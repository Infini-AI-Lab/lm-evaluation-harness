import os
from transformers.cache_utils import DynamicCache
from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np
import math
from einops import rearrange
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half
import falconn

torch.manual_seed(42)
np.random.seed(42)

class MagicPigCacheOrig(DynamicCache):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_attention_heads // config.num_key_value_heads
        self.sink_window = 4
        self.local_window = 64
        self.nsample = config.topk
        self.L = config.lsh_l
        self.K = config.lsh_k
        self.full_cache_layers = [0, 16]
        self.recalls = []
        self.num_unique = []

        device = kwargs.get("device")
        dtype = kwargs.get("dtype")

        self.sampling = False
        self.key_means = []
        self.hash_keys = []
        self.key_norms = []
        self.projection_matrix = torch.randn((self.num_kv_heads, self.L, self.K, self.head_dim), device=device, dtype=dtype)
        # self.projection_final = torch.randn((self.num_kv_heads, self.L, self.K), device=device, dtype=dtype)
        self.bin_2_int = torch.tensor([2**i for i in range(self.K)], device=device, dtype=torch.float32)

    def update(self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            query_states: Optional[torch.Tensor]=None,
            cache_kwargs: Optional[Dict[str, Any]] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        is_decode = key_states.shape[-2] == 1

        layer_offset = sum([t < layer_idx for t in self.full_cache_layers])
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        if layer_idx in self.full_cache_layers:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        if query_states is not None:
            assert is_decode
            num_key_value_groups = query_states.shape[1] // key_states.shape[1]
            query_states = rearrange(query_states, "b (h r) l d -> b h (r l) d", r=num_key_value_groups)
            
            key_cache = self.key_cache[layer_idx]
            last_non_local_key_states = key_cache[..., -self.local_window-1:-self.local_window, :]

            ##### non-aligned sampling methods #####
            d = query_states.shape[-1]
            W = torch.einsum("bhld,bhtd->bhlt", query_states, key_cache).float() 
        
            matches = self.sample(layer_idx, query_states, last_non_local_key_states, num_key_value_groups) # b h r n
            W_dynamic = W[..., self.sink_window : -self.local_window]   # b h r t
            
            topk_ids = W_dynamic.topk(10, dim=-1).indices
            overlap = torch.gather(matches, -1, topk_ids)
            recall = overlap.float().mean(dim=-1).view(-1)
            if len(self.recalls) <= layer_idx - layer_offset:
                self.recalls.append(recall.cpu().numpy())
            else:
                self.recalls[layer_idx - layer_offset] = np.concatenate([self.recalls[layer_idx - layer_offset], recall.cpu().numpy()], axis=-1)
            
            unique_ratio = matches.float().mean(dim=-1).view(-1)
            if len(self.num_unique) <= layer_idx - layer_offset:
                self.num_unique.append(unique_ratio.cpu().numpy())
            else:
                self.num_unique[layer_idx - layer_offset] = np.concatenate([self.num_unique[layer_idx - layer_offset], unique_ratio.cpu().numpy()])
            
            if self.sampling:
                W_dynamic = W_dynamic * matches

                norm_q = query_states.norm(dim=-1)  # h r
                norm_k = self.key_norms[layer_idx - layer_offset] # h t
                norm_qk = torch.einsum("bhr,bht->bhrt", norm_q, norm_k)  # b h r t
                sim_dynamic = W_dynamic  / norm_qk
                collision_probs = 1 - torch.arccos(sim_dynamic) / math.pi
                t = collision_probs**self.K
                sample_probs = 1 - (1-t)**self.L  # b h r t - single collision
                sample_probs = sample_probs - self.L * t * (1 - t)**(self.L-1)  # b h r t - 2 collisions

                W_dynamic.masked_fill_(matches.logical_not(), float("-inf"))  # not considering the ones that are not in sample

                dynamic_attn = (W_dynamic / math.sqrt(d)) - torch.log(sample_probs)  # W_s / P_s
                max_dynamic = dynamic_attn.max(dim=-1, keepdim=True).values
                dynamic_attn = dynamic_attn - max_dynamic
                lse_dynamic = torch.logsumexp(dynamic_attn, dim=-1, keepdim=True)  # log(Z_h)
                
                W_static = torch.cat([W[..., :self.sink_window], W[..., -self.local_window:]], dim=-1) 
                W_static = W_static / math.sqrt(d) 
                max_static = W_static.max(dim=-1, keepdim=True).values
                W_static = W_static - max_static
                lse_static = torch.logsumexp(W_static, dim=-1, keepdim=True)  # log(Z_h)
                
                dynamic_attn = torch.exp(dynamic_attn - lse_dynamic) / (1 + torch.exp(lse_static - lse_dynamic + max_static - max_dynamic))
                static_attn = torch.exp(W_static - lse_static) / (1 + torch.exp(lse_dynamic - lse_static + max_dynamic - max_static))

                attn = torch.cat([static_attn[..., :self.sink_window], dynamic_attn, static_attn[..., -self.local_window:]], dim=-1)
                output = torch.einsum("bhlt,bhtd->bhld", attn.to(query_states.dtype), self.value_cache[layer_idx])  # b h r d
                output = rearrange(output, "b h (r l) d -> b (h r) l d", r=num_key_value_groups)
                return None, output
            else:
                W_dynamic.masked_fill_(matches.logical_not(), float("-inf"))  
                attn = torch.cat([
                    W[..., :self.sink_window],
                    W_dynamic,
                    W[..., -self.local_window:]
                ], dim=-1) / math.sqrt(d)
                attn = torch.softmax(attn.float(), dim=-1).to(query_states.dtype)
                output = torch.einsum("bhlt,bhtd->bhld", attn, self.value_cache[layer_idx])
                output = rearrange(output, "b h (r l) d -> b (h r) l d", r=num_key_value_groups)
                return None, output
                                
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


    def sample(self, layer_idx, query_states, last_non_local_key_states, num_key_value_groups):
        layer_idx = layer_idx - sum([t < layer_idx for t in self.full_cache_layers])
        key_states = last_non_local_key_states - self.key_means[layer_idx]  # b h n d
        # key_norm = key_states.norm(dim=-1)  # b h n
        # self.key_norms[layer_idx] = torch.cat([self.key_norms[layer_idx], key_norm], dim=-1) # b h n   

        hash_scores = torch.einsum("hlkd,hnd->hnlk", self.projection_matrix, key_states[0]).unsqueeze(0) 
        hash_scores = hash_scores.gt(0).float()
        hash_k = torch.einsum("bhnlk,k->bhnl", hash_scores, self.bin_2_int).int()
        hash_k = self.hash_keys[layer_idx] = torch.cat([self.hash_keys[layer_idx], hash_k], dim=-2) # b h n l

        query_norm = query_states / query_states.norm(dim=-1, keepdim=True)
        hash_q = torch.einsum("hlkd,hnd->hnlk", self.projection_matrix, query_norm[0]).unsqueeze(0)
        hash_q = hash_q.gt(0).float()
        hash_q = torch.einsum("bhnlk,k->bhnl", hash_q, self.bin_2_int).int()    # b h r l

        # matches = (hash_q[..., None, :] == hash_k[..., None, :, :]).long().sum(-1).gt(1)   # b h r n
        nsample = max(hash_k.shape[-2] // 16, 64)
        hash_hits = (hash_q.unsqueeze(-2) == hash_k.unsqueeze(-3)).long().sum(dim=-1)
        topk_ids = hash_hits.topk(nsample, dim=-1).indices
        matches = torch.zeros_like(hash_hits, dtype=torch.bool)
        matches.scatter_(dim=-1, index=topk_ids, src=torch.ones_like(topk_ids, dtype=matches.dtype))

        return matches

    '''
    def prepare(self):
        for layer_idx in range(len(self.key_cache)):
            if layer_idx in self.full_cache_layers:
                continue
            non_sink_key_cache = self.key_cache[layer_idx][:, :, self.sink_window : -self.local_window, :]
            avg_key_states = non_sink_key_cache.mean(dim=-2, keepdim=True)  # b h 1 d
            self.key_means.append(avg_key_states)

            # subtract from all the key states
            self.key_cache[layer_idx] = self.key_cache[layer_idx] - avg_key_states

            key_states = self.key_cache[layer_idx][..., self.sink_window : -self.local_window, :] 
            key_norm = key_states.norm(dim=-1)  # b h n -> needed for sampling probs
            self.key_norms.append(key_norm)

            hash_scores = torch.einsum("hlkd,hnd->hnlk", self.projection_matrix, key_states[0]).unsqueeze(0) 

            # pack the hash keys into a single integer
            hash_codes = torch.zeros(*hash_scores.shape[:-1], dtype=torch.float32, device=hash_scores.device)
            for k in range(self.K):
                hash_codes += hash_scores[..., k].gt(0).float() * (2**k)
            self.hash_keys.append(hash_codes.int())   # b h n l
    '''
    def prepare(self):
        layer_offset = 0
        for layer_idx in range(len(self.key_cache)):
            if layer_idx in self.full_cache_layers:
                layer_offset += 1
                continue
            non_sink_key_cache = self.key_cache[layer_idx][:, :, self.sink_window : -self.local_window, :]
            avg_key_states = non_sink_key_cache.mean(dim=-2, keepdim=True)
            self.key_means.append(avg_key_states)

            key_states = non_sink_key_cache - avg_key_states
            # key_norms = key_states.norm(dim=-1)  # b h n
            # self.key_norms.append(key_norms)

            hash_scores = torch.einsum("hlkd,hnd->hnlk", self.projection_matrix, key_states[0]).unsqueeze(0) 

            # pack the hash keys into a single integer
            hash_codes = torch.zeros(*hash_scores.shape[:-1], dtype=torch.float32, device=hash_scores.device)
            for k in range(self.K):
                hash_codes += hash_scores[..., k].gt(0).float() * (2**k)
            self.hash_keys.append(hash_codes.int())   # b h n l

    def reset(self):
        self._seen_tokens = 0
        self.key_cache = []
        self.value_cache = []
        self.key_means = []
        self.hash_keys = []
        self.key_norms = []
        
class MagicPigCLSH(DynamicCache):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.sink_window = 4
        self.local_window = 64
        self.nsample = config.topk
        self.L = config.lsh_l
        self.K = config.lsh_k
        self.full_cache_layers = [0, 16]
        self.recalls = []
        self.num_unique = []

        device = kwargs.get("device")
        dtype = kwargs.get("dtype")

        self.sampling = False
        self.key_means = []
        self.hash_keys = []
        self.key_norms = []

        rotation = torch.rand((self.num_kv_heads, self.K, self.L, self.head_dim), device=device, dtype=dtype)
        self.rotation = rotation / rotation.norm(dim=-1, keepdim=True)

    def update(self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            query_states: Optional[torch.Tensor]=None,
            cache_kwargs: Optional[Dict[str, Any]] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        is_decode = key_states.shape[-2] == 1

        layer_offset = sum([t < layer_idx for t in self.full_cache_layers])
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        if layer_idx in self.full_cache_layers:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        if query_states is not None:
            assert is_decode
            num_key_value_groups = query_states.shape[1] // key_states.shape[1]
            query_states = rearrange(query_states, "b (h r) l d -> b h (r l) d", r=num_key_value_groups)
            
            key_cache = self.key_cache[layer_idx]
            last_non_local_key_states = key_cache[..., -self.local_window-1:-self.local_window, :]

            ##### non-aligned sampling methods #####
            d = query_states.shape[-1]
            W = torch.einsum("bhld,bhtd->bhlt", query_states, key_cache).float() 
        
            matches = self.sample(layer_idx, query_states, last_non_local_key_states, num_key_value_groups) # b h r n
            W_dynamic = W[..., self.sink_window : -self.local_window]   # b h r t
            
            topk_ids = W_dynamic.topk(10, dim=-1).indices
            overlap = torch.gather(matches, -1, topk_ids)
            recall = overlap.float().mean(dim=-1).view(-1)
            if len(self.recalls) <= layer_idx - layer_offset:
                self.recalls.append(recall.cpu().numpy())
            else:
                self.recalls[layer_idx - layer_offset] = np.concatenate([self.recalls[layer_idx - layer_offset], recall.cpu().numpy()], axis=-1)
            
            if self.sampling:
                # sampling probs
                W_avg = torch.einsum("bhld,bhtd->bhlt", query_states, self.key_means[layer_idx - layer_offset])
                norm_q = query_states.norm(dim=-1)  # h r
                norm_k = self.key_norms[layer_idx - layer_offset] # h t
                norm_qk = torch.einsum("bhr,bht->bhrt", norm_q, norm_k)  # b h r t
                sim_dynamic = (W_dynamic - W_avg)  / norm_qk
                collision_probs = 1 - torch.arccos(sim_dynamic) / math.pi
                nbit = int(math.log2(self.L))
                t = collision_probs**nbit
                sample_probs = 1 - (1-t)**self.K  # b h r t - single collision
                
                W_dynamic.masked_fill_(matches.logical_not(), float("-inf"))  # not considering the ones that are not in sample
                W_dynamic = W_dynamic / math.sqrt(query_states.shape[-1])
                W_dynamic = W_dynamic.float() - torch.log(sample_probs.float())
                max_dynamic = W_dynamic.max(dim=-1, keepdim=True).values
                W_dynamic = W_dynamic - max_dynamic
                lse_dynamic = torch.logsumexp(W_dynamic, dim=-1, keepdim=True)

                W_static = torch.cat([W[..., :self.sink_window], W[..., -self.local_window:]], dim=-1)
                W_static = W_static / math.sqrt(query_states.shape[-1])
                W_static = W_static.float()
                max_static = W_static.max(dim=-1, keepdim=True).values
                W_static = W_static - max_static
                lse_static = torch.logsumexp(W_static, dim=-1, keepdim=True)

                dynamic_attn = torch.exp(W_dynamic - lse_dynamic) / (1 + torch.exp(lse_static - lse_dynamic + max_static - max_dynamic))
                static_attn = torch.exp(W_static - lse_static) / (1 + torch.exp(lse_dynamic - lse_static + max_dynamic - max_static))

                attn = torch.cat([static_attn[..., :self.sink_window], dynamic_attn, static_attn[..., -self.local_window:]], dim=-1)
                output = torch.einsum("bhlt,bhtd->bhld", attn.to(query_states.dtype), self.value_cache[layer_idx])  # b h r d
                output = rearrange(output, "b h (r l) d -> b (h r) l d", r=num_key_value_groups)
                return None, output
            else:
                W_dynamic.masked_fill_(matches.logical_not(), float("-inf"))  
                attn = torch.cat([
                    W[..., :self.sink_window],
                    W_dynamic,
                    W[..., -self.local_window:],
                ], dim=-1) / math.sqrt(d)
                attn = torch.softmax(attn.float(), dim=-1)
                output = torch.einsum("bhlt,bhtd->bhld", attn.to(query_states.dtype), self.value_cache[layer_idx])
                output = rearrange(output, "b h (r l) d -> b (h r) l d", r=num_key_value_groups)
                return None, output
            
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def sample(self, layer_idx, query_states, last_non_local_key_states, num_key_value_groups):
        layer_idx = layer_idx - sum([t < layer_idx for t in self.full_cache_layers])
        key_states = last_non_local_key_states - self.key_means[layer_idx]  # b h n d
        # key_norm = key_states.norm(dim=-1)  # b h n
        # self.key_norms[layer_idx] = torch.cat([self.key_norms[layer_idx], key_norm], dim=-1) # b h n
        # dim_norm = torch.sqrt(self.key_norms_max[layer_idx].pow(2) - key_norm.pow(2))

        rotation = self.rotation.unsqueeze(0)
        # hash_code = torch.einsum("bhkld,bhnd->bhnkl", rotation, 
        #                          torch.cat([key_states, dim_norm.unsqueeze(-1)], dim=-1))   # b h n k l

        hash_code = torch.einsum("bhkld,bhnd->bhnkl", rotation, key_states)   # b h n k l
        hash_code = hash_code.argmax(dim=-1)  # b h n k
        hash_k = self.hash_keys[layer_idx] = torch.cat([self.hash_keys[layer_idx], hash_code], dim=-2) # b h n k

        # query_states = query_states / query_states.norm(dim=-1, keepdim=True)
        hash_code = torch.einsum("bhkld,bhnd->bhnkl", rotation, query_states)
        hash_q = hash_code.argmax(dim=-1)  # b h t k

        nsample = max(hash_k.shape[-2] // 16, 64)
        hash_hits = (hash_q.unsqueeze(-2) == hash_k.unsqueeze(-3)).long().sum(dim=-1)
        topk_ids = hash_hits.topk(nsample, dim=-1).indices
        matches = torch.zeros_like(hash_hits, dtype=torch.bool)
        matches.scatter_(dim=-1, index=topk_ids, src=torch.ones_like(topk_ids, dtype=matches.dtype))        
        # mask = matches.sum(dim=-1).gt(0)
        # matches.masked_fill_(mask.unsqueeze(-1).logical_not(), 0)
        # matches = (hash_q.unsqueeze(-2) == hash_k.unsqueeze(-3)).long().sum(dim=-1).gt(1)   # b h t n
        
        return matches

    def prepare(self):
        layer_offset = 0
        for layer_idx in range(len(self.key_cache)):
            if layer_idx in self.full_cache_layers:
                layer_offset += 1
                continue
            non_sink_key_cache = self.key_cache[layer_idx][:, :, self.sink_window : -self.local_window, :]
            avg_key_states = non_sink_key_cache.mean(dim=-2, keepdim=True)
            self.key_means.append(avg_key_states)

            key_states = non_sink_key_cache - avg_key_states
            # key_norms = key_states.norm(dim=-1)  # b h n
            # self.key_norms.append(key_norms)
            # key_norm_max = key_norms.max(dim=-1, keepdim=True).values
            # self.key_norms_max.append(key_norm_max)

            # dim_norm = torch.sqrt(key_norm_max.pow(2) - key_norms.pow(2))
            # key_states = torch.cat([key_states, dim_norm.unsqueeze(-1)], dim=-1)
            # hash_code = torch.einsum("bhkld,bhnd->bhnkl", self.rotation.unsqueeze(0), key_states)   # b h n k l
            
            hash_code = torch.einsum("bhkld,bhnd->bhnkl", self.rotation.unsqueeze(0), key_states) 
            hash_code = hash_code.argmax(dim=-1)  # b h n k
            self.hash_keys.append(hash_code)

    def reset(self):
        self._seen_tokens = 0
        self.key_cache = []
        self.value_cache = []
        self.key_means = []
        self.hash_keys = []
        self.key_norms = []
        self.key_norms_max = []


class MagicPigCPLSHOrig(DynamicCache):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.sink_window = 4
        self.local_window = 64
        self.nsample = config.topk
        self.L = config.lsh_l
        self.K = config.lsh_k
        self.full_cache_layers = [0, 16]
        self.recalls = []
        self.num_unique = []

        device = kwargs.get("device")
        dtype = kwargs.get("dtype")

        self.sampling = False
        self.key_means = []
        self.hash_keys = []
        self.key_norms = []

        self.rotation = torch.randn((self.num_kv_heads, self.L, self.K, self.head_dim//128, self.head_dim), device=device, dtype=dtype)

    def update(self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            query_states: Optional[torch.Tensor]=None,
            cache_kwargs: Optional[Dict[str, Any]] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        is_decode = key_states.shape[-2] == 1

        layer_offset = sum([t < layer_idx for t in self.full_cache_layers])
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        if layer_idx in self.full_cache_layers:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        if query_states is not None:
            assert is_decode
            num_key_value_groups = query_states.shape[1] // key_states.shape[1]
            query_states = rearrange(query_states, "b (h r) l d -> b h (r l) d", r=num_key_value_groups)
            
            key_cache = self.key_cache[layer_idx]
            last_non_local_key_states = key_cache[..., -self.local_window-1:-self.local_window, :]

            ##### non-aligned sampling methods #####
            d = query_states.shape[-1]
            W = torch.einsum("bhld,bhtd->bhlt", query_states, key_cache).float() 
        
            matches = self.sample(layer_idx, query_states, last_non_local_key_states, num_key_value_groups) # b h r n
            W_dynamic = W[..., self.sink_window : -self.local_window]   # b h r t
            
            topk_ids = W_dynamic.topk(10, dim=-1).indices
            overlap = torch.gather(matches, -1, topk_ids)
            recall = overlap.float().mean(dim=-1).view(-1)
            if len(self.recalls) <= layer_idx - layer_offset:
                self.recalls.append(recall.cpu().numpy())
            else:
                self.recalls[layer_idx - layer_offset] = np.concatenate([self.recalls[layer_idx - layer_offset], recall.cpu().numpy()], axis=-1)
            
            unique_ratio = matches.float().mean(dim=-1).view(-1)
            if len(self.num_unique) <= layer_idx - layer_offset:
                self.num_unique.append(unique_ratio.cpu().numpy())
            else:
                self.num_unique[layer_idx - layer_offset] = np.concatenate([self.num_unique[layer_idx - layer_offset], unique_ratio.cpu().numpy()])
            import pdb; pdb.set_trace()

            W_dynamic.masked_fill_(matches.logical_not(), float("-inf"))  
            attn = torch.cat([
                W[..., :self.sink_window],
                W_dynamic,
                W[..., -self.local_window:],
            ], dim=-1) / math.sqrt(d)
            attn = torch.softmax(attn.float(), dim=-1)
            output = torch.einsum("bhlt,bhtd->bhld", attn.to(query_states.dtype), self.value_cache[layer_idx])
            output = rearrange(output, "b h (r l) d -> b (h r) l d", r=num_key_value_groups)
            return None, output

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def sample(self, layer_idx, query_states, last_non_local_key_states, num_key_value_groups):
        layer_idx = layer_idx - sum([t < layer_idx for t in self.full_cache_layers])
        key_states = last_non_local_key_states - self.key_means[layer_idx]  # b h n d
        # key_norm = key_states.norm(dim=-1)  # b h n
        # self.key_norms[layer_idx] = torch.cat([self.key_norms[layer_idx], key_norm], dim=-1) # b h n   

        rotated_keys = torch.einsum("bhlktd,bhnd->bhnlkt", self.rotation.unsqueeze(0), key_states)
        rotated_keys = rotated_keys / rotated_keys.norm(dim=-1, keepdim=True)
        h = torch.argmax(torch.cat([-rotated_keys, rotated_keys], dim=-1), dim=-1)  # b h n l
        hash_k = self.hash_keys[layer_idx] = torch.cat([self.hash_keys[layer_idx], h], dim=2)

        query_states = query_states / query_states.norm(dim=-1, keepdim=True)
        rotated_queries = torch.einsum("bhlktd,bhnd->bhnlkt", self.rotation.unsqueeze(0), query_states)
        rotated_queries = rotated_queries / rotated_queries.norm(dim=-1, keepdim=True)
        hq = torch.argmax(torch.cat([-rotated_queries, rotated_queries], dim=-1), dim=-1)   # b h t l

        nsample = max(hash_k.shape[-2] // 16, 64)
        hash_hits = (hq.unsqueeze(3) == hash_k.unsqueeze(2)).all(dim=-1).long().sum(dim=-1)   # b h t n
        topk_ids = hash_hits.topk(nsample, dim=-1).indices
        matches = torch.zeros_like(hash_hits, dtype=torch.bool)
        matches.scatter_(dim=-1, index=topk_ids, src=torch.ones_like(topk_ids, dtype=matches.dtype))

        return matches

    def prepare(self):
        layer_offset = 0
        for layer_idx in range(len(self.key_cache)):
            if layer_idx in self.full_cache_layers:
                layer_offset += 1
                continue
            non_sink_key_cache = self.key_cache[layer_idx][:, :, self.sink_window : -self.local_window, :]
            avg_key_states = non_sink_key_cache.mean(dim=-2, keepdim=True)
            self.key_means.append(avg_key_states)

            key_states = non_sink_key_cache - avg_key_states
            # key_norms = key_states.norm(dim=-1)  # b h n
            # self.key_norms.append(key_norms)

            rotated_keys = torch.einsum("bhlktd,bhnd->bhnlkt", self.rotation.unsqueeze(0), key_states)
            rotated_keys = rotated_keys / rotated_keys.norm(dim=-1, keepdim=True)
            h = torch.argmax(torch.cat([-rotated_keys, rotated_keys], dim=-1), dim=-1)  # b h n l
            self.hash_keys.append(h)
            

    def reset(self):
        self._seen_tokens = 0
        self.key_cache = []
        self.value_cache = []
        self.key_means = []
        self.hash_keys = []
        self.key_norms = []

class SimHashRetrieveCache(DynamicCache):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_attention_heads // config.num_key_value_heads
        self.sink_window = 4
        self.local_window = 64
        self.nsample = config.topk
        self.L = config.lsh_l
        self.K = config.lsh_k
        self.full_cache_layers = [0, 16]
        self.recalls = []
        self.num_unique = []

        device = kwargs.get("device")
        dtype = kwargs.get("dtype")

        self.sampling = False
        self.key_means = []
        self.hash_keys = []
        self.key_norms = []

        self.indexes = []

    def update(self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            query_states: Optional[torch.Tensor]=None,
            cache_kwargs: Optional[Dict[str, Any]] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        is_decode = key_states.shape[-2] == 1

        layer_offset = sum([t < layer_idx for t in self.full_cache_layers])
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        if layer_idx in self.full_cache_layers:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        if query_states is not None:
            assert is_decode
            num_key_value_groups = query_states.shape[1] // key_states.shape[1]
            query_states = rearrange(query_states, "b (h r) l d -> b h (r l) d", r=num_key_value_groups)
            
            key_cache = self.key_cache[layer_idx]
            last_non_local_key_states = key_cache[..., -self.local_window-1:-self.local_window, :]

            ##### non-aligned sampling methods #####
            d = query_states.shape[-1]
            W = torch.einsum("bhld,bhtd->bhlt", query_states, key_cache).float() 
        
            matches = self.sample(layer_idx, query_states, last_non_local_key_states, num_key_value_groups) # b h r n
            W_dynamic = W[..., self.sink_window : -self.local_window]   # b h r t
            
            # matches = torch.zeros_like(W_dynamic, dtype=torch.bool)
            topk_ids = W_dynamic.topk(64, dim=-1).indices
            # matches.scatter_(dim=-1, index=topk_ids, src=torch.ones_like(topk_ids, dtype=matches.dtype))
            overlap = torch.gather(matches, -1, topk_ids)
            recall = overlap.float().mean(dim=-1).view(-1)
            if len(self.recalls) <= layer_idx - layer_offset:
                self.recalls.append(recall.cpu().numpy())
            else:
                self.recalls[layer_idx - layer_offset] = np.concatenate([self.recalls[layer_idx - layer_offset], recall.cpu().numpy()], axis=-1)
            
            unique_ratio = matches.float().mean(dim=-1).view(-1)
            if len(self.num_unique) <= layer_idx - layer_offset:
                self.num_unique.append(unique_ratio.cpu().numpy())
            else:
                self.num_unique[layer_idx - layer_offset] = np.concatenate([self.num_unique[layer_idx - layer_offset], unique_ratio.cpu().numpy()])
            
            W_dynamic.masked_fill_(matches.logical_not(), float("-inf"))  
            attn = torch.cat([
                W[..., :self.sink_window],
                W_dynamic,
                W[..., -self.local_window:]
            ], dim=-1) / math.sqrt(d)
            attn = torch.softmax(attn.float(), dim=-1).to(query_states.dtype)
            output = torch.einsum("bhlt,bhtd->bhld", attn, self.value_cache[layer_idx])
            output = rearrange(output, "b h (r l) d -> b (h r) l d", r=num_key_value_groups)
            return None, output
                                
        return self.key_cache[layer_idx], self.value_cache[layer_idx]            

    def sample(self, layer_idx, query_states, last_non_local_key_states, num_key_value_groups):
        nseq = self.get_seq_length(layer_idx) - self.sink_window - self.local_window
        layer_idx = layer_idx - sum([t < layer_idx for t in self.full_cache_layers])
        key_states = last_non_local_key_states - self.key_means[layer_idx]
        key_norm = key_states.norm(dim=-1)  # b h n
        self.key_norms[layer_idx] = torch.cat([self.key_norms[layer_idx], key_norm], dim=-1) # b h n

        index = self.indexes[layer_idx]
        query_norm = query_states / query_states.norm(dim=-1, keepdim=True)

        nsample = max(key_states.shape[-2] // 16, 64)
        matches = torch.zeros(*query_norm.shape[:-1], nseq, dtype=torch.bool, device=query_norm.device)
        
        for i in range(self.num_kv_heads):
            for j in range(self.num_kv_groups):
                # topk_ids = index[i].find_k_nearest_neighbors(query_norm[0][i][j].cpu().float().numpy(), nsample)
                topk_ids = index[i].get_unique_candidates(query_norm[0][i][j].cpu().float().numpy())
                matches[0][i][j].scatter_(0, torch.tensor(topk_ids, device=matches.device), 1)
        
        return matches

    def prepare(self):
        for layer_idx in range(len(self.key_cache)):
            if layer_idx in self.full_cache_layers:
                continue
            non_sink_key_cache = self.key_cache[layer_idx][:, :, self.sink_window : -self.local_window, :]
            avg_key_states = non_sink_key_cache.mean(dim=-2, keepdim=True)
            self.key_means.append(avg_key_states)

            key_states = non_sink_key_cache - avg_key_states
            key_norm = key_states.norm(dim=-1)  # b h n
            self.key_norms.append(key_norm)

            params_cp = falconn.LSHConstructionParameters()
            params_cp.dimension = key_states.shape[-1]
            params_cp.distance_function = falconn.DistanceFunction.NegativeInnerProduct
            params_cp.l = self.L
            params_cp.seed = 5721840
            params_cp.num_setup_threads = 0
            params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
            params_cp.num_rotations = 1

            params_cp.lsh_family = falconn.LSHFamily.Hyperplane
            params_cp.k = self.K

            local_indices = []
            for i in range(self.num_kv_heads):
                table = falconn.LSHIndex(params_cp)
                table.setup(key_states[0][i].cpu().float().numpy())
                query_object = table.construct_query_object()
                query_object.set_num_probes(self.L)
                local_indices.append(query_object)
                
            self.indexes.append(local_indices)

    def reset(self):
        self._seen_tokens = 0
        self.key_cache = []
        self.value_cache = []
        self.key_means = []
        self.hash_keys = []
        self.key_norms = []
        self.indexes = []

class CrossPolytopeRetrieveCache(SimHashRetrieveCache):
    def prepare(self):
        for layer_idx in range(len(self.key_cache)):
            if layer_idx in self.full_cache_layers:
                continue
            non_sink_key_cache = self.key_cache[layer_idx][:, :, self.sink_window : -self.local_window, :]
            avg_key_states = non_sink_key_cache.mean(dim=-2, keepdim=True)
            self.key_means.append(avg_key_states)

            key_states = non_sink_key_cache - avg_key_states
            key_norm = key_states.norm(dim=-1)  # b h n
            self.key_norms.append(key_norm)

            params_cp = falconn.LSHConstructionParameters()
            params_cp.dimension = key_states.shape[-1]
            params_cp.distance_function = falconn.DistanceFunction.NegativeInnerProduct
            params_cp.l = self.L
            params_cp.seed = 5721840
            params_cp.num_setup_threads = 0
            params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
            params_cp.num_rotations = 1

            params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
            # falconn.compute_number_of_hash_functions(self.K, params_cp)
            params_cp.k = 2
            params_cp.last_cp_dimension = 16

            local_indices = []
            for i in range(self.num_kv_heads):
                table = falconn.LSHIndex(params_cp)
                table.setup(key_states[0][i].cpu().float().numpy())
                query_object = table.construct_query_object()
                query_object.set_num_probes(self.L*4)
                local_indices.append(query_object)
                
            self.indexes.append(local_indices)

class MagicPigCache(DynamicCache):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_attention_heads // config.num_key_value_heads
        self.sink_window = 4
        self.local_window = 64
        self.nsample = config.topk
        self.L = config.lsh_l
        self.K = config.lsh_k
        self.full_cache_layers = np.array([0, 16])
        self.recalls = []
        self.num_unique = []

        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        self.key_means = []
        self.hash_keys = []
        self.key_norms = []
        self.projection_matrix = torch.randn((self.num_kv_heads, self.L, self.K, self.head_dim), device=device, dtype=dtype)
        self.bin_2_int = torch.tensor([2**i for i in range(self.K)], device=device, dtype=torch.float32)

    def update(self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            query_states: Optional[torch.Tensor]=None,
            cache_kwargs: Optional[Dict[str, Any]] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
                # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        is_decode = key_states.shape[-2] == 1

        layer_offset = sum([t < layer_idx for t in self.full_cache_layers])
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        if layer_idx in self.full_cache_layers:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        if query_states is not None:
            assert is_decode
            num_key_value_groups = query_states.shape[1] // key_states.shape[1]
            query_states = rearrange(query_states, "b (h r) l d -> b h (r l) d", r=num_key_value_groups)
            
            key_cache = self.key_cache[layer_idx]
            last_non_local_key_states = key_cache[..., -self.local_window-1:-self.local_window, :]

            ##### non-aligned sampling methods #####
            d = query_states.shape[-1]
            W = torch.einsum("bhld,bhtd->bhlt", query_states, key_cache).float() 
        
            matches = self.sample(layer_idx, query_states, last_non_local_key_states, num_key_value_groups) # b h r n
            W_dynamic = W[..., self.sink_window : -self.local_window]   # b h r t
            
            # matches = torch.zeros_like(W_dynamic, dtype=torch.bool)
            topk_ids = W_dynamic.topk(64, dim=-1).indices
            # matches.scatter_(dim=-1, index=topk_ids, src=torch.ones_like(topk_ids, dtype=matches.dtype))
            overlap = torch.gather(matches, -1, topk_ids)
            recall = overlap.float().mean(dim=-1).view(-1)
            if len(self.recalls) <= layer_idx - layer_offset:
                self.recalls.append(recall.cpu().numpy())
            else:
                self.recalls[layer_idx - layer_offset] = np.concatenate([self.recalls[layer_idx - layer_offset], recall.cpu().numpy()], axis=-1)
            
            unique_ratio = matches.float().mean(dim=-1).view(-1)
            if len(self.num_unique) <= layer_idx - layer_offset:
                self.num_unique.append(unique_ratio.cpu().numpy())
            else:
                self.num_unique[layer_idx - layer_offset] = np.concatenate([self.num_unique[layer_idx - layer_offset], unique_ratio.cpu().numpy()])
           
            W_dynamic.masked_fill_(matches.logical_not(), 0.)
            norm_q = query_states.norm(dim=-1)
            norm_k = self.key_norms[layer_idx - layer_offset]
            norm_qk = torch.einsum("bhr,bht->bhrt", norm_q, norm_k)
            sim_dynamic = W_dynamic / norm_qk
            collision_probs = 1 - torch.arccos(sim_dynamic) / math.pi
            t = collision_probs**self.K
            sample_probs = 1 - (1-t)**self.L
            sample_probs = sample_probs - self.L * t * (1 - t)**(self.L-1)

            W_dynamic.masked_fill_(matches.logical_not(), float("-inf"))
            dynamic_attn = W_dynamic / math.sqrt(d) - torch.log(sample_probs)
            max_dynamic = dynamic_attn.max(dim=-1, keepdim=True).values
            dynamic_attn = dynamic_attn - max_dynamic
            lse_dynamic = torch.logsumexp(dynamic_attn, dim=-1, keepdim=True)

            W_static = torch.cat([W[..., :self.sink_window], W[..., -self.local_window:]], dim=-1)
            W_static = W_static / math.sqrt(d)
            max_static = W_static.max(dim=-1, keepdim=True).values
            W_static = W_static - max_static
            lse_static = torch.logsumexp(W_static, dim=-1, keepdim=True)

            dynamic_attn = torch.exp(dynamic_attn - lse_dynamic) / (1 + torch.exp(lse_static - lse_dynamic + max_static - max_dynamic))
            static_attn = torch.exp(W_static - lse_static) / (1 + torch.exp(lse_dynamic - lse_static + max_dynamic - max_static))

            attn = torch.cat([static_attn[..., :self.sink_window], dynamic_attn, static_attn[..., -self.local_window:]], dim=-1)
            output = torch.einsum("bhlt,bhtd->bhld", attn.to(query_states.dtype), self.value_cache[layer_idx])  # b h r d
            output = rearrange(output, "b h (r l) d -> b (h r) l d", r=num_key_value_groups)
            return None, output
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def sample(self, layer_idx, query_states, last_non_local_key_states, num_key_value_groups):
        layer_offset = (self.full_cache_layers < layer_idx).sum()
        layer_idx = layer_idx - layer_offset

        key_states = last_non_local_key_states - self.key_means[layer_idx]  # b h n d
        key_norm = key_states.norm(dim=-1)  # b h n
        self.key_norms[layer_idx] = torch.cat([self.key_norms[layer_idx], key_norm], dim=-1) # b h n   

        hash_scores = torch.einsum("hlkd,hnd->hnlk", self.projection_matrix, key_states[0]).unsqueeze(0) 
        hash_scores = hash_scores.gt(0).float()
        hash_k = torch.einsum("bhnlk,k->bhnl", hash_scores, self.bin_2_int).int()
        hash_k = self.hash_keys[layer_idx] = torch.cat([self.hash_keys[layer_idx], hash_k], dim=-2) # b h n l

        query_norm = query_states / query_states.norm(dim=-1, keepdim=True)
        hash_q = torch.einsum("hlkd,hnd->hnlk", self.projection_matrix, query_norm[0]).unsqueeze(0)
        hash_q = hash_q.gt(0).float()
        hash_q = torch.einsum("bhnlk,k->bhnl", hash_q, self.bin_2_int).int()    # b h r l

        matches = (hash_q[..., None, :] == hash_k[..., None, :, :]).long().sum(-1).gt(1)   # b h r n
        
        return matches

    def prepare(self):
        for layer_idx in range(len(self.key_cache)):
            if layer_idx in self.full_cache_layers:
                continue
            non_sink_key_cache = self.key_cache[layer_idx][:, :, self.sink_window : -self.local_window, :]
            avg_key_states = non_sink_key_cache.mean(dim=-2, keepdim=True)  # b h 1 d
            self.key_means.append(avg_key_states)

            # subtract from all the key states
            key_cache = self.key_cache[layer_idx] - avg_key_states

            key_states = key_cache[..., self.sink_window : -self.local_window, :] 
            key_norm = key_states.norm(dim=-1)  # b h n -> needed for sampling probs
            self.key_norms.append(key_norm)

            hash_scores = torch.einsum("hlkd,hnd->hnlk", self.projection_matrix, key_states[0]).unsqueeze(0) 

            # pack the hash keys into a single integer
            hash_codes = torch.zeros(*hash_scores.shape[:-1], dtype=torch.float32, device=hash_scores.device)
            for k in range(self.K):
                hash_codes += hash_scores[..., k].gt(0).float() * (2**k)
            self.hash_keys.append(hash_codes.int())   # b h n l

    def reset(self):
        self._seen_tokens = 0
        self.key_cache = []
        self.value_cache = []
        self.key_means = []
        self.hash_keys = []
        self.key_norms = []

class MagicPigCache(DynamicCache):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_attention_heads // config.num_key_value_heads
        self.sink_window = 4
        self.local_window = 64
        self.nsample = config.topk
        self.L = config.lsh_l
        self.K = config.lsh_k
        self.full_cache_layers = np.array([0, 16])
        self.recalls = []
        self.num_unique = []

        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        self.key_means = []
        self.hash_keys = []
        self.key_norms = []
        self.projection_matrix = torch.randn((self.num_kv_heads, self.L, self.K, self.head_dim), device=device, dtype=dtype)
        # self.projection_final = torch.randn((self.num_kv_heads, self.L, self.K), device=device, dtype=dtype)
        self.bin_2_int = torch.tensor([2**i for i in range(self.K)], device=device, dtype=torch.float32)

    def update(self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            query_states: Optional[torch.Tensor]=None,
            cache_kwargs: Optional[Dict[str, Any]] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        is_decode = key_states.shape[-2] == 1
        layer_offset = (self.full_cache_layers < layer_idx).sum()
        if is_decode and layer_idx not in self.full_cache_layers:
            key_states = key_states - self.key_means[layer_idx - layer_offset] # b h t d

        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        if layer_idx in self.full_cache_layers:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        if query_states is not None:
            assert is_decode
            num_key_value_groups = query_states.shape[1] // key_states.shape[1]
            query_states = rearrange(query_states, "b (h r) l d -> b h (r l) d", r=num_key_value_groups)
            
            key_cache = self.key_cache[layer_idx]
            last_non_local_key_states = key_cache[..., -self.local_window-1:-self.local_window, :]

            ##### non-aligned sampling methods #####
            d = query_states.shape[-1]
            W = torch.einsum("bhld,bhtd->bhlt", query_states, key_cache).float() 
        
            matches = self.sample(layer_idx, query_states, last_non_local_key_states, num_key_value_groups) # b h r n
            W_dynamic = W[..., self.sink_window : -self.local_window]   # b h r t
            
            topk_ids = W_dynamic.topk(10, dim=-1).indices
            overlap = torch.gather(matches, -1, topk_ids)
            recall = overlap.float().mean(dim=-1).view(-1)
            if len(self.recalls) <= layer_idx - layer_offset:
                self.recalls.append(recall.cpu().numpy())
            else:
                self.recalls[layer_idx - layer_offset] = np.concatenate([self.recalls[layer_idx - layer_offset], recall.cpu().numpy()], axis=-1)
            
            unique_ratio = matches.float().mean(dim=-1).view(-1)
            if len(self.num_unique) <= layer_idx - layer_offset:
                self.num_unique.append(unique_ratio.cpu().numpy())
            else:
                self.num_unique[layer_idx - layer_offset] = np.concatenate([self.num_unique[layer_idx - layer_offset], unique_ratio.cpu().numpy()])
            
            '''
            W_dynamic = W_dynamic * matches
            norm_q = query_states.norm(dim=-1)  # h r
            norm_k = self.key_norms[layer_idx - layer_offset] # h t
            norm_qk = torch.einsum("bhr,bht->bhrt", norm_q, norm_k)  # b h r t
            sim_dynamic = W_dynamic  / norm_qk
            collision_probs = 1 - torch.arccos(sim_dynamic) / math.pi
            t = collision_probs**self.K
            sample_probs = 1 - (1-t)**self.L  # b h r t - single collision
            sample_probs = sample_probs - self.L * t * (1 - t)**(self.L-1)  # b h r t - 2 collisions
            '''

            W_dynamic.masked_fill_(matches.logical_not(), float("-inf"))  # not considering the ones that are not in sample

            dynamic_attn = (W_dynamic / math.sqrt(d)) #- torch.log(sample_probs)  # W_s / P_s
            max_dynamic = dynamic_attn.max(dim=-1, keepdim=True).values
            dynamic_attn = dynamic_attn - max_dynamic
            lse_dynamic = torch.logsumexp(dynamic_attn, dim=-1, keepdim=True)  # log(Z_h)
            
            W_static = torch.cat([W[..., :self.sink_window], W[..., -self.local_window:]], dim=-1) 
            W_static = W_static / math.sqrt(d) 
            max_static = W_static.max(dim=-1, keepdim=True).values
            W_static = W_static - max_static
            lse_static = torch.logsumexp(W_static, dim=-1, keepdim=True)  # log(Z_h)
            
            dynamic_attn = torch.exp(dynamic_attn - lse_dynamic) / (1 + torch.exp(lse_static - lse_dynamic + max_static - max_dynamic))
            static_attn = torch.exp(W_static - lse_static) / (1 + torch.exp(lse_dynamic - lse_static + max_dynamic - max_static))

            attn = torch.cat([static_attn[..., :self.sink_window], dynamic_attn, static_attn[..., -self.local_window:]], dim=-1)
            output = torch.einsum("bhlt,bhtd->bhld", attn.to(query_states.dtype), self.value_cache[layer_idx])  # b h r d
            output = rearrange(output, "b h (r l) d -> b (h r) l d", r=num_key_value_groups)
            return None, output
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


    def sample(self, layer_idx, query_states, last_non_local_key_states, num_key_value_groups):
        layer_offset = (self.full_cache_layers < layer_idx).sum()
        layer_idx = layer_idx - layer_offset

        key_states = last_non_local_key_states #- self.key_means[layer_idx]  # b h n d
        key_norm = key_states.norm(dim=-1)  # b h n
        self.key_norms[layer_idx] = torch.cat([self.key_norms[layer_idx], key_norm], dim=-1) # b h n   

        hash_scores = torch.einsum("hlkd,hnd->hnlk", self.projection_matrix, key_states[0]).unsqueeze(0) 
        hash_scores = hash_scores.gt(0).float()
        hash_k = torch.einsum("bhnlk,k->bhnl", hash_scores, self.bin_2_int).int()
        hash_k = self.hash_keys[layer_idx] = torch.cat([self.hash_keys[layer_idx], hash_k], dim=-2) # b h n l

        query_norm = query_states / query_states.norm(dim=-1, keepdim=True)
        hash_q = torch.einsum("hlkd,hnd->hnlk", self.projection_matrix, query_norm[0]).unsqueeze(0)
        hash_q = hash_q.gt(0).float()
        hash_q = torch.einsum("bhnlk,k->bhnl", hash_q, self.bin_2_int).int()    # b h r l

        matches = (hash_q[..., None, :] == hash_k[..., None, :, :]).long().sum(-1).gt(1)   # b h r n
        
        return matches


    def prepare(self):
        for layer_idx in range(len(self.key_cache)):
            if layer_idx in self.full_cache_layers:
                continue
            non_sink_key_cache = self.key_cache[layer_idx][:, :, self.sink_window : -self.local_window, :]
            avg_key_states = non_sink_key_cache.mean(dim=-2, keepdim=True)  # b h 1 d
            self.key_means.append(avg_key_states)

            # subtract from all the key states
            self.key_cache[layer_idx] = self.key_cache[layer_idx] - avg_key_states

            key_states = self.key_cache[layer_idx][..., self.sink_window : -self.local_window, :] 
            key_norm = key_states.norm(dim=-1)  # b h n -> needed for sampling probs
            self.key_norms.append(key_norm)

            hash_scores = torch.einsum("hlkd,hnd->hnlk", self.projection_matrix, key_states[0]).unsqueeze(0) 

            # pack the hash keys into a single integer
            hash_codes = torch.zeros(*hash_scores.shape[:-1], dtype=torch.float32, device=hash_scores.device)
            for k in range(self.K):
                hash_codes += hash_scores[..., k].gt(0).float() * (2**k)
            self.hash_keys.append(hash_codes.int())   # b h n l

    def reset(self):
        self._seen_tokens = 0
        self.key_cache = []
        self.value_cache = []
        self.key_means = []
        self.hash_keys = []
        self.key_norms = []

class MagicPigCPLSH(DynamicCache):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.sink_window = 4
        self.local_window = 64
        self.nsample = config.topk
        self.L = config.lsh_l
        self.K = config.lsh_k
        self.full_cache_layers = [0, 16]
        self.recalls = []
        self.num_unique = []

        device = kwargs.get("device")
        dtype = kwargs.get("dtype")

        self.key_means = []
        self.hash_keys = []
        self.key_norms = []
        self.key_norms_max = []

        self.cpd = 16
        rotation = torch.rand((self.num_kv_heads, self.L, self.K, self.cpd, self.head_dim), device=device, dtype=dtype)
        self.rotation = rotation / rotation.norm(dim=-1, keepdim=True)

        '''
        #### Calibration ####
        self.is_calibrate = True
        self.query_cache = []
        # self.sim = []
        self.sim_thresh = torch.from_numpy(np.load("clsh_sim_thresh.npy")).to(device) 
        self.sim_2_hitrate = []
        '''
        self.sim_thresh = torch.from_numpy(np.load("clsh_sim_thresh.npy")).to(device)
        self.sim_2_hitrate = torch.from_numpy(np.load("clsh_sim_2_probs.npy")).to(device)

    def update(self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            query_states: Optional[torch.Tensor]=None,
            cache_kwargs: Optional[Dict[str, Any]] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        is_decode = key_states.shape[-2] == 1

        layer_offset = sum([t < layer_idx for t in self.full_cache_layers])
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        if layer_idx in self.full_cache_layers:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        if query_states is not None:
            # assert is_decode
            num_key_value_groups = query_states.shape[1] // key_states.shape[1]
            query_states = rearrange(query_states, "b (h r) l d -> b h (r l) d", r=num_key_value_groups)
           
            '''
            if self.is_calibrate:
                assert len(self.query_cache) == layer_idx - layer_offset and not is_decode
                self.query_cache.append(query_states)
                return self.key_cache[layer_idx], self.value_cache[layer_idx]
            '''

            key_cache = self.key_cache[layer_idx]
            last_non_local_key_states = key_cache[..., -self.local_window-1:-self.local_window, :]

            matches = self.sample(layer_idx, query_states, last_non_local_key_states, num_key_value_groups) # b h r n

            W = torch.einsum("bhld,bhtd->bhlt", query_states, key_cache)
            W_dynamic = W[..., self.sink_window : -self.local_window]   # b h r t

            # sampling probs
            W_avg = torch.einsum("bhld,bhtd->bhlt", query_states, self.key_means[layer_idx - layer_offset])
            norm_q = query_states.norm(dim=-1)  # h r
            norm_k = self.key_norms[layer_idx - layer_offset] # h t
            norm_qk = torch.einsum("bhr,bht->bhrt", norm_q, norm_k)  # b h r t
            sim_dynamic = (W_dynamic - W_avg)  / norm_qk
            '''
            collision_probs_approx = 1 - torch.arccos(sim_dynamic) / math.pi
            nbit = int(math.log2(self.cpd)) * self.K
            t = collision_probs_approx**nbit
            sample_probs_approx = 1 - (1-t)**self.L  # b h r t - single collision
            sample_probs_approx = sample_probs_approx - self.L * t * (1 - t)**(self.L-1)
            
            sim_thresh = self.sim_thresh[layer_idx - layer_offset]
            hitrates = self.sim_2_hitrate[layer_idx - layer_offset]
            sim_quantized = (sim_dynamic[0].view(sim_dynamic.shape[1], -1).unsqueeze(-1) > sim_thresh.unsqueeze(-2)).long().sum(dim=-1) # h t*n 
            collision_probs = torch.gather(hitrates, -1, sim_quantized)
            collision_probs = collision_probs.reshape(sim_dynamic.shape)
            t = collision_probs**self.K
            sample_probs = 1 - (1-t)**self.L
            sample_probs = sample_probs - self.L * t * (1 - t)**(self.L-1)  # b h r t - 2 collisions
            '''

            # calculate recall
            topk_ids = W_dynamic.topk(10, dim=-1).indices
            overlap = torch.gather(matches, -1, topk_ids)
            recall = overlap.float().mean(dim=-1).view(-1)
            if len(self.recalls) <= layer_idx - layer_offset:
                self.recalls.append(recall.cpu().numpy())
            else:
                self.recalls[layer_idx - layer_offset] = np.concatenate([self.recalls[layer_idx - layer_offset], recall.cpu().numpy()], axis=-1)
            
            unique_ratio = matches.float().mean(dim=-1).view(-1)
            if len(self.num_unique) <= layer_idx - layer_offset:
                self.num_unique.append(unique_ratio.cpu().numpy())
            else:
                self.num_unique[layer_idx - layer_offset] = np.concatenate([self.num_unique[layer_idx - layer_offset], unique_ratio.cpu().numpy()])
            
            W_dynamic.masked_fill_(matches.logical_not(), float("-inf"))  # not considering the ones that are not in sample
            W_dynamic = W_dynamic / math.sqrt(query_states.shape[-1])
            W_dynamic = W_dynamic.float() #- torch.log(sample_probs.float())
            max_dynamic = W_dynamic.max(dim=-1, keepdim=True).values
            W_dynamic = W_dynamic - max_dynamic
            lse_dynamic = torch.logsumexp(W_dynamic, dim=-1, keepdim=True)

            W_static = torch.cat([W[..., :self.sink_window], W[..., -self.local_window:]], dim=-1)
            W_static = W_static / math.sqrt(query_states.shape[-1])
            W_static = W_static.float()
            max_static = W_static.max(dim=-1, keepdim=True).values
            W_static = W_static - max_static
            lse_static = torch.logsumexp(W_static, dim=-1, keepdim=True)

            dynamic_attn = torch.exp(W_dynamic - lse_dynamic) / (1 + torch.exp(lse_static - lse_dynamic + max_static - max_dynamic))
            static_attn = torch.exp(W_static - lse_static) / (1 + torch.exp(lse_dynamic - lse_static + max_dynamic - max_static))

            attn = torch.cat([static_attn[..., :self.sink_window], dynamic_attn, static_attn[..., -self.local_window:]], dim=-1)
            output = torch.einsum("bhlt,bhtd->bhld", attn.to(query_states.dtype), self.value_cache[layer_idx])  # b h r d
            output = rearrange(output, "b h (r l) d -> b (h r) l d", r=num_key_value_groups)
            return None, output

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def sample(self, layer_idx, query_states, last_non_local_key_states, num_key_value_groups):
        layer_idx = layer_idx - sum([t < layer_idx for t in self.full_cache_layers])
        key_states = last_non_local_key_states - self.key_means[layer_idx]  # b h n d
        key_norm = key_states.norm(dim=-1)  # b h n
        self.key_norms[layer_idx] = torch.cat([self.key_norms[layer_idx], key_norm], dim=-1) # b h n
        
        rotation = self.rotation.unsqueeze(0)
        hash_code = torch.einsum("bhlkcd,bhnd->bhnlkc", rotation, key_states)   # b h n l k
        hash_code = hash_code.argmax(dim=-1)  # b h n l k
        hash_k = self.hash_keys[layer_idx] = torch.cat([self.hash_keys[layer_idx], hash_code], dim=-3) # b h n l k

        query_states = query_states / query_states.norm(dim=-1, keepdim=True)
        hash_code = torch.einsum("bhlkcd,bhnd->bhnlkc", rotation, query_states)
        hash_q = hash_code.argmax(dim=-1)  # b h t l k
        
        # hash_hit = (hash_q.unsqueeze(-2) == hash_k.unsqueeze(-3)).long().sum(dim=-1)    # b h t n
        # topk_ids = hash_hit.topk(self.nsample, dim=-1).indices
        # matches = torch.zeros_like(hash_hit)
        # matches.scatter_(dim=-1, index=topk_ids, src=torch.ones_like(topk_ids, dtype=matches.dtype))
        # mask = matches.sum(dim=-1).gt(0)
        # matches.masked_fill_(mask.unsqueeze(-1).logical_not(), 0)
        # matches = (hash_q.unsqueeze(-3) == hash_k.unsqueeze(-4)).all(dim=-1).any(dim=-1)   # b h t n
        matches = (hash_q.unsqueeze(-3) == hash_k.unsqueeze(-4)).all(dim=-1).long().sum(-1).gt(1)

        return matches

    def prepare(self):
        layer_offset = 0
        for layer_idx in range(len(self.key_cache)):
            if layer_idx in self.full_cache_layers:
                layer_offset += 1
                continue
            non_sink_key_cache = self.key_cache[layer_idx][:, :, self.sink_window : -self.local_window, :]
            avg_key_states = non_sink_key_cache.mean(dim=-2, keepdim=True)
            self.key_means.append(avg_key_states)

            key_states = non_sink_key_cache - avg_key_states
            key_norms = key_states.norm(dim=-1)  # b h n
            self.key_norms.append(key_norms)
            
            hash_code = torch.einsum("bhlkcd,bhnd->bhnlkc", self.rotation.unsqueeze(0), key_states) 
            hash_code = hash_code.argmax(dim=-1)  # b h n l k
            self.hash_keys.append(hash_code)

    def calibrate(self):
        layer_offset = 0
        for layer_idx in range(len(self.key_cache)):
            if layer_idx in self.full_cache_layers:
                layer_offset += 1
                continue
            assert len(self.query_cache) > layer_idx - layer_offset, "Calibration is to be performed"
            query_states = self.query_cache[layer_idx - layer_offset][..., -64:, :]
            query_states = query_states / query_states.norm(dim=-1, keepdim=True)
            key_states = self.key_cache[layer_idx][..., self.sink_window : -self.local_window, :]
            key_states = key_states - self.key_means[layer_idx - layer_offset]
            key_states = key_states / self.key_norms[layer_idx - layer_offset].unsqueeze(-1)
            sim = torch.einsum("bhld,bhtd->bhlt", query_states, key_states).view(self.num_kv_heads, -1) # h t*n

            '''            
            if len(self.sim) <= layer_idx - layer_offset:
                self.sim.append(sim.cpu().float().numpy())
            else:
                self.sim[layer_idx - layer_offset] = np.concatenate([self.sim[layer_idx - layer_offset], sim.cpu().float().numpy()], axis=-1)
            '''
            nbuckets = 20
            sim_thresh = self.sim_thresh[layer_idx - layer_offset]
            sim_quantized = (sim.view(sim.shape[0], -1, 1) > sim_thresh.view(sim.shape[0], 1, -1)).long().sum(dim=-1)  # h x t*n
            sim_2_cnt = torch.zeros(self.num_kv_heads, nbuckets, device=query_states.device, dtype=torch.int32)
            sim_2_cnt.scatter_add_(1, sim_quantized, torch.ones_like(sim_quantized, dtype=torch.int32))

            hash_code = torch.einsum("bhlkcd,bhnd->bhnlkc", self.rotation.unsqueeze(0), query_states)
            hash_q = hash_code.argmax(dim=-1)  # b h t l k

            hash_k = self.hash_keys[layer_idx - layer_offset]

            hash_q = rearrange(hash_q, "b h n l k -> (b h) n (l k)")
            hash_k = rearrange(hash_k, "b h n l k -> (b h) n (l k)")
            hitrate = (hash_q.unsqueeze(-2) == hash_k.unsqueeze(-3)).float().mean(dim=-1)   # h t n
            hitrate = hitrate.view(self.num_kv_heads, -1)

            sim_2_hitrate = torch.zeros(self.num_kv_heads, nbuckets, device=query_states.device, dtype=torch.float32)
            sim_2_hitrate.scatter_add_(1, sim_quantized, hitrate)
            sim_2_hitrate = sim_2_hitrate / sim_2_cnt.float()
            sim_2_hitrate = sim_2_hitrate.unsqueeze(-1)

            if len(self.sim_2_hitrate) <= layer_idx - layer_offset:
                self.sim_2_hitrate.append(sim_2_hitrate.cpu().float().numpy())
            else:
                self.sim_2_hitrate[layer_idx - layer_offset] = np.concatenate([self.sim_2_hitrate[layer_idx - layer_offset], sim_2_hitrate.cpu().float().numpy()], axis=-1)
             

    def reset(self):
        self._seen_tokens = 0
        self.key_cache = []
        self.value_cache = []
        self.key_means = []
        self.hash_keys = []
        self.key_norms = []
        self.key_norms_max = []

        # self.query_cache = []
        # self.is_calibrate = True
