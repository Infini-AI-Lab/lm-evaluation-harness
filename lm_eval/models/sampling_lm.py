import copy
import os
from datetime import timedelta
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import LlamaForCausalLM 

import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from transformers.modeling_flash_attention_utils import _flash_attention_forward 
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as llama_apply_rotary_pos_emb
from transformers.models.llama.modeling_llama import repeat_kv as llama_repeat_kv
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.utils import GenerationConfig, LogitsProcessorList, StoppingCriteriaList, GenerateNonBeamOutput, GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput
from lm_eval.models.retrieval_cache_utils import SimHashRetrieveCache, CrossPolytopeRetrieveCache, MagicPigCache, MagicPigCPLSH
from lm_eval.models.retrieval_cache_utils import SimHashRetrieveCache, CrossPolytopeRetrieveCache, MagicPigCache, MagicPigCPLSH

from transformers.models.llama.modeling_llama import LlamaFlashAttention2
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model

# modifying _sample method to prepare the model anns indexes after every prefilling stage 
def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        is_prefill = True

        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits.clone()[:, -1, :].float()
            next_token_logits = next_token_logits.to(input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

            if is_prefill:
                self._cache.prepare()
                if getattr(self._cache, "is_calibrate", False):
                    self._cache.calibrate()
                    self._cache.is_calibrate = False
                is_prefill = False

            # else:
            #     print(f"sample_time: {self._cache.sample_time:.4f} min")

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

def _get_cache(
        self, cache_implementation: str, batch_size: int, max_cache_len: int, device: torch.device, model_kwargs
    ) -> Cache:
        """
        Sets a cache for `generate`, that will persist across calls. A new cache will only be initialized a
        new `generate` call requires a larger cache or uses a different batch size.

        Returns the resulting cache object.
        """
        if self.config.cache_implementation == 'simhash':
            cache_cls: Cache = SimHashRetrieveCache
        elif self.config.cache_implementation == "cplsh":
            cache_cls: Cache = CrossPolytopeRetrieveCache
        elif self.config.cache_implementation == "magicpig":
        if self.config.cache_implementation == 'simhash':
            cache_cls: Cache = SimHashRetrieveCache
        elif self.config.cache_implementation == "cplsh":
            cache_cls: Cache = CrossPolytopeRetrieveCache
        elif self.config.cache_implementation == "magicpig":
            cache_cls: Cache = MagicPigCache
        elif self.config.cache_implementation == "magicpig_clsh":
            cache_cls: Cache = MagicPigCPLSH
        else:
            raise ValueError(
                f"Cache implementation {self.config.cache_implementation} not recognized. "
                f"Choose one of 'retrieval', 'ivf_retrieval', 'sampling', 'ivf_sampling'."
            )
        need_new_cache = not hasattr(self, "_cache")

        if need_new_cache:
            if hasattr(self.config, "_pre_quantization_dtype"):
                cache_dtype = self.config._pre_quantization_dtype
            else:
                cache_dtype = self.get_output_embeddings().weight.dtype

            def get_layer_device_map(execution_device_map: Optional[dict] = None):
                if execution_device_map is None:
                    return None
                elif len(execution_device_map) == 1 and "" in execution_device_map:
                    return {idx: execution_device_map[""] for idx in range(self.config.num_hidden_layers)}
                layer_device_map = {}
                for layer in execution_device_map:
                    for idx in range(self.config.num_hidden_layers):
                        if f".{idx}." in f"{layer}.":
                            layer_device_map[idx] = execution_device_map[layer]
                            break
                for idx in range(self.config.num_hidden_layers):
                    if idx not in layer_device_map:
                        raise RuntimeError(f"layer {idx} has not been mapped to a device.")
                return layer_device_map

            execution_device_map = None
            if hasattr(self, "hf_device_map"):
                main_device = [d for d in self.hf_device_map.values() if d not in ["cpu", "disk"]][0]
                execution_device_map = {
                    name: main_device if device in ["cpu", "disk"] else device
                    for name, device in self.hf_device_map.items()
                }
            layer_device_map = get_layer_device_map(execution_device_map)

            cache_kwargs = {
                "config": self.config.get_text_config(),
                "batch_size": batch_size,
                "max_cache_len": max_cache_len,
                "device": device,
                "dtype": cache_dtype,
                "layer_device_map": layer_device_map,
            }
            self._cache = cache_cls(**cache_kwargs)
        else:
            self._cache.reset()
        return self._cache

def _prepare_cache_for_generation(
        self,
        generation_config: GenerationConfig,
        model_kwargs: Dict,
        assistant_model: "PreTrainedModel",
        batch_size: int,
        max_cache_length: int,
        device: torch.device,
    ) -> bool:
    cache_name = "past_key_values"
    if hasattr(self, "_cache"):
        self._cache.reset()
        model_kwargs[cache_name] = self._cache
    else:
        model_kwargs[cache_name] = self._get_cache(
            cache_implementation="store_cache",
            batch_size=max(generation_config.num_beams, generation_config.num_return_sequences) * batch_size,
            max_cache_len=max_cache_length,
            device=device,
            model_kwargs=model_kwargs,
        )
    return True

def llama_attn_forward(self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        output_attentions = False
        attn_weights = None

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = llama_apply_rotary_pos_emb(query_states, key_states, cos, sin)

        is_decode = query_states.shape[-2] == 1
        is_calibrate = getattr(past_key_value, "is_calibrate", False) # and isinstance(past_key_value, MagicPigCLSH) 

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}    
            if is_decode or is_calibrate:   # magicpig_clsh needs queries for sampling prob calibration
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, query_states, cache_kwargs)
            else:
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs=cache_kwargs)

        if key_states is None:
            attn_output = value_states.transpose(1, 2)
        else:
            # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
            # to be able to avoid many of these transpose/reshape/view.
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            dropout_rate = self.attention_dropout if self.training else 0.0

            input_dtype = query_states.dtype
            if input_dtype == torch.float32:
                if torch.is_autocast_enabled():
                    target_dtype = torch.get_autocast_gpu_dtype()
                # Handle the case where the model is quantized
                elif hasattr(self.config, "_pre_quantization_dtype"):
                    target_dtype = self.config._pre_quantization_dtype
                else:
                    target_dtype = self.q_proj.weight.dtype

                logger.warning_once(
                    f"The input hidden states seems to be silently casted in float32, this might be related to"
                    f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                    f" {target_dtype}."
                )

                query_states = query_states.to(target_dtype)
                key_states = key_states.to(target_dtype)
                value_states = value_states.to(target_dtype)

            attn_output = _flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                q_len,
                position_ids=position_ids,
                dropout=dropout_rate,
                use_top_left_mask=self._flash_attn_uses_top_left_mask,
                is_causal=self.is_causal,
            )
        
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights, past_key_value

LlamaFlashAttention2.forward = llama_attn_forward
LlamaForCausalLM._prepare_cache_for_generation = _prepare_cache_for_generation
LlamaForCausalLM._get_cache = _get_cache
LlamaForCausalLM._sample = _sample

@register_model("magicpig")
class SamplingLM(HFLM):
    def _create_model(
        self,
        pretrained: str,
        revision: Optional[str] = "main",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        trust_remote_code: Optional[bool] = False,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        # (accelerate naive PP (device_map) options)
        parallelize: Optional[bool] = False,
        gpus: Optional[int] = None,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        # PEFT, delta weights and quantization options
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        gptqmodel: Optional[bool] = False,
        **kwargs,
    ) -> None:
        model_kwargs = kwargs if kwargs else {}
        cache_impl = model_kwargs.pop("cache_impl", None)
        lsh_l = model_kwargs.pop("lsh_l", None)
        lsh_k = model_kwargs.pop("lsh_k", None)
        budget = model_kwargs.pop("budget", None)

        model_kwargs.update(
            self._get_accelerate_args(
                parallelize=parallelize,
                device_map=kwargs.get("device_map", None),
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                gpus=gpus,
            )
        )

        model = LlamaForCausalLM.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16,
                trust_remote_code=trust_remote_code,
                _attn_implementation = "flash_attention_2",
                **model_kwargs,
            )
        model.config.cache_implementation = cache_impl
        model.config.lsh_l = lsh_l
        model.config.lsh_k = lsh_k
        model.config.topk = budget
        self._model = model

    def generate_until(self, *args, **kwargs):
        res = super().generate_until(*args, **kwargs)

        # if hasattr(self.model._cache, "recalls"):
        #     recalls = np.stack(self.model._cache.recalls, axis=0)
        #     recalls = recalls.mean(axis=-1) # n_layer
        #     np.save(f"recalls_{self.model.config.cache_implementation}.npy", recalls)
        #     print(recalls)
        #     # plt.plot(recalls)
        #     # plt.xlabel("Layer")
        #     # plt.ylabel("Recall@10")
        #     # plt.title("Recall scores per layer")
        #     # plt.savefig("recall_scores.png")

        if hasattr(self.model._cache, "num_unique"):
            num_unique = np.stack(self.model._cache.num_unique, axis=0)
            total_unique = num_unique.mean(axis=-1)
            np.save(f"unique_tokens_{self.model.config.cache_implementation}.npy", total_unique)
            print(total_unique)
            # plt.plot(total_unique)
            # plt.xlabel("Layer")
            # plt.ylabel("Unique tokens")
            # plt.title("Unique tokens per layer")
            # plt.savefig("unique_tokens.png")

        return res
