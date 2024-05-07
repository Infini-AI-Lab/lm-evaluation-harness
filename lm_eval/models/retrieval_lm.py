import copy
import os
from datetime import timedelta
from pathlib import Path
import types
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

import transformers
import joblib
from tqdm import tqdm

from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model

def retrieval_attn(self, query, key, value, attention_mask=None, head_mask=None):
    batch_size, num_attention_heads, query_length, attn_head_size = query.size()
    key_length = key.size(-2)

    y_approx, approx_attn_scores, topk_ids = self._approx_attn(query, key, value, attention_mask, head_mask)

    if key_length < self.topk * 10:
        return y_approx, approx_attn_scores

    assert topk_ids is not None, "topk_ids must be returned by _approx_attn"

    if attention_mask is None and self.is_causal:
        attention_mask = torch.tril(
            torch.ones((query_length, key_length), device=query.device, dtype=torch.bool)
        ).logical_not()

    attn_scores = torch.zeros(
        batch_size * num_attention_heads,
        query_length,
        key_length,
        dtype=query.dtype,
        device=key.device,
    )
    attn_scores = torch.baddbmm(
        attn_scores,
        query.contiguous().view(-1, query_length, attn_head_size),
        key.contiguous().view(-1, key_length, attn_head_size).transpose(1, 2),
        beta=1.0,
        alpha=self.norm_factor if hasattr(self, "norm_factor") else np.sqrt(attn_head_size),
    )
    attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

    mask_value = torch.finfo(attn_scores.dtype).min
    mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
    attn_bias = torch.zeros((query_length, key_length), dtype=attn_scores.dtype, device=attn_scores.device)
    if attention_mask.dtype == torch.bool:
        attn_bias.masked_fill_(attention_mask, mask_value)
    else:
        attn_bias += attention_mask

    attn_scores += attn_bias[None, None, :, :]

    # topk recall (index approximation error)
    topk = self.topk
    min_length = topk * 10
    # do not consider the first min_length tokens, topk is not very interesting there
    _, gt_ids = torch.topk(attn_scores[..., min_length:, :], topk, dim=-1)
    topk_ids = topk_ids[..., min_length:, :]
    
    self.gt_ids.append(gt_ids.transpose(0, 1).reshape(num_attention_heads, -1, self.topk).cpu())
    self.ret_ids.append(topk_ids.transpose(0, 1).reshape(num_attention_heads, -1, self.topk).cpu()) 

    attn_scores = F.softmax(attn_scores, dim=-1).to(value.dtype)   

    # attention recall (what fraction of original attention is captured by exact topk and approx topk)
    exact_topk_attn_ratio = torch.sum(attn_scores[..., min_length:, :].gather(-1, gt_ids), dim=-1) # (batch_size, num_attention_heads, query_length)
    approx_topk_attn_ratio = torch.sum(attn_scores[..., min_length:, :].gather(-1, topk_ids), dim=-1) # (batch_size, num_attention_heads, query_length)
    self.exact_topk_attn_ratios.append(exact_topk_attn_ratio.transpose(0, 1).reshape(num_attention_heads, -1).cpu())
    self.approx_topk_attn_ratios.append(approx_topk_attn_ratio.transpose(0, 1).reshape(num_attention_heads, -1).cpu())

    attn_out = torch.bmm(attn_scores.view(-1, query_length, key_length), 
                            value.contiguous().view(-1, key_length, attn_head_size))
    attn_out = attn_out.view(batch_size, num_attention_heads, query_length, attn_head_size)
    
    # do not consider the first min_length tokens, topk is not very interesting there
    err_norm = torch.norm(attn_out[..., min_length:, :] - y_approx[..., min_length:, :], dim=-1) \
                / torch.norm(attn_out[..., min_length:, :], dim=-1)   # (batch_size, num_attention_heads, query_length)
    err_norm = err_norm.transpose(0, 1).reshape(num_attention_heads, -1)
    self.err_norm.append(err_norm.cpu())

    if self.layerwise_topk_analysis:
        return attn_out, attn_scores
    return y_approx, approx_attn_scores

def topk_attn(self, query, key, value, attention_mask=None, head_mask=None):
    batch_size, num_attention_heads, query_length, attn_head_size = query.size()
    key_length = key.size(-2)

    if attention_mask is None and self.is_causal:
        attention_mask = torch.tril(
            torch.ones((query_length, key_length), device=query.device, dtype=torch.bool)
        ).logical_not()

    attn_scores = torch.zeros(
        batch_size * num_attention_heads,
        query_length,
        key_length,
        dtype=query.dtype,
        device=key.device,
    )
    attn_scores = torch.baddbmm(
        attn_scores,
        query.contiguous().view(-1, query_length, attn_head_size),
        key.contiguous().view(-1, key_length, attn_head_size).transpose(1, 2),
        beta=1.0,
        alpha=self.norm_factor if hasattr(self, "norm_factor") else np.sqrt(attn_head_size),
    )
    attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)
    
    mask_value = torch.finfo(attn_scores.dtype).min
    mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
    attn_bias = torch.zeros((query_length, key_length), dtype=attn_scores.dtype, device=attn_scores.device)
    if attention_mask.dtype == torch.bool:
        attn_bias.masked_fill_(attention_mask, mask_value)
    else:
        attn_bias += attention_mask

    attn_scores += attn_bias[None, None, :, :]
    
    topk = min(self.topk, key_length)
    _, topk_ids = torch.topk(attn_scores, topk, dim=-1) 

    # obtain the top-k attention
    topk_bias = torch.full((batch_size, num_attention_heads, query_length, key_length), mask_value, dtype=attn_scores.dtype, device=attn_scores.device)
    topk_bias.scatter_(-1, topk_ids, 0.)
    topk_bias[..., :topk, :] = 0.
    attn_scores += topk_bias

    attn_scores = F.softmax(attn_scores, dim=-1).to(value.dtype)
    attn_out = torch.bmm(attn_scores.view(-1, query_length, key_length), 
                            value.contiguous().view(-1, key_length, attn_head_size))
    attn_out = attn_out.view(batch_size, num_attention_heads, query_length, attn_head_size)

    return attn_out, attn_scores, topk_ids

@register_model("ret-approx-hf")
class RetApproxLM(HFLM):
    AUTO_MODEL_CLASS = None
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: Optional[Union[str, transformers.PreTrainedModel]] = "gpt2",
        backend: Optional[Literal["default", "causal", "seq2seq"]] = "default",
        # override whether the model should be treated as decoder-only (causal) or encoder-decoder (seq2seq)
        revision: Optional[str] = "main",
        subfolder: Optional[str] = None,
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ] = None,
        truncation: Optional[bool] = False,
        logits_cache: bool = True,
        max_length: Optional[int] = None,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        max_batch_size: Optional[int] = 64,
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
        add_bos_token: Optional[bool] = False,
        prefix_token_id: Optional[int] = None,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        parallelize: Optional[bool] = False,
        device_map_option: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[Union[str, os.PathLike]] = "./offload",
        # PEFT, delta weights and quantization options
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        topk: Optional[int] = 10,
        layerwise_topk_analysis: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            backend=backend,
            revision=revision,
            subfolder=subfolder,
            tokenizer=tokenizer,
            truncation=truncation,
            logits_cache=logits_cache,
            max_length=max_length,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            trust_remote_code=trust_remote_code,
            use_fast_tokenizer=use_fast_tokenizer,
            add_bos_token=add_bos_token,
            prefix_token_id=prefix_token_id,
            parallelize=parallelize,
            device_map_option=device_map_option,
            max_memory_per_gpu=max_memory_per_gpu,
            max_cpu_memory=max_cpu_memory,
            offload_folder=offload_folder,
            peft=peft,
            delta=delta,
            autogptq=autogptq,
            **kwargs,
        )
        for name, module in self._model.named_modules():
            if hasattr(module, "_attn"):
                orig_attn = module._attn
                module._attn = types.MethodType(retrieval_attn, module)
                module._approx_attn = types.MethodType(topk_attn, module)
                module.gt_ids = []
                module.ret_ids = []
                module.err_norm = []
                module.exact_topk_attn_ratios = []
                module.approx_topk_attn_ratios = []
                module.topk = topk
                module.is_causal = self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
                module.layerwise_topk_analysis = layerwise_topk_analysis

    @staticmethod
    def calculate_recall(gt_ids, pred_ids):
        recall = 0
        for gt_id, pred_id in zip(gt_ids, pred_ids):
            recall += np.intersect1d(gt_id, pred_id).shape[0] / gt_id.shape[0]
        return recall / len(gt_ids)

    def summarize(self):
        results = {}
        for name, module in self._model.named_modules():
            if hasattr(module, "gt_ids"):
                gt_ids = torch.cat(module.gt_ids, dim=1).numpy()
                ret_ids = torch.cat(module.ret_ids, dim=1).numpy()
                err_norms = torch.cat(module.err_norm, dim=1).mean(dim=-1).numpy()
                exact_topk_attn_ratios = torch.cat(module.exact_topk_attn_ratios, dim=1).mean(dim=-1).numpy()
                approx_topk_attn_ratios = torch.cat(module.approx_topk_attn_ratios, dim=1).mean(dim=-1).numpy()
                # calculate recall for each head in parallel
                recalls = joblib.Parallel(n_jobs=-1)(
                    joblib.delayed(self.calculate_recall)(gt_id, ret_id)
                    for gt_id, ret_id in zip(gt_ids, ret_ids)
                )
                # recalls = np.zeros(gt_ids.shape[0])
                # for head_id in range(gt_ids.shape[0]):
                #     recalls[head_id] = self.calculate_recall(gt_ids[head_id], ret_ids[head_id])
                results[name] = {
                    "recall": recalls,
                    "err_norm": err_norms,
                    "exact_topk_attn_ratio": exact_topk_attn_ratios,
                    "approx_topk_attn_ratio": approx_topk_attn_ratios,
                }
                module.gt_ids = []
                module.ret_ids = []
                module.err_norm = []
                module.exact_topk_attn_ratios = []
                module.approx_topk_attn_ratios = []
        return results
