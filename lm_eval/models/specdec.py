import copy
import os
from datetime import timedelta
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from accelerate import (
    Accelerator,
    DistributedType,
    InitProcessGroupKwargs,
    find_executable_batch_size,
)
from packaging import version
from peft import PeftModel
from peft import __version__ as PEFT_VERSION
from tqdm import tqdm
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
)

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    clear_torch_cache,
    get_dtype,
    pad_and_concat,
    stop_sequences_criteria,
)
from .huggingface import HFLM
def get_sampling_logits(logits :torch.Tensor, top_p:float, T: float, replicate = False):
    if replicate:
        logits = logits.clone()
    if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                torch.nn.functional.softmax(sorted_logits / T, dim=-1), dim=-1)
                filter = cumulative_probs > top_p
                filter[..., 1:] = filter[..., :-1].clone()
                filter[..., 0] = 0
                indices_to_remove = filter.scatter(-1, sorted_indices, filter)
                logits[indices_to_remove] = float('-inf')
    return logits

def get_residual(p: torch.Tensor, q:torch.Tensor):
    residual = p - q
    residual[residual < 0] = 0.0
    residual = residual / (residual.sum(dim=-1).unsqueeze(-1) + 1e-9)
    return residual

eval_logger = utils.eval_logger


def _get_accelerate_args(
    device_map_option: Optional[str] = "auto",
    max_memory_per_gpu: Optional[Union[int, str]] = None,
    max_cpu_memory: Optional[Union[int, str]] = None,
    offload_folder: Optional[str] = "./offload",
) -> dict:
    """Returns the kwargs needed to apply `accelerate` in `AutoModel.from_pretrained`."""
    max_memory = {}
    if max_memory_per_gpu is not None:
        max_memory_per_gpu_map = {
            device_idx: max_memory_per_gpu
            for device_idx in range(torch.cuda.device_count())
        }
        max_memory.update(max_memory_per_gpu_map)
    if max_cpu_memory is not None:
        max_memory["cpu"] = max_cpu_memory

    args = {}
    if max_memory:
        args["max_memory"] = max_memory
    args["device_map"] = device_map_option
    args["offload_folder"] = offload_folder
    return args


@register_model("sd")
class SDLM(TemplateLM):
    """
    An abstracted Huggingface model class. Enables usage with both models of
    `transformers.AutoModelForCausalLM` and `transformers.AutoModelForSeq2SeqLM` classes.

    Supports data-parallel multi-GPU with HF Accelerate.
    """

    AUTO_MODEL_CLASS = None
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: Optional[Union[str, transformers.PreTrainedModel]] = "princeton-nlp/Sheared-LLaMA-1.3B",
        draft: Optional[Union[str, transformers.PreTrainedModel]] = "JackFram/llama-160m",
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
        **kwargs,
    ) -> None:
        super().__init__()

        # optionally: take in an already-initialized transformers.PreTrainedModel
        if not isinstance(pretrained, str):
            eval_logger.warning(
                "`pretrained` model kwarg is not of type `str`. Many other model arguments may be ignored. Please do not launch via accelerate or use `parallelize=True` if passing an existing model this way."
            )
            assert not parallelize, "`parallelize=True` is not compatible with passing pre-initialized model to `pretrained`"
            self._model = pretrained
            self._device = self._model.device
            self._config = self._model.config
            gpus = 0

            if tokenizer:
                assert isinstance(
                    tokenizer, transformers.PreTrainedTokenizer
                ) or isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
                self.tokenizer = tokenizer
            else:
                # Get tokenizer
                model_name = self._model.name_or_path
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_name,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    use_fast=use_fast_tokenizer,
                )

        else:
            assert isinstance(device, str)
            assert isinstance(pretrained, str)
            assert isinstance(batch_size, (int, str))

            gpus = torch.cuda.device_count()
            accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
            accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
            if accelerator.num_processes > 1:
                self.accelerator = accelerator

            if not (parallelize or accelerator.num_processes > 1):
                # use user-passed device
                device_list = set(
                    ["cuda", "cpu"]
                    + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
                    + ["mps", "mps:0"]
                )
                if device and device in device_list:
                    self._device = torch.device(device)
                    eval_logger.info(f"Using device '{device}'")
                    if device in ("mps", "mps:0") and version.parse(
                        torch.__version__
                    ) < version.parse("2.1"):
                        raise RuntimeError(
                            f"mps requires torch >= 2.1. You have {torch.__version__}"
                        )
                else:
                    eval_logger.info("Device not specified")
                    eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                    self._device = (
                        torch.device("cuda")
                        if torch.cuda.is_available()
                        else torch.device("cpu")
                    )
            else:
                if device != "cuda":
                    eval_logger.info(
                        f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model."
                    )
                # TODO: include in warning that `load_in_8bit` etc. affect this too
                self._device = torch.device(device)

            # TODO: update this to be less of a hack once subfolder is fixed in HF
            revision = revision + ("/" + subfolder if subfolder is not None else "")

            self._get_config(
                pretrained,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )

        # determine which of 'causal' and 'seq2seq' backends to use
        self._get_backend(
            config=self.config, backend=backend, trust_remote_code=trust_remote_code
        )

        # if we passed `pretrained` as a string, initialize our model now
        if isinstance(pretrained, str):
            
            self._create_model(
                pretrained=pretrained,
                draft=draft,
                revision=revision,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
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

        # access self._model through self.model property outside this method
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            self.model.tie_weights()
            
            self.draft.eval()
            self.draft.tie_weights()

        if isinstance(pretrained, str) and (gpus >= 1 or str(self.device) == "mps"):
            # TODO: can remove this whole snippet except in the mps case, perhaps?
            if not (parallelize or autogptq or hasattr(self, "accelerator")):
                # place model onto device requested manually,
                # if not using HF Accelerate or device_map
                # or any other option that preloads model onto device
                try:
                    self.draft.to(self.device)
                except ValueError:
                    eval_logger.debug(
                        "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
                    )

        self._create_tokenizer(
            pretrained,
            tokenizer,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast_tokenizer=use_fast_tokenizer,
        )

        self.truncation = truncation
        self.logits_cache = logits_cache
        self.vocab_size = self.tokenizer.vocab_size
        # select (or create) a pad token to use
        if self.tokenizer.pad_token:
            pass
        elif self.tokenizer.unk_token:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        elif self.tokenizer.eos_token:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        else:
            if getattr(self.config, "model_type", None) == "qwen":
                # Qwen's trust_remote_code tokenizer does not allow for adding special tokens
                self.tokenizer.pad_token = "<|endoftext|>"
            elif (
                self.tokenizer.__class__.__name__ == "RWKVWorldTokenizer"
                or self.tokenizer.__class__.__name__ == "Rwkv5Tokenizer"
            ):
                # The RWKV world tokenizer, does not allow for adding special tokens / setting the pad token (which is set as 0)
                # The additional tokenizer name check is needed, as there exists rwkv4 models with neox tokenizer
                # ---
                # Note that the world tokenizer class name, might change in the future for the final huggingface merge
                # https://github.com/huggingface/transformers/pull/26963
                assert self.tokenizer.pad_token_id == 0
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        # TODO: override this for Gemma
        self.add_bos_token = add_bos_token
        if getattr(self.config, "model_type", None) == "gemma":
            self.add_bos_token = True
            eval_logger.info(
                f"Model type is '{self.config.model_type}', a BOS token will be used as Gemma underperforms without it."
            )

        self._max_length = max_length

        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = max_batch_size

        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)



        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            eval_logger.info(
                f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}"
            )

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def model(self):
        return self._model
    @property
    def draft(self):
        return self._draft

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self) -> int:
        return 64

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _get_backend(
        self,
        config: Union[transformers.PretrainedConfig, transformers.AutoConfig],
        backend: Optional[Literal["default", "causal", "seq2seq"]] = "default",
        trust_remote_code: Optional[bool] = False,
    ) -> None:
        """
        Helper method during initialization.
        Determines the backend ("causal" (decoder-only) or "seq2seq" (encoder-decoder))
        model type to be used.
        """
        assert backend in ["default", "causal", "seq2seq"]

        if backend != "default":
            # if we've settled on non-default backend, use that manually
            if backend == "causal":
                self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
            elif backend == "seq2seq":
                self.AUTO_MODEL_CLASS = transformers.AutoModelForSeq2SeqLM
            eval_logger.info(
                f"Overrode HF model backend type, and using type '{backend}'"
            )
        else:
            # determine and use the default HF backend for this model, based on its config + metadata.
            if (
                getattr(config, "model_type")
                in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
            ):
                # first check if model type is listed under seq2seq models, since some
                # models like MBart are listed in both seq2seq and causal mistakenly in HF transformers.
                # these special cases should be treated as seq2seq models.
                self.AUTO_MODEL_CLASS = transformers.AutoModelForSeq2SeqLM
            elif (
                getattr(self.config, "model_type") in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
            ):
                self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
            else:
                if not trust_remote_code:
                    eval_logger.warning(
                        "HF model type is neither marked as CausalLM or Seq2SeqLM. \
                    This is expected if your model requires `trust_remote_code=True` but may be an error otherwise."
                    )
                # if model type is neither in HF transformers causal or seq2seq model registries
                # then we default to AutoModelForCausalLM
                self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

        assert self.AUTO_MODEL_CLASS in [
            transformers.AutoModelForCausalLM,
            transformers.AutoModelForSeq2SeqLM,
        ]
        return None

    def _get_config(
        self,
        pretrained: str,
        revision: str = "main",
        trust_remote_code: bool = False,
    ) -> None:
        self._config = transformers.AutoConfig.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

    def _create_model(
        self,
        pretrained: str,
        draft:str,
        temperature:float = 0.6,
        top_p: float = 1.0,
        greedy: bool = False,
        width: int = 4,
        revision: Optional[str] = "main",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        trust_remote_code: Optional[bool] = False,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        # (accelerate naive PP (device_map) options)
        parallelize: Optional[bool] = False,
        device_map_option: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        # PEFT, delta weights and quantization options
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        **kwargs,
    ) -> None:
        """
        Initializes an HF or HF-compatible PreTrainedModel from scratch
        inside HFLM, using the kwargs passed into self.__init__().

        Also handles functionality such as AutoGPTQ usage and PEFT wrapping.

        For future similar extensions to AutoGPTQ that are not core to HF's ecosystem,
        (such as PyTorch models that are nearly, but not quite, fully mirroring
        HF's public interface relied on in this HFLM class)
        please consider subclassing HFLM and overriding this and other methods as needed.
        """
       
        model_kwargs = kwargs if kwargs else {}
        
        self._model = self.AUTO_MODEL_CLASS.from_pretrained(
                pretrained,
                revision=revision,
                trust_remote_code=trust_remote_code,
                torch_dtype=get_dtype(dtype),
                device_map="auto",
                **model_kwargs,
                
            )
        
        self._draft = self.AUTO_MODEL_CLASS.from_pretrained(
                draft,
                torch_dtype=get_dtype(dtype),
                revision=revision,
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )
        
        self.temperature = temperature
        self.top_p = top_p
        self.greedy = greedy
        self.width = width
        return None

    def _create_tokenizer(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ],
        revision: Optional[str] = "main",
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
    ) -> None:
        """
        Helper method during initialization.

        Create a tokenizer object corresponding to the correct
        tokenizer for value of `pretrained`, or use the pre-initialized tokenizer passed.
        """

        if tokenizer:
            if isinstance(tokenizer, str):
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    tokenizer,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    use_fast=use_fast_tokenizer,
                )
            else:
                assert isinstance(
                    tokenizer, transformers.PreTrainedTokenizer
                ) or isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
                self.tokenizer = tokenizer
        else:
            # Get tokenizer based on 'pretrained'
            if isinstance(pretrained, str):
                model_name = pretrained
            else:
                # get the HF hub name via accessor on model
                model_name = self.model.name_or_path
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name,
                revision=revision,
                trust_remote_code=trust_remote_code,
                use_fast=use_fast_tokenizer,
            )
        return None

    def _detect_batch_size(self, requests=None, pos: int = 0):
         
        raise NotImplementedError
        
    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        """ """
        # default for None - empty dict, use predefined tokenizer param
        # used for all models except for CausalLM or predefined value
        special_tokens_kwargs = {}

        # by default for CausalLM - false or self.add_bos_token is set
        if add_special_tokens is None:
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                special_tokens_kwargs = {
                    "add_special_tokens": False or self.add_bos_token
                }
        # otherwise the method explicitly defines the value
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        add_special_tokens = {}
        if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
            add_special_tokens = {"add_special_tokens": False or self.add_bos_token}

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            **add_special_tokens,
        )
        if left_truncate_len:
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.tokenizer.padding_side = old_padding_side

        return encoding["input_ids"], encoding["attention_mask"]

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def loglikelihood(
        self, requests, disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        pass
    
    def _select_cont_toks(
        self, logits: torch.Tensor, contlen: int = None, inplen: int = None
    ) -> torch.Tensor:
        pass

    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        pass

    def _batch_scheduler(self, pos, n_reordered_requests):
        sched = pos // int(len(n_reordered_requests) / self.batch_schedule)
        if sched in self.batch_sizes:
            return self.batch_sizes[sched]
        if (len(self.batch_sizes) > 1) and (
            self.batch_sizes[sched - 1] == self.max_batch_size
        ):
            # if previous batch size is already maximal, skip recomputation
            self.batch_sizes[sched] = self.max_batch_size
            return self.batch_sizes[sched]
        print(
            f"Passed argument batch_size = auto:{self.batch_schedule}. Detecting largest batch size"
        )
        self.batch_sizes[sched] = self._detect_batch_size(n_reordered_requests, pos)
        print(f"Determined largest batch size: {self.batch_sizes[sched]}")
        return self.batch_sizes[sched]
    def _model_call(self, inps, attn_mask=None, labels=None):
       pass
    
    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: int = None,
    ) -> List[Tuple[float, bool]]:
        pass

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        pass
    def specdec_eval(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        res = []

        def _collate(req: Tuple[str, dict]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(req[0])
            return -len(toks), req[0]
        
        
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running specdec_eval requests",
        )
        
        batch_size = 1
        batch_fn = None

        
        re_ords = Collator(
            [reg.args for reg in requests],
            sort_fn=_collate,
            group_fn=lambda x: x[1],
        )
        chunks = re_ords.get_batched(n=batch_size, batch_fn=batch_fn)
        
        for chunk in chunks:
            contexts, _ = zip(*chunk)
            
            max_gen_toks = self.max_gen_toks
            max_ctx_len = self.max_length - max_gen_toks
            
            context_enc, _ = self.tok_batch_encode(
                contexts,
                left_truncate_len=max_ctx_len,
                truncation=self.truncation,
            )
            
            acceptance_rate, num_samples = self.specdec(context_enc)
            
            res.append({
                    "alpha": acceptance_rate.cumsum_(dim=0),
                    "num_samples": num_samples
                })
                
            pbar.update(1)
        # reorder this group of results back to original unsorted form
       
        res = re_ords.get_original(res)

        pbar.close()

        return res

    def specdec(self, context_enc: torch.LongTensor):
        
        input_ids = context_enc.to(self.device)
        initial_len = input_ids.shape[1]
        eos_tokens = [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id, self.tokenizer.bos_token_id]
        past_key_values = None
        tokens = input_ids.clone()
        T = self.temperature
        P = self.top_p
        greedy = self.greedy
        max_gen_toks = self.max_gen_toks
        
        with torch.inference_mode():
            for _ in range(max_gen_toks):
                    outputs = self._model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
                    past_key_values = outputs.past_key_values
                    logits :torch.Tensor = outputs.logits
                    logits = logits[...,-1,:]
                    logits = get_sampling_logits(logits, P, T, replicate=False)
                    logits = torch.nn.functional.softmax(logits/T, dim=-1)
                    
                    new_token = torch.multinomial(logits, num_samples=1)
                    
                    if new_token.item() in eos_tokens: break
                    input_ids = new_token
                    tokens = torch.cat([tokens, input_ids], dim=-1)

        #     print(tokens)
            #tokens = self._model.generate(inputs=input_ids, do_sample=True, temperature=T, top_k=32000, top_p=1.0, max_new_tokens=max_gen_toks)
            
            num_samples = tokens.shape[1] - initial_len
            acceptance_rate = torch.zeros(self.width).to(self.device)
            
            target_logits :torch.Tensor= self._model(input_ids=tokens).logits
            draft_logits :torch.Tensor= self._draft(input_ids=tokens).logits
                
            target_logits = get_sampling_logits(target_logits, P, T, replicate=False)
                
                
            target_logits = target_logits[...,initial_len:,:]
            draft_logits = draft_logits[...,initial_len:,:]
            target_proba = torch.nn.functional.softmax(target_logits/T, dim=-1)[0]
            draft_proba = torch.nn.functional.softmax(draft_logits/T, dim=-1)[0]
            
            for i in range(target_proba.shape[0]):
                token_acceptance_rate = torch.zeros(self.width).to(self.device)
                
                
                token_target_prob = target_proba[i]
                # token_draft_prob = q[i]
                #draft_model_prob.append(q[i].cpu())
                token_draft_logits = draft_logits[0][i]

                token_draft_prob = F.softmax(token_draft_logits / T, dim=-1).squeeze(0)
                sampled_token = token_draft_prob.multinomial(num_samples=1, replacement=True)
                
                
                token_acceptance_rate[0] = min(1.0, (token_target_prob[sampled_token]/ token_draft_prob[sampled_token]))

                token_target_prob = get_residual(token_target_prob, token_draft_prob)
                
                
                for j in range(self.width-1):
                    token_draft_logits[sampled_token] = - torch.inf
                    token_draft_prob = F.softmax(token_draft_logits / (T), dim=-1).squeeze(0)
                    if torch.isnan(token_draft_prob).long().sum() >= 1:
                        break
                    token_draft_prob = token_draft_prob / token_draft_prob.sum(-1)
                    sampled_token = token_draft_prob.multinomial(num_samples=1, replacement=True)
                    branch_token_acceptance_rate = min(1, token_target_prob[sampled_token]/ token_draft_prob[sampled_token])
                    token_acceptance_rate[j+1] = (1 - token_acceptance_rate.sum()) * branch_token_acceptance_rate
                    
                    token_target_prob = get_residual(token_target_prob, token_draft_prob)
                acceptance_rate = acceptance_rate + token_acceptance_rate
                 
        
        
        return acceptance_rate.cpu()/num_samples, num_samples