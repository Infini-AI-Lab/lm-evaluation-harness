## Analysing Performance of Retrieval-based Attention Approximations

### Usage

`python3 scripts/check_retrieved_perf.py --pretrained <model_name_or_path> --model_type <model_type> --tasks <tasks_list-comma separated> --batch 4 --limit 100 --topk 10 --layerwise_topk_analysis`

`topk`: topk used for recall calculation. </br>
`layerwise_topk_analysis`: restricts error propagation from bottom layer. approximations.

example cmd:

`python3 scripts/check_retrieved_perf.py --pretrained EleutherAI/pythia-70m  --model_type ret-approx-hf --tasks wikitext --batch 4 --limit 10 --topk 10 --layerwise_topk_analysis`

### Testing New Model
To test a new model, create a wrapper around the model using the parent class `lm_eval.models.retrieval_lm.RetApproxLM`

The new model's attention block is assumed to have a modified `_attn` function.<br />
**NOTE:** The modified function should return `approx_attn_output, approx_attn_scores(default=None), topk_ids` tuple.<br />
Replace `module._approx_attn = types.MethodType(topk_attn, module)` by `module._approx_attn = MethodType(module._attn, module)` as needed. </br>
**TODO:** Support for `FlashAttention` can be made if required.

### Sample output

`scripts/check_retrieved_perf.py` will output the following merics:
- **Recall@k**: Index approximation error metric
- **Output error norm**: QKV output error norm / QKV output norm ratio

The mean and std deviations are computed across heads.
Here the following output is for exact top-k search, hence recall scores are perfectly 1. 

|           Layer            |Mean Recall|Std Recall|Mean Error Norm|Std Error Norm|
|----------------------------|----------:|---------:|--------------:|-------------:|
|gpt_neox.layers.0.attention |          1|         0|          0.705|         0.409|
|gpt_neox.layers.1.attention |          1|         0|          0.727|         0.406|
|gpt_neox.layers.2.attention |          1|         0|          0.372|         0.257|
|gpt_neox.layers.3.attention |          1|         0|          0.226|         0.250|
|gpt_neox.layers.4.attention |          1|         0|          0.229|         0.231|
|gpt_neox.layers.5.attention |          1|         0|          0.358|         0.225|
|gpt_neox.layers.6.attention |          1|         0|          0.288|         0.203|
|gpt_neox.layers.7.attention |          1|         0|          0.143|         0.173|
|gpt_neox.layers.8.attention |          1|         0|          0.017|         0.044|
|gpt_neox.layers.9.attention |          1|         0|          0.001|         0.000|
|gpt_neox.layers.10.attention|          1|         0|          0.000|         0.000|
|gpt_neox.layers.11.attention|          1|         0|          0.000|         0.000|

In addition, the script will also output a visualization of accumulated attention fraction captured by top-k retrieved items. It will generate visualizations for both exact search and approximate search.

Here are sample outputs:

<div style="display: flex; justify-content: center;">
  <img src="../img/exact_topk_attn_ratio.png" alt="Exact top-k" style="width: 45%;">
  <img src="../img/approx_topk_attn_ratio.png" alt="Approx top-k" style="width: 45%;">
</div>