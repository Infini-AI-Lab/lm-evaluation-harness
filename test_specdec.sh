CUDA_VISIBLE_DEVICES=0 lm_eval --model sd \
    --model_args pretrained=meta-llama/Llama-2-7b-hf,draft=JackFram/llama-160m,temperature=0.6,width=4\
    --tasks gsm8k\
    --device cuda:0 \
    --batch_size 1 \
    --limit 0.1 \
    






