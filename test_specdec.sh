CUDA_VISIBLE_DEVICES=0,1,2,3 lm_eval --model sd \
    --model_args pretrained=meta-llama/Llama-2-7b-hf,draft=JackFram/llama-160m,temperature=0.7\
    --tasks openbookqa,gsm8k,coqa,truthfulqa\
    --device cuda:0 \
    --batch_size 1 \
    --limit 0.01 \
    






