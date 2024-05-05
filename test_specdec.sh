lm_eval --model sd \
    --model_args pretrained=meta-llama/Llama-2-7b-hf,draft=princeton-nlp/Sheared-LLaMA-1.3B\
    --tasks openbookqa,hellaswag\
    --device cuda:7 \
    --batch_size 1 \
    --limit 0.01 \
    






