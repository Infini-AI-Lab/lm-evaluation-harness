CUDA_VISIBLE_DEVICES=1,2,3,4 lm_eval --model sd \
    --model_args pretrained=meta-llama/Llama-2-70b-chat-hf,draft=meta-llama/Llama-2-7b-chat-hf,temperature=0.6,width=1\
    --tasks truthfulqa_gen\
    --device cuda:1,2,3,4 \
    --batch_size 1 \
    --num_fewshot 0 \
    --limit 0.1 \

CUDA_VISIBLE_DEVICES=1,2,3,4 lm_eval --model sd \
    --model_args pretrained=meta-llama/Llama-2-70b-chat-hf,draft=meta-llama/Llama-2-7b-chat-hf,temperature=0.6,width=1\
    --tasks truthfulqa_gen\
    --device cuda:1,2,3,4 \
    --batch_size 1 \
    --num_fewshot 1 \
    --limit 0.1 \

CUDA_VISIBLE_DEVICES=1,2,3,4 lm_eval --model sd \
    --model_args pretrained=meta-llama/Llama-2-70b-chat-hf,draft=meta-llama/Llama-2-7b-chat-hf,temperature=0.6,width=1\
    --tasks truthfulqa_gen\
    --device cuda:1,2,3,4 \
    --batch_size 1 \
    --num_fewshot 2 \
    --limit 0.1 \


CUDA_VISIBLE_DEVICES=1,2,3,4 lm_eval --model sd \
    --model_args pretrained=meta-llama/Llama-2-70b-chat-hf,draft=meta-llama/Llama-2-7b-chat-hf,temperature=0.6,width=1\
    --tasks truthfulqa_gen\
    --device cuda:1,2,3,4 \
    --batch_size 1 \
    --num_fewshot 3 \
    --limit 0.1 \

CUDA_VISIBLE_DEVICES=1,2,3,4 lm_eval --model sd \
    --model_args pretrained=meta-llama/Llama-2-70b-chat-hf,draft=meta-llama/Llama-2-7b-chat-hf,temperature=0.6,width=1\
    --tasks truthfulqa_gen\
    --device cuda:1,2,3,4 \
    --batch_size 1 \
    --num_fewshot 5 \
    --limit 0.1 \

    






