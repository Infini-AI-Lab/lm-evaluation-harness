# Xsum

To run the task,
'''
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks xsum \
    --device cuda:0 \
    --batch_size 2
'''