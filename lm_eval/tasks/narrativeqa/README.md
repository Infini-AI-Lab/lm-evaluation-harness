# Narrativeqa

To run the task,
'''
lm_eval --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct \
    --tasks narrativeqa \
    --device cuda:0 \
    --batch_size 2
'''