pip install fire -q
git clone https://github.com/openai/human-eval
pip install -e human-eval
# model inference
export CUDA_VISIBLE_DEVICES=0
NUM_ITERATIONS=10
rm -rf ./output_dir
for _ in $(seq $NUM_ITERATIONS);
do
    python generate.py \
        --output_dir ./output_dir \
        --base_model 'meta-llama/Meta-Llama-3-8B' \
        --batch_size 1 \
        --num_return_sequences 20 \
        --temperature 0.1 \
        --top_p 0.75 \
        --top_k 40
done

# Calculating pass@k with k=1,10,100
python eval_humaneval.py --prediction_dir ./output_dir