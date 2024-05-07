# HumanEval: Hand-Written Evaluation Set
This guide will help you quickly start evaluating your model using the HumanEval dataset by following the steps outlined below.

## TL;DR
Run the following commands to evaluate your model on the HumanEval dataset:
```bash
bash eval.sh
```

## Initial Setup and Installation
To begin, ensure that you have all necessary tools and libraries installed. You can use the following script to set up your environment:

```
pip install fire -q
git clone https://github.com/openai/human-eval
pip install -e human-eval
```

This script does the following:

- Installs fire, a library for automatically generating command line interfaces.
- Clones the human-eval repository from GitHub.
- Installs the human-eval package in editable mode (-e), which means changes to the source files will immediately affect the installed package.

## Running the Evaluation
Loop through the specified number of iterations to generate predictions using your model. After generating the predictions, you can calculate the metric pass@k for different values of k (e.g., 1, 10, 100).
```bash
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
```

n = NUM_ITERATIONS * batch_size * num_return_sequences, where n is used to estimate pass@k as in the Codex paper.

$${pass@k} = \underset{\text { Problems }}{\mathbb{E}}\left[1-\frac{C^{k}{n-c}}{C^{k}{n}}\right]$$

Here we choose n = 200 as employed in the paper, which results in `NUM_ITERATIONS=10`, `batch_size=1`, `num_return_sequences=20`.
