import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats
import torch

from pytablewriter import LatexTableWriter, MarkdownTableWriter
import matplotlib.pyplot as plt
import seaborn as sns

import lm_eval.api as api
import lm_eval.evaluator as evaluator
import lm_eval.models.utils
from lm_eval import tasks, utils


os.environ["TOKENIZERS_PARALLELISM"] = "false"
eval_logger = utils.eval_logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", default="ret-approx-hf", help="model type to study"
    )
    parser.add_argument(
        "--pretrained", default="EleutherAI/pythia-70m", help="name of model to compare"
    )
    parser.add_argument(
        "--model_args", help="huggingface model args <arg>=<value>", default=""
    )
    parser.add_argument("--tasks", type=str, default="arc_easy,hellaswag")
    parser.add_argument(
        "--limit",
        type=float,
        default=100,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=8,
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Top-k items to consider for recall calculation",
    )
    parser.add_argument(
        "--layerwise_topk_analysis",
        action="store_true",
        help="Perform layer-wise top-k analysis, otherwise errors from bottom layers will propagate",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        help="Logging verbosity",
    )
    return parser.parse_args()

def make_table(result_dict):
    """
    result_dict: dict
        Dictionary containing layer-wise results
        Each layer has a dictionary containing two keys:
            recalls: a recall np array containing head-wise recall scores
            err_norms: a np array containing head-wise error norms
    """
    md_writer = MarkdownTableWriter()
    md_writer.headers = ["Layer", "Mean Recall", "Std Recall", "Mean Error Norm", "Std Error Norm"]
    values = []
    for layer, results in result_dict.items():
        mean_recall = np.mean(results["recall"])
        std_recall = np.std(results["recall"])
        mean_err_norm = np.mean(results["err_norm"])
        std_err_norm = np.std(results["err_norm"])
        values.append([layer, f"{mean_recall:.3f}", f"{std_recall:.3f}", f"{mean_err_norm:.3f}", f"{std_err_norm:.3f}"])
    md_writer.value_matrix = values

    return md_writer.dumps()

def plot_heatmap(result_dict, out_dir):
    exact_topk_attn_ratios = np.stack(
        [result_dict[layer]["exact_topk_attn_ratio"] for layer in result_dict.keys()]
    )
    ax = sns.heatmap(exact_topk_attn_ratios, vmin=0, vmax=1)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("Exact Top-k Attention Ratio")
    plt.savefig(f"{out_dir}/exact_topk_attn_ratio.png")
    plt.clf()
    approx_topk_attn_ratios = np.stack(
        [result_dict[layer]["approx_topk_attn_ratio"] for layer in result_dict.keys()]
    )
    ax = sns.heatmap(approx_topk_attn_ratios, vmin=0, vmax=1)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")  
    ax.set_title("Approx Top-k Attention Ratio")
    plt.savefig(f"{out_dir}/approx_topk_attn_ratio.png")


if __name__ == "__main__":
    args = parse_args()
    task_names = args.tasks.split(",")
    task_manager = tasks.TaskManager()
    lm = api.registry.get_model(args.model_type).create_from_arg_string(
        args.model_args,
        {
            "pretrained": args.pretrained,
            "batch_size": args.batch,
            "max_batch_size": None,
            "device": args.device,
            "topk": args.topk,
            "layerwise_topk_analysis": args.layerwise_topk_analysis,
        },
    )

    os.makedirs(args.out_dir, exist_ok=True)
    for task_name in task_names:
        task_dict = tasks.get_task_dict(task_name, task_manager)
        e = evaluator.evaluate(
            lm, task_dict, limit=args.limit, verbosity=args.verbosity
        )
        retrieval_results = lm.summarize()
        print(make_table(retrieval_results))
        plot_heatmap(retrieval_results, args.out_dir)

        
        
        
    

