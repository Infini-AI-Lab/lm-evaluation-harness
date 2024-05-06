# this script is mainly used for measuring the perplexity of the model 
import torch 
import argparse 

##### the below packages are for utility functions ##### 
import datasets 
from datasets import load_dataset 
import sys 
import os 
from tqdm import tqdm 
import torch.nn.functional as F 
import time 
import numpy as np 
import inspect 
from termcolor import colored 
from torch import nn 
from torch.utils.data import random_split 
from packaging import version 
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union 
from itertools import chain 
seed_value = 42 # Set a global seed for reproducibility 
from transformers import set_seed 
set_seed(seed_value) 
import subprocess 
import warnings 

##### The following code is optional ##### 
current_dir = os.path.dirname(__file__) 
parent_dir = os.path.dirname(current_dir) 
src_folder = os.path.join(parent_dir, "src") 
sys.path.append(src_folder) 

##### The following imports only covers the common cases, please add more if needed ##### 
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM 
from transformers import GPTNeoXForCausalLM 
from transformers import LlamaConfig, LlamaPreTrainedModel 
from transformers import LlamaTokenizer 
from transformers import LlamaForCausalLM 
from transformers import Trainer, TrainingArguments 
from transformers import DataCollatorForLanguageModeling 
from transformers.generation.utils import GenerationConfig 
from transformers import BitsAndBytesConfig 

##### Setting the device ##### 
rank = os.environ.get("RANK") 
print("the rank is {}".format(rank)) 
if rank is None: 
    rank = 0 
torch_device = 'cuda:{}'.format(rank) if torch.cuda.is_available() else 'cpu' 

##### The default wandb is enabled ##### 
try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False 

##### Getting the commit hash (Optional) ##### 
commit_hash = None 
def get_git_commit_hash():
    try:
        # Run the git command to get the current commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip() 
        # Decode from bytes to string
        return commit_hash.decode('utf-8') 
    except subprocess.CalledProcessError:
        # Handle cases where the git command fails (e.g., not a git repository)
        return None 
commit_hash = get_git_commit_hash()[: 7] # only 7 digits 
print("the commit hash is {}".format(commit_hash)) 

##### Getting the hash of time ##### 
import datetime 
hash_of_time = str(datetime.datetime.now()).split('.')[-1] 
print("the hash of time is {}".format(hash_of_time)) 

##### Getting the hostname ##### 
import socket 
hostname = socket.gethostname()
print("Hostname:", hostname) 

##### Getting the arguments ##### 
parser = argparse.ArgumentParser(
                    prog='MeasuringPPL',
                    description='What the program does',
                    epilog='Text at the bottom of help') 

parser.add_argument("--loading_from_checkpoint", type = str, default = None) 
parser.add_argument("--batch_size", type = int, default = 8) 
parser.add_argument("--dataset_name", type = str, choices = ["c4", "OpenWebText", "wikitext", "Wikipedia"], default = None) 
# TODO: add the statistics of the Llama-2-7B model 

args = parser.parse_args() 

##### Setting the directories ##### 
if "lovelace" in hostname: 
    dir_sdata = "/home/yangzho6/slimpajama/SlimPajama-627B/test/chunk1/" 
    dir_c4 = "/home/yangzho6/c4_parts/downloads/" # for C4 dataset, I used the jsonl files to load in the data 
else: 
    dir_models = "/fsx-storygen/beidic/yang/model_checkpoints/" 
    dir_sdata = "/fsx-storygen/beidic/yang/c4llm_synthesized/" 

model_name = args.model_name 
text_eval = "evaluation_printout_{}_{}_{}.txt".format(commit_hash, hash_of_time, model_name) 

class CustomTrainer(Trainer): 
    # Sometimes custom models need to have its own forward function 
    # If that is the case, please override the compute_loss function 
    def compute_loss(self, model, inputs, return_outputs = False): 
        pass 

##### Below I gave an example of how to measure the perplexity of the model please replace the tokenizer and model for your use ##### 
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 
if tokenizer.pad_token is not None: 
    print("tokenizer has pad token {}".format(tokenizer.pad_token)) 
else: 
    tokenizer.pad_token = tokenizer.eos_token 
    print("We now use eos_token as pad token") 
tokenizer.padding_side = "left" 

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir = dir_models) 

datacollator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False) 
training_args = TrainingArguments(
    output_dir = dir_models, 
    per_device_eval_batch_size = 8, 
    do_train = False, 
    do_eval = True, 
    label_names = ["labels"], # this argument is optional for default models, but vital for custom models 
) 

trainer = Trainer(
    model = model, 
    args = training_args, 
    data_collator = datacollator, 
) 

def get_dataset(datasetname): 
    if datasetname == "c4llm_synthesized": 
        # datasetnew = load_dataset('json', data_files = dfiles, split = "train[:10000]") 
        dfiles = [] 
        if "lovelace" in hostname: 
            # filename = "c4synthesized_file1_kernel7_0.json" 
            filename = "c4synthesized_file1_1_0.json" 
            # filename = "c4synthesized_file11_1_0.json" 
            # filename = "c4synthesized_file1_kernel7.json" 
            # dfiles.append(dir_c4llmsynthesized + "{}/".format("tinyllama") + filename) 
            dfiles.append(dir_c4llmsynthesized + filename) 
            datasetnew = load_dataset("json", data_files = dfiles, split = "train[:10000]") 
        else: 
            filename = "c4synthesized_file1_kernel7_{}_combined.json".format(7) 
            dfiles.append(dir_c4llmsynthesized + "{}_topk{}/".format("tinyllama", "na") + filename) 
            datasetnew = load_dataset("json", data_files = dfiles, split = "train[:10000]") 
    elif datasetname == "c4": 
        dfiles = [] 
        # filename = "c4_file1.json" 
        filename = "c4_file150.json" 
        dfiles.append(dir_c4 + filename) 
        datasetnew = load_dataset("json", data_files = dfiles, split = "train[:10000]") 
    elif datasetname == "pg19": 
        datasetnew = load_dataset('emozilla/pg19', split = "train[:10000]") 
    elif datasetname == "cnn_dailymail": # we need to use special processing for this dataset 
        datasetnew = load_dataset("cnn_dailymail", "3.0.0", split = "test[:10000]") 
    elif datasetname == "openwebtext": 
        datasetnew = load_dataset("Skylion007/openwebtext", split = "train[:10000]") 
    elif datasetname == "xsum": # we need to use special processing for this dataset 
        datasetnew = load_dataset("xsum", split = "test[:10000]") 
    else: 
        raise ValueError("dataset_name is not recognized") 

    def encode_with_truncationspecialized(examples): 
        tokdictionary = tokenizer(examples['text'][100000 : 100000 + 3000], padding = "max_length", max_length = 260 if args.kernel_size == 7 else 259, 
                        return_attention_mask = True, return_tensors = "pt", truncation = True, 
                        add_special_tokens = True) 
        # tokdictionary = tokenizer(examples['text'], padding = "max_length", max_length = 260, 
        #                          return_attention_mask = True, return_tensors = "pt", truncation = True, 
        #                          add_special_tokens = True) 
        newdictionary = {} 
        newdictionary['input_ids'] = tokdictionary['input_ids'].squeeze(0) 
        newdictionary['attention_mask'] = tokdictionary['attention_mask'].squeeze(0) 
        return newdictionary 

    def encode_with_truncation(examples): 
        # tokdictionary = tokenizer(examples['text'][100000 : 100000 + 3000], padding = "max_length", max_length = 260, 
        #                  return_attention_mask = True, return_tensors = "pt", truncation = True, 
        #                  add_special_tokens = True) 
        tokdictionary = tokenizer(examples['text'], padding = "max_length", max_length = 260 if args.kernel_size == 7 else 259, 
                                return_attention_mask = True, return_tensors = "pt", truncation = True, 
                                add_special_tokens = True) 
        newdictionary = {} 
        newdictionary['input_ids'] = tokdictionary['input_ids'].squeeze(0) 
        newdictionary['attention_mask'] = tokdictionary['attention_mask'].squeeze(0) 
        return newdictionary 
    
    def encode_text_summary(examples): # cnn_dailymail uses "article" 
        tokdictionary = tokenizer(examples['article'], padding = "max_length", max_length = 260 if args.kernel_size == 7 else 259, 
                                return_attention_mask = True, return_tensors = "pt", truncation = True, 
                                add_special_tokens = True) 
        newdictionary = {} 
        newdictionary['input_ids'] = tokdictionary['input_ids'].squeeze(0) 
        newdictionary['attention_mask'] = tokdictionary['attention_mask'].squeeze(0) 
        return newdictionary 
    
    def encode_text_summary_xsum(examples): # xsum uses "document" 
        tokdictionary = tokenizer(examples["document"], padding = "max_length", max_length = 260 if args.kernel_size == 7 else 259, 
                                return_attention_mask = True, return_tensors = "pt", truncation = True, 
                                add_special_tokens = True) 
        newdictionary = {} 
        newdictionary['input_ids'] = tokdictionary['input_ids'].squeeze(0) 
        newdictionary['attention_mask'] = tokdictionary['attention_mask'].squeeze(0) 
        return newdictionary 

    def unflatten_list_func(examples): 
        examples['input_ids'] = examples['input_ids'].squeeze(0) 
        examples['attention_mask'] = examples['attention_mask'].squeeze(0) 

    # datasetnew = datasetnew.map(encode_with_truncation, batched = True, num_proc = 8) 
    if datasetname == "pg19": 
        datasetnew = datasetnew.map(encode_with_truncationspecialized, num_proc = 8) 
        datasetnew.set_format(type = "torch", columns = ["input_ids", "attention_mask", "text"]) 
    elif datasetname == "xsum": 
        datasetnew = datasetnew.map(encode_text_summary_xsum, num_proc = 8) 
        datasetnew.set_format(type = "torch", columns = ["input_ids", "attention_mask", "document"]) 
    elif datasetname == "cnn_dailymail": 
        datasetnew = datasetnew.map(encode_text_summary, num_proc = 8) 
        datasetnew.set_format(type = "torch", columns = ["input_ids", "attention_mask", "article"]) 
    else: 
        datasetnew = datasetnew.map(encode_with_truncation, num_proc = 8) 
        datasetnew.set_format(type = "torch", columns = ["input_ids", "attention_mask", "text"]) 
    # datasetnew = datasetnew.map(unflatten_list_func, num_proc = 8) 
    # datasetnew = datasetnew.map(unflatten_list_func, num_proc = 8) 
    return datasetnew 

datasetlist = ["c4", "OpenWebText", "wikitext", "Wikipedia"] 
ce_loss_list = [] 
ppl_list = [] 

for datasetname in datasetlist: 
    eval_dataset = get_dataset(datasetname) 
    results = trainer.evaluate(eval_dataset = eval_dataset) 
    ce_loss_list.append(results["eval_loss"]) 
    ppl_list.append(results["eval_perplexity"]) 
    print(results) 

for idx, datasetname in enumerate(datasetlist): 
    print("{}: ce_loss is {}, ppl is {}".format(datasetname, ce_loss_list[idx], ppl_list[idx])) 
