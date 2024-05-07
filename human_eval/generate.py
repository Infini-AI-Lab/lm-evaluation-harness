import os
import sys
import re
from tqdm import tqdm
import json
import math
import gzip
import fire
import torch
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

def create_filepath(filepath):
    if os.path.exists(filepath):
        filename, ext = os.path.splitext(filepath)
        file_index = re.search(r"\d+$", filename)
        if file_index is None:
            filename = "{}1".format(filename)
            file_index = 1
        else:
            file_index = int(file_index.group(0))
            file_index = int(file_index)
        while os.path.exists("{}{}".format(filename, ext)):
            filename = filename[:-len(str(file_index))] + str(file_index+1)
            file_index += 1
        filepath = "{}{}".format(filename, ext)
    return filepath


def load_data(dataset_name):
    data_path_mapping = {
            "humaneval": "./HumanEval.jsonl.gz"
            }
    data_path = data_path_mapping[dataset_name]
    data = []
    if data_path.endswith(".jsonl.gz"):
        with gzip.open(data_path, "rt") as f:
            data = [json.loads(line) for line in f]
    elif data_path.endswith(".json"):
        with open(data_path, "r") as f:
            data = json.load(f)
    else:
        with open(data_path, "r") as f:
            data = [json.loads(line) for line in f]

    instructions = []
    task_ids = []
    task_ids = [ex["task_id"] for ex in data]
    instructions = [ex["prompt"] for ex in data]

    return task_ids, instructions


def main(
        output_dir: str = "./output_dir",
        base_model: str = "",
        batch_size: int =1,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_return_sequences: int = 20,

        ):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='meta-llama/Meta-Llama-3-8B'"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, legacy=False)

    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map='cuda:0',
            )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    tokenizer.padding_size = "left"
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # loading data
    task_ids, instructions = load_data('humaneval')
    prompts = instructions

    num_samples_this_rank = math.ceil(len(prompts))
    _start_idx = 0
    _end_idx = num_samples_this_rank
    prompts = prompts[_start_idx:_end_idx]
    task_ids = task_ids[_start_idx:_end_idx]

    output_strings = []
    for idx in tqdm(range(0, len(prompts), batch_size), desc="Rank {}".format(0)):
        batch_prompts = prompts[batch_size*idx:batch_size*(idx+1)]
        # tokenization
        inputs = tokenizer(batch_prompts, 
                           truncation=False,
                           padding=False,
                           )
        input_ids = inputs["input_ids"]
        batch_max_length = max(len(_input_ids) for _input_ids in input_ids)
        new_input_ids, attention_mask = [], []
        for _input_ids in input_ids:
            padding_size = batch_max_length - len(_input_ids)
            new_input_ids.append([tokenizer.pad_token_id]*padding_size + _input_ids)
            attention_mask.append([False]*padding_size + [True]*len(_input_ids))
        input_ids = torch.LongTensor(new_input_ids)
        input_ids = input_ids.to('cuda:0')
        attention_mask = torch.BoolTensor(attention_mask)
        attention_mask = attention_mask.to('cuda:0')

        this_batch_size = input_ids.shape[0]

        try:
            generation_config = GenerationConfig(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    num_return_sequences=num_return_sequences,
                    )

            with torch.no_grad():
                output_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        generation_config=generation_config,
                        return_dict_in_generate=False,
                        max_new_tokens=128,
                        pad_token_id=tokenizer.eos_token_id,
                        )
        except torch.cuda.OutOfMemoryError:
            print("Rank {} out of memory ... continue ...".format(0))
            torch.cuda.empty_cache()
            output_strings.extend([""]*(this_batch_size*num_return_sequences))
            continue
        batch_output_strings = [tokenizer.decode(s,
                                           skip_special_tokens=True,
                                           ignore_tokenization_space=True)
                          for s in output_ids]
        output_strings.extend(batch_output_strings)

    filepath = create_filepath(os.path.join(output_dir, "generation.jsonl"))
    with open(filepath, "w") as f:
        for j, output_str in enumerate(output_strings):
            if output_str == "":
                continue
            task_id = task_ids[j//num_return_sequences]
            json.dump({"task_id": task_id, "completion": output_str, "rank": 0}, f)
            f.write("\n")

    print("Rank {} finished. Predictions saved to {}".format(0, filepath))
 
if __name__ == "__main__":
    fire.Fire(main)
