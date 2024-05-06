# Measuring PPL

## Usage 

Please run `python measuring_ppl.py --loading_from_checkpoint <arg1> --max_length <arg2> --batch_size <arg3> --dataset_name <arg4>` 
##### Arguments 

- loading_from_checkpoint: path to the checkpoint of the model (default is the pretrained checkpoint) 
- max_length: length of the tokenizer truncation max_length (default is 256) 
- batch_size: batch size used for evaluation 
- dataset_name: default None, but if specified to one specific, the evaluation will only run for the one with name given here 

## Dataset Support

- C4 (10k examples in the 150th json file) 
- OpenWebText (10k examples Skylion007/openwebtext) 
- WikiText (4.36k examples) 
- Wikipedia (10k examples at max from 20220301.en) 

## Example Score

In the following scores, xx(yy), xx refers to the CE loss, while yy refers to the ppl 
| Model Name   | Sequence Length | C4         | OpenWebText | WikiText   | Wikipedia  |
|--------------|-----------------|------------|------------|------------|-------------|
| Llama 2 7B   |       256       | 2.06 (7.83) | 2.00 (7.39) | 2.42 (11.19) | 1.50 (4.49) |
| Llama 2 13B  |       256       | 1.96 (7.08) | 1.91 (6.72) | 2.27 (9.65) | 1.30 (3.67) |
| Llama 2 7B   |       512       | 2.00 (7.42) | 1.92 (6.82) | 2.52 (12.38) | 1.54 (4.67) |
| Llama 2 13B  |       512       | 1.91 (6.73) | 1.82 (6.18) | 2.36 (10.59) | 1.35 (3.85) | 
