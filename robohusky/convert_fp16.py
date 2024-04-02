"""
Usage:
srun -p INTERN2 --job-name='convert_2_fp16' --gres=gpu:0 --cpus-per-task=8 --quotatype="auto" python -u husky/convert_fp16.py --in-checkpoint work_dirs/llm/husky-13b/zh_bell/checkpoint-9500 --out-checkpoint work_dirs/llm/husky-13b/zh_bell/
"""
import argparse
import os.path

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def convert_fp16(in_checkpoint, out_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(in_checkpoint, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        in_checkpoint, torch_dtype=torch.float16, low_cpu_mem_usage=False
    )
    if not os.path.exists(out_checkpoint):
        os.mkdir(out_checkpoint)
    model.save_pretrained(out_checkpoint)
    tokenizer.save_pretrained(out_checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-checkpoint", type=str, help="Path to the model")
    parser.add_argument("--out-checkpoint", type=str, help="Path to the output model")
    args = parser.parse_args()

    convert_fp16(args.in_checkpoint, args.out_checkpoint)
