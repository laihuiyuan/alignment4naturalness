# -*- coding: utf-8 -*-

import os
import torch
import argparse
from tqdm import tqdm
from torch import cuda

from transformers import (
    Trainer,
    AutoTokenizer,
    BartForConditionalGeneration,
    GenerationConfig
)

device = 'cuda' if cuda.is_available() else 'cpu'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--src_file', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--num_beams', default=5, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--max_new_tokens', default=128, type=int)

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = BartForConditionalGeneration.from_pretrained(args.model)
    model.to(device).eval()

    generation_config=GenerationConfig(
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        early_stopping=True,
        use_cache=True,
        do_sample=False,
    )

    src_seqs = []
    with open(args.src_file, 'r') as f:
        for line in f.readlines():
            src_seqs.append(' '.join(line.strip().split()[:150]))

    if os.path.exists(args.out_file):
        cur_num = len(open(args.out_file, 'r').readlines())
    else:
        cur_num = 0

    with open(args.out_file, 'a+') as f:
        for idx in tqdm(
            range(cur_num, len(src_seqs), args.batch_size)):

            inputs = tokenizer.batch_encode_plus(
                src_seqs[idx: idx + args.batch_size],
                padding=True, return_tensors='pt')
            inputs.to(device)

            outputs = model.generate(
                **inputs, 
                generation_config=generation_config)

            texts = tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False)

            for x in texts:
                f.write(x.strip() + '\n')


if __name__ == "__main__":
    main()
