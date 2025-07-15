# -*- coding:utf-8 _*-

import argparse
import numpy as np
import evaluate
import torch
import transformers
from tqdm import tqdm
from datasets import load_dataset

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_model', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()

    transformers.set_seed(args.seed)

#    print(
#        f"Finetuning with params:\n"
#        f"sedd: {args.seed}\n"
#        f"data_path: {args.data_path}\n"
#        f"save_model: {args.save_model}\n"
#        f"batch_size: {args.batch_size}\n"
#    )
    
    tokenizer = AutoTokenizer.from_pretrained("bert-nl")
    model = AutoModelForSequenceClassification.from_pretrained(args.save_model)
    model.to('cuda')


    data = []
    with open(args.data_path ,'r') as f:
        for line in f.readlines():
            data.append(line.strip())

    corr = 0
    for i in tqdm(range(0, len(data), args.batch_size)):
        inputs = tokenizer.batch_encode_plus(
            data[i:i+args.batch_size], 
            truncation=True, 
            padding=True, 
            return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs.to('cuda')).logits
            for x in logits:
                if x[1]>x[0]:
                    corr += 1
    print('[Info] total: {} | corr: {} | acc: {}'.format(
        len(data), corr, round(corr/len(data), 4)))





if __name__ == "__main__":
    main()

