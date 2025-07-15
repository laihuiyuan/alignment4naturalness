# -*- coding:utf-8 _*-

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math
import torch
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from sacrebleu.metrics import BLEU
from comet import download_model, load_from_checkpoint

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from biasmt_metrics import textToLFP
from biasmt_metrics import *
from sfa_metrics import compute_sfa

model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)

model_path = download_model("Unbabel/wmt22-cometkiwi-da")
cometkiwi_model = load_from_checkpoint(model_path)

tokenizer = AutoTokenizer.from_pretrained("bert_ht_or")
ht_or_model = AutoModelForSequenceClassification.from_pretrained("bert_ht_or")
mt_ht_model = AutoModelForSequenceClassification.from_pretrained("bert_mt_ht")
mt_or_model = AutoModelForSequenceClassification.from_pretrained("bert_mt_or")
ht_or_model.to('cuda')
mt_ht_model.to('cuda')
mt_or_model.to('cuda')


def compute_translationese(mt_list, model, batch_size=1024):

    corr = 0
    for i in tqdm(range(0, len(mt_list), batch_size)):
        inputs = tokenizer.batch_encode_plus(
            mt_list[i:i+batch_size], 
            truncation=True, 
            padding=True, 
            return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs.to('cuda')).logits
            for x in logits:
                if x[1]>x[0]:
                    corr += 1
    return corr/len(mt_list)


def compute_sacrebleu(mt_list, ref_list):
    bleu = BLEU()
    score = bleu.corpus_score(mt_list, ref_list)

    return score.score


def comput_comet(src_list, mt_list, ref_list, batch_size=512):
    data = []
    for s, o, r in zip(src_list, mt_list, ref_list):
        data.append(
            {
            "src": s.strip(),
            "mt": o.strip(),
            "ref": r.strip()
            }
        )

    score = comet_model.predict(data, batch_size=batch_size, gpus=1)[1]

    return score


def comput_cometkiwi(src_list, mt_list, batch_size=512):
    data = []
    for s, o in zip(src_list, mt_list):
        data.append(
            {
            "src": s.strip(),
            "mt": o.strip(),
            }
        )

    score = cometkiwi_model.predict(data, batch_size=batch_size, gpus=1)[1]

    return score


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True,
                        help="data dir")
    parser.add_argument("--model", required=True,
                        help="model name")
    parser.add_argument("--file_unseen", required=True,
                        help="Unseen text for generating word lists")
    parser.add_argument("--dictionary", required=True,
                        type=str, help="bilingual aligned dictionary.")
    args = parser.parse_args()

    names = ['Auster', 'Baldacci', 'Barnes', 'Boyne', 'Carre', 'Franzen', 'French',
             'Golding', 'Grisham', 'Hemingway', 'Highsmith', 'Hosseini', 'Irving',
             'James', 'Joyce', 'Kerouac', 'King', 'Kinsella', 'Mitchell', 'Orwell',
             'Patterson', 'Pynchon', 'Roth', 'Rowling', 'Salinger', 'Slaughter',
             'Steinbeck', 'Tolkien', 'Twain', 'Wilde', 'Yalom']

    with open(args.file_unseen, "r") as f:
        unseen = [line.strip() for line in f.readlines()]

    with open(args.dictionary, "r") as bilingual_dict:
        dict_data = [line.rstrip() for line in bilingual_dict.readlines()]


    eval_scores = []

    for i in range(len(names)):
        src_path = os.path.join(args.dir, 'test.{}.4spm.en'.format(names[i]))
        tgt_path = os.path.join(args.dir, 'test.{}.4spm.nl'.format(names[i]))
        # out_path = os.path.join(args.dir, 'test.{}.4spm.nl'.format(names[i]))
        out_path = os.path.join(args.dir, '{}_{}.en-nl'.format(args.model, names[i]))
        with open(src_path, "r") as f:
            src_list = [line.strip() for line in f.readlines()]
        with open(tgt_path, "r") as f:
            tgt_list = [line.strip() for line in f.readlines()]
        with open(out_path, "r") as f:
            out_list = [line.strip() for line in f.readlines()]

        [b1, b2, b3] = textToLFP(unseen, out_list)
        ttr = compute_ttr(out_list)
        yules = compute_yules_i(out_list)
        mtld = compute_mtld(out_list)
        synttr, ptf, cdu = compute_sfa(out_list, dict_data)
        ht_or = compute_translationese(out_list, ht_or_model)
        mt_ht = compute_translationese(out_list, mt_ht_model)
        mt_or = compute_translationese(out_list, mt_or_model)
        command = 'sacrebleu {} -i {} -m bleu -b -w 4'.format(tgt_path, out_path)
        bleu = float(subprocess.run(command, 
                     shell=True, capture_output=True, text=True).stdout.strip())
        comet = comput_comet(src_list, out_list, tgt_list)
        cometkiwi = comput_cometkiwi(src_list, out_list)
        eval_scores.append([ht_or, mt_ht, mt_or, bleu, comet, cometkiwi, b1, b2, b3, ttr, 
                            yules, mtld, synttr, ptf, cdu])
    metrics = ['HT_OR', 'MT_HT', 'MT_OR', 'BLEU', 'COMET', 'COMETKIWI', 'B1', 'B2', 'B3', 'TTR', 
               'Yules', 'MTLD', 'SynTTR', 'PTF', 'CDU']
    print('{:15}| {}'.format('Name', ' | '.join(["{:6}".format(x) for x in metrics])))
    for name, score in zip(names, eval_scores):
        info = ' | '.join([f"{x:.{4 if x < 10 else 3 if x < 100 else 2}f}" for x in score])
        print('{:15}| {}'.format(name, info))

    eval_scores = np.array(eval_scores)
    avg = [eval_scores[:,i].mean() for i in range(len(metrics))]
    info = ' | '.join([f"{x:.{4 if x < 10 else 3 if x < 100 else 2}f}" for x in avg])
    print('{:15}| {}'.format('Average', info))


if __name__ == "__main__":
    main()

