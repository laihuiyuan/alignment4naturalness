# -*- coding:utf-8 _*-

import warnings
import argparse

import torch
import transformers
from datasets import load_dataset

from torch.nn import CrossEntropyLoss

from transformers import (
    pipeline,
    BartConfig,
    Trainer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BartForConditionalGeneration,
    EarlyStoppingCallback,
    GenerationConfig,
)
from comet import download_model, load_from_checkpoint
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
generation_config = GenerationConfig(
#    num_beams=4,
    max_new_tokens=128,
    temperature=1.0,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=True,
)

model_path = download_model("Unbabel/wmt22-cometkiwi-da")
comet_model = load_from_checkpoint(model_path)

reward_tokenizer = AutoTokenizer.from_pretrained("bert_mt_or")
reward_model = AutoModelForSequenceClassification.from_pretrained("bert_mt_or").to('cuda')


def harmonic_mean(tensor1, tensor2):

    tensor_sum = 1 / tensor1 + 1 / tensor2
    harmonic_mean = 2 / tensor_sum

    return harmonic_mean


def tokenize(text, tokenizer, max_tokens=512, add_eos_token=True):

    result = tokenizer(
        text,
        truncation=True,
        max_length=max_tokens,
        padding=False,
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    return result


def preprocess(sample, tokenizer, max_tokens):
    tokenized_sample = tokenize(sample['src'], tokenizer, max_tokens)
    tokenized_sample['labels'] = tokenize(sample['tgt'], tokenizer, max_tokens)["input_ids"]

    return tokenized_sample


class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        outputs = model(
            input_ids=inputs["input_ids"],
           attention_mask=inputs["attention_mask"],
            labels=inputs["labels"]
        )
        loss_ref = outputs["loss"]
        # if not model.training:
        #     return (loss_ref, outputs) if return_outputs else loss_ref

        with torch.no_grad():
            outputs_g=model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=generation_config
            )

        outputs_g = outputs_g[:,1:].contiguous()

        rewards = self.compute_rewards(inputs['input_ids'].clone(), inputs['labels'].clone(), outputs_g)
        
        outputs_g[outputs_g==tokenizer.pad_token_id]=-100
        logits = model(
            input_ids=inputs["input_ids"].clone(),
            attention_mask=inputs["attention_mask"].clone(),
            labels=outputs_g,
        )["logits"]
        loss_fct = CrossEntropyLoss(reduction='none')
        loss_sam = loss_fct(logits.view(-1, logits.size(-1)), outputs_g.view(-1))
        loss_sam = loss_sam.view(outputs_g.size())
        
        loss_sam = loss_sam*rewards
        padding_mask = loss_sam.eq(0)
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        if num_active_elements>0:
            loss_sam = loss_sam.sum() / num_active_elements
        else:
            loss_sam = loss_sam.sum()
  
        loss = loss_ref*0.5 + loss_sam

        return (loss, outputs_g) if return_outputs else loss


    @staticmethod
    def compute_rewards(src, tgt, outputs):
        tgt[tgt==-100] = 2
        tgt_texts = tokenizer.batch_decode(tgt, skip_special_tokens=True)
        src_texts = tokenizer.batch_decode(src, skip_special_tokens=True)

        sampled_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        inputs = reward_tokenizer.batch_encode_plus(
            sampled_texts, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            tc_scores = torch.softmax(reward_model(**inputs.to('cuda')).logits,-1)[:,1]
        tc_scores[tc_scores<0.5] = 0

        data = [
                {
                    "src": s.strip(),
                    "mt":  m.strip(),
                } for s, m in zip(src_texts, sampled_texts)
            ]
        cm_scores = comet_model.predict(data, batch_size=256, gpus=1, progress_bar=False)
        cm_scores = torch.tensor(cm_scores.scores).to('cuda')
        cm_scores[cm_scores<0.85]=0

        rewards = harmonic_mean(tc_scores, cm_scores)
        rewards = rewards.unsqueeze(-1)

        return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_train', type=str, required=True)
    parser.add_argument('--data_valid', type=str, required=True)
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--save_model', type=str, required=True)
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--log_steps', type=int, default=100)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--gradient_checkpointing', type=bool, default=True)
    parser.add_argument('--group_by_length', type=bool, default=False)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    transformers.set_seed(args.seed)

    print(
        f"Finetuning with params:\n"
        f"sedd: {args.seed}\n"
        f"data_train: {args.data_train}\n"
        f"data_valid: {args.data_valid}\n"
        f"base_model: {args.base_model}\n"
        f"save_model: {args.save_model}\n"
        f"resume_checkpoin: {args.resume_checkpoint}\n"
        f"max_epochs: {args.max_epochs}\n"
        f"warmup_steps: {args.warmup_steps}\n"
        f"learning_rate: {args.learning_rate}\n"
        f"max_tokens: {args.max_tokens}\n"
        f"log_step: {args.log_steps}\n"
        f"eval_step: {args.eval_steps}\n"
        f"save_step: {args.save_steps}\n"        
        f"per_device_eval_batch_size: {args.eval_batch_size}\n"
        f"per_device_train_batch_size: {args.train_batch_size}\n"
        f"gradient_accumulation_steps: {args.gradient_accumulation_steps}\n"
        f"gradient_checkpointing: {args.gradient_checkpointing}\n"
        f"group_by_length: {args.group_by_length}\n"
    )

    model = BartForConditionalGeneration.from_pretrained(
        args.base_model,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.is_parallelizable = True
    model.model_parallel = True

    train = load_dataset("json", data_files=args.data_train)
    valid = load_dataset("json", data_files=args.data_valid)

    train_data = (
        valid["train"].shuffle().map(
            lambda example: preprocess(example, tokenizer, args.max_tokens))
    )
    valid_data = (
        valid["train"].shuffle().map(
            lambda example: preprocess(example, tokenizer, args.max_tokens))
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]

    print('[Info] {} train samples'.format(len(train_data)))
    print('[Info] {} valid samples'.format(len(valid_data)))


    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.Seq2SeqTrainingArguments(
            disable_tqdm=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            per_device_eval_batch_size=args.eval_batch_size,
            per_device_train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            bf16=True,
            tf32=True,
            optim="adamw_torch",
            lr_scheduler_type='constant',
            evaluation_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=10,
            load_best_model_at_end=True,
            logging_steps=args.log_steps,
            output_dir=args.save_model,
            group_by_length=args.group_by_length,
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={
                'use_reentrant':args.gradient_checkpointing},
            # deepspeed='deepspeed_config.json',
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=callbacks
    )
    model.config.use_cache = False
    eval_results = trainer.evaluate()
    print(f"Initial Validation Loss: {eval_results['eval_loss']:.4f}")
    exit()

    trainer.train(resume_from_checkpoint=args.resume_checkpoint)


if __name__ == "__main__":
    main()


