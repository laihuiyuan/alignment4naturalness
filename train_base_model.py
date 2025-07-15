# -*- coding:utf-8 _*-

import warnings
import argparse

import torch
import transformers
from datasets import load_dataset

from transformers import (
    BartConfig,
    Trainer,
    AutoTokenizer,
    BartForConditionalGeneration,
    EarlyStoppingCallback
)

warnings.filterwarnings("ignore")

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

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # loss = 0.2*loss + self.rewards_loss(labels, outputs, temperature=1.0)

        return (loss, outputs) if return_outputs else loss


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

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    config=BartConfig.from_pretrained(args.base_model)
    model = BartForConditionalGeneration(config)
    # model = BartForConditionalGeneration.from_pretrained(
    #     args.base_model,
    # )
    model.resize_token_embeddings(len(tokenizer))
    model.is_parallelizable = True
    model.model_parallel = True

    train = load_dataset("json", data_files=args.data_train)
    valid = load_dataset("json", data_files=args.data_valid)

    train_data = (
        train["train"].shuffle().map(
            lambda example: preprocess(example, tokenizer, args.max_tokens), num_proc=32)
    )
    valid_data = (
        valid["train"].shuffle().map(
            lambda example: preprocess(example, tokenizer, args.max_tokens), num_proc=32)
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
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            bf16=True,
            tf32=True,
            optim="adamw_torch",
            lr_scheduler_type='cosine',
            evaluation_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=1,
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

    trainer.train(resume_from_checkpoint=args.resume_checkpoint)


if __name__ == "__main__":
    main()

