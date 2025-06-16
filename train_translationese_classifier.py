# -*- coding:utf-8 _*-

import argparse
import numpy as np
import evaluate
import transformers
from datasets import load_dataset

from transformers import (
    Trainer,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback
)


def preprocess(sample, tokenizer, max_tokens=128):

    result = tokenizer(
        sample["text"],
        truncation=True,
        max_length=max_tokens,
        padding=False,
        return_tensors=None,
    )
    return result


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_train', type=str, required=True)
    parser.add_argument('--data_test', type=str, required=True)
    parser.add_argument('--val_set_size', type=float, default=0.02)
    parser.add_argument('--save_model', type=str, required=True)
    parser.add_argument('--base_model', type=str, default='GroNLP/bert-base-dutch-cased')
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--log_steps', type=int, default=100)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--eval_batch_size', type=int, default=1024)
    parser.add_argument('--train_batch_size', type=int, default=512)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--gradient_checkpointing', type=bool, default=True)
    parser.add_argument('--group_by_length', type=bool, default=False)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    transformers.set_seed(args.seed)

    print(
        f"Finetuning with params:\n"
        f"sedd: {args.seed}\n"
        f"data_path: {args.data_train}\n"
        f"data_test: {args.data_test}\n"
        f"val_set_size: {args.val_set_size}\n"
        f"save_model: {args.save_model}\n"
        f"base_model: {args.base_model}\n"
        f"resume_checkpoin: {args.resume_checkpoint}\n"
        f"max_epochs: {args.max_epochs}\n"
        f"warmup_steps: {args.warmup_steps}\n"
        f"learning_rate: {args.learning_rate}\n"
        f"max_tokens: {args.max_tokens}\n"
        f"log_steps: {args.log_steps}\n"
        f"eval_steps: {args.eval_steps}\n"
        f"save_steps: {args.save_steps}\n"   
        f"per_device_eval_batch_size: {args.eval_batch_size}\n"
        f"per_device_train_batch_size: {args.train_batch_size}\n"
        f"gradient_accumulation_steps: {args.gradient_accumulation_steps}\n"
        f"gradient_checkpointing: {args.gradient_checkpointing}\n"
        f"group_by_length: {args.group_by_length}\n"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label=id2label,
        label2id=label2id)

    data = load_dataset("json", data_files=args.data_path)

    train_valid = data["train"].train_test_split(
        test_size=args.val_set_size, shuffle=True, seed=args.seed
    )
    train_data = (
        train_valid["train"].shuffle().map(
            lambda example: preprocess(example, tokenizer, args.max_tokens), num_proc=32)
    )
    valid_data = (
        train_valid["test"].shuffle().map(
            lambda example: preprocess(example, tokenizer, args.max_tokens), num_proc=32)
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]

    print('[Info] {} train samples'.format(len(train_data)))
    print('[Info] {} valid samples'.format(len(valid_data)))


    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        compute_metrics=compute_metrics,
        args=transformers.TrainingArguments(
            disable_tqdm=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
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
        ),
        data_collator=transformers.DataCollatorWithPadding(tokenizer=tokenizer),
        callbacks=callbacks
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=args.resume_checkpoint)

    print('[Info] Evaluate on the test set')
    test_data = load_dataset("json", data_files=args.data_test)
    test_data = (
        test_data["train"].map(
            lambda example: preprocess(example, tokenizer, args.max_tokens), num_proc=32)
    )
    outputs = trainer.predict(test_data)
    print('[Info] {}'.format(outputs.metrics))


if __name__ == "__main__":
    main()
