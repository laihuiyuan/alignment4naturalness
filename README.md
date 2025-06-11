# [Multi-perspective Alignment for Increasing Naturalness in Neural Machine Translation (ACL 2025)](https://arxiv.org/abs/2412.08473)

> **Abstract:**
Neural machine translation (NMT) systems amplify lexical biases present in their training data, leading to artificially impoverished language in output translations. These language-level characteristics render automatic translations different from text originally written in a language and human translations, which hinders their usefulness in for example creating evaluation datasets. Attempts to increase naturalness in NMT can fall short in terms of content preservation, where increased lexical diversity comes at the cost of translation accuracy. Inspired by the reinforcement learning from human feedback framework, we introduce a novel method that rewards both naturalness and content preservation. We experiment with multiple perspectives to produce more natural translations, aiming at reducing machine and human translationese. We evaluate our method on English-to-Dutch literary translation, and find that our best model produces translations that are lexically richer and exhibit more properties of human-written language, without loss in translation accuracy.

## Quick Start

### Train Classifiers
```
python train_translationese_classifier.py \
    --max_epochs 2 \
    --max_tokens 256 \
    --val_set_size 0.008 \
    --data_train train_data_path \
    --data_test test_data_path \
    --save_model checkpoint_path \
```

### Train base MT Model
```
python train_base_model.py \
    --data_train train_data_path \
    --data_valid valid_data_path \
    --base_model facebook/bart-base \
    --save_model checkpoint_path \
    --max_epochs 10 \
    --warmup_steps 1000 \
    --learning_rate 1e-4 \
    --max_tokens 128 \
    --log_steps 100 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --eval_batch_size 512 \
    --train_batch_size 256 \
    --gradient_accumulation_steps 2 \
```

### Train alignment MT Model
```
python train_align_model.py \
    --data_train train_data_path \
    --data_valid valid_data_path \
    --base_model base_mt_model \
    --save_model checkpoint_path \
    --max_epochs 1 \
    --learning_rate 2e-5 \
    --max_tokens 128 \
    --log_steps 100 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --eval_batch_size 512 \
    --train_batch_size 256 \
    --gradient_accumulation_steps 2 \
```

### Inference
```
python infererence.py \
   --model model_path \
   --src_file source_file_path \
   --out_file output_file_path \
   --num_beams 5 \
   --batch_size 512 \
   --max_new_tokens 128
```

### Evaluate
```
python evaluation/eval.py \
  --dir data_path \
  --model model_path \
  --file_unseen unseen_file \
  --dictionary dictionary_file
```

## Citation
If you use any content from this repository, please cite our paper:
```
@article{lai-etal-2024-multi,
    title={Multi-perspective Alignment for Increasing Naturalness in
    Neural Machine Translation},
    author={Huiyuan Lai and Esther Ploeger and Rik van Noord and Antonio Toral},
    year={2025},
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association
    for Computational Linguistics",
    month = Jul,
    year = "2025",
    url={https://arxiv.org/abs/2412.08473}
}
```
