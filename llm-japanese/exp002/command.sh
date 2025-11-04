#!/bin/bash

python pretraining.py \
  --tokenizer hf \
  --hf_name rinna/japanese-gpt2-small \
  --data_dir ../dataset/001 \
  --use_pretokenized \
  --pretokenized_dir ../dataset/001_pretok \
  --n_epochs 5 \
  --batch_size 32 \
  --eval_freq 100 \
  --print_sample_iter 1000 \
  --save_ckpt_freq 10000 \
  --output_dir model_checkpoints \
  --lr 0.001 \
  --warmup_ratio 0.2 \
  --initial_lr 1e-5 \
  --min_lr 1e-5