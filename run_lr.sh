#!/bin/bash
# grpo_learning_rate: sweep around the default 1e-5 (default runs = run_seeds.sh)
for lr in 3e-6 3e-5 1e-4; do
  for s in 0 1; do
    sbatch -J grpo_lr${lr}_s$s -p hpg-b200 --gres=gpu:b200:2 --mem=128gb --time=03:00:00 run_gpu.sh grpo_train.py $s $lr
  done
done
