#!/bin/bash
# launch the 4-seed GRPO runs (each job: 2 B200, ~2 h) — run on hpg in the repo
for s in 0 1 2 3; do
  sbatch -J grpo_s$s -p hpg-b200 --gres=gpu:b200:2 --mem=128gb --time=03:00:00 run_gpu.sh grpo_train.py $s
done
