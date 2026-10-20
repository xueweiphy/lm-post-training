#!/bin/bash
# grpo_prompt_ablation: the other two prompts at the default lr (r1_zero = run_seeds.sh)
for p in question_only r1_zero_three_shot_gsm8k; do
  for s in 0 1 2; do
    sbatch -J grpo_${p:0:8}_s$s -p hpg-b200 --gres=gpu:b200:2 --mem=128gb --time=03:00:00 run_gpu.sh grpo_train.py $s 1e-5 $p
  done
done
