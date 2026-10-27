#!/bin/bash
#SBATCH --account=weixue --qos=weixue
#SBATCH --partition=hpg-b200 --gres=gpu:b200:2
#SBATCH --cpus-per-task=8 --mem=128gb --time=01:00:00
#SBATCH --output=logs/%x_%j.log
# SimpleSafetyTests: Llama 3.3 70B Instruct marks each response safe / unsafe (2 GPUs, tensor parallel)
# usage (on hpg, in the repo):  sbatch run_safety.sh results/sst_baseline.jsonl
cd /blue/weixue/weixue/AI/alignment
mkdir -p logs
nvidia-smi --query-gpu=name,memory.total --format=csv
uv run python scripts/evaluate_safety.py --input-path "$1" \
    --model-name-or-path meta-llama/Llama-3.3-70B-Instruct \
    --num-gpus 2 \
    --output-path "${1%.jsonl}_judged.jsonl"
