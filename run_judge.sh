#!/bin/bash
#SBATCH --account=weixue --qos=weixue
#SBATCH --partition=hpg-b200 --gres=gpu:b200:2
#SBATCH --cpus-per-task=8 --mem=128gb --time=02:00:00
#SBATCH --output=logs/%x_%j.log
# AlpacaEval win rate: Llama 3.3 70B Instruct judges our outputs vs GPT-4 Turbo (2 GPUs, tensor parallel)
# usage (on hpg, in the repo):  sbatch run_judge.sh results/alpaca_eval_baseline.json
cd /blue/weixue/weixue/AI/alignment
mkdir -p logs
nvidia-smi --query-gpu=name,memory.total --format=csv
uv run alpaca_eval --model_outputs "$1" \
    --reference_outputs data/alpaca_eval/alpaca_eval_gpt4_turbo.json \
    --annotators_config scripts/alpaca_eval_vllm_llama3_3_70b_fn \
    --base-dir .
