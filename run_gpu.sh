#!/bin/bash
#SBATCH --account=weixue --qos=weixue
#SBATCH --partition=hpg-rtx6000 --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8 --mem=64gb --time=02:00:00
#SBATCH --output=logs/%x_%j.log
# usage (on hpg, in the repo):  sbatch run_gpu.sh prompting_baselines.py
# B200 instead:  sbatch -p hpg-b200 --gres=gpu:b200:1 run_gpu.sh prompting_baselines.py
cd /blue/weixue/weixue/AI/alignment
mkdir -p logs results
nvidia-smi --query-gpu=name,memory.total --format=csv
uv run python "$@"
