#!/bin/bash
# Mac -> HiPerGator.  usage: ./push.sh grade_baselines.py results/r1_zero.prompt.jsonl
R=hpg:/blue/weixue/weixue/AI/alignment
cd "$(dirname "$0")"
for f in "$@"; do scp -r "$f" "$R/$(dirname "$f")/"; done
