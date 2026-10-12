#!/bin/bash
# HiPerGator -> Mac.  usage: ./pull.sh prompting_baselines.py run_baseline.sh results
R=hpg:/blue/weixue/weixue/AI/alignment
cd "$(dirname "$0")"
for f in "$@"; do mkdir -p "$(dirname "$f")"; scp -r "$R/$f" "$(dirname "$f")/"; done
