#!/usr/bin/env bash
# Runs INSIDE one pinned image (one GPU, --network none). Caps and selection come from the environment
# set by run_k_selection.sh. No `set -e`: the exit code is recorded.
set -u
nvidia-smi -L > /o/gpu.txt 2>&1
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv >> /o/gpu.txt 2>&1
timeout "$K_MAX_SECONDS" python3 /k/kselect/run_k.py --engine "$K_ENGINE" --plan /k/plan.json --bank /k/bank \
  --out /o --max-seconds "$(( K_MAX_SECONDS - 30 ))" --max-batches "${K_MAX_BATCHES:-0}" --cells "${K_CELLS:-}" \
  > /o/run.log 2>&1
echo "run_k_exit=$?" > /o/exit.txt
