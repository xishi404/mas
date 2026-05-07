#!/bin/bash
# Launch the full MaaS pipeline (Phase 1 → 1.5 → 2 → 3) on each dataset.
# Each dataset runs in its own tmux window so they execute concurrently.
#
# Usage:
#   ./run_all_experiments.sh                   # default: train + gate + test
#   ./run_all_experiments.sh --test_only       # skip Phase 1 (use existing checkpoint)
#
# Override defaults via env vars:
#   MODEL=gpt-4o-mini
#   LATENCY_WEIGHT=0.005
#   COST_WEIGHT=3.0
#   THRESHOLD=0.3
#   NUM_LAYERS=4
#   MAX_CONCURRENT=3

set -e

MODEL=${MODEL:-gpt-4o-mini}
LATENCY_WEIGHT=${LATENCY_WEIGHT:-0.005}
COST_WEIGHT=${COST_WEIGHT:-3.0}
THRESHOLD=${THRESHOLD:-0.3}
NUM_LAYERS=${NUM_LAYERS:-4}
MAX_CONCURRENT=${MAX_CONCURRENT:-3}

EXTRA_FLAGS=""
for arg in "$@"; do
    case "$arg" in
        --test_only)  EXTRA_FLAGS="$EXTRA_FLAGS --test_only" ;;
        --train_only) EXTRA_FLAGS="$EXTRA_FLAGS --train_only" ;;
        *) echo "Unknown flag: $arg" >&2; exit 1 ;;
    esac
done

timestamp=$(date +%Y%m%d_%H%M%S)
session="maas_${timestamp}"
mkdir -p logs

tmux new-session -d -s "$session"
window_num=0

for dataset in GSM8K MATH MMLU_Pro; do
    name="${dataset}"
    log="logs/${dataset}_${timestamp}.log"

    if [ $window_num -eq 0 ]; then
        tmux rename-window -t "${session}:0" "$name"
    else
        tmux new-window -t "$session" -n "$name"
    fi

    tmux send-keys -t "${session}:${name}" \
        "python -m experiments.run_main \
            --dataset $dataset \
            --exec_model_name $MODEL --opt_model_name $MODEL \
            --latency_weight $LATENCY_WEIGHT \
            --cost_weight $COST_WEIGHT \
            --threshold $THRESHOLD \
            --num_layers $NUM_LAYERS \
            --max_concurrent $MAX_CONCURRENT \
            $EXTRA_FLAGS \
            2>&1 | tee $log" C-m

    window_num=$((window_num + 1))
    sleep 1
done

echo "Launched in tmux session: $session"
echo "Attach: tmux attach -t $session"
