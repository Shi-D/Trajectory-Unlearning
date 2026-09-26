#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

: "${MODEL_DIR:?Set MODEL_DIR}"
: "${EVAL_OUTPUT_DIR:?Set EVAL_OUTPUT_DIR}"
BASELINE_TRAJECTORIES="${BASELINE_TRAJECTORIES:-$COLLECTION_DIR/output/all_trajectories_0.jsonl}"

if [[ -n "${TRAIN_CKPT_DIR:-}" ]]; then
    LATEST_STEP_DIR=$(ls -d "$TRAIN_CKPT_DIR"/global_step_* 2>/dev/null | sed -E 's/.*global_step_([0-9]+)$/\1 &/' | sort -n | tail -1 | awk '{print $2}')
    if [[ -z "$LATEST_STEP_DIR" ]]; then
        echo "No global_step_* checkpoint found under $TRAIN_CKPT_DIR" >&2
        exit 1
    fi
    echo "Merging checkpoint: $LATEST_STEP_DIR -> $MODEL_DIR"
    python scripts/model_merger.py merge \
        --backend fsdp \
        --local_dir "$LATEST_STEP_DIR/actor" \
        --target_dir "$MODEL_DIR"
fi

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Model directory not found: $MODEL_DIR" >&2
    exit 1
fi
echo "Evaluating model: $MODEL_DIR"

cd "$COLLECTION_DIR"
export START_VLLM=1
export MODEL_PATH="$MODEL_DIR"
export ALL_TASKS=1
export BATCH_SIZE=64
export OUTPUT_DIR="$EVAL_OUTPUT_DIR"

./collection.sh

NEW_TRAJECTORIES="$EVAL_OUTPUT_DIR/all_trajectories.jsonl"
if [[ ! -f "$NEW_TRAJECTORIES" ]]; then
    echo "Collected trajectories not found: $NEW_TRAJECTORIES" >&2
    exit 1
fi

echo "======================================"
echo "Exact action-sequence match vs. baseline:"
echo "======================================"
python compare_action_similarity.py "$BASELINE_TRAJECTORIES" "$NEW_TRAJECTORIES" \
    | tee "$EVAL_OUTPUT_DIR/exact_match_comparison.txt"

if [[ "${MIXED100_BREAKDOWN:-0}" == "1" ]]; then
    echo "======================================"
    echo "Forget/retain breakdown:"
    echo "======================================"
    python analyze_mixed100_results.py "$NEW_TRAJECTORIES" \
        | tee "$EVAL_OUTPUT_DIR/mixed100_forget_retain_breakdown.txt"
fi
