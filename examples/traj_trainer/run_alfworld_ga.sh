#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common/env.sh"

GAMMA="${GAMMA:-10.0}"
NPROC="${NPROC:-2}"
RUN_TAG="gamma${GAMMA//./p}"

MODEL_PATH="${MODEL_PATH:-$ALFWORLD_BASE_MODEL}"
DATA_PARQUET="${DATA_PARQUET:-$COLLECTION_DIR/output/npo_forget_all_tasks_mixed100.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$CKPT_ROOT/traj_alfworld_ga_${RUN_TAG}}"

torchrun --nproc_per_node="$NPROC" --standalone unlearn/ga_unlearn.py \
    --model_path "$MODEL_PATH" \
    --data_parquet "$DATA_PARQUET" \
    --forget_task_types mixed_forget \
    --retain_task_types auto \
    --output_dir "$OUTPUT_DIR" \
    --gamma "$GAMMA" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --gradient_checkpointing \
    --report_to_wandb \
    --run_name "traj_alfworld_ga_${RUN_TAG}"

echo "run tag: $RUN_TAG"
echo "output dir: $OUTPUT_DIR/final"
