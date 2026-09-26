#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common/env.sh"

BETA="${BETA:-0.1}"
NPROC="${NPROC:-2}"
RUN_TAG="beta${BETA//./p}"

MODEL_PATH="${MODEL_PATH:-$ALFWORLD_BASE_MODEL}"
DATA_PARQUET="${DATA_PARQUET:-$COLLECTION_DIR/output/dpo_forget_mixed100_pairs_exact.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$CKPT_ROOT/traj_alfworld_dpo_${RUN_TAG}}"

torchrun --nproc_per_node="$NPROC" --standalone unlearn/dpo_unlearn.py \
    --model_path "$MODEL_PATH" \
    --data_parquet "$DATA_PARQUET" \
    --output_dir "$OUTPUT_DIR" \
    --beta "$BETA" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-5}" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --gradient_checkpointing \
    --report_to_wandb \
    --run_name "traj_alfworld_dpo_${RUN_TAG}"

echo "run tag: $RUN_TAG"
echo "output dir: $OUTPUT_DIR/final"
