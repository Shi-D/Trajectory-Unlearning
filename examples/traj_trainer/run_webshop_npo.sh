#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common/env.sh"

BETA="${BETA:-0.1}"
GAMMA="${GAMMA:-1.0}"
NPROC="${NPROC:-4}"
RUN_TAG="beta${BETA//./p}_gamma${GAMMA//./p}"

MODEL_PATH="${MODEL_PATH:-$WEBSHOP_BASE_MODEL}"
DATA_PARQUET="${DATA_PARQUET:-$COLLECTION_WEBSHOP_DIR/output/npo_forget_webshop_1000_relabeled.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$CKPT_ROOT/traj_webshop_npo_${RUN_TAG}}"

PYTHONNOUSERSITE=1 torchrun --nproc_per_node="$NPROC" --standalone unlearn/npo_unlearn.py \
    --model_path "$MODEL_PATH" \
    --data_parquet "$DATA_PARQUET" \
    --forget_task_types mixed_forget \
    --retain_task_types auto \
    --output_dir "$OUTPUT_DIR" \
    --beta "$BETA" \
    --gamma "$GAMMA" \
    --max_length 3072 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --gradient_checkpointing \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap Qwen2DecoderLayer \
    --report_to_wandb \
    --run_name "traj_webshop_npo_${RUN_TAG}"

echo "run tag: $RUN_TAG"
echo "output dir: $OUTPUT_DIR/final"
