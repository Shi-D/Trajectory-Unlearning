#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common/env.sh"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-traj_alfworld_grpo_control}"
export TRAIN_CKPT_DIR="$CKPT_ROOT/$EXPERIMENT_NAME"
export MODEL_DIR="$MODELS_ROOT/Traj-AlfWorld-3B-grpo-control"
export EVAL_OUTPUT_DIR="$COLLECTION_DIR/output_grpo_control"
export MIXED100_BREAKDOWN=1

exec bash "$SCRIPT_DIR/_common/alfworld_eval.sh"
