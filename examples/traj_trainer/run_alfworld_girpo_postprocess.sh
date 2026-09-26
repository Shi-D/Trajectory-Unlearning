#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common/env.sh"

RUN_TAG="${RUN_TAG:-cov16}"
export TRAIN_CKPT_DIR="$CKPT_ROOT/traj_alfworld_girpo_${RUN_TAG}"
export MODEL_DIR="$MODELS_ROOT/Traj-AlfWorld-3B-girpo-${RUN_TAG}"
export EVAL_OUTPUT_DIR="$COLLECTION_DIR/output_girpo_${RUN_TAG}"

exec bash "$SCRIPT_DIR/_common/alfworld_eval.sh"
