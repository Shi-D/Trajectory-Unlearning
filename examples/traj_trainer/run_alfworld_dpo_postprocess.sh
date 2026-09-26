#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common/env.sh"

RUN_TAG="${RUN_TAG:-beta0p1}"

export MODEL_DIR="$CKPT_ROOT/traj_alfworld_dpo_${RUN_TAG}/final"
export EVAL_OUTPUT_DIR="$COLLECTION_DIR/output_dpo_${RUN_TAG}"

exec bash "$SCRIPT_DIR/_common/alfworld_eval.sh"
