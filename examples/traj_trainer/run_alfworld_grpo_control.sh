#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common/env.sh"

export MODEL_PATH="${MODEL_PATH:-$ALFWORLD_BASE_MODEL}"
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "Model not found: $MODEL_PATH" >&2
    exit 1
fi

EXCLUDE_GAME_FILES_PATH="${EXCLUDE_GAME_FILES_PATH:-$COLLECTION_DIR/output/mixed100_forget_game_files.txt}"

export TRAJ_STEP_ADV_W=0.0
export TRAJ_EPISODE_SKILL_TEACHER_ADV_W=0.0
export TRAJ_STEP_SKILL_TEACHER_ADV_W=0.0
export TRAJ_OPD_LOSS_COEF=0.0
export TRAJ_UNLEARN_ACTION_LOSS_COEF=0.0
export TRAJ_ENABLE_ANALYSIS=False
export TRAJ_SKILL_MODE="${TRAJ_SKILL_MODE:-episode_only}"
export TRAJ_MODE="${TRAJ_MODE:-mean_std_norm}"

HISTORY_LENGTH="${HISTORY_LENGTH:-5}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-41}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-5}"

export PROJECT_NAME="${PROJECT_NAME:-agentic_alfworld_unlearn}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-traj_alfworld_grpo_control}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$CKPT_ROOT/$EXPERIMENT_NAME}"
export history_length="$HISTORY_LENGTH"

echo "ALFWorld GRPO-control (retain pool only)"
echo "  model:       $MODEL_PATH"
echo "  excluded:    $EXCLUDE_GAME_FILES_PATH"
echo "  output dir:  $DEFAULT_LOCAL_DIR"

exec bash "$SCRIPT_DIR/_common/alfworld.sh" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    env.history_length="$HISTORY_LENGTH" \
    env.alfworld.exclude_game_files_path="$EXCLUDE_GAME_FILES_PATH" \
    "$@"
