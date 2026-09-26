#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common/env.sh"

DELTA="${DELTA:-0.0}"
RUN_TAG="delta${DELTA//./p}"
GUARANTEED_FORGET_SLOTS="${GUARANTEED_FORGET_SLOTS:-0}"
if [[ "$GUARANTEED_FORGET_SLOTS" -gt 0 ]]; then
    RUN_TAG="${RUN_TAG}_guaranteed${GUARANTEED_FORGET_SLOTS}"
fi

export MAIN_MODULE=verl.trainer.main_ppo_girpo_webshop
export MODEL_PATH="${MODEL_PATH:-$WEBSHOP_BASE_MODEL}"
export UNLEARN_TRAJECTORIES_PATH="${UNLEARN_TRAJECTORIES_PATH:-$COLLECTION_WEBSHOP_DIR/output/webshop_unlearn_forget100.jsonl}"
FORGET100_TASK_KEYS_PATH="${FORGET100_TASK_KEYS_PATH:-$COLLECTION_WEBSHOP_DIR/output/webshop_forget100_task_keys.json}"
export TRAJ_GIRPO_DELTA="$DELTA"

export TRAJ_STEP_ADV_W=0.0
export TRAJ_EPISODE_SKILL_TEACHER_ADV_W=0.0
export TRAJ_STEP_SKILL_TEACHER_ADV_W=0.0
export TRAJ_OPD_LOSS_COEF=0.0
export TRAJ_ENABLE_ANALYSIS=False

export TRAIN_DATA_SIZE=24
export VAL_DATA_SIZE=24
export GROUP_SIZE=4
export NUM_CPUS_PER_ENV_WORKER=0.1
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-41}"
export history_length=2

export PROJECT_NAME="${PROJECT_NAME:-agentic_webshop_unlearn}"
export EXPERIMENT_NAME="traj_webshop_girpo_${RUN_TAG}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$CKPT_ROOT/$EXPERIMENT_NAME}"

echo "  forget trajectories:     $UNLEARN_TRAJECTORIES_PATH"
echo "  delta (extra_penalty):   $DELTA"
echo "  guaranteed_forget_slots: $GUARANTEED_FORGET_SLOTS (0 = approximate coverage)"

exec bash "$SCRIPT_DIR/_common/webshop.sh" \
    trainer.n_gpus_per_node=2 \
    trainer.save_freq=5 \
    trainer.test_freq="$TOTAL_EPOCHS" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.max_actor_ckpt_to_keep=2 \
    env.webshop.guaranteed_forget_task_keys_path="$FORGET100_TASK_KEYS_PATH" \
    env.webshop.guaranteed_forget_slots="$GUARANTEED_FORGET_SLOTS" \
    "$@"
