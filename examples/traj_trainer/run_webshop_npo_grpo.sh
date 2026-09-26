#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common/env.sh"

NPO_COEF="${NPO_COEF:-0.05}"
RUN_TAG="npo${NPO_COEF//./p}"

export MAIN_MODULE=verl.trainer.main_ppo_npo_grpo
export MODEL_PATH="${MODEL_PATH:-$WEBSHOP_BASE_MODEL}"
export UNLEARN_TRAJECTORIES_PATH="${UNLEARN_TRAJECTORIES_PATH:-$COLLECTION_WEBSHOP_DIR/output/webshop_unlearn_forget100.jsonl}"
export TRAJ_UNLEARN_RETAIN_TRAJECTORIES_PATH="${TRAJ_UNLEARN_RETAIN_TRAJECTORIES_PATH:-$COLLECTION_WEBSHOP_DIR/output/webshop_retain900.jsonl}"

export TRAJ_UNLEARN_NPO_LOSS_COEF="$NPO_COEF"
export TRAJ_UNLEARN_NPO_BETA=0.1
export TRAJ_UNLEARN_NPO_GAMMA=1.0
export TRAJ_UNLEARN_NPO_MICRO_BSZ=8
export TRAJ_UNLEARN_OFFPOLICY_BATCH_SIZE=128
export TRAJ_UNLEARN_RETAIN_BATCH_SIZE=128

export TRAJ_STEP_ADV_W=0.0
export TRAJ_EPISODE_SKILL_TEACHER_ADV_W=0.0
export TRAJ_STEP_SKILL_TEACHER_ADV_W=0.0
export TRAJ_OPD_LOSS_COEF=0.0
export TRAJ_ENABLE_ANALYSIS=True
export TRAJ_SKILL_MODE=episode_only
export TRAJ_ANALYSIS_BACKEND="${TRAJ_ANALYSIS_BACKEND:-policy_vllm}"
export TRAJ_SELECTOR="${TRAJ_SELECTOR:-llm}"

export TRAIN_DATA_SIZE=24
export VAL_DATA_SIZE=24
export GROUP_SIZE=4
export NUM_CPUS_PER_ENV_WORKER=0.1
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-41}"
export history_length=2

export PROJECT_NAME="${PROJECT_NAME:-agentic_webshop_unlearn}"
export EXPERIMENT_NAME="traj_webshop_npo_grpo_${RUN_TAG}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$CKPT_ROOT/$EXPERIMENT_NAME}"

echo "  forget trajectories: $UNLEARN_TRAJECTORIES_PATH"
echo "  retain trajectories: $TRAJ_UNLEARN_RETAIN_TRAJECTORIES_PATH"
echo "  npo_loss_coef:       $TRAJ_UNLEARN_NPO_LOSS_COEF"

exec bash "$SCRIPT_DIR/_common/webshop.sh" \
    trainer.n_gpus_per_node=2 \
    trainer.save_freq=5 \
    trainer.test_freq="$TOTAL_EPOCHS" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.max_actor_ckpt_to_keep=2 \
    "$@"
