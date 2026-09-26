#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common/env.sh"

NPO_COEF="${NPO_COEF:-0.05}"
RUN_TAG="npo${NPO_COEF//./p}"

export MAIN_MODULE=verl.trainer.main_ppo_npo_grpo
export MODEL_PATH="${MODEL_PATH:-$ALFWORLD_BASE_MODEL}"
export UNLEARN_TRAJECTORIES_PATH="${UNLEARN_TRAJECTORIES_PATH:-$COLLECTION_DIR/output/unlearn_mixed100_forget.jsonl}"
export TRAJ_UNLEARN_RETAIN_TRAJECTORIES_PATH="${TRAJ_UNLEARN_RETAIN_TRAJECTORIES_PATH:-$COLLECTION_DIR/output/retain_no_mixed100.jsonl}"
for f in "$MODEL_PATH/config.json" "$UNLEARN_TRAJECTORIES_PATH" "$TRAJ_UNLEARN_RETAIN_TRAJECTORIES_PATH"; do
    [[ -f "$f" ]] || { echo "Not found: $f" >&2; exit 1; }
done

export TRAJ_STEP_ADV_W=0.0
export TRAJ_EPISODE_SKILL_TEACHER_ADV_W=0.0
export TRAJ_STEP_SKILL_TEACHER_ADV_W=0.0
export TRAJ_OPD_LOSS_COEF=0.0
export TRAJ_UNLEARN_ACTION_LOSS_COEF=0.0
export TRAJ_UNLEARN_OFFPOLICY_LOSS_COEF=0.0

export TRAJ_UNLEARN_NPO_LOSS_COEF="$NPO_COEF"
export TRAJ_UNLEARN_NPO_BETA="${TRAJ_UNLEARN_NPO_BETA:-0.1}"
export TRAJ_UNLEARN_NPO_GAMMA="${TRAJ_UNLEARN_NPO_GAMMA:-1.0}"
export TRAJ_UNLEARN_NPO_MICRO_BSZ="${TRAJ_UNLEARN_NPO_MICRO_BSZ:-8}"
export TRAJ_UNLEARN_OFFPOLICY_BATCH_SIZE="${TRAJ_UNLEARN_OFFPOLICY_BATCH_SIZE:-128}"
export TRAJ_UNLEARN_RETAIN_BATCH_SIZE="${TRAJ_UNLEARN_RETAIN_BATCH_SIZE:-128}"

export TRAJ_ENABLE_ANALYSIS=True
export TRAJ_SKILL_MODE="${TRAJ_SKILL_MODE:-episode_only}"
export TRAJ_ANALYSIS_BACKEND="${TRAJ_ANALYSIS_BACKEND:-openai}"
export TRAJ_MODE="${TRAJ_MODE:-mean_std_norm}"

export TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-24}"
export VAL_DATA_SIZE="${VAL_DATA_SIZE:-24}"
export GROUP_SIZE="${GROUP_SIZE:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-96}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-41}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
HISTORY_LENGTH="${HISTORY_LENGTH:-5}"
export history_length="$HISTORY_LENGTH"

export RAY_health_check_period_ms=15000
export RAY_health_check_timeout_ms=30000
export RAY_health_check_failure_threshold=20

export PROJECT_NAME="${PROJECT_NAME:-agentic_alfworld_unlearn}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-traj_alfworld_npo_grpo_${RUN_TAG}}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$CKPT_ROOT/$EXPERIMENT_NAME}"

echo "ALFWorld NPO+GRPO"
echo "  model:               $MODEL_PATH"
echo "  forget trajectories: $UNLEARN_TRAJECTORIES_PATH"
echo "  retain trajectories: $TRAJ_UNLEARN_RETAIN_TRAJECTORIES_PATH"
echo "  npo_loss_coef:       $TRAJ_UNLEARN_NPO_LOSS_COEF"
echo "  output dir:          $DEFAULT_LOCAL_DIR"

exec bash "$SCRIPT_DIR/_common/alfworld.sh" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.save_freq=5 \
    trainer.test_freq="$TOTAL_EPOCHS" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.max_actor_ckpt_to_keep=2 \
    env.history_length="$HISTORY_LENGTH" \
    "$@"
