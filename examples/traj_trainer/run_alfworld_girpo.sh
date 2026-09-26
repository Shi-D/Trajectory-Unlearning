#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common/env.sh"

FORGET_ENV_NUM="${FORGET_ENV_NUM-16}"
RUN_TAG="cov${FORGET_ENV_NUM:-0}"

export MAIN_MODULE=verl.trainer.main_ppo_girpo_alfworld
export MODEL_PATH="${MODEL_PATH:-$ALFWORLD_BASE_MODEL}"
export UNLEARN_TRAJECTORIES_PATH="${UNLEARN_TRAJECTORIES_PATH:-$COLLECTION_DIR/output/unlearn_mixed100_forget.jsonl}"
FORGET_CONFIG_PATH="${FORGET_CONFIG_PATH:-$PROJECT_ROOT/agent_system/environments/env_package/alfworld/configs/config_tw_unlearn_mixed100.yaml}"
FORGET_GAME_FILES_PATH="${FORGET_GAME_FILES_PATH:-$COLLECTION_DIR/output/mixed100_forget_game_files.txt}"
for f in "$MODEL_PATH/config.json" "$UNLEARN_TRAJECTORIES_PATH"; do
    [[ -f "$f" ]] || { echo "Not found: $f" >&2; exit 1; }
done

export TRAJ_STEP_ADV_W=0.0
export TRAJ_EPISODE_SKILL_TEACHER_ADV_W=0.0
export TRAJ_STEP_SKILL_TEACHER_ADV_W=0.0
export TRAJ_OPD_LOSS_COEF=0.0
export TRAJ_UNLEARN_ACTION_LOSS_COEF=0.0
export TRAJ_ENABLE_ANALYSIS=False
export TRAJ_SKILL_MODE="${TRAJ_SKILL_MODE:-episode_only}"
export TRAJ_MODE="${TRAJ_MODE:-mean_std_norm}"

export TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-24}"
export VAL_DATA_SIZE="${VAL_DATA_SIZE:-24}"
export GROUP_SIZE="${GROUP_SIZE:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-96}"
export PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-7}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
HISTORY_LENGTH="${HISTORY_LENGTH:-5}"
export history_length="$HISTORY_LENGTH"

export RAY_health_check_period_ms=15000
export RAY_health_check_timeout_ms=30000
export RAY_health_check_failure_threshold=20

export PROJECT_NAME="${PROJECT_NAME:-agentic_alfworld_unlearn}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-traj_alfworld_girpo_${RUN_TAG}}"
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$CKPT_ROOT/$EXPERIMENT_NAME}"

launch_args=(
    "trainer.n_gpus_per_node=$N_GPUS_PER_NODE"
    "trainer.save_freq=5"
    "trainer.test_freq=$TOTAL_EPOCHS"
    "trainer.total_epochs=$TOTAL_EPOCHS"
    "trainer.max_actor_ckpt_to_keep=2"
    "env.history_length=$HISTORY_LENGTH"
)
if [[ -n "$FORGET_ENV_NUM" ]]; then
    launch_args+=(
        "env.alfworld.forget_config_path=$FORGET_CONFIG_PATH"
        "env.alfworld.forget_env_num=$FORGET_ENV_NUM"
        "env.alfworld.forget_game_files_path=$FORGET_GAME_FILES_PATH"
    )
fi
if [[ -n "${ENV_SEED:-}" ]]; then
    launch_args+=("env.seed=$ENV_SEED")
fi

echo "ALFWorld GiRPO"
echo "  model:               $MODEL_PATH"
echo "  forget trajectories: $UNLEARN_TRAJECTORIES_PATH"
echo "  forget_env_num:      ${FORGET_ENV_NUM:-<none>} / $TRAIN_DATA_SIZE task-slots"
echo "  output dir:          $DEFAULT_LOCAL_DIR"

exec bash "$SCRIPT_DIR/_common/alfworld.sh" "${launch_args[@]}" "$@"
