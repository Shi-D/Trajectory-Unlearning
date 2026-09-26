#!/usr/bin/env bash

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
    set +x
    set -a
    source "$ENV_FILE"
    set +a
    set -x
fi

MAIN_MODULE="${MAIN_MODULE:-verl.trainer.main_ppo}"
UNLEARN_TRAJECTORIES_PATH="${UNLEARN_TRAJECTORIES_PATH:-}"
TRAJ_UNLEARN_OFFPOLICY_BATCH_SIZE="${TRAJ_UNLEARN_OFFPOLICY_BATCH_SIZE:-128}"
TRAJ_UNLEARN_NPO_LOSS_COEF="${TRAJ_UNLEARN_NPO_LOSS_COEF:-0.0}"
TRAJ_UNLEARN_NPO_BETA="${TRAJ_UNLEARN_NPO_BETA:-0.1}"
TRAJ_UNLEARN_NPO_GAMMA="${TRAJ_UNLEARN_NPO_GAMMA:-1.0}"
TRAJ_UNLEARN_NPO_MICRO_BSZ="${TRAJ_UNLEARN_NPO_MICRO_BSZ:-8}"
TRAJ_UNLEARN_RETAIN_TRAJECTORIES_PATH="${TRAJ_UNLEARN_RETAIN_TRAJECTORIES_PATH:-}"
TRAJ_UNLEARN_RETAIN_BATCH_SIZE="${TRAJ_UNLEARN_RETAIN_BATCH_SIZE:-128}"

ENGINE="${ENGINE:-vllm}"

ulimit -u 65536
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

export PYTHONNOUSERSITE=1

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MODELS_ROOT=${MODELS_ROOT:-}
if [[ -z "${MODEL_PATH:-}" ]]; then
    : "${MODELS_ROOT:?Please set MODEL_PATH through a public launcher, or set MODELS_ROOT}"
    MODEL_PATH="$MODELS_ROOT/Qwen2.5-3B-Instruct"
fi
TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-16}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-128}"
GROUP_SIZE="${GROUP_SIZE:-8}"
NUM_CPUS_PER_ENV_WORKER="${NUM_CPUS_PER_ENV_WORKER:-0.1}"

TRAJ_MODE="${TRAJ_MODE:-mean_norm}"
TRAJ_STEP_ADV_W="${TRAJ_STEP_ADV_W:-0.0}"
TRAJ_EPISODE_SKILL_TEACHER_ADV_W="${TRAJ_EPISODE_SKILL_TEACHER_ADV_W:-0.0}"
TRAJ_STEP_SKILL_TEACHER_ADV_W="${TRAJ_STEP_SKILL_TEACHER_ADV_W:-0.0}"
TRAJ_SKILL_MODE="${TRAJ_SKILL_MODE:-episode_step}"
TRAJ_SKILL_TEACHER_MODE="${TRAJ_SKILL_TEACHER_MODE:-step_priority}"
TRAJ_OPD_START_AFTER_STEPS="${TRAJ_OPD_START_AFTER_STEPS:-null}"
TRAJ_OPD_STOP_AFTER_STEPS="${TRAJ_OPD_STOP_AFTER_STEPS:-null}"
TRAJ_OPD_LOSS_COEF="${TRAJ_OPD_LOSS_COEF:-0.01}"
TRAJ_OPD_GATE_BETA="${TRAJ_OPD_GATE_BETA:-5.0}"

TRAJ_FAILED_ONLY="${TRAJ_FAILED_ONLY:-False}"
TRAJ_FAILED_ONLY_AFTER_STEPS="${TRAJ_FAILED_ONLY_AFTER_STEPS:-null}"
TRAJ_FAILURE_SUCCESS_THRESHOLD="${TRAJ_FAILURE_SUCCESS_THRESHOLD:-1.0}"

TRAJ_ENABLE_ANALYSIS="${TRAJ_ENABLE_ANALYSIS:-True}"
TRAJ_SELECTOR="${TRAJ_SELECTOR:-llm}"
TRAJ_ANALYSIS_BACKEND="${TRAJ_ANALYSIS_BACKEND:-policy_vllm}"
TRAJ_ANALYSIS_NUM_WORKERS="${TRAJ_ANALYSIS_NUM_WORKERS:-1}"
TRAJ_ANALYSIS_CONTEXT_LENGTH="${TRAJ_ANALYSIS_CONTEXT_LENGTH:-16384}"
TRAJ_ANALYSIS_MAX_COMPLETION_TOKENS="${TRAJ_ANALYSIS_MAX_COMPLETION_TOKENS:-4096}"
TRAJ_ANALYSIS_MAX_MODEL_LEN="${TRAJ_ANALYSIS_MAX_MODEL_LEN:-20480}"
TRAJ_ANALYSIS_MAX_STEP_SKILLS_PER_TRAJ="${TRAJ_ANALYSIS_MAX_STEP_SKILLS_PER_TRAJ:-5}"
TRAJ_ANALYSIS_PROMPT_VERSION="${TRAJ_ANALYSIS_PROMPT_VERSION:-traj}"

PROJECT_NAME="${PROJECT_NAME:-agentic_webshop}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-traj_qwen2.5_3b_webshop_sft}"
DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$MODELS_ROOT/ckpt/$EXPERIMENT_NAME}"

history_length="${history_length:-2}"

cd "$PROJECT_ROOT"

python3 -m examples.data_preprocess.prepare \
    --mode text \
    --train_data_size "$TRAIN_DATA_SIZE" \
    --val_data_size "$VAL_DATA_SIZE"

EXTRA_UNLEARN_ARGS=()
if [[ "$MAIN_MODULE" == "verl.trainer.main_ppo_npo_grpo" || "$MAIN_MODULE" == "verl.trainer.main_ppo_girpo_webshop" ]]; then
    : "${UNLEARN_TRAJECTORIES_PATH:?$MAIN_MODULE requires UNLEARN_TRAJECTORIES_PATH}"
    EXTRA_UNLEARN_ARGS+=("+unlearn.forget_trajectories_path=$UNLEARN_TRAJECTORIES_PATH")
fi
if [[ "$MAIN_MODULE" == "verl.trainer.main_ppo_girpo_webshop" ]]; then
    EXTRA_UNLEARN_ARGS+=("+unlearn.extra_penalty=${TRAJ_GIRPO_DELTA:-0.0}")
fi
if [[ "$MAIN_MODULE" == "verl.trainer.main_ppo_npo_grpo" ]]; then
    EXTRA_UNLEARN_ARGS+=("+unlearn.offpolicy_batch_size=$TRAJ_UNLEARN_OFFPOLICY_BATCH_SIZE")
    : "${TRAJ_UNLEARN_RETAIN_TRAJECTORIES_PATH:?NPO+GRPO requires TRAJ_UNLEARN_RETAIN_TRAJECTORIES_PATH}"
    EXTRA_UNLEARN_ARGS+=("+unlearn.retain_trajectories_path=$TRAJ_UNLEARN_RETAIN_TRAJECTORIES_PATH")
    EXTRA_UNLEARN_ARGS+=("+unlearn.retain_batch_size=$TRAJ_UNLEARN_RETAIN_BATCH_SIZE")
fi

python3 -m "$MAIN_MODULE" \
    algorithm.adv_estimator=traj \
    data.train_files="$HOME/data/verl-agent/text/train.parquet" \
    data.val_files="$HOME/data/verl-agent/text/test.parquet" \
    data.train_batch_size="$TRAIN_DATA_SIZE" \
    data.val_batch_size="$VAL_DATA_SIZE" \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.opd_loss_coef="$TRAJ_OPD_LOSS_COEF" \
    actor_rollout_ref.actor.opd_gate_beta="$TRAJ_OPD_GATE_BETA" \
    actor_rollout_ref.actor.unlearn_npo_loss_coef="$TRAJ_UNLEARN_NPO_LOSS_COEF" \
    actor_rollout_ref.actor.unlearn_npo_beta="$TRAJ_UNLEARN_NPO_BETA" \
    actor_rollout_ref.actor.unlearn_npo_gamma="$TRAJ_UNLEARN_NPO_GAMMA" \
    actor_rollout_ref.actor.unlearn_npo_micro_batch_size_per_gpu="$TRAJ_UNLEARN_NPO_MICRO_BSZ" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_model_len="$TRAJ_ANALYSIS_MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$TRAJ_ANALYSIS_MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.traj.step_advantage_w="$TRAJ_STEP_ADV_W" \
    algorithm.traj.episode_skill_teacher_advantage_w="$TRAJ_EPISODE_SKILL_TEACHER_ADV_W" \
    algorithm.traj.step_skill_teacher_advantage_w="$TRAJ_STEP_SKILL_TEACHER_ADV_W" \
    algorithm.traj.skill_mode="$TRAJ_SKILL_MODE" \
    algorithm.traj.skill_teacher_mode="$TRAJ_SKILL_TEACHER_MODE" \
    algorithm.traj.opd_start_after_steps="$TRAJ_OPD_START_AFTER_STEPS" \
    algorithm.traj.opd_stop_after_steps="$TRAJ_OPD_STOP_AFTER_STEPS" \
    algorithm.traj.failed_only="$TRAJ_FAILED_ONLY" \
    algorithm.traj.failed_only_after_steps="$TRAJ_FAILED_ONLY_AFTER_STEPS" \
    algorithm.traj.failure_success_threshold="$TRAJ_FAILURE_SUCCESS_THRESHOLD" \
    algorithm.traj.mode="$TRAJ_MODE" \
    algorithm.traj.enable_analysis="$TRAJ_ENABLE_ANALYSIS" \
    algorithm.traj.selector="$TRAJ_SELECTOR" \
    algorithm.traj.analysis_backend="$TRAJ_ANALYSIS_BACKEND" \
    algorithm.traj.analysis_num_workers="$TRAJ_ANALYSIS_NUM_WORKERS" \
    algorithm.traj.analysis_context_length="$TRAJ_ANALYSIS_CONTEXT_LENGTH" \
    algorithm.traj.analysis_max_completion_tokens="$TRAJ_ANALYSIS_MAX_COMPLETION_TOKENS" \
    algorithm.traj.analysis_max_step_skills_per_traj="$TRAJ_ANALYSIS_MAX_STEP_SKILLS_PER_TRAJ" \
    algorithm.traj.analysis_prompt_version="$TRAJ_ANALYSIS_PROMPT_VERSION" \
    algorithm.traj.normalize_teacher_adv=False \
    env.history_length="$history_length" \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n="$GROUP_SIZE" \
    env.resources_per_worker.num_cpus="$NUM_CPUS_PER_ENV_WORKER" \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=160 \
    trainer.val_before_train=False \
    trainer.default_local_dir="$DEFAULT_LOCAL_DIR" \
    trainer.rollout_data_dir="$DEFAULT_LOCAL_DIR" \
    "${EXTRA_UNLEARN_ARGS[@]}" \
    "$@"
