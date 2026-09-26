# GiRPO trajectory-unlearning entrypoint for WebShop.
#
# unlearn.unlearn_inject.ForgetOnlyBaselineExcludedAdvantageCollector injects
# the recorded forget trajectory as an extra synthetic sample into every GRPO
# group whose task matches the forget set (matched on the collector's
# match_key, e.g. WebShop's task_key), with reward forced to
# min(real rollout rewards in that group) - delta, and tags it with
# non_tensor_batch["is_forget_injected"]=True.
#
# verl.trainer.ppo.ray_trainer.compute_advantage /
# verl.trainer.ppo.core_algos.compute_grpo_outcome_advantage use that tag to
# EXCLUDE injected rows from each group's own mean/std baseline, while still
# normalizing the injected row's reward against that (real-only) baseline.
# Folding the injected row's artificially low reward into the baseline would
# shift the group mean down and hand every real rollout in the group a
# spurious positive advantage; excluding it keeps the real rows' zero-sum GRPO
# advantages undistorted and gives the injected row a clean negative advantage.
#
# The injected row's advantage is additionally floored at
# algorithm.traj.forget_advantage_min (lambda, default -0.1): when the real
# rollouts in a group are (near-)identical the group std collapses toward the
# epsilon floor and the normalized advantage could otherwise explode. Real
# rows are untouched. See also core_algos.py's kl_penalty fix (clamping the
# low_var_kl exponent) for a complementary grad-norm stability fix.
#
# Config (add via Hydra's `+`/`++` syntax, not in the base ppo_trainer.yaml schema):
#   +unlearn.forget_trajectories_path=/path/to/forget.jsonl
#   +unlearn.extra_penalty=0.0                 (delta)
#   algorithm.traj.forget_advantage_min=-0.1   (lambda; in ppo_trainer.yaml)
#
# See examples/traj_trainer/run_webshop_girpo.sh for a full launch example.

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo_girpo_webshop(config)


def run_ppo_girpo_webshop(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        forget_trajectories_path = OmegaConf.select(config, "unlearn.forget_trajectories_path")
        if not forget_trajectories_path:
            raise ValueError(
                "GiRPO requires +unlearn.forget_trajectories_path=<jsonl path> pointing at the "
                "collection/collect_trajectories.py --all-tasks output whose trajectories should "
                "be unlearned."
            )

        from unlearn.unlearn_inject import DEFAULT_BASELINE_ALL_TASKS_PATH, load_forget_full_trajectories

        # WebShop's rendered "initial_observation" is a near-constant string
        # (WebshopEnvironmentManager strips the instruction out of the
        # anchor observation for GiGPO's task-agnostic step grouping -- see
        # ForgetTrajectoryInjectingCollector's match_key docstring), so it
        # cannot serve as a per-task join key there. Use "task_desc" (the
        # raw instruction text) instead for WebShop; ALFWorld keeps the
        # default "initial_observation" key unchanged.
        env_name = OmegaConf.select(config, "env.env_name")
        is_webshop = str(env_name).lower() == "webshop"
        forget_trajectories = load_forget_full_trajectories(
            forget_trajectories_path,
            exclude_ambiguous_from=None if is_webshop else DEFAULT_BASELINE_ALL_TASKS_PATH,
            key_field="task_desc" if is_webshop else "initial_observation",
        )

        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        from agent_system.environments import make_envs
        envs, val_envs = make_envs(config)

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # Plain stock reward manager: episode_rewards for both real rows
        # (naturally computed by rollout) and injected synthetic rows
        # (forced by ForgetTrajectoryInjectingCollector, see below) are
        # already correct by the time this ever runs.
        from agent_system.reward_manager import EpisodeRewardManager

        reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        assert config.actor_rollout_ref.rollout.n == 1, "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"

        from agent_system.multi_turn_rollout import TrajectoryCollector
        from unlearn.unlearn_inject import ForgetOnlyBaselineExcludedAdvantageCollector

        extra_penalty = float(OmegaConf.select(config, "unlearn.extra_penalty") or 0.0)

        base_traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
        traj_collector = ForgetOnlyBaselineExcludedAdvantageCollector(
            base_collector=base_traj_collector,
            tokenizer=tokenizer,
            forget_trajectories=forget_trajectories,
            config=config,
            extra_penalty=extra_penalty,
            match_key="task_key" if is_webshop else "anchor_obs",
        )

        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
