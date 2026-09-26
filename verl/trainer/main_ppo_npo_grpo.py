# NPO+GRPO trajectory-unlearning entrypoint
# (verl/trainer/ppo/unlearn_npo_ray_trainer.py::TrajUnlearnNPORayPPOTrainer):
# plain on-policy GRPO plus an off-policy NPO auxiliary loss on the recorded
# forget / retain step pairs.
#
# Config keys (see examples/traj_trainer/run_alfworld_npo_grpo.sh):
#   +unlearn.forget_trajectories_path=<jsonl>
#   +unlearn.retain_trajectories_path=<jsonl>
#   +unlearn.offpolicy_batch_size=128           (forget batch size)
#   +unlearn.retain_batch_size=128
#   actor_rollout_ref.actor.unlearn_npo_loss_coef=...   (0 = off)
#   actor_rollout_ref.actor.unlearn_npo_beta=0.1
#   actor_rollout_ref.actor.unlearn_npo_gamma=1.0
#   actor_rollout_ref.actor.use_kl_loss=True    (required -- instantiates the
#       frozen reference-policy worker group reused for the NPO forget loss's
#       pi_ref(y|x) term; set by examples/traj_trainer/_common/*.sh)

import os

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo_npo_grpo(config)


def run_ppo_npo_grpo(config) -> None:
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
                "TrajUnlearnNPORayPPOTrainer requires +unlearn.forget_trajectories_path=<jsonl path>."
            )
        retain_trajectories_path = OmegaConf.select(config, "unlearn.retain_trajectories_path")
        if not retain_trajectories_path:
            raise ValueError(
                "TrajUnlearnNPORayPPOTrainer requires +unlearn.retain_trajectories_path=<jsonl path>."
            )
        off_policy_batch_size = OmegaConf.select(config, "unlearn.offpolicy_batch_size") or 128
        retain_batch_size = OmegaConf.select(config, "unlearn.retain_batch_size") or 128

        if not config.actor_rollout_ref.actor.use_kl_loss:
            raise ValueError(
                "NPO+GRPO (TrajUnlearnNPORayPPOTrainer) requires actor_rollout_ref.actor.use_kl_loss=True "
                "-- this is what instantiates the frozen reference-policy worker group the NPO forget "
                "loss reuses for its pi_ref(y|x) term."
            )

        from unlearn.unlearn_offpolicy import load_step_pairs

        forget_step_pairs = load_step_pairs(forget_trajectories_path, tag="NPO+GRPO forget")
        retain_step_pairs = load_step_pairs(retain_trajectories_path, tag="NPO+GRPO retain")

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

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

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

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == "episode":
            from agent_system.reward_manager import EpisodeRewardManager
            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError

        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        assert config.actor_rollout_ref.rollout.n == 1, "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"

        from agent_system.multi_turn_rollout import TrajectoryCollector
        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        from verl.trainer.ppo.unlearn_npo_ray_trainer import TrajUnlearnNPORayPPOTrainer

        trainer = TrajUnlearnNPORayPPOTrainer(
            forget_step_pairs=forget_step_pairs,
            retain_step_pairs=retain_step_pairs,
            off_policy_batch_size=int(off_policy_batch_size),
            retain_batch_size=int(retain_batch_size),
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
