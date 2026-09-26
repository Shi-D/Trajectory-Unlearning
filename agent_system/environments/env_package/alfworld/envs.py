# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

def _chunk_list(items, num_chunks):
    """Split items into num_chunks contiguous, disjoint, roughly-equal chunks.

    Used to guarantee full coverage of a "forget" game pool: TextWorld's
    TextworldBatchGymEnv.seed(seed) (called by AlfworldWorker with
    seed=base_seed + task_idx, identical across one task-slot's group_n
    replicas) builds a *deterministic* shuffled_cycle over whatever
    game_files list it was given -- every game in that list is visited
    exactly once before any repeat. So if each dedicated forget task-slot is
    given its own disjoint chunk (rather than the full pool), the union of
    all dedicated slots' chunks -- i.e. the whole forget pool -- is
    guaranteed fully visited within ceil(len(items) / num_chunks) resets,
    with zero cross-slot overlap. (Assigning the full pool to every slot
    instead would only give each slot a probabilistic chance of touching any
    given game within a fixed step budget, not a guarantee.)
    """
    if num_chunks <= 0:
        return []
    n = len(items)
    base, rem = divmod(n, num_chunks)
    chunks = []
    start = 0
    for i in range(num_chunks):
        size = base + (1 if i < rem else 0)
        chunks.append(items[start:start + size])
        start += size
    return chunks

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env):
        self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        self.env.seed(seed)
    
    def step(self, action):
        """Execute a step in the environment"""
        actions = [action] 
        
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        return obs, scores, dones, infos
    
    def reset(self):
        """Reset the environment"""
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        return obs, infos
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)

        # Traj-unlearn: optionally EXCLUDE an explicit list of games from
        # the main (non-forget-dedicated) pool entirely -- e.g. to train a
        # "retain-only" control that never samples a designated forget set
        # at all, unlike forget_env_num (which dedicates SOME slots to a
        # forget pool while the rest keep the normal, unrestricted pool).
        # Applies to both train and eval envs (env_kwargs is shared). Same
        # plain-text game_file-list format as forget_game_files_path below.
        exclude_game_files_path = env_kwargs.get('exclude_game_files_path')
        if exclude_game_files_path:
            with open(exclude_game_files_path) as f:
                excluded_game_files = {line.strip() for line in f if line.strip()}
            before_n = len(base_env.game_files)
            base_env.game_files = [g for g in base_env.game_files if g not in excluded_game_files]
            base_env.num_games = len(base_env.game_files)
            print(
                f"[AlfworldEnvs] exclude_game_files_path={exclude_game_files_path}: "
                f"pool {before_n} -> {len(base_env.game_files)} games after excluding "
                f"{len(excluded_game_files)} requested games."
            )

        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n

        # Traj-unlearn: optionally dedicate the first `forget_env_num`
        # task-slots to a guaranteed-coverage cycle over a separate
        # "forget" game pool (see _chunk_list above), while the remaining
        # task-slots keep using the normal (usually broader/unrestricted)
        # pool for ordinary reward-driven "retain" training. No-op unless
        # both forget_alf_config_path and forget_env_num are set.
        forget_alf_config_path = env_kwargs.get('forget_alf_config_path')
        forget_env_num = int(env_kwargs.get('forget_env_num', 0) or 0)
        forget_env_num = min(max(forget_env_num, 0), env_num) if is_train else 0

        forget_chunks = None
        forget_base_env = None
        forget_config = None
        if forget_alf_config_path and forget_env_num > 0:
            forget_config = load_config_file(forget_alf_config_path)
            forget_env_type = forget_config['env']['type']
            forget_base_env = get_environment(forget_env_type)(forget_config, train_eval='train')

            # Traj-unlearn: optional further filter down to an EXACT,
            # arbitrary subset of games (e.g. a fixed mixed-task-type
            # sample) instead of "every game of forget_config_path's
            # task_types". forget_game_files_path is a plain text file, one
            # absolute ALFWorld game_file path per line (same string format
            # as AlfredTWEnv.collect_game_files' game_file_path, i.e.
            # os.path.join(root, "game.tw-pddl")), so exact string
            # membership is sufficient -- no normalization needed.
            forget_game_files_path = env_kwargs.get('forget_game_files_path')
            if forget_game_files_path:
                with open(forget_game_files_path) as f:
                    allowed_game_files = {line.strip() for line in f if line.strip()}
                before_n = len(forget_base_env.game_files)
                forget_base_env.game_files = [g for g in forget_base_env.game_files if g in allowed_game_files]
                forget_base_env.num_games = len(forget_base_env.game_files)
                found = set(forget_base_env.game_files)
                missing = allowed_game_files - found
                print(
                    f"[AlfworldEnvs] forget_game_files_path={forget_game_files_path}: "
                    f"directory-scanned pool {before_n} -> {len(forget_base_env.game_files)} games after "
                    f"filtering to the requested {len(allowed_game_files)}-game list "
                    f"({len(missing)} requested games not found by the directory scan)."
                )
                assert not missing, (
                    f"forget_game_files_path requested {len(missing)} game(s) not found in "
                    f"forget_config_path's directory scan (check task_types covers all requested games' "
                    f"task types): {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
                )

            forget_chunks = _chunk_list(list(forget_base_env.game_files), forget_env_num)
            print(
                f"[AlfworldEnvs] Dedicating {forget_env_num}/{env_num} task-slots to guaranteed-coverage "
                f"cycling over {len(forget_base_env.game_files)} forget-task games "
                f"({[len(c) for c in forget_chunks]} games/slot; full coverage guaranteed within "
                f"{max(len(c) for c in forget_chunks)} resets)."
            )

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            task_idx = i // self.group_n
            if forget_chunks is not None and task_idx < forget_env_num:
                # Mutate-then-remote(): Ray serializes the object's state at
                # the moment .remote() is called, so each dedicated worker
                # snapshots its own disjoint chunk even though forget_base_env
                # is reused (not re-scanning the game directory per worker).
                forget_base_env.game_files = forget_chunks[task_idx]
                forget_base_env.num_games = len(forget_chunks[task_idx])
                worker = env_worker.remote(forget_config, seed + task_idx, forget_base_env)
            else:
                worker = env_worker.remote(config, seed + task_idx, base_env)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        images = ray.get(futures)
        return images

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)