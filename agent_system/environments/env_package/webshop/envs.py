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

import re
import ray
import gym
import numpy as np


# web_agent_site/engine/goal.py's get_human_goals/get_synthetic_goals append
# this exact suffix to instruction_text, with a price threshold drawn via
# `random.sample(price_range, 2)` BEFORE SimServer.__init__ calls
# `random.seed(seed)` -- i.e. drawn from whatever unseeded global `random`
# state the process happens to be in, so it differs across every separate
# process invocation (the offline extraction run, and each training-time
# worker process). Matching on raw instruction_text therefore silently drops
# ~50-70% of intended matches (any goal whose price_range had >=2
# candidates). Stripping this suffix before comparing restores a stable key.
_PRICE_SUFFIX_RE = re.compile(r", and price lower than [\d.]+ dollars$")


def _normalize_instruction_text(text):
    if text is None:
        return None
    return _PRICE_SUFFIX_RE.sub("", text)


def _load_task_key_set(path):
    """path: JSON file containing a list of [asin, instruction_text] pairs
    (see collection_webshop/build_webshop_task_keys.py). Returns a set of
    (asin, normalized_instruction_text) tuples for exact-membership
    filtering (see _normalize_instruction_text)."""
    import json
    with open(path) as f:
        pairs = json.load(f)
    return {(p[0], _normalize_instruction_text(p[1])) for p in pairs}


def _make_task_key_filter(include_path, exclude_path):
    """Builds a `filter_goals(i, goal) -> bool` callable for
    SimServer/WebAgentTextEnv, keyed on (asin, instruction_text) rather than
    goal index -- each Ray worker's SimServer independently shuffles its own
    goals list at a per-worker seed, so the same numeric index refers to a
    different underlying goal in different workers; only goal *content* is a
    stable identifier across workers. include/exclude are mutually
    exclusive; at most one should be set.

    instruction_text is normalized (price suffix stripped, see
    _normalize_instruction_text) on both sides before comparison, since that
    suffix embeds a randomly-drawn threshold that is not reproducible across
    process invocations.

    Known imprecision: ~9/6910 (asin, normalized instruction) keys collide
    across two distinct goals (same asin + near-identical instruction
    differing only in a field the price threshold used to coincidentally
    disambiguate before normalization) -- on the order of 1-2 goals per
    ~900-1000 task filter. Not worth a more precise (and more fragile) key
    for this project's purposes."""
    if include_path:
        allowed = _load_task_key_set(include_path)
        return lambda i, goal: (goal.get('asin'), _normalize_instruction_text(goal.get('instruction_text'))) in allowed
    if exclude_path:
        excluded = _load_task_key_set(exclude_path)
        return lambda i, goal: (goal.get('asin'), _normalize_instruction_text(goal.get('instruction_text'))) not in excluded
    return None


# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *WebAgentTextEnv* instance.
    """

    def __init__(self, seed, env_kwargs):
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webshop'))
        sys.path.append(project_root)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)

        # Traj-unlearn: optional include/exclude task filtering (see
        # _make_task_key_filter above). Both no-op (None) unless explicitly
        # set via env.webshop.include_task_keys_path / exclude_task_keys_path.
        include_task_keys_path = env_kwargs.pop('include_task_keys_path', None)
        exclude_task_keys_path = env_kwargs.pop('exclude_task_keys_path', None)
        filter_goals = _make_task_key_filter(include_task_keys_path, exclude_task_keys_path)
        if filter_goals is not None:
            env_kwargs['filter_goals'] = filter_goals

        env_kwargs['seed'] = seed
        self.env = gym.make('WebAgentTextEnv-v0', disable_env_checker=True, **env_kwargs)
    
    def step(self, action):
        """Execute a step in the environment"""
        obs, reward, done, info = self.env.step(action)
        info = dict(info or {})  # make a *copy* so we can mutate safely
        info['available_actions'] = self.env.get_available_actions()
        info['task_score'] = reward

        # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
        if done and reward == 1.0:
            info['won'] = True
            reward = 10.0
        else:
            info['won'] = False
            reward = 0

        return obs, reward, done, info
    
    def reset(self, idx):
        """Reset the environment with given session index"""
        obs, info = self.env.reset(session=idx)
        info = dict(info or {})
        info['available_actions'] = self.env.get_available_actions()
        info['won'] = False
        return obs, info
    
    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render)
        return rendered
    
    def get_available_actions(self):
        """Get available actions"""
        return self.env.get_available_actions()
    
    def get_goals(self):
        """Get environment goals"""
        return self.env.server.goals
    
    def close(self):
        """Close the environment"""
        self.env.close()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train: assert group_n == 1

        self._rng = np.random.RandomState(seed)

        # Copy (not alias) so this instance's own pops below never mutate a
        # dict the caller may reuse for a second WebshopMultiProcessEnv
        # (env_manager.py passes the SAME env_kwargs object to both the
        # train and val construction calls).
        self._env_kwargs = dict(env_kwargs) if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}

        # Traj-unlearn (GiRPO guaranteed-coverage): driver-level only --
        # WebshopWorker/gym.make don't understand these, so pop them off
        # before any worker_kwargs are built. See ppo_trainer.yaml's
        # env.webshop docstrings for the full design.
        guaranteed_forget_task_keys_path = self._env_kwargs.pop('guaranteed_forget_task_keys_path', None)
        guaranteed_forget_slots = int(self._env_kwargs.pop('guaranteed_forget_slots', 0) or 0)
        use_guaranteed_coverage = bool(is_train and guaranteed_forget_task_keys_path and guaranteed_forget_slots > 0)
        if use_guaranteed_coverage:
            assert guaranteed_forget_slots <= env_num, (
                f"guaranteed_forget_slots ({guaranteed_forget_slots}) must be <= env_num ({env_num})"
            )

        # -------------------------- Ray actors setup --------------------------
        # Traj-unlearn (GiRPO guaranteed-coverage): the first
        # `guaranteed_forget_slots` GROUPS (not individual workers -- every
        # worker in a group already shares one seed/goal-shuffle, by
        # design, so all `group_n` members get the same cohort) are
        # permanently restricted to guaranteed_forget_task_keys_path's pool
        # via a per-group include_task_keys_path override; every other
        # group keeps this env's ordinary (unrestricted, or
        # include/exclude_task_keys_path-filtered) env_kwargs.
        env_worker = ray.remote(**resources_per_worker)(WebshopWorker)
        self._workers = []
        self._is_forget_cohort_group = [False] * env_num
        for g in range(env_num):
            is_forget_group = use_guaranteed_coverage and g < guaranteed_forget_slots
            self._is_forget_cohort_group[g] = is_forget_group
            if is_forget_group:
                worker_kwargs = dict(self._env_kwargs)
                worker_kwargs['include_task_keys_path'] = guaranteed_forget_task_keys_path
                worker_kwargs['exclude_task_keys_path'] = None
            else:
                worker_kwargs = self._env_kwargs
            for _ in range(group_n):
                worker = env_worker.remote(seed + g, worker_kwargs)
                self._workers.append(worker)

        # ------- original ----------#
        # if args.num is None:
        #     if split == 'test':
        #         self.goal_idxs = range(500)
        #     elif split == 'eval':
        #         self.goal_idxs = range(500, 1500)
        #     elif split == 'train':
        #         self.goal_idxs = range(1500, len(self.env.server.goals))
        # else:
        #     self.goal_idxs = range(len(self.env.server.goals))

        # Traj-unlearn: when env.webshop.include_task_keys_path/
        # exclude_task_keys_path (or a group's guaranteed-coverage cohort
        # override) filtered `goals` down to a custom pool (see
        # _make_task_key_filter), that pool IS the intended train/eval set
        # in its entirety -- the original 500/1500 split-boundary constants
        # below assume the full ~6910-goal unfiltered corpus and are
        # meaningless (and can produce an out-of-range or empty range) once
        # filtering has shrunk `goals` below those constants. Each GROUP
        # can now have its own independently-filtered pool, so goal count
        # and filter-active status are tracked per group rather than once
        # globally (previously always identical across groups, since only
        # one shared env_kwargs existed).
        base_filter_active = bool(
            self._env_kwargs.get('include_task_keys_path') or self._env_kwargs.get('exclude_task_keys_path')
        )
        self._goal_count_per_group = []
        self._filter_active_per_group = []
        goals_per_group = []
        for g in range(env_num):
            goals = ray.get(self._workers[g * group_n].get_goals.remote())
            goals_per_group.append(goals)
            self._goal_count_per_group.append(len(goals))
            self._filter_active_per_group.append(self._is_forget_cohort_group[g] or base_filter_active)

        # Traj-unlearn (GiRPO guaranteed-coverage): partition
        # guaranteed_forget_task_keys_path's task set into
        # guaranteed_forget_slots roughly-equal chunks (shuffled once,
        # deterministically from `seed`), one per forget-cohort group; each
        # such group cycles through its own chunk one task per reset() (see
        # reset() below), so every listed task is visited at least once
        # within ceil(len(task set) / guaranteed_forget_slots) epochs.
        self._forget_chunk_per_group = [None] * env_num
        self._forget_ptr_per_group = [0] * env_num
        self._forget_content_to_idx_per_group = [None] * env_num
        if use_guaranteed_coverage:
            forget_keys = sorted(_load_task_key_set(guaranteed_forget_task_keys_path))
            order = np.random.RandomState(seed).permutation(len(forget_keys))
            shuffled_keys = [forget_keys[i] for i in order]
            chunks = np.array_split(np.arange(len(shuffled_keys)), guaranteed_forget_slots)
            for g in range(env_num):
                if not self._is_forget_cohort_group[g]:
                    continue
                self._forget_chunk_per_group[g] = [shuffled_keys[i] for i in chunks[g]]
                content_to_idx = {}
                for idx, goal in enumerate(goals_per_group[g]):
                    key = (goal.get('asin'), _normalize_instruction_text(goal.get('instruction_text')))
                    content_to_idx.setdefault(key, idx)
                self._forget_content_to_idx_per_group[g] = content_to_idx
            max_chunk = max(len(c) for c in self._forget_chunk_per_group if c is not None)
            print(
                "GiRPO guaranteed-coverage: %s/%s train groups dedicated to %s "
                "(%s tasks); full coverage guaranteed within %s epochs."
                % (guaranteed_forget_slots, env_num, guaranteed_forget_task_keys_path, len(shuffled_keys), max_chunk),
                flush=True,
            )

        print(self._goal_count_per_group)

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self._workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        # Traj-unlearn (GiRPO guaranteed-coverage): each group now draws its
        # own session index independently, since a forget-cohort group's
        # pool size/eligible range can differ from an ordinary group's (see
        # __init__'s per-group _goal_count_per_group/_filter_active_per_group).
        idx_per_group = []
        for g in range(self.env_num):
            if self._is_forget_cohort_group[g]:
                chunk = self._forget_chunk_per_group[g]
                ptr = self._forget_ptr_per_group[g]
                content_key = chunk[ptr % len(chunk)]
                self._forget_ptr_per_group[g] = ptr + 1
                group_idx = self._forget_content_to_idx_per_group[g].get(content_key)
                if group_idx is None:
                    # Rare: this task's normalized content didn't survive
                    # this group's own include_task_keys_path filter (e.g.
                    # an asin/instruction collision edge case) -- fall back
                    # to a random draw from this group's own (still
                    # forget-only) pool rather than crashing.
                    group_idx = int(self._rng.choice(self._goal_count_per_group[g]))
            else:
                n_goals = self._goal_count_per_group[g]
                if self._filter_active_per_group[g]:
                    candidates = range(n_goals)
                elif not self.is_train:
                    candidates = range(500)
                else:
                    candidates = range(500, n_goals)
                group_idx = int(self._rng.choice(candidates))
            idx_per_group.append(group_idx)
        idx = np.repeat(np.array(idx_per_group), self.group_n).tolist()

        # Send reset commands to all workers
        futures = []
        for worker, i in zip(self._workers, idx):
            future = worker.reset.remote(i)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            future = self._workers[env_idx].render.remote(mode)
            return ray.get(future)

        futures = []
        for worker in self._workers:
            future = worker.render.remote(mode)
            futures.append(future)
        
        return ray.get(futures)

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # Close all workers and kill Ray actors
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)
        
        # Wait for all workers to close
        ray.get(close_futures)
        
        # Kill all Ray actors
        for worker in self._workers:
            ray.kill(worker)
            
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )
