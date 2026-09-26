# v12 trajectory-unlearning: forced synthetic-sample injection.
#
# v9/v10/v11 (unlearn/unlearn_reward.py) all suppress the recorded forget
# trajectory only WHEN the on-policy rollout happens to reproduce it (fully
# or, for v11, partially). On an "easy" task type the base model already
# solves well, this is unreliable: many GRPO groups end up homogeneous (all
# samples take the exact recorded path, or none do), leaving nothing to
# contrast against.
#
# v12 removes this dependency on luck entirely: whenever a GRPO group's
# underlying game IS in the forget set (matched via the deterministic
# initial-observation join key, same technique as v1-v4/v9-v11), the
# RECORDED forget trajectory itself is built directly from its own saved
# (prompt, model_response) pairs and injected as an EXTRA synthetic
# "trajectory" into that SAME group -- with its reward forced to
# min(that group's real minimum reward this step, 0), i.e. adaptively at
# least as bad as the worst real outcome sampled for that game, guaranteed
# every single time the game is rolled out (not conditional on what the
# real samples happened to do). Standard GRPO/PPO then handles everything
# else unchanged: log_prob, ref KL, advantage (this synthetic row's reward
# pulls the group mean down, raising every real sample's relative
# advantage), and the policy-gradient update -- no new loss code anywhere.
#
# This module builds the synthetic tensors using the SAME tokenization
# conventions as real rollout rows (agent_system/multi_turn_rollout/
# rollout_loop.py::TrajectoryCollector.build_prompt_batch for the prompt
# side -- the same helper v6's off-policy payload construction already
# relies on -- and verl.utils.torch_functional.get_response_mask for the
# response side, matching vllm_rollout_spmd.py's own masking convention
# exactly), so the injected rows are structurally indistinguishable from a
# real "5th" GRPO sample to every downstream computation.

import json
import re
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from verl import DataProto
from verl.utils.torch_functional import get_response_mask


# WebShop-only: web_agent_site/engine/goal.py's get_human_goals/
# get_synthetic_goals append ", and price lower than X dollars" to
# instruction_text with a threshold drawn from the process's *unseeded*
# global `random` state (drawn before SimServer.__init__ calls
# random.seed(seed)) -- see agent_system/environments/env_package/
# webshop/envs.py's identical _PRICE_SUFFIX_RE. Since this text is rendered
# verbatim into the episode's initial observation, matching raw
# initial_observation strings across two separate process invocations (the
# forget-trajectories capture run vs. a live training rollout) silently
# fails for any task whose price suffix was non-empty -- normalizing it out
# before indexing/lookup restores a stable join key. No-op for ALFWorld
# (whose observations never contain this substring).
_WEBSHOP_PRICE_SUFFIX_RE = re.compile(r", and price lower than [\d.]+ dollars")


def _normalize_obs_key(obs: str) -> str:
    return _WEBSHOP_PRICE_SUFFIX_RE.sub("", obs)

# Tensor/non-tensor keys this injector understands how to build for its own
# synthetic rows. Any OTHER key found on the real rollout batch is either a
# known-and-explicitly-backfilled field (KNOWN_EXTRA_*) or an unexpected one
# that makes _inject refuse to run rather than silently mis-align data.
CORE_TENSOR_KEYS = ["prompts", "responses", "input_ids", "attention_mask", "position_ids"]
CORE_NON_TENSOR_KEYS = [
    "uid",
    "traj_uid",
    "step_num",
    "episode_rewards",
    "episode_lengths",
    "data_source",
    "rewards",
    "active_masks",
    "anchor_obs",
]
# Numeric CORE keys that must be a genuine numpy numeric dtype (not
# `object`) before concatenation -- see the dtype-promotion note in
# _split_known_keys. "rewards" is deliberately excluded: it's consumed only
# via whole-array `.astype()` (core_gigpo.compute_step_discounted_returns),
# which is safe even for an `object` array, and the real pipeline itself
# already produces it via `torch_to_numpy(..., is_object=True)`.
CORE_NUMERIC_NON_TENSOR_DTYPES = {
    "step_num": np.int64,
    "episode_rewards": np.float32,
    "episode_lengths": np.float32,
}
KNOWN_EXTRA_TENSOR_KEYS = {"rollout_log_probs"}
# "index", "obs_text", "obs_text_base", "raw_prompt" are all descriptive/
# provenance fields (dataset row index, templated observation text, raw
# chat-list prompt) never read by the active reward/advantage/loss
# computation for this config (Traj analysis and env_aux_loss are both
# disabled) -- safe to backfill generically (see
# _restore_known_extra_keys's fallback branch).
KNOWN_EXTRA_NON_TENSOR_KEYS = {
    "sample_id",
    "rollout_id",
    "step_id",
    "tool_callings",
    "is_action_valid",
    "index",
    "obs_text",
    "obs_text_base",
    "raw_prompt",
    # Threaded through by agent_system/environments/env_manager.py
    # (AlfWorldEnvironmentManager.reset/step) + rollout_loop.py's
    # preprocess_single_sample: the exact ALFWorld game file path for this
    # row's episode (stable across all steps of one episode, and identical
    # across every member of one GRPO group, same as anchor_obs). Used by
    # v17 (ForgetRetainFixedRewardInjectingCollector) for exact,
    # unambiguous forget/retain matching -- see
    # load_forget_full_trajectories_by_game_file /
    # load_retain_full_trajectories_by_game_file below. Not consumed by any
    # active reward/advantage/loss computation, so safe to backfill
    # generically on injected rows like the other KNOWN_EXTRA keys.
    "game_file",
    # WebShop-only batch-level metric (agent_system/environments/env_manager.py's
    # WebshopEnvironmentManager._process_batch -> success_evaluator()):
    # per-episode task_score (WebShop's continuous [0,1] partial-credit
    # score, distinct from the binary success_rate), identical on every
    # real row for a given rollout batch. Doesn't match the
    # "*_success_rate" pattern _split_known_keys/_restore_known_extra_keys
    # already special-case (its name ends in ")", not "_success_rate")
    # despite being the exact same kind of batch-level aggregate scalar --
    # handled the same way via the generic `fill_value = real_values[0]`
    # fallback below.
    "webshop_task_score (not success_rate)",
    # WebShop-only: per-episode task-identifying key (normalized
    # instruction text), populated by WebshopEnvironmentManager and used as
    # ForgetTrajectoryInjectingCollector's match_key -- see that class's
    # docstring and load_forget_full_trajectories's key_field parameter.
    "task_key",
}


DEFAULT_BASELINE_ALL_TASKS_PATH = str(Path(__file__).resolve().parents[1] / "collection" / "output" / "all_trajectories_0.jsonl")


def load_forget_full_trajectories(
    jsonl_path: str,
    exclude_ambiguous_from: Optional[str] = DEFAULT_BASELINE_ALL_TASKS_PATH,
    task_type_filter: Optional[str] = None,
    key_field: str = "initial_observation",
) -> Dict[str, Dict[str, Any]]:
    """Load a collection/collect_trajectories.py --all-tasks output file,
    indexed by the exact initial (reset) observation text of each episode
    (same deterministic join key as unlearn/unlearn.py::load_forget_index).

    `key_field` (default "initial_observation", ALFWorld's stable join key):
    pass "task_desc" for WebShop-sourced files (see collection_webshop/
    output/webshop_unlearn_forget100.jsonl), whose "initial_observation"
    field is a near-constant string (WebshopEnvironmentManager strips
    instruction text out of the anchor observation -- see
    ForgetTrajectoryInjectingCollector's match_key docstring) and therefore
    useless as a per-task key; "task_desc" carries the actual instruction
    text instead, matched against the caller-populated "task_key"
    non_tensor_batch field via the same price-suffix normalization
    (_normalize_obs_key) applied on both sides.

    Unlike load_forget_index (which only keeps the action list), this keeps
    each step's actual recorded `prompt` (the exact templated text fed to
    the model) and `model_response` (the full raw `<think>...</think>
    <action>...</action>` text), which is what's needed to rebuild proper
    training tensors for injection.

    Returns: {initial_observation: {"task_id", "task_type",
              "steps": [{"prompt", "model_response", "action",
                         "observation"}, ...]}}

    `exclude_ambiguous_from` (default: the project's standard all-6-task-types
    baseline collection): some ALFWorld trial files are internally
    inconsistent -- their folder-name-encoded task_type (e.g.
    "pick_cool_then_place_in_recep-WineBottle-...") does not match the
    actual task description/observation text TextWorld renders at reset
    (e.g. "Your task is to: clean some apple..."). When such a trial's
    observation text happens to collide with a genuine pick_clean entry's
    key, matching on initial_observation ALONE would inject the pick_clean
    forget trajectory into what is nominally a different task type's game
    slot. This cross-references every OTHER task type's initial_observation
    in the given baseline file and drops any forget-index key that also
    appears there, so a match only ever fires for keys that are
    unambiguously pick_clean-only across the full collected dataset. Pass
    None to skip this check (not recommended).

    `task_type_filter`: if set, only lines whose task_type equals this value
    are indexed (every other line is skipped entirely). Lets callers source
    the forget set directly from a mixed-task-type baseline collection (e.g.
    DEFAULT_BASELINE_ALL_TASKS_PATH itself) instead of a separately curated
    single-task-type file.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"forget_trajectories_path not found: {jsonl_path}")

    index: Dict[str, Dict[str, Any]] = {}
    num_lines = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            num_lines += 1
            traj = json.loads(line)
            if task_type_filter is not None and traj.get("task_type") != task_type_filter:
                continue
            obs0 = traj.get(key_field)
            if not obs0:
                continue
            obs0 = _normalize_obs_key(str(obs0))
            steps = []
            for step in traj.get("steps", []):
                prompt = step.get("prompt")
                response = step.get("model_response")
                action = step.get("action")
                if not prompt or not response or action is None:
                    continue
                steps.append(
                    {
                        "prompt": str(prompt),
                        "model_response": str(response),
                        "action": str(action).strip(),
                        "observation": str(step.get("observation", "")),
                    }
                )
            if not steps:
                continue
            if obs0 in index:
                # (silenced per-line to avoid spamming ~200+ collision
                # warnings at startup; see load count vs index size below)
                continue
            index[obs0] = {
                "task_id": traj.get("task_id", ""),
                "task_type": traj.get("task_type", ""),
                "steps": steps,
            }

    num_ambiguous_removed = 0
    if exclude_ambiguous_from is not None:
        baseline_path = Path(exclude_ambiguous_from)
        if baseline_path.exists():
            ambiguous_keys = set()
            with baseline_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    other = json.loads(line)
                    if other.get("task_type") == "pick_clean_then_place_in_recep":
                        continue
                    obs0 = other.get("initial_observation")
                    if obs0 in index:
                        ambiguous_keys.add(obs0)
            for key in ambiguous_keys:
                del index[key]
            num_ambiguous_removed = len(ambiguous_keys)
        else:
            print(
                "WARNING: exclude_ambiguous_from path not found (%s); skipping cross-task-type "
                "ambiguity check for the forget index." % exclude_ambiguous_from,
                flush=True,
            )

    print(
        "Loaded forget full-trajectory index from %s: %s trajectories read, %s unique game states "
        "indexed, %s removed for colliding with a non-pick_clean task's initial_observation "
        "(likely a mislabeled ALFWorld trial: folder name vs. actual task text mismatch)."
        % (jsonl_path, num_lines, len(index), num_ambiguous_removed),
        flush=True,
    )
    return index


def load_retain_full_trajectories(
    jsonl_path: str = DEFAULT_BASELINE_ALL_TASKS_PATH,
    exclude_task_type: str = "pick_clean_then_place_in_recep",
) -> Dict[str, Dict[str, Any]]:
    """Load every task whose task_type != exclude_task_type from a
    collection/collect_trajectories.py --all-tasks output file, indexed by
    initial_observation exactly like load_forget_full_trajectories. Used by
    ForgetRetainTrajectoryInjectingCollector to inject a forced "stay close
    to baseline" retain trajectory into every non-forget GRPO group.

    No ambiguity cross-check is needed here: any initial_observation that
    also appears under exclude_task_type is exactly the case
    load_forget_full_trajectories's own exclude_ambiguous_from logic strips
    from the FORGET index, so it naturally falls through to this retain
    index instead -- which is the desired behavior (a game with an
    ambiguous/mislabeled task type gets treated as retain, never as
    forget).
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"retain trajectories baseline path not found: {jsonl_path}")

    index: Dict[str, Dict[str, Any]] = {}
    num_lines = 0
    num_skipped_type = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            num_lines += 1
            traj = json.loads(line)
            if traj.get("task_type") == exclude_task_type:
                num_skipped_type += 1
                continue
            obs0 = traj.get("initial_observation")
            if not obs0:
                continue
            steps = []
            for step in traj.get("steps", []):
                prompt = step.get("prompt")
                response = step.get("model_response")
                action = step.get("action")
                if not prompt or not response or action is None:
                    continue
                steps.append(
                    {
                        "prompt": str(prompt),
                        "model_response": str(response),
                        "action": str(action).strip(),
                        "observation": str(step.get("observation", "")),
                    }
                )
            if not steps:
                continue
            if obs0 in index:
                continue
            index[obs0] = {
                "task_id": traj.get("task_id", ""),
                "task_type": traj.get("task_type", ""),
                "steps": steps,
            }

    print(
        "Loaded retain full-trajectory index from %s: %s trajectories read, %s excluded as %s, "
        "%s unique retain game states indexed."
        % (jsonl_path, num_lines, num_skipped_type, exclude_task_type, len(index)),
        flush=True,
    )
    return index


def load_forget_full_trajectories_by_game_file(
    jsonl_path: str,
    task_type_filter: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """v17 fix: same purpose as load_forget_full_trajectories, but indexed
    by the exact ALFWorld `game_file` path instead of the rendered
    initial_observation text.

    Why this matters: matching on rendered text is inherently lossy --
    ALFWorld reuses templated room/task descriptions across many distinct
    physical game files, so a large fraction of games share identical
    initial_observation text with at least one OTHER, different game
    (measured directly on this project's baseline collection: 60-68% of
    entries in every non-pick_clean task type collide with at least one
    other game's key). Matching on `game_file` instead is exact and
    unambiguous by construction -- two different physical ALFWorld trials
    never share a game_file path -- so this also makes the old
    exclude_ambiguous_from cross-referencing hack in
    load_forget_full_trajectories entirely unnecessary here: filtering by
    task_type_filter on load already guarantees every indexed entry is
    genuinely that task type, regardless of what text it happens to render.

    Requires the training run to actually expose `game_file` in its
    non_tensor_batch (see agent_system/environments/env_manager.py's
    AlfWorldEnvironmentManager.reset/step, threaded through by
    rollout_loop.py's preprocess_single_sample) -- older checkpoints/logs
    that predate this change won't have it.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"forget_trajectories_path not found: {jsonl_path}")

    index: Dict[str, Dict[str, Any]] = {}
    num_lines = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            num_lines += 1
            traj = json.loads(line)
            if task_type_filter is not None and traj.get("task_type") != task_type_filter:
                continue
            game_file = traj.get("game_file")
            if not game_file:
                continue
            steps = []
            for step in traj.get("steps", []):
                prompt = step.get("prompt")
                response = step.get("model_response")
                action = step.get("action")
                if not prompt or not response or action is None:
                    continue
                steps.append(
                    {
                        "prompt": str(prompt),
                        "model_response": str(response),
                        "action": str(action).strip(),
                        "observation": str(step.get("observation", "")),
                    }
                )
            if not steps:
                continue
            if game_file in index:
                continue
            index[game_file] = {
                "task_id": traj.get("task_id", ""),
                "task_type": traj.get("task_type", ""),
                "game_file": game_file,
                "steps": steps,
            }

    print(
        "Loaded forget full-trajectory index (by game_file) from %s: %s trajectories read, "
        "%s unique games indexed (task_type_filter=%r)."
        % (jsonl_path, num_lines, len(index), task_type_filter),
        flush=True,
    )
    return index


def load_retain_full_trajectories_by_game_file(
    jsonl_path: str = DEFAULT_BASELINE_ALL_TASKS_PATH,
    exclude_task_type: str = "pick_clean_then_place_in_recep",
) -> Dict[str, Dict[str, Any]]:
    """v17 fix: game_file-keyed counterpart to load_retain_full_trajectories.
    See load_forget_full_trajectories_by_game_file's docstring for why
    game_file is the correct join key (60-68% of retain entries collide on
    initial_observation text with a different, unrelated game). Disjoint
    from load_forget_full_trajectories_by_game_file by construction (a
    game_file can only ever have one task_type)."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"retain trajectories baseline path not found: {jsonl_path}")

    index: Dict[str, Dict[str, Any]] = {}
    num_lines = 0
    num_skipped_type = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            num_lines += 1
            traj = json.loads(line)
            if traj.get("task_type") == exclude_task_type:
                num_skipped_type += 1
                continue
            game_file = traj.get("game_file")
            if not game_file:
                continue
            steps = []
            for step in traj.get("steps", []):
                prompt = step.get("prompt")
                response = step.get("model_response")
                action = step.get("action")
                if not prompt or not response or action is None:
                    continue
                steps.append(
                    {
                        "prompt": str(prompt),
                        "model_response": str(response),
                        "action": str(action).strip(),
                        "observation": str(step.get("observation", "")),
                    }
                )
            if not steps:
                continue
            if game_file in index:
                continue
            index[game_file] = {
                "task_id": traj.get("task_id", ""),
                "task_type": traj.get("task_type", ""),
                "game_file": game_file,
                "steps": steps,
            }

    print(
        "Loaded retain full-trajectory index (by game_file) from %s: %s trajectories read, "
        "%s excluded as %s, %s unique retain games indexed."
        % (jsonl_path, num_lines, num_skipped_type, exclude_task_type, len(index)),
        flush=True,
    )
    return index


def _append_response_position_ids(prompt_position_ids: torch.Tensor, response_length: int) -> torch.Tensor:
    """Mirrors RayPPOTrainer._append_response_position_ids /
    vllm_rollout_spmd.py's own response-position-id extension exactly:
    continue incrementing from the prompt's last position id for the full
    (padded) response width."""
    batch_size = prompt_position_ids.size(0)
    delta = torch.arange(
        1,
        int(response_length) + 1,
        device=prompt_position_ids.device,
        dtype=prompt_position_ids.dtype,
    )
    delta = delta.unsqueeze(0).expand(batch_size, -1)
    response_position_ids = prompt_position_ids[..., -1:] + delta
    return torch.cat([prompt_position_ids, response_position_ids], dim=-1)


class ForgetTrajectoryInjectingCollector:
    """Wraps a TrajectoryCollector. After a real TRAINING rollout completes
    (is_train=True only -- validation/eval rollouts are untouched), for
    every GRPO group (uid) whose underlying game is in the forget set,
    builds and appends one synthetic trajectory from the recorded
    (prompt, model_response) pairs, with its reward forced to
    min(that group's real rollout rewards this step) - extra_penalty.

    NOTE: prior to this fix, the formula additionally clamped the inner
    min at 0.0 (`min(min(rewards), 0.0) - extra_penalty`). For any task
    whose episode-level reward is bounded below by 0 (e.g. pick_clean's
    binary success/fail signal), that outer clamp made the result a
    constant `-extra_penalty` regardless of what the real rollouts in the
    group actually scored -- silently defeating the "adaptive floor"
    design intent described below. Removed so forget_reward genuinely
    tracks the group's real minimum in every case.

    extra_penalty (default 0.0, i.e. v12's exact behavior): an additional
    fixed amount subtracted on top of the adaptive floor. v12 relies purely
    on "at least as bad as the worst real outcome this step" with no
    external magnitude to tune; v15 adds a small tunable margin below that
    floor for cases where "tied with the worst real sample" isn't a strong
    enough push (e.g. every real sample in the group also failed, so the
    adaptive floor is already low but not clearly WORSE than any of them)."""

    def __init__(
        self,
        base_collector,
        tokenizer,
        forget_trajectories: Dict[str, Dict[str, Any]],
        config,
        extra_penalty: float = 0.0,
        match_key: str = "anchor_obs",
    ):
        self._base = base_collector
        self.tokenizer = tokenizer
        self.forget_trajectories = forget_trajectories
        self.config = config
        self.extra_penalty = float(extra_penalty)
        # WebShop's anchor_obs has its instruction text stripped out by
        # WebshopEnvironmentManager.format_obs() (see agent_system/
        # environments/env_manager.py) before it's used as the GiGPO/Traj
        # "anchor" observation -- collapsing to a near-constant string
        # ("'Search'") across virtually every task at reset time, since the
        # instruction was the ONLY task-distinguishing content there. Using
        # anchor_obs to identify "is this rollout's game in the forget set"
        # for WebShop therefore spuriously matches ~100% of rollouts against
        # whatever single forget-trajectory entry happens to occupy that
        # collapsed key, rather than only the intended forget subset.
        # match_key="task_key" (populated by WebshopEnvironmentManager
        # specifically for this purpose, threaded through non_tensor_batch
        # like ALFWorld's "game_file") is the WebShop-appropriate join key;
        # ALFWorld callers leave this at its "anchor_obs" default, unchanged.
        self.match_key = match_key
        self._injected_rows = 0
        self._injected_groups = 0
        self._steps_seen = 0

    def __getattr__(self, name):
        # Proxy everything else (build_prompt_batch, build_prompt_sample,
        # etc.) straight through to the wrapped collector.
        return getattr(self._base, name)

    def multi_turn_loop(self, gen_batch, actor_rollout_wg, envs, is_train: bool = True):
        batch = self._base.multi_turn_loop(
            gen_batch=gen_batch, actor_rollout_wg=actor_rollout_wg, envs=envs, is_train=is_train
        )
        if not is_train:
            return batch
        return self._inject(batch)

    def _inject(self, batch: DataProto) -> DataProto:
        self._steps_seen += 1
        non_tensor = batch.non_tensor_batch
        required = ["uid", "traj_uid", self.match_key, "step_num", "episode_rewards"]
        missing = [key for key in required if key not in non_tensor]
        if missing:
            print(
                "v12 injector: rollout batch is missing required field(s) %s; skipping injection this step."
                % (missing,),
                file=sys.stderr,
                flush=True,
            )
            return batch

        uid_arr = non_tensor["uid"]
        traj_uid_arr = non_tensor["traj_uid"]
        anchor_obs_arr = non_tensor[self.match_key]
        step_num_arr = non_tensor["step_num"]
        episode_rewards_arr = non_tensor["episode_rewards"]
        n = len(batch)

        group_init_obs: Dict[object, str] = {}
        group_traj_rewards: Dict[object, Dict[object, float]] = defaultdict(dict)
        for i in range(n):
            g = uid_arr[i]
            t = traj_uid_arr[i]
            if g not in group_init_obs and int(step_num_arr[i]) == 0:
                obs = anchor_obs_arr[i]
                if isinstance(obs, (str, np.str_)):
                    group_init_obs[g] = str(obs)
            group_traj_rewards[g][t] = float(episode_rewards_arr[i])

        matching_groups = []
        for g, obs in group_init_obs.items():
            entry = self.forget_trajectories.get(_normalize_obs_key(obs))
            if entry is None:
                continue
            rewards = list(group_traj_rewards.get(g, {}).values())
            if not rewards:
                continue
            forget_reward = min(rewards) - self.extra_penalty
            matching_groups.append((g, entry, forget_reward))

        if not matching_groups:
            sample_obs = next(iter(group_init_obs.values()), None)
            print(
                "v12 injector DIAGNOSTIC: step=%s rows=%s groups_with_step0_str_obs=%s/%s "
                "forget_index_size=%s sample_group_obs=%r"
                % (
                    self._steps_seen,
                    n,
                    len(group_init_obs),
                    len(set(uid_arr.tolist())),
                    len(self.forget_trajectories),
                    (sample_obs[:200] if sample_obs else None),
                ),
                file=sys.stderr,
                flush=True,
            )
            return batch

        extra = self._build_extra_dataproto(batch, matching_groups)
        if extra is None or len(extra) == 0:
            return batch

        real_reduced, real_extra_kept = self._split_known_keys(batch)
        combined = DataProto.concat([real_reduced, extra])
        combined = self._restore_known_extra_keys(
            combined, real_extra_kept, n_real=len(batch), n_extra=len(extra)
        )

        self._injected_rows += len(extra)
        self._injected_groups += len(matching_groups)
        print(
            "Traj-unlearn (forget-inject): injected %s forget-trajectory row(s) across %s matching game "
            "instance(s) this step (cumulative rows=%s over %s steps, groups=%s)."
            % (
                len(extra),
                len(matching_groups),
                self._injected_rows,
                self._steps_seen,
                self._injected_groups,
            ),
            file=sys.stderr,
            flush=True,
        )
        return combined

    def _split_known_keys(self, batch: DataProto):
        non_tensor_keys = set(batch.non_tensor_batch.keys())
        extra_non_tensor = non_tensor_keys - set(CORE_NON_TENSOR_KEYS)
        success_rate_keys = {k for k in extra_non_tensor if k == "success_rate" or k.endswith("_success_rate")}
        unexpected = extra_non_tensor - KNOWN_EXTRA_NON_TENSOR_KEYS - success_rate_keys
        if unexpected:
            raise RuntimeError(
                f"v12 injector: real rollout batch has unexpected non_tensor_batch key(s) "
                f"{sorted(unexpected)} that unlearn/unlearn_inject.py does not know how to backfill "
                "for its synthetic rows. Refusing to inject silently -- update "
                "CORE_NON_TENSOR_KEYS/KNOWN_EXTRA_NON_TENSOR_KEYS to handle them."
            )

        tensor_keys = set(batch.batch.keys())
        extra_tensor = tensor_keys - set(CORE_TENSOR_KEYS)
        unexpected_tensor = extra_tensor - KNOWN_EXTRA_TENSOR_KEYS
        if unexpected_tensor:
            raise RuntimeError(
                f"v12 injector: real rollout batch has unexpected tensor key(s) "
                f"{sorted(unexpected_tensor)} that unlearn/unlearn_inject.py does not know how to "
                "backfill for its synthetic rows."
            )

        kept = {key: batch.non_tensor_batch[key] for key in extra_non_tensor}
        reduced = batch.select(batch_keys=CORE_TENSOR_KEYS, non_tensor_batch_keys=CORE_NON_TENSOR_KEYS)
        # Same dtype-promotion hazard as is_action_valid (see
        # _KNOWN_KEY_DTYPES below): episode_lengths/episode_rewards have
        # been observed to arrive as `object` arrays despite rollout_loop.py
        # constructing them as float32, apparently coerced somewhere in the
        # to_list_of_dict/collate_fn round-trip. Concatenating an `object`
        # array with our properly-typed synthetic float32 values silently
        # upcasts the WHOLE result (real rows included) to `object`, which
        # breaks verl's own compute_data_metrics
        # (`non_tensor_batch["episode_lengths"][idx].min().item()` requires
        # a real numpy scalar, not a plain Python float from an object
        # array's reduction). Force these to their known-correct dtype
        # up front, regardless of what dtype they happen to arrive in.
        for key, dtype in CORE_NUMERIC_NON_TENSOR_DTYPES.items():
            if key in reduced.non_tensor_batch and reduced.non_tensor_batch[key].dtype != dtype:
                reduced.non_tensor_batch[key] = reduced.non_tensor_batch[key].astype(dtype)
        return reduced, kept

    # Explicit dtypes for the fields we hardcode a fill value for. Not
    # inherited from real_values.dtype: is_action_valid in particular has
    # been observed to arrive as a numpy `object` array (holding plain
    # Python bools) rather than a `bool` array in some rollout paths --
    # concatenating that with a `bool`-dtype filler makes numpy fall back
    # to `object` for the WHOLE result (including the real rows' region),
    # which silently breaks apply_invalid_action_penalty's
    # `.astype(np.float32)` call downstream (plain Python bool/int have no
    # `.astype`). Forcing both sides to a known-correct dtype up front
    # avoids depending on which representation the real array happens to
    # be in.
    _KNOWN_KEY_DTYPES = {
        "sample_id": np.int64,
        "rollout_id": np.int64,
        "tool_callings": np.float32,
        "is_action_valid": np.bool_,
    }

    @classmethod
    def _restore_known_extra_keys(
        cls,
        combined: DataProto,
        real_extra_kept: Dict[str, np.ndarray],
        n_real: int,
        n_extra: int,
    ) -> DataProto:
        for key, real_values in real_extra_kept.items():
            if key == "success_rate" or key.endswith("_success_rate"):
                # Batch-level aggregate scalar, identical on every real row
                # -- just broadcast the same value onto the injected rows so
                # train-time success-rate metrics stay correct/unaffected.
                fill_value = real_values[0]
            elif key in ("sample_id", "rollout_id"):
                fill_value = -1
            elif key == "step_id":
                fill_value = "forget_inject"
            elif key == "tool_callings":
                fill_value = 0.0
            elif key == "is_action_valid":
                fill_value = True
            else:
                fill_value = real_values[0]

            target_dtype = cls._KNOWN_KEY_DTYPES.get(key, real_values.dtype)
            if real_values.dtype != target_dtype:
                real_values = real_values.astype(target_dtype)
            # real_values is not always 1-D: a field like "raw_prompt" (a
            # length-1 list of chat-message dicts, identical shape on every
            # row) gets auto-stacked by numpy into a genuinely 2-D object
            # array (n_real, 1), not a 1-D array of opaque objects. Build
            # the filler with the SAME trailing shape and use a plain
            # broadcast assignment (well-defined for any trailing shape,
            # unlike np.full(n, fill_value, ...), which tries to broadcast
            # an array-like fill_value into the output shape and either
            # raises or "succeeds" only by coincidence for length-1 values).
            filler = np.empty((n_extra,) + real_values.shape[1:], dtype=target_dtype)
            filler[:] = fill_value
            combined.non_tensor_batch[key] = np.concatenate([real_values, filler], axis=0)
        return combined

    def _build_extra_dataproto(self, real_batch: DataProto, matching_groups) -> Optional[DataProto]:
        rows = []
        for g, entry, forget_reward in matching_groups:
            steps = entry["steps"]
            traj_uid = f"forget-inject-{uuid.uuid4()}"
            num_steps = len(steps)
            for step_idx, step in enumerate(steps):
                rows.append(
                    {
                        "uid": g,
                        "traj_uid": traj_uid,
                        "step_num": step_idx,
                        "prompt_text": step["prompt"],
                        "response_text": step["model_response"],
                        "episode_rewards": forget_reward,
                        "episode_lengths": float(num_steps),
                        "reward": forget_reward if step_idx == num_steps - 1 else 0.0,
                        "anchor_obs": step.get("observation", ""),
                    }
                )
        if not rows:
            return None

        prompts_text = [r["prompt_text"] for r in rows]
        prompt_batch = self._base.build_prompt_batch(obs_contents=prompts_text, meta_info={})
        prompt_input_ids = prompt_batch.batch["input_ids"]
        prompt_attention_mask = prompt_batch.batch["attention_mask"]
        prompt_position_ids = prompt_batch.batch["position_ids"]

        max_response_length = int(self.config.data.max_response_length)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0
        eos_id = self.tokenizer.eos_token_id

        n = len(rows)
        responses = torch.full((n, max_response_length), pad_id, dtype=prompt_input_ids.dtype)
        for idx, row in enumerate(rows):
            ids = self.tokenizer(row["response_text"], add_special_tokens=False)["input_ids"]
            if eos_id is not None:
                ids = list(ids) + [eos_id]
            length = min(len(ids), max_response_length)
            if length > 0:
                responses[idx, :length] = torch.tensor(ids[:length], dtype=prompt_input_ids.dtype)

        eos_for_mask = eos_id if eos_id is not None else pad_id
        response_attention_mask = get_response_mask(
            response_id=responses, eos_token=eos_for_mask, dtype=prompt_attention_mask.dtype
        )

        input_ids = torch.cat([prompt_input_ids, responses], dim=-1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1)
        position_ids = _append_response_position_ids(prompt_position_ids, max_response_length)

        tensors = {
            "prompts": prompt_input_ids,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        non_tensors = {
            "uid": np.array([r["uid"] for r in rows], dtype=object),
            "traj_uid": np.array([r["traj_uid"] for r in rows], dtype=object),
            "step_num": np.array([r["step_num"] for r in rows], dtype=np.int64),
            "episode_rewards": np.array([r["episode_rewards"] for r in rows], dtype=np.float32),
            "episode_lengths": np.array([r["episode_lengths"] for r in rows], dtype=np.float32),
            "data_source": np.array(["alfworld"] * n, dtype=object),
            "rewards": np.array([r["reward"] for r in rows], dtype=object),
            "active_masks": np.array([True] * n, dtype=object),
            "anchor_obs": np.array([r["anchor_obs"] for r in rows], dtype=object),
        }
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=dict(real_batch.meta_info))


class ForgetRetainTrajectoryInjectingCollector(ForgetTrajectoryInjectingCollector):
    """v16 (forget+retain design): injects a forced synthetic trajectory
    into EVERY GRPO group whose underlying game is present in the baseline
    all-task collection, not just forget-set games.

    - forget-set (pick_clean) games: v12's suppressive floor,
      reward = min(group's real min reward this step, 0) - extra_penalty
      (unchanged from ForgetTrajectoryInjectingCollector).
    - every other (retain) game: a symmetric ENCOURAGING floor,
      reward = min(group's real max reward this step, 0) + retain_bonus.

    Both formulas are capped from above at their respective bonus/penalty
    offset (since min(..., 0) <= 0) -- the injected row never claims to be
    "better than perfect" or "worse than the worst real failure" beyond the
    tunable margin. Standard GRPO/PPO handles both kinds of injected row
    identically to a real sample (log_prob, ref KL, advantage, policy
    gradient) -- plain stock RayPPOTrainer, plain stock
    EpisodeRewardManager, no extra loss term (e.g. v13's asymmetric KL) is
    needed: the retain injection already supplies the "stay close to
    baseline" pressure directly through the advantage computation.

    A game's key can only ever match forget_trajectories OR
    retain_trajectories, never both -- see load_retain_full_trajectories's
    docstring on why the two indices are disjoint by construction.
    """

    def __init__(
        self,
        base_collector,
        tokenizer,
        forget_trajectories: Dict[str, Dict[str, Any]],
        retain_trajectories: Dict[str, Dict[str, Any]],
        config,
        extra_penalty: float = 0.0,
        retain_bonus: float = 0.0,
    ):
        super().__init__(
            base_collector=base_collector,
            tokenizer=tokenizer,
            forget_trajectories=forget_trajectories,
            config=config,
            extra_penalty=extra_penalty,
        )
        self.retain_trajectories = retain_trajectories
        self.retain_bonus = float(retain_bonus)
        self._injected_forget_rows = 0
        self._injected_retain_rows = 0

    def _inject(self, batch: DataProto) -> DataProto:
        self._steps_seen += 1
        non_tensor = batch.non_tensor_batch
        required = ["uid", "traj_uid", "anchor_obs", "step_num", "episode_rewards"]
        missing = [key for key in required if key not in non_tensor]
        if missing:
            print(
                "v16(forget+retain) injector: rollout batch is missing required field(s) %s; "
                "skipping injection this step." % (missing,),
                flush=True,
            )
            return batch

        uid_arr = non_tensor["uid"]
        traj_uid_arr = non_tensor["traj_uid"]
        anchor_obs_arr = non_tensor["anchor_obs"]
        step_num_arr = non_tensor["step_num"]
        episode_rewards_arr = non_tensor["episode_rewards"]
        n = len(batch)

        group_init_obs: Dict[object, str] = {}
        group_traj_rewards: Dict[object, Dict[object, float]] = defaultdict(dict)
        for i in range(n):
            g = uid_arr[i]
            t = traj_uid_arr[i]
            if g not in group_init_obs and int(step_num_arr[i]) == 0:
                obs = anchor_obs_arr[i]
                if isinstance(obs, (str, np.str_)):
                    group_init_obs[g] = str(obs)
            group_traj_rewards[g][t] = float(episode_rewards_arr[i])

        matching_groups = []
        n_forget_groups = 0
        n_retain_groups = 0
        for g, obs in group_init_obs.items():
            rewards = list(group_traj_rewards.get(g, {}).values())
            if not rewards:
                continue
            forget_entry = self.forget_trajectories.get(obs)
            if forget_entry is not None:
                forget_reward = min(min(rewards), 0.0) - self.extra_penalty
                matching_groups.append((g, forget_entry, forget_reward))
                n_forget_groups += 1
                continue
            retain_entry = self.retain_trajectories.get(obs)
            if retain_entry is not None:
                retain_reward = min(max(rewards), 0.0) + self.retain_bonus
                matching_groups.append((g, retain_entry, retain_reward))
                n_retain_groups += 1

        if not matching_groups:
            sample_obs = next(iter(group_init_obs.values()), None)
            print(
                "v16(forget+retain) injector DIAGNOSTIC: step=%s rows=%s groups_with_step0_str_obs=%s/%s "
                "forget_index_size=%s retain_index_size=%s sample_group_obs=%r"
                % (
                    self._steps_seen,
                    n,
                    len(group_init_obs),
                    len(set(uid_arr.tolist())),
                    len(self.forget_trajectories),
                    len(self.retain_trajectories),
                    (sample_obs[:200] if sample_obs else None),
                ),
                flush=True,
            )
            return batch

        extra = self._build_extra_dataproto(batch, matching_groups)
        if extra is None or len(extra) == 0:
            return batch

        real_reduced, real_extra_kept = self._split_known_keys(batch)
        combined = DataProto.concat([real_reduced, extra])
        combined = self._restore_known_extra_keys(
            combined, real_extra_kept, n_real=len(batch), n_extra=len(extra)
        )

        self._injected_rows += len(extra)
        self._injected_groups += len(matching_groups)
        print(
            "Traj-unlearn(forget+retain-inject): step=%s injected %s row(s) across %s matching "
            "game instance(s) (forget_groups=%s, retain_groups=%s; cumulative rows=%s over %s steps)."
            % (
                self._steps_seen,
                len(extra),
                len(matching_groups),
                n_forget_groups,
                n_retain_groups,
                self._injected_rows,
                self._steps_seen,
            ),
            flush=True,
        )
        return combined


class ForgetRetainFixedRewardInjectingCollector(ForgetTrajectoryInjectingCollector):
    """v17: like ForgetRetainTrajectoryInjectingCollector (both forget and
    retain games get a forced synthetic-sample injection from the baseline
    all-task collection), but the injected reward is a FIXED constant taken
    directly from ALFWorld's own native reward scale instead of an
    adaptive-floor-plus-tunable-penalty/bonus formula:

    - forget-set (pick_clean) games: injected reward = 0.0 -- exactly what
      `agent_system/environments/env_package/alfworld/envs.py::compute_reward`
      (`10.0 * float(info['won'])`, text-only case) gives a fully failed
      episode. No penalty hyperparameter to tune.
    - retain (every other) games: injected reward = 10.0 -- exactly what a
      fully successful episode gets. No bonus hyperparameter to tune.

    Conceptually: every injected trajectory carries an implicit
    info["forget"]/info["retain"] flag (mutually exclusive, see
    load_retain_full_trajectories's docstring for why the two source
    indices are disjoint by construction), and

        reward = 0.0            if info["forget"]
                 10.0            if info["retain"]
                 10.0*info["won"] otherwise (unchanged, real on-policy rows)

    -- i.e. the injected reward always lives on the SAME 0-10 scale real
    rollouts already use, sidestepping the v13/v15/v16 question of what
    penalty/bonus magnitude is "reasonable" relative to that scale.

    Meant to be paired with AlfworldEnvs' guaranteed-coverage forget-pool
    cycling (env.alfworld.forget_config_path / env.alfworld.forget_env_num,
    see envs.py) so that every forget-set game is actually rolled out (and
    therefore actually injected) at least once over the course of training,
    rather than depending on chance the way v12/v15/v16's default (fully
    random, unrestricted) sampling does.

    Matching fix: unlike v12/v15/v16 (which match via the rendered
    initial_observation text) and the first version of this class (same
    text-based matching), this version matches via the exact `game_file`
    path (see load_forget_full_trajectories_by_game_file /
    load_retain_full_trajectories_by_game_file). Text-based matching was
    found to be wrong ~60-68% of the time for retain games -- ALFWorld
    reuses templated room/task text across many distinct physical game
    files, so most retain entries collided with at least one other,
    different game sharing the same rendered text, meaning a large
    fraction of "encourage toward baseline" injections were actually
    forcing reward=10 onto an action sequence recorded for a DIFFERENT
    game than the one actually being trained on that step -- a likely
    major contributor to the first v17 run's broad success-rate/exact-match
    regressions on tasks with no semantic connection to what was injected.
    """

    FORGET_REWARD = 0.0
    RETAIN_REWARD = 10.0
    LOG_TAG = "v17"

    def __init__(
        self,
        base_collector,
        tokenizer,
        forget_trajectories: Dict[str, Dict[str, Any]],
        retain_trajectories: Dict[str, Dict[str, Any]],
        config,
    ):
        super().__init__(
            base_collector=base_collector,
            tokenizer=tokenizer,
            forget_trajectories=forget_trajectories,
            config=config,
            extra_penalty=0.0,
        )
        self.retain_trajectories = retain_trajectories

    def _inject(self, batch: DataProto) -> DataProto:
        self._steps_seen += 1
        non_tensor = batch.non_tensor_batch
        required = ["uid", "traj_uid", "game_file", "step_num", "episode_rewards"]
        missing = [key for key in required if key not in non_tensor]
        if missing:
            print(
                "%s(forget+retain, fixed-reward) injector: rollout batch is missing required "
                "field(s) %s; skipping injection this step." % (self.LOG_TAG, missing),
                flush=True,
            )
            return batch

        uid_arr = non_tensor["uid"]
        traj_uid_arr = non_tensor["traj_uid"]
        game_file_arr = non_tensor["game_file"]
        step_num_arr = non_tensor["step_num"]
        episode_rewards_arr = non_tensor["episode_rewards"]
        n = len(batch)

        # Exact, unambiguous matching key: the ALFWorld game_file path
        # (threaded through by env_manager.py/rollout_loop.py), not the
        # rendered initial_observation text -- see
        # load_forget_full_trajectories_by_game_file's docstring for why
        # text-based matching was wrong ~60-68% of the time for retain
        # games.
        group_game_file: Dict[object, str] = {}
        group_has_real_row: Dict[object, bool] = defaultdict(bool)
        for i in range(n):
            g = uid_arr[i]
            if g not in group_game_file and int(step_num_arr[i]) == 0:
                gf = game_file_arr[i]
                if isinstance(gf, (str, np.str_)) and gf:
                    group_game_file[g] = str(gf)
            group_has_real_row[g] = True
        _ = episode_rewards_arr  # unused: rewards are fixed constants, not group-adaptive

        matching_groups = []
        n_forget_groups = 0
        n_retain_groups = 0
        for g, gf in group_game_file.items():
            if not group_has_real_row.get(g):
                continue
            forget_entry = self.forget_trajectories.get(gf)
            if forget_entry is not None:
                matching_groups.append((g, forget_entry, self.FORGET_REWARD))
                n_forget_groups += 1
                continue
            retain_entry = self.retain_trajectories.get(gf)
            if retain_entry is not None:
                matching_groups.append((g, retain_entry, self.RETAIN_REWARD))
                n_retain_groups += 1

        if not matching_groups:
            sample_gf = next(iter(group_game_file.values()), None)
            print(
                "%s(forget+retain, fixed-reward) injector DIAGNOSTIC: step=%s rows=%s "
                "groups_with_step0_game_file=%s/%s forget_index_size=%s retain_index_size=%s "
                "sample_group_game_file=%r"
                % (
                    self.LOG_TAG,
                    self._steps_seen,
                    n,
                    len(group_game_file),
                    len(set(uid_arr.tolist())),
                    len(self.forget_trajectories),
                    len(self.retain_trajectories),
                    sample_gf,
                ),
                flush=True,
            )
            return batch

        extra = self._build_extra_dataproto(batch, matching_groups)
        if extra is None or len(extra) == 0:
            return batch

        real_reduced, real_extra_kept = self._split_known_keys(batch)
        combined = DataProto.concat([real_reduced, extra])
        combined = self._restore_known_extra_keys(
            combined, real_extra_kept, n_real=len(batch), n_extra=len(extra)
        )

        self._injected_rows += len(extra)
        self._injected_groups += len(matching_groups)
        print(
            "Traj-unlearn-%s(forget+retain, fixed-reward inject): step=%s injected %s row(s) "
            "across %s matching game instance(s) (forget_groups=%s [reward=%.1f], "
            "retain_groups=%s [reward=%.1f]; cumulative rows=%s over %s steps)."
            % (
                self.LOG_TAG,
                self._steps_seen,
                len(extra),
                len(matching_groups),
                n_forget_groups,
                self.FORGET_REWARD,
                n_retain_groups,
                self.RETAIN_REWARD,
                self._injected_rows,
                self._steps_seen,
            ),
            flush=True,
        )
        return combined


class ForgetRetainPartialRewardInjectingCollector(ForgetRetainFixedRewardInjectingCollector):
    """v18: identical mechanism to v17 (ForgetRetainFixedRewardInjectingCollector
    -- same game_file-exact matching, same guaranteed forget-pool coverage,
    same every-matching-group-gets-injected scope), but softens the retain
    reward from a full 10.0 down to a partial 5.0 (10.0 * 0.5):

        reward = 0.0             if info["forget"] == 1
                 10.0*info["retain"]  if info["retain"] == 0.5  (i.e. 5.0)
                 10.0*info["won"]     otherwise (unchanged, real on-policy rows)

    Motivation (from v17's own results, see collection/v17.md): forcing a
    full reward=10.0 on every retain-matching injected row destabilized
    training broadly (grad_norm went non-finite -- and the whole optimizer
    step was skipped, see dp_actor.py -- on 14/56 steps, 25%, vs. 0/41 for
    v15's much smaller forget-only perturbation) and dragged down success
    rate on task types that should have been purely reinforced, not just the
    forget target. Halving the retain reward's magnitude is a first, minimal
    knob to try to reduce that collateral perturbation while keeping the
    forget reward (0.0) unchanged.
    """

    FORGET_REWARD = 0.0
    RETAIN_REWARD = 5.0
    LOG_TAG = "v18"


class ForgetRetainSignedRewardInjectingCollector(ForgetRetainFixedRewardInjectingCollector):
    """v19: identical mechanism to v17/v18 (ForgetRetainFixedRewardInjectingCollector
    -- same game_file-exact matching, same guaranteed forget-pool coverage,
    same every-matching-group-gets-injected scope), but the forget reward is
    now NEGATIVE (explicitly worse than any real ALFWorld outcome, which is
    bounded below at 0.0) instead of v17/v18's floor of 0.0:

        reward = -5.0             if info["forget"] != 0
                 5.0               if info["retain"] != 0
                 10.0*info["won"]  otherwise (unchanged, real on-policy rows)

    i.e. forget reward = -5.0 (below ALFWorld's own native floor -- explicitly
    "worse than any real failure", not just "as bad as a real failure"),
    retain reward = 5.0 (same partial value as v18, not v17's full 10.0).
    Run alongside a 16:8 forget:retain task-slot split (FORGET_ENV_NUM=16,
    matching the very first v17 attempt's slot ratio, but with the game_file
    exact-matching fix in place) and TOTAL_EPOCHS=41 (ceil(650/16)=41 epochs
    needed for full pick_clean coverage at 16 dedicated slots).
    """

    FORGET_REWARD = -5.0
    RETAIN_REWARD = 5.0
    LOG_TAG = "v19"


class ForgetOnlyGameFileInjectingCollector(ForgetTrajectoryInjectingCollector):
    """v12 mechanism (forget-only, no retain injection at all -- identical
    reward formula to ForgetTrajectoryInjectingCollector:
    reward = min(group's real minimum reward this step, 0) - extra_penalty),
    but matching is done via the exact `game_file` path instead of the
    ambiguous rendered `initial_observation` text.

    Why this matters here specifically: the original v12 forget loader
    (load_forget_full_trajectories) EXCLUDES ambiguous entries (entries
    whose initial_observation text collides with another task_id's) rather
    than risk a silent mismatch -- safe, but it means the forget index only
    covers the UNIQUE-text subset of the forget-set pool, not all of it.
    For pick_clean this subset happened to be large enough (~414/650) that
    it wasn't investigated further at the time. But the later retain-
    matching collision analysis (see load_retain_full_trajectories_by_
    game_file's docstring) found initial_observation collision rates of
    60-68% for the OTHER task types when used as a forget/retain target --
    e.g. pick_heat_then_place_in_recep specifically measured at 65.8%. Using
    the exact `game_file` key instead means every game in the forget pool
    gets a real, unambiguous forget-index entry (verified: forget index
    size equals the full per-task-type trajectory count, no exclusions),
    which is what lets AlfworldEnvs' guaranteed-coverage forget_env_num
    cycling actually inject against EVERY game it dedicates a slot to,
    regardless of which task type is chosen as the forget target.
    """

    def _inject(self, batch: DataProto) -> DataProto:
        self._steps_seen += 1
        non_tensor = batch.non_tensor_batch
        required = ["uid", "traj_uid", "game_file", "step_num", "episode_rewards"]
        missing = [key for key in required if key not in non_tensor]
        if missing:
            print(
                "v12(game_file-matched, forget-only) injector: rollout batch is missing required "
                "field(s) %s; skipping injection this step." % (missing,),
                flush=True,
            )
            return batch

        uid_arr = non_tensor["uid"]
        traj_uid_arr = non_tensor["traj_uid"]
        game_file_arr = non_tensor["game_file"]
        step_num_arr = non_tensor["step_num"]
        episode_rewards_arr = non_tensor["episode_rewards"]
        n = len(batch)

        group_game_file: Dict[object, str] = {}
        group_traj_rewards: Dict[object, Dict[object, float]] = defaultdict(dict)
        for i in range(n):
            g = uid_arr[i]
            t = traj_uid_arr[i]
            if g not in group_game_file and int(step_num_arr[i]) == 0:
                gf = game_file_arr[i]
                if isinstance(gf, (str, np.str_)) and gf:
                    group_game_file[g] = str(gf)
            group_traj_rewards[g][t] = float(episode_rewards_arr[i])

        matching_groups = []
        for g, gf in group_game_file.items():
            entry = self.forget_trajectories.get(gf)
            if entry is None:
                continue
            rewards = list(group_traj_rewards.get(g, {}).values())
            if not rewards:
                continue
            forget_reward = min(min(rewards), 0.0) - self.extra_penalty
            matching_groups.append((g, entry, forget_reward))

        if not matching_groups:
            sample_gf = next(iter(group_game_file.values()), None)
            print(
                "v12(game_file-matched, forget-only) injector DIAGNOSTIC: step=%s rows=%s "
                "groups_with_step0_game_file=%s/%s forget_index_size=%s sample_group_game_file=%r"
                % (
                    self._steps_seen,
                    n,
                    len(group_game_file),
                    len(set(uid_arr.tolist())),
                    len(self.forget_trajectories),
                    sample_gf,
                ),
                flush=True,
            )
            return batch

        extra = self._build_extra_dataproto(batch, matching_groups)
        if extra is None or len(extra) == 0:
            return batch

        real_reduced, real_extra_kept = self._split_known_keys(batch)
        combined = DataProto.concat([real_reduced, extra])
        combined = self._restore_known_extra_keys(
            combined, real_extra_kept, n_real=len(batch), n_extra=len(extra)
        )

        self._injected_rows += len(extra)
        self._injected_groups += len(matching_groups)
        print(
            "Traj-unlearn(game_file-matched, forget-only): injected %s forget-trajectory row(s) "
            "across %s matching game instance(s) this step (cumulative rows=%s over %s steps, groups=%s)."
            % (
                len(extra),
                len(matching_groups),
                self._injected_rows,
                self._steps_seen,
                self._injected_groups,
            ),
            flush=True,
        )
        return combined


class ForgetOnlyFixedRewardInjectingCollector(ForgetTrajectoryInjectingCollector):
    """v21: identical mechanism to ForgetOnlyGameFileInjectingCollector (v20)
    -- forget-only, no retain injection at all, matched via the exact
    `game_file` path -- but the injected reward is a FIXED constant
    (FORGET_REWARD, default -0.1) instead of v12/v20's adaptive
    `min(group's real minimum reward this step, 0) - extra_penalty` formula.

    Rationale: v12/v20's adaptive floor means the actual forget-penalty
    magnitude applied varies step to step with whatever the on-policy
    rollout's worst reward happens to be (usually 0, but not always, if a
    real trajectory in the group failed with a worse-than-0 shaped
    penalty elsewhere in the reward pipeline). This version removes that
    dependency entirely: every injected forget row always gets exactly
    FORGET_REWARD, regardless of what the rest of its GRPO group scored
    this step.
    """

    FORGET_REWARD = -0.1
    LOG_TAG = "v21"

    def _inject(self, batch: DataProto) -> DataProto:
        self._steps_seen += 1
        non_tensor = batch.non_tensor_batch
        required = ["uid", "traj_uid", "game_file", "step_num"]
        missing = [key for key in required if key not in non_tensor]
        if missing:
            print(
                "%s(game_file-matched, forget-only, fixed-reward) injector: rollout batch is "
                "missing required field(s) %s; skipping injection this step." % (self.LOG_TAG, missing),
                flush=True,
            )
            return batch

        uid_arr = non_tensor["uid"]
        game_file_arr = non_tensor["game_file"]
        step_num_arr = non_tensor["step_num"]
        n = len(batch)

        group_game_file: Dict[object, str] = {}
        for i in range(n):
            g = uid_arr[i]
            if g not in group_game_file and int(step_num_arr[i]) == 0:
                gf = game_file_arr[i]
                if isinstance(gf, (str, np.str_)) and gf:
                    group_game_file[g] = str(gf)

        matching_groups = []
        for g, gf in group_game_file.items():
            entry = self.forget_trajectories.get(gf)
            if entry is None:
                continue
            matching_groups.append((g, entry, self.FORGET_REWARD))

        if not matching_groups:
            sample_gf = next(iter(group_game_file.values()), None)
            print(
                "%s(game_file-matched, forget-only, fixed-reward) injector DIAGNOSTIC: step=%s "
                "rows=%s groups_with_step0_game_file=%s/%s forget_index_size=%s sample_group_game_file=%r"
                % (
                    self.LOG_TAG,
                    self._steps_seen,
                    n,
                    len(group_game_file),
                    len(set(uid_arr.tolist())),
                    len(self.forget_trajectories),
                    sample_gf,
                ),
                flush=True,
            )
            return batch

        extra = self._build_extra_dataproto(batch, matching_groups)
        if extra is None or len(extra) == 0:
            return batch

        real_reduced, real_extra_kept = self._split_known_keys(batch)
        combined = DataProto.concat([real_reduced, extra])
        combined = self._restore_known_extra_keys(
            combined, real_extra_kept, n_real=len(batch), n_extra=len(extra)
        )

        self._injected_rows += len(extra)
        self._injected_groups += len(matching_groups)
        print(
            "Traj-unlearn-%s(game_file-matched, forget-only, fixed-reward=%.2f): injected %s "
            "forget-trajectory row(s) across %s matching game instance(s) this step (cumulative "
            "rows=%s over %s steps, groups=%s)."
            % (
                self.LOG_TAG,
                self.FORGET_REWARD,
                len(extra),
                len(matching_groups),
                self._injected_rows,
                self._steps_seen,
                self._injected_groups,
            ),
            flush=True,
        )
        return combined


class ForgetOnlyBaselineExcludedAdvantageCollector(ForgetTrajectoryInjectingCollector):
    """v22: IDENTICAL injection mechanism to v12/v12-reseed
    (ForgetTrajectoryInjectingCollector -- forget-only, no retain injection
    at all, extra_penalty=0.0, matched via the rendered initial_observation
    text). The only change is that injected forget rows are tagged with
    non_tensor_batch["is_forget_injected"] = True (real rows: False), which
    verl.trainer.ppo.core_algos.compute_grpo_outcome_advantage (via
    verl.trainer.ppo.ray_trainer.compute_advantage) uses to EXCLUDE injected
    rows from each GRPO group's own mean/std baseline -- while still
    normalizing the injected row's own (forced-low) reward against that
    (real-only) baseline.

    Rationale: GRPO's advantage is (score - group_mean) / group_std, which
    sums to exactly zero within a group by construction. Folding a
    synthetic, artificially-low-reward injected row INTO that same group's
    mean/std computation drags the mean down purely because of the
    injection, which silently manufactures a spurious POSITIVE advantage
    for whichever real on-policy rollouts happen to share its group that
    step -- even a group of real rollouts that all succeeded identically
    (which would otherwise get exactly zero advantage/signal, having
    nothing left to improve on) gets an artificial "reinforce this" signal
    purely as a mean-shift artifact of the injected row's presence. This
    also means the injected row's own advantage magnitude isn't a clean,
    independent measure of "how much to suppress this trajectory" -- it's
    entangled with whatever the real rollouts in its group happened to
    score. Excluding injected rows from the baseline (but still scoring
    them against it) removes both effects: real rows keep their ordinary,
    undistorted zero-sum GRPO advantages among themselves, and the injected
    row gets a negative advantage that is only a function of its own
    (forced) reward relative to the real rollouts' true mean/std, not an
    artifact of its own presence in that computation.

    Only affects the ONE collector used by v12/v12-reseed/v22 -- v17-v21's
    subclasses each override _inject() completely and are untouched by this
    change.
    """

    def _inject(self, batch: DataProto) -> DataProto:
        n_before = len(batch)
        combined = super()._inject(batch)
        if len(combined) == n_before:
            # No injection happened this step (missing required fields, or
            # no group matched a forget-set game) -- combined IS the
            # original batch, nothing to tag.
            return combined
        # ForgetTrajectoryInjectingCollector._inject always builds
        # `combined = DataProto.concat([real_reduced, extra])`, i.e. the
        # first n_before rows are the untouched real rollout and every row
        # from n_before onward is a synthetic injected row -- tag purely by
        # that position, no need to duplicate/intercept the parent's
        # internals.
        is_injected = np.zeros(len(combined), dtype=bool)
        is_injected[n_before:] = True
        combined.non_tensor_batch["is_forget_injected"] = is_injected
        return combined


class ForgetOnlyBaselineExcludedAdvantageCollectorByGameFile(ForgetOnlyGameFileInjectingCollector):
    """v22, mixed-task-type variant: IDENTICAL to
    ForgetOnlyBaselineExcludedAdvantageCollector (baseline-exclusion +
    forget-advantage-floor tagging via non_tensor_batch["is_forget_injected"]),
    but matched via the exact `game_file` path
    (ForgetOnlyGameFileInjectingCollector's _inject) instead of the
    ambiguous rendered `initial_observation` text
    (ForgetTrajectoryInjectingCollector's _inject).

    Needed because a forget set spanning MULTIPLE task types (e.g. a
    randomly-sampled 100-task mixed set) cannot safely rely on
    initial_observation text matching -- that ambiguity was measured at
    60-68% collision rates for individual non-pick_clean task types alone
    (see load_forget_full_trajectories_by_game_file's docstring), so a
    mixed-type pool would be even more exposed to silent cross-task-type
    collisions. Pair with unlearn.unlearn_inject.load_forget_full_trajectories_by_game_file
    (task_type_filter=None, since the forget jsonl is already pre-filtered
    to the desired subset) instead of load_forget_full_trajectories.
    """

    def _inject(self, batch: DataProto) -> DataProto:
        n_before = len(batch)
        combined = super()._inject(batch)
        if len(combined) == n_before:
            return combined
        is_injected = np.zeros(len(combined), dtype=bool)
        is_injected[n_before:] = True
        combined.non_tensor_batch["is_forget_injected"] = is_injected
        return combined
