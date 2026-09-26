# v6 trajectory-unlearning trainer: off-policy direct suppression.
#
# v5 (unlearn_action_ray_trainer.py) only suppresses pi_theta(a_t|s_t) at
# steps the on-policy rollout happens to visit that exactly match a to-forget
# (state, action) pair -- effectiveness decays as the policy drifts away from
# those states. v6 instead feeds EVERY recorded (prompt, action) pair from
# the to-forget trajectories directly into the current policy as a
# forced-decode target every training step, regardless of what that step's
# on-policy rollout contains, and applies the same unlikelihood loss
# (verl/workers/actor/dp_actor.py::backward_offpolicy_unlearn_loss) to push
# pi_theta(forget_action | forget_state) down.
#
# Part 1 (RL loss) is untouched plain on-policy GRPO/PPO across ALL 6 task
# types (no forget/retain dual-pool split like v3-v5 -- that architecture
# existed specifically to guarantee ON-POLICY exposure to forget-task
# states, which v6 no longer needs since the unlearn loss never depends on
# what the rollout visits).
#
# This subclass overrides _prepare_traj_teacher_signals (like v5) to skip
# all analyzer/OPD/on-policy-matching machinery entirely, and additionally
# builds the off-policy (prompt, action) batch each step and attaches it via
# batch.meta_info["traj_unlearn_offpolicy"] -- dp_actor.py::update_policy
# reads this key directly (mirroring the existing traj_skill_gen payload
# convention) and runs its own separate backward pass for it.

import logging
import random
from typing import Any, Dict, List

import torch

from verl.trainer.ppo.ray_trainer import RayPPOTrainer

module_logger = logging.getLogger(__name__)

ACTION_TEMPLATE = "<action>{action}</action>"


class TrajUnlearnOffPolicyRayPPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        *args,
        forget_step_pairs: List[Dict[str, str]],
        off_policy_batch_size: int = 128,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not forget_step_pairs:
            raise ValueError("forget_step_pairs must be non-empty for v6 off-policy unlearn training.")
        pairs = list(forget_step_pairs)
        random.Random(0).shuffle(pairs)
        self._offpolicy_pairs = pairs
        self._offpolicy_batch_size = max(int(off_policy_batch_size), 1)
        self._offpolicy_cursor = 0
        module_logger.info(
            "Traj-unlearn (off-policy) initialized with %s off-policy (prompt, action) pairs, batch_size=%s "
            "(one full deterministic pass every %s steps).",
            len(self._offpolicy_pairs),
            self._offpolicy_batch_size,
            -(-len(self._offpolicy_pairs) // self._offpolicy_batch_size),
        )

    def _next_offpolicy_batch(self) -> List[Dict[str, str]]:
        n = len(self._offpolicy_pairs)
        batch_size = min(self._offpolicy_batch_size, n)
        batch = []
        for _ in range(batch_size):
            batch.append(self._offpolicy_pairs[self._offpolicy_cursor])
            self._offpolicy_cursor = (self._offpolicy_cursor + 1) % n
        return batch

    def _build_offpolicy_payload(self, batch, pairs: List[Dict[str, str]]) -> Dict[str, Any]:
        prompts = [pair["prompt"] for pair in pairs]
        meta_info = dict(batch.meta_info)
        meta_info.pop("traj_skill_gen_samples", None)
        prompt_batch = self.traj_collector.build_prompt_batch(
            obs_contents=prompts,
            meta_info=meta_info,
        )
        prompt_input_ids = prompt_batch.batch["input_ids"]
        prompt_attention_mask = prompt_batch.batch["attention_mask"]
        prompt_position_ids = prompt_batch.batch["position_ids"]

        action_texts = [ACTION_TEMPLATE.format(action=pair["action"]) for pair in pairs]
        token_id_lists = [
            self.tokenizer(text, add_special_tokens=False)["input_ids"] for text in action_texts
        ]
        max_len = max((len(ids) for ids in token_id_lists), default=1)
        max_len = max(max_len, 1)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0

        responses = torch.full((len(pairs), max_len), pad_id, dtype=prompt_input_ids.dtype)
        response_masks = torch.zeros((len(pairs), max_len), dtype=prompt_attention_mask.dtype)
        for idx, ids in enumerate(token_id_lists):
            length = len(ids)
            if length == 0:
                continue
            responses[idx, :length] = torch.tensor(ids, dtype=prompt_input_ids.dtype)
            response_masks[idx, :length] = 1

        input_ids = torch.cat([prompt_input_ids, responses], dim=-1)
        attention_mask = torch.cat([prompt_attention_mask, response_masks], dim=-1)
        position_ids = self._append_response_position_ids(prompt_position_ids, responses.size(-1))

        return {
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def _prepare_traj_teacher_signals(self, batch, metrics, teacher_enabled):
        # No on-policy analysis/teacher/OPD machinery in v6 at all -- the
        # unlearn signal is entirely off-policy and independent of what this
        # rollout batch contains.
        batch_size = len(batch)
        device = batch.batch["responses"].device
        zero_teacher_log_prob = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
        zero_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        batch.batch["teacher_log_prob"] = zero_teacher_log_prob
        batch.batch["episode_teacher_log_prob"] = zero_teacher_log_prob.clone()
        batch.batch["step_teacher_log_prob"] = zero_teacher_log_prob.clone()
        batch.batch["critical_step_mask"] = zero_mask
        batch.batch["teacher_signal_mask"] = zero_mask.clone()
        batch.batch["step_skill_mask"] = zero_mask.clone()

        pairs = self._next_offpolicy_batch()
        payload = self._build_offpolicy_payload(batch, pairs)
        batch.meta_info["traj_unlearn_offpolicy"] = payload
        metrics["traj/unlearn_offpolicy_batch_size"] = float(len(pairs))
        metrics["traj/unlearn_offpolicy_cursor"] = float(self._offpolicy_cursor)
        metrics["traj/unlearn_offpolicy_total_pairs"] = float(len(self._offpolicy_pairs))
        module_logger.info(
            "Traj-unlearn (off-policy) built an off-policy batch of %s (prompt, forget-action) pairs (cursor=%s/%s).",
            len(pairs),
            self._offpolicy_cursor,
            len(self._offpolicy_pairs),
        )
        return batch

    def _merge_async_traj_teacher_signals(self, batch, teacher_signal_batch):
        batch = super()._merge_async_traj_teacher_signals(batch, teacher_signal_batch)
        if "traj_unlearn_offpolicy" in teacher_signal_batch.meta_info:
            batch.meta_info["traj_unlearn_offpolicy"] = teacher_signal_batch.meta_info["traj_unlearn_offpolicy"]
        return batch
