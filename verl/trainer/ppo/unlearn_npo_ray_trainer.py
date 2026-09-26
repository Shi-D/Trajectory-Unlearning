# v8 trajectory-unlearning trainer: GRPO + NPO.
#
# Motivation: the standalone text-only NPO baseline (unlearn/npo_unlearn.py)
# forgets the recorded forget-set trajectories very cleanly (similarity to
# baseline drops hard), but since it is pure SFT-style training with no
# reward signal at all, the model's TASK SUCCESS RATE on the forget task type
# also collapses -- there is nothing in that training loop pushing the model
# back towards *any* successful trajectory, only away from the recorded one.
#
# v8 fixes this by running NPO as an auxiliary loss ON TOP OF plain on-policy
# GRPO (exactly like v6's off-policy loss is layered on top of GRPO): GRPO's
# on-policy rollout already covers all 6 task types (including the forget
# type) every step, and its reward signal keeps pulling the policy towards
# *some* successful trajectory -- while the NPO term specifically pushes the
# policy's likelihood of reproducing the exact recorded (memorized) forget
# trajectory down relative to a frozen reference copy of the model. The goal
# is: forget the specific memorized path, but keep succeeding via a different
# one.
#
# Concretely this is v6's off-policy machinery
# (unlearn_offpolicy_ray_trainer.py / unlearn/unlearn_offpolicy.py) extended
# two ways:
#   1. A SECOND off-policy (prompt, action) pool is cycled independently for
#      the RETAIN task types (the unlearn/npo_unlearn.py "gamma" idea, but here
#      gamma weights a plain CE loss on the retain pool instead of a
#      full-corpus SFT epoch).
#   2. The forget batch's reference log-probs are computed once per step via
#      the EXISTING frozen reference-policy worker group (self.ref_policy_wg
#      -- already instantiated by every Traj-unlearn launcher through
#      actor_rollout_ref.actor.use_kl_loss=True in
#      examples/traj_trainer/_common/alfworld.sh) and threaded down to
#      dp_actor.py::backward_offpolicy_npo_loss via the payload dict, so no
#      new frozen model copy needs to be loaded anywhere.
#
# Like v6, this uses the SAME (prompt, action)-only convention (not the full
# <think>...</think><action>...</action> reasoning text used by the
# standalone text NPO baseline) -- consistent with how the rest of the
# off-policy unlearn infra (dp_actor.py's _forward_micro_batch on
# `responses`) already operates on action tokens only.

import logging
import random
from typing import Any, Dict, List

from verl.protocol import DataProto
from verl.trainer.ppo.unlearn_offpolicy_ray_trainer import TrajUnlearnOffPolicyRayPPOTrainer

module_logger = logging.getLogger(__name__)


class TrajUnlearnNPORayPPOTrainer(TrajUnlearnOffPolicyRayPPOTrainer):
    def __init__(
        self,
        *args,
        retain_step_pairs: List[Dict[str, str]],
        retain_batch_size: int = 128,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not retain_step_pairs:
            raise ValueError("retain_step_pairs must be non-empty for v8 GRPO+NPO training.")
        if not self.use_reference_policy:
            raise ValueError(
                "v8 (TrajUnlearnNPORayPPOTrainer) requires a frozen reference policy "
                "(actor_rollout_ref.actor.use_kl_loss=True) to compute the NPO forget "
                "loss's pi_ref(y|x) term."
            )
        pairs = list(retain_step_pairs)
        random.Random(1).shuffle(pairs)
        self._retain_pairs = pairs
        self._retain_batch_size = max(int(retain_batch_size), 1)
        self._retain_cursor = 0
        module_logger.info(
            "NPO+GRPO initialized with %s retain (prompt, action) pairs, batch_size=%s.",
            len(self._retain_pairs),
            self._retain_batch_size,
        )

    def _next_retain_batch(self) -> List[Dict[str, str]]:
        n = len(self._retain_pairs)
        batch_size = min(self._retain_batch_size, n)
        batch = []
        for _ in range(batch_size):
            batch.append(self._retain_pairs[self._retain_cursor])
            self._retain_cursor = (self._retain_cursor + 1) % n
        return batch

    def _prepare_traj_teacher_signals(self, batch, metrics, teacher_enabled):
        # Builds the forget payload (batch.meta_info["traj_unlearn_offpolicy"]
        # with responses/input_ids/attention_mask/position_ids) exactly like
        # v6, plus zeroed teacher-signal tensors.
        batch = super()._prepare_traj_teacher_signals(batch, metrics, teacher_enabled)

        forget_payload = batch.meta_info["traj_unlearn_offpolicy"]
        forget_proto = DataProto.from_dict(
            tensors={
                "responses": forget_payload["responses"],
                "input_ids": forget_payload["input_ids"],
                "attention_mask": forget_payload["attention_mask"],
                "position_ids": forget_payload["position_ids"],
            }
        )
        if not self.ref_in_actor:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(forget_proto)
        else:
            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(forget_proto)
        forget_payload["ref_log_prob"] = ref_log_prob.batch["ref_log_prob"]

        retain_pairs = self._next_retain_batch()
        retain_payload = self._build_offpolicy_payload(batch, retain_pairs)
        forget_payload["retain_responses"] = retain_payload["responses"]
        forget_payload["retain_input_ids"] = retain_payload["input_ids"]
        forget_payload["retain_attention_mask"] = retain_payload["attention_mask"]
        forget_payload["retain_position_ids"] = retain_payload["position_ids"]

        batch.meta_info["traj_unlearn_offpolicy"] = forget_payload
        metrics["traj/unlearn_npo_retain_batch_size"] = float(len(retain_pairs))
        metrics["traj/unlearn_npo_retain_cursor"] = float(self._retain_cursor)
        metrics["traj/unlearn_npo_retain_total_pairs"] = float(len(self._retain_pairs))
        return batch

    def _merge_async_traj_teacher_signals(self, batch, teacher_signal_batch):
        batch = super()._merge_async_traj_teacher_signals(batch, teacher_signal_batch)
        payload = teacher_signal_batch.meta_info.get("traj_unlearn_offpolicy")
        if isinstance(payload, dict) and "ref_log_prob" in payload:
            batch.meta_info["traj_unlearn_offpolicy"] = payload
        return batch
