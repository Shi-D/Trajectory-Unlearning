# v6: off-policy trajectory unlearning for Traj.
#
# v5 (unlearn_action_ray_trainer.py) is on-policy: it only ever suppresses
# pi_theta(a_t|s_t) at steps the CURRENT rollout actually visits and which
# happen to exactly match a recorded to-forget (state, action) pair. If the
# policy has already drifted away from the forget trajectories, those states
# are never revisited and the loss goes quiet (this is by design, but it
# means the loss's own effectiveness decays as training progresses).
#
# v6 instead feeds every recorded (prompt, action) step of the to-forget
# trajectories directly into the model as a forced-decode target -- entirely
# independent of what the on-policy rollout for this training step contains
# -- and computes an unlikelihood loss on the model's own probability of
# producing that recorded action. See
# verl/trainer/ppo/unlearn_offpolicy_ray_trainer.py for how this data is
# turned into a payload, and verl/workers/actor/dp_actor.py::
# backward_offpolicy_unlearn_loss for the actual loss/backward pass.

import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def load_forget_step_pairs(jsonl_path: str) -> List[Dict[str, str]]:
    """Load every (prompt, action) step pair from a
    collection/collect_trajectories.py --all-tasks output file.

    Unlike unlearn/unlearn.py::load_forget_state_action_index (deduped, one
    entry per unique state, used for v5's on-policy state matching), this
    keeps EVERY step of EVERY trajectory as a separate training example: v6
    feeds each pair to the model directly rather than matching it against a
    live rollout, so there is no reason to deduplicate repeated states.

    Returns: [{"prompt": str, "action": str}, ...]
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"forget_trajectories_path not found: {jsonl_path}")

    pairs: List[Dict[str, str]] = []
    num_trajs = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            num_trajs += 1
            traj = json.loads(line)
            for step in traj.get("steps", []):
                prompt = step.get("prompt")
                action = step.get("action")
                if not prompt or action is None:
                    continue
                pairs.append({"prompt": str(prompt), "action": str(action).strip()})

    if not pairs:
        raise ValueError(f"No (prompt, action) pairs found in {jsonl_path}")

    logger.info(
        "Loaded off-policy forget step pairs from %s: %s trajectories, %s (prompt, action) pairs.",
        jsonl_path,
        num_trajs,
        len(pairs),
    )
    return pairs


def load_step_pairs(jsonl_path: str, tag: str = "off-policy") -> List[Dict[str, str]]:
    """Same as load_forget_step_pairs, just with a configurable log tag --
    used by v8 (TrajUnlearnNPORayPPOTrainer) to load BOTH a forget file and a
    retain file through the same loader."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"trajectories path not found: {jsonl_path}")

    pairs: List[Dict[str, str]] = []
    num_trajs = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            num_trajs += 1
            traj = json.loads(line)
            for step in traj.get("steps", []):
                prompt = step.get("prompt")
                action = step.get("action")
                if not prompt or action is None:
                    continue
                pairs.append({"prompt": str(prompt), "action": str(action).strip()})

    if not pairs:
        raise ValueError(f"No (prompt, action) pairs found in {jsonl_path}")

    logger.info(
        "Loaded %s step pairs from %s: %s trajectories, %s (prompt, action) pairs.",
        tag,
        jsonl_path,
        num_trajs,
        len(pairs),
    )
    return pairs
