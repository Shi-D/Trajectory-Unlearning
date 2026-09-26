#!/usr/bin/env python3
"""DPO (Direct Preference Optimization) text unlearning baseline for Traj.

Forget-only baseline (no retain set, per instruction): treats each
(prompt, chosen_response, rejected_response) triplet pulled out of two
INDEPENDENT untrained-model baseline rollouts (see
collection/prepare_dpo_forget_pairs.py) as a preference pair --

    rejected_response = the recorded pick_clean action the model should stop
                         reproducing (from all_trajectories_0.jsonl)
    chosen_response   = the SAME task's action from a different, independent
                         rollout of the same untrained model
                         (from all_trajectories_1.jsonl)

-- and runs the standard DPO loss (Rafailov et al. 2023,
https://arxiv.org/abs/2305.18290) against a frozen reference copy of the
same base model:

    L_DPO = -E[log sigmoid(beta * (
                (logp_pi(y_w|x) - logp_ref(y_w|x))
              - (logp_pi(y_l|x) - logp_ref(y_l|x))
            ))]

where y_w = chosen_response, y_l = rejected_response, and logp is the SUM of
per-token log probs over the response span (standard DPO convention).

No environment interaction, no rollout, no retain-set term -- plain HF
Trainer loop over a parquet file, same infra style as unlearn/npo_unlearn.py.

Known approximation (see prepare_dpo_forget_pairs.py's docstring): x is
taken from the rejected side's recorded prompt. The two rollouts share an
exactly identical prompt at step_idx=0 for every task (both start from the
same game reset), but can diverge from step_idx>=1 onward once the two
independent rollouts pick different actions -- so for later steps, the
chosen_response was not literally elicited by the exact prompt string used
here. This is a documented, accepted approximation, not silently swept
under the rug -- `prompt_exact_match` in the input parquet flags which rows
are exact.
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


class DPOPairDataset(Dataset):
    """Tokenizes (prompt, chosen_response, rejected_response) triplets the
    same way unlearn/npo_unlearn.py::PromptResponseDataset does: prompt goes
    through `apply_chat_template` as a single user turn with
    `add_generation_prompt=True`, response is the model's raw completion
    text + eos, prompt tokens masked out of `labels`. If the combined
    sequence exceeds max_length, only the prompt is truncated (from the
    left) -- both chosen/rejected keep their full response likelihood."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.prompts = df["prompt"].tolist()
        self.chosen = df["chosen_response"].tolist()
        self.rejected = df["rejected_response"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def _build(self, prompt, response):
        tokenizer = self.tokenizer
        prompt_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        response_str = response + tokenizer.eos_token

        prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response_str, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + list(response_ids)

        if len(input_ids) > self.max_length:
            overflow = len(input_ids) - self.max_length
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        return {
            "chosen": self._build(prompt, self.chosen[idx]),
            "rejected": self._build(prompt, self.rejected[idx]),
        }


def pad_batch(features, pad_token_id):
    max_len = max(len(f["input_ids"]) for f in features)
    input_ids, attention_mask, labels = [], [], []
    for f in features:
        pad_len = max_len - len(f["input_ids"])
        input_ids.append(f["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(f["attention_mask"] + [0] * pad_len)
        labels.append(f["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def make_collate_fn(pad_token_id: int):
    def collate_fn(batch):
        return {
            "chosen": pad_batch([b["chosen"] for b in batch], pad_token_id),
            "rejected": pad_batch([b["rejected"] for b in batch], pad_token_id),
        }

    return collate_fn


def sequence_logps(model, inputs):
    """Sum of per-token log probs at label positions (prompt/pad excluded
    via labels == -100), one scalar per batch row -- the standard DPO
    convention (Rafailov et al. 2023 Appendix, and every reference
    implementation e.g. TRL's DPOTrainer)."""
    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    logits = outputs.logits[:, :-1, :]
    labels = inputs["labels"][:, 1:]
    loss_mask = labels != -100
    safe_labels = labels.clamp_min(0)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    per_token_logp = torch.gather(log_probs, dim=2, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    per_token_logp = per_token_logp * loss_mask
    return per_token_logp.sum(dim=-1)


class DPOTrainer(Trainer):
    def __init__(self, *args, ref_model=None, beta: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        assert ref_model is not None, "DPOTrainer requires a frozen ref_model"
        self.ref_model = ref_model.to(self.args.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        chosen_inputs = inputs["chosen"]
        rejected_inputs = inputs["rejected"]

        policy_chosen_logps = sequence_logps(model, chosen_inputs)
        policy_rejected_logps = sequence_logps(model, rejected_inputs)

        with torch.no_grad():
            ref_chosen_logps = sequence_logps(self.ref_model, chosen_inputs)
            ref_rejected_logps = sequence_logps(self.ref_model, rejected_inputs)

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_logratios - ref_logratios

        loss = -F.logsigmoid(self.beta * logits).mean()

        with torch.no_grad():
            chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps)
            rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps)
            reward_acc = (chosen_rewards > rejected_rewards).float().mean()

        self._last_metrics = {
            "dpo_loss": loss.detach().item(),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
            "reward_acc": reward_acc.item(),
        }

        return (loss, None) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if hasattr(self, "_last_metrics"):
            logs.update(self._last_metrics)
        super().log(logs, *args, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="DPO text-unlearning baseline for Traj (forget-only, pick_clean)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_parquet", type=str, required=True, help="Output of collection/prepare_dpo_forget_pairs.py")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature.")
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--num_train_epochs", type=float, default=5.0)
    parser.add_argument("--max_steps", type=int, default=-1, help="Override for smoke tests; -1 uses num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--report_to_wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fsdp", type=str, default="",
        help="Passed straight through to TrainingArguments.fsdp, e.g. 'full_shard auto_wrap'. "
        "Empty (default) keeps plain DDP, matching every existing (3B-scale) run of this script.",
    )
    parser.add_argument(
        "--fsdp_transformer_layer_cls_to_wrap", type=str, default="",
        help="Decoder layer class name to auto-wrap under FSDP, e.g. 'Qwen2DecoderLayer'. Required if --fsdp is set.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    df = pd.read_parquet(args.data_parquet)
    print(f"[dpo_unlearn] loaded {len(df)} (prompt, chosen, rejected) pairs from {args.data_parquet}")
    print(f"[dpo_unlearn] unique task_id: {df['task_id'].nunique()}")
    if "prompt_exact_match" in df.columns:
        print(f"[dpo_unlearn] prompt_exact_match rate: {df['prompt_exact_match'].mean():.3f}")

    train_dataset = DPOPairDataset(df, tokenizer, args.max_length)

    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    policy_model.config.use_cache = False

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    ref_model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        report_to=["wandb"] if args.report_to_wandb else [],
        run_name=args.run_name,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        seed=args.seed,
        fsdp=args.fsdp,
        fsdp_config={"transformer_layer_cls_to_wrap": [args.fsdp_transformer_layer_cls_to_wrap]} if args.fsdp else None,
    )

    trainer = DPOTrainer(
        model=policy_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=make_collate_fn(tokenizer.pad_token_id),
        ref_model=ref_model,
        beta=args.beta,
    )

    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)
        print(f"[dpo_unlearn] final model saved to {final_dir}")


if __name__ == "__main__":
    main()
