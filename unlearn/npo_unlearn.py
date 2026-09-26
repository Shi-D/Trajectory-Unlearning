#!/usr/bin/env python3
"""NPO (Negative Preference Optimization) text unlearning for Traj.

Unlike v1-v7 (agentic/on-policy trajectory unlearning: needs the live ALFWorld
env + vLLM rollout + verl/ray infra), NPO here is a pure text-unlearning
baseline. It treats each (prompt, model_response) step pulled out of
collected trajectories (see collection/prepare_npo_forget_data.py) as an
ordinary (x, y) supervised pair, and penalizes the policy's likelihood of
reproducing y given x relative to a frozen reference copy of the same model.
No environment interaction, no rollout -- just a plain HF Trainer loop over a
parquet file.

Loss (https://arxiv.org/abs/2404.05868):
    L_NPO = -(2/beta) * E_{(x,y)~D_f}[log sigmoid(-beta * log(pi_theta(y|x)/pi_ref(y|x)))]
optionally combined with a retain-set SFT regularizer:
    L = L_NPO + gamma * L_retain_sft

`current_forget_loss`/`ref_forget_loss` below are each a single batch-mean
cross-entropy (the model's own `outputs.loss`), not a per-example log-ratio --
this mirrors the reference NPO trainer implementation (TOFU-style) this
module was built from.
"""

import argparse
import os
import random

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


class PromptResponseDataset(Dataset):
    """Tokenizes (prompt, response) pairs exactly the way the live rollout
    builds them: prompt goes through `apply_chat_template` as a single user
    turn with `add_generation_prompt=True` (matching
    agent_system/multi_turn_rollout/rollout_loop.py), response is the model's
    raw completion text + eos. Prompt tokens are masked out of `labels`; if
    the combined sequence still exceeds max_length, only the PROMPT is
    truncated (from the left) -- the response is never touched, since its
    full likelihood is exactly what NPO/SFT need."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.prompts = df["prompt"].tolist()
        self.responses = df["response"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        tokenizer = self.tokenizer
        prompt_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": self.prompts[idx]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        response_str = self.responses[idx] + tokenizer.eos_token

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


class ForgetRetainDataset(Dataset):
    """Pairs each forget example with a randomly resampled retain example so
    every training step sees both halves in one batch. If `retain_dataset`
    is None (the "all task types are forget" setting has no retain pool
    left), each item carries only the forget half and NPOTrainer skips the
    retain loss term entirely."""

    def __init__(self, forget_dataset: PromptResponseDataset, retain_dataset: PromptResponseDataset = None):
        self.forget_dataset = forget_dataset
        self.retain_dataset = retain_dataset

    def __len__(self):
        return len(self.forget_dataset)

    def __getitem__(self, idx):
        item = {"forget": self.forget_dataset[idx]}
        if self.retain_dataset is not None:
            retain_idx = random.randrange(len(self.retain_dataset))
            item["retain"] = self.retain_dataset[retain_idx]
        return item


def make_collate_fn(pad_token_id: int):
    def collate_fn(batch):
        out = {"forget": pad_batch([b["forget"] for b in batch], pad_token_id)}
        if "retain" in batch[0]:
            out["retain"] = pad_batch([b["retain"] for b in batch], pad_token_id)
        return out

    return collate_fn


class NPOTrainer(Trainer):
    def __init__(self, *args, ref_model=None, beta: float = 0.1, gamma: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        assert ref_model is not None, "NPOTrainer requires a frozen ref_model"
        self.ref_model = ref_model.to(self.args.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.beta = beta
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget_inputs = inputs["forget"]

        outputs = model(
            input_ids=forget_inputs["input_ids"],
            attention_mask=forget_inputs["attention_mask"],
            labels=forget_inputs["labels"],
        )
        current_forget_loss = outputs.loss

        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=forget_inputs["input_ids"],
                attention_mask=forget_inputs["attention_mask"],
                labels=forget_inputs["labels"],
            )
            ref_forget_loss = ref_outputs.loss

        neg_log_ratios = current_forget_loss - ref_forget_loss
        forget_loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta

        if "retain" in inputs and self.gamma > 0:
            retain_inputs = inputs["retain"]
            retain_outputs = model(
                input_ids=retain_inputs["input_ids"],
                attention_mask=retain_inputs["attention_mask"],
                labels=retain_inputs["labels"],
            )
            retain_loss = retain_outputs.loss
        else:
            retain_loss = torch.zeros((), device=forget_loss.device)

        loss = forget_loss + self.gamma * retain_loss

        self._last_metrics = {
            "forget_loss": forget_loss.detach().item(),
            "retain_loss": retain_loss.detach().item(),
            "current_forget_ce": current_forget_loss.detach().item(),
            "ref_forget_ce": ref_forget_loss.detach().item(),
        }

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if hasattr(self, "_last_metrics"):
            logs.update(self._last_metrics)
        super().log(logs, *args, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="NPO text-unlearning baseline for Traj")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_parquet", type=str, required=True)
    parser.add_argument(
        "--forget_task_types",
        type=str,
        required=True,
        help="Comma-separated task_type list to forget, or 'all' for every task type present in data_parquet.",
    )
    parser.add_argument(
        "--retain_task_types",
        type=str,
        default="auto",
        help="Comma-separated task_type list to retain, 'auto' = complement of forget_task_types within the "
        "retain source (data_parquet, or retain_data_parquet if set), "
        "'none' = no retain set (pure forget-only NPO, gamma is ignored).",
    )
    parser.add_argument(
        "--retain_data_parquet",
        type=str,
        default=None,
        help="If set, retain samples are loaded from this parquet instead of --data_parquet -- e.g. a "
        "DIFFERENT independent rollout's trajectories for the SAME task_type as the forget set (rather "
        "than the usual 'other task types from the same rollout' retain set).",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--beta", type=float, default=0.1, help="NPO temperature (beta in the loss formula).")
    parser.add_argument("--gamma", type=float, default=1.0, help="Retain-loss coefficient.")
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
    all_task_types = sorted(df["task_type"].unique().tolist())

    if args.forget_task_types.strip().lower() == "all":
        forget_task_types = all_task_types
    else:
        forget_task_types = [t.strip() for t in args.forget_task_types.split(",") if t.strip()]
    unknown = set(forget_task_types) - set(all_task_types)
    assert not unknown, f"Unknown forget_task_types not present in {args.data_parquet}: {unknown}"

    if args.retain_data_parquet:
        retain_source_df = pd.read_parquet(args.retain_data_parquet)
        retain_source_label = args.retain_data_parquet
    else:
        retain_source_df = df
        retain_source_label = args.data_parquet
    retain_all_task_types = sorted(retain_source_df["task_type"].unique().tolist())

    retain_arg = args.retain_task_types.strip().lower()
    if retain_arg == "none":
        retain_task_types = []
    elif retain_arg == "auto":
        retain_task_types = [t for t in retain_all_task_types if t not in forget_task_types]
    else:
        retain_task_types = [t.strip() for t in args.retain_task_types.split(",") if t.strip()]
    unknown_retain = set(retain_task_types) - set(retain_all_task_types)
    assert not unknown_retain, f"Unknown retain_task_types not present in {retain_source_label}: {unknown_retain}"

    forget_df = df[df["task_type"].isin(forget_task_types)].reset_index(drop=True)
    retain_df = (
        retain_source_df[retain_source_df["task_type"].isin(retain_task_types)].reset_index(drop=True)
        if retain_task_types
        else None
    )

    print(f"[npo_unlearn] forget task types: {forget_task_types} ({len(forget_df)} samples) from {args.data_parquet}")
    if retain_df is not None:
        print(f"[npo_unlearn] retain task types: {retain_task_types} ({len(retain_df)} samples) from {retain_source_label}")
    else:
        print("[npo_unlearn] no retain set configured -- pure forget-only NPO (gamma is ignored)")

    forget_dataset = PromptResponseDataset(forget_df, tokenizer, args.max_length)
    retain_dataset = PromptResponseDataset(retain_df, tokenizer, args.max_length) if retain_df is not None else None
    train_dataset = ForgetRetainDataset(forget_dataset, retain_dataset)

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

    trainer = NPOTrainer(
        model=policy_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=make_collate_fn(tokenizer.pad_token_id),
        ref_model=ref_model,
        beta=args.beta,
        gamma=args.gamma,
    )

    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)
        print(f"[npo_unlearn] final model saved to {final_dir}")


if __name__ == "__main__":
    main()
