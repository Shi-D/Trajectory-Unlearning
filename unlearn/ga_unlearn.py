#!/usr/bin/env python3
"""GA (plain Gradient Ascent) text unlearning baseline for Traj.

Same data pipeline and base model as unlearn/npo_unlearn.py (reuses its dataset
classes directly) -- the only thing that differs is the loss: instead of NPO's
bounded log-sigmoid preference loss against a frozen reference model, GA just
ascends the forget-set cross-entropy directly (no reference model needed at
all) and regularizes with a plain SFT loss on the retain set:

    loss = -CE(forget) + gamma * CE(retain)

This is a well-known unstable baseline (unclamped ascent on cross-entropy is
unbounded and can blow up to NaN/Inf within a small number of steps) -- that
instability is the point of comparing it to NPO, not a bug to engineer away.
If there is no retain set (the "all task types are forget" setting), the
retain term is simply dropped (gamma has no effect).
"""

import argparse
import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

import pandas as pd

from unlearn.npo_unlearn import ForgetRetainDataset, PromptResponseDataset, make_collate_fn


class GATrainer(Trainer):
    def __init__(self, *args, gamma: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget_inputs = inputs["forget"]
        forget_outputs = model(
            input_ids=forget_inputs["input_ids"],
            attention_mask=forget_inputs["attention_mask"],
            labels=forget_inputs["labels"],
        )

        if "retain" in inputs and self.gamma > 0:
            retain_inputs = inputs["retain"]
            retain_outputs = model(
                input_ids=retain_inputs["input_ids"],
                attention_mask=retain_inputs["attention_mask"],
                labels=retain_inputs["labels"],
            )
            retain_loss = retain_outputs.loss
        else:
            retain_loss = torch.zeros((), device=forget_outputs.loss.device)

        loss = -forget_outputs.loss + self.gamma * retain_loss

        self._last_metrics = {
            "forget_ce": forget_outputs.loss.detach().item(),
            "retain_loss": retain_loss.detach().item(),
        }

        return (loss, forget_outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if hasattr(self, "_last_metrics"):
            logs.update(self._last_metrics)
        super().log(logs, *args, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="GA (gradient ascent) text-unlearning baseline for Traj")
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
        help="Comma-separated task_type list to retain, 'auto' = complement of forget_task_types, "
        "'none' = no retain set (pure gradient ascent, gamma is ignored).",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gamma", type=float, default=1.0, help="Retain-loss coefficient.")
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
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

    retain_arg = args.retain_task_types.strip().lower()
    if retain_arg == "none":
        retain_task_types = []
    elif retain_arg == "auto":
        retain_task_types = [t for t in all_task_types if t not in forget_task_types]
    else:
        retain_task_types = [t.strip() for t in args.retain_task_types.split(",") if t.strip()]

    forget_df = df[df["task_type"].isin(forget_task_types)].reset_index(drop=True)
    retain_df = df[df["task_type"].isin(retain_task_types)].reset_index(drop=True) if retain_task_types else None

    print(f"[ga_unlearn] forget task types: {forget_task_types} ({len(forget_df)} samples)")
    if retain_df is not None:
        print(f"[ga_unlearn] retain task types: {retain_task_types} ({len(retain_df)} samples)")
    else:
        print("[ga_unlearn] no retain set configured -- pure gradient ascent (gamma is ignored)")

    forget_dataset = PromptResponseDataset(forget_df, tokenizer, args.max_length)
    retain_dataset = PromptResponseDataset(retain_df, tokenizer, args.max_length) if retain_df is not None else None
    train_dataset = ForgetRetainDataset(forget_dataset, retain_dataset)

    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    policy_model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
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

    trainer = GATrainer(
        model=policy_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=make_collate_fn(tokenizer.pad_token_id),
        gamma=args.gamma,
    )

    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)
        print(f"[ga_unlearn] final model saved to {final_dir}")


if __name__ == "__main__":
    main()
