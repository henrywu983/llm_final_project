"""Translate our config into a TRL GRPOConfig, safely across TRL versions.

Settings that define the method (loss type, reward scaling, KL, group size,
completion length) are CRITICAL: if the installed TRL does not support one,
we stop instead of silently training a different algorithm.
"""

from __future__ import annotations

import dataclasses

CRITICAL = {"loss_type", "scale_rewards", "beta", "num_generations", "max_completion_length",
            "epsilon", "per_device_train_batch_size", "gradient_accumulation_steps"}


def batch_sizes(cfg: dict) -> tuple[int, int]:
    """Return (per_device_train_batch_size, gradient_accumulation_steps)."""
    g, t = cfg["grpo"], cfg["train"]
    completions = g["prompts_per_step"] * g["num_generations"]
    bs = t["per_device_train_batch_size"]
    if completions % bs:
        raise ValueError(f"prompts_per_step * num_generations ({completions}) must be divisible "
                         f"by per_device_train_batch_size ({bs})")
    return bs, completions // bs


def build_grpo_config(GRPOConfig, cfg: dict, output_dir: str, run_name: str, seed: int):
    g, t, log = cfg["grpo"], cfg["train"], cfg["logging"]
    bs, grad_acc = batch_sizes(cfg)
    fields = {f.name: f for f in dataclasses.fields(GRPOConfig)}

    # scale_rewards was a bool in older TRL and a string ("group"/"batch"/"none") in newer TRL.
    scale = g["scale_rewards"]
    default = fields["scale_rewards"].default if "scale_rewards" in fields else None
    if isinstance(default, bool):
        if scale == "batch":
            raise RuntimeError("This TRL version only supports group or no reward scaling.")
        scale = scale != "none"

    wanted = dict(
        output_dir=output_dir, run_name=run_name, seed=seed, data_seed=seed,
        learning_rate=t["learning_rate"], lr_scheduler_type=t["lr_scheduler_type"],
        warmup_steps=t["warmup_steps"], max_steps=t["max_steps"], max_grad_norm=t["max_grad_norm"],
        weight_decay=t["weight_decay"], optim=t["optim"],
        per_device_train_batch_size=bs, gradient_accumulation_steps=grad_acc,
        steps_per_generation=grad_acc,  # one generation round per optimizer step
        num_generations=g["num_generations"], max_completion_length=g["max_completion_length"],
        temperature=g["temperature"], top_p=g["top_p"],
        loss_type=g["loss_type"], scale_rewards=scale, beta=g["beta"], epsilon=g["epsilon"],
        mask_truncated_completions=g["mask_truncated_completions"],
        logging_steps=t["logging_steps"], save_strategy="steps", save_steps=t["save_steps"],
        save_total_limit=None,  # keep every checkpoint
        report_to=log["report_to"], log_completions=False, remove_unused_columns=False,
        num_iterations=1,
    )
    if g.get("epsilon_high") is not None:
        wanted["epsilon_high"] = g["epsilon_high"]

    kwargs, skipped = {}, []
    for key, value in wanted.items():
        if key in fields:
            kwargs[key] = value
        elif key in CRITICAL:
            raise RuntimeError(f"Installed TRL has no GRPOConfig option {key!r}, which defines the "
                               f"method. Use the TRL version in requirements.txt.")
        else:
            skipped.append(key)
    if skipped:
        print(f"[trl] option(s) not in this TRL version, skipped: {skipped}")
    return GRPOConfig(**kwargs)
