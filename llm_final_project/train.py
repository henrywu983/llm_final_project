#!/usr/bin/env python
"""Train one GRPO arm.

Examples
--------
    python train.py --config configs/arm1_grpo.yaml --seed 0
    python train.py --config configs/arm2_dr_grpo.yaml --seed 0 --set machine.gpu_memory_utilization=0.35
    python train.py --config configs/arm3_length_penalty.yaml --seed 0 --dry-run   # check setup, no GPU
    python train.py --config configs/arm1_grpo.yaml --seed 0 --resume             # continue a stopped run
    python train.py --config configs/arm1_grpo.yaml --suffix _test --set train.max_steps=5 --set train.save_steps=5

Everything for the run is saved in outputs/<arm>_seed<seed>/:
    config.yaml, run.json, pip_freeze.txt, nvidia_smi.txt, git_diff.patch   run record
    train_manifest.jsonl.gz      every training problem, in order
    trainer_config.json          the exact TRL settings used
    metrics.jsonl / metrics.csv  every logged metric, every step
    rollout_stats.csv            per-batch answer stats (accuracy, lengths, truncation, ...)
    rollouts.jsonl.gz            every answer the model wrote during training
    checkpoints/checkpoint-N/    LoRA adapter every save_steps, all kept
    final_lora/                  the final adapter
    train.log                    full terminal output
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

from src.config import apply_overrides, load_config
from src.data import build_train_rows, to_trl_dataset
from src.run_record import RunRecord, start_terminal_log


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="arm config, e.g. configs/arm1_grpo.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. machine.gpu_memory_utilization=0.4")
    p.add_argument("--suffix", default="", help="added to the run folder name, e.g. _test for a quick check")
    p.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    p.add_argument("--dry-run", action="store_true", help="load data and build settings, then stop")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    experiment_overrides = apply_overrides(cfg, args.set)
    cfg["experiment"]["seed"] = args.seed
    run_name = f"{cfg['experiment']['name']}_seed{args.seed}{args.suffix}"
    root = Path(cfg["experiment"]["output_root"])
    run_dir = root / "_dry_run" / run_name if args.dry_run else root / run_name

    if run_dir.exists() and any(run_dir.iterdir()) and not (args.resume or args.dry_run):
        sys.exit(f"{run_dir} already exists. Use --resume to continue it, or delete it to start over.")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = start_terminal_log(run_dir / ("dry_run.log" if args.dry_run else "train.log"))

    if experiment_overrides:
        print(f"[warning] these overrides change the experiment and break comparability "
              f"with other arms: {experiment_overrides}")

    record = RunRecord(run_dir, cfg, sys.argv, experiment_overrides)
    if not args.dry_run:
        record.start()
    status = "failed"
    try:
        # ---- data ---------------------------------------------------------
        rows = build_train_rows(cfg)
        with gzip.open(run_dir / "train_manifest.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: r[k] for k in ("question_id", "source", "level", "answer")}) + "\n")
        dataset = to_trl_dataset(rows, cfg["prompt"]["instruction"])
        by_source = {}
        for r in rows:
            by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        print(f"[data] {len(rows)} training problems: {by_source}")

        if args.dry_run:
            from trl import GRPOConfig

            from src.trl_setup import build_grpo_config

            grpo_args = build_grpo_config(GRPOConfig, cfg, str(run_dir / "checkpoints"), run_name, args.seed)
            _print_method(grpo_args, cfg)
            print("[dry-run] data and settings OK. Nothing was trained.")
            status = "dry_run"
            return

        # ---- model (Unsloth must be imported before TRL / transformers) ---
        from unsloth import FastLanguageModel  # noqa: I001
        from trl import GRPOConfig, GRPOTrainer

        from src.callbacks import MetricsCallback, write_metrics_csv
        from src.rewards import RewardEngine
        from src.trl_setup import build_grpo_config

        if cfg["logging"]["report_to"] == "wandb":
            os.environ.setdefault("WANDB_PROJECT", cfg["logging"]["wandb_project"])

        m = cfg["model"]
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=m["name"], max_seq_length=m["max_seq_length"], load_in_4bit=m["load_in_4bit"],
            fast_inference=True, max_lora_rank=m["lora_rank"],
            gpu_memory_utilization=cfg["machine"]["gpu_memory_utilization"],
        )
        model = FastLanguageModel.get_peft_model(
            model, r=m["lora_rank"], lora_alpha=m["lora_alpha"], target_modules=m["target_modules"],
            use_gradient_checkpointing="unsloth", random_state=args.seed,
        )

        grpo_args = build_grpo_config(GRPOConfig, cfg, str(run_dir / "checkpoints"), run_name, args.seed)
        _print_method(grpo_args, cfg)
        (run_dir / "trainer_config.json").write_text(
            json.dumps(grpo_args.to_dict(), indent=2, default=str), encoding="utf-8")

        engine = RewardEngine(cfg, tokenizer, run_dir)
        trainer = GRPOTrainer(
            model=model, processing_class=tokenizer, reward_funcs=engine.reward_functions(),
            args=grpo_args, train_dataset=dataset, callbacks=[MetricsCallback(run_dir)],
        )
        engine.trainer = trainer

        # Unsloth patches TRL's trainer; make sure it runs the loss we configured.
        actual = getattr(trainer, "loss_type", grpo_args.loss_type)
        if actual != cfg["grpo"]["loss_type"]:
            raise RuntimeError(f"Trainer uses loss_type={actual!r}, config says {cfg['grpo']['loss_type']!r}")

        trainer.train(resume_from_checkpoint=True if args.resume else None)

        final_dir = run_dir / "final_lora"
        if hasattr(model, "save_lora"):
            model.save_lora(str(final_dir))
        else:
            model.save_pretrained(str(final_dir))
        write_metrics_csv(run_dir / "metrics.jsonl", run_dir / "metrics.csv")
        print(f"[done] run saved in {run_dir}")
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        if not args.dry_run:
            record.finish(status)
        # Stop copying output to the log before closing it, so messages printed
        # while Python shuts down don't try to write to a closed file.
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        log_file.close()


def _penalty_desc(lp):
    from src.length_penalty import LENGTH_PENALTIES, extra_params

    params = dict(coef=lp["coef"], **extra_params(LENGTH_PENALTIES[lp["type"]], lp))
    return f"{lp['type']}(" + ", ".join(f"{k}={v}" for k, v in params.items()) + ")"


def _print_method(grpo_args, cfg):
    lp = cfg["reward"]["length_penalty"]
    print("[method] loss_type=%s scale_rewards=%s beta=%s epsilon=%s G=%s max_len=%s | length_penalty=%s"
          % (grpo_args.loss_type, grpo_args.scale_rewards, grpo_args.beta, grpo_args.epsilon,
             grpo_args.num_generations, grpo_args.max_completion_length,
             _penalty_desc(lp) if lp["enabled"] else "off"))
    print("[batch] %d questions x %d answers per step = %d answers; batch %d x grad_accum %d; %d steps"
          % (cfg["grpo"]["prompts_per_step"], grpo_args.num_generations,
             cfg["grpo"]["prompts_per_step"] * grpo_args.num_generations,
             grpo_args.per_device_train_batch_size, grpo_args.gradient_accumulation_steps,
             grpo_args.max_steps))


if __name__ == "__main__":
    main()
