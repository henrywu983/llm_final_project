#!/usr/bin/env python
"""Evaluate the base model or a run's checkpoints, saving every sampled answer.

Examples
--------
    # Base model (step 0), needed once per tag for AES:
    python evaluate.py --base --tag quick
    python evaluate.py --base --tag boundary

    # A training run: every checkpoint, cheap accuracy check (n=4, all problems)
    python evaluate.py --run outputs/arm1_grpo_seed0 --tag quick --checkpoints all

    # pass@k up to 256 on selected checkpoints
    python evaluate.py --run outputs/arm1_grpo_seed0 --tag boundary --checkpoints 100,300,final

Output, one folder per checkpoint:
    <run>/eval/<tag>/<checkpoint>/<benchmark>.jsonl.gz       one line per problem, with all samples
    <run>/eval/<tag>/<checkpoint>/<benchmark>_summary.json   pass@k, length, truncation, ...
    (base model: outputs/base_model/eval/<tag>/base/...)

Each sample also stores `nll`, the mean negative log-probability of the tokens it
sampled. Averaged over samples (weighted by tokens), this estimates the policy's
per-token entropy at the evaluation temperature, since E[-log p(token)] over
sampled tokens is the entropy. We use it because training does not log entropy.

Then run analyze.py to compute pass@k curves, AES and per-difficulty tables.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import time
from pathlib import Path

from src.config import load_config
from src.data import build_messages, load_benchmark
from src.grading import grade_many, has_boxed
from src.metrics import k_values, pass_at_k


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument("--run", help="run folder, e.g. outputs/arm1_grpo_seed0")
    who.add_argument("--base", action="store_true", help="evaluate the untrained base model")
    p.add_argument("--config", default=None, help="default: <run>/config.yaml, or configs/base.yaml with --base")
    p.add_argument("--tag", default="quick", help="evaluation mode from the config: quick | boundary")
    p.add_argument("--checkpoints", default="all", help="all | final | comma list of steps, e.g. 50,100,final")
    p.add_argument("--benchmarks", default=None, help="comma list, default: all in config")
    p.add_argument("--n-samples", type=int, default=None, help="override samples per problem")
    p.add_argument("--max-problems", type=int, default=None, help="override problems per benchmark")
    p.add_argument("--gpu-memory-utilization", type=float, default=None)
    p.add_argument("--workers", type=int, default=8, help="CPU processes for grading")
    p.add_argument("--overwrite", action="store_true", help="redo results that already exist")
    p.add_argument("--no-entropy", action="store_true", help="skip the per-token log-probabilities")
    return p.parse_args()


def find_targets(run_dir: Path, spec: str) -> list[tuple[str, str]]:
    """(name, adapter_path) pairs for the requested checkpoints."""
    ckpts = {}
    for d in (run_dir / "checkpoints").glob("checkpoint-*"):
        m = re.fullmatch(r"checkpoint-(\d+)", d.name)
        if m and (d / "adapter_config.json").exists():
            ckpts[int(m.group(1))] = d
    final = run_dir / "final_lora"
    targets = []
    wanted = [s.strip() for s in spec.split(",")]
    if "all" in wanted:
        targets = [(f"checkpoint-{s}", str(p)) for s, p in sorted(ckpts.items())]
        if final.exists():
            targets.append(("final", str(final)))
        return targets
    for w in wanted:
        if w == "final":
            if not final.exists():
                raise FileNotFoundError(f"{final} not found (training not finished?)")
            targets.append(("final", str(final)))
        else:
            step = int(w)
            if step not in ckpts:
                raise FileNotFoundError(f"checkpoint-{step} not found in {run_dir / 'checkpoints'}")
            targets.append((f"checkpoint-{step}", str(ckpts[step])))
    return targets


def main():
    args = parse_args()
    if args.base:
        cfg = load_config(args.config or "configs/base.yaml")
        out_root = Path(cfg["experiment"]["output_root"]) / "base_model" / "eval" / args.tag
        targets = [("base", None)]
    else:
        run_dir = Path(args.run)
        cfg = load_config(args.config or run_dir / "config.yaml")
        out_root = run_dir / "eval" / args.tag
        targets = find_targets(run_dir, args.checkpoints)
        if not targets:
            raise SystemExit(f"No checkpoints found in {run_dir}")

    ecfg = cfg["eval"]
    if args.tag not in ecfg:
        # Runs keep the config they were trained with; take modes added later from base.yaml.
        shared = load_config("configs/base.yaml")["eval"]
        if args.tag not in shared:
            raise SystemExit(f"Unknown --tag {args.tag!r}; define it under eval: in configs/base.yaml")
        ecfg[args.tag] = shared[args.tag]
    mode = ecfg[args.tag]
    n = args.n_samples or mode["n_samples"]
    max_problems = args.max_problems if args.max_problems is not None else mode["max_problems"]
    bench_names = args.benchmarks.split(",") if args.benchmarks else list(ecfg["benchmarks"])
    instruction = cfg["prompt"]["instruction"]

    # Skip work that is already done.
    todo = [(name, path, b) for name, path in targets for b in bench_names
            if args.overwrite or not (out_root / name / f"{b}_summary.json").exists()]
    if not todo:
        print("Everything already evaluated. Use --overwrite to redo.")
        return

    problems = {b: load_benchmark(b, ecfg["benchmarks"][b], max_problems, ecfg["subset_seed"])
                for b in bench_names}
    for b, rows in problems.items():
        print(f"[eval] {b}: {len(rows)} problems x {n} samples")

    # vLLM (imported late so --help is fast). The FlashInfer sampler fails on
    # RTX 50-series cards when the system CUDA toolkit is old; the default sampler is fine.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    use_lora = any(path for _, path, _ in todo)
    llm_kwargs = dict(
        model=ecfg["model_name"], enable_lora=use_lora,
        max_lora_rank=cfg["model"]["lora_rank"] if use_lora else 16,
        max_model_len=1024 + ecfg["max_tokens"], seed=ecfg["seed"],
        gpu_memory_utilization=args.gpu_memory_utilization or ecfg["gpu_memory_utilization"],
    )
    want_lp = not args.no_entropy
    logprobs_mode = None
    if want_lp:
        # "processed" = log-probs after temperature/top-p, i.e. of the distribution we
        # actually sample from. Older vLLM versions only return raw log-probs.
        try:
            llm = LLM(**llm_kwargs, logprobs_mode="processed_logprobs")
            logprobs_mode = "processed_logprobs"
        except TypeError:
            llm = LLM(**llm_kwargs)
            logprobs_mode = "raw_logprobs"
            print("[eval] this vLLM has no logprobs_mode; nll uses the raw (temperature 1) model "
                  "log-probs, so it is a cross-entropy rather than an exact entropy estimate")
    else:
        llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    prompts = {b: [tokenizer.apply_chat_template(build_messages(r["question"], instruction),
                                                 tokenize=False, add_generation_prompt=True)
                   for r in rows] for b, rows in problems.items()}
    sampling = SamplingParams(n=n, temperature=ecfg["temperature"], top_p=ecfg["top_p"],
                              max_tokens=ecfg["max_tokens"], seed=ecfg["seed"],
                              logprobs=0 if want_lp else None)

    lora_ids = {}
    for name, path, bench in todo:
        lora = None
        if path:
            lora_ids.setdefault(name, len(lora_ids) + 1)
            lora = LoRARequest(name, lora_ids[name], path)
        t0 = time.time()
        outputs = llm.generate(prompts[bench], sampling, lora_request=lora)
        gen_s = time.time() - t0

        rows = problems[bench]
        texts, golds = [], []
        for r, out in zip(rows, outputs):
            for o in out.outputs:
                texts.append(o.text)
                golds.append(r["answer"])
        correct = grade_many(texts, golds, workers=args.workers)

        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        records, idx = [], 0
        with gzip.open(out_dir / f"{bench}.jsonl.gz", "wt", encoding="utf-8") as f:
            for r, out in zip(rows, outputs):
                samples = []
                for o in out.outputs:
                    samples.append(dict(text=o.text, n_tokens=len(o.token_ids),
                                        finish_reason=o.finish_reason, correct=correct[idx],
                                        boxed=has_boxed(o.text), nll=sample_nll(o)))
                    idx += 1
                rec = dict(question_id=r["question_id"], level=r["level"], answer=r["answer"],
                           n=len(samples), n_correct=sum(s["correct"] for s in samples), samples=samples)
                records.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        summary = summarize(records)
        summary.update(logprobs_mode=logprobs_mode, checkpoint=name, benchmark=bench, tag=args.tag, adapter=path,
                       model=ecfg["model_name"], temperature=ecfg["temperature"], top_p=ecfg["top_p"],
                       max_tokens=ecfg["max_tokens"], generation_seconds=round(gen_s, 1))
        (out_dir / f"{bench}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        k_max = summary["n_samples"]
        ent = f"  entropy={summary['entropy']:.3f}" if summary.get("entropy") is not None else ""
        print(f"[eval] {name:>16s} {bench:>8s}  pass@1={summary['pass@1']:.3f}  "
              f"pass@{k_max}={summary['pass@' + str(k_max)]:.3f}  "
              f"tokens={summary['mean_tokens']:.0f}  truncated={summary['truncated_rate']:.3f}{ent}  "
              f"({gen_s / 60:.1f} min)")


def sample_nll(o) -> float | None:
    """Mean -log p of the tokens this sample generated (None if log-probs weren't requested)."""
    if not o.token_ids:
        return None
    total = None
    if o.logprobs:
        total = 0.0
        for tok, lp in zip(o.token_ids, o.logprobs):
            entry = lp.get(tok) if lp else None
            if entry is None:
                total = None
                break
            total += entry.logprob
    if total is None and getattr(o, "cumulative_logprob", None) is not None:
        total = o.cumulative_logprob
    return None if total is None else -total / len(o.token_ids)


def summarize(records: list[dict]) -> dict:
    n = min(r["n"] for r in records)
    all_samples = [s for r in records for s in r["samples"]]
    correct_tokens = [s["n_tokens"] for s in all_samples if s["correct"]]
    out = dict(
        n_problems=len(records), n_samples=n,
        mean_tokens=sum(s["n_tokens"] for s in all_samples) / len(all_samples),
        mean_tokens_correct=sum(correct_tokens) / len(correct_tokens) if correct_tokens else None,
        truncated_rate=sum(s["finish_reason"] == "length" for s in all_samples) / len(all_samples),
        boxed_rate=sum(s["boxed"] for s in all_samples) / len(all_samples),
        entropy=token_entropy(all_samples),
    )
    for k in k_values(n):
        out[f"pass@{k}"] = sum(pass_at_k(r["n"], r["n_correct"], k) for r in records) / len(records)
    return out


def token_entropy(samples: list[dict]) -> float | None:
    """Token-weighted mean of per-sample nll: an estimate of per-token policy entropy."""
    have = [s for s in samples if s.get("nll") is not None and s["n_tokens"] > 0]
    if not have:
        return None
    return sum(s["nll"] * s["n_tokens"] for s in have) / sum(s["n_tokens"] for s in have)


if __name__ == "__main__":
    main()
