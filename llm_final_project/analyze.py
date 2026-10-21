#!/usr/bin/env python
"""Turn saved evaluation samples into result tables (and optional plots).

    python analyze.py --tag boundary
    python analyze.py --tag quick --plots

Reads   outputs/base_model/eval/<tag>/base/*.jsonl.gz
        outputs/<run>/eval/<tag>/<checkpoint>/*.jsonl.gz
Writes  results/<tag>/summary.csv          one row per run x checkpoint x benchmark
        results/<tag>/summary_by_arm.csv   mean and std over seeds
        results/<tag>/by_level.csv         MATH500 results per difficulty level
        results/<tag>/*.png                with --plots

Metrics: pass@k for k = 1, 2, 4, ..., n (unbiased estimator), mean response
tokens (all / correct / incorrect), truncation rate, boxed rate, and AES
against the base model on the same benchmark and tag.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import pandas as pd
import yaml

from src.metrics import aes, k_values, pass_at_k


def token_entropy(samples):
    have = [s for s in samples if s.get("nll") is not None and s["n_tokens"] > 0]
    if not have:
        return None
    return sum(s["nll"] * s["n_tokens"] for s in have) / sum(s["n_tokens"] for s in have)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="outputs")
    p.add_argument("--tag", default="boundary")
    p.add_argument("--out", default=None, help="default: results/<tag>")
    p.add_argument("--plots", action="store_true", help="also save PNG figures")
    return p.parse_args()


def load_records(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def metrics_for(records: list[dict]) -> dict:
    n = min(r["n"] for r in records)
    samples = [s for r in records for s in r["samples"]]
    tok = [s["n_tokens"] for s in samples]
    tok_c = [s["n_tokens"] for s in samples if s["correct"]]
    tok_w = [s["n_tokens"] for s in samples if not s["correct"]]
    row = dict(
        n_problems=len(records), n_samples=n,
        mean_tokens=sum(tok) / len(tok),
        mean_tokens_correct=sum(tok_c) / len(tok_c) if tok_c else None,
        mean_tokens_incorrect=sum(tok_w) / len(tok_w) if tok_w else None,
        truncated_rate=sum(s["finish_reason"] == "length" for s in samples) / len(samples),
        boxed_rate=sum(s["boxed"] for s in samples) / len(samples),
        solved_by_any=sum(r["n_correct"] > 0 for r in records) / len(records),
        entropy=token_entropy(samples),
    )
    for k in k_values(n):
        row[f"pass@{k}"] = sum(pass_at_k(r["n"], r["n_correct"], k) for r in records) / len(records)
    return row


def checkpoint_step(name: str, cfg: dict | None) -> int:
    if name == "base":
        return 0
    if name == "final":
        return int(cfg["train"]["max_steps"]) if cfg else -1
    m = re.fullmatch(r"checkpoint-(\d+)", name)
    return int(m.group(1)) if m else -1


def collect(root: Path, tag: str):
    rows, level_rows = [], []
    sources = []
    base_dir = root / "base_model" / "eval" / tag / "base"
    if base_dir.exists():
        sources.append(("base", "base", None, base_dir.parent, None))
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in ("base_model", "_dry_run")):
        eval_dir = run_dir / "eval" / tag
        if not eval_dir.exists():
            continue
        cfg = yaml.safe_load(open(run_dir / "config.yaml")) if (run_dir / "config.yaml").exists() else None
        arm = cfg["experiment"]["name"] if cfg else run_dir.name
        seed = cfg["experiment"].get("seed") if cfg else None
        sources.append((run_dir.name, arm, seed, eval_dir, cfg))

    for run, arm, seed, eval_dir, cfg in sources:
        for ckpt_dir in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
            if run == "base" and ckpt_dir.name != "base":
                continue
            for f in sorted(ckpt_dir.glob("*.jsonl.gz")):
                bench = f.name[: -len(".jsonl.gz")]
                records = load_records(f)
                meta = dict(run=run, arm=arm, seed=seed, checkpoint=ckpt_dir.name,
                            step=checkpoint_step(ckpt_dir.name, cfg), benchmark=bench)
                rows.append({**meta, **metrics_for(records)})
                levels = sorted({r["level"] for r in records if r["level"]})
                for level in levels:
                    sub = [r for r in records if r["level"] == level]
                    level_rows.append({**meta, "level": level, **metrics_for(sub)})
    return pd.DataFrame(rows), pd.DataFrame(level_rows)


def add_aes(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df.run == "base"].set_index("benchmark")
    def row_aes(r):
        if r.benchmark not in base.index:
            return None
        b = base.loc[r.benchmark]
        return aes(r["pass@1"], r["mean_tokens"], b["pass@1"], b["mean_tokens"])
    df["aes"] = df.apply(row_aes, axis=1)
    return df


def main():
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out or f"results/{args.tag}")
    out.mkdir(parents=True, exist_ok=True)

    df, by_level = collect(root, args.tag)
    if df.empty:
        raise SystemExit(f"No evaluation results found for tag {args.tag!r} under {root}")
    if not (df.run == "base").any():
        print("[warning] no base-model results; run `python evaluate.py --base --tag "
              f"{args.tag}` to get AES.")
    df = add_aes(df)
    df.sort_values(["benchmark", "arm", "seed", "step"]).to_csv(out / "summary.csv", index=False)
    if not by_level.empty:
        by_level.to_csv(out / "by_level.csv", index=False)

    value_cols = [c for c in df.columns if c.startswith("pass@") or c in
                  ("mean_tokens", "mean_tokens_correct", "truncated_rate", "boxed_rate", "aes", "entropy")]
    agg = df.groupby(["benchmark", "arm", "checkpoint", "step"])[value_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index().sort_values(["benchmark", "arm", "step"])
    agg.to_csv(out / "summary_by_arm.csv", index=False)

    show = ["benchmark", "arm", "checkpoint", "pass@1", f"pass@{df.n_samples.min()}", "mean_tokens", "aes"]
    with pd.option_context("display.max_rows", 200, "display.width", 160, "display.precision", 3):
        print(df.sort_values(["benchmark", "arm", "step"])[show].to_string(index=False))
    print(f"\nSaved tables to {out}/")

    if args.plots:
        make_plots(df, root, out)


def make_plots(df: pd.DataFrame, root: Path, out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plots] matplotlib not installed; skipping")
        return

    for bench, d in df.groupby("benchmark"):
        # pass@k curves at the last checkpoint of each arm (mean over seeds) vs the base model
        fig, ax = plt.subplots(figsize=(5, 4))
        ks = [int(c.split("@")[1]) for c in d.columns if c.startswith("pass@") and d[c].notna().any()]
        ks = sorted(ks)
        for arm, a in d.groupby("arm"):
            last = a[a.step == a.step.max()]
            ys = [last[f"pass@{k}"].mean() for k in ks]
            ax.plot(ks, ys, marker="o", ms=3, label=arm, linestyle="--" if arm == "base" else "-")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("k (samples)")
        ax.set_ylabel("pass@k")
        ax.set_title(f"{bench}: pass@k at last checkpoint")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"passk_{bench}.png", dpi=150)
        plt.close(fig)

        # pass@1, pass@max, length and eval entropy over training
        kmax = max(ks)
        cols = ["pass@1", f"pass@{kmax}", "mean_tokens", "entropy"]
        fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
        for arm, a in d[d.arm != "base"].groupby("arm"):
            g = a.groupby("step")[cols].mean()
            for ax, col in zip(axes, cols):
                s = g[col].dropna()
                ax.plot(s.index, s, marker="o", ms=3, label=arm)
        base = d[d.arm == "base"]
        for ax, col in zip(axes, cols):
            if not base.empty and base[col].notna().any():
                ax.axhline(base[col].mean(), color="gray", linestyle="--", label="base")
            ax.set_xlabel("training step")
            ax.set_title(col)
        axes[0].legend(fontsize=8)
        fig.suptitle(bench)
        fig.tight_layout()
        fig.savefig(out / f"over_training_{bench}.png", dpi=150)
        plt.close(fig)

    # training curves from metrics.csv
    # Column names differ a little between TRL/Unsloth versions, so match by name pattern.
    panels = [("train reward", lambda c: c in ("reward", "train/reward")),
              ("answer length (tokens)", lambda c: c.endswith("completions/mean_length")),
              ("entropy", lambda c: "entropy" in c.lower())]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    found = [False] * len(panels)
    for run_dir in sorted(p for p in root.iterdir() if (p / "metrics.csv").exists()):
        m = pd.read_csv(run_dir / "metrics.csv")
        for i, (ax, (title, match)) in enumerate(zip(axes, panels)):
            col = next((c for c in m.columns if match(c)), None)
            if col is None:
                continue
            s = m[["step", col]].dropna()
            if s.empty:
                continue
            found[i] = True
            ax.plot(s.step, s[col].rolling(10, min_periods=1).mean(), label=run_dir.name)
            ax.set_title(title)
            ax.set_xlabel("training step")
    # Training did not log entropy: fall back to the evaluation estimate at each checkpoint.
    if not found[2] and "entropy" in df.columns and df["entropy"].notna().any():
        ev = df[df["entropy"].notna()]
        bench = "math500" if (ev.benchmark == "math500").any() else ev.benchmark.iloc[0]
        ev = ev[ev.benchmark == bench]
        base = ev[ev.run == "base"]
        for run, a in ev[ev.run != "base"].groupby("run"):
            g = a.groupby("step")["entropy"].mean().sort_index()
            if not base.empty:  # start each curve from the untrained model at step 0
                g = pd.concat([pd.Series({0: base["entropy"].mean()}), g])
            axes[2].plot(g.index, g.values, marker="o", ms=3, label=run)
        if not base.empty:
            axes[2].axhline(base["entropy"].mean(), color="gray", linestyle="--", label="base")
        axes[2].set_title(f"entropy (eval at checkpoints, {bench})")
        axes[2].set_xlabel("training step")
        axes[2].legend(fontsize=7)
        found[2] = True
    for ax, (title, _), ok in zip(axes, panels, found):
        if not ok:
            ax.set_title(title)
            ax.text(0.5, 0.5, "not logged", ha="center", va="center", transform=ax.transAxes)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=150)
    plt.close(fig)
    print(f"[plots] saved to {out}/")


if __name__ == "__main__":
    main()
