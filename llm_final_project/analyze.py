#!/usr/bin/env python
"""Turn saved evaluation samples into result tables (and optional plots).

    python analyze.py --tag boundary
    python analyze.py --tag quick --plots

Reads   outputs/base_model/eval/<tag>/base/*.jsonl.gz
        outputs/<run>/eval/<tag>/<checkpoint>/*.jsonl.gz
Writes  results/<tag>/summary.csv          one row per run x checkpoint x benchmark
        results/<tag>/summary_by_arm.csv   mean and std over seeds
        results/<tag>/by_level.csv         MATH500 results per difficulty level
        results/<tag>/lost_gained.csv      problems the base solved but a run never did (and back),
                                           at each run's last checkpoint, overall and per level
        results/<tag>/*.png                with --plots:
            tradeoff_<bench>.png       accuracy vs length, one point per arm (last checkpoint)
            passk_<bench>.png          pass@k curves, last checkpoint vs base
            over_training_<bench>.png  pass@1, pass@max, length, entropy over training
            by_level_<bench>.png       pass@1, pass@max, length per difficulty level
            training_curves.png        train reward, length, entropy during training

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


# Fixed colour per arm (taken from the arm number in its name), so an arm keeps the
# same colour in every figure no matter which other arms are present.
ARM_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BASE_COLOR = "#777777"


def arm_color(arm: str) -> str:
    if arm == "base":
        return BASE_COLOR
    m = re.match(r"arm(\d+)", arm)
    return ARM_COLORS[(int(m.group(1)) - 1) % len(ARM_COLORS)] if m else "#444444"


def arm_sort_key(arm: str):
    m = re.match(r"arm(\d+)", arm)
    return (0, 0, arm) if arm == "base" else (1, int(m.group(1)) if m else 99, arm)


def last_checkpoint(d: pd.DataFrame) -> pd.DataFrame:
    """Rows of each run's last evaluated checkpoint: "final" if present, else the highest step."""
    out = []
    for _, r in d.groupby("run"):
        pick = r[r.checkpoint == "final"]
        out.append(pick if not pick.empty else r[r.step == r.step.max()])
    return pd.concat(out) if out else d.iloc[0:0]


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
    """Return (summary rows, per-level rows, per-problem counts).

    The per-problem counts map (run, checkpoint, benchmark) to
    {question_id: (level, n_samples, n_correct)} for the lost/gained analysis.
    """
    rows, level_rows = [], []
    problems = {}
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
                problems[(run, ckpt_dir.name, bench)] = {
                    str(r.get("question_id", i)): (r.get("level"), r["n"], r["n_correct"])
                    for i, r in enumerate(records)}
                meta = dict(run=run, arm=arm, seed=seed, checkpoint=ckpt_dir.name,
                            step=checkpoint_step(ckpt_dir.name, cfg), benchmark=bench)
                rows.append({**meta, **metrics_for(records)})
                levels = sorted({r["level"] for r in records if r["level"]})
                for level in levels:
                    sub = [r for r in records if r["level"] == level]
                    level_rows.append({**meta, "level": level, **metrics_for(sub)})
    return pd.DataFrame(rows), pd.DataFrame(level_rows), problems


def lost_gained(df: pd.DataFrame, problems: dict) -> pd.DataFrame:
    """Per run (last checkpoint) and benchmark: which problems moved in or out of reach.

    lost   = solved at least once by the base model, never by the run
    gained = never solved by the base model, solved at least once by the run
    lost_beyond_chance counts lost problems where, if the run were exactly as good as
    the base on that problem, seeing zero successes would have had < 5% chance, i.e.
    (1 - c_base/n_base) ** n_run < 0.05. Problems the base solved only 1-2 times in
    256 can drop to zero by luck alone, so read `lost` together with this column.
    """
    base_rows = df[df.run == "base"]
    if base_rows.empty:
        return pd.DataFrame()
    out = []
    for _, r in last_checkpoint(df[df.run != "base"]).iterrows():
        base_ckpt = base_rows[base_rows.benchmark == r.benchmark]
        if base_ckpt.empty:
            continue
        base_p = problems.get(("base", base_ckpt.iloc[0].checkpoint, r.benchmark), {})
        run_p = problems.get((r.run, r.checkpoint, r.benchmark), {})
        shared = [q for q in run_p if q in base_p]
        if not shared:
            continue
        levels = sorted({base_p[q][0] for q in shared if base_p[q][0]})
        for level in ["all", *levels]:
            qs = [q for q in shared if level == "all" or base_p[q][0] == level]
            lost = [q for q in qs if base_p[q][2] > 0 and run_p[q][2] == 0]
            gained = [q for q in qs if base_p[q][2] == 0 and run_p[q][2] > 0]
            beyond = [q for q in lost
                      if (1 - base_p[q][2] / base_p[q][1]) ** run_p[q][1] < 0.05]
            out.append(dict(
                run=r.run, arm=r.arm, seed=r.seed, checkpoint=r.checkpoint, benchmark=r.benchmark,
                level=level, n_problems=len(qs),
                base_solved=sum(base_p[q][2] > 0 for q in qs),
                run_solved=sum(run_p[q][2] > 0 for q in qs),
                lost=len(lost), gained=len(gained), net=len(gained) - len(lost),
                lost_beyond_chance=len(beyond),
                base_pass1=sum(base_p[q][2] / base_p[q][1] for q in qs) / len(qs),
                run_pass1=sum(run_p[q][2] / run_p[q][1] for q in qs) / len(qs),
                lost_ids=" ".join(lost), gained_ids=" ".join(gained),
            ))
    return pd.DataFrame(out)


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

    df, by_level, problems = collect(root, args.tag)
    if df.empty:
        raise SystemExit(f"No evaluation results found for tag {args.tag!r} under {root}")
    if not (df.run == "base").any():
        print("[warning] no base-model results; run `python evaluate.py --base --tag "
              f"{args.tag}` to get AES.")
    df = add_aes(df)
    df.sort_values(["benchmark", "arm", "seed", "step"]).to_csv(out / "summary.csv", index=False)
    if not by_level.empty:
        by_level.to_csv(out / "by_level.csv", index=False)
    lg = lost_gained(df, problems)
    if not lg.empty:
        lg.to_csv(out / "lost_gained.csv", index=False)

    value_cols = [c for c in df.columns if c.startswith("pass@") or c in
                  ("mean_tokens", "mean_tokens_correct", "truncated_rate", "boxed_rate", "aes", "entropy")]
    agg = df.groupby(["benchmark", "arm", "checkpoint", "step"])[value_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index().sort_values(["benchmark", "arm", "step"])
    agg.to_csv(out / "summary_by_arm.csv", index=False)

    show = ["benchmark", "arm", "checkpoint", "pass@1", f"pass@{df.n_samples.min()}", "mean_tokens", "aes"]
    with pd.option_context("display.max_rows", 200, "display.width", 160, "display.precision", 3):
        print(df.sort_values(["benchmark", "arm", "step"])[show].to_string(index=False))
    if not lg.empty:
        print("\nLost / gained vs base (last checkpoint, all levels):")
        cols = ["benchmark", "arm", "seed", "base_solved", "run_solved", "lost", "lost_beyond_chance", "gained"]
        print(lg[lg.level == "all"][cols].to_string(index=False))
    print(f"\nSaved tables to {out}/")

    if args.plots:
        make_plots(df, root, out, by_level)


def make_plots(df: pd.DataFrame, root: Path, out: Path, by_level: pd.DataFrame | None = None) -> None:
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
        for arm, a in sorted(d.groupby("arm"), key=lambda t: arm_sort_key(t[0])):
            last = last_checkpoint(a)
            ys = [last[f"pass@{k}"].mean() for k in ks]
            ax.plot(ks, ys, marker="o", ms=3, label=arm, color=arm_color(arm),
                    linestyle="--" if arm == "base" else "-")
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
        for arm, a in sorted(d[d.arm != "base"].groupby("arm"), key=lambda t: arm_sort_key(t[0])):
            a = a[a.checkpoint != "final"] if (a.checkpoint != "final").any() else a
            g = a.groupby("step")[cols].mean()
            for ax, col in zip(axes, cols):
                s = g[col].dropna()
                ax.plot(s.index, s, marker="o", ms=3, label=arm, color=arm_color(arm))
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

        plot_tradeoff(plt, d, bench, out)
        if by_level is not None and not by_level.empty:
            plot_by_level(plt, by_level[by_level.benchmark == bench], bench, kmax, out)

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
            ax.plot(s.step, s[col].rolling(10, min_periods=1).mean(), label=run_dir.name,
                    color=arm_color(run_dir.name))
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
            axes[2].plot(g.index, g.values, marker="o", ms=3, label=run, color=arm_color(run))
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
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=150)
    plt.close(fig)
    print(f"[plots] saved to {out}/")


def plot_tradeoff(plt, d: pd.DataFrame, bench: str, out: Path) -> None:
    """Accuracy vs length at each arm's last checkpoint; up-and-left is better."""
    last = last_checkpoint(d)
    g = last.groupby("arm").agg(x=("mean_tokens", "mean"), y=("pass@1", "mean"),
                                xs=("mean_tokens", "std"), ys=("pass@1", "std"), n=("run", "count"))
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for arm in sorted(g.index, key=arm_sort_key):
        r = g.loc[arm]
        color = arm_color(arm)
        if r.n > 1:  # several seeds: show the spread
            ax.errorbar(r.x, 100 * r.y, xerr=r.xs, yerr=100 * r.ys, color=color, lw=1, capsize=3, zorder=1)
        ax.scatter(r.x, 100 * r.y, s=220 if arm == "base" else 70, color=color,
                   marker="*" if arm == "base" else "o", edgecolor="white", linewidth=1.2, zorder=2, label=arm)
        short = re.match(r"arm\d+", arm).group(0) if re.match(r"arm\d+", arm) else arm
        ax.annotate(short, (r.x, 100 * r.y), xytext=(7, 4), textcoords="offset points", fontsize=8, color="#333")
    ax.set_xlabel("mean response length (tokens)")
    ax.set_ylabel("pass@1 (%)")
    ax.set_title(f"{bench}: accuracy vs length (last checkpoint)\nup and left is better", fontsize=9)
    ax.grid(color="#e5e5e5", lw=0.6)
    ax.margins(x=0.15, y=0.15)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out / f"tradeoff_{bench}.png", dpi=150)
    plt.close(fig)


def plot_by_level(plt, lv: pd.DataFrame, bench: str, kmax: int, out: Path) -> None:
    """pass@1, pass@kmax and length per difficulty level at each arm's last checkpoint."""
    if lv.empty or lv.level.nunique() < 2:
        return
    last = pd.concat([last_checkpoint(g) for _, g in lv.groupby("level")])
    cols = [("pass@1", "pass@1 (%)", 100), (f"pass@{kmax}", f"pass@{kmax} (%)", 100),
            ("mean_tokens", "mean length (tokens)", 1)]
    cols = [c for c in cols if c[0] in last.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(4.4 * len(cols), 3.6))
    for arm, a in sorted(last.groupby("arm"), key=lambda t: arm_sort_key(t[0])):
        g = a.groupby("level")[[c[0] for c in cols]].mean().sort_index()
        for ax, (col, _, scale) in zip(axes, cols):
            ax.plot(g.index, g[col] * scale, marker="o", ms=4, lw=2, label=arm, color=arm_color(arm),
                    linestyle="--" if arm == "base" else "-", mfc="white" if arm == "base" else None)
    for ax, (_, title, _) in zip(axes, cols):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("difficulty level")
        ax.set_xticks(sorted(last.level.unique()))
        ax.grid(color="#e5e5e5", lw=0.6)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{bench}: by difficulty level (last checkpoint)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"by_level_{bench}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
