"""Trainer callback that saves every logged metric to metrics.jsonl / metrics.csv."""

from __future__ import annotations

import json
import time
from pathlib import Path

from transformers import TrainerCallback


class MetricsCallback(TrainerCallback):
    """Appends each log row (loss, rewards, entropy, KL, lengths, grad norm, lr, ...).

    Also records peak GPU memory, so memory use can be compared across methods.
    """

    def __init__(self, run_dir: Path):
        self.jsonl = Path(run_dir) / "metrics.jsonl"
        self.csv = Path(run_dir) / "metrics.csv"
        self.t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row = dict(step=state.global_step, elapsed_s=round(time.time() - self.t0, 1))
        try:
            import torch

            if torch.cuda.is_available():
                row["gpu_peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        except Exception:  # noqa: BLE001
            pass
        row.update(logs)
        with open(self.jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=float) + "\n")

    def on_train_end(self, args, state, control, **kwargs):
        write_metrics_csv(self.jsonl, self.csv)


def write_metrics_csv(jsonl: Path, csv: Path) -> None:
    try:
        import pandas as pd

        rows = [json.loads(line) for line in open(jsonl, encoding="utf-8")]
        pd.DataFrame(rows).to_csv(csv, index=False)
    except Exception as e:  # noqa: BLE001
        print(f"[metrics] could not write {csv}: {e}")
