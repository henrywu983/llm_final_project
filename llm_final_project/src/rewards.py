"""Reward functions for GRPO, plus logging of every training answer.

TRL calls each reward function with the same batch of answers. The
RewardEngine grades the batch once, caches the result, and each reward
function returns its own component. TRL logs each component separately
(rewards/correctness/mean, rewards/format/mean, rewards/length/mean).

Per batch it also:
  * appends every answer to rollouts.jsonl.gz (text, length, correct, ...)
  * appends summary stats to rollout_stats.csv
  * reports the same stats through TRL's `log_metric`, so they appear in
    metrics.csv and Weights & Biases under rollout/...
"""

from __future__ import annotations

import csv
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .grading import grade, has_boxed
from .length_penalty import LENGTH_PENALTIES


def _text(completion) -> str:
    if isinstance(completion, list):  # chat format: [{"role": "assistant", "content": ...}]
        return completion[-1]["content"] if completion else ""
    return completion


class RewardEngine:
    def __init__(self, cfg: dict, tokenizer, run_dir: Path):
        rcfg = cfg["reward"]
        self.correct_w = float(rcfg["correctness"])
        self.format_bonus = float(rcfg["format_bonus"])
        lp = rcfg["length_penalty"]
        self.length_enabled = bool(lp["enabled"])
        self.length_fn = LENGTH_PENALTIES[lp["type"]] if self.length_enabled else None
        self.length_coef = float(lp["coef"])
        self.max_len = int(cfg["grpo"]["max_completion_length"])
        self.tokenizer = tokenizer
        self.log_rollouts = bool(cfg["logging"].get("log_rollouts", True))
        self.rollout_path = Path(run_dir) / "rollouts.jsonl.gz"
        self.stats_path = Path(run_dir) / "rollout_stats.csv"
        self._cache_key = None
        self._cache = None
        self._batch = 0
        self.trainer = None  # set by train.py after the trainer is created

    # ---- reward functions handed to GRPOTrainer -------------------------
    def reward_functions(self) -> list:
        def correctness(prompts, completions, **kw):
            return self._evaluate(completions, kw)["r_correct"]

        def format(prompts, completions, **kw):  # noqa: A001 (name shows up in TRL logs)
            return self._evaluate(completions, kw)["r_format"]

        def length(prompts, completions, **kw):
            return self._evaluate(completions, kw)["r_length"]

        funcs = [correctness, format]
        if self.length_enabled:
            funcs.append(length)
        return funcs

    # ---- grading (once per batch) --------------------------------------
    def _lengths(self, texts, kw) -> list[int]:
        ids = kw.get("completion_ids")
        if ids is not None:
            return [len(x) for x in ids]
        return [len(x) for x in self.tokenizer(texts, add_special_tokens=False)["input_ids"]]

    def _evaluate(self, completions, kw) -> dict:
        texts = [_text(c) for c in completions]
        qids = list(kw.get("question_id") or range(len(texts)))
        key = (len(texts), hash(tuple(texts)), tuple(qids))
        if key == self._cache_key:
            return self._cache

        answers = kw["answer"]
        lengths = self._lengths(texts, kw)
        correct = [grade(t, a) for t, a in zip(texts, answers)]
        boxed = [has_boxed(t) for t in texts]
        truncated = [l >= self.max_len for l in lengths]
        r_correct = [self.correct_w * float(c) for c in correct]
        r_format = [self.format_bonus if b else 0.0 for b in boxed]
        if self.length_enabled:
            r_length = self.length_fn(qids, lengths, correct, self.max_len, self.length_coef)
        else:
            r_length = [0.0] * len(texts)

        result = dict(r_correct=r_correct, r_format=r_format, r_length=r_length)
        self._cache_key, self._cache = key, result

        state = kw.get("trainer_state")
        if state is None and self.trainer is not None:  # older/patched trainers
            state = self.trainer.state
        step = int(state.global_step) if state is not None else -1
        self._record(step, texts, qids, kw, lengths, correct, boxed, truncated, result,
                     kw.get("log_metric"))
        return result

    # ---- logging --------------------------------------------------------
    def _record(self, step, texts, qids, kw, lengths, correct, boxed, truncated, r, log_metric):
        self._batch += 1
        sources = kw.get("source") or [""] * len(texts)
        levels = kw.get("level") or [0] * len(texts)
        answers = kw["answer"]

        if self.log_rollouts:
            with gzip.open(self.rollout_path, "at", encoding="utf-8") as f:
                for i, t in enumerate(texts):
                    f.write(json.dumps(dict(
                        step=step, batch=self._batch, question_id=qids[i], source=sources[i],
                        level=levels[i], answer=answers[i], n_tokens=lengths[i],
                        truncated=truncated[i], correct=correct[i], boxed=boxed[i],
                        r_correct=r["r_correct"][i], r_format=r["r_format"][i],
                        r_length=r["r_length"][i], text=t,
                    ), ensure_ascii=False) + "\n")

        groups = defaultdict(list)
        for i, q in enumerate(qids):
            groups[q].append(correct[i])
        solve_rates = [sum(g) / len(g) for g in groups.values()]
        len_c = [l for l, c in zip(lengths, correct) if c]
        len_w = [l for l, c in zip(lengths, correct) if not c]
        stats = dict(
            accuracy=sum(correct) / len(correct),
            boxed_rate=sum(boxed) / len(boxed),
            truncated_rate=sum(truncated) / len(truncated),
            len_mean=statistics.fmean(lengths),
            len_std=statistics.pstdev(lengths),
            len_min=min(lengths),
            len_max=max(lengths),
            len_mean_correct=statistics.fmean(len_c) if len_c else float("nan"),
            len_mean_incorrect=statistics.fmean(len_w) if len_w else float("nan"),
            n_questions=len(groups),
            frac_all_correct=sum(s == 1.0 for s in solve_rates) / len(solve_rates),
            frac_all_wrong=sum(s == 0.0 for s in solve_rates) / len(solve_rates),
            reward_length_mean=statistics.fmean(r["r_length"]),
        )

        new_file = not self.stats_path.exists()
        with open(self.stats_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "batch", *stats])
            if new_file:
                writer.writeheader()
            writer.writerow(dict(step=step, batch=self._batch, **stats))

        if log_metric is not None:
            for name, value in stats.items():
                if value == value:  # skip NaN
                    log_metric(f"rollout/{name}", float(value))
