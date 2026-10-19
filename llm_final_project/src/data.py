"""Datasets: training problems (GSM8K + MATH train splits) and test benchmarks.

Every problem becomes a dict with:
    question_id  unique, stable id (used to group answers and to join results)
    source       gsm8k | math | math500 | amc23 | ...
    level        difficulty level 1-5 for MATH problems, 0 if unknown
    question     problem text
    answer       gold answer as a string
"""

from __future__ import annotations

import random
import re
from collections import defaultdict

from datasets import load_dataset

from .grading import last_boxed


def build_messages(question: str, instruction: str) -> list[dict]:
    """Chat-format prompt. Training and evaluation must use exactly this."""
    return [{"role": "user", "content": f"{question}\n{instruction}"}]


def _pick(row: dict, names: list[str]):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    raise KeyError(f"None of the fields {names} found; available: {list(row)}")


def _answer_str(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    s = str(value).strip()
    return s[:-2] if re.fullmatch(r"-?\d+\.0", s) else s


def _level(value) -> int:
    if isinstance(value, int):
        return value
    m = re.search(r"\d", str(value or ""))
    return int(m.group()) if m else 0


def _load(spec: dict):
    ds = load_dataset(spec["id"], spec.get("config"))
    split = spec.get("split", "train")
    if split not in ds:
        fallback = next(iter(ds))
        print(f"[data] split {split!r} not found in {spec['id']}; using {fallback!r}")
        split = fallback
    return ds[split], split


def load_gsm8k(spec: dict, tag: str = "gsm8k") -> list[dict]:
    ds, split = _load(spec)
    rows = []
    for i, r in enumerate(ds):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        rows.append(dict(question_id=f"{tag}-{split}-{i}", source=tag, level=0,
                         question=r["question"], answer=gold))
    return rows


def load_math_train(spec: dict) -> list[dict]:
    ds, split = _load(spec)
    levels = set(spec.get("levels") or [1, 2, 3, 4, 5])
    rows, skipped = [], 0
    for i, r in enumerate(ds):
        level = _level(r.get("level"))
        gold = r.get("answer") or last_boxed(_pick(r, ["solution"]))
        if not gold:
            skipped += 1
            continue
        if level and level not in levels:
            continue
        rows.append(dict(question_id=f"math-{split}-{i}", source="math", level=level,
                         question=_pick(r, ["problem", "question"]), answer=str(gold)))
    if skipped:
        print(f"[data] MATH train: skipped {skipped} problems without a boxed answer")
    return rows


def load_generic(spec: dict, tag: str) -> list[dict]:
    """Benchmarks with problem/question + answer fields (MATH-500, AMC23, ...)."""
    ds, split = _load(spec)
    rows = []
    for i, r in enumerate(ds):
        pid = r.get("unique_id") or r.get("id") or i
        rows.append(dict(question_id=f"{tag}-{pid}", source=tag, level=_level(r.get("level")),
                         question=_pick(r, ["problem", "question", "prompt"]),
                         answer=_answer_str(_pick(r, ["answer", "final_answer", "solution"]))))
    return rows


def build_train_rows(cfg: dict) -> list[dict]:
    dcfg = cfg["data"]
    rows = []
    for source in dcfg["train_sources"]:
        if source == "gsm8k":
            rows += load_gsm8k(dcfg["gsm8k"])
        elif source == "math":
            rows += load_math_train(dcfg["math"])
        else:
            raise ValueError(f"Unknown training source: {source}")
    random.Random(dcfg["shuffle_seed"]).shuffle(rows)
    if dcfg.get("max_train_examples"):
        rows = rows[: dcfg["max_train_examples"]]
    return rows


def to_trl_dataset(rows: list[dict], instruction: str):
    """Hugging Face Dataset with a chat `prompt` column plus metadata columns.

    TRL passes every extra column (answer, question_id, ...) to the reward
    functions as keyword arguments.
    """
    from datasets import Dataset

    return Dataset.from_list([
        dict(prompt=build_messages(r["question"], instruction), answer=r["answer"],
             question_id=r["question_id"], source=r["source"], level=r["level"])
        for r in rows
    ])


def load_benchmark(name: str, spec: dict, max_problems: int | None, seed: int) -> list[dict]:
    """Load a test benchmark and take a fixed (seeded) subset if requested.

    MATH-500 subsets are stratified by level so every difficulty stays represented.
    """
    rows = load_gsm8k(spec, tag=name) if "gsm8k" in spec["id"] else load_generic(spec, tag=name)
    if not max_problems or len(rows) <= max_problems:
        return rows
    rng = random.Random(seed)
    if any(r["level"] for r in rows):
        by_level = defaultdict(list)
        for r in rows:
            by_level[r["level"]].append(r)
        picked = []
        for level, items in sorted(by_level.items()):
            take = round(max_problems * len(items) / len(rows))
            picked += rng.sample(items, min(take, len(items)))
        rest = [r for r in rows if r not in picked]
        if len(picked) < max_problems:
            picked += rng.sample(rest, max_problems - len(picked))
        picked = picked[:max_problems]
    else:
        picked = rng.sample(rows, max_problems)
    order = {r["question_id"]: i for i, r in enumerate(rows)}
    return sorted(picked, key=lambda r: order[r["question_id"]])
