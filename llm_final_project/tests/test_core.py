"""CPU-only tests. Run with:  python -m pytest -q tests/"""

from __future__ import annotations

import gzip
import json
import math
import sys
from math import comb
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data as data_mod  # noqa: E402
from src.config import apply_overrides, load_config  # noqa: E402
from src.grading import extract_answer, grade, has_boxed, last_boxed  # noqa: E402
from src.length_penalty import arora_zanette, kimi, linear  # noqa: E402
from src.metrics import aes, k_values, pass_at_k  # noqa: E402

ARMS = ["arm1_grpo", "arm2_dr_grpo", "arm3_length_penalty"]


# ---------------------------------------------------------------- grading
def test_last_boxed_nested_and_truncated():
    assert last_boxed(r"so \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert last_boxed(r"\boxed{1} then \boxed{2}") == "2"
    assert last_boxed(r"cut off \boxed{\frac{1}{") is None
    assert not has_boxed("no box here 42")


@pytest.mark.parametrize("text,gold,ok", [
    (r"The answer is \boxed{18}.", "18", True),
    (r"\boxed{1,234}", "1234", True),
    ("so she has 18 apples", "18", True),                # no box -> last number
    (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}", True),
    (r"\boxed{0.5}", r"\frac{1}{2}", True),
    (r"\boxed{\dfrac{3}{4}}", r"\frac34", True),
    (r"\boxed{17}", "18", False),
    ("", "18", False),
    (r"\boxed{x=5}", "5", True),
])
def test_grade(text, gold, ok):
    assert grade(text, gold) is ok


def test_grade_without_math_verify(monkeypatch):
    import src.grading as g

    monkeypatch.setattr(g, "HAVE_MATH_VERIFY", False)
    assert g.grade(r"\boxed{1,234}", "1234") and g.grade(r"\boxed{\dfrac{1}{2}}", r"\frac{1}{2}")
    assert g.grade(r"\boxed{0.5}", r"\frac{1}{2}") and not g.grade(r"\boxed{3}", "4")


def test_extract_answer_prefers_box():
    assert extract_answer(r"3 + 4 = 7, \boxed{7} and 9") == "7"


# ---------------------------------------------------------------- metrics
@pytest.mark.parametrize("n,c,k", [(8, 0, 1), (8, 3, 1), (8, 3, 4), (256, 5, 64), (16, 16, 8), (16, 1, 16)])
def test_pass_at_k_matches_formula(n, c, k):
    exact = 1 - comb(n - c, k) / comb(n, k)
    assert math.isclose(pass_at_k(n, c, k), exact, rel_tol=1e-9, abs_tol=1e-12)


def test_k_values():
    assert k_values(256) == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert k_values(4) == [1, 2, 4]
    assert k_values(6) == [1, 2, 4, 6]


def test_aes():
    assert math.isclose(aes(0.5, 500, 0.5, 1000), 0.5)            # same accuracy, half the tokens
    assert math.isclose(aes(0.55, 1000, 0.5, 1000), 0.3)          # +10% accuracy -> 3 * 0.1
    assert math.isclose(aes(0.45, 1000, 0.5, 1000), -0.5)         # -10% accuracy -> -5 * 0.1
    assert math.isnan(aes(0.5, 500, 0.0, 1000))


# ---------------------------------------------------------------- length penalties
def test_arora_zanette():
    q = ["a"] * 4 + ["b"] * 2
    lengths = [100, 200, 300, 50, 80, 90]
    correct = [True, True, True, False, False, True]
    out = arora_zanette(q, lengths, correct, 1024, 0.2)
    assert out[3] == out[4] == 0.0                               # incorrect: no length term
    assert out[0] > out[1] > out[2]                              # longer correct -> bigger penalty
    assert math.isclose(out[1], -0.1)                            # mean length -> sigmoid(0)=0.5
    assert math.isclose(out[5], -0.1)                            # single correct answer
    assert all(-0.2 <= x <= 0 for x in out)


def test_linear_and_kimi():
    assert linear(["a", "a"], [512, 512], [True, False], 1024, 0.2) == [-0.1, 0.0]
    out = kimi(["a"] * 3, [100, 200, 300], [True, True, False], 1024, 1.0)
    assert out == [0.5, 0.0, -0.5]
    assert kimi(["a"] * 2, [100, 100], [True, False], 1024, 1.0) == [0.0, 0.0]


# ---------------------------------------------------------------- config
def test_arm_configs_differ_only_in_method():
    cfgs = {a: load_config(ROOT / "configs" / f"{a}.yaml") for a in ARMS}
    assert cfgs["arm1_grpo"]["grpo"]["loss_type"] == "grpo"
    assert cfgs["arm2_dr_grpo"]["grpo"]["loss_type"] == "dr_grpo"
    assert cfgs["arm2_dr_grpo"]["grpo"]["scale_rewards"] == "none"
    assert cfgs["arm3_length_penalty"]["reward"]["length_penalty"]["enabled"] is True
    assert cfgs["arm1_grpo"]["reward"]["length_penalty"]["enabled"] is False
    for section in ["model", "data", "prompt", "train", "eval"]:
        assert cfgs["arm1_grpo"][section] == cfgs["arm2_dr_grpo"][section] == cfgs["arm3_length_penalty"][section]


def test_overrides():
    cfg = load_config(ROOT / "configs" / "arm1_grpo.yaml")
    changed = apply_overrides(cfg, ["machine.gpu_memory_utilization=0.35", "train.max_steps=10"])
    assert cfg["machine"]["gpu_memory_utilization"] == 0.35
    assert cfg["train"]["max_steps"] == 10
    assert changed == ["train.max_steps=10"]
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["train.max_stepz=10"])


# ---------------------------------------------------------------- TRL settings
@pytest.mark.parametrize("arm,loss,scale", [("arm1_grpo", "grpo", "group"),
                                            ("arm2_dr_grpo", "dr_grpo", "none"),
                                            ("arm3_length_penalty", "grpo", "group")])
def test_grpo_config(tmp_path, arm, loss, scale):
    import dataclasses

    trl = pytest.importorskip("trl")
    import torch

    from src.trl_setup import build_grpo_config

    GRPOConfig = trl.GRPOConfig
    if not torch.cuda.is_available():  # test machine without a GPU: turn off bf16
        @dataclasses.dataclass
        class GRPOConfig(trl.GRPOConfig):  # noqa: N801
            bf16: bool = False
            use_cpu: bool = True

    cfg = load_config(ROOT / "configs" / f"{arm}.yaml")
    args = build_grpo_config(GRPOConfig, cfg, str(tmp_path), arm, 0)
    assert args.loss_type == loss
    assert str(args.scale_rewards).lower() in {scale, str(scale != "none").lower()}
    assert args.beta == 0.0 and args.num_generations == 8
    assert args.per_device_train_batch_size * args.gradient_accumulation_steps == 64
    assert args.save_total_limit is None and args.save_steps == 50


# ---------------------------------------------------------------- reward engine
class FakeTokenizer:
    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [t.split() for t in texts]}


def test_reward_engine(tmp_path):
    from src.rewards import RewardEngine

    cfg = load_config(ROOT / "configs" / "arm3_length_penalty.yaml")
    engine = RewardEngine(cfg, FakeTokenizer(), tmp_path)
    funcs = engine.reward_functions()
    assert [f.__name__ for f in funcs] == ["correctness", "format", "length"]

    comps = [[{"role": "assistant", "content": c}] for c in
             [r"short \boxed{4}", r"a much longer answer here \boxed{4}", "wrong 5", r"\boxed{9}"]]
    logged = {}
    kw = dict(answer=["4", "4", "4", "9"], question_id=["q1", "q1", "q1", "q2"], source=["gsm8k"] * 4,
              level=[0] * 4, completion_ids=[[0] * 3, [0] * 6, [0] * 2, [0] * 1],
              trainer_state=SimpleNamespace(global_step=7), log_metric=lambda k, v: logged.__setitem__(k, v))
    rc = funcs[0](None, comps, **kw)
    rf = funcs[1](None, comps, **kw)
    rl = funcs[2](None, comps, **kw)
    assert rc == [1.0, 1.0, 0.0, 1.0]
    assert rf == [0.1, 0.1, 0.0, 0.1]
    assert rl[0] > rl[1] and rl[2] == 0.0                         # shorter correct answer penalized less
    assert engine._batch == 1                                     # graded once, not three times
    lines = [json.loads(x) for x in gzip.open(tmp_path / "rollouts.jsonl.gz", "rt")]
    assert len(lines) == 4 and lines[0]["step"] == 7 and lines[0]["correct"] is True
    assert (tmp_path / "rollout_stats.csv").exists()
    assert math.isclose(logged["rollout/accuracy"], 0.75)
    assert math.isclose(logged["rollout/frac_all_correct"], 0.5)


# ---------------------------------------------------------------- data subsetting
def test_benchmark_subset_is_stratified_and_fixed(monkeypatch):
    rows = [dict(question_id=f"m-{i}", source="math500", level=1 + i % 5, question="q", answer="1")
            for i in range(500)]
    monkeypatch.setattr(data_mod, "load_generic", lambda spec, tag: list(rows))
    a = data_mod.load_benchmark("math500", {"id": "HuggingFaceH4/MATH-500"}, 200, seed=0)
    b = data_mod.load_benchmark("math500", {"id": "HuggingFaceH4/MATH-500"}, 200, seed=0)
    assert [r["question_id"] for r in a] == [r["question_id"] for r in b]
    assert len(a) == 200
    for level in range(1, 6):
        assert sum(r["level"] == level for r in a) == 40


# ---------------------------------------------------------------- evaluation + analysis
def _write_eval(path: Path, n_correct: list[int], n: int, tokens: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for i, c in enumerate(n_correct):
            samples = [dict(text="", n_tokens=tokens, finish_reason="stop", correct=j < c, boxed=True)
                       for j in range(n)]
            f.write(json.dumps(dict(question_id=f"q{i}", level=1 + i % 2, answer="1", n=n,
                                    n_correct=c, samples=samples)) + "\n")


def test_analysis_pipeline(tmp_path):
    import analyze
    import yaml

    root = tmp_path / "outputs"
    _write_eval(root / "base_model/eval/boundary/base/math500.jsonl.gz", [2, 0, 4, 1], 4, 1000)
    run = root / "arm1_grpo_seed0"
    run.mkdir(parents=True)
    cfg = load_config(ROOT / "configs/arm1_grpo.yaml")
    cfg["experiment"]["seed"] = 0
    yaml.safe_dump(cfg, open(run / "config.yaml", "w"))
    _write_eval(run / "eval/boundary/checkpoint-50/math500.jsonl.gz", [4, 0, 4, 0], 4, 500)
    df, by_level = analyze.collect(root, "boundary")
    df = analyze.add_aes(df)
    base = df[df.run == "base"].iloc[0]
    trained = df[df.run == "arm1_grpo_seed0"].iloc[0]
    assert trained.step == 50 and trained.arm == "arm1_grpo"
    assert math.isclose(base["pass@1"], 7 / 16) and math.isclose(trained["pass@1"], 0.5)
    assert math.isclose(trained["pass@4"], 0.5) and math.isclose(base["pass@4"], 0.75)  # boundary narrowed
    assert math.isclose(trained.aes, 0.5 + 3 * (0.5 - 7 / 16) / (7 / 16))
    assert set(by_level.level) == {1, 2}


def test_find_targets(tmp_path):
    import evaluate

    for step in (50, 100):
        d = tmp_path / "checkpoints" / f"checkpoint-{step}"
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text("{}")
    (tmp_path / "final_lora").mkdir()
    names = [n for n, _ in evaluate.find_targets(tmp_path, "all")]
    assert names == ["checkpoint-50", "checkpoint-100", "final"]
    assert [n for n, _ in evaluate.find_targets(tmp_path, "100,final")] == ["checkpoint-100", "final"]
    with pytest.raises(FileNotFoundError):
        evaluate.find_targets(tmp_path, "75")
