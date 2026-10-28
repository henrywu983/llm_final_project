"""Length terms added to the reward.

Total reward for one answer = correctness + format bonus + length term.
Each function returns one length term per answer. Answers to the same
question are compared with each other (grouped by `question_ids`).

To add a new variant, write a function with the same signature and add it to
LENGTH_PENALTIES; then set `reward.length_penalty.type` to its name in a config
file. Extra keyword arguments (like `tau`) are filled from the same config block.
"""

from __future__ import annotations

import math
from collections import defaultdict


def _groups(question_ids):
    groups = defaultdict(list)
    for i, q in enumerate(question_ids):
        groups[q].append(i)
    return groups.values()


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def arora_zanette(question_ids, lengths, correct, max_len, coef):
    """Arora & Zanette (2025), "Training Language Models to Reason Efficiently".

    Correct answers get -coef * sigmoid(z), where z is the answer's length
    z-score among the CORRECT answers to the same question. Incorrect answers
    get 0. So the correct reward is 1 - coef * sigmoid(z).
    """
    out = [0.0] * len(lengths)
    for idx in _groups(question_ids):
        good = [i for i in idx if correct[i]]
        if not good:
            continue
        ls = [lengths[i] for i in good]
        mean = sum(ls) / len(ls)
        std = (sum((x - mean) ** 2 for x in ls) / len(ls)) ** 0.5
        for i in good:
            z = (lengths[i] - mean) / std if std > 0 else 0.0
            out[i] = -coef * _sigmoid(z)
    return out


def linear(question_ids, lengths, correct, max_len, coef):
    """Linear penalty (the "+LP" baseline in DDCA): -coef * length / max_len for correct answers."""
    return [-coef * l / max_len if c else 0.0 for l, c in zip(lengths, correct)]


def kimi(question_ids, lengths, correct, max_len, coef):
    """Kimi k1.5 length reward, scaled by coef.

    lambda = 0.5 - (len - min) / (max - min) within each question's group.
    Correct answers get coef * lambda; incorrect get coef * min(0, lambda).
    """
    out = [0.0] * len(lengths)
    for idx in _groups(question_ids):
        ls = [lengths[i] for i in idx]
        lo, hi = min(ls), max(ls)
        if hi == lo:
            continue
        for i in idx:
            lam = 0.5 - (lengths[i] - lo) / (hi - lo)
            out[i] = coef * (lam if correct[i] else min(0.0, lam))
    return out


def _success_rates(question_ids, correct):
    """Share of correct answers in each question's group: p_hat = (#correct) / (group size)."""
    rate = {}
    for idx in _groups(question_ids):
        rate[question_ids[idx[0]]] = sum(bool(correct[i]) for i in idx) / len(idx)
    return rate


def gated_arora_zanette(question_ids, lengths, correct, max_len, coef, tau=0.5):
    """Arm 4, our difficulty-aware (DA) penalty: an on/off gate on the arm 3 penalty.

    p_hat = share of the group's answers that are correct. If p_hat >= tau the
    question counts as easy and gets exactly the arm 3 (Arora & Zanette) length
    term; otherwise (hard) it gets no length term at all.
    """
    base = arora_zanette(question_ids, lengths, correct, max_len, coef)
    rate = _success_rates(question_ids, correct)
    return [b if rate[q] >= tau else 0.0 for b, q in zip(base, question_ids)]


def scaled_arora_zanette(question_ids, lengths, correct, max_len, coef):
    """Arm 5, DDCA-inspired (reward-level only): the arm 3 penalty times p_hat.

    Like DDCA's n/N factor, the length pressure fades smoothly on hard questions
    and is full on easy ones. Unlike DDCA, it is added to the reward (so GRPO
    normalizes it together with correctness) rather than to a separate,
    correct-only length advantage.
    """
    base = arora_zanette(question_ids, lengths, correct, max_len, coef)
    rate = _success_rates(question_ids, correct)
    return [b * rate[q] for b, q in zip(base, question_ids)]


LENGTH_PENALTIES = {
    "arora_zanette": arora_zanette,
    "linear": linear,
    "kimi": kimi,
    "gated_arora_zanette": gated_arora_zanette,
    "scaled_arora_zanette": scaled_arora_zanette,
}


def extra_params(fn, lp_cfg: dict) -> dict:
    """Config values (e.g. tau) that this penalty function accepts as keyword arguments."""
    import inspect

    accepted = set(inspect.signature(fn).parameters) - {"question_ids", "lengths", "correct", "max_len", "coef"}
    return {k: lp_cfg[k] for k in accepted if k in lp_cfg}
