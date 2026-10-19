"""Length terms added to the reward.

Total reward for one answer = correctness + format bonus + length term.
Each function returns one length term per answer. Answers to the same
question are compared with each other (grouped by `question_ids`).

To add a new variant (e.g. our own method), write a function with the same
signature and add it to LENGTH_PENALTIES; then set
`reward.length_penalty.type` to its name in a config file.
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


LENGTH_PENALTIES = {
    "arora_zanette": arora_zanette,
    "linear": linear,
    "kimi": kimi,
}
