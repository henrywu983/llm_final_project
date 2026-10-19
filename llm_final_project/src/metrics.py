"""Evaluation metrics: unbiased pass@k and the Accuracy-Efficiency Score."""

from __future__ import annotations

import math


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021).

    n = samples drawn for the problem, c = how many were correct.
    Equals 1 - C(n-c, k) / C(n, k), computed in a numerically stable way.
    """
    if k > n:
        raise ValueError(f"k={k} is larger than the number of samples n={n}")
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def k_values(n: int) -> list[int]:
    """1, 2, 4, ... up to n (and n itself if it is not a power of two)."""
    ks, k = [], 1
    while k <= n:
        ks.append(k)
        k *= 2
    if ks[-1] != n:
        ks.append(n)
    return ks


def aes(p: float, length: float, p_base: float, length_base: float) -> float:
    """Accuracy-Efficiency Score (Luo et al., 2025; as used in DDCA).

    AES = (L_b - L) / L_b + 3 (p - p_b) / p_b   if p >= p_b
                            -5 (p_b - p) / p_b  if p <  p_b
    p = pass@1, L = mean response tokens, _b = base model.
    Accuracy drops are penalized more (x5) than gains are rewarded (x3).
    """
    if not p_base or not length_base:
        return math.nan
    token_term = (length_base - length) / length_base
    if p >= p_base:
        acc_term = 3.0 * (p - p_base) / p_base
    else:
        acc_term = -5.0 * (p_base - p) / p_base
    return token_term + acc_term
