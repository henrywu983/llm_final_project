"""Answer extraction and grading for GSM8K / MATH-style problems.

`grade(text, gold)` is the single function used by both the training reward
and the evaluation, so a "correct" answer means the same thing everywhere.

It uses Hugging Face's `math-verify` (symbolic equivalence: 1/2 == 0.5 ==
\\frac{1}{2}) when installed, and falls back to a simpler string/number check.
"""

from __future__ import annotations

import re

try:  # pip install math-verify
    from math_verify import parse as _mv_parse
    from math_verify import verify as _mv_verify

    HAVE_MATH_VERIFY = True
except ImportError:  # pragma: no cover
    HAVE_MATH_VERIFY = False

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (or \\fbox{...}), with nested braces handled."""
    if not text:
        return None
    start = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if start < 0:
        return None
    i = text.index("{", start) + 1
    depth = 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j].strip()
    return None  # unbalanced (e.g. cut off by the length limit)


def has_boxed(text: str) -> bool:
    return last_boxed(text) is not None


def extract_answer(text: str) -> str | None:
    """The boxed answer if present, otherwise the last number in the text."""
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed
    numbers = _NUMBER_RE.findall(text or "")
    return numbers[-1] if numbers else None


def _normalize(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    for old, new in [("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""),
                     ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("^\\circ", ""), ("^{\\circ}", ""),
                     ("\\%", ""), ("%", ""), ("\\$", "")]:
        s = s.replace(old, new)
    s = s.replace(" ", "").rstrip(".")
    if s.startswith("x=") or s.startswith("y="):
        s = s[2:]
    return s


def _to_float(s: str) -> float | None:
    s = s.replace(",", "")
    m = re.fullmatch(r"\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    m = re.fullmatch(r"(-?[\d.]+)/(-?[\d.]+)", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


def simple_equiv(pred: str | None, gold: str) -> bool:
    """Fallback check: numeric comparison, else normalized string equality."""
    if pred is None:
        return False
    p, g = _normalize(pred), _normalize(str(gold))
    if p == g:
        return True
    pf, gf = _to_float(p), _to_float(g)
    return pf is not None and gf is not None and abs(pf - gf) <= 1e-6 * max(1.0, abs(gf))


def _grade_pair(pair):
    return grade(*pair)


def grade_many(texts: list[str], golds: list[str], workers: int = 8) -> list[bool]:
    """Grade many answers in parallel (used by evaluation, which grades ~50k answers)."""
    pairs = list(zip(texts, golds))
    if workers <= 1 or len(pairs) < 2000:
        return [grade(t, g) for t, g in pairs]
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(workers) as pool:
        return pool.map(_grade_pair, pairs, chunksize=256)


def grade(text: str, gold: str) -> bool:
    """True if the model's final answer in `text` matches `gold`."""
    gold = str(gold).strip()
    if HAVE_MATH_VERIFY:
        try:
            if _mv_verify(_mv_parse(f"\\boxed{{{gold}}}"), _mv_parse(text)):
                return True
        except Exception:
            pass
    return simple_equiv(extract_answer(text), gold)
