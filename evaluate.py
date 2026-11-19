"""Metrics: target behavior, loop rate, regression, diversity."""

from __future__ import annotations

import numpy as np

from data import CHAR2IDX
from edit import collapse_rate
from model import CharLM, ids_to_text


def distinct_n(ids: list[int], n: int = 2) -> float:
    if len(ids) < n:
        return 0.0
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return float(len(set(grams)) / len(grams))


def evaluate(model: CharLM, xe: np.ndarray, ye: np.ndarray,
             trap_char: str, seed: int = 0) -> dict:
    """All metrics for one model on the frozen eval set."""
    acc = model.accuracy(xe, ye)
    gen = model.generate(n_chars=120, temperature=0.8, seed=seed)
    return {
        "target_acc": round(float(acc), 4),
        "loop_rate": round(collapse_rate(model, trap_char, seed=seed), 4),
        "distinct2": round(distinct_n(gen, 2), 4),
        "sample": ids_to_text(gen)[:120],
    }
