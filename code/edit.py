"""Controlled loop-direction injection and weight-edit methods.

Mirrors the paper's central contrast:
  - oracle edit        : subtract the exact injected direction (upper bound;
                         corresponds to "reasoning directly about weights"
                         with privileged knowledge of ground truth)
  - weight-only (b=0)  : no forward feedback; infer the direction from weight
                         statistics alone
  - feedback (b=5/20)  : use b dev forward passes to ESTIMATE the direction
                         and tune its scale -> "optimizing edits through
                         repeated model feedback"
  - random (C3)        : rank-1 edit of matched norm, wrong direction
  - finetune (C5)      : a few gradient steps on dev data (conventional)

Injection: add D = alpha * outer(u, v) to W2 (hidden->vocab).
  u = a fixed hidden direction (mean dev hidden activation, normalized)
  v = pushes logits toward a chosen "trap" character -> repetitive collapse
Removing D restores behavior (condition A) but cannot create Swedish ability
in a model that never learned it (condition B).
"""

from __future__ import annotations

import numpy as np

from data import CHAR2IDX, CHARS
from model import CharLM, softmax


# ---------------------------------------------------------------------------
# Injection (we, the experimenters, hold ground truth)
# ---------------------------------------------------------------------------

def mean_hidden_direction(model: CharLM, x: np.ndarray) -> np.ndarray:
    """Normalized mean hidden activation over a batch -- the direction u."""
    h = model.hidden(x).mean(axis=0)
    n = np.linalg.norm(h) + 1e-9
    return (h / n).astype(np.float32)


def inject_loop(model: CharLM, x_dev: np.ndarray, trap_char: str = "a",
                target_collapse: float = 0.55, seed: int = 0) -> dict:
    """Inject a rank-1 collapse direction into W2. Returns ground-truth dict.

    alpha is calibrated by bisection so that greedy generation on dev prompts
    collapses to `trap_char` at roughly `target_collapse` rate -- a clear but
    not total behavioral failure (so edits have room to help).
    """
    u = mean_hidden_direction(model, x_dev)
    v = np.zeros(model.V, dtype=np.float32)
    v[CHAR2IDX[trap_char]] = 1.0
    # center v so it is a pure "push toward trap" direction
    v = v - v.mean()

    def collapse_rate(alpha: float) -> float:
        # average over several seeds -> smooth, monotone calibration
        m = model.clone()
        m.W2 = (m.W2 + alpha * np.outer(u, v)).astype(np.float32)
        rates = []
        for sd in range(seed, seed + 4):
            gen = m.generate(n_chars=60, temperature=0.8, seed=sd)
            rates.append(float(np.mean([i == CHAR2IDX[trap_char] for i in gen])))
        return float(np.mean(rates))

    lo, hi = 0.0, 30.0
    for _ in range(16):
        mid = (lo + hi) / 2
        if collapse_rate(mid) < target_collapse:
            lo = mid
        else:
            hi = mid
    alpha = (lo + hi) / 2
    D = alpha * np.outer(u, v)
    model.W2 = (model.W2 + D).astype(np.float32)
    return {"u": u, "v": v, "alpha": float(alpha), "D": D.astype(np.float32),
            "trap_char": trap_char, "collapse_after": collapse_rate(alpha)}


# ---------------------------------------------------------------------------
# Estimate the collapse direction WITHOUT ground truth (the editor's view)
# ---------------------------------------------------------------------------

def estimate_direction_weight_only(broken: CharLM, x_dev: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """b=0: no forward pass allowed. Infer from weight statistics only.

    Heuristic: the injected column makes one W2 column disproportionately
    aligned with the mean hidden direction. We pick the most likely trap
    column by row-norm of W2 and build a centered one-hot v_hat; u_hat is the
    mean hidden direction (computable from weights + a fixed input batch is
    NOT allowed here, so use the row space of W2 itself as a proxy).
    """
    # column energy deviation from median -> suspect trap column
    colnorm = np.linalg.norm(broken.W2, axis=0)
    trap = int(np.argmax(colnorm))
    v = np.zeros(broken.V, dtype=np.float32)
    v[trap] = 1.0
    v = v - v.mean()
    # u proxy: leading right-singular direction of W2 correlated with trap col
    u = broken.W2[:, trap].copy()
    u = u / (np.linalg.norm(u) + 1e-9)
    alpha = 0.0  # cannot calibrate scale without feedback
    return u.astype(np.float32), v.astype(np.float32), alpha


def estimate_direction_feedback(broken: CharLM, x_dev: np.ndarray, budget: int,
                                seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """b>0: use `budget` dev forward passes to estimate u, v and alpha.

    - v_hat: character that is over-produced in generated dev samples
    - u_hat: mean hidden activation on dev (one forward over x_dev)
    - alpha_hat: line search over a grid, scored by dev collapse reduction
    """
    u = mean_hidden_direction(broken, x_dev)
    gen = broken.generate(n_chars=80, temperature=0.6, seed=seed)
    counts = np.bincount(gen, minlength=broken.V).astype(float)
    trap = int(np.argmax(counts))
    v = np.zeros(broken.V, dtype=np.float32)
    v[trap] = 1.0
    v = v - v.mean()

    # collapse score on a fixed dev probe; lower is better
    def score(alpha_hat: float) -> float:
        g = broken.generate(n_chars=60, temperature=0.6, seed=seed + 7)
        cand = broken.clone()
        cand.W2 = (cand.W2 - alpha_hat * np.outer(u, v)).astype(np.float32)
        g2 = cand.generate(n_chars=60, temperature=0.6, seed=seed + 7)
        return float(np.mean([i == trap for i in g2]) - np.mean([i == trap for i in g]))

    grid = np.linspace(0, 40, max(2, budget))
    best_a, best_s = 0.0, 1e9
    for a in grid[:budget]:
        s = score(float(a))
        if s < best_s:
            best_s, best_a = s, float(a)
    return u.astype(np.float32), v.astype(np.float32), best_a


# ---------------------------------------------------------------------------
# Edit methods (all return a NEW model; never mutate the broken baseline)
# ---------------------------------------------------------------------------

def edit_oracle(broken: CharLM, gt: dict) -> CharLM:
    m = broken.clone()
    m.W2 = (m.W2 - gt["D"]).astype(np.float32)
    return m


def edit_weight_only(broken: CharLM, x_dev: np.ndarray) -> CharLM:
    u, v, a = estimate_direction_weight_only(broken, x_dev)
    m = broken.clone()
    if a > 0:
        m.W2 = (m.W2 - a * np.outer(u, v)).astype(np.float32)
    return m


def edit_feedback(broken: CharLM, x_dev: np.ndarray, budget: int,
                  seed: int = 0) -> CharLM:
    u, v, a = estimate_direction_feedback(broken, x_dev, budget, seed)
    m = broken.clone()
    m.W2 = (m.W2 - a * np.outer(u, v)).astype(np.float32)
    return m


def edit_random(broken: CharLM, gt: dict, seed: int = 0) -> CharLM:
    """C3: rank-1 edit of matched norm but random direction."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=broken.H).astype(np.float32)
    u /= np.linalg.norm(u) + 1e-9
    v = rng.normal(size=broken.V).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    m = broken.clone()
    m.W2 = (m.W2 - gt["alpha"] * np.outer(u, v)).astype(np.float32)
    return m


def edit_finetune(broken: CharLM, x_dev: np.ndarray, y_dev: np.ndarray,
                  steps: int = 40, lr: float = 3e-3, seed: int = 0) -> CharLM:
    """C5: conventional gradient-based adaptation on a small dev slice."""
    m = broken.clone()
    rng = np.random.default_rng(seed)
    n = min(len(y_dev), 2048)
    idx = rng.choice(len(y_dev), size=n, replace=False)
    xb, yb = x_dev[idx], y_dev[idx]
    m.fit(xb, yb, epochs=max(1, steps // max(1, n // 256)),
          batch=256, lr=lr, seed=seed, verbose=False)
    return m


def collapse_rate(model: CharLM, trap_char: str, seed: int = 0) -> float:
    g = model.generate(n_chars=80, temperature=0.6, seed=seed)
    return float(np.mean([i == CHAR2IDX[trap_char] for i in g]))


if __name__ == "__main__":
    from data import get_dataset
    d = get_dataset("sv", seed=0)
    xd, yd = d["dev"]
    xe, ye = d["eval"]
    base = CharLM(seed=0).fit(d["train"][0], d["train"][1], epochs=8)
    print("base eval acc:", round(base.accuracy(xe, ye), 4))
    broken = base.clone()
    gt = inject_loop(broken, xd, trap_char="a", target_collapse=0.55, seed=0)
    print("injected alpha:", round(gt["alpha"], 3), "collapse:", round(gt["collapse_after"], 3))
    print("broken eval acc:", round(broken.accuracy(xe, ye), 4))
    for name, m in [
        ("oracle", edit_oracle(broken, gt)),
        ("random", edit_random(broken, gt, seed=0)),
        ("feedback b=5", edit_feedback(broken, xd, 5, seed=0)),
        ("feedback b=20", edit_feedback(broken, xd, 20, seed=0)),
    ]:
        print(f"  {name:14s} eval acc={m.accuracy(xe,ye):.4f} collapse={collapse_rate(m,'a'):.3f}")
