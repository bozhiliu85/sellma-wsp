"""C4 decoding control: can changing inference settings bypass the failure?

Next-token accuracy is argmax-based and cannot be changed by decoding params,
so C4 is assessed on GENERATION quality (loop rate / distinct-2): if a simple
repetition penalty removes the visible looping without touching weights, then
part of the "apparent improvement" is a decoding artifact, not a weight edit.

This mirrors the paper's own caveat that turning thinking off raises scores
without fixing the weights.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from data import get_dataset, CHAR2IDX
from edit import inject_loop
from evaluate import distinct_n
from model import CharLM, ids_to_text, softmax

RESULTS = Path(__file__).resolve().parent.parent / "results"
TRAP = "a"


def generate_with_penalty(model, n, penalty, seed):
    """Greedy-ish sampling with a repetition penalty on already-seen tokens."""
    rng = np.random.default_rng(seed)
    ctx = [0] * model.C
    out = []
    seen = {}
    for _ in range(n):
        z = model.logits(np.asarray(ctx)[None, :])[0] / 0.8
        for tok, cnt in seen.items():
            z[tok] -= penalty * cnt
        p = softmax(z)
        nxt = int(rng.choice(model.V, p=p))
        out.append(nxt)
        seen[nxt] = seen.get(nxt, 0) + 1
        ctx = ctx[1:] + [nxt]
    return out


def main():
    d = get_dataset("sv", seed=0)
    xd, yd = d["dev"]
    base = CharLM(seed=0).fit(d["train"][0], d["train"][1], epochs=10, seed=0)
    broken = base.clone()
    inject_loop(broken, xd, trap_char=TRAP, target_collapse=0.5, seed=0)

    rows = []
    for label, model, pen in [
        ("broken, no penalty", broken, 0.0),
        ("broken, rep_penalty=1", broken, 1.0),
        ("broken, rep_penalty=3", broken, 3.0),
        ("broken, rep_penalty=6", broken, 6.0),
    ]:
        gen = generate_with_penalty(model, 120, pen, seed=0)
        rows.append({
            "setting": label,
            "loop_rate": round(float(np.mean([i == CHAR2IDX[TRAP] for i in gen])), 4),
            "distinct2": round(distinct_n(gen, 2), 4),
            "sample": ids_to_text(gen)[:120],
        })

    with open(RESULTS / "c4_decode_control.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["setting", "loop_rate", "distinct2", "sample"])
        w.writeheader(); w.writerows(rows)

    print("C4 decoding control (generation quality):")
    for r in rows:
        print(f"  {r['setting']:24s} loop={r['loop_rate']:.3f} distinct2={r['distinct2']:.3f}")
    print("Wrote results/c4_decode_control.csv")


if __name__ == "__main__":
    main()
