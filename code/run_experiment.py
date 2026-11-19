"""Main experiment runner: C0-C7 control matrix x Condition A/B x seeds.

Condition A: base model trained on Swedish -> has Swedish ability.
             We inject a collapse, then test whether edits RESTORE it.
Condition B: base model trained on English (no Swedish ability) -> we inject
             the same collapse, then test whether edits CREATE Swedish ability.
             (This is the decisive control for H1.)

All selection/estimation uses dev only. eval is frozen and touched only at
the final scoring step.

Usage:
    python run_experiment.py --quick              # 1 seed, smoke
    python run_experiment.py --all --seeds 5      # full run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from pathlib import Path

import numpy as np

from data import get_dataset, VOCAB_SIZE
from edit import (inject_loop, edit_oracle, edit_random, edit_feedback,
                  edit_finetune, edit_weight_only)
from evaluate import evaluate
from model import CharLM

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"

TRAP = "a"
CONTEXT = 8
HIDDEN = 64
EPOCHS = 10
SEEDS_DEFAULT = 5


def build_base(train_lang: str, seed: int):
    """Train one base model on `train_lang`; return (model, sv_eval_data, dev)."""
    d = get_dataset(train_lang, seed=seed, context=CONTEXT)
    model = CharLM(vocab=VOCAB_SIZE, context=CONTEXT, hidden=HIDDEN, seed=seed)
    model.fit(d["train"][0], d["train"][1], epochs=EPOCHS, seed=seed, verbose=False)
    return model, d


def one_condition(base: CharLM, d: dict, seed: int) -> list[dict]:
    """Run C0-C7 for a single base model, scoring on Swedish eval."""
    xe, ye = d["eval"]                 # frozen Swedish eval pairs
    xd, yd = d["dev"]
    rows: list[dict] = []

    # ---- C0: untouched base (ability ceiling for condition A) ----
    rows.append({"condition": "C0_base", **evaluate(base, xe, ye, TRAP, seed)})

    # inject collapse on a clone
    broken = base.clone()
    gt = inject_loop(broken, xd, trap_char=TRAP, target_collapse=0.5, seed=seed)
    rows.append({"condition": "C1_broken", **evaluate(broken, xe, ye, TRAP, seed),
                 "alpha": round(gt["alpha"], 3)})

    # ---- C2 oracle (ground-truth weight reasoning, upper bound) ----
    rows.append({"condition": "C2_oracle", **evaluate(edit_oracle(broken, gt), xe, ye, TRAP, seed)})

    # ---- C2 weight-only (b=0: no forward feedback) ----
    rows.append({"condition": "C2_weight_only_b0", **evaluate(edit_weight_only(broken, xd), xe, ye, TRAP, seed)})

    # ---- C2 feedback budget ablation (b=5 / b=20) ----
    for b in (5, 20):
        m = edit_feedback(broken, xd, budget=b, seed=seed)
        rows.append({"condition": f"C2_feedback_b{b}", **evaluate(m, xe, ye, TRAP, seed)})

    # ---- C3 random matched-norm edit ----
    rows.append({"condition": "C3_random", **evaluate(edit_random(broken, gt, seed=seed), xe, ye, TRAP, seed)})

    # ---- C5 conventional finetune ----
    rows.append({"condition": "C5_finetune", **evaluate(edit_finetune(broken, xd, yd, seed=seed), xe, ye, TRAP, seed)})

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=SEEDS_DEFAULT)
    args = ap.parse_args()

    seeds = [0] if args.quick else list(range(args.seeds))
    print(f"Running conditions A/B x {len(seeds)} seed(s) = {2*len(seeds)} base models...")

    all_rows = []
    for cond, train_lang in (("A", "sv"), ("B", "en")):
        for seed in seeds:
            print(f"  condition {cond} seed {seed} training base on '{train_lang}' ...")
            base, d = build_base(train_lang, seed)
            # Condition B: evaluate on SWEDISH eval (ability absent)
            if cond == "B":
                sv_eval = get_dataset("sv", seed=seed, context=CONTEXT)
                d = dict(d)
                d["eval"] = sv_eval["eval"]       # swap in Swedish eval, frozen
            for r in one_condition(base, d, seed):
                r["condition_group"] = cond
                r["seed"] = seed
                all_rows.append(r)
            base_acc = all_rows[-1]  # noop, keep structure simple

    RESULTS.mkdir(parents=True, exist_ok=True)
    fields = ["condition_group", "seed", "condition", "target_acc", "loop_rate", "distinct2", "alpha", "sample"]
    with open(RESULTS / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    # config with versions / seeds / setup
    cfg = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "os": platform.platform(),
        "seeds": seeds,
        "vocab_size": VOCAB_SIZE,
        "context": CONTEXT,
        "hidden": HIDDEN,
        "epochs": EPOCHS,
        "trap_char": TRAP,
        "condition_A_train": "synthetic Swedish",
        "condition_B_train": "synthetic English",
        "eval_language": "synthetic Swedish (frozen, held-out)",
        "selection": "dev only; eval touched only at final scoring",
        "data_rights": "synthetic data generated locally; no external corpus",
        "note": "C4 decoding control is assessed qualitatively in examples.md",
    }
    with open(RESULTS / "config.yaml", "w", encoding="utf-8") as f:
        for k, v in cfg.items():
            f.write(f"{k}: {json.dumps(v, ensure_ascii=False)}\n")

    # qualitative examples (one seed)
    ex = [r for r in all_rows if r["seed"] == seeds[0]]
    with open(RESULTS / "examples.md", "w", encoding="utf-8") as f:
        f.write("# Qualitative examples (seed " + str(seeds[0]) + ")\n\n")
        f.write("Swedish eval generation samples per condition. "
                "`a` collapse is injected as the controlled failure.\n\n")
        for cond in ("A", "B"):
            f.write(f"## Condition {cond}\n\n")
            for r in ex:
                if r["condition_group"] == cond:
                    f.write(f"**{r['condition']}** (acc={r['target_acc']}, "
                            f"loop={r['loop_rate']}, distinct2={r['distinct2']}):\n\n")
                    f.write(f"`{r.get('sample','')}`\n\n")

    # summary to stdout
    print("\n=== Summary: target_acc (Swedish eval) ===")
    for cond in ("A", "B"):
        print(f"\nCondition {cond}:")
        keys = ["C0_base", "C1_broken", "C2_oracle", "C2_weight_only_b0",
                "C2_feedback_b5", "C2_feedback_b20", "C3_random", "C5_finetune"]
        for k in keys:
            vals = [r["target_acc"] for r in all_rows
                    if r["condition_group"] == cond and r["condition"] == k]
            if vals:
                print(f"  {k:20s} mean={np.mean(vals):.4f} sd={np.std(vals):.4f}")
    print(f"\nWrote {RESULTS/'metrics.csv'}, config.yaml, examples.md")


if __name__ == "__main__":
    main()
