# Weight editing for Swedish: a small controlled experiment

SE.LLMA candidate assignment — Weight-Space Programming

A small, CPU-only experiment testing one narrow claim from *Sparks of
Weight-Space Programming*: does a targeted weight edit **restore an existing
ability** or **create a new one**? We test it on synthetic Swedish data.

The whole thing runs in about 20 seconds and needs only Python and NumPy. No
model downloads, no paid compute, no other packages.

## What is in this repository

| File | What it is |
|---|---|
| `code/` + this file | Code: implementation and how to reproduce the experiment |
| `RESULTS.md` + `results/` | Results: examples, outcomes, config, seeds; includes an exploratory real-model check (`results/real_model_*`) |
| `RESEARCH_NOTE.md` | Research note, concise (~1–2 pages): hypothesis, experiment, results, interpretation, recommendation |
| `REPORT.md` + `REPORT.pdf` | Report source and its PDF: expands the note with more background, result reading and a related commentary |
| `AI_USAGE.md` | Short note on AI tools used |
| `RESEARCH_NOTE_CN.md`, `REPORT_CN.md` | Chinese versions, for reading only (not part of the submission) |

## Run it in VSCode

**1. Open the folder.** In VSCode: `File > Open Folder...` and select this
`sellma-wsp` folder.

**2. Select a Python interpreter.** Open any `.py` file in `code/`, then press
`Ctrl+Shift+P`, type `Python: Select Interpreter`, and pick your Python (3.10+
is fine). If VSCode offers to install the Python extension, accept it.

**3. Check NumPy is installed.** Open the integrated terminal with the backtick
key (`` ` ``) and run:

```
python -c "import numpy; print(numpy.__version__)"
```

If it says "No module named numpy", install it with `python -m pip install numpy`.
That is the only dependency.

**4. Run the main experiment.** In the terminal:

```
cd code
python run_experiment.py --all --seeds 5
```

It trains the models, applies the edits, and prints a summary table. About 20
seconds.

**5. Run the decoding control** (optional, a few seconds):

```
python c4_decode_control.py
```

**6. Look at the output.** Results are written to the `results/` folder one
level up:

- `metrics.csv` — all conditions × seeds, with accuracy, loop rate and distinct-2
- `config.yaml` — seeds and settings used
- `examples.md` — example Swedish text before, after damage, and after the edit
- `c4_decode_control.csv` — the decoding-only control

Double-click any of these in VSCode's file explorer to open them.

## What the code does, briefly

- `data.py` — builds the synthetic Swedish / English sentences and splits them
  into train / dev / eval (no overlap).
- `model.py` — a tiny character-level transformer in NumPy: training,
  generation, evaluation.
- `edit.py` — injects a known "collapse" direction into the weights, then
  implements the edits (oracle removal, random edit, feedback-based estimate).
- `evaluate.py` — accuracy and generation-quality metrics.
- `run_experiment.py` — runs the full condition matrix across 5 seeds.
- `c4_decode_control.py` — the decoding-only (repetition penalty) control.

Settings: 5 seeds (0–4), 4-layer model, context length 8, 10 epochs. See
`results/config.yaml` for the exact values.

## A note on `--quick`

`python run_experiment.py --quick` is a 1-seed smoke test (~5 s) for checking
the code runs. It writes to the same `results/metrics.csv`, so it will overwrite
the full 5-seed data. **For the reported numbers, always use
`python run_experiment.py --all --seeds 5`.** The `metrics.csv` shipped here is
the full 5-seed run (80 rows).

## Result in one line

A targeted edit restores Swedish accuracy from 0.50 back to 0.84 when the
ability was there, but cannot lift a model that never learned Swedish above its
own 0.16 ceiling — editing is a repair tool, not a way to teach a language.
