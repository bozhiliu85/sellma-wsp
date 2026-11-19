# Results

Candidate assignment — Weight-Space Programming for SE.LLMA

This file holds the evaluation examples, outcomes, and configuration. The
hypothesis and design are in `RESEARCH_NOTE.md`; the PDF made from that note is
`REPORT.pdf`. Raw machine-readable data is in `results/`.

## Setup and configuration

| Item | Value |
|---|---|
| Model | Tiny character-level transformer trained from scratch, no pretrained weights |
| Architecture | 4 attention layers, hidden 64, context 8, vocab 49 |
| Training | 10 epochs, synthetic templates, NumPy on CPU |
| Seeds | 0, 1, 2, 3, 4 (5 independent runs per condition) |
| Model revision | Local implementation in `code/model.py`; no external checkpoint |
| Injected failure | rank-1 direction into `W2`, trap character `a`, strength calibrated on dev |
| Selection | dev only; frozen Swedish eval touched once at final scoring |
| Software | Python 3.13.14, NumPy 2.5.5, Windows 11 |
| Data rights | Synthetic data generated locally; no external corpus or model weights |

Files: `results/metrics.csv` (all 80 rows), `results/config.yaml`,
`results/examples.md` (full generations), `results/c4_decode_control.csv`.

## Outcomes

Swedish held-out next-token accuracy, 5 seeds, mean ± sd:

| Condition | A: Swedish-trained | B: English-trained |
|---|---|---|
| C0 base | 0.8365 ± 0.0017 | 0.1641 ± 0.0075 |
| C1 injected, no edit | 0.4963 ± 0.0773 | 0.1314 ± 0.0149 |
| C2 oracle | 0.8365 ± 0.0017 | 0.1641 ± 0.0075 |
| C2 weight-only, b=0 | 0.4963 ± 0.0773 | 0.1314 ± 0.0149 |
| C2 feedback, b=5 | 0.8004 ± 0.0048 | 0.1658 ± 0.0066 |
| C2 feedback, b=20 | 0.8016 ± 0.0059 | 0.1661 ± 0.0069 |
| C3 random | 0.4930 ± 0.0826 | 0.1295 ± 0.0132 |
| C5 fine-tuning | 0.8354 ± 0.0031 | 0.1692 ± 0.0087 |

Decoding-only control (C4), no weight change, measured on the broken model:

| Setting | Loop rate | distinct-2 |
|---|---|---|
| Broken, no penalty | 0.4667 | 0.3529 |
| Broken, rep_penalty=3 | 0.0833 | 0.8235 |
| Broken, rep_penalty=6 | 0.0667 | 0.9160 |

Accuracy stays damaged under the penalty; only the surface changes.

## Evaluation examples

Condition A, seed 0, Swedish eval generations (full set in `results/examples.md`):

- **C0 base** (acc 0.8395): `nnagliser mat i  öteborg imorgon.gli sk ia ton ui ser en bil i  tockholm snabbt...`
- **C1 injected** (acc 0.4163): `aaaagrraan aaleaa.aaanraaa taalar aalealhar enabaa etaraaaträa...`
- **C2 oracle** (acc 0.8395): identical to C0 — removing the injected direction restores the base output.
- **C2 feedback b=20** (acc 0.8086): `nngilln sägbie i skolan långskmik o skrimnr nnskoi e lltt...`

The broken model collapses to the trap character `a`; the oracle removes it
exactly; the feedback edit recovers most of the loss without seeing the ground
truth.

## Supplementary: real open-weight model check (exploratory)

To check whether the effect is an artifact of the tiny model, the same rank-1
injection and edits were also run on two open-weight transformers, via
`hf-mirror.com`. This was exploratory and does **not** enter the main conclusion.

| Item | Value |
|---|---|
| Models | Qwen/Qwen2.5-0.5B (A, multilingual) and HuggingFaceTB/SmolLM2-135M (B, English-centric) |
| Injection | rank-1 delta on last-layer `mlp.down_proj`, alpha calibrated on dev |
| Seeds | 0, 1, 2 (3 per condition); 7 conditions × 2 models = 42 rows |
| Software | Python 3.13.14, torch 2.15.0+cpu, transformers 5.23.0 |
| Selection | dev only; eval touched only at final scoring |

Swedish next-token accuracy, mean ± sd (A / B):

| Condition | A: Qwen2.5-0.5B | B: SmolLM2-135M |
|---|---|---|
| C0 base | 0.2035 ± 0.0000 | 0.2023 ± 0.0000 |
| C1 broken | 0.1361 ± 0.0000 | 0.1189 ± 0.0000 |
| C2 oracle | 0.2035 (full recovery) | 0.2023 (full recovery) |
| C3 random | 0.2032 ± 0.0023 | 0.2031 ± 0.0009 |
| C5 feedback b=5 | 0.1903 ± 0.0345 | 0.2011 ± 0.0050 |
| C5 feedback b=20 | 0.1914 ± 0.0307 | 0.1994 ± 0.0035 |
| C4 decode(broken) | 0.1361 (unchanged) | 0.1189 (unchanged) |

**What held up.** Oracle removal restored the base score exactly on real
weights; the decode-only penalty lifted distinct-2 to 1.000 with accuracy
unchanged (surface repair, not weight repair); feedback edits were unstable
when the collapse token was guessed wrong (A sd ≈ 0.03).

**Where it fell short of the plan.** The intended A/B ability contrast did not
materialise: both models' base accuracy sat at ~0.20 — the ceiling of the
synthetic templates under BPE tokenisation, not a measure of language ability —
and a perplexity-ratio probe (SV/EN = 4.30 for A vs 4.06 for B) did not separate
them either. The "cannot create ability" half of the hypothesis therefore has
no evidence from this check; that evidence still comes only from the controlled
toy experiment. Reported honestly as a limitation, not as a result.

Raw data: `results/real_model_metrics.csv`, `results/real_model_config.yaml`,
`results/real_model_examples.md`.
