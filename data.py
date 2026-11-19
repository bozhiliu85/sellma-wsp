"""Synthetic corpora and disjoint train/dev/eval splits.

Only synthetic data is used, so data rights are unambiguous and the
experiment stays CPU-only and reproducible. We build three corpora with a
shared character vocabulary:

  - sv : synthetic Swedish sentences
  - en : synthetic English sentences
  - code : synthetic Python-like snippets

The key property we need: a model trained on `sv` reaches high next-char
accuracy on held-out Swedish, while a model trained on `en` stays low on
Swedish. That gap is the "ability" that a weight edit can only restore
(condition A) but never create (condition B).
"""

from __future__ import annotations

import numpy as np

# Shared character vocabulary. Index 0 is reserved as the BOS/pad token.
_ALPHABET = list("abcdefghijklmnopqrstuvwxyzåäö0123456789 _.,:;()\n")

CHARS = ["<BOS>"] + _ALPHABET
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS)
BOS = 0


# ---------------------------------------------------------------------------
# Synthetic corpora. Built from templates + word lists so that the char-level
# statistics of Swedish and English differ strongly (å/ä/ö, word shapes).
# ---------------------------------------------------------------------------

_SV_SUBJ = ["jag", "du", "han", "hon", "vi", "ni", "de", "en man", "en kvinna", "ett barn"]
_SV_VERB = ["är", "har", "går", "läser", "skriver", "ser", "gillar", "köper", "säger", "tänker"]
_SV_OBJ = ["en bok", "ett hus", "en bil", "kaffe", "mat", "vatten", "ett brev", "en vän", "musik", "ett äpple"]
_SV_PLACE = ["i Sverige", "i Stockholm", "här", "där", "i skolan", "på jobbet", "i trädgården", "i Göteborg"]
_SV_ADV = ["idag", "igår", "imorgon", "ofta", "aldrig", "alltid", "lite", "mycket", "snabbt", "långsamt"]

_EN_SUBJ = ["i", "you", "he", "she", "we", "they", "a man", "a woman", "a child"]
_EN_VERB = ["am", "have", "go", "read", "write", "see", "like", "buy", "say", "think"]
_EN_OBJ = ["a book", "a house", "a car", "coffee", "food", "water", "a letter", "a friend", "music", "an apple"]
_EN_PLACE = ["in Sweden", "in London", "here", "there", "at school", "at work", "in the garden", "in York"]
_EN_ADV = ["today", "yesterday", "tomorrow", "often", "never", "always", "a little", "a lot", "quickly", "slowly"]

_CODE_NAMES = ["x", "y", "n", "total", "count", "items", "value", "result", "index", "score"]
_CODE_OPS = ["+", "-", "*", "//", "%"]


def _sv_sentences(n: int, rng: np.random.Generator) -> list[str]:
    out = []
    for _ in range(n):
        parts = [
            rng.choice(_SV_SUBJ).capitalize(),
            rng.choice(_SV_VERB),
            rng.choice(_SV_OBJ),
            rng.choice(_SV_PLACE),
            rng.choice(_SV_ADV),
        ]
        out.append(" ".join(parts) + ".")
    return out


def _en_sentences(n: int, rng: np.random.Generator) -> list[str]:
    out = []
    for _ in range(n):
        parts = [
            rng.choice(_EN_SUBJ).capitalize(),
            rng.choice(_EN_VERB),
            rng.choice(_EN_OBJ),
            rng.choice(_EN_PLACE),
            rng.choice(_EN_ADV),
        ]
        out.append(" ".join(parts) + ".")
    return out


def _code_sentences(n: int, rng: np.random.Generator) -> list[str]:
    out = []
    for _ in range(n):
        a, b, c = rng.choice(_CODE_NAMES, 3, replace=True)
        op = rng.choice(_CODE_OPS)
        k = int(rng.integers(0, 10))
        templates = [
            f"{a} = {b} {op} {c}",
            f"if {a} > {k}: print({b})",
            f"for {a} in {b}: {c} += 1",
            f"total = sum({a}) + {k}",
            f"result.append({b})",
        ]
        out.append(rng.choice(templates))
    return out


_BUILDERS = {"sv": _sv_sentences, "en": _en_sentences, "code": _code_sentences}


def build_corpus(lang: str, n_sentences: int, seed: int) -> list[str]:
    """Return a list of sentences for the requested language."""
    if lang not in _BUILDERS:
        raise ValueError(f"unknown lang {lang!r}")
    rng = np.random.default_rng(seed)
    return _BUILDERS[lang](n_sentences, rng)


def split_sentences(sentences: list[str], seed: int, frac_dev: float = 0.15, frac_eval: float = 0.15):
    """Split by whole sentences (no char leakage) into train/dev/eval lists."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(sentences))
    n_eval = int(len(idx) * frac_eval)
    n_dev = int(len(idx) * frac_dev)
    eval_idx = idx[:n_eval]
    dev_idx = idx[n_eval : n_eval + n_dev]
    train_idx = idx[n_eval + n_dev :]
    return (
        [sentences[i] for i in train_idx],
        [sentences[i] for i in dev_idx],
        [sentences[i] for i in eval_idx],
    )


def encode(sentences: list[str]) -> np.ndarray:
    """Encode sentences to an int array of token ids, inserting BOS per line."""
    ids = [BOS]
    for s in sentences:
        for ch in s:
            ids.append(CHAR2IDX.get(ch, CHAR2IDX[" "]))
        ids.append(BOS)
    return np.asarray(ids, dtype=np.int64)


def make_pairs(ids: np.ndarray, context: int):
    """Build (x, y) pairs: x = context ids, y = next id. BOS skips as target."""
    xs, ys = [], []
    for i in range(context, len(ids)):
        y = ids[i]
        if y == BOS:
            continue
        xs.append(ids[i - context : i])
        ys.append(y)
    return np.asarray(xs, dtype=np.int64), np.asarray(ys, dtype=np.int64)


def get_dataset(lang: str, seed: int, n_sentences: int = 1200, context: int = 8):
    """Convenience: build, split, encode, and vectorize one language corpus."""
    sents = build_corpus(lang, n_sentences, seed)
    tr, dv, ev = split_sentences(sents, seed)
    tr_ids, dv_ids, ev_ids = encode(tr), encode(dv), encode(ev)
    xtr, ytr = make_pairs(tr_ids, context)
    xdv, ydv = make_pairs(dv_ids, context)
    xev, yev = make_pairs(ev_ids, context)
    return {
        "train": (xtr, ytr),
        "dev": (xdv, ydv),
        "eval": (xev, yev),
        "sentences": {"train": tr, "dev": dv, "eval": ev},
    }


if __name__ == "__main__":
    for lang in ("sv", "en", "code"):
        d = get_dataset(lang, seed=0)
        print(lang, VOCAB_SIZE, "vocab;",
              "train/dev/eval pairs:",
              len(d["train"][1]), len(d["dev"][1]), len(d["eval"][1]))
        print("  sample:", d["sentences"]["train"][0])
