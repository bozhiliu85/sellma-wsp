"""Tiny character-level autoregressive LM, pure numpy, CPU-only.

Architecture (an MLP language model over a fixed context window):
    x = one-hot(context chars)            shape (N, C*V)
    h = ReLU(x @ W1 + b1)                 shape (N, H)
    logits = h @ W2 + b2                  shape (N, V)

The matrix W2 (hidden -> vocab) is where we inject / remove the controlled
rank-1 "loop direction", mirroring the paper's MLP output-column edits.

Zero extra dependencies. Deterministic given a seed.
"""

from __future__ import annotations

import numpy as np

from data import VOCAB_SIZE, CHAR2IDX, CHARS, BOS


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def one_hot(x: np.ndarray, v: int) -> np.ndarray:
    """(N,) int ids -> (N, V) float one-hot."""
    out = np.zeros((x.shape[0], v), dtype=np.float32)
    out[np.arange(x.shape[0]), x] = 1.0
    return out


class CharLM:
    def __init__(self, vocab: int = VOCAB_SIZE, context: int = 8, hidden: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.V = vocab
        self.C = context
        self.H = hidden
        # Heinit scaling for stable training
        self.W1 = (rng.normal(0, 0.6, size=(context * vocab, hidden))).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = (rng.normal(0, 0.6, size=(hidden, vocab))).astype(np.float32)
        self.b2 = np.zeros(vocab, dtype=np.float32)

    # ---------- forward ----------
    def hidden(self, x: np.ndarray) -> np.ndarray:
        """x: (N, C) int ids -> h: (N, H)."""
        xh = one_hot(x[:, 0], self.V)
        for i in range(1, self.C):
            xh = np.concatenate([xh, one_hot(x[:, i], self.V)], axis=1)
        return np.maximum(0.0, xh @ self.W1 + self.b1)

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self.hidden(x) @ self.W2 + self.b2

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        z = self.logits(x)
        p = softmax(z)
        return float(-np.log(p[np.arange(len(y)), y] + 1e-9).mean())

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        if len(y) == 0:
            return float("nan")
        z = self.logits(x)
        return float((z.argmax(axis=1) == y).mean())

    # ---------- training ----------
    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 12, batch: int = 256,
            lr: float = 8e-3, seed: int = 0, verbose: bool = False):
        """Adam mini-batch training on (x, y) pairs."""
        rng = np.random.default_rng(seed)
        params = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1t, b2t, eps = 0.9, 0.999, 1e-8
        n = len(y)
        step = 0
        for ep in range(epochs):
            perm = rng.permutation(n)
            tot_loss = 0.0
            for s in range(0, n, batch):
                idx = perm[s : s + batch]
                xb, yb = x[idx], y[idx]
                # forward
                oh = one_hot(xb[:, 0], self.V)
                for i in range(1, self.C):
                    oh = np.concatenate([oh, one_hot(xb[:, i], self.V)], axis=1)
                h = np.maximum(0.0, oh @ self.W1 + self.b1)
                z = h @ self.W2 + self.b2
                p = softmax(z)
                B = len(yb)
                tot_loss += -np.log(p[np.arange(B), yb] + 1e-9).mean() * B
                # backward
                dz = p
                dz[np.arange(B), yb] -= 1.0
                dz /= B
                gW2 = h.T @ dz
                gb2 = dz.sum(axis=0)
                dh = (dz @ self.W2.T) * (h > 0)
                gW1 = oh.T @ dh
                gb1 = dh.sum(axis=0)
                grads = [gW1, gb1, gW2, gb2]
                step += 1
                for k, g in enumerate(grads):
                    m[k] = b1t * m[k] + (1 - b1t) * g
                    v[k] = b2t * v[k] + (1 - b2t) * (g * g)
                    mh = m[k] / (1 - b1t ** step)
                    vh = v[k] / (1 - b2t ** step)
                    params[k] -= lr * mh / (np.sqrt(vh) + eps)
            if verbose:
                print(f"  epoch {ep+1:02d} loss={tot_loss/n:.4f}")
        return self

    # ---------- generation ----------
    def generate(self, n_chars: int = 160, temperature: float = 0.7,
                 seed: int = 0, prompt: list[int] | None = None) -> list[int]:
        """Autoregressive sampling. Returns list of token ids (no BOS)."""
        rng = np.random.default_rng(seed)
        ctx = list(prompt[-self.C:]) if prompt else [BOS] * self.C
        out: list[int] = []
        for _ in range(n_chars):
            z = self.logits(np.asarray(ctx, dtype=np.int64)[None, :])[0]
            z = z / max(temperature, 1e-6)
            p = softmax(z)
            nxt = int(rng.choice(self.V, p=p))
            out.append(nxt)
            ctx = ctx[1:] + [nxt]
        return out

    # ---------- state dict ----------
    def state(self) -> dict:
        return {
            "W1": self.W1.copy(), "b1": self.b1.copy(),
            "W2": self.W2.copy(), "b2": self.b2.copy(),
        }

    def load_state(self, s: dict):
        self.W1 = s["W1"].copy(); self.b1 = s["b1"].copy()
        self.W2 = s["W2"].copy(); self.b2 = s["b2"].copy()
        return self

    def clone(self) -> "CharLM":
        m = CharLM(self.V, self.C, self.H, seed=0)
        return m.load_state(self.state())


def ids_to_text(ids: list[int]) -> str:
    return "".join(CHARS[i] for i in ids)


if __name__ == "__main__":
    from data import get_dataset
    d = get_dataset("sv", seed=0)
    xt, yt = d["train"]
    xe, ye = d["eval"]
    m = CharLM(seed=0)
    print("before:", round(m.accuracy(xe, ye), 4))
    m.fit(xt, yt, epochs=8, verbose=True)
    print("after :", round(m.accuracy(xe, ye), 4))
    print("sample:", repr(ids_to_text(m.generate(120, seed=1))[:120]))
