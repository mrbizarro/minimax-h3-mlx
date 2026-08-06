"""Is the range-fetched ``time_embedder`` the one DeepBeepMeep's pruned curve was built from?

DeepBeepMeep built ``adaln_t_table`` (1001 x 64) by taking the true modulation ``M(t)`` on a 1001-
point grid, centering it and keeping the top-64 SVD components. ``M(t)`` is *affine* in
``silu(temb(t))``, so the published curve must satisfy ``C = S P`` exactly for some ``P``, where
``S`` is the centered ``silu(temb(t))`` over the same grid. That means **every column of C lies in
the column space of S**.

``S`` has 2688 columns against 1001 grid rows, so an unrestricted fit is underdetermined and proves
nothing. The test that does discriminate: ``silu(temb(t))`` traces a smooth 1-D curve whose centered
Gram matrix has a fast-decaying spectrum, so ``col(S)`` is effectively spanned by its first ``k``
left singular vectors for ``k`` far below 1001. Projecting ``C`` onto that k-dimensional subspace
must recover it almost exactly if the embedder is right, and must not if it is wrong.

Two controls make the positive result mean something: the raw sinusoid with no MLP, and the same
MLP with shuffled weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np

from lora_numerics import SafeReader


def sinusoid(t: np.ndarray, dim: int = 256, max_period: float = 10000.0) -> np.ndarray:
    """Matches ``minimax_h3_mlx.dit.timestep_embedding`` (flip_sin_to_cos=True)."""
    half = dim // 2
    exponent = -np.log(max_period) * np.arange(half, dtype=np.float64) / half
    emb = t[:, None] * np.exp(exponent)[None, :]
    return np.concatenate([np.cos(emb), np.sin(emb)], axis=-1).astype(np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def subspace_r2(c: np.ndarray, s: np.ndarray, ks: list[int]) -> list[tuple[int, float]]:
    """Fraction of ``c``'s energy captured by the top-k left singular subspace of ``s``."""
    cc = c - c.mean(0, keepdims=True)
    ss = s - s.mean(0, keepdims=True)
    u, sv, _ = np.linalg.svd(ss, full_matrices=False)
    total = float((cc**2).sum())
    out = []
    for k in ks:
        proj = u[:, :k] @ (u[:, :k].T @ cc)
        out.append((k, 1.0 - float(((cc - proj) ** 2).sum()) / total))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dit", required=True)
    ap.add_argument("--embedder", required=True)
    ap.add_argument("--ks", type=int, nargs="*", default=[16, 32, 64, 96, 128, 200])
    args = ap.parse_args()

    dit = SafeReader(args.dit)
    table = np.asarray(dit.get("adaln_t_table").astype(mx.float32), dtype=np.float64)
    grid = table.shape[0]
    t = np.linspace(0.0, 1.0, grid)

    emb = SafeReader(args.embedder)
    w_in = np.asarray(emb.get("time_embedder.proj_in.weight"), dtype=np.float32)
    b_in = np.asarray(emb.get("time_embedder.proj_in.bias"), dtype=np.float32)
    w_out = np.asarray(emb.get("time_embedder.proj_out.weight"), dtype=np.float32)
    b_out = np.asarray(emb.get("time_embedder.proj_out.bias"), dtype=np.float32)

    sin = sinusoid(t)
    temb = silu(sin @ w_in.T + b_in) @ w_out.T + b_out      # TimestepEmbedder: proj_out(silu(proj_in))
    s_true = silu(temb).astype(np.float64)                  # AdaLN consumes silu(temb)

    rng = np.random.default_rng(0)
    w_in_s = w_in.copy().ravel(); rng.shuffle(w_in_s); w_in_s = w_in_s.reshape(w_in.shape)
    w_out_s = w_out.copy().ravel(); rng.shuffle(w_out_s); w_out_s = w_out_s.reshape(w_out.shape)
    s_shuf = silu(silu(sin @ w_in_s.T + b_in) @ w_out_s.T + b_out).astype(np.float64)

    print(f"grid {grid}, curve rank {table.shape[1]}")
    for name, s in (("recovered time_embedder", s_true),
                    ("control: raw sinusoid only", sin.astype(np.float64)),
                    ("control: shuffled MLP weights", s_shuf)):
        row = subspace_r2(table, s, args.ks)
        print(f"{name:<32} " + "  ".join(f"k={k}:{r:.6f}" for k, r in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
