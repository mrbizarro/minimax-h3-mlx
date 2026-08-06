"""Which row order does each checkpoint use for the fused qkv / SwiGLU projections?

Our DiT reads ``attn.qkv_proj`` as **per-head interleaved** ``(heads, 3, head_dim)`` (that is the
MiniMaxAI release layout the port was validated against). larryvrh's reference generator, which runs
the *ComfyUI* model definition, does ``qkv_proj(x).split(heads*head_dim, -1)`` — a contiguous
``(3, heads, head_dim)`` split. If the turbo LoRA was trained through the ComfyUI layout, its
``qkv_proj.lora_B`` rows are in the other order and applying them unpermuted would scramble q/k/v.

The test does not rely on either docstring. q, k and v have systematically different row scales in a
trained model (q/k are followed by an RMSNorm that absorbs their gain, v is not), so the *true*
grouping of the 21504 rows separates their L2 norms and the false grouping does not. The F-statistic
(between-group variance / within-group variance) makes that quantitative.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np

from lora_numerics import SafeReader


def f_stat(values: np.ndarray, groups: np.ndarray, k: int) -> float:
    """One-way ANOVA F for ``values`` split by integer ``groups`` into ``k`` levels."""
    n = values.size
    grand = values.mean()
    between = sum(((values[groups == g].mean() - grand) ** 2) * (groups == g).sum() for g in range(k))
    within = sum(((values[groups == g] - values[groups == g].mean()) ** 2).sum() for g in range(k))
    return float((between / (k - 1)) / (within / (n - k)))


def row_norms(w: mx.array) -> np.ndarray:
    return np.asarray(mx.sqrt(mx.sum(w.astype(mx.float32) ** 2, axis=1)), dtype=np.float64)


def report_qkv(name: str, w: mx.array, heads: int, head_dim: int) -> str:
    rn = row_norms(w)
    idx = np.arange(rn.size)
    per_head = (idx // head_dim) % 3          # (heads, 3, head_dim)
    contiguous = idx // (heads * head_dim)    # (3, heads, head_dim)
    fa, fb = f_stat(rn, per_head, 3), f_stat(rn, contiguous, 3)
    verdict = "(heads,3,hd) PER-HEAD" if fa > fb else "(3,heads,hd) CONTIGUOUS"
    means_a = [rn[per_head == g].mean() for g in range(3)]
    means_b = [rn[contiguous == g].mean() for g in range(3)]
    return (f"{name:<44} F_perhead={fa:12.2f}  F_contig={fb:12.2f}  -> {verdict}\n"
            f"{'':<44} group means per-head  {means_a[0]:.4f} {means_a[1]:.4f} {means_a[2]:.4f}\n"
            f"{'':<44} group means contig    {means_b[0]:.4f} {means_b[1]:.4f} {means_b[2]:.4f}")


def report_halves(name: str, w: mx.array) -> str:
    """fc1 is a fused [gate; value] SwiGLU projection; report whether the halves differ at all."""
    rn = row_norms(w)
    half = rn.size // 2
    return (f"{name:<44} gate-half mean {rn[:half].mean():.5f}  value-half mean {rn[half:].mean():.5f}  "
            f"ratio {rn[:half].mean()/rn[half:].mean():.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dit", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--blocks", type=int, nargs="*", default=[0, 25, 49])
    ap.add_argument("--heads", type=int, default=56)
    ap.add_argument("--head-dim", type=int, default=128)
    args = ap.parse_args()

    dit, lora = SafeReader(args.dit), SafeReader(args.lora)
    print("=== base checkpoint (deepbeep pruned, MiniMaxAI lineage) ===")
    for b in args.blocks:
        print(report_qkv(f"blocks.{b}.attn.qkv_proj.weight", dit.get(f"blocks.{b}.attn.qkv_proj.weight"),
                         args.heads, args.head_dim))
    print("\n=== turbo LoRA delta B@A ===")
    for b in args.blocks:
        a = lora.get(f"blocks.{b}.attn.qkv_proj.lora_A.weight").astype(mx.float32)
        bb = lora.get(f"blocks.{b}.attn.qkv_proj.lora_B.weight").astype(mx.float32)
        print(report_qkv(f"blocks.{b}.attn.qkv_proj  delta", bb @ a, args.heads, args.head_dim))
        del a, bb
        mx.clear_cache()
    print("\n=== turbo LoRA B alone (row order is B's own) ===")
    for b in args.blocks:
        bb = lora.get(f"blocks.{b}.attn.qkv_proj.lora_B.weight").astype(mx.float32)
        print(report_qkv(f"blocks.{b}.attn.qkv_proj  lora_B", bb, args.heads, args.head_dim))
        del bb
        mx.clear_cache()

    print("\n=== SwiGLU fc1 halves ===")
    for b in args.blocks:
        print(report_halves(f"blocks.{b}.mlp.fc1.weight base", dit.get(f"blocks.{b}.mlp.fc1.weight")))
        a = lora.get(f"blocks.{b}.mlp.fc1.lora_A.weight").astype(mx.float32)
        bb = lora.get(f"blocks.{b}.mlp.fc1.lora_B.weight").astype(mx.float32)
        print(report_halves(f"blocks.{b}.mlp.fc1 delta", bb @ a))
        del a, bb
        mx.clear_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
