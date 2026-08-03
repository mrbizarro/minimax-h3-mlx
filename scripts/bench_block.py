"""Time one real ``TransformerBlock`` at a target packed length and split it into its parts.

``bench_gemm.py`` gives the dense-projection speed of light. This adds the pieces that sit around
those projections — attention, the AdaLN row gathers, the rotary rotation — with random weights, so
the attackable share of a denoising step can be measured in seconds rather than a 7-minute render.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import mlx.nn as nn

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.config import DiTConfig
from minimax_h3_mlx.dit import Attention, FeedForward, TransformerBlock, RotaryPosEmbed3D


def timed(fn, warmup: int = 2, iters: int = 5) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    started = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - started) / iters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=7689)
    parser.add_argument("--modalities", type=int, default=3)
    args = parser.parse_args()

    S = args.rows
    config = DiTConfig(adaln_curve_grid=1001)
    block = TransformerBlock(config)
    block.set_dtype(mx.bfloat16)
    x = mx.random.normal((1, S, config.hidden_size)).astype(mx.bfloat16)

    rope = RotaryPosEmbed3D(config)
    position_ids = mx.random.uniform(0.0, 32.0, (S, 3)).astype(mx.float32)
    rotary = rope(position_ids)
    mx.eval(rotary)

    modulation = tuple(
        mx.random.normal((args.modalities, config.hidden_size)).astype(mx.bfloat16) * 0.02
        for _ in range(6)
    )
    idx = mx.array((mx.arange(S) % args.modalities).astype(mx.int32))
    mx.eval(x, idx, *modulation)

    n = config.num_attention_heads
    d = config.attention_head_dim
    q = mx.random.normal((1, n, S, d)).astype(mx.bfloat16)
    k = mx.random.normal((1, n, S, d)).astype(mx.bfloat16)
    v = mx.random.normal((1, n, S, d)).astype(mx.bfloat16)
    mx.eval(q, k, v)

    results = {}
    results["block_total"] = timed(lambda: block(x, modulation, idx, rotary))
    results["attn_full"] = timed(lambda: block.attn(x, rotary))
    results["attn_no_rope"] = timed(lambda: block.attn(x, None))
    results["sdpa_only"] = timed(
        lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=d**-0.5)
    )
    results["mlp"] = timed(lambda: block.mlp(x))
    results["norm1"] = timed(lambda: block.norm1(x))

    def modulate():
        h = block.norm1(x)
        return h * (1.0 + modulation[1][idx]) + modulation[0][idx]

    results["norm+adaln_gather"] = timed(modulate)

    def gathers_only():
        return tuple(m[idx] for m in modulation)

    results["6x_row_gather"] = timed(lambda: mx.stack(gathers_only()))

    # `mx.compile` can only ever recover the elementwise glue around the projections, so it is
    # measured here rather than paid for with a 7-minute render.
    compiled = mx.compile(lambda a, mod, i, cos, sin: block(a, mod, i, (cos, sin)))
    results["block_compiled"] = timed(
        lambda: compiled(x, modulation, idx, rotary[0], rotary[1])
    )

    print(f"rows={S}  hidden={config.hidden_size}  heads={n}x{d}")
    total = results["block_total"]
    for name, value in results.items():
        print(f"  {name:20s} {value*1000:8.2f} ms   x50 = {value*50:7.2f} s   {value/total*100:5.1f}%")

    sdpa_flops = 4.0 * S * S * n * d
    print(f"\nsdpa achieved: {sdpa_flops/results['sdpa_only']/1e12:.2f} TFLOP/s")
    print(f"projected step (50 blocks): {total*50:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
