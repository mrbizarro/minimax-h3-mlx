"""Micro-benchmark the raw GEMM ceiling this machine can reach at H3's block shapes.

The point is to separate "the pipeline is slow" from "MLX matmul is this fast". Every H3 block is
four dense projections over the packed sequence, so the achievable TFLOP/s on those exact shapes is
the speed-of-light for the denoiser. Quantized variants are timed against the same shapes so the
Q8 decision is made before 20 GB of weights are written to disk.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx


SHAPES = {
    "qkv_proj": (5376, 21504),
    "out_proj": (7168, 5376),
    "mlp.fc1": (5376, 28672),
    "mlp.fc2": (14336, 5376),
}


def timed(fn, warmup: int = 2, iters: int = 6) -> float:
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
    parser.add_argument("--bits", type=int, nargs="*", default=[8, 4])
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    S = args.rows
    print(f"rows={S}  mlx={mx.__version__}")
    totals = {"bf16": 0.0}
    for bits in args.bits:
        totals[f"q{bits}"] = 0.0

    for name, (k, n) in SHAPES.items():
        x = mx.random.normal((1, S, k)).astype(mx.bfloat16)
        w = mx.random.normal((n, k)).astype(mx.bfloat16)
        mx.eval(x, w)
        flops = 2.0 * S * k * n

        dt = timed(lambda: x @ w.T)
        totals["bf16"] += dt
        line = f"{name:10s} [{k}x{n}]  bf16 {dt*1000:8.1f} ms  {flops/dt/1e12:6.2f} TF/s"

        for bits in args.bits:
            wq, scales, biases = mx.quantize(w, group_size=args.group_size, bits=bits)
            mx.eval(wq, scales, biases)
            dtq = timed(
                lambda: mx.quantized_matmul(
                    x, wq, scales, biases, transpose=True, group_size=args.group_size, bits=bits
                )
            )
            totals[f"q{bits}"] += dtq
            line += f" | q{bits} {dtq*1000:7.1f} ms ({dt/dtq:4.2f}x)"
        print(line, flush=True)
        del x, w
        mx.clear_cache()

    print()
    for key, value in totals.items():
        per_block = value
        print(
            f"{key:5s}: {per_block*1000:8.1f} ms/block  ->  50 blocks = {per_block*50:6.2f} s/step"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
