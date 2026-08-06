"""How much of a LoRA update survives being folded into a bfloat16 base weight?

larryvrh's reference generator keeps the low-rank update as a run-time matmul and says folding it
into the bf16 base "would round most of the update away when it is small relative to the weight".
That claim decides our whole integration strategy, so it is measured here rather than believed:
for a sample of modules we compute the exact float32 sum, round it to bfloat16, and report what
fraction of the intended delta is left.

Reads individual tensors out of the two safetensors files by byte range, so nothing close to the
41 GB checkpoint is ever resident.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

_DTYPE = {
    "BF16": (mx.bfloat16, np.uint16, 2),
    "F16": (mx.float16, np.uint16, 2),
    "F32": (mx.float32, np.uint32, 4),
}


class SafeReader:
    """Minimal by-range safetensors reader (header JSON + mmap slice)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with open(self.path, "rb") as fh:
            length = int.from_bytes(fh.read(8), "little")
            self.header = json.loads(fh.read(length))
        self.start = 8 + length
        self.mmap = np.memmap(self.path, dtype=np.uint8, mode="r")

    def keys(self):
        return [k for k in self.header if k != "__metadata__"]

    def get(self, key: str) -> mx.array:
        entry = self.header[key]
        dtype, raw, size = _DTYPE[entry["dtype"]]
        a, b = entry["data_offsets"]
        buf = self.mmap[self.start + a : self.start + b]
        flat = np.frombuffer(buf.tobytes(), dtype=raw)
        return mx.array(flat).view(dtype).reshape(entry["shape"])


def survival(base: mx.array, delta: mx.array) -> dict[str, float]:
    """Fraction of ``delta`` that survives ``round_bf16(base + delta) - base``."""
    b32 = base.astype(mx.float32)
    exact = b32 + delta
    fused = exact.astype(mx.bfloat16).astype(mx.float32)
    realized = fused - b32
    residual = realized - delta
    dn = float(mx.sqrt(mx.sum(delta * delta)).item())
    rn = float(mx.sqrt(mx.sum(residual * residual)).item())
    # Also: how much does rounding the *base alone* cost? (it is already bf16, so zero) and how
    # large is the update relative to the weight it rides on.
    bn = float(mx.sqrt(mx.sum(b32 * b32)).item())
    return {
        "|W|": bn,
        "|dW|": dn,
        "dW/W": dn / bn if bn else float("nan"),
        "residual/|dW|": rn / dn if dn else float("nan"),
        "survival": 1.0 - (rn / dn if dn else 0.0),
        "corr": float(
            (mx.sum(realized * delta) / (mx.sqrt(mx.sum(realized**2)) * mx.sqrt(mx.sum(delta**2)))).item()
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dit", default="../models/deepbeep-pruned-bf16/MiniMax-H3-FL2VA-pruned_bf16.safetensors")
    ap.add_argument("--lora", default="../models/turbo-lora/minimax_h3_turbo_4step.safetensors")
    ap.add_argument("--modules", nargs="*", default=[
        "blocks.0.attn.qkv_proj", "blocks.0.attn.out_proj", "blocks.0.mlp.fc1", "blocks.0.mlp.fc2",
        "blocks.25.attn.qkv_proj", "blocks.25.mlp.fc2",
        "blocks.49.attn.qkv_proj", "blocks.49.attn.out_proj", "blocks.49.mlp.fc1", "blocks.49.mlp.fc2",
        "token_refiner.blocks.0.attn.qkv_proj", "token_refiner.blocks.1.mlp.fc1",
    ])
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    dit = SafeReader(here / args.dit if not Path(args.dit).is_absolute() else args.dit)
    lora = SafeReader(here / args.lora if not Path(args.lora).is_absolute() else args.lora)

    print(f"scale {args.scale}")
    print(f"{'module':<38} {'dW/W':>10} {'survival':>9} {'corr':>8} {'|dW|':>10}")
    rows = []
    for name in args.modules:
        w = dit.get(name + ".weight")
        a = lora.get(name + ".lora_A.weight").astype(mx.float32)
        b = lora.get(name + ".lora_B.weight").astype(mx.float32)
        delta = (b @ a) * args.scale
        if delta.shape != w.shape:
            print(f"{name:<38}  SHAPE {tuple(delta.shape)} vs {tuple(w.shape)}")
            continue
        s = survival(w, delta)
        rows.append(s)
        print(f"{name:<38} {s['dW/W']:>10.3e} {s['survival']:>9.4f} {s['corr']:>8.5f} {s['|dW|']:>10.3e}")
        del w, a, b, delta
        mx.clear_cache()
    if rows:
        print(f"\nmean survival {np.mean([r['survival'] for r in rows]):.4f} "
              f"| mean dW/W {np.mean([r['dW/W'] for r in rows]):.3e} "
              f"| min corr {min(r['corr'] for r in rows):.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
