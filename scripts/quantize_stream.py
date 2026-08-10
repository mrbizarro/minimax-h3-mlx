#!/usr/bin/env python3
"""Stream-quantize the pruned H3 DiT to the Q8 pack — bounded memory, any Mac.

Why this exists next to build_quant.py: that builder loads the WHOLE model and
quantizes it in place, which is fine on a 64 GB Mac and impossible on the 48 GB
machines the Q8 pack is FOR. This one never holds more than one tensor: it
reads the single-file pruned checkpoint (mmap), classifies each key with the
same QuantConfig recipe the loader replays, quantizes matching weights one at a
time on the CPU stream (bit-deterministic across runs — the GPU quant stream is
not, see uetuluk2's rebuild findings §3), and writes shards + config.json +
quant_config.json in exactly the layout load_dit() consumes.

    .venv/bin/python scripts/quantize_stream.py \
        --src  <models>/deepbeep-pruned-bf16/MiniMax-H3-FL2VA-pruned_bf16.safetensors \
        --out  <models>/h3-dit-q8

Validation before it reports success: the written pack is re-opened, one
quantized module is dequantized and compared to the source (rel tolerance), and
load_dit() must build the full tree from it.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import struct
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.config import DiTConfig
from minimax_h3_mlx.quantize import QuantConfig
from safetensors import safe_open

GROUP, BITS = 64, 8
MAX_SHARD = 5 * 1024**3
DTYPE_BYTES = {"BF16": 2, "F32": 4, "U32": 4}


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    t0 = time.time()
    hdr = read_header(a.src)
    with safe_open(str(a.src), framework="np") as h:
        grid, rank = h.get_slice("adaln_t_table").get_shape()
    cfg = DiTConfig(time_embed_dim=rank, adaln_curve_grid=grid)
    qc = QuantConfig(bits=BITS, group_size=GROUP, quantize_adaln=True, adaln_bits=BITS)

    # Classify every key with the loader's own recipe. Only 2-D `.weight`
    # tensors of modules the recipe names are quantized; everything else is
    # copied through untouched (norms, biases, tables, heads).
    plan: list[tuple[str, str, list[int], int | None]] = []   # key, dtype, shape, bits
    for k, meta in sorted(hdr.items()):
        base = k[: -len(".weight")] if k.endswith(".weight") else None
        bits = qc.bits_for(base) if base else None
        if bits and len(meta["shape"]) == 2:
            out_f, in_f = meta["shape"]
            if in_f % GROUP == 0:
                plan.append((k, "q", meta["shape"], bits))
                continue
        plan.append((k, meta["dtype"], meta["shape"], None))

    n_q = sum(1 for _, t, _, _ in plan if t == "q")
    print(f"{len(plan)} tensors, {n_q} quantized at {BITS}-bit g{GROUP}", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    src = mx.load(str(a.src))          # lazy, mmap-backed

    shard_idx, shard_bytes, shard = 1, 0, {}
    names: list[str] = []
    weight_map: dict[str, str] = {}

    def flush() -> None:
        nonlocal shard_idx, shard_bytes, shard
        if not shard:
            return
        name = f"model-{shard_idx:05d}.safetensors"
        mx.save_safetensors(str(a.out / name), shard)
        for kk in shard:
            weight_map[kk] = name
        names.append(name)
        print(f"  wrote {name} ({shard_bytes/1e9:.2f} GB, {len(shard)} tensors)", flush=True)
        shard_idx += 1
        shard_bytes, shard = 0, {}

    with mx.stream(mx.cpu):            # CPU quant: deterministic and RAM-bounded
        for k, kind, shape, bits in plan:
            if kind == "q":
                w, sc, bi = mx.quantize(src[k], group_size=GROUP, bits=BITS)
                base = k[: -len(".weight")]
                add = {base + ".weight": w,
                       base + ".scales": sc.astype(mx.bfloat16),
                       base + ".biases": bi.astype(mx.bfloat16)}
            else:
                add = {k: src[k]}
            mx.eval(list(add.values()))
            nbytes = sum(t.nbytes for t in add.values())
            if shard_bytes + nbytes > MAX_SHARD and shard:
                flush()
            shard.update(add)
            shard_bytes += nbytes
    flush()

    (a.out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_shards": len(names)}, "weight_map": weight_map}, indent=1))
    (a.out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=1))
    (a.out / "quant_config.json").write_text(json.dumps(
        {"bits": BITS, "group_size": GROUP, "quantize_adaln": True,
         "adaln_bits": BITS, "writer": "quantize_stream", "source": a.src.name}, indent=1))

    # ---- validation -----------------------------------------------------
    probe_key = next(k for k, t, _, _ in plan if t == "q")
    base = probe_key[: -len(".weight")]
    packed = mx.load(str(a.out / weight_map[base + ".weight"]))
    deq = mx.dequantize(packed[base + ".weight"],
                        packed[base + ".scales"].astype(mx.float32),
                        packed[base + ".biases"].astype(mx.float32),
                        group_size=GROUP, bits=BITS)
    ref = src[probe_key].astype(mx.float32)
    err = float(mx.abs(deq - ref).max())
    print(f"dequant probe {base}: max abs err {err:.5f}", flush=True)
    if err > 0.25:
        raise SystemExit("dequant error implausibly large — refusing to accept")

    from minimax_h3_mlx.load import load_dit
    load_dit(a.out, verbose=False)
    print(f"LOAD OK — pack valid. Total {time.time()-t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
