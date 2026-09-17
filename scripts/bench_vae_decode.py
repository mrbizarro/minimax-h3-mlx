#!/usr/bin/env python3
"""Decode saved Stage-A latents with the video VAE under several decoder precisions and compare.

Answers two questions without a new render: how long the full-quality VAE decode takes on this
machine for a given saved latent, and how far each faster decoder mode moves the pixels away from
the current (float32-arithmetic) decode.

    python scripts/bench_vae_decode.py --compact-root MODELS/ddalcu-q8 \
        --stage-a clip.stage_a.npz --modes float32,float16,float32 --out OUTDIR

Each mode decodes the same latent once (the first mode is also the reference). Every decode is
timed end-to-end the way ``generate_staged.py`` times it (unpatchify, denormalize, decode, host
copy, uint8), and the uint8 frames are compared against the reference: max abs, mean abs, PSNR.
Repeating a mode (``float32`` twice above) measures run-to-run determinism and warm-kernel cost.
``int8`` quantizes the decoder linears to 8 bits (group 64) — an approximate MLX analogue of
ComfyUI's int8_convrot VAE, weight-only because MLX has no int8 activation GEMM.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.load import load_compact_video_vae  # noqa: E402
from minimax_h3_mlx.stage_cache import load_stage_a  # noqa: E402

spec = importlib.util.spec_from_file_location("generate_staged", ROOT / "scripts" / "generate_staged.py")
staged = importlib.util.module_from_spec(spec)
spec.loader.exec_module(staged)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compact-root", type=Path, required=True)
    ap.add_argument("--stage-a", type=Path, required=True)
    ap.add_argument("--modes", default="float32,float16,float32")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save-frames", action="store_true", help="also write each mode's uint8 frames (.npy)")
    ap.add_argument("--memory-gb", type=float, default=40.0)
    args = ap.parse_args()

    mx.set_memory_limit(int(args.memory_gb * 1024**3))
    args.out.mkdir(parents=True, exist_ok=True)
    meta, arrays = load_stage_a(args.stage_a)
    frames, lf, lh, lw = (int(meta[k]) for k in ("frames", "latent_frames", "latent_h", "latent_w"))
    rows = mx.array(arrays["video_rows"])
    geometry = f"{meta['width']}x{meta['height']} {frames}f ({lf}x{lh}x{lw} latent)"
    print("latent:", args.stage_a.name, geometry, flush=True)

    t0 = time.perf_counter()
    vae = load_compact_video_vae(args.compact_root / "video_vae.safetensors")
    load_s = time.perf_counter() - t0
    print(f"vae load {load_s:.1f}s, decode batch {vae.decode_batch}", flush=True)

    results, reference = [], None
    quantized = False
    for index, mode in enumerate(m.strip() for m in args.modes.split(",")):
        if mode.startswith("int8"):
            if not quantized:
                nn.quantize(vae.decoder, group_size=64, bits=8,
                            class_predicate=lambda p, m: isinstance(m, nn.Linear) and ".transformer_blocks." in f".{p}")
                mx.eval(vae.decoder.parameters())
                quantized = True
            vae.decode_dtype = "float16" if mode == "int8" else mode.split("-", 1)[1]
        else:
            if quantized:
                raise SystemExit("int8 modes must come last")
            vae.decode_dtype = mode
        mx.clear_cache()
        mx.reset_peak_memory()
        t = time.perf_counter()
        video = staged.decode_video(vae, rows, frames, lf, lh, lw, (1, 2, 2))
        seconds = time.perf_counter() - t
        peak = mx.get_peak_memory() / 1024**3
        item = {"index": index, "mode": mode, "seconds": round(seconds, 2), "peak_gib": round(peak, 2),
                "frames": int(video.shape[0]), "shape": list(video.shape)}
        if reference is None:
            reference = video
        else:
            diff = np.abs(video.astype(np.int16) - reference.astype(np.int16))
            item.update(max_abs=int(diff.max()), mean_abs=round(float(diff.mean()), 4),
                        frac_pixels_changed=round(float((diff > 0).mean()), 4),
                        psnr_db=round(psnr(video, reference), 2), bit_identical=bool(diff.max() == 0))
        if args.save_frames:
            np.save(args.out / f"frames_{index}_{mode}.npy", video)
        results.append(item)
        print(json.dumps(item), flush=True)

    report = {"stage_a": str(args.stage_a), "geometry": geometry, "vae_load_s": round(load_s, 2),
              "decode_batch": vae.decode_batch, "mlx": mx.__version__, "results": results}
    out = args.out / f"bench_{args.stage_a.stem}.json"
    out.write_text(json.dumps(report, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
