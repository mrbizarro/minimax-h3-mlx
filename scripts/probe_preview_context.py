"""How much causal context does a one-frame TAE preview need to look like the real decode?

The live preview decodes ONE latent frame instead of the whole clip, which is what makes it cost
0.06 s instead of 1.1 s. The price is the tiny decoder's causal memory: ``MemBlock`` remembers the
previous frame's input, so a frame decoded alone is decoded as if it were the first frame of the
clip, with zero history. This measures that price against the ground truth — the same latents
decoded as one whole sequence, which is what ``--draft-decode tae`` delivers.

Context ``c`` means "hand the decoder latent frames ``f-c .. f`` and read the output frame that
belongs to ``f``". ``c = 1`` is free: the decoder pads its input up to five latent tokens anyway, so
one and two tokens cost exactly the same.

Runs entirely on cached clean Stage-A rows — no DiT, no denoise, one small GPU job.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.packing import unpatchify_video_tokens  # noqa: E402
from minimax_h3_mlx.tiny_video_vae import load_tiny_h3_video_decoder  # noqa: E402


def to_image(decoded: mx.array, index: int) -> np.ndarray:
    frame = np.array(decoded)[0, :, index].transpose(1, 2, 0)
    return (np.clip(frame, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def local_output_index(context: int) -> int:
    """Output frame that carries the last latent of a ``context + 1`` token window."""
    return 0 if context == 0 else 1 + 4 * (context - 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a", type=Path, required=True)
    parser.add_argument("--tae-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-frame", type=int, default=None)
    parser.add_argument("--contexts", type=int, nargs="+", default=(0, 1, 2, 4))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cached = np.load(args.stage_a, allow_pickle=False)
    meta = json.loads(str(cached["meta"]))
    latent_frames = int(meta["latent_frames"])
    latent_h, latent_w = int(meta["latent_h"]), int(meta["latent_w"])
    patch = tuple(int(value) for value in meta["patch"])
    frame = args.latent_frame if args.latent_frame is not None else latent_frames // 2

    rows = mx.array(cached["video_rows"])
    tae = load_tiny_h3_video_decoder(args.tae_checkpoint)
    latents = unpatchify_video_tokens(
        rows, latent_frames, latent_h, latent_w, tae.latent_channels, patch
    )
    mx.eval(latents)

    # Ground truth: the whole clip decoded in one pass, the way --draft-decode tae does it.
    started = time.perf_counter()
    full = tae.decode(latents, int(meta["frames"]))
    mx.eval(full)
    full_seconds = time.perf_counter() - started
    chunk, position = divmod(frame, 5)
    truth_index = chunk * 17 + (0 if position == 0 else 1 + 4 * (position - 1))
    truth = to_image(full, truth_index)
    del full
    mx.clear_cache()

    panels = [(f"FULL sequence decode\nframe {truth_index}  ({full_seconds:.2f}s)", truth)]
    report = []
    for context in args.contexts:
        low = max(0, frame - context)
        effective = frame - low
        window = latents[:, :, low : frame + 1]
        index = local_output_index(effective)
        # Warm, then time.
        decoded = tae.decode(window, index + 1)
        mx.eval(decoded)
        del decoded
        started = time.perf_counter()
        decoded = tae.decode(window, index + 1)
        mx.eval(decoded)
        seconds = time.perf_counter() - started
        image = to_image(decoded, index)
        del decoded
        mx.clear_cache()

        mae = float(np.abs(image.astype(np.float32) - truth.astype(np.float32)).mean())
        report.append(
            {
                "context": context,
                "latent_tokens": effective + 1,
                "seconds": round(seconds, 4),
                "speedup_vs_full": round(full_seconds / seconds, 1),
                "mean_abs_diff_vs_full": round(mae, 3),
            }
        )
        panels.append(
            (f"context {context} ({effective + 1} token{'s' if effective else ''})\n"
             f"{seconds:.3f}s   mean |diff| {mae:.2f}/255", image)
        )

    print(json.dumps(report, indent=2))

    from PIL import ImageDraw, ImageFont

    height, width = truth.shape[:2]
    band = 52
    pad = 8
    strip = Image.new(
        "RGB", (len(panels) * width + (len(panels) + 1) * pad, height + band + 2 * pad), (16, 16, 18)
    )
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    for order, (label, image) in enumerate(panels):
        x = pad + order * (width + pad)
        strip.paste(Image.fromarray(image), (x, pad))
        draw.multiline_text((x + 4, pad + height + 5), label, font=font, fill=(238, 238, 240))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    strip.save(args.out)
    print(f"\nwrote {args.out} ({strip.size[0]}x{strip.size[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
