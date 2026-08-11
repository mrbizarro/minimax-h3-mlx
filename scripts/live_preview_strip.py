"""Build the labelled preview-evolution strip: every forward's thumbnail, then the delivered frame.

The claim the early-abort feature rests on is that the FIRST forward already shows the composition.
That is not a claim to assert — it is a picture. This lays each forward's published preview beside
the frame the finished clip actually delivers at the same moment of the timeline, labelled with the
sigma the preview was taken at, and lets the eye settle it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.config import PipelineConfig  # noqa: E402
from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler  # noqa: E402

BAND = 46
PAD = 8
FONTS = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/SFNSDisplay.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)


def load_font(size: int):
    for candidate in FONTS:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    return ImageFont.load_default()


def extract_frame(mp4: Path, index: int, destination: Path) -> Image.Image:
    """Pull one frame out of the delivered mp4 — what a viewer would actually see."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(mp4),
            "-vf", f"select=eq(n\\,{index})",
            "-vsync", "0", "-frames:v", "1",
            str(destination),
        ],
        check=True,
        capture_output=True,
    )
    return Image.open(destination).convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-dir", type=Path, required=True)
    parser.add_argument("--mp4", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sigma-points", type=int, default=4)
    args = parser.parse_args()

    status = json.loads((args.live_dir / "status.json").read_text())
    previews = sorted(args.live_dir.glob("preview_[0-9][0-9].png"))
    if not previews:
        raise SystemExit(f"no previews in {args.live_dir}")

    scheduler = MiniMaxH3Scheduler(shift=PipelineConfig().sigma_shift_video)
    scheduler.set_timesteps(args.sigma_points)
    sigmas = [float(value) for value in scheduler.sigmas.tolist()]

    frame_index = int(status["approx_output_frame"])
    final = extract_frame(args.mp4, frame_index, args.out.with_name(f"{args.out.stem}_final.png"))

    panels = []
    for order, path in enumerate(previews, start=1):
        image = Image.open(path).convert("RGB")
        sigma_before = sigmas[order - 1]
        sigma_after = sigmas[order]
        panels.append(
            (
                image,
                f"after forward {order}/{len(previews)}   x0 estimate, TAE",
                f"sigma {sigma_before:.3f} -> {sigma_after:.3f}   {path.name}",
            )
        )
    panels.append(
        (
            final,
            f"DELIVERED frame {frame_index}   full video VAE",
            f"{args.mp4.name}   (same moment of the clip)",
        )
    )

    width, height = panels[0][0].size
    panels = [(image.resize((width, height), Image.Resampling.LANCZOS), a, b) for image, a, b in panels]

    strip = Image.new(
        "RGB",
        (len(panels) * width + (len(panels) + 1) * PAD, height + BAND + 2 * PAD),
        (16, 16, 18),
    )
    draw = ImageDraw.Draw(strip)
    title_font = load_font(19)
    sub_font = load_font(15)

    for index, (image, title, subtitle) in enumerate(panels):
        x = PAD + index * (width + PAD)
        strip.paste(image, (x, PAD))
        draw.text((x + 4, PAD + height + 5), title, font=title_font, fill=(245, 245, 245))
        draw.text((x + 4, PAD + height + 26), subtitle, font=sub_font, fill=(150, 152, 160))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    strip.save(args.out)
    print(f"wrote {args.out} ({strip.size[0]}x{strip.size[1]})")

    # A number to sit under the picture: how close each preview already is to the delivered frame.
    reference = np.asarray(final, dtype=np.float32)
    print("\nper-forward distance to the delivered frame (lower = closer):")
    for index, (image, title, _) in enumerate(panels[:-1], start=1):
        current = np.asarray(image, dtype=np.float32)
        mae = float(np.abs(current - reference).mean())
        # Downsampled correlation: composition agreement, not texture agreement.
        small_a = np.asarray(image.resize((40, 24)).convert("L"), dtype=np.float32).ravel()
        small_b = np.asarray(final.resize((40, 24)).convert("L"), dtype=np.float32).ravel()
        corr = float(np.corrcoef(small_a, small_b)[0, 1])
        print(f"  forward {index}: mean |diff| {mae:6.2f}/255   composition corr {corr:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
