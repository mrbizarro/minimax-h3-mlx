"""Build a same-seed A/B contact sheet from two or more renders of one config.

Every speed lever has to answer the same question — what did it cost the picture? — so the grid
samples the *same* normalized timestamps from each clip and stacks them one row per render, with
the run label burned into the left margin. Peak audio level is printed alongside, because a lever
that quietly kills the audio track would otherwise pass a purely visual review.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True,
        check=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def peak_db(path: Path) -> str:
    out = subprocess.run(
        ["ffmpeg", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True,
    )
    text = out.stderr.decode()
    peak = mean = "n/a"
    for line in text.splitlines():
        if "max_volume" in line:
            peak = line.split("max_volume:")[-1].strip()
        if "mean_volume" in line:
            mean = line.split("mean_volume:")[-1].strip()
    return f"peak {peak} / mean {mean}"


def extract(path: Path, fractions: list[float], workdir: Path) -> list[Image.Image]:
    duration = probe_duration(path)
    frames = []
    for index, fraction in enumerate(fractions):
        target = workdir / f"{path.stem}_{index}.png"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{max(0.0, duration * fraction):.3f}",
                "-i", str(path), "-frames:v", "1", str(target),
            ],
            check=True,
        )
        frames.append(Image.open(target).convert("RGB"))
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", nargs="+", help="label=path pairs, or bare paths")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--fractions", type=float, nargs="*", default=[0.02, 0.35, 0.65, 0.95])
    parser.add_argument("--width", type=int, default=384, help="per-tile width")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required")

    entries = []
    for item in args.clips:
        label, _, path = item.partition("=")
        if not path:
            label, path = Path(label).stem, label
        entries.append((label, Path(path)))

    margin = 240
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        rows = []
        for label, path in entries:
            frames = extract(path, args.fractions, workdir)
            scale = args.width / frames[0].width
            size = (args.width, int(round(frames[0].height * scale)))
            rows.append((label, path, [f.resize(size, Image.Resampling.LANCZOS) for f in frames]))

        tile_w, tile_h = rows[0][2][0].size
        sheet = Image.new(
            "RGB",
            (margin + tile_w * len(args.fractions), tile_h * len(rows)),
            (16, 16, 18),
        )
        draw = ImageDraw.Draw(sheet)
        for r, (label, path, frames) in enumerate(rows):
            for c, frame in enumerate(frames):
                sheet.paste(frame, (margin + c * tile_w, r * tile_h))
            draw.text((10, r * tile_h + 10), label, fill=(255, 255, 255))
            draw.text((10, r * tile_h + 26), peak_db(path), fill=(160, 160, 170))
            draw.text((10, r * tile_h + 42), path.name, fill=(110, 110, 120))

        args.output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(args.output)

    print(f"wrote {args.output} ({len(rows)} runs x {len(args.fractions)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
