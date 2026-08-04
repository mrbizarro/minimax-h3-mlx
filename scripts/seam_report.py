"""Grade the joins in a chained clip — the one thing window chaining can get wrong.

A chain buys duration by paying for it at the seams, so the seams are what has to be inspected.
This decodes the delivered clip, locates every boundary from the window geometry, and reports how
far out of line the boundary is against its own neighbourhood: a cut reads as a frame-to-frame
step several times the local motion, and an audio pop reads as a sample step far above the clip's
own typical slew. It also writes a contact sheet of the frames either side so the numbers can be
overruled by an eye, which is the only authority that actually matters here.

    python scripts/seam_report.py clip.mp4 --window-frames 124 --windows 2 -o seam.png
"""

from __future__ import annotations

import argparse
import json
import subprocess
import wave
from pathlib import Path

import numpy as np

FPS = 24


def probe(path: Path) -> dict:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v", "-count_frames",
            "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate",
            "-show_entries", "format=duration", "-of", "json", str(path),
        ],
        capture_output=True,
        check=True,
    )
    data = json.loads(out.stdout)
    stream = data["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream["nb_read_frames"]),
        "fps": eval(stream["r_frame_rate"]),  # noqa: S307 — ffprobe emits "24/1"
        "duration": float(data["format"]["duration"]),
    }


def decode(path: Path, width: int, height: int) -> np.ndarray:
    raw = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as fh:
        rate = fh.getframerate()
        channels = fh.getnchannels()
        pcm = np.frombuffer(fh.readframes(fh.getnframes()), "<i2").astype(np.float32) / 32767.0
    return pcm.reshape(-1, channels).T, rate


def boundaries(window_frames: int, windows: int, total: int) -> list[int]:
    """Output indices where a new window's material starts (one duplicate frame dropped each)."""
    found = []
    for k in range(2, windows + 1):
        index = window_frames + (k - 2) * (window_frames - 1)
        if 0 < index < total:
            found.append(index)
    return found


def video_grade(video: np.ndarray, index: int, context: int) -> dict:
    luma = video.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    lo, hi = max(1, index - context), min(len(video), index + context + 1)
    steps = {i: float(np.abs(luma[i] - luma[i - 1]).mean()) for i in range(lo, hi)}
    neighbours = [v for i, v in steps.items() if i != index]
    median = float(np.median(neighbours)) if neighbours else float("nan")
    brightness = {i: float(luma[i].mean()) for i in range(lo - 1, hi)}
    return {
        "frame": index,
        "seam_step": steps.get(index, float("nan")),
        "neighbour_median_step": median,
        "ratio": steps.get(index, float("nan")) / median if median else float("nan"),
        "luma_before": brightness.get(index - 1),
        "luma_after": brightness.get(index),
        "luma_delta": brightness.get(index, 0.0) - brightness.get(index - 1, 0.0),
        "neighbour_steps": [round(v, 3) for _, v in sorted(steps.items())],
    }


def audio_grade(audio: np.ndarray, rate: int, index: int) -> dict:
    sample = int(round(index / FPS * rate))
    diff = np.abs(np.diff(audio, axis=1))
    typical = float(np.percentile(diff, 99.9))
    window = diff[:, max(0, sample - 3) : sample + 3]
    quarter = int(0.25 * rate)
    before = audio[:, max(0, sample - quarter) : sample]
    after = audio[:, sample : sample + quarter]
    return {
        "sample": sample,
        "seam_step": float(window.max()) if window.size else float("nan"),
        "clip_p99.9_step": typical,
        "ratio": float(window.max()) / typical if typical else float("nan"),
        "rms_before_db": 20 * np.log10(max(float(np.sqrt((before**2).mean())), 1e-9)),
        "rms_after_db": 20 * np.log10(max(float(np.sqrt((after**2).mean())), 1e-9)),
    }


def contact_sheet(video: np.ndarray, index: int, context: int, path: Path, scale: float) -> None:
    from PIL import Image, ImageDraw

    lo, hi = max(0, index - context), min(len(video), index + context)
    frames = [Image.fromarray(video[i]) for i in range(lo, hi)]
    size = (int(frames[0].width * scale), int(frames[0].height * scale))
    frames = [f.resize(size, Image.Resampling.LANCZOS) for f in frames]
    sheet = Image.new("RGB", (size[0] * len(frames), size[1] + 22), (16, 16, 18))
    draw = ImageDraw.Draw(sheet)
    for column, (number, frame) in enumerate(zip(range(lo, hi), frames)):
        sheet.paste(frame, (column * size[0], 22))
        seam = number == index
        draw.text(
            (column * size[0] + 4, 6),
            f"{number}{'  << SEAM' if seam else ''}",
            fill=(255, 90, 90) if seam else (170, 170, 180),
        )
        if seam:
            draw.line([(column * size[0], 0), (column * size[0], sheet.height)], (255, 60, 60), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path)
    parser.add_argument("--window-frames", type=int, required=True)
    parser.add_argument("--windows", type=int, required=True)
    parser.add_argument("--context", type=int, default=7, help="frames each side (7 = +-0.29 s)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="contact sheet prefix")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--wav", type=Path, default=None, help="defaults to the clip's sidecar")
    args = parser.parse_args()

    info = probe(args.clip)
    video = decode(args.clip, info["width"], info["height"])
    marks = boundaries(args.window_frames, args.windows, len(video))
    print(f"{args.clip.name}: {len(video)} frames, {len(video)/FPS:.3f} s, seams at {marks}")

    wav = args.wav or args.clip.with_suffix(".wav")
    audio = rate = None
    if wav.exists():
        audio, rate = read_wav(wav)
        print(f"audio: {audio.shape[1]} samples @ {rate} Hz = {audio.shape[1]/rate:.3f} s")
        drift = audio.shape[1] / rate - len(video) / FPS
        print(f"a/v drift: {drift*1000:+.2f} ms")

    report = []
    for index in marks:
        grade = {"video": video_grade(video, index, args.context)}
        if audio is not None:
            grade["audio"] = audio_grade(audio, rate, index)
        report.append(grade)
        v = grade["video"]
        print(
            f"\nseam @ frame {index} ({index/FPS:.3f} s)\n"
            f"  frame-to-frame luma step: {v['seam_step']:.3f} "
            f"vs neighbour median {v['neighbour_median_step']:.3f}  -> {v['ratio']:.2f}x\n"
            f"  mean luma {v['luma_before']:.2f} -> {v['luma_after']:.2f} "
            f"({v['luma_delta']:+.2f})\n"
            f"  local steps: {v['neighbour_steps']}"
        )
        if audio is not None:
            a = grade["audio"]
            print(
                f"  audio sample step at seam: {a['seam_step']:.5f} vs clip p99.9 "
                f"{a['clip_p99.9_step']:.5f} -> {a['ratio']:.2f}x\n"
                f"  RMS {a['rms_before_db']:.1f} dB -> {a['rms_after_db']:.1f} dB"
            )
        if args.output is not None:
            sheet = args.output.with_name(f"{args.output.stem}_f{index}{args.output.suffix}")
            contact_sheet(video, index, args.context, sheet, args.scale)
            print(f"  wrote {sheet}")

    if args.output is not None:
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
