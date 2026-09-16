"""Muxed H3 clips carry exactly the frames that were asked for, and audio as long as the picture.

CPU only — tiny synthetic frames and a synthetic WAV through the real ``save_mp4`` / ffmpeg path,
then ffprobe on the encoded file. What it pins down: ``-shortest`` used to cut the LAST video frame
of every delivery (73 -> 72, 124 -> 123) because libx264 still held frames when the audio ended.
Covers the plain path, a short waveform (padded), a long one (trimmed), the ffmpeg ``atempo``
path, and the draft path's Apple time stretch when ``/usr/bin/swift`` is present.

Run: ``python tests/test_mux_frame_count.py`` (set PATH to test another ffmpeg).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.media import probe_video_frames, save_mp4  # noqa: E402

FPS = 24.0
SAMPLE_RATE = 32000
# One AAC frame at 32 kHz. The encoder works in 1024-sample packets, so this is the finest
# granularity at which the audio stream's reported duration can match the picture.
AAC_FRAME_S = 1024 / SAMPLE_RATE


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def frames_of(count: int) -> np.ndarray:
    # 32x32, a moving gradient so the encoder does real work on every frame.
    base = np.linspace(0, 255, 32, dtype=np.float32)
    video = np.empty((count, 32, 32, 3), dtype=np.uint8)
    for i in range(count):
        video[i] = ((base[None, :, None] + i * 3) % 256).astype(np.uint8)
    return video


def tone(seconds: float) -> np.ndarray:
    t = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    wave = 0.2 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    return np.stack([wave, wave])


def streams(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_packets", "-show_entries",
         "stream=codec_type,nb_read_packets,duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return {s["codec_type"]: s for s in json.loads(out)["streams"]}


def case(tmp: Path, name: str, frames: int, audio_seconds: float | None, **kw) -> bool:
    path = tmp / f"{name}.mp4"
    audio = None if audio_seconds is None else tone(audio_seconds)
    save_mp4(path, frames_of(frames), FPS, audio, SAMPLE_RATE, **kw)
    info = streams(path)
    video = info["video"]
    ok = check(f"{name}: {frames} frames encoded", int(video["nb_read_packets"]) == frames,
               f"ffprobe {video['nb_read_packets']}")
    ok &= check(f"{name}: probe_video_frames agrees", probe_video_frames(path) == frames,
                str(probe_video_frames(path)))
    if audio is not None:
        picture = frames / FPS
        sound = float(info["audio"]["duration"])
        ok &= check(
            f"{name}: audio ends with the picture",
            abs(sound - picture) <= AAC_FRAME_S + 1e-6,
            f"video {picture:.3f}s audio {sound:.3f}s",
        )
    return ok


def main() -> int:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("SKIP: ffmpeg/ffprobe not on PATH")
        return 0
    version = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout
    print(f"ffmpeg: {shutil.which('ffmpeg')} ({version.splitlines()[0]})")
    passed = True
    with tempfile.TemporaryDirectory(prefix="h3-mux-") as tmp_name:
        tmp = Path(tmp_name)
        for frames in (73, 124):
            seconds = frames / FPS
            # The model's own audio is a hair longer than the picture (the 42 ms the audit saw).
            passed &= case(tmp, f"f{frames}_long_audio", frames, seconds + 0.042)
            passed &= case(tmp, f"f{frames}_exact_audio", frames, seconds)
            # A crossfaded / stretched waveform can end early: pad, never shorten the picture.
            passed &= case(tmp, f"f{frames}_short_audio", frames, seconds - 0.3)
            passed &= case(tmp, f"f{frames}_no_audio", frames, None)
            # --playback-fps path: ffmpeg atempo, half speed, twice the audio.
            passed &= case(tmp, f"f{frames}_atempo", frames, seconds / 2, audio_tempo=0.5)
        passed &= case(tmp, "f1_single_frame", 1, 1 / FPS + 0.042)

        swift = ROOT / "scripts" / "time_stretch_audio.swift"
        if Path("/usr/bin/swift").is_file() and swift.is_file():
            # The draft path in generate_staged: 73 decoded frames spread over 124 delivered, the
            # waveform stretched by the same ratio plus a 50 ms tail that the mux cuts.
            delivered, source = 124, 73
            passed &= case(
                tmp, "draft_stretch_124", delivered, source / FPS + 0.042,
                audio_tempo=source / delivered, audio_stretch_script=swift,
                audio_output_frames=round((delivered / FPS + 0.05) * SAMPLE_RATE),
            )
        else:
            print("  [SKIP] draft time stretch (no /usr/bin/swift)")

    print("\nALL MUX FRAME COUNT TESTS PASSED" if passed else "\nMUX FRAME COUNT TESTS FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
