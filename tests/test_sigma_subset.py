"""``--sigma-subset``: a step-distilled adapter's published ladder, not a re-spaced grid.

CPU only, no weights. TaoMate-H3 (TaoLiveAIGC) distils H3 onto points (0, 16, 33, 49) of the
shifted 50-point grid — 3 forwards. This pins the parser, the schedule it builds for video (shift
12) and audio (shift 3) against the values the TaoMate recipe lists, that the default path (no
subset) is unchanged, and the argparse contract with ``--steps``.

Run: ``python tests/test_sigma_subset.py`` (or pytest).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)

from generate_staged import build_schedules, parse_sigma_subset  # noqa: E402
from minimax_h3_mlx.config import PipelineConfig  # noqa: E402

TAOMATE = "50:0,16,33,49"
# From the TaoMate recipe (video shift 12, audio shift 3), 3 forwards each.
VIDEO = [1.0, 0.96117, 0.85333, 0.0]
AUDIO = [1.0, 0.86087, 0.59259, 0.0]


def _close(got, want, tol=5e-5):
    return len(got) == len(want) and all(abs(a - b) <= tol for a, b in zip(got, want))


def test_parse():
    assert parse_sigma_subset(None) is None
    assert parse_sigma_subset("") is None
    assert parse_sigma_subset(TAOMATE) == (50, (0, 16, 33, 49))
    for bad in ("50:1,16,49", "50:0,16,33", "50:0,33,16,49", "50:0,16,16,49", "50:0", "x:0,1"):
        try:
            parse_sigma_subset(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should be refused")


def test_taomate_ladder():
    video, audio = build_schedules(4, PipelineConfig(), parse_sigma_subset(TAOMATE))
    v, a = video.sigmas.tolist(), audio.sigmas.tolist()
    assert _close(v, VIDEO), v
    assert _close(a, AUDIO), a
    # 3 forwards: one timestep per interval, terminal 0 excluded.
    assert len(video.timesteps.tolist()) == 3
    assert len(audio.timesteps.tolist()) == 3


def test_default_unchanged():
    cfg = PipelineConfig()
    video, audio = build_schedules(9, cfg)
    video2, audio2 = build_schedules(9, cfg, None)
    assert video.sigmas.tolist() == video2.sigmas.tolist()
    assert audio.sigmas.tolist() == audio2.sigmas.tolist()
    assert len(video.timesteps.tolist()) == 8


def test_cli_contract():
    script = str(ROOT / "scripts" / "generate_staged.py")
    base = [sys.executable, script, "p", "--dit", "/nonexistent", "--compact-root", "/nonexistent",
            "--text-config", "/nonexistent", "-o", "/nonexistent.mp4", "--metrics", "/nonexistent.json"]
    # --steps must equal the number of kept points; refused by argparse (exit 2) before any load.
    bad = subprocess.run(base + ["--steps", "9", "--sigma-subset", TAOMATE],
                         capture_output=True, text=True, timeout=120)
    assert bad.returncode == 2 and "pass --steps 4" in bad.stderr, bad.stderr[-400:]
    bad = subprocess.run(base + ["--steps", "4", "--sigma-subset", "50:1,16,33,49"],
                         capture_output=True, text=True, timeout=120)
    assert bad.returncode == 2 and "--sigma-subset" in bad.stderr, bad.stderr[-400:]
    helptext = subprocess.run([sys.executable, script, "--help"], capture_output=True,
                              text=True, timeout=120).stdout
    assert "--sigma-subset" in helptext and "--vae-dtype" in helptext


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  [PASS] {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  [FAIL] {name}: {exc}")
    sys.exit(1 if failures else 0)
