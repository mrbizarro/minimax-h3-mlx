"""Pika H3 Flex on MLX: the six-knot ladder, the re-noise step, and the CLI contract.

CPU only, no weights. Pika H3 Flex (Pika Labs, 2026-10-29) publishes its sampling in the
release manifest: base knots 1000,750,500,250,100,40,0, video shift 12, audio shift 3, method
"renoise". This pins:

* the shifted ladders the runner builds from those knots (values computed from the release's
  shift formula), so a schedule change cannot slip in silently;
* the re-noise step: x0 = x_t + sigma * v, re-noised to the next knot with FRESH noise, and the
  terminal knot returning x0 exactly (the model is never evaluated at sigma 0);
* determinism per seed, and independence of the video and audio noise streams;
* the argparse contract: --pika-flex owns --steps, refuses --revoice-steps / --step-cache /
  --revoice-steps, and --sigma-knots must match --steps.

Run: ``python tests/test_pika_flex.py`` (or pytest).
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

from generate_staged import PIKA_FLEX_KNOTS, build_schedules, parse_sigma_knots  # noqa: E402
from minimax_h3_mlx.config import PipelineConfig  # noqa: E402
from minimax_h3_mlx.scheduler import (  # noqa: E402
    MiniMaxH3RenoiseScheduler,
    MiniMaxH3Scheduler,
    shifted_knots,
)

# s*q / (1 + (s-1)*q) at q = knot/1000.
VIDEO = [1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0]
AUDIO = [1.0, 0.9, 0.75, 0.5, 0.25, 0.111111, 0.0]


def _close(got, want, tol=2e-6):
    return len(got) == len(want) and all(abs(a - b) <= tol for a, b in zip(got, want))


def test_ladders():
    assert PIKA_FLEX_KNOTS == (1000.0, 750.0, 500.0, 250.0, 100.0, 40.0, 0.0)
    assert _close(shifted_knots(PIKA_FLEX_KNOTS, 12.0), VIDEO)
    assert _close(shifted_knots(PIKA_FLEX_KNOTS, 3.0), AUDIO)
    video, audio = build_schedules(7, PipelineConfig(), None, knots=PIKA_FLEX_KNOTS,
                                   sampler="renoise", seed=5)
    assert isinstance(video, MiniMaxH3RenoiseScheduler)
    assert isinstance(audio, MiniMaxH3RenoiseScheduler)
    assert _close(video.sigmas.tolist(), VIDEO)
    assert _close(audio.sigmas.tolist(), AUDIO)
    assert len(video.timesteps.tolist()) == 6          # six forwards
    assert video.timesteps.tolist()[0] == 0.0          # t = 1 - sigma


def test_knot_parser():
    assert parse_sigma_knots(None) is None
    assert parse_sigma_knots("1000,750,500,250,100,40,0") == PIKA_FLEX_KNOTS
    for bad in ("900,500,0", "1000,500,100", "1000,500,600,0", "1000"):
        try:
            parse_sigma_knots(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should be refused")


def test_euler_default_unchanged():
    video, audio = build_schedules(9, PipelineConfig())
    assert type(video) is MiniMaxH3Scheduler and type(audio) is MiniMaxH3Scheduler
    ref = MiniMaxH3Scheduler(shift=12.0)
    ref.set_timesteps(9)
    assert video.sigmas.tolist() == ref.sigmas.tolist()


def test_renoise_step():
    sched = MiniMaxH3RenoiseScheduler(shift=12.0, seed=3)
    sched.set_timesteps(sigmas=shifted_knots(PIKA_FLEX_KNOTS, 12.0))
    x0 = mx.random.normal((64, 96), key=mx.random.key(11))
    eps = mx.random.normal((64, 96), key=mx.random.key(12))
    sigmas = sched.sigmas.tolist()
    x = eps
    for i, t in enumerate(sched.timesteps.tolist()):
        s = sigmas[i]
        v = x0 - eps                                       # the exact data-ward velocity
        x_t = (1.0 - s) * x0 + s * eps
        out = sched.step(v, float(t), x_t)
        nxt = sigmas[i + 1]
        if nxt == 0.0:
            # The last forward returns the clean prediction itself.
            assert mx.allclose(out, x0, atol=1e-5).item()
        else:
            # Re-noised to the next knot: the mean residual against x0 is the noise term,
            # whose standard deviation is sigma_next, and it is NOT the Euler blend.
            resid = (out - (1.0 - nxt) * x0) / nxt
            std = mx.sqrt(mx.mean(resid * resid)).item()
            assert 0.9 < std < 1.1, std
            euler_next = (nxt / s) * x_t + (1 - nxt / s) * x0
            assert not mx.allclose(out, euler_next, atol=1e-3).item()
        x = out
    del x


def test_determinism_and_streams():
    def run(seed, stream):
        sched = MiniMaxH3RenoiseScheduler(shift=12.0, seed=seed, stream=stream)
        sched.set_timesteps(sigmas=shifted_knots(PIKA_FLEX_KNOTS, 12.0))
        x = mx.zeros((8, 8))
        return sched.step(mx.zeros((8, 8)), float(sched.timesteps[0].item()), x)

    assert mx.array_equal(run(7, 0), run(7, 0)).item()
    assert not mx.array_equal(run(7, 0), run(8, 0)).item()
    assert not mx.array_equal(run(7, 0), run(7, 1)).item()


def _cli(*extra):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_staged.py"), "p",
         "--dit", "x", "--compact-root", "x", "--text-config", "x",
         "-o", "/nonexistent/o.mp4", "--metrics", "/nonexistent/m.json", *extra],
        capture_output=True, text=True, timeout=120,
    )


def test_cli_contract():
    for extra, needle in (
        (("--pika-flex", "--revoice-steps", "2"), "do not compose"),
        (("--pika-flex", "--step-cache", "0.1"), "do not compose"),
        (("--sigma-knots", "1000,500,0", "--steps", "9"), "pass --steps 3"),
        (("--sigma-knots", "1000,500,0", "--sigma-subset", "50:0,49", "--steps", "3"),
         "--sigma-subset keeps"),
        (("--video-shift", "0"), "must be positive"),
    ):
        r = _cli(*extra)
        assert r.returncode == 2 and needle in r.stderr, (extra, r.stderr[-400:])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("ALL OK")
