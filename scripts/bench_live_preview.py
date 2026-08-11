"""Price the live TAE preview before spending a render on it.

The whole case for the live preview is that it costs almost nothing, so that claim is the first
thing to falsify. This runs the exact preview path the runner runs — x0 from (x_t, v), one latent
frame unpatchified, TAE-decoded, PNG-encoded, atomically published — on synthetic rows at real
geometries, and reports seconds per preview against the measured seconds per forward at each tier.

It also exercises the abort sentinel and the status.json schema end to end, so the file contract is
tested without holding a 38 GB DiT resident.

Usage:
    bench_live_preview.py --tae-checkpoint models/tae/taeh3.safetensors [--repeats 5]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.live_preview import LivePreviewAborted, LivePreviewMonitor  # noqa: E402
from minimax_h3_mlx.packing import video_latent_num_frames  # noqa: E402

PATCH = (1, 2, 2)
LATENT_CHANNELS = 24

# (name, width, height, frames, measured seconds per forward on this machine)
# The forward times are real measurements: draft and high from this campaign's metrics JSONs, the
# native figure is the conservative floor implied by the 10 s Native renders in the notes.
TIERS = (
    ("draft 640x384 / 124f / 3 forwards", 640, 384, 124, 40.3),
    ("high 1024x576 / 124f / 3 forwards", 1024, 576, 124, 126.3),
    ("native 1344x768 / 124f / 3 forwards", 1344, 768, 124, 300.0),
)


def synthetic_rows(latent_frames: int, latent_h: int, latent_w: int, seed: int):
    """Packed video rows shaped exactly like the runner's, filled with plausible noise."""
    rows_per_frame = (latent_h // PATCH[1]) * (latent_w // PATCH[2])
    total = latent_frames * rows_per_frame
    width = LATENT_CHANNELS * PATCH[0] * PATCH[1] * PATCH[2]
    mx.random.seed(seed)
    x_t = mx.random.normal((total, width)).astype(mx.float32)
    v = mx.random.normal((total, width)).astype(mx.float32)
    mx.eval(x_t, v)
    return x_t, v


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tae-checkpoint", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/h3_live_preview_bench"))
    args = parser.parse_args()

    results = []
    for name, width, height, frames, forward_seconds in TIERS:
        latent_h, latent_w = height // 16, width // 16
        latent_frames = video_latent_num_frames(frames)
        x_t, v = synthetic_rows(latent_frames, latent_h, latent_w, seed=7)

        directory = args.work_dir / name.split()[0]
        monitor = LivePreviewMonitor(
            directory,
            args.tae_checkpoint,
            total_forwards=args.repeats,
            total_windows=1,
            sigma_points=4,
            output=directory / "unused.mp4",
        )
        monitor.start_window(
            window=1,
            latent_frames=latent_frames,
            latent_height=latent_h,
            latent_width=latent_w,
            patch=PATCH,
        )
        # Warm the kernels once; the first call compiles Metal pipelines that every later call
        # reuses, and charging that to the average would flatter nothing and mislead everything.
        monitor.after_forward(
            video_rows=x_t, video_pred=v, n_cond_v=0, timestep=0.5, forward_seconds=forward_seconds
        )
        mx.clear_cache()
        mx.reset_peak_memory()
        resident = mx.get_active_memory()
        timings = []
        for index in range(args.repeats):
            started = time.perf_counter()
            monitor.after_forward(
                video_rows=x_t,
                video_pred=v,
                n_cond_v=0,
                timestep=0.5 + index * 0.01,
                forward_seconds=forward_seconds,
            )
            timings.append(time.perf_counter() - started)
        monitor.finish("done")

        transient_gib = (mx.get_peak_memory() - resident) / 1024**3
        median = float(np.median(timings))
        results.append(
            {
                "tier": name,
                "canvas": f"{width}x{height}",
                "latent": f"{latent_w}x{latent_h}",
                "latent_frames": latent_frames,
                "context": monitor.context,
                "downscale": monitor.downscale,
                "preview_png": (
                    f"{latent_w // monitor.downscale * 16}x{latent_h // monitor.downscale * 16}"
                ),
                "transient_gib": round(transient_gib, 2),
                "tae_load_seconds": round(monitor.tae_load_seconds, 3),
                "median_seconds": round(median, 4),
                "min_seconds": round(float(np.min(timings)), 4),
                "max_seconds": round(float(np.max(timings)), 4),
                "forward_seconds": forward_seconds,
                "percent_of_forward": round(100.0 * median / forward_seconds, 3),
            }
        )
        del x_t, v, monitor
        mx.clear_cache()

    print(json.dumps(results, indent=2))

    # The abort contract, exercised without a render: drop the sentinel, assert the next check
    # raises, assert status.json says aborted, assert the sentinel was consumed.
    directory = args.work_dir / "abort"
    monitor = LivePreviewMonitor(
        directory,
        args.tae_checkpoint,
        total_forwards=3,
        total_windows=1,
        sigma_points=4,
        output=directory / "unused.mp4",
    )
    monitor.check_abort("no sentinel present")
    monitor.abort_path.write_text("")
    try:
        monitor.check_abort("sentinel present")
    except LivePreviewAborted as exc:
        status = json.loads(monitor.status_path.read_text())
        assert status["aborted"] is True, status
        assert status["status"] == "aborted", status
        assert not monitor.abort_path.exists(), "sentinel was not consumed"
        print(f"\nabort contract OK: {exc}")
    else:  # pragma: no cover - the assertion is the test
        raise SystemExit("ABORT sentinel did not stop the run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
