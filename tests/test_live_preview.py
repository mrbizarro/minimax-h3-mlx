"""Gate the live-preview file contract, because a UI will be coded against it.

The panel polls `status.json` and drops an `ABORT` file; both are promises, and a promise a test
does not hold is a promise the next refactor breaks. Everything here runs without the DiT; the
monitor half is skipped when the 23 MB TAE checkpoint is not on this machine.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.live_preview import (  # noqa: E402
    ABORT_EXIT_CODE,
    SCHEMA,
    LivePreviewAborted,
    LivePreviewMonitor,
    approximate_output_frame,
    auto_downscale,
    local_output_index,
)

TAE = Path(__file__).resolve().parents[2] / "models" / "tae" / "taeh3.safetensors"

failures = 0


def check(condition: bool, label: str, detail: str = "") -> None:
    global failures
    print(f"{'ok   ' if condition else 'FAIL '} {label}{(' — ' + detail) if detail else ''}")
    if not condition:
        failures += 1


# -- pure geometry ------------------------------------------------------------------------------

# The video VAE's (1, 4, 4, 4, 4) grouping: 5 latent frames per 17 pixel frames.
check(approximate_output_frame(0) == 0, "latent 0 is pixel frame 0")
check(approximate_output_frame(1) == 1, "latent 1 is pixel frame 1")
check(approximate_output_frame(4) == 13, "latent 4 is pixel frame 13")
check(approximate_output_frame(5) == 17, "latent 5 opens the second chunk at pixel 17")
check(approximate_output_frame(18) == 60, "the middle latent of a 124-frame clip is pixel 60")

check(local_output_index(0) == 0, "a 1-token window reads output 0")
check(local_output_index(1) == 1, "a 2-token window reads output 1")
check(local_output_index(4) == 13, "a 5-token window reads output 13")

check(auto_downscale(24, 40) == 1, "the draft tier is small enough to preview unpooled")
check(auto_downscale(36, 64) == 2, "1024x576 pools once")
check(auto_downscale(48, 84) == 2, "Native 1344x768 pools once")
check(auto_downscale(24, 40, budget=100) == 4, "a tiny budget keeps halving while both axes allow")
# 22 is not divisible by 4, so pooling must stop at 2 even though the budget is still exceeded —
# a preview that is too big is a cost, a preview whose reshape is wrong is a crash mid-render.
check(auto_downscale(22, 38, budget=100) == 2, "pooling stops when an axis cannot halve again")

check(ABORT_EXIT_CODE not in (0, 1), "the abort exit code is distinguishable from done and crashed")

# -- the file contract --------------------------------------------------------------------------

if not TAE.is_file():
    print(f"\nSKIP monitor contract: no TAE checkpoint at {TAE}")
else:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "live"
        # A sentinel left by an earlier job must not kill this one.
        directory.mkdir(parents=True)
        (directory / "ABORT").write_text("")

        monitor = LivePreviewMonitor(
            directory,
            TAE,
            total_forwards=3,
            total_windows=1,
            sigma_points=4,
            output=Path(tmp) / "clip.mp4",
        )
        check(monitor._stale_abort_cleared, "a stale ABORT is cleared at startup")
        check(not monitor.abort_path.exists(), "the stale sentinel is gone")

        status = json.loads(monitor.status_path.read_text())
        for field in (
            "schema", "status", "aborted", "forward", "total_forwards", "window",
            "total_windows", "abort_sentinel", "output", "pid", "updated_at",
        ):
            check(field in status, f"status.json carries `{field}`")
        check(status["schema"] == SCHEMA, "status.json declares its schema", status["schema"])
        check(status["aborted"] is False, "a fresh run is not aborted")

        monitor.check_abort("no sentinel")
        monitor.abort_path.write_text("")
        try:
            monitor.check_abort("sentinel dropped")
        except LivePreviewAborted as exc:
            aborted = json.loads(monitor.status_path.read_text())
            check(aborted["aborted"] is True, "status.json flips aborted to true")
            check(aborted["status"] == "aborted", "status.json's status is 'aborted'")
            check(not monitor.abort_path.exists(), "the sentinel is consumed, not left to re-fire")
            check("ABORT" in str(exc), "the exception says what happened", str(exc)[:60])
        else:
            check(False, "the sentinel stopped the run")

print()
if failures:
    print(f"{failures} LIVE PREVIEW CONTRACT CHECK(S) FAILED")
    raise SystemExit(1)
print("live preview contract passed")
