"""Live TAE preview of a running render, and the early-abort contract that goes with it.

A delivery render is minutes to an hour of GPU time, and its composition — who is in frame, how
they are framed, where the camera sits — is decided in the first forward or two. Today the only
way to learn that a take is wrong is to wait for it. This module turns each forward's *current
denoised estimate* into a thumbnail on disk, so a watcher (a UI, a shell loop, an eyeball) can
stop a bad take at minute two instead of minute fifty.

Two properties are load-bearing:

* **It is read-only.** The x0 estimate is recomputed here from the sampler's own algebra
  (``x0 = x_t + (1 - t) * v``, :meth:`minimax_h3_mlx.scheduler.MiniMaxH3Scheduler.step`); the
  scheduler object is never called, so its ``_step_index`` never advances and no tensor the
  denoiser owns is written. A render with the preview on must be byte-identical to the same
  render with it off.
* **Every file is written atomically** (write to ``<name>.tmp<pid>``, ``fsync``, ``os.replace``).
  A watcher polling at any frequency either sees the previous complete file or the next complete
  file, never a torn one.

The abort half is a sentinel file, not a signal: the runner checks for ``live/ABORT`` between
forwards and stops cleanly if it is there. A file is the one channel that works across a UI, a
shell, an ssh session and a supervisor without any of them holding the process handle.

See ``notes/LIVE_PREVIEW_2026-08-11.md`` for the file contract a panel is expected to code against.
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .packing import FRAMES_PER_CHUNK, LATENTS_PER_CHUNK, unpatchify_video_tokens

#: ``status.json``'s ``schema`` field. Bump it if a field changes meaning; add fields freely.
SCHEMA = "h3-live-preview/1"

#: Exit code for a render stopped by the ABORT sentinel. Distinct from 0 (done) and from 1
#: (the runner's ordinary traceback exit), so a supervisor can tell "the user stopped this"
#: apart from "this crashed" without parsing anything.
ABORT_EXIT_CODE = 75

ABORT_FILENAME = "ABORT"
STATUS_FILENAME = "status.json"
LATEST_FILENAME = "preview_latest.png"

#: The animated companion to LATEST_FILENAME: the warm-up frames the decode already
#: produced, written as one looping WebP. Same decode, so it costs no GPU.
LOOP_FILENAME = "preview_latest.webp"
#: Milliseconds per loop frame. The decoder emits four pixel frames per latent token
#: at 25 fps, so 40 ms plays the moments back at the speed they were generated at.
LOOP_FRAME_MS = 40
#: A monitor, not a deliverable — trade bytes and encode time for fidelity.
LOOP_QUALITY = 60


class LivePreviewAborted(RuntimeError):
    """Raised between forwards when the ABORT sentinel appears."""


def _atomic_write(path: Path, payload: bytes) -> None:
    """Write ``payload`` so a concurrent reader never observes a partial file."""
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


#: Latent frames of causal warm-up handed to the tiny decoder ahead of the previewed frame.
#: ``MemBlock`` remembers the previous frame's input, so a frame decoded alone is decoded as if it
#: were the first frame of the clip and smears. Measured against the full-sequence decode on real
#: Stage-A latents (``scripts/probe_preview_context.py``): 0 -> mean |diff| 32.0/255, 1 -> 18.1,
#: 2 -> 6.0, **4 -> 0.53**, i.e. 4 is indistinguishable. It is also the last free step — the decoder
#: pads its input to a multiple of five tokens, so 3, 4 and 5 tokens all cost one padded chunk.
DEFAULT_CONTEXT = 4

#: Latent cells (``latent_h * latent_w``) the preview decode is allowed to work on before it starts
#: pooling. The tiny decoder's transient working set scales with the *pixel* area it produces, and
#: measured on this M4 Max it is large: 3.10 GiB at the draft tier's 40x24 latent, 6.37 GiB at
#: 64x36, **10.70 GiB at Native's 84x48**. A Native render already peaks near 50 GiB on a 64 GB
#: machine, so an unbounded preview could be the allocation that pushes it into swap — a monitor
#: that slows the render it is monitoring is worse than no monitor. Above this budget the latent is
#: 2x2 average-pooled first, which costs 4x less memory and 4x less time for a half-size thumbnail.
#: 1200 is the smallest budget that leaves the draft tier (960 cells) untouched and still brings
#: Native (4032) under in a single halving (1008).
LATENT_CELL_BUDGET = 1200


def auto_downscale(latent_height: int, latent_width: int, budget: int = LATENT_CELL_BUDGET) -> int:
    """Smallest power-of-two pooling factor that fits the budget and divides both latent axes."""
    factor = 1
    while (latent_height // factor) * (latent_width // factor) > budget:
        nxt = factor * 2
        if latent_height % nxt or latent_width % nxt:
            break
        factor = nxt
    return factor


def pool_latent(latents: mx.array, factor: int) -> mx.array:
    """2x2-style average pool of ``(B, C, T, H, W)`` latents on the spatial axes."""
    if factor <= 1:
        return latents
    b, c, t, h, w = latents.shape
    return latents.reshape(b, c, t, h // factor, factor, w // factor, factor).mean(axis=(4, 6))


def local_output_index(context: int) -> int:
    """Output frame carrying the last latent of a ``context + 1`` token window.

    The decoder emits four raw frames per latent token and drops the three causal lead-in frames of
    every 20-frame chunk, so the tokens land on outputs ``0, 1, 5, 9, 13`` — the ``(1, 4, 4, 4, 4)``
    grouping the video VAE encodes with.
    """
    return 0 if context == 0 else 1 + 4 * (context - 1)


def approximate_output_frame(latent_frame: int) -> int:
    """First delivered pixel frame covered by ``latent_frame``.

    The video VAE groups ``17`` pixel frames into ``5`` latent frames with the spans
    ``(1, 4, 4, 4, 4)``, so a latent index maps to a pixel index without any decoding. Reported in
    ``status.json`` purely so a viewer knows *which* moment of the clip it is looking at.
    """
    chunk, position = divmod(int(latent_frame), LATENTS_PER_CHUNK)
    offset = 0 if position == 0 else 1 + 4 * (position - 1)
    return chunk * FRAMES_PER_CHUNK + offset


class LivePreviewMonitor:
    """Per-forward TAE thumbnails plus the abort sentinel, for one whole run.

    One monitor spans every window of a chain, so ``forward``/``total_forwards`` in
    ``status.json`` count the whole job rather than restarting per window — which is what a
    progress bar wants.
    """

    def __init__(
        self,
        directory: Path,
        tae_checkpoint: Path,
        *,
        total_forwards: int,
        total_windows: int,
        sigma_points: int,
        output: Path,
        every: int = 1,
        latent_frame: int | None = None,
        context: int = DEFAULT_CONTEXT,
        downscale: int = 0,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.abort_path = self.directory / ABORT_FILENAME
        self.status_path = self.directory / STATUS_FILENAME
        self.latest_path = self.directory / LATEST_FILENAME
        self.loop_path = self.directory / LOOP_FILENAME
        self._loop_frames = 0

        self.total_forwards = int(total_forwards)
        self.total_windows = int(total_windows)
        self.sigma_points = int(sigma_points)
        self.output = Path(output)
        self.every = max(1, int(every))
        self.requested_latent_frame = latent_frame
        self.context = max(0, int(context))
        self.requested_downscale = max(0, int(downscale))
        self.downscale = 1

        self.forward = 0
        self.window = 0
        self.forward_seconds: list[float] = []
        self.overhead_seconds: list[float] = []
        self.started_at = time.time()
        self._started_perf = time.perf_counter()
        self._geometry: dict | None = None
        self._last_preview: Path | None = None
        self._stale_abort_cleared = False

        # A sentinel left behind by a previous job would kill this one before its first forward.
        # Clearing it here is the only sane default: ABORT means "stop the render I am watching",
        # and this render did not exist when that file was made.
        if self.abort_path.exists():
            self.abort_path.unlink()
            self._stale_abort_cleared = True

        from .tiny_video_vae import load_tiny_h3_video_decoder

        load_started = time.perf_counter()
        self.tae = load_tiny_h3_video_decoder(tae_checkpoint)
        self.tae_load_seconds = time.perf_counter() - load_started
        self.tae_checkpoint = Path(tae_checkpoint)

        self.write_status("starting")

    # -- geometry -------------------------------------------------------------------------------

    def start_window(
        self,
        *,
        window: int,
        latent_frames: int,
        latent_height: int,
        latent_width: int,
        patch: tuple[int, int, int],
    ) -> None:
        """Record the window's latent geometry and pick the frame to preview.

        Raises if the patch is not the shipped ``(1, 2, 2)``: with a temporal patch the packed rows
        of one latent frame are no longer a contiguous block, and slicing them the easy way would
        silently preview the wrong moment.
        """
        pt, ph, pw = patch
        if pt != 1:
            raise ValueError(
                f"live preview needs a temporal patch of 1 to slice one latent frame, got {patch}"
            )
        frame = (
            latent_frames // 2
            if self.requested_latent_frame is None
            else int(self.requested_latent_frame)
        )
        if not 0 <= frame < latent_frames:
            raise ValueError(
                f"--live-preview-latent-frame {frame} is outside this render's 0..{latent_frames - 1}"
            )
        self.downscale = (
            auto_downscale(latent_height, latent_width)
            if self.requested_downscale == 0
            else max(1, self.requested_downscale)
        )
        if latent_height % self.downscale or latent_width % self.downscale:
            raise ValueError(
                f"--live-preview-downscale {self.downscale} does not divide this render's "
                f"{latent_width}x{latent_height} latent"
            )
        self.window = int(window)
        self._geometry = {
            "latent_frames": int(latent_frames),
            "latent_height": int(latent_height),
            "latent_width": int(latent_width),
            "patch": (int(pt), int(ph), int(pw)),
            "rows_per_frame": (latent_height // ph) * (latent_width // pw),
            "latent_frame": frame,
            "preview_width": latent_width // self.downscale * 16,
            "preview_height": latent_height // self.downscale * 16,
            "approx_output_frame": approximate_output_frame(frame),
        }

    # -- abort ----------------------------------------------------------------------------------

    def check_abort(self, stage: str = "between forwards") -> None:
        if self.abort_path.exists():
            # Consume it: the sentinel means "stop this render", and leaving it behind would abort
            # whatever runs into the same directory next.
            try:
                self.abort_path.unlink()
            except FileNotFoundError:
                pass
            self.write_status("aborted", extra={"aborted_at_stage": stage})
            raise LivePreviewAborted(
                f"live preview ABORT sentinel seen {stage} "
                f"(forward {self.forward}/{self.total_forwards}); stopping before any output is "
                f"written. Status: {self.status_path}"
            )

    # -- per forward ----------------------------------------------------------------------------

    def after_forward(
        self,
        *,
        video_rows: mx.array,
        video_pred: mx.array,
        n_cond_v: int,
        timestep: float,
        forward_seconds: float,
    ) -> float:
        """Decode and publish this forward's x0 thumbnail. Returns the seconds it cost.

        ``video_rows`` must be the sequence **as the forward saw it** (x_t), and ``video_pred`` the
        velocity the forward returned. Both are read, neither is written.
        """
        self.forward += 1
        self.forward_seconds.append(float(forward_seconds))
        if self._geometry is None:
            raise RuntimeError("start_window() must run before the first forward")

        if self.forward % self.every and self.forward != self.total_forwards:
            self.write_status("running", extra={"skipped_preview": True})
            self.overhead_seconds.append(0.0)
            return 0.0

        started = time.perf_counter()
        geometry = self._geometry
        rows_per_frame = geometry["rows_per_frame"]
        target = geometry["latent_frame"]
        first = max(0, target - self.context)
        context = target - first
        tokens = context + 1
        lo = n_cond_v + first * rows_per_frame
        hi = n_cond_v + (target + 1) * rows_per_frame

        # The sampler's own x0: `denoised = x_t + (1 - t) * v`, in the scheduler's float32 rounding
        # (minimax_h3_mlx/scheduler.py:146). Recomputed here rather than taken from the scheduler,
        # because calling scheduler.step() a second time would advance its step index.
        sigma = float(np.float32(1.0) - np.float32(timestep))
        x0_rows = video_rows[lo:hi].astype(mx.float32) + sigma * video_pred[lo:hi].astype(
            mx.float32
        )
        latents = unpatchify_video_tokens(
            x0_rows,
            tokens,
            geometry["latent_height"],
            geometry["latent_width"],
            self.tae.latent_channels,
            geometry["patch"],
        )
        latents = pool_latent(latents, self.downscale)
        index = local_output_index(context)
        decoded = self.tae.decode(latents, index + 1)
        mx.eval(decoded)
        stack = np.array(decoded)[0]  # (C, T, H, W)
        frame = stack[:, index].transpose(1, 2, 0)
        image = (np.clip(frame, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG", compress_level=1)
        payload = buffer.getvalue()

        path = self.directory / f"preview_{self.forward:02d}.png"
        _atomic_write(path, payload)
        _atomic_write(self.latest_path, payload)
        self._last_preview = path

        # THE LOOP IS ALREADY DECODED. `context` latent tokens of causal warm-up
        # exist so the target frame does not smear (see DEFAULT_CONTEXT), and the
        # decoder emits a PIXEL frame for every one of them — `index + 1` frames
        # in `stack`, of which the single line above used exactly one and threw
        # the rest away. They are the moments immediately BEFORE the previewed
        # one, in order, so writing them out as an animated WebP turns the still
        # into ~half a second of real motion for NO extra GPU work: same decode,
        # same tensors, only a handful of extra encodes.
        #
        # WebP because the panel already serves image/webp and a browser loops an
        # animated one on its own — no polling, no player, no client state. The
        # PNG above stays exactly as it was, so any consumer that wants a still
        # (and every older panel) is untouched.
        if stack.shape[1] > 1:
            try:
                seq = np.clip(stack.transpose(1, 2, 3, 0), 0.0, 1.0)
                seq = (seq * 255.0 + 0.5).astype(np.uint8)
                frames = [Image.fromarray(f) for f in seq]
                loop_buffer = io.BytesIO()
                frames[0].save(
                    loop_buffer,
                    format="WEBP",
                    save_all=True,
                    append_images=frames[1:],
                    duration=LOOP_FRAME_MS,
                    loop=0,
                    quality=LOOP_QUALITY,
                    method=0,  # fastest encoder setting: this is a monitor, not a deliverable
                )
                _atomic_write(self.loop_path, loop_buffer.getvalue())
                self._loop_frames = len(frames)
            except Exception:
                # A monitor must never be able to fail a render. If WebP is
                # unavailable in this Pillow build, the still above already
                # shipped and the panel falls back to it.
                self._loop_frames = 0

        del x0_rows, latents, decoded
        cost = time.perf_counter() - started
        self.overhead_seconds.append(cost)
        self.write_status("running", extra={"preview_seconds": round(cost, 4)})
        return cost

    # -- status ---------------------------------------------------------------------------------

    def write_status(self, status: str, extra: dict | None = None) -> None:
        done = len(self.forward_seconds)
        mean = sum(self.forward_seconds) / done if done else None
        remaining = max(0, self.total_forwards - self.forward)
        payload = {
            "schema": SCHEMA,
            "status": status,
            "aborted": status == "aborted",
            "forward": self.forward,
            "total_forwards": self.total_forwards,
            "window": self.window,
            "total_windows": self.total_windows,
            "sigma_points": self.sigma_points,
            "preview": self._last_preview.name if self._last_preview else None,
            # The animated companion, named only once it exists. A consumer that
            # does not know the key keeps using `preview` and sees a still.
            "preview_loop": LOOP_FILENAME if self._loop_frames else None,
            "preview_loop_frames": self._loop_frames or None,
            "preview_path": str(self._last_preview) if self._last_preview else None,
            "preview_latest_path": str(self.latest_path) if self._last_preview else None,
            "abort_sentinel": str(self.abort_path),
            "output": str(self.output),
            "pid": os.getpid(),
            "started_at": round(self.started_at, 3),
            "updated_at": round(time.time(), 3),
            "elapsed_seconds": round(time.perf_counter() - self._started_perf, 3),
            "mean_forward_seconds": round(mean, 3) if mean else None,
            "eta_seconds": round(mean * remaining, 1) if mean else None,
            "preview_overhead_seconds": round(sum(self.overhead_seconds), 3),
            "tae_load_seconds": round(self.tae_load_seconds, 3),
            "tae_checkpoint": str(self.tae_checkpoint),
            "stale_abort_cleared": self._stale_abort_cleared,
            "every": self.every,
            "context": self.context,
            "downscale": self.downscale,
        }
        if self._geometry is not None:
            payload.update(
                {
                    "latent_frame": self._geometry["latent_frame"],
                    "latent_frames": self._geometry["latent_frames"],
                    "approx_output_frame": self._geometry["approx_output_frame"],
                    "preview_width": self._geometry["preview_width"],
                    "preview_height": self._geometry["preview_height"],
                }
            )
        if extra:
            payload.update(extra)
        _atomic_write(self.status_path, (json.dumps(payload, indent=2) + "\n").encode())

    def finish(self, status: str = "done", extra: dict | None = None) -> None:
        self.write_status(status, extra=extra)

    def summary(self) -> dict:
        overhead = sum(self.overhead_seconds)
        written = sum(1 for value in self.overhead_seconds if value > 0.0)
        return {
            "directory": str(self.directory),
            "previews_written": written,
            "forwards": self.forward,
            "every": self.every,
            "context": self.context,
            "downscale": self.downscale,
            "latent_frame": self._geometry["latent_frame"] if self._geometry else None,
            "preview_size": (
                [self._geometry["preview_width"], self._geometry["preview_height"]]
                if self._geometry
                else None
            ),
            "tae_load_seconds": round(self.tae_load_seconds, 3),
            "overhead_seconds": round(overhead, 3),
            "overhead_seconds_per_preview": (
                round(overhead / written, 4) if written else None
            ),
            "per_forward_overhead_seconds": [round(value, 4) for value in self.overhead_seconds],
        }
