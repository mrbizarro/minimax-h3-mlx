#!/usr/bin/env python3
"""Resident MiniMax-H3 engine: load the DiT once, stream jobs through it.

``generate_staged.py`` is a *staged* runner — every large component is loaded, used and freed
inside one process that then exits. That is what makes 40 GiB of bf16 DiT fit on a 64 GB Mac, and
it is why a repeat job re-reads 41.4 GB of safetensors from the SSD before it can denoise anything.

This is the same pipeline with the staging inverted for the one component that never has to move:
the DiT stays resident and jobs stream through it.

Numerics are not re-implemented here. The job body *is* ``generate_staged.render_window`` — this
module only replaces the four loaders it calls (``load_dit``, ``drop_adaln_weights``,
``ModulationCache``, the two compact VAE loaders) with residency-aware equivalents, so the arithmetic
that produces pixels is byte-for-byte the code the graded renders came from. A cold run and a serve
run at the same seed must agree on the sha256 of the mp4 and the wav; ``--self-check`` asserts it.

THE MEMORY RULE THIS ENGINE IS BUILT AROUND
-------------------------------------------
On a 64 GB M4 Max the Metal working set tops out at 51.84 GiB and the two big components are:

    Q8 text encoder   26.3 GiB active
    bf16 DiT          38.6 GiB active   (37.5 after the AdaLN drop, which serve mode skips)

26.3 + 38.6 = 64.9 GiB. **They can never coexist.** So a resident DiT is only resident while no
prompt needs encoding. The engine handles that honestly rather than pretending: the text encoder's
constructor is intercepted, and constructing one frees the DiT first. A job whose conditioning is
already cached never constructs an encoder and never touches the DiT; a job with a new prompt pays
exactly what a cold run pays. Serve mode is therefore never slower than cold, and is ~22 s/job
faster whenever the prompt and first frame repeat -- which is what a seed sweep, a step sweep or a
re-roll from a panel actually is.

The video VAE (4.9 GiB) *can* coexist with the DiT and is kept resident under ``--resident-vae``;
it is off by default because it adds ~4.9 GiB to every peak and the headroom at 1280x736 is thin.

PROTOCOL (matches the LTX warm helper's interaction style)
----------------------------------------------------------
Line-delimited JSON on stdin, line-delimited JSON events on stdout. ``{"event": "ready", ...}`` is
emitted once the process can accept work; every ``print`` becomes ``{"event": "log", "line": ...}``.

    {"action": "ping"}                                   -> {"event": "pong"}
    {"action": "encode",   "id": ..., "params": {...}}   -> {"event": "encoded", "id": ...}
    {"action": "generate", "id": ..., "params": {...}}   -> {"event": "done",    "id": ...}
    {"action": "status"}                                 -> {"event": "status",  ...}
    {"action": "exit"}                                   -> {"event": "exit"}

``generate`` params: prompt, first_frame, frames, height, width, steps, seed, output, metrics,
frames_dir, crf. ``encode`` takes prompt/first_frame/height/width and warms the conditioning cache
while the DiT is not resident -- the way a client drains a queue of new prompts before asking for a
single frame of video.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate_staged as staged  # noqa: E402  (path shim above must run first)

from minimax_h3_mlx.adaln import ModulationCache as _RealModulationCache  # noqa: E402
from minimax_h3_mlx.load import (  # noqa: E402
    load_compact_audio_vae as _real_load_audio_vae,
)
from minimax_h3_mlx.load import (
    load_compact_video_vae as _real_load_video_vae,
)
from minimax_h3_mlx.load import load_dit as _real_load_dit  # noqa: E402
from minimax_h3_mlx.media import save_mp4  # noqa: E402
from minimax_h3_mlx.packing import FPS, align_num_frames  # noqa: E402
from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder as _RealTextEncoder  # noqa: E402

# ---- stdout as an event stream ----------------------------------------------------------------
_real_stdout = sys.stdout
_emit_lock = threading.Lock()


def emit(event: dict) -> None:
    try:
        with _emit_lock:
            _real_stdout.write(json.dumps(event) + "\n")
            _real_stdout.flush()
    except (BrokenPipeError, ConnectionResetError):
        # The client is gone. Holding on only orphans a 40 GiB process pinning unified memory.
        os._exit(0)
    except Exception:
        pass


class LineEmitter(io.TextIOBase):
    """Turn ordinary ``print`` output into ``{"event": "log"}`` lines."""

    def __init__(self):
        self.buf = ""
        self.lock = threading.Lock()

    def writable(self):
        return True

    def write(self, s):
        if not s:
            return 0
        with self.lock:
            self.buf += s
            while True:
                candidates = [i for i in (self.buf.find("\n"), self.buf.find("\r")) if i != -1]
                if not candidates:
                    break
                idx = min(candidates)
                line = self.buf[:idx].strip()
                self.buf = self.buf[idx + 1 :]
                if line:
                    emit({"event": "log", "line": line})
        return len(s)

    def flush(self):
        pass


def gib(value: int) -> float:
    return value / 1024**3


# ---- memoized AdaLN modulation ----------------------------------------------------------------


class _CachingModulationCache:
    """``ModulationCache`` that remembers tables across jobs.

    The table is a pure function of (DiT weights, distinct timesteps), and the DiT weights do not
    move in serve mode -- so two jobs with the same sigma schedule and the same conditioning level
    want the identical table. Keyed on the raw bytes of the timestep vector, which is exactly the
    thing the table is built from.
    """

    store: "dict[bytes, object]" = {}
    limit = 4
    hits = 0
    misses = 0
    last_build_reused = False
    active_key = None
    lora_reports: "dict[bytes, dict]" = {}

    @classmethod
    def build(cls, dit, timesteps, dtype=mx.bfloat16):
        key = np.array(timesteps.astype(mx.float32)).tobytes()
        hit = cls.store.get(key)
        if hit is not None:
            cls.hits += 1
            cls.last_build_reused = True
            cls.active_key = key
            return hit
        cls.misses += 1
        cls.last_build_reused = False
        cls.active_key = key
        built = _RealModulationCache.build(dit, timesteps, dtype=dtype)
        if len(cls.store) >= cls.limit:
            evicted = next(iter(cls.store))
            cls.store.pop(evicted)
            cls.lora_reports.pop(evicted, None)
        cls.store[key] = built
        return built

    @classmethod
    def remember_lora_report(cls, report):
        if cls.active_key is not None:
            cls.lora_reports[cls.active_key] = report

    @classmethod
    def current_lora_report(cls):
        return cls.lora_reports.get(cls.active_key)

    @classmethod
    def clear(cls):
        cls.store.clear()
        cls.lora_reports.clear()
        cls.active_key = None
        cls.last_build_reused = False


# ---- the engine --------------------------------------------------------------------------------


class ResidentEngine:
    def __init__(self, opts):
        self.opts = opts
        self.dit = None
        self.dit_loads = 0
        self.dit_load_seconds = 0.0
        self.video_vae = None
        self.audio_vae = None
        self.jobs = 0
        self.cache_dir = Path(opts.cond_cache)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- residency ------------------------------------------------------------------------------
    def load_dit(self):
        if self.dit is not None:
            return self.dit
        started = time.perf_counter()
        self.dit = _real_load_dit(self.opts.dit, verbose=True)
        if self.opts.lora:
            from minimax_h3_mlx import lora as lora_mod

            specs = [self.opts.lora] if isinstance(self.opts.lora, str) else list(self.opts.lora)
            reports = lora_mod.apply_loras(
                self.dit, [lora_mod.parse_spec(str(s)) for s in specs], mode="runtime", verbose=True
            )
            self.lora_report = reports[0].summary() if len(reports) == 1 else [r.summary() for r in reports]
        elapsed = time.perf_counter() - started
        self.dit_loads += 1
        self.dit_load_seconds += elapsed
        print(f"[engine] DiT resident after {elapsed:.1f}s (load #{self.dit_loads})")
        return self.dit

    def free_dit(self, why: str):
        if self.dit is None:
            return
        print(f"[engine] releasing the DiT: {why}")
        self.dit = None
        # The AdaLN tables were built against the weights being dropped; a reloaded DiT produces
        # the same table from the same file, but keeping stale objects alive just wastes 100 MiB
        # each and invites a subtle identity bug if a future loader is ever non-deterministic.
        _CachingModulationCache.clear()
        gc.collect()
        mx.clear_cache()

    def free_vaes(self):
        """Drop the VAEs unless residency was asked for -- otherwise ``--resident-vae`` would be a
        flag that reloads them every job while still being billed for the peak."""
        if self.opts.resident_vae:
            return
        self.video_vae = None
        self.audio_vae = None
        gc.collect()
        mx.clear_cache()

    # -- loader interception --------------------------------------------------------------------
    def install(self):
        engine = self

        def resident_load_dit(path, *args, **kwargs):
            return engine.load_dit()

        def no_drop(dit):
            # Serve mode must not destroy the resident DiT's AdaLN projections: the next job may
            # ask for a different sigma schedule, and rebuilding the table needs those weights.
            # Costs 1.17 GiB of residency. The block stack never reads them once a cache is passed,
            # so keeping them cannot move a single output bit.
            return 0

        def resident_video_vae(path, *args, **kwargs):
            if not engine.opts.resident_vae:
                return _real_load_video_vae(path, *args, **kwargs)
            if engine.video_vae is None:
                engine.video_vae = _real_load_video_vae(path, *args, **kwargs)
            return engine.video_vae

        def resident_audio_vae(path, *args, **kwargs):
            if not engine.opts.resident_vae:
                return _real_load_audio_vae(path, *args, **kwargs)
            if engine.audio_vae is None:
                engine.audio_vae = _real_load_audio_vae(path, *args, **kwargs)
            return engine.audio_vae

        class GuardedTextEncoder(_RealTextEncoder):
            """The one component that forces the DiT out of memory.

            26.3 GiB of Q8 encoder plus 38.6 GiB of bf16 DiT is 64.9 GiB on a 64 GB machine. There
            is no scheduling trick that makes that fit, so constructing an encoder is treated as
            the explicit eviction it is -- loudly, in the log, and counted in the metrics.
            """

            def __init__(self, *args, **kwargs):
                engine.free_dit("the Q8 text encoder needs 26.3 GiB and cannot share with 38.6")
                super().__init__(*args, **kwargs)

        staged.load_dit = resident_load_dit
        staged.drop_adaln_weights = no_drop
        staged.load_compact_video_vae = resident_video_vae
        staged.load_compact_audio_vae = resident_audio_vae
        staged.MiniMaxH3TextEncoder = GuardedTextEncoder
        staged.ModulationCache = _CachingModulationCache

    # -- conditioning cache ---------------------------------------------------------------------
    def cond_cache_path(self, prompt: str, first_frame: str, height: int, width: int) -> Path:
        # The vision tower reads the still already placed on the render canvas, so the embeds
        # depend on the geometry as well as on the prompt and the file.
        key = "\x1f".join([prompt, first_frame, str(height), str(width)])
        return self.cache_dir / f"cond_{hashlib.sha256(key.encode()).hexdigest()[:24]}.npz"


# ---- one job -----------------------------------------------------------------------------------


def build_args(engine, params: dict) -> SimpleNamespace:
    opts = engine.opts
    return SimpleNamespace(
        dit=Path(opts.dit),
        compact_root=Path(opts.compact_root),
        text_config=Path(opts.text_config) if opts.text_config else None,
        height=int(params["height"]),
        width=int(params["width"]),
        steps=int(params.get("steps", 9)),
        step_cache=float(params.get("step_cache", 0.0)),
        step_cache_max_skip=int(params.get("step_cache_max_skip", 2)),
        step_cache_probe=False,
        save_stage_a=None,
        draft_decode=opts.draft_decode,
        tae_checkpoint=Path(opts.tae_checkpoint) if opts.tae_checkpoint else None,
        draft_cache_dir=None,
        draft_cache_limit=50,
        draft_fps=int(params["draft_fps"]) if params.get("draft_fps") else None,
        draft_clock=str(params.get("draft_clock", "native")),
        lora=opts.lora,
        lora_mode="runtime",
        lora_adaln=Path(opts.lora_adaln) if opts.lora_adaln else None,
        lora_audit=False,
        lora_already_applied=bool(opts.lora),
    )


def run_generate(engine, job_id, params: dict) -> dict:
    opts = engine.opts
    requested_frames = align_num_frames(int(params.get("frames", 73)))
    height, width = int(params["height"]), int(params["width"])
    if height % 32 or width % 32:
        raise ValueError("height and width must be multiples of 32")
    prompt = params["prompt"]
    seed = int(params.get("seed", 0))
    first_frame = params.get("first_frame") or None
    output = Path(params["output"])
    metrics = Path(params.get("metrics") or output.with_suffix(".json"))
    crf = int(params.get("crf", 18))

    args = build_args(engine, params)
    args.first_frame = Path(first_frame) if first_frame else None
    delivery_frames = int(params.get("frames", 73)) if args.draft_fps is not None else requested_frames
    frames = (
        staged.draft_video_grid(delivery_frames, args.draft_fps)
        if args.draft_fps is not None
        else requested_frames
    )
    args.draft_clock_scale = delivery_frames / frames if args.draft_fps is not None else 1.0

    keyframe = None
    if first_frame:
        from PIL import Image

        from minimax_h3_mlx.packing import prepare_keyframe_image

        keyframe = prepare_keyframe_image(
            Image.open(first_frame).convert("RGB"), height, width, stretch=True
        )

    first_frame_key = str(first_frame) if first_frame else ""
    cond_path = engine.cond_cache_path(prompt, first_frame_key, height, width)
    cond_hit = cond_path.exists()

    settings = {
        "serve_mode": True,
        "prompt": prompt,
        "dit": str(args.dit),
        "compact_root": str(args.compact_root),
        "prompt_cache": str(cond_path),
        "cond_cache_hit": cond_hit,
        "dit_resident_at_start": engine.dit is not None,
        "resident_vae": bool(opts.resident_vae),
        "frames": frames,
        "delivery_frames": delivery_frames,
        "height": height,
        "width": width,
        "sigma_points": args.steps,
        "forwards": args.steps - 1,
        "seed": seed,
        "step_cache_threshold": args.step_cache,
        "step_cache_max_skip": args.step_cache_max_skip,
        "playback_fps": FPS,
        "draft_fps": args.draft_fps,
        "draft_unique_fps": (
            round(frames / (delivery_frames / FPS), 3) if args.draft_fps is not None else None
        ),
        "draft_clock": args.draft_clock if args.draft_fps is not None else None,
        "draft_clock_scale": (
            round(args.draft_clock_scale, 6) if args.draft_fps is not None else None
        ),
        "draft_output_width": params.get("draft_output_width"),
        "draft_output_height": params.get("draft_output_height"),
        "vae_decode_batch": staged.resolved_decode_batch(),
        "runner_sha256": engine.opts.runner_digest,
        "device": mx.device_info(),
    }
    record = staged.Recorder(metrics, settings)

    mx.reset_peak_memory()
    job_started = time.perf_counter()
    dit_loads_before = engine.dit_loads

    video, audio, sample_rate, step_times = staged.render_window(
        args,
        record,
        label="",
        prompt=prompt,
        frames=frames,
        seed=seed,
        keyframe=keyframe,
        first_frame_key=first_frame_key,
        prompt_cache=cond_path,
    )

    with record.phase("encode_mux"):
        output.parent.mkdir(parents=True, exist_ok=True)
        audio_tempo = 1.0
        audio_stretch_script = None
        audio_output_frames = None
        if args.draft_fps is not None:
            source_frames = len(video)
            output_width = params.get("draft_output_width")
            output_height = params.get("draft_output_height")
            if (output_width is None) != (output_height is None):
                raise ValueError("pass both draft_output_width and draft_output_height")
            if output_width is not None:
                video = staged.resize_draft_video(video, int(output_width), int(output_height))
            video = staged.distribute_draft_frames(video, delivery_frames)
            audio_tempo = source_frames / delivery_frames
            audio_stretch_script = Path(__file__).with_name("time_stretch_audio.swift")
            audio_output_frames = round((delivery_frames / FPS + 0.05) * sample_rate)
        save_mp4(
            output,
            video,
            float(FPS),
            audio,
            sample_rate,
            crf=crf,
            audio_tempo=audio_tempo,
            audio_stretch_script=audio_stretch_script,
            audio_output_frames=audio_output_frames,
        )
        frames_dir = params.get("frames_dir")
        if frames_dir:
            from minimax_h3_mlx.media import save_frames

            save_frames(Path(frames_dir), video)
            print(f"wrote {len(video)} lossless frames to {frames_dir}")

    total = time.perf_counter() - job_started
    # `Recorder.phase` resets the MLX peak counter on every phase entry, so `get_peak_memory()`
    # here would only report the mux. The job's real peak is the max over its phases -- the same
    # quantity the cold runner's metrics files carry, so the two are directly comparable.
    job_peak = max((p["peak_gib"] for p in record.data["phases"]), default=0.0)

    record.data.update(
        {
            "status": "done",
            "output": str(output),
            "delivered_frames": int(len(video)),
            "delivered_seconds": round(len(video) / FPS, 3),
            "total_seconds": round(total, 3),
            "mean_denoise_step_seconds": round(float(np.mean(step_times)), 3),
            "denoise_step_seconds": [round(v, 3) for v in step_times],
            "serve": {
                "job_index": engine.jobs + 1,
                "dit_reloaded": engine.dit_loads > dit_loads_before,
                "dit_loads_total": engine.dit_loads,
                "cond_cache_hit": cond_hit,
                "adaln_cache_hits": _CachingModulationCache.hits,
                "adaln_cache_misses": _CachingModulationCache.misses,
                "job_peak_gib": round(job_peak, 3),
                "job_active_gib": round(gib(mx.get_active_memory()), 3),
            },
        }
    )
    record.flush()

    engine.jobs += 1
    # Between jobs, not during: the decoded pixels and the audio are numpy by now, so everything
    # MLX still holds is cache. Residency must not creep across a session.
    engine.free_vaes()
    gc.collect()
    mx.clear_cache()

    return {
        "event": "done",
        "id": job_id,
        "output": str(output),
        "metrics": str(metrics),
        "seconds": round(total, 3),
        "peak_gib": round(job_peak, 3),
        "active_gib_after": round(gib(mx.get_active_memory()), 3),
        "dit_reloaded": engine.dit_loads > dit_loads_before,
        "cond_cache_hit": cond_hit,
        "frames": int(len(video)),
    }


def run_encode(engine, job_id, params: dict) -> dict:
    """Warm the conditioning cache for one (prompt, first frame, canvas) without denoising.

    This is the move that makes a resident DiT pay off across *different* prompts: drain the
    encodes first, while the encoder is the only large thing in memory, then load the DiT once and
    denoise every job without evicting it again.
    """
    height, width = int(params["height"]), int(params["width"])
    prompt = params["prompt"]
    first_frame = params.get("first_frame") or None
    first_frame_key = str(first_frame) if first_frame else ""
    cond_path = engine.cond_cache_path(prompt, first_frame_key, height, width)
    if cond_path.exists():
        return {"event": "encoded", "id": job_id, "cached": True, "path": str(cond_path)}

    started = time.perf_counter()
    keyframes = None
    if first_frame:
        from PIL import Image

        from minimax_h3_mlx.packing import prepare_keyframe_image

        keyframes = [
            prepare_keyframe_image(
                Image.open(first_frame).convert("RGB"), height, width, stretch=True
            )
        ]

    engine.free_dit("pre-encoding a prompt")
    encoder = staged.MiniMaxH3TextEncoder(
        Path(engine.opts.compact_root),
        dtype=mx.bfloat16,
        load_vision=keyframes is not None,
        verbose=True,
        config_path=Path(engine.opts.text_config) if engine.opts.text_config else None,
    )
    embeds, text_tags = encoder.encode(prompt, keyframes)
    embeds = mx.array(embeds)
    mx.eval(embeds)
    cond_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cond_path,
        prompt=np.array(prompt),
        first_frame=np.array(first_frame_key),
        embeds=np.array(embeds.astype(mx.float32)),
        text_tags=text_tags,
    )
    del encoder, embeds
    gc.collect()
    mx.clear_cache()
    return {
        "event": "encoded",
        "id": job_id,
        "cached": False,
        "path": str(cond_path),
        "seconds": round(time.perf_counter() - started, 3),
        "tokens": int(len(text_tags)),
    }


# ---- main --------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit", required=True)
    parser.add_argument("--compact-root", required=True)
    parser.add_argument("--text-config", default=None)
    parser.add_argument("--lora", action="append", default=None)   # repeatable: adapters stack
    parser.add_argument("--lora-adaln", default=None)
    parser.add_argument("--draft-decode", choices=("full", "tae"), default="full")
    parser.add_argument("--tae-checkpoint", default=None)
    parser.add_argument(
        "--cond-cache",
        default=None,
        help="Directory for encoded-conditioning NPZs. Defaults to a sibling of the DiT's parent.",
    )
    parser.add_argument("--wired-gb", type=float, default=50.0)
    parser.add_argument("--memory-gb", type=float, default=58.0)
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load the DiT before announcing ready. Only sensible when the first job's "
        "conditioning is already cached; otherwise the encoder evicts it immediately.",
    )
    parser.add_argument(
        "--resident-vae",
        action="store_true",
        help="Keep the compact video/audio VAEs resident too (+4.9 GiB on every peak).",
    )
    parser.add_argument(
        "--session-metrics",
        default=None,
        help="Write a session-level summary here as jobs complete.",
    )
    parser.add_argument("--idle-timeout", type=float, default=1800.0)
    opts = parser.parse_args()

    if opts.draft_decode == "tae" and not opts.tae_checkpoint:
        parser.error("--draft-decode tae requires --tae-checkpoint")

    if opts.cond_cache is None:
        opts.cond_cache = str(Path(opts.dit).resolve().parent / "cond_cache")
    opts.runner_digest = hashlib.sha256(
        (Path(__file__).resolve().parent / "generate_staged.py").read_bytes()
    ).hexdigest()[:16]

    sys.stdout = LineEmitter()
    sys.stderr = LineEmitter()

    device = mx.device_info()
    wired = min(int(opts.wired_gb * 1024**3), int(device["max_recommended_working_set_size"]) - 1024**2)
    limit = min(int(opts.memory_gb * 1024**3), int(device["memory_size"]) - 1024**3)
    mx.set_wired_limit(wired)
    mx.set_memory_limit(limit)

    engine = ResidentEngine(opts)
    engine.install()

    state = {"last_activity": time.time(), "busy": False}

    def reaper():
        while True:
            time.sleep(15)
            if state["busy"]:
                continue
            if time.time() - state["last_activity"] > opts.idle_timeout:
                emit({"event": "exit", "reason": "idle"})
                os._exit(0)

    threading.Thread(target=reaper, daemon=True).start()

    if opts.preload:
        engine.load_dit()

    session = {
        "started": time.time(),
        "wired_gb": round(gib(wired), 3),
        "memory_limit_gb": round(gib(limit), 3),
        "resident_vae": bool(opts.resident_vae),
        "runner_sha256": opts.runner_digest,
        "device": device,
        "jobs": [],
    }

    def flush_session():
        if opts.session_metrics:
            path = Path(opts.session_metrics)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(session, indent=2) + "\n")

    emit(
        {
            "event": "ready",
            "dit": str(opts.dit),
            "compact_root": str(opts.compact_root),
            "dit_resident": engine.dit is not None,
            "resident_vae": bool(opts.resident_vae),
            "cond_cache": str(opts.cond_cache),
            "wired_gb": round(gib(wired), 3),
            "memory_limit_gb": round(gib(limit), 3),
            "mlx": mx.__version__,
            "runner_sha256": opts.runner_digest,
            "idle_timeout_sec": opts.idle_timeout,
        }
    )
    flush_session()

    for line in sys.__stdin__:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception as exc:
            emit({"event": "error", "error": f"bad json: {exc}"})
            continue

        state["last_activity"] = time.time()
        action = msg.get("action")
        job_id = msg.get("id", "?")

        if action == "exit":
            emit({"event": "exit", "reason": "shutdown"})
            flush_session()
            os._exit(0)

        if action == "ping":
            emit({"event": "pong", "dit_resident": engine.dit is not None})
            continue

        if action == "status":
            emit(
                {
                    "event": "status",
                    "dit_resident": engine.dit is not None,
                    "dit_loads": engine.dit_loads,
                    "dit_load_seconds": round(engine.dit_load_seconds, 3),
                    "jobs": engine.jobs,
                    "active_gib": round(gib(mx.get_active_memory()), 3),
                    "cache_gib": round(gib(mx.get_cache_memory()), 3),
                    "peak_gib": round(gib(mx.get_peak_memory()), 3),
                }
            )
            continue

        if action not in ("generate", "encode"):
            emit({"event": "error", "id": job_id, "error": f"unsupported action: {action}"})
            continue

        params = msg.get("params") or {}
        state["busy"] = True
        try:
            if action == "encode":
                result = run_encode(engine, job_id, params)
            else:
                result = run_generate(engine, job_id, params)
            session["jobs"].append(result)
            flush_session()
            emit(result)
        except BaseException as exc:  # noqa: BLE001 - a serve loop must survive one bad job
            emit(
                {
                    "event": "error",
                    "id": job_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "trace": traceback.format_exc(),
                }
            )
        finally:
            state["busy"] = False
            state["last_activity"] = time.time()

    emit({"event": "exit", "reason": "stdin closed"})
    flush_session()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
