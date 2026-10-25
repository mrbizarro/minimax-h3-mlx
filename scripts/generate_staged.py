"""Generate MiniMax-H3 locally with one large component resident at a time.

This runner targets a 64 GB Apple Silicon desktop. It consumes the quality-first pruned BF16 DiT,
ddalcu's native MLX Q8 text encoder, and ddalcu's compact VAE exports. Every phase is materialized,
timed, recorded, and explicitly released before the next model is loaded.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

# Since ml-explore/mlx#4248 the runtime no longer registers its own exit-time
# cleanup: without this, stream/compile-cache teardown runs during static
# destruction after the interpreter is gone and aborts with
# "PyThreadState_Get: the function must be called with the GIL held" —
# AFTER the MP4 was written, so the panel called a finished render a failure
# (Phosphene #76, diagnosed and verified by @PhantombrainM). Guarded so an
# older mlx without clear_streams keeps working.
import atexit
if hasattr(mx, "clear_streams"):
    atexit.register(mx.clear_streams)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.adaln import ModulationCache, drop_adaln_weights
from minimax_h3_mlx.config import PipelineConfig
from minimax_h3_mlx.live_preview import (
    ABORT_EXIT_CODE,
    DEFAULT_CONTEXT as LIVE_PREVIEW_DEFAULT_CONTEXT,
    LivePreviewAborted,
    LivePreviewMonitor,
)
from minimax_h3_mlx.load import (
    dit_precision,
    load_compact_audio_vae,
    load_compact_video_vae,
    load_dit,
)
from minimax_h3_mlx.media import (
    mux_audio_onto_video,
    probe_media,
    probe_video_frames,
    read_audio,
    read_video_frames,
    save_mp4,
)
from minimax_h3_mlx.conditioning import (
    encode_audio_rows,
    encode_video_rows,
    frozen_row_mask,
    gate_ranked_skip_blocks,
)
from minimax_h3_mlx.packing import (
    AUDIO_CHANNELS,
    FPS,
    KEYFRAME_NOISE_AUG,
    PIXEL_MEAN,
    PIXEL_STD,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    patchify_video_latents,
    unpack_audio_tokens,
    unpatchify_video_tokens,
    video_latent_num_frames,
)
from minimax_h3_mlx.pipeline import encode_keyframe_rows
from minimax_h3_mlx.render_guard import (
    RenderIntegrityError,
    check_decoded_video,
    nonfinite_error,
    nonfinite_flag,
)
from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler
from minimax_h3_mlx.stepcache import StepResidualCache
from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder
from minimax_h3_mlx.video_vae import resolved_decode_batch, resolved_decode_dtype


def parse_lora_path(spec) -> Path:
    from minimax_h3_mlx import lora as lora_mod

    return Path(lora_mod.parse_spec(str(spec))[0])


def gb(value: int) -> float:
    return value / 1024**3


def release(*objects) -> None:
    del objects
    gc.collect()
    mx.clear_cache()


class Recorder:
    def __init__(self, path: Path, settings: dict):
        self.path = path
        self.data = {"status": "running", "settings": settings, "phases": []}
        self.flush()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2) + "\n")

    @contextmanager
    def phase(self, name: str):
        mx.reset_peak_memory()
        started = time.perf_counter()
        print(f"\n== {name} ==", flush=True)
        try:
            yield
        finally:
            item = {
                "name": name,
                "seconds": round(time.perf_counter() - started, 3),
                "active_gib": round(gb(mx.get_active_memory()), 3),
                "cache_gib": round(gb(mx.get_cache_memory()), 3),
                "peak_gib": round(gb(mx.get_peak_memory()), 3),
            }
            self.data["phases"].append(item)
            self.flush()
            print(
                f"{name}: {item['seconds']:.1f}s, peak {item['peak_gib']:.2f} GiB, "
                f"active {item['active_gib']:.2f} GiB",
                flush=True,
            )


def parse_sigma_subset(spec: str | None) -> tuple[int, tuple[int, ...]] | None:
    """``"50:0,16,33,49"`` -> ``(50, (0, 16, 33, 49))``: keep those points of an N-point grid.

    This is how step-distilled adapters that were trained on a sub-sampled schedule publish their
    ladder (TaoMate-H3: ``DISTILLED_STATE_INDICES = (0, 16, 33, 49)`` of the shifted 50-point grid,
    3 forwards). The indices must start at 0, end at N-1 and increase strictly.
    """
    if not spec:
        return None
    grid, _, tail = str(spec).partition(":")
    n = int(grid)
    indices = tuple(int(v) for v in tail.split(",") if v.strip())
    if len(indices) < 2 or indices[0] != 0 or indices[-1] != n - 1 or list(indices) != sorted(set(indices)):
        raise ValueError(f"--sigma-subset {spec!r}: indices must be sorted, unique, and span 0..{n - 1}")
    return n, indices


def build_schedules(points: int, config: PipelineConfig, subset=None):
    video = MiniMaxH3Scheduler(shift=config.sigma_shift_video)
    audio = MiniMaxH3Scheduler(shift=config.sigma_shift_audio)
    if subset is None:
        video.set_timesteps(points)
        audio.set_timesteps(points)
        return video, audio
    grid, indices = subset
    for sched in (video, audio):
        sched.set_timesteps(grid)
        full = sched.sigmas.tolist()
        if len(full) != grid:
            raise ValueError(f"the {grid}-point grid collapsed to {len(full)} points at this shift")
        sched.set_timesteps(sigmas=[full[i] for i in indices])
    return video, audio


# Timestep a held picture's rows run at (video-to-audio). 1.0 = clean target; the research knob
# H3_FROZEN_VIDEO_T lets an A/B try the keyframe level (0.999) instead.
import os as _os

FROZEN_VIDEO_TIMESTEP = float(_os.environ.get("H3_FROZEN_VIDEO_T", "1.0"))


def row_timestep_plan(layout, video_timesteps, audio_timesteps, frozen_video=None, frozen_audio=None):
    per_step = []
    for t, at in zip(video_timesteps.tolist(), audio_timesteps.tolist()):
        # Keyframe rows are pinned at their noise-augmentation level for the whole run. With no
        # keyframe the layout has no conditioning rows, so the extra level never appears in the
        # table and the schedule is unchanged. Frozen target rows (a supplied soundtrack or
        # picture) sit at t = 1, clean, for every step.
        distinct, inverse = build_row_timesteps(
            layout, float(t), float(at), max(float(t), KEYFRAME_NOISE_AUG), 1.0,
            frozen_video, frozen_audio, FROZEN_VIDEO_TIMESTEP,
        )
        per_step.append((np.array(distinct), np.array(inverse)))
    table = sorted({float(value) for distinct, _ in per_step for value in distinct})
    lookup = {value: index for index, value in enumerate(table)}
    plan = []
    for distinct, inverse in per_step:
        remap = np.array([lookup[float(value)] for value in distinct], dtype=np.int32)
        plan.append(mx.array(remap[inverse].astype(np.int32)))
    return mx.array(np.array(table, dtype=np.float32)), plan


def decode_video(model, rows, frames: int, latent_frames: int, latent_h: int, latent_w: int, patch):
    cfg = model.config
    latents = unpatchify_video_tokens(
        rows, latent_frames, latent_h, latent_w, cfg.latent_channels, patch
    )
    mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1, 1, 1)
    std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1, 1, 1)
    decoded = np.array(model.decode((latents * std + mean).astype(mx.float32)))
    pixel_mean = np.array(PIXEL_MEAN, np.float32).reshape(1, 3, 1, 1, 1)
    pixel_std = np.array(PIXEL_STD, np.float32).reshape(1, 3, 1, 1, 1)
    decoded = np.clip(decoded * pixel_std + pixel_mean, 0.0, 1.0)
    decoded = decoded[0].transpose(1, 2, 3, 0)[:frames]
    return (decoded * 255.0 + 0.5).astype(np.uint8)


def decode_video_tae(
    model, rows, frames: int, latent_frames: int, latent_h: int, latent_w: int, patch
):
    """Decode normalized transformer rows with the opt-in tiny H3 preview decoder."""
    latents = unpatchify_video_tokens(
        rows, latent_frames, latent_h, latent_w, model.latent_channels, patch
    )
    decoded = np.array(model.decode(latents.astype(mx.float32), frames))
    decoded = decoded[0].transpose(1, 2, 3, 0)[:frames]
    return (np.clip(decoded, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def decode_audio(model, rows, audio_latents: int, frames: int):
    cfg = model.config
    latents = unpack_audio_tokens(rows, audio_latents)
    mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1)
    std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1)
    waveform = np.array(model.decode((latents * std + mean).astype(mx.float32)))[:, 0, :]
    samples = round(frames / FPS * cfg.sampling_rate)
    return waveform[:, :samples].astype(np.float32)


def draft_video_grid(delivery_frames: int, draft_fps: int) -> int:
    """Choose the next-lower H3 grid without changing its joint audio/video tensor shape."""
    target_unique = delivery_frames * draft_fps // FPS
    return max(22, (target_unique - 5) // 17 * 17 + 5)


def distribute_draft_frames(video: np.ndarray, delivery_frames: int) -> np.ndarray:
    """Repeat native source frames across an exact-length 24 fps delivery timeline."""
    indices = np.rint(np.linspace(0, len(video) - 1, delivery_frames)).astype(np.int64)
    return video[indices]


def resize_draft_video(video: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize unique preview frames before temporal duplication."""
    from PIL import Image

    return np.stack(
        [
            np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS))
            for frame in video
        ]
    )


def spread_joint_draft_clock(layout, text_rows: int, scale: float) -> None:
    """Spread target audio and video positions together without adding packed rows."""
    positions = np.asarray(layout.position_ids, dtype=np.float32)
    target_video = np.asarray(layout.video_indices.tolist(), dtype=np.int64)[
        layout.num_condition_video_rows :
    ]
    target_audio = np.asarray(layout.audio_indices.tolist(), dtype=np.int64)[
        layout.num_condition_audio_rows :
    ]
    targets = np.concatenate([target_audio, target_video])
    origin = float(text_rows)
    positions[targets, 0] = origin + (positions[targets, 0] - origin) * scale
    layout.position_ids = mx.array(positions)


AUDIO_RATE = 32000  # the audio VAE's rate; every waveform this runner handles is at it

# The carried-context lengths a window can take: whole 17-frame VAE clips, so the history ends
# exactly where the next target starts on the rotary clock.
CONTEXT_FRAME_STEP = 17


def fit_audio(wave: np.ndarray, frames: int) -> np.ndarray:
    """Pad with silence or cut a ``(2, samples)`` waveform to exactly ``frames`` of 24 fps picture."""
    want = round(frames / FPS * AUDIO_RATE)
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 2 or wave.shape[0] != 2:
        wave = np.zeros((2, 0), dtype=np.float32)
    if wave.shape[1] < want:
        wave = np.pad(wave, ((0, 0), (0, want - wave.shape[1])))
    return np.ascontiguousarray(wave[:, :want])


def frame_samples(frame: int) -> int:
    """The audio sample at which 24 fps picture frame ``frame`` starts."""
    return round(frame / FPS * AUDIO_RATE)


def drive_slice(wave: np.ndarray, start_frame: int, num_frames: int) -> np.ndarray:
    """``num_frames`` of picture's worth of ``wave`` starting at ``start_frame``, silence-padded."""
    start = frame_samples(max(0, start_frame))
    want = frame_samples(num_frames)
    piece = np.asarray(wave[:, start : start + want], dtype=np.float32)
    if piece.shape[1] < want:
        piece = np.pad(piece, ((0, 0), (0, want - piece.shape[1])))
    return np.ascontiguousarray(piece)


# Longest --extend-from / --video-to-audio source, in seconds: a minute at 1024x576 is ~2.5 GB of
# decoded frames, which stays resident for the whole render.
MEDIA_SOURCE_MAX_SECONDS = 60.0


def media_mode_errors(args, parser) -> str:
    """Validate the media-conditioned flags together; return the run's mode name."""
    if args.chain_context_frames < 0 or args.chain_context_frames % CONTEXT_FRAME_STEP:
        parser.error(f"--chain-context-frames must be a multiple of {CONTEXT_FRAME_STEP} (0 = off)")
    keyframes = list(args.keyframe or [])
    media = [
        name for name, value in (
            ("--extend-from", args.extend_from),
            ("--video-to-audio", args.video_to_audio),
            ("--audio-drive", args.audio_drive),
        ) if value is not None
    ]
    for flag, value in (("--extend-from", args.extend_from), ("--video-to-audio", args.video_to_audio),
                        ("--audio-drive", args.audio_drive), ("--audio-mux", args.audio_mux),
                        ("--last-frame", args.last_frame)):
        if value is not None and not Path(value).is_file():
            parser.error(f"{flag} does not exist: {value}")
    if args.extend_frames is not None and args.extend_from is None:
        parser.error("--extend-frames only applies with --extend-from")
    for flag, value in (("--extend-from", args.extend_from), ("--video-to-audio", args.video_to_audio)):
        # The whole source is decoded onto the canvas and held through the render (it is stitched
        # in front of the new part, or is the held picture), so its length is bounded up front.
        if value is not None:
            try:
                seconds = float(probe_media(value).get("duration") or 0.0)
            except RuntimeError as exc:
                parser.error(f"{flag}: {exc}")
            if seconds > MEDIA_SOURCE_MAX_SECONDS:
                parser.error(f"{flag} takes clips up to {MEDIA_SOURCE_MAX_SECONDS:.0f}s; "
                             f"{value} is {seconds:.1f}s — cut it shorter first")
    if args.audio_mux is not None and args.audio_drive is None:
        parser.error("--audio-mux only applies with --audio-drive")
    if args.video_to_audio is not None and (
        args.first_frame or args.last_frame or keyframes or args.extend_from or args.audio_drive
    ):
        parser.error("--video-to-audio keeps the picture it is given; it takes no keyframes, "
                     "--extend-from or --audio-drive")
    if args.extend_from is not None and (args.first_frame is not None or keyframes):
        parser.error("--extend-from starts from the source's own last frame; pass --last-frame "
                     "for an end target, not --first-frame/--keyframe")
    if args.extend_from is not None and args.audio_drive is not None:
        parser.error("--extend-from and --audio-drive do not compose (yet); extend, then lip-sync")
    extra = media or args.last_frame is not None or keyframes or args.chain_context_frames
    if extra:
        for flag, value in (("--draft-fps", args.draft_fps), ("--playback-fps", args.playback_fps),
                            ("--draft-seconds", args.draft_seconds),
                            ("--draft-cache-dir", args.draft_cache_dir),
                            ("--save-stage-a", args.save_stage_a)):
            if value is not None:
                parser.error(f"{flag} does not compose with keyframes, carried context or media "
                             "conditioning")
    if args.extend_from is not None and not args.chain_context_frames:
        args.chain_context_frames = CONTEXT_FRAME_STEP
    if args.video_to_audio is not None and not args.chain_context_frames:
        args.chain_context_frames = CONTEXT_FRAME_STEP
    if args.chain_context_frames >= align_num_frames(args.frames):
        parser.error("--chain-context-frames must be shorter than a window")
    for spec in keyframes:
        path, _, when = str(spec).rpartition("@")
        if not path or not when:
            parser.error(f"--keyframe wants PATH@WHEN, got {spec!r}")
        if not Path(path).is_file():
            parser.error(f"--keyframe image does not exist: {path}")
    if args.video_to_audio is not None:
        return "video_to_audio"
    if args.audio_drive is not None:
        return "audio_drive"
    if args.extend_from is not None:
        return "extend"
    if args.last_frame is not None or keyframes:
        return "keyframes"
    return "text" if args.first_frame is None else "first_frame"


def resolve_keyframe_time(when: str, delivered: int) -> int:
    """``'48'`` -> frame 48; ``'2.5s'`` -> the frame at 2.5 s; ``'-1'`` -> the last frame."""
    when = str(when).strip().lower()
    if when.endswith("s"):
        frame = int(round(float(when[:-1]) * FPS))
    else:
        frame = int(when)
    if frame < 0:
        frame += delivered
    if not 0 <= frame < delivered:
        raise ValueError(f"keyframe time {when!r} is outside the clip's 0..{delivered - 1} frames")
    return frame


def window_of_frame(frame: int, window_frames: int) -> tuple[int, int]:
    """Global frame -> (window, frame within that window) for a chain of ``window_frames`` windows.

    Windows overlap by one frame (each opens on its predecessor's last frame), and that shared
    frame belongs to the EARLIER window, where it is a real last frame — the later window would
    only see it as the carried hand-over it already conditions on.
    """
    if frame <= 0:
        return 0, 0
    step = window_frames - 1
    window = (frame - 1) // step
    return window, frame - window * step


def plan_keyframes(args, window_frames: int, chain: int, delivered: int) -> list[list[tuple[Path, int]]]:
    """Route --first-frame / --keyframe / --last-frame to the windows that own them.

    Returns one list per window of ``(image path, frame within the window)``, sorted by time.
    Frame 0 of a window after the first is its carried hand-over, so a keyframe can only land
    there in window 1.
    """
    plan: list[list[tuple[Path, int]]] = [[] for _ in range(chain)]
    entries: list[tuple[int, Path]] = []
    if args.first_frame is not None:
        entries.append((0, Path(args.first_frame)))
    for spec in args.keyframe or []:
        path, _, when = str(spec).rpartition("@")
        entries.append((resolve_keyframe_time(when, delivered), Path(path)))
    if args.last_frame is not None:
        entries.append((delivered - 1, Path(args.last_frame)))
    seen: dict[int, Path] = {}
    for frame, path in entries:
        if frame in seen and seen[frame] != path:
            raise ValueError(f"two keyframes land on frame {frame} ({seen[frame].name}, {path.name})")
        seen[frame] = path
    for frame, path in sorted(seen.items()):
        window, local = window_of_frame(frame, window_frames)
        if window >= chain:
            raise ValueError(f"keyframe at frame {frame} is past the last window")
        plan[window].append((path, local))
    return plan


def anchor_name(local: int, window_frames: int) -> str:
    if local == 0:
        return "first"
    if local == window_frames - 1:
        return "last"
    return f"frame:{local}"


_ALIGNMENT_PREFIXES = ("for the target video", "how the reference pictures")


def alignment_line(anchors: list[str], window_frames: int, prompt: str) -> str | None:
    """MiniMax's own keyframe-alignment sentence for this window's pictures, or None.

    The official prompt guide opens a first+last (FL2VA) request with "How the reference pictures
    align with the target video — Picture 1 ... aligns with the 0.00-second mark ...; Picture 2
    ... aligns with the S.SS-second mark ..." and a last-frame-only one with the same sentence
    for ``<Picture 1>``. A plain first-frame window gets nothing, exactly as before.
    """
    if not anchors or all(anchor == "first" for anchor in anchors):
        return None
    if prompt.lstrip().lower().startswith(_ALIGNMENT_PREFIXES):
        return None
    import re

    shots = [int(n) for n in re.findall(r"\[Shot (\d+)\]", prompt)]
    final_shot = max(shots) if shots else 1

    def seconds(anchor: str) -> float:
        if anchor == "first":
            return 0.0
        if anchor == "last":
            return window_frames / FPS
        return int(anchor.split(":", 1)[1]) / FPS

    if anchors == ["last"]:
        return (
            "How the reference pictures align with the target video — <Picture 1> (from "
            f"[Shot {final_shot}]) aligns with the {seconds('last'):.2f}-second mark of the target video."
        )
    parts = [
        f"Picture {index} (from Shot {1 if anchor == 'first' else final_shot}) aligns with the "
        f"{seconds(anchor):.2f}-second mark of the target video"
        for index, anchor in enumerate(anchors, 1)
    ]
    return "How the reference pictures align with the target video — " + "; ".join(parts) + "."


CHAIN_PROMPT_SEPARATOR = " ||| "

# The shipped preview lengths all keep H3's native 24 fps motion/audio clock. Three and five
# seconds use one legal 17n+5 window; ten and fifteen seconds extend the approved five-second
# window through the ordinary first-frame chain and trim only the final tail. Nothing here is
# consulted unless --draft-seconds is explicit.
DRAFT_DURATION_PRESETS = {
    3: (73, 1, None),
    5: (124, 1, None),
    10: (124, 2, 240),
    15: (124, 3, 360),
}


def draft_duration_plan(seconds: int) -> tuple[int, int, int | None]:
    """Return (native frames per window, windows, delivered frames) for a draft preset."""
    try:
        return DRAFT_DURATION_PRESETS[seconds]
    except KeyError as exc:
        choices = ", ".join(str(value) for value in DRAFT_DURATION_PRESETS)
        raise ValueError(f"draft length must be one of {choices} seconds") from exc


def parse_chain_prompts(spec: str, windows: int) -> list[str]:
    """Turn one ``--chain-prompts`` value into exactly one prompt per window.

    A chain is a shot list, not one shot repeated: window *i* is a different moment of the same
    scene, so it wants its own text conditioning. ``spec`` is either a ``' ||| '``-separated string
    or the path to a ``.json`` file holding a list of strings — the same thing, spelled for a shell
    or for a file when the prompts are long enough that quoting them stops being reasonable.

    The count has to match the window count exactly. Silently recycling or truncating prompts is
    how a shot list turns into a clip whose dialogue lands in the wrong window.
    """
    path = Path(spec)
    if path.suffix.lower() == ".json":
        if not path.exists():
            raise ValueError(f"--chain-prompts names a .json file that does not exist: {path}")
        try:
            loaded = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"--chain-prompts file {path} is not valid JSON: {exc}") from exc
        if not isinstance(loaded, list) or not all(isinstance(item, str) for item in loaded):
            raise ValueError(
                f"--chain-prompts file {path} must hold a JSON list of strings, one per window"
            )
        prompts = [item.strip() for item in loaded]
        source = f"{path} holds"
    else:
        prompts = [part.strip() for part in spec.split(CHAIN_PROMPT_SEPARATOR.strip())]
        source = "--chain-prompts holds"

    if any(not prompt for prompt in prompts):
        empty = [index + 1 for index, prompt in enumerate(prompts) if not prompt]
        raise ValueError(
            f"--chain-prompts has an empty prompt at position(s) {empty}; every window needs one. "
            f"Separate prompts with '{CHAIN_PROMPT_SEPARATOR}'."
        )
    if len(prompts) != windows:
        raise ValueError(
            f"{source} {len(prompts)} prompt(s) but --chain-windows is {windows}. "
            f"Give exactly one prompt per window, separated by '{CHAIN_PROMPT_SEPARATOR}' "
            "(or a JSON list of that many strings)."
        )
    return prompts


def resolve_skip_blocks(spec: str, cache, num_layers: int) -> list[int]:
    """``--skip-blocks``: ``"auto:N"`` (the N smallest-gate blocks) or an explicit ``"3,17,22"``.

    Blocks 0, 1 and the last are never skipped, whichever spelling asks for them.
    """
    spec = str(spec).strip()
    protected = {0, 1, num_layers - 1}
    if spec.startswith("auto:"):
        return gate_ranked_skip_blocks(cache, num_layers, int(spec.split(":", 1)[1]))
    blocks = sorted({int(part) for part in spec.split(",") if part.strip()})
    bad = [b for b in blocks if b in protected or not 0 <= b < num_layers]
    if bad:
        raise ValueError(f"--skip-blocks cannot skip {bad} (0, 1 and {num_layers - 1} are kept)")
    return blocks


ADALN_SKIP_PREFIX = "adaLN projection"


def lora_accounting(data: dict, label: str, adaln_report: dict | None) -> dict:
    """Every LoRA module pair, accounted for: wrapped, absorbed into the AdaLN cache, or lost.

    ``apply_lora`` reports the adaLN pairs as skipped (the pruned DiT cannot wrap them);
    ``absorb_adaln_lora`` then folds them in when ``--lora-adaln`` is given. Neither report alone
    says whether anything was dropped, which is how Turbo v4 silently lost all 51 of its adaLN
    pairs. ``unaccounted`` is the number that must be zero.
    """
    summaries = data.get(f"{label}lora_stack") or [data.get(f"{label}lora") or {}]
    absorbed_by_path: dict[str, int] = {}
    for report in (adaln_report or {}).get("stack") or ([adaln_report] if adaln_report else []):
        count = int(report.get("blocks") or 0) + (1 if report.get("final") else 0)
        key = str(report.get("path", ""))
        absorbed_by_path[key] = absorbed_by_path.get(key, 0) + count
    adapters = []
    for summary in summaries:
        by_reason = summary.get("skipped_by_reason") or {}
        adaln_skipped = sum(n for r, n in by_reason.items() if r.startswith(ADALN_SKIP_PREFIX))
        other_skipped = sum(n for r, n in by_reason.items() if not r.startswith(ADALN_SKIP_PREFIX))
        path = str(summary.get("path", ""))
        absorbed = absorbed_by_path.get(path)
        if absorbed is None and len(summaries) == 1 and absorbed_by_path:
            absorbed = sum(absorbed_by_path.values())
        absorbed = min(int(absorbed or 0), adaln_skipped)
        adapters.append({
            "path": path,
            "applied": summary.get("applied"),
            "adaln_pairs": adaln_skipped,
            "adaln_absorbed": absorbed,
            "unaccounted": (adaln_skipped - absorbed) + other_skipped,
        })
    return {
        "adapters": adapters,
        "applied": sum(int(a["applied"] or 0) for a in adapters),
        "adaln_pairs": sum(a["adaln_pairs"] for a in adapters),
        "adaln_absorbed": sum(a["adaln_absorbed"] for a in adapters),
        "unaccounted": sum(a["unaccounted"] for a in adapters),
    }


def seam_colour_match(prev_last: np.ndarray, video: np.ndarray, frames: int) -> np.ndarray:
    """Ease a new window's colour into its predecessor's over ``frames`` frames.

    Every window is decoded on its own, and its re-render of the hand-over frame (``video[0]``,
    dropped at the join) lands a percent or two off the frame it re-renders — a small, visible
    exposure/tint step at the seam. The per-channel mean/std map that takes ``video[0]`` onto
    ``prev_last`` is applied in full at the seam and faded to nothing over ``frames`` frames, so
    the window drifts back to its own grade too slowly to see. Picture content is untouched.
    """
    if frames <= 0 or len(video) < 2:
        return video
    src = video[0].reshape(-1, 3).astype(np.float32)
    ref = prev_last.reshape(-1, 3).astype(np.float32)
    gain = (ref.std(0) + 1e-3) / (src.std(0) + 1e-3)
    gain = np.clip(gain, 0.8, 1.25)
    offset = ref.mean(0) - src.mean(0) * gain
    out = video.astype(np.float32)
    span = min(frames, len(video) - 1)
    for k in range(0, span + 1):
        w = 1.0 - k / float(span + 1)
        out[k] = out[k] * (1.0 + (gain - 1.0) * w) + offset * w
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def stitch_windows(segments, sample_rate: int, crossfade_seconds: float,
                   colour_match_frames: int = 0):
    """Butt-join chained windows in pixel space and cross-fade their audio at every seam.

    Each window after the first opens on a re-render of its predecessor's last frame, so that
    duplicate frame is dropped from the head — the reference behaviour. The audio of the dropped
    frame is *not* thrown away: it is the only material the chain owns that covers the same instant
    of scene time as the outgoing window's tail, which makes a true overlap cross-fade possible
    while keeping every later sample exactly where the video puts it.

    That bounds the fade at one frame (41.7 ms at 24 fps). Asking for more would mean either
    consuming audio the video still needs — WanGP's blind ``np.concatenate`` has the opposite
    failure, an untreated step discontinuity at every seam — or letting the whole tail of the clip
    drift ahead of the picture by the fade length per seam, which is worse than the click it fixes.
    """
    frame_samples = int(round(sample_rate / FPS))
    fade = min(int(round(crossfade_seconds * sample_rate)), frame_samples)

    video_parts = [segments[0][0]]
    audio_parts = [segments[0][1]]
    seams = []
    for video, audio in segments[1:]:
        if colour_match_frames:
            video = seam_colour_match(video_parts[-1][-1], video, colour_match_frames)
        video_parts.append(video[1:])
        tail = audio_parts[-1]
        span = min(fade, tail.shape[1], frame_samples, audio.shape[1])
        joined_at = int(sum(part.shape[1] for part in audio_parts))
        seam = {"sample": joined_at, "fade_samples": int(span)}
        if span > 0:
            # Equal power: two independently generated windows are uncorrelated across the seam, so
            # a linear pair would dip ~3 dB in the middle of the fade.
            ramp = (np.arange(span, dtype=np.float32) + 0.5) / span
            blended = tail[:, -span:] * np.cos(ramp * np.pi / 2) + (
                audio[:, frame_samples - span : frame_samples] * np.sin(ramp * np.pi / 2)
            )
            audio_parts[-1] = np.concatenate([tail[:, :-span], blended], axis=1)
        audio_parts.append(audio[:, frame_samples:])
        seams.append(seam)

    return (
        np.concatenate(video_parts, axis=0),
        np.concatenate(audio_parts, axis=1).astype(np.float32),
        seams,
    )


@dataclass
class WindowConditioning:
    """Everything a window is conditioned on beyond its prompt.

    ``keyframes``      (PIL image, anchor) pairs in packed order — ``"first"``, ``"last"`` or
                       ``"frame:<i>"`` (pixel frame i of this window). The images are also shown
                       to the text encoder's vision tower as ``<Picture n>``, in this order.
    ``history_frames`` (17*m, H, W, 3) uint8 — the carried tail of the clip this window
                       continues, ending on the frame BEFORE the window's first frame.
    ``history_audio``  (2, samples) at 32 kHz covering the history frames AND the window's first
                       frame (the boundary), so the carried sound ends where the carried picture
                       hands over.
    ``drive_audio``    (2, samples) — a soundtrack the window must follow: held clean in the
                       target audio rows (lip-sync, a song). Silence past its end.
    ``frozen_frames``  (F, H, W, 3) uint8 — a picture the window must keep: held clean in the
                       target video rows (video-to-audio). Only the sound is generated.
    """

    keyframes: list = field(default_factory=list)
    history_frames: np.ndarray | None = None
    history_audio: np.ndarray | None = None
    drive_audio: np.ndarray | None = None
    frozen_frames: np.ndarray | None = None
    references: list = field(default_factory=list)

    @property
    def decode_video(self) -> bool:
        return self.frozen_frames is None

    def summary(self) -> dict:
        return {
            "anchors": [anchor for _, anchor in self.keyframes],
            "history_frames": 0 if self.history_frames is None else int(len(self.history_frames)),
            "history_audio_seconds": (
                0.0 if self.history_audio is None
                else round(self.history_audio.shape[1] / 32000.0, 4)
            ),
            "drive_audio_seconds": (
                None if self.drive_audio is None else round(self.drive_audio.shape[1] / 32000.0, 4)
            ),
            "references": len(self.references),
            "frozen_video_frames": (
                None if self.frozen_frames is None else int(len(self.frozen_frames))
            ),
        }


def shift_reference_labels(prompt: str, base_keyframes: int, keyframes: int) -> str:
    """Move "<Picture n>" labels that name references (n > base_keyframes) to this window's
    numbering, where the references come after `keyframes` keyframes."""
    import re

    delta = int(keyframes) - int(base_keyframes)
    if not delta:
        return prompt

    def move(match):
        n = int(match.group(1))
        return f"<Picture {n + delta}>" if n > base_keyframes else match.group(0)

    return re.sub(r"<Picture (\d+)>", move, prompt)


def revoice_wanted(args, cond, lora_stack) -> bool:
    """Re-voice runs on a window whose sound the model makes with an adapter switched on.

    Skipped where it has nothing to do: no adapter to switch off, a held soundtrack (lip-sync),
    or a held picture (Add sound already makes the sound with no speed adapter).
    """
    if int(getattr(args, "revoice_steps", 0) or 0) <= 0:
        return False
    if not lora_stack or getattr(args, "lora_already_applied", False):
        return False
    return cond.drive_audio is None and cond.frozen_frames is None


def write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    import wave

    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(int(audio.shape[0]))
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(pcm.T.copy().tobytes())


def run_revoice(args, record, label, dit, rv, layout, video_rows, audio_rows, embeds, seed,
                n_cond_a, lora_stack):
    """Re-make this window's sound with the speed adapter off; the picture rows never move.

    Returns the new audio rows (conditioning rows first, untouched).
    """
    mod = rv["mod"]
    kv = getattr(args, "revoice_kv", "int8")
    started = time.perf_counter()
    with record.phase(f"{label}revoice"):
        switched = mod.set_adapter_scales(dit, rv["drop"], 0.0)
        print(
            f"re-voice: {rv['forwards']} audio steps with {', '.join(p.name for p in rv['drop'])} "
            f"off ({switched['entries']} adapter entries), KV {kv}",
            flush=True,
        )
        target_shape = (audio_rows.shape[0] - n_cond_a, audio_rows.shape[1])
        # Its own noise: the adapter's audio is not a starting point, the base model starts over.
        key = mx.random.key(int(seed) * 7919 + 104729)
        noise = mx.random.normal(target_shape, key=key).astype(mx.float32)
        strength = float(rv.get("strength", 1.0))
        if strength < 1.0:
            # x_t = t * x0 + (1 - t) * noise at sigma = strength (t = 1 - strength).
            noise = (1.0 - strength) * audio_rows[n_cond_a:].astype(mx.float32) + strength * noise
        cond_audio = audio_rows[:n_cond_a] if n_cond_a else None
        rows = mx.concatenate([cond_audio, noise]) if n_cond_a else noise
        rotary = None
        store = None
        step_seconds = []
        for index, timestep in enumerate(rv["audio_sched"].timesteps.tolist()):
            t0 = time.perf_counter()
            forward_args = (
                video_rows[None].astype(mx.bfloat16),
                rows[None].astype(mx.bfloat16),
                embeds.astype(mx.bfloat16),
                rv["table"],
                rv["plan"][index],
                layout.token_tags,
                layout.position_ids,
                layout.video_indices,
                layout.audio_indices,
                layout.text_indices,
            )
            if kv == "off":
                _, audio_out = dit(*forward_args, modulation_cache=rv["cache"])
                pred = audio_out[0].astype(mx.float32)
            elif store is None:
                audio_out, store = mod.store_forward(dit, forward_args, rv["cache"], rv["rows"], kv)
                pred = audio_out[0].astype(mx.float32)
                rotary = dit.rope(layout.position_ids)
            else:
                pred = mod.audio_forward(
                    dit, rows, rv["table"], rv["plan"][index], layout, rv["rows"], store,
                    rv["cache"], rotary,
                ).astype(mx.float32)
            stepped = rv["audio_sched"].step(pred[n_cond_a:], float(timestep), rows[n_cond_a:])
            rows = mx.concatenate([cond_audio, stepped]) if n_cond_a else stepped
            broken = nonfinite_flag(video_rows, rows)
            mx.eval(rows, broken)
            if broken.item():
                raise nonfinite_error(f"after re-voice step {index + 1}/{rv['forwards']}", lora_stack)
            step_seconds.append(time.perf_counter() - t0)
            print(f"re-voice step {index + 1}/{rv['forwards']}: {step_seconds[-1]:.2f}s "
                  f"(active {gb(mx.get_active_memory()):.1f} GiB)", flush=True)
        mod.set_adapter_scales(dit, rv["drop"], None)
        record.data[f"{label}revoice"] = {
            "steps": int(rv["forwards"]) + 1,
            "forwards": int(rv["forwards"]),
            "kv": kv,
            "strength": strength,
            "kv_gib": round(gb(store.nbytes), 3) if store is not None else 0.0,
            "dynamic_rows": int(rv["rows"].dynamic.shape[0]),
            "cached_rows": int(rv["rows"].cached.shape[0]),
            "dropped": [str(p) for p in rv["drop"]],
            "adapter_entries_off": int(switched["entries"]),
            "step_seconds": [round(v, 3) for v in step_seconds],
            "seconds": round(time.perf_counter() - started, 3),
        }
        record.flush()
        del store
    return rows


def render_window(
    args,
    record,
    *,
    label: str,
    prompt: str,
    frames: int,
    seed: int,
    keyframe,
    first_frame_key: str,
    prompt_cache: Path | None,
    live_preview: LivePreviewMonitor | None = None,
    window_index: int = 1,
    cond: WindowConditioning | None = None,
):
    """Render one window end to end and hand back decoded pixels, audio and the sample rate.

    Every large model is loaded, used and released inside this call, so a chain of windows runs in
    one process without the previous window's DiT still being resident. ``label`` prefixes the
    phase names so a chained metrics file stays readable. ``prompt`` is this window's own text —
    a chain that scripts a line in one window must not encode that line in the others.
    """
    if cond is None:
        cond = WindowConditioning(keyframes=[(keyframe, "first")] if keyframe is not None else [])
    keyframes = [image for image, _ in cond.keyframes] or None
    keyframe_anchors = tuple(anchor for _, anchor in cond.keyframes)
    # The vision tower sees the keyframes first (their "<Picture n>" numbering is what the
    # alignment line names), then the reference images, labelled after them.
    vision_images = (keyframes or []) + list(cond.references)
    if keyframe is None and keyframes:
        # The draft cache keys on "the" first frame; it is only ever used with a plain first-frame
        # window (media_mode_errors refuses it with anything richer).
        keyframe = keyframes[0]
    record.data[f"{label}conditioning"] = cond.summary()
    draft_cache = None
    text_request_key = None
    automatic_text = None
    draft_cache_dir = getattr(args, "draft_cache_dir", None)
    if draft_cache_dir is not None:
        from minimax_h3_mlx.draft_cache import DraftCache

        draft_cache = DraftCache(draft_cache_dir, getattr(args, "draft_cache_limit", 50))
        text_request_key = draft_cache.text_request_key(
            prompt=prompt,
            seed=seed,
            width=args.width,
            height=args.height,
            first_frame=keyframe,
            compact_root=args.compact_root,
            text_config=args.text_config,
        )
        if prompt_cache is None:
            automatic_text = draft_cache.load_text(text_request_key)

    if prompt_cache is not None and prompt_cache.exists():
        with record.phase(f"{label}text_cache_load"):
            cached = np.load(prompt_cache, allow_pickle=False)
            cached_prompt = str(cached["prompt"].item())
            # Keyed on this window's prompt, not the run's: a per-window chain has no single
            # prompt to cache against.
            if cached_prompt != prompt:
                raise ValueError("The prompt cache belongs to a different prompt.")
            cached_frame = str(cached["first_frame"].item()) if "first_frame" in cached else ""
            if cached_frame != first_frame_key:
                raise ValueError("The prompt cache was built for a different first frame.")
            embeds = mx.array(cached["embeds"]).astype(mx.bfloat16)
            text_tags = cached["text_tags"].astype(np.int64)
            mx.eval(embeds)
    elif automatic_text is not None:
        with record.phase(f"{label}text_draft_cache_load"):
            cached_embeds, text_tags, cached_digest = automatic_text
            embeds = mx.array(cached_embeds).astype(mx.bfloat16)
            mx.eval(embeds)
            record.data[f"{label}text_cache_hit"] = True
    else:
        with record.phase(f"{label}text_encode_q8"):
            # A keyframe request runs through the vision tower: H3 conditions on the still
            # twice, once as VL tokens in the text stream and once as VAE conditioning rows.
            encoder = MiniMaxH3TextEncoder(
                args.compact_root,
                dtype=mx.bfloat16,
                load_vision=bool(vision_images),
                verbose=True,
                config_path=args.text_config,
            )
            embeds, text_tags = encoder.encode(prompt, vision_images or None)
            embeds = mx.array(embeds)
            mx.eval(embeds)
            if prompt_cache is not None:
                prompt_cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    prompt_cache,
                    prompt=np.array(prompt),
                    first_frame=np.array(first_frame_key),
                    embeds=np.array(embeds.astype(mx.float32)),
                    text_tags=text_tags,
                )
        del encoder
    # A digest of the actual conditioning tensor, so "each window encoded its own prompt" is
    # something a metrics file proves rather than something the runner claims.
    embeds_host = np.array(embeds.astype(mx.float32))
    full_digest = hashlib.sha256(embeds_host.tobytes()).hexdigest()
    digest = full_digest[:16]
    if automatic_text is not None and cached_digest != full_digest:
        raise ValueError(
            f"draft text cache returned {cached_digest}, but the loaded embedding hashes to "
            f"{full_digest}"
        )
    if draft_cache is not None and automatic_text is None and prompt_cache is None:
        with record.phase(f"{label}text_draft_cache_store"):
            draft_cache.store_text(text_request_key, embeds_host, text_tags, full_digest)
            record.data[f"{label}text_cache_hit"] = False
    record.data[f"{label}prompt"] = prompt
    record.data[f"{label}prompt_tokens"] = int(len(text_tags))
    record.data[f"{label}text_embed_sha256"] = digest
    record.flush()
    print(
        f"text conditioning: {len(text_tags)} tokens, embeds {tuple(embeds.shape)} sha {digest}\n"
        f"  prompt: {prompt}",
        flush=True,
    )
    release()

    config = PipelineConfig()
    latent_frames = video_latent_num_frames(frames)
    latent_h, latent_w = args.height // 16, args.width // 16
    audio_latents = audio_latent_num_frames(frames)
    patch = (1, 2, 2)

    # The keyframe is encoded while only the 5 GB video VAE is resident, before the DiT lands.
    condition_rows = None
    anchors: tuple[str, ...] = ()
    if keyframes is not None:
        anchors = keyframe_anchors
        keyframe_key = None
        cached_keyframe = None
        if draft_cache is not None:
            keyframe_key = draft_cache.keyframe_key(
                first_frame=keyframe,
                width=args.width,
                height=args.height,
                compact_root=args.compact_root,
                patch_size=patch,
            )
            cached_keyframe = draft_cache.load_keyframe(keyframe_key)
        if cached_keyframe is not None:
            with record.phase(f"{label}keyframe_draft_cache_load"):
                condition_rows = mx.array(cached_keyframe)
                mx.eval(condition_rows)
                record.data[f"{label}keyframe_cache_hit"] = True
        else:
            with record.phase(f"{label}keyframe_encode"):
                keyframe_vae = load_compact_video_vae(args.compact_root / "video_vae.safetensors")
                condition_rows = encode_keyframe_rows(
                    keyframe_vae, keyframes, args.height, args.width, patch
                )
                mx.eval(condition_rows)
                print(f"keyframe conditioning rows: {condition_rows.shape}")
            del keyframe_vae
            if draft_cache is not None:
                with record.phase(f"{label}keyframe_draft_cache_store"):
                    draft_cache.store_keyframe(
                        keyframe_key, np.array(condition_rows.astype(mx.float32))
                    )
                    record.data[f"{label}keyframe_cache_hit"] = False
        release()

    reference_rows = None
    reference_grids: tuple = ()
    if cond.references:
        with record.phase(f"{label}reference_encode"):
            ref_vae = load_compact_video_vae(args.compact_root / "video_vae.safetensors")
            parts, grids = [], []
            for image in cond.references:
                w_ref, h_ref = image.size
                parts.append(encode_keyframe_rows(ref_vae, [image], h_ref, w_ref, patch))
                grids.append((h_ref // 16, w_ref // 16))
            reference_rows = mx.concatenate(parts) if len(parts) > 1 else parts[0]
            reference_grids = tuple(grids)
            mx.eval(reference_rows)
            print(f"reference images: {len(parts)} at {[g for g in grids]} latent, "
                  f"{reference_rows.shape[0]} rows", flush=True)
        del ref_vae
        release()

    # Media conditioning: carried history and/or a frozen picture, encoded while only the video VAE
    # is resident (it is unloaded again before the DiT lands, like the keyframe encode above).
    history_rows = None
    frozen_video_rows = None
    if cond.history_frames is not None or cond.frozen_frames is not None:
        with record.phase(f"{label}media_encode"):
            media_vae = load_compact_video_vae(args.compact_root / "video_vae.safetensors")
            if cond.history_frames is not None:
                history_rows = encode_video_rows(
                    media_vae, cond.history_frames, patch, sample=True
                )
            if cond.frozen_frames is not None:
                frozen_video_rows = encode_video_rows(
                    media_vae, cond.frozen_frames, patch, sample=False
                )
            print(
                "media conditioning rows: "
                f"history {None if history_rows is None else tuple(history_rows.shape)}, "
                f"frozen picture {None if frozen_video_rows is None else tuple(frozen_video_rows.shape)}",
                flush=True,
            )
        del media_vae
        release()

    history_audio_rows = None
    history_audio_latents = 0
    history_audio_offset = 0.0
    drive_audio_rows = None
    if cond.history_audio is not None or cond.drive_audio is not None:
        with record.phase(f"{label}audio_encode"):
            encode_vae = load_compact_audio_vae(args.compact_root / "audio_vae.safetensors")
            hop = encode_vae.config.hop_length
            rate = encode_vae.config.sampling_rate
            if cond.history_audio is not None and cond.history_audio.shape[1] >= hop:
                # Whole latents only, ending exactly where the carried picture hands over: the
                # carried span is (history + boundary) frames, which need not be a whole number of
                # 1/40 s latents, so the first one starts a fraction of a latent late instead.
                span_units = cond.history_audio.shape[1] / rate * 40.0
                history_audio_latents = int(np.floor(span_units + 1e-6))
                history_audio_offset = float(span_units - history_audio_latents)
                tail = cond.history_audio[:, -history_audio_latents * hop:]
                history_audio_rows = encode_audio_rows(encode_vae, tail, history_audio_latents)
            if cond.drive_audio is not None:
                drive_audio_rows = encode_audio_rows(encode_vae, cond.drive_audio, audio_latents)
            print(
                f"audio conditioning: history {history_audio_latents} latents "
                f"(offset {history_audio_offset:.3f}), "
                f"soundtrack {'held' if drive_audio_rows is not None else 'generated'}",
                flush=True,
            )
        del encode_vae
        release()

    # `--lora` repeats: every PATH[:SCALE] is one adapter, and they stack as a
    # sum of independent low-rank deltas on the same base layers (lora.py,
    # LoRALinear). Turbo plus a character LoRA is the case this exists for.
    # `lora_stack` is the ordered list of (path, scale); a single adapter is a
    # one-element list, so every consumer below handles one shape.
    lora_stack: list[tuple[Path, float]] = []
    # The phase is named for what is really loaded (`dit_load_q8` for the quantized build), so a
    # metrics file can no longer claim bf16 for a Q8 render.
    precision = dit_precision(args.dit)
    record.data[f"{label}dit_precision"] = precision
    with record.phase(f"{label}dit_load_{precision}"):
        dit = load_dit(args.dit, verbose=True)
        patch = dit.config.patch_size
        lora_spec = getattr(args, "lora", None)
        lora_specs = [lora_spec] if isinstance(lora_spec, (str, Path)) else list(lora_spec or [])
        lora_specs = [s for s in lora_specs if s]
        if lora_specs:
            from minimax_h3_mlx import lora as lora_mod

            lora_mod.AUDIT = bool(getattr(args, "lora_audit", False))
            lora_stack = [lora_mod.parse_spec(str(spec)) for spec in lora_specs]
            if not getattr(args, "lora_already_applied", False):
                reports = lora_mod.apply_loras(
                    dit,
                    lora_stack,
                    mode=getattr(args, "lora_mode", "runtime"),
                    verbose=True,
                    permute_qkv=bool(getattr(args, "lora_qkv_permute", False)),
                    swap_fc1=bool(getattr(args, "lora_fc1_swap", False)),
                )
                record.data[f"{label}lora"] = reports[0].summary()
                if len(reports) > 1:
                    record.data[f"{label}lora_stack"] = [r.summary() for r in reports]
            else:
                record.data[f"{label}lora"] = {
                    "path": str(lora_stack[0][0]),
                    "scale": float(lora_stack[0][1]),
                    "mode": "resident-preapplied",
                }
                if len(lora_stack) > 1:
                    record.data[f"{label}lora_stack"] = [
                        {"path": str(p), "scale": float(sc), "mode": "resident-preapplied"}
                        for p, sc in lora_stack]
            record.flush()

    if history_rows is not None:
        # Packed order is [history | keyframes]: the carried frames come first on the clock and
        # first in the conditioning block.
        condition_rows = (
            history_rows if condition_rows is None
            else mx.concatenate([history_rows, condition_rows])
        )
    history_latent_frames = 0
    if history_rows is not None:
        history_latent_frames = int(history_rows.shape[0]) // ((latent_h // patch[1]) * (latent_w // patch[2]))
    layout = build_packed_sequence(
        text_tags, latent_frames, latent_h, latent_w, audio_latents, patch, anchors,
        history_latent_frames=history_latent_frames,
        history_audio_latents=history_audio_latents,
        history_audio_offset=history_audio_offset,
        reference_grids=reference_grids,
    )
    n_target_video = int(layout.video_indices.shape[0]) - layout.num_condition_video_rows
    n_target_audio = int(layout.audio_indices.shape[0]) - layout.num_condition_audio_rows
    frozen_video_mask = None
    if frozen_video_rows is not None:
        if int(frozen_video_rows.shape[0]) != n_target_video:
            raise ValueError(
                f"the frozen picture encodes to {frozen_video_rows.shape[0]} rows, the window has "
                f"{n_target_video}"
            )
        frozen_video_mask = np.ones(n_target_video, dtype=bool)
    frozen_audio_mask = None
    if drive_audio_rows is not None:
        frozen_audio_mask = frozen_row_mask(n_target_audio, n_target_audio)
    if getattr(args, "draft_clock", "native") == "spread":
        spread_joint_draft_clock(layout, len(text_tags), args.draft_clock_scale)
        record.data[f"{label}draft_clock_scale"] = round(float(args.draft_clock_scale), 6)
    print(
        f"geometry: {args.width}x{args.height}, {frames} frames, "
        f"{layout.sequence_length:,} packed rows",
        flush=True,
    )
    record.data[f"{label}packed_rows"] = int(layout.sequence_length)
    record.data[f"{label}video_rows"] = int(layout.video_indices.shape[0])
    record.data[f"{label}audio_rows"] = int(layout.audio_indices.shape[0])
    record.flush()

    with record.phase(f"{label}adaln_cache_and_noise"):
        video_sched, audio_sched = build_schedules(args.steps, config, parse_sigma_subset(args.sigma_subset))
        record.data[f"{label}schedule"] = {
            "sigma_subset": args.sigma_subset,
            "video_sigmas": [round(float(v), 7) for v in video_sched.sigmas.tolist()],
            "audio_sigmas": [round(float(v), 7) for v in audio_sched.sigmas.tolist()],
        }
        print(f"schedule: video sigmas {[round(float(v), 4) for v in video_sched.sigmas.tolist()]}, "
              f"audio sigmas {[round(float(v), 4) for v in audio_sched.sigmas.tolist()]}", flush=True)
        timestep_table, plan = row_timestep_plan(
            layout, video_sched.timesteps, audio_sched.timesteps,
            frozen_video_mask, frozen_audio_mask,
        )
        adaln_key = None
        cached_adaln = None
        if draft_cache is not None:
            adaln_key = draft_cache.adaln_key(
                timestep_table=np.array(timestep_table),
                dit=args.dit,
                lora=lora_stack or None,
                lora_adaln=args.lora_adaln,
            )
            cached_adaln = draft_cache.load_adaln(adaln_key, dit)
        if cached_adaln is not None:
            cache, adaln_report = cached_adaln
            record.data[f"{label}adaln_cache_hit"] = True
            if adaln_report is not None:
                record.data[f"{label}lora_adaln"] = adaln_report
        else:
            cache = ModulationCache.build(dit, timestep_table, dtype=mx.bfloat16)
            mx.eval(cache.tables)
            adaln_report = None
            if lora_stack and getattr(args, "lora_adaln", None):
                from minimax_h3_mlx import lora as lora_mod

                reused_resident_table = bool(
                    getattr(ModulationCache, "last_build_reused", False)
                )
                if reused_resident_table and hasattr(ModulationCache, "current_lora_report"):
                    adaln_report = ModulationCache.current_lora_report()
                else:
                    # Each adapter's adaLN delta adds into the same tables; the
                    # whole stack is listed under "stack".
                    stack_reports = [
                        dict(lora_mod.absorb_adaln_lora(
                            dit, cache, lp, args.lora_adaln, ls, verbose=True
                        ), path=str(lp)) for lp, ls in lora_stack]
                    # Headline the adapter that actually carried adaLN pairs (Turbo, normally
                    # first), not whichever came last — a character LoRA stacked after Turbo has
                    # none, and its empty report used to stand in for the whole stack.
                    carriers = [r for r in stack_reports if r.get("blocks") or r.get("final")]
                    adaln_report = dict((carriers or stack_reports)[0])
                    if len(stack_reports) > 1:
                        adaln_report["stack"] = stack_reports
                    if hasattr(ModulationCache, "remember_lora_report"):
                        ModulationCache.remember_lora_report(adaln_report)
                record.data[f"{label}lora_adaln"] = adaln_report
                record.flush()
            if draft_cache is not None:
                draft_cache.store_adaln(adaln_key, cache, dit, adaln_report)
                record.data[f"{label}adaln_cache_hit"] = False
        if lora_stack and not getattr(args, "lora_already_applied", False):
            accounting = lora_accounting(record.data, label, adaln_report)
            record.data[f"{label}lora_accounting"] = accounting
            record.flush()
            print(
                f"LoRA accounting: {accounting['applied']} wrapped, "
                f"{accounting['adaln_absorbed']}/{accounting['adaln_pairs']} adaLN absorbed, "
                f"{accounting['unaccounted']} unaccounted"
                + ("" if accounting["unaccounted"] == 0 else
                   " — WARNING: these adapter modules were NOT applied"
                   + ("" if getattr(args, "lora_adaln", None) else
                      " (pass --lora-adaln TIME_EMBEDDER for adaLN pairs)")),
                flush=True,
            )
        revoice = None
        if revoice_wanted(args, cond, lora_stack):
            # Built here because the per-block adaLN projections are dropped just below: the
            # re-voice pass runs its own (audio-only) schedule, so it needs its own table.
            from minimax_h3_mlx import revoice as revoice_mod

            rv_video, rv_audio = build_schedules(args.revoice_steps, config)
            strength = float(getattr(args, "revoice_strength", 1.0) or 1.0)
            if strength < 1.0:
                # Start part-way down the audio schedule from the adapter's own sound, noised to
                # `strength`: the timing (what the lips do) is kept, the timbre is re-made.
                rv_audio.set_timesteps(sigmas=revoice_mod.start_sigmas(
                    [float(v) for v in rv_audio.sigmas.tolist()], strength))
            rv_forwards = len(rv_audio.timesteps)
            # The picture is held clean (t = 1), keyframe rows stay at their augmentation level,
            # and the text rows sit at t = 0 for every step - the level they see on step 0 of any
            # render - so nothing but the target audio rows changes across the pass and the
            # stored keys/values stay exact for every non-audio row.
            rv_table, rv_plan = row_timestep_plan(
                layout,
                mx.zeros((rv_forwards,), dtype=mx.float32),
                rv_audio.timesteps,
                np.ones(n_target_video, dtype=bool),
                None,
            )
            rv_cache = ModulationCache.build(dit, rv_table, dtype=mx.bfloat16)
            mx.eval(rv_cache.tables)
            drop = [Path(p) for p in (args.revoice_drop or [])] or ([lora_stack[0][0]] if lora_stack else [])
            revoice = {
                "mod": revoice_mod,
                "audio_sched": rv_audio,
                "table": rv_table,
                "plan": rv_plan,
                "cache": rv_cache,
                "forwards": rv_forwards,
                "drop": drop,
                "rows": revoice_mod.plan_rows(layout),
                "strength": strength,
            }
        skip_spec = getattr(args, "skip_blocks", None)
        if skip_spec:
            skipped_blocks = resolve_skip_blocks(skip_spec, cache, dit.config.num_layers)
            dit.skip_blocks = frozenset(skipped_blocks)
            record.data[f"{label}skip_blocks"] = skipped_blocks
            print(f"skipping DiT blocks {skipped_blocks} ({skip_spec})", flush=True)
        freed = drop_adaln_weights(dit)
        mx.clear_cache()
        print(f"AdaLN cache {cache.nbytes()/1024**2:.1f} MiB; dropped {gb(freed):.2f} GiB")
        # Draw order matches the reference pipeline: conditioning noise first, then video,
        # then audio, so a seed reproduces the same run with and without the staged loader.
        video_noise_shape = (1, dit.config.latents_dim, latent_frames, latent_h, latent_w)
        audio_noise_shape = (audio_latents * AUDIO_CHANNELS, dit.config.audio_latents_dim)
        noise_key = None
        cached_noise = None
        if draft_cache is not None:
            noise_key = draft_cache.noise_key(
                adaln_key=adaln_key,
                seed=seed,
                condition_shape=condition_rows.shape if condition_rows is not None else None,
                video_shape=video_noise_shape,
                audio_shape=audio_noise_shape,
            )
            cached_noise = draft_cache.load_noise(noise_key)
        if cached_noise is not None:
            condition_noise = (
                mx.array(cached_noise["condition"])
                if cached_noise["condition"] is not None
                else None
            )
            latents = mx.array(cached_noise["video"])
            audio_rows = mx.array(cached_noise["audio"])
            record.data[f"{label}noise_cache_hit"] = True
        else:
            mx.random.seed(seed)
            condition_noise = (
                mx.random.normal(condition_rows.shape).astype(mx.float32)
                if condition_rows is not None
                else None
            )
            latents = mx.random.normal(video_noise_shape).astype(mx.float32)
            audio_rows = mx.random.normal(audio_noise_shape).astype(mx.float32)
            if condition_noise is not None:
                mx.eval(condition_noise, latents, audio_rows)
            else:
                mx.eval(latents, audio_rows)
            if draft_cache is not None:
                draft_cache.store_noise(
                    noise_key,
                    video=np.array(latents),
                    audio=np.array(audio_rows),
                    condition=np.array(condition_noise) if condition_noise is not None else None,
                )
                record.data[f"{label}noise_cache_hit"] = False
        if condition_rows is not None:
            condition_rows = MiniMaxH3Scheduler(
                shift=config.sigma_shift_video
            ).scale_noise(condition_rows, KEYFRAME_NOISE_AUG, condition_noise)
        noised_refs = None
        if reference_rows is not None:
            # Their own draw, so a seed renders the same base noise with or without references.
            ref_noise = mx.random.normal(
                reference_rows.shape, key=mx.random.key(int(seed) * 131 + 17)
            ).astype(mx.float32)
            noised_refs = MiniMaxH3Scheduler(shift=config.sigma_shift_video).scale_noise(
                reference_rows, KEYFRAME_NOISE_AUG, ref_noise)
            condition_rows = (
                noised_refs if condition_rows is None
                else mx.concatenate([noised_refs, condition_rows])
            )
        video_rows = patchify_video_latents(latents, patch)
        frozen_video_values = None
        if frozen_video_rows is not None:
            # The supplied picture IS the target: held clean, decoded never (the source pixels are
            # the delivery). The noise above was still drawn so a seed keeps its meaning.
            frozen_video_values = frozen_video_rows.astype(mx.float32)
            video_rows = frozen_video_values
        frozen_audio_values = None
        if drive_audio_rows is not None:
            frozen_audio_values = drive_audio_rows.astype(mx.float32)
            audio_rows = frozen_audio_values
        if condition_rows is not None:
            video_rows = mx.concatenate([condition_rows, video_rows])
        if history_audio_rows is not None:
            # Carried sound is clean (t = 1), exactly like a reference audio row.
            audio_rows = mx.concatenate([history_audio_rows.astype(mx.float32), audio_rows])
        mx.eval(video_rows, audio_rows)

    n_cond_v = layout.num_condition_video_rows
    n_cond_a = layout.num_condition_audio_rows

    if live_preview is not None:
        live_preview.start_window(
            window=window_index,
            latent_frames=latent_frames,
            latent_height=latent_h,
            latent_width=latent_w,
            patch=patch,
        )

    num_forwards = len(video_sched.timesteps)
    step_cache = (
        StepResidualCache(args.step_cache, num_forwards, max_skip=args.step_cache_max_skip)
        if args.step_cache > 0 or args.step_cache_probe
        else None
    )

    step_times = []
    with record.phase(f"{label}joint_denoise"):
        for index, timestep in enumerate(video_sched.timesteps.tolist()):
            if live_preview is not None:
                # Checked before the forward is launched, so an abort costs at most the forward
                # already in flight — never a whole extra one.
                live_preview.check_abort(f"before forward {index + 1}/{num_forwards}")
            started = time.perf_counter()
            forward_args = (
                video_rows[None].astype(mx.bfloat16),
                audio_rows[None].astype(mx.bfloat16),
                embeds.astype(mx.bfloat16),
                timestep_table,
                plan[index],
                layout.token_tags,
                layout.position_ids,
                layout.video_indices,
                layout.audio_indices,
                layout.text_indices,
            )

            skip = False
            if step_cache is not None:
                x, _, adaln_indices, _ = dit.pack_inputs(*forward_args)
                indicator = dit.skip_indicator(x, adaln_indices, cache.get(0))
                mx.eval(indicator)
                del x
                skip = step_cache.decide(index, indicator)

            if skip:
                video_pred, audio_pred = step_cache.reuse(video_rows, audio_rows)
            else:
                video_out, audio_out = dit(*forward_args, modulation_cache=cache)
                video_pred = video_out[0].astype(mx.float32)
                audio_pred = audio_out[0].astype(mx.float32)
                if step_cache is not None:
                    step_cache.store(video_rows, audio_rows, video_pred, audio_pred)

            # The rows exactly as this forward saw them. `video_rows` is about to be rebound, and
            # the live preview's x0 estimate needs x_t and v from the *same* forward.
            forward_video_rows = video_rows

            # Only generated rows are written back; keyframe anchors survive untouched, so no
            # masking is needed. Rebind rather than assign into a slice — the stepped result is
            # a lazy graph over the very rows it would overwrite.
            stepped_video = video_sched.step(
                video_pred[n_cond_v:], float(timestep), video_rows[n_cond_v:]
            )
            stepped_audio = audio_sched.step(
                audio_pred[n_cond_a:],
                float(audio_sched.timesteps[index].item()),
                audio_rows[n_cond_a:],
            )
            # Frozen target rows are re-imposed after every step (the scheduler moved them along
            # with everything else; their timestep said "clean", so the model never asked to).
            if frozen_video_values is not None:
                stepped_video = frozen_video_values
            if frozen_audio_values is not None:
                stepped_audio = frozen_audio_values
            video_rows = (
                mx.concatenate([video_rows[:n_cond_v], stepped_video])
                if n_cond_v
                else stepped_video
            )
            audio_rows = (
                mx.concatenate([audio_rows[:n_cond_a], stepped_audio])
                if n_cond_a
                else stepped_audio
            )
            # One scalar rides the same eval: a NaN/inf latent (an overflowing adapter, 2026-09-17)
            # stops the render at the step it appears instead of decoding to a black "success".
            broken = nonfinite_flag(video_rows, audio_rows)
            mx.eval(video_rows, audio_rows, broken)
            if broken.item():
                raise nonfinite_error(
                    f"after denoise step {index + 1}/{num_forwards}", lora_stack)
            elapsed = time.perf_counter() - started
            step_times.append(elapsed)
            print(
                f"step {index+1}/{num_forwards}: {elapsed:.1f}s"
                f"{' [reused]' if skip else ''} "
                f"(active {gb(mx.get_active_memory()):.1f} GiB)",
                flush=True,
            )
            if live_preview is not None:
                # Deliberately outside the step timer: `step_times` stays comparable to every
                # render ever measured, and the preview's cost is reported as its own number.
                preview_seconds = live_preview.after_forward(
                    video_rows=forward_video_rows,
                    video_pred=video_pred,
                    n_cond_v=n_cond_v,
                    timestep=float(timestep),
                    forward_seconds=elapsed,
                )
                if preview_seconds:
                    print(
                        f"  live preview {live_preview.forward}/{live_preview.total_forwards} "
                        f"-> {live_preview.directory}/preview_"
                        f"{live_preview.forward:02d}.png ({preview_seconds:.2f}s)",
                        flush=True,
                    )
                live_preview.check_abort(f"after forward {index + 1}/{num_forwards}")
            del forward_video_rows
            if index == 0 and getattr(args, "lora_audit", False) and getattr(args, "lora", None):
                from minimax_h3_mlx import lora as lora_mod

                ratios = lora_mod.audit_output_scale(dit, limit=0)
                if ratios:
                    values = [r for _, r in ratios]
                    record.data[f"{label}lora_output_ratio"] = {
                        "n": len(values),
                        "median": float(np.median(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                    }
                    print(
                        f"lora |delta|/|base| on real activations: median {np.median(values):.3e}, "
                        f"min {np.min(values):.3e}, max {np.max(values):.3e} over {len(values)} layers "
                        f"(bf16 residual ULP is 3.9e-3)",
                        flush=True,
                    )
                    for name, ratio in ratios[:4]:
                        print(f"    {name}: {ratio:.3e}", flush=True)
    if step_cache is not None:
        record.data[f"{label}step_cache"] = step_cache.summary()
        print(f"step cache: reused {step_cache.skipped}/{num_forwards} forwards")

    adapter_audio_rows = None
    if revoice is not None:
        mx.eval(video_rows, audio_rows)
        if getattr(args, "revoice_keep_original", False):
            adapter_audio_rows = audio_rows
        audio_rows = run_revoice(
            args, record, label, dit, revoice, layout, video_rows, audio_rows, embeds, seed,
            n_cond_a, lora_stack,
        )
        del revoice

    mx.eval(video_rows, audio_rows)
    stage_a_cache = getattr(args, "save_stage_a", None)
    if stage_a_cache is not None:
        # Written before the VAE ever runs: the clean packed rows are the expensive artefact, and a
        # second pass wants them, not pixels. Conditioning rows are deliberately excluded — they
        # belong to this canvas's grid, and a refine at another canvas must rebuild them.
        with record.phase(f"{label}stage_a_save"):
            from minimax_h3_mlx.stage_cache import save_stage_a

            stored = save_stage_a(
                stage_a_cache,
                video_rows=np.array(video_rows[n_cond_v:].astype(mx.float32)),
                audio_rows=np.array(audio_rows[n_cond_a:].astype(mx.float32)),
                embeds=np.array(embeds.astype(mx.float32)),
                text_tags=np.asarray(text_tags),
                meta={
                    "prompt": prompt,
                    "seed": int(seed),
                    "frames": int(frames),
                    "height": int(args.height),
                    "width": int(args.width),
                    "latent_frames": int(latent_frames),
                    "latent_h": int(latent_h),
                    "latent_w": int(latent_w),
                    "audio_latents": int(audio_latents),
                    "patch": [int(value) for value in patch],
                    "latents_dim": int(dit.config.latents_dim),
                    "forwards": int(num_forwards),
                    "first_frame": first_frame_key,
                    "text_embed_sha256": digest,
                },
            )
            record.data[f"{label}stage_a_cache"] = {
                "path": str(stage_a_cache),
                "digests": stored["digests"],
                "shapes": stored["shapes"],
            }
            record.flush()
            print(f"stage-A cache: {stage_a_cache}", flush=True)
            for name, value in stored["digests"].items():
                print(f"  {name} {tuple(stored['shapes'][name])} sha256 {value[:16]}", flush=True)
    del dit, cache, plan, timestep_table, video_sched, audio_sched
    release()

    if not cond.decode_video:
        # Video-to-audio: the picture was held, so the delivery is the supplied pixels.
        video = np.asarray(cond.frozen_frames[:frames], dtype=np.uint8)
        del video_rows
    elif getattr(args, "draft_decode", "full") == "tae":
        with record.phase(f"{label}video_tae_decode"):
            from minimax_h3_mlx.tiny_video_vae import load_tiny_h3_video_decoder

            video_tae = load_tiny_h3_video_decoder(args.tae_checkpoint)
            video = decode_video_tae(
                video_tae, video_rows[n_cond_v:], frames, latent_frames, latent_h, latent_w, patch
            )
        del video_tae, video_rows
    else:
        # This is the dense/HQ path. Keep it isolated from preview changes: release validation
        # hashes this exact decoder's video stream against a pre-branch render.
        with record.phase(f"{label}video_vae_decode"):
            video_vae = load_compact_video_vae(args.compact_root / "video_vae.safetensors")
            video = decode_video(
                video_vae, video_rows[n_cond_v:], frames, latent_frames, latent_h, latent_w, patch
            )
        del video_vae, video_rows
    release()
    if cond.decode_video:
        check_decoded_video(video, lora_stack)

    with record.phase(f"{label}audio_vae_decode"):
        audio_vae = load_compact_audio_vae(args.compact_root / "audio_vae.safetensors")
        audio = decode_audio(audio_vae, audio_rows[n_cond_a:], audio_latents, frames)
        sample_rate = audio_vae.config.sampling_rate
        if adapter_audio_rows is not None:
            original = decode_audio(audio_vae, adapter_audio_rows[n_cond_a:], audio_latents, frames)
            keep = args.output.with_name(f"{args.output.stem}_{label}adapter_audio.wav")
            keep.parent.mkdir(parents=True, exist_ok=True)
            write_wav(keep, original, int(sample_rate))
            record.data[f"{label}revoice"]["adapter_audio"] = str(keep)
            del original
    del audio_vae, audio_rows, embeds, layout, adapter_audio_rows
    release()

    record.data[f"{label}mean_denoise_step_seconds"] = round(float(np.mean(step_times)), 3)
    record.data[f"{label}denoise_step_seconds"] = [round(value, 3) for value in step_times]
    record.flush()
    return video, audio, int(sample_rate), step_times


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="One prompt for the whole clip. Omit it and pass --chain-prompts to script a chain "
        "window by window.",
    )
    parser.add_argument("--dit", type=Path, required=True)
    parser.add_argument("--compact-root", type=Path, required=True)
    parser.add_argument("--text-config", type=Path, required=True)
    parser.add_argument("--prompt-cache", type=Path, default=None)
    parser.add_argument(
        "--draft-decode",
        choices=("full", "tae"),
        default="full",
        help="Video decoder for this render. 'full' is the untouched dense/HQ ViT VAE; 'tae' is "
        "the soft but fast preview decoder and requires --tae-checkpoint.",
    )
    parser.add_argument(
        "--tae-checkpoint",
        type=Path,
        default=None,
        help="madebyollin taeh3.safetensors, used only when --draft-decode tae is explicit.",
    )
    parser.add_argument(
        "--live-preview",
        choices=("off", "tae"),
        default="off",
        help="Publish a TAE thumbnail of the current x0 estimate after every denoising forward, "
        "so a bad take can be stopped at minute two instead of minute fifty. Reads the sampler "
        "state and writes only PNG/JSON, so the render itself is untouched. Requires "
        "--tae-checkpoint; composes with either --draft-decode.",
    )
    parser.add_argument(
        "--live-preview-every",
        type=int,
        default=1,
        metavar="N",
        help="Decode a preview on every Nth forward (default 1 = every forward). The last forward "
        "always publishes. Use this if the per-forward decode is a bigger share of a long render "
        "than you want to pay.",
    )
    parser.add_argument(
        "--live-preview-dir",
        type=Path,
        default=None,
        help="Where the preview PNGs, status.json and the ABORT sentinel live. Defaults to "
        "'live/' beside --output.",
    )
    parser.add_argument(
        "--live-preview-latent-frame",
        type=int,
        default=None,
        metavar="INDEX",
        help="Latent frame to preview; default is the middle of the window, which is the frame "
        "that says most about the take. status.json reports the pixel frame it corresponds to.",
    )
    parser.add_argument(
        "--live-preview-context",
        type=int,
        default=None,
        metavar="TOKENS",
        help="Latent frames of causal warm-up decoded ahead of the previewed frame. The tiny "
        "decoder's residual blocks remember the previous frame, so 0 renders the frame as if it "
        "opened the clip and smears it; 4 (the default) is indistinguishable from the full-sequence "
        "decode and costs the same as 2. See scripts/probe_preview_context.py.",
    )
    parser.add_argument(
        "--live-preview-downscale",
        type=int,
        default=0,
        metavar="N",
        help="Average-pool the previewed latent by N before decoding it. 0 (the default) picks the "
        "smallest power of two that keeps the decode's transient working set near 3 GiB — 1 at "
        "draft, 2 at 1024x576 and Native, where an unpooled preview would want 6-11 GiB on top of "
        "a render already peaking near 50. Pass 1 to force a full-resolution thumbnail.",
    )
    parser.add_argument(
        "--draft-cache-dir",
        type=Path,
        default=None,
        help="Opt-in bounded cache for TAE re-drafts. Reuses text embeddings by content SHA, "
        "AdaLN tables by sigma schedule, and seeded noise by geometry.",
    )
    parser.add_argument(
        "--draft-cache-limit",
        type=int,
        default=50,
        help="Maximum LRU entries per draft cache class (default: 50).",
    )
    parser.add_argument(
        "--draft-seconds",
        type=int,
        choices=tuple(DRAFT_DURATION_PRESETS),
        default=None,
        metavar="SECONDS",
        help="Recommended native-motion TAE preset: 3/5 seconds use one H3 window; 10/15 "
        "seconds use two/three conditioned windows and trim to exactly 240/360 frames. This "
        "overrides --frames and is only legal with --draft-decode tae.",
    )
    parser.add_argument(
        "--first-frame",
        type=Path,
        default=None,
        help="FL2VA first-frame conditioning: the still is fed to the vision tower as part of the "
        "request and video-VAE encoded into the packed keyframe slot.",
    )
    parser.add_argument(
        "--last-frame",
        type=Path,
        default=None,
        help="FL2VA last-frame conditioning: the clip ends on this still. With --first-frame it is "
        "the official first+last (FL2VA) task; on a chain it conditions the final window's last "
        "delivered frame.",
    )
    parser.add_argument(
        "--keyframe",
        action="append",
        default=None,
        metavar="PATH@WHEN",
        help="A timed keyframe: the clip passes through this still at WHEN — a delivered frame "
        "index ('48') or seconds ('2.5s'). Repeat for several. Frame 0 is the first frame and the "
        "last delivered frame is the last frame; anything between is pinned at its own instant "
        "on the target's clock. On a chain each keyframe is routed to the window that owns it.",
    )
    parser.add_argument(
        "--keyframe-alignment",
        choices=("auto", "off"),
        default="auto",
        help="'auto' prepends MiniMax's official alignment line (Picture N aligns with the S.SS-second "
        "mark of the target video) to a window that has a last-frame or timed keyframe and whose "
        "prompt does not already carry one. A plain first-frame window is left exactly as before.",
    )
    parser.add_argument(
        "--chain-context-frames",
        type=int,
        default=0,
        metavar="K",
        help="Carry the last K frames (a multiple of 17) of the previous window, plus their sound, "
        "into the next window as conditioning placed just before it on the clock — so motion and "
        "voice continue across the seam instead of restarting from one still. 0 (the default) is "
        "the original one-frame hand-over. 17 is the tested value.",
    )
    parser.add_argument(
        "--seam-colour-frames",
        type=int,
        default=0,
        metavar="N",
        help="Ease each new window's colour onto the previous window's last frame over N frames "
        "(each window decodes separately and lands a percent or two off). 0 = off; 24 is tested.",
    )
    parser.add_argument(
        "--extend-from",
        type=Path,
        default=None,
        metavar="VIDEO",
        help="Continue an existing clip: its tail (frames + sound) conditions --chain-windows new "
        "windows, and the output is the source followed by the continuation, on the render canvas. "
        "Uses --chain-context-frames (17 when 0 is given).",
    )
    parser.add_argument(
        "--extend-frames",
        type=int,
        default=None,
        metavar="N",
        help="With --extend-from: add exactly N frames after the source (trims the last new "
        "window). Without it every new window is delivered whole.",
    )
    parser.add_argument(
        "--audio-drive",
        type=Path,
        default=None,
        metavar="AUDIO",
        help="Lip-sync / music-driven video: this soundtrack is held in the target audio rows, so "
        "the picture is generated to it. Each window gets its own slice. The delivered audio is "
        "--audio-mux (default: this same file), never the VAE round trip.",
    )
    parser.add_argument("--audio-drive-offset", type=float, default=0.0, metavar="SECONDS")
    parser.add_argument(
        "--audio-mux",
        type=Path,
        default=None,
        metavar="AUDIO",
        help="With --audio-drive: the audio delivered in the mp4 (e.g. the full song when the drive "
        "is its vocal stem). Same offset.",
    )
    parser.add_argument(
        "--video-to-audio",
        type=Path,
        default=None,
        metavar="VIDEO",
        help="Generate sound for an existing clip: its frames are held in the target video rows and "
        "only audio is denoised. The output is the source's own video stream with the new sound. "
        "Long clips run as chained windows (sound carried across seams).",
    )
    parser.add_argument(
        "--skip-blocks",
        default=None,
        metavar="SPEC",
        help="EXPERIMENTAL speed lever: skip DiT blocks — 'auto:N' skips the N with the smallest "
        "residual gates, or list them ('12,23,31'). Blocks 0, 1 and 49 are never skipped.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        action="append",
        default=None,
        metavar="IMAGE",
        help="A reference image (a character's face or sheet): packed like MiniMax's ref2va image "
        "block - its own canvas, one slot on the clock before the target, labelled <Picture n> after "
        "any keyframes - so the prompt can name it. Repeat for several. Every window gets it.",
    )
    parser.add_argument(
        "--reference-size",
        type=int,
        default=2048,
        metavar="PIXELS",
        help="Short edge a reference image is encoded at (MiniMax's ref2va uses 2048). Smaller is "
        "cheaper: rows grow with the square of this.",
    )
    parser.add_argument(
        "--revoice-steps",
        type=int,
        default=0,
        metavar="POINTS",
        help="Re-voice: after the picture is made, switch the speed adapter off and re-make only the "
        "sound with the base model over POINTS sigma points (POINTS-1 audio steps), the picture held "
        "exactly as rendered. 0 (the default) is off. Runs only on windows rendered with --lora; "
        "skipped for --audio-drive and --video-to-audio.",
    )
    parser.add_argument(
        "--revoice-drop",
        type=Path,
        action="append",
        default=None,
        metavar="LORA",
        help="The adapter(s) re-voice switches off (paths as given to --lora). Default: the first "
        "--lora, which is where a speed adapter goes. Other LoRAs (a character's voice) stay on.",
    )
    parser.add_argument(
        "--revoice-kv",
        choices=("int8", "bf16", "off"),
        default="int8",
        help="Re-voice key/value store for the rows that do not change: int8 (default, half the "
        "memory of bf16), bf16, or off = run the exact full forward on every audio step (slow).",
    )
    parser.add_argument(
        "--revoice-strength",
        type=float,
        default=1.0,
        metavar="SIGMA",
        help="Where re-voice starts: 1.0 (default) re-makes the sound from noise; below 1 it starts "
        "from the adapter's own sound noised to this level, keeping its timing (lip-sync) and "
        "re-making its timbre. Only the schedule's points below SIGMA run.",
    )
    parser.add_argument(
        "--revoice-keep-original",
        action="store_true",
        help="Research: also write the adapter's own sound beside the output as "
        "<output>_adapter_audio.wav, for an A/B against the re-voiced track.",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Also write the delivered clip's frames as lossless PNGs here. An mp4 is the only "
        "artefact the runner leaves behind, so every downstream step (upscaling, grading, a second "
        "encode) starts from 4:2:0 and one generation of x264 loss. This is the escape hatch.",
    )
    parser.add_argument(
        "--save-stage-a",
        type=Path,
        default=None,
        metavar="NPZ",
        help="Write the clean Stage-A packed rows (generated video rows, frozen audio rows, text "
        "conditioning) to NPZ before decoding. A second pass — hires_refine.py, a re-decode, a "
        "resumed render — then costs its own forwards instead of the whole first pass again.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="x264 quality for the delivered mp4 (and the per-window previews). Lower is better; "
        "0 is lossless.",
    )
    parser.add_argument("--frames", type=int, default=22, help="snapped up to the 17n+5 grid")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--steps", type=int, default=9, help="sigma points; forwards = points - 1")
    parser.add_argument(
        "--lora",
        action="append",
        default=None,
        metavar="PATH[:SCALE]",
        help="Apply a low-rank adapter to the DiT. SCALE defaults to 1.0 (these checkpoints ship "
        "alpha == rank); SCALE 0 is a strict no-op and renders bit-identically to no --lora. "
        "Repeat the flag to stack adapters (Turbo plus a character LoRA, say); each keeps its "
        "own SCALE and the deltas add on the same base layers.",
    )
    parser.add_argument(
        "--lora-mode",
        choices=("runtime", "fuse"),
        default="runtime",
        help="'runtime' keeps the low-rank product as its own matmul (~1.5%% of a linear's FLOPs). "
        "'fuse' merges W += scale*(B@A) into the base weight at load — measured to lose ~87%% of "
        "this update to bfloat16 rounding, so it exists as a control arm, not a fast path.",
    )
    parser.add_argument(
        "--lora-adaln",
        type=Path,
        default=None,
        metavar="TIME_EMBEDDER",
        help="Also apply the LoRA's adaLN modules, which the pruned checkpoint cannot wrap. Needs "
        "the upstream time_embedder tensors (scripts/fetch_time_embedder.py); the delta is exact "
        "and is folded into the precomputed modulation cache, so it costs nothing per forward.",
    )
    parser.add_argument(
        "--lora-fc1-swap",
        action="store_true",
        help="Swap the fused mlp.fc1 SwiGLU halves of lora_B (OFF by default). Our FeedForward reads fc1 "
             "as [gate; value]; the ComfyUI definition stores [value; gate], so a ComfyUI-lineage "
             "LoRA applied verbatim puts the gate update on the value path. Kijai's conversions "
             "declare this in their own metadata (conversion: 'mlp.fc1 swiglu halves swapped').",
    )
    parser.add_argument(
        "--lora-qkv-permute",
        action="store_true",
        help="Re-order the fused qkv rows of lora_B (OFF by default; measured wrong for every converted file). The remap assumes the LoRA was "
             "trained through the ComfyUI model definition ((3, heads, head_dim)); this "
             "checkpoint stores (heads, 3, head_dim). Right for most CivitAI files, silent "
             "corruption for anything trained against the native layout — and corrupted "
             "attention presents as 'the effect works but faces and eyes are wrong'. Use this "
             "to test which lineage a file actually came from.",
    )
    parser.add_argument(
        "--lora-audit",
        action="store_true",
        help="Record |lora_out|/|base_out| on the first forward, per wrapped layer.",
    )
    parser.add_argument(
        "--sigma-subset",
        default=None,
        metavar="N:I0,I1,...",
        help="Use only these points of the shifted N-point sigma grid (e.g. 50:0,16,33,49 for "
        "TaoMate-H3's 3-forward ladder). --steps must equal the number of indices.",
    )
    parser.add_argument(
        "--vae-dtype",
        choices=("float32", "float16", "bfloat16"),
        default=None,
        help="ViT decoder arithmetic dtype (default float32, or H3_VAE_DTYPE). float16 matches the "
        "reference's FP16 autocast and ComfyUI's fp16 VAE.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wired-gb", type=float, default=50.0)
    parser.add_argument("--memory-gb", type=float, default=58.0)
    parser.add_argument(
        "--step-cache",
        type=float,
        default=0.0,
        metavar="THRESHOLD",
        help="TeaCache-class residual reuse: skip a forward while the accumulated relative L1 of "
        "the block-0 modulated input stays below THRESHOLD. 0 disables it.",
    )
    parser.add_argument(
        "--step-cache-max-skip",
        type=int,
        default=2,
        help="Longest run of consecutive skipped forwards allowed inside the error budget.",
    )
    parser.add_argument(
        "--step-cache-probe",
        action="store_true",
        help="Record the per-step relative L1 without ever skipping. Lets one reference render "
        "supply the curve every threshold would be chosen from, instead of guessing.",
    )
    parser.add_argument(
        "--playback-fps",
        type=float,
        default=None,
        help="Write the mp4 at this frame rate instead of 24. Use with a reduced --frames to trade "
        "temporal density for packed rows; audio is stretched to match.",
    )
    parser.add_argument(
        "--draft-fps",
        type=int,
        choices=(12, 15, 18),
        default=None,
        help="EXPERIMENTAL reduced-cadence TAE preview (not the recommended tier). --frames "
        "remains the delivery duration; the native "
        "joint audio/video grid is kept intact, then both streams are slowed by the same factor. "
        "Audio uses Apple's high-overlap time/pitch unit rather than ffmpeg atempo.",
    )
    parser.add_argument(
        "--draft-clock",
        choices=("native", "spread"),
        default="native",
        help="EXPERIMENTAL: with --draft-fps, keep the reduced joint clock native (slow-motion) or "
        "spread both audio and video positions over the requested timeline. No rows are added.",
    )
    parser.add_argument("--draft-output-width", type=int, default=None)
    parser.add_argument("--draft-output-height", type=int, default=None)
    parser.add_argument(
        "--chain-windows",
        type=int,
        default=1,
        metavar="N",
        help="Generate the clip as N chained --frames windows instead of one dense pass. Each "
        "window after the first is conditioned on its predecessor's last decoded frame through "
        "the ordinary first-frame keyframe path; the duplicate frame is dropped at the join. "
        "1 (the default) is the untouched dense path.",
    )
    parser.add_argument(
        "--chain-prompts",
        default=None,
        metavar="SPEC",
        help="One prompt per window instead of one prompt for the clip: either N prompts separated "
        f"by '{CHAIN_PROMPT_SEPARATOR}' or the path to a .json file holding a list of N strings, "
        "where N is --chain-windows. Use it to script a line of dialogue in one window only — a "
        "single prompt asks every window for that line, and each window delivers it. Mutually "
        "exclusive with the positional prompt; absent, every window gets the positional prompt.",
    )
    parser.add_argument(
        "--chain-total-frames",
        type=int,
        default=None,
        metavar="F",
        help="Trim the stitched clip to F frames. A chain delivers frames + (N-1)*(frames-1), so "
        "this is how a chain lands on an exact duration rather than a window multiple.",
    )
    parser.add_argument(
        "--chain-audio-crossfade",
        type=float,
        default=1.0 / FPS,
        metavar="SECONDS",
        help="Equal-power audio cross-fade at every seam, capped at one frame — the chain's real "
        "overlap. 0 reproduces WanGP's blind concatenation, seam pops included.",
    )
    parser.add_argument(
        "--chain-keep-windows",
        action="store_true",
        help="Also write each window's own mp4 (and its wav sidecar) beside the stitched output.",
    )
    args = parser.parse_args()

    if args.height % 32 or args.width % 32:
        parser.error("--height and --width must be multiples of 32")
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    try:
        subset = parse_sigma_subset(args.sigma_subset)
    except ValueError as exc:
        parser.error(str(exc))
    if subset is not None and len(subset[1]) != args.steps:
        parser.error(f"--sigma-subset keeps {len(subset[1])} points; pass --steps {len(subset[1])}")
    if args.reference:
        if args.draft_cache_dir is not None:
            parser.error("--reference and --draft-cache-dir do not compose")
        if args.reference_size < 256 or args.reference_size % 32:
            parser.error("--reference-size must be a multiple of 32, at least 256")
        for ref in args.reference:
            if not Path(ref).is_file():
                parser.error(f"--reference {ref} does not exist")
        # The prompt cache is keyed on prompt + first frame; a reference changes the embedding.
        args.prompt_cache = None
    if not 0.05 <= args.revoice_strength <= 1.0:
        parser.error("--revoice-strength must be in 0.05..1.0")
    if args.revoice_steps and args.revoice_steps < 2:
        parser.error("--revoice-steps must be 0 (off) or at least 2")
    if args.revoice_steps and not args.lora:
        parser.error("--revoice-steps re-makes the sound with an adapter switched off; pass --lora")
    if args.revoice_drop:
        given = {str(parse_lora_path(spec)) for spec in (args.lora or [])}
        missing = [str(p) for p in args.revoice_drop if str(p) not in given]
        if missing:
            parser.error(f"--revoice-drop names adapters not passed with --lora: {missing}")
    if args.vae_dtype:
        import os
        os.environ["H3_VAE_DTYPE"] = args.vae_dtype
    if args.draft_decode == "tae" and args.tae_checkpoint is None:
        parser.error("--draft-decode tae requires --tae-checkpoint")
    if args.draft_decode == "tae" and not args.tae_checkpoint.is_file():
        parser.error(f"--tae-checkpoint does not exist: {args.tae_checkpoint}")
    if args.live_preview != "off":
        if args.tae_checkpoint is None:
            parser.error("--live-preview tae requires --tae-checkpoint")
        if not args.tae_checkpoint.is_file():
            parser.error(f"--tae-checkpoint does not exist: {args.tae_checkpoint}")
        if args.live_preview_every < 1:
            parser.error("--live-preview-every must be at least 1")
        if args.live_preview_context is not None and args.live_preview_context < 0:
            parser.error("--live-preview-context cannot be negative")
        if args.live_preview_downscale < 0:
            parser.error("--live-preview-downscale cannot be negative")
    elif (
        args.live_preview_dir is not None
        or args.live_preview_latent_frame is not None
        or args.live_preview_context is not None
        or args.live_preview_downscale
    ):
        parser.error("the --live-preview-* options need --live-preview tae")
    if args.draft_cache_dir is not None and args.draft_decode != "tae":
        parser.error("--draft-cache-dir is draft-only; pass --draft-decode tae")
    if args.draft_cache_limit < 1:
        parser.error("--draft-cache-limit must be at least 1")
    if args.draft_seconds is not None:
        if args.draft_decode != "tae":
            parser.error("--draft-seconds is draft-only; pass --draft-decode tae")
        if args.draft_fps is not None or args.playback_fps is not None:
            parser.error("--draft-seconds keeps the native 24 fps clock; do not pass an fps flag")
        if args.chain_windows != 1 or args.chain_total_frames is not None:
            parser.error(
                "--draft-seconds owns the window count and exact trim; do not pass manual chain "
                "geometry"
            )
        args.frames, args.chain_windows, args.chain_total_frames = draft_duration_plan(
            args.draft_seconds
        )
        if not args.output.stem.endswith(f"_{args.draft_seconds}s"):
            args.output = args.output.with_name(
                f"{args.output.stem}_{args.draft_seconds}s{args.output.suffix}"
            )
    if args.draft_fps is not None and args.draft_decode != "tae":
        parser.error("--draft-fps is draft-only; pass --draft-decode tae")
    if args.draft_fps is not None and args.playback_fps is not None:
        parser.error("pass either --draft-fps or --playback-fps, not both")
    if args.draft_fps is not None and args.chain_windows > 1:
        parser.error("--draft-fps and --chain-windows do not compose")
    if args.draft_clock != "native" and args.draft_fps is None:
        parser.error("--draft-clock is only legal with --draft-fps")
    if (args.draft_output_width is None) != (args.draft_output_height is None):
        parser.error("pass both --draft-output-width and --draft-output-height")
    if args.draft_output_width is not None and args.draft_fps is None:
        parser.error("draft output resizing is only legal with --draft-fps")
    if args.chain_windows < 1:
        parser.error("--chain-windows must be at least 1")
    if args.chain_windows > 1 and args.playback_fps is not None:
        parser.error("--chain-windows and --playback-fps do not compose; the seam maths is 24 fps")
    if args.chain_windows > 1 and args.save_stage_a is not None:
        parser.error(
            "--save-stage-a holds one window's latents; with a chain every window would overwrite "
            "the last. Cache a single-window render."
        )
    media_mode = media_mode_errors(args, parser)
    requested_frames = align_num_frames(args.frames)
    draft_delivery_frames = int(args.frames) if args.draft_fps is not None else requested_frames
    frames = (
        draft_video_grid(draft_delivery_frames, args.draft_fps)
        if args.draft_fps is not None
        else requested_frames
    )
    if args.draft_fps is not None and not args.output.stem.endswith(f"_{args.draft_fps}fps"):
        args.output = args.output.with_name(
            f"{args.output.stem}_{args.draft_fps}fps{args.output.suffix}"
        )
    # Source media for the media-conditioned modes, read once, on the render canvas.
    source_video = source_audio = drive_wave = mux_wave = None
    try:
        if args.video_to_audio is not None:
            source_video = read_video_frames(args.video_to_audio, args.width, args.height)
            # The clip decides the shape: as many windows as it takes to cover it, trimmed to it.
            args.chain_windows = 1 + -(-max(0, len(source_video) - frames) // (frames - 1))
            args.chain_total_frames = len(source_video)
        if args.extend_from is not None:
            source_video = read_video_frames(args.extend_from, args.width, args.height)
            source_audio = fit_audio(read_audio(args.extend_from), len(source_video))
        if args.audio_drive is not None:
            drive_wave = read_audio(args.audio_drive, start_seconds=args.audio_drive_offset)
            if drive_wave.shape[1] < AUDIO_RATE // 4:
                parser.error(f"--audio-drive {args.audio_drive} has no usable audio")
            mux_wave = (
                read_audio(args.audio_mux, start_seconds=args.audio_drive_offset)
                if args.audio_mux is not None else drive_wave
            )
    except RuntimeError as exc:
        parser.error(str(exc))
    if args.extend_from is not None and args.extend_frames is not None:
        if args.extend_frames < 1:
            parser.error("--extend-frames must be at least 1")
        # The flag promises exactly N new frames: take as many windows as that needs rather
        # than silently capping it at what --chain-windows happened to say.
        needed = -(-int(args.extend_frames) // (frames - 1))
        if needed > args.chain_windows:
            if args.chain_prompts is not None:
                parser.error(f"--extend-frames {args.extend_frames} needs {needed} windows of "
                             f"{frames} frames; --chain-prompts names {args.chain_windows}")
            args.chain_windows = needed
    chain = args.chain_windows
    if args.extend_from is not None:
        # The source is "window 0": every new window drops its duplicated hand-over frame.
        chain_frames = len(source_video) + chain * (frames - 1)
        if args.extend_frames is not None:
            args.chain_total_frames = len(source_video) + args.extend_frames
    else:
        chain_frames = frames + (chain - 1) * (frames - 1)
    if args.chain_total_frames is not None and not 1 <= args.chain_total_frames <= chain_frames:
        parser.error(f"--chain-total-frames must be in 1..{chain_frames} for this chain")
    delivered_frames = (
        draft_delivery_frames
        if args.draft_fps is not None
        else (args.chain_total_frames or chain_frames)
    )
    args.draft_clock_scale = delivered_frames / frames if args.draft_fps is not None else 1.0
    # The new part's own timeline: frame 0 is the first frame this render conditions (the source's
    # last frame when extending), so keyframes and soundtrack slices are counted from there.
    timeline_offset = (len(source_video) - 1) if args.extend_from is not None else 0
    try:
        window_keyframes = plan_keyframes(
            args, frames, chain, delivered_frames - timeline_offset
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.chain_prompts is not None:
        if args.prompt is not None:
            parser.error(
                "pass either one positional prompt or --chain-prompts, not both — with both, "
                "which one a window should use is a guess"
            )
        try:
            prompts = parse_chain_prompts(args.chain_prompts, chain)
        except ValueError as exc:
            parser.error(str(exc))
    elif args.prompt is not None:
        # Backward compatible by construction: without --chain-prompts every window sees the same
        # text, which is exactly what the chain did before per-window prompts existed.
        prompts = [args.prompt] * chain
    else:
        parser.error(
            "a prompt is required: give one positionally, or one per window with --chain-prompts"
        )

    # Resolved after the --draft-seconds/--draft-fps output renames, so the preview directory
    # always sits beside the mp4 that will actually be written.
    live_preview_dir = None
    if args.live_preview != "off":
        live_preview_dir = (
            args.live_preview_dir
            if args.live_preview_dir is not None
            else args.output.parent / "live"
        )

    device = mx.device_info()
    max_wired = int(device["max_recommended_working_set_size"])
    wired_bytes = min(int(args.wired_gb * 1024**3), max_wired - 1024**2)
    memory_bytes = min(int(args.memory_gb * 1024**3), int(device["memory_size"]) - 1024**3)
    settings = {
        "prompt": prompts[0],
        "chain_prompts": prompts if args.chain_prompts is not None else None,
        "dit": str(args.dit),
        "compact_root": str(args.compact_root),
        "prompt_cache": str(args.prompt_cache) if args.prompt_cache else None,
        "draft_decode": args.draft_decode,
        "tae_checkpoint": str(args.tae_checkpoint) if args.tae_checkpoint else None,
        "live_preview": args.live_preview,
        "live_preview_every": args.live_preview_every if args.live_preview != "off" else None,
        "live_preview_dir": str(live_preview_dir) if live_preview_dir else None,
        "live_preview_context": (
            (
                LIVE_PREVIEW_DEFAULT_CONTEXT
                if args.live_preview_context is None
                else args.live_preview_context
            )
            if args.live_preview != "off"
            else None
        ),
        "live_preview_downscale": (
            args.live_preview_downscale if args.live_preview != "off" else None
        ),
        "draft_cache_dir": str(args.draft_cache_dir) if args.draft_cache_dir else None,
        "draft_cache_limit": args.draft_cache_limit if args.draft_cache_dir else None,
        "draft_seconds": args.draft_seconds,
        "save_stage_a": str(args.save_stage_a) if args.save_stage_a else None,
        "frames": frames,
        "delivery_frames": delivered_frames,
        "height": args.height,
        "width": args.width,
        "sigma_points": args.steps,
        "forwards": args.steps - 1,
        "seed": args.seed,
        "step_cache_threshold": args.step_cache,
        "step_cache_max_skip": args.step_cache_max_skip,
        "playback_fps": args.playback_fps or FPS,
        "draft_fps": args.draft_fps,
        "draft_unique_fps": (
            round(frames / (delivered_frames / FPS), 3) if args.draft_fps is not None else None
        ),
        "draft_clock": args.draft_clock if args.draft_fps is not None else None,
        "draft_clock_scale": (
            round(args.draft_clock_scale, 6) if args.draft_fps is not None else None
        ),
        "draft_output_width": args.draft_output_width,
        "draft_output_height": args.draft_output_height,
        "mux_fps": args.playback_fps or FPS,
        "chain_windows": chain,
        "chain_frames": chain_frames,
        "chain_delivered_frames": delivered_frames,
        "chain_audio_crossfade": args.chain_audio_crossfade if chain > 1 else None,
        "media_mode": media_mode,
        "chain_context_frames": int(args.chain_context_frames or 0),
        "first_frame": str(args.first_frame) if args.first_frame else None,
        "last_frame": str(args.last_frame) if args.last_frame else None,
        "keyframes": list(args.keyframe or []),
        "keyframe_alignment": args.keyframe_alignment,
        "extend_from": str(args.extend_from) if args.extend_from else None,
        "extend_source_frames": (
            len(source_video) if args.extend_from is not None and source_video is not None else None
        ),
        "audio_drive": str(args.audio_drive) if args.audio_drive else None,
        "audio_drive_offset": args.audio_drive_offset if args.audio_drive else None,
        "audio_mux": str(args.audio_mux) if args.audio_mux else None,
        "video_to_audio": str(args.video_to_audio) if args.video_to_audio else None,
        "skip_blocks": args.skip_blocks,
        "references": [str(p) for p in (args.reference or [])] or None,
        "reference_size": args.reference_size if args.reference else None,
        "revoice_steps": args.revoice_steps or None,
        "revoice_kv": args.revoice_kv if args.revoice_steps else None,
        "revoice_drop": [str(p) for p in (args.revoice_drop or [])] or None,
        # Which video-VAE decode mode produced this run, so a metrics file is self-describing
        # when someone compares decode phases across the campaign. 0/1 is the per-tile loop.
        "vae_decode_dtype": resolved_decode_dtype(),
        "vae_decode_batch": resolved_decode_batch(),
        "wired_gb": round(gb(wired_bytes), 3),
        "memory_limit_gb": round(gb(memory_bytes), 3),
        "device": device,
    }
    record = Recorder(args.metrics, settings)
    total_started = time.perf_counter()
    mx.set_wired_limit(wired_bytes)
    mx.set_memory_limit(memory_bytes)

    live_preview = None
    if args.live_preview != "off":
        # Built before any large model loads, so the 23 MB decoder is resident from the first
        # forward and its load never lands inside a measured step.
        live_preview = LivePreviewMonitor(
            live_preview_dir,
            args.tae_checkpoint,
            total_forwards=(args.steps - 1) * chain,
            total_windows=chain,
            sigma_points=args.steps,
            output=args.output,
            every=args.live_preview_every,
            latent_frame=args.live_preview_latent_frame,
            context=(
                LIVE_PREVIEW_DEFAULT_CONTEXT
                if args.live_preview_context is None
                else args.live_preview_context
            ),
            downscale=args.live_preview_downscale,
        )
        print(
            f"live preview: {live_preview.directory} "
            f"(every {live_preview.every} of {live_preview.total_forwards} forwards; "
            f"abort with `touch {live_preview.abort_path}`)",
            flush=True,
        )

    try:
        base_keyframe = None
        if args.first_frame is not None:
            from PIL import Image

            from minimax_h3_mlx.packing import prepare_keyframe_image

            # Put the still on the render canvas once, before either encoder sees it. The VAE would
            # do this anyway; doing it up front also means the vision tower's `smart_resize` is a
            # no-op on axes that are already multiples of 32, so both encoders read the same pixels.
            base_keyframe = prepare_keyframe_image(
                Image.open(args.first_frame).convert("RGB"),
                args.height,
                args.width,
                stretch=True,
            )

        reference_images = []
        if args.reference:
            from minimax_h3_mlx.reference import prepare_reference_image

            reference_images = [prepare_reference_image(p, args.reference_size) for p in args.reference]
        segments = []
        step_times = []
        window_report = []
        context = int(args.chain_context_frames or 0)
        if source_video is not None and args.extend_from is not None:
            # The source is window 0 of the timeline; it is delivered as decoded (on the canvas).
            segments.append((source_video, source_audio))
        keyframe_images: dict[Path, object] = {}

        def keyframe_image(path: Path):
            if path not in keyframe_images:
                from PIL import Image

                from minimax_h3_mlx.packing import prepare_keyframe_image

                # The first-frame still keeps the original stretch contract (the panel pre-fits
                # it); every other still is cover-cropped, the reference's rule for a follower.
                keyframe_images[path] = (
                    base_keyframe if (args.first_frame is not None and path == Path(args.first_frame))
                    else prepare_keyframe_image(
                        Image.open(path).convert("RGB"), args.height, args.width, stretch=False
                    )
                )
            return keyframe_images[path]

        for index in range(chain):
            label = "" if chain == 1 else f"w{index + 1}_"
            from PIL import Image

            cond = WindowConditioning()
            cond.references = reference_images
            window_start = index * (frames - 1)  # frame 0 of this window on the new timeline
            continuing = bool(segments)
            first_frame_key = ""
            prompt_cache = None
            if args.video_to_audio is not None:
                # The picture is given: hold it, show its first frame to the text encoder too.
                frozen = source_video[window_start : window_start + frames]
                if len(frozen) < frames:
                    frozen = np.concatenate(
                        [frozen, np.repeat(frozen[-1:], frames - len(frozen), axis=0)]
                    )
                cond.frozen_frames = frozen
                cond.keyframes.append((Image.fromarray(frozen[0]), "first"))
                first_frame_key = f"<video-to-audio window {index + 1}>"
                if index and context:
                    k = min(context, window_start) // CONTEXT_FRAME_STEP * CONTEXT_FRAME_STEP
                    if k:
                        cond.history_frames = source_video[window_start - k : window_start]
                        prev_audio = segments[-1][1]
                        cond.history_audio = prev_audio[:, -frame_samples(k + 1):]
            elif continuing:
                # The chain, in one line: the previous window's last decoded frame re-enters as
                # this window's first-frame keyframe. With carried context the frames before it —
                # and their sound — ride along as history rows just before the target on the
                # clock, so motion and voice continue instead of restarting from one still.
                prev_video, prev_audio = segments[-1]
                cond.keyframes.append((Image.fromarray(prev_video[-1]), "first"))
                first_frame_key = f"<chain window {index}>"
                k = min(context, len(prev_video) - 1) // CONTEXT_FRAME_STEP * CONTEXT_FRAME_STEP
                if k:
                    cond.history_frames = prev_video[-(k + 1) : -1]
                    cond.history_audio = (
                        drive_slice(drive_wave, window_start - k, k + 1)
                        if drive_wave is not None
                        else prev_audio[:, -frame_samples(k + 1):]
                    )
            for path, local in window_keyframes[index]:
                if continuing and local == 0:
                    continue  # the carried hand-over already owns frame 0
                cond.keyframes.append((keyframe_image(path), anchor_name(local, frames)))
            if index == 0 and not continuing and args.video_to_audio is None:
                anchors_now = [anchor for _, anchor in cond.keyframes]
                if anchors_now in ([], ["first"]):
                    # Exactly the old contract, so the prompt cache stays valid.
                    first_frame_key = str(args.first_frame) if args.first_frame else ""
                    prompt_cache = args.prompt_cache
            if drive_wave is not None:
                cond.drive_audio = drive_slice(drive_wave, window_start, frames)
            # Packed order is time order: first, then timed frames, then last.
            order = {"first": -1, "last": 10**9}
            cond.keyframes.sort(
                key=lambda item: order.get(item[1], int(item[1].split(":")[1]) if ":" in item[1] else 0)
            )
            window_prompt = prompts[index]
            if reference_images:
                # References are labelled after this window's keyframes. A chain's later windows
                # gain the carried hand-over frame, so "<Picture n>" naming a reference in window 1
                # names it one later there.
                if index == 0:
                    ref_base_keyframes = len(cond.keyframes)
                window_prompt = shift_reference_labels(
                    window_prompt, ref_base_keyframes, len(cond.keyframes))
            if args.keyframe_alignment == "auto":
                line = alignment_line([anchor for _, anchor in cond.keyframes], frames, window_prompt)
                if line:
                    window_prompt = f"{line}\n\n{window_prompt.lstrip()}"
            started = time.perf_counter()
            if chain > 1:
                print(f"\n### window {index + 1}/{chain} ###", flush=True)
            video, audio, sample_rate, window_steps = render_window(
                args,
                record,
                label=label,
                prompt=window_prompt,
                frames=frames,
                # A fresh seed per window: identical starting noise under a changed keyframe
                # correlates the windows' composition and camera motion, which is the one thing a
                # chain must not do.
                seed=args.seed + index,
                keyframe=None,
                first_frame_key=first_frame_key,
                prompt_cache=prompt_cache,
                # The x0 preview of a held picture is the picture; nothing to watch.
                live_preview=live_preview if cond.decode_video else None,
                window_index=index + 1,
                cond=cond,
            )
            segments.append((video, audio))
            step_times.extend(window_steps)
            window_report.append(
                {
                    "window": index + 1,
                    "seed": args.seed + index,
                    "seconds": round(time.perf_counter() - started, 3),
                    "keyframe": first_frame_key or None,
                    "anchors": [anchor for _, anchor in cond.keyframes],
                    "context_frames": 0 if cond.history_frames is None else len(cond.history_frames),
                    "prompt": window_prompt,
                }
            )
            record.data["windows"] = window_report
            record.flush()
            if args.chain_keep_windows and chain > 1:
                window_path = args.output.with_name(f"{args.output.stem}_w{index + 1}.mp4")
                window_path.parent.mkdir(parents=True, exist_ok=True)
                save_mp4(window_path, video, float(FPS), audio, sample_rate, crf=args.crf)
            release()

        with record.phase("encode_mux" if len(segments) == 1 else "stitch_and_mux"):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            playback = float(args.playback_fps or FPS)
            audio_tempo = playback / FPS
            # More than one segment = a seam to stitch: a chain, or a source clip
            # with its continuation (--extend-from, where the source is segment 0).
            if len(segments) > 1:
                video, audio, seams = stitch_windows(
                    segments, sample_rate, max(0.0, args.chain_audio_crossfade),
                    colour_match_frames=(
                        0 if args.video_to_audio is not None else args.seam_colour_frames
                    ),
                )
                video = video[:delivered_frames]
                audio = audio[:, : round(delivered_frames / FPS * sample_rate)]
                record.data["seams"] = seams
                print(
                    f"stitched {chain} windows into {len(video)} frames "
                    f"({len(video) / FPS:.2f} s) with {len(seams)} seam(s)",
                    flush=True,
                )
            else:
                video, audio = segments[0]
                if args.draft_fps is None and delivered_frames < len(video):
                    # A single window lands on its exact length too (a lip-sync of a 2.9 s line
                    # renders the 3 s window and delivers 2.9 s of it).
                    video = video[:delivered_frames]
                    audio = audio[:, : round(delivered_frames / FPS * sample_rate)]
            if mux_wave is not None:
                # Lip-sync: deliver the soundtrack the user gave (the full mix when the drive was a
                # vocal stem), sample-exact, never the VAE round trip of it.
                audio = fit_audio(mux_wave, len(video))
            audio_stretch_script = None
            audio_output_frames = None
            if args.draft_fps is not None:
                # Keep the exact joint tensor shape that passed the picture-quality gate. Only
                # after decoding, distribute its frames over the requested timeline and slow its
                # own synchronized audio by the identical ratio. AVAudioUnitTimePitch at maximum
                # overlap avoids the audible smearing seen with ffmpeg's 2x `atempo` stretch.
                source_frames = len(video)
                if args.draft_output_width is not None:
                    video = resize_draft_video(
                        video, args.draft_output_width, args.draft_output_height
                    )
                video = distribute_draft_frames(video, delivered_frames)
                playback = float(FPS)
                audio_tempo = source_frames / delivered_frames
                audio_stretch_script = Path(__file__).with_name("time_stretch_audio.swift")
                # Render 50 ms of the time-pitch unit's silent tail so the stretched waveform
                # never ends short of the picture; save_mp4 cuts the audio at exactly
                # frames / fps (it no longer relies on `-shortest`, which dropped a frame).
                audio_output_frames = round((delivered_frames / FPS + 0.05) * sample_rate)
            if args.video_to_audio is not None:
                # The source's own video stream, untouched, with the generated sound.
                mux_audio_onto_video(args.video_to_audio, audio, sample_rate, args.output)
            else:
                save_mp4(
                    args.output,
                    video,
                    playback,
                    audio,
                    sample_rate,
                    crf=args.crf,
                    audio_tempo=audio_tempo,
                    audio_stretch_script=audio_stretch_script,
                    audio_output_frames=audio_output_frames,
                )
            if args.frames_dir is not None:
                from minimax_h3_mlx.media import save_frames

                save_frames(args.frames_dir, video)
                print(f"wrote {len(video)} lossless frames to {args.frames_dir}", flush=True)
            # What the FILE holds, not what the array held: a mux that drops a frame must show
            # up here rather than be reported as a full delivery.
            encoded_frames = probe_video_frames(args.output)
            if args.video_to_audio is not None:
                # The delivery is the source's stream (its own frame rate); count what was copied.
                video = video if encoded_frames is None else video[:encoded_frames]
            if encoded_frames is not None and encoded_frames != len(video):
                print(f"WARNING: encoded {encoded_frames} frames, decoded {len(video)}",
                      flush=True)
            final_frames = encoded_frames if encoded_frames is not None else len(video)

        record.data.update(
            {
                "status": "done",
                "output": str(args.output),
                "delivered_frames": int(final_frames),
                "delivered_frames_source": "ffprobe" if encoded_frames is not None else "array",
                "decoded_frames": int(len(video)),
                "delivered_seconds": round(final_frames / playback, 3),
                "total_seconds": round(time.perf_counter() - total_started, 3),
                "mean_denoise_step_seconds": round(float(np.mean(step_times)), 3),
                "denoise_step_seconds": [round(value, 3) for value in step_times],
            }
        )
        if live_preview is not None:
            record.data["live_preview"] = live_preview.summary()
            live_preview.finish("done", extra={"output_written": str(args.output)})
        record.flush()
        print(f"\nwrote {args.output} in {record.data['total_seconds']/60:.1f} min")
        return 0
    except LivePreviewAborted as exc:
        # A clean stop, not a failure: nothing has been encoded yet, so there is no partial mp4 to
        # clean up, and the metrics file records where the run got to.
        record.data.update(
            {
                "status": "aborted",
                "aborted_by": "live-preview ABORT sentinel",
                "error": str(exc),
                "total_seconds": round(time.perf_counter() - total_started, 3),
            }
        )
        if live_preview is not None:
            record.data["live_preview"] = live_preview.summary()
        record.flush()
        print(f"\nABORTED: {exc}", flush=True)
        return ABORT_EXIT_CODE
    except BaseException as exc:
        if isinstance(exc, RenderIntegrityError):
            # Readable by the panel as-is: it shows `error_message`, not a traceback tail.
            record.data["error_kind"] = exc.kind
            record.data["error_message"] = str(exc)
            print(f"\nERROR: {exc}", flush=True)
        record.data.update(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "total_seconds": round(time.perf_counter() - total_started, 3),
            }
        )
        if live_preview is not None:
            record.data["live_preview"] = live_preview.summary()
            live_preview.finish("error", extra={"error": f"{type(exc).__name__}: {exc}"})
        record.flush()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
