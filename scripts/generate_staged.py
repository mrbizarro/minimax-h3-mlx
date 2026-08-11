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
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.adaln import ModulationCache, drop_adaln_weights
from minimax_h3_mlx.config import PipelineConfig
from minimax_h3_mlx.live_preview import (
    ABORT_EXIT_CODE,
    DEFAULT_CONTEXT as LIVE_PREVIEW_DEFAULT_CONTEXT,
    LivePreviewAborted,
    LivePreviewMonitor,
)
from minimax_h3_mlx.load import load_compact_audio_vae, load_compact_video_vae, load_dit
from minimax_h3_mlx.media import save_mp4
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
from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler
from minimax_h3_mlx.stepcache import StepResidualCache
from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder
from minimax_h3_mlx.video_vae import resolved_decode_batch


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


def build_schedules(points: int, config: PipelineConfig):
    video = MiniMaxH3Scheduler(shift=config.sigma_shift_video)
    audio = MiniMaxH3Scheduler(shift=config.sigma_shift_audio)
    video.set_timesteps(points)
    audio.set_timesteps(points)
    return video, audio


def row_timestep_plan(layout, video_timesteps, audio_timesteps):
    per_step = []
    for t, at in zip(video_timesteps.tolist(), audio_timesteps.tolist()):
        # Keyframe rows are pinned at their noise-augmentation level for the whole run. With no
        # keyframe the layout has no conditioning rows, so the extra level never appears in the
        # table and the schedule is unchanged.
        distinct, inverse = build_row_timesteps(
            layout, float(t), float(at), max(float(t), KEYFRAME_NOISE_AUG), 1.0
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


def stitch_windows(segments, sample_rate: int, crossfade_seconds: float):
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
):
    """Render one window end to end and hand back decoded pixels, audio and the sample rate.

    Every large model is loaded, used and released inside this call, so a chain of windows runs in
    one process without the previous window's DiT still being resident. ``label`` prefixes the
    phase names so a chained metrics file stays readable. ``prompt`` is this window's own text —
    a chain that scripts a line in one window must not encode that line in the others.
    """
    keyframes = [keyframe] if keyframe is not None else None
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
                load_vision=keyframes is not None,
                verbose=True,
                config_path=args.text_config,
            )
            embeds, text_tags = encoder.encode(prompt, keyframes)
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
        anchors = ("first",)
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

    lora_path = None
    lora_scale = None
    with record.phase(f"{label}dit_load_bf16"):
        dit = load_dit(args.dit, verbose=True)
        patch = dit.config.patch_size
        lora_spec = getattr(args, "lora", None)
        if lora_spec:
            from minimax_h3_mlx import lora as lora_mod

            lora_mod.AUDIT = bool(getattr(args, "lora_audit", False))
            lora_path, lora_scale = lora_mod.parse_spec(lora_spec)
            if not getattr(args, "lora_already_applied", False):
                lora_report = lora_mod.apply_lora(
                    dit,
                    lora_path,
                    lora_scale,
                    mode=getattr(args, "lora_mode", "runtime"),
                    verbose=True,
                )
                record.data[f"{label}lora"] = lora_report.summary()
            else:
                record.data[f"{label}lora"] = {
                    "path": str(lora_path),
                    "scale": float(lora_scale),
                    "mode": "resident-preapplied",
                }
            record.flush()

    layout = build_packed_sequence(
        text_tags, latent_frames, latent_h, latent_w, audio_latents, patch, anchors
    )
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
        video_sched, audio_sched = build_schedules(args.steps, config)
        timestep_table, plan = row_timestep_plan(
            layout, video_sched.timesteps, audio_sched.timesteps
        )
        adaln_key = None
        cached_adaln = None
        if draft_cache is not None:
            adaln_key = draft_cache.adaln_key(
                timestep_table=np.array(timestep_table),
                dit=args.dit,
                lora=(lora_path, lora_scale) if lora_path is not None else None,
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
            if getattr(args, "lora", None) and getattr(args, "lora_adaln", None):
                from minimax_h3_mlx import lora as lora_mod

                reused_resident_table = bool(
                    getattr(ModulationCache, "last_build_reused", False)
                )
                if reused_resident_table and hasattr(ModulationCache, "current_lora_report"):
                    adaln_report = ModulationCache.current_lora_report()
                else:
                    adaln_report = lora_mod.absorb_adaln_lora(
                        dit, cache, lora_path, args.lora_adaln, lora_scale, verbose=True
                    )
                    if hasattr(ModulationCache, "remember_lora_report"):
                        ModulationCache.remember_lora_report(adaln_report)
                record.data[f"{label}lora_adaln"] = adaln_report
                record.flush()
            if draft_cache is not None:
                draft_cache.store_adaln(adaln_key, cache, dit, adaln_report)
                record.data[f"{label}adaln_cache_hit"] = False
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
        video_rows = patchify_video_latents(latents, patch)
        if condition_rows is not None:
            video_rows = mx.concatenate([condition_rows, video_rows])
        mx.eval(video_rows, audio_rows)

    n_cond_v = layout.num_condition_video_rows

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
            audio_rows = audio_sched.step(
                audio_pred,
                float(audio_sched.timesteps[index].item()),
                audio_rows,
            )
            video_rows = (
                mx.concatenate([video_rows[:n_cond_v], stepped_video])
                if n_cond_v
                else stepped_video
            )
            mx.eval(video_rows, audio_rows)
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
                audio_rows=np.array(audio_rows.astype(mx.float32)),
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

    if getattr(args, "draft_decode", "full") == "tae":
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

    with record.phase(f"{label}audio_vae_decode"):
        audio_vae = load_compact_audio_vae(args.compact_root / "audio_vae.safetensors")
        audio = decode_audio(audio_vae, audio_rows, audio_latents, frames)
        sample_rate = audio_vae.config.sampling_rate
    del audio_vae, audio_rows, embeds, layout
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
        default=None,
        metavar="PATH[:SCALE]",
        help="Apply a low-rank adapter to the DiT. SCALE defaults to 1.0 (these checkpoints ship "
        "alpha == rank); SCALE 0 is a strict no-op and renders bit-identically to no --lora.",
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
        "--lora-audit",
        action="store_true",
        help="Record |lora_out|/|base_out| on the first forward, per wrapped layer.",
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
    chain = args.chain_windows
    chain_frames = frames + (chain - 1) * (frames - 1)
    if args.chain_total_frames is not None and not 1 <= args.chain_total_frames <= chain_frames:
        parser.error(f"--chain-total-frames must be in 1..{chain_frames} for this chain")
    delivered_frames = (
        draft_delivery_frames
        if args.draft_fps is not None
        else (args.chain_total_frames or chain_frames)
    )
    args.draft_clock_scale = delivered_frames / frames if args.draft_fps is not None else 1.0

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
        # Which video-VAE decode mode produced this run, so a metrics file is self-describing
        # when someone compares decode phases across the campaign. 0/1 is the per-tile loop.
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

        segments = []
        step_times = []
        window_report = []
        for index in range(chain):
            label = "" if chain == 1 else f"w{index + 1}_"
            if index == 0:
                keyframe = base_keyframe
                first_frame_key = str(args.first_frame) if args.first_frame else ""
                prompt_cache = args.prompt_cache
            else:
                from PIL import Image

                # The chain, in one line: the previous window's last decoded frame re-enters as this
                # window's first-frame keyframe. The ordinary keyframe path VAE-encodes it,
                # noises it to 0.999 and RoPE-anchors it to frame 0 of the new timeline, so
                # nothing inside the denoiser knows it is in a chain. It is already on the
                # canvas, so no resampling pass stands between one window and the next.
                keyframe = Image.fromarray(segments[-1][0][-1])
                first_frame_key = f"<chain window {index}>"
                # The keyframe changes every window and its vision tokens with it, so a prompt cache
                # keyed on (prompt, first frame) can only ever hit on window 1 — per-window prompts
                # do not change that, they only give the cache a second reason to miss.
                prompt_cache = None
            started = time.perf_counter()
            if chain > 1:
                print(f"\n### window {index + 1}/{chain} ###", flush=True)
            video, audio, sample_rate, window_steps = render_window(
                args,
                record,
                label=label,
                prompt=prompts[index],
                frames=frames,
                # A fresh seed per window: identical starting noise under a changed keyframe
                # correlates the windows' composition and camera motion, which is the one thing a
                # chain must not do.
                seed=args.seed + index,
                keyframe=keyframe,
                first_frame_key=first_frame_key,
                prompt_cache=prompt_cache,
                live_preview=live_preview,
                window_index=index + 1,
            )
            segments.append((video, audio))
            step_times.extend(window_steps)
            window_report.append(
                {
                    "window": index + 1,
                    "seed": args.seed + index,
                    "seconds": round(time.perf_counter() - started, 3),
                    "keyframe": first_frame_key or None,
                    "prompt": prompts[index],
                }
            )
            record.data["windows"] = window_report
            record.flush()
            if args.chain_keep_windows and chain > 1:
                window_path = args.output.with_name(f"{args.output.stem}_w{index + 1}.mp4")
                window_path.parent.mkdir(parents=True, exist_ok=True)
                save_mp4(window_path, video, float(FPS), audio, sample_rate, crf=args.crf)
            release()

        with record.phase("encode_mux" if chain == 1 else "stitch_and_mux"):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            playback = float(args.playback_fps or FPS)
            audio_tempo = playback / FPS
            if chain > 1:
                video, audio, seams = stitch_windows(
                    segments, sample_rate, max(0.0, args.chain_audio_crossfade)
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
                # AAC's frame boundary plus `-shortest` otherwise cuts a video frame when the
                # exact stretched waveform ends a fraction of a millisecond before the nominal
                # video boundary. Render 50 ms of the time-pitch unit's silent tail; ffmpeg trims
                # it against the 124-frame video instead of trimming the video to 123 frames.
                audio_output_frames = round((delivered_frames / FPS + 0.05) * sample_rate)
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

        record.data.update(
            {
                "status": "done",
                "output": str(args.output),
                "delivered_frames": int(len(video)),
                "delivered_seconds": round(len(video) / FPS, 3),
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
