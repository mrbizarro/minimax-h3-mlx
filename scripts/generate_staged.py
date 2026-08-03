"""Generate MiniMax-H3 locally with one large component resident at a time.

This runner targets a 64 GB Apple Silicon desktop. It consumes the quality-first pruned BF16 DiT,
ddalcu's native MLX Q8 text encoder, and ddalcu's compact VAE exports. Every phase is materialized,
timed, recorded, and explicitly released before the next model is loaded.
"""

from __future__ import annotations

import argparse
import gc
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


def decode_audio(model, rows, audio_latents: int, frames: int):
    cfg = model.config
    latents = unpack_audio_tokens(rows, audio_latents)
    mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1)
    std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1)
    waveform = np.array(model.decode((latents * std + mean).astype(mx.float32)))[:, 0, :]
    samples = round(frames / FPS * cfg.sampling_rate)
    return waveform[:, :samples].astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--dit", type=Path, required=True)
    parser.add_argument("--compact-root", type=Path, required=True)
    parser.add_argument("--text-config", type=Path, required=True)
    parser.add_argument("--prompt-cache", type=Path, default=None)
    parser.add_argument(
        "--first-frame",
        type=Path,
        default=None,
        help="FL2VA first-frame conditioning: the still is fed to the vision tower as part of the "
        "request and video-VAE encoded into the packed keyframe slot.",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=22, help="snapped up to the 17n+5 grid")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--steps", type=int, default=9, help="sigma points; forwards = points - 1")
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
    args = parser.parse_args()

    if args.height % 32 or args.width % 32:
        parser.error("--height and --width must be multiples of 32")
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    frames = align_num_frames(args.frames)
    device = mx.device_info()
    max_wired = int(device["max_recommended_working_set_size"])
    wired_bytes = min(int(args.wired_gb * 1024**3), max_wired - 1024**2)
    memory_bytes = min(int(args.memory_gb * 1024**3), int(device["memory_size"]) - 1024**3)
    settings = {
        "prompt": args.prompt,
        "dit": str(args.dit),
        "compact_root": str(args.compact_root),
        "prompt_cache": str(args.prompt_cache) if args.prompt_cache else None,
        "frames": frames,
        "height": args.height,
        "width": args.width,
        "sigma_points": args.steps,
        "forwards": args.steps - 1,
        "seed": args.seed,
        "step_cache_threshold": args.step_cache,
        "step_cache_max_skip": args.step_cache_max_skip,
        "playback_fps": args.playback_fps or FPS,
        "wired_gb": round(gb(wired_bytes), 3),
        "memory_limit_gb": round(gb(memory_bytes), 3),
        "device": device,
    }
    record = Recorder(args.metrics, settings)
    total_started = time.perf_counter()
    mx.set_wired_limit(wired_bytes)
    mx.set_memory_limit(memory_bytes)

    try:
        keyframes = None
        if args.first_frame is not None:
            from PIL import Image

            from minimax_h3_mlx.packing import prepare_keyframe_image

            # Put the still on the render canvas once, before either encoder sees it. The VAE would
            # do this anyway; doing it up front also means the vision tower's `smart_resize` is a
            # no-op on axes that are already multiples of 32, so both encoders read the same pixels.
            keyframes = [
                prepare_keyframe_image(
                    Image.open(args.first_frame).convert("RGB"),
                    args.height,
                    args.width,
                    stretch=True,
                )
            ]

        if args.prompt_cache is not None and args.prompt_cache.exists():
            with record.phase("text_cache_load"):
                cached = np.load(args.prompt_cache, allow_pickle=False)
                cached_prompt = str(cached["prompt"].item())
                if cached_prompt != args.prompt:
                    raise ValueError("The prompt cache belongs to a different prompt.")
                cached_frame = str(cached["first_frame"].item()) if "first_frame" in cached else ""
                if cached_frame != (str(args.first_frame) if args.first_frame else ""):
                    raise ValueError("The prompt cache was built for a different first frame.")
                embeds = mx.array(cached["embeds"]).astype(mx.bfloat16)
                text_tags = cached["text_tags"].astype(np.int64)
                mx.eval(embeds)
        else:
            with record.phase("text_encode_q8"):
                # A keyframe request runs through the vision tower: H3 conditions on the still
                # twice, once as VL tokens in the text stream and once as VAE conditioning rows.
                encoder = MiniMaxH3TextEncoder(
                    args.compact_root,
                    dtype=mx.bfloat16,
                    load_vision=keyframes is not None,
                    verbose=True,
                    config_path=args.text_config,
                )
                embeds, text_tags = encoder.encode(args.prompt, keyframes)
                embeds = mx.array(embeds)
                mx.eval(embeds)
                if args.prompt_cache is not None:
                    args.prompt_cache.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        args.prompt_cache,
                        prompt=np.array(args.prompt),
                        first_frame=np.array(str(args.first_frame) if args.first_frame else ""),
                        embeds=np.array(embeds.astype(mx.float32)),
                        text_tags=text_tags,
                    )
            del encoder
        record.data["prompt_tokens"] = int(len(text_tags))
        record.flush()
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
            with record.phase("keyframe_encode"):
                keyframe_vae = load_compact_video_vae(args.compact_root / "video_vae.safetensors")
                condition_rows = encode_keyframe_rows(
                    keyframe_vae, keyframes, args.height, args.width, patch
                )
                mx.eval(condition_rows)
                print(f"keyframe conditioning rows: {condition_rows.shape}")
            del keyframe_vae
            release()

        with record.phase("dit_load_bf16"):
            dit = load_dit(args.dit, verbose=True)
            patch = dit.config.patch_size

        layout = build_packed_sequence(
            text_tags, latent_frames, latent_h, latent_w, audio_latents, patch, anchors
        )
        print(
            f"geometry: {args.width}x{args.height}, {frames} frames, "
            f"{layout.sequence_length:,} packed rows",
            flush=True,
        )
        record.data["packed_rows"] = int(layout.sequence_length)
        record.data["video_rows"] = int(layout.video_indices.shape[0])
        record.data["audio_rows"] = int(layout.audio_indices.shape[0])
        record.flush()

        with record.phase("adaln_cache_and_noise"):
            video_sched, audio_sched = build_schedules(args.steps, config)
            timestep_table, plan = row_timestep_plan(
                layout, video_sched.timesteps, audio_sched.timesteps
            )
            cache = ModulationCache.build(dit, timestep_table, dtype=mx.bfloat16)
            mx.eval(cache.tables)
            freed = drop_adaln_weights(dit)
            mx.clear_cache()
            print(f"AdaLN cache {cache.nbytes()/1024**2:.1f} MiB; dropped {gb(freed):.2f} GiB")
            # Draw order matches the reference pipeline: conditioning noise first, then video,
            # then audio, so a seed reproduces the same run with and without the staged loader.
            mx.random.seed(args.seed)
            if condition_rows is not None:
                condition_noise = mx.random.normal(condition_rows.shape).astype(mx.float32)
                condition_rows = MiniMaxH3Scheduler(
                    shift=config.sigma_shift_video
                ).scale_noise(condition_rows, KEYFRAME_NOISE_AUG, condition_noise)
            latents = mx.random.normal(
                (1, dit.config.latents_dim, latent_frames, latent_h, latent_w)
            ).astype(mx.float32)
            video_rows = patchify_video_latents(latents, patch)
            audio_rows = mx.random.normal(
                (audio_latents * AUDIO_CHANNELS, dit.config.audio_latents_dim)
            ).astype(mx.float32)
            if condition_rows is not None:
                video_rows = mx.concatenate([condition_rows, video_rows])
            mx.eval(video_rows, audio_rows)

        n_cond_v = layout.num_condition_video_rows

        num_forwards = len(video_sched.timesteps)
        step_cache = (
            StepResidualCache(
                args.step_cache, num_forwards, max_skip=args.step_cache_max_skip
            )
            if args.step_cache > 0 or args.step_cache_probe
            else None
        )

        step_times = []
        with record.phase("joint_denoise"):
            for index, timestep in enumerate(video_sched.timesteps.tolist()):
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
        if step_cache is not None:
            record.data["step_cache"] = step_cache.summary()
            print(f"step cache: reused {step_cache.skipped}/{num_forwards} forwards")

        mx.eval(video_rows, audio_rows)
        del dit, cache, plan, timestep_table, video_sched, audio_sched
        release()

        with record.phase("video_vae_decode"):
            video_vae = load_compact_video_vae(args.compact_root / "video_vae.safetensors")
            video = decode_video(
                video_vae, video_rows[n_cond_v:], frames, latent_frames, latent_h, latent_w, patch
            )
        del video_vae, video_rows
        release()

        with record.phase("audio_vae_decode"):
            audio_vae = load_compact_audio_vae(args.compact_root / "audio_vae.safetensors")
            audio = decode_audio(audio_vae, audio_rows, audio_latents, frames)
            sample_rate = audio_vae.config.sampling_rate
        del audio_vae, audio_rows
        release()

        with record.phase("encode_mux"):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            playback = float(args.playback_fps or FPS)
            save_mp4(
                args.output,
                video,
                playback,
                audio,
                sample_rate,
                audio_tempo=playback / FPS,
            )

        record.data.update(
            {
                "status": "done",
                "output": str(args.output),
                "total_seconds": round(time.perf_counter() - total_started, 3),
                "mean_denoise_step_seconds": round(float(np.mean(step_times)), 3),
                "denoise_step_seconds": [round(value, 3) for value in step_times],
            }
        )
        record.flush()
        print(f"\nwrote {args.output} in {record.data['total_seconds']/60:.1f} min")
        return 0
    except BaseException as exc:
        record.data.update(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "total_seconds": round(time.perf_counter() - total_started, 3),
            }
        )
        record.flush()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
