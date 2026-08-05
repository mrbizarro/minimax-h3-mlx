"""Second-pass latent refine of a cached Stage-A render — the community hires-fix pattern.

The shape of the pass follows Tr1dae's ``ComfyUI-MiniMaxH3_LatentUpscaler`` (see
``notes/TR1DAE_NODE.md``): the clean Stage-A video latent is spatially interpolated onto a larger
grid, re-noised to a re-entry sigma with the scheduler's own geometry, and finished with the tail
of a reference-shaped schedule at the new canvas. Three details from that node are load-bearing and
are implemented here rather than assumed:

* **Every stream that carries grid metadata lives on the target grid.** The packed layout, the RoPE
  position grid and the keyframe conditioning rows are all rebuilt at the refine canvas; the
  keyframe is re-encoded from the original still at the target canvas rather than upsampled from a
  Stage-A latent, and a runtime assertion proves its row count equals the target layout's
  conditioning-row count. Reusing Stage-A-shaped conditioning is the documented "classic identity
  warp".
* **Axes snap to the DiT patch grid.** Cond patchify does not pad; an odd latent axis crashes the
  reshape. Both axes are required to be multiples of 32 (a multiple-of-2 latent grid).
* **``nan_to_num`` guards the re-noise mix**, and audio is frozen at sigma 0 — the node's
  ``audio_denoise = 0`` semantics. Audio is never re-noised, never stepped, never remixed.

Stage A is not re-run: it is read from a ``--save-stage-a`` cache, whose arrays are verified against
their write-time SHA-256s before a single forward is spent on them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_staged import Recorder, decode_audio, decode_video, gb, release  # noqa: E402
from minimax_h3_mlx.adaln import ModulationCache, drop_adaln_weights  # noqa: E402
from minimax_h3_mlx.config import PipelineConfig  # noqa: E402
from minimax_h3_mlx.load import (  # noqa: E402
    load_compact_audio_vae,
    load_compact_video_vae,
    load_dit,
)
from minimax_h3_mlx.media import save_mp4  # noqa: E402
from minimax_h3_mlx.packing import (  # noqa: E402
    AUDIO_CHANNELS,
    FPS,
    KEYFRAME_NOISE_AUG,
    build_packed_sequence,
    build_row_timesteps,
    patchify_video_latents,
    unpatchify_video_tokens,
)
from minimax_h3_mlx.pipeline import encode_keyframe_rows  # noqa: E402
from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler  # noqa: E402
from minimax_h3_mlx.stage_cache import load_stage_a  # noqa: E402

# Audio rows come out of Stage A clean, at sigma 0. In this port's convention that is t = 1.
CLEAN_TIMESTEP = 1.0


def shift_forward(base: np.ndarray, shift: float) -> np.ndarray:
    """The scheduler's exponential sigma shift, ``s*b / (1 + (s-1)*b)``."""
    shift32 = np.float32(shift)
    return (shift32 * base) / (np.float32(1.0) + np.float32(shift - 1.0) * base)


def shift_inverse(sigma: float, shift: float) -> float:
    """Undo the sigma shift: which point of the ``linspace(1, 0)`` grid maps to ``sigma``."""
    shift32 = np.float32(shift)
    sigma32 = np.float32(sigma)
    return float(sigma32 / (shift32 - np.float32(shift - 1.0) * sigma32))


def refine_sigmas(sigma: float, forwards: int, shift: float) -> list[float]:
    """A reference-shaped schedule suffix: ``forwards`` steps from ``sigma`` down to 0.

    The released schedule is a uniform ``linspace(1, 0, N)`` grid pushed through the shift, so the
    honest way to ask for "the tail from sigma" is to invert the shift, lay a uniform grid over
    ``[0, b0]`` in that base coordinate, and push it back through. Laying the grid on the shifted
    axis directly would hand the tail a geometry the model never sees.
    """
    if not 0.0 < sigma < 1.0:
        raise ValueError(f"the re-entry sigma must lie in (0, 1), got {sigma}")
    if forwards < 1:
        raise ValueError(f"a refine needs at least one forward, got {forwards}")
    base = np.linspace(shift_inverse(sigma, shift), 0.0, forwards + 1, dtype=np.float32)
    values = [float(value) for value in shift_forward(base, shift)]
    values[0] = float(np.float32(sigma))
    values[-1] = 0.0
    return values


def resize_latent_bilinear(latents: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinear spatial resize of ``(B, C, F, H, W)`` latents, ``align_corners=False``.

    The reference node's default and the one the campaign's own evidence exonerated: the dot
    lattice H1 produced is an underdosed-tail signature, not a property of interpolating latents.
    """
    b, c, f, h, w = latents.shape
    if (h, w) == (height, width):
        return latents.copy()

    def axis_weights(src: int, dst: int):
        scale = src / dst
        centres = (np.arange(dst, dtype=np.float64) + 0.5) * scale - 0.5
        centres = np.clip(centres, 0.0, src - 1)
        lo = np.floor(centres).astype(np.int64)
        hi = np.minimum(lo + 1, src - 1)
        frac = (centres - lo).astype(np.float32)
        return lo, hi, frac

    y_lo, y_hi, y_frac = axis_weights(h, height)
    x_lo, x_hi, x_frac = axis_weights(w, width)

    rows = latents[:, :, :, y_lo, :] * (1.0 - y_frac)[None, None, None, :, None] + latents[
        :, :, :, y_hi, :
    ] * y_frac[None, None, None, :, None]
    out = rows[:, :, :, :, x_lo] * (1.0 - x_frac)[None, None, None, None, :] + rows[
        :, :, :, :, x_hi
    ] * x_frac[None, None, None, None, :]
    return out.astype(latents.dtype)


def build_plan(layout, timesteps: list[float]):
    """Per-forward row-timestep table for a refine: video steps, audio and keyframe stay pinned."""
    per_step = []
    for value in timesteps:
        distinct, inverse = build_row_timesteps(
            layout, float(value), CLEAN_TIMESTEP, max(float(value), KEYFRAME_NOISE_AUG), 1.0
        )
        per_step.append((np.array(distinct), np.array(inverse)))
    table = sorted({float(value) for distinct, _ in per_step for value in distinct})
    lookup = {value: index for index, value in enumerate(table)}
    plan = []
    for distinct, inverse in per_step:
        remap = np.array([lookup[float(value)] for value in distinct], dtype=np.int32)
        plan.append(mx.array(remap[inverse].astype(np.int32)))
    return mx.array(np.array(table, dtype=np.float32)), plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-cache", type=Path, required=True)
    parser.add_argument("--dit", type=Path, required=True)
    parser.add_argument("--compact-root", type=Path, required=True)
    parser.add_argument(
        "--first-frame",
        type=Path,
        default=None,
        help="The still Stage A was conditioned on. It is re-encoded at the refine canvas, so the "
        "keyframe conditioning is native to the target grid instead of an upsampled small-canvas "
        "latent. Required unless --refine-forwards is 0.",
    )
    parser.add_argument("--refine-width", type=int, default=None)
    parser.add_argument("--refine-height", type=int, default=None)
    parser.add_argument(
        "--refine-sigma",
        type=float,
        default=0.55,
        help="Re-entry noise level. The tail has to carry enough sigma to dissolve the "
        "interpolation structure the upsample introduces.",
    )
    parser.add_argument(
        "--refine-forwards",
        type=int,
        default=2,
        help="Transformer forwards in the tail. 0 decodes the cached Stage-A latents untouched — "
        "the cache round-trip check.",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--wired-gb", type=float, default=50.0)
    parser.add_argument("--memory-gb", type=float, default=58.0)
    args = parser.parse_args()

    meta, arrays = load_stage_a(args.stage_a_cache)
    stage_h, stage_w = int(meta["height"]), int(meta["width"])
    height = args.refine_height or stage_h
    width = args.refine_width or stage_w
    if height % 32 or width % 32:
        parser.error("--refine-height/--refine-width must be multiples of 32 (even latent grid)")
    if args.refine_forwards < 0:
        parser.error("--refine-forwards cannot be negative")
    if args.refine_forwards > 0 and args.first_frame is None:
        parser.error("--first-frame is required for a refine; the keyframe is rebuilt at target")
    if args.refine_forwards == 0 and (height, width) != (stage_h, stage_w):
        parser.error("--refine-forwards 0 decodes the cache as-is; it cannot change the canvas")

    config = PipelineConfig()
    frames = int(meta["frames"])
    latent_frames = int(meta["latent_frames"])
    audio_latents = int(meta["audio_latents"])
    channels = int(meta["latents_dim"])
    patch = tuple(int(value) for value in meta["patch"])
    latent_h, latent_w = height // 16, width // 16
    sigmas = (
        refine_sigmas(args.refine_sigma, args.refine_forwards, config.sigma_shift_video)
        if args.refine_forwards
        else [1.0, 0.0]
    )
    timesteps = [float(np.float32(1.0) - np.float32(value)) for value in sigmas[:-1]]

    device = mx.device_info()
    wired_bytes = min(
        int(args.wired_gb * 1024**3), int(device["max_recommended_working_set_size"]) - 1024**2
    )
    memory_bytes = min(int(args.memory_gb * 1024**3), int(device["memory_size"]) - 1024**3)
    settings = {
        "stage_a_cache": str(args.stage_a_cache),
        "stage_a": {
            key: meta[key]
            for key in ("prompt", "seed", "frames", "height", "width", "forwards", "first_frame")
        },
        "stage_a_digests": meta["digests"],
        "refine": {
            "width": width,
            "height": height,
            "sigma": args.refine_sigma,
            "forwards": args.refine_forwards,
            "sigmas": [round(value, 6) for value in sigmas],
            "interpolation": "bilinear",
            "audio": "frozen",
            "keyframe": "re-encoded at target canvas",
        },
        "dit": str(args.dit),
        "compact_root": str(args.compact_root),
        "wired_gb": round(gb(wired_bytes), 3),
        "memory_limit_gb": round(gb(memory_bytes), 3),
        "device": device,
    }
    record = Recorder(args.metrics, settings)
    started_total = time.perf_counter()
    mx.set_wired_limit(wired_bytes)
    mx.set_memory_limit(memory_bytes)

    print(
        f"stage-A cache verified: {stage_w}x{stage_h}, {meta['forwards']} forwards, "
        f"seed {meta['seed']}\n"
        f"  video_rows {arrays['video_rows'].shape} sha256 {meta['digests']['video_rows'][:16]}\n"
        f"  audio_rows {arrays['audio_rows'].shape} sha256 {meta['digests']['audio_rows'][:16]}\n"
        f"  embeds     {arrays['embeds'].shape} sha256 {meta['digests']['embeds'][:16]}",
        flush=True,
    )

    try:
        text_tags = arrays["text_tags"]
        audio_rows = mx.array(arrays["audio_rows"])

        with record.phase("stage_a_upsample"):
            stage_latents = np.array(
                unpatchify_video_tokens(
                    mx.array(arrays["video_rows"]),
                    latent_frames,
                    stage_h // 16,
                    stage_w // 16,
                    channels,
                    patch,
                )
            )
            resized = resize_latent_bilinear(stage_latents, latent_h, latent_w)
            clean_rows = patchify_video_latents(mx.array(resized), patch)
            mx.eval(clean_rows)
            print(
                f"latent grid {stage_h//16}x{stage_w//16} -> {latent_h}x{latent_w} "
                f"({latent_w / (stage_w // 16):.3f}x), rows {clean_rows.shape}",
                flush=True,
            )

        if args.refine_forwards:
            with record.phase("refine_keyframe_encode"):
                from PIL import Image

                from minimax_h3_mlx.packing import prepare_keyframe_image

                keyframe = prepare_keyframe_image(
                    Image.open(args.first_frame).convert("RGB"), height, width, stretch=True
                )
                keyframe_vae = load_compact_video_vae(args.compact_root / "video_vae.safetensors")
                condition_rows = encode_keyframe_rows(
                    keyframe_vae, [keyframe], height, width, patch
                )
                mx.eval(condition_rows)
                print(f"keyframe conditioning rows: {condition_rows.shape}")
            del keyframe_vae
            release()

            with record.phase("refine_dit_load_bf16"):
                dit = load_dit(args.dit, verbose=True)
                patch = dit.config.patch_size

            layout = build_packed_sequence(
                text_tags, latent_frames, latent_h, latent_w, audio_latents, patch, ("first",)
            )
            n_cond_v = layout.num_condition_video_rows
            # The community node's headline failure mode, asserted rather than trusted: keyframe
            # rows that do not match the target layout's conditioning-row count mean the RoPE grid
            # and the tensor disagree.
            if condition_rows.shape[0] != n_cond_v:
                raise ValueError(
                    f"keyframe rows {condition_rows.shape[0]} != target layout conditioning rows "
                    f"{n_cond_v}; the keyframe is not on the refine grid"
                )
            if clean_rows.shape[0] != layout.video_indices.shape[0] - n_cond_v:
                raise ValueError("upsampled video rows do not fill the target layout")
            print(
                f"refine geometry: {width}x{height}, {layout.sequence_length:,} packed rows; "
                f"sigma {args.refine_sigma}, {args.refine_forwards} forwards",
                flush=True,
            )
            record.data.update(
                {
                    "packed_rows": int(layout.sequence_length),
                    "condition_rows": int(n_cond_v),
                    "refine_sigmas": [round(value, 6) for value in sigmas],
                }
            )
            record.flush()

            with record.phase("refine_adaln_cache_and_noise"):
                timestep_table, plan = build_plan(layout, timesteps)
                cache = ModulationCache.build(dit, timestep_table, dtype=mx.bfloat16)
                mx.eval(cache.tables)
                freed = drop_adaln_weights(dit)
                mx.clear_cache()
                print(f"AdaLN cache {cache.nbytes()/1024**2:.1f} MiB; dropped {gb(freed):.2f} GiB")
                # Same draw order as a fresh render — conditioning noise, then video — so the run
                # is reproducible from (cache, seed) alone.
                mx.random.seed(int(meta["seed"]))
                condition_noise = mx.random.normal(condition_rows.shape).astype(mx.float32)
                condition_rows = MiniMaxH3Scheduler(shift=config.sigma_shift_video).scale_noise(
                    condition_rows, KEYFRAME_NOISE_AUG, condition_noise
                )
                noise = mx.random.normal(clean_rows.shape).astype(mx.float32)
                video_rows = MiniMaxH3Scheduler(shift=config.sigma_shift_video).scale_noise(
                    clean_rows, timesteps[0], noise
                )
                video_rows = mx.nan_to_num(video_rows)
                video_rows = mx.concatenate([condition_rows, video_rows])
                mx.eval(video_rows, audio_rows)

            video_sched = MiniMaxH3Scheduler(shift=config.sigma_shift_video)
            video_sched.set_timesteps(sigmas=sigmas)
            embeds = mx.array(arrays["embeds"]).astype(mx.bfloat16)
            mx.eval(embeds)
            step_times = []
            with record.phase("refine_joint_denoise"):
                for index, timestep in enumerate(video_sched.timesteps.tolist()):
                    step_started = time.perf_counter()
                    video_out, audio_out = dit(
                        video_rows[None].astype(mx.bfloat16),
                        audio_rows[None].astype(mx.bfloat16),
                        embeds,
                        timestep_table,
                        plan[index],
                        layout.token_tags,
                        layout.position_ids,
                        layout.video_indices,
                        layout.audio_indices,
                        layout.text_indices,
                        modulation_cache=cache,
                    )
                    video_pred = video_out[0].astype(mx.float32)
                    # audio_out is discarded on purpose: audio_denoise = 0. The clean Stage-A audio
                    # is the delivered audio, so a refine can never garble speech.
                    del audio_out
                    stepped = video_sched.step(
                        video_pred[n_cond_v:], float(timestep), video_rows[n_cond_v:]
                    )
                    video_rows = mx.concatenate([video_rows[:n_cond_v], stepped])
                    mx.eval(video_rows)
                    elapsed = time.perf_counter() - step_started
                    step_times.append(elapsed)
                    print(
                        f"refine step {index+1}/{args.refine_forwards}: {elapsed:.1f}s "
                        f"(sigma {sigmas[index]:.4f} -> {sigmas[index+1]:.4f}, "
                        f"active {gb(mx.get_active_memory()):.1f} GiB)",
                        flush=True,
                    )
            record.data["refine_step_seconds"] = [round(value, 3) for value in step_times]
            record.data["mean_refine_step_seconds"] = round(float(np.mean(step_times)), 3)
            generated = video_rows[n_cond_v:]
            del dit, cache, plan, timestep_table, layout, video_rows
        else:
            print("refine forwards = 0: decoding the cached Stage-A latents untouched", flush=True)
            generated = clean_rows
        release()

        with record.phase("video_vae_decode"):
            video_vae = load_compact_video_vae(args.compact_root / "video_vae.safetensors")
            video = decode_video(
                video_vae, generated, frames, latent_frames, latent_h, latent_w, patch
            )
        del video_vae, generated
        release()

        with record.phase("audio_vae_decode"):
            audio_vae = load_compact_audio_vae(args.compact_root / "audio_vae.safetensors")
            audio = decode_audio(audio_vae, audio_rows, audio_latents, frames)
            sample_rate = int(audio_vae.config.sampling_rate)
        del audio_vae, audio_rows
        release()

        with record.phase("encode_mux"):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            save_mp4(args.output, video, float(FPS), audio, sample_rate, crf=args.crf)
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
                "total_seconds": round(time.perf_counter() - started_total, 3),
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
                "total_seconds": round(time.perf_counter() - started_total, 3),
            }
        )
        record.flush()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
