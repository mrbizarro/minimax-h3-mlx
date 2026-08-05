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


def decode_audio(model, rows, audio_latents: int, frames: int):
    cfg = model.config
    latents = unpack_audio_tokens(rows, audio_latents)
    mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1)
    std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1)
    waveform = np.array(model.decode((latents * std + mean).astype(mx.float32)))[:, 0, :]
    samples = round(frames / FPS * cfg.sampling_rate)
    return waveform[:, :samples].astype(np.float32)


CHAIN_PROMPT_SEPARATOR = " ||| "


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
):
    """Render one window end to end and hand back decoded pixels, audio and the sample rate.

    Every large model is loaded, used and released inside this call, so a chain of windows runs in
    one process without the previous window's DiT still being resident. ``label`` prefixes the
    phase names so a chained metrics file stays readable. ``prompt`` is this window's own text —
    a chain that scripts a line in one window must not encode that line in the others.
    """
    keyframes = [keyframe] if keyframe is not None else None

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
    digest = hashlib.sha256(np.array(embeds.astype(mx.float32)).tobytes()).hexdigest()[:16]
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
        with record.phase(f"{label}keyframe_encode"):
            keyframe_vae = load_compact_video_vae(args.compact_root / "video_vae.safetensors")
            condition_rows = encode_keyframe_rows(
                keyframe_vae, keyframes, args.height, args.width, patch
            )
            mx.eval(condition_rows)
            print(f"keyframe conditioning rows: {condition_rows.shape}")
        del keyframe_vae
        release()

    with record.phase(f"{label}dit_load_bf16"):
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
    record.data[f"{label}packed_rows"] = int(layout.sequence_length)
    record.data[f"{label}video_rows"] = int(layout.video_indices.shape[0])
    record.data[f"{label}audio_rows"] = int(layout.audio_indices.shape[0])
    record.flush()

    with record.phase(f"{label}adaln_cache_and_noise"):
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
        mx.random.seed(seed)
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
        StepResidualCache(args.step_cache, num_forwards, max_skip=args.step_cache_max_skip)
        if args.step_cache > 0 or args.step_cache_probe
        else None
    )

    step_times = []
    with record.phase(f"{label}joint_denoise"):
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
    if args.chain_windows < 1:
        parser.error("--chain-windows must be at least 1")
    if args.chain_windows > 1 and args.playback_fps is not None:
        parser.error("--chain-windows and --playback-fps do not compose; the seam maths is 24 fps")
    if args.chain_windows > 1 and args.save_stage_a is not None:
        parser.error(
            "--save-stage-a holds one window's latents; with a chain every window would overwrite "
            "the last. Cache a single-window render."
        )
    frames = align_num_frames(args.frames)
    chain = args.chain_windows
    chain_frames = frames + (chain - 1) * (frames - 1)
    if args.chain_total_frames is not None and not 1 <= args.chain_total_frames <= chain_frames:
        parser.error(f"--chain-total-frames must be in 1..{chain_frames} for this chain")
    delivered_frames = args.chain_total_frames or chain_frames

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
        "save_stage_a": str(args.save_stage_a) if args.save_stage_a else None,
        "frames": frames,
        "height": args.height,
        "width": args.width,
        "sigma_points": args.steps,
        "forwards": args.steps - 1,
        "seed": args.seed,
        "step_cache_threshold": args.step_cache,
        "step_cache_max_skip": args.step_cache_max_skip,
        "playback_fps": args.playback_fps or FPS,
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
            save_mp4(
                args.output,
                video,
                playback,
                audio,
                sample_rate,
                crf=args.crf,
                audio_tempo=playback / FPS,
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
