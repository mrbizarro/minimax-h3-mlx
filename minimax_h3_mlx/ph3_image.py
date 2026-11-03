"""P-H3-Image text-to-image on the MiniMax-H3 DiT, in MLX.

P-H3-Image is MiniMax-H3's FL2VA transformer plus a rank-128 adapter (``p-h3-image_lora``), run as
**one latent frame and two throw-away audio latents** over a fixed 8-forward grid with no guidance.
Everything around the DiT is new on the Mac: a Qwen3-VL-4B text encoder with a five-tap connector
(:mod:`ph3_image_text`) and a single-frame ViT decoder (:mod:`ph3_image_vae`).

``denoise_image`` is the part that needs no checkpoints beyond the DiT, so it is unit-testable with
a tiny DiT. ``PH3ImageRunner`` is the staged, memory-aware driver the Studio calls: text encoder
out before the DiT comes in, DiT out before the decoder comes in.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .adaln import ModulationCache, drop_adaln_weights
from .packing import (
    AUDIO_CHANNELS,
    build_packed_sequence,
    build_row_timesteps,
    patchify_video_latents,
    unpatchify_video_tokens,
)
from .scheduler import MiniMaxH3Scheduler

# The checkpoint's own sampling grid (diffusers ``sample_sigmas``), used as given: the scheduler's
# ``shift_provided_sigmas`` is false, and ComfyUI loads it with shift 1.0.
SAMPLE_SIGMAS = (1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0)
# Our ``--steps`` counts sigma POINTS; the number of forwards is one less.
STEPS = len(SAMPLE_SIGMAS)
FORWARDS = STEPS - 1

# Adapter layout, decided against the BASE weights (a remote probe of blocks.0 of Pruna's published
# transformer against our pruned FL2VA, 3 Nov 2026): delta = W_pruna - W_base is explained by B@A
# at scale 1.000 only with the qkv rows permuted (contiguous q|k|v -> our per-head interleave:
# cos +0.58 vs +0.005 as-is) and the fc1 rows AS-IS (cos +0.50 vs -0.002 swapped). The cosine is
# bounded by bf16 rounding of the published weights, the scalar is not. fc2 is as-is too.
ADAPTER_QKV_PERMUTE = True
ADAPTER_FC1_SWAP = False

NUM_AUDIO_LATENTS = 2  # per channel, discarded
CANVAS_MULTIPLE = 32  # 16x VAE x 2x2 patch

# The 8 MP sizes the model is recommended at (width, height).
SIZES_8MP = {
    "1:1": (2912, 2912), "4:3": (3360, 2496), "3:4": (2496, 3360), "3:2": (3552, 2368),
    "2:3": (2368, 3552), "16:9": (3872, 2176), "9:16": (2176, 3872),
}


def round_canvas(height: int, width: int) -> tuple[int, int]:
    m = CANVAS_MULTIPLE
    return max(m, round(height / m) * m), max(m, round(width / m) * m)


def size_for_megapixels(aspect: str, megapixels: float) -> tuple[int, int]:
    """``(width, height)`` for an aspect ratio at about ``megapixels`` (multiples of 32)."""
    if abs(megapixels - 8.0) < 1e-6 and aspect in SIZES_8MP:
        return SIZES_8MP[aspect]
    aw, ah = (float(x) for x in aspect.split(":"))
    ratio = aw / ah
    pixels = megapixels * 1024 * 1024
    width = (pixels * ratio) ** 0.5
    height = width / ratio
    h, w = round_canvas(int(round(height)), int(round(width)))
    return w, h


def denoise_image(
    dit,
    cond: mx.array,
    text_tags: np.ndarray,
    height: int,
    width: int,
    seed: int = 0,
    sigmas=SAMPLE_SIGMAS,
    latent_channels: int = 24,
    ratio: int = 16,
    drop_adaln: bool = True,
    verbose: bool = True,
    on_step=None,
) -> mx.array:
    """Run the 8-forward loop. Returns normalised latents ``(1, C, 1, H', W')`` (float32)."""
    height, width = round_canvas(height, width)
    lh, lw = height // ratio, width // ratio
    patch = tuple(dit.config.patch_size)
    layout = build_packed_sequence(text_tags, 1, lh, lw, NUM_AUDIO_LATENTS, patch, ())
    if verbose:
        print(f"canvas {width}x{height}: {layout.sequence_length:,} rows "
              f"({len(text_tags):,} text, {(lh // patch[1]) * (lw // patch[2]):,} image)", flush=True)

    mx.random.seed(seed)
    video_rows = patchify_video_latents(
        mx.random.normal((1, latent_channels, 1, lh, lw)).astype(mx.float32), patch
    )
    audio_rows = mx.random.normal(
        (NUM_AUDIO_LATENTS * AUDIO_CHANNELS, dit.config.audio_latents_dim)
    ).astype(mx.float32)

    # One grid for both modalities, used as given (no shift).
    video_sched, audio_sched = MiniMaxH3Scheduler(), MiniMaxH3Scheduler()
    video_sched.set_timesteps(sigmas=list(sigmas))
    audio_sched.set_timesteps(sigmas=list(sigmas))

    per_step = []
    for t, at in zip(video_sched.timesteps.tolist(), audio_sched.timesteps.tolist()):
        distinct, inverse = build_row_timesteps(layout, float(t), float(at), 0.999, 1.0)
        per_step.append((np.array(distinct), np.array(inverse)))
    table = sorted({float(v) for d, _ in per_step for v in d})
    lookup = {v: i for i, v in enumerate(table)}
    plan = [mx.array(np.array([lookup[float(v)] for v in d], dtype=np.int32)[inv].astype(np.int32))
            for d, inv in per_step]
    timestep_table = mx.array(np.array(table, dtype=np.float32))

    cache = ModulationCache.build(dit, timestep_table, dtype=mx.bfloat16)
    if drop_adaln:
        drop_adaln_weights(dit)
        mx.eval(dit.parameters())

    embeds = cond.astype(mx.bfloat16)
    times = []
    for i, t in enumerate(video_sched.timesteps.tolist()):
        started = time.perf_counter()
        video_pred, audio_pred = dit(
            video_rows[None].astype(mx.bfloat16), audio_rows[None].astype(mx.bfloat16), embeds,
            timestep_table, plan[i], layout.token_tags, layout.position_ids,
            layout.video_indices, layout.audio_indices, layout.text_indices,
            modulation_cache=cache,
        )
        video_rows = video_sched.step(video_pred[0].astype(mx.float32), float(t), video_rows)
        audio_rows = audio_sched.step(
            audio_pred[0].astype(mx.float32), float(audio_sched.timesteps[i].item()), audio_rows)
        mx.eval(video_rows, audio_rows)
        times.append(time.perf_counter() - started)
        if on_step:
            on_step(i + 1, len(video_sched.timesteps), times[-1])
        if verbose:
            mean = sum(times) / len(times)
            print(f"  step {i + 1}/{len(video_sched.timesteps)}  {times[-1]:.1f}s  "
                  f"eta {mean * (len(video_sched.timesteps) - i - 1) / 60:.1f} min", flush=True)
    denoise_image.last_step_seconds = times
    return unpatchify_video_tokens(video_rows, 1, lh, lw, latent_channels, patch)


class PH3ImageRunner:
    """Staged driver: encode -> (free encoder) -> DiT + adapter -> (free DiT) -> decode."""

    def __init__(self, root: str | Path, dit_path: str | Path, lora_path: str | Path | None,
                 lora_scale: float = 1.0, lora_mode: str = "runtime"):
        self.root, self.dit_path = Path(root), Path(dit_path)
        self.lora_path, self.lora_scale, self.lora_mode = lora_path, lora_scale, lora_mode
        self.metrics: dict = {}

    def generate(self, prompt: str, width: int, height: int, seed: int = 0,
                 verbose: bool = True, drop_adaln: bool = True) -> np.ndarray:
        import json

        from .lora import apply_loras, parse_spec
        from .load import load_dit
        from .ph3_image_text import PH3ImageTextEncoder
        from .ph3_image_vae import decode_to_uint8, load_image_vae

        m = self.metrics
        t0 = time.perf_counter()

        started = time.perf_counter()
        encoder = PH3ImageTextEncoder(self.root, load_vision=False, verbose=False)
        cond, tags = encoder.encode(prompt)
        mx.eval(cond)
        m["text_seconds"] = time.perf_counter() - started
        m["text_rows"] = int(cond.shape[1])
        del encoder
        gc.collect()
        mx.clear_cache()

        started = time.perf_counter()
        dit = load_dit(self.dit_path, verbose=verbose)
        m["dit_load_seconds"] = time.perf_counter() - started
        accounting = None
        if self.lora_path:
            reports = apply_loras(dit, [(Path(self.lora_path), float(self.lora_scale))],
                                  mode=self.lora_mode, verbose=verbose,
                                  permute_qkv=ADAPTER_QKV_PERMUTE, swap_fc1=ADAPTER_FC1_SWAP)
            accounting = reports[0].summary()
            m["lora"] = accounting
        started = time.perf_counter()
        latents = denoise_image(dit, cond, tags, height, width, seed=seed, drop_adaln=drop_adaln, verbose=verbose)
        mx.eval(latents)
        m["denoise_seconds"] = time.perf_counter() - started
        m["step_seconds"] = list(getattr(denoise_image, "last_step_seconds", []))
        del dit
        gc.collect()
        mx.clear_cache()

        started = time.perf_counter()
        vae = load_image_vae(self.root / "vae" / "decoder.safetensors",
                             json.loads((self.root / "vae" / "config.json").read_text()))
        image = decode_to_uint8(vae, latents)
        m["decode_seconds"] = time.perf_counter() - started
        m["total_seconds"] = time.perf_counter() - t0
        m["peak_gb"] = mx.get_peak_memory() / 1e9
        return image
