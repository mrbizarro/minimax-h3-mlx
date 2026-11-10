"""MiniMax-H3-Image text-to-image on the MiniMax-H3 DiT, in MLX.

MiniMax-H3-Image is the H3 family's own image model: the 33B H3 DiT with the audio branch removed,
the unchanged H3 text encoder (Qwen3-VL-32B, ``hidden_states[50]``), 6 forwards and no guidance.
The student was distilled in intervals ("PDD"); its release swaps the DiT's output layer for **six
fused heads**, one per forward, which :func:`denoise_h3_image` selects with ``dit.pdd_step``.

Packed layout: ``[text | target image]`` (references go between the two for edits). There are no
audio rows at all. The per-block AdaLN projections are not in the pack: their output for the six
timesteps (plus the 0.999 reference level) is a small precomputed table
(``dit/adaln_cache.safetensors``, built by ``scripts/fetch_h3_image.py`` from the UNQUANTISED
projections), exactly the trick :mod:`adaln` already plays for the other H3 builds.

Written from the publisher's inference code (SGLang ``minimax_h3_image``); nothing is copied.
"""

from __future__ import annotations

import gc
import json
import time
from math import sqrt
from pathlib import Path

import mlx.core as mx
import numpy as np

from .adaln import load_modulation_cache
from .packing import build_packed_sequence, build_row_timesteps, patchify_video_latents, unpatchify_video_tokens
from .scheduler import MiniMaxH3Scheduler

# pdd_config.json of the 6-step release (``sigmas``): used as given, no shift.
SAMPLE_SIGMAS = (1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724,
                 0.23200887441635132, 0.0)
# Our ``--steps`` counts sigma POINTS; the number of forwards is one less (the heads index it).
STEPS = len(SAMPLE_SIGMAS)
FORWARDS = STEPS - 1
REFERENCE_LEVEL = 0.999  # MINIMAX_H3_IMGVID_COND_TIMESTEP: references are held at 0.999 clean
CANVAS_MULTIPLE = 32  # 16x VAE x 2x2 patch
DEFAULT_MIN_GENERATION_AREA = 2048 * 2048
MIN_AREA_RANGE = (32 * 32, 4096 * 4096)

# Common ~16 MP sizes from the model card (width, height).
SIZES_16MP = {
    "1:1": (4096, 4096), "4:3": (4736, 3552), "3:4": (3552, 4736), "3:2": (5024, 3360),
    "2:3": (3360, 5024), "16:9": (5472, 3072), "9:16": (3072, 5472), "21:9": (6272, 2688),
}


def _nearest_multiple(value: float) -> int:
    # Python round: ties to even, like the reference.
    return max(CANVAS_MULTIPLE, int(round(value / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)


def generation_canvas(width: int, height: int, min_area: int | None = DEFAULT_MIN_GENERATION_AREA) -> tuple[int, int]:
    """The 32 px-aligned ``(width, height)`` the DiT generates for a request.

    A request smaller than ``min_area`` is generated on a larger canvas with the same aspect ratio
    (both sides scale by sqrt(area / (w*h)), each rounded to the nearest 32 px) and resized back by
    :func:`resize_output`. ``min_area=None`` turns the canvas off: sizes must then be 32-aligned.
    """
    if min_area is None:
        if width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
            raise ValueError("without a minimum generation area, width and height must be multiples of 32")
        return width, height
    if not MIN_AREA_RANGE[0] <= min_area <= MIN_AREA_RANGE[1]:
        raise ValueError(f"min_generation_area must be within {MIN_AREA_RANGE}")
    scale = max(1.0, sqrt(min_area / (width * height)))
    return _nearest_multiple(width * scale), _nearest_multiple(height * scale)


def resize_output(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a decoded uint8 ``(H, W, 3)`` image back to the requested size (PIL LANCZOS, 8-bit)."""
    if image.shape[1] == width and image.shape[0] == height:
        return image
    from PIL import Image

    return np.array(Image.fromarray(image).resize((width, height), Image.Resampling.LANCZOS))


def size_for_megapixels(aspect: str, megapixels: float) -> tuple[int, int]:
    """``(width, height)`` for an aspect ratio at about ``megapixels`` (multiples of 32)."""
    if abs(megapixels - 16.0) < 1e-6 and aspect in SIZES_16MP:
        return SIZES_16MP[aspect]
    aw, ah = (float(x) for x in aspect.split(":"))
    ratio = aw / ah
    pixels = megapixels * 1024 * 1024
    width = (pixels * ratio) ** 0.5
    return _nearest_multiple(width), _nearest_multiple(width / ratio)


def load_pdd_heads(path: str | Path) -> tuple[mx.array, mx.array]:
    """``(weight [6, 96, 5376], bias [6, 96])`` float32. Anything else is not this release."""
    raw = mx.load(str(path))
    if set(raw) != {"video_out.weight", "video_out.bias"}:
        raise ValueError(f"PDD heads must hold exactly video_out.weight/bias, got {sorted(raw)}")
    w, b = raw["video_out.weight"].astype(mx.float32), raw["video_out.bias"].astype(mx.float32)
    if w.shape[0] != FORWARDS or w.shape[:2] != b.shape:
        raise ValueError(f"PDD heads: expected {FORWARDS} heads, weight {w.shape}, bias {b.shape}")
    return w, b


def reference_timesteps(sigmas=SAMPLE_SIGMAS) -> list[float]:
    """The distinct timesteps the DiT will see: ``1 - sigma`` per forward (float32) + the 0.999 level.

    This is the table the AdaLN cache must have been built for.
    """
    sched = MiniMaxH3Scheduler()
    sched.set_timesteps(sigmas=list(sigmas))
    return sorted({float(v) for v in sched.timesteps.tolist()} | {float(np.float32(REFERENCE_LEVEL))})


def denoise_h3_image(
    dit,
    cond: mx.array,
    text_tags: np.ndarray,
    height: int,
    width: int,
    modulation_cache,
    heads: tuple[mx.array, mx.array],
    seed: int = 0,
    sigmas=SAMPLE_SIGMAS,
    latent_channels: int = 24,
    ratio: int = 16,
    verbose: bool = True,
    on_step=None,
) -> mx.array:
    """Run the 6-forward loop. Returns normalised latents ``(1, C, 1, H', W')`` (float32).

    ``height``/``width`` must already be a 32-aligned canvas (see :func:`generation_canvas`).
    """
    if len(sigmas) - 1 != heads[0].shape[0]:
        raise ValueError(f"{len(sigmas) - 1} forwards but {heads[0].shape[0]} fused heads: the heads are per step")
    if height % CANVAS_MULTIPLE or width % CANVAS_MULTIPLE:
        raise ValueError("canvas must be a multiple of 32")
    lh, lw = height // ratio, width // ratio
    patch = tuple(dit.config.patch_size)
    layout = build_packed_sequence(text_tags, 1, lh, lw, 0, patch, ())
    if verbose:
        print(f"canvas {width}x{height}: {layout.sequence_length:,} rows "
              f"({len(text_tags):,} text, {(lh // patch[1]) * (lw // patch[2]):,} image)", flush=True)

    mx.random.seed(seed)
    video_rows = patchify_video_latents(
        mx.random.normal((1, latent_channels, 1, lh, lw)).astype(mx.float32), patch
    )
    audio_rows = mx.zeros((0, dit.config.audio_latents_dim), dtype=mx.float32)

    sched = MiniMaxH3Scheduler()
    sched.set_timesteps(sigmas=list(sigmas))
    table = [float(v) for v in modulation_cache.timesteps.tolist()]
    lookup = {v: i for i, v in enumerate(table)}
    plan = []
    for t in sched.timesteps.tolist():
        distinct, inverse = build_row_timesteps(layout, float(t), float(t), float(np.float32(REFERENCE_LEVEL)), 1.0)
        try:
            idx = np.array([lookup[float(v)] for v in np.array(distinct)], dtype=np.int32)
        except KeyError as exc:
            raise ValueError(f"timestep {exc} is not in the AdaLN cache {table}: wrong cache for this schedule") from exc
        plan.append(mx.array(idx[np.array(inverse)].astype(np.int32)))
    timestep_table = modulation_cache.timesteps

    dit.pdd_heads = heads
    embeds = cond.astype(mx.bfloat16)
    times = []
    try:
        for i, t in enumerate(sched.timesteps.tolist()):
            started = time.perf_counter()
            dit.pdd_step = i
            video_pred, _ = dit(
                video_rows[None].astype(mx.bfloat16), audio_rows[None].astype(mx.bfloat16), embeds,
                timestep_table, plan[i], layout.token_tags, layout.position_ids,
                layout.video_indices, layout.audio_indices, layout.text_indices,
                modulation_cache=modulation_cache,
            )
            video_rows = sched.step(video_pred[0].astype(mx.float32), float(t), video_rows)
            mx.eval(video_rows)
            times.append(time.perf_counter() - started)
            if on_step:
                on_step(i + 1, FORWARDS, times[-1])
            if verbose:
                mean = sum(times) / len(times)
                print(f"  step {i + 1}/{FORWARDS}  {times[-1]:.1f}s  eta {mean * (FORWARDS - i - 1) / 60:.1f} min", flush=True)
    finally:
        dit.pdd_heads = None
    denoise_h3_image.last_step_seconds = times
    return unpatchify_video_tokens(video_rows, 1, lh, lw, latent_channels, patch)


class H3ImageRunner:
    """Staged driver: text encoder out before the DiT comes in, DiT out before the decoder.

    ``root`` is the folder ``scripts/fetch_h3_image.py`` builds (``dit/``, ``vae/``,
    ``pdd_fused_heads.safetensors``); ``text_dir`` is the H3 text-encoder folder the other H3 modes
    already use (the Q8 ``ddalcu-q8`` folder) and ``text_config`` the upstream ``text_encoder/config.json``.
    """

    def __init__(self, root: str | Path, text_dir: str | Path, text_config: str | Path | None = None,
                 compile_blocks: bool = False):
        self.root, self.text_dir = Path(root), Path(text_dir)
        self.text_config = Path(text_config) if text_config else None
        self.compile_blocks = compile_blocks
        self.metrics: dict = {}

    def close(self):
        gc.collect()
        mx.clear_cache()

    def generate(self, prompt: str, width: int, height: int, seed: int = 0, verbose: bool = True,
                 min_generation_area: int | None = DEFAULT_MIN_GENERATION_AREA) -> np.ndarray:
        from .h3_image_vae import decode_to_uint8, load_h3_image_vae
        from .load import load_dit
        from .text_encoder import MiniMaxH3TextEncoder

        m = self.metrics = {"compile_blocks": self.compile_blocks}
        t0 = time.perf_counter()
        cw, ch = generation_canvas(width, height, min_generation_area)
        m["canvas"] = [cw, ch]

        started = time.perf_counter()
        encoder = MiniMaxH3TextEncoder(self.text_dir, load_vision=False, verbose=False, config_path=self.text_config)
        cond, tags = encoder.encode(prompt)
        mx.eval(cond)
        m["text_seconds"], m["text_rows"] = time.perf_counter() - started, int(cond.shape[1])
        del encoder
        gc.collect()
        mx.clear_cache()

        started = time.perf_counter()
        dit = load_dit(self.root / "dit", verbose=verbose)
        cache = load_modulation_cache(self.root / "dit" / "adaln_cache.safetensors")
        heads = load_pdd_heads(self.root / "pdd_fused_heads.safetensors")
        dit.compile_image_blocks = self.compile_blocks
        m["dit_load_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        latents = denoise_h3_image(dit, cond, tags, ch, cw, cache, heads, seed=seed, verbose=verbose)
        mx.eval(latents)
        m["denoise_seconds"] = time.perf_counter() - started
        m["step_seconds"] = list(getattr(denoise_h3_image, "last_step_seconds", []))
        m["forwards"] = len(m["step_seconds"])
        del dit, cache
        gc.collect()
        mx.clear_cache()

        started = time.perf_counter()
        vae = load_h3_image_vae(self.root / "vae" / "diffusion_pytorch_model.safetensors",
                                json.loads((self.root / "vae" / "config.json").read_text()))
        image = resize_output(decode_to_uint8(vae, latents), width, height)
        m["decode_seconds"] = time.perf_counter() - started
        m["total_seconds"] = time.perf_counter() - t0
        m["peak_gb"] = mx.get_peak_memory() / 1e9
        return image
