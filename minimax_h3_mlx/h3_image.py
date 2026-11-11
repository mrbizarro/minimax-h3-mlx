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
from .packing import (PIXEL_MEAN, PIXEL_STD, build_packed_sequence, build_row_timesteps, patchify_video_latents,
                      unpatchify_video_tokens)
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


# Edit (image release): 1..16 reference pictures, Picture 1 is the one being edited.
MAX_REFERENCES = 16
EDIT_LABEL = "Picture {n}: "            # minimax_h3_image_presentation: no angle brackets
QWEN_SHORT_EDGE = (1024, 1024)          # edit_reference_qwen_{min,max}_short_edge
VAE_SHORT_EDGE = (1024, 2048)           # edit_reference_vae_{min,max}_short_edge


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


def bounded_reference_size(width: int, height: int, low: int, high: int) -> tuple[int, int] | None:
    """Where an edit reference is resized to for one consumer (Qwen or the VAE), or None.

    A short edge inside ``[low, high]`` keeps the original pixels (None). Otherwise the picture is
    scaled, ratio kept, so its short edge lands on the nearer bound, each side rounded to 32 px
    (``minimax_h3_image_reference_size``).
    """
    short = min(width, height)
    if low <= short <= high:
        return None
    scale = (low if short < low else high) / short
    return _nearest_multiple(width * scale), _nearest_multiple(height * scale)


def bound_reference(image, low: int, high: int):
    """``image`` (PIL RGB) resized per :func:`bounded_reference_size` (itself when inside)."""
    from PIL import Image

    size = bounded_reference_size(image.size[0], image.size[1], low, high)
    return image if size is None else image.resize(size, Image.Resampling.LANCZOS)


def reference_grid(width: int, height: int) -> tuple[int, int]:
    """``(latent_h, latent_w)`` of an edit reference: 16 px cells, an even count of each."""
    return int(height) // 16 // 2 * 2, int(width) // 16 // 2 * 2


def load_reference(path) -> "object":
    """An edit reference read upright (EXIF orientation applied), RGB, first frame only."""
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    if min(image.size) < 32:
        raise ValueError(f"a reference picture must be at least 32 px on each side, got {image.size[0]}x{image.size[1]}")
    return image


def vae_reference(image):
    """The VAE's copy of a reference: bounded to a 1024-2048 px short edge, then cropped top-left to
    an even 16 px grid. Returns ``(pixels, (latent_h, latent_w))``."""
    bounded = bound_reference(image, *VAE_SHORT_EDGE)
    gh, gw = reference_grid(*bounded.size)
    if (gw * 16, gh * 16) != bounded.size:
        bounded = bounded.crop((0, 0, gw * 16, gh * 16))
    return bounded, (gh, gw)


def size_matching(image, megapixels: float) -> tuple[int, int]:
    """``(width, height)`` at about ``megapixels`` in the shape of ``image`` (multiples of 32)."""
    w, h = image.size
    return size_for_megapixels(f"{w}:{h}", megapixels)


def encode_reference_rows(vae, image, patch_size, seed: int, index: int) -> mx.array:
    """One reference's packed, normalised clean rows (before the 0.999 anchor noise).

    The release's keyframe recipe: posterior SAMPLED (not its mode) and rounded through float16
    before ``latents_mean/std``. Upstream draws that sample from the request's generator; MLX's RNG is
    not torch's, so the draw is keyed on the seed and the reference index instead.
    """
    cfg = vae.config
    pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None, :, None] / 255.0
    pixels = (pixels - np.array(PIXEL_MEAN, np.float32).reshape(1, 3, 1, 1, 1)) / np.array(
        PIXEL_STD, np.float32).reshape(1, 3, 1, 1, 1)
    moments = vae._encode_clip(mx.array(pixels).transpose(0, 2, 3, 4, 1))
    channels = cfg.latent_channels
    mean, logvar = moments[..., :channels], mx.clip(moments[..., channels:], -30.0, 20.0)
    noise = mx.random.normal(mean.shape, key=mx.random.key(int(seed) * 7919 + 101 + index))
    latent = (mean + mx.exp(0.5 * logvar) * noise).transpose(0, 4, 1, 2, 3)
    latent = latent.astype(mx.float16).astype(mx.float32)
    lm = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1, 1, 1)
    ls = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1, 1, 1)
    rows = patchify_video_latents((latent - lm) / ls, patch_size)
    mx.eval(rows)
    return rows


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
    reference_rows: mx.array | None = None,
    reference_grids: tuple = (),
) -> mx.array:
    """Run the 6-forward loop. Returns normalised latents ``(1, C, 1, H', W')`` (float32).

    ``height``/``width`` must already be a 32-aligned canvas (see :func:`generation_canvas`).
    ``reference_rows`` (edit) are the references' clean rows in packed order, one block per entry
    of ``reference_grids``. They sit between the text and the target, are held at
    ``0.999 * clean + 0.001 * noise`` every forward, and only the target rows are stepped.
    """
    if len(sigmas) - 1 != heads[0].shape[0]:
        raise ValueError(f"{len(sigmas) - 1} forwards but {heads[0].shape[0]} fused heads: the heads are per step")
    if height % CANVAS_MULTIPLE or width % CANVAS_MULTIPLE:
        raise ValueError("canvas must be a multiple of 32")
    lh, lw = height // ratio, width // ratio
    patch = tuple(dit.config.patch_size)
    grids = tuple((int(h), int(w)) for h, w in reference_grids)
    if (reference_rows is None) != (not grids):
        raise ValueError("reference rows and reference grids must be given together")
    layout = build_packed_sequence(text_tags, 1, lh, lw, 0, patch, (), reference_grids=grids)
    n_cond = layout.num_condition_video_rows
    if reference_rows is not None and int(reference_rows.shape[0]) != n_cond:
        raise ValueError(f"{reference_rows.shape[0]} reference rows for a layout of {n_cond}")
    if verbose:
        print(f"canvas {width}x{height}: {layout.sequence_length:,} rows "
              f"({len(text_tags):,} text, {n_cond:,} reference, "
              f"{(lh // patch[1]) * (lw // patch[2]):,} image)", flush=True)

    mx.random.seed(seed)
    video_rows = patchify_video_latents(
        mx.random.normal((1, latent_channels, 1, lh, lw)).astype(mx.float32), patch
    )
    anchors = None
    if reference_rows is not None:
        # Drawn after the target noise, so a seed keeps its picture with or without references.
        drawn = mx.random.normal(reference_rows.shape).astype(mx.float32)
        level = float(np.float32(REFERENCE_LEVEL))
        anchors = level * reference_rows.astype(mx.float32) + (1.0 - level) * drawn
        mx.eval(anchors)
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
            packed = video_rows if anchors is None else mx.concatenate([anchors, video_rows])
            video_pred, _ = dit(
                packed[None].astype(mx.bfloat16), audio_rows[None].astype(mx.bfloat16), embeds,
                timestep_table, plan[i], layout.token_tags, layout.position_ids,
                layout.video_indices, layout.audio_indices, layout.text_indices,
                modulation_cache=modulation_cache,
            )
            video_rows = sched.step(video_pred[0][n_cond:].astype(mx.float32), float(t), video_rows)
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
                 min_generation_area: int | None = DEFAULT_MIN_GENERATION_AREA,
                 references: list | None = None) -> np.ndarray:
        """Text to image, or an edit when ``references`` (PIL RGB pictures, Picture 1 first) are given."""
        from .h3_image_vae import decode_to_uint8, load_h3_image_vae, load_h3_image_vae_encoder
        from .load import load_dit
        from .text_encoder import MiniMaxH3TextEncoder

        references = list(references or [])
        if len(references) > MAX_REFERENCES:
            raise ValueError(f"MiniMax-H3-Image takes at most {MAX_REFERENCES} reference pictures, got {len(references)}")
        m = self.metrics = {"compile_blocks": self.compile_blocks, "references": len(references)}
        t0 = time.perf_counter()
        cw, ch = generation_canvas(width, height, min_generation_area)
        m["canvas"] = [cw, ch]

        started = time.perf_counter()
        encoder = MiniMaxH3TextEncoder(self.text_dir, load_vision=bool(references), verbose=False,
                                       config_path=self.text_config)
        if references:
            qwen_images = [bound_reference(img, *QWEN_SHORT_EDGE) for img in references]
            cond, tags = encoder.encode(prompt, qwen_images, label_format=EDIT_LABEL)
        else:
            cond, tags = encoder.encode(prompt)
        mx.eval(cond)
        m["text_seconds"], m["text_rows"] = time.perf_counter() - started, int(cond.shape[1])
        del encoder
        gc.collect()
        mx.clear_cache()

        reference_rows, grids = None, ()
        if references:
            started = time.perf_counter()
            vae_json = json.loads((self.root / "vae" / "config.json").read_text())
            enc = load_h3_image_vae_encoder(self.root / "vae" / "diffusion_pytorch_model.safetensors", vae_json)
            parts, grid_list = [], []
            for index, img in enumerate(references):
                pixels, grid = vae_reference(img)
                parts.append(encode_reference_rows(enc, pixels, (1, 2, 2), seed, index))
                grid_list.append(grid)
            reference_rows = parts[0] if len(parts) == 1 else mx.concatenate(parts)
            grids = tuple(grid_list)
            mx.eval(reference_rows)
            m["reference_seconds"] = time.perf_counter() - started
            m["reference_grids"] = [list(g) for g in grids]
            if verbose:
                print(f"reference pictures: {len(parts)} at {[g for g in grids]} latent, "
                      f"{reference_rows.shape[0]} rows", flush=True)
            del enc
            gc.collect()
            mx.clear_cache()

        started = time.perf_counter()
        dit = load_dit(self.root / "dit", verbose=verbose)
        cache = load_modulation_cache(self.root / "dit" / "adaln_cache.safetensors")
        heads = load_pdd_heads(self.root / "pdd_fused_heads.safetensors")
        dit.compile_image_blocks = self.compile_blocks
        m["dit_load_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        latents = denoise_h3_image(dit, cond, tags, ch, cw, cache, heads, seed=seed, verbose=verbose,
                                   reference_rows=reference_rows, reference_grids=grids)
        mx.eval(latents)
        m["denoise_seconds"] = time.perf_counter() - started
        m["step_seconds"] = list(getattr(denoise_h3_image, "last_step_seconds", []))
        m["forwards"] = len(m["step_seconds"])
        del dit, cache, reference_rows
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
