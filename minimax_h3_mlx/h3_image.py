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
from dataclasses import asdict, dataclass
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


def vae_reference(image, short_edge: int | None = None):
    """The VAE's copy of a reference: bounded to a 1024-2048 px short edge (or exactly
    ``short_edge``, the light-references memory setting), then cropped top-left to an even 16 px
    grid. Returns ``(pixels, (latent_h, latent_w))``."""
    bounds = (short_edge, short_edge) if short_edge else VAE_SHORT_EDGE
    bounded = bound_reference(image, *bounds)
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


def process_footprint() -> int:
    """This process's physical footprint in bytes (what macOS's memory killer reads), or 0."""
    try:
        import ctypes
        import os

        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        buf = ctypes.create_string_buffer(512)
        if libc.proc_pid_rusage(os.getpid(), 2, buf) != 0:      # RUSAGE_INFO_V2
            return 0
        return int.from_bytes(buf.raw[72:80], "little")          # ri_phys_footprint
    except Exception:
        return 0


class _PhaseMeter:
    """Wall time, MLX peak and the highest process footprint of each phase of one picture.

    A 20 Hz sampler thread reads the footprint, because MLX's own peak does not count the page
    cache, the interpreter or the Metal driver, and jetsam acts on the footprint.
    """

    def __init__(self):
        import threading

        self.phases: dict = {}
        self._name, self._fp, self._t = None, 0, 0.0
        self._overall = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self):
        while not self._stop.is_set():
            fp = process_footprint()
            if fp > self._fp:
                self._fp = fp
            if fp > self._overall:
                self._overall = fp
            self._stop.wait(0.05)

    def begin(self, name: str):
        mx.synchronize()
        mx.reset_peak_memory()
        self._name, self._fp, self._t = name, process_footprint(), time.perf_counter()

    def end(self):
        mx.synchronize()
        fp = max(self._fp, process_footprint())
        self._overall = max(self._overall, fp)
        self.phases[self._name] = {"seconds": round(time.perf_counter() - self._t, 2),
                                   "mlx_peak_gb": round(mx.get_peak_memory() / 1e9, 2),
                                   "footprint_gb": round(fp / 1e9, 2)}

    def close(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=1.0)
        return {"phases": self.phases, "peak_footprint_gb": round(self._overall / 1e9, 2),
                "peak_mlx_gb": max((p["mlx_peak_gb"] for p in self.phases.values()), default=0.0)}


@dataclass(frozen=True)
class MemorySettings:
    """How one picture holds its weights. The default is the original, fully resident path.

    ``stream_text_encoder``: the Qwen3-VL encoder is read one decoder layer at a time (with the
    next layer read ahead), never resident as a whole. ``stream_dit``: the DiT's block linears are
    read per block during each forward (:mod:`dit_stream`), except the first ``dit_resident_gb``
    worth of blocks, which stay resident. ``cache_gb``: MLX's buffer-cache cap (None = MLX's own).
    ``decode_batch``: decoder tiles per call. Streaming moves evaluation boundaries only: the same
    bytes go through the same kernels, so the picture is the same (``tests/test_h3_image_lowmem``
    and the bench record the sha of both).
    """

    stream_text_encoder: bool = False
    stream_dit: bool = False
    dit_resident_gb: float = 0.0
    cache_gb: float | None = None
    decode_batch: int = 8
    name: str = "resident"
    #: The VAE copy of each edit reference is bounded to this short edge instead of the release's
    #: 1024-2048 px (None). 1024 is the release's own lower bound; a smaller copy means fewer rows
    #: in every forward, so less memory and a faster edit, at a measured cost in fidelity to the
    #: references. It is the only lossy setting: ``auto`` takes it only when nothing exact fits.
    reference_short_edge: int | None = None

    @property
    def lowmem(self) -> bool:
        return self.stream_text_encoder or self.stream_dit


RESIDENT = MemorySettings()

#: Measured peak footprint (GiB) of each phase on the resident path and per streamed setting; see
#: ``plan_memory``. Filled from the bench (M4 Max 64 GB, mlx 0.32.4) and inflated by MARGIN.
MEMORY_MARGIN = 1.10


def mac_budget_gib(ram_gib: float) -> float:
    """What a render may use on a Mac with ``ram_gib``: the studio's H3 rule (RAM less the larger of
    6 GB and an eighth of RAM, which macOS and an open browser keep)."""
    return float(ram_gib) - max(6.0, float(ram_gib) / 8.0)


# --- calibration: peak process footprint per phase, GiB (measured) -------------------------------
#: Measured, M4 Max 64 GB, mlx 0.32.4, 12 Nov 2026 (footprint of the largest phase):
#:   resident  4 MP: text 27.96 / denoise 23.54 / decode 4.84   16 MP: 27.98 / 27.99 / 6.21
#:   streamed  4 MP: text  1.88 / denoise  4.83 / decode 4.14   16 MP:  1.88 /  8.15 / 4.58
#:   streamed with 12 GB of blocks resident, 4 MP: denoise 17.53, same wall time (99.3 s).
#: Latents and pixels were byte-identical across all three at each size.
#: Resident text encoder (Qwen3-VL-32B truncated to 50 layers, Q8) and with the vision tower.
TEXT_RESIDENT_GIB = 28.0
TEXT_VISION_EXTRA_GIB = 1.0
#: Streamed text encoder: one decoder layer, the one read ahead, and the hidden states.
TEXT_STREAMED_GIB = 2.0
#: The DiT: everything outside the 50 blocks, and one block's streamed linears.
DIT_NONBLOCK_GIB = 0.8
DIT_BLOCK_GIB = 0.38
DIT_BLOCKS = 50
#: Streamed DiT holds three block slots (computing, queued, being read).
DIT_SLOTS = 3
#: Denoise working memory per 1,000 packed rows (attention projections, SwiGLU, residual).
#: Fitted 13 Nov over 16.9k rows (16 MP, 3.41 GiB of work), 25.1k (24 MP, 5.13) and 47.7k (the
#: six-reference edit, 10.34): 0.20-0.22 per 1,000, attention growing a little faster than the
#: rows. 0.20 with :data:`MEMORY_MARGIN` covers all three (the 0.185 of 12 Nov fell short on
#: the edit: 13.9 estimated, 14.08 measured).
DENOISE_GIB_PER_KROW = 0.20
#: Process outside MLX's counters (interpreter, driver, libraries).
RUNTIME_GIB = 0.8
#: Decoder weights and its working memory per megapixel at a batch of 8 / 1 tiles.
DECODE_WEIGHTS_GIB = 1.2
DECODE_GIB_PER_MP = {8: 0.25, 1: 0.12}


#: A reference whose size is not known yet is priced as the largest the release takes: a 16:9
#: picture at the 2048 px short edge (2048x3648 for the VAE, 1024x1824 for the text encoder).
WORST_REFERENCE = (3648, 2048)
#: Text rows outside the references (the prompt and its template).
TEXT_BASE_ROWS = 512


def _reference_sizes(references) -> list[tuple[int, int]]:
    if isinstance(references, int):
        return [WORST_REFERENCE] * references
    return [(int(w), int(h)) for w, h in (references or [])]


def _bounded(size: tuple[int, int], low: int, high: int) -> tuple[int, int]:
    return bounded_reference_size(size[0], size[1], low, high) or size


def reference_rows(references, reference_short_edge: int | None = None) -> tuple[int, int]:
    """``(vae_rows, vision_rows)`` the references add: each one's VAE copy (one row per 32x32 px
    of its 1024-2048 px short-edge copy, or of its ``reference_short_edge`` copy) and its 1024 px
    vision copy in the text (about one row per 32x32 px too). ``references`` is a count (priced as
    :data:`WORST_REFERENCE`) or the pictures' ``(width, height)``. Measured 13 Nov, six pictures:
    33,878 reference rows and 9,751 text rows."""
    bounds = (reference_short_edge, reference_short_edge) if reference_short_edge else VAE_SHORT_EDGE
    vae = vision = 0
    for size in _reference_sizes(references):
        w, h = _bounded(size, *bounds)
        vae += (w // 32) * (h // 32)
        qw, qh = _bounded(size, *QWEN_SHORT_EDGE)
        vision += int(qw * qh / 1024) + 16
    return vae, vision


def packed_rows_estimate(megapixels: float, references=0, text_rows: int = TEXT_BASE_ROWS,
                         reference_short_edge: int | None = None) -> int:
    """Rows the DiT carries: the target (one per 32x32 px), the references and the text."""
    vae, vision = reference_rows(references, reference_short_edge)
    return int(megapixels * 1024 + vae + text_rows + vision)


def estimate_peak_gib(settings: MemorySettings, megapixels: float, references=0) -> float:
    """Estimated peak footprint (GiB, margin included) of the largest phase under ``settings``.
    ``references``: a count or the pictures' ``(width, height)`` (see :func:`reference_rows`)."""
    sizes = _reference_sizes(references)
    text = TEXT_STREAMED_GIB if settings.stream_text_encoder else TEXT_RESIDENT_GIB
    if sizes:
        # The vision tower plus the hidden states of the vision rows (six pictures: 6.15 GB streamed).
        _, vision = reference_rows(sizes)
        text += (TEXT_VISION_EXTRA_GIB if not settings.stream_text_encoder else 0.5) + 0.35 * vision / 1000.0
    work = DENOISE_GIB_PER_KROW * packed_rows_estimate(
        megapixels, sizes, reference_short_edge=settings.reference_short_edge) / 1000.0
    if settings.stream_dit:
        kept = min(DIT_BLOCKS, int(settings.dit_resident_gb // DIT_BLOCK_GIB)) if settings.dit_resident_gb else 0
        weights = DIT_NONBLOCK_GIB + kept * DIT_BLOCK_GIB + DIT_SLOTS * DIT_BLOCK_GIB
    else:
        weights = DIT_NONBLOCK_GIB + DIT_BLOCKS * DIT_BLOCK_GIB
    cache = settings.cache_gb or 0.0
    denoise = weights + work + cache
    batch = 1 if settings.decode_batch <= 1 else 8
    decode = DECODE_WEIGHTS_GIB + DECODE_GIB_PER_MP[batch] * megapixels + min(cache, 1.0)
    return round((max(text, denoise, decode) + RUNTIME_GIB) * MEMORY_MARGIN, 2)


LOWMEM = MemorySettings(stream_text_encoder=True, stream_dit=True, cache_gb=2.0, name="lowmem")
LOWMEM_MIN = MemorySettings(stream_text_encoder=True, stream_dit=True, cache_gb=1.0, decode_batch=1,
                            name="lowmem-min")
#: The one lossy rung, edits only: references copied at the release's lower bound (1024 px short
#: edge) instead of up to 2048 px. A six-picture edit carries about 48,000 rows at full size and
#: about 23,000 here.
LOWMEM_LIGHTREF = MemorySettings(stream_text_encoder=True, stream_dit=True, cache_gb=1.0, decode_batch=1,
                                 name="lowmem-lightref", reference_short_edge=1024)


def memory_ladder(megapixels: float, references: int = 0) -> list[MemorySettings]:
    """The settings ``auto`` picks from, first fitting wins.

    The resident path is not on it: streaming was measured as fast or faster at every size (the
    reads hide under each block's compute, and nothing big is loaded up front: 4 MP 99 s vs 107 s,
    16 MP 525 s vs 528 s), byte-identical, at a third or less of the memory. Keeping blocks
    resident bought no time either (12 GB resident: 99.3 s, the same). ``resident`` stays an
    explicit choice. An edit gets one more rung, :data:`LOWMEM_LIGHTREF`, the only lossy one."""
    if _reference_sizes(references):
        return [LOWMEM, LOWMEM_MIN, LOWMEM_LIGHTREF]
    return [LOWMEM, LOWMEM_MIN]


def machine_ram_gib() -> float:
    import subprocess

    try:
        return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 1024**3
    except Exception:
        return 0.0


def memory_settings(mode: str, megapixels: float, references=0, ram_gib: float | None = None,
                    dit_resident_gb: float | None = None, budget_gib: float | None = None) -> MemorySettings:
    """``resident`` | ``lowmem`` | ``auto`` (the fastest rung that fits ``ram_gib``, default this
    Mac, and ``budget_gib`` if given)."""
    if mode == "resident":
        return RESIDENT
    if mode == "lowmem":
        settings = LOWMEM
    elif mode == "auto":
        settings = plan_memory(ram_gib or machine_ram_gib(), megapixels, references, budget_gib)
    else:
        raise ValueError(f"unknown memory mode {mode!r}")
    if dit_resident_gb is not None and settings.stream_dit:
        settings = MemorySettings(settings.stream_text_encoder, True, float(dit_resident_gb), settings.cache_gb,
                                  settings.decode_batch, f"{settings.name}@{dit_resident_gb:g}",
                                  settings.reference_short_edge)
    return settings


def plan_memory(ram_gib: float, megapixels: float, references=0,
                budget_gib: float | None = None) -> MemorySettings:
    """The fastest rung of :func:`memory_ladder` whose largest phase fits the budget: a Mac's
    (:func:`mac_budget_gib`), or ``budget_gib`` when the caller knows what is free right now."""
    budget = mac_budget_gib(ram_gib) if budget_gib is None else float(budget_gib)
    if ram_gib and budget_gib is not None:
        budget = min(budget, mac_budget_gib(ram_gib))
    for settings in memory_ladder(megapixels, references):
        if estimate_peak_gib(settings, megapixels, references) <= budget:
            return settings
    return memory_ladder(megapixels, references)[-1]


class H3ImageRunner:
    """Staged driver: text encoder out before the DiT comes in, DiT out before the decoder.

    ``root`` is the folder ``scripts/fetch_h3_image.py`` builds (``dit/``, ``vae/``,
    ``pdd_fused_heads.safetensors``); ``text_dir`` is the H3 text-encoder folder the other H3 modes
    already use (the Q8 ``ddalcu-q8`` folder) and ``text_config`` the upstream ``text_encoder/config.json``.
    """

    def __init__(self, root: str | Path, text_dir: str | Path, text_config: str | Path | None = None,
                 compile_blocks: bool = False, memory: MemorySettings = RESIDENT):
        self.root, self.text_dir = Path(root), Path(text_dir)
        self.text_config = Path(text_config) if text_config else None
        self.compile_blocks = compile_blocks
        self.memory = memory
        self.metrics: dict = {}

    def close(self):
        gc.collect()
        mx.clear_cache()

    def generate(self, prompt: str, width: int, height: int, seed: int = 0, verbose: bool = True,
                 min_generation_area: int | None = DEFAULT_MIN_GENERATION_AREA,
                 references: list | None = None) -> np.ndarray:
        """Text to image, or an edit when ``references`` (PIL RGB pictures, Picture 1 first) are given."""
        references = list(references or [])
        if len(references) > MAX_REFERENCES:
            raise ValueError(f"MiniMax-H3-Image takes at most {MAX_REFERENCES} reference pictures, got {len(references)}")
        mem = self.memory
        m = self.metrics = {"compile_blocks": self.compile_blocks, "references": len(references),
                            "memory": asdict(mem)}
        t0 = time.perf_counter()
        cw, ch = generation_canvas(width, height, min_generation_area)
        m["canvas"] = [cw, ch]
        meter = _PhaseMeter()
        previous_cache = None
        if mem.cache_gb is not None:
            previous_cache = mx.set_cache_limit(int(mem.cache_gb * 1024**3))

        try:
            image = self._generate(prompt, width, height, cw, ch, seed, verbose, references, meter, m)
        finally:
            m.update(meter.close())
            if previous_cache is not None:
                mx.set_cache_limit(previous_cache)
        m["total_seconds"] = time.perf_counter() - t0
        m["peak_gb"] = m["peak_mlx_gb"]
        if verbose:
            parts = ", ".join("%s %.1f" % (k, v["footprint_gb"]) for k, v in m["phases"].items())
            print("memory: %s, peak footprint %.1f GB (%s)" % (mem.name, m["peak_footprint_gb"], parts), flush=True)
        return image

    def _generate(self, prompt, width, height, cw, ch, seed, verbose, references, meter, m):
        from .h3_image_vae import decode_to_uint8, load_h3_image_vae, load_h3_image_vae_encoder
        from .load import load_dit
        from .text_encoder import MiniMaxH3TextEncoder

        mem = self.memory
        meter.begin("text")
        started = time.perf_counter()
        encoder = MiniMaxH3TextEncoder(self.text_dir, load_vision=bool(references), verbose=False,
                                       config_path=self.text_config, stream=mem.stream_text_encoder)
        if references:
            qwen_images = [bound_reference(img, *QWEN_SHORT_EDGE) for img in references]
            cond, tags = encoder.encode(prompt, qwen_images, label_format=EDIT_LABEL)
        else:
            cond, tags = encoder.encode(prompt)
        mx.eval(cond)
        m["text_seconds"], m["text_rows"] = time.perf_counter() - started, int(cond.shape[1])
        if getattr(encoder, "stream_stats", None):
            m["text_stream"] = dict(encoder.stream_stats)
        del encoder
        gc.collect()
        mx.clear_cache()
        meter.end()

        reference_rows, grids = None, ()
        if references:
            meter.begin("reference")
            started = time.perf_counter()
            vae_json = json.loads((self.root / "vae" / "config.json").read_text())
            enc = load_h3_image_vae_encoder(self.root / "vae" / "diffusion_pytorch_model.safetensors", vae_json)
            parts, grid_list = [], []
            for index, img in enumerate(references):
                pixels, grid = vae_reference(img, mem.reference_short_edge)
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
            meter.end()

        meter.begin("denoise")
        started = time.perf_counter()
        stream = None
        if mem.stream_dit:
            from .dit_stream import StreamPolicy

            stream = StreamPolicy(resident_gb=mem.dit_resident_gb or None, pipeline=True)
        dit = load_dit(self.root / "dit", verbose=verbose, stream=stream)
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
        streamer = getattr(dit, "streamer", None)
        if streamer is not None:
            m["dit_stream"] = streamer.summary()
            streamer.close()
        self.last_latents = latents
        del dit, cache, reference_rows, streamer
        gc.collect()
        mx.clear_cache()
        meter.end()

        meter.begin("decode")
        started = time.perf_counter()
        vae = load_h3_image_vae(self.root / "vae" / "diffusion_pytorch_model.safetensors",
                                json.loads((self.root / "vae" / "config.json").read_text()))
        vae.decode_batch = max(1, int(mem.decode_batch))
        image = resize_output(decode_to_uint8(vae, latents), width, height)
        del vae
        gc.collect()
        mx.clear_cache()
        m["decode_seconds"] = time.perf_counter() - started
        meter.end()
        return image
