"""Packed-sequence geometry for MiniMax-H3.

MiniMax-H3 runs its transformer over a single packed 1-D sequence holding every modality at once.
For the text/keyframe tasks the row order is

    [ text (L) | keyframe conditions (C) | target audio (A) | target video (V) ]

Every piece of geometry here exists to place a row in that sequence and give it its ``(t, h, w)``
rotary coordinate. The coordinates are built in **float64** because video and audio share one
40-units-per-second rotary clock — video advances ``5/3`` rotary units per pixel frame at 24 fps,
audio one unit per latent at 40 latents/s — and that shared clock *is* the audio/video alignment.

The grids are built with NumPy rather than MLX for two reasons the reference calls out: its
``linspace(..., endpoint=False)`` is ``start + arange(n) * (stop - start) / n``, which differs from
the torch/MLX formulation, and the temporal span is summed pairwise, which differs from a
sequential sum in the last ulp from 16 latent frames onwards. Only the finished grid crosses into
MLX, and the transformer casts it to float32 there — matching the reference exactly.

Padding is never emitted: the reference pads to a multiple of 64 and keeps the tail as a separate
attention document, but a padless sequence needs no attention mask at all, which keeps the fast
unmasked attention path available.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .config import TAG_AUDIO, TAG_TEXT, TAG_VIDEO

# MiniMax-H3 generates at a fixed 24 fps and was released for a 768-pixel short edge only, with a
# soft area cap of 768x1344 and both axes rounded to a multiple of 32.
FPS = 24
SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
CANVAS_MULTIPLE = 32
MIN_ASPECT_RATIO = 1 / 4
MAX_ASPECT_RATIO = 4
MIN_DURATION = 5.0
MAX_DURATION = 15.0

# The video VAE encodes 17 pixel frames per chunk and drops the 3 trailing latent frames of every
# chunk, so `17 * n + 5` pixel frames map to `5 * n + 2` latent frames.
FRAMES_PER_CHUNK = 17
LATENTS_PER_CHUNK = 5

# The pixel convention of the video VAE: ImageNet-normalized RGB over a [0, 1] base range.
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)

# MiniMax-H3 conditions on the *unnormalized* hidden state its Qwen3-VL conditioner produces after
# the 50th of its 64 decoder layers, i.e. `hidden_states[50]` (`hidden_states[0]` is the embedding).
TEXT_ENCODER_LAYER = 50

# The audio VAE hops 800 samples at 32 kHz: 40 latents per second. Stereo is carried as two
# channel-major blocks of audio rows.
AUDIO_LATENTS_PER_SECOND = 40
AUDIO_CHANNELS = 2

# Conditioning rows are not fully clean: keyframe latents are noised to t = 0.999 and run at that
# timestep for every denoising step.
KEYFRAME_NOISE_AUG = 0.999

# The seeded posterior sample of the keyframe VAE encode, fixed independently of the request seed.
KEYFRAME_ENCODE_SEED = 42

# Rotary-time constants. One latent frame spans `5/3 * frames_per_latent`, where the pattern
# `(1, 4, 4, 4, 4)` mirrors the VAE's 17-pixel-frames-to-5-latent-frames grouping; the spatial axes
# are normalized by the square root of the latent area and scaled by 32.
_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32


@dataclass
class PackedSequence:
    """The structural description of one packed MiniMax-H3 sequence."""

    sequence_length: int
    position_ids: mx.array  # (seq, 3) float32 — cast from the float64 host grid
    token_tags: mx.array  # (seq,) int32
    video_indices: mx.array  # conditioning rows first, then target rows
    audio_indices: mx.array  # reference rows first (ref2va), then target rows
    text_indices: mx.array
    num_condition_video_rows: int
    num_condition_audio_rows: int


def resolve_canvas_size(aspect_width: float, aspect_height: float) -> tuple[int, int]:
    """Resolve a display aspect ratio into a MiniMax-H3 canvas.

    The short edge starts at 768, the area is capped at ``768 * 1344`` and both axes are rounded to
    the nearest multiple of 32 — so the final area may end up slightly above the pre-rounding
    budget. Only the ratio of the arguments matters.

    Returns:
        ``(height, width)`` of the canvas.
    """
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"The aspect ratio must be positive, got {aspect_width}:{aspect_height}.")

    ratio = aspect_width / aspect_height
    if not MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO:
        raise ValueError(
            f"MiniMax-H3 supports aspect ratios from 1:4 to 4:1, got "
            f"{aspect_width}:{aspect_height} ({ratio:g})."
        )

    if ratio >= 1.0:
        width, height = SHORT_EDGE * ratio, float(SHORT_EDGE)
    else:
        width, height = float(SHORT_EDGE), SHORT_EDGE / ratio

    area = width * height
    if area > MAX_PIXELS:
        scale = (MAX_PIXELS / area) ** 0.5
        width, height = width * scale, height * scale

    m = CANVAS_MULTIPLE
    return max(m, round(height / m) * m), max(m, round(width / m) * m)


def align_num_frames(num_frames: int) -> int:
    """Snap a frame count up to the next ``17 * n + 5`` the video VAE can encode."""
    if num_frames < 1:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
    while num_frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        num_frames += 1
    return num_frames


def video_latent_num_frames(num_frames: int) -> int:
    """Latent frames produced for a ``17 * n + 5`` frame count: ``5 * n + 2``."""
    if num_frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        raise ValueError(f"`num_frames` must be of the form 17 * n + 5, got {num_frames}.")
    return (num_frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2


def audio_latent_num_frames(num_frames: int) -> int:
    """Audio latents covering ``num_frames`` video frames at 24 fps, on the 40 Hz latent grid."""
    return int(round(num_frames / FPS * AUDIO_LATENTS_PER_SECOND))


def prepare_keyframe_image(image, height: int, width: int, stretch: bool):
    """Put a keyframe onto the target canvas.

    The first keyframe of a request is the geometry anchor and is *stretched* onto the canvas; a
    second keyframe follows that canvas and is cover-cropped (aspect-preserving max-scale LANCZOS
    resize plus a centre crop). An image that already is the canvas is returned untouched, without a
    resampling pass.
    """
    from PIL import Image

    if image.size == (width, height):
        return image
    if stretch:
        return image.resize((width, height), Image.Resampling.LANCZOS)

    scale = max(width / image.size[0], height / image.size[1])
    resized_size = (max(width, round(image.size[0] * scale)), max(height, round(image.size[1] * scale)))
    left = max(0, (resized_size[0] - width) // 2)
    top = max(0, (resized_size[1] - height) // 2)
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    return resized.crop((left, top, left + width, top + height))


def patchify_video_latents(latents: mx.array, patch_size: tuple[int, int, int]) -> mx.array:
    """Pack ``(B, C, F, H, W)`` video latents into transformer rows, frame-major then row-major.

    Returns ``(B * num_patches, C * prod(patch_size))``.
    """
    pt, ph, pw = patch_size
    b, c, f, h, w = latents.shape
    if f % pt or h % ph or w % pw:
        raise ValueError(f"Latents of shape {latents.shape} are not divisible by the patch {patch_size}.")

    x = latents.reshape(b, c, f // pt, pt, h // ph, ph, w // pw, pw)
    x = x.transpose(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(-1, c * pt * ph * pw)


def unpatchify_video_tokens(
    rows: mx.array,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    channels: int,
    patch_size: tuple[int, int, int],
) -> mx.array:
    """Inverse of :func:`patchify_video_latents`. Returns ``(B, C, F, H, W)``."""
    pt, ph, pw = patch_size
    x = rows.reshape(
        -1,
        num_latent_frames // pt,
        latent_height // ph,
        latent_width // pw,
        channels,
        pt,
        ph,
        pw,
    )
    x = x.transpose(0, 4, 1, 5, 2, 6, 3, 7)
    return x.reshape(-1, channels, num_latent_frames, latent_height, latent_width)


def unpack_audio_tokens(rows: mx.array, num_audio_latents: int) -> mx.array:
    """Unpack channel-major audio rows into ``(2, latent_channels, num_audio_latents)``.

    One batch item per stereo channel, which is what the mono audio VAE consumes.
    """
    x = rows.reshape(AUDIO_CHANNELS, num_audio_latents, rows.shape[-1])
    return x.transpose(0, 2, 1)


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> np.ndarray:
    """One aspect-normalized spatial rotary axis.

    ``dim // patch`` coordinates centred on the unit interval, scaled by 32, right endpoint
    excluded — so a square canvas spans ``[0, 32)``.
    """
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE


def _temporal_position_grid(num_latent_frames: int, origin: float) -> np.ndarray:
    """Rotary time of every latent frame from ``origin``. Spacing is ``5/3 * (1, 4, 4, 4, 4)``."""
    spans = np.array(
        [
            _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[i % len(_ROPE_FRAMES_PER_LATENT)]
            for i in range(num_latent_frames)
        ],
        dtype=np.float64,
    )
    return origin + np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(spans[:-1])])


def _temporal_position_span(num_latent_frames: int) -> float:
    """Rotary time spanned by ``num_latent_frames`` latent frames.

    Summed pairwise by NumPy rather than sequentially: the reference computes the keyframe anchor
    this way and the two summation orders differ in the last ulp from 16 latent frames onwards.
    """
    spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
    for i in range(len(_ROPE_FRAMES_PER_LATENT)):
        spans[i :: len(_ROPE_FRAMES_PER_LATENT)] *= _ROPE_FRAMES_PER_LATENT[i]
    return float(spans.sum())


def parse_anchor(anchor: str, num_frames: int | None = None) -> tuple[str, int | None]:
    """``"first"`` / ``"last"`` / ``"frame:<pixel index>"`` -> ``(kind, pixel_index)``.

    ``"frame:0"`` is the first frame and ``"frame:<num_frames - 1>"`` the last one: the three
    spellings put a keyframe at exactly the same rotary time, so a timed keyframe that happens to
    land on an end of the window is the official first/last conditioning, not an approximation.
    """
    if anchor in ("first", "last"):
        return anchor, None
    kind, _, value = str(anchor).partition(":")
    if kind != "frame" or not value.strip().lstrip("-").isdigit():
        raise ValueError(
            f"A keyframe anchor must be 'first', 'last' or 'frame:<index>', got {anchor!r}."
        )
    index = int(value)
    if index < 0 or (num_frames is not None and index >= num_frames):
        raise ValueError(
            f"Keyframe anchor {anchor!r} is outside the window's 0..{(num_frames or 0) - 1} frames."
        )
    return "frame", index


def build_packed_sequence(
    text_token_tags: np.ndarray | list[int],
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
    keyframe_anchors: tuple[str, ...] = (),
    *,
    history_latent_frames: int = 0,
    history_audio_latents: int = 0,
    history_audio_offset: float = 0.0,
    reference_grids: tuple[tuple[int, int], ...] = (),
) -> PackedSequence:
    """Build the ``[text | video conditions | audio conditions | target audio | target video]`` layout.

    Args:
        text_token_tags: modality tag of every text row. Text is tagged ``1``, except the rows of a
            keyframe's vision block, which MiniMax-H3 tags ``0`` (video).
        keyframe_anchors: one entry per keyframe conditioning block, in packed order; ``"first"``
            anchors at the first latent frame, ``"last"`` at the last, ``"frame:<i>"`` at pixel
            frame ``i`` of the target (rotary time ``origin + i * 5/3``, the clock every target
            frame already lives on).
        history_latent_frames: latent frames of *carried context* — the tail of the clip this
            window continues — placed on the clock immediately BEFORE the target, so the target
            starts where the history ends. They are video conditioning rows like a keyframe (same
            noise level), packed ahead of the keyframe blocks. ``0`` reproduces the reference
            layout exactly.
        history_audio_latents: audio latents (per stereo channel) of carried context, starting at
            the same instant as the video history. Packed as conditioning audio rows, channel-major
            like the target audio, ahead of the target audio.
        history_audio_offset: rotary units (1/40 s) by which the carried audio starts after the
            carried video — non-zero only when the carried span is not a whole number of audio
            latents, so the audio can end exactly where the carried picture ends.

        reference_grids: ``(latent_height, latent_width)`` of every reference IMAGE block, in packed
            order. A reference is a single still on its own canvas and its own aspect-normalised
            spatial grid (it never binds the target geometry), one integer rotary slot each, placed
            on the clock before everything else - the official ``ref2va`` image block
            (``packing_ref2va.build_ref2va_packed_sequence``), here carried by the FL2VA layout.
            Packed first among the video conditioning rows.

    With no history the media clock starts at the text length, exactly as the reference builds it.
    With history the whole target moves later by the history's rotary span; the audio of both
    keeps the shared 40-units-per-second clock, so carried sound and carried picture stay in sync.
    """
    _, ph, pw = patch_size
    text_tags = np.asarray(text_token_tags, dtype=np.int64)
    rows_per_frame = (latent_height // ph) * (latent_width // pw)
    num_text = int(text_tags.shape[0])
    num_history_rows = int(history_latent_frames) * rows_per_frame
    ref_rows = [(int(rh) // ph) * (int(rw) // pw) for rh, rw in reference_grids]
    num_reference_rows = sum(ref_rows)
    num_condition_rows = num_reference_rows + num_history_rows + len(keyframe_anchors) * rows_per_frame
    num_condition_audio_rows = int(history_audio_latents) * AUDIO_CHANNELS
    num_audio_rows = num_audio_latents * AUDIO_CHANNELS
    num_video_rows = num_latent_frames * rows_per_frame
    seq_len = (
        num_text + num_condition_rows + num_condition_audio_rows + num_audio_rows + num_video_rows
    )
    target_num_frames = (
        (num_latent_frames - 2) // LATENTS_PER_CHUNK * FRAMES_PER_CHUNK + LATENTS_PER_CHUNK
        if num_latent_frames % LATENTS_PER_CHUNK == 2
        else None
    )

    reference_start = num_text
    condition_start = reference_start + num_reference_rows
    keyframe_start = condition_start + num_history_rows
    audio_condition_start = reference_start + num_condition_rows
    audio_start = audio_condition_start + num_condition_audio_rows
    video_start = audio_start + num_audio_rows

    # The media clock. Without history this is the reference's `num_text`; with history the target
    # begins one history span later, so the carried frames sit just before it in time.
    media_origin = float(num_text) + float(len(reference_grids))
    target_origin = media_origin + (
        _temporal_position_span(history_latent_frames) if history_latent_frames else 0.0
    )

    # 1. The (t, h, w) grid. Text rows sit on the time axis at their row index and the media rows
    #    continue from there, so text length shifts the whole media clock.
    position_ids = np.zeros((seq_len, 3), dtype=np.float64)
    position_ids[:num_text, 0] = np.arange(num_text, dtype=np.float64)

    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, ph, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, pw, sqrt_area)
    hh, ww = np.meshgrid(height_grid, width_grid, indexing="ij")
    frame_grid = np.stack([hh.reshape(-1), ww.reshape(-1)], axis=-1)

    start = reference_start
    for index, ((rh, rw), count) in enumerate(zip(reference_grids, ref_rows)):
        r_area = np.sqrt(rh * rw)
        r_hh, r_ww = np.meshgrid(
            _spatial_position_grid(int(rh), ph, r_area), _spatial_position_grid(int(rw), pw, r_area),
            indexing="ij",
        )
        position_ids[start:start + count, 0] = float(num_text) + index
        position_ids[start:start + count, 1] = r_hh.reshape(-1)
        position_ids[start:start + count, 2] = r_ww.reshape(-1)
        start += count

    if history_latent_frames:
        history_pos = np.empty((history_latent_frames, rows_per_frame, 3), dtype=np.float64)
        history_pos[:, :, 0] = _temporal_position_grid(history_latent_frames, media_origin)[:, None]
        history_pos[:, :, 1:] = frame_grid[None]
        position_ids[condition_start:keyframe_start] = history_pos.reshape(-1, 3)

    for index, anchor in enumerate(keyframe_anchors):
        kind, pixel = parse_anchor(anchor, target_num_frames)
        if kind == "first":
            anchor_time = target_origin
        elif kind == "last":
            anchor_time = (
                target_origin + _temporal_position_span(num_latent_frames) - _ROPE_FRAME_RESCALE
            )
        else:
            anchor_time = target_origin + pixel * _ROPE_FRAME_RESCALE
        lo = keyframe_start + index * rows_per_frame
        hi = keyframe_start + (index + 1) * rows_per_frame
        position_ids[lo:hi, 0] = anchor_time
        position_ids[lo:hi, 1:] = frame_grid

    # Audio rows are channel-major and share the video's rotary clock: one unit per latent at
    # 40 latents/s equals 24 fps * 5/3. They carry no height coordinate and are pinned to the two
    # extremes of the width grid.
    def _place_audio(start: int, count: int, origin: float) -> None:
        if not count:
            return
        stop = start + count * AUDIO_CHANNELS
        times = origin + np.arange(count, dtype=np.float64)
        position_ids[start:stop, 0] = np.tile(times, AUDIO_CHANNELS)
        position_ids[start:stop, 2] = np.concatenate(
            [
                np.full(count, float(width_grid[0]), dtype=np.float64),
                np.full(count * (AUDIO_CHANNELS - 1), float(width_grid[-1]), dtype=np.float64),
            ]
        )

    _place_audio(
        audio_condition_start, int(history_audio_latents), media_origin + float(history_audio_offset)
    )
    _place_audio(audio_start, num_audio_latents, target_origin)

    video_pos = np.empty((num_latent_frames, rows_per_frame, 3), dtype=np.float64)
    video_pos[:, :, 0] = _temporal_position_grid(num_latent_frames, target_origin)[:, None]
    video_pos[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_pos.reshape(-1, 3)

    # 2. Row indices and modality tags. Conditioning rows lead each modality's index list.
    video_idx = np.concatenate(
        [np.arange(reference_start, audio_condition_start), np.arange(video_start, seq_len)]
    )
    audio_idx = np.arange(audio_condition_start, video_start)
    text_idx = np.arange(num_text)

    tags = np.empty(seq_len, dtype=np.int64)
    tags[text_idx] = text_tags
    tags[audio_idx] = TAG_AUDIO
    tags[video_idx] = TAG_VIDEO

    return PackedSequence(
        sequence_length=seq_len,
        position_ids=mx.array(position_ids.astype(np.float32)),
        token_tags=mx.array(tags.astype(np.int32)),
        video_indices=mx.array(video_idx.astype(np.int32)),
        audio_indices=mx.array(audio_idx.astype(np.int32)),
        text_indices=mx.array(text_idx.astype(np.int32)),
        num_condition_video_rows=num_condition_rows,
        num_condition_audio_rows=num_condition_audio_rows,
    )


# Timestep of a FROZEN target row: clean data (t = 1 is sigma 0 in this checkpoint's convention).
# A frozen row is part of the target — it sits on the target's own clock and is decoded with it —
# but it is never stepped: the soundtrack a lip-sync render must follow, or the picture a
# video-to-audio render must score.
FROZEN_TARGET_TIMESTEP = 1.0


def build_row_timesteps(
    layout: PackedSequence,
    video_timestep: float,
    audio_timestep: float,
    condition_video_timestep: float,
    condition_audio_timestep: float,
    frozen_video_rows: np.ndarray | None = None,
    frozen_audio_rows: np.ndarray | None = None,
    frozen_video_timestep: float = FROZEN_TARGET_TIMESTEP,
) -> tuple[mx.array, mx.array]:
    """Assign a timestep to every row and reduce to the transformer's ``(timestep, indices)`` pair.

    One forward serves rows at different noise levels: generated video and audio rows step down
    their own schedules while conditioning rows stay pinned at their noise-augmentation level. Text
    rows never reach an output head and inherit the video timestep.

    ``frozen_video_rows`` / ``frozen_audio_rows`` are boolean masks over the TARGET rows (in target
    order) that are held clean at :data:`FROZEN_TARGET_TIMESTEP` instead of following a schedule.

    Returns:
        The distinct timesteps, sorted, and the index of every row into them.
    """
    seq_len = layout.sequence_length
    rows = np.full(seq_len, float(video_timestep), dtype=np.float32)

    video_idx = np.asarray(layout.video_indices.tolist(), dtype=np.int64)
    audio_idx = np.asarray(layout.audio_indices.tolist(), dtype=np.int64)
    n_cond_v = layout.num_condition_video_rows
    n_cond_a = layout.num_condition_audio_rows

    rows[video_idx[:n_cond_v]] = float(condition_video_timestep)
    rows[audio_idx[n_cond_a:]] = float(audio_timestep)
    rows[audio_idx[:n_cond_a]] = float(condition_audio_timestep)
    if frozen_video_rows is not None:
        mask = np.asarray(frozen_video_rows, dtype=bool)
        target = video_idx[n_cond_v:]
        if mask.shape != target.shape:
            raise ValueError(f"frozen_video_rows {mask.shape} does not match {target.shape} targets")
        rows[target[mask]] = float(frozen_video_timestep)
    if frozen_audio_rows is not None:
        mask = np.asarray(frozen_audio_rows, dtype=bool)
        target = audio_idx[n_cond_a:]
        if mask.shape != target.shape:
            raise ValueError(f"frozen_audio_rows {mask.shape} does not match {target.shape} targets")
        rows[target[mask]] = FROZEN_TARGET_TIMESTEP

    distinct, inverse = np.unique(rows, return_inverse=True)
    return mx.array(distinct.astype(np.float32)), mx.array(inverse.astype(np.int32))
