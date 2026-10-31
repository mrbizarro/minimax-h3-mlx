"""The ``ref2va`` packed layout: ``[text | reference blocks | target audio | target video]``.

A NumPy/MLX port of ``reference/diffusers/modular/packing_ref2va.py::build_ref2va_packed_sequence``,
position ids and modality tags only -- it takes the reference latents' GEOMETRY, never the latents
themselves, so a reference may come from the video/audio VAEs or from a saved RefMod alike.

Blocks are packed in request order. A video reference packs its soundtrack rows immediately before
its own video rows. Order is semantic: it advances the shared rotary clock -- an image takes one
slot, an audio reference as many as it has latents, a video the longer of its two spans.

Unlike a keyframe, a reference never binds the target geometry: each carries its own aspect-
normalized spatial grid. Returns the runner's own :class:`PackedSequence`, so the conditioning-row
timestep plumbing (``build_row_timesteps``) applies unchanged: reference video rows lead
``video_indices`` and reference audio rows lead ``audio_indices``.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .config import TAG_AUDIO, TAG_VIDEO
from .packing import (
    AUDIO_CHANNELS,
    PackedSequence,
    _ROPE_FRAME_RESCALE,
    _ROPE_FRAMES_PER_LATENT,
    _spatial_position_grid,
    _temporal_position_grid,
)

# Documented per-request limits of the omni-reference task.
MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_AUDIOS = 3
MAX_REFERENCES = 12


@dataclass(frozen=True)
class RefGeometry:
    """Latent geometry of one reference, in packed order.

    ``kind`` is "image", "video" or "audio". ``has_audio`` is implied for "audio" and optional for
    "video" (a soundtrack). Visual dims are latent (pre-patch) sizes; ``num_audio_latents`` is per
    stereo channel.
    """

    kind: str
    num_latent_frames: int = 1
    latent_height: int = 0
    latent_width: int = 0
    num_audio_latents: int = 0

    @property
    def has_audio(self) -> bool:
        return self.kind == "audio" or (self.kind == "video" and self.num_audio_latents > 0)

    def num_video_rows(self, patch_h: int, patch_w: int) -> int:
        if self.kind == "audio":
            return 0
        return self.num_latent_frames * (self.latent_height // patch_h) * (self.latent_width // patch_w)

    @property
    def num_audio_rows(self) -> int:
        return self.num_audio_latents * AUDIO_CHANNELS if self.has_audio else 0


def validate_references(references: list[RefGeometry]) -> None:
    if not references:
        raise ValueError("`ref2va` needs at least one reference.")
    counts = {"image": 0, "video": 0, "audio": 0}
    for index, ref in enumerate(references):
        if ref.kind not in counts:
            raise ValueError(f"Reference {index} must be an 'image', 'video' or 'audio', got {ref.kind!r}.")
        counts[ref.kind] += 1
    for kind, limit in (
        ("image", MAX_REFERENCE_IMAGES),
        ("video", MAX_REFERENCE_VIDEOS),
        ("audio", MAX_REFERENCE_AUDIOS),
    ):
        if counts[kind] > limit:
            raise ValueError(f"ref2va takes at most {limit} {kind} references, got {counts[kind]}.")
    if len(references) > MAX_REFERENCES:
        raise ValueError(f"ref2va takes at most {MAX_REFERENCES} references, got {len(references)}.")


def _reference_time_span(num_latent_frames: int) -> float:
    """Rotary time a video reference advances the clock by -- summed SEQUENTIALLY in float64.

    Deliberately not :func:`packing._temporal_position_span` (a pairwise NumPy sum): the reference
    keeps both orders, one per call site, and they differ in the last ulp from 16 frames on.
    """
    return sum(
        _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[i % len(_ROPE_FRAMES_PER_LATENT)]
        for i in range(num_latent_frames)
    )


def _frame_grid(latent_height: int, latent_width: int, patch_h: int, patch_w: int):
    """``(h, w)`` rotary coordinates of one latent frame, and the width axis they were built from."""
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    hh, ww = np.meshgrid(height_grid, width_grid, indexing="ij")
    return np.stack([hh.reshape(-1), ww.reshape(-1)], axis=-1), width_grid


def _fill_audio(position_ids, rows: slice, count: int, rotary_time: float, width_grid) -> None:
    """One channel-major audio block: no height coordinate, pinned to the width grid's two ends."""
    time = rotary_time + np.arange(count, dtype=np.float64)
    position_ids[rows, 0] = np.tile(time, AUDIO_CHANNELS)
    position_ids[rows, 2] = np.concatenate(
        [
            np.full(count, float(width_grid[0]), dtype=np.float64),
            np.full(count, float(width_grid[-1]), dtype=np.float64),
        ]
    )


def build_ref2va_packed_sequence(
    text_token_tags,
    references: list[RefGeometry],
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
) -> PackedSequence:
    """Build the ``[text | reference blocks | target audio | target video]`` layout.

    ``text_token_tags`` tags every text row (1, except a reference's vision-block rows, 0).
    """
    validate_references(references)
    _, patch_h, patch_w = patch_size
    text_tags = np.asarray(text_token_tags, dtype=np.int64)
    num_text = int(text_tags.shape[0])

    target_frame_grid, target_width_grid = _frame_grid(latent_height, latent_width, patch_h, patch_w)
    num_target_video = num_latent_frames * (latent_height // patch_h) * (latent_width // patch_w)
    num_target_audio = num_audio_latents * AUDIO_CHANNELS
    num_ref_video = sum(r.num_video_rows(patch_h, patch_w) for r in references)
    num_ref_audio = sum(r.num_audio_rows for r in references)
    seq_len = num_text + num_ref_video + num_ref_audio + num_target_audio + num_target_video

    position_ids = np.zeros((seq_len, 3), dtype=np.float64)
    position_ids[:num_text, 0] = np.arange(num_text, dtype=np.float64)

    video_indices: list[np.ndarray] = []
    audio_indices: list[np.ndarray] = []
    cursor = num_text
    rotary_time = float(num_text)  # the shared audio/video clock, starting where the text ends
    for ref in references:
        if ref.kind == "image":
            n = ref.num_video_rows(patch_h, patch_w)
            rows = slice(cursor, cursor + n)
            cursor = rows.stop
            video_indices.append(np.arange(rows.start, rows.stop))
            frame_grid, _ = _frame_grid(ref.latent_height, ref.latent_width, patch_h, patch_w)
            position_ids[rows, 0] = rotary_time
            position_ids[rows, 1:] = frame_grid
            rotary_time += 1.0  # an image takes one integer slot, not a latent frame's 5/3 units
        elif ref.kind == "audio":
            rows = slice(cursor, cursor + ref.num_audio_rows)
            cursor = rows.stop
            audio_indices.append(np.arange(rows.start, rows.stop))
            _fill_audio(position_ids, rows, ref.num_audio_latents, rotary_time, target_width_grid)
            rotary_time += float(ref.num_audio_latents)
        else:  # video: soundtrack rows first, sharing the video's origin
            audio_rows = slice(cursor, cursor + ref.num_audio_rows)
            video_rows = slice(audio_rows.stop, audio_rows.stop + ref.num_video_rows(patch_h, patch_w))
            cursor = video_rows.stop
            if ref.num_audio_rows:
                audio_indices.append(np.arange(audio_rows.start, audio_rows.stop))
            video_indices.append(np.arange(video_rows.start, video_rows.stop))
            frame_grid, width_grid = _frame_grid(ref.latent_height, ref.latent_width, patch_h, patch_w)
            if ref.num_audio_rows:
                _fill_audio(position_ids, audio_rows, ref.num_audio_latents, rotary_time, width_grid)
            frame_time = _temporal_position_grid(ref.num_latent_frames, rotary_time)
            position_ids[video_rows, 0] = np.repeat(frame_time, frame_grid.shape[0])
            position_ids[video_rows, 1:] = np.tile(frame_grid, (ref.num_latent_frames, 1))
            rotary_time += max(float(ref.num_audio_latents), _reference_time_span(ref.num_latent_frames))

    # Generated rows: target audio and video share the origin the reference blocks left behind.
    audio_start = cursor
    video_start = audio_start + num_target_audio
    _fill_audio(position_ids, slice(audio_start, video_start), num_audio_latents, rotary_time, target_width_grid)
    frame_time = _temporal_position_grid(num_latent_frames, rotary_time)
    position_ids[video_start:, 0] = np.repeat(frame_time, target_frame_grid.shape[0])
    position_ids[video_start:, 1:] = np.tile(target_frame_grid, (num_latent_frames, 1))

    video_idx = np.concatenate(video_indices + [np.arange(video_start, seq_len)])
    audio_idx = np.concatenate(audio_indices + [np.arange(audio_start, video_start)])
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
        num_condition_video_rows=num_ref_video,
        num_condition_audio_rows=num_ref_audio,
    )
