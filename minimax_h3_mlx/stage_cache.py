"""Persist the clean Stage-A state of a staged render so a second pass can resume from it.

A staged render throws away everything but pixels: the clean video latents, the frozen audio
latents and the text conditioning all die with the process. That makes every second-pass
experiment — a latent hires-fix, a different tail schedule, a re-decode — pay for the whole first
pass again. It is also the missing half of a resume/preview feature: the expensive part of a render
is the denoise, and the denoise is exactly what this file stores.

The payload is deliberately the *packed transformer rows*, not pixels and not a VAE posterior:

* ``video_rows``  — generated rows only, conditioning rows excluded. Conditioning rows are a
  function of the keyframe and the target canvas, so a second pass at a different canvas has to
  rebuild them anyway; storing them would invite reusing rows that belong to the old grid, which is
  the exact identity-warp failure the community upscaler documents.
* ``audio_rows``  — clean, at sigma 0. A second pass that freezes audio simply feeds these back.
* ``embeds`` / ``text_tags`` — the text (and vision) conditioning, so a resumed pass never reloads
  the 27 GB text encoder.

Every array carries a SHA-256 taken at write time and re-checked at read time, so "the cache is
what Stage A produced" is proven rather than assumed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

CACHE_VERSION = 1
ARRAY_NAMES = ("video_rows", "audio_rows", "embeds", "text_tags")


def digest(array: np.ndarray) -> str:
    """SHA-256 of an array's exact bytes, dtype and shape included."""
    contiguous = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(str(contiguous.dtype).encode())
    hasher.update(str(contiguous.shape).encode())
    hasher.update(contiguous.tobytes())
    return hasher.hexdigest()


def save_stage_a(
    path: str | Path,
    *,
    video_rows: np.ndarray,
    audio_rows: np.ndarray,
    embeds: np.ndarray,
    text_tags: np.ndarray,
    meta: dict,
) -> dict:
    """Write one Stage-A payload and return the metadata block that was stored with it."""
    arrays = {
        "video_rows": np.asarray(video_rows, dtype=np.float32),
        "audio_rows": np.asarray(audio_rows, dtype=np.float32),
        "embeds": np.asarray(embeds, dtype=np.float32),
        "text_tags": np.asarray(text_tags, dtype=np.int64),
    }
    stored = dict(meta)
    stored["cache_version"] = CACHE_VERSION
    stored["digests"] = {name: digest(array) for name, array in arrays.items()}
    stored["shapes"] = {name: list(array.shape) for name, array in arrays.items()}

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Uncompressed: the payload is tens of megabytes and this sits inside a render's wall clock.
    np.savez(path, meta=np.array(json.dumps(stored)), **arrays)
    return stored


def load_stage_a(path: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Read a Stage-A payload back and verify every array against its stored digest."""
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"].item()))
    if int(meta.get("cache_version", -1)) != CACHE_VERSION:
        raise ValueError(
            f"{path} is a version {meta.get('cache_version')} Stage-A cache; this build writes "
            f"version {CACHE_VERSION}."
        )

    arrays = {name: np.asarray(data[name]) for name in ARRAY_NAMES}
    mismatched = [
        name for name in ARRAY_NAMES if digest(arrays[name]) != meta["digests"].get(name)
    ]
    if mismatched:
        raise ValueError(
            f"{path} failed its digest check for {', '.join(mismatched)}: the cache does not hold "
            "the bytes Stage A wrote. Re-render Stage A rather than refining corrupt latents."
        )
    return meta, arrays
