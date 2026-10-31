"""Read RefMod files (Luisacaotica/ComfyUI-MiniMaxH3Mod) into ref2va conditioning rows.

A RefMod is a saved, already-VAE-normalised video latent ``(1, 24, T, H, W)`` plus a JSON
``refmod_meta`` header. Standalone files (format 4) carry one tensor, ``latent``; bundles
(format 5) carry ``ref_<i>`` per member. Only visual members (``image`` / ``video``) are read here;
audio members are skipped from bundles and refused as a standalone file.

``strength < 1`` weakens a reference the way the pack does: mix toward a heavily blurred copy of
itself (adaptive-average-pool the H/W grid down by 8, bilinear back up), never toward noise or
zero -- both fall off the latent manifold. :func:`blur_latent` reproduces torch's
``adaptive_avg_pool3d`` + trilinear ``align_corners=False`` exactly for the T-preserving case.

Nothing here touches the model: it hands back packed rows and the geometry
:func:`ref2va_packing.build_ref2va_packed_sequence` needs.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from .packing import patchify_video_latents
from .ref2va_packing import RefGeometry

BLUR_FACTOR = 8
LATENT_CHANNELS = 24


@dataclass
class RefModLatent:
    name: str
    kind: str  # "image" | "video"
    latent: np.ndarray  # (1, 24, T, H, W) float32

    @property
    def geometry(self) -> RefGeometry:
        _, _, t, h, w = self.latent.shape
        return RefGeometry(self.kind, t, h, w)


def read_header(path: Path) -> tuple[dict, dict]:
    """``(tensor header, refmod_meta)`` -- no tensor data is read."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    meta_raw = (header.pop("__metadata__", None) or {}).get("refmod_meta")
    if meta_raw is None:
        raise ValueError(f"{path.name}: not a RefMod (no refmod_meta header)")
    return header, json.loads(meta_raw)


def _checked(latent: mx.array, label: str) -> np.ndarray:
    arr = np.array(latent.astype(mx.float32))
    if arr.ndim != 5 or arr.shape[0] != 1 or arr.shape[1] != LATENT_CHANNELS:
        raise ValueError(f"{label}: expected a (1, {LATENT_CHANNELS}, T, H, W) latent, got {arr.shape}")
    if arr.shape[3] % 2 or arr.shape[4] % 2:
        raise ValueError(f"{label}: latent H/W must be even for the (1, 2, 2) patch, got {arr.shape[3:]}")
    return arr


def load_refmod(path: str | Path) -> list[RefModLatent]:
    """Every visual member of a RefMod file, in file order."""
    path = Path(path)
    if path.suffix != ".safetensors":
        path = path.with_name(path.name + ".safetensors")
    header, meta = read_header(path)
    tensors = mx.load(str(path))
    if meta.get("kind") == "bundle":
        out = []
        for i, member in enumerate(meta.get("members", [])):
            if member.get("kind") in ("image", "video"):
                out.append(RefModLatent(member.get("name", f"{path.stem}#{i}"), member["kind"],
                                        _checked(tensors[f"ref_{i}"], f"{path.name}[{i}]")))
        if not out:
            raise ValueError(f"{path.name}: bundle has no visual members (audio RefMods are not supported)")
        return out
    kind = meta.get("kind")
    if kind == "audio":
        raise ValueError(f"{path.name}: audio RefMods are not supported (voice transfer is unreliable)")
    if kind not in ("image", "video"):
        raise ValueError(f"{path.name}: unknown RefMod kind {kind!r}")
    return [RefModLatent(meta.get("name", path.stem), kind, _checked(tensors["latent"], path.name))]


def _adaptive_bins(size: int, out: int) -> list[tuple[int, int]]:
    return [(math.floor(i * size / out), math.ceil((i + 1) * size / out)) for i in range(out)]


def _interp_axis(x: np.ndarray, axis: int, out: int) -> np.ndarray:
    """Linear interpolation along ``axis`` with torch's ``align_corners=False`` mapping."""
    size = x.shape[axis]
    scale = size / out
    src = np.maximum((np.arange(out) + 0.5) * scale - 0.5, 0.0)
    lo = np.minimum(np.floor(src).astype(int), size - 1)
    hi = np.minimum(lo + 1, size - 1)
    frac = (src - lo).astype(x.dtype)
    shape = [1] * x.ndim
    shape[axis] = out
    frac = frac.reshape(shape)
    return np.take(x, lo, axis=axis) * (1 - frac) + np.take(x, hi, axis=axis) * frac


def blur_latent(z: np.ndarray, factor: int = BLUR_FACTOR) -> np.ndarray:
    """Spatial low-pass of a ``(1, C, T, H, W)`` latent: pool H/W down by ``factor``, bilinear back."""
    _, _, t, h, w = z.shape
    sh, sw = max(1, h // factor), max(1, w // factor)
    z = z.astype(np.float32)
    pooled = np.empty(z.shape[:3] + (sh, sw), dtype=np.float32)
    for i, (h0, h1) in enumerate(_adaptive_bins(h, sh)):
        for j, (w0, w1) in enumerate(_adaptive_bins(w, sw)):
            pooled[..., i, j] = z[..., h0:h1, w0:w1].mean(axis=(-2, -1))
    return _interp_axis(_interp_axis(pooled, 3, h), 4, w)


def apply_strength(latent: np.ndarray, strength: float) -> np.ndarray:
    """``strength`` 1.0 = untouched; lower mixes toward the blurred copy."""
    if strength >= 1.0:
        return latent
    return strength * latent + (1.0 - strength) * blur_latent(latent)


def parse_spec(spec: str) -> tuple[str, float]:
    """``PATH`` or ``PATH:STRENGTH`` -> (path, strength). A trailing ``:<number>`` only."""
    head, sep, tail = spec.rpartition(":")
    if sep:
        try:
            strength = float(tail)
        except ValueError:
            return spec, 1.0
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"RefMod strength must be 0..1, got {strength} in {spec!r}")
        return head, strength
    return spec, 1.0


def load_refmod_rows(specs: list[str], patch_size: tuple[int, int, int]):
    """Packed conditioning rows for ``--refmod`` specs, in request order.

    Returns ``(rows (N, C*patch), geometries, names)``. A strength-0 spec is dropped entirely.
    """
    geoms: list[RefGeometry] = []
    names: list[str] = []
    rows = []
    for spec in specs:
        path, strength = parse_spec(spec)
        if strength <= 0.0:
            continue
        for mod in load_refmod(path):
            latent = apply_strength(mod.latent, strength)
            rows.append(patchify_video_latents(mx.array(latent), patch_size))
            geoms.append(mod.geometry)
            names.append(f"{mod.name}@{strength:g}")
    if not rows:
        raise ValueError("every --refmod had strength 0; nothing to condition on")
    return mx.concatenate(rows), geoms, names
