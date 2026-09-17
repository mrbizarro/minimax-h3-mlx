"""Refuse to call a broken render a success.

2026-09-17: a Fast render with an F16 user LoRA overflowed inside the runtime adapter, every
latent became NaN, the VAE decoded a flat black clip, and the run still reported ``done``. Two
cheap checks close that class for good:

* :func:`nonfinite_flag` — one lazy scalar evaluated with each denoise step, so a NaN/inf latent
  stops the run at the step it appears instead of four minutes later.
* :func:`check_decoded_video` — a flat decode (every pixel of every frame within a couple of code
  values) is never a clip anyone asked for; it is what non-finite latents decode to.

Both raise :class:`RenderIntegrityError`, whose ``kind`` lands in the metrics JSON so the panel
can show the message instead of a traceback tail.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

NONFINITE_LATENTS = "nonfinite_latents"
BLANK_VIDEO = "blank_video"


class RenderIntegrityError(RuntimeError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def nonfinite_flag(*arrays):
    """Lazy mx scalar: True when any element of any array is NaN or +-inf."""
    import mlx.core as mx

    flag = None
    for array in arrays:
        bad = mx.any(mx.logical_not(mx.isfinite(array)))
        flag = bad if flag is None else mx.logical_or(flag, bad)
    return flag if flag is not None else mx.array(False)


def adapter_dtypes(path: str | Path) -> list[str]:
    """Distinct tensor dtypes a safetensors adapter stores, from its header alone."""
    try:
        with open(path, "rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
    except Exception:
        return []
    header.pop("__metadata__", None)
    return sorted({str(v.get("dtype")) for v in header.values() if isinstance(v, dict)})


def describe_adapters(lora_stack) -> str:
    parts = []
    for path, scale in lora_stack or []:
        dtypes = "/".join(adapter_dtypes(path)) or "?"
        parts.append(f"{Path(path).name} @ {float(scale):g} ({dtypes})")
    return ", ".join(parts)


def nonfinite_error(where: str, lora_stack=None) -> RenderIntegrityError:
    adapters = describe_adapters(lora_stack)
    if adapters:
        cause = (f"The most likely cause is a LoRA adapter: {adapters}. Render again without the "
                 f"user LoRA (or at a lower strength) to confirm, and report the file.")
    else:
        cause = "No LoRA was loaded, so this is an engine fault — please report it with the log."
    return RenderIntegrityError(
        NONFINITE_LATENTS,
        f"H3 render failed: the latents became NaN/inf {where}, so the clip would decode "
        f"black. Nothing was saved. {cause}",
    )


def check_decoded_video(video: np.ndarray, lora_stack=None, tolerance: int = 2) -> None:
    """Raise when the decoded uint8 clip is one flat colour from first frame to last."""
    if video is None or getattr(video, "size", 0) == 0:
        raise RenderIntegrityError(BLANK_VIDEO, "H3 render failed: the decoder returned no frames.")
    # Stride-sample the spatial grid: the verdict only needs the global min/max.
    sample = np.asarray(video)[:, ::4, ::4]
    spread = int(sample.max()) - int(sample.min())
    if spread > tolerance:
        return
    adapters = describe_adapters(lora_stack)
    raise RenderIntegrityError(
        BLANK_VIDEO,
        f"H3 render failed: every decoded frame is a single flat colour (pixel value "
        f"{int(sample.min())}-{int(sample.max())}), which is what broken latents decode to. "
        f"Nothing was saved."
        + (f" LoRA adapters in this render: {adapters} — try again without them." if adapters else ""),
    )
