"""Prepare reference images using the model's aspect and sizing rules."""
from __future__ import annotations

def reference_size(width: int, height: int, short_edge: int = 2048) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"a reference image must have a positive size, got {width}x{height}")
    if max(width, height) > 4 * min(width, height):
        raise ValueError(f"a reference image must be within 1:4 and 4:1, got {width}x{height}")
    scale = short_edge / min(width, height)
    return (max(32, int(round(width * scale / 32)) * 32), max(32, int(round(height * scale / 32)) * 32))


def prepare_reference_image(path, short_edge: int = 2048):
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = reference_size(image.size[0], image.size[1], short_edge)
    if image.size == (w, h):
        return image
    return image.resize((w, h), Image.Resampling.LANCZOS)


