"""The encoder's convolutions computed a few output frames at a time are bit-identical.

The video VAE's encoder is causal in time and normalises each frame on its own, so a 3D
convolution's output frames can be made in slices (set_conv_time_chunk) without changing a
value. This holds the low-memory carried-context encode to that: the latents must be the same
bytes for every slice size, on the clip shapes a chain and Add sound really encode.

    python3 tests/test_vae_encode_chunk.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from minimax_h3_mlx import video_vae as vv  # noqa: E402
from test_video_vae_smoke import tiny_config  # noqa: E402


def encode(model, x, chunk):
    vv.set_conv_time_chunk(chunk)
    try:
        out = model._encode_clip(x)
        mx.eval(out)
        return np.array(out)
    finally:
        vv.set_conv_time_chunk(0)


def test_chunked_encode_is_bit_identical():
    mx.random.seed(1)
    model = vv.VideoVAE(tiny_config())
    mx.eval(model.parameters())
    for frames, h, w in ((17, 64, 96), (17, 288, 320), (1, 64, 64)):
        x = mx.random.uniform(-1, 1, (1, frames, h, w, 3))
        ref = encode(model, x, 0)
        for chunk in (5, 6, 8, 16):
            got = encode(model, x, chunk)
            assert got.shape == ref.shape, (frames, chunk)
            assert np.array_equal(got, ref), (frames, h, w, chunk, float(np.abs(got - ref).max()))


def test_chunk_setting_is_restored_off():
    vv.set_conv_time_chunk(4)
    vv.set_conv_time_chunk(0)
    assert vv._CONV_TIME_CHUNK == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
