"""The streamed decode-only video VAE decodes byte-identically to the resident loader.

A tiny VideoVAE (the real tiling and chunking maths, shrunk widths) is saved in the compact
single-file layout; the resident loader and the streamed loader both read it and decode the same
latents. Equality is on bytes, never a tolerance.
"""

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx import load as load_mod  # noqa: E402
from minimax_h3_mlx.vae_stream import (  # noqa: E402
    _StreamedViTBlock,
    load_compact_video_vae_encoder_only,
    load_compact_video_vae_streamed,
)
from minimax_h3_mlx.video_vae import VideoVAE, VideoVAEConfig  # noqa: E402


def tiny_config() -> VideoVAEConfig:
    return VideoVAEConfig(
        in_channels=3, out_channels=3, latent_channels=8, block_out_channels=(32, 32),
        layers_per_block=1, spatial_downsample_factors=(2, 2, 2, 2),
        temporal_downsample_factors=(1, 2, 2, 1), decoder_num_layers=3,
        decoder_num_attention_heads=2, decoder_attention_head_dim=16,
        latents_mean=tuple([0.0] * 8), latents_std=tuple([1.0] * 8),
    )


def write_compact(path: Path, config: VideoVAEConfig) -> None:
    mx.random.seed(3)
    model = VideoVAE(config)
    weights = {}
    for key, value in tree_flatten(model.parameters()):
        if key.endswith(("scale1", "scale2")):
            # Initialised to zero; make the blocks actually do something.
            value = mx.random.normal(value.shape) * 0.1
        if value.ndim == 5:
            value = value.transpose(0, 4, 1, 2, 3)        # the file keeps torch's conv layout
        weights[key] = value.astype(mx.float16) if value.dtype == mx.float32 else value
    mx.eval(weights)
    mx.save_safetensors(str(path), weights)


def decode_bytes(model, z) -> bytes:
    return np.array(model.decode(z).astype(mx.float32)).tobytes()


class StreamedVAETests(unittest.TestCase):
    def test_streamed_decode_is_byte_identical(self):
        config = tiny_config()
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(load_mod, "load_compact_video_vae_config", lambda path: config):
            path = Path(tmp) / "video_vae.safetensors"
            write_compact(path, config)
            z = mx.random.normal((1, 8, 7, 8, 24), key=mx.random.key(5))
            for dtype in ("float32", "float16"):
                for batch in ("1", "8"):
                    with self.subTest(dtype=dtype, batch=batch), mock.patch.dict(
                            os.environ, {"H3_VAE_DTYPE": dtype, "H3_VAE_BATCH": batch}):
                        resident = load_mod.load_compact_video_vae(path)
                        streamed, stream = load_compact_video_vae_streamed(path)
                        try:
                            a, b = decode_bytes(resident, z), decode_bytes(streamed, z)
                            self.assertEqual(hashlib.sha256(a).hexdigest(), hashlib.sha256(b).hexdigest())
                            # No block weight and no encoder weight is held between calls.
                            keys = [k for k, _ in tree_flatten(streamed.parameters())]
                            self.assertFalse(any(k.startswith("decoder.transformer_blocks.") for k in keys))
                            self.assertFalse(any(k.startswith("encoder.") for k in keys))
                            self.assertTrue(all(isinstance(blk, _StreamedViTBlock)
                                                for blk in streamed.decoder.transformer_blocks))
                            info = stream.summary()
                            self.assertGreaterEqual(info["block_loads"], config.decoder_num_layers)
                            self.assertGreater(info["read_ahead_hits"], 0)
                            # A second decode rereads and still matches.
                            self.assertEqual(decode_bytes(streamed, z), a)
                        finally:
                            stream.close()
                        del resident, streamed

    def test_encoder_only_encode_is_byte_identical(self):
        config = tiny_config()
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(load_mod, "load_compact_video_vae_config", lambda path: config):
            path = Path(tmp) / "video_vae.safetensors"
            write_compact(path, config)
            x = mx.random.uniform(shape=(1, 3, 17, 64, 96), key=mx.random.key(8))
            resident = load_mod.load_compact_video_vae(path)
            enc_only = load_compact_video_vae_encoder_only(path)
            a = np.array(resident.encode(x).astype(mx.float32)).tobytes()
            b = np.array(enc_only.encode(x).astype(mx.float32)).tobytes()
            self.assertEqual(hashlib.sha256(a).hexdigest(), hashlib.sha256(b).hexdigest())
            keys = [k for k, _ in tree_flatten(enc_only.parameters())]
            self.assertFalse(any(k.startswith("decoder.") for k in keys))


if __name__ == "__main__":
    unittest.main()


def test_streamed_block_takes_the_mixed_precision_argument_and_is_never_compiled():
    """The engine head compiles decoder blocks as fn(x, cos, sin); a streamed block must opt out
    (its weights change on every call) and must accept `compute`. Broke every 16 GB run's decode."""
    import inspect

    from minimax_h3_mlx.vae_stream import _StreamedViTBlock

    params = inspect.signature(_StreamedViTBlock.__call__).parameters
    assert "compute" in params
    assert _StreamedViTBlock._streamed is True
    import minimax_h3_mlx.video_vae as vv

    assert "_streamed" in inspect.getsource(vv.ViTDecoder3d.__call__)
