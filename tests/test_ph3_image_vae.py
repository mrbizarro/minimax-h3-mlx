"""P-H3-Image decoder: MLX port vs Pruna's diffusers reference, on a tiny random config.

The reference (``AutoencoderKLPH3Image``) is only needed for validation. Point ``PH3_DIFFUSERS`` at
a diffusers tree that contains ``autoencoder_kl_p_h3_image.py``; without it the parity tests skip
and the pure-numpy checks (key conversion, tiling geometry) still run.

    PH3_DIFFUSERS=/tmp/dfx_target python tests/test_ph3_image_vae.py
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.ph3_image_vae import (  # noqa: E402
    PH3ImageVAE,
    convert_decoder_state,
    image_vae_config,
)

TINY = dict(
    in_channels=3, out_channels=3, latent_channels=24,
    block_out_channels=[32, 32, 32, 32, 32, 32], layers_per_block=1,
    spatial_downsample_factors=[2, 2, 2, 2, 1, 1], temporal_downsample_factors=[1, 2, 2, 1, 1, 1],
    norm_num_groups=32, norm_eps=1e-6,
    decoder_num_layers=2, decoder_num_attention_heads=4, decoder_attention_head_dim=24,
    decoder_num_register_tokens=4, decoder_ffn_mult=4, decoder_rope_theta=100.0,
    decoder_rope_dim_ratio=0.75, decoder_norm_eps=1e-5,
    latents_mean=[0.1] * 24, latents_std=[1.5] * 24,
)


def _reference(cfg: dict):
    path = os.environ.get("PH3_DIFFUSERS")
    if not path:
        return None
    sys.path.insert(0, path)
    try:
        import huggingface_hub.utils as u
        import httpx

        if not hasattr(u, "httpx"):
            u.httpx = httpx
        import torch
        from diffusers.models.autoencoders.autoencoder_kl_p_h3_image import AutoencoderKLPH3Image
    except Exception:  # pragma: no cover - environment dependent
        return None
    torch.manual_seed(0)
    model = AutoencoderKLPH3Image(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items()}).float().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("scale1") or name.endswith("scale2"):
                p.copy_(torch.randn_like(p) * 0.3)  # init is 0, which would hide every residual branch
            elif "register_tokens" in name:
                p.copy_(torch.randn_like(p) * 0.2)
    return model


class TestKeyConversion(unittest.TestCase):
    def test_fused_qkv_is_per_head_interleaved(self):
        from minimax_h3_mlx.ph3_image_vae import _fuse_qkv

        heads, hd, din = 2, 3, 5
        q, k, v = (np.random.randn(heads * hd, din).astype(np.float32) for _ in range(3))
        fused = _fuse_qkv(q, k, v, heads).reshape(heads, 3, hd, din)
        np.testing.assert_array_equal(fused[1, 0], q[hd : 2 * hd])
        np.testing.assert_array_equal(fused[1, 1], k[hd : 2 * hd])
        np.testing.assert_array_equal(fused[0, 2], v[:hd])

    def test_every_expected_key_is_produced(self):
        cfg = image_vae_config(TINY)
        model = PH3ImageVAE(cfg)
        from mlx.utils import tree_flatten

        expected = {k for k, _ in tree_flatten(model.parameters())}
        dim = cfg.decoder_dim
        t = {
            "decoder.proj_in.weight": np.zeros((dim, 24), np.float32), "decoder.proj_in.bias": np.zeros(dim, np.float32),
            "decoder.register_tokens": np.zeros((1, 4, dim), np.float32),
            "post_quant_conv.weight": np.zeros((24, 24, 1, 1, 1), np.float32), "post_quant_conv.bias": np.zeros(24, np.float32),
            "decoder.norm_out.weight": np.zeros(dim, np.float32), "decoder.norm_out.bias": np.zeros(dim, np.float32),
            "decoder.proj_out.weight": np.zeros((768, dim), np.float32), "decoder.proj_out.bias": np.zeros(768, np.float32),
        }
        for i in range(cfg.decoder_num_layers):
            p = f"decoder.transformer_blocks.{i}."
            for n in ("norm1.weight", "norm2.weight", "scale1", "scale2"):
                t[p + n] = np.zeros(dim, np.float32)
            for n in "qkv":
                t[p + f"attn.to_{n}.weight"] = np.zeros((dim, dim), np.float32)
                t[p + f"attn.to_{n}.bias"] = np.zeros(dim, np.float32)
            t[p + "attn.to_out.0.weight"] = np.zeros((dim, dim), np.float32)
            t[p + "attn.to_out.0.bias"] = np.zeros(dim, np.float32)
            t[p + "ff.net.0.proj.weight"] = np.zeros((8 * dim, dim), np.float32)
            t[p + "ff.net.0.proj.bias"] = np.zeros(8 * dim, np.float32)
            t[p + "ff.net.2.weight"] = np.zeros((dim, 4 * dim), np.float32)
            t[p + "ff.net.2.bias"] = np.zeros(dim, np.float32)
        out = convert_decoder_state(t, cfg)
        self.assertEqual(set(out), expected)


class TestParity(unittest.TestCase):
    def setUp(self):
        self.ref = _reference(TINY)
        if self.ref is None:
            self.skipTest("set PH3_DIFFUSERS to a diffusers tree with the P-H3-Image VAE")

    def _mlx_from_ref(self):
        import torch  # noqa: F401

        cfg = image_vae_config(TINY)
        sd = {k: v.detach().float().numpy() for k, v in self.ref.state_dict().items()}
        converted = convert_decoder_state(sd, cfg)
        model = PH3ImageVAE(cfg)
        from mlx.utils import tree_unflatten

        model.update(tree_unflatten([(k, mx.array(v)) for k, v in converted.items()]))
        mx.eval(model.parameters())
        return model

    def test_untiled_decode_matches_reference(self):
        import torch

        model = self._mlx_from_ref()
        model.use_tiling = False
        self.ref.use_tiling = False
        z = np.random.RandomState(1).randn(1, 24, 1, 6, 5).astype(np.float32)
        with torch.no_grad():
            want = self.ref.decode(torch.from_numpy(z), return_dict=False)[0].numpy()  # (1,3,1,H,W)
        got = model.decode_frame(mx.array(z.transpose(0, 2, 3, 4, 1)), compute=None)
        got = np.array(got).transpose(0, 4, 1, 2, 3)
        self.assertEqual(got.shape, want.shape)
        np.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-3)

    def test_tiled_decode_matches_reference(self):
        import torch

        model = self._mlx_from_ref()
        for m in (model, self.ref):
            m.use_tiling = True
            m.tile_sample_min_height = m.tile_sample_min_width = 64
            m.tile_sample_min_overlap_height = m.tile_sample_min_overlap_width = 32
        z = np.random.RandomState(2).randn(1, 24, 1, 9, 11).astype(np.float32)
        with torch.no_grad():
            want = self.ref.decode(torch.from_numpy(z), return_dict=False)[0].numpy()
        got = np.array(model.decode_frame(mx.array(z.transpose(0, 2, 3, 4, 1)), compute=None)).transpose(0, 4, 1, 2, 3)
        self.assertEqual(got.shape, want.shape)
        np.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
