"""MiniMax-H3-Image edit: reference sizing, the encoder loader and a tiny-DiT denoise with references."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from minimax_h3_mlx import h3_image as H  # noqa: E402
from test_h3_image import RELEASE_VAE_CONFIG, _tiny  # noqa: E402


class TestReferenceSizing(unittest.TestCase):
    def test_inside_the_bounds_keeps_the_original_pixels(self):
        self.assertIsNone(H.bounded_reference_size(1536, 1200, 1024, 2048))
        self.assertIsNone(H.bounded_reference_size(1024, 1024, 1024, 1024))

    def test_small_scales_up_large_scales_down_to_the_nearer_bound(self):
        w, h = H.bounded_reference_size(800, 600, 1024, 2048)
        self.assertEqual(h, 1024)
        self.assertEqual(w % 32, 0)
        self.assertAlmostEqual(w / h, 4 / 3, delta=0.02)
        w, h = H.bounded_reference_size(6000, 4000, 1024, 2048)
        self.assertEqual(h, 2048)
        self.assertEqual(w % 32, 0)
        # Qwen sees every reference with a 1024 px short edge
        self.assertEqual(min(H.bounded_reference_size(3000, 2000, *H.QWEN_SHORT_EDGE)), 1024)

    def test_grid_is_even_16px_cells(self):
        self.assertEqual(H.reference_grid(1500, 1100), (68, 92))
        gh, gw = H.reference_grid(1030, 1042)
        self.assertEqual((gh % 2, gw % 2), (0, 0))

    def test_vae_copy_is_cropped_top_left_to_the_grid(self):
        from PIL import Image

        img = Image.fromarray(np.arange(1100 * 1500 * 3, dtype=np.uint32).astype(np.uint8).reshape(1100, 1500, 3))
        pixels, (gh, gw) = H.vae_reference(img)
        self.assertEqual(pixels.size, (gw * 16, gh * 16))
        self.assertEqual(np.asarray(pixels)[0, 0].tolist(), np.asarray(img)[0, 0].tolist())

    def test_match_shape_follows_picture_one(self):
        from PIL import Image

        w, h = H.size_matching(Image.new("RGB", (900, 1600)), 4.0)
        self.assertEqual((w % 32, h % 32), (0, 0))
        self.assertAlmostEqual(w / h, 900 / 1600, delta=0.02)
        self.assertAlmostEqual(w * h / 2**20, 4.0, delta=0.2)

    def test_load_reference_is_upright_rgb_and_refuses_tiny(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.png"
            Image.new("RGBA", (64, 48), (10, 20, 30, 40)).save(p)
            self.assertEqual(H.load_reference(p).mode, "RGB")
            Image.new("RGB", (20, 64)).save(p)
            with self.assertRaises(ValueError):
                H.load_reference(p)

    def test_edit_label_has_no_angle_brackets(self):
        self.assertEqual(H.EDIT_LABEL.format(n=2), "Picture 2: ")
        self.assertEqual(H.MAX_REFERENCES, 16)


class TestEncoderLoader(unittest.TestCase):
    def test_release_encoder_tensors_load_into_the_video_encoder(self):
        """A file shaped like the release (PyTorch conv layout) loads, decoder keys ignored."""
        from minimax_h3_mlx.h3_image_vae import H3ImageEncoder, h3_image_vae_config, load_h3_image_vae_encoder
        from mlx.utils import tree_flatten

        cfg = h3_image_vae_config(RELEASE_VAE_CONFIG)
        ref = H3ImageEncoder(cfg)
        tensors = {k: (v.transpose(0, 4, 1, 2, 3) if v.ndim == 5 else v) for k, v in tree_flatten(ref.parameters())}
        tensors["decoder.mask_token"] = mx.zeros((1, 1024))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "vae.safetensors"
            mx.save_safetensors(str(path), tensors)
            enc = load_h3_image_vae_encoder(path, RELEASE_VAE_CONFIG)
            got = dict(tree_flatten(enc.parameters()))
            want = dict(tree_flatten(ref.parameters()))
            for k in ("encoder.conv_in.weight", "quant_conv.weight"):
                self.assertEqual(float(mx.max(mx.abs(got[k] - want[k])).item()), 0.0)
            # one 64x96 picture -> a 4x6 latent with 48 moments (mean + logvar)
            moments = enc._encode_clip(mx.zeros((1, 1, 64, 96, 3)))
            self.assertEqual(tuple(moments.shape), (1, 1, 4, 6, 48))
            del tensors["quant_conv.bias"]
            mx.save_safetensors(str(path), tensors)
            with self.assertRaises(KeyError):
                load_h3_image_vae_encoder(path, RELEASE_VAE_CONFIG)


class TestTinyEdit(unittest.TestCase):
    def test_references_are_held_and_only_the_target_is_stepped(self):
        cfg, dit, cache, heads = _tiny()
        cond, tags = mx.random.normal((1, 5, cfg.text_dim)), np.ones(5, dtype=np.int64)
        grids = ((4, 6), (2, 2))
        rows = mx.random.normal((6 + 1, 96))  # (4*6)/(2*2) + (2*2)/(2*2)
        seen = []
        orig = dit.__class__.__call__

        def spy(self, video, *a, **k):
            seen.append(np.array(video[0, :7].astype(mx.float32)))
            return orig(self, video, *a, **k)

        dit.__class__.__call__ = spy
        try:
            a = H.denoise_h3_image(dit, cond, tags, 64, 96, cache, heads, seed=3, verbose=False,
                                   reference_rows=rows, reference_grids=grids)
        finally:
            dit.__class__.__call__ = orig
        self.assertEqual(a.shape, (1, 24, 1, 4, 6))
        self.assertTrue(bool(mx.all(mx.isfinite(a)).item()))
        # every forward sees the same anchors: 0.999 * clean + 0.001 * noise
        for s in seen[1:]:
            np.testing.assert_array_equal(s, seen[0])
        expect = 0.999 * np.array(rows.astype(mx.bfloat16).astype(mx.float32))
        self.assertLess(float(np.max(np.abs(seen[0] - expect))), 0.05)
        plain = H.denoise_h3_image(dit, cond, tags, 64, 96, cache, heads, seed=3, verbose=False)
        self.assertGreater(float(mx.max(mx.abs(a - plain)).item()), 1e-4)

    def test_rows_and_grids_must_agree(self):
        cfg, dit, cache, heads = _tiny()
        cond, tags = mx.random.normal((1, 5, cfg.text_dim)), np.ones(5, dtype=np.int64)
        with self.assertRaises(ValueError):
            H.denoise_h3_image(dit, cond, tags, 64, 64, cache, heads, verbose=False,
                               reference_rows=mx.zeros((3, 96)), reference_grids=((4, 4),))
        with self.assertRaises(ValueError):
            H.denoise_h3_image(dit, cond, tags, 64, 64, cache, heads, verbose=False,
                               reference_rows=mx.zeros((4, 96)))


if __name__ == "__main__":
    unittest.main()
