"""MiniMax-H3-Image wiring on a tiny random DiT and decoder (no downloads)."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from minimax_h3_mlx import h3_image as H  # noqa: E402
from minimax_h3_mlx.adaln import ModulationCache, load_modulation_cache, save_modulation_cache  # noqa: E402
from minimax_h3_mlx.dit import MiniMaxH3DiT  # noqa: E402
from minimax_h3_mlx.h3_image_vae import H3ImageVAE, h3_image_vae_config, decode_to_uint8  # noqa: E402
from test_dit_smoke import tiny_config  # noqa: E402

RELEASE_VAE_CONFIG = {
    "latent_channels": 24,
    "latents_mean": [0.1 * i for i in range(24)],
    "latents_std": [1.0 + 0.1 * i for i in range(24)],
    "vit_decoder_kwargs": {"dim_head": 64, "heads": 16, "num_layers": 24, "rope_dim_ratio": 0.5,
                           "rope_theta": 100.0, "num_register_tokens": 4},
}


class TestSchedule(unittest.TestCase):
    def test_grid_is_the_release_and_six_forwards(self):
        self.assertEqual(H.FORWARDS, 6)
        self.assertEqual(H.STEPS, 7)
        self.assertEqual(H.SAMPLE_SIGMAS[0], 1.0)
        self.assertEqual(H.SAMPLE_SIGMAS[-1], 0.0)
        self.assertTrue(all(b < a for a, b in zip(H.SAMPLE_SIGMAS, H.SAMPLE_SIGMAS[1:])))

    def test_timesteps_are_one_minus_sigma_plus_reference_level(self):
        ts = H.reference_timesteps()
        self.assertEqual(len(ts), 7)  # 6 forwards + 0.999
        self.assertIn(float(np.float32(0.999)), ts)
        self.assertEqual(ts[0], 0.0)

    def test_canvas_matches_the_reference_rule(self):
        # at or above the minimum area: only aligned to the 32 px grid
        self.assertEqual(H.generation_canvas(4096, 4096), (4096, 4096))
        # below: both sides scale by sqrt(area / (w*h)); 1024x768 -> area 4 MP -> 2365 x 1774 -> 32 grid
        w, h = H.generation_canvas(1024, 768)
        self.assertEqual((w % 32, h % 32), (0, 0))
        self.assertAlmostEqual(w / h, 4 / 3, delta=0.02)
        self.assertAlmostEqual(w * h / 2**20, 4.0, delta=0.15)
        with self.assertRaises(ValueError):
            H.generation_canvas(1000, 700, None)  # canvas off needs 32-aligned sizes
        self.assertEqual(H.generation_canvas(1024, 1024, 1024), (1024, 1024))

    def test_sizes_are_multiples_of_32_and_16mp(self):
        for ratio, (w, h) in H.SIZES_16MP.items():
            self.assertEqual((w % 32, h % 32), (0, 0), ratio)
            self.assertAlmostEqual(w * h / 2**20, 16.0, delta=0.6, msg=ratio)

    def test_resize_back_is_lanczos_to_the_request(self):
        img = np.zeros((64, 96, 3), np.uint8)
        self.assertEqual(H.resize_output(img, 48, 32).shape, (32, 48, 3))
        self.assertIs(H.resize_output(img, 96, 64), img)


class TestPins(unittest.TestCase):
    def test_script_pins_equal_the_port_record(self):
        import json
        import fetch_h3_image as F

        self.assertEqual(len(F.SHARDS), 12)
        self.assertEqual(F.REVISION, "6534ea1c979a5f8cb00c2bd1025d5ca4ee8d8901")
        self.assertEqual(sum(b for b, _ in F.SHARDS.values()), 66_279_087_736)
        for name, (size, sha) in F.SHARDS.items():
            self.assertEqual(len(sha), 64, name)
            self.assertGreater(size, 4_000_000_000, name)
        self.assertEqual(F.CORE_FILES["pdd_fused_heads.safetensors"][1][:8], "c07093a0")
        self.assertIn("1 million USD", F.LICENCE_NOTE)
        self.assertIn("MiniMax-H3-Image is licensed under the MiniMax-H3-Image Community License Agreement", F.NOTICE_TEXT)
        self.assertIn("quantised", F.MODIFICATION_NOTICE)
        self.assertEqual(F.main(["--dir", "/nonexistent-h3img"]), 2)  # refuses without --accepted


class TestHeads(unittest.TestCase):
    def test_heads_are_per_step_and_refuse_any_other_layout(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            ok = Path(d) / "heads.safetensors"
            mx.save_safetensors(str(ok), {"video_out.weight": mx.zeros((6, 96, 16)), "video_out.bias": mx.zeros((6, 96))})
            w, b = H.load_pdd_heads(ok)
            self.assertEqual((w.shape, b.shape), ((6, 96, 16), (6, 96)))
            bad = Path(d) / "bad.safetensors"
            mx.save_safetensors(str(bad), {"video_out.weight": mx.zeros((8, 96, 16)), "video_out.bias": mx.zeros((8, 96))})
            with self.assertRaises(ValueError):
                H.load_pdd_heads(bad)
            extra = Path(d) / "extra.safetensors"
            mx.save_safetensors(str(extra), {"video_out.weight": mx.zeros((6, 96, 16)), "video_out.bias": mx.zeros((6, 96)),
                                             "audio_out.weight": mx.zeros((6, 32, 16))})
            with self.assertRaises(ValueError):
                H.load_pdd_heads(extra)


def _tiny(seed=0):
    cfg = replace(tiny_config(), latents_dim=24, rope_inv_freq_len=2, audio_enabled=False)
    mx.random.seed(seed)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    from minimax_h3_mlx.adaln import ModulationCache

    ts = mx.array(H.reference_timesteps(), dtype=mx.float32)
    cache = ModulationCache.build(dit, ts, dtype=mx.bfloat16)
    heads = (mx.random.normal((6, 96, cfg.hidden_size)) * 0.02, mx.zeros((6, 96)))
    return cfg, dit, cache, heads


class TestTinyEndToEnd(unittest.TestCase):
    def test_denoise_is_deterministic_with_no_audio_rows(self):
        cfg, dit, cache, heads = _tiny()
        cond = mx.random.normal((1, 5, cfg.text_dim))
        tags = np.ones(5, dtype=np.int64)
        a = H.denoise_h3_image(dit, cond, tags, 64, 96, cache, heads, seed=3, verbose=False)
        b = H.denoise_h3_image(dit, cond, tags, 64, 96, cache, heads, seed=3, verbose=False)
        c = H.denoise_h3_image(dit, cond, tags, 64, 96, cache, heads, seed=4, verbose=False)
        self.assertEqual(a.shape, (1, 24, 1, 4, 6))
        self.assertEqual(float(mx.max(mx.abs(a - b)).item()), 0.0)
        self.assertGreater(float(mx.max(mx.abs(a - c)).item()), 1e-3)
        self.assertTrue(bool(mx.all(mx.isfinite(a)).item()))
        self.assertIsNone(dit.pdd_heads)  # the hook is cleaned up after a run

    def test_each_forward_uses_its_own_head(self):
        cfg, dit, cache, heads = _tiny()
        seen = []
        orig = dit.__class__.__call__

        def spy(self, *a, **k):
            seen.append(self.pdd_step)
            return orig(self, *a, **k)

        dit.__class__.__call__ = spy
        try:
            H.denoise_h3_image(dit, mx.random.normal((1, 5, cfg.text_dim)), np.ones(5, dtype=np.int64), 64, 64,
                               cache, heads, seed=1, verbose=False)
        finally:
            dit.__class__.__call__ = orig
        self.assertEqual(seen, [0, 1, 2, 3, 4, 5])

    def test_wrong_head_count_and_wrong_cache_are_refused(self):
        cfg, dit, cache, heads = _tiny()
        cond, tags = mx.random.normal((1, 5, cfg.text_dim)), np.ones(5, dtype=np.int64)
        with self.assertRaises(ValueError):
            H.denoise_h3_image(dit, cond, tags, 64, 64, cache, (heads[0][:4], heads[1][:4]), verbose=False)
        wrong = ModulationCache(cache.tables, mx.array([0.0, 0.5, 1.0], dtype=mx.float32))
        with self.assertRaises(ValueError):
            H.denoise_h3_image(dit, cond, tags, 64, 64, wrong, heads, verbose=False)
        with self.assertRaises(ValueError):
            H.denoise_h3_image(dit, cond, tags, 60, 64, cache, heads, verbose=False)

    def test_cache_roundtrip_on_disk(self):
        import tempfile

        _, _, cache, _ = _tiny()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "adaln_cache.safetensors"
            save_modulation_cache(cache, p)
            back = load_modulation_cache(p)
        self.assertEqual(len(back.tables), len(cache.tables))
        self.assertEqual(float(mx.max(mx.abs(back.timesteps - cache.timesteps)).item()), 0.0)
        self.assertEqual(float(mx.max(mx.abs(back.get(0)[3].astype(mx.float32) - cache.get(0)[3].astype(mx.float32))).item()), 0.0)


class TestDecoder(unittest.TestCase):
    def test_tiny_decoder_uses_two_axis_rope_and_decodes_a_frame(self):
        raw = dict(RELEASE_VAE_CONFIG)
        raw["vit_decoder_kwargs"] = dict(raw["vit_decoder_kwargs"], heads=2, num_layers=2)
        cfg = h3_image_vae_config(raw)
        self.assertEqual(cfg.spatial_compression_ratio, 16)
        vae = H3ImageVAE(cfg)
        self.assertEqual(vae.decoder.rope.inv_freq.shape[0], 8)  # head 64 x 0.5 = 32 = 2 axes x 2 x 8
        mx.random.seed(0)
        lat = mx.random.normal((1, 24, 1, 4, 5))
        img = decode_to_uint8(vae, lat)
        self.assertEqual(img.shape, (64, 80, 3))
        self.assertEqual(img.dtype, np.uint8)

    def test_release_decoder_dimensions(self):
        cfg = h3_image_vae_config(RELEASE_VAE_CONFIG)
        self.assertEqual((cfg.decoder_num_layers, cfg.decoder_dim, cfg.decoder_num_register_tokens), (24, 1024, 4))
        self.assertEqual(cfg.decoder_rope_dim_ratio, 0.5)


if __name__ == "__main__":
    unittest.main()
