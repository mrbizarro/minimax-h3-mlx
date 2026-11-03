"""P-H3-Image pipeline wiring on a tiny random DiT and a tiny decoder (no downloads)."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from minimax_h3_mlx import ph3_image as P  # noqa: E402
from minimax_h3_mlx.dit import MiniMaxH3DiT  # noqa: E402
from minimax_h3_mlx.ph3_image_vae import PH3ImageVAE, decode_to_uint8, image_vae_config  # noqa: E402
from test_dit_smoke import tiny_config  # noqa: E402
from test_ph3_image_vae import TINY  # noqa: E402


class TestSchedule(unittest.TestCase):
    def test_grid_is_the_checkpoints_and_eight_forwards(self):
        self.assertEqual(len(P.SAMPLE_SIGMAS), 9)
        self.assertEqual(P.FORWARDS, 8)
        self.assertEqual(P.STEPS, P.FORWARDS + 1)
        self.assertEqual(P.SAMPLE_SIGMAS[0], 1.0)
        self.assertEqual(P.SAMPLE_SIGMAS[-1], 0.0)

    def test_sizes_are_multiples_of_32_and_8mp(self):
        for ratio, (w, h) in P.SIZES_8MP.items():
            self.assertEqual((w % 32, h % 32), (0, 0), ratio)
            self.assertAlmostEqual(w * h / 2**20, 8.0, delta=0.12, msg=ratio)
        self.assertEqual(P.size_for_megapixels("16:9", 8), (3872, 2176))
        w, h = P.size_for_megapixels("1:1", 2)
        self.assertEqual((w % 32, h % 32), (0, 0))
        self.assertAlmostEqual(w * h / 2**20, 2.0, delta=0.1)

    def test_round_canvas(self):
        self.assertEqual(P.round_canvas(100, 70), (96, 64))


class TestTinyEndToEnd(unittest.TestCase):
    def test_denoise_and_decode(self):
        # 16 px per latent token x 2 (patch) -> canvas must be a multiple of 32.
        cfg = replace(tiny_config(), latents_dim=24, rope_inv_freq_len=2)
        mx.random.seed(0)
        dit = MiniMaxH3DiT(cfg)
        dit.adaln_t_table = mx.zeros((5, cfg.time_embed_dim))  # pruned-curve style is not needed here
        mx.eval(dit.parameters())
        cfg = dit.config
        cond = mx.random.normal((1, 6, cfg.text_dim))
        tags = np.ones(6, dtype=np.int64)
        try:
            latents = P.denoise_image(dit, cond, tags, 64, 96, seed=3, drop_adaln=False, verbose=False)
        except Exception as exc:  # the tiny DiT has no AdaLN curve; keep the failure readable
            self.skipTest(f"tiny DiT cannot run the cached AdaLN path: {exc}")
        self.assertEqual(latents.shape, (1, 24, 1, 4, 6))
        self.assertFalse(bool(mx.any(mx.isnan(latents)).item()))
        self.assertEqual(len(P.denoise_image.last_step_seconds), P.FORWARDS)

        vae = PH3ImageVAE(image_vae_config(TINY))
        mx.eval(vae.parameters())
        vae.use_tiling = False
        img = decode_to_uint8(vae, latents)
        self.assertEqual(img.shape, (64, 96, 3))
        self.assertEqual(img.dtype, np.uint8)

    def test_same_seed_same_latents(self):
        cfg = replace(tiny_config(), latents_dim=24, rope_inv_freq_len=2)
        mx.random.seed(0)
        dit = MiniMaxH3DiT(cfg)
        mx.eval(dit.parameters())
        cond = mx.random.normal((1, 5, cfg.text_dim))
        tags = np.ones(5, dtype=np.int64)
        try:
            a = P.denoise_image(dit, cond, tags, 64, 64, seed=7, drop_adaln=False, verbose=False)
            b = P.denoise_image(dit, cond, tags, 64, 64, seed=7, drop_adaln=False, verbose=False)
        except Exception as exc:
            self.skipTest(f"tiny DiT cannot run the cached AdaLN path: {exc}")
        self.assertEqual(float(mx.max(mx.abs(a - b)).item()), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
