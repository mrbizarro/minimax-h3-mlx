"""MiniMax-H3-Image low-memory mode: streamed text encoder and streamed DiT.

Run on a Mac with Metal: python tests/test_h3_image_lowmem.py
The contract is byte equality: a tiny Q8 DiT is written as a checkpoint, then the 6-forward image
loop runs resident and streamed under several residency policies; the latents must be identical.
The planner and the runner's plumbing are checked without real weights.
"""

from __future__ import annotations

import gc
import json
import sys
import tempfile
import unittest

from PIL import Image
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from minimax_h3_mlx import h3_image as H  # noqa: E402
from minimax_h3_mlx.adaln import ModulationCache, drop_adaln_weights  # noqa: E402
from minimax_h3_mlx.dit import MiniMaxH3DiT  # noqa: E402
from minimax_h3_mlx.dit_stream import StreamPolicy  # noqa: E402
from minimax_h3_mlx.load import load_dit  # noqa: E402
from minimax_h3_mlx.quantize import QuantConfig, quantize_dit  # noqa: E402
from test_dit_stream import tiny_config  # noqa: E402


def image_config():
    return replace(tiny_config(), latents_dim=24, audio_enabled=False)


def write_image_checkpoint(root: Path) -> None:
    cfg = image_config()
    mx.random.seed(23)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    quantize_dit(dit, QuantConfig(bits=8, group_size=64, quantize_adaln=True, adaln_bits=8))
    flat = dict(tree_flatten(dit.parameters()))
    raw = asdict(cfg)
    raw["patch_size"] = list(raw["patch_size"])
    (root / "config.json").write_text(json.dumps(raw))
    (root / "quant_config.json").write_text(json.dumps(
        {"bits": 8, "group_size": 64, "quantize_adaln": True, "adaln_bits": 8}))
    keys = sorted(flat)
    half = len(keys) // 2
    for n, part in enumerate((keys[:half], keys[half:]), start=1):
        mx.save_safetensors(str(root / f"model-0000{n}.safetensors"), {k: flat[k] for k in part})


def denoise(root: Path, stream=None, references=False) -> bytes:
    dit = load_dit(root, stream=stream)
    cfg = dit.config
    ts = mx.array(H.reference_timesteps(), dtype=mx.float32)
    cache = ModulationCache.build(dit, ts, dtype=mx.bfloat16)
    drop_adaln_weights(dit)
    mx.random.seed(9)
    heads = (mx.random.normal((6, 96, cfg.hidden_size)) * 0.02, mx.zeros((6, 96)))
    cond = mx.random.normal((1, 5, cfg.text_dim))
    tags = np.ones(5, dtype=np.int64)
    kw = {}
    if references:
        rows = mx.random.normal((2 * 2, 96))          # one 4x4-latent reference -> 2x2 patches
        kw = {"reference_rows": rows, "reference_grids": ((4, 4),)}
    out = H.denoise_h3_image(dit, cond, tags, 64, 96, cache, heads, seed=3, verbose=False, **kw)
    mx.eval(out)
    data = np.array(out).tobytes()
    streamer = getattr(dit, "streamer", None)
    if streamer is not None:
        streamer.close()
    del dit, cache
    gc.collect()
    mx.clear_cache()
    return data


class StreamedImageLoopIsExact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        write_image_checkpoint(cls.root)
        cls.resident = denoise(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_resident_is_deterministic(self):
        self.assertEqual(denoise(self.root), self.resident)

    def test_streamed_policies_are_byte_identical(self):
        for policy in (StreamPolicy(pipeline=True), StreamPolicy(pipeline=False),
                       StreamPolicy(resident_blocks=frozenset({0, 2}), pipeline=True),
                       StreamPolicy(prefetch=2, workers=1, pipeline=True)):
            with self.subTest(policy=policy.describe()):
                self.assertEqual(denoise(self.root, policy), self.resident)

    def test_streamed_edit_is_byte_identical(self):
        self.assertEqual(denoise(self.root, StreamPolicy(pipeline=True), references=True),
                         denoise(self.root, references=True))


class Planner(unittest.TestCase):
    def test_budget_rule_matches_the_studio(self):
        self.assertEqual(H.mac_budget_gib(16), 10.0)
        self.assertEqual(H.mac_budget_gib(64), 56.0)

    def test_auto_streams_everywhere(self):
        # streaming measured as fast as resident and byte-identical: auto never picks resident
        for ram in (16, 24, 32, 64, 128):
            for mp in (4, 8, 16):
                self.assertEqual(H.plan_memory(ram, mp), H.LOWMEM)

    def test_small_macs_stream_both_and_fit(self):
        for ram in (16, 24, 32):
            for mp in (4, 8, 16):
                plan = H.plan_memory(ram, mp)
                with self.subTest(ram=ram, mp=mp):
                    self.assertTrue(plan.stream_text_encoder)
                    self.assertLessEqual(H.estimate_peak_gib(plan, mp), H.mac_budget_gib(ram))

    def test_ladder_gets_lighter_and_estimates_cover_the_measured_peaks(self):
        peaks = [H.estimate_peak_gib(s, 16) for s in H.memory_ladder(16)]
        self.assertEqual(peaks, sorted(peaks, reverse=True))
        # measured footprints (12 Nov): streamed 4 MP 4.83, 16 MP 8.15; resident 27.96 / 27.99
        self.assertGreaterEqual(H.estimate_peak_gib(H.LOWMEM, 4), 4.83)
        self.assertGreaterEqual(H.estimate_peak_gib(H.LOWMEM, 16), 8.15)
        self.assertGreaterEqual(H.estimate_peak_gib(H.RESIDENT, 16), 27.99)

    def test_modes(self):
        self.assertEqual(H.memory_settings("resident", 4), H.RESIDENT)
        low = H.memory_settings("lowmem", 4)
        self.assertTrue(low.stream_dit and low.stream_text_encoder)
        self.assertEqual(H.memory_settings("lowmem", 4, dit_resident_gb=5).dit_resident_gb, 5.0)
        self.assertEqual(H.memory_settings("auto", 16, ram_gib=16).name, H.plan_memory(16, 16).name)
        with self.assertRaises(ValueError):
            H.memory_settings("fast", 4)

    def test_a_tight_budget_takes_the_lightest_rung(self):
        self.assertEqual(H.plan_memory(64, 16, budget_gib=6.5), H.LOWMEM_MIN)
        self.assertEqual(H.plan_memory(64, 16, budget_gib=12), H.LOWMEM)
        # a budget never exceeds what the Mac itself can give
        self.assertEqual(H.plan_memory(16, 4, budget_gib=60), H.plan_memory(16, 4))


class PlannerReferences(unittest.TestCase):
    """Measured 13 Nov: a six-picture 4 MP edit (2720x1536 canvas) carried 9,751 text rows and
    33,878 reference rows and peaked at 14.08 GB on the lightest exact setting."""

    SIX = [(2720, 1536), (5024, 3360), (6144, 4096), (5472, 3072), (2368, 1760), (5024, 3360)]
    MP = 2720 * 1536 / 2**20

    def test_rows_match_the_measured_edit(self):
        vae, vision = H.reference_rows(self.SIX)
        self.assertEqual(vae, 33878)
        self.assertLess(abs(vision + H.TEXT_BASE_ROWS - 9751), 600)

    def test_rows_follow_the_vae_grid(self):
        for size in self.SIX:
            pixels, (gh, gw) = H.vae_reference(Image.new("RGB", size))
            self.assertEqual(H.reference_rows([size])[0], gh * gw // 4)
            pixels, (gh, gw) = H.vae_reference(Image.new("RGB", size), 1024)
            self.assertEqual(min(pixels.size), 1024)
            self.assertEqual(H.reference_rows([size], 1024)[0], gh * gw // 4)

    def test_estimate_covers_the_measured_edit(self):
        self.assertGreaterEqual(H.estimate_peak_gib(H.LOWMEM_MIN, self.MP, self.SIX), 14.08)

    def test_light_references_only_for_edits_and_only_when_nothing_exact_fits(self):
        self.assertNotIn(H.LOWMEM_LIGHTREF, H.memory_ladder(16))
        self.assertEqual(H.memory_ladder(4, self.SIX)[-1], H.LOWMEM_LIGHTREF)
        self.assertEqual(H.plan_memory(64, self.MP, self.SIX), H.LOWMEM)        # exact on a big Mac
        self.assertEqual(H.plan_memory(16, self.MP, self.SIX), H.LOWMEM_LIGHTREF)
        self.assertLessEqual(H.estimate_peak_gib(H.LOWMEM_LIGHTREF, self.MP, self.SIX), H.mac_budget_gib(16))
        self.assertEqual(H.plan_memory(16, 4, [(1536, 1024)]), H.LOWMEM)        # one small photo stays exact

    def test_a_count_prices_the_largest_reference(self):
        self.assertGreaterEqual(H.packed_rows_estimate(4, 2), H.packed_rows_estimate(4, self.SIX[:2]))

    def test_24_mp_still_fits_a_16_gb_mac(self):
        # measured 13 Nov: 6144x4096 on the 16 GB setting, lightest exact setting, 8.87 GB
        self.assertEqual(H.plan_memory(16, 24.1), H.LOWMEM_MIN)
        self.assertGreaterEqual(H.estimate_peak_gib(H.LOWMEM_MIN, 24.0), 8.87)


def _cache_limit() -> int:
    previous = mx.set_cache_limit(0)
    mx.set_cache_limit(previous)
    return previous


class RunnerPlumbing(unittest.TestCase):
    """The settings reach the encoder, the DiT loader, the cache cap and the decoder."""

    def _run(self, settings):
        seen = {}

        class Encoder:
            def __init__(self, *a, **kw):
                seen["encoder"] = kw

            def encode(self, prompt, *a, **kw):
                return mx.zeros((1, 4, 8)), np.ones(4, dtype=np.int64)

        class Streamer:
            def summary(self):
                return {"streamed_blocks": 3}

            def close(self):
                seen["streamer_closed"] = True

        class Dit:
            streamer = Streamer()

        def load(path, verbose=False, stream=None):
            seen["stream"] = stream
            return Dit()

        class Vae:
            decode_batch = 8

        def decode(vae, latents):
            seen["decode_batch"] = vae.decode_batch
            seen["cache_during_decode"] = _cache_limit()
            return np.zeros((64, 96, 3), np.uint8)

        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        (root / "vae").mkdir()
        (root / "vae" / "config.json").write_text("{}")
        before = _cache_limit()
        with mock.patch("minimax_h3_mlx.text_encoder.MiniMaxH3TextEncoder", Encoder), \
                mock.patch("minimax_h3_mlx.load.load_dit", load), \
                mock.patch.object(H, "load_modulation_cache", lambda p: None), \
                mock.patch.object(H, "load_pdd_heads", lambda p: None), \
                mock.patch.object(H, "denoise_h3_image", lambda *a, **k: mx.zeros((1, 24, 1, 4, 6))), \
                mock.patch("minimax_h3_mlx.h3_image_vae.load_h3_image_vae", lambda p, c: Vae()), \
                mock.patch("minimax_h3_mlx.h3_image_vae.decode_to_uint8", decode):
            runner = H.H3ImageRunner(root, root, memory=settings)
            image = runner.generate("a fox", 96, 64, verbose=False, min_generation_area=None)
        seen["cache_after"] = _cache_limit()
        seen["cache_before"] = before
        tmp.cleanup()
        return seen, runner.metrics, image

    def test_resident_is_the_original_path(self):
        seen, metrics, image = self._run(H.RESIDENT)
        self.assertFalse(seen["encoder"]["stream"])
        self.assertIsNone(seen["stream"])
        self.assertEqual(seen["decode_batch"], 8)
        self.assertEqual(image.shape, (64, 96, 3))
        self.assertEqual(set(metrics["phases"]), {"text", "denoise", "decode"})
        self.assertIn("peak_footprint_gb", metrics)

    def test_lowmem_streams_caps_and_restores(self):
        settings = H.MemorySettings(stream_text_encoder=True, stream_dit=True, dit_resident_gb=6.0,
                                    cache_gb=1.0, decode_batch=1, name="t")
        seen, metrics, _ = self._run(settings)
        self.assertTrue(seen["encoder"]["stream"])
        self.assertIsInstance(seen["stream"], StreamPolicy)
        self.assertEqual(seen["stream"].resident_gb, 6.0)
        self.assertTrue(seen["stream"].pipeline)
        self.assertEqual(seen["decode_batch"], 1)
        self.assertEqual(seen["cache_during_decode"], 1024**3)
        self.assertEqual(seen["cache_after"], seen["cache_before"])     # the cap is put back
        self.assertTrue(seen["streamer_closed"])
        self.assertEqual(metrics["memory"]["name"], "t")
        self.assertEqual(metrics["dit_stream"], {"streamed_blocks": 3})


class ProcessFootprint(unittest.TestCase):
    def test_reads_a_plausible_footprint(self):
        fp = H.process_footprint()
        self.assertGreater(fp, 10 * 1024**2)
        self.assertLess(fp, 1024**4)


if __name__ == "__main__":
    unittest.main()
