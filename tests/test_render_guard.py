"""The black-render class (2026-09-17): F16 adapters must not overflow, and broken output must fail.

An ai-toolkit F16 LoRA on the MLP rows computed its runtime delta in float16; the DiT's MLP inputs
exceed float16's 65504, so the delta went inf -> NaN latents -> a flat black clip reported "done".
"""

import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from minimax_h3_mlx import lora as lora_mod
from minimax_h3_mlx import render_guard as guard

IN, OUT, RANK = 32, 24, 4


def _pairs(dtype):
    mx.random.seed(3)
    a = (mx.random.normal((RANK, IN)) * 0.1).astype(dtype)
    b = (mx.random.normal((OUT, RANK)) * 0.1).astype(dtype)
    return a, b


def _big_input():
    # Well past float16's range, well inside bfloat16's — what a SwiGLU product can reach.
    x = mx.random.normal((5, IN)) * 3e5
    return x.astype(mx.bfloat16)


class AdapterDtypeRange(unittest.TestCase):
    def _check(self, dtype, expect_dtype):
        base = nn.Linear(IN, OUT, bias=False)
        base.weight = base.weight.astype(mx.bfloat16) * 0      # isolate the delta
        a, b = _pairs(dtype)
        layer = lora_mod.LoRALinear(base, a, b, 1.0)
        self.assertEqual(layer.lora_a[0].dtype, expect_dtype)
        self.assertEqual(layer.lora_b[0].dtype, expect_dtype)
        x = _big_input()
        y = layer(x)
        mx.eval(y)
        self.assertEqual(y.dtype, mx.bfloat16)
        self.assertTrue(bool(mx.all(mx.isfinite(y)).item()), f"{dtype} adapter overflowed")
        ref = (x.astype(mx.float32) @ a.astype(mx.float32).T) @ b.astype(mx.float32).T
        rel = float((mx.abs(y.astype(mx.float32) - ref).max() / mx.abs(ref).max()).item())
        self.assertLess(rel, 2e-2, f"{dtype} adapter delta wrong (rel err {rel})")

    def test_float16_adapter_is_promoted_and_finite(self):
        self._check(mx.float16, mx.bfloat16)

    def test_float32_adapter_is_kept_and_finite(self):
        self._check(mx.float32, mx.float32)

    def test_bfloat16_adapter_is_untouched(self):
        base = nn.Linear(IN, OUT, bias=False)
        a, b = _pairs(mx.bfloat16)
        layer = lora_mod.LoRALinear(base, a, b, 1.0)
        self.assertIs(layer.lora_a[0], a)
        self.assertIs(layer.lora_b[0], b)
        self._check(mx.bfloat16, mx.bfloat16)

    def test_stacked_float16_on_bfloat16_is_finite(self):
        base = nn.Linear(IN, OUT, bias=False)
        base.weight = base.weight.astype(mx.bfloat16)
        a1, b1 = _pairs(mx.bfloat16)
        a2, b2 = _pairs(mx.float16)
        layer = lora_mod.LoRALinear(base, a1, b1, 1.0)
        layer.add(a2, b2, 1.0)
        self.assertEqual(layer.lora_a[1].dtype, mx.bfloat16)
        y = layer(_big_input())
        self.assertTrue(bool(mx.all(mx.isfinite(y)).item()))

    def test_pre_fix_math_really_overflows(self):
        # Guards the test itself: the stored-dtype math on this input must break.
        a, b = _pairs(mx.float16)
        part = (_big_input().astype(mx.float16) @ a.T) @ b.T
        self.assertFalse(bool(mx.all(mx.isfinite(part)).item()))


class FileFormats(unittest.TestCase):
    """ai-toolkit / diffusers (prefixed lora_A/B) and bare files, each in F16, F32 and BF16."""

    def test_every_format_and_dtype_loads_finite(self):
        from tests.test_lora_qkv_permute import _Config, _Model

        dim = _Config.num_attention_heads * _Config.attention_head_dim
        for prefix in ("", "diffusion_model.", "transformer."):
            for dtype in (mx.float16, mx.float32, mx.bfloat16):
                with self.subTest(prefix=prefix, dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                    model = _Model()
                    name = next(n for n in lora_mod._iter_targets(model)
                                if n.endswith(lora_mod.QKV_SUFFIX))
                    mx.random.seed(5)
                    a = mx.random.normal((2, dim)).astype(dtype)
                    b = mx.random.normal((dim * 3, 2)).astype(dtype)
                    path = Path(tmp) / "adapter.safetensors"
                    mx.save_safetensors(str(path), {f"{prefix}{name}.lora_A.weight": a,
                                                    f"{prefix}{name}.lora_B.weight": b})
                    report = lora_mod.apply_lora(model, path, 1.0, verbose=False)
                    self.assertEqual(len(report.applied), 1)
                    layer = model.blocks[0].attn.qkv_proj
                    x = (mx.random.normal((3, dim)) * 1e5).astype(mx.bfloat16)
                    self.assertTrue(bool(mx.all(mx.isfinite(layer(x))).item()))


class Guards(unittest.TestCase):
    def test_nonfinite_flag(self):
        ok = mx.ones((4, 3))
        self.assertFalse(bool(guard.nonfinite_flag(ok, ok).item()))
        bad = mx.array([1.0, float("nan")])
        self.assertTrue(bool(guard.nonfinite_flag(ok, bad).item()))
        inf = mx.array([float("inf")], dtype=mx.bfloat16)
        self.assertTrue(bool(guard.nonfinite_flag(inf).item()))

    def test_nonfinite_error_names_the_adapter_and_dtype(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "user_style.safetensors"
            mx.save_safetensors(str(path), {"x.lora_A.weight": mx.zeros((2, 2), dtype=mx.float16)})
            err = guard.nonfinite_error("after denoise step 1/3", [(path, 1.0)])
        self.assertEqual(err.kind, guard.NONFINITE_LATENTS)
        text = str(err)
        self.assertIn("user_style.safetensors", text)
        self.assertIn("F16", text)
        self.assertIn("step 1/3", text)

    def test_flat_video_is_rejected(self):
        black = np.full((8, 16, 16, 3), 16, dtype=np.uint8)
        with self.assertRaises(guard.RenderIntegrityError) as ctx:
            guard.check_decoded_video(black, [])
        self.assertEqual(ctx.exception.kind, guard.BLANK_VIDEO)

    def test_real_video_passes(self):
        rng = np.random.default_rng(0)
        clip = rng.integers(0, 255, (8, 16, 16, 3), dtype=np.uint8)
        guard.check_decoded_video(clip, [])
        # A dark but not flat clip (a night scene) must pass too.
        dark = np.full((8, 16, 16, 3), 10, dtype=np.uint8)
        dark[3, 4:8, 4:8] = 40
        guard.check_decoded_video(dark, [])


class RunnerWiring(unittest.TestCase):
    """The runner must actually call both guards and write the readable error to metrics."""

    def test_generate_staged_uses_the_guards(self):
        source = (Path(__file__).resolve().parents[1] / "scripts" / "generate_staged.py").read_text()
        self.assertIn("nonfinite_flag(video_rows, audio_rows)", source)
        self.assertIn("check_decoded_video(video, lora_stack)", source)
        self.assertIn('record.data["error_message"]', source)


if __name__ == "__main__":
    unittest.main()
