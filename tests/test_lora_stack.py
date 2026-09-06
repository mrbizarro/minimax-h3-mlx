"""Stacked adapters: `--lora` repeats, and the deltas add.

A character LoRA and the Turbo distillation adapter are independent low-rank
updates to the same base layers, so applying both is `y = base(x) + d1 + d2`
with each adapter keeping its own scale. The trap this guards is the nested
wrapper: wrapping a LoRALinear in another LoRALinear hides the quantized
base's `scales` from `plan()`, which then compares the adapter against the
PACKED storage width and skips every module — the applied=0 failure that
once shipped as a render that "worked". A second apply must therefore find
the base under the first adapter and append to it.
"""

import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from minimax_h3_mlx import draft_cache as cache_mod
from minimax_h3_mlx import lora as lora_mod
from tests.test_lora_qkv_permute import _Config, _Model


def _save(path: Path, pairs: dict[str, tuple[mx.array, mx.array]]) -> Path:
    flat = {}
    for name, (a, b) in pairs.items():
        flat[f"{name}.lora_A.weight"] = a
        flat[f"{name}.lora_B.weight"] = b
    mx.save_safetensors(str(path), flat)
    return path


class StackedAdapters(unittest.TestCase):
    def setUp(self):
        self.model = _Model()
        self.dim = _Config.num_attention_heads * _Config.attention_head_dim
        targets = lora_mod._iter_targets(self.model)
        self.name = next(n for n in targets if n.endswith(lora_mod.QKV_SUFFIX))
        mx.random.seed(1)
        rank = 2
        self.a1 = mx.random.normal((rank, self.dim)); self.b1 = mx.random.normal((self.dim * 3, rank))
        self.a2 = mx.random.normal((rank, self.dim)); self.b2 = mx.random.normal((self.dim * 3, rank))
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.f1 = _save(root / "one.safetensors", {self.name: (self.a1, self.b1)})
        self.f2 = _save(root / "two.safetensors", {self.name: (self.a2, self.b2)})
        self.x = mx.random.normal((3, self.dim))
        self.base_out = self.model.blocks[0].attn.qkv_proj(self.x)

    def tearDown(self):
        self.tmp.cleanup()

    def _layer(self):
        return self.model.blocks[0].attn.qkv_proj

    def test_second_apply_stacks_instead_of_nesting(self):
        r1 = lora_mod.apply_lora(self.model, self.f1, 0.7, verbose=False)
        r2 = lora_mod.apply_lora(self.model, self.f2, 0.4, verbose=False)
        self.assertEqual([m.name for m in r1.applied], [self.name])
        self.assertEqual([m.name for m in r2.applied], [self.name], "second adapter skipped")
        self.assertEqual(r2.skipped, [])
        layer = self._layer()
        self.assertIsInstance(layer, lora_mod.LoRALinear)
        self.assertNotIsInstance(layer.base, lora_mod.LoRALinear, "nested wrapper")
        self.assertEqual(layer.adapters, 2)
        self.assertEqual(layer.lora_scales, [0.7, 0.4])

    def test_output_is_base_plus_both_deltas(self):
        lora_mod.apply_lora(self.model, self.f1, 0.7, verbose=False)
        lora_mod.apply_lora(self.model, self.f2, 0.4, verbose=False)
        d1 = ((self.x @ self.a1.T) @ self.b1.T) * 0.7
        d2 = ((self.x @ self.a2.T) @ self.b2.T) * 0.4
        got = self._layer()(self.x)
        self.assertTrue(mx.allclose(got, self.base_out + d1 + d2, atol=1e-4).item())

    def test_zero_scale_adapter_is_a_no_op_in_the_stack(self):
        lora_mod.apply_lora(self.model, self.f1, 0.7, verbose=False)
        lora_mod.apply_lora(self.model, self.f2, 0.0, verbose=False)
        d1 = ((self.x @ self.a1.T) @ self.b1.T) * 0.7
        got = self._layer()(self.x)
        self.assertTrue(mx.allclose(got, self.base_out + d1, atol=1e-4).item())

    def test_plan_sees_the_base_under_an_adapter(self):
        lora_mod.apply_lora(self.model, self.f1, 1.0, verbose=False)
        applicable, skipped = lora_mod.plan(self.model, {self.name: (self.a2, self.b2)})
        self.assertEqual(skipped, [])
        self.assertIn(self.name, applicable)

    def test_apply_loras_helper_reports_each(self):
        reports = lora_mod.apply_loras(self.model, [(self.f1, 1.0), (self.f2, 0.5)], verbose=False)
        self.assertEqual([r.scale for r in reports], [1.0, 0.5])
        self.assertEqual(self._layer().adapters, 2)

    def test_fuse_under_a_runtime_adapter_is_refused_not_corrupted(self):
        lora_mod.apply_lora(self.model, self.f1, 1.0, verbose=False)
        r = lora_mod.apply_lora(self.model, self.f2, 1.0, mode="fuse", verbose=False)
        self.assertEqual(r.applied, [])
        self.assertTrue(any("runtime" in m.reason for m in r.skipped))

    def test_cache_key_is_order_insensitive_and_distinct_per_stack(self):
        table = mx.zeros((4,), dtype=mx.float32)
        import numpy as np
        t = np.zeros((4,), dtype=np.float32)
        dit = self.f1                                  # any real file: fingerprint wants a stat
        k_one = cache_mod.DraftCache.adaln_key(timestep_table=t, dit=dit, lora=(self.f1, 1.0), lora_adaln=None)
        k_one_list = cache_mod.DraftCache.adaln_key(timestep_table=t, dit=dit, lora=[(self.f1, 1.0)], lora_adaln=None)
        k_ab = cache_mod.DraftCache.adaln_key(timestep_table=t, dit=dit, lora=[(self.f1, 1.0), (self.f2, 0.5)], lora_adaln=None)
        k_ba = cache_mod.DraftCache.adaln_key(timestep_table=t, dit=dit, lora=[(self.f2, 0.5), (self.f1, 1.0)], lora_adaln=None)
        k_ab2 = cache_mod.DraftCache.adaln_key(timestep_table=t, dit=dit, lora=[(self.f1, 1.0), (self.f2, 0.6)], lora_adaln=None)
        self.assertEqual(k_one, k_one_list, "a one-element stack must keep the historical key")
        self.assertEqual(k_ab, k_ba)
        self.assertNotEqual(k_ab, k_one)
        self.assertNotEqual(k_ab, k_ab2)


if __name__ == "__main__":
    unittest.main()
