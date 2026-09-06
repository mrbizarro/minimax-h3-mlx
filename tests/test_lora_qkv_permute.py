"""The qkv row-permute switch.

Background, so this is not re-litigated: the permute exists because a LoRA
trained through the ComfyUI model definition stores fused qkv rows as
(3, heads, head_dim) while this checkpoint stores (heads, 3, head_dim). We
apply the remap to every file because nothing in a safetensors LoRA states its
lineage.

SETTLED 2026-08-15 by measurement, not by eye. The same adapter exists as our
own repack (lightx2v_v1.0_768p_ourlayout) and as Kijai's rank-resized
conversion declaring `converted_layout: comfyui_minimax_h3`. Cosine of the
effective delta B@A between the two, per module:

    qkv_proj   as-is +0.975   permuted +0.012
    mlp.fc1    as-is +0.962   swapped  +0.002

The converted files are ALREADY in this checkpoint's layout, so both
transforms are wrong and both now default OFF. +0.012 is orthogonal: the
permute was not translating the adapter, it was replacing it. It ran on every
H3 LoRA including our shipped Turbo. Render-level it is subtle — the delta is
~1% of base — which is why it survived so long.
"""

import unittest

import mlx.core as mx
import mlx.nn as nn

from minimax_h3_mlx import lora as lora_mod


class _Config:
    num_attention_heads = 4
    attention_head_dim = 8
    time_embed_dim = 16


class _Attn(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)


class _Mlp(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2, bias=False)
        self.fc2 = nn.Linear(dim * 2, dim, bias=False)


class _Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = _Attn(dim)
        self.mlp = _Mlp(dim)


class _Refiner(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.blocks = [_Block(dim)]


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Config()
        dim = _Config.num_attention_heads * _Config.attention_head_dim
        self.blocks = [_Block(dim)]
        # _iter_targets walks both trees; a fixture missing one throws.
        self.token_refiner = _Refiner(dim)


class QkvPermuteSwitch(unittest.TestCase):
    def setUp(self):
        self.model = _Model()
        self.dim = _Config.num_attention_heads * _Config.attention_head_dim
        self.name = self._qkv_name()
        rank = 2
        mx.random.seed(0)
        self.a = mx.random.normal((rank, self.dim))
        self.b = mx.random.normal((self.dim * 3, rank))
        self.pairs = {self.name: (self.a, self.b)}

    def _qkv_name(self) -> str:
        targets = lora_mod._iter_targets(self.model)
        hits = [n for n in targets if n.endswith(lora_mod.QKV_SUFFIX)]
        self.assertTrue(hits, f"fixture grew no qkv module; targets={list(targets)}")
        return hits[0]

    def test_on_reorders_rows(self):
        applicable, skipped = lora_mod.plan(self.model, self.pairs, permute_qkv=True)
        self.assertEqual(skipped, [])
        _, _, _, b = applicable[self.name]
        expected = lora_mod._permute_qkv_rows(
            self.b, _Config.num_attention_heads, _Config.attention_head_dim
        )
        self.assertTrue(mx.array_equal(b, expected))
        # A permute that changed nothing would make this test vacuous.
        self.assertFalse(mx.array_equal(b, self.b))

    def test_off_passes_rows_through(self):
        applicable, skipped = lora_mod.plan(self.model, self.pairs, permute_qkv=False)
        self.assertEqual(skipped, [])
        _, _, _, b = applicable[self.name]
        self.assertTrue(mx.array_equal(b, self.b))

    def test_default_is_off(self):
        """Default must pass rows through — the transforms are opt-in triage."""
        default, _ = lora_mod.plan(self.model, self.pairs)
        self.assertTrue(mx.array_equal(default[self.name][3], self.b))

    def test_permute_is_its_own_inverse_on_this_shape(self):
        """3 and heads=4 do not commute, so OFF is not reachable by permuting twice.

        Guards against 'just run it twice' being proposed as an undo.
        """
        once = lora_mod._permute_qkv_rows(self.b, 4, 8)
        twice = lora_mod._permute_qkv_rows(once, 4, 8)
        self.assertFalse(mx.array_equal(twice, self.b))


if __name__ == "__main__":
    unittest.main()


class Fc1HalfSwap(unittest.TestCase):
    """The fused SwiGLU projection, and why it is left alone by default."""

    def test_swap_exchanges_the_halves(self):
        b = mx.arange(8).reshape(8, 1).astype(mx.float32)
        out = lora_mod._swap_fc1_halves(b)
        self.assertEqual([float(x) for x in out.reshape(-1)],
                         [4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0])

    def test_swap_is_its_own_inverse(self):
        b = mx.random.normal((16, 3))
        self.assertTrue(mx.array_equal(
            lora_mod._swap_fc1_halves(lora_mod._swap_fc1_halves(b)), b))

    def test_file_layout_is_empty_without_metadata(self):
        """A file we cannot read must not be reported as declaring a layout."""
        self.assertEqual(lora_mod.file_layout("/nonexistent/none.safetensors"), "")
