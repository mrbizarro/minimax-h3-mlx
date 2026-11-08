"""Byte-equality contract for the streamed DiT (low-memory): resident vs streamed.

Run on a Mac with Metal: python tests/test_dit_stream.py
A tiny Q8 DiT is written as a two-shard checkpoint with quant_config.json (the layout of
h3-dit-q8), then loaded resident and streamed under several residency policies, prefetch depths,
read modes, a stacked runtime LoRA with the qkv permute, and a skipped block. Every comparison is
byte equality of the forward outputs, never a tolerance.
"""

import gc
import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace

from minimax_h3_mlx import lora as lora_mod
from minimax_h3_mlx import revoice
from minimax_h3_mlx.adaln import ModulationCache, drop_adaln_weights
from minimax_h3_mlx.config import TAG_AUDIO, TAG_TEXT, TAG_VIDEO, DiTConfig
from minimax_h3_mlx.dit import MiniMaxH3DiT
from minimax_h3_mlx.dit_stream import StreamedParam, StreamPolicy
from minimax_h3_mlx.load import load_dit
from minimax_h3_mlx.quantize import QuantConfig, quantize_dit

LAYERS = 5


def tiny_config() -> DiTConfig:
    hidden = 128
    return DiTConfig(
        hidden_size=hidden, num_layers=LAYERS, token_refiner_num_layers=1,
        num_attention_heads=2, attention_head_dim=64, ffn_hidden_size=256,
        latents_dim=4, audio_latents_dim=8, patch_size=(1, 2, 2), text_dim=64,
        timestep_input_dim=16, time_embed_hidden_size=hidden, time_embed_dim=64,
        adaln_out_features=6 * 3 * hidden, final_adaln_out_features=2 * hidden,
        rope_inv_freq_len=2,
    )


def write_checkpoint(root: Path) -> None:
    cfg = tiny_config()
    mx.random.seed(11)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    quantize_dit(dit, QuantConfig(bits=8, group_size=64, quantize_adaln=True, adaln_bits=8))
    flat = dict(tree_flatten(dit.parameters()))
    raw = asdict(cfg)
    raw["patch_size"] = list(raw["patch_size"])
    (root / "config.json").write_text(json.dumps(raw))
    (root / "quant_config.json").write_text(json.dumps(
        {"bits": 8, "group_size": 64, "quantize_adaln": True, "adaln_bits": 8}))
    keys = sorted(flat)                      # alphabetical: block tensors interleave like the real shards
    half = len(keys) // 2
    for n, part in enumerate((keys[:half], keys[half:]), start=1):
        mx.save_safetensors(str(root / f"model-0000{n}.safetensors"), {k: flat[k] for k in part})


def write_lora(path: Path, cfg: DiTConfig, ranks=None) -> None:
    rng = np.random.default_rng(5)
    tensors = {}
    for i in range(cfg.num_layers):
        rank = (ranks or {}).get(i, 4)
        for name, (out_dim, in_dim) in {
            "attn.qkv_proj": (3 * cfg.inner_dim, cfg.hidden_size),
            "attn.out_proj": (cfg.hidden_size, cfg.inner_dim),
            "mlp.fc1": (2 * cfg.ffn_hidden_size, cfg.hidden_size),
            "mlp.fc2": (cfg.hidden_size, cfg.ffn_hidden_size),
        }.items():
            base = f"diffusion_model.blocks.{i}.{name}"
            tensors[base + ".lora_A.weight"] = mx.array(
                rng.standard_normal((rank, in_dim)).astype(np.float32) * 0.05).astype(mx.bfloat16)
            tensors[base + ".lora_B.weight"] = mx.array(
                rng.standard_normal((out_dim, rank)).astype(np.float32) * 0.05).astype(mx.bfloat16)
    mx.save_safetensors(str(path), tensors)


def forward_args(cfg: DiTConfig):
    n_text, n_video, n_audio = 7, 24, 5
    seq = n_text + n_video + n_audio
    mx.random.seed(3)
    tags = mx.concatenate([mx.full((n_text,), TAG_TEXT, dtype=mx.int32),
                           mx.full((n_video,), TAG_VIDEO, dtype=mx.int32),
                           mx.full((n_audio,), TAG_AUDIO, dtype=mx.int32)])
    ts_i = mx.concatenate([mx.zeros((n_text,), dtype=mx.int32), mx.ones((n_video,), dtype=mx.int32),
                           mx.ones((n_audio,), dtype=mx.int32)])
    pos = mx.stack([mx.arange(seq) % 3, mx.arange(seq) % 5, mx.arange(seq) % 7], axis=-1).astype(mx.int32)
    return (
        mx.random.normal((1, n_video, cfg.video_patch_dim)).astype(mx.bfloat16),
        mx.random.normal((1, n_audio, cfg.audio_latents_dim)).astype(mx.bfloat16),
        mx.random.normal((1, n_text, cfg.text_dim)).astype(mx.bfloat16),
        mx.array([0.0, 0.7], dtype=mx.float32),
        ts_i, tags, pos,
        mx.arange(n_text, n_text + n_video), mx.arange(n_text + n_video, seq), mx.arange(n_text),
    )


def digest(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        mx.eval(a)
        h.update(np.array(a.astype(mx.float32)).tobytes())
    return h.hexdigest()


def run(root: Path, stream=None, lora=None, skip=None, forwards=3, permute=True, expect_adapters=None):
    dit = load_dit(root, stream=stream)
    if lora:
        lora_mod.apply_loras(dit, lora, verbose=False, permute_qkv=permute)
        if getattr(dit, "streamer", None) is not None:
            info = dit.streamer.stream_adapters(transformed=permute)
            if expect_adapters is not None:
                assert info["streamed_factors"] == expect_adapters, info
    args = forward_args(dit.config)
    cache = ModulationCache.build(dit, args[3], dtype=mx.bfloat16)
    drop_adaln_weights(dit)
    if skip:
        dit.skip_blocks = frozenset(skip)
    out = []
    for _ in range(forwards):                  # several forwards: slot reuse wraps across them
        v, a = dit(*args, modulation_cache=cache)
        out.append(digest(v, a))
    streamer = getattr(dit, "streamer", None)
    if streamer is not None:
        for block in streamer.streamed:        # nothing stays attached between forwards
            for part, param, _ in streamer.plan[block]:
                w = dit.blocks[block]
                for name in part.split("."):
                    w = getattr(w, name)
                if param.startswith("lora_"):
                    which, _, k = param.partition(":")
                    assert isinstance(getattr(w, which)[int(k)], StreamedParam), (block, part, param)
                    continue
                w = getattr(w, "base", w)
                assert isinstance(getattr(w, param), StreamedParam) and param not in w, (block, part)
        streamer.close()
    del dit, cache
    gc.collect()
    mx.clear_cache()
    return out


class StreamedDiTTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "dit-q8"
        cls.root.mkdir()
        write_checkpoint(cls.root)
        cls.lora = Path(cls.tmp.name) / "lora.safetensors"
        write_lora(cls.lora, tiny_config())
        cls.resident = run(cls.root)
        cls.resident_lora = run(cls.root, lora=[(cls.lora, 1.0), (cls.lora, 0.5)])
        cls.resident_skip = run(cls.root, skip=[2])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_resident_is_deterministic(self):
        self.assertEqual(run(self.root), self.resident)

    def test_policies_byte_identical(self):
        for spec, prefetch, nocache in [
            ("none", 1, True), ("none", 0, True), ("none", 2, False), ("none", 9, True),
            ("out,fc2", 1, True), ("first:2", 1, True), ("blocks:1-3", 1, False),
            ("qkv,fc1,first:1", 2, True), ("gb:1000", 1, True),
        ]:
            for pipeline in (False, True):
                with self.subTest(spec=spec, prefetch=prefetch, nocache=nocache, pipeline=pipeline):
                    policy = StreamPolicy.parse(spec, prefetch=prefetch, workers=3, nocache=nocache,
                                                pipeline=pipeline)
                    self.assertEqual(run(self.root, stream=policy), self.resident)

    def test_lora_stack_with_qkv_permute(self):
        policy = StreamPolicy.parse("none", prefetch=1)
        got = run(self.root, stream=policy, lora=[(self.lora, 1.0), (self.lora, 0.5)])
        self.assertEqual(got, self.resident_lora)
        self.assertNotEqual(self.resident_lora, self.resident)

    def test_streamed_adapter_factors(self):
        stack = [(self.lora, 1.0), (self.lora, 0.5)]
        resident = run(self.root, lora=stack, permute=False)
        self.assertNotEqual(resident, self.resident)
        for spec in ("none", "out,first:2"):
            with self.subTest(spec=spec):
                parts = 4      # factors stream even where the base part is resident
                got = run(self.root, stream=StreamPolicy.parse(spec, prefetch=1), lora=stack,
                          permute=False, expect_adapters=parts * 2 * 2)
                self.assertEqual(got, resident)

    def test_variable_rank_adapter_stays_resident(self):
        varied = Path(self.tmp.name) / "lora_varied.safetensors"
        write_lora(varied, tiny_config(), ranks={3: 2, 4: 2})
        resident = run(self.root, lora=[(varied, 1.0)], permute=False)
        got = run(self.root, stream=StreamPolicy.parse("none", prefetch=1), lora=[(varied, 1.0)],
                  permute=False, expect_adapters=0)
        self.assertEqual(got, resident)

    def test_forward_recovers_after_an_exception_mid_block(self):
        from minimax_h3_mlx.dit import TransformerBlock

        class Boom(TransformerBlock):
            def __call__(self, *a, **k):
                raise RuntimeError("boom")

        dit = load_dit(self.root, stream=StreamPolicy.parse("none", prefetch=1, pipeline=True))
        args = forward_args(dit.config)
        cache = ModulationCache.build(dit, args[3], dtype=mx.bfloat16)
        drop_adaln_weights(dit)
        dit.blocks[2].__class__ = Boom
        with self.assertRaises(RuntimeError):
            dit(*args, modulation_cache=cache)
        dit.blocks[2].__class__ = TransformerBlock
        streamer = dit.streamer
        self.assertTrue(all(s.block is None and not s.in_use for s in streamer.slots))
        out = [digest(*dit(*args, modulation_cache=cache)) for _ in range(3)]
        self.assertEqual(out, self.resident)
        self.assertEqual(streamer.summary()["slots"], 3)
        streamer.close()
        self.assertEqual(streamer.summary()["slots"], 3)

    def test_skip_blocks(self):
        for spec in ("none", "first:3"):
            with self.subTest(spec=spec):
                for pipeline in (False, True):
                    got = run(self.root, stream=StreamPolicy.parse(spec, prefetch=1, pipeline=pipeline),
                              skip=[2])
                    self.assertEqual(got, self.resident_skip)

    def test_load_never_reads_streamed_tensors(self):
        gc.collect()
        mx.clear_cache()
        before = mx.get_active_memory()
        resident = load_dit(self.root)
        resident_bytes = mx.get_active_memory() - before
        del resident
        gc.collect()
        mx.clear_cache()
        before = mx.get_active_memory()
        streamed = load_dit(self.root, stream=StreamPolicy.parse("none", prefetch=1))
        streamed_bytes = mx.get_active_memory() - before
        streamer = streamed.streamer
        info = streamer.summary()
        streamed_total = sum(streamer.slot_bytes_for(b) for b in streamer.streamed)
        slots = len(streamer.slots) * streamer.slot_bytes
        # Resident minus every streamed tensor, plus the slot pool (rounding slack 1 %).
        expect = resident_bytes - streamed_total + slots
        self.assertLess(abs(streamed_bytes - expect), 0.01 * resident_bytes + 4096)
        self.assertEqual(info["streamed_blocks"], LAYERS)
        streamed.streamer.close()

    def test_revoice_paths(self):
        def revoice_digests(stream):
            dit = load_dit(self.root, stream=stream)
            args = forward_args(dit.config)
            cache = ModulationCache.build(dit, args[3], dtype=mx.bfloat16)
            drop_adaln_weights(dit)
            seq = int(args[6].shape[0])
            layout = SimpleNamespace(sequence_length=seq, position_ids=args[6], token_tags=args[5],
                                     video_indices=args[7], audio_indices=args[8], text_indices=args[9],
                                     num_condition_video_rows=0, num_condition_audio_rows=0)
            plan = revoice.plan_rows(layout)
            stored, store = revoice.store_forward(dit, args, cache, plan, "bf16")
            audio = revoice.audio_forward(dit, args[1][0], args[3], args[4], layout, plan, store, cache,
                                          dit.rope(layout.position_ids))
            out = [digest(stored), digest(audio)]
            if getattr(dit, "streamer", None) is not None:
                dit.streamer.close()
            return out

        resident = revoice_digests(None)
        for spec in ("none", "out,fc2,first:1"):
            with self.subTest(spec=spec):
                self.assertEqual(revoice_digests(StreamPolicy.parse(spec, prefetch=1)), resident)

    def test_policy_parse(self):
        p = StreamPolicy.parse("out,fc2,first:2,blocks:7-8,gb:1.5")
        self.assertEqual(p.resident_parts, frozenset({"out", "fc2"}))
        self.assertEqual(p.resident_blocks, frozenset({0, 1, 7, 8}))
        self.assertEqual(p.resident_gb, 1.5)
        with self.assertRaises(ValueError):
            StreamPolicy.parse("attn")


if __name__ == "__main__":
    unittest.main(verbosity=2)
