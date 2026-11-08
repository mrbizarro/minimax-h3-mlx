"""The low-memory plan and admission check (exp/h3-24gb).

Pure arithmetic on tiny synthetic safetensors headers: no model, no GPU.
"""

import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from minimax_h3_mlx import memory_plan as mp

GIB = 1024 ** 3


def _write_header(path: Path, tensors: dict) -> None:
    """A safetensors file whose header claims these byte sizes (payload is sparse zeros)."""
    header, offset = {}, 0
    for name, nbytes in tensors.items():
        header[name] = {"dtype": "U8", "shape": [nbytes], "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    blob = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.truncate(8 + len(blob) + offset)


def fixture(root: Path) -> None:
    compact = root / "compact"
    dit = root / "dit"
    compact.mkdir()
    dit.mkdir()
    mib = 1024 ** 2
    te = {"model.embed_tokens.weight": 300 * mib}
    for i in range(52):                      # 50 used, 2 beyond the conditioning layer
        te[f"model.layers.{i}.mlp.weight"] = 100 * mib
    te["visual.blocks.0.attn.weight"] = 40 * mib
    te["lm_head.weight"] = 500 * mib         # never counted
    _write_header(compact / "text_encoder.safetensors", te)
    vae = {f"decoder.transformer_blocks.{i}.w": 100 * mib for i in range(36)}
    vae["decoder.proj_out.weight"] = 1 * mib
    vae["encoder.conv_in.weight"] = 300 * mib
    _write_header(compact / "video_vae.safetensors", vae)
    _write_header(compact / "audio_vae.safetensors", {"w": 8 * mib})
    blocks = {}
    for b in range(50):
        for part in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
            blocks[f"blocks.{b}.{part}.weight"] = 2 * mib
        blocks[f"blocks.{b}.norm1.weight"] = 1 * mib
    blocks["final_layer.video_out.weight"] = 10 * mib
    _write_header(dit / "model-00001.safetensors", blocks)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        fixture(self.root)
        self.sizes = mp.checkpoint_sizes(self.root / "compact", self.root / "dit")

    def tearDown(self):
        self.tmp.cleanup()

    def test_sizes_come_from_headers(self):
        s = self.sizes
        mib = 1 / 1024
        self.assertAlmostEqual(s.te_total, (300 + 50 * 100) * mib, places=6)   # layers >= 50 skipped
        self.assertAlmostEqual(s.te_vision, 40 * mib, places=6)
        self.assertAlmostEqual(s.te_max_unit, 300 * mib, places=6)
        self.assertAlmostEqual(s.dit_block_stream, 8 * mib, places=6)
        self.assertAlmostEqual(s.dit_nonblock, (50 + 10) * mib, places=6)
        self.assertAlmostEqual(s.dit_total, (50 * 9 + 10) * mib, places=6)

    def test_rows_match_the_runner_layout(self):
        # 768x448x124 with 229 prompt tokens is 13,075 packed rows in the runner's own log.
        self.assertEqual(mp.packed_rows(768, 448, 124, 229), 13075)
        self.assertGreater(mp.packed_rows(1024, 576, 124, 229), mp.packed_rows(768, 448, 124, 229))

    def test_auto_keeps_todays_path_when_it_fits(self):
        rows = mp.packed_rows(768, 448, 124, 229)
        plan = mp.choose(self.sizes, rows, 768, 448, 1000.0, vision=False, mode="auto")
        self.assertEqual(plan.mode, "resident")
        self.assertFalse(plan.stream_dit)
        off = mp.choose(self.sizes, rows, 768, 448, 0.1, vision=False, mode="off")
        self.assertEqual(off.mode, "resident")

    def test_auto_streams_when_the_resident_peak_does_not_fit(self):
        rows = mp.packed_rows(768, 448, 124, 229)
        resident = mp.choose(self.sizes, rows, 768, 448, 1000.0, vision=False, mode="off")
        plan = mp.choose(self.sizes, rows, 768, 448, resident.peak_gib - 0.5, vision=False, mode="auto")
        self.assertEqual(plan.mode, "lowmem")
        self.assertTrue(plan.stream_text_encoder and plan.stream_dit and plan.dit_pipeline)
        self.assertLess(plan.peak_gib, resident.peak_gib)

    def test_auto_streams_when_other_apps_hold_the_memory(self):
        rows = mp.packed_rows(768, 448, 124, 229)
        resident = mp.choose(self.sizes, rows, 768, 448, 1000.0, vision=False, mode="off")
        busy_desk = mp.choose(self.sizes, rows, 768, 448, 1000.0, vision=False, mode="auto",
                              available_gib=resident.peak_gib - 0.5)
        self.assertEqual(busy_desk.mode, "lowmem")
        quiet = mp.choose(self.sizes, rows, 768, 448, 1000.0, vision=False, mode="auto",
                          available_gib=resident.peak_gib + 0.5)
        self.assertEqual(quiet.mode, "resident")

    def test_decode_ladder_only_steps_down_when_needed(self):
        rows = mp.packed_rows(768, 448, 124, 229)
        roomy = mp.choose(self.sizes, rows, 768, 448, 100.0, vision=False, mode="on")
        self.assertEqual((roomy.stream_vae_decoder, roomy.vae_batch), (False, 8))
        tight = mp.choose(self.sizes, rows, 768, 448, roomy.passes["decode"] - 0.01, vision=False, mode="on")
        self.assertEqual((tight.stream_vae_decoder, tight.vae_batch), (True, 8))
        self.assertLess(tight.passes["decode"], roomy.passes["decode"])
        tighter = mp.choose(self.sizes, rows, 768, 448, tight.passes["decode"] - 0.01, vision=False, mode="on")
        self.assertEqual((tighter.stream_vae_decoder, tighter.vae_batch), (True, 1))

    def test_vae_sizes_split_blocks_from_the_rest(self):
        mib = 1 / 1024
        self.assertAlmostEqual(self.sizes.vae_block, 100 * mib, places=6)
        self.assertAlmostEqual(self.sizes.vae_decode_fixed, 1 * mib, places=6)   # encoder excluded

    def test_admission_tells_cannot_from_busy(self):
        plan = mp.Plan("lowmem", peak_pass="decode", peak_gib=10.0)
        small = {"ram_gib": 15.0, "available_gib": 14.0}      # gives a render 15 - 6 = 9 GiB
        cannot = mp.admit(plan, small)
        self.assertEqual(cannot.verdict, "cannot")
        self.assertIn("cannot make this clip", cannot.message)
        mid = {"ram_gib": 24.0, "available_gib": 6.0}
        busy = mp.admit(plan, mid)
        self.assertEqual(busy.verdict, "busy")
        self.assertIn("other apps", busy.message)
        roomy = {"ram_gib": 24.0, "available_gib": 15.0}
        self.assertEqual(mp.admit(plan, roomy).verdict, "ok")

    def test_simulated_small_mac(self):
        with mock.patch.dict(os.environ, {"H3_SIMULATE_RAM_GB": "24"}):
            mem = mp.machine_memory({"max_recommended_working_set_size": 48 * GIB})
        self.assertEqual(mem["ram_gib"], 24.0)
        self.assertTrue(mem["simulated"])
        self.assertLessEqual(mem["available_gib"], 24.0 - mp.os_reserve_gib(24.0))

    def test_plan_for_request_off_changes_nothing(self):
        args = SimpleNamespace(
            compact_root=self.root / "compact", dit=self.root / "dit", lora=None, width=768,
            height=448, memory_gb=56.0, low_memory="off", reference=None, extend_from=None,
            lora_qkv_permute=False, lora_fc1_swap=False, revoice_steps=0, revoice_kv="int8",
            stream_text_encoder=False, stream_dit=False, dit_pipeline=False, cache_gb=None,
            decode_cache_gb=None)
        before = dict(vars(args))
        rec = mp.plan_for_request(args, {"max_recommended_working_set_size": 48 * GIB}, 124, 1, [[]], ["p" * 900])
        self.assertEqual(rec["mode"], "resident")
        self.assertEqual(vars(args), before)

    def test_plan_for_request_on_sets_the_low_memory_flags(self):
        args = SimpleNamespace(
            compact_root=self.root / "compact", dit=self.root / "dit", lora=None, width=768,
            height=448, memory_gb=12.0, low_memory="on", reference=None, extend_from=None,
            lora_qkv_permute=True, lora_fc1_swap=False, revoice_steps=0, revoice_kv="int8",
            stream_text_encoder=False, stream_dit=False, dit_pipeline=False, cache_gb=None,
            decode_cache_gb=None)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("H3_VAE_BATCH", None)
            rec = mp.plan_for_request(args, {"max_recommended_working_set_size": 48 * GIB}, 124, 1, [[]], ["p" * 900])
            os.environ.pop("H3_VAE_BATCH", None)
        self.assertEqual(rec["mode"], "lowmem")
        self.assertTrue(args.stream_text_encoder and args.stream_dit and args.dit_pipeline)
        self.assertEqual(args.cache_gb, 2.0)
        self.assertEqual(args.decode_cache_gb, 1.0)


if __name__ == "__main__":
    unittest.main()
