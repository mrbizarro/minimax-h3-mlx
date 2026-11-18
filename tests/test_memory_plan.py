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

    def test_decode_ladder_streams_first_and_steps_down_when_needed(self):
        # A low-memory plan always streams the decoder (the lower peak beats ~2 s on a small Mac);
        # one tile per call only when the big batch does not fit.
        rows = mp.packed_rows(768, 448, 124, 229)
        roomy = mp.choose(self.sizes, rows, 768, 448, 100.0, vision=False, mode="on")
        self.assertEqual((roomy.stream_vae_decoder, roomy.vae_batch), (True, 8))
        tighter = mp.choose(self.sizes, rows, 768, 448, roomy.passes["decode"] - 0.01, vision=False, mode="on")
        self.assertEqual((tighter.stream_vae_decoder, tighter.vae_batch), (True, 1))
        self.assertLess(tighter.passes["decode"], roomy.passes["decode"])

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
        self.assertEqual(args.cache_gb, mp.LOWMEM_CACHE_GIB)
        self.assertEqual(args.decode_cache_gb, mp.LOWMEM_DECODE_CACHE_GIB)


    def _revoice_args(self, memory_gb, low_memory):
        return SimpleNamespace(
            compact_root=self.root / "compact", dit=self.root / "dit", lora=None, width=768,
            height=448, memory_gb=memory_gb, low_memory=low_memory, reference=None,
            extend_from=None, lora_qkv_permute=True, lora_fc1_swap=False, revoice_steps=9,
            revoice_kv="int8", revoice_kv_store="auto", stream_text_encoder=False,
            stream_dit=False, dit_pipeline=False, cache_gb=None, decode_cache_gb=None)

    def test_revoice_store_goes_to_disk_when_it_does_not_fit(self):
        # Standard 5 s: the int8 store is ~9 GiB, more than a 16 GB Mac has beside the render.
        args = self._revoice_args(10.0, "on")
        env = {"H3_SIMULATE_RAM_GB": "16", "H3_SIMULATE_AVAILABLE_GB": "9"}
        with mock.patch.dict(os.environ, env):
            rec = mp.plan_for_request(args, {"max_recommended_working_set_size": 48 * GIB}, 124, 1, [[]], ["p" * 900])
            os.environ.pop("H3_VAE_BATCH", None)
        self.assertEqual(rec["mode"], "lowmem")
        self.assertEqual(args.revoice_kv_store, "disk")
        self.assertEqual(rec["revoice_kv_store"], "disk")
        # On disk the pass costs the denoise plus two blocks of the store, not the whole store.
        self.assertLess(rec["passes_gib"]["revoice"] - rec["passes_gib"]["denoise"], 1.0)

    def test_revoice_store_stays_in_memory_when_it_fits(self):
        args = self._revoice_args(56.0, "auto")
        with mock.patch.dict(os.environ, {"H3_SIMULATE_AVAILABLE_GB": "40"}):
            rec = mp.plan_for_request(args, {"max_recommended_working_set_size": 48 * GIB}, 124, 1, [[]], ["p" * 900])
        self.assertEqual(rec["mode"], "resident")
        self.assertEqual(args.revoice_kv_store, "memory")
        self.assertGreater(rec["passes_gib"]["revoice"] - rec["passes_gib"]["denoise"], 5.0)

    def test_an_explicit_store_choice_is_kept(self):
        args = self._revoice_args(10.0, "on")
        args.revoice_kv_store = "memory"
        with mock.patch.dict(os.environ, {"H3_SIMULATE_RAM_GB": "16", "H3_SIMULATE_AVAILABLE_GB": "9"}):
            rec = mp.plan_for_request(args, {"max_recommended_working_set_size": 48 * GIB}, 124, 1, [[]], ["p" * 900])
            os.environ.pop("H3_VAE_BATCH", None)
        self.assertEqual(args.revoice_kv_store, "memory")
        self.assertEqual(rec["admission"]["verdict"], "cannot")


    def test_a_chain_prices_the_carried_context_encode(self):
        args = self._revoice_args(10.0, "on")
        args.revoice_steps = 0
        with mock.patch.dict(os.environ, {"H3_SIMULATE_RAM_GB": "16", "H3_SIMULATE_AVAILABLE_GB": "10"}):
            rec = mp.plan_for_request(args, {"max_recommended_working_set_size": 48 * GIB}, 124, 2, [[], []], ["p" * 900])
            os.environ.pop("H3_VAE_BATCH", None)
        self.assertIn("media_encode", rec["passes_gib"])
        self.assertEqual(rec["admission"]["verdict"], "ok")
        one_call = round((mp.MEDIA_ENCODE_GIB[0] + mp.LOWMEM_CACHE_GIB + mp.RUNTIME_GIB) * mp.MARGIN, 2)
        self.assertLess(rec["passes_gib"]["media_encode"], one_call)
        # One clip, no carried frames: the pass does not exist.
        args2 = self._revoice_args(10.0, "on")
        args2.revoice_steps = 0
        rec2 = mp.plan_for_request(args2, {"max_recommended_working_set_size": 48 * GIB}, 124, 1, [[]], ["p" * 900])
        os.environ.pop("H3_VAE_BATCH", None)
        self.assertNotIn("media_encode", rec2["passes_gib"])


# Real H3 checkpoint sizes (GiB) as the runner reports them in its metrics (Q8 DiT, 17 Nov 2026).
REAL_SIZES = mp.CheckpointSizes(
    te_total=25.574, te_vision=0.71, te_max_unit=1.449, te_max_layer=0.483,
    te_max_vision_unit=0.044, dit_total=20.213, dit_block_stream=0.381, dit_nonblock=1.147,
    video_vae=4.85, audio_vae=0.564, vae_block=0.125, vae_decode_fixed=0.012)
STREAMED_WEIGHTS = 0.012 + 2 * 0.125

# Measured decoder peaks (MLX GiB, M4 Max, mlx 0.32.4, 124 frames), from
# today-2026-11-17/m5-next/out/decode.json and decode_streamed.json. Byte-identical frames in
# every mode. Key: (width, height, streamed, tile batch) -> peak including the decoder weights.
MEASURED_PEAKS = {
    (512, 288, False, 8): 6.704, (512, 288, False, 1): 6.245,
    (768, 448, False, 8): 7.126, (768, 448, False, 1): 6.529,
    (1024, 576, False, 8): 7.832, (1024, 576, False, 1): 7.058,
    (768, 448, True, 8): 2.276, (768, 448, True, 1): 1.683,
    (1024, 576, True, 8): 2.677, (1024, 576, True, 1): 2.232,
}
RESIDENT_WEIGHTS = 4.85


def _work(w, h, streamed, batch, frames=124):
    """The curve's MLX working set alone (the host clip term taken out)."""
    host = frames * w * h * 3 / GIB
    return mp.decode_work_gib(w, h, batch, streamed=streamed, frames=frames) - host


class DecodeCurveTests(unittest.TestCase):
    """The decoder estimate follows the measured curve (H3 planner over-estimate, 17 Nov 2026)."""

    def test_curve_covers_every_measured_point(self):
        for (w, h, streamed, batch), peak in MEASURED_PEAKS.items():
            weights = STREAMED_WEIGHTS if streamed else RESIDENT_WEIGHTS
            with self.subTest(w=w, h=h, streamed=streamed, batch=batch):
                self.assertGreaterEqual(_work(w, h, streamed, batch), peak - weights)

    def test_streamed_curve_is_tight_to_the_measurement(self):
        # The low-memory decoder is what 16-24 GB Macs run: at most 0.4 GiB of headroom per point
        # before MARGIN, where the old flat term put 1.2-3.0 GiB on top.
        for (w, h, streamed, batch), peak in MEASURED_PEAKS.items():
            if not streamed:
                continue
            with self.subTest(w=w, h=h, batch=batch):
                over = _work(w, h, True, batch) - (peak - STREAMED_WEIGHTS)
                self.assertLessEqual(over, 0.40)

    def test_1024x576_streamed_no_longer_budgets_5_gib(self):
        # Report: 5.4 GiB budgeted for decoder working memory, about 2.4-3.0 used.
        self.assertLess(_work(1024, 576, True, 8), 3.0)
        old = 9.2 * 1024 * 576 / 1e6
        self.assertGreater(old, 5.4)
        rows = mp.packed_rows(1024, 576, 124, 300)
        est = mp.estimate(REAL_SIZES, rows, 1024, 576, vision=False, lowmem=True, vae_batch=8,
                          stream_vae=True)
        # (0.26 weights + 2.8 work + 0.21 clip + 1.0 cache + 0.5 runtime) x 1.10
        self.assertLess(est["decode"], 5.5)
        self.assertGreaterEqual(est["decode"], round((MEASURED_PEAKS[(1024, 576, True, 8)]
                                                      + mp.RUNTIME_GIB) * mp.MARGIN, 2))

    def test_real_render_streamed_decode_is_covered(self):
        # weaker-macs-gpu/renders/fast_16: 640x384, streamed, batch 8, video_vae_decode 2.05 GiB.
        self.assertGreaterEqual(_work(640, 384, True, 8) + STREAMED_WEIGHTS, 2.05)

    def test_resident_stays_conservative(self):
        # Inside a full render the resident decoder peaked above the isolated bench: 8.00 GiB MLX
        # at 768x448 batch 8 (3.15 working) and 7.19 at batch 1 (2.34 working). Both covered.
        self.assertGreaterEqual(_work(768, 448, False, 8), 3.15)
        self.assertGreaterEqual(_work(768, 448, False, 1), 2.34)
        for w, h in ((512, 288), (768, 448), (1024, 576), (1280, 720)):
            for batch in (1, 8):
                with self.subTest(w=w, h=h, batch=batch):
                    self.assertGreater(_work(w, h, False, batch), _work(w, h, True, batch))

    def test_unmeasured_canvas_uses_the_old_slope_for_the_excess(self):
        top = _work(1024, 576, True, 8)
        mp_excess = (1280 * 720 - 1024 * 576) / 1e6
        self.assertAlmostEqual(_work(1280, 720, True, 8), top + 9.2 * mp_excess, places=6)
        self.assertGreater(_work(1920, 1088, True, 1), _work(1280, 720, True, 1))

    def test_longer_clips_add_their_finished_frames(self):
        short = mp.decode_work_gib(1024, 576, 8, streamed=True, frames=124)
        long = mp.decode_work_gib(1024, 576, 8, streamed=True, frames=244)
        self.assertAlmostEqual(long - short, 120 * 1024 * 576 * 3 / GIB, places=9)
        self.assertEqual(mp.decode_work_gib(1024, 576, 8, streamed=True),
                         mp.decode_work_gib(1024, 576, 8, streamed=True, frames=124))

    def test_chained_windows_price_all_their_frames(self):
        args = SimpleNamespace(compact_root=None, dit=None, lora=None, reference=None,
                               extend_from=None, width=1024, height=576, memory_gb=10.0,
                               low_memory="auto", lora_qkv_permute=False, lora_fc1_swap=False,
                               revoice_steps=0, stream_vae_decoder=False, cache_gb=None,
                               decode_cache_gb=None, stream_text_encoder=False, stream_dit=False,
                               dit_pipeline=False)
        mem = {"ram_gib": 16.0, "available_gib": 10.0, "gpu_working_set_gib": 12.0,
               "swap_used_gib": 0.0, "simulated": True}
        seen = {}

        def spy(*a, **kw):
            seen["frames"] = kw.get("frames")
            return mp.Plan("lowmem", passes={"decode": 1.0}, peak_pass="decode", peak_gib=1.0)

        with mock.patch.object(mp, "checkpoint_sizes", return_value=REAL_SIZES), \
                mock.patch.object(mp, "machine_memory", return_value=mem), \
                mock.patch.object(mp, "choose", side_effect=spy):
            mp.plan_for_request(args, {}, 124, 3, [[]], ["p"])
        self.assertEqual(seen["frames"], 372)


class DecodeLadderOnRealSizesTests(unittest.TestCase):
    """What a 16 GB Mac gets (budget 10 GiB, the Studio's h3_memory_budget)."""

    def _plan(self, w, h, available):
        rows = mp.packed_rows(w, h, 124, 300)
        return mp.choose(REAL_SIZES, rows, w, h, 10.0, vision=False, mode="auto",
                         available_gib=available)

    def test_1024x576_keeps_eight_tiles_when_the_decoder_fits(self):
        # Before: a tight denoise (9.75 GiB) walked the decoder down to one tile per call (+13 %
        # decode time) at 8 GiB free, though the streamed eight-tile decode needs about 5.3.
        for available in (10.0, 8.0, 7.0, 6.0):
            with self.subTest(available=available):
                plan = self._plan(1024, 576, available)
                self.assertEqual((plan.mode, plan.stream_vae_decoder, plan.vae_batch),
                                 ("lowmem", True, 8))
                self.assertLessEqual(plan.passes["decode"], available)

    def test_one_tile_only_when_the_decode_itself_does_not_fit(self):
        plan = self._plan(1280, 720, 8.0)
        self.assertEqual((plan.stream_vae_decoder, plan.vae_batch), (True, 1))
        rows = mp.packed_rows(1280, 720, 124, 300)
        b8 = mp.estimate(REAL_SIZES, rows, 1280, 720, vision=False, lowmem=True, vae_batch=8,
                         stream_vae=True)
        self.assertGreater(b8["decode"], 8.0)

    def test_admission_still_judges_the_largest_pass(self):
        plan = self._plan(1024, 576, 6.0)
        self.assertEqual(plan.peak_pass, "denoise")
        mem = {"ram_gib": 16.0, "available_gib": 6.0}
        self.assertEqual(mp.admit(plan, mem, 10.0).verdict, "busy")


if __name__ == "__main__":
    unittest.main()
