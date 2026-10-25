"""Re-voice: the KV-cached audio-only forward against the full forward, on a tiny DiT."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from minimax_h3_mlx import lora as lora_mod
from minimax_h3_mlx import revoice
from minimax_h3_mlx.adaln import ModulationCache
from minimax_h3_mlx.dit import MiniMaxH3DiT
from test_dit_smoke import build_packed_layout, tiny_config


def _setup(n_text=5, n_video=9, n_audio=4, n_cond_a=0):
    cfg = tiny_config()
    mx.random.seed(3)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())
    text_i, video_i, audio_i, tags, ts_i, pos = build_packed_layout(n_text, n_video, n_audio)
    layout = SimpleNamespace(
        sequence_length=n_text + n_video + n_audio, position_ids=pos, token_tags=tags,
        video_indices=video_i, audio_indices=audio_i, text_indices=text_i,
        num_condition_video_rows=0, num_condition_audio_rows=n_cond_a)
    table = mx.array([0.0, 0.3, 1.0])
    # audio rows on level 1 (moving), video on 2 (held), text on 0
    ts = np.zeros(layout.sequence_length, dtype=np.int32)
    ts[np.array(video_i.tolist())] = 2
    ts[np.array(audio_i.tolist())] = 1
    if n_cond_a:
        ts[np.array(audio_i.tolist())[:n_cond_a]] = 2
    cache = ModulationCache.build(dit, table, dtype=mx.float32)
    video = mx.random.normal((1, n_video, cfg.video_patch_dim))
    audio = mx.random.normal((1, n_audio, cfg.audio_latents_dim))
    text = mx.random.normal((1, n_text, cfg.text_dim))
    args = (video, audio, text, table, mx.array(ts), tags, pos, video_i, audio_i, text_i)
    return dit, layout, cache, args


def test_plan_rows_partitions_the_sequence():
    _, layout, _, _ = _setup(n_cond_a=1)
    plan = revoice.plan_rows(layout)
    dyn = set(plan.dynamic.tolist()); cached = set(plan.cached.tolist())
    assert not dyn & cached
    assert dyn | cached == set(range(layout.sequence_length))
    assert plan.dynamic.tolist() == layout.audio_indices.tolist()[1:]


def test_store_forward_is_the_full_forward():
    dit, layout, cache, args = _setup()
    _, full_audio = dit(*args, modulation_cache=cache)
    stored_audio, store = revoice.store_forward(dit, args, cache, revoice.plan_rows(layout), "bf16")
    np.testing.assert_allclose(np.array(stored_audio), np.array(full_audio), rtol=1e-5, atol=1e-5)
    assert len(store.k) == len(dit.blocks) and store.nbytes > 0


def _audio_only(dit, layout, cache, args, store, audio_rows):
    return revoice.audio_forward(dit, audio_rows, args[3], args[4], layout, revoice.plan_rows(layout),
                                 store, cache, dit.rope(layout.position_ids))


def test_audio_forward_matches_full_forward_on_the_stored_state():
    # Same audio as the store pass -> every cached K/V is exact, so the audio-only forward must
    # reproduce the full forward (the cache's indices, rotary and modulation rows are right).
    dit, layout, cache, args = _setup()
    _, full_audio = dit(*args, modulation_cache=cache)
    _, store = revoice.store_forward(dit, args, cache, revoice.plan_rows(layout), "bf16")
    got = _audio_only(dit, layout, cache, args, store, args[1][0])
    np.testing.assert_allclose(np.array(got), np.array(full_audio[0]), rtol=1e-4, atol=1e-4)


def test_int8_store_stays_close():
    dit, layout, cache, args = _setup()
    _, full_audio = dit(*args, modulation_cache=cache)
    _, store = revoice.store_forward(dit, args, cache, revoice.plan_rows(layout), "int8")
    got = np.array(_audio_only(dit, layout, cache, args, store, args[1][0]))
    ref = np.array(full_audio[0])
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel < 0.02, rel


def test_conditioning_audio_rows_are_returned_as_zeros_and_targets_match():
    dit, layout, cache, args = _setup(n_audio=5, n_cond_a=2)
    _, full_audio = dit(*args, modulation_cache=cache)
    _, store = revoice.store_forward(dit, args, cache, revoice.plan_rows(layout), "bf16")
    got = np.array(_audio_only(dit, layout, cache, args, store, args[1][0]))
    assert got.shape == np.array(full_audio[0]).shape
    assert np.all(got[:2] == 0)
    np.testing.assert_allclose(got[2:], np.array(full_audio[0])[2:], rtol=1e-4, atol=1e-4)


def test_audio_forward_moves_with_new_audio():
    dit, layout, cache, args = _setup()
    _, store = revoice.store_forward(dit, args, cache, revoice.plan_rows(layout), "bf16")
    a = np.array(_audio_only(dit, layout, cache, args, store, args[1][0]))
    b = np.array(_audio_only(dit, layout, cache, args, store, args[1][0] * 0.5))
    assert not np.allclose(a, b)


def _attach(dit, src, scale=1.0):
    # One adapter on every block's qkv_proj, recorded under `src`.
    for block in dit.blocks:
        base = block.attn.qkv_proj
        a = mx.random.normal((2, base.weight.shape[1])) * 0.1
        b = mx.random.normal((base.weight.shape[0], 2)) * 0.1
        if isinstance(base, lora_mod.LoRALinear):
            base.add(a, b, scale, source=src)
        else:
            block.attn.qkv_proj = lora_mod.LoRALinear(base, a, b, scale, source=src)


def test_set_adapter_scales_switches_only_the_named_adapter_and_restores():
    dit, layout, cache, args = _setup()
    _attach(dit, "/m/speed.safetensors")
    _attach(dit, "/m/character.safetensors", 0.8)
    out = revoice.set_adapter_scales(dit, ["/m/speed.safetensors"], 0.0)
    assert out["entries"] == len(dit.blocks)
    for block in dit.blocks:
        assert block.attn.qkv_proj.lora_scales == [0.0, 0.8]
    revoice.set_adapter_scales(dit, ["/m/speed.safetensors"], None)
    for block in dit.blocks:
        assert block.attn.qkv_proj.lora_scales == [1.0, 0.8]


def test_start_sigmas():
    full = [1.0, 0.955, 0.9, 0.833, 0.75, 0.643, 0.5, 0.3, 0.0]
    assert revoice.start_sigmas(full, 1.0) == full
    assert revoice.start_sigmas(full, 0.75) == [0.75, 0.643, 0.5, 0.3, 0.0]
    assert revoice.start_sigmas(full, 0.7) == [0.7, 0.643, 0.5, 0.3, 0.0]


def test_kv_estimate():
    assert revoice.kv_bytes_estimate(1000, 50, 7168, "off") == 0
    bf16 = revoice.kv_bytes_estimate(1000, 50, 7168, "bf16")
    int8 = revoice.kv_bytes_estimate(1000, 50, 7168, "int8")
    assert bf16 == 2 * 1000 * 7168 * 50 * 2 and int8 < bf16 * 0.6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
