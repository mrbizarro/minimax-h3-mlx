"""Re-voice: re-make a finished window's sound with the speed adapter switched off.

A step-distilled speed adapter (TaoMate's 3-step ladder, LightX2V Turbo) renders the picture well
but its SOUND has been reported harsh, thin and echo-like (a third-party Mac developer, Oct 2026,
on LightX2V Turbo). Their fix, reproduced here for our runner: keep the finished picture, switch
the adapter off, and denoise only the audio rows again with the base model.

The pass is cheap because almost every row of the packed sequence does not change while the sound
is re-made. The picture is held clean, the text and keyframe rows are pinned at constant noise
levels, so their attention keys and values are computed ONCE (the "store" forward) and every later
step runs only the audio rows - about 1-5 % of the sequence - through the 50 blocks, attending to
the stored keys/values plus their own. The one approximation: the stored rows saw the step-0
(noise) audio and are not refreshed as the sound forms. ``kv="off"`` runs the exact full forward
every step instead, which is what an A/B against the cached pass compares to.

Memory: the store holds ``2 x rows x 7168`` values per block. bf16 is ~1.43 MB per cached row over
50 blocks (Draft 5 s ~8 GB, Standard ~20 GB, High ~33 GB); ``int8`` (default) halves that with an
affine 8-bit quantisation per 128-wide head vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .dit import apply_rotary, param_dtype

KV_MODES = ("int8", "bf16", "off")


@dataclass
class RevoicePlan:
    """Which packed rows move (the target audio rows) and which are stored once."""

    dynamic: mx.array          # packed indices of the target audio rows, in audio-row order
    cached: mx.array           # packed indices of every other row
    num_audio_cond: int


def plan_rows(layout) -> RevoicePlan:
    audio_idx = np.asarray(layout.audio_indices.tolist(), dtype=np.int64)
    n_cond_a = int(layout.num_condition_audio_rows)
    dynamic = audio_idx[n_cond_a:]
    mask = np.ones(int(layout.sequence_length), dtype=bool)
    mask[dynamic] = False
    cached = np.nonzero(mask)[0]
    return RevoicePlan(
        dynamic=mx.array(dynamic.astype(np.int32)),
        cached=mx.array(cached.astype(np.int32)),
        num_audio_cond=n_cond_a,
    )


def start_sigmas(full: list[float], strength: float) -> list[float]:
    """The audio schedule from ``strength`` down: that level first, then the grid's points below it."""
    if strength >= 1.0:
        return list(full)
    return [float(strength)] + [v for v in full if v < strength]


def set_adapter_scales(dit, paths, scale: float | None) -> dict:
    """Switch the adapters loaded from ``paths`` off (``scale=0``) or restore them (``None``).

    Each ``LoRALinear`` wrapper stacks its adapters in load order; ``lora_sources`` (recorded at
    apply time) says which file each entry came from, so a character LoRA stacked after the speed
    adapter keeps its own scale. Returns a count of the entries touched.
    """
    from .lora import LoRALinear

    wanted = {str(p) for p in paths}
    touched = 0
    for _, module in dit.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        sources = getattr(module, "lora_sources", None) or []
        saved = getattr(module, "_revoice_saved", None)
        for i, src in enumerate(sources):
            if str(src) not in wanted:
                continue
            if scale is None:
                if saved is not None and i in saved:
                    module.lora_scales[i] = saved[i]
                    touched += 1
            else:
                if saved is None:
                    saved = {}
                    module._revoice_saved = saved
                saved.setdefault(i, module.lora_scales[i])
                module.lora_scales[i] = float(scale)
                touched += 1
        if scale is None and saved is not None:
            module._revoice_saved = None
    return {"entries": touched}


def _quantize_kv(t: mx.array):
    """(1, heads, rows, 128) bf16 -> int8 affine pack, one scale/bias per head vector."""
    shape = t.shape
    # One scale/bias per 128 values = per head vector on the real model (head_dim 128). MLX
    # supports groups of 32/64/128; a smaller head (tests) groups neighbouring vectors.
    group = next(g for g in (128, 64, 32) if t.size % g == 0)
    w, scales, biases = mx.quantize(t.reshape(-1, group), group_size=group, bits=8)
    return (w, scales, biases, shape, t.dtype, group)


def _dequantize_kv(packed) -> mx.array:
    w, scales, biases, shape, dtype, group = packed
    return mx.dequantize(w, scales, biases, group_size=group, bits=8).reshape(shape).astype(dtype)


class KVStore:
    def __init__(self, mode: str):
        if mode not in ("int8", "bf16"):
            raise ValueError(f"KV store mode {mode!r}")
        self.mode = mode
        self.k: list = []
        self.v: list = []
        self.nbytes = 0

    def add(self, k: mx.array, v: mx.array) -> None:
        if self.mode == "int8":
            pk, pv = _quantize_kv(k), _quantize_kv(v)
            mx.eval(pk[0], pk[1], pk[2], pv[0], pv[1], pv[2])
            self.nbytes += sum(a.nbytes for a in (pk[0], pk[1], pk[2], pv[0], pv[1], pv[2]))
        else:
            pk, pv = k, v
            mx.eval(pk, pv)
            self.nbytes += pk.nbytes + pv.nbytes
        self.k.append(pk)
        self.v.append(pv)

    def get(self, index: int) -> tuple[mx.array, mx.array]:
        k, v = self.k[index], self.v[index]
        if self.mode == "int8":
            return _dequantize_kv(k), _dequantize_kv(v)
        return k, v


def _qkv(attn, h: mx.array, cos: mx.array, sin: mx.array):
    B, S, _ = h.shape
    qkv = attn.qkv_proj(h).reshape(B, S, attn.heads, 3, attn.head_dim)
    q, k, v = qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]
    q = attn.q_norm(q).transpose(0, 2, 1, 3)
    k = attn.k_norm(k).transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin), v


def _attend(attn, q, k, v, dtype) -> mx.array:
    B, _, S, _ = q.shape
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, attn.heads * attn.head_dim)
    return attn.out_proj(out.astype(dtype))


def store_forward(dit, forward_args, modulation_cache, plan: RevoicePlan, kv_mode: str):
    """One exact full forward that also stores every block's K/V for the cached rows.

    Returns ``(audio_prediction_rows, store)``; the prediction is the same number ``dit(...)``
    would return for this input (it is the first re-voice step).
    """
    x, temb, adaln_indices, rotary = dit.pack_inputs(*forward_args)
    cos, sin = rotary
    store = KVStore(kv_mode)
    skip = getattr(dit, "skip_blocks", None) or ()
    for i, block in enumerate(dit.blocks):
        if i in skip:
            store.k.append(None)
            store.v.append(None)
            continue
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation_cache.get(i)
        h = block.norm1(x)
        h = h * (1.0 + scale_msa[adaln_indices]) + shift_msa[adaln_indices]
        q, k, v = _qkv(block.attn, h, cos, sin)
        store.add(k[:, :, plan.cached], v[:, :, plan.cached])
        x = x + gate_msa[adaln_indices] * _attend(block.attn, q, k, v, h.dtype)
        h = block.norm2(x)
        h = h * (1.0 + scale_mlp[adaln_indices]) + shift_mlp[adaln_indices]
        x = x + gate_mlp[adaln_indices] * block.mlp(h)
        mx.eval(x)
    timestep_indices = forward_args[4]
    x = dit.final_layer.norm_out(x, temb, timestep_indices)
    audio_out = dit.final_layer.audio_out(x.astype(param_dtype(dit.final_layer.audio_out)))
    audio_indices = forward_args[8]
    return audio_out[:, audio_indices], store


def audio_forward(dit, audio_rows, timestep_table, timestep_indices, layout, plan: RevoicePlan,
                  store: KVStore, modulation_cache, rotary) -> mx.array:
    """Audio rows only, attending to the stored keys/values plus their own.

    ``audio_rows`` is every audio row (conditioning first) so the return matches the shape a full
    forward hands back for ``audio_indices``; the conditioning audio rows' prediction is never used
    (they are held), so it is returned as zeros.
    """
    from .config import MODALITY_NUM, TAG_AUDIO

    n_cond = plan.num_audio_cond
    target = audio_rows[None, n_cond:]
    xa = dit.audio_patch_proj(target.astype(param_dtype(dit.audio_patch_proj)))
    temb = dit.embed_timesteps(timestep_table)
    ti = timestep_indices[plan.dynamic]
    ai = ti * MODALITY_NUM + TAG_AUDIO
    cos, sin = rotary
    cos_a, sin_a = cos[plan.dynamic], sin[plan.dynamic]
    # The packed buffer's dtype is the text stream's (the condition projection's); match it.
    xa = xa.astype(param_dtype(dit.condition_proj))
    skip = getattr(dit, "skip_blocks", None) or ()
    for i, block in enumerate(dit.blocks):
        if i in skip:
            continue
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation_cache.get(i)
        h = block.norm1(xa)
        h = h * (1.0 + scale_msa[ai]) + shift_msa[ai]
        q, k, v = _qkv(block.attn, h, cos_a, sin_a)
        kc, vc = store.get(i)
        keys = mx.concatenate([kc, k.astype(kc.dtype)], axis=2)
        values = mx.concatenate([vc, v.astype(vc.dtype)], axis=2)
        xa = xa + gate_msa[ai] * _attend(block.attn, q, keys, values, h.dtype)
        h = block.norm2(xa)
        h = h * (1.0 + scale_mlp[ai]) + shift_mlp[ai]
        xa = xa + gate_mlp[ai] * block.mlp(h)
        if i % 10 == 9:
            mx.eval(xa)
    xa = dit.final_layer.norm_out(xa, temb, ti)
    out = dit.final_layer.audio_out(xa.astype(param_dtype(dit.final_layer.audio_out)))[0]
    if n_cond:
        out = mx.concatenate([mx.zeros((n_cond, out.shape[-1]), dtype=out.dtype), out])
    return out


def kv_bytes_estimate(cached_rows: int, num_layers: int, inner_dim: int, mode: str) -> int:
    """Bytes the store will hold (what a memory gate prices before the pass starts)."""
    if mode == "off":
        return 0
    per = 2 * cached_rows * inner_dim * num_layers
    if mode == "bf16":
        return per * 2
    # int8 + one fp16-ish scale and bias per 128-wide vector (MLX stores them in the input dtype).
    return per + per // 128 * 4
