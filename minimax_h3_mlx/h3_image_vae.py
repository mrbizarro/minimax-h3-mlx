"""MiniMax-H3-Image's image decoder, in MLX.

The release ships ONE image VAE (``vae/diffusion_pytorch_model.safetensors``, F32): a CNN encoder
(edit mode) and a **24-layer 2D ViT decoder** in the original-checkpoint naming, which is already
our fused layout (``to_qkv``, ``ff.w1`` gate-first, ``scale1/2``), so the decoder loads 1:1.

It is the same family as the Pruna decoder (:mod:`ph3_image_vae`), with three differences that
matter and are written down here because each one gives noise when wrong:

* **2D rotary**: ids are ``(h, w)`` only (``n_dim = 2``), no constant time axis; the rotated part is
  half of the 64-wide head (``rope_dim_ratio 0.5``), theta 100;
* 24 blocks of 16 heads x 64 = 1024 wide, RMSNorm eps 1e-5 (the final norm is an affine LayerNorm);
* the checkpoint also carries a ``mask_token`` (a training buffer): the extra suffix token is a
  zero row, as in the reference, so it is ignored.

``decode_to_uint8`` is reused from the Pruna module: the pixel convention (ImageNet mean/std over
[0, 1]) and ``latents_mean/std`` normalisation are the same.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from .ph3_image_vae import PH3ImageVAE, _run_block, decode_to_uint8  # noqa: F401  (re-exported)
from .video_vae import VideoVAEConfig, ViTDecoder3d, ViTRotaryPosEmbed

SPATIAL_RATIO = 16


def h3_image_vae_config(raw: dict) -> VideoVAEConfig:
    """Build our config from the published ``vae/config.json`` (original keys)."""
    kw = raw["vit_decoder_kwargs"]
    heads, dim_head = kw["heads"], kw["dim_head"]
    cfg = VideoVAEConfig(
        latent_channels=raw.get("latent_channels", 24),
        spatial_downsample_factors=(2, 2, 2, 2, 1, 1),
        temporal_downsample_factors=(1,) * 6,
        decoder_num_layers=kw["num_layers"],
        decoder_num_attention_heads=heads,
        decoder_attention_head_dim=dim_head,
        decoder_num_register_tokens=kw.get("num_register_tokens", 4),
        decoder_ffn_mult=4,
        decoder_rope_theta=kw["rope_theta"],
        decoder_rope_dim_ratio=kw["rope_dim_ratio"],
        decoder_norm_eps=kw.get("eps", 1e-5),
        latents_mean=tuple(raw["latents_mean"]),
        latents_std=tuple(raw["latents_std"]),
    )
    assert cfg.spatial_compression_ratio == SPATIAL_RATIO
    return replace(cfg)


class H3ImageDecoder(ViTDecoder3d):
    """:class:`ViTDecoder3d` for one latent frame with a 2-axis (h, w) rotary embedding."""

    def __init__(self, config: VideoVAEConfig):
        # The parent builds a 3-axis rotary table, which needs the rotated width divisible by 6;
        # build it on a throw-away ratio and replace it with the real 2-axis one below.
        super().__init__(replace(config, decoder_rope_dim_ratio=0.75))
        self.rope = ViTRotaryPosEmbed(
            int(config.decoder_attention_head_dim * config.decoder_rope_dim_ratio),
            theta=config.decoder_rope_theta,
            num_axes=2,
        )

    def __call__(self, x: mx.array, compute=None) -> mx.array:
        b, d, h, w, c = x.shape
        if d != 1:
            raise ValueError(f"H3ImageDecoder decodes single latent frames, got {d}.")
        tokens = self.x_embedder(x.reshape(b, h * w, c))
        num_patches = tokens.shape[1]
        registers = mx.broadcast_to(self.register_tokens, (b, self.num_register_tokens, tokens.shape[-1]))
        cls_token = mx.zeros((b, 1, tokens.shape[-1]), dtype=tokens.dtype)
        tokens = mx.concatenate([tokens, registers.astype(tokens.dtype), cls_token], axis=1)

        grids = [2.0 * ((mx.arange(size, dtype=mx.float32) + 0.5) / size) - 1.0 for size in (h, w)]
        hh = mx.broadcast_to(grids[0].reshape(h, 1), (h, w))
        ww = mx.broadcast_to(grids[1].reshape(1, w), (h, w))
        position_ids = mx.stack([hh, ww], axis=-1).reshape(1, h * w, 2)
        position_ids = mx.broadcast_to(position_ids, (b, h * w, 2))
        suffix = mx.zeros((b, self.num_register_tokens + 1, 2), dtype=mx.float32)
        rotary = self.rope(mx.concatenate([position_ids, suffix], axis=1))

        for block in self.transformer_blocks:
            tokens = _run_block(block, tokens, rotary, compute)

        normed = self.norm_out(tokens.astype(mx.float32)).astype(tokens.dtype)
        tokens = self.proj_out(normed)[:, :num_patches, :]
        p = self.patch_size
        out = tokens.reshape(b, h, w, self.out_channels, p, p)
        out = out.transpose(0, 1, 4, 2, 5, 3)  # (b, h, p, w, p, c)
        return out.reshape(b, 1, h * p, w * p, self.out_channels)


class H3ImageVAE(PH3ImageVAE):
    """Decode-only image VAE: ``post_quant_conv`` + :class:`H3ImageDecoder`, tiled in pixel space
    (256 px tiles, 64 px overlap: the release's ``vae_tile_size`` / ``vae_tile_overlap_min``)."""

    def __init__(self, config: VideoVAEConfig):
        nn.Module.__init__(self)
        self.config = config
        self.decoder = H3ImageDecoder(config)
        self.post_quant_conv = nn.Linear(config.latent_channels, config.latent_channels)
        self.use_tiling = True
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_overlap_height = 64
        self.tile_sample_min_overlap_width = 64
        self.decode_batch = 8
        self.decode_dtype = "bfloat16"


def convert_decoder_state(tensors: dict, config: VideoVAEConfig) -> dict:
    """Original ``decoder.*`` / ``post_quant_conv.*`` tensors -> :class:`H3ImageVAE` names.

    Encoder tensors and the ``mask_token`` buffer are ignored. ``post_quant_conv`` is a 1x1 conv
    on one frame, so it is held as a Linear.
    """
    out = {}
    for k, v in tensors.items():
        if k.startswith("decoder.") and k != "decoder.mask_token":
            out[k] = v
    pq = tensors["post_quant_conv.weight"]
    out["post_quant_conv.weight"] = pq.reshape(pq.shape[0], pq.shape[1])
    out["post_quant_conv.bias"] = tensors["post_quant_conv.bias"]
    return out


def load_h3_image_vae(path: str | Path, config_json: dict, dtype: mx.Dtype = mx.bfloat16) -> H3ImageVAE:
    """Load the published VAE file (or a converted decoder-only file) for decoding."""
    cfg = h3_image_vae_config(config_json)
    model = H3ImageVAE(cfg)
    raw = mx.load(str(path))
    if any(k.startswith("encoder.") for k in raw) or raw["post_quant_conv.weight"].ndim == 4:
        raw = convert_decoder_state(raw, cfg)
    weights = {k: v.astype(dtype) for k, v in raw.items()}
    expected = {k for k, _ in tree_flatten(model.parameters())}
    missing, extra = sorted(expected - weights.keys()), sorted(weights.keys() - expected)
    if missing or extra:
        raise KeyError(f"H3-Image VAE mismatch: missing {missing[:4]}, unexpected {extra[:4]}")
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model
