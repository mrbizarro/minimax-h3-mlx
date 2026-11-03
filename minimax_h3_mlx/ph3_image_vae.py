"""P-H3-Image's image decoder, in MLX.

P-H3-Image (Pruna) keeps MiniMax-H3's video-VAE *encoder and latent space* and swaps the decoder
for a ViT that decodes **one latent frame**: every latent pixel becomes one token, four register
tokens plus one all-zero token are appended, 36 pre-norm blocks with per-channel layer scales run
full self-attention, and each token is expanded into a ``16 x 16`` pixel block.

That is the same family as :class:`minimax_h3_mlx.video_vae.ViTDecoder3d`, so this module reuses
its blocks and its tiling and only adds what differs:

* a single frame (``patch_size_t == 1``) at the constant rotary time ``2 * (0.5 / 7) - 1``;
* the Diffusers checkpoint layout (``to_q/to_k/to_v``, ``ff.net.0.proj`` value-first SwiGLU), which
  :func:`convert_decoder_state` maps onto our fused ``to_qkv`` / gate-first ``ff.w1`` blocks;
* a decode-only wrapper: for text-to-image the encoder is never needed.

The post-quant 1x1 convolution is a matrix product on a single frame, so it is held as a Linear.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from .video_vae import VideoVAE, VideoVAEConfig, ViTDecoder3d

# Rotary time shared by every token of the decoded frame (the reference's FRAME_POSITION).
FRAME_POSITION = 2.0 * (0.5 / 7) - 1.0

# Pixel convention of the H3 VAE family: ImageNet statistics over [0, 1].
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


def image_vae_config(raw: dict) -> VideoVAEConfig:
    """Build our config from the published ``vae/config.json`` (Diffusers keys)."""
    cfg = VideoVAEConfig(
        in_channels=raw.get("in_channels", 3),
        out_channels=raw.get("out_channels", 3),
        latent_channels=raw.get("latent_channels", 24),
        block_out_channels=tuple(raw["block_out_channels"]),
        layers_per_block=raw.get("layers_per_block", 2),
        spatial_downsample_factors=tuple(raw["spatial_downsample_factors"]),
        temporal_downsample_factors=tuple(raw.get("temporal_downsample_factors", (1,) * 6)),
        norm_num_groups=raw.get("norm_num_groups", 32),
        norm_eps=raw.get("norm_eps", 1e-6),
        decoder_num_layers=raw["decoder_num_layers"],
        decoder_num_attention_heads=raw["decoder_num_attention_heads"],
        decoder_attention_head_dim=raw["decoder_attention_head_dim"],
        decoder_num_register_tokens=raw["decoder_num_register_tokens"],
        decoder_ffn_mult=raw.get("decoder_ffn_mult", 4),
        decoder_rope_theta=raw["decoder_rope_theta"],
        decoder_rope_dim_ratio=raw["decoder_rope_dim_ratio"],
        decoder_norm_eps=raw["decoder_norm_eps"],
        latents_mean=tuple(raw["latents_mean"]),
        latents_std=tuple(raw["latents_std"]),
    )
    # One frame, no temporal packing in the decoder.
    return replace(cfg, temporal_downsample_factors=(1,) * len(cfg.spatial_downsample_factors))


class PH3ImageDecoder(ViTDecoder3d):
    """:class:`ViTDecoder3d` for one latent frame at the image model's constant rotary time."""

    def __call__(self, x: mx.array, compute=None) -> mx.array:
        b, d, h, w, c = x.shape
        if d != 1:
            raise ValueError(f"PH3ImageDecoder decodes single latent frames, got {d}.")
        tokens = self.x_embedder(x.reshape(b, h * w, c))
        num_patches = tokens.shape[1]

        registers = mx.broadcast_to(self.register_tokens, (b, self.num_register_tokens, tokens.shape[-1]))
        cls_token = mx.zeros((b, 1, tokens.shape[-1]), dtype=tokens.dtype)
        tokens = mx.concatenate([tokens, registers.astype(tokens.dtype), cls_token], axis=1)

        grids = [2.0 * ((mx.arange(size, dtype=mx.float32) + 0.5) / size) - 1.0 for size in (h, w)]
        tt = mx.full((h, w), FRAME_POSITION, dtype=mx.float32)
        hh = mx.broadcast_to(grids[0].reshape(h, 1), (h, w))
        ww = mx.broadcast_to(grids[1].reshape(1, w), (h, w))
        position_ids = mx.stack([tt, hh, ww], axis=-1).reshape(1, h * w, 3)
        position_ids = mx.broadcast_to(position_ids, (b, h * w, 3))
        suffix = mx.zeros((b, self.num_register_tokens + 1, 3), dtype=mx.float32)
        rotary = self.rope(mx.concatenate([position_ids, suffix], axis=1))

        for block in self.transformer_blocks:
            tokens = block(tokens, rotary, compute)

        normed = self.norm_out(tokens.astype(mx.float32)).astype(tokens.dtype)
        tokens = self.proj_out(normed)[:, :num_patches, :]
        p = self.patch_size
        out = tokens.reshape(b, h, w, self.out_channels, p, p)
        out = out.transpose(0, 1, 4, 2, 5, 3)  # (b, h, p, w, p, c)
        return out.reshape(b, 1, h * p, w * p, self.out_channels)


class PH3ImageVAE(VideoVAE):
    """Decode-only image VAE: ``post_quant_conv`` + :class:`PH3ImageDecoder`, tiled in pixel space."""

    def __init__(self, config: VideoVAEConfig):
        nn.Module.__init__(self)
        self.config = config
        self.decoder = PH3ImageDecoder(config)
        self.post_quant_conv = nn.Linear(config.latent_channels, config.latent_channels)
        self.use_tiling = True
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_overlap_height = 64
        self.tile_sample_min_overlap_width = 64
        self.decode_batch = 8
        self.decode_dtype = "bfloat16"

    def _decode_tiles_batched(self, tiles: list[mx.array], compute) -> list[mx.array]:
        out: list[mx.array] = []
        step = max(1, int(self.decode_batch))
        for start in range(0, len(tiles), step):
            chunk = tiles[start : start + step]
            batch = mx.concatenate(chunk, axis=0)
            decoded = self.decoder(self.post_quant_conv(batch), compute)
            mx.eval(decoded)
            out.extend(decoded[i : i + 1] for i in range(len(chunk)))
        return out

    def decode_frame(self, z: mx.array, compute="auto") -> mx.array:
        """``z``: normalised-undone latents ``(1, 1, H', W', C)`` channels-last -> ``(1, 1, H, W, 3)``."""
        if compute == "auto":
            compute = getattr(mx, self.decode_dtype) if self.decode_dtype in ("bfloat16", "float16") else None
        z = z.astype(mx.float32)
        ratio = self.config.spatial_compression_ratio
        height, width = z.shape[2] * ratio, z.shape[3] * ratio
        if not self.use_tiling:
            return self._decode_tiles_batched([z], compute)[0]

        y_idx, y_len, y_ov = self._split_tiles(height, self.tile_sample_min_height, self.tile_sample_min_overlap_height)
        x_idx, x_len, x_ov = self._split_tiles(width, self.tile_sample_min_width, self.tile_sample_min_overlap_width)
        tiles, shape = [], (len(y_idx), len(x_idx))
        for i_pos, i_len in zip(y_idx, y_len):
            for j_pos, j_len in zip(x_idx, x_len):
                tiles.append(
                    z[:, :, i_pos // ratio : (i_pos + i_len) // ratio, j_pos // ratio : (j_pos + j_len) // ratio, :]
                )
        decoded = self._decode_tiles_batched(tiles, compute)
        rows = [decoded[r * shape[1] : (r + 1) * shape[1]] for r in range(shape[0])]
        return self._stitch_tiles(rows, y_ov, x_ov, h_axis=2, w_axis=3)


# ---------------------------------------------------------------------------------------------
# Checkpoint layout: Diffusers -> ours
# ---------------------------------------------------------------------------------------------


def _xp(a):
    return mx if isinstance(a, mx.array) else np


def _fuse_qkv(q, k, v, heads: int):
    """Three ``(heads * hd, ...)`` projections -> our per-head interleaved ``(heads, 3, hd, ...)``.

    Works on numpy or MLX arrays (the converter streams MLX bf16 tensors, which numpy cannot hold).
    """
    hd = q.shape[0] // heads
    xp = _xp(q)
    stacked = xp.stack([a.reshape(heads, hd, *a.shape[1:]) for a in (q, k, v)], axis=1)
    return stacked.reshape(3 * heads * hd, *q.shape[1:])


def convert_decoder_state(tensors: dict, config: VideoVAEConfig) -> dict:
    """Map Diffusers ``decoder.*`` / ``post_quant_conv.*`` tensors onto :class:`PH3ImageVAE` names.

    Only the decoder side is converted; encoder tensors are ignored (they are not needed for
    text-to-image). Raises if a block is missing a tensor.
    """
    heads = config.decoder_num_attention_heads
    out: dict = {}

    def take(key: str):
        try:
            return tensors[key]
        except KeyError as exc:
            raise KeyError(f"image VAE checkpoint has no {key!r}") from exc

    out["decoder.x_embedder.weight"] = take("decoder.proj_in.weight")
    out["decoder.x_embedder.bias"] = take("decoder.proj_in.bias")
    out["decoder.register_tokens"] = take("decoder.register_tokens")
    for name in ("norm_out", "proj_out"):
        for leaf in ("weight", "bias"):
            out[f"decoder.{name}.{leaf}"] = take(f"decoder.{name}.{leaf}")

    for i in range(config.decoder_num_layers):
        src, dst = f"decoder.transformer_blocks.{i}.", f"decoder.transformer_blocks.{i}."
        out[dst + "norm1.weight"] = take(src + "norm1.weight")
        out[dst + "norm2.weight"] = take(src + "norm2.weight")
        out[dst + "scale1"] = take(src + "scale1")
        out[dst + "scale2"] = take(src + "scale2")
        for leaf in ("weight", "bias"):
            out[dst + f"attn.to_qkv.{leaf}"] = _fuse_qkv(
                take(src + f"attn.to_q.{leaf}"), take(src + f"attn.to_k.{leaf}"), take(src + f"attn.to_v.{leaf}"), heads
            )
            out[dst + f"attn.to_out.{leaf}"] = take(src + f"attn.to_out.0.{leaf}")
            # Diffusers SwiGLU is value-first, ``value * silu(gate)``; ours is ``[gate; up]``.
            proj = take(src + f"ff.net.0.proj.{leaf}")
            half = proj.shape[0] // 2
            out[dst + f"ff.w1.{leaf}"] = _xp(proj).concatenate([proj[half:], proj[:half]], axis=0)
            out[dst + f"ff.w2.{leaf}"] = take(src + f"ff.net.2.{leaf}")

    pq = take("post_quant_conv.weight")
    out["post_quant_conv.weight"] = pq.reshape(pq.shape[0], pq.shape[1])
    out["post_quant_conv.bias"] = take("post_quant_conv.bias")
    return out


def load_image_vae(path: str | Path, config_json: dict, dtype: mx.Dtype = mx.bfloat16) -> PH3ImageVAE:
    """Load a converted decoder (``save_converted_decoder``) or a Diffusers VAE file."""
    cfg = image_vae_config(config_json)
    model = PH3ImageVAE(cfg)
    raw = mx.load(str(path))
    if "decoder.x_embedder.weight" not in raw:  # Diffusers layout
        raw = convert_decoder_state(raw, cfg)
    weights = {k: v.astype(dtype) for k, v in raw.items()}
    expected = {k for k, _ in tree_flatten(model.parameters())}
    missing, extra = sorted(expected - weights.keys()), sorted(weights.keys() - expected)
    if missing or extra:
        raise KeyError(f"image VAE mismatch: missing {missing[:4]}, unexpected {extra[:4]}")
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model


def decode_to_uint8(vae: PH3ImageVAE, latents: mx.array) -> np.ndarray:
    """Normalised latents ``(1, C, 1, H', W')`` -> ``(H, W, 3)`` uint8.

    Undoes ``latents_mean/std``, decodes, then undoes the ImageNet pixel normalisation.
    """
    cfg = vae.config
    mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1, 1, 1)
    std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1, 1, 1)
    z = (latents * std + mean).transpose(0, 2, 3, 4, 1)  # -> (1, 1, H', W', C)
    pixels = vae.decode_frame(z)[0, 0].astype(mx.float32)  # (H, W, 3)
    pm = mx.array(np.array(PIXEL_MEAN, np.float32))
    ps = mx.array(np.array(PIXEL_STD, np.float32))
    img = mx.clip((pixels * ps + pm) * 255.0 + 0.5, 0.0, 255.0)
    mx.eval(img)
    return np.array(img).astype(np.uint8)
