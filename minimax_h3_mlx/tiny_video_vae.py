"""Decode MiniMax-H3 latents with madebyollin's tiny temporal autoencoder.

The full H3 video VAE uses a 36-layer, 2048-wide ViT decoder.  The tiny decoder is a causal
convolutional preview model with 22 MB of weights.  It consumes the transformer's normalized
24-channel video latents directly and produces display-ready RGB in ``[0, 1]``.

This is a decode-only MLX port of ``madebyollin/taehv``'s H3 decoder.  The source checkpoint is
linked from the Kijai/MiniMax-H3-TAE model card as the preferred H3 TAE.  Only the decoder tensors
are loaded; the checkpoint's encoder is deliberately ignored.
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten


class FrameConv2d(nn.Conv2d):
    """Apply an MLX 2D convolution independently to every video frame."""

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim != 5:
            raise ValueError(f"frame convolution expects (B, T, H, W, C), got {x.shape}")
        batch, frames, height, width, channels = x.shape
        out = super().__call__(x.reshape(batch * frames, height, width, channels))
        return out.reshape(batch, frames, out.shape[1], out.shape[2], out.shape[3])


class Clamp(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return mx.tanh(x / 3.0) * 3.0


class SpatialUpsample(nn.Module):
    """Nearest-neighbour 2x upsample, matching ``torch.nn.Upsample``'s default mode."""

    def __call__(self, x: mx.array) -> mx.array:
        return mx.repeat(mx.repeat(x, 2, axis=2), 2, axis=3)


class MemBlock(nn.Module):
    """Causal residual block whose memory is the previous frame's block input."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = [
            FrameConv2d(in_channels * 2, out_channels, 3, padding=1),
            nn.ReLU(),
            FrameConv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            FrameConv2d(out_channels, out_channels, 3, padding=1),
        ]
        self.skip = (
            FrameConv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def __call__(self, x: mx.array) -> mx.array:
        past = mx.concatenate([mx.zeros_like(x[:, :1]), x[:, :-1]], axis=1)
        h = mx.concatenate([x, past], axis=-1)
        for layer in self.conv:
            h = layer(h)
        return nn.relu(h + self.skip(x))


class TemporalGrow(nn.Module):
    """Learned 1x1 projection followed by channel-to-time rearrangement."""

    def __init__(self, channels: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = FrameConv2d(channels, channels * stride, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        if self.stride == 1:
            return self.conv(x)
        batch, frames, height, width, channels = x.shape
        x = self.conv(x).reshape(batch, frames, height, width, self.stride, channels)
        return x.transpose(0, 1, 4, 2, 3, 5).reshape(
            batch, frames * self.stride, height, width, channels
        )


def pixel_shuffle(x: mx.array, factor: int) -> mx.array:
    """Channels-last equivalent of ``torch.nn.functional.pixel_shuffle``."""

    batch, frames, height, width, channels = x.shape
    output_channels = channels // (factor * factor)
    if output_channels * factor * factor != channels:
        raise ValueError(f"{channels} channels cannot be pixel-shuffled by {factor}")
    x = x.reshape(batch, frames, height, width, output_channels, factor, factor)
    return x.transpose(0, 1, 2, 5, 3, 6, 4).reshape(
        batch, frames, height * factor, width * factor, output_channels
    )


class TinyH3VideoDecoder(nn.Module):
    """H3-specific TAE decoder: normalized latent video to display-ready RGB."""

    latent_channels = 24
    patch_size = 2
    temporal_upscale = 4
    latent_chunk = 5
    frames_per_chunk = 17

    def __init__(self):
        super().__init__()
        # Positional list indices intentionally reproduce the published checkpoint's
        # ``decoder.<index>`` names so loading stays a strict 1:1 mapping.
        self.decoder = [
            Clamp(),
            FrameConv2d(24, 256, 3, padding=1),
            nn.ReLU(),
            MemBlock(256, 256),
            MemBlock(256, 256),
            MemBlock(256, 256),
            SpatialUpsample(),
            TemporalGrow(256, 1),
            FrameConv2d(256, 128, 3, padding=1, bias=False),
            MemBlock(128, 128),
            MemBlock(128, 128),
            MemBlock(128, 128),
            SpatialUpsample(),
            TemporalGrow(128, 2),
            FrameConv2d(128, 64, 3, padding=1, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            SpatialUpsample(),
            TemporalGrow(64, 2),
            FrameConv2d(64, 64, 3, padding=1, bias=False),
            nn.ReLU(),
            FrameConv2d(64, 12, 3, padding=1),
        ]

    def decode(self, latents: mx.array, num_frames: int) -> mx.array:
        """Decode ``(B, 24, T, H, W)`` to ``(B, 3, num_frames, H*16, W*16)``.

        H3 discards three encoder tokens after processing its 17-frame chunks.  Repeating the
        final clean latent restores those tail positions before the tiny decoder runs.  Each
        five-token chunk then yields 20 raw frames; its three causal lead-in frames are removed,
        leaving H3's native 17-frame cadence.  Cutting to ``num_frames`` removes only the source
        clip's final grid padding.
        """
        if latents.ndim != 5 or latents.shape[1] != self.latent_channels:
            raise ValueError(
                f"tiny H3 decoder expects (B, 24, T, H, W), got {latents.shape}"
            )
        needed_tokens = math.ceil((latents.shape[2] + 3) / self.latent_chunk) * self.latent_chunk
        if needed_tokens > latents.shape[2]:
            tail = mx.broadcast_to(
                latents[:, :, -1:],
                (*latents.shape[:2], needed_tokens - latents.shape[2], *latents.shape[3:]),
            )
            latents = mx.concatenate([latents, tail], axis=2)

        x = latents.transpose(0, 2, 3, 4, 1)
        for layer in self.decoder:
            x = layer(x)

        raw_chunk = self.latent_chunk * self.temporal_upscale
        chunks = x.shape[1] // raw_chunk
        x = x[:, : chunks * raw_chunk].reshape(
            x.shape[0], chunks, raw_chunk, x.shape[2], x.shape[3], x.shape[4]
        )
        x = x[:, :, raw_chunk - self.frames_per_chunk :].reshape(
            x.shape[0], chunks * self.frames_per_chunk, x.shape[3], x.shape[4], x.shape[5]
        )
        if x.shape[1] < num_frames:
            raise ValueError(
                f"{latents.shape[2]} padded latent frames decode to only {x.shape[1]} RGB frames; "
                f"cannot deliver {num_frames}"
            )
        x = pixel_shuffle(x[:, :num_frames], self.patch_size)
        return mx.clip(x, 0.0, 1.0).transpose(0, 4, 1, 2, 3)


def load_tiny_h3_video_decoder(path: str | Path) -> TinyH3VideoDecoder:
    """Load the decoder half of a madebyollin ``taeh3.safetensors`` checkpoint."""

    model = TinyH3VideoDecoder()
    expected = {key for key, _ in tree_flatten(model.parameters())}
    weights = {}
    unexpected = []
    for key, tensor in mx.load(str(path)).items():
        if not key.startswith("decoder."):
            continue
        if key not in expected:
            unexpected.append(key)
            continue
        if tensor.ndim == 4:
            # torch OIHW -> MLX OHWI.  Materialize the small conversion on CPU so loading a
            # preview decoder never submits incidental work to the shared GPU.
            with mx.stream(mx.cpu):
                tensor = mx.contiguous(tensor.transpose(0, 2, 3, 1))
                mx.eval(tensor)
        weights[key] = tensor

    missing = sorted(expected - weights.keys())
    if missing or unexpected:
        raise KeyError(
            f"TAE checkpoint/module mismatch: {len(missing)} missing ({missing[:4]}), "
            f"{len(unexpected)} unexpected ({unexpected[:4]})"
        )
    model.update(tree_unflatten(list(weights.items())))
    return model
