"""MLX port of the MiniMax-H3 video VAE (``AutoencoderKLLegacy`` / ``MiniMaxH3VideoVAE``).

A causal 3D CNN **encoder** (16x spatial, 4x temporal, 24 latent channels) paired with a
non-causal 36-layer **ViT decoder** at dim 2048. There is no CNN decoder: every latent voxel
becomes one token and a single output projection expands it straight into a
``patch_size_t x patch_size x patch_size`` pixel block, doing all the upsampling at once.

Two MLX-specific departures from the reference, both forced by the framework rather than chosen:

* **Channels-last.** MLX convolutions take ``(N, D, H, W, C)`` and weights ``(C_out, kD, kH, kW,
  C_in)``, where torch uses ``(N, C, D, H, W)`` / ``(C_out, C_in, kD, kH, kW)``. The CNN runs
  channels-last internally and transposes only at the public encode/decode boundary, so the
  pipeline still sees the reference's ``(B, C, F, H, W)``.
* **Reflect padding.** ``mx.pad`` has no reflect mode, so it is done by gather (:func:`reflect_pad`).

The module tree reproduces the *original* checkpoint names (``encoder.down.{i}.block.{j}``,
``decoder.x_embedder``, ``attn.to_qkv``, ``ff.w1``), so the released weights load without renaming.
As in the DiT, ``attn.to_qkv`` is per-head interleaved and ``ff.w1`` is a fused ``[gate; up]``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

#: Spatial tiles pushed through the ViT decoder in one call. The decoder is per-item — attention
#: never crosses the batch axis and the rotary grid is derived from a tile's own ``(d, h, w)`` — so
#: stacking same-shaped tiles on the batch axis is exactly the arithmetic of decoding them one by
#: one, with a fraction of the launches. A 768x448 clip is 8 tiles, so the default batches a whole
#: clip in one call, which is what the PyTorch reference does. The cap exists because the fused
#: SwiGLU projection is the widest activation and grows linearly with the batch.
#: ``H3_VAE_BATCH=0`` (or 1) restores the one-call-per-tile loop.
DEFAULT_DECODE_BATCH = 8

#: Arithmetic dtype of the ViT decoder. The compact checkpoint stores FP16 weights, but the runner
#: hands the decoder FLOAT32 latents, and MLX promotes ``fp16 weight x fp32 input`` to float32 — so
#: until now every decoder linear, attention and FFN ran in float32 (with a per-call weight upcast).
#: The PyTorch reference decodes under FP16 autocast and ComfyUI runs the VAE in fp16, so
#: ``float16`` is the reference-equivalent precision; the RMS/QK norms stay float32 either way and
#: tiles are cast back to float32 before stitching. ``H3_VAE_DTYPE`` overrides the default.
#:
#: Default ``float16-mixed`` since 2026-10-30. Measured on an M4 Max, a real 1024x576 124-frame clip,
#: same session: float32 89.3 s, float16 (all-fp16 stream) 76.9 s at 62.9 dB / max 3/255,
#: float16-mixed 76.1 s at 69.6 dB / max 1/255 / mean 0.007/255 against float32, bfloat16-mixed
#: 60.3 dB. So against float32 it is ~15% faster; against the all-fp16 stream it is the same speed
#: and 6.7 dB closer to float32. On an M4 half-precision matmul is only ~11% faster than float32
#: (no M5 matmul units), which is why the gain is nowhere near h3.c's 3x on an M5.
#: ``float32`` is still selectable and still bit-for-bit the historical call.
#:
#: The ``*-mixed`` modes are the split h3.c uses (method only, our own code): the residual stream,
#: every norm, the Q/K RMSNorm and the rotary embedding stay float32, while the four projections
#: (``to_qkv``, ``to_out``, ``ff.w1``, ``ff.w2``) and the attention itself run in half precision.
#: Each projection's output is cast back to float32 before it is scaled into the residual, so the
#: stream that carries the picture through 36 blocks never rounds to 16 bits. ``float16-mixed`` is
#: the natural one here because the weights already ARE fp16 (no rounding of weights at all);
#: ``bfloat16-mixed`` rounds them to bf16 first and exists for comparison.
DEFAULT_DECODE_DTYPE = "float16-mixed"
DECODE_DTYPES = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "float16-mixed": mx.float32,
    "bfloat16-mixed": mx.float32,
}
#: Half dtype the projections and attention use in each ``*-mixed`` mode.
MIXED_COMPUTE = {"float16-mixed": mx.float16, "bfloat16-mixed": mx.bfloat16}


def resolved_decode_batch() -> int:
    """The decode batch a freshly built :class:`VideoVAE` will use, honouring ``H3_VAE_BATCH``.

    Unparseable values are ignored rather than allowed to fail a render that is already
    twenty minutes deep.
    """
    raw = os.environ.get("H3_VAE_BATCH")
    if raw is None:
        return DEFAULT_DECODE_BATCH
    try:
        return max(int(raw), 0)
    except ValueError:
        return DEFAULT_DECODE_BATCH


def resolved_decode_dtype() -> str:
    """The decoder compute dtype a fresh :class:`VideoVAE` will use (``H3_VAE_DTYPE`` or default)."""
    raw = (os.environ.get("H3_VAE_DTYPE") or DEFAULT_DECODE_DTYPE).strip().lower()
    return raw if raw in DECODE_DTYPES else DEFAULT_DECODE_DTYPE


@dataclass
class VideoVAEConfig:
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 24
    block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024)
    layers_per_block: int = 2
    spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1)
    temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1)
    norm_num_groups: int = 32
    norm_eps: float = 1e-6
    decoder_num_layers: int = 36
    decoder_num_attention_heads: int = 32
    decoder_attention_head_dim: int = 64
    decoder_num_register_tokens: int = 4
    decoder_ffn_mult: int = 4
    decoder_rope_theta: float = 100.0
    decoder_rope_dim_ratio: float = 0.75
    decoder_norm_eps: float = 1e-5
    clip_length: int = 17
    token_drop: int = 3
    latents_mean: tuple[float, ...] = field(default_factory=tuple)
    latents_std: tuple[float, ...] = field(default_factory=tuple)

    @property
    def spatial_compression_ratio(self) -> int:
        ratio = 1
        for f in self.spatial_downsample_factors:
            ratio *= f
        return ratio

    @property
    def temporal_compression_ratio(self) -> int:
        ratio = 1
        for f in self.temporal_downsample_factors:
            ratio *= f
        return ratio

    @property
    def decoder_dim(self) -> int:
        return self.decoder_num_attention_heads * self.decoder_attention_head_dim

    @property
    def patch_size(self) -> int:
        return self.spatial_compression_ratio

    @property
    def patch_size_t(self) -> int:
        return self.temporal_compression_ratio


def single_frame_conv2d() -> bool:
    """Whether one-frame encodes (keyframes) take the 2D-conv shortcut. OPT-IN: ``H3_VAE_ENCODER_CONV2D=1``.

    Measured 2026-10-30, M4 Max: two 1024x576 anchors 2.28 s -> 0.90 s, latent relative L2
    3.5e-6..3.9e-6 against the 3D conv. Small, but after the keyframe's seed-42 sample and float16
    round trip 0.7% of the conditioning values move by one fp16 step, and the denoise carries that
    into a different take: a 640x384 Flex Draft A/B (same seed) came out 37.8 dB apart with an audio
    correlation of 0.65 — the same scene, but not the same clip. For about one second per anchor
    that breaks seed reproducibility, so it stays off by default.
    """
    return os.environ.get("H3_VAE_ENCODER_CONV2D", "0") == "1"


def reflect_pad(x: mx.array, axis: int, left: int, right: int) -> mx.array:
    """Reflect padding without repeating the edge element, matching ``F.pad(mode="reflect")``.

    ``mx.pad`` supports only constant and edge, so the reflection is done by gather.
    """
    if left == 0 and right == 0:
        return x
    n = x.shape[axis]
    idx = list(range(left, 0, -1)) + list(range(n)) + list(range(n - 2, n - 2 - right, -1))
    return mx.take(x, mx.array(idx), axis=axis)


class CausalConv3d(nn.Module):
    """3D convolution over ``(N, D, H, W, C)``.

    Spatial padding is symmetric reflect; temporal padding is **causal** — ``temporal_padding``
    zero frames are prepended and nothing is appended.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        spatial_padding: int = 0,
        temporal_padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.stride = stride
        self.spatial_padding = spatial_padding
        self.temporal_padding = temporal_padding
        # MLX weight layout: (C_out, kD, kH, kW, C_in).
        scale = 1.0 / math.sqrt(in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2])
        self.weight = mx.random.uniform(-scale, scale, (out_channels, *kernel_size, in_channels))
        if bias:
            self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        if self.spatial_padding > 0:
            p = self.spatial_padding
            x = reflect_pad(x, 2, p, p)
            x = reflect_pad(x, 3, p, p)
        if single_frame_conv2d() and x.shape[1] == 1 and self.temporal_padding == self.weight.shape[1] - 1:
            # One input frame behind `kD - 1` causal zero frames: every temporal tap but the last
            # multiplies zeros, and the output is one frame deep whatever the temporal stride. So
            # the 3D conv is exactly a 2D conv with the last tap's kernel — two thirds fewer
            # multiply-adds for a keyframe. Same sum, different accumulation order, so the result
            # is equal to float32 rounding, not bit-identical, which is why it is opt-in
            # (see single_frame_conv2d).
            out = mx.conv2d(x[:, 0], self.weight[:, -1], stride=self.stride[1:], padding=0)[:, None]
            if "bias" in self:
                out = out + self.bias
            return out
        if self.temporal_padding > 0:
            x = mx.pad(x, ((0, 0), (self.temporal_padding, 0), (0, 0), (0, 0), (0, 0)))
        out = mx.conv3d(x, self.weight, stride=self.stride, padding=0)
        if "bias" in self:
            out = out + self.bias
        return out


class TIsolatedGroupNorm(nn.GroupNorm):
    """Group norm applied to each latent frame in isolation (``use_t_isolated_gn``).

    The temporal axis is folded into the batch axis so statistics never mix across frames.
    Subclasses ``nn.GroupNorm`` rather than wrapping it so the parameters stay at this module's own
    level and the checkpoint's ``norm1.weight`` / ``norm1.bias`` keep matching 1:1.
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-6):
        super().__init__(num_groups, num_channels, eps=eps, affine=True, pytorch_compatible=True)

    def __call__(self, x: mx.array) -> mx.array:
        b, d, h, w, c = x.shape
        folded = x.reshape(b * d, h, w, c)
        return super().__call__(folded).reshape(b, d, h, w, c)


class ResnetBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.norm1 = TIsolatedGroupNorm(num_groups, in_channels, eps)
        self.conv1 = CausalConv3d(in_channels, out_channels, 3, spatial_padding=1, temporal_padding=2)
        self.norm2 = TIsolatedGroupNorm(num_groups, out_channels, eps)
        self.conv2 = CausalConv3d(out_channels, out_channels, 3, spatial_padding=1, temporal_padding=2)
        if in_channels != out_channels:
            self.nin_shortcut = CausalConv3d(in_channels, out_channels, 1)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        h = self.conv1(nn.silu(self.norm1(x)))
        h = self.conv2(nn.silu(self.norm2(h)))
        if "nin_shortcut" in self:
            residual = self.nin_shortcut(residual)
        return residual + h


class Downsample3d(nn.Module):
    """Strided 3x3x3 downsample. A spatial stride of 2 is preceded by an asymmetric bottom/right
    reflect pad of 1 (the convolution carries no spatial padding), so the output is ``ceil(n / 2)``.
    """

    def __init__(self, in_channels: int, out_channels: int, temporal_stride: int = 1, spatial_stride: int = 2):
        super().__init__()
        self.spatial_stride = spatial_stride
        self.conv = CausalConv3d(
            in_channels,
            out_channels,
            3,
            stride=(temporal_stride, spatial_stride, spatial_stride),
            spatial_padding=0,
            temporal_padding=2,
        )

    def __call__(self, x: mx.array) -> mx.array:
        if self.spatial_stride == 2:
            x = reflect_pad(x, 2, 0, 1)
            x = reflect_pad(x, 3, 0, 1)
        return self.conv(x)


class DownBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        temporal_factor: int,
        spatial_factor: int,
        num_groups: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.block = [
            ResnetBlock3d(in_channels if i == 0 else out_channels, out_channels, num_groups, eps)
            for i in range(num_layers)
        ]
        if temporal_factor * spatial_factor > 1:
            self.downsample = Downsample3d(out_channels, out_channels, temporal_factor, spatial_factor)

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.block:
            x = resnet(x)
        if "downsample" in self:
            x = self.downsample(x)
        return x


class Encoder3d(nn.Module):
    """Causal 3D CNN encoder."""

    def __init__(self, config: VideoVAEConfig):
        super().__init__()
        ch = config.block_out_channels
        self.conv_in = CausalConv3d(config.in_channels, ch[0], 3, spatial_padding=1, temporal_padding=2)
        block_in = (ch[0],) + tuple(ch[:-1])
        self.down = [
            DownBlock3d(
                block_in[i],
                ch[i],
                config.layers_per_block,
                config.temporal_downsample_factors[i],
                config.spatial_downsample_factors[i],
                config.norm_num_groups,
                config.norm_eps,
            )
            for i in range(len(ch))
        ]
        self.norm_out = TIsolatedGroupNorm(config.norm_num_groups, ch[-1], config.norm_eps)
        self.conv_out = CausalConv3d(
            ch[-1], 2 * config.latent_channels, 3, spatial_padding=1, temporal_padding=2
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        for block in self.down:
            x = block(x)
        return self.conv_out(nn.silu(self.norm_out(x)))


class ViTRotaryPosEmbed:
    """3-axis rotary embedding for the ViT decoder.

    Coordinates are length-normalized to ``[-1, 1)`` per axis and scaled by ``2*pi``; the ``(t, h, w)``
    angle blocks are concatenated then duplicated, so the first ``rope_dim_ratio * head_dim``
    channels of every head are rotated.
    """

    def __init__(self, dim: int, theta: float = 100.0, num_axes: int = 3):
        if dim % (2 * num_axes) != 0:
            raise ValueError(f"`dim` {dim} must be divisible by `2 * num_axes` {2 * num_axes}.")
        step = 2 * num_axes / dim
        exponents = mx.array([i * step for i in range(int(math.ceil(1.0 / step)))], dtype=mx.float32)
        self.inv_freq = 1.0 / (theta**exponents)

    def __call__(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        # position_ids: (B, N, 3) -> angles (B, N, 1, 2 * 3 * len(inv_freq))
        angles = 2.0 * math.pi * position_ids[..., None] * self.inv_freq.reshape(1, 1, 1, -1)
        b, n = angles.shape[0], angles.shape[1]
        angles = angles.reshape(b, n, -1)
        angles = mx.concatenate([angles, angles], axis=-1)[:, :, None, :]
        return mx.cos(angles), mx.sin(angles)


def _apply_vit_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """``x`` is ``(B, N, heads, head_dim)``; ``cos``/``sin`` are ``(B, N, 1, rotary_dim)``."""
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    half = rotary_dim // 2
    first, second = x_rot[..., :half], x_rot[..., half:]
    rotated = mx.concatenate([-second, first], axis=-1)
    out = x_rot * cos.astype(x.dtype) + rotated * sin.astype(x.dtype)
    if x_pass.shape[-1] == 0:
        return out
    return mx.concatenate([out, x_pass], axis=-1)


def _half_linear(layer: nn.Linear, x: mx.array, compute) -> mx.array:
    """``layer(x)`` with weight and bias in ``compute`` (a no-op cast for the shipped fp16 weights)."""
    weight = layer.weight if layer.weight.dtype == compute else layer.weight.astype(compute)
    if "bias" in layer:
        bias = layer.bias if layer.bias.dtype == compute else layer.bias.astype(compute)
        return mx.addmm(bias, x, weight.T)
    return x @ weight.T


class ViTAttention(nn.Module):
    """Full self-attention with unaffine Q/K RMSNorm, computed in float32 as the reference does."""

    def __init__(self, dim: int, heads: int, dim_head: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.eps = eps
        self.scale = dim_head**-0.5
        inner = heads * dim_head
        self.to_qkv = nn.Linear(dim, 3 * inner, bias=bias)
        self.to_out = nn.Linear(inner, dim, bias=bias)

    def _rms_norm_f32(self, x: mx.array) -> mx.array:
        x32 = x.astype(mx.float32)
        normed = x32 * mx.rsqrt(mx.mean(x32 * x32, axis=-1, keepdims=True) + self.eps)
        return normed.astype(x.dtype)

    def mixed(self, h: mx.array, rotary, compute) -> mx.array:
        """Half-precision projections + attention; Q/K norm and rotary in float32.

        ``h`` is the normed block input already cast to ``compute``. Returns ``compute``.
        """
        b, n, _ = h.shape
        qkv = _half_linear(self.to_qkv, h, compute).reshape(b, n, self.heads, 3, self.dim_head)
        q, k, v = qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]
        q = self._rms_norm_f32(q.astype(mx.float32))
        k = self._rms_norm_f32(k.astype(mx.float32))
        if rotary is not None:
            q = _apply_vit_rotary(q, *rotary)
            k = _apply_vit_rotary(k, *rotary)
        q = q.astype(compute).transpose(0, 2, 1, 3)
        k = k.astype(compute).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(b, n, self.heads * self.dim_head)
        return _half_linear(self.to_out, out, compute)

    def __call__(self, x: mx.array, rotary: tuple[mx.array, mx.array] | None = None) -> mx.array:
        b, n, _ = x.shape
        # Per-head interleaved, as in the DiT: (..., heads, 3, head_dim).
        qkv = self.to_qkv(x).reshape(b, n, self.heads, 3, self.dim_head)
        q, k, v = qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]

        q = self._rms_norm_f32(q)
        k = self._rms_norm_f32(k)
        if rotary is not None:
            q = _apply_vit_rotary(q, *rotary)
            k = _apply_vit_rotary(k, *rotary)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(b, n, self.heads * self.dim_head)
        return self.to_out(out)


class ViTFeedForward(nn.Module):
    """Gated SwiGLU FFN. ``w1`` is the fused ``[gate; up]`` projection."""

    def __init__(self, dim: int, mult: int = 4, bias: bool = True):
        super().__init__()
        inner = dim * mult
        self.w1 = nn.Linear(dim, 2 * inner, bias=bias)
        self.w2 = nn.Linear(inner, dim, bias=bias)
        self._inner = inner

    def __call__(self, x: mx.array) -> mx.array:
        fused = self.w1(x)
        gate, up = fused[..., : self._inner], fused[..., self._inner :]
        return self.w2(nn.silu(gate) * up)

    def mixed(self, h: mx.array, compute) -> mx.array:
        fused = _half_linear(self.w1, h, compute)
        gate, up = fused[..., : self._inner], fused[..., self._inner :]
        return _half_linear(self.w2, nn.silu(gate) * up, compute)


class ViTBlock(nn.Module):
    """Pre-norm block with learned per-channel output scales (``scale1`` / ``scale2``, init 0).

    Both norms run in float32 regardless of the compute dtype, as in the reference.
    """

    def __init__(self, dim: int, heads: int, dim_head: int, ffn_mult: int = 4, eps: float = 1e-5):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps)
        self.attn = ViTAttention(dim, heads, dim_head, eps=eps, bias=True)
        self.scale1 = mx.zeros((dim,))
        self.norm2 = nn.RMSNorm(dim, eps=eps)
        self.ff = ViTFeedForward(dim, ffn_mult, bias=True)
        self.scale2 = mx.zeros((dim,))

    def __call__(
        self, x: mx.array, rotary: tuple[mx.array, mx.array] | None = None, compute=None
    ) -> mx.array:
        if compute is not None:
            # Mixed precision: ``x`` (the residual stream) stays float32 end to end; only what goes
            # into and comes out of the projections is half precision.
            h = self.norm1(x).astype(compute)
            x = x + self.attn.mixed(h, rotary, compute).astype(mx.float32) * self.scale1
            h = self.norm2(x).astype(compute)
            return x + self.ff.mixed(h, compute).astype(mx.float32) * self.scale2
        h = self.norm1(x.astype(mx.float32)).astype(x.dtype)
        x = x + self.attn(h, rotary) * self.scale1
        h = self.norm2(x.astype(mx.float32)).astype(x.dtype)
        return x + self.ff(h) * self.scale2


def compile_mixed_blocks() -> bool:
    """Whether the mixed-precision blocks run through ``mx.compile`` (``H3_VAE_COMPILE``, default on).

    Compiling fuses the block's elementwise tail (casts, SiLU x up, scale-and-add) into fewer
    kernels. Measured bit-identical to the uncompiled mixed block on a real 1024x576 clip.
    """
    return os.environ.get("H3_VAE_COMPILE", "1") != "0"


def _compiled_mixed(block: "ViTBlock", compute):
    """The block's mixed-precision call, compiled once per block and dtype.

    Kept off the module's parameter tree (a plain attribute holding a function, not an array or a
    container), so it never shows up in ``parameters()`` or a checkpoint.
    """
    key = f"_compiled_{compute}"
    fn = block.__dict__.get(key)
    if fn is None:
        # `inputs=block.state` makes the block's parameters live inputs of the compiled graph.
        # Without it they are traced as constants, and a block whose weights change after the
        # first call (re-loaded, quantized, edited by a test) would silently keep the old ones.
        fn = mx.compile(lambda x, cos, sin: block(x, (cos, sin), compute), inputs=block.state)
        object.__setattr__(block, key, fn)
    return fn


class ViTDecoder3d(nn.Module):
    """Non-causal ViT decoder.

    Every latent voxel becomes one token; ``num_register_tokens`` learned register tokens plus a
    single all-zero token are appended (all at position 0), attended over with full self-attention,
    and dropped again before the output projection expands each token into a
    ``patch_size_t x patch_size x patch_size`` pixel block.
    """

    def __init__(self, config: VideoVAEConfig):
        super().__init__()
        dim = config.decoder_dim
        self.patch_size = config.patch_size
        self.patch_size_t = config.patch_size_t
        self.out_channels = config.out_channels
        self.num_register_tokens = config.decoder_num_register_tokens

        self.rope = ViTRotaryPosEmbed(
            int(config.decoder_attention_head_dim * config.decoder_rope_dim_ratio),
            theta=config.decoder_rope_theta,
        )
        self.x_embedder = nn.Linear(config.latent_channels, dim)
        self.register_tokens = mx.zeros((1, config.decoder_num_register_tokens, dim))
        self.transformer_blocks = [
            ViTBlock(
                dim,
                config.decoder_num_attention_heads,
                config.decoder_attention_head_dim,
                config.decoder_ffn_mult,
                config.decoder_norm_eps,
            )
            for _ in range(config.decoder_num_layers)
        ]
        self.norm_out = nn.LayerNorm(dim, eps=config.decoder_norm_eps, affine=True)
        self.proj_out = nn.Linear(
            dim, config.out_channels * config.patch_size_t * config.patch_size * config.patch_size
        )

    def __call__(self, x: mx.array, compute=None) -> mx.array:
        """``x`` is ``(B, D, H, W, C)``; returns ``(B, D*pt, H*p, W*p, out_channels)``.

        ``compute`` (a half dtype) selects the mixed-precision blocks; ``x`` must then be float32.
        """
        b, d, h, w, c = x.shape
        tokens = self.x_embedder(x.reshape(b, d * h * w, c))
        num_patches = tokens.shape[1]

        registers = mx.broadcast_to(self.register_tokens, (b, self.num_register_tokens, tokens.shape[-1]))
        cls_token = mx.zeros((b, 1, tokens.shape[-1]), dtype=tokens.dtype)
        tokens = mx.concatenate([tokens, registers.astype(tokens.dtype), cls_token], axis=1)

        grids = [2.0 * ((mx.arange(size, dtype=mx.float32) + 0.5) / size) - 1.0 for size in (d, h, w)]
        tt = mx.broadcast_to(grids[0].reshape(d, 1, 1), (d, h, w))
        hh = mx.broadcast_to(grids[1].reshape(1, h, 1), (d, h, w))
        ww = mx.broadcast_to(grids[2].reshape(1, 1, w), (d, h, w))
        position_ids = mx.stack([tt, hh, ww], axis=-1).reshape(1, d * h * w, 3)
        position_ids = mx.broadcast_to(position_ids, (b, d * h * w, 3))
        suffix = mx.zeros((b, self.num_register_tokens + 1, 3), dtype=mx.float32)
        position_ids = mx.concatenate([position_ids, suffix], axis=1)
        rotary = self.rope(position_ids)

        for block in self.transformer_blocks:
            if compute is not None and compile_mixed_blocks():
                tokens = _compiled_mixed(block, compute)(tokens, *rotary)
            else:
                tokens = block(tokens, rotary, compute)

        # Final LayerNorm in float32, as the reference's autocast does; a no-op at float32.
        normed = self.norm_out(tokens.astype(mx.float32)).astype(tokens.dtype)
        tokens = self.proj_out(normed)[:, :num_patches, :]

        p, pt = self.patch_size, self.patch_size_t
        out = tokens.reshape(b, d, h, w, self.out_channels, pt, p, p)
        out = out.transpose(0, 1, 5, 2, 6, 3, 7, 4)
        return out.reshape(b, d * pt, h * p, w * p, self.out_channels)


class VideoVAE(nn.Module):
    """The MiniMax-H3 video autoencoder: causal CNN encoder + ViT decoder, tiled and chunked.

    The public :meth:`encode` / :meth:`decode` take and return the reference's ``(B, C, F, H, W)``;
    everything internal is channels-last.
    """

    def __init__(self, config: VideoVAEConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder3d(config)
        self.decoder = ViTDecoder3d(config)
        self.quant_conv = CausalConv3d(2 * config.latent_channels, 2 * config.latent_channels, 1)
        self.post_quant_conv = CausalConv3d(config.latent_channels, config.latent_channels, 1)

        ratio_t = config.temporal_compression_ratio
        self.frame_pre_padding = (-config.clip_length) % ratio_t
        self.tokens_chunk_size = math.ceil(config.clip_length / ratio_t)
        self.token_overlap = (-config.token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(self.token_overlap * ratio_t - self.frame_pre_padding, 0)

        self.use_tiling = True
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_overlap_height = 64
        self.tile_sample_min_overlap_width = 64
        self.decode_batch = resolved_decode_batch()
        self.decode_dtype = resolved_decode_dtype()

    # -- tiling -----------------------------------------------------------------------------

    def _split_tiles(self, length: int, tile_size: int, min_overlap: int):
        """Lay ``tile_size``-wide tiles over ``length`` pixels.

        The tile count is the smallest whose union covers ``length`` while keeping every overlap at
        least ``min_overlap``; the slack is distributed round-robin over the overlaps in whole
        ``spatial_compression_ratio`` steps so every boundary stays latent-aligned.
        """
        if tile_size >= length:
            return [0], [length], []
        ratio = self.config.spatial_compression_ratio
        num_tiles = math.ceil(length / tile_size)
        while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
            num_tiles += 1
        overlaps = [min_overlap] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        for i in range(remaining // ratio):
            overlaps[i % (num_tiles - 1)] += ratio
        starts = [0]
        for i in range(num_tiles - 1):
            starts.append(starts[-1] + tile_size - overlaps[i])
        return starts, [tile_size] * num_tiles, overlaps

    @staticmethod
    def _blend(a: mx.array, b: mx.array, blend_extent: int, axis: int) -> mx.array:
        blend_extent = min(a.shape[axis], b.shape[axis], blend_extent)
        if blend_extent <= 0:
            return b
        pos = mx.arange(blend_extent, dtype=mx.float32)
        shape = [1] * a.ndim
        shape[axis] = blend_extent
        wa = (1.0 - pos / blend_extent).reshape(shape).astype(b.dtype)
        wb = (pos / blend_extent).reshape(shape).astype(b.dtype)

        a_tail = mx.take(a, mx.arange(a.shape[axis] - blend_extent, a.shape[axis]), axis=axis)
        b_head = mx.take(b, mx.arange(blend_extent), axis=axis)
        blended = a_tail * wa + b_head * wb
        if blend_extent == b.shape[axis]:
            return blended
        rest = mx.take(b, mx.arange(blend_extent, b.shape[axis]), axis=axis)
        return mx.concatenate([blended, rest], axis=axis)

    def _stitch_tiles(self, tiles, height_overlaps, width_overlaps, h_axis: int, w_axis: int):
        rows = []
        for i, row in enumerate(tiles):
            out_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend(tiles[i - 1][j], tile, height_overlaps[i - 1], axis=h_axis)
                if j > 0:
                    tile = self._blend(row[j - 1], tile, width_overlaps[j - 1], axis=w_axis)
                if i < len(tiles) - 1:
                    tile = mx.take(tile, mx.arange(tile.shape[h_axis] - height_overlaps[i]), axis=h_axis)
                if j < len(row) - 1:
                    tile = mx.take(tile, mx.arange(tile.shape[w_axis] - width_overlaps[j]), axis=w_axis)
                out_row.append(tile)
            rows.append(mx.concatenate(out_row, axis=w_axis))
        return mx.concatenate(rows, axis=h_axis)

    # -- clip-level encode / decode ---------------------------------------------------------

    def _encode_clip(self, x: mx.array) -> mx.array:
        """``x`` is channels-last ``(B, D, H, W, C)``."""
        if not self.use_tiling:
            return self.quant_conv(self.encoder(x))
        h, w = x.shape[2], x.shape[3]
        y_idx, y_len, y_ov = self._split_tiles(h, self.tile_sample_min_height, self.tile_sample_min_overlap_height)
        x_idx, x_len, x_ov = self._split_tiles(w, self.tile_sample_min_width, self.tile_sample_min_overlap_width)

        rows = []
        for i_pos, i_len in zip(y_idx, y_len):
            row = []
            for j_pos, j_len in zip(x_idx, x_len):
                tile = x[:, :, i_pos : i_pos + i_len, j_pos : j_pos + j_len, :]
                encoded = self.quant_conv(self.encoder(tile))
                # One tile's activations at a time. A lazy graph over every tile evaluates them
                # all at once, which is invisible for a one-frame keyframe and is ~16 GiB for
                # 17 frames at 640x384 — the shape carried context and video-to-audio encode.
                # Values are unchanged; only the evaluation order is.
                mx.eval(encoded)
                row.append(encoded)
            rows.append(row)

        ratio = self.config.spatial_compression_ratio
        return self._stitch_tiles(rows, [o // ratio for o in y_ov], [o // ratio for o in x_ov], 2, 3)

    def _decode_tiles(self, tiles: list[mx.array]) -> list[mx.array]:
        """Push every tile through the ViT decoder, batching same-shaped tiles into one call.

        Tiles are grouped by shape — in practice :meth:`_split_tiles` makes them all identical, so
        this is one group — and each group is cut into ``decode_batch``-sized slices. Grouping is
        what makes the batching safe rather than merely convenient: the rotary grid a tile gets
        depends on its ``(d, h, w)``, so only same-shaped tiles may share a call.
        """
        if self.decode_batch <= 1 or len(tiles) < 2:
            return [self._run_decoder(tile) for tile in tiles]

        groups: dict[tuple[int, ...], list[int]] = {}
        for index, tile in enumerate(tiles):
            groups.setdefault(tuple(tile.shape[1:]), []).append(index)

        out: list[mx.array] = [None] * len(tiles)  # type: ignore[list-item]
        for members in groups.values():
            for start in range(0, len(members), self.decode_batch):
                part = members[start : start + self.decode_batch]
                if len(part) == 1:
                    out[part[0]] = self._run_decoder(tiles[part[0]])
                    continue
                stacked = mx.concatenate([tiles[index] for index in part], axis=0)
                decoded = self._run_decoder(stacked)
                cursor = 0
                for index in part:
                    size = tiles[index].shape[0]
                    out[index] = decoded[cursor : cursor + size]
                    cursor += size
        return out

    def _run_decoder(self, z: mx.array) -> mx.array:
        """``post_quant_conv`` + ViT decoder at :attr:`decode_dtype`, returned in the input dtype.

        With the default float32 this is exactly the historical call (the 1x1 conv and every
        linear promote to float32 anyway), so existing renders are bit-for-bit unchanged.
        """
        compute = DECODE_DTYPES[self.decode_dtype]
        h = self.post_quant_conv(z)
        half = MIXED_COMPUTE.get(self.decode_dtype)
        if half is not None:
            return self.decoder(h.astype(mx.float32), half).astype(z.dtype)
        if h.dtype == compute:
            return self.decoder(h)
        return self.decoder(h.astype(compute)).astype(z.dtype)

    def _decode_clip(self, z: mx.array) -> mx.array:
        if not self.use_tiling:
            return self._run_decoder(z)
        ratio = self.config.spatial_compression_ratio
        h, w = z.shape[2] * ratio, z.shape[3] * ratio
        y_idx, y_len, y_ov = self._split_tiles(h, self.tile_sample_min_height, self.tile_sample_min_overlap_height)
        x_idx, x_len, x_ov = self._split_tiles(w, self.tile_sample_min_width, self.tile_sample_min_overlap_width)

        tiles = [
            z[
                :,
                :,
                i_pos // ratio : i_pos // ratio + i_len // ratio,
                j_pos // ratio : j_pos // ratio + j_len // ratio,
                :,
            ]
            for i_pos, i_len in zip(y_idx, y_len)
            for j_pos, j_len in zip(x_idx, x_len)
        ]
        decoded = self._decode_tiles(tiles)
        width = len(x_idx)
        rows = [decoded[i * width : (i + 1) * width] for i in range(len(y_idx))]
        return self._stitch_tiles(rows, y_ov, x_ov, 2, 3)

    # -- public API (channels-first, matching the reference) --------------------------------

    def encode(self, x: mx.array) -> mx.array:
        """Encode ``(B, C, F, H, W)`` pixels into ``(B, 2*latent_channels, F', H', W')`` moments.

        The video is encoded in ``clip_length``-frame chunks and the ``token_drop`` trailing latent
        frames are dropped.
        """
        clip_length = self.config.clip_length
        x = x.transpose(0, 2, 3, 4, 1)  # -> (B, D, H, W, C)
        num_frames = x.shape[1]
        if num_frames % clip_length != 0:
            pad = (-num_frames) % clip_length
            tail = mx.broadcast_to(x[:, -1:], (x.shape[0], pad, *x.shape[2:]))
            x = mx.concatenate([x, tail], axis=1)

        moments = mx.concatenate(
            [
                self._encode_clip(x[:, i * clip_length : (i + 1) * clip_length])
                for i in range(x.shape[1] // clip_length)
            ],
            axis=1,
        )
        if self.config.token_drop > 0:
            moments = moments[:, : -self.config.token_drop]
        return moments.transpose(0, 4, 1, 2, 3)

    def _decode_plan(self, z: mx.array):
        """Shared set-up of :meth:`decode` / :meth:`decode_chunks`: pad and count the chunks."""
        z = z.transpose(0, 2, 3, 4, 1)  # -> (B, D, H, W, C)
        chunk_tokens = self.tokens_chunk_size
        token_drop = self.config.token_drop
        ratio_t = self.config.temporal_compression_ratio

        num_tokens = z.shape[1] + token_drop
        pad_tokens = (-num_tokens) % chunk_tokens
        num_chunks = (num_tokens + pad_tokens) // chunk_tokens - int(token_drop > 0)
        if num_chunks < 1:
            # `token_drop` consumes a whole chunk's worth of lead-in, so the shortest decodable
            # latent is one full chunk beyond it. The reference has the same floor; it just fails
            # with an empty concatenate instead of saying so.
            minimum = 2 * chunk_tokens - token_drop
            raise ValueError(
                f"Too few latent frames to decode: got {z.shape[1]}, need at least {minimum} "
                f"(chunk {chunk_tokens} latent frames, token_drop {token_drop})."
            )
        pad_frames = 0
        if pad_tokens > 0:
            # The repeated latent frames produce trailing pixel frames that were never requested.
            # A chunk's last latent frame only covers `clip_length % ratio_t` pixel frames; others
            # cover `ratio_t`.
            intra_tail = self.config.clip_length % ratio_t
            before_pad = z.shape[1]
            pad_frames = sum(
                intra_tail if intra_tail and (before_pad + k) % chunk_tokens == 0 else ratio_t
                for k in range(pad_tokens)
            )
            tail = mx.broadcast_to(z[:, -1:], (z.shape[0], pad_tokens, *z.shape[2:]))
            z = mx.concatenate([z, tail], axis=1)
        return z, num_chunks, pad_frames

    def _decoded_pieces(self, z: mx.array, num_chunks: int):
        """Yield the decoded pixel pieces, channels-last, in order, each one final.

        Piece ``i`` is chunk ``i`` already cross-faded into the tail of chunk ``i - 1``; the last
        piece is the final chunk's overlap tail. Concatenated they are exactly the whole decode.
        """
        chunk_tokens = self.tokens_chunk_size
        token_drop = self.config.token_drop
        chunk_num_frames = chunk_tokens * self.config.temporal_compression_ratio
        overlap = None
        for i in range(num_chunks):
            start = i * chunk_tokens
            clip = self._decode_clip(z[:, start : start + chunk_tokens + self.token_overlap])
            for j in range(int(token_drop > 0) + 1):
                frame_start = j * chunk_num_frames
                chunk = clip[:, frame_start : frame_start + chunk_num_frames]
                chunk = chunk[:, self.frame_pre_padding :]
                if j == 0:
                    if overlap is not None:
                        chunk = self._blend(overlap, chunk, self.frame_overlap, axis=1)
                    yield chunk
                else:
                    overlap = chunk
        if overlap is not None:
            yield overlap

    def decode(self, z: mx.array) -> mx.array:
        """Decode ``(B, latent_channels, F, H, W)`` latents into ``(B, C, F', H', W')`` pixels.

        Mirrors the chunking ``encode`` applied: ``token_drop`` removed the tail of every encoded
        chunk, so consecutive decoded chunks overlap by ``frame_overlap`` pixel frames and are
        linearly cross-faded. Latent frames are repeated at the end when the length is not a whole
        number of chunks, and the extra pixel frames are cut off again.
        """
        z, num_chunks, pad_frames = self._decode_plan(z)
        out = mx.concatenate(list(self._decoded_pieces(z, num_chunks)), axis=1)
        if pad_frames:
            out = out[:, :-pad_frames]
        return out.transpose(0, 4, 1, 2, 3)

    def decode_chunks(self, z: mx.array):
        """:meth:`decode`, one finished temporal piece at a time, pipelined.

        Yields channels-first ``(B, C, f, H', W')`` float32 pieces whose concatenation along the
        frame axis is bit-for-bit :meth:`decode` (same kernels, same shapes, only the evaluation is
        split). Piece ``i + 1`` is submitted to the GPU (``mx.async_eval``) before piece ``i`` is
        handed back, so whatever the caller does with a piece on the host (copy to numpy,
        uint8 conversion, writing it out) overlaps the decode of the next one instead of waiting
        for the whole clip. Peak memory also drops: a 5 s clip's float32 pixels never sit on the
        GPU all at once.
        """
        z, num_chunks, pad_frames = self._decode_plan(z)
        pieces = self._decoded_pieces(z, num_chunks)
        # Hold back one piece's worth of frames so the padded tail can be cut without knowing in
        # advance which piece it ends in (the overlap tail is shorter than `pad_frames` can be).
        held: list[mx.array] = []
        held_frames = 0

        def release(final: bool):
            nonlocal held_frames
            while held:
                frames = held[0].shape[1]
                # Keep one piece queued behind the one handed out (that is the pipelining), and
                # never hand out a frame that may still turn out to be padding.
                if not final and (len(held) < 2 or held_frames - frames < pad_frames):
                    return
                piece = held.pop(0)
                held_frames -= frames
                if final:
                    keep = max(0, frames - max(0, pad_frames - held_frames))
                    if keep == 0:
                        continue
                    piece = piece[:, :keep]
                yield piece.transpose(0, 4, 1, 2, 3)

        for piece in pieces:
            mx.async_eval(piece)
            held.append(piece)
            held_frames += piece.shape[1]
            yield from release(final=False)
        yield from release(final=True)
