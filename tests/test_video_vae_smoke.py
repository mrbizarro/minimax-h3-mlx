"""Tiny-config video-VAE tests that need only MLX — no weights, no torch reference.

The parity suite next door compares against diffusers and so needs the torch environment from
`requirements.txt`. This one has to stay runnable anywhere, because what it guards is cheap to
break and expensive to notice: the decoder is invoked once per *batch of spatial tiles*, and if
batching ever stopped being the loop's exact arithmetic the damage would be a subtle seam at the
tile boundaries of a thirty-minute render.

    ./.venv/bin/python tests/test_video_vae_smoke.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.video_vae import (  # noqa: E402
    DEFAULT_DECODE_BATCH,
    DEFAULT_DECODE_DTYPE,
    VideoVAE,
    VideoVAEConfig,
    resolved_decode_batch,
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def tiny_config() -> VideoVAEConfig:
    """Keeps the checkpoint-tied geometry (compression ratios, clip length, rotary ratio) and
    shrinks only what is free to shrink, so the tiling and chunking maths is the real one."""
    return VideoVAEConfig(
        in_channels=3,
        out_channels=3,
        latent_channels=8,
        block_out_channels=(8, 16),
        layers_per_block=1,
        spatial_downsample_factors=(2, 2, 2, 2),
        temporal_downsample_factors=(1, 2, 2, 1),
        norm_num_groups=4,
        decoder_num_layers=2,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=16,
        decoder_num_register_tokens=2,
        decoder_ffn_mult=2,
    )


def main() -> int:
    mx.random.seed(0)
    config = tiny_config()
    model = VideoVAE(config)
    mx.eval(model.parameters())

    ratio = config.spatial_compression_ratio
    check("compression ratios", (ratio, config.temporal_compression_ratio) == (16, 4),
          f"spatial {ratio}, temporal {config.temporal_compression_ratio}")
    check("default batch decodes a 768x448 clip in one call", DEFAULT_DECODE_BATCH >= 8,
          f"DEFAULT_DECODE_BATCH = {DEFAULT_DECODE_BATCH}")

    # `H3_VAE_BATCH` is the kill switch the report and the runner both promise.
    previous = os.environ.get("H3_VAE_BATCH")
    try:
        for raw, want in [("0", 0), ("4", 4), ("nonsense", DEFAULT_DECODE_BATCH), ("-3", 0)]:
            os.environ["H3_VAE_BATCH"] = raw
            got = resolved_decode_batch()
            check(f"H3_VAE_BATCH={raw!r}", got == want, f"resolved to {got}, wanted {want}")
        os.environ.pop("H3_VAE_BATCH")
        check("H3_VAE_BATCH unset", resolved_decode_batch() == DEFAULT_DECODE_BATCH)
    finally:
        os.environ.pop("H3_VAE_BATCH", None)
        if previous is not None:
            os.environ["H3_VAE_BATCH"] = previous

    # The decode floor is one full chunk beyond `token_drop`; 5 latent frames clears it.
    latent_frames = 5
    for label, (height, width), tile in [
        ("untiled 64x64", (64, 64), 256),
        ("tiled 96x96 (2x2)", (96, 96), 64),
        ("tiled 96x160 (2x3)", (96, 160), 64),
    ]:
        model.tile_sample_min_height = model.tile_sample_min_width = tile
        model.tile_sample_min_overlap_height = model.tile_sample_min_overlap_width = tile // 4
        tiles = len(model._split_tiles(height, tile, tile // 4)[0]) * \
            len(model._split_tiles(width, tile, tile // 4)[0])

        latent = mx.random.normal((1, config.latent_channels, latent_frames,
                                   height // ratio, width // ratio))
        mx.eval(latent)

        model.decode_batch = 0
        loop = model.decode(latent)
        mx.eval(loop)

        for batch in (2, DEFAULT_DECODE_BATCH, 64):
            model.decode_batch = batch
            batched = model.decode(latent)
            mx.eval(batched)
            same = batched.shape == loop.shape and bool(mx.array_equal(loop, batched).item())
            check(f"{label} batch {batch} == loop", same, f"{tiles} tile(s) per clip")

        model.decode_batch = DEFAULT_DECODE_BATCH

    # Decoder precision. The shipped checkpoint stores FP16 weights and the runner passes float32
    # latents, so the historical decode ran float32 arithmetic. `decode_dtype="float32"` must stay
    # that exact call; float16 must stay close and still hand back float32 pixels.
    model.set_dtype(mx.float16)
    model.tile_sample_min_height = model.tile_sample_min_width = 64
    model.tile_sample_min_overlap_height = model.tile_sample_min_overlap_width = 16
    latent = mx.random.normal((1, config.latent_channels, latent_frames, 96 // ratio, 160 // ratio))
    model.decode_batch = 0
    model.decode_dtype = "float32"
    tile = latent.transpose(0, 2, 3, 4, 1)[:, :, :4, :4, :]
    historical = model.decoder(model.post_quant_conv(tile))
    current = model._run_decoder(tile)
    check("float32 decode_dtype is the historical call",
          historical.dtype == mx.float32 and bool(mx.array_equal(historical, current).item()),
          f"historical dtype {historical.dtype}")
    ref = model.decode(latent)
    model.decode_dtype = "float16"
    half = model.decode(latent)
    mx.eval(ref, half)
    rel = (mx.abs(half - ref).max() / (mx.abs(ref).max() + 1e-6)).item()
    check("float16 decode returns float32 and stays close", half.dtype == mx.float32 and rel < 1e-2,
          f"max rel diff {rel:.2e}")
    # float16-mixed (the default since 2026-10-30): residual stream float32, projections and
    # attention fp16. Must hand back float32 pixels, stay closer to float32 than the all-fp16
    # stream does, and the compiled blocks must be the uncompiled mixed block bit for bit.
    check("default decode dtype is float16-mixed", DEFAULT_DECODE_DTYPE == "float16-mixed")
    # The block output scales initialise to zero, which would make every block the identity and
    # every precision trivially equal. Give them real values first.
    for block in model.decoder.transformer_blocks:
        block.scale1 = (mx.random.uniform(0.1, 1.0, block.scale1.shape)).astype(block.scale1.dtype)
        block.scale2 = (mx.random.uniform(0.1, 1.0, block.scale2.shape)).astype(block.scale2.dtype)
    model.decode_dtype = "float32"
    ref = model.decode(latent)
    model.decode_dtype = "float16"
    half = model.decode(latent)
    mx.eval(ref, half)
    rel = (mx.abs(half - ref).max() / (mx.abs(ref).max() + 1e-6)).item()
    model.decode_dtype = "float16-mixed"
    mixed = model.decode(latent)
    os.environ["H3_VAE_COMPILE"] = "0"
    try:
        mixed_eager = model.decode(latent)
        mx.eval(mixed, mixed_eager)
    finally:
        os.environ.pop("H3_VAE_COMPILE", None)
    rel_mixed = (mx.abs(mixed - ref).max() / (mx.abs(ref).max() + 1e-6)).item()
    check("float16-mixed returns float32 and stays close",
          mixed.dtype == mx.float32 and rel_mixed < 1e-2 and rel_mixed <= rel,
          f"max rel diff {rel_mixed:.2e} (all-fp16 {rel:.2e})")
    check("compiled mixed blocks == eager mixed blocks", bool(mx.array_equal(mixed, mixed_eager).item()))

    # Pipelined decode: the pieces concatenated are decode() bit for bit, padded tail included.
    for frames_in in (5, 6, 7, 9):
        z = mx.random.normal((1, config.latent_channels, frames_in, 96 // ratio, 160 // ratio))
        for dtype in ("float32", "float16-mixed"):
            model.decode_dtype = dtype
            whole = model.decode(z)
            pieces = list(model.decode_chunks(z))
            joined = mx.concatenate(pieces, axis=2)
            same = joined.shape == whole.shape and bool(mx.array_equal(joined, whole).item())
            check(f"decode_chunks == decode ({frames_in} latent frames, {dtype})", same,
                  f"{len(pieces)} pieces, {joined.shape} vs {whole.shape}")
    model.decode_dtype = "float32"
    model.decode_batch = DEFAULT_DECODE_BATCH

    # Single-frame encode: a causal 3x3x3 conv over one frame is a Conv2d with the last temporal
    # tap. Equal to float32 rounding (not bit-identical), and multi-frame encodes never take it.
    model.tile_sample_min_height = model.tile_sample_min_width = 64
    model.tile_sample_min_overlap_height = model.tile_sample_min_overlap_width = 16
    for frames_in, label in ((1, "one frame"), (17, "17 frames")):
        pixels = mx.random.uniform(-1, 1, (1, frames_in, 96, 160, 3))
        slow = model._encode_clip(pixels)  # the default: the 3D conv, as released
        os.environ["H3_VAE_ENCODER_CONV2D"] = "1"
        try:
            fast = model._encode_clip(pixels)
            mx.eval(slow, fast)
        finally:
            os.environ.pop("H3_VAE_ENCODER_CONV2D", None)
        rel_l2 = (mx.sqrt(((fast - slow) ** 2).sum()) / mx.sqrt((slow ** 2).sum())).item()
        if frames_in == 1:
            check("one-frame encode as Conv2d matches the 3D conv", fast.shape == slow.shape and rel_l2 < 1e-5,
                  f"rel L2 {rel_l2:.2e}")
        else:
            check("17-frame encode is untouched by the Conv2d shortcut",
                  bool(mx.array_equal(fast, slow).item()))

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("video VAE smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
