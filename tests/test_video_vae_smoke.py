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

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("video VAE smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
