"""MLX-only smoke tests for the opt-in draft decoder and disk cache."""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from minimax_h3_mlx.draft_cache import DraftCache  # noqa: E402
from minimax_h3_mlx.tiny_video_vae import TinyH3VideoDecoder  # noqa: E402
from generate_staged import (  # noqa: E402
    distribute_draft_frames,
    draft_video_grid,
    resize_draft_video,
    spread_joint_draft_clock,
)
from minimax_h3_mlx.packing import (  # noqa: E402
    audio_latent_num_frames,
    build_packed_sequence,
    video_latent_num_frames,
)


FAILURES = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    check("124-frame delivery selects the native 56-frame grid", draft_video_grid(124, 12) == 56)
    check("exact five-second delivery also selects 56 frames", draft_video_grid(120, 12) == 56)
    check("15 fps budget selects the native 73-frame grid", draft_video_grid(120, 15) == 73)
    check("18 fps budget selects the native 90-frame grid", draft_video_grid(120, 18) == 90)
    source = np.arange(56, dtype=np.uint8).reshape(56, 1, 1, 1)
    delivered = distribute_draft_frames(source, 124)
    check("draft mux preserves the requested duration", len(delivered) == 124)
    check("draft mux retains every source frame", len(np.unique(delivered)) == 56)
    resized = resize_draft_video(np.zeros((2, 4, 6, 3), dtype=np.uint8), 8, 10)
    check("draft delivery resize uses requested canvas", resized.shape == (2, 10, 8, 3))
    layout = build_packed_sequence(
        [1], video_latent_num_frames(56), 4, 4, audio_latent_num_frames(56), (1, 2, 2)
    )
    rows_before = layout.sequence_length
    times_before = np.asarray(layout.position_ids)[:, 0].copy()
    spread_joint_draft_clock(layout, 1, 120 / 56)
    times_after = np.asarray(layout.position_ids)[:, 0]
    check("joint clock spread adds no packed rows", layout.sequence_length == rows_before)
    check("joint clock spread advances the media tail", times_after.max() > times_before.max() * 2)

    mx.random.seed(0)
    decoder = TinyH3VideoDecoder()
    # Seven H3 tokens are the smallest native grid (22 pixel frames after restoring the three
    # encoder-tail tokens). A 1x1 latent pixel keeps the smoke cheap while exercising every layer,
    # both temporal grows, per-chunk trimming and the final 2x pixel shuffle.
    latent = mx.zeros((1, 24, 7, 1, 1), dtype=mx.float16)
    decoded = decoder.decode(latent, 22)
    mx.eval(decoded)
    check("tiny decoder recovers the 22-frame H3 grid", decoded.shape == (1, 3, 22, 16, 16))
    check(
        "tiny decoder clamps display RGB",
        float(mx.min(decoded).item()) >= 0.0 and float(mx.max(decoded).item()) <= 1.0,
    )

    with tempfile.TemporaryDirectory() as temporary:
        cache = DraftCache(temporary, limit=2)
        tags = np.array([1, 2, 3], dtype=np.int64)
        for index in range(3):
            embeds = np.full((1, 3, 4), index, dtype=np.float32)
            digest = hashlib.sha256(embeds.tobytes()).hexdigest()
            cache.store_text(f"request-{index}", embeds, tags, digest)
        requests = list((Path(temporary) / "text_requests").glob("*.json"))
        entries = list((Path(temporary) / "text").glob("*.npz"))
        check("text request LRU is bounded", len(requests) == 2, f"got {len(requests)}")
        check("unreferenced embedding payloads are pruned", len(entries) == 2, f"got {len(entries)}")

        embeds = np.full((1, 3, 4), 2, dtype=np.float32)
        digest = hashlib.sha256(embeds.tobytes()).hexdigest()
        hit = cache.load_text("request-2")
        check(
            "text hit is content-addressed by embedding SHA",
            hit is not None and hit[2] == digest and np.array_equal(hit[0], embeds),
        )

        cache.store_noise("noise", video=embeds, audio=tags)
        noise = cache.load_noise("noise")
        check(
            "seeded noise round-trips exactly",
            noise is not None
            and np.array_equal(noise["video"], embeds)
            and np.array_equal(noise["audio"], tags.astype(np.float32)),
        )

        keyframe_rows = np.arange(24, dtype=np.float32).reshape(3, 8)
        cache.store_keyframe("keyframe", keyframe_rows)
        check(
            "keyframe rows round-trip exactly",
            np.array_equal(cache.load_keyframe("keyframe"), keyframe_rows),
        )

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("draft speed smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
