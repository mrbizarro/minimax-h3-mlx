"""Resident-engine wiring: what serve mode replaces, and what it must not move.

MLX only — no weights, no 41 GB read. A serve session's whole claim to being safe is that it
changes *which* objects are in memory and nothing about the arithmetic, so the things worth pinning
down without a render are the seams: that the loader interception actually rebinds the names
``render_window`` calls, that the memoized AdaLN table is keyed on the timesteps it was built from
(a stale table silently denoises to the wrong schedule), that dropping the AdaLN weights really is a
no-op in serve mode, and that the conditioning cache cannot serve one canvas's embeds to another.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate_staged as staged  # noqa: E402
import serve_staged as serve  # noqa: E402

failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{' - ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


class FakeBlock:
    """Stands in for a transformer block's AdaLN projection."""

    def __init__(self, scale: float):
        self.scale = scale
        self.calls = 0

    def adaln_proj(self, temb):
        self.calls += 1
        return (temb * self.scale,)


class FakeDiT:
    def __init__(self):
        self.blocks = [FakeBlock(1.0), FakeBlock(2.0)]

    def embed_timesteps(self, timesteps):
        return timesteps[:, None] * mx.ones((1, 4))


def main() -> int:
    # --- 1. the memoized modulation cache -------------------------------------------------------
    serve._CachingModulationCache.clear()
    serve._CachingModulationCache.hits = 0
    serve._CachingModulationCache.misses = 0

    dit = FakeDiT()
    table_a = mx.array([0.0, 0.5, 1.0], dtype=mx.float32)
    table_b = mx.array([0.0, 0.25, 1.0], dtype=mx.float32)

    first = serve._CachingModulationCache.build(dit, table_a)
    again = serve._CachingModulationCache.build(dit, table_a)
    other = serve._CachingModulationCache.build(dit, table_b)

    check("same timesteps reuse one table", first is again)
    check("different timesteps rebuild", other is not first)
    check(
        "hit/miss accounting",
        (serve._CachingModulationCache.hits, serve._CachingModulationCache.misses) == (1, 2),
        f"{serve._CachingModulationCache.hits}/{serve._CachingModulationCache.misses}",
    )
    check(
        "a reused table is not merely equal but identical",
        bool(mx.array_equal(first.get(0)[0], again.get(0)[0])),
    )
    # The whole point of memoizing: the second identical schedule must not touch the weights.
    check("no redundant projection work", dit.blocks[0].calls == 2, f"{dit.blocks[0].calls} calls")

    # A schedule that differs only in the conditioning level is a different schedule. Serving the
    # first one's table would denoise every row at the wrong noise level.
    keyframe_table = mx.array([0.0, 0.5, 0.999, 1.0], dtype=mx.float32)
    check(
        "adding a conditioning level is a cache miss",
        serve._CachingModulationCache.build(dit, keyframe_table) is not first,
    )

    # --- 2. loader interception ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        opts = type(
            "Opts",
            (),
            {
                "dit": "/nonexistent/dit.safetensors",
                "compact_root": "/nonexistent",
                "text_config": None,
                "cond_cache": tmp,
                "resident_vae": False,
                "runner_digest": "test",
            },
        )()
        engine = serve.ResidentEngine(opts)

        before = (
            staged.load_dit,
            staged.drop_adaln_weights,
            staged.MiniMaxH3TextEncoder,
            staged.ModulationCache,
        )
        engine.install()
        after = (
            staged.load_dit,
            staged.drop_adaln_weights,
            staged.MiniMaxH3TextEncoder,
            staged.ModulationCache,
        )
        check("every loader render_window calls is rebound", all(a is not b for a, b in zip(before, after)))

        # Serve mode keeps the AdaLN projections so the next job can build a different schedule.
        # It must report zero freed bytes rather than lie about a saving it did not take.
        check("drop_adaln_weights is a no-op reporting 0", staged.drop_adaln_weights(dit) == 0)
        check("...and left the projections in place", hasattr(dit.blocks[0], "adaln_proj"))

        # --- 3. conditioning cache keying --------------------------------------------------------
        base = engine.cond_cache_path("a prompt", "/still.png", 384, 640)
        same = engine.cond_cache_path("a prompt", "/still.png", 384, 640)
        checks = {
            "prompt": engine.cond_cache_path("other prompt", "/still.png", 384, 640),
            "first frame": engine.cond_cache_path("a prompt", "/other.png", 384, 640),
            "height": engine.cond_cache_path("a prompt", "/still.png", 448, 640),
            "width": engine.cond_cache_path("a prompt", "/still.png", 384, 768),
        }
        check("same request hits the same cache file", base == same)
        for field, path in checks.items():
            # The vision tower reads the still already placed on the render canvas, so geometry is
            # part of the key just as much as the prompt is.
            check(f"a different {field} is a different cache entry", path != base)

        check("engine starts with nothing resident", engine.dit is None)
        engine.dit = object()
        engine.free_dit("test")
        check("free_dit releases the DiT", engine.dit is None)
        check("free_dit drops memoized tables too", not serve._CachingModulationCache.store)

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("serve engine wiring OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
