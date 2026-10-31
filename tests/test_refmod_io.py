"""RefMod reader + strength blur, against torch-recorded goldens and synthetic RefMod files.

    ./.venv/bin/python tests/test_refmod_io.py
The RefMod fixtures are generated on the fly (random latents), so no media is checked in.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.refmod_io import apply_strength, blur_latent, load_refmod, load_refmod_rows, parse_spec

DATA = ROOT / "tests" / "data"
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def test_blur() -> None:
    g = np.load(DATA / "refmod_blur_golden.npz")
    for i in range(5):
        got, want = blur_latent(g[f"in{i}"]), g[f"out{i}"]
        err = float(np.abs(got - want).max())
        check(f"blur {g[f'in{i}'].shape}", got.shape == want.shape and err < 1e-5, f"max abs err {err:.2e}")
    z = g["in0"]
    check("strength 1.0 untouched", apply_strength(z, 1.0) is z)
    mixed = apply_strength(z, 0.25)
    check("strength mix", np.allclose(mixed, 0.25 * z + 0.75 * blur_latent(z), atol=1e-6))


def test_specs() -> None:
    check("spec plain", parse_spec("a/b.safetensors") == ("a/b.safetensors", 1.0))
    check("spec strength", parse_spec("a/b:0.6") == ("a/b", 0.6))
    check("spec non-numeric colon", parse_spec("Vol:name/b") == ("Vol:name/b", 1.0))
    try:
        parse_spec("a:1.5")
        check("spec out of range", False)
    except ValueError:
        check("spec out of range", True)


def _write_refmod(path: Path, kind: str, shape: tuple, name: str = "synthetic") -> None:
    """A format-4 standalone RefMod: one ``latent`` tensor plus the JSON ``refmod_meta`` header."""
    rng = np.random.default_rng(0)
    meta = {"name": name, "kind": kind, "latent_t": shape[2], "latent_h": shape[3],
            "latent_w": shape[4], "_format_version": 4}
    mx.save_safetensors(str(path), {"latent": mx.array(rng.standard_normal(shape).astype(np.float16))},
                        metadata={"refmod_meta": json.dumps(meta)})


def test_synthetic_mod() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mod = Path(tmp) / "synthetic.safetensors"
        _write_refmod(mod, "video", (1, 24, 6, 16, 16))
        mods = load_refmod(mod)
        check("loads one video member", len(mods) == 1 and mods[0].kind == "video")
        check("latent shape", mods[0].latent.shape == (1, 24, 6, 16, 16), str(mods[0].latent.shape))
        rows, geoms, names = load_refmod_rows([f"{mod}:0.8"], (1, 2, 2))
        want_rows = 6 * (16 // 2) * (16 // 2)
        check("packed rows", tuple(rows.shape) == (want_rows, 24 * 4), str(rows.shape))
        check("geometry + name", geoms[0].num_video_rows(2, 2) == want_rows and names == ["synthetic@0.8"])

        image = Path(tmp) / "image.safetensors"
        _write_refmod(image, "image", (1, 24, 1, 20, 12))
        check("image kind", load_refmod(image)[0].geometry.kind == "image")

        audio = Path(tmp) / "audio.safetensors"
        _write_refmod(audio, "audio", (1, 24, 1, 8, 8))
        try:
            load_refmod(audio)
            check("refuses audio RefMod", False)
        except ValueError:
            check("refuses audio RefMod", True)

        odd = Path(tmp) / "odd.safetensors"
        _write_refmod(odd, "video", (1, 24, 2, 15, 16))
        try:
            load_refmod(odd)
            check("refuses odd latent H", False)
        except ValueError:
            check("refuses odd latent H", True)

        try:
            load_refmod_rows([f"{mod}:0"], (1, 2, 2))
            check("refuses all-zero strengths", False)
        except ValueError:
            check("refuses all-zero strengths", True)


if __name__ == "__main__":
    test_blur()
    test_specs()
    test_synthetic_mod()
    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nall ok")
    sys.exit(1 if FAILURES else 0)
