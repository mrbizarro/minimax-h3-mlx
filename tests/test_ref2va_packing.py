"""Parity of the MLX ref2va packed layout against the diffusers reference.

Goldens (tests/data/ref2va_packing_golden.*) were recorded from
reference/diffusers/modular/packing_ref2va.py::build_ref2va_packed_sequence. Position ids are
compared in float32 (the cast the transformer applies); indices and tags exactly.

    ./.venv/bin/python tests/test_ref2va_packing.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.ref2va_packing import RefGeometry, build_ref2va_packed_sequence, validate_references

DATA = ROOT / "tests" / "data"
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def test_golden() -> None:
    golden = np.load(DATA / "ref2va_packing_golden.npz")
    meta = json.loads((DATA / "ref2va_packing_golden.json").read_text())
    for name, m in meta.items():
        refs = [RefGeometry(r["kind"], r["num_latent_frames"], r["latent_height"], r["latent_width"],
                            r["num_audio_latents"]) for r in m["refs"]]
        tf, lh, lw, na = m["target"]
        got = build_ref2va_packed_sequence(np.ones(m["num_text"], dtype=np.int64), refs, tf, lh, lw, na, (1, 2, 2))
        check(f"{name}: length", got.sequence_length == m["seq"], f"{got.sequence_length} vs {m['seq']}")
        check(f"{name}: condition rows",
              (got.num_condition_video_rows, got.num_condition_audio_rows) == (m["cond_v"], m["cond_a"]))
        want_pos = golden[f"{name}/position_ids"].astype(np.float32)
        check(f"{name}: position ids", np.array_equal(np.array(got.position_ids), want_pos),
              f"max abs diff {np.abs(np.array(got.position_ids) - want_pos).max():.3g}")
        for key in ("token_tags", "video_indices", "audio_indices"):
            check(f"{name}: {key}", np.array_equal(np.array(getattr(got, key)), golden[f"{name}/{key}"]))


def test_limits() -> None:
    img = RefGeometry("image", 1, 64, 64)
    for label, refs in [("empty", []), ("10 images", [img] * 10), ("bad kind", [RefGeometry("gif")])]:
        try:
            validate_references(refs)
            check(f"rejects {label}", False)
        except ValueError:
            check(f"rejects {label}", True)
    validate_references([img] * 9)
    check("accepts 9 images", True)


if __name__ == "__main__":
    test_golden()
    test_limits()
    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nall ok")
    sys.exit(1 if FAILURES else 0)
