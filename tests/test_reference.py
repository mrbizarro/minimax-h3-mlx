"""Reference images on FL2VA: packing slots, image sizing, and carried-frame labels."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from minimax_h3_mlx.dit import MiniMaxH3DiT
from minimax_h3_mlx.packing import build_packed_sequence
from minimax_h3_mlx.reference import reference_size
from test_dit_smoke import tiny_config


def test_no_reference_is_the_old_layout():
    tags = np.ones(7, dtype=np.int64)
    a = build_packed_sequence(tags, 8, 24, 40, 50, (1, 2, 2), ("first",))
    b = build_packed_sequence(tags, 8, 24, 40, 50, (1, 2, 2), ("first",), reference_grids=())
    for f in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices"):
        assert np.array_equal(np.array(getattr(a, f)), np.array(getattr(b, f)))


def test_reference_blocks_lead_the_conditioning_and_the_clock():
    tags = np.ones(7, dtype=np.int64)
    base = build_packed_sequence(tags, 8, 24, 40, 50, (1, 2, 2), ("first",))
    ref = build_packed_sequence(tags, 8, 24, 40, 50, (1, 2, 2), ("first",),
                                reference_grids=((64, 48), (32, 32)))
    n1, n2 = 32 * 24, 16 * 16
    assert ref.num_condition_video_rows == base.num_condition_video_rows + n1 + n2
    assert ref.num_condition_audio_rows == base.num_condition_audio_rows
    pos = np.array(ref.position_ids)
    vid = np.array(ref.video_indices.tolist())
    # first reference: one slot at the text length, second at +1
    assert np.all(pos[vid[:n1], 0] == 7.0)
    assert np.all(pos[vid[n1:n1 + n2], 0] == 8.0)
    # the target (and its "first" keyframe) moved two slots later
    bpos = np.array(base.position_ids)
    bvid = np.array(base.video_indices.tolist())
    np.testing.assert_allclose(pos[vid[n1 + n2:], 0], bpos[bvid, 0] + 2.0, atol=1e-5)
    # its own aspect-normalised grid: the extent follows the reference's own aspect
    h = pos[vid[:n1], 1]; w = pos[vid[:n1], 2]
    span_h = (h.max() - h.min()) * 32 / 31; span_w = (w.max() - w.min()) * 24 / 23
    np.testing.assert_allclose(span_h / span_w, 64 / 48, rtol=1e-3)
    np.testing.assert_allclose(span_h, 32 * 64 / np.sqrt(64 * 48), rtol=1e-3)
    assert np.all(np.array(ref.token_tags)[vid] == 0)


def test_reference_size_follows_the_official_rule():
    assert reference_size(1086, 1448, 2048) == (2048, 2720)
    assert reference_size(1024, 598, 1024) == (1760, 1024)
    w, h = reference_size(333, 777, 512)
    assert w % 32 == 0 and h % 32 == 0 and min(w, h) == 512
    try:
        reference_size(100, 500)
    except ValueError:
        pass
    else:
        raise AssertionError


def test_reference_labels_follow_the_carried_frame():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gs", Path(__file__).resolve().parents[1] / "scripts" / "generate_staged.py")
    gs = importlib.util.module_from_spec(spec)
    sys.modules["gs"] = gs
    spec.loader.exec_module(gs)
    p = "The person from <Picture 1> walks. <Picture 1> smiles."
    assert gs.shift_reference_labels(p, 0, 0) == p
    assert gs.shift_reference_labels(p, 0, 1) == p.replace("<Picture 1>", "<Picture 2>")
    q = "Frame <Picture 1>, face <Picture 2>."
    assert gs.shift_reference_labels(q, 1, 1) == q
    assert gs.shift_reference_labels(q, 1, 2) == "Frame <Picture 1>, face <Picture 3>."


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
