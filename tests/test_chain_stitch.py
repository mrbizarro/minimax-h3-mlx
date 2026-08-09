"""Chained-window stitching: frame accounting, audio/video sync, the seam cross-fade, and the
per-window prompt fan-out.

Pure NumPy — no weights, no GPU. What it pins down is the part of window chaining that a render
cannot check for you: that the duplicate frame leaves exactly one frame of audio with it, so a
three-window clip's dialogue still lands on its own lip movements at the end, and that a shot list
of N prompts can only ever be spent on N windows.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_staged import draft_duration_plan, parse_chain_prompts, stitch_windows  # noqa: E402
from minimax_h3_mlx.packing import FPS  # noqa: E402

SAMPLE_RATE = 32000
FRAME_SAMPLES = round(SAMPLE_RATE / FPS)


def window(frames: int, fill: int, offset: float, seed: int):
    rng = np.random.default_rng(seed)
    video = np.full((frames, 8, 8, 3), fill, dtype=np.uint8)
    samples = round(frames / FPS * SAMPLE_RATE)
    audio = (rng.standard_normal((2, samples)) * 0.05 + offset).astype(np.float32)
    return video, audio


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{' — ' + detail if detail else ''}")
    return ok


def refusal(spec: str, windows: int) -> str:
    """The message ``parse_chain_prompts`` refuses with, or '' if it accepted the spec."""
    try:
        parse_chain_prompts(spec, windows)
    except ValueError as exc:
        return str(exc)
    return ""


def prompt_tests() -> bool:
    passed = True
    three = "Shot one. ||| Shot two. ||| Shot three, and he says: it runs on a Mac."

    passed &= check(
        "' ||| ' splits into one prompt per window",
        parse_chain_prompts(three, 3)
        == ["Shot one.", "Shot two.", "Shot three, and he says: it runs on a Mac."],
        str(parse_chain_prompts(three, 3)),
    )
    # A shell here-doc, a copy-paste, or a wrapped line all leave stray whitespace around the bars.
    passed &= check(
        "separator tolerates ragged whitespace",
        parse_chain_prompts("  A dune.   |||B dune.\n", 2) == ["A dune.", "B dune."],
    )
    passed &= check(
        "one prompt, one window still parses",
        parse_chain_prompts("Only shot.", 1) == ["Only shot."],
    )

    # The whole point of the flag is that window i gets prompt i. A count that does not match the
    # chain has no correct interpretation, so it must not be guessed at.
    short = refusal(three, 4)
    passed &= check(
        "too few prompts is refused, with both counts named",
        "3 prompt(s)" in short and "is 4" in short,
        short,
    )
    long = refusal(three, 2)
    passed &= check(
        "too many prompts is refused rather than truncated",
        "3 prompt(s)" in long and "is 2" in long,
        long,
    )
    passed &= check(
        "an empty prompt is refused with its position",
        "position(s) [2]" in refusal("Shot one. ||| ||| Shot three.", 3),
        refusal("Shot one. ||| ||| Shot three.", 3),
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shots.json"
        path.write_text(json.dumps(["Shot one.", "Shot two.", "Shot three."]))
        passed &= check(
            "a JSON list file is the same shot list",
            parse_chain_prompts(str(path), 3) == ["Shot one.", "Shot two.", "Shot three."],
        )
        passed &= check(
            "a JSON file with the wrong count is refused too",
            "3 prompt(s)" in refusal(str(path), 2),
            refusal(str(path), 2),
        )
        wrong = Path(tmp) / "wrong.json"
        wrong.write_text(json.dumps({"w1": "Shot one."}))
        passed &= check(
            "a JSON file that is not a list of strings is refused",
            "list of strings" in refusal(str(wrong), 1),
            refusal(str(wrong), 1),
        )
        passed &= check(
            "a missing .json path is named, not silently treated as a prompt",
            "does not exist" in refusal(str(Path(tmp) / "absent.json"), 1),
        )
    return passed


def main() -> int:
    passed = True
    frames = 124

    passed &= check(
        "3-second draft uses the lower native grid", draft_duration_plan(3) == (73, 1, None)
    )
    passed &= check(
        "5-second draft is the approved native grid", draft_duration_plan(5) == (124, 1, None)
    )
    passed &= check(
        "10-second draft is two windows trimmed to 240",
        draft_duration_plan(10) == (124, 2, 240),
    )
    passed &= check(
        "15-second draft is three windows trimmed to 360",
        draft_duration_plan(15) == (124, 3, 360),
    )

    for windows in (2, 3):
        segments = [window(frames, 10 * (i + 1), 0.4 * i, seed=i) for i in range(windows)]
        video, audio, seams = stitch_windows(segments, SAMPLE_RATE, 1.0 / FPS)

        expected_frames = frames + (windows - 1) * (frames - 1)
        passed &= check(
            f"{windows} windows deliver {expected_frames} frames",
            len(video) == expected_frames,
            f"got {len(video)}",
        )
        # The whole point of dropping one frame of audio with the duplicate frame: sync survives.
        drift = abs(audio.shape[1] / SAMPLE_RATE - len(video) / FPS)
        passed &= check(
            f"{windows} windows stay in sync",
            drift < 1.5 / SAMPLE_RATE,
            f"drift {drift * 1000:.4f} ms",
        )
        passed &= check(
            f"{windows} windows report {windows - 1} seam(s)",
            len(seams) == windows - 1 and all(s["fade_samples"] == FRAME_SAMPLES for s in seams),
            str(seams),
        )
        # The duplicate frame really is gone: window 2 opens on its own second frame.
        passed &= check(
            f"{windows} windows drop the duplicate frame",
            int(video[frames][0, 0, 0]) == 20,
            f"frame {frames} fill {int(video[frames][0, 0, 0])}",
        )

    # A DC step between two windows is exactly the blind-concatenation pop. The cross-fade has to
    # spread it over the fade, and 0 has to reproduce the pop so the two are comparable.
    segments = [window(frames, 10, 0.0, seed=11), window(frames, 20, 1.0, seed=12)]
    seam = round(frames / FPS * SAMPLE_RATE)
    _, faded, _ = stitch_windows(segments, SAMPLE_RATE, 1.0 / FPS)
    _, blunt, _ = stitch_windows(segments, SAMPLE_RATE, 0.0)
    step_faded = float(np.abs(np.diff(faded[:, seam - 2 : seam + 2], axis=1)).max())
    step_blunt = float(np.abs(np.diff(blunt[:, seam - 2 : seam + 2], axis=1)).max())
    passed &= check(
        "cross-fade flattens the seam step",
        step_faded < step_blunt / 3,
        f"{step_faded:.4f} vs {step_blunt:.4f}",
    )
    passed &= check(
        "cross-fade of 0 is a blind concatenation",
        np.array_equal(blunt[:, seam:], segments[1][1][:, FRAME_SAMPLES:]),
    )
    passed &= check(
        "cross-fade never changes the length",
        faded.shape == blunt.shape,
        f"{faded.shape} vs {blunt.shape}",
    )

    passed &= prompt_tests()

    print("\nALL CHAIN STITCH TESTS PASSED" if passed else "\nCHAIN STITCH TESTS FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
