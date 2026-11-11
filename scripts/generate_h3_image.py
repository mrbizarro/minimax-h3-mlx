#!/usr/bin/env python3
"""MiniMax-H3-Image text-to-image, and edit with reference pictures, from the command line.

    python scripts/generate_h3_image.py "a long, detailed English description" -o out.png \
        --root MODELS/h3-image --text-dir MODELS/ddalcu-q8 \
        --text-config MODELS/upstream-meta/FL2VA/text_encoder/config.json \
        --aspect 1:1 --mp 4 --seed 7 --metrics out.json

``--root`` is the folder ``fetch_h3_image.py`` builds. ``--text-dir`` is the H3 text-encoder folder the
other H3 modes already use. 6 forwards, no guidance, no negative prompt (the model has neither).
``--mp`` below 4 is generated on the 4 MP canvas and resized back, like the reference.

Edit: ``--reference PATH`` (repeat up to 16 times; the first is Picture 1, the one being edited).
Name them in the prompt as "Picture 1", "Picture 2", ... ``--match-reference`` makes the output the
shape of Picture 1 at ``--mp`` megapixels instead of ``--aspect``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("-o", "--output", default="out.png")
    ap.add_argument("--root", required=True)
    ap.add_argument("--text-dir", required=True)
    ap.add_argument("--text-config", required=True, help="upstream text_encoder/config.json (models/upstream-meta/FL2VA/text_encoder/)")
    ap.add_argument("--aspect", default="1:1")
    ap.add_argument("--mp", type=float, default=4.0, help="megapixels (the model works best at 16)")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--no-canvas", action="store_true", help="generate exactly the requested (32-aligned) size")
    ap.add_argument("--reference", action="append", default=[], help="a reference picture (edit); repeatable, Picture 1 first")
    ap.add_argument("--match-reference", action="store_true", help="output in the shape of Picture 1")
    args = ap.parse_args(argv)
    if not args.prompt.strip():
        raise SystemExit("a prompt is required")
    if bool(args.width) != bool(args.height):
        raise SystemExit("width and height must be supplied together")

    from PIL import Image
    from minimax_h3_mlx.h3_image import (DEFAULT_MIN_GENERATION_AREA, MAX_REFERENCES, H3ImageRunner,
                                         load_reference, size_for_megapixels, size_matching)

    if len(args.reference) > MAX_REFERENCES:
        raise SystemExit(f"at most {MAX_REFERENCES} reference pictures")
    if args.match_reference and not args.reference:
        raise SystemExit("--match-reference needs a --reference")
    references = [load_reference(p) for p in args.reference]
    if args.width:
        width, height = args.width, args.height
    elif args.match_reference:
        width, height = size_matching(references[0], args.mp)
    else:
        width, height = size_for_megapixels(args.aspect, args.mp)
    runner = H3ImageRunner(args.root, args.text_dir, args.text_config)
    started = time.perf_counter()
    image = runner.generate(args.prompt, width, height, seed=args.seed,
                            min_generation_area=None if args.no_canvas else DEFAULT_MIN_GENERATION_AREA,
                            references=references)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out)
    runner.metrics.update({"width": width, "height": height, "seed": args.seed, "prompt": args.prompt,
                           "reference_paths": list(args.reference),
                           "output": str(out), "request_seconds": time.perf_counter() - started})
    if args.metrics:
        Path(args.metrics).write_text(json.dumps(runner.metrics, indent=2))
    print(json.dumps({k: v for k, v in runner.metrics.items() if k != "step_seconds"}), flush=True)
    runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
