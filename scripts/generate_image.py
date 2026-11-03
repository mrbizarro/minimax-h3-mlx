#!/usr/bin/env python3
"""P-H3-Image text-to-image from the command line.

    python scripts/generate_image.py "a prompt" -o out.png --root MODELS/p-h3-image \
        --dit MODELS/h3-dit-q8 --lora MODELS/p-h3-image/loras/p-h3-image_lora_bf16.safetensors \
        --aspect 1:1 --mp 8 --seed 7 --metrics out.json

The DiT is MiniMax-H3's FL2VA transformer (our pruned/Q8 build); the adapter is applied at load
time (runtime mode, scale 1.0: the file's alpha equals its rank). 8 forwards, no guidance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("-o", "--output", default="out.png")
    ap.add_argument("--root", required=True, help="folder made by fetch_ph3_image.py")
    ap.add_argument("--dit", required=True, help="MiniMax-H3 FL2VA transformer (file or folder)")
    ap.add_argument("--lora", default=None, help="p-h3-image adapter; default: <root>/loras/p-h3-image_lora_bf16.safetensors")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--aspect", default="1:1", help="one of 1:1 4:3 3:4 3:2 2:3 16:9 9:16")
    ap.add_argument("--mp", type=float, default=8.0, help="megapixels (8 is what the model was trained for)")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metrics", default=None, help="write a JSON of timings and memory here")
    ap.add_argument("--keep-adaln", action="store_true")
    args = ap.parse_args(argv)

    from PIL import Image

    from minimax_h3_mlx.ph3_image import PH3ImageRunner, size_for_megapixels

    if args.width and args.height:
        width, height = args.width, args.height
    else:
        width, height = size_for_megapixels(args.aspect, args.mp)
    root = Path(args.root)
    lora = Path(args.lora) if args.lora else root / "loras" / "p-h3-image_lora_bf16.safetensors"
    if not lora.exists():
        print(f"adapter not found: {lora}", file=sys.stderr)
        return 3

    runner = PH3ImageRunner(root, args.dit, lora, lora_scale=args.lora_scale)
    image = runner.generate(args.prompt, width, height, seed=args.seed, drop_adaln=not args.keep_adaln)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out)
    runner.metrics.update({"width": width, "height": height, "seed": args.seed, "prompt": args.prompt,
                           "output": str(out), "forwards": 8})
    if args.metrics:
        Path(args.metrics).write_text(json.dumps(runner.metrics, indent=2))
    print(json.dumps({k: v for k, v in runner.metrics.items() if k != "step_seconds"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
