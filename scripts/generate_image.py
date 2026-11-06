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


def main(argv=None, *, runner_factory=None, default_mp=1.5) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", default="")
    ap.add_argument("--batch-json", help="experimental: JSON requests; keep models warm within this process")
    ap.add_argument("-o", "--output", default="out.png")
    ap.add_argument("--root", required=True, help="folder made by fetch_ph3_image.py")
    ap.add_argument("--dit", required=True, help="MiniMax-H3 FL2VA transformer (file or folder)")
    ap.add_argument("--lora", default=None, help="p-h3-image adapter; default: <root>/loras/p-h3-image_lora_bf16.safetensors")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--aspect", default="1:1", help="one of 1:1 4:3 3:4 3:2 2:3 16:9 9:16")
    ap.add_argument("--mp", type=float, default=default_mp, help="megapixels (8 is what the model was trained for)")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metrics", default=None, help="write a JSON of timings and memory here")
    ap.add_argument("--keep-adaln", action="store_true")
    ap.add_argument("--compile-blocks", action="store_true", help="compile image transformer blocks")
    ap.add_argument("--retain", choices=("all", "dit"), default="all", help="components kept within a batch")
    args = ap.parse_args(argv)

    requests = read_requests(args)
    return run_requests(args, requests, runner_factory=runner_factory)


def read_requests(args):
    """Validate the whole batch before loading any GPU weights or writing any output."""
    requests = json.loads(Path(args.batch_json).read_text()) if args.batch_json else [
        dict(prompt=args.prompt, output=args.output, metrics=args.metrics, seed=args.seed,
             aspect=args.aspect, mp=args.mp, width=args.width, height=args.height)]
    if not isinstance(requests, list) or not 1 <= len(requests) <= 16:
        raise ValueError("batch must contain between 1 and 16 requests")
    paths = set()
    for request in requests:
        if not isinstance(request, dict) or not str(request.get("prompt", "")).strip():
            raise ValueError("each request needs a nonempty prompt")
        if not request.get("output"):
            raise ValueError("each request needs an output path")
        for field in ("output", "metrics"):
            if request.get(field):
                path = Path(request[field]).resolve()
                if path in paths:
                    raise ValueError("batch output and metrics paths must be unique")
                paths.add(path)
        width, height = request.get("width"), request.get("height")
        if bool(width) != bool(height):
            raise ValueError("width and height must be supplied together")
        if width is not None and (not isinstance(width, int) or not 32 <= width <= 8192):
            raise ValueError("width must be an integer between 32 and 8192")
        if height is not None and (not isinstance(height, int) or not 32 <= height <= 8192):
            raise ValueError("height must be an integer between 32 and 8192")
        if not 0 < float(request.get("mp", args.mp)) <= 8:
            raise ValueError("megapixels must be in (0, 8]")
        if request.get("aspect", args.aspect) not in ("1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"):
            raise ValueError("unsupported aspect ratio")
        int(request.get("seed", args.seed))
    return requests


def run_requests(args, requests, *, runner_factory=None):
    import time
    from PIL import Image
    from minimax_h3_mlx.ph3_image import PH3ImageRunner, size_for_megapixels

    root = Path(args.root)
    lora = Path(args.lora) if args.lora else root / "loras" / "p-h3-image_lora_bf16.safetensors"
    if not lora.exists():
        print(f"adapter not found: {lora}", file=sys.stderr)
        return 3

    runner = (runner_factory or PH3ImageRunner)(root, args.dit, lora, lora_scale=args.lora_scale,
                            keep_warm=bool(args.batch_json), retain=getattr(args, "retain", "all"),
                            compile_blocks=getattr(args, "compile_blocks", False))
    try:
        for request in requests:
            started = time.perf_counter()
            if request.get("width") and request.get("height"):
                width, height = request["width"], request["height"]
            else:
                width, height = size_for_megapixels(request.get("aspect", args.aspect),
                                                    float(request.get("mp", args.mp)))
            seed = int(request.get("seed", args.seed))
            image = runner.generate(request["prompt"], width, height, seed=seed,
                                    drop_adaln=not args.keep_adaln)
            out = Path(request["output"])
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(out)
            runner.metrics.update({"width": width, "height": height, "seed": seed,
                                   "prompt": request["prompt"], "output": str(out), "forwards": runner.metrics.get("forwards", 8),
                                   "request_seconds": time.perf_counter() - started})
            if request.get("metrics"):
                Path(request["metrics"]).write_text(json.dumps(runner.metrics, indent=2))
            print(json.dumps({k: v for k, v in runner.metrics.items() if k != "step_seconds"}), flush=True)
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
