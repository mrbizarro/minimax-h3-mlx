"""Build an unquantized MLX copy of the MiniMax-H3 DiT.

    ./.venv/bin/python scripts/build_unquantized.py --dtype native --out .../MiniMax-H3-MLX-bf16
    ./.venv/bin/python scripts/build_unquantized.py --dtype float32 --out .../MiniMax-H3-MLX-f32

Three dtype policies, and the distinction matters:

* ``native`` (default) — preserve the release's **mixed** precision. MiniMax ships the two patch
  projections, the timestep MLP and the two output heads in float32 and everything else in
  bfloat16. This is the faithful conversion and what a "bf16" build should be.
* ``bfloat16`` — flatten everything to bfloat16. This is a *downgrade* from the release: it rounds
  the twelve float32 tensors the reference explicitly keeps in float32, including the timestep MLP,
  whose output feeds every block's modulation. Offered for completeness, not recommended.
* ``float32`` — upcast everything. Doubles the footprint and adds **no information**: the source
  weights are bfloat16, so this is a lossless widening, not extra precision. Useful if you want a
  float32 base for fine-tuning; pointless as a download if you only want to generate, since
  ``load_dit(dtype=mx.float32)`` does the same thing locally.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_quant import save_sharded

from minimax_h3_mlx.load import load_dit
from minimax_h3_mlx.quantize import resident_footprint

DTYPES = {"bfloat16": mx.bfloat16, "float32": mx.float32}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/Volumes/models/MiniMax-H3/FL2VA")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", choices=["native", "bfloat16", "float32"], default="native")
    args = parser.parse_args()

    source = Path(args.checkpoint)
    out_dir = Path(args.out)

    print(f"loading {source / 'transformer'} (dtype policy: {args.dtype})", flush=True)
    started = time.perf_counter()
    # `None` keeps the checkpoint's own mixed float32/bfloat16 split.
    model = load_dit(source / "transformer", dtype=DTYPES.get(args.dtype))
    print(f"  loaded in {time.perf_counter() - started:.1f}s", flush=True)

    counts: dict[str, int] = {}
    for _, value in tree_flatten(model.parameters()):
        counts[str(value.dtype).rsplit(".", 1)[-1]] = counts.get(str(value.dtype).rsplit(".", 1)[-1], 0) + 1
    footprint = resident_footprint(model)
    print(f"  dtypes: {counts}")
    print(f"  on disk {footprint['total_gb']:.1f} GB; resident after the adaln drop "
          f"{footprint['resident_gb']:.1f} GB", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(source / "transformer" / "config.json", out_dir / "config.json")

    meta = {
        "quantized": False,
        "dtype_policy": args.dtype,
        "tensor_dtypes": counts,
        "gb_on_disk": round(footprint["total_gb"], 2),
        "gb_resident_after_adaln_drop": round(footprint["resident_gb"], 2),
    }
    # Deliberately not `quant_config.json`: `load_dit` treats that file as a signal to rebuild a
    # quantized module tree, and this build has none.
    with open(out_dir / "build_config.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    started = time.perf_counter()
    names = save_sharded(model, out_dir, {"build": json.dumps(meta)})
    print(f"  wrote {len(names)} shards in {time.perf_counter() - started:.1f}s", flush=True)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
