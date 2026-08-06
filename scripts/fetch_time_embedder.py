"""Recover the upstream 2688-d timestep MLP without downloading a 66 GB checkpoint.

DeepBeepMeep's pruned export replaces the per-block AdaLN projection ``[96768, 2688]`` with
``[96768, 64]`` over a sampled rank-64 curve, and drops ``time_embedder`` entirely. That is what
makes the turbo LoRA's 51 adaLN modules inapplicable: their ``lora_A`` is ``[16, 2688]`` and the
pruned model has no 2688-d vector to feed it.

The delta they encode is a function of the timestep alone,

    dM_b(t) = B_b @ (A_b @ silu(temb(t)))

so it can be added straight into the precomputed modulation cache **if** we can evaluate
``temb(t)``. That needs only ``time_embedder.proj_in`` / ``proj_out`` — two tensors, ~63 MB of a
66 GB release. safetensors puts a JSON header at the front of the file with every tensor's byte
range, so a couple of HTTP range requests are enough.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

REPO = "MiniMaxAI/MiniMax-H3"
SUBDIR = "FL2VA/transformer"
WANTED = ("time_embedder.proj_in.weight", "time_embedder.proj_in.bias",
          "time_embedder.proj_out.weight", "time_embedder.proj_out.bias")


def _url(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_url

    return hf_hub_url(repo, filename=filename)


def _range(session: requests.Session, url: str, start: int, end: int) -> bytes:
    r = session.get(url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=120)
    r.raise_for_status()
    return r.content


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="safetensors file to write")
    ap.add_argument("--repo", default=REPO)
    args = ap.parse_args()

    # HF_HOME is deliberately NOT invented here. Only the index JSON goes through the hub cache
    # (the shards are read by range request and never cached), but a guessed cache root is how a
    # 60 MB fetch quietly re-downloads gigabytes into ~/.cache on somebody else's machine. Set
    # HF_HOME in the environment if you want the index to land somewhere specific.
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import save_file

    index = json.loads(Path(hf_hub_download(args.repo, f"{SUBDIR}/model.safetensors.index.json")).read_text())
    weight_map = index["weight_map"]
    missing = [k for k in WANTED if k not in weight_map]
    if missing:
        print("not in index:", missing)
        print("candidates:", [k for k in weight_map if "time" in k][:20])
        return 1

    session = requests.Session()
    tensors = {}
    for shard in sorted({weight_map[k] for k in WANTED}):
        url = _url(args.repo, f"{SUBDIR}/{shard}")
        head = _range(session, url, 0, 8)
        header_len = int.from_bytes(head, "little")
        header = json.loads(_range(session, url, 8, 8 + header_len))
        base = 8 + header_len
        for key in WANTED:
            if weight_map[key] != shard:
                continue
            entry = header[key]
            a, b = entry["data_offsets"]
            blob = _range(session, url, base + a, base + b)
            import numpy as np

            # Always land float32. The release is F32 today, but a bf16 re-upload read as raw
            # uint16 would sail through save_file() and then multiply as integers inside
            # `lora.silu_temb` — garbage that looks like a working file. The shift-into-float32
            # widening below is exact, so the output is one dtype whatever upstream publishes.
            kind = entry["dtype"]
            if kind == "BF16":
                bits = np.frombuffer(blob, dtype=np.uint16).astype(np.uint32) << 16
                arr = bits.view(np.float32).reshape(entry["shape"])
            elif kind in ("F32", "F16"):
                arr = np.frombuffer(blob, dtype={"F32": np.float32, "F16": np.float16}[kind])
                arr = arr.reshape(entry["shape"]).astype(np.float32)
            else:
                raise SystemExit(f"{key}: unhandled dtype {kind!r} — refusing to guess")
            tensors[key] = np.ascontiguousarray(arr, dtype=np.float32)
            print(f"{key}: {entry['dtype']} {entry['shape']} from {shard} "
                  f"({(b - a) / 1e6:.1f} MB)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out), metadata={"source": f"{args.repo}/{SUBDIR}", "note": "time_embedder only"})
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
