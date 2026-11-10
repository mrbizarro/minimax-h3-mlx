#!/usr/bin/env python3
"""Fetch MiniMax-H3-Image from its publisher and build the Mac pack, shard by shard.

Nothing is re-hosted. Every file comes from ``MiniMaxAI/MiniMax-H3-Image`` at a pinned revision,
is checked for size and sha256, and sits beside the upstream LICENSE and the NOTICE the licence
asks for. The only thing we change is the DiT's storage (see MODIFICATION_NOTICE):

* the 50 per-block AdaLN projections (26 GB) are replaced by the table their output makes for
  the release's 6 timesteps (+ the 0.999 reference level), computed from the UNQUANTISED weights;
* the remaining linears are quantised to 8 bit (group 64) on the CPU stream, bit-deterministic.

Each of the 12 source shards is downloaded, converted, and (unless ``--keep-source``) deleted before
the next one comes, so the disk needs one shard (<= 5.9 GB) plus the growing pack (~20 GB).
Re-running resumes after the last finished shard.

    python scripts/fetch_h3_image.py --dir MODELS/h3-image --accepted [--keep-source]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch_ph3_image as base  # noqa: E402  (fetch / sha256_of helpers)

REPO = "MiniMaxAI/MiniMax-H3-Image"
REVISION = "6534ea1c979a5f8cb00c2bd1025d5ca4ee8d8901"
TEXT_ENCODER_REPO = "MiniMaxAI/MiniMax-H3"
TEXT_ENCODER_REVISION = "0919fb684102ef115126612f3fbb84d1027e7317"

BITS, GROUP = 8, 64
MODIFICATION_NOTICE = (
    "Modified by Phosphene: the DiT of MiniMaxAI/MiniMax-H3-Image was quantised to 8 bit (group 64) and its "
    "50 per-block AdaLN projections were replaced by their precomputed output for the release's 6 timesteps "
    "(adaln_cache.safetensors). MiniMax-H3-Image is licensed under the MiniMax-H3-Image Community License "
    "Agreement, Copyright (c) 2026 MiniMax. All Rights Reserved."
)
NOTICE_TEXT = (
    "MiniMax-H3-Image is licensed under the MiniMax-H3-Image Community License Agreement, "
    "Copyright (c) 2026 MiniMax. All Rights Reserved.\n"
)
LICENCE_NOTE = (
    "MiniMax-H3-Image is released under the MiniMax-H3-Image Community License. It excludes the European Union, "
    "the United Kingdom, the Republic of Korea and the United States, and needs MiniMax's written authorisation "
    "above 1 million USD of annualised revenue or for a hosted service."
)

# path -> (bytes, sha256 or None for small text files), at REVISION
CORE_FILES = {
    "LICENSE": (18200, None),
    "model_index.json": (891, None),
    "pdd_config.json": (174, None),
    "pdd_fused_heads.safetensors": (12388808, "c07093a098e3bef4d43cd29c33e239a0c45424c5ff40dfcc864df00b8d664738"),
    "vae/config.json": (1826, None),
    "vae/diffusion_pytorch_model.safetensors": (2336964672, "a7c8bb1e181865e7cf87c29d3d6bc728d944ce4862b92d0e1cbdcc36b3b180b2"),
    "transformer/config.json": (630, None),
    "transformer/model.safetensors.index.json": (38049, None),
}

# transformer shard -> (bytes, sha256)
SHARDS = {
    "model-00001-of-00012.safetensors": (5441123104, "200667b632a6004bb8442008c4a8e975d64a7dbbc3d5569ed482650c3d61ff7b"),
    "model-00002-of-00012.safetensors": (5839159856, "2227018562b2d92db391bce2fe641ea7f887585b581d6a73092a8fcfc59ef348"),
    "model-00003-of-00012.safetensors": (5472861152, "165469f7dccbb9fb6cbdc7a88e89d2a0220697af4da7ee40a96b369ccfa5abc9"),
    "model-00004-of-00012.safetensors": (5472860360, "2bf7dff80d49203cee29834ce6bd86077b09c4b73174d1c4f696f8fecab83ed3"),
    "model-00005-of-00012.safetensors": (5839159904, "dcc7c79d46fa1ca910396fa001657985057db491af1bfadeda4cf3a82da329f2"),
    "model-00006-of-00012.safetensors": (5472861168, "12c76e39604f04d11d282172719bad6ad374e98479b8892f4ea0c9d2695b3086"),
    "model-00007-of-00012.safetensors": (5472860360, "59ed575a374c0304da09beb234225c17b5ecc1e491b76f72455f0b3b7a1fbbff"),
    "model-00008-of-00012.safetensors": (5839159904, "b71af5f2f42fee2530b4154caba6ce595ce98072d831090a6b63a546420c0072"),
    "model-00009-of-00012.safetensors": (5472861168, "a2764f44f306f6eecfd3d52567bcebe64ddfafb0a2fbdaf089efc1e74cbe82f8"),
    "model-00010-of-00012.safetensors": (5472860360, "0e82a47b5fd0498674718c7a0b3830d042d349f6b5fe16cd2b35cccca5063365"),
    "model-00011-of-00012.safetensors": (5839159904, "d728554ce210b10062e70db51080d2f6d6e41c6d0fa033dc77b021db20745d2b"),
    "model-00012-of-00012.safetensors": (4644160496, "8d9e1c67066a5fa646f2e87f0868f73cb50be66cc1739e51eb12791fd9123b83"),
}
SHARD_FILE = "transformer/{name}"


def shard_names() -> list[str]:
    return sorted(SHARDS)


def schedule_timesteps(pdd_config: dict) -> list[float]:
    """Distinct timesteps the DiT sees: ``1 - sigma`` per forward (float32) + the 0.999 reference level."""
    from minimax_h3_mlx.h3_image import reference_timesteps

    return reference_timesteps(tuple(pdd_config["sigmas"]))


def convert_shard(src: Path, out: Path, part: Path, temb, config, timesteps, quant, verbose=True) -> dict:
    """One source shard -> ``out`` (quantised tensors) + ``part`` (AdaLN cache pieces of its blocks)."""
    import mlx.core as mx
    import mlx.nn as nn

    from minimax_h3_mlx.dit import AdaLayerNormModulation
    from minimax_h3_mlx.load import is_fp32_key

    tensors = mx.load(str(src))
    kept, cache_parts, adaln_in = {}, {}, {}
    with mx.stream(mx.cpu):
        for key, t in sorted(tensors.items()):
            if key == "rope.inv_freq":
                continue
            if ".adaln_proj.linear." in key and key.startswith("blocks."):
                adaln_in[key] = t
                continue
            base_name = key[: -len(".weight")] if key.endswith(".weight") else None
            bits = quant.bits_for(base_name) if base_name else None
            if bits and t.ndim == 2 and t.shape[1] % GROUP == 0:
                w, sc, bi = mx.quantize(t, group_size=GROUP, bits=bits)
                kept[base_name + ".weight"] = w
                kept[base_name + ".scales"] = sc.astype(mx.bfloat16)
                kept[base_name + ".biases"] = bi.astype(mx.bfloat16)
            else:
                kept[key] = t.astype(mx.float32) if is_fp32_key(key) else t
            mx.eval(list(kept.values())[-1])
    # AdaLN: out = proj(silu(temb)) for every block of this shard, from the unquantised weights.
    blocks = sorted({int(k.split(".")[1]) for k in adaln_in})
    for i in blocks:
        mod = AdaLayerNormModulation(config)
        mod.linear.weight = adaln_in[f"blocks.{i}.adaln_proj.linear.weight"]
        mod.linear.bias = adaln_in[f"blocks.{i}.adaln_proj.linear.bias"]
        tables = mod(temb)
        for j, tab in enumerate(tables):
            cache_parts[f"blocks.{i}.{j}"] = tab.astype(mx.bfloat16)
        mx.eval(list(cache_parts.values()))
    mx.save_safetensors(str(out), kept, metadata={"modification_notice": MODIFICATION_NOTICE, "format": "mlx"})
    if cache_parts:
        mx.save_safetensors(str(part), cache_parts)
    return {"tensors": len(kept), "adaln_blocks": blocks, "bytes": out.stat().st_size}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, type=Path, help="output folder (the runner root)")
    ap.add_argument("--accepted", action="store_true",
                    help="the user accepted the MiniMax-H3-Image Community License (see LICENSE)")
    ap.add_argument("--keep-source", action="store_true", help="keep the downloaded source shards")
    ap.add_argument("--shards", default=None, help="only these shard numbers, e.g. 1,2 (testing)")
    ap.add_argument("--no-convert", action="store_true", help="fetch the small files only")
    args = ap.parse_args(argv)
    if not args.accepted:
        print(LICENCE_NOTE, file=sys.stderr)
        print("Pass --accepted once you have read the LICENSE that sits beside the files.", file=sys.stderr)
        return 2

    root = args.dir
    for rel, (size, sha) in CORE_FILES.items():
        base.fetch(REPO, REVISION, rel, root / rel if not rel.startswith("transformer/") else root / "source" / rel, size, sha)
    (root / "NOTICE").write_text(NOTICE_TEXT)
    if args.no_convert:
        print("small files fetched")
        return 0

    import mlx.core as mx

    from minimax_h3_mlx.adaln import ModulationCache, save_modulation_cache
    from minimax_h3_mlx.config import DiTConfig
    from minimax_h3_mlx.dit import TimestepEmbedder, timestep_embedding
    from minimax_h3_mlx.quantize import QuantConfig

    pdd = json.loads((root / "pdd_config.json").read_text())
    timesteps = schedule_timesteps(pdd)
    raw = json.loads((root / "source" / "transformer" / "config.json").read_text())
    config = DiTConfig.from_dict(raw)
    quant = QuantConfig(bits=BITS, group_size=GROUP, quantize_adaln=False)
    dit_dir = root / "dit"
    dit_dir.mkdir(parents=True, exist_ok=True)
    state_path = dit_dir / "convert_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"done": {}}

    names = shard_names()
    wanted = set(int(x) for x in args.shards.split(",")) if args.shards else None
    temb = None
    for n, name in enumerate(names, 1):
        if wanted and n not in wanted:
            continue
        if name in state["done"] and (dit_dir / name).exists():
            continue
        size, sha = SHARDS[name]
        rel = SHARD_FILE.format(name=name)
        print(f"[{n}/{len(names)}] fetching {name} ({size / 1e9:.2f} GB)", flush=True)
        src = base.fetch(REPO, REVISION, rel, root / "source" / rel, size, sha)
        if temb is None:
            # shard 1 carries time_embedder (checked against the index below); every AdaLN block needs it
            idx = json.loads((root / "source" / "transformer" / "model.safetensors.index.json").read_text())["weight_map"]
            holder = idx["time_embedder.proj_in.weight"]
            emb_src = src if holder == name else base.fetch(REPO, REVISION, SHARD_FILE.format(name=holder),
                                                           root / "source" / SHARD_FILE.format(name=holder),
                                                           SHARDS[holder][0], SHARDS[holder][1])
            te = TimestepEmbedder(config)
            w = mx.load(str(emb_src))
            te.proj_in.weight, te.proj_in.bias = w["time_embedder.proj_in.weight"].astype(mx.float32), w["time_embedder.proj_in.bias"].astype(mx.float32)
            te.proj_out.weight, te.proj_out.bias = w["time_embedder.proj_out.weight"].astype(mx.float32), w["time_embedder.proj_out.bias"].astype(mx.float32)
            ts = mx.array(timesteps, dtype=mx.float32)
            temb = te(timestep_embedding(ts, config.timestep_input_dim))
            mx.eval(temb)
        info = convert_shard(src, dit_dir / name, dit_dir / f"adaln_part-{n:02d}.safetensors", temb, config, timesteps, quant)
        state["done"][name] = info
        state_path.write_text(json.dumps(state, indent=1))
        if not args.keep_source:
            src.unlink()
        print(f"[{n}/{len(names)}] converted {name}: {info['tensors']} tensors, {info['bytes'] / 1e9:.2f} GB, "
              f"adaln blocks {info['adaln_blocks'][:1]}..{info['adaln_blocks'][-1:]}", flush=True)

    if all(n in state["done"] for n in names):
        finish(root, dit_dir, config, timesteps, names, state)
    return 0


def finish(root: Path, dit_dir: Path, config, timesteps, names, state) -> None:
    """Merge the AdaLN pieces, write config / quant recipe / index, and verify the pack loads."""
    import mlx.core as mx

    from minimax_h3_mlx.adaln import ModulationCache, save_modulation_cache

    tables = {}
    for part in sorted(dit_dir.glob("adaln_part-*.safetensors")):
        tables.update(mx.load(str(part)))
    cache = ModulationCache([tuple(tables[f"blocks.{i}.{j}"] for j in range(6)) for i in range(config.num_layers)],
                            mx.array(timesteps, dtype=mx.float32))
    save_modulation_cache(cache, dit_dir / "adaln_cache.safetensors")
    for part in dit_dir.glob("adaln_part-*.safetensors"):
        part.unlink()
    weight_map = {}
    for name in names:
        for k in mx.load(str(dit_dir / name)):
            weight_map[k] = name
    (dit_dir / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_shards": len(names)}, "weight_map": weight_map}, indent=1))
    cfg = dataclasses.replace(config, audio_enabled=False, adaln_precomputed=True)
    (dit_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=1))
    (dit_dir / "quant_config.json").write_text(json.dumps(
        {"bits": BITS, "group_size": GROUP, "quantize_adaln": False, "writer": "fetch_h3_image",
         "source": f"{REPO}@{REVISION}"}, indent=1))
    manifest = {n: base.sha256_of(dit_dir / n) for n in names}
    manifest["adaln_cache.safetensors"] = base.sha256_of(dit_dir / "adaln_cache.safetensors")
    (dit_dir / "ASSET_SHA256.json").write_text(json.dumps(manifest, indent=1))
    (root / "vae").mkdir(exist_ok=True)
    from minimax_h3_mlx.load import load_dit

    load_dit(dit_dir, verbose=False)
    print(f"PACK OK -> {dit_dir}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
