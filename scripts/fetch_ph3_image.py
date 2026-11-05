#!/usr/bin/env python3
"""Fetch P-H3-Image (Pruna) from its publisher and lay it out for the MLX runner.

Nothing here is re-hosted. Every file comes from ``PrunaAI`` at a pinned revision, is checked for
size and sha256, and lands beside the upstream LICENSE / LICENSE-APACHE / NOTICE. The only thing we
change is the image decoder's tensor layout (Diffusers names -> our fused blocks), which the
output's safetensors metadata says (``modification_notice``) and whose own sha is recorded.

    python scripts/fetch_ph3_image.py --dir MODELS/p-h3-image --accepted [--with-upsampler] [--keep-source]

The adapter that turns MiniMax-H3's FL2VA transformer into P-H3-Image is a separate 2 GB file from
Pruna's ComfyUI repo; it is applied at load time and is not merged into anything we write.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

DIFFUSERS_REPO = "PrunaAI/P-H3-Image"
DIFFUSERS_REVISION = "7f98acef503cf2bcfba821a78cb1cf1a9928b2df"
COMFY_REPO = "PrunaAI/P-H3-Image-Comfy"
COMFY_REVISION = "44a0fb73f299da1bb715bdb5ff59ce3fd3c6f543"

LORA_FILE = "loras/p-h3-image_lora_bf16.safetensors"
LORA_BYTES = 1_956_195_360
LORA_SHA256 = "0e9013ae2873e3415dcb7bb59a9677293bdbdb6f663e138a79bb9bdced00ea61"

VAE_SOURCE = "vae/diffusion_pytorch_model.safetensors"
DECODER_NAME = "decoder.safetensors"
# The converted decoder is deterministic (renamed tensors, q/k/v fused per head and the SwiGLU halves swapped;
# values unchanged). Verified by the digest of its TENSORS (tensors_sha256), which does not depend on how a given
# mlx lays out the file header. Measured 5 Nov 2026 on two independent conversions (bench + in-app install), whose
# whole-file sha256 was also identical: DECODER_FILE_SHA256 (mlx 0.32.x), kept for the record.
DECODER_TENSORS_SHA256 = "bc1d55c887b72dbe85caa941bc3e74c419617e5708f965d4291b300e605e3e4b"
DECODER_FILE_SHA256 = "0415dda9059a13d57b6b482b52cbe804f8ebed7c9786fae7f65562a2da7c5c94"
MODIFICATION_NOTICE = (
    "Modified by Phosphene: the image decoder tensors of PrunaAI/P-H3-Image (vae/) were renamed and "
    "re-fused (to_q/to_k/to_v -> to_qkv, SwiGLU halves swapped) for the MLX runner. Values unchanged. "
    "MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright (c) 2026 "
    "MiniMax. All Rights Reserved."
)

# path -> (bytes, sha256 or None for small text files), at DIFFUSERS_REVISION
CORE_FILES = {
    "LICENSE": (17604, None),
    "LICENSE-APACHE": (11544, None),
    "NOTICE": (696, None),
    "modular_model_index.json": (3525, None),
    "processor/chat_template.jinja": (5292, None),
    "processor/processor_config.json": (1220, None),
    "processor/tokenizer.json": (11422650, "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506"),
    "processor/tokenizer_config.json": (734, None),
    "scheduler/scheduler_config.json": (132, None),
    "text_connector/config.json": (208, None),
    "text_connector/diffusion_pytorch_model.safetensors": (262308352, "4c4ad8cbf37db8a2da22cdd9905e566e259cf3477ea242f6aee1cf3fb695f070"),
    "text_encoder/config.json": (1696, None),
    "text_encoder/generation_config.json": (242, None),
    "text_encoder/model-00001-of-00002.safetensors": (4990822120, "be8454bc80eb7b91da25273ab71e0f4a627cf1d5588e91da2e052297da8879cd"),
    "text_encoder/model-00002-of-00002.safetensors": (3884897208, "d3a4781fcec3eeaf8ba5adef647df3d21eed238dfc6be5e4aa96cf941cbe61b1"),
    "text_encoder/model.safetensors.index.json": (64778, None),
    "tokenizer/chat_template.jinja": (5292, None),
    "tokenizer/tokenizer.json": (11422650, "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506"),
    "tokenizer/tokenizer_config.json": (734, None),
    "vae/config.json": (1971, None),
    "vae/diffusion_pytorch_model.safetensors": (5559055392, "264964208bea153ca6366570b66d793c75b0fc64716fb6515a41b54e3dce6bf1"),
}

UPSAMPLER_FILES = {
    "prompt_upsampler/config.json": (2883, None),
    "prompt_upsampler/generation_config.json": (116, None),
    "prompt_upsampler/model-00001-of-00004.safetensors": (4942706120, "5af4cf75e63a8d7cb96d03d25add224b243380f04a8c83ef4b47c40a50e36962"),
    "prompt_upsampler/model-00002-of-00004.safetensors": (4987757928, "ffb3821101877085712f6d51510bf0796d7b5bd50d3804f299472ce94b449302"),
    "prompt_upsampler/model-00003-of-00004.safetensors": (4954810240, "078775ad9f57fac960fd5bf53754b9e4b07dd403e60b15de15998e876926a312"),
    "prompt_upsampler/model-00004-of-00004.safetensors": (3934446832, "24196f3a3680e0d72270fa6c7d27379690d5887fa4af5bcd1f5beecdd843fbcf"),
    "prompt_upsampler/model.safetensors.index.json": (69253, None),
    "prompt_upsampler_processor/chat_template.jinja": (7756, None),
    "prompt_upsampler_processor/processor_config.json": (1220, None),
    "prompt_upsampler_processor/tokenizer.json": (19989339, "a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f"),
    "prompt_upsampler_processor/tokenizer_config.json": (1507, None),
}

LICENCE_NOTE = (
    "P-H3-Image is built on MiniMax H3 and is released under the MiniMax H3 Community License. It "
    "excludes the European Union, the United Kingdom, the Republic of Korea and the United States."
)


def url(repo: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{path}"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def fetch(repo: str, revision: str, rel: str, dest: Path, size: int, sha: str | None) -> Path:
    """Download one pinned file (resumable) and verify it. Returns the path.

    A stalled connection times out (60 s without data) instead of hanging the install. A `.part`
    that is already complete goes straight to verification; one that is too big, or a server that
    ignores the Range request, starts the file over instead of appending to it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == size and (sha is None or sha256_of(dest) == sha):
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    if have < size:
        req = urllib.request.Request(url(repo, revision, rel))
        if have:
            req.add_header("Range", f"bytes={have}-")
        with urllib.request.urlopen(req, timeout=60) as resp:
            resume = bool(have) and getattr(resp, "status", 200) == 206
            with open(part, "ab" if resume else "wb") as out:
                while True:
                    block = resp.read(1 << 22)
                    if not block:
                        break
                    out.write(block)
    got = part.stat().st_size
    if got != size:
        if got > size:
            part.unlink()
        raise RuntimeError(f"{rel}: expected {size} bytes, got {got} (run again to "
                           + ("start it over)" if got > size else "resume)"))
    if sha is not None and sha256_of(part) != sha:
        part.unlink()
        raise RuntimeError(f"{rel}: sha256 mismatch (file removed, run again)")
    part.rename(dest)
    return dest


def _decoder_digest(path: Path) -> str:
    return tensors_sha256(path)


def tensors_sha256(path: Path) -> str:
    """Digest of a safetensors file's TENSORS (name, dtype, shape, raw bytes, in name order), not of
    its header layout, so a different mlx writing the same values gives the same digest."""
    import struct
    h = hashlib.sha256()
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        for name in sorted(k for k in header if k != "__metadata__"):
            t = header[name]
            h.update(f"{name}|{t['dtype']}|{t['shape']}\n".encode())
            start, end = t["data_offsets"]
            f.seek(base + start)
            left = end - start
            while left:
                block = f.read(min(left, 1 << 22))
                if not block:
                    raise RuntimeError(f"{path.name}: truncated tensor {name}")
                h.update(block)
                left -= len(block)
    return h.hexdigest()


def convert_decoder(source: Path, config_path: Path, dest: Path) -> str:
    """vae/diffusion_pytorch_model.safetensors -> decoder.safetensors in our layout."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import mlx.core as mx

    from minimax_h3_mlx.ph3_image_vae import convert_decoder_state, image_vae_config

    cfg = image_vae_config(json.loads(config_path.read_text()))
    tensors = mx.load(str(source))
    out = convert_decoder_state(tensors, cfg)
    mx.eval(list(out.values()))
    # mx.save_safetensors appends ".safetensors" to a name without it, so name the temp file fully.
    tmp = dest.with_name(dest.stem + ".tmp.safetensors")
    mx.save_safetensors(str(tmp), out, metadata={
        "modification_notice": MODIFICATION_NOTICE,
        "source_repo": DIFFUSERS_REPO, "source_revision": DIFFUSERS_REVISION,
        "source_file": VAE_SOURCE,
    })
    tmp.rename(dest)
    return sha256_of(dest)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="output folder (created)")
    ap.add_argument("--accepted", action="store_true", help="the user accepted the MiniMax H3 Community License")
    ap.add_argument("--with-upsampler", action="store_true", help="also fetch the 19 GB prompt rewriter")
    ap.add_argument("--keep-source", action="store_true", help="keep the 5.6 GB original VAE after converting")
    ap.add_argument("--lora-only", action="store_true", help="only the adapter (2 GB)")
    ap.add_argument("--list", action="store_true", help="print the pins and exit")
    args = ap.parse_args(argv)

    if args.list:
        print(json.dumps({"diffusers": [DIFFUSERS_REPO, DIFFUSERS_REVISION], "comfy": [COMFY_REPO, COMFY_REVISION],
                          "lora": [LORA_FILE, LORA_BYTES, LORA_SHA256], "files": CORE_FILES}, indent=1))
        return 0
    if not args.accepted:
        print("Refusing: " + LICENCE_NOTE + " Pass --accepted once the user has accepted the licence.", file=sys.stderr)
        return 2

    root = Path(args.dir)
    root.mkdir(parents=True, exist_ok=True)
    fetch(COMFY_REPO, COMFY_REVISION, LORA_FILE, root / LORA_FILE, LORA_BYTES, LORA_SHA256)
    print("adapter ok", flush=True)
    if args.lora_only:
        return 0

    decoder = root / "vae" / DECODER_NAME
    decoder_ok = decoder.exists() and _decoder_digest(decoder) == DECODER_TENSORS_SHA256
    if decoder.exists() and not decoder_ok:
        print("decoder on disk does not match its digest; building it again", flush=True)
        decoder.unlink()
    files = dict(CORE_FILES)
    if decoder_ok and not args.keep_source:
        files.pop(VAE_SOURCE, None)   # already converted: never fetch the 5.6 GB original again
    if args.with_upsampler:
        files.update(UPSAMPLER_FILES)
    for rel, (size, sha) in files.items():
        fetch(DIFFUSERS_REPO, DIFFUSERS_REVISION, rel, root / rel, size, sha)
        print("ok", rel, flush=True)

    if not decoder_ok:
        print("decoder converting (renamed and re-fused tensors, values unchanged)", flush=True)
        file_sha = convert_decoder(root / VAE_SOURCE, root / "vae" / "config.json", decoder)
        digest = _decoder_digest(decoder)
        if digest != DECODER_TENSORS_SHA256:
            decoder.unlink()
            print(f"error: the converted decoder's tensors digest {digest[:12]} is not the expected "
                  f"{DECODER_TENSORS_SHA256[:12]}; removed it. Report this line.", file=sys.stderr, flush=True)
            return 3
        (root / "vae" / (DECODER_NAME + ".sha256")).write_text(file_sha + "\n")
        print("decoder converted", digest[:12], "verified", flush=True)
    else:
        print("decoder ok", DECODER_TENSORS_SHA256[:12], flush=True)
    if not args.keep_source and (root / VAE_SOURCE).exists():
        (root / VAE_SOURCE).unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
