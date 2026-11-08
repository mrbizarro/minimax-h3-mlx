"""Decode-only video VAE whose ViT decoder streams one transformer block at a time.

Once the DiT streams, the video decode is the largest pass of a low-memory render: 4.5 GiB of
the 4.85 GiB compact VAE is 36 fp16 decoder blocks (128 MiB each) held for the whole decode.
Streamed, each decoder call reads the blocks in order from the file's byte ranges, the next
block on a worker thread while the current one computes, so about two blocks are resident.

The decode makes one decoder call per temporal chunk and tile batch, so every call rereads the
blocks; with the default tile batch that is a handful of calls per clip, and the read of block
i+1 overlaps block i's compute. The CNN encoder (only used to encode keyframes, never by a
decode) is not loaded at all.

Same tensors, same modules, same operation order as the resident loader; only evaluation
boundaries move (each block is evaluated before the next one loads), so decoded pixels are
byte-identical.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .tensor_stream import tensor_index

BLOCKS = "decoder.transformer_blocks."


class DecoderBlockStream:
    """Byte ranges of every decoder block, plus a one-block read-ahead (numpy only, no MLX)."""

    def __init__(self, path: Path, num_blocks: int):
        index = tensor_index([path])
        self.n = num_blocks
        self.blocks = {}
        for i in range(num_blocks):
            prefix = f"{BLOCKS}{i}."
            entries = {k[len(prefix):]: v for k, v in index.items() if k.startswith(prefix)}
            if not entries:
                raise KeyError(f"no tensors for decoder block {i} in {path}")
            if any(len(v.shape) == 5 for v in entries.values()):
                raise ValueError("a decoder block holds a 5-D (conv) tensor; it cannot stream as is")
            self.blocks[i] = entries
        self.block_bytes = max(sum(v.nbytes for v in e.values()) for e in self.blocks.values())
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vae-read")
        self._ahead: dict = {}
        self.stats = {"block_loads": 0, "bytes_read": 0, "read_ahead_hits": 0, "wait_seconds": 0.0}

    def _read(self, i: int) -> dict:
        return {k: v.numpy() for k, v in self.blocks[i].items()}

    def prefetch(self, i: int) -> None:
        i %= self.n
        if i not in self._ahead and self._pool is not None:
            self._ahead[i] = self._pool.submit(self._read, i)

    def weights(self, i: int) -> dict:
        fut = self._ahead.pop(i, None)
        started = time.perf_counter()
        if fut is not None:
            host = fut.result()
            self.stats["read_ahead_hits"] += 1
        else:
            host = self._read(i)
        self.stats["wait_seconds"] += time.perf_counter() - started
        # The next block (wrapping into the next decoder call) starts arriving now.
        self.prefetch(i + 1)
        out = {}
        for key, array in host.items():
            value = mx.array(array)
            if self.blocks[i][key].dtype == "BF16":
                value = value.view(mx.bfloat16)
            out[key] = value
        self.stats["block_loads"] += 1
        self.stats["bytes_read"] += sum(v.nbytes for v in self.blocks[i].values())
        return out

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        self._ahead.clear()

    def summary(self) -> dict:
        return {
            "blocks": self.n,
            "block_mib": round(self.block_bytes / 1024**2, 1),
            "block_loads": self.stats["block_loads"],
            "gib_read": round(self.stats["bytes_read"] / 1024**3, 2),
            "read_ahead_hits": self.stats["read_ahead_hits"],
            "wait_seconds": round(self.stats["wait_seconds"], 3),
        }


class _StreamedViTBlock(nn.Module):
    """Load block i, run the unchanged ViTBlock forward, evaluate, and let the weights go."""

    def __init__(self, stream: DecoderBlockStream, index: int, factory):
        super().__init__()
        self._stream = stream
        self._index = index
        self._factory = factory

    _streamed = True  # the decoder must not wrap this block in mx.compile: its weights change per call

    def __call__(self, x: mx.array, rotary=None, compute=None) -> mx.array:
        # Finish everything this block reads before its weights arrive, bounding live memory.
        mx.eval([x] + ([a for a in rotary if isinstance(a, mx.array)] if rotary else []))
        module = self._factory()
        weights = self._stream.weights(self._index)
        expected = {k for k, _ in tree_flatten(module.parameters())}
        if expected != set(weights):
            raise KeyError(f"decoder block {self._index}: tensors {sorted(set(weights) ^ expected)[:4]}")
        module.update(tree_unflatten(list(weights.items())))
        try:
            out = module(x, rotary, compute)
            mx.eval(out)
            return out
        finally:
            del module, weights


def load_compact_video_vae_streamed(path: str | Path):
    """Decode-only ``VideoVAE`` from the compact export: decoder blocks stream, encoder absent.

    Returns ``(model, stream)``; call ``stream.close()`` when the decode is done.
    """
    from .load import load_compact_video_vae_config
    from .video_vae import ViTBlock, VideoVAE

    path = Path(path)
    config = load_compact_video_vae_config(path)
    model = VideoVAE(config)
    stream = DecoderBlockStream(path, config.decoder_num_layers)
    factory = partial(ViTBlock, config.decoder_dim, config.decoder_num_attention_heads,
                      config.decoder_attention_head_dim, config.decoder_ffn_mult,
                      config.decoder_norm_eps)
    # Replace before any evaluation, so the blocks' (lazy) initial weights are never made.
    model.decoder.transformer_blocks = [
        _StreamedViTBlock(stream, i, factory) for i in range(config.decoder_num_layers)
    ]
    model.encoder = nn.Module()          # a decode never encodes
    expected = {key for key, _ in tree_flatten(model.parameters())}
    index = tensor_index([path])
    weights = {}
    for key, entry in index.items():
        if key.startswith(BLOCKS) or key.startswith("encoder.") or key in (
                "decoder.mask_token", "latents_mean", "latents_std"):
            continue
        if key not in expected:
            raise KeyError(f"unexpected tensor {key} in {path}")
        value = entry.mlx()
        if value.ndim == 5:
            with mx.stream(mx.cpu):
                value = mx.contiguous(value.transpose(0, 2, 3, 4, 1))
        weights[key] = value
    missing = sorted(expected - weights.keys())
    if missing:
        raise KeyError(f"streamed video VAE: {len(missing)} tensors missing, e.g. {missing[:4]}")
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    stream.prefetch(0)
    return model, stream


def load_compact_video_vae_encoder_only(path: str | Path):
    """Encode-only ``VideoVAE`` (keyframes, references, carried frames): the CNN encoder and its
    quant conv, without the 4.5 GiB ViT decoder an encode never runs. Same encoder tensors and
    layout as the resident loader, so encoded rows are byte-identical."""
    from .load import load_compact_video_vae_config
    from .video_vae import VideoVAE

    path = Path(path)
    model = VideoVAE(load_compact_video_vae_config(path))
    model.decoder = nn.Module()          # replaced before any evaluation: never materialized
    expected = {key for key, _ in tree_flatten(model.parameters())}
    weights = {}
    for key, entry in tensor_index([path]).items():
        if key.startswith("decoder.") or key in ("latents_mean", "latents_std"):
            continue
        if key not in expected:
            raise KeyError(f"unexpected tensor {key} in {path}")
        value = entry.mlx()
        if value.ndim == 5:
            with mx.stream(mx.cpu):
                value = mx.contiguous(value.transpose(0, 2, 3, 4, 1))
        weights[key] = value
    missing = sorted(expected - weights.keys())
    if missing:
        raise KeyError(f"encode-only video VAE: {len(missing)} tensors missing, e.g. {missing[:4]}")
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model
