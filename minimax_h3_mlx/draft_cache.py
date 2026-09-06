"""Bounded on-disk caches for iterative draft renders.

The cache is deliberately opt-in and runner-facing.  Dense/HQ renders never construct it.  Text
conditioning is content-addressed by the resulting embedding SHA-256, with a small request index
so a repeat can find that SHA without loading the 26 GB encoder.  AdaLN tables and seeded noise are
kept separately because they have different reuse keys and lifetimes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np

from .adaln import ModulationCache


CACHE_VERSION = 1


def _json_key(kind: str, payload: dict) -> str:
    encoded = json.dumps(
        {"version": CACHE_VERSION, "kind": kind, **payload}, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def path_fingerprint(path: str | Path | None) -> dict | None:
    """Cheap identity for large immutable checkpoints; hashing 40+ GB would erase the cache win."""
    if path is None:
        return None
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def image_digest(image) -> str:
    if image is None:
        return ""
    array = np.ascontiguousarray(np.asarray(image))
    hasher = hashlib.sha256()
    hasher.update(str(array.shape).encode())
    hasher.update(array.tobytes())
    return hasher.hexdigest()


class DraftCache:
    def __init__(self, root: str | Path, limit: int = 50):
        self.root = Path(root)
        self.limit = max(1, int(limit))
        for name in ("text", "text_requests", "keyframe", "adaln", "noise"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def text_request_key(
        *, prompt: str, seed: int, width: int, height: int, first_frame, compact_root, text_config
    ) -> str:
        return _json_key(
            "text_request",
            {
                "prompt": prompt,
                "seed": int(seed),
                "width": int(width),
                "height": int(height),
                "first_frame_sha256": image_digest(first_frame),
                "encoder": path_fingerprint(Path(compact_root) / "text_encoder.safetensors"),
                "text_config": path_fingerprint(text_config),
            },
        )

    @staticmethod
    def adaln_key(*, timestep_table: np.ndarray, dit, lora, lora_adaln) -> str:
        table = np.ascontiguousarray(timestep_table, dtype=np.float32)
        # `lora` is one (path, scale) pair or a list of them. A single adapter
        # keeps the historical key shape so existing caches stay valid; a stack
        # is keyed on the sorted set, so order of the flags does not matter.
        stack = []
        if lora:
            stack = list(lora) if isinstance(lora, (list, tuple)) and lora and isinstance(lora[0], (list, tuple)) else [lora]
        stack = [(p, float(s) if s is not None else None) for p, s in stack if p is not None]
        fields = {
            "timesteps_sha256": hashlib.sha256(table.tobytes()).hexdigest(),
            "dit": path_fingerprint(dit),
            "lora": path_fingerprint(stack[0][0]) if len(stack) == 1 else None,
            "lora_scale": stack[0][1] if len(stack) == 1 else None,
            "lora_adaln": path_fingerprint(lora_adaln),
        }
        if len(stack) > 1:
            fields["lora_stack"] = sorted(
                ({"lora": path_fingerprint(p), "scale": s} for p, s in stack),
                key=lambda d: d["lora"]["path"])
        return _json_key("adaln", fields)

    @staticmethod
    def noise_key(
        *, adaln_key: str, seed: int, condition_shape, video_shape, audio_shape
    ) -> str:
        return _json_key(
            "noise",
            {
                "adaln_key": adaln_key,
                "seed": int(seed),
                "condition_shape": list(condition_shape) if condition_shape else None,
                "video_shape": list(video_shape),
                "audio_shape": list(audio_shape),
            },
        )

    @staticmethod
    def keyframe_key(*, first_frame, width: int, height: int, compact_root, patch_size) -> str:
        return _json_key(
            "keyframe",
            {
                "first_frame_sha256": image_digest(first_frame),
                "width": int(width),
                "height": int(height),
                "video_vae": path_fingerprint(Path(compact_root) / "video_vae.safetensors"),
                "patch_size": list(patch_size),
            },
        )

    def load_keyframe(self, key: str):
        path = self.root / "keyframe" / f"{key}.npz"
        if not path.is_file():
            return None
        try:
            data = np.load(path, allow_pickle=False)
            rows = np.asarray(data["rows"], dtype=np.float32)
            os.utime(path, None)
            return rows
        except (OSError, KeyError, ValueError):
            path.unlink(missing_ok=True)
            return None

    def store_keyframe(self, key: str, rows) -> None:
        self._atomic_npz(
            self.root / "keyframe" / f"{key}.npz", rows=np.asarray(rows, dtype=np.float32)
        )
        self._prune(self.root / "keyframe", "*.npz")

    def load_text(self, request_key: str):
        request_path = self.root / "text_requests" / f"{request_key}.json"
        if not request_path.is_file():
            return None
        try:
            metadata = json.loads(request_path.read_text())
            digest = metadata["text_embed_sha256"]
            entry_path = self.root / "text" / f"{digest}.npz"
            data = np.load(entry_path, allow_pickle=False)
            embeds = np.asarray(data["embeds"], dtype=np.float32)
            actual = hashlib.sha256(embeds.tobytes()).hexdigest()
            if actual != digest:
                raise ValueError(f"embedding digest mismatch: expected {digest}, got {actual}")
            text_tags = np.asarray(data["text_tags"], dtype=np.int64)
            os.utime(request_path, None)
            os.utime(entry_path, None)
            return embeds, text_tags, digest
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            request_path.unlink(missing_ok=True)
            return None

    def store_text(
        self, request_key: str, embeds: np.ndarray, text_tags: np.ndarray, digest: str
    ) -> None:
        entry_path = self.root / "text" / f"{digest}.npz"
        if not entry_path.exists():
            self._atomic_npz(
                entry_path,
                embeds=np.asarray(embeds, dtype=np.float32),
                text_tags=np.asarray(text_tags, dtype=np.int64),
            )
        request_path = self.root / "text_requests" / f"{request_key}.json"
        self._atomic_json(request_path, {"text_embed_sha256": digest})
        self._prune(self.root / "text_requests", "*.json")
        self._prune_unreferenced_text()

    def load_adaln(self, key: str, dit):
        path = self.root / "adaln" / f"{key}.npz"
        if not path.is_file():
            return None
        try:
            data = np.load(path, allow_pickle=False)
            timesteps = mx.array(np.asarray(data["timesteps"], dtype=np.float32))
            tables = []
            for block in range(len(dit.blocks)):
                tables.append(
                    tuple(
                        mx.array(np.asarray(data[f"b{block}_{part}"], dtype=np.float32)).astype(
                            mx.bfloat16
                        )
                        for part in range(6)
                    )
                )
            cache = ModulationCache(tables, timesteps)
            if "final_lora_delta" in data:
                dit.final_layer.adaln_proj.lora_delta = mx.array(
                    np.asarray(data["final_lora_delta"], dtype=np.float32)
                )
            mx.eval(cache.tables)
            if hasattr(dit.final_layer.adaln_proj, "lora_delta"):
                mx.eval(dit.final_layer.adaln_proj.lora_delta)
            report = json.loads(str(data["report"].item())) if "report" in data else None
            os.utime(path, None)
            return cache, report
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            path.unlink(missing_ok=True)
            return None

    def store_adaln(self, key: str, cache: ModulationCache, dit, report: dict | None) -> None:
        arrays = {"timesteps": np.asarray(cache.timesteps, dtype=np.float32)}
        for block, table in enumerate(cache.tables):
            for part, value in enumerate(table):
                arrays[f"b{block}_{part}"] = np.array(value.astype(mx.float32))
        delta = getattr(dit.final_layer.adaln_proj, "lora_delta", None)
        if isinstance(delta, mx.array):
            arrays["final_lora_delta"] = np.array(delta.astype(mx.float32))
        arrays["report"] = np.array(json.dumps(report))
        self._atomic_npz(self.root / "adaln" / f"{key}.npz", **arrays)
        self._prune(self.root / "adaln", "*.npz")

    def load_noise(self, key: str):
        path = self.root / "noise" / f"{key}.npz"
        if not path.is_file():
            return None
        try:
            data = np.load(path, allow_pickle=False)
            out = {
                "video": np.asarray(data["video"], dtype=np.float32),
                "audio": np.asarray(data["audio"], dtype=np.float32),
                "condition": (
                    np.asarray(data["condition"], dtype=np.float32)
                    if "condition" in data
                    else None
                ),
            }
            os.utime(path, None)
            return out
        except (OSError, KeyError, ValueError):
            path.unlink(missing_ok=True)
            return None

    def store_noise(self, key: str, *, video, audio, condition=None) -> None:
        arrays = {
            "video": np.asarray(video, dtype=np.float32),
            "audio": np.asarray(audio, dtype=np.float32),
        }
        if condition is not None:
            arrays["condition"] = np.asarray(condition, dtype=np.float32)
        self._atomic_npz(self.root / "noise" / f"{key}.npz", **arrays)
        self._prune(self.root / "noise", "*.npz")

    def _prune_unreferenced_text(self) -> None:
        requests = []
        for path in (self.root / "text_requests").glob("*.json"):
            try:
                requests.append(json.loads(path.read_text())["text_embed_sha256"])
            except (OSError, KeyError, json.JSONDecodeError):
                path.unlink(missing_ok=True)
        referenced = set(requests)
        for entry in (self.root / "text").glob("*.npz"):
            if entry.stem not in referenced:
                entry.unlink(missing_ok=True)

    def _prune(self, directory: Path, pattern: str) -> None:
        entries = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime_ns, reverse=True)
        for path in entries[self.limit :]:
            path.unlink(missing_ok=True)

    @staticmethod
    def _atomic_npz(path: Path, **arrays) -> None:
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
        np.savez(temporary, **arrays)
        os.replace(temporary, path)

    @staticmethod
    def _atomic_json(path: Path, payload: dict) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
        os.replace(temporary, path)
