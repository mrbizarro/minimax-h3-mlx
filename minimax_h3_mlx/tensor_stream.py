"""Read individual safetensors ranges without materializing the surrounding shard.

In particular, a compact conditioner can be a single 26 GiB file. Keeping a
dictionary returned by a whole-file loader defeats layer eviction if any of its
arrays have been evaluated. This index holds only filenames, shapes and offsets.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_DTYPES = {
    "BF16": "<u2", "F16": "<f2", "F32": "<f4", "F64": "<f8",
    "U8": "u1", "I8": "i1", "U16": "<u2", "I16": "<i2",
    "U32": "<u4", "I32": "<i4", "U64": "<u8", "I64": "<i8", "BOOL": "?",
}


@dataclass(frozen=True)
class TensorRange:
    file: Path
    offset: int
    shape: tuple[int, ...]
    dtype: str
    nbytes: int

    def numpy(self) -> np.ndarray:
        """Read just this tensor; BF16 is returned as its original uint16 bits."""
        with self.file.open("rb") as handle:
            handle.seek(self.offset)
            array = np.fromfile(handle, dtype=_DTYPES[self.dtype], count=math.prod(self.shape))
        if array.nbytes != self.nbytes:
            raise ValueError(f"Truncated tensor payload in {self.file}")
        return array.reshape(self.shape)

    def mlx(self):
        import mlx.core as mx

        array = mx.array(self.numpy())
        # A numeric cast from uint16 would corrupt BF16. This is a bit reinterpretation.
        return array.view(mx.bfloat16) if self.dtype == "BF16" else array


def tensor_index(paths) -> dict[str, TensorRange]:
    """Index headers only, rejecting duplicate names and malformed byte ranges."""
    index = {}
    for path in map(Path, paths):
        size = path.stat().st_size
        with path.open("rb") as handle:
            length = handle.read(8)
            if len(length) != 8:
                raise ValueError(f"Missing safetensors header: {path}")
            header_size = struct.unpack("<Q", length)[0]
            if header_size > min(size - 8, 100_000_000):
                raise ValueError(f"Invalid safetensors header size: {path}")
            header = json.loads(handle.read(header_size))
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if key in index:
                raise ValueError(f"Duplicate tensor {key!r} in {path}")
            dtype, shape = meta["dtype"], tuple(meta["shape"])
            if dtype not in _DTYPES or any(d < 0 for d in shape):
                raise ValueError(f"Unsupported tensor {key!r}: {dtype}, {shape}")
            start, end = meta["data_offsets"]
            nbytes = math.prod(shape) * np.dtype(_DTYPES[dtype]).itemsize
            if start < 0 or end - start != nbytes or 8 + header_size + end > size:
                raise ValueError(f"Invalid tensor byte range: {key!r} in {path}")
            index[key] = TensorRange(path, 8 + header_size + start, shape, dtype, nbytes)
    return index
