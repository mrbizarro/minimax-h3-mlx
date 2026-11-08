"""Per-block DiT weight streaming (low-memory Macs).

The Q8 DiT is 50 blocks x 0.38 GiB of linear weights plus 0.8 GiB of everything else. Resident,
that is the 20.2 GiB that keeps H3 off 24 GB and 16 GB Macs. Streamed, a forward holds the
non-block weights, the blocks a :class:`StreamPolicy` keeps resident, and ``prefetch + 1`` block
slots.

How a streamed block gets its weights:

* At load, the streamed tensors (``weight``/``scales``/``biases`` of ``attn.qkv_proj``,
  ``attn.out_proj``, ``mlp.fc1``, ``mlp.fc2``) are never read. Their module attributes hold a
  :class:`StreamedParam` (shape and dtype only), so code that inspects shapes (``lora.plan``)
  keeps working and anything that tries to compute with a released block fails loudly
  (``QuantizedLinear`` indexes ``self["weight"]``, a ``KeyError``).
* A fixed pool of slots, each one evaluated MLX array per streamed tensor, is allocated once.
  A worker thread fills a slot with ``os.preadv`` straight into the array's own buffer (the
  buffer protocol hands out MLX's Metal shared memory): SSD to GPU-visible memory in one copy,
  no numpy staging, no allocation churn. With ``F_NOCACHE`` the read also skips the page cache,
  so streaming leaves no clean-page residue and writes nothing to swap. The worker makes no MLX
  call; only the main thread touches MLX.
* ``acquire(i)`` waits for block ``i``'s reads, attaches the slot arrays, and queues the next
  streamed block (skipped blocks respected, wrapping into the next forward) into a free slot.
  The caller evaluates the block's output and calls ``release(i)``: the placeholders go back and
  the slot returns to the pool. A slot is only refilled after the ``mx.eval`` that consumed it
  has returned, so no GPU work can still be reading it.

LoRA adapters are runtime wrappers around the base linear (``lora.LoRALinear``); only the base
arrays are swapped, so an adapter (and its qkv permute, applied once to ``B`` at attach) is
unaffected. The values a streamed block multiplies by are the same bytes the resident loader
reads, through the same modules and kernels; only evaluation boundaries move.
"""

from __future__ import annotations

import fcntl
import os
import re
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

PARTS = {"qkv": "attn.qkv_proj", "out": "attn.out_proj", "fc1": "mlp.fc1", "fc2": "mlp.fc2"}
PARAMS = ("weight", "scales", "biases", "bias")
_BLOCK_KEY = re.compile(r"^blocks\.(\d+)\.(attn\.qkv_proj|attn\.out_proj|mlp\.fc1|mlp\.fc2)\.(\w+)$")


class StreamedParam:
    """Stands in for a streamed tensor while its block is not loaded: shape and dtype only.

    Not an ``mx.array``, so ``Module.parameters()`` skips it and nothing can evaluate it into
    memory by accident.
    """

    __slots__ = ("shape", "dtype", "nbytes")

    def __init__(self, shape, dtype, nbytes):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.nbytes = int(nbytes)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __repr__(self) -> str:
        return f"StreamedParam(shape={self.shape}, dtype={self.dtype})"


@dataclass(frozen=True)
class StreamPolicy:
    """Which block tensors stay resident; everything else streams.

    ``resident_blocks``: whole blocks kept resident. ``resident_parts``: parts (``qkv``, ``out``,
    ``fc1``, ``fc2``) kept resident in every block. ``resident_gb``: if set, resolved at load into
    the first ``floor(gb / block_gib)`` blocks (whatever ``resident_parts`` leaves streamed).
    """

    resident_blocks: frozenset = frozenset()
    resident_parts: frozenset = frozenset()
    resident_gb: float | None = None
    prefetch: int = 1
    workers: int = 4
    nocache: bool = True
    #: Queue block i before confirming block i-1 (one more slot): no host-side gap between blocks.
    pipeline: bool = False

    @classmethod
    def parse(cls, spec: str | None, **kw) -> "StreamPolicy":
        """``none`` | comma list of ``qkv,out,fc1,fc2``, ``first:K``, ``blocks:3-7``, ``gb:X``."""
        blocks: set[int] = set()
        parts: set[str] = set()
        gb = None
        for token in (spec or "none").replace(" ", "").split(","):
            if token in ("", "none"):
                continue
            if token in PARTS:
                parts.add(token)
            elif token.startswith("first:"):
                blocks.update(range(int(token[6:])))
            elif token.startswith("blocks:"):
                lo, _, hi = token[7:].partition("-")
                blocks.update(range(int(lo), int(hi or lo) + 1))
            elif token.startswith("gb:"):
                gb = float(token[3:])
            else:
                raise ValueError(f"unknown DiT residency token {token!r}")
        return cls(frozenset(blocks), frozenset(parts), gb, **kw)

    def describe(self) -> dict:
        return {
            "resident_blocks": sorted(self.resident_blocks),
            "resident_parts": sorted(self.resident_parts),
            "resident_gb": self.resident_gb,
            "prefetch": self.prefetch,
            "workers": self.workers,
            "nocache": self.nocache,
            "pipeline": self.pipeline,
        }


def plan_streamed(index: dict, num_layers: int, policy: StreamPolicy) -> tuple[dict, StreamPolicy]:
    """``{block: [(part_path, param, TensorRange), ...]}`` for every streamed tensor.

    Resolves ``resident_gb`` against the real per-block streamable size and returns the policy
    with the resolved block set.
    """
    by_block: dict[int, list] = {}
    for key, rng in index.items():
        match = _BLOCK_KEY.match(key)
        if not match:
            continue
        block, part, param = int(match.group(1)), match.group(2), match.group(3)
        if param not in PARAMS or block >= num_layers:
            continue
        short = next(k for k, v in PARTS.items() if v == part)
        if short in policy.resident_parts:
            continue
        by_block.setdefault(block, []).append((part, param, rng))
    resident = set(policy.resident_blocks)
    if policy.resident_gb is not None and by_block:
        block_bytes = max(sum(r.nbytes for _, _, r in items) for items in by_block.values())
        keep = int(policy.resident_gb * 1024**3 // block_bytes) if block_bytes else 0
        resident.update(range(min(keep, num_layers)))
    plan = {b: sorted(items, key=lambda t: (t[0], t[1])) for b, items in by_block.items()
            if b not in resident}
    resolved = StreamPolicy(frozenset(resident), policy.resident_parts, policy.resident_gb,
                            policy.prefetch, policy.workers, policy.nocache, policy.pipeline)
    return plan, resolved


def _resolve(block, part_path: str):
    obj = block
    for name in part_path.split("."):
        obj = getattr(obj, name)
    return getattr(obj, "base", obj)          # through a runtime LoRA wrapper


def _mlx_dtype(code: str):
    import mlx.core as mx

    return {
        "BF16": mx.bfloat16, "F16": mx.float16, "F32": mx.float32, "U32": mx.uint32,
        "U8": mx.uint8, "I8": mx.int8, "I32": mx.int32, "U16": mx.uint16, "I16": mx.int16,
    }[code]


@dataclass
class _Slot:
    arrays: dict                       # (part, param) -> evaluated mx.array (the pool buffer)
    views: dict                        # (part, param) -> writable byte view of that buffer
    block: int | None = None
    futures: list = field(default_factory=list)
    in_use: bool = False


class BlockStreamer:
    """Owns the slot pool and the read thread for one loaded DiT."""

    def __init__(self, dit, plan: dict, policy: StreamPolicy):
        import mlx.core as mx

        # Weak: the DiT owns the streamer (`dit.streamer`), so dropping the DiT drops the streamer
        # and `__del__` closes its descriptors even on a render that died mid-forward.
        self._dit_ref = weakref.ref(dit)
        self.num_layers = len(dit.blocks)
        self.plan = plan
        self.policy = policy
        self.streamed = sorted(plan)
        self._fds: dict = {}
        self._lock = threading.Lock()
        try:
            self._sim_bps = float(os.environ.get("H3_STREAM_SIM_GBPS") or 0) * 1e9
        except ValueError:
            self._sim_bps = 0.0
        self._pool = ThreadPoolExecutor(max_workers=max(1, policy.workers),
                                        thread_name_prefix="dit-stream")
        # One evaluated buffer per streamed tensor shape. Every block has the same shapes; a block
        # that does not (none in the released checkpoints) is refused rather than half-handled.
        template = {}
        for block in self.streamed:
            for part, param, rng in plan[block]:
                key = (part, param)
                spec = (rng.shape, rng.dtype, rng.nbytes)
                if template.setdefault(key, spec) != spec:
                    raise ValueError(f"streamed tensor {part}.{param} differs across blocks")
        self.template = template
        slots = max(1, min(policy.prefetch + 1 + int(policy.pipeline), len(self.streamed)))
        self.slots: list[_Slot] = []
        for _ in range(slots if self.streamed else 0):
            arrays = {key: mx.zeros(shape, dtype=_mlx_dtype(dtype))
                      for key, (shape, dtype, _) in template.items()}
            mx.eval(list(arrays.values()))
            # Exporting a buffer evaluates the array, so the views are made once, here, on the
            # main thread; the read workers only ever receive these views.
            views = {key: memoryview(arr).cast("B") for key, arr in arrays.items()}
            self.slots.append(_Slot(arrays, views))
        self.slot_bytes = sum(nbytes for _, _, nbytes in template.values())
        self.slot_count = len(self.slots)
        self.stats = {"blocks_loaded": 0, "bytes_read": 0, "wait_seconds": 0.0,
                      "read_seconds": 0.0, "max_wait_seconds": 0.0}
        self._placeholders()

    @property
    def dit(self):
        dit = self._dit_ref()
        if dit is None:
            raise RuntimeError("the streamed DiT is gone")
        return dit

    # ------------------------------------------------------------------ attach / detach
    def _placeholders(self) -> None:
        for block in self.streamed:
            self._detach(block)

    def _set(self, block: int, part: str, param: str, value) -> None:
        if param.startswith("lora_"):
            # "lora_a:K" / "lora_b:K": adapter K's factor on the runtime wrapper (lora.LoRALinear).
            wrapper = self.dit.blocks[block]
            for name in part.split("."):
                wrapper = getattr(wrapper, name)
            which, _, k = param.partition(":")
            getattr(wrapper, which)[int(k)] = value
        else:
            setattr(_resolve(self.dit.blocks[block], part), param, value)

    def _detach(self, block: int) -> None:
        for part, param, rng in self.plan[block]:
            self._set(block, part, param, StreamedParam(rng.shape, _mlx_dtype(rng.dtype), rng.nbytes))

    def _attach(self, block: int, slot: _Slot) -> None:
        for part, param, _ in self.plan[block]:
            self._set(block, part, param, slot.arrays[(part, param)])

    def stream_adapters(self, transformed: bool = False) -> dict:
        """Stream the runtime LoRA factors of streamed blocks too (a rank-384 adapter is 1.8 GiB).

        Only factors whose in-memory bytes are exactly the file's bytes qualify: no qkv permute or
        fc1 swap (``transformed``), a range-safe stored dtype, and a byte check against the file
        on the first streamed block of every (part, adapter). Anything else stays resident.
        Call right after the adapters are attached, before the first forward.
        """
        import mlx.core as mx
        import numpy as np

        from . import lora as lora_mod
        from .tensor_stream import tensor_index

        if transformed or not self.streamed or any(s.block is not None for s in self.slots):
            return {"streamed_factors": 0, "reason": "transformed or busy" if self.streamed else "nothing streamed"}
        indexes: dict = {}
        added: dict[int, list] = {b: [] for b in self.streamed}
        template: dict = {}
        first = self.streamed[0]
        for part in PARTS.values():
            wrapper = self.dit.blocks[first]
            for name in part.split("."):
                wrapper = getattr(wrapper, name)
            if not isinstance(wrapper, lora_mod.LoRALinear):
                continue
            for k, source in enumerate(wrapper.lora_sources):
                if source is None:
                    continue
                index = indexes.get(source)
                if index is None:
                    index = indexes[source] = tensor_index([source])
                names = sorted({key[: -len(lora_mod.LORA_A)] for key in index if key.endswith(lora_mod.LORA_A)})
                prefix = next((c for c in sorted(lora_mod.KNOWN_PREFIXES, key=len, reverse=True)
                               if names and all(n.startswith(c) for n in names)), "")
                entries = {}
                ok = True
                for block in self.streamed:
                    w = self.dit.blocks[block]
                    for name in part.split("."):
                        w = getattr(w, name)
                    if not isinstance(w, lora_mod.LoRALinear) or len(w.lora_sources) <= k \
                            or w.lora_sources[k] != source:
                        ok = False
                        break
                    for which, suffix in (("lora_a", lora_mod.LORA_A), ("lora_b", lora_mod.LORA_B)):
                        rng = index.get(f"{prefix}blocks.{block}.{part}{suffix}")
                        held = getattr(w, which)[k]
                        if rng is None or not isinstance(held, mx.array) \
                                or tuple(held.shape) != rng.shape or _mlx_dtype(rng.dtype) != held.dtype:
                            ok = False
                            break
                        if block == first and np.array(held.view(mx.uint8)).tobytes() != \
                                rng.numpy().tobytes():
                            ok = False
                            break
                        entries.setdefault(block, []).append((part, f"{which}:{k}", rng))
                    if not ok:
                        break
                if ok:
                    # One slot buffer serves every block, so each factor must have the same shape
                    # and dtype in every streamed block (a variable-rank adapter stays resident).
                    specs: dict = {}
                    for items in entries.values():
                        for _, param, rng in items:
                            if specs.setdefault(param, (rng.shape, rng.dtype)) != (rng.shape, rng.dtype):
                                ok = False
                if not ok:
                    continue
                for block, items in entries.items():
                    added[block].extend(items)
                    for _, param, rng in items:
                        template[(part, param)] = (rng.shape, rng.dtype, rng.nbytes)
        streamed_bytes = 0
        for block, items in added.items():
            self.plan[block].extend(items)
            for part, param, rng in items:
                self._set(block, part, param, StreamedParam(rng.shape, _mlx_dtype(rng.dtype), rng.nbytes))
                streamed_bytes += rng.nbytes
        for slot in self.slots:
            for key, (shape, dtype, _) in template.items():
                arr = mx.zeros(shape, dtype=_mlx_dtype(dtype))
                mx.eval(arr)
                slot.arrays[key] = arr
                slot.views[key] = memoryview(arr).cast("B")
        self.template.update(template)
        self.slot_bytes += sum(nbytes for _, _, nbytes in template.values())
        mx.clear_cache()
        self.adapter_summary = {"streamed_factors": len(template),
                                "streamed_gib": round(streamed_bytes / 1024**3, 3)}
        return self.adapter_summary

    # ------------------------------------------------------------------ IO
    def _fd(self, path) -> int:
        with self._lock:
            fd = self._fds.get(path)
            if fd is None:
                fd = os.open(path, os.O_RDONLY)
                if self.policy.nocache:
                    fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                self._fds[path] = fd
            return fd

    def _read(self, rng, view: memoryview) -> float:
        started = time.perf_counter()
        fd = self._fd(rng.file)
        done = 0
        while done < rng.nbytes:
            n = os.preadv(fd, [view[done:]], rng.offset + done)
            if n <= 0:
                raise OSError(f"short read at {rng.file}:{rng.offset + done}")
            done += n
        if self._sim_bps:
            # Lab only (H3_STREAM_SIM_GBPS): pace each worker so the pool as a whole reads no
            # faster than a slower Mac's SSD. Off unless the variable is set.
            floor = rng.nbytes / (self._sim_bps / max(1, self.policy.workers))
            remaining = floor - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
        return time.perf_counter() - started

    def _schedule(self, block: int, slot: _Slot) -> None:
        slot.block = block
        slot.futures = []
        for part, param, rng in self.plan[block]:
            view = slot.views[(part, param)]
            if view.nbytes != rng.nbytes:
                raise ValueError(f"slot buffer {part}.{param} is {view.nbytes} B, tensor {rng.nbytes} B")
            slot.futures.append(self._pool.submit(self._read, rng, view))

    def _next_streamed(self, block: int) -> int | None:
        skip = getattr(self.dit, "skip_blocks", None) or ()
        order = [b for b in self.streamed if b not in skip]
        if not order:
            return None
        for b in order:
            if b > block:
                return b
        return order[0]

    def _slot_for(self, block: int) -> _Slot | None:
        return next((s for s in self.slots if s.block == block), None)

    # ------------------------------------------------------------------ public
    def acquire(self, block: int) -> None:
        if block not in self.plan:
            return
        slot = self._slot_for(block)
        if slot is None:
            free = next((s for s in self.slots if s.block is None), None)
            if free is None:
                raise RuntimeError(f"no free stream slot for block {block} (release missing?)")
            self._schedule(block, free)
            slot = free
        started = time.perf_counter()
        for fut in slot.futures:
            self.stats["read_seconds"] += fut.result()
        waited = time.perf_counter() - started
        self.stats["wait_seconds"] += waited
        self.stats["max_wait_seconds"] = max(self.stats["max_wait_seconds"], waited)
        self.stats["blocks_loaded"] += 1
        self.stats["bytes_read"] += self.slot_bytes_for(block)
        slot.futures = []
        self._attach(block, slot)
        slot.in_use = True
        # Queue what comes next into every other free slot (prefetch depth = slots - 1).
        nxt = block
        for _ in range(len(self.slots) - 1):
            nxt = self._next_streamed(nxt)
            if nxt is None or nxt == block or self._slot_for(nxt) is not None:
                break
            free = next((s for s in self.slots if s.block is None), None)
            if free is None:
                break
            self._schedule(nxt, free)

    def release(self, block: int) -> None:
        """Call only after the block's output has been evaluated."""
        if block not in self.plan:
            return
        slot = self._slot_for(block)
        if slot is None:
            return
        self._detach(block)
        slot.block = None
        slot.in_use = False

    def slot_bytes_for(self, block: int) -> int:
        return sum(rng.nbytes for _, _, rng in self.plan.get(block, ()))

    def drain(self) -> None:
        """Drop queued prefetches (end of the denoise: the next forward may never come)."""
        for slot in self.slots:
            for fut in slot.futures:
                try:
                    fut.result()
                except Exception:
                    pass
            if not slot.in_use:
                slot.block = None
                slot.futures = []

    def abort(self) -> None:
        """Recover the pool after a forward died between ``acquire`` and ``release``.

        Waits out in-flight reads and any GPU work that may still read a slot, then puts every
        block's placeholders back and frees every slot, so a retried forward starts clean.
        """
        import mlx.core as mx

        for slot in self.slots:
            for fut in slot.futures:
                try:
                    fut.result()
                except BaseException:
                    pass
            slot.futures = []
        try:
            mx.synchronize()
        except Exception:
            pass
        for slot in self.slots:
            if slot.block is not None and slot.in_use:
                self._detach(slot.block)
            slot.block = None
            slot.in_use = False

    def __del__(self):
        try:
            self._pool.shutdown(wait=False, cancel_futures=True)
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()
        except Exception:
            pass

    def close(self) -> None:
        self.drain()
        self._pool.shutdown(wait=True)
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()
        for slot in self.slots:
            for view in slot.views.values():
                view.release()
        self.slots = []

    def summary(self) -> dict:
        resident_blocks = sorted(set(range(self.num_layers)) - set(self.streamed))
        return {
            "policy": self.policy.describe(),
            "streamed_blocks": len(self.streamed),
            "resident_blocks": resident_blocks,
            "slots": self.slot_count,
            "slot_gib": round(self.slot_bytes / 1024**3, 3),
            "streamed_gib_per_forward": round(
                sum(self.slot_bytes_for(b) for b in self.streamed) / 1024**3, 3),
            "blocks_loaded": self.stats["blocks_loaded"],
            "gib_read": round(self.stats["bytes_read"] / 1024**3, 3),
            "read_seconds": round(self.stats["read_seconds"], 3),
            "wait_seconds": round(self.stats["wait_seconds"], 3),
            "max_wait_seconds": round(self.stats["max_wait_seconds"], 3),
            "adapters": getattr(self, "adapter_summary", None),
            "sim_gbps": self._sim_bps / 1e9 if self._sim_bps else None,
        }
