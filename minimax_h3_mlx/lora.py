"""Low-rank adapters for the DiT, and the two things that make this checkpoint family awkward.

Written for larryvrh's ``MiniMax-H3-Turbo-Lora`` (4-step joint audio-video distillation, Apache-2.0),
but nothing here is specific to it beyond the defaults.

Two facts drive the design, both measured rather than assumed — re-run
``scripts/lora_numerics.py`` and ``scripts/lora_layout_probe.py`` against any LoRA before trusting
either claim on a file this port has not seen:

1. **Folding the update into the bf16 base destroys it.** The turbo delta is ~3.4e-4 of the weight
   it rides on, while bfloat16's relative ULP is 2^-8 ~ 3.9e-3. Rounding ``W + dW`` back to bf16
   keeps ~13 % of the update (correlation as low as 0.19 with the intended delta). The reference
   generator says as much in a comment; we measured it before believing it. ``fuse`` mode is kept
   as a control — it is the honest "what if we had folded" arm — and ``runtime`` is the default.

2. **The fused qkv rows are in a different order in the two lineages.** MiniMaxAI's release (and
   therefore DeepBeepMeep's pruned export, and this port) stores ``attn.qkv_proj`` **per-head
   interleaved**, ``(heads, 3, head_dim)``. The ComfyUI model definition the LoRA was trained
   through splits contiguously, ``(3, heads, head_dim)``. Applying ``lora_B`` unpermuted writes the
   q update onto the first 18.7 heads' q/k/v triplets and so on — silent, total corruption that
   presents as "the LoRA does nothing good". :func:`_permute_qkv_rows` fixes it.

The pruned lineage also **cannot accept the adaLN LoRA** the way the others are taken: DeepBeepMeep
replaced the ``[96768, 2688]`` per-block timestep projection with a ``[96768, 64]`` projection over a
sampled rank-64 curve, so the ``[16, 2688]`` ``lora_A`` has no input to consume. Those 51 modules are
reported as skipped with the reason, never silently dropped — and :func:`absorb_adaln_lora` applies
them anyway, exactly, by folding their timestep-only delta into the precomputed modulation cache.
See the "Turbo LoRA" section of the README for the whole picture.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

LORA_A = ".lora_A.weight"
LORA_B = ".lora_B.weight"

# Modules whose fused output rows follow the attention head layout and therefore need remapping
# from the training lineage's (3, heads, head_dim) to this checkpoint's (heads, 3, head_dim).
QKV_SUFFIX = ".attn.qkv_proj"

# Set by `--lora-audit`: makes every wrapped layer record |lora_out| / |base_out| once.
AUDIT = False


@dataclass
class ModuleReport:
    name: str
    status: str          # "applied" | "skipped"
    reason: str = ""
    rank: int = 0
    shape: tuple[int, ...] = ()


@dataclass
class LoRAReport:
    path: str
    scale: float
    mode: str
    applied: list[ModuleReport] = field(default_factory=list)
    skipped: list[ModuleReport] = field(default_factory=list)
    seconds: float = 0.0
    permuted_qkv: int = 0

    def summary(self) -> dict:
        by_reason: dict[str, int] = {}
        for m in self.skipped:
            by_reason[m.reason] = by_reason.get(m.reason, 0) + 1
        return {
            "path": self.path,
            "scale": self.scale,
            "mode": self.mode,
            "applied": len(self.applied),
            "skipped": len(self.skipped),
            "skipped_by_reason": by_reason,
            "qkv_rows_permuted": self.permuted_qkv,
            "seconds": round(self.seconds, 2),
        }

    def render(self) -> str:
        s = self.summary()
        lines = [
            f"LoRA {Path(self.path).name} scale {self.scale:g} mode {self.mode}: "
            f"{s['applied']} applied, {s['skipped']} skipped, "
            f"{self.permuted_qkv} qkv row-blocks re-ordered, {s['seconds']:.2f}s"
        ]
        for reason, count in sorted(s["skipped_by_reason"].items()):
            lines.append(f"  skipped {count}: {reason}")
        return "\n".join(lines)


def parse_spec(spec: str) -> tuple[Path, float]:
    """``PATH`` or ``PATH:SCALE`` -> ``(path, scale)``. Default scale 1.0 (alpha == rank)."""
    text = str(spec)
    path, _, tail = text.rpartition(":")
    if path and tail:
        try:
            return Path(path), float(tail)
        except ValueError:
            pass
    return Path(text), 1.0


# Redistributions for other runtimes namespace the DiT under their own wrapper. Stripping a known
# prefix is safe because the suffix still has to match a module in our tree to be applied at all.
KNOWN_PREFIXES = ("diffusion_model.", "transformer.", "model.diffusion_model.")


def load_pairs(path: str | Path) -> dict[str, tuple[mx.array, mx.array]]:
    """Read a LoRA safetensors into ``{module_name: (A, B)}``, A ``[r, in]``, B ``[out, r]``."""
    raw = mx.load(str(path))
    names = sorted({k[: -len(LORA_A)] for k in raw if k.endswith(LORA_A)})
    prefix = ""
    for candidate in sorted(KNOWN_PREFIXES, key=len, reverse=True):
        if names and all(n.startswith(candidate) for n in names):
            prefix = candidate
            break
    pairs: dict[str, tuple[mx.array, mx.array]] = {}
    for name in names:
        a, b = raw.get(name + LORA_A), raw.get(name + LORA_B)
        if a is None or b is None:
            continue
        pairs[name[len(prefix):]] = (a, b)
    return pairs


def _permute_qkv_rows(b: mx.array, heads: int, head_dim: int) -> mx.array:
    """``lora_B`` rows ``(3, heads, head_dim)`` -> ``(heads, 3, head_dim)``.

    ``b`` is ``[3*heads*head_dim, rank]``; only the row axis is touched.
    """
    rank = b.shape[1]
    return b.reshape(3, heads, head_dim, rank).transpose(1, 0, 2, 3).reshape(3 * heads * head_dim, rank)


def _iter_targets(model) -> dict[str, tuple[nn.Module, str]]:
    """Every wrappable linear in the DiT, keyed by its checkpoint name."""
    targets: dict[str, tuple[nn.Module, str]] = {}
    for prefix, blocks in (("blocks", model.blocks), ("token_refiner.blocks", model.token_refiner.blocks)):
        for i, blk in enumerate(blocks):
            targets[f"{prefix}.{i}.attn.qkv_proj"] = (blk.attn, "qkv_proj")
            targets[f"{prefix}.{i}.attn.out_proj"] = (blk.attn, "out_proj")
            targets[f"{prefix}.{i}.mlp.fc1"] = (blk.mlp, "fc1")
            targets[f"{prefix}.{i}.mlp.fc2"] = (blk.mlp, "fc2")
    return targets


class LoRALinear(nn.Module):
    """``y = base(x) + scale * (x @ A^T) @ B^T``, the update kept out of the base weight.

    The low-rank product is evaluated at run time precisely because folding it into a bf16 weight
    rounds it away (see the module docstring). Cost is ~1.5 % of the linear's FLOPs at rank 64.
    """

    def __init__(self, base: nn.Module, a: mx.array, b: mx.array, scale: float):
        super().__init__()
        self.base = base
        self.lora_a = a
        self.lora_b = b
        self.lora_scale = float(scale)
        self._audit_ratio: float | None = None

    @property
    def weight(self) -> mx.array:          # keeps `param_dtype()` working through the wrapper
        return self.base.weight

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base(x)
        if self.lora_scale == 0.0:
            return y
        mid = x.astype(self.lora_a.dtype) @ self.lora_a.T
        delta = mid @ self.lora_b.T
        if self.lora_scale != 1.0:
            delta = delta * self.lora_scale
        delta = delta.astype(y.dtype)
        if AUDIT and self._audit_ratio is None:
            # Weight-space size (3.4e-4 of |W|) does not tell us whether the update survives the
            # bf16 residual add — only the ratio on *real* activations does. Measured once, on the
            # first forward, then never again.
            self._audit_ratio = float(
                (mx.sqrt(mx.sum(delta.astype(mx.float32) ** 2))
                 / mx.sqrt(mx.sum(y.astype(mx.float32) ** 2))).item()
            )
        return y + delta


def plan(model, pairs: dict[str, tuple[mx.array, mx.array]]) -> tuple[dict, list[ModuleReport]]:
    """Split the LoRA into what this checkpoint can take and what it cannot, with reasons."""
    targets = _iter_targets(model)
    applicable: dict[str, tuple[nn.Module, str, mx.array, mx.array]] = {}
    skipped: list[ModuleReport] = []
    heads, head_dim = model.config.num_attention_heads, model.config.attention_head_dim

    for name, (a, b) in pairs.items():
        rank = int(a.shape[0])
        if name not in targets:
            if name.endswith("adaln_proj.linear"):
                reason = (
                    f"adaLN projection: this checkpoint is the pruned lineage "
                    f"(adaln input dim {model.config.time_embed_dim}, LoRA expects {a.shape[1]})"
                )
            else:
                reason = "no matching module in the port's tree"
            skipped.append(ModuleReport(name, "skipped", reason, rank, tuple(b.shape)))
            continue
        parent, attr = targets[name]
        base = getattr(parent, attr)
        out_dim, in_dim = base.weight.shape
        if int(a.shape[1]) != in_dim or int(b.shape[0]) != out_dim:
            skipped.append(ModuleReport(
                name, "skipped",
                f"shape mismatch: base [{out_dim}, {in_dim}] vs B[{b.shape[0]}] A[{a.shape[1]}]",
                rank, tuple(b.shape)))
            continue
        if name.endswith(QKV_SUFFIX):
            b = _permute_qkv_rows(b, heads, head_dim)
        applicable[name] = (parent, attr, a, b)
    return applicable, skipped


def apply_lora(
    model,
    path: str | Path,
    scale: float = 1.0,
    mode: str = "runtime",
    verbose: bool = True,
) -> LoRAReport:
    """Attach (``runtime``) or merge (``fuse``) a LoRA onto a loaded DiT.

    ``scale = 0.0`` is a strict no-op in both modes: ``runtime`` short-circuits the delta and
    ``fuse`` skips the write, so a zero-scale render is bit-identical to no ``--lora`` at all.
    """
    if mode not in {"runtime", "fuse"}:
        raise ValueError(f"lora mode must be 'runtime' or 'fuse', got {mode!r}")
    started = time.perf_counter()
    pairs = load_pairs(path)
    applicable, skipped = plan(model, pairs)
    report = LoRAReport(path=str(path), scale=float(scale), mode=mode, skipped=skipped)

    for name, (parent, attr, a, b) in applicable.items():
        base = getattr(parent, attr)
        if name.endswith(QKV_SUFFIX):
            report.permuted_qkv += 1
        if mode == "runtime":
            setattr(parent, attr, LoRALinear(base, a, b, scale))
        elif scale != 0.0:
            # The merge itself is exact in float32; the loss is the single rounding back to the
            # base dtype, which is the whole point of the `fuse` control arm.
            w = base.weight
            merged = (w.astype(mx.float32) + (b.astype(mx.float32) @ a.astype(mx.float32)) * scale)
            base.weight = merged.astype(w.dtype)
            mx.eval(base.weight)
            del w, merged
        report.applied.append(ModuleReport(name, "applied", "", int(a.shape[0]), tuple(b.shape)))

    if mode == "fuse":
        mx.clear_cache()
    report.seconds = time.perf_counter() - started
    if verbose:
        print(report.render(), flush=True)
        for m in skipped[:2]:
            print(f"  e.g. {m.name}: {m.reason}", flush=True)
    return report


# ---------------------------------------------------------------------------------------------
# adaLN absorption
#
# The 51 adaLN modules cannot be wrapped — the pruned checkpoint's projection reads a 64-d curve,
# not the 2688-d timestep embedding their `lora_A` expects. They can still be applied *exactly*,
# because what they contribute depends on the timestep alone:
#
#     dM_b(t) = B_b @ (A_b @ silu(temb(t)))
#
# and the AdaLN table is already precomputed per distinct timestep. Feeding it needs the upstream
# `time_embedder`, which `scripts/fetch_time_embedder.py` range-fetches (63 MB of a 66 GB release)
# and `scripts/verify_time_embedder.py` proves is the right one: the recovered silu(temb) explains
# DeepBeepMeep's published `adaln_t_table` to a residual of 4e-12, against 1.8e-4 for the bare
# sinusoid and 4.7e-4 for shuffled weights.
# ---------------------------------------------------------------------------------------------

ADALN_SUFFIX = "adaln_proj.linear"


def _sinusoid(t: mx.array, dim: int = 256, max_period: float = 10000.0) -> mx.array:
    """The port's own ``timestep_embedding`` (flip_sin_to_cos), duplicated to avoid a cycle."""
    import math

    half = dim // 2
    exponent = -math.log(max_period) * mx.arange(half, dtype=mx.float32) / half
    emb = t.astype(mx.float32)[:, None] * mx.exp(exponent)[None, :]
    return mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)


def silu_temb(timesteps: mx.array, embedder_path: str | Path) -> mx.array:
    """``silu(time_embedder(sinusoid(t)))`` — the 2688-d vector the adaLN LoRA consumes."""
    w = mx.load(str(embedder_path))
    h = nn.silu(_sinusoid(timesteps) @ w["time_embedder.proj_in.weight"].T
                + w["time_embedder.proj_in.bias"])
    temb = h @ w["time_embedder.proj_out.weight"].T + w["time_embedder.proj_out.bias"]
    return nn.silu(temb.astype(mx.float32))


def absorb_adaln_lora(
    dit,
    cache,
    path: str | Path,
    embedder_path: str | Path,
    scale: float = 1.0,
    verbose: bool = True,
) -> dict:
    """Add the adaLN LoRA's timestep-dependent delta into a built :class:`ModulationCache`.

    Mutates ``cache.tables`` (and the final layer's projection) in place. Returns a report that
    includes the *measured* size of the delta against the modulation it corrects — the number that
    says whether the correction survives the table's bfloat16 storage at all.
    """
    if scale == 0.0:
        return {"blocks": 0, "final": False, "scale": 0.0}
    pairs = load_pairs(path)
    adaln = {k: v for k, v in pairs.items() if k.endswith(ADALN_SUFFIX)}
    if not adaln:
        if verbose:
            print("adaLN LoRA: this file carries no adaln_proj pairs — nothing to absorb. "
                  "(Re-distributions converted for pruned runtimes drop them.)", flush=True)
        return {"blocks": 0, "final": False, "scale": scale, "reason": "no adaln pairs in file"}

    hidden = dit.config.hidden_size
    st = silu_temb(cache.timesteps, embedder_path)          # (T, 2688) float32
    rel_num = rel_den = 0.0
    applied = 0

    for index in range(len(dit.blocks)):
        key = f"blocks.{index}.{ADALN_SUFFIX}"
        if key not in adaln:
            continue
        a, b = adaln[key]
        delta = ((st @ a.astype(mx.float32).T) @ b.astype(mx.float32).T) * scale   # (T, 96768)
        delta = delta.reshape(-1, 6 * hidden)                                      # (T*3, 32256)
        table = cache.tables[index]
        merged = []
        for i, part in enumerate(table):
            piece = delta[:, i * hidden : (i + 1) * hidden]
            rel_num += float(mx.sum(piece.astype(mx.float32) ** 2).item())
            rel_den += float(mx.sum(part.astype(mx.float32) ** 2).item())
            merged.append((part.astype(mx.float32) + piece).astype(part.dtype))
        cache.tables[index] = tuple(merged)
        mx.eval(cache.tables[index])
        del delta, merged
        applied += 1

    final_key = f"final_layer.{ADALN_SUFFIX}"
    final = False
    if final_key in adaln:
        a, b = adaln[final_key]
        # The final layer re-projects every forward instead of reading a table, so the delta is
        # parked on the module and added inside `FinalLayer.norm_out`.
        dit.final_layer.adaln_proj.lora_delta = (
            ((st @ a.astype(mx.float32).T) @ b.astype(mx.float32).T) * scale
        )
        mx.eval(dit.final_layer.adaln_proj.lora_delta)
        final = True

    mx.clear_cache()
    ratio = (rel_num / rel_den) ** 0.5 if rel_den else float("nan")
    report = {"blocks": applied, "final": final, "scale": float(scale),
              "delta_over_modulation": ratio, "timesteps": int(cache.num_timesteps)}
    if verbose:
        print(f"adaLN LoRA absorbed into the modulation cache: {applied} blocks"
              f"{' + final layer' if final else ''}, |dM|/|M| = {ratio:.3e} "
              f"(bf16 table ULP is 3.9e-3)", flush=True)
    return report


def audit_output_scale(model, limit: int = 6) -> list[tuple[str, float]]:
    """Ratio ``|lora_out| / |base_out|`` recorded by wrapped layers, if auditing was enabled."""
    out = []
    for name, (parent, attr) in _iter_targets(model).items():
        layer = getattr(parent, attr)
        ratio = getattr(layer, "_audit_ratio", None)
        if ratio is not None:
            out.append((name, float(ratio)))
    return out[:limit] if limit else out
