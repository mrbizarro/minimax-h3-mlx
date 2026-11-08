"""Memory plan for one H3 render, decided before anything large loads.

Three questions, answered from checkpoint headers (no tensor is read) and the request's shape:

1. What does each pass need? A render is a sequence of passes that never overlap: the prompt
   encoder, the denoise (DiT weights + adapters + working memory over the packed rows), and the
   decode (video VAE + tile working memory). The largest pass is the render's peak.
2. Which settings make the largest pass fit? ``resident`` is today's path, unchanged. ``lowmem``
   streams the prompt encoder and the DiT block weights, caps MLX's buffer cache, and picks a
   decode tile batch that fits.
3. Should the render start at all? ``admit`` separates "this Mac cannot hold the largest pass
   even with nothing else open" from "it could, but other apps are using the memory right now".

The working-memory terms are calibrated on measured renders (M4 Max, mlx 0.32.4, 5 s clips at
768x448 and 1024x576); every number carries a margin and is written to the metrics file next to
the measured peak, so the calibration can be checked on every render. Units are GiB throughout.
"""

from __future__ import annotations

import json
import os
import re
import struct
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

GIB = 1024 ** 3

# --- calibration (see LOWMEM_LOG.md for the runs these come from) ---------------------------------
# A pass needs its MLX peak (live arrays plus transient buffers), the buffer cache MLX is allowed
# to keep, and the process outside MLX. Uncapped (today's path), the cache is not counted: MLX
# returns it under its memory limit. Fitted to M4 Max, mlx 0.32.4, 768x448x124 Fast lane:
#   resident  encode 25.86 / denoise 22.79 / decode 8.00 GiB MLX peak  (footprint 26.29 / 26.82)
#   streamed  encode  0.54 / denoise  4.75 / decode 8.01 GiB MLX peak  (footprint  2.66 /  6.91 / 8.92)
#: Process memory outside MLX's counters: interpreter, Metal driver, libraries, tokenizer.
RUNTIME_GIB = 0.50
#: Denoise working memory per 1,000 packed rows (attention projections, SwiGLU intermediates,
#: residual stream): 2.41 GiB resident and 2.29 GiB streamed at 13,075 rows.
DENOISE_GIB_PER_KROW = 0.185
#: Encoder working memory over one unit (hidden states, dequantization scratch).
ENCODE_WORK_GIB = 0.30
#: Video-VAE decode working memory per megapixel of canvas, by tile batch (fp16 decode):
#: 3.15 GiB at 0.344 MP batch 8, 2.34 GiB at batch 1 (7.19 GiB MLX peak with 4.85 of weights).
DECODE_GIB_PER_MP = {1: 6.8, 8: 9.2}
#: Low-memory buffer-cache caps (MLX keeps freed buffers for reuse up to these).
LOWMEM_CACHE_GIB = 2.0
LOWMEM_DECODE_CACHE_GIB = 1.0
#: Decode settings tried in order until the plan fits: (stream the VAE decoder?, tile batch).
LOWMEM_DECODE_LADDER = ((False, 8), (True, 8), (True, 1))
#: Every estimate is inflated by this before it is compared with a budget.
MARGIN = 1.10


@dataclass
class CheckpointSizes:
    te_total: float              # every encoder tensor the resident loader keeps (text part)
    te_vision: float             # the vision tower, loaded only for image conditioning
    te_max_unit: float           # the largest unit the streamed encoder holds at once
    te_max_layer: float          # the largest decoder layer (the token table is read by row)
    te_max_vision_unit: float    # the largest vision-tower unit
    dit_total: float
    dit_block_stream: float      # streamed bytes of one block (qkv/out/fc1/fc2)
    dit_nonblock: float          # embeddings, norms, final layer, AdaLN, token refiner
    video_vae: float
    audio_vae: float
    vae_block: float = 0.0       # one ViT decoder block (the streamed decode holds two)
    vae_decode_fixed: float = 0.0  # what a streamed decode keeps: no blocks, no CNN encoder
    lora: list = field(default_factory=list)   # [(path, GiB)]


def _header(path: Path) -> dict:
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        head = json.loads(fh.read(n))
    head.pop("__metadata__", None)
    return head


def _nbytes(meta: dict) -> int:
    lo, hi = meta["data_offsets"]
    return hi - lo


_TE_LAYER = re.compile(r"^(?:model\.)?(?:language_model\.)?layers\.(\d+)\.")
_TE_UNIT = re.compile(
    r"^(?:model\.)?(?:language_model\.)?(embed_tokens|layers\.\d+)|"
    r"^(?:model\.)?visual\.(blocks\.\d+|patch_embed|merger|deepstack_merger_list\.\d+|pos_embed)"
)
_VAE_BLOCK = re.compile(r"^decoder\.transformer_blocks\.(\d+)\.")
_DIT_BLOCK = re.compile(r"^blocks\.(\d+)\.(attn\.qkv_proj|attn\.out_proj|mlp\.fc1|mlp\.fc2)\.")


def checkpoint_sizes(compact_root: Path, dit_dir: Path, loras=(), encoder_layers: int = 50) -> CheckpointSizes:
    compact_root, dit_dir = Path(compact_root), Path(dit_dir)
    te_text = te_vision = 0
    units: dict[str, int] = {}
    for key, meta in _header(compact_root / "text_encoder.safetensors").items():
        if key.startswith("lm_head"):
            continue
        m = _TE_LAYER.match(key)
        if m and "visual" not in key and int(m.group(1)) >= encoder_layers:
            continue
        size = _nbytes(meta)
        if "visual." in key:
            te_vision += size
        else:
            te_text += size
        u = _TE_UNIT.match(key)
        if u:
            name = u.group(1) or u.group(2)
            units[name] = units.get(name, 0) + size
    dit_total = block_stream = 0
    per_block: dict[int, int] = {}
    shards = sorted(p for p in dit_dir.glob("*.safetensors") if not p.name.startswith("."))
    for shard in shards:
        for key, meta in _header(shard).items():
            size = _nbytes(meta)
            dit_total += size
            m = _DIT_BLOCK.match(key)
            if m:
                per_block[int(m.group(1))] = per_block.get(int(m.group(1)), 0) + size
    block_stream = max(per_block.values()) if per_block else 0
    streamed_all = sum(per_block.values())
    vae_blocks: dict[int, int] = {}
    vae_fixed = 0
    for key, meta in _header(compact_root / "video_vae.safetensors").items():
        m = _VAE_BLOCK.match(key)
        if m:
            vae_blocks[int(m.group(1))] = vae_blocks.get(int(m.group(1)), 0) + _nbytes(meta)
        elif not key.startswith("encoder."):
            vae_fixed += _nbytes(meta)
    lora = []
    for spec in loras or ():
        path = Path(str(spec).rsplit(":", 1)[0]) if ":" in str(spec) and not Path(str(spec)).exists() else Path(str(spec))
        if path.is_file():
            lora.append((str(path), path.stat().st_size / GIB))
    return CheckpointSizes(
        te_total=te_text / GIB,
        te_vision=te_vision / GIB,
        te_max_unit=max(units.values()) / GIB if units else 0.0,
        te_max_layer=max((v for k, v in units.items() if k.startswith("layers.")), default=0) / GIB,
        te_max_vision_unit=max((v for k, v in units.items()
                                if not k.startswith(("layers.", "embed_tokens"))), default=0) / GIB,
        dit_total=dit_total / GIB,
        dit_block_stream=block_stream / GIB,
        dit_nonblock=(dit_total - streamed_all) / GIB,
        video_vae=(compact_root / "video_vae.safetensors").stat().st_size / GIB,
        audio_vae=(compact_root / "audio_vae.safetensors").stat().st_size / GIB,
        vae_block=max(vae_blocks.values(), default=0) / GIB,
        vae_decode_fixed=vae_fixed / GIB,
        lora=lora,
    )


def packed_rows(width: int, height: int, frames: int, text_tokens: int = 384,
                keyframes: int = 0, history: bool = False, reference_rows: int = 0) -> int:
    """Rows the DiT attends over: video latent patches, two rows per audio latent, the prompt.

    Matches the runner's own layout (768x448x124 with 229 prompt tokens is 13,075 rows).
    """
    from .packing import audio_latent_num_frames, video_latent_num_frames

    per_frame = (int(height) // 32) * (int(width) // 32)
    latent = video_latent_num_frames(int(frames))
    rows = per_frame * latent + 2 * audio_latent_num_frames(int(frames)) + int(text_tokens)
    rows += per_frame * (int(keyframes) + (5 if history else 0)) + int(reference_rows)
    return rows


@dataclass
class Plan:
    mode: str                      # "resident" (today's path) or "lowmem"
    stream_text_encoder: bool = False
    stream_dit: bool = False
    dit_pipeline: bool = False
    stream_vae_decoder: bool = False
    cache_gb: float | None = None
    decode_cache_gb: float | None = None
    vae_batch: int = 8
    passes: dict = field(default_factory=dict)   # name -> estimated footprint GiB
    peak_pass: str = ""
    peak_gib: float = 0.0

    def runner_args(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k not in ("passes", "peak_pass", "peak_gib")}


def estimate(sizes: CheckpointSizes, rows: int, width: int, height: int, *, vision: bool,
             lowmem: bool, vae_batch: int, lora_streams: bool = False,
             stream_vae: bool = False) -> dict:
    """Estimated process footprint (GiB) of each pass under one setting, margin included."""
    lora_gib = sum(g for _, g in sizes.lora)
    vis = sizes.te_vision if vision else 0.0
    if lowmem:
        # One unit at a time (the token table is gathered by row, so a decoder layer, or the
        # vision tower's largest block, is the biggest), plus the capped cache.
        encode = max(sizes.te_max_layer, sizes.te_max_vision_unit if vision else 0.0) + ENCODE_WORK_GIB
        # Slots: the block computing, the one queued behind it, and the one being read.
        slots = 3 * sizes.dit_block_stream
        lora_resident = 0.0 if lora_streams else lora_gib
        if lora_streams:
            slots += 3 * lora_gib / 50.0
        weights = sizes.dit_nonblock + slots + lora_resident
        cache, dcache = LOWMEM_CACHE_GIB, LOWMEM_DECODE_CACHE_GIB
    else:
        encode = sizes.te_total + vis + ENCODE_WORK_GIB
        weights = sizes.dit_total + lora_gib
        cache = dcache = 0.0
    denoise = weights + DENOISE_GIB_PER_KROW * rows / 1000.0
    b = 1 if vae_batch <= 1 else 8
    vae_weights = (sizes.vae_decode_fixed + 2 * sizes.vae_block) if (lowmem and stream_vae) \
        else sizes.video_vae
    decode = vae_weights + DECODE_GIB_PER_MP[b] * int(width) * int(height) / 1e6
    passes = {"encode": encode + cache, "denoise": denoise + cache, "decode": decode + dcache}
    return {k: round((v + RUNTIME_GIB) * MARGIN, 2) for k, v in passes.items()}


def choose(sizes: CheckpointSizes, rows: int, width: int, height: int, budget_gib: float, *,
           vision: bool, mode: str, lora_streams: bool = False,
           available_gib: float | None = None) -> Plan:
    """``mode``: ``off`` = today's path always; ``on`` = low memory always; ``auto`` = today's path
    when it fits, otherwise the low-memory settings.

    "Fits" is against the smaller of the budget and what other apps leave free right now, so a
    big Mac with a full desktop streams instead of pushing everything else into swap.
    """
    target = budget_gib if available_gib is None else min(budget_gib, available_gib)
    resident = estimate(sizes, rows, width, height, vision=vision, lowmem=False, vae_batch=8)
    if mode == "off" or (mode == "auto" and max(resident.values()) <= target):
        peak = max(resident, key=resident.get)
        return Plan("resident", passes=resident, peak_pass=peak, peak_gib=resident[peak])
    best = None
    # Cheapest first: the decoder resident (no rereads), then streamed at the big tile batch (a
    # handful of decoder calls, each rereading 4.5 GiB under compute), then one tile per call.
    for stream_vae, batch in LOWMEM_DECODE_LADDER:
        est = estimate(sizes, rows, width, height, vision=vision, lowmem=True, vae_batch=batch,
                       lora_streams=lora_streams, stream_vae=stream_vae)
        peak = max(est, key=est.get)
        plan = Plan("lowmem", stream_text_encoder=True, stream_dit=True, dit_pipeline=True,
                    stream_vae_decoder=stream_vae,
                    cache_gb=LOWMEM_CACHE_GIB, decode_cache_gb=LOWMEM_DECODE_CACHE_GIB,
                    vae_batch=batch, passes=est,
                    peak_pass=peak, peak_gib=est[peak])
        best = plan
        if est[peak] <= target:
            break
    return best


# --- the machine ---------------------------------------------------------------------------------
def _sysctl(name: str) -> str:
    return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()


def machine_memory(device: dict | None = None) -> dict:
    """RAM, what other apps leave free right now, and the GPU working-set ceiling.

    ``available`` counts free, inactive, purgeable and speculative pages: memory a new process
    gets without anything being compressed or swapped. Lab overrides (``H3_SIMULATE_RAM_GB``,
    ``H3_SIMULATE_AVAILABLE_GB``) let a big Mac rehearse a small one's decision.
    """
    ram = int(_sysctl("hw.memsize") or 0) / GIB
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1)) if "page size" in out else 16384
    pages = {}
    for line in out.splitlines()[1:]:
        key, _, val = line.partition(":")
        try:
            pages[key.strip()] = int(val.strip().rstrip("."))
        except ValueError:
            pass
    available = sum(pages.get(k, 0) for k in
                    ("Pages free", "Pages inactive", "Pages purgeable", "Pages speculative")) * page / GIB
    swap = re.search(r"used = ([\d.]+)M", _sysctl("vm.swapusage"))
    gpu = (device or {}).get("max_recommended_working_set_size")
    gpu = gpu / GIB if gpu else ram * 0.75
    simulated = False
    sim_ram = os.environ.get("H3_SIMULATE_RAM_GB")
    if sim_ram:
        scale = float(sim_ram) / ram
        ram, gpu, simulated = float(sim_ram), gpu * scale, True
        # A freshly booted small Mac with a browser and the studio open.
        available = min(available, float(sim_ram) - os_reserve_gib(float(sim_ram)))
    sim_avail = os.environ.get("H3_SIMULATE_AVAILABLE_GB")
    if sim_avail:
        available, simulated = float(sim_avail), True
    return {"ram_gib": round(ram, 2), "available_gib": round(available, 2),
            "gpu_working_set_gib": round(gpu, 2),
            "swap_used_gib": round(float(swap.group(1)) / 1024, 2) if swap else None,
            "simulated": simulated}


def os_reserve_gib(ram_gib: float) -> float:
    """What macOS and a couple of everyday apps keep: the studio's own H3 headroom rule
    (the larger of 6 GB and an eighth of RAM)."""
    return max(6.0, ram_gib / 8.0)


@dataclass
class Admission:
    verdict: str          # "ok" | "busy" | "cannot"
    need_gib: float
    peak_pass: str
    mac_can_give_gib: float
    free_now_gib: float
    message: str

    def as_dict(self) -> dict:
        return asdict(self)


def admit(plan: Plan, mem: dict, budget_gib: float | None = None) -> Admission:
    """Refuse honestly before twenty minutes of work, and say which kind of refusal it is."""
    need = plan.peak_gib
    can_give = mem["ram_gib"] - os_reserve_gib(mem["ram_gib"])
    if budget_gib is not None:
        can_give = min(can_give, budget_gib)
    free_now = mem["available_gib"]
    label = {"encode": "reading the prompt", "denoise": "making the clip",
             "decode": "finishing the video", "revoice": "cleaning up the sound"}.get(
                 plan.peak_pass, plan.peak_pass)
    if need > can_give:
        return Admission("cannot", need, plan.peak_pass, round(can_give, 2), free_now,
                         f"This Mac cannot make this clip: {label} needs about {need:.0f} GB and this "
                         f"{mem['ram_gib']:.0f} GB Mac can give a render about {can_give:.0f} GB. "
                         "Try a smaller size or a shorter clip.")
    if need > free_now:
        short = need - free_now
        return Admission("busy", need, plan.peak_pass, round(can_give, 2), free_now,
                         f"This Mac can make this clip, but other apps are using the memory right "
                         f"now: it needs about {need:.0f} GB and {free_now:.0f} GB is free. Close "
                         f"apps to free about {short:.0f} GB more, or try again later.")
    return Admission("ok", need, plan.peak_pass, round(can_give, 2), free_now, "")


# --- the runner's entry point --------------------------------------------------------------------
#: Bytes per packed row of the re-voice key/value store (int8 keys and values for 50 blocks plus
#: per-group scales); the studio prices "Clean up sound" with the same figure.
REVOICE_KV_BYTES_PER_ROW = 2 * 7168 * 50 * (1 + 4 / 128)
REFERENCE_ROWS = 2 * 1376
EXIT_CANNOT = 76          # this Mac cannot hold the largest pass (75 is the studio's stop code)
EXIT_BUSY = 77            # it could, but other apps leave too little free right now


def plan_for_request(args, device: dict, frames: int, chain: int, window_keyframes, prompts) -> dict:
    """Decide the memory plan for one runner invocation and apply it to ``args``.

    Returns a JSON-able record (sizes, estimate per pass, machine, budget, admission) for the
    metrics file. Only ``--low-memory on|auto`` changes any setting; ``off`` (the default) leaves
    every argument exactly as given.
    """
    loras = [str(s) for s in (getattr(args, "lora", None) or [])]
    if isinstance(getattr(args, "lora", None), (str, Path)):
        loras = [str(args.lora)]
    sizes = checkpoint_sizes(args.compact_root, args.dit, loras)
    # Tokens are only known after encoding; a prompt runs about 4.4 characters per token, so a
    # character count over 3 is a safe ceiling. Vision tokens ride the text stream too.
    text_tokens = max((len(p or "") for p in prompts), default=0) // 3 + 64
    keyframes = max((len(w) for w in (window_keyframes or [])), default=0)
    refs = len(getattr(args, "reference", None) or [])
    history = chain > 1 or getattr(args, "extend_from", None) is not None
    vision = bool(keyframes or refs)
    if vision:
        text_tokens += 1400 * max(1, keyframes + refs)
    rows = packed_rows(args.width, args.height, frames, text_tokens, keyframes=keyframes,
                       history=history, reference_rows=REFERENCE_ROWS * refs)
    mem = machine_memory(device)
    budget = min(float(args.memory_gb), mem["ram_gib"] - os_reserve_gib(mem["ram_gib"]))
    transformed = bool(getattr(args, "lora_qkv_permute", False) or getattr(args, "lora_fc1_swap", False))
    mode = getattr(args, "low_memory", "off")
    plan = choose(sizes, rows, args.width, args.height, budget, vision=vision,
                  mode="off" if mode == "off" else mode, lora_streams=not transformed,
                  available_gib=mem["available_gib"])
    if plan.mode == "lowmem" and getattr(args, "stream_vae_decoder", False) and not plan.stream_vae_decoder:
        # The caller asked for the streamed decoder: estimate what will actually run.
        batch = 1 if os.environ.get("H3_VAE_BATCH") in ("0", "1") else plan.vae_batch
        plan.passes = estimate(sizes, rows, args.width, args.height, vision=vision, lowmem=True,
                               vae_batch=batch, lora_streams=not transformed, stream_vae=True)
        plan.stream_vae_decoder, plan.vae_batch = True, batch
        plan.peak_pass = max(plan.passes, key=plan.passes.get)
        plan.peak_gib = plan.passes[plan.peak_pass]
    revoice = int(getattr(args, "revoice_steps", 0) or 0) > 0 and getattr(args, "revoice_kv", "int8") != "off"
    if revoice:
        kv = rows * REVOICE_KV_BYTES_PER_ROW * (2 if getattr(args, "revoice_kv", "int8") == "bf16" else 1) / GIB
        plan.passes["revoice"] = round(plan.passes["denoise"] + kv * MARGIN, 2)
        peak = max(plan.passes, key=plan.passes.get)
        plan.peak_pass, plan.peak_gib = peak, plan.passes[peak]
    if plan.mode == "lowmem":
        # Explicit flags win: a caller that already asked for a setting keeps it.
        args.stream_text_encoder = True
        args.stream_dit = True
        args.dit_pipeline = True
        if getattr(args, "cache_gb", None) is None:
            args.cache_gb = plan.cache_gb
        if getattr(args, "decode_cache_gb", None) is None:
            args.decode_cache_gb = plan.decode_cache_gb
        if plan.stream_vae_decoder:
            args.stream_vae_decoder = True
        if plan.vae_batch <= 1 and os.environ.get("H3_VAE_BATCH") is None:
            os.environ["H3_VAE_BATCH"] = "1"
    admission = admit(plan, mem, budget)
    return {
        "mode_requested": mode,
        "mode": plan.mode,
        "rows_estimate": rows,
        "budget_gib": round(budget, 2),
        "passes_gib": plan.passes,
        "peak_pass": plan.peak_pass,
        "peak_gib": plan.peak_gib,
        "vae_batch": plan.vae_batch,
        "stream_vae_decoder": plan.stream_vae_decoder,
        "sizes_gib": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(sizes).items()},
        "machine": mem,
        "admission": admission.as_dict(),
    }
