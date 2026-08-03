# MiniMax H3 MLX — speed campaign log

Machine: Apple M4 Max, 64 GiB unified memory. Branch `opt/speed-campaign`.
Baselines are the verified staged-pipeline numbers in `../BENCHMARKS.md` (bf16 DiT, no CFG pass).

Every timing run is taken with the Phosphene panel and its warm helper stopped, and memory is
always `mx.get_peak_memory()`, never RSS.

Iteration config for lever isolation: **768×448, 73 frames, 15 forwards, seed 314159**, the
astronaut capability prompt — baseline **7:36.314**, 27.589 s/step, 7,689 packed rows.
Hero config: **768×448, 243 frames, 15 forwards, seed 314159** — baseline **36:12.107**,
137.183 s/step, 25,138 packed rows.

---

## Phase 0 — where the time actually is

Before optimizing anything, two micro-benchmarks establish the speed of light. This turned out to
be the most important half hour of the campaign: it refuted three of the six planned levers on
measurement rather than on a seven-minute render each.

### `scripts/bench_gemm.py` — the four block projections in isolation, 7,689 rows

| Projection | shape | bf16 | TFLOP/s | q8 | q4 |
|---|---|---:|---:|---:|---:|
| `attn.qkv_proj` | 5376×21504 | 120.8 ms | 14.72 | 138.4 ms (0.87x) | 137.0 ms (0.88x) |
| `attn.out_proj` | 7168×5376 | 40.6 ms | 14.60 | 46.3 ms (0.88x) | 45.8 ms (0.89x) |
| `mlp.fc1` | 5376×28672 | 160.4 ms | 14.78 | 183.2 ms (0.88x) | 185.5 ms (0.86x) |
| `mlp.fc2` | 14336×5376 | 84.0 ms | 14.12 | 93.7 ms (0.90x) | 92.9 ms (0.90x) |
| **per block** | | **405.8 ms** | | 461.7 ms | 461.2 ms |
| **×50 blocks** | | **20.29 s** | | 23.08 s | 23.06 s |

### `scripts/bench_block.py` — one real `TransformerBlock`, 7,689 rows

| Part | ms | ×50 blocks | share |
|---|---:|---:|---:|
| **block total** | **549.0** | **27.45 s** | 100% |
| attention (qkv + sdpa + out) | 298.0 | 14.90 s | 54.3% |
| — of which `mx.fast.scaled_dot_product_attention` | 130.9 | 6.55 s | 23.8% |
| mlp (fc1 + silu·gate + fc2) | 250.0 | 12.50 s | 45.5% |
| `norm1` | 1.2 | 0.06 s | 0.2% |
| norm + AdaLN row gather | 2.3 | 0.12 s | 0.4% |
| 6× AdaLN row gather alone | 3.6 | 0.18 s | 0.7% |
| **block under `mx.compile`** | **546.5** | 27.32 s | 99.5% |

The projected 27.45 s/step lands on the measured 27.589 s/step, so the denoising step *is* the
block stack — there is no hidden overhead anywhere else in the runner.

**Conclusion: 74% of a step is dense GEMM at 14.7 TFLOP/s, 24% is SDPA at 13.0 TFLOP/s, and 2.3%
is everything else.** The pipeline already runs at ~94% of this machine's own matmul speed of
light. No implementation-level change can return more than a couple of percent.

---

## Lever results

### L1 — SDPA audit · **REFUTED, nothing to fix**

`dit.py:Attention.__call__` already calls `mx.fast.scaled_dot_product_attention`. There is no
hand-rolled `softmax(QK^T)V` anywhere in the file. SDPA reaches 13.0 TFLOP/s against the 14.7
TFLOP/s the dense GEMMs reach on the same hardware — 88% of local peak, so the fast path is
being used properly and there is no headroom to recover.

The redundant-cast audit came back clean too, and the commit 468afff lesson is already encoded in
`dit.py:param_dtype`: casting activations to `QuantizedLinear.weight` truncates them to integers
because that tensor is packed uint32 storage, so the function reads `scales.dtype` instead. Every
cast in the forward goes through it. Removing casts would in any case be chasing the 2.3% bucket.

**Quality cost: n/a. Keep: nothing to change.**

### L2 — `mx.compile` the per-step function · **REFUTED, 0.5%**

Measured on the real block: 546.5 ms compiled vs 549.0 ms eager, i.e. **1.005x**. This is the
expected result given Phase 0 — compile fuses elementwise work, and elementwise work is 2.3% of
the step. Shapes are static within a run so `shapeless=True` was not needed, and compile overhead
would have to amortize across only 4–15 steps.

**Quality cost: none (bit-identical class of change). Keep: no — not worth the graph-capture risk
for 0.5%.**

### L5 — Q8 DiT · **REFUTED before building it, would be 12% SLOWER**

The plan was to quantize the pruned bf16 DiT to MLX affine 8-bit group-64 (~20 GB on disk, about
an hour of work) and A/B it. `bench_gemm.py` shows this would have been a pure loss: at these
shapes `mx.quantized_matmul` is **0.86–0.90x** of bf16 at both 8 and 4 bits, so a Q8 DiT costs
23.08 s/step against bf16's 20.29 s of projection time.

The reason is Phase 0's finding. Quantization wins when a GEMM is *bandwidth* bound — the classic
LLM decode case, where M=1 and the weights are read once per token. Here M is 5,577–25,138 rows,
every weight is amortized over thousands of rows, and the kernel is compute bound. Dequantization
then shows up as pure added arithmetic.

Q8 remains the right lever for a *memory* problem (it would take the resident DiT from ~38.6 GB to
~10 GB and open the door to a 32 GB machine). It is the wrong lever for a speed problem on 64 GB.

**Quality cost: not measured — not built. Keep: no.** 20 GB of disk and an hour saved.
