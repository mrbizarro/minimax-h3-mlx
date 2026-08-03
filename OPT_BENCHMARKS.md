# MiniMax H3 MLX — speed campaign benchmarks

Machine: Mac Studio, Apple M4 Max, 64 GiB unified memory. Python 3.11.15, MLX 0.32.0.
Phosphene panel and warm helper stopped for every timing run. Memory is `mx.get_peak_memory()`.

## The ceiling this machine has

| Measurement | Value |
|---|---|
| bf16 GEMM on H3's block shapes, 7,689 rows | **14.7 TFLOP/s** |
| `mx.fast.scaled_dot_product_attention`, same rows | **13.0 TFLOP/s** |
| Pipeline's achieved throughput, 5,577 → 25,138 rows | **13.7–13.8 TFLOP/s** |
| `mx.quantized_matmul` (8-bit and 4-bit) vs bf16 | **0.86–0.90x — slower** |
| Whole block under `mx.compile` | **1.005x** |

The denoiser runs at ~94% of the machine's own matmul speed of light. Every remaining second is
arithmetic that has to happen, so the campaign's levers are all "do less of it", never "do it
faster".

## Cost model (validated)

A step is 74% dense projection (linear in packed rows) and 24% attention (quadratic in packed
rows), with 2.3% of glue. From the 7,689-row block measurement:

    step_seconds  ≈  50 × (0.4058 × rows/7689  +  0.1309 × (rows/7689)²)

| Config | Rows | Predicted | Measured | Error |
|---|---:|---:|---:|---:|
| 640×384 · 73f | 5,577 | 18.6 s | 18.79 s | −1.0% |
| 768×448 · 73f | 7,689 | 26.8 s | 27.59 s | −2.7% |
| 768×448 · 124f | 12,982 | 53.9 s | 53.71 s | +0.4% |
| 768×448 · 243f | 25,138 | 135.2 s | 137.18 s | −1.4% |
| 768×448 · 124f + keyframe | 13,662 | 57.2 s | 58.3 s | −1.9% |

The model is good to ~3%, which is what makes the honest projections below trustworthy without
burning a render on each one.

## Baselines (from `../BENCHMARKS.md`)

| Tier | Canvas | Frames | Forwards | Rows | s/step | Total | Peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| Iteration config | 768×448 | 73 | 15 | 7,689 | 27.589 | **7:36.314** | 39.53 GiB |
| HQ 5-second | 768×448 | 124 | 15 | 12,982 | 53.705 | **14:38.669** | 40.22 GiB |
| HQ 10-second (hero) | 768×448 | 243 | 15 | 25,138 | 137.183 | **36:12.107** | 42.64 GiB |

## Showcase renders (announcement deliverable, not campaign levers)

| Render | Rows | s/step | Total | Peak |
|---|---:|---:|---:|---:|
| Text-only, 768×448 · 124f · 15 fw · seed 271828 | 12,980 | 53.911 | **14:40.4** | 40.22 GiB |
| **FL2VA first-frame**, 768×448 · 124f · 15 fw · seed 271828 | 13,662 | 57.931 | **15:42.7** | 40.37 GiB |

First-frame conditioning costs +682 packed rows (336 keyframe VAE rows + 336 vision tokens + the
`<Picture 1>: ` label) and +1.5 s to encode the still, for **+7.5% wall clock** — a very cheap way
to fix identity, wardrobe and composition to a real photograph. Frame 0 of the render reproduces
the source still closely enough to overlay; the face then stays stable through a 5.2-second push-in
and the dialogue line lands at −0.0 dB peak / −9.1 dB mean, the loudest speech any run produced.

Both showcase renders were single takes. No re-roll was needed.
