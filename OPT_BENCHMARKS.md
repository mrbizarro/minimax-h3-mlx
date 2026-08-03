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

## Campaign results — iteration config (768×448, 73 frames, seed 314159)

| Run | Forwards | Wall | vs baseline | s/step | Peak | Audio peak/mean | Quality |
|---|---:|---:|---:|---:|---:|---|---|
| `opt_b0_fw15` | 15 | 7:39 | 1.00x | 27.76 | 39.63 GiB | −10.6 / −29.3 dB | reference |
| `opt_f8` | 8 | **4:24** | **1.74x** | 27.74 | 39.43 GiB | −12.9 / −32.1 dB | none–minor |
| `opt_f6` | 6 | 3:28 | 2.21x | 27.74 | 39.38 GiB | −13.0 / −32.7 dB | visible |
| `opt_f4` | 4 | 2:33 | 3.00x | 27.74 | 39.36 GiB | −14.6 / −33.1 dB | visible→unacceptable |
| `opt_c8` (step cache .08) | 8 real of 15 | 4:24 | 1.74x | — | 39.63 GiB | — | worse than `opt_f8` |
| `opt_c6` (step cache .12) | 6 real of 15 | 3:29 | 2.20x | — | 39.42 GiB | — | ≈ `opt_f6` |
| `opt_half39` (39f @ 12 fps) | 8 | 2:18 | 1.91x* | 13.84 | 38.99 GiB | −20.2 / −34.3 dB | visible→unacceptable |

\* against `opt_f8`, for a comparable clip length (3.19 s vs 3.04 s), not against the 15-forward baseline.

## Campaign results — production tiers

| Tier | Frames | Forwards | Wall | vs baseline | s/step | Rows | Peak | Quality |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 3.04 s | 73 | 15 | 7:39 | — | 27.76 | 7,689 | 39.63 GiB | reference |
| 3.04 s | 73 | **8** | **4:24** | **1.74x** | 27.74 | 7,689 | 39.43 GiB | **none–minor** |
| 5.17 s | 124 | 15 | 14:38.7 | — | 53.705 | 12,982 | 40.22 GiB | reference |
| 5.17 s | 124 | **8** | **8:19** | **1.76x** | 54.401 | 12,982 | 40.10 GiB | **none–minor** |
| 10.13 s | 243 | 15 | 36:12.1 | — | 137.183 | 25,138 | 42.64 GiB | reference |
| 10.13 s | 243 | 8 | 21:01.7 | 1.72x | 143.28 | 25,138 | 42.52 GiB | **visible — rejected** |

The 10-second row is the campaign's negative result: 1.72x is real, but the render puts two
astronauts on screen for its first 2.2 seconds. The forward-count lever has to be re-gated at every
duration, because a longer packed sequence needs more of the schedule to resolve subject count and
object identity.

The 143.28 s/step in that run is 4.4% above the baseline for identical geometry and drifts upward
within the run. A `bench_block` re-run immediately afterwards was within 0.5% of the campaign's
opening measurement, so short bursts are not throttled — this is sustained-load behaviour on a
144-second continuous full-occupancy step. Normalized to the baseline s/step the run is 20:13.

## Contact sheets

| File | Shows |
|---|---|
| `../opt_out/grid_forwards.png` | 15 / 8 / 6 / 4 forwards, four matched timestamps, audio levels |
| `../opt_out/grid_faces.png` | the same four at face magnification — the deciding evidence |
| `../opt_out/grid_stepcache_faces.png` | uniform 8 fw vs step cache at 8 real fw, matched wall clock |
| `../opt_out/grid_stepcache6_faces.png` | the same at 6 forwards |
| `../opt_out/grid_halfdensity.png` | full density vs half density + `minterpolate` |
| `../opt_out/grid_hero.png` | 10-second hero, 15 fw vs 8 fw |
| `../opt_out/grid_hero_open.png` | the hero's first 2.2 s — the duplicated subject |
| `../opt_out/grid_5s.png` | 5-second tier, 15 fw vs 8 fw — clean |
| `../opt_out/ff_first_vs_still.png` | FL2VA first frame vs the source still |
