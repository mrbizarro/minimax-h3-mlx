# H3 draft-tier wall-time campaign

Date: 2026-08-09

Branch: `codex/draft-speed`

Base commit: `fd6be7bfc291c40fae89e48b009f8c637b971090`

## Verdict

The selected 640x384, 5.17-second draft is **127.882 seconds** on the M4 Max: turbo LoRA, four
sigma points / three forwards, temporal TAE decode, the complete 124-frame motion/audio grid, and
the warm bounded re-draft cache.

This beats the original 178.919s full-VAE draft by 51.037s (28.5%), but it **misses the 2:00 target
by 7.882s**. It is nevertheless the selected tier because it preserves the native 124-frame joint
audio/video trajectory and its cached output is byte-identical to uncached TAE. No reduced-grid
result passed the combined picture, motion and sound gate.

Every kept optimization is gated. `--draft-decode` defaults to `full`; `--draft-cache-dir` is
rejected unless `--draft-decode tae` is explicit. Reduced-grid research remains behind explicit
`--draft-fps` / `--draft-clock` flags and is not a recommended or default tier. The full frame
grid's three DiT forwards alone take ~117.7s.

## Reproduction

Weights used:

- DiT: `models/deepbeep-pruned-bf16/MiniMax-H3-FL2VA-pruned_bf16.safetensors`
- turbo LoRA: `models/turbo-lora/minimax_h3_turbo_4step.safetensors`
- adaLN time embedder: `models/turbo-lora/upstream_time_embedder.safetensors`
- temporal TAE: `models/tae/taeh3.safetensors` (madebyollin's H3-specific replacement linked from
  the Kijai/MiniMax-H3-TAE model card)

The selected draft-specific arguments are:

```text
--width 640 --height 384 --frames 124 --steps 4
--draft-decode tae --tae-checkpoint models/tae/taeh3.safetensors
--draft-cache-dir /path/to/draft-cache
```

The runner generates and decodes all 124 H3 frames and native full-duration audio. No cadence or
audio-timing transform is part of the recommended path.

## Cheap latent compatibility gate

The requested note, `notes/TR1DAE_NODE.md`, is a study of the Tr1dae latent *upscaler*, not a TAE
implementation. It nevertheless confirms the native H3 transformer contract used here:
`(B, 24, T, H, W)` normalized video latents and grid-coherent temporal/spatial metadata.

Before runner integration, one existing Stage-A cache was unpatchified and center-cropped to a
real normalized latent with shape **(1, 24, 7, 12, 20)**, corresponding to 320x192 and 22 frames.
The full VAE and the MLX TAE decoded those exact bytes.

| Decoder | Load | Decode | Peak during decode | Result |
|---|---:|---:|---:|---|
| Full 36-layer ViT VAE | 0.223s | 1.350s | 6.308 GiB | reference detail |
| Temporal TAE | 0.009s | 0.240s | 1.532 GiB | recognizable face and pose; motion readable; soft |

The compatibility gate passed. The TAE consumes the transformer's normalized latent directly; it
must **not** receive the full VAE's per-channel mean/std reversal. H3's encoder discards three tail
tokens. The TAE path restores them by repeating the final clean token, decodes five-token chunks,
removes each chunk's three causal lead-in frames, and cuts only final grid padding. This recovers
the exact `17n+5` frame count.

Kijai's original 9.8 MB checkpoint is a 2D decoder: it has the correct 24 input channels but emits
one image per latent time slice. Its own model card now recommends madebyollin's 22 MB temporal H3
checkpoint, which reconstructs the pixel-time cadence instead of requiring latent-rate frame
duplication. The temporal checkpoint was therefore the integration candidate.

Probe artifacts:

- `opt_out/draft_speed/probe_320x192/full_vae.mp4`
- `opt_out/draft_speed/probe_320x192/tae.mp4`
- `opt_out/draft_speed/probe_320x192/side_by_side.png`
- `opt_out/draft_speed/probe_320x192/probe_metrics.json`

## Wall-time results

All 640x384 rows use the same prompt, keyframe, seed `161616`, turbo LoRA, adaLN LoRA and four
sigma points. `text_embed_sha256` is `3f63097e2c9f8c8c` throughout.

| Run | Text | Keyframe | DiT load | AdaLN + noise | Denoise | Video decode | Audio | Mux | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 124f full VAE baseline | 7.818 | 1.394 | 7.643 | 4.713 | 117.218 | 36.763 | 0.838 | 2.137 | **178.919** |
| 124f TAE, no cache | 7.097 | 1.368 | 7.209 | 4.430 | 117.464 | 1.099 | 0.881 | 2.241 | **142.196** |
| 124f TAE, cache cold | 7.279 | 1.370 | 7.444 | 4.154 | 117.699 | 0.715 | 0.869 | 2.180 | **142.083** |
| 124f TAE, cache warm | 0.006 | 1.344 | 7.427 | 3.643 | 117.683 | 0.702 | 0.868 | 2.199 | **134.157** |
| 124f TAE, all caches warm | 0.005 | 0.000 | 6.952 | 2.812 | 115.899 | 0.666 | 0.860 | 0.476 | **127.882** |
| 56f TAE, old shortened/time-stretched mux (rejected) | 0.006 | 1.347 | 6.980 | 3.625 | 48.932 | 0.343 | 0.452 | 1.939 | **63.857** |
| 56 unique / 124 delivered, native full audio (rejected) | 0.005 | 0.691 | 7.753 | 4.354 | 50.429 | 0.331 | 0.840 | 2.000 | **66.662** |

Seconds are the runner's own per-phase metrics. The final two rows are falsification evidence, not
candidates. The selected 124-frame warm TAE run carries the complete native model grid over a
5.167s container with 9,819 packed rows.

### Work item 1: TAE draft decode

At the real 124-frame shape, video decode falls **36.763s -> 1.099s** (33.5x, -35.664s). Peak
decode memory falls from 8.599 GiB to 7.074 GiB on the first full-size probe, and to 3.593 GiB on
the 56-frame final. The qualitative contract is met: identity, face, hand-held prop and mouth/pose
changes remain readable. Fine skin/fur texture is soft, which is acceptable for a preview.

Full-size A/B artifacts:

- full VAE: `opt_out/turbo/T3-preview.mp4`
- TAE: `opt_out/draft_speed/tae124.mp4`
- frame strip: `opt_out/draft_speed/full_vs_tae_strip.png`
- metrics: `opt_metrics/turbo_T3-preview.json`, `opt_metrics/draft_speed/tae124.json`

### Work item 2: bounded re-draft caching

The cache has four independent LRU namespaces, each bounded to 50 entries by default:

1. Text requests map `(prompt, seed, canvas, prepared first-frame pixels, encoder identity)` to the
   full embedding SHA-256. Payloads are named by that embedding SHA. A hit is found before the
   encoder is constructed, so the 26 GB Q8 encoder never loads.
2. First-frame VAE rows are keyed by the prepared pixels, canvas, VAE checkpoint and patch size.
   The encode uses fixed seed 42, so cached float32 rows round-trip exactly.
3. AdaLN tables are keyed by the exact timestep-table bytes plus DiT/LoRA/adaLN checkpoint
   fingerprints and LoRA scale. The optional final-layer LoRA delta is cached with the 50 block
   tables.
4. Seeded conditioning/video/audio noise is keyed by schedule, seed and exact tensor shapes.

Warm text load is 0.006s versus 7.277s cold. AdaLN+noise is 3.643s versus 4.154s cold: reuse works,
but the expected 2.7s saving was refuted on this pruned DiT. Its cached table is 46.1 MiB in bf16
(92 MiB serialized losslessly through float32), and loading/materializing it while 41.4 GB of DiT
weights are resident consumes much of the avoided projection time. The combined warm total saves
7.926s. Cold and warm video-stream hashes are exactly equal:

```text
e86ef488931e9d13c093fb59ab934990567898138a6b9c5ad10af2004426238d
```

This proves both cached conditioning and cached noise reproduce the uncached draft bytes.

The later keyframe-row cache removes the measured 1.344s keyframe phase. With every namespace warm,
the selected tier measured 127.882s. Its video-stream hash remains the same `e86ef4...` digest as
the uncached and earlier warm TAE runs, so the speedup does not move the selected picture.

### Work item 3: reduced-cadence draft — experimental, not selected

The first 56-frame probe duplicated frames and used a 2x `atempo` stretch. It measured 63.857s,
but delivered only 112 frames / 4.667s and audibly distorted the generated voice. It was invalid.

A corrected probe kept a 124-frame / 5.167-second delivery and native full-duration audio. Its 56
video frames were spread across the full model clock (2.307693x temporal scale) and distributed
over a 24 fps mux. It measured 66.662s, but visual review found the motion/image quality materially
degraded. That is the intended falsification gate: timing correctness alone is not acceptable.

Later probes kept audio/video row counts equal, used Apple's maximum-overlap time/pitch unit, and
tested progressively denser temporal grids. None replaced the full-motion TAE tier:

| Probe | Internal source | Delivery | Warm wall | Review |
|---|---|---|---:|---|
| native 56 | 640x384, 56f | 640x384, 124f / 5.17s | 50.999s | synchronized slow motion; rejected |
| joint-spread 56 | 640x384, 56f | 640x384, 120f / 5.0s | 50.993s | timing/image/sound not acceptable; rejected |
| joint-spread 73 | 576x352, 73f | 640x384, 120f / 5.0s | 54.782s | better cadence; not selected |
| joint-spread 90 | 512x320, 90f | 640x384, 120f / 5.0s | 53.749s | ongoing R&D; not selected |

The research flags remain explicit and default-off so these arms can keep improving without
changing the selected 124-frame tier or the dense/HQ path. Inspectable artifacts include:

- invalid short/audio-stretched probe: `opt_out/draft_speed/final_12fps.mp4`
- corrected-duration but visually degraded probe: `opt_out/draft_speed/corrected_12fps.mp4`
- 73-frame probe: `opt_out/draft_speed/resident_t73_warm.mp4`
- 90-frame probe: `opt_out/draft_speed/resident_t90_warm.mp4`
- metrics: `opt_metrics/draft_speed/final_12fps.json`,
  `opt_metrics/draft_speed/corrected_12fps.json`,
  `opt_metrics/draft_speed/resident_t73_warm.json`,
  `opt_metrics/draft_speed/resident_t90_warm.json`

## Visual decision set

| Choice | Wall | Duration / motion | Artifact | Decision |
|---|---:|---|---|---|
| Full VAE reference | 178.919s | 5.17s / native 24 fps | `opt_out/turbo/T3-preview.mp4` | detail reference |
| **TAE + all caches warm** | **127.882s** | **5.17s / native 24 fps** | `opt_out/draft_speed/tae124_keyframe_warm.mp4` | **selected** |
| 73-frame experiment | 54.782s | 5.0s / 14.6 unique fps | `opt_out/draft_speed/resident_t73_warm.mp4` | not selected |
| 90-frame experiment | 53.749s | 5.0s / 18 unique fps | `opt_out/draft_speed/resident_t90_warm.mp4` | ongoing R&D |

The full-VAE/TAE side-by-side strip is `opt_out/draft_speed/full_vs_tae_strip.png`.

## Full-VAE fallback investigation

Not pursued. TAE passed the cheap compatibility/quality gate and removed 35.7s from the real draft
decode, so internal profiling and changes to the full VAE would add risk without improving the
selected draft path. Existing `H3_VAE_BATCH=8` already batches a full 640x384 clip's spatial tiles.

## Dense/HQ byte regression

The pre-branch control is the existing full-quality `R1_1024x576_s9` render: 1024x576, 124 frames,
nine sigma points / eight forwards, seed 161616, full VAE, no LoRA and no prompt cache.

```text
pre-branch video-stream SHA-256:
d68c9725c8d11fae8500429088d6b97acc21f272a190db206edf2b81fb6aa00b
```

The post-change control completed in 1,139.916s. Its conditioning SHA, packed geometry and all
eight denoise-step outputs followed the pre-branch control, and its video stream hashes to:

```text
post-change video-stream SHA-256:
d68c9725c8d11fae8500429088d6b97acc21f272a190db206edf2b81fb6aa00b
```

**PASS: the hashes are identical.** The default full-VAE/dense/HQ path is byte-unchanged.

Control artifacts:

- pre-branch: `opt_out/hdloop/probes/R1_1024x576_s9.mp4`
- post-change: `opt_out/draft_speed/full_regression_1024.mp4`
- post-change metrics: `opt_metrics/draft_speed/full_regression_1024.json`

## Tests

- `tests/test_draft_speed.py`: H3 cadence/RGB, bounded text and keyframe caches, exact noise,
  reduced-grid selection, joint-clock row invariance and delivery resize — passed.
- `tests/test_dit_smoke.py` — passed.
- `tests/test_video_vae_smoke.py` — passed, including batched-vs-loop exactness.
- `tests/test_chain_stitch.py` — passed.
- `tests/test_serve_engine.py` — passed; the defaulted draft attributes preserve resident-engine
  callers that construct their own argument namespace.

The repository aggregate `scripts/run_tests.sh` could not run the PyTorch parity modules in this
worktree because its `.venv` lacks `torch`, and its first three commands use system `python3`
without MLX. The five relevant suites above were rerun explicitly with the same MLX interpreter
used by every benchmark and passed. The aggregate failure was missing dependencies, not a failed
assertion.
