# H3 draft-tier wall-time campaign

Date: 2026-08-09

Branch: `codex/draft-speed`

Base commit: `fd6be7bfc291c40fae89e48b009f8c637b971090`

## Verdict

The opt-in draft stack finishes a 640x384, 4.67-second clip in **63.857 seconds** on the M4 Max:

- turbo LoRA, four sigma points / three forwards;
- temporal TAE decode;
- warm bounded re-draft cache;
- 56 generated frames delivered as 112 duplicate-paired frames in a 24 fps mux (12 unique fps).

This clears the 2:00 requirement and the 1:30 stretch target. At the original 124-frame grid, TAE
alone cuts total wall from 178.919s to 142.196s; the warm re-draft cache cuts that to 134.157s. The
full frame grid therefore still misses 2:00 because its three DiT forwards alone take ~117.7s.

Every optimization is gated. `--draft-decode` defaults to `full`; `--draft-cache-dir` and
`--draft-fps` are rejected unless `--draft-decode tae` is explicit. The 12 fps path defaults off.

## Reproduction

Weights used:

- DiT: `models/deepbeep-pruned-bf16/MiniMax-H3-FL2VA-pruned_bf16.safetensors`
- turbo LoRA: `models/turbo-lora/minimax_h3_turbo_4step.safetensors`
- adaLN time embedder: `models/turbo-lora/upstream_time_embedder.safetensors`
- temporal TAE: `models/tae/taeh3.safetensors` (madebyollin's H3-specific replacement linked from
  the Kijai/MiniMax-H3-TAE model card)

The winning draft-specific arguments are:

```text
--width 640 --height 384 --frames 56 --steps 4
--draft-decode tae --tae-checkpoint models/tae/taeh3.safetensors
--draft-cache-dir /path/to/draft-cache --draft-fps 12
```

`--draft-fps 12` deliberately does not guess a frame grid. The caller requests the next-lower
`17n+5` point (56 for this approximately five-second clip). The runner duplicates every decoded
frame and writes a normal 24 fps H.264 stream whose filename is automatically suffixed `_12fps`.
Audio is pitch-preserving 2x `atempo` stretch. A 100 ms source-audio tail pad prevents ffmpeg's
filter latency plus `-shortest` from dropping the last duplicate pair; the validated file contains
exactly 112 frames and lasts 4.666667s.

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
| 56f TAE, warm, 12 unique fps | 0.006 | 1.347 | 6.980 | 3.625 | 48.932 | 0.343 | 0.452 | 1.939 | **63.857** |

Seconds are the runner's own per-phase metrics. The final row delivers 112 muxed frames / 4.667s.
Its three denoise forwards average 16.31s, versus 39.23s at 124 frames. Packed rows fall from
9,819 to 4,791.

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

The cache has three independent LRU namespaces, each bounded to 50 entries by default:

1. Text requests map `(prompt, seed, canvas, prepared first-frame pixels, encoder identity)` to the
   full embedding SHA-256. Payloads are named by that embedding SHA. A hit is found before the
   encoder is constructed, so the 26 GB Q8 encoder never loads.
2. AdaLN tables are keyed by the exact timestep-table bytes plus DiT/LoRA/adaLN checkpoint
   fingerprints and LoRA scale. The optional final-layer LoRA delta is cached with the 50 block
   tables.
3. Seeded conditioning/video/audio noise is keyed by schedule, seed and exact tensor shapes.

Warm text load is 0.006s versus 7.277s cold. AdaLN+noise is 3.643s versus 4.154s cold: reuse works,
but the expected 2.7s saving was refuted on this pruned DiT. Its cached table is 46.1 MiB in bf16
(92 MiB serialized losslessly through float32), and loading/materializing it while 41.4 GB of DiT
weights are resident consumes much of the avoided projection time. The combined warm total saves
7.926s. Cold and warm video-stream hashes are exactly equal:

```text
e86ef488931e9d13c093fb59ab934990567898138a6b9c5ad10af2004426238d
```

This proves both cached conditioning and cached noise reproduce the uncached draft bytes.

### Work item 3: 12 fps draft

The 56-frame grid, duplicate-paired at mux, cuts denoise **117.683s -> 48.932s** and TAE decode
**0.702s -> 0.343s**. The delivered strip preserves the man's identity, prop motion, head movement
and mouth poses. Motion is visibly less fluid but useful for composition/timing iteration. The
ergonomics pass for an explicitly labeled preview; the option remains off by default.

Final artifact and metrics:

- `opt_out/draft_speed/final_12fps.mp4`
- `opt_out/draft_speed/frames_final_12fps/`
- `opt_metrics/draft_speed/final_12fps.json`
- video stream SHA-256: `036c5cb69ef3a4f4b5afa14e914fd1aa5a9ec637a5c0be5440b702ab0cfae4a6`

## Full-VAE fallback investigation

Not pursued. TAE passed the cheap compatibility/quality gate and removed 35.7s from the real draft
decode, so internal profiling and changes to the full VAE would add risk without improving the
winning draft path. Existing `H3_VAE_BATCH=8` already batches a full 640x384 clip's spatial tiles.

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

- `tests/test_draft_speed.py`: H3 7-token -> 22-frame cadence, RGB clamp, text LRU/content
  addressing, and exact noise round-trip — passed.
- `tests/test_dit_smoke.py` — passed.
- `tests/test_video_vae_smoke.py` — passed, including batched-vs-loop exactness.
- `tests/test_chain_stitch.py` — passed.
- `tests/test_serve_engine.py` — passed; the defaulted draft attributes preserve resident-engine
  callers that construct their own argument namespace.
