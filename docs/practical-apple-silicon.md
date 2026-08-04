# Practical Apple Silicon tiers — measured on 64 GiB

The top-level README measures H3 at its native 1344x768 canvas on a 550 GB M3 Ultra and concludes
that local generation is impractical. That conclusion holds at native size. It does not hold at the
sub-native tiers below, which were measured end to end on a **64 GiB Apple M4 Max desktop**.

**High-quality H3 with jointly generated stereo audio runs locally at 768x448: 3.04 seconds in
7:36, 5.17 seconds in 14:39, and 10.13 seconds in 36:12.** A faster 640x384 tier produces 3.04
seconds in 3:05. Peak Metal allocation never exceeds 42.7 GiB.

This document covers only what was measured on this machine. It supersedes the README's
practicality conclusion at these canvases and nothing else.

## Component selection

Every component is fetched by the user from its original Hub repository under that repository's own
terms. No weights are redistributed here.

| Component | Source repository | File | On-disk | Role |
|---|---|---|---:|---|
| FL2VA DiT | `DeepBeepMeep/MiniMax-H3` | `MiniMax-H3-FL2VA-pruned_bf16.safetensors` | 41.399 GB | Pruned BF16 visual/audio core |
| Text encoder | `ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit` | `text_encoder.safetensors` | 28.223 GB | Native MLX affine Q8/g64, 50 layers |
| Video VAE | same repository | `video_vae.safetensors` | 5.208 GB | Compact fp16 export |
| Audio VAE | same repository | `audio_vae.safetensors` | 0.605 GB | Compact export, weight norm folded |
| Text config | `MiniMaxAI/MiniMax-H3` | `FL2VA/text_encoder/config.json` | small | Authoritative encoder config |

Header-level parity was checked against the MLX module trees before any inference: DiT 531/531
required parameters, video VAE 559/559, audio VAE 661/661, and 1,251/1,251 required tensors in the
truncated Q8 language tower. `rope.inv_freq` is a computed buffer and the unused final language
norm is intentionally absent. Loading stays strict; no key mismatch is tolerated.

## Why it fits

1. The pruned rank-64 AdaLN curve removes the original timestep projection path without quantizing
   the visual core, which stays BF16.
2. The Q8 text encoder runs first and is freed. Its precision affects only the conditioning
   embedding, not resident denoise memory or DiT compute.
3. The DiT, video VAE and audio VAE are loaded, used and released one at a time.
4. Packed-row count is controlled by a sub-native canvas rather than the ~38k-row 1344x768 shape.
5. Eight to fifteen forwards are enough for coherent output. The released weights are CFG-distilled,
   so a step is one forward, not two.

## Measured tiers

Apple M4 Max, 64 GiB unified memory. Canvas is `width x height`. "Forwards" is DiT evaluations;
the CLI's `--steps` is sigma grid points and is therefore one greater. Totals use a cached prompt
embedding.

| Tier | Canvas | Frames / duration | Forwards | Packed rows | Mean s/step | Total wall | Peak Metal |
|---|---:|---:|---:|---:|---:|---:|---:|
| Wiring proof | 128x128 | 22 / 0.917 s | 2 | 239 | 0.749 | 14.347 s | 38.586 GiB |
| Fast draft | 640x384 | 73 / 3.042 s | 8 | 5,577 | 18.783 | **3:05.050** | 38.890 GiB |
| Quality draft | 640x384 | 73 / 3.042 s | 15 | 5,577 | 18.787 | **5:18.317** | 39.046 GiB |
| Desktop quality | 768x448 | 73 / 3.042 s | 15 | 7,689 | 27.589 | **7:36.314** | 39.527 GiB |
| HQ 5-second | 768x448 | 124 / 5.167 s | 15 | 12,982 | 53.705 | **14:38.669** | 40.221 GiB |
| HQ 10-second | 768x448 | 243 / 10.125 s | 15 | 25,138 | 137.183 | **36:12.107** | 42.635 GiB |

Phase costs outside denoising: the first uncached Q8 text encode takes 10.574 s and peaks at
25.647 GiB, after which it is stored as a roughly 1 MB local cache; BF16 DiT load is 7-9 s; AdaLN
cache construction is 4-5 s; video decode is 22.4 s at 640x384 and 29.7 s at 768x448.

The duration curve is superlinear because both the dense linear work and the full attention grow
with the packed sequence. From 5.17 s to 10.13 s the packed rows nearly double and step time rises
2.55x.

### Recommended defaults

- **Fast:** 640x384, 73 frames, `--steps 9` (8 forwards) — about 3:05.
- **Best short:** 768x448, 73 frames, `--steps 16` (15 forwards) — about 7:36.
- **Best longer-form:** 768x448, 124 frames, `--steps 16` — about 14:39.
- **10 seconds:** 768x448, 243 frames, `--steps 16` — about 36:12; coherent, but batch-oriented
  rather than interactive.

Fifteen forwards improve fine structure over eight modestly; they do not transform the image.
768x448 gives the best result tested: full-body subject, stable limbs and direction of travel,
cleaner ground detail, consistent lighting.

### Geometry limits

The frame count snaps upward to the `17n + 5` grid and both canvas axes must be multiples of 32.
The wrapper advertises 5-15 seconds, but the transformer accepts any legal grid; the real
constraint is the video decoder, which needs at least seven latent frames. That makes **22 pixel
frames the true end-to-end minimum**, not 5.

### Long clips: chained windows

Duration in a single pass is quadratic — the attention term grows with the packed sequence, so the
second five seconds of a clip costs more than the first. `--chain-windows N` renders N windows
instead, each conditioned on the previous window's **last decoded frame** through the ordinary
first-frame keyframe path, dropping the duplicate frame at the join. The marginal cost of each
extra window is then flat.

| Delivered | Route | Forwards | Total wall | Peak Metal |
|---|---|---:|---:|---:|
| 243 frames / 10.125 s | one dense pass | 15 | 36:12.1 | 42.635 GiB |
| 243 frames / 10.125 s | **2 chained windows** | 8 per window | **17:04.9** | **40.254 GiB** |
| 362 frames / 15.083 s | **3 chained windows** | 8 per window | **26:34.0** | **40.236 GiB** |

The forward counts are not matched and the comparison is not a like-for-like speed claim: eight
forwards is the validated setting at the five-second tier and fifteen at ten seconds, and chaining
is what makes eight legitimate at ten seconds, because every pass is a five-second pass. Matched at
fifteen forwards per window, the same 10 s chain projects to about 30:03 (1.20x). What chaining
actually removes is the quadratic: a window that opens on a keyframe costs **8.6% more** than one
that does not (+680 packed rows, plus the still's VAE encode and a text encode the prompt cache
cannot serve), and that surcharge does not grow with clip length. Peak memory stops tracking
duration at all — a chained clip peaks at its window's peak, whatever its length.

Seams were measured, not assumed. On both clips the frame-to-frame luminance step **at the join**
was 0.64-1.25x the median step of its own neighbourhood, and smaller than the largest ordinary
step elsewhere in the same clip; audio is cross-faded over the one frame of real overlap the chain
owns, which drops the sample step at the seam by 7.6-10.5x to about 0.01-0.06x the clip's own
typical slew, with zero A/V drift.

Three honest limitations:

- **The camera can change direction at a seam.** A single still frame carries state, not momentum:
  the next window cannot know which way the camera was travelling. There is no positional jump —
  the discontinuity is in velocity.
- **Every window gets the same prompt**, so a prompt that scripts a spoken line asks for that line
  in each window, and it can be delivered once per window. Per-window prompts are the fix.
- **Audio level is not matched across seams.** Windows generate their ambience independently; one
  frame of overlap removes the click but cannot ramp a level change.

```bash
# 15 seconds as three windows, trimmed to an exact 362 frames
./.venv/bin/python scripts/generate_staged.py '<prompt>' \
  --dit ... --compact-root ... --text-config ... \
  --frames 124 --height 448 --width 768 --steps 9 --seed 161616 \
  --chain-windows 3 --chain-total-frames 362 \
  --output ../outputs/long.mp4 --metrics ../metrics/long.json

# grade the joins afterwards
./.venv/bin/python scripts/seam_report.py ../outputs/long.mp4 \
  --window-frames 124 --windows 3 -o ../outputs/long_seam.png
```

`--chain-windows 1` is the default and is the single-pass path unchanged.

## Reproduction

Fetch only the files this path needs, into `<experiment-root>/models`:

```bash
./.venv/bin/python scripts/download_selected.py --root /path/to/experiment-root
```

Then, with the repository checked out at `<experiment-root>/minimax-h3-mlx`:

```bash
./.venv/bin/python scripts/generate_staged.py \
  'A cinematic tracking shot of a red fox walking through a misty pine forest at sunrise, detailed orange fur moving naturally in the breeze, shallow depth of field, smooth camera motion. Audio: soft footsteps on wet leaves, distant birds, gentle wind through pine branches.' \
  --dit ../models/deepbeep-pruned-bf16/MiniMax-H3-FL2VA-pruned_bf16.safetensors \
  --compact-root ../models/ddalcu-q8 \
  --text-config ../models/upstream-meta/FL2VA/text_encoder/config.json \
  --prompt-cache ../cache/fox_prompt.npz \
  --frames 73 --height 448 --width 768 --steps 16 --seed 42 \
  --output ../outputs/fox_768x448_73f_s16.mp4 \
  --metrics ../metrics/fox_768x448_73f_s16.json
```

For the fastest good tier use `--height 384 --width 640 --steps 9`. For the longer-form tiers change
`--frames` to 124 or 243. The prompt cache is optional; it avoids reloading the 28 GB encoder
between matched runs and refuses to load against a different prompt. Cache, output and metrics
paths are deliberately outside the repository.

Wrapping the command in `caffeinate -dimsu` is worthwhile for the multi-minute tiers.

Weights stay outside the working tree throughout. Nothing in this path writes a checkpoint,
generated frame or metrics file into the repository.

### Environment

Measured with:

```text
Python 3.11.15
mlx 0.32.0 / mlx-metal 0.32.0
mlx-vlm 0.6.8
mlx-lm 0.31.3
transformers 5.14.1
huggingface-hub 1.26.0
safetensors 0.8.0
numpy 2.4.6
pillow 12.3.0
```

`requirements.txt` carries lower bounds rather than pins. `ffmpeg` must be on `PATH` for muxed
output.

## What is not claimed

- **Reference parity is not claimed.** There is no same-configuration CUDA render to compare
  against, and both tested canvases sit below H3's released 768-pixel short-edge envelope. These are
  honest high-quality draft tiers, not native-resolution parity.
- **Native 1344x768 remains impractical here.** Nothing in this path reduces attention FLOPs. Going
  further needs sparse attention, a genuinely distilled few-step checkpoint, or windowed/tiled
  attention designed around the packed multimodal sequence.
- **Prompt adherence is not perfect.** Long-form runs held subject identity, setting, lighting and
  camera axis across 10 seconds while completing a multi-stage object interaction, but individual
  requested actions were simplified.
- **This path is FL2VA only.** No claim is made here that the FL2VA and Ref2VA checkpoints are
  interchangeable, and nothing in this document depends on that being true.
- **Audio level is prompt-driven.** A deliberately ambient prompt produced a quiet mix (max
  -38.8 dB); an explicit dialogue control on the same pipeline peaked at -0.7 dB with a -15.8 dB
  mean, which is what confirms the audio path is healthy rather than attenuated.

## Licence

The port code is Apache-2.0. The weights are not: they are governed by the MiniMax H3 Community
License, which is territorially limited, requires redistributed copies to carry the agreement and
mark modified files, and requires separate authorization above $20M yearly revenue. Every user
fetches weights directly from the original Hub repositories and accepts those terms there.
