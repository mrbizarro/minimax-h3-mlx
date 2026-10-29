# minimax-h3-mlx

MLX (Apple Silicon) port of [**MiniMaxAI/MiniMax-H3**](https://huggingface.co/MiniMaxAI/MiniMax-H3) —
MiniMax's omni-modal generative system for synchronized **video + audio** generation.

> Powered by MiniMax H3.

H3 is **not** a language model. It is a diffusers pipeline: a 33B diffusion transformer denoising
video and audio latents jointly, conditioned by a frozen Qwen3-VL-32B encoder, with separate video
and audio VAEs. There is no autoregressive decoding and no `mlx_lm.convert` path — this repository
is a from-scratch MLX implementation of the pipeline.

## Architecture

| Component | Class | Size | Notes |
|---|---|---|---|
| `transformer` | `MiniMaxH3DiTModel` | 33B / 66.3 GB | 50 blocks, hidden 5376, 56x128 heads (inner dim 7168 > hidden), SwiGLU ffn 14336, 3D MM-RoPE |
| `text_encoder` | Qwen3-VL-32B | 66.7 GB | frozen conditioner; H3 reads the **unnormalized** hidden state after layer 50 of 64 |
| `video_vae` | `MiniMaxH3VideoVAE` | 10.4 GB | ViT+CNN KL VAE, 16x spatial / 4x temporal, 24 latent channels, tiled |
| `audio_vae` | `MiniMaxH3AudioVAE` | 0.6 GB | DAC/BigVGAN stereo 32 kHz, 32 latent channels, 40 Hz latent rate |

Everything runs over **one packed 1-D sequence** holding every modality at once:

```
[ text (L) | keyframe conditions (C) | target audio (A) | target video (V) ]
```

Attention is full self-attention over that sequence — no cross-attention, no per-modality block
weights. Modality-specific behaviour comes only from the two input patch projections, a per-row
AdaLN modality tag, and the two output heads.

### The two checkpoints are the same weights

`FL2VA/` and `Ref2VA/` are **byte-identical except for `model_index.json`** — all 80 weight and
code files share LFS hashes. The 288 GB repository is 144 GB of unique weights published twice;
only the pipeline metadata (`partition`, `tasks`) differs. One conversion covers both tasks.

### AdaLN precompute: 13B of the 33B need not be resident

13B parameters live in the per-block `adaln_proj.linear` projections (50 x `[96768, 2688]`). Their
only input is the timestep embedding — nothing sequence-dependent — so for a fixed sampler schedule
every modulation tensor a run will ever need can be computed once up front and the projections then
dropped. `ModulationCache` builds it and `drop_adaln_weights` frees the originals; the cache is
verified bit-exact against the live projection.

Measured on the real checkpoint, a 40-step run (video and audio use different sigma shifts, 12.0 and
3.0, so their schedules only partly coincide — 77 distinct timesteps in all, not 40):

| | params | resident |
|---|---:|---:|
| DiT as shipped | 33.12B | 66.3 GB |
| `adaln_proj` dropped | -13.01B | -26.0 GB |
| **after** | **20.11B** | **40.3 GB + 745 MB cache** |

A **25.3 GB net saving**, with the cache 35x smaller than the weights it replaces and built in 0.7 s.

The table is built from the float32 timestep MLP through the *unquantized* projections. Every block
reads the same `temb`, so an error there biases all 50 blocks identically at every step and
accumulates coherently along the trajectory — building it before quantization keeps that path exact.


### Encoder truncation: 14 of 64 layers are never evaluated

H3 reads the **unnormalized** hidden state after the 50th of Qwen3-VL-32B's 64 decoder layers and
feeds it straight to the DiT's `condition_proj`. The language-model head, the final norm and layers
50-63 are never touched, so the port loads only what it reads:

| | params | resident |
|---|---:|---:|
| text encoder on disk | — | 66.7 GB |
| layers 0-49 only, no `lm_head`, no vision tower | 25.16B | **50.3 GB** |

506 tensors are skipped. Reading pre-norm is a real distinction, not a detail — the parity test
asserts the returned state *differs* from `last_hidden_state`, since silently returning the normed
output would still look plausible.

Together with the AdaLN precompute, the two structural savings take the resident pipeline from
144 GB to about **102 GB before any quantization**.

## Performance: read this before converting anything

MiniMax has **not** released its sparse-attention implementation ("the initial open-source release
provides inference with full attention only"), so a run does dense attention over tens of thousands
of rows. Measured on an **M3 Ultra (550 GB unified memory)**, bfloat16, one transformer block timed
and multiplied by 50 (the blocks are identical):

| Request | Packed rows | Per block | **Per denoising step** | Peak activations |
|---|---:|---:|---:|---:|
| 5 s, 1344x768 | 37,966 | 10.5 s | **8.8 min** | 9.3 GB |
| 15 s, 1344x768 | 109,318 | 74.9 s | **1.04 h** | 24.4 GB |

A full 5 s generation confirms the extrapolation rather than relying on it: a real step of the
complete pipeline measured **531.6 s (8.86 min)** against the 8.8 min predicted from one block.

Per-step cost is the measured, assumption-free number. The released weights are **CFG-distilled**
("guidance baked into the weights, so there is no guider, no `negative_prompt` and no
`guidance_scale`"), so a step is one forward, not two — but MiniMax does not publish a recommended
step count, and the reference marks `num_inference_steps` required rather than defaulting it. Total
wall-clock therefore scales directly:

| Steps | 5 s clip | 15 s clip |
|---:|---:|---:|
| 8 | 1.2 h | 8.3 h |
| 16 | 2.3 h | 16.6 h |
| 50 (generic diffusers default) | 7.3 h | 52 h |

Peak memory is modest — MLX's attention is flash-style and never materializes the score matrix — so
**memory is not the constraint. Compute is.** 5 s is the shortest clip H3 supports and 15 s at 2K is
its flagship capability; 2K is out of reach locally.

This also changes what quantization buys. The bottleneck is attention FLOPs, which quantization
does not reduce. At 5 s the linear layers are ~42% of the work, at 15 s ~20%, so a 4-bit DiT is
worth roughly 1.2-1.4x end-to-end — useful for *fitting* the model on a smaller Mac, not for making
generation quick.

## Status

| Piece | State |
|---|---|
| DiT (`MiniMaxH3DiT`) | **done** — matches diffusers reference to 4.8e-07 |
| Video VAE | **done** — encode + decode match to 1.2e-06, tiled and untiled |
| Audio VAE | **done** — encode 6.6e-07, decode 2.1e-08 |
| AdaLN precompute + drop | **done** — bit-exact; verified on the real 33B checkpoint |
| Scheduler | **done** — bit-exact sigmas, timesteps and 16-step trajectory |
| Packed-sequence geometry | **done** — bit-exact `(t, h, w)` grid, tags, indices |
| Text encoder | **done** — `hidden_states[50]` matches HF to 5.0e-08 |
| Checkpoint loaders | **done** — all four components load from the release, zero key mismatches |
| Pipeline / denoise loop | **done** — generates prompt-faithful video + synced audio |
| Quant set | not started |

All four components were loaded from the released checkpoint and exercised:

* **DiT** — 33.12B over 534 tensors, every key matched, mixed precision intact (12 float32 tensors
  for the patch projections, timestep MLP and output heads; 522 bfloat16).
* **Audio VAE** — 151.3M params, 40 latents/s at 32 kHz as specified. Round-tripping real signals
  gives **33.2 dB SNR** on a 440 Hz tone and 0.9995 correlation on a decaying note; phase-accurate
  reconstruction is what proves the convolution transposes and the folded weight norm are right.
* **Video VAE** — 2.60B params, 16x spatial / 4x temporal. A 22-frame 128x128 clip round-trips
  through a `(1, 48, 7, 8, 8)` latent at **0.962 correlation / 21.9 dB PSNR**, and the decoded frame
  counts match `video_latent_num_frames` exactly (7 latent -> 22 frames, 12 -> 39) — the packing
  geometry and the VAE's temporal chunking were ported separately and agree.
* **Text encoder** — 25.16B params over 50 layers; a prompt encodes to `(1, N, 5120)` in 0.9 s.

The keyframe (`fl2va`) path is implemented — conditioning frames are encoded through the VAE's
spatial encoder, noised to `t = 0.999` and prepended as rows — but only the text-only `t2va` path has
been run end to end. Two details of the reference are load-bearing there and easy to get wrong: the
keyframe posterior is **sampled** rather than taken at its mode, under a generator seeded with 42
*independently of the request seed*, and the sampled latent is **rounded through float16** before
normalization — about 11 bits of every conditioning latent. MLX's RNG differs from torch's, so that
draw is not bit-identical to the reference's, though everything around it is.

### End to end

```bash
./.venv/bin/python scripts/generate.py "a red fox leaps over a mossy log" -o fox.mp4
```

A first run produces a misty forest with tall trunks, a mossy log in the foreground and an orange
fox form moving across the later frames — semantically faithful to the prompt, with audio muxed in.
Beyond eyeballing it, three properties are what say the wiring is right rather than merely plausible:

* **Temporal coherence** — adjacent frames differ less than distant ones (mean |Δ| 9.6 vs 12.7). A
  pipeline that packed the time axis wrongly would show no such gradient.
* **Stereo coherence** — the two audio channels correlate at **+0.947**: high, but not 1.0. That is
  what real stereo looks like, and it exercises the channel-major audio packing, where the two
  blocks of rows are pinned to opposite ends of the width grid.
* **Duration agreement** — video and audio land within 8 ms of each other (5.167 s vs 5.175 s), the
  residue of the 40 Hz audio latent grid, which is the shared rotary clock doing its job.

Output is written with no extra dependencies: stereo WAV through the standard library's `wave`
module, video piped as raw RGB into `ffmpeg` (with a PNG-sequence fallback if it is absent).

### Validation

Parity is checked against the `minimax-h3` branch of diffusers, not against a re-reading of it. The
MLX model is the source of truth and its parameters are pushed through the **official** conversion
script (`reorder_interleaved_qkv` + `convert_transformer_key`) into the reference module, so the
test exercises the two raw-checkpoint layout quirks the port handles by reshape rather than
assuming them:

* `attn.qkv_proj` rows are **per-head interleaved** — `[h0: q,k,v][h1: q,k,v]...` — so the
  projection output reshapes to `(..., heads, 3, head_dim)`.
* `mlp.fc1` is a fused **`[gate; value]`** SwiGLU projection; the reference computes
  `fc2(silu(gate) * value)`.

Both mean the released checkpoint loads **1:1 with no weight surgery**.

The video VAE is checked the same way, through `convert_video_vae_key`, on the reference's own tiny
CPU-parity config. Its `attn.to_qkv` is interleaved and its `ff.w1` fused exactly like the DiT's.

The audio VAE inverts the port's two departures instead: its MLX weights are converted back to
channels-first, to torch's transposed-conv axis order, and to a reconstructed `weight_g`/`weight_v`
pair (`v = w`, `g = ||w||`), which proves folding weight norm at load is equivalent rather than
assumed. Its recomputed Kaiser-sinc anti-aliasing filters are additionally checked against the ones
the released checkpoint actually ships, matching to 3.0e-08.

```bash
./.venv/bin/python tests/test_dit_parity.py        # 4.8e-07 vs reference
./.venv/bin/python tests/test_video_vae_parity.py  # 1.2e-06, tiled + untiled
./.venv/bin/python tests/test_audio_vae_parity.py  # 6.6e-07 encode, 2.1e-08 decode
./.venv/bin/python tests/test_text_encoder_parity.py  # 5.0e-08 vs transformers
./.venv/bin/python tests/test_packing_parity.py    # 81 checks, all bit-exact
python3 tests/test_dit_smoke.py                    # no torch needed
```

Two places needed care to stay bit-exact, both because a one-ulp difference is observable:

* **`linspace`** — ATen takes a float32 step, splits the range at the halfway point, and evaluates
  `start + step*i` with an FMA. The sigma grid is collapsed by a consecutive-duplicate check, so an
  ulp can change how many sigmas survive and therefore the *number of model evaluations*.
* **Scheduler scalars** — the reference does its arithmetic in float32 tensors. Computing the same
  expressions in Python floats rounds twice and drifts by an ulp per step.

The packed grid is built in NumPy float64 (as the reference does) because video and audio share one
40-units-per-second rotary clock, and that shared clock *is* the audio/video alignment. The
reference notes its temporal span must be summed pairwise, since sequential summation differs in
the last ulp from 16 latent frames onwards.

## Layout

```
minimax_h3_mlx/
  config.py      DiTConfig / PipelineConfig, original checkpoint field names
  dit.py         the 33B diffusion transformer
  adaln.py       ModulationCache, drop_adaln_weights
  scheduler.py   rectified-flow Euler with exponential sigma shift
  packing.py     packed-sequence geometry, patchify/unpatchify, row timesteps
  load.py        checkpoint loading, mixed fp32/bf16 split preserved
  video_vae.py   causal 3D CNN encoder + 36-layer ViT decoder, tiled
  audio_vae.py   DAC encoder + attention projection + BigVGAN decoder
  text_encoder.py Qwen3-VL-32B conditioner, truncated to the 50 layers H3 reads
  pipeline.py    packing, the joint denoise loop, decoding
  media.py       mp4 / wav writing, dependency-free
reference/       upstream sources, for validation only
scripts/         bench_dit.py
tests/           parity + smoke tests
```

## Pika H3 Flex on Apple Silicon

**Pika H3 Flex running natively on Apple Silicon (MLX) — first Mac port.**
[Pika H3 Flex](https://huggingface.co/Pika-Labs/Pika-H3-Flex) (Pika Labs / Mellis, Inc.,
2026-10-29) is a rank-128 LoRA on MiniMax H3 (base revision `42ed227`, whose transformer weights
are byte-identical to the release this port runs) plus the few-step sampler it was distilled for.
Pika ships it for NVIDIA GPUs only. This runner implements the method in its own MLX code:

```
--pika-flex   = --sampler renoise --sigma-knots 1000,750,500,250,100,40,0
                --video-shift 12 --audio-shift 3   (6 forwards)
--lora <pika_h3_flex_fl2va_phosphene.safetensors>:1.0
```

* `--sampler renoise` (`MiniMaxH3RenoiseScheduler`): each forward predicts the clean sample
  `x0 = x_t + sigma * v`, which is re-noised to the next knot with fresh noise instead of an
  Euler blend; the terminal knot is 0, so the last forward's `x0` is the result.
* `--sigma-knots`: base-grid knots pushed through each modality's own shift.
* The adapter is Pika's `fl2va/adapter.safetensors` re-laid out for this DiT (fused per-head qkv,
  SwiGLU halves swapped, float32 to bfloat16). Phosphene's `scripts/fetch_h3_flex.py` builds it
  from Pika's repo at a pinned revision; no weights are redistributed here.
* FL2VA only (text, first frame). Pika's Ref2VA adapter targets the separate Ref2VA trunk.

The adapter is a MiniMax H3 Model Derivative: MiniMax H3 Community License, copyright Mellis,
Inc.; Pika's code is Apache-2.0. NOTICE: "MiniMax H3 is licensed under the MiniMax H3 Community
License Agreement, Copyright © 2026 MiniMax. All Rights Reserved." Powered by MiniMax H3.

## License

The port is Apache-2.0. The **weights** are governed by the
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE),
which is not an open-source licence: redistribution must carry a copy of the agreement, mark
modified files, and display "Powered by MiniMax H3"; commercial use above $20M yearly revenue needs
separate authorization; and the grant is **territorially limited** (worldwide excluding the
Excluded Territories). Any republished weights inherit these terms.
