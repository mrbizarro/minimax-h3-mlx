# Speed campaign — verdict

## The headline

**The 5-second high-quality tier went from 14:39 to 8:19 — 1.76x, with no visible quality cost.
The 10-second clip did not: 8 forwards renders it in 21:01 instead of 36:12, but puts two
astronauts on screen for the first 2.2 seconds. The honest 10-second number is still 36:12.**

The owner's target was the 10-second clip at 5 minutes. That target is not reachable on this
machine, and the reason is arithmetic rather than engineering — see below. A true 8:19 for the
5-second tier is the result worth having.

## What was kept

| Lever | Verdict | Where it applies |
|---|---|---|
| **8 forwards instead of 15** | **KEPT** | up to ~5 seconds / ~13k packed rows |
| Everything else | rejected | — |

Recommended defaults after the campaign:

- **3.04 s, 768×448, 8 forwards — 4:24** (was 7:36)
- **5.17 s, 768×448, 8 forwards — 8:19** (was 14:39) ← the practical production tier
- **10.13 s, 768×448, 15 forwards — 36:12** (unchanged; 8 forwards is not safe here)

## What was rejected, and why it is worth knowing

| Lever | Result | Cost of finding out |
|---|---|---|
| SDPA audit | Already on `mx.fast.scaled_dot_product_attention` at 13.0 TFLOP/s, 88% of local GEMM peak | 30 min |
| `mx.compile` | **1.005x** | 1 min |
| Q8 / Q4 DiT | **0.86–0.90x — slower**, refuted before building it | 2 min (saved ~1 h + 20 GB) |
| TeaCache-class step reuse | Implemented fully; **loses to uniform step reduction at matched wall clock** | 2 renders |
| Half density + interpolation | 1.91x, but 7 dB quieter dialogue after the `atempo` stretch and half-rate motion | 1 render |
| 6 and 4 forwards | Waxy skin, undefined eyes, hero object never forms | 2 renders |

The single most valuable half hour was Phase 0: two micro-benchmarks (`bench_gemm.py`,
`bench_block.py`) that measured the machine's own ceiling before anything was optimized. They
refuted three of the six planned levers on measurement instead of on a seven-minute render each,
and they produced a cost model accurate to 3% that every projection below rests on.

## Why 5 minutes is not reachable for a 10-second clip

Not opinion — three measured numbers close the door.

1. **The machine's dense-GEMM ceiling is 14.7 TFLOP/s** at H3's block shapes, and the pipeline
   already achieves 13.7–13.8 TFLOP/s across a 4.5x range of sequence lengths. It runs at ~94% of
   local speed of light; 97.7% of a step is GEMM plus SDPA.
2. **One forward at 25,138 packed rows is 1,875 TFLOPs** — 969 in the projections, 906 in
   attention. At the 14.7 TFLOP/s ceiling that is **127.6 s minimum**; measured 137–143 s.
3. **Fixed non-denoise cost is 115 s**, dominated by 101 s of video VAE decode for 243 frames.

So a 300-second budget leaves 185 s of denoising: **1.3 forwards.**

The consequences are worth stating plainly, because they rule out the obvious remedies:

- A validated **4-step distilled checkpoint** — the usual answer — would land at **~11:00**, not
  5:00. Distillation alone cannot get there at this config.
- Making **attention entirely free** (it is 51% of the hero step) leaves 66 s/step: 8 forwards
  becomes 10:44, 4 forwards 6:19. Still not 5:00.
- Even **both together** — a 4-step model with free attention — gives ~4:40, and only by assuming
  two things that do not exist today.

The only levers that actually reach 5 minutes reduce packed rows, and every one of them changes
the deliverable rather than speeding it up:

| Route to ≈5 min | Projected | What you give up |
|---|---:|---|
| 640×384, 243f, 6 fw | 9:54 | resolution, and 6 fw is already *visible* at 3 s |
| 768×448, 124f @ 12 fps, 6 fw | 6:22 | half-rate motion, 2x-stretched dialogue |
| 640×384, 124f @ 12 fps, 6 fw | **4:19** | all of the above at once |

*(Projections from the validated cost model, accurate to 3%; the quality grades attached to them
are measured.)*

## What is next, in order of expected value

1. **Gate an intermediate forward count at 10 seconds.** 8 fails, 15 works; 10–12 forwards is
   untested and would land at 25:45–30:31 (projected). One render answers it.
2. **Sparse or windowed attention.** At the hero config attention is 51% of the step — the largest
   remaining single target, worth up to ~1.6x on its own. MiniMax withheld their pattern, so this
   means designing one against H3's packed multimodal layout and gating it on faces and object
   permanence. A boolean mask will not do: masked SDPA does not reduce FLOPs, so it has to be real
   windowing that reshapes the sequence.
3. **Cheaper video VAE decode.** 101 s of the 10-second render, and 12% of the *optimized*
   5-second render, is decode. It is now a visible share of the budget and has never been tuned.
4. **Q8 for a memory goal, not a speed goal.** It would take the resident DiT from ~38.6 GB to
   ~10 GB and put H3 on a 32 GB machine, at ~12% more time. That is a different product, and a
   good one.

## Reproduce

```bash
cd /Users/salo/AI/projects/hailuo-mlx/codex/opt
caffeinate -dimsu ../minimax-h3-mlx/.venv/bin/python scripts/generate_staged.py '<prompt>' \
  --dit ../models/deepbeep-pruned-bf16/MiniMax-H3-FL2VA-pruned_bf16.safetensors \
  --compact-root ../models/ddalcu-q8 \
  --text-config ../models/upstream-meta/FL2VA/text_encoder/config.json \
  --frames 124 --height 448 --width 768 --steps 9 --seed 314159 \
  --output ../opt_out/clip.mp4 --metrics ../opt_metrics/clip.json
```

`--steps 9` is the kept lever (9 sigma points = 8 forwards). Add `--first-frame <image>` for FL2VA
first-frame conditioning, `--step-cache <threshold>` for the rejected-but-retained step reuse, and
`--playback-fps` for the rejected half-density mode.
