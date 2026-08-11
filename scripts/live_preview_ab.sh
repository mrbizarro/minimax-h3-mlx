#!/bin/zsh
# Experiment 5 proof harness: the same draft-tier render with --live-preview off, then on.
#
# The two arms differ in exactly one flag. Everything the render consumes — prompt, keyframe, DiT,
# LoRA, canvas, frame count, sigma points, seed — is shared, so a difference in the delivered mp4
# can only come from the flag. The delivery decode is the dense/HQ video VAE, i.e. the path release
# validation hashes, not the TAE draft decoder.
#
# The caller owns the GPU lock (mkdir $ROOT/.gpu_lock). This script deliberately does not take it:
# it is meant to run inside one held lock so the two arms cannot be separated by someone else's
# render.
set -u
ROOT=/Users/salo/AI/projects/hailuo-mlx
CODEX=$ROOT/codex
HERE=${0:A:h:h}
PY=$CODEX/minimax-h3-mlx/.venv/bin/python

DIT=$CODEX/models/deepbeep-pruned-bf16/MiniMax-H3-FL2VA-pruned_bf16.safetensors
CR=$CODEX/models/ddalcu-q8
TC=$CODEX/models/upstream-meta/FL2VA/text_encoder/config.json
TAE=$CODEX/models/tae/taeh3.safetensors
LORA=$CODEX/models/turbo-lora/minimax_h3_turbo_4step_ema_ckpt500.safetensors
TE=$CODEX/models/turbo-lora/upstream_time_embedder.safetensors
KEY=$ROOT/../aurelius/video/clips/_keyframes/25_THE_STARE__1024x576.png
PROMPT_FILE=$ROOT/../aurelius/video/clips/25_THE_STARE/r2/25_THE_STARE_r2_turbo.prompt.txt

OUT=$CODEX/live_preview_outputs
mkdir -p "$OUT"
PROMPT=$(cat "$PROMPT_FILE")

common=(--dit $DIT --compact-root $CR --text-config $TC --first-frame $KEY
        --tae-checkpoint $TAE
        --lora "$LORA:1.0" --lora-adaln $TE
        --width 640 --height 384 --frames 124 --steps 4 --seed 20260825 --crf 14)

run () {  # run <name> <extra args...>
  local name=$1; shift
  echo "=== $name  $(date +%H:%M:%S) ==="
  caffeinate -dimsu "$PY" "$HERE/scripts/generate_staged.py" "$PROMPT" "${common[@]}" "$@" \
      -o "$OUT/${name}.mp4" --metrics "$OUT/${name}.json" >"$OUT/${name}.log" 2>&1
  echo "[done] rc=$? $(date +%H:%M:%S) -> $OUT/${name}.log"
}

# ARMS lets a re-run repeat one arm only; the OFF arm is deterministic and has been reproduced
# byte-for-byte across independent runs, so re-proving a code change only needs the ON arm.
ARMS=${ARMS:-both}
[[ $ARMS == both || $ARMS == off ]] && run lp_off
[[ $ARMS == both || $ARMS == on ]] && run lp_on --live-preview tae --live-preview-dir "$OUT/live_on"
echo "=== A/B done $(date +%H:%M:%S) ==="
