#!/bin/zsh
# ckpt500 sweep, 3 forwards, draft tier, in decision order. Uses the fast-poll runner and yields
# 25 s after each arm so the showcase batch reliably takes the alternate slot.
set -u
ROOT=/Users/salo/AI/projects/hailuo-mlx/codex
HERE=${0:A:h:h}
DIT=$ROOT/models/deepbeep-pruned-bf16/MiniMax-H3-FL2VA-pruned_bf16.safetensors
CR=$ROOT/models/ddalcu-q8; TC=$ROOT/models/upstream-meta/FL2VA/text_encoder/config.json
TE=$ROOT/models/turbo-lora/upstream_time_embedder.safetensors
PREVIEW=$ROOT/models/turbo-lora/minimax_h3_turbo_4step.safetensors
CK5=$ROOT/models/turbo-lora/minimax_h3_turbo_4step_ckpt500.safetensors
CK5E=$ROOT/models/turbo-lora/minimax_h3_turbo_4step_ema_ckpt500.safetensors
CKD=$ROOT/models/turbo-lora-drbaph/minimax_h3_turbo_4step_ckpt500_pruned_comfyui.safetensors
KEY=$ROOT/opt_out/hdloop/bizarro_furcoat_169_1344.png
OUT=$ROOT/opt_out/turbo; M=$ROOT/opt_metrics
PROMPT="The man in the fur coat faces the camera and says warmly: They said this was impossible. He holds eye contact, wind in the fur. Audio: desert wind, clear dialogue."
common=(--dit $DIT --compact-root $CR --text-config $TC --first-frame $KEY
        --width 640 --height 384 --frames 124 --steps 4 --seed 161616 --crf 14)
run () { local name=$1; shift
  [ -f "$OUT/${name}.mp4" ] && { echo "=== $name already done, skipping ==="; return 0; }
  echo "=== $name  $(date +%H:%M:%S) ==="
  "$HERE/scripts/turbo_run_fast.sh" "$OUT/${name}.log" "$PROMPT" "${common[@]}" "$@" \
      -o "$OUT/${name}.mp4" --metrics "$M/turbo_${name}.json" --frames-dir "$OUT/frames_${name}"
  sleep 25   # yield a slot to the showcase batch
}
run T3-preview       --lora "$PREVIEW:1.0" --lora-adaln "$TE"   # incumbent = what the showcase runs
run T3-ck500full     --lora "$CK5:1.0"     --lora-adaln "$TE"   # challenger, full 259/259
run T3-ck500full-ema --lora "$CK5E:1.0"    --lora-adaln "$TE"
run T3-ckpt500       --lora "$CKD:1.0"                          # ckpt500 minus adaLN (ComfyUI path)
run T3-preview-noad  --lora "$PREVIEW:1.0"                      # preview minus adaLN
echo "=== sweep done $(date +%H:%M:%S) ==="
