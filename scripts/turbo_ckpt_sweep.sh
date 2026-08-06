#!/bin/zsh
# ckpt500 sweep at the draft tier, 3 forwards — the step count where the only complaint about the
# turbo arm was softness. 640x384, 124 frames, seed 161616, fur-coat protocol, one arm at a time
# behind the shared GPU mutex (turbo_run.sh waits out the showcase batch's renders).
#
# Four arms, because comparing ckpt500 against "preview + absorbed adaLN" would confound two
# changes at once: the re-distributed ckpt500 files carry NO adaLN pairs, so a fair A/B needs a
# preview arm with the adaLN absorption switched off as well.
set -u
ROOT=/Users/salo/AI/projects/hailuo-mlx/codex
HERE=${0:A:h:h}
DIT=$ROOT/models/deepbeep-pruned-bf16/MiniMax-H3-FL2VA-pruned_bf16.safetensors
CR=$ROOT/models/ddalcu-q8
TC=$ROOT/models/upstream-meta/FL2VA/text_encoder/config.json
TE=$ROOT/models/turbo-lora/upstream_time_embedder.safetensors
PREVIEW=$ROOT/models/turbo-lora/minimax_h3_turbo_4step.safetensors
CK=$ROOT/models/turbo-lora-drbaph/minimax_h3_turbo_4step_ckpt500_pruned_comfyui.safetensors
CKE=$ROOT/models/turbo-lora-drbaph/minimax_h3_turbo_4step_ema_ckpt500_pruned_comfyui.safetensors
KEY=$ROOT/opt_out/hdloop/bizarro_furcoat_169_1344.png
OUT=$ROOT/opt_out/turbo
M=$ROOT/opt_metrics
PROMPT="The man in the fur coat faces the camera and says warmly: They said this was impossible. He holds eye contact, wind in the fur. Audio: desert wind, clear dialogue."

common=(--dit $DIT --compact-root $CR --text-config $TC --first-frame $KEY
        --width 640 --height 384 --frames 124 --steps 4 --seed 161616 --crf 14)

run () { local name=$1; shift
  echo "=== $name  $(date +%H:%M:%S) ==="
  "$HERE/scripts/turbo_run.sh" "$OUT/${name}.log" "$PROMPT" "${common[@]}" "$@" \
      -o "$OUT/${name}.mp4" --metrics "$M/turbo_${name}.json" --frames-dir "$OUT/frames_${name}"
}

run T3-preview       --lora "$PREVIEW:1.0" --lora-adaln "$TE"   # incumbent best config
run T3-ckpt500       --lora "$CK:1.0"                          # the challenger
run T3-ckpt500ema    --lora "$CKE:1.0"
run T3-preview-noad  --lora "$PREVIEW:1.0"                     # fair control for the two above
echo "=== sweep done $(date +%H:%M:%S) ==="
