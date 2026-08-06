#!/bin/zsh
# Draft-tier probe ladder for the turbo LoRA. 640x384, 124 frames, seed 161616, fur-coat protocol.
# One arm at a time under the shared GPU mutex; the 6-forward D0 anchor is NOT re-rendered.
set -u
ROOT=/Users/salo/AI/projects/hailuo-mlx/codex
HERE=${0:A:h:h}
DIT=$ROOT/models/deepbeep-pruned-bf16/MiniMax-H3-FL2VA-pruned_bf16.safetensors
CR=$ROOT/models/ddalcu-q8
TC=$ROOT/models/upstream-meta/FL2VA/text_encoder/config.json
TURBO=$ROOT/models/turbo-lora/minimax_h3_turbo_4step.safetensors
EMA=$ROOT/models/turbo-lora/minimax_h3_turbo_4step_ema.safetensors
KEY=$ROOT/opt_out/hdloop/bizarro_furcoat_169_1344.png
OUT=$ROOT/opt_out/turbo
M=$ROOT/opt_metrics
PROMPT="The man in the fur coat faces the camera and says warmly: They said this was impossible. He holds eye contact, wind in the fur. Audio: desert wind, clear dialogue."

common=(--dit $DIT --compact-root $CR --text-config $TC --first-frame $KEY
        --width 640 --height 384 --frames 124 --steps 5 --seed 161616 --crf 14)

run () {  # run <name> <extra args...>
  local name=$1; shift
  echo "=== $name  $(date +%H:%M:%S) ==="
  "$HERE/scripts/turbo_run.sh" "$OUT/${name}.log" "$PROMPT" "${common[@]}" "$@" \
      -o "$OUT/${name}.mp4" --metrics "$M/turbo_${name}.json" --frames-dir "$OUT/frames_${name}"
}

run D-base4
run D-turbo4 --lora "$TURBO:1.0" --lora-audit
run D-ema4   --lora "$EMA:1.0"
echo "=== ladder done $(date +%H:%M:%S) ==="
