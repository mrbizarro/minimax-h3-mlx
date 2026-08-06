#!/bin/zsh
# Follow-on arms: larryvrh's FULL-MODEL ckpt500 (published after the first listing), which unlike
# drbaph's repack still carries the 51 adaLN pairs — and this runner is the only one that can use
# them. Same protocol as turbo_ckpt_sweep.sh; the GPU mutex serialises against it and the showcase.
set -u
ROOT=/Users/salo/AI/projects/hailuo-mlx/codex
HERE=${0:A:h:h}
DIT=$ROOT/models/deepbeep-pruned-bf16/MiniMax-H3-FL2VA-pruned_bf16.safetensors
CR=$ROOT/models/ddalcu-q8; TC=$ROOT/models/upstream-meta/FL2VA/text_encoder/config.json
TE=$ROOT/models/turbo-lora/upstream_time_embedder.safetensors
CK5=$ROOT/models/turbo-lora/minimax_h3_turbo_4step_ckpt500.safetensors
CK5E=$ROOT/models/turbo-lora/minimax_h3_turbo_4step_ema_ckpt500.safetensors
KEY=$ROOT/opt_out/hdloop/bizarro_furcoat_169_1344.png
OUT=$ROOT/opt_out/turbo; M=$ROOT/opt_metrics
PROMPT="The man in the fur coat faces the camera and says warmly: They said this was impossible. He holds eye contact, wind in the fur. Audio: desert wind, clear dialogue."
common=(--dit $DIT --compact-root $CR --text-config $TC --first-frame $KEY
        --width 640 --height 384 --frames 124 --steps 4 --seed 161616 --crf 14)
run () { local name=$1; shift
  echo "=== $name  $(date +%H:%M:%S) ==="
  "$HERE/scripts/turbo_run.sh" "$OUT/${name}.log" "$PROMPT" "${common[@]}" "$@" \
      -o "$OUT/${name}.mp4" --metrics "$M/turbo_${name}.json" --frames-dir "$OUT/frames_${name}"
}
run T3-ck500full     --lora "$CK5:1.0"  --lora-adaln "$TE"
run T3-ck500full-ema --lora "$CK5E:1.0" --lora-adaln "$TE"
echo "=== sweep2 done $(date +%H:%M:%S) ==="
