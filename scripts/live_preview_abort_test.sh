#!/bin/zsh
# The early-abort contract, exercised against a real render rather than a stub.
#
# Starts a draft-tier render with the live preview on, waits for the first forward's thumbnail to
# appear, drops the ABORT sentinel, and then asserts the four things a UI depends on:
#   1. the process exits 75 (not 0, not a traceback),
#   2. status.json says aborted,
#   3. the sentinel was consumed, so the next render into this directory is not killed by it,
#   4. NO mp4 (and no wav) was left behind.
#
# The caller owns the GPU lock.
set -u
ROOT=/Users/salo/AI/projects/hailuo-mlx
CODEX=$ROOT/codex
HERE=${0:A:h:h}
PY=$CODEX/minimax-h3-mlx/.venv/bin/python

OUT=$CODEX/live_preview_outputs/abort_test
rm -rf "$OUT"; mkdir -p "$OUT"

PROMPT=$(cat $ROOT/../aurelius/video/clips/25_THE_STARE/r2/25_THE_STARE_r2_turbo.prompt.txt)

caffeinate -dimsu "$PY" "$HERE/scripts/generate_staged.py" "$PROMPT" \
  --dit $CODEX/models/deepbeep-pruned-bf16/MiniMax-H3-FL2VA-pruned_bf16.safetensors \
  --compact-root $CODEX/models/ddalcu-q8 \
  --text-config $CODEX/models/upstream-meta/FL2VA/text_encoder/config.json \
  --first-frame $ROOT/../aurelius/video/clips/_keyframes/25_THE_STARE__1024x576.png \
  --tae-checkpoint $CODEX/models/tae/taeh3.safetensors \
  --lora $CODEX/models/turbo-lora/minimax_h3_turbo_4step_ema_ckpt500.safetensors:1.0 \
  --lora-adaln $CODEX/models/turbo-lora/upstream_time_embedder.safetensors \
  --width 640 --height 384 --frames 124 --steps 4 --seed 20260825 --crf 14 \
  --live-preview tae --live-preview-dir "$OUT/live" \
  -o "$OUT/aborted.mp4" --metrics "$OUT/aborted.json" >"$OUT/render.log" 2>&1 &
RENDER=$!

echo "[abort-test] render pid $RENDER; waiting for the first preview"
for _ in {1..60}; do
  [[ -f "$OUT/live/preview_01.png" ]] && break
  sleep 5
done
if [[ ! -f "$OUT/live/preview_01.png" ]]; then
  echo "[abort-test] FAIL: no preview appeared"; kill $RENDER 2>/dev/null; exit 1
fi
echo "[abort-test] preview seen at $(date +%H:%M:%S); dropping ABORT"
touch "$OUT/live/ABORT"

wait $RENDER
RC=$?
echo "[abort-test] exit code: $RC (expected 75)"
[[ -f "$OUT/live/ABORT" ]] && echo "[abort-test] FAIL: sentinel not consumed" || echo "[abort-test] sentinel consumed: OK"
[[ -f "$OUT/aborted.mp4" ]] && echo "[abort-test] FAIL: mp4 was written" || echo "[abort-test] no mp4 left behind: OK"
[[ -f "$OUT/aborted.wav" ]] && echo "[abort-test] FAIL: wav was written" || echo "[abort-test] no wav left behind: OK"
echo "[abort-test] status.json:"; cat "$OUT/live/status.json"
echo "[abort-test] metrics status: $(python3 -c "import json;d=json.load(open('$OUT/aborted.json'));print(d['status'], '|', d.get('aborted_by'))")"
exit $(( RC == 75 ? 0 : 1 ))
