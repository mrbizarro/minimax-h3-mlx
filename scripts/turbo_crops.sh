#!/bin/zsh
# Mouth-and-eye strips for the turbo arms, built with the campaign's own `panel2.py` so the rects
# are fractions of the *content* area and a 640x384 draft arm, a 1152x640 C1p and a 1344x768 R2 all
# show the same piece of the picture at the same magnification.
#
# Frame 60 is the doctrine frame (mid-speech); 30 and 90 are there so a verdict is never taken from
# a single lucky frame.
#   usage: turbo_crops.sh DEST_SUBDIR LABEL=/abs/frames/dir [LABEL=...]
set -eu

ROOT=/Users/salo/AI/projects/hailuo-mlx/codex
PY=$ROOT/minimax-h3-mlx/.venv/bin/python
DEST=$ROOT/opt_out/quality2/turbo/crops/$1; shift
mkdir -p "$DEST"

build() {  # build OUTNAME RECT MAG FRAME LABEL=dir...
  local out=$1 rect=$2 mag=$3 frame=$4; shift 4
  local args=()
  for pair in "$@"; do
    args+=("${pair%%=*}=${pair#*=}/frame_$(printf '%05d' $frame).png")
  done
  $PY $ROOT/opt_out/hdloop/panel2.py "$DEST/${out}_f$(printf '%05d' $frame).png" "$rect" --mag $mag "${args[@]}"
}

for FRAME in 30 60 90; do
  # THE gate: mouth structure mid-speech. Lip line, vermillion border, moustache above the lip.
  build mouth 0.41,0.52,0.60,0.70 4 $FRAME "$@"
  # Second gate: eye structure. Lid line, iris, catchlight, lashes as lashes.
  build eyes  0.38,0.31,0.60,0.45 4 $FRAME "$@"
  # Context, so a "clean mouth" that came with a ruined face is visible.
  build face  0.36,0.26,0.66,0.78 2 $FRAME "$@"
done
echo "wrote panels to $DEST"
