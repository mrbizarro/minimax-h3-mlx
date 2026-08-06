#!/bin/zsh
# Owner-grading encode for a turbo arm — byte-for-byte the campaign recipe
# (`opt_out/quality2/upscale_ladder/encode_ladder.sh`, reused by the sparsity and fusion campaigns):
# lossless PNG -> libx264 CRF 14 preset slow, BT.709 at container *and* frame level with a trailing
# `setparams`, audio peak-normalised to -1 dB, never `-shortest` with copied audio.
#   usage: turbo_encode.sh <arm-name>     (reads opt_out/turbo/{frames_<name>,<name>.wav})
set -eu

NAME=$1
ROOT=/Users/salo/AI/projects/hailuo-mlx/codex
FRAMES=$ROOT/opt_out/turbo/frames_$NAME
WAV=$ROOT/opt_out/turbo/$NAME.wav
DEST=$ROOT/opt_out/quality2/turbo
OUT=$DEST/${NAME}_bt709.mp4

mkdir -p "$DEST"
[ -d "$FRAMES" ] || { echo "missing frames: $FRAMES" >&2; exit 2; }
[ -f "$WAV" ] || { echo "missing audio: $WAV" >&2; exit 2; }

PEAK=$(ffmpeg -hide_banner -i "$WAV" -af volumedetect -f null - 2>&1 | awk '/max_volume/{print $5}')
GAIN=$(python3 -c "print(f'{-1.0 - ($PEAK):.2f}')")

ffmpeg -y -v error -framerate 24 -i "$FRAMES/frame_%05d.png" -i "$WAV" \
  -vf "scale=out_color_matrix=bt709:out_range=tv,setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709" \
  -af "volume=${GAIN}dB" \
  -c:v libx264 -crf 14 -preset slow -pix_fmt yuv420p \
  -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv \
  -c:a aac -b:a 192k -movflags +faststart "$OUT"

ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,avg_frame_rate,color_range,color_space,color_transfer,color_primaries \
  -of default=nw=1 "$OUT"
echo "$OUT"
