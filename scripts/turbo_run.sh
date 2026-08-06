#!/bin/zsh
# One render at a time, under the shared GPU mutex, with the lock released on any exit.
# Usage: turbo_run.sh <logfile> <generate_staged args...>
set -u
ROOT=/Users/salo/AI/projects/hailuo-mlx
LOCK=$ROOT/.gpu_lock
PY=$ROOT/codex/minimax-h3-mlx/.venv/bin/python
HERE=${0:A:h:h}

LOG=$1; shift
mkdir -p "${LOG:h}"

# Never start on top of somebody else's render, lock or no lock.
while pgrep -f generate_staged >/dev/null 2>&1 || pgrep -f hires_refine >/dev/null 2>&1; do
  echo "[lock] a render is already running; waiting" >&2
  sleep 60
done
while ! mkdir "$LOCK" 2>/dev/null; do
  echo "[lock] held by another agent; waiting" >&2
  sleep 60
done
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM

echo "[lock] acquired $(date +%H:%M:%S)" >&2
START=$(date +%s)
caffeinate -dimsu "$PY" "$HERE/scripts/generate_staged.py" "$@" >"$LOG" 2>&1
RC=$?
END=$(date +%s)
echo "[done] rc=$RC wall=$((END-START))s -> $LOG" >&2
exit $RC
