#!/bin/zsh
# Same mutex protocol as turbo_run.sh, but contending at ~50 ms granularity.
# Reason: the other agent's batch launches its next render within ~100 ms of releasing the lock, so
# a 2 s (let alone 60 s) poll never wins a transition and the slower waiter is starved rather than
# interleaved. The caller sleeps YIELD seconds after each of our renders so the other batch reliably
# takes the alternate slot -- this contends for a fair share, it does not monopolise.
set -u
ROOT=/Users/salo/AI/projects/hailuo-mlx
LOCK=$ROOT/.gpu_lock
PY=$ROOT/codex/minimax-h3-mlx/.venv/bin/python
HERE=${0:A:h:h}
LOG=$1; shift
mkdir -p "${LOG:h}"
while :; do
  if mkdir "$LOCK" 2>/dev/null; then
    # Won the lock. Only proceed once no foreign render is still winding down.
    if pgrep -f generate_staged >/dev/null 2>&1 || pgrep -f hires_refine >/dev/null 2>&1; then
      rmdir "$LOCK" 2>/dev/null; sleep 0.5; continue
    fi
    break
  fi
  sleep 0.05
done
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM
echo "[lock] acquired $(date +%H:%M:%S)" >&2
START=$(date +%s)
caffeinate -dimsu "$PY" "$HERE/scripts/generate_staged.py" "$@" >"$LOG" 2>&1
RC=$?
echo "[done] rc=$RC wall=$(( $(date +%s) - START ))s -> $LOG" >&2
exit $RC
