#!/usr/bin/env bash
# Run the parity suite. The smoke tests need only MLX; the rest compare against the
# `minimax-h3` branch of diffusers and transformers in .venv (see requirements.txt).
set -u
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
fail=0
run() { echo; echo "=== $1 ==="; $PY "$1" 2>&1 | grep -vE "^(Modular|/opt/homebrew.*Warning|  WeightNorm)" || fail=1; }

python3 tests/test_dit_smoke.py || fail=1
python3 tests/test_video_vae_smoke.py || fail=1
python3 tests/test_draft_speed.py || fail=1
python3 tests/test_live_preview.py || fail=1
run tests/test_chain_stitch.py
run tests/test_dit_parity.py
run tests/test_video_vae_parity.py
run tests/test_audio_vae_parity.py
run tests/test_text_encoder_parity.py
run tests/test_quant_roundtrip.py
run tests/test_packing_parity.py
echo
[ $fail -eq 0 ] && echo "ALL SUITES PASSED" || echo "SOME SUITES FAILED"
exit $fail
