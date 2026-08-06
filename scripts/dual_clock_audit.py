"""Does this runner integrate audio on its own shifted schedule, or ride the flat approximation?

larryvrh's README warns that ComfyUI's stock samplers step both streams on ONE schedule, which the
model tolerates at ~20 steps and which blows the audio up at 4. The fix is a dual-clock sampler.
This checks, arithmetically rather than by reading the docstrings, that:

1. our audio sigma grid is *identical* to the reference generator's ``audio_sigma(sigma_v)`` map;
2. our Euler update on the audio stream equals the reference's ``xa += ha * (oa / slope)`` — i.e.
   whether our model output already is the raw audio velocity, or the slope-scaled one a flat
   sampler wants;
3. how far a flat sampler would have been off at 4 steps, so the size of the avoided error is a
   number rather than an adjective.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.config import PipelineConfig          # noqa: E402
from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler   # noqa: E402

SHIFT_V, SHIFT_A = 12.0, 3.0


def shift_sigma(u, shift):
    return shift * u / (1.0 + (shift - 1.0) * u)


def time_shift_sigma(sigma, frm, to):                      # reference generate.py
    base = sigma / (frm + sigma * (1.0 - frm))
    return to * base / (1.0 + (to - 1.0) * base)


def time_shift_slope(sigma, frm, to):
    base = sigma / (frm + sigma * (1.0 - frm))
    return (to * (1.0 + (frm - 1.0) * base) ** 2) / (frm * (1.0 + (to - 1.0) * base) ** 2)


def main() -> int:
    for forwards in (4, 6, 8, 20):
        points = forwards + 1
        cfg = PipelineConfig()
        v, a = MiniMaxH3Scheduler(shift=cfg.sigma_shift_video), MiniMaxH3Scheduler(shift=cfg.sigma_shift_audio)
        v.set_timesteps(points)
        a.set_timesteps(points)
        ours_v = np.array(v.sigmas.tolist(), dtype=np.float64)
        ours_a = np.array(a.sigmas.tolist(), dtype=np.float64)

        ref_v = np.array([shift_sigma(1.0 - i / forwards, SHIFT_V) for i in range(points)])
        ref_a = time_shift_sigma(ref_v, SHIFT_V, SHIFT_A)

        # Step-size comparison, in raw-audio-velocity units.
        #   ours      : x_next = x + (sigma_a - sigma_a_next) * v_data_ward   -> coefficient |ha|
        #   reference : x_next = x + (sigma_a_next - sigma_a) * (o_comfy / slope), and the README
        #               states o_comfy = slope * (raw velocity), so the slope cancels and the
        #               reference's coefficient in raw units is also |ha|.
        # The slope therefore never enters a pipeline whose model emits the raw audio velocity —
        # which reference/diffusers/modular/denoise.py confirms ours does, since it
        # steps `audio_scheduler.step(audio_noise_pred, audio_timesteps[i], ...)` with no division.
        h_ours = ours_a[:-1] - ours_a[1:]
        h_ref = np.abs(ref_a[1:] - ref_a[:-1])
        slope = time_shift_slope(np.maximum(ref_v[:-1], 1e-6), SHIFT_V, SHIFT_A)

        # A flat sampler steps the audio stream with the VIDEO delta and the pre-scaled velocity:
        #   hv * o_comfy = hv * slope * v_raw   against the correct   |ha| * v_raw.
        h_flat = (ours_v[:-1] - ours_v[1:]) * slope
        over = h_flat / h_ours

        print(f"--- {forwards} forwards ({points} sigma points) ---")
        print(f"  video sigmas ours vs reference : max |d| {np.abs(ours_v - ref_v).max():.3e}")
        print(f"  audio sigmas ours vs reference : max |d| {np.abs(ours_a - ref_a).max():.3e}")
        print(f"  audio step size ours vs ref    : max |d| {np.abs(h_ours - h_ref).max():.3e}")
        print(f"  our audio step coefficient     : {np.array2string(h_ours, precision=4)}")
        print(f"  a FLAT sampler would apply     : {np.array2string(h_flat, precision=4)}")
        print(f"  flat/dual per step             : {np.array2string(over, precision=3)}")
        print(f"  flat overshoot: last step {over[-1]:.2f}x, worst {over.max():.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
