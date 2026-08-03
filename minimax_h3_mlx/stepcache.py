"""Step-skipping for the joint denoiser — a TeaCache-class residual cache.

The observation TeaCache is built on is that a diffusion transformer's *output delta* changes
smoothly along the trajectory: consecutive steps often ask the stack to compute something very
close to what it just computed. When that is true the forward can be skipped entirely and the last
delta reused.

Two H3-specific choices:

* **Indicator.** The block-0 AdaLN-modulated input, as in the original. It is the first tensor in
  the stack that carries both the trajectory and the timestep, and producing it costs the two patch
  projections plus a 136-row text refiner — about 0.03% of a forward.
* **Residual.** H3's heads emit a *velocity* in the same shape as the rows they update, so the
  cached quantity is ``prediction - rows`` per modality. Video and audio ride one indicator because
  they ride one forward; skipping is all-or-nothing.

The accumulator is the honest part. Relative L1 between consecutive indicators is summed and the
forward is skipped only while that sum stays under ``threshold``; a real forward resets it. Error
therefore accrues on a budget rather than per-step, and ``max_skip`` caps how long the sampler can
coast even inside the budget. The first and last forwards are never skipped: the first has no
cached delta to reuse, and the last is what actually lands on the output manifold.
"""

from __future__ import annotations

import mlx.core as mx


class StepResidualCache:
    """Decide, per denoising step, whether the transformer forward can be reused."""

    def __init__(
        self,
        threshold: float,
        num_steps: int,
        max_skip: int = 2,
        warmup: int = 1,
        tail: int = 1,
    ):
        if threshold < 0:
            raise ValueError(f"`threshold` must be non-negative, got {threshold}.")
        self.threshold = float(threshold)
        self.num_steps = int(num_steps)
        self.max_skip = int(max_skip)
        self.warmup = int(warmup)
        self.tail = int(tail)

        self._previous: mx.array | None = None
        self._accumulated = 0.0
        self._consecutive = 0
        self._video_delta: mx.array | None = None
        self._audio_delta: mx.array | None = None
        self.history: list[dict] = []

    @property
    def enabled(self) -> bool:
        return self.threshold > 0.0

    @property
    def skipped(self) -> int:
        return sum(1 for item in self.history if item["skipped"])

    def decide(self, index: int, indicator: mx.array) -> bool:
        """Return whether step ``index`` may reuse the cached delta, and update the accumulator."""
        forced = (
            index < self.warmup
            or index >= self.num_steps - self.tail
            or self._video_delta is None
            or self._consecutive >= self.max_skip
        )

        relative = 0.0
        if self._previous is not None:
            numerator = mx.mean(mx.abs(indicator - self._previous))
            denominator = mx.mean(mx.abs(self._previous)) + 1e-8
            relative = float((numerator / denominator).item())
        self._previous = indicator

        if forced:
            skip = False
        else:
            skip = self._accumulated + relative < self.threshold

        if skip:
            self._accumulated += relative
            self._consecutive += 1
        else:
            self._accumulated = 0.0
            self._consecutive = 0

        self.history.append(
            {
                "step": index,
                "relative_l1": round(relative, 5),
                "accumulated": round(self._accumulated, 5),
                "skipped": bool(skip),
                "forced": bool(forced),
            }
        )
        return skip

    def store(self, video_rows: mx.array, audio_rows: mx.array, video_pred: mx.array, audio_pred: mx.array) -> None:
        """Record the delta a real forward produced, in the row shapes the sampler steps."""
        self._video_delta = video_pred - video_rows
        self._audio_delta = audio_pred - audio_rows

    def reuse(self, video_rows: mx.array, audio_rows: mx.array) -> tuple[mx.array, mx.array]:
        """Re-apply the last delta to the current rows."""
        if self._video_delta is None or self._audio_delta is None:
            raise RuntimeError("No cached residual: `decide` must force a real forward first.")
        return video_rows + self._video_delta, audio_rows + self._audio_delta

    def summary(self) -> dict:
        return {
            "threshold": self.threshold,
            "max_skip": self.max_skip,
            "warmup": self.warmup,
            "tail": self.tail,
            "steps": self.num_steps,
            "skipped": self.skipped,
            "history": self.history,
        }
