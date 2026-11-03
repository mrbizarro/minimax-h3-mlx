"""P-H3-Image's text side, in MLX: Qwen3-VL-4B hidden states -> text connector -> DiT conditioning.

The request is presented to the 4B encoder verbatim (no chat template, no special tokens). Five
hidden states of the *whole* stack are read (Diffusers numbering 13, 19, 25, 31 and 36, where 36 is
the final-normed state), concatenated, standardised, projected, de-standardised, and the first
token (an attention sink with the same conditioning for every prompt) is replaced by a learned
embedding.

An edit prefixes every image with ``"<Picture i>: "`` plus its vision block (rows tagged *video*)
and starts the instruction with ``"Edit <Picture 1>. "`` unless the caller already did.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open

from .text_encoder import MiniMaxH3TextEncoder

DEFAULT_TAPS = (13, 19, 25, 31, 36)
EDIT_PREFIX = "Edit <Picture 1>. "
_EDIT_STARTS = ("Edit <Picture 1>.", "subject_definitions:")


class PH3ImageTextConnector:
    """``(x - mean) / std -> Linear(no bias) -> * std + mean``, then the sink row."""

    def __init__(self, tensors: dict[str, mx.array]):
        self.input_mean = tensors["input_mean"].astype(mx.float32)
        self.input_std = tensors["input_std"].astype(mx.float32)
        self.proj = tensors["proj.weight"].astype(mx.float32)  # (out, in)
        self.output_mean = tensors["output_mean"].astype(mx.float32)
        self.output_std = tensors["output_std"].astype(mx.float32)
        self.sink = tensors["sink_embedding"].astype(mx.float32)

    @classmethod
    def from_dir(cls, directory: str | Path) -> "PH3ImageTextConnector":
        directory = Path(directory)
        path = directory / "diffusion_pytorch_model.safetensors"
        if not path.exists():
            path = directory / "model.safetensors"
        return cls(mx.load(str(path)))

    def __call__(self, tapped: mx.array) -> mx.array:
        """``tapped``: ``(1, N, 5 * 2560)`` concatenated hidden states -> ``(1, N, 5120)``."""
        x = (tapped.astype(mx.float32) - self.input_mean) / self.input_std
        y = (x @ self.proj.T) * self.output_std + self.output_mean
        return mx.concatenate([mx.broadcast_to(self.sink, (y.shape[0], 1, y.shape[-1])), y[:, 1:]], axis=1)


class PH3ImageTextEncoder(MiniMaxH3TextEncoder):
    """Qwen3-VL-4B with the five taps and the connector: ``encode`` returns ready DiT conditioning."""

    def __init__(self, root: str | Path, dtype: mx.Dtype = mx.bfloat16, load_vision: bool = False,
                 verbose: bool = False, connector_dir: str | Path | None = None):
        root = Path(root)
        cfg = json.loads((root / "text_connector" / "config.json").read_text()) if (root / "text_connector" / "config.json").exists() else {}
        taps = tuple(cfg.get("hidden_state_indices", DEFAULT_TAPS))
        with open(root / "text_encoder" / "config.json") as fh:
            layers = json.load(fh)["text_config"]["num_hidden_layers"]
        super().__init__(root / "text_encoder", num_layers=layers, dtype=dtype, load_vision=load_vision,
                         verbose=verbose, taps=taps)
        self.connector = PH3ImageTextConnector.from_dir(connector_dir or root / "text_connector")

    def build_request(self, prompt: str, images: list | None = None):
        if images and not prompt.lstrip().startswith(_EDIT_STARTS):
            prompt = EDIT_PREFIX + prompt
        return super().build_request(prompt, images)

    def encode(self, prompt: str, images: list | None = None):
        tapped, tags = super().encode(prompt, images)
        cond = self.connector(tapped)
        mx.eval(cond)
        return cond, tags
