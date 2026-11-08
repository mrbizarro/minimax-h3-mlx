"""Exact SHA256 contract for streamed language AND vision conditioning.

Run on a Mac with Metal: python tests/test_text_encoder_stream.py
No downloaded weights/tokenizer and no torch dependency. Fixtures exercise the
compact Q8 and upstream floating/sharded layouts through the real mlx-vlm stack.
The equality assertion is byte equality, never allclose or a tolerance.
"""

import gc
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_vlm.models.qwen3_vl.config import TextConfig, VisionConfig
from mlx_vlm.models.qwen3_vl.language import Qwen3VLModel
from mlx_vlm.models.qwen3_vl.vision import VisionModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder
from minimax_h3_mlx.tensor_stream import TensorRange


def fixture(path, quantized):
    raw = {
        "model_type": "qwen3_vl", "image_token_id": 250, "video_token_id": 251,
        "vision_start_token_id": 252, "vision_end_token_id": 253,
        "text_config": {
            "model_type": "qwen3_vl_text", "num_hidden_layers": 6, "hidden_size": 64,
            "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "rms_norm_eps": 1e-6, "vocab_size": 256,
            "rope_theta": 5000000., "max_position_embeddings": 4096,
            "rope_scaling": {"type": "default", "mrope_section": [4, 2, 2]},
        },
        "vision_config": {
            "model_type": "qwen3_vl", "depth": 3, "hidden_size": 64,
            "intermediate_size": 128, "num_heads": 4, "in_channels": 3,
            "patch_size": 4, "spatial_merge_size": 2, "temporal_patch_size": 2,
            "out_hidden_size": 64, "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0, 2],
        },
    }
    (path / "config.json").write_text(json.dumps(raw))
    mx.random.seed(7)
    language = Qwen3VLModel(TextConfig.from_dict(raw["text_config"]))
    vision = VisionModel(VisionConfig.from_dict(raw["vision_config"]))
    if quantized:
        for model in (language, vision):
            nn.quantize(model, bits=8, group_size=64,
                        class_predicate=lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
                        and p != "pos_embed" and m.weight.shape[-1] % 64 == 0)
    weights = {}
    for bucket, model in (("model" if quantized else "model.language_model", language),
                          ("visual" if quantized else "model.visual", vision)):
        for key, value in tree_flatten(model.parameters()):
            if value.dtype != mx.uint32:
                value = value.astype(mx.bfloat16)
            if not quantized and key == "patch_embed.proj.weight":
                value = value.transpose(0, 4, 1, 2, 3)  # upstream Conv3d layout
            weights[f"{bucket}.{key}"] = value
    weights["lm_head.weight"] = mx.zeros((256, 64), dtype=mx.bfloat16)
    mx.eval(weights)
    if quantized:
        mx.save_safetensors(str(path / "text_encoder.safetensors"), weights)
    else:
        # A single layer is deliberately split across shards.
        items = list(weights.items())
        for i in range(2):
            mx.save_safetensors(str(path / f"model-{i}.safetensors"), dict(items[i::2]))


def request(images):
    if not images:
        return mx.array([[3, 7, 9, 11, 12]], dtype=mx.int32), np.ones(5, np.int64), None
    ids = [3, 252] + [250] * 4 + [253, 5, 252] + [250] * 6 + [253, 7]
    tags = np.ones(len(ids), np.int64)
    tags[1:7] = 0
    tags[8:16] = 0
    pixels = np.random.default_rng(9).normal(size=(40, 3 * 2 * 4 * 4)).astype(np.float32)
    return mx.array([ids], dtype=mx.int32), tags, (pixels, np.array([[1, 4, 4], [1, 4, 6]]))


class StreamParityTests(unittest.TestCase):
    def test_exact_embeddings_and_eviction(self):
        for quantized in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                fixture(path, quantized)
                for dtype in (mx.bfloat16, mx.float32):
                    for vision in (False, True):
                        with self.subTest(quantized=quantized, dtype=dtype, vision=vision):
                            resident = MiniMaxH3TextEncoder(path, num_layers=4, dtype=dtype, load_vision=vision)
                            with patch.object(mx, "load", side_effect=AssertionError("whole-shard load")):
                                streamed = MiniMaxH3TextEncoder(path, num_layers=4, dtype=dtype,
                                                               load_vision=vision, stream=True)
                            encoders = [resident, streamed]
                            for enc in encoders:
                                enc.build_request = lambda prompt, images: request(images)
                            outputs = [enc.encode("same prompt", [object(), object()] if vision else None)
                                       for enc in encoders]
                            data = [np.array(out[0].astype(mx.float32)).tobytes() for out in outputs]
                            self.assertEqual(hashlib.sha256(data[0]).hexdigest(), hashlib.sha256(data[1]).hexdigest())
                            np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
                            # Wrappers must not retain a unit's loaded weights after evaluation.
                            keys = [k for k, _ in tree_flatten(encoders[1].language.parameters())]
                            self.assertFalse(any("q_proj.weight" in k or "embed_tokens.weight" in k for k in keys))
                            # A second encode must really reread weights and be byte-identical.
                            reads = []
                            # Count raw reads (TensorRange.numpy), which both the direct path
                            # (TensorRange.mlx) and the next-unit read-ahead go through.
                            original = TensorRange.numpy
                            def read(entry):
                                reads.append(entry.offset)
                                return original(entry)
                            with patch.object(TensorRange, "numpy", read), patch.object(
                                mx, "load", side_effect=AssertionError("whole-shard load")
                            ):
                                again, _ = encoders[1].encode("same prompt", [object()] if vision else None)
                                self.assertEqual(np.array(again.astype(mx.float32)).tobytes(), data[0])
                            self.assertGreater(len(reads), 4)
                            del encoders, outputs, again, resident, streamed, enc
                            gc.collect()
                            mx.clear_cache()

    def test_missing_layer_tensor_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            fixture(path, True)
            file = path / "text_encoder.safetensors"
            weights = mx.load(str(file))
            del weights["model.layers.2.self_attn.q_proj.scales"]
            # Drop weight instead too: ensure this cannot become a silent random layer.
            del weights["model.layers.2.self_attn.q_proj.weight"]
            mx.eval(weights)
            mx.save_safetensors(str(file), weights)
            encoder = MiniMaxH3TextEncoder(path, num_layers=4, load_vision=False, stream=True)
            encoder.build_request = lambda prompt, images: request(images)
            with self.assertRaisesRegex(KeyError, "missing"):
                encoder.encode("same prompt")


if __name__ == "__main__":
    unittest.main()
