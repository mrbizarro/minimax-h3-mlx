"""transformers-5 ``rope_parameters`` configs (P-H3-Image's Qwen3-VL) load like the old form (H3)."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minimax_h3_mlx.text_encoder import _legacy_rope  # noqa: E402


class TestLegacyRope(unittest.TestCase):
    def test_new_form_is_rewritten(self):
        raw = {"rope_parameters": {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_theta": 5000000, "rope_type": "default"}}
        out = _legacy_rope(dict(raw))
        self.assertEqual(out["rope_theta"], 5000000)
        self.assertEqual(out["rope_scaling"], {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"})
        self.assertNotIn("rope_parameters", out)

    def test_old_form_unchanged(self):
        raw = {"rope_scaling": {"mrope_section": [24, 20, 20], "rope_type": "default"}, "rope_theta": 5000000}
        self.assertEqual(_legacy_rope(dict(raw)), raw)


if __name__ == "__main__":
    unittest.main(verbosity=2)
