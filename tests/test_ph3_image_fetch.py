"""Pins and licence behaviour of scripts/fetch_ph3_image.py (no network)."""

import importlib.util
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fetch_ph3_image", ROOT / "scripts" / "fetch_ph3_image.py")
F = importlib.util.module_from_spec(spec)
sys.modules["fetch_ph3_image"] = F
spec.loader.exec_module(F)


class TestPins(unittest.TestCase):
    def test_revisions_are_full_shas(self):
        for rev in (F.DIFFUSERS_REVISION, F.COMFY_REVISION):
            self.assertRegex(rev, r"^[0-9a-f]{40}$")

    def test_every_big_file_has_a_sha_and_size(self):
        for rel, (size, sha) in {**F.CORE_FILES, **F.UPSAMPLER_FILES}.items():
            self.assertGreater(size, 0, rel)
            if rel.endswith(".safetensors") or rel.endswith("tokenizer.json"):
                self.assertRegex(sha or "", r"^[0-9a-f]{64}$", rel)
        self.assertRegex(F.LORA_SHA256, r"^[0-9a-f]{64}$")

    def test_licence_files_are_fetched_beside_the_weights(self):
        for name in ("LICENSE", "LICENSE-APACHE", "NOTICE"):
            self.assertIn(name, F.CORE_FILES)

    def test_refuses_without_acceptance(self):
        self.assertEqual(F.main(["--dir", "/nonexistent-should-not-be-created"]), 2)
        self.assertFalse(Path("/nonexistent-should-not-be-created").exists())

    def test_modification_notice_names_minimax(self):
        self.assertIn("MiniMax H3 is licensed under the MiniMax H3 Community License Agreement", F.MODIFICATION_NOTICE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
