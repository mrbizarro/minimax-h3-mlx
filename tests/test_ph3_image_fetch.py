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


class TestDecoderReuse(unittest.TestCase):
    """A second install never fetches the 5.6 GB original again, and a wrong decoder is rebuilt."""

    def _run(self, root, decoder_bytes, converted_sha):
        import tempfile, hashlib
        from unittest import mock
        fetched, converted = [], []
        (root / "vae").mkdir(parents=True, exist_ok=True)
        if decoder_bytes is not None:
            (root / "vae" / F.DECODER_NAME).write_bytes(decoder_bytes)
        real_sha = hashlib.sha256(decoder_bytes or b"").hexdigest()

        def fake_fetch(repo, rev, rel, dest, size, sha):
            fetched.append(rel)
            dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(b"x")
            return dest

        def fake_convert(src, cfg, dest):
            converted.append(src); dest.write_bytes(b"built"); return converted_sha

        with mock.patch.object(F, "fetch", fake_fetch), mock.patch.object(F, "convert_decoder", fake_convert), \
                mock.patch.object(F, "DECODER_SHA256", real_sha if decoder_bytes is not None else "f" * 64):
            rc = F.main(["--dir", str(root), "--accepted"])
        return rc, fetched, converted

    def test_converted_decoder_skips_the_source(self):
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            rc, fetched, converted = self._run(Path(t), b"decoder", None)
            self.assertEqual(rc, 0)
            self.assertNotIn(F.VAE_SOURCE, fetched)
            self.assertEqual(converted, [])

    def test_a_conversion_with_the_wrong_sha_is_refused_and_removed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            rc, fetched, converted = self._run(Path(t), None, "0" * 64)
            self.assertEqual(rc, 3)
            self.assertIn(F.VAE_SOURCE, fetched)
            self.assertFalse((Path(t) / "vae" / F.DECODER_NAME).exists())

    def test_decoder_sha_is_pinned(self):
        self.assertRegex(F.DECODER_SHA256, r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main(verbosity=2)
