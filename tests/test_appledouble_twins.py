"""AppleDouble twins must never end a render (2026-09-17).

macOS keeps a `._name` twin beside every file on an exFAT/SMB/NTFS volume — the
kind of external drive people keep 75 GB of weights on. `pathlib.Path.glob`
matches those twins (unlike `glob.glob`), and they are binary: reading one as
text raises UnicodeDecodeError, which is a ValueError and NOT a
json.JSONDecodeError. In the fleet that surfaced as
`'utf-8' codec can't decode byte 0xb0 in position 37: invalid start byte`
ending renders on three installs, and as a "shard" the loader could not read.
"""

import json
import tempfile
import unittest
from pathlib import Path

from minimax_h3_mlx.draft_cache import DraftCache
from minimax_h3_mlx.load import shard_paths

# The first 48 bytes macOS writes into an AppleDouble header: magic, version,
# "Mac OS X" filler, entry table. Byte 37 is where the decode blew up.
APPLEDOUBLE = bytes([0x00, 0x05, 0x16, 0x07, 0x00, 0x02, 0x00, 0x00]) + b"Mac OS X        " \
    + bytes([0x00, 0x02, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00,
             0x00, 0x32, 0x00, 0x00, 0x0E, 0xB0, 0x00, 0x00,
             0x00, 0x02, 0x00, 0x00, 0x00, 0x52, 0x00, 0x00]) + b"\x00" * 32


class TwinsAreNotData(unittest.TestCase):
    def test_text_cache_pruning_ignores_the_twin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = DraftCache(root)
            requests = root / "text_requests"
            requests.mkdir(parents=True, exist_ok=True)
            real = requests / "abc.json"
            real.write_text(json.dumps({"text_embed_sha256": "deadbeef"}))
            twin = requests / "._abc.json"
            twin.write_bytes(APPLEDOUBLE)
            (root / "text").mkdir(parents=True, exist_ok=True)
            kept = root / "text" / "deadbeef.npz"
            kept.write_bytes(b"not really an npz")
            cache._prune_unreferenced_text()          # must not raise
            self.assertTrue(real.is_file(), "a valid request was deleted")
            self.assertTrue(kept.is_file(), "a referenced embedding was pruned")
            self.assertTrue(twin.is_file(), "the twin is not ours to delete")

    def test_shard_discovery_skips_the_twin(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            (model / "model-00001-of-00001.safetensors").write_bytes(b"x")
            (model / "._model-00001-of-00001.safetensors").write_bytes(APPLEDOUBLE)
            self.assertEqual([p.name for p in shard_paths(model)],
                             ["model-00001-of-00001.safetensors"])


if __name__ == "__main__":
    unittest.main()
