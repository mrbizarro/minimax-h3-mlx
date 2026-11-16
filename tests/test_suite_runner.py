"""Check suite exit handling without loading models or a graphics runtime."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class SuiteExitStatus(unittest.TestCase):
    def run_fixture(self, mode):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            shutil.copy(Path(__file__).resolve().parents[1] / "scripts/run_tests.sh",
                        root / "scripts/run_tests.sh")
            binaries = root / ".venv/bin"
            binaries.mkdir(parents=True)
            interpreter = binaries / "python"
            interpreter.write_text('''#!/bin/sh
case "$SUITE_FIXTURE:$1" in
  failure:tests/test_dit_parity.py) echo "assertion failed"; exit 7;;
  silent:*) exit 0;;
  filtered:*) echo "Modular warning"; exit 0;;
  *) echo "passed"; exit 0;;
esac
''')
            interpreter.chmod(0o755)
            (binaries / "python3").symlink_to(interpreter)
            env = dict(os.environ, SUITE_FIXTURE=mode, PYTHON=str(interpreter),
                       PATH=str(binaries) + os.pathsep + os.environ.get("PATH", ""))
            return subprocess.run(["bash", "scripts/run_tests.sh"], cwd=root, env=env,
                                  capture_output=True, text=True)

    def test_failed_parity_suite_fails_the_runner(self):
        result = self.run_fixture("failure")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("SOME SUITES FAILED", result.stdout)

    def test_success_with_output_passes(self):
        self.assertEqual(self.run_fixture("normal").returncode, 0)

    def test_silent_success_passes(self):
        self.assertEqual(self.run_fixture("silent").returncode, 0)

    def test_filtered_success_passes(self):
        self.assertEqual(self.run_fixture("filtered").returncode, 0)


if __name__ == "__main__":
    unittest.main()
