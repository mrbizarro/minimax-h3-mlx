"""CPU-only batch contract; numerical warm/cold parity is in the GPU suite."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('image_cli', Path(__file__).resolve().parents[1] / 'scripts/generate_image.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class BatchContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        lora = self.root / 'loras/p-h3-image_lora_bf16.safetensors'
        lora.parent.mkdir(); lora.touch()
        self.args = argparse.Namespace(root=str(self.root), dit='dit', lora=None, lora_scale=1.0,
            batch_json=str(self.root / 'requests.json'), mp=2, aspect='1:1', seed=7, keep_adaln=False)
        self.requests = [dict(prompt=f'prompt {i}', output=str(self.root / f'{i}.png'),
            metrics=str(self.root / f'{i}.json'), seed=7+i) for i in range(2)]

    def read(self, requests):
        Path(self.args.batch_json).write_text(json.dumps(requests))
        return cli.read_requests(self.args)

    def test_duplicate_outputs_rejected_before_loading(self):
        self.requests[1]['output'] = self.requests[0]['metrics']
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.read(self.requests)

    def test_invalid_later_request_rejects_whole_batch(self):
        self.requests[1]['width'] = 64
        with self.assertRaisesRegex(ValueError, 'together'):
            self.read(self.requests)
        self.assertFalse((self.root / '0.png').exists())

    def test_empty_or_oversized_batch_rejected(self):
        for batch in ([], self.requests * 9, {}):
            with self.assertRaisesRegex(ValueError, 'between'):
                self.read(batch)

    def run_fake(self, fail=False):
        calls = []
        class Runner:
            def __init__(self, *a, **kw):
                calls.append(('load', kw['keep_warm']))
            def generate(self, prompt, width, height, **kw):
                calls.append(('generate', prompt, kw['seed']))
                if fail and prompt == 'prompt 1':
                    raise RuntimeError('decoder failed')
                self.metrics = {'total_seconds': 1.25, 'warm_hit': prompt == 'prompt 1'}
                return b'pixels'
            def close(self):
                calls.append(('close',))
        class Pixels:
            def save(self, path):
                Path(path).write_bytes(b'PNG test double')
        engine = types.ModuleType('minimax_h3_mlx.ph3_image')
        engine.PH3ImageRunner = Runner
        engine.size_for_megapixels = lambda *a: (64, 64)
        pil = types.ModuleType('PIL')
        pil.Image = types.SimpleNamespace(fromarray=lambda pixels: Pixels())
        with patch.dict(sys.modules, {'minimax_h3_mlx.ph3_image': engine, 'PIL': pil}):
            if fail:
                with self.assertRaisesRegex(RuntimeError, 'decoder failed'):
                    cli.run_requests(self.args, self.read(self.requests))
            else:
                self.assertEqual(cli.run_requests(self.args, self.read(self.requests)), 0)
        return calls

    def test_one_runner_multiple_prompts_metrics_and_release(self):
        calls = self.run_fake()
        self.assertEqual(calls, [('load', True), ('generate', 'prompt 0', 7),
                                ('generate', 'prompt 1', 8), ('close',)])
        a, b = [json.loads((self.root / f'{i}.json').read_text()) for i in range(2)]
        self.assertEqual((a['seed'], b['seed']), (7, 8))
        self.assertFalse(a['warm_hit']); self.assertTrue(b['warm_hit'])
        self.assertGreaterEqual(b['request_seconds'], 0)

    def test_failure_releases_runner_and_does_not_emit_success_metrics(self):
        calls = self.run_fake(fail=True)
        self.assertEqual(calls[-1], ('close',))
        self.assertTrue((self.root / '0.png').exists())
        self.assertFalse((self.root / '1.json').exists())

if __name__ == '__main__':
    unittest.main()
