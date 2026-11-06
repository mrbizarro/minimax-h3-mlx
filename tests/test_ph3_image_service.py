import argparse
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scripts.serve_image import serve


class WorkerProtocolTests(unittest.TestCase):
    def test_two_requests_share_runner_and_keep_metrics_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); made=[]
            class FakeRunner:
                def __init__(self,*a,**kw):
                    self.metrics={};self.calls=[];self.closed=False;made.append(self)
                    assert kw['keep_warm']
                def generate(self,prompt,width,height,seed):
                    self.calls.append((prompt,width,height,seed));self.metrics={'warm_hit':len(self.calls)>1}
                    return np.zeros((height,width,3),np.uint8)
                def close(self):self.closed=True
            requests=[dict(id=str(i),prompt=f'prompt {i}',mp=.01,aspect='1:1',seed=7+i,
                output=str(root/f'{i}.png'),metrics=str(root/f'{i}.json')) for i in range(2)]
            output=io.StringIO()
            with patch('sys.stdin',io.StringIO('\n'.join(map(json.dumps,requests))+'\n')),patch('sys.stdout',output):
                self.assertEqual(serve(argparse.Namespace(root=temp,dit=temp),FakeRunner),0)
            self.assertEqual(len(made),1);self.assertTrue(made[0].closed)
            self.assertEqual([x[-1] for x in made[0].calls],[7,8])
            self.assertFalse(json.loads((root/'0.json').read_text())['warm_hit'])
            self.assertTrue(json.loads((root/'1.json').read_text())['warm_hit'])
            receipts=[json.loads(x[len('PH3_RESULT '):]) for x in output.getvalue().splitlines()]
            self.assertEqual([r['id'] for r in receipts],['0','1'])
    def test_invalid_request_fails_before_generate_and_closes(self):
        made=[]
        class FakeRunner:
            def __init__(self,*a,**kw):made.append(self);self.closed=False
            def generate(self,*a,**kw):raise AssertionError('invalid request reached model')
            def close(self):self.closed=True
        with patch('sys.stdin',io.StringIO('{"id":"bad","prompt":"","output":"out.png"}\n')),patch('sys.stdout',io.StringIO()) as out:
            self.assertEqual(serve(argparse.Namespace(root='.',dit='.'),FakeRunner),1)
        self.assertTrue(made[0].closed)
        self.assertFalse(json.loads(out.getvalue().split('PH3_RESULT ')[1])['ok'])

if __name__=='__main__':unittest.main()
