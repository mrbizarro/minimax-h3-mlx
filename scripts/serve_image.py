#!/usr/bin/env python3
"""Stdio worker for a caller that serializes GPU jobs and evicts before switching engines."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def configure_memory(args):
    import mlx.core as mx
    # MLX defaults to zero wired memory. Retained weights must stay resident
    # to avoid paging delays on the next image; never change the system limit.
    cap = int(mx.device_info()["max_recommended_working_set_size"])
    args.wired_limit_bytes = min(cap, int(getattr(args, "wired_gb", 40) * 1024**3))
    mx.set_wired_limit(args.wired_limit_bytes)
    mx.set_cache_limit(2 * 1024**3)


def serve(args, runner_factory=None):
    from scripts.generate_image import read_requests
    if runner_factory is None:
        from minimax_h3_mlx.ph3_image import PH3ImageRunner
        configure_memory(args)
        runner_factory = PH3ImageRunner
    from minimax_h3_mlx.ph3_image import size_for_megapixels
    from PIL import Image
    root = Path(args.root)
    runner = runner_factory(root, args.dit, root / 'loras/p-h3-image_lora_bf16.safetensors', keep_warm=True,
        retain=getattr(args, 'retain', 'all'), compile_blocks=not getattr(args, 'no_compile', False))
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request.get('op') == 'close':
                return 0
            request_id = request.get('id')
            try:
                # Reuse CLI validation, with one request and no temporary request file.
                ns = argparse.Namespace(batch_json=None, prompt=request.get('prompt',''),
                    output=request.get('output'), metrics=request.get('metrics'),
                    seed=request.get('seed',0), aspect=request.get('aspect','1:1'),
                    mp=request.get('mp',1.5), width=None, height=None)
                checked = read_requests(ns)[0]
                start = time.perf_counter()
                width,height = size_for_megapixels(checked['aspect'],checked['mp'])
                pixels = runner.generate(checked['prompt'],width,height,seed=checked['seed'])
                out=Path(checked['output']); out.parent.mkdir(parents=True,exist_ok=True)
                temp=out.with_suffix('.pending.png'); Image.fromarray(pixels).save(temp); temp.replace(out)
                result=dict(runner.metrics, width=width,height=height,seed=checked['seed'],
                    wired_limit_bytes=getattr(args, 'wired_limit_bytes', None),
                    prompt=checked['prompt'],output=str(out),forwards=runner.metrics.get('forwards',8),request_seconds=time.perf_counter()-start)
                if checked.get('metrics'):
                    path=Path(checked['metrics']); tmp=path.with_suffix('.pending.json')
                    tmp.write_text(json.dumps(result,indent=2)); tmp.replace(path)
                print('PH3_RESULT '+json.dumps(dict(id=request_id,ok=True,metrics=result)),flush=True)
            except Exception as exc:
                print('PH3_RESULT '+json.dumps(dict(id=request_id,ok=False,error=str(exc))),flush=True)
                return 1
    finally:
        runner.close()
    return 0


def main(argv=None, *, runner_factory=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True); parser.add_argument('--dit',required=True)
    parser.add_argument('--retain', choices=('all','dit'), default='all')
    parser.add_argument('--wired-gb', type=float, default=40)
    parser.add_argument('--no-compile', action='store_true')
    args = parser.parse_args(argv)
    if args.wired_gb < 0:
        parser.error('--wired-gb cannot be negative')
    if runner_factory is not None:
        configure_memory(args)
    return serve(args, runner_factory=runner_factory)

if __name__=='__main__': raise SystemExit(main())
