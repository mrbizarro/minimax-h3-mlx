#!/usr/bin/env python3
"""Drive ``serve_staged.py`` over its line-JSON protocol.

Doubles as the executable description of that protocol: spawn the engine, wait for ``ready``, push
one JSON line per job, block on ``done``/``error``, and time each job from the line going in to the
event coming out -- which is the number a panel would actually feel, not an internal phase sum.

    serve_client.py --plan plan.json --summary out.json -- --dit ... --compact-root ...

``plan.json`` is a list of protocol messages, e.g.

    [{"action": "encode",   "id": "e1", "params": {...}},
     {"action": "generate", "id": "j1", "params": {...}}]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ENGINE = Path(__file__).resolve().parent / "serve_staged.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--quiet-logs", action="store_true")
    parser.add_argument("engine_args", nargs=argparse.REMAINDER)
    opts = parser.parse_args()

    engine_args = [a for a in opts.engine_args if a != "--"]
    plan = json.loads(Path(opts.plan).read_text())

    started = time.perf_counter()
    proc = subprocess.Popen(
        [opts.python, str(ENGINE), *engine_args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    results = []
    ready_at = None

    def read_until(*terminal):
        """Consume events until one of ``terminal`` arrives; stream logs through meanwhile."""
        while True:
            line = proc.stdout.readline()
            if not line:
                return {"event": "eof"}
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                print(f"[raw] {line.rstrip()}", flush=True)
                continue
            kind = event.get("event")
            if kind == "log":
                if not opts.quiet_logs:
                    print(f"  | {event.get('line','')}", flush=True)
                continue
            if kind in terminal:
                return event
            print(f"[event] {json.dumps(event)[:400]}", flush=True)

    ready = read_until("ready", "eof")
    ready_at = time.perf_counter() - started
    print(f"[client] ready after {ready_at:.2f}s: {json.dumps(ready)[:300]}", flush=True)

    # Every action has its own terminal event; waiting for "done" after a ping hangs the client.
    TERMINAL = {
        "generate": ("done",),
        "encode": ("encoded",),
        "ping": ("pong",),
        "status": ("status",),
    }

    for message in plan:
        label = message.get("id", message.get("action"))
        sent = time.perf_counter()
        proc.stdin.write(json.dumps(message) + "\n")
        proc.stdin.flush()
        expect = TERMINAL.get(message.get("action"), ("done",))
        event = read_until(*expect, "error", "eof")
        wall = time.perf_counter() - sent
        event["client_wall_seconds"] = round(wall, 3)
        event["request"] = message
        results.append(event)
        print(
            f"[client] {label}: {event.get('event')} in {wall:.1f}s "
            f"(peak {event.get('peak_gib','-')} GiB, "
            f"dit_reloaded={event.get('dit_reloaded','-')}, "
            f"cond_hit={event.get('cond_cache_hit', event.get('cached','-'))})",
            flush=True,
        )
        if event.get("event") in ("error", "eof"):
            print(f"[client] ABORTING: {json.dumps(event)[:2000]}", flush=True)
            break

    try:
        proc.stdin.write(json.dumps({"action": "exit"}) + "\n")
        proc.stdin.flush()
    except BrokenPipeError:
        pass
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()

    summary = {
        "engine_args": engine_args,
        "ready_seconds": round(ready_at, 3),
        "session_seconds": round(time.perf_counter() - started, 3),
        "results": results,
    }
    Path(opts.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(opts.summary).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[client] wrote {opts.summary}", flush=True)
    ok = all(r.get("event") in ("done", "encoded", "pong", "status") for r in results)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
