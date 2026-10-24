#!/usr/bin/env python3
"""MOCK=1 stand-in for the served engine (CPU only, no model, no GPU), for runner/run_p3b.sh session tests.

usage: mock_server_p3b.py ATTEMPT_DIR PORT
Serves /v1/models and /v1/chat/completions (sleeps max_tokens / MOCK_TOK_S seconds, MOCK_TOK_S default 400,
returns finish "length"), tracks concurrency and seeds (ATTEMPT_DIR/mock-stats.json), and plays the capture side:
ATTEMPT_DIR/mock-status.json gets quotas_met = true after MOCK_QUOTA_AFTER completed requests (default 6; 0 =
never), and the fabricated capture (tests/out/fake, env MOCK_FAKE) is copied to ATTEMPT_DIR/mock-caplog and
ATTEMPT_DIR/mock-private so the runner's copy-out and coverage steps run on real-format files.
"""
import json
import os
import shutil
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

AO, PORT = sys.argv[1], int(sys.argv[2])
TOK_S = float(os.environ.get("MOCK_TOK_S", "400"))
QUOTA_AFTER = int(os.environ.get("MOCK_QUOTA_AFTER", "6"))
FAKE = os.environ.get("MOCK_FAKE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "fake"))
lock = threading.Lock()
st = dict(inflight=0, peak=0, seeds=[], completed=0, aborted=0, max_tokens=[], order=[])
if os.path.isdir(FAKE):
    shutil.copytree(os.path.join(FAKE, "caplog"), os.path.join(AO, "mock-caplog"), dirs_exist_ok=True)
    shutil.copytree(os.path.join(FAKE, "cap"), os.path.join(AO, "mock-private"), dirs_exist_ok=True)
    flushed = dict(pending_at_flush=0, flush_wait_s=0.0, emitted_moe=30, emitted_att=5, torn=0, errors=0)
    json.dump(flushed, open(os.path.join(AO, "mock-caplog", "FLUSHED.json"), "w"))


def save():
    with lock:
        json.dump(st, open(os.path.join(AO, "mock-stats.json"), "w"))
        met = QUOTA_AFTER > 0 and st["completed"] >= QUOTA_AFTER
        json.dump(dict(quotas_met=met, completed=st["completed"]), open(os.path.join(AO, "mock-status.json"), "w"))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        try:
            self.wfile.write(b)
        except (BrokenPipeError, ConnectionResetError):
            with lock:
                st["aborted"] += 1

    def do_GET(self):
        self._send(dict(data=[dict(id="mock-model")]))

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        with lock:
            st["inflight"] += 1
            st["peak"] = max(st["peak"], st["inflight"])
            st["seeds"].append(body.get("seed"))
            st["max_tokens"].append(body.get("max_tokens"))
            st["order"].append([round(time.time(), 3), body.get("seed"), st["inflight"]])
        save()
        time.sleep(body.get("max_tokens", 16) / TOK_S)
        with lock:
            st["inflight"] -= 1
            st["completed"] += 1
        save()
        self._send(dict(choices=[dict(message=dict(content="x", reasoning_content="y"), finish_reason="length")],
                        usage=dict(completion_tokens=body.get("max_tokens"), prompt_tokens=10)))


save()
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
