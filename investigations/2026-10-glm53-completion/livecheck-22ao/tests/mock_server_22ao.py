#!/usr/bin/env python3
"""OpenAI-compatible MOCK server for the 22ao CPU session simulation (no model, no GPU), 127.0.0.1:18001 only.
/v1/models and /v1/chat/completions. A completion sleeps in proportion to max_tokens (MOCK_TOK_S tokens/s), then
returns finish 'length' or 'stop' at random (seeded by the request seed) with usage counts and a placeholder reply.
Tracks the peak number of concurrent requests and writes it to argv[1] after every request (JSON)."""
import json
import os
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATS = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mock-stats.json"
TOK_S = float(os.environ.get("MOCK_TOK_S", "4000"))
lock = threading.Lock()
state = dict(active=0, peak=0, served=0, seeds=[])


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, o):
        b = json.dumps(o).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._send({"data": [{"id": "mock-glm53"}]})
        else:
            self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        try:
            rnd = random.Random(body.get("seed", 0))
            mt = int(body.get("max_tokens", 16))
            gen = mt if rnd.random() < 0.5 else rnd.randint(1, mt)
            time.sleep(gen / TOK_S)
            self._send({"choices": [{"message": {"content": "x", "reasoning_content": "y"},
                                     "finish_reason": "length" if gen == mt else "stop"}],
                        "usage": {"completion_tokens": gen, "prompt_tokens": 300}})
        finally:
            with lock:
                state["active"] -= 1
                state["served"] += 1
                state["seeds"].append(body.get("seed"))
                json.dump(state, open(STATS, "w"))


ThreadingHTTPServer(("127.0.0.1", 18001), H).serve_forever()
