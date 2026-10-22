#!/usr/bin/env python3
"""FAKE model server for the B+ session simulation (tests/session_sim.sh). CPU only, 127.0.0.1 only, no model.
Answers /v1/models, /v1/chat/completions/render (returns the prefix's prompt ids, looked up by messages hash) and
/v1/completions (synthetic random continuations: finish with probability P_FINISH, else 65,536 tokens).
usage: fake_server.py PORT PKG_DIR   env: FAKE_SECS (per-request delay), FAKE_FAIL_K (answer HTTP 500 to the k-th request)
"""
import hashlib
import json
import os
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT, PKG = int(sys.argv[1]), sys.argv[2]
MODEL = "wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-v1"
pre = json.load(open(f"{PKG}/PRIVATE/prefixes.json"))["prefixes"]
prompts = json.load(open(f"{PKG}/PRIVATE/prompts.json"))["prompts"]
man = {m["prefix_id"]: m for m in json.load(open(f"{PKG}/out/prefix_manifest.json"))["prefixes"]}
by_msg = {}
for p, v in prompts.items():
    by_msg[json.dumps(v["messages"], sort_keys=True)] = pre[p][:man[p]["prompt_len"]]
SECS = float(os.environ.get("FAKE_SECS", "0.05"))
FAIL_K = int(os.environ.get("FAKE_FAIL_K", "0"))
P_FINISH = float(os.environ.get("FAKE_P_FINISH", "0.5"))
lock = threading.Lock()
count = [0]


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/v1/models":
            return self._send(200, {"data": [{"id": MODEL}]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/chat/completions/render":
            return self._send(200, {"token_ids": by_msg.get(json.dumps(body["messages"], sort_keys=True), [])})
        if self.path != "/v1/completions":
            return self._send(404, {"error": "not found"})
        with lock:
            count[0] += 1
            n = count[0]
        time.sleep(SECS)
        if FAIL_K and n == FAIL_K:
            return self._send(500, {"error": "fake failure"})
        rng = random.Random(body["seed"])
        if rng.random() < P_FINISH:
            L = rng.randint(1000, 60000)
            gen = [rng.randint(1000, 120000) for _ in range(L)] + [154842] + [rng.randint(1000, 120000) for _ in range(200)] + [154827]
            fr, sr = "stop", 154827
        else:
            gen = [rng.randint(1000, 120000) for _ in range(body["max_tokens"])]
            fr, sr = "length", None
        text = "x" * (3 * len(gen))
        self._send(200, {"id": f"cmpl-fake-{n}", "choices": [{"index": 0, "text": text, "finish_reason": fr, "stop_reason": sr,
                                                               "token_ids": gen, "prompt_token_ids": body["prompt"]}],
                         "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": len(gen),
                                   "prompt_tokens_details": {"cached_tokens": 0}}})

    def log_message(self, *a):
        pass


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
