#!/usr/bin/env python3
"""OpenAI-compatible MOCK server for the CPU dry run of client_mapguard.py (no model, no GPU).

127.0.0.1:8001 only.  /v1/models, /v1/completions (plain and stream=True SSE, aborted when the client
closes the stream), /v1/chat/completions (logprobs if asked), /metrics (vllm:num_requests_running /
waiting).  prompt_tokens: completions = len(prompt ids); chat = 2795 for the doc-127 messages hash (so
the 2d bookkeeping path can be exercised), else 100.  Mock values only -- never evidence of the engine.
Writes nothing but an access count summary to stdout on exit.
"""
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DOC127 = "b41ce17bb0d4ce27af85de277fdd16b86cb88ad66683d95c799146dced26ad62"
LOCK = threading.Lock()
STATE = dict(running=0, n=0, aborted=0)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, o, ctype="application/json"):
        b = o if isinstance(o, bytes) else json.dumps(o).encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/metrics"):
            with LOCK:
                r = STATE["running"]
            self._send(f"vllm:num_requests_running{{model_name=\"mock\"}} {float(r)}\n"
                       f"vllm:num_requests_waiting{{model_name=\"mock\"}} 0.0\n".encode(), "text/plain")
        else:
            self._send({"data": [{"id": "mock-model"}]})

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(n))
        with LOCK:
            STATE["running"] += 1
            STATE["n"] += 1
        try:
            mt = body.get("max_tokens", 1)
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    time.sleep(0.2)                      # "prefill"
                    for i in range(mt):
                        self.wfile.write(b'data: {"choices":[{"text":"z"}]}\n\n')
                        self.wfile.flush()
                        time.sleep(0.02)
                    self.wfile.write(b"data: [DONE]\n\n")
                except (BrokenPipeError, ConnectionResetError):
                    with LOCK:
                        STATE["aborted"] += 1
                self.close_connection = True
                return
            time.sleep(0.05)
            if self.path.endswith("/chat/completions"):
                h = hashlib.sha256(json.dumps(body["messages"], sort_keys=True, ensure_ascii=False,
                                              separators=(",", ":")).encode()).hexdigest()
                ch = {"message": {"content": "x", "reasoning_content": "y"}, "finish_reason": "length",
                      "logprobs": {"content": [{"token": "a", "logprob": -0.1}] * min(mt, 5)} if body.get("logprobs") else None}
                pt = 2795 if h == DOC127 else 100
            else:
                ch = {"text": "z", "finish_reason": "length"}
                pt = len(body["prompt"])
            self._send({"choices": [ch], "usage": {"prompt_tokens": pt, "completion_tokens": mt,
                                                   "prompt_tokens_details": {"cached_tokens": 0}}})
        finally:
            with LOCK:
                STATE["running"] -= 1


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", 8001), H)
    srv.daemon_threads = True
    try:
        srv.serve_forever()
    finally:
        print(json.dumps(STATE))
