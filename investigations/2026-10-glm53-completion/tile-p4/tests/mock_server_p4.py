#!/usr/bin/env python3
"""Mock of the P4 serving container for MOCK=1 runner sessions (no docker, no GPU).  Host python3, stdlib only.

usage: mock_server_p4.py CTRDIR PORT SESSION        (env MOCK_TILE: "" or "64,256,64,256"; MOCK_BAD_GATE=1)

CTRDIR plays the container's /tmp/p4: at start it writes the local-p4 init records (both ranks), forcewin
"registered" records, and bind logs (fabricated from captured-p3b's real launch fields via tests/test_gates.make:
stock rule, or the forced tile when MOCK_TILE is set; MOCK_BAD_GATE=1 leaves one m=24 launch unforced).
/v1/completions returns the forced ids for keyed requests (finish "length", cached_tokens by the EAGLE rule) and,
for every keyed request, appends a "flushed" record, writes a tiny PRIVATE npz and updates status_r0.json, as the
removal flush would.
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CTR, PORT, SESSION = sys.argv[1], int(sys.argv[2]), sys.argv[3]
TILE = os.environ.get("MOCK_TILE", "")
LOCK = threading.Lock()
FLUSHED = []


def setup():
    import test_gates as TG
    for d in ("log", "PRIVATE", "PRIVATE-fw"):
        os.makedirs(os.path.join(CTR, d), exist_ok=True)
    for rank in (0, 1):
        init = dict(kind="init", rank=rank, issues=[], stores_mib=186.0 if rank == 0 else 0.0,
                    stage_modules={f"m{k}": k for k in range(6)} if rank == 0 else {}, tile_env=TILE)
        open(os.path.join(CTR, "log", f"p4_r{rank}_p{10 + rank}.jsonl"), "w").write(json.dumps(init) + "\n")
        open(os.path.join(CTR, "PRIVATE-fw", f"fw_r{rank}_p{10 + rank}.jsonl"), "w").write(
            json.dumps({"kind": "init"}) + "\n" + json.dumps({"kind": "registered"}) + "\n")
    arm = "B" if TILE else "A"

    def bad(recs):
        r = next(x for x in recs if x["rank"] == 1 and x["layer"] == 20 and x["plan"] == "decode" and x["m"] == 24)
        r["launch"].update(fc1_tile_k=128, fc1_tile_n=128, fc2_tile_k=128, fc2_tile_n=128)
    recs = TG.make(arm, bad if os.environ.get("MOCK_BAD_GATE") == "1" else None)
    for rank in (0, 1):
        with open(os.path.join(CTR, "log", f"bind_r{rank}_p{10 + rank}.jsonl"), "w") as f:
            for r in recs:
                if r["rank"] == rank:
                    f.write(json.dumps(r) + "\n")
    status()


def status():
    with open(os.path.join(CTR, "log", ".s.tmp"), "w") as f:
        json.dump(dict(rank=0, flushed=len(FLUSHED), registered=True, serving=bool(FLUSHED)), f)
    os.replace(os.path.join(CTR, "log", ".s.tmp"), os.path.join(CTR, "log", "status_r0.json"))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/v1/models":
            return self._send(dict(data=[dict(id="mock-glm53")]))
        self._send({}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n).decode())
        xa = body.get("vllm_xargs") or {}
        P = len(body.get("prompt") or [])
        F = xa.get("forcewin_hooks")
        if F is not None:
            assert xa.get("local_fw_draft") == "list" and xa.get("local_fw_sched")
            tag = xa["local_fw_tag"]
            cached = max(0, ((P - 1) // 4352 - 1) * 4352)
            out = dict(choices=[dict(token_ids=list(F), finish_reason="length", text="")],
                       usage=dict(prompt_tokens=P, completion_tokens=len(F), prompt_tokens_details=dict(cached_tokens=cached)))
            with LOCK:
                FLUSHED.append(tag)
                with open(os.path.join(CTR, "log", "p4_r0_p10.jsonl"), "a") as f:
                    f.write(json.dumps(dict(kind="flushed", tag=tag, W=len(F))) + "\n")
                fd = os.open(os.path.join(CTR, "PRIVATE", f"p4_{tag}_r0.npz"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                os.write(fd, b"mock")
                os.close(fd)
                status()
            return self._send(out)
        return self._send(dict(choices=[dict(token_ids=[1], finish_reason="length", text="x")],
                               usage=dict(prompt_tokens=P, completion_tokens=1, prompt_tokens_details=dict(cached_tokens=0))))


if __name__ == "__main__":
    setup()
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
