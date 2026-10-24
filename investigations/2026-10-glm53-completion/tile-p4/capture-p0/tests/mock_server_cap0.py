#!/usr/bin/env python3
"""Mock of the capture-p0 serving container for MOCK=1 runs of runner/run_cap0.sh (no docker, no GPU).  Host python3,
stdlib only.  SYNTHETIC: it writes records in the hook modules' formats with fabricated values; no tensor bytes.

usage: mock_server_cap0.py CTRDIR PORT LOC(on|off)

CTRDIR plays the container's /tmp/p4.  At start: local-p4 init records (schema 2, s0 on, both ranks), cap0_init
records (rank 0: MLA/indexer 3, 7 + KDA 0/2/4; rank 1: KDA 0/2/4; MLA owner map split:25; 12 / 11 hooks when LOC=on),
forcewin "registered" records and stock-rule bind logs (tile-p4/tests/test_gates).
/v1/completions (keyed requests carry vllm_xargs forcewin_hooks [] and local_fw_tag):
  "CW-*"  warm      -> 1 token; "req" + "blocks" records; for "-L" a loc_flushed record per rank
  "CAP-*" capture   -> 1 token; a cap0_capture record per rank (n_computed = its cached tokens)
  unkeyed sentinel  -> 1 token
Prefix cache, modelled on the engine's measured rule (PROPOSAL-AMENDMENTS.md A3): per cache salt it remembers the
longest prompt seen; a request gets min(full blocks cached, floor((P - 1) / 4352)) - 1 blocks (one dropped on a hit).
It writes NO scheduler / step logs: those come only from p4_hooks's real flush path (tests/test_engine_path_cpu.py),
so a mock session can never stand in for them.
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TILE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(TILE, "tests"))
CTR, PORT, LOC = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "on"
LOCK = threading.Lock()
ST = dict(uid=0)
CACHE = {}                      # salt -> longest prompt seen (tokens)


def cached_for(salt, n):
    usable = min(CACHE.get(salt, 0) // 4352, (n - 1) // 4352)
    CACHE[salt] = max(CACHE.get(salt, 0), n)
    return max(0, usable - 1) * 4352


def log(rank, rec, prefix="p4"):
    with open(os.path.join(CTR, "log", f"{prefix}_r{rank}_p{10 + rank}.jsonl"), "a") as f:
        f.write(json.dumps(dict(rec, rank=rank)) + "\n")


def setup():
    import test_gates as TG
    for d in ("log", "PRIVATE", "PRIVATE-fw"):
        os.makedirs(os.path.join(CTR, d), exist_ok=True)
    for rank in (0, 1):
        log(rank, dict(kind="init", schema="local-p4/2", s0=True, issues=[], stores_mib=197.4 if rank == 0 else 0.0,
                       n_digest_layers=45 if rank == 0 else 0, tile_env=""))
        tk = ["kda0", "kda2", "kda4"] + (["mla3", "mla7", "idx3", "idx7"] if rank == 0 else [])
        log(rank, dict(kind="cap0_init", schema="local-cap0/1", loc=LOC, mla_owners="split:25", conv_patched=LOC,
                       hooks=(["h"] * (int(os.environ.get("MOCK_HOOKS_R0", "12")) if rank == 0 else 11)) if LOC else [],
                       targets={k: dict(kind=k[:3], layer=int(k[3:]), group=0 if k[:3] in ("mla", "idx") else
                                        {0: 2, 2: 4, 4: 5}[int(k[3:])], ok=True) for k in tk}, missing=[]), prefix="cap0")
        open(os.path.join(CTR, "PRIVATE-fw", f"fw_r{rank}_p{10 + rank}.jsonl"), "a").write(
            json.dumps({"kind": "init"}) + "\n" + json.dumps({"kind": "registered"}) + "\n")
    recs = TG.make("A", None)
    for rank in (0, 1):
        with open(os.path.join(CTR, "log", f"bind_r{rank}_p{10 + rank}.jsonl"), "w") as f:
            for r in recs:
                if r["rank"] == rank:
                    f.write(json.dumps(r) + "\n")


def keyed(tag, n_prompt, cached):
    with LOCK:
        ST["uid"] += 1
        uid = ST["uid"]
        log(0, dict(kind="req", uid=uid, tag=tag, req_idx=uid % 16, W=0))
        log(0, dict(kind="blocks", tag=tag, uid=uid, groups=[[10 + uid, 11 + uid, 12 + uid, 13 + uid]] * 6,
                    block_sizes=[4352] * 6))
        if tag.startswith("CW-") and tag.endswith("-L"):
            for rk in (0, 1):
                open(os.path.join(CTR, "PRIVATE", f"loc_{tag}_r{rk}.npz"), "wb").write(b"mock")
                log(rk, dict(kind="cap0_loc_flushed", tag=tag, uid=uid, is_ref=uid == 1, steps=[], points={},
                             file_sha256="0" * 64), prefix="cap0")
        if tag.startswith("CAP-"):
            for rk in (0, 1):
                open(os.path.join(CTR, "PRIVATE", f"cap0_{tag}_r{rk}.npz"), "wb").write(b"mock")
                log(rk, dict(kind="cap0_capture", tag=tag, uid=uid, ok=True, n_computed=cached, n_sched=2045,
                             slices={}, file_sha256="0" * 64), prefix="cap0")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            return self._send(200, dict(data=[dict(id="mock-glm53")]))
        self._send(404, {})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n).decode() or "{}")
        xa = body.get("vllm_xargs") or {}
        prompt = body.get("prompt") or []
        cached = cached_for(body.get("cache_salt", ""), len(prompt))
        if "forcewin_hooks" in xa:
            keyed(xa.get("local_fw_tag", ""), len(prompt), cached)
        self._send(200, dict(choices=[dict(index=0, text="", finish_reason="length")],
                             usage=dict(prompt_tokens=len(prompt), completion_tokens=1,
                                        prompt_tokens_details=dict(cached_tokens=cached))))


if __name__ == "__main__":
    setup()
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
