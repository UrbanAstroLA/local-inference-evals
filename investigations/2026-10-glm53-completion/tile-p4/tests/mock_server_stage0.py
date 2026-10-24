#!/usr/bin/env python3
"""Mock of the stage-0 serving container for MOCK=1 runs of runner/stage0/run_stage0.sh (no docker, no GPU).  Host
python3, stdlib only.

usage: mock_server_stage0.py CTRDIR PORT

CTRDIR plays the container's /tmp/p4.  At start: local-p4 init records (schema 2, s0 on, both ranks), entry_init
records (2 KV groups covered on both ranks), forcewin "registered" records and stock-rule bind logs (tests/test_gates).
/v1/completions:
  keyed warm  (forcewin_hooks [])        -> 1 token; "req" + "blocks" records; forcewin add record
  forced      (forcewin_hooks [...])     -> the forced ids, finish "length"; removal flush records (rank 0 store,
                                              entry digests on both ranks); status_r0.json flushed count
  multi       (local_fw_multi, prompt = list of prompts) -> one choice per prompt with ITS forced ids, picked by
              the prompt key exactly as the forcewin port does (a prompt without a unique key: empty ids, as unforced)
  unkeyed     (sentinel)                   -> 1 token
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CTR, PORT = sys.argv[1], int(sys.argv[2])
LOCK = threading.Lock()
ST = dict(flushed=0, uid=0, step=0)


def multi_key(prompt_ids):
    p = [int(x) for x in prompt_ids]
    return "%d:%s:%s" % (len(p), ",".join(map(str, p[:8])), ",".join(map(str, p[-8:])))


def log(rank, rec, prefix="p4"):
    with open(os.path.join(CTR, "log", f"{prefix}_r{rank}_p{10 + rank}.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def fwlog(rec):
    with open(os.path.join(CTR, "PRIVATE-fw", "fw_r0_p10.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def status():
    with open(os.path.join(CTR, "log", ".s.tmp"), "w") as f:
        json.dump(dict(rank=0, flushed=ST["flushed"], registered=True, serving=True, s0=True), f)
    os.replace(os.path.join(CTR, "log", ".s.tmp"), os.path.join(CTR, "log", "status_r0.json"))


def setup():
    import test_gates as TG
    for d in ("log", "PRIVATE", "PRIVATE-fw"):
        os.makedirs(os.path.join(CTR, d), exist_ok=True)
    for rank in (0, 1):
        log(rank, dict(kind="init", rank=rank, schema="local-p4/2", s0=True, issues=[],
                       stores_mib=192.0 if rank == 0 else 0.0, n_digest_layers=78 if rank == 0 else 0,
                       stage_modules={f"m{k}": k for k in range(6)} if rank == 0 else {}, tile_env=""))
        log(rank, dict(kind="entry_init", rank=rank, covered=3, uncovered=[], n_groups=2,
                       tensors=[dict(layer="mla.0", group=0, covered=True, ppb=68),
                                dict(layer="kda.0#0", group=1, covered=True, ppb=1),
                                dict(layer="kda.0#1", group=1, covered=True, ppb=1)]))
        open(os.path.join(CTR, "PRIVATE-fw", f"fw_r{rank}_p{10 + rank}.jsonl"), "a").write(
            json.dumps({"kind": "init"}) + "\n" + json.dumps({"kind": "registered"}) + "\n")
    recs = TG.make("A", None)
    for rank in (0, 1):
        with open(os.path.join(CTR, "log", f"bind_r{rank}_p{10 + rank}.jsonl"), "w") as f:
            for r in recs:
                if r["rank"] == rank:
                    f.write(json.dumps(r) + "\n")
    status()


def keyed(tag, W, nblocks):
    with LOCK:
        ST["uid"] += 1
        uid = ST["uid"]
        log(0, dict(kind="req", uid=uid, tag=tag, req_idx=uid % 16, W=W))
        fwlog(dict(kind="add", tag=tag, idx=uid % 16))
        log(0, dict(kind="blocks", tag=tag, uid=uid, groups=[list(range(nblocks)), list(range(nblocks))],
                    block_sizes=[4352, 4352]))
        if W > 0:
            ST["flushed"] += 1
            log(0, dict(kind="flushed", tag=tag, uid=uid, W=W))
            for rk in (0, 1):
                log(rk, dict(kind="entry_flushed", tag=tag, uid=uid, points=4, rank=rk))
            open(os.path.join(CTR, "PRIVATE", f"p4_{tag}_r0.npz"), "wb").write(b"mock")
        status()


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
        if "local_fw_multi" in xa:
            spec = json.loads(xa["local_fw_multi"])
            chs = []
            for i, p in enumerate(prompt):
                hit = [e for e in spec["entries"] if e["key"] == multi_key(p)]
                F = hit[0]["F"] if len(hit) == 1 else []
                if len(hit) == 1:
                    keyed(hit[0]["tag"], len(F), len(p) // 4352 + 1)
                chs.append(dict(index=i, text="", token_ids=list(F), finish_reason="length"))
            return self._send(200, dict(choices=chs, usage=dict(prompt_tokens=sum(len(p) for p in prompt),
                                                                completion_tokens=sum(len(c["token_ids"]) for c in chs))))
        if "forcewin_hooks" in xa:
            F = list(xa["forcewin_hooks"])
            keyed(xa.get("local_fw_tag", ""), len(F), len(prompt) // 4352 + 1)
            ids = F if F else [1]
            return self._send(200, dict(choices=[dict(index=0, text="", token_ids=ids, finish_reason="length")],
                                        usage=dict(prompt_tokens=len(prompt), completion_tokens=len(ids),
                                                   prompt_tokens_details=dict(cached_tokens=0))))
        self._send(200, dict(choices=[dict(index=0, text="", finish_reason="length")],
                             usage=dict(prompt_tokens=len(prompt), completion_tokens=1)))


if __name__ == "__main__":
    setup()
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
