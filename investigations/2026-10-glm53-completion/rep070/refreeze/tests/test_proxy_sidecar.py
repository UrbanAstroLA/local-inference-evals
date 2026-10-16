#!/usr/bin/env python3
"""Tests of runner/record_proxy_rep070.py against a fake upstream (CPU only; loopback ports 18911-18913; synthetic
messages; no model server). Cases:
  1. completed request: the copy's main-log row equals the ORIGINAL proxy's row for the same request (all fields and key
     order; only the timing fields t_start and secs differ); one sidecar start event with seed and max_tokens;
  2. the upstream is killed mid-request (SIGKILL, as a stopped server): sidecar start + interrupted with the error type;
     no main-log row; the client gets 502; the proxy stays up;
  3. the client goes away before a large response: the main-log row is written, the sidecar records client_gone, and
     the proxy stays up and serves the next request;
  4. SIGTERM to the proxy with a request in flight: sidecar interrupted, error proxy_terminated; no main-log row.
  5. raw-file writer: with a fixed clock (both proxies run with time.time patched to a constant), the copy's main log is
     BYTE-identical to the original proxy's on the same synthetic exchange; the raw file (gzip) holds the request and
     response bodies; it has mode 0600; across cases 1-4 every main-log row has its raw exchange record and every
     interrupted event its raw interrupted record (logs consistent after upstream failure, broken pipe and SIGTERM).
usage: test_proxy_sidecar.py WORK_DIR   (exit 0 only if every check passed)"""
import json, os, signal, socket, subprocess, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
COPY = os.path.join(HERE, "..", "runner", "record_proxy_rep070.py")
ORIG = "<eval-tools>/diag/record_proxy.py"
W = sys.argv[1]
os.makedirs(W, exist_ok=True)
import gzip, stat
UP, PC, PO, PF1, PF2 = 18911, 18912, 18913, 18914, 18915
FIXED = os.path.join(W, "fixed_clock.py")
open(FIXED, "w").write("import runpy, sys, time\ntime.time = lambda: 1760000000.0\nscript = sys.argv[1]\nsys.argv = sys.argv[1:]\n"
                       "runpy.run_path(script, run_name='__main__')\n")
UPSRC = os.path.join(W, "fake_upstream.py")
open(UPSRC, "w").write('''import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
DELAY, BIG = float(sys.argv[2]), sys.argv[3] == "big"
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        time.sleep(DELAY)
        body = json.dumps({"choices": [{"finish_reason": "length", "message": {"content": ("y" * 8000000) if BIG else "",
                           "reasoning_content": "x" * 10}}], "usage": {"prompt_tokens": 5, "completion_tokens": 327680}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)
    def log_message(self, *a):
        pass
s = ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H); s.daemon_threads = True; s.serve_forever()
''')
RES = []


def ok(name, cond, detail=""):
    RES.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'} {name}" + (f" ({detail})" if detail else ""), flush=True)


def free(p):
    s = socket.socket(); r = s.connect_ex(("127.0.0.1", p)); s.close(); return r != 0


def wait_port(p):
    for _ in range(100):
        if not free(p):
            return
        time.sleep(0.05)
    raise RuntimeError(f"port {p} did not open")


def upstream(delay, big=False):
    p = subprocess.Popen([sys.executable, UPSRC, str(UP), str(delay), "big" if big else "small"])
    wait_port(UP); return p


def proxy(script, port, out, fixed=False):
    pre = [FIXED] if fixed else []
    p = subprocess.Popen([sys.executable] + pre + [script, str(port), f"http://127.0.0.1:{UP}", out], stdout=subprocess.DEVNULL)
    wait_port(port); return p


def post(port, seed, timeout=60, out=None):
    body = json.dumps({"messages": [{"role": "user", "content": f"synthetic test {seed}"}], "seed": seed, "max_tokens": 327680,
                       "temperature": 1.0, "top_p": 0.95}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            v = r.status
    except urllib.error.HTTPError as e:
        v = e.code
    except Exception as e:  # noqa: BLE001
        v = type(e).__name__
    if out is not None:
        out.append(v)
    return v


def rows(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def stop(p, sig=signal.SIGTERM):
    if p.poll() is None:
        p.send_signal(sig); p.wait(10)


def rawrecs(p):
    out = []
    if os.path.exists(p):
        with gzip.open(p, "rt") as f:
            out = [json.loads(l) for l in f if l.strip()]
    return out


assert all(free(p) for p in (UP, PC, PO, PF1, PF2)), "test ports busy"
MAIN = os.path.join(W, "c.requests.jsonl"); SIDE = os.path.join(W, "c.requests.events.jsonl"); OMAIN = os.path.join(W, "o.requests.jsonl")
# ---- 1
u = upstream(0); pc = proxy(COPY, PC, MAIN); po = proxy(ORIG, PO, OMAIN)
post(PC, 1); post(PO, 1)
a, b = rows(MAIN), rows(OMAIN)
strip = lambda r: {k: v for k, v in r.items() if k not in ("t_start", "secs")}
ok("1 completed request: main-log row equals the original proxy's (timing fields excepted)", len(a) == 1 and len(b) == 1 and strip(a[0]) == strip(b[0]))
ok("1 same keys in the same order", bool(a and b) and list(a[0]) == list(b[0]), ",".join(a[0]) if a else "")
ev = rows(SIDE)
ok("1 sidecar: one start event (seed 1, max_tokens 327680), nothing else", len(ev) == 1 and ev[0]["event"] == "start" and ev[0]["seed"] == 1
   and ev[0]["max_tokens"] == 327680 and ev[0]["messages_sha256"] == a[0]["messages_sha256"])
stop(po); stop(u, signal.SIGKILL)
# ---- 2
u = upstream(30); got = []
th = threading.Thread(target=post, args=(PC, 2), kwargs={"out": got}); th.start()
time.sleep(1.0); stop(u, signal.SIGKILL); th.join(20)
ev = [e for e in rows(SIDE) if e.get("seed") == 2]
inter = [e for e in ev if e["event"] == "interrupted"]
ok("2 upstream killed mid-request: sidecar start + one interrupted event with the error type", [e["event"] for e in ev] == ["start", "interrupted"]
   and inter[0]["error"] not in ("", None, "proxy_terminated"), inter[0]["error"] if inter else "none")
ok("2 no main-log row for the interrupted request", not any(r.get("seed") == 2 for r in rows(MAIN)))
ok("2 client received 502 and the proxy is still up", got == [502] and pc.poll() is None, str(got))
# ---- 3
u = upstream(2, big=True); got = []
post(PC, 3, timeout=0.5, out=got)
time.sleep(4.0)
ok("3 client gone: main-log row written for the completed request", any(r.get("seed") == 3 for r in rows(MAIN)))
ok("3 sidecar client_gone recorded", any(e["event"] == "client_gone" and e.get("seed") == 3 for e in rows(SIDE)),
   [e["event"] for e in rows(SIDE) if e.get("seed") == 3])
stop(u, signal.SIGKILL); u = upstream(0)
ok("3 proxy still up and serves the next request", pc.poll() is None and post(PC, 4) == 200)
stop(u, signal.SIGKILL)
# ---- 4
u = upstream(30); got = []
th = threading.Thread(target=post, args=(PC, 5), kwargs={"out": got}); th.start()
time.sleep(1.0); stop(pc, signal.SIGTERM); th.join(20)
ev = [e for e in rows(SIDE) if e.get("seed") == 5]
ok("4 SIGTERM with a request in flight: sidecar interrupted, error proxy_terminated", [e["event"] for e in ev] == ["start", "interrupted"]
   and ev[-1]["error"] == "proxy_terminated")
ok("4 no main-log row; proxy exited", not any(r.get("seed") == 5 for r in rows(MAIN)) and pc.poll() is not None)
stop(u, signal.SIGKILL)
# ---- 5 raw-file writer
RAWF = os.path.join(W, "c.requests.raw.jsonl.gz")
ok("5 raw file mode is 0600", stat.S_IMODE(os.stat(RAWF).st_mode) == 0o600, oct(stat.S_IMODE(os.stat(RAWF).st_mode)))
rr = rawrecs(RAWF)
ex = {(r["messages_sha256"], r["seed"]) for r in rr if r["kind"] == "exchange"}
it = {(r["messages_sha256"], r["seed"]) for r in rr if r["kind"] == "interrupted"}
mains = {(r["messages_sha256"], r["seed"]) for r in rows(MAIN)}
ints = {(e["messages_sha256"], e["seed"]) for e in rows(SIDE) if e["event"] == "interrupted"}
ok("5 every main-log row has its raw exchange record, and no other exchange is recorded", ex == mains, f"{len(ex)} exchanges, {len(mains)} main rows")
ok("5 every interrupted event has its raw interrupted record (upstream kill, SIGTERM)", it == ints and len(it) == 2, f"{len(it)}")
r1 = [r for r in rr if r["kind"] == "exchange" and r["seed"] == 1][0]
ok("5 raw record holds the request body and the full response body (message, finish reason, usage)",
   r1["request"]["messages"][0]["content"] == "synthetic test 1" and r1["response"]["choices"][0]["finish_reason"] == "length"
   and r1["response"]["usage"]["completion_tokens"] == 327680 and "reasoning_content" in r1["response"]["choices"][0]["message"])
r3 = [r for r in rr if r["kind"] == "exchange" and r["seed"] == 3]
ok("5 broken-pipe exchange: raw body complete (8,000,000-char content)", len(r3) == 1 and len(r3[0]["response"]["choices"][0]["message"]["content"]) == 8000000)
FA, FB = os.path.join(W, "fixed-copy.requests.jsonl"), os.path.join(W, "fixed-orig.requests.jsonl")
u = upstream(0); pa = proxy(COPY, PF1, FA, fixed=True); pb = proxy(ORIG, PF2, FB, fixed=True)
post(PF1, 7); post(PF2, 7)
stop(pa); stop(pb); stop(u, signal.SIGKILL)
ba, bb = open(FA, "rb").read(), open(FB, "rb").read()
ok("5 fixed clock: the copy's main log is byte-identical to the original proxy's", ba == bb and len(ba) > 0, f"{len(ba)} bytes")
print(f"proxy sidecar tests: {sum(RES)} passed, {len(RES) - sum(RES)} failed")
sys.exit(0 if all(RES) else 1)
