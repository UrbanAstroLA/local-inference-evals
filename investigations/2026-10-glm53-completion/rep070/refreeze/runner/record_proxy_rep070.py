"""Recording pass-through for OpenAI-style chat completions: keeps what lm-eval throws away.

lm-eval's sample files keep only the answer text. This proxy sits between lm-eval and the server and appends one JSON
line per request: the request's message hash (joins to lm-eval's samples), seed, max_tokens, and the response's
finish_reason, token counts, reasoning length and an answer-text hash equal to the receipts' response_sha256.
Nothing in the request or response is changed. Non-streaming requests only (lm-eval's local-chat-completions).

rep070 copy of <eval-tools>/diag/record_proxy.py (sha256 946d880e...). The main log is unchanged (same rows, same
bytes, only for requests that got an upstream response). Added: a SIDECAR event log, OUT with '.jsonl' replaced by
'.events.jsonl' (e.g. x.requests.jsonl -> x.requests.events.jsonl), one line per event:
  start        on arrival of a chat-completion request: t, messages_sha256, seed, max_tokens
  interrupted  the upstream call raised (server stopped, connection reset, timeout): error type, secs, t_start,
               messages_sha256, seed; also written for every request still in flight when the proxy gets SIGTERM/SIGINT
               (error 'proxy_terminated')
  client_gone  the response could not be written back to the client (broken pipe or reset); the main-log row, if any,
               was already written
The proxy never crashes on a broken client pipe.
Added (review section 6, local only, never published): a RAW file with the full request body and the full upstream
response body of every chat-completion exchange (reasoning and final text, finish reason, usage), one gzip member per
exchange (gzip-readable as one stream). Path: 4th argument, default OUT with '.jsonl' replaced by '.raw.jsonl.gz'. It is
created under umask 077 with mode 0600 before the server starts. Raw, sidecar and main log are appended in that order
under one lock, so every main-log row has its raw record. The main log stays byte-compatible with the original proxy.

usage (copy): record_proxy_rep070.py LISTEN_PORT UPSTREAM_BASE OUT.jsonl [RAW.jsonl.gz]

usage: record_proxy.py LISTEN_PORT UPSTREAM_BASE OUT.jsonl     e.g. 8002 http://127.0.0.1:8001 /path/requests.jsonl
"""
import gzip, hashlib, json, os, signal, sys, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT, UPSTREAM, OUT = int(sys.argv[1]), sys.argv[2].rstrip("/"), sys.argv[3]
LOCK = threading.Lock()
BASE = OUT[:-len(".jsonl")] if OUT.endswith(".jsonl") else OUT
EVENTS = BASE + ".events.jsonl"
RAW = sys.argv[4] if len(sys.argv) > 4 else BASE + ".raw.jsonl.gz"
INFLIGHT = {}   # id -> (t_start, messages_sha256, seed, request body)


def event(rec):   # caller holds LOCK
    with open(EVENTS, "a") as f:
        f.write(json.dumps(rec) + "\n")


def raw_open():
    """Create the raw file under umask 077, mode 0600 (before any request is served)."""
    old = os.umask(0o077)
    try:
        fd = os.open(RAW, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        os.close(fd)
        os.chmod(RAW, 0o600)
    finally:
        os.umask(old)


def raw(rec):     # caller holds LOCK; one gzip member per exchange
    with open(RAW, "ab") as f, gzip.GzipFile(fileobj=f, mode="ab") as g:
        g.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))


def body_obj(b):
    try:
        return json.loads(b)
    except Exception:  # noqa: BLE001
        return {"unparsed_body_text": b.decode("utf-8", "replace")}


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def messages_hash(messages):
    return sha(json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _forward(self, method, body=None):
        req = urllib.request.Request(UPSTREAM + self.path, data=body, method=method,
                                     headers={"Content-Type": self.headers.get("Content-Type", "application/json")})
        try:
            with urllib.request.urlopen(req, timeout=86400) as r:
                return r.status, r.read(), r.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers.get("Content-Type", "application/json")

    def _reply(self, status, data, ctype, ident=None):
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError) as e:
            if ident is not None:
                with LOCK:
                    event({"event": "client_gone", "t": round(time.time(), 3), "error": type(e).__name__,
                           "messages_sha256": ident[1], "seed": ident[2]})

    def do_GET(self):
        self._reply(*self._forward("GET"))

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        t0 = time.time()
        chat = self.path.rstrip("/").endswith("/chat/completions")
        ident = None
        if chat:
            try:
                q0 = json.loads(body)
                ident = (round(t0, 3), messages_hash(q0.get("messages")), q0.get("seed"), q0.get("max_tokens"))
            except Exception:  # noqa: BLE001 - record, never break the request
                ident = (round(t0, 3), None, None, None)
            with LOCK:
                INFLIGHT[id(self)] = ident[:3] + (body,)
                event({"event": "start", "t": ident[0], "messages_sha256": ident[1], "seed": ident[2], "max_tokens": ident[3]})
        try:
            status, data, ctype = self._forward("POST", body)
        except Exception as e:  # noqa: BLE001 - upstream failure: sidecar only, the main log is untouched
            if chat:
                with LOCK:
                    INFLIGHT.pop(id(self), None)
                    raw({"kind": "interrupted", "t_start": ident[0], "secs": round(time.time() - t0, 3), "error": type(e).__name__,
                         "messages_sha256": ident[1], "seed": ident[2], "request": body_obj(body), "response": None})
                    event({"event": "interrupted", "t": round(time.time(), 3), "t_start": ident[0], "secs": round(time.time() - t0, 3),
                           "error": type(e).__name__, "messages_sha256": ident[1], "seed": ident[2]})
            self._reply(502, b'{"error": "upstream failure (rep070 proxy)"}', "application/json", ident)
            return
        if chat:
            rec = {"t_start": round(t0, 3), "secs": round(time.time() - t0, 3), "http_status": status}
            try:
                q = json.loads(body)
                rec.update(messages_sha256=messages_hash(q.get("messages")), seed=q.get("seed"),
                           max_tokens=q.get("max_tokens"), temperature=q.get("temperature"), top_p=q.get("top_p"))
            except Exception as e:  # noqa: BLE001 - record, never break the request
                rec["request_parse_error"] = repr(e)[:200]
            try:
                r = json.loads(data)
                c = r["choices"][0]; m = c.get("message") or {}
                content = m.get("content") or ""
                reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
                u = r.get("usage") or {}
                rec.update(finish_reason=c.get("finish_reason"), prompt_tokens=u.get("prompt_tokens"),
                           completion_tokens=u.get("completion_tokens"),
                           reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                           content_chars=len(content),
                           reasoning_chars=len(reasoning), content_sha256=sha(content))
            except Exception as e:  # noqa: BLE001
                rec["response_parse_error"] = repr(e)[:200]
            with LOCK:
                raw({"kind": "exchange", "t_start": rec["t_start"], "secs": rec["secs"], "http_status": status,
                     "messages_sha256": rec.get("messages_sha256"), "seed": rec.get("seed"), "request": body_obj(body), "response": body_obj(data)})
                with open(OUT, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                INFLIGHT.pop(id(self), None)
        self._reply(status, data, ctype, ident)

    def log_message(self, *a):
        pass


def _terminate(signum, frame):
    with LOCK:
        now = time.time()
        for t_start, msha, seed, body in list(INFLIGHT.values()):
            raw({"kind": "interrupted", "t_start": t_start, "secs": round(now - t_start, 3), "error": "proxy_terminated",
                 "messages_sha256": msha, "seed": seed, "request": body_obj(body), "response": None})
            event({"event": "interrupted", "t": round(now, 3), "t_start": t_start, "secs": round(now - t_start, 3),
                   "error": "proxy_terminated", "messages_sha256": msha, "seed": seed})
        INFLIGHT.clear()
    os._exit(0)


if __name__ == "__main__":
    raw_open()
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    srv.daemon_threads = True
    print(f"recording {UPSTREAM} on 127.0.0.1:{PORT} -> {OUT}", flush=True)
    srv.serve_forever()
