#!/usr/bin/env python3
"""serving-probe/v1 client. Usage: GPQA_SAMPLES=... serving_probe.py OUTDIR [--greedy-runs N] [--no-sampled]

Against the server on :8001, using 16 fixed GPQA prompts (first 16 doc ids of the K4 pass3 samples, the same
prompt source as the empties probes):
  greedy  : temperature 0, 2,048 tokens, one request at a time -> per-prompt text, for exactness checks
  sampled : temperature 1.0 / top_p 0.95 / seed 1234, 4,096 tokens, at concurrency 1 and 8
            -> server speculation counters (accepted / drafted / per position), TTFT, decode tok/s
Protocol: protocols/serving-probe/v1.md
"""
import json, sys, os, time, glob, re, statistics as st, threading, queue, requests

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
greedy_runs = int(sys.argv[sys.argv.index("--greedy-runs") + 1]) if "--greedy-runs" in sys.argv else 1
sampled = "--no-sampled" not in sys.argv
URL = os.environ.get("BASE_URL", "http://127.0.0.1:8001")
SRC = os.environ["GPQA_SAMPLES"]   # any gpqa-diamond/v1 lm-eval samples_*.jsonl (supplies rendered prompts)
prompts = {}
for line in open(SRC):
    r = json.loads(line)
    prompts.setdefault(r["doc_id"], json.loads(r["arguments"]["gen_args_0"]["arg_0"][0]))
DOCS = sorted(prompts)[:16]
MODEL = requests.get(f"{URL}/v1/models").json()["data"][0]["id"]


def counters():
    m = {}
    for line in requests.get(f"{URL}/metrics").text.splitlines():
        if line.startswith("vllm:spec_decode_num_"):
            name = line.split("{")[0].replace("vllm:spec_decode_num_", "")
            pos = re.search(r'position="(\d+)"', line)
            key = f"{name}[{pos.group(1)}]" if pos else name
            if key.endswith("_created") or "_created[" in key:
                continue
            m[key] = m.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
    return m


def one(doc, temperature, max_tokens, seed=None):
    body = dict(model=MODEL, messages=prompts[doc], temperature=temperature, max_tokens=max_tokens,
                stream=True, stream_options={"include_usage": True},
                chat_template_kwargs={"enable_thinking": True})
    if temperature > 0:
        body.update(top_p=0.95, seed=seed)
    t0 = time.time(); tfirst = None; parts = []; usage = None; fin = None
    with requests.post(f"{URL}/v1/chat/completions", json=body, stream=True, timeout=3600) as resp:
        for line in resp.iter_lines():
            if not line.startswith(b"data: ") or line[6:] == b"[DONE]":
                continue
            j = json.loads(line[6:])
            usage = j.get("usage") or usage
            for c in j.get("choices", []):
                d = c.get("delta", {})
                txt = (d.get("reasoning_content") or d.get("reasoning") or "") + (d.get("content") or "")
                if txt:
                    tfirst = tfirst or time.time(); parts.append(txt)
                fin = c.get("finish_reason") or fin
    t1 = time.time(); n = (usage or {}).get("completion_tokens") or 0
    return dict(doc=doc, text="".join(parts), tokens=n, finish=fin, ttft=round((tfirst or t1) - t0, 3),
                decode_tok_s=round(n / max(1e-6, t1 - (tfirst or t0)), 2), secs=round(t1 - t0, 2))


def batch(conc, temperature, max_tokens, seed=None):
    q = queue.Queue(); [q.put(d) for d in DOCS]; res = []; lock = threading.Lock()
    def w():
        while True:
            try: d = q.get_nowait()
            except queue.Empty: return
            r = one(d, temperature, max_tokens, seed)
            with lock: res.append(r)
    c0 = counters(); t0 = time.time()
    ts = [threading.Thread(target=w) for _ in range(conc)]; [t.start() for t in ts]; [t.join() for t in ts]
    wall = time.time() - t0; c1 = counters()
    delta = {k: c1.get(k, 0) - c0.get(k, 0) for k in c1}
    acc, drf, drafts = delta.get("accepted_tokens_total", 0), delta.get("draft_tokens_total", 0), delta.get("drafts_total", 0)
    spec = None
    if drf:
        per = [delta[k] / drafts for k in sorted(delta) if k.startswith("accepted_tokens_per_pos_total[")]
        spec = dict(accepted=acc, drafted=drf, acceptance_rate=round(acc / drf, 4),
                    mean_acceptance_length=round(1 + acc / drafts, 3), per_position=[round(x, 4) for x in per])
    gen = sum(r["tokens"] for r in res)
    return dict(concurrency=conc, temperature=temperature, max_tokens=max_tokens, n=len(res), wall_s=round(wall, 1),
                aggregate_tok_s=round(gen / wall, 1), median_decode_tok_s=st.median(r["decode_tok_s"] for r in res),
                median_ttft_s=st.median(r["ttft"] for r in res), spec=spec,
                finishes=dict((f, sum(r["finish"] == f for r in res)) for f in {r["finish"] for r in res}),
                requests=sorted(res, key=lambda r: r["doc"]))


report = dict(model=MODEL, docs=DOCS, started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
for g in range(1, greedy_runs + 1):
    report[f"greedy_run{g}"] = batch(1, 0.0, 2048)
    print(f"greedy run {g}: done", flush=True)
if sampled:
    for conc in (1, 8):
        report[f"sampled_c{conc}"] = b = batch(conc, 1.0, 4096, seed=1234)
        print(f"sampled c{conc}: aggregate {b['aggregate_tok_s']} tok/s, spec {b['spec']}", flush=True)
json.dump(report, open(f"{out}/probe.json", "w"), indent=1)
print("written", f"{out}/probe.json")
