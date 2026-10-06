#!/usr/bin/env python3
"""hard-prompt-screen/v1 client. Usage: GPQA_SAMPLES=<samples.jsonl> [BASE_URL=...] hard_prompt_screen.py OUTDIR REPEATS DOC [DOC ...]
Protocol: protocols/hard-prompt-screen/v1.md. Request body, seed (1234), order (Random(7)), concurrency (8), classes and
early stop as specified there. Prompts are the lm-eval-rendered chat messages taken from GPQA_SAMPLES."""
import json, sys, os, time, glob, threading, queue, random, zlib, requests
out, reps, docs = sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3:]]
os.makedirs(out, exist_ok=True)
SRC = os.environ["GPQA_SAMPLES"]   # any gpqa-diamond/v1 lm-eval samples_*.jsonl (supplies rendered prompts)
prompts = {}
for l in open(SRC):
    r = json.loads(l)
    prompts.setdefault(r['doc_id'], json.loads(r['arguments']['gen_args_0']['arg_0'][0]))
MODEL = requests.get(os.environ.get('BASE_URL', 'http://127.0.0.1:8001') + '/v1/models').json()['data'][0]['id']
lock = threading.Lock()
def log(m):
    with lock: print(time.strftime('%H:%M:%S'), m, flush=True)
def zr(s):
    t = s[-30000:].encode()
    return round(len(zlib.compress(t)) / max(1, len(t)), 4)
def run(doc, rep):
    fn = f'{out}/doc{doc}_rep{rep}.json'
    if os.path.exists(fn): return
    body = dict(model=MODEL, messages=prompts[doc], temperature=1.0, top_p=0.95, seed=1234,
                max_tokens=327680, stop=['</s>'], stream=True, stream_options={'include_usage': True},
                chat_template_kwargs={'enable_thinking': True})
    t0 = time.time(); reasoning = []; content = []; fin = None; usage = None; nchunk = 0
    nchars = 0; next_check = 30000; low_streak = 0; stopped_early = False; chk = []
    try:
        with requests.post(os.environ.get('BASE_URL', 'http://127.0.0.1:8001') + '/v1/chat/completions', json=body, stream=True, timeout=10800) as resp:
            for line in resp.iter_lines():
                if not line.startswith(b'data: '): continue
                p = line[6:]
                if p == b'[DONE]': break
                j = json.loads(p)
                if j.get('usage'): usage = j['usage']
                for c in j.get('choices', []):
                    d = c.get('delta', {})
                    rc = d.get('reasoning_content') or d.get('reasoning')
                    if rc: reasoning.append(rc); nchars += len(rc)
                    if d.get('content'): content.append(d['content'])
                    if c.get('finish_reason'): fin = c['finish_reason']
                nchunk += 1
                if nchars >= next_check:
                    next_check += 10000
                    q_ = zr(''.join(reasoning)); chk.append(q_)
                    low_streak = low_streak + 1 if q_ < 0.10 else 0
                    if low_streak >= 3:
                        stopped_early = True; fin = 'length'; break
                if nchunk % 20000 == 0: log(f'doc{doc} rep{rep} chunks={nchunk} {int(time.time()-t0)}s')
    except Exception as e:
        fin = f'ERROR {e!r}'
    r, c = ''.join(reasoning), ''.join(content)
    body_text = r if r else c
    ratio = zr(body_text)
    if fin is None:   # a healthy stream always ends with a finish reason (early stop sets 'length')
        fin = 'ERROR stream ended without a finish reason (server error or engine down)'
    cls = 'ok' if fin == 'stop' else ('error' if str(fin).startswith('ERROR') else ('loop' if (stopped_early or ratio < 0.15) else 'exhaust'))
    rec = dict(doc=doc, rep=rep, seed=1234, finish_reason=fin, cls=cls, stopped_early=stopped_early, zlib_checks=chk, usage=usage, chunks=nchunk,
               reasoning_chars=len(r), content_chars=len(c), secs=round(time.time()-t0),
               tail_zlib_ratio=ratio, tail=body_text[-3000:], content=c[-2000:], reasoning_head=r[:1500])
    json.dump(rec, open(fn, 'w'))
    open(f'{out}/doc{doc}_rep{rep}.reasoning.txt', 'w').write(r)
    log(f'DONE doc{doc} rep{rep} {cls}{" (stopped early)" if stopped_early else ""} finish={fin} tokens={(usage or {}).get("completion_tokens")} zlib={ratio} {rec["secs"]}s')
order = [(d, rep) for rep in range(1, reps+1) for d in docs]
random.Random(7).shuffle(order)
q = queue.Queue()
for x in order: q.put(x)
def worker():
    while True:
        try: d, rep = q.get_nowait()
        except queue.Empty: return
        run(d, rep)
ts = [threading.Thread(target=worker) for _ in range(8)]
[t.start() for t in ts]; [t.join() for t in ts]
log('ALL DONE')
