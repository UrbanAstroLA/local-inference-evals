#!/usr/bin/env python3
"""decode-prefill-consistency/v1 client (protocols/decode-prefill-consistency/v1.md). Needs a running server with
speculation off and prefix caching off.

Per prompt:
  1. Generate N tokens at concurrency 1, recording the decode-time logprobs.
     Settings: temperature 1.0, top_p 1.0, seed 1234 + doc id, logprobs 20, ignore_eos so every trace crosses 2,048 tokens.
  2. Re-score the same token ids by prefill, with prompt_logprobs 20:
     (i)  ids[:2048], the short-prefill path, for positions below 2,048;
     (ii) the full sequence, for positions >= 2048.
  3. For each generated position i (the decode step whose causal length is i), compare decode vs prefill:
       dlp    = |logprob(sampled token)| difference;
       top1   = whether the top-1 token agrees;
       kl_int = KL over the shared top-20 tokens, renormalised.

PROMPT_IDS is a JSON file {"ids": {"<doc_id>": [token ids]}}: the gpqa-diamond/v1 lm-eval chat messages rendered with the
served chat template (add_generation_prompt, enable_thinking=True). It is derived from the dataset and not published.

Usage: decode_prefill_consistency.py PROMPT_IDS OUT.json [--prompts 0,1,2,3,4,5] [--n 2600] [--base http://127.0.0.1:8001]
Stdlib only. The published rows rename dlp/top1/kl_int to abs_dlp/top1_agree/kl_top20 (see SCHEMA.md).
"""

import argparse
import json
import math
import time
import urllib.request



def post(base, path, body, timeout=3600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def model_id(base):
    with urllib.request.urlopen(base + "/v1/models", timeout=30) as r:
        return json.loads(r.read())["data"][0]["id"]


def tid(key):
    return int(key.split(":", 1)[1]) if isinstance(key, str) and key.startswith("token_id:") else int(key)


def prompt_lp(base, model, ids):
    r = post(base, "/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 1.0,
                                       "prompt_logprobs": 20, "seed": 1})
    out = []
    for d in r["choices"][0]["prompt_logprobs"]:
        out.append(None if d is None else {int(k): (v["logprob"] if isinstance(v, dict) else v) for k, v in d.items()})
    return out


def compare(dec_top, dec_tok, dec_lp, pre):
    d = dict(dec_top)
    dlp = abs(dec_lp - pre[dec_tok]) if dec_tok in pre else None
    top1 = max(d, key=d.get) == max(pre, key=pre.get)
    common = [t for t in d if t in pre]
    kl = None
    if len(common) >= 2:
        za = math.log(sum(math.exp(d[t]) for t in common))
        zb = math.log(sum(math.exp(pre[t]) for t in common))
        kl = sum(math.exp(pre[t] - zb) * ((pre[t] - zb) - (d[t] - za)) for t in common)
    return dlp, top1, kl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt_ids")
    ap.add_argument("out")
    ap.add_argument("--prompts", default="0,1,2,3,4,5")
    ap.add_argument("--n", type=int, default=2600)
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    a = ap.parse_args()
    ids_all = json.load(open(a.prompt_ids))["ids"]
    model = model_id(a.base)
    rows = []
    for p in a.prompts.split(","):
        prompt = ids_all[p]
        t0 = time.time()
        g = post(a.base, "/v1/completions", {
            "model": model, "prompt": prompt, "max_tokens": a.n, "temperature": 1.0, "top_p": 1.0,
            "seed": 1234 + int(p), "logprobs": 20, "ignore_eos": True,
            "return_tokens_as_token_ids": True, "return_token_ids": True})
        ch = g["choices"][0]
        lp = ch["logprobs"]
        gen = ch.get("token_ids") or [tid(t) for t in lp["tokens"]]
        full = list(prompt) + list(gen)
        pre_a = prompt_lp(a.base, model, full[:2048])
        pre_b = prompt_lp(a.base, model, full) if len(full) > 2048 else []
        for j, tok in enumerate(gen):
            i = len(prompt) + j                    # absolute index of the generated token
            src = pre_a if i < 2048 else pre_b
            pre = src[i] if i < len(src) else None
            top = lp["top_logprobs"][j]
            if pre is None or top is None or lp["token_logprobs"][j] is None:
                continue
            dlp, top1, kl = compare({tid(k): v for k, v in top.items()}, tok, lp["token_logprobs"][j], pre)
            rows.append({"prompt": p, "i": i, "mod4": i % 4,
                         "region": "lt2044" if i < 2044 else ("2044-2047" if i < 2048 else "ge2048"),
                         "dlp": dlp, "top1": top1, "kl_int": kl})
        print(f"prompt {p}: {len(gen)} tokens, {time.time() - t0:.0f} s", flush=True)
    summ = {}
    for r in rows:
        k = f"{r['region']}|mod{r['mod4']}"
        s = summ.setdefault(k, {"n": 0, "dlp": 0.0, "top1": 0, "kl": 0.0, "nkl": 0})
        s["n"] += 1
        s["dlp"] += r["dlp"] or 0.0
        s["top1"] += r["top1"]
        if r["kl_int"] is not None:
            s["kl"] += r["kl_int"]
            s["nkl"] += 1
    table = {k: {"n": s["n"], "mean_abs_dlp": s["dlp"] / s["n"], "top1_agree": s["top1"] / s["n"],
                 "mean_kl_int": (s["kl"] / s["nkl"]) if s["nkl"] else None} for k, s in sorted(summ.items())}
    json.dump({"model": model, "n": a.n, "prompts": a.prompts, "summary": table, "rows": rows}, open(a.out, "w"))
    for k, v in table.items():
        print(f"{k:16s} n={v['n']:6d} |dlp|={v['mean_abs_dlp']:.5f} top1={v['top1_agree']:.4f} kl={v['mean_kl_int']}")


if __name__ == "__main__":
    main()
