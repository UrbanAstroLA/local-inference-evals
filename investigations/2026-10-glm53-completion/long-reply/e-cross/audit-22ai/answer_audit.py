#!/usr/bin/env python3
"""Amendment 22ai supplementary answer audit (descriptive, never the primary). For every E continuation that stopped
(finish_reason 'stop'): does the text after the last reasoning close yield a final letter under the frozen extractor
(analysis_col511.final_answer), and is it correct? Counts only; no text is written."""
import sys, json, gzip, glob, collections
sys.dont_write_bytecode = True
E = "<runs>/long-reply/e-cross"
sys.path.insert(0, "<runs>/rep070/refreeze/frozen")
import analysis_col511 as M
man = {p["prefix_id"]: p for p in json.load(open(f"{E}/out/prefix_manifest.json"))["prefixes"]}
src = {"d091": "glm53-flash-exl3-k3.25@tpurtell+v0.9.1+d091", "d091b": "glm53-flash-exl3-k3.25@tpurtell+v0.9.1+d091b"}
tgt = {}
for s, tag in src.items():
    f = glob.glob(f"<gpqa-runs>/{tag}__pass1/*/samples_*.jsonl")[0]
    for l in open(f):
        r = json.loads(l)
        if r["filter"] == "flexible-extract":
            tgt[(s, r["doc_id"])] = r["target"].strip("()")
out = {}
for eng in ("v070", "v091"):
    req = {}
    for l in open(f"{E}/run/{eng}/requests.jsonl"):
        r = json.loads(l)
        if r.get("kind") == "request":
            req[(r["prefix_id"], r["j"])] = r
    c = collections.Counter(); by = collections.defaultdict(collections.Counter)
    with gzip.open(f"{E}/run/raw/{eng}.raw.jsonl.gz", "rt") as fh:
        for l in fh:
            r = json.loads(l); q = req[(r["prefix_id"], r["j"])]
            if q["finish_reason"] != "stop":
                continue
            p = man[r["prefix_id"]]; t = r["text"]
            closed = "</think>" in t
            content = t.rsplit("</think>", 1)[1] if closed else ""
            fa = M.final_answer(content) if content.strip() else None
            ok = fa is not None and fa == tgt[(p["source"], p["doc"])]
            for k, v in (("stopped", 1), ("reasoning_closed", closed), ("final_letter", fa is not None), ("correct", ok)):
                c[k] += int(v); by[p["stratum"]][k] += int(v)
    out[eng] = {"all": dict(c), "by_original_stratum": {k: dict(v) for k, v in by.items()}}
json.dump(out, open(f"{E}/audit-22ai/answer_audit.json", "w"), indent=1)
for e, v in out.items():
    print(e, v)
