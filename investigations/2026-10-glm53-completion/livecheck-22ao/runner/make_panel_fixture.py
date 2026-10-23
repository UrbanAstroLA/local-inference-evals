#!/usr/bin/env python3
"""Build the 22ao panel GPQA fixture (copy of bundle/gate/make_gpqa_fixture.py, DOCS changed) (LOCAL ONLY, never publish: contains benchmark questions).
Takes the chat messages of doc_ids 0-23 exactly as lm-eval sent them in the 0.9.1 record's pass 1 (seed 1234)
and checks each message list's hash against the recording proxy's request log (record_proxy.py messages_hash).
usage: make_gpqa_fixture.py OUT_JSON"""
import glob, hashlib, json, sys

SAMPLES = glob.glob("<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1__pass1/*/samples_gpqa_diamond_thinking_*.jsonl")
REQLOGS = ["<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1.requests.jsonl",
           "<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+col511fix.s1.requests.jsonl"]
DOCS = sorted([79, 127, 81, 71, 88, 109, 120, 147, 170, 8, 69, 121])   # 22ao panel (0.9.1 reach rate >= 0.4, f1/rows.jsonl)


def messages_hash(messages):  # = <eval-tools>/diag/record_proxy.py
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


assert len(SAMPLES) == 1, SAMPLES
logged = set()
for p in REQLOGS:
    try:
        for l in open(p):
            d = json.loads(l)
            if d.get("seed") == 1234:
                logged.add(d["messages_sha256"])
    except FileNotFoundError:
        pass
out = {"note": "LOCAL ONLY - GPQA questions; never publish", "source": SAMPLES[0], "items": []}
for l in open(SAMPLES[0]):
    r = json.loads(l)
    if r["doc_id"] not in DOCS or any(i["doc_id"] == r["doc_id"] for i in out["items"]):
        continue
    ga = r["arguments"]["gen_args_0"]
    msgs = json.loads(ga["arg_0"][0])
    kw = ga["arg_1"]
    h = messages_hash(msgs)
    out["items"].append(dict(doc_id=r["doc_id"], messages=msgs, messages_sha256=h, in_request_log=h in logged,
                             chat_template_kwargs=kw.get("chat_template_kwargs"), temperature=kw.get("temperature"),
                             top_p=kw.get("top_p")))
out["items"].sort(key=lambda i: i["doc_id"])
assert [i["doc_id"] for i in out["items"]] == DOCS
json.dump(out, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
print(f"{len(out['items'])} items; messages hash found in a seed-1234 request log: "
      f"{sum(i['in_request_log'] for i in out['items'])}/{len(out['items'])}; "
      f"chat kwargs {out['items'][0]['chat_template_kwargs']}, temperature {out['items'][0]['temperature']}, top_p {out['items'][0]['top_p']}")
