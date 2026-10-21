#!/usr/bin/env python3
"""Index of request lengths for every logged K3.25 GPQA configuration (host, stdlib only, zero GPU).

messages_sha256 identifies the question (the request messages are the same in every pass), so doc ids come from the
lm-eval samples via the frozen messages_hash() convention (analysis_col511.py). Writes ids, lengths and flags only.
Outputs: out/lengths_all.jsonl (one record per logged request), out/rep070F_index.json (rep070-F requests with
raw-file line positions), out/index_checks.json.
"""
import glob, gzip, hashlib, json, os, sys

GPQA = "<gpqa-runs>"
RAW = "<runs>/rep070/run-F/raw"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")
P = "glm53-flash-exl3-k3.25@tpurtell"
CAP = 327680

CONFIGS = [  # (group, request log, seeds -> lm-eval pass)
    ("stock 0.7.0 (orig)", f"{P}+ps.requests.jsonl", {1235: 2, 1236: 3}),
    ("stock 0.7.0 rep070-F", f"{P}+rep070-F.s1.requests.jsonl", {70701: 1}),
    ("stock 0.7.0 rep070-F", f"{P}+rep070-F.s2.requests.jsonl", {70702: 2, 70703: 3}),
    ("stock 0.9.1", f"{P}+v0.9.1.requests.jsonl", {1234: 1}),
    ("stock 0.9.1", f"{P}+v0.9.1+ps.requests.jsonl", {1235: 2, 1236: 3}),
    ("col511 arm", f"{P}+v0.9.1+col511fix.s1.requests.jsonl", {1234: 1}),
    ("col511 arm", f"{P}+v0.9.1+col511fix.s2.requests.jsonl", {1235: 2, 1236: 3}),
    ("B0", f"{P}+v0.9.1+bundle-d1234.s1.requests.jsonl", {1234: 1}),
    ("B0", f"{P}+v0.9.1+bundle-d1234.s2.requests.jsonl", {1235: 2}),
    ("B0", f"{P}+v0.9.1+bundle-d1234.s3.requests.jsonl", {1236: 3}),
    ("B1", f"{P}+v0.9.1+bundle-d1234+d7.s1.requests.jsonl", {1236: 3}),
]


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def messages_hash(messages):  # frozen convention, analysis_col511.py
    return sha(json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def main():
    os.makedirs(OUT, exist_ok=True)
    checks = []
    # messages hash -> doc id, from every samples file (must agree everywhere)
    m2d = {}
    conflicts = 0
    nfiles = 0
    for f in sorted(glob.glob(f"{GPQA}/{P}*__pass*/**/samples_*.jsonl", recursive=True)):
        nfiles += 1
        for line in open(f):
            r = json.loads(line)
            mh = messages_hash(json.loads(r["arguments"]["gen_args_0"]["arg_0"][0]))
            if mh in m2d and m2d[mh] != r["doc_id"]:
                conflicts += 1
            m2d[mh] = r["doc_id"]
    checks.append(("messages hash -> doc id is one-to-one across all samples files", conflicts == 0 and len(m2d) == 198,
                   f"{nfiles} files, {len(m2d)} hashes, {conflicts} conflicts"))
    recs = []
    for group, log, seeds in CONFIGS:
        rows = [json.loads(l) for l in open(f"{GPQA}/{log}")]
        for seed, lp in seeds.items():
            pr = [r for r in rows if r["seed"] == seed]
            docs = [m2d.get(r["messages_sha256"]) for r in pr]
            checks.append((f"{group} seed {seed}: 198 requests, all joined, 198 distinct docs",
                           len(pr) == 198 and None not in docs and len(set(docs)) == 198, f"{len(pr)} rows"))
            for r, d in zip(pr, docs):
                recs.append({"group": group, "log": log, "seed": seed, "pass": lp, "doc": d, "T": r["completion_tokens"],
                             "finish_reason": r["finish_reason"], "prompt_tokens": r["prompt_tokens"],
                             "capped": r["finish_reason"] == "length" and r["completion_tokens"] == CAP,
                             "content_chars": r.get("content_chars"), "reasoning_chars": r.get("reasoning_chars"),
                             "secs": r.get("secs")})
    with open(f"{OUT}/lengths_all.jsonl", "w") as fo:
        for r in recs:
            fo.write(json.dumps(r, sort_keys=True) + "\n")
    # cross-check against the tail audit's per-request T where both exist
    aud = json.load(open("<runs>/tail-audit/audit.json"))["records"]
    amap = {}
    for a in aud:
        g = a["group"].replace("stock 0.7.0", "stock 0.7.0 (orig)")
        amap[(g, a["seed"], a["doc"])] = a["T"]
    mism = sum(1 for r in recs if (r["group"], r["seed"], r["doc"]) in amap and amap[(r["group"], r["seed"], r["doc"])] != r["T"])
    both = sum(1 for r in recs if (r["group"], r["seed"], r["doc"]) in amap)
    checks.append(("lengths equal tail-audit records where both exist", mism == 0 and both > 0, f"{both} compared, {mism} differ"))

    # rep070-F raw index (line numbers into the raw files; no text)
    idx = []
    for fn in ("s1", "s2"):
        with gzip.open(f"{RAW}/{fn}.raw.jsonl.gz", "rt") as fi:
            for ln, line in enumerate(fi):
                d = json.loads(line)
                ch = d["response"]["choices"][0]
                u = d["response"]["usage"]
                idx.append({"file": fn, "line": ln, "seed": d["seed"], "pass": d["seed"] - 70700,
                            "doc": m2d.get(d["messages_sha256"]), "T": u["completion_tokens"],
                            "prompt_tokens": u["prompt_tokens"], "finish_reason": ch["finish_reason"],
                            "stop_reason": ch.get("stop_reason"),
                            "reasoning_chars": len(ch["message"].get("reasoning") or ""),
                            "content_chars": len(ch["message"].get("content") or ""),
                            "id": f"q{m2d.get(d['messages_sha256'])}p{d['seed'] - 70700}"})
    reqF = {(r["seed"], r["doc"]): r for r in recs if r["group"] == "stock 0.7.0 rep070-F"}
    agree = all(reqF[(x["seed"], x["doc"])]["T"] == x["T"] and reqF[(x["seed"], x["doc"])]["finish_reason"] == x["finish_reason"]
                for x in idx)
    checks.append(("rep070-F raw file: 594 requests, lengths and finish reasons equal the request logs",
                   len(idx) == 594 and agree, len(idx)))
    json.dump(idx, open(f"{OUT}/rep070F_index.json", "w"), indent=1)
    json.dump([{"check": c, "ok": ok, "detail": det} for c, ok, det in checks], open(f"{OUT}/index_checks.json", "w"), indent=1)
    for c, ok, det in checks:
        print("PASS" if ok else "FAIL", c, "|", det)
    at = [x for x in idx if x["T"] >= 100000]
    print("rep070-F at risk >=100k:", len(at), "finished", sum(x["finish_reason"] == "stop" for x in at),
          "capped", sum(x["finish_reason"] == "length" for x in at), "questions", len({x["doc"] for x in at}))
    cmp_ = [x for x in idx if 50000 <= x["T"] < 100000]
    print("comparison 50-100k:", len(cmp_))
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
