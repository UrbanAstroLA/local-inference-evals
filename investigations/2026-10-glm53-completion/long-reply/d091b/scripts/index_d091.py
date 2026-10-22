#!/usr/bin/env python3
"""Raw-file index for the discovery pipeline (host, stdlib, zero GPU). The index part of discovery's build_index.py
(sha256 43b6c311...) with its two hard-wired inputs made arguments: the raw directory / files and the request logs.
Everything else is the same code: the doc join through the frozen messages_hash over every K3.25 samples file, the record
fields, the id convention ("q{doc}p{seed - 70700}", so the d091b pass at seed 70705 has ids q{doc}p5), and the
lengths / finish-reason agreement check against the request logs.

The output file keeps discovery's name, out/rep070F_index.json, because trajectory.py and periods.py (byte-identical
copies, run unchanged) read exactly that path; for d091 it indexes the d091 raw file.

Proof of identity: with the rep070-F inputs (--raw-dir rep070/run-F/raw --files s1 s2 --reqlogs <rep070-F s1, s2>
--expect 594) the output is byte-identical to discovery/out/rep070F_index.json (tests/index-repro.txt).

usage: index_d091.py --raw-dir DIR --files s1 [s2 ...] --reqlogs LOG [LOG ...] --seeds S [S ...] --expect N --out-dir OUT
Writes ids, lengths and flags only.
"""
import argparse, glob, gzip, hashlib, json, os, sys

GPQA = "<gpqa-runs>"
P = "glm53-flash-exl3-k3.25@tpurtell"


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def messages_hash(messages):  # frozen convention, analysis_col511.py
    return sha(json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--reqlogs", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--expect", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    checks = []
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
    for log in a.reqlogs:
        rows = [json.loads(l) for l in open(log)]
        for seed in a.seeds:
            pr = [r for r in rows if r["seed"] == seed]
            if not pr:
                continue
            docs = [m2d.get(r["messages_sha256"]) for r in pr]
            checks.append((f"{os.path.basename(log)} seed {seed}: 198 requests, all joined, 198 distinct docs",
                           len(pr) == 198 and None not in docs and len(set(docs)) == 198, f"{len(pr)} rows"))
            for r, d in zip(pr, docs):
                recs.append({"seed": seed, "doc": d, "T": r["completion_tokens"], "finish_reason": r["finish_reason"]})
    idx = []
    for fn in a.files:
        with gzip.open(f"{a.raw_dir}/{fn}.raw.jsonl.gz", "rt") as fi:
            for ln, line in enumerate(fi):
                d = json.loads(line)
                if d.get("kind", "exchange") != "exchange":   # interrupted records (no response) are never indexed
                    continue
                ch = d["response"]["choices"][0]
                u = d["response"]["usage"]
                idx.append({"file": fn, "line": ln, "seed": d["seed"], "pass": d["seed"] - 70700,
                            "doc": m2d.get(d["messages_sha256"]), "T": u["completion_tokens"],
                            "prompt_tokens": u["prompt_tokens"], "finish_reason": ch["finish_reason"],
                            "stop_reason": ch.get("stop_reason"),
                            "reasoning_chars": len(ch["message"].get("reasoning") or ""),
                            "content_chars": len(ch["message"].get("content") or ""),
                            "id": f"q{m2d.get(d['messages_sha256'])}p{d['seed'] - 70700}"})
    req = {(r["seed"], r["doc"]): r for r in recs}
    agree = all((x["seed"], x["doc"]) in req and req[(x["seed"], x["doc"])]["T"] == x["T"]
                and req[(x["seed"], x["doc"])]["finish_reason"] == x["finish_reason"] for x in idx)
    checks.append((f"raw file: {a.expect} requests, lengths and finish reasons equal the request logs", len(idx) == a.expect and agree, len(idx)))
    checks.append(("raw file: request seeds are exactly the given seeds", {x["seed"] for x in idx} == set(a.seeds), sorted({x["seed"] for x in idx})))
    checks.append(("raw file: one record per (seed, doc)", len({(x["seed"], x["doc"]) for x in idx}) == len(idx), ""))
    json.dump(idx, open(f"{a.out_dir}/rep070F_index.json", "w"), indent=1)
    json.dump([{"check": c, "ok": ok, "detail": det} for c, ok, det in checks], open(f"{a.out_dir}/index_checks_d091.json", "w"), indent=1)
    for c, ok, det in checks:
        print("PASS" if ok else "FAIL", c, "|", det)
    at = [x for x in idx if x["T"] >= 100000]
    print("at risk >=100k:", len(at), "finished", sum(x["finish_reason"] == "stop" for x in at),
          "capped", sum(x["finish_reason"] == "length" for x in at), "questions", len({x["doc"] for x in at}))
    print("comparison 50-100k:", len([x for x in idx if 50000 <= x["T"] < 100000]))
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
