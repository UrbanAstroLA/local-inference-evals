#!/usr/bin/env python3
"""Supplement to seed_scan_d091.py: request seeds inside every gzip raw-body file (*.raw.jsonl.gz) under <data-runs>.
seed_scan_d091.py's text scan does not open .gz files; the raw files of the rep070-F replication and of B+ hold the full
request bodies, so their 'seed' fields are listed here. Read only, standard library; writes seeds and counts only (no
request or reply text is printed or stored).
usage: raw_seed_scan.py OUT.json CANDIDATE [CANDIDATE ...]
"""
import glob, gzip, json, sys
from collections import Counter

SKIP = "<runs>/long-reply/d091b"


def main():
    out_path, cands = sys.argv[1], [int(x) for x in sys.argv[2:]]
    files = {}
    allseeds = Counter()
    for p in sorted(glob.glob("<data-runs>/**/*.raw.jsonl.gz", recursive=True)):
        if p.startswith(SKIP):
            continue
        c, n, bad = Counter(), 0, 0
        try:
            with gzip.open(p, "rt") as f:
                for line in f:
                    if not line.strip():
                        continue
                    n += 1
                    try:
                        d = json.loads(line)
                    except ValueError:
                        bad += 1
                        continue
                    s = d.get("seed")
                    if s is None and isinstance(d.get("request"), dict):
                        s = d["request"].get("seed")
                    c[str(s)] += 1
                    if isinstance(d.get("request"), dict) and "seed" in d["request"] and str(d["request"]["seed"]) != str(s):
                        c[str(d["request"]["seed"])] += 1
        except OSError as e:
            files[p] = {"error": type(e).__name__}
            continue
        files[p] = {"records": n, "unparsed": bad, "seeds": dict(sorted(c.items()))}
        allseeds.update(c)
    res = {"what": "request seeds in gzip raw-body files under <data-runs> (seeds and counts only)", "files": files,
           "distinct_seeds": len(allseeds),
           "candidates": {str(x): {"in_raw_files": str(x) in allseeds} for x in cands}}
    json.dump(res, open(out_path, "w"), indent=1)
    for p, v in files.items():
        print(p, {k: v[k] for k in v if k != "seeds"}, "distinct seeds", len(v.get("seeds", {})))
    print(json.dumps(res["candidates"]))
    return 0 if not any(v["in_raw_files"] for v in res["candidates"].values()) else 3


if __name__ == "__main__":
    sys.exit(main())
