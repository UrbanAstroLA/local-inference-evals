#!/usr/bin/env python3
"""D7 exposure accounting, step 4: descriptive table (exposure class x empty reply), per pass and per record.
Purely descriptive: no tests, no causal reading. Reads gpqa_exposure_per_request.jsonl; writes exposure_tables.md."""
import collections
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["KNOWN", "POSSIBLE", "NOT-aligned-split", "NOT-unsplittable", "UNKNOWN"]
rows = [json.loads(l) for l in open(os.path.join(HERE, "gpqa_exposure_per_request.jsonl"))]
out = []
out.append("| Record | Pass (seed) | Timing | " + " | ".join(f"{c} E/A" for c in CLASSES) + " | Empty total |")
out.append("|---|---|---|" + "---:|" * (len(CLASSES) + 1))
recs = collections.OrderedDict()
for r in rows:
    recs.setdefault(r["record"], collections.OrderedDict()).setdefault((r["pass"], r["seed"]), []).append(r)
agg = collections.OrderedDict()
for rec, passes in recs.items():
    A = agg.setdefault(rec, collections.Counter())
    for (p, seed), rs in passes.items():
        c = collections.Counter((x["exposure"], x["empty"]) for x in rs)
        for k, v in c.items():
            A[k] += v
        timing = "logged (proxy)" if rs[0]["send_t_rel"] is not None else "not logged"
        cells = []
        for cl in CLASSES:
            e, a = c[(cl, True)], c[(cl, False)]
            cells.append("-" if e + a == 0 else f"{e}/{a}")
        out.append(f"| {rec} | {p} ({seed}) | {timing} | " + " | ".join(cells) + f" | {sum(x['empty'] for x in rs)} |")
out.append("")
out.append("Per record (all passes pooled; question-runs):")
out.append("")
out.append("| Record | " + " | ".join(f"{c} E/A" for c in CLASSES) + " | Empty / runs |")
out.append("|---|" + "---:|" * (len(CLASSES) + 1))
for rec, A in agg.items():
    cells = []
    for cl in CLASSES:
        e, a = A[(cl, True)], A[(cl, False)]
        cells.append("-" if e + a == 0 else f"{e}/{a}")
    n = sum(A.values())
    out.append(f"| {rec} | " + " | ".join(cells) + f" | {sum(v for (c, e), v in A.items() if e)}/{n} |")
out.append("")
out.append("Doc 127 (2,795 prompt tokens) per pass:")
out.append("")
out.append("| Record | Pass | Class | Ensemble exposed | Decode rows at admission (analytic range) | First chunk | Empty | Finish | Completion tokens |")
out.append("|---|---|---|---:|---|---:|---|---|---:|")
for r in rows:
    if r["doc_id"] != 127:
        continue
    a = r["analytic"]
    adm = r["central_schedule_admission"] or {}
    fr = "-" if r["ensemble_exposed_fraction"] is None else f"{r['ensemble_exposed_fraction']:.2f}"
    dr = a.get("d_range", "unknown")
    fc = adm.get("first_chunk", "unknown (exposed unless d in {1,5})" if r["engine"] == "0.7.0" else "?")
    out.append(f"| {r['record']} | {r['pass']} | {r['exposure']} | {fr} | {dr} | {fc} | {r['empty']} | {r['finish_reason']} | {r['completion_tokens']} |")
open(os.path.join(HERE, "exposure_tables.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
