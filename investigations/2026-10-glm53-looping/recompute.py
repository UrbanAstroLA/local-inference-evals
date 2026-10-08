#!/usr/bin/env python3
"""Recompute every number in this investigation's README from the published rows (standard library only).

    python3 investigations/2026-10-glm53-looping/recompute.py

Reads runs/*/results.jsonl and run.json (server-log aggregates are quoted from run notes, the logs are not published).
"""
import json, math, random, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from verify import config_label, wilson, summarize  # noqa: E402
from analyze import fisher, sign_test  # noqa: E402

F = "glm53-flash/"
AS, FIX, CTRL = F + "k3.25-v0.9.0-dflash3", F + "k3.25-v0.9.0-tailfix-dflash3", F + "k3.25-v0.9.0-ep2dcp2-dflash3"
NAMES = {AS: "0.9.0 as released", FIX: "0.9.0 + DCP1 tail fix ≈ 0.9.1", CTRL: "0.9.0, 0.7.0 layout (control)"}


def load(protocol, config):
    out = []
    for d in sorted((ROOT / "runs").iterdir()):
        m = json.loads((d / "run.json").read_text())
        if m["protocol"] == protocol and m["config"] == config:
            m["rows"] = [json.loads(l) for l in (d / "results.jsonl").read_text().splitlines() if l.strip()]
            m["cfg"] = json.loads((ROOT / "configs" / f"{config}.json").read_text())
            out.append(m)
    return out


def failed(r):
    return r["cls"] in ("loop", "exhaust")


print("## Configurations")
for c in (AS, FIX, CTRL):
    print(f"- {NAMES[c]}: {config_label(load('hard-prompt-screen/v1', c)[0]['cfg'])}  ({c})")

print("\n## Hard-question screen (hard-prompt-screen/v1), two screens per configuration")
screens = {c: load("hard-prompt-screen/v1", c) for c in (AS, FIX, CTRL)}
tot = {}
for c, ms in screens.items():
    rows = [r for m in ms for r in m["rows"]]; k = sum(map(failed, rows)); lo, hi = wilson(k, len(rows)); tot[c] = (k, len(rows))
    q = {d: sum(failed(r) for r in rows if r["doc_id"] == d) for d in (13, 79, 88, 121, 127)}
    print(f"{NAMES[c]}: failures {k}/{len(rows)} (95% Wilson {100*lo:.1f}-{100*hi:.1f}%), loops {sum(r['cls']=='loop' for r in rows)}, "
          f"exhaustions {sum(r['cls']=='exhaust' for r in rows)}, by question (of 16) {q}; runs {[m['id'] for m in ms]}")
for a, b in ((AS, CTRL), (AS, FIX), (FIX, CTRL)):
    (ka, na), (kb, nb) = tot[a], tot[b]
    print(f"Fisher exact, {NAMES[a]} vs {NAMES[b]}: {ka}/{na} vs {kb}/{nb}, p = {fisher(ka, na - ka, kb, nb - kb):.4f}")
for c, ms in screens.items():
    for m in ms:
        v = [r["completion_tokens"] / r["secs"] for r in m["rows"] if r["finish_reason"] == "stop" and r.get("completion_tokens")]
        print(f"{m['id']}: median completion tok/s of finished requests {st.median(v):.1f}; notes: {m['notes'].split('Server log')[1]}")

print("\n## Earlier screens, question 13 (failures of 8)")
for d in sorted((ROOT / "runs").iterdir()):
    m = json.loads((d / "run.json").read_text())
    if m["protocol"].startswith("hard-prompt-screen") and "INVALID" not in m["notes"] and "bisect" not in m["id"]:
        rows = [json.loads(l) for l in (d / "results.jsonl").read_text().splitlines() if l.strip()]
        print(f"{m['id']}: {sum(failed(r) for r in rows if r['doc_id'] == 13)}")

print("\n## Decode vs prefill (decode-prefill-consistency/v1)")
dp = {c: load("decode-prefill-consistency/v1", c)[0] for c in (F + "k3.25-v0.9.0-nospec-nocache", F + "k3.25-v0.9.0-tailfix-nospec-nocache")}
for c, m in dp.items():
    s = summarize(m["protocol"], m["rows"])["by_region"]
    print(f"{m['id']}: " + "; ".join(f"{k}: n {v['positions']}, KL {v['mean_kl_top20']:.4f}, top-1 {100*v['top1_agree']:.1f}%, |dlp| {v['mean_abs_dlp']:.4f}"
                                     for k, v in s.items()))
(sc, sm), (fc, fm) = dp.items()
for reg in ("lt2044", "ge2048"):
    per = []
    for d in range(6):
        a = [r["kl_top20"] for r in sm["rows"] if r["doc_id"] == d and r["region"] == reg and r["kl_top20"] is not None]
        b = [r["kl_top20"] for r in fm["rows"] if r["doc_id"] == d and r["region"] == reg and r["kl_top20"] is not None]
        per.append((sum(a) / len(a), sum(b) / len(b)))
    lower = sum(b < a for a, b in per)
    print(f"{reg}: per-prompt mean KL (stock, fix) {[(round(a, 4), round(b, 4)) for a, b in per]}; fix lower in {lower}/6 "
          f"(two-sided sign test p = {sign_test(lower, 6 - lower):.3f})")
print("prompt lengths (first generated position per prompt):", {d: min(r["i"] for r in sm["rows"] if r["doc_id"] == d) for d in range(6)})

print("\n## Index check (kpool-tail-index/v1)")
for c in (AS, FIX):
    m = load("kpool-tail-index/v1", c)[0]
    print(f"{m['id']}: {summarize(m['protocol'], m['rows'])['by_layout']}")
    print("   packed lengths with the tail dropped:", [r["length"] for r in m["rows"] if r["layout"] == "packed" and not r["tail_attended"]])
a, b = (load("kpool-tail-index/v1", c)[0]["rows"] for c in (AS, FIX))
same = [(x["layout"], x["length"]) for x, y in zip(a, b) if x["row_sha256_16"] == y["row_sha256_16"]]
ok = all(x["n_dropped"] == 0 and x["tail_attended"] for x in a if (x["layout"], x["length"]) in same)
print(f"rows byte-identical between images: {len(same)}; all of them rows the stock image already handled: {ok}")
print("fixed packed rows equal the dense prefix (L <= 2047):", all(r["equals_dense_prefix"] for r in b if r["layout"] == "packed" and r["length"] <= 2047))

print("\n## Kernel tests (kpool-kernel-tests/v1), upstream suite")
for d in sorted((ROOT / "runs").iterdir()):
    if d.name.endswith("_kpool-kernel-tests"):
        rows = [json.loads(l) for l in (d / "results.jsonl").read_text().splitlines() if l.strip()]
        up = [r for r in rows if r["suite"] == "upstream" and r["outcome"] != "skipped"]
        print(f"{d.name}: {sum(r['outcome'] == 'passed' for r in up)}/{len(up)} passed")

print("\n## tool-eval-bench (tool-eval-bench/v1)")
te = {c: load("tool-eval-bench/v1", c)[0] for c in (AS, FIX)}
for c, m in te.items():
    print(f"{m['id']}: {summarize(m['protocol'], m['rows'])['reps']}")
st_ = {c: {(r["rep"], r["scenario_id"]): r["status"] for r in m["rows"]} for c, m in te.items()}
ids = sorted({sid for (_, sid) in st_[AS]})
for sid in ("TC-80", "TC-88"):
    print(sid, {NAMES[c]: [st_[c][(rep, sid)] for rep in (1, 2)] for c in (AS, FIX)})
within = [sid for sid in ids if any(st_[c][(1, sid)] != st_[c][(2, sid)] for c in (AS, FIX))]
print(f"scenarios whose status differs between repeats of the same build: {len(within)} {within}")

print("\n## Serving probe (serving-probe/v1)")
for c in (AS, FIX):
    m = [x for x in load("serving-probe/v1", c)][0]
    b = summarize(m["protocol"], m["rows"])["batches"]
    print(f"{m['id']}: " + "; ".join(f"{k}: decode {v['median_decode_tok_s']} tok/s, acceptance {v['acceptance_rate']}" for k, v in b.items()))

print("\n## Sample size behind each comparison (normal approximation, two-sided p < 0.05, 80% power, ignores clustering)")
def n_per_arm(p1, p2, za=1.959964, zb=0.841621):
    pb = (p1 + p2) / 2
    return math.ceil((za * math.sqrt(2 * pb * (1 - pb)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p1 - p2) ** 2)
for a, b in ((AS, CTRL), (AS, FIX), (FIX, CTRL)):
    print(f"{NAMES[a]} vs {NAMES[b]}: {n_per_arm(tot[a][0] / tot[a][1], tot[b][0] / tot[b][1])} requests per configuration")
