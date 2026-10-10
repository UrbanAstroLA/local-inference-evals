#!/usr/bin/env python3
"""Recompute every number in this investigation's README from the published rows (standard library only).

    python3 investigations/2026-10-glm53-looping/recompute.py

Reads runs/*/results.jsonl, run.json and, for server-log figures, server_log.jsonl (numeric fields parsed from the engine's log;
the log text is not published).
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from verify import config_label, summarize  # noqa: E402
import analyze  # noqa: E402
from analyze import sign_test  # noqa: E402

F = "glm53-flash/"
AS, FIX = F + "k3.25-v0.9.0-dflash3", F + "k3.25-v0.9.0-tailfix-dflash3"
NAMES = {AS: "0.9.0 as released", FIX: "0.9.0 + DCP1 tail fix ≈ 0.9.1"}


def load(protocol, config):
    out = []
    for d in sorted((ROOT / "runs").iterdir()):
        m = json.loads((d / "run.json").read_text())
        if m["protocol"] == protocol and m["config"] == config:
            m["rows"] = [json.loads(l) for l in (d / "results.jsonl").read_text().splitlines() if l.strip()]
            m["cfg"] = json.loads((ROOT / "configs" / f"{config}.json").read_text())
            out.append(m)
    return out


print("# The DCP1 tail issue and its fix\n")
print("## Index check (kpool-tail-index/v1)")
for c in (AS, FIX):
    m = load("kpool-tail-index/v1", c)[0]
    print(f"{m['id']}: {summarize(m['protocol'], m['rows'])['by_layout']}")
    print("   packed lengths with the tail dropped:", [r["length"] for r in m["rows"] if r["layout"] == "packed" and not r["tail_attended"]])
a, b = (load("kpool-tail-index/v1", c)[0]["rows"] for c in (AS, FIX))
same = [(x["layout"], x["length"]) for x, y in zip(a, b) if x["row_sha256_16"] == y["row_sha256_16"]]
ok = all(x["n_dropped"] == 0 and x["tail_attended"] for x in a if (x["layout"], x["length"]) in same)
print(f"rows byte-identical between images: {len(same)}; all of them rows the stock image already handled: {ok}")
print("fixed packed rows equal the dense prefix (L <= 2047):", all(r["equals_dense_prefix"] for r in b if r["layout"] == "packed" and r["length"] <= 2047))

print("\n## Decode vs prefill (decode-prefill-consistency/v1), one run per build")
dp = {c: load("decode-prefill-consistency/v1", c)[0] for c in (F + "k3.25-v0.9.0-nospec-nocache", F + "k3.25-v0.9.0-tailfix-nospec-nocache")}
for c, m in dp.items():
    s = summarize(m["protocol"], m["rows"])["by_region"]
    print(f"{m['id']}: " + "; ".join(f"{k}: n {v['positions']}, KL {v['mean_kl_top20']:.4f}, top-1 {100*v['top1_agree']:.1f}%"
                                     for k, v in s.items() if "|" not in k))
(sc, sm), (fc, fm) = dp.items()
per = []
for d in range(6):
    x = [r["kl_top20"] for r in sm["rows"] if r["doc_id"] == d and r["region"] == "lt2044" and r["kl_top20"] is not None]
    y = [r["kl_top20"] for r in fm["rows"] if r["doc_id"] == d and r["region"] == "lt2044" and r["kl_top20"] is not None]
    per.append((sum(x) / len(x), sum(y) / len(y)))
lower = sum(y < x for x, y in per)
print(f"lt2044 per-prompt mean KL (stock, fix) {[(round(x, 4), round(y, 4)) for x, y in per]}; fix lower in {lower}/6 "
      f"(two-sided sign test p = {sign_test(lower, 6 - lower):.3f})")
print("prompt lengths (first generated position per prompt):", {d: min(r["i"] for r in sm["rows"] if r["doc_id"] == d) for d in range(6)})
for reg in ("lt2044|mod0", "lt2044|mod1", "lt2044|mod2", "lt2044|mod3"):
    print(f"  {reg}: " + ", ".join(f"{summarize(m['protocol'], m['rows'])['by_region'][reg]['mean_kl_top20']:.4f}" for m in dp.values()))

print("\n## Decode vs prefill, every run (prompts 0-5 shared by all), the 0.9.1 noise floor and single-request divergence")
analyze.decode_prefill()

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

print("\n## Serving probe (serving-probe/v1, single runs) and KV pools (server_log.jsonl -> summary.json 'server')")
for c in (AS, FIX):
    m = load("serving-probe/v1", c)[0]
    bt = summarize(m["protocol"], m["rows"])["batches"]
    print(f"{m['id']}: " + "; ".join(f"{k}: decode {v['median_decode_tok_s']} tok/s, acceptance {v['acceptance_rate']}" for k, v in bt.items()))
for d in sorted((ROOT / "runs").iterdir()):
    if (d / "server_log.jsonl").exists() and ("bisect1" in d.name or "screen-doc" in d.name or "gpqa-diamond" in d.name):
        sv = json.loads((d / "summary.json").read_text())["server"]
        print(f"KV pool, {d.name}: {sv['kv_pool_tokens']:,}; peak usage {sv['peak_kv_usage_pct']}%; at most {sv['max_running']} running; "
              f"waiting below max seqs {sv['waiting_below_max_seqs']} of {sv['status_lines']}")

print("\n## Kernel tests (kpool-kernel-tests/v1), upstream suite")
for d in sorted((ROOT / "runs").iterdir()):
    if d.name.endswith("_kpool-kernel-tests"):
        rows = [json.loads(l) for l in (d / "results.jsonl").read_text().splitlines() if l.strip()]
        up = [r for r in rows if r["suite"] == "upstream" and r["outcome"] != "skipped"]
        print(f"{d.name}: {sum(r['outcome'] == 'passed' for r in up)}/{len(up)} passed")

print("\n# Non-completion: component screen (hard-prompt-screen/v2) and the fixed-seed control\n")
analyze.screen_v2()
print("\n## Minimum detectable differences (tests, power and scope)")
analyze.power_table()
print("\n# Single draws from the earlier screens (LEDGER.md)\n")
analyze.screen_single()
print("\n# Across GPQA: three passes per configuration (one request seed per pass)\n")
analyze.gpqa_passes()
print()
analyze.gpqa_records([])
print()
analyze.gpqa_empty()
print("\n# GPQA Diamond, pass 1 of every configuration\n")
analyze.gpqa_table()
print()
analyze.gpqa_pairs([])
print("\n# How the failures look\n")
analyze.anatomy()
