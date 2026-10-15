#!/usr/bin/env python3
"""Post-inspection (diagnosis-fc1, CPU only, host python, reads frozen records, writes only into diagnosis-fc1/).

1. quick_struct sub-checks per C1 request (error / ids_equal / cached_tokens == cached_expected), recomputed.
2. Cached-token rule: planned c0 = floor(P/4352)*4352 vs the B1 engine rule for a prompt identical to (or inside the
   last partial block of) a cached warm prompt, with the DFlash EAGLE last-block drop and block-aligned lookups
   (fine-grained hits disabled by KpoolTailManager): (floor((P-1)/4352) - 1) * 4352.
3. Scheduler-side confirmation (mapguard ms_p274.jsonl): num_computed before the first scheduled chunk of each C1
   request; which warm MLA blocks each C1 request re-used; MLA blocks of a warm prefix re-allocated to another
   request before that fixture's C1 lookup (eviction evidence).
4. LRU horizon: prompt tokens scheduled between the warm-M3 free and the re-allocation of its MLA block, and the free
   queue length, i.e. blocks churned per 4,352 prompt tokens.
5. The same rule on earlier sessions of this stack (exact-repeat prompts).

usage: cache_rule.py   (paths fixed below)   -> ../cache_rule.json, stdout
"""
import collections
import json
import os

INV = "<runs>"
FC = INV + "/first-contrast"
RUN = FC + "/run-fc1"
OUT = FC + "/diagnosis-fc1/cache_rule.json"
B = 4352


def jl(p):
    return [json.loads(x) for x in open(p) if x.strip()]


def eagle_rule(P):
    return max(0, ((P - 1) // B - 1) * B)


res = {}
CL = jl(RUN + "/client.jsonl")
reqs = [r for r in CL if r.get("kind") == "request"]
warm = {r["fixture"]: r for r in reqs if r["phase"] == "warm-s1"}

# 1 + 2 -------------------------------------------------------------------------------------------------------
qs = []
for r in reqs:
    if r["phase"] != "C1":
        continue
    w = warm[r["fixture"]]
    qs.append(dict(tag=r["tag"], P=r["P"], warm_P=w["P"], same_prompt=r["prompt_sha256"] == w["prompt_sha256"],
                   salt_same=r["salt"] == w["salt"], error=r.get("error"), ids_equal=r.get("ids_equal"),
                   W=r["W"], completion_tokens=r["completion_tokens"], finish=r["finish"],
                   cached_expected=r["cached_expected"], cached=r["cached_tokens"],
                   blocks_expected=r["cached_expected"] // B, blocks_observed=r["cached_tokens"] // B,
                   eagle_rule=eagle_rule(r["P"]), matches_eagle_rule=r["cached_tokens"] == eagle_rule(r["P"]),
                   fails=[k for k, bad in (("error", bool(r.get("error"))), ("ids_equal", r.get("ids_equal") is not True),
                                           ("cached", r["cached_tokens"] != r["cached_expected"])) if bad]))
res["quick_struct_C1"] = qs
res["other_phases_cached"] = [dict(tag=r["tag"], P=r["P"], cached=r["cached_tokens"], expected=r["cached_expected"])
                              for r in reqs if r["phase"] in ("LW1", "LW2", "warm-s1")
                              and r["cached_tokens"] != r["cached_expected"]]

# 3 -----------------------------------------------------------------------------------------------------------
tags = {}
for p in ("fw_r0_p370.jsonl",):
    for r in jl(RUN + "/forcewin/" + p):
        if r.get("kind") == "add":
            tags[r["req"]] = (r["tag"], r["P"])
S = [r for r in jl(RUN + "/mapguard/ms_p274.jsonl") if r.get("kind") == "sched"]
info = {}
for r in S:
    for q, v in r["reqs"].items():
        d = info.setdefault(q, dict(first=r["seq"], computed_before=v[0] - v[1], mla=None, fin=None, last=v[0]))
        d["last"] = v[0]
        ch = r["changed"].get(q)
        if ch:
            d["mla"] = ch[0]
            d.setdefault("mla_first", ch[0])
    for q in r["finished"]:
        if q in info:
            info[q]["fin"] = r["seq"]
# warm requests (unkeyed, not in the hook log) identified by the scheduler: first seq in warm-s1's span, final length
# (A8 and L8 share P and prompt: equal lengths are matched by completion order, scheduler fin vs client t1)
warm_ids = {}
cand = sorted([q for q, d in info.items() if 1578 <= d["first"] <= 1960 and d["computed_before"] == 0],
              key=lambda q: info[q]["fin"])
for n, w in sorted(warm.items(), key=lambda kv: kv[1]["t1"]):
    for q in cand:
        if q not in warm_ids.values() and info[q]["last"] in (w["P"], w["P"] + 1):
            warm_ids[n] = q
            break
c1_ids = {tags[q][0][3:]: q for q in info if q in tags and tags[q][0].startswith("C1-")}
owner, events = {}, []
for r in S:
    for q, ch in r["changed"].items():
        for i, b in enumerate(ch[0]):
            pv = owner.get(b)
            if pv is None or pv[0] != q:
                events.append((r["seq"], b, q, i, pv))
            owner[b] = (q, i, r["seq"])
sched = {}
for n, cq in c1_ids.items():
    wq = warm_ids.get(n)
    wd, cd = info[wq], info[cq]
    reused_by_others = [dict(seq=s, block=b, warm_idx=wd["mla"].index(b), new_owner=q[-12:])
                        for (s, b, q, i, pv) in events if b in wd["mla"] and wd["fin"] < s < cd["first"] and q != cq]
    sched[n] = dict(c1_first_seq=cd["first"], computed_before=cd["computed_before"],
                    warm_first_seq=wd["first"], warm_fin_seq=wd["fin"], warm_mla_blocks=len(wd["mla"]),
                    c1_reused_warm_mla=len(set(wd["mla"]) & set(cd["mla_first"])),
                    warm_mla_reallocated_before_lookup=reused_by_others)
res["scheduler"] = sched
res["c1_admission_order"] = sorted(c1_ids, key=lambda n: info[c1_ids[n]]["first"])
res["warm_completion_order"] = sorted(warm_ids, key=lambda n: info[warm_ids[n]]["fin"])

# 4 -----------------------------------------------------------------------------------------------------------
m3 = sched["M3"]["warm_mla_reallocated_before_lookup"]
if m3:
    s_evict = m3[0]["seq"]
    s_free = sched["M3"]["warm_fin_seq"]
    toks = sum(v[1] for r in S if s_free < r["seq"] <= s_evict for v in r["reqs"].values())
    fr = [r["free_blocks"] for r in S if s_free <= r["seq"] <= s_evict]
    fr.sort()
    med = fr[len(fr) // 2]
    res["lru_horizon"] = dict(free_seq=s_free, realloc_seq=s_evict, prompt_and_decode_tokens_between=toks,
                              free_queue_median=med, blocks_per_4352_tokens=round(med / (toks / B), 2),
                              note="first re-allocation of a warm-M3 MLA block (idx %d); its later-freed tail blocks "
                                   "and its earlier-freed KDA state blocks precede it in the LRU queue"
                                   % m3[0]["warm_idx"])

# 5 -----------------------------------------------------------------------------------------------------------
prior = {}
for rel in ("b0-mapguard/run-a3/client.jsonl", "phase2-stage2/client.jsonl", "phase3/client.jsonl",
            "d7/step2/run-b1-a1/client.jsonl"):
    p = os.path.join(INV, rel)
    if not os.path.exists(p):
        continue
    seen, c = collections.defaultdict(list), collections.Counter()
    ex = []
    for r in jl(p):
        if r.get("kind") != "request" or r.get("prompt_tokens") is None:
            continue
        key = (r.get("id"), r["prompt_tokens"])
        if seen[key] and r.get("cached_tokens") is not None and r["prompt_tokens"] > B:
            ok = r["cached_tokens"] == eagle_rule(r["prompt_tokens"])
            c["eagle_rule" if ok else ("planned_rule" if r["cached_tokens"] == r["prompt_tokens"] // B * B
                                        else "other")] += 1
            if not ok and len(ex) < 5:
                ex.append((r.get("phase"), r.get("id"), r["prompt_tokens"], r["cached_tokens"]))
        seen[key].append(r.get("phase"))
    prior[rel] = dict(repeat_requests=dict(c), exceptions=ex)
res["prior_sessions_exact_repeats"] = prior

json.dump(res, open(OUT, "w"), indent=1)
print("== quick_struct C1 sub-checks")
for d in qs:
    print("%-7s P %6d warm %6d same %s  err %s ids_equal %s  W %d ct %d  cached %6d exp %6d (%d vs %d blocks) "
          "eagle-rule %6d match %s  FAILS %s" % (d["tag"], d["P"], d["warm_P"], d["same_prompt"], d["error"],
                                                  d["ids_equal"], d["W"], d["completion_tokens"], d["cached"],
                                                  d["cached_expected"], d["blocks_observed"], d["blocks_expected"],
                                                  d["eagle_rule"], d["matches_eagle_rule"], d["fails"]))
print("other phases with cached != expected:", res["other_phases_cached"])
print("== scheduler")
for n, d in sched.items():
    print(n, {k: v for k, v in d.items() if k != "warm_mla_reallocated_before_lookup"},
          "warm MLA re-allocated before lookup:", d["warm_mla_reallocated_before_lookup"])
print("C1 admission order", res["c1_admission_order"], "; warm completion order", res["warm_completion_order"])
print("== LRU horizon", res.get("lru_horizon"))
print("== prior sessions (exact repeats, P > 4352)")
for k, v in prior.items():
    print("  ", k, v)
