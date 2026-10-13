#!/usr/bin/env python3
"""B1 checkpoint pass, seed 1236 (PLAN.md amendment 19). FROZEN ANALYSIS, written before any B1 GPQA data. CPU only.
Mirrors ../bundle/screen2/screen2_analysis.py (sha256 cc3d3000...) and reuses the frozen arm loaders (analysis_col511.py,
sha256 b5b13ce2..., copied unchanged). The new pass is loaded as protocol pass 3 (request seed 1233 + 3 = 1236).
Outputs ids, hashes and counts only: no question, answer or reply text.

usage: b1screen_analysis.py --pass-dir DIR --reqlog FILE [--serverlog FILE] [--inspect FILE] [--gpu-hours X]
                            [--adjudication JSON] [--label TEXT] [--out JSON] [--dry-run-label TEXT]
Exit 0 valid, 3 invalid (no category is given for an invalid or watchdog-truncated pass).

B1 = recipe glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234+d7, image sha256:799d3cf8... (B0 image 08d1cb90... plus
only the D7 file). Checkpoint characterization, not a D7 attribution experiment: B0 has no seed-1236 pass, so no D7
individual effect is inferred from anything below.

PART A -- THE B1 PASS, ITS OWN CHECKPOINT RECORD (interpretation frozen in FROZEN-INTERPRETATION.md)
  Primary reference: the stock 0.9.1 record's seed-1236 pass (+v0.9.1+ps__pass3; second pass of the 2026-10-09 session):
      E = 6 (all cap exhaustion: docs 79, 81, 88, 127, 147, 170), raw 171, stated 174 (published rows).
  Context: the column-511 arm's seed-1236 pass (+col511fix__pass3): E = 6 (cap), raw 171, stated 174; and stock 0.7.0's
      seed-1236 pass (+ps__pass3 of the 0.7.0 record): E = 1 (cap), raw 175, stated 177 (published rows).
  E = empty final replies of 198, split into cap exhaustion (finish 'length' at 327,680 completion tokens) / premature
  stop (empty, finish 'stop') / other.
  ACCURACY STANDARD (unchanged from amendment 15b): the six-pass stock-0.9.1 family (record passes 1-3, column-511 arm
  passes 1-3; raw 164-176, mean 170.2, SD 3.9; stated 165-178, mean 172.8, SD 4.6):
      MATERIAL CONCERN (regression flag): raw <= 158 or stated <= 159 or >= 2 premature stops;
      ACCURACY CONCERN: raw 159-163 (below every family pass).
  Amendment-15b categories, baseline E = the matched stock pass (record pass 3, E = 6):
      SIMILAR OR WORSE                          material concern (any E), or E >= 6
      PROMISING FOLLOW-UP SIGNAL                E <= 2 and no accuracy concern
      MODEST IMPROVEMENT, NO CLEAR SEPARATION   otherwise (E = 3-5, or E <= 2 with an accuracy concern)
  Amendment-19 reading (FROZEN-INTERPRETATION.md), derived from the category, not a new test:
      FAVOURABLE      = PROMISING FOLLOW-UP SIGNAL
      UNFAVOURABLE    = a material concern, or E above every stock-family pass (E >= 7; family E 4-6) -> triggers review
      SIMILAR         = everything else (an accuracy concern is reported with it as a review item)
  Sensitivity readouts only (never category-determining): matched offsets against record pass 3 (flag raw <= 160 or
  stated <= 163; concern raw 161-165), and amendment 14c's absolute empty-count rule (E <= 2 / E = 3 / E >= 4).
  Stated adjudication of every row where the stated adjudicator differs from flexible-extract (--adjudication; hand-read
  rule of arm-col511/adjudications.json: hedged only if the reply commits to no single option) is required before the
  final category; otherwise PROVISIONAL.

PART B -- EXPLORATORY CORRECTED-FAMILY VIEW (two implementations; NOT a three-pass B1 result)
  B0 seed 1234 + B0 seed 1235 + B1 seed 1236 against the stock record's passes 1-3 (seeds 1234-1236), and against the
  column-511 arm's passes 1-3. Same statistics as screen2's part B (exact sign-flip over questions, 20,000-replicate
  question bootstrap). Labelled exploratory: it mixes two implementations (B0, B1), uses historical controls, and the
  B1 pass was chosen after inspecting B0's passes. Records stay separate: B0 seeds 1234/1235, B1 seed 1236.
"""
import argparse
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis_col511 as A  # noqa: E402

G = "<gpqa-runs>"
INV = "<runs>"
B1_R = "glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234+d7"
B1_IMAGE = "sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a"
B0_R = "glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234"
B0 = [dict(seed_pass=1, label="B0 seed 1234 (bundle screen 1)", dir=f"{G}/{B0_R}__pass1", reqlog=f"{G}/{B0_R}.s1.requests.jsonl",
           adjudication=f"{INV}/bundle/screen/run-d1234/adjudication.json"),
      dict(seed_pass=2, label="B0 seed 1235 (bundle screen 2)", dir=f"{G}/{B0_R}__pass2", reqlog=f"{G}/{B0_R}.s2.requests.jsonl",
           adjudication=f"{INV}/bundle/screen2/run-s1235/adjudication.json")]
B0_INSPECT = f"{INV}/bundle/screen2/run-s1235/inspect-s2.json"     # B0's seed-1235 session: the effective-settings reference
REC = dict(passes=[f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps__pass{p}" for p in (1, 2, 3)],
           reqlog=[f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1.requests.jsonl", f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps.requests.jsonl"],
           published="<this-repo>/runs/2026-10-08_glm53-flash_k3.25-v0.9.1-dflash3_gpqa-diamond/results.jsonl")
ARM = dict(passes=[f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+col511fix__pass{p}" for p in (1, 2, 3)],
           reqlog=[f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+col511fix.s1.requests.jsonl",
                   f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+col511fix.s2.requests.jsonl"],
           adjudication=f"{INV}/arm-col511/adjudications.json")
# 0.7.0: unlike screen2 (which loaded it without hand reads, giving stated 178 instead of 177 on pass 2), the published
# record's hand read is applied (pass 2 doc 25 hedged, 2026-10-10) and the derived rows must reproduce the published rows.
V070 = dict(passes=[f"{G}/glm53-flash-exl3-k3.25@tpurtell__pass1", f"{G}/glm53-flash-exl3-k3.25@tpurtell+ps__pass2",
                    f"{G}/glm53-flash-exl3-k3.25@tpurtell+ps__pass3"],
            reqlog=[f"{G}/glm53-flash-exl3-k3.25@tpurtell+ps.requests.jsonl"],
            published="<this-repo>/runs/2026-09-29_glm53-flash_k3.25-v0.7.0-dflash5_gpqa-diamond/results.jsonl",
            hedged=("2:25",))
SEED_PASS = 3                                       # request seed 1233 + 3 = 1236; lm-eval --seed 1234 + 3 = 1237
FAMILY_RAW = (169, 176, 171, 164, 170, 171)        # record p1-3, arm p1-3 (raw flexible-extract, all 198) -- amendment 15b
FAMILY_STATED = (170, 178, 174, 165, 176, 174)     # record p1-3, arm p1-3 (stated, adjudicated) -- amendment 15b
FAMILY_E = (5, 5, 6, 4, 5, 6)                      # record p1-3, arm p1-3 (empty replies)
RAW_FLAG, STATED_FLAG, RAW_CONCERN, PREMATURE_MAX = 158, 159, 163, 1
E_PROMISING, E_BASELINE, E_FAMILY_MAX = 2, 6, 6
OFFSET_FLAG, OFFSET_CONCERN = 11, 6                 # sensitivity readout only (as screen2)
BOOT_SEED = 20261013
# effective settings of every bundle/stock-0.9.1 GPQA session so far (B0 s1, B0 s2, arm s2, record s2, B1 mapguard a1)
SERVER_ARGS_SHA = "7025a96888a67642741db00aaecb9cbf786c2ba67cc4ded7d67c1cf4e909dcc5"
CUDAGRAPH = ["FULL_AND_PIECEWISE"]
KV_POOL = [4707515]
RELEASE_WARMUPS = 9
LMEVAL = dict(seed="1236", random_seed="1237", num_concurrent="8", timeout="10800", max_retries="3", max_gen_toks="327680")
INPUT_PINS = {   # informative (not gating): sha256 of the comparator inputs at freeze
    f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps__pass3": "bcaa1e011ce13b202205e12f37f399f57bdf39ad4b83ee8bdd885c8715d728b7",
    f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+col511fix__pass3": "1d00ade555e2567fb4b2b272a6e149c669458803d7f2fe81478fd2842984e351",
    f"{G}/glm53-flash-exl3-k3.25@tpurtell+ps__pass3": "aae579ab4587cd68003576d86a7ae1fa1096bc7681b3ad87bf65627222e9ef4e",
    f"{G}/{B0_R}__pass1": "8c413a97e412ea156ffc7093b3e85a5d04ff06e3fbba66b6f3077f18114caad8",
    f"{G}/{B0_R}__pass2": "d3eb4fa745c70ee8a91133e3d12e238a83e8aa9487f7989e3b42d6f157fa9c92",
    REC["published"]: "9d2f57c9c4c1c5fe0833f5d534a290bfc61793101ca0f5cf1cf6bb04cbc970b2",
    V070["published"]: "da381c6b175392d59d37f7922772bb9732dc1c17d3c1d72b22ad314d75dae837",
    ARM["adjudication"]: "27561557097519c98a074e8235632cb0c432e41cc6c33f8d47bcffc2987a043c",
    B0_INSPECT: "3e7391070a750cf990eb8712145cebe09fe42f6be437e463ec084cf6b6857f08",
}


def cause(r):
    if not r["empty"]:
        return None
    if not r["logged"]:
        return "not_logged"
    if r["finish_reason"] == "length" and r["completion_tokens"] == A.CAP:
        return "cap_exhaustion"
    if r["finish_reason"] == "stop":
        return "premature_stop"
    return "other_finish"


def categorize(E, raw, stated, premature):
    """Amendment 15b categories on the family accuracy standard, baseline E of the matched stock pass."""
    material = raw <= RAW_FLAG or stated <= STATED_FLAG or premature > PREMATURE_MAX
    concern = (not material) and RAW_FLAG < raw <= RAW_CONCERN
    if material or E >= E_BASELINE:
        return material, concern, "SIMILAR OR WORSE"
    if E <= E_PROMISING and not concern:
        return material, concern, "PROMISING FOLLOW-UP SIGNAL"
    return material, concern, "MODEST IMPROVEMENT, NO CLEAR SEPARATION"


def reading_a19(category, E, material, concern):
    """Amendment-19 reading of the 15b category (FROZEN-INTERPRETATION.md)."""
    if category == "PROMISING FOLLOW-UP SIGNAL":
        r = "FAVOURABLE"
    elif material or E > E_FAMILY_MAX:
        r = "UNFAVOURABLE"
    else:
        r = "SIMILAR"
    why = []
    if material:
        why.append("material concern")
    if E > E_FAMILY_MAX:
        why.append(f"E = {E} above every stock-family pass (max {E_FAMILY_MAX})")
    if concern:
        why.append("accuracy concern (raw 159-163): review item")
    return r, why


def reading_14c(E, raw, stated, premature):
    """Amendment 14c's absolute rule (seed-1234 screen), SENSITIVITY READOUT ONLY."""
    flag = raw <= RAW_FLAG or stated <= STATED_FLAG or premature > PREMATURE_MAX
    concern = RAW_FLAG < raw <= RAW_CONCERN
    if E >= 4:
        return "SIMILAR OR WORSE"
    if flag or concern:
        return "MIXED"
    if E <= 2:
        return "CLEAR PROMISING"
    return "SIMILAR / NO CLEAR SEPARATION"


def load_single(label, pass_dir, reqlogs, hedged_docs, seed_pass):
    """Load ONE pass directory as protocol pass `seed_pass` (request seed 1233 + seed_pass); rows keyed by doc."""
    saved = (A.N_PASSES, A.SEED_BASE)
    A.N_PASSES, A.SEED_BASE = 1, 1233 + seed_pass - 1
    try:
        rows, checks, _ = A.load_arm(label, [pass_dir], reqlogs, [f"1:{d}" for d in hedged_docs], None)
    finally:
        A.N_PASSES, A.SEED_BASE = saved
    out = {}
    for (d, _p), r in rows.items():
        out[d] = dict(r, **{"pass": seed_pass})
    return out, checks


def load_three(label, spec, hedged=()):
    rows, checks, _ = A.load_arm(label, spec["passes"], spec["reqlog"], list(hedged), spec.get("published"))
    by = {}
    for (d, p), r in rows.items():
        by.setdefault(p, {})[d] = r
    return by, checks


def compare(S, B):
    se = {d for d, r in S.items() if r["empty"]}
    be = {d for d, r in B.items() if r["empty"]}
    raw_s = sum(r["correct_flexible"] for r in S.values())
    raw_b = sum(r["correct_flexible"] for r in B.values())
    st_s = sum(r["correct_stated"] for r in S.values())
    st_b = sum(r["correct_stated"] for r in B.values())
    return dict(baseline_empties=len(be),
                baseline_causes={c: sum(1 for d in be if cause(B[d]) == c) for c in ("cap_exhaustion", "premature_stop", "not_logged", "other_finish")},
                rescued=sorted(be - se), newly_empty=sorted(se - be), empty_in_both=sorted(se & be),
                baseline_raw=raw_b, baseline_stated=st_b, raw_diff=raw_s - raw_b, stated_diff=st_s - st_b,
                raw_only_screen_correct=sum(1 for d in S if S[d]["correct_flexible"] and not B[d]["correct_flexible"]),
                raw_only_baseline_correct=sum(1 for d in S if B[d]["correct_flexible"] and not S[d]["correct_flexible"]))


def tokens(R):
    """Generated (completion) tokens over ALL requests, empties and cap exhaustions included."""
    allt = [r["completion_tokens"] for r in R.values() if r["logged"] and r["completion_tokens"] is not None]
    ans = [r["completion_tokens"] for r in R.values() if r["logged"] and not r["empty"] and r["completion_tokens"] is not None]
    return {"logged_requests": len(allt), "total_all_requests": sum(allt), "mean_all_requests": round(sum(allt) / len(allt), 1) if allt else None,
            "all_requests": A.quantiles(allt) if allt else None, "answered": A.quantiles(ans) if ans else None,
            "answered_above_32k_64k_128k_200k": [sum(t > x for t in ans) for x in A.LEN_THRESHOLDS]}


def window(R):
    if not R or not all(r["logged"] and r["t_start"] is not None for r in R.values()):
        return None
    t0 = min(r["t_start"] for r in R.values())
    t1 = max(r["t_start"] + r["secs"] for r in R.values())
    tok = sum(r["completion_tokens"] for r in R.values())
    rates = [r["completion_tokens"] / r["secs"] for r in R.values() if r["secs"] and not r["empty"]]
    return {"window_h": round((t1 - t0) / 3600, 4), "completion_tokens": tok, "aggregate_tok_s": round(tok / (t1 - t0), 1),
            "median_request_tok_s_answered": round(statistics.median(rates), 1) if rates else None,
            "request_window_gpu_h": round(2 * (t1 - t0) / 3600, 3)}


def combined(bundle_passes, cmp_passes, label):
    """EXPLORATORY: per-question sums over the paired seeds; questions are the unit (as screen2 part B)."""
    docs = sorted(bundle_passes[0])
    d_e = [sum(int(c[q]["empty"]) for c in cmp_passes) - sum(int(b[q]["empty"]) for b in bundle_passes) for q in docs]
    d_raw = [sum(int(b[q]["correct_flexible"]) for b in bundle_passes) - sum(int(c[q]["correct_flexible"]) for c in cmp_passes) for q in docs]
    d_st = [sum(int(b[q]["correct_stated"]) for b in bundle_passes) - sum(int(c[q]["correct_stated"]) for c in cmp_passes) for q in docs]
    n = len(bundle_passes) * len(docs)
    sf = A.signflip(d_e)
    lo, hi = A.boot_interval(d_e, n, BOOT_SEED)
    rlo, rhi = A.boot_interval(d_raw, n, BOOT_SEED + 1)
    slo, shi = A.boot_interval(d_st, n, BOOT_SEED + 2)
    return dict(label=f"EXPLORATORY ({label}): two implementations (B0 seeds 1234-1235, B1 seed 1236), NOT a three-pass B1 "
                      f"result; historical controls; B1 pass chosen after inspecting B0's passes",
                question_passes_per_side=n,
                corrected_family_empties=sum(int(b[q]["empty"]) for b in bundle_passes for q in docs),
                comparator_empties=sum(int(c[q]["empty"]) for c in cmp_passes for q in docs),
                empty_signflip_cmp_minus_family=sf,
                empty_rate_reduction_points=round(100 * sum(d_e) / n, 2),
                empty_rate_reduction_ci95_points=[round(100 * lo, 2), round(100 * hi, 2)],
                questions_d_pos=sum(1 for x in d_e if x > 0), questions_d_neg=sum(1 for x in d_e if x < 0),
                raw_diff_points=round(100 * sum(d_raw) / n, 2), raw_diff_ci95_points=[round(100 * rlo, 2), round(100 * rhi, 2)],
                raw_signflip_two_sided_reference=A.signflip(d_raw)["p_two_sided"],
                stated_diff_points=round(100 * sum(d_st) / n, 2), stated_diff_ci95_points=[round(100 * slo, 2), round(100 * shi, 2)])


def lmeval_config(pass_dir):
    res = glob.glob(os.path.join(pass_dir, "**", "results_*.json"), recursive=True)
    if len(res) != 1:
        return None
    j = json.load(open(res[0]))
    cfg = j.get("config", {})
    ma = cfg.get("model_args") or {}
    if isinstance(ma, str):
        ma = dict(kv.split("=", 1) for kv in ma.split(",") if "=" in kv)
    gk = cfg.get("gen_kwargs") or {}
    if isinstance(gk, str):
        gk = dict(kv.split("=", 1) for kv in gk.split(",") if "=" in kv)
    return {"seed": str(ma.get("seed")), "random_seed": str(cfg.get("random_seed")), "num_concurrent": str(ma.get("num_concurrent")),
            "timeout": str(ma.get("timeout")), "max_retries": str(ma.get("max_retries")), "max_gen_toks": str(gk.get("max_gen_toks")),
            "task_hash": (j.get("task_hashes") or {}).get("gpqa_diamond_thinking")}


def inspect_view(path):
    try:
        d = json.load(open(path))[0]
    except (OSError, ValueError, IndexError, KeyError):
        return None                                  # e.g. "[]" when the container no longer existed
    hc = d.get("HostConfig", {})
    return {"image": d.get("Image"), "env": sorted(set(d["Config"].get("Env") or [])), "cmd": d["Config"].get("Cmd"),
            "entrypoint": d["Config"].get("Entrypoint"), "path": d.get("Path"), "args": d.get("Args"),
            **{k: hc.get(k) for k in ("Binds", "PortBindings", "ShmSize", "IpcMode", "DeviceRequests", "NetworkMode", "Runtime")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass-dir", required=True)
    ap.add_argument("--reqlog", required=True)
    ap.add_argument("--serverlog", default=None)
    ap.add_argument("--inspect", default=None, help="docker inspect of the B1 server (required unless --dry-run-label)")
    ap.add_argument("--gpu-hours", type=float, default=None)
    ap.add_argument("--adjudication", default=None,
                    help="JSON {rows: [{doc, hedged}]}: hand reads of every row where stated differs from flexible-extract")
    ap.add_argument("--label", default="B1 checkpoint pass (seed 1236)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run-label", default=None)
    a = ap.parse_args()
    adj_rows = json.load(open(a.adjudication))["rows"] if a.adjudication else []
    have_log = os.path.isfile(a.reqlog)            # an aborted session may leave no request log: invalid, not a crash
    S, checks = load_single(a.label, a.pass_dir, [a.reqlog] if have_log else [], [int(r["doc"]) for r in adj_rows if r.get("hedged")], SEED_PASS)
    checks.append(("request log of the new pass present", have_log, a.reqlog))
    rec, c1 = load_three("0.9.1 record", REC)
    arm_adj = json.load(open(ARM["adjudication"]))["rows"]
    arm, c2 = load_three("col511 arm", ARM, [f"{r['pass']}:{r['doc']}" for r in arm_adj if r["hedged"]])
    v070, c3 = load_three("0.7.0 record (context)", V070, V070["hedged"])
    checks += [c for c in c1 + c2 + c3 if "pass 3" in c[0] or "published" in c[0]]
    # frozen constants reproduce from the data
    fam = [rec[1], rec[2], rec[3], arm[1], arm[2], arm[3]]
    checks.append(("frozen family constants reproduce (E, raw, stated of record p1-3 and arm p1-3; baseline E of record p3)",
                   tuple(sum(r["empty"] for r in R.values()) for R in fam) == FAMILY_E
                   and tuple(sum(r["correct_flexible"] for r in R.values()) for R in fam) == FAMILY_RAW
                   and tuple(sum(r["correct_stated"] for r in R.values()) for R in fam) == FAMILY_STATED
                   and sum(r["empty"] for r in rec[3].values()) == E_BASELINE and max(FAMILY_E) == E_FAMILY_MAX, ""))
    # B0 passes (part B only; separate checkpoint records)
    B0P = []
    for b in B0:
        hed = [int(r["doc"]) for r in json.load(open(b["adjudication"]))["rows"] if r.get("hedged")]
        R, cb = load_single(b["label"], b["dir"], [b["reqlog"]], hed, b["seed_pass"])
        checks += cb
        B0P.append(R)
    # the new pass: prompts, lm-eval configuration, stopping
    for name, B in (("record pass 3", rec[3]), ("arm pass 3", arm[3]), ("0.7.0 pass 3", v070[3])):
        mism = [d for d in S if d not in B or S[d]["messages_sha256"] != B[d]["messages_sha256"]]
        checks.append((f"prompts identical to {name} (messages_sha256, 198 questions)", not mism and len(S) == 198, f"{len(mism)} differ"))
    cfg, cfg_rec = lmeval_config(a.pass_dir), lmeval_config(REC["passes"][2])
    ok = cfg is not None and cfg_rec is not None and all(cfg[k] == v for k, v in LMEVAL.items()) and cfg["task_hash"] == cfg_rec["task_hash"]
    checks.append(("lm-eval: request seed 1236, --seed 1237, concurrency 8, timeout 10800, 3 retries, cap 327680; task hash = record pass 3",
                   ok, json.dumps({"new": cfg, "record_pass3_task_hash": cfg_rec and cfg_rec["task_hash"]})))
    odd = [d for d, r in S.items() if r["logged"] and r["finish_reason"] not in ("stop", "length")]
    checks.append(("no execution/stopping errors (finish reason stop or length on every request)", not odd, f"{odd[:5]}"))
    srv = None
    if a.serverlog and S and all(r["t_start"] for r in S.values()):
        t0 = min(r["t_start"] for r in S.values() if r["t_start"])
        t1 = max(r["t_start"] + r["secs"] for r in S.values() if r["t_start"])
        srv = A.parse_serverlogs([a.serverlog], [(t0, t1)], 0)[0]
        iw = srv.get("in_window", {})
        checks.append(("server: one start, no illegal memory access / preemption / NaN lines in the request window",
                       srv.get("server_starts") == 1 and not any(iw.values()), json.dumps(iw)))
        checks.append(("server: shipped configuration (non-default args as every 0.9.1-family session, cudagraph FULL_AND_PIECEWISE, "
                       "KV pool 4,707,515, 9 release warm-ups logged)",
                       srv.get("non_default_args_sha256") == SERVER_ARGS_SHA and srv.get("cudagraph_mode") == CUDAGRAPH
                       and srv.get("kv_pool_tokens") == KV_POOL and srv.get("release_warmups_logged") == RELEASE_WARMUPS,
                       json.dumps({k: srv.get(k) for k in ("non_default_args_sha256", "cudagraph_mode", "kv_pool_tokens", "release_warmups_logged")})))
    elif not a.dry_run_label or a.serverlog:
        checks.append(("server log supplied and every request logged with a start time", False, "missing"))
    if a.inspect:
        new, ref = inspect_view(a.inspect), inspect_view(B0_INSPECT)
        checks.append(("container image is B1", bool(new) and new["image"] == B1_IMAGE, new and new["image"]))
        diff = [k for k in ref if k != "image" and new[k] != ref[k]] if new else ["no inspect record"]
        checks.append(("container settings equal to B0's seed-1235 session apart from the image (env as a set, cmd, args, binds, ports, "
                       "shm, ipc, devices, network, runtime)", not diff, f"differ: {diff}"))
    elif not a.dry_run_label:
        checks.append(("container inspect record supplied", False, "missing"))
    valid = all(ok for _, ok, _ in checks)
    pins = {p: {"frozen": h, "now": (A.file_sha(p) if os.path.isfile(p) else
                                     (lambda f: A.file_sha(f[0]) if len(f) == 1 else None)(glob.glob(os.path.join(p, "**", "samples_*.jsonl"), recursive=True)))}
            for p, h in INPUT_PINS.items()}
    res = {"label": a.label, "dry_run": a.dry_run_label, "valid": valid, "checks": [list(c) for c in checks],
           "script_sha256": A.file_sha(os.path.abspath(__file__)), "analysis_col511_sha256": A.file_sha(A.__file__),
           "input_pins_informative": {"all_unchanged": all(v["frozen"] == v["now"] for v in pins.values()),
                                      "changed": [p for p, v in pins.items() if v["frozen"] != v["now"]]},
           "accuracy_standard": dict(family_raw=FAMILY_RAW, family_stated=FAMILY_STATED, family_empties=FAMILY_E,
                                     raw_mean=round(statistics.mean(FAMILY_RAW), 2), raw_sd=round(statistics.stdev(FAMILY_RAW), 2),
                                     stated_mean=round(statistics.mean(FAMILY_STATED), 2), stated_sd=round(statistics.stdev(FAMILY_STATED), 2),
                                     material=f"raw <= {RAW_FLAG} or stated <= {STATED_FLAG} or premature stops >= {PREMATURE_MAX + 1}",
                                     concern=f"raw {RAW_FLAG + 1}-{RAW_CONCERN}", baseline_E=E_BASELINE, family_E_max=E_FAMILY_MAX)}
    if valid:
        E = sum(r["empty"] for r in S.values())
        causes = {}
        for r in S.values():
            c = cause(r)
            if c:
                causes[c] = causes.get(c, 0) + 1
        raw = sum(r["correct_flexible"] for r in S.values())
        stated = sum(r["correct_stated"] for r in S.values())
        prem = causes.get("premature_stop", 0)
        material, concern, outcome = categorize(E, raw, stated, prem)
        a19, a19_why = reading_a19(outcome, E, material, concern)
        read = {int(r["doc"]) for r in adj_rows}
        missing = [d for d in sorted(d for d, r in S.items() if r["stated_differs_from_flexible"]) if d not in read]
        final = outcome if not missing else f"PROVISIONAL ({outcome}; stated adjudication pending for docs {missing})"
        final19 = a19 if not missing else f"PROVISIONAL ({a19})"
        rb, sb = sum(r["correct_flexible"] for r in rec[3].values()), sum(r["correct_stated"] for r in rec[3].values())
        sens_flag = raw <= rb - OFFSET_FLAG or stated <= sb - OFFSET_FLAG or prem > PREMATURE_MAX
        sens_concern = (not sens_flag) and raw <= rb - OFFSET_CONCERN
        A.N_PASSES = 1
        try:
            tim, _ = (A.timing({(d, 1): dict(r, **{"pass": 1}) for d, r in S.items()}, "pass3", None, a.gpu_hours, 2)
                      if all(r["logged"] for r in S.values()) else ({}, None))
        finally:
            A.N_PASSES = 3
        both = [d for d in sorted(S) if S[d]["logged"] and rec[3][d]["logged"]]
        dtok = [S[d]["completion_tokens"] - rec[3][d]["completion_tokens"] for d in both]
        tlo, thi = A.boot_interval(dtok, len(both), BOOT_SEED + 3) if both else (None, None)
        res["part_A_b1_pass"] = {
            "empties": E, "causes": causes, "category_15b": final, "category_15b_if_adjudication_complete": outcome,
            "reading_amendment19": final19, "reading_amendment19_if_adjudication_complete": a19, "reading_reasons": a19_why,
            "adjudication_complete": not missing, "adjudicated_docs": sorted(read),
            "flags_family_standard": {"material_concern": material, "accuracy_concern": concern,
                                      "detail": {"raw": raw, "stated": stated, "premature_stops": prem}},
            "sensitivity_matched_offsets_vs_record_pass3": {"baseline_raw": rb, "baseline_stated": sb, "flag": sens_flag, "concern": sens_concern,
                                                            "rule": f"flag raw <= {rb - OFFSET_FLAG} or stated <= {sb - OFFSET_FLAG} "
                                                                    f"or >= 2 premature; concern raw {rb - OFFSET_FLAG + 1}-{rb - OFFSET_CONCERN}"},
            "sensitivity_14c_absolute_rule": reading_14c(E, raw, stated, prem),
            "versus": {"stock 0.9.1 record pass 3 (primary reference, seed 1236)": compare(S, rec[3]),
                       "col511 arm pass 3 (context, seed 1236)": compare(S, arm[3]),
                       "stock 0.7.0 pass 3 (context, seed 1236)": compare(S, v070[3])},
            "secondary": {"raw_correct_all_198": raw, "stated_correct_all_198": stated,
                          "stated_differs_from_raw_rows": [d for d, r in S.items() if r["stated_differs_from_flexible"]],
                          "hedged_by_hand": sorted(d for d, r in S.items() if r["hedged_by_hand"]),
                          "generated_tokens": {"b1": tokens(S), "record_pass3": tokens(rec[3]), "arm_pass3": tokens(arm[3]),
                                               "v070_pass3": tokens(v070[3]),
                                               "paired_mean_per_request_b1_minus_record_pass3": {
                                                   "questions": len(both), "estimate": round(sum(dtok) / len(both), 1) if both else None,
                                                   "ci95": [round(tlo, 1), round(thi, 1)] if both else None,
                                                   "note": "all requests incl. cap exhaustions; question bootstrap; descriptive"}},
                          "timing": tim,
                          "comparator_windows_context": {"record_pass3": window(rec[3]), "arm_pass3": window(arm[3]), "v070_pass3": window(v070[3]),
                                                         "note": "comparators' pass 3 was the second pass of its server session; "
                                                                 "request windows only (no per-pass occupied GPU-hours)"},
                          "server": srv}}
        res["part_B_exploratory_corrected_family"] = {
            "vs_record_passes_1_3": combined([B0P[0], B0P[1], S], [rec[1], rec[2], rec[3]],
                                             "corrected family seeds 1234-1236 vs stock 0.9.1 record seeds 1234-1236"),
            "vs_arm_passes_1_3": combined([B0P[0], B0P[1], S], [arm[1], arm[2], arm[3]],
                                          "corrected family seeds 1234-1236 vs column-511 arm seeds 1234-1236"),
            "per_checkpoint_records": {"B0 seed 1234": {"E": sum(r["empty"] for r in B0P[0].values()),
                                                        "raw": sum(r["correct_flexible"] for r in B0P[0].values()),
                                                        "stated": sum(r["correct_stated"] for r in B0P[0].values())},
                                       "B0 seed 1235": {"E": sum(r["empty"] for r in B0P[1].values()),
                                                        "raw": sum(r["correct_flexible"] for r in B0P[1].values()),
                                                        "stated": sum(r["correct_stated"] for r in B0P[1].values())},
                                       "B1 seed 1236": {"E": E, "raw": raw, "stated": stated}}}
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=str)
    print(json.dumps({k: res[k] for k in res if k != "checks"}, indent=1, default=str)[:12000])
    bad = [c for c in res["checks"] if not c[1]]
    if bad:
        print("INVALID:", json.dumps(bad, default=str)[:1500])
    return 0 if valid else 3


if __name__ == "__main__":
    sys.exit(main())
