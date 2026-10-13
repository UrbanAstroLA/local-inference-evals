#!/usr/bin/env python3
"""B0 checkpoint pass, seed 1236 (review step after the UNFAVOURABLE B1 seed-1236 reading). FROZEN ANALYSIS, written
before any B0 seed-1236 GPQA data. CPU only.
Copy of ../b1-screen/b1screen_analysis.py (sha256 79268c47...), extended; every existing measure, rule and function
(cause, categorize, reading_a19, reading_14c, load_single, load_three, compare, tokens, window, lmeval_config,
inspect_view) is unchanged; combined() gains an optional description argument only. Reuses the frozen arm loaders
(analysis_col511.py, sha256 b5b13ce2..., copied unchanged). The new pass is loaded as protocol pass 3 (request seed
1233 + 3 = 1236). Outputs ids, hashes and counts only: no question, answer or reply text.

usage: b0s1236_analysis.py --pass-dir DIR --reqlog FILE [--serverlog FILE] [--inspect FILE] [--gpu-hours X]
                           [--adjudication JSON] [--label TEXT] [--out JSON] [--dry-run-label TEXT]
Exit 0 valid, 3 invalid (no category or reading is given for an invalid or watchdog-truncated pass).

B0 = recipe glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234, image sha256:08d1cb90... (D1-D4 bundle).
B1 = B0 + the D7 file only (image sha256:799d3cf8...). The D7 reconstruction (d7/exposure/REPORT.md) finds no D7 exposure
on GPQA prompts in the 0.9.1 family, and where D7 does not fire B1's writes are byte-identical to B0's (CPU-verified). So
on GPQA B0 and B1 should compute the same thing; B0-vs-B1 differences at seed 1236 then measure run-to-run variation at a
fixed seed (timing-dependent batch composition), unless the reconstruction is wrong. ONE PAIR OF PASSES CANNOT SEPARATE
D7 FROM RUN-TO-RUN VARIATION.

PART A -- THE B0 PASS AGAINST STOCK 0.9.1 AT SEED 1236 (continuity; rules of ../b1-screen/FROZEN-INTERPRETATION.md)
  Primary reference for continuity: stock 0.9.1 record pass 3 (E = 6, raw 171, stated 174); context: column-511 arm pass 3
  (E = 6, 171, 174) and stock 0.7.0 pass 3 (E = 1, 175, 177). Amendment-15b category on the six-pass family standard,
  baseline E = 6, unchanged. The amendment-19 reading and the 14c readout are reported for continuity only: they were
  defined for B1 and do not decide anything here.

PART C -- B0 VERSUS B1 AT SEED 1236 (the primary comparison of this review; FROZEN-INTERPRETATION.md)
  Both passes: first and only pass of a fresh server, request seed 1236, lm-eval --seed 1237, same protocol.
  B1 (final, adjudicated): E = 9, all cap exhaustion; raw 173; stated 174 (reproduced from its files as a validity check).
  Measures: empty replies by cause for both; rescued (empty in B1, not B0) / newly empty (empty in B0, not B1) / empty in
  both; per-question agreement of empty status (2x2); raw and stated (same hand-read adjudication rule); generated tokens
  (totals, quantiles, paired B0 - B1 per question with a question bootstrap, descriptive).
  Reply identity: questions whose final reply has identical content_sha256 AND identical completion_tokens, split into
  (i) non-empty identical replies (I_ne, the identity count that is READ) and (ii) both empty (identical by construction:
  the empty string's hash; at the cap also the same token count). As a stricter sub-count, reasoning length (reasoning
  characters and reasoning tokens from the request logs) must also agree. The same counts are given, as context, for the
  earlier pairs (B1 vs stock record pass 3, B1 vs arm pass 3, B0 seeds 1234/1235 vs record and arm, record vs arm).
  FIRST DIVERGENCE CANNOT BE MEASURED: no token-level logs exist (only final content hash, token counts and reasoning
  length per request), so where two replies differ it is unknown at which token they diverged.
  Frozen reading (empty count E of the B0 pass, B1's E = 9; stock family range 4-6):
      (a) E >= 8  SEED-1236 EXCESS SHARED BY B0 -> property of the corrected family at this seed, not attributable to D7
      (b) E <= 6  B0 WITHIN THE STOCK RANGE -> the B0/B1 difference is D7 (contradicting the zero-exposure reconstruction)
                  or run-to-run variation; distinguished only weakly by I_ne (lean, never a conclusion)
      (c) E = 7   NO READING
  Identity reading: I_ne >= 20 NEAR-DETERMINISTIC AT A FIXED SEED; I_ne <= 2 BATCHING NONDETERMINISM DOMINATES; else PARTIAL.

PART B -- EXPLORATORY VIEWS (questions as the unit; exact sign-flip; 20,000-replicate question bootstrap)
  B-1 (new): B0 three-seed record (seeds 1234, 1235, 1236) vs stock record passes 1-3 and vs column-511 arm passes 1-3, one
      implementation; EXPLORATORY: historical controls, seed reuse across engines, and the seed-1236 B0 pass was chosen
      after inspecting B1's pass. B1 stays separate.
  B-2 (continuity, unchanged): the two-implementation view of the B1 package (B0 1234 + B0 1235 + B1 1236); reproduced
      from the files as a validity check of the loaders.
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
B0_IMAGE = "sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548"
B0 = [dict(seed_pass=1, label="B0 seed 1234 (bundle screen 1)", dir=f"{G}/{B0_R}__pass1", reqlog=f"{G}/{B0_R}.s1.requests.jsonl",
           adjudication=f"{INV}/bundle/screen/run-d1234/adjudication.json"),
      dict(seed_pass=2, label="B0 seed 1235 (bundle screen 2)", dir=f"{G}/{B0_R}__pass2", reqlog=f"{G}/{B0_R}.s2.requests.jsonl",
           adjudication=f"{INV}/bundle/screen2/run-s1235/adjudication.json")]
B0_INSPECT = f"{INV}/bundle/screen2/run-s1235/inspect-s2.json"     # B0's seed-1235 session: the effective-settings reference
B1 = dict(label="B1 seed 1236 (b1-screen, final)", dir=f"{G}/{B1_R}__pass3", reqlog=f"{G}/{B1_R}.s1.requests.jsonl",
          adjudication=f"{INV}/b1-screen/run-s1236/adjudication.json", inspect=f"{INV}/b1-screen/run-s1236/inspect-s1.json",
          final_json=f"{INV}/b1-screen/run-s1236/b1screen-result-final.json")
B1_FINAL = dict(E=9, causes={"cap_exhaustion": 9}, raw=173, stated=174)   # b1screen-result-final.json (amendment-19 pass)
REC = dict(passes=[f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps__pass{p}" for p in (1, 2, 3)],
           reqlog=[f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1.requests.jsonl", f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps.requests.jsonl"],
           published="<this-repo>/runs/2026-10-08_glm53-flash_k3.25-v0.9.1-dflash3_gpqa-diamond/results.jsonl")
ARM = dict(passes=[f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+col511fix__pass{p}" for p in (1, 2, 3)],
           reqlog=[f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+col511fix.s1.requests.jsonl",
                   f"{G}/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+col511fix.s2.requests.jsonl"],
           adjudication=f"{INV}/arm-col511/adjudications.json")
# 0.7.0: the published record's hand read is applied (pass 2 doc 25 hedged, 2026-10-10) and the derived rows must
# reproduce the published rows.
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
# B0-vs-B1 reading (FROZEN-INTERPRETATION.md)
E_SHARED = 8                                        # (a) B0 E >= 8: at least 2 above every stock pass, within 1 of B1's 9
E_STOCK_MAX = E_FAMILY_MAX                          # (b) B0 E <= 6: within the stock family range (4-6)
IDENT_NEAR_DET = 20                                 # I_ne >= 20 (about 10% of 198): near-deterministic at a fixed seed
IDENT_NONDET_MAX = 2                                # I_ne <= 2: batching nondeterminism dominates
# effective settings of every bundle/stock-0.9.1 GPQA session so far (B0 s1, B0 s2, arm s2, record s2, B1 mapguard a1, B1 s1)
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
    f"{G}/{B1_R}__pass3": "478147cb7b87bdbb277e0ea565bb4cbc087af8c66d121cc9476349e01def695a",
    f"{G}/{B1_R}.s1.requests.jsonl": "7067834668c05b10e1781dc08a06657876d6a8bc46463782962ada577e970be3",
    B1["adjudication"]: "150f84f3c70ad58ff760d9aa03b9311e88c58179cf3bc602fb6c2558bd103668",
    B1["inspect"]: "4cd32f437cce0a40abee039b2054a5bc5721dcd7bc781df7c06723e7de10f9be",
    B1["final_json"]: "72d6c6edbb834b42f0a710d001150242f4fa779867f925f6702c3cbcc158b5c5",
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


def identity_reading(i_ne):
    """How the non-empty identity count I_ne between two same-seed passes is read (FROZEN-INTERPRETATION.md)."""
    if i_ne >= IDENT_NEAR_DET:
        return "NEAR-DETERMINISTIC AT A FIXED SEED"
    if i_ne <= IDENT_NONDET_MAX:
        return "BATCHING NONDETERMINISM DOMINATES"
    return "PARTIAL"


def reading_b0b1(E, i_ne):
    """Frozen B0-vs-B1 reading at seed 1236 (B1 E = 9). Returns (code, statement, lean or None)."""
    if E >= E_SHARED:
        return ("(a)", "SEED-1236 EXCESS SHARED BY B0: the excess is a property of the corrected family (B0 and B1) at this seed, "
                       "not attributable to D7 (B0, without D7, shows it too)", None)
    if E <= E_STOCK_MAX:
        ir = identity_reading(i_ne)
        if ir == "NEAR-DETERMINISTIC AT A FIXED SEED":
            lean = ("weak lean: the passes are near-deterministic, so the B0/B1 difference is less readily run-to-run variation; "
                    "predefined follow-up is a CPU re-check of the zero-exposure reconstruction; no D7 effect is claimed")
        elif ir == "BATCHING NONDETERMINISM DOMINATES":
            lean = ("weak lean: same-seed passes behave as distinct draws, so a difference of this size is expected from "
                    "run-to-run variation alone; no D7 effect is inferred")
        else:
            lean = "no lean (identity count in the partial band)"
        return ("(b)", "B0 WITHIN THE STOCK RANGE: the B0/B1 difference is either D7 (contradicting the zero-exposure "
                       "reconstruction) or run-to-run variation at a fixed seed; one pair of passes cannot separate them", lean)
    return ("(c)", "NO READING (E = 7: one above the stock range, two below B1)", None)


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


def reqlog_extra(paths, seed):
    """Reasoning length per request from the recording-proxy logs, keyed by messages_sha256 (counts only, no text)."""
    out = {}
    for r in A.load_reqlogs([p for p in paths if os.path.isfile(p)]):
        if r.get("seed") == seed:
            out[r.get("messages_sha256")] = {"reasoning_chars": r.get("reasoning_chars"), "reasoning_tokens": r.get("reasoning_tokens")}
    return out


def identity(S, B, XS=None, XB=None):
    """Reply identity between two passes at the same request seed: identical content_sha256 (= response_sha256, verified
    against the request log by the loader) AND identical completion_tokens. Split into non-empty identical replies (I_ne,
    the count that is read) and both-empty pairs (identical by construction). Strict sub-counts also require equal
    reasoning characters and reasoning tokens (request logs)."""
    docs = [d for d in sorted(S) if d in B and S[d]["logged"] and B[d]["logged"]]
    spec = [d for d in docs if S[d]["response_sha256"] == B[d]["response_sha256"] and S[d]["completion_tokens"] == B[d]["completion_tokens"]]
    ne = [d for d in spec if not S[d]["empty"]]
    both_empty = [d for d in spec if S[d]["empty"]]
    both_cap = [d for d in both_empty if cause(S[d]) == "cap_exhaustion" and cause(B[d]) == "cap_exhaustion"]
    out = dict(questions_compared=len(docs), identical_content_and_tokens=len(spec),
               nonempty_identical_I_ne=len(ne), nonempty_identical_docs=ne,
               both_empty_identical_by_construction=len(both_empty), both_empty_at_cap=len(both_cap),
               identity_reading=identity_reading(len(ne)))
    if XS is not None and XB is not None:
        def same_reasoning(d):
            a, b = XS.get(S[d]["messages_sha256"]), XB.get(B[d]["messages_sha256"])
            return bool(a) and bool(b) and a["reasoning_chars"] == b["reasoning_chars"] and a["reasoning_tokens"] == b["reasoning_tokens"]
        out["nonempty_identical_with_equal_reasoning_length"] = sum(1 for d in ne if same_reasoning(d))
        out["both_empty_with_equal_reasoning_length"] = sum(1 for d in both_empty if same_reasoning(d))
    return out


def empty_agreement(S, B):
    docs = [d for d in sorted(S) if d in B]
    both = sum(1 for d in docs if S[d]["empty"] and B[d]["empty"])
    s_only = sum(1 for d in docs if S[d]["empty"] and not B[d]["empty"])
    b_only = sum(1 for d in docs if B[d]["empty"] and not S[d]["empty"])
    neither = len(docs) - both - s_only - b_only
    return dict(questions=len(docs), empty_in_both=both, empty_only_in_new_pass=s_only, empty_only_in_reference=b_only,
                empty_in_neither=neither, empty_status_agrees=both + neither)


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


def paired_tokens(S, B, seed):
    both = [d for d in sorted(S) if d in B and S[d]["logged"] and B[d]["logged"]]
    dtok = [S[d]["completion_tokens"] - B[d]["completion_tokens"] for d in both]
    lo, hi = A.boot_interval(dtok, len(both), seed) if both else (None, None)
    return {"questions": len(both), "estimate": round(sum(dtok) / len(both), 1) if both else None,
            "ci95": [round(lo, 1), round(hi, 1)] if both else None,
            "note": "all requests incl. cap exhaustions; question bootstrap; descriptive"}


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


def combined(bundle_passes, cmp_passes, label, desc=None):
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
    if desc is None:
        desc = ("two implementations (B0 seeds 1234-1235, B1 seed 1236), NOT a three-pass B1 result; historical controls; "
                "B1 pass chosen after inspecting B0's passes")
    return dict(label=f"EXPLORATORY ({label}): {desc}",
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


def record(R):
    return {"E": sum(r["empty"] for r in R.values()), "raw": sum(r["correct_flexible"] for r in R.values()),
            "stated": sum(r["correct_stated"] for r in R.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass-dir", required=True)
    ap.add_argument("--reqlog", required=True)
    ap.add_argument("--serverlog", default=None)
    ap.add_argument("--inspect", default=None, help="docker inspect of the B0 server (required unless --dry-run-label)")
    ap.add_argument("--gpu-hours", type=float, default=None)
    ap.add_argument("--adjudication", default=None,
                    help="JSON {rows: [{doc, hedged}]}: hand reads of every row where stated differs from flexible-extract")
    ap.add_argument("--label", default="B0 checkpoint pass (seed 1236)")
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
    # B0 passes 1-2 (separate checkpoint records; part B)
    B0P = []
    for b in B0:
        hed = [int(r["doc"]) for r in json.load(open(b["adjudication"]))["rows"] if r.get("hedged")]
        R, cb = load_single(b["label"], b["dir"], [b["reqlog"]], hed, b["seed_pass"])
        checks += cb
        B0P.append(R)
    # B1 seed-1236 pass (final, adjudicated): must reproduce its frozen final result
    b1_hed = [int(r["doc"]) for r in json.load(open(B1["adjudication"]))["rows"] if r.get("hedged")]
    B1P, cb1 = load_single(B1["label"], B1["dir"], [B1["reqlog"]], b1_hed, SEED_PASS)
    checks += cb1
    b1_causes = {}
    for r in B1P.values():
        c = cause(r)
        if c:
            b1_causes[c] = b1_causes.get(c, 0) + 1
    b1_rec = record(B1P)
    checks.append(("B1 seed-1236 pass reproduces its final frozen result (E 9, all cap exhaustion, raw 173, stated 174)",
                   b1_rec == {"E": B1_FINAL["E"], "raw": B1_FINAL["raw"], "stated": B1_FINAL["stated"]} and b1_causes == B1_FINAL["causes"],
                   json.dumps({"now": b1_rec, "causes": b1_causes})))
    # the new pass: prompts, lm-eval configuration, stopping
    for name, B in (("record pass 3", rec[3]), ("arm pass 3", arm[3]), ("0.7.0 pass 3", v070[3]), ("B1 pass (seed 1236)", B1P)):
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
        new, ref, b1i = inspect_view(a.inspect), inspect_view(B0_INSPECT), inspect_view(B1["inspect"])
        checks.append(("container image is B0", bool(new) and new["image"] == B0_IMAGE, new and new["image"]))
        diff = [k for k in ref if new[k] != ref[k]] if new else ["no inspect record"]
        checks.append(("container settings equal to B0's seed-1235 session, image included (env as a set, cmd, args, binds, ports, "
                       "shm, ipc, devices, network, runtime)", not diff, f"differ: {diff}"))
        diff1 = [k for k in b1i if k != "image" and new[k] != b1i[k]] if (new and b1i) else ["no inspect record"]
        checks.append(("container settings equal to B1's seed-1236 session apart from the image", not diff1, f"differ: {diff1}"))
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
        # ---------------- PART C: B0 vs B1 at seed 1236 (primary for this review)
        xs = reqlog_extra([a.reqlog], 1236)
        x_b1 = reqlog_extra([B1["reqlog"]], 1236)
        x_rec = {s: reqlog_extra(REC["reqlog"], s) for s in (1234, 1235, 1236)}
        x_arm = {s: reqlog_extra(ARM["reqlog"], s) for s in (1234, 1235, 1236)}
        x_b0 = {1234: reqlog_extra([B0[0]["reqlog"]], 1234), 1235: reqlog_extra([B0[1]["reqlog"]], 1235)}
        ident = identity(S, B1P, xs, x_b1)
        code, statement, lean = reading_b0b1(E, ident["nonempty_identical_I_ne"])
        flags = []
        if material:
            flags.append("B0 pass has a material concern (family standard): review item")
        if concern:
            flags.append("B0 pass has an accuracy concern (raw 159-163): review item")
        b1_window = window(B1P)
        res["part_C_b0_vs_b1_seed1236"] = {
            "reading": {"code": code, "statement": statement, "identity_lean": lean, "review_items": flags,
                        "inputs": {"E_b0": E, "E_b1": B1_FINAL["E"], "I_ne": ident["nonempty_identical_I_ne"]},
                        "rule": f"(a) E >= {E_SHARED}; (b) E <= {E_STOCK_MAX} (identity lean: I_ne >= {IDENT_NEAR_DET} near-deterministic, "
                                f"I_ne <= {IDENT_NONDET_MAX} nondeterminism dominates); (c) otherwise (E = 7) no reading",
                        "depends_on_adjudication": False,
                        "not_claimed": "no significance or equivalence; one pair of passes cannot separate D7 from run-to-run variation; "
                                       "no D7 attribution beyond this reading"},
            "b0": {"E": E, "causes": causes, "raw": raw, "stated": stated, "stated_provisional": bool(missing)},
            "b1": {"E": b1_rec["E"], "causes": b1_causes, "raw": b1_rec["raw"], "stated": b1_rec["stated"]},
            "versus_b1": compare(S, B1P),
            "empty_status_agreement": empty_agreement(S, B1P),
            "reply_identity": ident,
            "first_divergence": "not measurable: no token-level logs (only final content hash, completion tokens and reasoning length "
                                "per request); where replies differ, the divergence position is unknown",
            "generated_tokens": {"b0": tokens(S), "b1": tokens(B1P),
                                 "paired_mean_per_request_b0_minus_b1": paired_tokens(S, B1P, BOOT_SEED + 5)},
            "windows": {"b0": window(S), "b1": b1_window},
            "identity_context_pairs": {
                "B1 s1236 vs stock record p3 (s1236)": identity(B1P, rec[3], x_b1, x_rec[1236]),
                "B1 s1236 vs col511 arm p3 (s1236)": identity(B1P, arm[3], x_b1, x_arm[1236]),
                "B0 s1234 vs stock record p1 (s1234)": identity(B0P[0], rec[1], x_b0[1234], x_rec[1234]),
                "B0 s1235 vs stock record p2 (s1235)": identity(B0P[1], rec[2], x_b0[1235], x_rec[1235]),
                "B0 s1236 vs stock record p3 (s1236)": identity(S, rec[3], xs, x_rec[1236]),
                "B0 s1234 vs col511 arm p1 (s1234)": identity(B0P[0], arm[1], x_b0[1234], x_arm[1234]),
                "B0 s1235 vs col511 arm p2 (s1235)": identity(B0P[1], arm[2], x_b0[1235], x_arm[1235]),
                "B0 s1236 vs col511 arm p3 (s1236)": identity(S, arm[3], xs, x_arm[1236]),
                "stock record p2 vs col511 arm p2 (s1235)": identity(rec[2], arm[2], x_rec[1235], x_arm[1235]),
                "stock record p3 vs col511 arm p3 (s1236)": identity(rec[3], arm[3], x_rec[1236], x_arm[1236]),
                "note": "different builds except B0/B1 (same computation on GPQA under the D7 reconstruction); both-empty pairs are "
                        "identical by construction and are not read"}}
        res["part_A_b0_pass_vs_stock"] = {
            "note": "continuity with the B1 package: 15b category on the family standard, baseline E 6; the amendment-19 reading "
                    "and the 14c readout are reported for continuity only and decide nothing here",
            "empties": E, "causes": causes, "category_15b": final, "category_15b_if_adjudication_complete": outcome,
            "reading_amendment19_continuity_only": final19, "reading_amendment19_if_adjudication_complete": a19, "reading_reasons": a19_why,
            "adjudication_complete": not missing, "adjudicated_docs": sorted(read),
            "flags_family_standard": {"material_concern": material, "accuracy_concern": concern,
                                      "detail": {"raw": raw, "stated": stated, "premature_stops": prem}},
            "sensitivity_matched_offsets_vs_record_pass3": {"baseline_raw": rb, "baseline_stated": sb, "flag": sens_flag, "concern": sens_concern,
                                                            "rule": f"flag raw <= {rb - OFFSET_FLAG} or stated <= {sb - OFFSET_FLAG} "
                                                                    f"or >= 2 premature; concern raw {rb - OFFSET_FLAG + 1}-{rb - OFFSET_CONCERN}"},
            "sensitivity_14c_absolute_rule": reading_14c(E, raw, stated, prem),
            "versus": {"stock 0.9.1 record pass 3 (primary reference for continuity, seed 1236)": compare(S, rec[3]),
                       "col511 arm pass 3 (context, seed 1236)": compare(S, arm[3]),
                       "stock 0.7.0 pass 3 (context, seed 1236)": compare(S, v070[3])},
            "secondary": {"raw_correct_all_198": raw, "stated_correct_all_198": stated,
                          "stated_differs_from_raw_rows": [d for d, r in S.items() if r["stated_differs_from_flexible"]],
                          "hedged_by_hand": sorted(d for d, r in S.items() if r["hedged_by_hand"]),
                          "generated_tokens": {"b0_s1236": tokens(S), "record_pass3": tokens(rec[3]), "arm_pass3": tokens(arm[3]),
                                               "v070_pass3": tokens(v070[3]),
                                               "paired_mean_per_request_b0_minus_record_pass3": {
                                                   "questions": len(both), "estimate": round(sum(dtok) / len(both), 1) if both else None,
                                                   "ci95": [round(tlo, 1), round(thi, 1)] if both else None,
                                                   "note": "all requests incl. cap exhaustions; question bootstrap; descriptive"}},
                          "timing": tim,
                          "comparator_windows_context": {"record_pass3": window(rec[3]), "arm_pass3": window(arm[3]), "v070_pass3": window(v070[3]),
                                                         "b1_s1236": b1_window,
                                                         "note": "comparators' pass 3 was the second pass of its server session (B1: first "
                                                                 "pass of a fresh server, as this pass); request windows only"},
                          "server": srv}}
        b0_desc = ("ONE implementation (B0 seeds 1234, 1235, 1236); historical controls; seed reuse across engines; the seed-1236 "
                   "B0 pass was chosen after inspecting B1's seed-1236 pass; B1 stays separate")
        cont_ref = json.load(open(B1["final_json"]))["part_B_exploratory_corrected_family"]
        cont = {"vs_record_passes_1_3": combined([B0P[0], B0P[1], B1P], [rec[1], rec[2], rec[3]],
                                                 "corrected family seeds 1234-1236 vs stock 0.9.1 record seeds 1234-1236"),
                "vs_arm_passes_1_3": combined([B0P[0], B0P[1], B1P], [arm[1], arm[2], arm[3]],
                                              "corrected family seeds 1234-1236 vs column-511 arm seeds 1234-1236")}
        cont_ok = all(json.loads(json.dumps(cont[k], default=str)) == cont_ref[k] for k in cont)
        res["part_B_exploratory"] = {
            "B-1_b0_three_seed_record": {
                "vs_record_passes_1_3": combined([B0P[0], B0P[1], S], [rec[1], rec[2], rec[3]],
                                                 "B0 seeds 1234-1236 vs stock 0.9.1 record seeds 1234-1236", b0_desc),
                "vs_arm_passes_1_3": combined([B0P[0], B0P[1], S], [arm[1], arm[2], arm[3]],
                                              "B0 seeds 1234-1236 vs column-511 arm seeds 1234-1236", b0_desc),
                "per_seed_records": {"B0 seed 1234": record(B0P[0]), "B0 seed 1235": record(B0P[1]), "B0 seed 1236": record(S),
                                     "stock record seeds 1234/1235/1236": [record(rec[p]) for p in (1, 2, 3)],
                                     "col511 arm seeds 1234/1235/1236": [record(arm[p]) for p in (1, 2, 3)],
                                     "B1 seed 1236 (separate checkpoint record)": b1_rec}},
            "B-2_continuity_two_implementations": dict(cont, reproduces_b1_package_part_B=cont_ok)}
        res["checks"].append(["B-2 continuity view reproduces the B1 package's final part B (informative)", cont_ok, ""])
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=str)
    order = ["label", "dry_run", "valid", "part_C_b0_vs_b1_seed1236", "part_A_b0_pass_vs_stock", "part_B_exploratory",
             "input_pins_informative", "script_sha256", "analysis_col511_sha256", "accuracy_standard"]
    print(json.dumps({k: res[k] for k in order if k in res}, indent=1, default=str)[:30000])
    bad = [c for c in res["checks"] if not c[1]]
    if bad:
        print("INVALID:" if not valid else "INFORMATIVE CHECK FAILED:", json.dumps(bad, default=str)[:1500])
    return 0 if valid else 3


if __name__ == "__main__":
    sys.exit(main())
