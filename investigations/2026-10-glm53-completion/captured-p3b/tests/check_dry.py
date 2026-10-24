#!/usr/bin/env python3
"""Assertions on the --dry replay outputs (tests/test_replay_dry.sh).  Host python3, json only."""
import json
import os
import sys

O = sys.argv[1]
res = []


def load(n):
    p = os.path.join(O, n)
    return json.load(open(p)) if os.path.exists(p) else None


def check(name, ok, **kw):
    res.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + "  " + json.dumps(kw, default=str)[:500])


exp = load("fake/expect.json")
a = load("dry-attn.json")
rows3 = a["layers"]["3"]["rows"] if a else []
na = [(r["seq"], r["row_slot"]) for r in rows3 if not r["anchored"]]
check("A1 attention dry: complete; only the injected row is not anchored (not reproduced)",
      a and a["status"] == "complete" and na == [(1, 2)] and
      all(r["anchored"] for r in a["layers"]["23"]["rows"]), not_anchored=na)
allrows = rows3 + (a["layers"]["23"]["rows"] if a else [])
check("A2 attention dry: compaction control reproduces every served POST table and nsa; Bf table equal on exact cols",
      allrows and all(r["compaction_ctrl"] and r["bf_table_same_on_exact_cols"] for r in allrows), rows=len(allrows))
check("A3 attention dry: P1 control detected on every row; Bc / Bf equal to A on every anchored row (masked -1 columns)",
      allrows and all(r["ctrl_halve"] for r in allrows if r["ctrl_halve"] is not None)
      and all(r["bc_equal"] and r["bf_equal"] for r in allrows if r["anchored"]))
regs = sorted({r["regime"] for r in rows3})
check("A4 attention dry: regimes classified from exact p and L (all four present on layer 3; r from the fabricated L)",
      regs == ["boundary_crossing", "clamped", "dense_r0", "dense_r>=1"] and
      sorted({r["r"] for r in rows3 if r["seq"] == 1 and r["regime"] != "clamped"}) == [2], regimes=regs)
check("A5 attention dry: coverage minima unmet -> INCOMPLETE (no EQUAL verdict from too few rows)",
      a and all(v.startswith("INCOMPLETE") or v in ("informational (control rows)", "no rows")
                for L in a["layers"].values() for v in L["summary"]["verdict"].values()))
pv = json.load(open(os.path.join(O, "fake-priv", "attn_rows_private.json"))) if os.path.exists(os.path.join(O, "fake-priv", "attn_rows_private.json")) else {}
pub_txt = open(os.path.join(O, "dry-attn.json")).read() if a else ""
check("A6 attention dry: positions / lengths only in the private file (0600), not in the public JSON",
      pv and '"p":' not in pub_txt and '"L":' not in pub_txt and
      oct(os.stat(os.path.join(O, "fake-priv", "attn_rows_private.json")).st_mode)[-3:] == "600")
ls = load("dry-attn-ls.json")
v3 = ls["layers"]["3"]["summary"]["verdict"] if ls else {}
check("A7 length-sensitive kernel: DIFFERENCES on inflated dense / boundary rows, EQUAL on clamped (nsa unchanged)",
      v3.get("dense_r>=1") == "DIFFERENCES (characterize)" and v3.get("boundary_crossing") == "DIFFERENCES (characterize)"
      and v3.get("clamped") == "EQUAL ON CAPTURED ROWS", verdicts=v3)
ref = load("dry-attn-refused.json")
check("A8 attention refuses dry-test switches outside --dry (no output written)", ref is None)

m = load("dry-moe-v091.json")
bad = []
emap = {}
for L, ld in (m or {}).get("layers", {}).items():
    for e in ld["entries"]:
        if not e["baseline"]["bitequal_all_rows"]:
            bad.append((int(L), e["seq"]))
        emap[(int(L), e["seq"])] = e
inj = (exp["moe"]["injected"]["layer"], exp["moe"]["injected"]["seq"])
check("M1 MoE v091 dry: BASELINE reproduces every entry except the injected one", m and bad == [inj], not_reproduced=bad)
check("M2 MoE v091 dry: E-plan equal at m <= 16, differs at m >= 24 (dry injection); repeats bit-equal",
      m and all((e["between"]["n_rows_differ"] == 0) == (e["m"] <= 16) for e in emap.values())
      and all(e["within"]["decode"]["bitequal"] and e["within"]["prefill"]["bitequal"] for e in emap.values()))
check("M3 MoE v091 dry: composed m = 40 / 48 present, chunk rows equal the served-shape decode, ulp control ok",
      m and all(len(ld.get("composed", [])) >= 2 and all(c["decode_chunk_rows_equal_served_shape"] for c in ld["composed"])
                and ld.get("control_ulp", {}).get("ok") for ld in m["layers"].values()))
check("M4 forced-tile dry: PREDICTION HOLDS when the forced tile computes as prefill",
      m and m["forced_tile_confirmation"]["verdict"].startswith("PREDICTION HOLDS"))
fd = load("dry-moe-v091-forceddiff.json")
check("M5 forced-tile dry: PREDICTION FAILS when it computes as decode",
      fd and fd["forced_tile_confirmation"]["verdict"].startswith("PREDICTION FAILS"))
check("M6 MoE refuses the dry switch outside --dry", load("dry-moe-refused.json") is None)
m70 = load("dry-moe-v070.json")
check("M7 MoE v070 dry: E-plan only (no BASELINE on another engine's slice)",
      m70 and m70["status"] == "complete" and all("baseline" not in e for ld in m70["layers"].values() for e in ld["entries"]))
pr = load("dry-moe-v091-probe.json")
p = (pr or {}).get("dry_forced_probe", {})
fx = (pr or {}).get("forced_tile_confirmation", {}).get("layers", {}).get("3", {}).get("p3_fixtures", [])
check("M8 forced-tile probe on CPU through the image code: plan_weights got None -> forced; weight plan tile forced; "
      "tile rule K128 at 10-32 stock, forced everywhere; P3 fixture inputs regenerated bit-identically",
      p.get("ok") and len(fx) == 6 and all(f["p3_input_digest_match"] for f in fx),
      reached=p.get("reached"), weight_plan_tile=p.get("weight_plan_tile"))
wi = load("dry-moe-wrongimage.json")
check("M9 MoE refuses the wrong image (exl3.py hash)", wi and wi["status"] == "error" and any("wrong image" in str(e) for e in wi["errors"]))
e2a, e2m = load("e2e/o/attn-v091.json"), load("e2e/o/moe-v091.json")
check("E1 in-container driver (production mounts, --dry): attention, MoE and forced-tile parts all ran",
      e2a and e2m and e2a["status"] == "complete" and e2m["status"] == "complete"
      and "forced_tile_confirmation" in e2m and os.path.exists(os.path.join(O, "e2e/po/attn_rows_private.json")))
print(f"{sum(res)}/{len(res)} passed")
sys.exit(0 if all(res) else 1)
