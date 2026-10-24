#!/usr/bin/env python3
"""G4 verdict from moe-g4a.json (gate) and moe-g4b.json (reported).  Host python3.
usage: g4_verdict.py AO      writes AO/verdict.json
G4 PASS iff the G4a replay completed for layers 3, 23 and 44, each with >= 1 captured entry, and EVERY entry has
baseline.bitequal_all_rows, baseline.repeat_bitequal and baseline.apply_bitequal_served true (the patched image,
env unset, reproduces the served stock MoE output bit for bit; captured-p3b's own baseline criterion).
G4b (reported): the decode launch map shows K64/N256 at m 16-32 on every layer, and on every captured traffic entry
with 16 <= m <= 32 the decode plan equals the prefill plan on all real rows ("between" n_rows_differ == 0).
"""
import json
import os
import sys


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ao = sys.argv[1]
    a, b = load(os.path.join(ao, "moe-g4a.json")), load(os.path.join(ao, "moe-g4b.json"))
    out = dict(G4="FAIL", G4b="NOT RUN", detail={})
    if a:
        lay = a.get("layers", {})
        per = {}
        ok = a.get("status") == "complete" and sorted(lay) == ["23", "3", "44"]
        for L, ld in lay.items():
            ents = ld.get("entries", [])
            bl = [e.get("baseline") or {} for e in ents]
            good = [x.get("bitequal_all_rows") is True and x.get("repeat_bitequal") is True
                    and x.get("apply_bitequal_served") is True for x in bl]
            per[L] = dict(entries=len(ents), reproduced=sum(good), errors=sum(1 for e in ents if e.get("error")))
            ok = ok and len(ents) > 0 and all(good)
        out["G4"] = "PASS" if ok else "FAIL"
        out["detail"]["G4a"] = dict(status=a.get("status"), layers=per, errors=a.get("errors"))
    if b:
        lay = b.get("layers", {})
        eff, eq, n = True, True, 0
        for L, ld in lay.items():
            lm = ld.get("launch_map", {})
            dec = {k: v for k, v in lm.items() if k.startswith("decode:m") and 16 <= int(k.split("m")[-1]) <= 32}
            eff = eff and bool(dec) and all(v.get("fc1_tile_k") == 64 and v.get("fc1_tile_n") == 256 and
                                            v.get("fc2_tile_k") == 64 and v.get("fc2_tile_n") == 256 for v in dec.values())
            for e in ld.get("entries", []):
                if e.get("phase") == "traffic" and 16 <= int(e.get("m", 0)) <= 32:
                    n += 1
                    eq = eq and (e.get("between") or {}).get("n_rows_differ") == 0
        out["G4b"] = ("AS PREDICTED" if (b.get("status") == "complete" and eff and eq and n) else
                      "NOT AS PREDICTED") + f" (forcing effective {eff}; decode == prefill on {n} entries: {eq})"
    json.dump(out, open(os.path.join(ao, "verdict.json"), "w"), indent=1)
    print("G4", out["G4"], "| G4b", out["G4b"], "|", json.dumps(out["detail"].get("G4a", {}).get("layers")))
    return 0 if out["G4"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
