#!/usr/bin/env python3
"""[A22-5] Conv-state comparison on the EXISTING attempt-2 V3 snapshots (post-inspection; CPU only; reported, not a
gate of attempt 2, whose frozen verdict is unchanged). Purpose: (1) establish the conv-state row <-> position mapping
used by fc_common.conv_entries and validate the first-contrast CV analysis functions on real engine data; (2) close,
as far as existing data allow, FOLLOWUP-a2's recorded limit "the conv state at the boundary is not compared".
Inputs (read-only): hook-validation/run-a2/forcewin/snap_V3-*.pt and snap_V3ref-*.pt (53 snapshot/reference pairs x 2
ranks; two requests cross the 4,352 block boundary). Writes conv_a2.json. Run with ./crun.sh conv_a2.py
"""
import collections
import json
import re
import sys

sys.path.insert(0, "/a")
import fc_common as fc  # noqa: E402

RUN = fc.Run("/run")
out = dict(pairs=0, snapshots=0, by_layer={}, screen={}, slots={}, persist={}, kda_vs_prefill={}, mapping={})
rows_conv, rows_kda, slot_bad, per_layer = [], [], [], collections.defaultdict(list)
alt = collections.Counter()
for (tag, F, win), ranks in sorted(RUN.SNAP.items()):
    if win or not tag.startswith("V3-"):
        continue
    rk = [(t, f, w) for (t, f, w) in RUN.SNAP if t == "V3ref-F%d-%s" % (F, tag) and not w]
    for rank, p in ranks.items():
        a = RUN.load(p)
        out["snapshots"] += 1
        for b_ in fc.slot_check(a):
            slot_bad.append((tag, F, rank) + tuple(b_))
        if not rk or rank not in RUN.SNAP[rk[0]]:
            continue
        b = RUN.load(RUN.SNAP[rk[0]][rank])
        out["pairs"] += 1
        for r in fc.conv_ident(a, b):
            r.update(tag=tag, F=F, rank=rank, cross=a["frontier_pos"] >= 4348)
            rows_conv.append(r)
            per_layer[r["layer"]].append(r)
        for r in fc.kda_ident(a, b):
            rows_kda.append(r)
        # mapping robustness: does any OTHER row offset match better at layer 0 (would indicate a wrong mapping)?
        L0 = fc.kda_layers(a).get("language_model.model.layers.0.self_attn")
        R0 = fc.kda_layers(b).get("language_model.model.layers.0.self_attn")
        if L0 is not None and R0 is not None:
            c = L0["conv"].float()
            r_ = R0["conv"].float()
            na = a["num_accepted"]
            best = min(range(0, 4), key=lambda s: sum(fc.rel(c[s + j], r_[j]) for j in range(3)))
            alt["mapped offset (num_accepted-1) is the best layer-0 match" if best == na - 1 else "OTHER"] += 1
ok_c, sc = fc.ident_screen(rows_conv)
ok_k, sk = fc.ident_screen(rows_kda)
out["screen"] = dict(conv_vs_prefill=dict(ok=ok_c, **sc), kda_vs_prefill_reproduced=dict(ok=ok_k, **sk))
out["mapping"] = dict(alt)
out["by_layer"] = {L: dict(n=len(v), ident=sum(r["ident"] for r in v) / len(v), rel_q50=fc.q([r["rel"] for r in v], .5),
                           rel_max=max(r["rel"] for r in v), margin_min=min(r["margin"] for r in v))
                   for L, v in sorted(per_layer.items())}
cr = [r for r in rows_conv if r["cross"]]
okx, sx = fc.ident_screen(cr)
out["screen"]["conv_vs_prefill_at_block_crossing"] = dict(ok=okx, **sx)
out["slots"] = dict(violations=slot_bad[:20], n_violations=len(slot_bad))
by = collections.defaultdict(dict)
for (tag, F, win), ranks in RUN.SNAP.items():
    if not win and tag.startswith("V3-"):
        for rank, p in ranks.items():
            by[(tag, rank)][F] = p
pers = {}
for (tag, rank), d in sorted(by.items()):
    s = [(F, RUN.load(d[F])) for F in sorted(d)]
    pers["%s|r%d" % (tag, rank)] = fc.conv_persist(s)
tot = dict(pairs=sum(v["pairs"] for v in pers.values()), rows=sum(v["rows"] for v in pers.values()),
           identical=sum(v["identical"] for v in pers.values()),
           straddling_pairs=sum(v["straddling_pairs"] for v in pers.values()))
out["persist"] = dict(total=tot, requests_with_shared_rows={k: v for k, v in pers.items() if v["rows"]})
json.dump(out, open("/c/conv_a2.json", "w"), indent=1, default=str)
print(json.dumps(dict(pairs=out["pairs"], snapshots=out["snapshots"], mapping=out["mapping"],
                      conv=out["screen"]["conv_vs_prefill"], crossing=out["screen"]["conv_vs_prefill_at_block_crossing"],
                      kda=out["screen"]["kda_vs_prefill_reproduced"], slots=out["slots"]["n_violations"],
                      persist=tot), default=str, indent=1))
print("per layer (n, ident, rel q50, rel max, margin min):")
for L, v in out["by_layer"].items():
    print("  L%-3d %4d %.3f %.4f %.4f %.2f" % (L, v["n"], v["ident"], v["rel_q50"], v["rel_max"], v["margin_min"]))
