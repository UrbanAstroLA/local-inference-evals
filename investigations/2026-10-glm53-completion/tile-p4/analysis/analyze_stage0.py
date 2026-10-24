#!/usr/bin/env python3
"""Stage-0 gates (STAGE0-PROTOCOL.md sections 4-5; assessor 22av section 5).  CPU only (torch reads the forcewin
records; run in a CPU container by analysis/run_stage0_analysis.sh).

usage: analyze_stage0.py AO [PO] [--out DIR]
  AO = runs-stage0/stage0-attempt-N (public logs), PO = its private twin (default: runs-stage0 -> PRIVATE-runs-stage0).
  Writes DIR/stage0-result.json and stage0-result.md (default DIR = AO): verdicts, counts, first-difference
  descriptors (window ids, position / step / point indices, layer indices, KV tensor names).  No token ids, no
  values of logits, hidden states or digests.

Per variant V in (V1 solitary, V2 co-batched), repeat 1 vs repeat 2 of the same window, gates IN ORDER; the first
failed gate ends that variant's reading (later gates: "not evaluated"):
  G-enq    the recorded engine-side execution is identical: the enqueue (add) order of the forced requests and, step by
           step from the replicate's first keyed step, the batch composition in batch order (window or warm label,
           scheduled tokens, computed tokens, computed prefill tokens, prefill length, prefilling flag, draft tokens),
           the step's real / padded token and request counts, the graph mode and the eagerly bound MoE plans
           (prefill chunk boundaries, the start of verification and the bound launch incl. padding follow from these).
           Request labels are replicate-free; engine request ids, req_idx, uids and physical block ids are not compared.
  G-entry  window-entry state: for every forced window, on both ranks, the KV-cache digests per tensor and LOGICAL
           block are equal at every digest point (each prefill step of the request and its first non-prefilling
           step, before the step's forward), the points have the same kind / computed / scheduled counts, and every
           cached block of the forced request was computed by its own replicate's warm request of the same reply
           (provenance from the public block records; groups with the largest block size).  Logical block 0 (the
           shared null block, D2's target) is reported but not required (shared scratch written by every request);
           where a request's row holds the null block as a placeholder (both repeats) it is skipped and counted; a
           placeholder in one repeat only is a difference.
  G-out    E1 (MoE output, layers 3/23/44) and E3 (every forcewin metric column but the absolute step id) are bit-equal
           for output positions 0..63 of every window; every one of these positions must be measured exactly once and
           staged in both repeats (missing coverage FAILS the gate, it never passes it).
For the first failed gate the first differing input / state / operation is reported: the first differing step and
field (G-enq); the first differing point, tensor and block, and whether the request's cached-prefix state (point 0)
was equal (G-entry); the first differing position and endpoint, the first differing layer of the per-layer state
digests at that position, and whether that step's recorded composition matched (G-out).
Interpretation (assessor 22av section 5; printed with the verdicts):
  V1 reproducible, V2 not  -> a batch/concurrency-dependent execution difference is implicated; mechanism unassigned
  neither reproducible     -> investigate the first differing input, state or operation; do not label stock kernels
                              nondeterministic
  V2 reproducible          -> proceed only after also establishing a common A/B starting construction
Outputs matching for 64 positions are an operational gate; they do not certify every persistent-state byte.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

W_GATE = 64
VARIANTS = ("V1", "V2")
NOT_EVAL = "not evaluated (stopped at an earlier gate)"
E3_SKIP = ("step",)


def jl(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        for line in open(p):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def norm_tag(tag):
    """Replicate-free label: 'V1-R01-close-r2' -> 'R01-close'; 'W-V2-r1-R03' -> 'W-R03'; '' -> 'unkeyed'."""
    if not tag:
        return "unkeyed"
    p = tag.split("-")
    if p[0] == "W" and len(p) >= 4:
        return "W-" + "-".join(p[3:])
    if p[0] in VARIANTS and len(p) >= 4:
        return "-".join(p[1:-1])
    return tag


def rep_of(tag):
    """'V1-R01-close-r2' -> 'V1-r2'; 'W-V1-r2-R01' -> 'V1-r2'; else None."""
    if not tag:
        return None
    p = tag.split("-")
    if p[0] == "W" and len(p) >= 4:
        return f"{p[1]}-{p[2]}"
    if p[0] in VARIANTS and len(p) >= 4:
        return f"{p[0]}-{p[-1]}"
    return None


class Attempt:
    def __init__(self, ao, po):
        self.ao, self.po = ao, po
        self.client = jl(os.path.join(ao, "client.jsonl"))
        logs0 = jl(os.path.join(ao, "p4log", "p4_r0_p*.jsonl"))
        logs1 = jl(os.path.join(ao, "p4log", "p4_r1_p*.jsonl"))
        self.logs = logs0 + logs1
        self.blocks = {r["tag"]: r for r in logs0 if r.get("kind") == "blocks"}
        self.reqs = {r["tag"]: r for r in logs0 if r.get("kind") == "req"}
        self.entry_init = {r.get("rank"): r for r in self.logs if r.get("kind") == "entry_init"}
        self.errors = [r for r in self.logs if r.get("kind") == "p4_error"]
        self.steps = []
        for r in jl(os.path.join(ao, "p4log", "sched_r0_p*.jsonl")):
            self.steps += r.get("steps", [])
        self.steps.sort(key=lambda d: d["step"])
        self.adds = [r for r in jl(os.path.join(po, "fw", "fw_r0_p*.jsonl")) if r.get("kind") == "add"]

    def fw(self, tag, rank=0):
        p = os.path.join(self.po, "fw", f"req_{tag}_r{rank}.pt")
        if not os.path.exists(p):
            return None
        import torch
        rec = torch.load(p, map_location="cpu", weights_only=False)
        return dict(cols=list(rec["cols"]), m=rec["metrics"].numpy(), nmeas=rec["nmeas"].numpy())

    def npz(self, name):
        p = os.path.join(self.po, "p4", name)
        if not os.path.exists(p):
            return None
        z = np.load(p, allow_pickle=False)
        return {k: z[k] for k in z.files}

    def replicate_steps(self, rname):
        out = []
        for d in self.steps:
            reps = {rep_of(r[3]) for r in d["rows"] if r[3]}
            if rname in reps:
                out.append(d)
        return out


def step_sig(d):
    rows = tuple((norm_tag(r[3]), r[4], r[5], r[6], r[7], r[8], r[9]) for r in d["rows"])
    return dict(nt=d["nt"], ntp=d["ntp"], nr=d["nr"], nrp=d["nrp"], ndraft=d.get("ndraft"), mode=d.get("mode"),
                binds=tuple(tuple(b) for b in (d.get("binds") or [])), rows=rows)


def g_enq(at, var):
    r1, r2 = f"{var}-r1", f"{var}-r2"
    s1, s2 = at.replicate_steps(r1), at.replicate_steps(r2)
    mixed = [d["step"] for d in at.steps if len({rep_of(r[3]) for r in d["rows"] if r[3]}) > 1]
    a1 = [norm_tag(a["tag"]) for a in at.adds if rep_of(a.get("tag")) == r1 and not a["tag"].startswith("W-")]
    a2 = [norm_tag(a["tag"]) for a in at.adds if rep_of(a.get("tag")) == r2 and not a["tag"].startswith("W-")]
    res = dict(steps=[len(s1), len(s2)], enqueue_order=[a1, a2], mixed_replicate_steps=mixed[:10])
    if not s1 or not s2:
        return dict(res, ok=False, why="no recorded steps for a replicate")
    if mixed:
        return dict(res, ok=False, why="steps mixing two replicates")
    if a1 != a2:
        k = next((i for i, (x, y) in enumerate(zip(a1, a2)) if x != y), min(len(a1), len(a2)))
        return dict(res, ok=False, why="enqueue order differs", first_difference=dict(index=k))
    for i, (x, y) in enumerate(zip(s1, s2)):
        sx, sy = step_sig(x), step_sig(y)
        if sx != sy:
            f = next(k for k in sx if sx[k] != sy[k])
            det = dict(step_offset=i, step_ids=[x["step"], y["step"]], field=f)
            if f == "rows":
                j = next((j for j, (u, v) in enumerate(zip(sx["rows"], sy["rows"])) if u != v),
                         min(len(sx["rows"]), len(sy["rows"])))
                det.update(row=j, labels=[sx["rows"][j][0] if j < len(sx["rows"]) else None,
                                          sy["rows"][j][0] if j < len(sy["rows"]) else None],
                           n_rows=[len(sx["rows"]), len(sy["rows"])])
                if j < len(sx["rows"]) and j < len(sy["rows"]):
                    names = ("label", "n_sched", "n_computed", "n_computed_prefill", "prefill_len", "prefilling",
                             "n_draft")
                    det["row_fields"] = [names[k] for k in range(7) if sx["rows"][j][k] != sy["rows"][j][k]]
            else:
                det["values"] = [str(sx[f])[:120], str(sy[f])[:120]]
            return dict(res, ok=False, why="step execution differs", first_difference=det)
    if len(s1) != len(s2):
        return dict(res, ok=False, why="different number of steps",
                    first_difference=dict(step_offset=min(len(s1), len(s2))))
    return dict(res, ok=True)


def windows_of(at):
    st = [d for d in at.client if d.get("kind") == "start"]
    return st[-1]["windows"] if st else []


def provenance(at, var, rep, wid):
    """Cached blocks of the forced request == the same replicate's warm request (same reply) at the same index."""
    tag = f"{var}-{wid}-r{rep}"
    reply = wid.split("-")[0]
    f = at.blocks.get(tag)
    w = at.blocks.get(f"W-{var}-r{rep}-{reply}")
    if f is None or w is None:
        return dict(ok=False, why="block record missing")
    e = at.npz(f"entry_{tag}_r0.npz")
    if e is None:
        return dict(ok=False, why="entry record missing")
    ncached = int(e["points"][0][2])                 # n_computed at the first scheduled step = cached tokens
    bs = f.get("block_sizes") or []
    big = max(bs) if bs else 0
    checked = bad = nulls = 0
    for g, ids in enumerate(f["groups"]):
        if not bs or bs[g] != big or big <= 0:
            continue
        k = ncached // big
        wid_ids = w["groups"][g]
        for i in range(min(k, len(ids))):
            if ids[i] == 0:                  # null-block placeholder (e.g. KDA state not kept for this block): not reused
                nulls += 1
                continue
            checked += 1
            bad += int(i >= len(wid_ids) or ids[i] != wid_ids[i])
    return dict(ok=checked > 0 and bad == 0, cached_tokens=ncached, blocks_checked=checked, not_from_warm=bad,
                null_placeholders=nulls)


def g_entry(at, var):
    wins = windows_of(at)
    out = dict(windows={}, null_block_equal=0, null_block_compared=0, null_in_row_skipped=0)
    first = None
    for wid in wins:
        t1, t2 = f"{var}-{wid}-r1", f"{var}-{wid}-r2"
        wres = dict(prov=[provenance(at, var, 1, wid), provenance(at, var, 2, wid)])
        for rank in (0, 1):
            e1, e2 = at.npz(f"entry_{t1}_r{rank}.npz"), at.npz(f"entry_{t2}_r{rank}.npz")
            if e1 is None or e2 is None:
                wres[f"r{rank}"] = dict(ok=False, why="entry record missing")
                first = first or dict(window=wid, rank=rank, why="entry record missing")
                continue
            p1, p2 = e1["points"], e2["points"]
            if p1.shape != p2.shape or not np.array_equal(p1[:, 1:], p2[:, 1:]) or int(p1[0][0]) < 0:
                wres[f"r{rank}"] = dict(ok=False, why="digest points differ", points=[p1.shape[0], p2.shape[0]])
                first = first or dict(window=wid, rank=rank, why="digest points differ (kind / computed / scheduled)")
                continue
            if list(e1["tensors"]) != list(e2["tensors"]):
                wres[f"r{rank}"] = dict(ok=False, why="tensor lists differ")
                first = first or dict(window=wid, rank=rank, why="tensor lists differ")
                continue
            d1, d2 = e1["edig"], e2["edig"]
            nb = d1.shape[2] - 1
            grp = e1["tensor_group"]
            diff = None
            for pt in range(p1.shape[0]):
                for ti in range(d1.shape[1]):
                    g = int(grp[ti])
                    nlog = int((e1["eblk"][pt, g] >= 0).sum()) if g >= 0 else 0
                    for j in range(nlog):
                        b1, b2 = int(e1["eblk"][pt, g, j]), int(e2["eblk"][pt, g, j])
                        if (b1 == 0) != (b2 == 0):
                            diff = dict(point=pt, point_kind="entry" if int(p1[pt][1]) == 1 else "prefill",
                                        tensor=str(e1["tensors"][ti]), group=g, logical_block=j,
                                        why="null-block placement differs")
                            break
                        if b1 == 0:              # the shared null block in the request's row: not the request's state
                            out["null_in_row_skipped"] += 1
                            continue
                        if not np.array_equal(d1[pt, ti, j], d2[pt, ti, j]):
                            diff = dict(point=pt, point_kind="entry" if int(p1[pt][1]) == 1 else "prefill",
                                        tensor=str(e1["tensors"][ti]), group=g, logical_block=j)
                            break
                    out["null_block_compared"] += 1
                    out["null_block_equal"] += int(np.array_equal(d1[pt, ti, nb], d2[pt, ti, nb]))
                    if diff:
                        break
                if diff:
                    break
            phys = bool(np.array_equal(e1["eblk"], e2["eblk"]))
            wres[f"r{rank}"] = dict(ok=diff is None, points=int(p1.shape[0]), tensors=int(d1.shape[1]),
                                    physical_block_ids_equal=phys, first_difference=diff)
            if diff and first is None:
                pt0_equal = diff["point"] > 0
                first = dict(window=wid, rank=rank, **diff, point0_cached_state_equal=pt0_equal)
        wres["ok"] = all(wres.get(f"r{k}", {}).get("ok") for k in (0, 1)) and all(p["ok"] for p in wres["prov"])
        if first is None and not all(p["ok"] for p in wres["prov"]):
            first = dict(window=wid, why="cached-state provenance", prov=wres["prov"])
        out["windows"][wid] = wres
    out["ok"] = bool(wins) and all(v["ok"] for v in out["windows"].values())
    out["first_difference"] = first
    return out


def g_out(at, var):
    wins = windows_of(at)
    out = dict(windows={})
    first = None
    for wid in wins:
        t1, t2 = f"{var}-{wid}-r1", f"{var}-{wid}-r2"
        f1, f2 = at.fw(t1), at.fw(t2)
        z1, z2 = at.npz(f"p4_{t1}_r0.npz"), at.npz(f"p4_{t2}_r0.npz")
        if f1 is None or f2 is None or z1 is None or z2 is None:
            out["windows"][wid] = dict(ok=False, why="record missing")
            first = first or dict(window=wid, why="record missing")
            continue
        cols = [i for i, c in enumerate(f1["cols"]) if c not in E3_SKIP]
        res = dict(ok=True, positions=W_GATE)
        for o in range(W_GATE):
            why = None
            if f1["nmeas"][o] != 1 or f2["nmeas"][o] != 1:
                why = "not measured once"
            elif not (z1["moe_have"][:, o] == 1).all() or not (z2["moe_have"][:, o] == 1).all():
                why = "E1 not staged"
            else:
                a, b = f1["m"][o, cols], f2["m"][o, cols]
                if not np.array_equal(np.nan_to_num(a, nan=7.0), np.nan_to_num(b, nan=7.0)) or \
                        not np.array_equal(np.isnan(a), np.isnan(b)):
                    k = next(i for i in range(len(cols)) if not (a[i] == b[i] or (a[i] != a[i] and b[i] != b[i])))
                    why = "E3 differs (%s)" % f1["cols"][cols[k]]
                else:
                    for k, n in enumerate(("moe3", "moe23", "moe44")):
                        if not np.array_equal(z1["moe"][k, o], z2["moe"][k, o]):
                            why = "E1 differs (%s)" % n
                            break
            if why:
                det = dict(position=o, endpoint=why)
                if "ldig" in z1 and "ldig" in z2 and z1["ldig_have"][o] == 1 and z2["ldig_have"][o] == 1:
                    dl = [i for i in range(z1["ldig"].shape[1]) if not np.array_equal(z1["ldig"][o, i], z2["ldig"][o, i])]
                    det["first_differing_layer_state"] = int(z1["ldig_layers"][dl[0]]) if dl else None
                    det["layer_digests_differ"] = len(dl)
                sidx = f1["cols"].index("step")
                s1, s2 = int(f1["m"][o, sidx]), int(f2["m"][o, sidx])
                st1 = next((d for d in at.steps if d["step"] == s1), None)
                st2 = next((d for d in at.steps if d["step"] == s2), None)
                det["step_composition_equal"] = (st1 is not None and st2 is not None and step_sig(st1) == step_sig(st2))
                res = dict(ok=False, first_difference=det)
                if first is None or o < first["position"]:
                    first = dict(window=wid, **det)
                break
        out["windows"][wid] = res
    out["ok"] = bool(wins) and all(v["ok"] for v in out["windows"].values())
    out["first_difference"] = first
    return out


def variant(at, var):
    reps = {d.get("replicate") for d in at.client if d.get("kind") == "replicate_done"}
    forced = [d for d in at.client if d.get("kind") == "forced" and str(d.get("replicate", "")).startswith(var)]
    res = dict(replicates_done=sorted(r for r in reps if r and r.startswith(var)),
               forced_ok=sum(1 for d in forced if d.get("ok")), forced=len(forced))
    if res["replicates_done"] != [f"{var}-r1", f"{var}-r2"] or res["forced_ok"] != 16:
        return dict(res, verdict="INCOMPLETE", first_failed="data", reason="both replicates with 8 forced ok each needed",
                    gates={g: NOT_EVAL for g in ("G-enq", "G-entry", "G-out")})
    gates = {}
    for name, fn in (("G-enq", g_enq), ("G-entry", g_entry), ("G-out", g_out)):
        r = fn(at, var)
        gates[name] = r
        if not r["ok"]:
            for later in ("G-enq", "G-entry", "G-out"):
                gates.setdefault(later, NOT_EVAL)
            return dict(res, verdict="NOT REPRODUCIBLE", first_failed=name, gates=gates)
    return dict(res, verdict="REPRODUCIBLE", first_failed=None, gates=gates)


def interpret(v1, v2):
    a, b = v1["verdict"], v2["verdict"]
    if "INCOMPLETE" in (a, b):
        return "INCOMPLETE: %s not evaluable; no interpretation" % ", ".join(
            n for n, v in (("V1", a), ("V2", b)) if v == "INCOMPLETE")
    if a == "REPRODUCIBLE" and b != "REPRODUCIBLE":
        return ("V1 reproducible, V2 not: a batch/concurrency-dependent execution difference is implicated; its "
                "mechanism remains unassigned (first difference: V2 %s)" % v2["first_failed"])
    if a != "REPRODUCIBLE" and b != "REPRODUCIBLE":
        return ("Neither reproducible: investigate the first differing input, state or operation (V1 %s, V2 %s); do "
                "not label stock kernels nondeterministic" % (v1["first_failed"], v2["first_failed"]))
    if b == "REPRODUCIBLE" and a == "REPRODUCIBLE":
        return ("V2 reproducible (and V1): proceed only after also establishing a common A/B starting construction")
    return ("V2 reproducible but V1 not (unexpected): investigate V1's first difference (%s) before any A/B; proceed only "
            "after also establishing a common A/B starting construction" % v1["first_failed"])


def to_md(doc):
    L = ["# Stage-0 result (analysis/analyze_stage0.py; aggregates and first-difference descriptors only)", "",
         f"**Interpretation:** {doc['interpretation']}", "",
         "Outputs matching for 64 positions are an operational gate; they do not certify every persistent-state byte. "
         "Entry digests cover the KV-cache tensors listed as covered in entry_init; logical block 0 (shared null "
         "block) is reported, not required.", "",
         "| variant | verdict | first failed gate | G-enq | G-entry | G-out |", "|---|---|---|---|---|---|"]
    for v in VARIANTS:
        r = doc["variants"][v]
        gs = [("PASS" if isinstance(r["gates"].get(g), dict) and r["gates"][g].get("ok") else
               ("FAIL" if isinstance(r["gates"].get(g), dict) else "-")) for g in ("G-enq", "G-entry", "G-out")]
        L.append(f"| {v} | {r['verdict']} | {r.get('first_failed')} | {gs[0]} | {gs[1]} | {gs[2]} |")
    for v in VARIANTS:
        r = doc["variants"][v]
        ff = r.get("first_failed")
        if ff and isinstance(r["gates"].get(ff), dict):
            g = r["gates"][ff]
            L += ["", f"## {v}: first failed gate {ff}", "", "```",
                  json.dumps(dict(why=g.get("why"), first_difference=g.get("first_difference")), indent=1,
                             default=str), "```"]
    L += ["", "## Instrument health", "", "```", json.dumps(doc["health"], indent=1, default=str), "```"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ao")
    ap.add_argument("po", nargs="?")
    ap.add_argument("--out")
    a = ap.parse_args()
    po = a.po or a.ao.replace(os.sep + "runs-stage0" + os.sep, os.sep + "PRIVATE-runs-stage0" + os.sep)
    at = Attempt(a.ao, po)
    v = {k: variant(at, k) for k in VARIANTS}
    cov = {rk: dict(covered=e.get("covered"), uncovered=len(e.get("uncovered", [])), n_groups=e.get("n_groups"),
                    groups_covered=sorted({int(t["group"]) for t in e.get("tensors", []) if t.get("covered")}),
                    uncovered_layers=[u.get("layer") for u in e.get("uncovered", [])][:32])
           for rk, e in at.entry_init.items()}
    doc = dict(attempt=os.path.basename(a.ao.rstrip("/")), variants=v, interpretation=interpret(v["V1"], v["V2"]),
               health=dict(hook_errors=len(at.errors), entry_coverage=cov, sched_steps=len(at.steps),
                           forced_adds=sum(1 for x in at.adds if not str(x.get("tag", "")).startswith("W-"))))
    out = a.out or a.ao
    os.makedirs(out, exist_ok=True)
    json.dump(doc, open(os.path.join(out, "stage0-result.json"), "w"), indent=1, default=str)
    open(os.path.join(out, "stage0-result.md"), "w").write(to_md(doc))
    print("STAGE0:", doc["interpretation"])
    for k in VARIANTS:
        print(f"  {k}: {v[k]['verdict']} (first failed: {v[k].get('first_failed')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
