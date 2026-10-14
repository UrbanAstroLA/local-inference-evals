"""Follow-up a2, check 2: identify the H4-meta disagreements (KDA layer-metadata num_accepted != previous step's
accepted + 1) in attempt 1 (/run1) and attempt 2 (/run); for each snapshot record step, F, frontier, schedule, rank,
layer, metadata values, the hook's per-step accepted history around the snapshot, the KDA group block ids, and the
committed-slot identification (H4b) at that snapshot. Replicates analyze_hookval.py r_h4's meta rule verbatim.
-> out/f2_meta.json"""
import glob, json, os, re, collections, sys
import numpy as np, torch

out = {}
for RUN in ("/run1", "/run"):
    REQ = {}
    for p in glob.glob(os.path.join(RUN, "forcewin", "req_V3*_r*.pt")):
        m = re.search(r"req_(.*)_r(\d+)\.pt$", os.path.basename(p))
        REQ.setdefault(m.group(1), {})[int(m.group(2))] = p
    SNAP = {}
    for p in glob.glob(os.path.join(RUN, "forcewin", "snap_V3*_r*.pt")):
        m = re.search(r"snap_(.*)_F(\d+)(win)?_r(\d+)\.pt$", os.path.basename(p))
        SNAP.setdefault((m.group(1), int(m.group(2)), bool(m.group(3))), {})[int(m.group(4))] = p
    CL = [json.loads(l) for l in open(os.path.join(RUN, "client.jsonl"))]
    creq = {r["tag"].replace(":", "_"): r for r in CL if r.get("kind") == "request"}
    rows, allsnap = [], []
    for (tag, F, win), ranks in sorted(SNAP.items()):
        if win or not tag.startswith("V3-"):
            continue
        ref = next((rr for (rt, rF, rw), rr in SNAP.items() if rt == f"V3ref-F{F}-{tag}" and not rw), None)
        if ref is None:
            continue
        for rank, p in sorted(ranks.items()):
            if rank not in ref:
                continue
            a = torch.load(p, weights_only=False)
            b = torch.load(ref[rank], weights_only=False)
            rq = torch.load(REQ[tag][rank] if rank in REQ[tag] else next(iter(REQ[tag].values())), weights_only=False)
            C = {n: i for i, n in enumerate(rq["cols"])}
            M = rq["metrics"].numpy()
            st = M[:, C["step"]]
            steps = sorted(set(st[np.isfinite(st)].astype(int).tolist()))
            prev = max([s for s in steps if s < a["step"]], default=None)
            acc = lambda s: sorted(set(M[st == s, C["accepted"]].astype(int).tolist()))
            kl = {n: L for n, L in a["layers"].items() if L.get("kind") == "kda"}
            lna = {L.get("num_accepted") for L in kl.values()}
            pa = acc(prev) if prev is not None else None
            ok = pa is not None and len(pa) == 1 and lna == {pa[0] + 1}
            # identification at this snapshot
            na = a["num_accepted"]
            ident = []
            for n, L in kl.items():
                Lb = b["layers"].get(n)
                if Lb is None or "ssm_sketch" not in L or "ssm_sketch" not in Lb:
                    continue
                r0 = Lb["ssm_sketch"][0]
                d = [float((s - r0).norm() / max(float(r0.norm()), 1e-12)) for s in L["ssm_sketch"]]
                c = d[na - 1]
                oth = [x for k, x in enumerate(d) if k != na - 1]
                ident.append(dict(layer=int(n.split(".")[3]), dists=[round(x, 5) for x in d], committed=na - 1,
                                  nearest=int(np.argmin(d)), margin=round(min(oth) / max(c, 1e-12), 3)))
            rec = dict(tag=tag, F=F, rank=rank, P=a["P"], step=a["step"], frontier_pos=a["frontier_pos"],
                       num_computed=a["num_computed"], total_len=a["total_len"], ms_num_accepted=na,
                       layer_meta_num_accepted=sorted(x for x in lna if x is not None), prev_step=prev,
                       prev_accepted=pa, snap_step_accepted=acc(a["step"]),
                       consistent=ok, mode=tag.split("-")[1], W=creq.get(tag, {}).get("W"),
                       next_block_boundary=((a["frontier_pos"] // 4352) + 1) * 4352,
                       dist_to_boundary=((a["frontier_pos"] // 4352) + 1) * 4352 - a["frontier_pos"],
                       ident_all=all(x["nearest"] == x["committed"] for x in ident), n_kda=len(ident),
                       margin_min_layers_0_2=min([x["margin"] for x in ident if x["layer"] <= 2], default=None))
            # source prediction (preprocess_mamba_align_fused_kernel, vllm/v1/worker/mamba_utils.py): the layer
            # metadata num_accepted is the previous step's accepted + 1, RESET TO 1 when this step's pre-forward
            # running block column state_idx = ceil(computed_after / 4352) - 1 changes, computed_after = num_computed
            # before the step + query length (1 + K drafts, K = 3)
            K, B = 3, 4352
            acc_now = rec["snap_step_accepted"][0] if len(rec["snap_step_accepted"]) == 1 else None
            if acc_now is not None and pa is not None and len(pa) == 1:
                nc_before = a["num_computed"] - (acc_now + 1)
                nc_before_prev = nc_before - (pa[0] + 1)
                col_now = -(-(nc_before + 1 + K) // B) - 1
                col_prev = -(-(nc_before_prev + 1 + K) // B) - 1
                rec["pred"] = dict(nc_before=nc_before, col_prev=col_prev, col_now=col_now,
                                   boundary_reset=col_prev != col_now,
                                   meta=1 if col_prev != col_now else pa[0] + 1)
                rec["pred_ok"] = rec["layer_meta_num_accepted"] == [rec["pred"]["meta"]]
            else:
                rec["pred"], rec["pred_ok"] = None, None
            allsnap.append(rec)
            if not ok:
                # step history around the snapshot: (step, accepted, n_rows) for the last 8 steps of this request
                hist = []
                for s in [s for s in steps if s <= a["step"]][-8:]:
                    hist.append((s, acc(s), int((st == s).sum())))
                rec["history"] = hist
                rec["kda_groups"] = [dict(g=g["g"], block_size=g["block_size"], ids=g["ids"].tolist())
                                     for g in a["groups"] if g["block_size"] == 4352 and g["bpk"] == 1]
                rec["ref_kda_groups"] = [dict(g=g["g"], ids=g["ids"].tolist()) for g in b["groups"]
                                         if g["block_size"] == 4352 and g["bpk"] == 1]
                L0 = next(iter(kl.values()))
                rec["layer_slots"] = {int(n.split(".")[3]): L["slots"].tolist() for n, L in list(kl.items())[:3]}
                rec["spec_rank"] = {L.get("spec_rank") for L in kl.values()}.pop()
                rec["ident"] = ident
                rows.append(rec)
    # all snapshots of the same request around the disagreeing F (context)
    ctx = {}
    for r in rows:
        ctx[r["tag"]] = [dict(F=s["F"], rank=s["rank"], step=s["step"], frontier=s["frontier_pos"],
                              ms=s["ms_num_accepted"], meta=s["layer_meta_num_accepted"], prev=s["prev_accepted"],
                              now=s["snap_step_accepted"], ok=s["consistent"]) for s in allsnap if s["tag"] == r["tag"]]
    near = [dict(tag=s["tag"], F=s["F"], rank=s["rank"], frontier=s["frontier_pos"], ok=s["consistent"])
            for s in allsnap if s["dist_to_boundary"] <= 8]
    pred = collections.Counter(str(s["pred_ok"]) for s in allsnap)
    resets = [dict(tag=s["tag"], F=s["F"], rank=s["rank"], frontier=s["frontier_pos"], pred=s["pred"],
                   meta=s["layer_meta_num_accepted"], prev=s["prev_accepted"], ok=s["consistent"])
              for s in allsnap if s["pred"] and s["pred"]["boundary_reset"]]
    print("prediction matches observed metadata:", dict(pred), "boundary resets predicted:", resets)
    out[RUN] = dict(n=len(allsnap), differs=len(rows), rows=rows, request_context=ctx, near_boundary=near,
                    prediction=dict(pred), predicted_resets=resets,
                    all=[{k: v for k, v in s.items() if k not in ("ident",)} for s in allsnap],
                    frontier_max=max(s["frontier_pos"] for s in allsnap))
    print(RUN, len(allsnap), "differs", len(rows))
    for r in rows:
        print(json.dumps({k: v for k, v in r.items() if k not in ("ident",)}))
    print("near boundary (<= 8):", near)
json.dump(out, open("/f/out/f2_meta.json", "w"), indent=1, default=str)
