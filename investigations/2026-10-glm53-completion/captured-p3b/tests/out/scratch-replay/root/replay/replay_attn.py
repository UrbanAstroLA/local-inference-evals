#!/usr/bin/env python3
"""captured-p3b, step B(2): sparse-MLA replay on captured served inputs (stock 0.9.1 image a9e985c0, GPU0).

usage: replay_attn.py --cap CAPDIR --caplog CAPLOGDIR --out OUT.json --priv PRIVDIR [--dry] [--deadline-s N]
                      [--layers 3,23]

For every captured attention entry (one armed verify step, up to 8 rows, layers 3 and 23 of rank 0):
  * compact cache: every KV record the PRE-compaction table references (gathered on device at consumption) plus
    the slot-0 record, re-addressed to compact slot ids; tables are re-addressed with the same map (-1 kept,
    slot 0 kept as 0).  Checks: a slot gathered twice holds the same bytes; every POST slot is covered by the
    PRE gather; no PRE entry pointed past the cache (header aux = cache slots).
  * A   = the served inputs exactly: q as bound, POST table, consumed nsa, consumed cache_lengths, through the
          served decode plan (Caps rebuilt from the capture log, splits pinned as served).  Must reproduce the
          served output bit for bit (row by row) before anything else is read; a row that does not reproduce is
          NOT ANCHORED and excluded from the A/B reading (counted).  A is run twice (determinism).
  * Bc  = count-only: the same POST table, nsa = min(p + 1, W), cache_lengths = p + 1.
  * Bf  = full path: the PRE table through the image's own _compact_dropped_selection and
          _mask_page_table_after_nsa_len with the exact count min(p + 1, W), then the kernel with cache_lengths p + 1.
          Control: the PRE table with the served count min(L, W) must reproduce the served POST table and nsa.
  * P1  = required positive control: the consumed count halved (nsa -> ceil(nsa / 2)); every row with nsa >= 2
          must change (shows the comparator and row attribution work on these inputs).
  * P2  = informational (the level-3 plant): one live slot of another captured row appended at column nsa
          (nsa + 1, cache + 1); its detected fraction is a sensitivity figure: a low-scoring foreign key can be
          invisible after the bf16 output rounding.
Rows are classified offline from the captured exact p and L: clamped (p + 1 >= W), dense (p + 1 < W), and in the
dense regime boundary-crossing (ceil(min(L, W) / 64) > ceil((p + 1) / 64)); r = L - (p + 1).

Outcome per layer and regime (PROPOSAL.md section 7): INCOMPLETE when anchored rows < the minimum, when a positive
control misses, or when a kernel errors; DIFFERENCES (characterize) when Bc or Bf differs from A on any anchored row;
EQUAL ON CAPTURED ROWS otherwise (not a general exclusion).  A P1 miss makes the layer INCOMPLETE.

Public OUT.json: counts, flags, categories (regime, r), comparison magnitudes; no positions, lengths or tensors.
PRIVDIR (0700): per-row positions / lengths / counts for audit.
--dry: CPU only; the real compaction kernels under TRITON_INTERPRET=1, a CPU reference attention instead of the
B12x kernel (the plumbing, maps, controls and outcome logic are what is tested).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_common as rc  # noqa: E402

SCHEMA = "p3b-replay-attn/1"
MIN_ROWS = {"dense_r>=1": 32, "clamped": 16, "boundary_crossing": 8}
if os.environ.get("P3B_DRY_MIN_ROWS_1") == "1":      # dry tests of the verdict logic only (refused outside --dry)
    MIN_ROWS = {k: 1 for k in MIN_ROWS}
CAPS_SERIAL_KEYS_IGNORED = ("device",)


def serialize_caps(caps, torch) -> dict:
    """Same rule as p3b_capture_hooks.serialize_caps (kept in step by tests/test_replay_dry.sh)."""
    import dataclasses
    import enum
    out = {}
    for f in dataclasses.fields(caps):
        v = getattr(caps, f.name, None)
        if isinstance(v, enum.Enum):
            val = v.value if isinstance(v.value, (int, str, bool)) else repr(v.value)
            out[f.name] = dict(enum=type(v).__name__, name=v.name, value=val)
        elif isinstance(v, bool) or v is None or isinstance(v, (int, str)):
            out[f.name] = v
        elif isinstance(v, float):
            out[f.name] = dict(float_hex=float(v).hex())
        elif isinstance(v, torch.dtype):
            out[f.name] = dict(dtype=str(v))
        elif isinstance(v, torch.device):
            out[f.name] = dict(device=str(v))
        else:
            out[f.name] = dict(_repr=repr(v)[:200])
    return out


class Kernel:
    """The served decode plan (GPU) or a CPU reference attention (--dry)."""

    def __init__(self, torch, dry, ai, dev, doc):
        self.torch, self.dry, self.ai, self.dev = torch, dry, ai, dev
        self.info = {}
        import vllm.v1.attention.backends.mla.b12x_mla_sparse as bm
        self.bm = bm
        if dry:
            self.info["kernel"] = "CPU reference attention (dry)"
            return
        from b12x.attention import sparse_mla
        self.sm = sparse_mla
        caps, skipped = rc.deserialize_caps(sparse_mla, torch, ai["decode_caps"], dev)
        ser = serialize_caps(caps, torch)
        diff = {k: (ai["decode_caps"].get(k), ser.get(k)) for k in set(ser) | set(ai["decode_caps"])
                if k not in CAPS_SERIAL_KEYS_IGNORED and ai["decode_caps"].get(k) != ser.get(k)}
        self.info.update(caps_ctor_skipped=skipped, caps_equal_to_served=not diff, caps_diff=diff)
        t0 = time.time()
        try:
            plan, _ = bm._prepare_b12x_sparse_mla_plan(sparse_mla, caps, forced_num_splits=int(ai["num_splits_cap"]))
            self.info["plan_path"] = "vllm _prepare_b12x_sparse_mla_plan (as served)"
        except Exception as exc:  # noqa: BLE001
            os.environ[bm._B12X_MLA_NUM_SPLITS_ENV] = str(int(ai["num_splits_cap"]))
            plan = sparse_mla.plan(caps)
            self.info["plan_path"] = f"fallback sparse_mla.plan with {bm._B12X_MLA_NUM_SPLITS_ENV} pinned ({exc!r:.160})"
        self.plan = plan
        self.info["plan_s"] = round(time.time() - t0, 2)
        try:
            from b12x.attention._shared.mla import kernel as K
            self.K = K
        except Exception:  # noqa: BLE001
            self.K = None

    def run(self, q, cache, sel, clen, nsa):
        torch = self.torch
        if self.dry:
            return self._fake(q, cache, sel, nsa), None
        spec = self.plan.scratch_specs()[0]
        scratch = torch.zeros(spec.shape, dtype=spec.dtype, device=self.dev)
        b = self.sm.bind(self.plan, scratch=scratch, q=q, kv_cache=cache, selected_indices=sel,
                         cache_lengths=clen, selected_lengths=nsa)
        r = self.sm.run(b)
        torch.cuda.synchronize(self.dev)
        if self.K is not None and "last_decode_plan" not in self.info:
            self.info["last_decode_plan"] = {k: str(v) for k, v in dict(getattr(self.K, "LAST_DECODE_PLAN", {})).items()}
        if isinstance(r, tuple):
            return r[0].contiguous().clone(), r[1].contiguous().clone()
        return r.contiguous().clone(), None

    def _fake(self, q, cache, sel, nsa):
        torch = self.torch
        rb = cache.shape[-1]
        flat = cache.reshape(-1, rb).float()
        d, dv = q.shape[-1], int(self.ai["v_head_dim"])
        K = flat[:, :d] / 16.0 - 8.0
        V = flat[:, :dv] / 16.0 - 8.0
        out = torch.zeros(q.shape[0], q.shape[1], dv, dtype=torch.float32)
        len_sensitive = os.environ.get("P3B_DRY_ATTN_LENSENSITIVE") == "1"   # dry test of the DIFFERENCES path
        for i in range(q.shape[0]):
            if len_sensitive:
                cols = torch.arange(sel.shape[1]) < int(nsa[i])
                idx = sel[i][cols].long().clamp_min(0)
            else:
                cols = (torch.arange(sel.shape[1]) < int(nsa[i])) & (sel[i] >= 0)
                idx = sel[i][cols].long()
            if idx.numel() == 0:
                continue
            s = q[i].float() @ K[idx].T * 0.125
            out[i] = torch.softmax(s, dim=-1) @ V[idx]
        return out.to(torch.bfloat16)


def row_eq(torch, a, b):
    ia, ib = a.view(torch.int16), b.view(torch.int16)
    return (ia == ib).reshape(a.shape[0], -1).all(dim=1)


def row_mag(torch, a, b):
    d = (a.float() - b.float())
    absd = torch.nan_to_num(d.abs(), nan=float("inf")).reshape(a.shape[0], -1)
    ref = b.float().reshape(b.shape[0], -1).norm(dim=1)
    rel = d.reshape(d.shape[0], -1).norm(dim=1) / ref.clamp_min(1e-30)
    return absd.max(dim=1).values, rel


def nan_rows(torch, a):
    return torch.isnan(a.float()).reshape(a.shape[0], -1).any(dim=1)


def replay_entry(torch, kern, ai, d, sites_q_heads, sites_out_heads, dev, Wd):
    """Returns (rows_public, rows_private, entry_info)."""
    bm = kern.bm
    hdr = d["hdr"]
    nslots = int(hdr[rc.H_AUX])
    scal = d["scal"]
    valid = np.where(scal[:, rc.S_VALID] == 1)[0]
    info = dict(n_valid=int(len(valid)), nslots=nslots, sha_ok=d.get("_sha_ok"))
    if len(valid) == 0:
        return [], [], dict(info, skipped="no valid rows")
    pre = d["pre"][valid].astype(np.int64)
    post = d["post"][valid].astype(np.int64)
    rec = d["rec"][valid]
    slot0 = d["slot0"][0]
    p = scal[valid, rc.S_POS]
    L = scal[valid, rc.S_L]
    nsa_s = scal[valid, rc.S_NSA]
    clen_s = scal[valid, rc.S_CLEN]
    nv = len(valid)
    oob_row = (pre >= nslots).any(axis=1) if nslots > 0 else np.zeros(nv, bool)
    # compact map over PRE (as gathered) and slot 0
    pre_ok = pre >= 0
    uniq, inv = np.unique(pre[pre_ok], return_inverse=True)
    recs_flat = rec[pre_ok]
    first = np.full(len(uniq), -1, dtype=np.int64)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    rec_u = recs_flat[first]
    consistent = bool((recs_flat == rec_u[inv]).all())
    if len(uniq) and uniq[0] == 0:
        consistent &= bool((rec_u[0] == slot0).all())
    nz = uniq[uniq != 0]
    order = np.concatenate([[0], nz]).astype(np.int64)
    recs = np.concatenate([slot0[None, :], rec_u[uniq != 0]], axis=0)
    post_cov = np.isin(post[post >= 0], order)
    post_missing_rows = np.zeros(nv, bool)
    if not post_cov.all():
        bad = np.zeros_like(post, bool)
        bad[post >= 0] = ~post_cov
        post_missing_rows = bad.any(axis=1)

    def remap(tab):
        out = np.full(tab.shape, -1, dtype=np.int32)
        ok = tab >= 0
        pos = np.searchsorted(order, tab[ok])
        pos = np.clip(pos, 0, len(order) - 1)
        hit = order[pos] == tab[ok]
        vals = np.where(hit, pos, -1)
        out[ok] = vals
        return out

    RB = rec.shape[-1]
    npages = max(1, math.ceil(len(order) / 64))
    cache = torch.zeros((npages, 64, RB), dtype=torch.uint8)
    cache.view(-1, RB)[: len(order)] = torch.from_numpy(recs)
    cache = cache.to(dev)
    qh = sites_q_heads or d["q"].shape[1]
    oh = sites_out_heads or d["out"].shape[1]
    q = rc.bf16(torch, d["q"][valid][:, :qh]).to(dev)
    served = rc.bf16(torch, d["out"][valid][:, :oh]).to(dev)
    i32 = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.int32, device=dev)  # noqa: E731
    sel_A = i32(remap(post))
    oA, _ = kern.run(q, cache, sel_A, i32(clen_s), i32(nsa_s))
    oA = oA[:, :oh]
    oA2, _ = kern.run(q, cache, sel_A, i32(clen_s), i32(nsa_s))
    oA2 = oA2[:, :oh]
    repro = row_eq(torch, oA, served)
    det = row_eq(torch, oA, oA2)
    ex = np.minimum(p + 1, Wd)
    oBc, _ = kern.run(q, cache, sel_A, i32(p + 1), i32(ex))
    oBc = oBc[:, :oh]
    # full path: the image's own compaction on the physical PRE table, exact count
    tab = torch.as_tensor(pre.astype(np.int32), device=dev).contiguous()
    nx = i32(ex).clone()
    bm._compact_dropped_selection(tab, nx)
    bm._mask_page_table_after_nsa_len(tab, nx)
    tabBf, nxBf = tab.cpu().numpy().astype(np.int64), nx.cpu().numpy().astype(np.int64)
    oBf, _ = kern.run(q, cache, i32(remap(tabBf)), i32(p + 1), i32(nxBf))
    oBf = oBf[:, :oh]
    # compaction control: PRE with the served count reproduces the served POST table and nsa
    tab2 = torch.as_tensor(pre.astype(np.int32), device=dev).contiguous()
    nc = i32(np.minimum(L, Wd)).clone()
    bm._compact_dropped_selection(tab2, nc)
    bm._mask_page_table_after_nsa_len(tab2, nc)
    tab2n, nc_n = tab2.cpu().numpy().astype(np.int64), nc.cpu().numpy().astype(np.int64)
    comp_ctrl = (tab2n == post).all(axis=1) & (nc_n == nsa_s)
    bf_table_same = np.array([(tabBf[i, :nxBf[i]] == post[i, :nxBf[i]]).all() for i in range(nv)])
    # positive controls.  P1 (required, instrument): halve the consumed count (nsa -> ceil(nsa / 2)); the row's
    # output must change for every row with nsa >= 2.  P2 (informational, level-3 analogue): one live slot of
    # another captured row appended at column nsa; a low-scoring foreign key can be invisible after the bf16
    # output rounding, so P2's detected fraction is reported as a sensitivity figure, not a pass criterion.
    sel_np = remap(post)
    p1_rows = [i for i in range(nv) if nsa_s[i] >= 2]
    p1_hit = []
    if p1_rows:
        nh = nsa_s.copy()
        nh[p1_rows] = (nsa_s[p1_rows] + 1) // 2
        oH, _ = kern.run(q, cache, sel_A, i32(clen_s), i32(nh))
        dH = ~row_eq(torch, oH[:, :oh], oA)
        p1_hit = [bool(dH[i]) for i in p1_rows]
    plant_rows, plant_hit = [], []
    selP = sel_np.copy()
    nP, cP = nsa_s.copy(), clen_s.copy()
    for i in range(nv):
        if nsa_s[i] >= Wd:
            continue
        own = set(sel_np[i][: nsa_s[i]].tolist())
        cand = [int(v) for v in np.unique(sel_np[sel_np >= 0]) if int(v) not in own and int(v) != 0]
        if not cand:
            continue
        selP[i, nsa_s[i]] = cand[len(cand) // 2]
        nP[i] += 1
        cP[i] += 1
        plant_rows.append(i)
    if plant_rows:
        oP, _ = kern.run(q, cache, i32(selP), i32(cP), i32(nP))
        oP = oP[:, :oh]
        diff_P = ~row_eq(torch, oP, oA)
        plant_hit = [bool(diff_P[i]) for i in plant_rows]
    eqBc, eqBf = row_eq(torch, oBc, oA), row_eq(torch, oBf, oA)
    magBc, relBc = row_mag(torch, oBc, oA)
    magBf, relBf = row_mag(torch, oBf, oA)
    magA, relA = row_mag(torch, oA, served)
    nanA, nanS = nan_rows(torch, oA), nan_rows(torch, served)
    pub, priv = [], []
    for i in range(nv):
        r_ = int(L[i] - (p[i] + 1))
        clamped = bool(p[i] + 1 >= Wd)
        crossing = (not clamped) and math.ceil(min(L[i], Wd) / 64) > math.ceil((p[i] + 1) / 64)
        regime = "clamped" if clamped else ("boundary_crossing" if crossing else ("dense_r>=1" if r_ >= 1 else "dense_r0"))
        anchored = bool(repro[i]) and bool(det[i]) and not oob_row[i] and not post_missing_rows[i] and consistent
        pub.append(dict(row_slot=int(valid[i]), regime=regime, r=r_, anchored=anchored, reproduced=bool(repro[i]),
                        deterministic=bool(det[i]), oob=bool(oob_row[i]), post_covered=not bool(post_missing_rows[i]),
                        compaction_ctrl=bool(comp_ctrl[i]), bf_table_same_on_exact_cols=bool(bf_table_same[i]),
                        bc_equal=bool(eqBc[i]), bf_equal=bool(eqBf[i]),
                        bc_max_abs=float(magBc[i]), bc_rel_l2=float(relBc[i]), bf_max_abs=float(magBf[i]),
                        bf_rel_l2=float(relBf[i]), a_vs_served_max_abs=float(magA[i]), a_vs_served_rel_l2=float(relA[i]),
                        nan_A=bool(nanA[i]), nan_served=bool(nanS[i]),
                        ctrl_halve=None if i not in p1_rows else p1_hit[p1_rows.index(i)],
                        plant=None if i not in plant_rows else plant_hit[plant_rows.index(i)],
                        nsa_changed_bc=bool(min(p[i] + 1, Wd) != nsa_s[i]), nsa_changed_bf=bool(nxBf[i] != nsa_s[i])))
        priv.append(dict(row_slot=int(valid[i]), p=int(p[i]), L=int(L[i]), nsa_served=int(nsa_s[i]),
                         clen_served=int(clen_s[i]), nsa_bf=int(nxBf[i]), nsa_ctrl=int(nc_n[i])))
    info.update(records_consistent=consistent, compact_slots=int(len(order)), plants=len(plant_rows))
    return pub, priv, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", required=True)
    ap.add_argument("--caplog", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--priv", required=True)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--deadline-s", type=float, default=None)
    ap.add_argument("--layers", default="3,23")
    args = ap.parse_args()
    if not args.dry and any(os.environ.get(k) for k in ("P3B_DRY_MIN_ROWS_1", "P3B_DRY_ATTN_LENSENSITIVE")):
        print("REFUSED: dry-test switches set outside --dry")
        return 2
    os.umask(0o077)
    import torch
    dl = rc.Deadline(args.deadline_s)
    doc = dict(schema=SCHEMA, mode="dry" if args.dry else "gpu", status="running", started=time.time(), layers={},
               min_rows=MIN_ROWS, errors=[])
    rc.write_json(args.out, doc)
    try:
        cl = rc.load_caplog(args.caplog)
        if cl["init"] is None:
            raise RuntimeError("no rank-0 init record in the capture log")
        dev = torch.device("cpu" if args.dry else "cuda:0")
        if not args.dry:
            doc["device"] = torch.cuda.get_device_name(dev)
        priv_rows = {}
        for L in [int(v) for v in args.layers.split(",")]:
            ai = cl["init"]["att_info"].get(str(L)) or cl["init"]["att_info"].get(L)
            ld = doc["layers"][str(L)] = dict(entries=[], rows=[], kernel={})
            if ai is None:
                ld["error"] = "layer not in capture init"
                continue
            kern = Kernel(torch, args.dry, ai, dev, doc)
            ld["kernel"] = kern.info
            qh = next((s["q"][0][1] for k, s in cl["sites"].items() if k[0] == "att_in" and k[1] == L), None)
            oh = next((s["out"][0][1] for k, s in cl["sites"].items() if k[0] == "att_out" and k[1] == L), None)
            for seq, path in rc.list_entries(args.cap, "att", L):
                if dl.passed():
                    doc["stopped_reason"] = "soft deadline"
                    break
                d = rc.load_npz(path, cl, "att", L, seq)
                try:
                    pub, priv, info = replay_entry(torch, kern, ai, d, qh, oh, dev, int(ai["topk_tokens"]))
                except Exception as exc:  # noqa: BLE001
                    doc["errors"].append(dict(layer=L, seq=seq, error=repr(exc)[:300], tb=traceback.format_exc()[-800:]))
                    continue
                for r_ in pub:
                    r_["seq"] = seq
                ld["rows"] += pub
                ld["entries"].append(dict(seq=seq, **info))
                priv_rows.setdefault(str(L), []).append(dict(seq=seq, rows=priv))
            ld["summary"] = summarize_layer(ld["rows"])
            rc.write_json(args.out, doc)
        os.makedirs(args.priv, mode=0o700, exist_ok=True)
        rc.write_json(os.path.join(args.priv, "attn_rows_private.json"), priv_rows)
        doc["status"] = "partial" if doc.get("stopped_reason") else ("error" if doc["errors"] else "complete")
    except Exception as exc:  # noqa: BLE001
        doc["status"] = "error"
        doc["errors"].append(dict(error=repr(exc)[:400], tb=traceback.format_exc()[-1500:]))
    doc["finished"] = time.time()
    rc.write_json(args.out, doc)
    print(json.dumps(dict(part="attn", status=doc["status"], layers={k: v.get("summary", {}).get("verdict")
                                                                       for k, v in doc["layers"].items()})))
    return 0 if doc["status"] == "complete" else (4 if doc["status"] == "partial" else 5)


def summarize_layer(rows: list) -> dict:
    out = dict(rows=len(rows), by_regime={})
    halved = [r for r in rows if r["ctrl_halve"] is not None]
    out["ctrl_halve_detected"] = sum(1 for r in halved if r["ctrl_halve"])
    out["ctrl_halve_rows"] = len(halved)
    planted = [r for r in rows if r["plant"] is not None]
    out["plant_detected"] = sum(1 for r in planted if r["plant"])
    out["plant_rows"] = len(planted)
    out["plant_detected_frac"] = round(out["plant_detected"] / len(planted), 4) if planted else None
    out["not_anchored"] = sum(1 for r in rows if not r["anchored"])
    out["not_reproduced"] = sum(1 for r in rows if not r["reproduced"])
    out["compaction_ctrl_fail"] = sum(1 for r in rows if not r["compaction_ctrl"])
    verdicts = {}
    for reg in ("dense_r>=1", "dense_r0", "clamped", "boundary_crossing"):
        rr = [r for r in rows if r["regime"] == reg and r["anchored"]]
        n_all = sum(1 for r in rows if r["regime"] == reg)
        bc_diff = [r for r in rr if not r["bc_equal"]]
        bf_diff = [r for r in rr if not r["bf_equal"]]
        e = dict(rows=n_all, anchored=len(rr), bc_differ=len(bc_diff), bf_differ=len(bf_diff),
                 bc_max_abs=max([r["bc_max_abs"] for r in rr], default=0.0),
                 bf_max_abs=max([r["bf_max_abs"] for r in rr], default=0.0),
                 bc_rel_l2_median=rc.median([r["bc_rel_l2"] for r in bc_diff]),
                 bf_rel_l2_median=rc.median([r["bf_rel_l2"] for r in bf_diff]),
                 r_hist={str(k): sum(1 for r in rr if r["r"] == k) for k in sorted({r["r"] for r in rr})},
                 nsa_changed_bc=sum(1 for r in rr if r["nsa_changed_bc"]))
        need = MIN_ROWS.get(reg)
        if halved and out["ctrl_halve_detected"] < len(halved):
            v = "INCOMPLETE (positive control P1 missed)"
        elif need is not None and len(rr) < need:
            v = f"INCOMPLETE ({len(rr)} anchored rows < {need})"
        elif bc_diff or bf_diff:
            v = "DIFFERENCES (characterize)"
        elif need is None:
            v = "informational (control rows)" if rr else "no rows"
        else:
            v = "EQUAL ON CAPTURED ROWS"
        e["verdict"] = v
        out["by_regime"][reg] = e
        verdicts[reg] = v
    out["verdict"] = verdicts
    return out


if __name__ == "__main__":
    sys.exit(main())
