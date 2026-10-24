#!/usr/bin/env python3
"""CPU harness test, run INSIDE the v0.9.1 image (torch available; CUDA hidden).  No served kernel is called here:
the R1 / R2 / R3 harness code (scripts/r1_topk.py, scripts/r2_mla.py) is driven with CPU stand-ins whose behaviour is
KNOWN, to show the harness records what each behaviour produces.  The served-call side is tested by
test_served_args.py and the dry runs.

usage: test_harness_cpu.py OUTDIR
Writes OUTDIR/<case>/v091|v070/{env,r1,r2}.json run directories for tests/test_analysis.py, which checks that the
analysis reads each case as pre-registered.  Exit 1 if a harness-level expectation fails here.

R1 stand-ins (same signature as torch.ops._C.top_k_per_row_prefill):
  sorted     exact top-k, logit-descending, deterministic                         -> REPEATABLE
  atomic     upstream layout: above-threshold block in a fresh random order per call, threshold block (16) sorted
             descending last                                                      -> A-ORDER, set and last col stable
  setvary    atomic + the last column alternates with the 513th pool                  -> A-SET
  ptrdep     order permuted by a seed taken from the output buffer's address          -> B-ALLOC
  nowindow   ignores cu_seqlen_ke (reads the whole row)                            -> INCOMPLETE (semantics check)
  badshort   rows with <= 512 pools returned reversed                              -> INCOMPLETE (negative control)
R2 stand-in kernels (same interface as V091 / V070 in r2_mla.py; tiny output [rows, 2, 8]):
  order      bf16 running sum over the index row in the given order               -> ORDER-SENSITIVE
  sym        integer sum of the index row (order-free)                            -> ORDER-INSENSITIVE
  noise      order + a random term per call                                       -> A-EXEC
  ptr        sym + a term from the cache tensor's address                         -> B-ALLOC
"""
from __future__ import annotations

import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/s")

import torch  # noqa: E402
import rt_common as rc  # noqa: E402
import r1_topk  # noqa: E402
import r2_mla  # noqa: E402

OUT = sys.argv[1]
DEV = torch.device("cpu")
FAILS = []
RNG = torch.Generator().manual_seed(99)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} harness {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def make_op(mode):
    state = dict(n=0)

    def op(logits, ks, ke, out, M, s0, s1, K):
        state["n"] += 1
        L = torch.as_strided(logits, (M, logits.shape[1]), (s0, s1))
        out.fill_(-1)
        for_rows = torch.arange(M)
        plen = (ke - ks)
        cols = torch.arange(L.shape[1])[None, :]
        inwin = (cols >= ks[:, None]) & (cols < (ke[:, None] if mode != "nowindow" else L.shape[1]))
        v = torch.where(inwin, L, torch.tensor(float("-inf")))
        top = torch.topk(v, K + 1, dim=1)
        idx, vals = top.indices, top.values
        short = plen <= K
        ar = torch.arange(K)[None, :]
        short_rows = torch.where(ar < plen[:, None], ar.expand(M, -1), torch.full((1, K), -1))
        if mode == "badshort":
            short_rows = torch.where(short_rows >= 0, (plen[:, None] - 1 - ar), short_rows)
        sel = idx[:, :K].clone()
        if mode in ("atomic", "setvary", "ptrdep"):
            nab = K - 16
            if mode == "ptrdep":
                g = torch.Generator().manual_seed(int(out.data_ptr()) % 100003)
            else:
                g = RNG
            perm = torch.argsort(torch.rand((M, nab), generator=g), dim=1)
            sel[:, :nab] = torch.gather(sel[:, :nab], 1, perm)
            if mode == "setvary" and state["n"] % 2 == 0:
                sel[:, K - 1] = idx[:, K]
        res = torch.where(short[:, None], short_rows, sel - ks[:, None])
        out.copy_(res.to(torch.int32))
        del for_rows, vals
    return op


def r1_doc(mode, n_calls, regimes=("K", "C", "T")):
    ks, ke = rc.formula_window(torch, DEV)
    doc = dict(part="r1", status="complete", label="SYNTHETIC", regimes={}, errors=[],
               window=dict(source="formula (cpu test)", equals_formula=True), stride_source="served producer output (K)")
    kept = None
    g = torch.Generator().manual_seed(5)
    base = torch.randn((rc.CHUNK_ROWS, rc.N_POOLS), generator=g)
    for name in regimes:
        lg = base.clone() if name != "T" else base.to(torch.bfloat16).float()
        r, k = r1_topk.run_regime(torch, DEV, name, lg, ks, ke, n_calls, 2, rc.Deadline(None),
                                  keep_outputs=(name == "K"), op=make_op(mode))
        doc["regimes"][name] = r
        if name == "K":
            kept = dict(regime="K", logits=lg, ks=ks, ke=ke, outputs=k)
    return doc, kept


class FakeKern:
    def __init__(self, mode, version):
        self.mode, self.version = mode, version
        self.info = dict(path=f"CPU stand-in ({mode})", prefill_mg_enabled=True)
        g = torch.arange(rc.CHUNK_START + 1, rc.SEQ_LEN + 1, dtype=torch.int32)
        self.per_token_local = (g + 1) // 2

    def build_cache(self, syn, poison, perm_seed):
        kv = torch.full((8, 8), poison, dtype=torch.uint8)
        return dict(kv=kv, bt=None, slots=torch.arange(0, rc.SEQ_LEN, 2), own=None)

    def build_index(self, syn, pool_topk, cache):
        n = rc.CHUNK_ROWS
        pt = pool_topk.long()[:, : rc.SELECT_K - 1]
        pl = ((syn.pos + 1) // 4).long()
        hist = (pt[:, :, None] * 4 + torch.arange(4)[None, None, :]).reshape(n, -1)
        hist = torch.where(((pt >= 0) & (pt < pl[:, None])).repeat_interleave(4, dim=1), hist, torch.full_like(hist, -1))
        tail = torch.arange(3)[None, :]
        tc = ((syn.pos + 1) % 4).long()[:, None]
        tl = torch.where(tail < tc, pl[:, None] * 4 + tail, torch.full_like(tail, -1))
        topk = torch.full((n, rc.W), -1, dtype=torch.int64)
        topk[:, : rc.EXPANDED_COLS] = torch.cat([hist, tl], dim=1)
        if self.version == "v091":
            sel = topk.to(torch.int32)
            nsa = (syn.pos + 1).clamp_max(rc.W).to(torch.int32)
        else:
            sel = torch.full((n, rc.W), -1, dtype=torch.int32)
            nsa = torch.zeros(n, dtype=torch.int32)
            for i in range(n):
                row = topk[i]
                own = row[(row >= 0) & (row % 2 == 0)] // 2
                sel[i, : len(own)] = own.to(torch.int32)
                nsa[i] = len(own)
        return dict(sel=sel, nsa=nsa, topk=topk.to(torch.int32), expanded=None, keep=())

    def make_q(self, syn):
        return syn.q_lat[:, :2, :8].contiguous(), None

    def run(self, q, cache, idx, scratch=None):
        sel = idx["sel"].float()
        valid = sel >= 0
        if self.mode in ("order", "noise"):
            acc = torch.zeros(sel.shape[0], dtype=torch.bfloat16)
            w = torch.where(valid, torch.sin(sel * 0.37) * 3.1, torch.zeros_like(sel)).to(torch.bfloat16)
            for c in range(0, sel.shape[1], 64):
                acc = acc + w[:, c:c + 64].float().sum(dim=1).to(torch.bfloat16)
            s = acc.float()
            if self.mode == "noise":
                s = s + torch.rand(s.shape, generator=RNG) * 1e-2
        else:
            si = idx["sel"].long()
            h = torch.where(si >= 0, (si * 2654435761) % 1000003, torch.zeros_like(si)).sum(dim=1) % 100000
            s = h.float() * 1e-3                       # exact integer hash of the SET: order-free
            if self.mode == "ptr":
                s = s + (int(cache["kv"].data_ptr()) % 977) * 1e-3
        out = q.float() * (1.0 + s[:, None, None])     # fp32 output: the stand-in's sensitivity is not the question
        lse = s[:, None].expand(-1, 2).contiguous()
        return out, lse, (scratch if scratch is not None else torch.empty(16, dtype=torch.uint8))

    def ref_rows(self, syn, cache, idx, rows):
        return syn.q_lat[rows][:, :2, :8].float()

    def decode_check(self, syn, cache):
        return 0.03


def r2_doc(mode, version, kept=None):
    cfg = dict(q_regimes=[("flat", 1.0), ("peaked", 4.0)], r2_warm=1, r2_repeats=5, r2_reloc=2, ref_rows=8,
               r3_calls=3, seed=20261024)
    kern = FakeKern(mode, version)
    doc = dict(part="r2", version=version, status="running", label="SYNTHETIC", regimes={}, errors=[], kernel=kern.info)
    for qname, qstd in cfg["q_regimes"]:
        doc["regimes"][qname] = r2_mla.run_q_regime(torch, DEV, kern, version, qname, qstd, cfg, rc.Deadline(None), print)
    if kept is not None:
        doc["r3"] = r2_mla.run_r3(torch, DEV, kern, cfg, kept, print)
    doc["status"] = "complete"
    return doc


def write_case(case, ver, r1, r2):
    d = os.path.join(OUT, case, ver)
    os.makedirs(d, exist_ok=True)
    rc.write_json(os.path.join(d, "env.json"), dict(version=ver, gpu="CPU stand-in", device="cpu"))
    if r1 is not None:
        rc.write_json(os.path.join(d, "r1.json"), r1)
    if r2 is not None:
        rc.write_json(os.path.join(d, "r2.json"), r2)


def main():
    os.makedirs(OUT, exist_ok=True)
    # R1 stand-ins (v091 regimes K, C, T) and R2 stand-ins; each case is one run directory
    r1 = {}
    keptK = {}
    for mode in ("sorted", "atomic", "setvary", "ptrdep", "nowindow", "badshort"):
        r1[mode], keptK[mode] = r1_doc(mode, 20)
        k = r1[mode]["regimes"]["K"]
        print(f"INFO r1 {mode}: same-alloc order-varied {k['same_alloc']['rows_order_varied']} set-varied "
              f"{k['same_alloc']['rows_set_varied']}; relocated order-varied {k['relocated']['rows_order_varied']}; "
              f"refset agree {k['refset_agree_untied']}/{k['refset_sample_rows']}")
    a = r1["atomic"]["regimes"]["C"]
    check("R1 atomic: order varies in same allocation", a["same_alloc"]["rows_order_varied_selecting"] > 2000)
    check("R1 atomic: set and last column stable", a["same_alloc"]["rows_set_varied"] == 0
          and a["same_alloc"]["rows_last_col_varied"] == 0)
    check("R1 atomic: shortcut rows never vary", a["same_alloc"]["rows_order_varied_shortcut"] == 0)
    s = r1["sorted"]["regimes"]["C"]
    check("R1 sorted: no variation anywhere", s["same_alloc"]["rows_order_varied"] == 0 and s["relocated"]["rows_order_varied"] == 0)
    check("R1 sorted: pattern fully descending", s["pattern_fully_desc"] == s["rows_selecting"])
    p = r1["ptrdep"]["regimes"]["C"]
    check("R1 ptrdep: same allocation repeatable, relocation varies",
          p["same_alloc"]["rows_order_varied"] == 0 and p["relocated"]["rows_order_varied"] > 0)
    check("R1 relocation achieved", p["relocated"]["calls_relocated_from_ref"] >= 10)
    check("R1 positive control", all(r1["sorted"]["regimes"][x]["pos_ctrl"]["order_flagged"] and
                                     r1["sorted"]["regimes"][x]["pos_ctrl"]["set_flagged"] for x in ("K", "C", "T")))
    check("R1 nowindow: semantics check fails", r1["nowindow"]["regimes"]["C"]["refset_agree_untied"]
          < r1["nowindow"]["regimes"]["C"]["refset_sample_rows"])
    check("R1 T regime has ties at the threshold", r1["sorted"]["regimes"]["T"].get("rows_tied_at_threshold", 0) > 0)
    # R2 stand-ins, both index paths
    r2 = {}
    for mode in ("order", "sym", "noise", "ptr"):
        for ver in ("v091", "v070"):
            r2[(mode, ver)] = r2_doc(mode, ver, kept=keptK["atomic"] if (ver == "v091" and mode == "order") else None)
    o = r2[("order", "v091")]["regimes"]["flat"]
    check("R2 order/v091: variants change selecting rows only",
          all(v["rows_differ_selecting"] > 0 and v["rows_differ_shortcut"] == 0 for v in o["variants"].values()))
    check("R2 order/v091: same set / nsa per variant", all(v["same_set"] and v["same_nsa"] for v in o["variants"].values()))
    check("R2 order/v091: repeat equal", o["repeat_same_alloc"]["all_out_equal"])
    check("R2 positive control detected", o["pos_ctrl_set_change"]["rows_differ_selecting"] >= 0.9 * o["pos_ctrl_set_change"]["selecting_rows"])
    sy = r2[("sym", "v070")]["regimes"]["flat"]
    check("R2 sym/v070: no variant changes output", all(v["rows_differ"] == 0 for v in sy["variants"].values()))
    r3 = r2[("order", "v091")]["r3"]
    check("R3 composed: outputs differ only where top-k order differs",
          any(c["rows_out_diff"] for c in r3["calls"]) and all(c["rows_out_diff_without_topk_diff"] == 0 for c in r3["calls"]))
    # the kernel-output normaliser: token-major passes through, head-major is transposed, rows kept
    info = {}
    tm = torch.arange(5 * 3 * 4, dtype=torch.float32).reshape(5, 3, 4)
    o1, _ = r2_mla._token_major(torch, tm, None, 5, info)
    o2, _ = r2_mla._token_major(torch, tm.permute(1, 0, 2).contiguous(), None, 5, {})
    o3, _ = r2_mla._token_major(torch, torch.cat([tm, tm]), None, 5, {})
    check("R2 output normaliser", torch.equal(o1, tm) and torch.equal(o2, tm) and torch.equal(o3, tm)
          and info.get("out_shape") == [5, 3, 4])
    # run directories for the analysis test
    write_case("h-order", "v091", r1["atomic"], r2[("order", "v091")])
    write_case("h-order", "v070", r1_doc("atomic", 20, regimes=("C", "T"))[0], r2[("order", "v070")])
    write_case("h-sym", "v091", r1["sorted"], r2[("sym", "v091")])
    write_case("h-sym", "v070", r1_doc("sorted", 20, regimes=("C", "T"))[0], r2[("sym", "v070")])
    write_case("h-noise", "v091", r1["setvary"], r2[("noise", "v091")])
    write_case("h-noise", "v070", None, r2[("noise", "v070")])
    write_case("h-ptr", "v091", r1["ptrdep"], r2[("ptr", "v091")])
    write_case("h-ptr", "v070", None, r2[("ptr", "v070")])
    write_case("h-bad", "v091", r1["nowindow"], None)
    write_case("h-bad", "v070", r1_doc("badshort", 20, regimes=("C", "T"))[0], None)
    print(f"harness: {len(FAILS)} failure(s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
