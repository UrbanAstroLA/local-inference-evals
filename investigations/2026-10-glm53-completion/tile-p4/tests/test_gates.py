#!/usr/bin/env python3
"""CPU tests of runner/gates_p4.py on fabricated bind logs (the recorder's record format).  Host python3.

The fabricated arm-A launches copy captured-p3b's REAL recorded launch fields (cap_p370.jsonl) for every layer, so
G3 runs against the real reference format.  Each planted defect must fail exactly its gate."""
import copy
import json
import os
import shutil
import subprocess
import sys

T = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(T)
GATES = os.path.join(ROOT, "runner", "gates_p4.py")
P3B = "<runs>/captured-p3b/runs/capture-attempt-2/caplog/cap_p370.jsonl"
OUT = os.path.join(T, "out", "gates-scratch")
RES = []


def check(name, ok, detail=""):
    ok = bool(ok)
    RES.append(ok)
    print(("PASS " if ok else "FAIL ") + name + (" " + detail if detail else ""), flush=True)


def p3b_launch():
    dec, pre = {}, None
    for line in open(P3B):
        r = json.loads(line)
        if r.get("kind") == "site" and isinstance(r.get("key"), list) and r["key"][0] == "moe_bind" and r["layer"] == 3:
            if r["plan"] == "decode":
                dec[r["m"]] = r["launch"]
            elif r["m"] == 64:
                pre = r["launch"]
    return dec, pre


def make(arm, mutate=None):
    dec, pre = p3b_launch()
    recs = []
    for rank in (0, 1):
        for L in range(3, 45):
            for m in (1, 2, 4, 8, 16, 24, 32):
                la = copy.deepcopy(dec[m])
                if arm == "B" and 10 <= m <= 32:
                    la.update(fc1_tile_k=64, fc1_tile_n=256, fc2_tile_k=64, fc2_tile_n=256)
                recs.append(dict(kind="bind", rank=rank, layer=L, plan="decode", m=m, draft=False, changed=False,
                                 launch=la, plan_kind="decode", capacity=32,
                                 weight_tile_config="(64, 256, 64, 256)" if arm == "B" else "None"))
            for m in (40, 48, 56, 64, 873, 1024):
                recs.append(dict(kind="bind", rank=rank, layer=L, plan="prefill", m=m, draft=False, changed=False,
                                 launch=copy.deepcopy(pre), plan_kind="prefill", capacity=1024,
                                 weight_tile_config="(64, 256, 64, 256)" if arm == "B" else "None"))
    recs.append(dict(kind="bind", rank=0, layer=-1, plan="decode", m=8, draft=True, changed=False, launch=dec[8]))
    if mutate:
        mutate(recs)
    return recs


def write(name, recs):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    for rank in (0, 1):
        with open(os.path.join(d, f"bind_r{rank}_p{100 + rank}.jsonl"), "w") as f:
            for r in recs:
                if r["rank"] == rank:
                    f.write(json.dumps(r) + "\n")
    return d


def gates(arm, binds, ref=None, p3b=None):
    cmd = [sys.executable, GATES, "--arm", arm, "--binds", binds, "--out", binds + ".json"]
    if ref:
        cmd += ["--ref-a", ref]
    if p3b:
        cmd += ["--p3b", p3b]
    rc = subprocess.run(cmd, capture_output=True, text=True).returncode
    return rc, json.load(open(binds + ".json"))


def find(recs, rank, L, plan, m):
    return next(r for r in recs if r["rank"] == rank and r["layer"] == L and r["plan"] == plan and r["m"] == m)


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    A = write("A", make("A"))
    rc, d = gates("A", A, p3b=P3B)
    check("G3 PASS: arm A built from captured-p3b's real launches (all layers, both ranks)", rc == 0, str(d["gates"]["G3"].get("n_mismatches")))

    def m1(recs):
        find(recs, 1, 17, "decode", 24)["launch"]["fc1_tile_k"] = 64
    rc, d = gates("A", write("A-rule", make("A", m1)), p3b=P3B)
    check("G3 FAIL: stock-rule violation on rank 1 layer 17 m 24", rc == 1 and d["gates"]["G3"]["n_stock_rule_violations"] == 1)

    def m2(recs):
        find(recs, 0, 23, "decode", 16)["launch"]["moe_block_size"] = 16
    rc, d = gates("A", write("A-p3b", make("A", m2)), p3b=P3B)
    check("G3 FAIL: rank-0 layer-23 launch differs from captured-p3b", rc == 1 and d["gates"]["G3"]["n_mismatches"] >= 1)

    def m3(recs):
        recs.remove(find(recs, 0, 44, "prefill", 48))
    rc, d = gates("A", write("A-miss", make("A", m3)), p3b=P3B)
    check("G3 FAIL: a capture-size key missing", rc == 1 and d["gates"]["G3"]["capture_size_keys_missing"])

    B = write("B", make("B"))
    rc, d = gates("B", B, ref=A)
    check("G1 + G2 PASS: arm B forces K64/N256 at m 10-32 everywhere, all else equal to A",
          rc == 0 and d["gates"]["G1"]["ok"] and d["gates"]["G2"]["ok"], json.dumps(d["gates"]["G2"]["weight_tile_config_pairs"]))

    def b1(recs):
        find(recs, 1, 30, "decode", 32)["launch"].update(fc1_tile_k=128, fc1_tile_n=128)
    rc, d = gates("B", write("B-g1", make("B", b1)), ref=A)
    check("G1 FAIL: one decode launch (rank 1, layer 30, m 32) not forced", rc == 1 and not d["gates"]["G1"]["ok"]
          and d["gates"]["G2"]["ok"])

    def b2(recs):
        find(recs, 0, 9, "prefill", 873)["launch"]["moe_block_size"] = 64
    rc, d = gates("B", write("B-g2", make("B", b2)), ref=A)
    check("G2 FAIL: a prefill launch of B differs from A (block size)", rc == 1 and d["gates"]["G1"]["ok"]
          and not d["gates"]["G2"]["ok"])

    def b3(recs):
        recs.remove(find(recs, 1, 44, "decode", 16))
    rc, d = gates("B", write("B-cov", make("B", b3)), ref=A)
    check("G1 FAIL: coverage (rank 1 layer 44 m 16 never bound)", rc == 1 and d["gates"]["G1"]["n_coverage_missing"] == 1)

    def b4(recs):
        r = copy.deepcopy(find(recs, 0, 5, "decode", 8))
        r["changed"] = True
        recs.append(r)
    rc, d = gates("B", write("B-chg", make("B", b4)), ref=A)
    check("G2 FAIL: a launch that changed during the session", rc == 1 and d["gates"]["G2"]["changed"])

    def b5(recs):
        recs.append(dict(kind="bind", rank=0, layer=6, plan="decode", m=6, draft=False, changed=False,
                         launch=copy.deepcopy(find(recs, 0, 6, "decode", 8)["launch"]), capacity=32))
    rc, d = gates("B", write("B-abs", make("B", b5)), ref=A)
    check("G2 PASS with a key absent from A judged by A's rule (decode m 6, K64/N256)", rc == 0 and
          d["gates"]["G2"]["n_absent_from_A"] == 1)

    rc, d = gates("A2", write("A2", make("A")), ref=A)
    check("A2 control PASS: identical launches", rc == 0)

    def c1(recs):
        find(recs, 0, 3, "decode", 32)["launch"]["max_m_blocks"] = 128
    rc, d = gates("A2", write("A2-bad", make("A", c1)), ref=A)
    check("A2 control FAIL: one launch differs from session A", rc == 1)
    rc, d = gates("B", B)
    check("G2 refuses without session A's binds", rc == 1 and "error" in d["gates"]["G2"])
    print("test_gates: %d/%d" % (sum(RES), len(RES)))
    return 0 if all(RES) else 1


if __name__ == "__main__":
    sys.exit(main())
