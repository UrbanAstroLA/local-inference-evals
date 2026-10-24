#!/usr/bin/env python3
"""Port of the validated local-forcewin module (B1 image, hook-validation attempt 2, sha256 15e93e23) to stock
0.9.1 for P4.  Host python3, no torch.  Every change is an exact, once-only anchored replacement, listed in PORT
below with its reason; the result is written to build/files/vllm/forcewin_hooks.py and the unified diff to
build/forcewin-port.diff.

  python3 port_forcewin.py            write the ported module and the diff
  python3 port_forcewin.py --check    regenerate in memory; require the files on disk to be identical

What does NOT change: every function the forcing and the per-position metrics rely on (row_plan, forced_override,
row_metrics, post_probs, pack_metrics, on_logits, on_processed, override, override_single, _commit_rows,
after_commit, after_propose, add_request / remove_request bookkeeping) keeps its validated statements; the port
only adds calls into vllm.p4_hooks (each guarded and exception-isolated), the "list" draft mode, and disables the
three B1-only features (D2 injection, shadow ledger, snapshots), whose call sites or assumptions do not exist on stock.
"""
import difflib
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "<runs>/hook-validation/hook/forcewin_hooks.py"
SRC_SHA = "15e93e236067567ffb9db7116986e8b732ef679e672c5b12b9d78f5884899d83"
OUT = os.path.join(HERE, "files", "vllm", "forcewin_hooks.py")
DIFF = os.path.join(HERE, "forcewin-port.diff")
MARK = "local-p4"

PORT = [
    # P1: header note (documentation only)
    ('"""local-forcewin: forced-history decode window',
     '"""[local-p4 PORT to stock 0.9.1 for tile-p4/: B1-only D2 injection, shadow ledger and snapshots are disabled\n'
     '(their call sites and B1 tail-ring assumptions do not exist on stock); adds the "list" draft mode (a fixed\n'
     'per-step accepted-count list, key local_fw_sched) and guarded calls into vllm.p4_hooks.  See\n'
     'tile-p4/build/port_forcewin.py and forcewin-port.diff.]\n\n'
     'local-forcewin: forced-history decode window'),
    # P2: B1-only features off whatever the env says (stock has no persistent tail-ring copy to inject into, and the
    #     shadow ledger / KDA snapshots model B1's corrected tail ring and B1's mamba metadata)
    ('D2_ALLOWED = ON and os.environ.get("FORCEWIN_D2", "") == "1"\n'
     'SHADOW = ON and os.environ.get("FORCEWIN_SHADOW", "") == "1"\n'
     'SNAP = ON and os.environ.get("FORCEWIN_SNAP", "") == "1"\n',
     'D2_ALLOWED = False   # local-p4: B1-only (stock has the D2 mapping natively; no injection site)\n'
     'SHADOW = False       # local-p4: B1-only (ledger models B1\'s corrected tail ring)\n'
     'SNAP = False         # local-p4: B1-only (validation snapshots; not needed for P4)\n'
     'B1_ONLY_ENV_SET = [k for k in ("FORCEWIN_D2", "FORCEWIN_SHADOW", "FORCEWIN_SNAP")\n'
     '                   if os.environ.get(k, "")]  # local-p4: logged as an error at init if any is set\n'
     'SCHED_MAX = int(os.environ.get("FORCEWIN_SCHED_MAX", "1024"))  # local-p4: "list" mode length\n'
     'try:  # local-p4\n'
     '    from vllm import p4_hooks as _p4b  # local-p4\n'
     '    if not _p4b.ENABLED:  # local-p4\n'
     '        _p4b = None  # local-p4\n'
     'except Exception:  # noqa: BLE001  local-p4\n'
     '    _p4b = None  # local-p4\n'),
    # P3: the list draft mode
    ('DRAFT_MODES = {"natural": 0, "oracle": 1, "poison": 2, "reject": 3, "sched": 4}\n',
     'DRAFT_MODES = {"natural": 0, "oracle": 1, "poison": 2, "reject": 3, "sched": 4, "list": 5}  # local-p4: list\n'),
    ('def draft_rows(mode, arg, forced, forced_len, n_out, K, step_ctr, vocab):\n',
     'def draft_rows(mode, arg, forced, forced_len, n_out, K, step_ctr, vocab, sched_tab=None, sched_len=None):\n'),
    ('    reject:<j>: the first j-1 equal, draft j does not (a = j-1); sched:<seed>: a_s pseudo-random in 0..K per step.\n'
     '    """\n',
     '    reject:<j>: the first j-1 equal, draft j does not (a = j-1); sched:<seed>: a_s pseudo-random in 0..K per step.\n'
     '    list (local-p4): a_s = sched_tab[r, step_ctr % sched_len[r]], the request\'s fixed accepted-count list.\n'
     '    """\n'),
    ('    a_sched = torch.remainder(step_ctr * 1103515245 + arg * 12345 + 7, K + 1)\n',
     '    a_sched = torch.remainder(step_ctr * 1103515245 + arg * 12345 + 7, K + 1)\n'
     '    if sched_tab is not None:  # local-p4: "list" mode (mode 5)\n'
     '        sl = sched_len.clamp(min=1)  # local-p4\n'
     '        a_list = sched_tab.gather(1, torch.remainder(step_ctr, sl)[:, None])[:, 0].clamp(0, K)  # local-p4\n'
     '        a_sched = torch.where(mode == 5, a_list, a_sched)  # local-p4\n'),
    # P4: persistent list table, allocated with the other per-request tables (before graph capture)
    ('    S.nstep = torch.zeros(R, dtype=torch.int64, device=dev)      # forced steps per request (sched mode)\n',
     '    S.nstep = torch.zeros(R, dtype=torch.int64, device=dev)      # forced steps per request (sched mode)\n'
     '    S.schedtab = torch.zeros(R, SCHED_MAX, dtype=torch.int64, device=dev)  # local-p4: list mode\n'
     '    S.schedlen = torch.ones(R, dtype=torch.int64, device=dev)  # local-p4\n'),
    ('          "shadow": SHADOW, "snap": SNAP, "cols": COLS, "xcols": XCOLS})\n',
     '          "shadow": SHADOW, "snap": SNAP, "cols": COLS, "xcols": XCOLS,\n'
     '          "port": "local-p4 stock 0.9.1", "p4_bridge": _p4b is not None, "sched_max": SCHED_MAX})  # local-p4\n'
     '    if B1_ONLY_ENV_SET:  # local-p4\n'
     '        _err("init", RuntimeError("B1-only forcewin features requested but not ported: %s" % B1_ONLY_ENV_SET))\n'),
    # P5: register -> P4 registration (same moment: capture_model, before capture)
    ('        _ensure(runner)\n        _log({"kind": "registered"})\n',
     '        _ensure(runner)\n        _log({"kind": "registered"})\n'
     '        if _p4b is not None:  # local-p4: stages and stores allocated before graph capture\n'
     '            _p4b.register(runner)  # local-p4\n'),
    # P6: add_request: list schedule table; P4 slot
    ('        S.mode[i] = r.mode\n        S.marg[i] = r.marg\n',
     '        S.mode[i] = r.mode\n        S.marg[i] = r.marg\n'
     '        if r.mode == 5:  # local-p4: "list" draft mode\n'
     '            sl = [int(x) for x in (_xarg(sp, "local_fw_sched") or [])][:SCHED_MAX]  # local-p4\n'
     '            if not sl:  # local-p4\n'
     '                raise ValueError("draft mode list without local_fw_sched")  # local-p4\n'
     '            S.schedtab[i, :len(sl)].copy_(torch.tensor(sl, dtype=torch.int64))  # local-p4\n'
     '            S.schedlen[i] = len(sl)  # local-p4\n'
     '        if _p4b is not None:  # local-p4\n'
     '            _p4b.on_add(i, r.tag, n)  # local-p4\n'),
    # P7: remove_request -> P4 flush (after the forcewin record is saved)
    ('        _log({"kind": "done", "req": req_id, "tag": r.tag, "errors": S.errors})\n        _flush_steplog()\n',
     '        _log({"kind": "done", "req": req_id, "tag": r.tag, "errors": S.errors})\n        _flush_steplog()\n'
     '        if _p4b is not None:  # local-p4: stores of this request -> PRIVATE npz; slot freed\n'
     '            _p4b.on_remove(i, r.tag, len(r.F))  # local-p4\n'),
    # P8: on_batch -> P4 host step log
    ('        S.step += 1\n        S.armed = True\n        S.fwd_armed = True\n',
     '        S.step += 1\n        S.armed = True\n        S.fwd_armed = True\n'
     '        if _p4b is not None:  # local-p4\n'
     '            _p4b.on_batch(input_batch, S.step)  # local-p4\n'),
    # P9: on_logits -> P4 token rows and logits staging (before the sampler)
    ('                    post_ok=torch.zeros(L, device=logits.device), L=L, nr=nr)\n',
     '                    post_ok=torch.zeros(L, device=logits.device), L=L, nr=nr)\n'
     '        if _p4b is not None:  # local-p4\n'
     '            _p4b.on_logits(S.sc, logits, input_batch)  # local-p4\n'),
    # P10: _commit_rows -> P4 stores (the same rows forcewin records)
    ('    S.nmeas.index_put_((tgt_req, tgt_o), torch.ones(L, dtype=torch.int32, device=packed.device),\n'
     '                       accumulate=True)\n\n\ndef override(',
     '    S.nmeas.index_put_((tgt_req, tgt_o), torch.ones(L, dtype=torch.int32, device=packed.device),\n'
     '                       accumulate=True)\n'
     '    if _p4b is not None:  # local-p4\n'
     '        _p4b.on_commit(sc, valid_flat)  # local-p4\n\n\ndef override('),
    # P11: override -> P4 step composition
    ('        sc["acc"], sc["valid"], sc["new_n"], sc["out_idx"] = acc, valid, new_n, out_idx\n',
     '        sc["acc"], sc["valid"], sc["new_n"], sc["out_idx"] = acc, valid, new_n, out_idx\n'
     '        if _p4b is not None:  # local-p4\n'
     '            _p4b.on_step(S.step, idx, out_idx[:, 0], n_rows, acc)  # local-p4\n'),
    # P12: override_single -> P4 step composition
    ('        sc["acc"] = torch.where(fwin[: sc["nr"]], torch.zeros_like(sc["idx"]), torch.full_like(sc["idx"], -1)) \\\n'
     '            if L == sc["nr"] else None\n',
     '        sc["acc"] = torch.where(fwin[: sc["nr"]], torch.zeros_like(sc["idx"]), torch.full_like(sc["idx"], -1)) \\\n'
     '            if L == sc["nr"] else None\n'
     '        if _p4b is not None and sc["acc"] is not None:  # local-p4\n'
     '            _p4b.on_step(S.step, sc["idx"], sc["o"][: sc["nr"]], n_rows, sc["acc"])  # local-p4\n'),
    # P13: after_propose passes the list table (mode 5); other modes unchanged
    ('        new = draft_rows(mode, S.marg[idx], S.forced[idx], S.flen[idx], nout, S.K, S.nstep[idx], S.vocab)\n',
     '        new = draft_rows(mode, S.marg[idx], S.forced[idx], S.flen[idx], nout, S.K, S.nstep[idx], S.vocab,\n'
     '                         S.schedtab[idx], S.schedlen[idx])  # local-p4: list mode table\n'),
]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def generate():
    raw = open(SRC, "rb").read()
    if sha(raw) != SRC_SHA:
        raise SystemExit(f"source module sha256 {sha(raw)} != validated {SRC_SHA}")
    s = raw.decode()
    for i, (a, b) in enumerate(PORT, 1):
        n = s.count(a)
        if n != 1:
            raise SystemExit(f"P{i}: anchor found {n} times: {a[:70]!r}")
        s = s.replace(a, b)
    return raw.decode(), s


def main():
    old, new = generate()
    diff = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                        "hook-validation/hook/forcewin_hooks.py (15e93e23, B1 attempt 2)",
                                        "tile-p4/build/files/vllm/forcewin_hooks.py (stock 0.9.1 port)"))
    if "--check" in sys.argv:
        ok = open(OUT).read() == new and open(DIFF).read() == diff
        print(("PASS" if ok else "FAIL") + f" forcewin port regenerates byte-identically ({sha(new.encode())[:16]})")
        return 0 if ok else 1
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(new)
    open(DIFF, "w").write(diff)
    added = sum(1 for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    removed = sum(1 for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---"))
    print(f"wrote {OUT} sha256 {sha(new.encode())} (+{added} / -{removed} lines vs 15e93e23)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
