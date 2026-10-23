#!/usr/bin/env python3
"""Build livecheck-22ao/guard/mapguard_hooks.py from the frozen B1 guard (d7/step2/guard, a8e59fa5...).
Every edit is an exact-match replacement that must occur exactly once."""
import hashlib, sys
SRC = "<runs>/d7/step2/guard/mapguard_hooks.py"
DST = sys.argv[1]
s = open(SRC).read()
assert hashlib.sha256(s.encode()).hexdigest().startswith("a8e59fa5"), "base guard changed"
def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:80], s.count(a)); s = s.replace(a, b)
parts = open(sys.argv[2]).read().split("=====")
blk = {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}
rep('Graph safety: hooks use fixed-shape device ops only', blk["HEADER"].strip("\n") + '\n\nGraph safety: hooks use fixed-shape device ops only')
rep('''SITES = ["mla_w", "idx_w_pf", "idx_w_dec", "tail_w_seed", "tail_w_stash",
         "idx_r_dec", "idx_r_pf", "mla_r", "pos"]''', '''SITES = ["mla_w", "idx_w_pf", "idx_w_dec", "tail_w_seed", "tail_w_stash",
         "idx_r_dec", "idx_r_pf", "mla_r", "pos", "mla_len"]''')
rep('''    S_MLA_R, S_POS = range(len(SITES))''', '''    S_MLA_R, S_POS, S_MLA_LEN = range(len(SITES))''')
rep('''       "pc_checked", "pc_mism", "d7_fallback", "legacy"]''', '''       "pc_checked", "pc_mism", "d7_fallback", "legacy",
       # 22ao amendment (append-only)
       "len_checked", "len_pos", "len_neg", "nsa_pos", "nsa_neg",
       "pa_planted", "pa_det", "pb_planted", "pb_det", "pc_planted", "pc_det", "pd_planted", "pd_det"]''')
rep('''ORACLE_MAX = 4096''', blk["CONSTS"].strip("\n") + '\nORACLE_MAX = 4096')
rep('''        self.scratch = None                     # oracle record pages, allocated on first use''',
    '''        self.scratch = None                     # oracle record pages, allocated on first use\n''' + blk["DEVBUF"].strip("\n"))
rep(blk["OLDMASKS"].strip("\n"), blk["NEWMASKS"].strip("\n"))
rep('''        record_rows(dev, rowbad & (armed != 0), rows)


# ------------------------------------------------------------------ host side: ledger''',
    '''        record_rows(dev, rowbad & (armed != 0), rows)\n''' + blk["CSTAIL"].rstrip("\n") + '''


''' + blk["HELPERS"].strip("\n") + '''


# ------------------------------------------------------------------ host side: ledger''')
rep('''# ------------------------------------------------------------------ worker install''',
    blk["HOOK"].strip("\n") + '''


# ------------------------------------------------------------------ worker install''')
rep('''    tensors = [dev.cnt, dev.rec, dev.nrec, dev.layer, dev.ring_step, dev.ring_p0,
               dev.nslog, dev.slog, dev.nbsnap, dev.bsnap, dev.bkey]''', '''    tensors = [dev.cnt, dev.rec, dev.nrec, dev.layer, dev.ring_step, dev.ring_p0,
               dev.nslog, dev.slog, dev.nbsnap, dev.bsnap, dev.bkey,
               dev.nlrec, dev.lrec, dev.lenhist, dev.nsahist]   # 22ao: counter before buffer''')
rep('''        cnt, rec, nrec, layer, rstep, rp0, nslog, slog, nbs, bsnap, bkey = _snapshot()''',
    '''        cnt, rec, nrec, layer, rstep, rp0, nslog, slog, nbs, bsnap, bkey, nlrec, lrec, lhist, nhist = _snapshot()
        nl = min(int(nlrec[0]), LCAP)
        lnew = lrec[S.lrec_emitted:nl].tolist()
        S.lrec_emitted = nl''')
rep('''                  n_slog=int(nslog[0]), slog=snew, n_bsnap=int(nbs[0]), bsnap=bnew))''',
    '''                  n_slog=int(nslog[0]), slog=snew, n_bsnap=int(nbs[0]), bsnap=bnew,
                  n_lrec=int(nlrec[0]), lrec=lnew, lenhist=lhist.tolist(), nsahist=nhist.tolist()))''')
rep('''        self.bsnap_emitted = 0''', '''        self.bsnap_emitted = 0\n        self.lrec_emitted = 0''')
# 22ao observer check: count host waits that actually block (the event was not yet complete)
rep('''        ev.synchronize()   # only waits if the GPU is >= 4 steps behind the host''',
    '''        S.uploads += 1
        if not ev.query():                     # 22ao: this wait blocks the host (GPU >= 4 steps behind)
            S.blocked_waits += 1
        ev.synchronize()   # only waits if the GPU is >= 4 steps behind the host''')
rep('''                  ring=sorted(ring), row_reuse=S.row_reuse, guard_errors=S.errors,''',
    '''                  ring=sorted(ring), row_reuse=S.row_reuse, guard_errors=S.errors,
                  uploads=S.uploads, blocked_waits=S.blocked_waits,''')
rep('''        self.row_reuse = 0''', '''        self.row_reuse = 0
        self.uploads = 0          # 22ao
        self.blocked_waits = 0    # 22ao''')
# 22ap: poll race fix. The pinned copy buffers were created by whichever thread polled first; when that was the
# engine thread's baseline poll (inside torch.inference_mode) they became inference tensors and every later poll from
# the poller thread failed on copy_. Buffers are now allocated with inference mode off, and polls are serialized.
rep('''        self.blocked_waits = 0    # 22ao''', '''        self.blocked_waits = 0    # 22ao
        self.poll_lock = threading.Lock()   # 22ap: engine-thread baseline and poller thread poll one at a time''')
rep('''            S.pins = [torch.empty(t.shape, dtype=t.dtype, pin_memory=True) for t in tensors]''',
    '''            S.pins = _alloc_pins(tensors, pin=True)''')
rep('''def _snapshot():''', '''def _alloc_pins(tensors, pin):
    """22ap: host copy buffers as normal tensors, whichever thread (and inference mode) allocates them."""
    with torch.inference_mode(False):
        return [torch.empty(t.shape, dtype=t.dtype, pin_memory=pin) for t in tensors]


def _snapshot():''')
rep('''def poll(reason: str) -> None:
    if S.dev is None:
        return''', '''def poll(reason: str) -> None:
    if not S.poll_lock.acquire(timeout=60.0):   # 22ap: bounded, so the exit poll cannot hang shutdown
        _err("poll_lock", RuntimeError("poll lock not acquired in 60 s (" + reason + ")"))
        return
    try:
        _poll_locked(reason)
    finally:
        S.poll_lock.release()


def _poll_locked(reason: str) -> None:
    if S.dev is None:
        return''')
open(DST, "w").write(s)
print("written", DST, hashlib.sha256(s.encode()).hexdigest()[:16])
