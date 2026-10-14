# SPDX-License-Identifier: Apache-2.0
"""local-forcewin: forced-history decode window, per-position termination metrics, draft-schedule control,
decode-window-only D2 fault injection and window-exposure counters for B1 (GLM-5.3-Flash, vLLM Model Runner V2).

PLAN.md amendment 21 sections 1-2, 4, 7 (hook-validation/PLAN-NOTES.md). Installed as ``vllm/forcewin_hooks.py``
in a derived diagnostic image, called from marked lines (``# local-forcewin``) in five engine files.

INERT unless FORCEWIN=on in the server environment AND a request carries the key ``forcewin_hooks``
(vllm_xargs -> SamplingParams.extra_args). Unkeyed requests are never touched; with the env unset every hook
returns before doing anything (module flag ON is False and the hooked call sites test it first).

Measurement ordering (what makes a measured row valid; VALIDATION.md section 2):
  * A step's logits are produced by the forward pass from inputs fixed BEFORE the step: the last committed
    token (row 0) and the drafts (rows 1..n-1). Nothing this module does can change them.
  * on_logits() records the RAW (pre-mask) metrics of every row from ``logits`` before the (rejection) sampler
    runs; on_processed() records the post-top_p probabilities from the processed logits inside
    RejectionSampler._verify / Sampler.sample, BEFORE rejection_sample / gumbel_sample draw.
  * override() runs after the sampler has decided; it replaces the decision for forced requests: draft j+1 is
    accepted iff it equals the history token predicted at row j (longest matching prefix, a drafts), the
    committed tokens are history tokens f_0..f_a. Only rows 0..a were conditioned on history alone; exactly
    those rows are written to the per-position buffer (each output position once; asserted by a counter).
  * The engine then consumes the overridden (sampled, num_sampled) -> num_rejected, post_update, KDA rollback,
    drafting, scheduler accounting: committed state advances only by history tokens.

Env:
  FORCEWIN          "" (off) | "on"
  FORCEWIN_DIR      output directory (default /tmp/forcewin)
  FORCEWIN_WMAX     max forced tokens per request (default 8448)
  FORCEWIN_D2       "" | "1": allow decode-window-only D2 injection for requests that ask for it
  FORCEWIN_SHADOW   "" | "1": host-side tail-ring shadow ledger (pool corruption classes; one small
                            D2H copy per step while a forced request is in the batch). Enable in BOTH arms.
  FORCEWIN_SNAP     "" | "1": committed-state snapshots (validation sessions only; one D2H per step)
  FORCEWIN_LAYERS   max indexer layers tracked per rank for exposure (default 16; B1 has 11 DSA layers,
                            split over the two ranks by VLLM_GLM53_MLA_OWNERS)
  FORCEWIN_NDESIG   max designated positions per request for exposure counts (default 1024)
  FORCEWIN_NSEL     max positions per request whose full final selection is recorded (default 64)

Request keys (vllm_xargs):
  forcewin_hooks  list[int]   forced output ids by output index (window + the history's next token)
  local_fw_tag    str         fixture label (recorded)
  local_fw_draft  str         natural | oracle | poison | reject:<j> (1..K) | sched:<seed>
  local_fw_d2     [lo, hi]    output-index window in which the tail ring uses the OLD (D2) mapping
  local_fw_snap   list[int]   committed-state snapshots: at the first step whose committed output count reaches
                                each value (actual count recorded; needs SNAP=1)
  local_fw_dumpwin int        1: at the last commit also dump every paged-cache page holding window positions
                                (index-K pools of the whole window, for decode-vs-prefill byte comparison)
  local_fw_dumprange [lo, hi] absolute positions for that dump (default the request's window); a prefill
                                rebuild uses it to dump the SAME positions it wrote by prefill
  local_fw_desig  list[int]   designated output indices for exposure counters (default: 16-grid + last 64)
  local_fw_selrec list[int]   output indices whose full final pool selection is recorded
  local_fw_ref    int         prefill reference: record metrics at every computed prompt row predicting
                                position >= this value (index = predicted position - value)
"""
from __future__ import annotations

import json
import os
import threading
import time

import numpy as np
import torch

ON = os.environ.get("FORCEWIN", "").strip().lower() == "on"
OUT_DIR = os.environ.get("FORCEWIN_DIR", "/tmp/forcewin")
WMAX = int(os.environ.get("FORCEWIN_WMAX", "8448"))
D2_ALLOWED = ON and os.environ.get("FORCEWIN_D2", "") == "1"
SHADOW = ON and os.environ.get("FORCEWIN_SHADOW", "") == "1"
SNAP = ON and os.environ.get("FORCEWIN_SNAP", "") == "1"
MAX_LAYERS = int(os.environ.get("FORCEWIN_LAYERS", "16"))
NDESIG = int(os.environ.get("FORCEWIN_NDESIG", "1024"))
NSEL = int(os.environ.get("FORCEWIN_NSEL", "64"))
SELW = 511                                       # final attended pools per row (select_k - 1)

TERM_IDS = (154842, 154820, 154827, 154829)   # </think>, <|endoftext|>, <|user|>, <|observation|>
NT = len(TERM_IDS) + 1                          # tracked ids per row: 4 termination ids + the forced token
KPOOL, RING, BLOCK = 4, 8, 4352
TOP_P = 0.95

# per-position metric columns (float32)
COLS = (["lp_pre_%d" % i for i in range(NT)] + ["mass_lt_%d" % i for i in range(NT)]
        + ["mass_le_%d" % i for i in range(NT)] + ["p_post_%d" % i for i in range(NT)]
        + ["elig_lo_%d" % i for i in range(NT)] + ["elig_hi_%d" % i for i in range(NT)]
        + ["entropy", "argmax", "top1_p", "argmax_is_forced", "argmax_is_term", "step", "row", "n_rows",
           "accepted", "kind", "post_ok", "out_of_nucleus"])
C = {n: i for i, n in enumerate(COLS)}
NM = len(COLS)
KIND_VERIFY, KIND_SINGLE, KIND_PREFILL_REF = 0.0, 1.0, 2.0
# exposure columns (int32), per (request, output index, indexer layer slot)
XCOLS = ["step", "n_sel", "n_sel_win", "n_sel_win_corrupt", "n_sel_win_race", "n_win", "n_win_corrupt",
         "n_win_race"]
NX = len(XCOLS)
# shadow pool classes
PC_NONE, PC_CLEAN, PC_CORRUPT, PC_RACE, PC_SEED = 0, 1, 2, 3, 4
DRAFT_MODES = {"natural": 0, "oracle": 1, "poison": 2, "reject": 3, "sched": 4}


# =====================================================================================================
# Pure functions (no engine state; exercised on CPU by tests/)
# =====================================================================================================
def row_plan(cu, pos, prompt_len_r, is_forced_r, forced_len_r):
    """Per logits row: request index, local row j, output index predicted, row-in-window flag.

    cu [R+1] cumulative logits per batch request; pos [L] input position of each row; *_r [R] per batch request.
    The row with input position p predicts output index o = p + 1 - prompt_len.
    """
    L = pos.shape[0]
    rows = torch.arange(L, device=pos.device)
    req = (torch.searchsorted(cu, rows, right=True) - 1).clamp(0, cu.shape[0] - 2)
    j = rows - cu[req]
    o = pos.to(torch.int64) + 1 - prompt_len_r[req]
    fwin = is_forced_r[req] & (o >= 0) & (o < forced_len_r[req])
    return req, j, o, fwin


def forced_override(sampled, num_sampled, draft_sampled, pos, cu, forced, forced_len, prompt_len, is_forced):
    """Override the sampler's decision for forced requests.

    sampled [R, K+1], num_sampled [R]: the real sampler's output. draft_sampled [L]: the input token of each
    logits row (row 0: last committed token, rows 1..: drafts). pos [L] its position. cu [R+1].
    forced [R, W] history tokens by output index; forced_len, prompt_len [R]; is_forced [R].
    Returns (sampled, num_sampled, valid [R, K+1], out_idx [R, K+1], accepted [R]).
    A request is forced in a step iff its row 0 predicts an output index inside its forced window.
    """
    R, K1 = sampled.shape
    dev = sampled.device
    n_rows = cu[1:] - cu[:-1]
    j = torch.arange(K1, device=dev)
    row = cu[:-1, None] + j[None, :]
    in_req = j[None, :] < n_rows[:, None]
    row_c = torch.where(in_req, row, cu[:-1, None].expand(R, K1)).clamp(max=max(pos.shape[0] - 1, 0))
    out_idx = pos[row_c].to(torch.int64) + 1 - prompt_len[:, None]
    o0 = out_idx[:, 0]
    act = is_forced & (o0 >= 0) & (o0 < forced_len)
    f = forced.gather(1, out_idx.clamp(min=0, max=forced.shape[1] - 1))
    has_next = (j[None, :] + 1) < n_rows[:, None]
    nxt = torch.where(has_next, row_c + 1, row_c).clamp(max=max(pos.shape[0] - 1, 0))
    match = (draft_sampled[nxt] == f) & has_next & ((out_idx + 1) < forced_len[:, None])
    acc = torch.cumprod(match.to(torch.int64), dim=1).sum(dim=1)
    remaining = (forced_len - o0).clamp(min=1)
    n_new = torch.minimum(acc + 1, remaining)
    new_sampled = torch.where(act[:, None], f.to(sampled.dtype), sampled)
    new_num = torch.where(act, n_new.to(num_sampled.dtype), num_sampled)
    valid = act[:, None] & in_req & (j[None, :] < n_new[:, None])
    return new_sampled, new_num, valid, out_idx, torch.where(act, acc, torch.full_like(acc, -1))


def row_metrics(raw_logits, forced_tok, top_p=TOP_P, ids=TERM_IDS):
    """Pre-mask metrics of each row (fp32). forced_tok [rows] (-1 = none: its columns become NaN).

    mass_lt: probability of tokens strictly less likely than the tracked token; mass_le: inclusive of ties.
    Served top_p rule (apply_top_k_top_p): a token is kept iff the ascending cumulative mass THROUGH it exceeds
    1 - top_p; the top token is always kept. Tie order is unknown, so elig_lo (certainly kept) =
    mass_lt + p > 1 - top_p, elig_hi (possibly kept) = mass_le > 1 - top_p.
    """
    x = raw_logits.float()
    lp = x - torch.logsumexp(x, dim=-1, keepdim=True)
    p = lp.exp()
    ft = forced_tok.to(torch.int64)
    track = torch.cat([torch.tensor(ids, device=x.device).expand(x.shape[0], -1),
                       ft.clamp(min=0)[:, None]], dim=1)
    lp_t = lp.gather(1, track)
    p_t = lp_t.exp()
    mass_lt = torch.stack([(p * (p < p_t[:, k:k + 1])).sum(-1) for k in range(track.shape[1])], dim=1)
    mass_le = torch.stack([(p * (p <= p_t[:, k:k + 1])).sum(-1) for k in range(track.shape[1])], dim=1)
    ent = -(p * lp).sum(-1)
    top1_p, amax = p.max(dim=-1)
    is_top = p_t >= top1_p[:, None]
    elig_lo = (((mass_lt + p_t) > (1.0 - top_p)) | is_top).float()
    elig_hi = ((mass_le > (1.0 - top_p)) | is_top).float()
    none = ft < 0
    for t in (lp_t, mass_lt, mass_le, elig_lo, elig_hi):
        t[:, NT - 1] = torch.where(none, torch.full_like(t[:, NT - 1], float("nan")), t[:, NT - 1])
    is_term = torch.zeros_like(amax, dtype=torch.bool)
    for tid in ids:
        is_term |= amax == tid
    return dict(lp_pre=lp_t, mass_lt=mass_lt, mass_le=mass_le, elig_lo=elig_lo, elig_hi=elig_hi,
                entropy=ent, argmax=amax, top1_p=top1_p,
                argmax_is_forced=(amax == ft) & ~none, argmax_is_term=is_term)


def post_probs(processed_logits, forced_tok, ids=TERM_IDS):
    """Post-mask probabilities of the tracked ids from the served processed logits (-inf = masked)."""
    x = processed_logits.float()
    pp = torch.softmax(x, dim=-1)
    ft = forced_tok.to(torch.int64)
    track = torch.cat([torch.tensor(ids, device=x.device).expand(x.shape[0], -1),
                       ft.clamp(min=0)[:, None]], dim=1)
    out = pp.gather(1, track)
    out[:, NT - 1] = torch.where(ft < 0, torch.full_like(out[:, NT - 1], float("nan")), out[:, NT - 1])
    return out


def pack_metrics(m, p_post, step, j, n_rows, accepted, kind, post_ok):
    """[rows, NM] float32 in COLS order."""
    rows = m["entropy"].shape[0]
    dev = m["entropy"].device
    f = lambda v: (v.float() if torch.is_tensor(v) else torch.full((rows,), float(v), device=dev))  # noqa: E731
    oon = (p_post[:, NT - 1] <= 0).float()
    parts = [m["lp_pre"], m["mass_lt"], m["mass_le"], p_post, m["elig_lo"], m["elig_hi"],
             torch.stack([f(m["entropy"]), f(m["argmax"]), f(m["top1_p"]), f(m["argmax_is_forced"]),
                          f(m["argmax_is_term"]), f(step), f(j), f(n_rows), f(accepted), f(kind), f(post_ok),
                          oon], dim=1)]
    return torch.cat(parts, dim=1)


def draft_rows(mode, arg, forced, forced_len, n_out, K, step_ctr, vocab):
    """Draft tokens [R, K] for the draft-schedule modes (natural rows are returned unchanged by the caller).

    The next step's row 0 predicts output index n_out; draft d (1..K) is accepted iff it equals F[n_out+d-1].
    oracle: all K equal history (a = K unless the window ends); poison: none equal (a = 0);
    reject:<j>: the first j-1 equal, draft j does not (a = j-1); sched:<seed>: a_s pseudo-random in 0..K per step.
    """
    R = forced.shape[0]
    dev = forced.device
    d = torch.arange(K, device=dev)
    idx = (n_out[:, None] + d[None, :]).clamp(min=0, max=forced.shape[1] - 1)
    good = forced.gather(1, idx)
    bad = torch.remainder(good + 1, vocab)                     # never equal to the history token
    a_sched = torch.remainder(step_ctr * 1103515245 + arg * 12345 + 7, K + 1)
    n_good = torch.where(mode == 1, torch.full_like(n_out, K),
             torch.where(mode == 2, torch.zeros_like(n_out),
             torch.where(mode == 3, (arg - 1).clamp(min=0, max=K), a_sched)))
    return torch.where(d[None, :] < n_good[:, None], good, bad)


def d2_mapping(generic, circular, qsl, positions, n_tok, n_reqs, inj_row, lo_row, hi_row):
    """Decode-window-only D2: per token, the OLD tail mapping (generic per-group slot kernel output; for
    pos >= 8 it collapses onto null tail block 0) where the token's request is injected and its position lies in
    [lo, hi); the corrected circular mapping (B1) everywhere else (prefill rows, other requests, padding)."""
    out = circular.clone()
    if n_tok == 0:
        return out, torch.zeros((), dtype=torch.int64, device=circular.device)
    t = torch.arange(n_tok, device=circular.device)
    req = (torch.searchsorted(qsl[: n_reqs + 1], t, right=True) - 1).clamp(0, max(n_reqs - 1, 0))
    p = positions[:n_tok].to(torch.int64)
    inj = inj_row[req] & (p >= lo_row[req]) & (p < hi_row[req])
    out[:n_tok] = torch.where(inj, generic[:n_tok], circular[:n_tok])
    return out, inj.sum()


def exposure_counts(pool_ids, pos, P, cls_row, pool0):
    """Window exposure of the FINAL attended pool selection of each decode row (graph-safe, fixed shapes).

    pool_ids [rows, S] selected logical pool indices after the col-511 / truncation policy (-1 = empty);
    pos [rows] the row's input position; P [rows] prompt length (window start); cls_row [rows, NPW] shadow class of
    window pools (index i - pool0); pool0 [rows] = P // 4. A pool i is window-written iff its completing position
    4i+3 >= P (decode writer). Returns [rows, 7]: n_sel, n_sel_win, n_sel_win_corrupt, n_sel_win_race, n_win,
    n_win_corrupt, n_win_race.
    """
    pid = pool_ids.to(torch.int64)
    sel = pid >= 0
    win = sel & ((4 * pid + 3) >= P[:, None])
    k = (pid - pool0[:, None]).clamp(min=0, max=cls_row.shape[1] - 1)
    c = cls_row.gather(1, k)
    sel_cor = win & (c == PC_CORRUPT)
    sel_race = win & (c == PC_RACE)
    n_complete = torch.div(pos.to(torch.int64) + 1, 4, rounding_mode="floor")      # pools 0..n_complete-1 exist
    first_win = torch.div(P, 4, rounding_mode="floor")                             # smallest i with 4i+3 >= P
    n_win = (n_complete - first_win).clamp(min=0)
    ar = torch.arange(cls_row.shape[1], device=pid.device)
    live = (ar[None, :] + pool0[:, None]) < n_complete[:, None]
    n_wc = ((cls_row == PC_CORRUPT) & live).sum(1)
    n_wr = ((cls_row == PC_RACE) & live).sum(1)
    return torch.stack([sel.sum(1), win.sum(1), sel_cor.sum(1), sel_race.sum(1), n_win, n_wc, n_wr], dim=1)


class ShadowRing:
    """Host-side ledger of tail-ring slot writes for forced requests' decode tokens; classifies every committed
    pool record written by the decode path.

    Per step the decode kernel (kpool_decode_update_and_maybe_write_cache_batched), per request in position order,
    stashes each token's raw key at its tail slot and, when (pos+1) % 4 == 0, compresses members pos-3..pos read
    from base + q % 8, base = the CURRENT token's slot - pos % 8. Requests run concurrently: two requests writing
    the same slot in one step race. A pool committed by a valid row is
      clean   every member slot holds the request's own valid key for that position (written this step by itself
              with no other writer of that slot this step, or earlier and not overwritten since),
      corrupt some member slot holds another request's key or a stale / rejected-draft key, with no race,
      race    some member slot is also written by another request in the same step (outcome unknown),
      seed    a member precedes the window (prefill seed; not tracked).
    """

    def __init__(self):
        self.owner = {}          # slot -> (req_key, pos, valid) | "RACE"
        self.cls = {}            # req_key -> {pool_index: class}

    def step(self, toks):
        """toks: list of dicts per request in batch order: key, P (prompt len), pos list, slot list, valid list
        (row validity: rows 0..a are committed). Returns list of (key, pool, class) for committed pools."""
        writers = {}
        for r in toks:
            for s in r["slot"]:
                if s >= 0:
                    writers.setdefault(s, set()).add(r["key"])
        out = []
        for r in toks:
            key = r["key"]
            local = {}
            for pos, s, v in zip(r["pos"], r["slot"], r["valid"]):
                if s < 0:
                    continue
                local[s] = (key, pos, v)
                if (pos + 1) % KPOOL != 0 or not v:
                    continue
                base = s - pos % RING
                cls = PC_CLEAN
                for q in range(pos - KPOOL + 1, pos + 1):
                    ms = base + q % RING
                    if q < r["P"]:
                        cls = max(cls, PC_SEED) if cls != PC_CORRUPT else cls
                        continue
                    if len(writers.get(ms, ())) > 1 or (ms in writers and key not in writers[ms]):
                        cls = PC_RACE if cls in (PC_CLEAN, PC_SEED) else cls
                        continue
                    got = local.get(ms, self.owner.get(ms))
                    if got != (key, q, True):
                        cls = PC_CORRUPT
                pool = pos // KPOOL
                self.cls.setdefault(key, {})[pool] = cls
                out.append((key, pool, cls))
            # apply this request's writes (its rows in position order)
            for s, w in local.items():
                if len(writers.get(s, ())) > 1:
                    self.owner[s] = "RACE"
                else:
                    self.owner[s] = w
        return out


# =====================================================================================================
# Engine glue
# =====================================================================================================
class _Req:
    __slots__ = ("req_id", "idx", "P", "F", "tag", "mode", "marg", "d2", "snap", "ref", "t0", "snaps_done",
                 "desig", "selpos", "dumpwin", "dumprange")


class _State:
    def __init__(self):
        self.lock = threading.Lock()
        self.dev = None
        self.rank = 0
        self.K = 0
        self.vocab = 0
        self.max_reqs = 0
        self.reqs = {}             # req_idx -> _Req
        self.by_id = {}
        self.step = 0
        self.armed = False
        self.fwd_armed = False     # True between on_batch and on_logits (the forward's metadata build)
        self.batch = None          # current InputBatch (host refs)
        self.errors = 0
        self.layer_slot = {}
        self.steplog = []          # host per-step batch facts (forced requests present only)
        self.chunk_lo = 0
        self.sc = None
        self.row_d2on = None
        self.d2_tok = None         # (final tail mapping, positions, qsl, n_tok) of the step, for the shadow
        self.shadow = ShadowRing() if SHADOW else None
        self.exp_step = 0


S = _State()


def _log(rec):
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        rec = dict(rec, t=round(time.time(), 3), rank=S.rank, pid=os.getpid())
        with open(os.path.join(OUT_DIR, "fw_r%d_p%d.jsonl" % (S.rank, os.getpid())), "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:  # noqa: BLE001
        pass


def _err(where, exc):
    S.errors += 1
    if S.errors <= 50:
        _log({"kind": "fw_error", "where": where, "error": repr(exc)[:400]})


def _ensure(runner):
    if S.dev is not None:
        return
    dev = runner.device
    S.dev = dev
    try:
        import torch.distributed as dist
        S.rank = dist.get_rank() if dist.is_initialized() else 0
    except Exception:  # noqa: BLE001
        S.rank = 0
    S.K = int(getattr(runner, "num_speculative_steps", 0) or 0)
    S.vocab = int(runner.model_config.get_vocab_size()) if hasattr(runner, "model_config") else 154880
    S.max_reqs = int(runner.max_num_reqs)
    R, W = S.max_reqs, WMAX + 1
    S.forced = torch.zeros(R, W, dtype=torch.int64, device=dev)
    S.flen = torch.zeros(R, dtype=torch.int64, device=dev)
    S.plen = torch.zeros(R, dtype=torch.int64, device=dev)
    S.isf = torch.zeros(R, dtype=torch.bool, device=dev)
    S.mode = torch.zeros(R, dtype=torch.int64, device=dev)
    S.marg = torch.zeros(R, dtype=torch.int64, device=dev)
    S.nstep = torch.zeros(R, dtype=torch.int64, device=dev)      # forced steps per request (sched mode)
    S.d2on = torch.zeros(R, dtype=torch.bool, device=dev)
    S.d2lo = torch.zeros(R, dtype=torch.int64, device=dev)
    S.d2hi = torch.zeros(R, dtype=torch.int64, device=dev)
    S.buf = torch.full((R + 1, W, NM), float("nan"), dtype=torch.float32, device=dev)   # row R = sink
    S.nmeas = torch.zeros(R + 1, W, dtype=torch.int32, device=dev)
    S.acc_hist = torch.zeros(R, S.K + 2, dtype=torch.int64, device=dev)              # accepted-count histogram
    S.inv = torch.zeros(R, 4, dtype=torch.int64, device=dev)                         # integer invariant counters
    S.d2count = torch.zeros(R, dtype=torch.int64, device=dev)
    S.npw = W // KPOOL + 2
    S.pcls = torch.zeros(R + 1, S.npw, dtype=torch.int8, device=dev)                 # shadow classes (window pools)
    # exposure counters only at designated positions (o -> slot tables), full selections at a smaller subset
    S.dslot = torch.full((R + 1, W), -1, dtype=torch.int64, device=dev)
    S.sslot = torch.full((R + 1, W), -1, dtype=torch.int64, device=dev)
    S.xbuf = torch.full((R + 1, NDESIG + 1, MAX_LAYERS + 1, NX), -1, dtype=torch.int32, device=dev)
    S.selbuf = torch.full((R + 1, NSEL + 1, MAX_LAYERS + 1, SELW + 1), -1, dtype=torch.int32, device=dev)
    S.xstep = torch.zeros((), dtype=torch.int32, device=dev)
    S.xarmed = torch.zeros((), dtype=torch.int32, device=dev)
    S.bidx = torch.full((2 * S.max_reqs + 1,), -1, dtype=torch.int64, device=dev)   # batch row -> req_idx
    S.sc = None      # per-step scratch
    _log({"kind": "init", "K": S.K, "vocab": S.vocab, "max_reqs": R, "wmax": WMAX, "d2_allowed": D2_ALLOWED,
          "shadow": SHADOW, "snap": SNAP, "cols": COLS, "xcols": XCOLS})


def register(runner):
    """GPUModelRunner.capture_model: allocate every persistent table BEFORE graph capture, so the in-graph exposure
    hook is captured with fixed addresses (allocated lazily it would be missing from every captured graph)."""
    if not ON:
        return
    try:
        _ensure(runner)
        _log({"kind": "registered"})
    except Exception as exc:  # noqa: BLE001
        _err("register", exc)


def _xarg(sp, key, default=None):
    ea = getattr(sp, "extra_args", None) or {}
    return ea.get(key, default)


# ---------------------------------------------------------------- request lifecycle
def add_request(runner, req_id, req_idx, prompt_len, new_req_data):
    if not ON:
        return
    try:
        _ensure(runner)
        sp = new_req_data.sampling_params
        F = _xarg(sp, "forcewin_hooks")
        ref = _xarg(sp, "local_fw_ref")
        if F is None and ref is None:
            S.isf[req_idx] = False
            S.d2on[req_idx] = False
            S.reqs.pop(req_idx, None)
            return
        r = _Req()
        r.req_id, r.idx, r.P = req_id, int(req_idx), int(prompt_len)
        r.F = [int(x) for x in (F or [])][: WMAX + 1]
        r.tag = str(_xarg(sp, "local_fw_tag", ""))
        dm = str(_xarg(sp, "local_fw_draft", "natural"))
        name, _, arg = dm.partition(":")
        r.mode, r.marg = DRAFT_MODES.get(name, 0), int(arg or 0)
        d2 = _xarg(sp, "local_fw_d2")
        r.d2 = [int(d2[0]), int(d2[1])] if (d2 and D2_ALLOWED) else None
        r.snap = sorted(int(x) for x in (_xarg(sp, "local_fw_snap") or [])) if SNAP else []
        r.snaps_done = []
        r.dumpwin = bool(_xarg(sp, "local_fw_dumpwin", 0))
        dr = _xarg(sp, "local_fw_dumprange")
        r.dumprange = [int(dr[0]), int(dr[1])] if dr else None
        r.ref = None if ref is None else int(ref)
        r.t0 = time.time()
        S.reqs[r.idx] = r
        S.by_id[req_id] = r
        i = r.idx
        S.buf[i].fill_(float("nan"))
        S.nmeas[i].zero_()
        S.acc_hist[i].zero_()
        S.inv[i].zero_()
        S.nstep[i] = 0
        S.d2count[i] = 0
        S.pcls[i].zero_()
        S.xbuf[i].fill_(-1)
        S.selbuf[i].fill_(-1)
        S.dslot[i].fill_(-1)
        S.sslot[i].fill_(-1)
        Wn = len(r.F)
        des = _xarg(sp, "local_fw_desig")
        if des is None:      # default: a 16-token grid plus the last 64 positions of the window
            des = sorted(set(range(0, Wn, 16)) | set(range(max(0, Wn - 64), Wn)))
        r.desig = [int(o) for o in des if 0 <= int(o) < Wn][:NDESIG]
        r.selpos = [int(o) for o in (_xarg(sp, "local_fw_selrec") or []) if 0 <= int(o) < Wn][:NSEL]
        if r.desig:
            S.dslot[i, torch.tensor(r.desig, device=S.dev)] = torch.arange(len(r.desig), device=S.dev)
        if r.selpos:
            S.sslot[i, torch.tensor(r.selpos, device=S.dev)] = torch.arange(len(r.selpos), device=S.dev)
        n = len(r.F)
        if n:
            S.forced[i, :n].copy_(torch.tensor(r.F, dtype=torch.int64), non_blocking=False)
        S.flen[i] = n
        S.plen[i] = r.P
        S.isf[i] = n > 0
        S.mode[i] = r.mode
        S.marg[i] = r.marg
        S.d2on[i] = r.d2 is not None
        if r.d2 is not None:
            S.d2lo[i] = r.P + r.d2[0]
            S.d2hi[i] = r.P + r.d2[1]
        _log({"kind": "add", "req": req_id, "idx": i, "tag": r.tag, "P": r.P, "W": n, "mode": dm, "d2": r.d2,
              "snap": r.snap, "ref": r.ref, "P_mod8": r.P % RING})
    except Exception as exc:  # noqa: BLE001
        _err("add_request", exc)


def remove_request(runner, req_id):
    if not ON:
        return
    r = S.by_id.pop(req_id, None)
    if r is None:
        return
    try:
        if r.dumpwin:
            # Window pages at removal: the request is no longer scheduled and this runs at the start of the next
            # execute_model, before any forward could write blocks the scheduler has just freed (no per-step sync).
            F = len(r.F) if r.F else 1
            lo, hi = r.dumprange if r.dumprange else (r.P, r.P + F)
            snapshot(runner, None, None, r, F, None, lo=lo, hi=hi, suffix="win")
            _log({"kind": "snap", "tag": r.tag, "targets": ["win"], "F": F, "P": r.P})
        i = r.idx
        S.reqs.pop(i, None)
        S.isf[i] = False
        S.d2on[i] = False
        n = max(len(r.F), 1)
        W = WMAX + 1
        rec = {
            "tag": r.tag, "req": req_id, "P": r.P, "W": len(r.F), "mode": r.mode, "marg": r.marg, "d2": r.d2,
            "ref": r.ref, "cols": COLS, "xcols": XCOLS,
            "metrics": S.buf[i, :n if r.ref is None else W].detach().cpu(),
            "nmeas": S.nmeas[i, :n if r.ref is None else W].detach().cpu(),
            "acc_hist": S.acc_hist[i].detach().cpu(), "inv": S.inv[i].detach().cpu(),
            "d2_injected_tokens": int(S.d2count[i].item()),
            "pool_class": S.pcls[i].detach().cpu(),
            "desig": getattr(r, "desig", []), "exposure": S.xbuf[i, : len(getattr(r, "desig", []))].detach().cpu(),
            "selpos": getattr(r, "selpos", []),
            "selections": S.selbuf[i, : len(getattr(r, "selpos", []))].detach().cpu(),
            "layer_slots": dict(S.layer_slot), "snaps": r.snaps_done, "wall": round(time.time() - r.t0, 3),
            "errors": S.errors,
        }
        if S.shadow is not None:
            rec["shadow_classes"] = S.shadow.cls.pop(i, {})
        os.makedirs(OUT_DIR, exist_ok=True)
        torch.save(rec, os.path.join(OUT_DIR, "req_%s_r%d.pt" % (_safe(r.tag or req_id), S.rank)))
        _log({"kind": "done", "req": req_id, "tag": r.tag, "errors": S.errors})
        _flush_steplog()
    except Exception as exc:  # noqa: BLE001
        _err("remove_request", exc)


def _safe(s):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(s))[:80]


# ---------------------------------------------------------------- per step
def on_batch(runner, input_batch):
    """prepare_attn (real batches only): arm the step; per-batch-row tables for the D2 builder and the
    in-graph exposure counters."""
    if not ON or not S.reqs:
        S.armed = False
        S.fwd_armed = False
        return
    try:
        S.step += 1
        S.armed = True
        S.fwd_armed = True
        S.batch = input_batch
        nr = input_batch.num_reqs
        idx = input_batch.idx_mapping[:nr].to(torch.int64)
        S.bidx.fill_(-1)
        S.bidx[:nr].copy_(idx)
        S.xstep.fill_(S.step)
        S.xarmed.fill_(1)
        # per batch row, padded with "not injected" so that a builder call whose num_reqs includes FULL-graph
        # padding requests sees them as uninjected
        S.row_d2on = torch.zeros(2 * S.max_reqs + 1, dtype=torch.bool, device=S.dev)
        S.row_d2lo = torch.zeros(2 * S.max_reqs + 1, dtype=torch.int64, device=S.dev)
        S.row_d2hi = torch.zeros(2 * S.max_reqs + 1, dtype=torch.int64, device=S.dev)
        S.row_d2on[:nr] = S.d2on[idx]
        S.row_d2lo[:nr] = S.d2lo[idx]
        S.row_d2hi[:nr] = S.d2hi[idx]
        S.d2_tok = None
        S.sc = None
        if any(int(x) in S.reqs for x in input_batch.idx_mapping_np[:nr]):
            npf = int(np.sum(input_batch.is_prefilling_np[:nr])) if input_batch.is_prefilling_np is not None else 0
            pad = int(getattr(input_batch, "num_tokens_after_padding", input_batch.num_tokens)) - \
                int(input_batch.num_tokens)
            S.steplog.append((S.step, nr, int(input_batch.num_tokens), int(input_batch.num_draft_tokens), npf, pad))
            if len(S.steplog) >= 4096:
                _flush_steplog()
    except Exception as exc:  # noqa: BLE001
        S.armed = False
        _err("on_batch", exc)


def _flush_steplog():
    if S.steplog:
        _log({"kind": "steps", "cols": ["step", "num_reqs", "num_tokens", "num_draft_tokens", "n_prefilling",
                                        "padded_tokens"],
              "rows": S.steplog})
        S.steplog = []


def d2_tail_mapping(generic, circular, qsl, positions, n_tok, n_reqs):
    """KpoolTailMetadataBuilder.build: replace the circular mapping by the old one for injected window tokens."""
    if not (D2_ALLOWED and S.fwd_armed) or getattr(S, "row_d2on", None) is None:
        return circular
    try:
        if S.row_d2on.shape[0] < n_reqs:
            return circular
        out, ninj = d2_mapping(generic, circular, qsl, positions, int(n_tok), int(n_reqs),
                               S.row_d2on, S.row_d2lo, S.row_d2hi)
        # per-request injected-token count (device; read at removal)
        t = torch.arange(int(n_tok), device=circular.device)
        req = (torch.searchsorted(qsl[: n_reqs + 1], t, right=True) - 1).clamp(0, max(n_reqs - 1, 0))
        inj = (out[: int(n_tok)] != circular[: int(n_tok)])
        S.d2count.index_add_(0, S.bidx[req].clamp(min=0), inj.to(torch.int64))
        if SHADOW:
            S.d2_tok = (out[: int(n_tok)].clone(), positions[: int(n_tok)].clone(), qsl[: n_reqs + 1].clone())
        return out
    except Exception as exc:  # noqa: BLE001
        _err("d2_tail_mapping", exc)
        return circular


def note_tail_mapping(final, positions, qsl, n_tok, n_reqs):
    """B1 arm with SHADOW: record the step's (uninjected) tail mapping for the ledger."""
    if not (SHADOW and S.fwd_armed) or S.d2_tok is not None:
        return
    try:
        S.d2_tok = (final[: int(n_tok)].clone(), positions[: int(n_tok)].clone(), qsl[: n_reqs + 1].clone())
    except Exception as exc:  # noqa: BLE001
        _err("note_tail_mapping", exc)


def on_logits(runner, logits, input_batch):
    """GPUModelRunner.sample, BEFORE the (rejection) sampler: raw metrics of every row of forced requests."""
    S.fwd_armed = False
    if ON and S.dev is not None:
        S.xarmed.fill_(0)                       # the forward is over: disarm the in-graph exposure counters
    if not (ON and S.armed and S.reqs):
        S.sc = None
        return
    try:
        nr = input_batch.num_reqs
        idx = input_batch.idx_mapping[:nr].to(torch.int64)
        cu = input_batch.cu_num_logits[: nr + 1].to(torch.int64)
        pos = input_batch.positions[input_batch.logits_indices]
        L = logits.shape[0]
        pos = pos[:L]
        req, j, o, fwin = row_plan(cu, pos, S.plen[idx], S.isf[idx], S.flen[idx])
        ridx = idx[req]
        ft = torch.where(fwin, S.forced[ridx, o.clamp(min=0, max=WMAX)], torch.full_like(o, -1))
        m = row_metrics(logits, ft)
        S.sc = dict(idx=idx, cu=cu, pos=pos, req=req, j=j, o=o, fwin=fwin, ridx=ridx, ft=ft, m=m,
                    p_post=torch.full((L, NT), float("nan"), device=logits.device),
                    post_ok=torch.zeros(L, device=logits.device), L=L, nr=nr)
    except Exception as exc:  # noqa: BLE001
        S.sc = None
        _err("on_logits", exc)


def set_chunk(lo):
    if ON and S.sc is not None:
        S.chunk_lo = int(lo)


def on_processed(processed_logits, single=False):
    """Inside RejectionSampler._verify / Sampler.sample, after top_p and BEFORE the random draw."""
    sc = S.sc
    if sc is None:
        return
    try:
        lo = 0 if single else S.chunk_lo
        n = processed_logits.shape[0]
        ft = sc["ft"][lo:lo + n]
        sc["p_post"][lo:lo + n] = post_probs(processed_logits, ft)
        sc["post_ok"][lo:lo + n] = 1.0
    except Exception as exc:  # noqa: BLE001
        _err("on_processed", exc)


def _commit_rows(valid_flat, acc_r, n_rows_r, kind):
    """Write the valid rows' metrics into the per-position buffer (sink row for the rest)."""
    sc = S.sc
    L = sc["L"]
    j = sc["j"]
    o = sc["o"]
    tgt_req = torch.where(valid_flat, sc["ridx"], torch.full_like(sc["ridx"], S.max_reqs))
    tgt_o = torch.where(valid_flat, o, torch.zeros_like(o)).clamp(0, WMAX)
    acc_row = acc_r[sc["req"]]
    nrow = n_rows_r[sc["req"]]
    packed = pack_metrics(sc["m"], sc["p_post"], S.step, j, nrow, acc_row, kind, sc["post_ok"])
    S.buf[tgt_req, tgt_o] = packed
    S.nmeas.index_put_((tgt_req, tgt_o), torch.ones(L, dtype=torch.int32, device=packed.device),
                       accumulate=True)


def override(sampled, num_sampled, input_batch, draft_sampled, pos):
    """RejectionSampler.__call__, before get_num_sampled_and_rejected."""
    sc = S.sc
    if sc is None:
        return sampled, num_sampled
    try:
        nr = sc["nr"]
        idx, cu = sc["idx"], sc["cu"]
        new_s, new_n, valid, out_idx, acc = forced_override(
            sampled, num_sampled, draft_sampled[: sc["L"]], pos[: sc["L"]], cu, S.forced[idx], S.flen[idx],
            S.plen[idx], S.isf[idx])
        # flatten [R, K+1] validity to logits rows
        R, K1 = valid.shape
        jj = torch.arange(K1, device=valid.device)
        rows = cu[:-1, None] + jj[None, :]
        vflat = torch.zeros(sc["L"] + 1, dtype=torch.bool, device=valid.device)
        vflat.index_put_((torch.where(valid, rows, torch.full_like(rows, sc["L"])).reshape(-1),),
                         torch.ones(R * K1, dtype=torch.bool, device=valid.device))
        vflat = vflat[: sc["L"]] & sc["fwin"]
        n_rows = cu[1:] - cu[:-1]
        _commit_rows(vflat, acc.clamp(min=0), n_rows, KIND_VERIFY)
        act = acc >= 0
        # integer invariants: committed tokens == forced ids; accepted drafts == history (by construction)
        S.acc_hist.index_put_((idx, acc.clamp(min=0, max=S.K + 1)), act.to(torch.int64), accumulate=True)
        S.nstep.index_add_(0, idx, act.to(torch.int64))
        sc["acc"], sc["valid"], sc["new_n"], sc["out_idx"] = acc, valid, new_n, out_idx
        return new_s, new_n
    except Exception as exc:  # noqa: BLE001
        _err("override", exc)
        return sampled, num_sampled


def override_single(sampled, input_batch):
    """Sampler.__call__ (no drafts in the batch): one row per request."""
    sc = S.sc
    if sc is None:
        return sampled
    try:
        fwin = sc["fwin"]
        L = sc["L"]
        if L != sampled.shape[0]:
            return sampled
        new = torch.where(fwin, sc["ft"].to(sampled.dtype), sampled)
        n_rows = sc["cu"][1:] - sc["cu"][:-1]
        _commit_rows(fwin, torch.zeros(sc["nr"], dtype=torch.int64, device=sampled.device), n_rows, KIND_SINGLE)
        S.acc_hist.index_put_((sc["idx"], torch.zeros_like(sc["idx"])), fwin[: sc["nr"]].to(torch.int64)
                              if L == sc["nr"] else torch.zeros_like(sc["idx"]), accumulate=True)
        sc["acc"] = torch.where(fwin[: sc["nr"]], torch.zeros_like(sc["idx"]), torch.full_like(sc["idx"], -1)) \
            if L == sc["nr"] else None
        return new
    except Exception as exc:  # noqa: BLE001
        _err("override_single", exc)
        return sampled


def after_commit(runner, input_batch, sampler_output, num_sampled, num_rejected, attn_metadata=None):
    """sample_tokens, after postprocess_sampled: integer invariants, shadow ledger, snapshots."""
    sc = S.sc
    if sc is None:
        return
    try:
        idx = sc["idx"]
        # invariant: the request's committed output count matches the forced commit and the stored tokens are
        # the history (device-side counters; read at removal)
        rs = runner.req_states
        tl = rs.total_len.gpu[idx].to(torch.int64)
        nout = tl - S.plen[idx]
        isf = S.isf[idx]
        last = rs.last_sampled_tokens[idx, 0].to(torch.int64)
        exp_last = S.forced[idx, (nout - 1).clamp(min=0, max=WMAX)]
        in_win = isf & (nout >= 1) & (nout <= S.flen[idx])
        bad_last = in_win & (last != exp_last)
        S.inv[:, 0].index_add_(0, idx, in_win.to(torch.int64))        # checked steps
        S.inv[:, 1].index_add_(0, idx, bad_last.to(torch.int64))      # last committed token != history
        if SHADOW and S.d2_tok is not None and sc.get("acc") is not None:
            _shadow_step(input_batch, sc)
        if SNAP:
            _maybe_snapshot(runner, input_batch, nout, attn_metadata)
    except Exception as exc:  # noqa: BLE001
        _err("after_commit", exc)


def after_propose(runner, input_batch):
    """After the speculator has written req_states.draft_tokens: draft-schedule modes for forced requests."""
    sc = S.sc
    if sc is None or S.K == 0:
        return
    try:
        idx = sc["idx"]
        mode = S.mode[idx]
        if not bool(S.reqs) or all(r.mode == 0 for r in S.reqs.values()):
            return
        rs = runner.req_states
        nout = rs.total_len.gpu[idx].to(torch.int64) - S.plen[idx]
        cur = rs.draft_tokens[idx]
        new = draft_rows(mode, S.marg[idx], S.forced[idx], S.flen[idx], nout, S.K, S.nstep[idx], S.vocab)
        use = (S.isf[idx] & (mode != 0))[:, None]
        rs.draft_tokens[idx] = torch.where(use, new.to(cur.dtype), cur)
    except Exception as exc:  # noqa: BLE001
        _err("after_propose", exc)


def prefill_ref(runner, hidden_states, input_batch):
    """Prefill reference: metrics at every computed prompt row of requests keyed local_fw_ref (eager)."""
    if not (ON and S.armed and S.reqs):
        return
    try:
        nr = input_batch.num_reqs
        for b in range(nr):
            i = int(input_batch.idx_mapping_np[b])
            r = S.reqs.get(i)
            if r is None or r.ref is None:
                continue
            s, e = int(input_batch.query_start_loc_np[b]), int(input_batch.query_start_loc_np[b + 1])
            c0 = int(input_batch.num_computed_prefill_tokens_np[b])
            for a in range(s, e, 256):
                z = min(e, a + 256)
                pos_next = torch.arange(c0 + (a - s) + 1, c0 + (z - s) + 1, device=hidden_states.device)
                keep = pos_next >= r.ref
                if not bool((c0 + (z - s)) >= r.ref):
                    continue
                logits = runner.model.compute_logits(hidden_states[a:z])
                nxt = torch.tensor(_prompt_slice(runner, i, c0 + (a - s) + 1, c0 + (z - s) + 1, r.P),
                                   dtype=torch.int64, device=logits.device)
                m = row_metrics(logits, nxt)
                pp = _ref_post(logits, nxt)
                n = z - a
                packed = pack_metrics(m, pp, S.step, -1, n, -1, KIND_PREFILL_REF, 1.0)
                o = (pos_next - r.ref).clamp(0, WMAX)
                tgt = torch.where(keep & ((pos_next - r.ref) <= WMAX), torch.full_like(o, i),
                                  torch.full_like(o, S.max_reqs))
                S.buf[tgt, o] = packed
                S.nmeas.index_put_((tgt, o), torch.ones(n, dtype=torch.int32, device=o.device), accumulate=True)
    except Exception as exc:  # noqa: BLE001
        _err("prefill_ref", exc)


_PROMPTS = {}


def _prompt_slice(runner, req_idx, a, z, P):
    """Prompt ids a..z-1 (-1 at/after the prompt end). Read once per request from req_states (host)."""
    key = (req_idx, P)
    ids = _PROMPTS.get(key)
    if ids is None:
        t = runner.req_states.all_token_ids
        ids = (t.np[req_idx, :P].tolist() if getattr(t, "np", None) is not None
               else t.gpu[req_idx, :P].cpu().tolist())
        _PROMPTS.clear()
        _PROMPTS[key] = ids
    return [ids[k] if k < P else -1 for k in range(a, z)]


def _ref_post(logits, nxt):
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
    p = torch.full((logits.shape[0],), TOP_P, dtype=torch.float32, device=logits.device)
    return post_probs(apply_top_k_top_p(logits.float().clone(), None, p), nxt)


# ---------------------------------------------------------------- in-graph exposure (indexer decode path)
def on_selection(layer_name, pool_ids, dec_seq, batch_size, next_n):
    """sparse_attn_indexer_kpool decode branch, after the final pool selection (cols :511), before expansion.
    Runs inside FULL CUDA graph capture/replay: fixed-shape device ops only; per-step values come from the
    persistent tables refreshed in on_batch (S.bidx, S.xstep, S.xarmed)."""
    if not ON or S.dev is None:
        return
    try:
        slot = S.layer_slot.get(layer_name)
        if slot is None:
            slot = len(S.layer_slot)
            if slot >= MAX_LAYERS:
                slot = MAX_LAYERS                # overflow layers share the sink slot (counted in layer_slots)
            S.layer_slot[layer_name] = slot
        rows = pool_ids.shape[0]
        dev = pool_ids.device
        b = torch.div(torch.arange(rows, device=dev), max(int(next_n), 1), rounding_mode="floor")
        ridx = S.bidx[b.clamp(max=2 * S.max_reqs)]
        okr = ridx >= 0
        rc = ridx.clamp(min=0)
        pos = dec_seq[:rows].to(torch.int64) - 1
        P = S.plen[rc]
        o = pos + 1 - P
        ok = okr & S.isf[rc] & (o >= 0) & (o <= WMAX) & (S.xarmed > 0)
        cnt = exposure_counts(pool_ids, pos, P, S.pcls[rc], torch.div(P, 4, rounding_mode="floor"))
        rec = torch.cat([S.xstep.to(torch.int64).expand(rows)[:, None], cnt], dim=1).to(torch.int32)
        oc = o.clamp(0, WMAX)
        ds = S.dslot[rc, oc]
        okd = ok & (ds >= 0)
        tr = torch.where(okd, rc, torch.full_like(rc, S.max_reqs))
        S.xbuf[tr, torch.where(okd, ds, torch.zeros_like(ds)), slot] = rec
        ss = S.sslot[rc, oc]
        oks = ok & (ss >= 0)
        tr2 = torch.where(oks, rc, torch.full_like(rc, S.max_reqs))
        sel = torch.cat([S.xstep.to(torch.int64).expand(rows)[:, None], pool_ids.to(torch.int64)], dim=1)
        S.selbuf[tr2, torch.where(oks, ss, torch.zeros_like(ss)), slot, : sel.shape[1]] = sel.to(torch.int32)
    except Exception as exc:  # noqa: BLE001
        _err("on_selection", exc)


# ---------------------------------------------------------------- shadow ledger step (host)
def _shadow_step(input_batch, sc):
    final, positions, qsl = S.d2_tok
    S.d2_tok = None
    nr = sc["nr"]
    acc = sc["acc"]
    host = torch.cat([final.to(torch.int64), positions.to(torch.int64), qsl.to(torch.int64),
                      acc.to(torch.int64), sc["idx"].to(torch.int64)]).cpu().numpy()
    n = final.shape[0]
    fin, pos, q = host[:n], host[n:2 * n], host[2 * n:2 * n + nr + 1]
    a = host[2 * n + nr + 1: 2 * n + 2 * nr + 1]
    idx = host[2 * n + 2 * nr + 1:]
    toks = []
    for bb in range(nr):
        i = int(idx[bb])
        r = S.reqs.get(i)
        if r is None or not r.F or a[bb] < 0:
            continue
        s, e = int(q[bb]), int(q[bb + 1])
        ps, sl = pos[s:e].tolist(), fin[s:e].tolist()
        if not ps or ps[0] < r.P:
            continue
        val = [jj <= a[bb] for jj in range(len(ps))]
        toks.append(dict(key=i, P=r.P, pos=ps, slot=sl, valid=val))
    for key, pool, cls in S.shadow.step(toks):
        r = S.reqs.get(key)
        k = pool - r.P // KPOOL
        if 0 <= k < S.npw:
            S.pcls[key, k] = cls


# ---------------------------------------------------------------- committed-state snapshots (validation only)
def _maybe_snapshot(runner, input_batch, nout_dev, attn_metadata):
    want = [(b, S.reqs.get(int(i))) for b, i in enumerate(input_batch.idx_mapping_np[: input_batch.num_reqs])]
    want = [(b, r) for b, r in want if r is not None and r.snap]
    if not want:
        return
    nout = nout_dev.cpu().tolist()                   # one D2H; validation sessions only
    for b, r in want:
        F = int(nout[b])
        due = [t for t in r.snap if t <= F and t not in [d[0] for d in r.snaps_done]]
        if due:
            for t in due:
                r.snaps_done.append((t, F))
            snapshot(runner, input_batch, b, r, F, attn_metadata)
            _log({"kind": "snap", "tag": r.tag, "targets": due, "F": F, "P": r.P})


def snapshot(runner, input_batch, b, r, F, attn_metadata, lo=None, hi=None, suffix=""):
    """Committed state at output count F: integers exactly; raw pages around the frontier per layer; KDA rows at
    every speculative slot of the request (committed slot = num_accepted - 1, the rest scratch)."""
    i = r.idx
    rs = runner.req_states
    rec = {"tag": r.tag, "F": F, "P": r.P, "step": S.step,
           "frontier_pos": r.P + F - 2,                   # last position whose state is committed
           "num_computed": int(rs.num_computed_tokens.gpu[i].item()),
           "total_len": int(rs.total_len.gpu[i].item()),
           "last_sampled": int(rs.last_sampled_tokens[i, 0].item()),
           "tokens_ok": bool(torch.equal(
               rs.all_token_ids.gpu[i, r.P:r.P + F].to(torch.int64).cpu() if hasattr(rs.all_token_ids, "gpu")
               else torch.tensor(rs.all_token_ids.np[i, r.P:r.P + F], dtype=torch.int64),
               torch.tensor(r.F[:F], dtype=torch.int64))),
           "num_accepted": None, "groups": [], "layers": {}}
    bt = runner.block_tables
    for g in range(bt.num_kv_cache_groups):
        nb = int(bt.num_blocks.np[g, i])
        rec["groups"].append({"g": g, "block_size": int(bt.block_sizes[g]),
                              "kernel_block": int(bt.kernel_block_sizes[g]),
                              "bpk": int(bt.blocks_per_kv_block[g]),
                              "ids": bt.block_tables[g].gpu[i, :nb].cpu()})
    try:
        ms = runner.model_state
        rec["num_accepted"] = int(ms.num_accepted_tokens_gpu[i].item()) \
            if hasattr(ms, "num_accepted_tokens_gpu") else None
    except Exception as exc:  # noqa: BLE001
        _err("snapshot_num_accepted", exc)
    ctx = runner.vllm_config.compilation_config.static_forward_context
    if lo is None:
        lo, hi = max(0, rec["frontier_pos"] - 64), rec["frontier_pos"] + S.K + 2
    rec["range"] = [lo, hi]
    for name, mod in ctx.items():
        try:
            kv = getattr(mod, "kv_cache", None)
            if kv is None:
                continue
            if isinstance(kv, (list, tuple)) and kv and isinstance(kv[0], (list, tuple)):
                kv = kv[0]                       # per-virtual-engine list of (conv, ssm)
            if isinstance(kv, (list, tuple)) and len(kv) == 1:
                kv = kv[0]                       # per-virtual-engine list of one tensor
            rec["layers"][name] = _layer_pages(mod, kv, rec, lo, hi, b, input_batch, attn_metadata, name)
        except Exception as exc:  # noqa: BLE001
            _err("snapshot_layer", exc)
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(rec, os.path.join(OUT_DIR, "snap_%s_F%d%s_r%d.pt" % (_safe(r.tag), F, suffix, S.rank)))


_SKETCH = {}


def _sketch(x, k=256):
    """Fixed random projection (seeded per size) of a flattened fp32 state: rel. L2 differences of sketches
    approximate those of the full states (Johnson-Lindenstrauss)."""
    v = x.reshape(-1).float()
    key = (v.numel(), str(v.device))
    M = _SKETCH.get(key)
    if M is None:
        g = torch.Generator(device="cpu").manual_seed(20261013 + v.numel() % 9973)
        M = (torch.randn(k, v.numel(), generator=g) / k ** 0.5).to(v.device)
        _SKETCH[key] = M
    return torch.mv(M, v).cpu()


def _layer_pages(mod, kv, rec, lo, hi, b, input_batch, attn_metadata, name):
    """Raw bytes of the pages that hold positions lo..hi for attention-like caches (page = kernel block), the
    request's whole tail block(s), or KDA state rows. Shapes, strides and dtypes are recorded so the offline
    analysis decodes them with the engine's own view code."""
    out = {"type": type(mod).__name__}
    if isinstance(kv, (list, tuple)) and len(kv) == 2 and kv[1].dim() >= 3 and kv[1].dtype == torch.float32:
        conv, ssm = kv                                  # KDA (conv_state, recurrent_state)
        md = attn_metadata.get(name) if isinstance(attn_metadata, dict) else None
        out["kind"] = "kda"
        if md is None or input_batch is None:
            return out
        nd = input_batch.num_draft_tokens_per_req
        is_spec = bool(nd is not None and nd[b] > 0)
        out["is_spec"] = is_spec
        if md is not None and not is_spec and getattr(md, "non_spec_state_indices_tensor", None) is not None:
            # prefill-built reference (sent alone, so its row is the only non-spec row of the step)
            ns = md.non_spec_state_indices_tensor.reshape(-1).to(torch.int64)
            out["non_spec_slots"] = ns.cpu()
            s0 = int(ns[0])
            out["slots"] = torch.tensor([s0])
            out["ssm_sketch"] = _sketch(ssm[s0])[None]
            out["ssm_norm"] = ssm[s0].float().norm().reshape(1).cpu()
            out["conv"] = conv[s0].detach().cpu().clone()
            return out
        if md is not None and getattr(md, "spec_state_indices_tensor", None) is not None:
            spec_rank = int(np.sum(nd[:b] > 0)) if nd is not None else b
            sidx = md.spec_state_indices_tensor[spec_rank].to(torch.int64)
            out["spec_rank"] = spec_rank
            out["slots"] = sidx.cpu()
            out["num_accepted"] = int(md.num_accepted_tokens[spec_rank].item()) \
                if getattr(md, "num_accepted_tokens", None) is not None else None
            out["ssm_sketch"] = torch.stack([_sketch(ssm[s]) for s in sidx.tolist()])
            out["ssm_norm"] = torch.stack([ssm[s].float().norm() for s in sidx.tolist()]).cpu()
            out["conv"] = conv[int(sidx[0])].detach().cpu().clone()
        return out
    t = kv if torch.is_tensor(kv) else None
    if t is None:
        return out
    out["shape"], out["stride"], out["dtype"] = list(t.shape), list(t.stride()), str(t.dtype)
    out["kind"] = "paged"
    # Candidate pages for positions lo..hi under each group's table (decoded offline with the engine's view code;
    # extra candidates cost space only): token-granular kernel pages (MLA: kernel block 64 tokens, ids already
    # expanded x68 per 4,352-token block), index-K pages (logical block * 17 + (pool % 1088) // 64, the layout the
    # B1 mapguard verified), and the request's tail block plus the null tail block 0 (D2's target).
    pages = {}

    def take(pg):
        if 0 <= pg < t.shape[0] and pg not in pages:
            pages[pg] = t[pg].detach().cpu().clone()

    for gr in rec["groups"]:
        ids = gr["ids"].tolist()
        kbs, bpk = gr["kernel_block"], max(gr["bpk"], 1)
        if kbs <= 0 or not ids:
            continue
        if gr["block_size"] == RING:
            take(int(ids[0]))
            take(0)
            continue
        seen = set()
        for p in range(lo, hi + 1, 1):
            col = p // kbs
            if col < len(ids) and col not in seen:
                seen.add(col)
                take(int(ids[col]))
            if bpk > 1:
                c0 = (p // (kbs * bpk)) * bpk
                if c0 < len(ids):
                    take((int(ids[c0]) // bpk) * 17 + ((p // KPOOL) % (BLOCK // KPOOL)) // 64)
    out["pages"] = pages
    return out
