"""Reference logic for the forced-decode window hook (PROPOSAL.md section 3). Pure torch, no engine state.

This is the arithmetic the engine hook would run on the GPU after the real sampler (or rejection sampler) has
produced its output for a step. It is written device-agnostic and without host synchronisation so that the same
functions can run inside the engine; here it is exercised on CPU only (test_forcewin_cpu.py).

Layout follows vLLM Model Runner V2 (v1/worker/gpu): a step has R requests; request r owns logits rows
cu[r]..cu[r+1]-1 (n_r = drafts_r + 1 rows). draft_sampled[row] is the input token at that row (row 0: the last
committed token; rows 1..: the draft tokens). pos[row] is that input's position; the row's logits predict position
pos[row] + 1. sampled is [R, K+1] and num_sampled is [R] (rejection_sampler_utils.rejection_sample).
"""
import torch

TERM_IDS = (154842, 154820, 154827, 154829)   # </think>, <|endoftext|>, <|user|>, <|observation|>


def forced_override(sampled, num_sampled, draft_sampled, pos, cu, forced, forced_len, prompt_len, is_forced):
    """Return (sampled, num_sampled, row_valid, row_out_idx) with forced requests overridden.

    forced      [R, Wmax] int64: the forced history tokens of each request (by output index), padded.
    forced_len  [R] int64: number of forced output tokens (= max_tokens).
    prompt_len  [R] int64: prompt length (output index o = predicted position - prompt_len).
    is_forced   [R] bool.
    A draft is accepted iff it equals the history token at its position; the token committed after the accepted
    prefix is the history token (so committed state advances only by history tokens). Rows after the first
    mismatch were conditioned on a wrong draft and are not valid measurements.
    """
    R, K1 = sampled.shape
    n_rows = cu[1:] - cu[:-1]                                  # [R]
    j = torch.arange(K1, device=sampled.device)                # local row index
    row = cu[:-1, None] + j[None, :]                           # [R, K1] global row ids
    in_req = j[None, :] < n_rows[:, None]
    row_c = torch.where(in_req, row, torch.zeros_like(row))
    out_idx = pos[row_c] + 1 - prompt_len[:, None]             # output index predicted by each row
    out_idx_c = out_idx.clamp(min=0, max=forced.shape[1] - 1)
    f = forced.gather(1, out_idx_c)                            # history token each row should predict
    # draft for row j+1 is the input at row j+1; it is accepted iff equal to the history token predicted at row j
    nxt = torch.where(j[None, :] + 1 < n_rows[:, None], row_c + 1, row_c)
    match = (draft_sampled[nxt] == f) & (j[None, :] + 1 < n_rows[:, None])
    acc = torch.cumprod(match.to(torch.int64), dim=1).sum(dim=1)          # accepted drafts a_r
    remaining = (forced_len - out_idx[:, 0]).clamp(min=1)                  # tokens still to force
    n_new = torch.minimum(acc + 1, remaining)
    new_sampled = torch.where(is_forced[:, None], f, sampled)
    new_num = torch.where(is_forced, n_new.to(num_sampled.dtype), num_sampled)
    valid = is_forced[:, None] & in_req & (j[None, :] < n_new[:, None])
    return new_sampled, new_num, valid, out_idx


def top1_p_of(p):
    return p.max(dim=-1).values


def row_metrics(raw_logits, processed_logits, forced_tok, top_p=0.95, ids=TERM_IDS):
    """Per-row measurements for rows that are valid (caller selects rows). fp32 throughout.

    raw_logits: target logits before any processing (temperature 1.0 is a no-op in the served config).
    processed_logits: the served logits after top_p (masked entries are -inf), i.e. exactly what was sampled from.
    Returns a dict of [rows, ...] tensors:
      lp_pre      log p of each tracked id and of the forced token (pre-mask)
      mass_lt     probability mass of tokens strictly less likely than each tracked token
      mass_le     the same, inclusive of tokens tied with it (and itself)
      elig_lo/hi  eligibility bounds from the served rule (a token is kept iff the inclusive ascending cumulative
                  mass through it exceeds 1 - top_p; the most likely token is always kept). Tie order is unknown,
                  so: certainly kept if mass_lt + p > 1 - top_p (elig_lo); possibly kept if mass_le > 1 - top_p
                  (elig_hi). The served value itself is p_post > 0.
      p_post      probability after masking and renormalisation, read from processed_logits
      entropy, argmax, top1_p
    """
    x = raw_logits.float()
    lse = torch.logsumexp(x, dim=-1, keepdim=True)
    lp = x - lse
    p = lp.exp()
    track = torch.cat([torch.tensor(ids, device=x.device).expand(x.shape[0], -1), forced_tok[:, None]], dim=1)
    lp_t = lp.gather(1, track)
    p_t = lp_t.exp()
    # mass of tokens no more likely than each tracked token: O(V) per tracked token, no sort
    mass_lt = torch.stack([(p * (p < p_t[:, k:k + 1])).sum(-1) for k in range(track.shape[1])], dim=1)
    mass_le = torch.stack([(p * (p <= p_t[:, k:k + 1])).sum(-1) for k in range(track.shape[1])], dim=1)
    pp = torch.softmax(processed_logits.float(), dim=-1)
    p_post = pp.gather(1, track)
    ent = -(p * lp).sum(-1)
    top1_p, amax = p.max(dim=-1)
    is_top = p_t >= top1_p_of(p)[:, None]
    elig_lo = ((mass_lt + p_t) > (1.0 - top_p)) | is_top
    elig_hi = (mass_le > (1.0 - top_p)) | is_top
    return dict(lp_pre=lp_t, mass_lt=mass_lt, mass_le=mass_le, elig_lo=elig_lo, elig_hi=elig_hi,
                p_post=p_post, entropy=ent, argmax=amax, top1_p=top1_p, argmax_is_forced=amax == forced_tok)
