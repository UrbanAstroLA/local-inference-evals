# P4: two-rank MoE tile intervention and its propagation to the target distribution (protocol)

Status: **the protocol as approved and run** (run 1, 2026-10-24; result `RESULT.md`, critique `CRITIQUE-run1.md`). It
answers the recommended next deliverable of the independent assessment of 22aq-22at. It was approved by the operator
(ceiling 4.0, later 7.0 GPU-hours) and not published before the run. Amendments made before any GPU step are in
`PROPOSAL-AMENDMENTS.md`. A section on later behavioural work is omitted.

## 1. Question and what is not claimed
**Question.** On stock 0.9.1, does forcing the routed-expert decode tile to K64/N256 (arm B), where the default
selector picks K128/N128 (arm A), change the target model's next-token distribution, compared on identical accepted
histories and identical serving shapes?

**Not claimed, whatever the result:**
- an effect on completion;
- an explanation of the 0.7.0-0.9.1 gap;
- anything about EP2/TP2 partitioning, D2 or DCP2.

A changed distribution establishes **computational propagation** only. Behaviour is a separate, later experiment
(section 8).

## 2. The intervention (exact implementation)
**Source.** Stock `exl3.py:2480` sets `tile_config = None` for projection-mixed EXL3 MoE. B12x then chooses
K128/N128 when `10 <= token_count <= 32` and the routes are packed, and K64/N256 otherwise
(`_impl.py:8310-8331`). That rule applies to every MoE layer on both ranks.

**Patch.** One derived image for both arms, so the arms differ only in an environment variable:
- line 2480 becomes `tile_config = _p4_forced_tile()   # local-p4`;
- the helper returns `None` unless `GLM53_P4_TILE=64,256,64,256`;
- with the variable unset, the code path and every launch are identical to stock (checked: gate G3 below).

**The tile is bound at weight planning**, in `process_weights_after_loading`, so each arm is its own server session.

**What B changes:** only the decode launches at 10-32 rows. Launches at m ≤ 9 and the 1,024-row prefill launch
already use K64/N256. This is to be verified, not assumed.

**Kept fixed:**
- K3.25 weights (snapshot `0490d2f7`) and loading;
- TP2 / DCP1 / EP1;
- DFlash2 K3, sampling, graph capture sizes, plan capacities and every other runtime setting;
- stock D2 and causal-length behaviour (no correction applied, per the assessment).

## 3. Launch verification (both ranks)
An observe-only hook at `_bind_rank_sliced` records the bound launch per (rank, layer, m): `size_m`, fc1/fc2 tile K/N,
`moe_block_size`, plan, capacity. It is the same field set as `captured-p3b`, extended to rank 1 and to all MoE layers.

**Gates before any measurement is read:**
- **G1:** arm B binds K64/N256 for every decode launch with 10 ≤ m ≤ 32, on both ranks and all MoE layers.
- **G2:** for every other (rank, layer, m), arm B binds exactly what arm A binds (tile, block, capacity).
- **G3:** arm A's bound launches equal those recorded from the unpatched stock image in `captured-p3b`, rank 0.
- **G4:** arm A's MoE output on one captured `captured-p3b` row set reproduces the served stock output bit-equal. This
  is an offline single-GPU check of the patched image with the variable unset.

## 4. Matched histories and state construction
**Tool.** The forcewin force-history hook (amendment 21; `hook-validation/`, validated on the B1 image), ported to
stock 0.9.1 alongside the P4 patch.
- It forces each request to commit an exact token history F through the served verify path (K + 1 = 4 rows per
  request, CUDA graphs, tail ring, KDA rollback).
- It records the target row for each position once, on the intended prefix, before forcing changes the decision.

**Acceptance schedule.** One fixed schedule (`sched`: the accepted counts of a recorded natural run) is used
identically in both arms.
- With 8 requests in flight, every verify step is 32 rows in both arms. The MoE tile difference is therefore
  exercised at the served shape, and both arms see the same batch composition and positions step for step.
- The step log records shape and composition; positions whose step shapes differ between arms are excluded and
  counted.

**State construction (documented per window).**
- The prompt and the history up to the window start P are built by prefill. The prefill launch is K64/N256 in both
  arms, so the starting state is the same arithmetic in A and B.
- The window [P, P + W) is then forced through decode in each arm, so each arm's state inside the window is built by
  its own decode arithmetic on identical tokens.
- Positions are compared only as "same accepted history, each arm's own state". Free-running generations are never
  compared at matching indices.

**Histories (local; benchmark-derived; never published).** Recorded stock GPQA replies already on disk, selected by a
rule fixed before data:
- 8 replies that finished after 100k tokens and 8 that were capped, from the stock 0.9.1 records;
- per reply, three windows of W = 512 tokens:
  - early, at P ≈ 16k;
  - deep, at P ≈ 96k;
  - near-close: the 512 tokens before the reply's `</think>`, or, for capped replies, before the cap.
- That is 48 windows and about 24.6k forced positions per pass.

## 5. Measured endpoints (per position, arm A vs arm B, against within-arm repeats)
| # | Endpoint | Recorded by |
|---|---|---|
| E1 | MoE output after the TP all-reduce: rel L2 and bit-equality per (layer, row), at layers 3, 23 and 44 | new graph-safe ring (`captured-p3b` pattern), both ranks' combined output on rank 0 |
| E2 | Propagation: final-norm hidden state (and layers 12 and 34), rel L2 | same ring, every 8th position |
| E3 | Target distribution, full-normalized: Δ ln P(forced token), Δ ln P(`</think>`), Δ ln P(end-of-turn ids), entropy, argmax change, top-1 p, nucleus eligibility (top_p 0.95) of the top tokens | forcewin row metrics (42 columns) |
| E4 | Full-vocabulary KL / TV on a sampled subset (every 64th position, about 400 per pass; full fp32 logits, about 620 KB each) | logits ring |
| E5 | Runtime: step time, graph mode, occupancy; acceptance (forced, so fixed by design) | step log |

A distance over renormalized intersecting top-k lists is reported only as a restricted diagnostic, never as the
endpoint.

**Within-arm controls:**
- every window is run twice per session (fresh requests; the prefix cache reuses the prefill-built prefix, which is
  identical by construction);
- a third session repeats arm A, so session-to-session reproducibility of A is measured directly.

**Session order:** A, B, A.

## 6. Decision rules (adopted from the assessment)
| Result | Reading |
|---|---|
| G1-G4 fail | INCOMPLETE; repair; no reading |
| E1 differs but E3 shows no A-vs-B difference beyond A-vs-A | No demonstrated propagation on these windows; investigate attenuation before any behavioural run |
| E3 differs reproducibly (A ≠ B where A = A, on the same positions in both repeats) | Computational propagation established; behavioural direction untested |
| E3 differences include terminal-token log-probabilities near close | Reported as such, still not a completion claim |

No threshold dismisses a reproducible small difference. A changed argmax demonstrates sensitivity only.

## 7. Cost
**Engineering (CPU), about 2-3 days:**
- port forcewin to stock 0.9.1 (its validated pieces carry over; the stock call sites are re-checked);
- add the tile switch, the both-rank launch recorder, and the MoE and hidden-state rings;
- window selection and fixture build;
- CPU tests: hook logic, graph-safety simulation, poller start after graph capture (the `captured-p3b` attempt-1
  lesson), runner gates, mock session, a DRY_RUN;
- a single-GPU check of graph capture with all hooks before any serving run (about 0.01 GPU-h).

**GPU, per session** (2 GPUs):

| Step | Wall time |
|---|---|
| Start-up | about 12.5 min |
| Prefill of 48 windows' prefixes (about 2.1M tokens at about 2,400 tok/s), partly cached on the repeat pass | about 15 min |
| Forced decode (2 × 24.6k positions at 8 in flight, about 4 positions per step) | about 5 min |
| Margin | about 3 min |
| **Total** | **about 35 min, about 1.2 GPU-h** |

**Three sessions (A, B, A):** about 3.5 GPU-h. **Proposed ceiling: 4.0 GPU-h**, including retries.

**Smaller option** (sessions A and B only, in-session repeats only): about 2.4 GPU-h, ceiling 3.0. It loses the
reversal and session-level control.

## 9. Privacy
Histories, logits and hidden states derive from GPQA replies and stay under a `PRIVATE-` directory (0700). Public
output, if approved, is limited to aggregates, shapes, digests and code.
