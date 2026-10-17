# Stock 0.7.0 replication at fresh seeds: result (amendments 22m-22q)

**Reading: O3, inconclusive.** Stock tpurtell 0.7.0 on K3.25 left **10 of 594** GPQA Diamond replies empty at fresh
request seeds. Its own three-pass record left 4, and the 0.9.1 record 16. The run separates from neither under the
frozen rules. The run is valid on every frozen check.

## Run
- **Design:** variant F (request seeds 70701 / 70702 / 70703; lm-eval process seeds 1235 / 1236 / 1237 as in every
  record). Two sessions as the record: pass 1, then passes 2-3 on a fresh server, through the recording proxy.
- **Package:** `refreeze/`, SHA256SUMS sha256 131e72ea…, verified at launch (`run-F/package-verify.txt`).
- **Cost:** **23.926 occupied GPU-hours** of the 36.5 authorized (session 1 6.693, session 2 17.233). No watchdog
  stop.
- **Server:** both sessions were configuration-identical to the old record's passes 2-3 server log (non-default
  arguments, vLLM version, snapshots, logged environment, CUDA-graph mode). Each had 9 release warm-ups before the
  first request, no preemption, no illegal-memory-access or NaN lines. The compile cache was unchanged apart from one
  file in session 1.
- **Requests:** all 594 returned HTTP 200 with the variant's seed, cap 327,680, temperature 1.0, top_p 0.95, with no
  re-sent request. The task hash and every prompt, doc, target and message hash equal both records'.

## Primary (frozen rules; `run-F/result-rep070-F-final.md`)

| Comparison | Empty replies | One-sided exact sign-flip p | Effect, 95% question bootstrap |
|---|---|---:|---|
| Replication | 10 of 594 (2 / 4 / 4 by pass) | | |
| C1: fewer than the 0.9.1 record (16) | | 0.122 | 1.01 points fewer (−0.34 to +2.53) |
| C2: more than the old 0.7.0 record (4) | | 0.108 | 1.01 points more (−0.17 to +2.36) |

- **The rule:** O1 needed ≤ 8 empties and p1 < 0.05; O2 needed ≥ 12 or p2 < 0.05. Neither holds, so the reading is O3:
  the prespecified evidence is inconclusive.
- **Assumptions:** both tests assume that, under the null, each question's pair of empty counts is exchangeable with
  the historical record's, and that questions are independent units. "Exact" refers to the calculation under those
  assumptions; the comparators are historical, so nothing is randomized.
- **Power:** the design was not powered to locate an intermediate rate. Its power analysis showed a halfway engine
  splitting across all three outcomes.

## Descriptive
- **Accuracy:**

  | | Raw (flexible-extract) | Stated (audited) |
  |---|---|---|
  | Replication | 87.2% | **87.9%** (522 / 594) |
  | 0.9.1 record | 86.9% | 87.9% |
  | Old 0.7.0 record | 88.2% | 89.4% |

  The stated figure is final: all 13 rows where it differed from flexible-extract were read by hand, and 3 replies
  committing to no single option are scored incorrect as stated.
- **Repeatability:** no answered reply is identical to the old record in any pass; one question-pass is empty in both.
  Log reply-length correlation with the old record at matched passes is 0.25-0.49, against 0.26-0.44 at mismatched
  passes, the different-seed level.
- **Secondary, P(finish | still generating at 100,000 tokens):** descriptive, never the reading.

  | | Finished / at risk | Questions | P |
  |---|---|---:|---:|
  | Replication | 19 / 29 | 18 | 0.66 |
  | 0.9.1 record | 10 / 25 | 15 | 0.40 |
  | Old 0.7.0 record, passes 2-3 | 10 / 14 | 11 | 0.71 |

  - **Differences:** +0.26 (95% +0.01 to +0.53) against 0.9.1, and −0.06 (−0.35 to +0.27) against the old 0.7.0
    record.
  - **Limits:** about 10-19 finishing events per record and wide intervals mean this cannot adjudicate a termination
    mechanism. Reaching 100k selects different trajectories on each engine, so the conditional comparison is not a
    causal engine effect.

## What it means, and what follows (amendment 22q)
- **What it suggests:** the original 4-versus-16 contrast looks partly like a favourable draw for 0.7.0. At fresh
  seeds its empty count more than doubled but stayed below the 0.9.1 record's, without separating from either.
- **What it does not show:** that no advantage remains, or that one does.
- **The historical-gap line closes here,** as the O3 next step allows. The verified correctness fixes keep their
  value independently.
- **Next:** the long-reply completion mechanism is taken up as its own question. The descriptive finishing rate is
  the measure on which 0.7.0 still looks different.

## Files
- `run-F/`: frozen results, provisional and final (`result-rep070-F*.{md,json}`), cost ledger, compile-cache
  inventories and diffs, GPU log, watchdog log, seeds planned, package verification, completeness probe.
- `run-F.out`: the runner log.
- Server logs, `docker inspect` output and the local raw request / response bodies (reasoning and final text) are
  not published. The first two are listed by hash; the raw bodies are kept locally only.
