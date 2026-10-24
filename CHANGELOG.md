# Changelog

## 2026-10-24, MoE tile intervention, run 1: INCOMPLETE (completion investigation, amendment 22av)

2.53 GPU-hours (cumulative 133.30).

- **New `tile-p4/`:**
  - `PROTOCOL.md`, `PROPOSAL-AMENDMENTS.md`, `RESULT.md` (INCOMPLETE) and `CRITIQUE-run1.md`;
  - the image build, runner, pre-check, G4, analysis and CPU tests;
  - run records and `diagnostics/`.
  - Benchmark-derived records (`PRIVATE-*`), test scratch and compile caches are withheld
    (`RAW-SHA256SUMS-tile-p4.txt`).
- **`PLAN.md`:** amendment 22av.
- **Investigation `README.md`:** a Status entry.
- **`LEDGER.md`**.

## 2026-10-23, wording corrections (completion investigation, amendment 22au)

No GPU use.

- **Accuracy wording:** "No accuracy advantage is established" replaces "Neither plan is more accurate".
- **Mechanism scope:** tile choice is the established cause on the tested inputs; the fp16-staging magnitude
  explanation rests on source analysis and CPU emulation.
- **Files corrected in place:** `moe-plan-p3/SOURCE-MECHANISM.md`, `captured-p3b/RESULT.md`, the investigation
  `README.md` and `LEDGER.md`.
- **`PLAN.md`:** amendment 22au.

## 2026-10-23, captured-input replay and the source of the MoE plan difference (completion investigation, amendment 22at)

0.8031 GPU-hours (cumulative 130.77).

- **New `captured-p3b/`:**
  - `PROPOSAL.md` (the protocol) and `RESULT.md`;
  - the capture image build (hooks, gates), runner, client, replay and summary scripts, and CPU and GPU tests;
  - the run records of both capture attempts and both replays;
  - the cost ledger and authorization.
  - Captured tensors (`PRIVATE-capture/`) and compile caches are withheld (`RAW-SHA256SUMS-captured-p3b.txt`).
- **`moe-plan-p3/`:** `SOURCE-MECHANISM.md` and the CPU emulation in `mechanism/`.
- **`PLAN.md`:** amendment 22at.
- **Investigation `README.md`:** a Status entry.
- **`LEDGER.md`**.

## 2026-10-23, routed-expert plan comparison on synthetic inputs (completion investigation, amendment 22as)

0.1458 GPU-hours (cumulative 129.97).

- **New `moe-plan-p3/`:**
  - `PROTOCOL.md` and `RESULT.md`;
  - the harness (`harness/`) and its CPU tests (`tests/`);
  - the run records of attempts 1 (smoke) and 2 (full) and the cost ledger.
  - Withheld: tensor files and the compile cache (hashes in `RAW-SHA256SUMS-moe-plan-p3.txt`).
- **`PLAN.md`:** amendment 22as.
- **Investigation `README.md`:** a Status entry.
- **`LEDGER.md`**.

## 2026-10-23, the slot-0 condition of level 3 (completion investigation, amendment 22ar)

No GPU use.

- **`livecheck-22ao/level3/`:**
  - `P2-SLOT0-AUDIT.md` (the bounded audit);
  - `P2-WRITER-CENSUS.md` with `p2-census/` (scripts and output);
  - `README.md` and `RESULT.md` updated with the outcome.
- **Erratum:** a runner citation in `level3/SOURCE-READING.md`, annotated in place.
- **`PLAN.md`:** amendment 22ar.
- **Investigation `README.md`:** the Status entry.
- **`LEDGER.md`**.

## 2026-10-23, level 3 of the causal-length check (completion investigation, amendment 22aq)

0.0022 GPU-hours (cumulative 129.82).

- **New `livecheck-22ao/level3/`:**
  - `README.md` (the three conclusions and their scope) and `SOURCE-READING.md`;
  - the CPU compaction harness and its output;
  - the GPU kernel test (`kernel_eq.py`, `run_level3.sh`), its result, log and cost line.
- **Correction:** the 22ap rule-of-three bounds are nominal (rows are dependent). Wording updated in the investigation
  `README.md`, `PLAN.md` and `LEDGER.md`.
- **`PLAN.md`:** amendment 22aq.
- **Investigation `README.md`:** the Status entry.
- **`LEDGER.md`**.

## 2026-10-23, live causal-length check run (completion investigation, amendment 22ap)

1.51 GPU-hours (cumulative 129.82).

- **`livecheck-22ao/run-1/`:** both arms' guard, client and server records, the frozen analysis (`result.md`) and
  notes (`NOTES-22ap.md`). The port-01 arm's first attempt is kept as `arm-c.attempt1-guard-race/` with its analysis
  (`result.attempt1.*`). Guard cache-byte snapshots are redacted (`withheld-kv-bytes`); local hashes of every changed
  copy are in `PUBLIC-COPIES-22ap.txt`.
- **`livecheck-22ao/AMENDMENT-22ap-preflight.md`:** the pre-flight change, the arm C continuation and retry, and the
  guard fix, with the frozen copies of each changed file (`*.frozen-22ao`, `*.22ap-attempt1`) and `AUTHORIZATION.txt`.
- **Guard fix:** `guard/make_guard_22ao.py`, `build/Dockerfile.C-22ap`, `tests/test_22ap_poll_race.py` and its
  output; the 22ao guard tests re-run (`tests/out/test-22ao-22ap.*`, identical).
- **`PLAN.md`:** amendment 22ap.
- **Investigation `README.md`:** the Status entry.
- **`LEDGER.md`**.

## 2026-10-23, live causal-length check frozen (completion investigation, amendment 22ao)

No run is added; no GPU use.

- **New `livecheck-22ao/`, the frozen package:**
  - `PROPOSAL.md` (the pre-registration) and `FREEZE.md`;
  - the image build inputs: Dockerfiles, hooked files, port 01;
  - the amended guard and its builder;
  - the client, runner and analysis;
  - the CPU tests and their outputs;
  - `SHA256SUMS`.

  The GPQA panel fixture is withheld (hash in `RAW-SHA256SUMS-livecheck-22ao.txt`).
- **`PLAN.md`:** amendment 22ao (adopts the assessment of 22an).
- **Investigation `README.md`:** a Status entry.
- **`LEDGER.md`**.

## 2026-10-22, review of the reconnaissance (completion investigation, amendment 22an)

No run is added; no GPU use.

- **`PLAN.md`:** amendment 22an (the independent review's disposition; reporting corrections; additions).
- **`recon-2026-10-22/`:**
  - `f1/result.md`: one- and two-sided exact permutation p; Monte Carlo label; reaching-100k wording;
  - `dispatch/`: client-side in-flight caveat; throughput field renamed (whole-request average); MoE launch geometry
    corrected; section 2a, the causal-length source for verify rows;
  - `draft-slot/`: deterministic lookahead control (new `lookahead_control.py` and result).
- **Wording, reaching 100k:** investigation `README.md`, `FINDINGS.md`, `LEDGER.md`.
- **`long-reply/README.md`:** the question-matched figure's caveat restored (the 0.9.1 family pools modified arms).
- **`LEDGER.md`**.

## 2026-10-22, reconnaissance after close (completion investigation, amendment 22am)

No run is added; no GPU use.

- **New `recon-2026-10-22/`:**
  - input manifest and stock comparisons (original, fresh-seed and pooled; post hoc);
  - per-version dispatch table;
  - draft-slot CPU reproduction on the stock 0.9.1 image;
  - coverage map of earlier checks.

  Engine source copies are withheld (hashes of cited files published).
- **`PLAN.md`:** amendment 22am.
- **Investigation `README.md`:** a Status entry for the reconnaissance; erratum, the sign in "B0 across three seeds"
  (a reduction of 0.84 points).
- **`long-reply/discovery/REPORT.md`, `long-reply/README.md`:** erratum, loop onsets after 100k were 6, not 8; 2
  replies never formed a terminal loop.
- **`FINDINGS.md`:** the 0.7.0 against 0.9.1 open-question row gains the pooled stock figure.
- **`LEDGER.md`**.

## 2026-10-22, upstream (completion investigation)

Contents unchanged.

- **Investigation `README.md`:** the Status and Next steps entries for upstream now link the submitted corrections
  (tpurtell PRs #7-#11; vLLM issue #63168) instead of describing them as held drafts.

## 2026-10-22, final (completion investigation): status sections brought up to date

Contents unchanged.

- **Investigation `README.md`:** the Status paragraph, which had stopped at amendment 20, is replaced with the status at
  close. The Answer table gains rows for the stock 0.7.0 replication and the long-reply line. Next steps now state that
  nothing is scheduled and list the open questions.
- **Repo `README.md`:** the completion row and summary run through the replication and the close-out.

## 2026-10-22, close-out (completion investigation, amendment 22al): long-reply line closed

Contents unchanged. No run is added.

- **`PLAN.md`**: amendment 22al (independent review's disposition: the long-reply allocation closes as unresolved,
  with positive findings; summary wording; answer audit over all attempts; requirements for any future interaction
  study).
- **New `long-reply/CLOSEOUT.md`**; `long-reply/README.md` gains its status at close.
- **Open-question rows updated:** investigation `README.md` (long-reply termination: closed; the pooled B0 + B1 0.12
  figure superseded) and `FINDINGS.md` (the 0.7.0 vs 0.9.1 empty-reply question: replication O3, both lines closed).
- **`LEDGER.md`**.

## 2026-10-22, later (completion investigation, amendment 22ak): prefix-source cross (E) result

Contents unchanged. One investigation run is added.

- **`PLAN.md`**: amendment 22ak (E: valid, R-I).
- **`long-reply/e-cross/`**: run outputs (request logs, frozen result, cache inventories, GPU and watchdog logs) and
  `audit-22ai/` (the answer audit: counts only). Raw outputs, prefixes, server logs and docker inspect output are
  withheld. Substitutions are in `PUBLIC-COPIES-22ak.txt`.
- **`costs.jsonl`** (10.2445) and **`recompute.py`**: cumulative 128.31.
- **`long-reply/README.md`**, investigation `README.md` and **`LEDGER.md`**.

## 2026-10-22, later (completion investigation, amendments 22ai-22aj): review disposition for E

Contents unchanged.

- **`PLAN.md`**: amendment 22ai (independent review: E proceeds unchanged; interpretation corrections; a fixed
  result-statement table; a supplementary answer audit; B+ citation wording; C held for a result-dependent review) and
  22aj (authorization).
- **Wording corrected** in the investigation `README.md`, `LEDGER.md` and `long-reply/README.md`: B+ is now stated in
  the review's citation wording; "identical states" becomes token prefixes reconstructed by each engine's prefill; η²
  describes between-prefix variation, not a causal share.

## 2026-10-22 (completion investigation, amendments 22ac-22ah): second 0.9.1 text pass; E frozen

Contents unchanged. One investigation run is added.

- **`PLAN.md`**: amendments 22ac-22ah (E's first freeze and its power; the decision to add prefixes; D2 package,
  authorization and result; E re-frozen on 13 prefixes).
- **`long-reply/`**: `d091b/` (D2) and `e-cross/` (the frozen E package), with the README updated. Withheld as before,
  plus E's superseded `.pre22ah` copies. Substitutions are in `PUBLIC-COPIES-22ah.txt`; a catch-all substitution for
  local paths was added.
- **`costs.jsonl`** (7.1633) and **`recompute.py`**: cumulative 118.07. The option-D check now names its step exactly
  (its prefix had also matched D2).
- **Investigation `README.md`** and **`LEDGER.md`**.

## 2026-10-21 (completion investigation, amendments 22r-22ab): long-reply completion

Contents unchanged. Two investigation runs are added (not protocol runs in `runs/`).

- **`PLAN.md`**: amendments 22r-22ab (discovery; B+ package, authorization, INVALID verdict, disclosed gate amendment,
  result; option D package, authorization, result; the decision to run option E).
- **New `long-reply/`**: `README.md`, `discovery/`, `bplus/` and `d091/`. Withheld: prefixes, prompts, token ids, raw
  bodies, server logs, docker inspect output, the seed-scan file inventory and the local eval-chain copy (by hash).
  Substitutions are in `PUBLIC-COPIES-22ab.txt`.
- **`costs.jsonl`** (13.0639 and 5.8844) and **`recompute.py`**: cumulative 110.90.
- **Investigation `README.md`** and **`LEDGER.md`**.

## 2026-10-17 (completion investigation, amendments 22o-22q): stock 0.7.0 replication result

One investigation run is added (not a protocol run in `runs/`). No published row changes.

- **`PLAN.md`**: amendments 22o (authorization), 22p (result: valid, O3) and 22q (stated figure final; the
  historical-gap line closes; long-reply completion taken up separately).
- **`rep070/`**: `RESULT.md`, the runner log and `run-F/`: frozen results (provisional and final after the hand read),
  cost ledger, compile-cache inventories and diffs, GPU and watchdog logs. Server logs and `docker inspect` output are
  withheld by hash; raw request and response bodies stay local. Substitutions are in `PUBLIC-COPIES-22q.txt`.
- **`costs.jsonl`** (23.9261) and **`recompute.py`**: cumulative 91.95.
- **Investigation `README.md`** and **`LEDGER.md`**.

## 2026-10-16 (completion investigation, amendments 22m-22n): stock 0.7.0 replication frozen before data

Contents unchanged. No run is added.

- **`PLAN.md`**: amendments 22m (the re-frozen replication package) and 22n (independent review's disposition: fresh
  seeds, 36.5 GPU-hour watchdog, incomplete-run handling, category meanings, local raw records).
- **New `rep070/`**: the 2026-10-11 draft and `refreeze/`, the frozen package (proposal, analysis, runner, proxy copy,
  incomplete-run report, power and budget models, dry runs, tests). Superseded copies, the seed-scan file inventory and
  the local eval-chain copy are withheld by hash; the chain copy's frozen diff is published. Substitutions are in
  `PUBLIC-COPIES-22n.txt`.
- **Investigation `README.md`** and **`LEDGER.md`**.

## 2026-10-16, later (completion investigation, amendments 22k-22l): contrast line closed; mechanism ranking

Contents unchanged. No run or reading is added.

- **`PLAN.md`**: amendments 22k (independent review's disposition after the retry) and 22l (the zero-GPU mechanism
  ranking and the operator's decision).
- **New `first-contrast/retry/review-22k/`**: the bounded CPU review of the failed gates and the prospective validation
  design.
- **New `mechanism-ranking/`**: the ranking, its sources with hashes, and CPU receipts (tail-ring harness exposure runs,
  served speculative-decoding metrics, a long-reply power simulation). Substitutions are in `PUBLIC-COPIES-22l.txt`.
- **Investigation `README.md`**: disposition, ranking, and new evidence in the step-6 open question. **`LEDGER.md`**.

## 2026-10-16 (completion investigation, amendments 22f-22j): first contrast retry, no reading

Contents unchanged. No reading is added.

- **`PLAN.md`**: amendments 22f (independent review of the retry package), 22g-22h (authorization: variant B, 6.2
  GPU-hour ceiling, batch guard), 22i (the frozen package) and 22j (the result).
- **`first-contrast/retry/`**: the package as frozen after 22f-22i, the session `run-fcr1/` (tensor records listed by
  count only; container logs and pre-flight listings by hash) and the post-hoc `diagnosis-fcr1/`. Substitutions are in
  `PUBLIC-COPIES-22j.txt`.
- **`costs.jsonl`** (2.9317) and **`recompute.py`**: cumulative 68.03.
- **Investigation `README.md`** and **`LEDGER.md`**.
- Removed `run-hv-a2.out` from the repository root: a one-line shell error committed by mistake on 2026-10-14.

## 2026-10-15, later (completion investigation, amendments 22d-22e): retry package for review

Contents unchanged. No run is added.

- **`PLAN.md`**: amendments 22d (independent review of the retry) and 22e (the retry package; a correction to 22c's
  account of the M3 eviction).
- **New `first-contrast/retry/`:** the cache replay through the image's own code, the feasible schedule, gates, F6
  calibration, runner, client, analysis and tests. Raw synthetic test data is withheld. Substitutions are in
  `PUBLIC-COPIES-22e.txt`.
- **Investigation `README.md`** and **`LEDGER.md`**: the correction and the retry status.

## 2026-10-15 (completion investigation, amendments 22-22c): first contrast attempt 1 stopped

Contents unchanged. No reading is added.

- **`PLAN.md`**: amendments 22 (first-contrast design from independent review), 22a (authorization), 22b (the
  structural stop) and 22c (diagnosis and proposed fixes).
- **`first-contrast/`** gains the frozen package, the run's records (tensor records listed by count only) and
  `diagnosis-fc1/`. Substitutions are in `PUBLIC-COPIES-22c.txt`.
- **`costs.jsonl`** (1.0861) and **`recompute.py`**: cumulative 65.10. The first push of this entry left five
  historical cumulative checks failing, because they excluded later entries by name. They now sum the entries before a
  named step in ledger order, and every check passes.
- **Investigation `README.md`** and **`LEDGER.md`**, including the owned planning errors.

## 2026-10-14, later (completion investigation, amendments 21g-21k): hook validation attempt 2, first-contrast proposal

Contents unchanged: 65 runs, 13 comparisons. Item 4o is new.

- **`PLAN.md`**:
  - 21g (independent guidance for attempt 2 and the first contrast);
  - 21h (coordinator interpretation and recorded limits);
  - 21i (frozen package);
  - 21j (attempt-2 result: revised VALID, original INVALID);
  - 21k (follow-ups; the first-contrast proposal).
- **`hook-validation/`** gains `prep-a2/`, `run-a2/` and `followup-a2/`.
- **New `first-contrast/`:** the proposal, position plan, exclusion-rule file, prior null and costing.
- **Substitutions** are listed in `PUBLIC-COPIES-21j.txt`. The forcing tensor records of both runs are listed by size
  only.
- **Investigation `README.md`:** the attempt-2 result and next step.
- **`costs.jsonl`** (attempt 2, 0.795) and **`recompute.py`**: cumulative 64.01.
- **`FINDINGS.md`** (item 4o) and **`LEDGER.md`**.

## 2026-10-14 (completion investigation, amendments 21-21f): review result, K, hook validation attempt 1

Contents unchanged: 65 runs, 13 comparisons. Item 4n is new.

- **`PLAN.md`**:
  - amendment 21 (revised first stage);
  - 21a (B0 seed-1236 result);
  - 21b-21d (K: authorization, diagnosis and post-inspection fixes, results);
  - 21e (frozen hook-validation package; F7 test change);
  - 21f (attempt-1 result).
- **Folders:**
  - `b0-screen-s1236/` gains its run;
  - `k-selection/` and `hook-validation/` are new.
  
  Binary tensors, engine-source copies, logs and the 32 GB forcing records are withheld and hashed or listed in
  RAW-SHA256SUMS files. Substitutions are in `PUBLIC-COPIES-21.txt`.
- **Investigation `README.md`:** the review-step result, and a new section "Instrument validation: K and the
  force-history hook (amendment 21)".
- **`costs.jsonl`**: three entries (5.866, 0.2825, 1.2117). `recompute.py` checks them and the cumulative 63.22.
- **`FINDINGS.md`** (item 4n) and **`LEDGER.md`**.

## 2026-10-13, night (completion investigation, amendment 20d): discovery reports and probe proposal

Contents unchanged. No run is added.

- **New folders:**
  - `fixture-inventory/`: existing histories, loop/exhaustion/close classes and termination-token semantics;
  - `source-review-lc/`: the focused review of context-length-dependent paths;
  - `probe-proposal/`: the first-stage force-decoded window proposal, with feasibility checks, fixture selection and
    costing.
  
  Ids, positions, metrics and hashes only. Substitutions are listed in `PUBLIC-COPIES-20d.txt`.
- **`PLAN.md`** (amendment 20d), **`LEDGER.md`** and the completion **`README.md`** (next steps) link them. No GPU use
  is authorized.

## 2026-10-13, night (completion investigation, amendment 20c): tail-accounting audit

Contents unchanged. No run is added.

- **New `tail-audit/`**: the stock-separate audit of the tail analysis required by amendment 20. Script, output, JSON
  and AUDIT.md; counts, ids and flags only. Substitutions are in `PUBLIC-COPIES-20.txt`.
- **Correction (owned):** "6 of about 100 still running beyond 150k" double-counted requests across bands; the unique
  count is 6 of 36. Fixed in the completion README, with a note.
- **Completion `README.md`**, **`LEDGER.md`** and **`PLAN.md`** (amendment 20c): stock-only figures, with the hypothesis
  standing at small counts and no range claimed.

## 2026-10-13, night (completion investigation, amendment 20b): non-completion is not looping

Contents unchanged. No result is added.

- **`PLAN.md`**: amendment 20b. Lower finishing probability is not higher looping probability; fixtures with text are to
  be classed as loop, exhaustion or natural close; the probe proposal states each fixture's state at its windows.
- **Completion `README.md`** (tail section) and **`LEDGER.md`**: the distinction stated, with the existing evidence (the
  pilot's capped replies were not loops; question 79 mostly exhausts, question 88 loops).
- **Looping `README.md`**: a status note pointing to the completion investigation, keeping loops and exhaustions
  distinct.

## 2026-10-13, late evening (completion investigation, amendment 20): plan, status and a correction of emphasis

Contents unchanged: 65 runs, 13 comparisons. No result is added.

- **`PLAN.md`**: amendment 20 (long-context termination localisation: finish the B0 review; zero-GPU discovery; a
  bounded fixed-history probe proposal; an execution-path control; interpretation and extension rules).
- **Correction of emphasis:** the tail analysis (amendment 19f) is reworded throughout as exploratory and
  hypothesis-generating, not a localisation of the gap. FINDINGS item 4m is regraded and its numbers given for stock
  0.9.1 alone. "The D7 path cannot run" is reworded as a reconstruction. B1's pass is noted as one pass under review.
- **`LEDGER.md`**: a "where things stand" entry (established / not established / in progress / hypothesis) and the
  owned correction.
- **Investigation `README.md`**, **`FINDINGS.md`**, **`tools/site.py`** and **`docs/`** updated accordingly.

## 2026-10-13, evening (completion investigation, amendments 19c-19f): B1 result, review step, long-context tail

Contents unchanged: 65 runs, 13 comparisons. Items 4l and 4m are new.

- **`PLAN.md`**: amendments 19c (B1 result), 19d (B0 seed-1236 authorization), 19e (frozen B0 package; identity-count
  correction) and 19f (exploratory tail analysis).
- **New folders:**
  - `b1-screen/`: runner, frozen analysis and interpretation, dry runs, result, hand read, post-hoc token reading.
    Server log and inspection are hashed; the GPQA samples and request log stay local.
  - `b0-screen-s1236/`: the package frozen before generation; its run was in progress at publication.
  - `tail-analysis/`: script, output and README.
  - Substitutions and two shortened hand-read notes are listed in `PUBLIC-COPIES-19.txt`.
- **Investigation `README.md`:** new section "B1 checkpoint pass and the long-context tail (amendment 19)"; status, open
  questions and next steps updated.
- **`costs.jsonl`**: the B1 pass (6.33). `recompute.py` checks it and the cumulative 55.85 GPU-hours.
- **`FINDINGS.md`** (items 4l-4m), **`LEDGER.md`**, **`tools/site.py`** and the rebuilt **`docs/`**.

## 2026-10-13, afternoon (completion investigation, amendments 19-19b): B1 GPQA pass running; a correction

Contents unchanged: 65 runs, 13 comparisons. No result is added yet.

- **`PLAN.md`**: amendments 19 (B1 checkpoint benchmark: preparation), 19a (frozen package: runner, analysis,
  interpretation and hashes; corrections) and 19b (launch authorization; upstream contributions stay drafts).
- **Investigation `README.md`**:
  - status and next steps say the B1 pass is running, results pending;
  - the seed-1235 table's 0.7.0 stated count is corrected from 178 to 177, matching the published rows, with a
    correction note.
- **`LEDGER.md`**: checkpoint status, the running pass and the correction.

## 2026-10-13, later (completion investigation, amendments 18-18c): D7 corrected and verified (B1)

Contents unchanged: 65 runs, 13 comparisons. No earlier run or number changed. Item 4k is new.

- **`PLAN.md`**: amendments 18 (D7 review), 18a (step-2 conditions; server-log correction), 18b (operator authorization:
  2.0 GPU-hours, one retry) and 18c (frozen package; seed correction).
- **New `d7/` folder:**
  - **Published:** the release and upstream comparison, the correction (`d7-fix.diff`) and its CPU tests, the GPQA
    exposure reconstruction (simulator, tokenizer, classes; token counts, hashes and offsets only), the frozen step-2
    package (guard, analysis, runner, client, CPU tests, build files), and the B1 session's verdict, phase 2d and gzip
    guard records.
  - **Hashed in `RAW-SHA256SUMS-d7.txt`:** fixture token ids, probe outputs, container logs, pre-flight listings and
    full engine files.
  - **Substitutions** are listed in `PUBLIC-COPIES-18.txt`.
- **Investigation `README.md`:**
  - new section "D7: correction and verification (amendment 18)";
  - the D7 row in the defect matrix updated;
  - status, open questions and next steps updated.
- **`b0-mapguard/README.md`:** the seed statements are corrected in place, with notes.
- **`costs.jsonl`**: one entry (0.6244). `recompute.py` checks it and the cumulative 49.52 GPU-hours.
- **`FINDINGS.md`** (item 4k, open questions), **`LEDGER.md`**, top-level **`README.md`**, **`tools/site.py`** and the
  rebuilt **`docs/`**.

## 2026-10-13 (completion investigation, amendments 17-17e): live slot mappings on B0

Contents unchanged: 65 runs, 13 comparisons. No earlier run or number changed. Item 4j is new.

- **`PLAN.md`**: amendments 17 (live scheduling and consumed slot mappings on B0), 17a (authorization), 17b (preparation;
  candidate D7 found on CPU before GPU data), 17c (pre-flight fix), 17d (attempt-2 diagnosis; D7 confirmed in serving)
  and 17e (zero-GPU checks before attempt 3; ceiling raised to 1.45).
- **Slot-mapping guard session** (new `b0-mapguard/`): VIOLATION, D7 only. 217 events in 50 requests at index-K prefill
  pools after chunk starts not aligned to 4 tokens; every other mapping clean over the frozen coverage. 0.9933
  GPU-hours, including an invalid attempt caused by the investigation's runner.
  - **Published:** the proposal, the guard module (current and as run in attempt 2), the hook diff and build files,
    the frozen analysis and a post-hoc attribution, the runner and client, CPU tests and their outputs, the allocator
    simulation, per-attempt budgets, gate output, client logs, verdicts and gzip-compressed guard records, the attempt-2
    diagnosis, cost lines and a README.
  - **Hashed in `RAW-SHA256SUMS-mapguard.txt`:** fixture token ids, probe outputs, container logs, pre-flight listings
    and full patched engine files.
  - **Substitutions** are listed in `PUBLIC-COPIES-17.txt`, including the guard module's public name
    `mapguard_hooks.py`.
- **Investigation `README.md`**:
  - new section "Live slot mappings on B0 (amendment 17)";
  - a D7 row in the defect matrix, now D1-D7;
  - status, open questions and next steps updated.
- **`costs.jsonl`**: one entry with three attempts (0, 0.4211, 0.5722). `recompute.py` checks it and the cumulative
  48.90 GPU-hours.
- **`FINDINGS.md`**: item 4j (D7, supported in serving; effect untested); open questions updated.
- **`LEDGER.md`**, top-level **`README.md`**, **`tools/site.py`** and the rebuilt **`docs/`**: D7 added to the
  engine-issues summary and table.

## 2026-10-12, evening (completion investigation, amendments 16-16e): checkpoint B0 and targeted invariants

Contents unchanged: 65 runs, 13 comparisons. No earlier run or number changed. Item 4f's grade is unchanged (descriptive)
with its classification added; items 4g-4i are new.

- **`PLAN.md`**: amendments 16 (checkpoint B0 and targeted invariants), 16a (authorizations), 16b (all-reduce and
  KV-writer results, operator decision on the KV-writer criteria, E1), 16c (cache fixture authorization), 16d
  (pre-flight fix) and 16e (cache fixture results).
- **Two-GPU all-reduce fixture** (new `b0-allreduce/`): CLEAN; 2,040,800 peer-push calls, 0 mismatches in 41 cases on
  both ranks against an exact bf16 reference; 0.07 GPU-hours. Published: proposal, fixture, input patterns, analysis,
  CPU tests, runner, frozen hashes, smoke and full results (per-rank results, plans, GPU log), CPU-check output, cost
  lines, a README; pre-flight process listings, container logs, step schedules and the compile cache are hashed in
  `RAW-SHA256SUMS-allreduce.txt`.
- **KV writer and query projection, and E1** (new `e1-kvwriter/`): within expected rounding under criteria errata
  recorded before GPU data (the operator's decision; the verdict as first written is reported beside it); 0.0069
  GPU-hours. E1 classified as expected numerical variation of the upstream router kernel (CPU only). Published: the
  frozen design, the errata, freeze hashes, input builder, GPU script, analysis (byte-identical to the frozen file),
  loader wrapper, runner, analysis output, GPU logs, the E1 assessment, scripts and outputs, SASS excerpts, a README;
  the prepared inputs, result tensors, pre-flight listings and full SASS listings are hashed in
  `RAW-SHA256SUMS-kvwriter.txt`.
- **Cache isolation and high-offset fixture** (new `b0-cache/`): every driven kernel passed after five fixture faults,
  found in the data, were fixed (no criterion loosened); one upstream test obsolete; one launch refused by the runner's
  pre-flight at zero cost and its check narrowed before any data; 0.0392 GPU-hours. Published: proposal, fixture code as
  fixed, hashes before and after, change notes, reachability simulation, CPU tests, per-run results and consoles, cost
  lines, a README; fixture backups (the versions that ran first) are hashed in `RAW-SHA256SUMS-cache.txt`.
- **`PUBLIC-COPIES-16.txt`**: local and public sha256 of the copies with paths, image prefixes or GPU ids replaced; the
  amendment and freeze hashes refer to the local files.
- **`costs.jsonl`**: three entries appended, one per item (0.07, 0.0069, 0.0392, the last including the refused launch at
  0). Cumulative 47.91 GPU-hours; `recompute.py` checks the new entries and total.
- **Investigation README**: status line, answer table, fast path, source audit table, prior probes, defect matrix (a
  second table for the audited items and E1), the new section "Checkpoint B0 and targeted invariants (amendment 16)",
  observations (E1), open questions (one added, evidence only), limits, next steps (nothing scheduled; no B1).
- **`FINDINGS.md`**: item 4f classified; new items 4g-4i with notes; open questions.
- **`LEDGER.md`**: dated entry for the tranche, with the KV-writer errata, the cache fixture faults and the pre-flight
  refusal.
- **`CONFOUNDS.md`**: entry 17 (E1) status and evidence.
- **Top-level README** and **site** (`tools/site.py`: the engine-issues text); pages regenerated.

## 2026-10-12, afternoon (completion investigation, amendments 15-15c): second bundle pass, seed 1235

Contents unchanged: 65 runs, 13 comparisons. No earlier run or number changed; grades unchanged (item 4e stays
descriptive, now covering two passes); the seed-1234 screen keeps its category.

- **`PLAN.md`**: amendments 15 (proposal), 15a (authorization), 15b (freeze) and 15c (result).

- **Second bundle pass** (new `bundle/screen2/`): the unchanged frozen bundle, one GPQA pass at seed 1235 (protocol pass 2,
  fresh server, the comparators' session conditions), runner and analysis frozen before generation. Valid; 3 of 198
  empty, all at the cap; no material or accuracy concern on the six-pass stock-0.9.1 family standard (raw 168, stated
  169); category MODEST IMPROVEMENT, NO CLEAR SEPARATION. Against the record's seed-1235 pass (5 empty): 2 rescued, 0
  newly empty, raw −8, stated −9; matched-offset readout (not deciding) "concern". Exploratory combination of both
  bundle passes against the matching stock passes: 6 vs 10 empty of 396 (one-sided p = 0.11). Published: runner,
  analysis, arm loaders (unchanged public copy), dry run, results before and after adjudication, adjudication ids and
  verdicts, watchdog and GPU logs, runner output, cost line, a README; the server log, container inspection, samples,
  request log and the local pre-run README are hashed in `RAW-SHA256SUMS-screen2.txt`. 4.541 GPU-hours.
- **`PUBLIC-COPIES-15.txt`**: local and public sha256 of the copies with paths replaced; amendment 15b hashes refer to the
  local files.
- **`costs.jsonl`**: one entry appended (4.541). Cumulative 47.79 GPU-hours; `recompute.py` checks the new total.
- **Investigation README**: status line, answer table (the bundle row covers both passes), fast path, prior probes,
  defect matrix, the screen section (second pass and the exploratory combined table), open questions (one added, as
  evidence only), limits, next steps (nothing scheduled). **`bundle/README.md`**: pointer to the second pass.
- **`FINDINGS.md`**: item 4e and its note, item 7 and its note, open questions.
- **`LEDGER.md`**: dated entry for the second pass.
- **`CONFOUNDS.md`**: new entry 18 (seed reuse across engines).
- **Top-level README** and **site** (`tools/site.py`: summary lines and the engine-issues text); pages regenerated.

## 2026-10-12 (completion investigation, amendments 13a-14e): stage A final, correctness bundle gate, one-pass screen

Contents unchanged: 65 runs, 13 comparisons. No earlier run, number or grade changed except as listed.

- **Stage A final** (`verify-stage-a/`, new `offline-a4/`): D3 controls (a)-(d) pass on all 1,593 records, control (c)
  with the serving-shaped router invocation adopted in amendment 14 before any comparison. The fp32-output router
  computation selects a different top-8 expert set on 3.15% of token × layer rows (95% over 176 requests 3.01-3.29%, over
  8 histories 2.90-3.50%; by layer 1.2-6.6%, by execution type 3.0-3.2%) and different weights on every row; the corrected
  path lies within τ_d of fp64 everywhere. FULL-graph decode covered by source only. Stage A 0.6929 of 2.0 GPU-hours.
  The stage A proposal copy gains its attempt-4 results; code gains the attempt-4 script, its diff and its CPU tests.
- **Bundle gate** (new `bundle/`): D1-D4 on 0.9.1; summaries, diffs, manifests, verdict and analysis outputs, code, D3 as
  served, D4 compile-cache evidence, production-image check, probe divergence (token strings withheld). Raw logs, request
  files, capture and probe records and the local GPQA fixture are hashed in `RAW-SHA256SUMS-bundle.txt`. Two original
  criteria failed as specified and were revised after inspecting gate data (amendment 14c). 1.6403 GPU-hours.
- **One-pass screen** (`bundle/screen/`): valid; 3 of 198 empty, all at the cap; SIMILAR / NO CLEAR SEPARATION under the
  frozen rules; rescued and newly empty replies against both baselines; secondary raw 174 / stated 176 (single pass, not
  a decision input). 4.841 GPU-hours.
- **`costs.jsonl`**: three entries appended (stage A attempt 4 0.0106, bundle gate 1.6403, screen 4.841); the earlier
  stage A entry is unchanged. Cumulative 43.25 GPU-hours; `recompute.py` checks the new totals.
- **`PUBLIC-COPIES-14.txt`**: local and public sha256 of every file whose public copy replaces local paths, image-name
  prefixes or module names; frozen hashes in amendment 14d refer to the local files.
- **Investigation README**: status line, answer table, fast path, prior probes, stage A (final), a D1-D6 defect matrix,
  bundle gate, screen, observations recorded separately (E1, long-context greedy non-repeatability of the stock
  release, NaN-byte index rows), open questions, limits, next steps (nothing further scheduled).
- **`FINDINGS.md`**: items 4c (router precision), 4d (SwiGLU limit), 4e (bundle screen), 4f (E1) and 19a (long-context
  non-repeatability); item 4b, item 7 and its note, open questions.
- **`LEDGER.md`**: the stage A entry moved from the end of the page into the timeline and completed; dated entries for
  the gate and the screen; owned corrections (the D4 compiled-kernel check's unverified assumption in the gate
  preparation; the coordinator's misstatement of the 16,240-token probe's cache coverage; three vs four early stops);
  three rows in "Figures corrected along the way".
- **`CONFOUNDS.md`**: entries 5, 11 and 13 updated; new entries 16 (long-context greedy non-repeatability) and 17
  (batch-size-dependent routing weights, E1).
- **Top-level README** and **site** (`tools/site.py`: summary lines and two rows in the engine-issues table); pages
  regenerated.

## 2026-10-11, late (completion investigation, amendment 12): final stated figure, reconstructed GPU-hours, two corrections

Contents unchanged: 65 runs, 13 comparisons. No run, number or grade changed except as listed; CPU only, no GPU.

- **Completion arm, stated answers final** (`arm-col511-result/`): the 12 arm rows where the stated answer differs from
  raw were read by hand under the protocol's rules; all commit to a single option, none hedged. The frozen script,
  re-run unchanged, gives identical output (`result-col511-final.md`, `result-col511-final.json`): stated 515 vs 522 of
  594, -1.18 points (95% -4.04 to +1.52). The receipts README lists ids, verdicts and categories (no reply text).
- **Occupied GPU-hours reconstructed** (investigation README, new section): start, ready, first and last request and
  stop per server session for the 0.7.0 and 0.9.1 K3.25 records and the arm; 22.97, 16.63 and 17.11 GPU-hours. Replaces
  16.59 (chain start to chain done) for the 0.9.1 record and a planning figure of about 15.5 (one session) for 0.7.0.
  `costs.jsonl` unchanged (cumulative 36.08). `tail-fix/RESULT.md` gains its measured cost beside the estimate.
- **Two interpretation corrections**, owned in `LEDGER.md` (new entry and two rows in "Figures corrected along the
  way"): the corrected 0.9.1 arm resembles the 0.9.1 record's rate but ran a modified engine, so it is not a replication
  of the unmodified record; greater tail-ring corruption in 0.7.0 does not exclude an accidental completion benefit, and
  the defect's role is untested. Edited in the investigation README, `FINDINGS.md` (items 4b and 7, open questions),
  `CONFOUNDS.md` (entries 7, 10 and 12), `LEDGER.md` and `tail-fix/RESULT.md`.
- **Site** regenerated.

## 2026-10-11, evening (completion investigation): completion arm finished; every page updated with its result

Contents unchanged: 65 runs, 13 comparisons. No earlier run, number or grade changed.

- **Completion arm result** (`investigations/2026-10-glm53-completion/arm-col511-result/`): output of the frozen
  analysis script, run unchanged (`result-col511.md`, `result-col511.json` with local paths replaced by placeholders),
  per-session costs and the GPU load log. All validity checks passed. Empty replies 15 vs 16 of 594 (one-sided p = 0.50;
  reduction +0.17 points, 95% -1.01 to +1.35); reading R3: the column-511 correction stands as a correctness fix but is
  not supported as the dominant completion explanation. Raw accuracy 85.0% vs 86.9% (-1.85 points, 95% -5.05 to +1.35),
  not significant. 17.11 GPU-hours; `costs.jsonl` appended (cumulative 36.08).
- **Completion investigation README**: status line; an Answer row for the arm (descriptive; the column-511 localized-defect
  row stays supported); section "Completion arm (running)" renamed "Completion arm" with the result, reading, secondary
  measures, cost and limits; open questions and next steps (nothing further scheduled).
- **Anchor**: links to `#completion-arm-running` now point to `#completion-arm` (FINDINGS, LEDGER, the investigation
  README).
- **`FINDINGS.md`**: item 4a carries the arm result; item 7, its note and the open questions updated.
- **`LEDGER.md`**: "Running." replaced by a pointer to a new entry dated 2026-10-11, evening, with the result and cost.
- **`CONFOUNDS.md`**: entries 7, 9 and 12 and the summary row for 9 updated.
- **Forward pointers**: looping investigation ("Why is open", open question 2), `comparisons/README.md`, the three-pass
  GPQA comparison, the top-level README.
- **Site** (`tools/site.py`): the lines that said an arm was running now give its result; pages regenerated.

## 2026-10-11 (completion investigation, amendments 7-11b): ledger, confounds, findings and site brought up to date; completion arm running

Contents unchanged: 65 runs, 13 comparisons. No earlier run, number or grade changed; no arm result is reported.

- **Completion investigation** (committed earlier on 2026-10-11 without a changelog entry): PLAN.md amendments 7-11b;
  README sections for the trace audit, completion pilot and replication, prior probes, kernel and DCP source audit,
  Phase 2 stages 1-2, Phase 3, the kpool tail ring, the comparison with upstream vLLM, the gate and the running
  completion arm; receipt folders `pilot/`, `replication/`, `audits/`, `phase2-stage1/`, `phase2-stage2/`, `phase3/`,
  `tail-ring/`, `tail-fix/`, `gate/`, `arm-col511-plan/`; costs appended (18.97 GPU-hours before the arm). One edit
  today: the router-precision row names its upstream reference as vllm#54048.
- **`LEDGER.md`**: dated entries from the trace audit (2026-10-10) through the arm's start (2026-10-11), with GPU-hours,
  and three owned corrections: the 511-pool slice premise in the looping investigation's code question 6, the arm's
  GPU-hour estimate (13 → 18.5 before launch) and a second server start described for a comparator log that holds one.
  Three rows added to "Figures corrected along the way".
- **`CONFOUNDS.md`**: seven new entries (9-15): requests in flight and the top-k route, the kpool tail-ring positions
  omission, NaN-byte index rows, the historical control of the completion arm, router precision and the routed-expert
  SwiGLU clamp, graph vs eager execution for instrumentation, and post-completion continuation in the long-context
  decode-vs-prefill measure. Entry 7 lists what the completion investigation has tested since.
- **`FINDINGS.md`**: new items 4a (column-511 drop) and 4b (kpool tail ring), both supported as localized defects with
  their effect on completion open; item 7 and the open questions updated (the running arm, the tail-ring fix, router
  precision, the SwiGLU clamp, the long-context measure); upstream references written as vllm-project/vllm#N.
- **`README.md`**: the completion investigation's row, a line on the two further issues, glossary entries for column
  511, the top-k route and the tail ring; the kpool entry now says the engine keeps 511 of the 512 selected pools.
- **Forward pointers, dated 2026-10-11, history unchanged**: looping investigation (code questions 6-8, open question
  2, "Why is open", the kpool fixes paragraph), kpool investigation (finding 1), the three-pass GPQA comparison.
- **Protocols**: dated notes in `decode-prefill-consistency/v2` (long bins measure forced continuation) and
  `kpool-kernel-tests/v1` (scope: kernels, not the engine's metadata call). Procedures unchanged.
- **`comparisons/README.md`**: points to the completion investigation's receipt folders.
- **Site** (`tools/site.py`): Results summary and the engine-issues table list the two further issues as found during
  evaluation, fixes prepared and not released; the 0.7.0 vs 0.9.1 lines point to the investigation and the running
  arm; glossary entries for column 511 and the tail ring.

## 2026-10-10 (completion investigation, Phase 1 steps 5-6): kpool fixes on 0.7.0; 0.7.0's layout on 0.9.1

Contents: 65 runs (4 new), 13 comparisons. No earlier run, number or grade changed.

- **4 new decode-vs-prefill runs** (`decode-prefill-consistency/v2`, 16,000 tokens, 2026-10-10), each on a fresh server
  and started after the engine's start-up warm-up: tpurtell 0.7.0 + kpool fixes (DFlash2 ×5, 2 runs) and tpurtell 0.9.1
  with 0.7.0's parallel layout (DFlash2 ×3, 2 runs).
- **New configs** `k3.25-v0.7.0-kpoolfix-dflash5-nocache` and `k3.25-v0.9.1-ep2dcp2-dflash3-nocache` (top-k owner
  exchange off, as in the layout bisection's arm D; stated in the config).
- **Client** `tools/clients/decode_prefill_consistency.py`: per-prompt timing in its output and an opt-in `--trace-dir`
  that keeps token ids and log-probabilities locally and records their sha256. Without it, requests, rows and summary
  are unchanged. Dated note in protocol v2.
- **Completion investigation**: steps 5 and 6 in the Phase 1 section, PLAN.md amendments 5, 6 and 6a (each written before
  its step's data), Answer table and new open questions, steps 5-6 and trace hashes in `manifest.json`, costs and checks
  appended, `recompute.py` extended to reproduce both steps' numbers and verdicts.

<details>
<summary>Verdicts (prompts 1-5, positions 8,000-15,999)</summary>

- Step 5: corrected 0.7.0 0.0439 and 0.0390 against stock 0.0558 and 0.0243; the corrected values remain at stock
  level, so these repairs did not explain the discrepancy under this protocol.
- Step 6: 0.9.1 with 0.7.0's layout 0.0604 and 0.0309 against 0.9.1 as shipped 0.0142 and 0.0119, higher in 5 and 4 of
  5 prompts; the old layout bundle can reproduce the discrepancy on newer code, without identifying which component.
- 3.37 GPU-hours (1.62 + 1.75); Phase 1 total 10.55.

</details>

## 2026-10-10 (completion investigation, Phase 1 steps 1-4): decode vs prefill on both engines

Contents: 61 runs (12 new), 13 comparisons. No earlier run, number or grade changed.

- **12 new decode-vs-prefill runs** (2026-10-10), each on a fresh server, with numeric server-log fields
  (`server_log.jsonl`): tpurtell 0.7.0 with speculation off (2 runs) and on (DFlash2 ×5, 2 runs); tpurtell 0.9.1 with
  speculation off (a third run), DFlash2 ×5 (2 runs) and ×3 (a second run); and both engines as shipped at 16,000 tokens
  (2 runs each).
- **New configs** `k3.25-v0.7.0-nospec-nocache`, `k3.25-v0.7.0-dflash5-nocache`, `k3.25-v0.9.1-dflash5-nocache`.
- **New protocol `decode-prefill-consistency/v2`**: v1 at 16,000 generated tokens on prompts 0-5, with context-length
  bins in the summary (`by_bin`). v1 is unchanged. `tools/verify.py` recomputes the bins; `SCHEMA.md` lists them.
- **Completion investigation**: new Phase 1 section in its `README.md` (question, rule, runs, numbers, verdict and cost
  per step), PLAN.md amendments 1-4 (each written before its step's data), Phase 1 arms in `manifest.json`, costs and
  checks appended, and `recompute.py` extended to reproduce every Phase 1 number and verdict and to check that the
  README's tables match.
- The looping page's decode-vs-prefill figure keeps to that investigation's runs (2026-10-07 to 10-09).

## 2026-10-10 (completion investigation, Phase 0): existing data only; one correction; no data or number changed

Contents unchanged: 49 runs, 13 comparisons.

- **New `investigations/2026-10-glm53-completion/`**: why tpurtell 0.7.0 left fewer GPQA answers empty than 0.9.1.
  `README.md` (Phase 0 answer, evidence, the effective configuration diff of the two servers, limits, ranked next
  steps), `PLAN.md` (the plan, frozen 2026-10-10), `manifest.json`, `checks.jsonl`, `costs.jsonl` and `recompute.py`.
- **New `tools/analyze.py gpqa-join [A B]`**: joins two multi-pass records on question and pass, checks seed, prompt,
  target and question hashes, and splits the score into pairs where both answered and pairs where one came back empty.
- **Results page and `FINDINGS.md` 7a**: one added sentence. The higher 0.7.0 score comes entirely from question-passes
  0.9.1 left empty; where both answered (577 of 594), raw accuracy is identical (516 vs 516; stated 523 vs 522). The
  pair list is in a tooltip and in the item 7 details. Numbers are computed by `tools/site.py`.
- **Correction (chat template).** `CONFOUNDS.md`, `FINDINGS.md` item 7, the looping investigation and the doc-79
  comparison said both servers loaded the same vendored chat template. 0.7.0 served the checkpoint's own template
  (sha256 `34d5ee66…`), 0.9.1 the corrected Z.ai template (sha256 `0c4099f3…`). Both render all 198 GPQA prompts
  byte-identically, so no result changes.
- Links to the new investigation from `README.md`, `LEDGER.md` and the open questions in `FINDINGS.md`.

## 2026-10-10 (restructure): results first; confounds and ledger pages; no data, number, grade or claim changed

Contents unchanged: 49 runs, 13 comparisons.

- **One summary.** The site's Results page (`docs/index.html`) carries the only summary: seven lines, each with its grade
  and a link to its evidence. `README.md` and `FINDINGS.md` link to it instead of repeating it.
- **Results page.** GPQA (three-pass records, pass 1 of every configuration), speed and acceptance, KV capacity, engine
  issues and their fixes, and kernel tests, each as a table beside its chart. The former GPQA, screen, speed and kernel
  pages are merged into Results and the investigation; their URLs forward to the new places, old anchors included.
- **New `CONFOUNDS.md`** (site: Confounds): eight confounds the data covers, what each does, how the runs control for it
  now, and its status. **New `LEDGER.md`** (site: Ledger): a dated, plain-language account of what was run, found,
  corrected and withdrawn. It holds the withdrawal notice, now naming what replaced the withdrawn data (the
  distinct-seed component screen; the three-pass GPQA records), and the single draws kept from the earlier screens.
- **`FINDINGS.md`** becomes graded tables (statement, grade, evidence) with the numbers and qualifications in folded
  blocks; statements keep their numbers 1-19. Grades are supported, descriptive or open; "unmeasured" is no longer a
  grade word ("Descriptive (effect unmeasured)").
- **Investigation** (`investigations/2026-10-glm53-looping`): answer first, then the DCP1 tail issue record, then the
  evidence; method and history moved to the ledger and confounds pages. Old section anchors still resolve.
- Engine findings are worded as issues found during evaluation and their fixes. Detail, statistics and receipts sit
  in folded `<details>` blocks; on the site, glossary terms and numbers' context show as tooltips on hover or focus.
- `tools/site.py` renders `CONFOUNDS.md` and `LEDGER.md` into the site; `recompute.py` section headings follow the new
  page.

## 2026-10-10 (readability): organization and charts; no data, number, grade or claim changed

Contents unchanged: 49 runs, 13 comparisons.

- **Reading paths.** `README.md` opens with a "Start here" table (30-second answer, the DCP1 tail bug record, the
  current non-completion summary, choosing a quant or runtime, checking a claim). `FINDINGS.md` opens with a summary
  table: every numbered statement's headline, its grade and its section; each section links to its charts on the site.
  The looping investigation gains a fast-path line; secondary documents link back to the README and findings.
- **Comparisons.** Every comparison README follows one template (question, grade with its findings item,
  configurations, result, caveats, recompute command); `comparisons/README.md` gains an index of all thirteen.
- **Site.** Every page leads with a graded "What it shows" list linking to its evidence; the overview ends with a
  "Where to go next" table, and "How to read this" with a reading path and the grade definitions. Chart titles state
  the takeaway and its grade; axes carry titles and units. The overview's two three-pass charts are merged into one
  small-multiple figure (accuracy and empty answers per pass, side by side; the full charts stay on the GPQA page), and
  the component-screen chart moves to the screen and investigation pages. The component screen is split into one panel
  per question on a shared scale; decode vs prefill is split into one panel per position region on a shared scale;
  serving charts put each fix's with/without pair on adjacent rows. The empty-answer grid uses darker shades so white
  text meets 4.5:1 contrast. Section anchors and a document footer on every page.

## 2026-10-10: three-pass GPQA records with one request seed per pass

Contents now: 49 runs, 13 comparisons. GPQA Diamond: 10 runs (one per configuration), three of them with three passes.

- **New data: three GPQA Diamond passes per configuration, request seeds 1234, 1235, 1236** (pass *p* sends 1233 + *p*),
  for `3.25bpw · tpurtell 0.7.0 · DFlash2 ×5` (pass 1 of 2026-09-29, passes 2-3 of 2026-10-10), `3.25bpw · tpurtell 0.9.1 ·
  DFlash2 ×3` (pass 1 of 2026-10-08, passes 2-3 of 2026-10-09) and `4bpw TR3 (Brandon) · tpurtell 0.9.1 · DFlash2 ×3 ·
  concurrency 4` (all three passes new, 2026-10-09/10). The first two extend their existing pass-1 runs, the 0.9.1 one
  renamed from `…_gpqa-diamond-pass1` to `…_gpqa-diamond`; the third is a new run. Every new pass has a passive request log
  (seed, finish reason and completion tokens per request).
- **Concurrency in configurations and labels:** new config field `serving.concurrency`, present only where the client kept
  a number of requests in flight other than the protocol's; labels gain ` · concurrency N`. New configuration
  `k4-v0.9.1-dflash3-c4`: 4bpw TR3 (Brandon) leaves a 1,377,179-token KV pool, which holds 4.17 requests at the token cap, so
  at 4 no request waits for KV (peak usage 91.7%, no preemption). The GPQA protocol notes the rule (dated note; procedure
  otherwise unchanged).
- **The KV-saturated 4bpw TR3 (Brandon) tpurtell 0.9.1 pass 1 at 8 concurrent requests is kept and labelled:** its server
  log shows requests waiting for KV memory in 653 of 1,417 status lines. The record at 4 replaces it as the configuration's
  record; it stays as the like-for-like partner of the 0.9.0 run, which (like the 0.8.0 run) used the same KV pool size at
  8 concurrent requests and kept no server log; their notes now say so.
- **`tools/verify.py`:** a GPQA run may hold several passes only when pass *p* carries request seed 1233 + *p* on every row,
  one row per question per pass, the same questions in every pass; passes that share a seed are rejected. GPQA summaries of
  multi-pass runs add `by_pass`, `pass_spread_flexible_points` and `questions_ever_empty`; intervals resample questions with
  all their passes.
- **Server logs:** GPQA server-log figures now cover only the measured requests' time window, so the engine's start-up
  warm-up requests are excluded (the KV-saturated run's peak and waiting counts change slightly: 653 of 1,417 status lines,
  was 656 of 1,424); `server_log.jsonl` gains `session` records naming the passes each server start served.
- **`correct_stated` for the new passes:** judged on 2026-10-10 by the 2026-10-09 audit's script, unchanged (it reproduces
  the earlier judgement on all 1,782 previously judged rows), and every one of the 31 replies whose stated answer disagrees
  with `flexible-extract` was read by hand (three commit to two options and count as not correct), plus a 16-reply spot
  check. The scoring note now covers all 16 published passes: 0.5 to 3.5 points low per pass, 1.5 on average.
- **Determinations** (`FINDINGS.md` section 2, the looping investigation, comparisons, site): one configuration's passes
  differ by 0.5 to 3.5 points, as much as pass 1 differs across configurations (descriptive); 3.25bpw and 4bpw TR3 (Brandon)
  on tpurtell 0.9.1 show no measurable difference in accuracy or empty answers (descriptive); tpurtell 0.7.0 as shipped
  left fewer questions unanswered than 0.9.1, 4 vs 16 of 594 on 4 vs 11 questions (supported by a question-clustered
  sign-flip test, p = 0.009; passes 2-3 alone p = 0.06), cause open. The pass-1 table is reframed against the measured
  pass-to-pass noise, and the 0.7.0 open question is restated with this evidence.
- **`tools/analyze.py`:** `gpqa-passes`, `gpqa-records` (question-clustered comparisons: question-level counts, exact
  sign-flip test over questions, bootstrap over questions; pooled tests for reference) and `gpqa-empty`; `gpqa-table` and
  `gpqa-pairs` use pass 1 and compute percentages from the rows. New comparison `glm53-flash-gpqa-records`.
- **Site:** accuracy and empty answers per pass with the three-pass mean and its interval, a question-by-record grid of
  empty answers, the clustered comparison table, glossary entries (pass, KV pool, question-clustered test), and a
  concurrency row in the label key. Pass-1 percentages are now computed from the rows (the site showed 84.9% for a 168/198
  run that the text gives as 84.8%).
- Glossary in `README.md` gains pass, KV pool / KV-saturated and question-clustered test.

## 2026-10-09 (review): receipts for every figure, qualified determinations

Contents now: 48 runs, 12 comparisons.

- Kernel tests on the release images (2026-10-09): `kpool-kernel-tests/v1` now covers the tpurtell 0.9.0 and 0.9.1
  release images (33 of 33 upstream tests pass; no rejected-draft corruption at 2, 3, 5 or 7 draft tokens), replacing
  the earlier reliance on file identity for those releases.
- **Decode vs prefill on the tpurtell 0.9.1 release, published:** two runs of one configuration with the same prompts
  and seeds (the noise floor) and one with DFlash2 ×3 on, 12 prompts each, protocol `decode-prefill-consistency/v1`.
  New comparisons `glm53-flash-v091-decode-prefill-repeat` and `glm53-flash-v091-decode-prefill-speculation`. Only the
  large tail-fix effect below 2,044 tokens is claimed; the fix's effect on answers is stated as unmeasured.
- **Server-log figures now have receipts:** runs whose notes quote KV pool size, throughput, acceptance or waiting
  requests publish `server_log.jsonl` (numeric fields parsed from the engine's log), and `tools/verify.py` recomputes a
  `server` block in their summaries. The figures that still rest on unpublished model output (greedy prefix lengths,
  the audited `correct_stated` judgement) are named as such.
- **GPQA: secondary, audited score `correct_stated`** per row (whether the reply's stated final answer is right) and
  accuracy among answered questions, beside the raw `flexible-extract` headline. The scoring note now uses the published
  runs: 0.5 to 2.0 points low per run.
- **Qualifications:** the shared pass-1 seed likely understates run-to-run spread; questions 88 and 79 were chosen
  from seed-1234 data; the component screen's fixed settings, its pairing of ownership with slot sharing, and its
  minimum detectable differences (`tools/analyze.py screen-power`); kernel tests cover the 0.7.0/0.8.0 images and
  local builds, not the 0.9.0/0.9.1 release images; speed figures are single runs; the engine is not bitwise
  reproducible even one request at a time.
- **Open questions added:** 0.7.0 as shipped and empty answers; single-request nondeterminism; no higher-precision
  reference on this hardware.
- Labels: `, prefix cache off` is appended when prefix caching is disabled. Glossary in `README.md` and on the site's
  method page, linked from first use on each site page; the overview leads with its summary.

## 2026-10-09: hard-question screens withdrawn as rate measures; component screen added

Contents now: 43 runs of GLM-5.3-Flash on 2x RTX PRO 6000: 9 GPQA Diamond runs (pass 1 each); 7 component-screen runs
(`hard-prompt-screen/v2`); 10 earlier screens kept as single draws (one kept as INVALID with its cause); 7 serving
probes; 4 kpool kernel-test runs; 2 kpool tail index checks; 2 decode-vs-prefill runs; 2 tool-eval-bench runs. 10
comparisons.

- **Withdrawn: the hard-question screen rates.** Every repeat of the `hard-prompt-screen/v0` and `v1` screens
  (2026-09-30 to 2026-10-08) sent request seed 1234. Concurrent batching still made the repeats vary, but every repeat
  drew on the same sampler noise, so that variation was not statistically meaningful and the effective sample behind
  each rate was far smaller than its request count. Evidence: on `3.25bpw · tpurtell 0.9.1 · DFlash2 ×3`, question 88 failed 3 of 12
  times with distinct seeds and 11 of 12 with seed 1234 on every repeat, all else equal. Withdrawn: the screen failure
  rates and intervals, the layout-bisection statistics and verdicts, and the question-level observations drawn from
  those screens, together with the text built on them in `FINDINGS.md`, the investigation, the comparisons and the site.
  Kept: repeat 1 of each question from each configuration's first screen, as a single draw. Removed runs (in git
  history): the 2026-10-04 `screen-a`, the 2026-10-05 `k3.25-v0.7.0-dflash5_screen-v07pair` and the three
  `screen-bisect2` runs; removed comparisons: `glm53-flash-engine-screen-v0`, `glm53-flash-kpool-screen`,
  `glm53-flash-kpool-screen-v070`, `glm53-flash-v090-layout-screen`, `glm53-flash-v090-tailfix-screen`. Notice:
  `investigations/2026-10-glm53-looping`, section 4. The layout-bisection preregistration is kept unchanged and marked
  withdrawn in a separate `WITHDRAWN.md`.
- **New protocol `hard-prompt-screen/v2`:** one question per run, 12 repeats with distinct request seeds (5000 +
  repeat), 12 concurrent, same early stop and classifier. `tools/clients/hard_prompt_screen.py` gains `SEED_BASE` and
  `CONC`; its v1 defaults are unchanged.
- **New data: the 2026-10-09 component screen** on tpurtell 0.9.1 (preregistered; plan and amendment in
  `investigations/2026-10-glm53-looping/component-screen`): question 88 under three layout switches (arms B, EN, EO,
  NO), question 79 on tpurtell 0.9.1 and on the tpurtell 0.7.0 image at three draft tokens (B79, V79), and a
  fixed-seed control (S1234). Five new configurations; new comparisons `glm53-flash-v091-component-screen-doc88`,
  `glm53-flash-doc79-image-screen`, `glm53-flash-fixed-seed-control`.
- GPQA runs now publish pass 1 only; further passes will use per-pass request seeds (1233 + pass). New: pass 1 of
  `3.25bpw · tpurtell 0.9.1 · DFlash2 ×3` and `4bpw TR3 (Brandon) · tpurtell 0.9.1 · DFlash2 ×3`, with `finish_reason`
  and `completion_tokens` from a passive request log. GPQA rows' `seed` is now the request seed.
- Decode vs prefill: each build was measured once, and a later repeat of the measurement varied by up to about 2x
  between runs; only the large effect below 2,044 tokens (KL 0.066 → 0.010) is claimed.
- `FINDINGS.md` rewritten to what the data supports, each statement graded supported / descriptive / open; the
  investigation reframed around what drives non-completion on hard questions; `tools/analyze.py` subcommands
  `gpqa-table`, `gpqa-pairs` (question-paired), `screen-v2`, `screen-single`, `screen-anatomy`, `screen-speed`;
  `tools/verify.py` enforces pass 1 for GPQA runs, repeat 1 for v0/v1 screens, distinct seeds in v2 runs, and no
  comparisons of v0/v1 runs; site rebuilt.

## Unreleased (first public version)
Contents: 39 runs of GLM-5.3-Flash on 2x RTX PRO 6000:
- 7 GPQA Diamond runs: 5 full 3-pass and 2 single-pass;
- 15 hard-question screens, one kept as INVALID with its cause;
- 7 serving probes;
- 4 kpool kernel-test runs;
- 2 kpool tail index checks;
- 2 decode-vs-prefill consistency runs;
- 2 tool-eval-bench runs.

Also: 11 comparisons, two investigations with their preregistrations, `FINDINGS.md`, `DATASHEET.md`, `SCHEMA.md`, and the
results site.

- Scoring note revised (2026-10-08): `flexible-extract` underscores by 0.5 to 2.5 points per pass, about 1.6, not about
  0.7. Most of it comes from replies that state the answer and then mention other options' labels; the earlier figure
  checked only replies where the two filters disagree. Notes added on `strict-match` (a phrasing check, not accuracy)
  and on answer order (fixed by the first `--seed 1235` load, cached). `tools/verify.py` now checks that compared GPQA
  runs share prompt and target hashes. No scores changed.
- Scoring note (2026-10-08): `flexible-extract` matches any parenthesised capital letter, so notation in a reply can
  be read as the answer. A deterministic misread that only lowers scores, about 0.7 points per pass. Stated in the GPQA
  protocol, DATASHEET, SCHEMA and site; no scores changed. FINDINGS 6 now credits unanswered questions at their own
  observed accuracy (0.8-2.8 points instead of 0.9-3.4).
- tpurtell 0.9.1 (2026-10-08): the DCP1 tail fix was merged (tpurtell/glm-5.3-flash-ext3-2x-rtx#6) and released in
  tpurtell 0.9.1. The tail-fix configurations, measured on a local build before the merge, are now labelled
  `tpurtell 0.9.0 + DCP1 tail fix ≈ 0.9.1` with the basis in `engine.equivalent_to`, as `0.8.0 + kpool fixes ≈ 0.9.0`
  is. Engine series `tpurtell 0.8.0-0.9.0` renamed `tpurtell 0.8.0-0.9.1`. Glossaries, the investigation and the site
  define 0.9.1. No data changed.
- Looping investigation (2026-10-08): `investigations/2026-10-glm53-looping`, and a site page.
  - **Layout bisection** on tpurtell 0.9.0, preregistered with three amendments: six screens across three
    configurations.
    - 0.9.0 as released vs v0.7.0's layout as a diagnostic control: 41 vs 29 failures of 80, p = 0.079 (not yet
      significant at this size).
    - With vs without the DCP1 tail fix (tpurtell/glm-5.3-flash-ext3-2x-rtx#6): 33 vs 41, p = 0.27, lower in both screens
      with the fix, not yet significant at this size.
  - **Tail-fix checks:** index check, decode-vs-prefill consistency (KL 0.066 → 0.010 below 2,044 tokens), serving
    probe, and tool-eval-bench (157/157 → 159/163 of 176).
  - New protocols `decode-prefill-consistency/v1`, `kpool-tail-index/v1` and `tool-eval-bench/v1`, with their clients.
  - New config field `serving.layout`. Its label appears only for non-default layouts.
  - `tools/analyze.py screen-pool`, and `recompute.py` in the investigation.
  - `FINDINGS.md` gains inference 8.
- Labels (2026-10-06): configurations are named `weights · engine version · speculation`, engine name first (for
  example `4bpw TR3 (Brandon) · tpurtell 0.8.0 + kpool fixes ≈ 0.9.0 · DFlash2 ×3`), built from new config fields (`model.label`,
  `engine.name`, `engine.built_on`, `engine.patches[].name`, `engine.equivalent_to`, `engine.series`,
  `serving.speculative.label`) by one rule in
  `tools/verify.py`, which also fails if a label is ambiguous. Defined in `SCHEMA.md`; key at the top of `FINDINGS.md`,
  in `README.md` and on every site page. Replaces "0.8.0 + patches ≈ 0.9.0". Brandon M. Music's TR3 checkpoint is labelled
  `4bpw TR3 (Brandon)`; speculation labels name the method (`DFlash2 ×3`, `no speculation`). Config and run notes use the labels.
- Consistency corrections (2026-10-06), no data changed: the 0.7.0-vs-0.8.0 completion gap is about a third on GPQA
  (6 vs 20 of 594, Fisher p = 0.009) and about half on the screen (12 vs 24 of 40); the earlier text merged the two and
  called overlapping exact intervals non-overlapping. "No completion cost" from the kpool fixes is now "no detectable
  change" with its interval (-3 to +19 more empty answers on 4bpw TR3 (Brandon), tpurtell 0.9.0). The gap to published scores is no longer
  called "not a runtime problem". Greedy divergence is stated as measured (median 318 characters). The serving probe's
  fix effect is given as measured (-7% to +3%, single runs). The published-context comparison covers all five full runs.
  See the commit history for each change.
- `tools/analyze.py` gains `gpqa-passes`, `gpqa-pairs` and `screen-speed`, so every number in `FINDINGS.md` is
  recomputable.
- Correction: the GPQA protocol said passes used seeds 1235-1237. Those are lm-eval's `--seed`; every request
  carried `seed: 1234`. Passes differ through batching nondeterminism. Protocol and schema now say so; no number changes.
- Results site in `docs/` (GitHub Pages), generated from the data by `tools/site.py`: overview, GPQA, screens,
  speed and acceptance, kernel tests, and a reading guide; interval plots with 95% intervals, table views, links to receipts.
- 4bpw TR3 (Brandon) on tpurtell 0.8.0 + kpool fixes ≈ 0.9.0, hard-question screen: 15/40.
- 4bpw TR3 (Brandon) on tpurtell 0.9.0 (official release): full GPQA Diamond run (84.7%, 23 of 594 empty), replacing the earlier
  partial exports. New comparison `glm53-flash-k4-v080-v090-gpqa`: no detectable difference from tpurtell 0.8.0 in
  accuracy (p = 0.90) or empty answers (23 vs 15, p = 0.25).
- Site labels runs with fewer than 3 passes by their pass count; accuracy ranges on the site are computed from the data.
- Client fix before release: a stream that ends without a finish reason is an `error`, not an `exhaust`. Affected only
  the INVALID run (engine crash); all other screens were audited and have finish reasons on every row.
