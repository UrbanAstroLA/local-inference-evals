# Hook validation, attempt 2: preparation (CPU only; NOT launched)

> **Superseded in part (PA2-24, 2026-10-14): PLAN.md amendment 21g is implemented in `ATTEMPT2-21G.md`.** That
> document wins where the two differ: criteria, analysis, client records, carry-over admissibility, coverage and the
> suites run. The hook, image, runner and cost below are unchanged.

2026-10-14. This prepares attempt 2 under:
- DIAGNOSIS-a1.md;
- PLAN.md amendments 21, 21c, 21e and 21f.

How it was done:
- **No GPU work.** No GPU or glm53 container was started, and nothing used `--gpus`. Only CPU containers ran, with
  `--network none` and `CUDA_VISIBLE_DEVICES=`.
- **No `sudo`.**
- **Attempt-1 frozen files were never overwritten in place.** Each modified file's attempt-1 version is kept as
  `<file>.a1`, and each copy hashes to its SHA256SUMS.a1 entry (checked by `prep-a2/pa2_changes.py`). Attempt-1
  outputs (`build/`, `tests/out/`, `run-a1/`, `diagnosis-a1/`) are untouched.
- **Every change is post-inspection and labelled PA2-n.** Before/after hashes are in `prep-a2/PA2-changes.json`
  (table in section 6).

**Attempt 2 is the last attempt under amendment 21c.** It needs, in order:
1. independent review of the criteria draft (`ATTEMPT2-CRITERIA-DRAFT.md`);
2. an amendment recording the post-inspection items and the adopted criteria, before any data;
3. coordinator verification of SHA256SUMS;
4. operator authorization.

## 1. Hook fixes (PA2-1, PA2-2) and CPU tests

**PA2-1: selection rows are mapped to requests through a per-token table** (`hook/forcewin_hooks.py`).
- `on_batch` fills a persistent `S.tokreq[t]` (query token → request) from `query_start_loc`. It does this before the
  forward, so a replayed graph reads this step's values from the capture-time address.
- `on_selection` then maps rows by layout:
  - with `next_n == 1` (the SM120 indexer's flattened `[B·4, 1]` layout, or plain decode), row t → `tokreq[t]`, and
    padding tokens → −1;
  - with `next_n > 1` (the native `[B, next_n]` layout), it keeps `row // next_n`.
- The same mapping serves the exposure records (`xbuf`) and the full selections (`selbuf`).

**PA2-2: the ledger stores the step's real counts.**
- `d2_tail_mapping` and `note_tail_mapping` keep the tail mapping cut to the real requests and tokens recorded by
  `on_batch`. FULL-graph padding requests come after the real ones.
- `_shadow_step` refuses a count mismatch: it raises, and the error is logged and flagged by H1i.
- It now runs in its own guard, so a ledger fault cannot skip that step's snapshot.

The five engine call sites are unchanged.

**CPU results.** All runs are in `tests/out-a2/`. The suites ran on B1 CPU containers with the hooked copies, then on
the built image.

| Suite | Result |
|---|---|
| `test_forcing.py`, 9 shards | **66/66**, run twice. The second run, `errscan_forcing.sh`, also scans the hook's error log: **0 `fw_error` records** in every shard |
| `test_d2_inject.py` (hooked copies; built image) | **22/22; 22/22**. These are the 16 attempt-1 checks plus D7p-a/b/c/d, D7pn and D0 |
| `test_exposure.py` (hooked copies; built image) | **12/12; 12/12**. These are the 6 attempt-1 checks plus X2f-a/b/c, X2fn, X2fn-b and X0 |
| I1 inertness: original B1 / hooked env unset / hooked env on, unkeyed / built image env unset | **Identical hash** `47e7bc7b…`, the same as attempt 1's |
| MG1 (mapguard import), IMG1 (installed files == reviewed copies, 6 files), IMG2 (image inert with env unset) | PASS, PASS, PASS |

**The new tests, and what they show.**
- **Flattened layout, positive.** X2f gives 8/8 records at their own (request, output index), with brute-force counts
  and correct selections. It stays exact with non-uniform lengths (4, 1, 4) plus 7 padding tokens, and in the native
  layout with a padding request.
- **Flattened layout, negative control.** X2fn runs the attempt-1 module (`.a1`) on the same input: 1 of 8 correct,
  7 missing, `(b, 66)` misattributed. That is the GPU signature.
- **Padded steps, positive.** D7p runs 3 requests in the 16-token graph through the real hooked builder, the real
  decode writer and the hook's own ledger:
  - B1 arm: 21/21 pools clean, under their own request, and byte-correct;
  - injected arm: clean pools are byte-correct and corrupt pools are wrong;
  - padded and unpadded runs give identical classes;
  - a count mismatch is refused.
- **Padded steps, negative control.** D7pn feeds the attempt-1 ledger the same padded inputs: 14 of 21 pools are
  missing, 6 are foreign, and 5 byte-correct pools are labelled corrupt. Unpadded, it is correct.

## 2. Image

| Field | Value |
|---|---|
| Tag (new; no tag replaced) | `local/glm53-v0.9.1-b1-forcehook-mg2:local` |
| Image id | **`sha256:ac05230bd94b3114cb6fb202b06402f9e2d45a61b20f041f314e045f967e97ee`** |
| Base | B1-mapguard `sha256:4d7bd73e…`, as attempt 1 (`build-a2/make_build.sh`) |
| Installed module | `forcewin_hooks.py` sha256 `15e93e23…` (attempt 1: `c079560e…`) |

Manifest checks:
- **Against the base:** the `.py` manifest differs in exactly the same 6 files as attempt 1 (`build-a2/manifest-diff.txt`).
- **Against the attempt-1 image** (`sha256:1b9f8b15…`): only `vllm/forcewin_hooks.py` differs
  (`build-a2/manifest-diff-vs-a1.txt`).
- **Engine files:** the five hooked engine files are byte-identical to attempt 1's. The build aborts otherwise.

**The image is not live until the coordinator verifies the package.** It runs only through `run_hookval.sh` after
authorization.

## 3. Analysis (PA2-7, PA2-8, PA2-11)

`analysis/analyze_hookval.py` writes two verdicts beside each other:
- **original**, in `verdict.{json,txt}`: the frozen attempt-1 code (`analyze_hookval.py.a1`) run verbatim;
- **revised**, in `verdict-a2.{json,txt}`: the labelled fixes plus the DRAFT criteria.

| Label | Fix |
|---|---|
| PA2-7a | H2 join by position, with top-1 counted on finite reference rows |
| PA2-7b | Argmax shift control beside the frozen lp rule |
| PA2-7c | Guard polls per rank, as deltas from the client's start dump |
| PA2-7d | H5h excludes o = 0 and W ≤ 1, and requires step-matched records (selections too) |
| PA2-7e | Window comparisons exclude pools completing at P + W − 1 |
| PA2-7f | H1f reports unmasked rows as their own class; gate "no violation with tracked mass > 1e-6" |

The H2 null and H3 carry over from attempt 1 (status CARRIED), through `analysis/carryover-a1.json` with source hashes.

**Validation of the analysis.**
- **Synthetic dry run** (`tests/make_synthetic_run.py`, now in the guard's real poll format: `rank`, no `pid`, the
  pre-client warm-up artefact):
  - **good** gives revised **VALID**. The original verdict is INVALID by design, because the run contains attempt 1's
    non-defects.
  - **bad** gives revised **INVALID**, failing exactly the 10 planted checks: H1c, H1f-r, H2c-r, H3-r, H4b-r,
    IDX-early, IDX-page, H5f-B1, H5g-r and H5h-r.
- **Replay on the real attempt-1 data** (`prep-a2/replay-a1/`):
  - the original verdict is reproduced **check-for-check**;
  - the revised checks reproduce the diagnosis: H1f-r, H2a-r, H2b-r, H2c-r, H4b-r, IDX-stable and H5g-r pass;
  - the hook defects fail: H5h-r (13,935 missing and 3,443 misattributed of 20,263 positions) and H5f-B1;
  - the V4n-dependent checks cannot run on attempt-1 data.

## 4. Revised criteria: DRAFT

`ATTEMPT2-CRITERIA-DRAFT.md` gives, for each criterion, the original, its attempt-1 value, the measured null, the
proposal, and the proposal evaluated on attempt-1 data. Points for the reviewer:

- **Index-K content.**
  - The literal "B1 ≤ 2 × the decode null" **fails on attempt-1 data**: 0.71 % against 2 × 0.11 %. That null was
    measured on one, unfragile trace.
  - The draft therefore gates at max(2 × max(N_dd, N_pp), 0.1 %), with both nulls in-session (V4n). On attempt-1 data
    this passes at 1.35 × N_pp.
- **KDA identification.**
  - "100 % in every layer" passes attempt 1 (3,604/3,604), but deep layers have near-ties (minimum margin 1.04).
  - A variant (100 % in layers 0-17, ≥ 99 % deeper) is reported beside it and recommended.
- **Structural addressing** separates completely on attempt-1 data:
  - aligned: page maximum 9.4 % and 0 gross in layers 3-15;
  - a planted wrong page: 100 % of pages over 25 % and 99.9 % of layer 3-15 pools gross.
  - It needs per-pool deduplication and a 16-pool page floor.
- **H5f** gates the B1 arms at 100 % clean (the positive control for PA2-2), and D2 clean pools at the content limit.

## 5. Scoped client and runner (PA2-9, PA2-10, PA2-12, PA2-13)

**Client** (`runner/client_hookval.py`). The phases run in this order:

| Phase | Content |
|---|---|
| V0 | Unkeyed probes |
| V2a | natural + oracle |
| V2c | Prefill references |
| V4 | As attempt 1 |
| **V4n-b1b** | The same 8 windows uninjected, in two batches of 4: the decode-path null |
| **V4n-rebuild2** | Every window prefilled at a different build length (+124 to +200 real ids of the same trace): the prefill null |
| V3 | As attempt 1 |
| V2a poison | **Optional**, only if the deadline leaves room (`optional_skip` is not a coverage skip) |

- The second arm and rebuild are tagged `V4n-…`, so the frozen H5 code ignores them instead of mis-parsing them.
- Requests now record `ref_from` and their batch.
- Plan (`runner/plan-dryrun-a2/plan.json`): 89 requests (81 mandatory and 8 optional), 72,692 forced tokens and
  93,845 prompt tokens.

**Runner** (`runner/run_hookval.sh`).
- Image pinned to `…-mg2` (`runner/images.txt`).
- **ATTEMPT must be 2, and the cost ledger must hold exactly the one attempt-1 line (2 of 2).**
- The budget gate is cumulative over the item, against 2.4 GPU-hours, so the wall cap is capped at **2,138 s**.
- MIN_WALL is 1,500 s.
- Everything else is as attempt 1: pre-flight, hooks registered on both ranks, watchdog, trap, cost line, and the
  analysis on CPU.

**Cost** (`runner/cost_a2.py`, from attempt-1 measured phase times and its 746 s fixed overhead):

| Scope | Client | Occupied | GPU-hours | Item cumulative |
|---|---:|---:|---:|---:|
| Mandatory | 514 s | 1,260 s | **0.700** | 1.912 |
| + optional poison | 674 s | 1,420 s | 0.789 | 2.000 |
| Worst case (wall cap) | | 2,138 s | 1.188 | 2.3995 |

- The second B1 arm is priced at 2 × attempt 1's batch of 8. That is conservative, and it is why these figures are
  above the diagnosis's 0.67.
- Client room at the cap is 1,236 s (cap minus the 180 s reserve minus attempt 1's 722 s readiness), against 674 s
  needed.

**Gate tests** (`tests/test_runner_a2.sh` → `tests/out-a2/runner-tests.log`): **16/16**. They run on a byte-identical
runner copy in a shim tree, where docker, nvidia-smi and curl shims refuse everything except `docker image inspect`;
no shim refusal occurred.

| Group | Cases |
|---|---|
| DRY_RUN passes (2,138 s wall, 1.188 cap) | Default attempt; other items' cost lines ignored; WALL_CAP env capped |
| Refused | No approval; 2.5 > ceiling; attempt 1 or 3; 2 attempts recorded; ledger missing attempt 1; spent too high; approval 2.0 (cumulative gate) |
| Aborted | Wrong image id; tampered frozen file; run-a2 exists |

**The final DRY_RUN of the real package** (after the last SHA256SUMS) is in `DRYRUN-a2.out`. That file is excluded
from the sums because it is written afterwards.

**Launch command, after an amendment and authorization only. Do not run before.**

```bash
APPROVED_GPU_H=2.4 <runs>/hook-validation/runner/run_hookval.sh 2
```

## 6. Change log (post-inspection, PA2-n)

`prep-a2/PA2-changes.json` lists, per item and file, the attempt-1 hash (from SHA256SUMS.a1, sha256 `67d8e54f…`) and
the current hash.

| Id | Change | Files (before → after) |
|---|---|---|
| PA2-1, PA2-2 | Hook: per-token selection table; ledger real counts | `hook/forcewin_hooks.py` c079560e → 15e93e23 |
| PA2-3 | Flattened-layout tests; forcing error scan | `tests/test_exposure.py` 6f77062a → f08bda70; `tests/errscan_forcing.sh` new |
| PA2-4 | Padded-step ledger tests | `tests/test_d2_inject.py` dd1f299b → f6fb7a18 |
| PA2-5 | Test runner: out-a2 and the new image | `tests/run_cpu_tests.sh` fb7378df → 2b3b4132 |
| PA2-6 | Image build | `build-a2/` new |
| PA2-7 | Analysis | `analysis/analyze_hookval.py` 6395e025 → afbf92b6 |
| PA2-8 | Synthetic runs | `tests/make_synthetic_run.py` 2fc3731d → 72c699d0 |
| PA2-9 | Scoped client | `runner/client_hookval.py` 422c6b35 → cda00d8b |
| PA2-10 | Runner and pin | `runner/run_hookval.sh` e6ed0236 → 79ae131c; `runner/images.txt` e077540a → 4a74b3fc |
| PA2-11 | Carry-over | `prep-a2/make_carryover.py`, `analysis/carryover-a1.json` new |
| PA2-12 | Cost and plan | `runner/cost_a2.{py,json}`, `runner/plan-dryrun-a2/` new |
| PA2-13 | Runner gate tests | `tests/test_runner_a2.sh`, `tests/out-a2/runner-tests.log` new |
| PA2-14 | Sums | `make_sums.sh` 756ab5da → 8140307f; SHA256SUMS regenerated |
| PA2-15 | Criteria basis and replay | `prep-a2/criteria_basis.py`, `prep-a2/out/`, `prep-a2/replay-a1/` new |

SHA256SUMS covers this document, so its own hash is recorded outside the package: in the hand-off and, when adopted,
in the amendment.

## 7. Files

| Path | What it is |
|---|---|
| `ATTEMPT2-PREP.md` | This document |
| `ATTEMPT2-CRITERIA-DRAFT.md` | Criteria draft |
| `prep-a2/` | `criteria_basis.py` + `out/criteria_basis.json`; `make_carryover.py`; `pa2_changes.py` + `PA2-changes.json`; `replay-a1/` |
| `build-a2/` | Image build: script, Dockerfile, log, id, manifests |
| `tests/out-a2/` | Every attempt-2 CPU log |
| `*.a1` | Attempt-1 versions of every modified file, including `SHA256SUMS.a1` |
