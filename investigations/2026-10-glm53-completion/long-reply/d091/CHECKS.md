# CHECKS: option D package (d091), CPU only, 2026-10-21

**Zero GPU.** No model server was started, no container ran with `--gpus`, no request went to a model server, and no
`sudo` was used. The CPU containers ran with `--network none`, `CUDA_VISIBLE_DEVICES=` and read-only inputs. Inputs that
held text existed only under `PRIVATE/` and were shredded afterwards; `PRIVATE/` is empty at freeze.

Labels: [M] measured here; [R] reproduced against an earlier package's frozen output.

| # | check | evidence | result |
|---|---|---|---|
| 1 | Fresh request seed | `checks/seed-scan.out`, `seed-inventory.json`, `raw-seed-scan.out` | 70704 (and 70705/70706) in no request log, results config, published row or text-scan value (20 logs, 50 results files, 10,903 text files, 1,340 values), nor in any gzip raw-body file [M] |
| 2 | Gate calibration vs the record's own variation | `checks/gate-calibration.txt`, `.json` | All 12 server gates (identity, warm-up count, KV floor, the 4 warm-up checks, health) pass on **both** 0.9.1 record sessions and on B+'s 0.9.1 session. Client config, per-question hashes, prompt tokens and task hash are identical across the record's 3 passes. KV pool: 4,707,515 in 10/10 default-layout 0.9.x sessions; 3 sizes over 7 sessions on 0.7.0 and 2 over 2 on 0.9.0 +ep2dcp2. Reported, not identity-gated; floor 2,646,016 gated [M] |
| 3 | Analysis dry runs | `tests/analysis-dryruns.txt`, `tests/dryruns/` | **17 / 17 as expected** (detail below) [M] |
| 4 | Discovery pipeline unchanged | `tests/repro-discovery.txt` | Byte-identical copies of build_index.py, trajectory.py, periods.py and analyse.py rerun on rep070-F: **12 / 12 outputs byte-identical** to `discovery/out` (prespec.sha256 is not a script output). Tolerance not needed [R] |
| 5 | Index adapter | `tests/index-repro.txt` | `scripts/index_d091.py` on rep070-F gives `rep070F_index.json` **byte-identical** to discovery's (f15153d4…) [R] |
| 6 | Side-by-side tables use discovery's definitions | `tests/compare-repro.txt` | `compare_d091.py --reproduce` matches discovery's `summary.json` exactly on T1 (6 groups), T3, T4, T12, T13 and drift: **ALL SAME** [R] |
| 7 | Prefix selection = B+'s rule | `tests/prefix-select-repro.txt` | The d091 selection function on B+'s inputs, and the byte-identical `select_prefixes.py` copy, both give `bplus/out/selection.json` **byte-identical** [R] |
| 8 | Prefix construction = B+'s code, unchanged | `tests/prefix-build-repro.txt` | Through the wrapper (only REQLOGS settable): build 86/86 checks; `prefix_manifest.json` **byte-identical**; `PRIVATE/prefixes.json` sha256 equal to B+'s; verify 170/170 in each of the 0.7.0, 0.9.1 and B1 images, each `prefix-verify-*.json` **byte-identical** to B+'s [R] |
| 9 | Whole pipeline, stand-in pass | `tests/standin/` | Detail below [M] |
| 10 | Runner refusals (DRY_RUN=1) | `tests/runner-refusals.txt` | **22 / 22**: the budget refusals, the overrides (IMAGE, PORT, GPU_MEMORY_UTILIZATION, GPQA_SEED_LIST, MAX_NUM_SEQS), a GLM container running, a wrong digest, a compute process, lm_eval running (a process only mentioning it does not block), port 8002 in use, an existing run/, a non-empty out/, PRIVATE not 0700, SHA256SUMS mismatch; the real machine passes at 10.0 and 8; DRY_RUN creates nothing [M] |
| 11 | Recording-proxy copy | `tests/proxy-sidecar.txt` | rep070's test, unchanged, on this package's byte-identical proxy copy: **17 / 17** [M] |
| 12 | Chain seed lines | `tests/chain-seed.txt` | **6 / 6**: pass 1 sends `,seed=70704`; process seed 1235; cap, concurrency, timeout and retries as the protocol; equal to the original up to the frozen diff; byte-identical to rep070's copy [M] |
| 13 | Session simulation (shims, no GPU) | `tests/session-sim.txt` | **67 / 67** over six scenarios (detail below) [M] |
| 14 | Timing reconstruction | `cost/timing-reconstruction.{json,md}` | rep070's `reconstruct_timing.py` (byte-identical) reproduces rep070's output byte for byte; record pass-1 session 5.147 GPU-h [R] |
| 15 | Costing | `cost/cost.txt`, `cost.json` | Central 5.15 (record) / 5.6 (model median); pessimistic 7.86 (p95); ceiling 10.0 (p99.9 9.74, rounded up); P(ceiling) 0.004% central, 0.05% pessimistic, 6.0% at twice the cap rate [M, model I] |
| 16 | Privacy | `checks/privacy-scan.txt` | 0 prompt-text, 0 reply-text and 0 token-id-run hits in every file outside `PRIVATE/`. `PRIVATE/` is 0700. Searched: windows of all 198 prompts and of all 594 rep070-F raw replies [M] |
| 17 | Real-machine DRY_RUN=1 | `tests/dry-run-real.txt` | Every refusal check passed on this machine (GPUs idle apart from the desktop); nothing created: no run/, no PRIVATE/raw/, empty out/, no `+d091` files in <data-runs> [M] |

## Detail for check 3 (analysis dry runs)

The stand-in is the 0.9.1 record's pass 1, relabelled to request seed 70704, read with its own session log.

**Valid, as expected:**
- D1, the stand-in: valid; empty 5, raw 169.
- D2, the KV pool edited to 4,698,112 (above the floor, not a size the record produced): valid. This shows the pool is
  not identity-gated.

**Invalid (exit 3), as expected, N1-N15:**

| case | what was altered |
|---|---|
| N1 | wrong request seed |
| N2 | missing question |
| N3 | wrong task hash |
| N4 | no ready line |
| N5 | no engine start |
| N6 | re-sent request |
| N7 | altered target hash |
| N8 | KV pool below the floor |
| N9 | wrong process seed |
| N10 | HTTP 500 |
| N11 | non-default args changed |
| N12 | no model_args seed |
| N13 | 0.7.0 server log |
| N14 | no proxy request log |
| N15 | no server log |

## Detail for check 9 (stand-in pipeline)

The stand-in is rep070-F's pass 1, with real raw bodies, run through `scripts/run_pipeline.sh`. All 9 steps returned
rc 0: index, trajectory, periods, compare, select, build, verify-v091, verify-b1, privacy. The checks passed:
- the per-reply, per-window and post hoc rows equal discovery's for the same 15 ids;
- the B+ rule selected 6 prefixes;
- the prefix hashes of the 4 replies B+ also selected equal B+'s manifest;
- build 44/44;
- verify 86/86 in the 0.9.1 image and 86/86 in the B1 image.

## Detail for check 13 (session simulation)

| scenario | outcome |
|---|---|
| (a) normal end of the session | one cost line; chain called with +d091 / 1 pass / 70704 / 8 / proxy; server stopped once; logs and inspect saved; loopback; cache inventories; seeds planned; placeholder pass reported INCOMPLETE and never classified |
| (b) SIGTERM | rc 4, cleanup, chain_rc null, INCOMPLETE |
| (c) watchdog | chain killed, server stopped, watchdog_fired true, INCOMPLETE with the watchdog cause |
| (d) chain failure | no retry, chain_rc 1, INCOMPLETE |
| (e) watchdog with a request in flight through the proxy | sidecar records start then interrupted; no main-log row; the raw interrupted record is in `PRIVATE/raw`, mode 0600; INCOMPLETE with interrupted 1, unstarted 197, never counted empty |
| (f) complete pass directory without proxy rows | probe COMPLETE, frozen analysis INVALID (rc 3), pipeline not run |

Across all scenarios: no refused docker call, and no process is left behind.

## How to rerun (all CPU only)

```bash
cd <runs>/long-reply/d091
python3 -B checks/seed_scan_d091.py --out /tmp/x.json --candidates 70704   # about 2 min
python3 -B checks/gate_calibration.py checks/gate-calibration.json
bash tests/run_analysis_dryruns.sh
bash tests/repro_discovery.sh            # about 20 min; writes tests/repro-discovery/ (delete after)
python3 -B scripts/compare_d091.py --reproduce
bash tests/prefix_build_repro.sh
bash tests/standin_pipeline.sh           # about 8 min; tests/standin/ must not exist
bash tests/test_runner_refusals.sh > /tmp/r.txt; bash tests/test_chain_seed.sh; bash tests/session_sim.sh
DRY_RUN=1 APPROVED_GPU_H=10 bash runner/run_d091.sh
```

Writing a test's output over a file that SHA256SUMS lists makes the runner's package check fail until `make_sums.sh`
is rerun. The frozen hash is the one recorded in the amendment.
