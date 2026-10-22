# CHECKS: option D2 package (d091b, amendment 22ad), CPU only, 2026-10-21

D2 is a copy of option D's frozen package (`d091/`, amendment 22x). It differs only in:
- the request seed, **70705**;
- the run tag, `+d091b`;
- its references to its own folder;
- its labels.

`PROPOSAL.md` section 13 lists every changed file with before/after sha256.

**Zero GPU.** Nothing here touched a model:
- no model server was started;
- no container ran with `--gpus`;
- no request went to a model server;
- no `sudo` was used.

Only read-only docker commands ran (`docker ps`, `docker image inspect`). Inputs that held text existed only under
`PRIVATE/` and were shredded afterwards. `PRIVATE/` is empty at freeze, and `out/` is empty.

Labels: **[M]** measured here (re-run for D2); **[D]** carried over from D's package. A [D] check was not re-run because the
code it exercises is byte-identical to D's, or differs only in seed or tag constants and docstrings.

| # | check | evidence | result |
|---|---|---|---|
| 1 | Fresh request seed | `checks/seed-scan.out`, `seed-inventory.json`, `raw-seed-scan.out`, `seed-text-hits.txt` | 70705 is in no request log, no lm-eval results config, no published row and no gzip raw body. Scanned: 22 request logs, 51 results files, 27 published-row files and 6 raw-body files, including D's request log, sidecar, lm-eval output and raw bodies, and B+'s three raw files. The text scan (11,550 files) finds 10 hits, **all of them D's synthetic negative N1** (and its public copy); none is a request. Under D's rule as written, `seed-scan.out` marks 70705 ineligible for that reason. Disclosed in PROPOSAL §3; 70706 is clean everywhere and is the fallback [M] |
| 2 | Gate calibration | `checks/gate-calibration.txt`, `.json` | All 12 server gates pass on both 0.9.1 record sessions, on B+'s session and **on D's own 2026-10-21 session** (new input). Client config, per-question hashes, prompt tokens and task hash are identical across the record's 3 passes. KV pool: 4,707,515 in 11/11 default-layout 0.9.x sessions, D's included. The gates and the floor (2,646,016) are unchanged; the script exits 0 [M] |
| 3 | Analysis dry runs | `tests/analysis-dryruns.txt`, `tests/dryruns/` | **17 / 17 as expected** on **D's real pass relabelled to 70705** (detail below) [M] |
| 4 | Discovery pipeline unchanged | `tests/repro-discovery.txt` | 12 / 12 outputs byte-identical (scripts byte-identical to D's) [D] |
| 5 | Index adapter | `tests/index-repro.txt` | Byte-identical to discovery's index. `index_d091.py` changed only in its docstring (`checks/changes-from-d091.diff`) [D] |
| 6 | Side-by-side tables use discovery's definitions | `tests/compare-repro.txt` | `compare_d091.py --reproduce` re-run: **ALL SAME**, and the output is byte-identical to D's. Only the docstring and the default label changed [M] |
| 7 | Prefix selection = B+'s rule | `tests/prefix-select-repro.txt` | Byte-identical to `bplus/out/selection.json`; `prefix_select_d091.py` is byte-identical to D's (salt unchanged) [D] |
| 8 | Prefix construction = B+'s code, unchanged | `tests/prefix-build-repro.txt` | B+'s outputs rebuilt byte for byte through the wrapper. In the wrapper only the default SEED and REQLOG constants changed; the reproduction used `--bplus-defaults` [D] |
| 9 | Whole pipeline, stand-in pass | `tests/standin/` | All 9 steps rc 0 on rep070-F pass 1 [D]. D's real run then took the same path on a 0.9.1 raw file (`d091/run-d091.out`: pipeline rc 0) |
| 10 | Runner refusals (DRY_RUN=1) | `tests/runner-refusals.txt` | **22 / 22**: the same refusal set as D. The real machine passes at 10.0 and at 8. DRY_RUN creates nothing (no `+d091b` file, no run/, no PRIVATE/raw/, empty out/) [M] |
| 11 | Recording-proxy copy | `tests/proxy-sidecar.txt` | 17 / 17; the proxy copy is byte-identical to D's and to rep070's [D] |
| 12 | Chain seed lines | `tests/chain-seed.txt` | **6 / 6**: pass 1 sends `,seed=70705` and the process seed is 1235. The chain copy equals the original up to the frozen diff and is byte-identical to rep070's copy [M] |
| 13 | Session simulation (shims, no GPU) | `tests/session-sim.txt` | **67 / 67** over the six scenarios (detail below) [M] |
| 14 | Output paths | `tests/output-paths.txt` | Every lm-eval, proxy, chain-log, raw, run, out, prefix-verify and prefix path differs from D's and does not exist yet. The runner's pre-flight glob for `+d091b` matches none of D's 5 record paths [M] |
| 15 | Costing | `cost/` | Unchanged from D: central 5.15 / pessimistic 7.86 / ceiling **10.0** GPU-h. D's real session used 5.884 GPU-h (`d091/run/costs.jsonl`), between central and pessimistic [D; D's actual M] |
| 16 | Privacy | `checks/privacy-scan.txt` | 0 prompt-text, 0 reply-text and 0 token-id-run hits in every file outside `PRIVATE/`. Needles: windows of all 198 prompts, of D's 198 raw replies (the stand-in pass), and, in a second scan, of rep070-F's 594 raw replies. `PRIVATE/` is 0700 [M] |
| 17 | Real-machine DRY_RUN=1 at 10.0 | `tests/dry-run-real.txt` | Every refusal check passed on this machine (GPUs idle apart from the desktop). The planned call shows `GPQA_TAG=+d091b`, `GPQA_SEED_LIST='70705'`, lm-eval `--seed 1235` and d091b paths. Nothing was created [M] |

## Detail for check 3 (analysis dry runs)

The stand-in is **option D's real 2026-10-21 pass**, read with its own session log (`d091/run/server-s1.log`). It comprises:
- the samples and results from `+d091__pass1`;
- the proxy request log `+d091.s1.requests.jsonl`.

Its request-log rows and lm-eval `model_args` seed were relabelled from 70704 to 70705 in a scratch copy under `PRIVATE/`.
D's own package used the 0.9.1 record's pass 1 here.

**Valid, as expected:**
- **D1**, the stand-in: valid; empty 6, raw 171. Its 198 result rows equal D's own `d091/run/rows-d091.jsonl` in every field
  except the seed [M].
- **D2-kv-other-size** (a dry-run name from D, unrelated to this package's name): the KV pool edited to 4,698,112, above the
  floor and not a size the record produced. Valid, which shows the pool is not identity-gated.

**Invalid (exit 3), as expected, N1-N15.** The cases are D's, applied to the new stand-in:

| case | what was altered |
|---|---|
| N1 | wrong request seed: **70704**, D's own seed (D used 70705 here) |
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

## Detail for check 13 (session simulation)

| scenario | outcome |
|---|---|
| (a) normal end of the session | one cost line; the chain is called with +d091b / 1 pass / 70705 / 8 / proxy; the server is stopped once; logs and inspect are saved; loopback; cache inventories; planned seeds 70705 / 1235; the placeholder pass is reported INCOMPLETE and never classified |
| (b) SIGTERM | rc 4, cleanup, chain_rc null, INCOMPLETE |
| (c) watchdog | chain killed, server stopped, watchdog_fired true, INCOMPLETE with the watchdog cause |
| (d) chain failure | no retry, chain_rc 1, INCOMPLETE |
| (e) watchdog with a request in flight through the proxy | the sidecar records start then interrupted for seed 70705; no main-log row; the interrupted raw record is in `PRIVATE/raw`, mode 0600; INCOMPLETE with interrupted 1, unstarted 197, never counted empty |
| (f) complete pass directory without proxy rows | probe COMPLETE, frozen analysis INVALID (rc 3), pipeline not run |

Across all scenarios: no refused docker call, and no process is left behind.

## Not verified

- **GPU behaviour of D2's session.** D's session passed every gate the same day on the same image and configuration, but
  that says nothing about D2's own run.
- **The 0.9.1 template render for any newly selected question.** It is checked only at run time, by the verify step.
- **The [D] checks** were not re-run (see the table).
- **Comparison against D.** Putting D2's analysis or tables beside D's would need a code change, and none was made.
  `out/tables_d091.md` and `run/result-d091-final.md` of the two packages can be read side by side. Both are against the
  same rep070-F side and the same record range.

## How to rerun (all CPU only)

```bash
cd <runs>/long-reply/d091b
python3 -B checks/seed_scan_d091.py --out PRIVATE/x.json --candidates 70705 70706   # about 2 min; exits 3 (text-scan hits, see check 1)
python3 -B checks/raw_seed_scan.py PRIVATE/y.json 70705 70706
python3 -B checks/gate_calibration.py PRIVATE/g.json
bash tests/run_analysis_dryruns.sh
python3 -B scripts/compare_d091.py --reproduce
bash tests/test_runner_refusals.sh > PRIVATE/r.txt; bash tests/test_chain_seed.sh; bash tests/session_sim.sh > PRIVATE/s.txt
DRY_RUN=1 APPROVED_GPU_H=10 bash runner/run_d091.sh
```

Writing a test's output over a file that `SHA256SUMS` lists makes the runner's package check fail until `make_sums.sh` is
rerun. Remove any `PRIVATE/` scratch file before a real run. The frozen hash is the one recorded in the amendment.
