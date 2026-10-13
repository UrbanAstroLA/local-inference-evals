# B1 checkpoint pass, seed 1236: proposal (PLAN.md amendment 19)

Status 2026-10-13: CPU preparation only. **Nothing has run on a GPU.** No container was started, and no B1 directory exists
in `<gpqa-runs>`. The launch needs the operator's GPU authorization.

## What runs

One GPQA Diamond pass of checkpoint **B1**:
- recipe `glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234+d7`;
- image `sha256:799d3cf8e5d14a615b89c734a6a38b4be5729bb44bb6b56effc7cf7df747b88a`, tag
  `local/glm53-v0.9.1-b1-d7:local`.

**Protocol and session:**
- all 198 questions; request seed 1236 (lm-eval `--seed` 1237);
- concurrency 8, 327,680-token cap, temperature 1.0, top_p 0.95, lm-eval parser;
- a fresh server with the shipped graph configuration, the chain's release warm-up gate, and the recording proxy;
- one pass; a 7.5 GPU-hour watchdog; no automatic retry.

**B1 composition, re-checked:**
- B1's `env.sh` differs from bundle-d1234's only in `IMAGE` and comments. Without comments, the diff is the single `IMAGE`
  line.
- B1's image layers extend B0's 104 layers unchanged (107 in all), with the same Env, Entrypoint and Cmd.
- The recorded filesystem diff shows only `sparse_attn_indexer_kpool.py` and its `.pyc` changed.

**How seed 1236 is reached** (same mechanism as screen2):
- `eval-chain.sh` sends seed 1233+pass and skips any existing `<recipe>__pass<N>` directory. Screen2 reached
  pass 2 because B0's real `__pass1` existed.
- B1 has no passes, so the runner creates two **empty placeholder** directories, `__pass1` and `__pass2`, just before the
  chain. It removes them with `rmdir` on every exit path. `GPQA_PASSES=3` then runs pass 3 only.
- After the run, `<gpqa-runs>` holds only `…+d7__pass3`. The dashboard will show one GPQA pass for B1.

**Launch:**
```bash
nohup bash <runs>/b1-screen/gpqa-b1-screen-s1236.sh \
  > <runs>/b1-screen/b1screen-run.log 2>&1 &
```

## Cost and ceiling

Occupied GPU-hours = 2 GPUs × wall time from server start to stop, with start-up and shutdown included.

| input | value |
|---|---|
| B0 seed 1234 session | 4.841 GPU-h (8,713 s; ready after 694 s; request window 2.222 h) |
| B0 seed 1235 session | 4.541 GPU-h (8,174 s; ready after 770 s; window 2.051 h) |
| overhead outside the request window | 714-790 s (0.40-0.44 GPU-h) |
| stock record seed 1236 window | 2.793 h, the longest in the 0.9.1 family (arm seed 1236: 2.744 h) |
| B0/stock window ratio at matched seeds | 0.939 (seed 1234), 0.752 (seed 1235) |

**Expected cost: 4.6-6.0 occupied GPU-hours, point estimate about 5.1.** The range comes from three window estimates:
- the scaled B1 window is 2.10-2.62 h (point 2.36 h at the mean ratio);
- the unscaled window is 2.79 h;
- adding the overhead gives 2.30-3.01 h of wall time.

The window is set by the slowest request: a cap exhaustion takes about 1.6-1.8 h.

**Ceiling: 7.5 GPU-hours** (13,500 s of wall time from the script's start), enforced by the watchdog.
- This leaves room for a request window of about 3.5 h, 26% above the longest 0.9.1-family window.
- It is 1.24× the unscaled worst case of 6.0 GPU-h and 1.55× the measured 4.84.
- A watchdog-truncated or invalid run is reported as invalid with its cost. There is no retry.

## Files (sha256)

Every file that runs is listed in `SHA256SUMS` (`73fa6661c7e4bc8ee820eb5afd50a670b33ae94bfe29a7298049f11465d05c4c`). The
runner refuses to start unless that file verifies. It also pins the external files: the chain, the proxy, the score tool,
the launcher, and B1's `env.sh` and `recipe.json`.

| file | sha256 |
|---|---|
| `gpqa-b1-screen-s1236.sh` (runner) | a4bb8e6b68c6b7adbd71b377eb3a361dfbaf2cb4864854d2d976c7f0c9af1928 |
| `b1screen_analysis.py` (frozen analysis) | 79268c4730d3c0f371cbc9afafa51680ff3d7bfbbfb1befd3827b957f2465b9c |
| `analysis_col511.py` (frozen loaders, unchanged copy) | b5b13ce2155085ca46fb24877f12b7cd0a5cdd16eb4cf19012958e7b86ae0033 |
| `FROZEN-INTERPRETATION.md` | bbfb63d7d448c6b66d7d92b87da8531254cf71df0eb682cacf7d901a854b1194 |
| `tests/refusal_tests.sh`, `tests/session_sim.sh`, `tests/test_categories.py` (CPU tests, not run at launch) | ee3d4b6d…, 52c9fb36…, f8875deb… |

## Frozen rules

The rules are in `FROZEN-INTERPRETATION.md`.

**References:**
- **Primary:** stock 0.9.1 record pass 3, seed 1236. E = 6, all at the cap; raw 171; stated 174.
- **Context:** column-511 arm pass 3, seed 1236 (E = 6, raw 171, stated 174), and stock 0.7.0 pass 3, seed 1236 (E = 1,
  raw 175, stated 177).
- The seeds were confirmed from the proxy logs (198/198 requests at 1236), from `results_*.json` (seed 1236,
  `random_seed` 1237), and from the published rows.

**Categories.** Amendment 15b's scheme is unchanged, with baseline E = 6 and the six-pass family accuracy standard.

**Amendment-19 reading:**
- **FAVOURABLE** = promising follow-up signal (E ≤ 2 with no accuracy concern);
- **UNFAVOURABLE** = a material concern (raw ≤ 158, stated ≤ 159, or 2 or more premature stops), or E ≥ 7, above every
  stock pass;
- **SIMILAR** = everything else. An accuracy concern (raw 159-163) is listed as a review item.

An unfavourable result triggers a review. It does not invalidate the independently verified corrections.

**Sensitivity readouts (not deciding):** matched offsets, and 14c's absolute rule.

**Stated answers:** hand reads of the rows where the stated adjudicator and flexible-extract differ, done after the run.
The result is PROVISIONAL until they are complete.

## Dry runs (CPU)

**Frozen analysis on existing seed-1236 data**, with arm pass 3 as a labelled stand-in (`dryrun-arm-pass3.*`):
- Valid: 41 of 41 checks. These include:
  - the published rows of the record and of 0.7.0 reproduced (0.7.0 with its published hand read, pass 2 doc 25);
  - the family constants reproduced;
  - the shipped-configuration server checks;
  - lm-eval seeds and task hash.
- E = 6, all at the cap. Category 15b: SIMILAR OR WORSE. Amendment-19 reading: **SIMILAR**. No flags.
- Against record pass 3: 2 rescued, 2 newly empty, raw and stated ±0.
- Against 0.7.0 pass 3: 1 rescued, 6 newly empty.
- Without an adjudication file, the result is PROVISIONAL, pending docs 70, 98 and 134.
- Part B (exploratory) uses B0 1234, B0 1235 and the stand-in: 12 against 16 empties, one-sided p = 0.17.

**Negative controls.** Each of these is invalid with exit code 3:
- the arm's inspect record (image ≠ B1);
- a missing server log or inspect record;
- an aborted session (no samples, no request log).

**Category boundaries:** 16 of 16 cases come out as specified (`dryrun-categories.txt`).

**Runner refusal paths** (`dryrun-refusals.txt`): 23 of 23 pass, with shims that answer read-only queries only and log
any other call. None was made. The paths covered:
- the ceiling above 7.5;
- the image missing or the tag moved;
- override variables set;
- a GLM container running;
- a port in use;
- `nvidia-smi` failing;
- a compute process present;
- a non-desktop graphics client;
- a GPU busy, or only one GPU;
- an earlier attempt, a cost line, or a pass directory (plain or tagged) already present;
- a request log already present;
- a changed frozen file.

**Session-path simulation** (`dryrun-session-sim.txt`, fake chain, no container): 30 of 30. Three scenarios were run:
normal completion, SIGTERM during the chain, and the watchdog. Each time:
- the placeholders were removed;
- there was exactly one cost line;
- the server-stop call was made;
- the chain and proxy were gone.

**Real-machine `DRY_RUN=1`:** all checks passed with the current desktop GPU load, and nothing was created
(`dryrun-runner-real.txt`).

## What is not claimed

- No significance or equivalence claim, and no D7 or individual-correction effect. B0 has no seed-1236 pass.
- No explanation of the 0.7.0 gap.
- **Seed reuse across engines** (CONFOUNDS #18) and **historical controls**: the comparators ran as the second pass of
  their sessions; this is the first pass of a fresh server.
- Records stay separate: B0 is seeds 1234 and 1235; B1 is seed 1236. Pooled scores are never called a three-pass B1
  result, and the family view is EXPLORATORY because it mixes two implementations.

## Notes for the coordinator

- **The watchdog in the earlier GPQA runners could not have stopped the chain.** It called
  `pkill -f "eval-chain.sh $R"`, where `+` in the recipe name acts as a regex quantifier, so the pattern never
  matches the chain's command line (tested).
  - Those runners would still have killed `lm_eval` and stopped the server, and the watchdog never fired in any B0
    session.
  - The B1 runner escapes the pattern, and the simulation shows the fix working.
- **B1's `recipe.json` points at B0's `env.sh`** in `launch.start`. This is cosmetic: the chain sources
  `<recipes>/<name>/env.sh` by recipe name. The file is not modified.
  - `recipe.json` is pinned in `SHA256SUMS`, so any edit to it before launch (a capture, for example) makes the runner
    refuse until it is re-hashed.
- **An attempt that aborts after creating `run-s1236`** (for example, the proxy does not come up) has used no GPU, but
  it blocks reruns until the directory is removed by hand. This is intended: no automatic retry.
- Nothing was added to `config/changes.json`.
