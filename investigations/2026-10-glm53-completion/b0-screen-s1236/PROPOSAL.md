# B0 checkpoint pass, seed 1236: proposal (review step after B1's UNFAVOURABLE seed-1236 reading)

Status 2026-10-13: CPU preparation only. **Nothing has run on a GPU.** No container was started, and no B0 pass-3
directory or seed-1236 B0 request log exists in `<gpqa-runs>`. The launch needs the operator's GPU authorization.

## Why

B1's seed-1236 pass came out at E = 9 against the stock family's 4-6, which is UNFAVOURABLE under amendment 19 and so
triggers a review.

The D7 reconstruction finds no D7 exposure on GPQA prompts. Where D7 does not fire, B1's writes are byte-identical to
B0's. One B0 pass at the same seed asks a narrow question: **does B0, without D7, show the same excess?**

## What runs

One GPQA Diamond pass of checkpoint **B0**:
- recipe `glm53-flash-exl3-k3.25@tpurtell+v0.9.1+bundle-d1234` (env.sh `3125683b…`, recipe.json `318996c4…`);
- image `sha256:08d1cb902ddb888f679b972714168bb7c0d2333b4c1865f3ebf860baa2f46548`, tag
  `local/glm53-v0.9.1-bundle-d1234:local`.

**Protocol and session:** unchanged from B1's pass.
- all 198 questions; request seed 1236 (lm-eval `--seed` 1237);
- concurrency 8, 327,680-token cap, temperature 1.0, top_p 0.95, lm-eval parser;
- a fresh server with the shipped graph configuration, the chain's release warm-up gate, and the recording proxy;
- one pass; 7.5 GPU-hour watchdog; no automatic retry.

**How seed 1236 is reached.** It is reached the natural way, with no placeholders.
- B0's real `__pass1` (seed 1234) and `__pass2` (seed 1235) exist.
- `eval-chain.sh` skips any existing `<recipe>__pass<N>` directory and sends seed 1233+pass.
- `GPQA_PASSES=3` therefore runs pass 3 only, as the first pass of a fresh server, as B1's pass was.
- The runner refuses unless both earlier passes are present with their frozen samples hashes, and re-checks them
  afterwards.
- After the run, `<gpqa-runs>` gains `…+bundle-d1234__pass3` and `…+bundle-d1234.s3.requests.jsonl`.
- The chain's score step will then score B0 passes 1-3 together. That display is what the dashboard shows; it is not
  this review's analysis.

**Launch:**
```bash
nohup bash <runs>/b0-screen-s1236/gpqa-b0-screen-s1236.sh \
  > <runs>/b0-screen-s1236/b0s1236-run.log 2>&1 &
```

## Runner changes against B1's runner

`gpqa-b0-screen-s1236.sh` is a copy of `../b1-screen/gpqa-b1-screen-s1236.sh` (`a4bb8e6b…`). The changes:
- **Pins:** the recipe name, image id and tag, and `env.sh` hash are B0's. The output directory is `b0-screen-s1236/`
  (`run-s1236`, `costs.jsonl`). The request log is `<B0>.s3.requests.jsonl`, the server log `server-s3.log` and the
  inspect record `inspect-s3.json`; the cost line records session 3, checkpoint B0. The frozen analysis is
  `b0s1236_analysis.py`.
- **Placeholder logic removed:** no `mkdir` or `rmdir` of `__pass1` or `__pass2`, and no placeholder cleanup.
- **Pass-state refusals replaced:**
  - `__pass1` and `__pass2` must be present, each with exactly one samples file at its frozen hash;
  - refuse if any `__pass3`-or-later directory exists, if `.s3.requests.jsonl` exists, or if any existing
    `<B0>.*requests.jsonl` holds a seed-1236 request.
  - B1's "tagged pass directory" glob `<R>+*__pass*` is dropped: for the B0 name it would match B1's own
    `+d7__pass3`. `GPQA_TAG` stays refused, so no tagged directory can be written.
- **After the session:** passes 1-2 are re-checked (prints "unchanged" or a WARN).

Everything else is unchanged:
- `SHA256SUMS` verification, including the chain, proxy, score tool, launcher files, and B0's `env.sh` and `recipe.json`;
- the override-variable, GLM-container, port and GPU refusals;
- the proxy-up gate;
- the escaped watchdog pattern;
- the chain run in the background and waited for;
- cleanup on EXIT, INT and TERM, with one cost line;
- `DRY_RUN=1`.

## Cost and ceiling

Occupied GPU-hours = 2 GPUs × wall time from server start to stop, with start-up and shutdown included.

| session | GPU-h | wall | request window | overhead outside the window | cap exhaustions (latest start → end) |
|---|---|---|---|---|---|
| B0 seed 1234 | 4.841 | 8,713 s | 2.222 h | 714 s | 3 (1.15 h → 2.22 h) |
| B0 seed 1235 | 4.541 | 8,174 s | 2.051 h | 790 s | 3 (1.11 h → 2.05 h) |
| B1 seed 1236 | 6.330 | 11,394 s | 2.967 h | 713 s | 9 (1.82 h → 2.90 h; longest 1.78 h) |
| stock record seed 1236 (context, request window only) | — | — | 2.793 h | — | 6 (1.95 h → 2.80 h) |

The request window is set by the last cap exhaustion to finish.

**Expected cost: 4.6-6.4 occupied GPU-hours, point estimate about 6.0.**
- If B0 behaves at this seed like its earlier passes (few caps; B0/stock window ratio 0.75-0.94 on 2.79 h), the window is
  2.1-2.6 h, or 4.6-5.7 GPU-h.
- If B0 behaves like B1 or stock at this seed, the window is 2.8-3.0 h, or 6.0-6.4 GPU-h.
- The point estimate leans to the second case because the seed and, by the reconstruction, the computation are B1's.

**Ceiling: 7.5 GPU-hours** (13,500 s of wall time from the script's start), enforced by the watchdog.
- With about 0.2 h of overhead, this allows a request window of about 3.55 h: 20% above B1's 2.97 h, and 1.17 GPU-h
  above B1's actual cost.
- **Truncation risk is not zero.** The latest observed cap start (1.95 h) plus the longest observed cap duration
  (1.78 h) gives a 3.73 h window, about 7.9 GPU-h. No observed pass came close to that combination; the largest
  measured window is B1's 2.97 h.
- A watchdog-truncated run is reported as invalid with its cost. There is no retry. Raising the ceiling is an operator
  decision and is not proposed here.

## Frozen rules (`FROZEN-INTERPRETATION.md`)

**Primary for this review:** B0 against B1 at seed 1236. B1 is E = 9, all at the cap; raw 173; stated 174.

| reading | condition | statement |
|---|---|---|
| **(a)** | E ≥ 8 | **The seed-1236 excess is shared by B0.** It is a property of the corrected family at this seed, not attributable to D7. |
| **(b)** | E ≤ 6 | **B0 is within the stock range (4-6).** The B0/B1 difference is either D7, which would contradict the zero-exposure reconstruction, or run-to-run variation. One pair of passes cannot separate them. |
| **(c)** | E = 7 | **No reading.** |

**Identity lean, used only in (b).**
- I_ne is the number of non-empty replies with identical `content_sha256` and `completion_tokens`.
- I_ne ≥ 20: near-deterministic. The predefined follow-up is a CPU re-check of the reconstruction; no D7 claim.
- I_ne ≤ 2: batching nondeterminism dominates. The difference is expected from run-to-run variation alone.
- I_ne 3-19: no lean.
- I_ne is also reported on its own: NEAR-DETERMINISTIC (≥ 20) / PARTIAL / BATCHING NONDETERMINISM DOMINATES (≤ 2).

**Why the threshold justification matters.** Every previously cited "identical reply" count is made only of pairs where
both replies are empty at the cap, with differing reasoning lengths:

| pair | identical replies | non-empty identical (I_ne) |
|---|---|---|
| B1 vs stock, seed 1236 | 5/198 | 0 |
| B1 vs arm, seed 1236 | 5/198 | 0 |
| B0 vs stock, seed 1234 | 2/198 | 0 |
| B0 vs stock, seed 1235 | 3/198 | 0 |
| stock vs arm, seed 1236 | 4/198 | 0 |
| stock vs arm, seed 1235 | 0/198 | 0 |

The raw identity count is therefore a function of the empty overlap, not of determinism. The analysis reads I_ne only.
B0/B1 is the first same-computation pair.

**Other measures:**
- empty replies by cause for both passes;
- rescued, newly empty and both-empty lists;
- empty-status agreement (2×2);
- raw and stated, with the same adjudicator and hand-read rule;
- generated tokens, with a paired B0 − B1 difference.

**First divergence cannot be measured:** there are no token-level logs.

**Review items** are listed with the reading and never change it: a material concern or an accuracy concern on the
family standard.

**Continuity (part A):** amendment 15b's category for B0 against stock 0.9.1 at seed 1236, unchanged (baseline E = 6,
six-pass family standard). The amendment-19 FAVOURABLE/SIMILAR/UNFAVOURABLE reading is also given, for continuity only.

**Exploratory (part B):**
- **B-1:** the B0 three-seed record (seeds 1234, 1235 and 1236) as one implementation, against stock record passes 1-3
  and arm passes 1-3. It uses the exact sign-flip and the 20,000-replicate question bootstrap. It is exploratory because
  the controls are historical, seeds are reused across engines, and the seed-1236 pass was chosen after B1's pass had
  been seen.
- **B-2:** the B1 package's two-implementation view, recomputed as a loader check.
- B1 stays a separate record.

**Stated answers:** rows where the stated adjudicator and flexible-extract differ are read by hand after the run. Until
then the category is PROVISIONAL. The B0/B1 reading depends only on E and I_ne, so it is final once the pass is valid.

## Files (sha256)

`SHA256SUMS` (`5615737c7955742f4217d1c1fd4126ca444232322bd0d5c75cfdbdd876dbac76`) covers every file that runs. The runner
refuses to start unless it verifies. Its external pins are identical to the B1 package's, except for B0's `env.sh` and
`recipe.json`.

| file | sha256 |
|---|---|
| `gpqa-b0-screen-s1236.sh` (runner) | 211bfff41c109825409542d9cbc483787ceb128fe4f82d97b8648f4d547d943d |
| `b0s1236_analysis.py` (frozen analysis) | f707d2003ddc7553235db8ef31071ce26e2adaec91294e43098cd945eb3a7477 |
| `analysis_col511.py` (frozen loaders, unchanged copy) | b5b13ce2155085ca46fb24877f12b7cd0a5cdd16eb4cf19012958e7b86ae0033 |
| `FROZEN-INTERPRETATION.md` | 42262e39707b3b461350c6f9c43ecc1673e61ebb53481cbc1c0c7fe29fea6f1c |
| `tests/refusal_tests.sh`, `tests/session_sim.sh`, `tests/test_readings.py` (CPU tests, not run at launch) | bfda6b7c…, d0946374…, 4cb2e3e9… |

## Dry runs (CPU)

**Analysis with the B1 pass as a labelled B0 stand-in** (`dryrun-b1-standin.*`):
- valid, 49 of 49 checks;
- the reading is (a) at E = 9;
- identity is 198/198 **by construction** (I_ne 189 plus 9 both-empty), because the stand-in is B1 itself;
- part A reproduces B1's final part A: against record pass 3, rescued 170 and newly empty 8, 69, 130 and 145;
- B-2 reproduces B1's part B.

**Analysis with the column-511 arm's pass 3 as a stand-in** (`dryrun-arm-standin.*`):
- valid, 49 of 49;
- E = 6, reading (b) with the lean "run-to-run";
- I_ne = 0 (5 both-empty), so the identity reading is BATCHING NONDETERMINISM DOMINATES;
- the B-1 view runs.

**Controls** (`dryrun-controls.txt`):
- B1's inspect record makes the run invalid (image ≠ B0), exit 3;
- B0's seed-1235 inspect record is accepted;
- missing server log and inspect record: invalid;
- an aborted session: invalid;
- with no adjudication, the category is PROVISIONAL and the reading final.

**Boundary tests** (`dryrun-readings.txt`): 41 of 41. These cover:
- the B1 package's 16 category cases through the unchanged functions;
- the (a)/(b)/(c) and identity-lean boundaries;
- `identity()` on synthetic rows.

**Runner refusal paths** (`dryrun-refusals.txt`): 30 of 30, with read-only shims; no other call was made. Added for B0:
- a pass 3, 4 or 12 directory exists;
- the s3 log exists;
- a B0 log holds a seed-1236 request;
- B0 logs at other seeds are accepted;
- B1's files are not mistaken for B0's;
- pass 1 is missing;
- pass 2 does not match its frozen hash.

**Session-path simulation** (`dryrun-session-sim.txt`): 34 of 34. The fake chain uses the real skip rule, and B0's real
passes 1-2 are symlinked read-only. Three scenarios were run: normal completion, SIGTERM, and the watchdog. In each:
- passes 1-2 were skipped, and pass 3 ran at seed 1236 / 1237;
- there was exactly one cost line;
- the server-stop call was made;
- the chain and proxy were gone;
- the real pass-1/2 samples were unchanged.

**Real-machine `DRY_RUN=1`** (`dryrun-runner-real.txt`): all checks passed with the current desktop GPU load (GPU0 20 MiB,
GPU1 2,245 MiB). Nothing was created.

## What is not claimed

- No significance or equivalence claim.
- No D7 effect or D7 absence beyond the predefined reading. One pair of passes cannot separate D7 from run-to-run
  variation.
- No explanation of the 0.7.0 gap.
- **Seed reuse across engines (CONFOUNDS #18) and historical controls.** The stock comparators were second passes of their
  sessions; B0 and B1 were each the first pass of a fresh server.
- **Records stay separate:** B0 at seeds 1234, 1235 and 1236; B1 at seed 1236. The B0 three-seed view is exploratory, and
  nothing is labelled a three-pass B1 result.

## Notes for the coordinator

- **The identity counts quoted in the brief are trivial.** The 5/198, 5/198 and 2/198 identical replies are all pairs
  where both replies are empty at the cap, and reasoning lengths differ in every one. No non-empty reply has ever matched
  across builds at a shared seed. The frozen reading therefore uses I_ne.
- The watchdog pattern for B0 (`local-eval-chain\.sh glm53-…\+bundle-d1234`) is a prefix of B1's chain command line.
  This is harmless: only one chain can run (GLM-container refusal), and the pattern construction is unchanged from the
  B1 runner.
- A stray `b1-screen/run-s1236/token-analysis-posthoc.md` (12:52) was not written by this preparation. `b1-screen/` is
  otherwise unchanged, and its `SHA256SUMS` verifies.
- Nothing was added to `config/changes.json`.
