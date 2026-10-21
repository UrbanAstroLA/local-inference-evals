# Option D: one stock 0.9.1 GPQA Diamond pass at a fresh seed, with raw bodies, through the discovery pipeline

Package `long-reply/d091/`, built 2026-10-21 for amendment 22x. **Prepared, not run. Zero GPU was used to build it.**
It needs the operator's GPU authorization (`APPROVED_GPU_H`) and a dated amendment that records the `SHA256SUMS` hash.

Labels: **[M]** measured here; **[S]** read from source, logs or config; **[R]** reconstructed from records; **[I]** inferred.

**What this is.** One pass. The question is descriptive (22x): do 0.9.1's long replies reach the states that 0.7.0's capped
replies showed before 100k, and are 0.9.1's capped replies the same exact loops? Nothing here tests anything against
the records. About 8 replies past 100k are expected [R]. The seed differs from every comparator.

## 1. Engine and configuration (specification 1)

- **Image:** stock tpurtell 0.9.1, `ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:a9e985c0…`. Its local Id is
  `sha256:a9e985c0…` [M].
- **Recipe:** `<recipes>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1/` (`env.sh` sha256 6186bbb6…). It sets
  MODEL_PROFILE k325, GPU_MEMORY_UTILIZATION 0.95 and IMAGE to the digest.
- **Stack:** `<stacks>/glm53-flash-v0.9.0/`, the stack the recipe's `launch.stack_dir` names. Its launcher,
  local-env.sh, start.sh and model-profiles.sh are pinned by hash, as in B+ [M].
- **Weights:** K3.25 snapshot 0490d2f7. Draft: DFlash2 bf582e4e. Both are checked file by file against frozen listings
  in `checks/` [M]. The compile cache is `<cache>/vllm-v090`; it is inventoried before and after the session and
  never cleared.
- **How the record was launched [S].** The record's pass 1 ran on 2026-10-08 through
  `<eval-tools>/diag/gpqa-k325-v091-pass1.sh`. That script:
  - started the recording proxy `record_proxy.py` on 8002 in front of 8001;
  - ran `GPQA_BASE=http://127.0.0.1:8002 GPQA_PASSES=1 <stacks>/glm53-flash-v0.8.0/eval-chain.sh <recipe> gpqa`,
    which gives a fresh server, the ready-file and /v1/models gate, lm-eval `--seed 1235`, and no request-seed argument
    (lm-eval's default, 1234);
  - saved the server log and stopped the container.
  Passes 2-3 ran on 2026-10-09 through `gpqa-clean-reruns.sh`, with per-pass seeds and the `+ps` tag. The chain logs are
  `<data-runs>/benchmarks/2026-10-08-…+v0.9.1/` and `2026-10-09-…+v0.9.1+ps/`.
- **Known differences from the record that are outside the engine [S]:**
  - The chain gained `GPQA_TAG` on 2026-10-08 at 20:15, after the record's pass 1. It only renames outputs.
  - The GLM stacks moved to a loopback publish (`PORT=127.0.0.1:8001`) on 2026-10-10.
  - B+'s stock 0.9.1 session ran after both changes (2026-10-21). It produced exactly the record's non-default args,
    vLLM version, snapshots, logged env names, CUDA-graph mode, warm-up count and KV pool [M]
    (`checks/gate-calibration.txt`).

## 2. Protocol (specification 2)

The protocol is identical to the records' gpqa-diamond/v1:
- 198 questions, 8 in flight, `max_gen_toks` 327,680, temperature 1.0, top_p 0.95;
- lm-eval timeout 10,800 s and max_retries 3;
- lm-eval HEAD b954108c with the same untracked task yaml, pinned. The task hash is gated at 87e807b3b27c8661….

**lm-eval process seed:** 1235, which is `--seed 1234+pass` for pass 1, as in the records' pass 1. The request seed
70704 is sent in every request through `GPQA_SEED_LIST`.

**One session.** A fresh server, then the release warm-up gate inside the chain (ready file, then /v1/models), then the
verified warm-up checks in the analysis.

The client and request-level configuration was checked against the record's own passes [M]:
- `model_args` without the seed and `gen_kwargs` are identical across the record's three passes;
- prompt, doc, target and message hashes, and the server-rendered prompt token counts, are identical across those passes.

## 3. Request seed (specification 3): **70704**

- **Inventory:** `checks/seed_scan_d091.py` is rep070's `seed_scan.py` with one change, `SKIP_DIRS`; the diff is in
  `checks/seed_scan_d091.diff`. It skips only this package and so now also scans the rep070 refreeze package.
- **What it scanned [M]:** 20 request logs, 50 lm-eval results files, the published rows, and a text scan of 10,903 files
  under <data-runs>, the public repo and <eval-tools> (1,340 distinct seed-like values).
- **Seeds already used by GPQA:** 1234-1236, 4242, 5001-5012, 70701-70703. Request seeds seen with any prompt: 1234-1236,
  4242, 70701-70703.
- **Result:** 70704, 70705 and 70706 appear nowhere.
- **Raw bodies:** `checks/raw_seed_scan.py` opens the five gzip raw-body files the text scan skips (rep070-F s1/s2 and
  B+ v070/v091/b1). It finds 70701-70703 and B+'s 8,170,101-8,191,204, and not 70704 [M].
- **Formula-based seed bases:** B+'s scan lists none below 7,810,000 [R] (`bplus/checks/seed-formula-scan.txt`).
- The inventory is frozen (`checks/seed-inventory.json`, sha256 1f58e7b3…). The analysis pins it and gates "70704 is in
  none of its seed sets".

## 4. Validity gates (specification 4)

Any failed gate means INVALID: the analysis exits 3, there is no reading, and the long-reply pipeline does not run. The
analysis is `analysis_d091.py`, frozen.

### 4.1 Request, task and pairing gates

These come from the frozen loader (`frozen/analysis_col511.py`, sha256 b5b13ce2…, hash-checked). It loads the one pass
exactly as rep070's variant-F loader loads each of its passes: `SEED_BASE = 70704 − 1`, keep pass 1, restore `SEED_BASE`
on every exit path.

- The proxy request log and the server log are both present.
- Exactly one samples file: 198 questions, ids 0-197, both filters.
- 198 requests, with no re-sent request (at most one per message).
- Every request returns HTTP 200 and carries seed 70704, max_tokens 327,680, temperature 1.0 and top_p 0.95. There are no
  parse errors.
- Every question has its proxy row, and each proxy content hash equals lm-eval's reply hash.
- **Seeds:**
  - the request log holds exactly {70704: 198};
  - lm-eval `model_args` seed is 70704 and the process seed is 1235 (equal to the record's pass-1 process seed);
  - 70704 is fresh against the frozen inventory.
- **Client:** `model_args` without the seed and `gen_kwargs` equal the record's pass 1.
- **Task hash:** 87e807b3… in the d091 pass, and also in the record's pass 1 and rep070-F's pass 1.
- **Pairing with the 0.9.1 record's pass 1 and with rep070-F's pass 1:** prompt, doc, target and message hashes are equal
  on all 198 questions, and so are the server-rendered prompt token counts.

### 4.2 Warm-up and server gates

- **Warm-up:** the gate is rep070's `warmup_checks`, unchanged:
  - one engine start;
  - 9 release warm-up lines ending with the ready line;
  - no chat-completion request before the ready line;
  - the first GPQA request comes after the last timestamped line before the ready line.
- **Configuration identity with the record's server log.** Each of non-default args, vLLM version, model snapshots,
  logged env names and CUDA-graph mode, plus the warm-up count, must be **one of the values the record's own two server
  sessions produced**. Each set has one element, because the two sessions agree [M].
- **KV pool:** reported, plus a functional floor (section 4.3).
- **Health:** no illegal-memory-access and no NaN-warning lines inside the request window. The record's sessions have
  0 and 0 [M]. Preemption lines are reported and not gated: with the floor in place, preemption cannot come from KV
  pressure.
- **Container image Id:** a9e985c0 (from `docker inspect`). The publish address is reported, not gated, because the
  record's pass 1 predates the loopback rule (section 1). The runner itself refuses any launcher that does not resolve
  to `127.0.0.1:8001`.

### 4.3 Calibration against the record's own variation (the 22u lesson)

`checks/gate_calibration.py` applies the analysis's **own gate functions** to three sessions: the record's two
(2026-10-08 and 2026-10-09) and B+'s stock 0.9.1 session (2026-10-21). Results [M], `checks/gate-calibration.txt`:
- **Every gate passes on both record sessions and on B+'s session.** None is stricter than the record's variation.
- **Request-level gates:** every one listed in section 4.1 holds across the record's three passes.
- **KV pool, reported without an identity gate.** Two pieces of evidence decide this:
  1. *The record:* both sessions produced 4,707,515 tokens, as did all 10 default-layout (DCP1) 0.9.x K3.25 sessions in
     the startup markers [M]. A gate of "a size the record produced" would therefore be {4,707,515}, which is plain
     equality.
  2. *Pools are known to move:* the same pool measure takes several values on the DCP2/EP2 layouts:
     - stock 0.7.0: 2,749,599 / 2,758,919 / 2,768,240 over 7 sessions;
     - 0.9.0 `+ep2dcp2`: 3,140,895 / 3,165,056 over 2 sessions [M].

     B+'s INVALID verdict (22u) was exactly such an equality gate, failing on a 0.34% difference with the same
     "Available KV cache memory".

  The pool cannot change this protocol's scheduling unless it falls below what 8 requests at the cap need. So the gate
  is that floor:
  - floor = 8 × 4,352 × ⌈(327,680 + 2,795 + 16) / 4,352⌉ = **2,646,016 tokens**;
  - 4,352 is the logged attention block size, and 2,795 is the record's longest prompt;
  - the record's pool clears it by 2.06M tokens.

  The analysis still reports whether the pool equals a size the record produced. Dry run D2 shows that a pool of
  4,698,112 is VALID. Dry run N8 shows that a pool of 2,000,000 is INVALID.

### 4.4 Incomplete-run path

`incomplete_d091.py` reuses `incomplete_rep070.py` (byte-identical) for three pieces: pass completeness, the ledger reader,
and the proxy-plus-sidecar counter.
- **COMPLETE** needs all of: the session in the ledger, chain rc 0, no watchdog fire, and one samples file holding
  198 questions with both filters.
- **Anything else is INCOMPLETE:** causes, counts (completed, cap-reached, other, interrupted, unstarted) and artifact
  hashes are reported.
- An incomplete run is never classified. There are no result rows, no accuracy and no long-reply pipeline, and
  interrupted or unstarted requests are never counted as empty.
- There is no retry and no replacement run.

## 5. Runner (`runner/run_d091.sh`)

### 5.1 How it differs from rep070's runner

The runner is derived from rep070's `run_rep070.sh`. The differences:
- **One session** (pass 1). There is no `VARIANT` and no session-2 budget rule. `PLAN_MAX_GPU_H` is 10.0.
- **The 0.9.1 engine and its pins:** recipe `env.sh`, the v0.9.0 stack files, the 0490d2f7 chat template, the
  0490d2f7 and draft snapshot listings, and the vllm-v090 cache.
- **The refusals also cover:**
  - a caller-set GPU_MEMORY_UTILIZATION;
  - a launcher that does not resolve to the digest, 127.0.0.1:8001, 0.95, k325 and the v090 cache;
  - an existing `run/` or `PRIVATE/raw/`, or a non-empty `out/`;
  - `PRIVATE/` not at mode 0700;
  - the chain or proxy copy differing from rep070's copies, checked by hash.
- **Raw bodies** go to `PRIVATE/raw/s1.raw.jsonl.gz` (umask 077, mode 0700/0600).
- **Finalize:**
  - COMPLETE → `analysis_d091.py`;
  - only rc 0 (valid) → `scripts/run_pipeline.sh` (CPU only);
  - rc 3 (INVALID) or any other rc → stop;
  - INCOMPLETE → `incomplete_d091.py`.

### 5.2 What the runner reuses unchanged

- **The rep070 eval-chain copy** (sha256 81ef3386…): the record's chain plus `GPQA_SEED_LIST` plus the run-tag log name.
  It equals the original exactly up to the frozen diff, and is byte-identical to rep070's copy.
- **The rep070 recording-proxy copy** (ce09016b…): its main log is byte-compatible with the record's proxy. It adds a
  start/interrupted sidecar and the 0600 raw file.
- **`cache_inventory.py`** (d317acd0…).
- **The structure:** the escaped watchdog pattern, the desktop-graphics GPU filter, the server stopped before the proxy
  (so in-flight requests are recorded as interrupted), and cleanup and finalize on every exit path.

### 5.3 The watchdog

The watchdog fires at `APPROVED_GPU_H` / 2 GPUs of wall time from script start. It is the only early stop. It kills the
chain and lm-eval and then stops the container.

### 5.4 Run-time locations

These follow the standard record layout:

| what | where |
|---|---|
| lm-eval output | `<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+d091__pass1/` |
| proxy log, sidecar and lm-eval log | `…+d091.s1.requests.jsonl`, `….s1.requests.events.jsonl`, `….log` |
| chain log | `<data-runs>/benchmarks/<date>-…+d091/` |
| raw bodies | `PRIVATE/raw/` |
| server log, inspect, ledger, GPU log, cache inventories, analysis | `run/` |
| pipeline outputs | `out/`, `checks/prefix-verify-*.json` |
| prefixes | `PRIVATE/prefixes.json`, `PRIVATE/prompts.json` |

**Note:** like every GPQA record, the lm-eval samples file holds the reply text, and it lives in `<gpqa-runs>`. That
is the standard protocol location and lies outside this package (see section 11).

### 5.5 Command (needs the operator's authorization; no sudo)

```bash
cd <runs>/long-reply/d091
DRY_RUN=1 APPROVED_GPU_H=10 bash runner/run_d091.sh      # read-only check; creates nothing
APPROVED_GPU_H=10 bash runner/run_d091.sh                 # the run: one session, about 2.6 h wall
```

## 6. Outputs (specifications 5-6)

### 6.1 GPQA result rows

`run/rows-d091.jsonl` holds 198 rows in the published schema:
- doc_id, pass, seed;
- empty, correct_flexible / strict / stated;
- finish_reason, completion_tokens, response_chars;
- the four hashes.

All of it is computed by the frozen module.

`run/result-d091.{json,md}` reports the empty count and accuracy **beside the 0.9.1 record's per-pass range**, which the
same loader computes [M]:

| measure | 0.9.1 record, passes 1 / 2 / 3 |
|---|---|
| empty | 5 / 5 / 6 |
| raw (flexible) | 85.35 / 88.89 / 86.36 % |
| stated | 85.86 / 89.90 / 87.88 % |
| capped | 4 / 5 / 6 |

rep070-F's per-pass figures are given for context. There is no test.

**The stated figure stays provisional until the coordinator's hand read.** The analysis lists the ids where stated
differs from flexible-extract. After the read, re-run the analysis with
`--hedged 1:DOC …` and `--out run/result-d091-final.json`.

### 6.2 P(finish | still generating at 100,000 tokens)

- The definitions and functions are rep070's secondary, unchanged.
- It is reported for d091 (pass 1), the 0.9.1 record (passes 1-3, and pass 1 alone) and rep070-F, each with numerator,
  denominator and question count.
- Each group gets a question-bootstrap 95% interval (2,000 replicates). The differences d091 − record and
  d091 − rep070-F use the joint question bootstrap.
- With one pass, the d091 bootstrap is a reply bootstrap.

### 6.3 Long-reply pipeline (CPU only, `scripts/run_pipeline.sh`)

The pipeline is run on the d091 replies of 50k tokens and longer, under the discovery's set definitions:
- at risk: T ≥ 100k;
- capped: finish length at 327,680;
- comparison: 50-100k.

Its steps:

| step | file | what it is | what it does |
|---|---|---|---|
| 1 | `scripts/index_d091.py` | the index part of discovery's `build_index.py`, with only the raw path and request logs as arguments | writes `out/rep070F_index.json`, which keeps discovery's name because the unchanged scripts read that path. On rep070-F it is byte-identical to discovery's. |
| 2 | `scripts/trajectory.py` | byte-identical; PRESPEC 97040a5c checked | per-window metrics, R50 terminal onset, episodes, the **prespecified** loop class |
| 3 | `scripts/periods.py` | byte-identical | the **post hoc** LRS and lag supplement |
| 4 | `scripts/compare_d091.py` | uses analyse.py's own definitions (section 6.4) | writes `out/tables_d091.{md,json}`, side by side against rep070-F |

Steps 2 and 3 run in discovery's CPU-only container (`local/glm53-v0.7.0-kpoolring:local`, Id f68e47a0…).

### 6.4 The side-by-side tables (`compare_d091.py`)

The helper definitions come from discovery's `analyse.py` itself: an AST extraction of `med`, `cls_set`,
`reply_medians`, `tw_cpt`, `tw`, `auc` and `METRICS` from the byte-identical copy. The groupings mirror T1, T3, T4, T12
and T13. The tables are:
1. **First 100k tokens.** Chars/token, digit share, and interjection (`chk_wait`) and verification (`chk_verify`) rates.
   Medians are given for capped and for finished ≥ 100k, with AUC. The T1 per-set medians are given as well.
2. **Capped replies, and finished replies with a terminal onset.** For each: the onset (R50 ≥ 0.5) with the 0.3 and 0.7
   sensitivity; R50 at 100k; the loop class under the prespecified rule [P]; the post hoc LRS class [PH]; and the
   episodes. Both labels are kept, as in discovery.
3. **High-repetition episodes and recovered episodes**, by set, plus the replies that finished from inside repetition.
4. **P(finish | at 100k)**, from section 6.2.
5. **Question overlap with the discovery's long-reply panel:**
   - d091's at-risk and capped questions;
   - which were ever long in the panel's 15 logged passes;
   - which are also at risk in rep070-F;
   - which are long in the stock 0.9.1 record or in both families;
   - which of discovery's 14 both-family questions go long again.

**Proof that compare_d091.py changes nothing:** `compare_d091.py --reproduce` recomputes the rep070-F side from
discovery/out and matches discovery's `summary.json` exactly on T1, T3, T4, T12, T13 and drift.

**All of this is descriptive.** It is one pass, with about 8 long replies expected and different seeds. It cannot
adjudicate a termination mechanism, and reaching 100k selects different trajectories on each engine.

## 7. Prefix export (specification 7). Nothing is continued in this package.

- **Rule:** B+'s rule (`scripts/prefix_select_d091.py`).
  - Pool: T ≥ 100k.
  - Eligibility: N_reasoning > 100,000, and R50 at 100k defined and < 0.05.
  - Strata: capped first, then finished; one reply per question.
  - Ranking: questions by discovery's panel count, with salted-hash ties (salt `d091-2026-10-21:`).
  - At most 6 per stratum.
- **Selection code:** the function is `select_prefixes.main()`'s body with its inputs as parameters. B+'s file asserts
  B+'s pool size of 29, so it cannot run on d091 unchanged. Run on B+'s inputs, the function reproduces
  `bplus/out/selection.json` **byte for byte**, and so does the byte-identical copy of `select_prefixes.py`.
- **Construction:** B+'s `build_prefixes.py`, **byte-identical** (sha256 1ebbec2e…). It is called through
  `scripts/prefix_build_d091.py`, which sets only the module constant `REQLOGS` to the d091 request log.
  - It builds in the 0.7.0 image, re-tokenizing from text and cutting at 100,000 reasoning tokens.
  - It verifies in the 0.9.1 and B1 images: the 0.9.1 template render must be identical and the hashes equal.
  - With B+'s inputs, the wrapper rebuilds B+'s `prefix_manifest.json` and all three `prefix-verify-*.json` byte for byte,
    and its `PRIVATE/prefixes.json` has B+'s sha256.
- **Files:** token ids and prompt messages stay in `PRIVATE/` (0600). Outside it there are only ids, hashes, lengths and
  covariates (`out/selection.json`, `out/prefix_manifest.json`).

## 8. Costing (specification 8): `cost/cost_d091.py`, `cost/cost.{json,txt}`

### 8.1 Reconstruction [R]

`cost/reconstruct_timing.py` is rep070's script, byte-identical, and its output is byte-identical to rep070's. It gives
the record's pass-1 session:
- **2.574 h wall = 5.15 GPU-h** (17:26:47 → 20:01:12);
- request window 142.0 min, plus 12.4 min of start-up and stop. "Ready" came after 725 s; B+'s 0.9.1 session was ready
  after 680 s.

### 8.2 Model [I]

The model is rep070's budget model applied to this pass:
- window = base + slope × caps;
- slope: empirical 12.78 min/cap, fitted to the record's three passes (4, 5 and 6 caps; 142.0, 163.7 and 167.6 min);
  token-based 12.58 min/cap (327,680 tokens at 434 tok/s);
- per-question cap probabilities: a beta-binomial (frozen `fit_bb`) fitted to the record's cap counts, giving 4.93
  expected caps per pass;
- pass noise: 5% (central) or 10% (pessimistic).

| | occupied GPU-hours |
|---|---|
| **central** | **5.15** (record session [R]); model median 5.6 |
| **pessimistic** (95th percentile; +15 min overhead, noise sd 10%) | **7.9** |
| **ceiling** (pessimistic 99.9th percentile 9.74, rounded up to 0.5) | **10.0** = `APPROVED_GPU_H` required = the runner's `PLAN_MAX_GPU_H` |

### 8.3 P(ceiling)

P(ceiling) is the probability that the watchdog stops the run and it is reported INCOMPLETE, never classified:

| scenario | P(ceiling) |
|---|---|
| 0.9.1-like long-reply rate, central | 0.004% |
| 0.9.1-like long-reply rate, pessimistic | 0.05% |
| twice the 0.9.1 cap rate (stress) | 6.0% |

There is no invalid-run allowance. An invalid or incomplete run means stop and amend.

### 8.4 Disk and timeout

- **Raw bodies:** about 3.4 MB central (the record's chars per pass × rep070-F's measured 0.287 gzip bytes/char),
  5.6 MB pessimistic, and about 102 MB in the worst case (all 198 capped). Add about 13 MB of request, server and
  lm-eval files. The runner refuses to start with less than 5 GiB free.
- **Timeout margin:** the record's longest request was 5,529 s, against lm-eval's 10,800 s timeout. A request past the
  timeout would be re-sent, and the "no re-sent request" gate would make the run INVALID.

## 9. CPU checks before freezing (specification 9)

These are in `CHECKS.md`. In summary:
- analysis dry runs: 2 valid stand-ins and 15 negative cases;
- the discovery pipeline on rep070-F: 12 of 12 outputs byte-identical;
- index, comparison, selection and construction reproductions;
- the stand-in pipeline end to end;
- runner refusals, the proxy copy, chain seeds and the session simulation, including a watchdog stop mid-request;
- the gate calibration, the privacy scan, and a real-machine `DRY_RUN=1`.

## 10. Reuse map (byte identity proven by hash)

| file here | source | sha256 |
|---|---|---|
| `frozen/analysis_col511.py`, `.md`, `power_col511.py` | rep070/refreeze/frozen | b5b13ce2…, 523ec228…, 92466804… |
| `analysis_rep070.py`, `incomplete_rep070.py` | rep070/refreeze | 54c7fa4f…, 5a682182… |
| `runner/local-eval-chain-rep070.sh` + `.diff`, `record_proxy_rep070.py` + `.diff`, `cache_inventory.py` | rep070/refreeze/runner | 81ef3386…, ce09016b…, d317acd0… |
| `cost/reconstruct_timing.py` | rep070/refreeze | b0ee57ba… |
| `checks/seed_scan.py` (and the `_d091` copy with its diff) | rep070/refreeze/checks | 3d8e80e4… |
| `PRESPEC.md` | discovery | 97040a5c… |
| `scripts/trajectory.py`, `periods.py`, `build_index.py`, `analyse.py`, `run_trajectory.sh` | discovery/scripts | c24267b1…, 7f5b7e88…, 43b6c311…, 1e599f2f…, bdd3fb7b… |
| `scripts/select_prefixes.py`, `build_prefixes.py` | bplus/scripts | b998bf67…, 1ebbec2e… |
| `checks/snapshot-0490d2f7.txt`, `snapshot-dflash2-bf582e4e.txt` | bplus/checks, rep070/refreeze/checks | 3d981610…, 10777a68… |
| `tests/test_proxy_sidecar.py` | rep070/refreeze/tests | (unchanged; it tests `../runner/record_proxy_rep070.py`) |

## 11. Privacy

- **Text stays in `PRIVATE/`.** GPQA question, answer, letter and reply text appears only there: the raw bodies at run
  time, and the prefixes and prompts. `PRIVATE/` is mode 0700 and its files 0600, written under umask 077.
- **Elsewhere:** files outside `PRIVATE/` hold ids, hashes, counts, positions and metrics only.
- **Scans [M]:** `checks/privacy_scan.py` checks every file outside `PRIVATE/` for:
  - 40-character windows of all 198 prompts;
  - windows of every raw reply;
  - 12-id token runs of every prefix.

  The privacy scan runs at the end of the pipeline too.
- **Scratch:** inputs that held text during the tests were built under `PRIVATE/` and shredded afterwards.
- **Exception (as for every record):** lm-eval writes its samples file, with the reply text, to `<gpqa-runs>/…__pass1`,
  the standard location that the analysis reads. Keeping it inside `PRIVATE/` would need a change to the chain copy,
  which would break the byte identity with rep070's copy. The coordinator decides.

## 12. Not verified, or open

- **Any GPU behaviour:** start-up, decode rates, whether the warm-up and the KV pool repeat. The checks were
  stand-ins, logs and shims only.
- **The positive pipeline path has not run on a 0.9.1 raw file,** because none exists yet. It ran on rep070-F's pass-1
  raw file. The 0.9.1 template render for any newly selected question is checked only at run time, by the verify step.
- **The session simulation uses a fake chain and a docker shim.** The real chain's interaction with the proxy copy was
  last exercised in rep070-F (2026-10-16) and B+ (without the chain).
- **Stated accuracy** needs the coordinator's hand read.
- **The cost model is linear in caps.** With 8 lanes, overlapping caps make it conservative [I].
