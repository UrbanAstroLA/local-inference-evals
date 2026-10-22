# Option D2 (amendment 22ad): a second stock 0.9.1 GPQA Diamond pass at a fresh seed, with raw bodies, through the discovery pipeline

Package `long-reply/d091b/` (**D2**), built 2026-10-21 for amendment 22ad (public PLAN.md: "D2: a second text-logged stock 0.9.1
GPQA pass, identical to option D apart from a fresh request seed and run tag"). It is a copy of option D's frozen package
`long-reply/d091/` (amendment 22x; run 2026-10-21, valid), changed only in the request seed (70705), the run tag (`+d091b`),
the package's references to its own folder, and these labels. Every changed file is listed with before/after sha256 in
section 13. The rest of this text is D's, with D's numbers where they are D's evidence.
**Prepared, not run. Zero GPU was used to build it.**
It needs the operator's GPU authorization (`APPROVED_GPU_H`) and a dated amendment that records the `SHA256SUMS` hash.

Labels: **[M]** measured here; **[S]** read from source, logs or config; **[R]** reconstructed from records; **[I]** inferred.

**What this is.** One pass. The question is descriptive (22x): do 0.9.1's long replies reach the states that 0.7.0's capped
replies showed before 100k, and are 0.9.1's capped replies the same exact loops? Nothing here tests anything against
the records. About 8 replies past 100k are expected [R]. The seed differs from every comparator.

**Why D2 (22ad).** The operator decided to "get more prefixes": D2 is a second pass of the same engine and protocol, so
that option E can draw on the union of D's and D2's eligible 0.9.1-built replies. D2 changes nothing in what is
measured or how. Its seed differs from D's (70704) as well, so D and D2 are two independent passes.

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
70705 is sent in every request through `GPQA_SEED_LIST`.

**One session.** A fresh server, then the release warm-up gate inside the chain (ready file, then /v1/models), then the
verified warm-up checks in the analysis.

The client and request-level configuration was checked against the record's own passes [M]:
- `model_args` without the seed and `gen_kwargs` are identical across the record's three passes;
- prompt, doc, target and message hashes, and the server-rendered prompt token counts, are identical across those passes.

## 3. Request seed (specification 3): **70705**

D2 re-runs D's seed scan, extended so that it also covers everything D produced.
- **Inventory:** `checks/seed_scan_d091.py` is still rep070's `seed_scan.py` with one change, `SKIP_DIRS`. It now skips only
  this package (`d091b`), so D's package folder is scanned like any other. The diff is regenerated in
  `checks/seed_scan_d091.diff`.
- **What it scanned [M]:** 22 request logs, 51 lm-eval results files, 27 published-row files, and a text scan of 11,550 files
  under <data-runs>, the public repo and <eval-tools> (1,415 distinct seed-like values). These include:
  - D's proxy request log and sidecar (`+d091.s1.requests.jsonl`, `.events.jsonl`): 198 rows each, all seed 70704;
  - D's lm-eval results (`+d091__pass1/…/results_*.json`): model_args seed 70704, process seed 1235;
  - D's `run/`, `out/` and package files, and the public repo's copy of D's package.
- **Seeds already used by GPQA:** 1234-1236, 4242, 5001-5012, 70701-70704. Request seeds seen with any prompt: 1234-1236,
  4242, 70701-70704. **70705 is in neither** [M].
- **Raw bodies:** `checks/raw_seed_scan.py` (only its skip path changed, to `d091b`) opens all six gzip raw-body files under
  <data-runs> [M]:
  - D's `d091/PRIVATE/raw/s1.raw.jsonl.gz`: 198 records, seed 70704 only;
  - B+'s v070/v091/b1 files: 48 records each, B+'s 8,170,101-8,191,204;
  - rep070-F's s1/s2: 70701-70703.

  It finds neither 70705 nor 70706.
- **Text scan: 70705 has 10 hits, all accounted for** (`checks/seed-text-hits.txt`). Every hit is option D's own negative dry
  run N1, which relabelled existing rows to "seed 70705" in a scratch file and fed them to the analysis. Nothing was sent to
  a server; the scratch was shredded. The hits are in D's `tests/dryruns/N1-wrongseed.{json,md}` and
  `tests/run_analysis_dryruns.sh`, in <data-runs> and in the public repo's copy. **No request, results config, published row
  or raw body carries 70705.**
- **Disclosure:** D's eligibility rule was "in none of the four sources", and the text scan is one of them. Under that rule
  as written, `seed-scan.out` marks 70705 not eligible (`in_text_scan: true`, exit 3). D2 uses 70705 because the only hits
  are those synthetic labels: 70705 has never been used by any GPQA request on this machine. 70706 is clean in all four
  sources and in the raw bodies. It is the fallback if the coordinator prefers the rule as written.
- **D2's own negative N1** uses D's seed 70704, a used seed, so this package adds no new mention of a fresh seed.
- **Formula-based seed bases:** B+'s scan lists none below 7,810,000 [R] (`bplus/checks/seed-formula-scan.txt`). Option E's
  planned bases (8,270,101 and up, in `e-cross/checks/seed-inventory.json`) are far from 70705 [M].
- **Frozen inventory:** `checks/seed-inventory.json`, sha256 b961a937…. The analysis pins it (the one changed pin) and gates
  "70705 is in none of its seed sets", that is, in neither GPQA-used nor any-prompt request seeds.

## 4. Validity gates (specification 4)

Any failed gate means INVALID: the analysis exits 3, there is no reading, and the long-reply pipeline does not run. The
analysis is `analysis_d091.py`, frozen.

### 4.1 Request, task and pairing gates

These come from the frozen loader (`frozen/analysis_col511.py`, sha256 b5b13ce2…, hash-checked). It loads the one pass
exactly as rep070's variant-F loader loads each of its passes: `SEED_BASE = 70705 − 1`, keep pass 1, restore `SEED_BASE`
on every exit path.

- The proxy request log and the server log are both present.
- Exactly one samples file: 198 questions, ids 0-197, both filters.
- 198 requests, with no re-sent request (at most one per message).
- Every request returns HTTP 200 and carries seed 70705, max_tokens 327,680, temperature 1.0 and top_p 0.95. There are no
  parse errors.
- Every question has its proxy row, and each proxy content hash equals lm-eval's reply hash.
- **Seeds:**
  - the request log holds exactly {70705: 198};
  - lm-eval `model_args` seed is 70705 and the process seed is 1235 (equal to the record's pass-1 process seed);
  - 70705 is fresh against the frozen inventory.
- **Client:** `model_args` without the seed and `gen_kwargs` equal the record's pass 1.
- **Task hash:** 87e807b3… in the D2 pass, and also in the record's pass 1 and rep070-F's pass 1.
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

`checks/gate_calibration.py` applies the analysis's **own gate functions** to four sessions: the record's two
(2026-10-08 and 2026-10-09), B+'s stock 0.9.1 session (2026-10-21), and, **new in D2**, option D's own GPQA session
(2026-10-21, `d091/run/server-s1.log`, with its request window). The gate set and its calibration are D's, unchanged. The
script gains only the D input and a D verdict, and it now exits non-zero unless every gate also passes on D. Results [M],
`checks/gate-calibration.txt`:
- **Every gate passes on both record sessions, on B+'s session and on D's session** (12 of 12 server gates on each). None is
  stricter than the variation those runs showed.
- **Request-level gates:** every one listed in section 4.1 holds across the record's three passes.
- **KV pool, reported without an identity gate.** Two pieces of evidence decide this:
  1. *The record:* both sessions produced 4,707,515 tokens, as did all 11 default-layout (DCP1) 0.9.x K3.25 sessions in
     the startup markers [M], D's 2026-10-21 session included. A gate of "a size the record produced" would therefore be {4,707,515}, which is plain
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

  The analysis still reports whether the pool equals a size the record produced. Dry run `D2-kv-other-size` (a dry-run
  name from D, unrelated to this package's name) shows that a pool of 4,698,112 is VALID. Dry run N8 shows that a pool of 2,000,000 is INVALID.

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
| lm-eval output | `<gpqa-runs>/glm53-flash-exl3-k3.25@tpurtell+v0.9.1+d091b__pass1/` |
| proxy log, sidecar and lm-eval log | `…+d091b.s1.requests.jsonl`, `….s1.requests.events.jsonl`, `…+d091b.log` |
| chain log | `<data-runs>/benchmarks/<date>-…+d091b/` |
| raw bodies | `PRIVATE/raw/` |
| server log, inspect, ledger, GPU log, cache inventories, analysis | `run/` |
| pipeline outputs | `out/`, `checks/prefix-verify-*.json` |
| prefixes | `PRIVATE/prefixes.json`, `PRIVATE/prompts.json` |

All of these are inside `d091b/` or carry the `+d091b` tag. `tests/output-paths.txt` checks each one [M]: it differs from
D's path, nothing exists at it yet, and the runner's pre-flight glob for `+d091b` matches none of D's five record paths.
File names inside the package keep D's names (`run/rows-d091.jsonl`, `out/tables_d091.md` and so on); they live in
`d091b/` and cannot collide with D's.

**Note:** like every GPQA record, the lm-eval samples file holds the reply text, and it lives in `<gpqa-runs>`. That
is the standard protocol location and lies outside this package (see section 11).

### 5.5 Command (needs the operator's authorization; no sudo)

```bash
cd <runs>/long-reply/d091b
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

**D's figures are not in this report.** The analysis compares against the 0.9.1 record and rep070-F only, so putting D's
pass beside D2's would need a code change. None was made. D's `run/result-d091-final.md` can be read beside D2's.

**The stated figure stays provisional until the coordinator's hand read.** The analysis lists the ids where stated
differs from flexible-extract. After the read, re-run the analysis with
`--hedged 1:DOC …` and `--out run/result-d091-final.json`.

### 6.2 P(finish | still generating at 100,000 tokens)

- The definitions and functions are rep070's secondary, unchanged.
- It is reported for this package's pass (the group is still named `d091` in the outputs; seed 70705), the 0.9.1 record (passes 1-3, and pass 1 alone) and rep070-F, each with numerator,
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

**Comparison against D (not added).** `compare_d091.py` fixes side A to discovery's rep070-F outputs, and side B is this
package's `out/`. Putting D2 beside D would need a code change (side A as an argument). None was made. Both packages'
`out/tables_d091.md` are against the same rep070-F side under the same definitions, so they can be read in parallel.
Only the default label changed (it carries the seed).

**Proof that compare_d091.py changes nothing:** `compare_d091.py --reproduce` recomputes the rep070-F side from
discovery/out and matches discovery's `summary.json` exactly on T1, T3, T4, T12, T13 and drift.

**All of this is descriptive.** It is one pass, with about 8 long replies expected and different seeds. It cannot
adjudicate a termination mechanism, and reaching 100k selects different trajectories on each engine.

## 7. Prefix export (specification 7). Nothing is continued in this package.

- **Rule:** B+'s rule (`scripts/prefix_select_d091.py`).
  - Pool: T ≥ 100k.
  - Eligibility: N_reasoning > 100,000, and R50 at 100k defined and < 0.05.
  - Strata: capped first, then finished; one reply per question.
  - Ranking: questions by discovery's panel count, with salted-hash ties (salt `d091-2026-10-21:`). The salt is D's,
    unchanged. It is neither a path, a seed nor a tag, and changing it would change the rule's tie-breaks. Reply ids differ
    from D's, though: `q{doc}p5` here, `q{doc}p4` in D, from seed − 70700.
  - At most 6 per stratum.
- **Selection code:** the function is `select_prefixes.main()`'s body with its inputs as parameters. B+'s file asserts
  B+'s pool size of 29, so it cannot run on d091 unchanged. Run on B+'s inputs, the function reproduces
  `bplus/out/selection.json` **byte for byte**, and so does the byte-identical copy of `select_prefixes.py`.
- **Construction:** B+'s `build_prefixes.py`, **byte-identical** (sha256 1ebbec2e…). It is called through
  `scripts/prefix_build_d091.py`, which sets only the module constant `REQLOGS` to this package's request log (`+d091b`,
  seed 70705).
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

These are in `CHECKS.md`, which marks each check as re-run for D2 or carried over from D (the inputs were unchanged). In
summary:
- analysis dry runs: 2 valid stand-ins and 15 negative cases, re-run on **D's real 2026-10-21 pass relabelled to 70705**.
  The stand-in's 198 result rows equal D's own `run/rows-d091.jsonl` in every field except the seed;
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
| `checks/seed_scan.py` (the `_d091` copy and its diff changed in D2: SKIP_DIRS only) | rep070/refreeze/checks | 3d8e80e4… |
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

D2 adds:
- **Seed eligibility:** the text-scan hits of 70705 are disclosed in section 3; 70706 is the fallback.
- **Carried over from D, not re-run:** the discovery reproduction, the index, selection and construction reproductions, the
  stand-in pipeline, the proxy-sidecar test, the costing and the timing reconstruction. Their code is byte-identical to
  D's, or differs only in seed and tag constants and docstrings (section 13).
- **Labels inside the code** still read "option D" / "d091" (headings, check names, JSON keys, file names). Only seeds,
  tags and paths changed. The outputs are identified by seed 70705 and tag `+d091b`.

From D:

- **Any GPU behaviour:** start-up, decode rates, whether the warm-up and the KV pool repeat. The checks were
  stand-ins, logs and shims only.
- **The positive pipeline path has not run on a 0.9.1 raw file,** because none exists yet. It ran on rep070-F's pass-1
  raw file. The 0.9.1 template render for any newly selected question is checked only at run time, by the verify step.
- **The session simulation uses a fake chain and a docker shim.** The real chain's interaction with the proxy copy was
  last exercised in rep070-F (2026-10-16) and B+ (without the chain).
- **Stated accuracy** needs the coordinator's hand read.
- **The cost model is linear in caps.** With 8 lanes, overlapping caps make it conservative [I].

## 13. Changes from d091

Before = `d091/SHA256SUMS` (sha256 b558b584…, D's frozen list); after = this package. **72 files differ or are new;**
the other 78 files are byte-identical to D's. No D file was removed. The code changes are in full in
`checks/changes-from-d091.diff`. Every one is a seed, tag or folder-path substitution, apart from four:
- the seed-inventory pin in the analysis;
- the D session added to the gate calibration;
- the dry-run stand-in switched to D's pass;
- the folder guards in the tests that were not re-run.

| file | before (d091) | after (d091b) | change |
|---|---|---|---|
| `CHECKS.md` | `4cc77195f54dd2ca7a53067687ea1017115e6b0530c0180c8e2a9c446eb150a0` | in `SHA256SUMS` (a file cannot hold its own hash) | rewritten for D2 |
| `PROPOSAL.md` | `be291ddfe8115599a934b611606760cae6e5d9a2c7df4c1da098ecb92392470d` | in `SHA256SUMS` (a file cannot hold its own hash) | labels, seed, paths, D2 sections |
| `analysis_d091.py` | `5317be99b6f65e99c61f78a3984a910ba7f27932226052df4bc8a14b368907e5` | `57b76ce85ffe963e47e4b7c3988778da3ae291176b81c9eab61c92ea0de825ca` | seed 70704 → 70705 (constant, docstring, check and label strings); TAG `+d091` → `+d091b`; seed-inventory pin → the D2 inventory |
| `checks/changes-from-d091.diff` | — (new) | `cc3d2c232eca098667dea3f5e81fcd7544e93e5a235f9e406f066510f132514d` | new: unified diff of every changed code file against d091 |
| `checks/gate-calibration.json` | `99430b1a3471391c96140bbbcf1b3226bba16f5ebdf0a75c78d01eb4afc8234a` | `4e52a1f8b215f4ad6b05c2df3c7c0d59aa7580dafc78824e9b9a58f210cab668` | re-run with D's session |
| `checks/gate-calibration.txt` | `a51b95897c0d4ec9681845e8f442aa748f182e9b148051a800571972d267c2cd` | `a6084df1030ae9d52328280f11e9624caeb94d3200acd4b7ed3cbdbf14e90553` | re-run with D's session |
| `checks/gate_calibration.py` | `1b1255e4e811ac30383ff4f9fdf3ca0be91f542c85b29dab9c7fc87d30cfd34e` | `8cb80eff8b4d209dbd27039fc86bfeddb00c2774121153146218820f5ac1df98` | adds D's 2026-10-21 session as a calibration input and its verdict; exit 0 needs D's session to pass too |
| `checks/privacy-scan.txt` | `a02a5c1089f17637b05e66f2f177585d459367005504b47125ad07d9f9f3a0cf` | `fdbcbe5dde004d0fc039bb05b784ec363067b7c0c598922c1b1aabd38b867d8b` | re-run (D's and rep070-F's raw replies as needles) |
| `checks/raw-seed-scan.json` | `d1072d43c2224cda147ab60ea762021ad34021ade1e06fba12099321bc7c03ea` | `813e54dd07aa9606e2924547654df3ac1f110d1675a123fbb122f4097f43476d` | re-scan (D2) |
| `checks/raw-seed-scan.out` | `64f1f5e38de92f6af4dfd1e9cded12ebddbc614e8e5347c60cc9f2d8c7fc1e19` | `fd1335a7ece57a4e44da227914745aafa520f1721e39927a72780de2e6deabaa` | re-scan (D2) |
| `checks/raw_seed_scan.py` | `a67d3e0abb90decf4aef3d22a483eb2f39997c88783551abb8c9193e94a38b4c` | `eea982a161cc50925593fc52602a98509066a0b5ac4c57379ae4f76196d92ef0` | SKIP d091 → d091b (D's raw bodies now scanned) |
| `checks/seed-inventory.json` | `1f58e7b3747b769133e112bfbe64c45890a1812742e1526d682bcefcda931c8e` | `b961a93781acbc728d53095c75d995bbeeef49b161c8ec96380c1856f89e6cb6` | re-scan (D2) |
| `checks/seed-scan.out` | `a1492f5d093af3202438f072270b95ae4328bd763bc433960d4bf122fbc40b75` | `6a8d8722c3fa417f7c8d9232825afb4dd63e39058ced711d0be3036a4cf32bab` | re-scan (D2) |
| `checks/seed-text-hits.txt` | — (new) | `631f22e1f7b8aef135918ce8ccf6c6e2ee45497a17b2a32d08a9b49430ca8099` | new: attribution of the text-scan hits of 70705/70706 |
| `checks/seed_scan_d091.diff` | `18a7b7e5e955b071dcd07973f3a6e2389ef39b14c4757642e46325e67e811010` | `cc28c2952689b6c7a88c68b6f42dac7804528f08d459feca95c25b034234e4a2` | regenerated against rep070's seed_scan.py |
| `checks/seed_scan_d091.py` | `e8716c6ff79e369d444a76520d6e1dc2992dbed9323b28b63a514ea0c9b84cb2` | `7dc4da515328173ebe2def06dea6ec18e70e3c989c74567c391a9dcb9bf211e1` | SKIP_DIRS d091 → d091b (D's folder now scanned); copy note |
| `incomplete_d091.py` | `192c852d045a9648404530c104fa9c28dbc90648bc6020155c4e68375ee4d2a4` | `06563285abc48ae2025cfe0f8ac1e391aeb900fa01e5b959059c12142c2f1f30` | seed 70704 → 70705 (constant, report heading) |
| `make_sums.sh` | `8af9399bc7078870e3a8981a28337f933282a806bb00bba34b825a668b88418f` | `ee9f0f967b3f94ab7e6d712e355fd91438fe74e60212797abf16066a835bb1e5` | folder guard d091 → d091b |
| `runner/run_d091.sh` | `4b94b6c4d3e246dcbc7692daf839f1751b4ece3553d1806f93af934e8f839b0e` | `bbb7f587545edc46ca47e7c71047c65f369d8b0b604b5b60e4db5dbe020c1a82` | folder guard d091 → d091b; SEED 70705; TAG `+d091b` (header comments too) |
| `scripts/compare_d091.py` | `9763a464f24fe4e93bcc457907131c9a95a0762d635a47685aa71a25966ca9ff` | `f86aa12b3a006ca1aea4ceb77e051ac9fcb413884b60a687e8f0e5b7bbed9a76` | docstring and default `--label` seed 70704 → 70705 (no code change) |
| `scripts/index_d091.py` | `0c0ecc4b1bd96f48fac2e992ecd8c42cacf315139e8c45516fb3002241866db9` | `f7fe081c1e7626d366fc983764d0899f95c12b2daab9e74cd3c1914e0489f5c5` | docstring only: seed 70705, ids `q{doc}p5` |
| `scripts/prefix_build_d091.py` | `7abf9bcc17fe26f9bb6dabd305ca38e578275deada23784b20d937b1c6f3931c` | `1b4adb50ec915ef8405a7c8082ad522620667596993b9fe4e9b76ae51ed0e8a3` | default SEED 70705 and REQLOG `+d091b` (docstring too) |
| `scripts/run_pipeline.sh` | `a467385e732ab84705f03e87579cf045711a2780b76138414d3f2245e1ad10fa` | `70b79e41ee5158c2e0f06a259720d878142ef8b054716fbcc0aeda196538c6bc` | folder guard d091 → d091b; default request log `+d091b`, SEED 70705 |
| `tests/analysis-dryruns.txt` | `435fc2b867025ecd7996f2d12c2b8fa3f58ba59fc7a7763f66e36fd378dcbbae` | `0ac9cf3cb85e476fd4360496bd6f0803288f98705ba579d946dda1ea709c5c8c` | re-run |
| `tests/chain-seed.txt` | `600b337af41aeb70ccd91a46ebbc94689b4b060da9919d694894cc5d4c77fe4c` | `19d84670013fed741fb1485b0dc1418e282dda17b0c6dd823dc0ddb849182001` | re-run |
| `tests/dry-run-real.txt` | `83b0a8f9901732afc629aa43b5d4339e05fd05ad6bcfc705ca8860664e0d58b3` | `b25b15b31bcc6a7c8c6bd1687242d8cd79dd7f439567cc61b20a14299301b445` | re-run (real machine, DRY_RUN=1, 10.0) |
| `tests/dryruns/D1-standin.json` | `f6fe84e23e5d57c089581132b8da73022b025bd35f144e9a9340844adde28307` | `84285852a33f8ffe727187617b77730eaad5ed8c23e72e6db176b89e93821484` | re-run (D's pass relabelled) |
| `tests/dryruns/D1-standin.md` | `380f7a470fdbee60937290077a58ec5e62a7ce434eff6ec8f98689ef6802dd58` | `bdc06241f6b329a83bff91cb25ea9550d598be60ac013a73d4dc39808e837284` | re-run (D's pass relabelled) |
| `tests/dryruns/D1-standin.rows.jsonl` | `49cc53725c0183b79a820286bb10829ce8560fe3a4d9e7ba969fbc1b39df22f6` | `804d82c25cca28e0dd95b2398d8f788231d11b9c03df32b2bcf685091549e2f5` | re-run (D's pass relabelled) |
| `tests/dryruns/D2-kv-other-size.json` | `94d19b07016c8cadc624d815ec8a365dcbc1980178b6a8146caee6ef7b199d90` | `2c1549db9224752b226d80358947010be6f501def7e98a911cdc73f037a30da2` | re-run (D's pass relabelled) |
| `tests/dryruns/D2-kv-other-size.md` | `ba63f4f0231f95cd0b0a5d2b66105d6b392f303c22346d7719f3a32ccf9539b2` | `f7d9d702462714dfc964e0a1d2617db46fbbd933272488d98512f792ce207ddc` | re-run (D's pass relabelled) |
| `tests/dryruns/N1-wrongseed.json` | `415e882e26df38bc78ba7e5360b0026464e2a629356858c51ded73580a3bb6fa` | `fe06c508abb4996209fbf8d6025c0e384ed79dbf395d2ce2b9d73847fc9810f1` | re-run (D's pass relabelled) |
| `tests/dryruns/N1-wrongseed.md` | `186e38dbebc4cc45e996d6aae3c1bbbdf5fe62c63d215996f5332b399e83db1f` | `7a5b291dd9b2b9d538cd05f54110dc45e342848f9ec82b3cc0cdb7d3ceec50ef` | re-run (D's pass relabelled) |
| `tests/dryruns/N10-http500.json` | `7cdb79dcd0135fff0cd2f41f15305772ed9e9fcc372fe3874762754df75621b9` | `b6663838164a47ece11109104f73b8b859ae5655f4f01e6bd3ce5ed5721e5b39` | re-run (D's pass relabelled) |
| `tests/dryruns/N10-http500.md` | `af4831a33aa48e44d6105d5f9c3653af7cfae64c77b7d7b37370f89091621212` | `7237219d0d6a1dbef3710175c743eecc3657e69d3d1d131384de24b1b19fb1fa` | re-run (D's pass relabelled) |
| `tests/dryruns/N11-nondefault.json` | `de718c141d0787d5dff2f77a95ffa5f8b748ae2044c7ade05a5202ea321775de` | `7cba1599580fef5671a91aca728a322e1eac24a66ae677c21eebffa7f1a171a7` | re-run (D's pass relabelled) |
| `tests/dryruns/N11-nondefault.md` | `f3252f721d9943326cd20bd2f7e895c615320f518b025ba032df198478401492` | `e12c8c666acdde75be3fc97cc9974dc1e16ce403bf0477401fcbf1e00401d5bf` | re-run (D's pass relabelled) |
| `tests/dryruns/N12-noargseed.json` | `016a05c993ef5156561d99a913e87622ee631dddbd2e9597060252c47b7b1729` | `21d73e5d8b595e00d862c4dcb45c5ceb886c60ae667d6f00911177b6ef57d787` | re-run (D's pass relabelled) |
| `tests/dryruns/N12-noargseed.md` | `1fb9b9ba277eb2ee38f42b0ce0a9456600898da32f2f6a88659e0050152e8644` | `aebdb1940a339f6aaf4737d7c40fb6e75a45b1fde1cfae89830df4ac639773a2` | re-run (D's pass relabelled) |
| `tests/dryruns/N13-rep070F-serverlog.json` | `992a7f5b7529f1549ab28882ab1984a028dcafdd00fbc1925f8a91a9d58726c6` | `4e8b045526f9ecc1e830c9952cbb56d5c36e6b7c64eef7486f192f15ea0372c7` | re-run (D's pass relabelled) |
| `tests/dryruns/N13-rep070F-serverlog.md` | `91f5fa4a4c0ad597ce8580fdb83c404f6a24eb8f4a3a5367d8a5937fd3b53f93` | `13068d510bf999dd2bd18105f8d7de0f8329d9de898ebb911e9bfc3dc47a04a9` | re-run (D's pass relabelled) |
| `tests/dryruns/N14-no-reqlog.json` | `79dc450ed25a8aab2efca9ddfc6dcaa80232caa1f19270d569d764880730076e` | `bf5dbebda9b950aa9f58ce435bb796ff81fc1f868955e4627551f0c9f4750e41` | re-run (D's pass relabelled) |
| `tests/dryruns/N14-no-reqlog.md` | `fb46ca6352f571a21368e206c118f72ca8aea731d050d6e5c0178c5c0b6c360f` | `9534b2f08db8c344998d25638f0259e2efc5c846ab79d77c77e1c6418720ca3b` | re-run (D's pass relabelled) |
| `tests/dryruns/N15-no-serverlog.json` | `d1195596c93450275ffe6671e432376e4e337141956f817f1a14ebf6befe3632` | `c6ac0ca15687ac72ea400c691dd04b4a1ccd2cf3906fe1af8dc65c31673fc6d7` | re-run (D's pass relabelled) |
| `tests/dryruns/N15-no-serverlog.md` | `532b0ab32d7c0fde07ccf8bd21db846d06eee0557e373a7cf10bb3add131c206` | `101c2ca35f10ec7b3633755ae9f67d11dc70944b1e75079c9a06bf7bf02578ce` | re-run (D's pass relabelled) |
| `tests/dryruns/N2-missingq.json` | `02afe31a1a79ef42814631b10c4f91d1316f9d8aabc4509495b41e6f7ed92531` | `5f2f720fc1ce9aa9374f55cfbe7797f9ea46b628ba3143a354146baf7eda5b80` | re-run (D's pass relabelled) |
| `tests/dryruns/N2-missingq.md` | `550219d4d3d8c7d1b8866729bbb903dba3707c5a50a996e0261b2faad8c0c91f` | `9fd4d46f7ac034d49a4b227c4d937b0cae19b4000d2d45dde00658d0cd2e87eb` | re-run (D's pass relabelled) |
| `tests/dryruns/N3-taskhash.json` | `a493154a09836fe3791fc14a69348bc5d063c1a9c67359df8ac90df0a743c8cd` | `5c2d40e4862d4849f89d0167b82e95bef3f9b9ec04b79c8ae41bb78081bec9fa` | re-run (D's pass relabelled) |
| `tests/dryruns/N3-taskhash.md` | `bef39716e5e35074f351aab002115d80ec1bb374160edcc142630adb90898631` | `1de9a7ef832fddac41fcd3c0b9a7afd943a02cf3620102b10dd9fc04d0ae6a9e` | re-run (D's pass relabelled) |
| `tests/dryruns/N4-noready.json` | `1539fc52375c3e76f8385da0d3146e60c02f72e06aefaf883541ae00671bdb93` | `e6b22df344a7303023bec7b521dd7ec77e5b34bc4f3a490cd77c91350b2229af` | re-run (D's pass relabelled) |
| `tests/dryruns/N4-noready.md` | `8d0487f38227f0986f95ec596dfe81308e47c69ddea8e37850ef4451e882c96b` | `22378c14ffc254c4d618b00e337f152ae6f86551ef4a656aa08fc4350633138c` | re-run (D's pass relabelled) |
| `tests/dryruns/N5-other-engine.json` | `679723d826af356220351aaf4159b9bc6a3214921a1408145d56d2dfe2d5e50f` | `cf55975513a82ab59d5852f57841269b1da145f20ff4d7a0107c5309ccebe836` | re-run (D's pass relabelled) |
| `tests/dryruns/N5-other-engine.md` | `310d179684f6b3ef8e9e1c6e2ee3e5b36a9e34e74ef1649a8f625a7a65214bb3` | `5c26a1558ee8de95c2a85f8de55320d6cbdc0688194389215df20797635fd9bc` | re-run (D's pass relabelled) |
| `tests/dryruns/N6-resend.json` | `5c5dd8a47a231568d16985d7c52cc39ea39c6f52d29a61c51386ffc8091da3e4` | `c54fb0dbc42c382008a6e387f97599257b09ef68deec7eced886881805158524` | re-run (D's pass relabelled) |
| `tests/dryruns/N6-resend.md` | `63469336cb0831bc193443649082bb500d9cd403f259bc066ec0c267eb424cf3` | `59f991d85c943b420e2ba9c26a8f8aff001ce1d396d184dcbbdfb75ee54fcf72` | re-run (D's pass relabelled) |
| `tests/dryruns/N7-target.json` | `2cabf1b54955014bf1474414416da741f709c40f2b35657b94670f9db2fafa84` | `bdfb376a4c3f21ecfc52ce5129cb502bbe0efaa6b0bf021ec7e01aa5170a8803` | re-run (D's pass relabelled) |
| `tests/dryruns/N7-target.md` | `9f7229b89a1591cb400baaa9b54ed77ec7cfcca7b4d37f0daa57169a8043c741` | `a224831db5c9f8063a3cb00ff87cbfbafc1a5a2d75dea1a0d06c50b1cc3b25aa` | re-run (D's pass relabelled) |
| `tests/dryruns/N8-kv-below-floor.json` | `023b16057f0900304a53c646021edb741a50549b8ba23959068d70f6adecd33d` | `f7e8aa5f2c07c4956e59d9f8675bb06bb273292b0dfdc7f10b166452ea819ee7` | re-run (D's pass relabelled) |
| `tests/dryruns/N8-kv-below-floor.md` | `f8ec25e03611991fc964464a1453f2caf4197e2660b230b24899627ebdd644ec` | `35ff9d3032c24731c58642ccdfa98c5ff5c0948a3fb1fd098495e3bb62fed009` | re-run (D's pass relabelled) |
| `tests/dryruns/N9-procseed.json` | `191a3f45a9cee1be357724fe1e322fe764ffba2f21d56817639ccac7166a406c` | `6ddbb0b7abf4b3d3d3aab7e1de3d53527e20183ca5eb855e7fa9cbb9476c0e4d` | re-run (D's pass relabelled) |
| `tests/dryruns/N9-procseed.md` | `adc90f5b9347f4da69a5ac13219207d8c0335657770f7a6e522e3902e648f8b1` | `dd0f1a78fb17ebf82aef93467ad3d9db6ed5d07fa48ec381d5bc5423f835e879` | re-run (D's pass relabelled) |
| `tests/dryruns/run_dryruns.out` | `84b2472e3d6cfe786e3bbc40e5bbe5e0b72aef7a42f729edaf19955bf54d337a` | `d91b5f6c2743dee7b5e7b24201ebb9f57da50d6649411683e3db3a8207cab7e5` | re-run (D's pass relabelled) |
| `tests/output-paths.txt` | — (new) | `1810ccf2eeebeeb3e6d21cf247a32f88608f5bc5584fdf7e43b63372cae64051` | new: D2 output paths differ from D's and do not exist |
| `tests/prefix_build_repro.sh` | `70391c72689895b6c859afaebcf18eddd1c7dc5a9b83c7b1fd9ae57eb7b972c0` | `f8e30af1fad778cc1180852dfa4edc93c6a0f161934090514d409cfa7d1c104e` | folder guards d091 → d091b (not re-run) |
| `tests/repro_discovery.sh` | `29db2bb7a27d511c0e3eac965014134ddb470615b43325daffa32a9a1e3df91f` | `c3cc1136ff0ed9aac0804641bf31fc76e424fa0cc5b5c59896c47e0665f481cc` | folder guard d091 → d091b (not re-run) |
| `tests/run_analysis_dryruns.sh` | `7731d102a7a1ecf5443ca90e4dd2415ae6a579ce283eba2f8952272712fd3ac7` | `c99c3846230722cfea2b7fcc6def90fc149f41f19f3a65765765bb36c996134f` | folder guards d091 → d091b; stand-in = D's real pass relabelled to 70705 with D's session log; N1 sends 70704 |
| `tests/runner-refusals.txt` | `f6db0bc312636ecc31b0ae8ebc3f9fe6fe15a552344533033e8853959b8ec534` | `58ce7edd4d8ccc07033b5a7abd6f1dbaa35c4d427185bf6d8c339ca21d0134a5` | re-run |
| `tests/session-sim.txt` | `f195b746946894647b36ac27d2d260fbb706d85594890d25aed99f7ac51bde07` | `49eceab97b5cc2d5f51c8a608d638c5154d54ea7b166728e179ea748a408081c` | re-run |
| `tests/session_sim.sh` | `955ab8f73a0085ba57fd45aa55d8c69948f10a3840e28e7faec360a7cdc68a38` | `13cce7998c405080316758870d6c3281e7478503f4b065cf7654f951fe3f646d` | folder guards d091 → d091b; tag `+d091b`; seed 70705 |
| `tests/standin_pipeline.sh` | `93f5cb5425629d686e2055d3473ab7b465eaf3aed08eeb5fd48eb27e2d854719` | `92efa65205c1498bfe8319a3e7b3f859708a3a0623d424f436968ba92ab6ae73` | folder guards d091 → d091b (not re-run) |
| `tests/test_chain_seed.sh` | `14f85ae3b4c5947b866185b1b15255d11d9af5edffbca9702ac7247b911aa69b` | `d3bfbf14205eafd8f3310c59199f8f41856e366365d11baaf8b5788d0584ca0d` | seed 70705 |
| `tests/test_runner_refusals.sh` | `d7375894fb1de23cdb81c4c26913f9f282f2ad5514214a3a78b9cb1bedf3363a` | `fa7bcc99721a4fea2069aaf2931282090fdb60d0c1de22b80a720fda9c62e723` | folder guards d091 → d091b; leftover-file glob `+d091b` |

**Not copied from D:** `run/`, `out/`, `run-d091.out`, `PRIVATE/` contents, and D's run-time `checks/prefix-verify-*.json`
(pipeline outputs, not in D's `SHA256SUMS`). `out/` and `PRIVATE/` (0700) exist empty.
