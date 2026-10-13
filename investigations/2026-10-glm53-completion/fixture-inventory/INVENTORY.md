# Fixture inventory: existing histories for the long-context termination probe

PLAN.md amendment 20, section 2, "Existing histories before new ones", with the amendment 20b loop/exhaustion
classification. CPU only, 2026-10-13. Local only.

This file holds no reasoning or GPQA text. It records ids, positions, lengths, metrics and hashes only.

- Machine-readable list: `inventory.json`. It has `exact_traces` (24), `text_rows` (901), `phase2_requests` (8) and a
  `shortlist` by window class.
- Raw scan: `scan-raw.json`. Tokenizer facts: `termination_tokens.json`.
- Scripts: `scripts/`.

## How this was measured

- **Tokenizer:** the K3.25 tokenizer, in the 0.9.1 release image (`ghcr.io/tpurtell/...@sha256:a9e985c0`), run as a
  CPU container with `--network none`, no `--gpus` and read-only mounts. Nothing touched a GPU.
  - `tokenizer.json`, `tokenizer_config.json` and `generation_config.json` are byte-identical in snapshots 0490d2f7
    and 701cd745. One tokenizer therefore serves 0.7.0, 0.9.1, B0 and B1.
- **Prompts:** rendered as in `fixtureB/prep_requests.py`: the zai template (= 0490d2f7 `chat_template.jinja`),
  `add_generation_prompt`, `enable_thinking`, no added special tokens. The prompt ends in `<think>` (154841).
  - Every text row that logged `prompt_tokens` (585) matches the rendered length. None mismatch.
  - All 24 exact traces have prompt ids equal to the rendered GPQA prompt of their doc.
  - 0.7.0's checkpoint template renders GPQA byte-identically (`local-fixture-index.json`, render_check 198/198).
- **Positions:** a token index within the reasoning. Index 0 is the first token after `<think>`. Absolute position
  = rendered prompt length + index.
- **Text rows:** the client-saved reasoning text, re-tokenized with `add_special_tokens=False`, exactly as the
  phase-2 requests were built.
  - **Re-tokenization drift:** for 606 rows, logged `completion_tokens` was compared with the re-tokenized
    reasoning + 1 (`</think>`) + content + 1 (terminator), or with the reasoning alone for unclosed rows. The count
    matched exactly in 262 rows. 584 of 606 are within ±7 tokens; the full range is −84 to +93.
  - So a re-tokenized text is a **shared counterfactual fixture, not a replay** of the generated ids. Equal
    counts (all pilot rows) are consistent with identical ids but do not prove it.
- **Repetition indicators (amendment 20b),** evaluated at token positions over the preceding 30,000 reasoning
  characters:
  - `zlib30k`: the clients' `zr()` (compressed / raw bytes of the last 30k characters).
    - The clients class a reply `loop` when its final value is < 0.15.
    - The early-stop detector fires after three consecutive periodic checks < 0.10.
  - `rep200`: the share of 200-character windows whose exact text already occurred earlier in the reply.
    - This is a reconstruction of the pilot's "repeated 200-character windows". The pilot's own script was not
      retained.
    - It reproduces the pilot's statement ("0-1%"). The two capped pilot replies score 0.0013 (q109) and 0.0000
      (q170) over their last 30k characters, and 0.0061 and 0.0002 over the whole reply.
  - **Repeating at a position:** `zlib30k < 0.15` or `rep200 >= 0.5`.
  - **Loop onset:** the earliest point on a 4k-token grid (starting at 8k) after which every grid point is repeating.
- **State per history:**
  - `natural_close`: finish `stop` with content, so `</think>` was emitted.
  - `loop`: the client early stop fired, or the reply is repeating at its end.
  - `exhaustion`: capped or cut, and not repeating at the end.
  - `exhaustion_cut120k`: the client cut it at 120k.
  - `error`: no stop-without-close row was found.
  - The client's own labels agree in 803 of the 807 rows that carry one. The 4 differences are 3 `exhaust` → `loop`
    (end window repeating by rep200) and 1 `unfinished_at_cut` → `loop`.
- **Commitment events:** regex classes over the reasoning text, recorded as token positions only. The chosen letter
  is never recorded.
  - Strong classes: `answer_letter` ("answer is/:" followed by an option letter), `go_with` ("go with / I'll choose"
    + letter), `final_answer`, `boxed`.
  - Tentative class: `so_answer` ("so/thus/therefore the answer").
  - Each strong event carries the repetition state at that position.
  - Regexes are in `inventory.json` → `params`.

## A. What exists

### A1. Exact generated token ids (only source)

`traces/step5/{v070fix-r1,r2}` and `traces/step6/{v091dcp2-r1,r2}`, 24 files (`prompt<doc>.json.gz`, docs 0-5).
Each holds prompt ids, 16,000 generated ids, and decode logprobs (sampled token plus top-20) at every position.

| Field | Value |
|---|---|
| Engines | step 5: 0.7.0 with the two kpool fixes (local, EP2/DCP2, DFlash2 x5); step 6: 0.9.1 EP2/DCP2 diagnostic bundle (x3). Neither is stock 0.7.0 nor B1. |
| Sampler | T 1.0, **top_p 1.0** (not the GPQA 0.95), `ignore_eos`, seed 1234 + doc, concurrency 1, prefix caching off |
| Natural reasoning close (`</think>` 154842) | at 176-4,795 generated tokens; all 24 traces close |
| Token before `</think>` | `.` (13) ×13, `).` (568) ×8, `)` (8) ×2, `$.` (12940) ×1. Never whitespace. |
| Close to first terminator | 303-632 content tokens; terminator `<|user|>` (154827) in 24/24 |
| Decode logprob of `</think>` where sampled | −0.69 to −0.00002 (median about −0.0002) |
| Positions in the last 2,048 before the close where `</think>` was in the decode top-20 | 0-21 per trace |
| Active segment | `gen_ids[:first_close]`. Everything after the first terminator is **forced continuation** (`ignore_eos`; 2-93 terminators per trace): **excluded**. |
| Decode → re-encode round trip of the active segment | exact in 22 of 24 (the other 2 differ by 0 and 1 tokens) |

These are the only exact-id active-reasoning histories on the machine. All are short (< 5k), so they serve as the
short control class only.

- Decode logprobs exist for the natural close, which gives a decode-path reference at short context.
- They were sampled at top_p 1.0.

### A2. Token-id fixtures derived from text (phase 2): `phase2-stage2/requests.json` r0-r7

Rendered GPQA prompt plus the first L tokens of a **re-tokenized pilot reasoning text** (`fixtureB/prep_requests.py`).

- Recomputed here: all 8 id lists are reproduced exactly from the pilot text (`ids_equal_recomputed` true; sha prefixes match).
- Identical copies are used by gate, tailfix, tailguard, verify13, bundle gate-runs, b0-mapguard and
  d7/step2/run-b1-a1. The last is a B1 server run, so tooling to load these prefixes into B1 already exists.
- Source engine for all: 0.9.1 (K3.25@0490d2f7). Arm A = default layout, arm B = EP2/DCP2 bundle.

| id | doc | source trace (state) | L (reasoning tokens) | total ids | position vs natural close | repeating at L? |
|---|---|---|---|---|---|---|
| r0 | 170 | A/doc170_repfresh, seed 7170: **exhaustion**, capped at 131,072 (re-tok 131,069) | 120,000 | 120,217 | no close; active | no (rep200 0.000 at 64k-128k checkpoints, zlib30k 0.36-0.37) |
| r1 | 170 | same | 64,000 | 64,217 | no close | no |
| r2 | 109 | A/doc109_repfresh, seed 7109: **exhaustion**, capped at 65,536 | 64,000 | 64,478 | no close; 1.5k before the cap | no (rep200 0.0013, zlib30k 0.306) |
| r3 | 109 | same | 32,000 | 32,478 | no close | no |
| r4 | 109 | B/doc109_reporig, seed 1234: natural close at 53,538 | 16,000 | 16,478 | 37.5k before the close | no |
| r5-r7 | 55 | B/doc55_reporig, seed 1235: natural close at 28,117 | 16,000 / 8,000 / 4,000 | 16,240 / 8,240 / 4,240 | 12k-24k before the close | no |

r0 and r1 are active reasoning in the 64k-128k band, not forced continuation. They are censored at the pilot's
131,072 cap and do not approach a close.

### A3. Text histories (901 rows; reasoning text + JSON per request)

`*.reasoning.txt` under `<data-runs>/diagnostics/*`, `<data-runs>/eval-matrix-2026-10-04/screens/*` and the completion
`pilot/` and `replication/`. Every text file is complete: its character count equals the logged `reasoning_chars` in
901/901.

- **Literal markers in the reasoning text** (`</think>`, `<think>`, `</s>`, special-token strings): none, in 901/901.
- **Re-tokenized special ids** (154841/154842/terminators): none.
- So no history contains a premature close or a stop-string hit.

Per arm. Engine, weights, layout, seeds and truncation come from each screen's PREREGISTRATION, gate and server logs.

| Arm (dir) | Engine | Weights | n | natural close | loop | exhaustion | stop w/o close | closed then capped | error | reached 64k | reached 128k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| diagnostics/component-screen-2026-10-09/B | v0.9.1 stock | K3.25@0490d2f7 | 12 | 9 | 3 | 0 | 0 | 0 | 0 | 8 | 4 |
| diagnostics/component-screen-2026-10-09/B79 | v0.9.1 stock | K3.25@0490d2f7 | 12 | 3 | 1 | 8 | 0 | 0 | 0 | 12 | 10 |
| diagnostics/component-screen-2026-10-09/EN | v0.9.1 EP2 NOPE0 | K3.25@0490d2f7 | 12 | 10 | 2 | 0 | 0 | 0 | 0 | 7 | 3 |
| diagnostics/component-screen-2026-10-09/EO | v0.9.1 EP2 owners-tp sharing0 | K3.25@0490d2f7 | 12 | 11 | 1 | 0 | 0 | 0 | 0 | 10 | 3 |
| diagnostics/component-screen-2026-10-09/NO | v0.9.1 owners-tp NOPE0 sharing0 | K3.25@0490d2f7 | 12 | 10 | 2 | 0 | 0 | 0 | 0 | 9 | 1 |
| diagnostics/component-screen-2026-10-09/S1234 | v0.9.1 stock | K3.25@0490d2f7 | 12 | 1 | 11 | 0 | 0 | 0 | 0 | 12 | 11 |
| diagnostics/component-screen-2026-10-09/V79 | v0.7.0 stock image, DFlash x3 | K3.25@701cd745 | 12 | 3 | 4 | 5 | 0 | 0 | 0 | 11 | 11 |
| diagnostics/empties-k325-v0.7.0-2026-09-30 | v0.7.0 stock | K3.25@701cd745 | 18 | 16 | 1 | 1 | 0 | 0 | 0 | 10 | 6 |
| diagnostics/empties-k325-v0.8.0-2026-09-30 | v0.8.0 stock | K3.25@0490d2f7 | 18 | 13 | 3 | 2 | 0 | 0 | 0 | 11 | 9 |
| diagnostics/empties-k4-v0.8.0-2026-09-29 | v0.8.0 (inferred; no server log) | K4 (inferred) | 18 | 16 | 1 | 1 | 0 | 0 | 0 | 8 | 5 |
| diagnostics/gpqa-retry-2026-10-01/retry-A-sharing0 | v0.8.0 stock, sharing 0 | K3.25@0490d2f7 | 20 | 12 | 8 | 0 | 0 | 0 | 0 | 11 | 8 |
| diagnostics/gpqa-retry-2026-10-01/retry-B-k5 | v0.8.0 stock | K3.25@0490d2f7 | 20 | 14 | 4 | 2 | 0 | 0 | 0 | 9 | 7 |
| diagnostics/gpqa-retry-2026-10-01/retry-C-default | v0.8.0 stock | K3.25@0490d2f7 | 20 | 14 | 6 | 0 | 0 | 0 | 0 | 10 | 8 |
| diagnostics/k4-k5-share-screen-2026-10-01/replay-C-k4-default | v0.8.0 stock | K4@a5fee929 | 32 | 14 | 0 | 18 (cut 120k) | 0 | 0 | 0 | 21 | 0 |
| diagnostics/k4-k5-share-screen-2026-10-01/replay-K5 | v0.8.0 stock | K4@a5fee929 | 7 | 7 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| diagnostics/kpool-k1-screen-2026-10-04/A | v0.8.0 stock | K3.25@0490d2f7 | 40 | 18 | 18 | 4 | 0 | 0 | 0 | 27 | 19 |
| diagnostics/kpool-k1-screen-2026-10-04/B | v0.8.0 stock (x1 draft) | K3.25@0490d2f7 | 40 | 19 | 12 | 9 | 0 | 0 | 0 | 22 | 18 |
| diagnostics/kpool-k1-screen-2026-10-04/C | v0.8.0 local kpoolring | K3.25@0490d2f7 | 40 | 20 | 15 | 5 | 0 | 0 | 0 | 26 | 22 |
| diagnostics/layout-bisect-2026-10-07/A1 | v0.9.0 stock | K3.25@0490d2f7 | 40 | 20 | 14 | 6 | 0 | 0 | 0 | 28 | 18 |
| diagnostics/layout-bisect-2026-10-07/A2 | v0.9.0 stock | K3.25@0490d2f7 | 40 | 19 | 16 | 5 | 0 | 0 | 0 | 25 | 20 |
| diagnostics/layout-bisect-2026-10-07/D1 | v0.9.0 EP2/DCP2 bundle | K3.25@0490d2f7 | 40 | 25 | 12 | 3 | 0 | 0 | 0 | 23 | 20 |
| diagnostics/layout-bisect-2026-10-07/D2 | v0.9.0 EP2/DCP2 bundle | K3.25@0490d2f7 | 40 | 26 | 9 | 5 | 0 | 0 | 0 | 24 | 20 |
| diagnostics/layout-bisect-2026-10-07/E1 | v0.9.0 local tailfix | K3.25@0490d2f7 | 40 | 22 | 15 | 3 | 0 | 0 | 0 | 22 | 15 |
| diagnostics/layout-bisect-2026-10-07/E2 | v0.9.0 local tailfix | K3.25@0490d2f7 | 40 | 25 | 10 | 5 | 0 | 0 | 0 | 23 | 17 |
| diagnostics/seed-replay-2026-09-30/v0.7.0 | v0.7.0 stock | K3.25@701cd745 | 40 | 28 | 9 | 3 | 0 | 0 | 0 | 23 | 19 |
| diagnostics/seed-replay-2026-09-30/v0.8.0 | v0.8.0 stock | K3.25@0490d2f7 | 40 | 16 | 19 | 5 | 0 | 0 | 0 | 31 | 27 |
| diagnostics/spec-k5-v080-2026-10-01/replay | v0.8.0 stock | K3.25@0490d2f7 | 16 | 12 | 0 | 4 | 0 | 0 | 0 | 6 | 6 |
| diagnostics/spec-screen-v070-k3-2026-10-01/replay | v0.7.0 stock (x3) | K3.25@701cd745 | 8 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| diagnostics/spec-slot-sharing-v080-2026-10-01/replay | v0.8.0 stock, sharing 0 | K3.25@0490d2f7 | 8 | 7 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| eval-matrix-2026-10-04/screens/S7 | v0.7.0 stock | K3.25@701cd745 | 40 | 27 | 9 | 4 | 0 | 0 | 0 | 20 | 16 |
| eval-matrix-2026-10-04/screens/S7K4p | v0.7.0 local kpoolring (INVALID: engine crash) | K4@a5fee929 | 40 | 0 | 0 | 0 | 0 | 0 | 40 | 2 | 2 |
| eval-matrix-2026-10-04/screens/S7p | v0.7.0 local kpoolring | K3.25@701cd745 | 40 | 27 | 7 | 6 | 0 | 0 | 0 | 20 | 16 |
| eval-matrix-2026-10-04/screens/S8K4p | v0.8.0 local kpoolring | K4@a5fee929 | 40 | 25 | 11 | 4 | 0 | 0 | 0 | 29 | 23 |
| completion/pilot/A | v0.9.1 stock | K3.25@0490d2f7 | 8 | 6 | 0 | 2 (cap 65k/131k) | 0 | 0 | 0 | 2 | 1 |
| completion/pilot/B | v0.9.1 EP2/DCP2 bundle | K3.25@0490d2f7 | 8 | 8 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| completion/replication/A | v0.9.1 stock | K3.25@0490d2f7 | 8 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| completion/replication/B | v0.9.1 EP2/DCP2 bundle | K3.25@0490d2f7 | 8 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**Totals:** 527 natural close, 224 loop, 92 exhaustion, 18 exhaustion cut at 120k, 40 error.

**Caveats on these rows:**
- **Shared seeds:** most screens send seed 1234 on every request, so their reps are not independent draws.
  - Distinct seeds: component-screen (5000 + rep; S1234 excepted), k4-k5 (1234 + rep − 1), and the pilot and
    replication manifests.
  - Empties: unseeded.
- **Client truncation:** zlib early stop in kpool, layout-bisect, eval-matrix and component; a 120k cut in
  spec-screen, slot-sharing and k4-k5.
- **Missing server logs:** the three empties dirs have none. The K4 empties engine is inferred.
- **Not the probe's engines:** only stock 0.7.0 (S7, seed-replay v0.7.0, empties v0.7.0, spec-screen-v070-k3, V79)
  and 0.9.x rows are relevant generators. No history was generated by B0 or B1: their GPQA passes kept no text.

**Repetition inside replies that still closed (20b):**
- 15 of 527 natural closes had a repeating checkpoint, and 14 were repeating at the close itself.
- Within natural closes, max checkpoint rep200 is p90 0.26 and p99 0.53.
- Exhaustions stay low: end rep200 p90 0.04, end zlib30k p50 0.30.
- Loops end at rep200 ≈ 1.0 with zlib30k p50 0.003.
- Loop onsets range from 8k to 324k tokens.
- 208 of the loops are still non-repeating at 64k and 135 at 128k. Their pre-onset windows are active reasoning.

### A4. No reasoning text or ids (counts only)

- **GPQA lm-eval samples (46 files, all passes):** final content only. No sample carries `<think>`/`</think>`.
  - One anomaly: 0.7.0 pass 1 (`glm53-flash-exl3-k3.25@tpurtell__pass1`, duplicated under `+ps__pass1`), doc 180.
    Its *content* is 190,446 characters (sha256 `04c5edd3…`); every other row is ≤ 3,023.
  - The reasoning before it was not kept, so the history cannot be reconstructed. Not a fixture.
- **Recording-proxy `*.requests.jsonl` (11 logged K3.25 passes, including B0 and B1):** finish reason, token counts
  and hashes only.
- **Steps 1-4 decode/prefill client outputs:** per-position metric rows only. Step 4 kept no traces.
- **Phase 2/3 `.pt` captures, e1-kvwriter, verify13:** indexer and KV state captured on the phase-2 prefixes. No
  new histories.
- **Bundle gate-runs and b0-mapguard `probe.jsonl`:** 96 decoded tokens per request after the phase-2 prefixes.
  Too short to be fixtures.
- **Short text probes, not scanned:** `eval-matrix-2026-10-04/{R0,R0n,R1,R1n,R2}/probe.json` and
  `diagnostics/tailfix-validation-2026-10-07/downstream/*-probe/probe.json`.
  - 16 GPQA prompts each; greedy 2,048 and sampled 4,096 tokens.
  - Usable only as extra short controls.
- **Excluded by rule:** the retired fixed-seed and other-runtime lm-eval samples in `archive/retired-2026-10/`. They
  hold content only and are cited nowhere.

## B. Usable fixtures by window class (`inventory.json` → `shortlist`)

All are K3.25-generated, active reasoning. Forced post-completion continuation is excluded.

| Class | Exact ids | Text (re-tokenized) | Engines that generated them | Questions |
|---|---|---|---|---|
| Short control (< 16k) | **24** step5/6 traces (closes at 176-4,795) | 114 natural closes < 16k (40 listed, distinct seeds first); phase-2 r5-r7 | exact: 0.7.0+kpoolfix, 0.9.1 bundle; text: all families | exact: docs 0-5; text: 12 docs |
| 64k-128k, active, no close in reach | 0 | phase-2 **r0 (120k) and r1 (64k)** ready as ids; 84 exhaustions non-repeating through 96k (0.7.0 19, 0.8.0 30, 0.9.0 26, 0.9.1 9); 121 loops whose onset is > 136k (pre-onset windows) | text only | exhaustions mostly doc 79 (74), plus 88, 127, 170 |
| Near a natural close, close at 64k-128k | 0 | **72** (0.7.0 14, 0.8.0 18, 0.9.0 15, 0.9.1 25), non-repeating at the close; 31 distinct-seed or unseeded | text only | 79, 88, 127, 13, 121, 90, 27, 170 |
| Near a natural close, close ≥ 128k | 0 | 73 (0.7.0 23, 0.9.x 36) | text only | 79, 88, 127, 13, 121, 27 |
| Near tentative conclusions in 56k-136k | 0 | 284 K3.25 rows hold 10,370 non-repeating strong commitment events in the band (natural close 139 rows, loop pre-onset 123, exhaustion 22) | text only | 13, 27, 79, 88, 90, 109, 121, 127, 170 |

For natural closes, the last strong commitment sits close to `</think>`. Over 497 closes with a commitment, the
distance from the last commitment to the close was:

| Statistic | Tokens to the close |
|---|---|
| Median | 28 |
| p75 | 362 |
| p90 | 1,254 |

30 closes have no strong commitment. A window ending at the close position is a positive control: `</think>` was
the actual next token.

**Matched-question cross-engine material:**
- Doc 79 has 0.7.0 histories (V79: 5 exhaustions, 3 closes) and stock 0.9.1 histories (B79: 8 exhaustions, 3
  closes) at the **same seeds 5001-5012**.
- Same seed on different engines is not a matched trajectory, but the question is shared and both sides have text.

**Suggested first set** (for the GPU proposal; positions in `inventory.json`):
1. Phase-2 r1/r0 (64k/120k, exact recomputed ids, already loaded by B1 tooling).
2. Two 0.7.0 and two 0.9.1 natural closes at 64k-128k, at windows ending at the close and at close − {2k, 8k}:
   - 0.7.0: e.g. `ca7fd76b2b7a` doc 13 close 102,579; `e87744068243` doc 88 close 121,500.
   - 0.9.1: `09ba09a88d7b` doc 79 seed 5004 close 97,408; `f0c2498e7b7f` doc 88 seed 5005 close 80,324.
3. Doc 79 exhaustions from V79 and B79 at 64k/96k/128k, plus a non-repeating commitment window in the band.
4. Step5/6 exact traces as the short control, scored at their natural close (decode logprobs exist there for a
   prefill-vs-decode check).

## Gaps

- **No exact generated ids beyond 5k tokens anywhere.** All 64k-128k and near-close-at-long-context fixtures are
  re-tokenized text: shared counterfactual histories, not replays.
  - 262 of 606 checkable rows reproduce the logged token count exactly; the rest differ by a few tokens.
- **The trailing whitespace before `</think>` is not in the saved text.** The parser strips it
  (`parser_engine.py:474-488`). The exact traces show the model closed directly after punctuation in 24/24, and
  count drift is 0 for many closes. Still, a text fixture's next-token-after-close state is reconstructed, not
  observed.
- **The exact traces are from modified engines** (0.7.0 + kpool fixes; 0.9.1 bundle) **at top_p 1.0.** None are
  from stock 0.7.0, B0 or B1 at the GPQA sampler.
- **No retained text from B0/B1 or from any stock GPQA pass.** The 0.7.0-vs-0.9.1 tail statistics of 19f have no
  text behind them.
- **Few questions:** the long histories come from 6-9 hard questions (79, 88, 127, 13, 121, 90 dominate), and
  several screens reuse seed 1234.
- **0.7.0 natural closes at 64k-128k:** 14 (5 questions). Near-close long-context material from stock 0.7.0 is thin.

## Hashes

Per-history sha256 of the reasoning text and of the re-tokenized ids (json.dumps convention, as in
`prep_requests.py`) are in `inventory.json`. The same file has per-trace sha256 of prompt, generated and active ids,
and per-request sha256 for r0-r7.
