# Replication of stock 0.7.0, variant R (re-freeze candidate analysis) -- DRY RUN: D1 old 0.7.0 record as rep, variant R (expect valid, O1)

valid: **True**; seeds [1234, 1235, 1236]; frozen module b5b13ce2; thresholds (proposed) O1 <= 8 with p1 < 0.05; O2 >= 12 or p2 < 0.05

## Checks
- [ok] old 0.7.0 record standing in as rep pass 1: exactly one samples file (1)
- [ok] old 0.7.0 record standing in as rep pass 1: 198 questions, ids 0-197, both filters (198)
- [ok] old 0.7.0 record standing in as rep pass 2: exactly one samples file (1)
- [ok] old 0.7.0 record standing in as rep pass 2: 198 questions, ids 0-197, both filters (198)
- [ok] old 0.7.0 record standing in as rep pass 2: request log has 198 requests, no re-sent request (198 requests, max per message 1)
- [ok] old 0.7.0 record standing in as rep pass 2: every request HTTP 200, seed 1235, max_tokens 327680, temperature 1.0, top_p 0.95 ()
- [ok] old 0.7.0 record standing in as rep pass 2: no request or response parse error ()
- [ok] old 0.7.0 record standing in as rep pass 3: exactly one samples file (1)
- [ok] old 0.7.0 record standing in as rep pass 3: 198 questions, ids 0-197, both filters (198)
- [ok] old 0.7.0 record standing in as rep pass 3: request log has 198 requests, no re-sent request (198 requests, max per message 1)
- [ok] old 0.7.0 record standing in as rep pass 3: every request HTTP 200, seed 1236, max_tokens 327680, temperature 1.0, top_p 0.95 ()
- [ok] old 0.7.0 record standing in as rep pass 3: no request or response parse error ()
- [ok] old 0.7.0 record standing in as rep: rows derived here reproduce the published rows (594 rows; empty, raw, strict, stated, hashes, finish reason, tokens) (0 field mismatches)
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 1: exactly one samples file (1)
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 1: 198 questions, ids 0-197, both filters (198)
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 1: request log has 198 requests, no re-sent request (198 requests, max per message 1)
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 1: every request HTTP 200, seed 1234, max_tokens 327680, temperature 1.0, top_p 0.95 ()
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 1: no request or response parse error ()
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 2: exactly one samples file (1)
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 2: 198 questions, ids 0-197, both filters (198)
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 2: request log has 198 requests, no re-sent request (198 requests, max per message 1)
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 2: every request HTTP 200, seed 1235, max_tokens 327680, temperature 1.0, top_p 0.95 ()
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 2: no request or response parse error ()
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 3: exactly one samples file (1)
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 3: 198 questions, ids 0-197, both filters (198)
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 3: request log has 198 requests, no re-sent request (198 requests, max per message 1)
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 3: every request HTTP 200, seed 1236, max_tokens 327680, temperature 1.0, top_p 0.95 ()
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 3: no request or response parse error ()
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record): rows derived here reproduce the published rows (594 rows; empty, raw, strict, stated, hashes, finish reason, tokens) (0 field mismatches)
- [ok] 0.7.0 K3.25 DFlash2 x5 (historical 3-pass record) pass 1: exactly one samples file (1)
- [ok] 0.7.0 K3.25 DFlash2 x5 (historical 3-pass record) pass 1: 198 questions, ids 0-197, both filters (198)
- [ok] 0.7.0 K3.25 DFlash2 x5 (historical 3-pass record) pass 2: exactly one samples file (1)
- [ok] 0.7.0 K3.25 DFlash2 x5 (historical 3-pass record) pass 2: 198 questions, ids 0-197, both filters (198)
- [ok] 0.7.0 K3.25 DFlash2 x5 (historical 3-pass record) pass 2: request log has 198 requests, no re-sent request (198 requests, max per message 1)
- [ok] 0.7.0 K3.25 DFlash2 x5 (historical 3-pass record) pass 2: every request HTTP 200, seed 1235, max_tokens 327680, temperature 1.0, top_p 0.95 ()
- [ok] 0.7.0 K3.25 DFlash2 x5 (historical 3-pass record) pass 2: no request or response parse error ()
- [ok] 0.7.0 K3.25 DFlash2 x5 (historical 3-pass record) pass 3: exactly one samples file (1)
- [ok] 0.7.0 K3.25 DFlash2 x5 (historical 3-pass record) pass 3: 198 questions, ids 0-197, both filters (198)
- [ok] 0.7.0 K3.25 DFlash2 x5 (historical 3-pass record) pass 3: request log has 198 requests, no re-sent request (198 requests, max per message 1)
- [ok] 0.7.0 K3.25 DFlash2 x5 (historical 3-pass record) pass 3: every request HTTP 200, seed 1236, max_tokens 327680, temperature 1.0, top_p 0.95 ()
- [ok] 0.7.0 K3.25 DFlash2 x5 (historical 3-pass record) pass 3: no request or response parse error ()
- [ok] 0.7.0 K3.25 DFlash2 x5 (historical 3-pass record): rows derived here reproduce the published rows (594 rows; empty, raw, strict, stated, hashes, finish reason, tokens) (0 field mismatches)
- [ok] variant R: rep seeds are the records' per-pass seeds 1234/1235/1236 ((1234, 1235, 1236))
- [FAIL] rep request logs: request seeds are exactly the variant's [1234, 1235, 1236], 198 requests each, nothing else (informative for this stand-in) ({"1235": 198, "1236": 198})
- [FAIL] rep pass 1: lm-eval model_args seed 1234 (sent in every request) and process seed 1235 (informative for this stand-in) (model_args seed None, process seed 1235)
- [ok] rep pass 2: lm-eval model_args seed 1235 (sent in every request) and process seed 1236 (informative for this stand-in) (model_args seed 1235, process seed 1236)
- [ok] rep pass 3: lm-eval model_args seed 1236 (sent in every request) and process seed 1237 (informative for this stand-in) (model_args seed 1236, process seed 1237)
- [ok] rep pass 1: lm-eval task hash 87e807b3b27c8661 (same task, rendering and answer ordering as both records) (87e807b3b27c8661)
- [ok] rep pass 2: lm-eval task hash 87e807b3b27c8661 (same task, rendering and answer ordering as both records) (87e807b3b27c8661)
- [ok] rep pass 3: lm-eval task hash 87e807b3b27c8661 (same task, rendering and answer ordering as both records) (87e807b3b27c8661)
- [ok] v091 record: task hash 87e807b3b27c8661 in all three passes (87e807b3,87e807b3,87e807b3)
- [ok] old070 record: task hash 87e807b3b27c8661 in all three passes (87e807b3,87e807b3,87e807b3)
- [ok] rep vs v091: both arms have all 594 question-passes (arm 594, cmp 594)
- [ok] rep vs v091: (question, pass) pairs agree on seed (0 mismatches)
- [ok] rep vs v091: (question, pass) pairs agree on prompt_hash (0 mismatches)
- [ok] rep vs v091: (question, pass) pairs agree on doc_hash (0 mismatches)
- [ok] rep vs v091: (question, pass) pairs agree on target_hash (0 mismatches)
- [ok] rep vs v091: (question, pass) pairs agree on messages_sha256 (0 mismatches)
- [ok] rep vs v091: server-rendered prompt token counts agree where both logged (396 compared, 0 differ)
- [ok] rep vs old070: both arms have all 594 question-passes (arm 594, cmp 594)
- [ok] rep vs old070: (question, pass) pairs agree on seed (0 mismatches)
- [ok] rep vs old070: (question, pass) pairs agree on prompt_hash (0 mismatches)
- [ok] rep vs old070: (question, pass) pairs agree on doc_hash (0 mismatches)
- [ok] rep vs old070: (question, pass) pairs agree on target_hash (0 mismatches)
- [ok] rep vs old070: (question, pass) pairs agree on messages_sha256 (0 mismatches)
- [ok] rep vs old070: server-rendered prompt token counts agree where both logged (396 compared, 0 differ)
- [ok] rep session 2 warm-up: exactly one engine start in the server log (1)
- [ok] rep session 2 warm-up: 9 release warm-up lines (record's count), ending with the ready line (9 warm-up lines, 1 ready lines)
- [ok] rep session 2 warm-up: no GPQA chat-completion request logged before the ready line (0 before)
- [ok] rep session 2 warm-up: first GPQA request after the last timestamped line before the ready line (first request 1791619684.386, last line before ready 1791619656.0)
- [ok] rep session 1: non_default_args_sha256 equals the old 0.7.0 record's passes 2-3 server log (74f26837e0e20391bbc8e39c vs 74f26837e0e20391bbc8e39c)
- [ok] rep session 1: vllm_version equals the old 0.7.0 record's passes 2-3 server log (['0.1.dev20051+g487ecf18 vs ['0.1.dev20051+g487ecf18)
- [ok] rep session 1: model_snapshots equals the old 0.7.0 record's passes 2-3 server log (['701cd7456c13d87bf0147a vs ['701cd7456c13d87bf0147a)
- [ok] rep session 1: env_names_logged_sha256 equals the old 0.7.0 record's passes 2-3 server log (014d31a48bdd11c1a48236e5 vs 014d31a48bdd11c1a48236e5)
- [ok] rep session 1: cudagraph_mode equals the old 0.7.0 record's passes 2-3 server log (['FULL_AND_PIECEWISE'] vs ['FULL_AND_PIECEWISE'])
- [ok] stated adjudicator source unchanged since freeze (informative) (731f27d4a436805baae367096db8b039ae2d723f7944dda52f81fd2a96d017f6)

## Reading
- rep empties 4 of 594 (old 0.7.0 record 4, 0.9.1 record 16)
- C1 (rep fewer than 0.9.1): one-sided p = 0.0044; reduction +2.02 points (95% +0.67 to +3.54)
- C2 (rep more than old 0.7.0): one-sided p = 1.0000; increase +0.00 points (95% +0.00 to +0.00)
- outcome **O1**: substantially fewer empties again: replication support for the 0.7.0 advantage relative to the 0.9.1 record (repeatability under the recorded seeds, historical comparators); the tail-mapping/DCP interaction on 0.7.0 becomes the concrete candidate (amendment 12)
- what O1 means: the recorded-seed repeat (variant R) sustains a low empty count and separates from the historical 0.9.1 comparator under the specified test. It does not establish equivalence to the original 0.7.0 rate, nor a contemporaneous causal engine advantage.
- prespecified next step: the historical advantage merits further investigation; consider a contemporaneous comparison before another broad mechanism search.
- tests: Both tests are exact sign-flip tests: 'exact' refers to the calculation of the p-value (full enumeration of the sign patterns), given the assumptions. The assumptions are that, under the null, each question's pair of per-question empty counts (rep, historical record) is exchangeable, i.e. the historical comparator behaves as a draw from the same process as the new run on that question, and that questions are the independent units. Neither is designed: the comparators are historical records, so time, state and other changes since are confounded with the comparison.
- limits: historical comparators for both C1 and C2; exchangeability assumed, not designed; thresholds are operational; repeatability under the recorded seeds (coupled to both records), not a fresh-seed estimate
- identical answered-reply hashes vs old 0.7.0 per pass: {'pass1': 198, 'pass2': 195, 'pass3': 197}
- empty in both rep and old 0.7.0: ['q79p2', 'q81p2', 'q121p2', 'q145p3']
- log reply-length correlation vs old 0.7.0, matched passes {'pass1': 1.0, 'pass2': 1.0, 'pass3': 1.0}

## Secondary (descriptive, never the reading): P(finish | still generating at 100,000 tokens)
- rep, passes 1-3: 10/14 finished (11 questions; P = 0.71); at >= 200k 4/8
- 0.9.1 record, passes 1-3: 10/25 finished (15 questions; P = 0.4); at >= 200k 2/17
- old 0.7.0 record, passes 2-3 (pass 1 unlogged: lengths unknown): 10/14 finished (11 questions; P = 0.71); at >= 200k 4/8
- rep, passes 2-3 (seed-matched to the old record's logged passes): 10/14 finished (11 questions; P = 0.71); at >= 200k 4/8
- rep_minus_v091: 0.3143 (95% question-bootstrap [0.0333, 0.5584], 0 undefined)
- rep_minus_old070_passes_2_3: 0.0 (95% question-bootstrap [0.0, 0.0], 0 undefined)
- rep_passes_2_3_minus_old070_passes_2_3_seed_matched: 0.0 (95% question-bootstrap [0.0, 0.0], 0 undefined)
- limits: Descriptive only. The small number of finishing events (about 10 per record) and the wide intervals prevent this measure from adjudicating a termination mechanism. Reaching 100,000 tokens selects different trajectories on different engines, so the conditional comparison is not a causal engine effect.

## Seeds and task identity observed per pass
- pass1: {'variant_request_seed': 1234, 'requests_logged_with_that_seed': 0, 'lm_eval_model_args_seed': None, 'lm_eval_process_seed': 1235, 'task_hash': '87e807b3b27c86618285efdab99466609ad12f022ae453146595fc8306b053b0'}
- pass2: {'variant_request_seed': 1235, 'requests_logged_with_that_seed': 198, 'lm_eval_model_args_seed': 1235, 'lm_eval_process_seed': 1236, 'task_hash': '87e807b3b27c86618285efdab99466609ad12f022ae453146595fc8306b053b0'}
- pass3: {'variant_request_seed': 1236, 'requests_logged_with_that_seed': 198, 'lm_eval_model_args_seed': 1236, 'lm_eval_process_seed': 1237, 'task_hash': '87e807b3b27c86618285efdab99466609ad12f022ae453146595fc8306b053b0'}
