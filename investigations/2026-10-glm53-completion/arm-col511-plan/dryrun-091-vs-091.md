# Column-511 arm analysis: 0.9.1 K3.25 DFlash2 x3 (same record as comparator) vs 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record)

**DRY RUN: 0.9.1 record against itself (sanity: zero effect expected). Not a result of the column-511 arm.**

script sha256 b5b13ce2155085ca46fb24877f12b7cd0a5cdd16eb4cf19012958e7b86ae0033; spec sha256 523ec22806059f2eb384fa2225af2db4d8b3c64a3b3f4d94ebed8312bb12d610; valid: **True**

## Checks
- [ok] 0.9.1 K3.25 DFlash2 x3 (same record as comparator) pass 1: exactly one samples file (1)
- [ok] 0.9.1 K3.25 DFlash2 x3 (same record as comparator) pass 1: 198 questions, ids 0-197, both filters (198)
- [ok] 0.9.1 K3.25 DFlash2 x3 (same record as comparator) pass 1: request log has 198 requests, no re-sent request (198 requests, max per message 1)
- [ok] 0.9.1 K3.25 DFlash2 x3 (same record as comparator) pass 1: every request HTTP 200, seed 1234, max_tokens 327680, temperature 1.0, top_p 0.95
- [ok] 0.9.1 K3.25 DFlash2 x3 (same record as comparator) pass 1: no request or response parse error
- [ok] 0.9.1 K3.25 DFlash2 x3 (same record as comparator) pass 2: exactly one samples file (1)
- [ok] 0.9.1 K3.25 DFlash2 x3 (same record as comparator) pass 2: 198 questions, ids 0-197, both filters (198)
- [ok] 0.9.1 K3.25 DFlash2 x3 (same record as comparator) pass 2: request log has 198 requests, no re-sent request (198 requests, max per message 1)
- [ok] 0.9.1 K3.25 DFlash2 x3 (same record as comparator) pass 2: every request HTTP 200, seed 1235, max_tokens 327680, temperature 1.0, top_p 0.95
- [ok] 0.9.1 K3.25 DFlash2 x3 (same record as comparator) pass 2: no request or response parse error
- [ok] 0.9.1 K3.25 DFlash2 x3 (same record as comparator) pass 3: exactly one samples file (1)
- [ok] 0.9.1 K3.25 DFlash2 x3 (same record as comparator) pass 3: 198 questions, ids 0-197, both filters (198)
- [ok] 0.9.1 K3.25 DFlash2 x3 (same record as comparator) pass 3: request log has 198 requests, no re-sent request (198 requests, max per message 1)
- [ok] 0.9.1 K3.25 DFlash2 x3 (same record as comparator) pass 3: every request HTTP 200, seed 1236, max_tokens 327680, temperature 1.0, top_p 0.95
- [ok] 0.9.1 K3.25 DFlash2 x3 (same record as comparator) pass 3: no request or response parse error
- [ok] 0.9.1 K3.25 DFlash2 x3 (same record as comparator): rows derived here reproduce the published rows (594 rows; empty, raw, strict, stated, hashes, finish reason, tokens) (0 field mismatches)
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 1: exactly one samples file (1)
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 1: 198 questions, ids 0-197, both filters (198)
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 1: request log has 198 requests, no re-sent request (198 requests, max per message 1)
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 1: every request HTTP 200, seed 1234, max_tokens 327680, temperature 1.0, top_p 0.95
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 1: no request or response parse error
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 2: exactly one samples file (1)
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 2: 198 questions, ids 0-197, both filters (198)
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 2: request log has 198 requests, no re-sent request (198 requests, max per message 1)
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 2: every request HTTP 200, seed 1235, max_tokens 327680, temperature 1.0, top_p 0.95
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 2: no request or response parse error
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 3: exactly one samples file (1)
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 3: 198 questions, ids 0-197, both filters (198)
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 3: request log has 198 requests, no re-sent request (198 requests, max per message 1)
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 3: every request HTTP 200, seed 1236, max_tokens 327680, temperature 1.0, top_p 0.95
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record) pass 3: no request or response parse error
- [ok] 0.9.1 K3.25 DFlash2 x3 (historical 3-pass record): rows derived here reproduce the published rows (594 rows; empty, raw, strict, stated, hashes, finish reason, tokens) (0 field mismatches)
- [ok] both arms have all 594 question-passes (arm 594, cmp 594)
- [ok] (question, pass) pairs agree on seed (0 mismatches)
- [ok] (question, pass) pairs agree on prompt_hash (0 mismatches)
- [ok] (question, pass) pairs agree on doc_hash (0 mismatches)
- [ok] (question, pass) pairs agree on target_hash (0 mismatches)
- [ok] (question, pass) pairs agree on messages_sha256 (0 mismatches)
- [ok] server-rendered prompt token counts agree where both logged (594 compared, 0 differ)
- [ok] stated adjudicator source unchanged since freeze (informative) (731f27d4a436805baae367096db8b039ae2d723f7944dda52f81fd2a96d017f6)

## Primary: empty replies, 594 question-passes per arm
- arm 16 (2.69%), comparator 16 (2.69%); per pass arm/cmp: pass1 5/5, pass2 5/5, pass3 6/6
- questions with any empty: arm 11, cmp 11; d_q > 0 on 0 questions, < 0 on 0
- exact sign-flip, T = 0, m = 0 nonzero questions (1 sign patterns): one-sided p = 1.0000 (two-sided 1.0000, reference)
- empty-rate reduction (cmp - arm) +0.00 points, 95% question-bootstrap interval +0.00 to +0.00 (20000 replicates, seed 20261011)
- relative reduction 0% (interval 0% to 0%; 0 replicates undefined)
- band: >= 13 (operational band: little improvement or deterioration)
- reading: R3: little improvement or deterioration = the localized correction stands but is not supported as the dominant completion explanation

## Secondary
- raw (flexible-extract), all 594 requests: arm 516 (86.9%), cmp 516 (86.9%); arm - cmp +0.00 points (95% +0.00 to +0.00); two-sided sign-flip p 1.0000 (reference). Per pass arm: 85.4, 88.9, 86.4; cmp: 85.4, 88.9, 86.4
- stated (audited method), all 594 requests: arm 522 (87.9%), cmp 522 (87.9%); arm - cmp +0.00 points (95% +0.00 to +0.00); two-sided sign-flip p 1.0000 (reference). Per pass arm: 85.9, 89.9, 87.9; cmp: 85.9, 89.9, 87.9
- A nonsignificant accuracy test does not establish no accuracy cost; read the interval.
- stated differs from raw on 8 arm rows (hand read required before the stated figure is final), 8 comparator rows
- rescued (cmp empty, arm answered): 0 (-); arm raw correct on them 0
- introduced (cmp answered, arm empty): 0 (-); cmp raw correct lost 0
- empty in both: 16 (q12p1, q55p2, q79p2, q79p3, q81p2, q81p3, q88p1, q88p3, q109p1, q120p1, q121p2, q127p1, q127p3, q147p2, q147p3, q170p3); answered in both: 578, raw correct arm 516 vs cmp 516
- arm empties by end: {'cap_exhaustion': 15, 'premature_stop': 1, 'other_finish': 0, 'not_logged': 0}; premature stops: [{'id': 'q12p1', 'completion_tokens': 38}]; logged rows 594; answered with finish 'length': 0
  - answered completion tokens: median 3508, p90 35394, p99 109684, max 274150 (n 578); answered above 32k/64k/128k/200k: 62/24/4/2
- cmp empties by end: {'cap_exhaustion': 15, 'premature_stop': 1, 'other_finish': 0, 'not_logged': 0}; premature stops: [{'id': 'q12p1', 'completion_tokens': 38}]; logged rows 594; answered with finish 'length': 0
  - answered completion tokens: median 3508, p90 35394, p99 109684, max 274150 (n 578); answered above 32k/64k/128k/200k: 62/24/4/2
- mean completion tokens per request, arm - cmp, 198 questions logged in all passes: +0 (95% +0 to +0)
- arm cost: request windows 7.89 h (15.78 GPU-h at 2 GPUs); occupied GPU-h n/a (not available); raw correct per GPU-h n/a
  - pass1: window 2.37 h, 3,810,906 tokens, 447.2 tok/s aggregate, median answered request 58.1 tok/s
  - pass2: window 2.73 h, 4,243,257 tokens, 431.9 tok/s aggregate, median answered request 57.4 tok/s
  - pass3: window 2.79 h, 4,251,266 tokens, 422.8 tok/s aggregate, median answered request 57.7 tok/s
  - route exposure pass1 (client-side in-flight, approximate): share of busy time with 1-4 in flight 0.050, 5+ 0.950; mean share of request life at 1-4: empty 0.033, answered 0.0018
  - route exposure pass2 (client-side in-flight, approximate): share of busy time with 1-4 in flight 0.143, 5+ 0.857; mean share of request life at 1-4: empty 0.0835, answered 0.0048
  - route exposure pass3 (client-side in-flight, approximate): share of busy time with 1-4 in flight 0.177, 5+ 0.823; mean share of request life at 1-4: empty 0.2047, answered 0.0027
- cmp cost: request windows 7.89 h (15.78 GPU-h at 2 GPUs); occupied GPU-h 16.593 (chain logs); raw correct per GPU-h 31.1
  - pass1: window 2.37 h, 3,810,906 tokens, 447.2 tok/s aggregate, median answered request 58.1 tok/s
  - pass2: window 2.73 h, 4,243,257 tokens, 431.9 tok/s aggregate, median answered request 57.4 tok/s
  - pass3: window 2.79 h, 4,251,266 tokens, 422.8 tok/s aggregate, median answered request 57.7 tok/s
  - route exposure pass1 (client-side in-flight, approximate): share of busy time with 1-4 in flight 0.050, 5+ 0.950; mean share of request life at 1-4: empty 0.033, answered 0.0018
  - route exposure pass2 (client-side in-flight, approximate): share of busy time with 1-4 in flight 0.143, 5+ 0.857; mean share of request life at 1-4: empty 0.0835, answered 0.0048
  - route exposure pass3 (client-side in-flight, approximate): share of busy time with 1-4 in flight 0.177, 5+ 0.823; mean share of request life at 1-4: empty 0.2047, answered 0.0027
- cmp server log glm53-flash-exl3-k3.25@tpurtell+v0.9.1.server.log: starts 1, cited block 0, non-default args sha256 7025a96888a67642, cudagraph ['FULL_AND_PIECEWISE'], KV pool [4707515], status {'lines': 853, 'max_running': 8, 'lines_waiting': 0, 'peak_kv_usage_pct': 39.6, 'mean_gen_tok_s_8plus_running': 459.4, 'share_lines_running_1to4': 0.0492, 'share_lines_running_5plus': 0.9496}, in window {'preemption_lines': 0, 'illegal_memory_access_lines': 0, 'nan_warning_lines': 0}
- cmp server log glm53-flash-exl3-k3.25@tpurtell+v0.9.1+ps.server.log: starts 1, cited block 0, non-default args sha256 7025a96888a67642, cudagraph ['FULL_AND_PIECEWISE'], KV pool [4707515], status {'lines': 1988, 'max_running': 8, 'lines_waiting': 0, 'peak_kv_usage_pct': 41.6, 'mean_gen_tok_s_8plus_running': 447.7, 'share_lines_running_1to4': 0.16, 'share_lines_running_5plus': 0.84}, in window {'preemption_lines': 0, 'illegal_memory_access_lines': 0, 'nan_warning_lines': 0}

Limits: historical control; exchangeability is assumed, not designed; matching recipes and seeds does not make this a randomized experiment; a positive result supports improvement relative to the recorded baseline only. Corrupted/NaN index rows are not measurable from GPQA outputs and are reported separately where instrumentation exists. The unresolved tail-ring defect is present in both arms and limits what a negative result could exclude.
