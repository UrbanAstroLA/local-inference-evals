# Post-hoc token analysis (operator request, 2026-10-13; after the B1 seed-1236 result; not part of the frozen analysis)

Source: the recording-proxy request logs (completion_tokens, finish_reason, content_chars per request), joined to doc
ids through d7/exposure/gpqa_prompt_tokens.json. Counts only; no reply text.

Totals per pass (all replies / capped replies / completed replies; median completed):
- seed 1234: stock 3.811M (4 capped, 1.311M) / completed 2.500M, median 3,570; B0 3.316M (3, 0.983M) / 2.333M, median 3,907
- seed 1235: stock 4.243M (5, 1.638M) / 2.605M, median 3,450; B0 3.040M (3, 0.983M) / 2.057M, median 3,145
- seed 1236: stock 4.251M (6, 1.966M) / 2.285M, median 3,884; B1 4.809M (9, 2.949M) / 1.860M, median 3,621;
  col511 4.286M (6) / 2.320M; 0.7.0 3.374M (1) / 3.046M
Each capped reply is 327,680 tokens; B1's +0.558M total against stock equals its 3 extra capped replies (+0.983M) minus
0.425M fewer tokens in completed replies.

Per-document deltas on mutually completed documents (both finished, both non-empty):
- B1 - stock @1236, n=188: median -30, p10 -7,710, p90 +4,803, 47% longer
- col511 - stock @1236, n=190: median -66, p10 -6,240, p90 +5,176
- B0 - stock @1234, n=192: median -12, p10 -8,063, p90 +7,811; @1235, n=193: median -31, p10 -10,662, p90 +2,990
- stock @1235 - stock @1236 (different seeds), n=190: median -114, p10 -8,071, p90 +5,571

Newly empty / rescued documents at seed 1236 (B1 vs stock): 8, 69, 130, 145 newly capped; 170 rescued. Prompt tokens
238, 148, 334, 285, 217; exposure class NOT-unsplittable in every 0.9.1-family pass (d7/exposure). Other runs: doc 69
also capped in the column-511 arm at 1236; doc 145 capped in stock 0.7.0 at 1236; doc 170 capped in stock 0.9.1 at
1236; all five finished in every B0 and stock pass at seeds 1234-1235.

Not measurable: first token of divergence (no token-level logs are kept; samples hold final text only, and capped
replies have none); D7 code-path activation (no instrumentation in GPQA runs).
