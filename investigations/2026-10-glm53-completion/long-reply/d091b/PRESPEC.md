# Pre-specified definitions (long-reply discovery)

Written 2026-10-16 **before any per-window metric was computed and before any reasoning text was read** for any
reply. Up to this point only `finish_reason`, `completion_tokens` and character counts from the raw file had been
listed. The sha256 of this file is recorded in `out/prespec.sha256` at the moment of writing; the scripts check it
before running. Anything not defined here and used later in REPORT.md is labelled post hoc there.

## Units and positions

- Text analysed: `response.choices[0].message.reasoning`, re-tokenized with the K3.25 tokenizer (snapshot
  701cd745, `tokenizer.json`), `add_special_tokens=False`. Re-tokenization can differ from the generated ids; the
  drift `completion_tokens - (retok(reasoning) + retok(content))` is reported for every reply.
- g = 0-based index into the re-tokenized reasoning ("generated position"). Absolute sequence position
  a = prompt_tokens + g (any template-supplied think-open token is part of the prompt tokens).
- Window grid: fixed, non-overlapping W = 4,096-token windows starting at g = 0. The last partial window is kept and
  flagged when shorter than 1,024 tokens.

## Per-window metrics

1. `cpt`: characters per token (characters from the tokenizer's offsets).
2. `rep20`, `rep50`, `rep100`: share of start positions i in the window (i <= N - n) whose n-gram t[i:i+n] also starts
   at some j < i in the same reply. 64-bit polynomial hashing; a random sample of hits is verified by direct
   comparison.
3. Exact repeated spans: for each i with a repeated 20-gram, d(i) = i - j*, where j* is the nearest earlier start
   with the same 20-gram. A span is a maximal run of consecutive i with a repeated 20-gram and constant d; its
   length is run + 19 tokens and its period is d. Per window: the longest span starting in it and its period.
4. `zr`: zlib (level 9) compressed bytes / UTF-8 bytes of the window text.
5. `dlr`: distinct / total non-empty stripped lines in the window. `line_seen`: share of the window's non-empty
   lines of at least 8 characters whose stripped text occurred as a line earlier in the reply.
6. `cjk`: share of characters in CJK ranges; `nonascii`: share of non-ASCII characters.
7. `chk`: count of verification-phrase matches (fixed regex list in `scripts/trajectory.py`), per 1,000 tokens;
   `chk_rep`: share of those matches whose containing sentence (lower-cased, whitespace collapsed) occurred
   earlier in the reply. Counts only, no text written.

## Change point (primary, on rep50)

- R50(i) = mean of the rep50 indicator over start positions [i, i + 4,096), at token resolution.
- **Terminal degeneration onset** O = the smallest i such that R50(j) >= 0.5 for every valid j >= i (valid: the
  4,096-token window fits inside the reasoning). Defined only when R50 at the last valid j is >= 0.5.
- **High-repetition episode**: a maximal interval of i with R50(i) >= 0.5 lasting at least 2,048 positions.
  An episode is **recovered** when it is not terminal (R50 falls below 0.5 again before the end of the reasoning).
- Sensitivity only (not primary): thresholds 0.3 and 0.7.

## Loop type (on [O, N) for replies with a terminal onset)

- E = share of tokens in [O, N) covered by exact repeated spans of length >= 256 tokens.
- P = the period carrying the most covered tokens among those spans.
- **short-period exact loop**: E >= 0.5 and P <= 4,096.
- **long-period exact cycling**: E >= 0.5 and P > 4,096.
- **fragmentary recycling**: E < 0.5 (50-gram repetitive without long exact copies); grouped with long-period
  cycling as "cycling without a short exact loop" when the question asks for three classes.
- **no terminal loop**: no terminal onset. For capped replies in this class, the final-window rep50 and max R50 are
  reported so "still-novel to the cap" can be judged.

## Finished long replies

- Distance from the end of the last fixed 4k grid window with rep50 >= 0.5 to the end of the reasoning (tokens);
  "none" if no such window. Same at 0.25 as sensitivity.
- Content section length: re-tokenized tokens and characters of `content`.

## Sets

- At risk: completion_tokens >= 100,000 in rep070-F (all three passes). Comparison: 50,000 <= T < 100,000 in rep070-F.
- Engine "long" for the panel: completion_tokens >= 100,000; capped: finish_reason `length` at 327,680.

## Geometry (descriptive)

For onsets (terminal and episode starts), on absolute positions a: residue mod 3,584 (0.7.0 attention block) and
mod 4 (pool / tail-ring width), distance to the nearest multiple of 2,043 and of 3,584, distance to 65,536 /
131,072 / 262,144. Uniformity of the mod-3,584 phase: mean resultant length with a Monte Carlo null (20,000 draws).
Resolution caveat: onset positions are defined by a 4,096-token rolling window on re-tokenized text, so any
alignment finer than a few hundred tokens (mod 4 in particular) is not resolvable and is reported for completeness
only.
