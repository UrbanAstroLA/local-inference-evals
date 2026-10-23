# Long-reply discovery: what capped and finished long GPQA replies look like (rep070-F, stock 0.7.0)

2026-10-16/17. **Zero GPU.** A CPU-only container (`--network none`, `CUDA_VISIBLE_DEVICES=`, no `--gpus`) did the
re-tokenization; everything else is host stdlib Python. No server, no model request, no upload.

Evidence labels: **[M]** measured here, **[I]** inferred, **[R]** read from another record. Everything is descriptive:
29 at-risk replies from 18 questions, 10 of them capped. No test is claimed.

**Privacy.** No question, answer, answer letter or reply text appears in any file here. Outputs hold ids (`q79p2` =
doc 79, pass 2), positions, counts and metrics.

(Report text written by the discovery agent and saved by the coordinator. The agent could not write this file itself.)

## 1. Inputs and method
- **Text:** `rep070/run-F/raw/s{1,2}.raw.jsonl.gz` (594 requests, seeds 70701-70703).
  - Doc ids come from the lm-eval samples via the frozen `messages_hash` convention: 18 of 18 join checks pass.
  - Every length equals the tail-audit records: 2,178 compared, 0 differ [M].
- **Sets:**
  - *at risk*: T ≥ 100,000 tokens. 29 replies (19 finished, 10 capped) from 18 questions.
  - *comparison*: 50,000 ≤ T < 100,000. 24 replies, all finished [M].
- **Re-tokenization:** K3.25 `tokenizer.json` (snapshot 701cd745), `add_special_tokens=False`.
  - Drift from the logged token count is 0-8 tokens for all 53 replies, so positions are accurate to within about 10
    tokens.
  - The 64-bit n-gram hashes were checked against direct comparison: 0 failures [M].
- **Older records** (0.7.0 orig, stock 0.9.1, col511 arm, B0 with all three seeds, B1) supply lengths and character
  counts only (`out/lengths_all.jsonl`).

**Pre-specification.** `PRESPEC.md` (sha256 `97040a5c…`) was written before any metric was computed or any reasoning
text was read. `trajectory.py` refuses to run if that file has changed. Capped replies were processed first; finished
replies followed with the definitions unchanged.

Definitions:
- **Per 4,096-token window:** chars/token; `rep20/50/100` (share of the window's n-gram starts already seen earlier in
  the reply); longest exact span; zlib ratio; distinct-line ratio; script shares; self-check counts. The self-check
  classes are verification phrases and interjections, counted only.
- **Onset:** R50(i) is the rolling share of repeated 50-grams over [i, i+4,096). The terminal onset is the first i
  after which R50 ≥ 0.5 holds to the end of the reasoning.
- **Episode:** R50 ≥ 0.5 for at least 2,048 positions. An episode is *recovered* if R50 drops below 0.5 again.
- Thresholds 0.3 and 0.7 are sensitivity checks only.

**Post hoc additions, labelled where used:**
- the longest repeated substring (LRS, via a suffix array) and lag-match rates. These were added because the
  pre-specified span measure splits internally repetitive cycles;
- precursor windows, within-question comparisons, question-matched finishing, chars/token by engine, and the cost
  model.

## 2. Trajectory metrics
Medians over replies of each reply's median over full 4k windows [M]. Distinct-line ratio is 1.00 in every set (period-2 loops contain no newlines), so it is not shown.

| set | n | chars/tok | rep50 | zlib | digit share | punct share | ≤1-char tokens | verify /1k | interjection /1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| comparison 50-100k | 24 | 3.09 | 0.000 | 0.37 | 0.034 | 0.086 | 0.29 | 1.10 | 3.1 |
| finished ≥ 100k | 19 | 3.15 | 0.000 | 0.36 | 0.033 | 0.086 | 0.28 | 0.98 | 5.4 |
| capped, before onset | 10 | 2.31 | 0.001 | 0.36 | 0.069 | 0.138 | 0.40 | 0.73 | 9.8 |
| capped, after onset | 8 | 2.50 | 1.000 | 0.008 | 0.000 | 0.175 | 0.44 | 0.00 | 277 |

**Low chars/token is already there in the first 100k tokens,** before any repetition: 2.31 against 3.15, AUC 0.12 [M].
- It goes with more digits (0.082 against 0.036, AUC 0.79), more punctuation and more single-character tokens.
- Within the same question, the capped pass is lower than the finished passes on q8, q69 and q88, but not on q127.
- So it is mostly a property of the trajectory before any loop [M]. That this reflects a computation-heavy path is
  [I].

**Interjection-class phrases** are highest in capped replies before the onset (9.8 per 1k).
- They rise about 2.5-4× in the last 32k tokens before the onset, in 6 of 9 terminal onsets.
- Verification-class phrases are not elevated.
- In all four period-2 loops, the 2-token unit is one interjection-class match plus a separator [M].

## 3. Capped replies: onset and loop type

| id | onset g (abs) | R50 at 100k | post hoc loop |
|---|---:|---:|---|
| q127p3 | 5,376 (8,171) | 1.00 | exact, period 35 |
| q88p2 | 54,812 (55,041) | 1.00 | exact, period 514 |
| q88p1 | 143,158 (143,387) | 0.00 | exact, period 2 |
| q69p2 | 181,125 (181,273) | 0.00 | exact, period 1,001 (lag match 0.80) |
| q120p3 | 185,274 (185,550) | 0.02 | exact, period 2 |
| q8p2 | 236,765 (237,003) | 0.00 | exact, period 2 |
| q158p3 | 280,750 (280,945) | 0.03 | exact, period 2 |
| q79p1 | 288,394 (288,485) | 0.01 | exact, period 246 |
| q79p3 | none (R50 max 0.60) | 0.00 | rising partial repetition |
| q79p2 | none (R50 max 0.07) | 0.00 | **novel text to the cap** |

- **Classification.** The pre-specified rule gave 5 short-period exact loops, 3 "fragmentary" and 2 with no terminal
  loop. The post hoc LRS shows the 3 "fragmentary" replies are exact loops whose cycles repeat internally.
- **Corrected reading:** 8 of 10 capped replies end in an exact loop with a period of 2-1,001 tokens. None shows
  long-period cycling. 2 (both q79) do not loop [M].
- **State at 100k:** only 2 of 10 capped replies were looping at 100k. 6 started 43k-188k tokens later and 2 (q79p2, q79p3) never formed a terminal loop (erratum 22am). All
  finished replies had R50 ≤ 0.018 at 100k [M].
- **Recovery:**
  - Recovered episodes: q127p1 and q127p2 (both then finished) and q79p3 (then capped).
  - Finished from inside repetition: q165p2 and q94p2.
  - No capped reply recovered from a terminal loop.
  - Episodes occur in 9 of 10 capped replies, against 2 of 19 finished and 1 of 24 in the comparison set [M].

## 4. Finished long replies
- **Novelty to the end:** 15 of 19 never exceed R50 0.04; they produce new text right up to the end of the reasoning.
- **Content section:** 194-780 tokens (median 506). All 19 have an extractable final letter; 11 of 19 are
  stated-correct.
- **Closing signature:** the verification-phrase rate in the last window is above the reply's own median in 16 of 19.
  This is weak, and post hoc [M].

## 5. Onset positions against engine geometry (descriptive)
- **Block size 3,584:** no clustering. Mean resultant length 0.12 for onsets (Monte Carlo p 0.87) and 0.22 for all
  23 events (p 0.32).
- **Powers of two:** no event lies within 8k tokens of 65,536, 131,072 or 262,144.
- **2,043-token prefill chunks:** distances to multiples look uniform.
- **mod 4:** not resolvable at this onset resolution.
- **Spread:** onsets spread over 8k-288k; 6 of 8 capped onsets fall after 140k.
- **q127:** the earliest onset is on q127, the only multi-chunk prefill prompt. But q127 is also long or capped on the
  0.9.1 family (9 of 10 passes long), so this does not single out the prefill [R].

## 6. Questions: the long-reply panel
- **Concentration:** 118 long events fall on 33 questions. The top 5 (79, 127, 147, 88, 81) hold 46% and the top 10
  hold 70% [M].
- **Persistence:** per-question long rates correlate between the 0.7.0 and 0.9.1 families (Spearman 0.60) [M].
- **Question-matched finishing** (14 questions long in both families): the 0.7.0 family finished 23 of 36 (0.64), the
  0.9.1 family 18 of 58 (0.31). 0.7.0 is higher on 8 questions, lower on 1 (q79) and equal on 5 [M].
  - The finishing gap is therefore not explained by which questions go long.
  - Whether it arises before 100k (the state reached) or after (continuation) is open [I].
  - Caveats: the 0.9.1 family pools stock 0.9.1 with modified arms (col511, B0, B1), and seeds 1234-1236 are shared
    across engines.
- **Chars/token by engine:** 0.9.1-family capped replies have a similar median (2.57-2.73), but 9 of 50 reach ≥ 3.0
  (B1: 4 of 9), against 1 of 14 on 0.7.0. Some 0.9.1 capped replies may be of a different kind; without text this
  cannot be checked [I].

## 7. Design implications (options only; the operator and the assessor decide)
At 100k, 8 of 10 eventual 0.7.0 capped replies are not yet repeating. Their outcome depends either on the engine's
continuation (E) or on the state the trajectory has reached (S). A fixed-prefix continuation holds S constant and
varies only the engine and the seed.

**Inputs [M]:**
- per-request decode on long replies, median (slowest): 0.7.0 41 (28) tok/s; stock 0.9.1 67 (47); B1 74 (51);
- KV pool: 0.7.0 2.76M tokens; 0.9.1 and B1 4.71M;
- rep-F finishing from 100k, within +32k / +64k / +100k / to the cap: 0.24 / 0.55 / 0.62 / 0.66.

**Options** (upper bounds, central / pessimistic GPU-hours):

| option | discriminates | GPU-h |
|---|---|---|
| A. 8 prefixes at 100k × 4 seeds to the cap, on 0.7.0, 0.9.1 and B1 | E against S on identical states | 25 / 36 |
| B. as A, horizon +64k (captures 84% of finishing events from 100k) | same | 5.7 / 7.3 |
| B+. 12 prefixes × 4 seeds, +64k | same, more questions (rough SE about 0.07 per engine before clustering) | 7.8 / 10.3 |
| C. B1 only: the same prefixes built by prefill against forced through decode (force-history hook) | whether decode-built state matters beyond content | 3.4 / 4.3 |
| D. one stock-0.9.1 pass with raw logging, then this pipeline | are 0.9.1's capped replies the same phenomenon? Also supplies 0.9.1 prefixes for a 2×2 | about 6 |

**Design notes [I]:**
1. **Prefixes:** use the at-risk prefixes not yet repeating at 100k, stratified by original outcome, one per question
   where possible.
2. **Second position:** onset − 16k for the late-onset capped replies would measure loop-entry hazard.
3. **Prefix KV is rebuilt by prefill:** state accumulated through the decode path in the original run is absent.
   Option C covers this. The hook exists only for the B1 runner.
4. **Seeds:** distinct per continuation, with the prefix as the unit.
5. **No early stop on loop detection:** 2 finished replies finished from inside repetition.
6. **Prefix source:** all prefixes come from 0.7.0, so A/B/C condition on 0.7.0-type states. Only D closes that.

**Most promising per GPU-hour [I]:** B+ (about 8-10 GPU-h), with C (about 4) as the add-on that separates decode-built
state. D is the cheapest way to learn whether 0.9.1's capped replies are the same phenomenon.

## 8. Limits
- **Small n:** 10 capped (3 from q79) and 19 finished at risk. AUCs and medians are descriptive.
- **Positions** come from re-tokenized text (drift ≤ 8 tokens).
- **Self-check counts** use a fixed English phrase list; they count phrases, not intent.
- **Loop classification** relies on a post hoc measure; both readings are reported.

## Files
- `PRESPEC.md`, `out/prespec.sha256`: the frozen definitions.
- `scripts/`: `build_index.py`, `run_trajectory.sh` and `trajectory.py`, `periods.py` (post hoc), `analyse.py`.
- `out/`: `tables.md` (T1-T14), `summary.json`, the per-reply and per-window metrics.
- `SHA256SUMS` covers the scripts and outputs; this file was added after it.
- About 20 minutes of CPU time.
