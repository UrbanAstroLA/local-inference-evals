# Long-reply completion (amendments 22q-22ah)

The question: what drives completion of GPQA replies that run past 100,000 generated tokens?

On the old records, P(finish | still generating at 100k) was 0.71 for stock 0.7.0 and 0.40 for stock 0.9.1. The
fresh-seed replication of stock 0.7.0 gave 0.66. All of these are descriptive, with about 10-19 finishing events per
record.

**Non-completion is not the same as looping. Both are measured here, not assumed.**

## 1. Discovery (zero GPU; `discovery/`)
Reasoning text from the stock 0.7.0 replication, with definitions hashed before any metric was computed (`PRESPEC.md`):

- **Capped replies:** 8 of 10 end in an exact loop (period 2-1,001 tokens). Four are 2-token loops: an interjection
  plus a separator. 2 replies do not loop.
- **Onsets:** only 2 were already looping at 100k; the rest began 43k-188k tokens later. No capped reply recovered from
  a terminal loop.
- **Finished replies:** 15 of 19 produce new text to the end of their reasoning.
- **Early signatures:** capped replies had lower characters per token in their first 100k tokens (2.31 against 3.15)
  and more interjections, before any repetition.
- **Geometry:** onset positions do not cluster on the attention block, prefill chunk or powers of two.
- **Question-matched:** on the 14 questions long on both engine families, the 0.7.0 family finished 23 / 36 long
  replies and the 0.9.1 family 18 / 58. Which questions go long does not explain the gap.

## 2. B+: fixed 0.7.0-built prefixes, continued on three engines (`bplus/`; 13.06 GPU-hours)
- **Design:** 12 prefixes cut at 100k from 0.7.0 replies, 4 seeds each, +64k tokens on stock 0.7.0, stock 0.9.1 and B1.
- **Validity:** the frozen analysis was INVALID on an over-strict KV-pool gate (a size the 0.7.0 record had itself
  produced). The gate was amended before any outcome was read (22v), and the result is reported as read after that
  disclosed amendment.
- **Reading: R-I, inconclusive.**

| engine | P(finish within +64k) |
|---|---:|
| stock 0.7.0 | 0.417 |
| stock 0.9.1 | 0.417 |
| B1 | 0.521 |

- **0.7.0 − 0.9.1:** 0.000 (95% −0.083 to +0.083).
- **Prefix share of outcome variance:** 0.67.
- **Why not "state-dominated":** the B1 contrasts missed the |mean| ≤ 0.10 rule by 0.004.
- **What it shows:** from the same 0.7.0-built state, 0.7.0 and 0.9.1 continue alike.
- **Limits:**
  - one 0.7.0-built state per question;
  - prefixes rebuilt by prefill, so any decode-built state is absent;
  - re-tokenized ids;
  - "engine" bundles decode path, defect exposure, layout and speculation depth.

## 3. Option D: one stock 0.9.1 pass with full text (`d091/`; 5.88 GPU-hours)
- **The pass:** valid. 6 of 198 empty, inside the record's range; stated accuracy 88.4%.
- **P(finish | 100k):** 4 / 10 (0.40), the same as the 0.9.1 record.
- **Capped replies:** all 6 end in short-period exact loops, the same kind as 0.7.0's.
- **Early signatures:**
  - the low characters-per-token signature is weak on 0.9.1 (2.97 against 3.27);
  - the interjection signature holds (capped 9.4 against finished 2.7 per 1k tokens).
- **Prefixes:** 8 0.9.1-built prefixes exported for a cross-continuation (option E).
- **Limits:** descriptive only. One pass, 10 replies at risk, seeds different from every comparator.

## 4. Option D2: a second stock 0.9.1 pass with full text (`d091b/`; 7.16 GPU-hours)
- **The pass:** valid. 8 of 198 empty (D: 6; record: 5-6 per pass); stated accuracy 85.4%.
- **P(finish | 100k):** 4 / 12 (0.33).
- **Capped replies:** 7 of 8 end in exact loops; none was repeating at 100k.
- **First 100k tokens, capped against finished:**
  - characters per token 2.68 against 3.37;
  - digit share 0.072 against 0.030;
  - interjections 11.6 against 2.3 per 1k.

  The digit-heavy early signature appears here, unlike in D.
- **Prefixes:** D2 adds 5 prefixes on new questions.

## 5. Option E: 0.9.1-built prefixes on both engines (`e-cross/`; frozen, not run)
- **Design:** 13 0.9.1-built prefixes (8 from D, 5 from D2; 8 originally capped, 5 finished), 4 seeds each, +64k
  tokens on stock 0.7.0 and stock 0.9.1, under B+'s design and screens.
- **KV pool:** reported with a floor, not gated.
- **Power:** P(engine effect read) 0.75-0.88 at the full discovery gap, 0.21-0.27 at half; false positive ≤ 0.013 per
  direction.
- **Cost:** central 10.2, ceiling 19.9 GPU-hours.
- **Matched questions:** eight questions overlap with B+'s prefixes, so the same question is seen with a 0.7.0-built
  and a 0.9.1-built prefix.

## Reading so far
- Once in trouble, both engines loop the same way.
- From the same 0.7.0-built state, they continue alike.
- The early signatures are not clearly engine-specific. The interjection signature appears on both engines. The
  low characters-per-token, digit-heavy signature appears on 0.7.0 and in D2, but only weakly in D.

This points to the difference arising before 100k (how often trajectories reach loop-prone states), not in
continuation from a given state.
It is not established: B+ used 0.7.0-built states only. Option E (frozen) continues 0.9.1-built prefixes on both
engines to complete the cross.

**Privacy.** Reply text, prefixes and token ids stay local (`PRIVATE/`, `raw/`) and are never published. The published
files hold ids, hashes, counts and metrics only.
