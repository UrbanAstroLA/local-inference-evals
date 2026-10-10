# Decode vs prefill: the same configuration twice (noise floor)

**Question.** How much does the decode-vs-prefill measurement vary between two runs of the same configuration
(tpurtell 0.9.1, speculation and prefix caching off, same prompts and seeds)? The noise floor for every
decode-vs-prefill comparison here.

**Grade.** That the engine is not bitwise reproducible one request at a time is supported
([`FINDINGS.md`](../../FINDINGS.md#4-serving), item 19); the spread is the noise floor used in item 3.

**Configurations.** `3.25bpw · tpurtell 0.9.1 · no speculation, prefix cache off`, run on 2026-10-08 and again on
2026-10-09, each on a fresh server, one request at a time, GPQA prompts 0-11 with the same request seeds (1234 + doc id),
2,600 generated tokens each (`protocols/decode-prefill-consistency/v1.md`).

## Result

| Positions | Mean KL, run 1 | Mean KL, run 2 | Prompts 0-5 only: run 1, run 2 |
|---|---|---|---|
| i < 2,044 | 0.0072 | 0.0077 | 0.0077, 0.0060 |
| i ≥ 2,048 | 0.0053 | 0.0098 | 0.0064, 0.0107 |

- The run means differ by up to about 2x (from 2,048 tokens), and single prompts by up to 4.2x (below 2,044) and 5.7x
  (from 2,048) between the two runs. A difference between two single runs smaller than that is not evidence of an effect.
- **Single-request nondeterminism.** With the same prompts and seeds, one request at a time, the two runs' per-position
  values (log-probability of the sampled token, KL) first differ after 1 to 130 generated tokens, depending on the
  prompt; the sampled continuations then differ. Batching cannot explain this (there was none).

## Caveats

- Use: the tail-fix comparison ([`glm53-flash-v090-tailfix-decode-prefill`](../glm53-flash-v090-tailfix-decode-prefill))
  has one run per build; this pair is the spread such single runs carry.

## Recompute

```bash
python3 tools/analyze.py decode-prefill
```
