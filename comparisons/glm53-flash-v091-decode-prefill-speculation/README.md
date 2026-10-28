# Decode vs prefill on tpurtell 0.9.1: speculation off (two runs) and on (one run)

**Question.** On tpurtell 0.9.1, does decoding through DFlash2 ×3's draft-and-verify path agree with prefill re-scoring
as well as plain decoding does?

**Grade.** Descriptive only ([`FINDINGS.md`](../../FINDINGS.md#1-engine-issues-found-during-evaluation-and-their-fixes),
item 3).

**Configurations.** `3.25bpw · tpurtell 0.9.1 · no speculation, prefix cache off` (two runs) vs
`3.25bpw · tpurtell 0.9.1 · DFlash2 ×3, prefix cache off` (one run). Same 12 GPQA prompts and request seeds, one request
at a time, fresh server per run. With speculation on, generated tokens pass through the draft-and-verify path, and their
decode distributions are compared with prefill re-scoring of the same tokens.

This KL is the engine's agreement with itself: one model and one set of weights, its decode distribution at each
position compared with prefill re-reading the same tokens, over the top-20 tokens both return (each renormalised). It
is not a KL between a quantized model and its BF16 base, and it is not comparable with such numbers.

## Result

| Positions | Speculation off, run 1 | Speculation off, run 2 | Speculation on |
|---|---|---|---|
| i < 2,044 | 0.0072 | 0.0077 | 0.0105 |
| i ≥ 2,048 | 0.0053 | 0.0098 | 0.0115 |

Speculation on sits somewhat above both speculation-off runs, by less than the spread between single prompts across
repeated runs ([`glm53-flash-v091-decode-prefill-repeat`](../glm53-flash-v091-decode-prefill-repeat)).

### Tails

Per-position KL and |dlp| (absolute difference in the sampled token's log-probability), all 12 prompts. p99 is the
value 1% of positions exceed (type-7 quantiles).

| Positions | Run | Positions (KL defined) | KL p99 | KL p99.9 | \|dlp\| p99 | \|dlp\| p99.9 |
|---|---|---|---|---|---|---|
| i < 2,044 | speculation off, run 1 | 21,923 | 0.081 | 0.547 | 0.501 | 1.434 |
|  | speculation off, run 2 | 21,923 | 0.096 | 0.415 | 0.526 | 1.437 |
|  | speculation on | 21,923 | 0.141 | 0.875 | 0.650 | 2.107 |
| i ≥ 2,048 | speculation off, run 1 | 9,229 | 0.070 | 0.290 | 0.430 | 1.054 |
|  | speculation off, run 2 | 9,229 | 0.144 | 0.756 | 0.642 | 2.021 |
|  | speculation on | 9,229 | 0.130 | 1.008 | 0.597 | 2.129 |

## Caveats

- One run with speculation on does not separate a difference of this size from run-to-run variation.

## Recompute

```bash
python3 tools/analyze.py decode-prefill
python3 tools/analyze.py decode-prefill-tails
```
