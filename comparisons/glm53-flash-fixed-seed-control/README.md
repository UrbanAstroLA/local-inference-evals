# Fixed-seed control: one request seed on every repeat vs distinct seeds

**Question.** Does sending request seed 1234 on every repeat, instead of a distinct seed per repeat, change how often
GPQA Diamond question 88 fails to finish on tpurtell 0.9.1 (3.25bpw), all else identical?

**Grade.** Supported (Fisher p = 0.003): repeats that share one request seed are not a meaningful sample
([`FINDINGS.md`](../../FINDINGS.md#3-non-completion-on-hard-questions), item 15). It is why the earlier screens were
withdrawn as rate measures.

**Configurations.** `3.25bpw · tpurtell 0.9.1 · DFlash2 ×3`, GPQA Diamond doc 88 × 12, 12 concurrent requests, fresh
server, same client and classifier. Only the request seed differs: 5001-5012 (arm B) vs 1234 on every repeat (arm S1234,
added by a dated amendment after the first arms were seen; `investigations/2026-10-glm53-looping/component-screen`).

## Result

| Arm | Request seeds | Failed to finish / 12 |
|---|---|---|
| B | 5001-5012 | 3 |
| S1234 | 1234 on every repeat | 11 |

Fisher p = 0.003. The amendment's reading rule (9 or more failures: the fixed seed explains most of the difference)
applies.

## Caveats

- **Why a shared seed is not a sample.** The engine draws its sampling noise from the request seed. Repeats that share a
  seed still vary, because concurrent batching changes the arithmetic and their texts diverge, but they draw on the same
  sampler noise, so that variation is not a statistically meaningful sample of the configuration.
- **A control, not a measurement.** S1234 is a control for the seed, not a measurement of the configuration. It is why
  the earlier screens, which sent seed 1234 on every repeat, were withdrawn as rate measures (notice:
  [`investigations/2026-10-glm53-looping`](../../investigations/2026-10-glm53-looping/README.md#4-method-note-on-seeds-and-withdrawal-notice)).

## Recompute

```bash
python3 tools/analyze.py screen-v2
python3 tools/verify.py
```
