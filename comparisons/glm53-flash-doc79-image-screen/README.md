# Question 79: tpurtell 0.9.1 vs the tpurtell 0.7.0 image at equal draft depth

**Question.** At equal draft depth (DFlash2 ×3), does the tpurtell 0.7.0 image with its default layout fail GPQA Diamond
question 79 less often than tpurtell 0.9.1 at its defaults (3.25bpw)? Preregistered
(`investigations/2026-10-glm53-looping/component-screen`, part 2).

**Grade.** Descriptive: 9 of 12 on both; the preregistered rule's verdict is "unresolved"
([`FINDINGS.md`](../../FINDINGS.md#3-non-completion-on-hard-questions), items 13-14).

**Configurations.** GPQA Diamond doc 79 × 12 repeats per arm, request seeds 5001-5012, 12 concurrent requests, fresh
server (`protocols/hard-prompt-screen/v2.md`).

## Result

| Arm | Configuration | Failed to finish / 12 (95% Wilson) | Loops | Exhaustions |
|---|---|---|---|---|
| B79 | `3.25bpw · tpurtell 0.9.1 · DFlash2 ×3` | 9 (47-91%) | 1 | 8 |
| V79 | `3.25bpw · tpurtell 0.7.0 · DFlash2 ×3` | 9 (47-91%) | 3 | 6 |

Fisher p = 1.00; difference 0 points (95% Newcombe -32.5 to +32.5). The preregistered rule's verdict is "unresolved":
neither of its outcomes (the 0.7.0 image as candidate, or draft depth implicated) was met at this size; draft depth
itself was not varied (both arms ran three draft tokens). Most failures on this question are exhaustion: reasoning that
keeps varying until the 327,680-token budget runs out.

## Caveats

- **What differs.** V79 differs from B79 in image, parallel layout (EP2 + DCP2) and vision. The model revisions differ
  only in files neither server uses (identical weight files; both servers load the same vendored chat template).
- **KV pool.** V79's KV pool (2,894,456 tokens) was full for part of the run, so fewer than 12 requests ran at once then
  (run notes).
- **Power** (`tools/analyze.py screen-power`): from 75% failing, 12 vs 12 draws detect with 80% power only a drop of
  about 60 points (two-sided Fisher, p < 0.05).
- **Held fixed:** 3.25bpw weights, DFlash2 ×3, temperature 1.0 / top_p 0.95, 12 concurrent requests, one question.
- **Not settled here:** tpurtell 0.7.0 ships with five draft tokens, and its GPQA runs used five; this screen ran it at
  three, so it says nothing about 0.7.0 as shipped. Draft depth, quantization and sampling were not varied.
- **Selection.** Question 79 was chosen as hard from seed-1234 data.

## Recompute

```bash
python3 tools/analyze.py screen-v2      # counts, Wilson and Newcombe intervals, Fisher test, the preregistered rule
python3 tools/analyze.py screen-power   # minimum detectable differences
python3 tools/verify.py                 # checks this comparison: same protocol and hardware, only the named fields differ
```
