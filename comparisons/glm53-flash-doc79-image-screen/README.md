# Question 79: tpurtell 0.9.1 vs the tpurtell 0.7.0 image at equal draft depth

Preregistered (`investigations/2026-10-glm53-looping/component-screen`, part 2). GPQA Diamond doc 79 × 12 repeats per
arm, request seeds 5001-5012, 12 concurrent requests, fresh server (`protocols/hard-prompt-screen/v2.md`).

| Arm | Configuration | Failed to finish / 12 (95% Wilson) | Loops | Exhaustions |
|---|---|---|---|---|
| B79 | `3.25bpw · tpurtell 0.9.1 · DFlash2 ×3` | 9 (47-91%) | 1 | 8 |
| V79 | `3.25bpw · tpurtell 0.7.0 · DFlash2 ×3` | 9 (47-91%) | 3 | 6 |

Fisher p = 1.00; difference 0 points (95% Newcombe -32.5 to +32.5). The preregistered rule's verdict is "unresolved":
neither the 0.7.0 image (with its layout) nor draft depth is implicated at this size. V79 differs from B79 in image,
parallel layout (EP2 + DCP2), vision and model revision (identical weight shards; only the chat template differs).
Its KV pool (2,894,456 tokens) was full for part of the run, so fewer than 12 requests ran at once then (run notes).
Most failures on this question are exhaustion: reasoning that keeps varying until the 327,680-token budget runs out.

**Power** (`tools/analyze.py screen-power`): from 75% failing, 12 vs 12 draws detect with 80% power only a drop of about
60 points (two-sided Fisher, p < 0.05). **Held fixed:** 3.25bpw weights, DFlash2 ×3, temperature 1.0 / top_p 0.95,
12 concurrent requests, one question. **Not settled here:** tpurtell 0.7.0 ships with five draft tokens, and its GPQA
runs used five; this screen ran it at three, so it says nothing about 0.7.0 as shipped. Draft depth, quantization and
sampling were not varied. Question 79 was chosen as hard from seed-1234 data. Recompute with `tools/analyze.py screen-v2`.
