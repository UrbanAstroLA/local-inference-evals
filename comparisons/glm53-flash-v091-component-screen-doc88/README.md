# Component screen, question 88: three layout switches on tpurtell 0.9.1

Preregistered (`investigations/2026-10-glm53-looping/component-screen`). Half-fraction factorial over three switches,
3.25bpw weights, DFlash2 ×3, GPQA Diamond doc 88 × 12 repeats per arm with request seeds 5001-5012, 12 concurrent
requests, a fresh server per arm (`protocols/hard-prompt-screen/v2.md`).

| Arm | Configuration | Failed to finish / 12 (95% Wilson) |
|---|---|---|
| B | `3.25bpw · tpurtell 0.9.1 · DFlash2 ×3` (defaults) | 3 (9-53%) |
| EN | `3.25bpw · tpurtell 0.9.1 · EP2 experts, NOPE records off · DFlash2 ×3` | 2 (5-45%) |
| EO | `3.25bpw · tpurtell 0.9.1 · EP2 experts, MLA owners tp · DFlash2 ×3, sharing off` | 1 (1-35%) |
| NO | `3.25bpw · tpurtell 0.9.1 · NOPE records off, MLA owners tp · DFlash2 ×3, sharing off` | 2 (5-45%) |

Every failure was a loop stopped by the early-stop detector. Each switch on (two arms, 24 requests) vs off (24):

| Switch | On | Off | On - off (95% Newcombe) | Fisher p | Preregistered rule |
|---|---|---|---|---|---|
| EP2 routed experts | 3/24 | 5/24 | -8.3 points (-29.6 to +13.5) | 0.70 | no candidate at this size |
| NOPE records off | 4/24 | 4/24 | 0.0 points (-21.6 to +21.6) | 1.00 | no candidate at this size |
| MLA ownership tp (+ slot sharing off) | 3/24 | 5/24 | -8.3 points (-29.6 to +13.5) | 0.70 | no candidate at this size |

"No candidate at this size" is the rule's wording, not evidence that a switch has no effect. The plan assumed about 95%
failing at the defaults; the defaults failed 3 of 12, so the screen could only see increases.

**Power** (`tools/analyze.py screen-power`; two-sided Fisher test, p < 0.05, 80% power): one arm against another
(12 vs 12, from 25% failing) detects only a rise of about 60 points, and no drop at any size; a switch on vs off
(24 vs 24, from 17%) detects only a rise of about 41 points, and no drop at any size. Only very large effects could show.

**Held fixed** in every arm: 3.25bpw weights, DFlash2 ×3, temperature 1.0 / top_p 0.95, the 327,680-token budget,
12 concurrent requests, one question per arm, a fresh server. Arms EO and NO also turned draft-slot sharing off (the
launcher pairs it with MLA ownership `tp`), so ownership and sharing are not separated. Each main effect is aliased
with the other two switches' interaction (resolution III). Draft depth, quantization and sampling were not varied.

**Selection.** Question 88 was chosen because it nearly always failed in the earlier screens, which sent request seed
1234 on every repeat; with distinct seeds it fails in 1-3 of 12 draws per arm. Recompute with `tools/analyze.py screen-v2`.

Server facts from the logs (run notes): KV pool 4,707,515 tokens at the defaults, 3,992,056 with EP2 and NOPE records
off, 2,215,158 with EP2 and ownership tp, 1,851,617 with NOPE records off and ownership tp. No preemption in any arm.
