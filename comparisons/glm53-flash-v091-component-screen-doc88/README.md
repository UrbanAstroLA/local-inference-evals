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

"No candidate at this size" is the rule's wording: the screen had about 50% power for a drop from 95% to 72% failing,
and the observed rates are far lower than that assumption. It is not evidence that a switch has no effect; differences
of 20 points either way are inside the intervals. Each main effect is aliased with the other two switches'
interaction (resolution III). Recompute with `tools/analyze.py screen-v2`.

Server facts from the logs (run notes): KV pool 4,707,515 tokens at the defaults, 3,992,056 with EP2 and NOPE records
off, 2,215,158 with EP2 and ownership tp, 1,851,617 with NOPE records off and ownership tp. No preemption in any arm.
