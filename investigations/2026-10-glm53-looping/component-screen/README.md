# Component screen (2026-10-09): preregistration and amendment

Both files are published byte-identical to the originals. SHA-256, recorded when each was written:

| File | SHA-256 | Written |
|---|---|---|
| `PREREGISTRATION.md` | `46fd9761988a153c38e7c312365cc5d83efa52191d2262d6fdff64967e79da92` | 2026-10-09 00:00 PDT, before any data; the first arm started at 00:51 |
| `AMENDMENT-1.md` | `ff4ea97f3ea813e0cf1754dcc861664d7bff2b997cdcbce9aceb76a65403e926` | 2026-10-09, after arms B, EN, EO, NO and B79 had been seen and before arm S1234 ran; it says so |

**Names used in these files.** Arms B, EN, EO, NO, B79 and S1234 are configurations of `3.25bpw · tpurtell 0.9.1 ·
DFlash2 ×3` (with the layout switches named in each config); V79 is `3.25bpw · tpurtell 0.7.0 · DFlash2 ×3`. A "recipe"
is a local name for a pinned configuration; the published configs are in `configs/glm53-flash/`. `empties_probe6.py` is
the local name of `tools/clients/hard_prompt_screen.py` run with `SEED_BASE=5000 CONC=12`; `empties_probe5.py` is the
same client in its v1 setting.

**Background figures in these files.** The plan's background (for example "62/64" and "29/40" failures on question 88,
and "tracks the layout") comes from the earlier fixed-seed screens, whose rates were withdrawn on 2026-10-09. They are
kept here because they are part of what was written before the data; they are not results. The amendment's "in the
2026-10-07/08 bisection" for the 62/64 figure is imprecise: it counts the eight fixed-seed screens of 3.25bpw DCP1
configurations from 2026-09-30 to 2026-10-08.

**Outcomes** (`tools/analyze.py screen-v2`):
- Part 1, question 88: no switch met the screening rule ("no candidate at this size"), and the plan's power assumption
  (failure near 95%) did not hold: the defaults failed 3 of 12.
- Part 2, question 79: 9 of 12 on both arms; the rule's verdict is "unresolved".
- Amendment 1: S1234 failed 11 of 12, against 3 of 12 for arm B; by the amendment's rule (9 or more), the fixed seed
  explains most of the difference.
- Validity gates: every arm had 12 of 12 records and no errors; the container's settings matched each arm; no arm logged
  a preemption. The V79 arm's KV pool was full for part of the run (run notes).

Results and context: [`../README.md`](../README.md).
