# tpurtell 0.9.0 with and without the DCP1 tail fix: tool calling

**Question.** Does the DCP1 tail fix change tool-calling results on tpurtell 0.9.0, 3.25bpw weights?

**Grade.** Descriptive; two repeats per build, not a significance test
([`FINDINGS.md`](../../FINDINGS.md#1-runtime-defects-and-their-deterministic-evidence), item 4).

**Configurations.** `3.25bpw · tpurtell 0.9.0 · DFlash2 ×3` vs `3.25bpw · tpurtell 0.9.0 + DCP1 tail fix ≈ 0.9.1 · DFlash2 ×3`.
tool-eval-bench, 88 scenarios, temperature 0, two repeats each (`protocols/tool-eval-bench/v1.md`).

## Result

Points of 176: 157 and 157 without the fix, 159 and 163 with it. TC-80 and TC-88 fail in both repeats without the fix
and pass in both with it.

## Caveats

- Ten other scenarios change status between repeats of the same build, so single scenarios other than these two are
  within run-to-run variation. Two repeats per build; not a significance test.

## Recompute

```bash
python3 tools/verify.py   # recomputes each run's per-repeat points from its rows
python3 investigations/2026-10-glm53-looping/recompute.py
```
