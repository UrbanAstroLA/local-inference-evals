# tpurtell 0.9.0 with and without the DCP1 tail fix: tool calling

tool-eval-bench, 88 scenarios, temperature 0, two repeats each (`protocols/tool-eval-bench/v1.md`).
Points of 176: 157 and 157 without the fix, 159 and 163 with it. TC-80 and TC-88 fail in both repeats without the fix
and pass in both with it. Ten other scenarios change status between repeats of the same build, so single scenarios
other than these two are within run-to-run variation. Two repeats per build; not a significance test.
