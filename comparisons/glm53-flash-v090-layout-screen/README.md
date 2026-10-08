# tpurtell 0.9.0: default layout vs v0.7.0's layout as a diagnostic control (hard-question screen)

- `3.25bpw · tpurtell 0.9.0 · DFlash2 ×3`: 0.9.0 as released (DCP1 + MLA layer ownership, TP2 experts).
- `3.25bpw · tpurtell 0.9.0 · 0.7.0 layout (DCP2, EP2) · DFlash2 ×3`: the same image with v0.7.0's parallel layout. It
  is a diagnostic control to separate parts of the 0.7.0 → 0.8.0 change, not a recommended configuration (see the
  config notes for the one deviation needed to start it).

Preregistered (`investigations/2026-10-glm53-looping`); two screens per configuration, interleaved, each on a fresh
server. Failures (loop or exhaustion): 41/80 (95% Wilson 40.5-61.9%) with the default layout vs 29/80 (26.6-47.2%)
with the control layout. Fisher exact p = 0.079. Preregistered verdict: **INCONCLUSIVE**; the rule asks for more
repeats rather than reinterpretation. Recompute with
`tools/analyze.py screen-pool k3.25-v0.9.0-dflash3 k3.25-v0.9.0-ep2dcp2-dflash3`.
