# kpool screen: arms A, B and C (2026-10-04, one session)

- A: `3.25bpw · tpurtell 0.8.0 · DFlash2 ×3` (control)
- B: `3.25bpw · tpurtell 0.8.0 · DFlash2 ×1` (the rejected-draft bug cannot fire)
- C: `3.25bpw · tpurtell 0.8.0 + kpool fixes ≈ 0.9.0 · DFlash2 ×3`

Preregistered rules: B "supported" iff A >= 2*B and A - B >= 4; C "fix helps" iff A >= 2*C and A - C >= 4.
Result: A 22/40, B 21/40, C 20/40 failed (loop or exhaustion). Neither rule met. See `investigations/2026-10-glm53-kpool-tail`.
