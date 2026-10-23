# Amendment 22ap: arm C pre-flight and continuation

Written 2026-10-23, after arm A ran and arm C was refused by the pre-flight, and **before any arm A outcome
(level 1/2 results, coverage counts, result.json, result.md) was read**. Only instrument-health lines from the run log
were seen: the guard was alive, the four plants were detected, the client served 24/24.

## What happened
Arm A ran to completion (0.5883 GPU-h, rc 0). Arm C's pre-flight then read GPU0 at 53% utilization with 20 MiB used,
seconds after arm A's container stopped, and refused (run-1/arm-c/refused.txt). The frozen pre-flight takes one sample.

## What changed (runner/run_22ao.sh only)
1. **Pre-flight utilization is re-sampled**: up to 12 samples, 5 s apart; it passes on the first sample with every
   GPU at <= 5%. Memory limits (1024 MiB, GPU1 4096 MiB), the container check and the GPU-client check are unchanged;
   a memory breach still aborts at once.
2. **ARMS=c continuation**: run-1 may exist when it holds arm A and no arm C data; exactly the existing arm-A cost
   line may exist and no arm-C line. Only run-1/arm-c is written. The refused pre-flight record is kept as
   run-1/arm-c.refused-preflight. Its budget and script-hash records go to budget-armc.txt and scripts-armc.sha256,
   so arm A's copies stay as written.
3. Unchanged: images and pins, client, seeds, panel, guard, analysis, outcome rules, wall cap, and the budget gate
   (APPROVED_GPU_H=2.0 total across both arms, so arm C has 1.41 GPU-h left).

## Hashes
SHA256SUMS.frozen-22ao is the frozen list. SHA256SUMS differs only in runner/run_22ao.sh:
- frozen  49f6448270ca74cc5f02b46548662cdd726ee2016a2cb5eb70ea5503c7375458
- amended f25c0b4b166f1615d7a661d4a87bf19803c968ccd10131dc45b776ac4f0dec8b
runner/run_22ao.sh.frozen-22ao is the frozen runner.

## Effect on inference
Arm A and arm C run 1-2 hours apart instead of back to back. Both arms use the same frozen images, client and seeds,
on an otherwise idle machine. The analysis compares arms by the frozen rules; the gap is disclosed here.

## Arm C attempt 1 and the approved retry
Attempt 1 of arm C (16:41Z, 0.4044 GPU-h) stopped at the start-up gate with 8 guard errors and sent no traffic. Cause:
a thread race in the guard. Its host copy buffers were allocated by whichever thread polled first; in attempt 1 the
engine thread's baseline poll (inside torch.inference_mode) came first, so the buffers were inference tensors and every
later poll from the poller thread failed on copy_. In arm A the poller came first. Arm A's records are complete
(guard errors 0; every length record emitted exactly once on both ranks).

The arm A results had been read when the retry was approved. The retry changes no rule, workload or analysis.

The operator approved (verbatim, AUTHORIZATION.txt) a guard fix, a rebuild of image C and a retry of arm C, with
2.0 GPU-h added (ceiling 4.0 total).
- **Guard fix** (in guard/make_guard_22ao.py, regenerated): buffers are allocated with inference mode off
  (`_alloc_pins`), and polls are serialized by a lock with a 60 s bounded wait. Guard b41e8581147c1c5c988d9bcca69c9ceb2f0cbd400d11f086076a1f4f8a3b2adb.
  CPU test tests/test_22ap_poll_race.py: the old allocation reproduces the failure under arm C's order; the fix gives
  no errors; 16 concurrent polls emit every length record exactly once.
- **Image C rebuilt** from build/Dockerfile.C-22ap (Dockerfile.C with only the guard file and its hash changed):
  sha256:450cbd4dd034cfde5eaef0402d8810f40d0bb82455e77b6d0af105c12d9de416. Of 2,435 vLLM .py files, only
  mapguard_hooks.py differs from the attempt-1 image (kept as local/glm53-v091-22ao-c:frozen-22ao). Image A is
  unchanged and keeps the original guard, so the two arms' guards differ by this fix only.
- **Runner** (RETRY_C=1 with ARMS=c): accepts exactly one earlier arm-C cost line with rc 8 and an attempt with no
  client records; keeps attempt 1 as run-1/arm-c.attempt1-guard-race and its analysis output as *.attempt1.*;
  MAX_GH 4.0. Runner 8f8b9d6f3bf1ce681354fd4bbc89f23d634634ab75fdadb5d08060fa1ba09f3f. The runner used for attempt 1 is runner/run_22ao.sh.22ap-attempt1, and its hash
  list is SHA256SUMS.22ap-attempt1.
