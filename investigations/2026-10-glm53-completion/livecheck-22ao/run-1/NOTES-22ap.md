# 22ap notes: run-1 outcome

## Arm C: INCOMPLETE (guard error, start-up), 0.4044 GPU-h, no traffic sent
- Cause: a thread race in the guard's `_snapshot()`. The pinned copy buffers (`S.pins`) are created by whichever
  thread polls first. In arm C the engine thread's `poll("baseline")` (host step 6) came first, inside
  `torch.inference_mode`, so the pins became inference tensors; every later `poll("periodic")` from the poller
  thread failed on `p_.copy_()` ("Inplace update to inference tensor outside InferenceMode"), 4 errors per rank.
- In arm A the poller's periodic poll came first at the same host step (records: periodic 6, then baseline 6), so the
  pins were normal tensors. Not related to port 01.
- The runner's start-up gate saw the guard errors and stopped before any request (client 0/24), as designed.
- Arm A is unaffected: guard errors 0; lrec emitted 24,626 = final n_lrec on both ranks (no duplicate or lost rows
  from the two polls at step 6).

## Arm A: frozen analysis (stock 0.9.1 + guard)
- Instrument ok: plants 4/4 detected in full, attribution 0.9985, async scheduling on, blocked waits 0/6,084.
- Level 1, POSITIVE FINDING: consumed per-token length exceeds p + 1 on 672,155 of 1,506,571 checked rows, never below.
  On verify rows d = L - (p + 1) equals the previous step's rejected-draft count in every joined row
  (0:18,230, 1:6,892, 2:9,750, 3:13,906; 180 rows with no previous step, all d = 0).
- Level 2: `sel_causal` 0, `sel_foreign` 0, `masked_live` 0 (CLEAN WITHIN COVERAGE, 95% upper bound 0.0001 per row,
  N = 48,782). `sel_len` / `nsa_pos` 99,495 (nsa > p + 1 in the dense regime), and the extra in-count columns are
  holes (-1; `sel_hole` 1,827,408, an info counter): no non-causal or foreign entry is consumed.
- Decision-table row: "Discrepancy (level 1) but no consumed non-causal or foreign entry (level 2 clean)": record the
  discrepancy without claiming leakage; identify whether another computational effect remains (whether in-count -1
  columns change the kernel's output, and tail-column compaction). Level 3 was not measured.
- Known D2 on A (recorded, not instrument failure): tail_w_stash mism 1,077,362 (null 1,077,076, foreign 286),
  tail_w_seed mism 9,080.

## Not established (after attempt 1; superseded by the retry below)
- Arm C (port 01) was not measured, so this run does not show that port 01 removes the discrepancy (expected from
  source: L = positions + 1).

## Arm C retry (17:13Z, image 450cbd4d, guard poll-race fix): CLEAN WITHIN COVERAGE at levels 1 and 2
- Instrument ok: guard errors 0 (15 polls per rank, all succeeded), plants 4/4 detected in full, attribution 0.9985,
  async on, blocked waits 0/6,120; client 24/24, no errors; 926 s x 2 GPU = 0.5144 GPU-h.
- Level 1: d = 0 on all 1,510,575 checked rows. On verify rows whose previous step rejected 1-3 drafts (the rows
  where arm A had d equal to that count), d = 0: 6,818 / 10,176 / 13,876 rows.
- Level 2: sel_causal, sel_foreign, masked_live, sel_len and nsa_pos all 0 (95% upper bound 0.0001 per row,
  N = 48,964). sel_hole 1,601,259 (info: holes within the consumed count, present on both arms).
- Known D2 findings: none on C (arm A: tail_w_stash mism 1,077,362; tail_w_seed mism 9,080).
- Every coverage minimum met on C, including all padded sizes and 20/20 transitions.
- Arm A's analysis is identical before and after the retry (result.attempt1.* vs result.*).

## Reading (decision table section 8, row 2)
Stock 0.9.1 passes the attention kernel a per-token length that exceeds p + 1 by exactly the previous step's
rejected-draft count; port 01 removes it in every checked row. On stock, the extra in-count columns are holes, not
future or foreign entries, so no consumed non-causal or foreign entry was observed: record the discrepancy without
claiming leakage. Open: whether in-count holes or tail-column compaction change the kernel's output (level 3, not
measured). Arm C differs from A by port 01 as a whole (D2 tail ring and the causal-length source), so this run does
not separate those two effects.
Session total: 1.5071 of 4.0 GPU-h (A 0.5883, C attempt 1 0.4044, C retry 0.5144).
