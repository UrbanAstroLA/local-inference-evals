# Amendment 1 (2026-10-07 13:35 PDT, before any arm-D data; A1 complete, its data unchanged)

Arm D (EP2 + DCP2 on v0.9.0) failed to start: the first forward pass crashes in the sparse indexer's DCP top-k
merge, `_merge_b12x_kpool_dcp_topk_by_owner` -> b12x/comm/pcie/_owner_preparation.py:
`TypeError: FrozenMapping.__init__() got an unexpected keyword argument 'dtype'` (log: D1-server-load-fail-1.log).
The direct PCIe owner exchange (`VLLM_B12X_DCP_TOPK_OWNER_EXCHANGE=1`, v0.7.0's and v0.9.0's default) no longer matches
the B12x fork that v0.8.0+ ships, so DCP2 cannot run on v0.9.0 as released.

Change for arm D only: `VLLM_B12X_DCP_TOPK_OWNER_EXCHANGE=0`. The indexer then falls back to `_merge_dcp_topk_global`
(gather every rank's candidates, then the same stable top-k with lowest-index ties). This changes the transport of the
top-k merge, not the selection rule. It is a deviation from v0.7.0's exact configuration and is reported as such.
Arm A is unaffected (DCP1 does not use this path). The decision rule is unchanged.
Runner change: a screen whose server fails to load now stops the run (it previously moved on to the next screen; an A2
attempt was started and stopped before any request, nothing recorded). Remaining order: D1, A2, D2.
