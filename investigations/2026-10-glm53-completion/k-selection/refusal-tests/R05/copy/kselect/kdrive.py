"""K: drive the image's OWN selection code (runs inside the release image; GPU for the B12x kernels).

Nothing is re-implemented on the selection path. Called, from the image:
  vllm.model_executor.layers.sparse_attn_indexer._run_b12x_paged_topk   (eager decode entry the kpool
      indexer calls: plan_indexer_scratch -> plan.bind -> index_topk_fp8; kpool.py b0:928-954)
  vllm.model_executor.layers.sparse_attn_indexer._merge_b12x_dcp_topk    (0.7.0 DCP2: the image's
      triton_convert_dcp_local_topk_to_global, candidate unpack, B12x run_row_topk over 1,024 candidates,
      triton_gather_topk_ids_by_position; kpool.py v070:924-931)
  vllm.model_executor.layers.sparse_attn_indexer_kpool._move_lowest_selected_pool_last   (B1 only)
  vllm.models.glm5next.nvidia.ops.kpool_compress.expand_pools_and_append_tail  (pool_ids[:, :511])
Replaced, and only these:
  current_workspace_manager()  -> torch.empty buffers pre-filled with a poison byte (0x00/0xFF/0x7F per
                                  repeat) so no route can depend on stale or zeroed scratch;
  _b12x_compat_dcp_group / _dcp_all_gather_first_dim_into -> a two-rank stub: rank r's slot is the
                                  call's own packed candidates, the other slot the other rank's candidates
                                  produced by the same image kernel (both perspectives are run and compared).
index_topk_fp8 is wrapped (pass-through) only to read the binding's route / ctas / merge threshold.
"""
from __future__ import annotations

import importlib
import os

import torch

TOPK = 512
CAPTURED = {}


class _WS:
    def __init__(self, device):
        self.device = device
        self.pattern = 0

    def get_simultaneous(self, *specs):
        out = []
        for shape, dtype in specs:
            t = torch.empty(tuple(shape), dtype=dtype, device=self.device)
            if t.numel():
                t.view(-1).view(torch.uint8).fill_(self.pattern)
            out.append(t)
        return out


class _Group:
    def __init__(self, n):
        self.world_size = n
        self.rank_in_group = 0


class Driver:
    def __init__(self, engine_cfg, device):
        self.cfg = engine_cfg
        self.dev = device
        # import order as the engine (and d7/mechanism tests) resolve it: avoids the glm5next
        # package's circular import when the kpool module is imported first
        self.SAI = importlib.import_module("vllm.model_executor.layers.sparse_attn_indexer")
        self.KC = importlib.import_module("vllm.models.glm5next.nvidia.ops.kpool_compress")
        self.KP = importlib.import_module("vllm.model_executor.layers.sparse_attn_indexer_kpool")
        self.PG = importlib.import_module("b12x.attention.dsa_indexer.paged")
        self.ws = _WS(device)
        self.SAI.current_workspace_manager = lambda: self.ws
        self.has_fix = hasattr(self.KP, "_move_lowest_selected_pool_last")
        if engine_cfg["col511_fix"] != self.has_fix:
            raise RuntimeError(f"image col511 fix presence {self.has_fix} != plan {engine_cfg['col511_fix']}")
        self._wrap_index_topk()
        self._cand = None
        self._ag_log = []

    # -------------------------------------------------------------- binding capture
    def _wrap_index_topk(self):
        orig = self.PG.index_topk_fp8

        def wrapped(*a, **kw):
            CAPTURED["binding"] = kw.get("binding")
            return orig(*a, **kw)
        wrapped._k_orig = orig
        self.PG.index_topk_fp8 = wrapped
        pkg = importlib.import_module("b12x.attention.dsa_indexer")
        try:
            porig = getattr(pkg, "index_topk_fp8")
        except Exception:  # noqa
            porig = None
        if porig is not None:
            def pwrapped(*a, **kw):
                CAPTURED["binding"] = kw.get("binding")
                return porig(*a, **kw)
            setattr(pkg, "index_topk_fp8", pwrapped)
        self.pkg_same_function = porig is orig

    def fold_plan(self, rows, width):
        sup = 32768 // 64
        try:
            fp = self.PG._plan_two_level_fold(q_rows=rows, topk=TOPK, page_size=64, page_table_width=width,
                                              supertile_pages=sup, output_physical_slots=False)
            return dict(slices=[list(x) for x in fp.slices] if fp.slices else None,
                        total_slices=int(fp.total_slices or 0), reason=str(fp.reason))
        except Exception as e:  # noqa
            return dict(error=repr(e))

    # -------------------------------------------------------------- selection
    def topk(self, q8, w, cache3d, bt_rows, seqlens_pool, pattern):
        """One _run_b12x_paged_topk call. Returns (ids int32 (R,512), scores f32 (R,512), meta)."""
        rows = q8.shape[0]
        self.ws.pattern = pattern
        out = torch.full((rows, TOPK), -1, dtype=torch.int32, device=self.dev)
        sc = torch.empty((rows, TOPK), dtype=torch.float32, device=self.dev)
        sc.view(torch.uint8).fill_(0xFF)                   # NaN: any slot a route leaves unwritten shows
        CAPTURED.clear()
        self.SAI._run_b12x_paged_topk(
            q_fp8=q8.view(torch.float8_e4m3fn).contiguous(), weights=w.contiguous(), kv_cache=cache3d,
            seq_lens=seqlens_pool.to(torch.int32).contiguous(), block_table=bt_rows.contiguous(),
            schedule_metadata=None, topk_indices=out, topk_tokens=TOPK, topk_scores=sc, active_width=None)
        b = CAPTURED.get("binding")
        s = getattr(b, "scratch", None)
        meta = dict(route=str(getattr(b, "route", getattr(s, "route", "?"))),   # as index_topk_fp8 resolves it
                    ctas_per_group=getattr(s, "fused_ctas_per_group", None),
                    merge_threshold=getattr(s, "fused_merge_threshold", None))
        return out, sc, meta

    # -------------------------------------------------------------- DCP2 merge (0.7.0)
    def _stub_group(self, n=None):
        return _Group(2)

    def _stub_all_gather(self, group, inp, out):
        r = self._me
        rows = inp.shape[0]
        self._ag_log.append(bool(torch.equal(inp, self._cand[r])))
        o = out.view(2, rows, 2, TOPK)
        o[r].copy_(inp)
        o[1 - r].copy_(self._cand[1 - r])

    def merge_dcp2(self, ids_loc, sc_loc, pattern):
        """ids_loc/sc_loc: per-rank kernel outputs (local ids). Returns per-perspective (ids, scores),
        and whether the stub's own-rank slot matched the independently produced candidates."""
        from vllm.v1.attention.backends.mla.b12x_dcp_topk import triton_convert_dcp_local_topk_to_global
        self.ws.pattern = pattern
        rows = ids_loc[0].shape[0]
        cand = []
        for r in range(2):
            i = ids_loc[r].clone()
            s = sc_loc[r].clone()
            triton_convert_dcp_local_topk_to_global(i, s, dcp_world_size=2, dcp_rank=r, cp_kv_cache_interleave_size=1)
            c = torch.empty((rows, 2, TOPK), dtype=torch.int32, device=self.dev)
            c[:, 0, :].copy_(i)
            c[:, 1, :].copy_(s.view(torch.int32))
            cand.append(c)
        self._cand = cand
        self._ag_log = []
        saved = (self.SAI._b12x_compat_dcp_group, self.SAI._dcp_all_gather_first_dim_into)
        self.SAI._b12x_compat_dcp_group = self._stub_group
        self.SAI._dcp_all_gather_first_dim_into = self._stub_all_gather
        res = []
        try:
            for r in range(2):
                self._me = r
                i = ids_loc[r].clone()
                s = sc_loc[r].clone()
                self.SAI._merge_b12x_dcp_topk(topk_indices=i, topk_scores=s, topk_tokens=TOPK,
                                              dcp_world_size=2, dcp_rank=r, cp_kv_cache_interleave_size=1)
                res.append((i, s))
        finally:
            self.SAI._b12x_compat_dcp_group, self.SAI._dcp_all_gather_first_dim_into = saved
        return res, all(self._ag_log) and len(self._ag_log) == 2, cand

    def merge_dcp1(self, ids, sc):
        # the engine calls it on every DCP size; world 1 returns immediately (kept for fidelity)
        self.SAI._merge_b12x_dcp_topk(topk_indices=ids, topk_scores=sc, topk_tokens=TOPK, dcp_world_size=1,
                                      dcp_rank=0, cp_kv_cache_interleave_size=1)

    # -------------------------------------------------------------- consumer
    def col511_fix(self, ids, sc):
        self.KP._move_lowest_selected_pool_last(ids, sc)

    def expand(self, ids, dec_seq):
        pool_ids = ids.to(torch.int64)
        return self.KC.expand_pools_and_append_tail(pool_ids[:, : TOPK - 1], dec_seq.to(torch.int32).contiguous(), 4)


def env_report():
    return {k: v for k, v in os.environ.items() if "B12X" in k or k.startswith("VLLM_") or "TRITON" in k}
