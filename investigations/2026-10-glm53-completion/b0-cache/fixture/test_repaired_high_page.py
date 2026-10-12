"""Repaired copy of B12x test_glm_next_writer_and_reader_use_int64_for_live_high_page.

The shipped test (tests/attention/test_glm_next_mla.py:1645-1725) calls
sparse_mla.concat_and_cache_glm_next_mla(sources, cache, slots) without the keyword-only
`plan=` that the shipped API requires (sparse_mla/api.py:133-141), so on a GPU with enough
free memory it should fail with TypeError instead of testing anything (inferred from the
signature). This wrapper supplies a writer plan (sparse_mla.plan_cache_writer, as vLLM's
_prepare_glm_next_writer_plan does) and otherwise runs the test body unchanged.
Covers the PADDED page-stride path (stride 37,888 B, page 56,681, > 2^31) which B0 does not
use; B0's contiguous path is covered by run_fixture.py parts D1-D3.
Run from /opt/b12x so that `tests` is importable.
"""
from tests.attention import test_glm_next_mla as T
from b12x.attention import sparse_mla

_orig = sparse_mla.concat_and_cache_glm_next_mla


def _with_plan(kv_c, kv_cache, slot_mapping, *, plan=None):
    if plan is None:
        plan = sparse_mla.plan_cache_writer(kv_c, kv_cache, slot_mapping)
    return _orig(kv_c, kv_cache, slot_mapping, plan=plan)


def test_repaired_glm_next_high_page(monkeypatch):
    monkeypatch.setattr(T.sparse_mla, "concat_and_cache_glm_next_mla", _with_plan)
    T.test_glm_next_writer_and_reader_use_int64_for_live_high_page()
