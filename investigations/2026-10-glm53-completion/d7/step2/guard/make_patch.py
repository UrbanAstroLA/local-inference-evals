#!/usr/bin/env python3
"""Build the mapguard hook copies of three B1 files (PLAN.md amendment 18a, step 2b; copy of the frozen
b0-mapguard/guard/make_patch.py with B1's index-K file and the amended prefill-write hooks).

Inputs: B1's sparse_attn_indexer_kpool.py (= B0 + d7-fix.diff, d7/step2/b1-build/, pinned) and B0's
b12x_mla_sparse.py and scheduler.py (unchanged in B1; sha256-pinned below).
Outputs: ../build/files/vllm/<path> (patched), ../build/mapguard.diff (unified diff).
Every insertion is an observe-only call guarded by _MAPGUARD_ON (False unless
GLM53_MAPGUARD=device), so with the variable unset the files compute exactly as B0.
"""
import difflib
import hashlib
import os
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "<b0-vllm-source>")
HERE = os.path.dirname(os.path.abspath(__file__))
# B1's index-K file comes from the B1 build directory (verified there == B0 file + d7-fix.diff)
SRC_OVERRIDE = {"model_executor/layers/sparse_attn_indexer_kpool.py":
                os.path.join(HERE, "..", "b1-build", "sparse_attn_indexer_kpool.py")}
OUT = os.path.join(HERE, "..", "build", "files", "vllm")

PIN = {
    "model_executor/layers/sparse_attn_indexer_kpool.py":
        "146a7c6b525d56e17e56b5c55d81cb0c19bfe7e1cfe3304a4974c1047246c6e3",   # B1 (B0 9104be37 + d7)
    "v1/attention/backends/mla/b12x_mla_sparse.py":
        "6e74c42cf4385bb0451243098ec8b4a84005675fa776169fddb2fb230bab0157",
    "v1/core/sched/scheduler.py":
        "e405e84eb3674b04bc6251b2736fe0dce8f66a907123c82dd31b4c2ebc73841f",
}

KP_HEADER_ANCHOR = "# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.\nMXFP4_BLOCK_SIZE = 32\n"
KP_HEADER = KP_HEADER_ANCHOR + """
# ---- mapguard (observe-only; inert unless GLM53_MAPGUARD=device) ----
try:
    from vllm import mapguard_hooks as _mapguard
    _mapguard.ensure_worker_install()
except Exception:  # noqa: BLE001
    _mapguard = None
_MAPGUARD_ON = _mapguard is not None and _mapguard.ENABLED
# ---- end mapguard ----
"""

EDITS = {
    "model_executor/layers/sparse_attn_indexer_kpool.py": [
        (KP_HEADER_ANCHOR, KP_HEADER),
        # S2 index-K prefill write (B1: after the D7 ring resolution, so the hook sees what the writer
        # receives; content oracle armed with the writer's inputs)
        ("""                    d7_tail = attn_metadata.get(_resolve_layer_name(tail_prefix))
                _kpool_compress_insert(
""", """                    d7_tail = attn_metadata.get(_resolve_layer_name(tail_prefix))
                if _MAPGUARD_ON and positions is not None:  # mapguard: observe-only
                    _mapguard.hook_idx_prefill_write(
                        k_cache_prefix, slot_mapping[prefill_slice], positions,
                        num_decode_tokens, num_tokens, index_kpool,
                        k=k[prefill_slice], gate=gate_score[prefill_slice], ape=compress_ape,
                        kv_cache=kv_cache, round_scale=(scale_fmt is not None),
                        ring_mode="none" if d7_tail is None else "ring",
                        ring_tslot=(None if d7_tail is None
                                    else d7_tail.slot_mapping[prefill_slice]))
                _kpool_compress_insert(
"""),
        # S2' index-K prefill write, after the writer: content oracle + shadow snapshot (B1 amendment)
        ("""                    tail_slot_mapping=(
                        None if d7_tail is None else d7_tail.slot_mapping[prefill_slice]
                    ),
                )
""", """                    tail_slot_mapping=(
                        None if d7_tail is None else d7_tail.slot_mapping[prefill_slice]
                    ),
                )
                if _MAPGUARD_ON and positions is not None:  # mapguard: observe-only
                    _mapguard.hook_idx_prefill_post(k_cache_prefix, kv_cache)
"""),
        # S4 tail seed
        ("""                        # pos % kpool offsets, so the result is identical.
                        kpool_seed_tail_cache(
""", """                        # pos % kpool offsets, so the result is identical.
                        if _MAPGUARD_ON and positions is not None:  # mapguard
                            _mapguard.hook_tail_seed(
                                k_cache_prefix, tail_meta.slot_mapping[prefill_slice],
                                positions, num_decode_tokens, index_kpool)
                        kpool_seed_tail_cache(
"""),
        # S6 index-K prefill read
        ("""            if not chunk.skip_kv_gather and chunk.local_total_seq_lens > 0:
                ops.cp_gather_indexer_k_quant_cache(
""", """            if not chunk.skip_kv_gather and chunk.local_total_seq_lens > 0:
                if _MAPGUARD_ON and positions is not None:  # mapguard: observe-only
                    _mapguard.hook_idx_prefill_read(k_cache_prefix, chunk, positions)
                ops.cp_gather_indexer_k_quant_cache(
"""),
        # S3 index-K decode write + tail stash
        ("""                # [B, lmax]) — no per-token .contiguous() copies needed.
                kpool_decode_update_and_maybe_write_cache_batched(
""", """                # [B, lmax]) — no per-token .contiguous() copies needed.
                if _MAPGUARD_ON:  # mapguard: observe-only
                    _mapguard.hook_idx_decode_write(
                        k_cache_prefix, dec_slot, dec_pos, dec_tail_slot, positions,
                        num_decode_tokens)
                kpool_decode_update_and_maybe_write_cache_batched(
"""),
        # S5 index-K decode read (B12x paged top-k)
        ("""            _run_b12x_paged_topk(
                q_fp8=padded_q_quant_decode_tokens.reshape(
""", """            if _MAPGUARD_ON and positions is not None:  # mapguard: observe-only
                _mapguard.hook_idx_decode_read(
                    k_cache_prefix, b12x_block_table[:num_rows],
                    b12x_seq_lens[:num_rows], positions, num_rows)
            _run_b12x_paged_topk(
                q_fp8=padded_q_quant_decode_tokens.reshape(
"""),
        # S5' index-K decode read (DeepGEMM fallback)
        ("""        else:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
""", """        else:
            if _MAPGUARD_ON and positions is not None:  # mapguard: observe-only
                _mapguard.hook_idx_decode_read(
                    k_cache_prefix, decode_metadata.block_table, seq_lens, positions,
                    num_padded_tokens)
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
"""),
    ],
    "v1/attention/backends/mla/b12x_mla_sparse.py": [
        ("""logger = init_logger(__name__)
""", """logger = init_logger(__name__)

# ---- mapguard (observe-only; inert unless GLM53_MAPGUARD=device) ----
try:
    from vllm import mapguard_hooks as _mapguard
except Exception:  # noqa: BLE001
    _mapguard = None
_MAPGUARD_ON = _mapguard is not None and _mapguard.ENABLED
# ---- end mapguard ----
"""),
        # S7 MLA read, physical-slot indexer branch
        ("""            if self._indexer_outputs_physical_slots:
                selected_indices = topk_indices
                nsa_cache_seqlens = per_token_cache
""", """            if self._indexer_outputs_physical_slots:
                selected_indices = topk_indices
                nsa_cache_seqlens = per_token_cache
                if _MAPGUARD_ON:  # mapguard: observe-only
                    _mapguard.hook_mla_read(
                        layer, selected_indices, nsa_cache_seqlens, num_actual_toks)
"""),
        # S7 MLA read, logical -> physical branch (GLM-5.3 kpool, DCP1)
        ("""                _compact_dropped_selection(
                    selected_indices, nsa_cache_seqlens
                )
                _mask_page_table_after_nsa_len(
                    selected_indices, nsa_cache_seqlens
                )
""", """                _compact_dropped_selection(
                    selected_indices, nsa_cache_seqlens
                )
                _mask_page_table_after_nsa_len(
                    selected_indices, nsa_cache_seqlens
                )
                if _MAPGUARD_ON:  # mapguard: observe-only
                    _mapguard.hook_mla_read(
                        layer, selected_indices, nsa_cache_seqlens, num_actual_toks)
"""),
    ],
    "v1/core/sched/scheduler.py": [
        ("""        # Return sync affected IDs to skip in update_from_output
        return sync_failed_req_ids
""", """        # Return sync affected IDs to skip in update_from_output
        return sync_failed_req_ids


# ---- mapguard (observe-only; inert unless GLM53_MAPGUARD=device) ----
try:
    from vllm import mapguard_hooks as _mapguard

    _mapguard.install_scheduler_side(Scheduler)
except Exception:  # noqa: BLE001
    pass
# ---- end mapguard ----
"""),
    ],
}

MARKERS = {  # expected count of "mapguard" lines per patched file
    "model_executor/layers/sparse_attn_indexer_kpool.py": 9,
    "v1/attention/backends/mla/b12x_mla_sparse.py": 4,
    "v1/core/sched/scheduler.py": 2,
}


def main():
    diffs = []
    for rel, edits in EDITS.items():
        src = open(SRC_OVERRIDE.get(rel, os.path.join(SRC, rel))).read()
        h = hashlib.sha256(src.encode()).hexdigest()
        if h != PIN[rel]:
            sys.exit(f"{rel}: sha256 {h} != pinned {PIN[rel]}")
        new = src
        for old, rep in edits:
            n = new.count(old)
            if n != 1:
                sys.exit(f"{rel}: anchor found {n} times:\n{old}")
            new = new.replace(old, rep)
        m = new.count("mapguard")
        if m != MARKERS[rel]:
            sys.exit(f"{rel}: {m} markers, expected {MARKERS[rel]}")
        dst = os.path.join(OUT, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "w").write(new)
        diffs.extend(difflib.unified_diff(src.splitlines(True), new.splitlines(True),
                                          f"a/vllm/{rel}", f"b/vllm/{rel}"))
        print(rel, "patched", hashlib.sha256(new.encode()).hexdigest())
    open(os.path.join(HERE, "..", "build", "mapguard.diff"), "w").write("".join(diffs))
    os.makedirs(os.path.join(OUT), exist_ok=True)
    g = open(os.path.join(HERE, "mapguard_hooks.py")).read()
    open(os.path.join(OUT, "mapguard_hooks.py"), "w").write(g)
    print("mapguard_hooks.py", hashlib.sha256(g.encode()).hexdigest())


if __name__ == "__main__":
    main()
