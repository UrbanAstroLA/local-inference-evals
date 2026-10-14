#!/usr/bin/env python3
"""Insert the local-forcewin hook lines into copies of the installed engine files (exact anchors, each asserted
to occur exactly once; pre-hashes asserted against the known B1 / B1-mapguard files).

  python3 make_patch.py <src_root> <out_root> [--base b1|b1-mapguard]

<src_root>/<rel> are the installed files (vllm/ relative paths), <out_root>/<rel> receive the hooked copies.
Every inserted line carries the marker "local-forcewin"; every call site is behind `if _FW_ON:`, so with
FORCEWIN unset the hooked files execute exactly the original statements plus one module-level import
and one boolean test per site.
"""
import hashlib
import os
import sys

MARK = "local-forcewin"

PRE = {   # installed-file sha256 for each supported base image
    "b1": {
        "v1/worker/gpu/model_runner.py": "101e456b28fbebd0cb73e3f43ae38b4726844a43ee96ec61239bbe455853205f",
        "v1/worker/gpu/spec_decode/rejection_sampler.py":
            "308d4171f892f27231655289ceb03cd4566533c7d26b8f98bd62ef0a34f986f0",
        "v1/worker/gpu/sample/sampler.py": "76070739447fc9f7563ef164ef80c0777208d1ea07dec3a6f6d2a0755188d7ae",
        "v1/attention/backends/mla/indexer.py": "00989c503028216cb349b506ccd7d4dc752bc9a54425a25b901c4905d95f5e0b",
        "model_executor/layers/sparse_attn_indexer_kpool.py":
            "146a7c6b525d56e17e56b5c55d81cb0c19bfe7e1cfe3304a4974c1047246c6e3",
    },
}
# B1-mapguard image (d7/step2/FROZEN.md): identical files except the guard-hooked index-K file
PRE["b1-mapguard"] = dict(PRE["b1"])
PRE["b1-mapguard"]["model_executor/layers/sparse_attn_indexer_kpool.py"] = \
    "30ff8bdea2bfba185bf0bf5b591e7999d451baad12238761681f60d56b55b35f"

IMPORT = (
    "# ---- local-forcewin (inert unless FORCEWIN=on) ----\n"
    "try:\n"
    "    from vllm import forcewin_hooks as _fw\n"
    "except Exception:  # noqa: BLE001  local-forcewin\n"
    "    _fw = None\n"
    "_FW_ON = _fw is not None and _fw.ON  # local-forcewin\n"
    "# ---- end local-forcewin ----\n\n\n"
)


def ins_before(anchor, text):
    return ("before", anchor, text)


def ins_after(anchor, text):
    return ("after", anchor, text)


EDITS = {
    "v1/worker/gpu/model_runner.py": [
        ins_before("class GPUModelRunner", IMPORT),
        ins_after(
            "                self.sampler.add_request(\n"
            "                    req_index, prompt_len, new_req_data.sampling_params\n"
            "                )\n",
            "                if _FW_ON:  # local-forcewin\n"
            "                    _fw.add_request(self, req_id, req_index, prompt_len, new_req_data)\n"),
        ins_after(
            "    def capture_model(self) -> int:\n",
            "        if _FW_ON:  # local-forcewin: persistent tables exist before CUDA-graph capture\n"
            "            _fw.register(self)\n"),
        ins_after(
            "    def _remove_request(self, req_id: str) -> bool:\n",
            "        if _FW_ON:  # local-forcewin: flush before the request's state is freed\n"
            "            _fw.remove_request(self, req_id)\n"),
        ins_before(
            "        return block_tables, slot_mappings\n\n    def prepare_dummy_attn(",
            "        if _FW_ON:  # local-forcewin: arm the step (real batches only)\n"
            "            _fw.on_batch(self, input_batch)\n"),
        ins_before(
            "        if input_batch.num_draft_tokens == 0 or self.rejection_sampler is None:\n",
            "        if _FW_ON:  # local-forcewin: raw metrics BEFORE the sampler decides\n"
            "            _fw.on_logits(self, logits, input_batch)\n"),
        ins_before(
            "        # Prepare the model runner output.\n",
            "        if _FW_ON:  # local-forcewin: prefill reference rows (keyed requests only)\n"
            "            _fw.prefill_ref(self, hidden_states, input_batch)\n\n"),
        ins_after(
            "        self.postprocess_sampled(\n"
            "            input_batch.idx_mapping,\n"
            "            sampler_output.sampled_token_ids,\n"
            "            num_sampled,\n"
            "            num_rejected,\n"
            "            input_batch.query_start_loc,\n"
            "        )\n",
            "        if _FW_ON:  # local-forcewin: invariants, shadow ledger, snapshots\n"
            "            _fw.after_commit(\n"
            "                self, input_batch, sampler_output, num_sampled, num_rejected, attn_metadata\n"
            "            )\n"),
        ins_after(
            "            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens\n",
            "            if _FW_ON:  # local-forcewin: draft-schedule modes (oracle / poison / reject:j)\n"
            "                _fw.after_propose(self, input_batch)\n"),
    ],
    "v1/worker/gpu/spec_decode/rejection_sampler.py": [
        ins_before("class RejectionSampler:", IMPORT),
        ins_before(
            "        sampled, num_sampled = rejection_sample(\n"
            "            processed_logits,\n",
            "        if _FW_ON:  # local-forcewin: post-top_p probabilities BEFORE the rejection draw\n"
            "            _fw.on_processed(processed_logits)\n"),
        ins_before(
            "            # draft_logits uses persistent request-state indices and stays global.\n",
            "            if _FW_ON:  # local-forcewin\n"
            "                _fw.set_chunk(lo)\n"),
        ins_before(
            "        num_sampled, num_rejected = get_num_sampled_and_rejected(\n"
            "            num_sampled,\n"
            "            input_batch.seq_lens,\n",
            "        if _FW_ON:  # local-forcewin: commit history tokens for forced requests\n"
            "            sampled, num_sampled = _fw.override(\n"
            "                sampled, num_sampled, input_batch, draft_sampled, pos\n"
            "            )\n"),
    ],
    "v1/worker/gpu/sample/sampler.py": [
        ins_before("class Sampler:", IMPORT),
        ins_after(
            "            processed_logits = apply_top_k_top_p(processed_logits, top_k, top_p)\n",
            "            if _FW_ON:  # local-forcewin: post-top_p probabilities BEFORE the draw\n"
            "                _fw.on_processed(processed_logits, single=True)\n"),
        ins_after(
            "        sampled, processed_logits = self.sample(\n"
            "            logits,\n"
            "            expanded_idx_mapping,\n"
            "            idx_mapping,\n"
            "            idx_mapping_np,\n"
            "            pos,\n"
            "            input_ids,\n"
            "            expanded_local_pos,\n"
            "            return_logprobs=return_logprobs,\n"
            "        )\n",
            "        if _FW_ON:  # local-forcewin: commit the history token for forced requests\n"
            "            sampled = _fw.override_single(sampled, input_batch)\n"),
    ],
    "v1/attention/backends/mla/indexer.py": [
        ins_before("def compute_kpool_tail_slot_mapping(", IMPORT),
        ins_before(
            "            common_attn_metadata.slot_mapping.copy_(slot_mapping)\n"
            "            slot_mapping = common_attn_metadata.slot_mapping\n",
            "            if _FW_ON:  # local-forcewin: decode-window-only D2 injection (keyed requests)\n"
            "                slot_mapping = _fw.d2_tail_mapping(\n"
            "                    common_attn_metadata.slot_mapping,\n"
            "                    slot_mapping,\n"
            "                    common_attn_metadata.query_start_loc,\n"
            "                    positions,\n"
            "                    common_attn_metadata.num_actual_tokens,\n"
            "                    common_attn_metadata.num_reqs,\n"
            "                )\n"
            "                _fw.note_tail_mapping(\n"
            "                    slot_mapping,\n"
            "                    positions,\n"
            "                    common_attn_metadata.query_start_loc,\n"
            "                    common_attn_metadata.num_actual_tokens,\n"
            "                    common_attn_metadata.num_reqs,\n"
            "                )\n"),
    ],
    "model_executor/layers/sparse_attn_indexer_kpool.py": [
        ins_before("def _move_lowest_selected_pool_last(", IMPORT),
        ins_before(
            "            out = expand_pools_and_append_tail(\n"
            "                pool_ids[:, : select_k - 1], dec_seq, index_kpool\n"
            "            )\n",
            "            if _FW_ON:  # local-forcewin: window exposure of the final selection (graph-safe)\n"
            "                _fw.on_selection(\n"
            "                    k_cache_prefix, pool_ids[:, : select_k - 1], dec_seq, batch_size, next_n\n"
            "                )\n"),
    ],
}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def apply(src_root, out_root, base="b1"):
    report = {}
    for rel, edits in EDITS.items():
        p = os.path.join(src_root, rel)
        raw = open(p, "rb").read()
        h = sha(raw)
        if h != PRE[base][rel]:
            raise SystemExit("pre-hash mismatch %s: %s != %s" % (rel, h, PRE[base][rel]))
        s = raw.decode()
        if MARK in s:
            raise SystemExit("already hooked: " + rel)
        for kind, anchor, text in edits:
            n = s.count(anchor)
            if n != 1:
                raise SystemExit("anchor count %d != 1 in %s: %r" % (n, rel, anchor[:80]))
            s = s.replace(anchor, text + anchor if kind == "before" else anchor + text)
        q = os.path.join(out_root, rel)
        os.makedirs(os.path.dirname(q), exist_ok=True)
        open(q, "w").write(s)
        added = [ln for ln in s.splitlines() if ln not in raw.decode().splitlines()]
        report[rel] = {"pre": h, "post": sha(s.encode()), "edits": len(edits),
                       "marker_lines": sum(1 for ln in s.splitlines() if MARK in ln),
                       "added_lines": len(added)}
    return report


if __name__ == "__main__":
    import json
    base = "b1"
    if "--base" in sys.argv:
        base = sys.argv[sys.argv.index("--base") + 1]
    print(json.dumps(apply(sys.argv[1], sys.argv[2], base), indent=1))
