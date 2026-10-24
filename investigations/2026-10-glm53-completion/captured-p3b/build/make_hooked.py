#!/usr/bin/env python3
"""Generate the two hooked vLLM files from the stock 0.9.1 copies by exact-anchor INSERTIONS only.

usage: make_hooked.py [--check]      (host python3; no torch needed)

Inputs:  build/stock/<path>   (copied out of the stock image a9e985c0 with `docker run --rm --network none`;
                               their sha256 are pinned below and re-checked inside the image by the Dockerfile)
Outputs: build/files/vllm/<path>
--check: regenerate in memory and require the files on disk to be byte-identical, then run the additive gate.

The additive gate (also used by tests and the Dockerfile's host-side twin) requires, per file:
  * difflib finds only insertions (no deleted or changed stock line);
  * every inserted line carries the marker "local-p3b";
  * the number of inserted lines equals the expected count.
"""
import difflib
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MARK = "local-p3b"
EXL3 = "model_executor/layers/quantization/exl3.py"
MLA = "v1/attention/backends/mla/b12x_mla_sparse.py"
STOCK_SHA = {
    EXL3: "3b4ca47cb1168bffebcfe3cf1db48daf86691c140cbf40f9c1f3aa6ad6a444d9",
    MLA: "6e74c42cf4385bb0451243098ec8b4a84005675fa776169fddb2fb230bab0157",
}

IMPORT_EXL3 = """# ---- local-p3b (observe-only capture; inert unless GLM53_P3B=capture) ----
try:  # local-p3b
    from vllm import p3b_capture_hooks as _p3b  # local-p3b
except Exception:  # noqa: BLE001  local-p3b
    _p3b = None  # local-p3b
_P3B_ON = _p3b is not None and _p3b.ENABLED  # local-p3b
# ---- end local-p3b ----
"""
IMPORT_MLA = """# ---- local-p3b (observe-only capture; inert unless GLM53_P3B=capture) ----
try:  # local-p3b
    from vllm import p3b_capture_hooks as _p3b  # local-p3b
    _p3b.ensure_worker_install()  # local-p3b
except Exception:  # noqa: BLE001  local-p3b
    _p3b = None  # local-p3b
_P3B_ON = _p3b is not None and _p3b.ENABLED  # local-p3b
# ---- end local-p3b ----
"""

# (file, anchor line (exact, full line), where: "before" | "after", text, occurrences expected of the anchor)
EDITS = [
    (EXL3, "logger = init_logger(__name__)", "after", "\n" + IMPORT_EXL3, 1),
    (EXL3, '            output = runtime["api"].run(binding=binding)', "before",
     "            if _P3B_ON:  # local-p3b\n"
     '                _p3b.hook_moe_bind(layer, "decode", m, binding, runtime)  # local-p3b\n', 1),
    (EXL3, '                output = runtime["api"].run(binding=binding)', "before",
     "                if _P3B_ON:  # local-p3b\n"
     '                    _p3b.hook_moe_bind(layer, "prefill", m, binding, runtime)  # local-p3b\n', 1),
    (EXL3, "            output = self._apply_rank_sliced(layer, x_2d, weights, ids)", "after",
     "            if _P3B_ON:  # local-p3b\n"
     "                _p3b.hook_moe(layer, x_2d, weights, ids, output)  # local-p3b\n", 1),
    (MLA, "logger = init_logger(__name__)", "after", "\n" + IMPORT_MLA, 1),
    (MLA, "                nsa_cache_seqlens.copy_(per_token_cache)", "before", None, 1),   # H1, see below
    (MLA, "            # Eager bind maps caller-owned scratch into views. forced_num_splits", "before",
     "            if _P3B_ON:  # local-p3b\n"
     "                _p3b.hook_attn_in(  # local-p3b\n"
     "                    self, decode_q, kv_cache, selected_indices, nsa_cache_seqlens,  # local-p3b\n"
     "                    cache_seqlens, num_actual_toks)  # local-p3b\n", 1),
    (MLA, "                return out, lse", "before",
     "                if _P3B_ON:  # local-p3b\n"
     "                    _p3b.hook_attn_out(self, out, lse)  # local-p3b\n", 1),
    (MLA, "            return out, None", "before",
     "            if _P3B_ON:  # local-p3b\n"
     "                _p3b.hook_attn_out(self, out, None)  # local-p3b\n", 1),
]
# H1 goes before the 3-line nsa block that follows _logical_topk_to_physical_slots(...), i.e. before
#     nsa_cache_seqlens = attn_metadata.nsa_cache_seqlens[
# on the DCP1 kpool branch.  That line is not unique in the file, so H1 is anchored on the block's 3rd line
# and inserted 3 lines above it (checked below).
H1_TEXT = ("                if _P3B_ON:  # local-p3b\n"
           "                    _p3b.hook_attn_pre(  # local-p3b\n"
           "                        self, topk_indices, selected_indices, per_token_cache,  # local-p3b\n"
           "                        num_actual_toks)  # local-p3b\n")
H1_ABOVE = ["                nsa_cache_seqlens = attn_metadata.nsa_cache_seqlens[",
            "                    :num_actual_toks",
            "                ]"]
EXPECTED_INSERTED = {EXL3: 7 + 2 + 2 + 2, MLA: 8 + 4 + 4 + 2 + 2}   # marked lines (one blank line before each import block)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def generate(path: str, stock: str) -> str:
    lines = stock.split("\n")
    for f, anchor, where, text, occ in EDITS:
        if f != path:
            continue
        idx = [i for i, ln in enumerate(lines) if ln == anchor]
        if len(idx) != occ:
            raise SystemExit(f"{path}: anchor {anchor!r} found {len(idx)} times, expected {occ}")
        i = idx[0]
        if text is None:          # H1
            if lines[i - 3:i] != H1_ABOVE:
                raise SystemExit(f"{path}: H1 context mismatch above line {i + 1}: {lines[i - 3:i]}")
            prev = "\n".join(lines[i - 12:i - 3])
            if "_logical_topk_to_physical_slots(" not in prev:
                raise SystemExit(f"{path}: H1 is not directly after _logical_topk_to_physical_slots")
            ins = H1_TEXT.rstrip("\n").split("\n")
            lines[i - 3:i - 3] = ins
            continue
        ins = text.rstrip("\n").split("\n")
        if where == "after":
            lines[i + 1:i + 1] = ins
        else:
            lines[i:i] = ins
    return "\n".join(lines)


def additive_gate(path: str, stock: str, hooked: str) -> dict:
    a, b = stock.split("\n"), hooked.split("\n")
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    inserted, bad = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag != "insert":
            bad.append((tag, i1 + 1, i2, j1 + 1, j2))
            continue
        inserted += b[j1:j2]
    unmarked = [ln for ln in inserted if MARK not in ln and ln.strip() != ""]
    blank = [ln for ln in inserted if ln.strip() == ""]
    ok = (not bad and not unmarked and len(inserted) - len(blank) == EXPECTED_INSERTED[path] and len(blank) <= 1)
    return dict(path=path, ok=ok, inserted=len(inserted), inserted_marked=len(inserted) - len(blank) - len(unmarked),
                blank_inserted=len(blank), expected=EXPECTED_INSERTED[path], non_insert_opcodes=bad,
                unmarked=unmarked[:5], stock_sha256=sha(stock.encode()), hooked_sha256=sha(hooked.encode()))


def main() -> int:
    check = "--check" in sys.argv
    res = []
    for path in (EXL3, MLA):
        stock = open(os.path.join(HERE, "stock", path), "rb").read()
        if sha(stock) != STOCK_SHA[path]:
            print(f"FAIL stock copy {path} sha256 {sha(stock)} != {STOCK_SHA[path]}")
            return 2
        stock_s = stock.decode()
        hooked = generate(path, stock_s)
        out = os.path.join(HERE, "files", "vllm", path)
        if check:
            on_disk = open(out).read()
            if on_disk != hooked:
                print(f"FAIL {out} differs from the regenerated file")
                return 2
        else:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as fh:
                fh.write(hooked)
        g = additive_gate(path, stock_s, hooked)
        res.append(g)
        print(("PASS " if g["ok"] else "FAIL ") + f"{path}: inserted {g['inserted']} lines "
              f"({g['inserted_marked']} marked, {g['blank_inserted']} blank), expected {g['expected']} marked; "
              f"non-insert opcodes {len(g['non_insert_opcodes'])}; hooked sha256 {g['hooked_sha256']}")
    return 0 if all(g["ok"] for g in res) else 1


if __name__ == "__main__":
    sys.exit(main())
