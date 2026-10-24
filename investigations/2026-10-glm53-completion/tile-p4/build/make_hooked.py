#!/usr/bin/env python3
"""Generate the five hooked vLLM files of the P4 image from the stock 0.9.1 copies in build/stock/ (extracted from
the local base image a9e985c0 with `docker run --rm --network none`; sha256 pinned below and re-checked inside the
image by the Dockerfile).  Host python3, no torch.

usage: make_hooked.py [--check]

  forcewin call sites (3 files): the validated hook-validation/hook/make_patch.py EDITS for these files, applied
      unchanged.  The stock files are byte-identical to B1's (same pre-hashes), so the outputs must be
      byte-identical to the validated attempt-2 hooked copies (hook-validation/build-a2/files); checked.
      Not applied: make_patch.py's edits to attention/backends/mla/indexer.py (D2 injection: stock lacks B1's
      persistent tail-buffer copy, the anchor does not exist) and layers/sparse_attn_indexer_kpool.py (window
      exposure counters: not a P4 endpoint).
  exl3.py: marked import block + the marked _p4_forced_tile() helper (after `logger = ...`); line 2480
      `tile_config = None` -> `tile_config = _p4_forced_tile()  # local-p4` (the ONLY changed stock line);
      bound-launch recorder before each of the 3 `runtime["api"].run(binding=binding)` sites.
  models/glm5next/nvidia/model.py: marked import block; MoE output after the all-reduce (decoder layer, MoE
      branch); post-layer hidden (model loop, after each layer); final norm (after self.norm).

Gates (also run by --check and by the tests):
  * forcewin files == validated copies, byte for byte;
  * additive gate for exl3.py and model.py: difflib finds only insertions, plus (exl3.py only) exactly one
    replacement: stock line 2480 -> the expected line; every inserted line carries "local-p4" (blank lines
    before import blocks allowed, counted); inserted counts equal EXPECTED;
  * stock-equivalence: dropping every inserted line and restoring line 2480 gives the stock bytes.
"""
import difflib
import hashlib
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
INV = "<runs>"
HV_PATCH = f"{INV}/hook-validation/hook/make_patch.py"
HV_PATCH_SHA = None   # recorded, not pinned (the outputs are compared byte for byte with the validated copies)
HV_A2_FILES = f"{INV}/hook-validation/build-a2/files/vllm"
MARK = "local-p4"
EXL3 = "model_executor/layers/quantization/exl3.py"
MODEL = "models/glm5next/nvidia/model.py"
FW_FILES = ("v1/worker/gpu/model_runner.py", "v1/worker/gpu/spec_decode/rejection_sampler.py",
            "v1/worker/gpu/sample/sampler.py")
STOCK_SHA = {
    EXL3: "3b4ca47cb1168bffebcfe3cf1db48daf86691c140cbf40f9c1f3aa6ad6a444d9",
    MODEL: "28c93ec6efa760fbf05dd6c36184c2a2ef1cba2902cc422b702c41f282c61ecb",
    "v1/worker/gpu/model_runner.py": "101e456b28fbebd0cb73e3f43ae38b4726844a43ee96ec61239bbe455853205f",
    "v1/worker/gpu/spec_decode/rejection_sampler.py": "308d4171f892f27231655289ceb03cd4566533c7d26b8f98bd62ef0a34f986f0",
    "v1/worker/gpu/sample/sampler.py": "76070739447fc9f7563ef164ef80c0777208d1ea07dec3a6f6d2a0755188d7ae",
}
VALIDATED_FW = {   # hook-validation/build-a2 hooked copies (validated on GPU, attempt 2)
    "v1/worker/gpu/model_runner.py": "8d276b3aac4bd6b5b2ef2c1f4b501140a4f1cd4e0dd1eb95f5a48d93e3a81d65",
    "v1/worker/gpu/spec_decode/rejection_sampler.py": "f5a875ef0af0f8e5ceda52d1272c56e2badd07893192f44a0421183d29f75539",
    "v1/worker/gpu/sample/sampler.py": "bbb1df8171c4750bea4f89ff76277a54b4b839caa25ef57d59313b2ce1e146c7",
}

TILE_OLD = "        tile_config = None"
TILE_NEW = "        tile_config = _p4_forced_tile()  # local-p4"
TILE_LINE = 2480

IMPORT_EXL3 = '''# ---- local-p4 (inert unless GLM53_P4=on / GLM53_P4_TILE is set) ----
try:  # local-p4
    from vllm import p4_hooks as _p4  # local-p4
except Exception:  # noqa: BLE001  local-p4
    _p4 = None  # local-p4
_P4_ON = _p4 is not None and _p4.ENABLED  # local-p4
_P4_TILE_ARM_B = (64, 256, 64, 256)  # local-p4: K64/N256 for fc1 and fc2 (the prefill tile)


def _p4_forced_tile():  # local-p4: P4 arm B; None (= stock) unless GLM53_P4_TILE is exactly "64,256,64,256"
    if os.environ.get("GLM53_P4_TILE", "") == "64,256,64,256":  # local-p4
        return _P4_TILE_ARM_B  # local-p4
    return None  # local-p4
# ---- end local-p4 ----'''

IMPORT_MODEL = '''# ---- local-p4 (observe-only; inert unless GLM53_P4=on) ----
try:  # local-p4
    from vllm import p4_hooks as _p4  # local-p4
except Exception:  # noqa: BLE001  local-p4
    _p4 = None  # local-p4
_P4_ON = _p4 is not None and _p4.ENABLED  # local-p4
# ---- end local-p4 ----'''

# (file, anchor: list of consecutive full lines, where: before|after, text)
EDITS = [
    (EXL3, ["logger = init_logger(__name__)"], "after", "\n" + IMPORT_EXL3),
    (EXL3, ['            output = runtime["api"].run(binding=binding)',
            "            return output.to(x.dtype)",
            "",
            '        if runtime["prefill_plan"] is not None and m > runtime["max_trellis_m"]:'], "before",
     "            if _P4_ON:  # local-p4: bound-launch recorder\n"
     '                _p4.hook_bind(layer, "decode", m, binding, runtime)  # local-p4'),
    (EXL3, ['                output = runtime["api"].run(binding=binding)'], "before",
     "                if _P4_ON:  # local-p4: bound-launch recorder\n"
     '                    _p4.hook_bind(layer, "prefill", m, binding, runtime)  # local-p4'),
    (EXL3, ['                tile_output = runtime["api"].run(binding=binding)'], "before",
     "                if _P4_ON:  # local-p4: bound-launch recorder (tiled prefill)\n"
     "                    _p4.hook_bind(  # local-p4\n"
     '                        layer, "decode_tile" if plan is runtime["trellis_plan"] else "prefill_tile",  # local-p4\n'
     "                        current_m, binding, runtime)  # local-p4"),
    (MODEL, ["logger = init_logger(__name__)"], "after", "\n" + IMPORT_MODEL),
    (MODEL, ["            x = self.mlp(x, already_sequence_parallel=self.is_sequence_parallel)"], "after",
     "            if _P4_ON:  # local-p4: MoE output after the TP all-reduce (routed + shared)\n"
     "                _p4.hook_moe_out(self, x)  # local-p4"),
    (MODEL, ["            hidden_states, residual, post, comb = layer(",
             "                positions, hidden_states, residual, post, comb",
             "            )"], "after",
     "            if _P4_ON:  # local-p4: contracted post-layer hidden (layers 12, 34)\n"
     "                _p4.hook_hidden(layer, hidden_states, residual, post, comb)  # local-p4"),
    (MODEL, ["        hidden_states = self.norm(hidden_states)"], "after",
     "        if _P4_ON:  # local-p4: final-norm hidden state\n"
     "            _p4.hook_final(self, hidden_states)  # local-p4"),
]
EXPECTED_INSERTED = {EXL3: 12 + 2 + 2 + 4, MODEL: 7 + 2 + 2 + 2}   # marked lines (blank lines counted separately)
EXPECTED_BLANK = {EXL3: 3, MODEL: 1}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def gen_p4(path: str, stock: str) -> str:
    lines = stock.split("\n")
    if path == EXL3:
        if lines[TILE_LINE - 1] != TILE_OLD:
            raise SystemExit(f"{path}:{TILE_LINE} is {lines[TILE_LINE - 1]!r}, expected {TILE_OLD!r}")
        if sum(1 for ln in lines if ln == TILE_OLD) != 1:
            raise SystemExit(f"{path}: `{TILE_OLD.strip()}` is not unique")
        lines[TILE_LINE - 1] = TILE_NEW
    for f, anchor, where, text in EDITS:
        if f != path:
            continue
        n = len(anchor)
        hits = [i for i in range(len(lines) - n + 1) if lines[i:i + n] == anchor]
        if len(hits) != 1:
            raise SystemExit(f"{path}: anchor {anchor[0]!r} (+{n - 1} lines) found {len(hits)} times")
        i = hits[0]
        ins = text.split("\n")
        if where == "after":
            lines[i + n:i + n] = ins
        else:
            lines[i:i] = ins
    return "\n".join(lines)


def additive_gate(path: str, stock: str, hooked: str) -> dict:
    a, b = stock.split("\n"), hooked.split("\n")
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    inserted, bad, replaced = [], [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert":
            inserted += b[j1:j2]
            continue
        if (tag == "replace" and path == EXL3 and i1 == TILE_LINE - 1 and i2 == TILE_LINE
                and a[i1:i2] == [TILE_OLD] and b[j1:j2] and b[j1:j2][0] == TILE_NEW
                and all(MARK in x or x.strip() == "" for x in b[j1 + 1:j2])):
            replaced.append((i1 + 1, a[i1], b[j1]))
            inserted += b[j1 + 1:j2]
            continue
        bad.append((tag, i1 + 1, i2, j1 + 1, j2))
    unmarked = [ln for ln in inserted if MARK not in ln and ln.strip() != ""]
    blank = [ln for ln in inserted if ln.strip() == ""]
    marked = len(inserted) - len(blank) - len(unmarked)
    exp_repl = 1 if path == EXL3 else 0
    ok = (not bad and not unmarked and marked == EXPECTED_INSERTED[path] and len(blank) == EXPECTED_BLANK[path]
          and len(replaced) == exp_repl)
    # stock equivalence: drop the inserted lines, restore the tile line
    rev = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            rev += b[j1:j2]
        elif tag == "replace":
            rev += a[i1:i2]
    equiv = "\n".join(rev) == stock
    return dict(path=path, ok=bool(ok and equiv), inserted_marked=marked, blank_inserted=len(blank),
                expected=EXPECTED_INSERTED[path], replaced=replaced, non_insert_opcodes=bad, unmarked=unmarked[:5],
                stock_equivalent_after_removal=equiv, stock_sha256=sha(stock.encode()),
                hooked_sha256=sha(hooked.encode()))


def gen_fw():
    spec = importlib.util.spec_from_file_location("hv_make_patch", HV_PATCH)
    mp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mp)
    out = {}
    for rel in FW_FILES:
        raw = open(os.path.join(HERE, "stock", rel), "rb").read()
        if sha(raw) != STOCK_SHA[rel] or sha(raw) != mp.PRE["b1"][rel]:
            raise SystemExit(f"{rel}: stock copy is not the file the validated hook was applied to")
        s = raw.decode()
        for kind, anchor, text in mp.EDITS[rel]:
            if s.count(anchor) != 1:
                raise SystemExit(f"{rel}: forcewin anchor count != 1: {anchor[:60]!r}")
            s = s.replace(anchor, text + anchor if kind == "before" else anchor + text)
        out[rel] = s
    return out, sha(open(HV_PATCH, "rb").read())


def main() -> int:
    check = "--check" in sys.argv
    res, allok = [], True
    fw, hv_sha = gen_fw()
    for rel, s in fw.items():
        val = open(os.path.join(HV_A2_FILES, rel)).read()
        ok = s == val and sha(s.encode()) == VALIDATED_FW[rel]
        out = os.path.join(HERE, "files", "vllm", rel)
        if check:
            ok = ok and open(out).read() == s
        else:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            open(out, "w").write(s)
        allok &= ok
        print(("PASS " if ok else "FAIL ") + f"{rel}: forcewin call sites == validated attempt-2 copy "
              f"({sha(s.encode())[:16]}; make_patch.py {hv_sha[:12]})")
    for path in (EXL3, MODEL):
        stock = open(os.path.join(HERE, "stock", path), "rb").read()
        if STOCK_SHA.get(path) and sha(stock) != STOCK_SHA[path]:
            print(f"FAIL stock copy {path} sha256 {sha(stock)} != {STOCK_SHA[path]}")
            return 2
        hooked = gen_p4(path, stock.decode())
        out = os.path.join(HERE, "files", "vllm", path)
        if check:
            if open(out).read() != hooked:
                print(f"FAIL {out} differs from the regenerated file")
                return 2
        else:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            open(out, "w").write(hooked)
        g = additive_gate(path, stock.decode(), hooked)
        res.append(g)
        allok &= g["ok"]
        print(("PASS " if g["ok"] else "FAIL ") + f"{path}: {g['inserted_marked']} marked + {g['blank_inserted']} blank "
              f"inserted (expected {g['expected']}), replaced {g['replaced']}, other opcodes "
              f"{len(g['non_insert_opcodes'])}, stock-equivalent after removal {g['stock_equivalent_after_removal']}; "
              f"stock {g['stock_sha256'][:16]} hooked {g['hooked_sha256'][:16]}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
