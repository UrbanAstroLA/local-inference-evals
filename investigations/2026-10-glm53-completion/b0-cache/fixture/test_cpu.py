"""CPU tests for the B0 cache fixture logic (no GPU). Run inside the B0 image with
VLLM_GLM53_NOPE_RECORD=1 and no --gpus. Writes a JSON summary to argv[1].

T1 geometry derivation vs the B0 server log
T2 real B0 view code (meta tensors, full size) -> byte offsets equal the footprint model
   at blocks 1, 934, 935, 1274 and the record straddling 2^31
T3 sentinel/shadow detection on a small tensor (correct writes pass; a misplaced write,
   a missing write and an overrun are detected and located)
T4 same at FULL size (2.93 GB CPU tensor x2): writes at high offsets incl. the straddling
   record are tracked exactly (checks the checker has no 32-bit limits)
T5 payload tags round-trip (bf16, fp32, E4M3 latent via the B12x pure-torch record
   reference pack/unpack)
"""
import json, os, sys, time, traceback

os.environ.setdefault("VLLM_GLM53_NOPE_RECORD", "1")
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b0geom as G
import b0shadow as SH

RES = {}


def t(name):
    def deco(fn):
        def run():
            t0 = time.time()
            try:
                out = fn()
                RES[name] = {"ok": True, "secs": round(time.time() - t0, 2), **(out or {})}
            except Exception as e:  # noqa: BLE001
                RES[name] = {"ok": False, "error": repr(e), "tb": traceback.format_exc()[-2000:]}
        return run
    return deco


@t("T1_geometry")
def t1():
    log = "/w/in/server-s1.log"
    line = None
    if os.path.exists(log):
        for l in open(log, errors="replace"):
            if "glm53-placement kv-capacity" in l:
                line = l
                break
    d = G.check_against_log(line)
    return {"log_checked": line is not None, "first_high_block": d["first_high_block"],
            "straddle": d["straddle_bytes"]}


BLOCKS = [1, 934, 935, 1274]


@t("T2_views_real_code")
def t2():
    import b0views as V
    raw = torch.empty(G.MLA_TENSOR, dtype=torch.int8, device="meta")
    rawi = torch.empty(G.IDX_TENSOR, dtype=torch.int8, device="meta")
    mla = V.mla_view(raw)
    kv = V.draft_view(raw)
    kc, vc = V.draft_kv_split(kv)
    conv, conv_k, ssm, _ = V.kda_views(raw)
    idx = V.idx_view(rawi)
    tail = V.tail_view(rawi)
    checks = []

    def eq(label, got, want):
        checks.append((label, got, want))
        assert got == want, (label, got, want)

    for b in BLOCKS:
        for tok in (0, 1, 4351):
            s = G.mla_slot(b, tok)
            eq(f"mla b{b} t{tok}", V.byte_offset(mla, (s // 64, s % 64, 0), raw),
               G.mla_record_range(s)[0])
        for tok in (0, 1087):
            for h in (0, 3):
                eq(f"draftK b{b} t{tok} h{h}", V.byte_offset(kc, (b, tok, h, 0), raw),
                   G.draft_k_range(b, tok, h)[0])
                eq(f"draftV b{b} t{tok} h{h}", V.byte_offset(vc, (b, tok, h, 0), raw),
                   G.draft_v_range(b, tok, h)[0])
        eq(f"kda conv b{b}", V.byte_offset(conv, (b, 0, 0), raw), G.kda_conv_range(b)[0])
        eq(f"kda conv end b{b}", V.byte_offset(conv, (b, 5, 12287), raw) + 2, G.kda_conv_range(b)[1])
        eq(f"kda ssm b{b}", V.byte_offset(ssm, (b, 0, 0, 0), raw), G.kda_ssm_range(b)[0])
        eq(f"kda ssm end b{b}", V.byte_offset(ssm, (b, 31, 127, 127), raw) + 4, G.kda_ssm_range(b)[1])
        for pool in (0, 63, 64, 1087):
            kp = b * G.IDX_KPAGES_PER_BLOCK + pool // 64
            flat = (pool % 64) * 128  # page-planar: K rows first, scales at byte 8,192
            eq(f"idxK b{b} p{pool}", V.byte_offset(idx, (kp, flat // 132, flat % 132), rawi),
               G.idx_k_range(b, pool)[0])
            flat = 64 * 128 + (pool % 64) * 4
            eq(f"idxS b{b} p{pool}", V.byte_offset(idx, (kp, flat // 132, flat % 132), rawi),
               G.idx_scale_range(b, pool)[0])
        eq(f"tail b{b}", V.byte_offset(tail, (b, 0, 0, 0), rawi), G.tail_range(b)[0])
        eq(f"tail end b{b}", V.byte_offset(tail, (b, 1, 7, 127), rawi) + 2, G.tail_range(b)[1])
    s = G.first_high_mla_slot()
    eq("straddle", V.byte_offset(mla, (s // 64, s % 64, 0), raw), G.mla_record_range(s)[0])
    # padded strides really are the page sizes
    eq("draft block stride B", kc.stride(0) * 2, G.MLA_PAGE)
    eq("ssm block stride B", ssm.stride(0) * 4, G.MLA_PAGE)
    eq("conv block stride B", conv.stride(0) * 2, G.MLA_PAGE)
    eq("tail block stride B", tail.stride(0) * 2, G.IDX_PAGE)
    eq("mla kpage stride", mla.stride(0), 64 * 528)
    eq("idx kpage stride", idx.stride(0), G.IDX_KPAGE_BYTES)
    return {"n_checks": len(checks), "draft_view": [list(kc.shape), list(kc.stride())],
            "conv_kernel_view": [list(conv_k.shape), list(conv_k.stride())],
            "ssm_view": [list(ssm.shape), list(ssm.stride())],
            "tail_view": [list(tail.shape), list(tail.stride())],
            "mla_view": [list(mla.shape), list(mla.stride())],
            "idx_view": [list(idx.shape), list(idx.stride())]}


@t("T3_shadow_small")
def t3():
    n = 64 << 20
    raw = torch.empty(n, dtype=torch.int8)
    sh = SH.Shadow(raw, salt=12345, kind="mla")
    r0 = sh.check("fresh")
    assert r0["ok"]
    # correct write: one 528-B record at slot 7
    data = torch.arange(528, dtype=torch.int16).to(torch.int8)
    raw[7 * 528 : 8 * 528] = data
    sh.expect_bytes(7 * 528, data)
    assert sh.check("correct")["ok"]
    # misplaced write: shifted by one record
    raw[9 * 528 : 10 * 528] = data
    sh.expect_bytes(8 * 528, data)
    r = sh.check("misplaced")
    assert not r["ok"] and r["where"][0]["mla_slot"] in (8, 9), r
    # overrun by 2 bytes
    raw[20 * 528 : 21 * 528 + 2] = 1
    sh.expect_bytes(20 * 528, torch.ones(528, dtype=torch.int8))
    r2 = sh.check("overrun")
    assert not r2["ok"] and r2["runs"][0][0] // 8 * 8 == (21 * 528) // 8 * 8, r2
    # missing write (shadow expects, raw untouched)
    sh.expect_zero(100 * 528, 101 * 528)
    r3 = sh.check("missing")
    assert not r3["ok"]
    # pattern regeneration equals fill
    pb = SH.pattern_bytes(1000, 1100, 12345)
    raw2 = torch.empty(4096, dtype=torch.int8)
    SH.fill_pattern(raw2, 12345)
    assert torch.equal(raw2[1000:1100], pb)
    return {"detections": [r["differing_words"], r2["differing_words"], r3["differing_words"]]}


@t("T4_shadow_full_size_high_offsets")
def t4():
    raw = torch.empty(G.MLA_TENSOR, dtype=torch.int8)
    sh = SH.Shadow(raw, salt=777, kind="mla")
    writes = []
    for b in BLOCKS:
        s, e = G.draft_token_range(b, 1087)
        writes.append((s, e))
        s, e = G.kda_ssm_range(b)
        writes.append((e - 4096, e))
    writes.append(G.mla_record_range(G.first_high_mla_slot()))
    writes.append(G.mla_record_range(G.mla_slot(1274, 4351)))  # last record of the tensor
    for i, (s, e) in enumerate(writes):
        d = torch.full((e - s,), (i % 100) + 1, dtype=torch.int8)
        raw[s:e] = d
        sh.expect_bytes(s, d)
    ok = sh.check("high writes tracked")
    assert ok["ok"], ok
    # a stray 8-byte write just above 2^31 and one in the last page's padding are found
    raw[G.TWO31 : G.TWO31 + 8] = 5
    raw[G.MLA_TENSOR - 8 : G.MLA_TENSOR] = 5
    r = sh.check("stray above 2^31")
    assert not r["ok"] and r["runs"][0][0] == G.TWO31 and r["runs"][-1][0] == G.MLA_TENSOR - 8, r
    return {"writes": len(writes), "stray_located": [x["block"] for x in r["where"]]}


@t("T5_payload_tags")
def t5():
    out = {}
    v = SH.bf16_tag_values("draft", 1274, 1087, 5, 128)
    assert SH.bf16_untag(v) == {"group": "draft", "epoch": 5, "block": 1274, "tok_mod128": 1087 % 128}
    f = SH.fp32_tag_values("kda", 935, 0, 3, 16)
    assert SH.untag_code(int(f[0])) == {"group": "kda", "epoch": 3, "block": 935, "tok_mod128": 0}
    tag = SH.tag_code("mla", 934, 2435, 2)
    lat = SH.e4m3_exact_latent(tag)
    try:
        from b12x.attention._shared.mla import reference as R
        rec = _pack_glm_next(R, lat)
        deq = _unpack_glm_next(R, rec)
        assert torch.equal(deq.float(), lat.float()), (deq[:8], lat[:8])
        assert SH.e4m3_untag(deq) == SH.untag_code(tag)
        out["record_bytes"] = int(rec.numel())
        out["reference_roundtrip_exact"] = True
    except ImportError as e:
        out["reference_import"] = repr(e)
        raise
    return out


def _pack_glm_next(R, lat):
    """B12x pure-torch packer; no RoPE lane -> 528-B GLM_NEXT record (reference.py:35-79)."""
    return R.pack_mla_kv_cache_reference(lat.view(1, 512))


def _unpack_glm_next(R, rec):
    return R.unpack_mla_kv_cache_reference(rec).reshape(-1)[:512]


if __name__ == "__main__":
    for fn in (t1, t2, t3, t4, t5):
        fn()
    json.dump(RES, open(sys.argv[1], "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "tb"} for k, v in RES.items()},
                     indent=1))
    sys.exit(0 if all(v["ok"] for v in RES.values()) else 1)
