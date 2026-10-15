#!/usr/bin/env python3
"""F1 [R-2]: tests of the scoped cache rule against the image's own lookup (CPU container).

The rule (analysis/fcr_geometry.rule_c) is asserted ONLY where it is claimed to hold: exact repeats and prefix /
extension prompts whose full blocks were all written by prefill under the same salt, without eviction. Everywhere
else the tests show what the image actually does and that the gate's safety properties still hold (block-aligned,
never beyond the permissible prefill coverage, never decode-written history, restored blocks attributed).

Families:
  T1 exact repeats at lengths k*4352 - 1, k*4352, k*4352 + 1, k*4352 + 2727 (k = 1, 2, 3, 15, 25)
  T2 extensions of a warm prompt: +1, to the next boundary - 1, to the boundary, +1 block, +2 blocks, +7,168
  T3 prompts shorter than the build (prefix of a longer prefill-written prompt; LW2 -> LC form)
  T4 cold / other salt / same tokens under another salt
  T5 partial eviction (MLA tail evicted; KDA state at c evicted, MLA kept) and full eviction
  T6 decode-written history: a prompt reaching a forced decode window's blocks restores decode-written blocks; the
     plan invariant forbids such prompts and the gate's provenance check flags them
  T7 the actual retry request forms on the real fixture ids: warm -> touch, C, D, CV, PS, LC (first and second LP
     user), LW2-built prefix -> LC, and C after LC (prefill coverage)
Writes /out/cache_rule_tests.json; prints PASS/FAIL per test.
"""
import json
import sys

sys.path.insert(0, "/replay")
sys.path.insert(0, "/inv/first-contrast/retry/analysis")
import engine_replay as E  # noqa: E402
import fcr_geometry as geo  # noqa: E402

B = E.BLOCK
RES = []


class Mini:
    """Model-mode driver: batches run to completion, decode commits 1 + 2 accepted (deterministic)."""

    def __init__(self, num_blocks=E.NUM_BLOCKS):
        self.s = E.make_scheduler(num_blocks=num_blocks)
        self.e = E.Engine(self.s)
        self.hbs = self.s._replay_block_sizes[1]
        self.label = {}
        self.trk = E.Tracker(self.s, self.e, lambda q: self.label.get(q, q))
        self.outs = {}
        self.pos = {}
        self.k = 0

    def batch(self, items):
        """items: list of dict(prompt, salt, max_tokens=1, forced=None, label). Returns {label: hit}."""
        rids = []
        for it in items:
            self.k += 1
            rid = "%s#%d" % (it.get("label", "r"), self.k)
            self.label[rid] = it.get("label", rid)
            self.outs[rid] = it.get("forced")
            self.pos[rid] = 0
            self.s.add_request(E.make_request(rid, it["prompt"], it.get("max_tokens", 1), it["salt"], self.hbs))
            rids.append(rid)
        pre = {}

        def commit(q, so, n):
            if pre.get((n, q), True):
                return []
            nspec = len((so.scheduled_spec_decode_tokens or {}).get(q, ()))
            kk = 1 + min(2, nspec)
            f = self.outs.get(q)
            p = self.pos[q]
            t = list(f[p:p + kk]) if f else []
            t += E.filler(kk - len(t), 5 + p)
            self.pos[q] += kk
            return t
        g = 0
        while self.s.has_requests() or self.e.queue:
            g += 1
            assert g < 300000
            if self.s.has_requests():
                so = self.e.schedule()
                for q in so.num_scheduled_tokens:
                    pre[(self.e.step_no, q)] = self.s.requests[q].is_prefill_chunk
                if len(self.e.queue) >= 2:
                    self.e.update(commit)
            else:
                self.e.update(commit)
        return {self.label[r]: self.trk.restores.get(r, {}).get("hit") for r in rids}, rids

    def provenance(self, rid):
        rs = self.trk.restores.get(rid, {})
        return sorted({(ow[0], ow[3]) for g, i, b, ow in rs.get("blocks", []) if ow})


def check(name, got, want, note=""):
    ok = got == want
    RES.append(dict(test=name, got=got, want=want, ok=ok, note=note))
    print("%s %-58s got %-8s want %-8s %s" % ("PASS" if ok else "FAIL", name, got, want, note))


def check_true(name, cond, detail):
    RES.append(dict(test=name, ok=bool(cond), detail=detail))
    print("%s %-58s %s" % ("PASS" if cond else "FAIL", name, json.dumps(detail, default=str)[:150]))


def toks(n, seed):
    return E.filler(n, seed)


def main():
    # ---------------- T1 exact repeats
    for k in (1, 2, 3, 15, 25):
        for d in (-1, 0, 1, 2727):
            P = k * B + d
            if P < 2:
                continue
            m = Mini()
            t = toks(P, 1000 * k + d + 7)
            m.batch([dict(prompt=t, salt="t1", label="warm")])
            h, _ = m.batch([dict(prompt=t, salt="t1", label="rep")])
            check("T1 repeat P=%d*4352%+d" % (k, d), h["rep"], geo.rule_c(P, P))
    # ---------------- T2 extensions
    base = toks(25 * B + 2727, 99)
    W = 15 * B + 1234
    for name, Q in (("+1", W + 1), ("to boundary-1", 16 * B - 1), ("to boundary", 16 * B), ("+1 block", W + B),
                    ("+2 blocks", W + 2 * B), ("+7168", W + 7168), ("+10 blocks", W + 10 * B)):
        m = Mini()
        m.batch([dict(prompt=base[:W], salt="t2", label="warm")])
        h, _ = m.batch([dict(prompt=base[:Q], salt="t2", label="ext")])
        check("T2 extension W=15*4352+1234 %s (Q=%d)" % (name, Q), h["ext"], geo.rule_c(W, Q))
    # ---------------- T3 shorter than the build
    for name, (Pb, Q) in (("LW2-like build P+8192 -> P_L=P+7168", (20 * B + 3000 + 8192, 20 * B + 3000 + 7168)),
                          ("build 25 blocks -> prompt 10 blocks + 5", (25 * B + 10, 10 * B + 5)),
                          ("build -> prompt at its boundary", (12 * B + 77, 12 * B))):
        m = Mini()
        m.batch([dict(prompt=base[:Pb], salt="t3", label="warm")])
        h, _ = m.batch([dict(prompt=base[:Q], salt="t3", label="short")])
        check("T3 %s" % name, h["short"], geo.rule_c(Pb, Q))
    # ---------------- T4 cold / salts
    m = Mini()
    h, _ = m.batch([dict(prompt=base[:20 * B], salt="t4", label="cold")])
    check("T4 cold first use", h["cold"], 0)
    h, _ = m.batch([dict(prompt=base[:20 * B + 5], salt="t4-other", label="other")])
    check("T4 same tokens, other salt", h["other"], 0)
    h, _ = m.batch([dict(prompt=base[:20 * B + 5], salt=None, label="nosalt")])
    check("T4 same tokens, no salt", h["nosalt"], 0)
    # ---------------- T5 eviction
    #   full eviction: build, then 0.9 M tokens of other-salt prefill traffic, then repeat -> 0
    m = Mini()
    P = 20 * B + 100
    m.batch([dict(prompt=base[:P], salt="t5", label="warm")])
    for i in range(8):
        m.batch([dict(prompt=toks(110000, 500 + i), salt="flood", label="flood")])
    h, rids = m.batch([dict(prompt=base[:P], salt="t5", label="rep")])
    check("T5 full eviction by 0.88 M tokens of other prefill", h["rep"], 0, "rule would say %d" % geo.rule_c(P, P))
    #   FIFO eviction of one build is all-or-nothing: the intermediate KDA states are freed (and evicted) before the
    #   boundary state at c, which is freed before the MLA/index-K blocks; a growing flood moves the hit from the rule
    #   value straight to 0
    series = []
    for n_flood in (5, 6, 7, 8, 9):
        m = Mini()
        m.batch([dict(prompt=base[:P], salt="t5", label="warm")])
        for i in range(n_flood):
            m.batch([dict(prompt=toks(110000, 700 + i), salt="flood", label="flood")])
        h, rids = m.batch([dict(prompt=base[:P], salt="t5", label="rep")])
        series.append((n_flood, h["rep"]))
    check_true("T5 FIFO eviction of a build: hit is the rule value or 0, block-aligned",
               all(x in (0, geo.rule_c(P, P)) for _, x in series) and any(x == 0 for _, x in series),
               dict(series=series, rule=geo.rule_c(P, P)))
    #   targeted partial evictions (BlockPool.evict_blocks, the image's own API): the lookup falls back to the last
    #   boundary whose MLA chain AND KDA state are both cached -> partial hit, block-aligned, < rule
    cases = (("KDA state at c evicted (7 groups)", "kda_c"), ("MLA block idx 5 evicted", "mla5"),
             ("MLA eagle-check block [c, c0) evicted", "mla_last"))
    for name, kind in cases:
        m = Mini()
        _, wr = m.batch([dict(prompt=base[:P], salt="t5", label="warm")])
        c = geo.rule_c(P, P)
        req = E.make_request("probe", base[:P], 1, "t5", m.hbs)
        pool = m.s.kv_cache_manager.block_pool
        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
        cache = pool.cached_block_hash_to_block._cache

        def ids_of(i, groups):
            out = set()
            for g in groups:
                v = cache.get(make_block_hash_with_group_id(req.block_hashes[i], g))
                if v is not None:
                    out |= ({v.block_id} if not isinstance(v, dict) else set(v))
            return out
        if kind == "kda_c":
            ev = ids_of(c // B - 1, range(2, 9))
            want = c - B
        elif kind == "mla5":
            ev = ids_of(5, [0])
            want = 4 * B
        else:
            ev = ids_of(c // B, [0])
            want = c - B
        pool.evict_blocks(ev)
        h, rids = m.batch([dict(prompt=base[:P], salt="t5", label="rep")])
        check("T5 partial: %s" % name, h["rep"], want, "rule %d -> partial (block-aligned)" % c)
    # ---------------- T6 decode-written history
    G = geo.load()
    ids = E.load_ids("/inv/first-contrast")
    n = "L4"
    pre, fo = ids[n]["prefix"], ids[n]["forced"]
    P = len(pre)
    m = Mini()
    m.batch([dict(prompt=pre, salt="t6", label="warm")])
    m.batch([dict(prompt=pre, salt="t6", max_tokens=len(fo), forced=fo, label="C")])
    x = 2 * B
    h, rids = m.batch([dict(prompt=pre + fo[:x], salt="t6", label="reach")])
    cov = geo.rule_c(P, P + x)
    prov = m.provenance(rids[0])
    check_true("T6 prompt reaching a decode window restores beyond prefill coverage",
               h["reach"] > cov, dict(hit=h["reach"], prefill_coverage_rule=cov))
    check_true("T6 ... and the restored blocks include decode-written history (provenance flags it)",
               any(k == "decode" for _, k in prov), dict(provenance=prov))
    plan = [dict(tag="x", requests=[dict(salt="t6", fixture=n, P=P, forced=(0, 1)),
                                    dict(salt="t6", fixture=n, P=P + x, forced=None)])]
    check_true("T6 the plan invariant refuses that prompt", len(geo.cache_invariant(plan)) > 0,
               dict(violations=geo.cache_invariant(plan)))
    # ---------------- T7 actual retry request forms (real ids)
    plan = geo.phases(G, "B")
    for nfx in ("M3", "L5", "L1"):
        pre, fo = ids[nfx]["prefix"], ids[nfx]["forced"]
        f = G["fx"][nfx]
        P, PL = f["P"], f["P_L"]
        m = Mini()
        m.batch([dict(prompt=pre, salt="s", label="warm")])
        h, _ = m.batch([dict(prompt=pre, salt="s", label="touch")])
        check("T7 %s touch (repeat of the build)" % nfx, h["touch"], geo.rule_c(P, P))
        h, r1 = m.batch([dict(prompt=pre + fo[:7168], salt="s", max_tokens=f["L_s"], forced=fo[7168:], label="LC1")])
        check("T7 %s LC first user (P_L on the P build)" % nfx, h["LC1"], geo.rule_c(P, PL))
        h, r2 = m.batch([dict(prompt=pre + fo[:7168], salt="s", max_tokens=f["L_s"], forced=fo[7168:], label="LC2")])
        check("T7 %s LC second user (P_L on LC1's prefill)" % nfx, h["LC2"], geo.rule_c(PL, PL))
        check_true("T7 %s LC second user restores prefill-written blocks only" % nfx,
                   all(k == "prefill" for _, k in m.provenance(r2[0])), dict(prov=m.provenance(r2[0])))
        h, rc = m.batch([dict(prompt=pre, salt="s", max_tokens=f["L"], forced=fo, label="C")])
        check("T7 %s C after the LP runs (prompt P)" % nfx, h["C"], geo.rule_c(PL, P))
        if nfx in G["fx"] and "cv" in G["fx"][nfx]:
            cv = G["fx"][nfx]["cv"]
            h, _ = m.batch([dict(prompt=pre + fo[:cv["x"]], salt="s", max_tokens=cv["list_len"],
                                 forced=fo[cv["x"]:cv["x"] + cv["list_len"]], label="CV")])
            check("T7 %s CV (P' = B1 - 63) after C" % nfx, h["CV"], geo.rule_c(PL, cv["P_cv"]))
        h, rp = m.batch([dict(prompt=pre, salt="s", max_tokens=1, forced=fo[:1], label="PS")])
        check("T7 %s PS after C (prompt P)" % nfx, h["PS"], geo.rule_repeat(P))
        h, rd = m.batch([dict(prompt=pre, salt="s", max_tokens=f["L"], forced=fo, label="D")])
        check("T7 %s D after C and PS (prompt P)" % nfx, h["D"], geo.rule_repeat(P))
        same = [b for g, i, b, ow in m.trk.restores[rc[0]]["blocks"] if g == 0] == \
            [b for g, i, b, ow in m.trk.restores[rd[0]]["blocks"] if g == 0]
        check_true("T7 %s C and D restore the identical physical MLA/index-K blocks" % nfx, same, {})
        # LW2-style build (P + 8192) then LC prompt P_L
        m = Mini()
        m.batch([dict(prompt=pre + fo[:8192], salt="l2", label="LW2")])
        h, _ = m.batch([dict(prompt=pre + fo[:7168], salt="l2", max_tokens=f["L_s"], forced=fo[7168:], label="LC")])
        check("T7 %s LC on an LW2 build (P+8192)" % nfx, h["LC"], geo.rule_c(P + 8192, PL))
    # plan expectations equal the engine for the plan's own sequences is shown by the session sweep (sim_*.json)
    npass = sum(1 for r in RES if r["ok"])
    print("cache_rule_tests: %d / %d PASS" % (npass, len(RES)))
    json.dump(dict(results=RES, passed=npass, total=len(RES)), open("/out/cache_rule_tests.json", "w"), indent=1,
              default=str)


if __name__ == "__main__":
    main()
