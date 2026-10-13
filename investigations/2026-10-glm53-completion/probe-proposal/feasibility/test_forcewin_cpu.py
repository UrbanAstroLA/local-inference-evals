#!/usr/bin/env python3
"""CPU feasibility checks for the forced-decode window hook, run INSIDE an engine image as a CPU container:

  docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -v <this dir>:/w:ro --entrypoint python3 <image> \
      /w/test_forcewin_cpu.py

No GPU, no server, no network, read-only mount; nothing is written outside the container's /tmp.
Checks (each prints PASS/FAIL):
  F1-F4  forced_override: acceptance = longest draft prefix equal to the history; committed tokens = history;
         unforced requests untouched; window end clamps the commit; K=3 and K=5 layouts; zero-draft rows.
  M1-M2  row_metrics against the image's own served top-p implementation (apply_top_k_top_p_pytorch):
         eligibility bounds bracket the served mask; post-mask probabilities equal the served softmax.
  E1     installed engine: Model Runner V2 refuses custom logits processors (source of the installed module).
  E2     installed engine: DFlash2 forces Model Runner V2 (source of the installed module).
  E3     installed engine: a completions request carries vllm_xargs (a list of ints) into SamplingParams.extra_args,
         and cache_salt is accepted; logprob_token_ids sets SamplingParams.logprobs to None (spec-decode gotcha).
  E4     installed engine: a general plugin declared by a .dist-info on sys.path is discovered by
         vllm.plugins.load_plugins_by_group (the delivery route that would leave engine files unmodified).
         Result (cpu-check.out): PASS on 0.7.0; FAIL on 0.9.1 because that image sets VLLM_PLUGINS=
         lora_filesystem_resolver (an allowlist). plugin_startup_check.sh (E4b) confirms the same at process
         start. The proposal therefore ships the hook as hook lines in copied files (guard-image precedent).
"""
import inspect, os, sys, tempfile, textwrap, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch  # noqa: E402
from forcewin_logic import TERM_IDS, forced_override, row_metrics  # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("%s %s %s" % ("PASS" if cond else "FAIL", name, detail))


def run(name, fn):
    try:
        fn()
    except Exception as exc:  # report and continue
        traceback.print_exc()
        check(name, False, "exception: %r" % (exc,))


def batch(K, reqs):
    """reqs: list of dict(prompt_len, o0, n_drafts, drafts(list), forced(list or None), sampled_row(list))."""
    R = len(reqs)
    cu, pos, inp = [0], [], []
    for q in reqs:
        n = q["n_drafts"] + 1
        cu.append(cu[-1] + n)
        last_pos = q["prompt_len"] + q["o0"] - 1            # position of the last committed token
        pos += [last_pos + j for j in range(n)]
        inp += [q.get("last", 7)] + q["drafts"]
    Wmax = max(len(q["forced"]) if q["forced"] else 1 for q in reqs)
    forced = torch.zeros(R, Wmax, dtype=torch.int64)
    flen = torch.zeros(R, dtype=torch.int64)
    for i, q in enumerate(reqs):
        if q["forced"]:
            forced[i, :len(q["forced"])] = torch.tensor(q["forced"])
            flen[i] = len(q["forced"])
    sampled = torch.full((R, K + 1), -5, dtype=torch.int64)
    for i, q in enumerate(reqs):
        sampled[i, :len(q["sampled_row"])] = torch.tensor(q["sampled_row"])
    num = torch.tensor([q["num_sampled"] for q in reqs], dtype=torch.int32)
    return dict(sampled=sampled, num_sampled=num, draft_sampled=torch.tensor(inp, dtype=torch.int64),
                pos=torch.tensor(pos, dtype=torch.int64), cu=torch.tensor(cu, dtype=torch.int64), forced=forced,
                forced_len=flen, prompt_len=torch.tensor([q["prompt_len"] for q in reqs], dtype=torch.int64),
                is_forced=torch.tensor([q["forced"] is not None for q in reqs]))


def t_forced():
    F = list(range(1000, 1100))                       # history tokens by output index
    # K = 3. req0: drafts match F[10], F[11], then a mismatch -> 2 accepted, commits F[10..12]
    # req1: unforced, real sampler output kept. req2: forced, no drafts this step. req3: near the window end.
    reqs = [
        dict(prompt_len=50, o0=10, n_drafts=3, drafts=[F[10], F[11], 999], forced=F, sampled_row=[1, 2, 3, 4], num_sampled=4),
        dict(prompt_len=70, o0=5, n_drafts=3, drafts=[11, 12, 13], forced=None, sampled_row=[21, 22, 23, 24], num_sampled=2),
        dict(prompt_len=60, o0=0, n_drafts=0, drafts=[], forced=F, sampled_row=[31], num_sampled=1),
        dict(prompt_len=40, o0=98, n_drafts=3, drafts=[F[98], F[99], F[0]], forced=F, sampled_row=[1, 1, 1, 1], num_sampled=4),
    ]
    b = batch(3, reqs)
    s, n, valid, oi = forced_override(**b)
    check("F1 accepted prefix and committed history (K=3)", n[0].item() == 3 and s[0, :3].tolist() == F[10:13],
          "num=%d committed=%s" % (n[0].item(), s[0, :3].tolist()))
    check("F2 unforced request untouched", n[1].item() == 2 and s[1].tolist() == [21, 22, 23, 24])
    check("F3 zero-draft forced row commits one history token", n[2].item() == 1 and s[2, 0].item() == F[0])
    check("F4a window end clamps the commit", n[3].item() == 2 and s[3, :2].tolist() == F[98:100],
          "num=%d" % n[3].item())
    check("F4b valid rows = rows conditioned only on history", valid[0].tolist() == [True, True, True, False]
          and valid[1].tolist() == [False] * 4 and valid[2].tolist() == [True, False, False, False]
          and oi[0, :4].tolist() == [10, 11, 12, 13])
    # K = 5 (0.7.0): all five drafts match -> 6 committed
    reqs5 = [dict(prompt_len=10, o0=20, n_drafts=5, drafts=F[20:25], forced=F, sampled_row=[0] * 6, num_sampled=1)]
    s5, n5, v5, _ = forced_override(**batch(5, reqs5))
    check("F4c K=5 full acceptance commits 6 history tokens", n5[0].item() == 6 and s5[0].tolist() == F[20:26])


def make_logits(V, rows, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, V, generator=g) * 2.0
    # heavy head so that top_p 0.95 masks a large tail; place tracked ids at chosen quantiles
    x[:, :50] += 9.0
    for r in range(rows):
        for k, tid in enumerate(TERM_IDS):
            x[r, tid] = x[r, (r * 7 + k * 13) % V] + [8.0, 3.0, 0.0, -2.0][(r + k) % 4]
    return x


def t_metrics():
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
    V, rows = 154880, 12
    x = make_logits(V, rows, 1234)
    forced_tok = torch.randint(0, V, (rows,), generator=torch.Generator().manual_seed(7))
    forced_tok[:4] = torch.tensor(TERM_IDS)
    served = apply_top_k_top_p_pytorch(x.clone(), None, torch.full((rows,), 0.95))
    m = row_metrics(x, served, forced_tok)
    track = torch.cat([torch.tensor(TERM_IDS).expand(rows, -1), forced_tok[:, None]], dim=1)
    kept = torch.isfinite(served.gather(1, track))
    ok_lo = bool((~m["elig_lo"] | kept).all())
    ok_hi = bool((~kept | m["elig_hi"]).all())
    n_amb = int((m["elig_hi"] & ~m["elig_lo"]).sum())
    check("M1 eligibility bounds bracket the served top-p mask", ok_lo and ok_hi,
          "kept=%d/%d, ambiguous(ties)=%d" % (int(kept.sum()), kept.numel(), n_amb))
    ref = torch.softmax(served, dim=-1).gather(1, track)
    check("M2 post-mask probabilities equal the served softmax", torch.allclose(m["p_post"], ref, atol=1e-7),
          "max|d|=%.2e; tracked kept with p_post>0: %d" % ((m["p_post"] - ref).abs().max().item(), int((m["p_post"] > 0).sum())))


def t_engine_sources():
    from vllm.config import vllm as cfg
    src = inspect.getsource(cfg.VllmConfig._get_v2_model_runner_unsupported_features)
    check("E1 V2 runner refuses custom logits processors", '"custom logits processors"' in src)
    src2 = inspect.getsource(cfg.VllmConfig.use_v2_model_runner.fget if isinstance(
        cfg.VllmConfig.__dict__.get("use_v2_model_runner"), property) else cfg.VllmConfig.use_v2_model_runner)
    check("E2 DFlash2 forces the V2 runner", "DFlash2 candidate selector exists only in the V2" in src2)


def t_protocol():
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    ids = list(range(100, 120))
    req = CompletionRequest(model="m", prompt=ids, max_tokens=9, temperature=1.0, top_p=0.95, seed=5,
                            cache_salt="probe-salt-1", vllm_xargs={"local_forcewin": [1, 2, 3], "fixture": "L1"})
    sp = req.to_sampling_params(max_tokens=9)
    check("E3a vllm_xargs list reaches SamplingParams.extra_args",
          sp.extra_args == {"local_forcewin": [1, 2, 3], "fixture": "L1"} and sp.seed == 5 and sp.top_p == 0.95,
          "extra_args=%s" % (sp.extra_args,))
    check("E3b cache_salt accepted on a token-id completions request", req.cache_salt == "probe-salt-1")
    req2 = CompletionRequest(model="m", prompt=ids, max_tokens=4, logprobs=5, logprob_token_ids=list(TERM_IDS))
    sp2 = req2.to_sampling_params(max_tokens=4)
    check("E3c logprob_token_ids sets SamplingParams.logprobs=None (not served by the V2 rejection sampler)",
          sp2.logprobs is None and list(sp2.logprob_token_ids) == list(TERM_IDS))


def t_plugin():
    import vllm.plugins as P
    d = tempfile.mkdtemp(prefix="plg")
    os.makedirs(os.path.join(d, "local_probe_selftest-0.0.dist-info"))
    open(os.path.join(d, "local_probe_selftest.py"), "w").write("HIT = []\ndef register():\n    HIT.append(1)\n")
    open(os.path.join(d, "local_probe_selftest-0.0.dist-info", "METADATA"), "w").write(
        "Metadata-Version: 2.1\nName: local-probe-selftest\nVersion: 0.0\n")
    open(os.path.join(d, "local_probe_selftest-0.0.dist-info", "entry_points.txt"), "w").write(
        textwrap.dedent("""\
        [vllm.general_plugins]
        local_probe_selftest = local_probe_selftest:register
        """))
    sys.path.insert(0, d)
    import importlib
    importlib.invalidate_caches()
    found = P.load_plugins_by_group("vllm.general_plugins")
    ok = "local_probe_selftest" in found
    if ok:
        found["local_probe_selftest"]()
        import local_probe_selftest as m
        ok = m.HIT == [1]
    check("E4 general plugin from a .dist-info on sys.path is discovered and runs", ok,
          "plugins found: %s" % sorted(found))


if __name__ == "__main__":
    import vllm
    print("vllm", getattr(vllm, "__version__", "?"), "torch", torch.__version__, "cuda available", torch.cuda.is_available())
    run("F", t_forced)
    run("M", t_metrics)
    run("E1/E2", t_engine_sources)
    run("E3", t_protocol)
    run("E4", t_plugin)
    n_ok = sum(ok for _, ok in RESULTS)
    print("%d/%d checks passed" % (n_ok, len(RESULTS)))
    sys.exit(0 if n_ok == len(RESULTS) else 1)
