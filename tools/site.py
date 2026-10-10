#!/usr/bin/env python3
"""Build the static results site in docs/ from the repository's own data (standard library only).

    python3 tools/site.py

Every chart and table is computed from runs/ at build time, so the site cannot drift from the receipts.
Intervals: GPQA accuracy = 95% bootstrap over questions; rates (empty answers, v2 screen failures for one question)
= 95% Wilson score intervals. Earlier fixed-seed screens (hard-prompt-screen v0/v1) are shown as single draws, never rates. Serve docs/ with GitHub Pages.
Labels, the label key and chart colours come from config fields (SCHEMA.md, "Labels"): a new engine needs data, not code.
"""
import html, json, math, random, re, statistics as st
from pathlib import Path

from verify import config_label, engine_label, weights_label
import analyze

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs"
REPO = "https://github.com/UrbanAstroLA/local-inference-evals"
esc = html.escape


# ---------------------------------------------------------------- data

def load_runs():
    runs = {}
    for d in sorted((ROOT / "runs").iterdir()):
        m = json.loads((d / "run.json").read_text())
        m["summary"] = json.loads((d / "summary.json").read_text())
        m["rows"] = [json.loads(l) for l in (d / "results.jsonl").read_text().splitlines() if l.strip()]
        m["cfg"] = json.loads((ROOT / "configs" / f"{m['config']}.json").read_text())
        runs[m["id"]] = m
    return runs


SERIES = {}            # engine.series -> colour, in order of each series' first run (new engines get the next colour)
PALETTE = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"]


def label(cfg):
    return config_label(cfg)


def color(cfg):
    return SERIES[cfg["engine"]["series"]]


def label_width(labels):
    """Left margin for chart row labels: about 7 px per character at 13 px (wide sans-serif fonts), plus the gap."""
    return int(18 + 7.0 * max((len(x) for x in labels), default=20))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def link(rid, text=None):
    return f'<a href="{REPO}/tree/main/runs/{esc(rid)}">{esc(text or "receipts")}</a>'


# ---------------------------------------------------------------- glossary (method page; first use on every other page links here)

GLOSSARY = [
    ("mla", r"\bMLA\b(?! (?:layer )?ownership| owners)", "MLA", "Multi-head latent attention: GLM-5.3-Flash's attention in 11 of its layers (the others are KDA "
     "layers). Keys and values are cached as a compressed latent, and a sparse-attention indexer picks which earlier tokens each decode step attends."),
    ("kpool", r"\bkpool\b", "kpool", "The sparse-attention indexer's pooled key cache: earlier tokens are grouped in pools of 4, and each decode step "
     "selects up to 512 pools (2,048 tokens). The <i>kpool tail</i> is the newest, incomplete pool: the current token and up to two before it."),
    ("dcp", r"\bDCP[12]\b", "DCP1 / DCP2", "Decode context parallelism. DCP2 splits each MLA layer's KV cache across both GPUs by token; DCP1 "
     "does not split it."),
    ("mla-ownership", r"\bMLA (?:layer )?(?:ownership|owners)\b", "MLA ownership", "tpurtell's DCP1 layout from 0.8.0 on: each MLA layer, with its "
     "indexer and caches, lives on one GPU only (<code>split:25</code>: layers 3-23 on the first GPU, 27-43 on the second) and runs all heads "
     "there. <code>tp</code> is the setting without ownership."),
    ("ep-tp", r"\b(?:EP2|TP2)\b", "EP2 / TP2", "Routed experts placed whole on one GPU each (expert parallel, EP2) or each expert split across "
     "both GPUs (tensor parallel, TP2)."),
    ("nope-record", r"\bNOPE records?\b", "NOPE record", "The KV cache record of an MLA latent. tpurtell 0.8.0 on stores a 528-byte record "
     "(512 FP8 bytes and four FP32 scales); <i>NOPE records off</i> uses the 656-byte record that carries an unused rotary tail. tpurtell reports "
     "that outputs match within BF16 rounding."),
    ("dflash2", r"\bDFlash2\b", "DFlash2 ×N", "Speculative decoding with the draft model incoai/GLM-5.3-Flash-DFlash2, which proposes N tokens "
     "per step; the served model checks them in one forward pass and keeps those it accepts. Acceptance rate = accepted / drafted tokens."),
    ("draft-slot-sharing", r"\b[Dd]raft[- ]slot sharing\b", "Draft-slot sharing", "From tpurtell 0.8.0 on, the DFlash2 draft cache is stored "
     "inside the MLA cache's slot tensors instead of separate tensors. <i>sharing off</i> disables it."),
    ("flexible-extract", r"\bflexible-extract\b", "flexible-extract", "lm-evaluation-harness's GPQA answer filter: the last parenthesised capital "
     "letter in the reply. The headline (raw) GPQA score. <code>correct_stated</code> is the secondary, audited score: whether the reply's stated "
     "final answer is right."),
    ("empty-answer", r"\b[Ee]mpty answers?\b", "Empty answer", "A GPQA reply with no content after its reasoning: the reasoning hit the "
     "327,680-token budget, or ended without an answer. Scored wrong."),
    ("loop", r"\b[Ll]oops?\b", "Loop", "A request that does not finish and repeats itself: stopped by the screen's early-stop detector (three "
     "consecutive checks with the last 30,000 reasoning characters compressing below 10% of their size), or ending at the budget with a tail that "
     "compresses below 15%."),
    ("exhaustion", r"\b[Ee]xhaustions?\b", "Exhaustion", "A request that runs to the 327,680-token budget with varied, non-repetitive reasoning."),
]
SKIP_TAGS = {"a", "svg", "h1", "h2", "code", "title", "style", "script", "th", "nav", "summary"}


def gloss(body, target="method.html"):
    """Link the first use of each glossary term on a page (text outside links, headings, charts and code) to its definition."""
    depth, done, out = [], set(), []
    for part in re.split(r"(<[^>]+>)", body):
        if part.startswith("<"):
            m = re.match(r"<(/?)([a-zA-Z0-9]+)", part)
            if m and m.group(2).lower() in SKIP_TAGS and not part.endswith("/>"):
                if m.group(1):
                    if depth: depth.pop()
                else:
                    depth.append(m.group(2).lower())
            out.append(part); continue
        segs = [part]
        if not depth:
            for slug, pat, _, _ in GLOSSARY:
                if slug in done: continue
                for n, seg in enumerate(segs):
                    if isinstance(seg, tuple): continue
                    m = re.search(pat, seg)
                    if m:
                        segs[n:n + 1] = [seg[:m.start()], (f'<a class="g" href="{target}#g-{slug}">{m.group(0)}</a>',), seg[m.end():]]
                        done.add(slug); break
        out.append("".join(x[0] if isinstance(x, tuple) else x for x in segs))
    return "".join(out)


def glossary_html():
    return ('<h2 id="glossary">Glossary</h2><dl class="gl">' + "".join(f'<dt id="g-{slug}">{term}</dt><dd>{d}</dd>' for slug, _, term, d in GLOSSARY)
            + '</dl>')


# ---------------------------------------------------------------- chart parts

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1a9a7a;--s4:#8a56d6;--good:#0ca30c;--crit:#d03b3b;
--seq1:#cde2fb;--seq2:#9ec5f4;--seq3:#6da7ec;--seq4:#3987e5;--seq5:#256abf;--seq6:#184f95;--seq7:#0d366b}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#2bb08d;--s4:#9a6be6}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;
--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#2bb08d;--s4:#9a6be6}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}
nav{display:flex;flex-wrap:wrap;gap:4px 18px;padding:14px 16px;max-width:980px;margin:0 auto;border-bottom:1px solid var(--grid)}
nav a{color:var(--ink2);text-decoration:none}nav a.on{color:var(--ink);font-weight:600}nav .brand{color:var(--ink);font-weight:600;margin-right:12px}
h1{font-size:28px;line-height:1.2;margin:8px 0 6px}h2{font-size:19px;margin:36px 0 6px}p,li{color:var(--ink2)}
a{color:var(--s1)}.lede{font-size:16px;max-width:760px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:22px 0}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:14px 16px}
.tile .lab{color:var(--ink2);font-size:13px}.tile .val{font-size:30px;font-weight:600;margin:2px 0}.tile .sub{color:var(--muted);font-size:13px}
figure{background:var(--surface);border:1px solid var(--ring);border-radius:10px;margin:16px 0;padding:16px 16px 10px}
figcaption .t{font-weight:600}figcaption .s{color:var(--ink2);font-size:13px;margin-top:2px}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:13px;color:var(--ink2);margin:10px 0 4px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.chart{overflow-x:auto}.chart svg{min-width:640px}svg{display:block;width:100%;height:auto;max-width:900px;font:13px system-ui,-apple-system,"Segoe UI",sans-serif}
svg text{fill:var(--ink2)}svg .muted{fill:var(--muted)}svg .grid{stroke:var(--grid);stroke-width:1}svg .axis{stroke:var(--axis);stroke-width:1}
svg .hit{fill:transparent;cursor:default}svg .hit:hover+g,svg .hit:focus+g{opacity:.75}
details{margin:6px 0 2px}summary{cursor:pointer;color:var(--ink2);font-size:13px}
table{border-collapse:collapse;width:100%;margin:8px 0;font-size:13.5px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}
th{color:var(--ink2);font-weight:600}td.n{text-align:right;font-variant-numeric:tabular-nums}
.tbl{overflow-x:auto}.note{border-left:3px solid var(--axis);padding:6px 12px;margin:14px 0;color:var(--ink2);font-size:14px}
.ok{color:var(--good)}.bad{color:var(--crit)}
.key{border:1px solid var(--ring);border-radius:10px;background:var(--surface);padding:8px 14px;margin:14px 0}.key summary{font-size:14px;color:var(--ink2)}
.key table{font-size:13px}.key td:first-child{white-space:nowrap;font-weight:600;color:var(--ink)}.key code{font-size:12px}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--ring);border-radius:8px;
padding:8px 10px;font-size:13px;box-shadow:0 4px 16px rgba(0,0,0,.12);display:none;max-width:320px;z-index:9}
a.g{color:inherit;text-decoration:underline dotted;text-underline-offset:2px}.gl dt{font-weight:600;margin-top:10px}.gl dd{margin:2px 0 0 0;color:var(--ink2)}
#tip b{display:block;font-size:15px}footer{max-width:980px;margin:0 auto;padding:0 16px 40px;color:var(--muted);font-size:13px}
"""

JS = """
const tip=document.getElementById('tip');
function show(e,el){const [v,l]=el.dataset.tip.split('||');tip.replaceChildren();const b=document.createElement('b');b.textContent=v;
tip.append(b,document.createTextNode(l||''));tip.style.display='block';const r=el.getBoundingClientRect();
const x=(e&&e.clientX)||r.left+r.width/2,y=(e&&e.clientY)||r.top;tip.style.left=Math.min(x+14,innerWidth-330)+'px';tip.style.top=(y+14)+'px';}
document.querySelectorAll('[data-tip]').forEach(el=>{el.addEventListener('pointermove',e=>show(e,el));
el.addEventListener('focus',()=>show(null,el));['pointerleave','blur'].forEach(t=>el.addEventListener(t,()=>tip.style.display='none'));});
"""

def legend(cfgs):
    """One entry per engine series present, listing the engine labels it covers; colours match the chart marks."""
    by = {}
    for c in cfgs:
        by.setdefault(c["engine"]["series"], {})[engine_label(c, with_name=False)] = c
    items = []
    for ser in sorted(by, key=list(SERIES).index):
        cs = sorted(by[ser].values(), key=lambda c: (c["engine"]["version"], len(c["engine"]["patches"])))
        text = cs[0]["engine"]["name"] + " " + ", ".join(engine_label(c, with_name=False) for c in cs)
        items.append(f'<span><i style="background:{SERIES[ser]}"></i>{esc(text)}</span>')
    return '<div class="legend">' + "".join(items) + "</div>"


def key(cfgs, open_=False):
    """What every label on the site means, generated from the configs."""
    eng, wts, spec = {}, {}, {}
    for c in cfgs:
        eng.setdefault(engine_label(c), c); wts.setdefault(weights_label(c), c)
        sp = c["serving"]["speculative"]
        if sp["tokens"]: spec.setdefault((sp["label"], sp["draft_model"]), 1)
    rows = []
    for lab, c in sorted(eng.items(), key=lambda t: (list(SERIES).index(t[1]["engine"]["series"]), t[1]["engine"]["version"], len(t[1]["engine"]["patches"]))):
        e = c["engine"]; proj = f'<a href="https://github.com/{esc(e["project"])}">{esc(e["project"])}</a>'
        digest = f'<code>{esc(e["image"].rsplit("@", 1)[-1][:19])}</code>'
        if e["patches"]:
            txt = f'{esc(e["version"])} of {proj} (built on {esc(e["built_on"])})'
            for pt in e["patches"]:
                m = re.match(r"https://github.com/([^/]+/[^/]+)/pull/(\d+)", pt["source"])
                src = f'<a href="{esc(pt["source"])}">{esc(m.group(1) + "#" + m.group(2)) if m else "source"}</a>'
                ports = " and ".join(esc(x) for x in pt.get("ports", []))
                txt += (f' with the {esc(pt["name"])} applied locally' + (f': {ports}, as ported in {src} ' if ports else f', from {src} ')
                        + f'(commit {esc(pt["commit"])})')
            txt += f'. A local build on the {esc(e["version"])} image {digest}, not a release'
            txt += (f'. <b>The same engine as {esc(e["name"])} {esc(e["equivalent_to"]["version"].lstrip("v"))} for every measurement here</b>: '
                    f'{esc(e["equivalent_to"]["basis"])}.' if e.get("equivalent_to") else ', and not equivalent to any release.')
        else:
            txt = f'Release {esc(e["version"])} of {proj} (built on {esc(e["built_on"])}), as published: image {digest}.'
        rows.append([esc(lab), txt])
    for lab, c in sorted(wts.items()):
        m = c["model"]
        rows.append([esc(lab), f'Weights <a href="https://huggingface.co/{esc(m["repo"])}">{esc(m["repo"])}</a>: {esc(m["quant"])}.'])
    lays = {}
    for c in cfgs:
        lay = c["serving"].get("layout") or {}
        if lay.get("label"): lays.setdefault(lay["label"], (c, lay))
    for lab, (c, lay) in sorted(lays.items()):
        rows.append([esc(lab), f'Parallel layout that differs from the engine release\'s default: EP{c["serving"]["ep"]} routed experts, '
                     f'DCP{c["serving"]["dcp"]}, MLA layer ownership "{esc(str(lay.get("mla_owners")))}", NOPE records '
                     f'{"on" if lay.get("nope_records") else "off"}' + (", DCP top-k owner exchange off" if lay.get("dcp_topk_owner_exchange") is False else "")
                     + '. A screen arm or diagnostic control, not a recommendation; see the configuration notes.'])
    for (lab, dm) in spec:
        rows.append([f"{esc(lab)} \u00d7N", f'Speculative decoding with {esc(lab)}, N draft tokens per step (draft model {esc(dm)}). '
                     '"sharing off": draft-slot sharing disabled. "no speculation": plain decoding.'])
    eqs = [(lab, c) for lab, c in eng.items() if c["engine"].get("equivalent_to")]
    lead = "".join(f' <b>{esc(lab)}</b> is the same engine as <b>{esc(c["engine"]["name"])} {esc(c["engine"]["equivalent_to"]["version"].lstrip("v"))}</b> for these measurements.' for lab, c in eqs[:1])
    return (f'<details class="key"{" open" if open_ else ""}><summary><b>Labels</b> read <i>weights · engine version · speculation</i>.{lead} '
            f'What each label means</summary>{table(["Label", "Meaning"], rows)}<p style="font-size:13px">Labels are built from the '
            f'configuration files by one rule (<a href="{REPO}/blob/main/SCHEMA.md#labels">SCHEMA.md</a>): engine name first, then its own version, '
            f'then any locally applied patches.</p></details>')


def interval_plot(rows, lo, hi, ticks, fmt, unit=""):
    """rows: dicts with label, est, lo, hi, color, tip. Horizontal dot-and-whisker, one row per config."""
    lw, rh, top = label_width(r["label"] for r in rows), 34, 10
    pw = 440; w = lw + pw + 30
    x = lambda v: lw + (v - lo) / (hi - lo) * pw
    h = top + rh * len(rows) + 30
    s = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">']
    for t in ticks:
        s.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" y2="{h - 26}"/>'
                 f'<text x="{x(t):.1f}" y="{h - 10}" text-anchor="middle" class="muted">{fmt(t)}{unit}</text>')
    for i, r in enumerate(rows):
        y = top + rh * i + rh / 2
        c = r["color"]
        s.append(f'<text x="{lw - 12}" y="{y + 4:.1f}" text-anchor="end">{esc(r["label"])}</text>')
        s.append(f'<rect class="hit" x="{lw}" y="{y - rh / 2:.1f}" width="{pw}" height="{rh}" tabindex="0" '
                 f'data-tip="{esc(r["tip"])}"/><g>'
                 f'<line x1="{x(r["lo"]):.1f}" x2="{x(r["hi"]):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{c}" stroke-width="2" stroke-linecap="round"/>'
                 f'<circle cx="{x(r["est"]):.1f}" cy="{y:.1f}" r="5" fill="{c}" stroke="var(--surface)" stroke-width="2"/></g>')
        s.append(f'<text x="{x(r["hi"]) + 8:.1f}" y="{y + 4:.1f}" class="muted">{esc(fmt(r["est"]))}{unit}</text>')
    s.append("</svg>")
    return "".join(s)


def table(headers, rows, numeric=()):
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f'<td class="{"n" if j in numeric else ""}">{c}</td>' for j, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="tbl"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def figure(title, sub, body, tbl, legend_html=""):
    return (f'<figure><figcaption><div class="t">{esc(title)}</div><div class="s">{sub}</div></figcaption>'
            f'{legend_html}<div class="chart">{body}</div>'
            f'<details><summary>Show the numbers as a table</summary>{tbl}</details></figure>')


# ---------------------------------------------------------------- pages

PAGES = [("index.html", "Overview"), ("gpqa.html", "GPQA Diamond"), ("screens.html", "Hard-question screen"),
         ("serving.html", "Speed & acceptance"), ("kernels.html", "Kernel tests"), ("looping.html", "Looping investigation"), ("method.html", "How to read this")]


def with_key(body, key_html):
    """Label key right after the page's lede (or right after its heading if a page has none)."""
    i = body.find('class="lede"')
    j = body.find("</p>", i) + 4 if i >= 0 else body.find("</h1>") + 5
    return body[:j] + key_html + body[j:]


def page(name, title, body, key_html):
    body_html = with_key(body, key_html)
    if name != "method.html":
        body_html = gloss(body_html)
    nav = "".join(f'<a href="{f}" class="{"on" if f == name else ""}">{esc(t)}</a>' for f, t in PAGES)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(title)} - local-inference-evals</title><style>{CSS}</style></head><body>'
            f'<nav><span class="brand">local-inference-evals</span>{nav}</nav><main>{body_html}</main>'
            f'<footer>Generated from the repository data by <code>tools/site.py</code>. Receipts, protocols and code: '
            f'<a href="{REPO}">{REPO.replace("https://", "")}</a>. Results CC BY 4.0, code Apache-2.0.</footer>'
            f'<div id="tip" role="status"></div><script>{JS}</script></body></html>')


def gpqa_rows(runs):
    out = []
    for rid, m in runs.items():
        if m["protocol"] != "gpqa-diamond/v1":
            continue
        s = m["summary"]
        out.append(dict(rid=rid, label=label(m["cfg"]), cfg=m["cfg"], color=color(m["cfg"]), s=s, n=s["samples"]))
    return sorted(out, key=lambda r: (list(SERIES).index(r["cfg"]["engine"]["series"]), r["label"]))


def gpqa_accuracy_fig(rows, title):
    data = [dict(label=r["label"], color=r["color"], est=100 * r["s"]["accuracy_flexible"], lo=100 * r["s"]["accuracy_flexible_ci95"][0],
                 hi=100 * r["s"]["accuracy_flexible_ci95"][1],
                 tip=f'{100 * r["s"]["accuracy_flexible"]:.1f}% (95% CI {100 * r["s"]["accuracy_flexible_ci95"][0]:.1f}-{100 * r["s"]["accuracy_flexible_ci95"][1]:.1f})||'
                     f'{r["label"]}, {r["n"]} answers') for r in rows]
    body = interval_plot(data, 70, 100, [70, 75, 80, 85, 90, 95, 100], lambda v: f"{v:.0f}" if v == int(v) else f"{v:.1f}", "%")
    pc = lambda x: f"{100 * x:.1f}%"
    tbl = table(["Configuration", "Raw (flexible-extract)", "95% CI", "Raw, answered only", "Stated answer (audited)", "Stated, answered only", "Receipts"],
                [[esc(r["label"]), pc(r["s"]["accuracy_flexible"]),
                  f'{100 * r["s"]["accuracy_flexible_ci95"][0]:.1f}-{100 * r["s"]["accuracy_flexible_ci95"][1]:.1f}',
                  pc(r["s"]["accuracy_flexible_answered"]), pc(r["s"]["accuracy_stated"]), pc(r["s"]["accuracy_stated_answered"]), link(r["rid"])]
                 for r in rows], numeric=(1, 3, 4, 5))
    gap = [100 * (r["s"]["accuracy_stated"] - r["s"]["accuracy_flexible"]) for r in rows]
    return figure(title, "Pass 1 of each configuration, 198 questions, request seed 1234. Dot = raw lm-eval flexible-extract score (the headline); line = "
                  "95% bootstrap interval over questions. The audited stated-answer score (table) is "
                  f"{min(gap):.1f}-{max(gap):.1f} points higher, because the raw filter reads some correct answers as wrong. All nine runs share one "
                  "request seed, so this spread likely understates how much independent runs vary. Overlapping lines: the configurations cannot be told apart.",
                  body, tbl, legend([r["cfg"] for r in rows]))


def gpqa_empty_fig(rows, title):
    data = []
    for r in rows:
        k, n = r["s"]["empty"], r["n"]; lo, hi = wilson(k, n)
        data.append(dict(label=r["label"], color=r["color"], est=100 * k / n, lo=100 * lo, hi=100 * hi,
                         tip=f'{k} of {n} ({100 * k / n:.1f}%)||{r["label"]}; 95% Wilson {100 * lo:.1f}-{100 * hi:.1f}%'))
    body = interval_plot(data, 0, 12, [0, 3, 6, 9, 12], lambda v: f"{v:.1f}" if v != int(v) else f"{v:.0f}", "%")
    fr = lambda r: ", ".join(f"{v} {k}" for k, v in r["s"]["empty_by_finish_reason"].items()) or "-"
    tbl = table(["Configuration", "Empty answers", "Answers", "Finish reasons", "95% Wilson", "Receipts"],
                [[esc(r["label"]), r["s"]["empty"], r["n"], esc(fr(r)),
                  "{:.1f}-{:.1f}%".format(*(100 * v for v in wilson(r["s"]["empty"], r["n"]))), link(r["rid"])] for r in rows], numeric=(1, 2))
    return figure(title, "Share of answers that came back empty (reasoning never finished; scored wrong), pass 1 of each configuration. "
                  "Line = 95% Wilson interval. One draw per question: empties cluster on a few hard questions, so true uncertainty is wider.",
                  body, tbl, legend([r["cfg"] for r in rows]))


# ---------------------------------------------------------------- hard-prompt-screen v2

def v2_runs(runs):
    """{arm: run} for the component screen, in the preregistered order; the fixed-seed control last."""
    out = {}
    for m in sorted((m for m in runs.values() if m["protocol"] == "hard-prompt-screen/v2"), key=lambda m: m["id"]):
        out[m["notes"].split("arm ", 1)[1].split(":", 1)[0]] = m
    order = ["B", "EN", "EO", "NO", "B79", "V79", "S1234"]
    return {a: out[a] for a in order if a in out}


def is_control(m):
    return "fixed-seed control" in m["notes"]


OUTC = [("ok", "Finished", "var(--seq2)"), ("loop", "Loop", "var(--s2)"), ("exhaust", "Exhaustion", "var(--s4)")]


def outcome_bars(items, title, sub):
    """items: dicts label, rows, rid, ci (bool). Stacked bars of 12 draws (finished / loop / exhaust) with a Wilson whisker
    for the failures (loop + exhaust) when ci. One bar per arm and question; never pooled across questions."""
    lw, rh, top = label_width(i["label"] for i in items), 34, 10; pw = 420; w = lw + pw + 70
    n_max = max(len(i["rows"]) for i in items)
    x = lambda v: lw + v / n_max * pw
    h = top + rh * len(items) + 30
    s = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">']
    for t in range(0, n_max + 1, 3):
        s.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" y2="{h - 26}"/>'
                 f'<text x="{x(t):.1f}" y="{h - 10}" text-anchor="middle" class="muted">{t}</text>')
    tr = []
    for i, it in enumerate(items):
        y = top + rh * i; rs = it["rows"]; n = len(rs)
        c = {k: sum(r["cls"] == k for r in rs) for k, _, _ in OUTC}; fail = c["loop"] + c["exhaust"]
        s.append(f'<text x="{lw - 12}" y="{y + rh / 2 + 4:.1f}" text-anchor="end">{esc(it["label"])}</text>')
        # failures first (from the left), then finished
        x0 = 0
        for k, name, col in (OUTC[1], OUTC[2], OUTC[0]):
            if c[k]:
                s.append(f'<rect x="{x(x0):.1f}" y="{y + 8}" width="{x(x0 + c[k]) - x(x0) - 1:.1f}" height="{rh - 16}" rx="3" style="fill:{col}"/>')
                x0 += c[k]
        lo, hi = wilson(fail, n)
        tipci = f"; 95% Wilson {100 * lo:.0f}-{100 * hi:.0f}%" if it["ci"] else " (fixed-seed control, not a rate)"
        s.append(f'<rect class="hit" x="{lw}" y="{y}" width="{pw}" height="{rh}" tabindex="0" data-tip="{fail} of {n} failed||'
                 f'{esc(it["label"])}: {c["loop"]} loops, {c["exhaust"]} exhaustions, {c["ok"]} finished{tipci}"/><g></g>')
        if it["ci"]:
            s.append(f'<line x1="{x(lo * n):.1f}" x2="{x(hi * n):.1f}" y1="{y + rh / 2:.1f}" y2="{y + rh / 2:.1f}" stroke="var(--ink)" stroke-width="1.5"/>'
                     f'<line x1="{x(lo * n):.1f}" x2="{x(lo * n):.1f}" y1="{y + rh / 2 - 5:.1f}" y2="{y + rh / 2 + 5:.1f}" stroke="var(--ink)" stroke-width="1.5"/>'
                     f'<line x1="{x(hi * n):.1f}" x2="{x(hi * n):.1f}" y1="{y + rh / 2 - 5:.1f}" y2="{y + rh / 2 + 5:.1f}" stroke="var(--ink)" stroke-width="1.5"/>')
        s.append(f'<text x="{x(n) + 8:.1f}" y="{y + rh / 2 + 4:.1f}" class="muted">{fail}/{n}</text>')
        tr.append([esc(it["label"]), c["ok"], c["loop"], c["exhaust"], f"{fail}/{n}",
                   f"{100 * lo:.0f}-{100 * hi:.0f}%" if it["ci"] else "control, not a rate", link(it["rid"])])
    s.append("</svg>")
    leg = ('<div class="legend">' + "".join(f'<span><i style="background:{col};border-radius:2px"></i>{name}</span>' for k, name, col in (OUTC[1], OUTC[2], OUTC[0]))
           + '<span>&#9474;&#8212;&#9474; 95% Wilson interval of failures</span></div>')
    return figure(title, sub, "".join(s), table(["Arm", "Finished", "Loop", "Exhaustion", "Failed", "95% Wilson", "Receipts"], tr, numeric=(1, 2, 3, 4)), leg)


def screen_caveats(v2):
    """What the component screen held fixed, what it did not test, and how large an effect it could see."""
    k = {a: sum(r["cls"] in ("loop", "exhaust") for r in m["rows"]) for a, m in v2.items()}
    base88 = sum(k[a] for a in analyze.ARMS88) / (12 * len(analyze.ARMS88))
    m1 = analyze.mdd(k["B"] / 12, 12, 12); m2 = analyze.mdd(base88, 24, 24); m3 = analyze.mdd(k["B79"] / 12, 12, 12)
    f = lambda x: "not reachable at all" if x is None else f"about {100 * x:.0f} points"
    return ('<div class="note"><b>Scope.</b> Held fixed in every arm: 3.25bpw weights, DFlash2 ×3, temperature 1.0 / top_p 0.95, the 327,680-token '
            'budget, 12 concurrent requests, one question per arm, a fresh server. Arms EO and NO also turned draft-slot sharing off (the launcher pairs '
            'it with MLA ownership <code>tp</code>), so ownership and sharing are not separated. Arm V79 differs from B79 in image, layout, vision and '
            'model revision together. Draft depth, quantization and sampling settings were not varied. '
            f'<b>Power</b> (two-sided Fisher test, p &lt; 0.05, 80% power; <code>tools/analyze.py screen-power</code>): one arm against another on '
            f'question 88 (12 vs 12, from {100 * k["B"] / 12:.0f}% failing) detects a rise of {f(m1[1])}, and a drop is {f(m1[0])}; a switch on '
            f'vs off (24 vs 24, from {100 * base88:.0f}%) detects a rise of {f(m2[1])}; question 79 (12 vs 12, from {100 * k["B79"] / 12:.0f}%) '
            f'detects a drop of {f(m3[0])}. Only very large effects could show. <b>Selection.</b> Questions 88 and 79 were chosen as hard from '
            f'empty answers and screens that all sent request seed 1234; with distinct seeds, question 88 fails in {min(k[a] for a in analyze.ARMS88)}-'
            f'{max(k[a] for a in analyze.ARMS88)} of 12 draws per arm. Two questions are not a benchmark-wide rate.</div>')


def arm_label(arm, m):
    return f'{arm} · q{m["rows"][0]["doc_id"]} · {label(m["cfg"])}'


def component_fig(v2):
    items = [dict(label=arm_label(a, m), rows=m["rows"], rid=m["id"], ci=True) for a, m in v2.items() if not is_control(m)]
    return outcome_bars(items, "Component screen, 2026-10-09: outcomes per arm and question (12 draws each)",
                        "Each bar is one question in one configuration, 12 repeats with distinct request seeds 5001-5012, 12 concurrent requests, "
                        "fresh server. Question 88: arms B, EN, EO, NO; question 79: B79 (tpurtell 0.9.1) and V79 (the tpurtell 0.7.0 image at three draft "
                        "tokens). Counts are never pooled across questions.")


def seed_fig(v2):
    if "S1234" not in v2:
        return ""
    items = [dict(label="B · seeds 5001-5012", rows=v2["B"]["rows"], rid=v2["B"]["id"], ci=True),
             dict(label="S1234 · seed 1234 on every repeat", rows=v2["S1234"]["rows"], rid=v2["S1234"]["id"], ci=False)]
    kb, ks = (sum(r["cls"] in ("loop", "exhaust") for r in v2[a]["rows"]) for a in ("B", "S1234"))
    return outcome_bars(items, "One seed on every repeat vs a distinct seed per repeat (question 88, same configuration)",
                        f"{esc(label(v2['B']['cfg']))}, question 88 x 12, 12 concurrent, fresh server; only the request seeds differ. "
                        f"Failed {kb} of 12 with distinct seeds vs {ks} of 12 with seed 1234 on every repeat (Fisher p = {fisher(kb, 12 - kb, ks, 12 - ks):.3f}). "
                        "The fixed-seed arm is a control for the seed, not a measurement of the configuration.")


def onset_hist(v2):
    """Where the early stop fired, in estimated tokens; the 2,044-token tail-bug region marked."""
    ctrl = {a for a, m in v2.items() if is_control(m)}
    cpq = {d: analyze.chars_per_token([r for a, m in v2.items() if a not in ctrl for r in m["rows"] if r["doc_id"] == d])
           for d in {m["rows"][0]["doc_id"] for m in v2.values()}}
    pts = [(r["reasoning_chars"] / cpq[r["doc_id"]], a in ctrl, a, r["doc_id"]) for a, m in v2.items() for r in m["rows"] if r["stopped_early"]]
    bw, hi = 20000, 340000; nb = hi // bw
    bins = [[0, 0] for _ in range(nb)]
    for t, c, _, _ in pts:
        bins[min(int(t // bw), nb - 1)][c] += 1
    lw, top, ph, pw = 40, 30, 170, 600; w, h = lw + pw + 20, top + ph + 44
    ymax = max(sum(b) for b in bins) or 1
    x = lambda v: lw + v / hi * pw; y = lambda v: top + ph - v / ymax * ph
    s = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">']
    for t in range(0, hi + 1, 40000):
        s.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" y2="{top + ph}"/>'
                 f'<text x="{x(t):.1f}" y="{top + ph + 16}" text-anchor="middle" class="muted">{t // 1000}k</text>')
    for v in range(0, ymax + 1, max(1, ymax // 4)):
        s.append(f'<text x="{lw - 8}" y="{y(v) + 4:.1f}" text-anchor="end" class="muted">{v}</text>')
    for i, (a, b) in enumerate(bins):
        for k, (cnt, col, base) in enumerate(((a, "var(--s1)", 0), (b, "var(--axis)", a))):
            if cnt:
                s.append(f'<rect x="{x(i * bw) + 1:.1f}" y="{y(base + cnt):.1f}" width="{x(bw) - x(0) - 2:.1f}" height="{y(base) - y(base + cnt):.1f}" rx="2" style="fill:{col}"/>')
        if a + b:
            s.append(f'<rect class="hit" x="{x(i * bw):.1f}" y="{top}" width="{x(bw) - x(0):.1f}" height="{ph}" tabindex="0" '
                     f'data-tip="{a + b} loops||about {i * bw // 1000}k-{(i + 1) * bw // 1000}k tokens: {a} distinct-seed, {b} fixed-seed control"/><g></g>')
    s.append(f'<line x1="{x(2044):.1f}" x2="{x(2044):.1f}" y1="{top - 22}" y2="{top + ph}" stroke="var(--crit)" stroke-width="2"/>'
             f'<text x="{x(2044) + 6:.1f}" y="{top - 10}" style="fill:var(--crit)">2,044 tokens: the tail bug acted only before this</text>'
             f'<text x="{lw + pw / 2}" y="{h - 6}" text-anchor="middle" class="muted">estimated tokens generated when the early stop fired</text>')
    s.append("</svg>")
    tr = [[esc(a), f"q{d}", f"{t:,.0f}", "fixed-seed control" if c else "distinct seeds"] for t, c, a, d in sorted(pts)]
    leg = ('<div class="legend"><span><i style="background:var(--s1);border-radius:2px"></i>distinct-seed arms</span>'
           '<span><i style="background:var(--axis);border-radius:2px"></i>fixed-seed control (S1234)</span></div>')
    return figure("Where loops were stopped", "Every loop that the early-stop detector stopped in the component screen. Tokens are estimated from "
                  "reasoning characters with each question's median characters per completion token in its finished distinct-seed requests "
                  f"({', '.join(f'q{d}: {v:.2f}' for d, v in sorted(cpq.items()))}). The detector stops a short-period loop within about 60,000 "
                  "characters of its start, so every loop here began far beyond the first 2,044 tokens.", "".join(s),
                  table(["Arm", "Question", "Estimated tokens at the stop", "Seeds"], tr, numeric=(2,)), leg)


def single_draw_table(runs):
    docs = [13, 79, 88, 121, 127]
    rows = []
    for m in sorted((m for m in runs.values() if m["protocol"] in ("hard-prompt-screen/v0", "hard-prompt-screen/v1")),
                    key=lambda m: (m["date"], label(m["cfg"]))):
        o = {r["doc_id"]: r["cls"] for r in m["rows"]}
        mark = lambda c: {"loop": '<span class="bad">loop</span>', "exhaust": '<span class="bad">exhaust</span>', "error": "error"}.get(c, c)
        rows.append([esc(label(m["cfg"])), m["date"][5:]] + [mark(o.get(d, "-")) for d in docs] + [link(m["id"])])
    return table(["Configuration", "Date"] + [f"q{d}" for d in docs] + ["Receipts"], rows)


WITHDRAWAL = ('<div class="note"><b>Withdrawal notice (2026-10-09).</b> Every repeat of the earlier hard-question screens (protocols v0 and v1, '
              '2026-09-30 to 2026-10-08) sent request seed 1234. Batching made the repeats vary, but every repeat drew on the same sampler noise, so that variation was not '
              'statistically meaningful. Their failure rates and intervals, '
              'the layout-bisection statistics and the question-level observations drawn from them are withdrawn. Only repeat 1 of each question '
              'from each configuration\'s first screen is kept, as a single draw. The withdrawn rows remain in the git history. Details and evidence: '
              '<a href="{inv}#4-method-note-on-seeds-and-withdrawal-notice">the investigation</a>.</div>')


# ---------------------------------------------------------------- looping investigation page

LOOP_CFG = {"as": "glm53-flash/k3.25-v0.9.0-dflash3", "fix": "glm53-flash/k3.25-v0.9.0-tailfix-dflash3"}
INV = REPO + "/tree/main/investigations/2026-10-glm53-looping"


def fisher(a, b, c, d):
    return analyze.fisher(a, b, c, d)


def decode_prefill_fig(runs):
    dps = sorted((m for m in runs.values() if m["protocol"] == "decode-prefill-consistency/v1"), key=lambda m: (m["config"], m["id"]))
    kd, ktr = [], []
    for reg, rname in (("lt2044", "i < 2,044"), ("ge2048", "i ≥ 2,048")):
        for m in dps:
            per = []
            for d in range(6):
                v = [r["kl_top20"] for r in m["rows"] if r["doc_id"] == d and r["region"] == reg and r["kl_top20"] is not None]
                per.append(sum(v) / len(v))
            allv = [r["kl_top20"] for r in m["rows"] if r["doc_id"] < 6 and r["region"] == reg and r["kl_top20"] is not None]
            est = sum(allv) / len(allv)
            lab = f"{rname} · {engine_label(m['cfg'])}{' · spec on' if m['cfg']['serving']['speculative']['tokens'] else ''} · {m['date'][5:]}"
            kd.append(dict(label=lab, color=color(m["cfg"]), est=est, lo=min(per), hi=max(per),
                           tip=f"KL {est:.4f}||{lab}; prompts 0-5; per-prompt range {min(per):.4f}-{max(per):.4f}"))
            ktr.append([esc(lab), f"{est:.4f}", f"{min(per):.4f}-{max(per):.4f}", link(m["id"])])
    return figure("Decode vs prefill re-scoring of the same tokens", "3.25bpw, prefix caching off, concurrency 1, GPQA prompts 0-5 (shared by every run; "
                  "the tpurtell 0.9.1 runs also cover prompts 6-11) x 2,600 decoded tokens. Dot = mean KL over the shared top-20 tokens (lower = decode "
                  "agrees with prefill); line = range of the six prompts' means. One run without the fix; three with it (the local build and two runs of "
                  "the 0.9.1 release, whose difference is the run-to-run spread), plus 0.9.1 with speculation on. Below 2,044 tokens the run without "
                  "the fix sits far above every run with it; from 2,048 tokens the runs with the fix themselves spread widely.",
                  interval_plot(kd, 0, 0.12, [0, 0.03, 0.06, 0.09, 0.12], lambda v: "0" if v == 0 else f"{v:.4f}".rstrip("0"), ""),
                  table(["Positions · engine · date", "Mean KL, prompts 0-5", "Per-prompt range", "Receipts"], ktr, numeric=(1,)),
                  legend([m["cfg"] for m in dps]))


def looping(runs):
    by = lambda proto, cfg: sorted((m for m in runs.values() if m["protocol"] == proto and m["config"] == cfg), key=lambda m: m["id"])
    v2 = v2_runs(runs)
    # decode vs prefill: every run, on prompts 0-5 (the prompts all runs share)
    fig_kl = decode_prefill_fig(runs)
    # index check
    ix = {c: by("kpool-tail-index/v1", c)[0] for c in (LOOP_CFG["as"], LOOP_CFG["fix"])}
    ia, ib = (sorted((r for r in ix[c]["rows"] if r["layout"] == "packed"), key=lambda r: r["length"]) for c in ix)
    itr = [[r["length"], ", ".join(map(str, r["tail"])) or "-", '<span class="bad">dropped</span>' if not r["tail_attended"] else "attended",
            '<span class="ok">attended</span>' if q["tail_attended"] else '<span class="bad">dropped</span>', "yes" if r["row_sha256_16"] == q["row_sha256_16"] else "no"]
           for r, q in zip(ia, ib) if r["length"] <= 2052]
    # tool eval
    te = {c: by("tool-eval-bench/v1", c)[0] for c in (LOOP_CFG["as"], LOOP_CFG["fix"])}
    stt = {c: {(r["rep"], r["scenario_id"]): r["status"] for r in m["rows"]} for c, m in te.items()}
    ids = sorted({sid for _, sid in stt[LOOP_CFG["as"]]})
    diff = [sid for sid in ids if len({stt[c][(rp, sid)] for c in te for rp in (1, 2)}) > 1]
    ttr = [[esc(sid)] + [stt[c][(rp, sid)] for c in te for rp in (1, 2)] for sid in diff]
    pts = [[esc(label(m["cfg"]))] + [f'{m["summary"]["reps"][str(rp)]["points"]}/{m["summary"]["reps"][str(rp)]["max_points"]}' for rp in (1, 2)] + [link(m["id"])]
           for m in te.values()]
    eh = lambda c: esc(engine_label(te[c]["cfg"]))
    kv = lambda m: re.search(r"KV pool ([\d,]+) tokens", m["notes"]).group(1)
    s1 = by("hard-prompt-screen/v1", LOOP_CFG["as"])[0]; s7 = by("hard-prompt-screen/v1", "glm53-flash/k3.25-v0.9.0-ep2dcp2-dflash3")[0]
    kb = {a: sum(r["cls"] in ("loop", "exhaust") for r in m["rows"]) for a, m in v2.items()}
    summary = table(["Conclusion", "Strength"], [
        ["Under tpurtell 0.8.0/0.9.0's DCP1 layout, decode attention skipped the newest 1-3 tokens at causal lengths up to 2,043 not divisible by 4; "
         "the fix (tpurtell PR #6, released in 0.9.1) removes it", "<b>Supported</b> (index check on the image's own kernels)"],
        ["With the fix, decode agrees much better with prefill below 2,044 tokens (KL 0.066 → 0.006-0.010 across three runs with the fix)",
         "<b>Supported</b> for this large effect (one run without the fix)"],
        ["The fix's effect on answers and tool calls", "<b>Unmeasured</b>: GPQA +0.5 points in one pass per release; tool-eval-bench TC-80 and "
         "TC-88 pass in both repeats with it, but ten other scenarios flip between repeats of one build"],
        [f"Repeats that share one request seed are not a meaningful sample: question 88 failed {kb.get('S1234', '-')} of 12 with seed 1234 on every "
         f"repeat vs {kb['B']} of 12 with distinct seeds", "<b>Supported</b>; the reason the earlier screens were withdrawn"],
        [f"On tpurtell 0.9.1, question 88 fails in 1-3 of 12 draws in every tested arm; question 79 in {kb['B79']} of 12 on 0.9.1 and {kb['V79']} of 12 "
         "on the 0.7.0 image at three draft tokens", "Descriptive"],
        ["None of the tested runtime parts moved either question at this size", "Descriptive; only very large effects were detectable "
         "(preregistered rules: no candidate / unresolved); draft depth, quantization and sampling untested"],
        ["Loops are stopped far beyond the 2,044-token tail-bug region; question 88 loops, question 79 mostly exhausts", "Descriptive"],
        ["What drives non-completion on these questions", "Open"]])
    return ('<h1>What drives non-completion on hard questions</h1>'
            '<p class="lede">GLM-5.3-Flash on 2x RTX PRO 6000, served by tpurtell\'s engine: on some hard GPQA questions the model does not finish '
            'its reasoning within the 327,680-token budget. Much of this is not yet conclusive; this page states what the clean data supports and '
            'will be updated as clean data arrives. Write-up, glossary, open questions and commands to recompute every number: '
            f'<a href="{INV}">investigations/2026-10-glm53-looping</a>.</p>'
            + '<h2>1. Summary</h2>' + summary
            + '<h2>2. Defect found and fixed: the DCP1 tail bug</h2>'
              '<p>For each decode step the sparse-attention indexer selects up to 2,048 earlier tokens in pools of 4; the newest, incomplete pool '
              '(the current token and up to two before it) sits in fixed columns 2044-2046. In the DCP1 branch the selection length becomes '
              'min(causal length, 2,048) and every column beyond it is masked, so at causal lengths up to 2,043 that are not a multiple of 4 every '
              'MLA layer missed those tokens. DCP1 with MLA layer ownership is tpurtell\'s layout choice for 0.8.0 onward and frees KV memory '
              f'({kv(s1)} tokens against {kv(s7)} for v0.7.0\'s layout on the same image); the masking path already existed in the vendored attention '
              'code, and only DCP1 exercises it. The fix, <a href="https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/6">tpurtell PR #6</a>, '
              'compacts the valid entries before the mask; it was merged on 2026-10-08 and released in tpurtell 0.9.1. The measurements below were '
              'taken on a local build of the fix before the merge (<i>tpurtell 0.9.0 + DCP1 tail fix ≈ 0.9.1</i>), and the decode-vs-prefill '
              'repeat on the 0.9.1 release. <b>Practical impact on answers is unmeasured:</b> 4bpw TR3 (Brandon) scored 0.5 points higher on 0.9.1 '
              'than on 0.9.0 in one GPQA pass each (well inside noise), and the tool-calling result below rests on two repeats per build.</p>'
            + '<h2>Which tokens a decode step attends (index check, the image\'s own kernels on GPU)</h2>'
            + table(["Causal length", "Tail tokens", eh(LOOP_CFG["as"]), eh(LOOP_CFG["fix"]), "Same row bytes"], itr, numeric=(0,))
            + fig_kl
            + '<h2>Tool calling (tool-eval-bench, 88 scenarios, temperature 0, two repeats)</h2>'
            + table(["Configuration", "Repeat 1", "Repeat 2", "Receipts"], pts)
            + '<p>Scenarios whose status differs anywhere. TC-80 and TC-88 fail in both repeats without the fix and pass in both with it; the '
              'others also vary between repeats of the same build.</p>'
            + table(["Scenario"] + [f"{eh(c)}, repeat {rp}" for c in te for rp in (1, 2)], ttr)
            + '<h2>3. What clean data shows about non-completion</h2>'
            + component_fig(v2)
            + '<p>Question 88: no layout switch (EP2 routed experts, NOPE records off, MLA ownership tp) met the preregistered screening rule; '
              'question 79: the 0.7.0 image at three draft tokens failed as often as 0.9.1 (rule verdict: unresolved). Clean 3-pass GPQA runs with a '
              'distinct request seed per pass are in progress for tpurtell 0.7.0 and 0.9.1, as records per recipe, not as a looping evaluator.</p>'
            + screen_caveats(v2)
            + '<p><b>Single draws from the earlier screens.</b> Repeat 1 of each question from each configuration\'s first fixed-seed screen: '
              'loops and exhaustion occur in every configuration tested. One draw per question, not a rate.</p>'
            + single_draw_table(runs)
            + '<h2>4. Method note on seeds</h2>' + seed_fig(v2) + WITHDRAWAL.format(inv=INV)
            + '<p>The rule now (<a href="' + REPO + '/blob/main/protocols/hard-prompt-screen/v2.md">hard-prompt-screen/v2</a>): a distinct request '
              'seed per repeat, one question per run, and the question as the unit of analysis.</p>'
            + '<h2>5. Anatomy of a failure</h2><p>Distinct-seed component screen only; the fixed-seed control is shown separately. Question 88 '
              'fails by looping; question 79 mostly by exhaustion (varied reasoning to the budget). No finished or exhausted request was stopped '
              'by the early-stop detector.</p>'
            + onset_hist(v2)
            + f'<h2>6. Open questions</h2><p>Each stated with its evidence, in the <a href="{INV}#6-open-questions">write-up</a>: what drives '
              'non-completion on questions 88 and 79; whether tpurtell 0.7.0 at its shipped five draft tokens leaves fewer empty GPQA answers (0 and 2 '
              'of 198 in pass 1, against 3-9 for every later build; the question-79 screen ran 0.7.0 at three); whether the tail fix changes answers; '
              'why the engine is not bitwise reproducible even one request at a time; whether these quants cost accuracy against a higher-precision '
              'reference (none was run on this hardware); and three code-level questions.</p>')


def build():
    runs = load_runs(); OUT.mkdir(exist_ok=True)
    first = {}
    for m in sorted(runs.values(), key=lambda m: (m["date"], m["cfg"]["engine"]["series"])):
        first.setdefault(m["cfg"]["engine"]["series"], m["date"])
    SERIES.clear(); SERIES.update((ser, PALETTE[i % len(PALETTE)]) for i, ser in enumerate(sorted(first, key=lambda x: (first[x], x))))
    cfgs = list({m["config"]: m["cfg"] for m in runs.values()}.values())
    key_open, key_closed = key(cfgs, open_=True), key(cfgs)
    allg = gpqa_rows(runs)
    accs = [100 * r["s"]["accuracy_flexible"] for r in allg]
    g = {m["config"].split("/", 1)[1]: m for m in runs.values() if m["protocol"] == "gpqa-diamond/v1"}
    pairs = [(a, b, what, analyze.paired(g[a], g[b])) for a, b, what in analyze.PAIRS]
    min_p = min(t[3]["p_acc"] for t in pairs)
    kern = {rid: m for rid, m in runs.items() if m["protocol"] == "kpool-kernel-tests/v1"}
    up = lambda s: (sum(1 for r in s if r["suite"] == "upstream" and r["outcome"] == "passed"), sum(1 for r in s if r["suite"] == "upstream" and r["outcome"] != "skipped"))
    up_un = up(next(m["rows"] for m in kern.values() if not m["cfg"]["engine"]["patches"]))
    up_pa = up(next(m["rows"] for m in kern.values() if m["cfg"]["engine"]["patches"]))
    dpk = {m["config"]: m["summary"]["by_region"]["lt2044"]["mean_kl_top20"] for m in runs.values()
           if m["protocol"] == "decode-prefill-consistency/v1" and m["config"].startswith("glm53-flash/k3.25-v0.9.0")}
    kl0, kl1 = dpk["glm53-flash/k3.25-v0.9.0-nospec-nocache"], dpk["glm53-flash/k3.25-v0.9.0-tailfix-nospec-nocache"]
    stated = [100 * r["s"]["accuracy_stated"] for r in allg]
    e07 = sorted(r["s"]["empty"] for r in allg if r["cfg"]["engine"]["version"] == "v0.7.0")
    elater = sorted(r["s"]["empty"] for r in allg if r["cfg"]["engine"]["version"] != "v0.7.0")
    v2 = v2_runs(runs); kb = {a: sum(r["cls"] in ("loop", "exhaust") for r in m["rows"]) for a, m in v2.items()}
    gap = [100 * (r["s"]["accuracy_stated"] - r["s"]["accuracy_flexible"]) for r in allg]
    par = json.loads((ROOT / "comparisons/glm53-flash-serving-probe/parity.json").read_text())
    floor = next(pp for pp in par["pairs"] if pp["a"].split("#")[0] == pp["b"].split("#")[0])

    # ---- index
    tiles = (f'<div class="tiles"><div class="tile"><div class="lab">Decode vs prefill KL below 2,044 tokens</div>'
             f'<div class="val">{kl0:.3f} → {kl1:.3f}</div><div class="sub">without vs with the DCP1 tail fix (released in tpurtell 0.9.1); one run without</div></div>'
             f'<div class="tile"><div class="lab">Upstream kpool regression tests</div><div class="val">{up_pa[0]}/{up_pa[1]}</div>'
             f'<div class="sub">with the kpool fixes (without: {up_un[0]}/{up_un[1]})</div></div>'
             f'<div class="tile"><div class="lab">GPQA Diamond accuracy, pass 1</div>'
             f'<div class="val">{min(accs):.1f}-{max(accs):.1f}%</div><div class="sub">{len(allg)} configurations, none distinguishable from another</div></div></div>')
    idx = (f'<h1>GLM-5.3-Flash on 2x RTX PRO 6000: end-to-end receipts</h1>'
           f'<p class="lede">Accuracy, completion, speed and kernel-correctness measurements of locally served GLM-5.3-Flash EXL3 quants '
           f'on tpurtell\'s vLLM-based engine releases and locally patched builds of them, under fixed, versioned protocols. Every number links to raw '
           f'per-item receipts and can be recomputed with the repository\'s tools (<a href="method.html#receipts">what has no row-level receipt</a>).</p>'
           + '<h2>What it shows</h2><ol>'
           '<li><b>A decode masking path in tpurtell 0.8.0/0.9.0\'s DCP1 layout skipped the newest 1-3 tokens</b> at causal lengths up to 2,043 '
           f'that are not a multiple of 4 (supported: index check on the image\'s own kernels). The fix (tpurtell PR #6, released in 0.9.1) removes it, '
           f'and decode then agrees much better with prefill below 2,044 tokens (KL {kl0:.3f} → {kl1:.3f}; two runs of the 0.9.1 release give 0.006-0.008 '
           'on the same prompts). Its effect on answers is unmeasured. See the <a href="looping.html">investigation</a>.</li>'
           '<li><b>The two upstream kpool bugs were present in the 0.7.0 and 0.8.0 images</b>; upstream\'s regression tests pass with the fixes '
           f'({up_pa[0]}/{up_pa[1]}, from {up_un[0]}/{up_un[1]}). The 0.9.0 and 0.9.1 release images pass them too (tested '
           '2026-10-09).</li>'
           f'<li><b>On two hard questions, non-completion varies by question</b> (descriptive): on tpurtell 0.9.1, question 88 fails in 1-3 of 12 draws '
           f'in every tested arm and question 79 in {kb["B79"]} of 12, the same as on the tpurtell 0.7.0 image at three draft tokens ({kb["V79"]} of 12). '
           'None of the tested runtime parts moved either question, but with 12 draws per arm only very large effects could show. Both questions '
           'were chosen as hard from seed-1234 data; they are not a benchmark-wide rate.</li>'
           f'<li><b>GPQA accuracy does not separate the configurations</b>: pass 1 of every configuration lands at {min(accs):.1f}-{max(accs):.1f}% raw '
           f'({min(stated):.1f}-{max(stated):.1f}% by the audited stated answer), and every question-paired comparison is consistent with noise '
           f'(McNemar p &ge; {min_p:.2f}). All passes used request seed 1234, so the configurations share sampler noise and this range likely '
           'understates independent run-to-run spread; clean passes 2-3 with per-pass seeds are in progress.</li>'
           f'<li><b>Open</b>, among others: the two tpurtell 0.7.0 builds left {e07[0]} and {e07[1]} of 198 GPQA answers empty in pass 1, every later '
           f'build {elater[0]}-{elater[-1]}; one pass each does not settle whether that differs. No higher-precision reference was run on this hardware, '
           'so whether these quants cost accuracy is not measured here.</li>'
           '</ol><p>Full write-up: <a href="' + REPO + '/blob/main/FINDINGS.md">FINDINGS.md</a>.</p>'
           + key_open + tiles + WITHDRAWAL.format(inv=INV)
           + component_fig(v2)
           + gpqa_accuracy_fig(allg, "GPQA Diamond accuracy (pass 1, 198 questions)"))
    (OUT / "index.html").write_text(page("index.html", "Overview", idx, ""))

    # ---- gpqa
    ptr = [[esc(label(g[a]["cfg"])), esc(label(g[b]["cfg"])), esc(what), f'{r["only_a"]} / {r["only_b"]}', f'{r["p_acc"]:.2f}',
            f'{r["diff"]:+.1f} ({r["lo"]:+.1f} to {r["hi"]:+.1f})', f'{r["empty_only_a"]} / {r["empty_only_b"]}', f'{r["p_empty"]:.2f}']
           for a, b, what, r in pairs]
    pub = table(["Source", "Weights", "GPQA Diamond", "Stated protocol"],
                [["NVIDIA model card", "BF16", "92.17", "temp 1.0, top_p 0.95, 327,680 max new tokens; harness not stated"],
                 ["NVIDIA model card", "NVFP4", "92.11", "same"],
                 ["Red Hat model card", "NVFP4", "90.57", "lm-eval / lighteval forks, vLLM, 3 seeds averaged"],
                 ["This repository", "EXL3 3.25bpw / 4bpw", f"{min(accs):.1f}-{max(accs):.1f}",
                  "pass 1 of each configuration (95% intervals {:.1f}-{:.1f}); see the GPQA protocol".format(
                      min(100 * r["s"]["accuracy_flexible_ci95"][0] for r in allg), max(100 * r["s"]["accuracy_flexible_ci95"][1] for r in allg))]])
    gp = ('<h1>GPQA Diamond</h1><p class="lede">198 graduate-level multiple-choice questions, lm-evaluation-harness, temperature 1.0, '
          'top_p 0.95, 327,680-token budget, thinking on, 8 concurrent requests, request seed 1234 (pass 1; further passes use per-pass seeds). '
          'Results carry question ids and hashes only: the dataset asks that its questions not be published in plain text.</p>'
          '<div class="note"><b>Shared seed.</b> Every pass 1 sent request seed 1234, so the configurations share sampler noise question by '
          f'question. The {min(accs):.1f}-{max(accs):.1f}% range therefore likely understates how much independent runs of one configuration '
          'vary. Clean passes 2 and 3 with per-pass seeds are in progress. <b>Scores.</b> The headline is the raw flexible-extract score '
          '(protocol v1); beside it, the audited stated-answer score <code>correct_stated</code> and accuracy among answered (non-empty) questions. '
          '<b>No reference.</b> No higher-precision version of the model was run on this hardware, so these runs do not measure what the quants '
          'cost in accuracy.</div>'
          + gpqa_accuracy_fig(allg, "Accuracy, pass 1") + gpqa_empty_fig(allg, "Empty answers, pass 1")
          + '<h2>Question by question</h2><p>Every run sent identical prompts with the same request seed, so configurations are compared per '
            'question: "only A right" counts questions A answered correctly and B did not; p is an exact McNemar test (no correction for multiple '
            'comparisons). One draw per question per configuration: each comparison resolves differences of about 5 points.</p>'
          + table(["A", "B", "What differs", "Only A right / only B right", "p", "B - A, points (95% interval)", "Only A empty / only B empty", "p"],
                  ptr, numeric=(4, 7))
          + '<h2>Published numbers, for context only</h2><div class="note">These use other weights (BF16, NVFP4) and an unstated or partly '
            'stated harness. The same NVFP4 weights score 92.1 (NVIDIA) and 90.6 (Red Hat), so harness alone moves the score by about 1.5 '
            'points. They are not plotted against the local runs.</div>' + pub)
    (OUT / "gpqa.html").write_text(page("gpqa.html", "GPQA Diamond", gp, key_closed))

    # ---- screens
    sc = ('<h1>Hard-question screen</h1><p class="lede">How often one hard GPQA question fails to finish in a configuration: 12 repeats with '
          'distinct request seeds, 12 concurrent requests, a fresh server (<a href="' + REPO + '/blob/main/protocols/hard-prompt-screen/v2.md">'
          'protocol v2</a>). A loop is repetitive reasoning stopped early or run to the budget; an exhaustion is varied reasoning that runs out of '
          'budget.</p>' + WITHDRAWAL.format(inv=INV)
          + component_fig(v2) + screen_caveats(v2) + seed_fig(v2)
          + '<h2>Single draws from the earlier fixed-seed screens</h2><p>Repeat 1 of each question from each configuration\'s first screen '
            '(protocols v0 and v1). One draw per question: these show that loops and exhaustion occur, not how often.</p>'
          + single_draw_table(runs))
    (OUT / "screens.html").write_text(page("screens.html", "Hard-question screen", sc, key_closed))

    # ---- serving
    probes = sorted(((label(m["cfg"]), m) for m in runs.values() if m["protocol"] == "serving-probe/v1"), key=lambda t: t[0])
    def bars(metric, batch, unit, title, sub, hi):
        lw, rh, top = label_width(lab for lab, _ in probes), 30, 8; pw = 460; w = lw + pw + 70; h = top + rh * len(probes) + 26
        s = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">']
        for t in [hi * f for f in (0, .25, .5, .75, 1)]:
            xx = lw + t / hi * pw
            s.append(f'<line class="grid" x1="{xx:.1f}" x2="{xx:.1f}" y1="{top}" y2="{h - 22}"/><text x="{xx:.1f}" y="{h - 6}" text-anchor="middle" class="muted">{t:g}</text>')
        tr = []
        for i, (lab, m) in enumerate(probes):
            b = m["summary"]["batches"].get(batch, {}); v = b.get(metric)
            y = top + rh * i
            s.append(f'<text x="{lw - 12}" y="{y + rh / 2 + 4}" text-anchor="end">{esc(lab)}</text>')
            if v is None:
                s.append(f'<text x="{lw + 6}" y="{y + rh / 2 + 4}" class="muted">not applicable</text>'); tr.append([esc(lab), "n/a"]); continue
            bw = v / hi * pw
            s.append(f'<rect class="hit" x="{lw}" y="{y}" width="{pw}" height="{rh}" tabindex="0" data-tip="{v:g}{unit}||{esc(lab)}"/>'
                     f'<path d="M{lw},{y + 7} h{max(bw - 4, 0):.1f} a4,4 0 0 1 4,4 v{rh - 22} a4,4 0 0 1 -4,4 h{-max(bw - 4, 0):.1f} z" fill="{color(m["cfg"])}"/>'
                     f'<text x="{lw + bw + 8:.1f}" y="{y + rh / 2 + 4}" class="muted">{v:g}{unit}</text>')
            tr.append([esc(lab), f"{v:g}{unit}"])
        s.append("</svg>")
        return figure(title, sub, "".join(s), table(["Configuration", title], tr, numeric=(1,)), legend([m["cfg"] for _, m in probes]))
    pc = {m["config"]: m["summary"]["batches"] for _, m in probes}
    moves = [100 * (pc[b][bt][k] / pc[a][bt][k] - 1) for a, b in (("glm53-flash/k3.25-v0.8.0-dflash3", "glm53-flash/k3.25-v0.8.0-kpoolfix-dflash3"),
                                                                ("glm53-flash/k3.25-v0.8.0-nospec", "glm53-flash/k3.25-v0.8.0-kpoolfix-nospec"))
             for bt, k in (("sampled_c1", "median_decode_tok_s"), ("sampled_c8", "median_decode_tok_s"), ("sampled_c8", "aggregate_tok_s"))]
    acc_move = max(abs(pc["glm53-flash/k3.25-v0.8.0-kpoolfix-dflash3"][bt]["acceptance_rate"] - pc["glm53-flash/k3.25-v0.8.0-dflash3"][bt]["acceptance_rate"])
                   for bt in ("sampled_c1", "sampled_c8"))
    kvt = []
    for m in sorted(runs.values(), key=lambda m: m["id"]):
        k = re.search(r"KV pool ([\d,]+) tokens", m["notes"])
        if k and m["protocol"] in ("hard-prompt-screen/v2", "gpqa-diamond/v1") or (k and m["id"].endswith("bisect1")):
            if any(r[0] == esc(label(m["cfg"])) for r in kvt): continue
            kvt.append([esc(label(m["cfg"])), k.group(1), link(m["id"])])
    a8 = {c: v["sampled_c8"]["acceptance_rate"] for c, v in pc.items() if v.get("sampled_c8", {}).get("acceptance_rate") and "dflash3" in c}
    a90 = a8["glm53-flash/k3.25-v0.9.0-dflash3"]; aoth = [v for c, v in a8.items() if c != "glm53-flash/k3.25-v0.9.0-dflash3"]
    sv = ('<h1>Speed, acceptance and KV capacity</h1><p class="lede">3.25bpw weights, 16 fixed GPQA prompts, 4,096 tokens, temperature 1.0, one run '
          'per configuration: tpurtell 0.8.0 with and without the kpool fixes (2026-10-04), and tpurtell 0.9.0 with and without the DCP1 tail fix '
          '(2026-10-08; see the <a href="looping.html">looping investigation</a>). The kpool fixes show no consistent change: with the same speculation '
          f'setting, decode speed and throughput move by {min(moves):+.0f}% to {max(moves):+.0f}% and acceptance by at most {acc_move:.3f} (see the '
          f'<a href="{REPO}/tree/main/comparisons/glm53-flash-serving-probe">comparison</a>).</p>'
          '<div class="note"><b>Single runs, no noise floor.</b> Every speed and acceptance figure on this page comes from one run per configuration; '
          'no configuration was repeated, so differences of a few percent cannot be told from run-to-run variation. Descriptively, draft acceptance '
          f'at 8 concurrent requests was {a90:.4f} on unpatched tpurtell 0.9.0 against {min(aoth):.4f}-{max(aoth):.4f} on the other DFlash2 ×3 builds.</div>'
          + bars("median_decode_tok_s", "sampled_c1", " tok/s", "Decode speed, one request at a time", "Median per-request decode tokens per second.", 180)
          + bars("aggregate_tok_s", "sampled_c8", " tok/s", "Total throughput, 8 requests at once", "Generated tokens per second across all requests.", 400)
          + bars("acceptance_rate", "sampled_c8", "", "Draft acceptance rate, 8 requests at once", "Accepted / drafted tokens from the server's counters.", 0.6)
          + '<h2>KV cache capacity</h2><p>KV pool reported by the server at start-up (server logs, quoted in each run\'s notes), same memory setting '
            '(0.95). tpurtell\'s default DCP1 layout with MLA layer ownership holds the most tokens.</p>'
          + table(["Configuration", "KV pool, tokens", "Receipts"], kvt, numeric=(1,))
          + '<h2>Greedy agreement</h2><p>Temperature 0, one request at a time: characters of identical output before two runs diverge. '
            f'Even the same configuration run twice, one request at a time, diverges early (median {floor["median_shared_prefix_chars"]:.0f} '
            'characters), so batching cannot explain it, and greedy parity cannot certify that speculative decoding is exact. The prefix lengths are '
            'derived from output text, which is not published (model output); each run publishes its outputs\' hashes.</p>'
          + table(["Comparison", "Identical outputs", "Median shared characters"],
                  [[esc(pp["meaning"]), f'{pp["identical_outputs"]}/{pp["n"]}', f'{pp["median_shared_prefix_chars"]:.0f}'] for pp in par["pairs"]], numeric=(1, 2)))
    (OUT / "serving.html").write_text(page("serving.html", "Speed and acceptance", sv, key_closed))

    # ---- kernels
    km = sorted(kern.values(), key=lambda m: (m["cfg"]["engine"]["version"], bool(m["cfg"]["engine"]["patches"])))
    tests = []
    for m in km:
        for r in m["rows"]:
            if (r["suite"], r["test"]) not in tests: tests.append((r["suite"], r["test"]))
    hdr = ["Test"] + [engine_label(m["cfg"]) for m in km]
    mark = {"passed": '<span class="ok">✓ pass</span>', "failed": '<span class="bad">✗ fail</span>', "skipped": "– skipped"}
    ktr = []
    for suite, t in tests:
        cells = []
        for m in km:
            o = next((r["outcome"] for r in m["rows"] if r["suite"] == suite and r["test"] == t), None)
            cells.append(mark.get(o, "-"))
        ktr.append([("upstream: " if suite == "upstream" else "rejected-draft: ") + esc(t)] + cells)
    kp = ('<h1>Kernel tests</h1><p class="lede">vLLM\'s own regression tests for the GLM-5.3-Flash kpool kernels '
          '(<code>tests/kernels/test_kpool_decode_update_batched.py</code>, pinned by hash), plus a rejected-draft reproduction, run inside each '
          'engine image. The four upstream failures are exactly the tests written for vllm#57477 and vllm#58454; the kpool fixes '
          '(<a href="https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/5">tpurtell PR #5</a>, shipped in tpurtell 0.9.0) fix all of them.</p>'
          + ('' if any(m["cfg"]["engine"]["version"] in ("v0.9.0", "v0.9.1") for m in km) else
             '<div class="note">The tests ran on the 0.7.0 and 0.8.0 release images and on local builds with the fixes. They have not yet run on the '
             '0.9.0 or 0.9.1 release images: that those releases carry the fixes rests, for now, on their kpool kernel files being byte-identical '
             'to the tested build.</div>')
          + table(hdr, ktr))
    (OUT / "kernels.html").write_text(page("kernels.html", "Kernel tests", kp, key_closed))

    # ---- looping investigation
    (OUT / "looping.html").write_text(page("looping.html", "Looping investigation", looping(runs), key_closed))

    # ---- method
    mt = ('<h1>How to read this</h1>'
          '<h2>Labels</h2><p>Every configuration is named <i>weights · engine version · speculation</i>, built from its configuration file by one rule '
          f'(<a href="{REPO}/blob/main/SCHEMA.md#labels">SCHEMA.md</a>). The engine name comes first because engines number their versions '
          'independently. Chart colours mark engine series (builds that share one code base); the legend under each chart lists the labels each colour covers.</p>'
          '<h2 id="scoring">Scoring</h2><p>GPQA scores are lm-eval\'s raw <code>flexible-extract</code> filter. It takes the last parenthesised '
          'capital letter in the reply, so a reply that states its answer and then mentions other options\' labels, or uses notation such as (H) '
          'or (R) in chemistry, is read as choosing the last one. The misread is deterministic: the same reply is always scored the same way, on '
          'particular questions, so repeating a run never reveals it. An audit of the stated final answer of every published reply, hand-checked '
          'wherever it disagrees with the filter, is published per row as <code>correct_stated</code>: it puts the raw score '
          f'{min(gap):.1f}-{max(gap):.1f} points low per run. Raw scores stay the headline (protocol v1) so runs remain comparable; the stated-answer '
          'score is shown beside them as a secondary, audited score. <code>strict-match</code> records whether the reply used the phrase "The answer is", which the prompt never '
          'asks for; it is not an accuracy measure.</p>'
          '<h2>Seeds</h2><p>The engine draws each request\'s sampling noise from its request seed. Repeats that share a seed still vary (see '
          'below) and their texts diverge, but they draw on the same sampler noise, so that variation is not a statistically meaningful sample. '
          'Screens therefore '
          'use a distinct seed per repeat (protocol v2), and GPQA passes a distinct seed per pass (pass <i>p</i> sends 1233 + <i>p</i>). The '
          'earlier fixed-seed screens are shown only as single draws.</p>'
          '<h2>Intervals</h2><p>GPQA accuracy: 95% bootstrap over questions. Rates (empty answers; screen failures of one question in one '
          'configuration): 95% Wilson score intervals. Configurations are compared question by question. Where intervals overlap, the '
          'configurations cannot be told apart.</p>'
          '<h2>The engine is not bitwise reproducible</h2><p>Even one request at a time, the engine does not repeat itself exactly: a greedy '
          f'(temperature 0) request diverges from its own rerun after a median of {floor["median_shared_prefix_chars"]:.0f} characters, and two '
          'decode-vs-prefill runs of tpurtell 0.9.1 with the same prompts and seeds, one request at a time, first differ in their '
          'per-position values after 1-130 generated tokens. Batching cannot explain that. At 8 or 12 concurrent requests, batch composition adds '
          'further variation. Why single requests vary is an open question. Compare outcomes over many draws, never individual transcripts. '
          'The analysis reproduces exactly: <code>tools/verify.py</code> and <code>tools/analyze.py</code> recompute every number from the published rows.</p>'
          '<h2 id="receipts">Receipts</h2><p>Every number on this site is recomputed from published files: per-item rows (<code>results.jsonl</code>), '
          'and, for figures from the engine\'s server log (KV pool size, throughput, acceptance, waiting requests), the numeric fields parsed from '
          'that log (<code>server_log.jsonl</code>, summarised in <code>summary.json</code>). Two kinds of figure rest on text that is not published, '
          'because it is model output or benchmark text: the greedy shared-prefix lengths (<code>comparisons/glm53-flash-serving-probe/parity.json</code>; '
          'the outputs\' hashes are published), and the per-row <code>correct_stated</code> judgement (the judgement is published, the reply text it '
          'was read from is not).</p>'
          '<h2>Comparability</h2><p>Comparisons only line up runs with the same protocol version and hardware, and declare which configuration fields differ; '
          '<code>tools/verify.py</code> enforces this. Published numbers from other harnesses are shown as context, never plotted as a ranking.</p>'
          f'<h2>Sources</h2><ul><li><a href="{REPO}/tree/main/protocols">Protocols</a> - exact settings per benchmark version</li>'
          f'<li><a href="{REPO}/tree/main/configs">Configurations</a> - engine image digests, model revisions, settings</li>'
          f'<li><a href="{REPO}/blob/main/DATASHEET.md">Datasheet</a> - what the data is, and is not, suitable for</li>'
          f'<li><a href="{REPO}/blob/main/FINDINGS.md">Findings</a> - the full write-up</li></ul>' + glossary_html())
    (OUT / "method.html").write_text(page("method.html", "How to read this", mt, key_open))
    print(f"built {len(PAGES)} pages into {OUT}")


if __name__ == "__main__":
    build()
