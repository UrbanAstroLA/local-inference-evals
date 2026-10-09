#!/usr/bin/env python3
"""Build the static results site in docs/ from the repository's own data (standard library only).

    python3 tools/site.py

Every chart and table is computed from runs/ at build time, so the site cannot drift from the receipts.
Intervals: GPQA accuracy = 95% bootstrap over questions (a question's passes resampled together);
rates (empty answers, screen failures) = 95% Wilson score intervals. Serve docs/ with GitHub Pages.
Labels, the label key and chart colours come from config fields (SCHEMA.md, "Labels"): a new engine needs data, not code.
"""
import html, json, math, random, re, statistics as st
from pathlib import Path

from verify import config_label, engine_label, weights_label

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
                     + '. Used as a diagnostic control; see the configuration notes.'])
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
    nav = "".join(f'<a href="{f}" class="{"on" if f == name else ""}">{esc(t)}</a>' for f, t in PAGES)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(title)} - local-inference-evals</title><style>{CSS}</style></head><body>'
            f'<nav><span class="brand">local-inference-evals</span>{nav}</nav><main>{with_key(body, key_html)}</main>'
            f'<footer>Generated from the repository data by <code>tools/site.py</code>. Receipts, protocols and code: '
            f'<a href="{REPO}">{REPO.replace("https://", "")}</a>. Results CC BY 4.0, code Apache-2.0.</footer>'
            f'<div id="tip" role="status"></div><script>{JS}</script></body></html>')


def gpqa_rows(runs, full_only=False):
    out = []
    for rid, m in runs.items():
        if m["protocol"] != "gpqa-diamond/v1":
            continue
        s = m["summary"]; npass = len(s["passes"])
        if full_only and npass < 3:
            continue
        lab = label(m["cfg"]) + ("" if npass == 3 else f" ({npass} pass{'es' if npass > 1 else ''})")
        out.append(dict(rid=rid, label=lab, cfg=m["cfg"], color=color(m["cfg"]), npass=npass, s=s, n=s["samples"]))
    return sorted(out, key=lambda r: (r["npass"] != 3, list(SERIES).index(r["cfg"]["engine"]["series"]), r["label"]))


def gpqa_accuracy_fig(rows, title):
    data = [dict(label=r["label"], color=r["color"], est=100 * r["s"]["accuracy_flexible"], lo=100 * r["s"]["accuracy_flexible_ci95"][0],
                 hi=100 * r["s"]["accuracy_flexible_ci95"][1],
                 tip=f'{100 * r["s"]["accuracy_flexible"]:.1f}% (95% CI {100 * r["s"]["accuracy_flexible_ci95"][0]:.1f}-{100 * r["s"]["accuracy_flexible_ci95"][1]:.1f})||'
                     f'{r["label"]}, {r["n"]} answers') for r in rows]
    body = interval_plot(data, 70, 100, [70, 75, 80, 85, 90, 95, 100], lambda v: f"{v:.0f}" if v == int(v) else f"{v:.1f}", "%")
    tbl = table(["Configuration", "Passes", "Accuracy", "95% CI", "Receipts"],
                [[esc(r["label"]), r["npass"], f'{100 * r["s"]["accuracy_flexible"]:.1f}%',
                  f'{100 * r["s"]["accuracy_flexible_ci95"][0]:.1f}-{100 * r["s"]["accuracy_flexible_ci95"][1]:.1f}', link(r["rid"])] for r in rows], numeric=(1, 2))
    return figure(title, "Raw lm-eval flexible-extract score, which reads some correct answers as wrong (about 0.7 points per pass; see the protocol). Dot = mean over passes; line = 95% bootstrap interval over questions "
                  "(each question's passes resampled together). Overlapping lines: the configurations cannot be told apart.", body, tbl,
                  legend([r["cfg"] for r in rows]))


def gpqa_empty_fig(rows, title):
    data = []
    for r in rows:
        k, n = r["s"]["empty"], r["n"]; lo, hi = wilson(k, n)
        data.append(dict(label=r["label"], color=r["color"], est=100 * k / n, lo=100 * lo, hi=100 * hi,
                         tip=f'{k} of {n} ({100 * k / n:.1f}%)||{r["label"]}; 95% Wilson {100 * lo:.1f}-{100 * hi:.1f}%'))
    body = interval_plot(data, 0, 8, [0, 2, 4, 6, 8], lambda v: f"{v:.1f}" if v != int(v) else f"{v:.0f}", "%")
    tbl = table(["Configuration", "Empty answers", "Answers", "Rate", "95% Wilson", "Receipts"],
                [[esc(r["label"]), r["s"]["empty"], r["n"], f'{100 * r["s"]["empty"] / r["n"]:.1f}%',
                  "{:.1f}-{:.1f}%".format(*(100 * v for v in wilson(r["s"]["empty"], r["n"]))), link(r["rid"])] for r in rows], numeric=(1, 2, 3))
    return figure(title, "Share of answers that came back empty (reasoning never finished within the token budget; scored wrong). "
                  "Line = 95% Wilson interval. Empties cluster on a few questions, so true uncertainty is somewhat wider.", body, tbl,
                  legend([r["cfg"] for r in rows]))


def screen_rows(runs, protocol):
    out = []
    for rid, m in runs.items():
        if m["protocol"] != protocol:
            continue
        s = m["summary"]
        out.append(dict(rid=rid, label=label(m["cfg"]) + f' ({m["date"][5:]})', cfg=m["cfg"], color=color(m["cfg"]), s=s,
                        invalid="INVALID" in m["notes"]))
    labs = [r["label"] for r in out]   # two screens of one configuration on one date: number them by run id order
    for r in out:
        if labs.count(r["label"]) > 1:
            same = sorted(x["rid"] for x in out if x["label"] == r["label"])
            r["label2"] = r["label"][:-1] + f', screen {same.index(r["rid"]) + 1})'
    for r in out:
        r["label"] = r.pop("label2", r["label"])
    return sorted(out, key=lambda r: (list(SERIES).index(r["cfg"]["engine"]["series"]), r["label"]))


def screen_fig(rows, title, sub):
    data, tr = [], []
    for r in rows:
        s = r["s"]
        if r["invalid"]:
            tr.append([esc(r["label"]), "-", "-", "-", "-", '<span class="bad">invalid: engine crashed (see receipts)</span>', link(r["rid"])])
            continue
        k, n = s["non_ok"], s["requests"]; lo, hi = wilson(k, n)
        data.append(dict(label=r["label"], color=r["color"], est=k, lo=lo * n, hi=hi * n,
                         tip=f'{k} of {n} failed||{r["label"]}: {s["loop"]} loops, {s["exhaust"]} exhaustions; 95% Wilson {lo * n:.1f}-{hi * n:.1f}'))
        tr.append([esc(r["label"]), s["ok"], s["loop"], s["exhaust"], k, f"{lo * n:.1f}-{hi * n:.1f}", link(r["rid"])])
    body = interval_plot(data, 0, 40, [0, 10, 20, 30, 40], lambda v: f"{v:.0f}", "")
    tbl = table(["Configuration", "OK", "Loop", "Exhaust", "Failures / 40", "95% Wilson", "Receipts"], tr, numeric=(1, 2, 3, 4))
    return figure(title, sub, body, tbl, legend([r["cfg"] for r in rows]))


def heatmap(runs, rows=None, caption=None):
    if rows is None:
        rows = [r for r in screen_rows(runs, "hard-prompt-screen/v1") + screen_rows(runs, "hard-prompt-screen/v0") if not r["invalid"]]
    docs = [79, 13, 127, 88, 121]
    cw, lw, rh, top = 92, label_width(r["label"] for r in rows), 30, 30
    w, h = lw + cw * len(docs) + 10, top + rh * len(rows) + 10
    seq = [None, "var(--seq1)", "var(--seq2)", "var(--seq3)", "var(--seq4)", "var(--seq5)", "var(--seq6)", "var(--seq7)", "var(--seq7)"]
    s = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">']
    for j, d in enumerate(docs):
        s.append(f'<text x="{lw + cw * j + cw / 2}" y="18" text-anchor="middle">question {d}</text>')
    tr = []
    for i, r in enumerate(rows):
        y = top + rh * i
        s.append(f'<text x="{lw - 12}" y="{y + rh / 2 + 4}" text-anchor="end">{esc(r["label"])}</text>')
        cells = []
        for j, d in enumerate(docs):
            pd = r["s"]["per_doc"].get(str(d), {}); fail = pd.get("loop", 0) + pd.get("exhaust", 0); tot = sum(pd.values())
            ink = "#ffffff" if fail >= 4 else "#0b0b0b"   # by fill luminance, not page theme
            s.append(f'<rect x="{lw + cw * j + 1}" y="{y + 1}" width="{cw - 2}" height="{rh - 2}" rx="4" style="fill:{seq[fail] if fail else "var(--grid)"}"/>'
                     f'<rect class="hit" x="{lw + cw * j + 1}" y="{y + 1}" width="{cw - 2}" height="{rh - 2}" rx="4" tabindex="0" '
                     f'data-tip="{fail} of {tot} failed||{r["label"]}, question {d}: {pd.get("loop", 0)} loops, {pd.get("exhaust", 0)} exhaustions"/>'
                     f'<text x="{lw + cw * j + cw / 2}" y="{y + rh / 2 + 4}" text-anchor="middle" style="fill:{ink if fail else "var(--ink2)"};pointer-events:none">{fail}</text>')
            cells.append(f"{fail}/{tot}")
        tr.append([esc(r["label"])] + cells)
    s.append("</svg>")
    tbl = table(["Configuration"] + [f"q{d}" for d in docs], tr)
    scale = ('<div class="legend">Failures out of 8: ' + "".join(f'<span><i style="background:{seq[v]};border-radius:2px;outline:1px solid var(--ring)"></i>{v}</span>' for v in (0, 2, 4, 6, 8)) + "</div>")
    q88 = [r["s"]["per_doc"].get("88", {}) for r in rows]
    f88 = [d.get("loop", 0) + d.get("exhaust", 0) for d in q88]
    caption = caption or (f'Question 88 fails in {min(f88)}-{max(f88)} of 8 repeats in every configuration, which points to the model '
                          f'as much as the runtime. Rows dated 09-30 used protocol v0 (no early stop); the others v1.')
    return (f'<figure><figcaption><div class="t">Failures by question</div><div class="s">Each cell: how many of the 8 repeats of that question failed '
            f'(loop or exhaustion). {caption}</div></figcaption>'
            f'{scale}<div class="chart">{"".join(s)}</div><details><summary>Show the numbers as a table</summary>{tbl}</details></figure>')


def fisher(a, b, c, d):
    """Two-sided Fisher exact test for the 2x2 table [[a, b], [c, d]]."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    p = lambda x: math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
    p0 = p(a)
    return sum(p(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1) if p(x) <= p0 * (1 + 1e-9))


def empty_diff_ci(a, b, n_boot=10000, seed=0):
    """95% interval of (empties in b - empties in a), resampling questions with all their passes (as tools/analyze.py)."""
    by = {}
    for r in a["rows"]: by.setdefault(r["doc_id"], [0, 0])[0] += r["empty"]
    for r in b["rows"]: by.setdefault(r["doc_id"], [0, 0])[1] += r["empty"]
    ids = sorted(by); rng = random.Random(seed)
    d = sorted(sum(by[i][1] - by[i][0] for i in (rng.choice(ids) for _ in ids)) for _ in range(n_boot))
    return d[int(0.025 * n_boot)], d[int(0.975 * n_boot) - 1]


def empties_vs(a, b):
    """(empties a, empties b, answers, Fisher p) for two GPQA runs over the passes both have."""
    ps = sorted(set(a["summary"]["passes"]) & set(b["summary"]["passes"]))
    ea, eb = (sum(r["empty"] for r in m["rows"] if r["pass"] in ps) for m in (a, b))
    n = 198 * len(ps)
    return ea, eb, n, fisher(ea, n - ea, eb, n - eb)


def by_config(runs, protocol):
    return {m["config"]: m for m in runs.values() if m["protocol"] == protocol}


def k4_signal(runs):
    """Empty answers, 4bpw tpurtell 0.9.0 vs 0.8.0 (no kpool fixes), over the GPQA passes both runs have."""
    g = by_config(runs, "gpqa-diamond/v1")
    a, b = g.get("glm53-flash/k4-v0.8.0-dflash3"), g.get("glm53-flash/k4-v0.9.0-dflash3")
    if not a or not b:
        return ""
    ea, eb, n, pv = empties_vs(a, b); lo, hi = empty_diff_ci(a, b)
    return (f' No detectable completion cost either: on {esc(weights_label(b["cfg"]))}, {esc(engine_label(b["cfg"]))} left {eb} of {n} answers empty '
            f'against {ea} for {esc(engine_label(a["cfg"]))} with the same prompts (Fisher p = {pv:.2f}). Resampling questions, the difference is '
            f'{lo:+d} to {hi:+d} answers, so a small cost is not excluded.')


LOOP_CFG = {"as": "glm53-flash/k3.25-v0.9.0-dflash3", "fix": "glm53-flash/k3.25-v0.9.0-tailfix-dflash3",
            "ctrl": "glm53-flash/k3.25-v0.9.0-ep2dcp2-dflash3"}
INV = REPO + "/tree/main/investigations/2026-10-glm53-looping"


def looping(runs):
    """The looping / empty-answer investigation across tpurtell runtimes (investigations/2026-10-glm53-looping)."""
    by = lambda proto, cfg: sorted((m for m in runs.values() if m["protocol"] == proto and m["config"] == cfg), key=lambda m: m["id"])
    order = [LOOP_CFG["as"], LOOP_CFG["fix"], LOOP_CFG["ctrl"]]
    screens = {c: by("hard-prompt-screen/v1", c) for c in order}
    data, tr, pooled = [], [], {}
    for c in order:
        ms = screens[c]; rows = [r for m in ms for r in m["rows"]]; cfg = ms[0]["cfg"]
        k = sum(r["cls"] in ("loop", "exhaust") for r in rows); n = len(rows); lo, hi = wilson(k, n); pooled[c] = (k, n)
        data.append(dict(label=label(cfg), color=color(cfg), est=100 * k / n, lo=100 * lo, hi=100 * hi,
                         tip=f"{k} of {n} failed ({100 * k / n:.0f}%)||{label(cfg)}; 95% Wilson {100 * lo:.1f}-{100 * hi:.1f}%"))
        tr.append([esc(label(cfg)), f"{k}/{n}", f"{100 * lo:.1f}-{100 * hi:.1f}%", sum(r["cls"] == "loop" for r in rows),
                   sum(r["cls"] == "exhaust" for r in rows), " ".join(link(m["id"], m["id"][-7:]) for m in ms)])
    fp = lambda a, b: fisher(pooled[a][0], pooled[a][1] - pooled[a][0], pooled[b][0], pooled[b][1] - pooled[b][0])
    fig1 = figure("Failures (loop or exhaustion) per configuration, 2 screens x 40", "Dot = share of 80 requests that did not finish; line = 95% Wilson "
                  f"interval (assumes independent requests; outcomes cluster by question, so it is too narrow). Fisher exact: as released vs control "
                  f"p = {fp(order[0], order[2]):.3f}; with vs without the fix p = {fp(order[0], order[1]):.2f}; fix vs control p = {fp(order[1], order[2]):.2f}. "
                  f"Both screens with the fix failed less than both without it; none of the differences is significant yet at 80 requests "
                  f"per configuration (the preregistered rule's labels: inconclusive / no detectable loop effect; see the write-up).",
                  interval_plot(data, 0, 100, [0, 20, 40, 60, 80, 100], lambda v: f"{v:.0f}", "%"),
                  table(["Configuration", "Failures", "95% Wilson", "Loops", "Exhaustions", "Receipts"], tr, numeric=(1, 3, 4)),
                  legend([screens[c][0]["cfg"] for c in order]))
    hrows = [r for r in screen_rows(runs, "hard-prompt-screen/v1") if r["cfg"]["id"] in order]
    hm = heatmap(runs, sorted(hrows, key=lambda r: (order.index(r["cfg"]["id"]), r["label"])),
                 "Two screens per configuration, each on a freshly started server. Question 13 (post hoc, not preregistered) failed only as released.")
    # decode vs prefill
    dp = {c: by("decode-prefill-consistency/v1", c)[0] for c in ("glm53-flash/k3.25-v0.9.0-nospec-nocache", "glm53-flash/k3.25-v0.9.0-tailfix-nospec-nocache")}
    kd, ktr = [], []
    for reg, name in (("lt2044", "i < 2,044"), ("ge2048", "i ≥ 2,048")):
        for c, m in dp.items():
            per = []
            for d in sorted({r["doc_id"] for r in m["rows"]}):
                v = [r["kl_top20"] for r in m["rows"] if r["doc_id"] == d and r["region"] == reg and r["kl_top20"] is not None]
                per.append(sum(v) / len(v))
            s_ = m["summary"]["by_region"][reg]; lab = f"{name} · {engine_label(m['cfg'])}"
            kd.append(dict(label=lab, color=color(m["cfg"]), est=s_["mean_kl_top20"], lo=min(per), hi=max(per),
                           tip=f"KL {s_['mean_kl_top20']:.4f}||{lab}; {s_['positions']} positions; per-prompt range {min(per):.4f}-{max(per):.4f}"))
            ktr.append([esc(lab), s_["positions"], f"{s_['mean_kl_top20']:.4f}", f"{100 * s_['top1_agree']:.1f}%", f"{s_['mean_abs_dlp']:.4f}", link(m["id"])])
    fig2 = figure("Decode vs prefill re-scoring of the same tokens", "3.25bpw, speculation and prefix caching off, 6 GPQA prompts x 2,600 decoded "
                  "tokens. Dot = mean KL over the shared top-20 tokens (lower = decode agrees with prefill); line = range of the six prompts' means. "
                  "The difference persists past 2,048 tokens, where the fix changes no selection, because keys and values written earlier stay in the cache.",
                  interval_plot(kd, 0, 0.12, [0, 0.03, 0.06, 0.09, 0.12], lambda v: f"{v:g}", ""),
                  table(["Positions · engine", "Positions", "Mean KL", "Top-1 agreement", "Mean abs Δlogprob", "Receipts"], ktr, numeric=(1, 2, 3, 4)),
                  legend([m["cfg"] for m in dp.values()]))
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
    # speed / capacity from run notes and rows
    sp = []
    for c in order:
        for m in screens[c]:
            v = [r["completion_tokens"] / r["secs"] for r in m["rows"] if r["finish_reason"] == "stop" and r.get("completion_tokens")]
            note = m["notes"]
            kv = re.search(r"KV pool ([\d,]+) tokens", note).group(1); thr = re.search(r"throughput (\d+) tok/s", note).group(1)
            mal = re.search(r"mean acceptance length ([\d.]+)", note).group(1)
            sp.append([esc(label(m["cfg"])), m["date"], f"{st.median(v):.1f}", thr, mal, kv, link(m["id"])])
    eh = lambda c: esc(engine_label(te[c]["cfg"]))
    return ('<h1>Looping, exhaustion and empty answers across tpurtell runtimes</h1>'
            '<p class="lede">On the same weights, tpurtell 0.8.0 and 0.9.0 leave more long reasoning unfinished than 0.7.0. This page shows the '
            'preregistered layout bisection on 0.9.0, and a decode bug found on the way and its fix '
            '(<a href="https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/6">tpurtell PR #6</a>, released in tpurtell 0.9.1). Write-up, glossary, open questions and '
            f'commands to recompute every number: <a href="{INV}">investigations/2026-10-glm53-looping</a>.</p>'
            '<div class="note"><b>Terms.</b> <i>DCP1 + MLA layer ownership</i> (0.8.0/0.9.0 default): each MLA layer runs on one GPU with its whole '
            'KV cache; <i>DCP2</i> (0.7.0) splits each layer\'s KV cache across both GPUs. <i>TP2 / EP2 experts</i>: routed experts split across '
            'both GPUs, or placed whole on one. <i>kpool tail</i>: the newest, incomplete 4-token pool of the sparse-attention selection. '
            '<b>0.7.0 layout (DCP2, EP2)</b> is a diagnostic control used to separate the tail bug from the rest of the layout change, not a '
            'recommendation: with the same weights and memory setting the default layout held 4,707,515 KV tokens and the control 3,140,895-3,165,056.</div>'
            + fig1 + hm + fig2
            + '<h2>Which tokens a decode step attends (index check, the image\'s own kernels on GPU)</h2>'
              '<p>Selections as the indexer emits them (valid pools first). Without the fix, the tail (current token and up to two before it) is '
              'masked out at every causal length up to 2,043 that is not a multiple of 4. With the fix, every selected token is attended; rows the '
              'stock code already handled are byte-identical.</p>'
            + table(["Causal length", "Tail tokens", eh(LOOP_CFG["as"]), eh(LOOP_CFG["fix"]), "Same row bytes"], itr, numeric=(0,))
            + '<h2>Tool calling (tool-eval-bench, 88 scenarios, temperature 0, two repeats)</h2>'
            + table(["Configuration", "Repeat 1", "Repeat 2", "Receipts"], pts)
            + '<p>Scenarios whose status differs anywhere. TC-80 and TC-88 fail in both repeats without the fix and pass in both with it; the '
              'others also vary between repeats of the same build.</p>'
            + table(["Scenario"] + [f"{eh(c)}, repeat {rp}" for c in te for rp in (1, 2)], ttr)
            + '<h2>Speed and KV capacity during the screens</h2><p>Per-request rate from the rows; server throughput (10-second intervals with at '
              'least 8 running requests), mean acceptance length and KV cache size from the server logs, as recorded in each run\'s notes.</p>'
            + table(["Configuration", "Date", "Median tok/s, finished requests", "Server tok/s", "Mean acceptance length", "KV cache tokens", "Receipts"],
                    sp, numeric=(2, 3, 4, 5)))


def build():
    runs = load_runs(); OUT.mkdir(exist_ok=True)
    first = {}
    for m in sorted(runs.values(), key=lambda m: (m["date"], m["cfg"]["engine"]["series"])):
        first.setdefault(m["cfg"]["engine"]["series"], m["date"])
    SERIES.clear(); SERIES.update((ser, PALETTE[i % len(PALETTE)]) for i, ser in enumerate(sorted(first, key=lambda x: (first[x], x))))
    cfgs = list({m["config"]: m["cfg"] for m in runs.values()}.values())
    key_open, key_closed = key(cfgs, open_=True), key(cfgs)
    full = gpqa_rows(runs, full_only=True); allg = gpqa_rows(runs)
    g = by_config(runs, "gpqa-diamond/v1")
    g07, g08 = g["glm53-flash/k3.25-v0.7.0-dflash5"], g["glm53-flash/k3.25-v0.8.0-dflash3"]
    e07, e08, n0708, p0708 = empties_vs(g07, g08)
    s0 = by_config(runs, "hard-prompt-screen/v0")
    sc07, sc08 = s0["glm53-flash/k3.25-v0.7.0-dflash5"]["summary"]["non_ok"], s0["glm53-flash/k3.25-v0.8.0-dflash3"]["summary"]["non_ok"]
    accs = [100 * r["s"]["accuracy_flexible"] for r in allg]
    p1 = {}
    for r in allg:
        p1[r["label"]] = {x["doc_id"]: x for x in runs[r["rid"]]["rows"] if x["pass"] == 1}
    def sign_p(a, b):
        x = sum(a[d]["correct_flexible"] and not b[d]["correct_flexible"] for d in a)
        y = sum(b[d]["correct_flexible"] and not a[d]["correct_flexible"] for d in a)
        n = x + y
        return x, y, (1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(x, y) + 1)) / 2 ** n))
    names = sorted(p1); pairs = [(a, b) + sign_p(p1[a], p1[b]) for i, a in enumerate(names) for b in names[i + 1:]]
    min_p = min(t[4] for t in pairs)
    kern = {rid: m for rid, m in runs.items() if m["protocol"] == "kpool-kernel-tests/v1"}
    up = lambda s: (sum(1 for r in s if r["suite"] == "upstream" and r["outcome"] == "passed"), sum(1 for r in s if r["suite"] == "upstream" and r["outcome"] != "skipped"))
    up_un = up(next(m["rows"] for m in kern.values() if not m["cfg"]["engine"]["patches"]))
    up_pa = up(next(m["rows"] for m in kern.values() if m["cfg"]["engine"]["patches"]))
    gain = [100 * (r["s"]["accuracy_flexible_answered"] - r["s"]["accuracy_flexible"]) for r in full]
    answered = [100 * r["s"]["accuracy_flexible_answered"] for r in full]
    qacc = {}                                           # each question's accuracy when answered, over all full runs
    for r in full:
        for x in runs[r["rid"]]["rows"]:
            if not x["empty"]: qacc.setdefault(x["doc_id"], []).append(x["correct_flexible"])
    hard = [v for d, vs in qacc.items() for v in vs if any(x["empty"] and x["doc_id"] == d for r in full for x in runs[r["rid"]]["rows"])]
    easy = [v for d, vs in qacc.items() for v in vs if not any(x["empty"] and x["doc_id"] == d for r in full for x in runs[r["rid"]]["rows"])]
    credited = [100 * sum(x["correct_flexible"] if not x["empty"] else sum(qacc.get(x["doc_id"], [0])) / max(1, len(qacc.get(x["doc_id"], [])))
                          for x in runs[r["rid"]]["rows"]) / len(runs[r["rid"]]["rows"]) for r in full]
    cgain = [c - 100 * r["s"]["accuracy_flexible"] for c, r in zip(credited, full)]
    per_pass = []
    for r in full:
        rows = runs[r["rid"]]["rows"]; lo, hi = (100 * v for v in r["s"]["accuracy_flexible_ci95"])
        acc = [100 * sum(x["correct_flexible"] for x in rows if x["pass"] == p) / 198 for p in (1, 2, 3)]
        per_pass.append((r, acc, max(acc) - min(acc), all(lo <= a <= hi for a in acc)))
    spread = (min(t[2] for t in per_pass), max(t[2] for t in per_pass)); inside = sum(3 for t in per_pass if t[3])
    par = json.loads((ROOT / "comparisons/glm53-flash-serving-probe/parity.json").read_text())
    floor = next(pp for pp in par["pairs"] if pp["a"].split("#")[0] == pp["b"].split("#")[0])

    # ---- index
    tiles = (f'<div class="tiles"><div class="tile"><div class="lab">GPQA Diamond accuracy, every configuration</div>'
             f'<div class="val">{min(accs):.1f}-{max(accs):.1f}%</div><div class="sub">no configuration distinguishable from another</div></div>'
             f'<div class="tile"><div class="lab">Empty GPQA answers per {n0708}, same {esc(weights_label(g07["cfg"]))} weights</div><div class="val">{e07} vs {e08}</div>'
             f'<div class="sub">{esc(engine_label(g07["cfg"]))} vs {esc(engine_label(g08["cfg"]))} (Fisher p = {p0708:.3f})</div></div>'
             f'<div class="tile"><div class="lab">Upstream kpool regression tests</div><div class="val">{up_pa[0]}/{up_pa[1]}</div>'
             f'<div class="sub">with the kpool fixes (without: {up_un[0]}/{up_un[1]})</div></div></div>')
    idx = (f'<h1>GLM-5.3-Flash on 2x RTX PRO 6000: end-to-end receipts</h1>'
           f'<p class="lede">Accuracy, completion, speed and kernel-correctness measurements of locally served GLM-5.3-Flash EXL3 quants '
           f'on tpurtell\'s vLLM-based engine releases and two locally patched builds of them, under fixed, versioned protocols. Every number links to raw per-item receipts and can be '
           f'recomputed with the repository\'s tools.</p>{tiles}'
           + gpqa_accuracy_fig(full, "GPQA Diamond accuracy (3 passes, 594 answers)")
           + gpqa_empty_fig(full, "Empty answers on GPQA Diamond (3 passes)")
           + screen_fig(screen_rows(runs, "hard-prompt-screen/v1"), "Hard-question screen: failures out of 40",
                        "Five of the hardest GPQA questions x 8 repeats. Line = 95% Wilson interval (assumes independent runs; see "
                        "<a href=\"method.html\">how to read this</a>). The questions were chosen from empty answers of tpurtell 0.8.0 (no kpool fixes), "
                        "so the gap between engine series here is an upper-end estimate; comparisons within one series are fair.")
           + '<h2>What it shows</h2><ol>'
           f'<li><b>Accuracy shows no dependence on engine, kpool fixes, draft depth or weights</b> here: every configuration lands at '
           f'{min(accs):.1f}-{max(accs):.1f}%, and every per-question comparison of first passes is consistent with noise (sign-test p &ge; {min_p:.2f}).</li>'
           f'<li><b>Finishing long reasoning depends on the engine.</b> On the same {esc(weights_label(g07["cfg"]))} weights, {esc(engine_label(g07["cfg"]))} left '
           f'{e07} of {n0708} GPQA answers empty against {e08} for {esc(engine_label(g08["cfg"]))}, about a third as many (Fisher p = {p0708:.3f}); on the '
           f'hard-question screen it failed about half as often ({sc07} vs {sc08} of 40 in the same session).</li>'
           '<li><b>The two kpool bugs are real but are not the main cause of the loops.</b> The fixes, shipped in tpurtell 0.9.0, pass upstream\'s '
           'regression tests, at no measurable accuracy cost. Their expected benefit is in long-lived servers with prefix caching, which these '
           'fresh-server runs rarely exercise.' + k4_signal(runs) + '</li>'
           f'<li><b>Empty answers explain only part of the gap to published scores</b> (NVIDIA 92.1, Red Hat 90.6): scoring only answered questions '
           f'would add {min(gain):.1f}-{max(gain):.1f} points, but the questions that go unanswered are harder ({100 * sum(hard) / len(hard):.1f}% correct when '
           f'answered, vs {100 * sum(easy) / len(easy):.1f}% for the others). Credited at their own observed accuracy, completing them would add '
           f'{min(cgain):.1f}-{max(cgain):.1f} points ({min(credited):.1f}-{max(credited):.1f}%), still below both. Raw scores also run about 0.7 points '
           f'low (see the <a href="method.html#scoring">scoring note</a>). The rest mixes quantization, harness and other runtime effects, which these runs cannot separate.</li>'
           '<li><b>Under tpurtell 0.8.0/0.9.0\'s default DCP1 layout, a masking path in the vendored attention code drops the newest 1-3 '
           'tokens from decode attention</b> until the context reaches 2,044 tokens. The DCP1 tail fix (tpurtell PR #6, released in 0.9.1) brings decode much closer to '
           'prefill and turned two failing tool-calling scenarios into passes. On a preregistered screen, loop failures fell from 41 to 33 of 80, '
           'toward the v0.7.0 layout\'s 29: a consistent direction, not yet statistically significant at this sample size. See the '
           '<a href="looping.html">looping investigation</a>.</li>'
           '</ol><p>Full write-up: <a href="' + REPO + '/blob/main/FINDINGS.md">FINDINGS.md</a>.</p>')
    (OUT / "index.html").write_text(page("index.html", "Overview", idx, key_open))

    # ---- gpqa
    ptr = [[esc(re.sub(r" \(\d pass(es)?\)$", "", a)), esc(re.sub(r" \(\d pass(es)?\)$", "", b)), x, y, f"{pv:.2f}"] for a, b, x, y, pv in pairs]
    pertr = [[esc(r["label"])] + [f"{a:.1f}%" for a in acc] for r, acc, _, _ in per_pass]
    pub = table(["Source", "Weights", "GPQA Diamond", "Stated protocol"],
                [["NVIDIA model card", "BF16", "92.17", "temp 1.0, top_p 0.95, 327,680 max new tokens; harness not stated"],
                 ["NVIDIA model card", "NVFP4", "92.11", "same"],
                 ["Red Hat model card", "NVFP4", "90.57", "lm-eval / lighteval forks, vLLM, 3 seeds averaged"],
                 ["This repository", "EXL3 3.25bpw / 4bpw", "{:.1f}-{:.1f}".format(*(100 * f(r["s"]["accuracy_flexible"] for r in full) for f in (min, max))),
                  "full 3-pass runs (95% intervals {:.1f}-{:.1f}); see the GPQA protocol".format(
                      min(100 * r["s"]["accuracy_flexible_ci95"][0] for r in full), max(100 * r["s"]["accuracy_flexible_ci95"][1] for r in full))]])
    gp = ('<h1>GPQA Diamond</h1><p class="lede">198 graduate-level multiple-choice questions, lm-evaluation-harness, temperature 1.0, '
          'top_p 0.95, 327,680-token budget, thinking on, 8 concurrent requests. Results carry question ids and hashes only: the dataset '
          'asks that its questions not be published in plain text.</p>'
          + gpqa_accuracy_fig(allg, "Accuracy, all runs") + gpqa_empty_fig(allg, "Empty answers, all runs")
          + f'<h2>Single passes</h2><p>Each full run\'s passes, to show how much one pass moves: {spread[0]:.1f}-{spread[1]:.1f} points between a run\'s '
            f'best and worst pass. {inside} of {3 * len(per_pass)} passes fall inside their run\'s 95% interval.</p>'
          + table(["Configuration", "Pass 1", "Pass 2", "Pass 3"], pertr, numeric=(1, 2, 3))
          + '<h2>Question-by-question, first pass</h2><p>Every pass of every run sent identical prompts with the same request seed (1234; see the '
            '<a href="' + REPO + '/blob/main/protocols/gpqa-diamond/v1.md">protocol</a>), so configurations can be compared per question. First passes '
            'are paired here because the single-pass runs have no other. "Only A right" counts questions A answered correctly and B did not. '
            'A sign-test p near 1 means no detectable difference.</p>'
          + table(["A", "B", "Only A right", "Only B right", "p"], ptr, numeric=(2, 3, 4))
          + '<h2>Published numbers, for context only</h2><div class="note">These use other weights (BF16, NVFP4) and an unstated or partly '
            'stated harness. The same NVFP4 weights score 92.1 (NVIDIA) and 90.6 (Red Hat), so harness alone moves the score by about 1.5 '
            'points. They are not plotted against the local runs.</div>' + pub)
    (OUT / "gpqa.html").write_text(page("gpqa.html", "GPQA Diamond", gp, key_closed))

    # ---- screens
    sc = ('<h1>Hard-question screen</h1><p class="lede">The five GPQA questions that most often came back empty, each run 8 times '
          '(40 runs per configuration), to compare how often configurations fail to finish long reasoning. A loop is repetitive reasoning '
          'to the cap; an exhaustion is varied reasoning that runs out of budget.</p>'
          '<div class="note"><b>Read with care.</b> The questions were chosen from empty answers of tpurtell 0.8.0 (no kpool fixes): comparisons '
          'within one engine series are fair, but the gap between series here is an upper-end estimate (the full GPQA runs carry that finding). '
          'Repeats use one fixed seed and still differ because concurrent batching changes the arithmetic. Outcomes cluster by question, so the '
          'Wilson intervals are too narrow.</div>'
          + screen_fig(screen_rows(runs, "hard-prompt-screen/v1"), "Failures out of 40 (protocol v1, with early loop stop)",
                       "Line = 95% Wilson interval. Rows with the same date ran in one session.")
          + screen_fig(screen_rows(runs, "hard-prompt-screen/v0"), "Failures out of 40 (protocol v0, 2026-09-30, no early stop)",
                       "Same questions and settings without the early stop; kept separate from v1 by the comparison rules. Line = 95% Wilson interval.")
          + heatmap(runs))
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
    sv = ('<h1>Speed and speculative acceptance</h1><p class="lede">3.25bpw weights, 16 fixed GPQA prompts, 4,096 tokens, temperature 1.0, one run '
          'per configuration: tpurtell 0.8.0 with and without the kpool fixes (2026-10-04), and tpurtell 0.9.0 with and without the DCP1 tail fix '
          '(2026-10-08; see the <a href="looping.html">looping investigation</a>). The kpool fixes show no consistent change: with the same speculation '
          f'setting, decode speed and throughput move by {min(moves):+.0f}% to {max(moves):+.0f}% and acceptance by at most {acc_move:.3f} (see the '
          f'<a href="{REPO}/tree/main/comparisons/glm53-flash-serving-probe">comparison</a>).</p>'
          + bars("median_decode_tok_s", "sampled_c1", " tok/s", "Decode speed, one request at a time", "Median per-request decode tokens per second.", 180)
          + bars("aggregate_tok_s", "sampled_c8", " tok/s", "Total throughput, 8 requests at once", "Generated tokens per second across all requests.", 400)
          + bars("acceptance_rate", "sampled_c8", "", "Draft acceptance rate, 8 requests at once", "Accepted / drafted tokens from the server's counters.", 0.6)
          + '<h2>Greedy agreement</h2><p>Temperature 0, one request at a time: characters of identical output before two runs diverge. '
            f'Even the same configuration run twice diverges early (median {floor["median_shared_prefix_chars"]:.0f} characters), so this cannot '
            'certify that speculative decoding is exact.</p>'
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
          + table(hdr, ktr))
    (OUT / "kernels.html").write_text(page("kernels.html", "Kernel tests", kp, key_closed))

    # ---- looping investigation
    (OUT / "looping.html").write_text(page("looping.html", "Looping investigation", looping(runs), key_closed))

    # ---- method
    mt = ('<h1>How to read this</h1>'
          '<h2>Labels</h2><p>Every configuration is named <i>weights · engine version · speculation</i>, built from its configuration file by one rule '
          f'(<a href="{REPO}/blob/main/SCHEMA.md#labels">SCHEMA.md</a>). The engine name comes first because engines number their versions '
          'independently. Chart colours mark engine series (builds that share one code base); the legend under each chart lists the labels each colour covers.</p>'
          '<h2 id="scoring">Scoring</h2><p>GPQA scores are lm-eval\'s raw <code>flexible-extract</code> filter. It matches any parenthesised '
          'capital letter, so notation in a reply, such as (H) or (R) in chemistry, can be read as the answer. The misread is deterministic: the '
          'same reply is always scored the same way, on particular questions, and it only lowers scores. A hand check of every reply where the two '
          'lm-eval filters disagree puts raw scores 0 to 2 points low per pass, about 0.7 on average. Raw scores stay the headline so runs remain '
          'comparable; read them as slightly low, not as exact accuracy.</p>'
          '<h2>Intervals</h2><p>GPQA accuracy: 95% bootstrap over questions, because each question\'s three passes are not independent. '
          'Rates (empty answers, screen failures): 95% Wilson score intervals. Where intervals overlap, the configurations cannot be told apart.</p>'
          '<h2>Concurrency changes the arithmetic</h2><p>Requests are served 8 at a time, and batch composition changes the numerics inside the engine. '
          'A fixed seed therefore does not reproduce a response, and even one greedy request at a time diverges from its own rerun after a median of '
          f'{floor["median_shared_prefix_chars"]:.0f} characters. Compare distributions (accuracy, failure rates), never individual transcripts.</p>'
          f'<h2>What reproduces</h2><p>Scores reproduce statistically: at temperature 1.0 every pass is a fresh draw, a full run\'s passes differ by '
          f'{spread[0]:.1f}-{spread[1]:.1f} points, and {inside} of {3 * len(per_pass)} fall inside their run\'s interval. Match the concurrency (8) as well as '
          'the sampling settings when rerunning. The analysis reproduces exactly: '
          '<code>tools/verify.py</code> and <code>tools/analyze.py</code> recompute every number from the published rows.</p>'
          '<h2>Comparability</h2><p>Comparisons only line up runs with the same protocol version and hardware, and declare which configuration fields differ; '
          '<code>tools/verify.py</code> enforces this. Published numbers from other harnesses are shown as context, never plotted as a ranking.</p>'
          f'<h2>Sources</h2><ul><li><a href="{REPO}/tree/main/protocols">Protocols</a> - exact settings per benchmark version</li>'
          f'<li><a href="{REPO}/tree/main/configs">Configurations</a> - engine image digests, model revisions, settings</li>'
          f'<li><a href="{REPO}/blob/main/DATASHEET.md">Datasheet</a> - what the data is, and is not, suitable for</li>'
          f'<li><a href="{REPO}/blob/main/FINDINGS.md">Findings</a> - the full write-up</li></ul>')
    (OUT / "method.html").write_text(page("method.html", "How to read this", mt, key_open))
    print(f"built {len(PAGES)} pages into {OUT}")


if __name__ == "__main__":
    build()
