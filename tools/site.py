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

from verify import config_label, engine_label, summarize, weights_label
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
    ("pass", r"\b[Pp]ass(?:es)? [123]\b", "Pass", "One run of all 198 GPQA questions. Pass <i>p</i> sends request seed 1233 + <i>p</i> (1234, 1235, 1236), so "
     "passes of one configuration are independent draws, and pass <i>p</i> of two configurations shares its seed, which pairs them question by question."),
    ("kv-pool", r"\bKV pool\b", "KV pool", "The KV cache capacity, in tokens, that the server reports at start-up. When running requests fill it, "
     "further requests wait (<i>KV-saturated</i>), so fewer run at once than the client sends."),
    ("clustered", r"\b(?:question-)?clustered\b", "Question-clustered test", "A comparison that treats a question answered in three passes as one unit, "
     "not three: here an exact sign-flip test over questions (each question's difference, summed over its passes, keeps or flips its sign) and "
     "intervals that resample questions. A pooled test that counts every question-pass separately overstates the evidence."),
]
SKIP_TAGS = {"a", "svg", "h1", "h2", "h3", "code", "title", "style", "script", "th", "nav", "summary"}
TERM = {slug: f"{re.sub('<[^>]+>', '', term)}||{re.sub('<[^>]+>', '', d)}" for slug, _, term, d in GLOSSARY}   # hover/focus tooltip text


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
                        segs[n:n + 1] = [seg[:m.start()], (f'<a class="g" href="{target}#g-{slug}" data-tip="{esc(TERM[slug])}">{m.group(0)}</a>',), seg[m.end():]]
                        done.add(slug); break
        out.append("".join(x[0] if isinstance(x, tuple) else x for x in segs))
    return "".join(out)


def glossary_html():
    return ('<h2 id="glossary">Glossary</h2><dl class="gl">' + "".join(f'<dt id="g-{slug}">{term}</dt><dd>{d}</dd>' for slug, _, term, d in GLOSSARY)
            + '</dl>')


# ---------------------------------------------------------------- markdown documents (CONFOUNDS.md, LEDGER.md) rendered as site pages

SITE = "https://urbanastrola.github.io/local-inference-evals/"
# repository documents that have a site page of their own, and how their section anchors map onto it
DOC_PAGES = {"LEDGER.md": ("ledger.html", {}), "CONFOUNDS.md": ("confounds.html", {}),
             "investigations/2026-10-glm53-looping/README.md": ("looping.html", {
                 "the-dcp1-tail-issue-and-its-fix": "tail-bug", "non-completion-what-the-clean-data-shows": "non-completion",
                 "open-questions": "open-questions"}),
             "investigations/2026-10-glm53-looping": ("looping.html", {}),
             "README.md": (REPO + "#readme", {"glossary": "method.html#glossary"})}


def slug(text):
    """GitHub's heading anchor: lower case, punctuation dropped, spaces to hyphens."""
    t = re.sub(r"<[^>]+>", "", text).strip().lower()
    return re.sub(r"\s", "-", re.sub(r"[^\w\- ]", "", t))


def md_href(href, src):
    """A link in a repository document, as the site should point it."""
    if href.startswith(SITE):
        rest = href[len(SITE):]
        return "index.html" + rest if not rest or rest.startswith("#") else rest
    if href.startswith(("http://", "https://", "mailto:")) or href.startswith("#"):
        return href
    path, _, frag = href.partition("#")
    full = (src.parent / path).resolve().relative_to(ROOT).as_posix() if path else src.relative_to(ROOT).as_posix()
    if full in DOC_PAGES:
        page_, frags = DOC_PAGES[full]
        if frag in frags:
            t = frags[frag]
            return t if "." in t.split("#")[0] else f"{page_}#{t}"
        return page_ + (f"#{frag}" if frag and "#readme" not in page_ else "")
    kind = "tree" if (ROOT / full).is_dir() else "blob"
    return f"{REPO}/{kind}/main/{full}" + (f"#{frag}" if frag else "")


def md_inline(t, src):
    codes = []
    def keep(m):
        codes.append(f"<code>{esc(m.group(1))}</code>"); return f"\x00{len(codes) - 1}\x00"
    t = esc(re.sub(r"`([^`]+)`", keep, t), quote=False)
    t = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", lambda m: f'<a href="{esc(md_href(m.group(2), src))}">{m.group(1)}</a>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*\s][^*]*)\*(?![\w*])", r"<i>\1</i>", t)
    return re.sub("\x00(\\d+)\x00", lambda m: codes[int(m.group(1))], t)


def md_html(src, figures):
    """Render a repository markdown document: headings, paragraphs, lists, tables, quotes, code, <details> blocks and
    <!-- figure:name --> placeholders (replaced by figures[name]). Enough for this repository's documents, not general."""
    lines = src.read_text().splitlines() + [""]
    out, i = [], 0
    inl = lambda t: md_inline(t, src)
    while i < len(lines):
        ln = lines[i]; st_ = ln.strip()
        if not st_:
            i += 1; continue
        m = re.match(r"<!-- figure:([\w-]+) -->", st_)
        if m:
            out.append(figures.get(m.group(1), "")); i += 1; continue
        if re.match(r"</?details>|<summary>|<a id=", st_):
            if st_.startswith("<summary>"):
                st_ = "<summary>" + inl(st_[len("<summary>"):-len("</summary>")]) + "</summary>"
            elif st_ == "<details>":
                st_ = '<details class="more">'
            out.append(st_); i += 1; continue
        m = re.match(r"(#{1,4}) (.*)", ln)
        if m:
            n = len(m.group(1)); out.append(f'<h{n} id="{slug(m.group(2))}">{inl(m.group(2))}</h{n}>'); i += 1; continue
        if st_.startswith("```"):
            j = i + 1
            while not lines[j].strip().startswith("```"): j += 1
            out.append("<pre><code>" + esc("\n".join(lines[i + 1:j])) + "</code></pre>"); i = j + 1; continue
        if st_.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split(" | ")]); i += 1
            th = "".join(f"<th>{inl(h)}</th>" for h in rows[0])
            trs = "".join("<tr>" + "".join(f"<td>{inl(c)}</td>" for c in r) + "</tr>" for r in rows[2:])
            out.append(f'<div class="tbl"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>')
            continue
        if st_.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip()); i += 1
            paras = "\n".join(buf).split("\n\n")
            out.append('<div class="note">' + "".join(f"<p>{inl(' '.join(q.split()))}</p>" for q in paras) + "</div>"); continue
        m = re.match(r"(\s*)(- |\d+\. )", ln)
        if m:
            ordered = m.group(2)[0].isdigit(); items = []
            start = int(m.group(2)[:-2]) if ordered else 1
            while i < len(lines) and lines[i].strip():
                mm = re.match(r"(\s*)(- |\d+\. )(.*)", lines[i])
                if mm and len(mm.group(1)) == len(m.group(1)):
                    items.append(mm.group(3))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f'<{tag}{f" start={chr(34)}{start}{chr(34)}" if ordered and start != 1 else ""}>' + "".join(f"<li>{inl(x)}</li>" for x in items) + f"</{tag}>")
            continue
        buf = []
        while i < len(lines) and lines[i].strip() and not re.match(r"\s*(#|\||>|- |\d+\. |```|<)", lines[i]):
            buf.append(lines[i].strip()); i += 1
        if not buf:
            out.append(inl(st_)); i += 1; continue
        out.append(f"<p>{inl(' '.join(buf))}</p>")
    return "\n".join(out)


def tip(text, detail):
    """Text with its context in a tooltip (hover or keyboard focus) instead of an inline parenthesis."""
    plain = re.sub(r"<[^>]+>", "", text)
    return f'<span class="tipped" tabindex="0" data-tip="{esc(plain)}||{esc(detail)}">{text}</span>'


def more(summary, body, open_=False):
    return f'<details class="more"{" open" if open_ else ""}><summary>{summary}</summary>{body}</details>'


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
.tile .lab{color:var(--ink2);font-size:13px}.tile .val{font-size:clamp(22px,2.4vw,26px);font-weight:600;margin:2px 0}.tile .sub{color:var(--muted);font-size:13px}
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
h3{font-size:16px;margin:26px 0 4px}.shows{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:4px 18px 8px;margin:16px 0}
.shows h2{margin:12px 0 4px}.shows li{margin:6px 0}.shows .ev{font-size:13px}
.path{font-size:14px;color:var(--ink2);margin:10px 0}.path a{margin-right:4px}
footer nav{padding:10px 0;border:0;margin:0;max-width:none}
details.more{border-left:3px solid var(--grid);padding:2px 0 2px 12px;margin:10px 0}details.more>summary{font-size:14px}
details.more[open]>summary{margin-bottom:6px}.tipped{text-decoration:underline dotted;text-underline-offset:2px;cursor:help}
.grade{font-weight:600;color:var(--ink)}.shows li .ev{white-space:nowrap}pre{overflow-x:auto;background:var(--surface);border:1px solid var(--ring);
border-radius:8px;padding:10px;font-size:12.5px}h4{font-size:15px;margin:18px 0 4px}.toc{font-size:14px;color:var(--ink2)}
"""

JS = """
const tip=document.getElementById('tip');
function show(e,el){const [v,l]=el.dataset.tip.split('||');tip.replaceChildren();const b=document.createElement('b');b.textContent=v;
tip.append(b,document.createTextNode(l||''));tip.style.display='block';const r=el.getBoundingClientRect();
const x=(e&&e.clientX)||r.left+r.width/2,y=(e&&e.clientY)||r.top;tip.style.left=Math.min(x+14,innerWidth-330)+'px';tip.style.top=(y+14)+'px';}
document.querySelectorAll('[data-tip]').forEach(el=>{el.addEventListener('pointermove',e=>show(e,el));
el.addEventListener('focus',()=>show(null,el));['pointerleave','blur'].forEach(t=>el.addEventListener(t,()=>tip.style.display='none'));});
document.addEventListener('keydown',e=>{if(e.key==='Escape')tip.style.display='none';});
function openTarget(){const id=decodeURIComponent(location.hash.slice(1));const t=id&&document.getElementById(id);if(!t)return;
let d=t.closest('details');while(d){d.open=true;d=d.parentElement.closest('details');}t.scrollIntoView();}
addEventListener('hashchange',openTarget);openTarget();
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
    concs = sorted({c["serving"]["concurrency"] for c in cfgs if c["serving"].get("concurrency")})
    for n in concs:
        rows.append([f"concurrency {n}", f'The client kept {n} requests in flight instead of the protocol\'s setting (GPQA: 8). Shown only where it '
                     'differs; the configuration notes say why.'])
    for (lab, dm) in spec:
        rows.append([f"{esc(lab)} \u00d7N", f'Speculative decoding with {esc(lab)}, N draft tokens per step (draft model {esc(dm)}). '
                     '"sharing off": draft-slot sharing disabled. "no speculation": plain decoding.'])
    eqs = [(lab, c) for lab, c in eng.items() if c["engine"].get("equivalent_to")]
    lead = "".join(f' <b>{esc(lab)}</b> is the same engine as <b>{esc(c["engine"]["name"])} {esc(c["engine"]["equivalent_to"]["version"].lstrip("v"))}</b> for these measurements.' for lab, c in eqs[:1])
    return (f'<details class="key"{" open" if open_ else ""}><summary><b>Labels</b> read <i>weights · engine version · speculation</i>.{lead} '
            f'What each label means</summary>{table(["Label", "Meaning"], rows)}<p style="font-size:13px">Labels are built from the '
            f'configuration files by one rule (<a href="{REPO}/blob/main/SCHEMA.md#labels">SCHEMA.md</a>): engine name first, then its own version, '
            f'then any locally applied patches.</p></details>')


def axis_title(x, y, text):
    return f'<text x="{x:.1f}" y="{y}" text-anchor="middle" class="muted">{esc(text)}</text>' if text else ""


def interval_plot(rows, lo, hi, ticks, fmt, unit="", axis=""):
    """rows: dicts with label, est, lo, hi, color, tip. Horizontal dot-and-whisker, one row per config; `axis` titles the x axis."""
    lw, rh, top = label_width(r["label"] for r in rows), 34, 10
    pw = 440; w = lw + pw + 30
    x = lambda v: lw + (v - lo) / (hi - lo) * pw
    h = top + rh * len(rows) + 30 + (20 if axis else 0)
    base = h - (20 if axis else 0)
    s = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">']
    for t in ticks:
        s.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" y2="{base - 26}"/>'
                 f'<text x="{x(t):.1f}" y="{base - 10}" text-anchor="middle" class="muted">{fmt(t)}{unit}</text>')
    s.append(axis_title(lw + pw / 2, h - 6, axis))
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


def figure(title, sub, body, tbl, legend_html="", fid=None):
    """A chart with its takeaway as the title, a subtitle (what is plotted, the grade), a legend, and a table view."""
    body = body.replace('role="img">', f'role="img" aria-label="{esc(title)}">')
    return (f'<figure{f" id={chr(34)}{fid}{chr(34)}" if fid else ""}><figcaption><div class="t">{esc(title)}</div><div class="s">{sub}</div></figcaption>'
            f'{legend_html}<div class="chart">{body}</div>'
            f'<details><summary>Show the numbers as a table</summary>{tbl}</details></figure>')


# ---------------------------------------------------------------- pages

PAGES = [("index.html", "Results"), ("confounds.html", "Confounds"), ("ledger.html", "Ledger"), ("looping.html", "Investigation"),
         ("method.html", "Method")]
# pages merged into others: old URL -> {old anchor: new place}, "" = where the page itself now lives
MOVED = {
    "gpqa.html": ("GPQA Diamond", {"": "index.html#gpqa", "records": "index.html#records", "records-accuracy": "index.html#records",
                                   "records-empty": "index.html#records", "empty-questions": "index.html#empty-questions",
                                   "records-compare": "index.html#records-compare", "pass1": "index.html#pass1",
                                   "pass1-accuracy": "index.html#pass1", "pass1-empty": "index.html#pass1",
                                   "pass1-pairs": "index.html#pass1-pairs", "published": "index.html#published"}),
    "screens.html": ("Hard-question screen", {"": "looping.html#non-completion", "screen-q88": "looping.html#screen-q88",
                                              "screen-q79": "looping.html#screen-q79", "screen-scope": "looping.html#screen-scope",
                                              "seed-control": "confounds.html#seed-control",
                                              "withdrawal": "ledger.html#withdrawn-and-what-replaced-it", "single-draws": "ledger.html#single-draws"}),
    "serving.html": ("Speed and acceptance", {"": "index.html#serving", "decode-speed": "index.html#decode-speed",
                                              "throughput": "index.html#throughput", "acceptance": "index.html#acceptance",
                                              "kv": "index.html#kv", "greedy": "confounds.html#greedy"}),
    "kernels.html": ("Kernel tests", {"": "index.html#kernels", "per-image": "index.html#kernels", "grid": "index.html#kernel-grid"}),
}


def with_key(body, key_html):
    """Label key right after the page's lede (or right after its heading if a page has none)."""
    i = body.find('class="lede"')
    j = body.find("</p>", i) + 4 if i >= 0 else body.find("</h1>") + 5
    return body[:j] + key_html + body[j:]


def shows(items, title="What it shows"):
    """The page's answer first: graded statements, each with a link to its evidence."""
    return (f'<div class="shows"><h2>{esc(title)}</h2><ol>' + "".join(f'<li>{t}' + (f' <span class="ev">{ev}</span>' if ev else "") + '</li>'
                                                                       for t, ev in items) + '</ol></div>')


def page(name, title, body, key_html):
    body_html = with_key(body, key_html)
    if name != "method.html":
        body_html = gloss(body_html)
    nav = "".join(f'<a href="{f}" class="{"on" if f == name else ""}">{esc(t)}</a>' for f, t in PAGES)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(title)} - local-inference-evals</title><style>{CSS}</style></head><body>'
            f'<nav><span class="brand">local-inference-evals</span>{nav}</nav><main>{body_html}</main>'
            f'<footer><nav aria-label="Repository documents"><a href="{REPO}#readme">README</a><a href="{REPO}/blob/main/FINDINGS.md">FINDINGS</a>'
            f'<a href="{INV}">Investigation (write-up)</a><a href="{REPO}/tree/main/comparisons">Comparisons</a>'
            f'<a href="{REPO}/blob/main/DATASHEET.md">Datasheet</a><a href="{REPO}/blob/main/CHANGELOG.md">Changelog</a></nav>'
            f'Generated from the repository data by <code>tools/site.py</code>. Receipts, protocols and code: '
            f'<a href="{REPO}">{REPO.replace("https://", "")}</a>. Results CC BY 4.0, code Apache-2.0.</footer>'
            f'<div id="tip" role="status"></div><script>{JS}</script></body></html>')


def gpqa_rows(runs):
    """Pass 1 of every GPQA run (request seed 1234, the pass every configuration has). s = pass-1 summary; pct from the rows."""
    out = []
    for rid, m in runs.items():
        if m["protocol"] != "gpqa-diamond/v1":
            continue
        rs = [r for r in m["rows"] if r["pass"] == 1]
        s = summarize(m["protocol"], rs)
        kv = ("KV-saturated at 8 concurrent (logged)" if "KV-SATURATED" in m["notes"] else
              "8 concurrent; KV pool holds about 4 requests at the cap (no server log)" if "whether requests waited for KV is not recorded" in m["notes"] else
              f'{m["cfg"]["serving"]["concurrency"]} concurrent' if m["cfg"]["serving"].get("concurrency") else "8 concurrent")
        out.append(dict(rid=rid, label=label(m["cfg"]), cfg=m["cfg"], color=color(m["cfg"]), s=s, n=len(rs), rows=rs, kv=kv,
                        acc=analyze.pct(rs, "correct_flexible"), stated=analyze.pct(rs, "correct_stated"),
                        acc_ans=analyze.pct(rs, "correct_flexible", True), stated_ans=analyze.pct(rs, "correct_stated", True),
                        passes=len({r["pass"] for r in m["rows"]})))
    return sorted(out, key=lambda r: (list(SERIES).index(r["cfg"]["engine"]["series"]), r["label"]))


# ---------------------------------------------------------------- GPQA records (several passes, one request seed per pass)

def record_rows(runs):
    out = []
    for rid, m in runs.items():
        if m["protocol"] != "gpqa-diamond/v1" or len(m["summary"]["passes"]) < 2:
            continue
        ps = m["summary"]["passes"]
        out.append(dict(rid=rid, m=m, label=label(m["cfg"]), cfg=m["cfg"], color=color(m["cfg"]), s=m["summary"], passes=ps,
                        per={p: [r for r in m["rows"] if r["pass"] == p] for p in ps}))
    return sorted(out, key=lambda r: (list(SERIES).index(r["cfg"]["engine"]["series"]), r["label"]))


MARKERS = ('<span><svg width="14" height="12" style="display:inline;width:14px;vertical-align:-1px"><circle cx="7" cy="6" r="4" fill="none" '
           'stroke="var(--ink2)" stroke-width="2"/></svg>one pass (number = pass; its own request seed)</span>'
           '<span><svg width="14" height="12" style="display:inline;width:14px;vertical-align:-1px"><circle cx="7" cy="6" r="5" fill="var(--ink2)"/>'
           '</svg>mean of the passes, with its 95% interval over questions</span>')


def records_summary_fig(recs):
    """Overview: the three-pass records as two small multiples sharing one row per record (accuracy | empty answers per pass).
    Per-pass numbers are in the table view and the table above the chart."""
    lines = [(r, [weights_label(r["cfg"]) + " · " + engine_label(r["cfg"]), label(r["cfg"]).split(engine_label(r["cfg"]) + " · ", 1)[-1]]) for r in recs]
    lw = int(18 + 7.0 * max(len(t) for _, ls in lines for t in ls)); pw, gap, rmargin, rh, top = 230, 36, 58, 46, 34
    panels = [("correct_flexible", "Raw accuracy per pass, %", 80, 95, [80, 85, 90, 95]), ("empty", "Empty answers per pass, of 198", 0, 10, [0, 2, 4, 6, 8, 10])]
    x0s = [lw, lw + pw + rmargin + gap]; w = x0s[1] + pw + rmargin; h = top + rh * len(recs) + 28
    s = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">']
    tr = []
    for (key, head, lo, hi, ticks), x0 in zip(panels, x0s):
        x = lambda v, x0=x0, lo=lo, hi=hi: x0 + (v - lo) / (hi - lo) * pw
        s.append(f'<text x="{x0 + pw / 2:.1f}" y="14" text-anchor="middle" style="font-weight:600">{esc(head)}</text>')
        for t in ticks:
            s.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 8}" y2="{h - 26}"/>'
                     f'<text x="{x(t):.1f}" y="{h - 10}" text-anchor="middle" class="muted">{t}</text>')
        for i, (r, _) in enumerate(lines):
            y = top + rh * i + rh / 2 + 4; c = r["color"]
            if key == "correct_flexible":
                pv = [(p, analyze.pct(r["per"][p], key)) for p in r["passes"]]
                mean = analyze.pct(r["m"]["rows"], key); lo_, hi_ = (100 * v for v in r["s"]["accuracy_flexible_ci95"]); f = lambda v: f"{v:.1f}%"
            else:
                pv = [(p, sum(x_[key] for x_ in r["per"][p])) for p in r["passes"]]
                mean = sum(v for _, v in pv) / len(pv); lo_, hi_ = bootstrap_mean(r["m"]["rows"], key); f = lambda v: f"{v:.1f}"
            tt = f'{head}: mean {f(mean)}||{r["label"]}: ' + "; ".join(f"pass {p} {v:.1f}" if key == "correct_flexible" else f"pass {p} {v}" for p, v in pv)
            s.append(f'<rect class="hit" x="{x0}" y="{y - rh / 2:.1f}" width="{pw}" height="{rh}" tabindex="0" data-tip="{esc(tt)}"/><g>'
                     f'<line x1="{x(lo_):.1f}" x2="{x(hi_):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{c}" stroke-width="2" stroke-linecap="round"/>')
            at = {}
            for p, v in pv: at.setdefault(round(v, 6), []).append(p)
            last = None
            for v, ps in sorted(at.items()):
                up = 9 if last is not None and x(v) - last < 14 else 0; last = x(v) if not up else None   # stagger labels of close markers
                s.append(f'<circle cx="{x(v):.1f}" cy="{y - 10:.1f}" r="4.5" fill="var(--surface)" stroke="{c}" stroke-width="2"/>'
                         f'<text x="{x(v):.1f}" y="{y - 17 - up:.1f}" text-anchor="middle" class="muted" style="font-size:10px">{",".join(map(str, ps))}</text>')
            s.append(f'<circle cx="{x(mean):.1f}" cy="{y:.1f}" r="5.5" fill="{c}" stroke="var(--surface)" stroke-width="2"/></g>'
                     f'<text x="{x0 + pw + 8:.1f}" y="{y + 4:.1f}" class="muted">{esc(f(mean))}</text>')
            if key == "correct_flexible":
                tr.append([esc(r["label"]), " / ".join(f"{v:.1f}" for _, v in pv) + "%", f"{mean:.1f}% ({lo_:.1f}-{hi_:.1f})"])
            else:
                tr[i] += [" / ".join(str(v) for _, v in pv), f'{sum(v for _, v in pv)} of {198 * len(pv)}', r["s"]["questions_ever_empty"], link(r["rid"])]
    for i, (r, ls) in enumerate(lines):
        y = top + rh * i + rh / 2 + 4
        s.append(f'<text x="{lw - 12}" y="{y - 3:.1f}" text-anchor="end">{esc(ls[0])}</text>'
                 f'<text x="{lw - 12}" y="{y + 12:.1f}" text-anchor="end" class="muted">{esc(ls[1])}</text>')
    s.append("</svg>")
    g = {r["m"]["config"].split("/", 1)[1]: r["m"] for r in recs}
    e7 = analyze.records(g["k3.25-v0.7.0-dflash5"], g["k3.25-v0.9.1-dflash3"], "empty")
    a7 = analyze.records(g["k3.25-v0.7.0-dflash5"], g["k3.25-v0.9.1-dflash3"], "correct_flexible")
    leg = legend([r["cfg"] for r in recs])[:-6] + MARKERS + "</div>"
    return figure(f"Three passes per configuration: tpurtell 0.7.0 left {e7['a']} empty answers of {e7['n']}, 0.9.1 left {e7['b']}; "
                  "accuracy does not differ measurably",
                  "One request seed per pass (1234, 1235, 1236). Hollow marker = one pass (numbered); filled dot = mean of the three passes; "
                  "line = 95% interval over questions. "
                  + tip("Statistics", f"Empty answers, 0.7.0 vs 0.9.1 (3.25bpw): supported, question-clustered p = {e7['p_cluster']:.3f}. "
                        f"Accuracy: clustered p = {a7['p_cluster']:.2f}. 3.25bpw vs 4bpw TR3 (Brandon) on 0.9.1: no measurable difference (descriptive).")
                  + ".", "".join(s),
                  table(["Configuration", "Raw, passes 1 / 2 / 3", "Raw, mean (95% CI)", "Empty, passes 1 / 2 / 3", "Empty, all passes",
                         "Questions ever empty", "Receipts"], tr, numeric=(5,)), leg, "records")


def bootstrap_mean(rows, key, n_boot=2000, seed=0):
    """95% interval of the per-pass mean count of `key`, resampling questions with all their passes (as verify.bootstrap_by_item)."""
    by = {}
    for r in rows: by.setdefault(r["doc_id"], []).append(1.0 if r[key] else 0.0)
    ids = sorted(by); rng = random.Random(seed); np_ = len(rows) / len(ids); means = []
    for _ in range(n_boot):
        pick = [by[rng.choice(ids)] for _ in ids]; means.append(sum(x for g in pick for x in g) / np_)
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


SHADE = {0: "transparent", 1: "var(--seq2)", 2: "var(--seq5)", 3: "var(--seq7)"}   # white text on 2 and 3 meets 4.5:1


def empty_grid_fig(recs):
    """Questions x records: how many of the passes came back empty, which passes, and how they ended."""
    docs = sorted({x["doc_id"] for r in recs for x in r["m"]["rows"] if x["empty"]})
    cw, rh, top, lw = 150, 24, 54, 70
    w, h = lw + cw * len(recs) + 10, top + rh * len(docs) + 8
    s = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" style="max-width:{w}px">']
    for j, r in enumerate(recs):
        cx = lw + cw * j + cw / 2
        for k, part in enumerate(short_label(r["cfg"])):
            s.append(f'<text x="{cx:.1f}" y="{16 + 14 * k}" text-anchor="middle" style="font-size:12px">{esc(part)}</text>')
    tr = []
    for i, d in enumerate(docs):
        y = top + rh * i
        s.append(f'<text x="{lw - 10}" y="{y + rh / 2 + 4:.1f}" text-anchor="end">q{d}</text>')
        row = [f"q{d}"]
        for j, r in enumerate(recs):
            e = sorted((x for x in r["m"]["rows"] if x["doc_id"] == d and x["empty"]), key=lambda x: x["pass"])
            n = len(e); x0 = lw + cw * j + 3
            txt = ", ".join(f"p{x['pass']}" for x in e) or "-"
            ends = ", ".join(f"pass {x['pass']}: " + {"length": "ran to the token cap", "stop": "stopped without an answer"}.get(x["finish_reason"], "not logged") for x in e)
            s.append(f'<rect x="{x0:.1f}" y="{y + 2}" width="{cw - 6}" height="{rh - 4}" rx="4" style="fill:{SHADE[n]};stroke:var(--grid)"/>'
                     f'<text x="{x0 + (cw - 6) / 2:.1f}" y="{y + rh / 2 + 4:.1f}" text-anchor="middle" style="fill:{"#fff" if n >= 2 else "var(--ink2)"};font-size:12px">{txt}</text>'
                     f'<rect class="hit" x="{x0:.1f}" y="{y + 2}" width="{cw - 6}" height="{rh - 4}" tabindex="0" data-tip="question {d}: empty in {n} of '
                     f'{len(r["passes"])} passes||{esc(r["label"])}' + (f"; {esc(ends)}" if ends else "") + '"/><g></g>')
            row.append(f"{n} ({txt})" if n else "-")
        tr.append(row)
    s.append("</svg>")
    leg = ('<div class="legend">' + "".join(f'<span><i style="background:{SHADE[n]};border-radius:2px;border:1px solid var(--grid)"></i>empty in {n} of 3 passes</span>'
                                           for n in (1, 2, 3)) + '<span>cell text: the passes that came back empty</span></div>')
    return figure("Empty answers concentrate on a few questions (descriptive)",
                  "Every question that came back empty in any pass of the three-pass records. Questions are GPQA Diamond doc ids (the questions "
                  "themselves are not published). Empty answers concentrate on a few questions; questions 79 and 81 came back empty in all three "
                  "configurations.", "".join(s), table(["Question"] + [r["label"] for r in recs], tr), leg, "empty-questions")


def short_label(cfg):
    """Two-line column header for narrow charts: weights, then engine and any concurrency."""
    c = cfg["serving"].get("concurrency")
    return [weights_label(cfg), engine_label(cfg) + (f", {c} concurrent" if c else "")]


def records_table(g):
    rows = []
    for a, b, what in analyze.RECORD_PAIRS:
        for key, nm in (("correct_flexible", "raw accuracy"), ("empty", "empty answers")):
            r = analyze.records(g[a], g[b], key)
            rows.append([esc(label(g[a]["cfg"])), esc(label(g[b]["cfg"])), esc(what), nm, f'{r["a"]} vs {r["b"]} of {r["n"]}',
                         f'{r["diff"]:+.1f} ({r["lo"]:+.1f} to {r["hi"]:+.1f})', f'{r["q_a"]} / {r["q_b"]}', f'{r["p_cluster"]:.3f}',
                         f'{r["p_pooled"]:.3f}'])
    return table(["A", "B", "What differs", "Outcome", "A vs B, question-passes", "B - A, points (95% interval over questions)",
                  "Questions where A / B had more", "Clustered p (sign-flip over questions)", "Pooled p (reference only)"], rows, numeric=(7, 8))


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


def outcome_bars(items, title, sub, fid=None, lw=None):
    """items: dicts label, rows, rid, ci (bool). Stacked bars of 12 draws (finished / loop / exhaust) with a Wilson whisker
    for the failures (loop + exhaust) when ci. One bar per arm and question; never pooled across questions."""
    lw, rh, top = lw or label_width(i["label"] for i in items), 34, 10; pw = 420; w = lw + pw + 70
    n_max = max(len(i["rows"]) for i in items)
    x = lambda v: lw + v / n_max * pw
    h = top + rh * len(items) + 50; base = h - 20
    s = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">']
    for t in range(0, n_max + 1, 3):
        s.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" y2="{base - 26}"/>'
                 f'<text x="{x(t):.1f}" y="{base - 10}" text-anchor="middle" class="muted">{t}</text>')
    s.append(axis_title(lw + pw / 2, h - 6, f"requests, of {n_max} (failures from the left)"))
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
    return figure(title, sub, "".join(s), table(["Arm", "Finished", "Loop", "Exhaustion", "Failed", "95% Wilson", "Receipts"], tr, numeric=(1, 2, 3, 4)), leg, fid)


def screen_caveats(v2):
    """What the component screen held fixed, what it did not test, and how large an effect it could see."""
    k = {a: sum(r["cls"] in ("loop", "exhaust") for r in m["rows"]) for a, m in v2.items()}
    base88 = sum(k[a] for a in analyze.ARMS88) / (12 * len(analyze.ARMS88))
    m1 = analyze.mdd(k["B"] / 12, 12, 12); m2 = analyze.mdd(base88, 24, 24); m3 = analyze.mdd(k["B79"] / 12, 12, 12)
    f = lambda x: "not reachable at all" if x is None else f"about {100 * x:.0f} points"
    rows = [["Held fixed", "In every arm: 3.25bpw weights, DFlash2 ×3, temperature 1.0 / top_p 0.95, the 327,680-token budget, 12 concurrent "
             "requests, one question per arm, a fresh server."],
            ["Not separated", "Arms EO and NO also turned draft-slot sharing off (the launcher pairs it with MLA ownership <code>tp</code>), so "
             "ownership and sharing are not separated. Arm V79 differs from B79 in image, layout and vision together."],
            ["Not varied", "Draft depth, quantization and sampling settings."],
            ["Power", f'Two-sided Fisher test, p &lt; 0.05, 80% power (<code>tools/analyze.py screen-power</code>): one arm against another on '
             f'question 88 (12 vs 12, from {100 * k["B"] / 12:.0f}% failing) detects a rise of {f(m1[1])}, and a drop is {f(m1[0])}; a switch on '
             f'vs off (24 vs 24, from {100 * base88:.0f}%) detects a rise of {f(m2[1])}; question 79 (12 vs 12, from {100 * k["B79"] / 12:.0f}%) '
             f'detects a drop of {f(m3[0])}. Only very large effects could show.'],
            ["Selection", f'Questions 88 and 79 were chosen as hard from empty answers and screens that all sent request seed 1234; with distinct '
             f'seeds, question 88 fails in {min(k[a] for a in analyze.ARMS88)}-{max(k[a] for a in analyze.ARMS88)} of 12 draws per arm. Two '
             'questions are not a benchmark-wide rate.']]
    return ('<div class="note" id="screen-scope"><b>Scope of the component screen.</b>'
            + "".join(f'<div style="margin-top:4px"><b>{h}.</b> {t}</div>' for h, t in rows) + '</div>')


def short_arm_label(arm, m):
    """Arm and configuration without the weights (every component-screen arm runs 3.25bpw; the subtitle says so)."""
    full, w = label(m["cfg"]), weights_label(m["cfg"]) + " · "
    return f'{arm} · {full[len(w):] if full.startswith(w) else full}'


def component_fig(v2):
    """Component screen as small multiples: one panel per question (counts are never pooled across questions)."""
    fail = lambda m: sum(r["cls"] in ("loop", "exhaust") for r in m["rows"])
    by = {}
    for a, m in v2.items():
        if not is_control(m): by.setdefault(m["rows"][0]["doc_id"], []).append((a, m))
    common = ("3.25bpw, 12 draws per arm with distinct request seeds, 12 concurrent, a fresh server per arm. Whisker = 95% Wilson interval of "
              "the failures. ")
    out, lw = [], label_width(short_arm_label(a, m) for arms in by.values() for a, m in arms)   # one scale for every panel
    for d, arms in sorted(by.items(), key=lambda t: -len(t[1])):
        ks = [fail(m) for _, m in arms]
        rng = f"{min(ks)}-{max(ks)}" if min(ks) != max(ks) else f"{ks[0]}"
        if len(arms) > 2:
            title = f"Question {d}: {rng} of 12 draws fail in every arm on tpurtell 0.9.1; no layout switch met the screening rule (descriptive)"
            note = tip("Note", "Arms EO and NO also turn draft-slot sharing off, so MLA ownership and sharing are not separated.") + "."
        else:
            title = f"Question {d}: {rng} of 12 draws fail on tpurtell 0.9.1 and on the tpurtell 0.7.0 image at three draft tokens (descriptive; verdict unresolved)"
            note = tip("Note", "V79 differs from B79 in image, layout and vision together, and ran three draft tokens, not the five 0.7.0 ships with.") + "."
        items = [dict(label=short_arm_label(a, m), rows=m["rows"], rid=m["id"], ci=True) for a, m in arms]
        out.append(outcome_bars(items, title, common + note, f"screen-q{d}", lw))
    return "".join(out)


def seed_fig(v2):
    if "S1234" not in v2:
        return ""
    items = [dict(label="B · seeds 5001-5012", rows=v2["B"]["rows"], rid=v2["B"]["id"], ci=True),
             dict(label="S1234 · seed 1234 on every repeat", rows=v2["S1234"]["rows"], rid=v2["S1234"]["id"], ci=False)]
    kb, ks = (sum(r["cls"] in ("loop", "exhaust") for r in v2[a]["rows"]) for a in ("B", "S1234"))
    return outcome_bars(items, f"Repeats that share one request seed are not a meaningful sample: {ks} of 12 failed with seed 1234 on every repeat, "
                        f"{kb} of 12 with distinct seeds (supported)",
                        f"{esc(label(v2['B']['cfg']))}, question 88 x 12, 12 concurrent, fresh server; only the request seeds differ. "
                        f"Failed {kb} of 12 with distinct seeds vs {ks} of 12 with seed 1234 on every repeat (Fisher p = {fisher(kb, 12 - kb, ks, 12 - ks):.3f}). "
                        "The fixed-seed arm is a control for the seed, not a measurement of the configuration.", "seed-control")


def onset_hist(v2):
    """Where the early stop fired, in estimated tokens; the 2,044-token tail-issue region marked."""
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
             f'<text x="{x(2044) + 6:.1f}" y="{top - 10}" style="fill:var(--crit)">2,044 tokens: the tail issue acted only before this</text>'
             f'<text x="{lw + pw / 2}" y="{h - 6}" text-anchor="middle" class="muted">estimated tokens generated when the early stop fired</text>')
    s.append("</svg>")
    tr = [[esc(a), f"q{d}", f"{t:,.0f}", "fixed-seed control" if c else "distinct seeds"] for t, c, a, d in sorted(pts)]
    leg = ('<div class="legend"><span><i style="background:var(--s1);border-radius:2px"></i>distinct-seed arms</span>'
           '<span><i style="background:var(--axis);border-radius:2px"></i>fixed-seed control (S1234)</span></div>')
    return figure("Every loop was stopped far beyond the 2,044-token region where the tail issue acted (descriptive)", "Every loop the early-stop detector stopped in the component screen, by estimated tokens generated. "
                  + tip("How tokens are estimated", "From reasoning characters, with each question's median characters per completion token in its "
                        f"finished distinct-seed requests ({', '.join(f'q{d}: {v:.2f}' for d, v in sorted(cpq.items()))}). The detector stops a "
                        "short-period loop within about 60,000 characters of its start, so every loop here began far beyond the first 2,044 tokens.")
                  + ".", "".join(s),
                  table(["Arm", "Question", "Estimated tokens at the stop", "Seeds"], tr, numeric=(2,)), leg, "loop-stops")


# ---------------------------------------------------------------- looping investigation page

LOOP_CFG = {"as": "glm53-flash/k3.25-v0.9.0-dflash3", "fix": "glm53-flash/k3.25-v0.9.0-tailfix-dflash3"}
INV = REPO + "/tree/main/investigations/2026-10-glm53-looping"


def fisher(a, b, c, d):
    return analyze.fisher(a, b, c, d)


def decode_prefill_fig(runs):
    """Two panels on one scale, one per position region: the run without the fix, the runs with it, then speculation on."""
    order = lambda m: (bool(m["cfg"]["serving"]["speculative"]["tokens"]), bool(m["cfg"]["engine"]["patches"]) or m["cfg"]["engine"]["version"] != "v0.9.0",
                       m["cfg"]["engine"]["version"], m["id"])
    dps = sorted((m for m in runs.values() if m["protocol"] == "decode-prefill-consistency/v1"), key=order)
    panels, ktr = [], []
    for reg, rname in (("lt2044", "i < 2,044"), ("ge2048", "i ≥ 2,048")):
        kd = []
        for m in dps:
            per = []
            for d in range(6):
                v = [r["kl_top20"] for r in m["rows"] if r["doc_id"] == d and r["region"] == reg and r["kl_top20"] is not None]
                per.append(sum(v) / len(v))
            allv = [r["kl_top20"] for r in m["rows"] if r["doc_id"] < 6 and r["region"] == reg and r["kl_top20"] is not None]
            est = sum(allv) / len(allv)
            lab = f"{engine_label(m['cfg'])}{' · spec on' if m['cfg']['serving']['speculative']['tokens'] else ''} · {m['date'][5:]}"
            kd.append(dict(label=lab, color=color(m["cfg"]), est=est, lo=min(per), hi=max(per),
                           tip=f"KL {est:.4f}||{rname} · {lab}; prompts 0-5; per-prompt range {min(per):.4f}-{max(per):.4f}"))
            ktr.append([esc(f"{rname} · {lab}"), f"{est:.4f}", f"{min(per):.4f}-{max(per):.4f}", link(m["id"])])
        head = ("Positions below 2,044, where the tail issue acted" if reg == "lt2044" else "Positions from 2,048")
        panels.append(f'<div class="s" style="margin:10px 0 0;font-weight:600;color:var(--ink2)">{head}</div>'
                      + interval_plot(kd, 0, 0.12, [0, 0.03, 0.06, 0.09, 0.12], lambda v: "0" if v == 0 else f"{v:.4f}".rstrip("0"), "",
                                      "mean KL, decode vs prefill (lower = closer agreement)"))
    return figure("Below 2,044 tokens the DCP1 tail fix brings decode much closer to prefill (supported); from 2,048 no effect is claimed",
                  "3.25bpw, prefix caching off, one request at a time, GPQA prompts 0-5 × 2,600 decoded tokens. Dot = mean KL over the shared "
                  "top-20 tokens (lower = decode agrees with prefill); line = range of the six prompts' means. "
                  + tip("Which runs", "One run without the fix; three with it: the local build and two runs of the 0.9.1 release, whose difference is "
                        "the run-to-run spread; plus 0.9.1 with speculation on. The 0.9.1 runs also cover prompts 6-11. From 2,048 tokens the runs "
                        "with the fix themselves spread widely.") + ".",
                  "".join(panels),
                  table(["Positions · engine · date", "Mean KL, prompts 0-5", "Per-prompt range", "Receipts"], ktr, numeric=(1,)),
                  legend([m["cfg"] for m in dps]), "decode-vs-prefill")


def says(items, title="What the results say"):
    """The page's answer: one line per result with its grade word, its context in a tooltip and a link to the evidence."""
    li = "".join(f'<li>{t} <span class="grade">{tip(g, gt) if gt else g}</span>' + (f' <span class="ev">{ev}</span>' if ev else "") + "</li>"
                 for t, g, gt, ev in items)
    return f'<div class="shows"><h2 id="summary">{esc(title)}</h2><ol>{li}</ol></div>'


def looping(runs):
    by = lambda proto, cfg: sorted((m for m in runs.values() if m["protocol"] == proto and m["config"] == cfg), key=lambda m: m["id"])
    v2 = v2_runs(runs)
    fig_kl = decode_prefill_fig(runs)
    ix = {c: by("kpool-tail-index/v1", c)[0] for c in (LOOP_CFG["as"], LOOP_CFG["fix"])}
    ia, ib = (sorted((r for r in ix[c]["rows"] if r["layout"] == "packed"), key=lambda r: r["length"]) for c in ix)
    dropped = sum(1 for r in ia if not r["tail_attended"]); dropped_fix = sum(1 for r in ib if not r["tail_attended"])
    itr = [[r["length"], ", ".join(map(str, r["tail"])) or "-", '<span class="bad">dropped</span>' if not r["tail_attended"] else "attended",
            '<span class="ok">attended</span>' if q["tail_attended"] else '<span class="bad">dropped</span>', "yes" if r["row_sha256_16"] == q["row_sha256_16"] else "no"]
           for r, q in zip(ia, ib) if r["length"] <= 2052]
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
    g = {m["config"].split("/", 1)[1]: m for m in runs.values() if m["protocol"] == "gpqa-diamond/v1"}
    e7 = analyze.records(g["k3.25-v0.7.0-dflash5"], g["k3.25-v0.9.1-dflash3"], "empty")
    e7b = analyze.records(g["k3.25-v0.7.0-dflash5"], g["k3.25-v0.9.1-dflash3"], "empty", (2, 3))
    a7 = analyze.records(g["k3.25-v0.7.0-dflash5"], g["k3.25-v0.9.1-dflash3"], "correct_flexible")
    q88 = [kb[a] for a in analyze.ARMS88]
    ev = lambda *ls: " · ".join(f'<a href="#{a}">{t}</a>' for a, t in ls)
    answer = table(["Finding", "Grade", "Evidence"], [
        ["The DCP1 layout of tpurtell 0.8.0 and 0.9.0 skipped the newest 1-3 tokens in decode attention at causal lengths up to 2,043 not "
         "divisible by 4. The fix (tpurtell PR #6) removes it; it shipped in 0.9.1.", tip("<b>Supported</b>", "Index check on the image's own kernels"),
         ev(("index-check", "index check"))],
        ["With the fix, decode agrees much better with prefill below 2,044 tokens.",
         tip("<b>Supported</b>", "Mean KL 0.066 without the fix (one run) vs 0.006-0.010 in three runs with it; two runs of the 0.9.1 release give the noise floor"),
         ev(("decode-vs-prefill", "chart"))],
        ["The fix's effect on answers and tool calls is within noise at the sizes run.",
         tip("<b>Descriptive</b> (effect unmeasured)", "GPQA +0.5 points on 4bpw TR3 (Brandon), one pass per release, both at 8 concurrent with a KV pool "
             "that holds about 4. Tool calling: TC-80 and TC-88 pass in both repeats with the fix, but ten other scenarios flip between repeats of one build"),
         ev(("impact", "detail"))],
        [f"On tpurtell 0.9.1, question 88 fails in {min(q88)}-{max(q88)} of 12 draws in every tested arm; question 79 in {kb['B79']} of 12, on 0.9.1 and "
         "on the 0.7.0 image at three draft tokens.", tip("<b>Descriptive</b>", "12 independent draws per arm, distinct request seeds"),
         ev(("screen-q88", "q88"), ("screen-q79", "q79"))],
        ["No tested runtime part moved either question. Only very large effects could have shown.",
         tip("<b>Descriptive</b>", "Preregistered rules: no candidate at this size / unresolved. Detectable: about 40-60 points. Draft depth, "
             "quantization and sampling were not varied"), ev(("screen-scope", "scope and power"))],
        ["Loops are stopped far beyond the first 2,044 tokens, where the tail issue acted. Question 88 loops; question 79 mostly exhausts.",
         "<b>Descriptive</b>", ev(("loop-stops", "chart"))],
        [f"Across GPQA, tpurtell 0.7.0 as shipped left fewer questions unanswered than 0.9.1: {e7['a']} vs {e7['b']} of {e7['n']} (3.25bpw, three passes each).",
         tip("<b>Supported</b>", f"Question-clustered test p = {e7['p_cluster']:.3f}. Raised by pass 1; passes 2-3 alone {e7b['a']} vs {e7b['b']} "
             f"(p = {e7b['p_cluster']:.2f})"), ev(("gpqa-records", "records"))],
        ["What drives non-completion, and which of 0.7.0's differences from 0.9.1 matters.", "<b>Open</b>", ev(("open-questions", "open questions"))]])
    scope = screen_caveats(v2)
    return ('<h1>Non-completion on hard questions</h1>'
            '<p class="lede">On some hard GPQA questions the model does not finish its reasoning within the 327,680-token budget: it loops or '
            'exhausts the budget. This page asks what drives that, and records an engine issue found along the way. '
            f'Write-up and recompute commands: <a href="{INV}">investigations/2026-10-glm53-looping</a>.</p>'
            '<p class="toc"><b>On this page:</b> <a href="#summary">answer</a> · <a href="#tail-bug">the DCP1 tail issue and its fix</a> · '
            '<a href="#non-completion">non-completion</a> · <a href="#open-questions">open questions</a></p>'
            + '<h2 id="summary">Answer</h2>' + answer
            + '<h2 id="tail-bug">The DCP1 tail issue and its fix</h2>'
              '<p><b>What happened.</b> Each decode step attends up to 2,048 earlier tokens, picked in pools of 4. The newest pool is incomplete: the '
              'current token and up to two before it (the kpool tail). In the DCP1 layout its columns were masked out at causal lengths up to 2,043 '
              'not divisible by 4, so every MLA layer missed those tokens.</p>'
              f'<p><b>Design context.</b> DCP1 with MLA layer ownership is tpurtell\'s layout from 0.8.0 on. It holds about 1.5x the KV cache of '
              f'v0.7.0\'s DCP2 layout ({kv(s1)} vs {kv(s7)} tokens on the same image). The masking path already existed in the vendored attention '
              'code; only this layout exercises it.</p>'
              '<p><b>The fix</b>, <a href="https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/6">tpurtell PR #6</a>, compacts the valid '
              'entries before the mask, in the DCP1 branch only. Merged 2026-10-08; released in tpurtell 0.9.1.</p>'
            + more("Mechanism in detail",
                   '<ul><li>The indexer writes the selected pools to columns 0-2043 and the tail to the fixed columns 2044-2046.</li>'
                   '<li>In the DCP1 branch the selection length becomes min(causal length, 2,048); every column at or beyond it is masked.</li>'
                   '<li>Not affected: DCP2, which compacts its selection, and the dense short-prefill path.</li>'
                   '<li>Measured on a local build of the fix before the merge (<i>tpurtell 0.9.0 + DCP1 tail fix ≈ 0.9.1</i>); the decode-vs-prefill '
                   'repeat ran on the 0.9.1 release.</li></ul>')
            + f'<h3 id="index-check">Index check: the tail is dropped in {dropped} of {len(ia)} packed cases without the fix, {dropped_fix} with it (supported)</h3>'
            + more("Which tokens a decode step attends, per causal length (the image's own kernels on GPU)",
                   table(["Causal length", "Tail tokens", eh(LOOP_CFG["as"]), eh(LOOP_CFG["fix"]), "Same row bytes"], itr, numeric=(0,)))
            + '<h3>Decode vs prefill</h3>' + fig_kl
            + '<h3 id="impact">Effect on answers: descriptive (effect unmeasured)</h3>'
              '<ul><li><b>GPQA:</b> 4bpw TR3 (Brandon) scored 0.5 points higher on 0.9.1 than on 0.9.0, one pass each. Passes of one configuration '
              'differ by 0.5-3.5 points.</li><li><b>Tool calling:</b> two repeats per build; ten scenarios flip between repeats of one build.</li></ul>'
            + more("Tool-calling results (tool-eval-bench, 88 scenarios, temperature 0, two repeats)",
                   '<div id="tool-calling"></div>' + table(["Configuration", "Repeat 1", "Repeat 2", "Receipts"], pts)
                   + '<p>Scenarios whose status differs anywhere. TC-80 and TC-88 fail in both repeats without the fix and pass in both with it; '
                     'the others also vary between repeats of the same build.</p>'
                   + table(["Scenario"] + [f"{eh(c)}, repeat {rp}" for c in te for rp in (1, 2)], ttr))
            + '<h2 id="non-completion">Non-completion: what the clean data shows</h2>'
              '<p>Component screen, 2026-10-09, preregistered: one question per arm, 12 draws with distinct request seeds.</p>'
            + component_fig(v2)
            + more("Scope, power and what was held fixed", scope)
            + '<h3 id="anatomy">How the failures look</h3><p>Question 88 fails by looping; question 79 mostly by exhaustion. No finished or '
              'exhausted request was stopped by the early-stop detector.</p>'
            + onset_hist(v2)
            + f'<h3 id="gpqa-records">Across GPQA: empty answers in three passes</h3><p>tpurtell 0.7.0 as shipped left {e7["a"]} of {e7["n"]} answers '
              f'empty, 0.9.1 left {e7["b"]} (3.25bpw; supported). Accuracy does not differ measurably (clustered p = {a7["p_cluster"]:.2f}). '
              'The two differ in several ways at once, so the cause is open (<a href="confounds.html#7-several-changes-between-releases-at-once">confounds</a>). '
              'Tables and the per-question grid: <a href="index.html#gpqa">Results</a>.</p>'
            + '<h2 id="seeds">Seeds and the earlier screens</h2><p>The earlier screens sent seed 1234 on every repeat. Their rates were withdrawn on '
              '2026-10-09 and replaced by the component screen above (<a href="ledger.html#withdrawn-and-what-replaced-it">ledger</a>; why: '
              '<a href="confounds.html#1-shared-request-seed-across-repeats">confounds</a>).</p>'
            + f'<h2 id="open-questions">Open questions</h2><p>Each stated with its evidence in the <a href="{INV}#open-questions">write-up</a>:</p><ul>'
              '<li>What drives non-completion on questions 88 and 79.</li>'
              f'<li>What makes tpurtell 0.7.0 as shipped leave fewer GPQA questions unanswered than 0.9.1 ({e7["a"]} vs {e7["b"]} of {e7["n"]}).</li>'
              '<li>Whether the tail fix changes answers.</li>'
              '<li>Why the engine is not bitwise reproducible even one request at a time.</li>'
              '<li>Whether these quants cost accuracy against a higher-precision reference (none was run on this hardware).</li>'
              '<li>Three code-level questions.</li></ul>')


def moved_page(name, title, where):
    """A merged page's old URL: links on, and forwards the browser (old anchors map to their new place)."""
    links = "".join(f'<li><a href="{esc(t)}">{esc(t)}</a>' + (f' (was <code>#{esc(a)}</code>)' if a else "") + "</li>" for a, t in where.items())
    js = ("const m=" + json.dumps(where) + ";const h=location.hash.slice(1);location.replace(m[h]||m['']);")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(title)} (moved) - local-inference-evals</title><style>{CSS}</style></head><body><main><h1>{esc(title)}: moved</h1>'
            f'<p class="lede">This page was merged into the site\'s main pages. Its content is now here:</p><ul>{links}</ul></main>'
            f'<script>{js}</script></body></html>')


def build():
    runs = load_runs(); OUT.mkdir(exist_ok=True)
    first = {}
    for m in sorted(runs.values(), key=lambda m: (m["date"], m["cfg"]["engine"]["series"])):
        first.setdefault(m["cfg"]["engine"]["series"], m["date"])
    SERIES.clear(); SERIES.update((ser, PALETTE[i % len(PALETTE)]) for i, ser in enumerate(sorted(first, key=lambda x: (first[x], x))))
    cfgs = list({m["config"]: m["cfg"] for m in runs.values()}.values())
    key_open, key_closed = key(cfgs, open_=True), key(cfgs)
    allg = gpqa_rows(runs); recs = record_rows(runs)
    accs = [r["acc"] for r in allg]; stated = [r["stated"] for r in allg]
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
    rg = {r["m"]["config"].split("/", 1)[1]: r for r in recs}
    spread = {c: max(analyze.pct(r["per"][p], "correct_flexible") for p in r["passes"]) - min(analyze.pct(r["per"][p], "correct_flexible") for p in r["passes"])
              for c, r in rg.items()}
    sp = sorted(spread.values())
    rec_acc = [analyze.pct(r["m"]["rows"], "correct_flexible") for r in recs]
    rec_st = [analyze.pct(r["m"]["rows"], "correct_stated") for r in recs]
    e7 = analyze.records(g["k3.25-v0.7.0-dflash5"], g["k3.25-v0.9.1-dflash3"], "empty")
    e7b = analyze.records(g["k3.25-v0.7.0-dflash5"], g["k3.25-v0.9.1-dflash3"], "empty", (2, 3))
    a7 = analyze.records(g["k3.25-v0.7.0-dflash5"], g["k3.25-v0.9.1-dflash3"], "correct_flexible")
    wk = {k: analyze.records(g["k3.25-v0.9.1-dflash3"], g["k4-v0.9.1-dflash3-c4"], k) for k in ("correct_flexible", "empty")}
    v2 = v2_runs(runs)
    gap = [r["stated"] - r["acc"] for r in allg]
    par = json.loads((ROOT / "comparisons/glm53-flash-serving-probe/parity.json").read_text())
    floor = next(pp for pp in par["pairs"] if pp["a"].split("#")[0] == pp["b"].split("#")[0])
    kvre = lambda m: re.search(r"KV pool ([\d,]+) tokens", m["notes"])
    kv091 = next(kvre(m).group(1) for m in v2.values() if m["config"] == "glm53-flash/k3.25-v0.9.1-dflash3")
    kv4 = g["k4-v0.9.1-dflash3-c4"]["summary"]["server"]["kv_pool_tokens"]

    # ---- serving data
    probes = sorted(((label(m["cfg"]), m) for m in runs.values() if m["protocol"] == "serving-probe/v1"),
                    key=lambda t: (t[1]["cfg"]["engine"]["version"], t[1]["cfg"]["serving"]["speculative"]["tokens"], len(t[1]["cfg"]["engine"]["patches"])))
    def bars(metric, batch, unit, title, sub, hi, col, axis, fid):
        lw, rh, top = label_width(lab for lab, _ in probes), 30, 8; pw = 460; w = lw + pw + 70; h = top + rh * len(probes) + 46; base = h - 20
        s = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">']
        for t in [hi * f for f in (0, .25, .5, .75, 1)]:
            xx = lw + t / hi * pw
            s.append(f'<line class="grid" x1="{xx:.1f}" x2="{xx:.1f}" y1="{top}" y2="{base - 22}"/><text x="{xx:.1f}" y="{base - 6}" text-anchor="middle" class="muted">{t:g}</text>')
        s.append(axis_title(lw + pw / 2, h - 4, axis))
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
        return figure(title, sub, "".join(s), table(["Configuration", col], tr, numeric=(1,)), legend([m["cfg"] for _, m in probes]), fid)
    pc = {m["config"]: m["summary"]["batches"] for _, m in probes}
    moves = [100 * (pc[b][bt][k] / pc[a][bt][k] - 1) for a, b in (("glm53-flash/k3.25-v0.8.0-dflash3", "glm53-flash/k3.25-v0.8.0-kpoolfix-dflash3"),
                                                                ("glm53-flash/k3.25-v0.8.0-nospec", "glm53-flash/k3.25-v0.8.0-kpoolfix-nospec"))
             for bt, k in (("sampled_c1", "median_decode_tok_s"), ("sampled_c8", "median_decode_tok_s"), ("sampled_c8", "aggregate_tok_s"))]
    acc_move = max(abs(pc["glm53-flash/k3.25-v0.8.0-kpoolfix-dflash3"][bt]["acceptance_rate"] - pc["glm53-flash/k3.25-v0.8.0-dflash3"][bt]["acceptance_rate"])
                   for bt in ("sampled_c1", "sampled_c8"))
    a8 = {c: v["sampled_c8"]["acceptance_rate"] for c, v in pc.items() if v.get("sampled_c8", {}).get("acceptance_rate") and "dflash3" in c}
    a90 = a8["glm53-flash/k3.25-v0.9.0-dflash3"]; aoth = [v for c, v in a8.items() if c != "glm53-flash/k3.25-v0.9.0-dflash3"]
    f_ = lambda v, u="": "-" if v is None else f"{v:g}{u}"
    stbl = table(["Configuration", "Decode, 1 request (tok/s)", "Decode, 8 at once (tok/s)", "Throughput, 8 at once (tok/s)", "Acceptance, 1",
                  "Acceptance, 8", "Receipts"],
                 [[esc(lab)] + [f_(m["summary"]["batches"].get(bt, {}).get(k)) for bt, k in (("sampled_c1", "median_decode_tok_s"),
                  ("sampled_c8", "median_decode_tok_s"), ("sampled_c8", "aggregate_tok_s"), ("sampled_c1", "acceptance_rate"), ("sampled_c8", "acceptance_rate"))]
                  + [link(m["id"])] for lab, m in probes], numeric=(1, 2, 3, 4, 5))
    kvt = []
    for m in sorted(runs.values(), key=lambda m: m["id"]):
        k = kvre(m)
        if k and m["protocol"] in ("hard-prompt-screen/v2", "gpqa-diamond/v1") or (k and m["id"].endswith("bisect1")):
            if any(r[0] == esc(label(m["cfg"])) for r in kvt): continue
            kvt.append([esc(label(m["cfg"])), k.group(1), link(m["id"])])

    # ---- kernel data
    km = sorted(kern.values(), key=lambda m: (m["cfg"]["engine"]["version"], bool(m["cfg"]["engine"]["patches"])))
    tests = []
    for m in km:
        for r in m["rows"]:
            if (r["suite"], r["test"]) not in tests: tests.append((r["suite"], r["test"]))
    mark = {"passed": '<span class="ok">✓ pass</span>', "failed": '<span class="bad">✗ fail</span>', "skipped": "– skipped"}
    ktr = [[("upstream: " if suite == "upstream" else "rejected-draft: ") + esc(t)]
           + [mark.get(next((r["outcome"] for r in m["rows"] if r["suite"] == suite and r["test"] == t), None), "-") for m in km] for suite, t in tests]
    per_image = table(["Engine image", "Upstream suite: passed / run", "Rejected-draft reproduction: passed / run"],
                      [[esc(engine_label(m["cfg"]))] + [("{} / {}".format(sum(r["outcome"] == "passed" for r in rs), len(rs)) if rs else "-")
                                                        for rs in ([r for r in m["rows"] if r["suite"] == su and r["outcome"] != "skipped"]
                                                                   for su in ("upstream", "rejected-draft"))] for m in km], numeric=(1, 2))

    # ---- GPQA tables
    rtr = []
    for r in recs:
        pv = [analyze.pct(r["per"][p], "correct_flexible") for p in r["passes"]]
        lo, hi = (100 * v for v in r["s"]["accuracy_flexible_ci95"])
        rtr.append([esc(r["label"]), " / ".join(f"{v:.1f}" for v in pv) + "%",
                    tip(f'{analyze.pct(r["m"]["rows"], "correct_flexible"):.1f}%', f"95% interval over questions: {lo:.1f}-{hi:.1f}%"),
                    f"{max(pv) - min(pv):.1f}", f'{analyze.pct(r["m"]["rows"], "correct_stated"):.1f}%',
                    " / ".join(str(sum(x["empty"] for x in r["per"][p])) for p in r["passes"]), r["s"]["questions_ever_empty"], link(r["rid"])])
    rec_tbl = table(["Configuration", "Raw, passes 1 / 2 / 3", "Raw, mean", "Spread, points", "Stated, mean", "Empty, passes 1 / 2 / 3",
                     "Questions ever empty", "Receipts"], rtr, numeric=(3, 6))
    p1 = table(["Configuration", "Raw", "Stated", "Empty of 198", "Concurrency, KV", "Receipts"],
               [[esc(r["label"]), tip(f'{r["acc"]:.1f}%', "95% interval over questions: {:.1f}-{:.1f}%".format(*(100 * v for v in r["s"]["accuracy_flexible_ci95"]))),
                 f'{r["stated"]:.1f}%', r["s"]["empty"], esc(r["kv"]), link(r["rid"])] for r in allg], numeric=(3,))
    fr = lambda r: ", ".join(f"{v} {k}" for k, v in r["s"]["empty_by_finish_reason"].items()) or "-"
    p1_full = table(["Configuration", "Raw (95% CI)", "Raw, answered only", "Stated", "Stated, answered only", "Empty", "Finish reasons of empty answers"],
                    [[esc(r["label"]), "{:.1f}% ({:.1f}-{:.1f})".format(r["acc"], *(100 * v for v in r["s"]["accuracy_flexible_ci95"])),
                      f'{r["acc_ans"]:.1f}%', f'{r["stated"]:.1f}%', f'{r["stated_ans"]:.1f}%', r["s"]["empty"], esc(fr(r))] for r in allg], numeric=(5,))
    ptr = [[esc(label(g[a]["cfg"])), esc(label(g[b]["cfg"])), esc(what), f'{r["only_a"]} / {r["only_b"]}', f'{r["p_acc"]:.2f}',
            f'{r["diff"]:+.1f} ({r["lo"]:+.1f} to {r["hi"]:+.1f})', f'{r["empty_only_a"]} / {r["empty_only_b"]}', f'{r["p_empty"]:.2f}']
           for a, b, what, r in pairs]
    pub = table(["Source", "Weights", "GPQA Diamond", "Stated protocol"],
                [["NVIDIA model card", "BF16", "92.17", "temp 1.0, top_p 0.95, 327,680 max new tokens; harness not stated"],
                 ["NVIDIA model card", "NVFP4", "92.11", "same"],
                 ["Red Hat model card", "NVFP4", "90.57", "lm-eval / lighteval forks, vLLM, 3 seeds averaged"],
                 ["This repository, three-pass records", "EXL3 3.25bpw / 4bpw", " / ".join(f"{v:.1f}" for v in rec_acc),
                  "mean of three passes (request seeds 1234-1236; 95% intervals {:.1f}-{:.1f}); see the GPQA protocol".format(
                      min(100 * r["s"]["accuracy_flexible_ci95"][0] for r in recs), max(100 * r["s"]["accuracy_flexible_ci95"][1] for r in recs))],
                 ["This repository, pass 1", "EXL3 3.25bpw / 4bpw", f"{min(accs):.1f}-{max(accs):.1f}",
                  "pass 1 of each configuration (95% intervals {:.1f}-{:.1f})".format(
                      min(100 * r["s"]["accuracy_flexible_ci95"][0] for r in allg), max(100 * r["s"]["accuracy_flexible_ci95"][1] for r in allg))]])

    acc3 = lambda c: "{:.1f} / {:.1f}%".format(analyze.pct(g[c]["rows"], "correct_flexible"), analyze.pct(g[c]["rows"], "correct_stated"))
    _l = [analyze.pct(g["k3.25-v0.7.0-dflash5"]["rows"], "correct_flexible") - analyze.pct(g[b]["rows"], "correct_flexible")
          for b in ("k3.25-v0.9.1-dflash3", "k4-v0.9.1-dflash3-c4")]
    lead_lo, lead_hi = min(_l), max(_l)
    # ---- Results (index)
    summary = says([
        (f'<b>3.25bpw on tpurtell 0.7.0 scored highest.</b> Three-pass means, raw / stated: '
         f'{acc3("k3.25-v0.7.0-dflash5")} (0.7.0), {acc3("k3.25-v0.9.1-dflash3")} (3.25bpw 0.9.1), {acc3("k4-v0.9.1-dflash3-c4")} (4bpw 0.9.1). '
         f'The lead is {lead_lo:.1f}-{lead_hi:.1f} points raw, consistent on both scores but within pass-to-pass noise, so not yet established. '
         'Why 0.7.0 leads is open.', "Descriptive",
         "Question-clustered tests, 0.7.0 vs 3.25bpw 0.9.1: raw p = {:.2f}, stated p = {:.2f}; vs 4bpw 0.9.1: raw p = {:.2f}, stated p = {:.2f}. "
         "Every 95% interval includes zero.".format(*[analyze.records(g["k3.25-v0.7.0-dflash5"], g[b], k)["p_cluster"]
                                                      for b in ("k3.25-v0.9.1-dflash3", "k4-v0.9.1-dflash3-c4") for k in ("correct_flexible", "correct_stated")]),
         '<a href="#records">records</a>'),
        (f'<b>One pass varies by {sp[0]:.1f}-{sp[-1]:.1f} points</b> between passes with their own seeds: as much as configurations differ.',
         "Descriptive", "Three configurations, three passes each", '<a href="#records">chart</a>'),
        (f'<b>tpurtell 0.7.0 as shipped left fewer questions unanswered than 0.9.1:</b> {e7["a"]} vs {e7["b"]} empty answers of {e7["n"]} (3.25bpw). '
         'The cause is open.', "Supported",
         f"Question-clustered test p = {e7['p_cluster']:.3f}. Raised by pass 1; passes 2-3 alone {e7b['a']} vs {e7b['b']} (p = {e7b['p_cluster']:.2f}). "
         "0.7.0 differs from 0.9.1 in draft depth, layout, kernels, vision and KV pool at once.",
         '<a href="#records-compare">comparisons</a> · <a href="confounds.html#7-several-changes-between-releases-at-once">confounds</a>'),
        (f'<b>3.25bpw and 4bpw TR3 (Brandon) perform alike on 0.9.1</b>, in accuracy and in completion. 4bpw\'s KV pool holds about 4 requests '
         'at the token cap.', "Descriptive",
         f"Raw accuracy {wk['correct_flexible']['diff']:+.1f} points ({wk['correct_flexible']['lo']:+.1f} to {wk['correct_flexible']['hi']:+.1f}); "
         f"empty answers {wk['empty']['a']} vs {wk['empty']['b']} of {wk['empty']['n']}. The 4bpw record ran 4 requests at once.", '<a href="#records">records</a> · <a href="#kv">KV</a>'),
        (f'<b>tpurtell\'s DCP1 layout with MLA layer ownership holds the most KV cache:</b> {kv091} tokens for 3.25bpw on 0.9.1.', "Supported",
         "Reported by the server at start-up, same memory setting", '<a href="#kv">KV table</a>'),
        ('<b>Neither engine fix costs speed</b>, in single runs.', "Descriptive",
         f"One run per configuration, no noise floor. Kpool fixes: {min(moves):+.0f}% to {max(moves):+.0f}%; DCP1 tail fix: 147.3 vs 146.9 tok/s at 1 request",
         '<a href="#serving">speed</a>'),
        (f'<b>Engine issues found during evaluation are fixed in current releases:</b> two upstream kpool issues (0.9.0) and the DCP1 tail '
         f'masking issue (0.9.1). The 0.9.0 and 0.9.1 images pass {up_pa[0]} of {up_pa[1]} upstream kernel tests.', "Supported",
         f"Without the kpool fixes {up_un[0]}/{up_un[1]}. Index check: the tail issue is gone with the fix; decode-vs-prefill KL below 2,044 tokens "
         f"{kl0:.3f} → {kl1:.3f}. Effect on answers: unmeasured (descriptive).", '<a href="#engine-issues">issues and fixes</a>')])
    issues = table(["Issue", "Found", "Fix", "Released in", "Evidence"], [
        ["Prefill wrote 2 KB of keys into another block's indexer region (kpool)", "upstream vLLM (vllm#57477)",
         'upstream fix, ported in <a href="https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/5">tpurtell PR #5</a>', "tpurtell 0.9.0",
         tip("<b>Supported</b>", f"Upstream regression tests: {up_un[0]}/{up_un[1]} on the 0.7.0 and 0.8.0 images, {up_pa[0]}/{up_pa[1]} with the fixes")],
        ["A rejected pool-completing draft could overwrite committed keys at 2 or more draft tokens (kpool)", "upstream vLLM (vllm#58454)",
         'upstream fix, ported in <a href="https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/5">tpurtell PR #5</a>', "tpurtell 0.9.0",
         tip("<b>Supported</b>", "The 0.9.0 and 0.9.1 release images reproduce no rejected-draft corruption at 2, 3, 5 or 7 draft tokens")],
        ["DCP1 decode skipped the newest 1-3 tokens at causal lengths up to 2,043 not divisible by 4", "during this evaluation",
         '<a href="https://github.com/tpurtell/glm-5.3-flash-ext3-2x-rtx/pull/6">tpurtell PR #6</a>', "tpurtell 0.9.1",
         tip("<b>Supported</b>", "Index check on the image's own kernels; decode agrees much better with prefill below 2,044 tokens") + ' · <a href="looping.html#tail-bug">record</a>']])
    idx = (f'<h1>GLM-5.3-Flash on 2x RTX PRO 6000: results</h1>'
           f'<p class="lede">Accuracy, completion, speed, KV capacity and kernel tests of GLM-5.3-Flash EXL3 quants on tpurtell\'s vLLM-based engine, '
           f'under fixed, versioned protocols. Every number links to its receipts. What else can move a result: '
           f'<a href="confounds.html">Confounds</a>. What changed and when, including one retraction: <a href="ledger.html">Ledger</a>.</p>'
           + summary
           + '<p class="toc"><b>On this page:</b> <a href="#gpqa">GPQA Diamond</a> · <a href="#serving">speed and acceptance</a> · '
             '<a href="#kv">KV capacity</a> · <a href="#engine-issues">engine issues and fixes</a> · <a href="#kernels">kernel tests</a>. '
             f'Every statement, graded: <a href="{REPO}/blob/main/FINDINGS.md">FINDINGS.md</a>.</p>'
           + key_closed
           + '<h2 id="gpqa">GPQA Diamond</h2><p>198 questions, temperature 1.0, a 327,680-token budget, 8 requests at once unless the label says '
             'otherwise. Raw = lm-eval flexible-extract (the headline). Stated = the audited stated answer.</p>'
           + '<h3 id="records-table">Three passes per configuration</h3>' + rec_tbl + records_summary_fig(recs)
           + more("Which questions came back empty", empty_grid_fig(recs))
           + more("Statistics: comparing the three-pass records",
                  '<div id="records-compare"></div><p>Paired by question and pass. A question answered three times counts once: the clustered p is an '
                  'exact sign-flip test over questions; the interval resamples questions. The pooled p treats passes as independent and is for '
                  f'reference only. No correction for multiple comparisons. Empty answers, 0.7.0 vs 0.9.1: passes 2 and 3 alone give {e7b["a"]} vs '
                  f'{e7b["b"]} ({e7b["q_a"]} vs {e7b["q_b"]} questions, clustered p = {e7b["p_cluster"]:.2f}).</p>' + records_table(g))
           + f'<h3 id="pass1">Pass 1 of every configuration</h3><p>Pass 1 shares one request seed across configurations, so they pair question '
             f'by question. Every pass 1 lands at {min(accs):.1f}-{max(accs):.1f}%: one pass does not separate them (descriptive).</p>' + p1
           + more("All pass-1 columns", p1_full)
           + more("Statistics: question-paired pass-1 comparisons",
                  '<div id="pass1-pairs"></div><p>"Only A right" counts questions A answered correctly in pass 1 and B did not; p is an exact McNemar '
                  'test (no correction for multiple comparisons). Each comparison resolves differences of about 5 points.</p>'
                  + table(["A", "B", "What differs", "Only A right / only B right", "p", "B - A, points (95% interval)", "Only A empty / only B empty", "p"],
                          ptr, numeric=(4, 7)))
           + more("Published scores, context only",
                  '<div id="published"></div><p>These use other weights (BF16, NVFP4) and an unstated or partly stated harness. The same NVFP4 weights '
                  'score 92.1 (NVIDIA) and 90.6 (Red Hat), so harness alone moves the score by about 1.5 points. Not plotted against the local runs.</p>' + pub)
           + more("Scoring and KV notes",
                  f'<p>The raw filter reads some correct answers as wrong; the stated-answer score is {min(gap):.1f}-{max(gap):.1f} points higher per '
                  'pass-1 run (<a href="method.html#scoring">scoring</a>). 4bpw TR3 (Brandon) leaves a KV pool of about 1.38 million tokens, about 4 '
                  'requests at the token cap: its 0.9.1 record runs 4 at once, and its pass-1 runs at 8 waited for KV at times (logged on 0.9.1; '
                  '0.8.0 and 0.9.0 kept no server log). No higher-precision version of the model was run on this hardware.</p>')
           + '<h2 id="serving">Speed and acceptance</h2><p>3.25bpw, 16 fixed GPQA prompts, 4,096 tokens, temperature 1.0. '
             + tip("One run per configuration", "No configuration was repeated, so differences of a few percent cannot be told from run-to-run variation")
             + f'. Neither fix shows a speed cost (descriptive): the kpool fixes move decode speed and throughput by {min(moves):+.0f}% to '
             f'{max(moves):+.0f}% and acceptance by at most {acc_move:.3f}.</p>' + stbl
           + bars("median_decode_tok_s", "sampled_c1", " tok/s", "Decode speed, one request at a time: neither fix shows a speed cost (single runs, descriptive)",
                  "Median per-request decode tokens per second. Each fix's with/without pair sits on adjacent rows.", 180,
                  "Decode speed, one request at a time", "median decode tokens per second per request", "decode-speed")
           + more("More charts: throughput and draft acceptance",
                  bars("aggregate_tok_s", "sampled_c8", " tok/s", "Total throughput, 8 requests at once: no consistent change with the kpool fixes (single runs)",
                       "Generated tokens per second across all requests. Each fix's with/without pair sits on adjacent rows.", 400,
                       "Total throughput, 8 requests at once", "generated tokens per second, all requests", "throughput")
                  + bars("acceptance_rate", "sampled_c8", "", f"Draft acceptance, 8 requests at once: {a90:.4f} on unpatched tpurtell 0.9.0, "
                         f"{min(aoth):.4f}-{max(aoth):.4f} on the other DFlash2 ×3 builds (single runs, descriptive)",
                         "Accepted / drafted tokens from the server's counters.", 0.6,
                         "Draft acceptance rate, 8 requests at once", "accepted / drafted tokens", "acceptance"))
           + f'<h2 id="kv">KV cache capacity</h2><p>Tokens the server reports at start-up, same memory setting. tpurtell\'s default DCP1 layout '
             f'with MLA layer ownership holds the most (supported). 4bpw TR3 (Brandon) on 0.9.1: {kv4:,} tokens, about 4 requests at the token cap.</p>'
           + table(["Configuration", "KV pool, tokens", "Receipts"], kvt, numeric=(1,))
           + '<h2 id="engine-issues">Engine issues found during evaluation, and their fixes</h2><p>All three are fixed in tpurtell 0.9.1. DCP1 '
             'with MLA layer ownership is tpurtell\'s design; it holds about 1.5x the KV cache of v0.7.0\'s DCP2 layout. Their effect on answers '
             'is unmeasured (descriptive).</p>' + issues
           + f'<h3 id="kernels">Kernel tests: {up_un[0]}/{up_un[1]} without the kpool fixes, {up_pa[0]}/{up_pa[1]} with them (supported)</h3>'
             '<p>vLLM\'s own regression tests for the kpool kernels, plus a rejected-draft reproduction, run inside each engine image.</p>' + per_image
           + more("Every test", '<div id="kernel-grid"></div>' + table(["Test"] + [engine_label(m["cfg"]) for m in km], ktr)))
    (OUT / "index.html").write_text(page("index.html", "Results", idx, ""))

    # ---- Confounds and Ledger, rendered from their markdown sources
    greedy = ('<div id="greedy"></div>' + more("Greedy agreement: characters of identical output before two runs diverge",
              table(["Comparison", "Identical outputs", "Median shared characters"],
                    [[esc(pp["meaning"]), f'{pp["identical_outputs"]}/{pp["n"]}', f'{pp["median_shared_prefix_chars"]:.0f}'] for pp in par["pairs"]],
                    numeric=(1, 2)) + '<p>Temperature 0, one request at a time. Derived from output text, which is not published; each run publishes '
              'its outputs\' hashes.</p>'))
    figs = {"seed-control": seed_fig(v2), "greedy": greedy}
    (OUT / "confounds.html").write_text(page("confounds.html", "Confounds", md_html(ROOT / "CONFOUNDS.md", figs), ""))
    (OUT / "ledger.html").write_text(page("ledger.html", "Ledger", md_html(ROOT / "LEDGER.md", figs), ""))

    # ---- investigation
    (OUT / "looping.html").write_text(page("looping.html", "Investigation", looping(runs), key_closed))

    # ---- method
    mt = ('<h1>Method</h1>'
          '<p class="lede">What the grades mean, how scores and intervals are made, and where every number comes from.</p>'
          '<h2 id="grades">Grades</h2><ul><li><b>Supported</b>: deterministic, or statistically clear.</li>'
          '<li><b>Descriptive</b>: what the data shows, without a test that separates it from chance. "Descriptive (effect unmeasured)" marks '
          'an effect no measurement here could show.</li><li><b>Open</b>: not answered by the data.</li></ul>'
          '<p>Withdrawn results are listed in the <a href="ledger.html">ledger</a> and kept in the git history.</p>'
          '<h2 id="labels">Labels</h2><p>Every configuration is named <i>weights · engine version · speculation</i>, built from its configuration '
          f'file by one rule (<a href="{REPO}/blob/main/SCHEMA.md#labels">SCHEMA.md</a>). Chart colours mark engine series.</p>'
          + key_open
          + '<h2 id="scoring">Scoring</h2><p>GPQA scores are lm-eval\'s raw <code>flexible-extract</code> filter: the last parenthesised capital '
            f'letter in the reply. It misreads some correct answers, so it runs {min(gap):.1f}-{max(gap):.1f} points low per pass-1 run. An audited '
            'stated-answer score, <code>correct_stated</code>, is published per row beside it. Raw stays the headline so runs stay comparable.</p>'
          + more("Scoring detail",
                 '<p>A reply that states its answer and then mentions other options\' labels, or uses notation such as (H) or (R) in chemistry, is '
                 'read as choosing the last one. The misread is deterministic: the same reply is always scored the same way, so repeating a run '
                 'never reveals it. The audit checks the stated final answer of every published reply, by hand wherever it disagrees with the '
                 'filter. <code>strict-match</code> records whether the reply used the phrase "The answer is", which the prompt never asks for; it '
                 'is not an accuracy measure.</p>')
          + '<h2 id="seeds">Seeds</h2><p>One request seed per screen repeat and per GPQA pass (pass <i>p</i> sends 1233 + <i>p</i>). Why: '
            '<a href="confounds.html#1-shared-request-seed-across-repeats">confounds</a>.</p>'
          + '<h2 id="intervals">Intervals and tests</h2><p>Questions are the unit. GPQA accuracy: 95% bootstrap over questions, each question '
            'resampled with all its passes. Rates (empty answers in one pass; screen failures of one question): 95% Wilson intervals. Overlapping '
            'intervals: the configurations cannot be told apart.</p>'
          + more("Tests and concurrency",
                 '<ul><li>Pass-1 comparisons pair questions (exact McNemar test).</li><li>Three-pass records: an exact sign-flip test over questions, '
                 'so a question answered three times counts once. Pooled tests are shown for reference only.</li><li>Screen rates are reported '
                 'per question and never pooled across questions.</li><li>GPQA runs at 8 concurrent requests unless the label says otherwise; '
                 'screens at 12. Match it when rerunning.</li></ul>')
          + '<h2 id="receipts">Receipts</h2><p>Every number is recomputed from published files: per-item rows (<code>results.jsonl</code>) and, '
            'for server-log figures, the numeric fields parsed from the log (<code>server_log.jsonl</code>). <code>tools/verify.py</code> and '
            '<code>tools/analyze.py</code> recompute them.</p>'
          + more("What has no row-level receipt",
                 '<p>Two kinds of figure rest on text that is not published, because it is model output or benchmark text: the greedy shared-prefix '
                 'lengths (<code>comparisons/glm53-flash-serving-probe/parity.json</code>; the outputs\' hashes are published), and the per-row '
                 '<code>correct_stated</code> judgement (the judgement is published, the reply text it was read from is not).</p>')
          + '<h2 id="comparability">Comparability</h2><p>Comparisons only line up runs with the same protocol version and hardware, and declare '
            'which configuration fields differ; <code>tools/verify.py</code> enforces this. Published numbers from other harnesses are context, never '
            'ranked.</p>'
          + f'<h2 id="sources">Sources</h2><ul><li><a href="{REPO}/tree/main/protocols">Protocols</a>: exact settings per benchmark version</li>'
            f'<li><a href="{REPO}/tree/main/configs">Configurations</a>: engine image digests, model revisions, settings</li>'
            f'<li><a href="{REPO}/blob/main/DATASHEET.md">Datasheet</a>: what the data is, and is not, suitable for</li>'
            f'<li><a href="{REPO}/blob/main/FINDINGS.md">Findings</a>: every statement, graded</li></ul>'
          + more("Glossary", glossary_html().replace('<h2 id="glossary">Glossary</h2>', '<div id="glossary"></div>')))
    (OUT / "method.html").write_text(page("method.html", "Method", mt, ""))

    # ---- merged pages keep their URLs
    for name, (title, where) in MOVED.items():
        (OUT / name).write_text(moved_page(name, title, where))
    print(f"built {len(PAGES)} pages and {len(MOVED)} forwarding pages into {OUT}")


if __name__ == "__main__":
    build()
