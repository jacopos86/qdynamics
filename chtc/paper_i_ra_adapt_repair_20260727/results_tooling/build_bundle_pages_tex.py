#!/usr/bin/env python3
"""Bundle pages, compiled by LaTeX.

Every PDF in this repository is compiled from LaTeX (see PDF_INVARIANT.md).
This builder emits PGF figures -- so all plot text is typeset by TeX, in the
document's own fonts -- plus booktabs tables, then runs latexmk.

Data and helpers come from _bundle_pages_data.py, which is the shared layer
extracted from the previous matplotlib-only builder.

Usage: python3 build_bundle_pages_tex.py [--only 3,9]
"""
import os, re, subprocess, sys, datetime, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("bpd", f"{HERE}/_bundle_pages_data.py")
D = importlib.util.module_from_spec(spec); sys.modules["bpd"] = D; spec.loader.exec_module(D)

import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "axes.titlesize": 9,
})
import matplotlib.pyplot as plt

OUTDIR = os.path.dirname(D.OUT)
BUILD = os.path.join(HERE, "_texbuild")
os.makedirs(BUILD, exist_ok=True); os.makedirs(OUTDIR, exist_ok=True)

TEX_SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
               "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
               "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}

def esc(s):
    return "".join(TEX_SPECIAL.get(c, c) for c in str(s)).replace("|", r"$\mid$")

def panel_figure(pageno, series, cluster, tmpl, traces_out):
    """Six regime panels for one bundle page, emitted as PGF."""
    fig = plt.figure(figsize=(13.6, 4.5))
    grid = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.26)
    handles, labels = [], []
    for idx, regime in enumerate(D.REGIMES):
        ax = fig.add_subplot(grid[idx // 3, idx % 3])
        av = D.aavqe.get(regime, [])
        if av:
            ax.semilogy([k for k, _ in av], [e for _, e in av], "--", lw=1.2,
                        color="#0072b2", label="AAVQE baseline")
        for arm, label, colour, marker in series:
            tr = traces_out.get((arm, regime))
            if not tr:
                continue
            ks = sorted(tr)
            live = traces_out.get(("__live__", arm, regime), False)
            ax.semilogy(ks, [tr[k] for k in ks], ls="--" if live else "-",
                        lw=1.0, color=colour, marker=marker, markersize=2.0,
                        markevery=max(1, len(ks) // 18),
                        label=(label + (r" [in flight]" if live else "")))
        ax.set_title(esc(regime.replace("_", " ")))
        ax.set_xlim(0, 78); ax.grid(True, which="major", lw=0.3, alpha=0.4)
        if idx // 3 == 1:
            ax.set_xlabel(r"accepted controller round $k$")
        if idx % 3 == 0:
            ax.set_ylabel(r"$|E_k - E_{\mathrm{exact}}|$")
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.06), frameon=False)
    path = os.path.join(BUILD, f"panels{pageno}.pgf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


HEAD = ["series", "regime", "state", r"$k^{*}$ / $k_{\mathrm{term}}$",
        r"$\Delta E$ $k^{*}$/term", r"$N_{2q}$ $\mid$ $(\hat C_{2q})$",
        r"$D_{2q}$ $\mid$ $(\hat C_{2qd})$", r"$D_c$ $\mid$ $(\hat D_c)$",
        r"k45 $\Delta E$ $\mid$ $(\hat C_{2q})$",
        r"$\mathcal{S}_{\mathrm{alg}}$ $k^{*}$/term"]


def table_tex(rows):
    """booktabs table; AAVQE rows shaded, in-flight rows italicised."""
    # No \resizebox here: colortbl's \rowcolor does not survive being
    # scaled, so the table is sized with a font command instead.
    out = [r"\begingroup\scriptsize\setlength{\tabcolsep}{3.2pt}",
           r"\begin{tabular}{@{}l l l r r r r r r r@{}}",
           r"\toprule",
           " & ".join(rf"\textbf{{{h}}}" for h in HEAD) + r" \\",
           r"\midrule"]
    for r in rows:
        cells = [esc(c) if i < 3 else esc(c) for i, c in enumerate(r)]
        if r[0] == "AAVQE":
            body = " & ".join(rf"\textbf{{{c}}}" for c in cells)
            out.append(r"\rowcolor{aavqeblue} " + body + r" \\")
        elif str(r[2]).startswith("RUNNING"):
            out.append(r"\rowcolor{runninggreen} " + " & ".join(
                rf"\textit{{{c}}}" for c in cells) + r" \\")
        elif str(r[2]).startswith("idle"):
            out.append(r"\rowcolor{idleamber} " + " & ".join(cells) + r" \\")
        else:
            out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\endgroup"]
    return "\n".join(out)


PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[landscape,a3paper,margin=10mm]{geometry}
\usepackage{array,booktabs,graphicx,xcolor,colortbl,amsmath,pgf}
\usepackage[T1]{fontenc}
% matplotlib's PGF output emits \mathdefault; it is a no-op wrapper.
\providecommand{\mathdefault}[1]{#1}
\definecolor{aavqeblue}{HTML}{DBE7F5}
\definecolor{runninggreen}{HTML}{D8F0DD}
\definecolor{idleamber}{HTML}{FDF1D6}
\definecolor{notered}{HTML}{A01B1B}
\setlength{\parindent}{0pt}
\pagestyle{empty}
\begin{document}
"""


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    want = {s.strip() for s in args.only.split(",") if s.strip()}
    pages = [p for p in D.PAGES
             if not want or re.search(r"Bundle (\d)", p[0]).group(1) in want]
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M CDT")

    body = [PREAMBLE]
    for pageno, (title, sect, cluster, series, tmpl, note) in enumerate(pages, 1):
        traces = {}
        for regime in D.REGIMES:
            for arm, label, _c, _m in series:
                tr = D.live_trace(cluster, arm, regime)
                if tr:
                    traces[(arm, regime)] = dict(tr)
                    traces[("__live__", arm, regime)] = True
                elif tmpl:
                    tr2 = D.stream_trace(tmpl.format(arm=arm, regime=regime), regime)
                    if tr2:
                        traces[(arm, regime)] = dict(tr2)
                ent = D.KT.get(sect, {}).get(f"{arm}|{regime}") if sect else None
                if (arm, regime) not in traces and ent and ent.get("trace"):
                    traces[(arm, regime)] = {int(k): float(v) for k, v in ent["trace"]}
        fig = panel_figure(pageno, series, cluster, tmpl, traces)

        rows = []
        for regime in D.REGIMES:
            av = D.KT.get("aavqe", {}).get(regime)
            if av:
                t = lambda side, f: (av.get(f"tuple_{side}", {}) or {}).get(f)
                rows.append(["AAVQE", regime, "published (tolmatch)",
                             f"{av.get('k_star')} / {av.get('k_term')}",
                             D.fmt_pair(av.get("dE_star"), av.get("dE_term")),
                             D.fmt_dual(av, "N2q", "c2q"), D.fmt_dual(av, "D2q", "cd"),
                             D.fmt_dual(av, "Dc", "dc"), D.fmt_k45(av, regime),
                             D.fmt_pair(av.get("salg_star"), av.get("salg_term"), "{}")])
            for arm, label, _c, _m in series:
                ent = D.KT.get(sect, {}).get(f"{arm}|{regime}") if sect else None
                st = D.cell_state(cluster, arm, regime)
                tr = traces.get((arm, regime))
                if st is None:
                    st = "done (archived)" if (ent or tr) else "-"
                d = dict(tr) if tr else {}
                ks = D.KSTAR[regime]; kterm = max(d) if d else None
                rows.append([label.split(" (")[0], regime, st,
                             f"{ks} / {kterm}" if kterm else f"{ks} / -",
                             D.fmt_pair(d.get(ks), d.get(kterm) if kterm else None),
                             D.fmt_dual(ent, "N2q", "c2q"), D.fmt_dual(ent, "D2q", "cd"),
                             D.fmt_dual(ent, "Dc", "dc"), D.fmt_k45(ent, regime),
                             (D.fmt_pair(ent.get("salg_star_corrected"),
                                         ent.get("salg_term_corrected"), "{}")
                              if ent and ent.get("salg_star_corrected") is not None else
                              (D.fmt_pair(ent.get("salg_star"), ent.get("salg_term"), "{}")
                               + "  [raw]" if ent and ent.get("salg_star") is not None
                               else "- / -"))])

        body.append(rf"\textbf{{Page {pageno} --- {esc(title)}}}\\[1mm]")
        body.append(rf"{{\small refreshed {esc(stamp)}}}\\[2mm]")
        body.append(r"\resizebox{\textwidth}{!}{\input{" + fig + r"}}\\[1mm]")
        body.append(r"{\scriptsize\color{notered}" + esc(note) + r"}\\[1.5mm]")
        body.append(table_tex(rows))
        if pageno < len(pages):
            body.append(r"\clearpage")
    body.append(r"\end{document}")

    tex = os.path.join(BUILD, "bundle_pages.tex")
    open(tex, "w").write("\n".join(body))
    # This machine has a stale array.sty in ~/Library/texmf that shadows TeX
    # Live's and breaks every tabular ("Undefined control sequence
    # \vcenter@text" inside \@@array) -- a bare four-line tabular document
    # fails too, so it is not specific to this build. Disabling TEXMFHOME for
    # the compile makes it reproducible without touching the user's tree; this
    # document needs only standard TeX Live packages.
    env = dict(os.environ, TEXMFHOME="/nonexistent")
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode",
                        "-halt-on-error", f"-outdir={BUILD}", tex],
                       capture_output=True, text=True, cwd=BUILD, env=env)
    produced = os.path.join(BUILD, "bundle_pages.pdf")
    if r.returncode != 0 or not os.path.exists(produced):
        log = os.path.join(BUILD, "bundle_pages.log")
        tail = open(log).read()[-2500:] if os.path.exists(log) else r.stdout[-2500:]
        raise SystemExit(f"latexmk failed:\n{tail}")
    import shutil
    shutil.copyfile(produced, D.OUT)
    print(f"wrote {D.OUT} ({len(pages)} pages, compiled by pdflatex)")


if __name__ == "__main__":
    main()
