#!/usr/bin/env python3
"""Render one matched-accuracy campaign JSON as gaps-then-cost-tables.

Reporting convention (user, 2026-08-26): the principal display is the
excitation-gap ladder per method, followed by tables reporting costs.
Costs are the deterministic graph-span proxy triple (Paper-II analog):
N2q (CX-ladder count), D2q (routed-chain two-qubit depth), and Dc, so no
transpiler routing randomness enters. Method identity is always labeled:
"ours" versus "benchmark". Output is LaTeX compiled with pdflatex.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _sci(value: Any) -> str:
    if value is None:
        return "--"
    mantissa, exponent = f"{float(value):.0e}".split("e")
    return rf"$10^{{{int(exponent)}}}$" if mantissa in ("1", "1.0") else \
        rf"${mantissa}\times10^{{{int(exponent)}}}$"


def _escape(text: Any) -> str:
    return str(text).replace("_", r"\_")


def _ours_rung(rec: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [r for r in rec["rungs"]["ours"] if r.get("root_energies")]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda r: (r["max_root_abs_error"] is None, r["max_root_abs_error"] or 0.0),
    )


def _gap_figure(payload: dict[str, Any], out_pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    regimes = list(payload["regimes"])
    exact_max = max(
        float(r) - float(payload["regimes"][g]["reference_ground_energy"])
        for g in regimes
        for r in payload["regimes"][g]["reference_excitations"]
    )
    ylim = exact_max * 1.32

    fig, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
    width, gap = 1.0, 0.6
    xticks: list[float] = []
    xlabels: list[str] = []
    offscale: list[tuple[float, float, str]] = []

    for slot, regime in enumerate(regimes):
        rec = payload["regimes"][regime]
        e0 = float(rec["reference_ground_energy"])
        refs = [float(x) for x in rec["reference_excitations"]]
        base = slot * (width + gap)
        offs = [base + width * (i / max(len(refs) - 1, 1)) for i in range(len(refs))]
        xticks.append(base + width / 2.0)
        xlabels.append(regime.replace("_", "-"))
        ax.plot(offs, [r - e0 for r in refs], "_", color="#222222", markersize=13,
                markeredgewidth=2.0,
                label="exact (reference)" if slot == 0 else None, zorder=1)

        our = _ours_rung(rec)
        series = [
            ("ours: selected + exchange", "#2673b8", "o",
             our.get("root_energies") if our else None),
            ("benchmark: fixed class", "#c05020", "s",
             rec["rungs"]["fixed_class"][0].get("root_energies")),
        ]
        for label, color, marker, values in series:
            if not values:
                continue
            xs: list[float] = []
            ys: list[float] = []
            for x, energy in zip(offs, values):
                g = float(energy) - e0
                if g > ylim:
                    offscale.append((x, g, color))
                else:
                    xs.append(x)
                    ys.append(g)
            ax.plot(xs, ys, marker, color=color, markersize=5, markerfacecolor="none",
                    markeredgewidth=1.3, label=label if slot == 0 else None, zorder=2)

    for x, value, color in offscale:
        ax.plot([x], [ylim * 0.97], marker="^", color=color, markersize=6, zorder=3)
        ax.annotate(f"{value:.1f}", (x, ylim * 0.97), textcoords="offset points",
                    xytext=(0, -10), ha="center", fontsize=6, color=color)

    ax.set_ylim(0.0, ylim)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7, rotation=18, ha="right")
    ax.set_ylabel(r"excitation gap $\omega_\nu = E_\nu - E_0$", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if offscale:
        handles.append(Line2D([], [], marker="^", color="#c05020", linestyle="none",
                              markersize=6))
        labels.append("off scale (value shown)")
    ax.legend(handles, labels, fontsize=7, loc="upper left", framealpha=0.95, ncol=2)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=200)
    plt.close(fig)


_ARMS = (
    ("ours", "ours"),
    ("fixed_class", "fixed class"),
    ("cheapest_first", "cheapest-first"),
    ("input_order", "input-order"),
)


def _cell(entry: dict[str, Any]) -> str:
    if entry["status"] == "REACHED":
        res = (entry.get("selected_rung") or {}).get("resources") or {}
        if res:
            return f"{res['n2q']:.0f}/{res['d2q']:.0f}/{res['dc']:.0f}"
        return f"{entry['cost_at_target']:.0f}"
    terminal = entry.get("terminal") or {}
    err = terminal.get("max_root_abs_error")
    tag = "unatt." if entry["status"].startswith("UNATT") else "n.r."
    suffix = rf"\,{_sci(err)}" if err is not None else ""
    return rf"\textit{{{tag}}}{suffix}"


def _cost_tables(payload: dict[str, Any]) -> str:
    targets = sorted(
        {e for rec in payload["regimes"].values() for e in rec["cells"]},
        key=lambda x: -float(x),
    )
    out: list[str] = []
    for eps in targets:
        out.append(r"\begin{center}")
        out.append(rf"\textbf{{Accuracy target $\varepsilon_E = $ {_sci(float(eps))}}}")
        out.append(r"\\[3pt]")
        out.append(r"\begin{tabular}{l" + "r" * len(_ARMS) + "}")
        out.append(r"\toprule")
        out.append("regime & " + " & ".join(label for _key, label in _ARMS) + r" \\")
        out.append(r"\midrule")
        for regime, rec in payload["regimes"].items():
            row = rec["cells"].get(eps)
            if row is None:
                continue
            cells = [_cell(row[key]) for key, _label in _ARMS]
            out.append(_escape(regime) + " & " + " & ".join(cells) + r" \\")
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}")
        out.append(r"\end{center}")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.campaign_json).read_text(encoding="utf-8"))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _gap_figure(payload, out / "campaign_gaps.pdf")

    tex = "\n".join([
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=1.6cm]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{amsmath}",
        r"\pagestyle{empty}",
        r"\setlength{\parindent}{0pt}",
        r"\begin{document}",
        rf"\section*{{Matched-accuracy campaign: {_escape(payload.get('regime_set', '?'))}}}",
        r"{\footnotesize Protocol: \texttt{agent\_guidance/qse/paper-iii-comparison-protocol.md}.",
        r"Identical record alphabet for every arm. Table cells give the deterministic",
        r"graph-span proxy $N_{2q}/D_{2q}/D_c$ at that arm's cheapest certified rung",
        r"(no transpiler routing randomness). \textit{unatt.} = unattainable with that",
        r"manifold; \textit{n.r.} = not reached within the shared pool, terminal error shown.\par}",
        r"\vspace{6pt}",
        r"\begin{center}\includegraphics[width=0.95\textwidth]{campaign_gaps.pdf}\end{center}",
        r"{\footnotesize",
        _cost_tables(payload),
        r"}",
        r"\end{document}",
    ])
    (out / "campaign_report.tex").write_text(tex, encoding="utf-8")
    proc = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "campaign_report.tex"],
        cwd=out, capture_output=True, text=True,
    )
    errors = [line for line in proc.stdout.splitlines() if line.startswith("!")]
    if errors:
        raise SystemExit("LaTeX errors: " + "; ".join(errors[:3]))
    print(f"wrote {out / 'campaign_report.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
