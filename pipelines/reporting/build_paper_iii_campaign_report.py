#!/usr/bin/env python3
"""Render one matched-accuracy campaign JSON as gaps-then-cost-tables.

Reporting convention (user, 2026-08-26): the principal display is the
excitation-gap ladder per method ("show the gaps"), followed by tables
reporting costs. Output is LaTeX compiled to PDF (no ad-hoc PDF backends)
plus the gap figure as PDF/PNG. Method identity is always labeled: "ours"
versus "benchmark".
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
    mantissa, exponent = f"{float(value):.1e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def _gap_figure(payload: dict[str, Any], out_pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regimes = list(payload["regimes"])
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    width = 1.0
    xticks, xlabels = [], []
    ours_label_done = fixed_label_done = False
    for slot, regime in enumerate(regimes):
        rec = payload["regimes"][regime]
        e0 = float(rec["reference_ground_energy"])
        refs = [float(x) for x in rec["reference_excitations"]]
        base = slot * (width + 0.45)
        offsets = [base + width * (i / max(len(refs) - 1, 1)) for i in range(len(refs))]
        xticks.append(base + width / 2.0)
        xlabels.append(regime.replace("_", "-"))
        ax.plot(offsets, [r - e0 for r in refs], "_", color="#222222",
                markersize=15, markeredgewidth=2.0,
                label="exact (reference)" if slot == 0 else None, zorder=1)
        our_rung = min(
            (r for r in rec["rungs"]["ours"] if r.get("root_energies")),
            key=lambda r: (r["max_root_abs_error"] is None, r["max_root_abs_error"] or 0),
            default=None,
        )
        if our_rung and our_rung.get("root_energies"):
            gaps = [float(e) - e0 for e in our_rung["root_energies"]]
            ax.plot(offsets[: len(gaps)], gaps, "o", color="#2673b8", markersize=5,
                    markerfacecolor="none", markeredgewidth=1.3,
                    label="ours: selected + exchange" if not ours_label_done else None, zorder=2)
            ours_label_done = True
        f = rec["rungs"]["fixed_class"][0]
        if f.get("root_energies"):
            gaps = [float(e) - e0 for e in f["root_energies"]]
            ax.plot(offsets[: len(gaps)], gaps, "s", color="#c05020", markersize=5,
                    markerfacecolor="none", markeredgewidth=1.3,
                    label="benchmark: fixed class" if not fixed_label_done else None, zorder=2)
            fixed_label_done = True
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel(r"excitation gap $\omega_\nu = E_\nu - E_0$")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8, loc="best")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=200)
    plt.close(fig)


def _cost_table(payload: dict[str, Any]) -> str:
    arms = ("ours", "fixed_class", "cheapest_first", "input_order")
    lines = [r"\begin{tabular}{ll" + "c" * len(arms) + "}", r"\hline"]
    lines.append("regime & $\\varepsilon_E$ & ours & benchmark: fixed class & "
                 r"benchmark: cheapest-first & benchmark: input-order \\ \hline")
    for regime, rec in payload["regimes"].items():
        for eps, row in sorted(rec["cells"].items(), key=lambda kv: -float(kv[0])):
            cells = []
            for arm in arms:
                c = row[arm]
                if c["status"] == "REACHED":
                    res = (c.get("selected_rung") or {}).get("resources") or {}
                    if res:
                        cells.append(
                            f"{res['n2q']:.0f} / {res['d2q']:.0f} / {res['dc']:.0f}"
                        )
                    else:
                        cells.append(f"{c['cost_at_target']:.0f}")
                else:
                    t = c.get("terminal") or {}
                    err = t.get("max_root_abs_error")
                    tag = "unatt." if c["status"].startswith("UNATT") else "n.r."
                    cells.append(rf"{tag} ({_sci(err)})" if err is not None else tag)
            lines.append(
                regime.replace("_", r"\_") + f" & {_sci(float(eps))} & " + " & ".join(cells) + r" \\"
            )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


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
        r"\usepackage[margin=2cm]{geometry}\usepackage{graphicx}\usepackage{booktabs}",
        r"\begin{document}",
        rf"\section*{{Matched-accuracy campaign: {str(payload.get('regime_set', '?')).replace(chr(95), chr(92)+chr(95))}}}",
        r"Protocol: \texttt{agent\_guidance/qse/paper-iii-comparison-protocol.md}. "
        r"Gaps first, then costs. `unatt.' = unattainable with that manifold; "
        r"`n.r.' = not reached within the shared pool (terminal error shown).",
        r"\subsection*{Excitation gaps}",
        r"\includegraphics[width=\textwidth]{campaign_gaps.pdf}",
        r"\subsection*{Proxy resources to reach each accuracy target: $N_{2q}$ / $D_{2q}$ / $D_c$}",
        r"Deterministic graph-span proxy (no routing randomness): $N_{2q}=\sum \hat c_{2q}$ (CX-ladder count), $D_{2q}=\sum \hat c_{d}$ (routed-chain two-qubit depth), $D_c=D_{2q}+\sum \hat c_{1q}$. Target reached = minimum-$N_{2q}$ certified rung.",
        r"{\small",
        _cost_table(payload),
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
        raise SystemExit(f"LaTeX errors: {errors[:3]}")
    print(f"wrote {out / 'campaign_report.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
