#!/usr/bin/env python3
"""Append a SNAKE--Append projected-singleton common-accuracy review page.

The singleton comparison uses the same reporting convention as the macro page:
the shared window ends at the earlier selected plateau, the common error is the
larger of the two within-window minima, and each method is costed at its first
crossing.  This script consumes only completed Paper-I tracker evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from pypdf import PdfReader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting.build_paper_i_hh_macro_common_accuracy_pdf import (  # noqa: E402
    _compile_comparator_at_k,
    _existing_prefix,
    _tex_sci,
    sha256,
)
from pipelines.reporting.build_paper_i_hh_tracking_plateau_costs import (  # noqa: E402
    _read_source_result,
    _snake_prefix,
)
from pipelines.reporting.paper_i_qiskit_cost_tuple import (  # noqa: E402
    PAPER_I_QISKIT_COST_TUPLE_LATEX,
)


TRACKER = REPO_ROOT / (
    "output/pdf/"
    "paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_novelty_tracking_20260715/"
    "paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_novelty_tracking_20260715.json"
)
OUTPUT_DIR = REPO_ROOT / (
    "MATH/paper_details/figures/paper_i_hh_macro_common_accuracy_20260723"
)
FINAL_STEM = "paper_i_hh_macro_common_accuracy_20260723"
FINAL_PDF = OUTPUT_DIR / f"{FINAL_STEM}.pdf"
BASE_TWO_PAGE_PDF = OUTPUT_DIR / f"{FINAL_STEM}_pages1_2.pdf"
BASE_THREE_PAGE_PDF = OUTPUT_DIR / f"{FINAL_STEM}_pages1_3.pdf"
SINGLETON_STEM = f"{FINAL_STEM}_singleton_page3"
PROVENANCE = OUTPUT_DIR / f"{FINAL_STEM}_provenance.json"

REGIMES = (
    ("weak_weak", "Weak--weak", "WW", 3),
    ("intermediate_weak", "Intermediate--weak", "IW", 3),
    ("strong_weak_u8", "Strong--weak", "SW", 3),
    ("weak_strong", "Weak--strong", "WS", 7),
    ("intermediate_strong", "Intermediate--strong", "IS", 7),
    ("strong_strong_u8", "Strong--strong", "SS", 7),
)
METHODS = (
    {
        "key": "snake_singleton",
        "route_id": "no_overlap_trust_projected_phase3_nph3_7",
        "label": "RA-ADAPT singleton",
        "short_label": "RA-ADAPT",
        "color": "#E45756",
        "marker": "*",
        "linewidth": 2.15,
    },
    {
        "key": "append_singleton",
        "route_id": "append_adapt_projected_singleton_nph3_7",
        "label": "Append-ADAPT singleton",
        "short_label": "Append-ADAPT",
        "color": "#4C78A8",
        "marker": "o",
        "linewidth": 1.45,
    },
)


def collect_rows(tracker: Mapping[str, Any]) -> list[dict[str, Any]]:
    routes = {route["id"]: route for route in tracker["routes"]}
    selected = {method["key"]: routes[method["route_id"]] for method in METHODS}
    rows: list[dict[str, Any]] = []
    for regime, title, abbreviation, n_ph in REGIMES:
        trajectories = {
            key: selected[key]["results"][regime]["trajectory"]
            for key in selected
        }
        common_window_end = min(
            int(selected[key]["plateau"][regime]["k_pl"])
            for key in selected
        )
        minima = {
            key: min(
                float(point["error"])
                for point in trajectory
                if int(point["round"]) <= common_window_end
            )
            for key, trajectory in trajectories.items()
        }
        common_error = max(minima.values())
        crossings = {
            key: next(
                int(point["round"])
                for point in trajectory
                if int(point["round"]) <= common_window_end
                and float(point["error"]) <= common_error
            )
            for key, trajectory in trajectories.items()
        }
        for method in METHODS:
            key = method["key"]
            route = selected[key]
            trajectory = trajectories[key]
            k = crossings[key]
            existing = _existing_prefix(route=route, regime=regime, k=k)
            if existing is not None:
                prefix, recovery = existing
                source_receipt = route["results"][regime]["source"]
            elif key == "append_singleton":
                prefix, source_receipt = _compile_comparator_at_k(
                    source=route["results"][regime]["source"],
                    trajectory=trajectory,
                    k=k,
                    representation="projected_singleton",
                )
                recovery = "exact bounded-memory prefix reconstruction and compile"
            else:
                source = route["results"][regime]["source"]
                payload, _runtime_seed, source_receipt = _read_source_result(
                    source,
                    need_runtime_seed=False,
                )
                prefix = _snake_prefix(
                    payload,
                    selection={
                        "history_position": k,
                        "k_pl": k,
                        "outer_iteration": int(trajectory[k - 1]["round"]),
                        "horizon": len(trajectory),
                        "error": float(trajectory[k - 1]["error"]),
                        "best_observed_error": min(
                            float(point["error"]) for point in trajectory
                        ),
                        "threshold": common_error,
                    },
                    source=source_receipt,
                    route_id=method["route_id"],
                    fallback_source_kind="paper_i_hh_snake_singleton_common_accuracy_prefix",
                )
                recovery = "exact signed-checkpoint reconstruction and compile"
            qiskit = prefix["qiskit"]
            row = {
                "regime": regime,
                "regime_title": title,
                "abbreviation": abbreviation,
                "n_ph": n_ph,
                "common_window_end": common_window_end,
                "snake_plateau_k": int(
                    selected["snake_singleton"]["plateau"][regime]["k_pl"]
                ),
                "append_plateau_k": int(
                    selected["append_singleton"]["plateau"][regime]["k_pl"]
                ),
                "method": key,
                "method_label": method["short_label"],
                "route_id": method["route_id"],
                "common_error": common_error,
                "method_minimum_error": minima[key],
                "k_cross": k,
                "crossing_error": float(trajectory[k - 1]["error"]),
                "active_depth": int(prefix["active_depth"]),
                "N2q": int(qiskit["N2q"]),
                "D2q": int(qiskit["D2q"]),
                "Dc": int(qiskit["Dc"]),
                "W1q": int(qiskit["W1q"]),
                "B1q": qiskit.get("B1q"),
                "qiskit_basis_work_status": qiskit[
                    "qiskit_basis_work_status"
                ],
                "qiskit_basis_work_schema": qiskit.get(
                    "qiskit_basis_work_schema"
                ),
                "S_alg": int(prefix["S_alg"]),
                "S_alg_scope": prefix["S_alg_scope"],
                "S_alg_components": prefix.get("S_alg_components"),
                "S_alg_receipt": prefix.get("S_alg_receipt"),
                "S_alg_reconstruction_status": prefix.get(
                    "S_alg_reconstruction_status"
                ),
                "qiskit_compile": prefix["qiskit_compile"],
                "prefix_receipt": prefix["prefix_receipt"],
                "source": source_receipt,
                "recovery": recovery,
            }
            rows.append(row)
            print(
                f"{abbreviation} {method['short_label']}: "
                f"Ecap={common_error:.8e}, k={k}, "
                f"N2q={row['N2q']}, S_alg={row['S_alg']}",
                flush=True,
            )
    return rows


def _style_axes(ax: Any, *, x_max: int) -> None:
    import numpy as np
    from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter

    ax.set_yscale("log")
    ax.set_xlim(0, x_max)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(which="major", color="#D8D8D8", linewidth=0.55)
    ax.grid(which="minor", axis="y", color="#EEEEEE", linewidth=0.35)
    ax.tick_params(axis="both", labelsize=8)


def make_plot(
    *,
    tracker: Mapping[str, Any],
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    routes = {route["id"]: route for route in tracker["routes"]}
    row_lookup = {(row["regime"], row["method"]): row for row in rows}
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.65, 4.75), dpi=300)
    for index, (regime, title, _abbreviation, n_ph) in enumerate(REGIMES):
        ax = axes.flat[index]
        _style_axes(ax, x_max=50)
        values: list[float] = []
        for method in METHODS:
            trajectory = routes[method["route_id"]]["results"][regime]["trajectory"]
            x = [int(point["round"]) for point in trajectory]
            y = [float(point["error"]) for point in trajectory]
            values.extend(y)
            ax.plot(
                x,
                y,
                color=method["color"],
                linewidth=method["linewidth"],
                solid_capstyle="round",
            )
            row = row_lookup[(regime, method["key"])]
            ax.scatter(
                [row["k_cross"]],
                [row["crossing_error"]],
                color=method["color"],
                marker=method["marker"],
                s=58 if method["marker"] == "*" else 42,
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
        common = row_lookup[(regime, "snake_singleton")]["common_error"]
        window_end = row_lookup[(regime, "snake_singleton")]["common_window_end"]
        ax.axhline(common, color="#555555", linestyle=(0, (3, 2)), linewidth=0.85)
        ax.axvspan(window_end, 50, color="#777777", alpha=0.06, linewidth=0)
        ax.axvline(
            window_end,
            color="#777777",
            linestyle=(0, (1.5, 2)),
            linewidth=0.7,
        )
        low = 10 ** np.floor(np.log10(min(value for value in values if value > 0)))
        high = 10 ** np.ceil(np.log10(max(values)))
        ax.set_ylim(low, high)
        ax.set_title(title, fontsize=9.2, pad=3)
        ax.text(
            0.02,
            0.95,
            f"({chr(ord('a') + index)})  $n_{{\\rm ph}}={n_ph}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
        )
        if index >= 3:
            ax.set_xlabel("ADAPT iteration, $k$", fontsize=8.5)
        if index % 3 == 0:
            ax.set_ylabel("Energy error, $\\Delta E$", fontsize=8.5)
    handles = [
        Line2D(
            [0],
            [0],
            color=method["color"],
            linewidth=method["linewidth"],
            marker=method["marker"],
            markersize=8 if method["marker"] == "*" else 6,
            markeredgecolor="white",
            label=method["label"],
        )
        for method in METHODS
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="#555555",
                linestyle=(0, (3, 2)),
                linewidth=0.9,
                label="$\\Delta E_\\cap$",
            ),
            Line2D(
                [0],
                [0],
                color="#777777",
                linestyle=(0, (1.5, 2)),
                linewidth=0.8,
                label="$K_\\cap$: earlier plateau",
            ),
        ]
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        fontsize=8.3,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.99,
        top=0.91,
        bottom=0.105,
        wspace=0.16,
        hspace=0.28,
    )
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def write_page_tex(
    *,
    rows: list[dict[str, Any]],
    plot_path: Path,
    tex_path: Path,
) -> None:
    lookup = {(row["regime"], row["method"]): row for row in rows}
    body: list[str] = []
    for regime, _title, abbreviation, _n_ph in REGIMES:
        for method_key in ("snake_singleton", "append_singleton"):
            row = lookup[(regime, method_key)]
            body.append(
                " & ".join(
                    [
                        abbreviation,
                        str(row["common_window_end"]),
                        _tex_sci(row["common_error"]),
                        row["method_label"],
                        str(row["k_cross"]),
                        _tex_sci(row["crossing_error"]),
                        f"{row['N2q']:,}",
                        f"{row['D2q']:,}",
                        f"{row['Dc']:,}",
                        f"{row['W1q']:,}",
                        f"{row['S_alg']:,}",
                    ]
                )
                + r" \\"
            )
        if regime != REGIMES[-1][0]:
            body.append(r"\addlinespace[1.5pt]")
    tex = rf"""
\documentclass[10pt,letterpaper]{{article}}
\usepackage[margin=0.33in]{{geometry}}
\usepackage{{amsmath,graphicx,booktabs,xcolor}}
\usepackage[T1]{{fontenc}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}
\begin{{center}}
{{\Large\bfseries RA-ADAPT--Append singleton costs at shared pre-plateau accuracy}}\par
\vspace{{2pt}}
{{\small For each regime, $K_\cap$ is the earlier selected singleton plateau and
$\Delta E_\cap=\max\{{\min_{{k\leq K_\cap}}\Delta E_{{\mathrm{{RA\text{{-}}ADAPT}}}}(k),
\min_{{k\leq K_\cap}}\Delta E_{{\rm Append}}(k)\}}$.
Each cost is evaluated at that method's first stored prefix satisfying
$\Delta E(k)\leq\Delta E_\cap$ within this shared window.}}\par
\vspace{{5pt}}
\includegraphics[width=7.75in]{{{plot_path.as_posix()}}}
\vspace{{3pt}}
\scriptsize
\setlength{{\tabcolsep}}{{3.0pt}}
\renewcommand{{\arraystretch}}{{1.05}}
\begin{{tabular}}{{@{{}}ccc l r c rrrrr@{{}}}}
\toprule
Reg. & $K_\cap$ & $\Delta E_\cap$ & Method & $k_\cap$ & $\Delta E(k_\cap)$
& $N_{{2q}}$ & $D_{{2q}}$ & $D_c$ & $W_{{1q}}$ & $S_{{\rm alg}}$ \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}
\par\vspace{{4pt}}
\begin{{minipage}}{{0.96\textwidth}}
\footnotesize
$N_{{2q}}$, $D_{{2q}}$, and $D_c$ are exact Qiskit-compiled costs for the
selected active prefix (optimization level 0, transpiler seed 7, reference
state included). $W_{{1q}}$ is the genuine Qiskit-emitted Pauli-rotation
one-qubit work before transpilation: basis changes plus the central
$R_z$ rotation, excluding reference preparation. $S_{{\rm alg}}$ is cumulative
logical estimator work, not physical shots or circuits. The canonical tuple is
${PAPER_I_QISKIT_COST_TUPLE_LATEX}$. RA-ADAPT is the active Paper-I support-projected
Phase-III configuration with no-overlap trust calibration and singleton
Pauli-child admission; Append-ADAPT uses the matched projected-singleton pool.
\end{{minipage}}
\end{{center}}
\end{{document}}
"""
    tex_path.write_text(tex.strip() + "\n", encoding="utf-8")


def compile_tex(tex_path: Path, output_dir: Path) -> Path:
    subprocess.run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-outdir={output_dir}",
            str(tex_path),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    return output_dir / f"{tex_path.stem}.pdf"


def ensure_two_page_base() -> None:
    if BASE_TWO_PAGE_PDF.is_file():
        if len(PdfReader(str(BASE_TWO_PAGE_PDF)).pages) != 2:
            raise RuntimeError("stored pages-1--2 PDF is not two pages")
        return
    if len(PdfReader(str(FINAL_PDF)).pages) != 2:
        raise RuntimeError(
            "cannot initialize pages-1--2 source from a non-two-page final PDF"
        )
    shutil.copy2(FINAL_PDF, BASE_TWO_PAGE_PDF)


def combine_with_page_three(page_three_pdf: Path) -> tuple[Path, Path]:
    combined_tex = OUTPUT_DIR / f"{FINAL_STEM}_combined.tex"
    combined_tex.write_text(
        rf"""
\documentclass[letterpaper]{{article}}
\usepackage{{pdfpages}}
\pagestyle{{empty}}
\begin{{document}}
\includepdf[pages=-,pagecommand={{}}]{{{BASE_TWO_PAGE_PDF.as_posix()}}}
\includepdf[pages=-,pagecommand={{}}]{{{page_three_pdf.as_posix()}}}
\end{{document}}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    combined_pdf = compile_tex(combined_tex, OUTPUT_DIR)
    if len(PdfReader(str(combined_pdf)).pages) != 3:
        raise RuntimeError("combined singleton review PDF is not three pages")
    shutil.copy2(combined_pdf, BASE_THREE_PAGE_PDF)
    shutil.copy2(combined_pdf, FINAL_PDF)
    return combined_tex, combined_pdf


def update_provenance(
    *,
    rows: list[dict[str, Any]],
    tracker_path: Path,
    plot_path: Path,
    page_tex: Path,
    page_pdf: Path,
    combined_tex: Path,
    combined_pdf: Path,
) -> None:
    payload = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    totals: dict[str, dict[str, int]] = {}
    for method_key in ("snake_singleton", "append_singleton"):
        method_rows = [row for row in rows if row["method"] == method_key]
        totals[method_key] = {
            field: sum(int(row[field]) for row in method_rows)
            for field in ("N2q", "D2q", "Dc", "W1q", "S_alg")
        }
    payload["singleton_common_accuracy"] = {
        "schema": "paper_i_hh_singleton_common_accuracy_comparison_v1",
        "definition": (
            "Per regime, K_cap is the earlier selected singleton plateau. "
            "DeltaE_cap is the larger within-window minimum of SNAKE and "
            "projected-singleton Append-ADAPT. Costs use each method's first crossing."
        ),
        "route_ids": [method["route_id"] for method in METHODS],
        "rows": rows,
        "summed_over_six_regimes": totals,
        "generated": {
            "plot_png": {
                "path": str(plot_path.relative_to(REPO_ROOT)),
                "sha256": sha256(plot_path),
            },
            "page_tex": {
                "path": str(page_tex.relative_to(REPO_ROOT)),
                "sha256": sha256(page_tex),
            },
            "page_pdf": {
                "path": str(page_pdf.relative_to(REPO_ROOT)),
                "sha256": sha256(page_pdf),
            },
        },
    }
    payload["generated"]["combined_tex"] = {
        "path": str(combined_tex.relative_to(REPO_ROOT)),
        "sha256": sha256(combined_tex),
    }
    payload["generated"]["combined_pdf"] = {
        "path": str(combined_pdf.relative_to(REPO_ROOT)),
        "sha256": sha256(combined_pdf),
    }
    payload["generated"]["pdf"] = {
        "path": str(FINAL_PDF.relative_to(REPO_ROOT)),
        "sha256": sha256(FINAL_PDF),
    }
    payload["validation"] = {
        "page_count": 3,
        "page_size": "letter",
        "rendered_pages_inspected": [],
        "latex_overfull_or_underfull_boxes": 0,
    }
    payload["source_tracking_json"] = {
        "path": str(tracker_path.relative_to(REPO_ROOT)),
        "sha256": hashlib.sha256(tracker_path.read_bytes()).hexdigest(),
    }
    PROVENANCE.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build(*, tracker_path: Path, output_dir: Path) -> Path:
    if output_dir != OUTPUT_DIR:
        raise ValueError("this append-only builder currently targets the canonical review PDF")
    ensure_two_page_base()
    tracker = json.loads(tracker_path.read_text(encoding="utf-8"))
    rows = collect_rows(tracker)
    plot_path = output_dir / f"{SINGLETON_STEM}_plot.png"
    page_tex = output_dir / f"{SINGLETON_STEM}.tex"
    make_plot(tracker=tracker, rows=rows, path=plot_path)
    write_page_tex(rows=rows, plot_path=plot_path, tex_path=page_tex)
    page_pdf = compile_tex(page_tex, output_dir)
    combined_tex, combined_pdf = combine_with_page_three(page_pdf)
    update_provenance(
        rows=rows,
        tracker_path=tracker_path,
        plot_path=plot_path,
        page_tex=page_tex,
        page_pdf=page_pdf,
        combined_tex=combined_tex,
        combined_pdf=combined_pdf,
    )
    return FINAL_PDF


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", type=Path, default=TRACKER)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    print(
        build(
            tracker_path=args.tracker.resolve(),
            output_dir=args.output_dir.resolve(),
        )
    )


if __name__ == "__main__":
    main()
