#!/usr/bin/env python3
"""Build a compact five-page report of all retrieved Geo-ADAPT results.

The report is deliberately separate from the Paper-I manuscript.  It consumes
the validated 40-row Geo inventory (six Hubbard--Holstein ``L=2`` rows followed
by the 34 explicitly ordered scaling rows), reconstructs the selected plateau
prefixes, compiles coefficient-aware Qiskit costs, and lays out three compact
plot-and-cost panels per row.  It never reads or writes ``Paper_I.tex`` or
``Paper_I.pdf``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting.build_paper_i_geo_scaling_evidence import (
    EXPECTED_SCALING_CASE_IDS,
    GEO_COLOR,
    PLOT_ERROR_FLOOR,
    analyze_case,
    compile_latex,
    format_error_tex,
    format_integer_tex,
    read_json,
    rel,
    sha256,
    write_csv,
    write_json,
)
from pipelines.reporting.paper_i_geo_compact_method_overlays import (
    build_overlay_rows,
    trajectory_status_is_displayable,
    write_overlay_csv,
)


SCHEMA = "paper_i_geo_retrieved_compact_v1"
STEM = "paper_i_geo_retrieved_compact_20260711"
DEFAULT_INVENTORY = REPO_ROOT / (
    "output/pdf/paper_i_geo_evidence_inventory_20260711/"
    "paper_i_geo_evidence_inventory_20260711.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / f"output/pdf/{STEM}"
DEFAULT_GROUPED_EXACT_MAX_ACTIVE_QUBITS = 8
PAGE_CASE_COUNTS: tuple[int, ...] = (6, 9, 9, 9, 7)
PAPER_I_METHOD_STYLES: dict[str, dict[str, Any]] = {
    "Append-ADAPT": {"color": "#4C78A8", "marker": "o", "width": 1.15, "size": 18},
    "Geo-ADAPT": {"color": GEO_COLOR, "marker": "^", "width": 1.25, "size": 24},
    "SNAKE": {"color": "#E45756", "marker": "*", "width": 1.55, "size": 34},
}
OVERLAY_METHOD_ORDER = ("Append-ADAPT", "Geo-ADAPT", "SNAKE")

EXPECTED_MAIN_CASE_IDS: tuple[str, ...] = (
    "hh_L2_nph2_three_model_sym_weak_weak",
    "hh_L2_nph2_three_model_sym_strong_weak",
    "hh_L2_nph2_three_model_sym_u8_strong_weak",
    "hh_L2_nph4_three_model_sym_weak_strong",
    "hh_L2_nph4_three_model_sym_strong_strong",
    "hh_L2_nph4_three_model_sym_u8_strong_strong",
)
EXPECTED_CASE_IDS: tuple[str, ...] = EXPECTED_MAIN_CASE_IDS + EXPECTED_SCALING_CASE_IDS


def retained_overlay_methods(
    *, include_append: bool, include_snake: bool
) -> tuple[str, ...]:
    """Return the explicit comparator surface in stable display order."""

    retained = {"Geo-ADAPT"}
    if include_append:
        retained.add("Append-ADAPT")
    if include_snake:
        retained.add("SNAKE")
    return tuple(method for method in OVERLAY_METHOD_ORDER if method in retained)


def filter_overlay_methods(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    include_append: bool,
    include_snake: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove unrequested method payloads and rebuild method-level provenance."""

    active_methods = retained_overlay_methods(
        include_append=include_append,
        include_snake=include_snake,
    )
    output: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    trajectory_status_counts: Counter[str] = Counter()
    cost_status_counts: Counter[str] = Counter()
    for raw_row in rows:
        row = dict(raw_row)
        raw_methods = row.get("method_overlays")
        if not isinstance(raw_methods, Mapping):
            raise ValueError(f"Overlay row has no method payloads: {row.get('case_id')}")
        methods = {
            method: raw_methods[method]
            for method in active_methods
            if isinstance(raw_methods.get(method), Mapping)
        }
        if tuple(method for method in active_methods if method in methods) != active_methods:
            raise ValueError(
                f"Overlay row is missing a retained method: {row.get('case_id')}"
            )
        row["method_overlays"] = methods
        row["overlay_methods"] = list(active_methods)
        for method, method_row in methods.items():
            status_counts[
                f"{method}:{str(method_row.get('status')).split(':', 1)[0]}"
            ] += 1
            trajectory_status_counts[
                f"{method}:{str(method_row.get('trajectory_status')).split(':', 1)[0]}"
            ] += 1
            cost_status_counts[
                f"{method}:{str(method_row.get('cost_status')).split(':', 1)[0]}"
            ] += 1
        output.append(row)

    def all_status(key: str, expected: str) -> int:
        return sum(
            all(str(row["method_overlays"][method].get(key)) == expected for method in active_methods)
            for row in output
        )

    displayable_count = sum(
        all(
            trajectory_status_is_displayable(
                row["method_overlays"][method].get("trajectory_status")
            )
            for method in active_methods
        )
        for row in output
    )
    displayable_cost_count = sum(
        all(
            trajectory_status_is_displayable(
                row["method_overlays"][method].get("trajectory_status")
            )
            and row["method_overlays"][method].get("cost_status") == "ok"
            for method in active_methods
        )
        for row in output
    )
    filtered_summary: dict[str, Any] = {
        "schema": str(summary.get("schema") or "paper_i_geo_compact_overlay_summary_v1"),
        "active_methods": list(active_methods),
        "row_count": len(output),
        "status_counts": dict(sorted(status_counts.items())),
        "trajectory_status_counts": dict(sorted(trajectory_status_counts.items())),
        "cost_status_counts": dict(sorted(cost_status_counts.items())),
        "complete_case_count": all_status("status", "ok"),
        "complete_trajectory_case_count": all_status("trajectory_status", "ok"),
        "displayable_method_case_count": int(displayable_count),
        "displayable_method_cost_complete_case_count": int(displayable_cost_count),
        "cost_scope_note": (
            "Each retained Geo/Append method uses its own k_pl structural-prefix cost. "
            "Every displayed cost uses coefficient-aware Table-I basis-gate compilation "
            "at optimization level 0 and cap 8."
            if not include_snake
            else str(summary.get("cost_scope_note") or "")
        ),
    }
    if include_append:
        filtered_summary["append_root_audit"] = summary.get("append_root_audit", [])
    if include_snake:
        for key in (
            "snake_root_audit",
            "mixed_policy_case_ids",
            "snake_history_round_audit",
            "snake_cost_semantics_counts",
            "displayable_three_method_case_count",
            "displayable_three_method_cost_complete_case_count",
        ):
            if key in summary:
                filtered_summary[key] = summary[key]
    return output, filtered_summary


def ordered_inventory_rows(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the exact requested 40 rows, rejecting reordered/expanded input."""

    raw_rows = inventory.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError("Inventory has no rows list")
    rows = [dict(row) for row in raw_rows if isinstance(row, Mapping)]
    observed = tuple(str(row.get("case_id")) for row in rows)
    if observed != EXPECTED_CASE_IDS:
        raise ValueError(
            "Inventory is not the exact ordered 40-case Geo contract; "
            f"expected={EXPECTED_CASE_IDS!r}, observed={observed!r}"
        )
    if len(set(observed)) != 40:
        raise ValueError("Geo inventory contains duplicate case ids")
    main_placements = {
        str(row.get("paper_placement")) for row in rows[: len(EXPECTED_MAIN_CASE_IDS)]
    }
    scaling_placements = {
        str(row.get("paper_placement")) for row in rows[len(EXPECTED_MAIN_CASE_IDS) :]
    }
    if main_placements != {"main_results_hubbard_holstein_L2"}:
        raise ValueError(f"Unexpected L=2 placement contract: {main_placements}")
    if scaling_placements != {"appendix_scaling_results"}:
        raise ValueError(f"Unexpected scaling placement contract: {scaling_placements}")
    return rows


def page_chunks(rows: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    if len(rows) != sum(PAGE_CASE_COUNTS):
        raise ValueError(f"Expected 40 rows, got {len(rows)}")
    chunks: list[list[Mapping[str, Any]]] = []
    start = 0
    for count in PAGE_CASE_COUNTS:
        chunks.append(list(rows[start : start + count]))
        start += count
    return chunks


def retrieval_archive_provenance(paths: Sequence[Path]) -> list[dict[str, Any]]:
    provenance: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw).resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            raise FileNotFoundError(f"Retrieval archive does not exist: {path}")
        provenance.append(
            {
                "path": rel(path),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return provenance


def compact_title_tex(row: Mapping[str, Any]) -> str:
    family = {
        "hh": "Hubbard--Holstein",
        "hubbard": "Hubbard",
        "spin_boson": "spin--boson",
        "bose_hubbard": "Bose--Hubbard",
    }.get(str(row.get("family")), str(row.get("family")))
    title = rf"{family} $L={int(row.get('L') or 0)}$"
    cutoff = row.get("cutoff_pair")
    if isinstance(cutoff, Mapping) and cutoff.get("n_ph_work") is not None:
        title += rf", $n_{{\rm ph}}^{{\max}}={int(cutoff['n_ph_work'])}$"
    regime = str(row.get("display_regime") or "").replace("-", "--")
    if regime:
        title += rf"; {regime}"
    return title


def plot_compact_case(row: Mapping[str, Any], *, plot_dir: Path) -> dict[str, Any]:
    """Render one compact axes, optionally with completed method overlays."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    plot_dir.mkdir(parents=True, exist_ok=True)
    case_id = str(row["case_id"])
    overlay_mode = bool(row.get("overlay_mode"))
    if overlay_mode:
        raw_methods = row.get("method_overlays")
        if not isinstance(raw_methods, Mapping):
            raise ValueError(f"Overlay row has no method_overlays: {case_id}")
        method_rows = [
            (method, raw_methods[method])
            for method in OVERLAY_METHOD_ORDER
            if isinstance(raw_methods.get(method), Mapping)
            and trajectory_status_is_displayable(
                raw_methods[method].get("trajectory_status")
                or raw_methods[method].get("status")
            )
        ]
        if not method_rows:
            raise ValueError(f"Overlay row has no completed display methods: {case_id}")
        suffix = "_".join(
            {
                "Append-ADAPT": "append",
                "Geo-ADAPT": "geo",
                "SNAKE": "snake",
            }[method]
            for method, _ in method_rows
        ) + "_compact"
    else:
        suffix = "geo_compact"
        method_rows = [
            (
                "Geo-ADAPT",
                {
                    "curve": list(row["trajectory_points"]),
                    "marker": dict(row["marker"]),
                },
            )
        ]
    pdf_path = plot_dir / f"{case_id}__{suffix}.pdf"
    png_path = plot_dir / f"{case_id}__{suffix}.png"
    includes_snake = any(method == "SNAKE" for method, _ in method_rows)

    with plt.rc_context(
        {
            "font.size": 6.4,
            "axes.labelsize": 6.6,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.7,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
        }
    ):
        fig, ax = plt.subplots(figsize=(2.55, 1.68), constrained_layout=True)
        legend_handles: list[Line2D] = []
        x_terminal = 0
        marker_audit: dict[str, Any] = {}
        for method, method_row in method_rows:
            style = PAPER_I_METHOD_STYLES[method]
            points = list(method_row["curve"])
            marker = method_row["marker"]
            x = [int(point["k"]) for point in points]
            y = [float(point["error_plotted"]) for point in points]
            x_terminal = max(x_terminal, max(x))
            ax.plot(
                x,
                y,
                color=str(style["color"]),
                linewidth=float(style["width"]),
                linestyle="-",
            )
            ax.scatter(
                [int(marker["k"])],
                [float(marker["error_plotted"])],
                marker=str(style["marker"]),
                s=float(style["size"]),
                color=str(style["color"]),
                edgecolors="black",
                linewidths=0.35,
                zorder=4,
            )
            label = method
            if not overlay_mode:
                label = r"Geo-ADAPT ($\triangle$: $k_{\rm pl}$)"
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=str(style["color"]),
                    linewidth=float(style["width"]),
                    linestyle="-",
                    marker=str(style["marker"]),
                    markersize=4.2 if method != "SNAKE" else 5.2,
                    markeredgecolor="black",
                    markeredgewidth=0.35,
                    label=label,
                )
            )
            marker_audit[method] = {
                "count_on_curve": 1,
                "policy": str(marker.get("policy") or "first_plateau_prefix"),
                "label": marker.get("label") or ("k_pl" if not overlay_mode else None),
                "k": int(marker["k"]),
                "error_raw": float(marker["error_raw"]),
                "error_plotted": float(marker["error_plotted"]),
            }
        ax.set_yscale("log")
        ax.set_xlim(0, int(x_terminal))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, min_n_ticks=4))
        ax.set_xlabel(
            "reported outer round" if includes_snake else "ADAPT iteration",
            labelpad=1.0,
        )
        ax.set_ylabel(r"same-cutoff $|\Delta E|$", labelpad=1.2)
        ax.grid(True, which="major", alpha=0.20, linewidth=0.4)
        ax.legend(
            handles=legend_handles,
            loc="best",
            frameon=False,
            borderaxespad=0.15,
            handlelength=1.8,
            ncol=max(1, len(method_rows)),
            columnspacing=0.55,
            handletextpad=0.25,
        )
        fig.savefig(
            pdf_path,
            metadata={
                "Title": case_id,
                "Creator": Path(__file__).name,
                "CreationDate": None,
                "ModDate": None,
            },
        )
        fig.savefig(png_path, dpi=240, metadata={"Software": Path(__file__).name})
        plt.close(fig)

    return {
        "pdf": rel(pdf_path),
        "pdf_sha256": sha256(pdf_path),
        "png": rel(png_path),
        "png_sha256": sha256(png_path),
        "layout": "compact_single_axes",
        "x_axis": (
            "integer_reported_outer_round"
            if includes_snake
            else "integer_adapt_iteration"
        ),
        "y_axis": "log_same_cutoff_abs_delta_e",
        "target_line": False,
        "curve_line_style": "solid",
        "curve_repeated_markers": False,
        "marker_count_on_curve": 1,
        "display_method_count": len(method_rows),
        "markers": marker_audit,
        "marker_shape": "method_specific" if overlay_mode else "triangle",
        "marker_k": None if overlay_mode else int(row["marker"]["k"]),
        "marker_error_raw": None if overlay_mode else float(row["marker"]["error_raw"]),
        "marker_error_plotted": None
        if overlay_mode
        else float(row["marker"]["error_plotted"]),
        "marker_policy": (
            "one_method_specific_marker_at_each_retained_method_point"
            if overlay_mode
            else "one_first_plateau_marker"
        ),
        "zero_error_plot_floor": PLOT_ERROR_FLOOR,
    }


def _panel_tex(row: Mapping[str, Any], *, output_dir: Path) -> list[str]:
    plot_path = Path(str(row["compact_plot"]["pdf"]))
    resolved_plot = plot_path if plot_path.is_absolute() else REPO_ROOT / plot_path
    try:
        plot_rel = resolved_plot.resolve().relative_to(output_dir.resolve())
    except ValueError:
        plot_rel = Path("plots") / resolved_plot.name
    lines = [
        r"\begin{minipage}[t][2.68in][t]{0.322\textwidth}",
        r"  \centering",
        rf"  {{\fontsize{{7.0}}{{7.8}}\selectfont\bfseries {row['compact_title_tex']}\par}}",
        r"  \vspace{0.4mm}",
        rf"  \includegraphics[width=0.995\linewidth]{{{plot_rel.as_posix()}}}",
        r"  \vspace{-1.0mm}",
    ]
    if bool(row.get("overlay_mode")):
        methods = row.get("method_overlays")
        if not isinstance(methods, Mapping):
            raise ValueError(f"Missing method overlays for {row['case_id']}")
        completed = [
            (method, methods[method])
            for method in OVERLAY_METHOD_ORDER
            if isinstance(methods.get(method), Mapping)
            and trajectory_status_is_displayable(
                methods[method].get("trajectory_status") or methods[method].get("status")
            )
        ]
        if not completed:
            raise ValueError(f"No completed method rows for {row['case_id']}")
        display_label = {
            "Append-ADAPT": r"Append ($k_{\rm pl}$)",
            "Geo-ADAPT": r"Geo ($k_{\rm pl}$)",
            "SNAKE": r"SNAKE ($r_{\rm hist}$)",
        }
        lines.extend(
            [
                r"  {\fontsize{5.15}{5.7}\selectfont",
                r"  \setlength{\tabcolsep}{1.6pt}",
                r"  \resizebox{0.995\linewidth}{!}{%",
                r"  \begin{tabular}{@{}lrrrrrr@{}}",
                r"    \toprule",
                r"    Method & index & $|\Delta E|$ & $N_{2q}$ & $D_{2q}$ & $D_{\rm circ}$ & $S$ \\",
                r"    \midrule",
            ]
        )
        for method, overlay in completed:
            marker = overlay["marker"]
            qiskit = overlay["qiskit_cost"]
            ledger = overlay["query_ledger"]
            method_label = display_label[method]
            if method == "SNAKE" and str(
                overlay.get("trajectory_status") or ""
            ).startswith("diagnostic:"):
                method_label = r"SNAKE diag. ($r_{\rm hist}$)"
            cost_ok = str(qiskit.get("status")) == "ok"
            cost_cells = [
                format_integer_tex(int(qiskit[key])) if cost_ok else r"--"
                for key in ("N2q", "D2q", "Dcirc")
            ]
            lines.append(
                "    "
                + method_label
                + " & "
                + str(int(marker["k"]))
                + " & $"
                + format_error_tex(float(marker["error_raw"]))
                + "$ & "
                + cost_cells[0]
                + " & "
                + cost_cells[1]
                + " & "
                + cost_cells[2]
                + " & "
                + format_integer_tex(int(ledger["S"]))
                + r" \\"
            )
        lines.extend(
            [
                r"    \bottomrule",
                r"  \end{tabular}}",
                r"  }",
            ]
        )
    else:
        qiskit = row["qiskit_prefix_cost"]
        if str(qiskit.get("status")) != "ok":
            raise ValueError(f"Qiskit cost is not complete for {row['case_id']}")
        lines.extend(
            [
                r"  {\fontsize{5.8}{6.4}\selectfont",
                r"  \setlength{\tabcolsep}{2.2pt}",
                r"  \begin{tabular}{@{}rrrrrr@{}}",
                r"    \toprule",
                r"    $k_{\rm pl}$ & $|\Delta E|$ & $N_{2q}$ & $D_{2q}$ & $D_{\rm circ}$ & $S$ \\",
                r"    \midrule",
                (
                    "    "
                    + str(int(row["marker"]["k"]))
                    + " & $"
                    + format_error_tex(float(row["marker"]["error_raw"]))
                    + "$ & "
                    + format_integer_tex(int(qiskit["N2q"]))
                    + " & "
                    + format_integer_tex(int(qiskit["D2q"]))
                    + " & "
                    + format_integer_tex(int(qiskit["Dcirc"]))
                    + " & "
                    + format_integer_tex(int(row["query_ledger"]["S"]))
                    + r" \\"
                ),
                r"    \bottomrule",
                r"  \end{tabular}",
                r"  }",
            ]
        )
    lines.append(r"\end{minipage}")
    return lines


def _panel_rows_tex(rows: Sequence[Mapping[str, Any]], *, output_dir: Path) -> list[str]:
    lines: list[str] = []
    for start in range(0, len(rows), 3):
        triplet = list(rows[start : start + 3])
        lines.append(r"\par\noindent")
        for column, row in enumerate(triplet):
            if column:
                lines.append(r"\hfill")
            lines.extend(_panel_tex(row, output_dir=output_dir))
        lines.append(r"\par")
    return lines


def write_report_tex(
    path: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    inventory_path: Path,
    inventory_sha256: str,
    provenance_json: Path,
    provenance_csv: Path,
    manifest_json: Path,
    overlay_summary: Mapping[str, Any] | None = None,
    append_roots: Sequence[Path] = (),
    snake_roots: Sequence[Path] = (),
    retrieval_archives: Sequence[Mapping[str, Any]] = (),
    condor_evidence_timestamp: str | None = None,
) -> None:
    chunks = page_chunks(rows)
    output_dir = path.parent
    source_name = inventory_path.name
    overlay_mode = any(bool(row.get("overlay_mode")) for row in rows)
    active_overlay_methods = tuple(
        method
        for method in OVERLAY_METHOD_ORDER
        if overlay_mode
        and all(
            isinstance(row.get("method_overlays"), Mapping)
            and method in row["method_overlays"]
            for row in rows
        )
    )
    includes_snake = "SNAKE" in active_overlay_methods
    if overlay_mode:
        if includes_snake:
            history_audit = dict(
                (overlay_summary or {}).get("snake_history_round_audit") or {}
            )
            audited_snake_count = int(history_audit.get("audited_row_count") or 0)
            title = "Retrieved Adaptive-Comparator Diagnostics"
            subtitle = (
                "Validated Geo/Append trajectories with retrieved SNAKE "
                "history-round diagnostics"
            )
            header = "retrieved adaptive-comparator diagnostics"
            right_header = "retrieved comparator diagnostics"
            method_text = (
                "Append-ADAPT, Geo-ADAPT, and SNAKE. SNAKE history audit: "
                f"{audited_snake_count} rows retain the requested controller horizon."
            )
            selection_text = (
                "Append and SNAKE select with replacement; Geo blocks immediate repeats. "
                "SNAKE curves use source history rounds."
            )
            marker_label = "Selected marker"
            marker_text = (
                "Geo/Append: first plateau prefix $k_{\rm pl}$; SNAKE: terminal history "
                "round $r_{\rm hist}$ (diagnostic, not an admission depth)."
            )
            complete_count = int(
                (overlay_summary or {}).get("displayable_three_method_case_count") or 0
            )
            cost_text = (
                "Geo/Append costs are at $k_{\rm pl}$; SNAKE cost is for the final native "
                "structure after the reported history horizon. All use coefficient-aware "
                "Table-I Qiskit compilation (optimization level 0; cap 8); device-mapped "
                "compile scouts are excluded; dashes mark blocked/missing common costs. "
                f"Displayable three-method panels: {complete_count}/40."
            )
        else:
            title = "Retrieved Geo/Append Comparator Results"
            subtitle = "Validated Geo-ADAPT and completed Append-ADAPT trajectories"
            header = "retrieved Geo/Append comparator results"
            right_header = "completed retrieved runs"
            method_text = "Append-ADAPT and Geo-ADAPT; full-meta macro/parent generators only."
            selection_text = (
                "Append selects with replacement; Geo selects with replacement while blocking "
                "an immediate repeat."
            )
            marker_label = "Plateau marker"
            marker_text = (
                "Each displayed method uses its own first prefix within 10\\% of its best "
                "same-cutoff error."
            )
            complete_count = int(
                (overlay_summary or {}).get("displayable_method_case_count") or 0
            )
            cost_text = (
                "Geo/Append costs are at each method's own $k_{\rm pl}$. All use "
                "coefficient-aware Table-I Qiskit compilation (optimization level 0; cap 8); "
                "dashes mark blocked/missing common costs. "
                f"Displayable two-method panels: {complete_count}/40."
            )
    else:
        title = "Geo-ADAPT Retrieved Results"
        subtitle = "Forty completed append-position Geo-ADAPT trajectories"
        header = "Geo-ADAPT retrieved results"
        right_header = "completed retrieved runs"
        method_text = "Geo-ADAPT; full-meta macro/parent generators only."
        selection_text = "With replacement except that an immediate repeat is disabled."
        marker_label = "Plateau marker"
        marker_text = (
            "First prefix within 10\\% of the best same-cutoff error over the completed trajectory."
        )
        cost_text = (
            "Coefficient-aware Qiskit synthesis, optimization level 0, transpiler seed 7, "
            "grouped-exact active-qubit cap 8; all 40 rows validated."
        )
    input_comments = [
        f"% append_source_root={rel(Path(root))}" for root in append_roots
    ] + [f"% snake_source_root={rel(Path(root))}" for root in snake_roots]
    input_comments.extend(
        f"% retrieval_archive={item.get('path')} sha256={item.get('sha256')} size_bytes={item.get('size_bytes')}"
        for item in retrieval_archives
    )
    if condor_evidence_timestamp:
        input_comments.append(f"% condor_evidence_timestamp={condor_evidence_timestamp}")
    archive_display = "; ".join(
        f"{Path(str(item.get('path'))).name} ({str(item.get('sha256'))[:12]})"
        for item in retrieval_archives
    )
    tex = [
        "% BEGIN_MACHINE_READABLE_GEO_RETRIEVED_COMPACT",
        f"% schema={SCHEMA}",
        f"% inventory_json={rel(inventory_path)}",
        f"% inventory_sha256={inventory_sha256}",
        f"% provenance_json={rel(provenance_json)}",
        f"% provenance_csv={rel(provenance_csv)}",
        f"% manifest_json={rel(manifest_json)}",
        f"% overlay_mode={'true' if overlay_mode else 'false'}",
        *input_comments,
        "% row_order=6_hh_L2_then_34_ordered_scaling_cases",
        "% page_case_counts=6,9,9,9,7",
        "% END_MACHINE_READABLE_GEO_RETRIEVED_COMPACT",
        r"\documentclass[10pt]{article}",
        r"\ifdefined\pdfinfoomitdate\pdfinfoomitdate=1\fi",
        r"\ifdefined\pdftrailerid\pdftrailerid{}\fi",
        r"\ifdefined\pdfsuppressptexinfo\pdfsuppressptexinfo=15\fi",
        r"\usepackage[letterpaper,left=0.30in,right=0.30in,top=0.28in,bottom=0.30in,headheight=9pt,headsep=3pt,footskip=12pt]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{xcolor}",
        r"\usepackage{fancyhdr}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\definecolor{geogreen}{HTML}{54A24B}",
        r"\definecolor{manifestbg}{HTML}{F3F6F3}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0pt}",
        r"\pagestyle{fancy}",
        r"\fancyhf{}",
        rf"\fancyhead[L]{{\fontsize{{6.8}}{{7.2}}\selectfont {header}}}",
        rf"\fancyhead[R]{{\fontsize{{6.8}}{{7.2}}\selectfont {right_header}}}",
        r"\fancyfoot[C]{\fontsize{7}{7}\selectfont\thepage}",
        r"\renewcommand{\headrulewidth}{0.25pt}",
        r"\renewcommand{\footrulewidth}{0pt}",
        r"\begin{document}",
        r"\thispagestyle{fancy}",
        rf"{{\centering\fontsize{{13}}{{14}}\selectfont\bfseries {title}\par}}",
        rf"{{\centering\fontsize{{7.5}}{{8.2}}\selectfont {subtitle}\par}}",
        r"\vspace{1.2mm}",
        r"\noindent\colorbox{manifestbg}{%",
        r"\begin{minipage}{0.984\textwidth}",
        r"\fontsize{6.35}{7.1}\selectfont",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.06}",
        r"\begin{tabular}{@{}p{0.12\linewidth}p{0.35\linewidth}p{0.12\linewidth}p{0.35\linewidth}@{}}",
        (
            r"\textbf{Scope} & 40 Geo anchor rows: six Hubbard--Holstein $L=2$ cases, then 34 explicitly ordered scaling cases; fetched comparators shown where available. & \textbf{Method} & "
            if overlay_mode
            else r"\textbf{Scope} & 40 completed rows: six Hubbard--Holstein $L=2$ cases, then 34 explicitly ordered scaling cases (no Cartesian expansion). & \textbf{Method} & "
        )
        + method_text
        + r" \\",
        r"\textbf{Selection} & "
        + selection_text
        + r" & \textbf{Optimization} & Powell, maximum 200 inner iterations; fixed source horizon (20, 30, or 50). \\",
        rf"\textbf{{{marker_label}}} & {marker_text} & \textbf{{Error}} & Same-cutoff $|E_{{\rm alg}}-E_{{\rm ED}}|$; logarithmic plot. \\",
        r"\textbf{Compiled cost} & "
        + cost_text
        + r" & \textbf{Estimator work} & $S=N_{H,{\rm outer}}+N_{H,{\rm refit}}+N_{\rm grad}+N_{\rm metric}+N_{\rm other}$. \\",
        r"\textbf{State/drive} & Source reference state per retrieved case; no time-dependent drive. & \textbf{Cutoffs} & Working phonon cutoff shown in each phonon-bearing panel; same-cutoff ED reference. \\",
        rf"\textbf{{Source}} & \multicolumn{{3}}{{p{{0.82\linewidth}}}}{{\texttt{{\detokenize{{{source_name}}}}}; SHA-256 \texttt{{{inventory_sha256}}}.}} \\",
        (
            r"\textbf{Retrieval} & \multicolumn{3}{p{0.82\linewidth}}{"
            + (archive_display.replace("_", r"\_") if archive_display else "No archive supplied")
            + (
                rf"; Condor evidence {condor_evidence_timestamp}."
                if condor_evidence_timestamp
                else "."
            )
            + r"} \\"
            if overlay_mode
            else r""
        ),
        r"\end{tabular}",
        r"\end{minipage}}",
        r"\vspace{1.0mm}",
    ]
    tex.extend(_panel_rows_tex(chunks[0], output_dir=output_dir))
    for chunk in chunks[1:]:
        tex.append(r"\clearpage")
        tex.extend(_panel_rows_tex(chunk, output_dir=output_dir))
    tex.append(r"\end{document}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(tex) + "\n", encoding="utf-8")


def _pdf_page_count(path: Path) -> int:
    from pypdf import PdfReader

    return len(PdfReader(str(path)).pages)


def build_report(
    *,
    inventory_path: Path = DEFAULT_INVENTORY,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_stem: str = STEM,
    compile_qiskit: bool = True,
    grouped_exact_max_active_qubits: int = DEFAULT_GROUPED_EXACT_MAX_ACTIVE_QUBITS,
    append_source_roots: Sequence[Path] = (),
    snake_source_roots: Sequence[Path] = (),
    require_complete_overlays: bool = False,
    retrieval_archives: Sequence[Path] = (),
    condor_evidence_timestamp: str | None = None,
) -> dict[str, Any]:
    inventory_path = inventory_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = read_json(inventory_path)
    inventory_rows = ordered_inventory_rows(inventory)
    plot_dir = output_dir / "plots"
    prefix_dir = output_dir / "prefixes"

    rows: list[dict[str, Any]] = []
    for index, inventory_row in enumerate(inventory_rows):
        row = analyze_case(
            inventory_row,
            order_index=index,
            prefix_dir=prefix_dir,
            compile_qiskit=compile_qiskit,
            grouped_exact_max_active_qubits=grouped_exact_max_active_qubits,
        )
        row["compact_title_tex"] = compact_title_tex(row)
        rows.append(row)

    qiskit_counts = Counter(str(row["qiskit_prefix_cost"]["status"]) for row in rows)
    if qiskit_counts != Counter({"ok": 40}):
        raise ValueError(f"All 40 Qiskit prefix compiles must succeed: {qiskit_counts}")
    if not all(row["prefix_reconstruction"]["status"] == "pass" for row in rows):
        raise ValueError("At least one structural-prefix reconstruction failed")
    if not all(
        row["query_ledger"]["terminal_identity"]["status"] == "pass" for row in rows
    ):
        raise ValueError("At least one terminal query-accounting identity failed")

    overlay_mode = bool(append_source_roots or snake_source_roots)
    archive_provenance = retrieval_archive_provenance(retrieval_archives)
    overlay_summary: dict[str, Any] | None = None
    if overlay_mode:
        rows, overlay_summary = build_overlay_rows(
            geo_rows=rows,
            inventory_rows=inventory_rows,
            append_roots=append_source_roots,
            snake_roots=snake_source_roots,
            grouped_exact_max_active_qubits=grouped_exact_max_active_qubits,
        )
        rows, overlay_summary = filter_overlay_methods(
            rows,
            overlay_summary,
            include_append=bool(append_source_roots),
            include_snake=bool(snake_source_roots),
        )
        if require_complete_overlays and int(overlay_summary["complete_case_count"]) != 40:
            raise ValueError(
                "Complete requested-method overlays required, but only "
                f"{overlay_summary['complete_case_count']}/40 cases are complete; "
                f"status_counts={overlay_summary['status_counts']}"
            )

    for row in rows:
        row["compact_plot"] = plot_compact_case(row, plot_dir=plot_dir)
        # Retain the shared Geo evidence plot key while using compact assets.
        row["plot"] = dict(row["compact_plot"])

    provenance_json = output_dir / f"{output_stem}_provenance.json"
    provenance_csv = output_dir / f"{output_stem}_provenance.csv"
    manifest_json = output_dir / f"{output_stem}_manifest.json"
    report_tex = output_dir / f"{output_stem}.tex"
    report_pdf = output_dir / f"{output_stem}.pdf"
    if overlay_mode:
        write_overlay_csv(provenance_csv, rows)
    else:
        write_csv(provenance_csv, rows)
    write_report_tex(
        report_tex,
        rows=rows,
        inventory_path=inventory_path,
        inventory_sha256=sha256(inventory_path),
        provenance_json=provenance_json,
        provenance_csv=provenance_csv,
        manifest_json=manifest_json,
        overlay_summary=overlay_summary,
        append_roots=append_source_roots,
        snake_roots=snake_source_roots,
        retrieval_archives=archive_provenance,
        condor_evidence_timestamp=condor_evidence_timestamp,
    )
    latex = compile_latex(report_tex)
    page_count = _pdf_page_count(report_pdf)
    if page_count != 5:
        raise ValueError(f"Compact report must be exactly 5 pages, got {page_count}")

    generated_utc = datetime.now(timezone.utc).isoformat()
    strict_counts = Counter(str(row["strict_replay"]["status"]) for row in rows)
    active_overlay_methods = (
        retained_overlay_methods(
            include_append=bool(append_source_roots),
            include_snake=bool(snake_source_roots),
        )
        if overlay_mode
        else ()
    )
    summary = {
        "row_count": len(rows),
        "page_count": page_count,
        "page_case_counts": list(PAGE_CASE_COUNTS),
        "family_counts": dict(sorted(Counter(str(row["family"]) for row in rows).items())),
        "qiskit_status_counts": dict(sorted(qiskit_counts.items())),
        "strict_replay_status_counts": dict(sorted(strict_counts.items())),
        "all_prefix_reconstructions_pass": True,
        "all_terminal_query_identities_pass": True,
        "all_plot_marker_counts_one": all(
            (
                int(row["compact_plot"]["display_method_count"])
                if overlay_mode
                else 1
            )
            == sum(
                int(marker["count_on_curve"])
                for marker in row["compact_plot"]["markers"].values()
            )
            for row in rows
        ),
        "overlay_mode": overlay_mode,
        "overlay_summary": overlay_summary,
    }
    provenance = {
        "schema": SCHEMA,
        "generated_utc": generated_utc,
        "scope": (
            (
                "Exactly 40 ordered Geo-ADAPT anchor rows with explicitly supplied Append-ADAPT "
                "overlays and provenance-classified SNAKE history-round diagnostics where available"
                if "SNAKE" in active_overlay_methods
                else "Exactly 40 ordered Geo-ADAPT anchor rows with explicitly supplied "
                "Append-ADAPT overlays where available"
            )
            if overlay_mode
            else "Exactly 40 completed retrieved Geo-ADAPT rows: six Hubbard-Holstein L=2 "
            "cases followed by the 34 explicitly ordered scaling cases"
        ),
        "case_order_contract": list(EXPECTED_CASE_IDS),
        "page_case_counts": list(PAGE_CASE_COUNTS),
        "method_contract": (
            {
                "methods": {
                    method: {
                        "Append-ADAPT": "static_full_meta_append_adapt_vqe",
                        "Geo-ADAPT": "static_geo_adapt_vqe",
                        "SNAKE": "static_family_native_adapt_phase3",
                    }[method]
                    for method in active_overlay_methods
                },
                "optimizer": "powell",
                "optimizer_maxiter": 200,
                "cost_point_policy": {
                    method: (
                        "native_terminal_structure_after_reported_history_horizon"
                        if method == "SNAKE"
                        else "own_first_plateau_prefix_k_pl"
                    )
                    for method in active_overlay_methods
                },
                "pool_policy": (
                    "full_meta_parent_generators_only_except_explicit_l2_snake_child_set_diagnostic"
                    if "SNAKE" in active_overlay_methods
                    else "full_meta_parent_generators_only"
                ),
                "grouped_exact_max_active_qubits": int(grouped_exact_max_active_qubits),
            }
            if overlay_mode
            else {
                "algorithm_id": "static_geo_adapt_vqe",
                "method": "Geo-ADAPT",
                "optimizer": "powell",
                "optimizer_maxiter": 200,
                "selection_with_replacement": True,
                "immediate_repeat_allowed": False,
                "immediate_repeat_policy": "with_replacement_except_immediate_repeat",
                "pool_policy": "full_meta_parent_generators_only",
                "grouped_exact_max_active_qubits": int(grouped_exact_max_active_qubits),
            }
        ),
        "plateau_policy": {
            "id": "first_prefix_within_10_percent_of_best_observed_error_v1",
            "relative_tolerance": 0.10,
            "selection_domain": "each_completed_source_horizon",
            "applies_to": (
                [method for method in active_overlay_methods if method != "SNAKE"]
                if overlay_mode
                else ["Geo-ADAPT"]
            ),
        },
        "plot_policy": {
            "layout": "three_plot_and_cost_panels_per_row",
            "method": list(active_overlay_methods) if overlay_mode else "Geo-ADAPT",
            "colors": (
                {
                    method: PAPER_I_METHOD_STYLES[method]["color"]
                    for method in active_overlay_methods
                }
                if overlay_mode
                else {"Geo-ADAPT": GEO_COLOR}
            ),
            "line_style": "solid",
            "curve_repeated_markers": False,
            "marker_policy": (
                (
                    "Geo_and_Append_at_k_pl;SNAKE_at_terminal_history_round_r_hist"
                    if "SNAKE" in active_overlay_methods
                    else "Geo_and_Append_at_own_k_pl"
                )
                if overlay_mode
                else "one_triangle_at_first_plateau_prefix"
            ),
            "integer_x_ticks": True,
            "log_y": True,
            "target_line": False,
            "zero_error_plot_floor": PLOT_ERROR_FLOOR,
        },
        "source": {
            "inventory_json": rel(inventory_path),
            "inventory_sha256": sha256(inventory_path),
            "inventory_generated_utc": inventory.get("generated_utc"),
            **(
                {"append_source_roots": [rel(Path(root)) for root in append_source_roots]}
                if append_source_roots
                else {}
            ),
            **(
                {"snake_source_roots": [rel(Path(root)) for root in snake_source_roots]}
                if snake_source_roots
                else {}
            ),
            "retrieval_archives": archive_provenance,
            "condor_evidence_timestamp": condor_evidence_timestamp,
        },
        "overlay_contract": (
            {
                "summary": overlay_summary,
                "missing_rows_are_never_invented_or_proxied": True,
                "require_complete_overlays": bool(require_complete_overlays),
                **(
                    {
                        "l2_snake_policy_note": (
                            "Retrieved L=2 SNAKE current-forward rows use archival child-set "
                            "selection; those panels are explicitly mixed-policy diagnostics "
                            "relative to parent-only Geo/Append."
                        ),
                        "snake_prefix_reconstruction_policy": (
                            "Geo and Append retain their own validated k_pl prefix semantics. "
                            "No arbitrary SNAKE prefix is reconstructed; a SNAKE terminal-structure "
                            "cost row requires a hash-linked native terminal structure and "
                            "coefficient-aware common Table-I compilation; FakeMarrakesh "
                            "optimization-level-1 compile scouts are excluded from the table."
                        ),
                        "snake_history_round_semantics": (
                            "SNAKE x coordinates are source outer-history rounds. They are not "
                            "committed-admission counts; zero-gain duplicate structural rollbacks "
                            "are counted and classified in every retrieved SNAKE row."
                        ),
                    }
                    if "SNAKE" in active_overlay_methods
                    else {
                        "prefix_reconstruction_policy": (
                            "Geo and Append each retain their own validated k_pl structural-prefix "
                            "semantics and coefficient-aware common Table-I compilation."
                        )
                    }
                ),
            }
            if overlay_mode
            else None
        ),
        "summary": summary,
        "artifacts": {
            "report_pdf": rel(report_pdf),
            "report_pdf_sha256": sha256(report_pdf),
            "report_tex": rel(report_tex),
            "report_tex_sha256": sha256(report_tex),
            "provenance_csv": rel(provenance_csv),
            "provenance_csv_sha256": sha256(provenance_csv),
            "manifest_json": rel(manifest_json),
            "plot_directory": rel(plot_dir),
            "prefix_directory": rel(prefix_dir),
            "latex_build": latex,
        },
        "rows": rows,
    }
    write_json(provenance_json, provenance)
    manifest = {
        "schema": f"{SCHEMA}_manifest_v1",
        "generated_utc": generated_utc,
        "parameter_manifest": provenance["method_contract"],
        "case_order_contract": list(EXPECTED_CASE_IDS),
        "source": provenance["source"],
        "summary": summary,
        "artifacts": {
            **provenance["artifacts"],
            "provenance_json": rel(provenance_json),
            "provenance_json_sha256": sha256(provenance_json),
        },
    }
    write_json(manifest_json, manifest)
    return manifest


def audit_overlay_sources(
    *,
    inventory_path: Path,
    append_source_roots: Sequence[Path],
    snake_source_roots: Sequence[Path],
    grouped_exact_max_active_qubits: int,
) -> dict[str, Any]:
    """Validate explicit overlay roots without writing or compiling a PDF."""

    inventory_path = inventory_path.resolve()
    inventory = read_json(inventory_path)
    inventory_rows = ordered_inventory_rows(inventory)
    with tempfile.TemporaryDirectory(prefix="paper_i_geo_overlay_audit_") as tmp:
        prefix_dir = Path(tmp) / "prefixes"
        geo_rows = [
            analyze_case(
                inventory_row,
                order_index=index,
                prefix_dir=prefix_dir,
                compile_qiskit=True,
                grouped_exact_max_active_qubits=grouped_exact_max_active_qubits,
            )
            for index, inventory_row in enumerate(inventory_rows)
        ]
        rows, summary = build_overlay_rows(
            geo_rows=geo_rows,
            inventory_rows=inventory_rows,
            append_roots=append_source_roots,
            snake_roots=snake_source_roots,
            grouped_exact_max_active_qubits=grouped_exact_max_active_qubits,
        )
        rows, summary = filter_overlay_methods(
            rows,
            summary,
            include_append=bool(append_source_roots),
            include_snake=bool(snake_source_roots),
        )
    summary = dict(summary)
    summary["inventory_json"] = rel(inventory_path)
    summary["inventory_sha256"] = sha256(inventory_path)
    summary["report_built"] = False
    summary["rows"] = [
        {
            "case_id": row["case_id"],
            "methods": {
                method: row["method_overlays"][method]["status"]
                for method in retained_overlay_methods(
                    include_append=bool(append_source_roots),
                    include_snake=bool(snake_source_roots),
                )
            },
        }
        for row in rows
    ]
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default=STEM)
    parser.add_argument("--skip-qiskit", action="store_true")
    parser.add_argument(
        "--append-source-root",
        type=Path,
        action="append",
        default=[],
        help="Explicit fetched root to search for Append-ADAPT results; repeatable.",
    )
    parser.add_argument(
        "--snake-source-root",
        type=Path,
        action="append",
        default=[],
        help="Explicit fetched root to search for SNAKE results; repeatable.",
    )
    parser.add_argument(
        "--require-complete-overlays",
        action="store_true",
        help="Fail before plotting unless all 40 rows have every explicitly requested method.",
    )
    parser.add_argument(
        "--retrieval-archive",
        type=Path,
        action="append",
        default=[],
        help="Fetched archive to hash into overlay provenance; repeatable.",
    )
    parser.add_argument(
        "--condor-evidence-timestamp",
        default=None,
        help="Timestamp of the queue/history evidence used to classify fetched rows.",
    )
    parser.add_argument(
        "--audit-overlays-only",
        action="store_true",
        help="Validate explicit overlay roots and print statuses without building a PDF.",
    )
    parser.add_argument(
        "--grouped-exact-max-active-qubits",
        type=int,
        default=DEFAULT_GROUPED_EXACT_MAX_ACTIVE_QUBITS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.audit_overlays_only:
        if not args.append_source_root and not args.snake_source_root:
            raise ValueError("--audit-overlays-only requires at least one explicit source root")
        audit = audit_overlay_sources(
            inventory_path=args.inventory,
            append_source_roots=args.append_source_root,
            snake_source_roots=args.snake_source_root,
            grouped_exact_max_active_qubits=args.grouped_exact_max_active_qubits,
        )
        audit["retrieval_archives"] = retrieval_archive_provenance(args.retrieval_archive)
        audit["condor_evidence_timestamp"] = args.condor_evidence_timestamp
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0
    manifest = build_report(
        inventory_path=args.inventory,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
        compile_qiskit=not args.skip_qiskit,
        grouped_exact_max_active_qubits=args.grouped_exact_max_active_qubits,
        append_source_roots=args.append_source_root,
        snake_source_roots=args.snake_source_root,
        require_complete_overlays=args.require_complete_overlays,
        retrieval_archives=args.retrieval_archive,
        condor_evidence_timestamp=args.condor_evidence_timestamp,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
