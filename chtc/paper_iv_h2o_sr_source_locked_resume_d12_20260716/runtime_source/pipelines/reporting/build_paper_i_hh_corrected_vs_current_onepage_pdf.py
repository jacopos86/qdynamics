#!/usr/bin/env python3
"""Build a one-page Paper-I HH corrected-versus-current comparison PDF.

The artifact overlays completed corrected parent-generator Geo/Append reruns on
the currently visible Paper-I Geo/Append trajectories and costs.  SNAKE is the
unchanged Paper-I reference.  Strong--strong corrected rows are deliberately
deferred by default and are never synthesized from the current Paper-I rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting import (  # noqa: E402
    build_paper_i_hh_corrected_parent_comparator_page13_pdf as base,
)


SCHEMA = "paper_i_hh_corrected_vs_current_onepage_report_v1"
STEM = "paper_i_hh_corrected_vs_current_onepage_20260710"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/pdf/paper_i_hh_corrected_vs_current_20260710"
RECOVERY_REPO_ROOT = Path(
    "/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3"
)
CURRENT_PAPER_PLOT_PROVENANCE = base.SNAKE_PLOT_PROVENANCE
CURRENT_PAPER_SUPPORT_CSV = RECOVERY_REPO_ROOT / (
    "output/pdf/paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630/"
    "paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630_"
    "powell_pool_exposure_support.csv"
)
CURRENT_ROLE_KEYS = {"geo": "geo_macro_c", "append": "append_macro_c"}
DEFERRED_PAIRS = frozenset({("strong-strong", "geo"), ("strong-strong", "append")})
RUNNING_PAIRS = frozenset({("weak-strong", "append"), ("intermediate-strong", "append")})


@dataclass(frozen=True)
class CurrentPaperRow:
    regime: str
    method: str
    k_pl: int
    table_error: float
    plot_marker_error: float
    n2q: int
    d2q: int
    dc: int
    s_alg: int
    curve: tuple[base.CurvePoint, ...]
    source_json: str
    source_sha256: str
    support_csv: str
    support_csv_sha256: str
    cost_source: str
    validation: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "method": self.method,
            "method_display": base.METHOD_DISPLAY[self.method],
            "role": "current_paper_i",
            "k_pl": self.k_pl,
            "table_abs_delta_e": self.table_error,
            "plot_marker_abs_delta_e": self.plot_marker_error,
            "N2q": self.n2q,
            "D2q": self.d2q,
            "Dc": self.dc,
            "S_alg": self.s_alg,
            "trajectory_points": [point.__dict__ for point in self.curve],
            "source_json": self.source_json,
            "source_sha256": self.source_sha256,
            "support_csv": self.support_csv,
            "support_csv_sha256": self.support_csv_sha256,
            "cost_source": self.cost_source,
            "validation": dict(self.validation),
        }


def _current_paper_cells() -> dict[tuple[str, str], dict[str, int | float]]:
    """Parse the 18 active mini-table rows in the page-13 composite."""

    source = base.PAPER_I_TEX.read_text(encoding="utf-8")
    label_index = source.index(r"\label{fig:hh_main_results_composite}")
    start_index = source.rfind(r"\onecolumngrid", 0, label_index)
    if start_index < 0:
        raise ValueError("Could not isolate the active Paper-I HH composite")
    block = source[start_index:label_index]
    pattern = re.compile(
        r"^(SNAKE|Geo|Append)\s*&\s*(\d+)\s*&\s*([0-9.eE+\-]+)\s*&\s*"
        r"([0-9,]+)\s*&\s*([0-9,]+)\s*&\s*([0-9,]+)\s*&\s*([0-9,]+)\s*\\\\\s*$",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(block))
    if len(matches) != 3 * len(base.REGIME_ORDER):
        raise ValueError(f"Expected 18 active Paper-I HH rows, found {len(matches)}")
    method_key = {"SNAKE": "snake", "Geo": "geo", "Append": "append"}
    cells: dict[tuple[str, str], dict[str, int | float]] = {}
    for index, match in enumerate(matches):
        regime = base.REGIME_ORDER[index // 3]
        method = method_key[match.group(1)]
        cells[(regime, method)] = {
            "k_pl": int(match.group(2)),
            "abs_delta_e": float(match.group(3)),
            "N2q": int(match.group(4).replace(",", "")),
            "D2q": int(match.group(5).replace(",", "")),
            "Dc": int(match.group(6).replace(",", "")),
            "S_alg": int(match.group(7).replace(",", "")),
        }
    return cells


def _resolve_recovery_source(raw: str | Path) -> Path:
    raw_path = Path(raw)
    if raw_path.is_absolute():
        candidates = [raw_path]
    else:
        candidates = [REPO_ROOT / raw_path, RECOVERY_REPO_ROOT / raw_path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(candidates[-1])


def _read_support_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_current_paper_rows() -> list[CurrentPaperRow]:
    plot_provenance = base.read_json(CURRENT_PAPER_PLOT_PROVENANCE)
    support_path = _resolve_recovery_source(plot_provenance["support_csv"])
    expected_support_hash = str(plot_provenance["support_csv_sha256"])
    if support_path.resolve() != CURRENT_PAPER_SUPPORT_CSV.resolve():
        raise ValueError("Current Paper-I support CSV resolved to an unexpected path")
    if base.sha256(support_path) != expected_support_hash:
        raise ValueError("Current Paper-I support CSV hash mismatch")
    support_rows = _read_support_rows(support_path)
    support_by_key = {
        (str(row["regime"]), str(row["method"])): row
        for row in support_rows
        if str(row.get("role_key")) in set(CURRENT_ROLE_KEYS.values())
    }
    plot_by_regime = {str(row["regime"]): row for row in plot_provenance["plots"]}
    cells = _current_paper_cells()
    rows: list[CurrentPaperRow] = []
    for regime in base.REGIME_ORDER:
        for method in ("geo", "append"):
            support = support_by_key[(regime, method)]
            role_key = CURRENT_ROLE_KEYS[method]
            method_provenance = next(
                row for row in plot_by_regime[regime]["methods"] if str(row["role_key"]) == role_key
            )
            cell = cells[(regime, method)]
            source_path = _resolve_recovery_source(method_provenance["source_json"])
            source_hash = base.sha256(source_path)
            trajectory_raw = json.loads(str(support["trajectory_points_json"]))
            curve = tuple(
                base.CurvePoint(int(round(float(point[0]))), base._positive_error(float(point[1])))
                for point in trajectory_raw
            )
            checks = {
                "support_status_done": str(support.get("status")) == "done",
                "selection_status_ok": str(support.get("selection_status")) == "ok",
                "qiskit_cost_status_ok": str(support.get("cost_status")) == "ok",
                "qiskit_cost_source": str(support.get("cost_source"))
                == "qiskit_selected_generic_prefix_compile",
                "s_work_status_ok": str(support.get("s_work_status")) == "ok",
                "role_key": str(support.get("role_key")) == role_key,
                "source_path_match": str(support.get("source_json"))
                == str(method_provenance.get("source_json")),
                "source_hash_match": source_hash
                == str(support.get("source_sha256"))
                == str(method_provenance.get("source_sha256")),
                "curve_begins_at_zero": bool(curve) and curve[0].k == 0,
                "curve_point_count": len(curve) == int(method_provenance["point_count"]),
                "active_k_matches_plot_marker": int(cell["k_pl"])
                == int(method_provenance["marker_k"]),
                "active_error_matches_table_source": math.isclose(
                    float(cell["abs_delta_e"]),
                    float(method_provenance["table_error"]),
                    rel_tol=5.0e-4,
                    abs_tol=5.0e-12,
                ),
            }
            override = method_provenance if all(
                key in method_provenance for key in ("N2q", "D2q", "Dc", "S_alg")
            ) else None
            if override is None:
                checks.update(
                    active_prefix_matches_support=int(cell["k_pl"])
                    == int(float(support["selected_prefix_k"])),
                    active_n2q_matches_support=int(cell["N2q"]) == int(float(support["N2q"])),
                    active_d2q_matches_support=int(cell["D2q"]) == int(float(support["D2q"])),
                    active_dc_matches_support=int(cell["Dc"]) == int(float(support["Dc"])),
                    active_s_matches_support=int(cell["S_alg"]) == int(float(support["S_alg"])),
                )
                cost_source = str(support["cost_source"])
            else:
                override_lock = plot_provenance.get("strong_strong_append_k8_update") or {}
                override_json = _resolve_recovery_source(str(override_lock["source_json"]))
                override_csv = _resolve_recovery_source(str(override_lock["source_csv"]))
                checks.update(
                    override_is_strong_strong_append=(regime, method) == ("strong-strong", "append"),
                    override_compile_status_ok=str(override_lock.get("compile_status")) == "ok",
                    override_k_matches=int(cell["k_pl"]) == int(override_lock["plot_iteration_k"]),
                    override_error_matches=math.isclose(
                        float(method_provenance["table_error"]),
                        float(override_lock["abs_delta_e"]),
                        rel_tol=0.0,
                        abs_tol=1.0e-15,
                    ),
                    active_n2q_matches_override=int(cell["N2q"]) == int(override["N2q"]),
                    active_d2q_matches_override=int(cell["D2q"]) == int(override["D2q"]),
                    active_dc_matches_override=int(cell["Dc"]) == int(override["Dc"]),
                    active_s_matches_override=int(cell["S_alg"]) == int(override["S_alg"]),
                    override_json_hash_match=base.sha256(override_json)
                    == str(override_lock["source_json_sha256"]),
                    override_csv_hash_match=base.sha256(override_csv)
                    == str(override_lock["source_csv_sha256"]),
                )
                cost_source = str(override.get("table_cost_source") or "paper_i_explicit_prefix_override")
            if not all(checks.values()):
                failed = [key for key, value in checks.items() if not value]
                raise ValueError(f"Current Paper-I row validation failed for {regime}/{method}: {failed}")
            rows.append(
                CurrentPaperRow(
                    regime=regime,
                    method=method,
                    k_pl=int(cell["k_pl"]),
                    table_error=float(method_provenance["table_error"]),
                    plot_marker_error=float(method_provenance["marker_error"]),
                    n2q=int(cell["N2q"]),
                    d2q=int(cell["D2q"]),
                    dc=int(cell["Dc"]),
                    s_alg=int(cell["S_alg"]),
                    curve=curve,
                    source_json=str(method_provenance["source_json"]),
                    source_sha256=source_hash,
                    support_csv=base.rel(support_path),
                    support_csv_sha256=expected_support_hash,
                    cost_source=cost_source,
                    validation=checks,
                )
            )
    return rows


def build_completed_corrected_rows(
    *,
    weak_weak_root: Path,
    corrected_root: Path,
    deferred_pairs: frozenset[tuple[str, str]] = DEFERRED_PAIRS,
) -> list[base.DisplayRow]:
    rows = base.build_snake_rows()
    for regime in base.REGIME_ORDER:
        for method in ("geo", "append"):
            if (regime, method) in deferred_pairs:
                continue
            result_path = base.corrected_result_path(
                regime,
                method,
                weak_weak_root=weak_weak_root,
                corrected_root=corrected_root,
            )
            if not result_path.is_file():
                progress_path = corrected_root / regime / method / "adapt_iteration_progress.jsonl"
                if (regime, method) in RUNNING_PAIRS and progress_path.is_file():
                    continue
                raise FileNotFoundError(
                    f"Corrected {regime}/{method} result is incomplete; refusing progress-JSON substitution: "
                    f"{result_path}"
                )
            rows.append(base.build_corrected_row(regime, method, result_path))
    rows.sort(
        key=lambda row: (
            base.REGIME_ORDER.index(row.regime),
            base.METHOD_ORDER.index(row.method),
        )
    )
    return rows


def validate_rows(
    corrected_rows: Sequence[base.DisplayRow],
    current_rows: Sequence[CurrentPaperRow],
    *,
    deferred_pairs: frozenset[tuple[str, str]],
) -> dict[str, Any]:
    snake_pairs = {(row.regime, row.method) for row in corrected_rows if row.method == "snake"}
    corrected_pairs = {(row.regime, row.method) for row in corrected_rows if row.method != "snake"}
    expected_corrected = {
        (regime, method)
        for regime in base.REGIME_ORDER
        for method in ("geo", "append")
        if (regime, method) not in deferred_pairs
    }
    missing_corrected = expected_corrected - corrected_pairs
    current_pairs = {(row.regime, row.method) for row in current_rows}
    checks = {
        "six_current_snake_rows": snake_pairs == {(regime, "snake") for regime in base.REGIME_ORDER},
        "all_available_corrected_rows": corrected_pairs.issubset(expected_corrected),
        "only_running_rows_lack_completed_results": missing_corrected.issubset(RUNNING_PAIRS),
        "twelve_current_paper_comparator_rows": current_pairs
        == {(regime, method) for regime in base.REGIME_ORDER for method in ("geo", "append")},
        "deferred_pairs_are_strong_strong_only": deferred_pairs == DEFERRED_PAIRS,
        "no_corrected_row_fabricated_for_deferred_pairs": corrected_pairs.isdisjoint(deferred_pairs),
        "all_current_costs_nonnegative": all(
            min(row.n2q, row.d2q, row.dc, row.s_alg) >= 0 for row in current_rows
        ),
        "all_current_curves_begin_at_zero": all(row.curve and row.curve[0].k == 0 for row in current_rows),
    }
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise ValueError(f"Corrected/current comparison validation failed: {failed}")
    return checks


def _plot_regime(
    regime: str,
    corrected_rows: Sequence[base.DisplayRow],
    current_rows: Sequence[CurrentPaperRow],
    *,
    figure_dir: Path,
    stem: str,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure_dir.mkdir(parents=True, exist_ok=True)
    corrected = {row.method: row for row in corrected_rows if row.regime == regime}
    current = {row.method: row for row in current_rows if row.regime == regime}
    fig, ax = plt.subplots(figsize=(3.15, 2.05))
    for method in ("geo", "append"):
        row = current[method]
        style = base.METHOD_STYLE[method]
        ax.plot(
            [point.k for point in row.curve],
            [point.error for point in row.curve],
            color=style["color"],
            linestyle="--",
            linewidth=0.95,
            alpha=0.62,
        )
        ax.scatter(
            [row.k_pl],
            [row.plot_marker_error],
            facecolors="none",
            edgecolors=style["color"],
            marker=style["marker"],
            s=25,
            linewidth=0.8,
            zorder=4,
        )
    snake = corrected["snake"]
    snake_style = base.METHOD_STYLE["snake"]
    ax.plot(
        [point.k for point in snake.curve],
        [point.error for point in snake.curve],
        color=snake_style["color"],
        linestyle="-",
        linewidth=1.35,
        alpha=0.94,
    )
    ax.scatter(
        [snake.k_pl],
        [snake.abs_delta_e],
        color=snake_style["color"],
        marker=snake_style["marker"],
        s=43,
        edgecolor="black",
        linewidth=0.3,
        zorder=5,
    )
    for method in ("geo", "append"):
        row = corrected.get(method)
        if row is None:
            continue
        style = base.METHOD_STYLE[method]
        ax.plot(
            [point.k for point in row.curve],
            [point.error for point in row.curve],
            color=style["color"],
            linestyle="-",
            linewidth=1.45,
            alpha=0.98,
        )
        ax.scatter(
            [row.k_pl],
            [row.abs_delta_e],
            color=style["color"],
            marker=style["marker"],
            s=26,
            edgecolor="black",
            linewidth=0.3,
            zorder=5,
        )
    ax.set_yscale("log")
    ax.set_xlim(left=0)
    ax.set_xlabel("ADAPT outer iteration $k$", fontsize=7.5)
    ax.set_ylabel(r"$|\Delta E|$", fontsize=7.5)
    title = base.REGIME_DISPLAY[regime]
    if regime == "strong-strong":
        title += " (corrected deferred)"
    ax.set_title(title, fontsize=8.7)
    ax.tick_params(axis="both", labelsize=6.4)
    ax.grid(True, which="major", alpha=0.22, linewidth=0.4)
    handles = [
        Line2D(
            [0],
            [0],
            color=snake_style["color"],
            linestyle="-",
            marker=snake_style["marker"],
            markersize=5.2,
            label="SNAKE P-I",
        )
    ]
    for method in ("geo", "append"):
        style = base.METHOD_STYLE[method]
        if method in corrected:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=style["color"],
                    linestyle="-",
                    marker=style["marker"],
                    markersize=3.8,
                    label=f"{base.METHOD_DISPLAY[method]} corr.",
                )
            )
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle="--",
                marker=style["marker"],
                markerfacecolor="none",
                markersize=3.8,
                label=f"{base.METHOD_DISPLAY[method]} P-I",
            )
        )
    ax.legend(
        handles=handles,
        loc="best",
        fontsize=4.2,
        frameon=False,
        handlelength=1.35,
        ncol=2,
        columnspacing=0.6,
        labelspacing=0.3,
    )
    fig.tight_layout(pad=0.35)
    safe = regime.replace("-", "_")
    png = figure_dir / f"{stem}__{safe}.png"
    pdf = figure_dir / f"{stem}__{safe}.pdf"
    fig.savefig(png, dpi=280)
    fig.savefig(pdf)
    plt.close(fig)
    return {
        "regime": regime,
        "png": base.rel(png),
        "png_sha256": base.sha256(png),
        "pdf": base.rel(pdf),
        "pdf_sha256": base.sha256(pdf),
    }


def _table_tex(
    regime: str,
    corrected_rows: Sequence[base.DisplayRow],
    current_rows: Sequence[CurrentPaperRow],
) -> str:
    corrected = {row.method: row for row in corrected_rows if row.regime == regime}
    current = {row.method: row for row in current_rows if row.regime == regime}
    lines = [
        r"\begin{tabular*}{\linewidth}{@{}l@{\extracolsep{\fill}}rrrrrr@{}}",
        r"\toprule",
        r"Row & $k$ & $|\Delta E|$ & $N_{2q}$ & $D_{2q}$ & $D_c$ & $S$\\",
        r"\midrule",
    ]
    snake = corrected["snake"]
    lines.append(
        f"SNAKE P-I & {snake.k_pl} & {base.format_error(snake.abs_delta_e)} & "
        f"{snake.n2q:,} & {snake.d2q:,} & {snake.dc:,} & {snake.s_alg:,} \\\\"
    )
    for method, short in (("geo", "Geo"), ("append", "App.")):
        paper = current[method]
        lines.append(
            f"{short} P-I & {paper.k_pl} & {base.format_error(paper.table_error)} & "
            f"{paper.n2q:,} & {paper.d2q:,} & {paper.dc:,} & {paper.s_alg:,} \\\\"
        )
        corrected_row = corrected.get(method)
        if corrected_row is None:
            pair = (regime, method)
            if pair in DEFERRED_PAIRS:
                status = "deferred"
            elif pair in RUNNING_PAIRS:
                status = "running"
            else:
                raise ValueError(f"Unclassified missing corrected row: {pair}")
            lines.append(f"{short} corr. & \\multicolumn{{6}}{{c}}{{{status}}} \\\\")
        else:
            lines.append(
                f"{short} corr. & {corrected_row.k_pl} & {base.format_error(corrected_row.abs_delta_e)} & "
                f"{corrected_row.n2q:,} & {corrected_row.d2q:,} & {corrected_row.dc:,} & "
                f"{corrected_row.s_alg:,} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular*}"])
    return "\n".join(lines)


def _write_csv(
    path: Path,
    corrected_rows: Sequence[base.DisplayRow],
    current_rows: Sequence[CurrentPaperRow],
) -> None:
    fields = [
        "regime",
        "method",
        "role",
        "k_pl",
        "table_abs_delta_e",
        "plot_marker_abs_delta_e",
        "N2q",
        "D2q",
        "Dc",
        "S_alg",
        "source_json",
        "source_sha256",
        "cost_source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in corrected_rows:
            writer.writerow(
                {
                    "regime": row.regime,
                    "method": base.METHOD_DISPLAY[row.method],
                    "role": "current_paper_i" if row.method == "snake" else "corrected",
                    "k_pl": row.k_pl,
                    "table_abs_delta_e": f"{row.abs_delta_e:.17g}",
                    "plot_marker_abs_delta_e": f"{row.abs_delta_e:.17g}",
                    "N2q": row.n2q,
                    "D2q": row.d2q,
                    "Dc": row.dc,
                    "S_alg": row.s_alg,
                    "source_json": row.source_json,
                    "source_sha256": row.source_sha256,
                    "cost_source": row.cost_source,
                }
            )
        for row in current_rows:
            writer.writerow(
                {
                    "regime": row.regime,
                    "method": base.METHOD_DISPLAY[row.method],
                    "role": "current_paper_i",
                    "k_pl": row.k_pl,
                    "table_abs_delta_e": f"{row.table_error:.17g}",
                    "plot_marker_abs_delta_e": f"{row.plot_marker_error:.17g}",
                    "N2q": row.n2q,
                    "D2q": row.d2q,
                    "Dc": row.dc,
                    "S_alg": row.s_alg,
                    "source_json": row.source_json,
                    "source_sha256": row.source_sha256,
                    "cost_source": row.cost_source,
                }
            )


def _write_onepage_tex(
    path: Path,
    *,
    corrected_rows: Sequence[base.DisplayRow],
    current_rows: Sequence[CurrentPaperRow],
    figures: Sequence[Mapping[str, Any]],
    report_json: Path,
    report_csv: Path,
    generated_utc: str,
) -> None:
    figure_by_regime = {str(row["regime"]): row for row in figures}
    source_comment = json.dumps(
        {
            "schema": SCHEMA,
            "report_json": base.rel(report_json),
            "report_csv": base.rel(report_csv),
            "run_class": "candidate",
            "strong_strong_corrected": "deferred_by_user_scope",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    panels: list[str] = []
    for index, regime in enumerate(base.REGIME_ORDER):
        figure = base.resolve_source_path(str(figure_by_regime[regime]["pdf"]))
        panels.extend(
            [
                r"\begin{minipage}[t]{0.322\textwidth}",
                r"\centering",
                f"\\includegraphics[width=\\linewidth]{{{base.latex_graphics_path(figure)}}}",
                r"\par\vspace{-0.4ex}",
                r"{\fontsize{4.8}{5.35}\selectfont",
                _table_tex(regime, corrected_rows, current_rows),
                r"}",
                r"\end{minipage}",
            ]
        )
        if index in {0, 1, 3, 4}:
            panels.append(r"\hfill")
        elif index == 2:
            panels.append(r"\par\vspace{1.0ex}")
    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[letterpaper,margin=0.26in]{{geometry}}
\usepackage{{booktabs,graphicx,caption,microtype,xcolor}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
% BEGIN_MACHINE_READABLE_CORRECTED_VS_CURRENT_REPORT
% {source_comment}
% END_MACHINE_READABLE_CORRECTED_VS_CURRENT_REPORT
\begin{{document}}
\begin{{center}}
{{\large\bfseries Paper-I Hubbard--Holstein: corrected parent comparators vs current Paper-I values}}\\[-0.2ex]
{{\fontsize{{5.8}}{{6.5}}\selectfont
\begin{{tabular*}}{{0.99\linewidth}}{{@{{}}l@{{\extracolsep{{\fill}}}}p{{0.88\linewidth}}@{{}}}}
\toprule
Manifest & Candidate report generated {base.latex_escape(generated_utc)}; no manuscript source edited.\\
Corrected contract & Powell 200; 30 fixed selector scans; full-meta parent macro generators; HVA included; same-cutoff $|E_k-E_0|$; Geo scores the full pool then blocks an immediate repeated append.\\
Comparison & Solid/filled Geo and Append are corrected reruns; dashed/open Geo and Append reproduce the current Paper-I trajectories and plotted markers; SNAKE is the current Paper-I reference.\\
Deferred scope & Strong--strong corrected Geo/Append were not run at the user's request; its current Paper-I rows remain visible and corrected rows are marked deferred.\\
Running scope & Weak--strong and intermediate--strong corrected Append rows are included only when a completed result JSON exists; otherwise they are marked running and no progress JSONL is used as evidence.\\
Costs & $N_{{2q}}$, $D_{{2q}}$, and $D_c$ are Qiskit prefix costs; $S$ is the logical estimator-query count at the same table prefix, not physical hardware shots.\\
\bottomrule
\end{{tabular*}}}}
\end{{center}}
\vspace{{0.4ex}}
\begin{{center}}
\scriptsize
\setlength{{\tabcolsep}}{{0pt}}
\renewcommand{{\arraystretch}}{{0.88}}
{chr(10).join(panels)}
\par\vspace{{0.6ex}}
{{\fontsize{{5.7}}{{6.3}}\selectfont
Current Paper-I open-marker $y$ values follow the frozen plotted trajectories, while P-I mini-table errors reproduce the active table cells; those legacy values can differ because the current figure and table use distinct locked error fields. Completed corrected filled markers, errors, Qiskit costs, and $S$ are prefix-aligned; running/deferred rows are not synthesized from progress data.}}
\captionof{{figure}}{{\fontsize{{6.0}}{{6.7}}\selectfont Error versus ADAPT iteration and prefix Qiskit/query costs for all six Hubbard--Holstein regimes. Completed corrected Geo/Append reruns are overlaid on the current Paper-I values; weak--strong and intermediate--strong corrected Append are marked running until their result JSONs exist, and strong--strong corrected rows are deferred.}}
\end{{center}}
\end{{document}}
"""
    path.write_text(tex, encoding="utf-8")


def _pdf_page_count(path: Path) -> int:
    executable = shutil.which("pdfinfo")
    if executable is None:
        raise RuntimeError("pdfinfo is required to enforce the one-page contract")
    completed = subprocess.run(
        [executable, str(path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    match = re.search(r"^Pages:\s+(\d+)\s*$", completed.stdout, re.MULTILINE)
    if match is None:
        raise ValueError("pdfinfo did not report a page count")
    return int(match.group(1))


def build(
    *,
    weak_weak_root: Path,
    corrected_root: Path,
    output_dir: Path,
    stem: str = STEM,
    deferred_pairs: frozenset[tuple[str, str]] = DEFERRED_PAIRS,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_utc = base.utc_now()
    corrected_rows = build_completed_corrected_rows(
        weak_weak_root=weak_weak_root,
        corrected_root=corrected_root,
        deferred_pairs=deferred_pairs,
    )
    current_rows = build_current_paper_rows()
    validation = validate_rows(
        corrected_rows,
        current_rows,
        deferred_pairs=deferred_pairs,
    )
    figure_dir = output_dir / "figures"
    figures = [
        _plot_regime(
            regime,
            corrected_rows,
            current_rows,
            figure_dir=figure_dir,
            stem=stem,
        )
        for regime in base.REGIME_ORDER
    ]
    report_json = output_dir / f"{stem}.json"
    report_csv = output_dir / f"{stem}.csv"
    report_tex = output_dir / f"{stem}.tex"
    _write_csv(report_csv, corrected_rows, current_rows)
    _write_onepage_tex(
        report_tex,
        corrected_rows=corrected_rows,
        current_rows=current_rows,
        figures=figures,
        report_json=report_json,
        report_csv=report_csv,
        generated_utc=generated_utc,
    )
    report_pdf = base.compile_latex(report_tex)
    page_count = _pdf_page_count(report_pdf)
    if page_count != 1:
        raise ValueError(f"One-page PDF contract failed: rendered {page_count} pages")
    visible_source_map = corrected_root / "visible_source_map.json"
    batch_manifest = corrected_root / "batch_manifest.json"
    resolver_traces = sorted((corrected_root / "source_locks").glob("*.json"))
    if not visible_source_map.is_file() or not batch_manifest.is_file() or len(resolver_traces) != 12:
        raise ValueError("Corrected source map, batch manifest, or 12 resolver locks are missing")
    plot_provenance = base.read_json(CURRENT_PAPER_PLOT_PROVENANCE)
    support_path = _resolve_recovery_source(plot_provenance["support_csv"])
    strong_strong_override = plot_provenance["strong_strong_append_k8_update"]
    strong_strong_override_json = _resolve_recovery_source(strong_strong_override["source_json"])
    strong_strong_override_csv = _resolve_recovery_source(strong_strong_override["source_csv"])
    paper_i_provenance = base.read_json(base.PAPER_I_PROVENANCE)
    snake_comparison = base.resolve_source_path(paper_i_provenance["comparison_json"])
    if base.sha256(snake_comparison) != str(paper_i_provenance["comparison_json_sha256"]):
        raise ValueError("SNAKE Qiskit comparison hash drifted after row validation")
    corrected_pairs = {
        (row.regime, row.method)
        for row in corrected_rows
        if row.method in {"geo", "append"}
    }
    all_comparator_pairs = {
        (regime, method)
        for regime in base.REGIME_ORDER
        for method in ("geo", "append")
    }
    unavailable_pairs = all_comparator_pairs - corrected_pairs
    payload = {
        "schema": SCHEMA,
        "generated_utc": generated_utc,
        "run_class": "candidate",
        "manuscript_edited": False,
        "one_page": True,
        "deferred_corrected_pairs": [
            {"regime": regime, "method": method, "reason": "deferred_by_user_scope"}
            for regime, method in sorted(deferred_pairs)
        ],
        "unavailable_corrected_pairs": [
            {
                "regime": regime,
                "method": method,
                "status": (
                    "deferred_by_user_scope"
                    if (regime, method) in deferred_pairs
                    else "running_no_completed_result_omitted"
                ),
                "progress_json_used_as_evidence": False,
            }
            for regime, method in sorted(unavailable_pairs)
        ],
        "contract": {
            "optimizer": "POWELL",
            "optimizer_maxiter": 200,
            "selector_scans": 30,
            "pool": "full_meta_parent_macro_generators",
            "primary_error": "same_cutoff_abs_delta_e",
            "qiskit_compile_convention": base.QISKIT_COMPILE_CONVENTION,
            "S_formula": "N_H_outer + N_H_refit + N_grad + N_metric + N_other_quantum",
            "overlay_style": {
                "corrected": "solid_curve_filled_marker",
                "current_paper_i": "dashed_curve_open_marker",
                "snake": "current_paper_i_reference_only",
            },
            "legacy_paper_i_marker_table_policy": (
                "open_marker_y_from_frozen_plot_trajectory; mini_table_error_from_active_Paper_I_cell"
            ),
        },
        "source_locks": {
            "weak_weak_root": base.rel(weak_weak_root),
            "corrected_root": base.rel(corrected_root),
            "visible_source_map": base.rel(visible_source_map),
            "visible_source_map_sha256": base.sha256(visible_source_map),
            "batch_manifest": base.rel(batch_manifest),
            "batch_manifest_sha256": base.sha256(batch_manifest),
            "resolver_traces": [
                {"path": base.rel(path), "sha256": base.sha256(path)} for path in resolver_traces
            ],
            "active_paper_i_tex": base.rel(base.PAPER_I_TEX),
            "active_paper_i_tex_sha256": base.sha256(base.PAPER_I_TEX),
            "paper_i_provenance": base.rel(base.PAPER_I_PROVENANCE),
            "paper_i_provenance_sha256": base.sha256(base.PAPER_I_PROVENANCE),
            "snake_qiskit_comparison_json": base.rel(snake_comparison),
            "snake_qiskit_comparison_json_sha256": base.sha256(snake_comparison),
            "snake_s_accounting_shadow": base.rel(base.SNAKE_S_SHADOW),
            "snake_s_accounting_shadow_sha256": base.sha256(base.SNAKE_S_SHADOW),
            "current_paper_plot_provenance": base.rel(CURRENT_PAPER_PLOT_PROVENANCE),
            "current_paper_plot_provenance_sha256": base.sha256(CURRENT_PAPER_PLOT_PROVENANCE),
            "current_paper_support_csv": base.rel(support_path),
            "current_paper_support_csv_sha256": base.sha256(support_path),
            "strong_strong_append_k8_override_json": base.rel(strong_strong_override_json),
            "strong_strong_append_k8_override_json_sha256": base.sha256(strong_strong_override_json),
            "strong_strong_append_k8_override_csv": base.rel(strong_strong_override_csv),
            "strong_strong_append_k8_override_csv_sha256": base.sha256(strong_strong_override_csv),
        },
        "corrected_and_snake_rows": [row.as_dict() for row in corrected_rows],
        "current_paper_i_comparator_rows": [row.as_dict() for row in current_rows],
        "figures": figures,
        "validation": validation,
        "artifacts": {
            "pdf": base.rel(report_pdf),
            "pdf_sha256": base.sha256(report_pdf),
            "pdf_page_count": page_count,
            "tex": base.rel(report_tex),
            "tex_sha256": base.sha256(report_tex),
            "csv": base.rel(report_csv),
            "csv_sha256": base.sha256(report_csv),
            "json": base.rel(report_json),
        },
    }
    base.write_json(report_json, payload)
    payload["artifacts"]["json_sha256"] = base.sha256(report_json)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-weak-root", type=Path, default=base.DEFAULT_WEAK_WEAK_ROOT)
    parser.add_argument("--corrected-root", type=Path, default=base.DEFAULT_CORRECTED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default=STEM)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build(
        weak_weak_root=args.weak_weak_root.resolve(),
        corrected_root=args.corrected_root.resolve(),
        output_dir=args.output_dir.resolve(),
        stem=str(args.stem),
    )
    print(json.dumps({"status": "ok", **payload["artifacts"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
