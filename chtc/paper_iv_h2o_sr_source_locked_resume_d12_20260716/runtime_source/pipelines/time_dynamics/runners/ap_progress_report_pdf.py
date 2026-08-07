"""Build LaTeX APM diagnostic progression PDFs from trajectory JSONs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPORT_SCHEMA_V1 = "ap_mclachlan_progress_report_v2"
DEFAULT_THRESHOLDS = (1.0e-3, 1.0e-2, 1.0e-1, 1.0, 2.0, 3.0)
DEFAULT_COMPACT_PANEL_PAGE_ROWS = 6
APPEND_MARKER_COLOR = "#1F77B4"
PRUNE_MARKER_COLOR = "#C62828"
STABILIZATION_BAND_COLOR = "#F2C94C"


@dataclass(frozen=True)
class PatchEvent:
    index: int
    time: float
    patch_kind: str
    selected_label: str
    reason: str
    rank_score: float | None
    abs_energy_error: float | None
    theta_dot_l2: float | None
    residual_ratio: float | None


@dataclass(frozen=True)
class ThresholdHit:
    threshold: float
    index: int
    time: float
    abs_energy_error: float


@dataclass(frozen=True)
class RunSummary:
    run_index: int
    path: str
    label: str
    schema: str
    route_label: str
    drive_enabled: bool
    point_count: int
    integrator_method: str
    initial_abs_energy_error: float | None
    final_abs_energy_error: float | None
    max_abs_energy_error: float | None
    max_abs_energy_error_time: float | None
    seed_final_abs_energy_error: float | None
    seed_max_abs_energy_error: float | None
    final_abs_doublon_error: float | None
    max_abs_doublon_error: float | None
    seed_final_abs_doublon_error: float | None
    seed_max_abs_doublon_error: float | None
    final_site_occupations_abs_error_max: float | None
    max_abs_site_occupations_error: float | None
    seed_final_site_occupations_abs_error_max: float | None
    seed_max_abs_site_occupations_error: float | None
    accepted_patch_count: int
    prune_patch_smoothness_deferred_count: int
    prune_patch_smoothness_passed_count: int
    prune_patch_smoothness_retry_count: int
    prune_patch_smoothness_accepted_after_retry_count: int
    max_prune_patch_smoothness_eta: float | None
    max_prune_patch_smoothness_severity: float | None
    events: tuple[PatchEvent, ...]
    threshold_hits: tuple[ThresholdHit, ...]


def build_ap_progress_report_pdf(
    *,
    trajectory_jsons: Sequence[str | Path],
    plot_pngs: Sequence[str | Path] = (),
    cost_table_jsons: Sequence[str | Path] = (),
    output_pdf: str | Path,
    output_manifest: str | Path | None = None,
    title: str = "APM Diagnostic Progress Report",
    notes: Sequence[str] = (),
    plot_grid_columns: int = 3,
    compact_panel_page_rows: int = DEFAULT_COMPACT_PANEL_PAGE_ROWS,
    build_pdf: bool = True,
) -> dict[str, Any]:
    """Create a plot-first APM diagnostic report.

    The report is written as LaTeX and built with tectonic. It intentionally does
    not use ReportLab tables.
    """

    trajectory_paths = [Path(path) for path in trajectory_jsons]
    run_count = len(trajectory_paths)
    summaries = [
        _summarize_run(path, run_index=run_count - index)
        for index, path in enumerate(trajectory_paths)
    ]
    output_path = Path(output_pdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_tex = output_path.with_suffix(".tex")
    compact_panel_base = output_path.with_name(f"{output_path.stem}_compact_panels")
    image_paths = _existing_plot_paths(plot_pngs)
    cost_table_paths = _existing_plot_paths(cost_table_jsons)

    cost_lookup = _load_qiskit_cost_lookup(cost_table_paths)
    if summaries:
        compact_panel_pngs = _write_compact_panels(
            trajectory_jsons=[Path(summary.path) for summary in summaries],
            output_base=compact_panel_base,
            title="Total energy, doublon, and expressivity miss",
            max_rows_per_page=int(compact_panel_page_rows),
            qiskit_cost_lookup=cost_lookup,
        )
    else:
        compact_panel_pngs = ()

    output_tex.write_text(
        _render_latex(
            title=title,
            summaries=summaries,
            compact_panel_pngs=compact_panel_pngs,
            plot_pngs=image_paths,
            cost_table_jsons=cost_table_paths,
            notes=notes,
            plot_grid_columns=max(1, int(plot_grid_columns)),
        ),
        encoding="utf-8",
    )
    if build_pdf:
        _build_latex_pdf(output_tex=output_tex, output_pdf=output_path)

    manifest = {
        "schema": REPORT_SCHEMA_V1,
        "output_pdf": str(output_path),
        "output_tex": str(output_tex),
        "compact_panel_png": str(compact_panel_pngs[0]) if compact_panel_pngs else None,
        "compact_panel_pngs": [str(path) for path in compact_panel_pngs],
        "compact_panel_page_rows": int(compact_panel_page_rows),
        "trajectory_jsons": [str(path) for path in trajectory_jsons],
        "plot_pngs": [str(path) for path in image_paths],
        "cost_table_jsons": [str(path) for path in cost_table_paths],
        "notes": [str(note) for note in notes],
        "diagnostic_only": True,
        "exact_reference_scope": "post_run_reporting_overlay_only",
        "uses_reportlab_tables": False,
        "latex_built": bool(build_pdf),
        "marker_convention": {
            "append": "blue vertical bar",
            "prune": "red diamond",
            "numerical_stabilization": "faint amber time band",
        },
        "run_summaries": [_summary_to_json(summary) for summary in summaries],
    }
    if output_manifest not in {None, ""}:
        manifest_path = Path(str(output_manifest))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def discover_plot_pngs(plot_dirs: Sequence[str | Path]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for plot_dir in plot_dirs:
        root = Path(plot_dir)
        if root.is_dir():
            paths.extend(sorted(root.glob("*.png")))
    return tuple(_sort_plot_paths(paths))


def _write_compact_panels(
    *,
    trajectory_jsons: Sequence[Path],
    output_base: Path,
    title: str,
    max_rows_per_page: int,
    qiskit_cost_lookup: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Path, ...]:
    run_count = len(trajectory_jsons)
    runs = [
        _load_run(
            path,
            run_index=run_count - index,
            qiskit_cost_lookup=qiskit_cost_lookup or {},
        )
        for index, path in enumerate(trajectory_jsons)
    ]
    if not runs:
        return ()
    page_rows = max(1, int(max_rows_per_page))
    pages = [runs[index : index + page_rows] for index in range(0, len(runs), page_rows)]
    output_paths: list[Path] = []
    for page_index, page_runs in enumerate(pages, start=1):
        if len(pages) == 1:
            output_png = Path(str(output_base) + ".png")
            page_title = title
        else:
            output_png = Path(str(output_base) + f"_page{page_index:02d}.png")
            page_title = f"{title} - page {page_index}/{len(pages)}"
        _write_compact_panel_from_runs(runs=page_runs, output_png=output_png, title=page_title)
        output_paths.append(output_png)
    return tuple(output_paths)


def _write_compact_panel_from_runs(
    *,
    runs: Sequence[dict[str, Any]],
    output_png: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not runs:
        return
    shared_energy_ylim = _shared_compact_panel_ylim(
        runs,
        fields=("energy_expectation", "reference_energy", "seed_reference_energy"),
    )
    fig_width = 15.2
    fig_height = max(3.3, 2.42 * len(runs))
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    height_ratios: list[float] = []
    for _run in runs:
        height_ratios.extend([1.0, 0.18])
    grid = fig.add_gridspec(
        2 * len(runs),
        3,
        height_ratios=height_ratios,
    )
    for row_index, run in enumerate(runs):
        axes = [
            fig.add_subplot(grid[2 * row_index, 0]),
            fig.add_subplot(grid[2 * row_index, 1]),
            fig.add_subplot(grid[2 * row_index, 2]),
        ]
        cost_axis = fig.add_subplot(grid[2 * row_index + 1, :])
        rows = run["rows"]
        times = np.asarray([float(row["time"]) for row in rows], dtype=float)
        energy = np.asarray([float(row["energy_expectation"]) for row in rows], dtype=float)
        reference = np.asarray(
            [
                np.nan if row.get("reference_energy") is None else float(row["reference_energy"])
                for row in rows
            ],
            dtype=float,
        )
        seed_reference = np.asarray(
            [
                np.nan
                if row.get("seed_reference_energy") is None
                else float(row["seed_reference_energy"])
                for row in rows
            ],
            dtype=float,
        )
        rho_expr = np.asarray(
            [
                np.nan
                if row.get("mclachlan_rho_expr") is None
                else float(row["mclachlan_rho_expr"])
                for row in rows
            ],
            dtype=float,
        )
        residual_ratio = np.asarray(
            [
                np.nan
                if row.get("mclachlan_residual_ratio") is None
                else float(row["mclachlan_residual_ratio"])
                for row in rows
            ],
            dtype=float,
        )
        doublon = np.asarray(
            [
                np.nan if row.get("doublon") is None else float(row["doublon"])
                for row in rows
            ],
            dtype=float,
        )
        doublon_exact = np.asarray(
            [
                np.nan if row.get("doublon_exact") is None else float(row["doublon_exact"])
                for row in rows
            ],
            dtype=float,
        )
        seed_doublon_exact = np.asarray(
            [
                np.nan
                if row.get("seed_doublon_exact") is None
                else float(row["seed_doublon_exact"])
                for row in rows
            ],
            dtype=float,
        )
        events = [row for row in rows if bool(row.get("patch_accepted"))]

        ax_energy = axes[0]
        ax_doublon = axes[1]
        ax_miss = axes[2]
        ax_energy.plot(times, energy, linewidth=1.5, label="algorithm")
        reference_mask = np.isfinite(reference)
        if np.any(reference_mask):
            ax_energy.plot(
                times[reference_mask],
                reference[reference_mask],
                linewidth=1.1,
                linestyle="--",
                label="ED exact",
            )
        seed_reference_mask = np.isfinite(seed_reference)
        if np.any(seed_reference_mask):
            ax_energy.plot(
                times[seed_reference_mask],
                seed_reference[seed_reference_mask],
                linewidth=1.05,
                linestyle="-.",
                label="seed exact",
            )
        if shared_energy_ylim is not None:
            ax_energy.set_ylim(shared_energy_ylim)
        _annotate_energy_endpoints(ax_energy, times=times, energy=energy)
        doublon_mask = np.isfinite(doublon)
        if np.any(doublon_mask):
            ax_doublon.plot(times[doublon_mask], doublon[doublon_mask], linewidth=1.5, label="algorithm")
        exact_doublon_mask = np.isfinite(doublon_exact)
        if np.any(exact_doublon_mask):
            ax_doublon.plot(
                times[exact_doublon_mask],
                doublon_exact[exact_doublon_mask],
                linewidth=1.1,
                linestyle="--",
                label="ED exact",
            )
        seed_doublon_mask = np.isfinite(seed_doublon_exact)
        if np.any(seed_doublon_mask):
            ax_doublon.plot(
                times[seed_doublon_mask],
                seed_doublon_exact[seed_doublon_mask],
                linewidth=1.05,
                linestyle="-.",
                label="seed exact",
            )
        if not np.any(doublon_mask):
            ax_doublon.text(
                0.5,
                0.5,
                "doublon unavailable",
                transform=ax_doublon.transAxes,
                ha="center",
                va="center",
                fontsize=7,
                color="#555555",
            )
        rho_expr_mask = np.isfinite(rho_expr)
        if np.any(rho_expr_mask):
            ax_miss.semilogy(
                times[rho_expr_mask],
                np.maximum(rho_expr[rho_expr_mask], 1.0e-16),
                linewidth=1.5,
                label="rho_expr",
            )
        residual_mask = np.isfinite(residual_ratio)
        if np.any(residual_mask):
            ax_miss.semilogy(
                times[residual_mask],
                np.maximum(residual_ratio[residual_mask], 1.0e-16),
                linewidth=1.1,
                linestyle=":",
                label="residual ratio",
            )
        _annotate_numerical_stabilization(ax_energy, rows)
        _annotate_numerical_stabilization(ax_doublon, rows)
        _annotate_numerical_stabilization(ax_miss, rows)
        _annotate_patch_events(ax_energy, events, times=times, values=energy)
        _annotate_patch_events(ax_doublon, events, times=times, values=doublon)
        miss_marker_values = np.where(np.isfinite(rho_expr), rho_expr, residual_ratio)
        _annotate_patch_events(ax_miss, events, times=times, values=miss_marker_values)
        run_title = _run_title_for_axis(run)
        ax_energy.set_title(
            _wrapped_axis_title(run_title, "total energy"),
            fontsize=8,
            pad=3,
        )
        ax_doublon.set_title(
            _wrapped_axis_title(run_title, "doublon"),
            fontsize=8,
            pad=3,
        )
        ax_miss.set_title(
            _wrapped_axis_title(run_title, "expressivity miss"),
            fontsize=8,
            pad=3,
        )
        ax_energy.set_ylabel("energy")
        ax_doublon.set_ylabel("doublon")
        ax_miss.set_ylabel("miss")
        ax_energy.grid(True, alpha=0.25)
        ax_doublon.grid(True, alpha=0.25)
        ax_miss.grid(True, alpha=0.25, which="both")
        if row_index == len(runs) - 1:
            ax_energy.set_xlabel("time")
            ax_doublon.set_xlabel("time")
            ax_miss.set_xlabel("time")
        if row_index == 0:
            ax_energy.legend(fontsize=7, loc="best")
            if np.any(doublon_mask) or np.any(exact_doublon_mask) or np.any(seed_doublon_mask):
                ax_doublon.legend(fontsize=7, loc="best")
            if np.any(rho_expr_mask) or np.any(residual_mask):
                ax_miss.legend(fontsize=7, loc="best")
        _draw_qiskit_cost_strip(cost_axis, run)
    fig.suptitle(title, fontsize=12)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=190)
    plt.close(fig)


def _draw_qiskit_cost_strip(ax: Any, run: Mapping[str, Any]) -> None:
    ax.axis("off")
    summary = dict(run.get("summary", {}) or {})
    qiskit_cost = dict(run.get("qiskit_cost", {}) or {})
    cost_status = "Qiskit ok" if qiskit_cost.get("N2q") is not None else "Qiskit --"
    match_kind = str(qiskit_cost.get("_qiskit_cost_match") or "")
    if match_kind == "summary_exact":
        cost_status = "Qiskit matched"
    values = [
        cost_status,
        _fmt(_first_present(summary, qiskit_cost, ("logical_parameter_count_final",), ("logical_parameter_count",))),
        _fmt(_first_present(summary, qiskit_cost, ("runtime_parameter_count_final",), ("runtime_parameter_count",))),
        _fmt(_first_present(summary, qiskit_cost, ("accepted_append_count",), ("accepted_append_count",))),
        _fmt(_first_present(summary, qiskit_cost, ("accepted_appended_coordinate_count",), ("accepted_appended_coordinate_count",))),
        _fmt(_first_present(summary, qiskit_cost, ("accepted_delete_count",), ())),
        _fmt(_first_present(summary, qiskit_cost, ("accepted_deleted_coordinate_count",), ())),
        _fmt(_first_present(summary, qiskit_cost, ("final_abs_energy_error",), ("final_abs_energy_error",))),
        _fmt(_first_present(summary, qiskit_cost, ("final_abs_doublon_error",), ("final_abs_doublon_error",))),
        _fmt(qiskit_cost.get("N2q")),
        _fmt(qiskit_cost.get("D2q")),
        _fmt(qiskit_cost.get("Dc")),
    ]
    headers = [
        "resource",
        "logical",
        "runtime",
        "app",
        "add",
        "prune",
        "drop",
        "dE(T)",
        "dbl(T)",
        "N2q",
        "D2q",
        "Dc",
    ]
    table = ax.table(
        cellText=[values],
        colLabels=headers,
        cellLoc="center",
        loc="center",
        bbox=[0.015, 0.04, 0.97, 0.90],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.4)
    for (row_index, _col_index), cell in table.get_celld().items():
        cell.set_linewidth(0.28)
        cell.set_edgecolor("#9A9A9A")
        if row_index == 0:
            cell.set_facecolor("#F0F0F0")
        else:
            cell.set_facecolor("#FFFFFF")


def _annotate_energy_endpoints(
    ax: Any,
    *,
    times: np.ndarray,
    energy: np.ndarray,
) -> None:
    mask = np.isfinite(times) & np.isfinite(energy)
    if not np.any(mask):
        return
    finite_energy = energy[mask]
    if finite_energy.size == 0:
        return
    start_energy = float(finite_energy[0])
    final_energy = float(finite_energy[-1])
    ax.text(
        0.015,
        0.965,
        f"E_ans(0)={start_energy:.3g}\nE_ans(T)={final_energy:.3g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        color="#202020",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#777777",
            "linewidth": 0.35,
            "alpha": 0.78,
        },
        zorder=9,
    )


def _shared_compact_panel_ylim(
    runs: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str],
) -> tuple[float, float] | None:
    values: list[float] = []
    for run in runs:
        for row in run.get("rows", ()):
            if not isinstance(row, Mapping):
                continue
            for field in fields:
                value = _float_or_none(row.get(field))
                if value is not None:
                    values.append(float(value))
    if not values:
        return None
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    if y_min == y_max:
        pad = max(1.0e-6, abs(y_min) * 0.05)
    else:
        pad = 0.04 * (y_max - y_min)
    return float(y_min - pad), float(y_max + pad)


def _annotate_patch_events(
    ax: Any,
    events: Sequence[Mapping[str, Any]],
    *,
    times: Sequence[float] | np.ndarray | None = None,
    values: Sequence[float] | np.ndarray | None = None,
) -> None:
    """Mark accepted support-patch events with stable compact glyphs."""

    if not events:
        return
    original_ylim = ax.get_ylim()
    for event in events:
        event_time = float(event.get("time") or 0.0)
        kind = str(event.get("patch_kind") or "")
        if kind in {"append", "insert", "exchange"}:
            y_center = _event_curve_value(times, values, event_time)
            if y_center is None:
                ax.plot(
                    [event_time, event_time],
                    [0.84, 0.98],
                    color=APPEND_MARKER_COLOR,
                    linestyle="-",
                    linewidth=1.45,
                    alpha=0.95,
                    transform=ax.get_xaxis_transform(),
                    solid_capstyle="butt",
                    clip_on=False,
                    zorder=7,
                )
            else:
                y_low, y_high = _event_curve_marker_span(ax, float(y_center), original_ylim)
                ax.plot(
                    [event_time, event_time],
                    [y_low, y_high],
                    color=APPEND_MARKER_COLOR,
                    linestyle="-",
                    linewidth=1.55,
                    alpha=0.95,
                    solid_capstyle="butt",
                    clip_on=False,
                    zorder=7,
                )
        if kind in {"delete", "exchange"}:
            y_center = _event_curve_value(times, values, event_time)
            if y_center is None:
                ax.scatter(
                    [event_time],
                    [0.90],
                    marker="D",
                    s=22,
                    color=PRUNE_MARKER_COLOR,
                    edgecolors="white",
                    linewidths=0.35,
                    transform=ax.get_xaxis_transform(),
                    clip_on=False,
                    zorder=7,
                )
            else:
                ax.scatter(
                    [event_time],
                    [float(y_center)],
                    marker="D",
                    s=22,
                    color=PRUNE_MARKER_COLOR,
                    edgecolors="white",
                    linewidths=0.35,
                    clip_on=False,
                    zorder=7,
                )
    ax.set_ylim(original_ylim)


def _event_curve_value(
    times: Sequence[float] | np.ndarray | None,
    values: Sequence[float] | np.ndarray | None,
    event_time: float,
) -> float | None:
    if times is None or values is None:
        return None
    x = np.asarray(times, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return None
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return None
    x_f = x[mask]
    y_f = y[mask]
    exact = np.where(np.isclose(x_f, float(event_time), rtol=0.0, atol=1.0e-12))[0]
    if exact.size:
        return float(y_f[int(exact[0])])
    if float(event_time) <= float(np.min(x_f)) or float(event_time) >= float(np.max(x_f)):
        nearest = int(np.argmin(np.abs(x_f - float(event_time))))
        return float(y_f[nearest])
    order = np.argsort(x_f)
    return float(np.interp(float(event_time), x_f[order], y_f[order]))


def _event_curve_marker_span(
    ax: Any,
    y_center: float,
    ylim: tuple[float, float],
) -> tuple[float, float]:
    y_min, y_max = float(ylim[0]), float(ylim[1])
    if str(ax.get_yscale()) == "log" and y_center > 0.0 and y_min > 0.0 and y_max > 0.0:
        log_span = max(1.0e-12, np.log10(y_max) - np.log10(y_min))
        factor = 10.0 ** (0.04 * log_span)
        return float(y_center / factor), float(y_center * factor)
    span = max(1.0e-12, abs(y_max - y_min))
    half_height = 0.035 * span
    return float(y_center - half_height), float(y_center + half_height)


def _annotate_numerical_stabilization(
    ax: Any,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    windows = _numerical_stabilization_windows(rows)
    if not windows:
        return
    for start, end, _count, _max_depth in windows:
        ax.axvspan(
            float(start),
            float(end),
            color=STABILIZATION_BAND_COLOR,
            alpha=0.12,
            linewidth=0.0,
            zorder=0,
        )
    count = sum(count for _start, _end, count, _max_depth in windows)
    max_depth = max((depth for _start, _end, _count, depth in windows), default=0)
    label = f"stab x{count}" if max_depth <= 0 else f"stab x{count}, d<={max_depth}"
    start, end, _count, _max_depth = max(
        windows,
        key=lambda window: (int(window[2]), float(window[1]) - float(window[0])),
    )
    x_text = 0.5 * (float(start) + float(end))
    ax.text(
        x_text,
        0.965,
        label,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=5.6,
        color="#6A5600",
        clip_on=False,
        zorder=8,
        bbox={
            "boxstyle": "round,pad=0.12",
            "facecolor": "white",
            "edgecolor": STABILIZATION_BAND_COLOR,
            "linewidth": 0.35,
            "alpha": 0.72,
        },
    )


def _numerical_stabilization_windows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[float, float, int, int], ...]:
    row_list = tuple(rows)
    if not row_list:
        return ()
    times = [float(row.get("time") or 0.0) for row in row_list]
    finite_steps = [
        abs(float(right - left))
        for left, right in zip(times[:-1], times[1:])
        if np.isfinite(float(right - left)) and abs(float(right - left)) > 0.0
    ]
    dt_default = min(finite_steps) if finite_steps else 1.0e-6
    active: list[tuple[int, float, float, int]] = []
    for index, row in enumerate(row_list):
        if not _row_has_numerical_stabilization(row):
            continue
        start = times[index]
        if bool(row.get("integration_local_subdivision_applied")) and index + 1 < len(times):
            end = times[index + 1]
        else:
            end = start + dt_default
        depth = _int_or_zero(row.get("integration_local_subdivision_depth"))
        active.append((index, float(start), float(end), depth))
    if not active:
        return ()
    windows: list[tuple[float, float, int, int]] = []
    current_start = active[0][1]
    current_end = active[0][2]
    current_count = 1
    current_depth = active[0][3]
    previous_index = active[0][0]
    for index, start, end, depth in active[1:]:
        if int(index) == int(previous_index) + 1 and float(start) <= float(current_end) + 1.0e-12:
            current_end = max(float(current_end), float(end))
            current_count += 1
            current_depth = max(int(current_depth), int(depth))
        else:
            windows.append((float(current_start), float(current_end), int(current_count), int(current_depth)))
            current_start = float(start)
            current_end = float(end)
            current_count = 1
            current_depth = int(depth)
        previous_index = int(index)
    windows.append((float(current_start), float(current_end), int(current_count), int(current_depth)))
    return tuple(windows)


def _row_has_numerical_stabilization(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("integration_local_subdivision_applied")
        or row.get("solve_repair_applied")
        or row.get("solve_repair_unsupported")
    )


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _event_marker_shape(event: Mapping[str, Any]) -> str:
    kind = str(event.get("patch_kind") or "")
    if kind in {"append", "insert"}:
        return "^"
    if kind == "delete":
        return "D"
    if kind == "exchange":
        return "D"
    return "o"


def _event_marker_color(event: Mapping[str, Any]) -> str:
    kind = str(event.get("patch_kind") or "")
    if kind in {"append", "insert"}:
        return "#146C43"
    if kind == "delete":
        return "#9B1C31"
    if kind == "exchange":
        return "#6B4C9A"
    return "#404040"


def _event_axis_label(event: Mapping[str, Any]) -> str:
    kind = str(event.get("patch_kind") or "")
    if kind in {"append", "insert"}:
        count = _event_appended_count(event)
        return f"append +{count} {_append_event_kind(event, count)}"
    if kind == "delete":
        return f"prune -{_event_deleted_count(event)} term"
    if kind == "exchange":
        return f"exchange -{_event_deleted_count(event)} +{_event_appended_count(event)}"
    return "patch"


def _event_appended_count(event: Mapping[str, Any]) -> int:
    for key in (
        "patch_appended_count",
        "support_patch_appended_count",
        "patch_inserted_count",
        "support_patch_inserted_count",
        "patch_selected_rung_size",
    ):
        value = event.get(key)
        if value not in {None, ""}:
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                pass
    return 1


def _event_deleted_count(event: Mapping[str, Any]) -> int:
    for key in (
        "patch_deleted_count",
        "support_patch_deleted_count",
        "patch_removed_count",
        "support_patch_removed_count",
    ):
        value = event.get(key)
        if value not in {None, ""}:
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                pass
    return 1


def _append_event_kind(event: Mapping[str, Any], count: int) -> str:
    ladder_mode = str(event.get("patch_append_ladder_mode") or "")
    rung_size = event.get("patch_selected_rung_size")
    selection_policy = str(event.get("patch_selection_policy") or "")
    if ladder_mode == "combinatorial" or "append_ladder" in selection_policy:
        if int(count) == 1:
            return "Pauli term"
        return "Pauli batch"
    if int(count) > 1:
        return "macro generator"
    if rung_size not in {None, ""}:
        return "Pauli term"
    return "macro generator"


def _run_title_for_axis(run: Mapping[str, Any]) -> str:
    return f"Run {int(run['run_index'])}: {run['label']}"


def _wrapped_axis_title(label: str, suffix: str, *, width: int = 58, max_lines: int = 3) -> str:
    text = f"{label} - {suffix}"
    lines = textwrap.wrap(
        text,
        width=max(16, int(width)),
        break_long_words=False,
        break_on_hyphens=False,
    )
    if len(lines) <= int(max_lines):
        return "\n".join(lines)
    kept = lines[: max(1, int(max_lines))]
    remainder = " ".join(lines[max(1, int(max_lines)) :])
    kept[-1] = textwrap.shorten(
        f"{kept[-1]} {remainder}",
        width=max(16, int(width)),
        placeholder="...",
    )
    return "\n".join(kept)


def _render_latex(
    *,
    title: str,
    summaries: Sequence[RunSummary],
    compact_panel_pngs: Sequence[Path],
    plot_pngs: Sequence[Path],
    cost_table_jsons: Sequence[Path],
    notes: Sequence[str],
    plot_grid_columns: int,
) -> str:
    lines: list[str] = [
        r"\documentclass[10pt,landscape]{article}",
        r"\usepackage[margin=0.18in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{xcolor}",
        r"\usepackage{enumitem}",
        r"\pagestyle{empty}",
        r"\setlength{\parindent}{0pt}",
        r"\begin{document}",
        rf"% title: {_tex(title)}",
        r"% diagnostic_only: true",
        r"% exact_reference_scope: post_run_reporting_overlay_only",
        r"% naming: exact reference = diagnostic ED/reference curve; fixed APM = support edits disabled; patch-enabled APM = append/prune/swap active.",
    ]
    for note in notes:
        lines.append(rf"% note: {_tex(note)}")
    if summaries:
        lines.append(rf"% event_audit: {_event_sentence(summaries)}")
        lines.append(rf"% threshold_audit: {_threshold_sentence(summaries)}")
    if cost_table_jsons and not compact_panel_pngs:
        for cost_table_json in cost_table_jsons:
            lines.extend(_render_cost_table_latex(cost_table_json))
        if plot_pngs or compact_panel_pngs:
            lines.append(r"\clearpage")
    if plot_pngs:
        width = 0.98 / float(max(1, plot_grid_columns))
        height = "7.65in" if int(plot_grid_columns) == 1 else "3.65in"
        for index, path in enumerate(plot_pngs):
            if index and index % plot_grid_columns == 0:
                lines.append(r"\par\vspace{0.25em}")
            lines.append(
                rf"\begin{{minipage}}[t]{{{width:.3f}\linewidth}}\centering"
                rf"\includegraphics[width=\linewidth,height={height},keepaspectratio]{{{_tex_path(path)}}}"
                r"\end{minipage}\hfill"
            )
        lines.append(r"\par")
    for index, compact_panel_png in enumerate(compact_panel_pngs):
        if index or plot_pngs:
            lines.append(r"\clearpage")
        lines.append(
            rf"\noindent\includegraphics[width=\linewidth,height=8.05in,keepaspectratio]{{{_tex_path(compact_panel_png)}}}"
        )
    lines.extend([r"\end{document}", ""])
    return "\n".join(lines)


def _render_cost_table_latex(path: Path) -> list[str]:
    payload = _load_json_object(path)
    rows_raw = payload.get("rows", ())
    rows = [dict(row) for row in rows_raw if isinstance(row, Mapping)]
    if not rows:
        return []
    compile_defaults = payload.get("compile_defaults", {})
    backend = ""
    if isinstance(compile_defaults, Mapping):
        backend = str(compile_defaults.get("backend_name") or "")
    lines = [
        r"\begin{center}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        rf"\textbf{{Qiskit cost table}}\\[-0.2em]",
    ]
    if backend:
        lines.append(rf"\emph{{backend: {_tex(backend)}}}\\[0.25em]")
    lines.extend(
        [
            r"\begin{tabular}{lrrrrrrrrr}",
            r"Run & logical & runtime & appends & added & $|\Delta E|$ & doublon err. & $N_{2q}$ & $D_{2q}$ & $D_{\rm c}$ \\",
            r"\hline",
        ]
    )
    for row in rows:
        append_count = row.get("accepted_append_count")
        added_count = row.get("accepted_appended_coordinate_count")
        line = " & ".join(
            [
                _tex(row.get("label", "")),
                _fmt(row.get("logical_parameter_count")),
                _fmt(row.get("runtime_parameter_count")),
                _fmt(append_count),
                _fmt(added_count),
                _fmt(row.get("final_abs_energy_error")),
                _fmt(row.get("final_abs_doublon_error")),
                _fmt(row.get("N2q")),
                _fmt(row.get("D2q")),
                _fmt(row.get("Dc")),
            ]
        )
        lines.append(line + r" \\")
    lines.extend(
        [
            r"\end{tabular}",
            r"\\[-0.1em]\emph{Qiskit columns use the same final-state ansatz circuit convention as the diagnostic sidecar; exact/reference trajectories are reporting-only.}",
            r"\end{center}",
        ]
    )
    return lines


def _build_latex_pdf(*, output_tex: Path, output_pdf: Path) -> None:
    tectonic = shutil.which("tectonic")
    if tectonic is None:
        raise RuntimeError("tectonic is required to build the AP progress PDF from LaTeX.")
    subprocess.run(
        [tectonic, "--keep-logs", "--reruns", "2", output_tex.name],
        cwd=str(output_tex.parent),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    built_pdf = output_tex.with_suffix(".pdf")
    if built_pdf != output_pdf:
        built_pdf.replace(output_pdf)


def _summarize_run(path: Path, *, run_index: int) -> RunSummary:
    payload = _load_json_object(path)
    rows = _plot_rows(payload, path)
    summary = dict(payload.get("summary", {}) or {})
    schema = str(payload.get("schema") or "")
    events = tuple(_patch_events(rows))
    max_error_row = _max_row(rows, "abs_energy_error")
    return RunSummary(
        run_index=int(run_index),
        path=str(path),
        label=_run_label(path, payload),
        schema=schema,
        route_label=_route_label(schema),
        drive_enabled=bool(summary.get("drive_enabled")),
        point_count=int(summary.get("point_count") or len(rows)),
        integrator_method=str(summary.get("integrator_method") or ""),
        initial_abs_energy_error=_float_or_none(rows[0].get("abs_energy_error") if rows else None),
        final_abs_energy_error=_float_or_none(summary.get("final_abs_energy_error")),
        max_abs_energy_error=_float_or_none(summary.get("max_abs_energy_error")),
        max_abs_energy_error_time=_float_or_none(max_error_row.get("time") if max_error_row else None),
        seed_final_abs_energy_error=_float_or_none(summary.get("seed_final_abs_energy_error")),
        seed_max_abs_energy_error=_float_or_none(summary.get("seed_max_abs_energy_error")),
        final_abs_doublon_error=_float_or_none(summary.get("final_abs_doublon_error")),
        max_abs_doublon_error=_float_or_none(summary.get("max_abs_doublon_error")),
        seed_final_abs_doublon_error=_float_or_none(summary.get("seed_final_abs_doublon_error")),
        seed_max_abs_doublon_error=_float_or_none(summary.get("seed_max_abs_doublon_error")),
        final_site_occupations_abs_error_max=_float_or_none(
            summary.get("final_site_occupations_abs_error_max")
        ),
        max_abs_site_occupations_error=_float_or_none(
            summary.get("max_abs_site_occupations_error")
        ),
        seed_final_site_occupations_abs_error_max=_float_or_none(
            summary.get("seed_final_site_occupations_abs_error_max")
        ),
        seed_max_abs_site_occupations_error=_float_or_none(
            summary.get("seed_max_abs_site_occupations_error")
        ),
        accepted_patch_count=int(summary.get("accepted_patch_count") or len(events)),
        prune_patch_smoothness_deferred_count=int(
            summary.get("prune_patch_smoothness_deferred_count") or 0
        ),
        prune_patch_smoothness_passed_count=int(
            summary.get("prune_patch_smoothness_passed_count") or 0
        ),
        prune_patch_smoothness_retry_count=int(
            summary.get("prune_patch_smoothness_retry_count") or 0
        ),
        prune_patch_smoothness_accepted_after_retry_count=int(
            summary.get("prune_patch_smoothness_accepted_after_retry_count") or 0
        ),
        max_prune_patch_smoothness_eta=_float_or_none(
            summary.get("max_prune_patch_smoothness_eta")
        ),
        max_prune_patch_smoothness_severity=_float_or_none(
            summary.get("max_prune_patch_smoothness_severity")
        ),
        events=events,
        threshold_hits=tuple(_threshold_hits(rows)),
    )


def _load_run(
    path: Path,
    *,
    run_index: int,
    qiskit_cost_lookup: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = _load_json_object(path)
    summary = dict(payload.get("summary", {}) or {})
    qiskit_cost = _qiskit_cost_for_run(
        path=path,
        payload=payload,
        summary=summary,
        lookup=qiskit_cost_lookup or {},
    )
    return {
        "run_index": int(run_index),
        "path": str(path),
        "label": _run_label(path, payload),
        "rows": _plot_rows(payload, path),
        "summary": summary,
        "qiskit_cost": qiskit_cost,
    }


def _load_json_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Trajectory JSON must be an object: {path}")
    return payload


def _load_qiskit_cost_lookup(paths: Sequence[Path]) -> dict[str, Any]:
    lookup: dict[str, Any] = {"__rows__": []}
    for path in paths:
        payload = _load_json_object(path)
        rows_raw = payload.get("rows", ())
        if not isinstance(rows_raw, Sequence) or isinstance(rows_raw, (str, bytes)):
            continue
        for row_raw in rows_raw:
            if not isinstance(row_raw, Mapping):
                continue
            row = dict(row_raw)
            row["_qiskit_cost_table_json"] = str(path)
            lookup["__rows__"].append(row)
            for key in _path_lookup_keys(row.get("trajectory_json")):
                lookup[key] = row
    return lookup


def _qiskit_cost_for_run(
    *,
    path: Path,
    payload: Mapping[str, Any],
    summary: Mapping[str, Any],
    lookup: Mapping[str, Any],
) -> dict[str, Any]:
    for key in _path_lookup_keys(path):
        row = lookup.get(key)
        if isinstance(row, Mapping):
            return {**dict(row), "_qiskit_cost_match": "path"}
    raw_source = None
    report_slim_source = payload.get("report_slim_source")
    if isinstance(report_slim_source, Mapping):
        raw_source = report_slim_source.get("raw_trajectory_json")
    for key in _path_lookup_keys(raw_source):
        row = lookup.get(key)
        if isinstance(row, Mapping):
            return {**dict(row), "_qiskit_cost_match": "raw_path"}
    matched = _match_qiskit_cost_by_summary(summary, lookup.get("__rows__", ()))
    if matched is None:
        return {}
    return {**dict(matched), "_qiskit_cost_match": "summary_exact"}


def _match_qiskit_cost_by_summary(
    summary: Mapping[str, Any],
    rows: Any,
) -> Mapping[str, Any] | None:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return None
    for row_raw in rows:
        if not isinstance(row_raw, Mapping):
            continue
        if _summary_matches_cost_row(summary, row_raw):
            return row_raw
    return None


def _summary_matches_cost_row(summary: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    count_pairs = (
        ("logical_parameter_count_final", "logical_parameter_count"),
        ("runtime_parameter_count_final", "runtime_parameter_count"),
        ("accepted_append_count", "accepted_append_count"),
        ("accepted_appended_coordinate_count", "accepted_appended_coordinate_count"),
    )
    for summary_key, row_key in count_pairs:
        left = summary.get(summary_key)
        right = row.get(row_key)
        if left in {None, ""} or right in {None, ""}:
            return False
        try:
            if int(left) != int(right):
                return False
        except (TypeError, ValueError):
            return False
    float_pairs = (
        ("final_abs_energy_error", "final_abs_energy_error"),
        ("final_abs_doublon_error", "final_abs_doublon_error"),
    )
    for summary_key, row_key in float_pairs:
        left = _float_or_none(summary.get(summary_key))
        right = _float_or_none(row.get(row_key))
        if left is None or right is None:
            return False
        if not np.isclose(float(left), float(right), rtol=1.0e-12, atol=1.0e-14):
            return False
    return True


def _path_lookup_keys(value: Any) -> tuple[str, ...]:
    if value in {None, ""}:
        return ()
    text = str(value)
    path = Path(text)
    keys = {text, path.as_posix()}
    try:
        keys.add(str(path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()))
    except Exception:
        pass
    return tuple(key for key in keys if key)


def _plot_rows(payload: Mapping[str, Any], path: Path) -> list[dict[str, Any]]:
    rows = payload.get("plot_rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError(f"Trajectory JSON is missing sequence `plot_rows`: {path}")
    return [dict(row) for row in rows]


def _patch_events(rows: Sequence[Mapping[str, Any]]) -> list[PatchEvent]:
    events: list[PatchEvent] = []
    for row in rows:
        if not bool(row.get("patch_accepted")):
            continue
        events.append(
            PatchEvent(
                index=int(row.get("index") or 0),
                time=float(row.get("time") or 0.0),
                patch_kind=str(row.get("patch_kind") or ""),
                selected_label=str(row.get("patch_selected_label") or ""),
                reason=str(row.get("patch_reason") or row.get("patch_batch_reason") or ""),
                rank_score=_float_or_none(row.get("patch_rank_score")),
                abs_energy_error=_float_or_none(row.get("abs_energy_error")),
                theta_dot_l2=_float_or_none(row.get("theta_dot_l2")),
                residual_ratio=_float_or_none(row.get("mclachlan_residual_ratio")),
            )
        )
    return events


def _threshold_hits(rows: Sequence[Mapping[str, Any]]) -> list[ThresholdHit]:
    hits: list[ThresholdHit] = []
    for threshold in DEFAULT_THRESHOLDS:
        hit_row = next(
            (
                row
                for row in rows
                if row.get("abs_energy_error") is not None
                and abs(float(row["abs_energy_error"])) >= threshold
            ),
            None,
        )
        if hit_row is None:
            continue
        hits.append(
            ThresholdHit(
                threshold=threshold,
                index=int(hit_row.get("index") or 0),
                time=float(hit_row.get("time") or 0.0),
                abs_energy_error=float(hit_row["abs_energy_error"]),
            )
        )
    return hits


def _event_sentence(summaries: Sequence[RunSummary]) -> str:
    parts: list[str] = []
    for summary in summaries:
        label = f"Run {summary.run_index}: {summary.label}"
        if not summary.events:
            parts.append(f"{label}: no accepted support patch")
            continue
        event_bits = []
        for event in summary.events:
            kind = (
                "append"
                if event.patch_kind in {"append", "insert"}
                else event.patch_kind or "patch"
            )
            event_bits.append(
                f"{kind} {event.selected_label} at t={_fmt(event.time)} "
                f"(score={_fmt(event.rank_score)}, abs dE={_fmt(event.abs_energy_error)})"
            )
        parts.append(f"{label}: " + "; ".join(event_bits))
    return _tex("; ".join(parts) + ".")


def _threshold_sentence(summaries: Sequence[RunSummary]) -> str:
    parts: list[str] = []
    for summary in summaries:
        label = f"Run {summary.run_index}: {summary.label}"
        if not summary.threshold_hits:
            parts.append(f"{label}: no listed thresholds crossed")
            continue
        compact = ", ".join(
            f"{_fmt(hit.threshold)} at t={_fmt(hit.time)}" for hit in summary.threshold_hits[:4]
        )
        parts.append(f"{label}: {compact}")
    return _tex("; ".join(parts) + ".")


def _run_label(path: Path, payload: Mapping[str, Any]) -> str:
    summary_label_overrides = {
        "powellA1_weakweak_k10_driveA0p8_batch5_pauli_append_costweighted_res1em3_no_solve_repair_t3_n601_dualref": (
            "k=10 cost-weighted <=5, no stabilization"
        ),
        "weak_weak_snake_exchange_smoke_t0p02_n5": (
            "exchange-family smoke, prune score only"
        ),
        "run115_exchange_macro_scout_parentcap20_prunehistfix_k10_A0p8_t3_n601_dualref_report_slim": (
            "macro scout cap=20, prune history preserved"
        ),
        "run115_macro_scout_parent16_exchange_parallel4_k10_A0p8_t3_n601_dualref_report_slim": (
            "macro scout cap=16, exchange fail-open off"
        ),
        "run117_cached_exchange_nomacroscout_k10_A0p8_t3_n601_dualref_report_slim": (
            "cached exchange rerun, macro scout off"
        ),
        "powellA1_weakweak_k10_driveA0p8_batch5_pauli_append_costweighted_res1em3_t3_n601_dualref": (
            "k=10 cost-weighted max batch <= 5"
        ),
        "powellA1_weakweak_k10_driveA0p8_batch5_pauli_append_costweighted_prune_score_res1em3_t3_n601_dualref_report_slim": (
            "prune scoring only, no commits"
        ),
        "powellA1_weakweak_k10_driveA0p8_batch5_pauli_append_costweighted_prune_commit_shadowoff_uncapped_res1em3_t3_n601_dualref_report_slim": (
            "prune commit, shadow off"
        ),
        "powellA1_weakweak_k10_driveA0p8_batch5_pauli_append_costweighted_prune_conditioned_commit_shadowoff_res1em3_t3_n601_dualref_report_slim": (
            "condition-aware prune commit"
        ),
        "powellA1_weakweak_k10_driveA0p8_batch5_pauli_append_costweighted_prune_conditioned_persist5_hist5_softcond_res1em3_t3_n601_dualref_report_slim": (
            "prune persistence=5, softer conditioning"
        ),
        "powellA1_weakweak_k10_driveA0p8_batch5_pauli_append_costweighted_prune_conditioned_persist3_hist3_midcond_res1em3_t3_n601_dualref_report_slim": (
            "prune persistence=3, mid conditioning"
        ),
        "powellA1_weakweak_k10_driveA0p8_batch5_pauli_append_costweighted_prune_conditioned_atompersist3_frac1_hist3_midcond_res1em3_t3_n601_dualref_report_slim": (
            "atom-history persistence q=1.0"
        ),
        "a2_append_prune_commit_batchmax10_atompersist3_shadowoff_t3_n601_report_slim": (
            "A2 prune commit, atom history"
        ),
        "a2_run110_prune_smoothness_deferred_t3_n601_report_slim": (
            "Run 110 replay, prune smoothness guard"
        ),
        "a2_run110_prune_smoothness_eta1em3_t3_n601_report_slim": (
            "prune smoothness eta<=1e-3"
        ),
        "a2_append_prune_score_batchmax10_atompersist3_t3_n601_report_slim": (
            "A2 prune scoring only"
        ),
        "a2_append_only_batchmax10_res1em3_t3_n601_report_slim": (
            "A2 append only, max batch <= 10"
        ),
        "a2_fixed_no_tested_append_t3_n601_report_slim": (
            "A2 fixed support"
        ),
        "powellA1_weakweak_k10_driveA0p8_fixed_no_tested_append_costweighted_companion_t3_n601_dualref": (
            "k=10 fixed companion"
        ),
        "powellA1_weakweak_k15_driveA0p8_batch3_pauli_append_augsolveconfirm_t3_n601_dualref": (
            "k15 A=0.8 max batch <= 3, aug-solve confirm"
        ),
        "powellA1_weakweak_k15_driveA0p8_batch5_multiappend_pauli_append_augsolveconfirm_t3_n601_dualref": (
            "k15 A=0.8 max batch <= 5, aug-solve confirm"
        ),
        "powellA1_weakweak_k10_driveA0p8_batch5_pauli_append_augsolveconfirm_res1em3_t3_n601_dualref": (
            "k=10 max batch <= 5, rho>=1e-3"
        ),
        "powellA1_weakweak_k10_driveA0p8_fixed_no_tested_append_t3_n601_dualref": (
            "k=10 fixed support"
        ),
        "powellA1_weakweak_k15_driveA0p8_batch3_pauli_append_t3_n601_dualref": (
            "k15 A=0.8 max batch <= 3 Pauli, dual refs"
        ),
        "powellA1_weakweak_k3_driveA0p8_batch3_pauli_append_sector_guarded_t3_n601_dualref": (
            "k3 A=0.8 max batch <= 3, sector guard"
        ),
        "powellA1_weakweak_k3_driveA0p8_batch3_pauli_append_guarded_t3_n601_dualref": (
            "k3 A=0.8 max batch <= 3, legal guard"
        ),
        "powellA1_weakweak_k1_driveA0p8_batch3_pauli_append_t3_n601_dualref": (
            "k1 A=0.8 max batch <= 3 Pauli, dual refs"
        ),
        "powellA1_weakweak_k3_driveA0p8_batch3_pauli_append_t3_n601_dualref": (
            "k3 A=0.8 max batch <= 3 Pauli, dual refs"
        ),
        "powellA1_weakweak_k5_driveA0p8_batch3_pauli_append_t3_n601_dualref": (
            "k5 A=0.8 max batch <= 3 Pauli, dual refs"
        ),
    }
    if path.stem in summary_label_overrides:
        return summary_label_overrides[path.stem]
    summary = payload.get("summary", {}) or {}
    if isinstance(summary, Mapping):
        seed = summary.get("seed_kind") or summary.get("label")
        if seed not in {None, ""}:
            return str(seed)
    replacements = {
        "snake_nodrive_t5_n501_fixed": "snake no-drive fixed APM",
        "snake_nodrive_t5_n501_apm_append_euler_default_ridge": (
            "snake no-drive APM append, selected-only pool, Euler ridge=1e-7"
        ),
        "snake_nodrive_t5_n501_apm_append_complete_pool_euler_default_ridge": (
            "snake no-drive APM append, complete pool, Euler ridge=1e-7"
        ),
        "snake_nodrive_t5_n501_apm_append_prune_complete_pool_euler_default_ridge_pruneloss1e-2": (
            "snake no-drive APM append-prune, complete pool, Euler ridge=1e-7, prune loss<=1e-2"
        ),
        "snake_nodrive_t5_n501": "snake no-drive patch APM",
        "snake_driveA0p6_t5_n501_apm_append_drive_aligned_euler_default_ridge": (
            "snake drive A=0.6 APM append, selected-only pool + drive generator, Euler ridge=1e-7"
        ),
        "snake_driveA0p6_t5_n501_apm_append_complete_pool_drive_aligned_euler_default_ridge": (
            "snake drive A=0.6 APM append, complete pool + drive generator, Euler ridge=1e-7"
        ),
        "snake_driveA0p6_t5_n501_apm_append_prune_complete_pool_drive_aligned_euler_default_ridge_pruneloss1e-2": (
            "snake drive A=0.6 APM append-prune, complete pool + drive generator, Euler ridge=1e-7, prune loss<=1e-2"
        ),
        "snake_driveA0p6_t5_n501_fixed_drive_aligned_euler_ridge1e-6": (
            "snake drive A=0.6 fixed APM + drive generator, Euler ridge=1e-6"
        ),
        "snake_driveA0p6_t5_n501_fixed_drive_aligned_euler_ridge1e-8": (
            "snake drive A=0.6 fixed APM + drive generator, Euler ridge=1e-8"
        ),
        "snake_driveA0p6_t5_n501_fixed_drive_aligned_rk4_ridge1e-7": (
            "snake drive A=0.6 fixed APM + drive generator, RK4 ridge=1e-7"
        ),
        "snake_driveA0p6_t5_n501_fixed_drive_aligned_rk4_ridge1e-8": (
            "snake drive A=0.6 fixed APM + drive generator, RK4 ridge=1e-8"
        ),
        "snake_driveA0p6_t5_n501_fixed_drive_aligned_rk4_ridge0": (
            "snake drive A=0.6 fixed APM + drive generator, RK4 ridge=0"
        ),
        "snake_driveA0p6_t5_n501_fixed_drive_aligned": (
            "snake drive A=0.6 fixed APM + drive generator, Euler ridge=0"
        ),
        "snake_driveA0p6_t5_n501_fixed": "snake drive A=0.6 fixed APM",
        "snake_driveA0p6_t5_n501": "snake drive A=0.6 patch APM",
        "seed1_snake_driveA0p6_t0p05_n6_batch_combinatorial_pool_diag": (
            "short-grid max batch <= 3 Pauli"
        ),
        "seed1_snake_driveA0p6_t0p05_n6_support_atom_singleton_pool_diag": (
            "short-grid singleton Pauli insert"
        ),
        "seed1_snake_driveA0p6_t0p05_n6_legacy_singleton_pool_diag": (
            "short-grid legacy macro singleton"
        ),
        "seed1_snake_driveA0p6_t0p05_n6_fixed_pool_diag": (
            "short-grid fixed support"
        ),
        "powellA1_seed1_k15_fixed_no_tested_append_t3_n601_diag": (
            "Seed1 k=15 fixed support"
        ),
        "powellA1_seed1_k15_singleton_pauli_append_t3_n601_diag": (
            "Seed1 k=15 singleton Pauli insert"
        ),
        "powellA1_seed1_k15_batch3_pauli_append_t3_n601_diag": (
            "Seed1 k=15 max batch <= 3 Pauli"
        ),
        "powellA1_seed1_k15_fixed_no_tested_append_t3_n601_obs": (
            "Seed1 k=15 fixed support + observables"
        ),
        "powellA1_seed1_k15_singleton_pauli_append_t3_n601_obs": (
            "Seed1 k=15 singleton Pauli insert + observables"
        ),
        "powellA1_seed1_k15_batch3_pauli_append_t3_n601_obs": (
            "Seed1 k=15 max batch <= 3 Pauli + observables"
        ),
        "powellA1_weakweak_k15_driveA0p8_fixed_no_tested_append_t3_n601_dualref": (
            "k=15 A=0.8 fixed"
        ),
        "powellA1_weakweak_k15_driveA0p8_singleton_pauli_append_t3_n601_dualref": (
            "k=15 A=0.8 singleton"
        ),
        "powellA1_weakweak_k15_driveA0p8_batch3_pauli_append_t3_n601_dualref": (
            "k=15 A=0.8 max batch <= 3"
        ),
        "powellA1_weakweak_k15_driveA0p8_batch3_pauli_append_augsolveconfirm_t3_n601_dualref": (
            "k=15 A=0.8 max batch <= 3, aug-solve confirm"
        ),
        "powellA1_weakweak_k15_driveA0p8_batch5_multiappend_pauli_append_augsolveconfirm_t3_n601_dualref": (
            "k=15 A=0.8 max batch <= 5, aug-solve confirm"
        ),
        "powellA1_weakweak_k15_driveA0p8_batch5_multiappend_pauli_append_schurguard_sector_guarded_t3_n601_dualref": (
            "k=15 A=0.8 max batch <= 5, multiappend, Schur guard"
        ),
        "powellA1_seed2_k5_fixed_no_tested_append_t3_n601_diag": (
            "Seed2 k=5 fixed support"
        ),
        "powellA1_seed2_k5_singleton_pauli_append_t3_n601_diag": (
            "Seed2 k=5 singleton Pauli insert"
        ),
        "powellA1_seed2_k5_batch3_pauli_append_t3_n601_diag": (
            "Seed2 k=5 max batch <= 3 Pauli"
        ),
        "powellA1_seed2_k5_fixed_no_tested_append_t3_n601_obs": (
            "Seed2 k=5 fixed support + observables"
        ),
        "powellA1_seed2_k5_singleton_pauli_append_t3_n601_obs": (
            "Seed2 k=5 singleton Pauli insert + observables"
        ),
        "powellA1_seed2_k5_batch3_pauli_append_t3_n601_obs": (
            "Seed2 k=5 max batch <= 3 Pauli + observables"
        ),
        "powellA1_weakweak_k1_driveA0p8_fixed_no_tested_append_sector_guarded_t3_n601_dualref": (
            "k=1 Pauli fixed, sector guard"
        ),
        "powellA1_weakweak_k1_driveA0p8_singleton_pauli_append_sector_guarded_t3_n601_dualref": (
            "k=1 Pauli singleton, sector guard"
        ),
        "powellA1_weakweak_k1_driveA0p8_batch3_pauli_append_sector_guarded_t3_n601_dualref": (
            "k=1 Pauli max batch <= 3, sector guard"
        ),
        "powellA1_weakweak_k1_driveA0p8_singleton_pauli_append_schurguard_sector_guarded_t3_n601_dualref": (
            "k=1 singleton, Schur guard"
        ),
        "powellA1_weakweak_k1_driveA0p8_batch3_pauli_append_schurguard_sector_guarded_t3_n601_dualref": (
            "k=1 max batch <= 3, Schur guard"
        ),
        "powellA1_weakweak_k1_driveA0p8_batch10_multiappend_pauli_append_schurguard_sector_guarded_t3_n601_dualref": (
            "k=1 max batch <= 10, multiappend, Schur guard"
        ),
        "powellA1_weakweak_k1_driveA0p8_fixed_macro_sector_guarded_t3_n601_dualref": (
            "k=1 macro fixed, sector guard"
        ),
        "powellA1_weakweak_k1_driveA0p8_singleton_macro_append_sector_guarded_t3_n601_dualref": (
            "k=1 macro singleton, sector guard"
        ),
        "powellA1_weakweak_k1_driveA0p8_batch3_macro_append_sector_guarded_t3_n601_dualref": (
            "k=1 macro max batch <= 3, sector guard"
        ),
        "powellA1_weakweak_k2_driveA0p8_fixed_no_tested_append_sector_guarded_t3_n601_dualref": (
            "k=2 Pauli fixed, sector guard"
        ),
        "powellA1_weakweak_k2_driveA0p8_singleton_pauli_append_sector_guarded_t3_n601_dualref": (
            "k=2 Pauli singleton, sector guard"
        ),
        "powellA1_weakweak_k2_driveA0p8_batch3_pauli_append_sector_guarded_t3_n601_dualref": (
            "k=2 Pauli max batch <= 3, sector guard"
        ),
        "powellA1_weakweak_k2_driveA0p8_singleton_pauli_append_schurguard_sector_guarded_t3_n601_dualref": (
            "k=2 singleton, Schur guard"
        ),
        "powellA1_weakweak_k2_driveA0p8_batch3_pauli_append_schurguard_sector_guarded_t3_n601_dualref": (
            "k=2 max batch <= 3, Schur guard"
        ),
        "powellA1_weakweak_k2_driveA0p8_batch10_multiappend_pauli_append_schurguard_sector_guarded_t3_n601_dualref": (
            "k=2 max batch <= 10, multiappend, Schur guard"
        ),
        "powellA1_weakweak_k2_driveA0p8_fixed_macro_sector_guarded_t3_n601_dualref": (
            "k=2 macro fixed, sector guard"
        ),
        "powellA1_weakweak_k2_driveA0p8_singleton_macro_append_sector_guarded_t3_n601_dualref": (
            "k=2 macro singleton, sector guard"
        ),
        "powellA1_weakweak_k2_driveA0p8_batch3_macro_append_sector_guarded_t3_n601_dualref": (
            "k=2 macro max batch <= 3, sector guard"
        ),
        "powellA1_weakweak_k3_driveA0p8_fixed_no_tested_append_sector_guarded_t3_n601_dualref": (
            "k=3 sector guard, fixed"
        ),
        "powellA1_weakweak_k3_driveA0p8_singleton_pauli_append_sector_guarded_t3_n601_dualref": (
            "k=3 sector guard, singleton"
        ),
        "powellA1_weakweak_k3_driveA0p8_batch3_pauli_append_sector_guarded_t3_n601_dualref": (
            "k=3 sector guard, max batch <= 3"
        ),
        "powellA1_weakweak_k3_driveA0p8_fixed_no_tested_append_guarded_t3_n601_dualref": (
            "k=3 dropped-parent guard, fixed"
        ),
        "powellA1_weakweak_k3_driveA0p8_singleton_pauli_append_guarded_t3_n601_dualref": (
            "k=3 dropped-parent guard, singleton"
        ),
        "powellA1_weakweak_k3_driveA0p8_batch3_pauli_append_guarded_t3_n601_dualref": (
            "k=3 dropped-parent guard, max batch <= 3"
        ),
        "seed1_snake_driveA0p6_t3_n301_batch_combinatorial_pool_diag": (
            "max batch <= 3 Pauli, dt=0.01"
        ),
        "seed1_snake_driveA0p6_t3_n601_batch_combinatorial_pool_diag": (
            "max batch <= 3 Pauli, dt=0.005"
        ),
        "seed1_snake_driveA0p6_t3_n601_batch_combinatorial_pool_minappend0p005_euler_diag": (
            "Euler ridge=1e-7 baseline"
        ),
        "seed1_snake_driveA0p6_t3_n601_batch_combinatorial_pool_minappend0p005_euler_current_no_repair_baseline_diag": (
            "current-code no-repair baseline"
        ),
        "seed1_snake_driveA0p6_t3_n601_batch_combinatorial_pool_minappend0p005_euler_paperii_state_repair_rho1_diag": (
            "Paper-II state repair, no forced subdivision"
        ),
        "seed1_snake_driveA0p6_t3_n601_batch_combinatorial_pool_minappend0p005_euler_paperii_state_repair_rho1_subdivide_diag": (
            "Paper-II state repair + kink subdivision"
        ),
        "seed1_snake_driveA0p6_t3_n601_batch_combinatorial_pool_minappend0p005_euler_ridge1e-8_current_no_repair_stress_diag": (
            "stress ridge=1e-8 no repair"
        ),
        "seed1_snake_driveA0p6_t3_n601_batch_combinatorial_pool_minappend0p005_euler_ridge1e-8_paperii_state_repair_stress_diag": (
            "stress ridge=1e-8 + state repair"
        ),
        "seed1_snake_driveA0p6_t3_n601_batch_combinatorial_pool_minappend0p005_euler_ridge1e-8_paperii_state_repair_severity_scaled_diag": (
            "stress ridge=1e-8 + severity-scaled repair"
        ),
        "seed1_snake_driveA0p6_t3_n601_batch_combinatorial_pool_minappend0p005_euler_ridge1e-8_paperii_state_repair_severity_state_only_diag": (
            "rho-free subdivision severity"
        ),
        "seed1_snake_driveA0p6_t3_n601_batch_combinatorial_pool_minappend0p005_euler_ridge1e-8_paperii_scheduler_diag": (
            "Paper-II scheduler"
        ),
        "seed1_snake_driveA0p6_t3_n601_batch_combinatorial_pool_minappend0p005_euler_baseline_no_repair_nonabort_diag": (
            "passive no-repair baseline"
        ),
        "seed1_snake_driveA0p6_t3_n601_batch_combinatorial_pool_minappend0p005_euler_solve_repair_nonabort_cap1p65e6_diag": (
            "repair cap=1.65e6, non-aborting"
        ),
        "seed1_snake_driveA0p6_t3_n601_batch_combinatorial_pool_minappend0p005_euler_solve_repair_state_rho5e4_cond1e12_diag": (
            "rho_num cap=5e-4, kappa off"
        ),
        "seed1_snake_driveA0p6_t3_n601_batch_combinatorial_pool_minappend0p005_euler_solve_repair_state_rho1e4_cond1e12_diag": (
            "rho_num cap=1e-4, kappa off"
        ),
        "seed1_snake_driveA0p6_t3_n601_batch_combinatorial_pool_minappend0p005_euler_solve_repair_state_kappa1p705e7_rho1_diag": (
            "kappa cap=1.705e7, rho off"
        ),
        "seed1_snake_driveA0p6_t3_n601_batch_combinatorial_pool_minappend0p005_euler_solve_repair_v2_cap1p75e6_rho0p015_diag": (
            "repair cap=1.75e6, rho<=0.015 (global)"
        ),
        "seed1_snake_driveA0p6_t3_n601_batch_combinatorial_pool_minappend0p005_rk4_diag": (
            "RK4 ridge=1e-7"
        ),
        "seed1_snake_driveA0p6_t3_n601_batch_combinatorial_pool_minappend0p005_ridge1e-6_euler_diag": (
            "Euler ridge=1e-6"
        ),
        "seed1_snake_driveA0p6_t3_n601_batch_combinatorial_pool_minappend0p005_ridge1e-6_rk4_diag": (
            "RK4 ridge=1e-6"
        ),
        "seed1_snake_driveA0p6_t3_n601_batch_combinatorial_pool_minappend0p005_ridge1e-6_rk4_solve_repair_diag": (
            "RK4 ridge=1e-6 + solve repair"
        ),
        "seed1_snake_driveA0p6_t3_n601_batch_combinatorial_pool_minappend0p005_ridge1e-6_rk4_solve_repair_cap1p65e6_diag": (
            "RK4 ridge=1e-6 + repair cap=1.65e6"
        ),
        "seed1_snake_driveA0p6_t3_n601_batch_combinatorial_pool_minappend0p005_euler_solve_repair_tdot50_diag": (
            "solve repair theta-dot cap=50"
        ),
        "seed1_snake_driveA0p6_t3_n301_support_atom_singleton_pool_diag": (
            "singleton Pauli insert"
        ),
        "seed1_snake_driveA0p6_t3_n301_legacy_singleton_pool_diag": (
            "legacy macro singleton"
        ),
        "seed1_snake_driveA0p6_t3_n301_fixed_pool_diag": (
            "fixed support"
        ),
    }
    return replacements.get(path.stem, path.stem.replace("_", " "))


def _route_label(schema: str) -> str:
    if "fixed_from" in schema or "fixed_trajectory" in schema:
        return "fixed APM"
    if "append_from" in schema or "append_trajectory" in schema:
        return "patch-enabled APM"
    return schema or "unknown"


def _max_row(rows: Sequence[Mapping[str, Any]], key: str) -> Mapping[str, Any] | None:
    candidates = [row for row in rows if row.get(key) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row[key]))


def _existing_plot_paths(paths: Sequence[str | Path]) -> tuple[Path, ...]:
    return tuple(Path(path) for path in paths if Path(path).is_file())


def _sort_plot_paths(paths: Sequence[Path]) -> list[Path]:
    def priority(path: Path) -> tuple[int, str]:
        stem = path.stem
        if "fixed_vs_append" in stem:
            return (0, stem)
        if "fixed" in stem:
            return (1, stem)
        if "nodrive" in stem:
            return (2, stem)
        if "drive" in stem:
            return (3, stem)
        return (9, stem)

    return sorted(paths, key=priority)


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "-"
    value_f = float(value)
    if value_f == 0:
        return "0"
    if abs(value_f) >= 1000 or abs(value_f) < 0.001:
        return f"{value_f:.3e}"
    return f"{value_f:.6g}"


def _first_present(
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
    primary_keys: Sequence[str],
    secondary_keys: Sequence[str],
) -> Any:
    for key in primary_keys:
        value = primary.get(key)
        if value not in {None, ""}:
            return value
    for key in secondary_keys:
        value = secondary.get(key)
        if value not in {None, ""}:
            return value
    return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _tex(text: Any) -> str:
    result = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


def _tex_path(path: Path) -> str:
    return _tex(str(path.resolve()))


def _summary_to_json(summary: RunSummary) -> dict[str, Any]:
    return {
        "run_index": summary.run_index,
        "path": summary.path,
        "label": summary.label,
        "schema": summary.schema,
        "route_label": summary.route_label,
        "drive_enabled": summary.drive_enabled,
        "point_count": summary.point_count,
        "integrator_method": summary.integrator_method,
        "initial_abs_energy_error": summary.initial_abs_energy_error,
        "final_abs_energy_error": summary.final_abs_energy_error,
        "max_abs_energy_error": summary.max_abs_energy_error,
        "max_abs_energy_error_time": summary.max_abs_energy_error_time,
        "seed_final_abs_energy_error": summary.seed_final_abs_energy_error,
        "seed_max_abs_energy_error": summary.seed_max_abs_energy_error,
        "final_abs_doublon_error": summary.final_abs_doublon_error,
        "max_abs_doublon_error": summary.max_abs_doublon_error,
        "seed_final_abs_doublon_error": summary.seed_final_abs_doublon_error,
        "seed_max_abs_doublon_error": summary.seed_max_abs_doublon_error,
        "final_site_occupations_abs_error_max": summary.final_site_occupations_abs_error_max,
        "max_abs_site_occupations_error": summary.max_abs_site_occupations_error,
        "seed_final_site_occupations_abs_error_max": summary.seed_final_site_occupations_abs_error_max,
        "seed_max_abs_site_occupations_error": summary.seed_max_abs_site_occupations_error,
        "accepted_patch_count": summary.accepted_patch_count,
        "prune_patch_smoothness_deferred_count": summary.prune_patch_smoothness_deferred_count,
        "prune_patch_smoothness_passed_count": summary.prune_patch_smoothness_passed_count,
        "prune_patch_smoothness_retry_count": summary.prune_patch_smoothness_retry_count,
        "prune_patch_smoothness_accepted_after_retry_count": summary.prune_patch_smoothness_accepted_after_retry_count,
        "max_prune_patch_smoothness_eta": summary.max_prune_patch_smoothness_eta,
        "max_prune_patch_smoothness_severity": summary.max_prune_patch_smoothness_severity,
        "events": [event.__dict__ for event in summary.events],
        "threshold_hits": [hit.__dict__ for hit in summary.threshold_hits],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a LaTeX APM diagnostic progress PDF.")
    parser.add_argument("--trajectory-json", action="append", default=[])
    parser.add_argument("--plot-png", action="append", default=[])
    parser.add_argument("--plot-dir", action="append", default=[])
    parser.add_argument("--cost-table-json", action="append", default=[])
    parser.add_argument("--output-pdf", required=True)
    parser.add_argument("--output-manifest", default=None)
    parser.add_argument("--title", default="APM Diagnostic Progress Report")
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--plot-grid-columns", type=int, default=3)
    parser.add_argument("--compact-panel-page-rows", type=int, default=DEFAULT_COMPACT_PANEL_PAGE_ROWS)
    parser.add_argument("--no-build", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    plot_pngs = [Path(path) for path in args.plot_png]
    plot_pngs.extend(discover_plot_pngs(args.plot_dir))
    try:
        build_ap_progress_report_pdf(
            trajectory_jsons=tuple(args.trajectory_json),
            plot_pngs=tuple(plot_pngs),
            cost_table_jsons=tuple(args.cost_table_json),
            output_pdf=args.output_pdf,
            output_manifest=args.output_manifest,
            title=str(args.title),
            notes=tuple(args.note),
            plot_grid_columns=int(args.plot_grid_columns),
            compact_panel_page_rows=int(args.compact_panel_page_rows),
            build_pdf=not bool(args.no_build),
        )
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_COMPACT_PANEL_PAGE_ROWS",
    "REPORT_SCHEMA_V1",
    "build_ap_progress_report_pdf",
    "discover_plot_pngs",
    "main",
]
