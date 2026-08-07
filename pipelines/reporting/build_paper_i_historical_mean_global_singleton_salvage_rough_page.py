#!/usr/bin/env python3
"""Build a standalone six-panel RA-salvage/Append-R70 diagnostic page.

The three nph=3 historical-mean global-singleton RA workers completed their
scientific transitions, but their post-run EXDEV publication failure destroyed
the result, checkpoint, optimized parameters, and estimator ledger.  Scheduler
stdout retains the energy entering transition depth ``d``.  Consequently the
recoverable energy history is k=0..49, not k=1..50.  This diagnostic overlays
that history with the authenticated fresh Append-ADAPT singleton k=0..70 trace
without modifying the evolving report PDF or inventing RA cost tuples.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / (
    "output/pdf/"
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving"
)
SOURCE_PACKAGE = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau6_"
    "r50_20260801_v1_chtc"
)
SALVAGE_ROOT = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau6_"
    "r50_20260801_v1_chtc_runtime/salvage"
)
APPEND_ADAPTER = OUTPUT_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_"
    "evolving_append_singleton_r70_all6_adapter.json"
)
OUTPUT_STEM = (
    "paper_i_historical_mean_global_singleton_salvage_"
    "vs_append_r70_rough_six_panel"
)

PACKAGE_ID = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau6_"
    "r50_20260801_v1_chtc"
)
CLUSTER_ID = 9_400_252
APPEND_SCHEMA = "paper_i_append_adapt_singleton_r70_progress_adapter_v1"
PROVENANCE_SCHEMA = (
    "paper_i_historical_mean_global_singleton_salvage_"
    "vs_append_r70_rough_six_panel_v1"
)
REGIMES = (
    ("weak_weak", "Weak--weak", 3, 0),
    ("intermediate_weak", "Intermediate--weak", 3, 1),
    ("strong_weak_u8", "Strong--weak", 3, 2),
    ("weak_strong", "Weak--strong", 7, None),
    ("intermediate_strong", "Intermediate--strong", 7, None),
    ("strong_strong_u8", "Strong--strong", 7, None),
)
COST_FIELDS = ("N2q", "D2q", "Dc", "W1q", "S_alg")
PLOT_FLOOR = 1.0e-16


class RoughPageError(ValueError):
    """Raised when the sources cannot support this diagnostic page."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digested(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "sha256" in result:
        raise RoughPageError("self-digest input already contains sha256")
    result["sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    observed = value.get("sha256")
    unsigned = copy.deepcopy(dict(value))
    unsigned.pop("sha256", None)
    expected = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if observed != expected:
        raise RoughPageError(f"{label} self-digest drifted")
    return str(observed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RoughPageError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise RoughPageError(f"{label} must be a JSON object")
    return value


def finite(value: Any, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RoughPageError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RoughPageError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise RoughPageError(f"{label} must be >= {minimum}")
    return result


def execution_id(regime: str) -> str:
    return (
        f"historical_mean_global_singleton_v1_r50__{regime}__nph3__"
        "ra_global_singleton_plateau"
    )


def stdout_name(regime: str, proc: int) -> str:
    return f"{CLUSTER_ID}.{proc}__{execution_id(regime)}.out"


def salvage_log_root(salvage_root: Path) -> Path:
    candidates = list(salvage_root.rglob(f"{CLUSTER_ID}.0__*.out"))
    if len(candidates) != 1:
        raise RoughPageError("salvage root does not resolve exactly one proc-0 stdout")
    root = candidates[0].parent
    for regime, _display, _nph, proc in REGIMES[:3]:
        assert proc is not None
        if not (root / stdout_name(regime, proc)).is_file():
            raise RoughPageError(f"missing recovered stdout for {regime}")
    return root


def parse_entering_energy_trace(path: Path) -> list[dict[str, Any]]:
    """Parse depth 1..50 events as accepted energies at k=0..49."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise RoughPageError(f"scheduler stdout is unreadable: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.startswith("AI_LOG "):
            continue
        try:
            event = json.loads(line[len("AI_LOG ") :])
        except json.JSONDecodeError as exc:
            raise RoughPageError("scheduler stdout contains malformed AI_LOG") from exc
        if not isinstance(event, dict) or event.get("event") != "hardcoded_adapt_iter":
            continue
        depth = event.get("depth")
        position = event.get("selected_position")
        if (
            isinstance(depth, bool)
            or not isinstance(depth, int)
            or isinstance(position, bool)
            or not isinstance(position, int)
            or depth < 1
            or position < 0
            or position >= depth
        ):
            raise RoughPageError("iteration depth/position telemetry is invalid")
        rows.append(
            {
                "round": depth - 1,
                "transition_depth": depth,
                "energy": finite(event.get("energy"), label="iteration energy"),
                "next_selected_position": position,
                "next_selected_operator": str(event.get("best_op", "")),
                "next_max_gradient": finite(
                    event.get("max_grad"), label="max gradient", minimum=0.0
                ),
                "next_placement": "append" if position == depth - 1 else "interior",
            }
        )
    if [row["transition_depth"] for row in rows] != list(range(1, 51)):
        raise RoughPageError("stdout does not contain exactly transition depths 1..50")
    if [row["round"] for row in rows] != list(range(50)):
        raise RoughPageError("recovered energy trace is not contiguous k=0..49")
    return rows


def validate_costs(value: Any, *, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(COST_FIELDS):
        raise RoughPageError(f"{label} cost tuple fields drifted")
    result: dict[str, int] = {}
    for field in COST_FIELDS:
        item = value[field]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise RoughPageError(f"{label}.{field} must be a nonnegative integer")
        result[field] = item
    return result


def validate_append_adapter(path: Path) -> tuple[dict[str, Any], str]:
    adapter = load_json(path, label="Append R70 adapter")
    canonical = verify_self_digest(adapter, label="Append R70 adapter")
    if (
        adapter.get("schema") != APPEND_SCHEMA
        or adapter.get("status") != "passed"
        or adapter.get("pending_regimes") != []
        or len(adapter.get("cells", [])) != 6
    ):
        raise RoughPageError("Append R70 adapter identity or closure drifted")
    expected_order = [regime for regime, _display, _nph, _proc in REGIMES]
    if adapter.get("regime_order") != expected_order:
        raise RoughPageError("Append R70 regime order drifted")
    for cell, (regime, _display, nph, _proc) in zip(
        adapter["cells"], REGIMES, strict=True
    ):
        if (
            not isinstance(cell, Mapping)
            or cell.get("regime_id") != regime
            or cell.get("nph") != nph
        ):
            raise RoughPageError(f"Append cell identity drifted for {regime}")
        exact = finite(
            cell.get("exact_same_cutoff_energy"), label=f"{regime} Append exact"
        )
        points = cell.get("points")
        if not isinstance(points, list) or len(points) != 71:
            raise RoughPageError(f"{regime} Append trace is not k=0..70")
        for expected_round, point in enumerate(points):
            if not isinstance(point, Mapping) or point.get("round") != expected_round:
                raise RoughPageError(f"{regime} Append trace is not contiguous")
            energy = finite(point.get("energy"), label=f"{regime} Append energy")
            error = finite(
                point.get("delta_e"), label=f"{regime} Append error", minimum=0.0
            )
            if not math.isclose(
                error, abs(energy - exact), rel_tol=1.0e-10, abs_tol=1.0e-12
            ):
                raise RoughPageError(f"{regime} Append same-cutoff error drifted")
        endpoints = cell.get("endpoints")
        if not isinstance(endpoints, Mapping):
            raise RoughPageError(f"{regime} Append endpoints are missing")
        for key, round_number in (("round_50", 50), ("round_70", 70)):
            endpoint = endpoints.get(key)
            if (
                not isinstance(endpoint, Mapping)
                or endpoint.get("round") != round_number
            ):
                raise RoughPageError(f"{regime} Append {key} identity drifted")
            validate_costs(endpoint.get("costs"), label=f"{regime} Append {key}")
    return adapter, canonical


def first_shared_crossings(
    ra_points: Sequence[Mapping[str, Any]],
    append_points: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ra_min = min(float(point["delta_e"]) for point in ra_points)
    append_min = min(float(point["delta_e"]) for point in append_points)
    target = max(ra_min, append_min)
    inclusive = math.nextafter(target, math.inf)
    ra_crossing = next(
        point for point in ra_points if float(point["delta_e"]) <= inclusive
    )
    append_crossing = next(
        point for point in append_points if float(point["delta_e"]) <= inclusive
    )
    return {
        "policy": (
            "first_crossing_at_shared_attainable_same_cutoff_error_"
            "over_ra_k0_49_and_append_k0_70_v1"
        ),
        "target_delta_e": target,
        "ra_minimum_delta_e": ra_min,
        "append_minimum_delta_e": append_min,
        "ra_crossing_round": int(ra_crossing["round"]),
        "append_crossing_round": int(append_crossing["round"]),
        "ra_cost_available": False,
        "append_matched_prefix_cost_compiled": False,
    }


def build_cells(
    *, package_dir: Path, salvage_root: Path, append_adapter: Mapping[str, Any]
) -> list[dict[str, Any]]:
    manifest = load_json(package_dir / "package_manifest.json", label="package manifest")
    verify_self_digest(manifest, label="package manifest")
    if manifest.get("package_id") != PACKAGE_ID or manifest.get("row_count") != 6:
        raise RoughPageError("source package identity drifted")
    append_by_regime = {
        str(cell["regime_id"]): cell for cell in append_adapter["cells"]
    }
    log_root = salvage_log_root(salvage_root)
    cells: list[dict[str, Any]] = []
    for regime, display, nph, proc in REGIMES:
        append = append_by_regime[regime]
        cell: dict[str, Any] = {
            "regime_id": regime,
            "display_name": display,
            "nph": nph,
            "exact_same_cutoff_energy": float(append["exact_same_cutoff_energy"]),
            "append": copy.deepcopy(append),
            "ra": None,
        }
        if proc is not None:
            eid = execution_id(regime)
            job_path = package_dir / "jobs" / f"{eid}.json"
            job = load_json(job_path, label=f"{regime} job")
            verify_self_digest(job, label=f"{regime} job")
            exact = finite(job.get("exact_same_cutoff_energy"), label=f"{regime} exact")
            if (
                job.get("execution_id") != eid
                or job.get("regime_id") != regime
                or job.get("nph") != nph
                or not math.isclose(
                    exact,
                    float(append["exact_same_cutoff_energy"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-10,
                )
            ):
                raise RoughPageError(f"{regime} RA/Append identity drifted")
            log_path = log_root / stdout_name(regime, proc)
            points = parse_entering_energy_trace(log_path)
            for point in points:
                point["delta_e"] = abs(float(point["energy"]) - exact)
            if not math.isclose(
                float(points[0]["energy"]),
                float(append["points"][0]["energy"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise RoughPageError(f"{regime} initial energy closure drifted")
            shared = first_shared_crossings(points, append["points"])
            cell["ra"] = {
                "execution_id": eid,
                "cluster_id": CLUSTER_ID,
                "proc_id": proc,
                "points": points,
                "point_count": 50,
                "round_span": [0, 49],
                "marker_round": 49,
                "marker_policy": "terminal_recoverable_energy_point",
                "scientific_transition_depths_observed": [1, 50],
                "post_refit_round_50_energy_available": False,
                "qiskit_cost_available": False,
                "s_alg_available": False,
                "source": {
                    "scheduler_stdout": file_binding(log_path),
                    "job": file_binding(job_path),
                    "job_canonical_sha256": job["sha256"],
                },
            }
            cell["shared_attainable_selection"] = shared
        else:
            cell["shared_attainable_selection"] = None
        cells.append(cell)
    return cells


def format_cost(costs: Mapping[str, Any]) -> str:
    values = {field: int(costs[field]) for field in COST_FIELDS}
    return (
        f"({values['N2q']},{values['D2q']},{values['Dc']},"
        f"{values['W1q']},{values['S_alg']:.2e})"
    )


def render_page(cells: Sequence[Mapping[str, Any]], *, png: Path, pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, NullFormatter

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.4,
            "axes.labelsize": 7.8,
            "axes.titlesize": 8.8,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
        }
    )
    fig = plt.figure(figsize=(11.0, 8.5))
    grid = fig.add_gridspec(
        3,
        3,
        height_ratios=(1.0, 1.0, 0.76),
        left=0.055,
        right=0.985,
        top=0.875,
        bottom=0.055,
        hspace=0.34,
        wspace=0.20,
    )
    axes = [fig.add_subplot(grid[index // 3, index % 3]) for index in range(6)]
    for row in range(2):
        row_axes = axes[row * 3 : (row + 1) * 3]
        for axis in row_axes[1:]:
            axis.sharey(row_axes[0])

    for index, (axis, cell) in enumerate(zip(axes, cells, strict=True)):
        append_points = cell["append"]["points"]
        append_x = [int(point["round"]) for point in append_points]
        append_y = [max(float(point["delta_e"]), PLOT_FLOOR) for point in append_points]
        axis.plot(append_x, append_y, color="#4C78A8", linewidth=1.45)
        axis.scatter(
            [append_x[-1]],
            [append_y[-1]],
            color="#4C78A8",
            marker="o",
            s=24,
            zorder=4,
        )
        ra = cell.get("ra")
        if isinstance(ra, Mapping):
            ra_points = ra["points"]
            ra_x = [int(point["round"]) for point in ra_points]
            ra_y = [max(float(point["delta_e"]), PLOT_FLOOR) for point in ra_points]
            axis.plot(ra_x, ra_y, color="#E45756", linewidth=1.55)
            axis.scatter(
                [ra_x[-1]],
                [ra_y[-1]],
                color="#E45756",
                marker="D",
                s=25,
                zorder=4,
            )
            shared = cell["shared_attainable_selection"]
            axis.text(
                0.03,
                0.055,
                (
                    f"RA recovered k=0..49\n"
                    f"shared target {float(shared['target_delta_e']):.1e}: "
                    f"k_RA={shared['ra_crossing_round']}, "
                    f"k_A={shared['append_crossing_round']}"
                ),
                transform=axis.transAxes,
                fontsize=5.9,
                va="bottom",
                color="#444444",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "#CCCCCC",
                    "alpha": 0.84,
                },
            )
        else:
            axis.text(
                0.5,
                0.14,
                "RA repaired rerun pending\nAppend-ADAPT shown through k=70",
                transform=axis.transAxes,
                ha="center",
                va="bottom",
                fontsize=6.3,
                color="#555555",
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": "#CCCCCC",
                    "alpha": 0.86,
                },
            )
        axis.set_title(str(cell["display_name"]))
        axis.set_xlim(0, 70)
        axis.set_xticks(range(0, 71, 10))
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(LogLocator(base=10.0))
        axis.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.grid(True, which="major", linewidth=0.42, alpha=0.30)
        axis.grid(True, which="minor", linewidth=0.22, alpha=0.12)
        if index // 3 == 1:
            axis.set_xlabel("ADAPT iteration k")
        if index % 3 == 0:
            axis.set_ylabel(r"Same-cutoff $|\Delta E|$")

    fig.suptitle(
        "Historical-mean global-singleton RA stdout salvage vs fresh Append-ADAPT R70",
        x=0.5,
        y=0.982,
        fontsize=11.2,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.951,
        "ROUGH DIAGNOSTIC - RA k=50 post-refit results and RA cost tuples were lost",
        ha="center",
        va="center",
        color="#9C2F2F",
        fontsize=7.8,
        fontweight="bold",
    )
    fig.legend(
        handles=(
            Line2D(
                [0],
                [0],
                color="#4C78A8",
                marker="o",
                label="Fresh Append-ADAPT singleton (marker: terminal k=70)",
            ),
            Line2D(
                [0],
                [0],
                color="#E45756",
                marker="D",
                label="RA stdout salvage (marker: last recoverable k=49)",
            ),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.932),
        ncol=2,
        frameon=False,
        fontsize=7.2,
        title="Exactly one marker per curve; marker denotes terminal plotted point",
        title_fontsize=6.5,
    )

    table_axis = fig.add_subplot(grid[2, :])
    table_axis.axis("off")
    table_rows: list[list[str]] = []
    for cell in cells:
        append_endpoints = cell["append"]["endpoints"]
        ra = cell.get("ra")
        if isinstance(ra, Mapping):
            last = ra["points"][-1]
            ra_trace = f"k49; DE={float(last['delta_e']):.2e}"
            shared = cell["shared_attainable_selection"]
            shared_text = (
                f"{float(shared['target_delta_e']):.2e}; "
                f"k={shared['ra_crossing_round']}/{shared['append_crossing_round']}"
            )
        else:
            ra_trace = "pending"
            shared_text = "--"
        table_rows.append(
            [
                str(cell["display_name"]),
                ra_trace,
                shared_text,
                "-- (lost)",
                format_cost(append_endpoints["round_50"]["costs"]),
                format_cost(append_endpoints["round_70"]["costs"]),
            ]
        )
    table = table_axis.table(
        cellText=table_rows,
        colLabels=(
            "Regime",
            "RA observed",
            "shared DE; k_RA/k_A",
            "C_RA,last",
            "C_Append,50",
            "C_Append,70",
        ),
        colWidths=(0.12, 0.16, 0.18, 0.11, 0.215, 0.215),
        cellLoc="center",
        loc="upper center",
        bbox=(0.0, 0.19, 1.0, 0.79),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.75)
    for (row, _column), table_cell in table.get_celld().items():
        table_cell.set_linewidth(0.35)
        table_cell.set_edgecolor("#AAAAAA")
        if row == 0:
            table_cell.set_facecolor("#EAEFF5")
            table_cell.set_text_props(weight="bold")
    table_axis.text(
        0.5,
        0.10,
        "C = (N2q, D2q, Dc, W1q, S_alg). Append costs are authenticated; RA costs and matched-prefix costs are not inferred.",
        transform=table_axis.transAxes,
        ha="center",
        va="center",
        fontsize=6.1,
    )
    table_axis.text(
        0.5,
        0.015,
        "Each RA stdout event at transition depth d stores the energy entering that transition, hence the plotted RA trace is k=0..49. This page is not adopted Paper-I evidence.",
        transform=table_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.9,
        color="#444444",
    )

    png.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "Paper-I RA salvage vs Append R70 rough six-panel diagnostic",
        "Creator": "Holstein_test_fullclone_3 reporting pipeline",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(png, dpi=240, metadata={"Software": metadata["Creator"]})
    fig.savefig(pdf, metadata=metadata)
    plt.close(fig)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def build(
    *,
    package_dir: Path,
    salvage_root: Path,
    append_adapter_path: Path,
    output_png: Path,
    output_pdf: Path,
    output_provenance: Path,
) -> dict[str, Any]:
    append_adapter, append_canonical = validate_append_adapter(append_adapter_path)
    cells = build_cells(
        package_dir=package_dir,
        salvage_root=salvage_root,
        append_adapter=append_adapter,
    )
    render_page(cells, png=output_png, pdf=output_pdf)
    provenance = digested(
        {
            "schema": PROVENANCE_SCHEMA,
            "status": "passed_standalone_rough_diagnostic",
            "classification": "stdout_salvage_not_adopted_paper_evidence",
            "metric": "same_cutoff_absolute_energy_error",
            "layout": {
                "page_count": 1,
                "panel_count": 6,
                "grid": "2x3",
                "x_range": [0, 70],
                "shared_y_axis": "within_each_Holstein_row",
                "plot_floor": PLOT_FLOOR,
            },
            "method_style": {
                "append": {
                    "color": "#4C78A8",
                    "line": "solid",
                    "marker": "circle",
                    "marker_round": 70,
                },
                "ra_salvage": {
                    "color": "#E45756",
                    "line": "solid",
                    "marker": "diamond",
                    "marker_round": 49,
                },
            },
            "sources": {
                "source_package_manifest": file_binding(
                    package_dir / "package_manifest.json"
                ),
                "append_adapter": {
                    **file_binding(append_adapter_path),
                    "canonical_sha256": append_canonical,
                },
            },
            "cells": [
                {
                    "regime_id": cell["regime_id"],
                    "display_name": cell["display_name"],
                    "nph": cell["nph"],
                    "append": {
                        "point_count": 71,
                        "round_span": [0, 70],
                        "marker_round": 70,
                        "terminal_delta_e": cell["append"]["points"][-1][
                            "delta_e"
                        ],
                        "round_50_costs": copy.deepcopy(
                            cell["append"]["endpoints"]["round_50"]["costs"]
                        ),
                        "round_70_costs": copy.deepcopy(
                            cell["append"]["endpoints"]["round_70"]["costs"]
                        ),
                    },
                    "ra": copy.deepcopy(cell["ra"]),
                    "shared_attainable_selection": copy.deepcopy(
                        cell["shared_attainable_selection"]
                    ),
                }
                for cell in cells
            ],
            "cost_boundary": {
                "tuple_fields": list(COST_FIELDS),
                "append_round_50_and_70": "authenticated",
                "ra_round_49": "unavailable_checkpoint_parameters_and_ledger_lost",
                "matched_prefix_costs": "not_compiled_in_rough_diagnostic",
            },
            "outputs": {
                "png": file_binding(output_png),
                "pdf": file_binding(output_pdf),
            },
            "limitations": [
                "RA stdout supplies energy points k=0..49 only; post-refit k=50 was lost.",
                "RA Qiskit and S_alg cost tuples are unavailable and are not inferred.",
                "The nph=7 historical-mean global-singleton RA reruns remain pending.",
                "This standalone page does not modify the evolving report PDF.",
            ],
        }
    )
    atomic_write_json(output_provenance, provenance)
    return {
        "status": "built",
        "pdf": str(output_pdf),
        "png": str(output_png),
        "provenance": str(output_provenance),
        "pdf_sha256": sha256_file(output_pdf),
        "provenance_sha256": provenance["sha256"],
        "panel_count": 6,
        "ra_point_counts": {
            str(cell["regime_id"]): (
                int(cell["ra"]["point_count"])
                if isinstance(cell.get("ra"), Mapping)
                else 0
            )
            for cell in cells
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--package-dir", type=Path, default=SOURCE_PACKAGE)
    result.add_argument("--salvage-root", type=Path, default=SALVAGE_ROOT)
    result.add_argument("--append-adapter", type=Path, default=APPEND_ADAPTER)
    result.add_argument(
        "--output-png", type=Path, default=OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    )
    result.add_argument(
        "--output-pdf", type=Path, default=OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
    )
    result.add_argument(
        "--output-provenance",
        type=Path,
        default=OUTPUT_DIR / f"{OUTPUT_STEM}_provenance.json",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = build(
            package_dir=args.package_dir.resolve(),
            salvage_root=args.salvage_root.resolve(),
            append_adapter_path=args.append_adapter.resolve(),
            output_png=args.output_png.resolve(),
            output_pdf=args.output_pdf.resolve(),
            output_provenance=args.output_provenance.resolve(),
        )
    except (OSError, RuntimeError, RoughPageError, ValueError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
