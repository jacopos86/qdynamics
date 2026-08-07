#!/usr/bin/env python3
"""Regenerate Paper-I alternate-Hamiltonian appendix iteration plots.

This is a display-only artifact builder. It reads the active appendix source
inventory from ``build_paper_i_hh_s_accounting_shadow.py`` and rewrites the
corresponding PDF/PNG error-vs-ADAPT-iteration plots. Histories are not
truncated; ``--x-max`` only crops the displayed x-axis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting.build_paper_i_hh_s_accounting_shadow import (  # noqa: E402
    APPENDIX_S_INVENTORY,
)

SNAKE_COLOR = "#E45756"
Y_FLOOR = 1.0e-16
DEFAULT_OUTPUT_DIR = REPO_ROOT / "MATH/paper_details/figures/paper_i_alt_hamiltonian_1p75_20260709"
DEFAULT_PROVENANCE = DEFAULT_OUTPUT_DIR / "paper_i_alt_snake_appendix_error_vs_iteration_xmax5_provenance.json"


@dataclass(frozen=True)
class PlotPoint:
    iteration: int
    abs_delta_e: float
    energy: float | None
    selected_op: str | None


@dataclass(frozen=True)
class PlotSummary:
    surface: str
    family: str
    regime: str
    source_json: str
    source_json_sha256: str
    output_pdf: str
    output_png: str
    point_count: int
    displayed_point_count: int
    marker_policy: str
    marker_k: int
    marker_abs_delta_e: float | None
    marker_visible: bool
    terminal_full_k: int
    terminal_full_abs_delta_e: float | None
    terminal_displayed_k: int
    terminal_displayed_abs_delta_e: float | None
    x_min: int
    x_max: int
    y_floor: float
    status: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _adapt_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    adapt = payload.get("adapt_vqe")
    if isinstance(adapt, Mapping):
        return adapt
    return payload


def _history(adapt: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if "history" not in adapt:
        raise ValueError("source JSON must contain full adapt_vqe.history; history_tail is not accepted")
    raw = adapt.get("history")
    if not isinstance(raw, list) or not raw:
        raise ValueError("source JSON adapt_vqe.history is missing or empty")
    return [row for row in raw if isinstance(row, Mapping)]


def _point_error(row: Mapping[str, Any], *, exact_energy: float | None) -> float | None:
    for key in ("delta_abs_current", "abs_delta_e", "abs_delta", "error"):
        value = _as_float(row.get(key))
        if value is not None:
            return abs(value)
    energy = _as_float(row.get("energy_after_opt") or row.get("energy"))
    if energy is not None and exact_energy is not None:
        return abs(energy - exact_energy)
    return None


def _initial_error(first: Mapping[str, Any], *, exact_energy: float | None) -> tuple[float | None, float | None]:
    for key in ("delta_abs_prev", "delta_abs_initial", "initial_abs_delta_e"):
        value = _as_float(first.get(key))
        if value is not None:
            energy = _as_float(first.get("energy_before_opt"))
            return abs(value), energy
    energy = _as_float(first.get("energy_before_opt"))
    if energy is not None and exact_energy is not None:
        return abs(energy - exact_energy), energy
    return None, energy


def extract_points(payload: Mapping[str, Any]) -> list[PlotPoint]:
    adapt = _adapt_payload(payload)
    hist = _history(adapt)
    if not hist:
        raise ValueError("source JSON has no ADAPT history/history_tail")
    exact_energy = _as_float(adapt.get("exact_gs_energy") or adapt.get("exact_energy"))
    points: list[PlotPoint] = []
    initial_err, initial_energy = _initial_error(hist[0], exact_energy=exact_energy)
    if initial_err is not None:
        points.append(
            PlotPoint(
                iteration=0,
                abs_delta_e=max(initial_err, Y_FLOOR),
                energy=initial_energy,
                selected_op="initial_state",
            )
        )
    for idx, row in enumerate(hist, start=1):
        iteration_raw = row.get("depth") or row.get("iteration") or row.get("adapt_iteration") or idx
        try:
            iteration = int(iteration_raw)
        except (TypeError, ValueError):
            iteration = idx
        err = _point_error(row, exact_energy=exact_energy)
        if err is None:
            continue
        energy = _as_float(row.get("energy_after_opt") or row.get("energy"))
        selected = row.get("selected_op") or row.get("operator_label") or row.get("selected_operator")
        points.append(
            PlotPoint(
                iteration=iteration,
                abs_delta_e=max(abs(err), Y_FLOOR),
                energy=energy,
                selected_op=str(selected) if selected is not None else None,
            )
        )
    if not points:
        raise ValueError("no plottable error points extracted")
    # Keep first occurrence for each iteration, preserving order.
    dedup: dict[int, PlotPoint] = {}
    for point in points:
        dedup[point.iteration] = point
    return [dedup[k] for k in sorted(dedup)]


def _title(item: Mapping[str, Any]) -> str:
    return f"{item['family']} {item['regime']}"


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_plot(item: Mapping[str, Any], *, x_max: int, dpi: int) -> PlotSummary:
    source = REPO_ROOT / str(item["source_json"])
    if not source.exists():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text())
    points = extract_points(payload)
    xs = [p.iteration for p in points]
    ys = [p.abs_delta_e for p in points]
    displayed = [p for p in points if 0 <= p.iteration <= x_max]
    if not displayed:
        raise ValueError(f"no points visible in x range 0..{x_max}: {source}")

    output_pdf = REPO_ROOT / str(item["visible_figure_pdf"])
    output_png = output_pdf.with_suffix(".png")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    k_pl = int(item["k_pl"])
    point_by_k = {p.iteration: p for p in points}
    marker_point = point_by_k.get(k_pl)
    marker_policy = "first_plateau_prefix_from_appendix_inventory"
    if marker_point is None:
        marker_point = points[-1]
        marker_policy = "terminal_full_point_plateau_prefix_missing"

    fig, ax = plt.subplots(figsize=(3.35, 2.45), constrained_layout=True)
    ax.plot(xs, ys, color=SNAKE_COLOR, linewidth=2.0, label="RA-ADAPT")
    ax.scatter(
        [marker_point.iteration],
        [marker_point.abs_delta_e],
        color=SNAKE_COLOR,
        marker="*",
        s=90,
        zorder=5,
        label="plateau/terminal marker",
    )
    ax.set_yscale("log")
    ax.set_xlim(left=0, right=x_max)
    # Let y-limits follow the displayed crop, not hidden post-crop points.
    visible_ys = [p.abs_delta_e for p in displayed]
    ymin = max(min(visible_ys) / 2.5, Y_FLOOR)
    ymax = max(visible_ys) * 2.5
    if ymin >= ymax:
        ymin = max(min(visible_ys) / 10.0, Y_FLOOR)
        ymax = max(visible_ys) * 10.0
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("ADAPT iteration")
    ax.set_ylabel(r"$|E - E_0|$")
    ax.set_title(_title(item), fontsize=9)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(loc="best", fontsize=7, frameon=False)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)

    terminal_full = points[-1]
    terminal_displayed = displayed[-1]
    return PlotSummary(
        surface=str(item["surface"]),
        family=str(item["family"]),
        regime=str(item["regime"]),
        source_json=_relative(source),
        source_json_sha256=_sha256(source),
        output_pdf=_relative(output_pdf),
        output_png=_relative(output_png),
        point_count=len(points),
        displayed_point_count=len(displayed),
        marker_policy=marker_policy,
        marker_k=marker_point.iteration,
        marker_abs_delta_e=marker_point.abs_delta_e,
        marker_visible=0 <= marker_point.iteration <= x_max,
        terminal_full_k=terminal_full.iteration,
        terminal_full_abs_delta_e=terminal_full.abs_delta_e,
        terminal_displayed_k=terminal_displayed.iteration,
        terminal_displayed_abs_delta_e=terminal_displayed.abs_delta_e,
        x_min=0,
        x_max=x_max,
        y_floor=Y_FLOOR,
        status="ok",
    )


def write_provenance(path: Path, *, summaries: Sequence[PlotSummary], x_max: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "paper_i_alt_hamiltonian_appendix_iteration_plots_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "active_paper_i_tex_modified": False,
        "display_policy": {
            "x_min": 0,
            "x_max": x_max,
            "x_crop_only": True,
            "history_source_field": "adapt_vqe.history",
            "history_tail_accepted": False,
            "history_truncated": False,
            "y_axis": "log",
            "y_limits": "computed from displayed points after x crop",
            "marker_policy": "SNAKE star at APPENDIX_S_INVENTORY k_pl; marker may be outside visible x range",
        },
        "source_inventory": "pipelines/reporting/build_paper_i_hh_s_accounting_shadow.py::APPENDIX_S_INVENTORY",
        "plots": [asdict(summary) for summary in summaries],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-max", type=int, default=5, help="display-only right x-axis crop")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    args = parser.parse_args(argv)

    if args.x_max < 1:
        raise SystemExit("--x-max must be >= 1")
    summaries = [write_plot(item, x_max=args.x_max, dpi=args.dpi) for item in APPENDIX_S_INVENTORY]
    provenance = args.provenance
    if not provenance.is_absolute():
        provenance = REPO_ROOT / provenance
    write_provenance(provenance, summaries=summaries, x_max=args.x_max)
    for summary in summaries:
        print(
            f"{summary.surface}\tpoints={summary.point_count}\tdisplayed={summary.displayed_point_count}"
            f"\tmarker_k={summary.marker_k}\tmarker_visible={summary.marker_visible}"
            f"\tterminal_displayed={summary.terminal_displayed_abs_delta_e:.6g}"
            f"\tpdf={summary.output_pdf}"
        )
    print(f"provenance={_relative(provenance)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
