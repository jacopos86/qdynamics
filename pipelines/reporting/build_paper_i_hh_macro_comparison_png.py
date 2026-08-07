#!/usr/bin/env python3
"""Build the Paper-I macro-generator trajectory and Qiskit-cost composite.

This is a deterministic paper-facing figure builder.  It consumes only the
validated Paper-I tracking JSON and does not launch runs or alter manuscript
sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting.build_paper_i_hh_macro_common_accuracy_pdf import (  # noqa: E402
    _clean_s_alg_receipt_closes,
    _compile_comparator_at_k,
    _selection,
    collect_plateau_rows,
)
from pipelines.reporting.build_paper_i_hh_tracking_plateau_costs import (  # noqa: E402
    _read_source_result,
    _snake_prefix,
)


DEFAULT_SOURCE = (
    REPO_ROOT
    / "output/pdf/"
    "paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_novelty_tracking_20260715/"
    "paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_novelty_tracking_20260715.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "MATH/paper_details/figures/paper_i_hh_macro_comparison_20260723"
)
PNG_NAME = "paper_i_hh_macro_comparison_20260723.png"
PROVENANCE_NAME = "paper_i_hh_macro_comparison_20260723_provenance.json"

REGIMES = (
    ("weak_weak", "Weak–weak", "WW"),
    ("intermediate_weak", "Intermediate–weak", "IW"),
    ("strong_weak_u8", "Strong–weak", "SW"),
    ("weak_strong", "Weak–strong", "WS"),
    ("intermediate_strong", "Intermediate–strong", "IS"),
    ("strong_strong_u8", "Strong–strong", "SS"),
)

METHODS = (
    {
        "key": "append",
        "route_id": "append_adapt_macro_nph3_7",
        "label": "Append-ADAPT",
        "short": "Append",
        "color": "#4C78A8",
        "marker": "o",
        "linewidth": 1.45,
    },
    {
        "key": "geo",
        "route_id": "geo_adapt_macro_nph3_7",
        "label": "Geo-ADAPT",
        "short": "Geo",
        "color": "#54A24B",
        "marker": "^",
        "linewidth": 1.45,
    },
    {
        "key": "snake",
        "route_id": "sr_macro_physical_lanes_nph3_7",
        "label": "SNAKE",
        "short": "SNAKE",
        "color": "#E45756",
        "marker": "*",
        "linewidth": 2.15,
    },
)

TARGET = 2.0e-4
S_ALG_CORRECTED_METHODS = frozenset({"append", "snake"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sci(value: float) -> str:
    return f"{value:.2e}"


def load_routes(source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(source.read_text())
    routes_by_id = {route["id"]: route for route in payload["routes"]}
    selected = {}
    for method in METHODS:
        route_id = method["route_id"]
        if route_id not in routes_by_id:
            raise KeyError(f"missing route {route_id!r} in {source}")
        selected[method["key"]] = routes_by_id[route_id]
    return payload, selected


def validate(routes: dict[str, Any]) -> None:
    for method in METHODS:
        route = routes[method["key"]]
        for regime, _, _ in REGIMES:
            result = route["results"][regime]
            trajectory = result["trajectory"]
            if result.get("status") != "complete":
                raise RuntimeError(
                    f"{method['label']} {regime}: result is not complete"
                )
            if len(trajectory) != 50:
                raise RuntimeError(
                    f"{method['label']} {regime}: expected 50 points, "
                    f"found {len(trajectory)}"
                )
            rounds = [int(point["round"]) for point in trajectory]
            if rounds != list(range(1, 51)):
                raise RuntimeError(
                    f"{method['label']} {regime}: stored rounds are not 1..50"
                )
            plateau = route["plateau"][regime]
            if plateau.get("status") != "complete":
                raise RuntimeError(
                    f"{method['label']} {regime}: plateau prefix is incomplete"
                )
            if "qiskit" not in plateau:
                raise RuntimeError(
                    f"{method['label']} {regime}: plateau Qiskit costs missing"
                )
            if "S_alg" not in plateau:
                raise RuntimeError(
                    f"{method['label']} {regime}: plateau oracle count missing"
                )


def clean_s_alg_overrides(
    *,
    tracker: Mapping[str, Any],
    routes: Mapping[str, Mapping[str, Any]],
    plateau_rows: list[dict[str, Any]] | None = None,
) -> dict[tuple[str, str, str], int]:
    """Reconstruct Append/SNAKE counts without mutating the locked tracker.

    Geo-ADAPT remains outside the current clean Append/SNAKE recount because
    its surviving support exposes only the historical unique-primitive
    receipt.  Its displayed value is therefore intentionally preserved.
    """

    overrides: dict[tuple[str, str, str], int] = {}
    clean_plateau_rows = (
        plateau_rows
        if plateau_rows is not None
        else collect_plateau_rows(tracker)
    )
    for row in clean_plateau_rows:
        method = str(row["method"])
        if method in S_ALG_CORRECTED_METHODS:
            overrides[("plateau", method, str(row["regime"]))] = int(
                row["S_alg"]
            )

    method_by_key = {method["key"]: method for method in METHODS}
    for method_key in ("append", "snake"):
        method = method_by_key[method_key]
        route = routes[method_key]
        for regime, _title, _abbreviation in REGIMES:
            target = route["target_energy"][regime]
            if target.get("status") != "complete":
                continue
            k = int(target["k_target"])
            trajectory = route["results"][regime]["trajectory"]
            if method_key == "append":
                prefix, _source_receipt = _compile_comparator_at_k(
                    source=route["results"][regime]["source"],
                    trajectory=trajectory,
                    k=k,
                    representation="intact_macro",
                )
            else:
                source = route["results"][regime]["source"]
                payload, _runtime_seed, source_receipt = _read_source_result(
                    source,
                    need_runtime_seed=False,
                )
                prefix = _snake_prefix(
                    payload,
                    selection=_selection(trajectory=trajectory, k=k),
                    source=source_receipt,
                    route_id=method["route_id"],
                    fallback_source_kind=(
                        "paper_i_hh_snake_macro_target_prefix"
                    ),
                )
            receipt = prefix.get("S_alg_receipt")
            if not isinstance(receipt, Mapping) or not _clean_s_alg_receipt_closes(
                receipt=receipt,
                scalar=prefix.get("S_alg"),
                accepted_prefix_length=k,
            ):
                raise ValueError(
                    f"{method['label']} {regime} target prefix lacks a "
                    "closed clean-algorithm S_alg receipt"
                )
            overrides[("target", method_key, regime)] = int(prefix["S_alg"])
    return overrides


def row_limits(
    routes: dict[str, Any], regime_keys: list[str]
) -> tuple[float, float]:
    values = []
    for method in METHODS:
        route = routes[method["key"]]
        for regime in regime_keys:
            values.extend(
                float(point["error"])
                for point in route["results"][regime]["trajectory"]
                if float(point["error"]) > 0
            )
    low = 10 ** np.floor(np.log10(min(values)))
    high = 10 ** np.ceil(np.log10(max(values)))
    return low, high


def style_axes(ax: mpl.axes.Axes) -> None:
    ax.set_yscale("log")
    ax.set_xlim(0, 50)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(which="major", axis="both", color="#D8D8D8", linewidth=0.55)
    ax.grid(which="minor", axis="y", color="#EEEEEE", linewidth=0.35)
    ax.tick_params(axis="both", which="major", labelsize=8, width=0.7)
    ax.tick_params(axis="both", which="minor", width=0.45, length=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#555555")


def make_plot_panels(
    fig: mpl.figure.Figure,
    grid: mpl.gridspec.GridSpecFromSubplotSpec,
    routes: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    top_limits = row_limits(routes, [entry[0] for entry in REGIMES[:3]])
    bottom_limits = row_limits(routes, [entry[0] for entry in REGIMES[3:]])
    axes: list[mpl.axes.Axes] = []

    for index, (regime, title, abbreviation) in enumerate(REGIMES):
        row, col = divmod(index, 3)
        share_y = axes[row * 3] if col > 0 else None
        ax = fig.add_subplot(grid[row, col], sharey=share_y)
        axes.append(ax)
        style_axes(ax)
        ax.set_ylim(*(top_limits if row == 0 else bottom_limits))
        if col > 0:
            ax.tick_params(labelleft=False)
        ax.set_title(title, fontsize=9.2, pad=4.0)
        ax.text(
            0.015,
            0.96,
            f"({chr(ord('a') + index)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            fontweight="bold",
        )

        panel_provenance: dict[str, Any] = {
            "regime": regime,
            "abbreviation": abbreviation,
            "n_ph": 3 if row == 0 else 7,
            "curves": {},
        }
        for method in METHODS:
            route = routes[method["key"]]
            trajectory = route["results"][regime]["trajectory"]
            x = np.asarray([int(point["round"]) for point in trajectory])
            y = np.asarray([float(point["error"]) for point in trajectory])
            ax.plot(
                x,
                y,
                color=method["color"],
                linewidth=method["linewidth"],
                solid_capstyle="round",
                zorder=2,
            )

            plateau = route["plateau"][regime]
            k_plateau = int(plateau["k_pl"])
            error_plateau = float(plateau["error"])
            ax.scatter(
                [k_plateau],
                [error_plateau],
                color=method["color"],
                marker=method["marker"],
                s=42 if method["marker"] != "*" else 68,
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
            panel_provenance["curves"][method["key"]] = {
                "route_id": method["route_id"],
                "point_count": len(trajectory),
                "stored_round_range": [int(x.min()), int(x.max())],
                "initial_state_synthesized": False,
                "marker_policy": "first plateau prefix from route plateau receipt",
                "marker": {
                    "round": k_plateau,
                    "error": error_plateau,
                },
                "result_source": route["results"][regime]["source"],
            }

        if row == 1:
            ax.set_xlabel("ADAPT iteration, $k$", fontsize=8.5)
        if col == 0:
            ax.set_ylabel("Same-cutoff energy error, $\\Delta E$", fontsize=8.5)
            ax.text(
                -0.28,
                0.5,
                f"$n_{{\\rm ph}}={3 if row == 0 else 7}$",
                transform=ax.transAxes,
                rotation=90,
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
            )
        provenance["trajectory_panels"].append(panel_provenance)

    handles = [
        Line2D(
            [0],
            [0],
            color=method["color"],
            linewidth=method["linewidth"],
            marker=method["marker"],
            markersize=6.5 if method["marker"] != "*" else 9,
            markerfacecolor=method["color"],
            markeredgecolor="white",
            label=method["label"],
        )
        for method in METHODS
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="none",
            marker="o",
            markerfacecolor="#666666",
            markeredgecolor="white",
            markersize=6,
            label="marker: selected plateau prefix",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.983),
        ncol=4,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.4,
        fontsize=8.5,
    )


def style_table(
    table: mpl.table.Table,
    *,
    header_color: str,
    header_rows: tuple[int, ...] = (0,),
    font_size: float = 7.5,
) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#B8B8B8")
        cell.set_linewidth(0.45)
        if row in header_rows:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F4F4F4")
        else:
            cell.set_facecolor("white")
        cell.PAD = 0.035


def make_plateau_tables(
    fig: mpl.figure.Figure,
    grid: mpl.gridspec.GridSpecFromSubplotSpec,
    routes: dict[str, Any],
    provenance: dict[str, Any],
    s_alg_overrides: Mapping[tuple[str, str, str], int],
    qiskit_cost_overrides: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> None:
    for col, method in enumerate(METHODS):
        ax = fig.add_subplot(grid[0, col])
        ax.axis("off")
        route = routes[method["key"]]
        rows = []
        for regime, _, abbreviation in REGIMES:
            record = route["plateau"][regime]
            qiskit = {
                **record["qiskit"],
                **qiskit_cost_overrides.get(
                    ("plateau", method["key"], regime),
                    {},
                ),
            }
            if qiskit.get("W1q") is None:
                raise ValueError(
                    f"{method['label']} {regime} plateau lacks exact W1q"
                )
            original_s_alg = int(record["S_alg"])
            s_alg = int(
                s_alg_overrides.get(
                    ("plateau", method["key"], regime),
                    original_s_alg,
                )
            )
            row = {
                "regime": regime,
                "abbreviation": abbreviation,
                "k_pl": int(record["k_pl"]),
                "error": float(record["error"]),
                "N2q": int(qiskit["N2q"]),
                "D2q": int(qiskit["D2q"]),
                "Dc": int(qiskit["Dc"]),
                "W1q": int(qiskit["W1q"]),
                "S_alg": s_alg,
                "S_alg_original_tracker_value": original_s_alg,
                "S_alg_accounting": (
                    "clean_algorithm_recount"
                    if method["key"] in S_ALG_CORRECTED_METHODS
                    else "historical_geo_support_receipt_preserved"
                ),
            }
            rows.append(row)
            provenance["plateau_costs"].append(
                {"method": method["key"], "route_id": method["route_id"], **row}
            )

        cell_text = [
            [
                row["abbreviation"],
                str(row["k_pl"]),
                sci(row["error"]),
                f"{row['N2q']:,}",
                f"{row['D2q']:,}",
                f"{row['Dc']:,}",
                f"{row['W1q']:,}",
                f"{row['S_alg']:,}",
            ]
            for row in rows
        ]
        table = ax.table(
            cellText=cell_text,
            colLabels=[
                "Reg.",
                "$k_{\\rm pl}$",
                "$\\Delta E$",
                "$N_{2q}$",
                "$D_{2q}$",
                "$D_c$",
                "$W_{1q}$",
                "$S_{\\rm alg}$",
            ],
            cellLoc="center",
            colLoc="center",
            bbox=[0.0, 0.0, 1.0, 0.88],
            colWidths=[0.09, 0.11, 0.17, 0.12, 0.12, 0.12, 0.13, 0.14],
        )
        style_table(table, header_color=method["color"], font_size=5.9)
        ax.text(
            0.5,
            0.965,
            method["label"],
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            color=method["color"],
        )


def make_target_table(
    fig: mpl.figure.Figure,
    grid: mpl.gridspec.GridSpecFromSubplotSpec,
    routes: dict[str, Any],
    provenance: dict[str, Any],
    s_alg_overrides: Mapping[tuple[str, str, str], int],
    qiskit_cost_overrides: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> None:
    ax = fig.add_subplot(grid[0, 0])
    ax.axis("off")
    rows = []
    method_rank = {method["key"]: index for index, method in enumerate(METHODS)}
    regime_rank = {regime: index for index, (regime, _, _) in enumerate(REGIMES)}
    method_by_key = {method["key"]: method for method in METHODS}

    for method in METHODS:
        route = routes[method["key"]]
        for regime, _, abbreviation in REGIMES:
            record = route["target_energy"][regime]
            if record.get("status") != "complete":
                continue
            qiskit = {
                **record["qiskit"],
                **qiskit_cost_overrides.get(
                    ("target", method["key"], regime),
                    {},
                ),
            }
            if qiskit.get("W1q") is None:
                raise ValueError(
                    f"{method['label']} {regime} target lacks exact W1q"
                )
            original_s_alg = int(record["S_alg"])
            s_alg = int(
                s_alg_overrides.get(
                    ("target", method["key"], regime),
                    original_s_alg,
                )
            )
            rows.append(
                {
                    "method": method["key"],
                    "method_label": method["label"],
                    "route_id": method["route_id"],
                    "regime": regime,
                    "abbreviation": abbreviation,
                    "k_target": int(record["k_target"]),
                    "error": float(record["error"]),
                    "N2q": int(qiskit["N2q"]),
                    "D2q": int(qiskit["D2q"]),
                    "Dc": int(qiskit["Dc"]),
                    "W1q": int(qiskit["W1q"]),
                    "S_alg": s_alg,
                    "S_alg_original_tracker_value": original_s_alg,
                    "S_alg_accounting": (
                        "clean_algorithm_recount"
                        if method["key"] in S_ALG_CORRECTED_METHODS
                        else "historical_geo_support_receipt_preserved"
                    ),
                }
            )
    rows.sort(key=lambda row: (regime_rank[row["regime"]], method_rank[row["method"]]))
    provenance["target_costs"] = rows

    cell_text = [
        [
            row["method_label"],
            row["abbreviation"],
            str(row["k_target"]),
            sci(row["error"]),
            f"{row['N2q']:,}",
            f"{row['D2q']:,}",
            f"{row['Dc']:,}",
            f"{row['W1q']:,}",
            f"{row['S_alg']:,}",
        ]
        for row in rows
    ]
    table = ax.table(
        cellText=cell_text,
        colLabels=[
            "Method",
            "Reg.",
            "$k_T$",
            "$\\Delta E(k_T)$",
            "$N_{2q}$",
            "$D_{2q}$",
            "$D_c$",
            "$W_{1q}$",
            "$S_{\\rm alg}$",
        ],
        cellLoc="center",
        colLoc="center",
        bbox=[0.025, 0.15, 0.95, 0.73],
        colWidths=[0.19, 0.07, 0.07, 0.16, 0.09, 0.09, 0.10, 0.10, 0.13],
    )
    style_table(table, header_color="#444444", font_size=6.75)

    for row_index, row in enumerate(rows, start=1):
        method = method_by_key[row["method"]]
        table[(row_index, 0)].get_text().set_color(method["color"])
        table[(row_index, 0)].get_text().set_fontweight("bold")

    ax.text(
        0.5,
        0.98,
        "First same-cutoff target hit: $\\Delta E \\leq 2\\times10^{-4}$",
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.035,
        "Only SW and SS are reached within 50 stored iterations; Geo-ADAPT does not reach SS.",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        fontsize=7.3,
        color="#444444",
    )


def build(
    source: Path,
    output_dir: Path,
    *,
    clean_plateau_rows: list[dict[str, Any]] | None = None,
    precomputed_s_alg_overrides: Mapping[tuple[str, str, str], int] | None = None,
    qiskit_cost_overrides: Mapping[
        tuple[str, str, str], Mapping[str, Any]
    ] | None = None,
) -> tuple[Path, Path]:
    payload, routes = load_routes(source)
    validate(routes)
    s_alg_overrides = (
        dict(precomputed_s_alg_overrides)
        if precomputed_s_alg_overrides is not None
        else clean_s_alg_overrides(
            tracker=payload,
            routes=routes,
            plateau_rows=clean_plateau_rows,
        )
    )
    qiskit_cost_overrides = dict(qiskit_cost_overrides or {})
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / PNG_NAME
    provenance_path = output_dir / PROVENANCE_NAME

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

    provenance: dict[str, Any] = {
        "schema": "paper_i_hh_macro_comparison_png_v1",
        "source_tracking_json": str(source.relative_to(REPO_ROOT)),
        "source_tracking_json_sha256": sha256(source),
        "source_schema": payload.get("schema"),
        "route_ids": [method["route_id"] for method in METHODS],
        "metric": "same-cutoff absolute ground-state energy error",
        "target_abs_error": TARGET,
        "plot_policy": {
            "x_axis": "stored ADAPT iteration",
            "x_limits": [0, 50],
            "stored_rounds": [1, 50],
            "initial_state_synthesized": False,
            "y_axis": "logarithmic",
            "shared_y_axis": "within each n_ph row",
            "target_line_drawn": False,
            "curve_markers": "one marker per curve at selected plateau prefix",
        },
        "qiskit_cost_convention": {
            "N2q": "Qiskit two-qubit gate count",
            "D2q": "Qiskit two-qubit depth",
            "Dc": "Qiskit total circuit depth",
            "W1q": (
                "Qiskit-emitted Pauli-rotation one-qubit work before "
                "transpilation: basis changes plus central Rz"
            ),
            "identity": "table_i_basis_gate_transpile_v1",
            "optimization_level": 0,
            "seed_transpiler": 7,
            "reference_state_included": True,
            "backend": None,
        },
        "oracle_query_convention": {
            "symbol": "S_alg",
            "meaning": "logical scalar estimator invocation count",
            "not_physical_shots_or_circuits": True,
            "selected_prefix_scope": "display_prefix",
            "append_snake_accounting": "clean_algorithm_recount",
            "geo_accounting": "historical support receipt preserved",
        },
        "trajectory_panels": [],
        "plateau_costs": [],
        "target_costs": [],
        "blockers": [],
    }

    fig = plt.figure(figsize=(8.1, 10.2), dpi=300)
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[5.05, 2.16, 1.72],
        left=0.09,
        right=0.985,
        top=0.94,
        bottom=0.025,
        hspace=0.18,
    )
    plot_grid = outer[0].subgridspec(2, 3, wspace=0.10, hspace=0.26)
    plateau_grid = outer[1].subgridspec(1, 3, wspace=0.08)
    target_grid = outer[2].subgridspec(1, 1)

    make_plot_panels(fig, plot_grid, routes, provenance)
    fig.text(
        0.09,
        0.427,
        "Selected plateau-prefix resources",
        ha="left",
        va="bottom",
        fontsize=9.4,
        fontweight="bold",
    )
    fig.text(
        0.985,
        0.427,
        "$S_{\\rm alg}$: logical quantum-oracle queries",
        ha="right",
        va="bottom",
        fontsize=7.6,
        color="#444444",
    )
    make_plateau_tables(
        fig,
        plateau_grid,
        routes,
        provenance,
        s_alg_overrides,
        qiskit_cost_overrides,
    )
    make_target_table(
        fig,
        target_grid,
        routes,
        provenance,
        s_alg_overrides,
        qiskit_cost_overrides,
    )

    fig.savefig(png_path, dpi=300, bbox_inches=None, facecolor="white")
    plt.close(fig)

    provenance["generated_png"] = {
        "path": str(png_path.relative_to(REPO_ROOT)),
        "sha256": sha256(png_path),
        "pixel_dimensions": [2430, 3060],
        "dpi": 300,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return png_path, provenance_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    png_path, provenance_path = build(args.source.resolve(), args.output_dir.resolve())
    print(png_path)
    print(provenance_path)


if __name__ == "__main__":
    main()
