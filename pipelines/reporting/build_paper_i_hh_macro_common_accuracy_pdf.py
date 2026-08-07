#!/usr/bin/env python3
"""Build a macro-comparator review PDF with a final manifest appendix.

Page 1 reports the selected intact-macro plateau prefixes for SNAKE and
Append-ADAPT.  Page 2 compares the same methods at the lowest energy error
attained by both trajectories before the earlier selected plateau.  Every
displayed curve ends at the prefix whose costs are reported, with a small
panel-specific blank margin beyond it.  Geo-ADAPT is excluded because the
surviving support exposes only an identity-collapsed diagnostic, not the clean
logical-estimator invocation count required for ``S_alg``.

The distinct ``*_macro_base_with_manifest.pdf`` output appends its normalized
parameter manifest after the two reader-facing result pages.  A separate
``*_pages1_2.pdf`` compatibility artifact contains exactly those first two
pages for the established downstream page-assembly scripts.  This base builder
never overwrites the canonical multi-review PDF or its provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.paper_i_s_alg_accounting import (
    PAPER_I_S_ALG_ACCOUNTING_SCHEMA,
    PAPER_I_S_ALG_CONTRACT,
)
from pipelines.reporting.build_paper_i_hh_tracking_plateau_costs import (
    _comparator_prefix,
    _finite,
    _read_source_result,
    _sha256_path,
    _snake_prefix,
    _source_path,
)
from pipelines.reporting.paper_i_run_summary import (
    PaperIErrorTracePoint,
    select_paper_i_common_accuracy,
    select_paper_i_effective_plateau,
)
from pipelines.reporting.paper_i_qiskit_cost_tuple import (
    PAPER_I_QISKIT_COST_TUPLE_LATEX,
    paper_i_cost_tuple_latex,
)


TRACKER = REPO_ROOT / (
    "output/pdf/"
    "paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_novelty_tracking_20260715/"
    "paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_novelty_tracking_20260715.json"
)
OUTPUT_DIR = REPO_ROOT / (
    "MATH/paper_details/figures/paper_i_hh_macro_common_accuracy_20260723"
)
STEM = "paper_i_hh_macro_common_accuracy_20260723"
BASE_REPORT_STEM = f"{STEM}_macro_base_with_manifest"

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
        "key": "snake",
        "route_id": "sr_macro_physical_lanes_nph3_7",
        "label": "RA-ADAPT",
        "color": "#E45756",
        "marker": "*",
        "linewidth": 2.15,
    },
    {
        "key": "append",
        "route_id": "append_adapt_macro_nph3_7",
        "label": "Append-ADAPT",
        "color": "#4C78A8",
        "marker": "o",
        "linewidth": 1.45,
    },
)
PAGE_ONE_METHODS = METHODS
DISPLAY_PADDING = 0.15


def _required_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object.")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_reader_pages_copy(*, source_pdf: Path, target_pdf: Path) -> None:
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(source_pdf))
    if len(reader.pages) < 2:
        raise ValueError(
            "macro-comparator PDF must contain at least two reader pages."
        )
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    writer.add_page(reader.pages[1])
    with target_pdf.open("wb") as handle:
        writer.write(handle)


def _selection(*, trajectory: list[dict[str, Any]], k: int) -> dict[str, Any]:
    point = trajectory[k - 1]
    return {
        "history_position": k,
        "k_pl": k,
        "outer_iteration": int(point["round"]),
        "horizon": len(trajectory),
        "error": float(point["error"]),
        "best_observed_error": min(float(row["error"]) for row in trajectory),
        "threshold": float(point["error"]),
    }


def _compile_comparator_at_k(
    *,
    source: Mapping[str, Any],
    trajectory: list[dict[str, Any]],
    k: int,
    representation: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compile one exact comparator prefix with bounded-memory archive reads."""

    from pipelines.reporting.build_paper_i_hh_comparator_tracking_summary import (
        _tar_array_item,
        _tar_json_member,
    )

    path = _source_path(source)
    observed_sha = _sha256_path(path)
    expected_sha = str(source.get("sha256") or "")
    if not expected_sha or observed_sha != expected_sha:
        raise ValueError(f"comparator archive hash drift: {path}")
    member_name = str(source.get("member") or "")
    if not member_name:
        raise ValueError("comparator source lacks result member")
    zero_index = k - 1
    history_row = _tar_array_item(
        path,
        member_name=member_name,
        array_key="adapt_history",
        zero_index=zero_index,
    )
    receipt_row = _tar_array_item(
        path,
        member_name=member_name,
        array_key="estimator_call_round_receipts",
        zero_index=zero_index,
    )
    seed_member = member_name.rsplit("/", 1)[0] + "/runtime_seed.json"
    runtime_seed = _tar_json_member(path, member_name=seed_member)
    expected_error = float(trajectory[zero_index]["error"])
    source_error = history_row.get("abs_delta_e_same_cutoff_after")
    if source_error is None:
        source_error = history_row.get("abs_delta_e_after")
    if source_error is None or not math.isclose(
        abs(_finite(source_error, label="comparator crossing error")),
        expected_error,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("comparator crossing row disagrees with tracker trajectory")
    payload = {
        "status": "completed",
        "result": {
            "adapt_history": [{} for _ in range(zero_index)] + [history_row],
            "estimator_call_round_receipts": [{} for _ in range(zero_index)]
            + [receipt_row],
        },
    }
    selection = _selection(trajectory=trajectory, k=k)
    prefix = _comparator_prefix(
        payload,
        runtime_seed=runtime_seed,
        selection=selection,
        representation=str(representation),
        source_kind="paper_i_hh_comparator_exact_common_accuracy_prefix",
    )
    source_receipt = {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": observed_sha,
        "result_member": member_name,
        "runtime_seed_member": seed_member,
        "streaming_bounded_memory": True,
    }
    return prefix, source_receipt


def _clean_s_alg_receipt_closes(
    *,
    receipt: Mapping[str, Any],
    scalar: Any,
    accepted_prefix_length: int,
) -> bool:
    components = receipt.get("components")
    if not isinstance(components, Mapping):
        return False
    try:
        component_total = sum(
            int(components[key])
            for key in ("N_H_outer", "N_H_refit", "N_grad", "N_metric")
        )
        receipt_total = int(receipt.get("S_alg"))
        scalar_total = int(scalar)
        receipt_prefix = int(receipt.get("accepted_prefix_length"))
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        receipt.get("schema") == PAPER_I_S_ALG_ACCOUNTING_SCHEMA
        and receipt.get("contract") == PAPER_I_S_ALG_CONTRACT
        and receipt.get("representation") not in {None, ""}
        and receipt_prefix == int(accepted_prefix_length)
        and component_total == receipt_total == scalar_total
    )


def _existing_prefix(
    *,
    route: Mapping[str, Any],
    regime: str,
    k: int,
) -> tuple[dict[str, Any], str] | None:
    result = route["results"][regime]
    plateau = route["plateau"][regime]
    plateau_s_alg_receipt = plateau.get("S_alg_receipt")
    if (
        int(plateau["k_pl"]) == k
        and isinstance(plateau_s_alg_receipt, Mapping)
        and _clean_s_alg_receipt_closes(
            receipt=plateau_s_alg_receipt,
            scalar=plateau["S_alg"],
            accepted_prefix_length=k,
        )
    ):
        return {
            "active_depth": int(plateau["active_depth"]),
            "S_alg": int(plateau["S_alg"]),
            "S_alg_scope": plateau["S_alg_scope"],
            "S_alg_components": plateau.get("S_alg_components"),
            "S_alg_receipt": dict(plateau_s_alg_receipt),
            "S_alg_reconstruction_status": plateau.get("S_alg_reconstruction_status"),
            "qiskit": dict(plateau["qiskit"]),
            "qiskit_compile": dict(plateau["qiskit_compile"]),
            "prefix_receipt": dict(plateau["prefix_receipt"]),
        }, "reused validated plateau prefix"
    result_s_alg_receipt = result.get("S_alg_receipt")
    if (
        k == len(result["trajectory"])
        and isinstance(result_s_alg_receipt, Mapping)
        and _clean_s_alg_receipt_closes(
            receipt=result_s_alg_receipt,
            scalar=result["s_alg"],
            accepted_prefix_length=k,
        )
    ):
        return {
            "active_depth": int(result.get("active_depth") or k),
            "S_alg": int(result["s_alg"]),
            "S_alg_scope": result["s_alg_scope"],
            "S_alg_components": result_s_alg_receipt.get("components"),
            "S_alg_receipt": dict(result_s_alg_receipt),
            "S_alg_reconstruction_status": (
                "validated clean-algorithm terminal result"
            ),
            "qiskit": dict(route["costs"][regime]),
            "qiskit_compile": {
                "identity": "table_i_basis_gate_transpile_v1",
                "optimization_level": 0,
                "seed_transpiler": 7,
                "backend": None,
                "reference_state_included": True,
                "source_kind": "validated terminal prefix",
            },
            "prefix_receipt": {"mode": "validated terminal result"},
        }, "reused validated terminal prefix"
    return None


def _typed_error_trace(
    trajectory: Any,
    *,
    owner: str,
) -> tuple[PaperIErrorTracePoint, ...]:
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError(f"{owner} lacks a complete trajectory.")
    rows: list[PaperIErrorTracePoint] = []
    for index, point in enumerate(trajectory, start=1):
        if not isinstance(point, Mapping):
            raise TypeError(f"{owner} trajectory row {index} is not an object.")
        rows.append(
            PaperIErrorTracePoint(
                controller_round=int(point.get("round") or 0),
                absolute_energy_error=_finite(
                    point.get("error"),
                    label=f"{owner} trajectory row {index} error",
                ),
            )
        )
    return tuple(rows)


def collect_rows(tracker: Mapping[str, Any]) -> list[dict[str, Any]]:
    routes = {route["id"]: route for route in tracker["routes"]}
    selected = {
        method["key"]: routes[method["route_id"]]
        for method in METHODS
    }
    rows: list[dict[str, Any]] = []
    for regime, title, abbreviation, n_ph in REGIMES:
        trajectories = {
            key: selected[key]["results"][regime]["trajectory"]
            for key in ("snake", "append")
        }
        typed_traces = {
            key: _typed_error_trace(
                trajectory,
                owner=f"{regime} {key}",
            )
            for key, trajectory in trajectories.items()
        }
        common = select_paper_i_common_accuracy(
            typed_traces["snake"],
            typed_traces["append"],
        )
        recorded_plateaus = (
            int(selected["snake"]["plateau"][regime]["k_pl"]),
            int(selected["append"]["plateau"][regime]["k_pl"]),
        )
        derived_plateaus = (
            common.sr_snake_plateau_controller_round,
            common.append_adapt_plateau_controller_round,
        )
        if recorded_plateaus != derived_plateaus:
            raise ValueError(
                f"{regime} tracker plateau drift: recorded="
                f"{recorded_plateaus!r}, canonical={derived_plateaus!r}."
            )
        common_window_end = common.shared_window_end_controller_round
        common_error = common.common_target_absolute_error
        minima = {
            "snake": common.sr_snake_window_minimum_error,
            "append": common.append_adapt_window_minimum_error,
        }
        crossings = {
            "snake": common.sr_snake_crossing_controller_round,
            "append": common.append_adapt_crossing_controller_round,
        }
        for method in METHODS:
            key = method["key"]
            route = selected[key]
            k = crossings[key]
            trajectory = trajectories[key]
            existing = _existing_prefix(route=route, regime=regime, k=k)
            if existing is not None:
                prefix, recovery = existing
                source_receipt = route["results"][regime]["source"]
            elif key == "append":
                prefix, source_receipt = _compile_comparator_at_k(
                    source=route["results"][regime]["source"],
                    trajectory=trajectory,
                    k=k,
                    representation="intact_macro",
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
                    selection=_selection(trajectory=trajectory, k=k),
                    source=source_receipt,
                    route_id=method["route_id"],
                    fallback_source_kind="paper_i_hh_snake_common_accuracy_prefix",
                )
                recovery = "exact signed-checkpoint reconstruction and compile"
            receipt = prefix.get("S_alg_receipt")
            if not isinstance(receipt, Mapping) or not _clean_s_alg_receipt_closes(
                receipt=receipt,
                scalar=prefix.get("S_alg"),
                accepted_prefix_length=k,
            ):
                raise ValueError(
                    f"{abbreviation} {method['label']} common-accuracy prefix "
                    "lacks a clean-v2 logical-estimator receipt."
                )
            qiskit = prefix["qiskit"]
            rows.append(
                {
                    "regime": regime,
                    "regime_title": title,
                    "abbreviation": abbreviation,
                    "n_ph": n_ph,
                    "common_window_end": common_window_end,
                    "snake_plateau_k": (
                        common.sr_snake_plateau_controller_round
                    ),
                    "append_plateau_k": (
                        common.append_adapt_plateau_controller_round
                    ),
                    "method": key,
                    "method_label": method["label"],
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
            )
            print(
                f"{abbreviation} {method['label']}: Ecap={common_error:.8e}, "
                f"k={k}, N2q={qiskit['N2q']}, S_alg={prefix['S_alg']}",
                flush=True,
            )
    return rows


def collect_plateau_rows(tracker: Mapping[str, Any]) -> list[dict[str, Any]]:
    routes = {route["id"]: route for route in tracker["routes"]}
    rows: list[dict[str, Any]] = []
    for regime, title, abbreviation, n_ph in REGIMES:
        for method in PAGE_ONE_METHODS:
            route = routes[method["route_id"]]
            plateau = route["plateau"][regime]
            trajectory = route["results"][regime]["trajectory"]
            canonical_plateau = select_paper_i_effective_plateau(
                _typed_error_trace(
                    trajectory,
                    owner=f"{regime} {method['key']}",
                )
            )
            k = canonical_plateau.controller_round
            if int(plateau["k_pl"]) != k or not math.isclose(
                float(plateau["error"]),
                canonical_plateau.absolute_energy_error,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    f"{regime} {method['label']} tracker plateau disagrees "
                    "with paper_i_effective_plateau_v1."
                )
            if (
                plateau.get("best_observed_error") is not None
                and not math.isclose(
                    float(plateau["best_observed_error"]),
                    canonical_plateau.best_observed_error,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ):
                raise ValueError(
                    f"{regime} {method['label']} tracker plateau best-error "
                    "receipt drifted."
                )
            if (
                plateau.get("threshold") is not None
                and not math.isclose(
                    float(plateau["threshold"]),
                    canonical_plateau.selection_threshold,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ):
                raise ValueError(
                    f"{regime} {method['label']} tracker plateau threshold "
                    "receipt drifted."
                )
            existing = _existing_prefix(route=route, regime=regime, k=k)
            if existing is not None:
                prefix, recovery = existing
                source_receipt = route["results"][regime]["source"]
            elif method["key"] == "append":
                prefix, source_receipt = _compile_comparator_at_k(
                    source=route["results"][regime]["source"],
                    trajectory=trajectory,
                    k=k,
                    representation="intact_macro",
                )
                recovery = (
                    "exact bounded-memory plateau-prefix reconstruction and compile"
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
                        "paper_i_hh_snake_macro_plateau_prefix"
                    ),
                )
                recovery = (
                    "exact signed-checkpoint plateau-prefix reconstruction and compile"
                )
            receipt = prefix.get("S_alg_receipt")
            if not isinstance(receipt, Mapping) or not _clean_s_alg_receipt_closes(
                receipt=receipt,
                scalar=prefix.get("S_alg"),
                accepted_prefix_length=k,
            ):
                raise ValueError(
                    f"{abbreviation} {method['label']} plateau lacks a "
                    "clean-v2 logical-estimator receipt."
                )
            qiskit = prefix["qiskit"]
            rows.append(
                {
                    "regime": regime,
                    "regime_title": title,
                    "abbreviation": abbreviation,
                    "n_ph": n_ph,
                    "method": method["key"],
                    "method_label": method["label"],
                    "route_id": method["route_id"],
                    "k_pl": int(k),
                    "error": canonical_plateau.absolute_energy_error,
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
                    "S_alg_receipt": dict(receipt),
                    "S_alg_reconstruction_status": prefix.get(
                        "S_alg_reconstruction_status"
                    ),
                    "qiskit_compile": prefix["qiskit_compile"],
                    "prefix_receipt": prefix["prefix_receipt"],
                    "source": source_receipt,
                    "recovery": recovery,
                }
            )
    return rows


def _display_x_max(*iterations: int) -> int:
    return max(2, int(math.ceil((1.0 + DISPLAY_PADDING) * max(iterations))))


def _math_compact_sci(value: int) -> str:
    """Format an integer in compact two-significant-digit e notation."""

    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}\\mathrm{{e}}{int(exponent)}"


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


def make_crossing_plot(
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
    fig, axes = plt.subplots(2, 3, figsize=(7.65, 6.10), dpi=300)
    for index, (regime, title, _abbreviation, _n_ph) in enumerate(REGIMES):
        ax = axes.flat[index]
        x_max = _display_x_max(
            *(
                row_lookup[(regime, method["key"])]["k_cross"]
                for method in METHODS
            )
        )
        _style_axes(ax, x_max=x_max)
        all_values: list[float] = []
        for method in METHODS:
            route = routes[method["route_id"]]
            trajectory = route["results"][regime]["trajectory"]
            row = row_lookup[(regime, method["key"])]
            visible = [
                point
                for point in trajectory
                if int(point["round"]) <= row["k_cross"]
            ]
            x = [int(point["round"]) for point in visible]
            y = [float(point["error"]) for point in visible]
            all_values.extend(y)
            ax.plot(
                x,
                y,
                color=method["color"],
                linewidth=method["linewidth"],
                solid_capstyle="round",
            )
            ax.scatter(
                [row["k_cross"]],
                [row["crossing_error"]],
                color=method["color"],
                marker=method["marker"],
                s=58 if method["marker"] == "*" else 38,
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
            tuple_text = paper_i_cost_tuple_latex(
                row,
                marker=(
                    r"\star" if method["marker"] == "*" else r"\bullet"
                ),
                format_s_alg=_math_compact_sci,
            )
            ax.text(
                0.98,
                0.95 if method["key"] == "snake" else 0.87,
                tuple_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6.8,
                color=method["color"],
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                    "pad": 0.5,
                },
                zorder=6,
            )
        common = row_lookup[(regime, "snake")]["common_error"]
        ax.axhline(common, color="#555555", linestyle=(0, (3, 2)), linewidth=0.85)
        low = 10 ** np.floor(np.log10(min(value for value in all_values if value > 0)))
        high = 10 ** np.ceil(np.log10(max(all_values)))
        if regime == "intermediate_weak":
            low = 5.0e-2
        ax.set_ylim(low, high)
        ax.set_title(title, fontsize=9.2, pad=3)
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
    handles.append(
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle=(0, (3, 2)),
            linewidth=0.9,
            label="$\\Delta E_\\cap$",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.91, bottom=0.105, wspace=0.16, hspace=0.28)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def make_plateau_plot(
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
    for index, (regime, title, _abbreviation, _n_ph) in enumerate(REGIMES):
        ax = axes.flat[index]
        x_max = _display_x_max(
            *(
                row_lookup[(regime, method["key"])]["k_pl"]
                for method in PAGE_ONE_METHODS
            )
        )
        _style_axes(ax, x_max=x_max)
        values: list[float] = []
        for method in PAGE_ONE_METHODS:
            row = row_lookup[(regime, method["key"])]
            trajectory = routes[method["route_id"]]["results"][regime]["trajectory"]
            visible = [
                point
                for point in trajectory
                if int(point["round"]) <= row["k_pl"]
            ]
            x = [int(point["round"]) for point in visible]
            y = [float(point["error"]) for point in visible]
            values.extend(y)
            ax.plot(
                x,
                y,
                color=method["color"],
                linewidth=method["linewidth"],
                solid_capstyle="round",
            )
            ax.scatter(
                [row["k_pl"]],
                [row["error"]],
                color=method["color"],
                marker=method["marker"],
                s=58 if method["marker"] == "*" else 42,
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
        low = 10 ** np.floor(np.log10(min(value for value in values if value > 0)))
        high = 10 ** np.ceil(np.log10(max(values)))
        ax.set_ylim(low, high)
        ax.set_title(title, fontsize=9.2, pad=3)
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
        for method in PAGE_ONE_METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(PAGE_ONE_METHODS),
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


def _tex_sci(value: float) -> str:
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10**exponent)
    return rf"${mantissa:.3f}\times 10^{{{exponent}}}$"


def _tex_text(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _tex_breakable_text(value: Any) -> str:
    break_after = {"_", "/", ";", ":", "-"}
    return "".join(
        _tex_text(char) + (r"\allowbreak{}" if char in break_after else "")
        for char in str(value)
    )


def _tex_breakable_hash(value: Any) -> str:
    raw = str(value)
    if len(raw) != 64:
        raise ValueError("expected a 64-character SHA-256 digest.")
    return "".join(
        rf"\texttt{{{_tex_text(raw[index:index + 8])}}}\allowbreak{{}}"
        for index in range(0, len(raw), 8)
    )


def _read_normalized_source_manifest(
    source: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one small normalized manifest without materializing result JSON."""

    import tarfile

    path = _source_path(source)
    result_member = str(source.get("member") or "")
    if not result_member:
        raise ValueError("macro comparison source lacks a result member.")
    result_parent = result_member.rsplit("/", 1)[0]
    run_parent = result_parent.rsplit("/", 1)[0]
    candidates = {
        f"{result_parent}/normalized_job_manifest.json",
        f"{result_parent}/normalized_run_manifest.json",
        f"{run_parent}/normalized_run_manifest.json",
    }
    raw: bytes | None = None
    member_name: str | None = None
    with tarfile.open(path, "r|gz") as archive:
        for info in archive:
            if info.name in candidates:
                handle = archive.extractfile(info)
                if handle is None:
                    raise RuntimeError(
                        f"cannot extract {info.name} from {path}"
                    )
                raw = handle.read()
                member_name = info.name
                break
            archive.members.clear()
    if raw is None or member_name is None:
        raise ValueError(
            f"macro comparison source lacks a normalized manifest: {path}"
        )
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise TypeError("normalized source manifest must be an object.")
    return dict(payload), {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": _sha256_path(path),
        "manifest_member": member_name,
        "manifest_member_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _parameter_manifest_row(
    *,
    regime: str,
    method: Mapping[str, Any],
    normalized_manifest: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
    compile_policy: Mapping[str, Any],
) -> dict[str, Any]:
    physics = _required_mapping(
        normalized_manifest.get("physics"),
        name=f"{regime} {method['key']} physics",
    )
    for drive_value in (
        normalized_manifest.get("drive_enabled"),
        physics.get("drive_enabled"),
    ):
        if drive_value is not None and drive_value is not False:
            raise ValueError(
                f"{regime} {method['label']} is not a static no-drive "
                "artifact."
            )
    schema = str(normalized_manifest.get("schema") or "")
    if schema in {
        "paper_i_hh_append_completion_job_v1",
        "paper_i_hh_geo_completion_job_v1",
    }:
        optimizer = _required_mapping(
            normalized_manifest.get("optimizer"),
            name=f"{regime} append optimizer",
        )
        candidate = _required_mapping(
            normalized_manifest.get("candidate_pool"),
            name=f"{regime} append candidate_pool",
        )
        variant = _required_mapping(
            normalized_manifest.get("variant"),
            name=f"{regime} append variant",
        )
        optimizer_kind = str(optimizer.get("kind") or "")
        optimizer_maxiter = int(optimizer.get("maxiter") or 0)
        seed = int(normalized_manifest.get("seed") or 0)
        candidate_identity = str(
            variant.get("candidate_representation") or ""
        )
        pool_identity = str(candidate.get("parent_pool") or "")
        insertion_policy = (
            str(optimizer.get("position_policy") or "append")
            if schema == "paper_i_hh_geo_completion_job_v1"
            else "append_only"
        )
        candidate_representation = candidate_identity
        hva_policy = (
            "included"
            if bool(candidate.get("hva_included"))
            else "excluded_or_unreported"
        )
        sector_policy = str(
            candidate.get("shared_pauli_pool_symmetry_policy") or "off"
        )
        padding_policy = (
            "authenticated_by_projected_child_identity"
            if "padding_valid" in candidate_identity
            else "not_applicable_intact_macro"
        )
    else:
        route_identity = _required_mapping(
            normalized_manifest.get("route_identity"),
            name=f"{regime} SNAKE route_identity",
        )
        profile_contract = _required_mapping(
            route_identity.get("profile_contract"),
            name=f"{regime} SNAKE profile_contract",
        )
        execution = _required_mapping(
            profile_contract.get("execution_settings"),
            name=f"{regime} SNAKE execution_settings",
        )
        optimizer_kind = str(execution.get("adapt_inner_optimizer") or "")
        optimizer_maxiter = int(execution.get("adapt_maxiter") or 0)
        seed = int(execution.get("adapt_seed") or 0)
        candidate_identity = str(
            route_identity.get("profile_resolved")
            or route_identity.get("profile")
            or method["route_id"]
        )
        pool_identity = str(execution.get("adapt_pool") or "")
        insertion_policy = str(
            execution.get("adapt_insertion_mode") or ""
        )
        semantic = _required_mapping(
            profile_contract.get("semantic_invariants"),
            name=f"{regime} SNAKE semantic_invariants",
        )
        split_mode = str(
            execution.get("phase3_runtime_split_mode") or "off"
        )
        if split_mode == "off":
            candidate_representation = "intact_macro"
            sector_policy = "not_applicable_intact_macro"
            padding_policy = "not_applicable_intact_macro"
        elif int(
            execution.get("phase3_runtime_split_max_subset_size") or 0
        ) == 1:
            candidate_representation = "projected_singleton"
            sector_policy = str(
                execution.get(
                    "phase3_runtime_split_child_set_symmetry_policy"
                )
                or "unreported"
            )
            padding_policy = str(
                execution.get(
                    "phase3_runtime_split_child_padding_policy"
                )
                or "unreported"
            )
        else:
            candidate_representation = split_mode
            sector_policy = str(
                execution.get(
                    "phase3_runtime_split_child_set_symmetry_policy"
                )
                or "unreported"
            )
            padding_policy = str(
                execution.get(
                    "phase3_runtime_split_child_padding_policy"
                )
                or "unreported"
            )
        hva_policy = str(
            semantic.get("full_meta_hva_policy") or "unreported"
        )
    t_value = float(physics.get("t") or 0.0)
    u_value = physics.get("u")
    if u_value is None:
        u_value = float(physics.get("u_over_t") or 0.0) * t_value
    n_ph = int(
        physics.get("n_ph_work", physics.get("n_ph_max", -1))
    )
    exact_reference = normalized_manifest.get("exact_reference")
    exact_energy = physics.get("expected_exact_energy")
    exact_usage = "reporting_only"
    if isinstance(exact_reference, Mapping):
        if exact_energy is None:
            exact_energy = exact_reference.get("energy")
        exact_usage = str(
            exact_reference.get("usage") or exact_usage
        )
    if exact_energy is None:
        raise ValueError(
            f"{regime} {method['label']} lacks its same-cutoff exact target."
        )
    accounting = (
        {
            "schema": "paper_i_historical_geo_accounting_v1",
            "contract": "closed_cumulative_unique_estimator_prefix",
            "canonical_s_alg": False,
        }
        if schema == "paper_i_hh_geo_completion_job_v1"
        else {
            "schema": PAPER_I_S_ALG_ACCOUNTING_SCHEMA,
            "contract": PAPER_I_S_ALG_CONTRACT,
            "canonical_s_alg": True,
        }
    )
    return {
        "regime": regime,
        "method": str(method["key"]),
        "method_label": str(method["label"]),
        "route_id": str(method["route_id"]),
        "physics": {
            "family": str(
                normalized_manifest.get("family")
                or physics.get("problem")
                or "hh"
            ),
            "L": int(physics.get("L") or 0),
            "t": t_value,
            "u": float(u_value),
            "dv": float(physics.get("dv") or 0.0),
            "omega0": float(physics.get("omega0") or 0.0),
            "g_ep": float(physics.get("g_ep") or 0.0),
            "n_ph_max": n_ph,
            "boson_encoding": str(
                physics.get("boson_encoding") or "binary"
            ),
            "ordering": str(physics.get("ordering") or "blocked"),
            "boundary": str(physics.get("boundary") or "open"),
            "drive_enabled": False,
            "same_cutoff_reference": bool(
                physics.get("same_cutoff_reference")
            ),
            "exact_gs_energy": float(exact_energy),
            "exact_reference_usage": exact_usage,
        },
        "optimizer": {
            "kind": optimizer_kind,
            "maxiter": optimizer_maxiter,
            "seed": seed,
        },
        "candidate": {
            "identity": candidate_identity,
            "pool": pool_identity,
            "insertion": insertion_policy,
            "representation": candidate_representation,
            "hva_policy": hva_policy,
            "sector_policy": sector_policy,
            "padding_policy": padding_policy,
        },
        "compile": {
            "identity": str(compile_policy.get("identity") or ""),
            "optimization_level": int(
                compile_policy.get("optimization_level") or 0
            ),
            "seed_transpiler": int(
                compile_policy.get("seed_transpiler") or 0
            ),
            "reference_state_included": bool(
                compile_policy.get("reference_state_included")
            ),
        },
        "accounting": accounting,
        "source": dict(source_receipt),
    }


def collect_parameter_manifest(
    tracker: Mapping[str, Any],
    *,
    methods: Sequence[Mapping[str, Any]] = METHODS,
) -> list[dict[str, Any]]:
    routes = {route["id"]: route for route in tracker["routes"]}
    compile_policy = _required_mapping(
        tracker.get("plateau_compile_policy"),
        name="tracker plateau_compile_policy",
    )
    rows: list[dict[str, Any]] = []
    for regime, _title, _abbreviation, expected_n_ph in REGIMES:
        for method in methods:
            source = _required_mapping(
                routes[method["route_id"]]["results"][regime].get("source"),
                name=f"{regime} {method['key']} source",
            )
            normalized, source_receipt = (
                _read_normalized_source_manifest(source)
            )
            row = _parameter_manifest_row(
                regime=regime,
                method=method,
                normalized_manifest=normalized,
                source_receipt=source_receipt,
                compile_policy=compile_policy,
            )
            if row["physics"]["n_ph_max"] != expected_n_ph:
                raise ValueError(
                    f"{regime} {method['label']} normalized n_ph drift."
                )
            rows.append(row)
    return rows


def _evidence_status() -> dict[str, Any]:
    return {
        "artifact_role": "historical_macro_comparator_report",
        "validation_state": "source_locked_prefix_rows",
        "projected_singleton_included": False,
        "source_target_reference": "MATH/paper_details/Paper_I.tex",
    }


def _parameter_manifest_fragment(
    parameter_manifest: Sequence[Mapping[str, Any]],
    *,
    retained_page_manifest: Sequence[Mapping[str, Any]] = (),
) -> str:
    manifest_body: list[str] = []
    source_body: list[str] = []
    method_contracts: dict[str, Mapping[str, Any]] = {}
    physical_conventions: set[tuple[str, str, str, bool, bool]] = set()
    for raw_row in parameter_manifest:
        manifest_row = _required_mapping(
            raw_row,
            name="parameter manifest row",
        )
        physics = _required_mapping(
            manifest_row.get("physics"),
            name="parameter manifest physics",
        )
        source = _required_mapping(
            manifest_row.get("source"),
            name="parameter manifest source",
        )
        regime = str(manifest_row["regime"])
        abbreviation = next(
            row[2] for row in REGIMES if row[0] == regime
        )
        method_label = str(manifest_row["method_label"])
        physical_conventions.add(
            (
                str(physics["boson_encoding"]),
                str(physics["ordering"]),
                str(physics["boundary"]),
                bool(physics["drive_enabled"]),
                bool(physics["same_cutoff_reference"]),
            )
        )
        manifest_body.append(
            " & ".join(
                (
                    abbreviation,
                    _tex_text(method_label),
                    str(physics["L"]),
                    f"{float(physics['t']):.3g}",
                    f"{float(physics['u']):.3g}",
                    f"{float(physics['dv']):.3g}",
                    f"{float(physics['omega0']):.3g}",
                    f"{float(physics['g_ep']):.6g}",
                    str(physics["n_ph_max"]),
                    f"{float(physics['exact_gs_energy']):.9g}",
                )
            )
            + r" \\"
        )
        source_body.append(
            " & ".join(
                (
                    abbreviation,
                    _tex_text(method_label),
                    _tex_breakable_hash(source["sha256"]),
                    _tex_breakable_hash(
                        source["manifest_member_sha256"]
                    ),
                )
            )
            + r" \\"
        )
        method_key = str(manifest_row["method"])
        existing = method_contracts.setdefault(method_key, manifest_row)
        for contract_key in (
            "route_id",
            "optimizer",
            "candidate",
            "compile",
            "accounting",
        ):
            if existing[contract_key] != manifest_row[contract_key]:
                raise ValueError(
                    f"{method_key} {contract_key} drifts across regimes."
                )
    method_body: list[str] = []
    for method_key in method_contracts:
        manifest_row = method_contracts[method_key]
        candidate = _required_mapping(
            manifest_row.get("candidate"),
            name=f"{method_key} candidate contract",
        )
        optimizer = _required_mapping(
            manifest_row.get("optimizer"),
            name=f"{method_key} optimizer contract",
        )
        compile_contract = _required_mapping(
            manifest_row.get("compile"),
            name=f"{method_key} compile contract",
        )
        accounting_contract = _required_mapping(
            manifest_row.get("accounting"),
            name=f"{method_key} accounting contract",
        )
        contract_rows = (
            ("Route", manifest_row["route_id"]),
            (
                "Optimizer/maxiter/seed",
                f"{optimizer['kind']}/"
                f"{optimizer['maxiter']}/"
                f"{optimizer['seed']}",
            ),
            ("Candidate identity", candidate["identity"]),
            (
                "Pool/insertion/representation",
                f"{candidate['pool']}; {candidate['insertion']}; "
                f"{candidate['representation']}",
            ),
            (
                "HVA/sector/padding",
                f"{candidate['hva_policy']}; "
                f"{candidate['sector_policy']}; "
                f"{candidate['padding_policy']}",
            ),
            (
                "Compile convention",
                f"{compile_contract['identity']}; opt="
                f"{compile_contract['optimization_level']}; seed="
                f"{compile_contract['seed_transpiler']}; ref="
                f"{compile_contract['reference_state_included']}",
            ),
            (
                "Exact-reference use",
                _required_mapping(
                    manifest_row.get("physics"),
                    name=f"{method_key} physics contract",
                )["exact_reference_usage"],
            ),
            (
                "Estimator accounting",
                (
                    "canonical: "
                    if bool(accounting_contract["canonical_s_alg"])
                    else "historical/noncanonical: "
                )
                + f"{accounting_contract['schema']}; "
                + str(accounting_contract["contract"]),
            ),
        )
        for index, (field_name, value) in enumerate(contract_rows):
            method_body.append(
                " & ".join(
                    (
                        (
                            _tex_text(manifest_row["method_label"])
                            if index == 0
                            else ""
                        ),
                        _tex_text(field_name),
                        _tex_breakable_text(value),
                    )
                )
                + r" \\"
            )
        method_body.append(r"\addlinespace[2pt]")
    if len(physical_conventions) != 1:
        raise ValueError(
            "parameter-manifest encoding, ordering, boundary, drive, or "
            "same-cutoff convention drifts across displayed routes."
        )
    encoding, ordering, boundary, drive_enabled, same_cutoff = next(
        iter(physical_conventions)
    )
    retained_page_body: list[str] = []
    for raw_page in retained_page_manifest:
        page = _required_mapping(
            raw_page,
            name="retained diagnostic page manifest row",
        )
        provenance = _required_mapping(
            page.get("provenance"),
            name="retained diagnostic page provenance",
        )
        route_ids = page.get("route_ids")
        if (
            not isinstance(route_ids, Sequence)
            or isinstance(route_ids, (str, bytes))
            or not route_ids
        ):
            raise TypeError(
                "retained diagnostic page route_ids must be non-empty."
            )
        retained_page_body.append(
            " & ".join(
                (
                    _tex_text(page["pages"]),
                    _tex_text(page["label"]),
                    _tex_breakable_text("; ".join(str(item) for item in route_ids)),
                    _tex_breakable_text(provenance["path"]),
                    _tex_breakable_hash(provenance["sha256"]),
                )
            )
            + r" \\"
        )
    retained_page_fragment = ""
    if retained_page_body:
        retained_page_fragment = rf"""
\begin{{longtable}}{{@{{}}>{{\raggedright\arraybackslash}}p{{0.35in}}>{{\raggedright\arraybackslash}}p{{1.0in}}>{{\raggedright\arraybackslash}}p{{2.15in}}>{{\raggedright\arraybackslash}}p{{2.45in}}>{{\raggedright\arraybackslash}}p{{1.4in}}@{{}}}}
\toprule
Pages & Diagnostic & Routes & Provenance & SHA-256 \\
\midrule
{chr(10).join(retained_page_body)}
\bottomrule
\end{{longtable}}
""".strip()
    return rf"""
\section*{{Parameter manifest}}
\small
All energies use same-cutoff exact diagonalization. Reference, fidelity,
plotting, and compilation are reporting-only. Canonical SNAKE and Append rows
use logical-estimator accounting
\texttt{{{_tex_breakable_text(PAPER_I_S_ALG_ACCOUNTING_SCHEMA)}}} under
\texttt{{{_tex_breakable_text(PAPER_I_S_ALG_CONTRACT)}}}. Any historical Geo-ADAPT
support accounting is labelled noncanonical in its method contract.
Bosons use \texttt{{{_tex_text(encoding)}}} encoding; Pauli words use
\texttt{{{_tex_text(ordering)}}} ordering with
\texttt{{{_tex_text(boundary)}}} boundaries. Drive enabled:
\texttt{{{str(drive_enabled).lower()}}}. Same-cutoff reference:
\texttt{{{str(same_cutoff).lower()}}}.

\vspace{{4pt}}
\scriptsize
\setlength{{\tabcolsep}}{{2.6pt}}
\begin{{longtable}}{{@{{}}llrrrrrrrr@{{}}}}
\toprule
Reg. & Method & $L$ & $t$ & $U$ & $\Delta v$ & $\omega_0$ & $g$
& $n_{{\rm ph}}$ & $E_{{\rm exact}}$ \\
\midrule
{chr(10).join(manifest_body)}
\bottomrule
\end{{longtable}}

\begin{{longtable}}{{@{{}}>{{\raggedright\arraybackslash}}p{{0.9in}}>{{\raggedright\arraybackslash}}p{{1.65in}}>{{\raggedright\arraybackslash}}p{{4.7in}}@{{}}}}
\toprule
Method & Contract field & Value \\
\midrule
{chr(10).join(method_body)}
\bottomrule
\end{{longtable}}

\tiny
\begin{{longtable}}{{@{{}}llp{{2.55in}}p{{2.55in}}@{{}}}}
\toprule
Reg. & Method & Source artifact SHA-256 & Normalized contract SHA-256 \\
\midrule
{chr(10).join(source_body)}
\bottomrule
\end{{longtable}}

{retained_page_fragment}
""".strip()


def write_parameter_manifest_tex(
    *,
    parameter_manifest: Sequence[Mapping[str, Any]],
    tex_path: Path,
    retained_page_manifest: Sequence[Mapping[str, Any]] = (),
) -> None:
    fragment = _parameter_manifest_fragment(
        parameter_manifest,
        retained_page_manifest=retained_page_manifest,
    )
    tex = rf"""
\documentclass[10pt,letterpaper]{{article}}
\usepackage[margin=0.33in]{{geometry}}
\usepackage{{amsmath,booktabs,longtable,array}}
\makeatletter
\providecommand{{\vcenter@text}}{{\vcenter}}
\makeatother
\usepackage[T1]{{fontenc}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}
{fragment}
\end{{document}}
"""
    tex_path.write_text(tex.strip() + "\n", encoding="utf-8")


def write_tex(
    *,
    rows: list[dict[str, Any]],
    plateau_rows: list[dict[str, Any]],
    parameter_manifest: Sequence[Mapping[str, Any]],
    plateau_plot: Path,
    crossing_plot: Path,
    tex_path: Path,
) -> None:
    grouped = {(row["regime"], row["method"]): row for row in rows}
    plateau_grouped = {
        (row["regime"], row["method"]): row for row in plateau_rows
    }
    plateau_body: list[str] = []
    for regime, _title, abbreviation, _n_ph in REGIMES:
        for method in PAGE_ONE_METHODS:
            method_key = str(method["key"])
            row = plateau_grouped[(regime, method_key)]
            plateau_body.append(
                " & ".join(
                    [
                        abbreviation,
                        row["method_label"],
                        str(row["k_pl"]),
                        _tex_sci(row["error"]),
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
            plateau_body.append(r"\addlinespace[1.5pt]")
    body: list[str] = []
    for regime, _title, abbreviation, _n_ph in REGIMES:
        for method_key in ("snake", "append"):
            row = grouped[(regime, method_key)]
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
    manifest_fragment = _parameter_manifest_fragment(parameter_manifest)
    tex = rf"""
\documentclass[10pt,letterpaper]{{article}}
\usepackage[margin=0.33in]{{geometry}}
\usepackage{{amsmath,graphicx,booktabs,xcolor,longtable,array}}
\makeatletter
\providecommand{{\vcenter@text}}{{\vcenter}}
\makeatother
\usepackage[T1]{{fontenc}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}
\begin{{center}}
{{\Large\bfseries Macro-generator plateau-prefix comparison}}\par
\vspace{{2pt}}
{{\small Each trajectory terminates at the selected plateau prefix whose
compiled circuit resources and cumulative logical estimator work are reported
below. The panel-specific horizontal range includes a 15\% blank margin beyond
the latest displayed prefix.}}\par
\vspace{{5pt}}
\includegraphics[width=7.75in]{{{plateau_plot.as_posix()}}}
\vspace{{3pt}}
\scriptsize
\setlength{{\tabcolsep}}{{3.4pt}}
\renewcommand{{\arraystretch}}{{0.94}}
\begin{{tabular}}{{@{{}}cl r c rrrrr@{{}}}}
\toprule
Reg. & Method & $k_{{\rm pl}}$ & $\Delta E(k_{{\rm pl}})$
& $N_{{2q}}$ & $D_{{2q}}$ & $D_c$ & $W_{{1q}}$ & $S_{{\rm alg}}$ \\
\midrule
{chr(10).join(plateau_body)}
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
logical estimator work, not physical shots or circuits. Panel tuples follow
${PAPER_I_QISKIT_COST_TUPLE_LATEX}$. Both methods use intact macro generators.
\end{{minipage}}
\end{{center}}
\newpage
\begin{{center}}
{{\Large\bfseries Macro-generator costs at shared pre-plateau accuracy}}\par
\vspace{{2pt}}
{{\small For each regime, $K_\cap$ is the earlier selected plateau and
$\Delta E_\cap=\max\{{\min_{{k\leq K_\cap}}\Delta E_{{\mathrm{{RA\text{{-}}ADAPT}}}}(k),
\min_{{k\leq K_\cap}}\Delta E_{{\rm Append}}(k)\}}$.
Each cost is evaluated at that method's first stored prefix satisfying
$\Delta E(k)\leq\Delta E_\cap$ within this shared window. Each displayed
trajectory terminates at that costed prefix.}}\par
\vspace{{5pt}}
\includegraphics[width=7.75in]{{{crossing_plot.as_posix()}}}
\end{{center}}
\clearpage
{manifest_fragment}
\end{{document}}
"""
    tex_path.write_text(tex.strip() + "\n", encoding="utf-8")


def build(*, tracker_path: Path, output_dir: Path) -> Path:
    tracker = json.loads(tracker_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(tracker)
    plateau_rows = collect_plateau_rows(tracker)
    parameter_manifest = collect_parameter_manifest(tracker)
    page_one_plot_path = output_dir / f"{STEM}_page1_plot.png"
    page_two_plot_path = output_dir / f"{STEM}_page2_plot.png"
    tex_path = output_dir / f"{BASE_REPORT_STEM}.tex"
    pdf_path = output_dir / f"{BASE_REPORT_STEM}.pdf"
    provenance_path = output_dir / f"{BASE_REPORT_STEM}_provenance.json"
    make_plateau_plot(
        tracker=tracker,
        rows=plateau_rows,
        path=page_one_plot_path,
    )
    make_crossing_plot(tracker=tracker, rows=rows, path=page_two_plot_path)
    write_tex(
        rows=rows,
        plateau_rows=plateau_rows,
        parameter_manifest=parameter_manifest,
        plateau_plot=page_one_plot_path,
        crossing_plot=page_two_plot_path,
        tex_path=tex_path,
    )
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
    two_page_base = output_dir / f"{STEM}_pages1_2.pdf"
    _write_reader_pages_copy(
        source_pdf=pdf_path,
        target_pdf=two_page_base,
    )
    provenance = {
        "schema": "paper_i_hh_macro_common_accuracy_comparison_v3",
        "definition": (
            "Per regime, K_cap is the earlier selected plateau. DeltaE_cap is "
            "the larger of the minimum SNAKE and Append-ADAPT errors over "
            "stored prefixes k <= K_cap; costs use each method's first crossing "
            "within that shared window."
        ),
        "geo_adapt_excluded": True,
        "tracker": {
            "path": str(tracker_path.relative_to(REPO_ROOT)),
            "sha256": sha256(tracker_path),
        },
        "display_policy": {
            "curve_end": "reported cost prefix",
            "x_padding_fraction": DISPLAY_PADDING,
            "x_limit_rule": "ceil((1 + padding) * latest displayed prefix)",
            "source_histories_truncated": False,
            "tuple_font_size_pt": 6.8,
            "tuple_s_alg_format": "two-significant-digit compact e notation",
            "panel_identifiers": False,
            "in_panel_n_ph_labels": False,
        },
        "plateau_rows": plateau_rows,
        "rows": rows,
        "parameter_manifest": parameter_manifest,
        "evidence_status": _evidence_status(),
        "generated": {
            "page_one_plot_png": {
                "path": str(page_one_plot_path.relative_to(REPO_ROOT)),
                "sha256": sha256(page_one_plot_path),
            },
            "page_two_plot_png": {
                "path": str(page_two_plot_path.relative_to(REPO_ROOT)),
                "sha256": sha256(page_two_plot_path),
            },
            "tex": {
                "path": str(tex_path.relative_to(REPO_ROOT)),
                "sha256": sha256(tex_path),
            },
            "pdf": {
                "path": str(pdf_path.relative_to(REPO_ROOT)),
                "sha256": sha256(pdf_path),
            },
            "pages1_2_pdf": {
                "path": str(two_page_base.relative_to(REPO_ROOT)),
                "sha256": sha256(two_page_base),
            },
        },
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", type=Path, default=TRACKER)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    pdf = build(
        tracker_path=args.tracker.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(pdf)


if __name__ == "__main__":
    main()
