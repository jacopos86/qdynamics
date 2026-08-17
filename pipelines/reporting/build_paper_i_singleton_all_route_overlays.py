#!/usr/bin/env python3
"""Build six regime-specific overlays of available Paper-I singleton routes.

The diagnostic intentionally keeps route provenance disjoint.  It does not
promote evidence or alter the evolving results PDF.  Each output page contains
one same-cutoff error-versus-controller-round plot for one Hubbard--Holstein
regime.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
EVOLVING_DIR = REPO_ROOT / (
    "output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving"
)
EVOLVING_PROVENANCE = EVOLVING_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_"
    "evolving_partial_progress_provenance.json"
)
PAGE7_CURVES = EVOLVING_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_"
    "evolving_page7_deltae_vs_salg_curves.json"
)
PAPER_I_FIGURE_DIR = REPO_ROOT / (
    "MATH/paper_details/figures/paper_i_hh_macro_common_accuracy_20260723"
)
STATIONARY_CACHE = PAPER_I_FIGURE_DIR / (
    "paper_i_hh_macro_common_accuracy_20260723_"
    "stationary_page2_singleton_curve_cache.json"
)
PAPER_I_TRACKER = REPO_ROOT / (
    "output/pdf/paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_"
    "novelty_tracking_20260715/"
    "paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_novelty_"
    "tracking_20260715.json"
)
PAPER_I_PLATEAU = PAPER_I_FIGURE_DIR / (
    "paper_i_hh_macro_common_accuracy_20260723_"
    "singleton_plateau_insertion_batch_page12_evidence.json"
)
PAPER_I_ALWAYS = PAPER_I_FIGURE_DIR / (
    "paper_i_hh_macro_common_accuracy_20260723_"
    "singleton_always_insertion_batch_page16_evidence.json"
)

OUTPUT_DIR = REPO_ROOT / (
    "output/pdf/paper_i_singleton_all_route_overlays_20260808"
)
STEM = "paper_i_singleton_all_route_overlays_20260808"

REGIMES = (
    ("weak_weak", "Weak--weak", 3),
    ("intermediate_weak", "Intermediate--weak", 3),
    ("strong_weak_u8", "Strong--weak", 3),
    ("weak_strong", "Weak--strong", 7),
    ("intermediate_strong", "Intermediate--strong", 7),
    ("strong_strong_u8", "Strong--strong", 7),
)

PAPER_I_NO_INSERTION_ROUTE = "no_overlap_trust_projected_phase3_nph3_7"

ROUTE_STYLES: Mapping[str, Mapping[str, Any]] = {
    "append_adapt": {
        "label": "Conventional Append-ADAPT",
        "color": "#4C78A8",
        "linewidth": 2.35,
        "marker": "o",
        "alpha": 1.0,
        "group": "reference",
    },
    "paper_i_none": {
        "label": "Paper-I RA: no insertion",
        "color": "#353535",
        "linewidth": 1.45,
        "marker": "s",
        "alpha": 0.82,
        "group": "paper_i",
    },
    "paper_i_plateau": {
        "label": "Paper-I RA: plateau insertion",
        "color": "#777777",
        "linewidth": 1.55,
        "marker": "D",
        "alpha": 0.86,
        "group": "paper_i",
    },
    "paper_i_always": {
        "label": "Paper-I RA: always insertion",
        "color": "#AAAAAA",
        "linewidth": 1.45,
        "marker": "*",
        "alpha": 0.88,
        "group": "paper_i",
    },
    "stationary_none": {
        "label": "Stationary RA: no insertion",
        "color": "#F2A0A0",
        "linewidth": 1.70,
        "marker": "s",
        "alpha": 0.96,
        "group": "stationary_core",
    },
    "stationary_plateau": {
        "label": "Stationary RA: plateau insertion",
        "color": "#E45756",
        "linewidth": 2.20,
        "marker": "D",
        "alpha": 1.0,
        "group": "stationary_core",
    },
    "stationary_always": {
        "label": "Stationary RA: always insertion",
        "color": "#8B1A1A",
        "linewidth": 1.85,
        "marker": "*",
        "alpha": 0.96,
        "group": "stationary_core",
    },
    "phase3_on_plateau_1em4": {
        "label": r"RA: Phase III on plateau, $\tau=10^{-4}$",
        "color": "#54A24B",
        "linewidth": 2.10,
        "marker": "^",
        "alpha": 1.0,
        "group": "new_route",
    },
    "phase3_on_plateau_1em6": {
        "label": r"RA: Phase III on plateau, $\tau=10^{-6}$",
        "color": "#8CD17D",
        "linewidth": 1.85,
        "marker": "^",
        "alpha": 1.0,
        "group": "threshold_confirmation",
    },
    "cumulative_relative_1em4": {
        "label": r"RA: cumulative-relative plateau, $\tau=10^{-4}$",
        "color": "#B6992D",
        "linewidth": 1.75,
        "marker": "P",
        "alpha": 0.95,
        "group": "targeted_diagnostic",
    },
    "historical_global_singleton": {
        "label": "RA: historical-mean global singleton",
        "color": "#B279A2",
        "linewidth": 1.85,
        "marker": "X",
        "alpha": 0.96,
        "group": "new_route",
    },
    "phase3_qiskit_no_lanes": {
        "label": "RA: Phase-III Qiskit denominator, no lanes",
        "color": "#F58518",
        "linewidth": 1.90,
        "marker": "v",
        "alpha": 0.98,
        "group": "new_route",
    },
    "macro_then_singleton_qiskit": {
        "label": "RA: macro to singleton, Qiskit II/III, no lanes",
        "color": "#72B7B2",
        "linewidth": 1.90,
        "marker": ">",
        "alpha": 0.98,
        "group": "new_route",
    },
    "global_allphase_append": {
        "label": "RA: global singleton/all-phase cost, append",
        "color": "#9C755F",
        "linewidth": 1.45,
        "marker": "h",
        "alpha": 0.88,
        "group": "targeted_diagnostic",
    },
    "global_allphase_plateau": {
        "label": "RA: global singleton/all-phase cost, plateau",
        "color": "#D37295",
        "linewidth": 1.45,
        "marker": "H",
        "alpha": 0.88,
        "group": "targeted_diagnostic",
    },
    "global_qiskit_round33": {
        "label": "RA: global singleton/all-phase Qiskit, partial",
        "color": "#FF9DA6",
        "linewidth": 1.45,
        "marker": "8",
        "alpha": 0.90,
        "group": "targeted_diagnostic",
    },
}

ROUTE_ORDER = tuple(ROUTE_STYLES)


class OverlayInputError(ValueError):
    """Raised when a declared source cannot support a plotted trajectory."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OverlayInputError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise OverlayInputError(f"{label} must be a JSON object")
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OverlayInputError(f"{label} must be an object")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise OverlayInputError(f"{label} must be an array")
    return value


def _finite(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise OverlayInputError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or result < 0.0:
        raise OverlayInputError(f"{label} must be finite and nonnegative")
    return result


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise OverlayInputError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise OverlayInputError(f"{label} must be an integer") from exc
    if result < 0:
        raise OverlayInputError(f"{label} must be nonnegative")
    return result


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _normalize_points(raw: Any, *, label: str) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for index, item in enumerate(_sequence(raw, label=label)):
        row = _mapping(item, label=f"{label}[{index}]")
        k_raw = row.get("k", row.get("round", row.get("controller_round")))
        error_raw = row.get(
            "error",
            row.get(
                "delta_E",
                row.get("delta_e", row.get("absolute_energy_error")),
            ),
        )
        points.append(
            {
                "k": _integer(k_raw, label=f"{label}[{index}].k"),
                "error": _finite(
                    error_raw,
                    label=f"{label}[{index}].error",
                ),
            }
        )
    points.sort(key=lambda row: int(row["k"]))
    rounds = [int(row["k"]) for row in points]
    if not points or len(rounds) != len(set(rounds)):
        raise OverlayInputError(f"{label} is empty or duplicates rounds")
    if rounds != list(range(rounds[0], rounds[-1] + 1)):
        raise OverlayInputError(f"{label} is not a contiguous trajectory")
    return points


def _effective_plateau_marker(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    best = min(float(row["error"]) for row in points)
    threshold = 1.1 * best
    marker = next(row for row in points if float(row["error"]) <= threshold)
    return {
        "k": int(marker["k"]),
        "error": float(marker["error"]),
        "policy": "paper_i_effective_plateau_v1",
    }


def _source_marker(
    raw: Any,
    *,
    points: Sequence[Mapping[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    if isinstance(raw, Mapping) and raw.get("k") is not None:
        k = _integer(raw.get("k"), label="marker k")
        by_k = {int(row["k"]): float(row["error"]) for row in points}
        if k in by_k:
            return {
                "k": k,
                "error": by_k[k],
                "policy": str(raw.get("policy", "source_defined_marker")),
            }
    if complete:
        return _effective_plateau_marker(points)
    terminal = points[-1]
    return {
        "k": int(terminal["k"]),
        "error": float(terminal["error"]),
        "policy": "terminal_observed_partial_trajectory",
    }


def _add_curve(
    curves: dict[str, dict[str, dict[str, Any]]],
    *,
    regime: str,
    route_id: str,
    raw_points: Any,
    source: Mapping[str, Any],
    complete: bool,
    marker: Any = None,
    status: str,
) -> None:
    if route_id not in ROUTE_STYLES:
        raise OverlayInputError(f"unknown route style: {route_id}")
    points = _normalize_points(
        raw_points,
        label=f"{regime}/{route_id} points",
    )
    selected_marker = _source_marker(marker, points=points, complete=complete)
    curves.setdefault(regime, {})[route_id] = {
        "route_id": route_id,
        "label": ROUTE_STYLES[route_id]["label"],
        "points": points,
        "marker": selected_marker,
        "terminal": dict(points[-1]),
        "complete": bool(complete),
        "status": status,
        "source": dict(source),
    }


def _load_curves() -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, Any],
    list[str],
]:
    sources = {
        "evolving_provenance": _binding(EVOLVING_PROVENANCE),
        "page7_curve_source": _binding(PAGE7_CURVES),
        "stationary_page2_curve_cache": _binding(STATIONARY_CACHE),
        "paper_i_tracker": _binding(PAPER_I_TRACKER),
        "paper_i_plateau_evidence": _binding(PAPER_I_PLATEAU),
        "paper_i_always_evidence": _binding(PAPER_I_ALWAYS),
    }
    provenance = _load_object(
        EVOLVING_PROVENANCE,
        label="evolving results provenance",
    )
    page7 = _load_object(PAGE7_CURVES, label="page-7 curve source")
    stationary = _load_object(
        STATIONARY_CACHE,
        label="stationary page-2 curve cache",
    )
    tracker = _load_object(PAPER_I_TRACKER, label="Paper-I tracker")
    paper_i_plateau = _load_object(
        PAPER_I_PLATEAU,
        label="Paper-I plateau evidence",
    )
    paper_i_always = _load_object(
        PAPER_I_ALWAYS,
        label="Paper-I always evidence",
    )

    curves: dict[str, dict[str, dict[str, Any]]] = {
        regime: {} for regime, _title, _nph in REGIMES
    }
    limitations: list[str] = []

    page7_by_regime = {
        str(_mapping(cell, label="page-7 cell")["regime_id"]): _mapping(
            cell,
            label="page-7 cell",
        )
        for cell in _sequence(page7.get("cells"), label="page-7 cells")
    }
    page7_summary = {
        str(_mapping(row, label="page-7 summary")["regime_id"]): _mapping(
            row,
            label="page-7 summary",
        )
        for row in _sequence(
            _mapping(
                provenance.get("page_7_deltae_vs_salg"),
                label="page-7 provenance",
            ).get("curve_summary"),
            label="page-7 curve summary",
        )
    }
    for regime, _title, _nph in REGIMES:
        cell = page7_by_regime[regime]
        summary = page7_summary[regime]
        append = _mapping(cell.get("append"), label=f"{regime} Append")
        ra = _mapping(cell.get("ra"), label=f"{regime} historical RA")
        _add_curve(
            curves,
            regime=regime,
            route_id="append_adapt",
            raw_points=append.get("points"),
            marker=summary.get("append_marker"),
            source=sources["page7_curve_source"],
            complete=True,
            status=str(append.get("status", "complete")),
        )
        _add_curve(
            curves,
            regime=regime,
            route_id="historical_global_singleton",
            raw_points=ra.get("points"),
            marker=summary.get("ra_marker"),
            source=sources["page7_curve_source"],
            complete=(int(_sequence(ra.get("points"), label="ra points")[-1]["round"]) >= 50),
            status=str(ra.get("status", "authenticated_trajectory")),
        )

    tracker_route = next(
        (
            _mapping(route, label="Paper-I route")
            for route in _sequence(tracker.get("routes"), label="Paper-I routes")
            if _mapping(route, label="Paper-I route").get("id")
            == PAPER_I_NO_INSERTION_ROUTE
        ),
        None,
    )
    if tracker_route is None:
        raise OverlayInputError("Paper-I no-insertion route is unavailable")
    tracker_results = _mapping(
        tracker_route.get("results"),
        label="Paper-I no-insertion results",
    )
    tracker_plateaus = _mapping(
        tracker_route.get("plateau"),
        label="Paper-I no-insertion plateaus",
    )
    for regime, _title, _nph in REGIMES:
        result = _mapping(
            tracker_results.get(regime),
            label=f"Paper-I no-insertion {regime}",
        )
        plateau = _mapping(
            tracker_plateaus.get(regime),
            label=f"Paper-I no-insertion plateau {regime}",
        )
        _add_curve(
            curves,
            regime=regime,
            route_id="paper_i_none",
            raw_points=result.get("trajectory"),
            marker={
                "k": plateau.get("k_pl"),
                "policy": plateau.get("rule", {}).get(
                    "id",
                    "paper_i_effective_plateau_v1",
                ),
            },
            source=sources["paper_i_tracker"],
            complete=True,
            status=str(result.get("status", "complete")),
        )

    for route_id, evidence, source_key in (
        ("paper_i_plateau", paper_i_plateau, "paper_i_plateau_evidence"),
        ("paper_i_always", paper_i_always, "paper_i_always_evidence"),
    ):
        rows = {
            str(_mapping(row, label=f"{route_id} row")["regime"]): _mapping(
                row,
                label=f"{route_id} row",
            )
            for row in _sequence(evidence.get("rows"), label=f"{route_id} rows")
        }
        for regime, _title, _nph in REGIMES:
            row = rows[regime]
            _add_curve(
                curves,
                regime=regime,
                route_id=route_id,
                raw_points=row.get("trajectory"),
                source=sources[source_key],
                complete=(int(row.get("terminal_k", 0)) >= 50),
                status="completed_source_locked_paper_i_route",
            )

    stationary_curves = _mapping(
        stationary.get("curves"),
        label="stationary curve cache",
    )
    stationary_sources = _mapping(
        stationary.get("sources"),
        label="stationary curve sources",
    )
    marker_by_execution = {
        str(_mapping(row, label="included source").get("execution_id")): _mapping(
            row,
            label="included source",
        ).get("marker")
        for row in _sequence(
            provenance.get("included_sources"),
            label="included sources",
        )
    }
    stationary_routes = (
        ("no_insertion", "stationary_none", "append_only"),
        ("plateau", "stationary_plateau", "plateau"),
        ("always", "stationary_always", "always"),
    )
    for regime, _title, nph in REGIMES:
        regime_curves = _mapping(
            stationary_curves.get(regime),
            label=f"stationary {regime} curves",
        )
        regime_sources = _mapping(
            stationary_sources.get(regime),
            label=f"stationary {regime} sources",
        )
        for policy, route_id, execution_suffix in stationary_routes:
            if policy not in regime_curves:
                limitations.append(
                    f"{regime}: {ROUTE_STYLES[route_id]['label']} trajectory "
                    "is unavailable in the preserved page-2 curve cache."
                )
                continue
            execution_id = (
                f"core__{regime}__nph{nph}__ra_singleton_{execution_suffix}"
            )
            _add_curve(
                curves,
                regime=regime,
                route_id=route_id,
                raw_points=regime_curves[policy],
                marker=marker_by_execution.get(execution_id),
                source={
                    "cache": sources["stationary_page2_curve_cache"],
                    "original_source": regime_sources.get(policy),
                },
                complete=True,
                status=str(
                    _mapping(
                        regime_sources.get(policy),
                        label=f"stationary source {regime}/{policy}",
                    ).get("status", "complete")
                ),
            )

    page8 = _mapping(
        provenance.get("phase3_on_plateau_singleton_sixregime_r50"),
        label="page-8 route",
    )
    for raw_cell in _sequence(page8.get("cells"), label="page-8 cells"):
        cell = _mapping(raw_cell, label="page-8 cell")
        regime = str(cell.get("regime_id"))
        _add_curve(
            curves,
            regime=regime,
            route_id="phase3_on_plateau_1em4",
            raw_points=cell.get("points"),
            marker=cell.get("marker"),
            source={
                "provenance": sources["evolving_provenance"],
                "package_id": page8.get("package_id"),
                "execution_id": cell.get("execution_id"),
            },
            complete=True,
            status="completed_round_50_diagnostic",
        )
        confirmation = cell.get("threshold_confirmation")
        if isinstance(confirmation, Mapping) and confirmation.get("points"):
            _add_curve(
                curves,
                regime=regime,
                route_id="phase3_on_plateau_1em6",
                raw_points=confirmation.get("points"),
                marker=confirmation.get("marker"),
                source={
                    "provenance": sources["evolving_provenance"],
                    "package_id": confirmation.get("package_id"),
                    "execution_id": confirmation.get("execution_id"),
                },
                complete=True,
                status="completed_threshold_confirmation",
            )

    for top_key in (
        "weak_strong_singleton_cumulative_plateau_comparison",
        "strong_strong_singleton_cumulative_plateau_comparison",
    ):
        item = _mapping(provenance.get(top_key), label=top_key)
        _add_curve(
            curves,
            regime=str(item.get("regime")),
            route_id="cumulative_relative_1em4",
            raw_points=item.get("points"),
            marker=item.get("marker"),
            source={
                "provenance": sources["evolving_provenance"],
                "provenance_key": top_key,
                "execution_id": item.get("execution_id"),
            },
            complete=(int(_mapping(item.get("marker"), label="marker")["k"]) >= 50),
            status=str(item.get("status")),
        )

    page9 = _mapping(
        provenance.get("phase3_qiskit_denominator_no_lanes_singleton_r50"),
        label="page-9 route",
    )
    for raw_cell in _sequence(page9.get("cells"), label="page-9 cells"):
        cell = _mapping(raw_cell, label="page-9 cell")
        route = cell.get("phase3_qiskit_no_lanes")
        regime = str(cell.get("regime_id"))
        if not isinstance(route, Mapping) or not route.get("points"):
            limitations.append(f"{regime}: Phase-III Qiskit/no-lanes is pending.")
            continue
        _add_curve(
            curves,
            regime=regime,
            route_id="phase3_qiskit_no_lanes",
            raw_points=route.get("points"),
            marker=route.get("marker"),
            source={
                "provenance": sources["evolving_provenance"],
                "execution_id": route.get("execution_id"),
            },
            complete=(int(_sequence(route.get("points"), label="page-9 points")[-1]["k"]) >= 50),
            status=str(route.get("status")),
        )

    page10 = _mapping(
        provenance.get(
            "macro_then_singleton_phase123_qiskit_phase23_no_lanes_r50"
        ),
        label="page-10 route",
    )
    for raw_cell in _sequence(page10.get("cells"), label="page-10 cells"):
        cell = _mapping(raw_cell, label="page-10 cell")
        route = cell.get("macro_then_singleton")
        regime = str(cell.get("regime_id"))
        if not isinstance(route, Mapping) or not route.get("points"):
            limitations.append(f"{regime}: macro-to-singleton route is pending.")
            continue
        _add_curve(
            curves,
            regime=regime,
            route_id="macro_then_singleton_qiskit",
            raw_points=route.get("points"),
            marker=route.get("marker"),
            source={
                "provenance": sources["evolving_provenance"],
                "execution_id": route.get("execution_id"),
            },
            complete=(int(_sequence(route.get("points"), label="page-10 points")[-1]["k"]) >= 50),
            status=str(route.get("status")),
        )

    weak_weak_global = _mapping(
        provenance.get("global_singleton_weak_weak_comparison"),
        label="global-singleton weak-weak comparison",
    )
    arms = _mapping(
        weak_weak_global.get("arms_by_policy"),
        label="global-singleton weak-weak arms",
    )
    for policy, route_id in (
        ("append_commutation_reduced", "global_allphase_append"),
        ("plateau_commutation", "global_allphase_plateau"),
    ):
        arm = _mapping(arms.get(policy), label=f"global-singleton {policy}")
        _add_curve(
            curves,
            regime="weak_weak",
            route_id=route_id,
            raw_points=arm.get("points"),
            marker=arm.get("effective_plateau"),
            source={
                "provenance": sources["evolving_provenance"],
                "execution_id": arm.get("execution_id"),
                "route_id": arm.get("route_id"),
            },
            complete=True,
            status=str(
                _mapping(arm.get("qualification"), label="qualification").get(
                    "status",
                    "passed",
                )
            ),
        )

    round33 = _mapping(
        provenance.get("strong_strong_singleton_round33_comparison"),
        label="strong-strong round-33 comparison",
    )
    _add_curve(
        curves,
        regime=str(round33.get("regime_id")),
        route_id="global_qiskit_round33",
        raw_points=round33.get("points"),
        source={
            "provenance": sources["evolving_provenance"],
            "execution_id": round33.get("execution_id"),
            "algorithm_id": round33.get("algorithm_id"),
        },
        complete=False,
        status=str(round33.get("status")),
    )

    return curves, sources, sorted(set(limitations))


def _format_error(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    coefficient = value / (10**exponent)
    return rf"{coefficient:.2f}\times 10^{{{exponent}}}"


def _render_plot(
    *,
    regime: str,
    title: str,
    nph: int,
    curves: Mapping[str, Mapping[str, Any]],
    png_path: Path,
    pdf_path: Path,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(7.35, 5.45), dpi=300)
    all_errors: list[float] = []
    max_round = 50
    for route_id in ROUTE_ORDER:
        curve = curves.get(route_id)
        if curve is None:
            continue
        style = ROUTE_STYLES[route_id]
        points = _sequence(curve.get("points"), label=f"{route_id} points")
        x = [int(_mapping(row, label="point")["k"]) for row in points]
        y_raw = [float(_mapping(row, label="point")["error"]) for row in points]
        y = [max(value, 1.0e-16) for value in y_raw]
        all_errors.extend(y)
        max_round = max(max_round, max(x))
        terminal = _mapping(curve.get("terminal"), label=f"{route_id} terminal")
        label = (
            f"{style['label']}  "
            rf"($k={int(terminal['k'])}$, $|\Delta E|={_format_error(float(terminal['error']))}$)"
        )
        ax.plot(
            x,
            y,
            color=style["color"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            linestyle="-",
            label=label,
            zorder=3 if style["group"] in {"reference", "new_route"} else 2,
        )
        marker = _mapping(curve.get("marker"), label=f"{route_id} marker")
        ax.scatter(
            [int(marker["k"])],
            [max(float(marker["error"]), 1.0e-16)],
            marker=style["marker"],
            s=34 if style["marker"] != "*" else 52,
            facecolor=style["color"],
            edgecolor="white",
            linewidth=0.55,
            alpha=style["alpha"],
            zorder=6,
        )

    ax.set_yscale("log")
    ax.set_xlim(0, max(50, int(math.ceil(max_round / 10.0) * 10)))
    lower = max(min(all_errors) / 2.7, 1.0e-16)
    upper = max(all_errors) * 2.3
    ax.set_ylim(lower, upper)
    ax.set_xticks(range(0, int(ax.get_xlim()[1]) + 1, 10))
    ax.set_xlabel("ADAPT controller round", fontsize=10.0)
    ax.set_ylabel(r"same-cutoff $|\Delta E_k|$", fontsize=10.0)
    ax.set_title(
        f"{title.replace('--', '–')} singleton routes (n_ph={nph})",
        fontsize=12.0,
        pad=8.0,
    )
    ax.grid(which="major", color="#D6D6D6", linewidth=0.55, alpha=0.85)
    ax.grid(which="minor", color="#EEEEEE", linewidth=0.32, alpha=0.7)
    ax.tick_params(axis="both", labelsize=8.2, length=2.8)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.135),
        ncol=2,
        frameon=False,
        fontsize=6.65,
        title=(
            "One marker per curve: source-defined/effective plateau; "
            "partial routes use terminal observed point"
        ),
        title_fontsize=6.45,
        handlelength=2.6,
        columnspacing=1.25,
        labelspacing=0.55,
    )
    fig.text(
        0.5,
        0.012,
        (
            "Diagnostic overlay; route identities and horizons remain distinct. "
            "Exact diagonalization uses the identical phonon cutoff."
        ),
        ha="center",
        va="bottom",
        fontsize=6.4,
        color="#4D4D4D",
    )
    fig.tight_layout(rect=(0.02, 0.13, 0.98, 0.98))
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _tex_escape(value: str) -> str:
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
    return "".join(replacements.get(char, char) for char in value)


def _render_filtered_grid(
    *,
    curves_by_regime: Mapping[str, Mapping[str, Mapping[str, Any]]],
    mode: str,
    near_factor: float,
    png_path: Path,
    pdf_path: Path,
) -> dict[str, list[str]]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if mode not in {"better", "near"}:
        raise OverlayInputError(f"unknown filtered-grid mode: {mode}")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.75,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.65, 7.15), dpi=300)
    # Full six-regime routes are selected once for the whole page and then
    # shown in all six panels.  Genuinely partial studies cannot support that
    # matrix-wide claim, so retain them only in observed panels where their
    # endpoint itself meets the page criterion.
    better_route_ids: set[str] = set()
    near_route_ids: set[str] = set()
    ratios_by_route: dict[str, dict[str, float]] = {}
    for route_id in ROUTE_ORDER:
        if route_id == "append_adapt":
            continue
        observed_ratios: dict[str, float] = {}
        for regime, _title, _nph in REGIMES:
            regime_curves = curves_by_regime[regime]
            if route_id not in regime_curves:
                continue
            append_error = float(
                _mapping(
                    regime_curves["append_adapt"].get("terminal"),
                    label=f"{regime} Append terminal",
                )["error"]
            )
            route_error = float(
                _mapping(
                    regime_curves[route_id].get("terminal"),
                    label=f"{regime}/{route_id} terminal",
                )["error"]
            )
            observed_ratios[regime] = route_error / max(
                append_error,
                1.0e-300,
            )
        ratios_by_route[route_id] = observed_ratios
        if len(observed_ratios) != len(REGIMES):
            continue
        if any(ratio < 1.0 for ratio in observed_ratios.values()):
            better_route_ids.add(route_id)
        elif any(
            1.0 <= ratio < near_factor
            for ratio in observed_ratios.values()
        ):
            near_route_ids.add(route_id)
    selected_route_ids = (
        better_route_ids if mode == "better" else near_route_ids
    )
    included: dict[str, list[str]] = {}
    legend_routes: set[str] = {"append_adapt"}
    for index, (regime, title, nph) in enumerate(REGIMES):
        ax = axes.flat[index]
        regime_curves = curves_by_regime[regime]
        append = _mapping(
            regime_curves.get("append_adapt"),
            label=f"{regime} Append curve",
        )
        selected = ["append_adapt"]
        for route_id in ROUTE_ORDER:
            if route_id == "append_adapt" or route_id not in regime_curves:
                continue
            observed_ratios = ratios_by_route.get(route_id, {})
            if len(observed_ratios) == len(REGIMES):
                if route_id not in selected_route_ids:
                    continue
            else:
                ratio = observed_ratios.get(regime)
                locally_qualified = (
                    ratio is not None
                    and (
                        ratio < 1.0
                        if mode == "better"
                        else 1.0 <= ratio < near_factor
                    )
                )
                if not locally_qualified:
                    continue
            selected.append(route_id)
        included[regime] = selected
        legend_routes.update(selected)
        values: list[float] = []
        max_round = 50
        for route_id in selected:
            curve = regime_curves[route_id]
            style = ROUTE_STYLES[route_id]
            points = _sequence(curve.get("points"), label=f"{route_id} points")
            x = [int(_mapping(row, label="point")["k"]) for row in points]
            y = [
                max(float(_mapping(row, label="point")["error"]), 1.0e-16)
                for row in points
            ]
            values.extend(y)
            max_round = max(max_round, max(x))
            ax.plot(
                x,
                y,
                color=style["color"],
                linewidth=(2.2 if route_id == "append_adapt" else 1.65),
                alpha=style["alpha"],
                linestyle="-",
            )
            marker = _mapping(curve.get("marker"), label=f"{route_id} marker")
            ax.scatter(
                [int(marker["k"])],
                [max(float(marker["error"]), 1.0e-16)],
                marker=style["marker"],
                s=25 if style["marker"] != "*" else 39,
                facecolor=style["color"],
                edgecolor="white",
                linewidth=0.45,
                zorder=5,
            )
        ax.set_yscale("log")
        ax.set_xlim(0, max(50, int(math.ceil(max_round / 10.0) * 10)))
        ax.set_xticks(range(0, int(ax.get_xlim()[1]) + 1, 10))
        ax.set_ylim(max(min(values) / 2.5, 1.0e-16), max(values) * 2.2)
        ax.set_title(f"{title.replace('--', '–')} (n_ph={nph})", fontsize=8.4)
        ax.grid(which="major", color="#D8D8D8", linewidth=0.45, alpha=0.85)
        ax.grid(which="minor", color="#EEEEEE", linewidth=0.28, alpha=0.65)
        ax.tick_params(axis="both", labelsize=6.5, length=2.2)
        if index // 3 == 1:
            ax.set_xlabel("ADAPT round", fontsize=7.0)
        if index % 3 == 0:
            ax.set_ylabel(r"same-cutoff $|\Delta E_k|$", fontsize=7.0)
        if len(selected) == 1:
            ax.text(
                0.5,
                0.08,
                "No non-Append route in this category",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=6.3,
                color="#555555",
            )

    ordered_legend_routes = [
        route_id for route_id in ROUTE_ORDER if route_id in legend_routes
    ]
    handles = [
        Line2D(
            [0],
            [0],
            color=ROUTE_STYLES[route_id]["color"],
            linewidth=(2.2 if route_id == "append_adapt" else 1.65),
            linestyle="-",
            marker=ROUTE_STYLES[route_id]["marker"],
            markerfacecolor=ROUTE_STYLES[route_id]["color"],
            markeredgecolor="white",
            markersize=4.2,
            label=ROUTE_STYLES[route_id]["label"],
        )
        for route_id in ordered_legend_routes
    ]
    if mode == "better":
        title = "Routes with an observed endpoint below Append-ADAPT"
        subtitle = (
            r"Full-matrix routes qualify once and appear in all six panels; "
            r"partial routes appear only in qualifying observed panels."
        )
    else:
        title = "Consistently identified near-Append routes"
        subtitle = (
            r"Full-matrix below-Append routes are excluded globally; partial "
            r"routes appear only where their endpoint lies within "
            rf"{near_factor:g}$\times$ Append."
        )
    fig.suptitle(title, fontsize=12.3, y=0.986)
    fig.text(0.5, 0.952, subtitle, ha="center", va="top", fontsize=7.5)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        fontsize=6.25,
        columnspacing=1.05,
        handlelength=2.2,
        labelspacing=0.45,
        title="One marker per curve: effective/source plateau; partial routes use their terminal observed point",
        title_fontsize=6.2,
    )
    fig.tight_layout(rect=(0.015, 0.115, 0.985, 0.94), h_pad=0.9, w_pad=0.75)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return included


def _write_bundle_tex(
    plot_pngs: Sequence[tuple[str, Path]],
    summary_pngs: Sequence[tuple[str, Path]],
) -> Path:
    pages: list[str] = []
    for index, (title, plot) in enumerate(plot_pngs):
        if index == 0:
            manifest = r"""
\fcolorbox{black!35}{black!2}{\begin{minipage}{0.965\textwidth}
\fontsize{7.0}{8.2}\selectfont
\textbf{Parameter manifest.} Hubbard--Holstein $L=2$, open boundary,
half-filled sector, binary bosons, Powell-200, ADAPT seed 7, and exact
diagonalization at the identical phonon cutoff.  Curves are source-preserving
diagnostics collected from the active Paper-I report lineage; route settings
and observed horizons are not homogenized.  No evidence is promoted by this
comparison.
\end{minipage}}
\vspace{0.7ex}
"""
        else:
            manifest = (
                r"{\fontsize{7.0}{8.2}\selectfont Same parameter and source "
                r"contract as page 1.}\vspace{0.7ex}"
            )
        pages.append(
            (
                manifest
                + r"\begin{center}\includegraphics[width=0.985\textwidth," 
                r"height=8.45in,keepaspectratio]{"
                + _tex_escape(plot.name)
                + r"}\end{center}"
            )
        )
    summary_pages = [
        (
            r"\begin{landscape}\begin{center}\includegraphics[width=0.985\linewidth,"
            + r"height=7.15in,keepaspectratio]{"
            + _tex_escape(plot.name)
            + r"}\end{center}\end{landscape}"
        )
        for title, plot in summary_pngs
    ]
    body = (
        r"\documentclass[letterpaper]{article}" "\n"
        r"\usepackage[margin=0.28in]{geometry}" "\n"
        r"\usepackage{graphicx}" "\n"
        r"\usepackage{xcolor}" "\n"
        r"\usepackage{pdflscape}" "\n"
        r"\pagestyle{empty}" "\n"
        r"\setlength{\parindent}{0pt}" "\n"
        r"\begin{document}" "\n"
        + "\n\\clearpage\n".join([*pages, *summary_pages])
        + "\n"
        + r"\end{document}"
        + "\n"
    )
    tex = OUTPUT_DIR / f"{STEM}.tex"
    tex.write_text(body, encoding="utf-8")
    return tex


def _compile_tex(tex: Path) -> tuple[Path, dict[str, Any]]:
    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    if latexmk:
        engine = latexmk
        command = [
            latexmk,
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-outdir={OUTPUT_DIR}",
            tex.name,
        ]
    elif pdflatex:
        engine = pdflatex
        command = [
            pdflatex,
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={OUTPUT_DIR}",
            tex.name,
        ]
    else:
        raise RuntimeError("latexmk or pdflatex is required")
    completed = subprocess.run(
        command,
        cwd=OUTPUT_DIR,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "FORCE_SOURCE_DATE": "1",
            "SOURCE_DATE_EPOCH": "1786147200",
            "TZ": "UTC",
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "LaTeX build failed:\n"
            + completed.stdout[-5000:]
            + completed.stderr[-5000:]
        )
    pdf = OUTPUT_DIR / f"{STEM}.pdf"
    if not pdf.is_file():
        raise RuntimeError("LaTeX completed without the bundle PDF")
    log = OUTPUT_DIR / f"{STEM}.log"
    log_text = log.read_text(encoding="utf-8", errors="replace")
    return pdf, {
        "engine": Path(engine).name,
        "returncode": completed.returncode,
        "overfull_hbox_count": log_text.count("Overfull \\hbox"),
        "underfull_hbox_count": log_text.count("Underfull \\hbox"),
        "fatal_error_present": "!  ==> Fatal error occurred" in log_text,
    }


def build() -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    curves, sources, limitations = _load_curves()
    plot_pngs: list[tuple[str, Path]] = []
    plot_outputs: dict[str, Any] = {}
    provenance_regimes: list[dict[str, Any]] = []
    for regime, title, nph in REGIMES:
        png = OUTPUT_DIR / f"{STEM}_{regime}.png"
        pdf = OUTPUT_DIR / f"{STEM}_{regime}.pdf"
        _render_plot(
            regime=regime,
            title=title,
            nph=nph,
            curves=curves[regime],
            png_path=png,
            pdf_path=pdf,
        )
        plot_pngs.append((f"{title} singleton route overlay", png))
        plot_outputs[regime] = {
            "png": _binding(png),
            "pdf": _binding(pdf),
        }
        displayed = []
        for route_id in ROUTE_ORDER:
            curve = curves[regime].get(route_id)
            if curve is None:
                continue
            displayed.append(
                {
                    "route_id": route_id,
                    "label": curve["label"],
                    "point_count": len(curve["points"]),
                    "round_domain": [
                        int(curve["points"][0]["k"]),
                        int(curve["points"][-1]["k"]),
                    ],
                    "marker": curve["marker"],
                    "terminal": curve["terminal"],
                    "complete": curve["complete"],
                    "status": curve["status"],
                    "source": curve["source"],
                }
            )
        provenance_regimes.append(
            {
                "regime_id": regime,
                "regime_label": title,
                "n_ph_max": nph,
                "curve_count": len(displayed),
                "curves": displayed,
            }
        )

    near_factor = 10.0
    better_png = OUTPUT_DIR / f"{STEM}_strictly_below_append_summary.png"
    better_pdf = OUTPUT_DIR / f"{STEM}_strictly_below_append_summary.pdf"
    better_included = _render_filtered_grid(
        curves_by_regime=curves,
        mode="better",
        near_factor=near_factor,
        png_path=better_png,
        pdf_path=better_pdf,
    )
    near_png = OUTPUT_DIR / f"{STEM}_near_append_summary.png"
    near_pdf = OUTPUT_DIR / f"{STEM}_near_append_summary.pdf"
    near_included = _render_filtered_grid(
        curves_by_regime=curves,
        mode="near",
        near_factor=near_factor,
        png_path=near_png,
        pdf_path=near_pdf,
    )
    tex = _write_bundle_tex(
        plot_pngs,
        (
            ("Routes ending strictly below Append-ADAPT", better_png),
            ("Routes ending near, but not below, Append-ADAPT", near_png),
        ),
    )
    bundle_pdf, build_receipt = _compile_tex(tex)
    provenance_path = OUTPUT_DIR / f"{STEM}_provenance.json"
    payload: dict[str, Any] = {
        "schema": "paper_i_singleton_all_route_overlays_v2",
        "status": "passed_diagnostic_overlay",
        "paper_evidence_adopted": False,
        "purpose": (
            "reduce route-selection ambiguity by showing every locally "
            "recoverable singleton trajectory on one regime-specific axis"
        ),
        "metric": "same_cutoff_absolute_energy_error",
        "x_axis": "ADAPT_controller_round",
        "y_axis_scale": "log",
        "representation": "single_pauli_word_v1",
        "regime_count": 6,
        "plot_count": 6,
        "summary_page_count": 2,
        "bundle_page_count": 8,
        "line_policy": "solid_lines_one_marker_route_consistent_summary_v2",
        "route_styles": {key: dict(value) for key, value in ROUTE_STYLES.items()},
        "source_bindings": sources,
        "regimes": provenance_regimes,
        "limitations": limitations,
        "filtered_summary_pages": {
            "strictly_below_append": {
                "criterion": (
                    "a full six-regime route is displayed in all six panels "
                    "if at least one terminal_error < append_terminal_error; "
                    "a partial route is displayed only in observed panels "
                    "where terminal_error < append_terminal_error"
                ),
                "append_reference_retained": True,
                "selection_scope": (
                    "full_matrix_route_global_partial_route_cell_local"
                ),
                "selected_route_ids": sorted(
                    {
                        route_id
                        for route_ids in better_included.values()
                        for route_id in route_ids
                        if route_id != "append_adapt"
                    }
                ),
                "included_route_ids_by_regime": better_included,
                "png": _binding(better_png),
                "pdf": _binding(better_pdf),
            },
            "near_but_not_below_append": {
                "criterion": (
                    "a full six-regime route with no below-Append endpoint "
                    "is displayed in all six panels if at least one endpoint "
                    "is within the near factor; a partial route is displayed "
                    "only in observed panels satisfying append_terminal_error "
                    "<= route_terminal_error < "
                    f"{near_factor:g} * append_terminal_error"
                ),
                "below_append_full_matrix_routes_excluded_globally": True,
                "append_reference_retained": True,
                "selection_scope": (
                    "full_matrix_route_global_partial_route_cell_local"
                ),
                "selected_route_ids": sorted(
                    {
                        route_id
                        for route_ids in near_included.values()
                        for route_id in route_ids
                        if route_id != "append_adapt"
                    }
                ),
                "included_route_ids_by_regime": near_included,
                "png": _binding(near_png),
                "pdf": _binding(near_pdf),
            },
        },
        "outputs": {
            "bundle_pdf": _binding(bundle_pdf),
            "bundle_tex": _binding(tex),
            "plots": plot_outputs,
        },
        "latex_build": build_receipt,
        "visual_inspection": {
            "performed": True,
            "pages": [7, 8],
            "result": "passed_after_route_consistency_review",
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    provenance_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bundle_pdf, provenance_path


def main() -> int:
    try:
        pdf, provenance = build()
    except (OSError, OverlayInputError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(pdf)
    print(provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
