#!/usr/bin/env python3
"""Append a fourth singleton page using method-specific plateau windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from pypdf import PdfReader, PdfWriter


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting.add_paper_i_hh_singleton_common_accuracy_page import (  # noqa: E402
    FINAL_PDF,
    FINAL_STEM,
    METHODS,
    OUTPUT_DIR,
    PROVENANCE,
    REGIMES,
    TRACKER,
    _style_axes,
    compile_tex,
    collect_rows as collect_singleton_common_rows,
)
from pipelines.reporting.build_paper_i_hh_macro_common_accuracy_pdf import (  # noqa: E402
    METHODS as MACRO_METHODS,
    _compile_comparator_at_k,
    _display_x_max,
    _evidence_status as _macro_evidence_status,
    _existing_prefix,
    _math_compact_sci,
    _parameter_manifest_row,
    _required_mapping,
    _tex_sci,
    collect_parameter_manifest,
    collect_rows as collect_macro_common_rows,
    sha256,
    write_parameter_manifest_tex,
)
from pipelines.reporting.build_paper_i_hh_tracking_plateau_costs import (  # noqa: E402
    _read_source_result,
    _snake_prefix,
)
from pipelines.reporting.paper_i_qiskit_cost_tuple import (  # noqa: E402
    PAPER_I_QISKIT_COST_TUPLE_LATEX,
    paper_i_cost_tuple_latex,
)


MACRO_TWO_PAGE_PDF = OUTPUT_DIR / f"{FINAL_STEM}_pages1_2.pdf"
PAST_RESULTS_PAGES_1_3 = OUTPUT_DIR / f"{FINAL_STEM}_past_results_pages1_3.pdf"
PAST_RESULTS_PAGE_4 = OUTPUT_DIR / f"{FINAL_STEM}_past_results_page4.pdf"
PAGE_STEM = f"{FINAL_STEM}_singleton_page4_own_plateau"
ACTIVE_THREE_PAGE_STEM = f"{FINAL_STEM}_active3"
CORRECTED_REVIEW4_STEM = f"{FINAL_STEM}_review4_s_alg_corrected"
CORRECTED_REVIEW4_TEX = OUTPUT_DIR / f"{CORRECTED_REVIEW4_STEM}.tex"
CORRECTED_REVIEW4_PDF = OUTPUT_DIR / f"{CORRECTED_REVIEW4_STEM}.pdf"
REVIEW6_TEX = OUTPUT_DIR / f"{FINAL_STEM}_review6.tex"
REVIEW6_PROVENANCE = OUTPUT_DIR / f"{FINAL_STEM}_review6_provenance.json"
PRESERVED_INSERTION_REVIEW_PDF = OUTPUT_DIR / f"{FINAL_STEM}_review9.pdf"
PRESERVED_SINGLETON_INSERTION_PAGE_PDF = (
    OUTPUT_DIR
    / f"{FINAL_STEM}_singleton_plateau_insertion_page10.pdf"
)
MACRO_INSERTION_TRAJECTORY_PROVENANCE = (
    OUTPUT_DIR
    / f"{FINAL_STEM}_macro_commutation_reduced_insertion_provenance.json"
)
MACRO_INSERTION_COST_PROVENANCE = (
    OUTPUT_DIR
    / f"{FINAL_STEM}_macro_insertion_cost_provenance.json"
)
SINGLETON_INSERTION_PAGE_PROVENANCE = (
    OUTPUT_DIR
    / f"{FINAL_STEM}_singleton_plateau_insertion_page10_provenance.json"
)
PARAMETER_MANIFEST_TEX = OUTPUT_DIR / f"{FINAL_STEM}_parameter_manifest.tex"
PARAMETER_MANIFEST_PDF = OUTPUT_DIR / f"{FINAL_STEM}_parameter_manifest.pdf"
MACRO_COMPARISON_OUTPUT_DIR = (
    REPO_ROOT
    / "MATH/paper_details/figures/paper_i_hh_macro_comparison_20260723"
)
MACRO_COMPARISON_PNG_NAME = "paper_i_hh_macro_comparison_20260723.png"
MACRO_COMPARISON_PNG = (
    MACRO_COMPARISON_OUTPUT_DIR / MACRO_COMPARISON_PNG_NAME
)
GEO_METHOD = {
    "key": "geo",
    "route_id": "geo_adapt_macro_nph3_7",
    "label": "Geo-ADAPT",
}


def _repo_source_path(raw_path: Any) -> Path:
    relative = Path(str(raw_path))
    if relative.is_absolute():
        raise ValueError("retained diagnostic sources must be repo-relative.")
    resolved = (REPO_ROOT / relative).resolve()
    if not resolved.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("retained diagnostic source escapes the repository.")
    return resolved


def _authenticated_json(
    receipt: Mapping[str, Any],
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _repo_source_path(receipt.get("path"))
    expected_sha = str(receipt.get("sha256") or "")
    if len(expected_sha) != 64:
        raise ValueError(f"{label} lacks an authenticated SHA-256.")
    observed_sha = sha256(path)
    if observed_sha != expected_sha:
        raise ValueError(
            f"{label} hash drift: expected {expected_sha}, got "
            f"{observed_sha} for {path}."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object.")
    return payload, {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": observed_sha,
    }


def _route_contract_parameter_row(
    *,
    regime: str,
    method_key: str,
    method_label: str,
    route_id: str,
    result_payload: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
    compile_policy: Mapping[str, Any],
) -> dict[str, Any]:
    settings = _required_mapping(
        result_payload.get("settings"),
        name=f"{regime} retained diagnostic settings",
    )
    route_contract = _required_mapping(
        settings.get("sr_route_profile_contract"),
        name=f"{regime} retained diagnostic route contract",
    )
    route_profile = str(
        settings.get("sr_route_profile_resolved")
        or settings.get("route_profile")
        or route_contract.get("route_profile")
        or ""
    )
    if not route_profile:
        raise ValueError(f"{regime} retained diagnostic route is unidentified.")
    contract_profile = str(route_contract.get("route_profile") or "")
    if contract_profile and contract_profile != route_profile:
        raise ValueError(
            f"{regime} retained diagnostic route-profile drift."
        )
    adapt_vqe = _required_mapping(
        result_payload.get("adapt_vqe"),
        name=f"{regime} retained diagnostic ADAPT result",
    )
    exact_energy = float(adapt_vqe["exact_gs_energy"])
    benchmark_reference = float(
        adapt_vqe["benchmark_stop_reference_energy"]
    )
    if not (
        math.isfinite(exact_energy)
        and math.isclose(
            exact_energy,
            benchmark_reference,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(
            f"{regime} retained diagnostic same-cutoff reference drift."
        )
    physics = {
        "problem": "hh",
        "L": int(settings["L"]),
        "t": float(settings["t"]),
        "u_over_t": float(settings["u"]) / float(settings["t"]),
        "dv": float(settings["dv"]),
        "omega0": float(settings["omega0"]),
        "g_ep": float(settings["g_ep"]),
        "n_ph_work": int(settings["n_ph_max"]),
        "boson_encoding": str(settings["boson_encoding"]),
        "ordering": str(settings["ordering"]),
        "boundary": str(settings["boundary"]),
        "drive_enabled": False,
        "same_cutoff_reference": True,
        "expected_exact_energy": exact_energy,
    }
    normalized_contract = {
        "schema": "paper_i_hh_authenticated_result_route_contract_v1",
        "family": "hh",
        "physics": physics,
        "route_identity": {
            "profile_resolved": route_profile,
            "profile_contract": route_contract,
        },
        "exact_reference": {
            "energy": exact_energy,
            "usage": "reporting_only_after_optimization",
        },
    }
    contract_raw = json.dumps(
        route_contract,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    projected_receipt = dict(source_receipt)
    projected_receipt.update(
        {
            "manifest_member": "settings.sr_route_profile_contract",
            "manifest_member_sha256": hashlib.sha256(contract_raw).hexdigest(),
        }
    )
    return _parameter_manifest_row(
        regime=regime,
        method={
            "key": method_key,
            "label": method_label,
            "route_id": route_id,
        },
        normalized_manifest=normalized_contract,
        source_receipt=projected_receipt,
        compile_policy=compile_policy,
    )


def _compile_contract_matches(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    return all(
        observed.get(key) == expected.get(key)
        for key in (
            "identity",
            "optimization_level",
            "seed_transpiler",
            "reference_state_included",
        )
    )


def collect_retained_diagnostic_parameter_manifest(
    tracker: Mapping[str, Any],
    *,
    tracker_path: Path = TRACKER,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Authenticate and project the run contracts behind retained pages 7--10."""

    compile_policy = _required_mapping(
        tracker.get("plateau_compile_policy"),
        name="tracker plateau_compile_policy",
    )
    macro_trajectory = json.loads(
        MACRO_INSERTION_TRAJECTORY_PROVENANCE.read_text(encoding="utf-8")
    )
    macro_cost = json.loads(
        MACRO_INSERTION_COST_PROVENANCE.read_text(encoding="utf-8")
    )
    singleton_page = json.loads(
        SINGLETON_INSERTION_PAGE_PROVENANCE.read_text(encoding="utf-8")
    )
    if (
        macro_trajectory.get("schema")
        != "paper_i_hh_macro_commutation_reduced_insertion_diagnostic_v1"
    ):
        raise ValueError("unexpected macro-insertion trajectory provenance.")
    if (
        macro_cost.get("schema")
        != "paper_i_hh_macro_insertion_cost_comparison_v3"
    ):
        raise ValueError("unexpected macro-insertion cost provenance.")
    if (
        singleton_page.get("schema")
        != "paper_i_hh_singleton_plateau_insertion_page_v1"
        or singleton_page.get("classification") != "diagnostic"
    ):
        raise ValueError("unexpected singleton-insertion page provenance.")
    tracker_path = tracker_path.resolve()
    tracker_sha = sha256(tracker_path)

    def validate_tracker_receipt(
        receipt: Mapping[str, Any],
        *,
        label: str,
    ) -> None:
        source_path = _repo_source_path(receipt.get("path"))
        if source_path != tracker_path:
            raise ValueError(f"{label} tracker path drift.")
        if str(receipt.get("sha256") or "") != tracker_sha:
            raise ValueError(f"{label} tracker hash drift.")

    validate_tracker_receipt(
        _required_mapping(
            _required_mapping(
                macro_trajectory.get("inputs"),
                name="macro-insertion trajectory inputs",
            ).get("tracker"),
            name="macro-insertion trajectory tracker receipt",
        ),
        label="macro-insertion trajectory",
    )
    validate_tracker_receipt(
        _required_mapping(
            _required_mapping(
                macro_cost.get("inputs"),
                name="macro-insertion cost inputs",
            ).get("tracker"),
            name="macro-insertion cost tracker receipt",
        ),
        label="macro-insertion cost",
    )

    routes = _required_mapping(
        macro_cost.get("routes"),
        name="macro-insertion routes",
    )
    expected_routes = {
        "append_adapt": "append_adapt_macro_nph3_7",
        "append_only_snake": "sr_macro_physical_lanes_nph3_7",
        "insertion": "sr_macro_commutation_reduced_insertion_nph3_7",
    }
    if dict(routes) != expected_routes:
        raise ValueError("macro-insertion route identity drift.")
    macro_generated = _required_mapping(
        macro_cost.get("generated"),
        name="macro-insertion generated artifacts",
    )
    preserved_review_receipt = _required_mapping(
        macro_generated.get("review_pdf"),
        name="macro-insertion nine-page review receipt",
    )
    preserved_review_path = _repo_source_path(
        preserved_review_receipt.get("path")
    )
    if preserved_review_path != PRESERVED_INSERTION_REVIEW_PDF.resolve():
        raise ValueError("preserved insertion-review path drift.")
    if (
        int(preserved_review_receipt.get("pages") or 0) != 9
        or sha256(preserved_review_path)
        != preserved_review_receipt.get("sha256")
        or len(PdfReader(str(preserved_review_path), strict=False).pages) != 9
    ):
        raise ValueError("preserved insertion-review PDF hash/page drift.")
    trajectory_comparison = _required_mapping(
        macro_trajectory.get("comparison"),
        name="macro-insertion trajectory comparison",
    )
    if (
        trajectory_comparison.get("append_adapt_route")
        != routes["append_adapt"]
        or trajectory_comparison.get("baseline_route")
        != routes["append_only_snake"]
    ):
        raise ValueError("macro-insertion trajectory baseline drift.")

    trajectory_rows = _required_mapping(
        {"rows": trajectory_comparison.get("rows")},
        name="macro-insertion trajectory rows",
    )["rows"]
    terminal_rows = macro_cost.get("terminal_insertion_rows")
    if not isinstance(trajectory_rows, list) or not isinstance(
        terminal_rows, list
    ):
        raise TypeError("macro-insertion provenance rows must be arrays.")
    trajectory_sources = {
        str(row["regime"]): dict(
            _required_mapping(
                row.get("result_json"),
                name="macro-insertion trajectory result receipt",
            )
        )
        for row in trajectory_rows
    }
    terminal_by_regime = {
        str(row["regime"]): _required_mapping(
            row,
            name="macro-insertion terminal row",
        )
        for row in terminal_rows
    }
    expected_regimes = {row[0] for row in REGIMES}
    if (
        set(trajectory_sources) != expected_regimes
        or set(terminal_by_regime) != expected_regimes
    ):
        raise ValueError("macro-insertion provenance lacks all six regimes.")

    rows: list[dict[str, Any]] = []
    for regime, _title, _abbreviation, expected_n_ph in REGIMES:
        terminal = terminal_by_regime[regime]
        terminal_source = _required_mapping(
            terminal.get("source"),
            name=f"{regime} macro-insertion source",
        )
        trajectory_source = trajectory_sources[regime]
        if any(
            terminal_source.get(key) != trajectory_source.get(key)
            for key in ("path", "sha256")
        ):
            raise ValueError(
                f"{regime} macro-insertion source receipt drift."
            )
        result_payload, source_receipt = _authenticated_json(
            terminal_source,
            label=f"{regime} macro-insertion result",
        )
        qiskit_compile = _required_mapping(
            terminal.get("qiskit_compile"),
            name=f"{regime} macro-insertion compile contract",
        )
        if not _compile_contract_matches(qiskit_compile, compile_policy):
            raise ValueError(
                f"{regime} macro-insertion compile-contract drift."
            )
        row = _route_contract_parameter_row(
            regime=regime,
            method_key="macro_insertion",
            method_label="Macro insertion RA-ADAPT",
            route_id=str(routes["insertion"]),
            result_payload=result_payload,
            source_receipt=source_receipt,
            compile_policy=compile_policy,
        )
        if row["physics"]["n_ph_max"] != expected_n_ph:
            raise ValueError(
                f"{regime} macro-insertion phonon-cutoff drift."
            )
        rows.append(row)

    singleton_sources = _required_mapping(
        singleton_page.get("sources"),
        name="singleton-insertion page sources",
    )
    validate_tracker_receipt(
        _required_mapping(
            singleton_sources.get("tracker"),
            name="singleton-insertion tracker receipt",
        ),
        label="singleton-insertion page",
    )
    singleton_result, singleton_receipt = _authenticated_json(
        _required_mapping(
            singleton_sources.get("insertion_current"),
            name="singleton-insertion result receipt",
        ),
        label="singleton-insertion result",
    )
    singleton_settings = _required_mapping(
        singleton_result.get("settings"),
        name="singleton-insertion settings",
    )
    singleton_route_id = str(
        singleton_settings.get("sr_route_profile_resolved") or ""
    )
    if not singleton_route_id:
        raise ValueError("singleton-insertion route profile is absent.")
    singleton_row = _route_contract_parameter_row(
        regime="weak_weak",
        method_key="singleton_plateau_insertion",
        method_label="Singleton plateau-insertion RA-ADAPT",
        route_id=singleton_route_id,
        result_payload=singleton_result,
        source_receipt=singleton_receipt,
        compile_policy=compile_policy,
    )
    singleton_data = _required_mapping(
        singleton_page.get("data"),
        name="singleton-insertion page data",
    )
    if not math.isclose(
        float(singleton_data["exact_energy"]),
        float(singleton_row["physics"]["exact_gs_energy"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("singleton-insertion page exact-energy drift.")
    rows.append(singleton_row)

    generated_page = _required_mapping(
        _required_mapping(
            singleton_page.get("generated"),
            name="singleton-insertion generated artifacts",
        ).get("page_pdf"),
        name="singleton-insertion page PDF receipt",
    )
    page_pdf_path = _repo_source_path(generated_page.get("path"))
    if page_pdf_path != PRESERVED_SINGLETON_INSERTION_PAGE_PDF.resolve():
        raise ValueError("singleton-insertion page path drift.")
    if sha256(page_pdf_path) != generated_page.get("sha256"):
        raise ValueError("singleton-insertion page PDF hash drift.")
    if len(PdfReader(str(page_pdf_path), strict=False).pages) != 1:
        raise ValueError("singleton-insertion page PDF is not one page.")

    def provenance_receipt(path: Path) -> dict[str, Any]:
        return {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": sha256(path),
        }

    all_macro_routes = [
        str(routes["append_adapt"]),
        str(routes["append_only_snake"]),
        str(routes["insertion"]),
    ]
    retained_pages = [
        {
            "pages": "7",
            "label": "Macro insertion trajectory",
            "route_ids": all_macro_routes,
            "provenance": provenance_receipt(
                MACRO_INSERTION_TRAJECTORY_PROVENANCE
            ),
        },
        {
            "pages": "8--9",
            "label": "Macro insertion costs",
            "route_ids": all_macro_routes,
            "provenance": provenance_receipt(
                MACRO_INSERTION_COST_PROVENANCE
            ),
        },
        {
            "pages": "10",
            "label": "Singleton plateau insertion",
            "route_ids": [
                METHODS[0]["route_id"],
                METHODS[1]["route_id"],
                singleton_route_id,
            ],
            "provenance": provenance_receipt(
                SINGLETON_INSERTION_PAGE_PROVENANCE
            ),
        },
    ]
    return rows, retained_pages


def _closed_cached_rows(
    payload: Mapping[str, Any],
    *,
    section: str,
) -> list[dict[str, Any]] | None:
    raw_rows: Any
    if section == "rows":
        raw_rows = payload.get("rows")
    else:
        raw_section = payload.get(section)
        raw_rows = (
            raw_section.get("rows")
            if isinstance(raw_section, Mapping)
            else None
        )
    if not isinstance(raw_rows, list) or len(raw_rows) != 12:
        return None
    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            return None
        receipt = raw_row.get("S_alg_receipt")
        components = (
            receipt.get("components")
            if isinstance(receipt, Mapping)
            else None
        )
        if not isinstance(components, Mapping):
            return None
        if (
            raw_row.get("W1q") is None
            or raw_row.get("qiskit_basis_work_status") != "ok"
        ):
            return None
        try:
            scalar = int(raw_row["S_alg"])
            receipt_scalar = int(receipt["S_alg"])
            component_sum = sum(int(value) for value in components.values())
        except (KeyError, TypeError, ValueError):
            return None
        if scalar != receipt_scalar or scalar != component_sum:
            return None
        if not str(raw_row.get("S_alg_reconstruction_status") or "").startswith(
            "clean_algorithm_recount_closed_"
        ):
            return None
        rows.append(dict(raw_row))
    return rows


def collect_rows(tracker: Mapping[str, Any]) -> list[dict[str, Any]]:
    routes = {route["id"]: route for route in tracker["routes"]}
    selected = {method["key"]: routes[method["route_id"]] for method in METHODS}
    rows: list[dict[str, Any]] = []
    for regime, title, abbreviation, n_ph in REGIMES:
        trajectories = {
            key: selected[key]["results"][regime]["trajectory"]
            for key in selected
        }
        plateau_ends = {
            key: int(selected[key]["plateau"][regime]["k_pl"])
            for key in selected
        }
        minima = {
            key: min(
                float(point["error"])
                for point in trajectory
                if int(point["round"]) <= plateau_ends[key]
            )
            for key, trajectory in trajectories.items()
        }
        common_error = max(minima.values())
        crossings = {
            key: next(
                int(point["round"])
                for point in trajectory
                if int(point["round"]) <= plateau_ends[key]
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
                    fallback_source_kind=(
                        "paper_i_hh_snake_singleton_own_plateau_common_accuracy_prefix"
                    ),
                )
                recovery = "exact signed-checkpoint reconstruction and compile"
            qiskit = prefix["qiskit"]
            row = {
                "regime": regime,
                "regime_title": title,
                "abbreviation": abbreviation,
                "n_ph": n_ph,
                "method_plateau_k": plateau_ends[key],
                "snake_plateau_k": plateau_ends["snake_singleton"],
                "append_plateau_k": plateau_ends["append_singleton"],
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
                f"k_pl={plateau_ends[key]}, Ecap={common_error:.8e}, k={k}, "
                f"N2q={row['N2q']}, S_alg={row['S_alg']}",
                flush=True,
            )
    return rows


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
        values: list[float] = []
        for method in METHODS:
            trajectory = routes[method["route_id"]]["results"][regime]["trajectory"]
            row = row_lookup[(regime, method["key"])]
            visible = [
                point
                for point in trajectory
                if int(point["round"]) <= row["k_cross"]
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
                [row["k_cross"]],
                [row["crossing_error"]],
                color=method["color"],
                marker=method["marker"],
                s=58 if method["marker"] == "*" else 42,
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
                0.95 if method["key"] == "snake_singleton" else 0.87,
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
        common = row_lookup[(regime, "snake_singleton")]["common_error"]
        ax.axhline(common, color="#555555", linestyle=(0, (3, 2)), linewidth=0.85)
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
                        str(row["method_plateau_k"]),
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
{{\Large\bfseries RA-ADAPT--Append singleton costs at common attained accuracy}}\par
\vspace{{2pt}}
{{\small For each regime,
$\Delta E_\cap=\max\{{\min_{{k\leq k_{{\rm pl}}^{{\mathrm{{RA\text{{-}}ADAPT}}}}}}
\Delta E_{{\mathrm{{RA\text{{-}}ADAPT}}}}(k),
\min_{{k\leq k_{{\rm pl}}^{{\rm Append}}}}
\Delta E_{{\rm Append}}(k)\}}$.
Each cost is evaluated at that method's first stored prefix satisfying
$\Delta E(k)\leq\Delta E_\cap$ no later than its own selected plateau.
Each displayed trajectory terminates at that costed prefix.}}\par
\vspace{{5pt}}
\includegraphics[width=7.75in]{{{plot_path.as_posix()}}}
\end{{center}}
\end{{document}}
"""
    tex_path.write_text(tex.strip() + "\n", encoding="utf-8")


def ensure_macro_two_page_base() -> None:
    if not MACRO_TWO_PAGE_PDF.is_file():
        raise FileNotFoundError(
            f"missing active macro two-page source: {MACRO_TWO_PAGE_PDF}"
        )
    if len(PdfReader(str(MACRO_TWO_PAGE_PDF)).pages) != 2:
        raise RuntimeError("active macro source is not two pages")


def combine_active_pages(singleton_pdf: Path) -> tuple[Path, Path]:
    combined_tex = OUTPUT_DIR / f"{ACTIVE_THREE_PAGE_STEM}.tex"
    combined_tex.write_text(
        rf"""
\documentclass[letterpaper]{{article}}
\usepackage{{pdfpages}}
\pagestyle{{empty}}
\begin{{document}}
\includepdf[pages=-,pagecommand={{}}]{{{MACRO_TWO_PAGE_PDF.as_posix()}}}
\includepdf[pages=-,pagecommand={{}}]{{{singleton_pdf.as_posix()}}}
\end{{document}}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    combined_pdf = compile_tex(combined_tex, OUTPUT_DIR)
    if len(PdfReader(str(combined_pdf)).pages) != 3:
        raise RuntimeError("active comparison PDF is not three pages")
    return combined_tex, combined_pdf


def _review_table_body(
    rows: list[dict[str, Any]],
    *,
    window_field: str,
) -> str:
    body: list[str] = []
    for index, row in enumerate(rows):
        body.append(
            " & ".join(
                [
                    str(row["abbreviation"]),
                    str(row[window_field]),
                    _tex_sci(float(row["common_error"])),
                    str(row["method_label"]),
                    str(row["k_cross"]),
                    _tex_sci(float(row["crossing_error"])),
                    f"{int(row['N2q']):,}",
                    f"{int(row['D2q']):,}",
                    f"{int(row['Dc']):,}",
                    f"{int(row['W1q']):,}",
                    f"{int(row['S_alg']):,}",
                ]
            )
            + r" \\"
        )
        if index % 2 == 1 and index != len(rows) - 1:
            body.append(r"\addlinespace[1.5pt]")
    return "\n".join(body)


def _overlay_table_tex(
    rows: list[dict[str, Any]],
    *,
    window_field: str,
    window_heading: str,
) -> str:
    return rf"""
\resizebox{{319pt}}{{!}}{{%
\begin{{tabular}}{{@{{}}ccc l r c rrrrr@{{}}}}
\toprule
Reg. & {window_heading} & $\Delta E_\cap$ & Method
& $k_\cap$ & $\Delta E(k_\cap)$
& $N_{{2q}}$ & $D_{{2q}}$ & $D_c$ & $W_{{1q}}$ & $S_{{\rm alg}}$ \\
\midrule
{_review_table_body(rows, window_field=window_field)}
\bottomrule
\end{{tabular}}%
}}
""".strip()


def write_corrected_review4_tex(
    *,
    macro_rows: list[dict[str, Any]],
    singleton_common_rows: list[dict[str, Any]],
    singleton_own_rows: list[dict[str, Any]],
) -> None:
    for path in (
        PAST_RESULTS_PAGES_1_3,
        PAST_RESULTS_PAGE_4,
        MACRO_COMPARISON_PNG,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if len(PdfReader(str(PAST_RESULTS_PAGES_1_3)).pages) != 3:
        raise RuntimeError("preserved pages 1--3 source is not three pages")
    if len(PdfReader(str(PAST_RESULTS_PAGE_4)).pages) != 1:
        raise RuntimeError("preserved page-4 source is not one page")

    macro_table = _overlay_table_tex(
        macro_rows,
        window_field="common_window_end",
        window_heading=r"$K_\cap$",
    )
    singleton_common_table = _overlay_table_tex(
        singleton_common_rows,
        window_field="common_window_end",
        window_heading=r"$K_\cap$",
    )
    singleton_own_table = _overlay_table_tex(
        singleton_own_rows,
        window_field="method_plateau_k",
        window_heading=r"$k_{\rm pl}$",
    )
    CORRECTED_REVIEW4_TEX.write_text(
        rf"""
\documentclass[10pt,letterpaper]{{article}}
\usepackage[margin=0in]{{geometry}}
\usepackage{{amsmath,graphicx,booktabs,pdfpages,tikz}}
\usepackage[T1]{{fontenc}}
\usetikzlibrary{{calc}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}
\thispagestyle{{empty}}
\begin{{tikzpicture}}[remember picture,overlay]
\node[anchor=south west,inner sep=0pt] at
  ($ (current page.south west) + (24.479pt,59.223pt) $)
  {{\includegraphics[width=563.05pt]{{{MACRO_COMPARISON_PNG.as_posix()}}}}};
\end{{tikzpicture}}
\null
\clearpage
\includepdf[
  pages=2,
  fitpaper=true,
  noautoscale=true,
  pagecommand={{
    \begin{{tikzpicture}}[remember picture,overlay]
    \fill[white] ($ (current page.south west) + (143pt,238pt) $)
      rectangle ($ (current page.south west) + (469pt,380pt) $);
    \node[anchor=north west,inner sep=0pt] at
      ($ (current page.south west) + (146pt,377pt) $)
      {{\scriptsize {macro_table}}};
    \end{{tikzpicture}}
  }}
]{{{PAST_RESULTS_PAGES_1_3.as_posix()}}}
\includepdf[
  pages=3,
  fitpaper=true,
  noautoscale=true,
  pagecommand={{
    \begin{{tikzpicture}}[remember picture,overlay]
    \fill[white] ($ (current page.south west) + (143pt,238pt) $)
      rectangle ($ (current page.south west) + (469pt,380pt) $);
    \node[anchor=north west,inner sep=0pt] at
      ($ (current page.south west) + (146pt,377pt) $)
      {{\scriptsize {singleton_common_table}}};
    \end{{tikzpicture}}
  }}
]{{{PAST_RESULTS_PAGES_1_3.as_posix()}}}
\includepdf[
  pages=1,
  fitpaper=true,
  noautoscale=true,
  pagecommand={{
    \begin{{tikzpicture}}[remember picture,overlay]
    \fill[white] ($ (current page.south west) + (143pt,238pt) $)
      rectangle ($ (current page.south west) + (469pt,380pt) $);
    \node[anchor=north west,inner sep=0pt] at
      ($ (current page.south west) + (146pt,377pt) $)
      {{\scriptsize {singleton_own_table}}};
    \end{{tikzpicture}}
  }}
]{{{PAST_RESULTS_PAGE_4.as_posix()}}}
\end{{document}}
""".strip()
        + "\n",
        encoding="utf-8",
    )


def build_corrected_review4(
    *,
    macro_rows: list[dict[str, Any]],
    singleton_common_rows: list[dict[str, Any]],
    singleton_own_rows: list[dict[str, Any]],
) -> Path:
    write_corrected_review4_tex(
        macro_rows=macro_rows,
        singleton_common_rows=singleton_common_rows,
        singleton_own_rows=singleton_own_rows,
    )
    corrected_pdf = compile_tex(CORRECTED_REVIEW4_TEX, OUTPUT_DIR)
    if corrected_pdf != CORRECTED_REVIEW4_PDF:
        raise RuntimeError("unexpected corrected four-page review output path")
    if len(PdfReader(str(corrected_pdf)).pages) != 4:
        raise RuntimeError("corrected historical review is not four pages")
    return corrected_pdf


def combine_review6(*, corrected_review4_pdf: Path, singleton_pdf: Path) -> Path:
    if len(PdfReader(str(corrected_review4_pdf)).pages) != 4:
        raise RuntimeError("corrected review input is not four pages")
    if len(PdfReader(str(MACRO_TWO_PAGE_PDF)).pages) != 2:
        raise RuntimeError("macro source is not two pages")
    if len(PdfReader(str(singleton_pdf)).pages) != 1:
        raise RuntimeError("singleton source is not one page")
    REVIEW6_TEX.write_text(
        rf"""
\documentclass[letterpaper]{{article}}
\usepackage{{pdfpages}}
\pagestyle{{empty}}
\begin{{document}}
\includepdf[pages=-,pagecommand={{}}]{{{corrected_review4_pdf.as_posix()}}}
\includepdf[pages=2,pagecommand={{}}]{{{MACRO_TWO_PAGE_PDF.as_posix()}}}
\includepdf[pages=-,pagecommand={{}}]{{{singleton_pdf.as_posix()}}}
\end{{document}}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    review_pdf = compile_tex(REVIEW6_TEX, OUTPUT_DIR)
    if len(PdfReader(str(review_pdf)).pages) != 6:
        raise RuntimeError("corrected review PDF is not six pages")
    return review_pdf


def promote_canonical_pdf(
    *,
    review_pdf: Path,
    preserved_review_pdf: Path,
    preserved_terminal_page_pdf: Path,
    parameter_manifest_pdf: Path,
    final_pdf: Path,
) -> int:
    manifest_reader = PdfReader(str(parameter_manifest_pdf), strict=False)
    if not manifest_reader.pages:
        raise RuntimeError("parameter manifest input has no pages")

    if not preserved_review_pdf.is_file():
        raise FileNotFoundError(
            "canonical assembly requires the authenticated nine-page "
            f"insertion review: {preserved_review_pdf}"
        )
    if not preserved_terminal_page_pdf.is_file():
        raise FileNotFoundError(
            "canonical assembly requires the authenticated singleton "
            f"insertion page: {preserved_terminal_page_pdf}"
        )
    refreshed_reader = PdfReader(str(review_pdf), strict=False)
    preserved_reader = PdfReader(
        str(preserved_review_pdf),
        strict=False,
    )
    terminal_reader = PdfReader(
        str(preserved_terminal_page_pdf),
        strict=False,
    )
    if len(refreshed_reader.pages) != 6:
        raise RuntimeError("refreshed review input is not six pages")
    if len(preserved_reader.pages) != 9:
        raise RuntimeError("preserved insertion review is not nine pages")
    if len(terminal_reader.pages) != 1:
        raise RuntimeError(
            "preserved singleton insertion input is not one page"
        )

    writer = PdfWriter()
    for page in refreshed_reader.pages:
        writer.add_page(page)
    for page in preserved_reader.pages[6:]:
        writer.add_page(page)
    writer.add_page(terminal_reader.pages[0])
    for page in manifest_reader.pages:
        writer.add_page(page)

    temporary_pdf = final_pdf.with_name(f".{final_pdf.name}.tmp")
    with temporary_pdf.open("wb") as handle:
        writer.write(handle)
    try:
        validation = _validate_canonical_assembly(
            review_pdf=review_pdf,
            preserved_review_pdf=preserved_review_pdf,
            preserved_terminal_page_pdf=preserved_terminal_page_pdf,
            parameter_manifest_pdf=parameter_manifest_pdf,
            final_pdf=temporary_pdf,
        )
    except Exception:
        temporary_pdf.unlink(missing_ok=True)
        raise
    temporary_pdf.replace(final_pdf)
    return int(validation["page_count"])


def _page_content_sha256(page: Any) -> str:
    contents = page.get_contents()
    content_bytes = b"" if contents is None else contents.get_data()
    return hashlib.sha256(content_bytes).hexdigest()


def _page_box(page: Any) -> tuple[float, float, float, float]:
    box = page.mediabox
    return tuple(
        float(value)
        for value in (
            box.left,
            box.bottom,
            box.right,
            box.top,
        )
    )


def _validate_canonical_assembly(
    *,
    review_pdf: Path,
    preserved_review_pdf: Path,
    preserved_terminal_page_pdf: Path,
    parameter_manifest_pdf: Path,
    final_pdf: Path,
) -> dict[str, Any]:
    refreshed_reader = PdfReader(str(review_pdf), strict=False)
    preserved_reader = PdfReader(str(preserved_review_pdf), strict=False)
    terminal_reader = PdfReader(
        str(preserved_terminal_page_pdf),
        strict=False,
    )
    manifest_reader = PdfReader(str(parameter_manifest_pdf), strict=False)
    final_reader = PdfReader(str(final_pdf), strict=False)
    expected_pages = [
        *refreshed_reader.pages,
        *preserved_reader.pages[6:9],
        terminal_reader.pages[0],
        *manifest_reader.pages,
    ]
    if len(final_reader.pages) != len(expected_pages):
        raise RuntimeError(
            "canonical assembly page-count drift: expected "
            f"{len(expected_pages)}, got {len(final_reader.pages)}."
        )
    for page_number, (observed, expected) in enumerate(
        zip(final_reader.pages, expected_pages, strict=True),
        start=1,
    ):
        if _page_content_sha256(observed) != _page_content_sha256(expected):
            raise RuntimeError(
                "canonical assembly content-stream drift on page "
                f"{page_number}."
            )
        if _page_box(observed) != _page_box(expected):
            raise RuntimeError(
                f"canonical assembly page-box drift on page {page_number}."
            )
        if not (
            math.isclose(
                float(observed.mediabox.width),
                612.0,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
            and math.isclose(
                float(observed.mediabox.height),
                792.0,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
        ):
            raise RuntimeError(
                f"canonical assembly page {page_number} is not letter size."
            )
    return {
        "page_count": len(final_reader.pages),
        "page_size": "letter",
        "source_page_content_streams_verified": True,
        "source_page_boxes_verified": True,
        "source_page_map": {
            "1--6": "refreshed_review6",
            "7--9": "authenticated_review9_pages_7_through_9",
            "10": "authenticated_singleton_insertion_page",
            f"11--{len(expected_pages)}": "parameter_manifest",
        },
    }


def _latex_box_validation(*tex_paths: Path) -> dict[str, Any]:
    patterns = (
        r"Overfull \hbox",
        r"Overfull \vbox",
        r"Underfull \hbox",
        r"Underfull \vbox",
    )
    logs: list[dict[str, Any]] = []
    total = 0
    for tex_path in tex_paths:
        log_path = tex_path.with_suffix(".log")
        if not log_path.is_file():
            raise FileNotFoundError(
                f"missing LaTeX log for validation: {log_path}"
            )
        text = log_path.read_text(encoding="utf-8", errors="replace")
        count = sum(text.count(pattern) for pattern in patterns)
        total += count
        try:
            displayed_path = str(log_path.relative_to(REPO_ROOT))
        except ValueError:
            displayed_path = str(log_path)
        logs.append(
            {
                "path": displayed_path,
                "sha256": sha256(log_path),
                "overfull_or_underfull_boxes": count,
            }
        )
    if total:
        raise RuntimeError(
            f"generated LaTeX sources contain {total} overfull or "
            "underfull boxes."
        )
    return {
        "scope": "newly_compiled_latex_sources_in_this_build",
        "logs": logs,
        "overfull_or_underfull_boxes": total,
    }


def _replace_promotion_state_with_evidence_status(
    payload: dict[str, Any],
) -> None:
    payload.pop("manuscript_promotion", None)
    status = dict(_macro_evidence_status())
    status.update(
        {
            "artifact_role": "historical_macro_and_projected_singleton_report",
            "validation_state": "source_locked_prefix_rows",
            "projected_singleton_included": True,
            "final_parameter_manifest_appended": True,
            "retained_diagnostic_pages_manifested": True,
        }
    )
    payload["evidence_status"] = status


def _validated_geo_adapt_manifest_rows(
    parameter_manifest: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    geo_rows: list[Mapping[str, Any]] = []
    for raw_row in parameter_manifest:
        row = _required_mapping(
            raw_row,
            name="parameter-manifest row",
        )
        if str(row.get("method") or "") == "geo":
            geo_rows.append(row)
    expected_regimes = {row[0] for row in REGIMES}
    observed_regimes = {str(row.get("regime") or "") for row in geo_rows}
    if observed_regimes != expected_regimes:
        raise ValueError(
            "final report provenance requires one Geo-ADAPT manifest row "
            "for every displayed regime."
        )
    return geo_rows


def _scope_geo_adapt_page_roles(
    payload: dict[str, Any],
    *,
    parameter_manifest: Sequence[Mapping[str, Any]],
) -> None:
    _validated_geo_adapt_manifest_rows(parameter_manifest)
    payload.pop("geo_adapt_excluded", None)
    page_roles = dict(payload.get("method_page_roles") or {})
    page_roles["geo_adapt"] = {
        "artifact_included": True,
        "route_id": GEO_METHOD["route_id"],
        "historical_macro_comparison_page_1": "included",
        "clean_common_accuracy_pages": "excluded",
        "final_parameter_manifest": (
            "included_as_six_historical_noncanonical_rows"
        ),
        "accounting_status": "historical_noncanonical",
    }
    payload["method_page_roles"] = page_roles


def update_provenance(
    *,
    rows: list[dict[str, Any]],
    singleton_common_rows: list[dict[str, Any]],
    tracker_path: Path,
    plot_path: Path,
    page_tex: Path,
    page_pdf: Path,
    combined_tex: Path,
    combined_pdf: Path,
    corrected_review4_pdf: Path,
    review_pdf: Path,
    parameter_manifest: list[dict[str, Any]],
    retained_page_manifest: list[dict[str, Any]],
    parameter_manifest_tex: Path,
    parameter_manifest_pdf: Path,
    latex_box_validation: Mapping[str, Any],
) -> None:
    payload = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    verified_box_validation = _required_mapping(
        latex_box_validation,
        name="pre-promotion LaTeX box validation",
    )
    if int(verified_box_validation.get("overfull_or_underfull_boxes", -1)) != 0:
        raise ValueError(
            "provenance requires a closed zero-warning LaTeX box receipt."
        )
    totals: dict[str, dict[str, int]] = {}
    for method_key in ("snake_singleton", "append_singleton"):
        method_rows = [row for row in rows if row["method"] == method_key]
        totals[method_key] = {
            field: sum(int(row[field]) for row in method_rows)
            for field in ("N2q", "D2q", "Dc", "W1q", "S_alg")
        }
    payload["singleton_own_plateau_common_accuracy"] = {
        "schema": "paper_i_hh_singleton_own_plateau_common_accuracy_v2",
        "projected_phase_order": (
            "phase1_parent_shortlist_then_split_then_phase2_children_then_phase3"
        ),
        "parent_phase2_dead_work_included": False,
        "definition": (
            "Each method's comparison window ends at its own selected plateau. "
            "DeltaE_cap is the larger of the two within-window minima. Costs use "
            "each method's first crossing within its own window."
        ),
        "display_policy": {
            "curve_end": "reported cost prefix",
            "x_padding_fraction": 0.15,
            "x_limit_rule": "ceil((1 + padding) * latest displayed prefix)",
            "source_histories_truncated": False,
            "tuple_font_size_pt": 6.8,
            "tuple_s_alg_format": "two-significant-digit compact e notation",
            "panel_identifiers": False,
            "in_panel_n_ph_labels": False,
        },
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
    singleton_common_totals: dict[str, dict[str, int]] = {}
    for method_key in ("snake_singleton", "append_singleton"):
        method_rows = [
            row for row in singleton_common_rows if row["method"] == method_key
        ]
        singleton_common_totals[method_key] = {
            field: sum(int(row[field]) for row in method_rows)
            for field in ("N2q", "D2q", "Dc", "W1q", "S_alg")
        }
    payload["singleton_common_accuracy"] = {
        "schema": "paper_i_hh_singleton_common_accuracy_comparison_v2",
        "projected_phase_order": (
            "phase1_parent_shortlist_then_split_then_phase2_children_then_phase3"
        ),
        "parent_phase2_dead_work_included": False,
        "definition": (
            "Per regime, K_cap is the earlier selected singleton plateau. "
            "DeltaE_cap is the larger within-window minimum of SNAKE and "
            "projected-singleton Append-ADAPT. Costs use each method's first crossing."
        ),
        "route_ids": [method["route_id"] for method in METHODS],
        "rows": singleton_common_rows,
        "summed_over_six_regimes": singleton_common_totals,
        "generated": {
            "review4_tex": {
                "path": str(CORRECTED_REVIEW4_TEX.relative_to(REPO_ROOT)),
                "sha256": sha256(CORRECTED_REVIEW4_TEX),
            },
            "review4_pdf": {
                "path": str(corrected_review4_pdf.relative_to(REPO_ROOT)),
                "sha256": sha256(corrected_review4_pdf),
            },
        },
    }
    _replace_promotion_state_with_evidence_status(payload)
    _scope_geo_adapt_page_roles(
        payload,
        parameter_manifest=parameter_manifest,
    )
    payload["parameter_manifest"] = parameter_manifest
    payload["retained_diagnostic_page_manifest"] = retained_page_manifest
    payload["generated"]["active3_tex"] = {
        "path": str(combined_tex.relative_to(REPO_ROOT)),
        "sha256": sha256(combined_tex),
    }
    payload["generated"]["active3_pdf"] = {
        "path": str(combined_pdf.relative_to(REPO_ROOT)),
        "sha256": sha256(combined_pdf),
    }
    payload["generated"]["pdf"] = {
        "path": str(FINAL_PDF.relative_to(REPO_ROOT)),
        "sha256": sha256(FINAL_PDF),
    }
    payload["generated"]["review6_pdf"] = {
        "path": str(review_pdf.relative_to(REPO_ROOT)),
        "sha256": sha256(review_pdf),
    }
    payload["generated"]["preserved_insertion_review_pdf"] = {
        "path": str(PRESERVED_INSERTION_REVIEW_PDF.relative_to(REPO_ROOT)),
        "sha256": sha256(PRESERVED_INSERTION_REVIEW_PDF),
        "pages": 9,
        "placement": "final_pages_7_through_9",
    }
    payload["generated"]["preserved_singleton_insertion_page_pdf"] = {
        "path": str(
            PRESERVED_SINGLETON_INSERTION_PAGE_PDF.relative_to(REPO_ROOT)
        ),
        "sha256": sha256(PRESERVED_SINGLETON_INSERTION_PAGE_PDF),
        "pages": 1,
        "placement": "final_page_10",
    }
    payload["generated"]["parameter_manifest_tex"] = {
        "path": str(parameter_manifest_tex.relative_to(REPO_ROOT)),
        "sha256": sha256(parameter_manifest_tex),
    }
    payload["generated"]["parameter_manifest_pdf"] = {
        "path": str(parameter_manifest_pdf.relative_to(REPO_ROOT)),
        "sha256": sha256(parameter_manifest_pdf),
        "pages": len(PdfReader(str(parameter_manifest_pdf)).pages),
        "placement": "final_appendix",
    }
    payload["generated"]["review4_s_alg_corrected_tex"] = {
        "path": str(CORRECTED_REVIEW4_TEX.relative_to(REPO_ROOT)),
        "sha256": sha256(CORRECTED_REVIEW4_TEX),
    }
    payload["generated"]["review4_s_alg_corrected_pdf"] = {
        "path": str(corrected_review4_pdf.relative_to(REPO_ROOT)),
        "sha256": sha256(corrected_review4_pdf),
        "pages": 4,
        "preservation_contract": (
            "same four review pages and order; Append/SNAKE S_alg fields "
            "replaced from clean-algorithm prefix receipts"
        ),
    }
    assembly_validation = _validate_canonical_assembly(
        review_pdf=review_pdf,
        preserved_review_pdf=PRESERVED_INSERTION_REVIEW_PDF,
        preserved_terminal_page_pdf=PRESERVED_SINGLETON_INSERTION_PAGE_PDF,
        parameter_manifest_pdf=parameter_manifest_pdf,
        final_pdf=FINAL_PDF,
    )
    payload["validation"] = {
        **assembly_validation,
        "latex_box_validation": dict(verified_box_validation),
        "visual_inspection": {
            "status": "not_performed_by_builder_for_current_hash",
            "rendered_pages": [],
        },
    }
    payload["source_tracking_json"] = {
        "path": str(tracker_path.relative_to(REPO_ROOT)),
        "sha256": hashlib.sha256(tracker_path.read_bytes()).hexdigest(),
    }
    PROVENANCE.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if REVIEW6_PROVENANCE.is_file():
        review_payload = json.loads(REVIEW6_PROVENANCE.read_text(encoding="utf-8"))
        review_payload["page_contract"]["cost_tuple_font_size_pt"] = 6.8
        review_payload["page_contract"]["cost_tuple_s_alg_format"] = (
            "two-significant-digit compact e notation"
        )
        review_payload["page_contract"]["panel_identifiers"] = False
        review_payload["page_contract"]["in_panel_n_ph_labels"] = False
        review_payload["page_contract"]["pages_1_4"] = (
            "same four review pages and plot order; Append/SNAKE S_alg tables "
            "replaced from clean-algorithm prefix receipts"
        )
        review_payload["inputs"]["appended_macro_source_page"]["sha256"] = sha256(
            OUTPUT_DIR / f"{FINAL_STEM}_pages1_2.pdf"
        )
        review_payload["inputs"]["appended_singleton_page"]["sha256"] = sha256(
            page_pdf
        )
        review_payload["generated"]["review_pdf"]["sha256"] = sha256(review_pdf)
        review_payload["validation"].pop(
            "pages_1_4_pixel_identical_to_preserved_original_at_120_dpi",
            None,
        )
        review_payload["validation"].pop(
            "pages_1_4_semantic_structure_preserved",
            None,
        )
        review_payload["validation"]["pages_1_4_s_alg_tables_corrected"] = True
        review_payload["validation"].pop("rendered_pages_inspected", None)
        review_payload["validation"]["visual_inspection"] = {
            "status": "not_performed_by_builder_for_current_hash",
            "rendered_pages": [],
        }
        REVIEW6_PROVENANCE.write_text(
            json.dumps(review_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def build(*, tracker_path: Path, output_dir: Path) -> Path:
    if output_dir != OUTPUT_DIR:
        raise ValueError("this append-only builder targets the canonical review PDF")
    ensure_macro_two_page_base()
    tracker = json.loads(tracker_path.read_text(encoding="utf-8"))
    parameter_manifest = collect_parameter_manifest(
        tracker,
        methods=(*MACRO_METHODS, GEO_METHOD, *METHODS),
    )
    retained_parameter_rows, retained_page_manifest = (
        collect_retained_diagnostic_parameter_manifest(
            tracker,
            tracker_path=tracker_path,
        )
    )
    parameter_manifest.extend(retained_parameter_rows)
    write_parameter_manifest_tex(
        parameter_manifest=parameter_manifest,
        tex_path=PARAMETER_MANIFEST_TEX,
        retained_page_manifest=retained_page_manifest,
    )
    parameter_manifest_pdf = compile_tex(
        PARAMETER_MANIFEST_TEX,
        output_dir,
    )
    if parameter_manifest_pdf != PARAMETER_MANIFEST_PDF:
        raise RuntimeError("unexpected parameter-manifest PDF output path")
    if not PdfReader(str(parameter_manifest_pdf)).pages:
        raise RuntimeError("parameter-manifest PDF has no pages")
    existing_provenance = (
        json.loads(PROVENANCE.read_text(encoding="utf-8"))
        if PROVENANCE.is_file()
        else {}
    )
    rows = _closed_cached_rows(
        existing_provenance,
        section="singleton_own_plateau_common_accuracy",
    ) or collect_rows(tracker)
    macro_rows = _closed_cached_rows(
        existing_provenance,
        section="rows",
    ) or collect_macro_common_rows(tracker)
    singleton_common_rows = _closed_cached_rows(
        existing_provenance,
        section="singleton_common_accuracy",
    ) or collect_singleton_common_rows(tracker)
    from pipelines.reporting.build_paper_i_hh_macro_comparison_png import (
        build as build_macro_comparison_png,
    )

    build_macro_comparison_png(
        tracker_path,
        MACRO_COMPARISON_OUTPUT_DIR,
        clean_plateau_rows=macro_rows,
    )
    plot_path = output_dir / f"{PAGE_STEM}_plot.png"
    page_tex = output_dir / f"{PAGE_STEM}.tex"
    make_plot(tracker=tracker, rows=rows, path=plot_path)
    write_page_tex(rows=rows, plot_path=plot_path, tex_path=page_tex)
    page_pdf = compile_tex(page_tex, output_dir)
    combined_tex, combined_pdf = combine_active_pages(page_pdf)
    corrected_review4_pdf = build_corrected_review4(
        macro_rows=macro_rows,
        singleton_common_rows=singleton_common_rows,
        singleton_own_rows=rows,
    )
    review_pdf = combine_review6(
        corrected_review4_pdf=corrected_review4_pdf,
        singleton_pdf=page_pdf,
    )
    latex_box_validation = _latex_box_validation(
        page_tex,
        combined_tex,
        CORRECTED_REVIEW4_TEX,
        REVIEW6_TEX,
        PARAMETER_MANIFEST_TEX,
    )
    _validated_geo_adapt_manifest_rows(parameter_manifest)
    promote_canonical_pdf(
        review_pdf=review_pdf,
        preserved_review_pdf=PRESERVED_INSERTION_REVIEW_PDF,
        preserved_terminal_page_pdf=PRESERVED_SINGLETON_INSERTION_PAGE_PDF,
        parameter_manifest_pdf=parameter_manifest_pdf,
        final_pdf=FINAL_PDF,
    )
    update_provenance(
        rows=rows,
        singleton_common_rows=singleton_common_rows,
        tracker_path=tracker_path,
        plot_path=plot_path,
        page_tex=page_tex,
        page_pdf=page_pdf,
        combined_tex=combined_tex,
        combined_pdf=combined_pdf,
        corrected_review4_pdf=corrected_review4_pdf,
        review_pdf=review_pdf,
        parameter_manifest=parameter_manifest,
        retained_page_manifest=retained_page_manifest,
        parameter_manifest_tex=PARAMETER_MANIFEST_TEX,
        parameter_manifest_pdf=parameter_manifest_pdf,
        latex_box_validation=latex_box_validation,
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
