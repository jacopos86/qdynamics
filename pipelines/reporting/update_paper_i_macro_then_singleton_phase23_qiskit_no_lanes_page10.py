#!/usr/bin/env python3
"""Append the completed macro-to-singleton Phase-II/III Qiskit cells as page 10."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting.update_paper_i_phase3_qiskit_no_lanes_page9 import (
    UpdateError,
    binding,
    compile_terminal,
    format_cost,
    format_error,
    format_s,
    load,
    load_stationary,
    sha256,
    trace,
)
from pipelines.reporting.paper_i_mixed_horizon_continuation import (
    MixedHorizonContinuationError,
    STRONG_HOLSTEIN_REGIMES,
    coalesce_continuation_points,
    decorate_route,
    horizon_policy,
    missing_route_continuation_status,
    validate_continuation_adapter,
)


REPORT_DIR = REPO_ROOT / (
    "output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving"
)
TARGET_PDF = REPORT_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "partial_progress.pdf"
)
TARGET_PROVENANCE = TARGET_PDF.with_name(
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "partial_progress_provenance.json"
)
ASSET_STEM = (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "macro_then_singleton_phase23_qiskit_no_lanes_page10"
)
PAGE_PDF = REPORT_DIR / f"{ASSET_STEM}.pdf"
PAGE_PNG = REPORT_DIR / f"{ASSET_STEM}.png"
ADAPTER_PATH = REPORT_DIR / f"{ASSET_STEM}_adapter.json"
RECOVERABLE_CONTINUATIONS = REPO_ROOT / (
    "output/pdf/paper_i_stationary_vs_paper_i_route_comparison_20260729/"
    "paper_i_page10_recoverable_continuations_20260808.json"
)
CONTINUATION_ADAPTER_PATH = REPORT_DIR / (
    f"{ASSET_STEM}_strong_sector_r70_continuations.json"
)

PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_macro_then_singleton_phase123_qiskit_phase23_"
    "no_lanes_tau1em4_r50_20260807_v1_chtc"
)
RETRIEVED_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "retrieved_chtc_20260807_macro_then_singleton_phase123_qiskit_"
    "phase23_no_lanes_v1"
)
LOCAL_RUN_DIRS = {
    "weak_sector": REPO_ROOT
    / (
        "output/local_runs/"
        "paper_i_page10_macro_then_singleton_phase23_qiskit_no_lanes_"
        "weak_sector_20260808_v1"
    ),
    "strong_weak": REPO_ROOT
    / (
        "output/local_runs/"
        "paper_i_page10_macro_then_singleton_phase23_qiskit_no_lanes_"
        "strong_weak_20260808_v1"
    ),
}

REGIME_ORDER = (
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
)
REGIME_LABELS = {
    "weak_weak": "Weak--weak",
    "intermediate_weak": "Intermediate--weak",
    "strong_weak_u8": "Strong--weak",
    "weak_strong": "Weak--strong",
    "intermediate_strong": "Intermediate--strong",
    "strong_strong_u8": "Strong--strong",
}
NPH = {
    "weak_weak": 3,
    "intermediate_weak": 3,
    "strong_weak_u8": 3,
    "weak_strong": 7,
    "intermediate_strong": 7,
    "strong_strong_u8": 7,
}
COMPLETED = {
    "weak_weak": {
        "source_kind": "authorized_local_run",
        "local_root": "weak_sector",
        "directory": "weak_weak",
    },
    "intermediate_weak": {
        "source_kind": "authorized_local_run",
        "local_root": "weak_sector",
        "directory": "intermediate_weak",
    },
    "strong_weak_u8": {
        "source_kind": "authorized_local_run",
        "local_root": "strong_weak",
        "directory": ".",
    },
    "weak_strong": {
        "source_kind": "verified_chtc_fetch",
        "directory": "9600705.3_weak_strong",
        "archive": "9600705.3_weak_strong.tar.gz",
        "cluster_id": 9600705,
        "proc_id": 3,
        "archive_sha256": (
            "afb7e9dcd5b28d876940d71df0783f158c428e12e62ceedaf2dfb441c32211fe"
        ),
        "archive_size_bytes": 454493230,
        "remote_state": "exact_deleted_after_verified_fetch",
    },
    "intermediate_strong": {
        "source_kind": "verified_chtc_fetch",
        "directory": "9600705.4_intermediate_strong",
        "archive": "9600705.4_intermediate_strong.tar.gz",
        "cluster_id": 9600705,
        "proc_id": 4,
        "archive_sha256": (
            "150b0c3d2739f57f8c4c07475b39bf1ea77c3743596e3194f0b4cff403526cf3"
        ),
        "archive_size_bytes": 514816542,
        "remote_state": "exact_deleted_after_verified_fetch",
    },
    "strong_strong_u8": {
        "source_kind": "verified_chtc_fetch",
        "directory": "9600705.5_strong_strong_u8",
        "archive": "9600705.5_strong_strong_u8.tar.gz",
        "cluster_id": 9600705,
        "proc_id": 5,
        "archive_sha256": (
            "c2ca5029334be90d664e4798483de9b095fe4af024d44027ef5377632efe1d7b"
        ),
        "archive_size_bytes": 737842102,
        "remote_state": "exact_deleted_after_verified_fetch",
    },
}
EXPECTED_ROUTE = (
    "paper_i_ra_adapt__single_pauli_word_v1__"
    "insertion_commutation_plateau_v2__stationary_source_response_v1__"
    "all_phase_resource_weighting_v1__macro_phase1_then_singleton_phase1_"
    "then_qiskit_phase2_phase3_no_lanes_v1"
)
EXPECTED_ROUTE_SHA256 = (
    "83b5e5cb17bdfbfc8e8efb22a586d952b3343f430de15ffb58550082d17e3cf0"
)
PAGE_ID = "macro_then_singleton_phase123_qiskit_phase23_no_lanes_r50_partial_v1"
REPORT_KEY = "macro_then_singleton_phase123_qiskit_phase23_no_lanes_r50"
PLOT_FLOOR = 1.0e-16


def _artifact_binding(worker: Mapping[str, Any], path: Path) -> Mapping[str, Any]:
    parts = path.parts
    try:
        suffix = Path(*parts[parts.index("runs") :]).as_posix()
    except ValueError as exc:
        raise UpdateError(f"artifact path has no runs root: {path}") from exc
    matches = [
        row
        for row in worker.get("artifacts", ())
        if isinstance(row, Mapping) and str(row.get("path", "")).endswith(suffix)
    ]
    if len(matches) != 1:
        raise UpdateError(f"worker artifact binding is absent: {suffix}")
    return matches[0]


def _repo_binding(raw_path: Any, *, label: str) -> dict[str, Any]:
    relative = Path(str(raw_path))
    if relative.is_absolute() or ".." in relative.parts:
        raise UpdateError(f"{label} path escapes the repository")
    return binding(REPO_ROOT / relative)


def load_recoverable_continuations() -> dict[str, dict[str, Any]]:
    """Load the authenticated k>50 prefixes retained after local ENOSPC."""

    if not RECOVERABLE_CONTINUATIONS.exists():
        return {}
    source = load(RECOVERABLE_CONTINUATIONS)
    if (
        source.get("schema") != "paper_i_page10_recoverable_continuations_v1"
        or source.get("status") != "recoverable_prefixes_after_local_enospc"
        or source.get("paper_evidence_adopted") is not False
    ):
        raise UpdateError("Page-10 recoverable continuation identity drifted")
    cells = source.get("cells")
    if not isinstance(cells, list):
        raise UpdateError("Page-10 recoverable continuation cells are invalid")
    source_binding = binding(RECOVERABLE_CONTINUATIONS)
    result: dict[str, dict[str, Any]] = {}
    for raw in cells:
        if not isinstance(raw, Mapping):
            raise UpdateError("Page-10 continuation cell is invalid")
        regime = str(raw.get("regime_id", ""))
        if regime not in STRONG_HOLSTEIN_REGIMES or regime in result:
            raise UpdateError(f"unsupported Page-10 continuation regime: {regime}")
        exact = float(raw.get("exact_same_cutoff_energy"))
        recoverable_round = int(raw.get("recoverable_round", -1))
        points: list[dict[str, Any]] = []
        for expected, row in enumerate(raw.get("points", ()), 51):
            if not isinstance(row, Mapping) or int(row.get("k", -1)) != expected:
                raise UpdateError(f"{regime}: continuation points are noncanonical")
            energy = float(row["energy"])
            error = float(row["error"])
            if not math.isclose(
                error,
                abs(energy - exact),
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ):
                raise UpdateError(f"{regime}: continuation error drifted")
            points.append({"k": expected, "energy": energy, "error": error})
        if (
            not points
            or points[-1]["k"] != recoverable_round
            or recoverable_round >= 70
        ):
            raise UpdateError(f"{regime}: recoverable continuation boundary drifted")

        checkpoint_raw = raw.get("checkpoint")
        failure_raw = raw.get("failure_receipt")
        if not isinstance(checkpoint_raw, Mapping) or not isinstance(
            failure_raw, Mapping
        ):
            raise UpdateError(f"{regime}: continuation bindings are invalid")
        checkpoint = _repo_binding(
            checkpoint_raw.get("path", ""),
            label=f"{regime} continuation checkpoint",
        )
        if (
            checkpoint["sha256"] != checkpoint_raw.get("sha256")
            or checkpoint["size_bytes"] != checkpoint_raw.get("size_bytes")
        ):
            raise UpdateError(f"{regime}: continuation checkpoint drifted")
        failure = _repo_binding(
            failure_raw.get("path", ""),
            label=f"{regime} continuation failure receipt",
        )
        if failure["sha256"] != failure_raw.get("sha256"):
            raise UpdateError(f"{regime}: continuation failure receipt drifted")
        result[regime] = {
            "points": points,
            "status": "recoverable_prefix_incomplete",
            "exact_same_cutoff_energy": exact,
            "source": {
                "kind": "validated_recoverable_prefix_after_local_enospc",
                "continuation_adapter": source_binding,
                "checkpoint": checkpoint,
                "failure_receipt": failure,
            },
        }
    return result


def load_continuation_adapter() -> dict[str, dict[str, Any]]:
    """Load authenticated completed/running k=70 results when available."""

    if not CONTINUATION_ADAPTER_PATH.exists():
        return {}
    try:
        cells = validate_continuation_adapter(
            load(CONTINUATION_ADAPTER_PATH),
            expected_route_contract_sha256=EXPECTED_ROUTE_SHA256,
        )
    except MixedHorizonContinuationError as exc:
        raise UpdateError(str(exc)) from exc
    adapter_binding = binding(CONTINUATION_ADAPTER_PATH)
    for cell in cells.values():
        cell["source"]["continuation_adapter"] = adapter_binding
    return cells


def load_current(regime: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    source_kind = str(spec["source_kind"])
    if source_kind == "verified_chtc_fetch":
        directory = RETRIEVED_DIR / str(spec["directory"])
        archive = RETRIEVED_DIR / str(spec["archive"])
        if (
            sha256(archive) != spec["archive_sha256"]
            or archive.stat().st_size != spec["archive_size_bytes"]
        ):
            raise UpdateError(f"{regime}: fetched archive drifted")
    elif source_kind == "authorized_local_run":
        local_root = str(spec.get("local_root", ""))
        if local_root not in LOCAL_RUN_DIRS:
            raise UpdateError(f"{regime}: unsupported local root {local_root}")
        directory = LOCAL_RUN_DIRS[local_root] / str(spec["directory"])
        archive = None
    else:
        raise UpdateError(f"{regime}: unsupported source kind {source_kind}")
    worker_path = directory / "worker_receipt.json"
    manifest_path = next(directory.glob("runs/*/execution_manifest.json"))
    summary_path = next(directory.glob("runs/*/summary/summary.json"))
    result_path = next(directory.glob("runs/*/result/result.json"), None)
    if source_kind == "authorized_local_run" and result_path is None:
        raise UpdateError(f"{regime}: local result payload is absent")
    worker = load(worker_path)
    manifest = load(manifest_path)
    summary = load(summary_path)
    execution_id = str(worker.get("execution_id"))
    job_path = PACKAGE_DIR / "jobs" / f"{execution_id}.json"
    job = load(job_path)
    if (
        worker.get("status") != "passed"
        or worker.get("controller_rounds_completed") != 50
        or worker.get("job_spec_sha256") != job.get("sha256")
        or manifest.get("status") != "passed"
        or manifest.get("execution_id") != execution_id
        or manifest.get("controller_rounds_completed") != 50
        or manifest.get("route_contract_sha256") != EXPECTED_ROUTE_SHA256
        or job.get("regime_id") != regime
        or job.get("nph") != NPH[regime]
        or job.get("target_horizon") != 50
        or job.get("candidate_representation") != "single_pauli_word_v1"
        or job.get("route_contract_sha256") != EXPECTED_ROUTE_SHA256
    ):
        raise UpdateError(f"{regime}: package/worker identity drifted")
    bound_artifacts = [manifest_path, summary_path]
    if result_path is not None:
        bound_artifacts.append(result_path)
    for path in bound_artifacts:
        artifact = _artifact_binding(worker, path)
        if (
            artifact.get("sha256") != sha256(path)
            or artifact.get("size_bytes") != path.stat().st_size
        ):
            raise UpdateError(f"{regime}: worker artifact binding drifted")
    provenance = summary.get("provenance", {})
    if (
        provenance.get("route_profile") != EXPECTED_ROUTE
        or provenance.get("route_contract_sha256") != EXPECTED_ROUTE_SHA256
        or provenance.get("candidate_representation") != "single_pauli_word_v1"
    ):
        raise UpdateError(f"{regime}: scientific route identity drifted")
    points = trace(summary, label=f"{regime} macro-to-singleton")
    compiled = compile_terminal(summary)
    plateau = summary["effective_plateau"]
    source_bindings = {
        "admission_mode": (
            "verified_chtc_fetch_v1"
            if source_kind == "verified_chtc_fetch"
            else "authorized_local_worker_receipt_v1"
        ),
        "worker_receipt": binding(worker_path),
        "execution_manifest": binding(manifest_path),
        "summary": binding(summary_path),
        "job": binding(job_path),
    }
    if result_path is not None:
        source_bindings["result"] = binding(result_path)
    if archive is not None:
        source_bindings.update(
            {
                "archive": binding(archive),
                "remote_archive": {
                    "path": (
                        "/home/jsstrobel/Holstein_phase3_optuna_chtc/transfer/"
                        + execution_id
                        + f"__{spec['cluster_id']}__{spec['proc_id']}.tar.gz"
                    ),
                    "sha256": spec["archive_sha256"],
                    "size_bytes": spec["archive_size_bytes"],
                    "state": spec["remote_state"],
                },
            }
        )
    return {
        "status": "complete",
        "execution_id": execution_id,
        "execution_scope": (
            "chtc" if source_kind == "verified_chtc_fetch" else "local"
        ),
        "cluster_id": (
            int(spec["cluster_id"])
            if source_kind == "verified_chtc_fetch"
            else None
        ),
        "proc_id": (
            int(spec["proc_id"])
            if source_kind == "verified_chtc_fetch"
            else None
        ),
        "points": points,
        "marker": {
            "k": int(plateau["controller_round"]),
            "error": float(plateau["absolute_energy_error"]),
        },
        "terminal": {
            "k": 50,
            "error": points[-1]["error"],
            **compiled,
            "S_alg": int(summary["canonical_all_work"]["s_alg"]),
        },
        "exact_same_cutoff_energy": float(
            summary["provenance"]["exact_same_cutoff_energy"]
        ),
        "source_bindings": source_bindings,
    }


def build_adapter(provenance: Mapping[str, Any]) -> dict[str, Any]:
    page8 = provenance["phase3_on_plateau_singleton_sixregime_r50"]
    page8_cells = {row["regime_id"]: row for row in page8["cells"]}
    recoverable = load_recoverable_continuations()
    continued = load_continuation_adapter()
    cells = []
    for regime in REGIME_ORDER:
        current = load_current(regime, COMPLETED[regime]) if regime in COMPLETED else None
        continuation = recoverable.get(regime)
        completed_or_live = continued.get(regime)
        if current is not None:
            if continuation is not None and not math.isclose(
                current["exact_same_cutoff_energy"],
                continuation["exact_same_cutoff_energy"],
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise UpdateError(f"{regime}: continuation exact reference drifted")
            continuation_sources = [
                source["points"]
                for source in (continuation, completed_or_live)
                if source is not None
            ]
            try:
                merged_continuation = (
                    coalesce_continuation_points(
                        current["points"],
                        continuation_sources,
                        label=regime,
                    )
                    if continuation_sources
                    else None
                )
            except MixedHorizonContinuationError as exc:
                raise UpdateError(str(exc)) from exc
            selected_source = completed_or_live or continuation
            source_receipt = None
            if continuation is not None or completed_or_live is not None:
                source_receipt = {
                    "sources": [
                        source["source"]
                        for source in (continuation, completed_or_live)
                        if source is not None
                    ]
                }
            current = decorate_route(
                current,
                regime_id=regime,
                continuation_points=merged_continuation,
                continuation_status=(
                    selected_source["status"]
                    if selected_source is not None
                    else "pending"
                    if regime in STRONG_HOLSTEIN_REGIMES
                    else None
                ),
                continuation_source=source_receipt,
            )
        continuation_status = (
            copy.deepcopy(current["continuation"])
            if current is not None
            else missing_route_continuation_status(regime_id=regime)
        )
        stationary_available = any(
            row.get("regime_id") == regime
            and row.get("route_id") == "ra_singleton_plateau"
            for row in provenance["included_sources"]
        )
        cells.append(
            {
                "regime_id": regime,
                "regime_label": REGIME_LABELS[regime],
                "nph": NPH[regime],
                "append_adapt": copy.deepcopy(page8_cells[regime]["append_adapt"]),
                "stationary_ra_plateau": (
                    load_stationary(provenance, regime)
                    if current and stationary_available
                    else None
                ),
                "stationary_comparator_status": (
                    "available" if current and stationary_available else "unavailable"
                ),
                "macro_then_singleton": current,
                "current_status": "complete" if current else "pending_on_chtc",
                "continuation_status": continuation_status,
            }
        )
    unsigned = {
        "schema": "paper_i_macro_then_singleton_phase23_qiskit_page10_adapter_v1",
        "page_id": PAGE_ID,
        "status": f"partial_{len(COMPLETED)}_of_6_complete",
        "paper_evidence_adopted": False,
        "route_profile": EXPECTED_ROUTE,
        "route_contract_sha256": EXPECTED_ROUTE_SHA256,
        "comparison": "macro-to-singleton route vs Append-ADAPT vs page-2 stationary RA plateau",
        "horizon_policy": horizon_policy(),
        "cells": cells,
    }
    unsigned["sha256"] = hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    ADAPTER_PATH.write_text(
        json.dumps(unsigned, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return unsigned


def render(adapter: Mapping[str, Any]) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    mpl.rcParams.update({"font.family": "serif", "font.size": 7.5})
    fig = plt.figure(figsize=(11, 8.5))
    grid = fig.add_gridspec(
        3, 3, height_ratios=(1.0, 1.0, 0.72), hspace=0.34, wspace=0.25
    )
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]
    for index, (axis, cell) in enumerate(zip(axes, adapter["cells"], strict=True)):
        append = cell["append_adapt"]
        axis.plot(
            [p["k"] for p in append["points"]],
            [max(float(p["error"]), PLOT_FLOOR) for p in append["points"]],
            color="#4C78A8",
            lw=1.45,
        )
        stationary = cell["stationary_ra_plateau"]
        if stationary:
            axis.plot(
                [p["k"] for p in stationary["points"]],
                [max(float(p["error"]), PLOT_FLOOR) for p in stationary["points"]],
                color="#009E73",
                lw=1.45,
                ls="--",
            )
        current = cell["macro_then_singleton"]
        if current:
            trajectory = current["trajectory_points"]
            axis.plot(
                [p["k"] for p in trajectory],
                [max(float(p["error"]), PLOT_FLOOR) for p in trajectory],
                color="#CC79A7",
                lw=1.8,
            )
            axis.scatter(
                [current["marker"]["k"]],
                [max(float(current["marker"]["error"]), PLOT_FLOOR)],
                color="#CC79A7",
                marker="*",
                s=42,
                zorder=4,
            )
        else:
            axis.text(
                0.5,
                0.12,
                "base k=50 result pending; continuation unavailable",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#CC79A7",
                fontsize=7.0,
                bbox={"facecolor": "white", "edgecolor": "#CC79A7", "alpha": 0.85},
            )
        if stationary is None:
            axis.text(
                0.02,
                0.97,
                "page-2 stationary comparator unavailable",
                transform=axis.transAxes,
                ha="left",
                va="top",
                color="#666666",
                fontsize=6.2,
            )
        axis.set_yscale("log")
        axis.set_xlim(0, 70)
        axis.grid(True, which="major", alpha=0.22, lw=0.5)
        axis.set_title(
            f"{cell['regime_label']} ($n_{{ph}}={cell['nph']}$)", fontsize=8.5
        )
        if index // 3 == 1:
            axis.set_xlabel("ADAPT controller round")
        if index % 3 == 0:
            axis.set_ylabel(r"same-cutoff $|\Delta E|$")
    fig.legend(
        handles=[
            Line2D([0], [0], color="#4C78A8", lw=1.45, label="Append-ADAPT"),
            Line2D(
                [0],
                [0],
                color="#009E73",
                lw=1.45,
                ls="--",
                label="Page-2 stationary RA plateau",
            ),
            Line2D(
                [0],
                [0],
                color="#CC79A7",
                lw=1.8,
                marker="*",
                markersize=6,
                label="Macro shortlist to singleton; Qiskit Phase II/III; no lanes",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Macro Phase I to singleton Phases I/II/III: Qiskit denominator in Phases II/III",
        fontsize=11.0,
        fontweight="bold",
        y=0.988,
    )
    table_axis = fig.add_subplot(grid[2, :])
    table_axis.axis("off")
    rows = []
    for cell in adapter["cells"]:
        current = cell["macro_then_singleton"]
        if not current:
            continue
        for label, route in (
            ("Macro->singleton", current),
            ("stationary RA", cell["stationary_ra_plateau"]),
            ("Append-ADAPT", cell["append_adapt"]),
        ):
            if route is None:
                continue
            terminal = route.get("paper_facing_fixed_round_50", route["terminal"])
            rows.append(
                [
                    cell["regime_label"],
                    label,
                    format_error(float(terminal["error"])),
                    format_cost(terminal),
                    format_s(int(terminal["S_alg"])),
                ]
            )
    table = table_axis.table(
        cellText=rows,
        colLabels=[
            "Regime",
            "Method",
            r"$|\Delta E_{50}|$",
            r"$(N_{2q},D_{2q},D_c,W_{1q})$",
            r"$S_{\rm alg}$",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=(0.18, 0.20, 0.16, 0.28, 0.12),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1.0, 0.82)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EAEAEA")
    fig.text(
        0.5,
        0.248,
        (
            "Mixed horizon: selected strong-sector curves may extend to k=70; "
            "all table errors, Qiskit tuples, and S_alg are fixed at k=50."
        ),
        ha="center",
        va="bottom",
        fontsize=6.7,
        color="#444444",
    )
    fig.savefig(PAGE_PDF, bbox_inches="tight")
    fig.savefig(PAGE_PNG, dpi=240, bbox_inches="tight")
    plt.close(fig)


def append_page(adapter: Mapping[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    from pypdf import PdfReader, PdfWriter

    current_binding = binding(TARGET_PDF)
    declared = provenance["outputs"]["partial_progress_pdf"]
    page_count = int(provenance["layout"].get("page_count", -1))
    if (
        current_binding["sha256"] != declared["sha256"]
        or current_binding["size_bytes"] != declared["size_bytes"]
        or page_count < 9
        or (page_count >= 10 and provenance["layout"].get("page_10") != PAGE_ID)
    ):
        raise UpdateError("target PDF/provenance is not a supported page-10 state")
    old = PdfReader(str(TARGET_PDF), strict=False)
    new = PdfReader(str(PAGE_PDF), strict=False)
    if len(old.pages) != page_count or len(new.pages) != 1:
        raise UpdateError("unexpected PDF page count")
    writer = PdfWriter()
    for page in old.pages[:9]:
        writer.add_page(page)
    writer.add_page(new.pages[0])
    for page in old.pages[10:]:
        writer.add_page(page)
    page_count_after = 10 if page_count == 9 else page_count
    temporary_pdf = TARGET_PDF.with_name(f".{TARGET_PDF.name}.page10.tmp")
    temporary_provenance = TARGET_PROVENANCE.with_name(
        f".{TARGET_PROVENANCE.name}.page10.tmp"
    )
    rollback = TARGET_PDF.with_name(f".{TARGET_PDF.name}.page10.rollback")
    for path in (temporary_pdf, temporary_provenance, rollback):
        if path.exists() or path.is_symlink():
            raise UpdateError(f"stale temporary exists: {path}")
    try:
        with temporary_pdf.open("xb") as stream:
            writer.write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        if (
            len(PdfReader(str(temporary_pdf), strict=False).pages)
            != page_count_after
        ):
            raise UpdateError("combined PDF page count drifted")
        updated = copy.deepcopy(provenance)
        updated["layout"]["page_10"] = PAGE_ID
        updated["layout"]["page_count"] = page_count_after
        updated[REPORT_KEY] = {
            "schema": "paper_i_macro_then_singleton_phase23_qiskit_page10_report_v1",
            "page_id": PAGE_ID,
            "status": adapter["status"],
            "paper_evidence_adopted": False,
            "horizon_policy": copy.deepcopy(adapter["horizon_policy"]),
            "adapter": {
                **binding(ADAPTER_PATH),
                "canonical_sha256": adapter["sha256"],
            },
            "cells": copy.deepcopy(adapter["cells"]),
            "completed_regimes": sorted(COMPLETED),
            "pending_regimes": [r for r in REGIME_ORDER if r not in COMPLETED],
            "outputs": {
                "page_pdf": binding(PAGE_PDF),
                "page_png": binding(PAGE_PNG),
            },
            "structural_validation": {
                "pages_before": page_count,
                "pages_after": page_count_after,
                "preserved_pages": page_count_after - 1,
                "preserved_trailing_pages": max(0, page_count - 10),
                "page_10_operation": (
                    "append" if page_count == 9 else "replace"
                ),
            },
        }
        combined = binding(temporary_pdf)
        combined["path"] = str(TARGET_PDF.resolve())
        updated["outputs"]["partial_progress_pdf"] = combined
        updated["outputs"][
            "macro_then_singleton_phase23_qiskit_page10_pdf"
        ] = binding(PAGE_PDF)
        updated["outputs"][
            "macro_then_singleton_phase23_qiskit_page10_png"
        ] = binding(PAGE_PNG)
        updated["outputs"]["macro_then_singleton_phase23_qiskit_page10_adapter"] = {
            **binding(ADAPTER_PATH),
            "canonical_sha256": adapter["sha256"],
        }
        with temporary_provenance.open("xb") as stream:
            stream.write(
                json.dumps(updated, indent=2, sort_keys=True, allow_nan=False).encode()
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.link(TARGET_PDF, rollback)
        os.replace(temporary_pdf, TARGET_PDF)
        try:
            os.replace(temporary_provenance, TARGET_PROVENANCE)
        except Exception:
            os.replace(rollback, TARGET_PDF)
            raise
        rollback.unlink(missing_ok=True)
    except Exception:
        temporary_pdf.unlink(missing_ok=True)
        temporary_provenance.unlink(missing_ok=True)
        rollback.unlink(missing_ok=True)
        raise
    return {
        "status": "updated_existing_report_in_place",
        "page_count": page_count_after,
        "completed_regimes": sorted(COMPLETED),
        "pdf": binding(TARGET_PDF),
    }


def main() -> int:
    provenance = load(TARGET_PROVENANCE)
    adapter = build_adapter(provenance)
    render(adapter)
    result = append_page(adapter, provenance)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
