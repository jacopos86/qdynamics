#!/usr/bin/env python3
"""Append/replace the two Phase-0 diagnostic route pages in the evolving PDF."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
SNAPSHOT_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "retrieved_phase0_progress_20260808"
)
COMPLETED_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "retrieved_phase0_completed_20260809"
)
PAGE12_R70_CONTINUATION_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "retrieved_chtc_20260812_page12_strong_r70_continuations_v1"
)
ED_SOURCE = REPO_ROOT / (
    "MATH/paper_facing/paper_I_static_scaffold/"
    "paper_i_hh_ed_cutoff_reference_six_regime_20260727.json"
)

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
PLOT_FLOOR = 1.0e-16

MACRO_PACKAGE = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_macro_gradient_phase0_then_singleton_phase123_"
    "qiskit_phase23_no_lanes_cap24_tau1em4_r50_20260807_v1_chtc"
)
GLOBAL_PACKAGE = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_global_singleton_gradient_phase0_phase123_"
    "qiskit_phase23_no_lanes_cap24_tau1em4_r50_20260807_v1_chtc"
)

MACRO_TERMINAL_ONLY = {
    "weak_strong": {
        "k": 50,
        "energy": -1.1384594010338258,
        "cluster_id": 9605117,
        "proc_id": 3,
        "status": "completed_science_archive_lost",
        "source_limitation": (
            "terminal hardcoded_adapt_iter and passed worker receipt were observed "
            "before the OSDF 403 transfer failure; the execute sandbox and full "
            "trajectory archive are no longer available"
        ),
    },
    "intermediate_strong": {
        "k": 50,
        "energy": -0.6239270440664035,
        "cluster_id": 9605117,
        "proc_id": 4,
        "status": "completed_science_archive_lost",
        "source_limitation": (
            "terminal hardcoded_adapt_iter and passed worker receipt were observed "
            "before the OSDF 403 transfer failure; the execute sandbox and full "
            "trajectory archive are no longer available"
        ),
    },
}

ROUTES = (
    {
        "key": "macro_gradient_phase0_then_singleton",
        "page": 11,
        "page_id": "macro_gradient_phase0_then_singleton_partial_v1",
        "title": (
            "Macro-gradient Phase 0, then singleton Phase I/II/III "
            "(Qiskit II/III, no lanes)"
        ),
        "short_label": "Macro Phase-0 -> singleton",
        "color": "#D55E00",
        "package": MACRO_PACKAGE,
        "route_sha256": (
            "bb07440ed21d8e663817642124344d800a2e9bb556fd66209e7def5ca9e7b73b"
        ),
        "snapshots": {"strong_strong_u8": "9605117.5.adapt_iter.jsonl"},
        "completed": {},
        "terminal_only": MACRO_TERMINAL_ONLY,
        "cluster_id": 9605117,
    },
    {
        "key": "global_singleton_gradient_phase0",
        "page": 12,
        "page_id": "global_singleton_gradient_phase0_partial_v1",
        "title": (
            "Initialized global-singleton gradient Phase 0, then Phase I/II/III "
            "(Qiskit II/III, no lanes)"
        ),
        "short_label": "Global-singleton Phase 0",
        "color": "#CC79A7",
        "package": GLOBAL_PACKAGE,
        "route_sha256": (
            "9811652b332b592bee048a8e5f3048972256abae186921ed7efea52bfd5f3dd8"
        ),
        "snapshots": {
            "weak_strong": "9605157.3.adapt_iter.jsonl",
            "intermediate_strong": "9605157.4.adapt_iter.jsonl",
            "strong_strong_u8": "9605157.5.adapt_iter.jsonl",
        },
        "completed": {
            "weak_weak": "9605157.0_completed_report_adapter.json",
            "intermediate_weak": "9605157.1_completed_report_adapter.json",
            "strong_weak_u8": "9605157.2_completed_report_adapter.json",
            "weak_strong": "9605157.3_completed_report_adapter.json",
            "intermediate_strong": "9605157.4_completed_report_adapter.json",
            "strong_strong_u8": "9605157.5_completed_report_adapter.json",
        },
        "continuations": {
            "weak_strong": "9629628.0_page12_r70_continuation_adapter.json",
            "intermediate_strong": (
                "9629628.1_page12_r70_continuation_adapter.json"
            ),
            "strong_strong_u8": (
                "9629628.2_page12_r70_continuation_adapter.json"
            ),
        },
        "terminal_only": {},
        "cluster_id": 9605157,
    },
)


class UpdateError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise UpdateError(f"unsafe or missing file: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise UpdateError(f"JSON object required: {path}")
    return value


def verify_self_digest(value: Mapping[str, Any], *, label: str) -> None:
    claimed = value.get("sha256")
    unsigned = {key: row for key, row in value.items() if key != "sha256"}
    observed = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    if claimed != observed:
        raise UpdateError(f"{label}: self digest drifted")


def bind_retrieved_full_archive(
    completed: Mapping[str, Any], *, adapter_name: str
) -> dict[str, Any]:
    source = completed.get("source")
    if not isinstance(source, Mapping):
        raise UpdateError(f"{adapter_name}: completed source is invalid")
    remote = source.get("full_archive")
    if not isinstance(remote, Mapping):
        raise UpdateError(f"{adapter_name}: remote full-archive identity is absent")
    local_path = COMPLETED_DIR / (
        f"{int(completed['cluster_id'])}.{int(completed['proc_id'])}_full.tar.gz"
    )
    local = binding(local_path)
    if (
        local["sha256"] != remote.get("sha256")
        or local["size_bytes"] != remote.get("size_bytes")
    ):
        raise UpdateError(
            f"{adapter_name}: retrieved full archive differs from remote identity"
        )
    return {
        "local_archive": local,
        "remote_archive_at_retrieval": copy.deepcopy(dict(remote)),
        "remote_local_sha256_size_identity": "passed",
        "local_retrieval_state": "complete_verified_local_archive",
    }


def _authenticated_completion(status: Any) -> bool:
    return str(status).startswith("completed_authenticated_remote_summary")


def bind_page12_r70_continuation_archive(
    continuation: Mapping[str, Any], *, adapter_name: str
) -> dict[str, Any]:
    source = continuation.get("source")
    if not isinstance(source, Mapping):
        raise UpdateError(f"{adapter_name}: continuation source is invalid")
    raw_local = source.get("local_archive")
    raw_remote = source.get("full_archive")
    if not isinstance(raw_local, Mapping) or not isinstance(raw_remote, Mapping):
        raise UpdateError(f"{adapter_name}: continuation archive bindings are absent")
    local_path = Path(str(raw_local.get("path", "")))
    if not local_path.is_absolute():
        local_path = REPO_ROOT / local_path
    local = binding(local_path)
    if (
        local["sha256"] != raw_local.get("sha256")
        or local["size_bytes"] != raw_local.get("size_bytes")
        or local["sha256"] != raw_remote.get("sha256")
        or local["size_bytes"] != raw_remote.get("size_bytes")
    ):
        raise UpdateError(
            f"{adapter_name}: Page-12 continuation archive identity drifted"
        )
    return {
        "local_archive": local,
        "remote_archive_at_retrieval": copy.deepcopy(dict(raw_remote)),
        "remote_local_sha256_size_identity": "passed",
        "local_retrieval_state": "complete_verified_local_archive",
    }


def merge_page12_r70_continuation(
    current: Mapping[str, Any],
    continuation: Mapping[str, Any],
    *,
    regime: str,
    completed_adapter_binding: Mapping[str, Any],
    continuation_adapter_binding: Mapping[str, Any],
    continuation_archive_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge rounds 51--70 while retaining the fixed round-50 cost tuple."""

    fixed = continuation.get("fixed_round_50_reporting")
    source = continuation.get("source")
    if not isinstance(fixed, Mapping) or not isinstance(source, Mapping):
        raise UpdateError(f"{regime}: Page-12 continuation adapter is incomplete")
    base_binding = source.get("base_completed_adapter")
    merged_points = continuation.get("merged_points")
    continuation_points = continuation.get("continuation_points")
    latest = continuation.get("latest")
    if (
        continuation.get("schema")
        != "paper_i_page12_r70_continuation_adapter_v1"
        or continuation.get("status")
        != "passed_authenticated_round70_continuation"
        or continuation.get("regime_id") != regime
        or continuation.get("source_horizon") != 50
        or continuation.get("target_horizon") != 70
        or not isinstance(base_binding, Mapping)
        or not isinstance(merged_points, list)
        or not isinstance(continuation_points, list)
        or not isinstance(latest, Mapping)
        or [row.get("k") for row in merged_points] != list(range(1, 71))
        or [row.get("k") for row in continuation_points] != list(range(51, 71))
        or latest.get("k") != 70
        or fixed.get("controller_round") != 50
        or fixed.get("costs") != current.get("costs")
        or fixed.get("compile") != current.get("compile")
        or fixed.get("work_components") != current.get("work_components")
        or base_binding.get("sha256") != completed_adapter_binding.get("sha256")
        or base_binding.get("size_bytes")
        != completed_adapter_binding.get("size_bytes")
    ):
        raise UpdateError(f"{regime}: Page-12 continuation identity drifted")
    current_points = current.get("points")
    if (
        not isinstance(current_points, list)
        or [row.get("k") for row in current_points] != list(range(1, 51))
        or any(
            not math.isclose(
                float(left[field]),
                float(right[field]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            for left, right in zip(current_points, merged_points[:50], strict=True)
            for field in ("energy", "error")
        )
    ):
        raise UpdateError(f"{regime}: authenticated round-50 prefix drifted")

    merged = copy.deepcopy(dict(current))
    merged["status"] = "completed_authenticated_remote_summary_r70_continuation"
    merged["points"] = copy.deepcopy(merged_points)
    merged["latest"] = copy.deepcopy(dict(latest))
    merged["fixed_resource_controller_round"] = 50
    merged["trajectory_controller_round"] = 70
    merged_source = copy.deepcopy(dict(merged.get("source", {})))
    merged_source["round70_continuation"] = {
        "adapter": copy.deepcopy(dict(continuation_adapter_binding)),
        "archive": copy.deepcopy(dict(continuation_archive_binding)),
        "continuation_terminal": copy.deepcopy(
            continuation.get("continuation_terminal")
        ),
        "reporting_policy": copy.deepcopy(continuation.get("reporting_policy")),
    }
    merged["source"] = merged_source
    return merged


def exact_references() -> dict[str, float]:
    source = load(ED_SOURCE)
    result: dict[str, float] = {}
    for row in source["regimes"]:
        regime = str(row["name"]).replace("-", "_")
        if regime == "strong_weak":
            regime = "strong_weak_u8"
        elif regime == "strong_strong":
            regime = "strong_strong_u8"
        cutoff = int(row["working_cutoff"])
        cell = next(cell for cell in row["cells"] if int(cell["M"]) == cutoff)
        result[regime] = float(cell["E_ED"])
    if set(result) != set(REGIME_ORDER):
        raise UpdateError("same-cutoff exact-reference coverage drifted")
    return result


def job_for(package: Path, regime: str) -> tuple[Path, dict[str, Any]]:
    matches = sorted((package / "jobs").glob(f"*__{regime}__*.json"))
    if len(matches) != 1:
        raise UpdateError(f"{package.name}/{regime}: expected one job spec")
    value = load(matches[0])
    return matches[0], value


def parse_snapshot(path: Path, exact: float) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for expected, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.startswith("AI_LOG "):
            raise UpdateError(f"unexpected progress line: {path}")
        row = json.loads(raw.removeprefix("AI_LOG "))
        if row.get("event") != "hardcoded_adapt_iter" or row.get("depth") != expected:
            raise UpdateError(f"noncanonical accepted-round sequence: {path}")
        energy = float(row["energy"])
        points.append(
            {
                "k": expected,
                "energy": energy,
                "error": abs(energy - exact),
                "selected_position": int(row["selected_position"]),
                "timestamp_utc": str(row["ts_utc"]),
            }
        )
    if not points:
        raise UpdateError(f"empty progress snapshot: {path}")
    return points


def build_route_adapter(
    route: Mapping[str, Any], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    exact = exact_references()
    page8_cells = {
        row["regime_id"]: row
        for row in provenance["phase3_on_plateau_singleton_sixregime_r50"]["cells"]
    }
    cells = []
    for regime in REGIME_ORDER:
        job_path, job = job_for(Path(route["package"]), regime)
        if (
            job.get("regime_id") != regime
            or job.get("nph") != NPH[regime]
            or job.get("target_horizon") != 50
            or job.get("candidate_representation") != "single_pauli_word_v1"
            or job.get("route_contract_sha256") != route["route_sha256"]
        ):
            raise UpdateError(f"{route['key']}/{regime}: job identity drifted")
        current: dict[str, Any] | None = None
        snapshot_name: str | None = None
        completed_name = route["completed"].get(regime)
        if completed_name:
            completed_path = COMPLETED_DIR / str(completed_name)
            completed = load(completed_path)
            verify_self_digest(
                completed,
                label=f"{route['key']}/{regime} completed adapter",
            )
            points = completed.get("points")
            terminal = completed.get("terminal")
            costs = terminal.get("costs") if isinstance(terminal, Mapping) else None
            if (
                completed.get("status")
                != "passed_remote_summary_extract_full_archive_preserved"
                or completed.get("cluster_id") != route["cluster_id"]
                or completed.get("regime_id") != regime
                or completed.get("nph") != NPH[regime]
                or completed.get("controller_rounds_completed") != 50
                or not isinstance(points, list)
                or [row.get("k") for row in points] != list(range(1, 51))
                or not isinstance(terminal, Mapping)
                or terminal.get("k") != 50
                or not isinstance(costs, Mapping)
                or set(costs) != {"N2q", "D2q", "Dc", "W1q", "S_alg"}
                or not math.isclose(
                    float(completed.get("exact_same_cutoff_energy")),
                    exact[regime],
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ):
                raise UpdateError(
                    f"{route['key']}/{regime}: completed adapter identity drifted"
                )
            retrieved_archive = bind_retrieved_full_archive(
                completed, adapter_name=str(completed_name)
            )
            current = {
                "status": "completed_authenticated_remote_summary",
                "points": copy.deepcopy(points),
                "latest": {
                    "k": int(terminal["k"]),
                    "energy": float(terminal["energy"]),
                    "error": float(terminal["error"]),
                },
                "cluster_id": int(completed["cluster_id"]),
                "proc_id": int(completed["proc_id"]),
                "costs": {key: int(costs[key]) for key in costs},
                "compile": copy.deepcopy(terminal["compile"]),
                "work_components": copy.deepcopy(terminal["work_components"]),
                "source": {
                    "completed_adapter": binding(completed_path),
                    "remote_evidence": copy.deepcopy(completed["source"]),
                    "retrieved_full_archive": retrieved_archive,
                },
                "cost_status": "available_shared_locked_qiskit_compile",
                "s_alg_status": "available_canonical_all_work",
            }
            continuation_name = route.get("continuations", {}).get(regime)
            if continuation_name:
                continuation_path = PAGE12_R70_CONTINUATION_DIR / str(
                    continuation_name
                )
                if continuation_path.is_file():
                    continuation = load(continuation_path)
                    verify_self_digest(
                        continuation,
                        label=f"{route['key']}/{regime} continuation adapter",
                    )
                    continuation_archive = bind_page12_r70_continuation_archive(
                        continuation, adapter_name=str(continuation_name)
                    )
                    current = merge_page12_r70_continuation(
                        current,
                        continuation,
                        regime=regime,
                        completed_adapter_binding=binding(completed_path),
                        continuation_adapter_binding={
                            **binding(continuation_path),
                            "canonical_sha256": continuation["sha256"],
                        },
                        continuation_archive_binding=continuation_archive,
                    )
        else:
            snapshot_name = route["snapshots"].get(regime)
        if current is None and snapshot_name:
            snapshot_path = SNAPSHOT_DIR / str(snapshot_name)
            points = parse_snapshot(snapshot_path, exact[regime])
            current = {
                "status": "live_snapshot_incomplete",
                "points": points,
                "latest": copy.deepcopy(points[-1]),
                "cluster_id": route["cluster_id"],
                "proc_id": int(str(snapshot_name).split(".")[1]),
                "source": binding(snapshot_path),
                "cost_status": "unavailable_until_validated_run_summary",
                "s_alg_status": "unavailable_until_validated_run_summary",
            }
        elif regime in route["terminal_only"]:
            observed = copy.deepcopy(route["terminal_only"][regime])
            observed["error"] = abs(float(observed["energy"]) - exact[regime])
            current = {
                "status": observed.pop("status"),
                "points": [
                    {
                        "k": int(observed["k"]),
                        "energy": float(observed["energy"]),
                        "error": float(observed["error"]),
                    }
                ],
                "latest": {
                    "k": int(observed["k"]),
                    "energy": float(observed["energy"]),
                    "error": float(observed["error"]),
                },
                "cluster_id": int(observed["cluster_id"]),
                "proc_id": int(observed["proc_id"]),
                "source_limitation": observed["source_limitation"],
                "cost_status": "unrecoverable_from_lost_archive",
                "s_alg_status": "unrecoverable_from_lost_archive",
            }
        cells.append(
            {
                "regime_id": regime,
                "regime_label": REGIME_LABELS[regime],
                "nph": NPH[regime],
                "exact_same_cutoff_energy": exact[regime],
                "append_adapt": copy.deepcopy(page8_cells[regime]["append_adapt"]),
                "phase0_route": current,
                "status": current["status"] if current else "pending_not_started",
                "job": binding(job_path),
            }
        )
    all_completed = all(_authenticated_completion(cell["status"]) for cell in cells)
    unsigned = {
        "schema": "paper_i_phase0_route_progress_adapter_v1",
        "page_id": route["page_id"],
        "route_key": route["key"],
        "route_contract_sha256": route["route_sha256"],
        "cluster_id": route["cluster_id"],
        "status": (
            "completed_six_regime_evidence_ready"
            if all_completed
            else "partial_progress_diagnostic"
        ),
        "paper_evidence_adopted": False,
        "same_cutoff_reference": binding(ED_SOURCE),
        "cells": cells,
        "limitations": [
            "live snapshots are incomplete progress diagnostics, not completed evidence",
            "terminal-only points lack their lost archive trajectory, Qiskit tuple, and S_alg",
            "pending cells have no accepted rounds",
            "completed-summary cells bind a local full archive only after its SHA-256 and size match the preserved remote archive identity",
            "round-70 continuation trajectories retain the authenticated round-50 Qiskit and S_alg tuple; they do not substitute round-70 resources",
        ],
    }
    unsigned["sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return unsigned


def format_error(value: float) -> str:
    return f"{value:.2e}"


def format_s_alg(value: int) -> str:
    mantissa, exponent = f"{int(value):.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def format_cost_tuple(value: Mapping[str, Any] | None) -> str:
    if not isinstance(value, Mapping):
        return "--"
    fields = ("N2q", "D2q", "Dc", "W1q", "S_alg")
    if any(value.get(field) is None for field in fields):
        return "--"
    return "(" + ",".join(
        format_s_alg(int(value[field]))
        if field == "S_alg"
        else str(int(value[field]))
        for field in fields
    ) + ")"


def render_route(route: Mapping[str, Any], adapter: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    stem = (
        "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
        f"{route['key']}_page{route['page']}"
    )
    page_pdf = REPORT_DIR / f"{stem}.pdf"
    page_png = REPORT_DIR / f"{stem}.png"
    adapter_path = REPORT_DIR / f"{stem}_adapter.json"
    adapter_path.write_text(
        json.dumps(adapter, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    mpl.rcParams.update({"font.family": "serif", "font.size": 7.4})
    fig = plt.figure(figsize=(11, 8.5))
    grid = fig.add_gridspec(3, 3, height_ratios=(1.0, 1.0, 0.58), hspace=0.34, wspace=0.25)
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]
    for index, (axis, cell) in enumerate(zip(axes, adapter["cells"], strict=True)):
        append = cell["append_adapt"]
        axis.plot(
            [row["k"] for row in append["points"]],
            [max(float(row["error"]), PLOT_FLOOR) for row in append["points"]],
            color="#4C78A8",
            lw=1.45,
        )
        current = cell["phase0_route"]
        if current:
            points = current["points"]
            completed = _authenticated_completion(current["status"])
            if len(points) > 1:
                axis.plot(
                    [row["k"] for row in points],
                    [max(float(row["error"]), PLOT_FLOOR) for row in points],
                    color=route["color"],
                    lw=1.8,
                )
            marker = "X" if current["status"] == "completed_science_archive_lost" else "o"
            axis.scatter(
                [current["latest"]["k"]],
                [max(float(current["latest"]["error"]), PLOT_FLOOR)],
                color=route["color"],
                marker=marker,
                s=38,
                zorder=4,
            )
            axis.text(
                0.97,
                0.08,
                f"{'complete' if completed else current['status'].replace('_', ' ')}\n"
                f"k={current['latest']['k']}, |dE|={format_error(float(current['latest']['error']))}",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.2,
                color=route["color"],
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            )
        else:
            axis.text(
                0.5,
                0.14,
                "Phase-0 cell pending / not started",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=6.8,
                color=route["color"],
                bbox={"facecolor": "white", "edgecolor": route["color"], "alpha": 0.85},
            )
        axis.set_yscale("log")
        axis.set_xlim(0, 70)
        axis.grid(True, which="major", alpha=0.22, lw=0.5)
        axis.set_title(f"{cell['regime_label']} ($n_{{ph}}={cell['nph']}$)", fontsize=8.4)
        if index // 3 == 1:
            axis.set_xlabel("ADAPT controller round")
        if index % 3 == 0:
            axis.set_ylabel(r"same-cutoff $|\Delta E|$")
    legend_handles = [
            Line2D([0], [0], color="#4C78A8", lw=1.45, label="Append-ADAPT"),
            Line2D(
                [0], [0], color=route["color"], lw=1.8, marker="o", markersize=5,
                label=f"{route['short_label']} (available trajectory)",
            ),
    ]
    if any(
        cell["phase0_route"]
        and cell["phase0_route"]["status"] == "completed_science_archive_lost"
        for cell in adapter["cells"]
    ):
        legend_handles.append(
            Line2D(
                [0], [0], color=route["color"], lw=0, marker="X", markersize=6,
                label="terminal observation; archive lost",
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.953),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(route["title"], fontsize=10.8, fontweight="bold", y=0.988)

    table_axis = fig.add_subplot(grid[2, :])
    table_axis.axis("off")
    rows = []
    for cell in adapter["cells"]:
        current = cell["phase0_route"]
        if not current:
            continue
        route_error = float(current["latest"]["error"])
        append_error = float(cell["append_adapt"]["terminal"]["error"])
        rows.append(
            [
                cell["regime_label"],
                (
                    "complete"
                    if _authenticated_completion(current["status"])
                    else current["status"].replace("_", " ")
                ),
                str(current["latest"]["k"]),
                format_error(route_error),
                format_error(append_error),
                format_cost_tuple(current.get("costs")),
                format_cost_tuple(cell["append_adapt"]["terminal"]),
            ]
        )
    table = table_axis.table(
        cellText=rows,
        colLabels=[
            "Regime",
            "Phase-0 status",
            "latest k",
            r"Phase-0 $|\Delta E|$",
            r"Append $|\Delta E_{50}|$",
            "Phase-0 Qiskit / $S_{alg}$",
            "Append Qiskit / $S_{alg}$",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=(0.13, 0.18, 0.06, 0.12, 0.12, 0.20, 0.20),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.9)
    table.scale(1.0, 0.82)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EAEAEA")
    all_completed = all(_authenticated_completion(cell["status"]) for cell in adapter["cells"])
    footer = (
        r"Tuple order: $(N_{2q},D_{2q},D_c,W_{1q},S_{alg})$; "
        r"$S_{alg}$ uses X.YeZ notation."
        if all_completed
        else r"Tuple order: $(N_{2q},D_{2q},D_c,W_{1q},S_{alg})$; "
        r"$S_{alg}$ uses X.YeZ notation. Pending/lost summaries remain unreported."
    )
    fig.text(
        0.5,
        0.018,
        footer,
        ha="center",
        fontsize=6.5,
    )
    fig.savefig(page_png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    # Matplotlib's independently generated vector pages can reuse PDF
    # resource identifiers when concatenated with this long-lived report.
    # Embed the visually verified high-resolution render to keep each new
    # page self-contained and collision-free.
    from PIL import Image

    with Image.open(page_png) as source:
        source.convert("RGB").save(page_pdf, format="PDF", resolution=240.0)
    return page_pdf, page_png, adapter_path


def update_combined(
    provenance: dict[str, Any],
    rendered: list[tuple[Mapping[str, Any], Mapping[str, Any], Path, Path, Path]],
) -> dict[str, Any]:
    from pypdf import PdfReader, PdfWriter

    current = binding(TARGET_PDF)
    declared = provenance["outputs"]["partial_progress_pdf"]
    page_count = int(provenance["layout"].get("page_count", -1))
    supported = page_count in (10, 12)
    if (
        current["sha256"] != declared["sha256"]
        or current["size_bytes"] != declared["size_bytes"]
        or not supported
        or (page_count == 12 and provenance["layout"].get("page_11") != ROUTES[0]["page_id"])
        or (page_count == 12 and provenance["layout"].get("page_12") != ROUTES[1]["page_id"])
    ):
        raise UpdateError("target PDF/provenance is not a supported Phase-0 page state")
    original = PdfReader(str(TARGET_PDF), strict=False)
    if len(original.pages) != page_count:
        raise UpdateError("target PDF page count drifted")
    writer = PdfWriter()
    for page in original.pages[:10]:
        writer.add_page(page)
    page_readers = []
    for _, _, page_pdf, _, _ in rendered:
        page_reader = PdfReader(str(page_pdf), strict=False)
        page_readers.append(page_reader)
        if len(page_reader.pages) != 1:
            raise UpdateError(f"single-page artifact required: {page_pdf}")
        writer.add_page(page_reader.pages[0])
    temporary_pdf = TARGET_PDF.with_name(f".{TARGET_PDF.name}.phase0.tmp")
    temporary_provenance = TARGET_PROVENANCE.with_name(
        f".{TARGET_PROVENANCE.name}.phase0.tmp"
    )
    rollback = TARGET_PDF.with_name(f".{TARGET_PDF.name}.phase0.rollback")
    for path in (temporary_pdf, temporary_provenance, rollback):
        if path.exists() or path.is_symlink():
            raise UpdateError(f"stale temporary exists: {path}")
    try:
        with temporary_pdf.open("xb") as stream:
            writer.write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        if len(PdfReader(str(temporary_pdf), strict=False).pages) != 12:
            raise UpdateError("combined PDF must have 12 pages")
        updated = copy.deepcopy(provenance)
        updated["layout"]["page_11"] = ROUTES[0]["page_id"]
        updated["layout"]["page_12"] = ROUTES[1]["page_id"]
        updated["layout"]["page_count"] = 12
        for route, adapter, page_pdf, page_png, adapter_path in rendered:
            key = f"phase0_{route['key']}_progress"
            updated[key] = {
                "schema": "paper_i_phase0_route_progress_report_v1",
                "page_id": route["page_id"],
                "status": adapter["status"],
                "paper_evidence_adopted": False,
                "adapter": {**binding(adapter_path), "canonical_sha256": adapter["sha256"]},
                "cells": copy.deepcopy(adapter["cells"]),
                "limitations": copy.deepcopy(adapter["limitations"]),
                "outputs": {"page_pdf": binding(page_pdf), "page_png": binding(page_png)},
            }
            updated["outputs"][f"{key}_pdf"] = binding(page_pdf)
            updated["outputs"][f"{key}_png"] = binding(page_png)
            updated["outputs"][f"{key}_adapter"] = {
                **binding(adapter_path),
                "canonical_sha256": adapter["sha256"],
            }
        combined = binding(temporary_pdf)
        combined["path"] = str(TARGET_PDF.resolve())
        updated["outputs"]["partial_progress_pdf"] = combined
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
        "page_count": 12,
        "pdf": binding(TARGET_PDF),
    }


def _page_content_sha256(page: Any) -> str:
    contents = page.get_contents()
    data = b"" if contents is None else contents.get_data()
    return hashlib.sha256(data).hexdigest()


def replace_route_page(
    route: Mapping[str, Any],
    adapter: Mapping[str, Any],
    page_pdf: Path,
    page_png: Path,
    adapter_path: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace one Phase-0 route page while preserving every other page."""
    from pypdf import PdfReader, PdfWriter

    current = binding(TARGET_PDF)
    declared = provenance["outputs"]["partial_progress_pdf"]
    page_count = int(provenance["layout"].get("page_count", -1))
    page_number = int(route["page"])
    if (
        current["sha256"] != declared["sha256"]
        or current["size_bytes"] != declared["size_bytes"]
        or page_count < page_number
        or provenance["layout"].get(f"page_{page_number}") != route["page_id"]
    ):
        raise UpdateError("target PDF/provenance is not a supported route-page state")
    original = PdfReader(str(TARGET_PDF), strict=False)
    replacement_reader = PdfReader(str(page_pdf), strict=False)
    if len(original.pages) != page_count or len(replacement_reader.pages) != 1:
        raise UpdateError("route-page replacement requires the declared report and one page")
    before = [_page_content_sha256(page) for page in original.pages]

    writer = PdfWriter()
    replacement_index = page_number - 1
    for index, page in enumerate(original.pages):
        writer.add_page(replacement_reader.pages[0] if index == replacement_index else page)

    token = str(route["key"])
    temporary_pdf = TARGET_PDF.with_name(f".{TARGET_PDF.name}.{token}.tmp")
    temporary_provenance = TARGET_PROVENANCE.with_name(
        f".{TARGET_PROVENANCE.name}.{token}.tmp"
    )
    rollback = TARGET_PDF.with_name(f".{TARGET_PDF.name}.{token}.rollback")
    for path in (temporary_pdf, temporary_provenance, rollback):
        if path.exists() or path.is_symlink():
            raise UpdateError(f"stale temporary exists: {path}")
    try:
        with temporary_pdf.open("xb") as stream:
            writer.write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        combined_reader = PdfReader(str(temporary_pdf), strict=False)
        if len(combined_reader.pages) != page_count:
            raise UpdateError("route-page replacement changed the page count")
        after = [_page_content_sha256(page) for page in combined_reader.pages]
        if any(
            before[index] != after[index]
            for index in range(page_count)
            if index != replacement_index
        ):
            raise UpdateError("route-page replacement changed a preserved page")

        updated = copy.deepcopy(dict(provenance))
        key = f"phase0_{route['key']}_progress"
        updated[key] = {
            "schema": "paper_i_phase0_route_progress_report_v1",
            "page_id": route["page_id"],
            "status": adapter["status"],
            "paper_evidence_adopted": False,
            "adapter": {**binding(adapter_path), "canonical_sha256": adapter["sha256"]},
            "cells": copy.deepcopy(adapter["cells"]),
            "limitations": copy.deepcopy(adapter["limitations"]),
            "outputs": {"page_pdf": binding(page_pdf), "page_png": binding(page_png)},
        }
        updated["outputs"][f"{key}_pdf"] = binding(page_pdf)
        updated["outputs"][f"{key}_png"] = binding(page_png)
        updated["outputs"][f"{key}_adapter"] = {
            **binding(adapter_path),
            "canonical_sha256": adapter["sha256"],
        }
        combined = binding(temporary_pdf)
        combined["path"] = str(TARGET_PDF.resolve())
        updated["outputs"]["partial_progress_pdf"] = combined
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
        except BaseException:
            os.replace(rollback, TARGET_PDF)
            raise
        rollback.unlink(missing_ok=True)
    except BaseException:
        temporary_pdf.unlink(missing_ok=True)
        temporary_provenance.unlink(missing_ok=True)
        rollback.unlink(missing_ok=True)
        raise
    return {
        "status": "replaced_route_page_in_place",
        "page": page_number,
        "page_count": page_count,
        "pdf": binding(TARGET_PDF),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--render-route",
        choices=[str(route["key"]) for route in ROUTES],
        help="internal isolated-render entrypoint",
    )
    mode.add_argument(
        "--replace-route",
        choices=[str(route["key"]) for route in ROUTES],
        help="render and replace only the selected route page",
    )
    args = parser.parse_args()
    provenance = load(TARGET_PROVENANCE)
    if args.render_route:
        route = next(route for route in ROUTES if route["key"] == args.render_route)
        adapter = build_route_adapter(route, provenance)
        page_pdf, page_png, adapter_path = render_route(route, adapter)
        print(
            json.dumps(
                {
                    "page_pdf": str(page_pdf),
                    "page_png": str(page_png),
                    "adapter": str(adapter_path),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.replace_route:
        route = next(route for route in ROUTES if route["key"] == args.replace_route)
        adapter = build_route_adapter(route, provenance)
        page_pdf, page_png, adapter_path = render_route(route, adapter)
        result = replace_route_page(
            route, adapter, page_pdf, page_png, adapter_path, provenance
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    rendered = []
    for route in ROUTES:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--render-route", str(route["key"])],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        paths = json.loads(completed.stdout)
        page_pdf = Path(paths["page_pdf"])
        page_png = Path(paths["page_png"])
        adapter_path = Path(paths["adapter"])
        adapter = load(adapter_path)
        rendered.append((route, adapter, page_pdf, page_png, adapter_path))
    result = update_combined(provenance, rendered)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
