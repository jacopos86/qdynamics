#!/usr/bin/env python3
"""Append the completed Phase-III Qiskit/no-lanes singleton cells as page 9."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tarfile
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting.add_paper_i_phase3_on_plateau_singleton_page import (
    _compile_prefix_mapping,
    _normalize_compiled_cost,
)
from pipelines.reporting.paper_i_mixed_horizon_continuation import (
    MixedHorizonContinuationError,
    STRONG_HOLSTEIN_REGIMES,
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
    "phase3_qiskit_denominator_no_lanes_page9"
)
PAGE_PDF = REPORT_DIR / f"{ASSET_STEM}.pdf"
PAGE_PNG = REPORT_DIR / f"{ASSET_STEM}.png"
ADAPTER_PATH = REPORT_DIR / f"{ASSET_STEM}_adapter.json"
CONTINUATION_ADAPTER_PATH = REPORT_DIR / (
    f"{ASSET_STEM}_strong_sector_r70_continuations.json"
)

PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_global_singleton_phase3_qiskit_denominator_no_lanes_"
    "tau1em6_r50_20260807_v3_chtc"
)
RETRIEVED_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "retrieved_chtc_20260807_phase3_qiskit_denominator_no_lanes_v1"
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
COMPLETED = {
    "weak_weak": {
        "directory": "9588784.0_weak_weak_20260807",
        "cluster_id": 9588784,
        "proc_id": 0,
        "archive_sha256": (
            "0b9a8903da202b0e995ffb47078cfc8b63c680b054aa25983603d079ab7223a0"
        ),
        "archive_size_bytes": 2049328364,
        "compact_archive_sha256": (
            "aca94f1898c78feb87eea2f21a6b79158c0e341f24d30f6b9c9e158c5e88e7e9"
        ),
    },
    "intermediate_weak": {
        "directory": "9588784.1_intermediate_weak_20260807",
        "cluster_id": 9588784,
        "proc_id": 1,
        "archive_sha256": (
            "a6f8dd854a006d4a732982109216622b5f9fe76182c232657a7b612b5204c0d1"
        ),
        "archive_size_bytes": 2184236282,
        "compact_archive_sha256": (
            "8f5343028e3a926c1587b6c2fb90eb1c36f2900b9697e946de1c38f135535e66"
        ),
    },
    "strong_weak_u8": {
        "directory": "9588784.2_strong_weak_u8_20260807",
        "cluster_id": 9588784,
        "proc_id": 2,
        "archive_sha256": (
            "ce14dbaa18cc3db62a92bbb62322620598e14d53bface93e1abeb96e352887b1"
        ),
        "archive_size_bytes": 2183079540,
        "compact_archive_sha256": (
            "14f9685b8b9fef5d4021a1c71016a3146f852b3a351db0be7f5f046701e9cc44"
        ),
    },
    "weak_strong": {
        "directory": "9588784.3_weak_strong_20260808",
        "cluster_id": 9588784,
        "proc_id": 3,
        "archive_sha256": (
            "c7bf22315829bdff310d7afa839c8e7bca280a822198e5b6d90d416783f0205f"
        ),
        "archive_size_bytes": 3360915798,
        "compact_archive_sha256": (
            "3ea4db41e212167b45d68d4f28f93de5c8cb9595383c89c7182b266266cc13ee"
        ),
        "remote_archive_path": (
            "/staging/jsstrobel/paper_i_ra_adapt_completed_20260808/raw/"
            "phase3_qiskit_denominator_no_lanes__weak_strong__nph7__"
            "ra_global_singleton_plateau_commutation__9588784__3.tar.gz"
        ),
    },
    "intermediate_strong": {
        "directory": "9588784.4_intermediate_strong_20260808",
        "cluster_id": 9588784,
        "proc_id": 4,
        "archive_sha256": (
            "d47d6bd0be68946c2959f22aa381a6dfb50feff75eaf99dfa90fd328d1f60246"
        ),
        "archive_size_bytes": 5342296973,
        "compact_archive_sha256": (
            "3dc3dbe65caa7f44e059f494885dca73437b2d5bf039135edf65c7d57515caf0"
        ),
        "remote_archive_path": (
            "/staging/jsstrobel/paper_i_ra_adapt_completed_20260808/raw/"
            "phase3_qiskit_denominator_no_lanes__intermediate_strong__nph7__"
            "ra_global_singleton_plateau_commutation__9588784__4.tar.gz"
        ),
    },
}
EXPECTED_ROUTE = (
    "paper_i_ra_adapt__single_pauli_word_v1__"
    "insertion_commutation_plateau_v2__global_guarded_singleton_phase_i__"
    "identity_phase_ii__stationary_source_response_v1__"
    "all_phase_resource_weighting_v1__"
    "qiskit_full_ansatz_positive_marginal_denominator_phase3_only_"
    "no_lanes_tau1em6_v1"
)
EXPECTED_ROUTE_SHA256 = (
    "e649eaa50428f6f396c4ab6cf25542a21add58115beb61d42df32408ad1399b6"
)
PAGE_ID = "phase3_qiskit_denominator_no_lanes_singleton_r50_partial_v1"
REPORT_KEY = "phase3_qiskit_denominator_no_lanes_singleton_r50"
PLOT_FLOOR = 1.0e-16


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


def load_continuation_adapter() -> dict[str, dict[str, Any]]:
    """Load a fetched k=70 overlay when the retrieval workflow has written it."""

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


def trace(summary: Mapping[str, Any], *, label: str) -> list[dict[str, Any]]:
    rows = summary.get("accepted_error_trace")
    if summary.get("schema") != "paper_i_run_summary_v1" or not isinstance(rows, list):
        raise UpdateError(f"{label}: invalid Paper-I summary")
    if summary.get("available_controller_rounds") != 50 or len(rows) != 50:
        raise UpdateError(f"{label}: expected exactly 50 rounds")
    result: list[dict[str, Any]] = []
    for expected, raw in enumerate(rows, 1):
        if not isinstance(raw, Mapping) or raw.get("controller_round") != expected:
            raise UpdateError(f"{label}: noncanonical trace")
        energy = float(raw["accepted_energy"])
        error = float(raw["absolute_energy_error"])
        exact = float(summary["provenance"]["exact_same_cutoff_energy"])
        if not math.isclose(error, abs(energy - exact), rel_tol=1e-11, abs_tol=1e-12):
            raise UpdateError(f"{label}: error trace math drifted")
        result.append({"k": expected, "error": error})
    return result


def compile_terminal(summary: Mapping[str, Any]) -> dict[str, Any]:
    requested = summary.get("requested_rounds")
    if not isinstance(requested, list) or len(requested) != 1:
        raise UpdateError("new route lacks its unique round-50 observation")
    row = requested[0]
    if row.get("controller_round") != 50 or row.get("status") != "available":
        raise UpdateError("new route round-50 observation drifted")
    compiled = _normalize_compiled_cost(_compile_prefix_mapping(row["prefix"]))
    resources = row["resources"]
    expected = {
        "N2q": resources["compiled_two_qubit_count"],
        "D2q": resources["compiled_two_qubit_depth"],
        "Dc": resources["compiled_total_depth"],
    }
    if any(compiled[key] != value for key, value in expected.items()):
        raise UpdateError("serialized and recompiled Qiskit costs disagree")
    return {
        key: compiled[key]
        for key in ("N2q", "D2q", "Dc", "W1q", "B1q", "qiskit_version")
    }


def load_current(regime: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    directory = RETRIEVED_DIR / str(spec["directory"])
    compact = RETRIEVED_DIR / f"{directory.name}.tar.gz"
    if sha256(compact) != spec["compact_archive_sha256"]:
        raise UpdateError(f"{regime}: compact retrieval archive drifted")
    worker = load(directory / "worker_receipt.json")
    manifest_path = next(directory.glob("runs/*/execution_manifest.json"))
    summary_path = next(directory.glob("runs/*/summary/summary.json"))
    manifest = load(manifest_path)
    summary = load(summary_path)
    job_path = next((PACKAGE_DIR / "jobs").glob(f"*__{regime}__*.json"))
    job = load(job_path)
    execution_id = str(job["execution_id"])
    if (
        worker.get("status") != "passed"
        or worker.get("execution_id") != execution_id
        or worker.get("controller_rounds_completed") != 50
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
    artifact = next(
        row
        for row in worker["artifacts"]
        if row["path"].endswith("/summary/summary.json")
    )
    if (
        artifact["sha256"] != sha256(summary_path)
        or artifact["size_bytes"] != summary_path.stat().st_size
    ):
        raise UpdateError(f"{regime}: worker-to-summary binding drifted")
    provenance = summary["provenance"]
    if (
        provenance.get("route_profile") != EXPECTED_ROUTE
        or provenance.get("route_contract_sha256") != EXPECTED_ROUTE_SHA256
        or provenance.get("candidate_representation") != "single_pauli_word_v1"
    ):
        raise UpdateError(f"{regime}: scientific route identity drifted")
    points = trace(summary, label=f"{regime} current")
    compiled = compile_terminal(summary)
    plateau = summary["effective_plateau"]
    return {
        "status": "complete",
        "execution_id": execution_id,
        "cluster_id": spec["cluster_id"],
        "proc_id": spec["proc_id"],
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
        "source_bindings": {
            "compact_archive": binding(compact),
            "worker_receipt": binding(directory / "worker_receipt.json"),
            "execution_manifest": binding(manifest_path),
            "summary": binding(summary_path),
            "job": binding(job_path),
            "remote_full_archive": {
                "sha256": spec["archive_sha256"],
                "size_bytes": spec["archive_size_bytes"],
                "preserved_location": spec.get(
                    "remote_archive_path",
                    (
                        "/staging/jsstrobel/"
                        "paper_i_ra_adapt_completed_20260807/"
                        + execution_id
                        + f"__{spec['cluster_id']}__{spec['proc_id']}.tar.gz"
                    ),
                ),
            },
        },
    }


def archive_summary(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with tarfile.open(path, "r:gz") as archive:
        member = archive.getmember("worker_outputs/summary.json")
        stream = archive.extractfile(member)
        if stream is None:
            raise UpdateError(f"summary is unreadable: {path}")
        raw = stream.read()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise UpdateError(f"summary is invalid: {path}")
    return value, {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "summary_sha256": hashlib.sha256(raw).hexdigest(),
        "summary_size_bytes": len(raw),
    }


def load_stationary(
    provenance: Mapping[str, Any], regime: str
) -> dict[str, Any] | None:
    source = next(
        (
            row
            for row in provenance["included_sources"]
            if row.get("regime_id") == regime
            and row.get("route_id") == "ra_singleton_plateau"
        ),
        None,
    )
    if source is None:
        return None
    candidates = sorted((REPO_ROOT / "raw_outputs").rglob(source["attempt_path"]))
    archive = next(
        (path for path in candidates if sha256(path) == source["attempt_sha256"]),
        None,
    )
    if archive is None:
        raise UpdateError(f"{regime}: page-2 source archive is unavailable")
    summary, source_binding = archive_summary(archive)
    if (
        source_binding["sha256"] != source["attempt_sha256"]
        or source_binding["summary_sha256"] != source["summary_file_sha256"]
    ):
        raise UpdateError(f"{regime}: page-2 source binding drifted")
    points = trace(summary, label=f"{regime} stationary")
    if not math.isclose(
        points[-1]["error"],
        source["terminal"]["error"],
        rel_tol=1e-11,
        abs_tol=1e-12,
    ):
        raise UpdateError(f"{regime}: page-2 terminal drifted")
    return {
        "execution_id": source["execution_id"],
        "points": points,
        "marker": copy.deepcopy(source["marker"]),
        "terminal": copy.deepcopy(source["terminal"]),
        "source": source_binding,
    }


def build_adapter(provenance: Mapping[str, Any]) -> dict[str, Any]:
    page8 = provenance["phase3_on_plateau_singleton_sixregime_r50"]
    page8_cells = {row["regime_id"]: row for row in page8["cells"]}
    continuations = load_continuation_adapter()
    cells = []
    for regime in REGIME_ORDER:
        append = copy.deepcopy(page8_cells[regime]["append_adapt"])
        stationary = load_stationary(provenance, regime) if regime in COMPLETED else None
        current = load_current(regime, COMPLETED[regime]) if regime in COMPLETED else None
        continuation_overlay = continuations.get(regime)
        if continuation_overlay is not None and current is None:
            raise UpdateError(
                f"{regime}: continuation is present before its k=50 base"
            )
        if current is not None:
            current = decorate_route(
                current,
                regime_id=regime,
                continuation_points=(
                    continuation_overlay["points"]
                    if continuation_overlay is not None
                    else None
                ),
                continuation_status=(
                    continuation_overlay["status"]
                    if continuation_overlay is not None
                    else "pending"
                    if regime in STRONG_HOLSTEIN_REGIMES
                    else None
                ),
                continuation_source=(
                    continuation_overlay["source"]
                    if continuation_overlay is not None
                    else None
                ),
            )
        continuation = (
            copy.deepcopy(current["continuation"])
            if current is not None
            else missing_route_continuation_status(regime_id=regime)
        )
        cells.append(
            {
                "regime_id": regime,
                "regime_label": REGIME_LABELS[regime],
                "nph": NPH[regime],
                "append_adapt": append,
                "stationary_ra_plateau": stationary,
                "stationary_comparator_status": (
                    "available" if stationary else "absent_from_page_2"
                ),
                "phase3_qiskit_no_lanes": current,
                "current_status": "complete" if current else "pending_on_chtc",
                "continuation_status": continuation,
            }
        )
    unsigned = {
        "schema": "paper_i_phase3_qiskit_no_lanes_page9_adapter_v1",
        "page_id": PAGE_ID,
        "status": f"partial_{len(COMPLETED)}_of_6_complete",
        "paper_evidence_adopted": False,
        "route_profile": EXPECTED_ROUTE,
        "route_contract_sha256": EXPECTED_ROUTE_SHA256,
        "comparison": "new route vs Append-ADAPT vs page-2 stationary RA plateau",
        "horizon_policy": horizon_policy(),
        "cells": cells,
    }
    unsigned["sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    ADAPTER_PATH.write_text(json.dumps(unsigned, indent=2, sort_keys=True) + "\n")
    return unsigned


def format_s(value: int) -> str:
    mantissa, exponent = f"{float(value):.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def format_error(value: float) -> str:
    return f"{value:.2e}"


def format_cost(value: Mapping[str, Any]) -> str:
    return "(" + ",".join(str(value[key]) for key in ("N2q", "D2q", "Dc", "W1q")) + ")"


def render(adapter: Mapping[str, Any]) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    mpl.rcParams.update({"font.family": "serif", "font.size": 7.5})
    fig = plt.figure(figsize=(11, 8.5))
    grid = fig.add_gridspec(3, 3, height_ratios=(1.0, 1.0, 0.72), hspace=0.34, wspace=0.25)
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]
    for index, (axis, cell) in enumerate(zip(axes, adapter["cells"], strict=True)):
        append = cell["append_adapt"]
        stationary = cell["stationary_ra_plateau"]
        axis.plot(
            [p["k"] for p in append["points"]],
            [max(float(p["error"]), PLOT_FLOOR) for p in append["points"]],
            color="#4C78A8", lw=1.45,
        )
        if stationary:
            axis.plot(
                [p["k"] for p in stationary["points"]],
                [max(float(p["error"]), PLOT_FLOOR) for p in stationary["points"]],
                color="#009E73", lw=1.45, ls="--",
            )
        current = cell["phase3_qiskit_no_lanes"]
        if current:
            trajectory = current["trajectory_points"]
            axis.plot(
                [p["k"] for p in trajectory],
                [max(float(p["error"]), PLOT_FLOOR) for p in trajectory],
                color="#D55E00", lw=1.8,
            )
            axis.scatter(
                [current["marker"]["k"]],
                [max(float(current["marker"]["error"]), PLOT_FLOOR)],
                color="#D55E00", marker="*", s=42, zorder=4,
            )
        else:
            axis.text(
                0.5, 0.12, "base k=50 result pending; continuation unavailable",
                transform=axis.transAxes, ha="center", va="center",
                color="#D55E00", fontsize=7.0,
                bbox={"facecolor": "white", "edgecolor": "#D55E00", "alpha": 0.85},
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
        axis.set_title(f"{cell['regime_label']} ($n_{{ph}}={cell['nph']}$)", fontsize=8.5)
        if index // 3 == 1:
            axis.set_xlabel("ADAPT controller round")
        if index % 3 == 0:
            axis.set_ylabel(r"same-cutoff $|\Delta E|$")
    legend = [
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
            color="#D55E00",
            lw=1.8,
            marker="*",
            markersize=6,
            label="Phase-III Qiskit denominator, no lanes",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        (
            "Global-singleton plateau insertion: Phase-III Qiskit marginal "
            "denominator without lanes"
        ),
        fontsize=11.0, fontweight="bold", y=0.988,
    )

    table_axis = fig.add_subplot(grid[2, :])
    table_axis.axis("off")
    rows = []
    for cell in adapter["cells"]:
        current = cell["phase3_qiskit_no_lanes"]
        if not current:
            continue
        compared_routes = [
            ("Qiskit/no lanes", current),
            ("Append-ADAPT", cell["append_adapt"]),
        ]
        if cell["stationary_ra_plateau"] is not None:
            compared_routes.insert(
                1, ("stationary RA", cell["stationary_ra_plateau"])
            )
        for label, route in compared_routes:
            terminal = route.get(
                "paper_facing_fixed_round_50", route["terminal"]
            )
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
        colWidths=(0.18, 0.18, 0.16, 0.28, 0.12),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    table.scale(1.0, 0.68)
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
        or page_count < 8
        or (page_count >= 9 and provenance["layout"].get("page_9") != PAGE_ID)
    ):
        raise UpdateError("target PDF/provenance is not a supported page-9 state")
    old = PdfReader(str(TARGET_PDF), strict=False)
    new = PdfReader(str(PAGE_PDF), strict=False)
    if len(old.pages) != page_count or len(new.pages) != 1:
        raise UpdateError("unexpected PDF page count")
    writer = PdfWriter()
    for page in old.pages[:8]:
        writer.add_page(page)
    writer.add_page(new.pages[0])
    for page in old.pages[9:]:
        writer.add_page(page)
    page_count_after = 9 if page_count == 8 else page_count
    temporary_pdf = TARGET_PDF.with_name(f".{TARGET_PDF.name}.page9.tmp")
    temporary_provenance = TARGET_PROVENANCE.with_name(
        f".{TARGET_PROVENANCE.name}.page9.tmp"
    )
    rollback = TARGET_PDF.with_name(f".{TARGET_PDF.name}.page9.rollback")
    for path in (temporary_pdf, temporary_provenance, rollback):
        if path.exists() or path.is_symlink():
            raise UpdateError(f"stale temporary exists: {path}")
    try:
        with temporary_pdf.open("xb") as stream:
            writer.write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        check = PdfReader(str(temporary_pdf), strict=False)
        if len(check.pages) != page_count_after:
            raise UpdateError("combined PDF page count drifted")
        updated = copy.deepcopy(provenance)
        updated["layout"]["page_9"] = PAGE_ID
        updated["layout"]["page_count"] = page_count_after
        updated[REPORT_KEY] = {
            "schema": "paper_i_phase3_qiskit_no_lanes_page9_report_v1",
            "page_id": PAGE_ID,
            "status": adapter["status"],
            "paper_evidence_adopted": False,
            "horizon_policy": copy.deepcopy(adapter["horizon_policy"]),
            "adapter": {**binding(ADAPTER_PATH), "canonical_sha256": adapter["sha256"]},
            "cells": copy.deepcopy(adapter["cells"]),
            "completed_regimes": sorted(COMPLETED),
            "pending_regimes": [r for r in REGIME_ORDER if r not in COMPLETED],
            "outputs": {"page_pdf": binding(PAGE_PDF), "page_png": binding(PAGE_PNG)},
            "structural_validation": {
                "pages_before": page_count,
                "pages_after": page_count_after,
                "preserved_pages": (
                    8 if page_count == 8 else page_count - 1
                ),
                "preserved_trailing_pages": max(0, page_count - 9),
                "page_9_operation": "append" if page_count == 8 else "replace",
            },
        }
        combined = binding(temporary_pdf)
        combined["path"] = str(TARGET_PDF.resolve())
        updated["outputs"]["partial_progress_pdf"] = combined
        updated["outputs"]["phase3_qiskit_no_lanes_page9_pdf"] = binding(PAGE_PDF)
        updated["outputs"]["phase3_qiskit_no_lanes_page9_png"] = binding(PAGE_PNG)
        updated["outputs"]["phase3_qiskit_no_lanes_page9_adapter"] = {
            **binding(ADAPTER_PATH), "canonical_sha256": adapter["sha256"]
        }
        with temporary_provenance.open("xb") as stream:
            stream.write(
                json.dumps(
                    updated, indent=2, sort_keys=True, allow_nan=False
                ).encode()
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
