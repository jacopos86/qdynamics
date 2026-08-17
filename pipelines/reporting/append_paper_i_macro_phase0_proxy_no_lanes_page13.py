#!/usr/bin/env python3
"""Append or refresh Page 13 of the evolving Paper-I result report.

Page 13 compares the exact Page-1 conventional unwhitened macro-ADAPT
trajectories with the local macro-only gradient-Phase-0 RA campaign.  The
local campaign is serial, so the adapter deliberately supports completed,
live-prefix, and pending cells without promoting a live checkpoint to final
evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
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
PAGE_STEM = (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "macro_phase0_proxy_no_lanes_page13"
)
PAGE_PDF = REPORT_DIR / f"{PAGE_STEM}.pdf"
PAGE_PNG = REPORT_DIR / f"{PAGE_STEM}.png"
ADAPTER_PATH = REPORT_DIR / f"{PAGE_STEM}_adapter.json"
PAGE_ID = "macro_gradient_phase0_macro_phase123_proxy_no_lanes_partial_v1"

CURVE_CACHE = REPO_ROOT / (
    "MATH/paper_details/figures/paper_i_hh_macro_common_accuracy_20260723/"
    "paper_i_hh_macro_common_accuracy_20260723_stationary_page1_macro_"
    "curve_cache.json"
)
ED_SOURCE = REPO_ROOT / (
    "MATH/paper_facing/paper_I_static_scaffold/"
    "paper_i_hh_ed_cutoff_reference_six_regime_20260727.json"
)
PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_macro_gradient_phase0_macro_phase123_proxy_no_lanes_"
    "cap24_tau1em4_r50_20260810_v3_chtc"
)
ACTIVATION_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "macro_gradient_phase0_macro_phase123_proxy_no_lanes_local_20260810_v1/"
    "activation"
)
RUNTIME_DIR = REPO_ROOT / (
    "output/local_runs/"
    "paper_i_macro_gradient_phase0_macro_phase123_proxy_no_lanes_r50_"
    "serial_20260810_v1"
)

PACKAGE_MANIFEST_SHA256 = (
    "1ec606f6162a6a5c83b8f618112cb36ed271d2d233543c8c70d81f5098e3f7fb"
)
SOURCE_ARCHIVE_SHA256 = (
    "b70723e52058275ab31ef654f638d9510f0b81707f731d0b750aa395e539027c"
)
ROUTE_CONTRACT_SHA256 = (
    "1b2f7254a96a27a7f2a262f1b4bc19c886b421a9cbaa5e24c95e354a02f2cf45"
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
BLUE = "#4C78A8"
ROUTE_COLOR = "#009E73"


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


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def verify_self_digest(value: Mapping[str, Any], *, label: str) -> None:
    claimed = value.get("sha256")
    unsigned = {key: row for key, row in value.items() if key != "sha256"}
    if claimed != _canonical_sha256(unsigned):
        raise UpdateError(f"{label}: self digest drifted")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise UpdateError(f"stale temporary exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(
                json.dumps(
                    value,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                ).encode()
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
    if tuple(result) != REGIME_ORDER and set(result) != set(REGIME_ORDER):
        raise UpdateError("same-cutoff exact-reference coverage drifted")
    return result


def parse_ai_log_points(
    text: str, *, exact: float, maximum_k: int | None = None
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for raw in text.splitlines():
        if not raw.startswith("AI_LOG "):
            continue
        try:
            row = json.loads(raw.removeprefix("AI_LOG "))
        except json.JSONDecodeError:
            continue
        if row.get("event") != "hardcoded_adapt_iter":
            continue
        k = int(row["depth"])
        if maximum_k is not None and k > maximum_k:
            continue
        if k != len(points) + 1:
            raise UpdateError("live accepted-round log is not contiguous")
        energy = float(row["energy"])
        points.append(
            {
                "k": k,
                "energy": energy,
                "error": abs(energy - exact),
                "selected_position": int(row["selected_position"]),
                "timestamp_utc": str(row["ts_utc"]),
            }
        )
    return points


def _jobs() -> dict[str, tuple[Path, dict[str, Any]]]:
    result: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted((PACKAGE_DIR / "jobs").glob("*.json")):
        job = load(path)
        regime = str(job.get("regime_id"))
        if regime in result:
            raise UpdateError(f"duplicate local route job for {regime}")
        if (
            regime not in REGIME_ORDER
            or job.get("nph") != NPH[regime]
            or job.get("target_horizon") != 50
            or job.get("candidate_representation") != "macro_generator_v1"
            or job.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
            or job.get("execution_authorized") is not False
            or job.get("submission_authorized") is not False
        ):
            raise UpdateError(f"local route job identity drifted: {path}")
        result[regime] = (path, job)
    if set(result) != set(REGIME_ORDER):
        raise UpdateError("local route job coverage drifted")
    return result


def _page1_blue_sources(
    provenance: Mapping[str, Any], cache: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    included = provenance.get("included_sources")
    curves = cache.get("curves")
    if not isinstance(included, list) or not isinstance(curves, Mapping):
        raise UpdateError("Page-1 blue-curve authority is incomplete")
    result: dict[str, dict[str, Any]] = {}
    for regime in REGIME_ORDER:
        execution_id = f"core__{regime}__nph{NPH[regime]}__append_macro"
        rows = [row for row in included if row.get("execution_id") == execution_id]
        if len(rows) != 1:
            raise UpdateError(f"Page-1 blue source coverage drifted: {regime}")
        source = rows[0]
        terminal = source.get("terminal")
        regime_curves = curves.get(regime)
        raw_points = (
            regime_curves.get("append")
            if isinstance(regime_curves, Mapping)
            else None
        )
        if (
            source.get("candidate_representation") != "macro_generator_v1"
            or source.get("method_family") != "append"
            or source.get("route_id") != "append_macro"
            or source.get("plotted_point_count") != 51
            or not isinstance(terminal, Mapping)
            or terminal.get("k") != 50
            or not isinstance(raw_points, list)
            or [row.get("k") for row in raw_points] != list(range(1, 51))
            or not math.isclose(
                float(raw_points[-1]["error"]),
                float(terminal["error"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise UpdateError(f"Page-1 blue source identity drifted: {regime}")
        result[regime] = {
            "execution_id": execution_id,
            "label": "Conventional unwhitened ADAPT",
            "points": copy.deepcopy(raw_points),
            "terminal": copy.deepcopy(dict(terminal)),
            "marker": copy.deepcopy(source["marker"]),
            "source_binding": {
                key: copy.deepcopy(source[key])
                for key in (
                    "attempt_path",
                    "attempt_sha256",
                    "result_file_sha256",
                    "summary_file_sha256",
                    "worker_receipt_sha256",
                    "source_receipt_index",
                )
            },
        }
    return result


def _completed_route(execution_id: str, *, exact: float) -> dict[str, Any] | None:
    run_root = RUNTIME_DIR / "runs" / execution_id
    summary_path = run_root / "summary/summary.json"
    if not summary_path.is_file():
        return None
    manifest_path = run_root / "execution_manifest.json"
    receipt_path = RUNTIME_DIR / "worker_receipts" / f"{execution_id}.json"
    payload_paths = {
        "checkpoint": run_root / "checkpoints/current.json",
        "result": run_root / "result/result.json",
        "estimator_ledger": run_root / "result/estimator_ledger.json",
        "summary": summary_path,
    }
    required_paths = (manifest_path, receipt_path, *payload_paths.values())
    if any(not path.is_file() or path.is_symlink() for path in required_paths):
        return None
    manifest = load(manifest_path)
    receipt = load(receipt_path)
    verify_self_digest(manifest, label=f"{execution_id} execution manifest")
    verify_self_digest(receipt, label=f"{execution_id} worker receipt")
    if (
        manifest.get("status") != "passed"
        or manifest.get("execution_id") != execution_id
        or manifest.get("controller_rounds_completed") != 50
        or manifest.get("target_horizon") != 50
        or receipt.get("status") != "passed"
        or receipt.get("execution_id") != execution_id
        or receipt.get("controller_rounds_completed") != 50
    ):
        raise UpdateError(f"completed closure drifted: {execution_id}")
    raw_payloads = manifest.get("output_payloads")
    if not isinstance(raw_payloads, Mapping) or set(raw_payloads) != set(
        payload_paths
    ):
        raise UpdateError(f"completed output inventory drifted: {execution_id}")
    payload_sources: dict[str, Any] = {}
    for role, path in payload_paths.items():
        declared = raw_payloads.get(role)
        if not isinstance(declared, Mapping):
            raise UpdateError(f"completed {role} binding is absent: {execution_id}")
        declared_path = RUNTIME_DIR / str(declared.get("path"))
        if (
            declared_path.resolve() != path.resolve()
            or int(declared.get("size_bytes", -1)) != path.stat().st_size
        ):
            raise UpdateError(f"completed {role} path/size drifted: {execution_id}")
        payload_sources[role] = {
            "path": str(path.resolve()),
            "sha256": str(declared.get("sha256")),
            "size_bytes": path.stat().st_size,
            "binding_source": "passed_self_digested_execution_manifest",
        }
    summary = load(summary_path)
    if sha256(summary_path) != raw_payloads["summary"].get("sha256"):
        raise UpdateError(f"completed summary bytes drifted: {execution_id}")
    trace = summary.get("accepted_error_trace")
    if (
        summary.get("schema") != "paper_i_run_summary_v1"
        or not isinstance(trace, list)
        or [row.get("controller_round") for row in trace] != list(range(1, 51))
    ):
        raise UpdateError(f"completed summary trajectory drifted: {execution_id}")
    points = []
    for row in trace:
        if not math.isclose(
            float(row["exact_same_cutoff_energy"]), exact, rel_tol=0.0, abs_tol=1e-12
        ):
            raise UpdateError(f"completed exact reference drifted: {execution_id}")
        points.append(
            {
                "k": int(row["controller_round"]),
                "energy": float(row["accepted_energy"]),
                "error": float(row["absolute_energy_error"]),
            }
        )
    work = summary.get("canonical_all_work")
    if not isinstance(work, Mapping) or work.get("s_alg") is None:
        raise UpdateError(f"completed S_alg is unavailable: {execution_id}")
    requested = [
        row
        for row in summary.get("requested_rounds", [])
        if row.get("controller_round") == 50
    ]
    costs: dict[str, Any] = {
        "N2q": None,
        "D2q": None,
        "Dc": None,
        "W1q": None,
        "S_alg": int(work["s_alg"]),
    }
    cost_status = "terminal_qiskit_unavailable"
    if len(requested) == 1 and requested[0].get("status") == "available":
        resources = requested[0].get("resources")
        if isinstance(resources, Mapping):
            costs.update(
                {
                    "N2q": int(resources["compiled_two_qubit_count"]),
                    "D2q": int(resources["compiled_two_qubit_depth"]),
                    "Dc": int(resources["compiled_total_depth"]),
                }
            )
            cost_status = "available_except_W1q_not_serialized_by_summary"
    return {
        "status": "completed_authenticated_local_summary",
        "points": points,
        "latest": copy.deepcopy(points[-1]),
        "costs": costs,
        "cost_status": cost_status,
        "sources": {
            "execution_manifest": binding(manifest_path),
            "worker_receipt": binding(receipt_path),
            "output_payloads": payload_sources,
            "large_payload_verification_policy": (
                "path_and_size_match_passed_manifest; manifest supplies sha256; "
                "summary bytes rehashed for this diagnostic refresh"
            ),
        },
    }


def _live_route(
    execution_id: str, *, exact: float, serial_status: Mapping[str, Any]
) -> dict[str, Any] | None:
    running_ids = serial_status.get("running_execution_ids")
    active = serial_status.get("current_execution_id") == execution_id
    if isinstance(running_ids, list):
        active = active or execution_id in running_ids
    if serial_status.get("status") != "running" or not active:
        return None
    log_path = RUNTIME_DIR / "logs" / f"{execution_id}.out"
    candidates = sorted(
        RUNTIME_DIR.glob(
            f"in_progress/*{execution_id}*/cell_output/checkpoints/"
            "current.verified_singleton_resume.*.json"
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not log_path.is_file() or not candidates:
        return None
    sidecar_path = candidates[0]
    sidecar = load(sidecar_path)
    state = sidecar.get("controller_state")
    source_result = sidecar.get("source_result_json")
    if not isinstance(state, Mapping) or not isinstance(source_result, str):
        raise UpdateError("live checkpoint sidecar is incomplete")
    k = int(state["controller_round"])
    current_path = Path(source_result)
    try:
        current_path.resolve().relative_to((RUNTIME_DIR / "in_progress").resolve())
    except ValueError as exc:
        raise UpdateError("live checkpoint escapes the local runtime") from exc
    snapshot_path = REPORT_DIR / f".page13-live-{os.getpid()}-{uuid.uuid4().hex}.json"
    try:
        os.link(current_path, snapshot_path)
        observed_sha = sha256(snapshot_path)
        extracted = subprocess.run(
            [
                "jq",
                "-c",
                "{history_count:.adapt_vqe.history_count,"
                "energy:.adapt_vqe.energy,S_alg:.adapt_vqe.S_alg}",
                str(snapshot_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        checkpoint = json.loads(extracted.stdout)
        snapshot_size = snapshot_path.stat().st_size
    finally:
        snapshot_path.unlink(missing_ok=True)
    if int(checkpoint["history_count"]) != k:
        return None
    log_bytes = log_path.read_bytes()
    log_text = log_bytes.decode("utf-8")
    points = parse_ai_log_points(log_text, exact=exact, maximum_k=k)
    if not points or points[-1]["k"] != k:
        return None
    # The iteration log is emitted before the final checkpoint publication
    # seam and can differ by a tiny terminal refit.  Preserve its prefix but
    # use the frozen checkpoint's accepted energy for the latest point.
    points[-1]["logged_energy_before_checkpoint_publication"] = points[-1][
        "energy"
    ]
    points[-1]["energy"] = float(checkpoint["energy"])
    points[-1]["error"] = abs(float(checkpoint["energy"]) - exact)
    return {
        "status": "live_snapshot_incomplete",
        "points": points,
        "latest": copy.deepcopy(points[-1]),
        "costs": {
            "N2q": None,
            "D2q": None,
            "Dc": None,
            "W1q": None,
            "S_alg": int(checkpoint["S_alg"]),
        },
        "cost_status": "qiskit_unavailable_until_completed_summary",
        "sources": {
            "verified_resume_sidecar": binding(sidecar_path),
            "source_result": {
                "path": str(current_path.resolve()),
                "sha256": observed_sha,
                "size_bytes": snapshot_size,
                "verified_resume_projection_sha256": sidecar.get(
                    "source_result_sha256"
                ),
                "verified_resume_projection_scope": sidecar.get(
                    "source_result_digest_scope"
                ),
            },
            "accepted_log_prefix": {
                "path": str(log_path.resolve()),
                "sha256": hashlib.sha256(log_bytes).hexdigest(),
                "size_bytes": len(log_bytes),
                "used_through_controller_round": k,
            },
        },
    }


def build_adapter(provenance: Mapping[str, Any]) -> dict[str, Any]:
    package_manifest = load(PACKAGE_DIR / "package_manifest.json")
    source_archive = package_manifest.get("source_archive")
    if (
        package_manifest.get("sha256") != PACKAGE_MANIFEST_SHA256
        or not isinstance(source_archive, Mapping)
        or source_archive.get("sha256") != SOURCE_ARCHIVE_SHA256
        or package_manifest.get("child_route_contract_sha256")
        != ROUTE_CONTRACT_SHA256
        or package_manifest.get("submitted") is not False
    ):
        raise UpdateError("sealed local source-package identity drifted")
    cache = load(CURVE_CACHE)
    if cache.get("schema") != "paper_i_stationary_page1_macro_curve_cache_v1":
        raise UpdateError("Page-1 macro curve cache identity drifted")
    exact = exact_references()
    blue = _page1_blue_sources(provenance, cache)
    jobs = _jobs()
    serial_status = load(RUNTIME_DIR / "serial_status.json")
    cells = []
    for regime in REGIME_ORDER:
        job_path, job = jobs[regime]
        execution_id = str(job["execution_id"])
        route = _completed_route(execution_id, exact=exact[regime])
        if route is None:
            route = _live_route(
                execution_id,
                exact=exact[regime],
                serial_status=serial_status,
            )
        status = "pending_serial_not_started" if route is None else route["status"]
        cells.append(
            {
                "regime_id": regime,
                "regime_label": REGIME_LABELS[regime],
                "nph": NPH[regime],
                "exact_same_cutoff_energy": exact[regime],
                "conventional_unwhitened_adapt": blue[regime],
                "macro_phase0_route": route,
                "status": status,
                "job": binding(job_path),
            }
        )
    unsigned = {
        "schema": "paper_i_macro_phase0_proxy_no_lanes_page13_adapter_v1",
        "page_id": PAGE_ID,
        "status": "partial_progress_diagnostic",
        "paper_evidence_adopted": False,
        "route": {
            "candidate_representation": "macro_generator_v1",
            "pool_size": 102,
            "phase0": "standard_absolute_energy_gradient",
            "phase0_cap": 24,
            "phase1_cap": 24,
            "phase2_cap": 12,
            "phase3_cap": 12,
            "phase123_cost_source": "measurement_proxy",
            "qiskit_selector_cost": False,
            "lane_shortlisting": False,
            "gradient_policy": "stationary_source_response_v1",
            "insertion": "commutation_reduced_relative_plateau",
            "relative_plateau_threshold": 1.0e-4,
            "plateau_patience": 1,
            "optimizer": "powell",
            "optimizer_maxiter": 200,
            "seed": 7,
            "target_horizon": 50,
        },
        "source_package": {
            "package_manifest": binding(PACKAGE_DIR / "package_manifest.json"),
            "canonical_sha256": PACKAGE_MANIFEST_SHA256,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "route_contract_sha256": ROUTE_CONTRACT_SHA256,
            "submitted": False,
        },
        "local_execution": {
            "activation_manifest": binding(ACTIVATION_DIR / "activation_manifest.json"),
            "serial_manifest": binding(RUNTIME_DIR / "serial_manifest.json"),
            "serial_status": binding(RUNTIME_DIR / "serial_status.json"),
            "execution_target": "local_mac_serial",
        },
        "page1_blue_curve_cache": binding(CURVE_CACHE),
        "same_cutoff_reference": binding(ED_SOURCE),
        "cells": cells,
        "limitations": [
            "live checkpoints are incomplete diagnostic prefixes, not completed evidence",
            "pending serial cells have no local RA trajectory",
            "the Page-1 blue cache intentionally stores k=1..50; Page 13 therefore begins both displayed families at k=1",
            "the live RA Qiskit tuple is unavailable until the completed round-50 summary is published",
            "the completed local summary does not serialize W1q; any such tuple remains explicitly partial until a common compiler adapter supplies it",
        ],
    }
    unsigned["sha256"] = _canonical_sha256(unsigned)
    return unsigned


def format_error(value: float) -> str:
    return f"{value:.2e}"


def format_s_alg(value: int) -> str:
    mantissa, exponent = f"{int(value):.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def format_cost_tuple(value: Mapping[str, Any] | None) -> str:
    if not isinstance(value, Mapping):
        return "--"
    values = []
    for field in ("N2q", "D2q", "Dc", "W1q", "S_alg"):
        item = value.get(field)
        if item is None:
            values.append("--")
        elif field == "S_alg":
            values.append(format_s_alg(int(item)))
        else:
            values.append(str(int(item)))
    return "(" + ",".join(values) + ")"


def format_route_status(value: str) -> str:
    if value.startswith("completed_"):
        return "complete"
    if value == "live_snapshot_incomplete":
        return "live prefix"
    if value == "pending_serial_not_started":
        return "pending"
    return value.replace("_", " ")


def render_page(adapter: Mapping[str, Any]) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from PIL import Image

    _atomic_json(ADAPTER_PATH, adapter)
    mpl.rcParams.update({"font.family": "serif", "font.size": 7.4})
    fig = plt.figure(figsize=(11, 8.5))
    grid = fig.add_gridspec(
        3,
        3,
        height_ratios=(1.0, 1.0, 0.64),
        hspace=0.34,
        wspace=0.25,
    )
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]
    for index, (axis, cell) in enumerate(zip(axes, adapter["cells"], strict=True)):
        baseline = cell["conventional_unwhitened_adapt"]
        axis.plot(
            [row["k"] for row in baseline["points"]],
            [max(float(row["error"]), PLOT_FLOOR) for row in baseline["points"]],
            color=BLUE,
            lw=1.55,
        )
        axis.scatter(
            [baseline["terminal"]["k"]],
            [max(float(baseline["terminal"]["error"]), PLOT_FLOOR)],
            color=BLUE,
            marker="o",
            s=28,
            zorder=4,
        )
        route = cell["macro_phase0_route"]
        if route:
            points = route["points"]
            axis.plot(
                [row["k"] for row in points],
                [max(float(row["error"]), PLOT_FLOOR) for row in points],
                color=ROUTE_COLOR,
                lw=1.8,
            )
            axis.scatter(
                [route["latest"]["k"]],
                [max(float(route["latest"]["error"]), PLOT_FLOOR)],
                color=ROUTE_COLOR,
                marker="s" if route["status"].startswith("completed") else "o",
                s=34,
                zorder=4,
            )
            status = "complete" if route["status"].startswith("completed") else "live prefix"
            axis.text(
                0.97,
                0.08,
                f"{status}: k={route['latest']['k']}\n"
                f"|dE|={format_error(float(route['latest']['error']))}",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.4,
                color=ROUTE_COLOR,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            )
        else:
            axis.text(
                0.5,
                0.13,
                "local RA cell pending (serial)",
                transform=axis.transAxes,
                ha="center",
                fontsize=6.8,
                color=ROUTE_COLOR,
                bbox={"facecolor": "white", "edgecolor": ROUTE_COLOR, "alpha": 0.82},
            )
        axis.set_yscale("log")
        axis.set_xlim(0, 50)
        axis.grid(True, which="major", alpha=0.22, lw=0.5)
        axis.set_title(
            f"{cell['regime_label']} ($n_{{ph}}={cell['nph']}$)", fontsize=8.4
        )
        if index // 3 == 1:
            axis.set_xlabel("ADAPT controller round")
        if index % 3 == 0:
            axis.set_ylabel(r"same-cutoff $|\Delta E|$")
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=BLUE,
                lw=1.55,
                marker="o",
                markersize=4.5,
                label="Conventional unwhitened ADAPT (Page 1 blue)",
            ),
            Line2D(
                [0],
                [0],
                color=ROUTE_COLOR,
                lw=1.8,
                marker="o",
                markersize=4.5,
                label="RA: macro Phase 0 -> macro Phase I/II/III (proxy, no lanes)",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.953),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Macro-only gradient Phase 0, then macro Phase I/II/III "
        "(proxy costs, no lanes)",
        fontsize=10.8,
        fontweight="bold",
        y=0.988,
    )

    table_axis = fig.add_subplot(grid[2, :])
    table_axis.axis("off")
    rows = []
    for cell in adapter["cells"]:
        route = cell["macro_phase0_route"]
        baseline = cell["conventional_unwhitened_adapt"]
        rows.append(
            [
                cell["regime_label"],
                format_route_status(str(cell["status"])),
                "--" if not route else str(route["latest"]["k"]),
                "--" if not route else format_error(float(route["latest"]["error"])),
                format_error(float(baseline["terminal"]["error"])),
                "--" if not route else format_cost_tuple(route.get("costs")),
                format_cost_tuple(baseline["terminal"]),
            ]
        )
    table = table_axis.table(
        cellText=rows,
        colLabels=[
            "Regime",
            "Local RA status",
            "latest k",
            r"RA $|\Delta E_k|$",
            r"ADAPT $|\Delta E_{50}|$",
            r"RA Qiskit / $S_{alg}$",
            r"ADAPT Qiskit / $S_{alg}$",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=(0.14, 0.10, 0.06, 0.12, 0.12, 0.23, 0.23),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.8)
    table.scale(1.0, 0.84)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EAEAEA")
    fig.text(
        0.5,
        0.018,
        r"Tuple order: $(N_{2q},D_{2q},D_c,W_{1q},S_{alg})$. "
        r"-- means not yet serialized; live prefixes are diagnostic only.",
        ha="center",
        fontsize=6.4,
    )
    fig.savefig(PAGE_PNG, dpi=240, bbox_inches="tight")
    plt.close(fig)
    with Image.open(PAGE_PNG) as source:
        source.convert("RGB").save(PAGE_PDF, format="PDF", resolution=240.0)


def _page_content_sha256(page: Any) -> str:
    contents = page.get_contents()
    data = b"" if contents is None else contents.get_data()
    return hashlib.sha256(data).hexdigest()


def append_or_replace_page(
    adapter: Mapping[str, Any], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    from pypdf import PdfReader, PdfWriter

    current = binding(TARGET_PDF)
    declared = provenance["outputs"]["partial_progress_pdf"]
    page_count = int(provenance["layout"].get("page_count", -1))
    if (
        current["sha256"] != declared["sha256"]
        or current["size_bytes"] != declared["size_bytes"]
        or page_count not in (12, 13)
        or provenance["layout"].get("page_11")
        != "macro_gradient_phase0_then_singleton_partial_v1"
        or provenance["layout"].get("page_12")
        != "global_singleton_gradient_phase0_partial_v1"
        or (page_count == 13 and provenance["layout"].get("page_13") != PAGE_ID)
    ):
        raise UpdateError("target PDF/provenance is not a supported Page-13 state")
    original = PdfReader(str(TARGET_PDF), strict=False)
    if len(original.pages) != page_count:
        raise UpdateError("target PDF page count drifted")
    prior_digests = [_page_content_sha256(page) for page in original.pages[:12]]
    page_reader = PdfReader(str(PAGE_PDF), strict=False)
    if len(page_reader.pages) != 1:
        raise UpdateError("Page-13 artifact must have exactly one page")
    writer = PdfWriter()
    for page in original.pages[:12]:
        writer.add_page(page)
    writer.add_page(page_reader.pages[0])

    temporary_pdf = TARGET_PDF.with_name(f".{TARGET_PDF.name}.page13.tmp")
    temporary_provenance = TARGET_PROVENANCE.with_name(
        f".{TARGET_PROVENANCE.name}.page13.tmp"
    )
    rollback = TARGET_PDF.with_name(f".{TARGET_PDF.name}.page13.rollback")
    for path in (temporary_pdf, temporary_provenance, rollback):
        if path.exists() or path.is_symlink():
            raise UpdateError(f"stale temporary exists: {path}")
    try:
        with temporary_pdf.open("xb") as stream:
            writer.write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        combined_reader = PdfReader(str(temporary_pdf), strict=False)
        if len(combined_reader.pages) != 13:
            raise UpdateError("combined PDF must have 13 pages")
        if [
            _page_content_sha256(page) for page in combined_reader.pages[:12]
        ] != prior_digests:
            raise UpdateError("Pages 1--12 changed while adding Page 13")

        updated = copy.deepcopy(dict(provenance))
        updated["layout"]["page_13"] = PAGE_ID
        updated["layout"]["page_count"] = 13
        updated["macro_phase0_macro_only_proxy_no_lanes_progress"] = {
            "schema": "paper_i_macro_phase0_proxy_no_lanes_page13_report_v1",
            "page_id": PAGE_ID,
            "status": adapter["status"],
            "paper_evidence_adopted": False,
            "adapter": {**binding(ADAPTER_PATH), "canonical_sha256": adapter["sha256"]},
            "cells": copy.deepcopy(adapter["cells"]),
            "limitations": copy.deepcopy(adapter["limitations"]),
            "outputs": {
                "page_pdf": binding(PAGE_PDF),
                "page_png": binding(PAGE_PNG),
            },
        }
        updated["outputs"]["macro_phase0_proxy_no_lanes_page13_pdf"] = binding(
            PAGE_PDF
        )
        updated["outputs"]["macro_phase0_proxy_no_lanes_page13_png"] = binding(
            PAGE_PNG
        )
        updated["outputs"]["macro_phase0_proxy_no_lanes_page13_adapter"] = {
            **binding(ADAPTER_PATH),
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
        "status": "updated_existing_report_in_place",
        "page_count": 13,
        "pdf": binding(TARGET_PDF),
    }


def main() -> int:
    provenance = load(TARGET_PROVENANCE)
    adapter = build_adapter(provenance)
    render_page(adapter)
    result = append_or_replace_page(adapter, provenance)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
