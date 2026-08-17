#!/usr/bin/env python3
"""Append or refresh the authenticated Page-16 macro Qiskit comparison.

The updater preserves Pages 1--15 at the PDF content-stream level.  A Page-16
cell is plotted only after its CHTC archive closes against the sealed job,
worker receipt, execution manifest, and every receipt-bound payload.  Missing
cells remain visibly pending.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import uuid
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting import (
    append_paper_i_completed_beam_noise_pages as completed_pages,
)
from pipelines.reporting import (
    append_paper_i_macro_phase0_proxy_no_lanes_page13 as page13,
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
PAGE_STEM = (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "macro_phase0_phase23_qiskit_no_lanes_page16"
)
PAGE_PDF = REPORT_DIR / f"{PAGE_STEM}.pdf"
PAGE_PNG = REPORT_DIR / f"{PAGE_STEM}.png"
ADAPTER_PATH = REPORT_DIR / f"{PAGE_STEM}_adapter.json"
PAGE_ID = "macro_gradient_phase0_macro_phase123_qiskit_phase23_no_lanes_partial_v1"

PAGE13_ADAPTER = REPORT_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "macro_phase0_proxy_no_lanes_page13_adapter.json"
)
PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_page16_macro_gradient_phase0_macro_phase123_qiskit_"
    "phase23_no_lanes_cap24_tau1em4_weak50_strong30_20260811_v1_chtc"
)
RETRIEVED_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "retrieved_chtc_20260811_page16_macro_phase23_qiskit_v1"
)

PACKAGE_MANIFEST_SHA256 = (
    "ef1b626932f95af6408b65e2ef69d477fe5e21091ca0cb66cee40881ab3109d2"
)
SOURCE_ARCHIVE_SHA256 = (
    "95b3ea575a4590961b6a57337eb1c58ef3ba3855d9d342b179657973c129ef26"
)
ROUTE_CONTRACT_SHA256 = (
    "a97b5dce0fcb5e2b53a69fd404eb3e87f595494c269923958799af793b99b6e0"
)
CLUSTER_ID = 9636624

REGIME_ORDER = page13.REGIME_ORDER
REGIME_LABELS = page13.REGIME_LABELS
NPH = page13.NPH
TARGET_HORIZON = {
    "weak_weak": 50,
    "intermediate_weak": 50,
    "strong_weak_u8": 50,
    "weak_strong": 30,
    "intermediate_strong": 30,
    "strong_strong_u8": 30,
}

WEAK_WEAK_EXECUTION_ID = (
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__weak_weak__nph3__"
    "ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_no_lanes_"
    "plateau"
)
INTERMEDIATE_WEAK_EXECUTION_ID = (
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__"
    "intermediate_weak__nph3__ra_page16_macro_gradient_phase0_macro_phase123_"
    "qiskit_phase23_no_lanes_plateau"
)
STRONG_WEAK_EXECUTION_ID = (
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__"
    "strong_weak_u8__nph3__ra_page16_macro_gradient_phase0_macro_phase123_"
    "qiskit_phase23_no_lanes_plateau"
)
WEAK_STRONG_EXECUTION_ID = (
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__"
    "weak_strong__nph7__ra_page16_macro_gradient_phase0_macro_phase123_"
    "qiskit_phase23_no_lanes_plateau"
)
INTERMEDIATE_STRONG_EXECUTION_ID = (
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__"
    "intermediate_strong__nph7__ra_page16_macro_gradient_phase0_macro_"
    "phase123_qiskit_phase23_no_lanes_plateau"
)
STRONG_STRONG_EXECUTION_ID = (
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__"
    "strong_strong_u8__nph7__ra_page16_macro_gradient_phase0_macro_phase123_"
    "qiskit_phase23_no_lanes_plateau"
)
ARCHIVES: dict[str, dict[str, Any]] = {
    "weak_weak": {
        "proc_id": 0,
        "filename": f"{WEAK_WEAK_EXECUTION_ID}__{CLUSTER_ID}__0.tar.gz",
        "remote_path": (
            "/home/jsstrobel/Holstein_phase3_optuna_chtc/transfer/"
            f"{WEAK_WEAK_EXECUTION_ID}__{CLUSTER_ID}__0.tar.gz"
        ),
        "size_bytes": 370_246_859,
        "sha256": (
            "00cdac5c57d976956aa8ca527635c4ca9c4d77736abc29dbdb4620e8fc8125e1"
        ),
    },
    "intermediate_weak": {
        "proc_id": 1,
        "filename": (
            f"{INTERMEDIATE_WEAK_EXECUTION_ID}__{CLUSTER_ID}__1.tar.gz"
        ),
        "remote_path": (
            "/home/jsstrobel/Holstein_phase3_optuna_chtc/transfer/"
            f"{INTERMEDIATE_WEAK_EXECUTION_ID}__{CLUSTER_ID}__1.tar.gz"
        ),
        "size_bytes": 407_785_859,
        "sha256": (
            "1e34eb0009ee75afaa36db33f742b082f507be57313fc3e98e65fee1b57e8b0b"
        ),
    },
    "strong_weak_u8": {
        "proc_id": 2,
        "filename": f"{STRONG_WEAK_EXECUTION_ID}__{CLUSTER_ID}__2.tar.gz",
        "remote_path": (
            "/home/jsstrobel/Holstein_phase3_optuna_chtc/transfer/"
            f"{STRONG_WEAK_EXECUTION_ID}__{CLUSTER_ID}__2.tar.gz"
        ),
        "size_bytes": 285_830_975,
        "sha256": (
            "7ed65d7c4a43e38122a3f7b288851ee57f7ccb12ba79347f38cccd6b182b6ba0"
        ),
    },
    "weak_strong": {
        "proc_id": 3,
        "filename": f"{WEAK_STRONG_EXECUTION_ID}__{CLUSTER_ID}__3.tar.gz",
        "remote_path": (
            "/home/jsstrobel/Holstein_phase3_optuna_chtc/transfer/"
            f"{WEAK_STRONG_EXECUTION_ID}__{CLUSTER_ID}__3.tar.gz"
        ),
        "size_bytes": 101_154_867,
        "sha256": (
            "0f21a5a752c6d28b693b242fa20ff84e6776d244f9c5c3245255d93f68c2cd5e"
        ),
    },
    "intermediate_strong": {
        "proc_id": 4,
        "filename": (
            f"{INTERMEDIATE_STRONG_EXECUTION_ID}__{CLUSTER_ID}__4.tar.gz"
        ),
        "remote_path": (
            "/home/jsstrobel/Holstein_phase3_optuna_chtc/transfer/"
            f"{INTERMEDIATE_STRONG_EXECUTION_ID}__{CLUSTER_ID}__4.tar.gz"
        ),
        "size_bytes": 105_627_736,
        "sha256": (
            "da095c3dc0a57b4f27fc3d39d76f4d45e2c55370ae3db79c35a3b822886436f8"
        ),
    },
    "strong_strong_u8": {
        "proc_id": 5,
        "filename": (
            f"{STRONG_STRONG_EXECUTION_ID}__{CLUSTER_ID}__5.tar.gz"
        ),
        "remote_path": (
            "/home/jsstrobel/Holstein_phase3_optuna_chtc/transfer/"
            f"{STRONG_STRONG_EXECUTION_ID}__{CLUSTER_ID}__5.tar.gz"
        ),
        "size_bytes": 94_818_319,
        "sha256": (
            "d379ec9816fb4689394a26e3eedb9830104ae910cccc13d7fecf464a49949b0a"
        ),
    },
}

PLOT_FLOOR = 1.0e-16
BLUE = "#4C78A8"
GREEN = "#009E73"
ORANGE = "#E69F00"


class UpdateError(ValueError):
    pass


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise UpdateError(f"JSON object required: {path}")
    return value


def binding(path: Path) -> dict[str, Any]:
    return completed_pages.binding(path)


def verify_self_digest(value: Mapping[str, Any], *, label: str) -> None:
    claimed = value.get("sha256")
    unsigned = {key: row for key, row in value.items() if key != "sha256"}
    if claimed != _canonical_sha256(unsigned):
        raise UpdateError(f"{label}: self digest drifted")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(
                json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _safe_member_name(name: str) -> str:
    raw = name.removeprefix("./")
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise UpdateError(f"unsafe archive member: {name}")
    return path.as_posix()


def _sha256_stream(stream: Any) -> tuple[str, int, bytes | None]:
    digest = hashlib.sha256()
    size = 0
    captured: bytes | None = b""
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
        size += len(block)
        if captured is not None:
            if size <= 4 * 1024 * 1024:
                captured += block
            else:
                captured = None
    return digest.hexdigest(), size, captured


def _close_page16_archive(
    *,
    path: Path,
    expected: Mapping[str, Any],
    job_path: Path,
    job: Mapping[str, Any],
    cluster_id: int = CLUSTER_ID,
    expected_route_contract_sha256: str = ROUTE_CONTRACT_SHA256,
) -> dict[str, Any]:
    """Validate every archive file before delegating summary reconstruction."""

    archive_binding = binding(path)
    if (
        archive_binding["sha256"] != expected["sha256"]
        or archive_binding["size_bytes"] != expected["size_bytes"]
    ):
        raise UpdateError(f"archive identity drifted: {path}")

    observed: dict[str, dict[str, Any]] = {}
    directories: set[str] = set()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            relative = _safe_member_name(member.name)
            if relative in observed or relative in directories:
                raise UpdateError(f"duplicate archive member: {relative}")
            if member.issym() or member.islnk():
                raise UpdateError(f"linked archive member is forbidden: {relative}")
            if member.isdir():
                directories.add(relative)
                continue
            if not member.isfile():
                raise UpdateError(f"unsafe archive member type: {relative}")
            stream = archive.extractfile(member)
            if stream is None:
                raise UpdateError(f"unreadable archive member: {relative}")
            digest, size, captured = _sha256_stream(stream)
            observed[relative] = {
                "sha256": digest,
                "size_bytes": size,
                "captured": captured,
            }

    required_roots = {"worker_exit_status.txt", "worker_receipt.json"}
    if not required_roots.issubset(observed):
        raise UpdateError("Page-16 archive lacks worker root receipts")
    exit_raw = observed["worker_exit_status.txt"]["captured"]
    if exit_raw is None or exit_raw.strip() != b"0":
        raise UpdateError("Page-16 worker exit status is nonzero or unreadable")
    worker_raw = observed["worker_receipt.json"]["captured"]
    if worker_raw is None:
        raise UpdateError("Page-16 worker receipt is unexpectedly large")
    worker = json.loads(worker_raw)
    if not isinstance(worker, dict):
        raise UpdateError("Page-16 worker receipt must be a JSON object")
    verify_self_digest(worker, label="Page-16 worker receipt")
    execution_id = str(job["execution_id"])
    target_horizon = int(job["target_horizon"])
    if (
        worker.get("schema")
        != "paper_i_ra_adapt_page16_macro_phase23_qiskit_worker_receipt_v1"
        or worker.get("status") != "passed"
        or worker.get("package_id") != job["package_id"]
        or worker.get("campaign_id") != job["campaign_id"]
        or worker.get("execution_id") != execution_id
        or worker.get("job_spec_sha256") != job["sha256"]
        or worker.get("controller_rounds_completed") != target_horizon
    ):
        raise UpdateError(f"Page-16 worker identity drifted: {execution_id}")
    raw_artifacts = worker.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise UpdateError("Page-16 worker artifact inventory is absent")
    declared: dict[str, Mapping[str, Any]] = {}
    for row in raw_artifacts:
        if not isinstance(row, Mapping):
            raise UpdateError("Page-16 worker artifact row is invalid")
        relative = _safe_member_name(str(row.get("path", "")))
        if relative in declared:
            raise UpdateError(f"duplicate worker artifact: {relative}")
        declared[relative] = row
    if set(observed) != required_roots | set(declared):
        raise UpdateError("Page-16 archive contains missing or unbound files")
    for relative, row in declared.items():
        actual = observed[relative]
        if (
            actual["sha256"] != row.get("sha256")
            or actual["size_bytes"] != row.get("size_bytes")
        ):
            raise UpdateError(f"Page-16 artifact binding drifted: {relative}")

    manifest_name = f"runs/{execution_id}/execution_manifest.json"
    manifest_raw = observed.get(manifest_name, {}).get("captured")
    if manifest_raw is None:
        raise UpdateError("Page-16 execution manifest is absent or too large")
    manifest = json.loads(manifest_raw)
    if not isinstance(manifest, dict):
        raise UpdateError("Page-16 execution manifest must be a JSON object")
    verify_self_digest(manifest, label="Page-16 execution manifest")
    if (
        manifest.get("schema")
        != "paper_i_ra_adapt_page16_macro_phase23_qiskit_execution_manifest_v1"
        or manifest.get("status") != "passed"
        or manifest.get("execution_id") != execution_id
        or manifest.get("package_id") != job["package_id"]
        or manifest.get("job_spec_sha256") != job["sha256"]
        or manifest.get("route_contract_sha256")
        != expected_route_contract_sha256
        or manifest.get("controller_rounds_completed") != target_horizon
        or manifest.get("target_horizon") != target_horizon
        or manifest.get("sha256") != worker.get("execution_manifest_sha256")
    ):
        raise UpdateError(f"Page-16 execution closure drifted: {execution_id}")

    result = completed_pages._archive_result(
        path=path,
        expected=expected,
        cluster_id=cluster_id,
        job_path=job_path,
        job=job,
    )
    result["sources"]["retrieval_identity"]["remote_state"] = (
        "preserved_after_exact_size_sha256_verified_fetch"
    )
    result["sources"]["archive_closure"] = {
        "worker_exit_status": 0,
        "declared_payload_count": len(declared),
        "all_declared_payload_hashes_verified": True,
        "unbound_file_count": 0,
        "worker_receipt_canonical_sha256": worker["sha256"],
        "execution_manifest_canonical_sha256": manifest["sha256"],
        "authorization_sha256_bound_by_worker": worker["authorization_sha256"],
    }
    return result


def _source_page13() -> dict[str, Any]:
    adapter = load(PAGE13_ADAPTER)
    verify_self_digest(adapter, label="Page-13 adapter")
    cells = adapter.get("cells")
    if (
        adapter.get("page_id")
        != "macro_gradient_phase0_macro_phase123_proxy_no_lanes_partial_v1"
        or not isinstance(cells, list)
        or [row.get("regime_id") for row in cells] != list(REGIME_ORDER)
        or any(
            not isinstance(row.get("macro_phase0_route"), Mapping)
            or row["macro_phase0_route"].get("status")
            != "completed_authenticated_local_summary"
            for row in cells
        )
    ):
        raise UpdateError("Page-13 comparison authority drifted")
    return adapter


def build_adapter() -> dict[str, Any]:
    manifest = load(PACKAGE_DIR / "package_manifest.json")
    verify_self_digest(manifest, label="Page-16 package manifest")
    source_archive = manifest.get("source_archive")
    if (
        manifest.get("sha256") != PACKAGE_MANIFEST_SHA256
        or manifest.get("status") != "passed_inert_six_cells"
        or manifest.get("row_count") != 6
        or manifest.get("submitted") is not False
        or not isinstance(source_archive, Mapping)
        or source_archive.get("sha256") != SOURCE_ARCHIVE_SHA256
        or manifest.get("child_route_contract_sha256") != ROUTE_CONTRACT_SHA256
    ):
        raise UpdateError("Page-16 sealed package identity drifted")
    jobs = completed_pages._package_jobs(PACKAGE_DIR, PACKAGE_MANIFEST_SHA256)
    page13_adapter = _source_page13()
    page13_cells = {row["regime_id"]: row for row in page13_adapter["cells"]}
    cells = []
    for regime in REGIME_ORDER:
        matches = [
            value
            for execution_id, value in jobs.items()
            if f"__{regime}__nph{NPH[regime]}__" in execution_id
        ]
        if len(matches) != 1:
            raise UpdateError(f"Page-16 job coverage drifted: {regime}")
        job_path, job = matches[0]
        if (
            job.get("regime_id") != regime
            or job.get("nph") != NPH[regime]
            or job.get("target_horizon") != TARGET_HORIZON[regime]
            or job.get("candidate_representation") != "macro_generator_v1"
            or job.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        ):
            raise UpdateError(f"Page-16 job identity drifted: {regime}")
        route = None
        archive_spec = ARCHIVES.get(regime)
        if archive_spec is not None:
            route = _close_page16_archive(
                path=RETRIEVED_DIR / str(archive_spec["filename"]),
                expected=archive_spec,
                job_path=job_path,
                job=job,
            )
        page13_cell = page13_cells[regime]
        cells.append(
            {
                "regime_id": regime,
                "regime_label": REGIME_LABELS[regime],
                "nph": NPH[regime],
                "target_horizon": TARGET_HORIZON[regime],
                "conventional_unwhitened_adapt": copy.deepcopy(
                    page13_cell["conventional_unwhitened_adapt"]
                ),
                "page13_proxy_route": copy.deepcopy(
                    page13_cell["macro_phase0_route"]
                ),
                "page16_qiskit_route": route,
                "status": (
                    "completed_authenticated_chtc_archive"
                    if route is not None
                    else "pending_no_completed_archive"
                ),
                "job": binding(job_path),
            }
        )
    completed_count = sum(cell["page16_qiskit_route"] is not None for cell in cells)
    unsigned = {
        "schema": "paper_i_macro_phase0_phase23_qiskit_no_lanes_page16_adapter_v1",
        "page_id": PAGE_ID,
        "status": (
            "completed_6_of_6_mixed_horizon"
            if completed_count == len(REGIME_ORDER)
            else f"partial_{completed_count}_of_6_completed"
        ),
        "paper_evidence_adopted": False,
        "cluster_id": CLUSTER_ID,
        "route": {
            "candidate_representation": "macro_generator_v1",
            "pool_size": 102,
            "phase0": "standard_absolute_energy_gradient",
            "phase0_cap": 24,
            "phase1_cap": 24,
            "phase2_cap": 24,
            "phase3_cap": 24,
            "phase1_cost_source": "measurement_proxy",
            "phase2_cost_source": "signed_qiskit_compiled_marginal",
            "phase3_cost_source": "signed_qiskit_compiled_marginal",
            "lane_shortlisting": False,
            "gradient_policy": "stationary_source_response_v1",
            "insertion": "commutation_reduced_relative_plateau",
            "relative_plateau_threshold": 1.0e-4,
            "plateau_patience": 1,
            "optimizer": "powell",
            "optimizer_maxiter": 200,
            "seed": 7,
            "weak_holstein_horizon": 50,
            "strong_holstein_horizon": 30,
        },
        "source_package": {
            "package_manifest": binding(PACKAGE_DIR / "package_manifest.json"),
            "canonical_sha256": PACKAGE_MANIFEST_SHA256,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "route_contract_sha256": ROUTE_CONTRACT_SHA256,
            "sealed_state": "inert_pre_activation_package",
        },
        "source_page13_adapter": binding(PAGE13_ADAPTER),
        "cells": cells,
        "limitations": [
            "only completed, authenticated CHTC archives are plotted; running and idle cells remain pending",
            "the remote activation manifest and authorization bytes are not copied into this report; their SHA-256 binding is retained in the worker receipt",
            "the sealed source package is inert; CHTC submission authority was supplied through a separate remote activation",
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
    return "(" + ",".join(
        format_s_alg(int(value[field])) if field == "S_alg" else str(int(value[field]))
        for field in ("N2q", "D2q", "Dc", "W1q", "S_alg")
    ) + ")"


def render_page(adapter: Mapping[str, Any]) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _atomic_json(ADAPTER_PATH, adapter)
    mpl.rcParams.update({"font.family": "serif", "font.size": 7.2})
    fig = plt.figure(figsize=(11, 8.5))
    grid = fig.add_gridspec(
        3,
        3,
        height_ratios=(1.0, 1.0, 0.62),
        hspace=0.34,
        wspace=0.25,
    )
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]
    for index, (axis, cell) in enumerate(zip(axes, adapter["cells"], strict=True)):
        horizon = int(cell["target_horizon"])
        for source, color, width in (
            (cell["conventional_unwhitened_adapt"], BLUE, 1.25),
            (cell["page13_proxy_route"], GREEN, 1.5),
        ):
            points = [row for row in source["points"] if int(row["k"]) <= horizon]
            axis.plot(
                [row["k"] for row in points],
                [max(float(row["error"]), PLOT_FLOOR) for row in points],
                color=color,
                lw=width,
            )
        route = cell["page16_qiskit_route"]
        if route is not None:
            points = route["points"]
            axis.plot(
                [row["k"] for row in points],
                [max(float(row["error"]), PLOT_FLOOR) for row in points],
                color=ORANGE,
                lw=1.9,
            )
            terminal = route["terminal"]
            axis.scatter(
                [terminal["k"]],
                [max(float(terminal["error"]), PLOT_FLOOR)],
                color=ORANGE,
                marker="D",
                s=30,
                zorder=5,
            )
            axis.text(
                0.97,
                0.07,
                rf"complete: $k={terminal['k']}$" "\n"
                + rf"$|\Delta E|={format_error(float(terminal['error']))}$",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.2,
                color=ORANGE,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            )
        else:
            axis.text(
                0.5,
                0.12,
                "Page-16 CHTC cell pending",
                transform=axis.transAxes,
                ha="center",
                fontsize=6.7,
                color=ORANGE,
                bbox={"facecolor": "white", "edgecolor": ORANGE, "alpha": 0.82},
            )
        axis.set_yscale("log")
        axis.set_xlim(0, horizon)
        axis.grid(True, which="major", alpha=0.22, lw=0.5)
        axis.set_title(
            f"{cell['regime_label']} ($n_{{ph}}={cell['nph']}$; $k_{{max}}={horizon}$)",
            fontsize=8.2,
        )
        if index // 3 == 1:
            axis.set_xlabel("ADAPT controller round")
        if index % 3 == 0:
            axis.set_ylabel(r"same-cutoff $|\Delta E|$")
    fig.legend(
        handles=[
            Line2D([0], [0], color=BLUE, lw=1.25, label="Conventional unwhitened ADAPT"),
            Line2D([0], [0], color=GREEN, lw=1.5, label="Page 13: proxy Phase I/II/III"),
            Line2D([0], [0], color=ORANGE, lw=1.9, label="Page 16: Qiskit Phase II/III"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.953),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Macro Phase 0 and macro Phase I/II/III: Qiskit costs in Phases II/III",
        fontsize=10.8,
        fontweight="bold",
        y=0.988,
    )

    table_axis = fig.add_subplot(grid[2, :])
    table_axis.axis("off")
    rows = []
    for cell in adapter["cells"]:
        route = cell["page16_qiskit_route"]
        baseline = cell["conventional_unwhitened_adapt"]
        proxy = cell["page13_proxy_route"]
        rows.append(
            [
                cell["regime_label"],
                "complete" if route is not None else "pending",
                "--" if route is None else str(route["terminal"]["k"]),
                "--" if route is None else format_error(float(route["terminal"]["error"])),
                format_error(float(proxy["latest"]["error"])),
                format_error(float(baseline["terminal"]["error"])),
                "--" if route is None else format_cost_tuple(route["costs"]),
            ]
        )
    table = table_axis.table(
        cellText=rows,
        colLabels=[
            "Regime",
            "Page 16",
            "latest k",
            r"Page-16 $|\Delta E_k|$",
            r"Page-13 $|\Delta E_{50}|$",
            r"ADAPT $|\Delta E_{50}|$",
            r"Page-16 Qiskit / $S_{alg}$",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=(0.14, 0.09, 0.07, 0.14, 0.14, 0.14, 0.28),
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
        r"Tuple order: $(N_{2q},D_{2q},D_c,W_{1q},S_{alg})$; "
        r"shared locked Table-I compiler; $S_{alg}$ uses X.YeZ notation.",
        ha="center",
        fontsize=6.4,
    )
    completed_pages._save_page(fig, png_path=PAGE_PNG, pdf_path=PAGE_PDF)
    plt.close(fig)


def _page_content_sha256(page: Any) -> str:
    contents = page.get_contents()
    data = b"" if contents is None else contents.get_data()
    return hashlib.sha256(data).hexdigest()


def append_or_replace_page(
    adapter: Mapping[str, Any], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    from pypdf import PdfReader, PdfWriter

    current = binding(TARGET_PDF)
    outputs = provenance.get("outputs")
    layout = provenance.get("layout")
    declared = outputs.get("partial_progress_pdf") if isinstance(outputs, Mapping) else None
    if not isinstance(layout, Mapping) or not isinstance(declared, Mapping):
        raise UpdateError("report provenance is incomplete")
    page_count = int(layout.get("page_count", -1))
    if (
        current["sha256"] != declared.get("sha256")
        or current["size_bytes"] != declared.get("size_bytes")
        or page_count not in (15, 16)
        or layout.get("page_13")
        != "macro_gradient_phase0_macro_phase123_proxy_no_lanes_partial_v1"
        or layout.get("page_14") != completed_pages.PAGE14_ID
        or layout.get("page_15") != completed_pages.PAGE15_ID
        or (page_count == 16 and layout.get("page_16") != PAGE_ID)
    ):
        raise UpdateError("target PDF/provenance is not a supported Page-16 state")
    original = PdfReader(str(TARGET_PDF), strict=False)
    page = PdfReader(str(PAGE_PDF), strict=False)
    if len(original.pages) != page_count or len(page.pages) != 1:
        raise UpdateError("Page-16 update requires one one-page input")
    preserved_hashes = [_page_content_sha256(row) for row in original.pages[:15]]
    writer = PdfWriter()
    for row in original.pages[:15]:
        writer.add_page(row)
    writer.add_page(page.pages[0])

    token = uuid.uuid4().hex
    temporary_pdf = TARGET_PDF.with_name(f".{TARGET_PDF.name}.{token}.tmp")
    temporary_provenance = TARGET_PROVENANCE.with_name(
        f".{TARGET_PROVENANCE.name}.{token}.tmp"
    )
    rollback_pdf = TARGET_PDF.with_name(f".{TARGET_PDF.name}.{token}.rollback")
    rollback_provenance = TARGET_PROVENANCE.with_name(
        f".{TARGET_PROVENANCE.name}.{token}.rollback"
    )
    try:
        with temporary_pdf.open("xb") as stream:
            writer.write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        combined = PdfReader(str(temporary_pdf), strict=False)
        if len(combined.pages) != 16:
            raise UpdateError("combined report must contain exactly 16 pages")
        if [_page_content_sha256(row) for row in combined.pages[:15]] != preserved_hashes:
            raise UpdateError("Page-16 update changed a preserved page")

        updated = copy.deepcopy(dict(provenance))
        updated["layout"]["page_16"] = PAGE_ID
        updated["layout"]["page_count"] = 16
        updated["macro_phase0_phase23_qiskit_no_lanes_progress"] = {
            "schema": "paper_i_macro_phase0_phase23_qiskit_no_lanes_page16_report_v1",
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
        updated["outputs"]["macro_phase0_phase23_qiskit_page16_pdf"] = binding(
            PAGE_PDF
        )
        updated["outputs"]["macro_phase0_phase23_qiskit_page16_png"] = binding(
            PAGE_PNG
        )
        updated["outputs"]["macro_phase0_phase23_qiskit_page16_adapter"] = {
            **binding(ADAPTER_PATH),
            "canonical_sha256": adapter["sha256"],
        }
        combined_binding = binding(temporary_pdf)
        combined_binding["path"] = str(TARGET_PDF.resolve())
        updated["outputs"]["partial_progress_pdf"] = combined_binding
        with temporary_provenance.open("xb") as stream:
            stream.write(
                json.dumps(updated, indent=2, sort_keys=True, allow_nan=False).encode()
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.link(TARGET_PDF, rollback_pdf)
        os.link(TARGET_PROVENANCE, rollback_provenance)
        os.replace(temporary_pdf, TARGET_PDF)
        try:
            os.replace(temporary_provenance, TARGET_PROVENANCE)
        except BaseException:
            os.replace(rollback_pdf, TARGET_PDF)
            os.replace(rollback_provenance, TARGET_PROVENANCE)
            raise
        rollback_pdf.unlink(missing_ok=True)
        rollback_provenance.unlink(missing_ok=True)
    except BaseException:
        temporary_pdf.unlink(missing_ok=True)
        temporary_provenance.unlink(missing_ok=True)
        rollback_pdf.unlink(missing_ok=True)
        rollback_provenance.unlink(missing_ok=True)
        raise
    return {
        "status": "updated_existing_report_in_place",
        "page_count": 16,
        "completed_cells": sum(
            cell["page16_qiskit_route"] is not None for cell in adapter["cells"]
        ),
        "pdf": binding(TARGET_PDF),
        "provenance": binding(TARGET_PROVENANCE),
    }


def main() -> int:
    provenance = load(TARGET_PROVENANCE)
    adapter = build_adapter()
    render_page(adapter)
    result = append_or_replace_page(adapter, provenance)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
