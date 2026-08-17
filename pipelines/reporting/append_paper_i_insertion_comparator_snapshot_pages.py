#!/usr/bin/env python3
"""Append one provisional Page-16 insertion-comparator progress page.

The update is deliberately diagnostic.  It reuses the authenticated Page-16
plateau adapter, closes each available comparator archive against its sealed
job and worker receipts, authenticates any closed local round-30 cells, and
preserves Pages 1--16 at the PDF content-stream level.  Any older two-page
insertion snapshot is replaced by this single dense 2-by-3 page.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import uuid
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting import (  # noqa: E402
    append_paper_i_completed_beam_noise_pages as completed_pages,
)
from pipelines.reporting import (  # noqa: E402
    append_paper_i_macro_phase23_qiskit_no_lanes_page16 as page16_completed,
)


REPORT_DIR = REPO_ROOT / (
    "output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving"
)
TARGET_PDF = REPORT_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_partial_progress.pdf"
)
TARGET_PROVENANCE = TARGET_PDF.with_name(
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "partial_progress_provenance.json"
)
STEM = (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "insertion_comparator_live_snapshot"
)
PAGE17_PDF = REPORT_DIR / f"{STEM}_page17.pdf"
PAGE17_PNG = REPORT_DIR / f"{STEM}_page17.png"
ADAPTER_PATH = REPORT_DIR / f"{STEM}_adapter.json"
REPORT_MUTATION_LOCK_PATH = REPORT_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "partial_progress_mutation.lock"
)

PAGE17_ID = "phase0_insertion_comparator_page16_six_regime_progress_snapshot_v3"
CURRENT_PAGE18_ID = (
    "phase0_insertion_comparator_page12_six_regime_progress_snapshot_v1"
)
LEGACY_PAGE17_ID = "phase0_insertion_comparator_page12_six_regime_live_snapshot_v2"
LEGACY_PAGE18_ID = "phase0_insertion_comparator_page16_six_regime_live_snapshot_v2"
OLD_PAGE17_ID = "phase0_insertion_comparator_campaign_live_overview_v1"
OLD_PAGE18_ID = "phase0_insertion_comparator_weak_weak_live_trajectory_v1"
PAGE16_ID = "macro_gradient_phase0_macro_phase123_qiskit_phase23_no_lanes_partial_v1"

PAGE16_ADAPTER = REPORT_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "macro_phase0_phase23_qiskit_no_lanes_page16_adapter.json"
)
PACKAGE_ROOT = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
PAGE16_PACKAGE = PACKAGE_ROOT / (
    "paper_i_ra_adapt_page16_insertion_comparators_weak50_strong30_20260812_v1_chtc"
)
CANARY_SUBMISSION_RECEIPT = PACKAGE_ROOT / (
    "paper_i_ra_adapt_page16_insertion_comparators_weak50_strong30_"
    "20260812_v1_chtc_submission_receipt_9644571.json"
)
SUBMISSION_RECEIPT = PACKAGE_ROOT / (
    "paper_i_ra_adapt_insertion_comparators_all24_20260812_"
    "submission_receipt_9644571_9647385_9647386.json"
)
RETRIEVED_DIR = PACKAGE_ROOT / ("retrieved_page16_insertion_comparators_20260812")
SW_ALWAYS_CLOSURE_RECEIPT = PACKAGE_ROOT / (
    "paper_i_ra_adapt_page16_cluster9647386_sw_always_"
    "remote_materialization_exclusion_receipt_20260813.json"
)
SW_ALWAYS_CLOSURE_SCHEMA = (
    "paper_i_ra_adapt_page16_sw_always_"
    "remote_materialization_exclusion_receipt_v2"
)
SW_ALWAYS_CLOSURE_STATUS = (
    "passed_sw_always_k50_closed_remote_materialization_excluded"
)
SW_ALWAYS_EXECUTION_ID = (
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__strong_weak_u8__"
    "nph3__ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_"
    "no_lanes_always_commutation_reduced"
)

COMPARATOR_ROUTE_CONTRACT_SHA256 = (
    "9b9d6bdbb9edb6128e2f0973dd740b44d0daa00d55ecd910fd587f091ae81338"
)
COMPARATOR_ROUTE_CONTRACT_SHA256_BY_POLICY = {
    "always_commutation_reduced": COMPARATOR_ROUTE_CONTRACT_SHA256,
    "append_only": (
        "4e0ec32dafbe7566090d217b690e7bf2a6b7a804b8c1b605e73b0c62a0878c54"
    ),
}
COMPLETED_ARCHIVES: dict[str, dict[str, Any]] = {
    "weak_weak": {
        "cluster_id": 9644571,
        "proc_id": 0,
        "filename": "weak_weak_always__9644571__0.tar.gz",
        "remote_path": (
            "osdf:///chtc/staging/j/jsstrobel/"
            "paper_i_ra_adapt_page16_insertion_comparators_20260812_v1/"
            "outputs/transfer/page16_macro_gradient_phase0_phase23_qiskit_no_lanes__"
            "weak_weak__nph3__ra_page16_macro_gradient_phase0_macro_phase123_"
            "qiskit_phase23_no_lanes_always_commutation_reduced__9644571__0.tar.gz"
        ),
        "size_bytes": 428_656_803,
        "sha256": ("30ee791a285c7f4413e2f69f9e244053c81354707b55fb39f8054a97a00dc0c0"),
    },
    "intermediate_weak": {
        "cluster_id": 9647386,
        "proc_id": 0,
        "filename": "intermediate_weak_always__9647386__0.tar.gz",
        "remote_path": (
            "osdf:///chtc/staging/j/jsstrobel/"
            "paper_i_ra_adapt_page16_insertion_comparators_20260812_v1/"
            "outputs/transfer/page16_macro_gradient_phase0_phase23_qiskit_no_lanes__"
            "intermediate_weak__nph3__ra_page16_macro_gradient_phase0_macro_"
            "phase123_qiskit_phase23_no_lanes_always_commutation_reduced__"
            "9647386__0.tar.gz"
        ),
        "size_bytes": 395_123_818,
        "sha256": ("ff20380198dd907b86308832851fe6a450ece26d75c1fccebd60626507066d08"),
    },
}


def _optional_sw_always_archive(
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return the third archive only after strict remote-exclusion closure."""

    if (
        not SW_ALWAYS_CLOSURE_RECEIPT.exists()
        and not SW_ALWAYS_CLOSURE_RECEIPT.is_symlink()
    ):
        return None, None
    receipt = _load_digested_file(
        SW_ALWAYS_CLOSURE_RECEIPT,
        label="strong-weak always-open CHTC closure and remote exclusion",
    )
    continuation_adapter = _load_continuation_adapter()
    worker = _continuation_call(
        continuation_adapter,
        "strong-weak CHTC strict authentication failed",
        continuation_adapter.k30._load_worker,
    )
    jobs = _continuation_call(
        continuation_adapter,
        "strong-weak CHTC strict authentication failed",
        continuation_adapter._job_by_id,
        worker,
    )
    if (
        continuation_adapter.SW_ALWAYS_CHTC_EXECUTION_ID
        != SW_ALWAYS_EXECUTION_ID
        or not isinstance(jobs, Mapping)
        or SW_ALWAYS_EXECUTION_ID not in jobs
    ):
        raise UpdateError("strong-weak CHTC strict authority drifted")
    strict_terminal = _continuation_call(
        continuation_adapter,
        "strong-weak CHTC strict authentication failed",
        continuation_adapter._authenticate_sw_always_closure,
        worker,
        job=jobs[SW_ALWAYS_EXECUTION_ID],
    )
    strict_archive = (
        strict_terminal.get("archive")
        if isinstance(strict_terminal, Mapping)
        else None
    )
    cell = receipt.get("completed_remote_cell")
    exclusion = receipt.get("remote_materialization_exclusion")
    if not isinstance(cell, Mapping) or not isinstance(exclusion, Mapping):
        raise UpdateError("strong-weak CHTC closure receipt is incomplete")
    archive = cell.get("archive")
    history = cell.get("history")
    worker = cell.get("worker_receipt")
    expected_path = (
        "chtc/paper_i_ra_adapt_repair_20260727/"
        "retrieved_page16_insertion_comparators_20260812/"
        "strong_weak_u8_always__9647386__1.tar.gz"
    )
    if (
        receipt.get("schema") != SW_ALWAYS_CLOSURE_SCHEMA
        or receipt.get("status") != SW_ALWAYS_CLOSURE_STATUS
        or receipt.get("scientific_execution_performed_by_action") is not False
        or cell.get("regime_id") != "strong_weak_u8"
        or cell.get("comparator_policy") != "always_commutation_reduced"
        or cell.get("execution_id") != SW_ALWAYS_EXECUTION_ID
        or cell.get("cluster_id") != 9647386
        or cell.get("proc_id") != 1
        or cell.get("controller_rounds_completed") != 50
        or cell.get("authenticated_full_sealed_closure") is not True
        or not isinstance(archive, Mapping)
        or archive.get("path") != expected_path
        or not isinstance(archive.get("remote_path"), str)
        or "/outputs/transfer/" not in str(archive.get("remote_path"))
        or not isinstance(archive.get("size_bytes"), int)
        or int(archive.get("size_bytes", -1)) <= 0
        or not isinstance(archive.get("sha256"), str)
        or len(str(archive.get("sha256"))) != 64
        or not isinstance(history, Mapping)
        or history.get("exit_code") != 0
        or not isinstance(worker, Mapping)
        or not isinstance(worker.get("canonical_sha256"), str)
        or len(str(worker.get("canonical_sha256"))) != 64
        or exclusion.get("removal_command") != "condor_rm 9647386"
        or exclusion.get("removal_attempts_authenticated") is not True
        or exclusion.get("latent_proc_ids_never_materialized")
        != list(range(2, 11))
        or exclusion.get("queue_cluster_absent") is not True
        or exclusion.get("remote_materialization_excluded") is not True
        or not isinstance(strict_terminal, Mapping)
        or not isinstance(strict_archive, Mapping)
        or strict_terminal.get("execution_id") != SW_ALWAYS_EXECUTION_ID
        or strict_terminal.get("cluster_id") != 9647386
        or strict_terminal.get("proc_id") != 1
        or strict_terminal.get("controller_rounds_completed") != 50
        or strict_terminal.get("source_closure_receipt_sha256")
        != receipt.get("sha256")
        or strict_terminal.get("authenticated_full_sealed_closure") is not True
        or strict_terminal.get("remote_materialization_exclusion_outcome")
        != exclusion.get("outcome")
        or strict_terminal.get(
            "remote_materialization_exclusion_authenticated"
        )
        is not True
        or strict_archive.get("path") != archive.get("path")
        or strict_archive.get("remote_path") != archive.get("remote_path")
        or strict_archive.get("size_bytes") != archive.get("size_bytes")
        or strict_archive.get("sha256") != archive.get("sha256")
    ):
        raise UpdateError("strong-weak CHTC closure receipt drifted")
    archive_path = REPO_ROOT / expected_path
    if (
        not archive_path.is_file()
        or archive_path.is_symlink()
        or archive_path.stat().st_size != archive["size_bytes"]
        or _sha256_file(archive_path) != archive["sha256"]
    ):
        raise UpdateError("strong-weak CHTC archive binding drifted")
    return {
        "cluster_id": 9647386,
        "proc_id": 1,
        "filename": archive_path.name,
        "remote_path": archive["remote_path"],
        "size_bytes": archive["size_bytes"],
        "sha256": archive["sha256"],
    }, receipt

LOCAL_RUNTIME_DIR = REPO_ROOT / (
    "output/local_runs/"
    "paper_i_page16_insertion_comparators_k30_20260812_v2"
)
LOCAL_ACTIVATION_DIR = PACKAGE_ROOT / (
    "paper_i_ra_adapt_page16_insertion_comparators_k30_"
    "20260812_v2_local_activation"
)
LOCAL_ADAPTER_PATH = PACKAGE_ROOT / (
    "run_local_page16_insertion_comparators_20260812.py"
)
EXPECTED_LOCAL_ADAPTER_SHA256 = (
    "bd9d61fb98b48911c3da04faf8b6c38eb391b1a02ab3362e22ef02316a414c4e"
)
LOCAL_EXECUTION_TARGET = "local_mac_two_regime_waves_v2"
LOCAL_TARGET_HORIZON = 30
LOCAL_REQUEST_SCHEMA = (
    "paper_i_page16_insertion_comparator_local_k30_activation_request_v2"
)
LOCAL_PREFLIGHT_SCHEMA = (
    "paper_i_page16_insertion_comparator_local_k30_host_preflight_v2"
)
LOCAL_RUNTIME_SCHEMA = (
    "paper_i_page16_insertion_comparator_local_k30_runtime_manifest_v2"
)
LOCAL_ACTIVATION_SCHEMA = (
    "paper_i_page16_insertion_comparator_local_k30_activation_manifest_v2"
)
LOCAL_AUTHORIZATION_SCHEMA = (
    "paper_i_page16_insertion_comparator_local_k30_authorization_v2"
)
LOCAL_EXECUTION_SCHEMA = (
    "paper_i_page16_insertion_comparator_local_k30_execution_manifest_v2"
)
LOCAL_WORKER_RECEIPT_SCHEMA = (
    "paper_i_page16_insertion_comparator_local_k30_worker_receipt_v2"
)
LOCAL_PLATEAU_GATE_SCHEMA = (
    "paper_i_page16_insertion_comparator_k30_effective_plateau_gate_v2"
)
CONTINUATION_ADAPTER_PATH = PACKAGE_ROOT / (
    "continue_local_page16_insertion_comparators_k30_to_k50_20260813.py"
)
EXPECTED_CONTINUATION_ADAPTER_SHA256 = (
    "56c50f046759d4299d768cb609f08fce8c79e3190aaadb6609afdde4f5452e07"
)
CONTINUATION_SUPERVISOR_PATH = PACKAGE_ROOT / (
    "supervise_local_page16_insertion_comparator_k50_continuations_20260813.py"
)
EXPECTED_CONTINUATION_SUPERVISOR_SHA256 = (
    "0e3a342fa21d925c941a4c3b8e0476c23907ba52d46d459310527a6e0123d761"
)
CONTINUATION_ACTIVATION_DIR = PACKAGE_ROOT / (
    "paper_i_ra_adapt_page16_insertion_comparators_k30_to_k50_"
    "20260813_v2_local_activation"
)
CONTINUATION_RUNTIME_DIR = REPO_ROOT / (
    "output/local_runs/"
    "paper_i_page16_insertion_comparators_k30_to_k50_20260813_v2"
)
MACRO_TERMINAL_RECEIPT = PACKAGE_ROOT / (
    "paper_i_ra_adapt_page16_macro_k30_k50_terminal_clearance_20260813.json"
)
MACRO_TERMINAL_SCHEMA = (
    "paper_i_page16_insertion_comparator_macro_k30_k50_terminal_clearance_v1"
)
MACRO_TERMINAL_STATUS = "passed_all_required_macro_k30_k50_work_terminal"
CONTINUATION_TARGET_HORIZON = 50
_CONTINUATION_ADAPTER: Any | None = None

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
EXPECTED_POLICIES = ("always_commutation_reduced", "append_only")
ORANGE = "#E69F00"
BLUE = "#4C78A8"
RED = "#B22222"
GRAY = "#666666"
MAGENTA = "#CC79A7"
PLOT_FLOOR = 1.0e-16


class UpdateError(ValueError):
    pass


def _load_continuation_adapter() -> Any:
    """Lazily load the exact source-locked continuation authority."""

    global _CONTINUATION_ADAPTER

    continuation_name = "paper_i_page16_reporting_continuation_adapter"
    k30_name = "paper_i_page16_pinned_k30_runner_for_k50_continuation"

    if (
        not CONTINUATION_ADAPTER_PATH.is_file()
        or CONTINUATION_ADAPTER_PATH.is_symlink()
        or _sha256_file(CONTINUATION_ADAPTER_PATH)
        != EXPECTED_CONTINUATION_ADAPTER_SHA256
    ):
        raise UpdateError("pinned continuation adapter is absent or unsafe")
    if _CONTINUATION_ADAPTER is not None:
        existing_path = getattr(_CONTINUATION_ADAPTER, "__file__", None)
        k30_module = getattr(_CONTINUATION_ADAPTER, "k30", None)
        k30_path = getattr(k30_module, "__file__", None)
        if (
            sys.modules.get(continuation_name) is not _CONTINUATION_ADAPTER
            or not isinstance(existing_path, str)
            or Path(existing_path).resolve() != CONTINUATION_ADAPTER_PATH.resolve()
            or sys.modules.get(k30_name) is not k30_module
            or not isinstance(k30_path, str)
            or Path(k30_path).resolve() != LOCAL_ADAPTER_PATH.resolve()
            or not LOCAL_ADAPTER_PATH.is_file()
            or LOCAL_ADAPTER_PATH.is_symlink()
            or _sha256_file(LOCAL_ADAPTER_PATH)
            != EXPECTED_LOCAL_ADAPTER_SHA256
        ):
            raise UpdateError("cached continuation adapter identity drifted")
        return _CONTINUATION_ADAPTER
    if continuation_name in sys.modules:
        raise UpdateError("untrusted continuation adapter is preloaded")
    if k30_name in sys.modules:
        raise UpdateError("untrusted continuation k30 authority is preloaded")
    spec = importlib.util.spec_from_file_location(
        continuation_name, CONTINUATION_ADAPTER_PATH
    )
    if spec is None or spec.loader is None:
        raise UpdateError("pinned continuation adapter cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[continuation_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(continuation_name, None)
        sys.modules.pop(k30_name, None)
        raise UpdateError(f"pinned continuation adapter failed to load: {exc}") from exc
    except BaseException:
        sys.modules.pop(continuation_name, None)
        sys.modules.pop(k30_name, None)
        raise
    k30_module = getattr(module, "k30", None)
    k30_path = getattr(k30_module, "__file__", None)
    if (
        sys.modules.get(k30_name) is not k30_module
        or not isinstance(k30_path, str)
        or Path(k30_path).resolve() != LOCAL_ADAPTER_PATH.resolve()
        or not LOCAL_ADAPTER_PATH.is_file()
        or LOCAL_ADAPTER_PATH.is_symlink()
        or _sha256_file(LOCAL_ADAPTER_PATH) != EXPECTED_LOCAL_ADAPTER_SHA256
    ):
        sys.modules.pop(continuation_name, None)
        sys.modules.pop(k30_name, None)
        raise UpdateError("loaded continuation k30 authority identity drifted")
    _CONTINUATION_ADAPTER = module
    return _CONTINUATION_ADAPTER


def _continuation_call(
    continuation_adapter: Any,
    label: str,
    function: Any,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Normalize producer contract failures at the reporting boundary."""

    try:
        return function(*args, **kwargs)
    except UpdateError:
        raise
    except Exception as exc:
        raise UpdateError(f"{label}: {exc}") from exc


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(raw: Any, *, label: str) -> PurePosixPath:
    if not isinstance(raw, str) or not raw:
        raise UpdateError(f"{label}: relative path is absent")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise UpdateError(f"{label}: unsafe relative path {raw!r}")
    return path


def _load_digested_file(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise UpdateError(f"{label}: file is absent or unsafe: {path}")
    value = load(path)
    verify_self_digest(value, label=label)
    return value


def _verify_local_binding(
    root: Path,
    raw: Any,
    *,
    expected_path: str,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(raw, Mapping):
        raise UpdateError(f"{label}: binding is absent")
    relative = _safe_relative_path(raw.get("path"), label=label)
    if relative.as_posix() != expected_path:
        raise UpdateError(f"{label}: bound path drifted")
    path = root / relative
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != raw.get("size_bytes")
        or _sha256_file(path) != raw.get("sha256")
    ):
        raise UpdateError(f"{label}: byte binding drifted")
    value = _load_digested_file(path, label=label)
    if value.get("sha256") != raw.get("canonical_sha256"):
        raise UpdateError(f"{label}: canonical binding drifted")
    return path, value


def _expected_local_waves(
    jobs_by_cell: Mapping[tuple[str, str], tuple[Path, Mapping[str, Any]]],
) -> tuple[tuple[str, str], ...]:
    def execution_id(regime: str, policy: str) -> str:
        return str(jobs_by_cell[(regime, policy)][1]["execution_id"])

    return (
        (
            execution_id("weak_weak", "append_only"),
            execution_id("intermediate_weak", "append_only"),
        ),
        (
            execution_id("strong_weak_u8", "always_commutation_reduced"),
            execution_id("weak_strong", "always_commutation_reduced"),
        ),
        (
            execution_id("strong_weak_u8", "append_only"),
            execution_id("weak_strong", "append_only"),
        ),
        (
            execution_id("intermediate_strong", "always_commutation_reduced"),
            execution_id("strong_strong_u8", "always_commutation_reduced"),
        ),
        (
            execution_id("intermediate_strong", "append_only"),
            execution_id("strong_strong_u8", "append_only"),
        ),
    )


def _local_campaign_authority(
    *,
    runtime_dir: Path,
    activation_dir: Path,
    expected_adapter_sha256: str,
    package_manifest_sha256: str,
    jobs_by_cell: Mapping[tuple[str, str], tuple[Path, Mapping[str, Any]]],
) -> dict[str, Any]:
    if len(expected_adapter_sha256) != 64:
        raise UpdateError("expected local adapter SHA-256 is not pinned")
    if activation_dir.is_symlink() or not activation_dir.is_dir():
        raise UpdateError("local activation directory is absent or unsafe")
    activation = _load_digested_file(
        activation_dir / "activation_manifest.json",
        label="local activation manifest",
    )
    waves = _expected_local_waves(jobs_by_cell)
    execution_ids = tuple(row for wave in waves for row in wave)
    excluded = tuple(
        str(jobs_by_cell[(regime, EXPECTED_POLICIES[0])][1]["execution_id"])
        for regime in ("weak_weak", "intermediate_weak")
    )
    _request_path, request = _verify_local_binding(
        activation_dir,
        activation.get("activation_request"),
        expected_path="activation_request.json",
        label="local activation request",
    )
    _preflight_path, preflight = _verify_local_binding(
        activation_dir,
        activation.get("host_preflight"),
        expected_path="host_preflight.json",
        label="local host preflight",
    )
    if (
        activation.get("schema") != LOCAL_ACTIVATION_SCHEMA
        or activation.get("status") != "passed_local_activation_prepared"
        or activation.get("package_manifest_sha256") != package_manifest_sha256
        or activation.get("local_adapter_sha256") != expected_adapter_sha256
        or tuple(activation.get("execution_ids", ())) != execution_ids
        or tuple(activation.get("excluded_completed_execution_ids", ())) != excluded
        or tuple(tuple(row) for row in activation.get("waves", ())) != waves
        or activation.get("authorization_count") != len(execution_ids)
        or activation.get("local_operational_target_horizon")
        != LOCAL_TARGET_HORIZON
        or activation.get("wave_size") != 2
        or activation.get("maximum_concurrency") != 1
        or activation.get("host_memory_safe_serialization") is not True
        or activation.get("execution_target") != LOCAL_EXECUTION_TARGET
        or activation.get("execution_authorized") is not True
        or activation.get("submission_authorized") is not False
        or activation.get("paper_evidence_adoption_authorized") is not False
        or request.get("schema") != LOCAL_REQUEST_SCHEMA
        or request.get("status") != "authorized_local_execution"
        or tuple(request.get("requested_execution_ids", ())) != execution_ids
        or request.get("local_adapter_sha256") != expected_adapter_sha256
        or request.get("execution_target") != LOCAL_EXECUTION_TARGET
        or request.get("execution_authorized") is not True
        or request.get("submission_authorized") is not False
        or request.get("round50_continuation_authorized_for_execution") is not False
        or request.get("paper_evidence_adoption_authorized") is not False
        or preflight.get("schema") != LOCAL_PREFLIGHT_SCHEMA
        or preflight.get("status") != "passed_inert_local_host_preflight"
        or preflight.get("sealed_worker_preflight_count") != len(execution_ids)
        or preflight.get("local_adapter_sha256") != expected_adapter_sha256
        or preflight.get("scientific_execution_performed") is not False
    ):
        raise UpdateError("local activation identity drifted")
    authorization_rows = activation.get("authorizations")
    if (
        not isinstance(authorization_rows, list)
        or [row.get("execution_id") for row in authorization_rows] != list(execution_ids)
    ):
        raise UpdateError("local authorization inventory drifted")
    authorizations: dict[str, dict[str, Any]] = {}
    job_by_execution_id = {
        str(job["execution_id"]): job for _, job in jobs_by_cell.values()
    }
    wave_by_execution_id = {
        execution_id: wave_number
        for wave_number, wave in enumerate(waves, start=1)
        for execution_id in wave
    }
    for row in authorization_rows:
        execution_id = str(row["execution_id"])
        _path, authority = _verify_local_binding(
            activation_dir,
            row,
            expected_path=f"authorizations/{execution_id}.json",
            label=f"local authorization {execution_id}",
        )
        job = job_by_execution_id[execution_id]
        if (
            authority.get("schema") != LOCAL_AUTHORIZATION_SCHEMA
            or authority.get("status") != "authorized_local_cell_execution"
            or authority.get("execution_id") != execution_id
            or authority.get("job_spec_sha256") != job.get("sha256")
            or authority.get("protocol_sha256") != job.get("protocol_sha256")
            or authority.get("route_contract_sha256")
            != job.get("route_contract_sha256")
            or authority.get("package_manifest_sha256")
            != package_manifest_sha256
            or authority.get("local_adapter_sha256")
            != expected_adapter_sha256
            or authority.get("wave") != wave_by_execution_id[execution_id]
            or authority.get("local_operational_target_horizon")
            != LOCAL_TARGET_HORIZON
            or authority.get("fresh_start") is not True
            or authority.get("execution_target")
            != LOCAL_EXECUTION_TARGET
            or authority.get("execution_authorized") is not True
            or authority.get("submission_authorized") is not False
            or authority.get("round50_continuation_authorized_for_execution")
            is not False
            or authority.get("paper_evidence_adoption_authorized") is not False
        ):
            raise UpdateError(f"local authorization drifted: {execution_id}")
        authorizations[execution_id] = authority

    runtime: dict[str, Any] | None = None
    if runtime_dir.exists() or runtime_dir.is_symlink():
        if runtime_dir.is_symlink() or not runtime_dir.is_dir():
            raise UpdateError("local runtime directory is unsafe")
        runtime = _load_digested_file(
            runtime_dir / "runtime_manifest.json",
            label="local runtime manifest",
        )
        expected_runtime_waves = tuple(
            (int(row.get("wave", -1)), tuple(row.get("execution_ids", ())))
            for row in runtime.get("waves", ())
            if isinstance(row, Mapping)
        )
        if (
            runtime.get("schema") != LOCAL_RUNTIME_SCHEMA
            or runtime.get("status") != "authorized_pending_waves"
            or runtime.get("adapter_sha256") != expected_adapter_sha256
            or runtime.get("activation_manifest_sha256") != activation.get("sha256")
            or runtime.get("package_manifest_sha256") != package_manifest_sha256
            or tuple(runtime.get("execution_ids", ())) != execution_ids
            or tuple(runtime.get("excluded_completed_execution_ids", ())) != excluded
            or expected_runtime_waves
            != tuple((index, wave) for index, wave in enumerate(waves, start=1))
            or runtime.get("local_operational_target_horizon")
            != LOCAL_TARGET_HORIZON
            or runtime.get("wave_size") != 2
            or runtime.get("maximum_concurrency") != 1
            or runtime.get("host_memory_safe_serialization") is not True
            or runtime.get("execution_target") != LOCAL_EXECUTION_TARGET
            or runtime.get("round50_continuation_execution_in_scope") is not False
            or runtime.get("execution_authorized") is not True
            or runtime.get("submission_authorized") is not False
            or runtime.get("paper_evidence_adoption_authorized") is not False
        ):
            raise UpdateError("local runtime manifest drifted")
    return {
        "activation": activation,
        "runtime": runtime,
        "execution_ids": execution_ids,
        "excluded_execution_ids": excluded,
        "waves": waves,
        "authorizations": authorizations,
    }


def _verify_local_artifact_inventory(
    *, runtime_dir: Path, run_root: Path, receipt: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    rows = receipt.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise UpdateError("local worker artifact inventory is absent")
    inventory: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise UpdateError("local worker artifact row is malformed")
        relative = _safe_relative_path(row.get("path"), label="local artifact")
        name = relative.as_posix()
        path = runtime_dir / relative
        if (
            name in inventory
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != row.get("size_bytes")
            or _sha256_file(path) != row.get("sha256")
        ):
            raise UpdateError(f"local artifact binding drifted: {name}")
        try:
            path.relative_to(run_root)
        except ValueError as exc:
            raise UpdateError(f"local artifact escapes run root: {name}") from exc
        inventory[name] = row
    observed = {
        path.relative_to(runtime_dir).as_posix()
        for path in run_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if any(path.is_symlink() for path in run_root.rglob("*")):
        raise UpdateError("local run root contains a symbolic link")
    if observed != set(inventory):
        raise UpdateError("local run root contains missing or unbound artifacts")
    return inventory


def _local_summary_result(
    *,
    runtime_dir: Path,
    execution_id: str,
    job_path: Path,
    job: Mapping[str, Any],
    authority: Mapping[str, Any],
    compile_costs: bool,
) -> tuple[str, dict[str, Any] | None]:
    run_root = runtime_dir / "runs" / execution_id
    receipt_path = runtime_dir / "worker_receipts" / f"{execution_id}.json"
    gate_path = runtime_dir / "plateau_gates" / f"{execution_id}.json"
    closure_paths = (run_root, receipt_path, gate_path)
    present = tuple(path.exists() or path.is_symlink() for path in closure_paths)
    if not any(present):
        return "pending_not_published", None
    if not (
        run_root.is_dir()
        and not run_root.is_symlink()
        and receipt_path.is_file()
        and not receipt_path.is_symlink()
        and gate_path.is_file()
        and not gate_path.is_symlink()
    ):
        return "published_partial_unclosed", None

    manifest_path = run_root / "execution_manifest.json"
    manifest = _load_digested_file(
        manifest_path, label=f"local execution manifest {execution_id}"
    )
    receipt = _load_digested_file(
        receipt_path, label=f"local worker receipt {execution_id}"
    )
    gate = _load_digested_file(gate_path, label=f"local plateau gate {execution_id}")
    internal_gate = run_root / "gate/round30_effective_plateau.json"
    expected_route = COMPARATOR_ROUTE_CONTRACT_SHA256_BY_POLICY[
        str(job["comparator_policy"])
    ]
    if (
        manifest.get("schema") != LOCAL_EXECUTION_SCHEMA
        or manifest.get("status") != "passed"
        or manifest.get("execution_target") != LOCAL_EXECUTION_TARGET
        or manifest.get("execution_id") != execution_id
        or manifest.get("job_spec_sha256") != job.get("sha256")
        or manifest.get("authorization_sha256") != authority.get("sha256")
        or manifest.get("protocol_sha256") != job.get("protocol_sha256")
        or manifest.get("route_contract_sha256") != expected_route
        or manifest.get("comparator_policy") != job.get("comparator_policy")
        or manifest.get("local_operational_target_horizon")
        != LOCAL_TARGET_HORIZON
        or manifest.get("controller_rounds_completed") != LOCAL_TARGET_HORIZON
        or manifest.get("fresh_start") is not True
        or manifest.get("source_checkpoint_consumed") is not False
        or manifest.get("round50_continuation_executed") is not False
        or manifest.get("paper_evidence_adoption_authorized") is not False
        or receipt.get("schema") != LOCAL_WORKER_RECEIPT_SCHEMA
        or receipt.get("status") != "passed"
        or receipt.get("execution_target") != LOCAL_EXECUTION_TARGET
        or receipt.get("execution_id") != execution_id
        or receipt.get("job_spec_sha256") != job.get("sha256")
        or receipt.get("authorization_sha256") != authority.get("sha256")
        or receipt.get("execution_manifest_sha256") != manifest.get("sha256")
        or receipt.get("local_operational_target_horizon")
        != LOCAL_TARGET_HORIZON
        or receipt.get("controller_rounds_completed") != LOCAL_TARGET_HORIZON
        or receipt.get("fresh_start") is not True
        or receipt.get("round50_continuation_executed") is not False
        or gate.get("schema") != LOCAL_PLATEAU_GATE_SCHEMA
        or gate.get("status") != "passed"
        or gate.get("execution_id") != execution_id
        or gate.get("regime_id") != job.get("regime_id")
        or gate.get("comparator_policy") != job.get("comparator_policy")
        or gate.get("policy") != "paper_i_effective_plateau_v1"
        or gate.get("available_horizon_controller_rounds")
        != LOCAL_TARGET_HORIZON
        or gate.get("horizon_scope") != "deliberately_stopped_prefix"
        or gate.get("resume_execution_performed") is not False
        or gate.get("round50_protocol_derived") is not False
        or gate.get("paper_evidence_adoption_authorized") is not False
        or receipt.get("plateau_gate_sha256") != gate.get("sha256")
        or manifest.get("plateau_gate_sha256") != gate.get("sha256")
        or not internal_gate.is_file()
        or internal_gate.is_symlink()
        or _sha256_file(internal_gate) != _sha256_file(gate_path)
    ):
        raise UpdateError(f"local completed-cell identity drifted: {execution_id}")
    inventory = _verify_local_artifact_inventory(
        runtime_dir=runtime_dir, run_root=run_root, receipt=receipt
    )
    output_payloads = manifest.get("output_payloads")
    if not isinstance(output_payloads, Mapping):
        raise UpdateError(f"local output inventory is absent: {execution_id}")
    for role in ("checkpoint", "result", "estimator_ledger", "summary"):
        row = output_payloads.get(role)
        if not isinstance(row, Mapping):
            raise UpdateError(f"local output inventory lacks {role}: {execution_id}")
        relative = _safe_relative_path(row.get("path"), label=f"local {role}")
        artifact = inventory.get(relative.as_posix())
        if (
            artifact is None
            or artifact.get("sha256") != row.get("sha256")
            or artifact.get("size_bytes") != row.get("size_bytes")
        ):
            raise UpdateError(f"local {role} binding drifted: {execution_id}")
    gate_binding = output_payloads.get("round30_effective_plateau_gate")
    expected_gate_relative = (
        PurePosixPath("runs")
        / execution_id
        / "gate/round30_effective_plateau.json"
    ).as_posix()
    if (
        not isinstance(gate_binding, Mapping)
        or gate_binding.get("path") != expected_gate_relative
        or gate_binding.get("sha256") != _sha256_file(internal_gate)
        or gate_binding.get("size_bytes") != internal_gate.stat().st_size
        or expected_gate_relative not in inventory
    ):
        raise UpdateError(f"local gate output binding drifted: {execution_id}")

    summary_row = output_payloads["summary"]
    summary_path = runtime_dir / _safe_relative_path(
        summary_row["path"], label="local summary"
    )
    summary = load(summary_path)
    trace = summary.get("accepted_error_trace")
    plateau = summary.get("effective_plateau")
    if (
        summary.get("schema") != "paper_i_run_summary_v1"
        or summary.get("horizon_scope") != "deliberately_stopped_prefix"
        or summary.get("available_controller_rounds") != LOCAL_TARGET_HORIZON
        or not isinstance(trace, list)
        or [row.get("controller_round") for row in trace]
        != list(range(1, LOCAL_TARGET_HORIZON + 1))
        or not isinstance(plateau, Mapping)
    ):
        raise UpdateError(f"local round-30 summary drifted: {execution_id}")
    exact = float(summary["provenance"]["exact_same_cutoff_energy"])
    points: list[dict[str, Any]] = []
    for row in trace:
        fingerprint = row.get("projective_state_fingerprint")
        if not math.isclose(
            float(row["exact_same_cutoff_energy"]),
            exact,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise UpdateError(f"local same-cutoff reference drifted: {execution_id}")
        error = float(row["absolute_energy_error"])
        if (
            not math.isfinite(error)
            or error < 0.0
            or not _valid_projective_state_fingerprint(fingerprint)
        ):
            raise UpdateError(f"local trajectory error is invalid: {execution_id}")
        points.append(
            {
                "k": int(row["controller_round"]),
                "energy": float(row["accepted_energy"]),
                "error": error,
                "active_ansatz_depth": int(row["active_ansatz_depth"]),
                "projective_state_fingerprint": fingerprint,
            }
        )
    errors = [float(row["error"]) for row in points]
    best = min(errors)
    threshold = 1.10 * best
    selected_round = next(
        index for index, error in enumerate(errors, start=1) if error <= threshold
    )
    selected_at_cap = selected_round == LOCAL_TARGET_HORIZON
    terminal_in_band = errors[-1] <= threshold
    expected_extension_decision = (
        "eligible_for_authenticated_resume_to_k50"
        if not terminal_in_band or selected_at_cap
        else "stop_at_k30"
    )
    expected_classification = (
        "endpoint_outside_effective_band_at_k30"
        if not terminal_in_band
        else (
            "right_censored_at_k30"
            if selected_at_cap
            else "effective_plateau_observed_within_k30"
        )
    )
    source_horizon = int(job["target_horizon"])
    expected_materialization = (
        "authenticated_resume_adapter_only"
        if source_horizon >= CONTINUATION_TARGET_HORIZON
        else "new_source_locked_k50_protocol_required"
    )
    if (
        plateau.get("policy") != "paper_i_effective_plateau_v1"
        or plateau.get("controller_round") != selected_round
        or plateau.get("available_horizon_controller_rounds")
        != LOCAL_TARGET_HORIZON
        or plateau.get("horizon_scope") != "deliberately_stopped_prefix"
        or not math.isclose(
            float(plateau.get("absolute_energy_error")),
            errors[selected_round - 1],
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or gate.get("selected_controller_round") != selected_round
        or not _same_float(
            gate.get("selected_absolute_energy_error"),
            errors[selected_round - 1],
            tolerance=1.0e-15,
        )
        or not math.isclose(
            float(gate.get("best_observed_absolute_energy_error")),
            best,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or not _same_float(
            gate.get("selection_threshold"), threshold, tolerance=1.0e-15
        )
        or not _same_float(
            gate.get("terminal_absolute_energy_error"),
            errors[-1],
            tolerance=1.0e-15,
        )
        or gate.get("terminal_in_effective_band") is not terminal_in_band
        or gate.get("selected_at_cap") is not selected_at_cap
        or gate.get("classification") != expected_classification
        or gate.get("extension_decision") != expected_extension_decision
        or gate.get("source_authorized_horizon") != source_horizon
        or gate.get("continuation_target_horizon")
        != CONTINUATION_TARGET_HORIZON
        or gate.get("continuation_materialization_requirement")
        != expected_materialization
        or gate.get("summary_effective_plateau_matches_recomputation") is not True
        or gate.get("accepted_error_trace_canonical_sha256")
        != hashlib.sha256(
            json.dumps(
                trace,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    ):
        raise UpdateError(f"local effective-plateau gate drifted: {execution_id}")
    costs: dict[str, int] | None = None
    compile_receipt: dict[str, Any] | None = None
    if compile_costs:
        costs, compile_receipt = completed_pages._compile_cost_tuple(
            summary, round_index=LOCAL_TARGET_HORIZON
        )
    extension_decision = expected_extension_decision
    reporting_status = (
        "completed_authenticated_local_k30"
        if extension_decision == "stop_at_k30"
        else "authenticated_local_k30_right_censored_partial"
    )
    return "closed_authenticated_local_receipt", {
        "status": reporting_status,
        "execution_id": execution_id,
        "regime_id": str(job["regime_id"]),
        "comparator_policy": str(job["comparator_policy"]),
        "wave": int(authority["wave"]),
        "target_horizon": LOCAL_TARGET_HORIZON,
        "source_authorized_horizon": int(job["target_horizon"]),
        "points": points,
        "terminal": copy.deepcopy(points[-1]),
        "costs": costs,
        "compile": compile_receipt,
        "effective_plateau_gate": {
            "selected_controller_round": selected_round,
            "classification": gate.get("classification"),
            "extension_decision": extension_decision,
            "canonical_sha256": gate["sha256"],
        },
        "sources": {
            "execution_manifest": {
                **binding(manifest_path),
                "canonical_sha256": manifest["sha256"],
            },
            "worker_receipt": {
                **binding(receipt_path),
                "canonical_sha256": receipt["sha256"],
            },
            "plateau_gate": {
                **binding(gate_path),
                "canonical_sha256": gate["sha256"],
            },
            "summary": binding(summary_path),
            "authorization_canonical_sha256": authority["sha256"],
            "job": binding(job_path),
            "all_receipt_artifact_hashes_verified": True,
            "unbound_run_file_count": 0,
        },
    }


def local_comparator_inventory(
    *,
    runtime_dir: Path,
    activation_dir: Path,
    expected_adapter_sha256: str,
    package_manifest_sha256: str,
    jobs_by_cell: Mapping[tuple[str, str], tuple[Path, Mapping[str, Any]]],
    compile_costs: bool,
) -> dict[str, Any]:
    if not activation_dir.exists() and not activation_dir.is_symlink():
        if runtime_dir.exists() or runtime_dir.is_symlink():
            raise UpdateError("local runtime exists without its activation authority")
        return {
            "campaign_state": "not_activated",
            "execution_ids": [],
            "cell_states": {},
            "completed": {},
            "sources": {},
        }
    authority = _local_campaign_authority(
        runtime_dir=runtime_dir,
        activation_dir=activation_dir,
        expected_adapter_sha256=expected_adapter_sha256,
        package_manifest_sha256=package_manifest_sha256,
        jobs_by_cell=jobs_by_cell,
    )
    completed: dict[str, dict[str, Any]] = {}
    states: dict[str, str] = {}
    jobs_by_execution_id = {
        str(job["execution_id"]): (path, job)
        for path, job in jobs_by_cell.values()
    }
    if authority["runtime"] is None:
        states = {
            execution_id: "pending_runtime_not_materialized"
            for execution_id in authority["execution_ids"]
        }
    else:
        for execution_id in authority["execution_ids"]:
            job_path, job = jobs_by_execution_id[execution_id]
            state, result = _local_summary_result(
                runtime_dir=runtime_dir,
                execution_id=execution_id,
                job_path=job_path,
                job=job,
                authority=authority["authorizations"][execution_id],
                compile_costs=compile_costs,
            )
            states[execution_id] = state
            if result is not None:
                result["evidence_revision"] = _result_evidence_revision(result)
                completed[execution_id] = result
    sources: dict[str, Any] = {
        "activation_manifest": {
            **binding(activation_dir / "activation_manifest.json"),
            "canonical_sha256": authority["activation"]["sha256"],
        },
        "expected_local_adapter_sha256": expected_adapter_sha256,
    }
    if authority["runtime"] is not None:
        sources["runtime_manifest"] = {
            **binding(runtime_dir / "runtime_manifest.json"),
            "canonical_sha256": authority["runtime"]["sha256"],
        }
    return {
        "campaign_state": (
            "runtime_materialized"
            if authority["runtime"] is not None
            else "activated_runtime_pending"
        ),
        "execution_ids": list(authority["execution_ids"]),
        "cell_states": states,
        "completed": completed,
        "sources": sources,
    }


def authenticated_local_comparator_inventory(
    *,
    runtime_dir: Path = LOCAL_RUNTIME_DIR,
    activation_dir: Path = LOCAL_ACTIVATION_DIR,
    expected_adapter_sha256: str = EXPECTED_LOCAL_ADAPTER_SHA256,
    compile_costs: bool = False,
) -> dict[str, Any]:
    """Authenticate the fixed ten-cell local campaign without rendering."""

    package16 = load(PAGE16_PACKAGE / "package_manifest.json")
    verify_self_digest(package16, label="Page-16 comparator package")
    if (
        tuple(package16.get("comparator_policies", ())) != EXPECTED_POLICIES
        or len(package16.get("jobs", ())) != 12
    ):
        raise UpdateError("Page-16 comparator package coverage drifted")
    jobs = completed_pages._package_jobs(PAGE16_PACKAGE, package16["sha256"])
    jobs_by_cell: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    for regime in REGIME_ORDER:
        nph = 3 if regime in REGIME_ORDER[:3] else 7
        for policy in EXPECTED_POLICIES:
            matches = [
                (path, job)
                for execution_id, (path, job) in jobs.items()
                if f"__{regime}__nph{nph}__" in execution_id
                and execution_id.endswith(f"_{policy}")
            ]
            if len(matches) != 1:
                raise UpdateError(
                    f"Page-16 local job coverage drifted: {regime}/{policy}"
                )
            job_path, job = matches[0]
            if (
                job.get("regime_id") != regime
                or job.get("nph") != nph
                or job.get("comparator_policy") != policy
                or job.get("route_contract_sha256")
                != COMPARATOR_ROUTE_CONTRACT_SHA256_BY_POLICY[policy]
            ):
                raise UpdateError(
                    f"Page-16 local job identity drifted: {regime}/{policy}"
                )
            jobs_by_cell[(regime, policy)] = (job_path, job)
    return local_comparator_inventory(
        runtime_dir=runtime_dir,
        activation_dir=activation_dir,
        expected_adapter_sha256=expected_adapter_sha256,
        package_manifest_sha256=package16["sha256"],
        jobs_by_cell=jobs_by_cell,
        compile_costs=compile_costs,
    )


def _same_float(left: Any, right: Any, *, tolerance: float = 1.0e-12) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=tolerance
        )
    except (TypeError, ValueError):
        return False


def _valid_projective_state_fingerprint(value: Any) -> bool:
    prefix = "projective_state_v1:"
    return (
        isinstance(value, str)
        and value.startswith(prefix)
        and len(value) == len(prefix) + 64
        and all(character in "0123456789abcdef" for character in value[len(prefix):])
    )


def _finite_float(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _result_evidence_revision(result: Mapping[str, Any]) -> str:
    sources = result.get("sources")
    if not isinstance(sources, Mapping):
        raise UpdateError("authenticated comparator result lacks sources")
    return _canonical_sha256(
        {
            "execution_id": result.get("execution_id"),
            "status": result.get("status"),
            "target_horizon": result.get("target_horizon"),
            "terminal": result.get("terminal"),
            "sources": sources,
        }
    )


def _validate_decision_snapshot(
    continuation_adapter: Any,
    snapshot_value: Mapping[str, Any],
    *,
    k30_completed: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    conditional = tuple(continuation_adapter.CONDITIONAL_EXECUTION_IDS)
    decisions = snapshot_value.get("decisions")
    eligible = tuple(str(row) for row in snapshot_value.get("eligible_execution_ids", ()))
    stopped = tuple(str(row) for row in snapshot_value.get("stop_at_k30_execution_ids", ()))
    pending = tuple(str(row) for row in snapshot_value.get("pending_execution_ids", ()))
    if (
        not isinstance(decisions, list)
        or snapshot_value.get("schema")
        != continuation_adapter.DECISION_STATUS_SCHEMA
        or snapshot_value.get("sha256")
        != _canonical_sha256(
            {key: value for key, value in snapshot_value.items() if key != "sha256"}
        )
        or snapshot_value.get("conditional_execution_ids") != list(conditional)
        or snapshot_value.get("terminal_chtc_k50_execution_ids")
        != list(continuation_adapter.TERMINAL_CHTC_EXECUTION_IDS)
        or snapshot_value.get("scientific_execution_performed") is not False
        or len(eligible) != len(set(eligible))
        or len(stopped) != len(set(stopped))
        or len(pending) != len(set(pending))
        or set(eligible).intersection(stopped)
        or set(eligible).intersection(pending)
        or set(stopped).intersection(pending)
        or set(eligible).union(stopped, pending) != set(conditional)
        or [row.get("execution_id") for row in decisions]
        != [execution_id for execution_id in conditional if execution_id not in pending]
        or snapshot_value.get("closed_decision_count") != len(decisions)
        or snapshot_value.get("all_decisions_closed") is not (not bool(pending))
        or snapshot_value.get("status")
        != (
            "waiting_for_all_k30_decisions"
            if pending
            else "passed_all_k30_decisions_closed"
        )
    ):
        raise UpdateError("continuation k30 decision inventory drifted")
    decision_by_id = {str(row["execution_id"]): row for row in decisions}
    for execution_id in (*eligible, *stopped):
        decision = decision_by_id[execution_id]
        source = k30_completed.get(execution_id)
        expected = (
            "eligible_for_authenticated_resume_to_k50"
            if execution_id in eligible
            else "stop_at_k30"
        )
        if (
            source is None
            or decision.get("extension_decision") != expected
            or source.get("effective_plateau_gate", {}).get("extension_decision")
            != expected
            or source.get("effective_plateau_gate", {}).get("canonical_sha256")
            != decision.get("k30_plateau_gate_sha256")
        ):
            raise UpdateError(f"continuation k30 decision drifted: {execution_id}")
    return eligible, stopped, pending


def _continuation_summary_result(
    *,
    continuation_adapter: Any,
    runtime_dir: Path,
    runtime: Mapping[str, Any],
    execution_id: str,
    authority: Mapping[str, Any],
    source_result: Mapping[str, Any],
    compile_costs: bool,
) -> dict[str, Any]:
    run_root = runtime_dir / "runs" / execution_id
    manifest_path = run_root / "execution_manifest.json"
    receipt_path = runtime_dir / "worker_receipts" / f"{execution_id}.json"
    manifest = _load_digested_file(
        manifest_path, label=f"continuation execution manifest {execution_id}"
    )
    receipt = _load_digested_file(
        receipt_path, label=f"continuation worker receipt {execution_id}"
    )
    worker = _continuation_call(
        continuation_adapter,
        "continuation worker load failed",
        continuation_adapter.k30._load_worker,
    )
    jobs = _continuation_call(
        continuation_adapter,
        "continuation job inventory failed",
        continuation_adapter._job_by_id,
        worker,
    )
    job = jobs[execution_id]
    target_protocol = authority.get("target_protocol")
    if not isinstance(target_protocol, Mapping):
        raise UpdateError(f"continuation target protocol is absent: {execution_id}")
    expected_target_protocol = target_protocol.get("target_protocol_sha256")
    prefix = manifest.get("accepted_prefix_preservation")
    if (
        manifest.get("schema") != continuation_adapter.EXECUTION_SCHEMA
        or manifest.get("status") != "passed"
        or manifest.get("execution_target")
        != continuation_adapter.LOCAL_EXECUTION_TARGET
        or manifest.get("source_package_manifest_sha256")
        != continuation_adapter.k30.PACKAGE_MANIFEST_CANONICAL_SHA256
        or manifest.get("adapter_sha256")
        != EXPECTED_CONTINUATION_ADAPTER_SHA256
        or manifest.get("activation_manifest_sha256")
        != runtime.get("activation_manifest_sha256")
        or manifest.get("execution_id") != execution_id
        or manifest.get("job_spec_sha256") != job.get("sha256")
        or manifest.get("job_spec_sha256") != authority.get("job_spec_sha256")
        or manifest.get("resume_authorization_sha256") != authority.get("sha256")
        or manifest.get("source_protocol_sha256")
        != authority.get("source_protocol_sha256")
        or manifest.get("protocol_sha256") != expected_target_protocol
        or manifest.get("route_contract_sha256")
        != authority.get("route_contract_sha256")
        or manifest.get("comparator_policy") != authority.get("comparator_policy")
        or manifest.get("resume_round") != 30
        or manifest.get("target_horizon") != CONTINUATION_TARGET_HORIZON
        or manifest.get("controller_rounds_completed")
        != CONTINUATION_TARGET_HORIZON
        or manifest.get("source_checkpoint_sha256")
        != authority.get("resume_checkpoint", {}).get("sha256")
        or manifest.get("source_plateau_gate_sha256")
        != authority.get("k30_plateau_gate_sha256")
        or manifest.get("accepted_state_resume") is not True
        or manifest.get("fresh_start") is not False
        or not isinstance(prefix, Mapping)
        or prefix.get("status") != "passed"
        or prefix.get("source_round") != 30
        or prefix.get("all_non_energy_fields_exact") is not True
        or prefix.get("energy_comparison") != "128_ulp_roundoff_only"
        or not _valid_projective_state_fingerprint(
            prefix.get("terminal_state_fingerprint")
        )
        or not _finite_float(prefix.get("terminal_energy"))
        or manifest.get("paper_evidence_adoption_authorized") is not False
        or receipt.get("schema") != continuation_adapter.WORKER_RECEIPT_SCHEMA
        or receipt.get("status") != "passed"
        or receipt.get("execution_id") != execution_id
        or receipt.get("job_spec_sha256") != job.get("sha256")
        or receipt.get("resume_authorization_sha256") != authority.get("sha256")
        or receipt.get("execution_manifest_sha256") != manifest.get("sha256")
        or receipt.get("resume_round") != 30
        or receipt.get("controller_rounds_completed")
        != CONTINUATION_TARGET_HORIZON
        or receipt.get("accepted_state_resume") is not True
        or receipt.get("fresh_start") is not False
    ):
        raise UpdateError(f"continuation closure identity drifted: {execution_id}")
    inventory = _verify_local_artifact_inventory(
        runtime_dir=runtime_dir, run_root=run_root, receipt=receipt
    )
    output_payloads = manifest.get("output_payloads")
    if not isinstance(output_payloads, Mapping):
        raise UpdateError(f"continuation output inventory is absent: {execution_id}")
    run_inventory = {
        PurePosixPath(path)
        .relative_to(PurePosixPath("runs") / execution_id)
        .as_posix(): row
        for path, row in inventory.items()
        if path != f"runs/{execution_id}/execution_manifest.json"
    }
    if set(output_payloads) != set(run_inventory):
        raise UpdateError(f"continuation output inventory drifted: {execution_id}")
    for relative, row in output_payloads.items():
        if (
            not isinstance(row, Mapping)
            or run_inventory.get(str(relative), {}).get("sha256") != row.get("sha256")
            or run_inventory.get(str(relative), {}).get("size_bytes")
            != row.get("size_bytes")
        ):
            raise UpdateError(
                f"continuation output binding drifted: {execution_id}/{relative}"
            )
    required = {
        "checkpoints/current.json",
        "result/estimator_ledger.json",
        "result/result.json",
        "summary/summary.json",
        "continuation/resolved_protocol.json",
        "continuation/resume_authorization.json",
        "continuation/source_lock_audit.json",
    }
    if not required.issubset(output_payloads):
        raise UpdateError(f"continuation required payload is absent: {execution_id}")

    source_audit_path = run_root / "continuation/source_lock_audit.json"
    source_audit = _load_digested_file(
        source_audit_path, label=f"continuation source-lock audit {execution_id}"
    )
    resolved_protocol = _load_digested_file(
        run_root / "continuation/resolved_protocol.json",
        label=f"continuation resolved protocol {execution_id}",
    )
    embedded_authority = _load_digested_file(
        run_root / "continuation/resume_authorization.json",
        label=f"continuation embedded authorization {execution_id}",
    )
    expected_derivation_kind = (
        "source_authorized_k50_protocol_reused_exactly"
        if int(job["target_horizon"]) >= CONTINUATION_TARGET_HORIZON
        else "source_locked_sole_horizon_delta_30_to_50"
    )
    if (
        embedded_authority != authority
        or resolved_protocol.get("sha256") != manifest.get("protocol_sha256")
        or source_audit.get("schema")
        != "paper_i_page16_k30_to_k50_source_lock_audit_v2"
        or source_audit.get("status") != "passed"
        or source_audit.get("execution_id") != execution_id
        or source_audit.get("source_protocol_sha256")
        != manifest.get("source_protocol_sha256")
        or source_audit.get("target_protocol_sha256")
        != manifest.get("protocol_sha256")
        or source_audit.get("common_route_contract_sha256")
        != manifest.get("route_contract_sha256")
        or source_audit.get("comparator_policy")
        != manifest.get("comparator_policy")
        or source_audit.get("source_horizon") != int(job["target_horizon"])
        or source_audit.get("resume_round") != 30
        or source_audit.get("target_horizon") != CONTINUATION_TARGET_HORIZON
        or source_audit.get("protocol_derivation_kind")
        != expected_derivation_kind
        or manifest.get("protocol_derivation_kind")
        != expected_derivation_kind
        or source_audit.get("non_horizon_protocol_diff") != []
        or source_audit.get("source_locks_exact") is not True
        or source_audit.get("resume_checkpoint_sha256")
        != authority.get("resume_checkpoint", {}).get("sha256")
        or source_audit.get("resume_checkpoint_siblings")
        != authority.get("resume_checkpoint_siblings")
        or source_audit.get("accepted_prefix_preservation") != prefix
        or source_audit.get("sha256") != manifest.get("source_lock_audit_sha256")
    ):
        raise UpdateError(f"continuation source-lock audit drifted: {execution_id}")

    summary_path = run_root / "summary/summary.json"
    summary = load(summary_path)
    trace = summary.get("accepted_error_trace")
    plateau = summary.get("effective_plateau")
    if (
        summary.get("schema") != "paper_i_run_summary_v1"
        or summary.get("horizon_scope") != "deliberately_stopped_prefix"
        or summary.get("available_controller_rounds")
        != CONTINUATION_TARGET_HORIZON
        or not isinstance(trace, list)
        or [row.get("controller_round") for row in trace]
        != list(range(1, CONTINUATION_TARGET_HORIZON + 1))
        or not isinstance(plateau, Mapping)
    ):
        raise UpdateError(f"continuation round-50 summary drifted: {execution_id}")
    exact = float(summary.get("provenance", {}).get("exact_same_cutoff_energy"))
    if not math.isfinite(exact):
        raise UpdateError(f"continuation exact reference is invalid: {execution_id}")
    points: list[dict[str, Any]] = []
    for row in trace:
        energy = float(row["accepted_energy"])
        error = float(row["absolute_energy_error"])
        fingerprint = row.get("projective_state_fingerprint")
        if (
            not math.isfinite(energy)
            or not math.isfinite(error)
            or error < 0.0
            or not _same_float(row.get("exact_same_cutoff_energy"), exact)
            or not _same_float(error, abs(energy - exact))
            or not _valid_projective_state_fingerprint(fingerprint)
        ):
            raise UpdateError(f"continuation trajectory drifted: {execution_id}")
        points.append(
            {
                "k": int(row["controller_round"]),
                "energy": energy,
                "error": error,
                "active_ansatz_depth": int(row["active_ansatz_depth"]),
                "projective_state_fingerprint": fingerprint,
            }
        )
    source_points = source_result.get("points")
    if not isinstance(source_points, list) or len(source_points) != 30:
        raise UpdateError(f"continuation k30 source trace drifted: {execution_id}")
    for source, continued in zip(source_points, points[:30], strict=True):
        energy_tolerance = 128.0 * math.ulp(
            max(1.0, abs(float(source["energy"])), abs(float(continued["energy"])))
        )
        if (
            source.get("k") != continued["k"]
            or source.get("active_ansatz_depth") != continued["active_ansatz_depth"]
            or source.get("projective_state_fingerprint")
            != continued["projective_state_fingerprint"]
            or not _same_float(
                source.get("energy"), continued["energy"], tolerance=energy_tolerance
            )
            or not _same_float(
                source.get("error"), continued["error"], tolerance=energy_tolerance
            )
        ):
            raise UpdateError(f"continuation accepted prefix drifted: {execution_id}")
    terminal_prefix_point = points[29]
    prefix_energy_tolerance = 128.0 * math.ulp(
        max(
            1.0,
            abs(float(prefix["terminal_energy"])),
            abs(float(terminal_prefix_point["energy"])),
        )
    )
    if (
        not _same_float(
            prefix.get("terminal_energy"),
            terminal_prefix_point["energy"],
            tolerance=prefix_energy_tolerance,
        )
        or prefix.get("terminal_state_fingerprint")
        != terminal_prefix_point["projective_state_fingerprint"]
    ):
        raise UpdateError(f"continuation prefix terminal drifted: {execution_id}")
    errors = [row["error"] for row in points]
    best = min(errors)
    threshold = 1.10 * best
    selected_round = next(
        index for index, error in enumerate(errors, start=1) if error <= threshold
    )
    if (
        plateau.get("policy") != "paper_i_effective_plateau_v1"
        or plateau.get("controller_round") != selected_round
        or plateau.get("available_horizon_controller_rounds")
        != CONTINUATION_TARGET_HORIZON
        or plateau.get("horizon_scope") != "deliberately_stopped_prefix"
        or not _same_float(
            plateau.get("absolute_energy_error"), errors[selected_round - 1],
            tolerance=1.0e-15,
        )
        or not _same_float(plateau.get("best_observed_error"), best, tolerance=1.0e-15)
        or not _same_float(
            plateau.get("selection_threshold"), threshold, tolerance=1.0e-15
        )
    ):
        raise UpdateError(f"continuation effective plateau drifted: {execution_id}")
    costs: dict[str, int] | None = None
    compile_receipt: dict[str, Any] | None = None
    if compile_costs:
        costs, compile_receipt = completed_pages._compile_cost_tuple(
            summary, round_index=CONTINUATION_TARGET_HORIZON
        )
    result: dict[str, Any] = {
        "status": "completed_authenticated_local_k50_continuation",
        "execution_id": execution_id,
        "regime_id": str(authority["regime_id"]),
        "comparator_policy": str(authority["comparator_policy"]),
        "target_horizon": CONTINUATION_TARGET_HORIZON,
        "resume_round": 30,
        "source_authorized_horizon": int(job["target_horizon"]),
        "points": points,
        "terminal": copy.deepcopy(points[-1]),
        "costs": costs,
        "compile": compile_receipt,
        "effective_plateau": {
            "selected_controller_round": selected_round,
            "selected_absolute_energy_error": errors[selected_round - 1],
            "best_observed_absolute_energy_error": best,
            "selection_threshold": threshold,
        },
        "source_k30_effective_plateau_gate": copy.deepcopy(
            source_result["effective_plateau_gate"]
        ),
        "sources": {
            "execution_manifest": {
                **binding(manifest_path),
                "canonical_sha256": manifest["sha256"],
            },
            "worker_receipt": {
                **binding(receipt_path),
                "canonical_sha256": receipt["sha256"],
            },
            "summary": binding(summary_path),
            "source_lock_audit": {
                **binding(source_audit_path),
                "canonical_sha256": source_audit["sha256"],
            },
            "resume_authorization_canonical_sha256": authority["sha256"],
            "accepted_prefix_preservation_authenticated": True,
            "all_receipt_artifact_hashes_verified": True,
            "unbound_run_file_count": 0,
        },
    }
    result["evidence_revision"] = _result_evidence_revision(result)
    return result


def _authenticate_macro_terminal_receipt(
    *,
    continuation_adapter: Any,
    activation: Mapping[str, Any],
    runtime: Mapping[str, Any],
    decision_snapshot: Mapping[str, Any],
    path: Path,
) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    if (
        not CONTINUATION_SUPERVISOR_PATH.is_file()
        or CONTINUATION_SUPERVISOR_PATH.is_symlink()
        or _sha256_file(CONTINUATION_SUPERVISOR_PATH)
        != EXPECTED_CONTINUATION_SUPERVISOR_SHA256
    ):
        raise UpdateError("pinned continuation supervisor is absent or unsafe")
    receipt = _load_digested_file(path, label="macro k30/k50 terminal receipt")
    terminal = _continuation_call(
        continuation_adapter,
        "terminal CHTC authentication failed",
        continuation_adapter.terminal_chtc_status,
        cached={},
    )
    eligible = list(decision_snapshot["eligible_execution_ids"])
    stopped = list(decision_snapshot["stop_at_k30_execution_ids"])
    if (
        receipt.get("schema") != MACRO_TERMINAL_SCHEMA
        or receipt.get("status") != MACRO_TERMINAL_STATUS
        or receipt.get("adapter_sha256")
        != EXPECTED_CONTINUATION_ADAPTER_SHA256
        or receipt.get("activation_manifest_sha256") != activation.get("sha256")
        or receipt.get("runtime_manifest_sha256") != runtime.get("sha256")
        or receipt.get("k30_runtime_manifest_sha256")
        != runtime.get("k30_runtime_manifest_sha256")
        or receipt.get("decision_status_sha256") != decision_snapshot.get("sha256")
        or receipt.get("terminal_chtc_status_sha256") != terminal.get("sha256")
        or receipt.get("conditional_execution_ids")
        != list(continuation_adapter.CONDITIONAL_EXECUTION_IDS)
        or receipt.get("terminal_chtc_k50_execution_ids")
        != list(continuation_adapter.TERMINAL_CHTC_EXECUTION_IDS)
        or receipt.get("eligible_k50_continuation_execution_ids") != eligible
        or receipt.get("stop_at_k30_execution_ids") != stopped
        or receipt.get("closed_k50_continuation_execution_ids") != eligible
        or receipt.get("all_k30_cells_closed") is not True
        or receipt.get("all_extension_required_cells_closed_at_k50") is not True
        or receipt.get("remaining_macro_execution_ids") != []
        or receipt.get("active_macro_execution_ids") != []
        or receipt.get("scientific_execution_performed_by_receipt") is not False
        or terminal.get("status")
        != "passed_all_three_authenticated_chtc_k50_terminals"
        or terminal.get("all_terminal_cells_authenticated") is not True
    ):
        raise UpdateError("macro k30/k50 terminal receipt drifted")
    return receipt


def authenticated_continuation_inventory(
    *,
    k30_inventory: Mapping[str, Any],
    activation_dir: Path = CONTINUATION_ACTIVATION_DIR,
    runtime_dir: Path = CONTINUATION_RUNTIME_DIR,
    macro_terminal_receipt: Path = MACRO_TERMINAL_RECEIPT,
    compile_costs: bool = False,
) -> dict[str, Any]:
    """Authenticate conditional k30-to-k50 state without launching work."""

    continuation_adapter = _load_continuation_adapter()
    snapshot_value = _continuation_call(
        continuation_adapter,
        "continuation decision authentication failed",
        continuation_adapter.decision_snapshot,
        cached={},
    )
    k30_completed = {
        str(execution_id): result
        for execution_id, result in k30_inventory.get("completed", {}).items()
        if execution_id in continuation_adapter.CONDITIONAL_EXECUTION_IDS
    }
    eligible, stopped, pending = _continuation_call(
        continuation_adapter,
        "continuation decision inventory validation failed",
        _validate_decision_snapshot,
        continuation_adapter,
        snapshot_value,
        k30_completed=k30_completed,
    )
    activation_present = activation_dir.exists() or activation_dir.is_symlink()
    runtime_present = runtime_dir.exists() or runtime_dir.is_symlink()
    macro_present = macro_terminal_receipt.exists() or macro_terminal_receipt.is_symlink()
    if not activation_present:
        if runtime_present or macro_present:
            raise UpdateError("continuation runtime/terminal exists without activation")
        return {
            "campaign_state": "not_activated",
            "decision_snapshot": snapshot_value,
            "eligible_execution_ids": list(eligible),
            "stop_at_k30_execution_ids": list(stopped),
            "pending_decision_execution_ids": list(pending),
            "closed_execution_ids": [],
            "cell_states": {
                **{execution_id: "pending_continuation_activation" for execution_id in eligible},
                **{execution_id: "closed_at_k30_plateau_stop" for execution_id in stopped},
            },
            "completed": {},
            "all_required_continuations_closed": not eligible and not pending,
            "macro_terminal_authenticated": False,
            "sources": {
                "expected_continuation_adapter_sha256": (
                    EXPECTED_CONTINUATION_ADAPTER_SHA256
                )
            },
        }
    worker = _continuation_call(
        continuation_adapter,
        "continuation worker load failed",
        continuation_adapter.k30._load_worker,
    )
    activation, _bundle = _continuation_call(
        continuation_adapter,
        "continuation activation authentication failed",
        continuation_adapter._validate_activation,
        worker,
        activation_dir,
    )
    if not runtime_present:
        if macro_present:
            raise UpdateError("macro terminal exists without continuation runtime")
        return {
            "campaign_state": "activated_runtime_pending",
            "decision_snapshot": snapshot_value,
            "eligible_execution_ids": list(eligible),
            "stop_at_k30_execution_ids": list(stopped),
            "pending_decision_execution_ids": list(pending),
            "closed_execution_ids": [],
            "cell_states": {
                **{execution_id: "pending_continuation_runtime" for execution_id in eligible},
                **{execution_id: "closed_at_k30_plateau_stop" for execution_id in stopped},
            },
            "completed": {},
            "all_required_continuations_closed": not eligible and not pending,
            "macro_terminal_authenticated": False,
            "sources": {
                "activation_manifest": {
                    **binding(activation_dir / "activation_manifest.json"),
                    "canonical_sha256": activation["sha256"],
                },
                "expected_continuation_adapter_sha256": (
                    EXPECTED_CONTINUATION_ADAPTER_SHA256
                ),
            },
        }
    runtime, validated_activation, _bundle = _continuation_call(
        continuation_adapter,
        "continuation runtime authentication failed",
        continuation_adapter._validate_runtime,
        worker,
        activation_dir=activation_dir,
        runtime_dir=runtime_dir,
    )
    if validated_activation != activation:
        raise UpdateError("continuation activation validation drifted")
    if (
        runtime.get("decision_status_sha256") != snapshot_value.get("sha256")
        or tuple(runtime.get("eligible_execution_ids", ())) != eligible
        or tuple(runtime.get("stop_at_k30_execution_ids", ())) != stopped
    ):
        raise UpdateError("continuation runtime decision snapshot drifted")
    completed: dict[str, dict[str, Any]] = {}
    states = {execution_id: "closed_at_k30_plateau_stop" for execution_id in stopped}
    for execution_id in eligible:
        authority, authority_runtime, _authority_bundle = _continuation_call(
            continuation_adapter,
            f"continuation authorization failed for {execution_id}",
            continuation_adapter._resume_authorization,
            worker,
            activation_dir=activation_dir,
            runtime_dir=runtime_dir,
            execution_id=execution_id,
        )
        if authority_runtime != runtime:
            raise UpdateError(f"continuation runtime authority drifted: {execution_id}")
        run_root = runtime_dir / "runs" / execution_id
        receipt_path = runtime_dir / "worker_receipts" / f"{execution_id}.json"
        quarantine = runtime_dir / "quarantine" / execution_id
        in_progress = runtime_dir / "in_progress" / execution_id
        if quarantine.exists() or quarantine.is_symlink():
            if quarantine.is_symlink() or not quarantine.is_dir():
                raise UpdateError(f"unsafe continuation quarantine: {execution_id}")
            quarantine_receipt = _load_digested_file(
                quarantine / "quarantine_receipt.json",
                label=f"continuation quarantine {execution_id}",
            )
            if (
                quarantine_receipt.get("schema")
                != continuation_adapter.QUARANTINE_SCHEMA
                or quarantine_receipt.get("execution_id") != execution_id
                or quarantine_receipt.get("scientific_execution_completed") is not True
                or quarantine_receipt.get("paper_evidence_adoption_authorized") is not False
            ):
                raise UpdateError(f"continuation quarantine drifted: {execution_id}")
            states[execution_id] = "quarantined_unclosed_k50_continuation"
            continue
        if in_progress.exists() or in_progress.is_symlink():
            if in_progress.is_symlink():
                raise UpdateError(f"unsafe continuation in-progress state: {execution_id}")
            states[execution_id] = "in_progress_unclosed_k50_continuation"
            continue
        present = (
            run_root.exists() or run_root.is_symlink(),
            receipt_path.exists() or receipt_path.is_symlink(),
        )
        if any(present) and not all(present):
            states[execution_id] = "published_partial_unclosed_k50_continuation"
            continue
        if not any(present):
            states[execution_id] = "pending_authenticated_k50_continuation"
            continue
        closed = _continuation_call(
            continuation_adapter,
            f"continuation closure failed for {execution_id}",
            continuation_adapter.closed_continuation_cell,
            runtime_dir=runtime_dir,
            execution_id=execution_id,
        )
        if not closed:
            states[execution_id] = "pending_authenticated_k50_continuation"
            continue
        completed[execution_id] = _continuation_call(
            continuation_adapter,
            f"continuation result authentication failed for {execution_id}",
            _continuation_summary_result,
            continuation_adapter=continuation_adapter,
            runtime_dir=runtime_dir,
            runtime=runtime,
            execution_id=execution_id,
            authority=authority,
            source_result=k30_completed[execution_id],
            compile_costs=compile_costs,
        )
        states[execution_id] = "closed_authenticated_k50_continuation"
    closed_ids = tuple(execution_id for execution_id in eligible if execution_id in completed)
    all_required = not pending and closed_ids == eligible
    macro_receipt = _continuation_call(
        continuation_adapter,
        "macro terminal authentication failed",
        _authenticate_macro_terminal_receipt,
        continuation_adapter=continuation_adapter,
        activation=activation,
        runtime=runtime,
        decision_snapshot=snapshot_value,
        path=macro_terminal_receipt,
    )
    if macro_receipt is not None and not all_required:
        raise UpdateError("macro terminal predates required continuation closure")
    sources: dict[str, Any] = {
        "activation_manifest": {
            **binding(activation_dir / "activation_manifest.json"),
            "canonical_sha256": activation["sha256"],
        },
        "runtime_manifest": {
            **binding(runtime_dir / "runtime_manifest.json"),
            "canonical_sha256": runtime["sha256"],
        },
        "decision_snapshot_canonical_sha256": snapshot_value["sha256"],
        "expected_continuation_adapter_sha256": EXPECTED_CONTINUATION_ADAPTER_SHA256,
    }
    if macro_receipt is not None:
        sources["macro_terminal_receipt"] = {
            **binding(macro_terminal_receipt),
            "canonical_sha256": macro_receipt["sha256"],
        }
    return {
        "campaign_state": "runtime_materialized",
        "decision_snapshot": snapshot_value,
        "eligible_execution_ids": list(eligible),
        "stop_at_k30_execution_ids": list(stopped),
        "pending_decision_execution_ids": list(pending),
        "closed_execution_ids": list(closed_ids),
        "cell_states": states,
        "completed": completed,
        "all_required_continuations_closed": all_required,
        "macro_terminal_authenticated": macro_receipt is not None,
        "sources": sources,
    }


def _merge_authenticated_continuations(
    completed: Mapping[str, Mapping[str, Any]],
    continuation: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Overlay only authenticated k=50 closures on their k=30 source rows."""

    merged = {
        str(execution_id): copy.deepcopy(dict(result))
        for execution_id, result in completed.items()
    }
    eligible = tuple(str(row) for row in continuation["eligible_execution_ids"])
    stopped = tuple(str(row) for row in continuation["stop_at_k30_execution_ids"])
    closed = tuple(str(row) for row in continuation["closed_execution_ids"])
    continued = continuation["completed"]
    if (
        len(eligible) != len(set(eligible))
        or len(stopped) != len(set(stopped))
        or set(eligible).intersection(stopped)
        or set(eligible).union(stopped) != set(merged)
        or len(closed) != len(set(closed))
        or not set(closed).issubset(eligible)
        or not isinstance(continued, Mapping)
        or set(continued) != set(closed)
    ):
        raise UpdateError("continuation decision inventory drifted")
    for execution_id in eligible:
        source = merged[execution_id]
        if (
            source.get("status")
            != "authenticated_local_k30_right_censored_partial"
            or source.get("effective_plateau_gate", {}).get("extension_decision")
            != "eligible_for_authenticated_resume_to_k50"
        ):
            raise UpdateError(f"continuation source gate drifted: {execution_id}")
        if execution_id not in continued:
            continue
        replacement = copy.deepcopy(dict(continued[execution_id]))
        if (
            replacement.get("status")
            != "completed_authenticated_local_k50_continuation"
            or replacement.get("execution_id") != execution_id
            or replacement.get("regime_id") != source.get("regime_id")
            or replacement.get("comparator_policy")
            != source.get("comparator_policy")
            or replacement.get("target_horizon") != 50
            or replacement.get("terminal", {}).get("k") != 50
        ):
            raise UpdateError(f"continuation result identity drifted: {execution_id}")
        merged[execution_id] = replacement
    return merged


def _continuation_evidence_revision(continuation: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {
            "campaign_state": continuation.get("campaign_state"),
            "eligible_execution_ids": continuation.get(
                "eligible_execution_ids", []
            ),
            "stop_at_k30_execution_ids": continuation.get(
                "stop_at_k30_execution_ids", []
            ),
            "pending_decision_execution_ids": continuation.get(
                "pending_decision_execution_ids", []
            ),
            "closed_execution_ids": continuation.get("closed_execution_ids", []),
            "cell_states": continuation.get("cell_states", {}),
            "all_required_continuations_closed": continuation.get(
                "all_required_continuations_closed"
            ),
            "macro_terminal_authenticated": continuation.get(
                "macro_terminal_authenticated"
            ),
            "sources": continuation.get("sources", {}),
        }
    )


def _display_horizon(
    reference_horizon: int,
    comparators: Mapping[str, Mapping[str, Any]],
) -> int:
    terminals = [
        int(result.get("terminal", {}).get("k", -1))
        for result in comparators.values()
    ]
    if reference_horizon <= 0 or any(round_index <= 0 for round_index in terminals):
        raise UpdateError("display horizon inventory drifted")
    return max([reference_horizon, *terminals])


def authenticated_effective_local_comparator_inventory(
    *,
    runtime_dir: Path = LOCAL_RUNTIME_DIR,
    activation_dir: Path = LOCAL_ACTIVATION_DIR,
    expected_adapter_sha256: str = EXPECTED_LOCAL_ADAPTER_SHA256,
    continuation_activation_dir: Path = CONTINUATION_ACTIVATION_DIR,
    continuation_runtime_dir: Path = CONTINUATION_RUNTIME_DIR,
    macro_terminal_receipt: Path = MACRO_TERMINAL_RECEIPT,
    compile_costs: bool = False,
) -> dict[str, Any]:
    """Return effective k30/k50 local evidence with authenticated revisions."""

    k30_inventory = authenticated_local_comparator_inventory(
        runtime_dir=runtime_dir,
        activation_dir=activation_dir,
        expected_adapter_sha256=expected_adapter_sha256,
        compile_costs=compile_costs,
    )
    continuation = authenticated_continuation_inventory(
        k30_inventory=k30_inventory,
        activation_dir=continuation_activation_dir,
        runtime_dir=continuation_runtime_dir,
        macro_terminal_receipt=macro_terminal_receipt,
        compile_costs=compile_costs,
    )
    completed = _merge_authenticated_continuations(
        k30_inventory["completed"], continuation
    )
    continuation_revision = _continuation_evidence_revision(continuation)
    revisions = {
        execution_id: str(result["evidence_revision"])
        for execution_id, result in completed.items()
    }
    cell_states = dict(k30_inventory["cell_states"])
    cell_states.update(continuation["cell_states"])
    sources = copy.deepcopy(k30_inventory["sources"])
    sources["continuation_campaign"] = copy.deepcopy(continuation["sources"])
    return {
        **copy.deepcopy(k30_inventory),
        "cell_states": cell_states,
        "completed": completed,
        "evidence_revisions": revisions,
        "continuation": continuation,
        "all_required_continuations_closed": continuation[
            "all_required_continuations_closed"
        ],
        "macro_terminal_authenticated": continuation[
            "macro_terminal_authenticated"
        ],
        "continuation_evidence_revision": continuation_revision,
        "sources": sources,
    }


def build_adapter(
    *,
    completed_regimes: tuple[str, ...] | None = None,
    include_local: bool = True,
    runtime_dir: Path = LOCAL_RUNTIME_DIR,
    activation_dir: Path = LOCAL_ACTIVATION_DIR,
    expected_adapter_sha256: str = EXPECTED_LOCAL_ADAPTER_SHA256,
) -> dict[str, Any]:
    """Build the dense Page-17 adapter from closed CHTC and local cells."""

    page16 = load(PAGE16_ADAPTER)
    package16 = load(PAGE16_PACKAGE / "package_manifest.json")
    receipt = load(SUBMISSION_RECEIPT)
    canary_receipt = load(CANARY_SUBMISSION_RECEIPT)
    for label, value in (
        ("Page-16 adapter", page16),
        ("Page-16 comparator package", package16),
    ):
        verify_self_digest(value, label=label)
    if page16.get("status") != "completed_6_of_6_mixed_horizon":
        raise UpdateError("Page-16 plateau reference is not complete")
    if tuple(package16.get("comparator_policies", ())) != EXPECTED_POLICIES:
        raise UpdateError("Page-16 comparator policies drifted")
    if len(package16.get("jobs", ())) != 12:
        raise UpdateError("expected twelve Page-16 comparator jobs")
    if (
        package16.get("run_class") != "diagnostic"
        or package16.get("plateau_reference_reused_not_rerun") is not True
    ):
        raise UpdateError("Page-16 package is outside the diagnostic reuse scope")
    clusters = receipt.get("clusters")
    if (
        receipt.get("schema")
        != "paper_i_ra_adapt_insertion_comparator_all24_submission_receipt_v1"
        or receipt.get("scientific_scope", {}).get("total_submitted_cells") != 24
        or not isinstance(clusters, list)
        or [row.get("cluster_id") for row in clusters]
        != [9644571, 9647385, 9647386]
        or canary_receipt.get("submission", {}).get("cluster_id") != 9644571
    ):
        raise UpdateError("all-24 submission authority drifted")
    page16_factory = next(row for row in clusters if row.get("cluster_id") == 9647386)
    if (
        page16_factory.get("submitted_procs") != 11
        or page16_factory.get("factory", {}).get("total_submit_procs") != 11
        or page16_factory.get("factory", {}).get("job_materialize_limit") != 1
        or page16_factory.get("factory", {}).get("job_materialize_max_idle") != 0
    ):
        raise UpdateError("Page-16 frozen factory contract drifted")

    jobs = completed_pages._package_jobs(PAGE16_PACKAGE, package16["sha256"])
    route_contracts = package16.get("route_contract_sha256_by_execution_id")
    if not isinstance(route_contracts, Mapping):
        raise UpdateError("Page-16 route-contract map is absent")
    jobs_by_cell: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    for regime in REGIME_ORDER:
        nph = 3 if regime in REGIME_ORDER[:3] else 7
        source_horizon = 50 if nph == 3 else 30
        expected_resources = {
            "request_cpus": 4,
            "request_memory_mb": 32_768 if nph == 3 else 49_152,
            "request_disk_mb": 61_440 if nph == 3 else 81_920,
            "max_runtime_seconds": 259_200,
        }
        for policy in EXPECTED_POLICIES:
            matches = [
                (path, job)
                for execution_id, (path, job) in jobs.items()
                if f"__{regime}__" in execution_id
                and execution_id.endswith(f"_{policy}")
            ]
            if len(matches) != 1:
                raise UpdateError(
                    f"Page-16 comparator job coverage drifted: {regime}/{policy}"
                )
            job_path, job = matches[0]
            execution_id = str(job["execution_id"])
            resources = job.get("resources")
            if (
                job.get("regime_id") != regime
                or job.get("nph") != nph
                or job.get("target_horizon") != source_horizon
                or job.get("candidate_representation") != "macro_generator_v1"
                or job.get("comparator_policy") != policy
                or job.get("runtime_insertion_mode")
                != (
                    "full_commutation_reduced"
                    if policy == EXPECTED_POLICIES[0]
                    else "append_only"
                )
                or job.get("route_contract_sha256")
                != route_contracts.get(execution_id)
                or job.get("route_contract_sha256")
                != COMPARATOR_ROUTE_CONTRACT_SHA256_BY_POLICY[policy]
                or not isinstance(resources, Mapping)
                or any(
                    resources.get(key) != value
                    for key, value in expected_resources.items()
                )
            ):
                raise UpdateError(
                    f"Page-16 comparator job identity drifted: {regime}/{policy}"
                )
            jobs_by_cell[(regime, policy)] = (job_path, job)

    archive_inventory = copy.deepcopy(COMPLETED_ARCHIVES)
    optional_sw_archive, optional_sw_receipt = _optional_sw_always_archive()
    if optional_sw_archive is not None:
        archive_inventory["strong_weak_u8"] = optional_sw_archive
    selected_regimes = (
        tuple(archive_inventory)
        if completed_regimes is None
        else tuple(completed_regimes)
    )
    unknown_regimes = set(selected_regimes) - set(archive_inventory)
    if unknown_regimes:
        raise UpdateError(
            f"unknown completed comparator regimes: {sorted(unknown_regimes)}"
        )
    completed_comparators: dict[str, dict[str, dict[str, Any]]] = {
        regime: {} for regime in REGIME_ORDER
    }
    for regime in selected_regimes:
        archive_spec = archive_inventory[regime]
        job_path, job = jobs_by_cell[(regime, EXPECTED_POLICIES[0])]
        result = page16_completed._close_page16_archive(
            path=RETRIEVED_DIR / str(archive_spec["filename"]),
            expected=archive_spec,
            job_path=job_path,
            job=job,
            cluster_id=int(archive_spec["cluster_id"]),
            expected_route_contract_sha256=COMPARATOR_ROUTE_CONTRACT_SHA256,
        )
        result.update(
            {
                "regime_id": regime,
                "comparator_policy": EXPECTED_POLICIES[0],
                "execution_origin": "CHTC",
            }
        )
        completed_comparators[regime][EXPECTED_POLICIES[0]] = result

    if include_local:
        local_inventory = local_comparator_inventory(
            runtime_dir=runtime_dir,
            activation_dir=activation_dir,
            expected_adapter_sha256=expected_adapter_sha256,
            package_manifest_sha256=package16["sha256"],
            jobs_by_cell=jobs_by_cell,
            compile_costs=True,
        )
        continuation_inventory = authenticated_continuation_inventory(
            k30_inventory=local_inventory,
            compile_costs=True,
        )
        effective_local_completed = _merge_authenticated_continuations(
            local_inventory["completed"], continuation_inventory
        )
    else:
        local_inventory = {
            "campaign_state": "excluded_by_caller",
            "execution_ids": [],
            "cell_states": {},
            "completed": {},
            "sources": {},
        }
        continuation_inventory = {
            "campaign_state": "excluded_by_caller",
            "eligible_execution_ids": [],
            "stop_at_k30_execution_ids": [],
            "pending_decision_execution_ids": [],
            "closed_execution_ids": [],
            "cell_states": {},
            "completed": {},
            "all_required_continuations_closed": False,
            "macro_terminal_authenticated": False,
            "sources": {},
        }
        effective_local_completed = {}
    for result in effective_local_completed.values():
        regime = str(result["regime_id"])
        policy = str(result["comparator_policy"])
        if policy in completed_comparators[regime]:
            raise UpdateError(f"duplicate comparator evidence: {regime}/{policy}")
        result["execution_origin"] = "local"
        completed_comparators[regime][policy] = result

    page16_references: list[dict[str, Any]] = []
    page16_by_regime = {row["regime_id"]: row for row in page16["cells"]}
    for regime in REGIME_ORDER:
        nph = 3 if regime in REGIME_ORDER[:3] else 7
        page16_horizon = 50 if nph == 3 else 30
        page16_route = page16_by_regime[regime]["page16_qiskit_route"]
        page16_points = [
            copy.deepcopy(point)
            for point in page16_route["points"]
            if int(point["k"]) <= page16_horizon
        ]
        if not page16_points or int(page16_points[-1]["k"]) != page16_horizon:
            raise UpdateError(
                f"Page-16 {regime}: plateau reference lacks k={page16_horizon}"
            )
        page16_references.append(
            {
                "regime_id": regime,
                "regime_label": REGIME_LABELS[regime],
                "nph": nph,
                "target_horizon": page16_horizon,
                "points": page16_points,
                "terminal": copy.deepcopy(page16_points[-1]),
                "status": page16_route["status"],
            }
        )

    local_state_by_cell: dict[tuple[str, str], str] = {}
    for (regime, policy), (_path, job) in jobs_by_cell.items():
        execution_id = str(job["execution_id"])
        local_state_by_cell[(regime, policy)] = str(
            continuation_inventory["cell_states"].get(
                execution_id,
                local_inventory["cell_states"].get(
                    execution_id,
                    (
                        "completed_authenticated_chtc_archive"
                        if regime in selected_regimes
                        and policy == EXPECTED_POLICIES[0]
                        else "pending_local_campaign"
                    ),
                ),
            )
        )

    def cell_label(regime: str, policy: str) -> str:
        result = completed_comparators[regime].get(policy)
        if result is not None:
            k = int(result["terminal"]["k"])
            if result["status"] == "authenticated_local_k30_right_censored_partial":
                continuation_state = continuation_inventory["cell_states"].get(
                    str(result["execution_id"]), "pending_authenticated_k50_continuation"
                )
                return (
                    f"authenticated partial / right-censored k={k}; "
                    f"k=50 {str(continuation_state).replace('_', ' ')}"
                )
            return f"complete / authenticated k={k}"
        if local_state_by_cell[(regime, policy)] == "published_partial_unclosed":
            return "closing locally / unclosed (not plotted)"
        state = local_state_by_cell[(regime, policy)]
        if state in {
            "pending_continuation_activation",
            "pending_continuation_runtime",
            "pending_authenticated_k50_continuation",
            "in_progress_unclosed_k50_continuation",
            "published_partial_unclosed_k50_continuation",
            "quarantined_unclosed_k50_continuation",
        }:
            return "authenticated partial / right-censored k=30; k=50 pending"
        return "pending / local k=30 campaign"

    rows: list[dict[str, Any]] = []
    for regime in REGIME_ORDER:
        nph = 3 if regime in REGIME_ORDER[:3] else 7
        horizon = 50 if nph == 3 else 30
        always_job = jobs_by_cell[(regime, EXPECTED_POLICIES[0])][1]
        append_job = jobs_by_cell[(regime, EXPECTED_POLICIES[1])][1]
        if always_job["resources"] != append_job["resources"]:
            raise UpdateError(f"Page-16 policy resources drifted: {regime}")
        rows.append(
            {
                "representation": "Page 16 / intact macro",
                "regime_id": regime,
                "regime_label": REGIME_LABELS[regime],
                "nph": nph,
                "target_horizon": horizon,
                "reference_target_horizon": horizon,
                "display_horizon": _display_horizon(
                    horizon, completed_comparators[regime]
                ),
                "plateau_reference": "complete / reused",
                "always_open": cell_label(regime, EXPECTED_POLICIES[0]),
                "append_only": cell_label(regime, EXPECTED_POLICIES[1]),
                "local_cell_states": {
                    policy: local_state_by_cell[(regime, policy)]
                    for policy in EXPECTED_POLICIES
                },
                "resources": copy.deepcopy(always_job["resources"]),
            }
        )

    flat_completed = [
        result
        for regime in REGIME_ORDER
        for result in completed_comparators[regime].values()
    ]
    authenticated_count = len(flat_completed)
    local_closed_count = sum(
        result.get("execution_origin") == "local" for result in flat_completed
    )
    local_complete_count = sum(
        result.get("status") == "completed_authenticated_local_k30"
        for result in flat_completed
    )
    local_partial_count = sum(
        result.get("status") == "authenticated_local_k30_right_censored_partial"
        for result in flat_completed
    )
    local_k50_count = sum(
        result.get("status")
        == "completed_authenticated_local_k50_continuation"
        for result in flat_completed
    )
    always_count = sum(
        EXPECTED_POLICIES[0] in completed_comparators[regime]
        for regime in REGIME_ORDER
    )
    append_count = sum(
        EXPECTED_POLICIES[1] in completed_comparators[regime]
        for regime in REGIME_ORDER
    )
    unclosed_count = sum(
        state == "published_partial_unclosed"
        for state in local_inventory["cell_states"].values()
    )
    pending_count = 12 - authenticated_count
    unsigned: dict[str, Any] = {
        "schema": "paper_i_ra_adapt_page16_insertion_comparator_progress_adapter_v4",
        "status": (
            f"provisional_page16_{authenticated_count}_authenticated_"
            f"{local_complete_count}_local_complete_"
            f"{local_partial_count}_local_right_censored"
        ),
        "page_ids": [PAGE17_ID],
        "submission_receipt_created_at_utc": receipt["created_at_utc"],
        "run_class": "diagnostic",
        "paper_evidence_adopted": False,
        "fresh_source_value_anchor": False,
        "plateau_reference_reused_not_rerun": True,
        "parameter_manifest": {
            "model": "Hubbard--Holstein L=2",
            "boundary": "open",
            "boson_encoding": "binary",
            "optimizer": "Powell",
            "error_metric": "same-cutoff absolute energy error",
            "representation": "Page 16 intact macro Phase-0 route",
            "comparator_policies": list(EXPECTED_POLICIES),
            "runtime_always_open_mode": "full_commutation_reduced",
            "page16_reference_weak_horizon": 50,
            "page16_reference_strong_horizon": 30,
            "local_comparator_operational_horizon": 30,
            "conditional_continuation_target_horizon": 50,
        },
        "campaign_counts": {
            "page16_comparator_jobs_planned": 12,
            "chtc_completed_authenticated": len(selected_regimes),
            "local_cells_closed_authenticated": local_closed_count,
            "local_cells_completed_at_k30": local_complete_count,
            "local_cells_right_censored_at_k30": local_partial_count,
            "local_cells_completed_at_k50": local_k50_count,
            "authenticated_curves_plotted": authenticated_count,
            "completed_validated": authenticated_count,
            "always_open_authenticated": always_count,
            "append_only_authenticated": append_count,
            "pending_or_unclosed": pending_count,
            "published_partial_unclosed_not_plotted": unclosed_count,
            "plateau_reference_jobs_rerun": 0,
        },
        "campaign_execution_state": {
            "page16_canary_cluster_id": 9644571,
            "page16_remaining_cluster_id": 9647386,
            "page16_remaining_chtc_factory": (
                optional_sw_receipt["remote_materialization_exclusion"][
                    "outcome"
                ]
                if optional_sw_receipt is not None
                else "frozen_no_further_materialization"
            ),
            "local_campaign_state": local_inventory["campaign_state"],
            "continuation_campaign_state": continuation_inventory["campaign_state"],
            "local_cell_states": copy.deepcopy(local_inventory["cell_states"]),
            "continuation_cell_states": copy.deepcopy(
                continuation_inventory["cell_states"]
            ),
            "local_closed_execution_ids": list(local_inventory["completed"]),
            "eligible_k50_continuation_execution_ids": list(
                continuation_inventory["eligible_execution_ids"]
            ),
            "closed_k50_continuation_execution_ids": list(
                continuation_inventory["closed_execution_ids"]
            ),
            "all_required_k50_continuations_closed": continuation_inventory[
                "all_required_continuations_closed"
            ],
            "macro_terminal_authenticated": continuation_inventory[
                "macro_terminal_authenticated"
            ],
            "continuation_evidence_revision": _continuation_evidence_revision(
                continuation_inventory
            ),
            "pending_or_unclosed_page16_cell_count": pending_count,
        },
        "matrix": rows,
        "completed_comparators": {
            regime: copy.deepcopy(completed_comparators[regime])
            for regime in REGIME_ORDER
            if completed_comparators[regime]
        },
        "reference_cells": page16_references,
        "sources": {
            "page16_adapter": {
                **binding(PAGE16_ADAPTER),
                "canonical_sha256": page16["sha256"],
            },
            "page16_package_manifest": {
                **binding(PAGE16_PACKAGE / "package_manifest.json"),
                "canonical_sha256": package16["sha256"],
            },
            "submission_receipt": binding(SUBMISSION_RECEIPT),
            "canary_submission_receipt": binding(CANARY_SUBMISSION_RECEIPT),
            "local_campaign": copy.deepcopy(local_inventory["sources"]),
            "continuation_campaign": copy.deepcopy(
                continuation_inventory["sources"]
            ),
        },
        "limitations": [
            (
                f"{authenticated_count}/12 comparator cells have authenticated "
                "curves; closed local k=30 cells are plotted immediately."
            ),
            (
                f"{local_partial_count} closed local cell(s) still require an "
                "authenticated k=50 continuation and remain partial/right-censored; "
                f"{local_k50_count} authenticated continuation(s) are complete at k=50."
            ),
            (
                "Unclosed or failed local attempts are status-only and are never "
                "rendered as completed curves."
            ),
            "Existing plateau references are reused and were not rerun.",
            "No paper-evidence adoption or insertion-policy conclusion is implied.",
        ],
    }
    if optional_sw_receipt is not None:
        unsigned["sources"]["sw_always_remote_materialization_exclusion_receipt"] = {
            **binding(SW_ALWAYS_CLOSURE_RECEIPT),
            "canonical_sha256": optional_sw_receipt["sha256"],
        }
    return {**unsigned, "sha256": _canonical_sha256(unsigned)}


def _style_table(table: Any, *, header_color: str = "#E8E8E8") -> None:
    table.auto_set_font_size(False)
    for (row, _), cell in table.get_celld().items():
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F7F7F7")


def render_page(adapter: Mapping[str, Any]) -> None:
    """Render the established single-page dense six-regime matrix."""

    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _atomic_json(ADAPTER_PATH, adapter)
    references = adapter["reference_cells"]
    completed = adapter["completed_comparators"]
    matrix_by_regime = {row["regime_id"]: row for row in adapter["matrix"]}
    policy_style = {
        EXPECTED_POLICIES[0]: {"color": ORANGE, "marker": "D", "label": "always-open"},
        EXPECTED_POLICIES[1]: {"color": MAGENTA, "marker": "o", "label": "append-only"},
    }

    def status_line(regime: str, policy: str) -> str:
        result = completed.get(regime, {}).get(policy)
        short = "always" if policy == EXPECTED_POLICIES[0] else "append"
        if result is None:
            state = matrix_by_regime[regime]["local_cell_states"][policy]
            if state == "published_partial_unclosed":
                return f"{short}: CLOSING (unclosed; not plotted)"
            return f"{short}: PENDING local k=30"
        k = int(result["terminal"]["k"])
        error = float(result["terminal"]["error"])
        if result["status"] == "authenticated_local_k30_right_censored_partial":
            return f"{short}: PARTIAL k={k}, |dE|={error:.2e}"
        origin = "CHTC" if result.get("execution_origin") == "CHTC" else "local"
        return f"{short}: COMPLETE {origin} k={k}, |dE|={error:.2e}"

    mpl.rcParams.update({"font.family": "serif", "font.size": 7.2})
    fig = plt.figure(figsize=(11, 8.5))
    grid = fig.add_gridspec(
        3,
        3,
        left=0.065,
        right=0.96,
        top=0.84,
        bottom=0.10,
        height_ratios=(1.0, 1.0, 0.56),
        hspace=0.40,
        wspace=0.25,
    )
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]
    for index, (axis, cell) in enumerate(zip(axes, references, strict=True)):
        regime = str(cell["regime_id"])
        points = cell["points"]
        horizon = int(cell["target_horizon"])
        display_horizon = int(matrix_by_regime[regime]["display_horizon"])
        axis.plot(
            [row["k"] for row in points],
            [max(float(row["error"]), PLOT_FLOOR) for row in points],
            color=BLUE,
            lw=1.55,
        )
        terminal = cell["terminal"]
        axis.scatter(
            [terminal["k"]],
            [max(float(terminal["error"]), PLOT_FLOOR)],
            color=BLUE,
            marker="s",
            s=20,
            zorder=5,
        )
        for policy in EXPECTED_POLICIES:
            result = completed.get(regime, {}).get(policy)
            if result is None:
                continue
            style = policy_style[policy]
            comparator_points = result["points"]
            axis.plot(
                [row["k"] for row in comparator_points],
                [max(float(row["error"]), PLOT_FLOOR) for row in comparator_points],
                color=style["color"],
                lw=1.75,
            )
            axis.scatter(
                [result["terminal"]["k"]],
                [max(float(result["terminal"]["error"]), PLOT_FLOOR)],
                color=style["color"],
                marker=style["marker"],
                s=27,
                zorder=6,
            )
        status = "\n".join(status_line(regime, policy) for policy in EXPECTED_POLICIES)
        axis.text(
            0.97,
            0.07,
            status,
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.7,
            color=GRAY,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84},
        )
        axis.set_yscale("log")
        axis.set_xlim(0, display_horizon)
        axis.grid(True, which="major", alpha=0.22, lw=0.5)
        axis.set_title(
            f"{cell['regime_label']} ($n_{{ph}}={cell['nph']}$; ref $k_{{max}}={horizon}$)",
            fontsize=8.1,
        )
        if index // 3 == 1:
            axis.set_xlabel("ADAPT controller round")
        if index % 3 == 0:
            axis.set_ylabel(r"same-cutoff $|\Delta E|$")

    counts = adapter["campaign_counts"]
    fig.suptitle(
        "Page 16 intact-macro insertion-policy comparators: authenticated progress",
        fontsize=11.2,
        fontweight="bold",
        y=0.982,
    )
    fig.text(
        0.5,
        0.948,
        (
            f"Authenticated curves: {counts['authenticated_curves_plotted']}/12 "
            f"(CHTC {counts['chtc_completed_authenticated']} + local "
            f"{counts['local_cells_closed_authenticated']}); local k=30 complete "
            f"{counts['local_cells_completed_at_k30']}, right-censored partial "
            f"{counts['local_cells_right_censored_at_k30']}, continued k=50 "
            f"{counts['local_cells_completed_at_k50']}"
        ),
        ha="center",
        color=RED,
        fontsize=7.2,
        fontweight="bold",
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=BLUE,
                lw=1.55,
                marker="s",
                markevery=[1],
                label="existing plateau reference (complete; reused)",
            ),
            Line2D(
                [0], [0], color=ORANGE, lw=1.75, marker="D", label="always-open comparator"
            ),
            Line2D(
                [0], [0], color=MAGENTA, lw=1.75, marker="o", label="append-only comparator"
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.918),
        ncol=3,
        frameon=False,
        fontsize=6.9,
    )

    table_axis = fig.add_subplot(grid[2, :])
    table_axis.axis("off")
    rows = []
    for cell in references:
        regime = str(cell["regime_id"])
        matrix_row = matrix_by_regime[regime]
        resources = matrix_row["resources"]
        rows.append(
            [
                cell["regime_label"],
                str(cell["target_horizon"]),
                f"{float(cell['terminal']['error']):.2e}",
                matrix_row["always_open"].replace("authenticated ", "auth. "),
                matrix_row["append_only"].replace("authenticated ", "auth. "),
                (
                    f"{resources['request_cpus']} / "
                    f"{resources['request_memory_mb'] // 1024} / "
                    f"{resources['request_disk_mb'] // 1024} / "
                    f"{resources['max_runtime_seconds'] // 3600}"
                ),
            ]
        )
    table = table_axis.table(
        cellText=rows,
        colLabels=[
            "Regime",
            "ref k",
            r"plateau $|\Delta E|$",
            "always-open",
            "append-only",
            "request: CPU / GiB / GiB / h",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=(0.17, 0.07, 0.14, 0.23, 0.21, 0.18),
    )
    _style_table(table)
    table.set_fontsize(5.8)
    table.scale(1.0, 0.92)
    for row_index, cell in enumerate(references, 1):
        regime = str(cell["regime_id"])
        for column, policy, color in (
            (3, EXPECTED_POLICIES[0], "#FFF0D9"),
            (4, EXPECTED_POLICIES[1], "#FCEAF5"),
        ):
            result = completed.get(regime, {}).get(policy)
            if result is not None:
                table[(row_index, column)].set_facecolor(color)
                table[(row_index, column)].set_text_props(weight="bold")

    sources = adapter["sources"]
    footer = (
        "Page 16 intact macro; HH L=2; open boundary; binary bosons; Powell; "
        f"package {sources['page16_package_manifest']['canonical_sha256'][:12]}...; "
        "same-cutoff exact reference; plateau source reused, not rerun. "
        "Local curves require closed manifests, receipts, gates, full artifact hashes, "
        "and source-locked continuation closure when extended."
    )
    fig.text(0.5, 0.050, footer, ha="center", fontsize=5.8, color=GRAY)
    fig.text(
        0.5,
        0.024,
        (
            "PROVISIONAL DIAGNOSTIC - k=30 continuation-eligible cells remain "
            "PARTIAL/right-censored until authenticated k=50 closure; unclosed attempts "
            "are not plotted; no "
            "paper-evidence adoption or insertion-policy conclusion is implied."
        ),
        ha="center",
        fontsize=6.1,
        color=RED,
        fontweight="bold",
    )
    completed_pages._save_page(fig, png_path=PAGE17_PNG, pdf_path=PAGE17_PDF)
    plt.close(fig)


def _page_content_sha256(page: Any) -> str:
    contents = page.get_contents()
    data = b"" if contents is None else contents.get_data()
    return hashlib.sha256(data).hexdigest()


def _with_report_mutation_lock(function: Any) -> Any:
    """Serialize Page-17/Page-18 replacements without changing run contracts."""

    def locked(*args: Any, **kwargs: Any) -> Any:
        REPORT_MUTATION_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REPORT_MUTATION_LOCK_PATH.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            return function(*args, **kwargs)

    return locked


@_with_report_mutation_lock
def append_or_replace_pages(
    adapter: Mapping[str, Any], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    from pypdf import PdfReader, PdfWriter

    current = binding(TARGET_PDF)
    outputs = provenance.get("outputs")
    layout = provenance.get("layout")
    declared = (
        outputs.get("partial_progress_pdf") if isinstance(outputs, Mapping) else None
    )
    if not isinstance(layout, Mapping) or not isinstance(declared, Mapping):
        raise UpdateError("report provenance is incomplete")
    page_count = int(layout.get("page_count", -1))
    supported = (
        current["sha256"] == declared.get("sha256")
        and current["size_bytes"] == declared.get("size_bytes")
        and layout.get("page_16") == PAGE16_ID
        and (
            page_count == 16
            or (
                page_count == 17
                and layout.get("page_17")
                in {PAGE17_ID, LEGACY_PAGE17_ID, OLD_PAGE17_ID}
            )
            or (
                page_count == 18
                and (
                    (
                        layout.get("page_17") == PAGE17_ID
                        and layout.get("page_18") == CURRENT_PAGE18_ID
                    )
                    or
                    (
                        layout.get("page_17") == LEGACY_PAGE17_ID
                        and layout.get("page_18") == LEGACY_PAGE18_ID
                    )
                    or (
                        layout.get("page_17") == OLD_PAGE17_ID
                        and layout.get("page_18") == OLD_PAGE18_ID
                    )
                )
            )
        )
    )
    if not supported:
        raise UpdateError(
            "target PDF/provenance is not a supported snapshot-page state"
        )
    original = PdfReader(str(TARGET_PDF), strict=False)
    page17 = PdfReader(str(PAGE17_PDF), strict=False)
    if len(original.pages) != page_count or len(page17.pages) != 1:
        raise UpdateError("snapshot update requires one one-page input")
    preserve_current_page18 = (
        page_count == 18
        and layout.get("page_17") == PAGE17_ID
        and layout.get("page_18") == CURRENT_PAGE18_ID
    )
    preserved_hashes = [_page_content_sha256(row) for row in original.pages[:16]]
    preserved_page18_hash = (
        _page_content_sha256(original.pages[17])
        if preserve_current_page18
        else None
    )
    writer = PdfWriter()
    for row in original.pages[:16]:
        writer.add_page(row)
    writer.add_page(page17.pages[0])
    if preserve_current_page18:
        writer.add_page(original.pages[17])

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
        expected_page_count = 18 if preserve_current_page18 else 17
        if len(combined.pages) != expected_page_count:
            raise UpdateError(
                f"combined report must contain exactly {expected_page_count} pages"
            )
        if [
            _page_content_sha256(row) for row in combined.pages[:16]
        ] != preserved_hashes:
            raise UpdateError("snapshot update changed a preserved page")
        if (
            preserve_current_page18
            and _page_content_sha256(combined.pages[17]) != preserved_page18_hash
        ):
            raise UpdateError("Page-17 refresh changed the current Page 18")

        updated = copy.deepcopy(dict(provenance))
        updated["layout"]["page_17"] = PAGE17_ID
        if preserve_current_page18:
            updated["layout"]["page_18"] = CURRENT_PAGE18_ID
            updated["layout"]["page_count"] = 18
        else:
            updated["layout"].pop("page_18", None)
            updated["layout"]["page_count"] = 17
        updated.pop("phase0_insertion_comparator_live_snapshot", None)
        updated["phase0_insertion_comparator_snapshot"] = {
            "schema": "paper_i_ra_adapt_page16_insertion_comparator_progress_report_v5",
            "status": adapter["status"],
            "page_ids": [PAGE17_ID],
            "run_class": "diagnostic",
            "paper_evidence_adopted": False,
            "adapter": {**binding(ADAPTER_PATH), "canonical_sha256": adapter["sha256"]},
            "campaign_counts": copy.deepcopy(adapter["campaign_counts"]),
            "campaign_execution_state": copy.deepcopy(
                adapter["campaign_execution_state"]
            ),
            "matrix": copy.deepcopy(adapter["matrix"]),
            "completed_comparators": copy.deepcopy(adapter["completed_comparators"]),
            "sources": copy.deepcopy(adapter["sources"]),
            "limitations": copy.deepcopy(adapter["limitations"]),
            "outputs": {
                "page17_pdf": binding(PAGE17_PDF),
                "page17_png": binding(PAGE17_PNG),
            },
        }
        updated["outputs"]["insertion_comparator_snapshot_adapter"] = {
            **binding(ADAPTER_PATH),
            "canonical_sha256": adapter["sha256"],
        }
        updated["outputs"]["insertion_comparator_snapshot_page17_pdf"] = binding(
            PAGE17_PDF
        )
        updated["outputs"]["insertion_comparator_snapshot_page17_png"] = binding(
            PAGE17_PNG
        )
        if not preserve_current_page18:
            updated["outputs"].pop("insertion_comparator_snapshot_page18_pdf", None)
            updated["outputs"].pop("insertion_comparator_snapshot_page18_png", None)
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
        "page_count": 18 if preserve_current_page18 else 17,
        "preserved_page_count": 17 if preserve_current_page18 else 16,
        "completed_comparator_count": adapter["campaign_counts"][
            "authenticated_curves_plotted"
        ],
        "completed_local_comparator_count": adapter["campaign_counts"][
            "local_cells_closed_authenticated"
        ],
        "completed_local_k30_count": adapter["campaign_counts"][
            "local_cells_completed_at_k30"
        ],
        "right_censored_local_k30_count": adapter["campaign_counts"][
            "local_cells_right_censored_at_k30"
        ],
        "pdf": binding(TARGET_PDF),
        "provenance": binding(TARGET_PROVENANCE),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, default=LOCAL_RUNTIME_DIR)
    parser.add_argument("--activation-dir", type=Path, default=LOCAL_ACTIVATION_DIR)
    parser.add_argument(
        "--expected-local-adapter-sha256",
        default=EXPECTED_LOCAL_ADAPTER_SHA256,
    )
    args = parser.parse_args()
    provenance = load(TARGET_PROVENANCE)
    optional_sw_archive, _optional_sw_receipt = _optional_sw_always_archive()
    archive_inventory = copy.deepcopy(COMPLETED_ARCHIVES)
    if optional_sw_archive is not None:
        archive_inventory["strong_weak_u8"] = optional_sw_archive
    available = tuple(
        regime
        for regime, spec in archive_inventory.items()
        if (RETRIEVED_DIR / str(spec["filename"])).is_file()
    )
    missing = tuple(regime for regime in archive_inventory if regime not in available)
    if missing:
        raise UpdateError(
            "required completed comparator archives are unavailable: "
            + ", ".join(missing)
        )
    adapter = build_adapter(
        completed_regimes=available,
        runtime_dir=args.runtime_dir.resolve(),
        activation_dir=args.activation_dir.resolve(),
        expected_adapter_sha256=args.expected_local_adapter_sha256,
    )
    render_page(adapter)
    result = append_or_replace_pages(adapter, provenance)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
