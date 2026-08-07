#!/usr/bin/env python3
"""Shared fail-closed contract for the minimal Paper-I Study-1 package.

This module is deliberately data/validation only.  It neither creates an
authorization receipt nor executes a scientific cell.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


PACKAGE_ID = "paper_i_ra_adapt_study1_minimal_20260728_v3_chtc"
PACKAGE_RELATIVE_ROOT = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "study1_minimal_20260728_v1_chtc"
)
CAMPAIGN_ID = "paper_i_ra_adapt_stationarity_comparison_v1"
RUN_CLASS = "candidate"
EXECUTION_TARGET = "chtc"
AUTHORIZATION_SCHEMA = (
    "paper_i_ra_adapt_study1_execution_authorization_v3"
)
EXECUTION_PLAN_SCHEMA = "paper_i_ra_adapt_study1_execution_plan_v3"
JOB_SPEC_SCHEMA = "paper_i_ra_adapt_study1_job_spec_v3"
PACKAGE_MANIFEST_SCHEMA = "paper_i_ra_adapt_study1_package_manifest_v3"
WORKER_RECEIPT_SCHEMA = "paper_i_ra_adapt_study1_worker_receipt_v3"
PACKAGE_CONTROL_PLANE_SCHEMA = (
    "paper_i_ra_adapt_study1_package_control_plane_v3"
)
ATTEMPT_SELECTION_SCHEMA = (
    "paper_i_ra_adapt_study1_attempt_selection_v3"
)
COMPLETION_MATRIX_SCHEMA = (
    "paper_i_ra_adapt_study1_completion_matrix_v3"
)
G11_JOB_DIAGNOSTIC_SCHEMA = (
    "paper_i_ra_adapt_study1_g11_job_diagnostic_v3"
)
FETCH_VALIDATION_SCHEMA = (
    "paper_i_ra_adapt_study1_fetched_validation_v3"
)
SHARED_APPEND_EQUIVALENCE_SCHEMA = (
    "paper_i_ra_adapt_shared_append_equivalence_v3"
)
SCIENTIFIC_PREFLIGHT_SCHEMA = (
    "paper_i_ra_adapt_study1_scientific_preflight_v3"
)
SUBMISSION_PREFLIGHT_SCHEMA = (
    "paper_i_ra_adapt_study1_submission_preflight_v3"
)
SUBMISSION_PREFLIGHT_OVERLAY_SCHEMA = (
    "paper_i_ra_adapt_study1_submission_preflight_overlay_contract_v1"
)

V8_REVISION = "ra_adapt_unification_post_refactor_v8"
V8_RELATIVE_ROOT = (
    "chtc/paper_i_ra_adapt_repair_20260727/bundles/materializations/"
    f"{V8_REVISION}"
)
V8_FINAL_RECEIPT_NAME = "final_materialization_receipt.json"
OBJECTIVE_GATE_AUTHORITY_NAME = (
    "study1_objective_gate_authority_receipt.json"
)
SCIENTIFIC_PREFLIGHT_RELATIVE = (
    "authority/scientific_preflight_receipt.json"
)
SUBMISSION_PREFLIGHT_RELATIVE = (
    "authority/submission_preflight_receipt.json"
)
STATIONARY_BUNDLE_ID = "ra_repair_stationary_late_v1"
MEASURED_BUNDLE_ID = "ra_repair_measured_late_v1"
BUNDLE_IDS = (STATIONARY_BUNDLE_ID, MEASURED_BUNDLE_ID)
PACKAGE_CONTROL_PLANE_FILES = (
    "build_attempt_selection.py",
    "build_package.py",
    "package_contract.py",
    "run_cell.py",
    "execute_source_locked_job.sh",
    "run_scientific_preflight_smokes.py",
    "publish_staged_package.sh",
    "stage_transferred_executable.py",
    "submit.sub",
    "validate_package.py",
    "validate_fetched.py",
    "link_shared_append.py",
    "objective_gates.py",
)

VALIDATION_REGIMES = ("strong_weak_u8", "strong_strong_u8")
VALIDATION_ROUTES = (
    "append_macro",
    "ra_macro_append_only",
    "ra_macro_plateau",
    "ra_macro_always",
    "singleton_plateau",
)
MEASURED_DIRECT_ROUTES = (
    "ra_macro_append_only",
    "ra_macro_plateau",
    "ra_macro_always",
    "singleton_plateau",
)
EXPECTED_ARTIFACT_ROLES = (
    "execution_manifest",
    "checkpoint",
    "estimator_ledger",
    "result",
    "summary",
)
EXPECTED_ARTIFACT_SUFFIXES = {
    "execution_manifest": "execution_manifest.json",
    "checkpoint": "checkpoints/current.json",
    "estimator_ledger": "result/estimator_ledger.json",
    "result": "result/result.json",
    "summary": "summary/summary.json",
}

REMOTE_IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
REMOTE_IMAGE_PATH = "chtc/phase3_optuna/image.sif"
REQUEST_CPUS = 4
MAX_RUNTIME_SECONDS = 72 * 60 * 60
BATCH_NAME = "paper-i-ra-adapt-study1-minimal-20260728-v2"
P4_EXECUTION_ID = (
    "ra_repair_stationary_late_v1__validation__strong_weak_u8__"
    "nph3__append_macro"
)
REMOTE_REPOSITORY_ROOT_BASENAME = "Holstein_phase3_optuna_chtc"

# These are deliberately conservative envelopes derived from the completed
# Paper-I nph=3 macro and guarded-singleton CHTC families.  The singleton
# strong-strong row retains the largest observed guarded-singleton envelope.
RESOURCE_ENVELOPES_MB = {
    ("strong_weak_u8", "macro"): (49_152, 61_440),
    ("strong_weak_u8", "singleton"): (57_344, 61_440),
    ("strong_strong_u8", "macro"): (65_536, 81_920),
    ("strong_strong_u8", "singleton"): (90_112, 98_304),
}

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
MACOS_UF_COMPRESSED = 0x00000020
MACOS_SF_DATALESS = 0x40000000


class PackageContractError(ValueError):
    """Raised when immutable package authority or content drifts."""


def repo_root_from_script(script_path: str | Path) -> Path:
    """Resolve this package's repository root without consulting cwd."""

    return Path(script_path).resolve().parents[3]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_control_plane_receipt(
    package_dir: str | Path,
) -> dict[str, Any]:
    """Hash the complete pre-build package code/wrapper/validator surface."""

    root = Path(package_dir).resolve()
    rows: list[dict[str, Any]] = []
    for relative in PACKAGE_CONTROL_PLANE_FILES:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise PackageContractError(
                f"Package control-plane member is unavailable: {path}"
            )
        rows.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return digested(
        {
            "schema": PACKAGE_CONTROL_PLANE_SCHEMA,
            "package_id": PACKAGE_ID,
            "files": rows,
            "file_count": len(rows),
            "all_files_verified": True,
        }
    )


def stage_packaged_runtime_tree(
    *,
    package_dir: str | Path,
    source_root: str | Path,
    job_relative: str,
) -> Path:
    """Mirror the CHTC-preserved package control plane into a temp root."""

    package = Path(package_dir).resolve()
    root = Path(source_root).resolve()
    target_package = root / PACKAGE_RELATIVE_ROOT
    relatives = (
        *PACKAGE_CONTROL_PLANE_FILES,
        "package_manifest.json",
        "execution_plan.json",
        "authority/execution_authorization_receipt.json",
        "authority/v8_final_materialization_receipt.json",
        "authority/study1_objective_gate_authority_receipt.json",
        SCIENTIFIC_PREFLIGHT_RELATIVE,
        safe_relative_path(job_relative, label="runtime job spec").as_posix(),
    )
    if len(relatives) != len(set(relatives)):
        raise PackageContractError(
            "Packaged runtime staging paths contain duplicates."
        )
    for relative in relatives:
        source = package / safe_relative_path(
            relative, label="packaged runtime source"
        )
        destination = target_package / safe_relative_path(
            relative, label="packaged runtime destination"
        )
        if not source.is_file() or source.is_symlink():
            raise PackageContractError(
                f"Packaged runtime source is unavailable: {source}"
            )
        if destination.exists():
            raise PackageContractError(
                f"Packaged runtime staging collision: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return target_package


def require_sha256(value: Any, *, label: str) -> str:
    resolved = str(value)
    if SHA256_RE.fullmatch(resolved) is None:
        raise PackageContractError(f"{label} is not a lowercase SHA-256.")
    return resolved


def require_safe_id(value: Any, *, label: str) -> str:
    resolved = str(value)
    if SAFE_ID_RE.fullmatch(resolved) is None:
        raise PackageContractError(f"{label} is not a safe identifier.")
    return resolved


def safe_relative_path(value: Any, *, label: str) -> PurePosixPath:
    raw = str(value)
    path = PurePosixPath(raw)
    if (
        not raw
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or "\\" in raw
        or any(not part for part in path.parts)
    ):
        raise PackageContractError(f"{label} is not a safe relative path.")
    return path


def load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PackageContractError(f"Missing {label}: {source}") from exc
    except json.JSONDecodeError as exc:
        raise PackageContractError(f"Invalid JSON in {label}: {source}") from exc
    if not isinstance(value, dict):
        raise PackageContractError(f"{label} must be a JSON object: {source}")
    return value


def verify_self_digest(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> str:
    expected = require_sha256(payload.get("sha256"), label=f"{label}.sha256")
    unsigned = dict(payload)
    del unsigned["sha256"]
    actual = canonical_sha256(unsigned)
    if actual != expected:
        raise PackageContractError(
            f"{label} canonical digest mismatch: {actual} != {expected}."
        )
    return expected


def digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    if "sha256" in value:
        raise PackageContractError("Cannot digest a payload that already has sha256.")
    value["sha256"] = canonical_sha256(value)
    return value


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write canonical JSON through a same-directory atomic replacement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise PackageContractError(
            f"Refusing to overwrite stale atomic temporary: {temporary}"
        )
    data = canonical_json_bytes(dict(payload)) + b"\n"
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def validation_cell_id(regime_id: str, route_id: str) -> str:
    return f"validation__{regime_id}__nph3__{route_id}"


def logical_cell_keys() -> tuple[str, ...]:
    return tuple(
        f"{bundle_id}::{validation_cell_id(regime_id, route_id)}"
        for bundle_id in BUNDLE_IDS
        for regime_id in VALIDATION_REGIMES
        for route_id in VALIDATION_ROUTES
    )


def direct_execution_rows() -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    for bundle_id in BUNDLE_IDS:
        routes = (
            VALIDATION_ROUTES
            if bundle_id == STATIONARY_BUNDLE_ID
            else MEASURED_DIRECT_ROUTES
        )
        for regime_id in VALIDATION_REGIMES:
            for route_id in routes:
                cell_id = validation_cell_id(regime_id, route_id)
                execution_id = f"{bundle_id}__{cell_id}"
                rows.append(
                    {
                        "execution_id": execution_id,
                        "bundle_id": bundle_id,
                        "cell_id": cell_id,
                        "regime_id": regime_id,
                        "route_id": route_id,
                    }
                )
    if len(rows) != 18 or len({row["execution_id"] for row in rows}) != 18:
        raise AssertionError("The minimal Study-1 direct matrix must be 18 rows.")
    return tuple(rows)


def direct_execution_ids() -> tuple[str, ...]:
    return tuple(row["execution_id"] for row in direct_execution_rows())


def objective_gate_diagnostic_contract(
    *,
    bundle_id: str,
    regime_id: str,
    route_id: str,
) -> dict[str, Any]:
    """Return the fixed, outcome-independent in-job replay diagnostic plan."""

    if bundle_id not in BUNDLE_IDS:
        raise PackageContractError(f"Unknown diagnostic bundle: {bundle_id}")
    if regime_id not in VALIDATION_REGIMES:
        raise PackageContractError(f"Unknown diagnostic regime: {regime_id}")
    if route_id not in VALIDATION_ROUTES:
        raise PackageContractError(f"Unknown diagnostic route: {route_id}")
    method = "append_adapt" if route_id == "append_macro" else "ra_adapt"
    selected = route_id == "singleton_plateau" or (
        bundle_id == STATIONARY_BUNDLE_ID and route_id == "append_macro"
    )
    purposes: list[str] = []
    if selected:
        purposes.append("g11_method_regime_coverage")
    if route_id == "singleton_plateau":
        purposes.append("g13_same_problem_deterministic_replay")
    return {
        "schema": G11_JOB_DIAGNOSTIC_SCHEMA,
        "selected": selected,
        "method_family": method,
        "regime_id": regime_id,
        "bounded_controller_round": 1 if selected else None,
        "independent_fresh_execution_required": selected,
        "authenticated_ra_continuation_required": (
            selected and method == "ra_adapt"
        ),
        "ra_fresh_leg_maximum_controller_rounds": (
            2 if selected and method == "ra_adapt" else None
        ),
        "ra_resumed_maximum_controller_rounds": (
            3 if selected and method == "ra_adapt" else None
        ),
        "append_resume_boundary": (
            "authenticated_reconstruction_only_v1"
            if selected and method == "append_adapt"
            else None
        ),
        "purposes": purposes,
    }


def shared_append_rows() -> tuple[dict[str, str], ...]:
    rows = []
    for regime_id in VALIDATION_REGIMES:
        cell_id = validation_cell_id(regime_id, "append_macro")
        rows.append(
            {
                "regime_id": regime_id,
                "cell_id": cell_id,
                "canonical_bundle_id": STATIONARY_BUNDLE_ID,
                "reference_bundle_id": MEASURED_BUNDLE_ID,
                "canonical_execution_id": (
                    f"{STATIONARY_BUNDLE_ID}__{cell_id}"
                ),
                "reference_logical_key": f"{MEASURED_BUNDLE_ID}::{cell_id}",
            }
        )
    return tuple(rows)


def expected_artifact_path(cell_id: str, role: str) -> str:
    try:
        suffix = EXPECTED_ARTIFACT_SUFFIXES[role]
    except KeyError as exc:
        raise PackageContractError(f"Unknown artifact role: {role}") from exc
    return f"runs/{cell_id}/{suffix}"


def _load_digested_file(path: Path, *, label: str) -> dict[str, Any]:
    payload = load_json_object(path, label=label)
    verify_self_digest(payload, label=label)
    return payload


def _file_binding(
    path: Path,
    *,
    payload: Mapping[str, Any],
    display_path: str | None = None,
) -> dict[str, Any]:
    return {
        "path": path.as_posix() if display_path is None else display_path,
        "canonical_sha256": str(payload["sha256"]),
        "file_sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _manifest_validation_rows(
    manifest: Mapping[str, Any],
    *,
    bundle_id: str,
) -> dict[str, Mapping[str, Any]]:
    raw_rows = manifest.get("cells")
    if not isinstance(raw_rows, list):
        raise PackageContractError(f"{bundle_id} manifest has no ordered cells.")
    rows: dict[str, Mapping[str, Any]] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise PackageContractError(f"{bundle_id} manifest has a non-object cell.")
        cell_id = str(raw.get("cell_id", ""))
        if cell_id in rows:
            raise PackageContractError(f"Duplicate bundle cell: {bundle_id}::{cell_id}")
        if str(raw.get("stage", "")) == "validation":
            rows[cell_id] = raw
    expected = {
        validation_cell_id(regime_id, route_id)
        for regime_id in VALIDATION_REGIMES
        for route_id in VALIDATION_ROUTES
    }
    if set(rows) != expected:
        raise PackageContractError(
            f"{bundle_id} validation matrix drifted: "
            f"missing={sorted(expected - set(rows))}, "
            f"extra={sorted(set(rows) - expected)}"
        )
    return rows


def _validate_dedupe(
    dedupe: Mapping[str, Any],
) -> str:
    digest = verify_self_digest(dedupe, label="Study-1 dedupe contract")
    if (
        dedupe.get("schema")
        != "paper_i_ra_adapt_study1_execution_dedupe_v1"
        or int(dedupe.get("materialized_validation_cell_count", -1)) != 20
        or int(dedupe.get("unique_validation_execution_count", -1)) != 18
        or int(dedupe.get("shared_execution_savings", -1)) != 2
        or dedupe.get("execution_authorized") is not False
        or dedupe.get("submitted") is not False
    ):
        raise PackageContractError("Study-1 dedupe count/state contract drifted.")
    groups = dedupe.get("groups")
    if not isinstance(groups, list) or len(groups) != 2:
        raise PackageContractError("Study-1 dedupe must contain two Append groups.")
    by_cell = {}
    for group in groups:
        if not isinstance(group, Mapping):
            raise PackageContractError("Study-1 dedupe group is not an object.")
        by_cell[str(group.get("scientific_cell_id", ""))] = group
    for shared in shared_append_rows():
        group = by_cell.get(shared["cell_id"])
        if not isinstance(group, Mapping):
            raise PackageContractError(
                f"Missing Append dedupe group for {shared['cell_id']}."
            )
        canonical = group.get("canonical_execution")
        reference = group.get("shared_result_reference")
        if (
            not isinstance(canonical, Mapping)
            or canonical.get("bundle_id") != STATIONARY_BUNDLE_ID
            or canonical.get("cell_id") != shared["cell_id"]
            or not isinstance(reference, Mapping)
            or reference.get("bundle_id") != MEASURED_BUNDLE_ID
            or reference.get("cell_id") != shared["cell_id"]
            or reference.get("fulfillment_kind")
            != "shared_result_reference_v1"
        ):
            raise PackageContractError(
                f"Append dedupe identity drifted for {shared['cell_id']}."
            )
    projection = dedupe.get("scientific_equivalence_projection")
    if (
        not isinstance(projection, Mapping)
        or not isinstance(projection.get("required_equal_fields"), list)
        or not projection["required_equal_fields"]
    ):
        raise PackageContractError(
            "Study-1 dedupe lacks required scientific-equivalence fields."
        )
    return digest


def _validate_artifact_contract(
    *,
    bundle_id: str,
    cell_id: str,
    route_id: str,
    expected_cell: Mapping[str, Any],
    template: Mapping[str, Any],
    dedupe_sha256: str,
) -> None:
    expected_entrypoint = (
        "pipelines.static_adapt.ra_adapt.run_append_adapt"
        if route_id == "append_macro"
        else "pipelines.static_adapt.ra_adapt.run_ra_adapt"
    )
    if template.get("execution_entrypoint") != expected_entrypoint:
        raise PackageContractError(
            f"Execution entrypoint drifted for {bundle_id}::{cell_id}."
        )
    fulfillment = expected_cell.get("execution_fulfillment")
    template_fulfillment = template.get("execution_fulfillment")
    if not isinstance(fulfillment, Mapping) or fulfillment != template_fulfillment:
        raise PackageContractError(
            f"Execution fulfillment drifted for {bundle_id}::{cell_id}."
        )
    if fulfillment.get("dedupe_contract_sha256") != dedupe_sha256:
        raise PackageContractError(
            f"Dedupe hash drifted for {bundle_id}::{cell_id}."
        )
    reference = bundle_id == MEASURED_BUNDLE_ID and route_id == "append_macro"
    expected_kind = (
        "shared_result_reference_v1"
        if reference
        else "canonical_shared_execution_v1"
        if bundle_id == STATIONARY_BUNDLE_ID and route_id == "append_macro"
        else "direct_execution_v1"
    )
    if fulfillment.get("fulfillment_kind") != expected_kind:
        raise PackageContractError(
            f"Wrong fulfillment kind for {bundle_id}::{cell_id}."
        )
    artifacts = expected_cell.get("expected_run_artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        EXPECTED_ARTIFACT_ROLES
    ):
        raise PackageContractError(
            f"Expected-artifact role set drifted for {bundle_id}::{cell_id}."
        )
    for role in EXPECTED_ARTIFACT_ROLES:
        record = artifacts.get(role)
        if (
            not isinstance(record, Mapping)
            or record.get("path") != expected_artifact_path(cell_id, role)
            or record.get("required") is not True
            or record.get("fulfillment_kind") != expected_kind
            or record.get("direct_file_required") is not (not reference)
            or record.get("reference_receipt_required") is not reference
        ):
            raise PackageContractError(
                f"Artifact fulfillment drifted for "
                f"{bundle_id}::{cell_id}:{role}."
            )
    template_outputs = template.get("output_artifacts")
    template_roles = {
        "checkpoint",
        "estimator_ledger",
        "result",
        "summary",
    }
    if not isinstance(template_outputs, Mapping) or set(
        template_outputs
    ) != template_roles:
        raise PackageContractError(
            f"Execution-template output roles drifted for "
            f"{bundle_id}::{cell_id}."
        )
    for role in template_roles:
        row = template_outputs.get(role)
        if (
            not isinstance(row, Mapping)
            or row.get("path") != expected_artifact_path(cell_id, role)
            or row.get("sha256") is not None
            or row.get("status") != "not_produced"
        ):
            raise PackageContractError(
                f"Execution-template output drifted for "
                f"{bundle_id}::{cell_id}:{role}."
            )


def validate_v8_authority(
    repo_root: str | Path,
    *,
    v8_root: str | Path | None = None,
    final_receipt_path: str | Path | None = None,
    objective_gate_authority_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate immutable v8, both bundles, and objective-gate authority."""

    root = Path(repo_root).resolve()
    revision_root = (
        Path(v8_root).resolve()
        if v8_root is not None
        else (root / V8_RELATIVE_ROOT).resolve()
    )
    try:
        revision_root.relative_to(root)
    except ValueError as exc:
        raise PackageContractError(
            "The immutable v8 bundle root must remain inside the active "
            "repository/source-archive root."
        ) from exc
    final_path = (
        Path(final_receipt_path).resolve()
        if final_receipt_path is not None
        else revision_root / V8_FINAL_RECEIPT_NAME
    )
    if not final_path.is_file():
        raise PackageContractError(
            "Immutable v8 final receipt is not available yet; wait for "
            f"{final_path} and do not build from v7 or a partial v8."
        )
    final = _load_digested_file(final_path, label="v8 final materialization receipt")
    if (
        final.get("schema") != "ra_adapt_final_materialization_receipt_v1"
        or final.get("materialization_revision") != "v8"
        or final.get("status") != "passed"
        or final.get("execution_authorized") is not False
        or final.get("submission_state") != "not_submitted"
        or final.get("submitted") is not False
        or final.get("stationarity_winner_selected") is not False
    ):
        raise PackageContractError(
            "The v8 final receipt is not the passed, immutable, non-authorizing "
            "Study-1 materialization."
        )

    raw_final_bundles = final.get("bundles")
    if (
        not isinstance(raw_final_bundles, list)
        or any(not isinstance(row, Mapping) for row in raw_final_bundles)
    ):
        raise PackageContractError(
            "v8 final receipt does not contain flat bundle summaries."
        )
    final_bundles = {
        str(row.get("bundle_id", "")): row for row in raw_final_bundles
    }
    if (
        len(final_bundles) != len(raw_final_bundles)
        or set(final_bundles) != set(BUNDLE_IDS)
    ):
        raise PackageContractError("v8 final receipt does not bind exactly two bundles.")

    bundle_bindings: dict[str, Any] = {}
    dedupe_sha256: str | None = None
    source_inventory: dict[str, Any] | None = None
    for bundle_id in BUNDLE_IDS:
        bundle_dir = revision_root / bundle_id
        if not bundle_dir.is_dir() or bundle_dir.is_symlink():
            raise PackageContractError(f"Missing immutable v8 bundle: {bundle_dir}")
        manifest_path = bundle_dir / "bundle_manifest.json"
        source_locks_path = bundle_dir / "source_locks.json"
        expected_path = bundle_dir / "expected_artifacts.json"
        validation_path = bundle_dir / "validation_report.json"
        manifest = _load_digested_file(
            manifest_path, label=f"{bundle_id} bundle manifest"
        )
        source_locks = _load_digested_file(
            source_locks_path, label=f"{bundle_id} source locks"
        )
        expected = _load_digested_file(
            expected_path, label=f"{bundle_id} expected artifacts"
        )
        validation = _load_digested_file(
            validation_path, label=f"{bundle_id} validation report"
        )
        if (
            manifest.get("bundle_id") != bundle_id
            or manifest.get("campaign_id") != CAMPAIGN_ID
            or manifest.get("run_class") != RUN_CLASS
            or manifest.get("execution_target") != EXECUTION_TARGET
            or manifest.get("execution_authorized") is not False
            or manifest.get("submitted") is not False
            or int(manifest.get("validation_cell_count", -1)) != 10
            or int(manifest.get("cell_count", -1)) != 58
            or expected.get("bundle_id") != bundle_id
            or int(expected.get("cell_count", -1)) != 58
            or validation.get("bundle_id") != bundle_id
            or validation.get("materialization_status") != "passed"
        ):
            raise PackageContractError(f"v8 bundle state drifted for {bundle_id}.")

        final_bundle = final_bundles[bundle_id]
        if not isinstance(final_bundle, Mapping):
            raise PackageContractError(f"v8 final bundle row is invalid: {bundle_id}")
        for key, payload in (
            ("bundle_manifest_sha256", manifest),
            ("source_locks_sha256", source_locks),
            ("expected_artifacts_sha256", expected),
            ("validation_report_sha256", validation),
        ):
            if final_bundle.get(key) != payload["sha256"]:
                raise PackageContractError(
                    f"v8 final receipt {bundle_id}.{key} hash drifted."
                )

        dedupe = manifest.get("study1_shared_execution_dedupe")
        if not isinstance(dedupe, Mapping):
            raise PackageContractError(f"{bundle_id} has no dedupe contract.")
        observed_dedupe = _validate_dedupe(dedupe)
        if dedupe_sha256 is None:
            dedupe_sha256 = observed_dedupe
        elif dedupe_sha256 != observed_dedupe:
            raise PackageContractError("Bundle dedupe digests disagree.")

        inventory = source_locks.get("implementation_sources")
        if not isinstance(inventory, Mapping):
            raise PackageContractError(
                f"{bundle_id} has no implementation source inventory."
            )
        verify_self_digest(inventory, label=f"{bundle_id} source inventory")
        if (
            inventory.get("all_files_verified") is not True
            or int(inventory.get("file_count", -1))
            != len(inventory.get("files", ()))
        ):
            raise PackageContractError(
                f"{bundle_id} implementation source inventory is not verified."
            )
        if source_inventory is None:
            source_inventory = dict(inventory)
        elif source_inventory != dict(inventory):
            raise PackageContractError(
                "Stationary and measured implementation inventories differ."
            )

        manifest_rows = _manifest_validation_rows(
            manifest, bundle_id=bundle_id
        )
        expected_cells = expected.get("cells")
        if not isinstance(expected_cells, Mapping):
            raise PackageContractError(f"{bundle_id} expected cells are invalid.")
        protocols: dict[str, Any] = {}
        templates: dict[str, Any] = {}
        for regime_id in VALIDATION_REGIMES:
            for route_id in VALIDATION_ROUTES:
                cell_id = validation_cell_id(regime_id, route_id)
                row = manifest_rows[cell_id]
                if (
                    row.get("regime_id") != regime_id
                    or row.get("route_id") != route_id
                    or int(row.get("nph", -1)) != 3
                    or int(row.get("horizon", -1)) != 23
                ):
                    raise PackageContractError(
                        f"Validation cell identity drifted: {bundle_id}::{cell_id}"
                    )
                protocol_rel = safe_relative_path(
                    row.get("protocol_path"),
                    label=f"{bundle_id}::{cell_id} protocol path",
                )
                template_rel = safe_relative_path(
                    row.get("execution_template_path"),
                    label=f"{bundle_id}::{cell_id} template path",
                )
                if (
                    protocol_rel.as_posix() != f"protocols/{cell_id}.json"
                    or template_rel.as_posix()
                    != f"execution_templates/{cell_id}.json"
                ):
                    raise PackageContractError(
                        f"Validation protocol/template path drifted: "
                        f"{bundle_id}::{cell_id}"
                    )
                protocol_path = bundle_dir / protocol_rel
                template_path = bundle_dir / template_rel
                protocol = _load_digested_file(
                    protocol_path, label=f"{bundle_id}::{cell_id} protocol"
                )
                template = _load_digested_file(
                    template_path, label=f"{bundle_id}::{cell_id} template"
                )
                expected_cell = expected_cells.get(cell_id)
                if not isinstance(expected_cell, Mapping):
                    raise PackageContractError(
                        f"Missing expected-artifact cell: {bundle_id}::{cell_id}"
                    )
                expected_protocol = expected_cell.get("protocol")
                expected_template = expected_cell.get("execution_template")
                if (
                    not isinstance(expected_protocol, Mapping)
                    or expected_protocol.get("path") != protocol_rel.as_posix()
                    or expected_protocol.get("sha256") != protocol["sha256"]
                    or expected_protocol.get("status") != "resolved"
                    or not isinstance(expected_template, Mapping)
                    or expected_template.get("path") != template_rel.as_posix()
                    or expected_template.get("sha256") != template["sha256"]
                    or protocol.get("bundle_id") != bundle_id
                    or protocol.get("bundle_manifest_sha256")
                    != manifest["sha256"]
                    or protocol.get("execution_authorized") is not False
                    or template.get("execution_authorized") is not False
                    or template.get("submitted") is not False
                ):
                    raise PackageContractError(
                        f"Protocol/template binding drifted: {bundle_id}::{cell_id}"
                    )
                _validate_artifact_contract(
                    bundle_id=bundle_id,
                    cell_id=cell_id,
                    route_id=route_id,
                    expected_cell=expected_cell,
                    template=template,
                    dedupe_sha256=observed_dedupe,
                )
                protocols[cell_id] = _file_binding(
                    protocol_path,
                    payload=protocol,
                    display_path=protocol_path.relative_to(root).as_posix(),
                )
                templates[cell_id] = _file_binding(
                    template_path,
                    payload=template,
                    display_path=template_path.relative_to(root).as_posix(),
                )

        bundle_bindings[bundle_id] = {
            "bundle_root": bundle_dir.relative_to(root).as_posix(),
            "bundle_manifest": _file_binding(
                manifest_path,
                payload=manifest,
                display_path=manifest_path.relative_to(root).as_posix(),
            ),
            "source_locks": _file_binding(
                source_locks_path,
                payload=source_locks,
                display_path=source_locks_path.relative_to(root).as_posix(),
            ),
            "expected_artifacts": _file_binding(
                expected_path,
                payload=expected,
                display_path=expected_path.relative_to(root).as_posix(),
            ),
            "validation_report": _file_binding(
                validation_path,
                payload=validation,
                display_path=validation_path.relative_to(root).as_posix(),
            ),
            "validation_protocols": protocols,
            "validation_execution_templates": templates,
        }

    if source_inventory is None or dedupe_sha256 is None:
        raise AssertionError("v8 validation did not resolve source/dedupe authority.")
    final_inventory = final.get("implementation_inventory")
    if (
        not isinstance(final_inventory, Mapping)
        or final_inventory.get("stable") is not True
        or final_inventory.get("preflight_sha256") != source_inventory["sha256"]
        or final_inventory.get("post_staged_loader_sha256")
        != source_inventory["sha256"]
        or final_inventory.get("post_final_loader_sha256")
        != source_inventory["sha256"]
        or int(final_inventory.get("file_count", -1))
        != int(source_inventory["file_count"])
    ):
        raise PackageContractError(
            "v8 final receipt implementation inventory binding drifted."
        )

    verified_source_files: list[dict[str, Any]] = []
    seen: set[str] = set()
    raw_files = source_inventory.get("files")
    if not isinstance(raw_files, list):
        raise PackageContractError("Implementation source inventory has no files.")
    for raw in raw_files:
        if not isinstance(raw, Mapping):
            raise PackageContractError("Implementation inventory row is invalid.")
        relative = safe_relative_path(
            raw.get("path"), label="implementation source path"
        ).as_posix()
        expected_hash = require_sha256(
            raw.get("sha256"), label=f"implementation source {relative}"
        )
        if relative in seen:
            raise PackageContractError(
                f"Duplicate implementation inventory path: {relative}"
            )
        seen.add(relative)
        source = root / relative
        if not source.is_file() or source.is_symlink():
            raise PackageContractError(
                f"Verified implementation source is unavailable: {source}"
            )
        source_flags = int(getattr(source.stat(), "st_flags", 0))
        if source_flags & (MACOS_UF_COMPRESSED | MACOS_SF_DATALESS):
            raise PackageContractError(
                "Verified implementation source is compressed/dataless and "
                f"must be materialized before packaging: {relative}"
            )
        actual_hash = sha256_file(source)
        if actual_hash != expected_hash:
            raise PackageContractError(
                f"Implementation source drifted: {relative}: "
                f"{actual_hash} != {expected_hash}"
            )
        verified_source_files.append(
            {
                "path": relative,
                "sha256": expected_hash,
                "size_bytes": source.stat().st_size,
            }
        )
    if len(verified_source_files) != int(source_inventory["file_count"]):
        raise PackageContractError("Implementation source file count drifted.")

    from objective_gates import validate_objective_gate_authority

    objective_path = (
        Path(objective_gate_authority_path).resolve()
        if objective_gate_authority_path is not None
        else revision_root / OBJECTIVE_GATE_AUTHORITY_NAME
    )
    objective_gate_authority = validate_objective_gate_authority(
        receipt_path=objective_path,
        revision_root=revision_root,
        final_receipt=final,
    )

    return {
        "v8_root": revision_root,
        "final_receipt": final,
        "final_receipt_binding": {
            "path": (
                final_path.relative_to(root).as_posix()
                if final_path.is_relative_to(root)
                else final_path.as_posix()
            ),
            "canonical_sha256": final["sha256"],
            "file_sha256": sha256_file(final_path),
            "size_bytes": final_path.stat().st_size,
        },
        "bundle_bindings": bundle_bindings,
        "dedupe_contract": dict(
            load_json_object(
                revision_root / STATIONARY_BUNDLE_ID / "bundle_manifest.json",
                label="stationary bundle manifest",
            )["study1_shared_execution_dedupe"]
        ),
        "dedupe_sha256": dedupe_sha256,
        "source_inventory": source_inventory,
        "verified_source_files": verified_source_files,
        "objective_gate_authority": objective_gate_authority,
        "objective_gate_authority_binding": {
            "path": (
                objective_path.relative_to(root).as_posix()
                if objective_path.is_relative_to(root)
                else objective_path.as_posix()
            ),
            "canonical_sha256": objective_gate_authority["sha256"],
            "file_sha256": objective_gate_authority["file_sha256"],
            "size_bytes": objective_path.stat().st_size,
        },
    }


def submission_preflight_overlay_contract() -> dict[str, Any]:
    """Return the fixed one-way post-seal P4 overlay declaration."""

    return {
        "schema": SUBMISSION_PREFLIGHT_OVERLAY_SCHEMA,
        "path": SUBMISSION_PREFLIGHT_RELATIVE,
        "receipt_schema": SUBMISSION_PREFLIGHT_SCHEMA,
        "fixed_p4_execution_id": P4_EXECUTION_ID,
        "excluded_from_frozen_package_digest": True,
        "excluded_from_authorization_digest_as_realized_bytes": True,
        "back_binds_frozen_package": True,
        "required_before_staging_dry_run_and_submission": True,
        "only_admitted_post_seal_overlay": True,
    }


def validate_scientific_preflight_receipt(
    receipt: Mapping[str, Any],
    *,
    receipt_file_sha256: str,
    v8_authority: Mapping[str, Any],
    package_control_plane_sha256: str,
) -> dict[str, Any]:
    """Validate and bind the frozen P2/P3 receipt used by authorization."""

    digest = verify_self_digest(
        receipt,
        label="scientific P2/P3 preflight receipt",
    )
    final_binding = v8_authority["final_receipt_binding"]
    required = {
        "schema": SCIENTIFIC_PREFLIGHT_SCHEMA,
        "package_id": PACKAGE_ID,
        "materialization_revision": "v8",
        "v8_final_receipt_canonical_sha256": final_binding[
            "canonical_sha256"
        ],
        "v8_final_receipt_file_sha256": final_binding["file_sha256"],
        "study1_objective_gate_authority_sha256": v8_authority[
            "objective_gate_authority"
        ]["sha256"],
        "package_control_plane_sha256": require_sha256(
            package_control_plane_sha256,
            label="package control-plane SHA-256",
        ),
        "p2_passed": True,
        "p3_passed": True,
        "all_preflight_smokes_passed": True,
    }
    for field, expected in required.items():
        if receipt.get(field) != expected:
            raise PackageContractError(
                f"Scientific P2/P3 preflight drifted at {field}."
            )
    p2 = receipt.get("p2")
    p3 = receipt.get("p3")
    if (
        not isinstance(p2, Mapping)
        or p2.get("status") != "passed"
        or not isinstance(p3, Mapping)
        or p3.get("status") != "passed"
        or int(p3.get("actual_facade_execution_count", -1)) != 12
        or int(p3.get("case_count", -1)) != 4
    ):
        raise PackageContractError(
            "Scientific P2/P3 preflight did not close its exact smoke matrix."
        )
    return {
        "path": SCIENTIFIC_PREFLIGHT_RELATIVE,
        "canonical_sha256": digest,
        "file_sha256": require_sha256(
            receipt_file_sha256,
            label="scientific P2/P3 preflight file SHA-256",
        ),
    }


def validate_authorization_receipt(
    receipt: Mapping[str, Any],
    *,
    v8_authority: Mapping[str, Any],
    package_control_plane_sha256: str,
    scientific_preflight_binding: Mapping[str, Any],
) -> str:
    """Validate externally supplied user authorization without minting it."""

    digest = verify_self_digest(receipt, label="execution authorization receipt")
    final_binding = v8_authority["final_receipt_binding"]
    required = {
        "schema": AUTHORIZATION_SCHEMA,
        "authorization_id": receipt.get("authorization_id"),
        "authorized_utc": receipt.get("authorized_utc"),
        "package_id": PACKAGE_ID,
        "campaign_id": CAMPAIGN_ID,
        "run_class": RUN_CLASS,
        "execution_target": EXECUTION_TARGET,
        "execution_authorized": True,
        "submission_authorized": True,
        "v8_final_receipt_file_sha256": final_binding["file_sha256"],
        "v8_final_receipt_canonical_sha256": final_binding[
            "canonical_sha256"
        ],
        "study1_objective_gate_authority_sha256": v8_authority[
            "objective_gate_authority"
        ]["sha256"],
        "study1_dedupe_sha256": v8_authority["dedupe_sha256"],
        "package_control_plane_sha256": require_sha256(
            package_control_plane_sha256,
            label="package control-plane SHA-256",
        ),
        "scientific_preflight_receipt_canonical_sha256": require_sha256(
            scientific_preflight_binding.get("canonical_sha256"),
            label="scientific P2/P3 preflight canonical SHA-256",
        ),
        "scientific_preflight_receipt_file_sha256": require_sha256(
            scientific_preflight_binding.get("file_sha256"),
            label="scientific P2/P3 preflight file SHA-256",
        ),
        "submission_preflight_overlay_contract": (
            submission_preflight_overlay_contract()
        ),
        "logical_cell_count": 20,
        "direct_execution_count": 18,
        "authorized_logical_cell_keys": list(logical_cell_keys()),
        "authorized_direct_execution_ids": list(direct_execution_ids()),
        "remote_image_sha256": REMOTE_IMAGE_SHA256,
    }
    for field, expected in required.items():
        observed = receipt.get(field)
        if field in {"authorization_id", "authorized_utc"}:
            if not isinstance(observed, str) or not observed.strip():
                raise PackageContractError(
                    f"Authorization receipt has no nonempty {field}."
                )
        elif observed != expected:
            raise PackageContractError(
                f"Authorization receipt drifted at {field}: "
                f"{observed!r} != {expected!r}."
            )
    try:
        authorized_at = datetime.fromisoformat(
            str(receipt["authorized_utc"]).replace("Z", "+00:00")
        )
        finalized_at = datetime.fromisoformat(
            str(v8_authority["final_receipt"]["finalized_utc"]).replace(
                "Z", "+00:00"
            )
        )
        if authorized_at.tzinfo is None or finalized_at.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
    except (KeyError, TypeError, ValueError) as exc:
        raise PackageContractError(
            "Authorization/finalization timestamps are invalid."
        ) from exc
    if authorized_at.astimezone(timezone.utc) <= finalized_at.astimezone(
        timezone.utc
    ):
        raise PackageContractError(
            "Execution authorization must postdate the immutable v8 "
            "finalization receipt."
        )
    return digest


def resource_envelope(regime_id: str, route_id: str) -> tuple[int, int]:
    representation = "singleton" if route_id == "singleton_plateau" else "macro"
    try:
        return RESOURCE_ENVELOPES_MB[(regime_id, representation)]
    except KeyError as exc:
        raise PackageContractError(
            f"No reviewed CHTC resource envelope for {regime_id}/{route_id}."
        ) from exc


def verify_exact_key_set(
    observed: Iterable[str],
    expected: Sequence[str],
    *,
    label: str,
) -> None:
    observed_set = set(observed)
    expected_set = set(expected)
    if observed_set != expected_set:
        raise PackageContractError(
            f"{label} key set drifted: missing={sorted(expected_set-observed_set)}, "
            f"extra={sorted(observed_set-expected_set)}"
        )


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "ATTEMPT_SELECTION_SCHEMA",
    "BATCH_NAME",
    "BUNDLE_IDS",
    "CAMPAIGN_ID",
    "COMPLETION_MATRIX_SCHEMA",
    "EXECUTION_PLAN_SCHEMA",
    "EXECUTION_TARGET",
    "EXPECTED_ARTIFACT_ROLES",
    "FETCH_VALIDATION_SCHEMA",
    "G11_JOB_DIAGNOSTIC_SCHEMA",
    "JOB_SPEC_SCHEMA",
    "MAX_RUNTIME_SECONDS",
    "MEASURED_BUNDLE_ID",
    "OBJECTIVE_GATE_AUTHORITY_NAME",
    "PACKAGE_ID",
    "PACKAGE_RELATIVE_ROOT",
    "PACKAGE_CONTROL_PLANE_FILES",
    "PACKAGE_CONTROL_PLANE_SCHEMA",
    "PACKAGE_MANIFEST_SCHEMA",
    "P4_EXECUTION_ID",
    "PackageContractError",
    "REMOTE_IMAGE_PATH",
    "REMOTE_IMAGE_SHA256",
    "REMOTE_REPOSITORY_ROOT_BASENAME",
    "REQUEST_CPUS",
    "RUN_CLASS",
    "SCIENTIFIC_PREFLIGHT_RELATIVE",
    "SCIENTIFIC_PREFLIGHT_SCHEMA",
    "SHARED_APPEND_EQUIVALENCE_SCHEMA",
    "STATIONARY_BUNDLE_ID",
    "SUBMISSION_PREFLIGHT_OVERLAY_SCHEMA",
    "SUBMISSION_PREFLIGHT_RELATIVE",
    "SUBMISSION_PREFLIGHT_SCHEMA",
    "V8_FINAL_RECEIPT_NAME",
    "V8_RELATIVE_ROOT",
    "V8_REVISION",
    "VALIDATION_REGIMES",
    "VALIDATION_ROUTES",
    "WORKER_RECEIPT_SCHEMA",
    "atomic_write_json",
    "canonical_json_bytes",
    "canonical_sha256",
    "digested",
    "direct_execution_ids",
    "direct_execution_rows",
    "expected_artifact_path",
    "load_json_object",
    "logical_cell_keys",
    "objective_gate_diagnostic_contract",
    "package_control_plane_receipt",
    "repo_root_from_script",
    "require_safe_id",
    "require_sha256",
    "resource_envelope",
    "safe_relative_path",
    "sha256_file",
    "shared_append_rows",
    "stage_packaged_runtime_tree",
    "submission_preflight_overlay_contract",
    "validate_authorization_receipt",
    "validate_scientific_preflight_receipt",
    "validate_v8_authority",
    "validation_cell_id",
    "verify_exact_key_set",
    "verify_self_digest",
]
