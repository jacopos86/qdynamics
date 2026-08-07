#!/usr/bin/env python3
"""Fail-closed contract for the inert corrected-always 48-cell package."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

PACKAGE_ID = (
    "paper_i_ra_adapt_always_factorial48_r50_20260730_v1_chtc"
)
PACKAGE_RELATIVE_ROOT = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "ra_always_factorial48_r50_20260730_v1_chtc"
)
MATERIALIZATION_ID = "ra_adapt_always_factorial48_v1"
MATERIALIZATION_RELATIVE_ROOT = (
    "chtc/paper_i_ra_adapt_repair_20260727/bundles/materializations/"
    f"{MATERIALIZATION_ID}"
)
MATERIALIZATION_RECEIPT_NAME = "factorial_materialization_receipt.json"
MATERIALIZER_RELATIVE_PATH = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "materialize_ra_always_factorial48_v1.py"
)
V13_FINAL_RECEIPT_RELATIVE_PATH = (
    "chtc/paper_i_ra_adapt_repair_20260727/bundles/materializations/"
    "ra_adapt_stationary_late_core_v13/final_publication_receipt.json"
)

CAMPAIGN_ID = (
    "paper_i_ra_adapt_always_stationarity_phase1_cost_factorial_v1"
)
RUN_CLASS = "diagnostic"
EXECUTION_TARGET = "chtc"
HORIZON = 50
OPTIMIZER = "powell"
OPTIMIZER_MAXITER = 200
SEED = 7
ALWAYS_INSERTION_KIND = "always_commutation_reduced"
ALWAYS_INSERTION_MODE = "full_commutation_reduced"
INSERTION_EQUIVALENCE_POLICY = (
    "termwise_cross_component_commutation_earliest_representative_v1"
)
INSERTION_POSITION_SCOPE = (
    "full_logical_ansatz_commutation_classes_every_depth_v2"
)
ACTIVE_GRADIENT_STATIONARY = "stationary_source_response_v1"
ACTIVE_GRADIENT_MEASURED = "measured_residual_response_v1"
RESOURCE_WEIGHTING_LATE = "late_resource_weighting_v1"
RESOURCE_WEIGHTING_ALL_PHASE = "all_phase_resource_weighting_v1"
REGIME_CUTOFF_PAIRS = (
    ("weak_weak", 3),
    ("intermediate_weak", 3),
    ("strong_weak_u8", 3),
    ("weak_strong", 7),
    ("intermediate_strong", 7),
    ("strong_strong_u8", 7),
)
ROUTE_IDS = ("ra_macro_always", "ra_singleton_always")
BUNDLE_POLICIES = (
    (
        "ra_repair_always_factorial_stationary_late_v1",
        ACTIVE_GRADIENT_STATIONARY,
        RESOURCE_WEIGHTING_LATE,
        "gradient_stationary__phase1_cost_off",
    ),
    (
        "ra_repair_always_factorial_measured_late_v1",
        ACTIVE_GRADIENT_MEASURED,
        RESOURCE_WEIGHTING_LATE,
        "gradient_measured__phase1_cost_off",
    ),
    (
        "ra_repair_always_factorial_stationary_all_phase_v1",
        ACTIVE_GRADIENT_STATIONARY,
        RESOURCE_WEIGHTING_ALL_PHASE,
        "gradient_stationary__phase1_cost_on",
    ),
    (
        "ra_repair_always_factorial_measured_all_phase_v1",
        ACTIVE_GRADIENT_MEASURED,
        RESOURCE_WEIGHTING_ALL_PHASE,
        "gradient_measured__phase1_cost_on",
    ),
)
DIRECT_EXECUTION_COUNT = 48
SMOKE_ROUNDS = 2
SMOKE_EXECUTION_IDS = tuple(
    (
        "core__strong_weak_u8__nph3__"
        f"{route_id}__{suffix}"
    )
    for _bundle_id, _gradient, _scope, suffix in BUNDLE_POLICIES
    for route_id in ROUTE_IDS
)
RESOURCE_ENVELOPES = {
    ("ra_macro_always", 3): (4, 49_152, 61_440),
    ("ra_singleton_always", 3): (4, 57_344, 61_440),
    ("ra_macro_always", 7): (4, 65_536, 81_920),
    ("ra_singleton_always", 7): (4, 90_112, 98_304),
}
MAX_RUNTIME_SECONDS = 72 * 60 * 60
REMOTE_IMAGE_PATH = "chtc/phase3_optuna/image.sif"
REMOTE_IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)

SMOKE_RECEIPT_NAME = "two_round_smoke_receipt.json"
SOURCE_ARCHIVE_NAME = "source_locked.tar.gz"
SOURCE_ARCHIVE_MANIFEST_NAME = "source_archive_manifest.json"
EXECUTION_PLAN_NAME = "execution_plan.json"
QUEUE_NAME = "queue.tsv"
PACKAGE_MANIFEST_NAME = "package_manifest.json"
JOB_SCHEMA = "paper_i_ra_always_factorial_job_v1"
SMOKE_SCHEMA = "paper_i_ra_always_factorial_smoke_v1"
PLAN_SCHEMA = "paper_i_ra_always_factorial_execution_plan_v1"
MANIFEST_SCHEMA = "paper_i_ra_always_factorial_package_v1"
ARCHIVE_SCHEMA = "paper_i_ra_always_factorial_source_archive_v1"

CONTROL_FILES = (
    "package_contract.py",
    "run_semantic_preflight.py",
    "build_package.py",
    "validate_package.py",
    "run_cell.py",
    "execute_source_locked_job.sh",
    "submit.sub",
)
GENERATED_FILES = (
    SMOKE_RECEIPT_NAME,
    SOURCE_ARCHIVE_NAME,
    SOURCE_ARCHIVE_MANIFEST_NAME,
    EXECUTION_PLAN_NAME,
    QUEUE_NAME,
    PACKAGE_MANIFEST_NAME,
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PackageContractError(ValueError):
    """Raised when the factorial package contract does not close."""


def repo_root_from_script(script_path: str | Path) -> Path:
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


def digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("sha256", None)
    result["sha256"] = canonical_sha256(result)
    return result


def load_json(path: str | Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        raise PackageContractError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackageContractError(f"{label} is not valid JSON.") from exc
    if not isinstance(value, dict):
        raise PackageContractError(f"{label} must be a JSON object.")
    return value


def verify_self_digest(
    payload: Mapping[str, Any], *, label: str
) -> str:
    unsigned = dict(payload)
    observed = unsigned.pop("sha256", None)
    if (
        not isinstance(observed, str)
        or SHA256_RE.fullmatch(observed) is None
        or canonical_sha256(unsigned) != observed
    ):
        raise PackageContractError(f"{label} self digest drifted.")
    return observed


def safe_relative_path(value: Any, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise PackageContractError(f"{label} must be a nonempty path.")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or any(not part for part in path.parts)
    ):
        raise PackageContractError(f"{label} is unsafe: {value!r}.")
    return path


def _base_execution_id(
    regime_id: str, nph: int, route_id: str
) -> str:
    return f"core__{regime_id}__nph{nph}__{route_id}"


def direct_execution_rows() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for (
        bundle_id,
        active_gradient_policy,
        resource_weighting_scope,
        suffix,
    ) in BUNDLE_POLICIES:
        for regime_id, nph in REGIME_CUTOFF_PAIRS:
            for route_id in ROUTE_IDS:
                representation = (
                    "macro_generator_v1"
                    if route_id == "ra_macro_always"
                    else "single_pauli_word_v1"
                )
                cpus, memory_mb, disk_mb = RESOURCE_ENVELOPES[
                    (route_id, nph)
                ]
                base_id = _base_execution_id(
                    regime_id, nph, route_id
                )
                execution_id = f"{base_id}__{suffix}"
                rows.append(
                    {
                        "execution_id": execution_id,
                        "cell_id": execution_id,
                        "base_cell_id": base_id,
                        "source_lock_id": (
                            f"{regime_id}__nph{nph}__{route_id}"
                        ),
                        "regime_id": regime_id,
                        "nph": nph,
                        "route_id": route_id,
                        "candidate_representation": representation,
                        "bundle_id": bundle_id,
                        "active_gradient_policy": (
                            active_gradient_policy
                        ),
                        "resource_weighting_scope": (
                            resource_weighting_scope
                        ),
                        "phase1_cost_term": (
                            "disabled_for_phase1_only"
                            if resource_weighting_scope
                            == RESOURCE_WEIGHTING_LATE
                            else "enabled"
                        ),
                        "execution_entrypoint": "run_ra_adapt",
                        "resources": {
                            "request_cpus": cpus,
                            "request_memory_mb": memory_mb,
                            "request_disk_mb": disk_mb,
                            "max_runtime_seconds": (
                                MAX_RUNTIME_SECONDS
                            ),
                        },
                    }
                )
    if len(rows) != DIRECT_EXECUTION_COUNT:
        raise AssertionError("Factorial execution cardinality drifted.")
    return tuple(rows)


def _validate_protocol(
    protocol: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
) -> None:
    verify_self_digest(protocol, label=f"{row['cell_id']} protocol")
    request = protocol.get("request")
    method = request.get("method") if isinstance(request, Mapping) else None
    insertion = (
        method.get("insertion") if isinstance(method, Mapping) else None
    )
    route_contract = protocol.get("route_contract")
    execution_settings = (
        route_contract.get("execution_settings")
        if isinstance(route_contract, Mapping)
        else None
    )
    invariants = (
        route_contract.get("semantic_invariants")
        if isinstance(route_contract, Mapping)
        else None
    )
    materialization = protocol.get("bundle_materialization")
    if (
        not isinstance(materialization, Mapping)
        or materialization.get("cell_id") != row["cell_id"]
        or materialization.get("source_lock_id")
        != row["source_lock_id"]
        or materialization.get("bundle_id") != row["bundle_id"]
        or materialization.get("active_gradient_policy")
        != row["active_gradient_policy"]
        or materialization.get("resource_weighting_scope")
        != row["resource_weighting_scope"]
        or protocol.get("bundle_id") != row["bundle_id"]
        or protocol.get("candidate_representation")
        != row["candidate_representation"]
        or protocol.get("active_gradient_policy")
        != row["active_gradient_policy"]
        or protocol.get("resource_weighting_scope")
        != row["resource_weighting_scope"]
        or int(protocol.get("horizon", -1)) != HORIZON
        or str(protocol.get("optimizer", "")).lower() != OPTIMIZER
        or int(protocol.get("optimizer_maxiter", -1))
        != OPTIMIZER_MAXITER
        or protocol.get("seeds")
        != {"adapt": SEED, "transpiler": SEED}
        or not isinstance(insertion, Mapping)
        or insertion.get("kind") != ALWAYS_INSERTION_KIND
        or not isinstance(execution_settings, Mapping)
        or execution_settings.get("adapt_insertion_mode")
        != ALWAYS_INSERTION_MODE
        or not isinstance(invariants, Mapping)
        or invariants.get("active_gradient_policy")
        != row["active_gradient_policy"]
        or invariants.get("resource_weighting_scope")
        != row["resource_weighting_scope"]
        or invariants.get("insertion_position_scope")
        != INSERTION_POSITION_SCOPE
        or invariants.get("insertion_equivalence_policy")
        != INSERTION_EQUIVALENCE_POLICY
        or (
            invariants.get("canonical_insertion_policy") is not None
            and invariants.get("canonical_insertion_policy")
            != ALWAYS_INSERTION_KIND
        )
        or protocol.get("execution_authorized") is not False
    ):
        raise PackageContractError(
            f"Factorial protocol drifted: {row['cell_id']}."
        )


def _normalize_non_axis_protocol(
    value: Any,
    *,
    row: Mapping[str, Any],
) -> Any:
    """Remove only axis identities and their deterministic hash cascade."""

    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if (
                key == "sha256"
                or key.endswith("_sha256")
                or key
                in {
                    "active_gradient_policy",
                    "resource_weighting_scope",
                    "bundle_id",
                    "bundle_manifest_sha256",
                }
            ):
                continue
            normalized[str(key)] = _normalize_non_axis_protocol(
                item, row=row
            )
        return normalized
    if isinstance(value, list):
        return [
            _normalize_non_axis_protocol(item, row=row)
            for item in value
        ]
    if isinstance(value, str):
        replacements = (
            (str(row["cell_id"]), str(row["base_cell_id"])),
            (str(row["bundle_id"]), "<factorial_bundle>"),
            (ACTIVE_GRADIENT_STATIONARY, "<active_gradient_policy>"),
            (ACTIVE_GRADIENT_MEASURED, "<active_gradient_policy>"),
            (RESOURCE_WEIGHTING_LATE, "<resource_weighting_scope>"),
            (
                RESOURCE_WEIGHTING_ALL_PHASE,
                "<resource_weighting_scope>",
            ),
        )
        result = value
        for before, after in replacements:
            result = result.replace(before, after)
        return result
    return value


def _protocol_binding(
    path: Path, *, repo_root: Path
) -> dict[str, Any]:
    protocol = load_json(path, label=f"{path.stem} protocol")
    return {
        "path": path.relative_to(repo_root).as_posix(),
        "sha256": sha256_file(path),
        "canonical_sha256": str(protocol["sha256"]),
        "size_bytes": path.stat().st_size,
    }


def _normalize_non_axis_source_locks(
    source_locks: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = json.loads(json.dumps(source_locks))
    normalized.pop("sha256", None)
    cells = normalized.get("cell_locks")
    if not isinstance(cells, dict):
        raise PackageContractError(
            "Factorial source-lock normalization has no cell map."
        )
    for lock_id, lock in cells.items():
        if isinstance(lock, dict):
            lock.pop("sha256", None)
        trace = (
            lock.get("resolver_trace")
            if isinstance(lock, Mapping)
            else None
        )
        changes = (
            trace.get("settings_changed")
            if isinstance(trace, Mapping)
            else None
        )
        if not isinstance(changes, list):
            raise PackageContractError(
                f"Factorial source lock has no change list: {lock_id}."
            )
        for row in changes:
            if not isinstance(row, dict):
                continue
            if row.get("field") == "active_gradient_policy":
                if "to" in row:
                    row["to"] = "<active_gradient_policy>"
                if "to_bundle_values" in row:
                    row["to_bundle_values"] = [
                        "<active_gradient_policy>"
                    ]
            elif row.get("field") == "resource_weighting_scope":
                row["to"] = "<resource_weighting_scope>"
    return normalized


def validate_factorial_authority(
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    materialization_root = root / MATERIALIZATION_RELATIVE_ROOT
    final_path = materialization_root / MATERIALIZATION_RECEIPT_NAME
    final = load_json(final_path, label="factorial materialization receipt")
    verify_self_digest(final, label="factorial materialization receipt")
    false_fields = (
        "execution_authorized",
        "submission_authorized",
        "remote_stage",
        "condor_submit",
    )
    if (
        final.get("schema")
        != (
            "paper_i_ra_adapt_always_factorial48_"
            "materialization_receipt_v1"
        )
        or final.get("status") != "passed"
        or final.get("materialization_id") != MATERIALIZATION_ID
        or final.get("campaign_id") != CAMPAIGN_ID
        or final.get("run_class") != RUN_CLASS
        or int(final.get("arm_count", -1)) != len(BUNDLE_POLICIES)
        or int(final.get("cell_count_per_arm", -1)) != 12
        or int(final.get("total_cell_count", -1))
        != DIRECT_EXECUTION_COUNT
        or final.get("submission_state") != "not_submitted"
        or final.get("submitted") is not False
        or any(final.get(field) is not False for field in false_fields)
    ):
        raise PackageContractError(
            "Factorial materialization receipt drifted."
        )
    factor_delta_path = (
        materialization_root
        / "source_materialization/factor_delta_receipt.json"
    )
    factor_delta = load_json(
        factor_delta_path, label="factor source-lock delta receipt"
    )
    verify_self_digest(
        factor_delta, label="factor source-lock delta receipt"
    )
    factor_delta_binding = _protocol_binding(
        factor_delta_path, repo_root=root
    )
    final_factor_binding = final.get("source_anchor", {}).get(
        "factor_delta_receipt"
    )
    expected_final_factor_path = (
        factor_delta_path.relative_to(materialization_root).as_posix()
    )
    if (
        factor_delta.get("schema")
        != "paper_i_ra_adapt_always_factorial_source_lock_delta_v1"
        or factor_delta.get("status") != "passed"
        or factor_delta.get("allowed_changed_fields")
        != [
            "active_gradient_policy",
            "resource_weighting_scope",
        ]
        or int(factor_delta.get("arm_count", -1)) != 4
        or int(factor_delta.get("row_count", -1)) != 48
        or factor_delta.get(
            "all_non_axis_source_lock_fields_equal"
        )
        is not True
        or factor_delta.get("all_archive_bindings_preserved") is not True
        or factor_delta.get("all_member_bindings_preserved") is not True
        or factor_delta.get(
            "all_global_source_bindings_preserved"
        )
        is not True
        or not isinstance(final_factor_binding, Mapping)
        or final_factor_binding.get("path")
        != expected_final_factor_path
        or final_factor_binding.get("sha256")
        != factor_delta_binding["sha256"]
        or final_factor_binding.get("canonical_sha256")
        != factor_delta_binding["canonical_sha256"]
        or int(final_factor_binding.get("size_bytes", -1))
        != int(factor_delta_binding["size_bytes"])
    ):
        raise PackageContractError(
            "Factorial source-lock delta provenance drifted."
        )

    rows = direct_execution_rows()
    rows_by_bundle = {
        bundle_id: [
            row for row in rows if row["bundle_id"] == bundle_id
        ]
        for bundle_id, *_rest in BUNDLE_POLICIES
    }
    protocol_bindings: dict[str, dict[str, Any]] = {}
    normalized_by_base: dict[str, list[tuple[str, Any]]] = {}
    implementation_inventory: dict[str, Any] | None = None
    source_locks_reference: dict[str, Any] | None = None
    bundle_bindings: list[dict[str, Any]] = []
    artifact_destinations: dict[str, list[str]] = {
        "result": [],
        "checkpoint": [],
        "estimator_ledger": [],
    }
    artifact_destinations_by_execution_id: dict[
        str, dict[str, str]
    ] = {}

    for (
        bundle_id,
        active_gradient_policy,
        resource_weighting_scope,
        _suffix,
    ) in BUNDLE_POLICIES:
        bundle_root = materialization_root / bundle_id
        manifest_path = bundle_root / "bundle_manifest.json"
        manifest = load_json(
            manifest_path, label=f"{bundle_id} manifest"
        )
        verify_self_digest(manifest, label=f"{bundle_id} manifest")
        source_locks_path = bundle_root / "source_locks.json"
        source_locks = load_json(
            source_locks_path, label=f"{bundle_id} source locks"
        )
        verify_self_digest(
            source_locks, label=f"{bundle_id} source locks"
        )
        validation_path = bundle_root / "validation_report.json"
        validation = load_json(
            validation_path, label=f"{bundle_id} validation"
        )
        verify_self_digest(
            validation, label=f"{bundle_id} validation"
        )
        expected_path = bundle_root / "expected_artifacts.json"
        expected = load_json(
            expected_path, label=f"{bundle_id} expected artifacts"
        )
        verify_self_digest(
            expected, label=f"{bundle_id} expected artifacts"
        )
        if (
            manifest.get("bundle_id") != bundle_id
            or manifest.get("campaign_id") != CAMPAIGN_ID
            or manifest.get("run_class") != RUN_CLASS
            or manifest.get("active_gradient_policy")
            != active_gradient_policy
            or manifest.get("resource_weighting_scope")
            != resource_weighting_scope
            or int(manifest.get("cell_count", -1)) != 12
            or manifest.get("execution_authorized") is not False
            or manifest.get("submission_state") != "not_submitted"
            or manifest.get("submitted") is not False
            or validation.get("materialization_status") != "passed"
            or validation.get("execution_authorized") is not False
            or validation.get("submission_state") != "not_submitted"
            or validation.get("submitted") is not False
        ):
            raise PackageContractError(
                f"Factorial bundle envelope drifted: {bundle_id}."
            )
        inventory = source_locks.get("implementation_sources")
        cell_locks = source_locks.get("cell_locks")
        if (
            not isinstance(inventory, Mapping)
            or not isinstance(cell_locks, Mapping)
            or len(cell_locks) != 12
        ):
            raise PackageContractError(
                f"Factorial source locks drifted: {bundle_id}."
            )
        verify_self_digest(
            inventory, label=f"{bundle_id} implementation inventory"
        )
        if implementation_inventory is None:
            implementation_inventory = dict(inventory)
            source_locks_reference = dict(source_locks)
        elif (
            dict(inventory) != implementation_inventory
            or _normalize_non_axis_source_locks(source_locks)
            != _normalize_non_axis_source_locks(
                source_locks_reference or {}
            )
        ):
            raise PackageContractError(
                "Factorial arms do not share one exact source lock."
            )

        bundle_bindings.append(
            {
                "bundle_id": bundle_id,
                "active_gradient_policy": active_gradient_policy,
                "resource_weighting_scope": resource_weighting_scope,
                "manifest": _protocol_binding(
                    manifest_path, repo_root=root
                ),
                "source_locks": _protocol_binding(
                    source_locks_path, repo_root=root
                ),
                "validation_report": _protocol_binding(
                    validation_path, repo_root=root
                ),
                "expected_artifacts": _protocol_binding(
                    expected_path, repo_root=root
                ),
            }
        )
        for row in rows_by_bundle[bundle_id]:
            protocol_path = (
                bundle_root / "protocols" / f"{row['cell_id']}.json"
            )
            protocol = load_json(
                protocol_path, label=f"{row['cell_id']} protocol"
            )
            _validate_protocol(protocol, row=row)
            protocol_bindings[str(row["execution_id"])] = (
                _protocol_binding(protocol_path, repo_root=root)
            )
            normalized_by_base.setdefault(
                str(row["base_cell_id"]), []
            ).append(
                (
                    str(row["execution_id"]),
                    _normalize_non_axis_protocol(protocol, row=row),
                )
            )
            expected_cell = expected.get("cells", {}).get(
                row["cell_id"]
            )
            expected_roles = (
                expected_cell.get("expected_run_artifacts")
                if isinstance(expected_cell, Mapping)
                else None
            )
            if not isinstance(expected_roles, Mapping):
                raise PackageContractError(
                    "Expected artifact contract is missing for "
                    f"{row['cell_id']}."
                )
            execution_destinations: dict[str, str] = {}
            for role in artifact_destinations:
                artifact = expected_roles.get(role)
                path_text = (
                    artifact.get("path")
                    if isinstance(artifact, Mapping)
                    else None
                )
                if (
                    not isinstance(path_text, str)
                    or not path_text.startswith(
                        f"runs/{row['cell_id']}/"
                    )
                ):
                    raise PackageContractError(
                        "Expected artifact destination drifted for "
                        f"{row['cell_id']}.{role}."
                    )
                artifact_destinations[role].append(path_text)
                execution_destinations[role] = path_text
            artifact_destinations_by_execution_id[
                str(row["execution_id"])
            ] = execution_destinations

    equality_rows: list[dict[str, Any]] = []
    if (
        len(normalized_by_base) != 12
        or len(protocol_bindings) != DIRECT_EXECUTION_COUNT
    ):
        raise PackageContractError(
            "Factorial protocol cardinality drifted."
        )
    for base_cell_id, variants in sorted(normalized_by_base.items()):
        if len(variants) != len(BUNDLE_POLICIES):
            raise PackageContractError(
                f"Factorial arm projection is incomplete: {base_cell_id}."
            )
        reference = variants[0][1]
        if any(candidate != reference for _, candidate in variants[1:]):
            raise PackageContractError(
                "Non-axis protocol fields differ across factorial arms: "
                f"{base_cell_id}."
            )
        equality_rows.append(
            {
                "base_cell_id": base_cell_id,
                "variant_execution_ids": [
                    execution_id for execution_id, _ in variants
                ],
                "normalized_non_axis_sha256": canonical_sha256(
                    reference
                ),
                "status": "passed",
            }
        )

    if implementation_inventory is None or source_locks_reference is None:
        raise PackageContractError(
            "Factorial implementation/source inventory is missing."
        )
    for role, paths in artifact_destinations.items():
        if (
            len(paths) != DIRECT_EXECUTION_COUNT
            or len(set(paths)) != DIRECT_EXECUTION_COUNT
        ):
            raise PackageContractError(
                f"Factorial {role} destinations are not unique."
            )
    return {
        "materialization_root": materialization_root,
        "final": final,
        "final_binding": _protocol_binding(
            final_path, repo_root=root
        ),
        "factor_delta_binding": factor_delta_binding,
        "source_locks": source_locks_reference,
        "implementation_inventory": implementation_inventory,
        "protocol_bindings": protocol_bindings,
        "bundle_bindings": bundle_bindings,
        "artifact_destinations": artifact_destinations,
        "artifact_destinations_by_execution_id": (
            artifact_destinations_by_execution_id
        ),
        "equality_audit": digested(
            {
                "schema": (
                    "paper_i_ra_adapt_always_factorial_"
                    "cross_arm_equality_v1"
                ),
                "status": "passed",
                "arm_count": len(BUNDLE_POLICIES),
                "base_cell_count": len(equality_rows),
                "variant_count": DIRECT_EXECUTION_COUNT,
                "allowed_axes": [
                    "active_gradient_policy",
                    "resource_weighting_scope",
                ],
                "derived_identity_fields_normalized": True,
                "rows": equality_rows,
            }
        ),
        "rows": rows,
    }


def _validate_candidate_plan(
    plan: Mapping[str, Any],
    *,
    requested: Sequence[int],
) -> tuple[bool, bool]:
    requested_positions = [
        int(value) for value in plan.get("requested_positions", ())
    ]
    representatives = [
        int(value)
        for value in plan.get("representative_positions", ())
    ]
    members = plan.get("members_by_representative")
    representative_by_position = plan.get(
        "representative_by_position"
    )
    if (
        plan.get("schema")
        != "commutation_reduced_insertion_positions_v1"
        or requested_positions != list(requested)
        or not representatives
        or not isinstance(members, Mapping)
        or not isinstance(representative_by_position, Mapping)
    ):
        raise PackageContractError(
            "Smoke candidate plan is not a reduced full-domain plan."
        )
    normalized_members = {
        int(key): [int(value) for value in values]
        for key, values in members.items()
        if isinstance(values, list)
    }
    if (
        sorted(normalized_members) != representatives
        or any(
            not values or representative != min(values)
            for representative, values in normalized_members.items()
        )
        or sorted(
            position
            for values in normalized_members.values()
            for position in values
        )
        != requested_positions
        or {
            int(key): int(value)
            for key, value in representative_by_position.items()
        }
        != {
            position: representative
            for representative, values in normalized_members.items()
            for position in values
        }
        or int(plan.get("collapsed_position_count", -1))
        != len(requested_positions) - len(representatives)
    ):
        raise PackageContractError(
            "Smoke commutation-equivalence membership does not close."
        )
    crossings = plan.get("commuting_crossings")
    if (
        not isinstance(crossings, list)
        or len(crossings) != max(requested_positions)
        or any(not isinstance(value, bool) for value in crossings)
    ):
        raise PackageContractError(
            "Smoke commuting-crossing certificate drifted."
        )
    class_start_by_position = {0: 0}
    class_start = 0
    for crossing_index, crossing in enumerate(crossings):
        if not crossing:
            class_start = crossing_index + 1
        class_start_by_position[crossing_index + 1] = class_start
    requested_by_class: dict[int, list[int]] = {}
    for position in requested_positions:
        requested_by_class.setdefault(
            class_start_by_position[position], []
        ).append(position)
    expected_members = {
        min(values): values for values in requested_by_class.values()
    }
    if normalized_members != expected_members:
        raise PackageContractError(
            "Smoke equivalence classes disagree with their "
            "commuting-crossing certificate."
        )
    return (
        len(representatives) < len(requested_positions),
        len(representatives) == len(requested_positions),
    )


def validate_smoke_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    verify_self_digest(receipt, label="two-round smoke receipt")
    observations = receipt.get("observations")
    if (
        receipt.get("schema") != SMOKE_SCHEMA
        or receipt.get("status") != "passed"
        or receipt.get("package_id") != PACKAGE_ID
        or receipt.get("maximum_controller_rounds") != SMOKE_ROUNDS
        or receipt.get("paper_evidence_allowed") is not False
        or receipt.get("execution_authorized") is not False
        or receipt.get("submission_authorized") is not False
        or not isinstance(observations, list)
        or [row.get("execution_id") for row in observations]
        != list(SMOKE_EXECUTION_IDS)
    ):
        raise PackageContractError("Two-round smoke envelope drifted.")

    expected_by_id = {
        row["execution_id"]: row
        for row in direct_execution_rows()
        if row["execution_id"] in SMOKE_EXECUTION_IDS
    }
    for observation in observations:
        execution_id = str(observation["execution_id"])
        row = expected_by_id[execution_id]
        rounds = observation.get("accepted_round_reduction_receipts")
        trajectory_policy = observation.get(
            "trajectory_policy_observation"
        )
        policy = observation.get("active_gradient_observation")
        cost = observation.get("phase1_cost_observation")
        if (
            observation.get("active_gradient_policy")
            != row["active_gradient_policy"]
            or observation.get("resource_weighting_scope")
            != row["resource_weighting_scope"]
            or observation.get("typed_insertion_policy")
            != ALWAYS_INSERTION_KIND
            or not isinstance(rounds, list)
            or len(rounds) != SMOKE_ROUNDS
            or observation.get("controller_round_count")
            != SMOKE_ROUNDS
            or not isinstance(trajectory_policy, Mapping)
            or not isinstance(policy, Mapping)
            or not isinstance(cost, Mapping)
        ):
            raise PackageContractError(
                f"Smoke observation drifted: {execution_id}."
            )
        if (
            trajectory_policy.get("active_gradient_policy")
            != row["active_gradient_policy"]
            or trajectory_policy.get("resource_weighting_scope")
            != row["resource_weighting_scope"]
            or policy.get("probe_kind")
            != (
                "production_phase3_active_gradient_"
                "promotion_depth1_v1"
            )
            or int(policy.get("active_depth", -1)) != 1
            or policy.get("active_gradient_policy")
            != row["active_gradient_policy"]
        ):
            raise PackageContractError(
                f"Active-gradient smoke binding drifted: {execution_id}."
            )
        if row["active_gradient_policy"] == ACTIVE_GRADIENT_STATIONARY:
            if (
                policy.get("active_gradient_indices_acquired") != []
                or int(policy.get("active_gradient_charge", -1)) != 0
                or policy.get("active_gradient_source")
                != "not_acquired_stationary_source_protocol"
                or policy.get("g_A") != [0.0]
                or trajectory_policy.get(
                    "active_gradient_indices_acquired"
                )
                != []
                or int(
                    trajectory_policy.get(
                        "active_gradient_charge", -1
                    )
                )
                != 0
            ):
                raise PackageContractError(
                    "Stationary smoke acquired active gradients."
                )
        elif (
            not policy.get("active_gradient_indices_acquired")
            or int(policy.get("active_gradient_charge", 0)) <= 0
            or policy.get("active_gradient_source")
            != "measured_active_residual_response_v1"
        ):
            raise PackageContractError(
                "Measured smoke did not acquire active gradients."
            )
        raw_burden = float(cost.get("phase1_raw_burden", 0.0))
        effective_burden = float(
            cost.get("phase1_effective_burden", 0.0)
        )
        if (
            raw_burden <= 0.0
            or cost.get("probe_kind")
            != (
                "production_phase1_score_payload_"
                "nonunit_raw_burden_v1"
            )
            or cost.get("resource_weighting_scope")
            != row["resource_weighting_scope"]
            or (
                row["resource_weighting_scope"]
                == RESOURCE_WEIGHTING_LATE
                and (
                    cost.get("phase1_resource_weighting_active")
                    is not False
                    or effective_burden != 1.0
                )
            )
            or (
                row["resource_weighting_scope"]
                == RESOURCE_WEIGHTING_ALL_PHASE
                and (
                    cost.get("phase1_resource_weighting_active")
                    is not True
                    or effective_burden != raw_burden
                )
            )
        ):
            raise PackageContractError(
                f"Phase-I cost smoke drifted: {execution_id}."
            )
        for round_index, reduction in enumerate(rounds, start=1):
            expected_requested = list(range(round_index))
            plans = reduction.get("candidate_position_plans")
            candidate_count = int(reduction.get("candidate_count", -1))
            collapsed_count = int(
                reduction.get("collapsed_position_count", -1)
            )
            retained_count = int(
                reduction.get("retained_representative_count", -1)
            )
            if (
                reduction.get("schema")
                != "commutation_reduced_insertion_domain_receipt_v1"
                or reduction.get("policy") != ALWAYS_INSERTION_KIND
                or reduction.get("domain_open") is not True
                or reduction.get("domain_state") != "open"
                or reduction.get("effective_insertion_mode")
                != ALWAYS_INSERTION_MODE
                or reduction.get("requested_positions")
                != expected_requested
                or int(reduction.get("requested_position_count", -1))
                != len(expected_requested)
                or not isinstance(plans, list)
                or len(plans) != candidate_count
                or candidate_count <= 0
                or retained_count + collapsed_count
                != candidate_count * len(expected_requested)
            ):
                raise PackageContractError(
                    f"Round-{round_index} reduced-domain closure failed."
                )
            for plan in plans:
                if not isinstance(plan, Mapping):
                    raise PackageContractError(
                        "Smoke candidate plan is malformed."
                    )
                _validate_candidate_plan(
                    plan, requested=expected_requested
                )
        second = rounds[1]
        plans = second.get("candidate_position_plans")
        if (
            not isinstance(plans, list)
            or int(second.get("collapsed_position_count", -1)) <= 0
        ):
            raise PackageContractError(
                "Second-round reduced-domain closure failed."
            )
        states = [
            _validate_candidate_plan(plan, requested=(0, 1))
            for plan in plans
            if isinstance(plan, Mapping)
        ]
        if not any(collapsed for collapsed, _ in states) or not any(
            uncollapsed for _, uncollapsed in states
        ):
            raise PackageContractError(
                "Smoke lacks collapsed and noncommuting candidates."
            )
    return dict(receipt)


__all__ = [
    "ACTIVE_GRADIENT_MEASURED",
    "ACTIVE_GRADIENT_STATIONARY",
    "ALWAYS_INSERTION_KIND",
    "ALWAYS_INSERTION_MODE",
    "ARCHIVE_SCHEMA",
    "BUNDLE_POLICIES",
    "CAMPAIGN_ID",
    "CONTROL_FILES",
    "DIRECT_EXECUTION_COUNT",
    "EXECUTION_PLAN_NAME",
    "EXECUTION_TARGET",
    "GENERATED_FILES",
    "HORIZON",
    "JOB_SCHEMA",
    "MANIFEST_SCHEMA",
    "MATERIALIZATION_RELATIVE_ROOT",
    "PACKAGE_ID",
    "PACKAGE_MANIFEST_NAME",
    "PLAN_SCHEMA",
    "PackageContractError",
    "QUEUE_NAME",
    "REMOTE_IMAGE_PATH",
    "REMOTE_IMAGE_SHA256",
    "RESOURCE_WEIGHTING_ALL_PHASE",
    "RESOURCE_WEIGHTING_LATE",
    "ROUTE_IDS",
    "RUN_CLASS",
    "SMOKE_EXECUTION_IDS",
    "SMOKE_RECEIPT_NAME",
    "SMOKE_ROUNDS",
    "SMOKE_SCHEMA",
    "SOURCE_ARCHIVE_MANIFEST_NAME",
    "SOURCE_ARCHIVE_NAME",
    "V13_FINAL_RECEIPT_RELATIVE_PATH",
    "MATERIALIZER_RELATIVE_PATH",
    "canonical_json_bytes",
    "canonical_sha256",
    "digested",
    "direct_execution_rows",
    "load_json",
    "repo_root_from_script",
    "safe_relative_path",
    "sha256_file",
    "validate_factorial_authority",
    "validate_smoke_receipt",
    "verify_self_digest",
]
