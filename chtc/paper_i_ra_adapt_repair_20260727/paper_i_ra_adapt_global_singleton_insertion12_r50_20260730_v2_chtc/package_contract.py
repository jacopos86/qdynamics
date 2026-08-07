#!/usr/bin/env python3
"""Fail-closed contract for the inert 12-cell global-singleton package."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

PACKAGE_ID = (
    "paper_i_ra_adapt_global_singleton_insertion12_r50_20260730_v1_chtc"
)
PACKAGE_RELATIVE_ROOT = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    f"{PACKAGE_ID}"
)
MATERIALIZATION_ID = "ra_adapt_global_singleton_insertion12_v1"
MATERIALIZATION_RELATIVE_ROOT = (
    "chtc/paper_i_ra_adapt_repair_20260727/bundles/materializations/"
    f"{MATERIALIZATION_ID}"
)
MATERIALIZATION_RECEIPT_NAME = (
    "global_singleton_insertion12_materialization_receipt.json"
)
MATERIALIZATION_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_global_singleton_insertion12_"
    "materialization_receipt_v1"
)
MATERIALIZER_RELATIVE_PATH = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "materialize_ra_global_singleton_insertion12_v1.py"
)
V13_FINAL_RECEIPT_RELATIVE_PATH = (
    "chtc/paper_i_ra_adapt_repair_20260727/bundles/materializations/"
    "ra_adapt_stationary_late_core_v13/final_publication_receipt.json"
)

CAMPAIGN_ID = (
    "paper_i_ra_adapt_global_singleton_insertion_comparison_v1"
)
BUNDLE_ID = (
    "ra_repair_global_singleton_stationary_all_phase_insertion_v1"
)
RUN_CLASS = "diagnostic"
EXECUTION_TARGET = "chtc"
DIRECT_EXECUTION_COUNT = 12
HORIZON = 50
OPTIMIZER = "powell"
OPTIMIZER_MAXITER = 200
SEED = 7
ACTIVE_GRADIENT_POLICY = "stationary_source_response_v1"
RESOURCE_WEIGHTING_SCOPE = "all_phase_resource_weighting_v1"
CANDIDATE_REPRESENTATION = "single_pauli_word_v1"
CANDIDATE_ADAPTER_ID = (
    "paper_i_ra_adapt_global_single_pauli_word_candidate_adapter_v1"
)
PHASE_I_CANDIDATE_SUPPLY = "global_guarded_singleton_pool_v1"
PHASE_I_CANDIDATE_VISIBILITY = "all_executable_candidates_v1"
PHASE_II_CANDIDATE_EXPOSURE = (
    "identity_on_retained_singletons_v1"
)
INSERTION_EQUIVALENCE_POLICY = (
    "termwise_cross_component_commutation_earliest_representative_v1"
)
APPEND_ROUTE_ID = "ra_global_singleton_append_commutation_reduced"
PLATEAU_ROUTE_ID = "ra_global_singleton_plateau_commutation"
ROUTE_IDS = (APPEND_ROUTE_ID, PLATEAU_ROUTE_ID)
ALGORITHM_IDS = {
    APPEND_ROUTE_ID: (
        "paper_i_ra_adapt_global_singleton_"
        "append_commutation_reduced_v1"
    ),
    PLATEAU_ROUTE_ID: (
        "paper_i_ra_adapt_global_singleton_plateau_commutation_v1"
    ),
}
INSERTION_CONTRACTS = {
    APPEND_ROUTE_ID: {
        "typed_kind": "append_commutation_reduced",
        "runtime_mode": "append_commutation_reduced",
        "position_scope": "append_endpoint_only_every_depth_v1",
        "equivalence_policy": INSERTION_EQUIVALENCE_POLICY,
    },
    PLATEAU_ROUTE_ID: {
        "typed_kind": "plateau_commutation",
        "runtime_mode": "insertion_commutation_plateau_v1",
        "position_scope": (
            "append_only_or_immediate_plateau_full_logical_domain_v1"
        ),
        "equivalence_policy": INSERTION_EQUIVALENCE_POLICY,
        "energy_decrease_threshold": 1.0e-8,
        "threshold_comparison": "strictly_below_v1",
        "patience": 1,
        "hysteresis_active": False,
    },
}
REGIME_CUTOFF_PAIRS = (
    ("weak_weak", 3),
    ("intermediate_weak", 3),
    ("strong_weak_u8", 3),
    ("weak_strong", 7),
    ("intermediate_strong", 7),
    ("strong_strong_u8", 7),
)
PARENT_INVENTORY_BY_NPH = {
    3: {
        "count": 123,
        "ordered_labels_sha256": (
            "17cc97b744f8e6b50b686b24edd28426ca2c055bc2c31054fd353ddfa10efbe3"
        ),
    },
    7: {
        "count": 171,
        "ordered_labels_sha256": (
            "389ce1382b57b916e15e170c641f3884ed1ce33e9913d6eb709f24490739e93f"
        ),
    },
}
GLOBAL_POOL_BY_NPH = {
    3: {
        "count": 948,
        "ordered_labels_sha256": (
            "02995a2c570d4322e46e55e3a532381ff7eff85dc3c2de8cb2b30ed888b76906"
        ),
    },
    7: {
        "count": 6508,
        "ordered_labels_sha256": (
            "079478057eea213139dc2f3c7486097496454421a44677c290b5dc55860accb7"
        ),
    },
}
ORDERED_POOL_SHA256_BY_REGIME = {
    "weak_weak": (
        "74880d215fd350fba57c2560eef6b6225d1caa69a7103d32624f70f6f3dfce84"
    ),
    "intermediate_weak": (
        "816dfa970a2b40e7c781f5440fcdfb33690a236f9ae853dfdd155e2f53c7e67f"
    ),
    "strong_weak_u8": (
        "62a24f68adc8a71f78fa5d3afb28356d15b988a2003e1c97e69871a65726e90c"
    ),
    "weak_strong": (
        "2b7416a82f70814e5d507ef6524bd8c8bd436c624dfc25495f7a4974188152c0"
    ),
    "intermediate_strong": (
        "078aa89647ee0449b73e3951d1c367d61a41eefeed67565ea3d8caecd81ded1a"
    ),
    "strong_strong_u8": (
        "7a0e3dacc93ef0e5af82c4f76d6956d5113844e2517723c06f26fb41a8568c59"
    ),
}
RESOURCE_ENVELOPES = {
    3: (4, 57_344, 61_440),
    7: (4, 90_112, 98_304),
}
MAX_RUNTIME_SECONDS = 72 * 60 * 60
REMOTE_IMAGE_PATH = "chtc/phase3_optuna/image.sif"
REMOTE_IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)

SMOKE_ROUNDS = 2
SMOKE_EXECUTION_IDS = (
    f"global_singleton__weak_strong__nph7__{APPEND_ROUTE_ID}",
    f"global_singleton__weak_strong__nph7__{PLATEAU_ROUTE_ID}",
)
SMOKE_RECEIPT_NAME = "two_round_semantic_preflight_receipt.json"
CALIBRATION_RECEIPT_NAME = "plateau_open_domain_calibration_receipt.json"
SOURCE_ARCHIVE_NAME = "source_locked.tar.gz"
SOURCE_ARCHIVE_MANIFEST_NAME = "source_archive_manifest.json"
EXECUTION_PLAN_NAME = "execution_plan.json"
QUEUE_NAME = "queue.tsv"
PACKAGE_MANIFEST_NAME = "package_manifest.json"
JOB_SCHEMA = "paper_i_ra_global_singleton_insertion_job_v1"
SMOKE_SCHEMA = "paper_i_ra_global_singleton_two_round_smoke_v1"
CALIBRATION_SCHEMA = "plateau_open_domain_calibration_v1"
PLAN_SCHEMA = (
    "paper_i_ra_global_singleton_insertion12_execution_plan_v1"
)
MANIFEST_SCHEMA = (
    "paper_i_ra_global_singleton_insertion12_package_v1"
)
ARCHIVE_SCHEMA = (
    "paper_i_ra_global_singleton_insertion12_source_archive_v1"
)
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
    CALIBRATION_RECEIPT_NAME,
    SOURCE_ARCHIVE_NAME,
    SOURCE_ARCHIVE_MANIFEST_NAME,
    EXECUTION_PLAN_NAME,
    QUEUE_NAME,
    PACKAGE_MANIFEST_NAME,
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PackageContractError(ValueError):
    """Raised when the global-singleton package contract does not close."""


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


def _binding(path: Path, *, repo_root: Path) -> dict[str, Any]:
    payload = load_json(path, label=f"{path.name} binding")
    return {
        "path": path.relative_to(repo_root).as_posix(),
        "sha256": sha256_file(path),
        "canonical_sha256": str(payload["sha256"]),
        "size_bytes": path.stat().st_size,
    }


def direct_execution_rows() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for regime_id, nph in REGIME_CUTOFF_PAIRS:
        for route_id in ROUTE_IDS:
            execution_id = (
                f"global_singleton__{regime_id}__nph{nph}__{route_id}"
            )
            cpus, memory_mb, disk_mb = RESOURCE_ENVELOPES[nph]
            insertion = INSERTION_CONTRACTS[route_id]
            rows.append(
                {
                    "execution_id": execution_id,
                    "cell_id": execution_id,
                    "source_lock_id": (
                        f"{regime_id}__nph{nph}__{route_id}"
                    ),
                    "regime_id": regime_id,
                    "nph": nph,
                    "route_id": route_id,
                    "algorithm_id": ALGORITHM_IDS[route_id],
                    "candidate_representation": (
                        CANDIDATE_REPRESENTATION
                    ),
                    "candidate_adapter_id": CANDIDATE_ADAPTER_ID,
                    "insertion_policy": insertion["typed_kind"],
                    "insertion_runtime_mode": (
                        insertion["runtime_mode"]
                    ),
                    "active_gradient_policy": (
                        ACTIVE_GRADIENT_POLICY
                    ),
                    "resource_weighting_scope": (
                        RESOURCE_WEIGHTING_SCOPE
                    ),
                    "phase1_cost_term": "enabled",
                    "execution_entrypoint": "run_ra_adapt",
                    "resources": {
                        "request_cpus": cpus,
                        "request_memory_mb": memory_mb,
                        "request_disk_mb": disk_mb,
                        "max_runtime_seconds": MAX_RUNTIME_SECONDS,
                        "status": (
                            "provisional_not_demonstrated_by_"
                            "bounded_calibration"
                        ),
                    },
                }
            )
    if len(rows) != DIRECT_EXECUTION_COUNT:
        raise AssertionError("Global-singleton cell count drifted.")
    return tuple(rows)


def _pool_contract(
    protocol: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
) -> None:
    parent = protocol.get("parent_inventory")
    executable = protocol.get("executable_pool")
    expected_parent = PARENT_INVENTORY_BY_NPH[int(row["nph"])]
    expected_pool = GLOBAL_POOL_BY_NPH[int(row["nph"])]
    if (
        not isinstance(parent, Mapping)
        or not isinstance(executable, Mapping)
        or int(parent.get("count", -1)) != expected_parent["count"]
        or parent.get("ordered_labels_sha256")
        != expected_parent["ordered_labels_sha256"]
        or int(executable.get("count", -1)) != expected_pool["count"]
        or executable.get("ordered_labels_sha256")
        != expected_pool["ordered_labels_sha256"]
        or executable.get("ordered_pool_sha256")
        != ORDERED_POOL_SHA256_BY_REGIME[row["regime_id"]]
    ):
        raise PackageContractError(
            f"Global-singleton pool drifted: {row['cell_id']}."
        )


def _validate_protocol(
    protocol: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
) -> None:
    verify_self_digest(protocol, label=f"{row['cell_id']} protocol")
    request = protocol.get("request")
    adapter = (
        request.get("adapter") if isinstance(request, Mapping) else None
    )
    method = (
        request.get("method") if isinstance(request, Mapping) else None
    )
    insertion = (
        method.get("insertion") if isinstance(method, Mapping) else None
    )
    admission = (
        method.get("admission") if isinstance(method, Mapping) else None
    )
    pruning = (
        method.get("pruning") if isinstance(method, Mapping) else None
    )
    beam = method.get("beam") if isinstance(method, Mapping) else None
    route_contract = protocol.get("route_contract")
    execution = (
        route_contract.get("execution_settings")
        if isinstance(route_contract, Mapping)
        else None
    )
    invariants = (
        route_contract.get("semantic_invariants")
        if isinstance(route_contract, Mapping)
        else None
    )
    lineage = protocol.get("lineage_authority")
    supply = (
        lineage.get("candidate_supply")
        if isinstance(lineage, Mapping)
        else None
    )
    materialization = protocol.get("bundle_materialization")
    expected_insertion = INSERTION_CONTRACTS[str(row["route_id"])]
    if (
        not isinstance(materialization, Mapping)
        or materialization.get("cell_id") != row["cell_id"]
        or materialization.get("source_lock_id")
        != row["source_lock_id"]
        or materialization.get("bundle_id") != BUNDLE_ID
        or materialization.get("algorithm_id")
        != row["algorithm_id"]
        or protocol.get("bundle_id") != BUNDLE_ID
        or protocol.get("algorithm_id") != row["algorithm_id"]
        or protocol.get("candidate_representation")
        != CANDIDATE_REPRESENTATION
        or protocol.get("adapter_id") != CANDIDATE_ADAPTER_ID
        or protocol.get("active_gradient_policy")
        != ACTIVE_GRADIENT_POLICY
        or protocol.get("resource_weighting_scope")
        != RESOURCE_WEIGHTING_SCOPE
        or int(protocol.get("horizon", -1)) != HORIZON
        or str(protocol.get("optimizer", "")).lower() != OPTIMIZER
        or int(protocol.get("optimizer_maxiter", -1))
        != OPTIMIZER_MAXITER
        or protocol.get("seeds")
        != {"adapt": SEED, "transpiler": SEED}
        or not isinstance(adapter, Mapping)
        or adapter.get("adapter_id") != CANDIDATE_ADAPTER_ID
        or adapter.get("candidate_representation_id")
        != CANDIDATE_REPRESENTATION
        or not isinstance(insertion, Mapping)
        or insertion.get("kind")
        != expected_insertion["typed_kind"]
        or not isinstance(admission, Mapping)
        or admission.get("kind") != "singleton"
        or not isinstance(pruning, Mapping)
        or pruning.get("kind") != "off"
        or not isinstance(beam, Mapping)
        or beam.get("kind") != "off"
        or not isinstance(execution, Mapping)
        or execution.get("adapt_insertion_mode")
        != expected_insertion["runtime_mode"]
        or int(execution.get("phase1_shortlist_size", -1)) != 24
        or int(execution.get("phase2_shortlist_size", -1)) != 12
        or not isinstance(invariants, Mapping)
        or invariants.get("active_gradient_policy")
        != ACTIVE_GRADIENT_POLICY
        or invariants.get("resource_weighting_scope")
        != RESOURCE_WEIGHTING_SCOPE
        or int(invariants.get("admission_cardinality", -1)) != 1
        or invariants.get("online_exact_reference_used", False)
        is not False
        or invariants.get("compatibility_resolution_active", False)
        is not False
        or invariants.get("insertion_position_scope")
        != expected_insertion["position_scope"]
        or invariants.get("insertion_equivalence_policy")
        != expected_insertion["equivalence_policy"]
        or not isinstance(supply, Mapping)
        or supply.get("candidate_adapter_id")
        != CANDIDATE_ADAPTER_ID
        or supply.get("phase_i_candidate_supply")
        != PHASE_I_CANDIDATE_SUPPLY
        or supply.get("phase_i_candidate_visibility")
        != PHASE_I_CANDIDATE_VISIBILITY
        or supply.get("phase_ii_candidate_exposure")
        != PHASE_II_CANDIDATE_EXPOSURE
        or protocol.get("execution_authorized") is not False
    ):
        raise PackageContractError(
            f"Global-singleton protocol drifted: {row['cell_id']}."
        )
    _pool_contract(protocol, row=row)


def _normalize_insertion_axis(
    value: Any,
    *,
    row: Mapping[str, Any],
) -> Any:
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if key == "sha256" or key.endswith("_sha256"):
                continue
            if (
                key == "insertion"
                or key == "adapt_insertion_mode"
                or key == "canonical_insertion_policy"
                or key.startswith("insertion_")
                or key.startswith("plateau_")
                or key == "parent_route_profile"
                or key
                in {
                    "canonical_admission_policy",
                    "canonical_beam_policy",
                    "canonical_composition_schema",
                    "canonical_pruning_policy",
                    "compatibility_resolution_active",
                    "diagnostic_position_ablation",
                    "online_exact_reference_used",
                }
            ):
                continue
            if key == "settled_change_ids" and isinstance(item, list):
                normalized[str(key)] = [
                    value
                    for value in item
                    if value
                    != "global_singleton_insertion_policy_variant"
                ]
                continue
            if key in {
                "route_id",
                "algorithm_id",
                "cell_id",
                "source_lock_id",
            }:
                normalized[str(key)] = f"<{key}>"
                continue
            normalized[str(key)] = _normalize_insertion_axis(
                item, row=row
            )
        return normalized
    if isinstance(value, list):
        return [
            _normalize_insertion_axis(item, row=row) for item in value
        ]
    if isinstance(value, str):
        result = value
        replacements = (
            (str(row["cell_id"]), "<cell_id>"),
            (str(row["source_lock_id"]), "<source_lock_id>"),
            (str(row["route_id"]), "<route_id>"),
            (str(row["algorithm_id"]), "<algorithm_id>"),
            ("append_commutation_reduced", "<insertion_axis>"),
            ("insertion_commutation_plateau_v1", "<insertion_axis>"),
            ("plateau_commutation", "<insertion_axis>"),
            (
                "append_endpoint_only_every_depth_v1",
                "<insertion_position_scope>",
            ),
            (
                "append_only_or_immediate_plateau_full_logical_domain_v1",
                "<insertion_position_scope>",
            ),
        )
        for before, after in replacements:
            result = result.replace(before, after)
        return result
    return value


def validate_materialization_authority(
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    materialization_root = root / MATERIALIZATION_RELATIVE_ROOT
    final_path = (
        materialization_root / MATERIALIZATION_RECEIPT_NAME
    )
    final = load_json(
        final_path, label="global-singleton materialization receipt"
    )
    verify_self_digest(
        final, label="global-singleton materialization receipt"
    )
    false_fields = (
        "execution_authorized",
        "submission_authorized",
        "remote_stage",
        "condor_submit",
    )
    if (
        final.get("schema") != MATERIALIZATION_RECEIPT_SCHEMA
        or final.get("status") != "passed"
        or final.get("materialization_id") != MATERIALIZATION_ID
        or final.get("campaign_id") != CAMPAIGN_ID
        or final.get("bundle_id") != BUNDLE_ID
        or final.get("run_class") != RUN_CLASS
        or int(final.get("cell_count", -1))
        != DIRECT_EXECUTION_COUNT
        or final.get("submission_state") != "not_submitted"
        or final.get("submitted") is not False
        or any(final.get(field) is not False for field in false_fields)
    ):
        raise PackageContractError(
            "Global-singleton materialization receipt drifted."
        )

    delta_path = (
        materialization_root
        / "source_materialization/source_lock_delta_receipt.json"
    )
    delta = load_json(
        delta_path, label="global-singleton source-lock delta"
    )
    verify_self_digest(
        delta, label="global-singleton source-lock delta"
    )
    delta_binding = _binding(delta_path, repo_root=root)
    final_delta = final.get("source_anchor", {}).get(
        "source_lock_delta_receipt"
    )
    if (
        delta.get("schema")
        != (
            "paper_i_ra_adapt_global_singleton_insertion_"
            "source_lock_delta_v1"
        )
        or delta.get("status") != "passed"
        or int(delta.get("source_cell_count", -1)) != 6
        or int(delta.get("derived_cell_count", -1)) != 12
        or delta.get("all_archive_bindings_preserved") is not True
        or delta.get("all_member_bindings_preserved") is not True
        or delta.get("all_global_source_bindings_preserved") is not True
        or delta.get("execution_authorized") is not False
        or delta.get("submission_authorized") is not False
        or delta.get("submitted") is not False
        or not isinstance(final_delta, Mapping)
        or final_delta.get("path")
        != (
            "source_materialization/"
            "source_lock_delta_receipt.json"
        )
        or final_delta.get("sha256")
        != delta_binding["sha256"]
        or final_delta.get("canonical_sha256")
        != delta_binding["canonical_sha256"]
        or int(final_delta.get("size_bytes", -1))
        != int(delta_binding["size_bytes"])
    ):
        raise PackageContractError(
            "Global-singleton source-lock delta drifted."
        )

    bundle_root = materialization_root / BUNDLE_ID
    manifest_path = bundle_root / "bundle_manifest.json"
    locks_path = bundle_root / "source_locks.json"
    validation_path = bundle_root / "validation_report.json"
    expected_path = bundle_root / "expected_artifacts.json"
    manifest = load_json(manifest_path, label="bundle manifest")
    source_locks = load_json(locks_path, label="source locks")
    validation = load_json(validation_path, label="validation report")
    expected = load_json(expected_path, label="expected artifacts")
    for payload, label in (
        (manifest, "bundle manifest"),
        (source_locks, "source locks"),
        (validation, "validation report"),
        (expected, "expected artifacts"),
    ):
        verify_self_digest(payload, label=label)
    comparison = manifest.get("global_singleton_insertion_contract")
    if (
        manifest.get("bundle_id") != BUNDLE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("run_class") != RUN_CLASS
        or manifest.get("active_gradient_policy")
        != ACTIVE_GRADIENT_POLICY
        or manifest.get("resource_weighting_scope")
        != RESOURCE_WEIGHTING_SCOPE
        or manifest.get("stationarity_condition")
        != "always_applied_v1"
        or manifest.get("phase1_cost_term")
        != "always_applied_v1"
        or int(manifest.get("cell_count", -1))
        != DIRECT_EXECUTION_COUNT
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_state") != "not_submitted"
        or manifest.get("submitted") is not False
        or validation.get("materialization_status") != "passed"
        or validation.get("execution_authorized") is not False
        or not isinstance(comparison, Mapping)
        or comparison.get("candidate_adapter_id")
        != CANDIDATE_ADAPTER_ID
        or comparison.get("phase_i_candidate_supply")
        != PHASE_I_CANDIDATE_SUPPLY
        or comparison.get("phase_i_candidate_visibility")
        != PHASE_I_CANDIDATE_VISIBILITY
        or comparison.get("phase_ii_candidate_exposure")
        != PHASE_II_CANDIDATE_EXPOSURE
        or int(comparison.get("phase_i_shortlist_size", -1)) != 24
        or int(comparison.get("phase_ii_shortlist_size", -1)) != 12
        or int(
            comparison.get("phase_iii_admission_cardinality", -1)
        )
        != 1
        or comparison.get("global_executable_pool_membership_by_nph")
        != {str(key): value for key, value in GLOBAL_POOL_BY_NPH.items()}
        or comparison.get("ordered_pool_sha256_by_regime")
        != ORDERED_POOL_SHA256_BY_REGIME
    ):
        raise PackageContractError(
            "Global-singleton bundle envelope drifted."
        )

    implementation = source_locks.get("implementation_sources")
    cell_locks = source_locks.get("cell_locks")
    global_sources = source_locks.get("global_sources")
    if (
        not isinstance(implementation, Mapping)
        or not isinstance(cell_locks, Mapping)
        or len(cell_locks) != DIRECT_EXECUTION_COUNT
        or not isinstance(global_sources, Mapping)
    ):
        raise PackageContractError(
            "Global-singleton source locks drifted."
        )
    verify_self_digest(
        implementation, label="implementation inventory"
    )

    rows = direct_execution_rows()
    protocol_bindings: dict[str, dict[str, Any]] = {}
    artifact_destinations: dict[str, list[str]] = {
        "result": [],
        "checkpoint": [],
        "estimator_ledger": [],
    }
    destinations_by_id: dict[str, dict[str, str]] = {}
    normalized_by_regime: dict[str, list[Any]] = {}
    for row in rows:
        protocol_path = (
            bundle_root / "protocols" / f"{row['cell_id']}.json"
        )
        protocol = load_json(
            protocol_path, label=f"{row['cell_id']} protocol"
        )
        _validate_protocol(protocol, row=row)
        protocol_bindings[str(row["execution_id"])] = _binding(
            protocol_path, repo_root=root
        )
        normalized_by_regime.setdefault(
            str(row["regime_id"]), []
        ).append(_normalize_insertion_axis(protocol, row=row))
        expected_cell = expected.get("cells", {}).get(row["cell_id"])
        roles = (
            expected_cell.get("expected_run_artifacts")
            if isinstance(expected_cell, Mapping)
            else None
        )
        if not isinstance(roles, Mapping):
            raise PackageContractError(
                f"Expected artifacts missing: {row['cell_id']}."
            )
        destinations: dict[str, str] = {}
        for role in artifact_destinations:
            artifact = roles.get(role)
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
                    f"Artifact destination drifted: "
                    f"{row['cell_id']}.{role}."
                )
            artifact_destinations[role].append(path_text)
            destinations[role] = path_text
        destinations_by_id[str(row["execution_id"])] = destinations

    equality_rows: list[dict[str, Any]] = []
    for regime_id, variants in sorted(normalized_by_regime.items()):
        if len(variants) != 2 or variants[0] != variants[1]:
            raise PackageContractError(
                "Insertion arms differ outside the declared axis: "
                f"{regime_id}."
            )
        equality_rows.append(
            {
                "regime_id": regime_id,
                "normalized_common_sha256": canonical_sha256(
                    variants[0]
                ),
                "status": "passed",
            }
        )
    for role, paths in artifact_destinations.items():
        if (
            len(paths) != DIRECT_EXECUTION_COUNT
            or len(set(paths)) != DIRECT_EXECUTION_COUNT
        ):
            raise PackageContractError(
                f"{role} destinations are not unique."
            )
    return {
        "materialization_root": materialization_root,
        "final": final,
        "final_binding": _binding(final_path, repo_root=root),
        "source_lock_delta": delta,
        "source_lock_delta_binding": delta_binding,
        "manifest": manifest,
        "bundle_bindings": {
            "manifest": _binding(manifest_path, repo_root=root),
            "source_locks": _binding(locks_path, repo_root=root),
            "validation_report": _binding(
                validation_path, repo_root=root
            ),
            "expected_artifacts": _binding(
                expected_path, repo_root=root
            ),
        },
        "source_locks": source_locks,
        "implementation_inventory": implementation,
        "protocol_bindings": protocol_bindings,
        "artifact_destinations": artifact_destinations,
        "artifact_destinations_by_execution_id": (
            destinations_by_id
        ),
        "equality_audit": digested(
            {
                "schema": (
                    "paper_i_ra_global_singleton_"
                    "insertion_cross_arm_equality_v1"
                ),
                "status": "passed",
                "allowed_axis": "insertion_policy",
                "regime_pair_count": len(equality_rows),
                "variant_count": DIRECT_EXECUTION_COUNT,
                "rows": equality_rows,
            }
        ),
        "rows": rows,
    }


def _validate_reduction(
    receipt: Mapping[str, Any],
    *,
    route_id: str,
    append_position: int,
) -> None:
    expected = INSERTION_CONTRACTS[route_id]
    domain_open = bool(receipt.get("domain_open"))
    if route_id == APPEND_ROUTE_ID and domain_open:
        raise PackageContractError(
            "Append-reduced smoke unexpectedly opened its domain."
        )
    requested = (
        list(range(append_position + 1))
        if domain_open
        else [append_position]
    )
    expected_schema = (
        "commutation_reduced_insertion_domain_receipt_v1"
        if route_id == APPEND_ROUTE_ID
        else "insertion_commutation_plateau_round_policy_v1"
    )
    expected_effective_mode = (
        "append_commutation_reduced"
        if route_id == APPEND_ROUTE_ID
        else (
            "full_commutation_reduced"
            if domain_open
            else "append_only"
        )
    )
    if (
        receipt.get("schema") != expected_schema
        or receipt.get("policy") != expected["runtime_mode"]
        or receipt.get("domain_state")
        != ("open" if domain_open else "closed")
        or receipt.get("effective_insertion_mode")
        != expected_effective_mode
        or receipt.get("requested_positions") != requested
        or int(receipt.get("requested_position_count", -1))
        != len(requested)
        or int(receipt.get("retained_representative_count", -1))
        + int(receipt.get("collapsed_position_count", -1))
        != int(receipt.get("candidate_count", -2)) * len(requested)
    ):
        raise PackageContractError(
            "Two-round endpoint reduction receipt drifted."
        )
    if (
        route_id == APPEND_ROUTE_ID
        and int(receipt.get("append_position", -1))
        != append_position
    ):
        raise PackageContractError(
            "Append-reduced endpoint binding drifted."
        )
    if route_id == PLATEAU_ROUTE_ID:
        decrease = receipt.get("trigger_energy_decrease")
        expected_open = bool(
            decrease is not None and float(decrease) < 1.0e-8
        )
        if (
            domain_open != expected_open
            or receipt.get("energy_decrease_threshold") != 1.0e-8
            or receipt.get("threshold_comparison") != "strictly_below"
            or receipt.get("patience") != 1
            or receipt.get("hysteresis_active") is not False
        ):
            raise PackageContractError(
                "Plateau smoke trigger semantics drifted."
            )


def validate_smoke_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    verify_self_digest(receipt, label="two-round semantic preflight")
    observations = receipt.get("observations")
    if (
        receipt.get("schema") != SMOKE_SCHEMA
        or receipt.get("package_id") != PACKAGE_ID
        or receipt.get("status") != "passed"
        or receipt.get("maximum_controller_rounds") != SMOKE_ROUNDS
        or receipt.get("scientific_result") is not False
        or receipt.get("execution_evidence") is not False
        or receipt.get("paper_evidence_allowed") is not False
        or receipt.get("execution_authorized") is not False
        or receipt.get("submission_authorized") is not False
        or not isinstance(observations, list)
        or [row.get("execution_id") for row in observations]
        != list(SMOKE_EXECUTION_IDS)
    ):
        raise PackageContractError(
            "Two-round semantic preflight envelope drifted."
        )
    rows_by_id = {
        row["execution_id"]: row for row in direct_execution_rows()
    }
    for observation in observations:
        row = rows_by_id[str(observation["execution_id"])]
        rounds = observation.get("accepted_round_insertion_receipts")
        if (
            observation.get("controller_round_count") != SMOKE_ROUNDS
            or observation.get("global_pool_count") != 6508
            or observation.get("global_pool_ordered_labels_sha256")
            != GLOBAL_POOL_BY_NPH[7]["ordered_labels_sha256"]
            or observation.get("global_pool_ordered_pool_sha256")
            != ORDERED_POOL_SHA256_BY_REGIME[row["regime_id"]]
            or observation.get("active_gradient_policy")
            != ACTIVE_GRADIENT_POLICY
            or observation.get("resource_weighting_scope")
            != RESOURCE_WEIGHTING_SCOPE
            or not isinstance(rounds, list)
            or len(rounds) != SMOKE_ROUNDS
        ):
            raise PackageContractError(
                f"Two-round smoke drifted: {row['execution_id']}."
            )
        for round_index, reduction in enumerate(rounds):
            if not isinstance(reduction, Mapping):
                raise PackageContractError(
                    "Two-round smoke lost an insertion receipt."
                )
            _validate_reduction(
                reduction,
                route_id=str(row["route_id"]),
                append_position=round_index,
            )
    return dict(receipt)


def validate_calibration_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    verify_self_digest(receipt, label="open-plateau calibration")
    domain = receipt.get("open_domain_receipt")
    resources = receipt.get("resource_observation")
    if (
        receipt.get("schema") != CALIBRATION_SCHEMA
        or receipt.get("package_id") != PACKAGE_ID
        or receipt.get("status") != "passed"
        or receipt.get("scientific_result") is not False
        or receipt.get("execution_evidence") is not False
        or receipt.get("checkpoint_emitted") is not False
        or receipt.get("result_promotable") is not False
        or receipt.get("synthetic_trigger_only") is not True
        or receipt.get("nph") != 7
        or receipt.get("candidate_count") != 6508
        or receipt.get("requested_positions") != [0, 1]
        or receipt.get("precollapse_candidate_position_pair_count")
        != 13_016
        or not isinstance(domain, Mapping)
        or domain.get("schema")
        != "insertion_commutation_plateau_round_policy_v1"
        or domain.get("policy")
        != "insertion_commutation_plateau_v1"
        or domain.get("domain_open") is not True
        or domain.get("domain_state") != "open"
        or domain.get("effective_insertion_mode")
        != "full_commutation_reduced"
        or domain.get("energy_decrease_threshold") != 1.0e-8
        or domain.get("threshold_comparison") != "strictly_below"
        or domain.get("patience") != 1
        or domain.get("hysteresis_active") is not False
        or domain.get("requested_positions") != [0, 1]
        or int(domain.get("candidate_count", -1)) != 6508
        or int(domain.get("retained_representative_count", -1))
        + int(domain.get("collapsed_position_count", -1))
        != 13_016
        or not isinstance(resources, Mapping)
        or int(resources.get("peak_rss_bytes", 0)) <= 0
        or float(resources.get("elapsed_seconds", 0.0)) <= 0.0
        or int(resources.get("serialized_receipt_bytes", 0)) <= 0
        or receipt.get("package_resources_demonstrated") is not False
        or receipt.get("package_resource_status")
        != "provisional_not_demonstrated"
    ):
        raise PackageContractError(
            "Open-plateau calibration receipt drifted."
        )
    return dict(receipt)


__all__ = [
    "ACTIVE_GRADIENT_POLICY",
    "APPEND_ROUTE_ID",
    "ARCHIVE_SCHEMA",
    "BUNDLE_ID",
    "CALIBRATION_RECEIPT_NAME",
    "CALIBRATION_SCHEMA",
    "CAMPAIGN_ID",
    "CANDIDATE_ADAPTER_ID",
    "CONTROL_FILES",
    "DIRECT_EXECUTION_COUNT",
    "EXECUTION_PLAN_NAME",
    "EXECUTION_TARGET",
    "GENERATED_FILES",
    "GLOBAL_POOL_BY_NPH",
    "HORIZON",
    "JOB_SCHEMA",
    "MANIFEST_SCHEMA",
    "MATERIALIZATION_RECEIPT_NAME",
    "MATERIALIZATION_RELATIVE_ROOT",
    "MATERIALIZER_RELATIVE_PATH",
    "ORDERED_POOL_SHA256_BY_REGIME",
    "PACKAGE_ID",
    "PACKAGE_MANIFEST_NAME",
    "PLAN_SCHEMA",
    "PLATEAU_ROUTE_ID",
    "PackageContractError",
    "QUEUE_NAME",
    "REGIME_CUTOFF_PAIRS",
    "REMOTE_IMAGE_PATH",
    "REMOTE_IMAGE_SHA256",
    "RESOURCE_WEIGHTING_SCOPE",
    "ROUTE_IDS",
    "RUN_CLASS",
    "SMOKE_EXECUTION_IDS",
    "SMOKE_RECEIPT_NAME",
    "SMOKE_ROUNDS",
    "SMOKE_SCHEMA",
    "SOURCE_ARCHIVE_MANIFEST_NAME",
    "SOURCE_ARCHIVE_NAME",
    "V13_FINAL_RECEIPT_RELATIVE_PATH",
    "canonical_json_bytes",
    "canonical_sha256",
    "digested",
    "direct_execution_rows",
    "load_json",
    "repo_root_from_script",
    "safe_relative_path",
    "sha256_file",
    "validate_calibration_receipt",
    "validate_materialization_authority",
    "validate_smoke_receipt",
    "verify_self_digest",
]
