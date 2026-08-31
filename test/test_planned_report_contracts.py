"""Predeclared report-slot contracts for future Paper-II benchmark lanes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import pytest

from pipelines.time_dynamics.planned_report_contracts import (
    DEFAULT_REPORT_PATHS,
    PlannedReportContractError,
    build_planned_report_contract,
    update_exact_reference,
    update_method_slot,
    validate_planned_report_contract,
    write_default_planned_report_contracts,
)
from pipelines.time_dynamics.acquisition_supervisor import (
    CampaignRegistry,
    LaneRegistration,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    EXACT_REFERENCE_DATA_V1,
    EXACT_REFERENCE_MANIFEST_V1,
    EXACT_REFERENCE_RECEIPT_V1,
    RUN_STATE_EVIDENCE_V1,
    SAFEGUARD_QUALIFICATION_RECEIPT_V1,
    SCORE_METRICS_V1,
    SCORE_QUALIFICATION_RECEIPT_V1,
    SEED_ARTIFACT_V1,
    SEED_QUALIFICATION_RECEIPT_V1,
    SOURCE_EPOCH_RECEIPT_V1,
    SOURCE_TREE_MANIFEST_V1,
    ExactReferenceReceipt,
    ExecutionRequest,
    RunLifecycleReceipt,
    SafeguardQualificationReceipt,
    ScoreQualificationReceipt,
    ScientificPreSubmitAudit,
    SourceEpochReceipt,
    SeedQualificationReceipt,
    VerifiedFile,
    build_package_execution_request,
    build_scientific_bindings,
    load_scientific_state,
    resolve_scientific_action,
    update_report_slot_execution_request,
    write_action_registry_entry,
    write_run_lifecycle_receipt,
    write_scientific_pre_submit_audit,
    write_scientific_state,
)
from pipelines.time_dynamics.parallel_benchmark_matrix import ParallelBenchmarkMatrix
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    SourceEpochIdentity,
    build_identity_registry,
)
from pipelines.time_dynamics.prospective_scientific_planner import (
    ARM_APM,
    ARM_AVQDS,
    HORIZON_FULL,
    HORIZON_PREFIX,
    KIND_RECORD_SAFEGUARD,
    KIND_RUN,
    ProspectiveNextAction,
    ProspectiveScientificPlan,
    SAFEGUARD_LADDER_UNRESOLVED,
    prospective_run_action_id,
    prospective_safeguard_id,
)
from pipelines.time_dynamics.staged_campaign import AccuracyPolicy


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = Path(
    "pipelines/time_dynamics/campaigns/"
    "paper_ii_parallel_benchmark_matrix_v1.json"
)

_AUDIT_CAMPAIGN_REGISTRIES: dict[
    tuple[str, str], CampaignRegistry
] = {}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _canonical_sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()


def _sealed(payload: dict, key: str) -> dict:
    result = dict(payload)
    result.pop(key, None)
    result[key] = _canonical_sha(result)
    return result


def _campaign_registry_for_audit(
    audit: ScientificPreSubmitAudit,
) -> CampaignRegistry:
    return _AUDIT_CAMPAIGN_REGISTRIES[
        (audit.receipt_path, audit.receipt_sha256)
    ]


def _copy_matrix(root: Path) -> Path:
    target = root / MATRIX_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((REPOSITORY_ROOT / MATRIX_PATH).read_bytes())
    return target


def _policy(drive: str) -> AccuracyPolicy:
    scales = {drive: _reference_response_scales()}
    return AccuracyPolicy.response_normalized_matched_pair(
        response_scales=scales,
        response_relative_target=0.05,
        response_relative_guard=0.15,
        matched_accuracy_factor=2.0,
        matched_component_deadband=0.10,
        infidelity_ceiling=1.0e-4,
        absolute_energy_error_ceiling=1.0e-2,
        absolute_doublon_error_ceiling=5.0e-3,
        absolute_site_occupation_error_ceiling=5.0e-3,
    )


def _reference_response_scales() -> dict[str, dict[str, float]]:
    return {
        horizon: {
            "energy_rms_excursion": 0.2,
            "doublon_rms_excursion": 0.1,
            "max_site_occupation_rms_excursion": 0.1,
        }
        for horizon in ("t5", HORIZON_FULL)
    }


def _prospective_exact_data(reference, *, materialized_key: str) -> dict:
    grid = reference.grid
    points = int(grid["points"])
    prefix_points = int(grid["prefix_points"])
    site_count = int(reference.physical["lattice_sites"])
    times = [
        float(grid["start"]) + index * float(grid["step"])
        for index in range(points)
    ]

    def _series(scale: float) -> list[float]:
        prefix_value = scale * math.sqrt(
            float(prefix_points) / float(prefix_points - 1)
        )
        return [0.0, *([prefix_value] * (prefix_points - 1)), *(
            [scale] * (points - prefix_points)
        )]

    energy = _series(0.2)
    doublon = _series(0.1)
    occupation = _series(0.1)
    occupation_rows = [
        [value, -value, *([0.0] * (site_count - 2))]
        for value in occupation
    ]
    return {
        "schema": EXACT_REFERENCE_DATA_V1,
        "reference_identity_sha256": reference.identity_sha256,
        "materialized_reference_key": materialized_key,
        "observable_schema": reference.observable_schema,
        "ordered_times_sha256": grid["ordered_times_sha256"],
        "time": times,
        "energy": energy,
        "doublon": doublon,
        "site_occupations": occupation_rows,
        "statevector_real": [[1.0, 0.0] for _ in range(points)],
        "statevector_imag": [[0.0, 0.0] for _ in range(points)],
    }


def _registered_context(root: Path, cell_id: str, *, token: str = "one") -> dict:
    matrix = ParallelBenchmarkMatrix.read(root / MATRIX_PATH)
    identities = build_identity_registry(matrix)
    reference_identity = next(
        row for row in identities.references if row.cell_id == cell_id
    )
    seed_identity = next(
        row for row in identities.seeds if row.seed_id == reference_identity.seed_id
    )
    assert seed_identity.initializer is not None
    assert seed_identity.restarts is not None
    assert seed_identity.optimizer_seed is not None
    source_epoch = SourceEpochIdentity(
        lane_id=reference_identity.lane_id,
        source_epoch_id=f"{reference_identity.lane_id}-test-source",
        source_revision="a" * 40,
    )
    source_manifest = root / f"evidence/{token}_source_manifest.json"
    _write_json(
        source_manifest,
        _sealed(
            {
                "schema": SOURCE_TREE_MANIFEST_V1,
                "lane_id": source_epoch.lane_id,
                "source_epoch_identity_sha256": source_epoch.identity_sha256,
                "source_revision": source_epoch.source_revision,
                "source_archive_sha256": source_epoch.source_archive_sha256,
                "source_tree_sha256": "d" * 64,
                "provenance_kind": "git_archive",
            },
            "source_tree_manifest_sha256",
        ),
    )
    source_path = root / f"receipts/{token}_source.json"
    _write_json(
        source_path,
        _sealed(
            {
                "schema": SOURCE_EPOCH_RECEIPT_V1,
                "source_epoch": source_epoch.to_json_dict(),
                "source_tree_manifest": {
                    "path": source_manifest.relative_to(root).as_posix(),
                    "sha256": _sha(source_manifest),
                },
                "source_tree_sha256": "d" * 64,
            },
            "source_epoch_receipt_sha256",
        ),
    )
    source_receipt = SourceEpochReceipt.load(
        source_path,
        repository_root=root,
        expected_epoch=source_epoch,
    )

    exact_energy = -1.25
    energy_error = 5.0e-7
    variational_energy = exact_energy + energy_error
    optimizer_execution_id = f"optimizer::{seed_identity.seed_id}::{token}"
    seed_artifact = root / f"evidence/{token}_seed_artifact.json"
    _write_json(
        seed_artifact,
        _sealed(
            {
                "schema": SEED_ARTIFACT_V1,
                "seed_id": seed_identity.seed_id,
                "seed_identity_sha256": seed_identity.identity_sha256,
                "lane_id": seed_identity.lane_id,
                "model": seed_identity.model,
                "physical": dict(seed_identity.physical),
                "seed_policy": seed_identity.policy,
                "initializer": seed_identity.initializer,
                "restarts": seed_identity.restarts,
                "optimizer_seed": seed_identity.optimizer_seed,
                "optimizer_execution_id": optimizer_execution_id,
                "source_epoch_identity_sha256": source_epoch.identity_sha256,
                "variational_energy": variational_energy,
                "exact_energy": exact_energy,
                "exact_state_infidelity": 2.0e-9,
            },
            "seed_artifact_sha256",
        ),
    )
    seed_path = root / f"receipts/{token}_seed.json"
    _write_json(
        seed_path,
        _sealed(
            {
                "schema": SEED_QUALIFICATION_RECEIPT_V1,
                "seed_id": seed_identity.seed_id,
                "seed_identity_sha256": seed_identity.identity_sha256,
                "artifact": {
                    "path": seed_artifact.relative_to(root).as_posix(),
                    "sha256": _sha(seed_artifact),
                },
                "energy_error": energy_error,
                "exact_state_infidelity": 2.0e-9,
                "initializer": seed_identity.initializer,
                "restarts": seed_identity.restarts,
                "optimizer_seed": seed_identity.optimizer_seed,
                "optimizer_execution_id": optimizer_execution_id,
                "variational_energy": variational_energy,
                "exact_energy": exact_energy,
                "source_epoch_identity_sha256": source_epoch.identity_sha256,
            },
            "seed_receipt_sha256",
        ),
    )
    seed_receipt = SeedQualificationReceipt.load(
        seed_path,
        repository_root=root,
        expected_seed=seed_identity,
        expected_source_epoch=source_epoch,
    )

    reference_contract = {
        "reference_id": reference_identity.reference_id,
        "reference_identity_sha256": reference_identity.identity_sha256,
        "lane_id": reference_identity.lane_id,
        "cell_id": reference_identity.cell_id,
        "seed_id": reference_identity.seed_id,
        "seed_identity_sha256": seed_identity.identity_sha256,
        "model": reference_identity.model,
        "physical": dict(reference_identity.physical),
        "drive": dict(reference_identity.drive),
        "grid": dict(reference_identity.grid),
        "propagator": reference_identity.propagator,
        "observable_schema": reference_identity.observable_schema,
    }
    materialized_key = reference_identity.materialized_key(seed_receipt.artifact.sha256)
    data = root / f"evidence/{token}_exact.json"
    _write_json(
        data,
        _prospective_exact_data(
            reference_identity,
            materialized_key=materialized_key,
        ),
    )
    response_scales = _reference_response_scales()
    response_scales_sha256 = _canonical_sha(response_scales)
    manifest = root / f"evidence/{token}_exact_manifest.json"
    _write_json(
        manifest,
        _sealed(
            {
                "schema": EXACT_REFERENCE_MANIFEST_V1,
                "reference_contract": reference_contract,
                "materialized_reference_key": materialized_key,
                "source_epoch_identity_sha256": source_epoch.identity_sha256,
                "seed_artifact_sha256": seed_receipt.artifact.sha256,
                "data_sha256": _sha(data),
                "response_scales": response_scales,
                "response_scales_sha256": response_scales_sha256,
            },
            "exact_manifest_sha256",
        ),
    )
    reference_path = root / f"receipts/{token}_reference.json"
    _write_json(
        reference_path,
        _sealed(
            {
                "schema": EXACT_REFERENCE_RECEIPT_V1,
                "reference_id": reference_identity.reference_id,
                "reference_identity_sha256": reference_identity.identity_sha256,
                "lane_id": reference_identity.lane_id,
                "cell_id": reference_identity.cell_id,
                "seed_id": seed_identity.seed_id,
                "seed_artifact_sha256": seed_receipt.artifact.sha256,
                "manifest": {
                    "path": manifest.relative_to(root).as_posix(),
                    "sha256": _sha(manifest),
                },
                "data": {
                    "path": data.relative_to(root).as_posix(),
                    "sha256": _sha(data),
                },
                "source_epoch_identity_sha256": source_epoch.identity_sha256,
                "materialized_reference_key": materialized_key,
                "reference_contract": reference_contract,
                "response_scales": response_scales,
                "response_scales_sha256": response_scales_sha256,
            },
            "exact_reference_receipt_sha256",
        ),
    )
    reference_receipt = ExactReferenceReceipt.load(
        reference_path,
        repository_root=root,
        expected_reference=reference_identity,
        expected_seed_identity=seed_identity,
        expected_seed_receipt=seed_receipt,
        expected_source_epoch=source_epoch,
    )
    policy = _policy(str(reference_identity.drive["drive_id"]))
    bindings = build_scientific_bindings(
        reference=reference_identity,
        seed=seed_identity,
        seed_receipt=seed_receipt,
        reference_receipt=reference_receipt,
        source_epoch=source_epoch,
        source_epoch_receipt=source_receipt,
        policy=policy,
        matrix=matrix,
    )
    return {
        "matrix": matrix,
        "identities": identities,
        "seed_identity": seed_identity,
        "reference_identity": reference_identity,
        "source_epoch": source_epoch,
        "source_receipt": source_receipt,
        "seed_receipt": seed_receipt,
        "reference_receipt": reference_receipt,
        "policy": policy,
        "bindings": bindings,
    }


def _audit(
    root: Path,
    contract: dict,
    context: dict,
    *,
    token: str,
    action_entries: tuple[VerifiedFile, ...] = (),
    run_receipts: tuple[RunLifecycleReceipt, ...] = (),
    score_receipts: tuple[ScoreQualificationReceipt, ...] = (),
    safeguard_receipts: tuple[SafeguardQualificationReceipt, ...] = (),
    warnings: tuple[str, ...] = (),
    audit_scope: str = "update_report_slot",
) -> ScientificPreSubmitAudit:
    bindings = context["bindings"]
    state_path = root / f"state/{token}_audit_scientific_state.json"
    write_scientific_state(
        state_path,
        bindings=bindings,
        action_entries=action_entries,
        run_receipts=run_receipts,
        score_receipts=score_receipts,
        safeguard_receipts=safeguard_receipts,
        repository_root=root,
        expected_seed=context["seed_identity"],
        expected_reference=context["reference_identity"],
        expected_source_epoch=context["source_epoch"],
        warnings=warnings,
    )
    state = load_scientific_state(
        state_path,
        repository_root=root,
        expected_seed=context["seed_identity"],
        expected_reference=context["reference_identity"],
        expected_source_epoch=context["source_epoch"],
    )
    contract_token = contract["report_contract_sha256"][:12]
    contract_path = (
        root / f"state/{token}_{contract_token}_audit_report_contract.json"
    )
    _write_json(contract_path, contract)
    contract_file = VerifiedFile(
        contract_path.relative_to(root).as_posix(), _sha(contract_path)
    )
    matrix = ParallelBenchmarkMatrix.read(root / MATRIX_PATH)
    lane = LaneRegistration(
        lane_id=bindings.lane_id,
        kind="paper_ii_prospective_test",
        monitor_enabled=False,
        submission_enabled=False,
        attempts_root=root / f"attempts/{token}",
        state_root=root / f"state/{token}_supervisor",
        scientific_state=state_path,
        report_contract=contract_path,
        planner_authority=(
            "pipelines.time_dynamics.prospective_scientific_planner:"
            "plan_prospective_cell"
        ),
    )
    campaign_registry = CampaignRegistry(
        registry_id=f"planned_report_test_{token}",
        repository_root=root,
        supervisor_state_root=root / f"state/{token}_campaign",
        lanes=(lane,),
        scientific_contracts={
            "matrix": {
                "raw_sha256": matrix.raw_sha256,
                "payload_sha256": matrix.payload_sha256,
                "expanded_semantic_sha256": (
                    matrix.expanded_semantic_sha256
                ),
                "accuracy_policy_spec_sha256": (
                    matrix.accuracy_policy_spec_sha256
                ),
            },
            "planned_report_contracts": {
                bindings.lane_id: {
                    "lane_id": bindings.lane_id,
                    "path": contract_file.path,
                    "file_sha256": contract_file.sha256,
                    "report_contract_sha256": contract[
                        "report_contract_sha256"
                    ],
                    "tuple_structure_sha256": contract[
                        "tuple_structure_sha256"
                    ],
                }
            },
        },
        production_operations=None,
    )
    request_factory = (
        build_package_execution_request
        if audit_scope == "build_package"
        else update_report_slot_execution_request
    )
    report_request = request_factory(
        bindings,
        scientific_state=state,
        planned_report_contract=contract_file,
        campaign_registry=campaign_registry,
    )
    audit = write_scientific_pre_submit_audit(
        root / f"evidence/{token}_{contract_token}_audit.json",
        request=report_request,
        scientific_state=VerifiedFile(state.receipt_path, state.receipt_sha256),
        planned_report_contract=contract_file,
        source_epoch_receipts=(
            VerifiedFile(
                context["source_receipt"].receipt_path,
                context["source_receipt"].receipt_sha256,
            ),
        ),
        seed_receipts=(
            VerifiedFile(
                context["seed_receipt"].receipt_path,
                context["seed_receipt"].receipt_sha256,
            ),
        ),
        exact_reference_receipts=(
            VerifiedFile(
                context["reference_receipt"].receipt_path,
                context["reference_receipt"].receipt_sha256,
            ),
        ),
        campaign_registry=campaign_registry,
        repository_root=root,
        matrix_path=MATRIX_PATH,
        warnings=warnings,
    )
    _AUDIT_CAMPAIGN_REGISTRIES[
        (audit.receipt_path, audit.receipt_sha256)
    ] = campaign_registry
    return audit


def _method_evidence(
    root: Path,
    contract: dict,
    cell_id: str,
    method: str = "avqds",
    *,
    context: dict | None = None,
    token: str = "one",
    activation_cut: float = 1.0e-3,
    planner_sequence: int = 1,
    unresolved: bool = False,
):
    context = context or _registered_context(root, cell_id, token="shared")
    bindings = context["bindings"]
    arm = ARM_APM if method == "apm" else ARM_AVQDS
    action_id = prospective_run_action_id(
        bindings,
        arm=arm,
        horizon=HORIZON_FULL,
        activation_cut=activation_cut,
    )
    requested_memory_gib = 4
    resume_from_action_id = prospective_run_action_id(
        bindings,
        arm=arm,
        horizon=HORIZON_PREFIX,
        activation_cut=activation_cut,
    )
    resume_run_receipt_sha256 = "8" * 64
    resume_run_sha256 = "9" * 64
    action_inputs = {
        **bindings.to_json_dict(),
        "arm": arm,
        "horizon": HORIZON_FULL,
        "activation_cut": activation_cut,
        "resume_from_action_id": resume_from_action_id,
        "resume_run_receipt_sha256": resume_run_receipt_sha256,
        "resume_run_sha256": resume_run_sha256,
    }
    action = ProspectiveNextAction(
        kind=KIND_RUN,
        action_id=action_id,
        execution_id=(
            f"{action_id}::attempt0::mem{requested_memory_gib}g"
        ),
        arm=arm,
        horizon=HORIZON_FULL,
        activation_cut=activation_cut,
        ladder_index=0,
        attempt=0,
        planner_sequence=planner_sequence,
        requested_memory_gib=requested_memory_gib,
        reason="test full-horizon evidence",
        inputs=action_inputs,
        exact_reference_blind=True,
    )
    request = resolve_scientific_action(bindings, action)
    action_path = root / f"state/{token}_action.json"
    write_action_registry_entry(
        action_path,
        request,
        source_receipts=(
            VerifiedFile(
                context["source_receipt"].receipt_path,
                context["source_receipt"].receipt_sha256,
            ),
            VerifiedFile(
                context["seed_receipt"].receipt_path,
                context["seed_receipt"].receipt_sha256,
            ),
            VerifiedFile(
                context["reference_receipt"].receipt_path,
                context["reference_receipt"].receipt_sha256,
            ),
        ),
        repository_root=root,
        bindings=bindings,
        expected_seed=context["seed_identity"],
        expected_reference=context["reference_identity"],
        expected_source_epoch=context["source_epoch"],
    )
    action_binding = VerifiedFile(action_path.relative_to(root).as_posix(), _sha(action_path))

    run = root / f"evidence/{token}_run.json"
    metrics = root / f"evidence/{token}_metrics.json"
    planner = root / f"evidence/{token}_planner.json"
    _write_json(run, {"status": "complete", "action_id": action_id})
    submitted_state = root / f"evidence/{token}_run_submitted_state.json"
    _write_json(
        submitted_state,
        _sealed(
            {
                "schema": RUN_STATE_EVIDENCE_V1,
                "scientific_action_id": action_id,
                "execution_id": action.execution_id,
                "state": "submitted",
            },
            "run_state_evidence_sha256",
        ),
    )
    submitted_receipt = write_run_lifecycle_receipt(
        root / f"evidence/{token}_run_submitted_receipt.json",
        request=request,
        state="submitted",
        action_registry=action_binding,
        state_evidence=VerifiedFile(
            submitted_state.relative_to(root).as_posix(), _sha(submitted_state)
        ),
        repository_root=root,
    )
    completed_state = root / f"evidence/{token}_run_completed_state.json"
    _write_json(
        completed_state,
        _sealed(
            {
                "schema": RUN_STATE_EVIDENCE_V1,
                "scientific_action_id": action_id,
                "execution_id": action.execution_id,
                "state": "completed",
            },
            "run_state_evidence_sha256",
        ),
    )
    run_receipt = write_run_lifecycle_receipt(
        root / f"evidence/{token}_run_completed_receipt.json",
        request=request,
        state="completed",
        action_registry=action_binding,
        state_evidence=VerifiedFile(
            completed_state.relative_to(root).as_posix(), _sha(completed_state)
        ),
        previous_receipt=VerifiedFile(
            submitted_receipt.receipt_path,
            submitted_receipt.receipt_sha256,
        ),
        run=VerifiedFile(run.relative_to(root).as_posix(), _sha(run)),
        repository_root=root,
    )
    plan = ProspectiveScientificPlan(
        bindings_sha256=bindings.digest,
        phase="ladder_unresolved" if unresolved else "score_recorded",
        next_actions=(),
        dispositions=(),
        selected_full_action_ids={arm: action_id},
        complete=unresolved,
        unresolved=unresolved,
        blocked=False,
        cost_comparison_admissible=False,
    )
    _write_json(planner, plan.to_json_dict())
    accuracy = {
        "mean_abs_energy_error": 1.0e-4,
        "mean_abs_doublon_error": 5.0e-5,
        "mean_max_abs_site_occupation_error": 5.0e-5,
        "mean_infidelity": 1.0e-5,
    }
    run_receipt_sha256 = run_receipt.receipt_sha256
    metrics_payload = {
        "schema": SCORE_METRICS_V1,
        "score_action_id": f"score::{action_id}",
        "scientific_action_id": action_id,
        "execution_id": action.execution_id,
        "cell_id": cell_id,
        "method": arm,
        "horizon": HORIZON_FULL,
        "activation_cut": activation_cut,
        "attempt": action.attempt,
        "planner_sequence": action.planner_sequence,
        "requested_memory_gib": requested_memory_gib,
        "run_receipt_sha256": run_receipt_sha256,
        "resume_from_action_id": resume_from_action_id,
        "resume_run_receipt_sha256": resume_run_receipt_sha256,
        "resume_run_sha256": resume_run_sha256,
        "bindings_sha256": bindings.digest,
        "source_epoch_identity_sha256": bindings.source_epoch_identity_sha256,
        "exact_reference_id": bindings.reference_id,
        "exact_reference_identity_sha256": bindings.reference_identity_sha256,
        "exact_reference_receipt_sha256": context["reference_receipt"].receipt_sha256,
        "exact_manifest_sha256": bindings.reference_manifest_sha256,
        "exact_data_sha256": bindings.reference_data_sha256,
        "run_sha256": _sha(run),
        "accuracy_policy_sha256": bindings.accuracy_policy_sha256,
        "accuracy_policy": context["policy"].to_json_dict(),
        "metrics": accuracy,
        "report_values": {**accuracy, "accepted_steps": 42, "terminal_n_theta": 17},
    }
    _write_json(metrics, _sealed(metrics_payload, "score_metrics_sha256"))
    score_path = root / f"evidence/{token}_score.json"
    score_payload = {
        "schema": SCORE_QUALIFICATION_RECEIPT_V1,
        "score_action_id": f"score::{action_id}",
        "scientific_action_id": action_id,
        "execution_id": action.execution_id,
        "cell_id": cell_id,
        "method": arm,
        "horizon": HORIZON_FULL,
        "activation_cut": activation_cut,
        "attempt": action.attempt,
        "planner_sequence": action.planner_sequence,
        "requested_memory_gib": requested_memory_gib,
        "run_receipt": {
            "path": run_receipt.receipt_path,
            "sha256": run_receipt.receipt_sha256,
        },
        "action_registry": action_binding.to_json_dict(),
        "exact_reference_id": bindings.reference_id,
        "exact_reference_identity_sha256": bindings.reference_identity_sha256,
        "exact_reference_receipt": {
            "path": context["reference_receipt"].receipt_path,
            "sha256": context["reference_receipt"].receipt_sha256,
        },
        "exact_manifest_sha256": bindings.reference_manifest_sha256,
        "exact_data_sha256": bindings.reference_data_sha256,
        "bindings_sha256": bindings.digest,
        "source_epoch_identity_sha256": bindings.source_epoch_identity_sha256,
        "accuracy_policy_sha256": bindings.accuracy_policy_sha256,
        "metrics": {"path": metrics.relative_to(root).as_posix(), "sha256": _sha(metrics)},
        "planner_receipt": {
            "path": planner.relative_to(root).as_posix(),
            "sha256": _sha(planner),
        },
    }
    _write_json(score_path, _sealed(score_payload, "score_receipt_sha256"))
    score = ScoreQualificationReceipt.load(score_path, repository_root=root)
    safeguard = None
    audit_actions = [action_binding]
    if unresolved:
        safeguard_reason = SAFEGUARD_LADDER_UNRESOLVED
        safeguard_id = prospective_safeguard_id(
            bindings,
            reason=safeguard_reason,
            subject_action_id=action_id,
        )
        safeguard_sequence = action.planner_sequence + 1
        safeguard_action = ProspectiveNextAction(
            kind=KIND_RECORD_SAFEGUARD,
            action_id=safeguard_id,
            execution_id=safeguard_id,
            arm=None,
            horizon=None,
            activation_cut=None,
            ladder_index=None,
            attempt=0,
            planner_sequence=safeguard_sequence,
            requested_memory_gib=None,
            reason="record exhausted ladder",
            inputs={
                **bindings.to_json_dict(),
                "safeguard_reason": safeguard_reason,
                "subject_action_id": action_id,
                "retain_all_numerical_evidence": True,
            },
            exact_reference_blind=False,
        )
        safeguard_request = resolve_scientific_action(
            bindings, safeguard_action
        )
        safeguard_action_path = root / f"state/{token}_safeguard_action.json"
        write_action_registry_entry(
            safeguard_action_path,
            safeguard_request,
            source_receipts=(
                VerifiedFile(
                    context["source_receipt"].receipt_path,
                    context["source_receipt"].receipt_sha256,
                ),
                VerifiedFile(
                    context["seed_receipt"].receipt_path,
                    context["seed_receipt"].receipt_sha256,
                ),
                VerifiedFile(
                    context["reference_receipt"].receipt_path,
                    context["reference_receipt"].receipt_sha256,
                ),
            ),
            repository_root=root,
            bindings=bindings,
            expected_seed=context["seed_identity"],
            expected_reference=context["reference_identity"],
            expected_source_epoch=context["source_epoch"],
        )
        safeguard_action_binding = VerifiedFile(
            safeguard_action_path.relative_to(root).as_posix(),
            _sha(safeguard_action_path),
        )
        safeguard_path = root / f"evidence/{token}_safeguard.json"
        _write_json(
            safeguard_path,
            _sealed(
                {
                    "schema": SAFEGUARD_QUALIFICATION_RECEIPT_V1,
                    "safeguard_action_id": safeguard_id,
                    "reason": safeguard_reason,
                    "subject_action_id": action_id,
                    "planner_sequence": safeguard_sequence,
                    "cell_id": cell_id,
                    "method": arm,
                    "bindings_sha256": bindings.digest,
                    "action_registry": safeguard_action_binding.to_json_dict(),
                    "subject_run_receipts": [
                        score.run_receipt.to_json_dict()
                    ],
                    "subject_score_receipt": {
                        "path": score.receipt_path,
                        "sha256": score.receipt_sha256,
                    },
                    "planner_receipt": {
                        "path": planner.relative_to(root).as_posix(),
                        "sha256": _sha(planner),
                    },
                    "source_epoch_identity_sha256": bindings.source_epoch_identity_sha256,
                    "qiskit_compilation": False,
                },
                "safeguard_receipt_sha256",
            ),
        )
        safeguard = SafeguardQualificationReceipt.load(
            safeguard_path, repository_root=root
        )
        audit_actions.append(safeguard_action_binding)
    audit = _audit(
        root,
        contract,
        context,
        token=token,
        action_entries=tuple(audit_actions),
        score_receipts=(score,),
        safeguard_receipts=(() if safeguard is None else (safeguard,)),
    )
    return action_binding, score, audit, context, safeguard


def _exact_evidence(root: Path, contract: dict, cell_id: str, *, token: str = "one"):
    context = _registered_context(root, cell_id, token=token)
    audit = _audit(root, contract, context, token=token)
    return (
        context["seed_receipt"],
        context["reference_receipt"],
        context["source_receipt"],
        audit,
    )


def _rebind_score_planner(
    root: Path, score: ScoreQualificationReceipt, planner: Path
) -> ScoreQualificationReceipt:
    receipt_path = root / score.receipt_path
    payload = json.loads(receipt_path.read_text())
    payload["planner_receipt"] = {
        "path": planner.relative_to(root).as_posix(),
        "sha256": _sha(planner),
    }
    _write_json(receipt_path, _sealed(payload, "score_receipt_sha256"))
    return ScoreQualificationReceipt.load(receipt_path, repository_root=root)


def _qualified_pair_evidence(
    root: Path,
    contract: dict,
    cell_id: str,
    *,
    token: str,
):
    av_action, av_score, _av_audit, context, _ = _method_evidence(
        root,
        contract,
        cell_id,
        method="avqds",
        token=f"{token}-avqds",
        planner_sequence=1,
    )
    ap_action, ap_score, _ap_audit, _context2, _ = _method_evidence(
        root,
        contract,
        cell_id,
        method="apm",
        context=context,
        token=f"{token}-apm",
        planner_sequence=2,
    )
    planner = root / f"evidence/{token}-pair-planner.json"
    _write_json(
        planner,
        ProspectiveScientificPlan(
            bindings_sha256=context["bindings"].digest,
            phase="complete",
            next_actions=(),
            dispositions=(),
            selected_full_action_ids={
                ARM_AVQDS: av_score.scientific_action_id,
                ARM_APM: ap_score.scientific_action_id,
            },
            complete=True,
            unresolved=False,
            blocked=False,
            cost_comparison_admissible=True,
        ).to_json_dict(),
    )
    av_score = _rebind_score_planner(root, av_score, planner)
    ap_score = _rebind_score_planner(root, ap_score, planner)
    audit = _audit(
        root,
        contract,
        context,
        token=f"{token}-pair",
        action_entries=(av_action, ap_action),
        score_receipts=(av_score, ap_score),
    )
    return av_action, av_score, ap_score, audit, context


def _registry_with_wrong_source_epoch(
    root: Path,
    score: ScoreQualificationReceipt,
    context: dict,
    *,
    token: str,
) -> VerifiedFile:
    wrong_epoch = SourceEpochIdentity(
        lane_id=context["source_epoch"].lane_id,
        source_epoch_id=f"{context['source_epoch'].source_epoch_id}-wrong",
        source_revision="b" * 40,
    )
    manifest = root / f"evidence/{token}-wrong-source-manifest.json"
    _write_json(
        manifest,
        _sealed(
            {
                "schema": SOURCE_TREE_MANIFEST_V1,
                "lane_id": wrong_epoch.lane_id,
                "source_epoch_identity_sha256": wrong_epoch.identity_sha256,
                "source_revision": wrong_epoch.source_revision,
                "source_archive_sha256": wrong_epoch.source_archive_sha256,
                "source_tree_sha256": "c" * 64,
                "provenance_kind": "git_archive",
            },
            "source_tree_manifest_sha256",
        ),
    )
    source = root / f"receipts/{token}-wrong-source.json"
    _write_json(
        source,
        _sealed(
            {
                "schema": SOURCE_EPOCH_RECEIPT_V1,
                "source_epoch": wrong_epoch.to_json_dict(),
                "source_tree_manifest": {
                    "path": manifest.relative_to(root).as_posix(),
                    "sha256": _sha(manifest),
                },
                "source_tree_sha256": "c" * 64,
            },
            "source_epoch_receipt_sha256",
        ),
    )
    wrong_source = {
        "path": source.relative_to(root).as_posix(),
        "sha256": _sha(source),
    }
    registry_payload = json.loads((root / score.action_registry.path).read_text())
    rebound_receipts = []
    for binding in registry_payload["source_receipts"]:
        receipt_payload = json.loads((root / binding["path"]).read_text())
        rebound_receipts.append(
            wrong_source
            if receipt_payload.get("schema") == SOURCE_EPOCH_RECEIPT_V1
            else binding
        )
    registry_payload["source_receipts"] = rebound_receipts
    registry_payload["source_receipt_sha256"] = wrong_source["sha256"]
    registry = root / f"state/{token}-wrong-source-action.json"
    _write_json(registry, _sealed(registry_payload, "entry_sha256"))
    return VerifiedFile(registry.relative_to(root).as_posix(), _sha(registry))


def _rebind_score_action_registry(
    root: Path,
    score: ScoreQualificationReceipt,
    registry: VerifiedFile,
    *,
    token: str,
) -> ScoreQualificationReceipt:
    request_payload = json.loads((root / registry.path).read_text())["execution_request"]
    request = ExecutionRequest.from_mapping(request_payload)
    lifecycle = score.run_lifecycle(root)
    assert lifecycle.previous_receipt is not None
    previous = RunLifecycleReceipt.load(
        root / lifecycle.previous_receipt.path,
        repository_root=root,
    )
    submitted = write_run_lifecycle_receipt(
        root / f"evidence/{token}-rebound-submitted.json",
        request=request,
        state="submitted",
        action_registry=registry,
        state_evidence=previous.state_evidence,
        repository_root=root,
    )
    assert lifecycle.run is not None
    completed = write_run_lifecycle_receipt(
        root / f"evidence/{token}-rebound-completed.json",
        request=request,
        state="completed",
        action_registry=registry,
        state_evidence=lifecycle.state_evidence,
        previous_receipt=VerifiedFile(
            submitted.receipt_path,
            submitted.receipt_sha256,
        ),
        run=lifecycle.run,
        repository_root=root,
    )

    metrics_payload = json.loads((root / score.metrics.path).read_text())
    metrics_payload["run_receipt_sha256"] = completed.receipt_sha256
    metrics = root / f"evidence/{token}-rebound-metrics.json"
    _write_json(metrics, _sealed(metrics_payload, "score_metrics_sha256"))

    score_payload = json.loads((root / score.receipt_path).read_text())
    score_payload["run_receipt"] = {
        "path": completed.receipt_path,
        "sha256": completed.receipt_sha256,
    }
    score_payload["action_registry"] = registry.to_json_dict()
    score_payload["metrics"] = {
        "path": metrics.relative_to(root).as_posix(),
        "sha256": _sha(metrics),
    }
    receipt = root / f"evidence/{token}-rebound-score.json"
    _write_json(receipt, _sealed(score_payload, "score_receipt_sha256"))
    return ScoreQualificationReceipt.load(receipt, repository_root=root)


@pytest.mark.parametrize(
    ("lane_id", "cell_count", "slot_count", "horizons"),
    [
        ("paper_ii_hh_product_v1", 36, 72, {20.0}),
        ("siam_nb4_pilot_v1", 1, 2, {5.0}),
        ("paper_ii_hubbard_l4_scale_v1", 2, 4, {20.0}),
    ],
)
def test_builder_precreates_every_pending_tuple_and_method_slot(
    lane_id: str,
    cell_count: int,
    slot_count: int,
    horizons: set[float],
) -> None:
    contract = build_planned_report_contract(
        lane_id=lane_id,
        repository_root=REPOSITORY_ROOT,
        matrix_path=MATRIX_PATH,
    )
    validate_planned_report_contract(
        contract,
        repository_root=REPOSITORY_ROOT,
        matrix_path=MATRIX_PATH,
    )

    assert contract["counts"] == {
        "planned_cells": cell_count,
        "planned_exact_references": cell_count,
        "planned_method_slots": slot_count,
        "pending_exact_references": cell_count,
        "pending_method_slots": slot_count,
        "candidate_method_slots": 0,
        "qualified_method_slots": 0,
        "unresolved_method_slots": 0,
    }
    assert len({cell["cell_id"] for cell in contract["cells"]}) == cell_count
    assert {cell["horizon"] for cell in contract["cells"]} == horizons
    assert all(cell["controller"] == "state_motion_5e-5" for cell in contract["cells"])
    assert all(cell["grid"]["points"] == 501 for cell in contract["cells"])
    assert all(cell["grid"]["stop"] == cell["horizon"] for cell in contract["cells"])
    assert all(cell["exact_reference"]["state"] == "pending" for cell in contract["cells"])
    assert all(
        [slot["method"] for slot in cell["method_slots"]] == ["apm", "avqds"]
        for cell in contract["cells"]
    )
    assert all(
        slot["state"] == "pending"
        for cell in contract["cells"]
        for slot in cell["method_slots"]
    )
    assert contract["qiskit_compilation"] == {
        "requested": False,
        "used": False,
        "canonical": False,
    }
    matrix_file = REPOSITORY_ROOT / MATRIX_PATH
    assert contract["matrix"]["path"] == MATRIX_PATH.as_posix()
    assert contract["matrix"]["raw_sha256"] == hashlib.sha256(
        matrix_file.read_bytes()
    ).hexdigest()
    assert set(contract["matrix"]) == {
        "path",
        "raw_sha256",
        "payload_sha256",
        "expanded_semantic_sha256",
        "matrix_id",
    }


def test_ingestion_update_changes_one_slot_without_changing_tuple_structure(
    tmp_path: Path,
) -> None:
    _copy_matrix(tmp_path)
    contract = build_planned_report_contract(
        lane_id="paper_ii_hubbard_l4_scale_v1",
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    cell_id = contract["cells"][0]["cell_id"]
    action, score, audit, _context, _safeguard = _method_evidence(
        tmp_path, contract, cell_id
    )
    updated = update_method_slot(
        contract,
        cell_id=cell_id,
        method="avqds",
        state="candidate",
        action_registry=action,
        score_receipt=score,
        scientific_pre_submit_audit=audit,
        campaign_registry=_campaign_registry_for_audit(audit),
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )

    assert updated["tuple_structure_sha256"] == contract["tuple_structure_sha256"]
    assert updated["counts"]["pending_method_slots"] == 3
    assert updated["counts"]["candidate_method_slots"] == 1
    slots = {
        slot["slot_id"]: slot
        for cell in updated["cells"]
        for slot in cell["method_slots"]
    }
    changed = slots[f"{cell_id}::avqds"]
    assert changed["values"]["terminal_n_theta"] == 17
    assert changed["provenance"]["score_receipt"]["sha256"] == (
        score.receipt_sha256
    )
    validate_planned_report_contract(
        updated,
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )


def test_method_update_rejects_audit_without_promoted_evidence(
    tmp_path: Path,
) -> None:
    _copy_matrix(tmp_path)
    contract = build_planned_report_contract(
        lane_id="paper_ii_hubbard_l4_scale_v1",
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    cell_id = contract["cells"][0]["cell_id"]
    action, score, _score_audit, context, _safeguard = _method_evidence(
        tmp_path,
        contract,
        cell_id,
        token="promoted-evidence",
    )
    bootstrap_only_audit = _audit(
        tmp_path,
        contract,
        context,
        token="bootstrap-only",
    )

    with pytest.raises(
        PlannedReportContractError,
        match="audit.*primary|primary.*audit",
    ):
        update_method_slot(
            contract,
            cell_id=cell_id,
            method="avqds",
            state="candidate",
            action_registry=action,
            score_receipt=score,
            scientific_pre_submit_audit=bootstrap_only_audit,
            campaign_registry=_campaign_registry_for_audit(bootstrap_only_audit),
            repository_root=tmp_path,
            matrix_path=MATRIX_PATH,
        )


def test_method_update_rejects_package_scope_audit(tmp_path: Path) -> None:
    _copy_matrix(tmp_path)
    contract = build_planned_report_contract(
        lane_id="paper_ii_hubbard_l4_scale_v1",
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    cell_id = contract["cells"][0]["cell_id"]
    action, score, _report_audit, context, _safeguard = _method_evidence(
        tmp_path,
        contract,
        cell_id,
        token="wrong-audit-scope",
    )
    package_audit = _audit(
        tmp_path,
        contract,
        context,
        token="wrong-audit-scope-package",
        action_entries=(action,),
        score_receipts=(score,),
        audit_scope="build_package",
    )

    with pytest.raises(
        PlannedReportContractError,
        match="not report-slot scoped",
    ):
        update_method_slot(
            contract,
            cell_id=cell_id,
            method="avqds",
            state="candidate",
            action_registry=action,
            score_receipt=score,
            scientific_pre_submit_audit=package_audit,
            campaign_registry=_campaign_registry_for_audit(package_audit),
            repository_root=tmp_path,
            matrix_path=MATRIX_PATH,
        )


def test_method_update_rejects_current_registry_authority_drift(
    tmp_path: Path,
) -> None:
    _copy_matrix(tmp_path)
    contract = build_planned_report_contract(
        lane_id="paper_ii_hubbard_l4_scale_v1",
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    cell_id = contract["cells"][0]["cell_id"]
    action, score, audit, _context, _safeguard = _method_evidence(
        tmp_path,
        contract,
        cell_id,
        token="registry-authority-drift",
    )
    drifted = replace(
        _campaign_registry_for_audit(audit),
        registry_id="drifted_planned_report_registry",
    )
    drift_audit = audit

    with pytest.raises(
        PlannedReportContractError,
        match="current campaign registry|lane authority drifted",
    ):
        update_method_slot(
            contract,
            cell_id=cell_id,
            method="avqds",
            state="candidate",
            action_registry=action,
            score_receipt=score,
            scientific_pre_submit_audit=drift_audit,
            campaign_registry=drifted,
            repository_root=tmp_path,
            matrix_path=MATRIX_PATH,
        )


def test_method_update_carries_audit_warnings_into_slot_provenance(
    tmp_path: Path,
) -> None:
    _copy_matrix(tmp_path)
    contract = build_planned_report_contract(
        lane_id="paper_ii_hubbard_l4_scale_v1",
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    cell_id = contract["cells"][0]["cell_id"]
    action, score, _plain_audit, context, _safeguard = _method_evidence(
        tmp_path,
        contract,
        cell_id,
        token="warning-provenance",
    )
    warning_audit = _audit(
        tmp_path,
        contract,
        context,
        token="warning-provenance-audit",
        action_entries=(action,),
        score_receipts=(score,),
        warnings=("post-fetch caveat retained", "scheduler caveat retained"),
    )

    updated = update_method_slot(
        contract,
        cell_id=cell_id,
        method="avqds",
        state="candidate",
        action_registry=action,
        score_receipt=score,
        scientific_pre_submit_audit=warning_audit,
        campaign_registry=_campaign_registry_for_audit(warning_audit),
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    slot = next(
        row
        for cell in updated["cells"]
        if cell["cell_id"] == cell_id
        for row in cell["method_slots"]
        if row["method"] == "avqds"
    )
    assert slot["provenance"]["audit_warnings"] == [
        "post-fetch caveat retained",
        "scheduler caveat retained",
    ]

    changed = deepcopy(updated)
    changed_slot = next(
        row
        for cell in changed["cells"]
        if cell["cell_id"] == cell_id
        for row in cell["method_slots"]
        if row["method"] == "avqds"
    )
    changed_slot["provenance"]["audit_warnings"].reverse()
    changed = _sealed(changed, "report_contract_sha256")
    with pytest.raises(
        PlannedReportContractError,
        match="audit contract binding|audit warnings",
    ):
        update_method_slot(
            changed,
            cell_id=cell_id,
            method="avqds",
            state="candidate",
            action_registry=action,
            score_receipt=score,
            scientific_pre_submit_audit=warning_audit,
            campaign_registry=_campaign_registry_for_audit(warning_audit),
            repository_root=tmp_path,
            matrix_path=MATRIX_PATH,
        )


def test_method_update_requires_supplied_registry_to_equal_score_registry(
    tmp_path: Path,
) -> None:
    _copy_matrix(tmp_path)
    contract = build_planned_report_contract(
        lane_id="paper_ii_hubbard_l4_scale_v1",
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    cell_id = contract["cells"][0]["cell_id"]
    action, score, audit, _context, _safeguard = _method_evidence(
        tmp_path,
        contract,
        cell_id,
        token="duplicate-registry",
    )
    duplicate_path = tmp_path / "state/duplicate-registry-copy.json"
    duplicate_path.parent.mkdir(parents=True, exist_ok=True)
    duplicate_path.write_bytes((tmp_path / action.path).read_bytes())
    duplicate = VerifiedFile(
        duplicate_path.relative_to(tmp_path).as_posix(),
        _sha(duplicate_path),
    )

    with pytest.raises(PlannedReportContractError, match="score action registry"):
        update_method_slot(
            contract,
            cell_id=cell_id,
            method="avqds",
            state="candidate",
            action_registry=duplicate,
            score_receipt=score,
            scientific_pre_submit_audit=audit,
            campaign_registry=_campaign_registry_for_audit(audit),
            repository_root=tmp_path,
            matrix_path=MATRIX_PATH,
        )


def test_exact_reference_update_preserves_predeclared_tuple(tmp_path: Path) -> None:
    _copy_matrix(tmp_path)
    contract = build_planned_report_contract(
        lane_id="paper_ii_hubbard_l4_scale_v1",
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    cell_id = contract["cells"][0]["cell_id"]
    seed, reference, source, audit = _exact_evidence(tmp_path, contract, cell_id)
    updated = update_exact_reference(
        contract,
        cell_id=cell_id,
        reference_receipt=reference,
        seed_receipt=seed,
        source_epoch_receipt=source,
        scientific_pre_submit_audit=audit,
        campaign_registry=_campaign_registry_for_audit(audit),
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    assert updated["tuple_structure_sha256"] == contract["tuple_structure_sha256"]
    assert updated["counts"]["pending_exact_references"] == (
        contract["counts"]["pending_exact_references"] - 1
    )
    updated_cell = next(row for row in updated["cells"] if row["cell_id"] == cell_id)
    assert updated_cell["exact_reference"]["artifact"]["data"]["sha256"] == (
        reference.data.sha256
    )


def test_exact_reference_update_requires_exact_audit_receipt_path(
    tmp_path: Path,
) -> None:
    _copy_matrix(tmp_path)
    contract = build_planned_report_contract(
        lane_id="paper_ii_hubbard_l4_scale_v1",
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    cell_id = contract["cells"][0]["cell_id"]
    seed, reference, source, audit = _exact_evidence(
        tmp_path,
        contract,
        cell_id,
        token="exact-audit-path",
    )
    duplicate_path = tmp_path / "receipts/exact-audit-path-duplicate-reference.json"
    duplicate_path.write_bytes((tmp_path / reference.receipt_path).read_bytes())
    duplicate_reference = ExactReferenceReceipt.load(
        duplicate_path,
        repository_root=tmp_path,
        expected_reference=next(
            row
            for row in build_identity_registry(
                ParallelBenchmarkMatrix.read(tmp_path / MATRIX_PATH)
            ).references
            if row.cell_id == cell_id
        ),
        expected_seed_identity=next(
            row
            for row in build_identity_registry(
                ParallelBenchmarkMatrix.read(tmp_path / MATRIX_PATH)
            ).seeds
            if row.seed_id == reference.seed_id
        ),
        expected_seed_receipt=seed,
        expected_source_epoch=source.source_epoch,
    )

    with pytest.raises(
        PlannedReportContractError,
        match="exact-reference report evidence",
    ):
        update_exact_reference(
            contract,
            cell_id=cell_id,
            reference_receipt=duplicate_reference,
            seed_receipt=seed,
            source_epoch_receipt=source,
            scientific_pre_submit_audit=audit,
            campaign_registry=_campaign_registry_for_audit(audit),
            repository_root=tmp_path,
            matrix_path=MATRIX_PATH,
        )


def test_exact_reference_update_rejects_package_scope_audit(
    tmp_path: Path,
) -> None:
    _copy_matrix(tmp_path)
    contract = build_planned_report_contract(
        lane_id="paper_ii_hubbard_l4_scale_v1",
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    cell_id = contract["cells"][0]["cell_id"]
    context = _registered_context(tmp_path, cell_id, token="exact-wrong-scope")
    package_audit = _audit(
        tmp_path,
        contract,
        context,
        token="exact-wrong-scope-package",
        audit_scope="build_package",
    )

    with pytest.raises(
        PlannedReportContractError,
        match="not report-slot scoped",
    ):
        update_exact_reference(
            contract,
            cell_id=cell_id,
            reference_receipt=context["reference_receipt"],
            seed_receipt=context["seed_receipt"],
            source_epoch_receipt=context["source_receipt"],
            scientific_pre_submit_audit=package_audit,
            campaign_registry=_campaign_registry_for_audit(package_audit),
            repository_root=tmp_path,
            matrix_path=MATRIX_PATH,
        )


def test_exact_reference_update_rejects_off_registry_authority(
    tmp_path: Path,
) -> None:
    _copy_matrix(tmp_path)
    contract = build_planned_report_contract(
        lane_id="paper_ii_hubbard_l4_scale_v1",
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    cell_id = contract["cells"][0]["cell_id"]
    seed, reference, source, audit = _exact_evidence(
        tmp_path,
        contract,
        cell_id,
        token="exact-off-registry",
    )
    registry = _campaign_registry_for_audit(audit)
    lane = registry.lanes[0]
    off_registry = replace(
        registry,
        lanes=(
            replace(
                lane,
                scientific_state=tmp_path / "state/off-registry-state.json",
            ),
        ),
    )
    off_registry_audit = audit

    with pytest.raises(
        PlannedReportContractError,
        match="current campaign registry|lane authority drifted",
    ):
        update_exact_reference(
            contract,
            cell_id=cell_id,
            reference_receipt=reference,
            seed_receipt=seed,
            source_epoch_receipt=source,
            scientific_pre_submit_audit=off_registry_audit,
            campaign_registry=off_registry,
            repository_root=tmp_path,
            matrix_path=MATRIX_PATH,
        )


def test_method_update_rejects_arbitrary_or_mismatched_evidence(
    tmp_path: Path,
) -> None:
    _copy_matrix(tmp_path)
    contract = build_planned_report_contract(
        lane_id="paper_ii_hubbard_l4_scale_v1",
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    cell_id = contract["cells"][0]["cell_id"]
    action, score, audit, _context, _safeguard = _method_evidence(
        tmp_path, contract, cell_id
    )

    with pytest.raises(PlannedReportContractError, match="requested report state"):
        update_method_slot(
            contract,
            cell_id=cell_id,
            method="avqds",
            state="qualified",
            action_registry=action,
            score_receipt=score,
            scientific_pre_submit_audit=audit,
            campaign_registry=_campaign_registry_for_audit(audit),
            repository_root=tmp_path,
            matrix_path=MATRIX_PATH,
        )

    with pytest.raises(PlannedReportContractError, match="pre-submit audit"):
        update_method_slot(
            contract,
            cell_id=cell_id,
            method="avqds",
            state="candidate",
            action_registry=action,
            score_receipt=score,
            scientific_pre_submit_audit=replace(
                audit, receipt_sha256="e" * 64
            ),
            campaign_registry=_campaign_registry_for_audit(audit),
            repository_root=tmp_path,
            matrix_path=MATRIX_PATH,
        )


def test_exact_slot_idempotent_replay_and_collision_rejection(tmp_path: Path) -> None:
    _copy_matrix(tmp_path)
    contract = build_planned_report_contract(
        lane_id="paper_ii_hubbard_l4_scale_v1",
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    cell_id = contract["cells"][0]["cell_id"]
    seed, reference, source, audit = _exact_evidence(
        tmp_path, contract, cell_id, token="first"
    )
    updated = update_exact_reference(
        contract,
        cell_id=cell_id,
        reference_receipt=reference,
        seed_receipt=seed,
        source_epoch_receipt=source,
        scientific_pre_submit_audit=audit,
        campaign_registry=_campaign_registry_for_audit(audit),
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    replayed = update_exact_reference(
        updated,
        cell_id=cell_id,
        reference_receipt=reference,
        seed_receipt=seed,
        source_epoch_receipt=source,
        scientific_pre_submit_audit=audit,
        campaign_registry=_campaign_registry_for_audit(audit),
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    assert replayed == updated

    seed2, reference2, source2, audit2 = _exact_evidence(
        tmp_path, updated, cell_id, token="second"
    )
    with pytest.raises(PlannedReportContractError, match="immutable"):
        update_exact_reference(
            updated,
            cell_id=cell_id,
            reference_receipt=reference2,
            seed_receipt=seed2,
            source_epoch_receipt=source2,
        scientific_pre_submit_audit=audit2,
        campaign_registry=_campaign_registry_for_audit(audit2),
        repository_root=tmp_path,
            matrix_path=MATRIX_PATH,
        )


def test_terminal_slot_idempotent_replay_and_collision_rejection(tmp_path: Path) -> None:
    _copy_matrix(tmp_path)
    contract = build_planned_report_contract(
        lane_id="paper_ii_hubbard_l4_scale_v1",
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    cell_id = contract["cells"][0]["cell_id"]
    action, score, audit, context, safeguard = _method_evidence(
        tmp_path, contract, cell_id, token="terminal-first", unresolved=True
    )
    assert safeguard is not None
    updated = update_method_slot(
        contract,
        cell_id=cell_id,
        method="avqds",
        state="unresolved",
        action_registry=action,
        score_receipt=score,
        safeguard_receipt=safeguard,
        scientific_pre_submit_audit=audit,
        campaign_registry=_campaign_registry_for_audit(audit),
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    replayed = update_method_slot(
        updated,
        cell_id=cell_id,
        method="avqds",
        state="unresolved",
        action_registry=action,
        score_receipt=score,
        safeguard_receipt=safeguard,
        scientific_pre_submit_audit=audit,
        campaign_registry=_campaign_registry_for_audit(audit),
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    assert replayed == updated

    action2, score2, audit2, _context2, safeguard2 = _method_evidence(
        tmp_path,
        updated,
        cell_id,
        context=context,
        token="terminal-second",
        activation_cut=3.0e-4,
        planner_sequence=2,
        unresolved=True,
    )
    assert safeguard2 is not None
    with pytest.raises(PlannedReportContractError, match="immutable"):
        update_method_slot(
            updated,
            cell_id=cell_id,
            method="avqds",
            state="unresolved",
            action_registry=action2,
            score_receipt=score2,
            safeguard_receipt=safeguard2,
            scientific_pre_submit_audit=audit2,
            campaign_registry=_campaign_registry_for_audit(audit2),
            repository_root=tmp_path,
            matrix_path=MATRIX_PATH,
        )


def test_unresolved_promotion_rejects_audit_without_safeguard_evidence(
    tmp_path: Path,
) -> None:
    _copy_matrix(tmp_path)
    contract = build_planned_report_contract(
        lane_id="paper_ii_hubbard_l4_scale_v1",
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    cell_id = contract["cells"][0]["cell_id"]
    action, score, _complete_audit, context, safeguard = _method_evidence(
        tmp_path,
        contract,
        cell_id,
        token="missing-safeguard-audit",
        unresolved=True,
    )
    assert safeguard is not None
    primary_only_audit = _audit(
        tmp_path,
        contract,
        context,
        token="missing-safeguard-audit-primary-only",
        action_entries=(action,),
        score_receipts=(score,),
    )

    with pytest.raises(
        PlannedReportContractError,
        match="safeguard report evidence",
    ):
        update_method_slot(
            contract,
            cell_id=cell_id,
            method="avqds",
            state="unresolved",
            action_registry=action,
            score_receipt=score,
            safeguard_receipt=safeguard,
            scientific_pre_submit_audit=primary_only_audit,
            campaign_registry=_campaign_registry_for_audit(primary_only_audit),
            repository_root=tmp_path,
            matrix_path=MATRIX_PATH,
        )


def test_candidate_supersession_requires_newer_selected_planner_proof(
    tmp_path: Path,
) -> None:
    _copy_matrix(tmp_path)
    contract = build_planned_report_contract(
        lane_id="paper_ii_hubbard_l4_scale_v1",
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    cell_id = contract["cells"][0]["cell_id"]
    action1, score1, audit1, context, _ = _method_evidence(
        tmp_path, contract, cell_id, token="candidate-first", planner_sequence=1
    )
    updated = update_method_slot(
        contract,
        cell_id=cell_id,
        method="avqds",
        state="candidate",
        action_registry=action1,
        score_receipt=score1,
        scientific_pre_submit_audit=audit1,
        campaign_registry=_campaign_registry_for_audit(audit1),
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )

    stale_action, stale_score, stale_audit, _context2, _ = _method_evidence(
        tmp_path,
        updated,
        cell_id,
        context=context,
        token="candidate-stale",
        activation_cut=3.0e-4,
        planner_sequence=1,
    )
    with pytest.raises(PlannedReportContractError, match="newer selected planner"):
        update_method_slot(
            updated,
            cell_id=cell_id,
            method="avqds",
            state="candidate",
            action_registry=stale_action,
            score_receipt=stale_score,
            scientific_pre_submit_audit=stale_audit,
            campaign_registry=_campaign_registry_for_audit(stale_audit),
            repository_root=tmp_path,
            matrix_path=MATRIX_PATH,
        )

    newer_action, newer_score, newer_audit, _context3, _ = _method_evidence(
        tmp_path,
        updated,
        cell_id,
        context=context,
        token="candidate-newer",
        activation_cut=1.0e-4,
        planner_sequence=2,
    )
    superseded = update_method_slot(
        updated,
        cell_id=cell_id,
        method="avqds",
        state="candidate",
        action_registry=newer_action,
        score_receipt=newer_score,
        scientific_pre_submit_audit=newer_audit,
        campaign_registry=_campaign_registry_for_audit(newer_audit),
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    slot = next(
        slot
        for cell in superseded["cells"]
        if cell["cell_id"] == cell_id
        for slot in cell["method_slots"]
        if slot["method"] == "avqds"
    )
    assert slot["provenance"]["score_receipt"]["sha256"] == newer_score.receipt_sha256
    assert slot["provenance"]["superseded_candidates"][0]["score_receipt"][
        "sha256"
    ] == score1.receipt_sha256


def test_candidate_replay_rejects_resealed_superseded_history(
    tmp_path: Path,
) -> None:
    _copy_matrix(tmp_path)
    contract = build_planned_report_contract(
        lane_id="paper_ii_hubbard_l4_scale_v1",
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    cell_id = contract["cells"][0]["cell_id"]
    action1, score1, audit1, context, _ = _method_evidence(
        tmp_path,
        contract,
        cell_id,
        token="history-first",
        planner_sequence=1,
    )
    first = update_method_slot(
        contract,
        cell_id=cell_id,
        method="avqds",
        state="candidate",
        action_registry=action1,
        score_receipt=score1,
        scientific_pre_submit_audit=audit1,
        campaign_registry=_campaign_registry_for_audit(audit1),
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    action2, score2, audit2, _context2, _ = _method_evidence(
        tmp_path,
        first,
        cell_id,
        context=context,
        token="history-second",
        activation_cut=1.0e-4,
        planner_sequence=2,
    )
    current = update_method_slot(
        first,
        cell_id=cell_id,
        method="avqds",
        state="candidate",
        action_registry=action2,
        score_receipt=score2,
            scientific_pre_submit_audit=audit2,
            campaign_registry=_campaign_registry_for_audit(audit2),
            repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )

    def candidate_slot(payload: dict) -> dict:
        return next(
            row
            for cell in payload["cells"]
            if cell["cell_id"] == cell_id
            for row in cell["method_slots"]
            if row["method"] == "avqds"
        )

    for mutate in (
        lambda history, provenance: history[0].__setitem__(
            "audit_warnings", ["forged historical warning"]
        ),
        lambda history, provenance: history[0].__setitem__(
            "superseded_candidates", []
        ),
        lambda history, provenance: history[0].__setitem__(
            "planner_sequence", provenance["planner_sequence"]
        ),
    ):
        forged = deepcopy(current)
        provenance = candidate_slot(forged)["provenance"]
        history = provenance["superseded_candidates"]
        mutate(history, provenance)
        forged = _sealed(forged, "report_contract_sha256")
        with pytest.raises(
            PlannedReportContractError,
            match="superseded|history|audit|planner sequence",
        ):
            update_method_slot(
                forged,
                cell_id=cell_id,
                method="avqds",
                state="candidate",
                action_registry=action2,
                score_receipt=score2,
                scientific_pre_submit_audit=audit2,
                campaign_registry=_campaign_registry_for_audit(audit2),
                repository_root=tmp_path,
                matrix_path=MATRIX_PATH,
            )


def test_candidate_replay_rejects_resealed_numerical_value_change(
    tmp_path: Path,
) -> None:
    _copy_matrix(tmp_path)
    contract = build_planned_report_contract(
        lane_id="paper_ii_hubbard_l4_scale_v1",
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    cell_id = contract["cells"][0]["cell_id"]
    action, score, audit, _context, _safeguard = _method_evidence(
        tmp_path,
        contract,
        cell_id,
        token="replay-value-forgery",
    )
    updated = update_method_slot(
        contract,
        cell_id=cell_id,
        method="avqds",
        state="candidate",
        action_registry=action,
        score_receipt=score,
        scientific_pre_submit_audit=audit,
        campaign_registry=_campaign_registry_for_audit(audit),
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    changed = deepcopy(updated)
    slot = next(
        row
        for cell in changed["cells"]
        if cell["cell_id"] == cell_id
        for row in cell["method_slots"]
        if row["method"] == "avqds"
    )
    slot["values"]["terminal_n_theta"] = 999
    changed = _sealed(changed, "report_contract_sha256")

    with pytest.raises(
        PlannedReportContractError,
        match="audit contract binding|newer selected planner|replay",
    ):
        update_method_slot(
            changed,
            cell_id=cell_id,
            method="avqds",
            state="candidate",
            action_registry=action,
            score_receipt=score,
            scientific_pre_submit_audit=audit,
            campaign_registry=_campaign_registry_for_audit(audit),
            repository_root=tmp_path,
            matrix_path=MATRIX_PATH,
        )


def test_qualified_slot_is_immutable_except_same_evidence_replay(
    tmp_path: Path,
) -> None:
    _copy_matrix(tmp_path)
    contract = build_planned_report_contract(
        lane_id="paper_ii_hubbard_l4_scale_v1",
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    cell_id = contract["cells"][0]["cell_id"]
    av_action, av_score, _av_audit, context, _ = _method_evidence(
        tmp_path,
        contract,
        cell_id,
        method="avqds",
        token="qualified-avqds",
        planner_sequence=1,
    )
    ap_action, ap_score, _ap_audit, _context2, _ = _method_evidence(
        tmp_path,
        contract,
        cell_id,
        method="apm",
        context=context,
        token="qualified-apm",
        planner_sequence=2,
    )
    planner = tmp_path / "evidence/qualified-pair-planner.json"
    _write_json(
        planner,
        ProspectiveScientificPlan(
            bindings_sha256=context["bindings"].digest,
            phase="complete",
            next_actions=(),
            dispositions=(),
            selected_full_action_ids={
                ARM_AVQDS: av_score.scientific_action_id,
                ARM_APM: ap_score.scientific_action_id,
            },
            complete=True,
            unresolved=False,
            blocked=False,
            cost_comparison_admissible=True,
        ).to_json_dict(),
    )
    av_score = _rebind_score_planner(tmp_path, av_score, planner)
    ap_score = _rebind_score_planner(tmp_path, ap_score, planner)
    audit = _audit(
        tmp_path,
        contract,
        context,
        token="qualified-pair-rebound",
        action_entries=(av_action, ap_action),
        score_receipts=(av_score, ap_score),
    )
    qualified = update_method_slot(
        contract,
        cell_id=cell_id,
        method="avqds",
        state="qualified",
        action_registry=av_action,
        score_receipt=av_score,
        paired_score_receipt=ap_score,
        scientific_pre_submit_audit=audit,
        campaign_registry=_campaign_registry_for_audit(audit),
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    assert update_method_slot(
        qualified,
        cell_id=cell_id,
        method="avqds",
        state="qualified",
        action_registry=av_action,
        score_receipt=av_score,
        paired_score_receipt=ap_score,
        scientific_pre_submit_audit=audit,
        campaign_registry=_campaign_registry_for_audit(audit),
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    ) == qualified

    replacement_action, replacement_score, replacement_audit, _context3, _ = (
        _method_evidence(
            tmp_path,
            qualified,
            cell_id,
            method="avqds",
            context=context,
            token="qualified-overwrite",
            activation_cut=3.0e-4,
            planner_sequence=3,
        )
    )
    with pytest.raises(PlannedReportContractError, match="terminal qualified"):
        update_method_slot(
            qualified,
            cell_id=cell_id,
            method="avqds",
            state="candidate",
            action_registry=replacement_action,
            score_receipt=replacement_score,
            scientific_pre_submit_audit=replacement_audit,
            campaign_registry=_campaign_registry_for_audit(replacement_audit),
            repository_root=tmp_path,
            matrix_path=MATRIX_PATH,
        )


def test_qualified_promotion_rejects_peer_registry_from_wrong_source_epoch(
    tmp_path: Path,
) -> None:
    _copy_matrix(tmp_path)
    contract = build_planned_report_contract(
        lane_id="paper_ii_hubbard_l4_scale_v1",
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    cell_id = contract["cells"][0]["cell_id"]
    av_action, av_score, ap_score, audit, context = _qualified_pair_evidence(
        tmp_path,
        contract,
        cell_id,
        token="wrong-peer-registry",
    )
    wrong_registry = _registry_with_wrong_source_epoch(
        tmp_path,
        ap_score,
        context,
        token="wrong-peer-registry",
    )
    forged_peer = _rebind_score_action_registry(
        tmp_path,
        ap_score,
        wrong_registry,
        token="wrong-peer-registry",
    )

    with pytest.raises(
        PlannedReportContractError,
        match="qualified peer action registry",
    ):
        update_method_slot(
            contract,
            cell_id=cell_id,
            method="avqds",
            state="qualified",
            action_registry=av_action,
            score_receipt=av_score,
            paired_score_receipt=forged_peer,
            scientific_pre_submit_audit=audit,
            campaign_registry=_campaign_registry_for_audit(audit),
            repository_root=tmp_path,
            matrix_path=MATRIX_PATH,
        )


def test_qualified_promotion_rejects_audit_without_peer_evidence(
    tmp_path: Path,
) -> None:
    _copy_matrix(tmp_path)
    contract = build_planned_report_contract(
        lane_id="paper_ii_hubbard_l4_scale_v1",
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    cell_id = contract["cells"][0]["cell_id"]
    av_action, av_score, ap_score, _pair_audit, context = (
        _qualified_pair_evidence(
            tmp_path,
            contract,
            cell_id,
            token="missing-peer-audit",
        )
    )
    primary_only_audit = _audit(
        tmp_path,
        contract,
        context,
        token="missing-peer-audit-primary-only",
        action_entries=(av_action,),
        score_receipts=(av_score,),
    )

    with pytest.raises(
        PlannedReportContractError,
        match="peer report evidence",
    ):
        update_method_slot(
            contract,
            cell_id=cell_id,
            method="avqds",
            state="qualified",
            action_registry=av_action,
            score_receipt=av_score,
            paired_score_receipt=ap_score,
            scientific_pre_submit_audit=primary_only_audit,
            campaign_registry=_campaign_registry_for_audit(primary_only_audit),
            repository_root=tmp_path,
            matrix_path=MATRIX_PATH,
        )


def test_qualified_replay_requires_both_registries_run_receipts_and_policy_digests(
    tmp_path: Path,
) -> None:
    _copy_matrix(tmp_path)
    contract = build_planned_report_contract(
        lane_id="paper_ii_hubbard_l4_scale_v1",
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    cell_id = contract["cells"][0]["cell_id"]
    av_action, av_score, ap_score, audit, _context = _qualified_pair_evidence(
        tmp_path,
        contract,
        cell_id,
        token="complete-replay-provenance",
    )
    qualified = update_method_slot(
        contract,
        cell_id=cell_id,
        method="avqds",
        state="qualified",
        action_registry=av_action,
        score_receipt=av_score,
        paired_score_receipt=ap_score,
        scientific_pre_submit_audit=audit,
        campaign_registry=_campaign_registry_for_audit(audit),
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    slot = next(
        slot
        for cell in qualified["cells"]
        if cell["cell_id"] == cell_id
        for slot in cell["method_slots"]
        if slot["method"] == "avqds"
    )
    required = {
        "action_registry",
        "run_receipt",
        "paired_action_registry",
        "paired_run_receipt",
        "accuracy_policy_sha256",
        "matrix_accuracy_policy_sha256",
    }
    assert required.issubset(slot["provenance"])

    for field in (
        "run_receipt",
        "paired_action_registry",
        "paired_run_receipt",
        "accuracy_policy_sha256",
        "matrix_accuracy_policy_sha256",
    ):
        changed = deepcopy(qualified)
        changed_slot = next(
            row
            for cell in changed["cells"]
            if cell["cell_id"] == cell_id
            for row in cell["method_slots"]
            if row["method"] == "avqds"
        )
        del changed_slot["provenance"][field]
        changed = _sealed(changed, "report_contract_sha256")
        with pytest.raises(
            PlannedReportContractError,
            match="incomplete evidence provenance",
        ):
            update_method_slot(
                changed,
                cell_id=cell_id,
                method="avqds",
                state="qualified",
                action_registry=av_action,
                score_receipt=av_score,
                paired_score_receipt=ap_score,
                scientific_pre_submit_audit=audit,
                campaign_registry=_campaign_registry_for_audit(audit),
                repository_root=tmp_path,
                matrix_path=MATRIX_PATH,
            )


def test_validator_rejects_tuple_mutation_and_qiskit_enablement() -> None:
    contract = build_planned_report_contract(
        lane_id="siam_nb4_pilot_v1",
        repository_root=REPOSITORY_ROOT,
        matrix_path=MATRIX_PATH,
    )
    changed = deepcopy(contract)
    changed["cells"][0]["cell_id"] = "different-cell"
    with pytest.raises(PlannedReportContractError, match="tuple structure"):
        validate_planned_report_contract(
            changed,
            repository_root=REPOSITORY_ROOT,
            matrix_path=MATRIX_PATH,
        )

    changed = deepcopy(contract)
    changed["qiskit_compilation"]["requested"] = True
    with pytest.raises(PlannedReportContractError, match="Qiskit"):
        validate_planned_report_contract(
            changed,
            repository_root=REPOSITORY_ROOT,
            matrix_path=MATRIX_PATH,
        )


def test_writer_materializes_and_checks_stable_paths(tmp_path: Path) -> None:
    matrix_source = REPOSITORY_ROOT / MATRIX_PATH
    copied_matrix = tmp_path / MATRIX_PATH
    copied_matrix.parent.mkdir(parents=True)
    copied_matrix.write_bytes(matrix_source.read_bytes())

    summary = write_default_planned_report_contracts(
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
    )
    assert set(summary) == set(DEFAULT_REPORT_PATHS)
    assert sum(row["planned_cells"] for row in summary.values()) == 39
    assert sum(row["planned_method_slots"] for row in summary.values()) == 78

    for lane_id, relative_path in DEFAULT_REPORT_PATHS.items():
        payload = json.loads((tmp_path / relative_path).read_text(encoding="utf-8"))
        assert payload["lane_id"] == lane_id
    assert write_default_planned_report_contracts(
        repository_root=tmp_path,
        matrix_path=MATRIX_PATH,
        check_only=True,
    ) == summary

    stale_path = tmp_path / DEFAULT_REPORT_PATHS["paper_ii_hh_product_v1"]
    stale_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(PlannedReportContractError, match="absent or stale"):
        write_default_planned_report_contracts(
            repository_root=tmp_path,
            matrix_path=MATRIX_PATH,
            check_only=True,
        )
