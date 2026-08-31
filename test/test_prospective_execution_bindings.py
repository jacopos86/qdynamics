"""Typed executor and bootstrap/scientific bridge contracts."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import pytest

from pipelines.time_dynamics.acquisition_supervisor import (
    CampaignRegistry,
    LaneRegistration,
)
from pipelines.time_dynamics.parallel_benchmark_matrix import (
    DEFAULT_MATRIX_PATH,
    ParallelBenchmarkMatrix,
)
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    LANE_HH,
    BootstrapEvidence,
    SourceEpochIdentity,
    build_identity_registry,
    plan_bootstrap_actions,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    EXACT_REFERENCE_DATA_V1,
    EXACT_REFERENCE_MANIFEST_V1,
    EXACT_REFERENCE_RECEIPT_V1,
    EXEC_PROSPECTIVE_SCORE,
    EXEC_PROSPECTIVE_AVQDS,
    RUN_FAILURE_EVIDENCE_V1,
    RUN_LIFECYCLE_RECEIPT_V1,
    RUN_STATE_EVIDENCE_V1,
    SCORE_METRICS_V1,
    SCORE_QUALIFICATION_RECEIPT_V1,
    SEED_ARTIFACT_V1,
    SEED_QUALIFICATION_RECEIPT_V1,
    SAFEGUARD_QUALIFICATION_RECEIPT_V1,
    SCIENTIFIC_PRE_SUBMIT_AUDIT_V1,
    SOURCE_EPOCH_RECEIPT_V1,
    SOURCE_TREE_MANIFEST_V1,
    ExactReferenceReceipt,
    ExecutorBinding,
    ProspectiveExecutionError,
    RunLifecycleReceipt,
    ScoreQualificationReceipt,
    SafeguardQualificationReceipt,
    ScientificPreSubmitAudit,
    SourceEpochReceipt,
    SeedQualificationReceipt,
    VerifiedFile,
    binding_registry_payload,
    build_package_execution_request,
    build_scientific_bindings,
    durable_receipts_to_evidence,
    executor_bindings,
    load_action_registry_entry,
    load_scientific_state,
    package_request,
    report_slot_request,
    resolve_bootstrap_action,
    resolve_scientific_action,
    score_receipts_to_evidence,
    validate_seed_receipt_collection,
    write_action_registry_entry,
    write_run_lifecycle_receipt,
    write_scientific_pre_submit_audit,
    write_scientific_state,
    update_report_slot_execution_request,
)
from pipelines.time_dynamics.planned_report_contracts import (
    build_planned_report_contract,
    update_exact_reference,
    update_method_slot,
)
from pipelines.time_dynamics.prospective_scientific_planner import (
    FAILURE_MEMORY,
    HORIZON_FULL,
    HORIZON_PREFIX,
    KIND_RECORD_SAFEGUARD,
    ProspectiveEvidence,
    ProspectiveNextAction,
    ProspectivePlannerError,
    SAFEGUARD_LADDER_UNRESOLVED,
    prospective_safeguard_id,
    plan_prospective_cell,
)
from pipelines.time_dynamics.staged_campaign import AccuracyPolicy


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return _sha(path)


def _canonical_sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


def _sealed(payload: dict, key: str) -> dict:
    result = dict(payload)
    result.pop(key, None)
    result[key] = _canonical_sha(result)
    return result


def _terminal_campaign_registry(
    root: Path,
    *,
    bindings,
    scientific_state,
    planned_report_contract: VerifiedFile,
    matrix_path: Path,
) -> CampaignRegistry:
    matrix = ParallelBenchmarkMatrix.read(matrix_path)
    report_payload = json.loads(
        (root / planned_report_contract.path).read_text()
    )
    lane = LaneRegistration(
        lane_id=bindings.lane_id,
        kind="prospective_test_lane",
        monitor_enabled=False,
        submission_enabled=False,
        attempts_root=root / "output/terminal/attempts",
        state_root=root / "output/terminal/state",
        scientific_state=root / scientific_state.receipt_path,
        report_contract=root / planned_report_contract.path,
        planner_authority=(
            "pipelines.time_dynamics.prospective_scientific_planner:"
            "plan_prospective_cell"
        ),
        blocked_by=("scientific_pre_submit_audit_binding",),
    )
    return CampaignRegistry(
        registry_id="prospective_test_registry",
        repository_root=root,
        supervisor_state_root=root / "output/terminal/supervisor",
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
                    "path": planned_report_contract.path,
                    "file_sha256": planned_report_contract.sha256,
                    "report_contract_sha256": report_payload[
                        "report_contract_sha256"
                    ],
                    "tuple_structure_sha256": report_payload[
                        "tuple_structure_sha256"
                    ],
                }
            },
        },
    )


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


def _source_epoch() -> SourceEpochIdentity:
    return SourceEpochIdentity(
        lane_id=LANE_HH,
        source_epoch_id="paper-ii-hh-product-test-source",
        source_revision="a" * 40,
    )


def _prospective_rows():
    registry = build_identity_registry()
    reference = next(row for row in registry.references if row.lane_id == LANE_HH)
    seed = next(row for row in registry.seeds if row.seed_id == reference.seed_id)
    return registry, seed, reference


def _source_receipt(
    tmp_path: Path,
    source_epoch: SourceEpochIdentity | None = None,
    *,
    token: str = "",
) -> SourceEpochReceipt:
    epoch = source_epoch or _source_epoch()
    tree_sha = "d" * 64
    manifest = tmp_path / (
        f"source/{token}_tree_manifest.json" if token else "source/tree_manifest.json"
    )
    _write(
        manifest,
        _sealed(
            {
                "schema": SOURCE_TREE_MANIFEST_V1,
                "lane_id": epoch.lane_id,
                "source_epoch_identity_sha256": epoch.identity_sha256,
                "source_revision": epoch.source_revision,
                "source_archive_sha256": epoch.source_archive_sha256,
                "source_tree_sha256": tree_sha,
                "provenance_kind": "git_archive",
            },
            "source_tree_manifest_sha256",
        ),
    )
    receipt = tmp_path / (
        f"receipts/{token}_source_epoch.json" if token else "receipts/source_epoch.json"
    )
    _write(
        receipt,
        _sealed(
            {
                "schema": SOURCE_EPOCH_RECEIPT_V1,
                "source_epoch": epoch.to_json_dict(),
                "source_tree_manifest": {
                    "path": manifest.relative_to(tmp_path).as_posix(),
                    "sha256": _sha(manifest),
                },
                "source_tree_sha256": tree_sha,
            },
            "source_epoch_receipt_sha256",
        ),
    )
    return SourceEpochReceipt.load(
        receipt,
        repository_root=tmp_path,
        expected_epoch=epoch,
    )


def _seed_receipt(
    tmp_path: Path,
    *,
    energy_error: float = 5.0e-7,
    source_epoch: SourceEpochIdentity | None = None,
    seed_identity=None,
    token: str = "",
) -> SeedQualificationReceipt:
    _registry, default_seed, _reference = _prospective_rows()
    seed = seed_identity or default_seed
    epoch = source_epoch or _source_epoch()
    variational = -1.25 + energy_error
    exact = -1.25
    optimizer_execution = f"optimizer::{seed.seed_id}"
    artifact = tmp_path / (
        f"evidence/{token}_seed_artifact.json" if token else seed.artifact_path
    )
    _write(
        artifact,
        _sealed(
            {
                "schema": SEED_ARTIFACT_V1,
                "seed_id": seed.seed_id,
                "seed_identity_sha256": seed.identity_sha256,
                "lane_id": seed.lane_id,
                "model": seed.model,
                "physical": dict(seed.physical),
                "seed_policy": seed.policy,
                "initializer": seed.initializer,
                "restarts": seed.restarts,
                "optimizer_seed": seed.optimizer_seed,
                "optimizer_execution_id": optimizer_execution,
                "source_epoch_identity_sha256": epoch.identity_sha256,
                "variational_energy": variational,
                "exact_energy": exact,
                "exact_state_infidelity": 2.0e-9,
            },
            "seed_artifact_sha256",
        ),
    )
    receipt = tmp_path / (
        f"receipts/{token}_seed.json" if token else "receipts/seed.json"
    )
    _write(
        receipt,
        _sealed(
            {
                "schema": SEED_QUALIFICATION_RECEIPT_V1,
                "seed_id": seed.seed_id,
                "seed_identity_sha256": seed.identity_sha256,
                "artifact": {
                    "path": artifact.relative_to(tmp_path).as_posix(),
                    "sha256": _sha(artifact),
                },
                "energy_error": energy_error,
                "exact_state_infidelity": 2.0e-9,
                "initializer": seed.initializer,
                "restarts": seed.restarts,
                "optimizer_seed": seed.optimizer_seed,
                "optimizer_execution_id": optimizer_execution,
                "variational_energy": variational,
                "exact_energy": exact,
                "source_epoch_identity_sha256": epoch.identity_sha256,
            },
            "seed_receipt_sha256",
        ),
    )
    return SeedQualificationReceipt.load(
        receipt,
        repository_root=tmp_path,
        expected_seed=seed,
        expected_source_epoch=epoch,
    )


def _reference_receipt(
    tmp_path: Path,
    seed_receipt: SeedQualificationReceipt,
    *,
    source_epoch: SourceEpochIdentity | None = None,
    reference_identity=None,
    seed_identity=None,
    token: str = "",
) -> ExactReferenceReceipt:
    _registry, default_seed, default_reference = _prospective_rows()
    seed = seed_identity or default_seed
    reference = reference_identity or default_reference
    epoch = source_epoch or _source_epoch()
    materialized_key = reference.materialized_key(seed_receipt.artifact.sha256)
    data = tmp_path / (
        f"exact/{token}_reference.json" if token else "exact/reference.json"
    )
    _write(
        data,
        _prospective_exact_data(
            reference,
            materialized_key=materialized_key,
        ),
    )
    response_scales = {
        horizon: {
            "energy_rms_excursion": 0.2,
            "doublon_rms_excursion": 0.1,
            "max_site_occupation_rms_excursion": 0.1,
        }
        for horizon in (HORIZON_PREFIX, HORIZON_FULL)
    }
    response_scales_sha256 = _canonical_sha(response_scales)
    manifest = tmp_path / (
        f"exact/{token}_manifest.json" if token else "exact/manifest.json"
    )
    _write(
        manifest,
        _sealed(
            {
                "schema": EXACT_REFERENCE_MANIFEST_V1,
                "reference_contract": {
                    "reference_id": reference.reference_id,
                    "reference_identity_sha256": reference.identity_sha256,
                    "lane_id": reference.lane_id,
                    "cell_id": reference.cell_id,
                    "seed_id": reference.seed_id,
                    "seed_identity_sha256": seed.identity_sha256,
                    "model": reference.model,
                    "physical": dict(reference.physical),
                    "drive": dict(reference.drive),
                    "grid": dict(reference.grid),
                    "propagator": reference.propagator,
                    "observable_schema": reference.observable_schema,
                },
                "materialized_reference_key": materialized_key,
                "source_epoch_identity_sha256": epoch.identity_sha256,
                "seed_artifact_sha256": seed_receipt.artifact.sha256,
                "data_sha256": _sha(data),
                "response_scales": response_scales,
                "response_scales_sha256": response_scales_sha256,
            },
            "exact_manifest_sha256",
        ),
    )
    receipt = tmp_path / (
        f"receipts/{token}_reference.json" if token else "receipts/reference.json"
    )
    _write(
        receipt,
        _sealed({
            "schema": EXACT_REFERENCE_RECEIPT_V1,
            "reference_id": reference.reference_id,
            "reference_identity_sha256": reference.identity_sha256,
            "lane_id": reference.lane_id,
            "cell_id": reference.cell_id,
            "seed_id": seed.seed_id,
            "seed_artifact_sha256": seed_receipt.artifact.sha256,
            "manifest": {
                "path": manifest.relative_to(tmp_path).as_posix(),
                "sha256": _sha(manifest),
            },
            "data": {
                "path": data.relative_to(tmp_path).as_posix(),
                "sha256": _sha(data),
            },
            "source_epoch_identity_sha256": epoch.identity_sha256,
            "materialized_reference_key": materialized_key,
            "reference_contract": {
                "reference_id": reference.reference_id,
                "reference_identity_sha256": reference.identity_sha256,
                "lane_id": reference.lane_id,
                "cell_id": reference.cell_id,
                "seed_id": reference.seed_id,
                "seed_identity_sha256": seed.identity_sha256,
                "model": reference.model,
                "physical": dict(reference.physical),
                "drive": dict(reference.drive),
                "grid": dict(reference.grid),
                "propagator": reference.propagator,
                "observable_schema": reference.observable_schema,
            },
            "response_scales": response_scales,
            "response_scales_sha256": response_scales_sha256,
        }, "exact_reference_receipt_sha256"),
    )
    return ExactReferenceReceipt.load(
        receipt,
        repository_root=tmp_path,
        expected_reference=reference,
        expected_seed_identity=seed,
        expected_seed_receipt=seed_receipt,
        expected_source_epoch=epoch,
    )


def _policy(drive: str) -> AccuracyPolicy:
    scales = {
        drive: {
            horizon: {
                "energy_rms_excursion": 0.2,
                "doublon_rms_excursion": 0.1,
                "max_site_occupation_rms_excursion": 0.1,
            }
            for horizon in (HORIZON_PREFIX, HORIZON_FULL)
        }
    }
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


def _scientific_context(
    tmp_path: Path,
    *,
    reference_identity=None,
    token: str = "",
) -> dict:
    registry, _default_seed, default_reference = _prospective_rows()
    reference_identity = reference_identity or default_reference
    seed_identity = next(
        row
        for row in registry.seeds
        if row.seed_id == reference_identity.seed_id
    )
    source = _source_epoch()
    seed = _seed_receipt(
        tmp_path,
        source_epoch=source,
        seed_identity=seed_identity,
        token=token,
    )
    reference = _reference_receipt(
        tmp_path,
        seed,
        source_epoch=source,
        reference_identity=reference_identity,
        seed_identity=seed_identity,
        token=token,
    )
    source_receipt = _source_receipt(tmp_path, source, token=token)
    policy = _policy(str(reference_identity.drive["drive_id"]))
    bindings = build_scientific_bindings(
        reference=reference_identity,
        seed=seed_identity,
        seed_receipt=seed,
        reference_receipt=reference,
        source_epoch=source,
        source_epoch_receipt=source_receipt,
    )
    return {
        "seed_identity": seed_identity,
        "reference_identity": reference_identity,
        "source": source,
        "seed": seed,
        "reference": reference,
        "source_receipt": source_receipt,
        "policy": policy,
        "bindings": bindings,
    }


def _action_registry_binding(
    tmp_path: Path, context: dict, action, *, token: str = "run"
) -> tuple[object, VerifiedFile]:
    request = resolve_scientific_action(context["bindings"], action)
    action_path = tmp_path / f"state/actions/{token}.json"
    write_action_registry_entry(
        action_path,
        request,
        source_receipts=(
            VerifiedFile(
                context["source_receipt"].receipt_path,
                context["source_receipt"].receipt_sha256,
            ),
            VerifiedFile(
                context["seed"].receipt_path,
                context["seed"].receipt_sha256,
            ),
            VerifiedFile(
                context["reference"].receipt_path,
                context["reference"].receipt_sha256,
            ),
        ),
        repository_root=tmp_path,
        bindings=context["bindings"],
        expected_seed=context["seed_identity"],
        expected_reference=context["reference_identity"],
        expected_source_epoch=context["source"],
    )
    return request, VerifiedFile(
        action_path.relative_to(tmp_path).as_posix(), _sha(action_path)
    )


def _run_state_evidence(
    tmp_path: Path, action, state: str, *, token: str
) -> VerifiedFile:
    target = tmp_path / f"evidence/state_{token}.json"
    _write(
        target,
        _sealed(
            {
                "schema": RUN_STATE_EVIDENCE_V1,
                "scientific_action_id": action.action_id,
                "execution_id": action.execution_id,
                "state": state,
            },
            "run_state_evidence_sha256",
        ),
    )
    return VerifiedFile(target.relative_to(tmp_path).as_posix(), _sha(target))


def _run_failure_evidence(
    tmp_path: Path,
    action,
    *,
    failure_kind: str,
    failure_reason: str,
    observed_peak_gib: float | None,
    token: str,
) -> VerifiedFile:
    target = tmp_path / f"evidence/failure_{token}.json"
    _write(
        target,
        _sealed(
            {
                "schema": RUN_FAILURE_EVIDENCE_V1,
                "scientific_action_id": action.action_id,
                "execution_id": action.execution_id,
                "failure_kind": failure_kind,
                "failure_reason": failure_reason,
                "observed_peak_gib": observed_peak_gib,
            },
            "run_failure_evidence_sha256",
        ),
    )
    return VerifiedFile(target.relative_to(tmp_path).as_posix(), _sha(target))


def test_symbolic_executor_ids_resolve_to_real_callables() -> None:
    payload = binding_registry_payload()
    assert payload["production_submission_enabled"] is False
    assert payload["qiskit_compilation"] is False
    assert len(payload["bindings"]) == len(executor_bindings())
    assert len({row["binding_id"] for row in payload["bindings"]}) == len(
        payload["bindings"]
    )
    assert all(row.resolve() for row in executor_bindings())
    score = next(
        row for row in executor_bindings()
        if row.binding_id == EXEC_PROSPECTIVE_SCORE
    )
    assert score.handler == (
        "pipelines.time_dynamics.prospective_trajectory_scoring_worker:"
        "materialization_request"
    )
    avqds = next(
        row for row in executor_bindings()
        if row.binding_id == EXEC_PROSPECTIVE_AVQDS
    )
    assert avqds.handler == (
        "pipelines.time_dynamics.prospective_avqds_trajectory_worker:"
        "materialization_request"
    )


def test_missing_executor_handler_fails_closed() -> None:
    binding = ExecutorBinding(
        binding_id="missing",
        lanes=(LANE_HH,),
        kinds=("build_seed",),
        handler="pipelines.time_dynamics.prospective_execution_bindings:no_such_handler",
        output_role="none",
    )
    with pytest.raises(ProspectiveExecutionError, match="not callable"):
        binding.resolve()


def test_bootstrap_action_resolver_enforces_source_and_no_qiskit() -> None:
    registry = build_identity_registry()
    available = tuple(row.binding_id for row in executor_bindings())
    plan = plan_bootstrap_actions(
        registry,
        evidence=BootstrapEvidence(source_epochs={LANE_HH: _source_epoch()}),
        available_executors=available,
    )
    action = next(row for row in plan.ready_actions if row.lane_id == LANE_HH)
    request = resolve_bootstrap_action(action)
    assert request.action_id == action.action_id
    assert request.declared_outputs == (
        "seed_artifact",
        "seed_qualification_receipt",
    )
    assert request.to_json_dict()["qiskit_compilation_requested"] is False
    request_handler = next(
        row for row in executor_bindings() if row.binding_id == request.binding_id
    ).resolve()
    prepared = request_handler(request)
    assert prepared["status"] == "prepared_not_executed"
    assert prepared["completion_receipt"] is None
    assert prepared["execution_request"]["execution_request_sha256"] == request.digest


def test_avqds_binding_resolves_to_inert_operational_materialization_adapter() -> None:
    registry = build_identity_registry()
    seed = next(row for row in registry.seeds if row.lane_id == LANE_HH)
    plan = plan_bootstrap_actions(
        registry,
        evidence=BootstrapEvidence(
            seed_artifact_sha256={seed.seed_id: "a" * 64},
            source_epochs={LANE_HH: _source_epoch()},
        ),
        available_executors=tuple(row.binding_id for row in executor_bindings()),
    )
    action = next(
        row
        for row in plan.ready_actions
        if row.kind == "screen_avqds" and row.lane_id == LANE_HH
    )
    request = resolve_bootstrap_action(action)
    binding = next(
        row for row in executor_bindings() if row.binding_id == request.binding_id
    )

    assert binding.handler == (
        "pipelines.time_dynamics.prospective_avqds_scout_worker:"
        "materialization_request"
    )
    adapter = binding.resolve()(request)
    assert adapter["status"] == "operational_materialization_ready"
    assert adapter["runner_module"] == (
        "pipelines.time_dynamics.runners.run_prospective_avqds_scout"
    )
    assert adapter["required_receipts"] == [
        "source_epoch_receipt",
        "seed_qualification_receipt",
    ]
    assert adapter["scheduler_contacted"] is False
    assert adapter["scheduler_submission_authorized"] is False
    assert adapter["production_submission_enabled"] is False


def test_exact_binding_resolves_to_authenticated_inert_materialization_adapter() -> None:
    registry = build_identity_registry()
    seed = next(row for row in registry.seeds if row.lane_id == LANE_HH)
    plan = plan_bootstrap_actions(
        registry,
        evidence=BootstrapEvidence(
            seed_artifact_sha256={seed.seed_id: "a" * 64},
            source_epochs={LANE_HH: _source_epoch()},
        ),
        available_executors=tuple(row.binding_id for row in executor_bindings()),
    )
    action = next(
        row
        for row in plan.ready_actions
        if row.kind == "build_exact_reference" and row.lane_id == LANE_HH
    )
    request = resolve_bootstrap_action(action)
    binding = next(
        row for row in executor_bindings() if row.binding_id == request.binding_id
    )

    assert binding.handler == (
        "pipelines.time_dynamics.prospective_exact_reference_worker:"
        "materialization_request"
    )
    adapter = binding.resolve()(request)
    assert adapter["status"] == "operational_materialization_ready"
    assert adapter["runner_module"] == (
        "pipelines.time_dynamics.runners.build_prospective_exact_reference"
    )
    assert adapter["execution_request_sha256"] == request.digest
    assert adapter["required_receipts"] == [
        "source_epoch_receipt",
        "seed_qualification_receipt",
    ]
    assert adapter["scheduler_contacted"] is False
    assert adapter["scheduler_submission_authorized"] is False
    assert adapter["production_submission_enabled"] is False


def test_seed_receipt_enforces_energy_gate_and_verified_artifact(tmp_path: Path) -> None:
    good = _seed_receipt(tmp_path)
    assert good.energy_error <= 1.0e-6
    assert good.exact_state_infidelity == pytest.approx(2.0e-9)
    assert good.restarts == 4

    with pytest.raises(ProspectiveExecutionError, match="Delta E"):
        _seed_receipt(tmp_path / "bad", energy_error=1.1e-6)

    (tmp_path / good.artifact.path).write_bytes(b"tampered")
    with pytest.raises(ProspectiveExecutionError, match="artifact digest mismatch"):
        SeedQualificationReceipt.load(
            tmp_path / good.receipt_path,
            repository_root=tmp_path,
            expected_seed=_prospective_rows()[1],
            expected_source_epoch=_source_epoch(),
        )


def test_seed_receipt_rejects_self_valid_forged_energy_claim(tmp_path: Path) -> None:
    good = _seed_receipt(tmp_path)
    receipt_path = tmp_path / good.receipt_path
    payload = json.loads(receipt_path.read_text())
    payload["energy_error"] = good.energy_error / 2.0
    _write(receipt_path, _sealed(payload, "seed_receipt_sha256"))

    with pytest.raises(ProspectiveExecutionError, match="not reproducible"):
        SeedQualificationReceipt.load(
            receipt_path,
            repository_root=tmp_path,
            expected_seed=_prospective_rows()[1],
            expected_source_epoch=_source_epoch(),
        )


def test_exact_reference_receipt_is_bound_to_the_same_seed(tmp_path: Path) -> None:
    seed = _seed_receipt(tmp_path)
    reference = _reference_receipt(tmp_path, seed)
    assert reference.seed_artifact_sha256 == seed.artifact.sha256

    manifest = tmp_path / reference.manifest.path
    payload = json.loads(manifest.read_text())
    payload["reference_contract"]["drive"] = {
        **payload["reference_contract"]["drive"],
        "amplitude": 99.0,
    }
    _write(manifest, _sealed(payload, "exact_manifest_sha256"))
    receipt_path = tmp_path / reference.receipt_path
    receipt_payload = json.loads(receipt_path.read_text())
    receipt_payload["manifest"]["sha256"] = _sha(manifest)
    _write(
        receipt_path,
        _sealed(receipt_payload, "exact_reference_receipt_sha256"),
    )
    with pytest.raises(ProspectiveExecutionError, match="registered identity"):
        ExactReferenceReceipt.load(
            receipt_path,
            repository_root=tmp_path,
            expected_reference=_prospective_rows()[2],
            expected_seed_identity=_prospective_rows()[1],
            expected_seed_receipt=seed,
            expected_source_epoch=_source_epoch(),
        )


def test_exact_reference_rejects_wrong_drive_with_same_seed(tmp_path: Path) -> None:
    registry, seed_identity, expected = _prospective_rows()
    wrong_drive = next(
        row
        for row in registry.references
        if row.lane_id == expected.lane_id
        and row.seed_id == expected.seed_id
        and row.cell_id != expected.cell_id
        and row.drive != expected.drive
    )
    seed = _seed_receipt(tmp_path)
    forged = _reference_receipt(
        tmp_path,
        seed,
        reference_identity=wrong_drive,
        seed_identity=seed_identity,
    )

    with pytest.raises(ProspectiveExecutionError, match="registered identity"):
        ExactReferenceReceipt.load(
            tmp_path / forged.receipt_path,
            repository_root=tmp_path,
            expected_reference=expected,
            expected_seed_identity=seed_identity,
            expected_seed_receipt=seed,
            expected_source_epoch=_source_epoch(),
        )


def test_exact_reference_response_scales_are_digest_bound(tmp_path: Path) -> None:
    seed = _seed_receipt(tmp_path)
    reference = _reference_receipt(tmp_path, seed)
    receipt_path = tmp_path / reference.receipt_path
    payload = json.loads(receipt_path.read_text())
    payload["response_scales"][HORIZON_PREFIX][
        "energy_rms_excursion"
    ] *= 10.0
    _write(receipt_path, _sealed(payload, "exact_reference_receipt_sha256"))

    with pytest.raises(ProspectiveExecutionError, match="response-scale digest"):
        ExactReferenceReceipt.load(
            receipt_path,
            repository_root=tmp_path,
            expected_reference=_prospective_rows()[2],
            expected_seed_identity=_prospective_rows()[1],
            expected_seed_receipt=seed,
            expected_source_epoch=_source_epoch(),
        )


def test_exact_reference_rejects_self_consistent_forged_response_scales(
    tmp_path: Path,
) -> None:
    seed = _seed_receipt(tmp_path)
    reference = _reference_receipt(tmp_path, seed)
    manifest_path = tmp_path / reference.manifest.path
    receipt_path = tmp_path / reference.receipt_path

    manifest = json.loads(manifest_path.read_text())
    forged = {
        horizon: {name: value * 1000.0 for name, value in row.items()}
        for horizon, row in manifest["response_scales"].items()
    }
    manifest["response_scales"] = forged
    manifest["response_scales_sha256"] = _canonical_sha(forged)
    _write(manifest_path, _sealed(manifest, "exact_manifest_sha256"))

    receipt = json.loads(receipt_path.read_text())
    receipt["response_scales"] = forged
    receipt["response_scales_sha256"] = _canonical_sha(forged)
    receipt["manifest"]["sha256"] = _sha(manifest_path)
    _write(receipt_path, _sealed(receipt, "exact_reference_receipt_sha256"))

    with pytest.raises(ProspectiveExecutionError, match="exact trajectory data"):
        ExactReferenceReceipt.load(
            receipt_path,
            repository_root=tmp_path,
            expected_reference=_prospective_rows()[2],
            expected_seed_identity=_prospective_rows()[1],
            expected_seed_receipt=seed,
            expected_source_epoch=_source_epoch(),
        )


def test_loaded_exact_response_scales_are_deeply_immutable(tmp_path: Path) -> None:
    reference = _reference_receipt(tmp_path, _seed_receipt(tmp_path))

    with pytest.raises(TypeError):
        reference.response_scales[HORIZON_PREFIX][
            "energy_rms_excursion"
        ] = 999.0


def test_exact_reference_rejects_self_consistent_grid_substitution(
    tmp_path: Path,
) -> None:
    seed = _seed_receipt(tmp_path)
    reference = _reference_receipt(tmp_path, seed)
    data_path = tmp_path / reference.data.path
    manifest_path = tmp_path / reference.manifest.path
    receipt_path = tmp_path / reference.receipt_path

    data = json.loads(data_path.read_text())
    data["time"][1] += 1.0e-3
    _write(data_path, data)
    manifest = json.loads(manifest_path.read_text())
    manifest["data_sha256"] = _sha(data_path)
    _write(manifest_path, _sealed(manifest, "exact_manifest_sha256"))
    receipt = json.loads(receipt_path.read_text())
    receipt["data"]["sha256"] = _sha(data_path)
    receipt["manifest"]["sha256"] = _sha(manifest_path)
    _write(receipt_path, _sealed(receipt, "exact_reference_receipt_sha256"))

    with pytest.raises(ProspectiveExecutionError, match="expected grid"):
        ExactReferenceReceipt.load(
            receipt_path,
            repository_root=tmp_path,
            expected_reference=_prospective_rows()[2],
            expected_seed_identity=_prospective_rows()[1],
            expected_seed_receipt=seed,
            expected_source_epoch=_source_epoch(),
        )


def test_bridge_rejects_same_named_cell_with_altered_waveform(tmp_path: Path) -> None:
    _registry, seed_identity, reference_identity = _prospective_rows()
    source = _source_epoch()
    seed = _seed_receipt(tmp_path, source_epoch=source)
    forged_identity = replace(
        reference_identity,
        drive={**dict(reference_identity.drive), "amplitude": 99.0},
    )
    forged_receipt = _reference_receipt(
        tmp_path,
        seed,
        source_epoch=source,
        reference_identity=forged_identity,
        seed_identity=seed_identity,
    )

    with pytest.raises(ProspectiveExecutionError, match="canonical matrix reference"):
        build_scientific_bindings(
            reference=forged_identity,
            seed=seed_identity,
            seed_receipt=seed,
            reference_receipt=forged_receipt,
            source_epoch=source,
            source_epoch_receipt=_source_receipt(tmp_path, source),
        )


def test_bridge_rejects_noncanonical_seed_identity(tmp_path: Path) -> None:
    _registry, seed_identity, reference_identity = _prospective_rows()
    source = _source_epoch()
    seed = _seed_receipt(tmp_path, source_epoch=source)
    reference = _reference_receipt(tmp_path, seed, source_epoch=source)

    with pytest.raises(ProspectiveExecutionError, match="canonical matrix seed"):
        build_scientific_bindings(
            reference=reference_identity,
            seed=replace(seed_identity, policy="cross_regime_warm_start"),
            seed_receipt=seed,
            reference_receipt=reference,
            source_epoch=source,
            source_epoch_receipt=_source_receipt(tmp_path, source),
        )


def test_planner_rejects_self_valid_bindings_with_forged_matrix_digests(
    tmp_path: Path,
) -> None:
    context = _scientific_context(tmp_path)
    forged = replace(
        context["bindings"],
        matrix_raw_sha256="0" * 64,
        matrix_payload_sha256="1" * 64,
        matrix_expanded_semantic_sha256="2" * 64,
        matrix_accuracy_policy_sha256="3" * 64,
    )

    with pytest.raises(ProspectivePlannerError, match="matrix"):
        plan_prospective_cell(forged, context["policy"])


def test_persisted_bindings_reject_self_valid_weakened_policy(tmp_path: Path) -> None:
    context = _scientific_context(tmp_path)
    weak_policy = replace(
        context["policy"],
        absolute_energy_error_ceiling=1.0,
        absolute_doublon_error_ceiling=1.0,
        absolute_site_occupation_error_ceiling=1.0,
    )
    forged = replace(
        context["bindings"],
        accuracy_policy=weak_policy.to_json_dict(),
        accuracy_policy_sha256=_canonical_sha(weak_policy.to_json_dict()),
    )

    with pytest.raises(ProspectivePlannerError, match="weaken|matrix policy"):
        type(forged).from_json_dict(forged.to_json_dict())


def test_action_registry_rejoins_policy_to_exact_reference_scales(
    tmp_path: Path,
) -> None:
    context = _scientific_context(tmp_path)
    payload = context["bindings"].to_json_dict()
    policy_payload = payload["accuracy_policy"]
    drive = context["bindings"].drive
    policy_payload["response_scales"][drive][HORIZON_PREFIX][
        "energy_rms_excursion"
    ] *= 10.0
    payload["accuracy_policy_sha256"] = _canonical_sha(policy_payload)
    payload.pop("bindings_sha256")
    payload["bindings_sha256"] = _canonical_sha(payload)
    forged_bindings = type(context["bindings"]).from_json_dict(payload)
    forged_policy = AccuracyPolicy.from_json_dict(policy_payload)
    action = plan_prospective_cell(forged_bindings, forged_policy).next_actions[0]
    request = resolve_scientific_action(forged_bindings, action)

    with pytest.raises(ProspectiveExecutionError, match="reference response scales"):
        write_action_registry_entry(
            tmp_path / "state/actions/forged_scales.json",
            request,
            source_receipts=(
                VerifiedFile(
                    context["source_receipt"].receipt_path,
                    context["source_receipt"].receipt_sha256,
                ),
                VerifiedFile(
                    context["seed"].receipt_path,
                    context["seed"].receipt_sha256,
                ),
                VerifiedFile(
                    context["reference"].receipt_path,
                    context["reference"].receipt_sha256,
                ),
            ),
            repository_root=tmp_path,
            bindings=forged_bindings,
            expected_seed=context["seed_identity"],
            expected_reference=context["reference_identity"],
            expected_source_epoch=context["source"],
        )


def test_scientific_resolver_rejects_unsupported_arm_and_preserves_retry_identity(
    tmp_path: Path,
) -> None:
    context = _scientific_context(tmp_path)
    action = plan_prospective_cell(
        context["bindings"], context["policy"]
    ).next_actions[0]
    unsupported = replace(
        action,
        arm="insertion_mclachlan",
        inputs={**action.inputs, "arm": "insertion_mclachlan"},
    )
    with pytest.raises(ProspectiveExecutionError, match="unsupported arm"):
        resolve_scientific_action(context["bindings"], unsupported)

    retry = replace(
        action,
        execution_id=f"{action.action_id}::attempt1::mem16g",
        attempt=1,
        planner_sequence=7,
        requested_memory_gib=16,
        inputs={**action.inputs, "resource_only_retry": True},
    )
    request = resolve_scientific_action(context["bindings"], retry)
    assert request.execution_id == retry.execution_id
    assert request.attempt == 1
    assert request.planner_sequence == 7
    assert request.requested_memory_gib == 16


def test_typed_receipts_bridge_into_scientific_planner_and_resolver(
    tmp_path: Path,
) -> None:
    _registry, seed_identity, reference_identity = _prospective_rows()
    source = _source_epoch()
    seed = _seed_receipt(tmp_path, source_epoch=source)
    reference = _reference_receipt(tmp_path, seed, source_epoch=source)
    source_receipt = _source_receipt(tmp_path, source)
    policy = _policy(str(reference_identity.drive["drive_id"]))
    bindings = build_scientific_bindings(
        reference=reference_identity,
        seed=seed_identity,
        seed_receipt=seed,
        reference_receipt=reference,
        source_epoch=source,
        source_epoch_receipt=source_receipt,
        policy=policy,
    )
    plan = plan_prospective_cell(bindings, policy)
    request = resolve_scientific_action(bindings, plan.next_actions[0])
    assert request.kind == "run"
    assert request.exact_reference_blind is True
    assert request.inputs["seed_artifact_sha256"] == seed.artifact.sha256
    assert request.inputs["reference_data_sha256"] == reference.data.sha256
    assert request.inputs["controller"] == "state_motion_5e-5"
    matrix = ParallelBenchmarkMatrix.read()
    assert bindings.matrix_raw_sha256 == matrix.raw_sha256
    assert bindings.matrix_payload_sha256 == matrix.payload_sha256
    assert (
        bindings.matrix_expanded_semantic_sha256
        == matrix.expanded_semantic_sha256
    )
    assert (
        bindings.matrix_accuracy_policy_sha256
        == matrix.accuracy_policy_spec_sha256
    )
    assert bindings.matrix_accuracy_policy_id == (
        "response_normalized_matched_pair_v2"
    )
    assert (
        bindings.to_json_dict()["accuracy_policy"]
        == policy.to_json_dict()
    )


def test_bridge_rejects_caller_policy_that_weakens_matrix_guards(
    tmp_path: Path,
) -> None:
    context = _scientific_context(tmp_path)
    weak_policy = replace(
        context["policy"],
        absolute_energy_error_ceiling=1.0,
        absolute_doublon_error_ceiling=1.0,
        absolute_site_occupation_error_ceiling=1.0,
    )
    with pytest.raises(ProspectiveExecutionError, match="caller-supplied"):
        build_scientific_bindings(
            reference=context["reference_identity"],
            seed=context["seed_identity"],
            seed_receipt=context["seed"],
            reference_receipt=context["reference"],
            source_epoch=context["source"],
            source_epoch_receipt=context["source_receipt"],
            policy=weak_policy,
        )


def test_bridge_rejects_cross_seed_and_cross_epoch_receipts(tmp_path: Path) -> None:
    _registry, seed_identity, reference_identity = _prospective_rows()
    source = _source_epoch()
    seed = _seed_receipt(tmp_path, source_epoch=source)
    reference = _reference_receipt(tmp_path, seed, source_epoch=source)
    source_receipt = _source_receipt(tmp_path, source)
    policy = _policy(str(reference_identity.drive["drive_id"]))
    with pytest.raises(ProspectiveExecutionError, match="another seed"):
        build_scientific_bindings(
            reference=reference_identity,
            seed=seed_identity,
            seed_receipt=seed,
            reference_receipt=replace(reference, seed_artifact_sha256="e" * 64),
            source_epoch=source,
            source_epoch_receipt=source_receipt,
            policy=policy,
        )
    with pytest.raises(ProspectiveExecutionError, match="source epochs"):
        build_scientific_bindings(
            reference=reference_identity,
            seed=seed_identity,
            seed_receipt=replace(seed, source_epoch_identity_sha256="e" * 64),
            reference_receipt=reference,
            source_epoch=source,
            source_epoch_receipt=source_receipt,
            policy=policy,
        )


def _score_receipt(
    tmp_path: Path,
    *,
    bindings,
    reference_receipt: ExactReferenceReceipt,
    policy: AccuracyPolicy,
    action,
    action_registry: VerifiedFile,
    planner_payload: dict | None = None,
) -> ScoreQualificationReceipt:
    token = hashlib.sha256(action.action_id.encode()).hexdigest()[:10]
    run = tmp_path / f"evidence/run_{token}.json"
    planner = tmp_path / f"evidence/planner_{token}.json"
    metrics = tmp_path / f"evidence/metrics_{token}.json"
    _write(run, {"status": "complete"})
    request = resolve_scientific_action(bindings, action)
    submitted = write_run_lifecycle_receipt(
        tmp_path / f"evidence/run_receipt_{token}_submitted.json",
        request=request,
        state="submitted",
        action_registry=action_registry,
        state_evidence=_run_state_evidence(
            tmp_path, action, "submitted", token=f"{token}_submitted"
        ),
        repository_root=tmp_path,
    )
    run_receipt = write_run_lifecycle_receipt(
        tmp_path / f"evidence/run_receipt_{token}_completed.json",
        request=request,
        state="completed",
        action_registry=action_registry,
        state_evidence=_run_state_evidence(
            tmp_path, action, "completed", token=f"{token}_completed"
        ),
        previous_receipt=VerifiedFile(
            submitted.receipt_path, submitted.receipt_sha256
        ),
        run=VerifiedFile(run.relative_to(tmp_path).as_posix(), _sha(run)),
        repository_root=tmp_path,
    )
    if planner_payload is None:
        planner_payload = {
            "schema": "paper_ii_prospective_scientific_plan_v1",
            "bindings_sha256": bindings.digest,
            "phase": "score_recorded",
            "next_actions": [],
            "dispositions": [],
            "selected_prefix_action_ids": {},
            "selected_full_action_ids": (
                {action.arm: action.action_id}
                if action.horizon == HORIZON_FULL
                else {}
            ),
            "complete": False,
            "unresolved": False,
            "blocked": False,
            "cost_comparison_admissible": False,
            "qiskit_compilation_requested": False,
        }
    _write(planner, _sealed(planner_payload, "planner_receipt_sha256"))
    accuracy = {
        "mean_abs_energy_error": 1.0e-4,
        "mean_abs_doublon_error": 5.0e-5,
        "mean_max_abs_site_occupation_error": 5.0e-5,
        "mean_infidelity": 1.0e-5,
    }
    run_receipt_sha = run_receipt.receipt_sha256
    resume_action = action.inputs.get("resume_from_action_id")
    resume_receipt = action.inputs.get("resume_run_receipt_sha256")
    resume_run = action.inputs.get("resume_run_sha256")
    _write(
        metrics,
        _sealed({
            "schema": SCORE_METRICS_V1,
            "score_action_id": f"score::{action.action_id}",
            "scientific_action_id": action.action_id,
            "execution_id": action.execution_id,
            "cell_id": bindings.cell_id,
            "method": action.arm,
            "horizon": action.horizon,
            "activation_cut": action.activation_cut,
            "attempt": action.attempt,
            "planner_sequence": action.planner_sequence,
            "requested_memory_gib": action.requested_memory_gib,
            "run_receipt_sha256": run_receipt_sha,
            "resume_from_action_id": resume_action,
            "resume_run_receipt_sha256": resume_receipt,
            "resume_run_sha256": resume_run,
            "bindings_sha256": bindings.digest,
            "source_epoch_identity_sha256": bindings.source_epoch_identity_sha256,
            "exact_reference_id": bindings.reference_id,
            "exact_reference_identity_sha256": bindings.reference_identity_sha256,
            "exact_reference_receipt_sha256": reference_receipt.receipt_sha256,
            "exact_manifest_sha256": bindings.reference_manifest_sha256,
            "exact_data_sha256": bindings.reference_data_sha256,
            "run_sha256": _sha(run),
            "accuracy_policy_sha256": bindings.accuracy_policy_sha256,
            "accuracy_policy": policy.to_json_dict(),
            "metrics": accuracy,
            "report_values": {**accuracy, "accepted_steps": 42},
        }, "score_metrics_sha256"),
    )
    receipt = tmp_path / f"evidence/score_receipt_{token}.json"
    _write(
        receipt,
        _sealed({
            "schema": SCORE_QUALIFICATION_RECEIPT_V1,
            "score_action_id": f"score::{action.action_id}",
            "scientific_action_id": action.action_id,
            "execution_id": action.execution_id,
            "cell_id": bindings.cell_id,
            "method": action.arm,
            "horizon": action.horizon,
            "activation_cut": action.activation_cut,
            "attempt": action.attempt,
            "planner_sequence": action.planner_sequence,
            "requested_memory_gib": action.requested_memory_gib,
            "run_receipt": {
                "path": run_receipt.receipt_path,
                "sha256": run_receipt.receipt_sha256,
            },
            "action_registry": action_registry.to_json_dict(),
            "exact_reference_id": bindings.reference_id,
            "exact_reference_identity_sha256": bindings.reference_identity_sha256,
            "exact_reference_receipt": {
                "path": reference_receipt.receipt_path,
                "sha256": reference_receipt.receipt_sha256,
            },
            "exact_manifest_sha256": bindings.reference_manifest_sha256,
            "exact_data_sha256": bindings.reference_data_sha256,
            "bindings_sha256": bindings.digest,
            "source_epoch_identity_sha256": bindings.source_epoch_identity_sha256,
            "accuracy_policy_sha256": bindings.accuracy_policy_sha256,
            "metrics": {
                "path": metrics.relative_to(tmp_path).as_posix(),
                "sha256": _sha(metrics),
            },
            "planner_receipt": {
                "path": planner.relative_to(tmp_path).as_posix(),
                "sha256": _sha(planner),
            },
        }, "score_receipt_sha256"),
    )
    return ScoreQualificationReceipt.load(receipt, repository_root=tmp_path)


def _typed_safeguard_receipt(
    tmp_path: Path,
    *,
    context: dict,
    score: ScoreQualificationReceipt,
    token: str,
) -> tuple[SafeguardQualificationReceipt, VerifiedFile]:
    reason = SAFEGUARD_LADDER_UNRESOLVED
    safeguard_id = prospective_safeguard_id(
        context["bindings"],
        reason=reason,
        subject_action_id=score.scientific_action_id,
    )
    planner_sequence = score.planner_sequence + 1
    action = ProspectiveNextAction(
        kind=KIND_RECORD_SAFEGUARD,
        action_id=safeguard_id,
        execution_id=safeguard_id,
        arm=None,
        horizon=None,
        activation_cut=None,
        ladder_index=None,
        attempt=0,
        planner_sequence=planner_sequence,
        requested_memory_gib=None,
        reason="record exhausted ladder",
        inputs={
            **context["bindings"].to_json_dict(),
            "safeguard_reason": reason,
            "subject_action_id": score.scientific_action_id,
            "retain_all_numerical_evidence": True,
        },
        exact_reference_blind=False,
    )
    request = resolve_scientific_action(context["bindings"], action)
    action_path = tmp_path / f"state/actions/{token}_safeguard.json"
    write_action_registry_entry(
        action_path,
        request,
        source_receipts=(
            VerifiedFile(
                context["source_receipt"].receipt_path,
                context["source_receipt"].receipt_sha256,
            ),
            VerifiedFile(
                context["seed"].receipt_path,
                context["seed"].receipt_sha256,
            ),
            VerifiedFile(
                context["reference"].receipt_path,
                context["reference"].receipt_sha256,
            ),
        ),
        repository_root=tmp_path,
        bindings=context["bindings"],
        expected_seed=context["seed_identity"],
        expected_reference=context["reference_identity"],
        expected_source_epoch=context["source"],
    )
    action_registry = VerifiedFile(
        action_path.relative_to(tmp_path).as_posix(), _sha(action_path)
    )
    planner = tmp_path / f"evidence/{token}_terminal_planner.json"
    _write(
        planner,
        _sealed(
            {
                "schema": "paper_ii_prospective_scientific_plan_v1",
                "bindings_sha256": context["bindings"].digest,
                "phase": SAFEGUARD_LADDER_UNRESOLVED,
                "next_actions": [],
                "dispositions": [],
                "selected_prefix_action_ids": {
                    score.method: score.scientific_action_id
                },
                "selected_full_action_ids": {},
                "complete": True,
                "unresolved": True,
                "blocked": False,
                "cost_comparison_admissible": False,
                "qiskit_compilation_requested": False,
            },
            "planner_receipt_sha256",
        ),
    )
    receipt = tmp_path / f"evidence/{token}_safeguard_receipt.json"
    _write(
        receipt,
        _sealed(
            {
                "schema": SAFEGUARD_QUALIFICATION_RECEIPT_V1,
                "safeguard_action_id": safeguard_id,
                "reason": reason,
                "subject_action_id": score.scientific_action_id,
                "planner_sequence": planner_sequence,
                "cell_id": score.cell_id,
                "method": score.method,
                "bindings_sha256": context["bindings"].digest,
                "action_registry": action_registry.to_json_dict(),
                "subject_run_receipts": [score.run_receipt.to_json_dict()],
                "subject_score_receipt": {
                    "path": score.receipt_path,
                    "sha256": score.receipt_sha256,
                },
                "planner_receipt": {
                    "path": planner.relative_to(tmp_path).as_posix(),
                    "sha256": _sha(planner),
                },
                "source_epoch_identity_sha256": (
                    context["bindings"].source_epoch_identity_sha256
                ),
                "qiskit_compilation": False,
            },
            "safeguard_receipt_sha256",
        ),
    )
    return (
        SafeguardQualificationReceipt.load(receipt, repository_root=tmp_path),
        action_registry,
    )


def test_action_registry_and_scientific_state_are_digest_bound(tmp_path: Path) -> None:
    _registry, seed_identity, reference_identity = _prospective_rows()
    source = _source_epoch()
    seed = _seed_receipt(tmp_path, source_epoch=source)
    reference = _reference_receipt(tmp_path, seed, source_epoch=source)
    source_receipt = _source_receipt(tmp_path, source)
    policy = _policy(str(reference_identity.drive["drive_id"]))
    bindings = build_scientific_bindings(
        reference=reference_identity,
        seed=seed_identity,
        seed_receipt=seed,
        reference_receipt=reference,
        source_epoch=source,
        source_epoch_receipt=source_receipt,
        policy=policy,
    )
    action = plan_prospective_cell(bindings, policy).next_actions[0]
    request = resolve_scientific_action(bindings, action)
    registry_path = tmp_path / "state/actions/action.json"
    entry = write_action_registry_entry(
        registry_path,
        request,
        source_receipts=(
            VerifiedFile(source_receipt.receipt_path, source_receipt.receipt_sha256),
            VerifiedFile(seed.receipt_path, seed.receipt_sha256),
            VerifiedFile(reference.receipt_path, reference.receipt_sha256),
        ),
        repository_root=tmp_path,
        bindings=bindings,
        expected_seed=seed_identity,
        expected_reference=reference_identity,
        expected_source_epoch=source,
    )
    assert entry["action_id"] == action.action_id
    action_binding = VerifiedFile(
        registry_path.relative_to(tmp_path).as_posix(), _sha(registry_path)
    )
    score = _score_receipt(
        tmp_path,
        bindings=bindings,
        reference_receipt=reference,
        policy=policy,
        action=action,
        action_registry=action_binding,
    )
    state_path = tmp_path / "state/scientific_state.json"
    state = write_scientific_state(
        state_path,
        bindings=bindings,
        action_entries=(action_binding,),
        score_receipts=(score,),
        repository_root=tmp_path,
        expected_seed=seed_identity,
        expected_reference=reference_identity,
        expected_source_epoch=source,
    )
    assert state["bindings"]["seed_artifact_sha256"] == seed.artifact.sha256
    assert state["score_receipts"][0]["sha256"] == score.receipt_sha256
    assert state["run_lifecycle_tips"][0]["sha256"] == score.run_receipt.sha256
    assert state["safeguard_receipts"] == []
    assert state["qiskit_compilation"] is False
    loaded = load_scientific_state(
        state_path,
        repository_root=tmp_path,
        expected_seed=seed_identity,
        expected_reference=reference_identity,
        expected_source_epoch=source,
    )
    assert loaded.bindings == bindings
    assert loaded.run_lifecycle_tips == (score.run_receipt,)
    assert loaded.score_receipts == (
        VerifiedFile(score.receipt_path, score.receipt_sha256),
    )
    assert loaded.evidence.runs == (
        score.to_planner_evidence(tmp_path, bindings),
    )


def test_scientific_state_round_trip_includes_canonical_safeguard_projection(
    tmp_path: Path,
) -> None:
    context = _scientific_context(tmp_path)
    action = plan_prospective_cell(
        context["bindings"], context["policy"]
    ).next_actions[0]
    _request, action_registry = _action_registry_binding(
        tmp_path, context, action, token="state_subject"
    )
    score = _score_receipt(
        tmp_path,
        bindings=context["bindings"],
        reference_receipt=context["reference"],
        policy=context["policy"],
        action=action,
        action_registry=action_registry,
    )
    safeguard, safeguard_registry = _typed_safeguard_receipt(
        tmp_path, context=context, score=score, token="state_terminal"
    )
    state_path = tmp_path / "state/terminal_scientific_state.json"
    payload = write_scientific_state(
        state_path,
        bindings=context["bindings"],
        action_entries=(safeguard_registry, action_registry),
        run_receipts=(score.run_lifecycle(tmp_path),),
        score_receipts=(score,),
        safeguard_receipts=(safeguard,),
        repository_root=tmp_path,
        expected_seed=context["seed_identity"],
        expected_reference=context["reference_identity"],
        expected_source_epoch=context["source"],
    )
    loaded = load_scientific_state(
        state_path,
        repository_root=tmp_path,
        expected_seed=context["seed_identity"],
        expected_reference=context["reference_identity"],
        expected_source_epoch=context["source"],
    )
    assert loaded.evidence == durable_receipts_to_evidence(
        (score.run_lifecycle(tmp_path),),
        score_receipts=(score,),
        safeguard_receipts=(safeguard,),
        repository_root=tmp_path,
        bindings=context["bindings"],
    )
    assert payload["planner_evidence"] == {
        "runs": [
            {
                "action_id": action.action_id,
                "activation_cut": action.activation_cut,
                "arm": action.arm,
                "attempt": action.attempt,
                "bindings_sha256": context["bindings"].digest,
                "failure_kind": None,
                "failure_reason": "",
                "failure_receipt_sha256": None,
                "horizon": action.horizon,
                "metrics": score.accuracy_metrics(tmp_path).to_json_dict(),
                "observed_peak_gib": None,
                "planner_sequence": action.planner_sequence,
                "requested_memory_gib": action.requested_memory_gib,
                "resume_from_action_id": None,
                "resume_run_receipt_sha256": None,
                "resume_run_sha256": None,
                "run_receipt_sha256": score.run_receipt.sha256,
                "run_sha256": score.run.sha256,
                "score_receipt_sha256": score.receipt_sha256,
                "score_sha256": score.metrics.sha256,
                "score_state": "completed",
                "state": "completed",
            }
        ],
        "safeguards": [
            {
                "bindings_sha256": context["bindings"].digest,
                "planner_sequence": safeguard.planner_sequence,
                "reason": SAFEGUARD_LADDER_UNRESOLVED,
                "receipt_sha256": safeguard.receipt_sha256,
                "safeguard_id": safeguard.safeguard_action_id,
                "subject_action_id": action.action_id,
            }
        ],
    }


def test_scientific_state_loader_rejects_resealed_projection_and_qiskit_tamper(
    tmp_path: Path,
) -> None:
    context = _scientific_context(tmp_path)
    action = plan_prospective_cell(
        context["bindings"], context["policy"]
    ).next_actions[0]
    _request, action_registry = _action_registry_binding(
        tmp_path, context, action, token="state_tamper"
    )
    score = _score_receipt(
        tmp_path,
        bindings=context["bindings"],
        reference_receipt=context["reference"],
        policy=context["policy"],
        action=action,
        action_registry=action_registry,
    )
    state_path = tmp_path / "state/tamper_scientific_state.json"
    write_scientific_state(
        state_path,
        bindings=context["bindings"],
        action_entries=(action_registry,),
        score_receipts=(score,),
        repository_root=tmp_path,
        expected_seed=context["seed_identity"],
        expected_reference=context["reference_identity"],
        expected_source_epoch=context["source"],
    )
    original = json.loads(state_path.read_text())
    projection_tamper = json.loads(json.dumps(original))
    projection_tamper["planner_evidence"]["runs"][0]["planner_sequence"] += 1
    _write(
        state_path,
        _sealed(projection_tamper, "scientific_state_sha256"),
    )
    with pytest.raises(ProspectiveExecutionError, match="planner evidence projection"):
        load_scientific_state(
            state_path,
            repository_root=tmp_path,
            expected_seed=context["seed_identity"],
            expected_reference=context["reference_identity"],
            expected_source_epoch=context["source"],
        )

    qiskit_tamper = dict(original)
    qiskit_tamper["qiskit_compilation"] = True
    _write(state_path, _sealed(qiskit_tamper, "scientific_state_sha256"))
    with pytest.raises(ProspectiveExecutionError, match="Qiskit"):
        load_scientific_state(
            state_path,
            repository_root=tmp_path,
            expected_seed=context["seed_identity"],
            expected_reference=context["reference_identity"],
            expected_source_epoch=context["source"],
        )

    orphan_tamper = json.loads(json.dumps(original))
    orphan_tamper["action_entries"] = []
    _write(state_path, _sealed(orphan_tamper, "scientific_state_sha256"))
    with pytest.raises(ProspectiveExecutionError, match="no matching action-registry"):
        load_scientific_state(
            state_path,
            repository_root=tmp_path,
            expected_seed=context["seed_identity"],
            expected_reference=context["reference_identity"],
            expected_source_epoch=context["source"],
        )

    duplicate_tamper = json.loads(json.dumps(original))
    duplicate_tamper["action_entries"].append(
        duplicate_tamper["action_entries"][0]
    )
    _write(state_path, _sealed(duplicate_tamper, "scientific_state_sha256"))
    with pytest.raises(ProspectiveExecutionError, match="repeats.*action_entries"):
        load_scientific_state(
            state_path,
            repository_root=tmp_path,
            expected_seed=context["seed_identity"],
            expected_reference=context["reference_identity"],
            expected_source_epoch=context["source"],
        )

    cross_cell_tamper = json.loads(json.dumps(original))
    other_reference = next(
        row
        for row in build_identity_registry().references
        if row.lane_id == context["bindings"].lane_id
        and row.cell_id != context["bindings"].cell_id
    )
    cross_cell_tamper["bindings"]["cell_id"] = other_reference.cell_id
    unsigned_bindings = dict(cross_cell_tamper["bindings"])
    unsigned_bindings.pop("bindings_sha256")
    cross_cell_tamper["bindings"]["bindings_sha256"] = _canonical_sha(
        unsigned_bindings
    )
    _write(state_path, _sealed(cross_cell_tamper, "scientific_state_sha256"))
    with pytest.raises(ProspectiveExecutionError, match="matrix-bound bindings"):
        load_scientific_state(
            state_path,
            repository_root=tmp_path,
            expected_seed=context["seed_identity"],
            expected_reference=context["reference_identity"],
            expected_source_epoch=context["source"],
        )


def test_scientific_state_rejects_duplicate_and_orphan_action_entries(
    tmp_path: Path,
) -> None:
    context = _scientific_context(tmp_path)
    actions = plan_prospective_cell(
        context["bindings"], context["policy"]
    ).next_actions
    action = actions[0]
    _request, action_registry = _action_registry_binding(
        tmp_path, context, action, token="state_duplicate"
    )
    score = _score_receipt(
        tmp_path,
        bindings=context["bindings"],
        reference_receipt=context["reference"],
        policy=context["policy"],
        action=action,
        action_registry=action_registry,
    )
    with pytest.raises(ProspectiveExecutionError, match="repeats an action entry"):
        write_scientific_state(
            tmp_path / "state/duplicate.json",
            bindings=context["bindings"],
            action_entries=(action_registry, action_registry),
            score_receipts=(score,),
            repository_root=tmp_path,
            expected_seed=context["seed_identity"],
            expected_reference=context["reference_identity"],
            expected_source_epoch=context["source"],
        )

    orphan_id = prospective_safeguard_id(
        context["bindings"],
        reason=SAFEGUARD_LADDER_UNRESOLVED,
        subject_action_id=action.action_id,
    )
    orphan_action = ProspectiveNextAction(
        kind=KIND_RECORD_SAFEGUARD,
        action_id=orphan_id,
        execution_id=orphan_id,
        arm=None,
        horizon=None,
        activation_cut=None,
        ladder_index=None,
        attempt=0,
        planner_sequence=action.planner_sequence + 1,
        requested_memory_gib=None,
        reason="orphan safeguard registry fixture",
        inputs={
            **context["bindings"].to_json_dict(),
            "safeguard_reason": SAFEGUARD_LADDER_UNRESOLVED,
            "subject_action_id": action.action_id,
            "retain_all_numerical_evidence": True,
        },
        exact_reference_blind=False,
    )
    _other_request, orphan_registry = _action_registry_binding(
        tmp_path,
        context,
        orphan_action,
        token="state_orphan",
    )
    with pytest.raises(ProspectiveExecutionError, match="orphan action entry"):
        write_scientific_state(
            tmp_path / "state/orphan.json",
            bindings=context["bindings"],
            action_entries=(action_registry, orphan_registry),
            score_receipts=(score,),
            repository_root=tmp_path,
            expected_seed=context["seed_identity"],
            expected_reference=context["reference_identity"],
            expected_source_epoch=context["source"],
        )


def test_action_registry_rejects_request_and_source_digest_mismatches(
    tmp_path: Path,
) -> None:
    context = _scientific_context(tmp_path)
    action = plan_prospective_cell(
        context["bindings"], context["policy"]
    ).next_actions[0]
    request = resolve_scientific_action(context["bindings"], action)
    receipts = (
        VerifiedFile(
            context["source_receipt"].receipt_path,
            context["source_receipt"].receipt_sha256,
        ),
        VerifiedFile(context["seed"].receipt_path, context["seed"].receipt_sha256),
        VerifiedFile(
            context["reference"].receipt_path,
            context["reference"].receipt_sha256,
        ),
    )
    action_path = tmp_path / "state/action.json"
    write_action_registry_entry(
        action_path,
        request,
        source_receipts=receipts,
        repository_root=tmp_path,
        bindings=context["bindings"],
        expected_seed=context["seed_identity"],
        expected_reference=context["reference_identity"],
        expected_source_epoch=context["source"],
    )
    payload = json.loads(action_path.read_text())
    payload["execution_request_sha256"] = "e" * 64
    _write(action_path, _sealed(payload, "entry_sha256"))
    with pytest.raises(ProspectiveExecutionError, match="scientific bindings"):
        load_action_registry_entry(
            VerifiedFile(action_path.relative_to(tmp_path).as_posix(), _sha(action_path)),
            repository_root=tmp_path,
            bindings=context["bindings"],
            expected_seed=context["seed_identity"],
            expected_reference=context["reference_identity"],
            expected_source_epoch=context["source"],
        )

    source_path = tmp_path / context["source_receipt"].receipt_path
    source_payload = json.loads(source_path.read_text())
    source_payload["source_tree_sha256"] = "f" * 64
    _write(source_path, _sealed(source_payload, "source_epoch_receipt_sha256"))
    with pytest.raises(ProspectiveExecutionError, match="source tree"):
        write_action_registry_entry(
            tmp_path / "state/source-mismatch-action.json",
            request,
            source_receipts=(
                VerifiedFile(
                    context["source_receipt"].receipt_path, _sha(source_path)
                ),
                receipts[1],
                receipts[2],
            ),
            repository_root=tmp_path,
            bindings=context["bindings"],
            expected_seed=context["seed_identity"],
            expected_reference=context["reference_identity"],
            expected_source_epoch=context["source"],
        )


def test_score_receipt_rejects_run_and_metrics_digest_mismatch(tmp_path: Path) -> None:
    context = _scientific_context(tmp_path)
    action = plan_prospective_cell(
        context["bindings"], context["policy"]
    ).next_actions[0]
    _request, action_registry = _action_registry_binding(
        tmp_path, context, action, token="score_tamper"
    )
    score = _score_receipt(
        tmp_path,
        bindings=context["bindings"],
        reference_receipt=context["reference"],
        policy=context["policy"],
        action=action,
        action_registry=action_registry,
    )
    metrics_path = tmp_path / score.metrics.path
    metrics = json.loads(metrics_path.read_text())
    metrics["run_sha256"] = "e" * 64
    _write(metrics_path, _sealed(metrics, "score_metrics_sha256"))
    receipt_path = tmp_path / score.receipt_path
    receipt = json.loads(receipt_path.read_text())
    receipt["metrics"]["sha256"] = _sha(metrics_path)
    _write(receipt_path, _sealed(receipt, "score_receipt_sha256"))
    with pytest.raises(ProspectiveExecutionError, match="score metrics identity"):
        ScoreQualificationReceipt.load(receipt_path, repository_root=tmp_path)


def test_score_receipt_requires_typed_run_and_its_exact_action_registry(
    tmp_path: Path,
) -> None:
    context = _scientific_context(tmp_path)
    action = plan_prospective_cell(
        context["bindings"], context["policy"]
    ).next_actions[0]
    _request, action_registry = _action_registry_binding(
        tmp_path, context, action, token="score_authority"
    )
    score = _score_receipt(
        tmp_path,
        bindings=context["bindings"],
        reference_receipt=context["reference"],
        policy=context["policy"],
        action=action,
        action_registry=action_registry,
    )
    receipt_path = tmp_path / score.receipt_path
    original = json.loads(receipt_path.read_text())

    naked = dict(original)
    naked.pop("run_receipt")
    naked["run_receipt_sha256"] = score.run_receipt_sha256
    naked["run"] = score.run.to_json_dict()
    _write(receipt_path, _sealed(naked, "score_receipt_sha256"))
    with pytest.raises(ProspectiveExecutionError, match="no run_receipt binding"):
        ScoreQualificationReceipt.load(receipt_path, repository_root=tmp_path)

    _write(receipt_path, original)
    _duplicate_request, duplicate_registry = _action_registry_binding(
        tmp_path, context, action, token="score_authority_duplicate"
    )
    substituted = dict(original)
    substituted["action_registry"] = duplicate_registry.to_json_dict()
    _write(receipt_path, _sealed(substituted, "score_receipt_sha256"))
    with pytest.raises(ProspectiveExecutionError, match="action registry diverges"):
        ScoreQualificationReceipt.load(receipt_path, repository_root=tmp_path)


def test_score_receipt_rejects_tampered_run_behind_typed_lifecycle(
    tmp_path: Path,
) -> None:
    context = _scientific_context(tmp_path)
    action = plan_prospective_cell(
        context["bindings"], context["policy"]
    ).next_actions[0]
    _request, action_registry = _action_registry_binding(
        tmp_path, context, action, token="score_run_tamper"
    )
    score = _score_receipt(
        tmp_path,
        bindings=context["bindings"],
        reference_receipt=context["reference"],
        policy=context["policy"],
        action=action,
        action_registry=action_registry,
    )
    (tmp_path / score.run.path).write_text("tampered\n")
    with pytest.raises(ProspectiveExecutionError, match="artifact digest mismatch"):
        ScoreQualificationReceipt.load(
            tmp_path / score.receipt_path, repository_root=tmp_path
        )


def test_typed_safeguard_joins_canonical_action_subject_and_run(
    tmp_path: Path,
) -> None:
    context = _scientific_context(tmp_path)
    action = plan_prospective_cell(
        context["bindings"], context["policy"]
    ).next_actions[0]
    _request, action_registry = _action_registry_binding(
        tmp_path, context, action, token="safeguard_subject"
    )
    score = _score_receipt(
        tmp_path,
        bindings=context["bindings"],
        reference_receipt=context["reference"],
        policy=context["policy"],
        action=action,
        action_registry=action_registry,
    )
    safeguard, safeguard_registry = _typed_safeguard_receipt(
        tmp_path, context=context, score=score, token="canonical"
    )
    assert safeguard.action_registry == safeguard_registry
    assert safeguard.subject_action_id == score.scientific_action_id
    assert safeguard.subject_run_receipts == (score.run_receipt,)
    evidence = safeguard.to_planner_evidence(
        tmp_path, context["bindings"]
    )
    assert evidence.safeguard_id == safeguard.safeguard_action_id
    assert evidence.reason == SAFEGUARD_LADDER_UNRESOLVED
    assert evidence.subject_action_id == score.scientific_action_id
    replay = durable_receipts_to_evidence(
        (score.run_lifecycle(tmp_path),),
        score_receipts=(score,),
        safeguard_receipts=(safeguard,),
        repository_root=tmp_path,
        bindings=context["bindings"],
    )
    assert replay.safeguards == (evidence,)
    with pytest.raises(ProspectiveExecutionError, match="repeat one canonical"):
        durable_receipts_to_evidence(
            (score.run_lifecycle(tmp_path),),
            score_receipts=(score,),
            safeguard_receipts=(safeguard, safeguard),
            repository_root=tmp_path,
            bindings=context["bindings"],
        )
    with pytest.raises(ProspectiveExecutionError, match="no subject run evidence"):
        durable_receipts_to_evidence(
            (),
            score_receipts=(),
            safeguard_receipts=(safeguard,),
            repository_root=tmp_path,
            bindings=context["bindings"],
        )


def test_scientific_state_rejects_score_from_another_exact_reference(
    tmp_path: Path,
) -> None:
    _registry, seed_identity, reference_identity = _prospective_rows()
    source = _source_epoch()
    seed = _seed_receipt(tmp_path, source_epoch=source)
    reference = _reference_receipt(tmp_path, seed, source_epoch=source)
    source_receipt = _source_receipt(tmp_path, source)
    policy = _policy(str(reference_identity.drive["drive_id"]))
    bindings = build_scientific_bindings(
        reference=reference_identity,
        seed=seed_identity,
        seed_receipt=seed,
        reference_receipt=reference,
        source_epoch=source,
        source_epoch_receipt=source_receipt,
        policy=policy,
    )
    action = plan_prospective_cell(bindings, policy).next_actions[0]
    request = resolve_scientific_action(bindings, action)
    action_path = tmp_path / "evidence/action.json"
    write_action_registry_entry(
        action_path,
        request,
        source_receipts=(
            VerifiedFile(source_receipt.receipt_path, source_receipt.receipt_sha256),
            VerifiedFile(seed.receipt_path, seed.receipt_sha256),
            VerifiedFile(reference.receipt_path, reference.receipt_sha256),
        ),
        repository_root=tmp_path,
        bindings=bindings,
        expected_seed=seed_identity,
        expected_reference=reference_identity,
        expected_source_epoch=source,
    )
    action_binding = VerifiedFile(
        action_path.relative_to(tmp_path).as_posix(), _sha(action_path)
    )
    score = _score_receipt(
        tmp_path,
        bindings=bindings,
        reference_receipt=reference,
        policy=policy,
        action=action,
        action_registry=action_binding,
    )
    score_path = tmp_path / score.receipt_path
    score_payload = json.loads(score_path.read_text())
    metrics_path = tmp_path / score.metrics.path
    metrics_payload = json.loads(metrics_path.read_text())
    metrics_payload["exact_data_sha256"] = "f" * 64
    _write(metrics_path, _sealed(metrics_payload, "score_metrics_sha256"))
    score_payload["exact_data_sha256"] = "f" * 64
    score_payload["metrics"]["sha256"] = _sha(metrics_path)
    _write(score_path, _sealed(score_payload, "score_receipt_sha256"))
    forged = ScoreQualificationReceipt.load(score_path, repository_root=tmp_path)
    with pytest.raises(ProspectiveExecutionError, match="scientific bindings"):
        write_scientific_state(
            tmp_path / "state.json",
            bindings=bindings,
            action_entries=(
                VerifiedFile(
                    action_path.relative_to(tmp_path).as_posix(), _sha(action_path)
                ),
            ),
            score_receipts=(forged,),
            repository_root=tmp_path,
            expected_seed=seed_identity,
            expected_reference=reference_identity,
            expected_source_epoch=source,
        )


def test_terminal_safeguard_receipt_is_score_and_planner_bound(
    tmp_path: Path,
) -> None:
    context = _scientific_context(tmp_path)
    action = plan_prospective_cell(
        context["bindings"], context["policy"]
    ).next_actions[0]
    _request, action_registry = _action_registry_binding(
        tmp_path, context, action, token="tamper_subject"
    )
    score = _score_receipt(
        tmp_path,
        bindings=context["bindings"],
        reference_receipt=context["reference"],
        policy=context["policy"],
        action=action,
        action_registry=action_registry,
    )
    loaded, _safeguard_registry = _typed_safeguard_receipt(
        tmp_path, context=context, score=score, token="tamper"
    )
    assert loaded.outcome == "threshold_ladder_exhausted"
    (tmp_path / loaded.planner_receipt.path).write_text("tampered\n")
    with pytest.raises(ProspectiveExecutionError, match="artifact digest mismatch"):
        SafeguardQualificationReceipt.load(
            tmp_path / loaded.receipt_path, repository_root=tmp_path
        )


def _auditable_execution(
    tmp_path: Path,
    *,
    token: str,
    scope: str = "build_package",
) -> dict:
    context = _scientific_context(tmp_path)
    action = plan_prospective_cell(
        context["bindings"], context["policy"]
    ).next_actions[0]
    _run_request, action_registry = _action_registry_binding(
        tmp_path, context, action, token=f"{token}_run"
    )
    score = _score_receipt(
        tmp_path,
        bindings=context["bindings"],
        reference_receipt=context["reference"],
        policy=context["policy"],
        action=action,
        action_registry=action_registry,
    )
    state_path = tmp_path / f"state/{token}_scientific_state.json"
    write_scientific_state(
        state_path,
        bindings=context["bindings"],
        action_entries=(action_registry,),
        run_receipts=(score.run_lifecycle(tmp_path),),
        score_receipts=(score,),
        warnings=("warning carried unchanged",),
        repository_root=tmp_path,
        expected_seed=context["seed_identity"],
        expected_reference=context["reference_identity"],
        expected_source_epoch=context["source"],
    )
    state = load_scientific_state(
        state_path,
        repository_root=tmp_path,
        expected_seed=context["seed_identity"],
        expected_reference=context["reference_identity"],
        expected_source_epoch=context["source"],
    )
    matrix_path = (
        tmp_path
        / "pipelines/time_dynamics/campaigns/"
        "paper_ii_parallel_benchmark_matrix_v1.json"
    )
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_bytes(DEFAULT_MATRIX_PATH.read_bytes())
    contract = build_planned_report_contract(
        lane_id=context["bindings"].lane_id,
        repository_root=tmp_path,
        matrix_path=matrix_path,
    )
    report_path = tmp_path / f"state/{token}_planned_report_contract.json"
    _write(report_path, contract)
    report = VerifiedFile(
        report_path.relative_to(tmp_path).as_posix(), _sha(report_path)
    )
    campaign_registry = _terminal_campaign_registry(
        tmp_path,
        bindings=context["bindings"],
        scientific_state=state,
        planned_report_contract=report,
        matrix_path=matrix_path,
    )
    request_factory = (
        build_package_execution_request
        if scope == "build_package"
        else update_report_slot_execution_request
    )
    request = request_factory(
        context["bindings"],
        scientific_state=state,
        planned_report_contract=report,
        campaign_registry=campaign_registry,
    )
    audit_path = tmp_path / f"evidence/{token}_{scope}_audit.json"
    audit = write_scientific_pre_submit_audit(
        audit_path,
        request=request,
        scientific_state=VerifiedFile(
            state.receipt_path, state.receipt_sha256
        ),
        planned_report_contract=report,
        source_epoch_receipts=(
            VerifiedFile(
                context["source_receipt"].receipt_path,
                context["source_receipt"].receipt_sha256,
            ),
        ),
        seed_receipts=(
            VerifiedFile(
                context["seed"].receipt_path,
                context["seed"].receipt_sha256,
            ),
        ),
        exact_reference_receipts=(
            VerifiedFile(
                context["reference"].receipt_path,
                context["reference"].receipt_sha256,
            ),
        ),
        campaign_registry=campaign_registry,
        repository_root=tmp_path,
        matrix_path=matrix_path,
        warnings=("warning carried unchanged",),
    )
    return {
        **context,
        "action": action,
        "action_registry": action_registry,
        "score": score,
        "state": state,
        "state_path": state_path,
        "matrix_path": matrix_path,
        "contract": contract,
        "report": report,
        "report_path": report_path,
        "campaign_registry": campaign_registry,
        "request": request,
        "audit": audit,
        "audit_path": audit_path,
    }


def test_canonical_scientific_pre_submit_audit_gates_package_and_report(
    tmp_path: Path,
) -> None:
    package = _auditable_execution(tmp_path / "package", token="canonical")
    audit = package["audit"]
    authority = package["campaign_registry"].lane_scientific_authority(
        package["bindings"].lane_id
    )
    assert audit.audit_scope == "build_package"
    assert audit.action_id == package["request"].action_id
    assert audit.scientific_state == VerifiedFile(
        package["state"].receipt_path, package["state"].receipt_sha256
    )
    assert audit.warnings == ("warning carried unchanged",)
    assert audit.lane_scientific_authority == authority
    assert package["request"].inputs["lane_scientific_authority"] == (
        authority.to_json_dict()
    )
    materialization = package_request(
        package["request"],
        scientific_pre_submit_audit=audit,
        campaign_registry=package["campaign_registry"],
        repository_root=tmp_path / "package",
    )
    assert materialization["status"] == "package_materialization_required"
    assert materialization["lane_scientific_authority_sha256"] == (
        authority.authority_sha256
    )
    assert materialization["completion_receipt"] is None

    report = _auditable_execution(
        tmp_path / "report", token="canonical", scope="update_report_slot"
    )
    update = report_slot_request(
        report["request"],
        scientific_pre_submit_audit=report["audit"],
        campaign_registry=report["campaign_registry"],
        repository_root=tmp_path / "report",
    )
    assert update["status"] == "evidence_binding_required"
    assert update["completion_receipt"] is None
    assert binding_registry_payload()["production_submission_enabled"] is False


def test_terminal_requests_reject_off_registry_copied_evidence(
    tmp_path: Path,
) -> None:
    fixture = _auditable_execution(tmp_path, token="off_registry")
    copied_state_path = tmp_path / "state/copied_scientific_state.json"
    copied_state_path.write_bytes(fixture["state_path"].read_bytes())
    copied_state = load_scientific_state(
        copied_state_path,
        repository_root=tmp_path,
        expected_seed=fixture["seed_identity"],
        expected_reference=fixture["reference_identity"],
        expected_source_epoch=fixture["source"],
    )
    with pytest.raises(ProspectiveExecutionError, match="authority"):
        build_package_execution_request(
            fixture["bindings"],
            scientific_state=copied_state,
            planned_report_contract=fixture["report"],
            campaign_registry=fixture["campaign_registry"],
        )

    copied_report_path = tmp_path / "state/copied_planned_report_contract.json"
    copied_report_path.write_bytes(fixture["report_path"].read_bytes())
    copied_report = VerifiedFile(
        copied_report_path.relative_to(tmp_path).as_posix(),
        _sha(copied_report_path),
    )
    with pytest.raises(ProspectiveExecutionError, match="authority"):
        build_package_execution_request(
            fixture["bindings"],
            scientific_state=fixture["state"],
            planned_report_contract=copied_report,
            campaign_registry=fixture["campaign_registry"],
        )


def test_terminal_request_checks_every_registry_report_and_matrix_digest(
    tmp_path: Path,
) -> None:
    fixture = _auditable_execution(tmp_path, token="contract_digests")
    lane_id = fixture["bindings"].lane_id
    mutations = (
        ("planned_report_contracts", "file_sha256"),
        ("planned_report_contracts", "report_contract_sha256"),
        ("planned_report_contracts", "tuple_structure_sha256"),
        ("matrix", "raw_sha256"),
        ("matrix", "payload_sha256"),
        ("matrix", "expanded_semantic_sha256"),
        ("matrix", "accuracy_policy_spec_sha256"),
    )
    for section, field in mutations:
        contracts = json.loads(
            json.dumps(fixture["campaign_registry"].scientific_contracts)
        )
        target = contracts[section]
        if section == "planned_report_contracts":
            target = target[lane_id]
        target[field] = "f" * 64
        drifted = replace(
            fixture["campaign_registry"], scientific_contracts=contracts
        )
        with pytest.raises(ProspectiveExecutionError, match="authority"):
            build_package_execution_request(
                fixture["bindings"],
                scientific_state=fixture["state"],
                planned_report_contract=fixture["report"],
                campaign_registry=drifted,
            )


def test_terminal_pipeline_requires_current_campaign_registry(
    tmp_path: Path,
) -> None:
    fixture = _auditable_execution(tmp_path, token="registry_required")
    with pytest.raises(ProspectiveExecutionError, match="current campaign registry"):
        build_package_execution_request(
            fixture["bindings"],
            scientific_state=fixture["state"],
            planned_report_contract=fixture["report"],
            campaign_registry=None,  # type: ignore[arg-type]
        )
    with pytest.raises(ProspectiveExecutionError, match="current campaign registry"):
        write_scientific_pre_submit_audit(
            tmp_path / "evidence/no_registry_audit.json",
            request=fixture["request"],
            scientific_state=VerifiedFile(
                fixture["state"].receipt_path,
                fixture["state"].receipt_sha256,
            ),
            planned_report_contract=fixture["report"],
            source_epoch_receipts=(
                VerifiedFile(
                    fixture["source_receipt"].receipt_path,
                    fixture["source_receipt"].receipt_sha256,
                ),
            ),
            seed_receipts=(
                VerifiedFile(
                    fixture["seed"].receipt_path,
                    fixture["seed"].receipt_sha256,
                ),
            ),
            exact_reference_receipts=(
                VerifiedFile(
                    fixture["reference"].receipt_path,
                    fixture["reference"].receipt_sha256,
                ),
            ),
            campaign_registry=None,  # type: ignore[arg-type]
            repository_root=tmp_path,
            matrix_path=fixture["matrix_path"],
            warnings=("warning carried unchanged",),
        )
    with pytest.raises(ProspectiveExecutionError, match="current campaign registry"):
        package_request(
            fixture["request"],
            scientific_pre_submit_audit=fixture["audit"],
            campaign_registry=None,  # type: ignore[arg-type]
            repository_root=tmp_path,
        )


@pytest.mark.parametrize("drift", ("planner", "cross_lane", "matrix"))
def test_terminal_handlers_reject_current_registry_scientific_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    fixture = _auditable_execution(tmp_path, token=f"registry_{drift}")
    registry = fixture["campaign_registry"]
    lane = registry.lane(fixture["bindings"].lane_id)
    if drift == "planner":
        drifted = replace(
            registry,
            lanes=(
                replace(
                    lane,
                    planner_authority=(
                        "pipelines.time_dynamics.other:plan_prospective_cell"
                    ),
                ),
            ),
        )
    elif drift == "cross_lane":
        drifted = replace(
            registry,
            lanes=(replace(lane, lane_id="different_lane"),),
        )
    else:
        contracts = json.loads(json.dumps(registry.scientific_contracts))
        contracts["matrix"]["raw_sha256"] = "f" * 64
        drifted = replace(registry, scientific_contracts=contracts)

    # Loading authenticates only the immutable projection captured by the audit.
    assert fixture["audit"].reload(tmp_path) == fixture["audit"]
    with pytest.raises(ProspectiveExecutionError, match="authority"):
        package_request(
            fixture["request"],
            scientific_pre_submit_audit=fixture["audit"],
            campaign_registry=drifted,
            repository_root=tmp_path,
        )


def test_terminal_handler_authority_is_stable_across_control_plane_changes(
    tmp_path: Path,
) -> None:
    fixture = _auditable_execution(tmp_path, token="control_plane")
    registry = fixture["campaign_registry"]
    lane = registry.lane(fixture["bindings"].lane_id)
    control_changed = replace(
        registry,
        supervisor_state_root=tmp_path / "output/new_supervisor_state",
        lanes=(
            replace(
                lane,
                monitor_enabled=True,
                submission_enabled=False,
                blocked_by=(),
                metadata={"control_revision": 2},
            ),
        ),
    )
    before = registry.lane_scientific_authority(lane.lane_id)
    after = control_changed.lane_scientific_authority(lane.lane_id)
    assert after == before
    assert after.authority_sha256 == before.authority_sha256
    materialization = package_request(
        fixture["request"],
        scientific_pre_submit_audit=fixture["audit"],
        campaign_registry=control_changed,
        repository_root=tmp_path,
    )
    assert materialization["status"] == "package_materialization_required"


def test_pre_submit_audit_rejects_hand_authored_resealed_and_wrong_scope(
    tmp_path: Path,
) -> None:
    fixture = _auditable_execution(tmp_path, token="tamper")
    path = fixture["audit_path"]
    canonical = json.loads(path.read_text())

    hand_authored = {
        "schema": SCIENTIFIC_PRE_SUBMIT_AUDIT_V1,
        "status": "passed",
        "lane_id": LANE_HH,
        "errors": [],
        "warnings": [],
    }
    hand_authored["audit_sha256"] = _canonical_sha(hand_authored)
    _write(path, hand_authored)
    with pytest.raises(ProspectiveExecutionError, match="canonical audit contract"):
        ScientificPreSubmitAudit.load(path, repository_root=tmp_path)

    resealed = json.loads(json.dumps(canonical))
    resealed["executor_registry_sha256"] = "e" * 64
    _write(path, _sealed(resealed, "audit_sha256"))
    with pytest.raises(ProspectiveExecutionError, match="executor registry"):
        ScientificPreSubmitAudit.load(path, repository_root=tmp_path)

    warning_erased = json.loads(json.dumps(canonical))
    warning_erased["warnings"] = []
    _write(path, _sealed(warning_erased, "audit_sha256"))
    with pytest.raises(ProspectiveExecutionError, match="warning|canonical evidence"):
        ScientificPreSubmitAudit.load(path, repository_root=tmp_path)

    authority_tampered = json.loads(json.dumps(canonical))
    embedded = authority_tampered["lane_scientific_authority"]
    embedded["planner_authority"] = (
        "pipelines.time_dynamics.other:plan_prospective_cell"
    )
    embedded["authority_sha256"] = _canonical_sha(
        {
            key: value
            for key, value in embedded.items()
            if key != "authority_sha256"
        }
    )
    _write(path, _sealed(authority_tampered, "audit_sha256"))
    with pytest.raises(ProspectiveExecutionError, match="authority projection"):
        ScientificPreSubmitAudit.load(path, repository_root=tmp_path)

    _write(path, canonical)
    with pytest.raises(ProspectiveExecutionError, match="warnings.*scientific state"):
        write_scientific_pre_submit_audit(
            tmp_path / "evidence/mismatched_warnings_audit.json",
            request=fixture["request"],
            scientific_state=VerifiedFile(
                fixture["state"].receipt_path,
                fixture["state"].receipt_sha256,
            ),
            planned_report_contract=fixture["report"],
            source_epoch_receipts=(
                VerifiedFile(
                    fixture["source_receipt"].receipt_path,
                    fixture["source_receipt"].receipt_sha256,
                ),
            ),
            seed_receipts=(
                VerifiedFile(
                    fixture["seed"].receipt_path,
                    fixture["seed"].receipt_sha256,
                ),
            ),
            exact_reference_receipts=(
                VerifiedFile(
                    fixture["reference"].receipt_path,
                    fixture["reference"].receipt_sha256,
                ),
            ),
            campaign_registry=fixture["campaign_registry"],
            repository_root=tmp_path,
            matrix_path=fixture["matrix_path"],
            warnings=(),
        )

    _write(path, canonical)
    report_request = update_report_slot_execution_request(
        fixture["bindings"],
        scientific_state=fixture["state"],
        planned_report_contract=fixture["report"],
        campaign_registry=fixture["campaign_registry"],
    )
    with pytest.raises(ProspectiveExecutionError, match="scope|request"):
        report_slot_request(
            report_request,
            scientific_pre_submit_audit=fixture["audit"],
            campaign_registry=fixture["campaign_registry"],
            repository_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("artifact", "error"),
    (
        ("matrix_path", "matrix"),
        ("report_path", "report|artifact digest"),
        ("source_receipt", "source|artifact digest"),
        ("score", "score|artifact digest"),
        ("action_registry", "action|artifact digest"),
    ),
)
def test_pre_submit_audit_reloads_every_scientific_artifact(
    tmp_path: Path,
    artifact: str,
    error: str,
) -> None:
    fixture = _auditable_execution(tmp_path, token=f"stale_{artifact}")
    if artifact == "source_receipt":
        target = tmp_path / fixture[artifact].receipt_path
    elif artifact == "score":
        target = tmp_path / fixture[artifact].receipt_path
    elif artifact == "action_registry":
        target = tmp_path / fixture[artifact].path
    else:
        target = fixture[artifact]
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ProspectiveExecutionError, match=error):
        fixture["audit"].reload(tmp_path)


def _populated_report_execution(tmp_path: Path) -> dict:
    context = _scientific_context(tmp_path)
    evidence_rows = []
    scores = []
    action_registries = []
    for index in range(2):
        plan = plan_prospective_cell(
            context["bindings"],
            context["policy"],
            evidence=ProspectiveEvidence(tuple(evidence_rows), ()),
        )
        action = plan.next_actions[0]
        _request, registry = _action_registry_binding(
            tmp_path, context, action, token=f"slot_prefix_{index}"
        )
        score = _score_receipt(
            tmp_path,
            bindings=context["bindings"],
            reference_receipt=context["reference"],
            policy=context["policy"],
            action=action,
            action_registry=registry,
        )
        action_registries.append(registry)
        scores.append(score)
        evidence_rows.append(
            score.to_planner_evidence(tmp_path, context["bindings"])
        )

    continuation = plan_prospective_cell(
        context["bindings"],
        context["policy"],
        evidence=ProspectiveEvidence(tuple(evidence_rows), ()),
    )
    full_action = next(
        row
        for row in continuation.next_actions
        if row.arm == "avqds" and row.horizon == HORIZON_FULL
    )
    _request, full_registry = _action_registry_binding(
        tmp_path, context, full_action, token="slot_full"
    )
    full_score = _score_receipt(
        tmp_path,
        bindings=context["bindings"],
        reference_receipt=context["reference"],
        policy=context["policy"],
        action=full_action,
        action_registry=full_registry,
    )
    action_registries.append(full_registry)
    scores.append(full_score)

    matrix_path = (
        tmp_path
        / "pipelines/time_dynamics/campaigns/"
        "paper_ii_parallel_benchmark_matrix_v1.json"
    )
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_bytes(DEFAULT_MATRIX_PATH.read_bytes())
    contract = build_planned_report_contract(
        lane_id=context["bindings"].lane_id,
        repository_root=tmp_path,
        matrix_path=matrix_path,
    )
    pending_report_path = tmp_path / "state/slot_pending_contract.json"
    _write(pending_report_path, contract)
    pending_report = VerifiedFile(
        pending_report_path.relative_to(tmp_path).as_posix(),
        _sha(pending_report_path),
    )
    rich_state_path = tmp_path / "state/slot_rich_state.json"
    write_scientific_state(
        rich_state_path,
        bindings=context["bindings"],
        action_entries=tuple(action_registries),
        score_receipts=tuple(scores),
        repository_root=tmp_path,
        expected_seed=context["seed_identity"],
        expected_reference=context["reference_identity"],
        expected_source_epoch=context["source"],
    )
    rich_state = load_scientific_state(
        rich_state_path,
        repository_root=tmp_path,
        expected_seed=context["seed_identity"],
        expected_reference=context["reference_identity"],
        expected_source_epoch=context["source"],
    )
    promotion_campaign_registry = _terminal_campaign_registry(
        tmp_path,
        bindings=context["bindings"],
        scientific_state=rich_state,
        planned_report_contract=pending_report,
        matrix_path=matrix_path,
    )
    promotion_request = update_report_slot_execution_request(
        context["bindings"],
        scientific_state=rich_state,
        planned_report_contract=pending_report,
        campaign_registry=promotion_campaign_registry,
    )
    promotion_audit_path = tmp_path / "evidence/slot_promotion_audit.json"
    promotion_audit = write_scientific_pre_submit_audit(
        promotion_audit_path,
        request=promotion_request,
        scientific_state=VerifiedFile(
            rich_state.receipt_path, rich_state.receipt_sha256
        ),
        planned_report_contract=pending_report,
        source_epoch_receipts=(
            VerifiedFile(
                context["source_receipt"].receipt_path,
                context["source_receipt"].receipt_sha256,
            ),
        ),
        seed_receipts=(
            VerifiedFile(context["seed"].receipt_path, context["seed"].receipt_sha256),
        ),
        exact_reference_receipts=(
            VerifiedFile(
                context["reference"].receipt_path,
                context["reference"].receipt_sha256,
            ),
        ),
        campaign_registry=promotion_campaign_registry,
        repository_root=tmp_path,
        matrix_path=matrix_path,
    )
    populated = update_method_slot(
        contract,
        cell_id=context["bindings"].cell_id,
        method="avqds",
        action_registry=full_registry,
        score_receipt=full_score,
        scientific_pre_submit_audit=promotion_audit,
        campaign_registry=promotion_campaign_registry,
        repository_root=tmp_path,
        matrix_path=matrix_path,
    )
    populated_path = tmp_path / "state/slot_populated_contract.json"
    _write(populated_path, populated)
    populated_file = VerifiedFile(
        populated_path.relative_to(tmp_path).as_posix(), _sha(populated_path)
    )

    sparse_state_path = tmp_path / "state/slot_sparse_state.json"
    write_scientific_state(
        sparse_state_path,
        bindings=context["bindings"],
        action_entries=(action_registries[0],),
        score_receipts=(scores[0],),
        repository_root=tmp_path,
        expected_seed=context["seed_identity"],
        expected_reference=context["reference_identity"],
        expected_source_epoch=context["source"],
    )
    sparse_state = load_scientific_state(
        sparse_state_path,
        repository_root=tmp_path,
        expected_seed=context["seed_identity"],
        expected_reference=context["reference_identity"],
        expected_source_epoch=context["source"],
    )
    campaign_registry = _terminal_campaign_registry(
        tmp_path,
        bindings=context["bindings"],
        scientific_state=sparse_state,
        planned_report_contract=populated_file,
        matrix_path=matrix_path,
    )
    request = update_report_slot_execution_request(
        context["bindings"],
        scientific_state=sparse_state,
        planned_report_contract=populated_file,
        campaign_registry=campaign_registry,
    )
    audit_path = tmp_path / "evidence/slot_current_audit.json"
    audit = write_scientific_pre_submit_audit(
        audit_path,
        request=request,
        scientific_state=VerifiedFile(
            sparse_state.receipt_path, sparse_state.receipt_sha256
        ),
        planned_report_contract=populated_file,
        source_epoch_receipts=(
            VerifiedFile(
                context["source_receipt"].receipt_path,
                context["source_receipt"].receipt_sha256,
            ),
        ),
        seed_receipts=(
            VerifiedFile(context["seed"].receipt_path, context["seed"].receipt_sha256),
        ),
        exact_reference_receipts=(
            VerifiedFile(
                context["reference"].receipt_path,
                context["reference"].receipt_sha256,
            ),
        ),
        campaign_registry=campaign_registry,
        repository_root=tmp_path,
        matrix_path=matrix_path,
    )
    return {
        "audit": audit,
        "full_score": full_score,
        "full_registry": full_registry,
        "promotion_audit": promotion_audit,
    }


def test_pre_submit_audit_transitively_inventories_populated_report_slot(
    tmp_path: Path,
) -> None:
    fixture = _populated_report_execution(tmp_path)
    expected = {
        VerifiedFile(
            fixture["full_score"].receipt_path,
            fixture["full_score"].receipt_sha256,
        ),
        fixture["full_score"].run_receipt,
        fixture["full_registry"],
        VerifiedFile(
            fixture["promotion_audit"].receipt_path,
            fixture["promotion_audit"].receipt_sha256,
        ),
    }
    assert expected.issubset(set(fixture["audit"].evidence_files))


@pytest.mark.parametrize(
    ("artifact", "operation"),
    (
        ("score", "tamper"),
        ("run", "missing"),
        ("action", "tamper"),
        ("audit", "tamper"),
    ),
)
def test_pre_submit_audit_rejects_stale_populated_report_slot_evidence(
    tmp_path: Path,
    artifact: str,
    operation: str,
) -> None:
    fixture = _populated_report_execution(tmp_path)
    targets = {
        "score": tmp_path / fixture["full_score"].receipt_path,
        "run": tmp_path / fixture["full_score"].run_receipt.path,
        "action": tmp_path / fixture["full_registry"].path,
        "audit": tmp_path / fixture["promotion_audit"].receipt_path,
    }
    target = targets[artifact]
    if operation == "missing":
        target.unlink()
    else:
        target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ProspectiveExecutionError, match="report slot|artifact digest"):
        fixture["audit"].reload(tmp_path)


def _populated_exact_report_execution(tmp_path: Path) -> dict:
    identities = build_identity_registry()
    references = tuple(
        row for row in identities.references if row.lane_id == LANE_HH
    )
    current_context = _scientific_context(
        tmp_path,
        reference_identity=references[0],
        token="exact_current",
    )
    slot_context = _scientific_context(
        tmp_path,
        reference_identity=references[1],
        token="exact_slot",
    )
    matrix_path = (
        tmp_path
        / "pipelines/time_dynamics/campaigns/"
        "paper_ii_parallel_benchmark_matrix_v1.json"
    )
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_bytes(DEFAULT_MATRIX_PATH.read_bytes())
    contract = build_planned_report_contract(
        lane_id=LANE_HH,
        repository_root=tmp_path,
        matrix_path=matrix_path,
    )
    pending_path = tmp_path / "state/exact_pending_contract.json"
    _write(pending_path, contract)
    pending_file = VerifiedFile(
        pending_path.relative_to(tmp_path).as_posix(), _sha(pending_path)
    )
    slot_state_path = tmp_path / "state/exact_slot_state.json"
    write_scientific_state(
        slot_state_path,
        bindings=slot_context["bindings"],
        action_entries=(),
        score_receipts=(),
        repository_root=tmp_path,
        expected_seed=slot_context["seed_identity"],
        expected_reference=slot_context["reference_identity"],
        expected_source_epoch=slot_context["source"],
    )
    slot_state = load_scientific_state(
        slot_state_path,
        repository_root=tmp_path,
        expected_seed=slot_context["seed_identity"],
        expected_reference=slot_context["reference_identity"],
        expected_source_epoch=slot_context["source"],
    )
    slot_campaign_registry = _terminal_campaign_registry(
        tmp_path,
        bindings=slot_context["bindings"],
        scientific_state=slot_state,
        planned_report_contract=pending_file,
        matrix_path=matrix_path,
    )
    slot_request = update_report_slot_execution_request(
        slot_context["bindings"],
        scientific_state=slot_state,
        planned_report_contract=pending_file,
        campaign_registry=slot_campaign_registry,
    )
    slot_audit_path = tmp_path / "evidence/exact_slot_audit.json"
    slot_audit = write_scientific_pre_submit_audit(
        slot_audit_path,
        request=slot_request,
        scientific_state=VerifiedFile(
            slot_state.receipt_path, slot_state.receipt_sha256
        ),
        planned_report_contract=pending_file,
        source_epoch_receipts=(
            VerifiedFile(
                slot_context["source_receipt"].receipt_path,
                slot_context["source_receipt"].receipt_sha256,
            ),
        ),
        seed_receipts=(
            VerifiedFile(
                slot_context["seed"].receipt_path,
                slot_context["seed"].receipt_sha256,
            ),
        ),
        exact_reference_receipts=(
            VerifiedFile(
                slot_context["reference"].receipt_path,
                slot_context["reference"].receipt_sha256,
            ),
        ),
        campaign_registry=slot_campaign_registry,
        repository_root=tmp_path,
        matrix_path=matrix_path,
    )
    populated = update_exact_reference(
        contract,
        cell_id=slot_context["bindings"].cell_id,
        reference_receipt=slot_context["reference"],
        seed_receipt=slot_context["seed"],
        source_epoch_receipt=slot_context["source_receipt"],
        scientific_pre_submit_audit=slot_audit,
        campaign_registry=slot_campaign_registry,
        repository_root=tmp_path,
        matrix_path=matrix_path,
    )
    populated_path = tmp_path / "state/exact_populated_contract.json"
    _write(populated_path, populated)
    populated_file = VerifiedFile(
        populated_path.relative_to(tmp_path).as_posix(), _sha(populated_path)
    )
    current_state_path = tmp_path / "state/exact_current_state.json"
    write_scientific_state(
        current_state_path,
        bindings=current_context["bindings"],
        action_entries=(),
        score_receipts=(),
        repository_root=tmp_path,
        expected_seed=current_context["seed_identity"],
        expected_reference=current_context["reference_identity"],
        expected_source_epoch=current_context["source"],
    )
    current_state = load_scientific_state(
        current_state_path,
        repository_root=tmp_path,
        expected_seed=current_context["seed_identity"],
        expected_reference=current_context["reference_identity"],
        expected_source_epoch=current_context["source"],
    )
    campaign_registry = _terminal_campaign_registry(
        tmp_path,
        bindings=current_context["bindings"],
        scientific_state=current_state,
        planned_report_contract=populated_file,
        matrix_path=matrix_path,
    )
    request = update_report_slot_execution_request(
        current_context["bindings"],
        scientific_state=current_state,
        planned_report_contract=populated_file,
        campaign_registry=campaign_registry,
    )
    audit_path = tmp_path / "evidence/exact_current_audit.json"
    audit = write_scientific_pre_submit_audit(
        audit_path,
        request=request,
        scientific_state=VerifiedFile(
            current_state.receipt_path, current_state.receipt_sha256
        ),
        planned_report_contract=populated_file,
        source_epoch_receipts=(
            VerifiedFile(
                current_context["source_receipt"].receipt_path,
                current_context["source_receipt"].receipt_sha256,
            ),
        ),
        seed_receipts=(
            VerifiedFile(
                current_context["seed"].receipt_path,
                current_context["seed"].receipt_sha256,
            ),
        ),
        exact_reference_receipts=(
            VerifiedFile(
                current_context["reference"].receipt_path,
                current_context["reference"].receipt_sha256,
            ),
        ),
        campaign_registry=campaign_registry,
        repository_root=tmp_path,
        matrix_path=matrix_path,
    )
    return {
        "audit": audit,
        "slot_context": slot_context,
        "slot_audit": slot_audit,
    }


def test_pre_submit_audit_transitively_inventories_populated_exact_slot(
    tmp_path: Path,
) -> None:
    fixture = _populated_exact_report_execution(tmp_path)
    context = fixture["slot_context"]
    expected = {
        VerifiedFile(
            context["source_receipt"].receipt_path,
            context["source_receipt"].receipt_sha256,
        ),
        context["source_receipt"].source_tree_manifest,
        VerifiedFile(context["seed"].receipt_path, context["seed"].receipt_sha256),
        context["seed"].artifact,
        VerifiedFile(
            context["reference"].receipt_path,
            context["reference"].receipt_sha256,
        ),
        context["reference"].manifest,
        context["reference"].data,
        VerifiedFile(
            fixture["slot_audit"].receipt_path,
            fixture["slot_audit"].receipt_sha256,
        ),
    }
    assert expected.issubset(set(fixture["audit"].evidence_files))


@pytest.mark.parametrize(
    ("artifact", "operation"),
    (
        ("source_manifest", "missing"),
        ("seed_artifact", "tamper"),
        ("reference_data", "tamper"),
        ("audit", "tamper"),
    ),
)
def test_pre_submit_audit_rejects_stale_populated_exact_slot_evidence(
    tmp_path: Path,
    artifact: str,
    operation: str,
) -> None:
    fixture = _populated_exact_report_execution(tmp_path)
    context = fixture["slot_context"]
    targets = {
        "source_manifest": (
            tmp_path / context["source_receipt"].source_tree_manifest.path
        ),
        "seed_artifact": tmp_path / context["seed"].artifact.path,
        "reference_data": tmp_path / context["reference"].data.path,
        "audit": tmp_path / fixture["slot_audit"].receipt_path,
    }
    target = targets[artifact]
    if operation == "missing":
        target.unlink()
    else:
        target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ProspectiveExecutionError, match="exact slot|artifact digest"):
        fixture["audit"].reload(tmp_path)


def test_terminal_executor_requests_reject_qiskit_and_missing_handler_route(
    tmp_path: Path,
) -> None:
    fixture = _auditable_execution(tmp_path, token="route")
    payload = fixture["request"].to_json_dict()
    payload["inputs"] = {**payload["inputs"], "qiskit_compilation": True}
    payload["execution_request_sha256"] = _canonical_sha(
        {
            key: value
            for key, value in payload.items()
            if key != "execution_request_sha256"
        }
    )
    with pytest.raises(ProspectiveExecutionError, match="Qiskit"):
        type(fixture["request"]).from_mapping(payload)

    missing = ExecutorBinding(
        binding_id="missing-package-route",
        lanes=(LANE_HH,),
        kinds=("build_package",),
        handler=(
            "pipelines.time_dynamics.prospective_execution_bindings:"
            "missing_package_request"
        ),
        output_role="package_manifest",
    )
    with pytest.raises(ProspectiveExecutionError, match="not callable"):
        missing.resolve()


def test_completed_run_lifecycle_receipt_cannot_be_an_orphan(
    tmp_path: Path,
) -> None:
    from pipelines.time_dynamics.prospective_execution_bindings import (
        RUN_LIFECYCLE_RECEIPT_V1,
        RUN_STATE_EVIDENCE_V1,
        RunLifecycleReceipt,
    )

    context = _scientific_context(tmp_path)
    action = plan_prospective_cell(
        context["bindings"], context["policy"]
    ).next_actions[0]
    request = resolve_scientific_action(context["bindings"], action)
    action_path = tmp_path / "state/actions/run.json"
    write_action_registry_entry(
        action_path,
        request,
        source_receipts=(
            VerifiedFile(
                context["source_receipt"].receipt_path,
                context["source_receipt"].receipt_sha256,
            ),
            VerifiedFile(
                context["seed"].receipt_path,
                context["seed"].receipt_sha256,
            ),
            VerifiedFile(
                context["reference"].receipt_path,
                context["reference"].receipt_sha256,
            ),
        ),
        repository_root=tmp_path,
        bindings=context["bindings"],
        expected_seed=context["seed_identity"],
        expected_reference=context["reference_identity"],
        expected_source_epoch=context["source"],
    )
    action_binding = VerifiedFile(
        action_path.relative_to(tmp_path).as_posix(), _sha(action_path)
    )
    state_path = tmp_path / "evidence/completed_state.json"
    _write(
        state_path,
        _sealed(
            {
                "schema": RUN_STATE_EVIDENCE_V1,
                "scientific_action_id": action.action_id,
                "execution_id": action.execution_id,
                "state": "completed",
            },
            "run_state_evidence_sha256",
        ),
    )
    run_path = tmp_path / "evidence/run.json"
    _write(run_path, {"status": "complete"})
    payload = {
        "schema": RUN_LIFECYCLE_RECEIPT_V1,
        "scientific_action_id": action.action_id,
        "execution_id": action.execution_id,
        "cell_id": context["bindings"].cell_id,
        "method": action.arm,
        "horizon": action.horizon,
        "activation_cut": action.activation_cut,
        "attempt": action.attempt,
        "planner_sequence": action.planner_sequence,
        "requested_memory_gib": action.requested_memory_gib,
        "state": "completed",
        "bindings_sha256": context["bindings"].digest,
        "execution_request_sha256": request.digest,
        "action_registry": action_binding.to_json_dict(),
        "source_epoch_identity_sha256": (
            context["bindings"].source_epoch_identity_sha256
        ),
        "seed_receipt_sha256": context["bindings"].seed_receipt_sha256,
        "exact_reference_receipt_sha256": (
            context["bindings"].reference_receipt_sha256
        ),
        "accuracy_policy_sha256": (
            context["bindings"].accuracy_policy_sha256
        ),
        "resume_from_action_id": None,
        "resume_run_receipt_sha256": None,
        "resume_run_sha256": None,
        "previous_receipt": None,
        "state_evidence": {
            "path": state_path.relative_to(tmp_path).as_posix(),
            "sha256": _sha(state_path),
        },
        "run": {
            "path": run_path.relative_to(tmp_path).as_posix(),
            "sha256": _sha(run_path),
        },
        "failure_kind": None,
        "failure_reason": "",
        "failure_evidence": None,
        "observed_peak_gib": None,
    }
    receipt_path = tmp_path / "evidence/orphan_completed.json"
    _write(
        receipt_path,
        _sealed(payload, "run_lifecycle_receipt_sha256"),
    )
    with pytest.raises(ProspectiveExecutionError, match="completed.*previous"):
        RunLifecycleReceipt.load(receipt_path, repository_root=tmp_path)


def test_run_lifecycle_accepts_only_durable_legal_transitions(
    tmp_path: Path,
) -> None:
    context = _scientific_context(tmp_path)
    action = plan_prospective_cell(
        context["bindings"], context["policy"]
    ).next_actions[0]
    request, action_registry = _action_registry_binding(
        tmp_path, context, action
    )
    submitted = write_run_lifecycle_receipt(
        tmp_path / "evidence/run_submitted.json",
        request=request,
        state="submitted",
        action_registry=action_registry,
        state_evidence=_run_state_evidence(
            tmp_path, action, "submitted", token="submitted"
        ),
        repository_root=tmp_path,
    )
    submitted_file = VerifiedFile(
        submitted.receipt_path, submitted.receipt_sha256
    )
    in_flight = write_run_lifecycle_receipt(
        tmp_path / "evidence/run_in_flight.json",
        request=request,
        state="in_flight",
        action_registry=action_registry,
        state_evidence=_run_state_evidence(
            tmp_path, action, "in_flight", token="in_flight"
        ),
        previous_receipt=submitted_file,
        repository_root=tmp_path,
    )
    run_path = tmp_path / "evidence/run_complete.json"
    _write(run_path, {"status": "complete"})
    completed = write_run_lifecycle_receipt(
        tmp_path / "evidence/run_completed.json",
        request=request,
        state="completed",
        action_registry=action_registry,
        state_evidence=_run_state_evidence(
            tmp_path, action, "completed", token="completed"
        ),
        previous_receipt=VerifiedFile(
            in_flight.receipt_path, in_flight.receipt_sha256
        ),
        run=VerifiedFile(run_path.relative_to(tmp_path).as_posix(), _sha(run_path)),
        repository_root=tmp_path,
    )
    evidence = completed.to_planner_evidence(
        tmp_path, context["bindings"]
    )
    assert evidence.state == "completed"
    assert evidence.run_receipt_sha256 == completed.receipt_sha256
    assert evidence.run_sha256 == _sha(run_path)
    rebuilt = durable_receipts_to_evidence(
        (completed,),
        repository_root=tmp_path,
        bindings=context["bindings"],
    )
    assert rebuilt.runs == (evidence,)
    with pytest.raises(ProspectiveExecutionError, match="changed after validation"):
        replace(completed, state="failed").to_planner_evidence(
            tmp_path, context["bindings"]
        )

    direct_run_path = tmp_path / "evidence/run_complete_direct.json"
    _write(direct_run_path, {"status": "complete", "route": "direct"})
    direct_completed = write_run_lifecycle_receipt(
        tmp_path / "evidence/run_completed_direct.json",
        request=request,
        state="completed",
        action_registry=action_registry,
        state_evidence=_run_state_evidence(
            tmp_path, action, "completed", token="completed_direct"
        ),
        previous_receipt=submitted_file,
        run=VerifiedFile(
            direct_run_path.relative_to(tmp_path).as_posix(),
            _sha(direct_run_path),
        ),
        repository_root=tmp_path,
    )
    with pytest.raises(ProspectiveExecutionError, match="divergent attempt tips"):
        durable_receipts_to_evidence(
            (completed, direct_completed),
            repository_root=tmp_path,
            bindings=context["bindings"],
        )

    illegal_state = _run_state_evidence(
        tmp_path, action, "in_flight", token="late_in_flight"
    )
    with pytest.raises(ProspectiveExecutionError, match="illegal.*completed->in_flight"):
        write_run_lifecycle_receipt(
            tmp_path / "evidence/illegal_late_in_flight.json",
            request=request,
            state="in_flight",
            action_registry=action_registry,
            state_evidence=illegal_state,
            previous_receipt=VerifiedFile(
                completed.receipt_path, completed.receipt_sha256
            ),
            repository_root=tmp_path,
        )

    state_path = tmp_path / submitted.state_evidence.path
    state_path.write_text("tampered\n")
    with pytest.raises(ProspectiveExecutionError, match="artifact digest mismatch"):
        completed.reload(tmp_path)


def test_memory_failure_receipt_authorizes_exactly_one_typed_retry(
    tmp_path: Path,
) -> None:
    context = _scientific_context(tmp_path)
    action = plan_prospective_cell(
        context["bindings"], context["policy"]
    ).next_actions[0]
    request, action_registry = _action_registry_binding(
        tmp_path, context, action, token="attempt0"
    )
    submitted = write_run_lifecycle_receipt(
        tmp_path / "evidence/attempt0_submitted.json",
        request=request,
        state="submitted",
        action_registry=action_registry,
        state_evidence=_run_state_evidence(
            tmp_path, action, "submitted", token="attempt0_submitted"
        ),
        repository_root=tmp_path,
    )
    reason = "verified scheduler memory exhaustion"
    failed = write_run_lifecycle_receipt(
        tmp_path / "evidence/attempt0_failed.json",
        request=request,
        state="failed",
        action_registry=action_registry,
        state_evidence=_run_state_evidence(
            tmp_path, action, "failed", token="attempt0_failed"
        ),
        previous_receipt=VerifiedFile(
            submitted.receipt_path, submitted.receipt_sha256
        ),
        failure_kind=FAILURE_MEMORY,
        failure_reason=reason,
        failure_evidence=_run_failure_evidence(
            tmp_path,
            action,
            failure_kind=FAILURE_MEMORY,
            failure_reason=reason,
            observed_peak_gib=5.5,
            token="attempt0",
        ),
        observed_peak_gib=5.5,
        repository_root=tmp_path,
    )
    failed_evidence = failed.to_planner_evidence(
        tmp_path, context["bindings"]
    )
    assert failed_evidence.failure_receipt_sha256 == failed.receipt_sha256
    retry = plan_prospective_cell(
        context["bindings"],
        context["policy"],
        evidence=ProspectiveEvidence(runs=(failed_evidence,)),
    ).next_actions[0]
    assert retry.attempt == 1
    assert retry.requested_memory_gib == 12
    retry_request, retry_registry = _action_registry_binding(
        tmp_path, context, retry, token="attempt1"
    )
    retry_submitted = write_run_lifecycle_receipt(
        tmp_path / "evidence/attempt1_submitted.json",
        request=retry_request,
        state="submitted",
        action_registry=retry_registry,
        state_evidence=_run_state_evidence(
            tmp_path, retry, "submitted", token="attempt1_submitted"
        ),
        previous_receipt=VerifiedFile(failed.receipt_path, failed.receipt_sha256),
        repository_root=tmp_path,
    )
    assert retry_submitted.attempt == 1
    assert retry_submitted.requested_memory_gib == 12
    rebuilt = durable_receipts_to_evidence(
        (retry_submitted,),
        repository_root=tmp_path,
        bindings=context["bindings"],
    )
    assert [(row.attempt, row.state) for row in rebuilt.runs] == [
        (0, "failed"),
        (1, "submitted"),
    ]

    forged_retry = replace(
        retry,
        requested_memory_gib=16,
        execution_id=f"{retry.action_id}::attempt1::mem16g",
    )
    forged_request, forged_registry = _action_registry_binding(
        tmp_path, context, forged_retry, token="forged_attempt1"
    )
    with pytest.raises(ProspectiveExecutionError, match="unauthorized memory"):
        write_run_lifecycle_receipt(
            tmp_path / "evidence/forged_retry.json",
            request=forged_request,
            state="submitted",
            action_registry=forged_registry,
            state_evidence=_run_state_evidence(
                tmp_path, forged_retry, "submitted", token="forged_retry"
            ),
            previous_receipt=VerifiedFile(
                failed.receipt_path, failed.receipt_sha256
            ),
            repository_root=tmp_path,
        )
