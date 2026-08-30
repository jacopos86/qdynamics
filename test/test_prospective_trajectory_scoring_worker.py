"""Focused exact-join tests for prospective trajectory scoring."""

from __future__ import annotations

import copy
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pytest

from pipelines.time_dynamics.adapters.observables import hh_observable_snapshot
from pipelines.time_dynamics.prospective_exact_reference_worker import (
    _drive_diagonal,
    _static_hamiltonian,
    _waveform,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    ExactReferenceReceipt,
    ProspectiveExecutionError,
    VerifiedFile,
    build_scientific_bindings,
    resolve_scientific_action,
    write_run_lifecycle_receipt,
)
from pipelines.time_dynamics.prospective_scientific_planner import (
    ProspectiveEvidence,
    plan_prospective_cell,
)
from pipelines.time_dynamics.prospective_trajectory_scoring_worker import (
    APM_COMPLETION_SCHEMA_V1,
    AVQDS_COMPLETION_SCHEMA_V1,
    SCIENTIFIC_AVQDS_COMPLETION_SCHEMA_V1,
    materialization_request,
    score_completed_trajectory,
)
from pipelines.time_dynamics.runners.score_prospective_trajectory import main as score_main
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
    RUNNER_SCHEMA_V1,
)
from test_prospective_execution_bindings import (
    _action_registry_binding,
    _canonical_sha,
    _policy,
    _prospective_rows,
    _reference_receipt,
    _run_state_evidence,
    _sealed,
    _seed_receipt,
    _sha,
    _source_epoch,
    _source_receipt,
    _write,
)


def _dimension(reference) -> int:
    sites = int(reference.physical["lattice_sites"])
    if reference.model == "hubbard":
        return 1 << (2 * sites)
    levels = int(reference.physical["n_ph_max"]) + 1
    qpb = max(1, (levels - 1).bit_length())
    return 1 << (2 * sites + sites * qpb)


def _expand_exact_states(
    root: Path,
    *,
    reference,
    seed_identity,
    seed_receipt,
    source_epoch,
    loaded_reference,
) -> ExactReferenceReceipt:
    data_path = root / loaded_reference.data.path
    manifest_path = root / loaded_reference.manifest.path
    receipt_path = root / loaded_reference.receipt_path
    data = json.loads(data_path.read_text(encoding="utf-8"))
    dimension = _dimension(reference)
    real = [1.0, *([0.0] * (dimension - 1))]
    imag = [0.0] * dimension
    data["statevector_real"] = [list(real) for _ in data["time"]]
    data["statevector_imag"] = [list(imag) for _ in data["time"]]
    _write(data_path, data)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["data_sha256"] = _sha(data_path)
    _write(manifest_path, _sealed(manifest, "exact_manifest_sha256"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["data"]["sha256"] = _sha(data_path)
    receipt["manifest"]["sha256"] = _sha(manifest_path)
    _write(
        receipt_path,
        _sealed(receipt, "exact_reference_receipt_sha256"),
    )
    return ExactReferenceReceipt.load(
        receipt_path,
        repository_root=root,
        expected_reference=reference,
        expected_seed_identity=seed_identity,
        expected_seed_receipt=seed_receipt,
        expected_source_epoch=source_epoch,
    )


def _raw_run(
    reference,
    seed,
    *,
    bad_grid: bool,
    bad_dimension: bool,
    recorded_observables: str = "full",
) -> dict:
    grid = reference.grid
    times = tuple(
        float(grid["start"]) + index * float(grid["step"])
        for index in range(int(grid["prefix_points"]))
    )
    dimension = _dimension(reference)
    state = np.zeros(dimension, dtype=complex)
    state[0] = 1.0
    static = _static_hamiltonian(reference)
    drive_diagonal = _drive_diagonal(reference, dimension)
    diagonal = np.diag_indices_from(static)
    snapshot = hh_observable_snapshot(
        state,
        num_sites=int(reference.physical["lattice_sites"]),
        ordering=str(reference.physical["ordering"]),
    )
    rows = []
    for time_value in times:
        instantaneous = np.array(static, copy=True)
        instantaneous[diagonal] += _waveform(reference, time_value) * drive_diagonal
        rows.append(
            {
                "time": time_value,
                "energy_expectation": float(
                    np.real(np.vdot(state, instantaneous @ state))
                ),
                "doublon": float(snapshot["doublon"]),
                "site_occupations": list(snapshot["site_occupations"]),
                "psi_real": [float(value) for value in state.real],
                "psi_imag": [float(value) for value in state.imag],
                "runtime_parameter_count": 1,
                "logical_parameter_count": 1,
            }
        )
    if bad_grid:
        rows[-1]["time"] += 1.0e-3
    if bad_dimension:
        rows[0]["psi_real"] = rows[0]["psi_real"][:-1]
        rows[0]["psi_imag"] = rows[0]["psi_imag"][:-1]
    if recorded_observables == "missing":
        for row in rows:
            row.pop("doublon")
            row.pop("site_occupations")
    elif recorded_observables == "partial":
        for row in rows:
            row.pop("site_occupations")
    elif recorded_observables != "full":
        raise ValueError("unsupported recorded-observable fixture mode")
    nq = int(math.log2(dimension))
    coordinate = {
        "label": "seed::r0::" + "x" + "e" * (nq - 1),
        "pauli_exyz": "x" + "e" * (nq - 1),
        "coeff_real": 1.0,
        "nq": nq,
    }
    return {
        "schema": RUNNER_SCHEMA_V1,
        "run_lock": {
            "physics": {
                "family_key": "hh",
                "seed_sha256": seed.artifact.sha256,
                "time_grid": {
                    "point_count": len(times),
                    "t_initial": times[0],
                    "t_final": times[-1],
                    "ordered_values_sha256": grid["ordered_prefix_sha256"],
                },
                "exact_reference_json": None,
                "exact_reference_sha256": None,
            },
            "code_revision": "b" * 40,
        },
        "resume_state": {
            "schema": "paper_ii_resume_state_v1",
            "time": times[-1],
            "problem_family": "hh",
            "parameterization_mode": "per_pauli_term",
            "coordinates": [coordinate],
            "theta_runtime": [0.0],
        },
        "initial_state": {
            "parameterization_mode": "per_pauli_term",
            "runtime_parameter_count": 1,
        },
        "final_state": {
            "parameterization_mode": "per_pauli_term",
            "runtime_parameter_count": 1,
            "selected_term_labels": [coordinate["label"]],
        },
        "summary": {
            "point_count": len(times),
            "time_initial": times[0],
            "time_final": times[-1],
            "accepted_internal_substep_count": len(times) - 1,
            "runtime_parameter_count_final": 1,
        },
        "plot_rows": rows,
        "decision_data_flow": {
            "uses_reference_for_decision": False,
            "uses_exact_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
            "uses_statevector_as_ideal_observable_estimator": True,
        },
    }


def _write_state_evidence(
    root: Path,
    *,
    action,
    state: str,
    token: str,
    extras: dict | None = None,
) -> VerifiedFile:
    path = root / f"evidence/state_{token}.json"
    payload = {
        "schema": "paper_ii_run_state_evidence_v1",
        "scientific_action_id": action.action_id,
        "execution_id": action.execution_id,
        "state": state,
        **(extras or {}),
    }
    _write(path, _sealed(payload, "run_state_evidence_sha256"))
    return VerifiedFile(path.relative_to(root).as_posix(), _sha(path))


def _context(
    root: Path,
    *,
    bad_grid: bool = False,
    bad_dimension: bool = False,
    completion_overrides: dict | None = None,
    adopted_bootstrap: bool = False,
    recorded_observables: str = "full",
) -> dict:
    _registry, seed_identity, reference_identity = _prospective_rows()
    source_epoch = _source_epoch()
    seed = _seed_receipt(root, source_epoch=source_epoch)
    provisional_reference = _reference_receipt(
        root,
        seed,
        source_epoch=source_epoch,
        reference_identity=reference_identity,
        seed_identity=seed_identity,
    )
    reference = _expand_exact_states(
        root,
        reference=reference_identity,
        seed_identity=seed_identity,
        seed_receipt=seed,
        source_epoch=source_epoch,
        loaded_reference=provisional_reference,
    )
    source = _source_receipt(root, source_epoch)
    policy = _policy(str(reference_identity.drive["drive_id"]))
    bindings = build_scientific_bindings(
        reference=reference_identity,
        seed=seed_identity,
        seed_receipt=seed,
        reference_receipt=reference,
        source_epoch=source_epoch,
        source_epoch_receipt=source,
    )
    initial = plan_prospective_cell(bindings, policy)
    action = initial.next_actions[0]
    assert action.arm == "avqds" and action.horizon == "t5"
    request, action_registry = _action_registry_binding(
        root, {
            "source_receipt": source,
            "seed": seed,
            "reference": reference,
            "bindings": bindings,
            "seed_identity": seed_identity,
            "reference_identity": reference_identity,
            "source": source_epoch,
        }, action,
    )
    run_path = root / "raw/run.json"
    _write(
        run_path,
        _raw_run(
            reference_identity,
            seed,
            bad_grid=bad_grid,
            bad_dimension=bad_dimension,
            recorded_observables=recorded_observables,
        ),
    )
    run_file = VerifiedFile(run_path.relative_to(root).as_posix(), _sha(run_path))
    bootstrap_action = "bootstrap::screen::test"
    bootstrap_request_sha = "c" * 64
    completion_core = {
        "schema": AVQDS_COMPLETION_SCHEMA_V1,
        "action_id": (
            bootstrap_action if adopted_bootstrap else request.action_id
        ),
        "execution_request_sha256": (
            bootstrap_request_sha if adopted_bootstrap else request.digest
        ),
        "lane_id": bindings.lane_id,
        "cell_id": bindings.cell_id,
        "method": "avqds",
        "horizon": "t5",
        "status": "completed_unscored",
        "run": run_file.to_json_dict(),
        "work_order_sha256": "d" * 64,
        "bindings_sha256": bindings.digest,
        "seed_identity_sha256": seed_identity.identity_sha256,
        "seed_artifact_sha256": seed.artifact.sha256,
        "seed_receipt_sha256": seed.receipt_sha256,
        "source_epoch_identity_sha256": source_epoch.identity_sha256,
        "source_epoch_receipt_sha256": source.receipt_sha256,
        "executor_overlay_identity_sha256": "e" * 64,
        "executor_overlay_revision": "b" * 40,
        "composite_source_identity_sha256": "f" * 64,
        "activation_ladder_sha256": "a" * 64,
        "activation_cut": action.activation_cut,
        "scientific_qualification": None,
        "score_receipt": None,
        "scoring_required_before_qualification": True,
        "remote_raw_evidence_policy": "preserve",
        "remote_cleanup_authorized": False,
        "qiskit_compilation_used": False,
    }
    completion_core.update(completion_overrides or {})
    completion_path = root / "raw/completion_receipt.json"
    _write(
        completion_path,
        _sealed(completion_core, "completion_receipt_sha256"),
    )
    completion_file = VerifiedFile(
        completion_path.relative_to(root).as_posix(), _sha(completion_path)
    )

    submitted = write_run_lifecycle_receipt(
        root / "evidence/submitted_lifecycle.json",
        request=request,
        state="submitted",
        action_registry=action_registry,
        state_evidence=_run_state_evidence(
            root, action, "submitted", token="score_submitted"
        ),
        repository_root=root,
    )
    adoption_extras = None
    if adopted_bootstrap:
        adoption_extras = {
            "evidence_kind": "verified_bootstrap_scout_evidence_reuse",
            "scheduler_submission_performed": False,
            "bootstrap_action_id": bootstrap_action,
            "bootstrap_execution_request_sha256": bootstrap_request_sha,
            "bootstrap_completion_receipt": completion_file.to_json_dict(),
            "reused_run": run_file.to_json_dict(),
            "remote_raw_evidence_policy": "preserve",
            "scientific_qualification": None,
        }
    completed = write_run_lifecycle_receipt(
        root / "evidence/completed_lifecycle.json",
        request=request,
        state="completed",
        action_registry=action_registry,
        state_evidence=_write_state_evidence(
            root,
            action=action,
            state="completed",
            token="score_completed",
            extras=adoption_extras,
        ),
        repository_root=root,
        previous_receipt=VerifiedFile(
            submitted.receipt_path, submitted.receipt_sha256
        ),
        run=run_file,
    )
    completed_file = VerifiedFile(
        completed.receipt_path, completed.receipt_sha256
    )
    evidence = completed.to_planner_evidence(root, bindings)
    score_plan = plan_prospective_cell(
        bindings,
        policy,
        evidence=ProspectiveEvidence(runs=(evidence,)),
    )
    assert len(score_plan.next_actions) == 1
    score_action = score_plan.next_actions[0]
    assert score_action.kind == "score"
    score_request = resolve_scientific_action(bindings, score_action)
    planner_path = root / "evidence/score_plan.json"
    _write(planner_path, score_plan.to_json_dict())
    planner_file = VerifiedFile(
        planner_path.relative_to(root).as_posix(), _sha(planner_path)
    )
    return {
        "bindings": bindings,
        "reference": reference,
        "score_request": score_request,
        "completed": completed_file,
        "completion": completion_file,
        "planner": planner_file,
        "run": run_file,
    }


def _score(root: Path, context: dict):
    return score_completed_trajectory(
        context["score_request"],
        bindings=context["bindings"],
        completed_run_receipt=context["completed"],
        raw_completion_receipt=context["completion"],
        exact_reference_receipt=VerifiedFile(
            context["reference"].receipt_path,
            context["reference"].receipt_sha256,
        ),
        planner_receipt=context["planner"],
        metrics_path="scores/metrics.json",
        score_receipt_path="scores/score_receipt.json",
        repository_root=root,
    )


def test_scores_exact_join_with_qiskit_free_costs_and_idempotent_replay(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    first = _score(tmp_path, context)
    second = _score(tmp_path, context)

    assert first == second
    metrics = json.loads((tmp_path / first.metrics.path).read_text())
    assert metrics["schema"] == "paper_ii_exact_scored_metrics_v1"
    assert metrics["raw_completion_receipt_sha256"] == context["completion"].sha256
    assert metrics["response_normalized_evaluation"]["pair_matching"][
        "state"
    ] == "not_evaluated_single_arm"
    assert metrics["response_normalized_evaluation"]["pair_matching"][
        "monotonicity_inference_used"
    ] is False
    assert metrics["canonical_costs"]["qiskit_compilation"]["performed"] is False
    assert metrics["canonical_costs"]["logical_coordinates"][
        "accepted_integration_steps"
    ] == 125
    assert metrics["canonical_costs"]["routing_independent_graph_tuple"][
        "named"
    ]["N_theta"] == 1


def test_accepts_digest_bound_bootstrap_scout_adoption(tmp_path: Path) -> None:
    receipt = _score(tmp_path, _context(tmp_path, adopted_bootstrap=True))
    assert receipt.method == "avqds"
    assert receipt.horizon == "t5"


def test_accepts_direct_scientific_avqds_completion_schema(
    tmp_path: Path,
) -> None:
    receipt = _score(
        tmp_path,
        _context(
            tmp_path,
            completion_overrides={
                "schema": SCIENTIFIC_AVQDS_COMPLETION_SCHEMA_V1
            },
        ),
    )
    assert receipt.method == "avqds"
    assert receipt.horizon == "t5"


def test_rejects_stale_exact_data_digest(tmp_path: Path) -> None:
    context = _context(tmp_path)
    data_path = tmp_path / context["reference"].data.path
    data_path.write_bytes(data_path.read_bytes() + b" ")
    with pytest.raises(ProspectiveExecutionError, match="artifact digest mismatch"):
        _score(tmp_path, context)


def test_rejects_wrong_seed_or_source_epoch(tmp_path: Path) -> None:
    context = _context(
        tmp_path,
        completion_overrides={"source_epoch_identity_sha256": "0" * 64},
    )
    with pytest.raises(ProspectiveExecutionError, match="diverges"):
        _score(tmp_path, context)


def test_rejects_wrong_grid_for_declared_horizon(tmp_path: Path) -> None:
    context = _context(tmp_path, bad_grid=True)
    with pytest.raises(ProspectiveExecutionError, match="grid|horizon"):
        _score(tmp_path, context)


def test_rejects_malformed_state_dimension(tmp_path: Path) -> None:
    context = _context(tmp_path, bad_dimension=True)
    with pytest.raises(ProspectiveExecutionError, match="state dimension"):
        _score(tmp_path, context)


def test_scores_statevector_only_rows_without_recorded_observables(
    tmp_path: Path,
) -> None:
    receipt = _score(
        tmp_path,
        _context(tmp_path, recorded_observables="missing"),
    )
    metrics = json.loads((tmp_path / receipt.metrics.path).read_text())
    assert metrics["metrics"]["mean_abs_doublon_error"] >= 0.0
    assert metrics["metrics"]["mean_max_abs_site_occupation_error"] >= 0.0


def test_rejects_partially_recorded_observables(tmp_path: Path) -> None:
    context = _context(tmp_path, recorded_observables="partial")
    with pytest.raises(ProspectiveExecutionError, match="partially present"):
        _score(tmp_path, context)


def test_rejects_cross_arm_raw_completion(tmp_path: Path) -> None:
    context = _context(
        tmp_path,
        completion_overrides={
            "schema": APM_COMPLETION_SCHEMA_V1,
            "method": "apm",
        },
    )
    with pytest.raises(ProspectiveExecutionError, match="diverges"):
        _score(tmp_path, context)


def test_score_materialization_adapter_is_operational_and_inert(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    recipe = materialization_request(context["score_request"])
    assert recipe["status"] == "operational_materialization_ready"
    assert recipe["worker"].endswith(":score_completed_trajectory")
    assert recipe["required_receipts"] == [
        "completed_run_lifecycle",
        "completed_unscored_raw_receipt",
        "exact_reference_receipt",
        "scientific_planner_receipt",
    ]
    assert recipe["scheduler_contacted"] is False
    assert recipe["scientific_pair_qualification"] is None
    assert recipe["qiskit_compilation_requested"] is False

    inputs = dict(context["score_request"].inputs)
    inputs["run_sha256"] = "not-a-digest"
    with pytest.raises(ProspectiveExecutionError, match="canonical"):
        materialization_request(replace(context["score_request"], inputs=inputs))


def test_score_cli_materializes_then_validates_without_replacement(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    request_path = tmp_path / "score_cli/request.json"
    bindings_path = tmp_path / "score_cli/bindings.json"
    _write(request_path, context["score_request"].to_json_dict())
    _write(bindings_path, context["bindings"].to_json_dict())
    arguments = [
        "--repository-root",
        str(tmp_path),
        "--request-json",
        str(request_path),
        "--bindings-json",
        str(bindings_path),
        "--completed-run-receipt",
        context["completed"].path,
        "--raw-completion-receipt",
        context["completion"].path,
        "--exact-reference-receipt",
        context["reference"].receipt_path,
        "--planner-receipt",
        context["planner"].path,
        "--metrics-output",
        "score_cli/metrics.json",
        "--score-receipt-output",
        "score_cli/score_receipt.json",
    ]
    assert score_main(arguments) == 0
    metrics_before = (tmp_path / "score_cli/metrics.json").read_bytes()
    receipt_before = (tmp_path / "score_cli/score_receipt.json").read_bytes()
    assert score_main([*arguments, "--validate-only"]) == 0
    assert (tmp_path / "score_cli/metrics.json").read_bytes() == metrics_before
    assert (tmp_path / "score_cli/score_receipt.json").read_bytes() == receipt_before
