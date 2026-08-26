from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
from collections import Counter
import hashlib
import json
import sys
from typing import Mapping

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.scaffold.handoff_state_bundle import build_statevector_manifest
from pipelines.scaffold.runtime_contract import ScaffoldRuntimeInput
from pipelines.scaffold.hh_continuation_types import PhaseControllerSnapshot
from pipelines.static_adapt.output_artifacts import (
    _resolved_output_powell_coordinate_chart_policy,
)
from pipelines.static_adapt.resume_scaffold import (
    ResumeScaffoldSource,
    _assert_repeated_resume_contract_consistency,
    _load_verified_singleton_controller_state,
    _load_verified_singleton_selection_state,
    assert_no_secret_material,
    build_credential_audit,
    build_resume_import_summary,
    digest_jsonable,
    _validate_authenticated_v4_pruned_lineage,
    extract_best_frontier_resume_checkpoint,
    extract_verified_singleton_resume_checkpoint,
    extract_resume_history,
    load_static_resume_source,
    match_resume_scaffold_to_pool,
    validate_resume_powell_coordinate_chart_policy,
    validate_resume_phase12_energy_model_policies,
    validate_resume_phase3_response_coordinate_scope,
    validate_static_hh_resume_source,
)
from pipelines.static_adapt.estimator_call_ledger import (
    EstimatorCallKey,
    EstimatorCallLedger,
    projective_state_fingerprint,
)
from pipelines.static_adapt.selector_measurement_proxy import (
    ControllerMeasurementWorkAccumulator,
    controller_proxy_from_history_rows,
)
from pipelines.static_adapt.sr_snake_escape_controller import (
    SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1,
    SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1,
    SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1,
    SR_ROUTE_PROFILE_CONFORMANCE_REGISTERED,
    SR_ROUTE_PROFILE_CONFORMANCE_UNPROMOTED_EXPLICIT_ABLATION,
    SR_ROUTE_PROFILE_PHASE2_PHASE3_WHITENED,
    resolve_sr_powell_route_instance,
)
from pipelines.static_adapt.sr_snake_route_profile import (
    PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1,
    PHASE3_RESPONSE_COORDINATE_SCOPE_LEGACY_REOPT_COUPLED_V1,
    SR_ROUTE_PROFILE_CONVENTIONAL_V2,
    SR_ROUTE_PROFILE_CONVENTIONAL_V3,
    SR_ROUTE_PROFILE_CANDIDATE_V4,
    canonical_sr_snake_v4_contract,
    canonical_sr_snake_v4_contract_sha256,
)
from pipelines.static_adapt.sr_snake_phase12_policy import (
    PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1,
    PHASE1_SCORE_MODE_TRUST_REGION_V1,
    PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF,
    PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1,
)
from src.quantum.ansatz_parameterization import build_parameter_layout, serialize_layout
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import compile_polynomial_action, energy_via_one_apply
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, PauliTerm


def _one_term(label: str = "resume_test_x") -> AnsatzTerm:
    poly = PauliPolynomial("JW")
    poly.add_term(PauliTerm(1, ps="x", pc=1.0))
    poly._reduce()
    return AnsatzTerm(label=label, polynomial=poly)


def _resume_source() -> ResumeScaffoldSource:
    term = _one_term()
    layout = build_parameter_layout([term], ignore_identity=True, coefficient_tolerance=1e-12, sort_terms=True)
    theta = np.array([0.25], dtype=float)
    psi_ref = np.array([1.0 + 0.0j, 0.0 + 0.0j])
    psi_initial = CompiledAnsatzExecutor(
        [term],
        parameterization_layout=layout,
        parameterization_mode="per_pauli_term",
    ).prepare_state(theta, psi_ref)
    payload = {
        "settings": {
            "problem": "hh",
            "L": 2,
            "t": 1.0,
            "u": 2.0,
            "dv": 0.0,
            "omega0": 1.0,
            "g_ep": 0.5,
            "n_ph_max": 1,
            "boson_encoding": "binary",
            "ordering": "blocked",
            "boundary": "open",
            "include_zero_point": True,
            "adapt_pool": "pareto_lean",
            "adapt_continuation_mode": "phase3_v1",
        },
        "adapt_vqe": {
            "operators": [term.label],
            "optimal_point": [0.25],
            "logical_optimal_point": [0.25],
            "parameterization": serialize_layout(layout),
            "parameterization_execution_mode": "per_pauli_term",
            "ansatz_depth": 1,
            "num_parameters": 1,
            "pool_type": "pareto_lean",
            "continuation_mode": "phase3_v1",
        },
        "ansatz_input_state": build_statevector_manifest(
            psi_state=psi_ref,
            source="hf",
            handoff_state_kind="reference_state",
            amplitude_cutoff=1e-12,
        ),
        "initial_state": build_statevector_manifest(
            psi_state=psi_initial,
            source="adapt_vqe",
            handoff_state_kind="prepared_state",
            amplitude_cutoff=1e-12,
        ),
    }
    runtime_input = ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(),
        psi_ref=psi_ref,
        psi_initial=psi_initial,
        base_layout=layout,
        theta_runtime=theta,
        theta_logical=np.array([0.25], dtype=float),
        structure_locked=False,
        exact_energy=None,
        selected_terms=(term,),
    )
    return ResumeScaffoldSource(
        artifact_json=Path("resume.json"),
        artifact_sha256="a" * 64,
        payload=payload,
        runtime_input=runtime_input,
        import_summary={},
    )


def _best_frontier_resume_source() -> ResumeScaffoldSource:
    source = _resume_source()
    payload = json.loads(json.dumps(source.payload))
    last_trust_update = {
        "schema": "route_a_trust_region_update_v1",
        "policy": "fixed",
        "radius_before": 0.5,
        "radius_after": 0.5,
    }
    trust_state = {
        "schema": "route_a_trust_region_state_v1",
        "radius": 0.5,
        "reference_radius": 0.5,
        "update_count": 1,
        "last_update": dict(last_trust_update),
    }
    history_row = {
        "depth": 1,
        "batch_size": 1,
        "branch_id": 7,
        "parent_branch_id": 0,
        "selected_op": "resume_test_x",
        "selected_position": 0,
        "energy_before_opt": -0.25,
        "energy_after_opt": -0.5,
        "post_admission_prune": {"accepted_count": 0},
        "route_a_trust_region_update": dict(last_trust_update),
        "structural_rollback": False,
        "preserved_nested_telemetry": {"sentinel": [1, 2, 3]},
    }
    frontier_prune_key = {
        "labels": ["resume_test_x"],
        "theta_round10": [0.25],
        "theta_round10_digits": 10,
        "energy_root": -0.25,
        "cumulative_selector_score": 1.0,
        "cumulative_selector_burden": 2.0,
        "cumulative_beam_cost": 1.0,
    }
    beam_checkpoint = {
        "status": "frontier",
        "terminated": False,
        "branch_id": 7,
        "parent_branch_id": 0,
        "depth_local": 1,
        "history_count": 1,
        "history_tail_count": 1,
        "history_tail": [dict(history_row)],
        "ansatz_depth": 1,
        "operator_labels": ["resume_test_x"],
        "energy": -0.5,
        "route_a_trust_region_state": dict(trust_state),
        "frontier_prune_key": dict(frontier_prune_key),
    }
    payload["checkpoint"] = {
        "reason": "beam_round_done",
        "checkpoint_branch_policy": "best_frontier_branch",
        "beam_enabled": True,
        "complete": False,
        "depth": 1,
        "ansatz_depth": 1,
        "branch_id": 7,
        "parent_branch_id": 0,
    }
    payload["adapt_vqe"].update(
        {
            "partial_checkpoint": True,
            "adapt_beam_enabled": True,
            "branch_id": 7,
            "parent_branch_id": 0,
            "history_checkpoint_complete": True,
            "history_count": 1,
            "history": [dict(history_row)],
            "history_tail_count": 1,
            "history_tail": [dict(history_row)],
            "logical_num_parameters": 1,
            "pool_size": 1,
            "energy": -0.5,
            "route_a_trust_region_state": dict(trust_state),
            "beam_replay_telemetry": {
                "checkpoint_branch": beam_checkpoint,
            },
        }
    )
    return ResumeScaffoldSource(
        artifact_json=source.artifact_json,
        artifact_sha256=source.artifact_sha256,
        payload=payload,
        runtime_input=source.runtime_input,
        import_summary=source.import_summary,
    )


def _verified_singleton_resume_source(tmp_path: Path) -> ResumeScaffoldSource:
    source = _resume_source()
    payload = json.loads(json.dumps(source.payload))
    work = ControllerMeasurementWorkAccumulator()
    work.record_event(
        phase="phase1",
        event_kind="candidate_scoring",
        group_keys=["x"],
        records_evaluated=1,
        records_with_group_keys=1,
        depth=1,
        candidate_count=1,
        evaluated_count=1,
        pre_shortlist_count=1,
        shortlist_size=1,
        retained_count=1,
        rejected_count=0,
    )
    history_row = {
        "depth": 1,
        "batch_size": 1,
        "branch_id": None,
        "parent_branch_id": None,
        "selected_op": "resume_test_x",
        "selected_position": 0,
        "energy_before_opt": -0.25,
        "energy_after_opt": -0.5,
        "nfev_total_after_step": 3,
        "post_admission_prune": {"accepted_count": 0},
        "route_a_trust_region_update": None,
        "controller_measurement_work_proxy": work.summary(
            include_events=False
        ),
    }
    controller_summary = controller_proxy_from_history_rows([history_row])
    controller_summary.update(
        {
            "beam_run_scope": "single_route",
            "winner_history_scope": "checkpoint_lineage_only",
            "checkpoint_depth": 1,
        }
    )
    trust_state = {
        "schema": "route_a_trust_region_state_v1",
        "radius": 0.5,
        "reference_radius": 0.5,
        "update_count": 0,
        "last_update": None,
    }
    payload["checkpoint"] = {
        "reason": "iteration_done",
        "checkpoint_branch_policy": None,
        "beam_enabled": False,
        "complete": False,
        "depth": 1,
        "ansatz_depth": 1,
        "branch_id": None,
        "parent_branch_id": None,
    }
    payload["adapt_vqe"].update(
        {
            "partial_checkpoint": True,
            "checkpoint_reason": "iteration_done",
            "adapt_beam_enabled": False,
            "branch_id": None,
            "parent_branch_id": None,
            "history_checkpoint_complete": True,
            "history_count": 1,
            "history": [dict(history_row)],
            "history_tail_count": 1,
            "history_tail": [dict(history_row)],
            "logical_num_parameters": 1,
            "pool_size": 1,
            "energy": -0.5,
            "nfev_total": 3,
            "stop_reason": None,
            "strict_replay": {
                "schema": "static_adapt_strict_state_replay_v1",
                "source": "current_checkpoint",
                "passed": True,
                "tolerance": 1.0e-10,
                "phase_aligned_l2": 0.0,
                "fidelity": 1.0,
            },
            "route_a_trust_region_state": dict(trust_state),
            "controller_measurement_work_summary": controller_summary,
            "beam_replay_telemetry": None,
            "formal_manifold_runtime_checkpoint": None,
            "formal_manifold_warm_state_checkpoint": None,
            "formal_manifold_query_closure_checkpoint": None,
            "final_full_refit": {
                "schema_version": "adapt_final_full_refit_v1",
                "attempted": False,
                "executed": False,
                "nfev": 0,
                "skipped_reason": "checkpoint_before_final_refit",
            },
        }
    )

    artifact_json = tmp_path / "current.json"
    artifact_json.write_text(json.dumps(payload), encoding="utf-8")
    controller_snapshot = asdict(
        PhaseControllerSnapshot(
            step_index=0,
            depth_local=0,
            depth_left=1,
            runway_ratio=1.0,
            early_coordinate=1.0,
            late_coordinate=0.0,
            frontier_ratio=1.0,
            snapshot_version="phase123_controller_maturity_v2",
            depth_runway_ratio=1.0,
            phase_live={"phase1": True, "phase2": True, "phase3": True},
            terminal_phase=3,
            phase_null_reasons={
                "phase1": "always_live",
                "phase2": "live",
                "phase3": "live",
            },
            phase_null_streaks={"phase2": 0, "phase3": 0},
            phase_caps={"phase1": 1, "phase2": 1, "phase3": 1},
            phase_shots={"phase1": 1, "phase2": 1, "phase3": 1},
            phase_shots_effective={
                "phase1": 1,
                "phase2": 1,
                "phase3": 1,
            },
        )
    )
    controller_evidence = {
        "depth": 1,
        "drop_policy_enabled": False,
        "drop_plateau_hits": 0,
        "stage_name": "core",
        "stage_transition_reason": "stay_core",
        "controller_snapshot_count": 1,
        "selected_feature_row_index": 0,
    }
    (tmp_path / "signed_active_prefix_checkpoint.json").write_text(
        json.dumps(
            {
                "schema": "static_adapt_signed_active_prefix_resume_sidecar_v1",
                "source_result_json": "preserved/result.json",
                "source_result_sha256": "c" * 64,
                "controller_snapshot": controller_snapshot,
                "controller_snapshot_sha256": digest_jsonable(
                    controller_snapshot
                ),
                "controller_state": {
                    "schema": "static_adapt_singleton_controller_resume_state_v1",
                    "controller_round": 1,
                    "source_max_depth": 1,
                    "phase1_residual_opened": False,
                    "phase1_stage_name": "core",
                    "source_history_row_evidence": controller_evidence,
                    "source_history_row_evidence_sha256": digest_jsonable(
                        controller_evidence
                    ),
                },
                "selection_state": {
                    "schema": "static_adapt_singleton_selection_count_resume_state_v1",
                    "controller_round": 1,
                    "pool_size": 1,
                    "seq2p_logical_mode": False,
                    "ordered_parent_pool_indices": [0],
                    "ordered_parent_pool_indices_sha256": digest_jsonable([0]),
                    "selected_feature_row_count_per_round": [1],
                    "ordered_logical_candidate_indices": [],
                    "ordered_logical_candidate_indices_sha256": digest_jsonable([]),
                },
            }
        ),
        encoding="utf-8",
    )
    ledger = EstimatorCallLedger()
    identity = EstimatorCallKey(
        projective_state_fingerprint="state:test",
        hamiltonian_fingerprint="hamiltonian:test",
        backend_fingerprint="backend:test",
        precision_contract="precision:test",
        primitive_kind="hamiltonian_expectation",
        observable_or_formula_identity="energy:test",
    )
    ledger.record_call(
        identity,
        component="N_H_outer",
        consumer_scope="energy:initial_state",
    )
    ledger.record_call(
        identity,
        component="N_H_refit",
        consumer_scope="energy:final_full_refit",
    )
    full_payload = ledger.to_payload()
    occurrence_summary = full_payload["occurrence_summary"]
    (tmp_path / "estimator_call_ledger.json").write_text(
        json.dumps(
            {
                "schema": "paper_i_estimator_call_ledger_sidecar_v2",
                "adapt_success": True,
                "adapt_error": None,
                "accounting": {
                    "schema": "paper_i_current_s_alg_accounting_v2",
                    "enabled": True,
                    "complete": True,
                    "exact_blockers": [],
                    "all_branch_search_work": {
                        "components": dict(
                            occurrence_summary["component_occurrence_counts"]
                        ),
                        "S_alg": int(
                            occurrence_summary["total_call_occurrences"]
                        ),
                    },
                },
                "ledger": full_payload,
            }
        ),
        encoding="utf-8",
    )
    return ResumeScaffoldSource(
        artifact_json=artifact_json,
        artifact_sha256="b" * 64,
        payload=payload,
        runtime_input=source.runtime_input,
        import_summary=source.import_summary,
    )


def _v4_prune_state(
    *,
    update_count: int,
    radius: float = 0.125,
    metric_damping: float = 0.0,
) -> dict[str, object]:
    return {
        "schema": "affine_deletion_fs_trust_state_v1",
        "radius": float(radius),
        "metric_damping": float(metric_damping),
        "update_count": int(update_count),
        "source": "test",
    }


def _v4_prune_payload(
    *,
    accepted: bool,
    trust_before: Mapping[str, object],
    trust_after: Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "enabled": True,
        "executed": bool(accepted),
        "accepted_count": int(bool(accepted)),
        "phase1_prune_trust_state_before": dict(trust_before),
        "phase1_prune_trust_state_after": dict(trust_after),
        "decisions": [],
    }
    if accepted:
        trial_id = "sr-v4-prune:round=2:index=0:label=resume_test_x"
        payload.update(
            {
                "selected_index": 0,
                "selected_label": "resume_test_x",
                "rollback_mutation_scope": "committed_prune",
                "phase1_prune_trial_receipt": {
                    "schema": "affine_deletion_fs_trust_same_trial_receipt_v1",
                    "trial_id": trial_id,
                    "prediction_trial_id": trial_id,
                    "realization_trial_id": trial_id,
                    "prediction_complete": True,
                    "realization_complete": True,
                    "energy_receipt_complete": True,
                    "measured_delete_refit_is_acceptance_authority": True,
                    "predicted_energy_change": 0.0,
                    "realized_energy_change": 0.0,
                    "energy_comparison_width": 1.0e-9,
                },
                "decisions": [
                    {
                        "index": 0,
                        "label": "resume_test_x",
                        "accepted": True,
                        "energy_before": -0.5,
                        "energy_after": -0.5,
                    }
                ],
            }
        )
    return payload


def _v4_active_prefix_checkpoint(
    *,
    outer_iteration: int,
    prune_payload: Mapping[str, object],
) -> dict[str, object]:
    checkpoint: dict[str, object] = {
        "schema": "paper_i_signed_active_prefix_checkpoint_v1",
        "checkpoint_kind": "post_admission_prune",
        "outer_iteration": int(outer_iteration),
        "active_ansatz_depth": 1,
        "ordered_active_operator_labels": ["resume_test_x"],
        "signed_unwrapped_runtime_parameters": [0.25],
        "signed_unwrapped_logical_parameters": [0.25],
        "post_admission_prune": json.loads(json.dumps(prune_payload)),
    }
    checkpoint["checkpoint_sha256"] = digest_jsonable(checkpoint)
    return checkpoint


def _attach_v4_profile(payload: dict[str, object]) -> None:
    contract = canonical_sr_snake_v4_contract()
    contract_sha256 = canonical_sr_snake_v4_contract_sha256()
    scope = PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
    phase12_policies = {
        "phase1_score_mode": PHASE1_SCORE_MODE_TRUST_REGION_V1,
        "phase1_energy_model": PHASE1_ENERGY_MODEL_FIRST_ORDER_FS_TRUST_V1,
        "phase2_curvature_policy": (
            PHASE2_CURVATURE_POLICY_MEASURED_REQUIRED_FAIL_CLOSED_V1
        ),
        "phase2_cheap_curvature_proxy_policy": (
            PHASE2_CHEAP_CURVATURE_PROXY_POLICY_OFF
        ),
    }
    settings = payload["settings"]
    adapt = payload["adapt_vqe"]
    checkpoint = payload["checkpoint"]
    assert isinstance(settings, dict)
    assert isinstance(adapt, dict)
    assert isinstance(checkpoint, dict)
    for block in (settings, adapt, checkpoint):
        block["sr_route_profile_request"] = SR_ROUTE_PROFILE_CANDIDATE_V4
        block["sr_route_profile_contract"] = json.loads(json.dumps(contract))
        block["sr_route_profile_contract_sha256"] = contract_sha256
        block["phase3_response_coordinate_scope"] = scope
        block.update(phase12_policies)
        block["sr_powell_coordinate_chart_policy"] = (
            SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
        )


def _v4_pruned_history(*, beam: bool) -> list[dict[str, object]]:
    state0 = _v4_prune_state(update_count=0)
    state1 = _v4_prune_state(update_count=1)
    prune0 = _v4_prune_payload(
        accepted=False,
        trust_before=state0,
        trust_after=state0,
    )
    prune1 = _v4_prune_payload(
        accepted=True,
        trust_before=state0,
        trust_after=state1,
    )
    rows = [
        {
            "depth": 1,
            "batch_size": 1,
            "branch_id": 7 if beam else None,
            "parent_branch_id": 0 if beam else None,
            "selected_op": "resume_test_x",
            "selected_position": 0,
            "energy_before_opt": -0.25,
            "energy_after_opt": -0.5,
            "nfev_total_after_step": 3,
            "post_admission_prune": prune0,
            "active_prefix_checkpoint": _v4_active_prefix_checkpoint(
                outer_iteration=1,
                prune_payload=prune0,
            ),
        },
        {
            "depth": 2,
            "batch_size": 1,
            "branch_id": 8 if beam else None,
            "parent_branch_id": 7 if beam else None,
            "selected_op": "resume_test_x",
            "selected_position": 1,
            "energy_before_opt": -0.5,
            "energy_after_opt": -0.5,
            "nfev_total_after_step": 6,
            "post_admission_prune": prune1,
            "active_prefix_checkpoint": _v4_active_prefix_checkpoint(
                outer_iteration=2,
                prune_payload=prune1,
            ),
        },
    ]
    return rows


def _v4_pruned_best_frontier_source() -> ResumeScaffoldSource:
    source = _best_frontier_resume_source()
    payload = json.loads(json.dumps(source.payload))
    history = _v4_pruned_history(beam=True)
    route_updates = [
        {
            "schema": "route_a_trust_region_update_v1",
            "policy": "fixed",
            "radius_before": 0.5,
            "radius_after": 0.5,
        }
        for _ in history
    ]
    for row, update in zip(history, route_updates):
        row["route_a_trust_region_update"] = update
    route_trust = {
        "schema": "route_a_trust_region_state_v1",
        "radius": 0.5,
        "reference_radius": 0.5,
        "update_count": 2,
        "last_update": dict(route_updates[-1]),
    }
    payload["checkpoint"].update(
        {
            "depth": 2,
            "ansatz_depth": 1,
            "branch_id": 8,
            "parent_branch_id": 7,
        }
    )
    payload["adapt_vqe"].update(
        {
            "branch_id": 8,
            "parent_branch_id": 7,
            "history_count": 2,
            "history": history,
            "history_tail_count": 2,
            "history_tail": history,
            "route_a_trust_region_state": route_trust,
        }
    )
    beam_checkpoint = payload["adapt_vqe"]["beam_replay_telemetry"][
        "checkpoint_branch"
    ]
    beam_checkpoint.update(
        {
            "branch_id": 8,
            "parent_branch_id": 7,
            "depth_local": 2,
            "history_count": 2,
            "history_tail_count": 2,
            "history_tail": history,
            "ansatz_depth": 1,
            "route_a_trust_region_state": route_trust,
        }
    )
    _attach_v4_profile(payload)
    return ResumeScaffoldSource(
        artifact_json=source.artifact_json,
        artifact_sha256=source.artifact_sha256,
        payload=payload,
        runtime_input=source.runtime_input,
        import_summary=source.import_summary,
    )


def _v4_pruned_singleton_source(tmp_path: Path) -> ResumeScaffoldSource:
    source = _verified_singleton_resume_source(tmp_path)
    payload = json.loads(json.dumps(source.payload))
    history = _v4_pruned_history(beam=False)
    work = history[0].get("controller_measurement_work_proxy")
    if not isinstance(work, Mapping):
        work = source.payload["adapt_vqe"]["history"][0][
            "controller_measurement_work_proxy"
        ]
    for row in history:
        row["controller_measurement_work_proxy"] = dict(work)
    controller_summary = controller_proxy_from_history_rows(history)
    controller_summary.update(
        {
            "beam_run_scope": "single_route",
            "winner_history_scope": "checkpoint_lineage_only",
            "checkpoint_depth": 2,
        }
    )
    payload["checkpoint"].update({"depth": 2, "ansatz_depth": 1})
    payload["adapt_vqe"].update(
        {
            "history_count": 2,
            "history": history,
            "history_tail_count": 2,
            "history_tail": history,
            "nfev_total": 6,
            "controller_measurement_work_summary": controller_summary,
        }
    )
    _attach_v4_profile(payload)
    source.artifact_json.write_text(json.dumps(payload), encoding="utf-8")

    sidecar_path = tmp_path / "signed_active_prefix_checkpoint.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["controller_snapshot"].update(
        {"step_index": 1, "depth_local": 1, "depth_left": 1}
    )
    sidecar["controller_snapshot_sha256"] = digest_jsonable(
        sidecar["controller_snapshot"]
    )
    sidecar["controller_state"].update(
        {
            "controller_round": 2,
            "source_max_depth": 2,
            "source_history_row_evidence": {
                "depth": 2,
                "drop_policy_enabled": False,
                "drop_plateau_hits": 0,
                "stage_name": "core",
                "stage_transition_reason": "stay_core",
                "controller_snapshot_count": 1,
                "selected_feature_row_index": 0,
            },
        }
    )
    sidecar["controller_state"]["source_history_row_evidence_sha256"] = (
        digest_jsonable(
            sidecar["controller_state"]["source_history_row_evidence"]
        )
    )
    sidecar["selection_state"].update(
        {
            "controller_round": 2,
            "ordered_parent_pool_indices": [0, 0],
            "ordered_parent_pool_indices_sha256": digest_jsonable([0, 0]),
            "selected_feature_row_count_per_round": [1, 1],
        }
    )
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return ResumeScaffoldSource(
        artifact_json=source.artifact_json,
        artifact_sha256=source.artifact_sha256,
        payload=payload,
        runtime_input=source.runtime_input,
        import_summary=source.import_summary,
    )


def _v4_unpruned_singleton_source(tmp_path: Path) -> ResumeScaffoldSource:
    """Build a real beam-disabled v4 schema with a held prune trust state."""

    source = _verified_singleton_resume_source(tmp_path)
    payload = json.loads(json.dumps(source.payload))
    state = _v4_prune_state(update_count=0)
    for history_field in ("history", "history_tail"):
        history_rows = payload["adapt_vqe"][history_field]
        history_rows[-1]["post_admission_prune"].update(
            {
                "enabled": True,
                "executed": False,
                "accepted_count": 0,
                "phase1_prune_trust_state_before": dict(state),
                "phase1_prune_trust_state_after": dict(state),
                "phase1_prune_trust_update": {
                    "schema": "affine_deletion_fs_trust_state_update_v1",
                    "status": "held",
                    "reason": "all_affine_deletion_models_infeasible",
                    "update_count_before": 0,
                    "update_count_after": 0,
                    "classical_quantum_query_charge": 0,
                },
                "phase1_prune_no_feasible_model": {
                    "schema": "sr_v4_no_feasible_affine_deletion_models_v1",
                    "status": "skipped_no_feasible_affine_deletion_models",
                    "legacy_nomination_fallback_used": False,
                    "exact_delete_refit_trial_count": 0,
                },
            }
        )
    _attach_v4_profile(payload)
    source.artifact_json.write_text(json.dumps(payload), encoding="utf-8")
    return ResumeScaffoldSource(
        artifact_json=source.artifact_json,
        artifact_sha256=source.artifact_sha256,
        payload=payload,
        runtime_input=source.runtime_input,
        import_summary=source.import_summary,
    )




def _with_sr_powell_chart(
    source: ResumeScaffoldSource,
    policy: str = SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1,
) -> ResumeScaffoldSource:
    payload = json.loads(json.dumps(source.payload))
    payload["settings"]["route_family"] = "singleton_response_snake"
    payload["settings"]["sr_powell_coordinate_chart_policy"] = policy
    payload["adapt_vqe"]["static_route_identity"] = {
        "route_family": "singleton_response_snake",
        "route_profile": "supported_whitened_adaptive_trust_v1",
        "powell_coordinate_chart_policy": policy,
    }
    checkpoint = payload.get("checkpoint")
    if isinstance(checkpoint, dict):
        checkpoint["optimizer_coordinate_chart"] = {
            "powell_coordinate_chart_policy": policy,
        }
    return ResumeScaffoldSource(
        artifact_json=source.artifact_json,
        artifact_sha256=source.artifact_sha256,
        payload=payload,
        runtime_input=source.runtime_input,
        import_summary=source.import_summary,
    )


def _with_sr_phase2_phase3_expanded_ablation(
    source: ResumeScaffoldSource,
    *,
    include_conformance: bool = True,
    conformance: str = (
        SR_ROUTE_PROFILE_CONFORMANCE_UNPROMOTED_EXPLICIT_ABLATION
    ),
    include_scope: bool = True,
) -> ResumeScaffoldSource:
    resolution = resolve_sr_powell_route_instance(
        "disabled",
        coordinate_solve_scope=(
            SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1
        ),
        requested_policy=(
            SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
        ),
    )
    resolution["route_profile_conformance"] = str(conformance)
    seeded = _with_sr_powell_chart(
        source,
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1,
    )
    payload = json.loads(json.dumps(seeded.payload))
    payload["settings"]["route_profile"] = (
        SR_ROUTE_PROFILE_PHASE2_PHASE3_WHITENED
    )
    payload["adapt_vqe"]["static_route_identity"]["route_profile"] = (
        SR_ROUTE_PROFILE_PHASE2_PHASE3_WHITENED
    )
    if include_conformance:
        payload["settings"]["route_profile_conformance"] = str(conformance)
        payload["adapt_vqe"]["static_route_identity"][
            "route_profile_conformance"
        ] = str(conformance)
    if include_scope:
        payload["settings"]["historical_singleton_coordinate_solve_scope"] = (
            SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1
        )
        payload["adapt_vqe"]["static_route_identity"][
            "coordinate_solve_scope"
        ] = SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1
    payload["settings"]["sr_powell_route_instance"] = dict(resolution)
    if not include_conformance:
        payload["settings"]["sr_powell_route_instance"].pop(
            "route_profile_conformance", None
        )
    if not include_scope:
        payload["settings"]["sr_powell_route_instance"].pop(
            "coordinate_solve_scope", None
        )
    return ResumeScaffoldSource(
        artifact_json=seeded.artifact_json,
        artifact_sha256=seeded.artifact_sha256,
        payload=payload,
        runtime_input=seeded.runtime_input,
        import_summary=seeded.import_summary,
    )


# Structural note: the argparse resume-flag surface (mutual exclusion, secret-flag
# guard, batch-order CLI aliases) was retired with the adapt_pipeline argparse entrypoint; the
# secret-material guard itself is covered CLI-free by
# test_resume_secret_scan_and_stable_digests below.


def test_resume_secret_scan_and_stable_digests() -> None:
    assert digest_jsonable({"b": 2, "a": [1, 2]}) == digest_jsonable({"a": [1, 2], "b": 2})
    assert_no_secret_material({"safe": "FakeMarrakesh"}, context="unit")
    assert_no_secret_material(build_credential_audit(), context="unit_audit")
    with pytest.raises(ValueError):
        assert_no_secret_material({"bad": "api_key=abc123"}, context="unit")


def test_strict_best_frontier_checkpoint_preserves_complete_cleaned_history() -> None:
    checkpoint = extract_best_frontier_resume_checkpoint(
        _best_frontier_resume_source()
    )

    assert checkpoint.controller_round == 1
    assert checkpoint.ansatz_depth == 1
    assert checkpoint.branch_id == 7
    assert checkpoint.parent_branch_id == 0
    assert checkpoint.operator_labels == ("resume_test_x",)
    assert checkpoint.theta_runtime == (0.25,)
    assert checkpoint.theta_logical == (0.25,)
    assert len(checkpoint.history) == 1
    assert "structural_rollback" not in checkpoint.history[0]
    assert checkpoint.history[0]["preserved_nested_telemetry"] == {
        "sentinel": [1, 2, 3]
    }
    assert checkpoint.route_a_trust_region_state["update_count"] == 1
    assert checkpoint.frontier_prune_key["cumulative_beam_cost"] == 1.0
    assert checkpoint.validation["lineage_scope"] == (
        "preserved_best_frontier_branch_only"
    )
    assert checkpoint.validation["discarded_frontier_reconstructed"] is False


def test_best_frontier_resume_preserves_authenticated_seed_prefix() -> None:
    source = _best_frontier_resume_source()
    payload = json.loads(json.dumps(source.payload))
    seed_term = _one_term("seed_x")
    admitted_term = _one_term("resume_test_x")
    terms = (seed_term, admitted_term)
    layout = build_parameter_layout(
        terms,
        ignore_identity=True,
        coefficient_tolerance=1e-12,
        sort_terms=True,
    )
    theta = np.array([0.1, 0.25], dtype=float)
    psi_initial = CompiledAnsatzExecutor(
        terms,
        parameterization_layout=layout,
        parameterization_mode="per_pauli_term",
    ).prepare_state(theta, source.runtime_input.psi_ref)

    payload["checkpoint"]["ansatz_depth"] = 2
    adapt = payload["adapt_vqe"]
    adapt.update(
        {
            "operators": [term.label for term in terms],
            "optimal_point": theta.tolist(),
            "logical_optimal_point": theta.tolist(),
            "parameterization": serialize_layout(layout),
            "ansatz_depth": 2,
            "num_parameters": 2,
            "logical_num_parameters": 2,
        }
    )
    adapt["history"][0]["selected_position"] = 1
    adapt["history_tail"][0]["selected_position"] = 1
    checkpoint_branch = adapt["beam_replay_telemetry"][
        "checkpoint_branch"
    ]
    checkpoint_branch.update(
        {
            "ansatz_depth": 2,
            "operator_labels": [term.label for term in terms],
        }
    )
    checkpoint_branch["history_tail"][0]["selected_position"] = 1
    checkpoint_branch["frontier_prune_key"].update(
        {
            "labels": [term.label for term in terms],
            "theta_round10": theta.tolist(),
        }
    )
    payload["initial_state"] = build_statevector_manifest(
        psi_state=psi_initial,
        source="adapt_vqe",
        handoff_state_kind="prepared_state",
        amplitude_cutoff=1e-12,
    )
    seeded_source = ResumeScaffoldSource(
        artifact_json=source.artifact_json,
        artifact_sha256=source.artifact_sha256,
        payload=payload,
        runtime_input=ScaffoldRuntimeInput(
            resolved_problem=source.runtime_input.resolved_problem,
            psi_ref=source.runtime_input.psi_ref,
            psi_initial=psi_initial,
            base_layout=layout,
            theta_runtime=theta,
            theta_logical=theta.copy(),
            structure_locked=False,
            exact_energy=None,
            selected_terms=terms,
        ),
        import_summary=source.import_summary,
    )

    checkpoint = extract_best_frontier_resume_checkpoint(seeded_source)
    assert checkpoint.ansatz_depth == 2
    assert checkpoint.controller_round == 1
    assert checkpoint.operator_labels == ("seed_x", "resume_test_x")
    assert checkpoint.validation["preserved_seed_prefix_depth"] == 1


def test_best_frontier_resume_accepts_typed_history_only_overlap_receipt() -> None:
    source = _best_frontier_resume_source()
    payload = json.loads(json.dumps(source.payload))
    overlap_receipt = {
        "schema": "adaptive_trust_overlap_query_accounting_v1",
        "enabled": True,
        "status": "complete",
        "component": "N_metric",
        "formal_query_category": "N_cross",
        "primitive_id": "overlap-primitive-1",
    }
    for history_row in (
        payload["adapt_vqe"]["history"][-1],
        payload["adapt_vqe"]["history_tail"][-1],
        payload["adapt_vqe"]["beam_replay_telemetry"]["checkpoint_branch"][
            "history_tail"
        ][-1],
    ):
        history_row["route_a_trust_region_update"][
            "endpoint_overlap_query_accounting"
        ] = dict(overlap_receipt)
    augmented = ResumeScaffoldSource(
        artifact_json=source.artifact_json,
        artifact_sha256=source.artifact_sha256,
        payload=payload,
        runtime_input=source.runtime_input,
        import_summary=source.import_summary,
    )

    checkpoint = extract_best_frontier_resume_checkpoint(augmented)
    assert checkpoint.controller_round == 1


def test_best_frontier_resume_rejects_incomplete_history_overlap_receipt() -> None:
    source = _best_frontier_resume_source()
    payload = json.loads(json.dumps(source.payload))
    payload["adapt_vqe"]["history"][-1][
        "route_a_trust_region_update"
    ]["endpoint_overlap_query_accounting"] = {
        "schema": "adaptive_trust_overlap_query_accounting_v1",
        "enabled": True,
        "status": "incomplete",
        "component": "N_metric",
        "formal_query_category": "N_cross",
        "primitive_id": "overlap-primitive-1",
    }
    augmented = ResumeScaffoldSource(
        artifact_json=source.artifact_json,
        artifact_sha256=source.artifact_sha256,
        payload=payload,
        runtime_input=source.runtime_input,
        import_summary=source.import_summary,
    )

    with pytest.raises(ValueError, match="complete typed receipt"):
        extract_best_frontier_resume_checkpoint(augmented)


def test_best_frontier_resume_accepts_authenticated_v4_pruned_lineage() -> None:
    checkpoint = extract_best_frontier_resume_checkpoint(
        _v4_pruned_best_frontier_source(),
        expected_sr_route_profile_request=SR_ROUTE_PROFILE_CANDIDATE_V4,
        expected_sr_route_profile_contract=canonical_sr_snake_v4_contract(),
        expected_sr_route_profile_contract_sha256=(
            canonical_sr_snake_v4_contract_sha256()
        ),
    )

    assert checkpoint.controller_round == 2
    assert checkpoint.ansatz_depth == 1
    assert checkpoint.operator_labels == ("resume_test_x",)
    lineage = checkpoint.validation["authenticated_v4_pruned_lineage"]
    assert lineage["accepted_prune_count"] == 1
    assert lineage["restored_prune_trust_state"] == {
        "schema": "affine_deletion_fs_trust_state_v1",
        "radius": 0.125,
        "metric_damping": 0.0,
        "update_count": 1,
    }


def test_best_frontier_resume_rejects_tampered_v4_prune_prefix() -> None:
    source = _v4_pruned_best_frontier_source()
    payload = json.loads(json.dumps(source.payload))
    payload["adapt_vqe"]["history"][1]["active_prefix_checkpoint"][
        "ordered_active_operator_labels"
    ] = ["tampered"]
    corrupted = ResumeScaffoldSource(
        artifact_json=source.artifact_json,
        artifact_sha256=source.artifact_sha256,
        payload=payload,
        runtime_input=source.runtime_input,
        import_summary=source.import_summary,
    )

    with pytest.raises(ValueError, match="active-prefix checksum failed"):
        extract_best_frontier_resume_checkpoint(
            corrupted,
            expected_sr_route_profile_request=SR_ROUTE_PROFILE_CANDIDATE_V4,
            expected_sr_route_profile_contract=canonical_sr_snake_v4_contract(),
            expected_sr_route_profile_contract_sha256=(
                canonical_sr_snake_v4_contract_sha256()
            ),
        )


def test_verified_singleton_checkpoint_restores_complete_state_and_ledger(
    tmp_path: Path,
) -> None:
    checkpoint = extract_verified_singleton_resume_checkpoint(
        _verified_singleton_resume_source(tmp_path)
    )

    assert checkpoint.controller_round == 1
    assert checkpoint.ansatz_depth == 1
    assert checkpoint.branch_id is None
    assert checkpoint.parent_branch_id is None
    assert checkpoint.operator_labels == ("resume_test_x",)
    assert checkpoint.route_a_trust_region_state["update_count"] == 0
    assert checkpoint.phase1_residual_opened is False
    assert checkpoint.phase1_stage_name == "core"
    assert checkpoint.maturity_controller_snapshot["phase_live"] == {
        "phase1": True,
        "phase2": True,
        "phase3": True,
    }
    assert checkpoint.selection_parent_pool_size == 1
    assert checkpoint.selected_parent_pool_indices == (0,)
    assert checkpoint.selected_logical_candidate_indices == ()
    assert checkpoint.validation["strict_replay_passed"] is True
    assert checkpoint.validation["controller_measurement_work_closed"] is True
    assert checkpoint.validation["lineage_scope"] == (
        "complete_singleton_branch_only"
    )
    assert checkpoint.validation["discarded_frontier_reconstructed"] is False
    provenance = checkpoint.estimator_call_ledger_provenance
    assert provenance["restored_prefix_occurrence_count"] == 1
    assert provenance["excluded_terminal_occurrence_count"] == 1
    assert checkpoint.estimator_call_ledger_payload["summary"]["S_unique"] == 1


def test_verified_singleton_checkpoint_prefers_hash_linked_round_ledger(
    tmp_path: Path,
) -> None:
    source = _verified_singleton_resume_source(tmp_path)
    ledger = EstimatorCallLedger()
    identity = EstimatorCallKey(
        projective_state_fingerprint="state:round-checkpoint",
        hamiltonian_fingerprint="hamiltonian:test",
        backend_fingerprint="backend:test",
        precision_contract="precision:test",
        primitive_kind="hamiltonian_expectation",
        observable_or_formula_identity="energy:accepted-round",
    )
    ledger.record_call(
        identity,
        component="N_H_outer",
        consumer_scope="energy:accepted_round",
    )
    ledger_payload = ledger.to_payload()
    sidecar_payload = {
        "schema": "paper_i_estimator_call_ledger_checkpoint_sidecar_v1",
        "generated_utc": "2026-07-18T00:00:00+00:00",
        "checkpoint": {
            "reason": "iteration_done",
            "depth": 1,
            "branch_id": None,
            "parent_branch_id": None,
            "current_round_finalized": True,
        },
        "ledger": ledger_payload,
        "ledger_fingerprint": ledger_payload["ledger_fingerprint"],
        "unique_primitive_count": 1,
        "raw_occurrence_count": 1,
        "S_alg": 1,
        "no_credentials_serialized": True,
    }
    sidecar_bytes = (
        json.dumps(sidecar_payload, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    sidecar_sha256 = hashlib.sha256(sidecar_bytes).hexdigest()
    sidecar_path = tmp_path / (
        "current.estimator_call_ledger_checkpoint."
        f"{sidecar_sha256[:16]}.json"
    )
    sidecar_path.write_bytes(sidecar_bytes)

    payload = json.loads(json.dumps(source.payload))
    payload["adapt_vqe"]["estimator_call_ledger_checkpoint"] = {
        "schema": "paper_i_estimator_call_ledger_checkpoint_pointer_v1",
        "enabled": True,
        "status": "complete",
        "path": sidecar_path.name,
        "sha256": sidecar_sha256,
        "ledger_schema": ledger_payload["schema"],
        "checkpoint_reason": "iteration_done",
        "ledger_fingerprint": ledger_payload["ledger_fingerprint"],
        "unique_primitive_count": 1,
        "raw_occurrence_count": 1,
        "S_alg": 1,
        "checkpoint_depth": 1,
        "current_round_finalized": True,
    }
    checkpoint = extract_verified_singleton_resume_checkpoint(
        ResumeScaffoldSource(
            artifact_json=source.artifact_json,
            artifact_sha256=source.artifact_sha256,
            payload=payload,
            runtime_input=source.runtime_input,
            import_summary=source.import_summary,
        )
    )

    assert checkpoint.estimator_call_ledger_payload["ledger_fingerprint"] == (
        ledger_payload["ledger_fingerprint"]
    )
    assert checkpoint.estimator_call_ledger_provenance["source_kind"] == (
        "completed_round_checkpoint_sidecar"
    )
    assert checkpoint.estimator_call_ledger_provenance[
        "excluded_terminal_occurrence_count"
    ] == 0


def test_verified_singleton_round_ledger_fails_closed_on_hash_mismatch(
    tmp_path: Path,
) -> None:
    source = _verified_singleton_resume_source(tmp_path)
    payload = json.loads(json.dumps(source.payload))
    sidecar_path = tmp_path / "current.estimator_call_ledger_checkpoint.bad.json"
    sidecar_path.write_text("{}\n", encoding="utf-8")
    payload["adapt_vqe"]["estimator_call_ledger_checkpoint"] = {
        "schema": "paper_i_estimator_call_ledger_checkpoint_pointer_v1",
        "enabled": True,
        "status": "complete",
        "path": sidecar_path.name,
        "sha256": "0" * 64,
        "ledger_fingerprint": "1" * 64,
        "unique_primitive_count": 0,
        "raw_occurrence_count": 0,
        "S_alg": 0,
        "checkpoint_depth": 1,
        "current_round_finalized": True,
    }

    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        extract_verified_singleton_resume_checkpoint(
            ResumeScaffoldSource(
                artifact_json=source.artifact_json,
                artifact_sha256=source.artifact_sha256,
                payload=payload,
                runtime_input=source.runtime_input,
                import_summary=source.import_summary,
            )
        )


def test_verified_singleton_resume_accepts_authenticated_v4_pruned_lineage(
    tmp_path: Path,
) -> None:
    checkpoint = extract_verified_singleton_resume_checkpoint(
        _v4_pruned_singleton_source(tmp_path),
        expected_sr_route_profile_request=SR_ROUTE_PROFILE_CANDIDATE_V4,
        expected_sr_route_profile_contract=canonical_sr_snake_v4_contract(),
        expected_sr_route_profile_contract_sha256=(
            canonical_sr_snake_v4_contract_sha256()
        ),
    )

    assert checkpoint.controller_round == 2
    assert checkpoint.ansatz_depth == 1
    assert checkpoint.operator_labels == ("resume_test_x",)
    assert checkpoint.selected_parent_pool_indices == (0, 0)
    lineage = checkpoint.validation["authenticated_v4_pruned_lineage"]
    assert lineage["accepted_prune_count"] == 1
    assert lineage["restored_prune_trust_state"]["radius"] == pytest.approx(
        0.125
    )
    assert lineage["restored_prune_trust_state"][
        "metric_damping"
    ] == pytest.approx(0.0)


def test_verified_singleton_v4_resume_round_trips_latest_prune_trust_state(
    tmp_path: Path,
) -> None:
    checkpoint = extract_verified_singleton_resume_checkpoint(
        _v4_unpruned_singleton_source(tmp_path),
        expected_sr_route_profile_request=SR_ROUTE_PROFILE_CANDIDATE_V4,
        expected_sr_route_profile_contract=canonical_sr_snake_v4_contract(),
        expected_sr_route_profile_contract_sha256=(
            canonical_sr_snake_v4_contract_sha256()
        ),
    )

    assert checkpoint.validation["latest_v4_prune_trust_state"] == {
        "schema": "affine_deletion_fs_trust_state_v1",
        "radius": 0.125,
        "metric_damping": 0.0,
        "update_count": 0,
    }
    assert checkpoint.history[-1]["post_admission_prune"][
        "phase1_prune_no_feasible_model"
    ]["legacy_nomination_fallback_used"] is False


@pytest.mark.parametrize(
    "missing_field",
    ["radius", "metric_damping", "update_count"],
)
def test_verified_singleton_v4_resume_rejects_missing_latest_prune_trust_field(
    tmp_path: Path,
    missing_field: str,
) -> None:
    source = _v4_unpruned_singleton_source(tmp_path)
    payload = json.loads(json.dumps(source.payload))
    for history_field in ("history", "history_tail"):
        del payload["adapt_vqe"][history_field][-1]["post_admission_prune"][
            "phase1_prune_trust_state_after"
        ][missing_field]
    corrupted = ResumeScaffoldSource(
        artifact_json=source.artifact_json,
        artifact_sha256=source.artifact_sha256,
        payload=payload,
        runtime_input=source.runtime_input,
        import_summary=source.import_summary,
    )

    with pytest.raises(ValueError, match="radius/mu/update_count"):
        extract_verified_singleton_resume_checkpoint(
            corrupted,
            expected_sr_route_profile_request=SR_ROUTE_PROFILE_CANDIDATE_V4,
            expected_sr_route_profile_contract=canonical_sr_snake_v4_contract(),
            expected_sr_route_profile_contract_sha256=(
                canonical_sr_snake_v4_contract_sha256()
            ),
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("adapt_vqe", "history_checkpoint_complete"),
            False,
            "history_checkpoint_complete",
        ),
        (
            ("adapt_vqe", "operators", 0),
            "wrong_operator",
            "history operators",
        ),
        (
            ("adapt_vqe", "strict_replay", "passed"),
            False,
            "strict replay did not pass",
        ),
        (
            ("adapt_vqe", "strict_replay", "phase_aligned_l2"),
            1.0e-4,
            "strict-replay receipt",
        ),
        (
            ("adapt_vqe", "history", 0, "post_admission_prune", "accepted_count"),
            1,
            "accepted prune deletion",
        ),
    ],
)
def test_verified_singleton_checkpoint_fails_closed_on_drift(
    tmp_path: Path,
    path: tuple[str | int, ...],
    value: object,
    message: str,
) -> None:
    source = _verified_singleton_resume_source(tmp_path)
    payload = json.loads(json.dumps(source.payload))
    cursor: object = payload
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]
    if path[:2] == ("adapt_vqe", "history") and len(path) > 3:
        tail_cursor: object = payload["adapt_vqe"]["history_tail"]
        for key in path[2:-1]:
            tail_cursor = tail_cursor[key]  # type: ignore[index]
        tail_cursor[path[-1]] = value  # type: ignore[index]
    corrupted = ResumeScaffoldSource(
        artifact_json=source.artifact_json,
        artifact_sha256=source.artifact_sha256,
        payload=payload,
        runtime_input=source.runtime_input,
        import_summary=source.import_summary,
    )

    with pytest.raises(ValueError, match=message):
        extract_verified_singleton_resume_checkpoint(corrupted)


def test_verified_singleton_checkpoint_requires_state_keyed_ledger_sidecar(
    tmp_path: Path,
) -> None:
    source = _verified_singleton_resume_source(tmp_path)
    (tmp_path / "estimator_call_ledger.json").unlink()

    with pytest.raises(ValueError, match="requires sibling estimator_call_ledger"):
        extract_verified_singleton_resume_checkpoint(source)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda payload: payload["controller_state"].update(
                {"phase1_residual_opened": True}
            ),
            "controller_state is inconsistent",
        ),
        (
            lambda payload: payload["selection_state"].update(
                {
                    "ordered_parent_pool_indices": [1],
                    "ordered_parent_pool_indices_sha256": digest_jsonable([1]),
                }
            ),
            "incomplete/out of range",
        ),
        (
            lambda payload: payload["selection_state"].update(
                {
                    "ordered_logical_candidate_indices": [0],
                    "ordered_logical_candidate_indices_sha256": digest_jsonable([0]),
                }
            ),
            "authenticated empty logical sequence",
        ),
    ],
)
def test_verified_singleton_controller_and_selection_state_fail_closed(
    tmp_path: Path,
    mutator: object,
    message: str,
) -> None:
    source = _verified_singleton_resume_source(tmp_path)
    sidecar_path = tmp_path / "signed_active_prefix_checkpoint.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    mutator(sidecar)  # type: ignore[operator]
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        extract_verified_singleton_resume_checkpoint(source)








def test_sr_resume_summary_and_best_frontier_preserve_explicit_powell_chart() -> None:
    source = _with_sr_powell_chart(_best_frontier_resume_source())
    summary = build_resume_import_summary(source)
    assert summary["powell_coordinate_chart_policy"] == (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    )
    assert summary["powell_coordinate_chart_policy_validation"]["inferred"] is False

    checkpoint = extract_best_frontier_resume_checkpoint(
        source,
        expected_powell_coordinate_chart_policy=(
            SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
        ),
    )
    assert checkpoint.powell_coordinate_chart_policy == (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    )
    assert checkpoint.validation["powell_coordinate_chart_policy"] == (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    )


def test_sr_phase2_phase3_expanded_ablation_resume_requires_and_preserves_marker() -> None:
    source = _with_sr_phase2_phase3_expanded_ablation(
        _best_frontier_resume_source()
    )
    validation = validate_resume_powell_coordinate_chart_policy(
        source.payload,
        expected_policy=(
            SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
        ),
        expected_route_profile_conformance=(
            SR_ROUTE_PROFILE_CONFORMANCE_UNPROMOTED_EXPLICIT_ABLATION
        ),
    )
    assert validation["explicit_unpromoted_ablation"] is True
    assert validation["route_profile_conformance"] == (
        SR_ROUTE_PROFILE_CONFORMANCE_UNPROMOTED_EXPLICIT_ABLATION
    )
    assert validation["coordinate_solve_scope"] == (
        SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1
    )

    summary = build_resume_import_summary(source)
    assert summary["route_profile_conformance"] == (
        SR_ROUTE_PROFILE_CONFORMANCE_UNPROMOTED_EXPLICIT_ABLATION
    )
    checkpoint = extract_best_frontier_resume_checkpoint(
        source,
        expected_powell_coordinate_chart_policy=(
            SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
        ),
        expected_route_profile_conformance=(
            SR_ROUTE_PROFILE_CONFORMANCE_UNPROMOTED_EXPLICIT_ABLATION
        ),
    )
    assert checkpoint.route_profile_conformance == (
        SR_ROUTE_PROFILE_CONFORMANCE_UNPROMOTED_EXPLICIT_ABLATION
    )
    assert checkpoint.validation["route_profile_conformance"] == (
        SR_ROUTE_PROFILE_CONFORMANCE_UNPROMOTED_EXPLICIT_ABLATION
    )


@pytest.mark.parametrize(
    ("include_conformance", "conformance", "include_scope", "match"),
    [
        (
            False,
            SR_ROUTE_PROFILE_CONFORMANCE_UNPROMOTED_EXPLICIT_ABLATION,
            True,
            "requires the serialized unpromoted",
        ),
        (
            True,
            SR_ROUTE_PROFILE_CONFORMANCE_REGISTERED,
            True,
            "requires the serialized unpromoted",
        ),
        (
            True,
            SR_ROUTE_PROFILE_CONFORMANCE_UNPROMOTED_EXPLICIT_ABLATION,
            False,
            "refuses to infer",
        ),
    ],
)
def test_sr_phase2_phase3_expanded_ablation_resume_fails_closed_without_exact_marker(
    include_conformance: bool,
    conformance: str,
    include_scope: bool,
    match: str,
) -> None:
    source = _with_sr_phase2_phase3_expanded_ablation(
        _resume_source(),
        include_conformance=include_conformance,
        conformance=conformance,
        include_scope=include_scope,
    )
    with pytest.raises(ValueError, match=match):
        validate_resume_powell_coordinate_chart_policy(source.payload)


def test_sr_resume_powell_chart_policy_fails_closed_on_missing_unknown_and_conflict() -> None:
    missing = _resume_source().payload
    missing["settings"]["route_family"] = "singleton_response_snake"
    with pytest.raises(ValueError, match="missing the explicit Powell"):
        validate_resume_powell_coordinate_chart_policy(missing)

    profile_only = json.loads(json.dumps(_resume_source().payload))
    profile_only["settings"]["route_profile"] = (
        "supported_whitened_adaptive_trust_v1"
    )
    with pytest.raises(ValueError, match="missing the explicit Powell"):
        validate_resume_powell_coordinate_chart_policy(profile_only)

    unknown = json.loads(json.dumps(missing))
    unknown["settings"]["sr_powell_coordinate_chart_policy"] = "implicit_latest"
    with pytest.raises(ValueError, match="unknown Powell"):
        validate_resume_powell_coordinate_chart_policy(unknown)

    conflict_source = _with_sr_powell_chart(_resume_source())
    conflict = json.loads(json.dumps(conflict_source.payload))
    conflict["adapt_vqe"]["optimizer_coordinate_chart"] = {
        "powell_coordinate_chart_policy": (
            SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
        )
    }
    with pytest.raises(ValueError, match="conflicting Powell"):
        validate_resume_powell_coordinate_chart_policy(conflict)

    profile_mismatch = json.loads(json.dumps(_resume_source().payload))
    profile_mismatch["settings"]["route_family"] = "singleton_response_snake"
    profile_mismatch["settings"]["route_profile"] = (
        "supported_whitened_adaptive_trust_v1"
    )
    profile_mismatch["settings"]["sr_powell_coordinate_chart_policy"] = (
        SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
    )
    profile_mismatch["adapt_vqe"]["static_route_identity"] = {
        "route_family": "singleton_response_snake",
        "route_profile": "supported_whitened_adaptive_trust_v1",
        "powell_coordinate_chart_policy": (
            SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
        ),
    }
    with pytest.raises(ValueError, match="route-profile/Powell-chart mismatch"):
        validate_resume_powell_coordinate_chart_policy(profile_mismatch)


def test_sr_resume_powell_chart_policy_rejects_current_route_mismatch() -> None:
    source = _with_sr_powell_chart(_resume_source())
    with pytest.raises(ValueError, match="policy mismatch"):
        validate_static_hh_resume_source(
            source,
            expected_powell_coordinate_chart_policy=(
                SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
            ),
        )


def test_v3_resume_response_scope_fails_closed_on_missing_or_legacy() -> None:
    missing = {
        "settings": {
            "sr_route_profile_request": SR_ROUTE_PROFILE_CONVENTIONAL_V3,
        }
    }
    with pytest.raises(ValueError, match="missing phase3_response_coordinate_scope"):
        validate_resume_phase3_response_coordinate_scope(missing)

    legacy = json.loads(json.dumps(missing))
    legacy["settings"]["phase3_response_coordinate_scope"] = (
        PHASE3_RESPONSE_COORDINATE_SCOPE_LEGACY_REOPT_COUPLED_V1
    )
    with pytest.raises(ValueError, match="requires full_active_plus_singleton_v1"):
        validate_resume_phase3_response_coordinate_scope(legacy)


def test_v4_resume_rejects_legacy_phase1_score_mode() -> None:
    payload = json.loads(json.dumps(_resume_source().payload))
    payload.setdefault("checkpoint", {})
    _attach_v4_profile(payload)
    payload["adapt_vqe"]["phase1_score_mode"] = "legacy_simple_v1"

    with pytest.raises(ValueError, match="conflicting phase1_score_mode"):
        validate_resume_phase12_energy_model_policies(payload)


def test_v3_resume_response_scope_round_trip_and_historical_v2_resolution() -> None:
    v3 = {
        "settings": {
            "sr_route_profile_request": SR_ROUTE_PROFILE_CONVENTIONAL_V3,
            "phase3_response_coordinate_scope": (
                PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
            ),
        }
    }
    validation = validate_resume_phase3_response_coordinate_scope(v3)
    assert validation["status"] == "pass"
    assert validation["resolved_scope"] == (
        PHASE3_RESPONSE_COORDINATE_SCOPE_FULL_ACTIVE_PLUS_SINGLETON_V1
    )
    assert validation["resolution_source"] == "serialized_artifact"
    assert validation["inferred_from_window_or_refit_schedule"] is False

    historical = {
        "settings": {
            "sr_route_profile_request": SR_ROUTE_PROFILE_CONVENTIONAL_V2,
        }
    }
    validation = validate_resume_phase3_response_coordinate_scope(historical)
    assert validation["resolved_scope"] == (
        PHASE3_RESPONSE_COORDINATE_SCOPE_LEGACY_REOPT_COUPLED_V1
    )
    assert validation["resolution_source"] == (
        "versioned_historical_profile_contract"
    )


def test_source_locked_resume_without_powell_chart_fails_closed() -> None:
    payload = _resume_source().payload
    payload["settings"]["phase3_source_lock_preferred_sequence"] = "op_a"
    with pytest.raises(ValueError, match="SR/source-locked"):
        validate_resume_powell_coordinate_chart_policy(payload)


@pytest.mark.parametrize(
    "checkpoint_location",
    [
        "adapt_vqe",
        "top_level",
        "checkpoint",
    ],
)
@pytest.mark.parametrize(
    "resume_entrypoint",
    [
        "validate_static_hh_resume_source",
        "extract_best_frontier_resume_checkpoint",
    ],
)
def test_legacy_resume_paths_reject_modeled_minimum_execution_checkpoint(
    checkpoint_location: str,
    resume_entrypoint: str,
) -> None:
    source = (
        _resume_source()
        if resume_entrypoint == "validate_static_hh_resume_source"
        else _best_frontier_resume_source()
    )
    payload = json.loads(json.dumps(source.payload))
    execution_checkpoint = {
        "schema": "sr_snake_modeled_minimum_execution_checkpoint_test_v1",
        "incumbent_state": {"energy": -0.5},
        "working_state": {"energy": -0.49},
        "scheduler_state": {"service_count": 3},
    }
    if checkpoint_location == "adapt_vqe":
        payload["adapt_vqe"]["modeled_minimum_execution_checkpoint"] = (
            execution_checkpoint
        )
    elif checkpoint_location == "top_level":
        payload["modeled_minimum_execution_checkpoint"] = execution_checkpoint
    else:
        payload.setdefault("checkpoint", {})[
            "modeled_minimum_execution_checkpoint"
        ] = execution_checkpoint
    guarded_source = ResumeScaffoldSource(
        artifact_json=source.artifact_json,
        artifact_sha256=source.artifact_sha256,
        payload=payload,
        runtime_input=source.runtime_input,
        import_summary=source.import_summary,
    )

    with pytest.raises(
        ValueError,
        match="cannot consume a modeled-minimum execution checkpoint",
    ):
        if resume_entrypoint == "validate_static_hh_resume_source":
            validate_static_hh_resume_source(
                guarded_source,
                continuation_mode="phase3_v1",
            )
        else:
            extract_best_frontier_resume_checkpoint(guarded_source)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("adapt_vqe", "history_checkpoint_complete"), False, "history_checkpoint_complete"),
        (("adapt_vqe", "history_count"), 2, "history_count"),
        (("checkpoint", "depth"), 2, "checkpoint depth"),
        (("adapt_vqe", "operators", 0), "wrong_operator", "history operators"),
        (
            (
                "adapt_vqe",
                "beam_replay_telemetry",
                "checkpoint_branch",
                "frontier_prune_key",
                "theta_round10",
                0,
            ),
            0.5,
            "frontier-prune theta",
        ),
        (("initial_state", "nq_total"), 2, "dimension disagrees"),
        (
            ("adapt_vqe", "route_a_trust_region_state", "update_count"),
            2,
            "trust update_count",
        ),
    ],
)
def test_strict_best_frontier_checkpoint_fails_closed_on_structural_drift(
    path: tuple[str | int, ...],
    value: object,
    message: str,
) -> None:
    source = _best_frontier_resume_source()
    payload = json.loads(json.dumps(source.payload))
    cursor: object = payload
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]
    corrupted = ResumeScaffoldSource(
        artifact_json=source.artifact_json,
        artifact_sha256=source.artifact_sha256,
        payload=payload,
        runtime_input=source.runtime_input,
        import_summary=source.import_summary,
    )

    with pytest.raises(ValueError, match=message):
        extract_best_frontier_resume_checkpoint(corrupted)


@pytest.mark.parametrize(
    "relative_path",
    [
        (
            "raw_outputs/paper_i_hh_new_paper_i_route4_two_stage_20260712/"
            "route4_exact_hessian_schur/weak_strong/full/json/current.json"
        ),
        (
            "raw_outputs/paper_i_hh_new_paper_i_route4_two_stage_20260712/"
            "route4_exact_hessian_schur/intermediate_strong/full/json/current.json"
        ),
    ],
)
def test_preserved_round21_route4_best_frontier_checkpoint_loads_when_present(
    relative_path: str,
) -> None:
    artifact = REPO_ROOT / relative_path
    if not artifact.exists():
        pytest.skip("preserved Paper-I Route-4 round-21 checkpoint is not present")

    checkpoint = extract_best_frontier_resume_checkpoint(
        load_static_resume_source(artifact)
    )

    assert checkpoint.controller_round == 21
    assert checkpoint.ansatz_depth == 21
    assert len(checkpoint.history) == 21
    assert checkpoint.route_a_trust_region_state["update_count"] == 21


def _preserved_depth30_sr_resume_source(
    regime: str,
    tmp_path: Path,
) -> ResumeScaffoldSource | None:
    artifact = REPO_ROOT / (
        "raw_outputs/paper_i_hh_sr_snake_noprune_nobeam_no_ordinary_novelty_"
        f"five_20260715_v1_chtc/{regime}/json/current.json"
    )
    ledger = artifact.with_name("estimator_call_ledger.json")
    sidecar = REPO_ROOT / (
        "chtc/phase3_optuna/input/paper_i_hh_sr_snake_noprune_nobeam_no_"
        "ordinary_novelty_r50_continuations_20260715_v1_chtc/resume_inputs/"
        f"{regime}.round30.signed_active_prefix_checkpoint.json"
    )
    if not artifact.exists() or not ledger.exists() or not sidecar.exists():
        return None
    source = load_static_resume_source(artifact)
    staged_current = tmp_path / "current.json"
    staged_sidecar = tmp_path / "signed_active_prefix_checkpoint.json"
    staged_ledger = tmp_path / "estimator_call_ledger.json"
    staged_sidecar.symlink_to(sidecar)
    staged_ledger.symlink_to(ledger)
    return ResumeScaffoldSource(
        artifact_json=staged_current,
        artifact_sha256=source.artifact_sha256,
        payload=source.payload,
        runtime_input=source.runtime_input,
        import_summary=source.import_summary,
    )


def test_preserved_depth30_sr_singleton_checkpoint_loads_when_present(
    tmp_path: Path,
) -> None:
    source = _preserved_depth30_sr_resume_source("strong_weak_u8", tmp_path)
    if source is None:
        pytest.skip("preserved depth-30 SR singleton checkpoint is not present")

    checkpoint = extract_verified_singleton_resume_checkpoint(
        source
    )

    assert checkpoint.controller_round == 30
    assert checkpoint.ansatz_depth == 30
    assert len(checkpoint.history) == 30
    assert checkpoint.branch_id is None
    assert checkpoint.parent_branch_id is None
    assert checkpoint.phase1_residual_opened is False
    assert checkpoint.phase1_stage_name == "core"
    assert checkpoint.maturity_controller_snapshot["snapshot_version"] == (
        "phase123_controller_maturity_v2"
    )
    assert len(checkpoint.selected_parent_pool_indices) == 30
    assert sum(
        1 for value in checkpoint.selected_parent_pool_indices if value == 34
    ) == 9
    assert sum(
        1 for value in checkpoint.selected_parent_pool_indices if value == 40
    ) == 11
    assert checkpoint.validation["controller_measurement_work_closed"] is True
    assert checkpoint.estimator_call_ledger_provenance[
        "restored_prefix_occurrence_count"
    ] == 32599


@pytest.mark.parametrize(
    ("regime", "expected_counts", "expected_phase_live", "expected_null_streaks"),
    [
        (
            "strong_weak_u8",
            {0: 1, 1: 1, 2: 1, 22: 1, 29: 1, 34: 9, 40: 11, 63: 1, 76: 2, 77: 2},
            {"phase1": True, "phase2": False, "phase3": False},
            {"phase2": 4, "phase3": 9},
        ),
        (
            "weak_strong",
            {0: 1, 1: 1, 28: 2, 35: 2, 40: 4, 46: 11, 67: 1, 82: 4, 83: 4},
            {"phase1": True, "phase2": True, "phase3": True},
            {"phase2": 0, "phase3": 0},
        ),
        (
            "intermediate_strong",
            {0: 1, 1: 1, 28: 2, 35: 2, 40: 8, 46: 7, 54: 1, 71: 1, 82: 2, 83: 5},
            {"phase1": True, "phase2": True, "phase3": True},
            {"phase2": 0, "phase3": 0},
        ),
        (
            "strong_strong_u8",
            {0: 1, 1: 1, 2: 1, 28: 2, 35: 5, 40: 6, 46: 5, 56: 3, 82: 4, 83: 2},
            {"phase1": True, "phase2": True, "phase3": True},
            {"phase2": 0, "phase3": 0},
        ),
    ],
)
def test_preserved_depth30_sr_controller_and_selection_states_when_present(
    tmp_path: Path,
    regime: str,
    expected_counts: dict[int, int],
    expected_phase_live: dict[str, bool],
    expected_null_streaks: dict[str, int],
) -> None:
    source = _preserved_depth30_sr_resume_source(regime, tmp_path)
    if source is None:
        pytest.skip(f"preserved {regime} controller sidecar is not present")

    controller = _load_verified_singleton_controller_state(source)
    selection = _load_verified_singleton_selection_state(source)

    assert controller["phase1_residual_opened"] is False
    assert controller["phase1_stage_name"] == "core"
    assert controller["controller_snapshot"]["phase_live"] == expected_phase_live
    assert controller["controller_snapshot"]["phase_null_streaks"] == (
        expected_null_streaks
    )
    assert Counter(selection["ordered_parent_pool_indices"]) == Counter(
        expected_counts
    )
    assert len(selection["ordered_parent_pool_indices"]) == 30
    assert selection["ordered_logical_candidate_indices"] == ()


def test_current_checkpoint_continuation_fields_pass_strict_resume_validation() -> None:
    source = _resume_source()
    payload = dict(source.payload)
    settings = dict(payload["settings"])
    settings.pop("adapt_continuation_mode", None)
    settings["continuation_mode"] = "phase3_v1"
    adapt = dict(payload["adapt_vqe"])
    adapt.pop("continuation_mode", None)
    adapt["continuation"] = {
        "continuation_mode": "phase3_v1",
        "oracle_gradient_config": None,
        "noise_floor_stop": {"policy": "off", "enabled": False},
    }
    payload["settings"] = settings
    payload["adapt_vqe"] = adapt
    source = ResumeScaffoldSource(
        artifact_json=source.artifact_json,
        artifact_sha256=source.artifact_sha256,
        payload=payload,
        runtime_input=source.runtime_input,
        import_summary=source.import_summary,
    )

    validation = validate_static_hh_resume_source(source, continuation_mode="phase3_v1")

    assert validation["settings_match"] is True
    assert validation["continuation_mode"] == "phase3_v1"
    assert validation["current_continuation_mode"] == "phase3_v1"
    assert validation["no_credentials_serialized"] is True


def test_resume_scope_defaults_legacy_to_phase3_and_blocks_midstream_opt_in() -> None:
    source = _resume_source()
    common_args = dict(
        problem="hh",
        adapt_resume_mode="scaffold_v1",
        L=2,
        t=1.0,
        u=2.0,
        dv=0.0,
        omega0=1.0,
        g_ep=0.5,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        include_zero_point=True,
        adapt_pool="pareto_lean",
    )

    legacy_validation = validate_static_hh_resume_source(
        source,
        args=SimpleNamespace(
            **common_args,
            historical_singleton_coordinate_solve_scope="phase3_only_v1",
        ),
        continuation_mode="phase3_v1",
    )
    assert legacy_validation["settings_match"] is True

    with pytest.raises(
        ValueError,
        match="historical_singleton_coordinate_solve_scope",
    ):
        validate_static_hh_resume_source(
            source,
            args=SimpleNamespace(
                **common_args,
                historical_singleton_coordinate_solve_scope=(
                    "phase2_and_phase3_v1"
                ),
            ),
            continuation_mode="phase3_v1",
        )


def test_molecular_h2o_linear_fd_resume_validation_accepts_matching_problem() -> None:
    source = _resume_source()
    payload = dict(source.payload)
    settings = dict(payload["settings"])
    settings.update(
        {
            "problem": "molecular_vibronic_h2o_linear_fd",
            "L": 6,
            "n_ph_max": 1,
            "adapt_pool": "full_meta",
        }
    )
    payload["settings"] = settings
    source = ResumeScaffoldSource(
        artifact_json=source.artifact_json,
        artifact_sha256=source.artifact_sha256,
        payload=payload,
        runtime_input=source.runtime_input,
        import_summary=source.import_summary,
    )
    args = SimpleNamespace(
        problem="molecular_vibronic_h2o_linear_fd",
        adapt_resume_mode="scaffold_v1",
        L=6,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        include_zero_point=True,
        adapt_pool="full_meta",
        molecular_vibronic_h2o_linear_fd_fixture_json=None,
    )

    validation = validate_static_hh_resume_source(
        source,
        args=args,
        continuation_mode="phase3_v1",
    )

    assert validation["problem"] == "molecular_vibronic_h2o_linear_fd"
    assert validation["settings_match"] is True
    assert validation["selected_term_count"] == 1


def test_conflicting_current_checkpoint_continuation_fields_fail_strict_validation() -> None:
    source = _resume_source()
    payload = dict(source.payload)
    settings = dict(payload["settings"])
    settings["adapt_continuation_mode"] = "phase3_v1"
    settings["continuation_mode"] = "phase3_v2"
    payload["settings"] = settings
    source = ResumeScaffoldSource(
        artifact_json=source.artifact_json,
        artifact_sha256=source.artifact_sha256,
        payload=payload,
        runtime_input=source.runtime_input,
        import_summary=source.import_summary,
    )

    with pytest.raises(ValueError, match="conflicting continuation mode fields"):
        validate_static_hh_resume_source(source, continuation_mode="phase3_v1")


def test_resume_summary_validation_and_pool_match_contract() -> None:
    source = _resume_source()
    args = SimpleNamespace(
        problem="hh",
        adapt_resume_mode="scaffold_v1",
        L=2,
        t=1.0,
        u=2.0,
        dv=0.0,
        omega0=1.0,
        g_ep=0.5,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        include_zero_point=True,
        adapt_pool="pareto_lean",
    )
    validation = validate_static_hh_resume_source(
        source,
        args=args,
        continuation_mode="phase3_v1",
    )
    summary = build_resume_import_summary(source, validation=validation)
    assert summary["schema_version"] == "static_hh_adapt_resume_import_v1"
    assert summary["source_ansatz_depth"] == 1
    assert summary["no_credentials_serialized"] is True
    assert summary["validation"]["settings_match"] is True

    match = match_resume_scaffold_to_pool(
        source,
        pool=[source.runtime_input.selected_terms[0]],
        build_selected_layout=lambda ops: build_parameter_layout(
            ops,
            ignore_identity=True,
            coefficient_tolerance=1e-12,
            sort_terms=True,
        ),
        expected_parameterization_mode="per_pauli_term",
    )
    assert match.selected_pool_indices == (0,)
    assert match.validation["runtime_parameter_count"] == 1
    assert match.validation["strict_expected_mode_replay"]["passed"] is True


def test_resume_pool_match_restores_legacy_omitted_pool_execution_mode() -> None:
    source = _resume_source()
    payload = json.loads(json.dumps(source.payload))
    payload["adapt_vqe"]["parameterization"]["blocks"][0].pop(
        "execution_mode", None
    )
    payload["adapt_vqe"]["parameterization_execution_mode"] = (
        "logical_shared"
    )
    serialized_term = source.runtime_input.selected_terms[0]
    grouped_pool_term = AnsatzTerm(
        label=str(serialized_term.label),
        polynomial=serialized_term.polynomial,
        execution_mode="grouped_exact",
    )
    legacy_source = ResumeScaffoldSource(
        artifact_json=source.artifact_json,
        artifact_sha256=source.artifact_sha256,
        payload=payload,
        runtime_input=source.runtime_input,
        import_summary=source.import_summary,
    )

    match = match_resume_scaffold_to_pool(
        legacy_source,
        pool=[grouped_pool_term],
        build_selected_layout=lambda ops: build_parameter_layout(
            ops,
            ignore_identity=True,
            coefficient_tolerance=1e-12,
            sort_terms=True,
        ),
        expected_parameterization_mode="logical_shared",
    )

    assert match.selected_pool_indices == (0,)
    assert match.selected_ops[0].execution_mode == "grouped_exact"
    assert match.validation["legacy_execution_mode_omission_repaired"] is True
    assert len(match.validation["legacy_execution_mode_rebind_records"]) == 1


def test_verified_resume_restores_projected_child_guard_and_execution_contract(
    tmp_path: Path,
) -> None:
    label = "parent::child_set[0]::legal_projected"
    polynomial = PauliPolynomial("JW")
    polynomial.add_term(PauliTerm(2, ps="xx", pc=0.5))
    polynomial.add_term(PauliTerm(2, ps="yy", pc=0.5))
    polynomial._reduce()
    serialized_term = AnsatzTerm(
        label=label,
        polynomial=polynomial,
        execution_mode="termwise_product",
    )
    grouped_term = AnsatzTerm(
        label=label,
        polynomial=polynomial,
        execution_mode="grouped_exact",
    )
    layout = build_parameter_layout(
        [serialized_term],
        ignore_identity=True,
        coefficient_tolerance=1e-12,
        sort_terms=True,
    )
    parameterization = serialize_layout(layout)
    parameterization["blocks"][0].pop("execution_mode", None)
    theta_runtime = np.array([0.2, 0.2], dtype=float)
    theta_logical = np.array([0.2], dtype=float)
    psi_ref = np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j])
    grouped_layout = build_parameter_layout(
        [grouped_term],
        ignore_identity=True,
        coefficient_tolerance=1e-12,
        sort_terms=True,
    )
    psi_initial = CompiledAnsatzExecutor(
        [grouped_term],
        parameterization_layout=grouped_layout,
        parameterization_mode="logical_shared",
    ).prepare_state(theta_logical, psi_ref)
    payload = {
        "adapt_vqe": {
            "operators": [label],
            "optimal_point": theta_runtime.tolist(),
            "logical_optimal_point": theta_logical.tolist(),
            "parameterization": parameterization,
            "parameterization_execution_mode": "logical_shared",
            "history": [
                {
                    "depth": 1,
                    "selected_op": label,
                    "selected_records": [
                        {
                            "operator_label": label,
                            "generator_id": "child-generator",
                            "parent_generator_id": "parent-generator",
                            "template_id": "template",
                            "runtime_split_mode": "shortlist_pauli_children_v1",
                            "runtime_split_chosen_representation": "child_set",
                            "runtime_split_child_generator_ids": [
                                "child-generator"
                            ],
                        }
                    ],
                }
            ],
            "active_generator_sector_contract": {
                "fixed_sector_guarded_generator_labels": [label]
            },
        }
    }
    runtime_split = {
        "mode": "shortlist_pauli_children_v1",
        "representation": "child_set",
        "recommended_execution_mode": "grouped_exact",
        "symmetry_gate": {
            "passed": True,
            "hard_guard_present": True,
            "hard_guard_required": True,
        },
    }
    padding_lineage = {
        "status": "projected",
        "projection": {
            "active": True,
            "policy": "exact_projected_grouped_v1",
        },
    }
    checkpoint = {
        "schema": "paper_i_signed_active_prefix_checkpoint_v1",
        "checkpoint_kind": "post_admission_prune",
        "outer_iteration": 1,
        "active_ansatz_depth": 1,
        "ordered_active_operator_labels": [label],
        "ordered_active_operators": [
            {
                "active_position": 0,
                "label": label,
                "generator_id": "child-generator",
                "parent_generator_id": "parent-generator",
                "execution_mode": "grouped_exact",
                "serialized_terms_exyz_in_execution_order": [
                    {
                        "pauli_exyz": "xx",
                        "coeff_re": 0.5,
                        "coeff_im": 0.0,
                        "nq": 2,
                    },
                    {
                        "pauli_exyz": "yy",
                        "coeff_re": 0.5,
                        "coeff_im": 0.0,
                        "nq": 2,
                    },
                ],
                "runtime_split": runtime_split,
                "route_a_child_padding_lineage": padding_lineage,
            }
        ],
        "signed_unwrapped_runtime_parameters": theta_runtime.tolist(),
        "signed_unwrapped_logical_parameters": theta_logical.tolist(),
        "parameterization_execution_mode": "logical_shared",
        "active_generator_sector_contract": {
            "passed_with_parameterization": True,
            "fixed_sector_guarded_generator_count": 1,
            "fixed_sector_guarded_generator_labels": [label],
        },
        "state_sector_contract": {"passed": True},
        "strict_replay": {"passed": True},
        "projective_state_fingerprint": projective_state_fingerprint(
            psi_initial
        ),
    }
    checkpoint["checkpoint_sha256"] = digest_jsonable(checkpoint)
    current_path = tmp_path / "current.json"
    (tmp_path / "signed_active_prefix_checkpoint.json").write_text(
        json.dumps(
            {
                "schema": "static_adapt_signed_active_prefix_resume_sidecar_v1",
                "source_result_json": "preserved/result.json",
                "source_result_sha256": "c" * 64,
                "checkpoint": checkpoint,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    source = ResumeScaffoldSource(
        artifact_json=current_path,
        artifact_sha256="b" * 64,
        payload=payload,
        runtime_input=ScaffoldRuntimeInput(
            resolved_problem=SimpleNamespace(),
            psi_ref=psi_ref,
            psi_initial=psi_initial,
            base_layout=layout,
            theta_runtime=theta_runtime,
            theta_logical=theta_logical,
            structure_locked=False,
            exact_energy=None,
            selected_terms=(serialized_term,),
        ),
        import_summary={},
    )

    match = match_resume_scaffold_to_pool(
        source,
        pool=[],
        build_selected_layout=lambda ops: build_parameter_layout(
            ops,
            ignore_identity=True,
            coefficient_tolerance=1e-12,
            sort_terms=True,
        ),
        expected_parameterization_mode="logical_shared",
        require_source_generator_contract=True,
    )

    assert match.selected_ops[0].execution_mode == "grouped_exact"
    assert match.validation["strict_expected_mode_replay"]["passed"] is True
    contract = match.selected_generator_contracts[label]
    assert contract["symmetry_spec"]["hard_guard"] is True
    assert contract["compile_metadata"]["runtime_split"] == runtime_split
    assert contract["compile_metadata"]["route_a_child_padding_lineage"] == (
        padding_lineage
    )
    assert match.validation["signed_active_prefix_sidecar"][
        "checkpoint_sha256"
    ] == checkpoint["checkpoint_sha256"]

    payload["adapt_vqe"]["history"][-1]["active_prefix_checkpoint"] = checkpoint
    current_text = json.dumps(payload, sort_keys=True)
    current_path.write_text(current_text, encoding="utf-8")
    (tmp_path / "signed_active_prefix_checkpoint.json").unlink()
    embedded_source = ResumeScaffoldSource(
        artifact_json=current_path,
        artifact_sha256=hashlib.sha256(current_text.encode("utf-8")).hexdigest(),
        payload=payload,
        runtime_input=source.runtime_input,
        import_summary={},
    )
    embedded_match = match_resume_scaffold_to_pool(
        embedded_source,
        pool=[],
        build_selected_layout=lambda ops: build_parameter_layout(
            ops,
            ignore_identity=True,
            coefficient_tolerance=1e-12,
            sort_terms=True,
        ),
        expected_parameterization_mode="logical_shared",
        require_source_generator_contract=True,
    )

    embedded_provenance = embedded_match.validation[
        "signed_active_prefix_sidecar"
    ]
    assert embedded_provenance["resume_prefix_source"] == (
        "embedded_current_winning_history_v1"
    )
    assert embedded_provenance["checkpoint_sha256"] == (
        checkpoint["checkpoint_sha256"]
    )


def test_verified_resume_rejects_conflicting_repeated_label_contracts() -> None:
    base = {
        "label": "repeated",
        "execution_mode": "grouped_exact",
        "generator_id": "generator-a",
        "parent_generator_id": "parent-a",
        "symmetry_spec": {"hard_guard": True},
        "compile_metadata": {
            "serialized_terms_exyz": [
                {
                    "pauli_exyz": "xx",
                    "coeff_re": 1.0,
                    "coeff_im": 0.0,
                    "nq": 2,
                }
            ],
            "runtime_split": {"mode": "shortlist_pauli_children_v1"},
            "route_a_child_padding_lineage": {"status": "projected"},
        },
    }
    conflicting = json.loads(json.dumps(base))
    conflicting["generator_id"] = "generator-b"

    with pytest.raises(ValueError, match="conflicting execution/guard"):
        _assert_repeated_resume_contract_consistency([base, conflicting])


def test_obsolete_admission_rollback_state_is_dropped_with_migration_note(
    tmp_path: Path,
) -> None:
    source = _resume_source()
    payload = dict(source.payload)
    payload["settings"] = {
        **dict(payload["settings"]),
        "adapt_rollback_mode": "structural",
        "adapt_rollback_tolerance": 0.0,
        "route_a_funnel_config": {
            "duplicate_cooldown_policy": "one_round_exact_record_pre_child_phase1_v1"
        },
    }
    payload["continuation"] = {
        "selected_scaffold_history": [
            {
                "selected_op": "resume_test_x",
                "structural_rollback": True,
                "depth_rollback": True,
                "zero_gain_duplicate_guard": {"triggered": True},
                "suppressed_reason": "structural_rollback",
            }
        ]
    }
    payload["adapt_vqe"] = {
        **dict(payload["adapt_vqe"]),
        "final_full_refit": {"executed": True, "rollback": True},
    }
    executor = CompiledAnsatzExecutor(
        list(source.runtime_input.selected_terms),
        parameterization_layout=source.runtime_input.base_layout,
        parameterization_mode="per_pauli_term",
    )
    prepared = executor.prepare_state(
        source.runtime_input.theta_runtime,
        source.runtime_input.psi_ref,
    )
    payload["initial_state"] = build_statevector_manifest(
        psi_state=prepared,
        source="adapt_vqe",
        handoff_state_kind="prepared_state",
        amplitude_cutoff=1e-12,
    )
    artifact = tmp_path / "legacy_resume.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_static_resume_source(artifact)

    assert "adapt_rollback_mode" not in loaded.payload["settings"]
    assert "adapt_rollback_tolerance" not in loaded.payload["settings"]
    assert (
        "duplicate_cooldown_policy"
        not in loaded.payload["settings"]["route_a_funnel_config"]
    )
    history = extract_resume_history(loaded.payload)
    assert history == [{"selected_op": "resume_test_x"}]
    assert "rollback" not in loaded.payload["adapt_vqe"]["final_full_refit"]
    migration = loaded.import_summary[
        "obsolete_admission_rollback_state_migration"
    ]
    assert migration["applied"] is True
    assert migration["behavior"] == "ignored_and_dropped_before_resume"
    assert migration["removed_field_counts"]["structural_rollback"] == 1


def test_actual_hh_weak_weak_snake_trial0011_resume_artifact_loads_when_present() -> None:
    artifact = REPO_ROOT / (
        "raw_outputs/local_hh_symmetric_snake_energy_geom_nocost_routefix_20260530_v6_tmux/"
        "routeA_paper_i_three_model_hh_l2_nph2_three_model_sym_weak_weak_full_meta_energygeom_nocost_routefix_v6/"
        "run/hh_L2_nph2_three_model_sym_weak_weak/trial_0011/"
        "hh_L2_nph2_three_model_sym_weak_weak/json/result.json"
    )
    if not artifact.exists():
        pytest.skip("local HH weak-weak trial0011 resume artifact is not present")
    source = load_static_resume_source(artifact)
    validation = validate_static_hh_resume_source(source, continuation_mode="phase3_v1")
    assert validation["selected_term_count"] == 22
    assert validation["runtime_parameter_count"] == 134
    assert validation["settings_match"] is True


def test_pdf_tableiii_strong_weak_snake_provenance_replay_matches_when_present() -> None:
    source_map = REPO_ROOT / "MATH/paper_facing/paper_I_static_scaffold/hh_tableiii_convergence_sources.json"
    if not source_map.exists():
        pytest.skip("Paper-I HH Table-III convergence source map is not present")
    source_map_payload = json.loads(source_map.read_text(encoding="utf-8"))
    regime_payload = source_map_payload["regimes"]["strong_weak"]
    row = regime_payload["methods"]["SNAKE"]
    artifact = REPO_ROOT / str(row["strict_replay_json"])
    if not artifact.exists():
        pytest.skip("local PDF-provenance HH strong-weak SNAKE replay artifact is not present")

    source = load_static_resume_source(artifact)
    validation = validate_static_hh_resume_source(source, continuation_mode="phase3_v1")
    assert source.artifact_sha256 == row["strict_replay_sha256"]
    assert source.artifact_sha256 == row["source_sha256"]
    assert validation["selected_term_count"] == 11
    assert validation["runtime_parameter_count"] == 60
    assert validation["logical_parameter_count"] == 11
    assert validation["settings_match"] is True

    runtime_input = source.runtime_input
    layout = runtime_input.base_layout
    executor = CompiledAnsatzExecutor(
        list(runtime_input.selected_terms),
        coefficient_tolerance=float(layout.coefficient_tolerance),
        ignore_identity=bool(layout.ignore_identity),
        sort_terms=(str(layout.term_order).strip().lower() == "sorted"),
        parameterization_mode="per_pauli_term",
        parameterization_layout=layout,
    )
    psi = np.asarray(
        executor.prepare_state(
            np.asarray(runtime_input.theta_runtime, dtype=float).reshape(-1),
            np.asarray(runtime_input.psi_ref, dtype=complex).reshape(-1),
        ),
        dtype=complex,
    ).reshape(-1)
    assert np.linalg.norm(
        psi - np.asarray(runtime_input.psi_initial, dtype=complex).reshape(-1)
    ) == pytest.approx(0.0, abs=1e-12)

    replay_energy, _ = energy_via_one_apply(
        psi,
        compile_polynomial_action(runtime_input.resolved_problem.hamiltonian),
    )
    replay_energy = float(replay_energy)
    saved_energy = float(source.payload["adapt_vqe"]["energy"])
    reference_energy = float(regime_payload["reference_energy"])
    assert replay_energy == pytest.approx(float(row["strict_replay_energy_local"]), abs=1e-12)
    assert abs(saved_energy - replay_energy) == pytest.approx(
        float(row["strict_replay_energy_abs_diff_vs_result"]),
        abs=1e-12,
    )
    assert abs(saved_energy - reference_energy) == pytest.approx(
        float(row["last_final_abs_delta_e"]),
        abs=1e-12,
    )
    assert max(0.0, abs(saved_energy - reference_energy) - 2.0e-4) == 0.0


def test_output_rejects_canonical_profile_with_reduced_powell_chart() -> None:
    args = SimpleNamespace()
    adapt_payload = {
        "static_route_identity": {
            "route_family": "singleton_response_snake",
            "route_profile": "supported_whitened_adaptive_trust_v1",
            "powell_coordinate_chart_policy": (
                SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
            ),
        },
        "optimizer_coordinate_chart": {
            "powell_coordinate_chart_policy": (
                SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
            ),
        },
    }
    with pytest.raises(ValueError, match="route-profile/Powell-chart mismatch"):
        _resolved_output_powell_coordinate_chart_policy(
            args=args,
            adapt_payload=adapt_payload,
        )
