from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from pipelines.contracts.problem import ProblemRequest
from pipelines.scaffold.handoff_state_bundle import build_statevector_manifest
from pipelines.static_adapt.builders.problem_registry import (
    resolve_problem_context,
)
from pipelines.static_adapt.estimator_call_ledger import (
    EstimatorCallKey,
    EstimatorCallLedger,
    projective_state_fingerprint,
)
from pipelines.static_adapt.sr_snake._resume import (
    CanonicalAcceptedStateHydration,
    CanonicalResumeError,
    load_canonical_accepted_state_resume,
)
from pipelines.static_adapt.sr_snake.contracts import AcceptedStateResume
from pipelines.static_adapt.sr_snake_route_profile import (
    canonical_sr_snake_greedy_batch_projected_phase3_no_overlap_trust_v1_contract,
    canonical_sr_snake_greedy_batch_projected_phase3_no_overlap_trust_v1_contract_sha256,
    canonical_sr_snake_insertion_commutation_plateau_v1_contract,
    canonical_sr_snake_insertion_commutation_plateau_v1_contract_sha256,
)


def _problem(*, n_fermions: int | None = None) -> Any:
    return resolve_problem_context(
        ProblemRequest(
            problem_key="hh",
            num_sites=2,
            t=1.0,
            u=0.5,
            dv=0.0,
            omega0=1.0,
            g_ep=0.2,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            n_fermions=n_fermions,
        ),
        hamiltonian=object(),
        exact_energy_impl=lambda **_kwargs: -1.0,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(
        (
            json.dumps(
                value,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    )


def _scored_position_receipt() -> dict[str, Any]:
    phases: list[dict[str, Any]] = []
    for phase in ("phase_i", "phase_ii", "phase_iii"):
        records = [
            {
                "domain_record_id": f"{phase}:fixture_child_x:0",
                "generator_id": "generator:fixture_child_x",
                "pool_index": 0,
                "pool_label": "fixture_child_x",
                "insertion_position": 0,
                "position_class": "append",
            }
        ]
        phases.append(
            {
                "phase": phase,
                "population_count": 1,
                "records": records,
                "ordered_population_sha256": _digest(records),
            }
        )
    payload = {
        "schema": "paper_i_scored_insertion_position_population_v1",
        "coordinate_chart": "exact_ordered_insertion_zero_angle_v1",
        "append_position": 0,
        "phase_order": ["phase_i", "phase_ii", "phase_iii"],
        "phases": phases,
        "scored_record_count": 3,
        "interior_scored_count": 0,
        "append_scored_count": 3,
    }
    payload["sha256"] = _digest(payload)
    return payload


def _prefix_receipt(*, terminal: bool = False) -> dict[str, Any]:
    cumulative_components = {
        "N_H_outer": 1,
        "N_H_refit": 0,
        "N_grad": 0,
        "N_metric": 0,
    }
    delta_components = (
        {component: 0 for component in cumulative_components}
        if terminal
        else dict(cumulative_components)
    )
    delta_total = 0 if terminal else 1
    return {
        "schema": "paper_i_active_prefix_estimator_ledger_receipt_v2",
        "enabled": True,
        "status": "complete",
        "checkpoint_sequence": 2 if terminal else 1,
        "occurrence_sequence_start_exclusive": 1 if terminal else 0,
        "occurrence_sequence_end_inclusive": 1,
        "raw_occurrence_delta": {
            "components": dict(delta_components),
            "total": delta_total,
        },
        "executed_query_delta": {
            "components": dict(delta_components),
            "S_alg": delta_total,
        },
        "unique_primitive_delta": {
            "components": dict(delta_components),
            "S_unique": delta_total,
        },
        "cumulative_raw_occurrences": {
            "components": dict(cumulative_components),
            "total": 1,
        },
        "cumulative_executed_queries": {
            "components": dict(cumulative_components),
            "S_alg": 1,
            "unit": "executed_logical_scalar_estimator_invocation",
        },
        "cumulative_unique_primitives": {
            "components": dict(cumulative_components),
            "S_unique": 1,
        },
        "runtime_estimator_occurrence_contract": (
            "all_instrumented_logical_scalar_estimator_calls_v1"
        ),
        "physical_identity_collapse_is_diagnostic_only": True,
        "raw_occurrences_preserved": True,
        "outer_iteration": 1,
        "checkpoint_kind": (
            "terminal_post_final_refit_and_prune"
            if terminal
            else "post_admission_prune"
        ),
        "branch_id": None,
        "parent_branch_id": None,
    }


def _signed_prefix(
    *,
    problem: Any,
    route_profile: str,
    route_digest: str,
    ledger_receipt: dict[str, Any],
    checkpoint_kind: str = "post_admission_prune",
) -> dict[str, Any]:
    nq = int(problem.layout.total_qubits)
    pauli = "e" * (nq - 1) + "x"
    runtime_term = {
        "pauli_exyz": pauli,
        "coeff_re": 0.0,
        "coeff_im": 1.0,
        "nq": nq,
    }
    parameterization = {
        "mode": "per_pauli_term_v1",
        "term_order": "sorted",
        "ignore_identity": True,
        "coefficient_tolerance": 1.0e-12,
        "logical_operator_count": 1,
        "runtime_parameter_count": 1,
        "blocks": [
            {
                "candidate_label": "fixture_child_x",
                "logical_index": 0,
                "runtime_start": 0,
                "runtime_count": 1,
                "execution_mode": "termwise_product",
                "runtime_terms_exyz": [dict(runtime_term)],
            }
        ],
    }
    state = problem.reference_state.build_state()
    checkpoint = {
        "schema": "paper_i_signed_active_prefix_checkpoint_v1",
        "checkpoint_kind": checkpoint_kind,
        "outer_iteration": 1,
        "active_ansatz_depth": 1,
        "ordered_active_operator_labels": ["fixture_child_x"],
        "ordered_active_operators": [
            {
                "active_position": 0,
                "label": "fixture_child_x",
                "generator_id": "generator:fixture_child_x",
                "parent_generator_id": "generator:fixture_parent",
                "execution_mode": "termwise_product",
                "serialized_terms_exyz_in_execution_order": [
                    dict(runtime_term)
                ],
                "runtime_split": {
                    "mode": "shortlist_pauli_children_v1",
                },
                "route_a_child_padding_lineage": {
                    "guard": "hard",
                },
            }
        ],
        "signed_unwrapped_runtime_parameters": [0.0],
        "signed_unwrapped_logical_parameters": [0.0],
        "parameterization_mode": "per_pauli_term_v1",
        "parameterization_execution_mode": "per_pauli_term_v1",
        "parameterization_execution_mode_source": "canonical_route",
        "optimizer_coordinate_chart": {
            "powell_coordinate_chart_policy": (
                "expanded_runtime_projected_logical_v1"
            )
        },
        "sr_route_profile": route_profile,
        "sr_route_profile_contract_sha256": route_digest,
        "phase1_score_mode": "trust_region_v1",
        "phase1_energy_model": "first_order_fs_trust_v1",
        "phase2_curvature_policy": "measured_required_fail_closed_v1",
        "phase2_cheap_curvature_proxy_policy": "off",
        "parameterization": parameterization,
        "generator_sector_contract": {"passed": True},
        "generator_pool_sector_contract": {"passed": True},
        "active_generator_sector_contract": {
            "passed": True,
            "execution_passed": True,
            "passed_with_parameterization": True,
        },
        "state_sector_contract": {
            "schema": "static_adapt_state_sector_contract_v1",
            "passed": True,
        },
        "strict_replay": {
            "schema": "static_adapt_strict_state_replay_v1",
            "source": "active_prefix_checkpoint",
            "passed": True,
            "fidelity": 1.0,
            "phase_aligned_l2": 0.0,
            "tolerance": 1.0e-10,
        },
        "projective_state_fingerprint": projective_state_fingerprint(state),
        "estimator_ledger_receipt": dict(ledger_receipt),
        "fixed_spin_sector_probability": 1.0,
        "fixed_spin_sector_illegal_probability": 0.0,
        "boson_legal_codeword_probability": 1.0,
        "boson_illegal_codeword_probability": 0.0,
        "boson_legal_subspace": {},
        "admission_at_outer_iteration": {
            "selected_batch_labels": ["fixture_child_x"],
            "selected_batch_positions": [0],
            "selected_batch_effective_positions": [0],
        },
        "post_admission_prune": {
            "schema": "direct_no_prune_transition_v1",
            "enabled": False,
            "accepted_count": 0,
            "deleted_indices": [],
            "deleted_labels": [],
        },
    }
    checkpoint["checkpoint_sha256"] = _digest(checkpoint)
    return checkpoint


def _accounting(ledger_payload: dict[str, Any]) -> dict[str, Any]:
    occurrence = ledger_payload["occurrence_summary"]
    components = dict(occurrence["component_occurrence_counts"])
    work = {
        "schema": "paper_i_executed_logical_scalar_estimator_work_v2",
        "contract": (
            "required_executed_logical_scalar_estimator_invocations_v1"
        ),
        "scope": "accepted prefix through the current finalized round",
        "unit": "logical_scalar_estimator_invocation",
        "components": dict(components),
        **dict(components),
        "S_alg": 1,
        "S_unique": 1,
        "identity_repeat_occurrence_count": 0,
        "includes_rejected_evaluated_candidates": True,
        "persistent_or_prior_run_cache_reductions_allowed": False,
    }
    return {
        "schema": "paper_i_current_s_alg_accounting_v2",
        "enabled": True,
        "status": "resolved_from_live_state_keyed_instrumentation",
        "complete": True,
        "exact_blockers": [],
        "definition": "S_alg = N_H_outer + N_H_refit + N_grad + N_metric",
        "unit": "logical_scalar_estimator_invocation",
        "identity_collapsed_count_name": "S_unique",
        "components": dict(components),
        "S_alg": 1,
        "S_unique": 1,
        "all_branch_search_work": dict(work),
        "winning_lineage": dict(work),
        "all_branch_unique_primitive_diagnostic": ledger_payload["summary"],
        "winning_lineage_unique_primitive_diagnostic": (
            ledger_payload["summary"]
        ),
        "executed_occurrence_accounting": {
            "schema": "paper_i_estimator_execution_occurrences_v2",
            "canonical_s_alg_source": (
                "all_executed_logical_scalar_estimator_occurrences"
            ),
            "canonical_s_alg_is_unique_primitive_based": False,
            "optimizer_and_guard_nfev_reported": 3,
            "all_execution": occurrence,
            "allocation_is_disjoint_by_consumer": True,
        },
    }


def _write_checkpoint_bundle(tmp_path: Path) -> tuple[
    Any,
    str,
    str,
    Path,
    Path,
]:
    problem = _problem()
    contract = (
        canonical_sr_snake_insertion_commutation_plateau_v1_contract()
    )
    route_profile = str(contract["route_profile"])
    route_digest = (
        canonical_sr_snake_insertion_commutation_plateau_v1_contract_sha256()
    )

    ledger = EstimatorCallLedger()
    state_fingerprint = projective_state_fingerprint(
        problem.reference_state.build_state()
    )
    ledger.record_call(
        EstimatorCallKey(
            projective_state_fingerprint=state_fingerprint,
            hamiltonian_fingerprint="hamiltonian:fixture",
            backend_fingerprint="backend:compiled",
            precision_contract="precision:float64",
            primitive_kind="hamiltonian_expectation",
            observable_or_formula_identity="hamiltonian_expectation_v1",
        ),
        component="N_H_outer",
        consumer_scope="outer_state_refresh",
    )
    ledger_payload = ledger.to_payload()
    accounting = _accounting(ledger_payload)
    receipt = _prefix_receipt()
    terminal_receipt = _prefix_receipt(terminal=True)
    prefix = _signed_prefix(
        problem=problem,
        route_profile=route_profile,
        route_digest=route_digest,
        ledger_receipt=receipt,
    )
    terminal_prefix = _signed_prefix(
        problem=problem,
        route_profile=route_profile,
        route_digest=route_digest,
        ledger_receipt=terminal_receipt,
        checkpoint_kind="terminal_post_final_refit_and_prune",
    )
    snapshot = {
        "snapshot_version": "phase123_controller_maturity_v2",
        "step_index": 0,
        "depth_local": 0,
        "depth_left": 1,
        "runway_ratio": 1.0,
        "early_coordinate": 1.0,
        "late_coordinate": 0.0,
        "frontier_ratio": 1.0,
        "phase_live": {
            "phase1": True,
            "phase2": True,
            "phase3": True,
        },
    }
    no_prune = {
        "schema": "direct_no_prune_transition_v1",
        "enabled": False,
        "accepted_count": 0,
        "deleted_indices": [],
        "deleted_labels": [],
    }
    insertion_receipt = {
        "schema": "insertion_commutation_plateau_round_policy_v1",
        "policy": "insertion_commutation_plateau_v1",
        "domain_state": "closed",
        "domain_open": False,
        "effective_insertion_mode": "append_only",
        "calibration_status": (
            "source_locked_completed_trajectory_replay_v1"
        ),
        "exact_reference_used": False,
        "requested_positions": [0],
        "candidate_count": 1,
        "requested_position_count": 1,
        "retained_representative_count": 1,
        "collapsed_position_count": 0,
        "candidate_position_plans": [
            {
                "schema": "commutation_reduced_insertion_positions_v1",
                "candidate_pool_index": 0,
                "candidate_label": "fixture_child_x",
                "requested_positions": [0],
                "representative_positions": [0],
                "representative_by_position": {0: 0},
                "members_by_representative": {0: [0]},
                "commuting_crossings": [],
                "collapsed_position_count": 0,
            }
        ],
        "retained_representatives": [
            {
                "candidate_pool_index": 0,
                "candidate_label": "fixture_child_x",
                "positions": [0],
            }
        ],
    }
    history_row = {
        "depth": 1,
        "batch_size": 1,
        "branch_id": None,
        "parent_branch_id": None,
        "selected_op": "fixture_child_x",
        "selected_logical_op": "fixture_child_x",
        "selected_logical_size": 1,
        "selected_logical_pool_indices": [0],
        "pool_index": 0,
        "selected_ops": ["fixture_child_x"],
        "selected_pool_indices": [0],
        "selected_position": 0,
        "selected_positions": [0],
        "selected_effective_positions": [0],
        "selected_batch_labels": ["fixture_child_x"],
        "selected_batch_positions": [0],
        "selected_batch_effective_positions": [0],
        "generator_id": "generator:fixture_child_x",
        "phase1_energy_model": "first_order_fs_trust_v1",
        "selected_feature_rows": [
            {
                "candidate_label": "fixture_child_x",
                "candidate_pool_index": 0,
                "generator_id": "generator:fixture_child_x",
                "controller_snapshot": dict(snapshot),
            }
        ],
        "energy_before_opt": -0.25,
        "energy_after_opt": -0.5,
        "nfev_opt": 3,
        "nfev_total_after_step": 3,
        "insertion_commutation_plateau": insertion_receipt,
        "scored_insertion_position_population": (
            _scored_position_receipt()
        ),
        "post_admission_prune": dict(no_prune),
        "active_prefix_checkpoint": prefix,
    }

    current_path = tmp_path / "current.json"
    sidecar_payload = {
        "schema": "paper_i_estimator_call_ledger_checkpoint_sidecar_v2",
        "ledger_scope": "single_route",
        "generated_utc": "2026-07-25T00:00:00+00:00",
        "checkpoint": {
            "reason": "iteration_done",
            "depth": 1,
            "ledger_scope": "single_route",
            "beam_enabled": False,
            "checkpoint_branch_policy": None,
            "branch_id": None,
            "parent_branch_id": None,
            "current_round_finalized": True,
        },
        "ledger": ledger_payload,
        "ledger_fingerprint": ledger_payload["ledger_fingerprint"],
        "unique_primitive_count": 1,
        "raw_occurrence_count": 1,
        "S_alg": 1,
        "S_unique": 1,
        "consumer_complete_projection": {},
        "no_credentials_serialized": True,
    }
    sidecar_bytes = (
        json.dumps(
            sidecar_payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    sidecar_sha = hashlib.sha256(sidecar_bytes).hexdigest()
    sidecar_path = tmp_path / (
        "current.estimator_call_ledger_checkpoint."
        f"{sidecar_sha[:16]}.json"
    )
    sidecar_path.write_bytes(sidecar_bytes)
    pointer = {
        "schema": "paper_i_estimator_call_ledger_checkpoint_pointer_v2",
        "enabled": True,
        "status": "complete",
        "path": sidecar_path.name,
        "sha256": sidecar_sha,
        "ledger_schema": ledger_payload["schema"],
        "checkpoint_reason": "iteration_done",
        "ledger_scope": "single_route",
        "beam_enabled": False,
        "checkpoint_branch_policy": None,
        "branch_id": None,
        "parent_branch_id": None,
        "ledger_fingerprint": ledger_payload["ledger_fingerprint"],
        "unique_primitive_count": 1,
        "raw_occurrence_count": 1,
        "S_alg": 1,
        "S_unique": 1,
        "checkpoint_depth": 1,
        "current_round_finalized": True,
    }
    continuation = {
        "active_prefix_checkpoints": [prefix],
        "terminal_active_prefix_checkpoint": terminal_prefix,
        "all_active_prefix_estimator_ledger_receipts": [
            receipt,
            terminal_receipt,
        ],
        "active_prefix_estimator_ledger_closure": {
            "schema": "paper_i_active_prefix_estimator_ledger_closure_v1",
            "enabled": True,
            "status": "complete",
            "passed": True,
            "receipt_count": 2,
            "summed_raw_occurrences": {
                "components": dict(accounting["components"]),
                "total": 1,
            },
            "summed_unique_primitives": {
                "components": dict(accounting["components"]),
                "S_unique": 1,
            },
            "terminal_raw_occurrences": {
                "components": dict(accounting["components"]),
                "total": 1,
            },
            "terminal_unique_primitives": {
                "components": dict(accounting["components"]),
                "S_unique": 1,
            },
            "includes_discarded_branch_checkpoints": False,
        },
        "estimator_call_accounting": accounting,
    }
    adapt = {
        "success": False,
        "method": "hardcoded_adapt_vqe_full_meta",
        "route_family": contract["route_family"],
        "route_profile": route_profile,
        "sr_route_profile_request": route_profile,
        "sr_route_profile_resolved": route_profile,
        "sr_route_profile_contract": contract,
        "sr_route_profile_contract_sha256": route_digest,
        "energy": -0.5,
        "operators": ["fixture_child_x"],
        "ansatz_depth": 1,
        "num_parameters": 1,
        "logical_num_parameters": 1,
        "optimal_point": [0.0],
        "logical_optimal_point": [0.0],
        "parameterization": prefix["parameterization"],
        "parameterization_mode": "per_pauli_term_v1",
        "parameterization_execution_mode": "per_pauli_term_v1",
        "parameterization_execution_mode_source": "canonical_route",
        "pool_type": "full_meta",
        "pool_size": 2,
        "adapt_inner_optimizer": "POWELL",
        "phase1_score_mode": "trust_region_v1",
        "phase1_energy_model": "first_order_fs_trust_v1",
        "phase2_curvature_policy": "measured_required_fail_closed_v1",
        "phase2_cheap_curvature_proxy_policy": "off",
        "phase3_response_coordinate_scope": (
            "full_active_plus_singleton_v1"
        ),
        "history": [history_row],
        "history_tail": [history_row],
        "history_count": 1,
        "history_tail_count": 1,
        "history_checkpoint_complete": True,
        "stop_reason": None,
        "nfev_total": 3,
        "estimator_call_accounting": accounting,
        "S_alg": 1,
        "S_alg_components": accounting["components"],
        "S_unique": 1,
        "active_prefix_checkpoints": [prefix],
        "terminal_active_prefix_checkpoint": terminal_prefix,
        "continuation": continuation,
        "strict_replay": prefix["strict_replay"],
        "route_a_trust_region_state": {
            "schema": "route_a_trust_region_state_v1",
            "radius": 0.25,
            "reference_radius": 0.25,
            "update_count": 0,
            "last_update": None,
            "initialization_reason": "configured_initial_radius",
        },
        "controller_measurement_work_summary": {
            "schema": "controller_measurement_work_proxy_v1",
        },
        "partial_checkpoint": True,
        "checkpoint_reason": "iteration_done",
        "adapt_beam_enabled": False,
        "branch_id": None,
        "parent_branch_id": None,
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
        "estimator_call_ledger_checkpoint": pointer,
    }
    request = problem.request
    settings = {
        "problem": request.problem_key,
        "L": request.num_sites,
        "t": request.t,
        "u": request.u,
        "dv": request.dv,
        "omega0": request.omega0,
        "g_ep": request.g_ep,
        "n_ph_max": request.n_ph_max,
        "boson_encoding": request.boson_encoding,
        "ordering": request.ordering,
        "boundary": request.boundary,
        "include_zero_point": request.include_zero_point,
        "adapt_pool": "full_meta",
        "adapt_inner_optimizer": "POWELL",
        "adapt_continuation_mode": problem.default_continuation_mode,
        "sr_route_profile_request": route_profile,
        "sr_route_profile_resolved": route_profile,
        "sr_route_profile_contract": contract,
        "sr_route_profile_contract_sha256": route_digest,
        "phase1_score_mode": "trust_region_v1",
        "phase1_energy_model": "first_order_fs_trust_v1",
        "phase2_curvature_policy": "measured_required_fail_closed_v1",
        "phase2_cheap_curvature_proxy_policy": "off",
        "phase3_response_coordinate_scope": (
            "full_active_plus_singleton_v1"
        ),
    }
    reference_state = problem.reference_state.build_state()
    envelope = {
        "schema_version": "static_adapt_current_checkpoint_v1",
        "settings": settings,
        "adapt_vqe": adapt,
        "ansatz_input_state": build_statevector_manifest(
            psi_state=reference_state,
            source=problem.reference_state.source_label,
            handoff_state_kind="reference_state",
            amplitude_cutoff=1.0e-12,
        ),
        "initial_state": build_statevector_manifest(
            psi_state=reference_state,
            source="active_sr_snake_accepted_checkpoint",
            handoff_state_kind="prepared_state",
            amplitude_cutoff=1.0e-12,
        ),
        "no_credentials_serialized": True,
        "checkpoint": {
            "complete": False,
            "reason": "iteration_done",
            "beam_enabled": False,
            "checkpoint_branch_policy": None,
            "branch_id": None,
            "parent_branch_id": None,
            "depth": 1,
            "ansatz_depth": 1,
            "stop_reason": None,
            "sr_route_profile_contract_sha256": route_digest,
            "phase3_response_coordinate_scope": (
                "full_active_plus_singleton_v1"
            ),
            "estimator_call_ledger_checkpoint": pointer,
            "path": str(current_path),
        },
    }
    _write_json(current_path, envelope)
    return problem, route_profile, route_digest, current_path, sidecar_path


def _resume(path: Path) -> AcceptedStateResume:
    return AcceptedStateResume(
        checkpoint_path=path,
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _convert_bundle_to_two_member_greedy(
    *,
    problem: Any,
    current: Path,
) -> tuple[str, str]:
    payload = json.loads(current.read_text(encoding="utf-8"))
    contract = (
        canonical_sr_snake_greedy_batch_projected_phase3_no_overlap_trust_v1_contract(
            maximum_size=3,
            search_window_size=None,
        )
    )
    profile = str(contract["route_profile"])
    digest = (
        canonical_sr_snake_greedy_batch_projected_phase3_no_overlap_trust_v1_contract_sha256(
            maximum_size=3,
            search_window_size=None,
        )
    )
    adapt = payload["adapt_vqe"]
    settings = payload["settings"]
    for block in (settings, adapt):
        block["sr_route_profile_request"] = profile
        block["sr_route_profile_resolved"] = profile
        block["sr_route_profile_contract"] = contract
        block["sr_route_profile_contract_sha256"] = digest
    adapt["route_profile"] = profile
    adapt["route_family"] = "greedy_batch_response_snake"
    payload["checkpoint"]["sr_route_profile_contract_sha256"] = digest
    payload["checkpoint"]["ansatz_depth"] = 2

    history_prefix = adapt["active_prefix_checkpoints"][0]
    terminal_prefix = adapt["terminal_active_prefix_checkpoint"]
    nq = int(problem.layout.total_qubits)
    second_term = {
        "pauli_exyz": "e" * (nq - 1) + "y",
        "coeff_re": 0.0,
        "coeff_im": 1.0,
        "nq": nq,
    }
    for prefix in (history_prefix, terminal_prefix):
        prefix["sr_route_profile"] = profile
        prefix["sr_route_profile_contract_sha256"] = digest
        prefix["active_ansatz_depth"] = 2
        prefix["ordered_active_operator_labels"] = [
            "fixture_child_x",
            "fixture_child_y",
        ]
        prefix["ordered_active_operators"].append(
            {
                "active_position": 1,
                "label": "fixture_child_y",
                "generator_id": "generator:fixture_child_y",
                "parent_generator_id": "generator:fixture_parent",
                "execution_mode": "termwise_product",
                "serialized_terms_exyz_in_execution_order": [
                    dict(second_term)
                ],
                "runtime_split": {
                    "mode": "shortlist_pauli_children_v1",
                },
                "route_a_child_padding_lineage": {"guard": "hard"},
            }
        )
        prefix["signed_unwrapped_runtime_parameters"] = [0.0, 0.0]
        prefix["signed_unwrapped_logical_parameters"] = [0.0, 0.0]
        parameterization = prefix["parameterization"]
        parameterization["logical_operator_count"] = 2
        parameterization["runtime_parameter_count"] = 2
        parameterization["blocks"].append(
            {
                "candidate_label": "fixture_child_y",
                "logical_index": 1,
                "runtime_start": 1,
                "runtime_count": 1,
                "execution_mode": "termwise_product",
                "runtime_terms_exyz": [dict(second_term)],
            }
        )
        prefix["admission_at_outer_iteration"] = {
            "selected_batch_labels": [
                "fixture_child_x",
                "fixture_child_y",
            ],
            "selected_batch_positions": [0, 0],
            "selected_batch_effective_positions": [0, 1],
        }
        unsigned = dict(prefix)
        unsigned.pop("checkpoint_sha256")
        prefix["checkpoint_sha256"] = _digest(unsigned)
    parameterization = terminal_prefix["parameterization"]

    row = adapt["history"][0]
    row.update(
        {
            "selected_logical_size": 2,
            "selected_logical_pool_indices": [0, 1],
            "selected_ops": ["fixture_child_x", "fixture_child_y"],
            "selected_pool_indices": [0, 1],
            "selected_positions": [0, 0],
            "selected_effective_positions": [0, 1],
            "selected_batch_labels": [
                "fixture_child_x",
                "fixture_child_y",
            ],
            "selected_batch_positions": [0, 0],
            "selected_batch_effective_positions": [0, 1],
            "active_prefix_checkpoint": history_prefix,
            "greedy_batch_admission": {
                "schema": "sr_snake_greedy_batch_admission_v1",
                "maximum_size": 3,
                "search_window_size": None,
                "selected_record_ids": ["record:x", "record:y"],
                "selected_generator_ids": [
                    "generator:fixture_child_x",
                    "generator:fixture_child_y",
                ],
                "selected_original_positions": [0, 0],
                "selected_effective_positions": [0, 1],
            },
        }
    )
    row.pop("insertion_commutation_plateau")
    snapshot = row["selected_feature_rows"][0]["controller_snapshot"]
    row["selected_feature_rows"].append(
        {
            "candidate_label": "fixture_child_y",
            "candidate_pool_index": 1,
            "generator_id": "generator:fixture_child_y",
            "controller_snapshot": snapshot,
        }
    )
    adapt["history_tail"] = [row]
    adapt["operators"] = ["fixture_child_x", "fixture_child_y"]
    adapt["ansatz_depth"] = 2
    adapt["num_parameters"] = 2
    adapt["logical_num_parameters"] = 2
    adapt["optimal_point"] = [0.0, 0.0]
    adapt["logical_optimal_point"] = [0.0, 0.0]
    adapt["parameterization"] = parameterization
    adapt["active_prefix_checkpoints"] = [history_prefix]
    adapt["terminal_active_prefix_checkpoint"] = terminal_prefix
    adapt["continuation"]["active_prefix_checkpoints"] = [history_prefix]
    adapt["continuation"][
        "terminal_active_prefix_checkpoint"
    ] = terminal_prefix
    _write_json(current, payload)
    return profile, digest


def _convert_bundle_to_one_round_beam(
    *,
    problem: Any,
    current: Path,
) -> tuple[str, str]:
    payload = json.loads(current.read_text(encoding="utf-8"))
    adapt = payload["adapt_vqe"]
    settings = payload["settings"]
    contract = json.loads(
        json.dumps(settings["sr_route_profile_contract"])
    )
    contract["execution_settings"].update(
        {
            "adapt_beam_live_branches": 2,
            "adapt_beam_children_per_parent": 2,
            "adapt_beam_lambda": 0.01,
        }
    )
    contract["semantic_invariants"].update(
        {
            "canonical_beam_policy": "fork_local",
            "beam_comparison": (
                "accepted_energy_plus_weight_times_fork_local_s_alg_v1"
            ),
            "beam_global_accounting": (
                "all_executed_branch_occurrences_in_global_s_alg_v1"
            ),
            "beam_live_parent_branches": 2,
            "beam_admission_children_per_parent": 2,
            "beam_maximum_admission_children_per_round": 2,
            "beam_s_alg_weight": 0.01,
            "beam_calibration_status": "uncalibrated_default",
            "beam_unchanged_parent_survival": False,
            "beam_phase_live_hysteresis": False,
        }
    )
    profile = str(contract["route_profile"])
    digest = _digest(contract)
    for block in (settings, adapt):
        block["sr_route_profile_contract"] = contract
        block["sr_route_profile_contract_sha256"] = digest
    payload["checkpoint"]["sr_route_profile_contract_sha256"] = digest

    winner_id = "canonical_beam:r1:p0:c0:n1"
    discarded_id = "canonical_beam:r1:p0:c1:n2"
    state_fingerprint = projective_state_fingerprint(
        problem.reference_state.build_state()
    )
    ledger = EstimatorCallLedger()
    for index, (component, scope, branch_id) in enumerate(
        (
            ("N_H_outer", "outer_state_refresh", None),
            ("N_H_refit", "energy:winner", winner_id),
            ("N_H_refit", "energy:discarded", discarded_id),
        )
    ):
        ledger.record_call(
            EstimatorCallKey(
                projective_state_fingerprint=state_fingerprint,
                hamiltonian_fingerprint="hamiltonian:beam-fixture",
                backend_fingerprint="backend:compiled",
                precision_contract="precision:float64",
                primitive_kind="hamiltonian_expectation",
                observable_or_formula_identity=(
                    f"hamiltonian_expectation_v1:{index}"
                ),
            ),
            component=component,
            consumer_scope=scope,
            branch_id=branch_id,
        )
    ledger_payload = ledger.to_payload()
    full_occurrence = ledger.occurrence_summary()
    full_unique = ledger.summary()
    winning_occurrence = ledger.occurrence_summary(
        branch_ids=(winner_id,),
        include_unbranched=True,
    )
    winning_unique = ledger.summary(
        branch_ids=(winner_id,),
        include_unbranched=True,
    )
    full_components = dict(
        full_occurrence["component_occurrence_counts"]
    )
    winning_components = dict(
        winning_occurrence["component_occurrence_counts"]
    )

    def _work(
        *,
        components: dict[str, int],
        s_alg: int,
        s_unique: int,
        scope: str,
    ) -> dict[str, Any]:
        return {
            "schema": (
                "paper_i_executed_logical_scalar_estimator_work_v2"
            ),
            "contract": (
                "required_executed_logical_scalar_estimator_invocations_v1"
            ),
            "scope": scope,
            "unit": "logical_scalar_estimator_invocation",
            "components": dict(components),
            **dict(components),
            "S_alg": s_alg,
            "S_unique": s_unique,
            "identity_repeat_occurrence_count": s_alg - s_unique,
            "includes_rejected_evaluated_candidates": True,
            "persistent_or_prior_run_cache_reductions_allowed": False,
        }

    all_work = _work(
        components=full_components,
        s_alg=3,
        s_unique=3,
        scope="all executed beam work",
    )
    winning_work = _work(
        components=winning_components,
        s_alg=2,
        s_unique=2,
        scope="accepted beam lineage",
    )
    accounting = {
        "schema": "paper_i_current_s_alg_accounting_v2",
        "enabled": True,
        "status": "resolved_from_live_state_keyed_instrumentation",
        "complete": True,
        "exact_blockers": [],
        "definition": "S_alg = N_H_outer + N_H_refit + N_grad + N_metric",
        "unit": "logical_scalar_estimator_invocation",
        "identity_collapsed_count_name": "S_unique",
        "components": dict(full_components),
        "S_alg": 3,
        "S_unique": 3,
        "all_branch_search_work": all_work,
        "winning_lineage": winning_work,
        "all_branch_unique_primitive_diagnostic": full_unique,
        "winning_lineage_unique_primitive_diagnostic": winning_unique,
        "executed_occurrence_accounting": {
            "schema": "paper_i_estimator_execution_occurrences_v2",
            "canonical_s_alg_source": (
                "all_executed_logical_scalar_estimator_occurrences"
            ),
            "canonical_s_alg_is_unique_primitive_based": False,
            "optimizer_and_guard_nfev_reported": 3,
            "all_execution": full_occurrence,
            "allocation_is_disjoint_by_consumer": True,
        },
        "beam_accounting": {
            "schema": "paper_i_fork_local_beam_accounting_v1",
            "all_executed_search_work_included": True,
            "winning_branch_ids": [winner_id],
            "winning_lineage": winning_work,
            "discarded_s_alg": 1,
            "unchanged_parent_survival": False,
        },
    }

    def _beam_receipt(
        *,
        sequence: int,
        start: int,
        delta_components: dict[str, int],
        cumulative_components: dict[str, int],
        branch_id: str,
        kind: str,
    ) -> dict[str, Any]:
        delta = sum(delta_components.values())
        cumulative = sum(cumulative_components.values())
        return {
            "schema": "paper_i_active_prefix_estimator_ledger_receipt_v2",
            "enabled": True,
            "status": "complete",
            "checkpoint_sequence": sequence,
            "occurrence_sequence_start_exclusive": start,
            "occurrence_sequence_end_inclusive": cumulative,
            "raw_occurrence_delta": {
                "components": dict(delta_components),
                "total": delta,
            },
            "executed_query_delta": {
                "components": dict(delta_components),
                "S_alg": delta,
            },
            "unique_primitive_delta": {
                "components": dict(delta_components),
                "S_unique": delta,
            },
            "cumulative_raw_occurrences": {
                "components": dict(cumulative_components),
                "total": cumulative,
            },
            "cumulative_executed_queries": {
                "components": dict(cumulative_components),
                "S_alg": cumulative,
                "unit": (
                    "executed_logical_scalar_estimator_invocation"
                ),
            },
            "cumulative_unique_primitives": {
                "components": dict(cumulative_components),
                "S_unique": cumulative,
            },
            "runtime_estimator_occurrence_contract": (
                "all_instrumented_logical_scalar_estimator_calls_v1"
            ),
            "physical_identity_collapse_is_diagnostic_only": True,
            "raw_occurrences_preserved": True,
            "outer_iteration": 1,
            "checkpoint_kind": kind,
            "branch_id": branch_id,
            "parent_branch_id": None,
        }

    zero = {component: 0 for component in full_components}
    winner_delta = dict(zero)
    winner_delta.update({"N_H_outer": 1, "N_H_refit": 1})
    winner_cumulative = dict(winner_delta)
    discarded_delta = dict(zero)
    discarded_delta["N_H_refit"] = 1
    full_cumulative = dict(full_components)
    winner_receipt = _beam_receipt(
        sequence=1,
        start=0,
        delta_components=winner_delta,
        cumulative_components=winner_cumulative,
        branch_id=winner_id,
        kind="post_admission_prune",
    )
    discarded_receipt = _beam_receipt(
        sequence=2,
        start=2,
        delta_components=discarded_delta,
        cumulative_components=full_cumulative,
        branch_id=discarded_id,
        kind="post_admission_prune",
    )
    terminal_receipt = _beam_receipt(
        sequence=3,
        start=3,
        delta_components=zero,
        cumulative_components=full_cumulative,
        branch_id=winner_id,
        kind="terminal_post_final_refit_and_prune",
    )

    history_prefix = adapt["active_prefix_checkpoints"][0]
    terminal_prefix = adapt["terminal_active_prefix_checkpoint"]
    for prefix, receipt, kind in (
        (history_prefix, winner_receipt, "post_admission_prune"),
        (
            terminal_prefix,
            terminal_receipt,
            "terminal_post_final_refit_and_prune",
        ),
    ):
        prefix["sr_route_profile_contract_sha256"] = digest
        prefix["estimator_ledger_receipt"] = receipt
        prefix["checkpoint_kind"] = kind
        unsigned = dict(prefix)
        unsigned.pop("checkpoint_sha256")
        prefix["checkpoint_sha256"] = _digest(unsigned)

    history_row = adapt["history"][0]
    history_row["branch_id"] = winner_id
    history_row["parent_branch_id"] = None
    history_row["active_prefix_checkpoint"] = history_prefix
    adapt["history_tail"] = [
        json.loads(json.dumps(history_row))
    ]
    adapt["active_prefix_checkpoints"] = [history_prefix]
    adapt["terminal_active_prefix_checkpoint"] = terminal_prefix
    adapt["adapt_beam_enabled"] = True
    adapt["branch_id"] = winner_id
    adapt["parent_branch_id"] = None
    adapt["estimator_call_accounting"] = accounting
    adapt["S_alg"] = 3
    adapt["S_alg_components"] = dict(full_components)
    adapt["S_unique"] = 3
    diagnostics = {
        "schema": "paper_i_canonical_fork_local_beam_search_v1",
        "comparison": "accepted_energy_plus_weight_times_lineage_s_alg",
        "s_alg_scope": "fork_local_cumulative_lineage",
        "s_alg_weight": 0.01,
        "calibration_status": "uncalibrated_default",
        "live_parent_branches": 2,
        "admission_children_per_parent": 2,
        "maximum_admission_children_per_round": 2,
        "unchanged_parent_survival": False,
        "phase_live_hysteresis": False,
        "initial_unbranched_s_alg": 1,
        "all_executed_s_alg": 3,
        "resume_winning_branch_ids": [],
        "resume_winning_lineage_s_alg": 0,
        "winning_branch_ids": [winner_id],
        "winning_lineage_s_alg": 1,
        "winning_comparison_score": -0.49,
        "rounds": [
            {
                "controller_round": 1,
                "parent_rows": [
                    {
                        "parent_branch_id": None,
                        "children_executed": 2,
                        "unchanged_parent_retained": False,
                    }
                ],
                "children": [
                    {
                        "branch_id": winner_id,
                        "parent_branch_id": None,
                        "accepted_energy": -0.5,
                        "fork_local_s_alg_delta": 1,
                        "lineage_s_alg": 1,
                        "comparison_score": -0.49,
                        "selected_pool_indices": [0],
                        "stop_reasons": ["maximum_controller_rounds"],
                    },
                    {
                        "branch_id": discarded_id,
                        "parent_branch_id": None,
                        "accepted_energy": -0.4,
                        "fork_local_s_alg_delta": 1,
                        "lineage_s_alg": 1,
                        "comparison_score": -0.39,
                        "selected_pool_indices": [1],
                        "stop_reasons": ["maximum_controller_rounds"],
                    },
                ],
                "children_executed": 2,
                "survivor_branch_ids": [winner_id],
                "terminal_reason": "maximum_controller_rounds",
            }
        ],
    }
    adapt["beam_search_diagnostics"] = diagnostics
    closure = {
        "schema": "paper_i_active_prefix_estimator_ledger_closure_v1",
        "enabled": True,
        "status": "complete",
        "passed": True,
        "receipt_count": 3,
        "summed_raw_occurrences": {
            "components": dict(full_components),
            "total": 3,
        },
        "summed_unique_primitives": {
            "components": dict(full_components),
            "S_unique": 3,
        },
        "terminal_raw_occurrences": {
            "components": dict(full_components),
            "total": 3,
        },
        "terminal_unique_primitives": {
            "components": dict(full_components),
            "S_unique": 3,
        },
        "includes_discarded_branch_checkpoints": True,
    }
    adapt["continuation"] = {
        "active_prefix_checkpoints": [history_prefix],
        "terminal_active_prefix_checkpoint": terminal_prefix,
        "all_active_prefix_estimator_ledger_receipts": [
            winner_receipt,
            discarded_receipt,
            terminal_receipt,
        ],
        "active_prefix_estimator_ledger_closure": closure,
        "estimator_call_accounting": accounting,
    }

    sidecar_payload = {
        "schema": "paper_i_estimator_call_ledger_checkpoint_sidecar_v2",
        "ledger_scope": "all_executed_branches",
        "generated_utc": "2026-07-25T00:00:00+00:00",
        "checkpoint": {
            "reason": "iteration_done",
            "depth": 1,
            "ledger_scope": "all_executed_branches",
            "beam_enabled": True,
            "checkpoint_branch_policy": (
                "canonical_terminal_winning_lineage"
            ),
            "branch_id": winner_id,
            "parent_branch_id": None,
            "current_round_finalized": True,
        },
        "ledger": ledger_payload,
        "ledger_fingerprint": ledger_payload["ledger_fingerprint"],
        "unique_primitive_count": 3,
        "raw_occurrence_count": 3,
        "S_alg": 3,
        "S_unique": 3,
        "consumer_complete_projection": {},
        "no_credentials_serialized": True,
    }
    sidecar_bytes = (
        json.dumps(
            sidecar_payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    sidecar_sha = hashlib.sha256(sidecar_bytes).hexdigest()
    sidecar_path = current.with_name(
        "current.estimator_call_ledger_checkpoint."
        f"{sidecar_sha[:16]}.json"
    )
    sidecar_path.write_bytes(sidecar_bytes)
    pointer = {
        "schema": "paper_i_estimator_call_ledger_checkpoint_pointer_v2",
        "enabled": True,
        "status": "complete",
        "path": sidecar_path.name,
        "sha256": sidecar_sha,
        "ledger_schema": ledger_payload["schema"],
        "checkpoint_reason": "iteration_done",
        "ledger_scope": "all_executed_branches",
        "beam_enabled": True,
        "checkpoint_branch_policy": "canonical_terminal_winning_lineage",
        "branch_id": winner_id,
        "parent_branch_id": None,
        "ledger_fingerprint": ledger_payload["ledger_fingerprint"],
        "unique_primitive_count": 3,
        "raw_occurrence_count": 3,
        "S_alg": 3,
        "S_unique": 3,
        "checkpoint_depth": 1,
        "current_round_finalized": True,
    }
    adapt["estimator_call_ledger_checkpoint"] = pointer
    payload["checkpoint"].update(
        {
            "beam_enabled": True,
            "checkpoint_branch_policy": (
                "canonical_terminal_winning_lineage"
            ),
            "branch_id": winner_id,
            "parent_branch_id": None,
            "estimator_call_ledger_checkpoint": pointer,
        }
    )
    _write_json(current, payload)
    return profile, digest


def test_adapter_returns_immutable_replay_complete_hydration(
    tmp_path: Path,
) -> None:
    problem, profile, digest, current, _sidecar = (
        _write_checkpoint_bundle(tmp_path)
    )

    hydration = load_canonical_accepted_state_resume(
        _resume(current),
        expected_problem=problem,
        expected_route_profile=profile,
        expected_route_contract_sha256=digest,
    )

    assert isinstance(hydration, CanonicalAcceptedStateHydration)
    assert hydration.controller_round == 1
    assert hydration.route_family == "singleton_response_snake"
    assert hydration.accepted_energy == pytest.approx(-0.5)
    assert hydration.s_alg == 1
    assert hydration.s_unique == 1
    assert hydration.selection_counts_by_pool_index == (1, 0)
    assert hydration.available_pool_indices == (0, 1)
    assert hydration.selected_parent_pool_indices == (0,)
    assert hydration.operators[0].admission_round == 1
    assert hydration.operators[0].runtime_terms[0].pauli_exyz.endswith("x")
    assert hydration.logical_parameters == (0.0,)
    assert hydration.runtime_parameters == (0.0,)
    assert hydration.accepted_state_fingerprint.startswith(
        "projective_state_v1:"
    )
    assert hydration.estimator_prefix_checkpoint_cursor[
        "raw_occurrence_count"
    ] == 1
    assert hydration.mutable_estimator_call_ledger_payload()[
        "ledger_fingerprint"
    ]
    with pytest.raises(TypeError):
        hydration.history[0]["depth"] = 2  # type: ignore[index]


def test_adapter_accepts_round_finalized_current_checkpoint_shape(
    tmp_path: Path,
) -> None:
    problem, profile, digest, current, _sidecar = (
        _write_checkpoint_bundle(tmp_path)
    )
    payload = json.loads(current.read_text(encoding="utf-8"))
    adapt = payload["adapt_vqe"]
    current_prefix = adapt["active_prefix_checkpoints"][-1]
    adapt["terminal_active_prefix_checkpoint"] = current_prefix
    continuation = adapt["continuation"]
    continuation["terminal_active_prefix_checkpoint"] = current_prefix
    continuation["all_active_prefix_estimator_ledger_receipts"] = [
        current_prefix["estimator_ledger_receipt"]
    ]
    continuation.pop("active_prefix_estimator_ledger_closure")
    _write_json(current, payload)

    hydration = load_canonical_accepted_state_resume(
        _resume(current),
        expected_problem=problem,
        expected_route_profile=profile,
        expected_route_contract_sha256=digest,
    )

    assert hydration.controller_round == 1
    assert hydration.estimator_prefix_checkpoint_cursor[
        "checkpoint_sequence"
    ] == 1


def test_adapter_still_requires_closure_for_terminalized_receipt(
    tmp_path: Path,
) -> None:
    problem, profile, digest, current, _sidecar = (
        _write_checkpoint_bundle(tmp_path)
    )
    payload = json.loads(current.read_text(encoding="utf-8"))
    payload["adapt_vqe"]["continuation"].pop(
        "active_prefix_estimator_ledger_closure"
    )
    _write_json(current, payload)

    with pytest.raises(
        CanonicalResumeError,
        match="estimator-prefix closure summary",
    ):
        load_canonical_accepted_state_resume(
            _resume(current),
            expected_problem=problem,
            expected_route_profile=profile,
            expected_route_contract_sha256=digest,
        )


def test_adapter_accepts_equivalent_explicit_particle_total(
    tmp_path: Path,
) -> None:
    _problem_default, profile, digest, current, _sidecar = (
        _write_checkpoint_bundle(tmp_path)
    )

    hydration = load_canonical_accepted_state_resume(
        _resume(current),
        expected_problem=_problem(n_fermions=2),
        expected_route_profile=profile,
        expected_route_contract_sha256=digest,
    )

    assert hydration.controller_round == 1


def test_adapter_hydrates_atomic_greedy_batch_from_same_deep_format(
    tmp_path: Path,
) -> None:
    problem, _profile, _digest_value, current, _sidecar = (
        _write_checkpoint_bundle(tmp_path)
    )
    profile, digest = _convert_bundle_to_two_member_greedy(
        problem=problem,
        current=current,
    )

    hydration = load_canonical_accepted_state_resume(
        _resume(current),
        expected_problem=problem,
        expected_route_profile=profile,
        expected_route_contract_sha256=digest,
    )

    assert hydration.route_family == "greedy_batch_response_snake"
    assert tuple(operator.label for operator in hydration.operators) == (
        "fixture_child_x",
        "fixture_child_y",
    )
    assert tuple(
        operator.admission_round for operator in hydration.operators
    ) == (1, 1)
    assert hydration.selection_counts_by_pool_index == (1, 1)
    assert hydration.selected_parent_pool_indices == (0, 1)


def test_adapter_rejects_outer_artifact_hash_mismatch(
    tmp_path: Path,
) -> None:
    problem, profile, digest, current, _sidecar = (
        _write_checkpoint_bundle(tmp_path)
    )
    resume = AcceptedStateResume(
        checkpoint_path=current,
        checkpoint_sha256="0" * 64,
    )

    with pytest.raises(CanonicalResumeError, match="checkpoint SHA-256"):
        load_canonical_accepted_state_resume(
            resume,
            expected_problem=problem,
            expected_route_profile=profile,
            expected_route_contract_sha256=digest,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["adapt_vqe"].update(
                {"history_checkpoint_complete": False}
            ),
            "partial",
        ),
        (
            lambda payload: payload["settings"].update({"t": 9.0}),
            "different physical problem",
        ),
        (
            lambda payload: payload["adapt_vqe"].update(
                {"sr_route_profile_contract_sha256": "0" * 64}
            ),
            "route-contract digest",
        ),
        (
            lambda payload: payload["adapt_vqe"][
                "estimator_call_accounting"
            ].update({"S_alg": 2}),
            "accounting",
        ),
    ],
)
def test_adapter_rejects_partial_problem_route_and_accounting_drift(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    problem, profile, digest, current, _sidecar = (
        _write_checkpoint_bundle(tmp_path)
    )
    payload = json.loads(current.read_text(encoding="utf-8"))
    mutate(payload)
    _write_json(current, payload)

    with pytest.raises(CanonicalResumeError, match=message):
        load_canonical_accepted_state_resume(
            _resume(current),
            expected_problem=problem,
            expected_route_profile=profile,
            expected_route_contract_sha256=digest,
        )


def test_adapter_rejects_semantically_failed_signed_replay(
    tmp_path: Path,
) -> None:
    problem, profile, digest, current, _sidecar = (
        _write_checkpoint_bundle(tmp_path)
    )
    payload = json.loads(current.read_text(encoding="utf-8"))
    adapt = payload["adapt_vqe"]
    checkpoint = adapt["terminal_active_prefix_checkpoint"]
    checkpoint["strict_replay"]["passed"] = False
    checkpoint_without_sha = dict(checkpoint)
    checkpoint_without_sha.pop("checkpoint_sha256")
    checkpoint["checkpoint_sha256"] = _digest(checkpoint_without_sha)
    adapt["history"][0]["active_prefix_checkpoint"] = checkpoint
    adapt["history_tail"][0]["active_prefix_checkpoint"] = checkpoint
    adapt["active_prefix_checkpoints"] = [checkpoint]
    adapt["continuation"]["active_prefix_checkpoints"] = [checkpoint]
    adapt["continuation"]["terminal_active_prefix_checkpoint"] = checkpoint
    _write_json(current, payload)

    with pytest.raises(CanonicalResumeError, match="strict.*replay"):
        load_canonical_accepted_state_resume(
            _resume(current),
            expected_problem=problem,
            expected_route_profile=profile,
            expected_route_contract_sha256=digest,
        )


def test_adapter_rejects_terminal_operator_replay_drift(
    tmp_path: Path,
) -> None:
    problem, profile, digest, current, _sidecar = (
        _write_checkpoint_bundle(tmp_path)
    )
    payload = json.loads(current.read_text(encoding="utf-8"))
    adapt = payload["adapt_vqe"]
    terminal = adapt["terminal_active_prefix_checkpoint"]
    terminal["ordered_active_operators"][0][
        "serialized_terms_exyz_in_execution_order"
    ][0]["coeff_im"] = 2.0
    terminal["parameterization"]["blocks"][0]["runtime_terms_exyz"][0][
        "coeff_im"
    ] = 2.0
    unsigned = dict(terminal)
    unsigned.pop("checkpoint_sha256")
    terminal["checkpoint_sha256"] = _digest(unsigned)
    adapt["parameterization"] = terminal["parameterization"]
    adapt["continuation"]["terminal_active_prefix_checkpoint"] = terminal
    _write_json(current, payload)

    with pytest.raises(CanonicalResumeError, match="Terminal signed"):
        load_canonical_accepted_state_resume(
            _resume(current),
            expected_problem=problem,
            expected_route_profile=profile,
            expected_route_contract_sha256=digest,
        )


def test_adapter_rejects_inconsistent_terminal_receipt_cumulative(
    tmp_path: Path,
) -> None:
    problem, profile, digest, current, _sidecar = (
        _write_checkpoint_bundle(tmp_path)
    )
    payload = json.loads(current.read_text(encoding="utf-8"))
    adapt = payload["adapt_vqe"]
    terminal = adapt["terminal_active_prefix_checkpoint"]
    receipt = terminal["estimator_ledger_receipt"]
    receipt["cumulative_raw_occurrences"]["total"] = 2
    unsigned = dict(terminal)
    unsigned.pop("checkpoint_sha256")
    terminal["checkpoint_sha256"] = _digest(unsigned)
    adapt["continuation"]["terminal_active_prefix_checkpoint"] = terminal
    adapt["continuation"][
        "all_active_prefix_estimator_ledger_receipts"
    ][-1] = receipt
    _write_json(current, payload)

    with pytest.raises(CanonicalResumeError, match="cumulative counters"):
        load_canonical_accepted_state_resume(
            _resume(current),
            expected_problem=problem,
            expected_route_profile=profile,
            expected_route_contract_sha256=digest,
        )


def test_adapter_rejects_hash_linked_ledger_sidecar_tampering(
    tmp_path: Path,
) -> None:
    problem, profile, digest, current, sidecar = (
        _write_checkpoint_bundle(tmp_path)
    )
    sidecar.write_bytes(sidecar.read_bytes() + b" ")

    with pytest.raises(CanonicalResumeError, match="sidecar SHA-256"):
        load_canonical_accepted_state_resume(
            _resume(current),
            expected_problem=problem,
            expected_route_profile=profile,
            expected_route_contract_sha256=digest,
        )
