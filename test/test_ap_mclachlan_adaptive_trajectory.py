from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory as aptraj
from pipelines.contracts.scaffold import CandidatePoolSource, ScaffoldRuntimeInput
from pipelines.time_dynamics.ap_mclachlan.append_cost import (
    AP_APPEND_COST_NORMALIZATION_RAW_LEGACY_V1,
    AP_APPEND_RANK_SCORE_KIND_V1,
    estimate_append_atom_set_cost,
)
from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import (
    APPEND_LADDER_PREFILTER_POLICY_V1,
    APPEND_LADDER_SELECTION_POLICY_V1,
    PRUNE_PERSISTENCE_ATOM_HISTORY,
    PRUNE_PERSISTENCE_EXACT_BATCH,
    SUPPORT_PATCH_EXCHANGE_SELECTION_POLICY_V1,
    AppendControllerConfig,
    PatchCandidateScore,
    SolveRepairConfig,
    SupportPatchControllerConfig,
    _checkpoint_local_subdivision_request,
    _PruneControllerRuntimeState,
    _repair_severity_for_step,
)
from pipelines.time_dynamics.ap_mclachlan.fixed_step import SolveRepairResponseSchedule
from pipelines.time_dynamics.ap_mclachlan.geometry import McLachlanGeometry
from pipelines.time_dynamics.ap_mclachlan.geometry_eval import (
    GeometryEvaluation,
    evaluate_mclachlan_geometry,
)
from pipelines.time_dynamics.ap_mclachlan.fixed_step import FixedMcLachlanStep
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import (
    time_dependent_hamiltonian_from_runtime_input,
)
from pipelines.time_dynamics.ap_mclachlan.inverse import McLachlanInversePolicy
from pipelines.time_dynamics.ap_mclachlan.state import state_from_scaffold_runtime_input
from pipelines.time_dynamics.ap_mclachlan.support_atoms import (
    APPEND_OCCURRENCE_POLICY_UNIQUE_SUPPORT,
    ActiveSupportAtom,
    SupportAtom,
    active_support_atoms,
    candidate_append_atoms,
)
from pipelines.time_dynamics.ap_mclachlan.support_patch import (
    PATCH_APPEND,
    PATCH_DELETE,
    PATCH_EXCHANGE,
    SupportPatch,
    SupportPatchGeometry,
    SupportPatchScore,
    build_support_patch_before_cache,
)
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
    run_append_ap_mclachlan_from_runtime_input,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _poly(label: str, coeff: float = 1.0) -> PauliPolynomial:
    poly = PauliPolynomial("JW")
    poly.add_term(PauliTerm(1, ps=str(label), pc=float(coeff)))
    poly._reduce()
    return poly


def _runtime_input_with_candidates(
    *candidates: AnsatzTerm,
    hamiltonian: PauliPolynomial | None = None,
    candidate_pool_filter_payload: dict | None = None,
) -> ScaffoldRuntimeInput:
    return ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(
            family_key="toy",
            hamiltonian=_poly("x") if hamiltonian is None else hamiltonian,
        ),
        psi_ref=np.array([1.0, 0.0], dtype=complex),
        psi_initial=np.array([1.0, 0.0], dtype=complex),
        base_layout=build_parameter_layout([]),
        theta_runtime=np.zeros(0, dtype=float),
        theta_logical=np.zeros(0, dtype=float),
        structure_locked=False,
        exact_energy=None,
        selected_terms=(),
        candidate_pool_terms=tuple(candidates),
        candidate_pool_source=CandidatePoolSource(
            source_kind="resolved_pool",
            pool_key="toy_pool",
            completeness="complete",
            filter_payload=dict(candidate_pool_filter_payload or {}),
        ),
        provenance={"artifact_json": "toy.json"},
    )


def _runtime_input_with_selected(*selected: AnsatzTerm) -> ScaffoldRuntimeInput:
    layout = build_parameter_layout(tuple(selected))
    theta = np.zeros(int(layout.runtime_parameter_count), dtype=float)
    psi_ref = np.array([1.0, 0.0], dtype=complex)
    executor = CompiledAnsatzExecutor(
        tuple(selected),
        parameterization_layout=layout,
        parameterization_mode="per_pauli_term",
    )
    psi_initial = executor.prepare_state(theta, psi_ref)
    return ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(family_key="toy", hamiltonian=_poly("x")),
        psi_ref=psi_ref,
        psi_initial=psi_initial,
        base_layout=layout,
        theta_runtime=theta,
        theta_logical=np.zeros(int(layout.logical_parameter_count), dtype=float),
        structure_locked=False,
        exact_energy=None,
        selected_terms=tuple(selected),
        candidate_pool_terms=tuple(selected),
        candidate_pool_source=CandidatePoolSource(
            source_kind="resolved_pool",
            pool_key="toy_pool",
            completeness="complete",
        ),
        provenance={"artifact_json": "toy.json"},
    )


def test_append_trajectory_inserts_candidate_without_exact_reference() -> None:
    runtime_input = _runtime_input_with_candidates(
        AnsatzTerm(label="candidate_x", polynomial=_poly("x")),
        AnsatzTerm(label="candidate_y", polynomial=_poly("y")),
    )

    payload = run_append_ap_mclachlan_from_runtime_input(
        runtime_input,
        times=(0.0, 0.1),
        controller_config=AppendControllerConfig(
            max_append_candidates=2,
            append_gain_threshold=0.0,
        ),
        support_patch_config=SupportPatchControllerConfig(
            append_ladder_mode="combinatorial",
            residual_ratio_threshold=0.0,
        ),
    )

    assert payload["summary"]["accepted_append_count"] == 1
    assert payload["summary"]["accepted_insert_count"] == 1
    assert payload["summary"]["runtime_parameter_count_initial"] == 0
    assert payload["summary"]["runtime_parameter_count_final"] == 1
    assert payload["plot_rows"][0]["patch_accepted"] is True
    assert payload["plot_rows"][0]["patch_kind"] == "append"
    decision = payload["trajectory"]["points"][0]["patch_decision"]
    metadata = decision["metadata"]
    assert metadata["selection_policy"].startswith("paper_ii_deletion_conditioned")
    assert metadata["kind"] == "insert"
    assert metadata["committed"]["inserted_selection"]
    assert decision["reason"] == "accepted_deletion_conditioned_insert"
    assert payload["plot_rows"][0]["theta_dot_l2"] > 0.0
    assert payload["decision_data_flow"]["uses_exact_reference_for_decision"] is False


def test_prune_atom_history_drops_deleted_atoms_after_support_change() -> None:
    old_state = state_from_scaffold_runtime_input(
        _runtime_input_with_selected(
            AnsatzTerm(label="seed_x", polynomial=_poly("x")),
            AnsatzTerm(label="seed_z", polynomial=_poly("z")),
        )
    )
    kept_state = state_from_scaffold_runtime_input(
        _runtime_input_with_selected(AnsatzTerm(label="seed_z", polynomial=_poly("z")))
    )
    old_atom_ids = tuple(str(atom.atom_id) for atom in active_support_atoms(old_state))
    kept_atom_ids = {str(atom.atom_id) for atom in active_support_atoms(kept_state)}
    deleted_atom_ids = tuple(atom_id for atom_id in old_atom_ids if atom_id not in kept_atom_ids)
    assert len(deleted_atom_ids) == 1

    runtime_state = _PruneControllerRuntimeState()
    for atom_id in old_atom_ids:
        runtime_state.atom_seen_history[atom_id] = [0, 1]
        runtime_state.cooldown_until_index[atom_id] = 7
    runtime_state.loss_history["old_batch"] = [(1, 0.01)]

    metadata = runtime_state.update_after_support_change(
        new_state=kept_state,
        theta_runtime=kept_state.theta_runtime,
        patch_kind=PATCH_DELETE,
    )

    assert metadata["prune_history_transition"] == "delete_preserved_surviving_atom_history"
    assert deleted_atom_ids[0] not in runtime_state.atom_seen_history
    assert deleted_atom_ids[0] not in runtime_state.cooldown_until_index
    assert set(runtime_state.atom_seen_history) == kept_atom_ids
    assert set(runtime_state.cooldown_until_index) == kept_atom_ids
    assert runtime_state.loss_history == {}


def test_append_min_time_skips_initial_grid_point_then_allows_append() -> None:
    runtime_input = _runtime_input_with_candidates(
        AnsatzTerm(label="candidate_x", polynomial=_poly("x")),
        AnsatzTerm(label="candidate_y", polynomial=_poly("y")),
    )

    payload = run_append_ap_mclachlan_from_runtime_input(
        runtime_input,
        times=(0.0, 0.1, 0.2),
        controller_config=AppendControllerConfig(
            max_append_candidates=2,
            append_gain_threshold=0.0,
            append_min_time=0.1,
        ),
        support_patch_config=SupportPatchControllerConfig(
            append_ladder_mode="combinatorial",
            residual_ratio_threshold=0.0,
            append_min_time=0.1,
        ),
    )

    assert payload["summary"]["accepted_insert_count"] >= 1
    assert payload["plot_rows"][0]["patch_accepted"] is False
    assert payload["plot_rows"][0]["patch_reason"] == "append_before_min_time"
    assert payload["plot_rows"][1]["patch_accepted"] is True
    assert payload["plot_rows"][1]["time"] == 0.1


def test_append_runner_records_solve_repair_policy_telemetry() -> None:
    runtime_input = _runtime_input_with_selected(
        AnsatzTerm(label="active_x", polynomial=_poly("x")),
    )

    payload = run_append_ap_mclachlan_from_runtime_input(
        runtime_input,
        times=(0.0, 0.1),
        solve_damping=1.0e-8,
        controller_config=AppendControllerConfig(
            max_append_candidates=0,
        ),
        solve_repair_config=SolveRepairConfig(
            enabled=True,
            condition_number_max=1.0e9,
            ridge_ladder=(1.0e-7,),
            pinv_rcond_ladder=(1.0e-10, 1.0e-8),
            solve_damping_ladder=(1.0e-8,),
        ),
    )

    row = payload["plot_rows"][0]
    assert payload["summary"]["solve_repair_enabled"] is True
    assert payload["summary"]["solve_damping"] == 1.0e-8
    assert row["solve_repair_enabled"] is True
    assert row["solve_repair_attempt_count"] >= 1
    assert row["effective_solve_damping"] == 1.0e-8
    assert payload["trajectory"]["solve_repair_config"]["enabled"] is True


def test_solve_repair_subdivides_interval_for_state_motion_guard() -> None:
    runtime_input = _runtime_input_with_selected(
        AnsatzTerm(label="active_x", polynomial=_poly("x")),
    )

    payload = run_append_ap_mclachlan_from_runtime_input(
        runtime_input,
        times=(0.0, 0.1),
        controller_config=AppendControllerConfig(
            max_append_candidates=0,
        ),
        solve_repair_config=SolveRepairConfig(
            enabled=True,
            condition_number_max=1.0e9,
            rho_num_max=1.0,
            state_motion_l2_step_max=7.5e-2,
            max_local_subdivisions=2,
            local_subdivision_factor=2,
            ridge_ladder=(1.0e-7,),
            pinv_rcond_ladder=(1.0e-10,),
            solve_damping_ladder=(0.0,),
        ),
    )

    row = payload["plot_rows"][0]
    integration = payload["trajectory"]["points"][0]["integration_to_next"]
    assert row["integration_local_subdivision_applied"] is True
    assert row["integration_local_subdivision_depth"] == 1
    assert row["integration_local_substep_count"] == 2
    assert integration["repair_summary"]["max_state_motion_l2_step"] < 7.5e-2
    assert payload["summary"]["local_subdivision_applied_count"] == 1


def test_solve_repair_subdivision_depth_scales_with_motion_severity() -> None:
    runtime_input = _runtime_input_with_selected(
        AnsatzTerm(label="active_x", polynomial=_poly("x")),
    )

    payload = run_append_ap_mclachlan_from_runtime_input(
        runtime_input,
        times=(0.0, 0.1),
        controller_config=AppendControllerConfig(
            max_append_candidates=0,
        ),
        solve_repair_config=SolveRepairConfig(
            enabled=True,
            condition_number_max=1.0e9,
            rho_num_max=1.0,
            state_motion_l2_step_max=2.0e-2,
            max_local_subdivisions=4,
            local_subdivision_factor=2,
            ridge_ladder=(1.0e-7,),
            pinv_rcond_ladder=(1.0e-10,),
            solve_damping_ladder=(0.0,),
        ),
    )

    row = payload["plot_rows"][0]
    integration = payload["trajectory"]["points"][0]["integration_to_next"]
    assert row["integration_local_subdivision_applied"] is True
    assert row["integration_local_subdivision_depth"] >= 3
    assert integration["repair_summary"]["local_subdivision_min_depth_requested"] >= 3
    assert integration["repair_summary"]["local_subdivision_severity"] > 1.0


def test_solve_repair_subdivides_for_nonlinear_trial_state_motion() -> None:
    selected = tuple(
        AnsatzTerm(label=f"active_{index}_{pauli}", polynomial=_poly(pauli))
        for index, pauli in enumerate(("x", "y", "x", "z"))
    )
    layout = build_parameter_layout(selected)
    theta = np.array(
        [0.77795273, 1.58657212, 2.34570360, -2.37495135],
        dtype=float,
    )
    psi_ref = np.array([1.0, 0.0], dtype=complex)
    executor = CompiledAnsatzExecutor(
        selected,
        parameterization_layout=layout,
        parameterization_mode="per_pauli_term",
    )
    runtime_input = ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(family_key="toy", hamiltonian=_poly("x")),
        psi_ref=psi_ref,
        psi_initial=executor.prepare_state(theta, psi_ref),
        base_layout=layout,
        theta_runtime=theta,
        theta_logical=theta,
        structure_locked=False,
        exact_energy=None,
        selected_terms=selected,
        candidate_pool_terms=selected,
        candidate_pool_source=CandidatePoolSource(
            source_kind="resolved_pool",
            pool_key="toy_pool",
            completeness="complete",
        ),
        provenance={"artifact_json": "toy_nonlinear_step.json"},
    )

    payload = run_append_ap_mclachlan_from_runtime_input(
        runtime_input,
        times=(0.0, 0.05),
        pinv_rcond=1.0e-12,
        ridge_lambda=1.0e-10,
        controller_config=AppendControllerConfig(max_append_candidates=0),
        # Pin the 4-parameter scenario: prune-only search now runs below the
        # residual threshold and would delete the ray-redundant z rotation.
        support_patch_config=SupportPatchControllerConfig(
            min_runtime_parameter_count=4,
        ),
        solve_repair_config=SolveRepairConfig(
            enabled=True,
            condition_number_max=1.0e12,
            rho_num_max=1.0,
            state_motion_l2_step_max=6.0e-2,
            state_space_kink_eta_max=1.0,
            max_local_subdivisions=6,
            local_subdivision_factor=2,
            ridge_ladder=(1.0e-10,),
            pinv_rcond_ladder=(1.0e-12,),
            solve_damping_ladder=(0.0,),
        ),
    )

    row = payload["plot_rows"][0]
    integration = payload["trajectory"]["points"][0]["integration_to_next"]
    assert integration["repair_summary"]["max_state_motion_l2_step"] < 6.0e-2
    assert integration["local_subdivision_applied"] is True
    assert integration["repair_summary"]["prospective_state_motion_triggered"] is True
    assert row["integration_prospective_state_motion_triggered"] is True
    assert (
        integration["repair_summary"]["max_prospective_state_motion_l2_step"]
        <= 6.0e-2
    )


def test_subdivision_severity_ignores_numerical_miss() -> None:
    step = SimpleNamespace(
        solve_repair_enabled=True,
        solve_guard_g_empty=False,
        solve_guard_g_delta=False,
        solve_guard_g_kink=False,
        state_space_kink_eta=None,
        state_motion_l2_step=1.0e-2,
        rho_num=1.0e6,
        solve_repair_response_schedule=SolveRepairResponseSchedule(
            active_lanes=("rho",),
            severity=1.0e12,
            breadth=40,
            inverse_policy_breadth=40,
            local_subdivision_breadth=0,
        ),
    )
    config = SolveRepairConfig(
        enabled=True,
        rho_num_max=1.0e-6,
        state_motion_l2_step_max=1.0e-1,
        state_space_kink_eta_max=1.0e-2,
    )

    assert _repair_severity_for_step(step, solve_repair_config=config) == 1.0
    assert _checkpoint_local_subdivision_request(step, solve_repair_config=config) is None


def test_coordinate_only_kink_diagnostic_is_not_runtime_code() -> None:
    root = Path("pipelines/time_dynamics/ap_mclachlan")
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))

    assert "J_coord" not in text
    assert "coordinate_only" not in text


def test_append_prune_trajectory_deletes_candidate_without_exact_reference() -> None:
    runtime_input = _runtime_input_with_selected(
        AnsatzTerm(label="active_x", polynomial=_poly("x")),
        AnsatzTerm(label="active_y", polynomial=_poly("y")),
    )

    payload = run_append_ap_mclachlan_from_runtime_input(
        runtime_input,
        times=(0.0, 0.1),
        controller_config=AppendControllerConfig(max_append_candidates=0),
        support_patch_config=SupportPatchControllerConfig(
            append_ladder_mode="combinatorial",
            residual_ratio_threshold=0.0,
            # Huge saved-cost pressure makes deletion-bearing candidates the
            # top structural scores; loose gates let the best one certify.
            prune_cost_alpha=1.0,
            prune_ray_distance_tol=1.0,
            prune_patch_smoothness_eta_max=1.0e6,
            min_runtime_parameter_count=1,
        ),
    )

    kinds = {row["patch_kind"] for row in payload["plot_rows"] if row["patch_accepted"]}
    assert kinds <= {"delete", "exchange", "append"}
    decision = payload["trajectory"]["points"][0]["patch_decision"]
    assert decision["metadata"]["selection_policy"].startswith(
        "paper_ii_deletion_conditioned"
    )
    assert payload["decision_data_flow"]["uses_exact_reference_for_decision"] is False
