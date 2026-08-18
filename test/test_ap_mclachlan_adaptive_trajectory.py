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
    APPEND_MACRO_SCOUT_SCORE_MODE_FULL_CHILD_BLOCK_DIAGNOSTIC,
    APPEND_MACRO_SCOUT_SCORE_MODE_PARENT_LINEAR_RESIDUAL_V1,
    PRUNE_PERSISTENCE_ATOM_HISTORY,
    PRUNE_PERSISTENCE_EXACT_BATCH,
    SUPPORT_PATCH_EXCHANGE_SELECTION_POLICY_V1,
    AppendControllerConfig,
    PatchCandidateScore,
    SolveRepairConfig,
    SupportPatchControllerConfig,
    _checkpoint_local_subdivision_request,
    _apply_append_cost_scores,
    _build_append_candidate_geometry_cache,
    _score_append_atom_set,
    _PruneControllerRuntimeState,
    _prune_persistence_metadata,
    _prune_persistence_passed,
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


def _poly_sum(*labels: str) -> PauliPolynomial:
    poly = PauliPolynomial("JW")
    for label in labels:
        poly.add_term(PauliTerm(1, ps=str(label), pc=1.0))
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


def _runtime_input_with_selected_and_candidates(
    selected: tuple[AnsatzTerm, ...],
    candidates: tuple[AnsatzTerm, ...],
    *,
    hamiltonian: PauliPolynomial | None = None,
) -> ScaffoldRuntimeInput:
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
        resolved_problem=SimpleNamespace(
            family_key="toy",
            hamiltonian=_poly("x") if hamiltonian is None else hamiltonian,
        ),
        psi_ref=psi_ref,
        psi_initial=psi_initial,
        base_layout=layout,
        theta_runtime=theta,
        theta_logical=np.zeros(int(layout.logical_parameter_count), dtype=float),
        structure_locked=False,
        exact_energy=None,
        selected_terms=tuple(selected),
        candidate_pool_terms=tuple(selected) + tuple(candidates),
        candidate_pool_source=CandidatePoolSource(
            source_kind="resolved_pool",
            pool_key="toy_pool",
            completeness="complete",
        ),
        provenance={"artifact_json": "toy.json"},
    )


def _candidate_atom(label: str) -> SupportAtom:
    poly = PauliPolynomial("JW")
    poly.add_term(PauliTerm(len(str(label)), ps=str(label), pc=1.0))
    poly._reduce()
    term = AnsatzTerm(label=f"candidate_{label}", polynomial=poly)
    return SupportAtom(
        atom_id=f"pauli:{label}",
        atom_label=str(term.label),
        parent_label=str(term.label),
        term=term,
        parameterization_mode="per_pauli_term",
        runtime_count=1,
        origin_kind="test",
    )


def _raw_append_score(label: str, insertion_gain: float, score_index: int) -> PatchCandidateScore:
    atom = _candidate_atom(label)
    return PatchCandidateScore(
        candidate_kind=PATCH_APPEND,
        candidate_label=str(atom.atom_label),
        patch=SupportPatch(inserted_count=1, inserted_labels=(str(atom.atom_label),)),
        score=SupportPatchScore(
            patch_kind=PATCH_APPEND,
            before_indices=(),
            after_indices_before_part=(),
            removed_runtime_indices=(),
            inserted_count=1,
            inserted_labels=(str(atom.atom_label),),
            before_gain=0.0,
            after_gain=float(insertion_gain),
            signed_delta_gain=float(insertion_gain),
            normalized_score=float(insertion_gain),
            insertion_gain=float(insertion_gain),
            deletion_loss=None,
            denominator=1.0,
            denominator_kind="test_denominator",
            support_kind="test_support",
            score_kind="test_support_patch_score",
            pinv_policy_id="test_inverse",
            pinv_rcond=1.0e-10,
            ridge_lambda=0.0,
            solve_damping=0.0,
            rank_before=0,
            rank_after=1,
            rank_score=float(insertion_gain),
        ),
        rank_score=float(insertion_gain),
        accepted_eligible=True,
        rejection_reason="eligible",
        metadata={
            "rung_size": 1,
            "atom_ids": [str(atom.atom_id)],
            "candidate_set_index": int(score_index),
            "score_index": int(score_index),
            "augmented_solve_confirmation_reason": "eligible",
            "schur_guard_reason": "eligible",
            "append_cost_raw": estimate_append_atom_set_cost((atom,)).to_json_dict(),
        },
    )


def _raw_prune_score(
    atom_ids: tuple[str, ...],
    *,
    persistence_atom_ids: tuple[str, ...] | None = None,
) -> PatchCandidateScore:
    return PatchCandidateScore(
        candidate_kind=PATCH_DELETE,
        candidate_label="delete_" + "_".join(atom_ids),
        patch=SupportPatch(removed_runtime_indices=tuple(range(len(atom_ids)))),
        score=None,
        rank_score=None,
        accepted_eligible=True,
        rejection_reason="eligible",
        metadata={
            "candidate_key": "|".join(atom_ids),
            "atom_ids": list(atom_ids),
            **(
                {}
                if persistence_atom_ids is None
                else {"persistence_atom_ids": list(persistence_atom_ids)}
            ),
        },
    )


def _strip_support_patch_scoring_workers(value):
    if isinstance(value, dict):
        return {
            str(key): _strip_support_patch_scoring_workers(item)
            for key, item in value.items()
            if str(key) != "support_patch_scoring_workers"
        }
    if isinstance(value, list):
        return [_strip_support_patch_scoring_workers(item) for item in value]
    return value


def _strip_append_geometry_implementation_mode(value):
    if isinstance(value, dict):
        return {
            str(key): _strip_append_geometry_implementation_mode(item)
            for key, item in value.items()
            if str(key) != "append_candidate_geometry_mode"
        }
    if isinstance(value, list):
        return [_strip_append_geometry_implementation_mode(item) for item in value]
    return value


def test_append_finalist_prefers_lower_ranked_eligible_candidate() -> None:
    rejected = replace(
        _raw_append_score("x", insertion_gain=2.0, score_index=0),
        accepted_eligible=False,
        rejection_reason="append_schur_rank_deficient",
    )
    eligible = _raw_append_score("y", insertion_gain=1.0, score_index=1)

    selected = aptraj._preferred_append_finalist_branches(
        SimpleNamespace(candidate_scores=(rejected, eligible)),
        score_min=0.0,
    )

    assert selected == (eligible,)


def test_append_finalist_preserves_top_rejection_when_none_are_eligible() -> None:
    top_rejected = replace(
        _raw_append_score("x", insertion_gain=2.0, score_index=0),
        accepted_eligible=False,
        rejection_reason="append_schur_rank_deficient",
    )
    lower_rejected = replace(
        _raw_append_score("y", insertion_gain=1.0, score_index=1),
        accepted_eligible=False,
        rejection_reason="append_augmented_solve_not_confirmed",
    )

    selected = aptraj._preferred_append_finalist_branches(
        SimpleNamespace(candidate_scores=(top_rejected, lower_rejected)),
        score_min=0.0,
    )

    assert selected == (top_rejected,)
    assert selected[0].rejection_reason == "append_schur_rank_deficient"


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


def test_prune_atom_history_persistence_allows_rotating_batches() -> None:
    exact_state = _PruneControllerRuntimeState()
    exact_config = SupportPatchControllerConfig(
        prune_persistence_required=2,
        prune_persistence_mode=PRUNE_PERSISTENCE_EXACT_BATCH,
    )
    first_exact = _prune_persistence_metadata(
        exact_state,
        _raw_prune_score(("a", "b", "c")),
        support_config=exact_config,
        time_index=0,
    )
    second_exact = _prune_persistence_metadata(
        exact_state,
        _raw_prune_score(("a", "b", "d")),
        support_config=exact_config,
        time_index=1,
    )

    assert first_exact["prune_persistence_count"] == 1
    assert second_exact["prune_persistence_count"] == 1
    assert _prune_persistence_passed(second_exact) is False

    atom_state = _PruneControllerRuntimeState()
    atom_config = SupportPatchControllerConfig(
        prune_persistence_required=2,
        prune_persistence_mode=PRUNE_PERSISTENCE_ATOM_HISTORY,
        prune_atom_history_fraction=0.5,
        prune_history_window=3,
    )
    first_atom = _prune_persistence_metadata(
        atom_state,
        _raw_prune_score(("a", "b", "c")),
        support_config=atom_config,
        time_index=0,
    )
    second_atom = _prune_persistence_metadata(
        atom_state,
        _raw_prune_score(("a", "b", "d")),
        support_config=atom_config,
        time_index=1,
    )

    assert first_atom["prune_atom_history_pass_count"] == 0
    assert _prune_persistence_passed(first_atom) is False
    assert second_atom["prune_atom_history_pass_count"] == 2
    assert second_atom["prune_atom_history_total_count"] == 3
    assert second_atom["prune_atom_history_fraction"] >= 0.5
    assert _prune_persistence_passed(second_atom) is True


def test_prune_atom_history_aggregates_repeated_occurrences_by_base_atom() -> None:
    runtime_state = _PruneControllerRuntimeState()
    config = SupportPatchControllerConfig(
        prune_persistence_required=2,
        prune_persistence_mode=PRUNE_PERSISTENCE_ATOM_HISTORY,
        prune_atom_history_fraction=1.0,
        prune_history_window=3,
    )
    first = _prune_persistence_metadata(
        runtime_state,
        _raw_prune_score(
            ("pauli:g::occ1",),
            persistence_atom_ids=("pauli:g",),
        ),
        support_config=config,
        time_index=0,
    )
    second = _prune_persistence_metadata(
        runtime_state,
        _raw_prune_score(
            ("pauli:g::occ2",),
            persistence_atom_ids=("pauli:g",),
        ),
        support_config=config,
        time_index=1,
    )

    assert _prune_persistence_passed(first) is False
    assert second["prune_atom_history_counts"] == {"pauli:g": 2}
    assert _prune_persistence_passed(second) is True


def test_prune_atom_history_survives_append_only_support_change() -> None:
    base_state = state_from_scaffold_runtime_input(
        _runtime_input_with_selected(
            AnsatzTerm(label="seed_x", polynomial=_poly("x")),
            AnsatzTerm(label="seed_z", polynomial=_poly("z")),
        )
    )
    appended_state = state_from_scaffold_runtime_input(
        _runtime_input_with_selected(
            AnsatzTerm(label="seed_x", polynomial=_poly("x")),
            AnsatzTerm(label="seed_z", polynomial=_poly("z")),
            AnsatzTerm(label="candidate_y", polynomial=_poly("y")),
        )
    )
    atom_ids = tuple(str(atom.atom_id) for atom in active_support_atoms(base_state))
    runtime_state = _PruneControllerRuntimeState()
    runtime_state.atom_seen_history[atom_ids[0]] = [0]
    runtime_state.cooldown_until_index[atom_ids[0]] = 5
    runtime_state.loss_history["old_batch"] = [(0, 0.01)]
    runtime_state.conditioning_history["old_batch"] = [(0, 0.2)]
    runtime_state.eligible_streak["old_batch"] = 2
    runtime_state.last_seen_index["old_batch"] = 0
    runtime_state.smoothness_deferred["old_batch"] = object()  # type: ignore[assignment]

    metadata = runtime_state.update_after_support_change(
        new_state=appended_state,
        theta_runtime=appended_state.theta_runtime,
        patch_kind=PATCH_APPEND,
    )
    persistence = _prune_persistence_metadata(
        runtime_state,
        _raw_prune_score((atom_ids[0],)),
        support_config=SupportPatchControllerConfig(
            prune_persistence_required=2,
            prune_persistence_mode=PRUNE_PERSISTENCE_ATOM_HISTORY,
            prune_history_window=3,
        ),
        time_index=1,
    )

    assert metadata["prune_history_transition"] == "append_preserved_atom_history"
    assert metadata["prune_atom_history_preserved_count"] == 1
    assert metadata["prune_atom_history_dropped_count"] == 0
    assert runtime_state.cooldown_until_index[atom_ids[0]] == 5
    assert runtime_state.loss_history == {}
    assert runtime_state.conditioning_history == {}
    assert runtime_state.eligible_streak == {}
    assert runtime_state.last_seen_index == {}
    assert runtime_state.smoothness_deferred == {}
    assert persistence["prune_atom_history_counts"][atom_ids[0]] == 2
    assert _prune_persistence_passed(persistence) is True


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


def test_zero_angle_append_geometry_cache_matches_full_augmented_recompute() -> None:
    runtime_input = _runtime_input_with_selected_and_candidates(
        selected=(AnsatzTerm(label="seed_y", polynomial=_poly("y")),),
        candidates=(
            AnsatzTerm(label="candidate_x", polynomial=_poly("x", coeff=0.7)),
            AnsatzTerm(label="candidate_z", polynomial=_poly("z", coeff=-0.4)),
        ),
        hamiltonian=_poly_sum("x", "z"),
    )
    state = state_from_scaffold_runtime_input(
        runtime_input,
        parameterization_mode="per_pauli_term",
    )
    theta = np.array([0.37], dtype=float)
    hamiltonian = time_dependent_hamiltonian_from_runtime_input(runtime_input)
    base_evaluation = evaluate_mclachlan_geometry(
        state=state,
        hamiltonian=hamiltonian,
        theta_runtime=theta,
        time=0.2,
        include_tangent_matrix=True,
    )
    atoms = candidate_append_atoms(state)
    assert len(atoms) == 2
    inverse_policy = McLachlanInversePolicy(
        pinv_rcond=1.0e-10,
        ridge_lambda=1.0e-7,
        solve_damping=0.0,
    )
    support_config = SupportPatchControllerConfig(
        append_gain_threshold=0.0,
        append_batch_score_threshold=0.0,
        cost_required_for_decisions=False,
    )
    before_cache = build_support_patch_before_cache(
        geometry=SupportPatchGeometry(
            K_before=base_evaluation.geometry.K,
            f_before=base_evaluation.geometry.f,
            norm_b_sq=base_evaluation.geometry.norm_b_sq,
        ),
        inverse_policy=inverse_policy,
    )
    candidate_cache = _build_append_candidate_geometry_cache(
        state=state,
        base_evaluation=base_evaluation,
        atoms=atoms,
        schur_inverse_policy=aptraj._append_schur_inverse_policy(
            inverse_policy,
            support_config=support_config,
        ),
    )
    assert candidate_cache is not None

    for atom_set in ((atoms[0],), atoms):
        common = {
            "state": state,
            "hamiltonian": hamiltonian,
            "theta_runtime": theta,
            "time": 0.2,
            "base_K": base_evaluation.geometry.K,
            "base_f": base_evaluation.geometry.f,
            "norm_b_sq": base_evaluation.geometry.norm_b_sq,
            "n_before": state.runtime_parameter_count,
            "atoms": atom_set,
            "inverse_policy": inverse_policy,
            "support_config": support_config,
            "candidate_set_index": 0,
            "score_index": 0,
            "before_cache": before_cache,
        }
        full = _score_append_atom_set(**common, candidate_geometry_cache=None)
        cached = _score_append_atom_set(
            **common,
            candidate_geometry_cache=candidate_cache,
        )

        assert cached.patch == full.patch
        assert cached.accepted_eligible is full.accepted_eligible
        assert cached.rejection_reason == full.rejection_reason
        assert cached.rank_score == pytest.approx(full.rank_score, abs=1.0e-12)
        assert cached.score is not None
        assert full.score is not None
        assert cached.score.insertion_gain == pytest.approx(
            full.score.insertion_gain,
            abs=1.0e-12,
        )
        assert cached.score.schur_novelty is not None
        assert full.score.schur_novelty is not None
        assert np.allclose(
            cached.score.schur_novelty.matrix,
            full.score.schur_novelty.matrix,
            atol=1.0e-12,
            rtol=0.0,
        )
        cached_confirmation = cached.score.augmented_solve_confirmation
        full_confirmation = full.score.augmented_solve_confirmation
        assert cached_confirmation is not None
        assert full_confirmation is not None
        assert cached_confirmation.confirmed is full_confirmation.confirmed
        assert cached_confirmation.gamma == pytest.approx(
            full_confirmation.gamma,
            abs=1.0e-12,
        )
        assert cached_confirmation.residual_ratio == pytest.approx(
            full_confirmation.residual_ratio,
            abs=1.0e-12,
        )


def test_parallel_support_patch_scoring_matches_serial_exchange_family(monkeypatch) -> None:
    runtime_input = _runtime_input_with_selected_and_candidates(
        selected=(AnsatzTerm(label="seed_x", polynomial=_poly("x")),),
        candidates=(AnsatzTerm(label="candidate_y", polynomial=_poly("y")),),
        hamiltonian=_poly("y"),
    )

    def run(workers: int):
        return run_append_ap_mclachlan_from_runtime_input(
            runtime_input,
            times=(0.0, 0.1),
            support_patch_config=SupportPatchControllerConfig(
                append_ladder_mode="combinatorial",
                max_append_batch_size=1,
                append_rung_set_cap=0,
                append_prefilter_size=0,
                append_gain_threshold=0.0,
                append_batch_score_threshold=0.0,
                residual_ratio_threshold=0.0,
                prune_enabled=True,
                prune_commit_enabled=False,
                max_prune_batch_size=1,
                prune_loss_threshold=10.0,
                min_runtime_parameter_count=0,
                prune_shadow_enabled=False,
                max_prune_commits=1,
                exchange_enabled=True,
                max_exchange_append_branches=3,
                max_exchange_prune_branches=3,
                support_patch_scoring_workers=workers,
            ),
        )

    serial = _strip_support_patch_scoring_workers(run(1))
    parallel = _strip_support_patch_scoring_workers(run(2))

    assert parallel["trajectory"]["points"][0]["patch_decision"] == serial[
        "trajectory"
    ]["points"][0]["patch_decision"]
    assert parallel["plot_rows"] == serial["plot_rows"]
    assert parallel["summary"] == serial["summary"]
    assert (
        parallel["decision_data_flow"]["uses_exact_reference_for_decision"]
        is False
    )

    monkeypatch.setattr(
        aptraj,
        "_build_append_candidate_geometry_cache",
        lambda **_kwargs: None,
    )
    compatibility = _strip_support_patch_scoring_workers(run(1))
    assert _strip_append_geometry_implementation_mode(serial) == (
        _strip_append_geometry_implementation_mode(compatibility)
    )


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


def test_append_cost_weighting_can_flip_raw_gain_ranking() -> None:
    scores = [
        _raw_append_score("xz", insertion_gain=1.2, score_index=0),
        _raw_append_score("x", insertion_gain=1.0, score_index=1),
    ]
    config = SupportPatchControllerConfig(
        append_ladder_mode="combinatorial",
        append_gain_threshold=0.0,
        append_batch_score_threshold=0.0,
        cost_normalization_mode=AP_APPEND_COST_NORMALIZATION_RAW_LEGACY_V1,
        append_cost_lambda_2q=1.0,
        append_cost_lambda_d=1.0,
        append_cost_lambda_1q=0.0,
        append_cost_lambda_theta=0.0,
        append_cost_lambda_shot=0.0,
    )

    assert scores[0].score is not None
    assert scores[1].score is not None
    assert scores[0].score.insertion_gain > scores[1].score.insertion_gain

    _apply_append_cost_scores(scores, support_config=config)

    assert scores[1].rank_score > scores[0].rank_score
    assert scores[1].metadata["append_cost"]["rank_utility"] == scores[1].rank_score
    assert scores[0].score.insertion_gain == 1.2
    assert scores[0].score.rank_score == 1.2


def test_prune_patch_smoothness_defers_large_state_velocity_jump() -> None:
    geometry = McLachlanGeometry(
        K=np.array([[1.0]]),
        f=np.array([0.0]),
        norm_b_sq=1.0,
        support_indices=(0,),
        support_labels=("a",),
        time=0.0,
    )
    policy = McLachlanInversePolicy()
    base_step = FixedMcLachlanStep(
        theta_dot=np.array([1.0]),
        gamma=0.0,
        residual_sq=1.0,
        residual_ratio=1.0,
        rank=1,
        condition_number=1.0,
        geometry=geometry,
        inverse_policy=policy,
    )
    patched_step = FixedMcLachlanStep(
        theta_dot=np.array([3.0]),
        gamma=0.0,
        residual_sq=1.0,
        residual_ratio=1.0,
        rank=1,
        condition_number=1.0,
        geometry=geometry,
        inverse_policy=policy,
    )
    psi = np.array([1.0, 0.0], dtype=complex)
    base_eval = GeometryEvaluation(
        geometry=geometry,
        psi=psi,
        h_psi=np.zeros(2, dtype=complex),
        energy_expectation=0.0,
        theta_runtime=np.array([0.0]),
        tangent_matrix=np.array([[0.0], [-1.0j]], dtype=complex),
    )
    patched_eval = GeometryEvaluation(
        geometry=geometry,
        psi=psi,
        h_psi=np.zeros(2, dtype=complex),
        energy_expectation=0.0,
        theta_runtime=np.array([0.0]),
        tangent_matrix=np.array([[0.0], [1.0j]], dtype=complex),
    )
    config = SupportPatchControllerConfig(
        append_ladder_mode="combinatorial",
        prune_enabled=True,
        prune_commit_enabled=True,
        max_prune_batch_size=1,
        prune_patch_smoothness_eta_max=1.0e-2,
        prune_cooldown_steps=2,
        prune_patch_smoothness_cooldown_max_steps=8,
    )

    smoothness = aptraj._evaluate_prune_patch_smoothness(
        base_evaluation=base_eval,
        base_step=base_step,
        patched_evaluation=patched_eval,
        patched_step=patched_step,
        support_config=config,
    )

    assert smoothness.available is True
    assert smoothness.passed is False
    assert smoothness.defer is True
    assert smoothness.eta is not None and smoothness.eta > config.prune_patch_smoothness_eta_max
    assert aptraj._prune_smoothness_cooldown_steps(
        smoothness,
        support_config=config,
    ) == 8

    runtime_state = _PruneControllerRuntimeState()
    atom = ActiveSupportAtom(
        atom_id="pauli:a",
        atom_label="a",
        parent_label="parent",
        term=None,
        parameterization_mode="per_pauli_term",
        runtime_count=1,
        origin_kind="test",
        runtime_indices=(0,),
        theta_values=(0.1,),
    )
    record = aptraj._record_prune_smoothness_deferred(
        runtime_state,
        (atom,),
        time_index=4,
        cooldown_steps=8,
        smoothness=smoothness,
    )

    assert record.attempt_count == 1
    assert record.cooldown_until_index == 12
    assert runtime_state.smoothness_deferred[record.candidate_key] is record
