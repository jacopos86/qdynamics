from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.contracts.scaffold import CandidatePoolSource, ScaffoldRuntimeInput
from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import (
    AppendControllerConfig,
    LEGACY_APPEND_CONTROLLER_PROFILE_V1,
    SupportPatchControllerConfig,
)
from pipelines.time_dynamics.diagnostics.ap_pool_accounting import (
    build_pool_accounting_audit,
)
from pipelines.time_dynamics.ap_mclachlan.state import (
    AP_PARAMETERIZATION_LOGICAL_SHARED,
    AP_PARAMETERIZATION_PER_PAULI_TERM,
    runtime_coordinate_records,
    state_from_scaffold_runtime_input,
    state_with_runtime_coordinate_patch,
    state_without_runtime_indices,
)
from pipelines.time_dynamics.ap_mclachlan.support_atoms import (
    APPEND_OCCURRENCE_POLICY_LAYER_REUSE,
    APPEND_OCCURRENCE_POLICY_UNIQUE_SUPPORT,
    active_support_atoms,
    append_occurrence_base_label,
    candidate_append_atoms,
    state_with_appended_atoms,
    state_without_active_atoms,
)
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _poly(labels: tuple[tuple[str, float], ...], *, nq: int = 1) -> PauliPolynomial:
    poly = PauliPolynomial("JW")
    for label, coeff in labels:
        poly.add_term(PauliTerm(int(nq), ps=str(label), pc=float(coeff)))
    poly._reduce()
    return poly


def _runtime_input(
    *,
    selected: tuple[AnsatzTerm, ...],
    theta_runtime: np.ndarray,
    candidate_pool_terms: tuple[AnsatzTerm, ...] = (),
    parameterization_mode: str = AP_PARAMETERIZATION_PER_PAULI_TERM,
    candidate_pool_complete: bool = True,
    candidate_pool_filter_payload: dict | None = None,
) -> ScaffoldRuntimeInput:
    layout = build_parameter_layout(selected)
    theta_logical = np.zeros(int(layout.logical_parameter_count), dtype=float)
    if parameterization_mode == AP_PARAMETERIZATION_LOGICAL_SHARED:
        theta_for_executor = theta_logical
    else:
        theta_for_executor = np.asarray(theta_runtime, dtype=float).reshape(-1)
        for block in layout.blocks:
            if int(block.runtime_count) > 0:
                vals = theta_for_executor[int(block.runtime_start) : int(block.runtime_stop)]
                theta_logical[int(block.logical_index)] = float(np.mean(vals))
    psi_ref = np.array([1.0, 0.0], dtype=complex)
    executor = CompiledAnsatzExecutor(
        selected,
        parameterization_mode=parameterization_mode,
        parameterization_layout=layout,
    )
    psi_initial = executor.prepare_state(theta_for_executor, psi_ref)
    return ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(family_key="toy", hamiltonian=_poly((("z", 1.0),))),
        psi_ref=psi_ref,
        psi_initial=psi_initial,
        base_layout=layout,
        theta_runtime=np.asarray(theta_runtime, dtype=float).reshape(-1),
        theta_logical=theta_logical,
        structure_locked=False,
        exact_energy=None,
        selected_terms=selected,
        candidate_pool_terms=candidate_pool_terms,
        candidate_pool_source=CandidatePoolSource(
            source_kind="resolved_pool",
            pool_key="toy_pool",
            completeness="complete" if candidate_pool_complete else "partial",
            filter_payload=dict(candidate_pool_filter_payload or {}),
        ),
        provenance={"artifact_json": "toy.json"},
    )


def test_per_pauli_support_atoms_use_child_coordinates() -> None:
    selected = (
        AnsatzTerm(
            label="seed_xz",
            polynomial=_poly((("x", 1.0), ("z", 0.5))),
        ),
    )
    candidates = (
        AnsatzTerm(
            label="candidate_yz",
            polynomial=_poly((("y", 1.0), ("z", -0.25))),
        ),
    )
    state = state_from_scaffold_runtime_input(
        _runtime_input(
            selected=selected,
            theta_runtime=np.array([0.0, 0.0]),
            candidate_pool_terms=candidates,
        )
    )

    records = runtime_coordinate_records(state)
    assert [record.runtime_label for record in records] == [
        "seed_xz::r0::x",
        "seed_xz::r1::z",
    ]
    active = active_support_atoms(state)
    assert [atom.atom_label for atom in active] == [
        "seed_xz::r0::x",
        "seed_xz::r1::z",
    ]
    assert [atom.runtime_indices for atom in active] == [(0,), (1,)]

    candidate_atoms = candidate_append_atoms(state)
    assert [atom.atom_label for atom in candidate_atoms] == [
        "candidate_yz::r0::y",
        "candidate_yz::r1::z",
    ]
    assert all(atom.runtime_count == 1 for atom in candidate_atoms)


def test_pool_accounting_distinguishes_parents_children_available_and_sidecar_labels() -> None:
    selected = (
        AnsatzTerm(
            label="candidate_xy",
            polynomial=_poly((("x", 1.0),)),
        ),
    )
    candidates = (
        AnsatzTerm(
            label="candidate_xy",
            polynomial=_poly((("x", 1.0), ("y", -0.5))),
        ),
        AnsatzTerm(
            label="candidate_z",
            polynomial=_poly((("z", 0.25),)),
        ),
    )
    state = state_from_scaffold_runtime_input(
        _runtime_input(
            selected=selected,
            theta_runtime=np.array([0.0]),
            candidate_pool_terms=candidates,
        )
    )
    paper_i_sidecar_payload = {
        "result": {
            "pool_pauli_labels_exyz": {
                "candidate_xy": ["x", "y"],
                "candidate_xy::r0::x": ["x"],
                "candidate_xy::r1::y": ["y"],
                "candidate_z::r0::z": ["z"],
            }
        }
    }

    audit = build_pool_accounting_audit(
        state,
        paper_i_pool_payload=paper_i_sidecar_payload,
    )

    counts = audit["counts"]
    assert counts["selected_seed_terms"] == 1
    assert counts["runtime_parameter_count"] == 1
    assert counts["candidate_parent_pool_terms_after_loader"] == 2
    assert counts["all_pauli_child_atoms"] == 3
    assert counts["active_pauli_child_atoms"] == 1
    assert counts["available_append_atoms"] == 2
    assert counts["reusable_append_frontier_atoms"] == 3
    assert counts["paper_i_sidecar_pool_labels_raw"] == 4
    assert counts["paper_i_sidecar_single_child_labels"] == 3
    assert counts["paper_i_sidecar_multi_child_labels"] == 1

    all_vs_sidecar = audit["comparisons"][
        "ap_all_children_vs_paper_i_single_child_pauli_multiset"
    ]
    assert all_vs_sidecar["common_representation"] == (
        "pauli_exyz_multiset_without_coefficients"
    )
    assert all_vs_sidecar["count_match"] is True
    assert all_vs_sidecar["digest_match"] is True
    assert all_vs_sidecar["unique_pauli_set_digest_match"] is True

    available_vs_sidecar = audit["comparisons"][
        "ap_available_children_vs_paper_i_single_child_pauli_multiset"
    ]
    assert available_vs_sidecar["count_match"] is False
    assert available_vs_sidecar["only_right_occurrences"] == 1


def test_logical_shared_support_atoms_use_macro_generators() -> None:
    selected = (
        AnsatzTerm(
            label="seed_xz",
            polynomial=_poly((("x", 1.0), ("z", 0.5))),
        ),
    )
    candidates = (
        AnsatzTerm(
            label="candidate_yz",
            polynomial=_poly((("y", 1.0), ("z", -0.25))),
        ),
    )
    state = state_from_scaffold_runtime_input(
        _runtime_input(
            selected=selected,
            theta_runtime=np.array([0.0, 0.0]),
            candidate_pool_terms=candidates,
            parameterization_mode=AP_PARAMETERIZATION_LOGICAL_SHARED,
        ),
        parameterization_mode=AP_PARAMETERIZATION_LOGICAL_SHARED,
    )

    active = active_support_atoms(state)
    assert [atom.atom_label for atom in active] == ["seed_xz"]
    assert active[0].runtime_indices == (0,)
    assert active[0].metadata["runtime_child_count"] == 2

    candidate_atoms = candidate_append_atoms(state)
    assert [atom.atom_label for atom in candidate_atoms] == ["candidate_yz"]
    assert candidate_atoms[0].runtime_count == 1


def test_legal_subspace_guard_blocks_pauli_split_but_allows_macro_candidate() -> None:
    selected = (AnsatzTerm(label="seed_x", polynomial=_poly((("x", 1.0),))),)
    grouped_candidate = AnsatzTerm(
        label="grouped_safe_macro",
        polynomial=_poly((("y", 1.0), ("z", 0.5))),
    )
    filter_payload = {
        "legal_subspace_append_guard": {
            "no_pauli_split_parent_labels": ["grouped_safe_macro"],
        }
    }
    per_pauli_state = state_from_scaffold_runtime_input(
        _runtime_input(
            selected=selected,
            theta_runtime=np.array([0.0]),
            candidate_pool_terms=(grouped_candidate,),
            candidate_pool_filter_payload=filter_payload,
        )
    )

    assert candidate_append_atoms(per_pauli_state) == ()

    logical_state = state_from_scaffold_runtime_input(
        _runtime_input(
            selected=selected,
            theta_runtime=np.array([0.0]),
            candidate_pool_terms=(grouped_candidate,),
            parameterization_mode=AP_PARAMETERIZATION_LOGICAL_SHARED,
            candidate_pool_filter_payload=filter_payload,
        ),
        parameterization_mode=AP_PARAMETERIZATION_LOGICAL_SHARED,
    )

    logical_atoms = candidate_append_atoms(logical_state)
    assert [atom.atom_label for atom in logical_atoms] == ["grouped_safe_macro"]
    assert logical_atoms[0].runtime_count == 1


def test_logical_shared_inserted_atom_uses_canonical_runtime_label() -> None:
    selected = (AnsatzTerm(label="seed_x", polynomial=_poly((("x", 1.0),))),)
    candidate = AnsatzTerm(
        label="candidate_yz",
        polynomial=_poly((("y", 1.0), ("z", 0.5))),
    )
    state = state_from_scaffold_runtime_input(
        _runtime_input(
            selected=selected,
            theta_runtime=np.array([0.0]),
            candidate_pool_terms=(candidate,),
            parameterization_mode=AP_PARAMETERIZATION_LOGICAL_SHARED,
        ),
        parameterization_mode=AP_PARAMETERIZATION_LOGICAL_SHARED,
    )
    atom = candidate_append_atoms(state)[0]

    appended_state, theta_aug = state_with_appended_atoms(
        state,
        (atom,),
        theta_runtime=state.theta_runtime,
    )

    assert appended_state.runtime_coordinate_labels[-1] == (
        "candidate_yz::logical::generator"
    )
    assert theta_aug.shape == (2,)
    np.testing.assert_allclose(
        appended_state.prepare_state(theta_aug),
        appended_state.psi_initial,
    )


def test_state_without_runtime_indices_deletes_one_pauli_child() -> None:
    selected = (
        AnsatzTerm(
            label="seed_xz",
            polynomial=_poly((("x", 1.0), ("z", 0.5))),
        ),
    )
    state = state_from_scaffold_runtime_input(
        _runtime_input(
            selected=selected,
            theta_runtime=np.array([0.2, -0.1]),
        )
    )

    pruned_state, pruned_theta = state_without_runtime_indices(
        state,
        (0,),
        theta_runtime=state.theta_runtime,
    )

    assert state.runtime_coordinate_labels == ("seed_xz::r0::x", "seed_xz::r1::z")
    assert pruned_theta.tolist() == [-0.1]
    assert pruned_state.runtime_coordinate_labels == ("seed_xz::r1::z",)
    assert pruned_state.runtime_parameter_count == 1
    np.testing.assert_allclose(pruned_state.prepare_state(pruned_theta), pruned_state.psi_initial)


def test_state_without_active_atoms_deletes_pauli_child_batch() -> None:
    selected = (
        AnsatzTerm(
            label="seed_xyz",
            polynomial=_poly((("x", 1.0), ("y", 0.25), ("z", 0.5))),
        ),
    )
    state = state_from_scaffold_runtime_input(
        _runtime_input(
            selected=selected,
            theta_runtime=np.array([0.2, 0.3, -0.1]),
        )
    )
    atoms = active_support_atoms(state, state.theta_runtime)

    pruned_state, pruned_theta = state_without_active_atoms(
        state,
        (atoms[0], atoms[2]),
        theta_runtime=state.theta_runtime,
    )

    assert pruned_theta.tolist() == [0.3]
    assert pruned_state.runtime_coordinate_labels == ("seed_xyz::r1::y",)
    assert pruned_state.runtime_parameter_count == 1
    np.testing.assert_allclose(
        pruned_state.prepare_state(pruned_theta),
        pruned_state.psi_initial,
    )


def test_state_with_support_patch_atoms_deletes_and_inserts_children() -> None:
    selected = (
        AnsatzTerm(
            label="seed_xz",
            polynomial=_poly((("x", 1.0), ("z", 0.5))),
        ),
    )
    candidates = (AnsatzTerm(label="candidate_y", polynomial=_poly((("y", 1.0),))),)
    state = state_from_scaffold_runtime_input(
        _runtime_input(
            selected=selected,
            theta_runtime=np.array([0.2, -0.1]),
            candidate_pool_terms=candidates,
        )
    )
    candidate_atoms = candidate_append_atoms(state)

    patched_state, patched_theta = state_with_runtime_coordinate_patch(
        state,
        removed_runtime_indices=(0,),
        inserted_coordinate_terms=(candidate_atoms[0].term,),
        inserted_coordinate_labels=(candidate_atoms[0].atom_label,),
        theta_runtime=state.theta_runtime,
    )

    assert patched_theta.tolist() == [-0.1, 0.0]
    assert patched_state.runtime_coordinate_labels == (
        "seed_xz::r1::z",
        "candidate_y::r0::y",
    )
    np.testing.assert_allclose(
        patched_state.prepare_state(patched_theta),
        patched_state.psi_initial,
    )


def test_state_with_appended_atoms_appends_zero_child_coordinate() -> None:
    selected = (AnsatzTerm(label="seed_x", polynomial=_poly((("x", 1.0),))),)
    candidates = (AnsatzTerm(label="candidate_y", polynomial=_poly((("y", 1.0),))),)
    state = state_from_scaffold_runtime_input(
        _runtime_input(
            selected=selected,
            theta_runtime=np.array([0.1]),
            candidate_pool_terms=candidates,
        )
    )
    atom = candidate_append_atoms(state)[0]

    appended_state, appended_theta = state_with_appended_atoms(
        state,
        (atom,),
        theta_runtime=state.theta_runtime,
    )

    assert appended_theta.tolist() == [0.1, 0.0]
    assert appended_state.runtime_coordinate_labels == (
        "seed_x::r0::x",
        "candidate_y::r0::y",
    )
    np.testing.assert_allclose(appended_state.psi_initial, state.prepare_state())
    assert candidate_append_atoms(appended_state) == ()


def test_layer_reuse_appends_same_base_atom_at_later_ansatz_positions() -> None:
    selected = (AnsatzTerm(label="seed_x", polynomial=_poly((("x", 1.0),))),)
    candidates = (AnsatzTerm(label="candidate_y", polynomial=_poly((("y", 1.0),))),)
    state = state_from_scaffold_runtime_input(
        _runtime_input(
            selected=selected,
            theta_runtime=np.array([0.1]),
            candidate_pool_terms=candidates,
        )
    )

    first = candidate_append_atoms(
        state,
        occurrence_policy=APPEND_OCCURRENCE_POLICY_LAYER_REUSE,
    )[0]
    state_1, theta_1 = state_with_appended_atoms(
        state,
        (first,),
        theta_runtime=state.theta_runtime,
    )
    second = candidate_append_atoms(
        state_1,
        occurrence_policy=APPEND_OCCURRENCE_POLICY_LAYER_REUSE,
    )[0]
    state_2, theta_2 = state_with_appended_atoms(
        state_1,
        (second,),
        theta_runtime=theta_1,
    )

    assert first.atom_label == "candidate_y::r0::y"
    assert second.atom_label == "candidate_y::ap_occ2::r0::y"
    assert second.metadata["base_atom_id"] == first.metadata["base_atom_id"]
    assert second.metadata["occurrence_index"] == 2
    assert state_2.runtime_coordinate_labels[-2:] == (
        "candidate_y::r0::y",
        "candidate_y::ap_occ2::r0::y",
    )
    active_y = [
        atom
        for atom in active_support_atoms(state_2, theta_2)
        if atom.metadata["base_atom_id"] == "pauli:candidate_y::r0::y"
    ]
    assert [atom.metadata["occurrence_index"] for atom in active_y] == [1, 2]
    assert append_occurrence_base_label(second.atom_label) == first.atom_label


def test_layer_reuse_occurrence_counter_survives_deletion() -> None:
    selected = (AnsatzTerm(label="seed_x", polynomial=_poly((("x", 1.0),))),)
    candidates = (AnsatzTerm(label="candidate_y", polynomial=_poly((("y", 1.0),))),)
    state = state_from_scaffold_runtime_input(
        _runtime_input(
            selected=selected,
            theta_runtime=np.array([0.1]),
            candidate_pool_terms=candidates,
        )
    )
    first = candidate_append_atoms(
        state,
        occurrence_policy=APPEND_OCCURRENCE_POLICY_LAYER_REUSE,
    )[0]
    appended, theta = state_with_appended_atoms(
        state,
        (first,),
        theta_runtime=state.theta_runtime,
    )
    active_first = next(
        atom for atom in active_support_atoms(appended, theta) if atom.atom_label == first.atom_label
    )
    pruned, pruned_theta = state_without_active_atoms(
        appended,
        (active_first,),
        theta_runtime=theta,
    )

    next_atom = candidate_append_atoms(
        pruned,
        occurrence_policy=APPEND_OCCURRENCE_POLICY_LAYER_REUSE,
    )[0]

    assert pruned_theta.shape == (1,)
    assert next_atom.atom_label == "candidate_y::ap_occ2::r0::y"


def test_append_batch_rejects_duplicate_base_atom_occurrences() -> None:
    selected = (AnsatzTerm(label="seed_x", polynomial=_poly((("x", 1.0),))),)
    candidates = (AnsatzTerm(label="candidate_y", polynomial=_poly((("y", 1.0),))),)
    state = state_from_scaffold_runtime_input(
        _runtime_input(
            selected=selected,
            theta_runtime=np.array([0.1]),
            candidate_pool_terms=candidates,
        )
    )
    atom = candidate_append_atoms(
        state,
        occurrence_policy=APPEND_OCCURRENCE_POLICY_LAYER_REUSE,
    )[0]

    with pytest.raises(ValueError, match="each base support atom at most once"):
        state_with_appended_atoms(
            state,
            (atom, atom),
            theta_runtime=state.theta_runtime,
        )


def test_insertion_named_aliases_for_append_are_gone() -> None:
    """``*_inserted_*`` names must never again be silent append aliases.

    The support-atoms alias stays removed.  The state-module name now exists
    as *real* positional insertion (2026-08-15); the functional check pins
    that a cut-0 insertion lands first, which the old append alias could not
    do.
    """

    import pipelines.time_dynamics.ap_mclachlan.state as state_mod
    import pipelines.time_dynamics.ap_mclachlan.support_atoms as atoms_mod

    assert not hasattr(atoms_mod, "state_with_inserted_atoms")
    assert hasattr(state_mod, "state_with_inserted_runtime_coordinates")

    selected = (AnsatzTerm(label="seed_x", polynomial=_poly((("x", 1.0),))),)
    state = state_from_scaffold_runtime_input(
        _runtime_input(selected=selected, theta_runtime=np.array([0.1]))
    )
    inserted_state, theta = state_mod.state_with_inserted_runtime_coordinates(
        state,
        insertions=(
            (0, AnsatzTerm(label="front", polynomial=_poly((("y", 1.0),))), "front::r0::y"),
        ),
    )
    assert inserted_state.runtime_coordinate_labels[0] == "front::r0::y"
    assert theta.tolist() == [0.0, 0.1]


def test_candidate_append_atoms_rejects_incomplete_pool_by_default() -> None:
    selected = (AnsatzTerm(label="seed_x", polynomial=_poly((("x", 1.0),))),)
    candidates = (AnsatzTerm(label="candidate_y", polynomial=_poly((("y", 1.0),))),)
    state = state_from_scaffold_runtime_input(
        _runtime_input(
            selected=selected,
            theta_runtime=np.array([0.0]),
            candidate_pool_terms=candidates,
            candidate_pool_complete=False,
        )
    )

    with pytest.raises(ValueError, match="incomplete candidate pool"):
        candidate_append_atoms(state)

    assert candidate_append_atoms(state, allow_incomplete_candidate_pool=True)


def test_append_controller_config_maps_to_legacy_support_patch_profile() -> None:
    config = AppendControllerConfig(
        max_append_candidates=5,
        max_prune_candidates=2,
        max_total_prunes=1,
        append_gain_threshold=0.25,
        allow_incomplete_candidate_pool=True,
    )

    support_config = config.to_support_patch_config()

    assert isinstance(support_config, SupportPatchControllerConfig)
    assert support_config.controller_profile == LEGACY_APPEND_CONTROLLER_PROFILE_V1
    assert support_config.exchange_enabled is False
    assert support_config.branch_scoring_enabled is False
    assert support_config.append_ladder_mode == "legacy_singleton"
    assert support_config.append_occurrence_policy == (
        APPEND_OCCURRENCE_POLICY_UNIQUE_SUPPORT
    )
    assert support_config.max_append_batch_size == 1
    assert support_config.cost_required_for_decisions is False
    assert support_config.allow_incomplete_candidate_pool is True


def test_support_patch_controller_defaults_are_exchange_family_combinatorial() -> None:
    config = SupportPatchControllerConfig()

    assert config.controller_profile == "support_patch_exchange_family_v1"
    assert config.parameterization_mode_default == AP_PARAMETERIZATION_PER_PAULI_TERM
    assert config.append_ladder_mode == "combinatorial"
    assert config.append_occurrence_policy == APPEND_OCCURRENCE_POLICY_LAYER_REUSE
    assert config.max_append_batch_size == 10
    # The Schur guard, rank fraction, and novelty ridge belonged to the
    # retired append route and were removed; the surviving cap is the
    # certification conditioning bound.
    assert config.append_schur_max_condition_number == 1.0e12
    # failed-append-reuse was removed 2026-08-15; the attribute must be gone.
    assert not hasattr(config, "failed_append_reuse_enabled")
    assert config.exchange_enabled is True
    assert config.branch_scoring_enabled is True
    assert config.support_patch_scoring_workers == 1
    assert config.max_exchange_append_branches == 3
    assert config.max_exchange_prune_branches == 3
    assert config.prune_enabled is False
    assert config.prune_commit_enabled is False
    assert config.max_prune_batch_size == 0


def test_support_patch_controller_rejects_retired_macro_scout_settings() -> None:
    # The macro-scout surface was seed-construction machinery the exchange
    # selector never read; its settings were removed, so passing one is now
    # an error rather than a silently inert configuration.
    for field in (
        "append_macro_scout_score_mode",
        "append_macro_scout_audit_parent_count",
        "append_macro_scout_audit_parent_fraction",
        "append_macro_scout_parent_cost_alpha",
    ):
        with pytest.raises(TypeError):
            SupportPatchControllerConfig(**{field: 1})


