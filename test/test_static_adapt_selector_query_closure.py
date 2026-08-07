from __future__ import annotations

from dataclasses import replace
import itertools

import numpy as np
import pytest

from pipelines.static_adapt.joint_linear_solve import JointLinearSolveConfig
from pipelines.static_adapt.selector_query_closure import (
    CAPABILITY_ACTIVE_CANDIDATE_GRAM,
    CAPABILITY_COMMON_TANGENT_CONTRACTION,
    EstimatorPrimitiveIdentity,
    FormalAdmissionCurvatureReceipt,
    GEOMETRY_ELEMENT_CROSS_STATE_TANGENT,
    GEOMETRY_ELEMENT_FULL_SYMMETRIC_GRAM,
    GrowthReceiptExpectation,
    OPTIMIZER_INVERSE_CURVATURE_PROVENANCE,
    OPTIMIZER_MIXED_BLOCK_STATUS,
    ORDINARY_HESSIAN_PROVENANCE,
    Phase2OrdinaryHessianBlocks,
    QueryPrimitiveLedger,
    QueryReceipt,
    SelectorGeometryAnchor,
    build_candidate_tangent_record,
    build_formal_admission_curvature_receipt,
    build_formal_growth_geometry_receipt,
    build_optimizer_inverse_curvature_prior,
    build_query_closed_population_workspace,
    evaluate_phase1_query_closed_score,
    normalize_serialized_matrix_payload,
    reconcile_primitive_id_sets,
    select_combinatorial_query_closed_batch,
    solve_phase2_query_closed_subset,
    validate_formal_growth_geometry_receipt,
)


def _primitive(
    kind: str,
    formula: str,
    *,
    state: str = "state-1",
    branch: str = "branch-1",
    candidate: str = "",
    insertion: int | None = None,
    tie: str = "tie-1",
) -> EstimatorPrimitiveIdentity:
    return EstimatorPrimitiveIdentity(
        primitive_kind=kind,
        physical_state_fingerprint=state,
        branch_id=branch,
        ordered_scaffold_fingerprint="scaffold-1",
        theta_fingerprint="theta-1",
        coordinate_registry_fingerprint="registry-1",
        candidate_generator_fingerprint=candidate,
        candidate_insertion_position=insertion,
        parameterization_tie_map_fingerprint=tie,
        hamiltonian_fingerprint="hamiltonian-1",
        provider_backend_id="exact-provider-v1",
        estimator_precision_contract="float64-exact",
        formula_primitive_identity=formula,
    )


def _receipt(
    *,
    requested=(),
    reused=(),
    fields=("value",),
    capabilities=(),
    shortcut=False,
) -> QueryReceipt:
    return QueryReceipt.from_primitives(
        requested=tuple(requested),
        reused=tuple(reused),
        returned_fields=tuple(fields),
        closure_capabilities=tuple(capabilities),
        provenance_by_field={field: "exact_provider" for field in fields},
        provider_kind="exact_state",
        statevector_shortcut_used=shortcut,
    )


def _anchor(*, empty: bool = False) -> tuple[SelectorGeometryAnchor, tuple]:
    gradient = _primitive("coordinate_gradient", "active-gradient")
    metric = _primitive("tangent_or_metric", "active-tangent-frame")
    receipt = _receipt(
        requested=(gradient, metric),
        fields=("b_A", "G_AA", "active_tangent_frame"),
        capabilities=(CAPABILITY_COMMON_TANGENT_CONTRACTION,),
        shortcut=True,
    )
    if empty:
        anchor = SelectorGeometryAnchor(
            state_fingerprint="state-1",
            branch_id="branch-1",
            manifold_id="manifold-1",
            ordered_scaffold_fingerprint="scaffold-1",
            theta_fingerprint="theta-1",
            coordinate_registry_fingerprint="registry-1",
            parameterization_mode="logical_shared",
            parameterization_tie_map_fingerprint="tie-1",
            hamiltonian_fingerprint="hamiltonian-1",
            active_coordinate_indices=(),
            active_tangent_handles=(),
            G_AA=np.zeros((0, 0)),
            b_A=np.zeros(0),
            gram_provenance="exact_fubini_study_metric",
            differential_provenance="exact_coordinate_differential",
            source_query_receipts=(receipt,),
        )
        return anchor, (gradient, metric)
    active_tangents = (
        np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
        np.array([0.0, 1.0, 0.0, 0.0], dtype=complex),
    )
    anchor = SelectorGeometryAnchor(
        state_fingerprint="state-1",
        branch_id="branch-1",
        manifold_id="manifold-1",
        ordered_scaffold_fingerprint="scaffold-1",
        theta_fingerprint="theta-1",
        coordinate_registry_fingerprint="registry-1",
        parameterization_mode="logical_shared",
        parameterization_tie_map_fingerprint="tie-1",
        hamiltonian_fingerprint="hamiltonian-1",
        active_coordinate_indices=(0, 1),
        active_tangent_handles=active_tangents,
        G_AA=np.eye(2),
        b_A=np.array([0.3, -0.1]),
        gram_provenance="exact_fubini_study_metric",
        differential_provenance="exact_coordinate_differential",
        source_query_receipts=(receipt,),
    )
    return anchor, (gradient, metric)


def _candidate(
    anchor: SelectorGeometryAnchor,
    index: int,
    tangent: np.ndarray,
    differential: float,
):
    gradient = _primitive(
        "coordinate_gradient",
        f"candidate-gradient-{index}",
        candidate=f"candidate-{index}",
        insertion=index,
    )
    metric = _primitive(
        "tangent_or_metric",
        f"candidate-tangent-{index}",
        candidate=f"candidate-{index}",
        insertion=index,
    )
    receipt = _receipt(
        requested=(gradient, metric),
        fields=("b_B", "tangent_handle"),
        capabilities=(CAPABILITY_COMMON_TANGENT_CONTRACTION,),
        shortcut=True,
    )
    record = build_candidate_tangent_record(
        anchor=anchor,
        candidate_fingerprint=f"candidate-{index}",
        candidate_registry_entry_fingerprint=f"registry-entry-{index}",
        insertion_position=index,
        tangent_handle=np.asarray(tangent, dtype=complex),
        differential=differential,
        query_receipts=(receipt,),
    )
    return record, (gradient, metric)


def _population():
    anchor, anchor_primitives = _anchor()
    candidates_and_primitives = (
        _candidate(anchor, 0, np.array([0.2, 0.1, 0.95, 0.0]), -0.5),
        _candidate(anchor, 1, np.array([-0.1, 0.3, 0.2, 0.9]), 0.35),
        _candidate(anchor, 2, np.array([0.0, -0.2, 0.4, 0.8]), -0.25),
    )
    records = tuple(row[0] for row in candidates_and_primitives)
    primitives = anchor_primitives + tuple(
        primitive
        for _, candidate_primitives in candidates_and_primitives
        for primitive in candidate_primitives
    )
    workspace = build_query_closed_population_workspace(
        anchor=anchor,
        candidate_records=records,
    )
    return anchor, records, workspace, primitives


def _ordinary_hessian(workspace, primitives):
    q_primitive = _primitive(
        "coordinate_second_derivative", "phase2-full-population-Q"
    )
    receipt = _receipt(
        requested=(q_primitive,),
        reused=tuple(primitives),
        fields=("Q_AA", "Q_AC", "Q_CC", "G_AA", "G_AC", "G_CC"),
    )
    active = workspace.anchor.active_dimension
    candidate_count = len(workspace.candidate_records)
    Q_AA = 1.8 * np.eye(active)
    Q_AC = np.array(
        [[0.08, -0.03, 0.02], [0.01, 0.06, -0.04]], dtype=float
    )[:, :candidate_count]
    Q_CC = 1.4 * np.eye(candidate_count) + 0.03 * (
        np.ones((candidate_count, candidate_count)) - np.eye(candidate_count)
    )
    blocks = Phase2OrdinaryHessianBlocks(
        workspace_fingerprint=workspace.workspace_fingerprint,
        candidate_keys=workspace.candidate_keys,
        Q_AA=Q_AA,
        Q_AC=Q_AC,
        Q_CC=Q_CC,
        source_query_receipts=(receipt,),
        provenance_by_block={
            "Q_AA": ORDINARY_HESSIAN_PROVENANCE,
            "Q_AC": ORDINARY_HESSIAN_PROVENANCE,
            "Q_CC": ORDINARY_HESSIAN_PROVENANCE,
        },
    )
    return blocks, q_primitive, receipt


def test_primitive_identity_is_deterministic_and_fingerprint_complete():
    primitive = _primitive(
        "tangent_or_metric",
        "candidate-tangent",
        candidate="candidate-a",
        insertion=2,
    )
    assert primitive.primitive_id == replace(primitive).primitive_id
    for mutation in (
        {"physical_state_fingerprint": "state-2"},
        {"branch_id": "branch-2"},
        {"candidate_insertion_position": 3},
        {"parameterization_tie_map_fingerprint": "tie-2"},
        {"hamiltonian_fingerprint": "hamiltonian-2"},
        {"estimator_precision_contract": "shots-4096"},
    ):
        assert replace(primitive, **mutation).primitive_id != primitive.primitive_id


def test_query_receipt_is_portable_and_rejects_ambiguous_charge():
    primitive = _primitive("energy", "energy")
    receipt = _receipt(
        requested=(primitive,), fields=("energy",), shortcut=True
    )
    payload = receipt.portable_payload()
    assert payload["primitive_ids_requested"] == [primitive.primitive_id]
    assert payload["statevector_shortcut_used"] is True
    with pytest.raises(ValueError, match="requested and reused"):
        _receipt(requested=(primitive,), reused=(primitive,))


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (4, 0)])
def test_serialized_zero_extent_matrix_restores_typed_shape(shape):
    restored = normalize_serialized_matrix_payload(
        [], expected_shape=shape, field_name="G_AA_raw"
    )
    assert restored.shape == shape
    assert restored.size == 0
    assert not restored.flags.writeable


@pytest.mark.parametrize(
    "payload,shape,error",
    [
        ([], (1, 1), "shape"),
        ([[1.0]], (0, 0), "shape"),
        ([[float("nan")]], (1, 1), "nonfinite"),
    ],
)
def test_serialized_matrix_normalization_fails_closed(
    payload, shape, error
):
    with pytest.raises(ValueError, match=error):
        normalize_serialized_matrix_payload(
            payload, expected_shape=shape, field_name="G_AA_raw"
        )


def test_phase1_augmented_gram_score_is_query_closed_and_reconciled():
    anchor, anchor_primitives = _anchor()
    candidate, candidate_primitives = _candidate(
        anchor, 0, np.array([0.2, 0.1, 0.95, 0.0]), -0.5
    )
    baseline_ids = {
        primitive.primitive_id
        for primitive in (*anchor_primitives, *candidate_primitives)
    }
    score = evaluate_phase1_query_closed_score(
        anchor=anchor,
        candidate=candidate,
        trust_radius=0.25,
        resource_burden=0.2,
        metric_regularization=1e-9,
        baseline_primitive_ids=baseline_ids,
    )
    expected_s = candidate.G_BB - candidate.G_AB @ candidate.G_AB
    expected_b = candidate.b_B - candidate.G_AB @ anchor.b_A
    expected_response = expected_b**2 / (expected_s + 1e-9)
    assert score.feasible
    assert score.schur_metric == pytest.approx(expected_s)
    assert score.residual_differential == pytest.approx(expected_b)
    assert score.response == pytest.approx(expected_response)
    assert score.trust_gain == pytest.approx(0.25 * np.sqrt(expected_response))
    assert score.score == pytest.approx(score.trust_gain / 1.2)
    assert score.primitive_set_reconciled is True
    assert score.incremental_query_charge == 0
    assert "schur_residual_metric" in score.query_free_derived_fields
    portable = candidate.portable_payload()
    assert "tangent_handle" not in portable
    assert "tangent_handle_payload" not in str(portable).lower()
    assert "statevector_payload" not in str(portable).lower()


@pytest.mark.parametrize(
    "field,value",
    [
        ("state_fingerprint", "state-2"),
        ("branch_id", "branch-2"),
        ("ordered_scaffold_fingerprint", "scaffold-2"),
        ("theta_fingerprint", "theta-2"),
        ("coordinate_registry_fingerprint", "registry-2"),
        ("parameterization_tie_map_fingerprint", "tie-2"),
        ("hamiltonian_fingerprint", "hamiltonian-2"),
    ],
)
def test_phase1_fails_closed_on_scope_mismatch(field, value):
    anchor, _ = _anchor()
    candidate, _ = _candidate(
        anchor, 0, np.array([0.2, 0.1, 0.95, 0.0]), -0.5
    )
    result = evaluate_phase1_query_closed_score(
        anchor=anchor,
        candidate=replace(candidate, **{field: value}),
        trust_radius=0.25,
    )
    assert not result.feasible
    assert field in result.reason


def test_phase1_fails_closed_without_tangent_or_explicit_cross_gram():
    anchor, _ = _anchor()
    candidate, _ = _candidate(anchor, 0, np.array([0.2, 0.1, 0.95, 0.0]), -0.5)
    scalar_only = replace(
        candidate,
        tangent_handle=None,
        closure_capabilities=(),
    )
    result = evaluate_phase1_query_closed_score(
        anchor=anchor, candidate=scalar_only, trust_radius=0.25
    )
    assert not result.feasible
    assert result.reason == "missing_tangent_handle_or_active_candidate_gram"


def test_phase1_rank_gates_redundant_and_near_threshold_directions():
    anchor, _ = _anchor()
    redundant, _ = _candidate(anchor, 0, np.array([1.0, 0.0, 0.0, 0.0]), -0.5)
    result = evaluate_phase1_query_closed_score(
        anchor=anchor, candidate=redundant, trust_radius=0.25
    )
    assert not result.feasible
    assert result.rank_gain == 0
    near = replace(
        redundant,
        G_BB=1.0 + 1e-12,
        tangent_handle=None,
        closure_capabilities=(CAPABILITY_ACTIVE_CANDIDATE_GRAM,),
    )
    near_result = evaluate_phase1_query_closed_score(
        anchor=anchor,
        candidate=near,
        trust_radius=0.25,
        rank_relative_tolerance=1e-6,
    )
    assert not near_result.feasible
    assert near_result.schur_metric == pytest.approx(1e-12, abs=1e-14)


def test_phase1_empty_active_context_is_supported():
    anchor, _ = _anchor(empty=True)
    candidate, _ = _candidate(anchor, 0, np.array([0.0, 1.0]), -0.4)
    result = evaluate_phase1_query_closed_score(
        anchor=anchor, candidate=candidate, trust_radius=0.3
    )
    assert result.feasible
    assert result.schur_metric == pytest.approx(1.0)
    assert result.residual_differential == pytest.approx(-0.4)


def test_population_closes_all_candidate_pair_grams_without_queries():
    _, records, workspace, _ = _population()
    assert workspace.complete_pair_gram
    assert workspace.missing_primitive_requests == ()
    direct = np.array(
        [
            [
                np.real(np.vdot(left.tangent_handle, right.tangent_handle))
                for right in records
            ]
            for left in records
        ]
    )
    np.testing.assert_allclose(workspace.G_CC, direct)
    assert workspace.derived_feature_cache[
        "candidate_pair_classical_contraction_count"
    ] == 3
    assert "G_CC_live_tangent_pair_contractions" in workspace.query_free_derived_fields


def test_scalar_provider_declares_and_charges_only_missing_pair_primitives():
    anchor, records, _, _ = _population()
    scalar_records = tuple(
        replace(
            record,
            tangent_handle=None,
            closure_capabilities=(CAPABILITY_ACTIVE_CANDIDATE_GRAM,),
        )
        for record in records
    )

    def missing(left, right):
        return _primitive(
            "tangent_or_metric",
            f"pair:{left.candidate_key}:{right.candidate_key}",
            candidate=f"{left.candidate_fingerprint}+{right.candidate_fingerprint}",
            insertion=min(left.insertion_position, right.insertion_position),
        )

    incomplete = build_query_closed_population_workspace(
        anchor=anchor,
        candidate_records=scalar_records,
        missing_pair_primitive_factory=missing,
    )
    assert not incomplete.complete_pair_gram
    assert len(incomplete.missing_primitive_requests) == 3
    pair_values = {}
    pair_receipts = {}
    for left, right in itertools.combinations(scalar_records, 2):
        pair_key = tuple(sorted((left.candidate_key, right.candidate_key)))
        primitive = missing(left, right)
        pair_values[pair_key] = np.real(
            np.vdot(records[left.insertion_position].tangent_handle, records[right.insertion_position].tangent_handle)
        )
        pair_receipts[pair_key] = _receipt(
            requested=(primitive,), fields=("G_candidate_pair",)
        )
    complete = build_query_closed_population_workspace(
        anchor=anchor,
        candidate_records=scalar_records,
        provided_pair_gram=pair_values,
        provided_pair_receipts=pair_receipts,
    )
    assert complete.complete_pair_gram
    assert complete.derived_feature_cache["candidate_pair_measured_entry_count"] == 3
    assert len(complete.source_primitive_ids - incomplete.source_primitive_ids) == 3


def test_phase2_ordinary_hessian_trust_solve_and_schur_parity():
    _, _, workspace, primitives = _population()
    blocks, q_primitive, _ = _ordinary_hessian(workspace, primitives)
    config = JointLinearSolveConfig(
        rank_relative_tolerance=1e-10,
        metric_regularization=0.0,
        energy_regularization=1e-12,
        max_fubini_study_step=0.5,
    )
    result = solve_phase2_query_closed_subset(
        workspace=workspace,
        ordinary_hessian=blocks,
        candidate_indices=(0, 2),
        resource_burden=0.4,
        solve_config=config,
    )
    assert result.feasible
    assert result.ordinary_hessian_provenance == ORDINARY_HESSIAN_PROVENANCE
    assert result.optimizer_curvature_used is False
    assert result.predicted_reduction > 0.0
    assert result.score == pytest.approx(result.predicted_reduction / 1.4)
    assert result.direct_schur_step_difference < 1e-9
    assert result.supported_active_rank == workspace.anchor.active_dimension
    assert result.supported_candidate_rank == 2
    assert result.active_subspace_embedding_residual < 1e-10
    assert result.structured_whitening_identity_residual < 1e-10
    assert q_primitive.primitive_id in result.source_primitive_ids
    assert len(result.source_primitive_ids) == len(set(result.source_primitive_ids))


def test_phase2_schur_partition_uses_old_supported_rank_not_raw_coordinate_count():
    gradient = _primitive("coordinate_gradient", "rank-deficient-active-gradient")
    metric = _primitive("tangent_or_metric", "rank-deficient-active-metric")
    anchor_receipt = _receipt(
        requested=(gradient, metric),
        fields=("b_A", "G_AA", "active_tangent_frame"),
        capabilities=(CAPABILITY_COMMON_TANGENT_CONTRACTION,),
        shortcut=True,
    )
    repeated = np.array([1.0, 0.0, 0.0], dtype=complex)
    anchor = SelectorGeometryAnchor(
        state_fingerprint="state-1",
        branch_id="branch-1",
        manifold_id="manifold-1",
        ordered_scaffold_fingerprint="scaffold-1",
        theta_fingerprint="theta-1",
        coordinate_registry_fingerprint="registry-1",
        parameterization_mode="logical_shared",
        parameterization_tie_map_fingerprint="tie-1",
        hamiltonian_fingerprint="hamiltonian-1",
        active_coordinate_indices=(0, 1),
        active_tangent_handles=(repeated, repeated.copy()),
        G_AA=np.ones((2, 2)),
        b_A=np.array([0.2, 0.2]),
        gram_provenance="exact_fubini_study_metric",
        differential_provenance="exact_coordinate_differential",
        source_query_receipts=(anchor_receipt,),
    )
    candidate, candidate_primitives = _candidate(
        anchor,
        0,
        np.array([0.0, 1.0, 0.0]),
        -0.4,
    )
    workspace = build_query_closed_population_workspace(
        anchor=anchor, candidate_records=(candidate,)
    )
    blocks, _, _ = _ordinary_hessian(
        workspace, (gradient, metric, *candidate_primitives)
    )
    result = solve_phase2_query_closed_subset(
        workspace=workspace,
        ordinary_hessian=blocks,
        candidate_indices=(0,),
        solve_config=JointLinearSolveConfig(
            rank_relative_tolerance=1e-10,
            metric_regularization=1e-9,
            energy_regularization=1e-12,
            max_fubini_study_step=0.4,
        ),
    )
    assert result.feasible
    assert result.supported_active_rank == 1
    assert result.supported_candidate_rank == 1
    assert result.active_subspace_embedding_residual < 1e-10
    assert result.structured_whitening_identity_residual < 1e-10
    assert result.direct_schur_step_difference < 1e-9


def test_phase2_schur_parity_does_not_reapply_energy_floor_as_pinv_rcond():
    _, _, workspace, primitives = _population()
    blocks, _, _ = _ordinary_hessian(workspace, primitives)
    conditioned = replace(
        blocks,
        Q_AA=np.diag([-0.1, 2.0]),
        Q_AC=np.zeros_like(blocks.Q_AC),
        Q_CC=np.eye(blocks.Q_CC.shape[0]),
    )
    result = solve_phase2_query_closed_subset(
        workspace=workspace,
        ordinary_hessian=conditioned,
        candidate_indices=(0,),
        solve_config=JointLinearSolveConfig(
            rank_relative_tolerance=1e-10,
            metric_regularization=0.0,
            energy_regularization=0.2,
            max_fubini_study_step=10.0,
        ),
    )
    assert result.feasible
    assert result.direct_schur_step_difference < 1e-9
    assert result.shared_direct_step_difference < 1e-9
    assert result.direct_kkt_residual < 1e-9
    assert result.schur_kkt_residual < 1e-9


def test_phase2_schur_parity_uses_scale_invariant_backward_error():
    gradient = _primitive("coordinate_gradient", "scaled-active-gradient")
    metric = _primitive("tangent_or_metric", "scaled-active-metric")
    anchor_receipt = _receipt(
        requested=(gradient, metric),
        fields=("b_A", "G_AA", "active_tangent_frame"),
        capabilities=(CAPABILITY_COMMON_TANGENT_CONTRACTION,),
        shortcut=True,
    )
    rhs = np.array([-0.8877660096825283, 4.597522373516844])
    anchor = SelectorGeometryAnchor(
        state_fingerprint="state-1",
        branch_id="branch-1",
        manifold_id="manifold-1",
        ordered_scaffold_fingerprint="scaffold-1",
        theta_fingerprint="theta-1",
        coordinate_registry_fingerprint="registry-1",
        parameterization_mode="logical_shared",
        parameterization_tie_map_fingerprint="tie-1",
        hamiltonian_fingerprint="hamiltonian-1",
        active_coordinate_indices=(0,),
        active_tangent_handles=(np.array([1.0, 0.0], dtype=complex),),
        G_AA=np.eye(1),
        b_A=np.array([-rhs[0]]),
        gram_provenance="exact_fubini_study_metric",
        differential_provenance="exact_coordinate_differential",
        source_query_receipts=(anchor_receipt,),
    )
    candidate, candidate_primitives = _candidate(
        anchor,
        0,
        np.array([0.0, 1.0]),
        -rhs[1],
    )
    workspace = build_query_closed_population_workspace(
        anchor=anchor, candidate_records=(candidate,)
    )
    matrix = np.array(
        [
            [176100269.00731426, 442894035.94602495],
            [442894035.946025, 1113883250.7414658],
        ]
    )
    q_primitive = _primitive(
        "coordinate_second_derivative", "scaled-phase2-Q"
    )
    q_receipt = _receipt(
        requested=(q_primitive,),
        reused=(gradient, metric, *candidate_primitives),
        fields=("Q_AA", "Q_AC", "Q_CC", "G_AA", "G_AC", "G_CC"),
    )
    blocks = Phase2OrdinaryHessianBlocks(
        workspace_fingerprint=workspace.workspace_fingerprint,
        candidate_keys=workspace.candidate_keys,
        Q_AA=matrix[:1, :1],
        Q_AC=matrix[:1, 1:],
        Q_CC=matrix[1:, 1:],
        source_query_receipts=(q_receipt,),
        provenance_by_block={
            "Q_AA": ORDINARY_HESSIAN_PROVENANCE,
            "Q_AC": ORDINARY_HESSIAN_PROVENANCE,
            "Q_CC": ORDINARY_HESSIAN_PROVENANCE,
        },
    )
    result = solve_phase2_query_closed_subset(
        workspace=workspace,
        ordinary_hessian=blocks,
        candidate_indices=(0,),
        solve_config=JointLinearSolveConfig(
            rank_relative_tolerance=1e-12,
            metric_regularization=0.0,
            energy_regularization=0.0,
            max_fubini_study_step=10.0,
        ),
    )
    assert result.feasible
    assert result.schur_kkt_backward_error < 1e-12
    assert result.direct_kkt_backward_error < 1e-12
    assert result.schur_kkt_residual > 1e-10


def test_phase2_fails_closed_for_workspace_or_pair_geometry_mismatch():
    anchor, records, workspace, primitives = _population()
    blocks, _, _ = _ordinary_hessian(workspace, primitives)
    mismatch = replace(blocks, workspace_fingerprint="different-workspace")
    result = solve_phase2_query_closed_subset(
        workspace=workspace,
        ordinary_hessian=mismatch,
        candidate_indices=(0,),
    )
    assert not result.feasible
    assert result.reason == "ordinary_hessian_workspace_fingerprint_mismatch"

    scalar_records = tuple(
        replace(
            record,
            tangent_handle=None,
            closure_capabilities=(CAPABILITY_ACTIVE_CANDIDATE_GRAM,),
        )
        for record in records
    )
    incomplete = build_query_closed_population_workspace(
        anchor=anchor, candidate_records=scalar_records
    )
    incomplete_blocks, _, _ = _ordinary_hessian(incomplete, primitives)
    singleton = solve_phase2_query_closed_subset(
        workspace=incomplete,
        ordinary_hessian=incomplete_blocks,
        candidate_indices=(0,),
    )
    batch = solve_phase2_query_closed_subset(
        workspace=incomplete,
        ordinary_hessian=incomplete_blocks,
        candidate_indices=(0, 1),
    )
    assert singleton.feasible
    assert not batch.feasible
    assert batch.reason == "missing_candidate_pair_gram_primitive"


def test_ordinary_hessian_cannot_be_relabelled_optimizer_curvature():
    _, _, workspace, primitives = _population()
    blocks, _, _ = _ordinary_hessian(workspace, primitives)
    with pytest.raises(ValueError, match="provenance"):
        replace(blocks, hessian_provenance=OPTIMIZER_INVERSE_CURVATURE_PROVENANCE)
    with pytest.raises(ValueError, match="Q_AC provenance"):
        replace(
            blocks,
            provenance_by_block={
                "Q_AA": ORDINARY_HESSIAN_PROVENANCE,
                "Q_AC": OPTIMIZER_INVERSE_CURVATURE_PROVENANCE,
                "Q_CC": ORDINARY_HESSIAN_PROVENANCE,
            },
        )


def test_optimizer_growth_prior_is_mandatory_distinct_and_reset_safe():
    active = np.array([[2.0, 0.2], [0.2, 1.5]])
    prior = build_optimizer_inverse_curvature_prior(
        active_inverse_curvature=active,
        active_rank=2,
        candidate_rank=2,
        candidate_scale=0.7,
    )
    np.testing.assert_allclose(prior.B_plus[:2, :2], active)
    np.testing.assert_allclose(prior.B_plus[:2, 2:], 0.0)
    np.testing.assert_allclose(prior.B_plus[2:, 2:], 0.7 * np.eye(2))
    assert prior.provenance == OPTIMIZER_INVERSE_CURVATURE_PROVENANCE
    assert prior.mixed_block_status == OPTIMIZER_MIXED_BLOCK_STATUS
    reset = build_optimizer_inverse_curvature_prior(
        active_inverse_curvature=None,
        active_rank=2,
        candidate_rank=1,
        candidate_scale=0.7,
        reset_active_scale=1.2,
    )
    np.testing.assert_allclose(reset.B_plus[:2, :2], 1.2 * np.eye(2))
    assert reset.active_source == "regularized_isotropic_reset_prior"
    rank_zero = build_optimizer_inverse_curvature_prior(
        active_inverse_curvature=None,
        active_rank=0,
        candidate_rank=1,
        candidate_scale=0.5,
    )
    np.testing.assert_allclose(rank_zero.B_plus, [[0.5]])


def test_combinatorial_argmax_searches_every_subset_and_reuses_cache():
    _, _, workspace, primitives = _population()
    blocks, _, _ = _ordinary_hessian(workspace, primitives)
    config = JointLinearSolveConfig(
        rank_relative_tolerance=1e-10,
        metric_regularization=0.0,
        energy_regularization=1e-12,
        max_fubini_study_step=0.5,
    )
    selection = select_combinatorial_query_closed_batch(
        workspace=workspace,
        ordinary_hessian=blocks,
        max_batch_size=2,
        candidate_resource_burdens={key: 0.1 for key in workspace.candidate_keys},
        solve_config=config,
    )
    assert selection.feasible
    assert selection.subsets_searched == 6
    assert selection.feasible_subset_count == 6
    assert selection.selected is not None
    assert selection.selected.score == max(
        result.score for result in selection.ordered_results if result.feasible
    )
    cache_size = len(workspace.subset_solve_cache)
    repeated = select_combinatorial_query_closed_batch(
        workspace=workspace,
        ordinary_hessian=blocks,
        max_batch_size=2,
        candidate_resource_burdens={key: 0.1 for key in workspace.candidate_keys},
        solve_config=config,
    )
    assert repeated.selected.candidate_keys == selection.selected.candidate_keys
    assert len(workspace.subset_solve_cache) == cache_size


def test_permuted_population_keeps_physical_subset_score_and_determinism():
    anchor, records, workspace, primitives = _population()
    blocks, _, _ = _ordinary_hessian(workspace, primitives)
    original = solve_phase2_query_closed_subset(
        workspace=workspace,
        ordinary_hessian=blocks,
        candidate_indices=(0, 2),
        solve_config=JointLinearSolveConfig(metric_regularization=0.0),
    )
    permutation = (2, 1, 0)
    permuted_records = tuple(records[index] for index in permutation)
    permuted_workspace = build_query_closed_population_workspace(
        anchor=anchor, candidate_records=permuted_records
    )
    inverse = {old: new for new, old in enumerate(permutation)}
    permuted_blocks = Phase2OrdinaryHessianBlocks(
        workspace_fingerprint=permuted_workspace.workspace_fingerprint,
        candidate_keys=permuted_workspace.candidate_keys,
        Q_AA=blocks.Q_AA,
        Q_AC=blocks.Q_AC[:, permutation],
        Q_CC=blocks.Q_CC[np.ix_(permutation, permutation)],
        source_query_receipts=blocks.source_query_receipts,
        provenance_by_block=dict(blocks.provenance_by_block),
    )
    permuted = solve_phase2_query_closed_subset(
        workspace=permuted_workspace,
        ordinary_hessian=permuted_blocks,
        candidate_indices=(inverse[0], inverse[2]),
        solve_config=JointLinearSolveConfig(metric_regularization=0.0),
    )
    assert original.feasible and permuted.feasible
    assert permuted.predicted_reduction == pytest.approx(original.predicted_reduction)


def _growth_fixture():
    _, _, workspace, primitives = _population()
    blocks, _, _ = _ordinary_hessian(workspace, primitives)
    selected = solve_phase2_query_closed_subset(
        workspace=workspace,
        ordinary_hessian=blocks,
        candidate_indices=(0, 1),
        solve_config=JointLinearSolveConfig(metric_regularization=0.0),
    )
    assert selected.feasible
    receipt = build_formal_growth_geometry_receipt(
        workspace=workspace,
        selected=selected,
        old_to_new_registry_mapping=(0, 1),
        new_coordinate_registry_fingerprint="registry-plus",
        rank_relative_tolerance=1e-6,
        metric_regularization=1e-9,
        zero_new_coordinates=True,
        old_gate_subsequence_unchanged=True,
    )
    return receipt


def _curvature_fixture(*, feasible: bool = True, reason: str | None = None):
    _, _, workspace, primitives = _population()
    blocks, q_primitive, _ = _ordinary_hessian(workspace, primitives)
    selected = solve_phase2_query_closed_subset(
        workspace=workspace,
        ordinary_hessian=blocks,
        candidate_indices=(0,),
        solve_config=JointLinearSolveConfig(metric_regularization=0.0),
    )
    assert selected.feasible
    growth = build_formal_growth_geometry_receipt(
        workspace=workspace,
        selected=selected,
        old_to_new_registry_mapping=(1, 2),
        new_coordinate_registry_fingerprint="registry-plus-curvature",
        rank_relative_tolerance=1e-6,
        metric_regularization=1e-9,
        zero_new_coordinates=True,
        old_gate_subsequence_unchanged=True,
    )
    candidate_index = selected.candidate_indices[0]
    summary = {
        "schema": "historical_singleton_coordinate_model_v1",
        "scope": "historical_phase3_whitening",
        "authority": "historical_phase3_benefit_overlay_only",
        "feasible": bool(feasible),
        "reason": (
            str(reason)
            if reason is not None
            else (
                "supported_metric_whitened_eigh_solve"
                if feasible
                else "rank_gate"
            )
        ),
        "geometry_mode": "full_residual_gram_hessian_v1",
        "joint_batch_context_mode": "full_ansatz_v1",
        "joint_linear_solve_policy_effective": (
            "supported_metric_whitened_eigh_v1"
        ),
        "supported_metric_whitening_policy": (
            "supported_metric_whitened_eigh_v1"
        ),
        "supported_metric_whitening_provenance_id": "whitening-source-1",
        "active_coordinate_count": workspace.anchor.active_dimension,
        "batch_coordinate_count": 1,
        "active_coordinate_identities": ["active-0", "active-1"],
        "batch_coordinate_identities": [
            {
                "candidate_label": "candidate-label-0",
                "candidate_pool_index": 17,
                "position_id": int(growth.insertion_positions[0]),
                "global_child_identity": "child-0",
            }
        ],
        "candidate_label": "candidate-label-0",
        "candidate_pool_index": 17,
        "position_id": int(growth.insertion_positions[0]),
        "G_AA_raw": growth.G_AA.tolist(),
        "G_AB_raw": growth.G_AB.tolist(),
        "G_BB_raw": growth.G_BB.tolist(),
        "H_AA_raw": blocks.Q_AA.tolist(),
        "H_AB_raw": blocks.Q_AC[:, [candidate_index]].tolist(),
        "H_BB_raw": blocks.Q_CC[
            np.ix_((candidate_index,), (candidate_index,))
        ].tolist(),
        # Historical SR summaries store descent gradients, whereas the formal
        # growth receipt stores the positive coordinate energy gradient.
        "g_A": (-workspace.anchor.b_A).tolist(),
        "g_B": (-growth.candidate_gradients).tolist(),
    }
    return growth, summary, q_primitive


def test_growth_receipt_reuses_selected_blocks_and_is_portable():
    receipt = _growth_fixture()
    expectation = GrowthReceiptExpectation.from_receipt(receipt)
    validation = validate_formal_growth_geometry_receipt(receipt, expectation)
    assert validation.valid
    assert validation.query_reuse_allowed
    assert validation.incremental_query_charge == 0
    payload = receipt.portable_payload()
    assert payload["G_AB"] == receipt.G_AB.tolist()
    assert payload["receipt_fingerprint"] == receipt.receipt_fingerprint
    assert "tangent_handle" not in str(payload).lower()
    assert "statevector_payload" not in str(payload).lower()


@pytest.mark.parametrize(
    "field,value",
    [
        ("state_fingerprint", "state-2"),
        ("branch_id", "branch-2"),
        ("manifold_id", "manifold-2"),
        ("ordered_scaffold_fingerprint", "scaffold-2"),
        ("theta_fingerprint", "theta-2"),
        ("old_coordinate_registry_fingerprint", "registry-2"),
        ("new_coordinate_registry_fingerprint", "registry-plus-2"),
        ("parameterization_tie_map_fingerprint", "tie-2"),
        ("hamiltonian_fingerprint", "hamiltonian-2"),
        ("candidate_keys", ("candidate-key-2", "candidate-key-3")),
        (
            "candidate_generator_fingerprints",
            ("candidate-generator-2", "candidate-generator-3"),
        ),
        ("insertion_positions", (7, 8)),
        ("old_to_new_registry_mapping", (1, 0)),
        ("rank_rule_fingerprint", "rank-rule-2"),
        ("metric_convention", "metric-convention-2"),
        ("zero_new_coordinates", False),
        ("old_gate_subsequence_unchanged", False),
    ],
)
def test_growth_receipt_fails_closed_on_every_identity_mismatch(field, value):
    receipt = _growth_fixture()
    expectation = replace(
        GrowthReceiptExpectation.from_receipt(receipt), **{field: value}
    )
    validation = validate_formal_growth_geometry_receipt(receipt, expectation)
    assert not validation.valid
    assert not validation.query_reuse_allowed
    assert validation.incremental_query_charge is None
    assert field in validation.mismatched_fields


def test_formal_admission_curvature_receipt_preserves_full_raw_model_and_sign():
    growth, summary, q_primitive = _curvature_fixture()
    receipt = build_formal_admission_curvature_receipt(
        growth_receipt=growth,
        phase3_summary=summary,
        ordinary_hessian_primitive_ids=(
            q_primitive.primitive_id,
            q_primitive.primitive_id,
        ),
    )
    assert isinstance(receipt, FormalAdmissionCurvatureReceipt)
    assert receipt.growth_receipt_fingerprint == growth.receipt_fingerprint
    assert receipt.active_coordinate_identities == ("active-0", "active-1")
    assert receipt.candidate_coordinate_identities == (
        {
            "candidate_label": "candidate-label-0",
            "candidate_pool_index": 17,
            "position_id": 0,
            "global_child_identity": "child-0",
        },
    )
    np.testing.assert_allclose(receipt.G_AA, growth.G_AA)
    np.testing.assert_allclose(receipt.G_AB, growth.G_AB)
    np.testing.assert_allclose(receipt.G_BB, growth.G_BB)
    np.testing.assert_allclose(
        -receipt.descent_gradient_B, growth.candidate_gradients
    )
    assert receipt.ordinary_hessian_primitive_ids == (
        q_primitive.primitive_id,
    )
    assert receipt.ordinary_hessian_provenance == ORDINARY_HESSIAN_PROVENANCE
    assert receipt.selector_feasible is True
    assert receipt.receipt_fingerprint == receipt.receipt_fingerprint

    payload = receipt.portable_payload()
    restored = FormalAdmissionCurvatureReceipt.from_portable_payload(payload)
    assert restored.receipt_fingerprint == receipt.receipt_fingerprint
    assert restored.portable_payload() == payload
    tampered = dict(payload)
    tampered["H_BB"] = [[float(receipt.H_BB[0, 0]) + 0.5]]
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        FormalAdmissionCurvatureReceipt.from_portable_payload(tampered)


def test_formal_admission_curvature_receipt_accepts_array_complete_rank_gate():
    growth, summary, q_primitive = _curvature_fixture(feasible=False)
    # The historical producer omitted these invariant labels only on rank-gate
    # exits while retaining the full raw arrays and whitening telemetry.
    summary.pop("geometry_mode")
    summary.pop("joint_batch_context_mode")
    summary["batch_coordinate_identities"][0]["candidate_label"] = ""
    receipt = build_formal_admission_curvature_receipt(
        growth_receipt=growth,
        phase3_summary=summary,
        ordinary_hessian_primitive_ids=(q_primitive.primitive_id,),
    )
    assert receipt.selector_feasible is False
    assert receipt.selector_reason == "rank_gate"
    assert receipt.geometry_mode == "full_residual_gram_hessian_v1"
    assert receipt.joint_batch_context_mode == "full_ansatz_v1"
    assert (
        receipt.candidate_coordinate_identities[0]["candidate_label"]
        == summary["candidate_label"]
    )


@pytest.mark.parametrize(
    "mutation,error",
    [
        ({"schema": "wrong"}, "schema"),
        ({"scope": "historical_phase2"}, "scope"),
        ({"authority": "diagnostic"}, "authority"),
        ({"joint_batch_context_mode": "windowed_v1"}, "full_ansatz"),
        ({"geometry_mode": "diagonal"}, "full_residual"),
        ({"joint_linear_solve_policy_effective": "legacy_block_pinv_v1"}, "supported-metric"),
    ],
)
def test_formal_admission_curvature_receipt_rejects_wrong_phase3_contract(
    mutation, error
):
    growth, summary, _ = _curvature_fixture()
    summary.update(mutation)
    if "joint_linear_solve_policy_effective" in mutation:
        summary.pop("supported_metric_whitening_policy")
    with pytest.raises(ValueError, match=error):
        build_formal_admission_curvature_receipt(
            growth_receipt=growth,
            phase3_summary=summary,
        )


def test_formal_admission_curvature_receipt_rejects_geometry_or_sign_drift():
    growth, summary, _ = _curvature_fixture()
    wrong_metric = dict(summary)
    wrong_metric["G_BB_raw"] = (
        np.asarray(summary["G_BB_raw"], dtype=float) + 0.1
    ).tolist()
    with pytest.raises(ValueError, match="G_BB disagrees"):
        build_formal_admission_curvature_receipt(
            growth_receipt=growth,
            phase3_summary=wrong_metric,
        )

    wrong_sign = dict(summary)
    wrong_sign["g_B"] = growth.candidate_gradients.tolist()
    with pytest.raises(ValueError, match="wrong sign"):
        build_formal_admission_curvature_receipt(
            growth_receipt=growth,
            phase3_summary=wrong_sign,
        )

    wrong_position = dict(summary)
    wrong_position["batch_coordinate_identities"] = [
        {
            **summary["batch_coordinate_identities"][0],
            "position_id": 7,
        }
    ]
    with pytest.raises(ValueError, match="position"):
        build_formal_admission_curvature_receipt(
            growth_receipt=growth,
            phase3_summary=wrong_position,
        )

    nonsymmetric = dict(summary)
    nonsymmetric["H_AA_raw"] = [[1.0, 0.5], [0.0, 1.0]]
    with pytest.raises(ValueError, match="H_AA_raw must be symmetric"):
        build_formal_admission_curvature_receipt(
            growth_receipt=growth,
            phase3_summary=nonsymmetric,
        )


def test_formal_admission_curvature_receipt_rejects_non_rank_infeasibility():
    growth, summary, _ = _curvature_fixture(
        feasible=False, reason="conditioning_gate"
    )
    with pytest.raises(ValueError, match="feasible or an array-complete rank_gate"):
        build_formal_admission_curvature_receipt(
            growth_receipt=growth,
            phase3_summary=summary,
        )


def test_query_ledger_reconciles_categories_reuse_and_classical_free_work():
    energy = _primitive("energy", "energy")
    gradient = _primitive("coordinate_gradient", "gradient")
    metric = _primitive("tangent_or_metric", "metric")
    q = _primitive("coordinate_second_derivative", "Q")
    hv = _primitive("hessian_vector", "Hv")
    cross = _primitive("cross_state_tangent", "cross")
    phase1 = _receipt(
        requested=(energy, gradient, metric),
        fields=("energy", "gradient", "metric", "tangent_handle"),
        shortcut=True,
    )
    phase2 = _receipt(
        requested=(q, hv),
        reused=(gradient, metric),
        fields=("Q", "Hv", "metric"),
    )
    batch = _receipt(
        requested=(cross,),
        reused=(q, metric),
        fields=("cross", "Q", "G_CC"),
    )
    growth = _receipt(
        reused=(metric, q), fields=("G_AB", "G_BB", "Q")
    )
    ledger = QueryPrimitiveLedger()
    ledger.consume_receipt(phase1, consumer_phase="phase1")
    ledger.consume_receipt(phase2, consumer_phase="phase2")
    ledger.consume_receipt(batch, consumer_phase="batch")
    ledger.consume_receipt(growth, consumer_phase="growth")
    ledger.consume_receipt(growth, consumer_phase="growth")
    ledger.record_query_free_derived_fields(
        ("pseudoinverse", "eigendecomposition", "subset_search")
    )
    ledger.record_matrix_element_diagnostic("G_CC_elements", 9)
    telemetry = ledger.telemetry(expected_actual_operator_probe_count=6)
    assert telemetry["actual_operator_probe_count"] == 6
    assert telemetry["N_E"] == 1
    assert telemetry["N_grad"] == 1
    assert telemetry["N_G"] == 1
    assert telemetry["N_Q"] == 1
    assert telemetry["N_Hv"] == 1
    assert telemetry["N_cross"] == 1
    assert telemetry["phase1_to_phase2_reuse_count"] == 2
    assert telemetry["phase2_to_batch_reuse_count"] == 2
    assert telemetry["batch_to_growth_reuse_count"] == 2
    assert telemetry["matrix_element_diagnostics"]["G_CC_elements"] == 9
    assert telemetry["statevector_shortcut_used"] is True
    assert telemetry["primitive_count_reconciliation"]["count_equal"] is True


def test_same_value_different_state_or_branch_is_a_distinct_query():
    baseline = _primitive("energy", "energy", state="state-1", branch="branch-1")
    different_state = _primitive(
        "energy", "energy", state="state-2", branch="branch-1"
    )
    different_branch = _primitive(
        "energy", "energy", state="state-1", branch="branch-2"
    )
    ledger = QueryPrimitiveLedger()
    ledger.consume_receipt(
        _receipt(requested=(baseline, different_state, different_branch)),
        consumer_phase="phase1",
    )
    assert ledger.telemetry()["N_E"] == 3


def test_primitive_set_reconciliation_requires_set_equality_not_scalar_count():
    left = {_primitive("energy", "e1").primitive_id, _primitive("energy", "e2").primitive_id}
    right = {_primitive("energy", "e1").primitive_id, _primitive("energy", "e3").primitive_id}
    mismatch = reconcile_primitive_id_sets(
        baseline_primitive_ids=left, enriched_primitive_ids=right
    )
    assert mismatch["baseline_count"] == mismatch["enriched_count"] == 2
    assert not mismatch["set_equal"]
    assert not mismatch["zero_incremental_queries"]
    exact = reconcile_primitive_id_sets(
        baseline_primitive_ids=left, enriched_primitive_ids=reversed(sorted(left))
    )
    assert exact["set_equal"]
    assert exact["zero_incremental_queries"]


def test_phase2_rejects_finite_hessian_without_second_derivative_receipt():
    _, _, workspace, _ = _population()
    gradient = _primitive("coordinate_gradient", "not-Q")
    with pytest.raises(ValueError, match="coordinate_second_derivative"):
        Phase2OrdinaryHessianBlocks(
            workspace_fingerprint=workspace.workspace_fingerprint,
            candidate_keys=workspace.candidate_keys,
            Q_AA=np.eye(workspace.anchor.active_dimension),
            Q_AC=np.zeros(
                (workspace.anchor.active_dimension, len(workspace.candidate_keys))
            ),
            Q_CC=np.eye(len(workspace.candidate_keys)),
            source_query_receipts=(_receipt(requested=(gradient,)),),
            provenance_by_block={
                "Q_AA": ORDINARY_HESSIAN_PROVENANCE,
                "Q_AC": ORDINARY_HESSIAN_PROVENANCE,
                "Q_CC": ORDINARY_HESSIAN_PROVENANCE,
            },
        )


def test_redundant_candidate_subset_fails_supported_rank_gate():
    anchor, _ = _anchor()
    tangent = np.array([0.2, 0.1, 0.95, 0.0])
    left, _ = _candidate(anchor, 0, tangent, -0.5)
    right, _ = _candidate(anchor, 1, tangent, 0.35)
    workspace = build_query_closed_population_workspace(
        anchor=anchor,
        candidate_records=(left, right),
    )
    blocks, _, _ = _ordinary_hessian(workspace, ())
    result = solve_phase2_query_closed_subset(
        workspace=workspace,
        ordinary_hessian=blocks,
        candidate_indices=(0, 1),
    )
    assert result.feasible is False
    assert result.reason == "candidate_subset_supported_rank_gate_failed"


def test_query_ledger_checkpoint_merge_and_later_reuse_are_exact():
    gradient = _primitive("coordinate_gradient", "checkpoint-gradient")
    source = QueryPrimitiveLedger()
    source.consume_receipt(
        _receipt(requested=(gradient,), fields=("gradient",)),
        consumer_phase="phase1",
    )
    source.reuse_known_primitives(
        (gradient.primitive_id,), consumer_phase="growth"
    )
    restored = QueryPrimitiveLedger.from_checkpoint_payload(
        source.checkpoint_payload()
    )
    assert restored.checkpoint_payload() == source.checkpoint_payload()
    telemetry = restored.telemetry()
    assert telemetry["N_grad"] == 1
    assert gradient.primitive_id in telemetry["unique_primitive_ids_reused"]
    assert telemetry["primitive_to_consumer_phases"][gradient.primitive_id] == [
        "growth",
        "phase1",
    ]

    energy = _primitive("energy", "discarded-energy")
    branch = QueryPrimitiveLedger()
    branch.consume_receipt(
        _receipt(requested=(energy,), fields=("energy",)),
        consumer_phase="discarded_trial",
    )
    restored.merge(branch)
    merged = restored.telemetry()
    assert merged["N_grad"] == 1
    assert merged["N_E"] == 1


def test_geometry_element_accounting_is_dimension_aware_and_deduplicated():
    metric = _primitive("tangent_or_metric", "full-gram")
    cross = _primitive("cross_state_tangent", "frame-cross")
    ledger = QueryPrimitiveLedger()
    ledger.consume_receipt(
        _receipt(requested=(metric, cross), fields=("metric", "cross")),
        consumer_phase="optimizer_exact_anchor",
    )

    assert ledger.record_geometry_element_accounting(
        metric.primitive_id,
        geometry_kind=GEOMETRY_ELEMENT_FULL_SYMMETRIC_GRAM,
        row_dimension=3,
        column_dimension=3,
    )
    assert ledger.record_geometry_element_accounting(
        cross.primitive_id,
        geometry_kind=GEOMETRY_ELEMENT_CROSS_STATE_TANGENT,
        row_dimension=2,
        column_dimension=3,
    )
    # Reusing either logical primitive cannot recharge its matrix elements.
    ledger.reuse_known_primitives(
        (metric.primitive_id, cross.primitive_id), consumer_phase="growth"
    )
    assert not ledger.record_geometry_element_accounting(
        metric.primitive_id,
        geometry_kind=GEOMETRY_ELEMENT_FULL_SYMMETRIC_GRAM,
        row_dimension=3,
        column_dimension=3,
    )

    accounting = ledger.telemetry()["geometry_element_accounting"]
    assert accounting["is_total_S_alg"] is False
    assert accounting["full_symmetric_gram_elements"] == 6
    assert accounting["cross_state_tangent_elements"] == 6
    assert accounting["total_geometry_elements"] == 12
    assert accounting["geometry_primitive_count"] == 2

    checkpoint = ledger.checkpoint_payload()
    assert checkpoint["geometry_element_accounting"] == accounting
    restored = QueryPrimitiveLedger.from_checkpoint_payload(checkpoint)
    assert restored.checkpoint_payload() == checkpoint

    # Branch merge is a set union by primitive identity, not arithmetic sum.
    ledger.merge(restored)
    assert ledger.telemetry()["geometry_element_accounting"] == accounting
    with pytest.raises(ValueError, match="conflicting dimensions"):
        ledger.record_geometry_element_accounting(
            metric.primitive_id,
            geometry_kind=GEOMETRY_ELEMENT_FULL_SYMMETRIC_GRAM,
            row_dimension=4,
            column_dimension=4,
        )


def test_beam_ledger_clone_and_difference_are_branch_isolated():
    common = _primitive(
        "tangent_or_metric", "beam-common", branch="beam_branch:0"
    )
    winner = _primitive(
        "coordinate_gradient", "beam-winner", branch="beam_branch:1"
    )
    discarded = _primitive(
        "coordinate_second_derivative",
        "beam-discarded",
        branch="beam_branch:2",
    )
    root = QueryPrimitiveLedger()
    root.consume_receipt(
        _receipt(requested=(common,), fields=("metric",)),
        consumer_phase="phase1",
    )
    root.record_geometry_element_accounting(
        common.primitive_id,
        geometry_kind=GEOMETRY_ELEMENT_FULL_SYMMETRIC_GRAM,
        row_dimension=2,
        column_dimension=2,
    )
    winner_lineage = root.clone()
    winner_lineage.consume_receipt(
        _receipt(requested=(winner,), fields=("gradient",)),
        consumer_phase="growth",
    )
    discarded_lineage = root.clone()
    discarded_lineage.consume_receipt(
        _receipt(requested=(discarded,), fields=("Q",)),
        consumer_phase="discarded_branch",
    )

    all_executed = QueryPrimitiveLedger()
    all_executed.merge(winner_lineage)
    all_executed.merge(discarded_lineage)
    discarded_only = all_executed.difference(
        winner_lineage.unique_primitive_ids
    )

    assert set(discarded_only.unique_primitive_ids) == {
        discarded.primitive_id
    }
    assert winner_lineage.unique_primitive_ids.isdisjoint(
        discarded_only.unique_primitive_ids
    )
    assert discarded_only.telemetry()["geometry_element_accounting"][
        "total_geometry_elements"
    ] == 0
    # Mutating either child cannot alter the root or its sibling.
    assert discarded.primitive_id not in root.unique_primitive_ids
    assert discarded.primitive_id not in winner_lineage.unique_primitive_ids
