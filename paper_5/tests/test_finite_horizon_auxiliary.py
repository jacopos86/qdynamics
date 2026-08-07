from __future__ import annotations

import numpy as np

from paper5.stability.archive_auxiliary_memory import (
    build_archive_auxiliary_frame_from_observables,
)
from paper5.stability.exact_reference import (
    _build_exact_dimer_model,
    _contract_matrix_state,
    _ground_state,
)
from paper5.stability.finite_horizon_auxiliary import (
    FiniteHorizonScenario,
    finite_horizon_reachable_observable_audit,
)
from paper5.stability.hubbard_dimer import DimerParameters
from paper5.stability.matrix_reference import (
    matrix_state_to_closed_scalar_coordinates,
)
from paper5.stability.reachability_observability import (
    build_drive_aware_word_envelope,
)


def _small_audit():
    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5)
    model = _build_exact_dimer_model(parameters, phonon_cutoff=2)
    _, ground_state = _ground_state(model, eigensolver_tolerance=1e-12)
    closed = matrix_state_to_closed_scalar_coordinates(
        _contract_matrix_state(model, ground_state)
    )
    envelope = build_drive_aware_word_envelope(
        parameters,
        phonon_cutoff=2,
        maximum_word_depth=1,
        rank_tolerance=1e-10,
        preparation_state_vectors=(ground_state,),
    )
    frame = build_archive_auxiliary_frame_from_observables(
        envelope.construction,
        envelope.hidden_observables,
    )
    times = np.linspace(0.0, 0.4, 5)
    scenario = FiniteHorizonScenario(
        label="small",
        times=times,
        closed_coordinates=np.repeat(closed[None, :], times.size, axis=0),
        drive_values=np.array([0.0, 0.1, 0.2, 0.1, 0.0]),
        initial_memory_coordinates=np.linspace(
            0.1,
            0.2,
            frame.hidden_dimension,
        ),
    )
    audit = finite_horizon_reachable_observable_audit(
        frame,
        (scenario,),
        np.ones(31),
        split_times=(0.1, 0.2, 0.3),
        mandatory_dimension=envelope.layer_dimensions[0],
        relative_tolerance=1e-9,
    )
    return audit


def test_finite_horizon_audit_builds_supported_balanced_splits() -> None:
    audit = _small_audit()

    assert len(audit.split_audits) == 3
    assert audit.supported_pair_count > 0
    for split in audit.split_audits:
        assert split.reachability_rank > 0
        assert split.observability_rank > 0
        assert np.all(np.diff(split.hankel_singular_values) <= 0.0)
        np.testing.assert_allclose(
            split.dual_directions.T @ split.primal_directions,
            np.eye(split.hankel_singular_values.size),
            atol=2e-8,
        )


def test_orthogonal_candidate_frames_retain_mandatory_entrance_range() -> None:
    audit = _small_audit()
    pair_count = min(3, audit.supported_pair_count)
    frame = audit.orthogonal_frame(pair_count)
    mandatory = np.eye(audit.hidden_dimension, audit.mandatory_dimension)

    np.testing.assert_allclose(frame.T @ frame, np.eye(frame.shape[1]), atol=1e-11)
    np.testing.assert_allclose(
        mandatory - frame @ (frame.T @ mandatory),
        0.0,
        atol=1e-10,
    )
    orders = audit.actual_order_curve()
    assert orders[0] == audit.mandatory_dimension
    assert np.all(np.diff(orders) >= 0)
    assert orders[-1] <= audit.hidden_dimension

    local = audit.split_audits[0].orthogonal_frame(
        hidden_dimension=audit.hidden_dimension,
        mandatory_dimension=audit.mandatory_dimension,
        pair_count=pair_count,
        relative_tolerance=audit.relative_tolerance,
    )
    np.testing.assert_allclose(
        local.T @ local,
        np.eye(local.shape[1]),
        atol=1e-11,
    )
    reachability_residual, observability_residual = (
        audit.split_audits[0].projection_residuals(local)
    )
    assert 0.0 <= reachability_residual <= 1.0
    assert 0.0 <= observability_residual <= 1.0


def test_optimal_hankel_tail_is_monotone() -> None:
    audit = _small_audit()
    defects = np.asarray(
        [
            audit.worst_optimal_relative_defect(rank)
            for rank in range(audit.hidden_dimension + 1)
        ]
    )

    assert np.all(np.diff(defects) <= 1e-14)
    assert defects[-1] == 0.0
