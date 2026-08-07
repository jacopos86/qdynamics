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
    pauli_repaired_closed_scalar_rhs,
)
from paper5.stability.moving_frame_viability import (
    moving_frame_viability_audit,
)
from paper5.stability.reachability_observability import (
    build_drive_aware_word_envelope,
)


class _FixedDriveParameters:
    def __init__(self, parameters: DimerParameters, drive_value: float) -> None:
        self._parameters = parameters
        self._drive_value = float(drive_value)

    def __getattr__(self, name: str):
        return getattr(self._parameters, name)

    def drive_difference(self, time_value: float) -> float:
        del time_value
        return self._drive_value


def test_moving_frame_audit_reaches_rank_saturation_without_a_cap() -> None:
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
    drives = np.array([0.0, 0.1, 0.2, 0.1, 0.0])

    def archive_field(state: np.ndarray, drive_value: float) -> np.ndarray:
        return pauli_repaired_closed_scalar_rhs(
            0.0,
            state,
            _FixedDriveParameters(parameters, drive_value),  # type: ignore[arg-type]
        )

    initial = frame.initialize_memory(
        closed,
        ground_state,
        archive_field,
        drive_value=0.0,
        relative_tolerance=1e-10,
    )
    scenario = FiniteHorizonScenario(
        label="small",
        times=times,
        closed_coordinates=np.repeat(closed[None, :], times.size, axis=0),
        drive_values=drives,
        initial_memory_coordinates=initial.memory_coordinates,
    )
    finite_horizon = finite_horizon_reachable_observable_audit(
        frame,
        (scenario,),
        np.ones(31),
        split_times=(0.1, 0.2, 0.3),
        mandatory_dimension=envelope.layer_dimensions[0],
        relative_tolerance=1e-9,
    )
    audit = moving_frame_viability_audit(
        frame,
        finite_horizon,
        (scenario,),
        np.ones(31),
        archive_field,
    )

    assert audit.orders
    assert audit.orders[0].pair_count == 0
    assert audit.orders[0].minimum_local_order == audit.mandatory_dimension
    assert all(
        0.0 <= order.worst_reachability_residual <= 1.0
        and 0.0 <= order.worst_observability_residual <= 1.0
        and 0.0 <= order.maximum_input_leakage_ratio <= 1.0 + 1e-12
        for order in audit.orders
    )
    assert all(
        left.minimum_local_order < right.minimum_local_order
        or left.maximum_local_order < right.maximum_local_order
        for left, right in zip(audit.orders[:-1], audit.orders[1:], strict=True)
    )
    if audit.first_full_pair_count is not None:
        saturated = next(
            order
            for order in audit.orders
            if order.pair_count == audit.first_full_pair_count
        )
        assert saturated.minimum_local_order == frame.hidden_dimension
        assert saturated.worst_reachability_residual < 1e-7
        assert saturated.worst_observability_residual < 1e-7
        assert saturated.maximum_input_leakage_ratio < 1e-7
        assert saturated.minimum_neighbor_blend_gap > 1.0 - 1e-10
