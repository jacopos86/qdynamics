"""Exact-propagation checks for the SIAM screening reference.

The post-quench Hamiltonian is time-independent, so two independent
propagators must agree exactly: spectral evaluation and Krylov. Where a check
is about physics rather than numerics it is written against a quantity derived
independently of the code under test -- a Heisenberg commutator, a conservation
law, or a finite difference.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.quantum.siam_model import (
    QUENCHED_HYBRIDIZATION,
    SIAMLayout,
    UP,
    DOWN,
    build_initial_state,
    number_operator,
    project_to_sector,
    sector_vector,
)
from src.quantum.siam_observables import (
    impurity_current_operator,
    integrated_screening,
    spin_correlation_operator,
)
from src.quantum.siam_reference import (
    exact_reference,
    propagate_sector,
    sector_hamiltonian,
)
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix

N_BATH = 2
U = 1.5
TIMES = (0.0, 0.5, 1.5, 3.0, 5.0)


def _dense(poly) -> np.ndarray:
    return np.asarray(hamiltonian_matrix(poly), dtype=complex)


def test_sector_hamiltonian_has_the_fixture_dimension() -> None:
    assert sector_hamiltonian(N_BATH, u=U, v=QUENCHED_HYBRIDIZATION).shape == (9, 9)


def test_spectral_and_krylov_propagation_agree() -> None:
    spectral = propagate_sector(N_BATH, u=U, times=TIMES, method="eigh")
    krylov = propagate_sector(N_BATH, u=U, times=TIMES, method="krylov")
    for a, b in zip(spectral, krylov):
        assert np.linalg.norm(a - b) < 1e-10


def test_propagation_preserves_the_norm() -> None:
    for state in propagate_sector(N_BATH, u=U, times=TIMES):
        assert float(np.linalg.norm(state)) == pytest.approx(1.0, abs=1e-12)


def test_propagation_at_zero_time_returns_the_prepared_state() -> None:
    first = propagate_sector(N_BATH, u=U, times=[0.0])[0]
    expected = sector_vector(build_initial_state(N_BATH), N_BATH)
    assert np.allclose(first, expected, atol=1e-12)


def test_energy_is_conserved_after_the_quench() -> None:
    h = sector_hamiltonian(N_BATH, u=U, v=QUENCHED_HYBRIDIZATION)
    energies = [
        float(np.real(np.vdot(s, h @ s)))
        for s in propagate_sector(N_BATH, u=U, times=TIMES)
    ]
    assert max(energies) - min(energies) < 1e-10


def test_unknown_propagation_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown propagation method"):
        propagate_sector(N_BATH, u=U, times=[0.0], method="nope")


def test_reference_reports_every_gated_observable() -> None:
    ref = exact_reference(N_BATH, u=U, times=TIMES)
    assert set(ref.observables) >= {"m_d", "D_d", "C_1", "C_2"}
    for series in ref.observables.values():
        assert len(series) == len(TIMES)


def test_screening_starts_from_the_decoupled_values() -> None:
    ref = exact_reference(N_BATH, u=U, times=TIMES)
    assert ref.at("m_d")[0] == pytest.approx(1.0, abs=1e-12)
    assert ref.at("D_d")[0] == pytest.approx(0.0, abs=1e-12)
    assert ref.at("C_1")[0] == pytest.approx(0.0, abs=1e-12)


def test_nearest_shell_correlation_turns_antiferromagnetic() -> None:
    """The screening signature: the impurity antialigns with the near shell."""

    ref = exact_reference(N_BATH, u=U, times=(0.0, 1.0, 1.5, 2.0, 3.0))
    assert min(ref.at("C_1")) < -1e-3


def test_short_time_derivative_of_double_occupancy_vanishes() -> None:
    """d<D_d>/dt = 0 at t=0: the hybridization needs one order to act."""

    h = 1e-4
    ref = exact_reference(N_BATH, u=U, times=(0.0, h, 2 * h))
    d0, d1, d2 = ref.at("D_d")
    assert (d1 - d0) / h == pytest.approx(0.0, abs=1e-3)
    assert d2 > d0  # but it does grow at second order


def test_impurity_occupation_is_pinned_by_particle_hole_symmetry() -> None:
    """<N_d> = 1 exactly, so the impurity current vanishes identically.

    This is a property of the particle-hole-symmetric impurity at half filling,
    not an artifact of the fixture: it holds at every U and every bath size
    checked. The current is therefore reported as a diagnostic that carries no
    screening information here.
    """

    layout = SIAMLayout(N_BATH)
    n_d = project_to_sector(
        _dense(
            number_operator(layout, layout.impurity(UP))
            + number_operator(layout, layout.impurity(DOWN))
        ),
        N_BATH,
    )
    current = project_to_sector(
        _dense(impurity_current_operator(layout, v=QUENCHED_HYBRIDIZATION)), N_BATH
    )
    for u in (0.0, U, 4.0):
        for state in propagate_sector(N_BATH, u=u, times=TIMES):
            assert float(np.real(np.vdot(state, n_d @ state))) == pytest.approx(
                1.0, abs=1e-10
            )
            assert float(np.real(np.vdot(state, current @ state))) == pytest.approx(
                0.0, abs=1e-10
            )


def test_impurity_current_equals_the_heisenberg_derivative() -> None:
    """<I_d> = <i[H, N_d]> operator-wise, independent of the state."""

    layout = SIAMLayout(N_BATH)
    h = sector_hamiltonian(N_BATH, u=U, v=QUENCHED_HYBRIDIZATION)
    n_d = project_to_sector(
        _dense(
            number_operator(layout, layout.impurity(UP))
            + number_operator(layout, layout.impurity(DOWN))
        ),
        N_BATH,
    )
    current = project_to_sector(
        _dense(impurity_current_operator(layout, v=QUENCHED_HYBRIDIZATION)), N_BATH
    )
    assert np.allclose(current, 1.0j * (h @ n_d - n_d @ h), atol=1e-10)


def test_integrated_screening_accumulates_shells() -> None:
    ref = exact_reference(N_BATH, u=U, times=(3.0,))
    shells = {"C_1": ref.at("C_1")[0], "C_2": ref.at("C_2")[0]}
    assert integrated_screening(shells, 1) == pytest.approx(shells["C_1"])
    assert integrated_screening(shells, 2) == pytest.approx(
        shells["C_1"] + shells["C_2"]
    )
    with pytest.raises(KeyError, match="C_3"):
        integrated_screening(shells, 3)


def test_reference_is_reporting_only_and_takes_no_controller_input() -> None:
    """The reference API accepts physics and times, never a controller state."""

    import inspect

    params = set(inspect.signature(exact_reference).parameters)
    assert params == {"n_bath", "u", "v", "times", "method"}


def test_reference_preserves_custom_hopping_and_one_shot_time_iterables() -> None:
    times = (value for value in (0.0, 0.25, 0.5))
    reference = exact_reference(N_BATH, u=U, hopping=0.7, times=times)
    assert reference.hopping == pytest.approx(0.7)
    assert reference.times == pytest.approx((0.0, 0.25, 0.5))
    assert all(len(series) == 3 for series in reference.observables.values())
