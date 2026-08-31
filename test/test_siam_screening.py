"""The public SIAM screening configuration surface.

These tests pin the pilot's declared shape (10 qubits, sector 100) and check
that a caller can build every artifact through the public API without reaching
into a register layout, and without touching Hubbard-Holstein code paths.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.time_dynamics.siam_screening import (
    FIXTURE_NB2,
    PILOT_NB4,
    SIAMScreeningConfig,
    build_screening_problem,
)
from src.quantum.siam_model import QUENCHED_HYBRIDIZATION
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix


def test_pilot_declares_the_specified_experiment() -> None:
    p = PILOT_NB4.provenance()
    assert p["n_bath"] == 4
    assert p["u"] == pytest.approx(1.5)
    assert p["v_post_quench"] == pytest.approx(1.0 / np.sqrt(2.0))
    assert p["v_pre_quench"] == 0.0
    assert p["t_final"] == pytest.approx(5.0)
    assert p["num_times"] == 501
    assert p["n_qubits"] == 10
    assert p["sector_dimension"] == 100
    assert p["electron_counts"] == {"up": 3, "down": 2}
    assert p["mode_order"] == "spin_blocked_jordan_wigner"


def test_fixture_declares_the_proof_case() -> None:
    p = FIXTURE_NB2.provenance()
    assert (p["n_qubits"], p["sector_dimension"]) == (6, 9)


def test_reporting_grid_spans_zero_to_t_final() -> None:
    times = PILOT_NB4.times
    assert len(times) == 501
    assert times[0] == 0.0
    assert times[-1] == pytest.approx(5.0)


def test_invalid_configs_are_rejected() -> None:
    with pytest.raises(ValueError, match="num_times"):
        SIAMScreeningConfig(num_times=1)
    with pytest.raises(ValueError, match="t_final"):
        SIAMScreeningConfig(t_final=0.0)
    with pytest.raises(ValueError, match="closed shell"):
        SIAMScreeningConfig(n_bath=3)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("n_bath", 4.0),
        ("n_bath", True),
        ("u", "1.5"),
        ("v_final", False),
        ("hopping", "1.0"),
        ("num_times", 3.0),
        ("t_final", "5.0"),
        ("label", 17),
    ),
)
def test_config_rejects_implicitly_coercible_types(field: str, value) -> None:
    with pytest.raises(TypeError, match=field):
        SIAMScreeningConfig(**{field: value})


@pytest.mark.parametrize("field", ("u", "v_final", "hopping", "t_final"))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_config_rejects_nonfinite_physical_values(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        SIAMScreeningConfig(**{field: value})


def test_problem_builds_every_artifact_through_the_public_api() -> None:
    problem = build_screening_problem(FIXTURE_NB2)
    assert len(problem.pool) == 14
    assert set(problem.observables) >= {
        "m_d",
        "D_d",
        "C_1",
        "C_2",
        "Sigma_1",
        "Sigma_2",
    }
    assert float(np.linalg.norm(problem.initial_state)) == pytest.approx(1.0)
    assert problem.provenance["sector_dimension"] == 9


def test_problem_exact_reference_uses_the_configured_hopping() -> None:
    problem = build_screening_problem(
        SIAMScreeningConfig(
            n_bath=2,
            u=1.5,
            hopping=0.7,
            t_final=0.5,
            num_times=3,
        )
    )
    reference = problem.exact_reference()
    assert reference.hopping == pytest.approx(0.7)


def test_pre_quench_and_post_quench_differ_only_by_hybridization() -> None:
    problem = build_screening_problem(FIXTURE_NB2)
    pre = np.asarray(hamiltonian_matrix(problem.hamiltonian_pre_quench), dtype=complex)
    post = np.asarray(hamiltonian_matrix(problem.hamiltonian), dtype=complex)
    assert not np.allclose(pre, post)
    from src.quantum.siam_model import build_hybridization

    hyb = np.asarray(
        hamiltonian_matrix(
            build_hybridization(FIXTURE_NB2.layout, v=QUENCHED_HYBRIDIZATION)
        ),
        dtype=complex,
    )
    assert np.allclose(post - pre, hyb, atol=1e-12)


def test_prepared_state_is_an_eigenstate_of_the_pre_quench_hamiltonian() -> None:
    """Nothing happens before the quench: the state is stationary at V=0."""

    problem = build_screening_problem(FIXTURE_NB2)
    pre = np.asarray(hamiltonian_matrix(problem.hamiltonian_pre_quench), dtype=complex)
    psi = problem.initial_state
    residual = pre @ psi - (np.vdot(psi, pre @ psi)) * psi
    assert float(np.linalg.norm(residual)) < 1e-10


def test_screening_profile_decays_with_distance() -> None:
    """|C_1| > |C_last|: correlation is strongest at the near shell."""

    problem = build_screening_problem(
        SIAMScreeningConfig(n_bath=4, u=1.5, t_final=5.0, num_times=51)
    )
    ref = problem.exact_reference()
    near = abs(ref.at("C_1")[-1])
    far = abs(ref.at("C_4")[-1])
    assert near > far
    assert ref.at("C_1")[-1] < 0.0
