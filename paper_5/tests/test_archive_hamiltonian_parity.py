"""Coefficient parity of the committed archive right-hand side.

Two checks, each independent of ``closed_scalar_rhs``:

1. **Hamiltonian parity** on the 20 rows that need no factorization.  At exact
   cutoff-16 and cutoff-20 driven states (``t <= 4``), the exact derivative of
   the closed coordinates is contracted from the wavefunction derivative
   (``exact_reference``; no archive code).  The rows of ``rho`` and the real
   shared-trace and diagonal-difference rows of ``C`` are exact Ehrenfest
   equations and must agree to rounding; the ``B``, ``N`` and ``A`` rows carry
   only the phonon-cutoff remainder, which must fall with the cutoff.
2. **Second transcription** of Eqs. (14a)--(14e) of the Paper V draft, written
   from the equations (general coupling tensors ``g^{q+-}_{12}``, explicit
   index sums, spin degeneracy 2 in Eqs. (14b), (14c) and (14e)), compared row
   by row with ``closed_scalar_rhs``.  This covers the 11 factorized ``C``
   rows that the Hamiltonian cannot check.

Each check is shown to catch the mutation ``+ 0.3 x_0 x_3`` in a row it
covers, including row 21 (``correlation_0_01_real``), which every assertion of
``test_quantum_target_form.py`` passes.
"""

from __future__ import annotations

import numpy as np
import pytest

from paper5.stability.exact_reference import exact_holstein_driven_trajectory
from paper5.stability.hubbard_dimer import DimerParameters
from paper5.stability.matrix_reference import (
    CLOSED_SCALAR_STATE_NAMES,
    closed_scalar_rhs,
    closed_scalar_to_matrix_state,
    matrix_state_to_closed_scalar_coordinates,
)

PARAMETERS = DimerParameters()
EHRENFEST_EXACT_ROWS = (0, 1, 2, 17, 19, 25)  # rho; C real shared trace and diagonal differences
CUTOFF_ROWS = {"B": range(3, 7), "N": range(7, 11), "A": range(11, 17)}
FACTORIZED_C_ROWS = (18, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30)
MUTATION_ROW_21 = CLOSED_SCALAR_STATE_NAMES.index("correlation_0_01_real")


def _mutated(row: int):
    def rhs(time, x, parameters):
        out = closed_scalar_rhs(time, x, parameters).copy()
        out[row] += 0.3 * x[0] * x[3]
        return out

    return rhs


@pytest.fixture(scope="module")
def exact_samples():
    times = np.round(np.arange(0.0, 4.0 + 1e-9, 0.05), 10)
    samples = {}
    for cutoff in (16, 20):
        trajectory = exact_holstein_driven_trajectory(
            PARAMETERS, sample_times=times, phonon_cutoff=cutoff, maximum_step=0.01
        )
        states = np.asarray(
            [matrix_state_to_closed_scalar_coordinates(s) for s in trajectory.matrix_states]
        )
        derivatives = np.asarray(
            [matrix_state_to_closed_scalar_coordinates(s) for s in trajectory.matrix_derivatives]
        )
        samples[cutoff] = (times, states, derivatives)
    return samples


def hamiltonian_row_defect(rhs, sample) -> np.ndarray:
    """Per-row max |exact derivative - rhs(t, x_exact)| over the samples."""

    times, states, derivatives = sample
    values = np.asarray([rhs(float(t), x, PARAMETERS) for t, x in zip(times, states)])
    return np.max(np.abs(derivatives - values), axis=0)


def test_ehrenfest_rows_match_the_exact_hamiltonian(exact_samples):
    defect16 = hamiltonian_row_defect(closed_scalar_rhs, exact_samples[16])
    defect20 = hamiltonian_row_defect(closed_scalar_rhs, exact_samples[20])
    for row in EHRENFEST_EXACT_ROWS:
        assert defect16[row] <= 1e-13, CLOSED_SCALAR_STATE_NAMES[row]
        assert defect20[row] <= 1e-13, CLOSED_SCALAR_STATE_NAMES[row]
    bounds16 = {"B": 2e-4, "N": 1e-4, "A": 2e-3}
    bounds20 = {"B": 5e-6, "N": 5e-6, "A": 5e-5}
    for block, rows in CUTOFF_ROWS.items():
        rows = list(rows)
        assert defect16[rows].max() <= bounds16[block], block
        assert defect20[rows].max() <= bounds20[block], block
        # the remainder is the phonon cutoff's: it falls by at least 20x from 16 to 20
        assert defect20[rows].max() <= defect16[rows].max() / 20.0, block


@pytest.mark.parametrize("row", [0, 3, 7, 11, 17, 19])
def test_hamiltonian_parity_detects_a_mutation(exact_samples, row):
    defect = hamiltonian_row_defect(_mutated(row), exact_samples[20])
    assert defect[row] > 1e-2


# ---------------------------------------------------------------------------
# Second transcription of Eqs. (14a)--(14e)
# ---------------------------------------------------------------------------


def transcribed_closed_rhs(time: float, x: np.ndarray, parameters: DimerParameters) -> np.ndarray:
    """Eqs. (14a)--(14e) and (15) of the draft, transcribed independently.

    Two sites, two local modes; ``g^{q-}_{12} = g^{q+}_{12} = g delta_{1q}
    delta_{2q}``; ``v_ext = diag(f/2, -f/2)``; one spin channel for ``rho`` and
    ``delta rho^q``, spin degeneracy 2 in the phonon and coherent-field sums.
    """

    s = 2.0
    state = closed_scalar_to_matrix_state(np.asarray(x, dtype=float))
    rho = np.asarray(state.electron_density, dtype=complex)
    b = np.asarray(state.coherent_phonon, dtype=complex)
    nmat = np.asarray(state.phonon_density, dtype=complex)
    amat = np.asarray(state.anomalous_phonon_density, dtype=complex)
    corr = np.asarray(state.electron_phonon_correlation, dtype=complex)
    g = np.zeros((2, 2, 2))
    for q in range(2):
        g[q, q, q] = parameters.coupling
    gm = gp = g
    w = np.full(2, parameters.omega_ph)
    f = parameters.drive_difference(time)
    one = np.eye(2)

    # Eq. (15)
    h = np.array([[0.0, -parameters.hopping], [-parameters.hopping, 0.0]], dtype=complex)
    h = h + np.diag([0.5 * f, -0.5 * f])
    for q in range(2):
        h = h + gm[q] * (b[q] + np.conj(b[q]))

    # Eq. (14a)
    drho = -1j * (h @ rho - rho @ h)
    bracket = np.zeros((2, 2), dtype=complex)
    for a in range(2):
        for c in range(2):
            for q in range(2):
                for k in range(2):
                    bracket[a, c] += -1j * (
                        gm[q, a, k] * corr[q, k, c] + gp[q, k, a] * np.conj(corr[q, c, k])
                    )
    drho = drho + bracket + bracket.conj().T

    # Eq. (14b)
    dn = np.zeros((2, 2), dtype=complex)
    for q in range(2):
        for qp in range(2):
            total = -1j * (w[q] - w[qp]) * nmat[q, qp]
            for a in range(2):
                for c in range(2):
                    total += -1j * s * (
                        -gm[qp, a, c] * corr[q, c, a] + gp[q, a, c] * np.conj(corr[qp, c, a])
                    )
            dn[q, qp] = total

    # Eq. (14c)
    da = np.zeros((2, 2), dtype=complex)
    for qp in range(2):
        for q in range(2):
            total = -1j * (w[qp] + w[q]) * amat[qp, q]
            for a in range(2):
                for c in range(2):
                    total += -1j * s * (gp[q, a, c] * corr[qp, a, c] + gp[qp, a, c] * corr[q, a, c])
            da[qp, q] = total

    # Eq. (14d)
    dc = np.zeros((2, 2, 2), dtype=complex)
    for q in range(2):
        dc[q] = -1j * (h @ corr[q] - corr[q] @ h + w[q] * corr[q])
        for a in range(2):
            for bb in range(2):
                total = 0.0j
                for qp in range(2):
                    occupation = (1.0 if q == qp else 0.0) + nmat[q, qp]
                    for c in range(2):
                        for d in range(2):
                            total += -1j * gp[qp, c, d] * occupation * (one[a, d] - rho[a, d]) * rho[c, bb]
                            total += 1j * gp[qp, c, d] * nmat[q, qp] * (one[c, bb] - rho[c, bb]) * rho[a, d]
                    for c in range(2):
                        total += -1j * gm[qp, bb, c] * amat[qp, q] * rho[c, a]
                        total += 1j * gm[qp, c, bb] * amat[qp, q] * rho[a, c]
                dc[q, a, bb] += total

    # Eq. (14e)
    db = np.zeros(2, dtype=complex)
    for q in range(2):
        total = w[q] * b[q]
        for a in range(2):
            for c in range(2):
                total += s * gp[q, a, c] * rho[a, c]
        db[q] = -1j * total

    derivative = type(state)(
        electron_density=drho,
        coherent_phonon=db,
        phonon_density=dn,
        anomalous_phonon_density=da,
        electron_phonon_correlation=dc,
    )
    return matrix_state_to_closed_scalar_coordinates(derivative)


def transcription_row_defect(rhs, *, samples: int = 60, seed: int = 41) -> np.ndarray:
    rng = np.random.default_rng(seed)
    worst = np.zeros(31)
    for _ in range(samples):
        x = rng.normal(size=31) * float(rng.choice([0.3, 1.0, 2.0]))
        t = float(rng.uniform(0.0, 4.0))
        reference = transcribed_closed_rhs(t, x, PARAMETERS)
        scale = max(1.0, float(np.max(np.abs(reference))))
        worst = np.maximum(worst, np.abs(rhs(t, x, PARAMETERS) - reference) / scale)
    return worst


def test_second_transcription_matches_every_row():
    defect = transcription_row_defect(closed_scalar_rhs)
    assert defect.max() <= 1e-13
    # including the 11 factorized correlation rows the Hamiltonian cannot check
    assert defect[list(FACTORIZED_C_ROWS)].max() <= 1e-13


@pytest.mark.parametrize("row", [MUTATION_ROW_21, 18, 30])
def test_second_transcription_detects_a_mutation(row):
    defect = transcription_row_defect(_mutated(row), samples=20)
    assert defect[row] > 1e-3
