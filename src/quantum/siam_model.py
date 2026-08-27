r"""Nonequilibrium single-impurity Anderson model (SIAM) on a finite bath.

The application is **finite-box impurity screening after a sudden hybridization
quench** --- a mesoscopic screening precursor, never a thermodynamic Kondo
cloud.  A finite bath of :math:`N_b` sites cannot host a Kondo cloud; what it
can show is the onset of screening correlations propagating outward from the
impurity after the coupling is switched on.

Hamiltonian, particle-hole symmetric at the impurity:

.. math::

    H = -t_b\sum_{j=1}^{N_b-1}\sum_\sigma
            (c^\dagger_{j\sigma}c_{j+1,\sigma} + \mathrm{h.c.})
        + U\,(n_{d\uparrow}-\tfrac12)(n_{d\downarrow}-\tfrac12)
        - V(t)\sum_\sigma (d^\dagger_\sigma c_{1\sigma} + \mathrm{h.c.})

with :math:`t_b = 1`, :math:`V(t<0) = 0`, and :math:`V(t\ge0) = 1/\sqrt2`.

Mode ordering is **spin-blocked** Jordan--Wigner, without tapering::

    (d_up, c1_up, ..., cNb_up,  d_down, c1_down, ..., cNb_down)

so the register is :math:`2(N_b+1)` qubits.  Spin-blocking (rather than
interleaving) keeps every same-spin hopping string short, since a hop between
adjacent chain sites spans one Jordan--Wigner Z and never crosses the spin
boundary.

The prepared state is the decoupled closed-shell bath Fermi sea with one
spin-up electron on the impurity,

.. math::

    |\psi_0\rangle = d^\dagger_\uparrow
        \prod_{m=1}^{N_b/2} f^\dagger_{m\uparrow} f^\dagger_{m\downarrow}
        |0\rangle,

where :math:`f_m` are the single-particle eigenmodes of the *decoupled* open
bath chain.  The target symmetry sector is therefore
:math:`N_\uparrow = N_b/2 + 1`, :math:`N_\downarrow = N_b/2`; for
:math:`N_b = 2` that is 6 qubits and sector dimension 9, and for
:math:`N_b = 4` it is 10 qubits and sector dimension 100.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator, Sequence

import numpy as np

from src.quantum.pauli_polynomial_class import (
    PauliPolynomial,
    fermion_minus_operator,
    fermion_plus_operator,
)

SIAM_MODEL_V1 = "siam_finite_box_screening_v1"
DEFAULT_HOPPING = 1.0
QUENCHED_HYBRIDIZATION = 1.0 / np.sqrt(2.0)

UP = "up"
DOWN = "down"
SPINS = (UP, DOWN)


# ---------------------------------------------------------------------------
# Register layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SIAMLayout:
    """Spin-blocked Jordan--Wigner mode layout for a bath of ``n_bath`` sites."""

    n_bath: int

    def __post_init__(self) -> None:
        if int(self.n_bath) < 2 or int(self.n_bath) % 2 != 0:
            raise ValueError(
                "n_bath must be a positive even number so the decoupled bath "
                f"is closed shell; got {self.n_bath}."
            )

    @property
    def modes_per_spin(self) -> int:
        return int(self.n_bath) + 1

    @property
    def n_qubits(self) -> int:
        return 2 * self.modes_per_spin

    def spin_offset(self, spin: str) -> int:
        if spin == UP:
            return 0
        if spin == DOWN:
            return self.modes_per_spin
        raise ValueError(f"spin must be 'up' or 'down'; got {spin!r}")

    def impurity(self, spin: str) -> int:
        """Mode index of the impurity orbital for ``spin``."""

        return self.spin_offset(spin)

    def bath(self, site: int, spin: str) -> int:
        """Mode index of bath site ``site`` (1-based, nearest the impurity is 1)."""

        if not 1 <= int(site) <= int(self.n_bath):
            raise ValueError(
                f"bath site must lie in 1..{self.n_bath}; got {site}."
            )
        return self.spin_offset(spin) + int(site)

    def electron_counts(self) -> tuple[int, int]:
        """``(N_up, N_down)`` of the target sector."""

        half = int(self.n_bath) // 2
        return half + 1, half


# ---------------------------------------------------------------------------
# Fermionic primitives
# ---------------------------------------------------------------------------


def _plus(layout: SIAMLayout, mode: int) -> PauliPolynomial:
    return fermion_plus_operator("JW", layout.n_qubits, int(mode))


def _minus(layout: SIAMLayout, mode: int) -> PauliPolynomial:
    return fermion_minus_operator("JW", layout.n_qubits, int(mode))


def number_operator(layout: SIAMLayout, mode: int) -> PauliPolynomial:
    r""":math:`n_p = c^\dagger_p c_p`."""

    return _plus(layout, mode) * _minus(layout, mode)


def hopping_term(
    layout: SIAMLayout, mode_a: int, mode_b: int, *, amplitude: float
) -> PauliPolynomial:
    r""":math:`-\,\text{amplitude}\,(c^\dagger_a c_b + \mathrm{h.c.})`."""

    forward = _plus(layout, mode_a) * _minus(layout, mode_b)
    backward = _plus(layout, mode_b) * _minus(layout, mode_a)
    return (-float(amplitude)) * (forward + backward)


def total_number_operator(layout: SIAMLayout, spin: str) -> PauliPolynomial:
    r""":math:`N_\sigma`, summed over the impurity and every bath site."""

    total = number_operator(layout, layout.impurity(spin))
    for site in range(1, layout.n_bath + 1):
        total = total + number_operator(layout, layout.bath(site, spin))
    return total


# ---------------------------------------------------------------------------
# Hamiltonian
# ---------------------------------------------------------------------------


def build_bath_kinetic(
    layout: SIAMLayout, *, hopping: float = DEFAULT_HOPPING
) -> PauliPolynomial:
    r""":math:`-t_b\sum_{j=1}^{N_b-1}\sum_\sigma(c^\dagger_{j\sigma}c_{j+1,\sigma}+\mathrm{h.c.})`."""

    terms: PauliPolynomial | None = None
    for spin in SPINS:
        for site in range(1, layout.n_bath):
            block = hopping_term(
                layout,
                layout.bath(site, spin),
                layout.bath(site + 1, spin),
                amplitude=hopping,
            )
            terms = block if terms is None else terms + block
    if terms is None:  # pragma: no cover - n_bath >= 2 guarantees a term
        raise ValueError("bath kinetic term is empty; n_bath must exceed 1.")
    return terms


def build_impurity_interaction(
    layout: SIAMLayout, *, u: float
) -> PauliPolynomial:
    r""":math:`U(n_{d\uparrow}-\tfrac12)(n_{d\downarrow}-\tfrac12)`.

    Written in centered form so the impurity is particle-hole symmetric: the
    expansion contributes :math:`U n_\uparrow n_\downarrow`, the two
    :math:`-U/2` chemical-potential pieces, and the constant :math:`U/4`.
    """

    identity = identity_polynomial(layout)
    centered_up = number_operator(layout, layout.impurity(UP)) - 0.5 * identity
    centered_dn = number_operator(layout, layout.impurity(DOWN)) - 0.5 * identity
    return float(u) * (centered_up * centered_dn)


def build_hybridization(
    layout: SIAMLayout, *, v: float
) -> PauliPolynomial:
    r""":math:`-V\sum_\sigma(d^\dagger_\sigma c_{1\sigma}+\mathrm{h.c.})`."""

    terms: PauliPolynomial | None = None
    for spin in SPINS:
        block = hopping_term(
            layout,
            layout.impurity(spin),
            layout.bath(1, spin),
            amplitude=v,
        )
        terms = block if terms is None else terms + block
    assert terms is not None
    return terms


def build_siam_hamiltonian(
    n_bath: int,
    *,
    u: float,
    v: float,
    hopping: float = DEFAULT_HOPPING,
) -> PauliPolynomial:
    """Full SIAM Hamiltonian at hybridization ``v``.

    The quench is expressed by evaluating this at ``v=0`` before the quench and
    at ``v=QUENCHED_HYBRIDIZATION`` after it; the interaction and bath terms are
    unchanged, so the two share every operator except the hybridization block.
    """

    layout = SIAMLayout(int(n_bath))
    return (
        build_bath_kinetic(layout, hopping=hopping)
        + build_impurity_interaction(layout, u=u)
        + build_hybridization(layout, v=v)
    )


def identity_polynomial(layout: SIAMLayout) -> PauliPolynomial:
    """Identity on the full register, via the shared Hubbard-module helper."""

    from src.quantum.hubbard_latex_python_pairs import _identity_pauli_polynomial

    return _identity_pauli_polynomial("JW", layout.n_qubits)


# ---------------------------------------------------------------------------
# Symmetry sector
# ---------------------------------------------------------------------------


def _occupied_modes(index: int, n_qubits: int) -> tuple[int, ...]:
    """Modes occupied in computational basis state ``index`` (mode 0 = LSB)."""

    return tuple(m for m in range(n_qubits) if (index >> m) & 1)


@lru_cache(maxsize=None)
def sector_indices(n_bath: int) -> tuple[int, ...]:
    """Computational basis indices spanning the target ``(N_up, N_down)`` sector."""

    layout = SIAMLayout(int(n_bath))
    n_up, n_dn = layout.electron_counts()
    per_spin = layout.modes_per_spin
    out = []
    for index in range(2 ** layout.n_qubits):
        occ = _occupied_modes(index, layout.n_qubits)
        up = sum(1 for m in occ if m < per_spin)
        if up != n_up:
            continue
        if len(occ) - up != n_dn:
            continue
        out.append(index)
    return tuple(out)


def project_to_sector(matrix: np.ndarray, n_bath: int) -> np.ndarray:
    """Restrict a full-register operator to the target sector."""

    idx = np.asarray(sector_indices(int(n_bath)), dtype=int)
    return np.asarray(matrix, dtype=complex)[np.ix_(idx, idx)]


def sector_vector(state: np.ndarray, n_bath: int) -> np.ndarray:
    idx = np.asarray(sector_indices(int(n_bath)), dtype=int)
    return np.asarray(state, dtype=complex).reshape(-1)[idx]


def embed_sector_vector(vector: np.ndarray, n_bath: int) -> np.ndarray:
    layout = SIAMLayout(int(n_bath))
    out = np.zeros(2 ** layout.n_qubits, dtype=complex)
    out[np.asarray(sector_indices(int(n_bath)), dtype=int)] = np.asarray(
        vector, dtype=complex
    ).reshape(-1)
    return out


# ---------------------------------------------------------------------------
# Prepared state
# ---------------------------------------------------------------------------


def bath_eigenmodes(n_bath: int, *, hopping: float = DEFAULT_HOPPING):
    r"""Single-particle eigenmodes of the decoupled open bath chain.

    For an open chain of ``L`` sites with hopping ``-t_b`` the eigenvalues are
    :math:`\varepsilon_m = -2 t_b\cos(k_m)` with :math:`k_m = \pi m/(L+1)` and
    eigenvectors :math:`\phi_m(j) = \sqrt{2/(L+1)}\,\sin(k_m j)`.  Returned in
    ascending energy so filling the first ``L/2`` gives the closed shell.
    """

    size = int(n_bath)
    chain = np.zeros((size, size), dtype=float)
    for j in range(size - 1):
        chain[j, j + 1] = chain[j + 1, j] = -float(hopping)
    energies, vectors = np.linalg.eigh(chain)
    return energies, vectors


def build_initial_state(
    n_bath: int, *, hopping: float = DEFAULT_HOPPING
) -> np.ndarray:
    r"""``d^dagger_up prod_m f^dagger_{m up} f^dagger_{m down} |0>``.

    Built by applying Jordan--Wigner creation operators to the vacuum in a fixed
    order, so the fermionic sign is carried by the operators themselves rather
    than by a hand-tracked permutation.
    """

    from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix

    layout = SIAMLayout(int(n_bath))
    _energies, vectors = bath_eigenmodes(int(n_bath), hopping=hopping)
    filled = int(n_bath) // 2

    state = np.zeros(2 ** layout.n_qubits, dtype=complex)
    state[0] = 1.0  # vacuum: every mode empty

    # Bath modes first, then the impurity electron, matching the written order
    # d^dagger_up prod_m f^dagger_{m,up} f^dagger_{m,down} |0>: operators are
    # applied right-to-left, so the product over m acts before d^dagger_up.
    for mode in reversed(range(filled)):
        for spin in (DOWN, UP):
            amplitudes = vectors[:, mode]
            creator: PauliPolynomial | None = None
            for site in range(1, layout.n_bath + 1):
                weight = float(amplitudes[site - 1])
                if abs(weight) < 1.0e-15:
                    continue
                block = weight * _plus(layout, layout.bath(site, spin))
                creator = block if creator is None else creator + block
            assert creator is not None
            state = np.asarray(hamiltonian_matrix(creator), dtype=complex) @ state

    state = np.asarray(
        hamiltonian_matrix(_plus(layout, layout.impurity(UP))), dtype=complex
    ) @ state

    norm = float(np.linalg.norm(state))
    if norm <= 0.0:  # pragma: no cover - filled shell always has weight
        raise ValueError("prepared SIAM state has zero norm.")
    return state / norm


__all__ = [
    "DEFAULT_HOPPING",
    "DOWN",
    "QUENCHED_HYBRIDIZATION",
    "SIAMLayout",
    "SIAM_MODEL_V1",
    "SPINS",
    "UP",
    "bath_eigenmodes",
    "build_bath_kinetic",
    "build_hybridization",
    "build_impurity_interaction",
    "build_initial_state",
    "build_siam_hamiltonian",
    "embed_sector_vector",
    "hopping_term",
    "identity_polynomial",
    "number_operator",
    "project_to_sector",
    "sector_indices",
    "sector_vector",
    "total_number_operator",
]
