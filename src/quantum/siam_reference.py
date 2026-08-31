r"""Exact fixed-sector reference propagation for finite-box SIAM screening.

The quench is a sudden change of a *time-independent* Hamiltonian: :math:`V=0`
for :math:`t<0` and :math:`V=1/\sqrt2` for :math:`t\ge0`.  After the quench the
generator is constant, so the exact solution is a single matrix exponential
rather than a time-ordered product, and it may be evaluated directly at any
reporting time without stepping.

Everything runs inside the :math:`(N_\uparrow,N_\downarrow)` sector: for
:math:`N_b=4` that is a 100-dimensional problem instead of 1024, and the
saving grows quickly with bath size.

**This reference is reporting-only.**  It must never be handed to an online
controller, a selector, or any decision that the method makes during
propagation; it exists to score a trajectory after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import expm_multiply

from src.quantum.siam_model import (
    DEFAULT_HOPPING,
    QUENCHED_HYBRIDIZATION,
    SIAMLayout,
    build_initial_state,
    build_siam_hamiltonian,
    embed_sector_vector,
    project_to_sector,
    sector_vector,
)
from src.quantum.siam_observables import (
    SIAM_OBSERVABLES_V2,
    screening_observable_operators,
)
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix

SIAM_REFERENCE_V1 = "siam_exact_sector_reference_v1"
SIAM_REFERENCE_V2 = "siam_exact_sector_reference_v2"


@dataclass(frozen=True)
class SIAMReference:
    """Exact post-quench trajectory in the target sector."""

    n_bath: int
    u: float
    v: float
    hopping: float
    times: tuple[float, ...]
    observables: dict[str, tuple[float, ...]]
    schema: str = SIAM_REFERENCE_V2
    observable_schema: str = SIAM_OBSERVABLES_V2

    def at(self, name: str) -> tuple[float, ...]:
        return self.observables[name]


def sector_hamiltonian(
    n_bath: int,
    *,
    u: float,
    v: float,
    hopping: float = DEFAULT_HOPPING,
) -> np.ndarray:
    poly = build_siam_hamiltonian(
        int(n_bath), u=float(u), v=float(v), hopping=float(hopping)
    )
    return project_to_sector(
        np.asarray(hamiltonian_matrix(poly), dtype=complex), int(n_bath)
    )


def propagate_sector(
    n_bath: int,
    *,
    u: float,
    v: float = QUENCHED_HYBRIDIZATION,
    hopping: float = DEFAULT_HOPPING,
    times: Iterable[float],
    method: str = "eigh",
) -> list[np.ndarray]:
    r"""Exact :math:`e^{-iHt}|\psi_0\rangle` at each reporting time.

    ``method='eigh'`` diagonalizes once and evaluates every time from the same
    spectrum, which is both faster and more accurate than stepping when the
    post-quench Hamiltonian is constant.  ``method='krylov'`` uses
    ``expm_multiply`` per time and exists as an independent check.
    """

    reporting_times = tuple(float(t) for t in times)
    h_sector = sector_hamiltonian(
        int(n_bath), u=u, v=v, hopping=float(hopping)
    )
    psi0 = sector_vector(
        build_initial_state(int(n_bath), hopping=float(hopping)), int(n_bath)
    )
    out: list[np.ndarray] = []
    if str(method) == "eigh":
        evals, evecs = eigh(h_sector)
        coeffs = evecs.conj().T @ psi0
        for t in reporting_times:
            phased = np.exp(-1.0j * evals * float(t)) * coeffs
            out.append(evecs @ phased)
        return out
    if str(method) == "krylov":
        for t in reporting_times:
            out.append(expm_multiply(-1.0j * float(t) * h_sector, psi0))
        return out
    raise ValueError(f"unknown propagation method {method!r}")


def exact_reference(
    n_bath: int,
    *,
    u: float,
    v: float = QUENCHED_HYBRIDIZATION,
    hopping: float = DEFAULT_HOPPING,
    times: Iterable[float],
    method: str = "eigh",
) -> SIAMReference:
    """Exact screening observables at every reporting time."""

    reporting_times = tuple(float(t) for t in times)
    layout = SIAMLayout(int(n_bath))
    states = propagate_sector(
        int(n_bath),
        u=u,
        v=v,
        hopping=float(hopping),
        times=reporting_times,
        method=method,
    )
    operators = {
        name: project_to_sector(
            np.asarray(hamiltonian_matrix(op), dtype=complex), int(n_bath)
        )
        for name, op in screening_observable_operators(layout, v=float(v)).items()
    }
    series: dict[str, list[float]] = {name: [] for name in operators}
    for state in states:
        for name, matrix in operators.items():
            series[name].append(float(np.real(np.vdot(state, matrix @ state))))
    return SIAMReference(
        n_bath=int(n_bath),
        u=float(u),
        v=float(v),
        hopping=float(hopping),
        times=reporting_times,
        observables={k: tuple(v_) for k, v_ in series.items()},
    )


__all__ = [
    "SIAMReference",
    "SIAM_REFERENCE_V1",
    "SIAM_REFERENCE_V2",
    "exact_reference",
    "propagate_sector",
    "sector_hamiltonian",
]
