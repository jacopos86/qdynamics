r"""Observables for finite-box SIAM impurity screening.

Every quantity here is built from the same Jordan--Wigner layout as
:mod:`src.quantum.siam_model`, so an operator and the Hamiltonian always act on
one register with one mode ordering.

The screening observables are:

* impurity magnetization :math:`m_d = \langle n_{d\uparrow}-n_{d\downarrow}\rangle`;
* impurity double occupancy :math:`D_d = \langle n_{d\uparrow}n_{d\downarrow}\rangle`;
* shell correlations :math:`C_j = \langle \mathbf S_d\cdot\mathbf S_j\rangle`
  for every bath site :math:`j`;
* integrated screening :math:`\Sigma(r,t) = \sum_{j\le r} C_j(t)`;
* the impurity current :math:`I_d`, optional.

:math:`\mathbf S_d\cdot\mathbf S_j` is the **full** scalar product, including
the transverse exchange:

.. math::

    \mathbf S_d\cdot\mathbf S_j
      = S_d^z S_j^z + \tfrac12\bigl(S_d^+S_j^- + S_d^-S_j^+\bigr).

Substituting :math:`3 S_d^z S_j^z` is wrong here and is not offered.  That
substitution is exact only for an isotropic *ensemble average* over a spin
singlet; the quench state is not isotropic, and the transverse channel is
precisely where screening builds up, so dropping it would remove the signal
the application exists to measure.
"""

from __future__ import annotations

import numpy as np

from src.quantum.pauli_polynomial_class import (
    PauliPolynomial,
    fermion_minus_operator,
    fermion_plus_operator,
)
from src.quantum.siam_model import (
    DOWN,
    UP,
    SIAMLayout,
    identity_polynomial,
    number_operator,
)

SIAM_OBSERVABLES_V1 = "siam_screening_observables_v1"


def _plus(layout: SIAMLayout, mode: int) -> PauliPolynomial:
    return fermion_plus_operator("JW", layout.n_qubits, int(mode))


def _minus(layout: SIAMLayout, mode: int) -> PauliPolynomial:
    return fermion_minus_operator("JW", layout.n_qubits, int(mode))


def _modes(layout: SIAMLayout, orbital: str | int) -> tuple[int, int]:
    """``(up_mode, down_mode)`` for ``'d'`` or a 1-based bath site."""

    if orbital == "d":
        return layout.impurity(UP), layout.impurity(DOWN)
    return layout.bath(int(orbital), UP), layout.bath(int(orbital), DOWN)


def spin_z_operator(layout: SIAMLayout, orbital: str | int) -> PauliPolynomial:
    r""":math:`S^z = \tfrac12(n_\uparrow - n_\downarrow)`."""

    up, dn = _modes(layout, orbital)
    return 0.5 * (number_operator(layout, up) - number_operator(layout, dn))


def spin_raise_operator(layout: SIAMLayout, orbital: str | int) -> PauliPolynomial:
    r""":math:`S^+ = c^\dagger_\uparrow c_\downarrow`."""

    up, dn = _modes(layout, orbital)
    return _plus(layout, up) * _minus(layout, dn)


def spin_lower_operator(layout: SIAMLayout, orbital: str | int) -> PauliPolynomial:
    r""":math:`S^- = c^\dagger_\downarrow c_\uparrow`."""

    up, dn = _modes(layout, orbital)
    return _plus(layout, dn) * _minus(layout, up)


def spin_correlation_operator(
    layout: SIAMLayout, site: int
) -> PauliPolynomial:
    r""":math:`\mathbf S_d\cdot\mathbf S_j`, transverse channel included."""

    longitudinal = spin_z_operator(layout, "d") * spin_z_operator(layout, site)
    transverse = 0.5 * (
        spin_raise_operator(layout, "d") * spin_lower_operator(layout, site)
        + spin_lower_operator(layout, "d") * spin_raise_operator(layout, site)
    )
    return longitudinal + transverse


def impurity_magnetization_operator(layout: SIAMLayout) -> PauliPolynomial:
    r""":math:`n_{d\uparrow}-n_{d\downarrow}`."""

    up, dn = _modes(layout, "d")
    return number_operator(layout, up) - number_operator(layout, dn)


def impurity_double_occupancy_operator(layout: SIAMLayout) -> PauliPolynomial:
    r""":math:`n_{d\uparrow}n_{d\downarrow}`."""

    up, dn = _modes(layout, "d")
    return number_operator(layout, up) * number_operator(layout, dn)


def impurity_current_operator(
    layout: SIAMLayout, *, v: float
) -> PauliPolynomial:
    r"""Impurity current :math:`I_d = i V\sum_\sigma(d^\dagger_\sigma c_{1\sigma} - \mathrm{h.c.})`.

    This is :math:`\mathrm{d}N_d/\mathrm{d}t` from the Heisenberg equation with
    the hybridization term, so it vanishes identically before the quench.
    """

    terms: PauliPolynomial | None = None
    for spin in (UP, DOWN):
        d_mode = layout.impurity(spin)
        b_mode = layout.bath(1, spin)
        block = (1.0j * float(v)) * (
            _plus(layout, d_mode) * _minus(layout, b_mode)
            - _plus(layout, b_mode) * _minus(layout, d_mode)
        )
        terms = block if terms is None else terms + block
    assert terms is not None
    return terms


def screening_observable_operators(
    layout: SIAMLayout, *, v: float | None = None
) -> dict[str, PauliPolynomial]:
    """Every gated screening observable, keyed by report name."""

    ops: dict[str, PauliPolynomial] = {
        "m_d": impurity_magnetization_operator(layout),
        "D_d": impurity_double_occupancy_operator(layout),
    }
    for site in range(1, layout.n_bath + 1):
        ops[f"C_{site}"] = spin_correlation_operator(layout, site)
    if v is not None:
        ops["I_d"] = impurity_current_operator(layout, v=float(v))
    return ops


def integrated_screening(correlations: dict[str, float], radius: int) -> float:
    r""":math:`\Sigma(r) = \sum_{j\le r} C_j`."""

    total = 0.0
    for site in range(1, int(radius) + 1):
        key = f"C_{site}"
        if key not in correlations:
            raise KeyError(f"missing shell correlation {key}")
        total += float(correlations[key])
    return total


__all__ = [
    "SIAM_OBSERVABLES_V1",
    "identity_polynomial",
    "impurity_current_operator",
    "impurity_double_occupancy_operator",
    "impurity_magnetization_operator",
    "integrated_screening",
    "screening_observable_operators",
    "spin_correlation_operator",
    "spin_lower_operator",
    "spin_raise_operator",
    "spin_z_operator",
]
