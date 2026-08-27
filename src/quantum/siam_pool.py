r"""Logical generator pool for finite-box SIAM impurity screening.

Six block families, every one Hermitian and preserving both :math:`N_\uparrow`
and :math:`N_\downarrow` separately:

======  ==================================================================
``K``   Hamiltonian-edge hopping, :math:`c^\dagger_a c_b + \mathrm{h.c.}`
``J``   current quadrature, :math:`i(c^\dagger_a c_b - \mathrm{h.c.})`
``Qd``  centered impurity interaction, :math:`(n_{d\uparrow}-\tfrac12)(n_{d\downarrow}-\tfrac12)`
``A``   impurity density-assisted current on the hybridization edge
``X``   impurity--shell spin exchange, :math:`\mathbf S_d\cdot\mathbf S_j`
``P``   pair transfer, :math:`d^\dagger_\uparrow d^\dagger_\downarrow c_{1\downarrow}c_{1\uparrow} + \mathrm{h.c.}`
======  ==================================================================

``K`` and ``J`` are the Hermitian and anti-Hermitian quadratures of the same
edge; both are kept because a variational generator :math:`e^{-i\theta A}`
reaches different states from each, and the quench breaks the symmetry that
would otherwise make one redundant.

Unrestricted generalized excitations are deliberately **not** included: they
would leave the :math:`(N_\uparrow, N_\downarrow)` sector, which both the exact
reference and the screening observables assume.  Every block below is checked
against that in :mod:`test.test_siam_pool`.

Labels are deterministic and carry their physical content, e.g. ``K_bath_up_1_2``
or ``X_d_3``, so a pool entry can be traced back to the operator it encodes
without consulting an index table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from src.quantum.pauli_polynomial_class import (
    PauliPolynomial,
    fermion_minus_operator,
    fermion_plus_operator,
)
from src.quantum.siam_model import DOWN, SIAMLayout, UP, number_operator
from src.quantum.siam_observables import spin_correlation_operator

SIAM_POOL_V1 = "siam_screening_logical_pool_v1"


@dataclass(frozen=True)
class PoolBlock:
    """One labeled Hermitian generator."""

    label: str
    family: str
    operator: PauliPolynomial


def _plus(layout: SIAMLayout, mode: int) -> PauliPolynomial:
    return fermion_plus_operator("JW", layout.n_qubits, int(mode))


def _minus(layout: SIAMLayout, mode: int) -> PauliPolynomial:
    return fermion_minus_operator("JW", layout.n_qubits, int(mode))


def hopping_quadrature(
    layout: SIAMLayout, mode_a: int, mode_b: int
) -> PauliPolynomial:
    r""":math:`c^\dagger_a c_b + c^\dagger_b c_a`."""

    return _plus(layout, mode_a) * _minus(layout, mode_b) + _plus(
        layout, mode_b
    ) * _minus(layout, mode_a)


def current_quadrature(
    layout: SIAMLayout, mode_a: int, mode_b: int
) -> PauliPolynomial:
    r""":math:`i(c^\dagger_a c_b - c^\dagger_b c_a)`."""

    return 1.0j * (
        _plus(layout, mode_a) * _minus(layout, mode_b)
        - _plus(layout, mode_b) * _minus(layout, mode_a)
    )


def _edges(layout: SIAMLayout) -> Iterator[tuple[str, int, int, str]]:
    """``(name, mode_a, mode_b, spin)`` over hybridization and bath edges."""

    for spin in (UP, DOWN):
        yield ("hyb", layout.impurity(spin), layout.bath(1, spin), spin)
        for site in range(1, layout.n_bath):
            yield (
                "bath",
                layout.bath(site, spin),
                layout.bath(site + 1, spin),
                spin,
            )


def _edge_label(kind: str, name: str, spin: str, a: int, b: int) -> str:
    return f"{kind}_{name}_{spin}_{a}_{b}"


def kinetic_blocks(layout: SIAMLayout) -> list[PoolBlock]:
    return [
        PoolBlock(_edge_label("K", name, spin, a, b), "K", hopping_quadrature(layout, a, b))
        for name, a, b, spin in _edges(layout)
    ]


def current_blocks(layout: SIAMLayout) -> list[PoolBlock]:
    return [
        PoolBlock(_edge_label("J", name, spin, a, b), "J", current_quadrature(layout, a, b))
        for name, a, b, spin in _edges(layout)
    ]


def impurity_interaction_block(layout: SIAMLayout) -> PoolBlock:
    from src.quantum.siam_model import identity_polynomial

    identity = identity_polynomial(layout)
    centered_up = number_operator(layout, layout.impurity(UP)) - 0.5 * identity
    centered_dn = number_operator(layout, layout.impurity(DOWN)) - 0.5 * identity
    return PoolBlock("Qd_centered", "Qd", centered_up * centered_dn)


def density_assisted_blocks(layout: SIAMLayout) -> list[PoolBlock]:
    r"""``n_{d,-sigma} * i(d^dagger_sigma c_{1 sigma} - h.c.)``.

    The impurity current conditioned on the opposite-spin impurity occupation:
    the channel by which double occupancy gates charge transfer, which a bare
    hopping generator cannot express.
    """

    blocks: list[PoolBlock] = []
    for spin, other in ((UP, DOWN), (DOWN, UP)):
        gate = number_operator(layout, layout.impurity(other))
        current = current_quadrature(
            layout, layout.impurity(spin), layout.bath(1, spin)
        )
        # gate and current commute: the gate touches only the opposite spin
        # block, so the product is Hermitian without symmetrization.
        blocks.append(PoolBlock(f"A_hyb_{spin}", "A", gate * current))
    return blocks


def spin_exchange_blocks(layout: SIAMLayout) -> list[PoolBlock]:
    return [
        PoolBlock(f"X_d_{site}", "X", spin_correlation_operator(layout, site))
        for site in range(1, layout.n_bath + 1)
    ]


def pair_transfer_blocks(layout: SIAMLayout) -> list[PoolBlock]:
    r""":math:`d^\dagger_\uparrow d^\dagger_\downarrow c_{1\downarrow}c_{1\uparrow} + \mathrm{h.c.}`"""

    d_up, d_dn = layout.impurity(UP), layout.impurity(DOWN)
    b_up, b_dn = layout.bath(1, UP), layout.bath(1, DOWN)
    forward = (
        _plus(layout, d_up)
        * _plus(layout, d_dn)
        * _minus(layout, b_dn)
        * _minus(layout, b_up)
    )
    backward = (
        _plus(layout, b_up)
        * _plus(layout, b_dn)
        * _minus(layout, d_dn)
        * _minus(layout, d_up)
    )
    return [PoolBlock("P_d_1", "P", forward + backward)]


def build_siam_pool(n_bath: int) -> list[PoolBlock]:
    """The full logical pool in a deterministic order."""

    layout = SIAMLayout(int(n_bath))
    blocks: list[PoolBlock] = []
    blocks.extend(kinetic_blocks(layout))
    blocks.extend(current_blocks(layout))
    blocks.append(impurity_interaction_block(layout))
    blocks.extend(density_assisted_blocks(layout))
    blocks.extend(spin_exchange_blocks(layout))
    blocks.extend(pair_transfer_blocks(layout))
    labels = [b.label for b in blocks]
    if len(set(labels)) != len(labels):  # pragma: no cover - guarded by tests
        raise ValueError("SIAM pool labels are not unique.")
    return blocks


__all__ = [
    "PoolBlock",
    "SIAM_POOL_V1",
    "build_siam_pool",
    "current_blocks",
    "current_quadrature",
    "density_assisted_blocks",
    "hopping_quadrature",
    "impurity_interaction_block",
    "kinetic_blocks",
    "pair_transfer_blocks",
    "spin_exchange_blocks",
]
