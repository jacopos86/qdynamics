"""Additional lattice Hamiltonian builders for registered static-ADAPT problem families."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.quantum.hartree_fock_reference_state import bitstring_qn1_to_q0
from src.quantum.hubbard_latex_python_pairs import (
    SPIN_DN,
    SPIN_UP,
    bravais_nearest_neighbor_edges,
    build_hubbard_kinetic,
    build_hubbard_onsite,
    build_hubbard_potential,
    jw_number_operator,
    mode_index,
)
from src.quantum.pauli_polynomial_class import (
    PauliPolynomial,
    fermion_minus_operator,
    fermion_plus_operator,
)
from src.quantum.qubitization_module import PauliTerm


def _boundary_to_pbc(boundary: str) -> bool:
    return str(boundary).strip().lower() == "periodic"


def _chain_edges(*, num_sites: int, hop_distance: int, boundary: str) -> tuple[tuple[int, int], ...]:
    if int(num_sites) <= 0:
        raise ValueError("num_sites must be positive")
    if int(hop_distance) <= 0:
        raise ValueError("hop_distance must be positive")
    periodic = _boundary_to_pbc(boundary)
    edges: set[tuple[int, int]] = set()
    for site in range(int(num_sites)):
        neighbor = int(site) + int(hop_distance)
        if neighbor < int(num_sites):
            a, b = int(site), int(neighbor)
        elif periodic:
            wrapped = int(neighbor % int(num_sites))
            if wrapped == int(site):
                continue
            a, b = sorted((int(site), int(wrapped)))
        else:
            continue
        if a != b:
            edges.add((int(a), int(b)))
    return tuple(sorted(edges))


def _chain_triples(*, num_sites: int, boundary: str) -> tuple[tuple[int, int, int], ...]:
    if int(num_sites) < 3:
        return tuple()
    periodic = _boundary_to_pbc(boundary)
    triples: set[tuple[int, int, int]] = set()
    for site in range(int(num_sites)):
        j = int(site) + 1
        k = int(site) + 2
        if k < int(num_sites):
            triples.add((int(site), int(j), int(k)))
            continue
        if not periodic:
            continue
        triples.add((int(site), int(j % int(num_sites)), int(k % int(num_sites))))
    return tuple(sorted(triples))


def _staggered_site_potential(*, num_sites: int, amplitude: float) -> tuple[float, ...]:
    return tuple(float(amplitude) if (site % 2 == 0) else -float(amplitude) for site in range(int(num_sites)))


def _identity_polynomial(*, nq: int) -> PauliPolynomial:
    return PauliPolynomial("JW", [PauliTerm(int(nq), ps="e" * int(nq), pc=1.0)])


MATH_SPINFUL_NN_DENSITY = r"H_V = V\sum_{\langle i,j\rangle}(n_{i\uparrow}+n_{i\downarrow})(n_{j\uparrow}+n_{j\downarrow})"


def build_spinful_nearest_neighbor_density_interaction(
    *,
    num_sites: int,
    v_nn: float,
    ordering: str,
    boundary: str,
) -> PauliPolynomial:
    """Build the spinful nearest-neighbor density-density interaction block."""
    nq = 2 * int(num_sites)
    edges = bravais_nearest_neighbor_edges(int(num_sites), pbc=_boundary_to_pbc(boundary))
    h_v = PauliPolynomial("JW")
    if abs(float(v_nn)) <= 1e-15:
        return h_v
    n_cache: dict[int, PauliPolynomial] = {}

    def n_op(mode: int) -> PauliPolynomial:
        if int(mode) not in n_cache:
            n_cache[int(mode)] = jw_number_operator("JW", nq, int(mode))
        return n_cache[int(mode)]

    for site_i, site_j in edges:
        n_i = (
            n_op(mode_index(int(site_i), SPIN_UP, indexing=str(ordering), n_sites=int(num_sites)))
            + n_op(mode_index(int(site_i), SPIN_DN, indexing=str(ordering), n_sites=int(num_sites)))
        )
        n_j = (
            n_op(mode_index(int(site_j), SPIN_UP, indexing=str(ordering), n_sites=int(num_sites)))
            + n_op(mode_index(int(site_j), SPIN_DN, indexing=str(ordering), n_sites=int(num_sites)))
        )
        h_v += float(v_nn) * (n_i * n_j)
    return h_v


MATH_IONIC_HUBBARD = r"H_{\mathrm{ionic}} = -t\sum_{\langle i,j\rangle,\sigma}(c^\dagger_{i\sigma}c_{j\sigma}+\mathrm{h.c.}) + U\sum_i n_{i\uparrow}n_{i\downarrow} - \sum_{i,\sigma} (-1)^i\,\Delta\, n_{i\sigma}"


MATH_SPINFUL_EDGE_HOP = r"T_{ij,\sigma} = -t(c^\dagger_{i\sigma}c_{j\sigma}+c^\dagger_{j\sigma}c_{i\sigma})"


def build_spinful_edge_hopping_primitive(
    *,
    num_sites: int,
    site_i: int,
    site_j: int,
    spin: str,
    amplitude: float,
    ordering: str,
) -> PauliPolynomial:
    """Build one spin-resolved hopping primitive for a single edge."""
    nq = 2 * int(num_sites)
    spin_key = str(spin).strip().lower()
    if spin_key in {"up", "alpha"}:
        spin_value = SPIN_UP
    elif spin_key in {"dn", "down", "beta"}:
        spin_value = SPIN_DN
    else:
        raise ValueError(f"Unsupported spin label '{spin}'.")
    p_mode = mode_index(int(site_i), spin_value, indexing=str(ordering), n_sites=int(num_sites))
    q_mode = mode_index(int(site_j), spin_value, indexing=str(ordering), n_sites=int(num_sites))
    cd = lambda mode: fermion_plus_operator("JW", nq, int(mode))
    cm = lambda mode: fermion_minus_operator("JW", nq, int(mode))
    return (-float(amplitude)) * ((cd(p_mode) * cm(q_mode)) + (cd(q_mode) * cm(p_mode)))


MATH_SPINFUL_EDGE_CURRENT = r"J_{ij,\sigma} = i\,t(c^\dagger_{i\sigma}c_{j\sigma}-c^\dagger_{j\sigma}c_{i\sigma})"


def build_spinful_edge_current_primitive(
    *,
    num_sites: int,
    site_i: int,
    site_j: int,
    spin: str,
    amplitude: float,
    ordering: str,
) -> PauliPolynomial:
    """Build the current-like quadrature partner for one spin-resolved edge hop."""
    nq = 2 * int(num_sites)
    spin_key = str(spin).strip().lower()
    if spin_key in {"up", "alpha"}:
        spin_value = SPIN_UP
    elif spin_key in {"dn", "down", "beta"}:
        spin_value = SPIN_DN
    else:
        raise ValueError(f"Unsupported spin label '{spin}'.")
    p_mode = mode_index(int(site_i), spin_value, indexing=str(ordering), n_sites=int(num_sites))
    q_mode = mode_index(int(site_j), spin_value, indexing=str(ordering), n_sites=int(num_sites))
    cd = lambda mode: fermion_plus_operator("JW", nq, int(mode))
    cm = lambda mode: fermion_minus_operator("JW", nq, int(mode))
    return (1.0j * float(amplitude)) * ((cd(p_mode) * cm(q_mode)) - (cd(q_mode) * cm(p_mode)))


MATH_SPINFUL_SITE_DENSITY = r"N_i = n_{i\uparrow}+n_{i\downarrow}"


def build_spinful_site_density_primitive(
    *,
    num_sites: int,
    site: int,
    amplitude: float,
    ordering: str,
) -> PauliPolynomial:
    """Build one site-density primitive summed over spin."""
    nq = 2 * int(num_sites)
    site_i = int(site)
    if abs(float(amplitude)) <= 1e-15:
        return PauliPolynomial("JW")
    up_mode = mode_index(site_i, SPIN_UP, indexing=str(ordering), n_sites=int(num_sites))
    dn_mode = mode_index(site_i, SPIN_DN, indexing=str(ordering), n_sites=int(num_sites))
    return float(amplitude) * (
        jw_number_operator("JW", nq, int(up_mode)) + jw_number_operator("JW", nq, int(dn_mode))
    )


MATH_SPINFUL_ONSITE_SITE = r"U_i = U\,n_{i\uparrow}n_{i\downarrow}"


def build_spinful_onsite_primitive(
    *,
    num_sites: int,
    site: int,
    u: float,
    ordering: str,
) -> PauliPolynomial:
    """Build one onsite interaction primitive for a single site."""
    nq = 2 * int(num_sites)
    site_i = int(site)
    up_mode = mode_index(site_i, SPIN_UP, indexing=str(ordering), n_sites=int(num_sites))
    dn_mode = mode_index(site_i, SPIN_DN, indexing=str(ordering), n_sites=int(num_sites))
    return float(u) * (
        jw_number_operator("JW", nq, int(up_mode)) * jw_number_operator("JW", nq, int(dn_mode))
    )


MATH_SPINFUL_EDGE_DENSITY = r"V_{ij} = V\,N_i N_j"


def build_spinful_edge_density_primitive(
    *,
    num_sites: int,
    site_i: int,
    site_j: int,
    amplitude: float,
    ordering: str,
    ) -> PauliPolynomial:
    """Build one nearest-neighbor density-density primitive."""
    if abs(float(amplitude)) <= 1e-15:
        return PauliPolynomial("JW")
    density_i = build_spinful_site_density_primitive(
        num_sites=int(num_sites),
        site=int(site_i),
        amplitude=1.0,
        ordering=str(ordering),
    )
    density_j = build_spinful_site_density_primitive(
        num_sites=int(num_sites),
        site=int(site_j),
        amplitude=1.0,
        ordering=str(ordering),
    )
    return float(amplitude) * (density_i * density_j)


MATH_SPINFUL_BOND_DENSITY = r"B_{ij} = N_i + N_j"


def build_spinful_bond_density_primitive(
    *,
    num_sites: int,
    site_i: int,
    site_j: int,
    amplitude: float,
    ordering: str,
) -> PauliPolynomial:
    """Build the spin-summed bond-density primitive B_ij = N_i + N_j."""
    if abs(float(amplitude)) <= 1e-15:
        return PauliPolynomial("JW")
    return float(amplitude) * (
        build_spinful_site_density_primitive(
            num_sites=int(num_sites),
            site=int(site_i),
            amplitude=1.0,
            ordering=str(ordering),
        )
        + build_spinful_site_density_primitive(
            num_sites=int(num_sites),
            site=int(site_j),
            amplitude=1.0,
            ordering=str(ordering),
        )
    )


MATH_SPINFUL_BOND_CHARGE_HOP = r"\widetilde{T}_{ij,\sigma}^{(B)} = (N_i+N_j)\,T_{ij,\sigma}"


def build_spinful_bond_charge_hopping_primitive(
    *,
    num_sites: int,
    site_i: int,
    site_j: int,
    spin: str,
    amplitude: float,
    ordering: str,
) -> PauliPolynomial:
    """Build a bond-charge-assisted hopping primitive (N_i + N_j) T_{ij,σ}."""
    bond_density = build_spinful_bond_density_primitive(
        num_sites=int(num_sites),
        site_i=int(site_i),
        site_j=int(site_j),
        amplitude=1.0,
        ordering=str(ordering),
    )
    hop = build_spinful_edge_hopping_primitive(
        num_sites=int(num_sites),
        site_i=int(site_i),
        site_j=int(site_j),
        spin=str(spin),
        amplitude=float(amplitude),
        ordering=str(ordering),
    )
    return bond_density * hop


MATH_SPINFUL_BOND_CHARGE_CURRENT = r"\widetilde{J}_{ij,\sigma}^{(B)} = (N_i+N_j)\,J_{ij,\sigma}"


def build_spinful_bond_charge_current_primitive(
    *,
    num_sites: int,
    site_i: int,
    site_j: int,
    spin: str,
    amplitude: float,
    ordering: str,
) -> PauliPolynomial:
    """Build a bond-charge-assisted current primitive (N_i + N_j) J_{ij,σ}."""
    bond_density = build_spinful_bond_density_primitive(
        num_sites=int(num_sites),
        site_i=int(site_i),
        site_j=int(site_j),
        amplitude=1.0,
        ordering=str(ordering),
    )
    current = build_spinful_edge_current_primitive(
        num_sites=int(num_sites),
        site_i=int(site_i),
        site_j=int(site_j),
        spin=str(spin),
        amplitude=float(amplitude),
        ordering=str(ordering),
    )
    return bond_density * current


MATH_SPINFUL_OPPOSITE_SPIN_BOND_DENSITY = r"D_{ij,\bar{\sigma}} = n_{i\bar{\sigma}} + n_{j\bar{\sigma}}"


def build_spinful_opposite_spin_bond_density_primitive(
    *,
    num_sites: int,
    site_i: int,
    site_j: int,
    spin: str,
    amplitude: float,
    ordering: str,
) -> PauliPolynomial:
    """Build the opposite-spin bond density n_{i,opp} + n_{j,opp}."""
    spin_key = str(spin).strip().lower()
    if spin_key in {"up", "alpha"}:
        opposite_spin = SPIN_DN
    elif spin_key in {"dn", "down", "beta"}:
        opposite_spin = SPIN_UP
    else:
        raise ValueError(f"Unsupported spin label '{spin}'.")
    nq = 2 * int(num_sites)
    if abs(float(amplitude)) <= 1e-15:
        return PauliPolynomial("JW")
    mode_i = mode_index(int(site_i), opposite_spin, indexing=str(ordering), n_sites=int(num_sites))
    mode_j = mode_index(int(site_j), opposite_spin, indexing=str(ordering), n_sites=int(num_sites))
    return float(amplitude) * (
        jw_number_operator("JW", nq, int(mode_i)) + jw_number_operator("JW", nq, int(mode_j))
    )


MATH_SPINFUL_OPPOSITE_SPIN_ASSIST_HOP = r"\widetilde{T}_{ij,\sigma}^{(\bar{\sigma})} = (n_{i\bar{\sigma}} + n_{j\bar{\sigma}})\,T_{ij,\sigma}"


def build_spinful_opposite_spin_assisted_hopping_primitive(
    *,
    num_sites: int,
    site_i: int,
    site_j: int,
    spin: str,
    amplitude: float,
    ordering: str,
) -> PauliPolynomial:
    """Build an opposite-spin-assisted hopping primitive."""
    opposite_spin_density = build_spinful_opposite_spin_bond_density_primitive(
        num_sites=int(num_sites),
        site_i=int(site_i),
        site_j=int(site_j),
        spin=str(spin),
        amplitude=1.0,
        ordering=str(ordering),
    )
    hop = build_spinful_edge_hopping_primitive(
        num_sites=int(num_sites),
        site_i=int(site_i),
        site_j=int(site_j),
        spin=str(spin),
        amplitude=float(amplitude),
        ordering=str(ordering),
    )
    return opposite_spin_density * hop


MATH_SPINFUL_OPPOSITE_SPIN_ASSIST_CURRENT = r"\widetilde{J}_{ij,\sigma}^{(\bar{\sigma})} = (n_{i\bar{\sigma}} + n_{j\bar{\sigma}})\,J_{ij,\sigma}"


def build_spinful_opposite_spin_assisted_current_primitive(
    *,
    num_sites: int,
    site_i: int,
    site_j: int,
    spin: str,
    amplitude: float,
    ordering: str,
) -> PauliPolynomial:
    """Build an opposite-spin-assisted current primitive."""
    opposite_spin_density = build_spinful_opposite_spin_bond_density_primitive(
        num_sites=int(num_sites),
        site_i=int(site_i),
        site_j=int(site_j),
        spin=str(spin),
        amplitude=1.0,
        ordering=str(ordering),
    )
    current = build_spinful_edge_current_primitive(
        num_sites=int(num_sites),
        site_i=int(site_i),
        site_j=int(site_j),
        spin=str(spin),
        amplitude=float(amplitude),
        ordering=str(ordering),
    )
    return opposite_spin_density * current


MATH_SPINFUL_EDGE_PAIR_HOP = r"W_{ij} = c^\dagger_{i\uparrow} c^\dagger_{i\downarrow} c_{j\downarrow} c_{j\uparrow} + \mathrm{h.c.}"


def build_spinful_edge_pair_hop_primitive(
    *,
    num_sites: int,
    site_i: int,
    site_j: int,
    amplitude: float,
    ordering: str,
) -> PauliPolynomial:
    """Build a pair-hopping primitive on one edge."""
    nq = 2 * int(num_sites)
    iu = mode_index(int(site_i), SPIN_UP, indexing=str(ordering), n_sites=int(num_sites))
    idn = mode_index(int(site_i), SPIN_DN, indexing=str(ordering), n_sites=int(num_sites))
    ju = mode_index(int(site_j), SPIN_UP, indexing=str(ordering), n_sites=int(num_sites))
    jdn = mode_index(int(site_j), SPIN_DN, indexing=str(ordering), n_sites=int(num_sites))
    cd = lambda mode: fermion_plus_operator("JW", nq, int(mode))
    cm = lambda mode: fermion_minus_operator("JW", nq, int(mode))
    forward = cd(iu) * cd(idn) * cm(jdn) * cm(ju)
    backward = cd(ju) * cd(jdn) * cm(idn) * cm(iu)
    return float(amplitude) * (forward + backward)


MATH_SPINFUL_EDGE_PAIR_CURRENT = r"J^{(W)}_{ij} = i(c^\dagger_{i\uparrow} c^\dagger_{i\downarrow} c_{j\downarrow} c_{j\uparrow} - \mathrm{h.c.})"


def build_spinful_edge_pair_current_primitive(
    *,
    num_sites: int,
    site_i: int,
    site_j: int,
    amplitude: float,
    ordering: str,
) -> PauliPolynomial:
    """Build the quadrature partner for one pair-hopping primitive."""
    nq = 2 * int(num_sites)
    iu = mode_index(int(site_i), SPIN_UP, indexing=str(ordering), n_sites=int(num_sites))
    idn = mode_index(int(site_i), SPIN_DN, indexing=str(ordering), n_sites=int(num_sites))
    ju = mode_index(int(site_j), SPIN_UP, indexing=str(ordering), n_sites=int(num_sites))
    jdn = mode_index(int(site_j), SPIN_DN, indexing=str(ordering), n_sites=int(num_sites))
    cd = lambda mode: fermion_plus_operator("JW", nq, int(mode))
    cm = lambda mode: fermion_minus_operator("JW", nq, int(mode))
    forward = cd(iu) * cd(idn) * cm(jdn) * cm(ju)
    backward = cd(ju) * cd(jdn) * cm(idn) * cm(iu)
    return (1.0j * float(amplitude)) * (forward - backward)


MATH_SPINFUL_EDGE_EXCHANGE = r"X_{ij} = c^\dagger_{i\uparrow} c_{j\uparrow} c^\dagger_{j\downarrow} c_{i\downarrow} + \mathrm{h.c.}"


def build_spinful_edge_exchange_primitive(
    *,
    num_sites: int,
    site_i: int,
    site_j: int,
    amplitude: float,
    ordering: str,
) -> PauliPolynomial:
    """Build a spin-exchange primitive on one edge."""
    nq = 2 * int(num_sites)
    iu = mode_index(int(site_i), SPIN_UP, indexing=str(ordering), n_sites=int(num_sites))
    idn = mode_index(int(site_i), SPIN_DN, indexing=str(ordering), n_sites=int(num_sites))
    ju = mode_index(int(site_j), SPIN_UP, indexing=str(ordering), n_sites=int(num_sites))
    jdn = mode_index(int(site_j), SPIN_DN, indexing=str(ordering), n_sites=int(num_sites))
    cd = lambda mode: fermion_plus_operator("JW", nq, int(mode))
    cm = lambda mode: fermion_minus_operator("JW", nq, int(mode))
    forward = cd(iu) * cm(ju) * cd(jdn) * cm(idn)
    backward = cd(idn) * cm(jdn) * cd(ju) * cm(iu)
    return float(amplitude) * (forward + backward)


MATH_SPINFUL_EDGE_EXCHANGE_CURRENT = r"J^{(X)}_{ij} = i(c^\dagger_{i\uparrow} c_{j\uparrow} c^\dagger_{j\downarrow} c_{i\downarrow} - \mathrm{h.c.})"


def build_spinful_edge_exchange_current_primitive(
    *,
    num_sites: int,
    site_i: int,
    site_j: int,
    amplitude: float,
    ordering: str,
) -> PauliPolynomial:
    """Build the quadrature partner for one spin-exchange primitive."""
    nq = 2 * int(num_sites)
    iu = mode_index(int(site_i), SPIN_UP, indexing=str(ordering), n_sites=int(num_sites))
    idn = mode_index(int(site_i), SPIN_DN, indexing=str(ordering), n_sites=int(num_sites))
    ju = mode_index(int(site_j), SPIN_UP, indexing=str(ordering), n_sites=int(num_sites))
    jdn = mode_index(int(site_j), SPIN_DN, indexing=str(ordering), n_sites=int(num_sites))
    cd = lambda mode: fermion_plus_operator("JW", nq, int(mode))
    cm = lambda mode: fermion_minus_operator("JW", nq, int(mode))
    forward = cd(iu) * cm(ju) * cd(jdn) * cm(idn)
    backward = cd(idn) * cm(jdn) * cd(ju) * cm(iu)
    return (1.0j * float(amplitude)) * (forward - backward)


MATH_SPINFUL_CENTERED_OPPOSITE_SPIN_SITE_DENSITY = r"\widetilde{n}_{j\bar{\sigma}} = n_{j\bar{\sigma}} - \tfrac12"


def build_spinful_centered_opposite_spin_site_density_primitive(
    *,
    num_sites: int,
    site: int,
    spin: str,
    ordering: str,
) -> PauliPolynomial:
    """Build the centered opposite-spin density n_{j,opp} - 1/2 on one site."""
    nq = 2 * int(num_sites)
    spin_key = str(spin).strip().lower()
    if spin_key in {"up", "alpha"}:
        opposite_spin = SPIN_DN
    elif spin_key in {"dn", "down", "beta"}:
        opposite_spin = SPIN_UP
    else:
        raise ValueError(f"Unsupported spin label '{spin}'.")
    mode = mode_index(int(site), opposite_spin, indexing=str(ordering), n_sites=int(num_sites))
    return jw_number_operator("JW", nq, int(mode)) - (0.5 * _identity_polynomial(nq=nq))


MATH_SPINFUL_THREE_SITE_BRIDGE_HOP = r"\widetilde{T}_{ik|j,\sigma}^{(3)} = (n_{j\bar{\sigma}}-\tfrac12)\,T_{ik,\sigma}"


def build_spinful_three_site_bridge_hopping_primitive(
    *,
    num_sites: int,
    site_i: int,
    site_j: int,
    site_k: int,
    spin: str,
    amplitude: float,
    ordering: str,
) -> PauliPolynomial:
    """Build a three-site bridge hopping primitive on a contiguous triple."""
    center = build_spinful_centered_opposite_spin_site_density_primitive(
        num_sites=int(num_sites),
        site=int(site_j),
        spin=str(spin),
        ordering=str(ordering),
    )
    hop = build_spinful_edge_hopping_primitive(
        num_sites=int(num_sites),
        site_i=int(site_i),
        site_j=int(site_k),
        spin=str(spin),
        amplitude=float(amplitude),
        ordering=str(ordering),
    )
    return center * hop


MATH_SPINFUL_THREE_SITE_BRIDGE_CURRENT = r"\widetilde{J}_{ik|j,\sigma}^{(3)} = (n_{j\bar{\sigma}}-\tfrac12)\,J_{ik,\sigma}"


def build_spinful_three_site_bridge_current_primitive(
    *,
    num_sites: int,
    site_i: int,
    site_j: int,
    site_k: int,
    spin: str,
    amplitude: float,
    ordering: str,
) -> PauliPolynomial:
    """Build a three-site bridge current primitive on a contiguous triple."""
    center = build_spinful_centered_opposite_spin_site_density_primitive(
        num_sites=int(num_sites),
        site=int(site_j),
        spin=str(spin),
        ordering=str(ordering),
    )
    current = build_spinful_edge_current_primitive(
        num_sites=int(num_sites),
        site_i=int(site_i),
        site_j=int(site_k),
        spin=str(spin),
        amplitude=float(amplitude),
        ordering=str(ordering),
    )
    return center * current


def _build_spinful_termwise_family(
    *,
    num_sites: int,
    ordering: str,
    boundary: str,
    onsite_u: float,
    site_density_weights: Sequence[float] | None,
    nn_density_weight: float = 0.0,
    include_quadratures: bool,
    nn_hop_weight: float,
    nnn_hop_weight: float = 0.0,
    include_bond_charge_terms: bool = False,
    include_opposite_spin_assisted_terms: bool = False,
    include_exchange_terms: bool = False,
    include_three_site_bridge_terms: bool = False,
) -> list[tuple[str, PauliPolynomial]]:
    primitives: list[tuple[str, PauliPolynomial]] = []
    nn_edges = bravais_nearest_neighbor_edges(int(num_sites), pbc=_boundary_to_pbc(boundary))
    nnn_edges = _chain_edges(num_sites=int(num_sites), hop_distance=2, boundary=str(boundary))
    for edge_kind, edge_weight, edges in (
        ("nn", float(nn_hop_weight), tuple(nn_edges)),
        ("nnn", float(nnn_hop_weight), tuple(nnn_edges)),
    ):
        if abs(float(edge_weight)) <= 1e-15:
            continue
        for site_i, site_j in edges:
            for spin_label in ("up", "dn"):
                primitives.append(
                    (
                        f"hop_{edge_kind}_{spin_label}({int(site_i)},{int(site_j)})",
                        build_spinful_edge_hopping_primitive(
                            num_sites=int(num_sites),
                            site_i=int(site_i),
                            site_j=int(site_j),
                            spin=str(spin_label),
                            amplitude=float(edge_weight),
                            ordering=str(ordering),
                        ),
                    )
                )
                if bool(include_quadratures):
                    primitives.append(
                        (
                            f"hop_{edge_kind}_quadrature_{spin_label}({int(site_i)},{int(site_j)})",
                            build_spinful_edge_current_primitive(
                                num_sites=int(num_sites),
                                site_i=int(site_i),
                                site_j=int(site_j),
                                spin=str(spin_label),
                                amplitude=float(edge_weight),
                                ordering=str(ordering),
                            ),
                        )
                    )
                if bool(include_bond_charge_terms):
                    primitives.append(
                        (
                            f"bond_charge_hop_{edge_kind}_{spin_label}({int(site_i)},{int(site_j)})",
                            build_spinful_bond_charge_hopping_primitive(
                                num_sites=int(num_sites),
                                site_i=int(site_i),
                                site_j=int(site_j),
                                spin=str(spin_label),
                                amplitude=float(edge_weight),
                                ordering=str(ordering),
                            ),
                        )
                    )
                    if bool(include_quadratures):
                        primitives.append(
                            (
                                f"bond_charge_current_{edge_kind}_{spin_label}({int(site_i)},{int(site_j)})",
                                build_spinful_bond_charge_current_primitive(
                                    num_sites=int(num_sites),
                                    site_i=int(site_i),
                                    site_j=int(site_j),
                                    spin=str(spin_label),
                                    amplitude=float(edge_weight),
                                    ordering=str(ordering),
                                ),
                            )
                        )
                if bool(include_opposite_spin_assisted_terms):
                    primitives.append(
                        (
                            f"opp_spin_assist_hop_{edge_kind}_{spin_label}({int(site_i)},{int(site_j)})",
                            build_spinful_opposite_spin_assisted_hopping_primitive(
                                num_sites=int(num_sites),
                                site_i=int(site_i),
                                site_j=int(site_j),
                                spin=str(spin_label),
                                amplitude=float(edge_weight),
                                ordering=str(ordering),
                            ),
                        )
                    )
                    if bool(include_quadratures):
                        primitives.append(
                            (
                                f"opp_spin_assist_current_{edge_kind}_{spin_label}({int(site_i)},{int(site_j)})",
                                build_spinful_opposite_spin_assisted_current_primitive(
                                    num_sites=int(num_sites),
                                    site_i=int(site_i),
                                    site_j=int(site_j),
                                    spin=str(spin_label),
                                    amplitude=float(edge_weight),
                                    ordering=str(ordering),
                                ),
                            )
                        )
            if bool(include_exchange_terms):
                primitives.append(
                    (
                        f"pair_hop_{edge_kind}({int(site_i)},{int(site_j)})",
                        build_spinful_edge_pair_hop_primitive(
                            num_sites=int(num_sites),
                            site_i=int(site_i),
                            site_j=int(site_j),
                            amplitude=1.0,
                            ordering=str(ordering),
                        ),
                    )
                )
                primitives.append(
                    (
                        f"exchange_{edge_kind}({int(site_i)},{int(site_j)})",
                        build_spinful_edge_exchange_primitive(
                            num_sites=int(num_sites),
                            site_i=int(site_i),
                            site_j=int(site_j),
                            amplitude=1.0,
                            ordering=str(ordering),
                        ),
                    )
                )
                if bool(include_quadratures):
                    primitives.append(
                        (
                            f"pair_current_{edge_kind}({int(site_i)},{int(site_j)})",
                            build_spinful_edge_pair_current_primitive(
                                num_sites=int(num_sites),
                                site_i=int(site_i),
                                site_j=int(site_j),
                                amplitude=1.0,
                                ordering=str(ordering),
                            ),
                        )
                    )
                    primitives.append(
                        (
                            f"exchange_current_{edge_kind}({int(site_i)},{int(site_j)})",
                            build_spinful_edge_exchange_current_primitive(
                                num_sites=int(num_sites),
                                site_i=int(site_i),
                                site_j=int(site_j),
                                amplitude=1.0,
                                ordering=str(ordering),
                            ),
                        )
                    )
    if bool(include_three_site_bridge_terms):
        bridge_weight = float(nnn_hop_weight if abs(float(nnn_hop_weight)) > 1e-15 else nn_hop_weight)
        for site_i, site_j, site_k in _chain_triples(num_sites=int(num_sites), boundary=str(boundary)):
            for spin_label in ("up", "dn"):
                primitives.append(
                    (
                        f"three_site_bridge_hop_{spin_label}({int(site_i)},{int(site_j)},{int(site_k)})",
                        build_spinful_three_site_bridge_hopping_primitive(
                            num_sites=int(num_sites),
                            site_i=int(site_i),
                            site_j=int(site_j),
                            site_k=int(site_k),
                            spin=str(spin_label),
                            amplitude=float(bridge_weight),
                            ordering=str(ordering),
                        ),
                    )
                )
                if bool(include_quadratures):
                    primitives.append(
                        (
                            f"three_site_bridge_current_{spin_label}({int(site_i)},{int(site_j)},{int(site_k)})",
                            build_spinful_three_site_bridge_current_primitive(
                                num_sites=int(num_sites),
                                site_i=int(site_i),
                                site_j=int(site_j),
                                site_k=int(site_k),
                                spin=str(spin_label),
                                amplitude=float(bridge_weight),
                                ordering=str(ordering),
                            ),
                        )
                    )
    for site in range(int(num_sites)):
        primitives.append(
            (
                f"onsite_site({int(site)})",
                build_spinful_onsite_primitive(
                    num_sites=int(num_sites),
                    site=int(site),
                    u=float(onsite_u),
                    ordering=str(ordering),
                ),
            )
        )
    weights = (
        [0.0] * int(num_sites)
        if site_density_weights is None
        else [float(x) for x in site_density_weights]
    )
    if len(weights) != int(num_sites):
        raise ValueError("site_density_weights must match num_sites.")
    for site, weight in enumerate(weights):
        if abs(float(weight)) <= 1e-15:
            continue
        primitives.append(
            (
                f"site_density({int(site)})",
                build_spinful_site_density_primitive(
                    num_sites=int(num_sites),
                    site=int(site),
                    amplitude=float(weight),
                    ordering=str(ordering),
                ),
            )
        )
    if abs(float(nn_density_weight)) > 1e-15:
        for site_i, site_j in nn_edges:
            primitives.append(
                (
                    f"nn_density({int(site_i)},{int(site_j)})",
                    build_spinful_edge_density_primitive(
                        num_sites=int(num_sites),
                        site_i=int(site_i),
                        site_j=int(site_j),
                        amplitude=float(nn_density_weight),
                        ordering=str(ordering),
                    ),
                )
            )
    return primitives


MATH_IONIC_HUBBARD_HVA = r"\{T_{ij,\sigma},\,U_i,\,(-1)^i\Delta N_i\}"


def build_ionic_hubbard_hva_terms(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    ordering: str,
    boundary: str,
) -> list[tuple[str, PauliPolynomial]]:
    """Build physical Hamiltonian primitives for the ionic Hubbard family."""
    return _build_spinful_termwise_family(
        num_sites=int(num_sites),
        ordering=str(ordering),
        boundary=str(boundary),
        onsite_u=float(u),
        site_density_weights=_staggered_site_potential(num_sites=int(num_sites), amplitude=-float(dv)),
        nn_density_weight=0.0,
        include_quadratures=False,
        nn_hop_weight=float(t),
        nnn_hop_weight=0.0,
    )


MATH_IONIC_HUBBARD_QUADRATURES = r"\{T_{ij,\sigma},\,J_{ij,\sigma},\,U_i,\,(-1)^i\Delta N_i\}"


def build_ionic_hubbard_quadratures(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    ordering: str,
    boundary: str,
) -> list[tuple[str, PauliPolynomial]]:
    """Build ionic Hubbard physical primitives plus hopping quadrature partners."""
    return _build_spinful_termwise_family(
        num_sites=int(num_sites),
        ordering=str(ordering),
        boundary=str(boundary),
        onsite_u=float(u),
        site_density_weights=_staggered_site_potential(num_sites=int(num_sites), amplitude=-float(dv)),
        nn_density_weight=0.0,
        include_quadratures=True,
        nn_hop_weight=float(t),
        nnn_hop_weight=0.0,
    )


def build_ionic_hubbard_hamiltonian(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    ordering: str,
    boundary: str,
) -> PauliPolynomial:
    """Build the ionic Hubbard Hamiltonian with a staggered onsite potential of amplitude dv."""
    return (
        build_hubbard_kinetic(
            dims=int(num_sites),
            t=float(t),
            repr_mode="JW",
            indexing=str(ordering),
            pbc=_boundary_to_pbc(boundary),
        )
        + build_hubbard_onsite(
            dims=int(num_sites),
            U=float(u),
            repr_mode="JW",
            indexing=str(ordering),
        )
        + build_hubbard_potential(
            dims=int(num_sites),
            v=_staggered_site_potential(num_sites=int(num_sites), amplitude=float(dv)),
            repr_mode="JW",
            indexing=str(ordering),
        )
    )


def build_ionic_hubbard_blocks(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    ordering: str,
    boundary: str,
) -> list[tuple[str, PauliPolynomial]]:
    return [
        (
            "hop_layer",
            build_hubbard_kinetic(
                dims=int(num_sites),
                t=float(t),
                repr_mode="JW",
                indexing=str(ordering),
                pbc=_boundary_to_pbc(boundary),
            ),
        ),
        (
            "onsite_layer",
            build_hubbard_onsite(
                dims=int(num_sites),
                U=float(u),
                repr_mode="JW",
                indexing=str(ordering),
            ),
        ),
        (
            "ionic_potential_layer",
            build_hubbard_potential(
                dims=int(num_sites),
                v=_staggered_site_potential(num_sites=int(num_sites), amplitude=float(dv)),
                repr_mode="JW",
                indexing=str(ordering),
            ),
        ),
    ]


MATH_EXTENDED_HUBBARD = r"H_{\mathrm{ext}} = -t\sum_{\langle i,j\rangle,\sigma}(c^\dagger_{i\sigma}c_{j\sigma}+\mathrm{h.c.}) + U\sum_i n_{i\uparrow}n_{i\downarrow} + V\sum_{\langle i,j\rangle} n_i n_j - \sum_{i,\sigma} v_i n_{i\sigma}"


def build_extended_hubbard_hamiltonian(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    v_nn: float,
    ordering: str,
    boundary: str,
) -> PauliPolynomial:
    """Build the extended Hubbard Hamiltonian with nearest-neighbor density interaction v_nn."""
    return (
        build_hubbard_hamiltonian_like(
            num_sites=int(num_sites),
            t=float(t),
            u=float(u),
            dv=float(dv),
            ordering=str(ordering),
            boundary=str(boundary),
        )
        + build_spinful_nearest_neighbor_density_interaction(
            num_sites=int(num_sites),
            v_nn=float(v_nn),
            ordering=str(ordering),
            boundary=str(boundary),
        )
    )


def build_extended_hubbard_blocks(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    v_nn: float,
    ordering: str,
    boundary: str,
) -> list[tuple[str, PauliPolynomial]]:
    return [
        (
            "hop_layer",
            build_hubbard_kinetic(
                dims=int(num_sites),
                t=float(t),
                repr_mode="JW",
                indexing=str(ordering),
                pbc=_boundary_to_pbc(boundary),
            ),
        ),
        (
            "onsite_layer",
            build_hubbard_onsite(
                dims=int(num_sites),
                U=float(u),
                repr_mode="JW",
                indexing=str(ordering),
            ),
        ),
        (
            "potential_layer",
            build_hubbard_potential(
                dims=int(num_sites),
                v=float(dv),
                repr_mode="JW",
                indexing=str(ordering),
            ),
        ),
        (
            "nn_density_layer",
            build_spinful_nearest_neighbor_density_interaction(
                num_sites=int(num_sites),
                v_nn=float(v_nn),
                ordering=str(ordering),
                boundary=str(boundary),
            ),
        ),
    ]


MATH_EXTENDED_HUBBARD_HVA = r"\{T_{ij,\sigma},\,W_{ij},\,X_{ij},\,U_i,\,v N_i,\,V N_iN_j\}"


def build_extended_hubbard_hva_terms(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    v_nn: float,
    ordering: str,
    boundary: str,
) -> list[tuple[str, PauliPolynomial]]:
    """Build physical Hamiltonian primitives for the extended Hubbard family."""
    return _build_spinful_termwise_family(
        num_sites=int(num_sites),
        ordering=str(ordering),
        boundary=str(boundary),
        onsite_u=float(u),
        site_density_weights=[-float(dv)] * int(num_sites),
        nn_density_weight=float(v_nn),
        include_quadratures=False,
        nn_hop_weight=float(t),
        nnn_hop_weight=0.0,
        include_bond_charge_terms=False,
        include_opposite_spin_assisted_terms=False,
        include_exchange_terms=True,
        include_three_site_bridge_terms=False,
    )


MATH_EXTENDED_HUBBARD_QUADRATURES = r"\{T_{ij,\sigma},\,J_{ij,\sigma},\,W_{ij},\,J^{(W)}_{ij},\,X_{ij},\,J^{(X)}_{ij},\,U_i,\,v N_i,\,V N_iN_j\}"


def build_extended_hubbard_quadratures(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    v_nn: float,
    ordering: str,
    boundary: str,
) -> list[tuple[str, PauliPolynomial]]:
    """Build extended Hubbard physical primitives plus hopping quadrature partners."""
    return _build_spinful_termwise_family(
        num_sites=int(num_sites),
        ordering=str(ordering),
        boundary=str(boundary),
        onsite_u=float(u),
        site_density_weights=[-float(dv)] * int(num_sites),
        nn_density_weight=float(v_nn),
        include_quadratures=True,
        nn_hop_weight=float(t),
        nnn_hop_weight=0.0,
        include_bond_charge_terms=False,
        include_opposite_spin_assisted_terms=False,
        include_exchange_terms=True,
        include_three_site_bridge_terms=False,
    )


MATH_TTPRIME_HUBBARD = r"H_{t,t',U} = -t\sum_{\langle i,j\rangle,\sigma}(c^\dagger_{i\sigma}c_{j\sigma}+\mathrm{h.c.}) - t'\sum_{\langle\!\langle i,j\rangle\!\rangle,\sigma}(c^\dagger_{i\sigma}c_{j\sigma}+\mathrm{h.c.}) + U\sum_i n_{i\uparrow}n_{i\downarrow} - \sum_{i,\sigma} v_i n_{i\sigma}"


def build_ttprime_hubbard_hamiltonian(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    t_prime: float,
    ordering: str,
    boundary: str,
) -> PauliPolynomial:
    """Build a 1D t-t'-U Hubbard chain with optional local potential dv."""
    nn_edges = bravais_nearest_neighbor_edges(int(num_sites), pbc=_boundary_to_pbc(boundary))
    nnn_edges = _chain_edges(num_sites=int(num_sites), hop_distance=2, boundary=str(boundary))
    return (
        build_hubbard_kinetic(
            dims=int(num_sites),
            t=float(t),
            repr_mode="JW",
            indexing=str(ordering),
            edges=nn_edges,
            pbc=_boundary_to_pbc(boundary),
        )
        + build_hubbard_kinetic(
            dims=int(num_sites),
            t=float(t_prime),
            repr_mode="JW",
            indexing=str(ordering),
            edges=nnn_edges,
            pbc=_boundary_to_pbc(boundary),
        )
        + build_hubbard_onsite(
            dims=int(num_sites),
            U=float(u),
            repr_mode="JW",
            indexing=str(ordering),
        )
        + build_hubbard_potential(
            dims=int(num_sites),
            v=float(dv),
            repr_mode="JW",
            indexing=str(ordering),
        )
    )


def build_ttprime_hubbard_blocks(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    t_prime: float,
    ordering: str,
    boundary: str,
) -> list[tuple[str, PauliPolynomial]]:
    nn_edges = bravais_nearest_neighbor_edges(int(num_sites), pbc=_boundary_to_pbc(boundary))
    nnn_edges = _chain_edges(num_sites=int(num_sites), hop_distance=2, boundary=str(boundary))
    return [
        (
            "hop_nn_layer",
            build_hubbard_kinetic(
                dims=int(num_sites),
                t=float(t),
                repr_mode="JW",
                indexing=str(ordering),
                edges=nn_edges,
                pbc=_boundary_to_pbc(boundary),
            ),
        ),
        (
            "hop_nnn_layer",
            build_hubbard_kinetic(
                dims=int(num_sites),
                t=float(t_prime),
                repr_mode="JW",
                indexing=str(ordering),
                edges=nnn_edges,
                pbc=_boundary_to_pbc(boundary),
            ),
        ),
        (
            "onsite_layer",
            build_hubbard_onsite(
                dims=int(num_sites),
                U=float(u),
                repr_mode="JW",
                indexing=str(ordering),
            ),
        ),
        (
            "potential_layer",
            build_hubbard_potential(
                dims=int(num_sites),
                v=float(dv),
                repr_mode="JW",
                indexing=str(ordering),
            ),
        ),
    ]


MATH_TTPRIME_HUBBARD_HVA = r"\{T_{ij,\sigma}^{(1)},\,T_{ij,\sigma}^{(2)},\,\widetilde{T}_{ij,\sigma}^{(\bar{\sigma})},\,\widetilde{T}_{ik|j,\sigma}^{(3)},\,U_i,\,v N_i\}"


def build_ttprime_hubbard_hva_terms(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    t_prime: float,
    ordering: str,
    boundary: str,
) -> list[tuple[str, PauliPolynomial]]:
    """Build physical Hamiltonian primitives for the t-t'-U Hubbard family."""
    return _build_spinful_termwise_family(
        num_sites=int(num_sites),
        ordering=str(ordering),
        boundary=str(boundary),
        onsite_u=float(u),
        site_density_weights=[-float(dv)] * int(num_sites),
        nn_density_weight=0.0,
        include_quadratures=False,
        nn_hop_weight=float(t),
        nnn_hop_weight=float(t_prime),
        include_bond_charge_terms=False,
        include_opposite_spin_assisted_terms=True,
        include_exchange_terms=False,
        include_three_site_bridge_terms=True,
    )


MATH_TTPRIME_HUBBARD_QUADRATURES = r"\{T_{ij,\sigma}^{(1)},\,J_{ij,\sigma}^{(1)},\,T_{ij,\sigma}^{(2)},\,J_{ij,\sigma}^{(2)},\,\widetilde{T}_{ij,\sigma}^{(\bar{\sigma})},\,\widetilde{J}_{ij,\sigma}^{(\bar{\sigma})},\,\widetilde{T}_{ik|j,\sigma}^{(3)},\,\widetilde{J}_{ik|j,\sigma}^{(3)},\,U_i,\,v N_i\}"


def build_ttprime_hubbard_quadratures(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    t_prime: float,
    ordering: str,
    boundary: str,
) -> list[tuple[str, PauliPolynomial]]:
    """Build t-t'-U Hubbard primitives plus hopping quadrature partners."""
    return _build_spinful_termwise_family(
        num_sites=int(num_sites),
        ordering=str(ordering),
        boundary=str(boundary),
        onsite_u=float(u),
        site_density_weights=[-float(dv)] * int(num_sites),
        nn_density_weight=0.0,
        include_quadratures=True,
        nn_hop_weight=float(t),
        nnn_hop_weight=float(t_prime),
        include_bond_charge_terms=False,
        include_opposite_spin_assisted_terms=True,
        include_exchange_terms=False,
        include_three_site_bridge_terms=True,
    )


MATH_SPINLESS_TV = r"H_{tV} = -t\sum_{\langle i,j\rangle}(c^\dagger_i c_j + \mathrm{h.c.}) + V\sum_{\langle i,j\rangle} n_i n_j - \sum_i v_i n_i"


def _spinless_number_operator(*, nq: int, mode: int) -> PauliPolynomial:
    return jw_number_operator("JW", int(nq), int(mode))


def build_spinless_kinetic(
    *,
    num_sites: int,
    t: float,
    boundary: str,
) -> PauliPolynomial:
    nq = int(num_sites)
    edges = bravais_nearest_neighbor_edges(int(num_sites), pbc=_boundary_to_pbc(boundary))
    c_dag: dict[int, PauliPolynomial] = {}
    c: dict[int, PauliPolynomial] = {}

    def cd(mode: int) -> PauliPolynomial:
        if int(mode) not in c_dag:
            c_dag[int(mode)] = fermion_plus_operator("JW", nq, int(mode))
        return c_dag[int(mode)]

    def cm(mode: int) -> PauliPolynomial:
        if int(mode) not in c:
            c[int(mode)] = fermion_minus_operator("JW", nq, int(mode))
        return c[int(mode)]

    h_t = PauliPolynomial("JW")
    for site_i, site_j in edges:
        h_t += (-float(t)) * ((cd(int(site_i)) * cm(int(site_j))) + (cd(int(site_j)) * cm(int(site_i))))
    return h_t


MATH_SPINLESS_EDGE_CURRENT = r"J_{ij} = i\,t(c^\dagger_i c_j - c^\dagger_j c_i)"


def build_spinless_edge_current_primitive(
    *,
    num_sites: int,
    site_i: int,
    site_j: int,
    amplitude: float,
) -> PauliPolynomial:
    nq = int(num_sites)
    cd = lambda mode: fermion_plus_operator("JW", nq, int(mode))
    cm = lambda mode: fermion_minus_operator("JW", nq, int(mode))
    return (1.0j * float(amplitude)) * (
        (cd(int(site_i)) * cm(int(site_j))) - (cd(int(site_j)) * cm(int(site_i)))
    )


def build_spinless_edge_hopping_primitive(
    *,
    num_sites: int,
    site_i: int,
    site_j: int,
    amplitude: float,
) -> PauliPolynomial:
    nq = int(num_sites)
    cd = lambda mode: fermion_plus_operator("JW", nq, int(mode))
    cm = lambda mode: fermion_minus_operator("JW", nq, int(mode))
    return (-float(amplitude)) * (
        (cd(int(site_i)) * cm(int(site_j))) + (cd(int(site_j)) * cm(int(site_i)))
    )


def build_spinless_density_interaction(
    *,
    num_sites: int,
    v_nn: float,
    boundary: str,
) -> PauliPolynomial:
    nq = int(num_sites)
    edges = bravais_nearest_neighbor_edges(int(num_sites), pbc=_boundary_to_pbc(boundary))
    h_v = PauliPolynomial("JW")
    if abs(float(v_nn)) <= 1e-15:
        return h_v
    for site_i, site_j in edges:
        h_v += float(v_nn) * (
            _spinless_number_operator(nq=nq, mode=int(site_i))
            * _spinless_number_operator(nq=nq, mode=int(site_j))
        )
    return h_v


def build_spinless_potential(
    *,
    num_sites: int,
    dv: float,
) -> PauliPolynomial:
    nq = int(num_sites)
    h_v = PauliPolynomial("JW")
    if abs(float(dv)) <= 1e-15:
        return h_v
    for mode in range(int(num_sites)):
        h_v += (-float(dv)) * _spinless_number_operator(nq=nq, mode=int(mode))
    return h_v


def build_spinless_site_density_primitive(
    *,
    num_sites: int,
    site: int,
    amplitude: float,
) -> PauliPolynomial:
    if abs(float(amplitude)) <= 1e-15:
        return PauliPolynomial("JW")
    return float(amplitude) * _spinless_number_operator(nq=int(num_sites), mode=int(site))


def build_spinless_edge_density_primitive(
    *,
    num_sites: int,
    site_i: int,
    site_j: int,
    amplitude: float,
) -> PauliPolynomial:
    if abs(float(amplitude)) <= 1e-15:
        return PauliPolynomial("JW")
    return float(amplitude) * (
        _spinless_number_operator(nq=int(num_sites), mode=int(site_i))
        * _spinless_number_operator(nq=int(num_sites), mode=int(site_j))
    )


MATH_SPINLESS_BOND_DENSITY = r"B_{ij}^{(f)} = n_i + n_j"


def build_spinless_bond_density_primitive(
    *,
    num_sites: int,
    site_i: int,
    site_j: int,
    amplitude: float,
) -> PauliPolynomial:
    """Build the spinless bond-density primitive n_i + n_j."""
    if abs(float(amplitude)) <= 1e-15:
        return PauliPolynomial("JW")
    return float(amplitude) * (
        _spinless_number_operator(nq=int(num_sites), mode=int(site_i))
        + _spinless_number_operator(nq=int(num_sites), mode=int(site_j))
    )


MATH_SPINLESS_BOND_CHARGE_HOP = r"\widetilde{T}_{ij}^{(B)} = (n_i+n_j)\,T_{ij}"


def build_spinless_bond_charge_hopping_primitive(
    *,
    num_sites: int,
    site_i: int,
    site_j: int,
    amplitude: float,
) -> PauliPolynomial:
    """Build a spinless bond-charge-assisted hopping primitive."""
    bond_density = build_spinless_bond_density_primitive(
        num_sites=int(num_sites),
        site_i=int(site_i),
        site_j=int(site_j),
        amplitude=1.0,
    )
    hop = build_spinless_edge_hopping_primitive(
        num_sites=int(num_sites),
        site_i=int(site_i),
        site_j=int(site_j),
        amplitude=float(amplitude),
    )
    return bond_density * hop


MATH_SPINLESS_BOND_CHARGE_CURRENT = r"\widetilde{J}_{ij}^{(B)} = (n_i+n_j)\,J_{ij}"


def build_spinless_bond_charge_current_primitive(
    *,
    num_sites: int,
    site_i: int,
    site_j: int,
    amplitude: float,
) -> PauliPolynomial:
    """Build a spinless bond-charge-assisted current primitive."""
    bond_density = build_spinless_bond_density_primitive(
        num_sites=int(num_sites),
        site_i=int(site_i),
        site_j=int(site_j),
        amplitude=1.0,
    )
    current = build_spinless_edge_current_primitive(
        num_sites=int(num_sites),
        site_i=int(site_i),
        site_j=int(site_j),
        amplitude=float(amplitude),
    )
    return bond_density * current


MATH_SPINLESS_CENTERED_SITE_DENSITY = r"\widetilde{n}_j = n_j - \tfrac12"


def build_spinless_centered_site_density_primitive(
    *,
    num_sites: int,
    site: int,
) -> PauliPolynomial:
    """Build the centered spinless site density n_j - 1/2."""
    return _spinless_number_operator(nq=int(num_sites), mode=int(site)) - (
        0.5 * _identity_polynomial(nq=int(num_sites))
    )


MATH_SPINLESS_THREE_SITE_BRIDGE_HOP = r"\widetilde{T}_{ik|j}^{(3)} = (n_j-\tfrac12)\,T_{ik}"


def build_spinless_three_site_bridge_hopping_primitive(
    *,
    num_sites: int,
    site_i: int,
    site_j: int,
    site_k: int,
    amplitude: float,
) -> PauliPolynomial:
    """Build a spinless three-site bridge hopping primitive."""
    center = build_spinless_centered_site_density_primitive(
        num_sites=int(num_sites),
        site=int(site_j),
    )
    hop = build_spinless_edge_hopping_primitive(
        num_sites=int(num_sites),
        site_i=int(site_i),
        site_j=int(site_k),
        amplitude=float(amplitude),
    )
    return center * hop


MATH_SPINLESS_THREE_SITE_BRIDGE_CURRENT = r"\widetilde{J}_{ik|j}^{(3)} = (n_j-\tfrac12)\,J_{ik}"


def build_spinless_three_site_bridge_current_primitive(
    *,
    num_sites: int,
    site_i: int,
    site_j: int,
    site_k: int,
    amplitude: float,
) -> PauliPolynomial:
    """Build a spinless three-site bridge current primitive."""
    center = build_spinless_centered_site_density_primitive(
        num_sites=int(num_sites),
        site=int(site_j),
    )
    current = build_spinless_edge_current_primitive(
        num_sites=int(num_sites),
        site_i=int(site_i),
        site_j=int(site_k),
        amplitude=float(amplitude),
    )
    return center * current


def _build_spinless_termwise_family(
    *,
    num_sites: int,
    boundary: str,
    nn_hop_weight: float,
    nn_density_weight: float,
    site_density_weights: Sequence[float] | None,
    include_quadratures: bool,
    include_bond_charge_terms: bool = False,
    include_three_site_bridge_terms: bool = False,
) -> list[tuple[str, PauliPolynomial]]:
    primitives: list[tuple[str, PauliPolynomial]] = []
    edges = bravais_nearest_neighbor_edges(int(num_sites), pbc=_boundary_to_pbc(boundary))
    if abs(float(nn_hop_weight)) > 1e-15:
        for site_i, site_j in edges:
            primitives.append(
                (
                    f"hop_nn({int(site_i)},{int(site_j)})",
                    build_spinless_edge_hopping_primitive(
                        num_sites=int(num_sites),
                        site_i=int(site_i),
                        site_j=int(site_j),
                        amplitude=float(nn_hop_weight),
                    ),
                )
            )
            if bool(include_quadratures):
                primitives.append(
                    (
                        f"hop_nn_quadrature({int(site_i)},{int(site_j)})",
                        build_spinless_edge_current_primitive(
                            num_sites=int(num_sites),
                            site_i=int(site_i),
                            site_j=int(site_j),
                            amplitude=float(nn_hop_weight),
                        ),
                    )
                )
            if bool(include_bond_charge_terms):
                primitives.append(
                    (
                        f"bond_charge_hop_nn({int(site_i)},{int(site_j)})",
                        build_spinless_bond_charge_hopping_primitive(
                            num_sites=int(num_sites),
                            site_i=int(site_i),
                            site_j=int(site_j),
                            amplitude=float(nn_hop_weight),
                        ),
                    )
                )
                if bool(include_quadratures):
                    primitives.append(
                        (
                            f"bond_charge_current_nn({int(site_i)},{int(site_j)})",
                            build_spinless_bond_charge_current_primitive(
                                num_sites=int(num_sites),
                                site_i=int(site_i),
                                site_j=int(site_j),
                                amplitude=float(nn_hop_weight),
                            ),
                        )
                    )
    if bool(include_three_site_bridge_terms):
        for site_i, site_j, site_k in _chain_triples(num_sites=int(num_sites), boundary=str(boundary)):
            primitives.append(
                (
                    f"three_site_bridge_hop({int(site_i)},{int(site_j)},{int(site_k)})",
                    build_spinless_three_site_bridge_hopping_primitive(
                        num_sites=int(num_sites),
                        site_i=int(site_i),
                        site_j=int(site_j),
                        site_k=int(site_k),
                        amplitude=float(nn_hop_weight),
                    ),
                )
            )
            if bool(include_quadratures):
                primitives.append(
                    (
                        f"three_site_bridge_current({int(site_i)},{int(site_j)},{int(site_k)})",
                        build_spinless_three_site_bridge_current_primitive(
                            num_sites=int(num_sites),
                            site_i=int(site_i),
                            site_j=int(site_j),
                            site_k=int(site_k),
                            amplitude=float(nn_hop_weight),
                        ),
                    )
                )
    if abs(float(nn_density_weight)) > 1e-15:
        for site_i, site_j in edges:
            primitives.append(
                (
                    f"nn_density({int(site_i)},{int(site_j)})",
                    build_spinless_edge_density_primitive(
                        num_sites=int(num_sites),
                        site_i=int(site_i),
                        site_j=int(site_j),
                        amplitude=float(nn_density_weight),
                    ),
                )
            )
    weights = (
        [0.0] * int(num_sites)
        if site_density_weights is None
        else [float(x) for x in site_density_weights]
    )
    if len(weights) != int(num_sites):
        raise ValueError("site_density_weights must match num_sites.")
    for site, weight in enumerate(weights):
        if abs(float(weight)) <= 1e-15:
            continue
        primitives.append(
            (
                f"site_density({int(site)})",
                build_spinless_site_density_primitive(
                    num_sites=int(num_sites),
                    site=int(site),
                    amplitude=float(weight),
                ),
            )
        )
    return primitives


def build_spinless_tv_hamiltonian(
    *,
    num_sites: int,
    t: float,
    v_nn: float,
    dv: float,
    boundary: str,
) -> PauliPolynomial:
    """Build the spinless t-V Hamiltonian with an optional uniform local potential dv."""
    return (
        build_spinless_kinetic(
            num_sites=int(num_sites),
            t=float(t),
            boundary=str(boundary),
        )
        + build_spinless_density_interaction(
            num_sites=int(num_sites),
            v_nn=float(v_nn),
            boundary=str(boundary),
        )
        + build_spinless_potential(
            num_sites=int(num_sites),
            dv=float(dv),
        )
    )


def build_spinless_tv_blocks(
    *,
    num_sites: int,
    t: float,
    v_nn: float,
    dv: float,
    boundary: str,
) -> list[tuple[str, PauliPolynomial]]:
    return [
        (
            "spinless_hop_layer",
            build_spinless_kinetic(
                num_sites=int(num_sites),
                t=float(t),
                boundary=str(boundary),
            ),
        ),
        (
            "spinless_nn_density_layer",
            build_spinless_density_interaction(
                num_sites=int(num_sites),
                v_nn=float(v_nn),
                boundary=str(boundary),
            ),
        ),
        (
            "spinless_potential_layer",
            build_spinless_potential(
                num_sites=int(num_sites),
                dv=float(dv),
            ),
        ),
    ]


MATH_SPINLESS_TV_HVA = r"\{T_{ij},\,\widetilde{T}_{ik|j}^{(3)},\,V n_i n_j,\,v n_i\}"


def build_spinless_tv_hva_terms(
    *,
    num_sites: int,
    t: float,
    v_nn: float,
    dv: float,
    boundary: str,
) -> list[tuple[str, PauliPolynomial]]:
    """Build physical Hamiltonian primitives for the spinless t-V family."""
    return _build_spinless_termwise_family(
        num_sites=int(num_sites),
        boundary=str(boundary),
        nn_hop_weight=float(t),
        nn_density_weight=float(v_nn),
        site_density_weights=[-float(dv)] * int(num_sites),
        include_quadratures=False,
        include_bond_charge_terms=False,
        include_three_site_bridge_terms=True,
    )


MATH_SPINLESS_TV_QUADRATURES = r"\{T_{ij},\,J_{ij},\,\widetilde{T}_{ik|j}^{(3)},\,\widetilde{J}_{ik|j}^{(3)},\,V n_i n_j,\,v n_i\}"


def build_spinless_tv_quadratures(
    *,
    num_sites: int,
    t: float,
    v_nn: float,
    dv: float,
    boundary: str,
) -> list[tuple[str, PauliPolynomial]]:
    """Build spinless t-V physical primitives plus hopping quadrature partners."""
    return _build_spinless_termwise_family(
        num_sites=int(num_sites),
        boundary=str(boundary),
        nn_hop_weight=float(t),
        nn_density_weight=float(v_nn),
        site_density_weights=[-float(dv)] * int(num_sites),
        include_quadratures=True,
        include_bond_charge_terms=False,
        include_three_site_bridge_terms=True,
    )


def build_hubbard_hamiltonian_like(
    *,
    num_sites: int,
    t: float,
    u: float,
    dv: float,
    ordering: str,
    boundary: str,
) -> PauliPolynomial:
    return (
        build_hubbard_kinetic(
            dims=int(num_sites),
            t=float(t),
            repr_mode="JW",
            indexing=str(ordering),
            pbc=_boundary_to_pbc(boundary),
        )
        + build_hubbard_onsite(
            dims=int(num_sites),
            U=float(u),
            repr_mode="JW",
            indexing=str(ordering),
        )
        + build_hubbard_potential(
            dims=int(num_sites),
            v=float(dv),
            repr_mode="JW",
            indexing=str(ordering),
        )
    )


def spinless_reference_statevector(*, num_sites: int, n_fermions: int) -> np.ndarray:
    if int(num_sites) <= 0:
        raise ValueError("num_sites must be positive")
    if int(n_fermions) < 0 or int(n_fermions) > int(num_sites):
        raise ValueError("n_fermions must satisfy 0 <= n_fermions <= num_sites")
    basis_index = 0
    for mode in range(int(n_fermions)):
        basis_index |= (1 << int(mode))
    psi = np.zeros(1 << int(num_sites), dtype=complex)
    psi[int(basis_index)] = 1.0 + 0.0j
    return psi


def spinless_reference_bitstring(*, num_sites: int, n_fermions: int) -> str:
    return bitstring_qn1_to_q0(int(num_sites), tuple(range(int(n_fermions))))


__all__ = [
    "build_extended_hubbard_blocks",
    "build_extended_hubbard_hva_terms",
    "build_extended_hubbard_hamiltonian",
    "build_extended_hubbard_quadratures",
    "build_hubbard_hamiltonian_like",
    "build_ionic_hubbard_blocks",
    "build_ionic_hubbard_hva_terms",
    "build_ionic_hubbard_hamiltonian",
    "build_ionic_hubbard_quadratures",
    "build_spinless_density_interaction",
    "build_spinless_edge_current_primitive",
    "build_spinless_centered_site_density_primitive",
    "build_spinless_edge_density_primitive",
    "build_spinless_edge_hopping_primitive",
    "build_spinless_kinetic",
    "build_spinless_potential",
    "build_spinless_three_site_bridge_current_primitive",
    "build_spinless_three_site_bridge_hopping_primitive",
    "build_spinless_tv_blocks",
    "build_spinless_tv_hva_terms",
    "build_spinless_tv_hamiltonian",
    "build_spinless_tv_quadratures",
    "build_spinless_site_density_primitive",
    "build_spinful_edge_current_primitive",
    "build_spinful_edge_density_primitive",
    "build_spinful_edge_exchange_current_primitive",
    "build_spinful_edge_exchange_primitive",
    "build_spinful_edge_hopping_primitive",
    "build_spinful_edge_pair_current_primitive",
    "build_spinful_edge_pair_hop_primitive",
    "build_spinful_nearest_neighbor_density_interaction",
    "build_spinful_opposite_spin_assisted_current_primitive",
    "build_spinful_opposite_spin_assisted_hopping_primitive",
    "build_spinful_centered_opposite_spin_site_density_primitive",
    "build_spinful_onsite_primitive",
    "build_spinful_site_density_primitive",
    "build_spinful_three_site_bridge_current_primitive",
    "build_spinful_three_site_bridge_hopping_primitive",
    "build_ttprime_hubbard_blocks",
    "build_ttprime_hubbard_hva_terms",
    "build_ttprime_hubbard_hamiltonian",
    "build_ttprime_hubbard_quadratures",
    "spinless_reference_bitstring",
    "spinless_reference_statevector",
]
