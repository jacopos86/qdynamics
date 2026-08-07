"""
Hartree-Fock (Slater-determinant) reference state helpers for JW-mapped Hubbard-style models.

Design intent:
- Keep ordering conventions consistent with the JSON metadata exported by export_hubbard_jw_reference.py:
    pauli_string_qubit_order: left_to_right = q_(n-1) ... q_0
  i.e. qubit 0 is the least-significant / rightmost character in bitstrings.

- Support the two spin-orbital orderings used in this code base:
    * "blocked":     alpha0..alpha(L-1), beta0..beta(L-1)
    * "interleaved": alpha0,beta0,alpha1,beta1,...

The "Hartree-Fock" state here is the occupation-basis determinant obtained by filling the first
n_alpha alpha orbitals and first n_beta beta orbitals in the chosen ordering. This matches
Qiskit Nature's HartreeFock initial-state convention for lattice models when the Hamiltonian
is expressed in that same spin-orbital ordering.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

# HH qubit-count helper – imported lazily to avoid circular imports at module level.
try:
    from src.quantum.hubbard_latex_python_pairs import boson_qubits_per_site, n_sites_from_dims
except Exception:  # pragma: no cover
    def boson_qubits_per_site(n_ph_max: int, encoding: str = "binary") -> int:
        d = int(n_ph_max) + 1
        return max(1, int(math.ceil(math.log2(d))))

    def n_sites_from_dims(dims):
        if isinstance(dims, int):
            return int(dims)
        out = 1
        for L in dims:
            out *= int(L)
        return out

Dims = Union[int, Tuple[int, ...]]


SpinParticles = Tuple[int, int]  # (n_alpha, n_beta)


def _validate_indexing(indexing: str) -> str:
    if not isinstance(indexing, str):
        raise TypeError("indexing must be a string")
    normalized = indexing.strip().lower()
    if normalized not in {"blocked", "interleaved"}:
        raise ValueError("indexing must be 'blocked' or 'interleaved'")
    return normalized


def mode_index(site: int, spin: int, *, n_sites: int, indexing: str) -> int:
    """
    Map (site, spin) -> fermionic mode index p (== qubit index under JW).

    spin: 0 for alpha/up, 1 for beta/down.
    """
    if spin not in (0, 1):
        raise ValueError("spin must be 0 (alpha/up) or 1 (beta/down)")
    if site < 0 or site >= n_sites:
        raise ValueError("site out of range")
    idx = _validate_indexing(indexing)

    if idx == "interleaved":
        return 2 * site + spin
    # idx == "blocked"
    return site if spin == 0 else n_sites + site


def hartree_fock_occupied_qubits(
    n_sites: int,
    num_particles: SpinParticles,
    *,
    indexing: str = "blocked",
) -> List[int]:
    """
    Return the list of qubit indices (== JW modes) occupied in the HF determinant.

    Convention: occupy alpha sites 0..n_alpha-1 and beta sites 0..n_beta-1.
    """
    if n_sites <= 0:
        raise ValueError("n_sites must be positive")
    n_alpha, n_beta = (int(num_particles[0]), int(num_particles[1]))
    if n_alpha < 0 or n_beta < 0:
        raise ValueError("num_particles entries must be non-negative")
    if n_alpha > n_sites or n_beta > n_sites:
        raise ValueError("cannot occupy more than n_sites orbitals per spin")

    idx = _validate_indexing(indexing)
    occ: List[int] = []
    for i in range(n_alpha):
        occ.append(mode_index(i, 0, n_sites=n_sites, indexing=idx))
    for i in range(n_beta):
        occ.append(mode_index(i, 1, n_sites=n_sites, indexing=idx))
    return sorted(set(occ))


def bitstring_qn1_to_q0(n_qubits: int, occupied_qubits: Sequence[int]) -> str:
    """
    Return a computational-basis label in q_(n-1)...q_0 order.

    Example (n_qubits=4, occupied=[0,2]) -> "0101" i.e. |q3 q2 q1 q0> = |0 1 0 1>.
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    bits = ["0"] * n_qubits
    for q in occupied_qubits:
        if q < 0 or q >= n_qubits:
            raise ValueError("occupied qubit index out of range")
        bits[n_qubits - 1 - q] = "1"
    return "".join(bits)


def hartree_fock_bitstring(
    n_sites: int,
    num_particles: SpinParticles,
    *,
    indexing: str = "blocked",
) -> str:
    """
    HF bitstring in q_(n-1)...q_0 order (compatible with Qiskit label convention).
    """
    n_qubits = 2 * int(n_sites)
    occ = hartree_fock_occupied_qubits(n_sites, num_particles, indexing=indexing)
    return bitstring_qn1_to_q0(n_qubits, occ)


def hartree_fock_statevector(
    n_sites: int,
    num_particles: SpinParticles,
    *,
    indexing: str = "blocked",
) -> np.ndarray:
    """
    Return the HF statevector |Phi_HF> as a length-2^(2*n_sites) complex numpy array.

    Little-endian convention: basis index = sum_q bit_q * 2^q, with qubit 0 the least significant.
    """
    n_qubits = 2 * int(n_sites)
    occ = hartree_fock_occupied_qubits(n_sites, num_particles, indexing=indexing)
    basis_index = 0
    for q in occ:
        basis_index |= (1 << q)

    dim = 1 << n_qubits
    psi = np.zeros(dim, dtype=complex)
    psi[basis_index] = 1.0 + 0.0j
    return psi


# ---------------------------------------------------------------------------
# Hubbard-Holstein reference state
# ---------------------------------------------------------------------------

def _half_filled_num_particles(n_sites: int) -> Tuple[int, int]:
    """Default half-filling: ceil(L/2) alpha, floor(L/2) beta."""
    L = int(n_sites)
    if L <= 0:
        raise ValueError("n_sites must be positive")
    return ((L + 1) // 2, L // 2)


def _phonon_vacuum_bitstring(n_sites: int, qpb: int, boson_encoding: str) -> str:
    """
    Phonon vacuum bitstring for the full boson register (q_{n-1}...q_0 order).

    - binary:  all phonon qubits = 0  →  "0" * n_bos
    - unary:   per-site one-hot |n=0⟩ = qubit[0]=1, rest=0
               In bitstring notation the lowest-index qubit appears rightmost
               inside each site's block, so per-site: "0"*(qpb-1) + "1".
               Sites are ordered high-to-low (site L-1 leftmost).
    """
    encoding = str(boson_encoding).strip().lower()
    if encoding == "binary":
        return "0" * (int(n_sites) * int(qpb))
    if encoding == "unary":
        # Each site contributes qpb bits; vacuum = qubit[0]=1, rest=0.
        # In bitstring (left=high, right=low), this is "0"*(qpb-1) + "1".
        site_vac = ("0" * (int(qpb) - 1)) + "1"
        # Sites ordered high-to-low: site (L-1) is leftmost.
        return site_vac * int(n_sites)
    raise ValueError(f"Unknown boson encoding '{boson_encoding}'")


def hubbard_holstein_reference_state(
    *,
    dims: Dims,
    num_particles: Optional[SpinParticles] = None,
    n_ph_max: int,
    boson_encoding: str = "binary",
    indexing: str = "blocked",
) -> np.ndarray:
    r"""
    Hubbard-Holstein reference state = (fermionic HF determinant) ⊗ (phonon vacuum).

    Qubit register layout (left = high, right = low):
        [boson site-(L-1) | … | boson site-0 | fermion qubits]

    Bitstring in q_{n-1}…q_0 order:
        full_bitstring = phonon_vacuum_bitstring + hf_fermion_bitstring

    Phonon vacuum:
      - binary: all phonon qubits = 0
      - unary:  per-site one-hot |n=0⟩ (qubit[0] = 1 within each site block)

    Dimension: 2^{n_ferm + n_bos}, normalised to ||ψ|| = 1.
    """
    n_sites = int(n_sites_from_dims(dims))
    n_ferm = 2 * n_sites
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    n_bos = n_sites * qpb
    n_total = n_ferm + n_bos

    if num_particles is None:
        num_particles_i = _half_filled_num_particles(n_sites)
    else:
        num_particles_i = (int(num_particles[0]), int(num_particles[1]))

    hf_fermion_bs = str(
        hartree_fock_bitstring(
            n_sites=n_sites,
            num_particles=num_particles_i,
            indexing=str(indexing),
        )
    )
    phonon_vac_bs = _phonon_vacuum_bitstring(n_sites, qpb, str(boson_encoding))
    full_bitstring = phonon_vac_bs + hf_fermion_bs

    # Build statevector: little-endian index = int(bitstring, 2)
    dim = 1 << n_total
    idx = int(full_bitstring, 2)
    psi = np.zeros(dim, dtype=complex)
    psi[idx] = 1.0 + 0.0j
    return psi


if __name__ == "__main__":
    # Minimal self-checks (not exhaustive).
    assert hartree_fock_bitstring(2, (1, 1), indexing="blocked") == "0101"
    assert hartree_fock_bitstring(2, (1, 1), indexing="interleaved") == "0011"
    assert hartree_fock_bitstring(3, (2, 1), indexing="blocked") == "001011"
    assert hartree_fock_bitstring(3, (2, 1), indexing="interleaved") == "000111"