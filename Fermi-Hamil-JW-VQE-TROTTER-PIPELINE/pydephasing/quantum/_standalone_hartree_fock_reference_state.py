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

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


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


def hartree_fock_circuit(
    n_sites: int,
    num_particles: SpinParticles,
    *,
    indexing: str = "blocked",
):
    """
    Return a Qiskit QuantumCircuit that prepares the HF determinant by X-flipping occupied qubits.

    Kept optional: Qiskit is imported inside the function.
    """
    from qiskit import QuantumCircuit  # local import keeps core module dependency-light

    n_qubits = 2 * int(n_sites)
    occ = hartree_fock_occupied_qubits(n_sites, num_particles, indexing=indexing)

    qc = QuantumCircuit(n_qubits, name="HF")
    for q in occ:
        qc.x(q)
    return qc


if __name__ == "__main__":
    # Minimal self-checks (not exhaustive).
    assert hartree_fock_bitstring(2, (1, 1), indexing="blocked") == "0101"
    assert hartree_fock_bitstring(2, (1, 1), indexing="interleaved") == "0011"
    assert hartree_fock_bitstring(3, (2, 1), indexing="blocked") == "001011"
    assert hartree_fock_bitstring(3, (2, 1), indexing="interleaved") == "000111"