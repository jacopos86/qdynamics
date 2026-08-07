"""
Independent exact diagonalization (ED) for the Hubbard-Holstein model.

This module builds the HH Hamiltonian directly in the occupation-number
(Fock) basis — no Pauli algebra.  It serves as a completely independent
cross-check for the Pauli-polynomial Hamiltonian construction.

Supports both binary and unary (one-hot) boson-to-qubit encodings for
the qubit-index mapping.  The physical Hamiltonian matrix elements are
encoding-agnostic; only ``encode_state_to_qubit_index`` changes.

Adapted from ``_imports/.../pydephasing/quantum/ed_hubbard_holstein.py``.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from scipy.sparse import coo_matrix, spmatrix
except Exception:  # pragma: no cover - SciPy is optional
    coo_matrix = None  # type: ignore[assignment]
    spmatrix = None  # type: ignore[assignment]

from src.quantum.hubbard_latex_python_pairs import (
    Dims,
    SPIN_DN,
    SPIN_UP,
    boson_qubits_per_site,
    bravais_nearest_neighbor_edges,
    mode_index,
    n_sites_from_dims,
)

SitePotential = Optional[Union[float, Sequence[float], Dict[int, float]]]
MatrixLike = Union[np.ndarray, "spmatrix"]

__all__ = [
    # Data structures
    "HHPhysicalState",
    "HHEDBasis",
    # Encoding / basis construction
    "encode_state_to_qubit_index",
    "build_hh_sector_basis",
    # Hamiltonian
    "build_hh_sector_hamiltonian_ed",
    # Utilities
    "matrix_to_dense",
    "hermiticity_residual",
]


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HHPhysicalState:
    """A single physical basis state: fermion occupation + phonon occupations."""
    fermion_bits: int
    phonons: Tuple[int, ...]


@dataclass
class HHEDBasis:
    """Sector-restricted basis for HH exact diagonalization."""
    dims: Dims
    n_sites: int
    n_ph_max: int
    num_particles: Tuple[int, int]
    indexing: str
    boson_encoding: str
    qpb: int
    fermion_qubits: int
    total_qubits: int
    basis_states: List[HHPhysicalState]
    basis_indices: List[int]
    state_to_index: Dict[Tuple[int, Tuple[int, ...]], int]

    @property
    def dimension(self) -> int:
        return int(len(self.basis_states))


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _parse_site_potential(v: SitePotential, n_sites: int) -> List[float]:
    if v is None:
        return [0.0] * int(n_sites)
    if isinstance(v, (int, float, complex)):
        return [float(v)] * int(n_sites)
    if isinstance(v, dict):
        out = [0.0] * int(n_sites)
        for k, val in v.items():
            idx = int(k)
            if idx < 0 or idx >= int(n_sites):
                raise ValueError("site-potential key out of bounds")
            out[idx] = float(val)
        return out
    if len(v) != int(n_sites):
        raise ValueError("site potential must be scalar, dict, or length n_sites")
    return [float(val) for val in v]


def _spin_orbital_index_sets(
    n_sites: int, indexing: str,
) -> Tuple[List[int], List[int]]:
    n_sites_i = int(n_sites)
    up = [mode_index(i, SPIN_UP, indexing=indexing, n_sites=n_sites_i)
          for i in range(n_sites_i)]
    dn = [mode_index(i, SPIN_DN, indexing=indexing, n_sites=n_sites_i)
          for i in range(n_sites_i)]
    return up, dn


def _fermion_sector_bitstrings(
    *,
    n_sites: int,
    num_particles: Tuple[int, int],
    indexing: str,
) -> List[int]:
    n_sites_i = int(n_sites)
    n_up = int(num_particles[0])
    n_dn = int(num_particles[1])
    if n_up < 0 or n_dn < 0 or n_up > n_sites_i or n_dn > n_sites_i:
        raise ValueError("num_particles must satisfy 0 <= N_up,N_dn <= n_sites")

    up_idx, dn_idx = _spin_orbital_index_sets(n_sites_i, indexing=indexing)
    n_ferm = 2 * n_sites_i
    out: List[int] = []
    for bits in range(1 << n_ferm):
        na = sum((bits >> q) & 1 for q in up_idx)
        if na != n_up:
            continue
        nb = sum((bits >> q) & 1 for q in dn_idx)
        if nb == n_dn:
            out.append(int(bits))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Qubit-index encoding  (encoding-aware)
# ═══════════════════════════════════════════════════════════════════════════

def encode_state_to_qubit_index(
    *,
    fermion_bits: int,
    phonons: Sequence[int],
    n_sites: int,
    n_ph_max: int,
    qpb: int,
    boson_encoding: str = "binary",
) -> int:
    """Map a physical state to its computational-basis index.

    binary:  phonon level n is stored as the integer n in the qpb-bit register.
    unary:   phonon level n is one-hot: qubit n is |1⟩, rest are |0⟩.
    """
    n_sites_i = int(n_sites)
    n_ph_i = int(n_ph_max)
    qpb_i = int(qpb)
    if qpb_i <= 0:
        raise ValueError("qpb must be positive")
    if len(phonons) != n_sites_i:
        raise ValueError("phonons length must equal n_sites")

    fermion_qubits = 2 * n_sites_i
    idx = int(fermion_bits)
    enc = str(boson_encoding).strip().lower()

    for site, nloc in enumerate(phonons):
        n_i = int(nloc)
        if n_i < 0 or n_i > n_ph_i:
            raise ValueError("phonon occupation out of bounds")
        base = fermion_qubits + site * qpb_i

        if enc == "binary":
            idx |= (n_i << base)
        elif enc == "unary":
            # One-hot: set qubit (base + n_i) to |1⟩
            idx |= (1 << (base + n_i))
        else:
            raise ValueError(f"Unknown boson encoding '{boson_encoding}'")
    return idx


# ═══════════════════════════════════════════════════════════════════════════
# Basis construction
# ═══════════════════════════════════════════════════════════════════════════

def build_hh_sector_basis(
    *,
    dims: Dims,
    n_ph_max: int,
    num_particles: Tuple[int, int],
    indexing: str = "blocked",
    boson_encoding: str = "binary",
) -> HHEDBasis:
    """Build the sector-restricted basis for HH exact diagonalization.

    Supports both binary and unary boson encodings.
    """
    n_sites = int(n_sites_from_dims(dims))
    n_ph_i = int(n_ph_max)
    if n_ph_i < 0:
        raise ValueError("n_ph_max must be >= 0")
    enc = str(boson_encoding).strip().lower()
    if enc not in ("binary", "unary"):
        raise ValueError(f"Unknown boson encoding '{boson_encoding}'")

    qpb = int(boson_qubits_per_site(n_ph_i, enc))
    fermion_qubits = 2 * n_sites
    total_qubits = fermion_qubits + n_sites * qpb

    fermion_bits = _fermion_sector_bitstrings(
        n_sites=n_sites,
        num_particles=(int(num_particles[0]), int(num_particles[1])),
        indexing=str(indexing),
    )
    phonon_configs = list(itertools.product(range(n_ph_i + 1), repeat=n_sites))

    packed: List[Tuple[int, HHPhysicalState]] = []
    for f_bits in fermion_bits:
        for ph in phonon_configs:
            ph_t = tuple(int(x) for x in ph)
            comp_idx = encode_state_to_qubit_index(
                fermion_bits=int(f_bits),
                phonons=ph_t,
                n_sites=n_sites,
                n_ph_max=n_ph_i,
                qpb=qpb,
                boson_encoding=enc,
            )
            packed.append((comp_idx, HHPhysicalState(
                fermion_bits=int(f_bits), phonons=ph_t,
            )))

    packed.sort(key=lambda t: t[0])
    basis_indices = [int(idx) for idx, _ in packed]
    basis_states = [st for _, st in packed]
    state_to_index: Dict[Tuple[int, Tuple[int, ...]], int] = {
        (int(st.fermion_bits), tuple(st.phonons)): i
        for i, st in enumerate(basis_states)
    }
    return HHEDBasis(
        dims=dims,
        n_sites=n_sites,
        n_ph_max=n_ph_i,
        num_particles=(int(num_particles[0]), int(num_particles[1])),
        indexing=str(indexing),
        boson_encoding=enc,
        qpb=qpb,
        fermion_qubits=fermion_qubits,
        total_qubits=total_qubits,
        basis_states=basis_states,
        basis_indices=basis_indices,
        state_to_index=state_to_index,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Fermion hop helper
# ═══════════════════════════════════════════════════════════════════════════

def _annihilation_phase(fermion_bits: int, q: int) -> int:
    parity = (int(fermion_bits) & ((1 << int(q)) - 1)).bit_count() & 1
    return -1 if parity else 1


def _creation_phase(fermion_bits: int, q: int) -> int:
    parity = (int(fermion_bits) & ((1 << int(q)) - 1)).bit_count() & 1
    return -1 if parity else 1


def _apply_cdag_c(
    fermion_bits: int, p: int, q: int,
) -> Optional[Tuple[int, float]]:
    """Apply c_p^† c_q to a fermion bitstring.  Return (new_bits, sign) or None."""
    bits = int(fermion_bits)
    p_i, q_i = int(p), int(q)
    if ((bits >> q_i) & 1) == 0:
        return None
    if ((bits >> p_i) & 1) == 1:
        return None
    sign_a = _annihilation_phase(bits, q_i)
    after_a = bits ^ (1 << q_i)
    sign_c = _creation_phase(after_a, p_i)
    after_c = after_a | (1 << p_i)
    return int(after_c), float(sign_a * sign_c)


# ═══════════════════════════════════════════════════════════════════════════
# Sparse entry accumulator
# ═══════════════════════════════════════════════════════════════════════════

def _append_entry(
    rows: List[int], cols: List[int], data: List[complex],
    row: int, col: int, value: complex, *, tol: float,
) -> None:
    if abs(complex(value)) <= float(tol):
        return
    rows.append(int(row))
    cols.append(int(col))
    data.append(complex(value))


# ═══════════════════════════════════════════════════════════════════════════
# Main ED Hamiltonian builder
# ═══════════════════════════════════════════════════════════════════════════

def build_hh_sector_hamiltonian_ed(
    *,
    dims: Dims,
    J: float,
    U: float,
    omega0: float,
    g: float,
    n_ph_max: int,
    num_particles: Tuple[int, int],
    indexing: str = "blocked",
    boson_encoding: str = "binary",
    pbc: Union[bool, Sequence[bool]] = True,
    edges: Optional[Sequence[Tuple[int, int]]] = None,
    delta_v: SitePotential = None,
    include_zero_point: bool = True,
    basis: Optional[HHEDBasis] = None,
    sparse: bool = True,
    tol: float = 1e-14,
    return_basis: bool = False,
) -> Union[MatrixLike, Tuple[MatrixLike, HHEDBasis]]:
    r"""
    Build HH Hamiltonian directly in the occupation-number basis (no Pauli algebra).

    H = H_t + H_U + H_{ph} + H_g + H_{drive}

    H_t     = -J  \sum_{<ij>,\sigma} (c^\dagger_{i\sigma} c_{j\sigma} + h.c.)
    H_U     =  U  \sum_i  n_{i\uparrow} n_{i\downarrow}
    H_{ph}  = \omega_0 \sum_i (n_{b,i} + 1/2)          [if include_zero_point]
    H_g     =  g  \sum_i  x_i (n_i - 1)
    H_{drv} = \sum_{i\sigma} \delta v_i \, n_{i\sigma}
    """
    if basis is None:
        basis_i = build_hh_sector_basis(
            dims=dims,
            n_ph_max=int(n_ph_max),
            num_particles=(int(num_particles[0]), int(num_particles[1])),
            indexing=str(indexing),
            boson_encoding=str(boson_encoding),
        )
    else:
        basis_i = basis

    n_sites = int(basis_i.n_sites)
    if edges is None:
        edges_i = list(bravais_nearest_neighbor_edges(dims, pbc=pbc))
    else:
        edges_i = [(int(a), int(b)) for (a, b) in edges]

    delta_v_i = _parse_site_potential(delta_v, n_sites=n_sites)
    dim = int(basis_i.dimension)

    rows: List[int] = []
    cols: List[int] = []
    data: List[complex] = []

    J_i = float(J)
    U_i = float(U)
    omega_i = float(omega0)
    g_i = float(g)
    n_ph_i = int(n_ph_max)
    idx_mode = str(indexing)

    for col, st in enumerate(basis_i.basis_states):
        f_bits = int(st.fermion_bits)
        phon = tuple(int(x) for x in st.phonons)

        # ── diagonal: H_U + H_drive + H_ph ────────────────────────────
        diag = 0.0
        for site in range(n_sites):
            p_up = mode_index(site, SPIN_UP, indexing=idx_mode, n_sites=n_sites)
            p_dn = mode_index(site, SPIN_DN, indexing=idx_mode, n_sites=n_sites)
            n_up = (f_bits >> int(p_up)) & 1
            n_dn = (f_bits >> int(p_dn)) & 1

            diag += U_i * float(n_up * n_dn)
            diag += float(delta_v_i[site]) * float(n_up + n_dn)
            diag += omega_i * float(phon[site])

        if include_zero_point:
            diag += 0.5 * omega_i * float(n_sites)
        _append_entry(rows, cols, data, col, col, complex(diag), tol=tol)

        # ── hopping: H_t ──────────────────────────────────────────────
        if abs(J_i) > tol:
            for (i_site, j_site) in edges_i:
                for spin in (SPIN_UP, SPIN_DN):
                    p_i = mode_index(i_site, spin, indexing=idx_mode, n_sites=n_sites)
                    p_j = mode_index(j_site, spin, indexing=idx_mode, n_sites=n_sites)
                    for p_mode, q_mode in ((p_i, p_j), (p_j, p_i)):
                        hop = _apply_cdag_c(f_bits, p_mode, q_mode)
                        if hop is None:
                            continue
                        f_new, sign = hop
                        row = basis_i.state_to_index.get((int(f_new), phon))
                        if row is None:
                            continue
                        amp = -J_i * float(sign)
                        _append_entry(rows, cols, data, row, col, complex(amp), tol=tol)

        # ── electron-phonon coupling: H_g = g * x_i * (n_i - 1) ──────
        if abs(g_i) > tol:
            for site in range(n_sites):
                p_up = mode_index(site, SPIN_UP, indexing=idx_mode, n_sites=n_sites)
                p_dn = mode_index(site, SPIN_DN, indexing=idx_mode, n_sites=n_sites)
                n_i = ((f_bits >> int(p_up)) & 1) + ((f_bits >> int(p_dn)) & 1)
                pref = g_i * (float(n_i) - 1.0)
                if abs(pref) <= tol:
                    continue

                n_loc = int(phon[site])
                # x_i = b + b†:  <n+1|x|n> = sqrt(n+1),  <n-1|x|n> = sqrt(n)
                if n_loc < n_ph_i:
                    ph_up = list(phon)
                    ph_up[site] = n_loc + 1
                    row_up = basis_i.state_to_index.get((f_bits, tuple(ph_up)))
                    if row_up is not None:
                        amp_up = pref * math.sqrt(float(n_loc + 1))
                        _append_entry(rows, cols, data, row_up, col, complex(amp_up), tol=tol)

                if n_loc > 0:
                    ph_dn = list(phon)
                    ph_dn[site] = n_loc - 1
                    row_dn = basis_i.state_to_index.get((f_bits, tuple(ph_dn)))
                    if row_dn is not None:
                        amp_dn = pref * math.sqrt(float(n_loc))
                        _append_entry(rows, cols, data, row_dn, col, complex(amp_dn), tol=tol)

    # ── assemble matrix ───────────────────────────────────────────────
    if sparse and coo_matrix is not None:
        h_mat = coo_matrix(
            (np.array(data, dtype=complex), (rows, cols)),
            shape=(dim, dim), dtype=complex,
        ).tocsr()
        h_mat.sum_duplicates()
    else:
        h_mat = np.zeros((dim, dim), dtype=complex)
        for r, c, val in zip(rows, cols, data):
            h_mat[int(r), int(c)] += complex(val)

    if return_basis:
        return h_mat, basis_i
    return h_mat


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def matrix_to_dense(h_mat: MatrixLike) -> np.ndarray:
    if spmatrix is not None and isinstance(h_mat, spmatrix):
        return np.asarray(h_mat.toarray(), dtype=complex)
    return np.asarray(h_mat, dtype=complex)


def hermiticity_residual(h_mat: MatrixLike) -> float:
    if spmatrix is not None and isinstance(h_mat, spmatrix):
        diff = (h_mat - h_mat.getH()).tocsr()
        if diff.nnz == 0:
            return 0.0
        return float(np.max(np.abs(diff.data)))
    dense = matrix_to_dense(h_mat)
    return float(np.max(np.abs(dense - dense.conj().T))) if dense.size else 0.0
