"""Matrix-free Pauli-polynomial action on a bridge array backend.

The action reuses the repository's compiled-Pauli representation
(``src.quantum.pauli_actions``: ``P|i> = i**n_y (-1)**popcount(i & phase_mask)
|i ^ flip_mask>``) and the repository's polynomial compilation
(``src.quantum.compiled_polynomial.compile_polynomial_action``: merge equal
words, drop |c| <= tol).  Each word is materialized once into a permutation and
a complex weight ``c_k * phase_k``; applying the polynomial is then
``sum_k weight_k ⊙ psi[perm_k]`` plus the identity coefficient, on whichever
backend the caller binds.

The CPU float64 numpy path is bit-for-bit the same arithmetic the repository's
``apply_compiled_polynomial`` performs (same permutation/phase tables); the
tests check that identity directly.  Everything a method runner needs that is
linear in the Hamiltonian action lives here: expectation values, the
fixed-particle sector matrix, ADAPT commutator gradients and the QSE pencil.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Any, Sequence

import numpy as np

from pipelines.chemistry_bridge.array_backend import NumpyBackend, TorchBackend
from src.quantum.compiled_polynomial import CompiledPolynomialAction, compile_polynomial_action
from src.quantum.pauli_actions import materialize_compiled_pauli_action
from src.quantum.pauli_polynomial_class import PauliPolynomial


Backend = NumpyBackend | TorchBackend


@dataclass(frozen=True)
class MaterializedTerm:
    label: str
    coeff: complex
    perm: np.ndarray | None
    weight: np.ndarray | None  # coeff * phase, complex128, length 2**nq; None for identity


@dataclass(frozen=True)
class BridgePolynomialAction:
    """Backend-neutral compiled polynomial: ordered words, permutations, weights."""

    nq: int
    terms: tuple[MaterializedTerm, ...]
    compile_tol: float
    action_sha256: str

    @property
    def dimension(self) -> int:
        return 1 << int(self.nq)

    @property
    def term_count(self) -> int:
        return len(self.terms)

    @property
    def identity_coefficient(self) -> complex:
        return sum((term.coeff for term in self.terms if term.perm is None), 0.0 + 0.0j)

    @classmethod
    def from_compiled(cls, compiled: CompiledPolynomialAction, *, compile_tol: float) -> "BridgePolynomialAction":
        materialized: list[MaterializedTerm] = []
        digest = hashlib.sha256()
        digest.update(struct.pack(">Q", int(compiled.nq)))
        for term in compiled.terms:
            coeff = complex(term.coeff)
            if term.action is None:
                label = "e" * int(compiled.nq)
                materialized.append(MaterializedTerm(label=label, coeff=coeff, perm=None, weight=None))
            else:
                label = str(term.action.label_exyz)
                # materialize_* returns the scatter form P|i> = phase[i] |perm[i]>.
                # In gather form out[j] = w[j] * psi[perm[j]] the weight is the
                # phase at the source index perm[j] (perm is an XOR involution).
                perm, phase = materialize_compiled_pauli_action(term.action)
                perm = np.asarray(perm, dtype=np.int64)
                weight = np.asarray(coeff * np.asarray(phase, dtype=np.complex128)[perm], dtype=np.complex128)
                materialized.append(MaterializedTerm(label=label, coeff=coeff, perm=np.asarray(perm, dtype=np.int64), weight=weight))
            digest.update(label.encode("ascii"))
            digest.update(struct.pack("<dd", coeff.real, coeff.imag))
        return cls(nq=int(compiled.nq), terms=tuple(materialized), compile_tol=float(compile_tol), action_sha256=digest.hexdigest())

    @classmethod
    def from_polynomial(cls, poly: PauliPolynomial, *, compile_tol: float = 1.0e-15) -> "BridgePolynomialAction":
        """Compile with the repository's merge-then-drop semantics (``tol`` as in QSE core)."""

        compiled = compile_polynomial_action(poly, tol=float(compile_tol))
        return cls.from_compiled(compiled, compile_tol=float(compile_tol))

    def bind(self, backend: Backend) -> "BoundPolynomialAction":
        return BoundPolynomialAction(self, backend)


class BoundPolynomialAction:
    """A ``BridgePolynomialAction`` whose tables live on one backend."""

    def __init__(self, action: BridgePolynomialAction, backend: Backend) -> None:
        self.action = action
        self.backend = backend
        self._perms = []
        self._weights = []
        self._identity = complex(action.identity_coefficient)
        for term in action.terms:
            if term.perm is None:
                continue
            self._perms.append(backend.index_array(term.perm))
            self._weights.append(backend.from_complex(term.weight))

    @property
    def nq(self) -> int:
        return int(self.action.nq)

    def apply(self, psi_handle: Any) -> Any:
        out = self.backend.zeros(self.action.dimension)
        if self._identity != 0.0:
            self.backend.axpy_weight(out, self._identity, psi_handle)
        for perm, weight in zip(self._perms, self._weights):
            self.backend.axpy_weight(out, weight, self.backend.gather(psi_handle, perm))
        return out

    def apply_numpy(self, psi: np.ndarray) -> np.ndarray:
        handle = self.backend.from_complex(np.asarray(psi, dtype=np.complex128).reshape(-1))
        out = self.apply(handle)
        self.backend.synchronize()
        return self.backend.to_complex(out)

    def expectation(self, psi_handle: Any) -> complex:
        return self.backend.vdot(psi_handle, self.apply(psi_handle))

    def expectation_numpy(self, psi: np.ndarray) -> complex:
        handle = self.backend.from_complex(np.asarray(psi, dtype=np.complex128).reshape(-1))
        value = self.expectation(handle)
        self.backend.synchronize()
        return value


# ----------------------------------------------------------------------------
# Sector bookkeeping (blocked spin-orbital order, qubit 0 rightmost)
# ----------------------------------------------------------------------------


def blocked_spin_orbital_masks(n_spatial: int) -> tuple[int, int]:
    """Bit masks of the alpha (qubits 0..n-1) and beta (qubits n..2n-1) registers."""

    n = int(n_spatial)
    alpha_mask = (1 << n) - 1
    beta_mask = alpha_mask << n
    return alpha_mask, beta_mask


def fixed_sector_basis_indices(n_spatial: int, num_particles: tuple[int, int]) -> np.ndarray:
    """Full-register basis indices with exactly (n_alpha, n_beta) occupations (blocked order)."""

    n = int(n_spatial)
    nq = 2 * n
    alpha_mask, beta_mask = blocked_spin_orbital_masks(n)
    indices = np.arange(1 << nq, dtype=np.int64)
    n_alpha = np.bitwise_count(np.bitwise_and(indices, np.int64(alpha_mask)))
    n_beta = np.bitwise_count(np.bitwise_and(indices, np.int64(beta_mask)))
    mask = (n_alpha == int(num_particles[0])) & (n_beta == int(num_particles[1]))
    return indices[mask]


def out_of_sector_norm(psi: np.ndarray, basis_indices: np.ndarray) -> float:
    """Norm of the component of ``psi`` outside the span of ``basis_indices``."""

    vector = np.asarray(psi, dtype=np.complex128).reshape(-1)
    inside = np.zeros(vector.shape, dtype=bool)
    inside[np.asarray(basis_indices, dtype=np.int64)] = True
    return float(np.linalg.norm(vector[~inside]))


def sector_matrix(bound: BoundPolynomialAction, basis_indices: np.ndarray) -> tuple[np.ndarray, float]:
    """Dense Hamiltonian block on the sector basis, assembled by one action per basis state.

    Returns the block and the largest out-of-sector leakage over all columns
    (a nonzero value means the operator does not preserve the sector).
    """

    basis = np.asarray(basis_indices, dtype=np.int64)
    dim = int(basis.size)
    columns = np.zeros((1 << bound.nq, dim), dtype=np.complex128)
    for col, index in enumerate(basis):
        unit = np.zeros(1 << bound.nq, dtype=np.complex128)
        unit[int(index)] = 1.0
        columns[:, col] = bound.apply_numpy(unit)
    block = columns[basis, :]
    leakage = max((out_of_sector_norm(columns[:, col], basis) for col in range(dim)), default=0.0)
    return np.asarray(block, dtype=np.complex128), float(leakage)


def sector_spectrum(block: np.ndarray, *, hermitian_atol: float = 1.0e-10) -> np.ndarray:
    """Ascending real eigenvalues of a Hermitian block (checks Hermiticity first)."""

    matrix = np.asarray(block, dtype=np.complex128)
    residual = float(np.max(np.abs(matrix - matrix.conj().T))) if matrix.size else 0.0
    if residual > float(hermitian_atol):
        raise ValueError(f"sector block is not Hermitian: residual {residual:.3e} > {hermitian_atol:.1e}")
    return np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T))


# ----------------------------------------------------------------------------
# ADAPT commutator gradient (repository formula g = 2 Im <H psi | A psi>)
# ----------------------------------------------------------------------------


def adapt_commutator_gradients(
    hamiltonian: BoundPolynomialAction,
    generators: Sequence[BoundPolynomialAction],
    psi: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return (gradients, energy) with g_k = 2 Im <H psi | A_k psi> at theta = 0.

    This is the ADAPT selection score of ``src.quantum.compiled_polynomial``
    (``adapt_commutator_grad_from_hpsi``), evaluated with one Hamiltonian action
    and one generator action per candidate on the bound backend.
    """

    backend = hamiltonian.backend
    psi_handle = backend.from_complex(np.asarray(psi, dtype=np.complex128).reshape(-1))
    h_psi = hamiltonian.apply(psi_handle)
    energy = backend.vdot(psi_handle, h_psi)
    gradients = np.zeros(len(generators), dtype=np.float64)
    for k, generator in enumerate(generators):
        a_psi = generator.apply(psi_handle)
        overlap = backend.vdot(h_psi, a_psi)
        gradients[k] = 2.0 * overlap.imag
    backend.synchronize()
    return gradients, float(energy.real)


# ----------------------------------------------------------------------------
# QSE pencil (mirrors pipelines.qse_spectra.core.build_qse_matrices)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class BridgeQSEPencil:
    overlap: np.ndarray
    hamiltonian: np.ndarray
    reference_energy: float
    basis_action_norms: tuple[float, ...]
    basis_projected_norms: tuple[float, ...]
    basis_matrix_vector_norms: tuple[float, ...]
    overlap_hermitian_residual_raw: float
    hamiltonian_hermitian_residual_raw: float


def qse_pencil(
    hamiltonian: BoundPolynomialAction,
    basis_actions: Sequence[BoundPolynomialAction | None],
    psi: np.ndarray,
    *,
    reference_projection: str = "none",
    basis_vector_normalization: str = "normalized",
) -> BridgeQSEPencil:
    """Projected QSE overlap/Hamiltonian matrices on the bound backend.

    ``basis_actions[i] is None`` denotes the identity record.  The policy
    arguments mirror ``QSEBasisVectorPolicy``: ``reference_projection`` in
    {"none", "q0"} and ``basis_vector_normalization`` in {"normalized",
    "raw_projected"}.  The returned matrices are Hermitian-symmetrized exactly
    as the core does (``0.5 * (M + M^H)``) after recording the raw residuals.
    """

    if reference_projection not in {"none", "q0"}:
        raise ValueError("reference_projection must be 'none' or 'q0'")
    if basis_vector_normalization not in {"normalized", "raw_projected"}:
        raise ValueError("basis_vector_normalization must be 'normalized' or 'raw_projected'")
    backend = hamiltonian.backend
    psi_np = np.asarray(psi, dtype=np.complex128).reshape(-1)
    norm = float(np.linalg.norm(psi_np))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("reference state norm must be positive and finite")
    psi_np = psi_np / norm
    psi_handle = backend.from_complex(psi_np)
    h_psi = hamiltonian.apply(psi_handle)
    reference_energy = backend.vdot(psi_handle, h_psi).real
    matrix_vectors: list[Any] = []
    action_norms: list[float] = []
    projected_norms: list[float] = []
    matrix_norms: list[float] = []
    for action in basis_actions:
        raw = psi_handle if action is None else action.apply(psi_handle)
        if action is None:
            raw = backend.zeros(1 << hamiltonian.nq)
            backend.axpy_weight(raw, 1.0 + 0.0j, psi_handle)
        raw_norm = backend.norm(raw)
        if reference_projection == "q0":
            overlap_before = backend.vdot(psi_handle, raw)
            projected = backend.zeros(1 << hamiltonian.nq)
            backend.axpy_weight(projected, 1.0 + 0.0j, raw)
            backend.axpy_weight(projected, -overlap_before, psi_handle)
        else:
            projected = raw
        projected_norm = backend.norm(projected)
        if basis_vector_normalization == "normalized" and projected_norm > 0.0:
            matrix_vector = backend.zeros(1 << hamiltonian.nq)
            backend.axpy_weight(matrix_vector, complex(1.0 / projected_norm), projected)
        else:
            matrix_vector = projected
        matrix_vectors.append(matrix_vector)
        action_norms.append(float(raw_norm))
        projected_norms.append(float(projected_norm))
        matrix_norms.append(float(backend.norm(matrix_vector)))
    count = len(matrix_vectors)
    overlap_raw = np.zeros((count, count), dtype=np.complex128)
    hamiltonian_raw = np.zeros((count, count), dtype=np.complex128)
    h_columns = [hamiltonian.apply(vector) for vector in matrix_vectors]
    for col in range(count):
        for row in range(count):
            overlap_raw[row, col] = backend.vdot(matrix_vectors[row], matrix_vectors[col])
            hamiltonian_raw[row, col] = backend.vdot(matrix_vectors[row], h_columns[col])
    backend.synchronize()
    overlap_residual = float(np.max(np.abs(overlap_raw - overlap_raw.conj().T))) if count else 0.0
    hamiltonian_residual = float(np.max(np.abs(hamiltonian_raw - hamiltonian_raw.conj().T))) if count else 0.0
    return BridgeQSEPencil(
        overlap=0.5 * (overlap_raw + overlap_raw.conj().T),
        hamiltonian=0.5 * (hamiltonian_raw + hamiltonian_raw.conj().T),
        reference_energy=float(reference_energy),
        basis_action_norms=tuple(action_norms),
        basis_projected_norms=tuple(projected_norms),
        basis_matrix_vector_norms=tuple(matrix_norms),
        overlap_hermitian_residual_raw=overlap_residual,
        hamiltonian_hermitian_residual_raw=hamiltonian_residual,
    )


__all__ = [
    "BoundPolynomialAction",
    "BridgePolynomialAction",
    "BridgeQSEPencil",
    "MaterializedTerm",
    "adapt_commutator_gradients",
    "blocked_spin_orbital_masks",
    "fixed_sector_basis_indices",
    "out_of_sector_norm",
    "qse_pencil",
    "sector_matrix",
    "sector_spectrum",
]
