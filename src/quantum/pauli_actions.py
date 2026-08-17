"""Compiled Pauli-string action helpers (exyz convention).

These utilities implement statevector action for a single Pauli string
without forming dense matrices.

For an input basis index ``i``, a Pauli word acts as

``P|i> = i**n_y (-1)**popcount(i & phase_mask) |i ^ flip_mask>``.

The two integer masks are therefore a complete, constant-size compiled
representation, and they remain the stored form of every action.

The permutation/sign arrays derived from those masks are ``2**n`` long, so
retaining one per Pauli word does not scale: at 20 qubits a table is 9.2 MB and
an operator pool holds hundreds of words.  They are therefore materialized on
demand, and memoized only below ``_PERMUTATION_TABLE_CACHE_MAX_NQ`` where a
table is small enough to be free (36 KB at 12 qubits, 0.6 KB at 6).  Above that
ceiling the arrays are rebuilt per call exactly as before.  The cache changes no
value it returns; it only avoids recomputing them.

Math:
    P|psi>  via bit-mask permutation + phase
    exp(-i * dt * c * P)|psi> = cos(theta)|psi> - i sin(theta) P|psi>
    where theta = dt * Re(c)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True, slots=True)
class CompiledPauliAction:
    label_exyz: str
    nq: int
    flip_mask: int
    phase_mask: int
    y_count_mod4: int

    @property
    def dimension(self) -> int:
        return 1 << int(self.nq)

    @property
    def retained_bytes(self) -> int:
        """Return the compact numeric payload size, excluding Python headers."""

        mask_bytes = max(1, (int(self.nq) + 7) // 8)
        return int(2 * mask_bytes + 1)

    @property
    def perm(self) -> np.ndarray:
        """Materialize the legacy input-to-output permutation on demand."""

        permutation, _ = materialize_compiled_pauli_action(self)
        return permutation

    @property
    def phase(self) -> np.ndarray:
        """Materialize the legacy input-index phase table on demand."""

        _, phase = materialize_compiled_pauli_action(self)
        return phase


@lru_cache(maxsize=None)
def _basis_indices(nq: int) -> np.ndarray:
    indices = np.arange(1 << int(nq), dtype=np.int64)
    indices.flags.writeable = False
    return indices


def _phase_prefactor(action: CompiledPauliAction) -> complex:
    return (1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j, 0.0 - 1.0j)[
        int(action.y_count_mod4)
    ]


def _phase_signs(indices: np.ndarray, phase_mask: int) -> np.ndarray | None:
    if int(phase_mask) == 0:
        return None
    masked = np.bitwise_and(indices, np.int64(phase_mask))
    parity = np.bitwise_and(np.bitwise_count(masked), np.uint8(1)).astype(
        np.int8,
        copy=False,
    )
    return np.asarray(1 - 2 * parity, dtype=np.int8)


#: Largest qubit count whose permutation/sign tables are memoized.  One table is
#: ``9 * 2**nq`` bytes, so 12 qubits is 36 KB per Pauli word; beyond this the
#: arrays are rebuilt per call to keep memory O(1) per operator.
_PERMUTATION_TABLE_CACHE_MAX_NQ = 12

#: Bound on retained tables, so a large operator pool cannot grow without limit.
_PERMUTATION_TABLE_CACHE_MAX_ENTRIES = 512


def _build_permutation_and_signs(
    nq: int,
    flip_mask: int,
    phase_mask: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Materialize the source permutation and phase signs for one action."""

    indices = _basis_indices(int(nq))
    source_indices = (
        indices
        if int(flip_mask) == 0
        else np.bitwise_xor(indices, np.int64(flip_mask))
    )
    if source_indices is not indices:
        source_indices.flags.writeable = False
    signs = _phase_signs(source_indices, int(phase_mask))
    if signs is not None:
        signs.flags.writeable = False
    return source_indices, signs


@lru_cache(maxsize=_PERMUTATION_TABLE_CACHE_MAX_ENTRIES)
def _cached_permutation_and_signs(
    nq: int,
    flip_mask: int,
    phase_mask: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    return _build_permutation_and_signs(nq, flip_mask, phase_mask)


def _permutation_and_signs(
    action: CompiledPauliAction,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return read-only permutation/sign arrays for ``action``.

    Both arrays are pure functions of the action's masks, and the same handful
    of actions is applied repeatedly during tangent transport, so small systems
    reuse them.  Large systems rebuild, because the table is what does not
    scale.  Returned arrays are read-only; callers index with them and must not
    write through them.
    """

    nq = int(action.nq)
    masks = (nq, int(action.flip_mask), int(action.phase_mask))
    if nq <= _PERMUTATION_TABLE_CACHE_MAX_NQ:
        return _cached_permutation_and_signs(*masks)
    return _build_permutation_and_signs(*masks)


def materialize_compiled_pauli_action(
    action: CompiledPauliAction,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize legacy permutation/phase arrays for one transient use."""

    indices = _basis_indices(int(action.nq))
    return materialize_compiled_pauli_action_on_inputs(action, indices)


def materialize_compiled_pauli_action_on_inputs(
    action: CompiledPauliAction,
    input_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return output indices and phases for selected full-space inputs."""

    indices = np.asarray(input_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError(
            f"input_indices must be one-dimensional; got {indices.shape}."
        )
    if indices.size and (
        int(np.min(indices)) < 0
        or int(np.max(indices)) >= int(action.dimension)
    ):
        raise ValueError(
            "input_indices lie outside the compiled Pauli action dimension."
        )
    permutation = np.bitwise_xor(indices, np.int64(action.flip_mask))
    signs = _phase_signs(indices, int(action.phase_mask))
    prefactor = _phase_prefactor(action)
    if signs is None:
        phase = np.full(indices.shape, prefactor, dtype=complex)
    else:
        phase = np.asarray(prefactor * signs, dtype=complex)
    return np.asarray(permutation, dtype=np.int64), phase


def compile_pauli_action_exyz(label_exyz: str, nq: int) -> CompiledPauliAction:
    """Compile an exyz Pauli label into constant-size action masks.

    The label convention is q_(n-1)...q_0 (qubit 0 rightmost).
    """
    nq_value = int(nq)
    label = str(label_exyz)
    if nq_value < 0:
        raise ValueError(f"nq must be nonnegative, got {nq_value}.")
    if len(label) != nq_value:
        raise ValueError(
            f"Pauli label length mismatch: got {len(label)}, expected {nq_value}."
        )
    flip_mask = 0
    phase_mask = 0
    y_count = 0
    for q, op in enumerate(reversed(label)):
        bit = 1 << int(q)
        if op == "e":
            continue
        if op == "x":
            flip_mask |= bit
            continue
        if op == "y":
            flip_mask |= bit
            phase_mask |= bit
            y_count += 1
            continue
        if op == "z":
            phase_mask |= bit
            continue
        raise ValueError(f"Unsupported Pauli symbol '{op}' in '{label}'.")

    return CompiledPauliAction(
        label_exyz=label,
        nq=nq_value,
        flip_mask=int(flip_mask),
        phase_mask=int(phase_mask),
        y_count_mod4=int(y_count % 4),
    )


def apply_compiled_pauli(psi: np.ndarray, action: CompiledPauliAction) -> np.ndarray:
    """Apply a compiled Pauli action to a statevector."""
    psi_vec = np.asarray(psi, dtype=complex).reshape(-1)
    if psi_vec.size != int(action.dimension):
        raise ValueError(
            "Statevector length does not match compiled Pauli action: "
            f"{psi_vec.size} vs {action.dimension}."
        )
    source_indices, signs = _permutation_and_signs(action)
    # Fancy indexing returns a fresh array, so the in-place scaling below never
    # writes through the shared read-only tables.
    out = np.asarray(psi_vec[source_indices], dtype=complex)
    if signs is not None:
        out *= signs
    prefactor = _phase_prefactor(action)
    if prefactor != 1.0 + 0.0j:
        out *= prefactor
    return out


def apply_compiled_pauli_to_columns(
    values: np.ndarray,
    action: CompiledPauliAction,
) -> np.ndarray:
    """Apply one compact Pauli action to statevector columns."""

    matrix = np.asarray(values, dtype=complex)
    if matrix.ndim != 2:
        raise ValueError(
            f"values must be a rank-2 column matrix; got {matrix.shape}."
        )
    if int(matrix.shape[0]) != int(action.dimension):
        raise ValueError(
            "Pauli action dimension does not match tangent column matrix: "
            f"{action.dimension} vs {matrix.shape[0]}."
        )
    source_indices, signs = _permutation_and_signs(action)
    out = np.asarray(matrix[source_indices, :], dtype=complex)
    if signs is not None:
        out *= signs[:, None]
    prefactor = _phase_prefactor(action)
    if prefactor != 1.0 + 0.0j:
        out *= prefactor
    return out


def apply_exp_term(
    psi: np.ndarray,
    action: CompiledPauliAction,
    coeff: complex,
    dt: float,
    tol: float = 1e-12,
) -> np.ndarray:
    """Apply exp(-i * dt * coeff * P) using the compiled Pauli action."""
    coeff_c = complex(coeff)
    if abs(coeff_c.imag) > float(tol):
        raise ValueError(f"Imaginary coefficient encountered for {action.label_exyz}: {coeff_c}")
    theta = float(dt) * float(coeff_c.real)
    ppsi = apply_compiled_pauli(psi, action)
    return math.cos(theta) * psi - 1j * math.sin(theta) * ppsi


__all__ = [
    "CompiledPauliAction",
    "compile_pauli_action_exyz",
    "apply_compiled_pauli",
    "apply_compiled_pauli_to_columns",
    "materialize_compiled_pauli_action",
    "materialize_compiled_pauli_action_on_inputs",
    "apply_exp_term",
]
