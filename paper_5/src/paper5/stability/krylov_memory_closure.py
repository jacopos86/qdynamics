"""Fixed Hilbert--Schmidt block-Krylov closure for the Holstein dimer.

The construction follows the raw-moment projection proposed in
``autonomous_holstein_moment_closure_memorandum.md``.  The physical 31-slot
centered state has two fixed correlation-trace coordinates, so its independent
raw chart has 29 real coordinates.  Projection is performed on the associated
Hermitian raw-observable operators; centering is applied only after the
projected raw velocity has been evaluated.

Exact wavefunctions are accepted only by offline construction and diagnostic
helpers.  The coefficient object and its online velocity methods contain no
exact-state or exact-trajectory input.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh, svd
from scipy.sparse import csc_matrix, eye, hstack, vstack

from .exact_reference import _ExactDimerModel, _build_exact_dimer_model
from .hubbard_dimer import DimerParameters
from .matrix_reference import CLOSED_SCALAR_STATE_NAMES

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
SparseOperator = csc_matrix
OperatorBlock = tuple[SparseOperator, ...]


RAW_MOMENT_NAMES = (
    "r_x",
    "r_y",
    "r_z",
    "beta_0_real",
    "beta_0_imag",
    "beta_1_real",
    "beta_1_imag",
    "M_00",
    "M_11",
    "M_01_real",
    "M_01_imag",
    "U_00_real",
    "U_00_imag",
    "U_01_real",
    "U_01_imag",
    "U_11_real",
    "U_11_imag",
    "t_0x_real",
    "t_0x_imag",
    "t_0y_real",
    "t_0y_imag",
    "t_0z_real",
    "t_0z_imag",
    "t_1x_real",
    "t_1x_imag",
    "t_1y_real",
    "t_1y_imag",
    "t_1z_real",
    "t_1z_imag",
)

_RAW_DIMENSION = len(RAW_MOMENT_NAMES)
_CLOSED_DIMENSION = len(CLOSED_SCALAR_STATE_NAMES)


def _require_vector(
    values: FloatArray,
    *,
    length: int,
    name: str,
) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape {(length,)}, got {array.shape}")
    return array


def _unpack_raw_moments(
    raw: FloatArray,
) -> tuple[FloatArray, ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
    values = _require_vector(raw, length=_RAW_DIMENSION, name="raw moments")
    bloch = values[:3].copy()
    coherent = np.array(
        [values[3] + 1j * values[4], values[5] + 1j * values[6]],
        dtype=complex,
    )
    normal = np.array(
        [
            [values[7], values[9] + 1j * values[10]],
            [values[9] - 1j * values[10], values[8]],
        ],
        dtype=complex,
    )
    anomalous = np.array(
        [
            [values[11] + 1j * values[12], values[13] + 1j * values[14]],
            [values[13] + 1j * values[14], values[15] + 1j * values[16]],
        ],
        dtype=complex,
    )
    mixed_pauli = np.empty((2, 3), dtype=complex)
    for phonon_index, offset in enumerate((17, 23)):
        for pauli_index in range(3):
            entry = offset + 2 * pauli_index
            mixed_pauli[phonon_index, pauli_index] = (
                values[entry] + 1j * values[entry + 1]
            )
    return bloch, coherent, normal, anomalous, mixed_pauli


def _pack_raw_moments(
    bloch: FloatArray,
    coherent: ComplexArray,
    normal: ComplexArray,
    anomalous: ComplexArray,
    mixed_pauli: ComplexArray,
) -> FloatArray:
    values = np.empty(_RAW_DIMENSION, dtype=float)
    values[:3] = np.asarray(bloch, dtype=float)
    values[3:7] = (
        coherent[0].real,
        coherent[0].imag,
        coherent[1].real,
        coherent[1].imag,
    )
    values[7:11] = (
        normal[0, 0].real,
        normal[1, 1].real,
        normal[0, 1].real,
        normal[0, 1].imag,
    )
    values[11:17] = (
        anomalous[0, 0].real,
        anomalous[0, 0].imag,
        anomalous[0, 1].real,
        anomalous[0, 1].imag,
        anomalous[1, 1].real,
        anomalous[1, 1].imag,
    )
    for phonon_index, offset in enumerate((17, 23)):
        for pauli_index in range(3):
            entry = offset + 2 * pauli_index
            values[entry] = mixed_pauli[phonon_index, pauli_index].real
            values[entry + 1] = mixed_pauli[phonon_index, pauli_index].imag
    return values


def raw_moments_to_closed_coordinates(raw: FloatArray) -> FloatArray:
    """Apply the exact raw-to-centered map into the 31-slot state chart."""

    bloch, coherent, normal, anomalous, mixed_pauli = _unpack_raw_moments(raw)
    centered_normal = normal - np.outer(coherent, coherent.conjugate())
    centered_anomalous = anomalous - np.outer(coherent, coherent)
    centered_mixed = mixed_pauli - coherent[:, None] * bloch[None, :]

    return _pack_closed_coordinates(
        bloch,
        coherent,
        centered_normal,
        centered_anomalous,
        centered_mixed,
    )


def _pack_closed_coordinates(
    bloch: FloatArray,
    coherent: ComplexArray,
    centered_normal: ComplexArray,
    centered_anomalous: ComplexArray,
    centered_mixed: ComplexArray,
) -> FloatArray:
    """Pack already-centered blocks without applying centering a second time."""

    closed = np.zeros(_CLOSED_DIMENSION, dtype=float)
    closed[:3] = (bloch[2], 0.5 * bloch[0], -0.5 * bloch[1])
    closed[3:7] = (
        coherent[0].real,
        coherent[0].imag,
        coherent[1].real,
        coherent[1].imag,
    )
    closed[7:11] = (
        centered_normal[0, 0].real,
        centered_normal[1, 1].real,
        centered_normal[0, 1].real,
        centered_normal[0, 1].imag,
    )
    closed[11:17] = (
        centered_anomalous[0, 0].real,
        centered_anomalous[0, 0].imag,
        centered_anomalous[1, 1].real,
        centered_anomalous[1, 1].imag,
        centered_anomalous[0, 1].real,
        centered_anomalous[0, 1].imag,
    )

    # The raw fixed-number packing enforces Tr(C^q) = 0 identically.
    closed[17:19] = 0.0
    for phonon_index, offset in enumerate((19, 25)):
        c_x, c_y, c_z = centered_mixed[phonon_index]
        correlation_01 = 0.5 * (c_x - 1j * c_y)
        correlation_10 = 0.5 * (c_x + 1j * c_y)
        closed[offset : offset + 6] = (
            c_z.real,
            c_z.imag,
            correlation_01.real,
            correlation_01.imag,
            correlation_10.real,
            correlation_10.imag,
        )
    return closed


def closed_coordinates_to_raw_moments(
    closed: FloatArray,
    *,
    trace_tolerance: float = 1e-12,
) -> FloatArray:
    """Invert the centered chart on its fixed-correlation-trace manifold."""

    values = _require_vector(
        closed,
        length=_CLOSED_DIMENSION,
        name="closed coordinates",
    )
    if np.linalg.norm(values[17:19], ord=np.inf) > trace_tolerance:
        raise ValueError(
            "the 29-coordinate raw chart requires the common C trace to be zero"
        )

    bloch = np.array([2.0 * values[1], -2.0 * values[2], values[0]])
    coherent = np.array(
        [values[3] + 1j * values[4], values[5] + 1j * values[6]],
        dtype=complex,
    )
    centered_normal = np.array(
        [
            [values[7], values[9] + 1j * values[10]],
            [values[9] - 1j * values[10], values[8]],
        ],
        dtype=complex,
    )
    normal = centered_normal + np.outer(coherent, coherent.conjugate())
    centered_anomalous = np.array(
        [
            [values[11] + 1j * values[12], values[15] + 1j * values[16]],
            [values[15] + 1j * values[16], values[13] + 1j * values[14]],
        ],
        dtype=complex,
    )
    anomalous = centered_anomalous + np.outer(coherent, coherent)

    mixed_pauli = np.empty((2, 3), dtype=complex)
    for phonon_index, offset in enumerate((19, 25)):
        c_z = values[offset] + 1j * values[offset + 1]
        correlation_01 = values[offset + 2] + 1j * values[offset + 3]
        correlation_10 = values[offset + 4] + 1j * values[offset + 5]
        c_x = correlation_01 + correlation_10
        c_y = 1j * (correlation_01 - correlation_10)
        mixed_pauli[phonon_index] = (
            np.array([c_x, c_y, c_z], dtype=complex)
            + coherent[phonon_index] * bloch
        )
    return _pack_raw_moments(
        bloch,
        coherent,
        normal,
        anomalous,
        mixed_pauli,
    )


def raw_velocity_to_closed_velocity(
    raw: FloatArray,
    raw_velocity: FloatArray,
) -> FloatArray:
    """Apply the analytic differential of the raw-to-centered map."""

    bloch, coherent, normal, anomalous, mixed_pauli = _unpack_raw_moments(raw)
    (
        bloch_velocity,
        coherent_velocity,
        normal_velocity,
        anomalous_velocity,
        mixed_velocity,
    ) = _unpack_raw_moments(raw_velocity)

    centered_normal_velocity = (
        normal_velocity
        - np.outer(coherent_velocity, coherent.conjugate())
        - np.outer(coherent, coherent_velocity.conjugate())
    )
    centered_anomalous_velocity = (
        anomalous_velocity
        - np.outer(coherent_velocity, coherent)
        - np.outer(coherent, coherent_velocity)
    )
    centered_mixed_velocity = (
        mixed_velocity
        - coherent_velocity[:, None] * bloch[None, :]
        - coherent[:, None] * bloch_velocity[None, :]
    )

    return _pack_closed_coordinates(
        bloch_velocity,
        coherent_velocity,
        centered_normal_velocity,
        centered_anomalous_velocity,
        centered_mixed_velocity,
    )


def raw_to_closed_jacobian(raw: FloatArray) -> FloatArray:
    """Return the analytic ``31 x 29`` Jacobian of the centering map."""

    values = _require_vector(raw, length=_RAW_DIMENSION, name="raw moments")
    jacobian = np.empty((_CLOSED_DIMENSION, _RAW_DIMENSION), dtype=float)
    for column in range(_RAW_DIMENSION):
        direction = np.zeros(_RAW_DIMENSION, dtype=float)
        direction[column] = 1.0
        jacobian[:, column] = raw_velocity_to_closed_velocity(values, direction)
    return jacobian


def _hermitian_real_part(operator: SparseOperator) -> SparseOperator:
    return (0.5 * (operator + operator.getH())).tocsc()


def _hermitian_imaginary_part(operator: SparseOperator) -> SparseOperator:
    return ((operator - operator.getH()) / (2j)).tocsc()


def _hs_inner(
    left: SparseOperator,
    right: SparseOperator,
    *,
    dimension: int,
) -> complex:
    return complex(left.conjugate().multiply(right).sum()) / dimension


def _block_gram(
    left: Sequence[SparseOperator],
    right: Sequence[SparseOperator],
    *,
    dimension: int,
) -> ComplexArray:
    matrix = np.empty((len(left), len(right)), dtype=complex)
    for row, left_operator in enumerate(left):
        for column, right_operator in enumerate(right):
            matrix[row, column] = _hs_inner(
                left_operator,
                right_operator,
                dimension=dimension,
            )
    return matrix


def _real_block_gram(
    left: Sequence[SparseOperator],
    right: Sequence[SparseOperator],
    *,
    dimension: int,
    tolerance: float = 1e-11,
) -> FloatArray:
    matrix = _block_gram(left, right, dimension=dimension)
    imaginary_norm = float(np.linalg.norm(matrix.imag))
    scale = max(1.0, float(np.linalg.norm(matrix.real)))
    if imaginary_norm > tolerance * scale:
        raise RuntimeError(
            "Hermitian operator projection developed a complex coefficient: "
            f"relative imaginary norm {imaginary_norm / scale:.3e}"
        )
    return np.asarray(matrix.real, dtype=float)


def _linear_combination(
    operators: Sequence[SparseOperator],
    coefficients: FloatArray,
) -> SparseOperator:
    if len(operators) != len(coefficients):
        raise ValueError("operator and coefficient counts must agree")
    if not operators:
        raise ValueError("cannot combine an empty operator block")
    result = csc_matrix(operators[0].shape, dtype=complex)
    for coefficient, operator in zip(coefficients, operators, strict=True):
        if coefficient != 0.0:
            result = result + float(coefficient) * operator
    result = result.tocsc()
    result.eliminate_zeros()
    return result


def _combine_block(
    operators: Sequence[SparseOperator],
    coefficients: FloatArray,
) -> OperatorBlock:
    matrix = np.asarray(coefficients, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(operators):
        raise ValueError("coefficient matrix has incompatible shape")
    return tuple(
        _linear_combination(operators, matrix[:, column])
        for column in range(matrix.shape[1])
    )


@dataclass(frozen=True)
class RawMomentBasis:
    """Whitened fixed Hilbert--Schmidt basis for the 29 raw moments."""

    phonon_cutoff: int
    hilbert_dimension: int
    observables: OperatorBlock
    identity_expectations: FloatArray
    centered_observables: OperatorBlock
    gram: FloatArray
    square_root: FloatArray
    inverse_square_root: FloatArray
    orthonormal_observables: OperatorBlock

    def raw_to_orthonormal(self, raw: FloatArray) -> FloatArray:
        values = _require_vector(raw, length=_RAW_DIMENSION, name="raw moments")
        return self.inverse_square_root @ (values - self.identity_expectations)

    def orthonormal_to_raw(self, coordinates: FloatArray) -> FloatArray:
        values = _require_vector(
            coordinates,
            length=_RAW_DIMENSION,
            name="orthonormal coordinates",
        )
        return self.identity_expectations + self.square_root @ values

    def contract_state(self, state_vector: ComplexArray) -> FloatArray:
        state = np.asarray(state_vector, dtype=complex)
        if state.shape != (self.hilbert_dimension,):
            raise ValueError("state vector has incompatible Hilbert dimension")
        return np.asarray(
            [
                np.vdot(state, operator @ state).real
                for operator in self.orthonormal_observables
            ],
            dtype=float,
        )


def _raw_observables(model: _ExactDimerModel) -> OperatorBlock:
    spin_average = tuple(
        0.5
        * (
            model.spin_pauli_observables[0][pauli_index]
            + model.spin_pauli_observables[1][pauli_index]
        )
        for pauli_index in range(4)
    )
    observables: list[SparseOperator] = list(spin_average[1:4])

    for annihilation in model.phonon_annihilation:
        observables.extend(
            (
                _hermitian_real_part(annihilation),
                _hermitian_imaginary_part(annihilation),
            )
        )

    observables.extend(
        (
            _hermitian_real_part(model.normal_phonon_observables[0][0]),
            _hermitian_real_part(model.normal_phonon_observables[1][1]),
            _hermitian_real_part(model.normal_phonon_observables[0][1]),
            _hermitian_imaginary_part(model.normal_phonon_observables[0][1]),
        )
    )
    for row, column in ((0, 0), (0, 1), (1, 1)):
        operator = model.anomalous_phonon_observables[row][column]
        observables.extend(
            (
                _hermitian_real_part(operator),
                _hermitian_imaginary_part(operator),
            )
        )

    for phonon_index in range(2):
        annihilation = model.phonon_annihilation[phonon_index]
        for pauli_operator in spin_average[1:4]:
            mixed = (annihilation @ pauli_operator).tocsc()
            observables.extend(
                (
                    _hermitian_real_part(mixed),
                    _hermitian_imaginary_part(mixed),
                )
            )

    if len(observables) != _RAW_DIMENSION:
        raise RuntimeError(
            f"expected {_RAW_DIMENSION} raw observables, built {len(observables)}"
        )
    return tuple(operator.tocsc() for operator in observables)


def build_raw_moment_basis(
    parameters: DimerParameters,
    *,
    phonon_cutoff: int,
) -> RawMomentBasis:
    """Build and whiten the symmetry-adapted raw-observable basis."""

    model = _build_exact_dimer_model(parameters, phonon_cutoff=phonon_cutoff)
    return _build_raw_moment_basis_from_model(
        model,
        phonon_cutoff=phonon_cutoff,
    )


def _build_raw_moment_basis_from_model(
    model: _ExactDimerModel,
    *,
    phonon_cutoff: int,
) -> RawMomentBasis:
    observables = _raw_observables(model)
    dimension = model.static_hamiltonian.shape[0]
    identity = eye(dimension, format="csc", dtype=complex)
    identity_expectations = np.asarray(
        [
            _hs_inner(identity, operator, dimension=dimension).real
            for operator in observables
        ],
        dtype=float,
    )
    centered = tuple(
        (operator - expectation * identity).tocsc()
        for operator, expectation in zip(
            observables,
            identity_expectations,
            strict=True,
        )
    )
    gram = _real_block_gram(centered, centered, dimension=dimension)
    eigenvalues, eigenvectors = eigh(0.5 * (gram + gram.T))
    if eigenvalues[0] <= 0.0:
        raise RuntimeError(
            "raw-moment Gram matrix is not positive definite: "
            f"minimum eigenvalue {eigenvalues[0]:.3e}"
        )
    square_root = (
        eigenvectors * np.sqrt(eigenvalues)[None, :]
    ) @ eigenvectors.T
    inverse_square_root = (
        eigenvectors * (1.0 / np.sqrt(eigenvalues))[None, :]
    ) @ eigenvectors.T
    orthonormal = _combine_block(centered, inverse_square_root)
    return RawMomentBasis(
        phonon_cutoff=phonon_cutoff,
        hilbert_dimension=dimension,
        observables=observables,
        identity_expectations=identity_expectations,
        centered_observables=centered,
        gram=gram,
        square_root=square_root,
        inverse_square_root=inverse_square_root,
        orthonormal_observables=orthonormal,
    )


def centered_jacobian_from_orthonormal(
    basis: RawMomentBasis,
    orthonormal_coordinates: FloatArray,
) -> FloatArray:
    """Return ``D chi_R(a)`` for the 31-slot centered representation."""

    raw = basis.orthonormal_to_raw(orthonormal_coordinates)
    return raw_to_closed_jacobian(raw) @ basis.square_root


def orthonormal_to_closed_coordinates(
    basis: RawMomentBasis,
    orthonormal_coordinates: FloatArray,
) -> FloatArray:
    """Reconstruct the centered state from whitened raw coordinates."""

    return raw_moments_to_closed_coordinates(
        basis.orthonormal_to_raw(orthonormal_coordinates)
    )


def closed_coordinates_to_orthonormal(
    basis: RawMomentBasis,
    closed: FloatArray,
) -> FloatArray:
    """Extract whitened raw coordinates from a physical centered state."""

    return basis.raw_to_orthonormal(closed_coordinates_to_raw_moments(closed))


def _density_liouvillian(
    hamiltonian: SparseOperator,
    operator: SparseOperator,
) -> SparseOperator:
    result = (-1j * (hamiltonian @ operator - operator @ hamiltonian)).tocsc()
    result.eliminate_zeros()
    return result


def _apply_liouvillian(
    hamiltonian: SparseOperator,
    operators: Sequence[SparseOperator],
) -> OperatorBlock:
    return tuple(_density_liouvillian(hamiltonian, operator) for operator in operators)


def _project_block_out(
    operators: Sequence[SparseOperator],
    basis: Sequence[SparseOperator],
    *,
    dimension: int,
    passes: int = 2,
) -> OperatorBlock:
    residual = tuple(operator.copy().tocsc() for operator in operators)
    if not basis:
        return residual
    for _ in range(passes):
        coefficients = _real_block_gram(
            basis,
            residual,
            dimension=dimension,
        )
        updated: list[SparseOperator] = []
        for column, operator in enumerate(residual):
            projection = _linear_combination(basis, coefficients[:, column])
            difference = (operator - projection).tocsc()
            difference.eliminate_zeros()
            updated.append(difference)
        residual = tuple(updated)
    return residual


def _compressed_real_vectorization(
    operators: Sequence[SparseOperator],
    *,
    dimension: int,
) -> FloatArray:
    if not operators:
        return np.empty((0, 0), dtype=float)
    complex_columns = hstack(
        [operator.reshape((dimension * dimension, 1)) for operator in operators],
        format="csc",
    )
    real_columns = vstack(
        (complex_columns.real, complex_columns.imag),
        format="csc",
    ) / sqrt(float(dimension))
    occupied_rows = np.flatnonzero(
        np.asarray(real_columns.getnnz(axis=1)).reshape(-1)
    )
    if occupied_rows.size == 0:
        return np.empty((0, len(operators)), dtype=float)
    return np.asarray(real_columns[occupied_rows, :].toarray(), dtype=float)


def _orthonormalize_block(
    candidates: Sequence[SparseOperator],
    against: Sequence[SparseOperator],
    *,
    dimension: int,
    rank_tolerance: float,
) -> tuple[OperatorBlock, FloatArray]:
    residual = _project_block_out(
        candidates,
        against,
        dimension=dimension,
        passes=2,
    )
    vectors = _compressed_real_vectorization(residual, dimension=dimension)
    if vectors.size == 0:
        return (), np.empty(0, dtype=float)
    _, singular_values, right_adjoint = svd(
        vectors,
        full_matrices=False,
        check_finite=False,
        lapack_driver="gesdd",
    )
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return (), singular_values
    candidate_scale = max(
        1.0,
        _operator_block_norm(candidates, dimension=dimension),
    )
    if singular_values[0] <= rank_tolerance * candidate_scale:
        return (), np.empty(0, dtype=float)
    keep = singular_values / singular_values[0] >= rank_tolerance
    retained_singular_values = singular_values[keep]
    if retained_singular_values.size == 0:
        return (), retained_singular_values
    transformation = (
        right_adjoint.T[:, keep] / retained_singular_values[None, :]
    )
    block = _combine_block(residual, transformation)

    # Remove roundoff from the sparse combinations without changing rank.
    overlap = _real_block_gram(block, block, dimension=dimension)
    eigenvalues, eigenvectors = eigh(0.5 * (overlap + overlap.T))
    if eigenvalues[0] <= 0.0:
        raise RuntimeError("orthonormal block lost positive rank after SVD")
    inverse_square_root = (
        eigenvectors * (1.0 / np.sqrt(eigenvalues))[None, :]
    ) @ eigenvectors.T
    block = _combine_block(block, inverse_square_root)
    return block, np.asarray(retained_singular_values, dtype=float)


def _operator_block_norm(
    operators: Sequence[SparseOperator],
    *,
    dimension: int,
) -> float:
    if not operators:
        return 0.0
    gram = _real_block_gram(operators, operators, dimension=dimension)
    return sqrt(max(0.0, float(np.trace(gram))))


def _skew_projection(matrix: FloatArray) -> tuple[FloatArray, float]:
    values = np.asarray(matrix, dtype=float)
    skew = 0.5 * (values - values.T)
    symmetric = 0.5 * (values + values.T)
    scale = max(float(np.linalg.norm(values)), np.finfo(float).tiny)
    return skew, float(np.linalg.norm(symmetric) / scale)


@dataclass(frozen=True)
class KrylovClosureCoefficients:
    """Online-safe arrays for one fixed order of the projected closure."""

    phonon_cutoff: int
    order: int
    shell_dimensions: tuple[int, ...]
    raw_basis: RawMomentBasis
    retained_static: FloatArray
    retained_drive: FloatArray
    retained_to_auxiliary: FloatArray
    auxiliary_static: FloatArray
    auxiliary_drive: FloatArray
    auxiliary_observables: OperatorBlock
    static_residual_observables: OperatorBlock
    drive_residual_observables: OperatorBlock
    static_residual_gram: FloatArray
    drive_residual_gram: FloatArray
    cross_residual_gram: FloatArray
    symmetric_leakage: dict[str, float]

    @property
    def auxiliary_dimension(self) -> int:
        return len(self.auxiliary_observables)

    def orthonormal_velocity(
        self,
        orthonormal_coordinates: FloatArray,
        auxiliary_coordinates: FloatArray,
        *,
        drive_value: float,
    ) -> tuple[FloatArray, FloatArray]:
        retained = _require_vector(
            orthonormal_coordinates,
            length=_RAW_DIMENSION,
            name="orthonormal coordinates",
        )
        auxiliary = _require_vector(
            auxiliary_coordinates,
            length=self.auxiliary_dimension,
            name="auxiliary coordinates",
        )
        retained_generator = self.retained_static + drive_value * self.retained_drive
        auxiliary_generator = self.auxiliary_static + drive_value * self.auxiliary_drive
        retained_velocity = (
            retained_generator @ retained
            - self.retained_to_auxiliary.T @ auxiliary
        )
        auxiliary_velocity = (
            self.retained_to_auxiliary @ retained
            + auxiliary_generator @ auxiliary
        )
        return retained_velocity, auxiliary_velocity

    def centered_velocity(
        self,
        closed_coordinates: FloatArray,
        auxiliary_coordinates: FloatArray,
        *,
        drive_value: float,
    ) -> tuple[FloatArray, FloatArray]:
        retained = closed_coordinates_to_orthonormal(
            self.raw_basis,
            closed_coordinates,
        )
        retained_velocity, auxiliary_velocity = self.orthonormal_velocity(
            retained,
            auxiliary_coordinates,
            drive_value=drive_value,
        )
        centered_velocity = (
            centered_jacobian_from_orthonormal(self.raw_basis, retained)
            @ retained_velocity
        )
        return centered_velocity, auxiliary_velocity

    def contract_auxiliary_state(self, state_vector: ComplexArray) -> FloatArray:
        state = np.asarray(state_vector, dtype=complex)
        if state.shape != (self.raw_basis.hilbert_dimension,):
            raise ValueError("state vector has incompatible Hilbert dimension")
        return np.asarray(
            [
                np.vdot(state, operator @ state).real
                for operator in self.auxiliary_observables
            ],
            dtype=float,
        )

    def residual_norms(
        self,
        auxiliary_coordinates: FloatArray,
        *,
        drive_value: float,
    ) -> tuple[float, float, float]:
        auxiliary = _require_vector(
            auxiliary_coordinates,
            length=self.auxiliary_dimension,
            name="auxiliary coordinates",
        )
        static_squared = float(auxiliary @ self.static_residual_gram @ auxiliary)
        drive_squared = float(auxiliary @ self.drive_residual_gram @ auxiliary)
        cross = float(auxiliary @ self.cross_residual_gram @ auxiliary)
        total_squared = (
            static_squared
            + 2.0 * drive_value * cross
            + drive_value**2 * drive_squared
        )
        return (
            sqrt(max(0.0, static_squared)),
            abs(drive_value) * sqrt(max(0.0, drive_squared)),
            sqrt(max(0.0, total_squared)),
        )


@dataclass(frozen=True)
class KrylovClosureConstruction:
    """Offline operator construction from which fixed-order arrays are cut."""

    phonon_cutoff: int
    hilbert_dimension: int
    rank_tolerance: float
    raw_basis: RawMomentBasis
    retained_static_raw: FloatArray
    retained_drive_raw: FloatArray
    retained_static: FloatArray
    retained_drive: FloatArray
    static_force: OperatorBlock
    drive_force: OperatorBlock
    force_singular_values: FloatArray
    shells: tuple[OperatorBlock, ...]
    shell_singular_values: tuple[FloatArray, ...]
    static_hamiltonian: SparseOperator
    drive_hamiltonian: SparseOperator
    retained_symmetric_leakage: dict[str, float]

    @property
    def force_rank(self) -> int:
        return len(self.shells[0]) if self.shells else 0

    @property
    def shell_dimensions(self) -> tuple[int, ...]:
        return tuple(len(shell) for shell in self.shells)

    def coefficients(self, order: int) -> KrylovClosureCoefficients:
        if order <= 0:
            raise ValueError("order must be positive")
        if order > len(self.shells):
            raise ValueError(
                f"order {order} exceeds the {len(self.shells)} constructed shells"
            )
        auxiliary = tuple(
            operator for shell in self.shells[:order] for operator in shell
        )
        dimension = self.hilbert_dimension
        coupling = _real_block_gram(
            auxiliary,
            self.static_force,
            dimension=dimension,
        )
        static_action = _apply_liouvillian(self.static_hamiltonian, auxiliary)
        drive_action = _apply_liouvillian(self.drive_hamiltonian, auxiliary)
        static_raw = _real_block_gram(
            auxiliary,
            static_action,
            dimension=dimension,
        )
        drive_raw = _real_block_gram(
            auxiliary,
            drive_action,
            dimension=dimension,
        )
        static, static_leakage = _skew_projection(static_raw)
        drive, drive_leakage = _skew_projection(drive_raw)

        retained_basis = self.raw_basis.orthonormal_observables
        static_residual = _project_block_out(
            static_action,
            (*retained_basis, *auxiliary),
            dimension=dimension,
            passes=2,
        )
        drive_residual = _project_block_out(
            drive_action,
            (*retained_basis, *auxiliary),
            dimension=dimension,
            passes=2,
        )
        static_residual_gram = _real_block_gram(
            static_residual,
            static_residual,
            dimension=dimension,
        )
        drive_residual_gram = _real_block_gram(
            drive_residual,
            drive_residual,
            dimension=dimension,
        )
        cross_residual_gram = _real_block_gram(
            static_residual,
            drive_residual,
            dimension=dimension,
        )
        return KrylovClosureCoefficients(
            phonon_cutoff=self.phonon_cutoff,
            order=order,
            shell_dimensions=tuple(len(shell) for shell in self.shells[:order]),
            raw_basis=self.raw_basis,
            retained_static=self.retained_static,
            retained_drive=self.retained_drive,
            retained_to_auxiliary=coupling,
            auxiliary_static=static,
            auxiliary_drive=drive,
            auxiliary_observables=auxiliary,
            static_residual_observables=static_residual,
            drive_residual_observables=drive_residual,
            static_residual_gram=static_residual_gram,
            drive_residual_gram=drive_residual_gram,
            cross_residual_gram=cross_residual_gram,
            symmetric_leakage={
                **self.retained_symmetric_leakage,
                "auxiliary_static": static_leakage,
                "auxiliary_drive": drive_leakage,
            },
        )


def build_krylov_closure_construction(
    parameters: DimerParameters,
    *,
    phonon_cutoff: int,
    shell_count: int = 5,
    rank_tolerance: float = 1e-12,
) -> KrylovClosureConstruction:
    """Construct retained and orthogonal block-Krylov operator spaces."""

    if shell_count <= 0:
        raise ValueError("shell_count must be positive")
    if not 0.0 < rank_tolerance < 1.0:
        raise ValueError("rank_tolerance must lie between zero and one")
    model = _build_exact_dimer_model(parameters, phonon_cutoff=phonon_cutoff)
    raw_basis = _build_raw_moment_basis_from_model(
        model,
        phonon_cutoff=phonon_cutoff,
    )
    retained_basis = raw_basis.orthonormal_observables
    dimension = raw_basis.hilbert_dimension
    static_action = _apply_liouvillian(model.static_hamiltonian, retained_basis)
    drive_action = _apply_liouvillian(model.drive_operator, retained_basis)
    retained_static_raw = _real_block_gram(
        retained_basis,
        static_action,
        dimension=dimension,
    )
    retained_drive_raw = _real_block_gram(
        retained_basis,
        drive_action,
        dimension=dimension,
    )
    retained_static, retained_static_leakage = _skew_projection(
        retained_static_raw
    )
    retained_drive, retained_drive_leakage = _skew_projection(
        retained_drive_raw
    )
    static_force = _project_block_out(
        static_action,
        retained_basis,
        dimension=dimension,
        passes=2,
    )
    drive_force = _project_block_out(
        drive_action,
        retained_basis,
        dimension=dimension,
        passes=2,
    )

    first_shell, force_singular_values = _orthonormalize_block(
        static_force,
        retained_basis,
        dimension=dimension,
        rank_tolerance=rank_tolerance,
    )
    shells: list[OperatorBlock] = []
    singular_values: list[FloatArray] = []
    if first_shell:
        shells.append(first_shell)
        singular_values.append(force_singular_values)
    while shells and len(shells) < shell_count:
        candidates = _apply_liouvillian(model.static_hamiltonian, shells[-1])
        against = tuple(
            [*retained_basis]
            + [operator for shell in shells for operator in shell]
        )
        shell, shell_values = _orthonormalize_block(
            candidates,
            against,
            dimension=dimension,
            rank_tolerance=rank_tolerance,
        )
        if not shell:
            break
        shells.append(shell)
        singular_values.append(shell_values)

    return KrylovClosureConstruction(
        phonon_cutoff=phonon_cutoff,
        hilbert_dimension=dimension,
        rank_tolerance=rank_tolerance,
        raw_basis=raw_basis,
        retained_static_raw=retained_static_raw,
        retained_drive_raw=retained_drive_raw,
        retained_static=retained_static,
        retained_drive=retained_drive,
        static_force=static_force,
        drive_force=drive_force,
        force_singular_values=force_singular_values,
        shells=tuple(shells),
        shell_singular_values=tuple(singular_values),
        static_hamiltonian=model.static_hamiltonian,
        drive_hamiltonian=model.drive_operator,
        retained_symmetric_leakage={
            "retained_static": retained_static_leakage,
            "retained_drive": retained_drive_leakage,
        },
    )
