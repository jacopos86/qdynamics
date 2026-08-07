"""Finite-cutoff witness against an exact instantaneous 31-moment closure.

The diagnostic constructs two full-rank, spin-exchange-symmetric density
operators with identical raw expectations for every observable underlying
(rho, B, N, A, C). It additionally forces the exact rho, B, N, and A
velocities to agree. A component of the exact C velocity is then projected
onto the orthogonal complement of those constraints. A nonzero residual
supplies a Hermitian perturbation that separates the exact C velocities
without changing the retained 31-coordinate state.

try: higher g. 1.5,2,; try without and with.

This is a finite-cutoff counterexample to a globally exact single-valued
instantaneous closure. It does not rule out an exact closure on a narrower
state manifold or an autonomous closure augmented by memory variables.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .exact_reference import (
    _ExactDimerModel,
    _build_exact_dimer_model,
    _ground_state,
)
from .hubbard_dimer import DimerParameters
from .matrix_reference import (
    MatrixDimerState,
    electron_phonon_moment_matrix,
    matrix_state_to_closed_scalar_coordinates,
)

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
TargetComponent = Literal["real", "imag"]


@dataclass(frozen=True)
class ClosureIdentifiabilityWitness:
    """Two physical states with one retained state and two exact velocities."""

    phonon_cutoff: int
    hilbert_dimension: int
    time: float
    reference_identity_weight: float
    constraint_count: int
    constraint_rank: int
    target_phonon: int
    target_row: int
    target_column: int
    target_component: TargetComponent
    target_residual_norm: float
    target_relative_residual: float
    perturbation_scale: float
    perturbation: ComplexArray
    density_plus: ComplexArray
    density_minus: ComplexArray
    coordinates_plus: FloatArray
    coordinates_minus: FloatArray
    derivatives_plus: FloatArray
    derivatives_minus: FloatArray
    minimum_density_eigenvalue: float
    minimum_joint_gram_eigenvalue: float
    spin_swap_residual: float
    maximum_constraint_overlap: float

    @property
    def coordinate_difference(self) -> FloatArray:
        """Return the difference between the two retained states."""

        return self.coordinates_plus - self.coordinates_minus

    @property
    def derivative_difference(self) -> FloatArray:
        """Return the difference between the two exact 31-coordinate rates."""

        return self.derivatives_plus - self.derivatives_minus

    @property
    def maximum_coordinate_difference(self) -> float:
        """Return the largest retained-coordinate mismatch."""

        return float(np.max(np.abs(self.coordinate_difference)))

    @property
    def lower_derivative_difference_norm(self) -> float:
        """Return the combined rho, B, N, and A velocity mismatch."""

        return float(np.linalg.norm(self.derivative_difference[:17]))

    @property
    def correlation_derivative_difference_norm(self) -> float:
        """Return the Euclidean difference in the fourteen C rates."""

        return float(np.linalg.norm(self.derivative_difference[17:]))

    @property
    def maximum_correlation_derivative_difference(self) -> float:
        """Return the largest componentwise difference in the C rate."""

        return float(np.max(np.abs(self.derivative_difference[17:])))

    def summary(self) -> dict[str, float | int | str]:
        """Return a compact JSON-serializable certificate."""

        return {
            "phonon_cutoff": self.phonon_cutoff,
            "hilbert_dimension": self.hilbert_dimension,
            "time": self.time,
            "reference_identity_weight": self.reference_identity_weight,
            "constraint_count": self.constraint_count,
            "constraint_rank": self.constraint_rank,
            "target": (
                f"C[{self.target_phonon},{self.target_row},"
                f"{self.target_column}].{self.target_component}"
            ),
            "target_residual_norm": self.target_residual_norm,
            "target_relative_residual": self.target_relative_residual,
            "perturbation_scale": self.perturbation_scale,
            "minimum_density_eigenvalue": self.minimum_density_eigenvalue,
            "minimum_joint_gram_eigenvalue": (
                self.minimum_joint_gram_eigenvalue
            ),
            "spin_swap_residual": self.spin_swap_residual,
            "maximum_constraint_overlap": self.maximum_constraint_overlap,
            "maximum_coordinate_difference": (
                self.maximum_coordinate_difference
            ),
            "lower_derivative_difference_norm": (
                self.lower_derivative_difference_norm
            ),
            "correlation_derivative_difference_norm": (
                self.correlation_derivative_difference_norm
            ),
            "maximum_correlation_derivative_difference": (
                self.maximum_correlation_derivative_difference
            ),
        }


def _matrix_units(
    model: _ExactDimerModel,
    spin: int,
) -> tuple[tuple[ComplexArray, ComplexArray], ...]:
    identity, sigma_x, sigma_y, sigma_z = (
        operator.toarray()
        for operator in model.spin_pauli_observables[spin]
    )
    return (
        (
            0.5 * (identity + sigma_z),
            0.5 * (sigma_x - 1j * sigma_y),
        ),
        (
            0.5 * (sigma_x + 1j * sigma_y),
            0.5 * (identity - sigma_z),
        ),
    )


def _hermitian_components(
    operator: ComplexArray,
) -> tuple[tuple[TargetComponent, ComplexArray], ...]:
    parts: list[tuple[TargetComponent, ComplexArray]] = []
    candidates: tuple[tuple[TargetComponent, ComplexArray], ...] = (
        ("real", 0.5 * (operator + operator.conjugate().T)),
        ("imag", (operator - operator.conjugate().T) / (2j)),
    )
    for name, matrix in candidates:
        hermitian = 0.5 * (matrix + matrix.conjugate().T)
        if np.linalg.norm(hermitian) > 1e-12:
            parts.append((name, hermitian))
    return tuple(parts)


def _heisenberg_derivative(
    hamiltonian: ComplexArray,
    operator: ComplexArray,
) -> ComplexArray:
    return 1j * (hamiltonian @ operator - operator @ hamiltonian)


def _real_vector(matrix: ComplexArray) -> FloatArray:
    return np.concatenate((matrix.real.ravel(), matrix.imag.ravel()))


def _matrix_from_real_vector(
    vector: FloatArray,
    dimension: int,
) -> ComplexArray:
    entry_count = dimension * dimension
    matrix = (
        vector[:entry_count] + 1j * vector[entry_count:]
    ).reshape(dimension, dimension)
    return 0.5 * (matrix + matrix.conjugate().T)


def _orthogonal_projection_basis(
    operators: tuple[ComplexArray, ...],
    *,
    rank_tolerance: float,
) -> tuple[FloatArray, int]:
    vectors = np.column_stack(
        [_real_vector(operator) for operator in operators]
    )
    left_vectors, singular_values, _ = np.linalg.svd(
        vectors,
        full_matrices=False,
    )
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        raise RuntimeError("the retained-observable constraint span is empty")
    rank = int(
        np.sum(singular_values > rank_tolerance * singular_values[0])
    )
    return np.asarray(left_vectors[:, :rank], dtype=float), rank


def _project_to_complement(
    matrix: ComplexArray,
    basis: FloatArray,
) -> ComplexArray:
    vector = _real_vector(matrix)
    residual = vector - basis @ (basis.T @ vector)
    return _matrix_from_real_vector(residual, matrix.shape[0])


def _spin_swap(dimension: int) -> ComplexArray:
    if dimension % 4:
        raise ValueError("the fixed-sector Hilbert dimension must divide by 4")
    electron_swap = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=complex,
    )
    return np.kron(electron_swap, np.eye(dimension // 4, dtype=complex))


def _expectation(
    density: ComplexArray,
    operator: ComplexArray,
) -> complex:
    return complex(np.trace(density @ operator))


def _contract_density_state(
    model: _ExactDimerModel,
    density: ComplexArray,
    electron_units: tuple[tuple[ComplexArray, ComplexArray], ...],
) -> MatrixDimerState:
    annihilation = tuple(
        operator.toarray() for operator in model.phonon_annihilation
    )
    coherent = np.array(
        [_expectation(density, operator) for operator in annihilation],
        dtype=complex,
    )
    electron = np.array(
        [
            [
                _expectation(density, electron_units[row][column])
                for column in range(2)
            ]
            for row in range(2)
        ],
        dtype=complex,
    )
    normal = np.empty((2, 2), dtype=complex)
    anomalous = np.empty((2, 2), dtype=complex)
    for q in range(2):
        for r in range(2):
            normal[q, r] = (
                _expectation(
                    density,
                    model.normal_phonon_observables[q][r].toarray(),
                )
                - coherent[q] * coherent[r].conjugate()
            )
            anomalous[r, q] = (
                _expectation(
                    density,
                    model.anomalous_phonon_observables[r][q].toarray(),
                )
                - coherent[r] * coherent[q]
            )
    correlation = np.empty((2, 2, 2), dtype=complex)
    for q in range(2):
        for row in range(2):
            for column in range(2):
                mixed_operator = (
                    annihilation[q] @ electron_units[row][column]
                )
                correlation[q, row, column] = (
                    _expectation(density, mixed_operator)
                    - electron[row, column] * coherent[q]
                )
    return MatrixDimerState(
        electron_density=electron,
        coherent_phonon=coherent,
        phonon_density=normal,
        anomalous_phonon_density=anomalous,
        electron_phonon_correlation=correlation,
    )


def _contract_density_derivative(
    model: _ExactDimerModel,
    density: ComplexArray,
    hamiltonian: ComplexArray,
    electron_units: tuple[tuple[ComplexArray, ComplexArray], ...],
) -> MatrixDimerState:
    state = _contract_density_state(model, density, electron_units)
    density_derivative = -1j * (
        hamiltonian @ density - density @ hamiltonian
    )
    electron_derivative = np.array(
        [
            [
                _expectation(
                    density_derivative,
                    electron_units[row][column],
                )
                for column in range(2)
            ]
            for row in range(2)
        ],
        dtype=complex,
    )
    annihilation = tuple(
        operator.toarray() for operator in model.phonon_annihilation
    )
    coherent_derivative = np.array(
        [
            _expectation(density_derivative, operator)
            for operator in annihilation
        ],
        dtype=complex,
    )
    coherent = state.coherent_phonon
    normal_derivative = np.empty((2, 2), dtype=complex)
    anomalous_derivative = np.empty((2, 2), dtype=complex)
    for q in range(2):
        for r in range(2):
            normal_derivative[q, r] = (
                _expectation(
                    density_derivative,
                    model.normal_phonon_observables[q][r].toarray(),
                )
                - coherent_derivative[q] * coherent[r].conjugate()
                - coherent[q] * coherent_derivative[r].conjugate()
            )
            anomalous_derivative[r, q] = (
                _expectation(
                    density_derivative,
                    model.anomalous_phonon_observables[r][q].toarray(),
                )
                - coherent_derivative[r] * coherent[q]
                - coherent[r] * coherent_derivative[q]
            )
    correlation_derivative = np.empty((2, 2, 2), dtype=complex)
    electron = state.electron_density
    for q in range(2):
        for row in range(2):
            for column in range(2):
                mixed_operator = (
                    annihilation[q] @ electron_units[row][column]
                )
                correlation_derivative[q, row, column] = (
                    _expectation(density_derivative, mixed_operator)
                    - electron_derivative[row, column] * coherent[q]
                    - electron[row, column] * coherent_derivative[q]
                )
    return MatrixDimerState(
        electron_density=electron_derivative,
        coherent_phonon=coherent_derivative,
        phonon_density=normal_derivative,
        anomalous_phonon_density=anomalous_derivative,
        electron_phonon_correlation=correlation_derivative,
    )


def closure_identifiability_witness(
    parameters: DimerParameters,
    *,
    phonon_cutoff: int = 3,
    time: float = 0.0,
    reference_identity_weight: float = 0.1,
    positivity_fraction: float = 0.45,
    rank_tolerance: float = 1e-10,
) -> ClosureIdentifiabilityWitness:
    """Construct a physical counterexample to exact instantaneous closure.

    The reference is (1-mu)|ground><ground| + mu I/d. The identity component
    makes the state interior to the positive cone, so both signs of a
    retained-null perturbation remain physical. The witness does not assert
    that the exact driven trajectory visits either constructed state.
    """

    if phonon_cutoff < 1:
        raise ValueError("phonon_cutoff must be at least one")
    if not 0.0 < reference_identity_weight <= 1.0:
        raise ValueError("reference_identity_weight must lie in (0, 1]")
    if not 0.0 < positivity_fraction < 1.0:
        raise ValueError("positivity_fraction must lie in (0, 1)")
    if rank_tolerance <= 0.0:
        raise ValueError("rank_tolerance must be positive")

    model = _build_exact_dimer_model(
        parameters,
        phonon_cutoff=phonon_cutoff,
    )
    hamiltonian = (
        model.static_hamiltonian
        + parameters.drive_difference(time) * model.drive_operator
    ).toarray()
    dimension = hamiltonian.shape[0]
    identity = np.eye(dimension, dtype=complex)
    spin_swap = _spin_swap(dimension)
    electron_units = (_matrix_units(model, 0), _matrix_units(model, 1))
    annihilation = tuple(
        operator.toarray() for operator in model.phonon_annihilation
    )
    normal = tuple(
        tuple(operator.toarray() for operator in row)
        for row in model.normal_phonon_observables
    )
    anomalous = tuple(
        tuple(operator.toarray() for operator in row)
        for row in model.anomalous_phonon_observables
    )

    lower_raw: list[ComplexArray] = []
    all_raw: list[ComplexArray] = [identity]
    for spin_units in electron_units:
        for row in spin_units:
            for operator in row:
                lower_raw.append(operator)
                all_raw.append(operator)
    for operator in annihilation:
        lower_raw.append(operator)
        all_raw.append(operator)
    for matrices in (normal, anomalous):
        for row in matrices:
            for operator in row:
                lower_raw.append(operator)
                all_raw.append(operator)

    mixed_operators: list[tuple] = []
    for spin_units in electron_units:
        spin_mixed = tuple(
            tuple(
                tuple(
                    annihilation[q] @ spin_units[row][column]
                    for column in range(2)
                )
                for row in range(2)
            )
            for q in range(2)
        )
        mixed_operators.append(spin_mixed)
        for mode in spin_mixed:
            for row in mode:
                all_raw.extend(row)

    constraints: list[ComplexArray] = []
    for operator in all_raw:
        constraints.extend(
            matrix for _, matrix in _hermitian_components(operator)
        )
    for operator in lower_raw:
        constraints.extend(
            matrix
            for _, matrix in _hermitian_components(
                _heisenberg_derivative(hamiltonian, operator)
            )
        )
    constraint_basis, constraint_rank = _orthogonal_projection_basis(
        tuple(constraints),
        rank_tolerance=rank_tolerance,
    )

    _, ground_state = _ground_state(
        model,
        eigensolver_tolerance=1e-12,
    )
    ground_density = np.outer(ground_state, ground_state.conjugate())
    ground_density = 0.5 * (
        ground_density + spin_swap @ ground_density @ spin_swap
    )
    mu = reference_identity_weight
    reference_density = (
        (1.0 - mu) * ground_density + mu * identity / dimension
    )
    reference_state = _contract_density_state(
        model,
        reference_density,
        electron_units[0],
    )

    candidates: list[tuple] = []
    for q in range(2):
        annihilation_derivative = _heisenberg_derivative(
            hamiltonian,
            annihilation[q],
        )
        for row in range(2):
            for column in range(2):
                centered_targets: list[ComplexArray] = []
                for spin in range(2):
                    electron_derivative = _heisenberg_derivative(
                        hamiltonian,
                        electron_units[spin][row][column],
                    )
                    centered_targets.append(
                        _heisenberg_derivative(
                            hamiltonian,
                            mixed_operators[spin][q][row][column],
                        )
                        - reference_state.electron_density[row, column]
                        * annihilation_derivative
                        - reference_state.coherent_phonon[q]
                        * electron_derivative
                    )
                spin_symmetric_target = 0.5 * (
                    centered_targets[0] + centered_targets[1]
                )
                for component, target in _hermitian_components(
                    spin_symmetric_target
                ):
                    residual = _project_to_complement(
                        target,
                        constraint_basis,
                    )
                    residual = 0.5 * (
                        residual + spin_swap @ residual @ spin_swap
                    )
                    residual = _project_to_complement(
                        residual,
                        constraint_basis,
                    )
                    candidates.append(
                        (
                            float(np.linalg.norm(residual)),
                            float(np.linalg.norm(target)),
                            q,
                            row,
                            column,
                            component,
                            residual,
                        )
                    )
    if not candidates:
        raise RuntimeError("no nonzero correlation-velocity target was found")
    (
        residual_norm,
        target_norm,
        target_q,
        target_row,
        target_column,
        target_component,
        perturbation,
    ) = max(candidates, key=lambda candidate: candidate[0])
    if residual_norm <= rank_tolerance * max(1.0, target_norm):
        raise RuntimeError(
            "the exact C velocity lies in the tested retained constraint span"
        )
    perturbation = perturbation / residual_norm

    operator_norm = float(
        np.max(np.abs(np.linalg.eigvalsh(perturbation)))
    )
    minimum_reference_eigenvalue = float(
        np.min(np.linalg.eigvalsh(reference_density))
    )
    perturbation_scale = (
        positivity_fraction * minimum_reference_eigenvalue / operator_norm
    )
    density_plus = reference_density + perturbation_scale * perturbation
    density_minus = reference_density - perturbation_scale * perturbation

    state_plus = _contract_density_state(
        model,
        density_plus,
        electron_units[0],
    )
    state_minus = _contract_density_state(
        model,
        density_minus,
        electron_units[0],
    )
    derivative_plus = _contract_density_derivative(
        model,
        density_plus,
        hamiltonian,
        electron_units[0],
    )
    derivative_minus = _contract_density_derivative(
        model,
        density_minus,
        hamiltonian,
        electron_units[0],
    )
    coordinates_plus = matrix_state_to_closed_scalar_coordinates(state_plus)
    coordinates_minus = matrix_state_to_closed_scalar_coordinates(state_minus)
    derivatives_plus = matrix_state_to_closed_scalar_coordinates(
        derivative_plus
    )
    derivatives_minus = matrix_state_to_closed_scalar_coordinates(
        derivative_minus
    )
    minimum_density_eigenvalue = min(
        float(np.min(np.linalg.eigvalsh(density_plus))),
        float(np.min(np.linalg.eigvalsh(density_minus))),
    )
    minimum_joint_gram_eigenvalue = min(
        float(
            np.min(
                np.linalg.eigvalsh(
                    electron_phonon_moment_matrix(state_plus)
                )
            )
        ),
        float(
            np.min(
                np.linalg.eigvalsh(
                    electron_phonon_moment_matrix(state_minus)
                )
            )
        ),
    )
    maximum_constraint_overlap = float(
        np.max(
            np.abs(
                constraint_basis.T @ _real_vector(perturbation)
            )
        )
    )

    return ClosureIdentifiabilityWitness(
        phonon_cutoff=phonon_cutoff,
        hilbert_dimension=dimension,
        time=float(time),
        reference_identity_weight=reference_identity_weight,
        constraint_count=len(constraints),
        constraint_rank=constraint_rank,
        target_phonon=target_q,
        target_row=target_row,
        target_column=target_column,
        target_component=target_component,
        target_residual_norm=residual_norm,
        target_relative_residual=residual_norm / target_norm,
        perturbation_scale=perturbation_scale,
        perturbation=np.asarray(perturbation, dtype=complex),
        density_plus=np.asarray(density_plus, dtype=complex),
        density_minus=np.asarray(density_minus, dtype=complex),
        coordinates_plus=np.asarray(coordinates_plus, dtype=float),
        coordinates_minus=np.asarray(coordinates_minus, dtype=float),
        derivatives_plus=np.asarray(derivatives_plus, dtype=float),
        derivatives_minus=np.asarray(derivatives_minus, dtype=float),
        minimum_density_eigenvalue=minimum_density_eigenvalue,
        minimum_joint_gram_eigenvalue=minimum_joint_gram_eigenvalue,
        spin_swap_residual=float(
            np.linalg.norm(
                perturbation - spin_swap @ perturbation @ spin_swap
            )
        ),
        maximum_constraint_overlap=maximum_constraint_overlap,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phonon-cutoff", type=int, default=3)
    parser.add_argument("--reference-identity-weight", type=float, default=0.1)
    return parser


def main() -> int:
    args = _parser().parse_args()
    witness = closure_identifiability_witness(
        DimerParameters(
            lambda_ep=1.5,
            gamma=0.5,
            drive_amplitude=1.0,
        ),
        phonon_cutoff=args.phonon_cutoff,
        reference_identity_weight=args.reference_identity_weight,
    )
    print(json.dumps(witness.summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
