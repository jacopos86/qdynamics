"""Preparation-conditioned unitary residual-Galerkin reduction.

The reduced state is a ket in a deterministic Hilbert-space trial basis.  The
online model contains only compressed Hamiltonians, compressed raw-moment
operators, and the reduced initial ket.  Full-cutoff operators and the exact
ground-state preparation remain in :class:`PurgProjection`, which is an
offline construction and certification object.

This module deliberately does not use exact driven trajectories.  They are
reserved for later, separately gated scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh, qr
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import eigsh, splu

from .exact_reference import _build_exact_dimer_model
from .hubbard_dimer import DimerParameters
from .krylov_memory_closure import (
    RAW_MOMENT_NAMES,
    RawMomentBasis,
    _operator_block_norm,
    _build_raw_moment_basis_from_model,
    build_krylov_closure_construction,
    closed_coordinates_to_raw_moments,
    raw_moments_to_closed_coordinates,
    raw_to_closed_jacobian,
    raw_velocity_to_closed_velocity,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
SparseOperator = csc_matrix

_RAW_DIMENSION = len(RAW_MOMENT_NAMES)
_CENTERED_BLOCKS: dict[str, slice] = {
    "rho": slice(0, 3),
    "B": slice(3, 7),
    "N": slice(7, 11),
    "A": slice(11, 17),
    "C": slice(17, 31),
}


def _require_complex_vector(
    values: ComplexArray,
    *,
    length: int,
    name: str,
) -> ComplexArray:
    array = np.asarray(values, dtype=complex)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape {(length,)}, got {array.shape}")
    return array


def _normalize(values: ComplexArray, *, name: str) -> ComplexArray:
    vector = np.asarray(values, dtype=complex)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must have nonzero norm")
    return vector / norm


def _fix_vector_phase(vector: ComplexArray) -> ComplexArray:
    result = np.asarray(vector, dtype=complex).copy()
    if result.size == 0:
        return result
    pivot = int(np.argmax(np.abs(result)))
    magnitude = abs(result[pivot])
    if magnitude > 0.0:
        result /= result[pivot] / magnitude
    result[pivot] = abs(result[pivot]) + 0.0j
    return result


def _phase_fix_columns(matrix: ComplexArray) -> ComplexArray:
    result = np.asarray(matrix, dtype=complex).copy()
    for column in range(result.shape[1]):
        result[:, column] = _fix_vector_phase(result[:, column])
    return result


def _project_out(
    candidates: ComplexArray,
    basis: ComplexArray,
    *,
    passes: int = 2,
) -> ComplexArray:
    residual = np.asarray(candidates, dtype=complex).copy()
    if residual.ndim == 1:
        residual = residual[:, None]
    if basis.size == 0:
        return residual
    for _ in range(passes):
        residual -= basis @ (basis.conj().T @ residual)
    return residual


@dataclass(frozen=True)
class RrqrAppendResult:
    """Independent columns accepted from one deterministic RRQR packet."""

    basis: ComplexArray
    accepted: int
    deflated: int
    truncated: int
    pivot_order: tuple[int, ...]
    diagonal: FloatArray


def _rrqr_append(
    basis: ComplexArray,
    candidates: ComplexArray,
    *,
    relative_tolerance: float,
    maximum_new_columns: int | None = None,
) -> RrqrAppendResult:
    if relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive")
    current = np.asarray(basis, dtype=complex)
    packet = np.asarray(candidates, dtype=complex)
    if current.ndim != 2 or packet.ndim != 2:
        raise ValueError("basis and candidates must be matrices")
    if current.shape[0] != packet.shape[0]:
        raise ValueError("basis and candidates have different row dimensions")
    if packet.shape[1] == 0:
        return RrqrAppendResult(
            basis=current.copy(),
            accepted=0,
            deflated=0,
            truncated=0,
            pivot_order=(),
            diagonal=np.zeros(0, dtype=float),
        )

    residual = _project_out(packet, current, passes=2)
    q_columns, triangular, pivots = qr(
        residual,
        mode="economic",
        pivoting=True,
        check_finite=True,
    )
    diagonal = np.abs(np.diag(triangular)).astype(float)
    reference = float(diagonal[0]) if diagonal.size else 0.0
    if reference <= 0.0:
        numerical_rank = 0
    else:
        numerical_rank = int(
            np.count_nonzero(diagonal > relative_tolerance * reference)
        )
    rank = numerical_rank
    if maximum_new_columns is not None:
        if maximum_new_columns < 0:
            raise ValueError("maximum_new_columns must be nonnegative")
        rank = min(rank, maximum_new_columns)

    accepted = _project_out(q_columns[:, :rank], current, passes=2)
    if rank:
        accepted, _ = np.linalg.qr(accepted, mode="reduced")
        accepted = _phase_fix_columns(accepted)
        combined = np.column_stack((current, accepted))
    else:
        combined = current.copy()
    return RrqrAppendResult(
        basis=combined,
        accepted=rank,
        deflated=packet.shape[1] - numerical_rank,
        truncated=numerical_rank - rank,
        pivot_order=tuple(int(value) for value in pivots),
        diagonal=diagonal,
    )


@dataclass(frozen=True)
class PurgConstructionSettings:
    """Fixed deterministic settings for one PURG basis family."""

    caps: tuple[int, ...] = (32, 64, 96, 128)
    shift: float = 0.5
    rrqr_relative_tolerance: float = 1e-12
    solve_relative_tolerance: float = 1e-11
    eigensolver_tolerance: float = 1e-12
    final_time: float = 4.0
    construction_step: float = 0.0025

    def __post_init__(self) -> None:
        if not self.caps or any(cap <= 0 for cap in self.caps):
            raise ValueError("caps must contain positive dimensions")
        if tuple(sorted(set(self.caps))) != self.caps:
            raise ValueError("caps must be strictly increasing")
        if self.shift <= 0.0:
            raise ValueError("shift must be positive")
        if self.rrqr_relative_tolerance <= 0.0:
            raise ValueError("rrqr_relative_tolerance must be positive")
        if self.solve_relative_tolerance <= 0.0:
            raise ValueError("solve_relative_tolerance must be positive")
        if self.eigensolver_tolerance <= 0.0:
            raise ValueError("eigensolver_tolerance must be positive")
        if self.final_time <= 0.0:
            raise ValueError("final_time must be positive")
        if self.construction_step <= 0.0:
            raise ValueError("construction_step must be positive")


@dataclass(frozen=True)
class PurgReducedModel:
    """Online-safe Hermitian compression and its reduced initial ket."""

    phonon_cutoff: int
    cap_label: int
    static_hamiltonian: ComplexArray
    drive_hamiltonian: ComplexArray
    raw_observables: ComplexArray
    initial_state: ComplexArray

    @property
    def dimension(self) -> int:
        return int(self.static_hamiltonian.shape[0])

    def hamiltonian(self, drive_value: float) -> ComplexArray:
        return self.static_hamiltonian + drive_value * self.drive_hamiltonian

    def rhs(self, state: ComplexArray, *, drive_value: float) -> ComplexArray:
        vector = _require_complex_vector(
            state,
            length=self.dimension,
            name="reduced state",
        )
        return -1j * (self.hamiltonian(drive_value) @ vector)

    def raw_coordinates(self, state: ComplexArray) -> FloatArray:
        vector = _require_complex_vector(
            state,
            length=self.dimension,
            name="reduced state",
        )
        return np.asarray(
            [
                np.vdot(vector, operator @ vector).real
                for operator in self.raw_observables
            ],
            dtype=float,
        )

    def raw_velocity(
        self,
        state: ComplexArray,
        *,
        drive_value: float,
    ) -> FloatArray:
        vector = _require_complex_vector(
            state,
            length=self.dimension,
            name="reduced state",
        )
        velocity = self.rhs(vector, drive_value=drive_value)
        return np.asarray(
            [
                2.0 * np.vdot(velocity, operator @ vector).real
                for operator in self.raw_observables
            ],
            dtype=float,
        )

    def centered_coordinates(self, state: ComplexArray) -> FloatArray:
        return raw_moments_to_closed_coordinates(self.raw_coordinates(state))

    def centered_velocity(
        self,
        state: ComplexArray,
        *,
        drive_value: float,
    ) -> FloatArray:
        raw = self.raw_coordinates(state)
        raw_velocity = self.raw_velocity(state, drive_value=drive_value)
        return raw_velocity_to_closed_velocity(raw, raw_velocity)

    def energy(self, state: ComplexArray, *, drive_value: float) -> float:
        vector = _require_complex_vector(
            state,
            length=self.dimension,
            name="reduced state",
        )
        return float(
            np.vdot(vector, self.hamiltonian(drive_value) @ vector).real
        )


@dataclass(frozen=True)
class PurgProjection:
    """Offline full-space lift and residual data for one reduced model."""

    model: PurgReducedModel
    basis: ComplexArray
    static_hamiltonian: SparseOperator
    drive_hamiltonian: SparseOperator
    full_raw_observables: tuple[SparseOperator, ...]
    reference_initial_state: ComplexArray
    static_residual_gram: ComplexArray
    cross_residual_gram: ComplexArray
    drive_residual_gram: ComplexArray
    compression_hermitian_leakage: dict[str, float]

    @property
    def full_dimension(self) -> int:
        return int(self.basis.shape[0])

    def lift(self, state: ComplexArray) -> ComplexArray:
        vector = _require_complex_vector(
            state,
            length=self.model.dimension,
            name="reduced state",
        )
        return self.basis @ vector

    def projection_residual(
        self,
        state: ComplexArray,
        *,
        drive_value: float,
    ) -> ComplexArray:
        vector = _require_complex_vector(
            state,
            length=self.model.dimension,
            name="reduced state",
        )
        lifted = self.basis @ vector
        full_action = (
            self.static_hamiltonian @ lifted
            + drive_value * (self.drive_hamiltonian @ lifted)
        )
        return full_action - self.basis @ (
            self.model.hamiltonian(drive_value) @ vector
        )

    def projection_residual_norm(
        self,
        state: ComplexArray,
        *,
        drive_value: float,
    ) -> float:
        vector = _require_complex_vector(
            state,
            length=self.model.dimension,
            name="reduced state",
        )
        gram = (
            self.static_residual_gram
            + drive_value
            * (self.cross_residual_gram + self.cross_residual_gram.conj().T)
            + drive_value**2 * self.drive_residual_gram
        )
        squared = float(np.vdot(vector, gram @ vector).real)
        return sqrt(max(0.0, squared))


@dataclass(frozen=True)
class PurgCapRecord:
    """Construction record at one preregistered cap label."""

    cap_label: int
    actual_rank: int
    projection: PurgProjection | None
    residual_peak: float | None
    residual_peak_time: float | None
    greedy_packets: int
    deflated_columns: int
    truncated_columns: int


@dataclass(frozen=True)
class PurgConstruction:
    """Complete offline basis family and its construction diagnostics."""

    parameters: DimerParameters
    phonon_cutoff: int
    settings: PurgConstructionSettings
    raw_basis: RawMomentBasis
    ground_energy: float
    ground_state: ComplexArray
    ground_residual: float
    initial_rank: int
    solve_relative_residuals: FloatArray
    records: tuple[PurgCapRecord, ...]

    def record(self, cap_label: int) -> PurgCapRecord:
        for record in self.records:
            if record.cap_label == cap_label:
                return record
        raise KeyError(f"no PURG record for cap {cap_label}")


@dataclass(frozen=True)
class PurgMidpointTrajectory:
    """Unitary exponential-midpoint trajectory sampled at step endpoints."""

    times: FloatArray
    states: ComplexArray
    norm_defect: float


@dataclass(frozen=True)
class PurgOperatorBounds:
    """Spectral norms used by the a posteriori moment-error certificate."""

    raw: FloatArray
    static_derivative: FloatArray
    drive_derivative: FloatArray


@dataclass(frozen=True)
class PurgCertificate:
    """Construction-only Duhamel and centered-derivative certificate."""

    times: FloatArray
    states: ComplexArray
    projection_residual_norms: FloatArray
    total_defect_norms: FloatArray
    cumulative_defect: FloatArray
    initial_state_error: float
    state_error_bound: FloatArray
    raw_derivative_absolute_bounds: FloatArray
    centered_derivative_absolute_bounds: FloatArray
    block_derivative_metrics: dict[str, dict[str, float]]
    quadrature_error_estimate: float
    continuous_norm_defect: float


def _deterministic_ground_state(
    hamiltonian: SparseOperator,
    *,
    tolerance: float,
) -> tuple[float, ComplexArray, float]:
    dimension = hamiltonian.shape[0]
    v0 = np.linspace(1.0, 2.0, dimension, dtype=float).astype(complex)
    v0 /= np.linalg.norm(v0)
    eigenvalues, eigenvectors = eigsh(
        hamiltonian,
        k=1,
        which="SA",
        tol=tolerance,
        v0=v0,
    )
    energy = float(eigenvalues[0].real)
    state = _normalize(eigenvectors[:, 0], name="ground state")
    state = _fix_vector_phase(state)
    residual = float(np.linalg.norm(hamiltonian @ state - energy * state))
    return energy, state, residual


def _compress_hermitian(
    basis: ComplexArray,
    operator: SparseOperator,
) -> tuple[ComplexArray, float]:
    raw = np.asarray(basis.conj().T @ (operator @ basis), dtype=complex)
    leakage = float(np.linalg.norm(raw - raw.conj().T))
    scale = max(1.0, float(np.linalg.norm(raw)))
    return 0.5 * (raw + raw.conj().T), leakage / scale


def _make_projection(
    *,
    basis: ComplexArray,
    cap_label: int,
    phonon_cutoff: int,
    static_hamiltonian: SparseOperator,
    drive_hamiltonian: SparseOperator,
    raw_observables: tuple[SparseOperator, ...],
    reference_initial_state: ComplexArray,
) -> tuple[PurgProjection, dict[str, float]]:
    static_reduced, static_leakage = _compress_hermitian(
        basis,
        static_hamiltonian,
    )
    drive_reduced, drive_leakage = _compress_hermitian(
        basis,
        drive_hamiltonian,
    )
    compressed_raw: list[ComplexArray] = []
    maximum_raw_leakage = 0.0
    for operator in raw_observables:
        compressed, leakage = _compress_hermitian(basis, operator)
        compressed_raw.append(compressed)
        maximum_raw_leakage = max(maximum_raw_leakage, leakage)

    projected_initial = basis.conj().T @ reference_initial_state
    initial_state = _normalize(projected_initial, name="projected initial state")
    model = PurgReducedModel(
        phonon_cutoff=phonon_cutoff,
        cap_label=cap_label,
        static_hamiltonian=static_reduced,
        drive_hamiltonian=drive_reduced,
        raw_observables=np.asarray(compressed_raw, dtype=complex),
        initial_state=initial_state,
    )

    static_residual = static_hamiltonian @ basis - basis @ static_reduced
    drive_residual = drive_hamiltonian @ basis - basis @ drive_reduced
    static_gram = np.asarray(static_residual.conj().T @ static_residual)
    cross_gram = np.asarray(static_residual.conj().T @ drive_residual)
    drive_gram = np.asarray(drive_residual.conj().T @ drive_residual)
    projection = PurgProjection(
        model=model,
        basis=np.asarray(basis, dtype=complex),
        static_hamiltonian=static_hamiltonian,
        drive_hamiltonian=drive_hamiltonian,
        full_raw_observables=raw_observables,
        reference_initial_state=np.asarray(reference_initial_state, dtype=complex),
        static_residual_gram=0.5 * (static_gram + static_gram.conj().T),
        cross_residual_gram=cross_gram,
        drive_residual_gram=0.5 * (drive_gram + drive_gram.conj().T),
        compression_hermitian_leakage={
            "static_hamiltonian": static_leakage,
            "drive_hamiltonian": drive_leakage,
            "raw_observables": maximum_raw_leakage,
        },
    )
    return projection, {
        "static_hamiltonian": static_leakage,
        "drive_hamiltonian": drive_leakage,
        "raw_observables": maximum_raw_leakage,
    }


def _time_grid(final_time: float, step: float) -> FloatArray:
    count = int(round(final_time / step))
    if count <= 0 or not np.isclose(count * step, final_time, atol=1e-13):
        raise ValueError("final_time must be an integer multiple of step")
    return np.linspace(0.0, final_time, count + 1, dtype=float)


def propagate_purg_midpoint(
    model: PurgReducedModel,
    parameters: DimerParameters,
    *,
    final_time: float,
    step: float,
) -> PurgMidpointTrajectory:
    """Propagate the reduced ket with unitary exponential midpoint steps."""

    times = _time_grid(final_time, step)
    states = np.empty((times.size, model.dimension), dtype=complex)
    states[0] = model.initial_state
    maximum_norm_defect = abs(float(np.vdot(states[0], states[0]).real) - 1.0)
    for index in range(times.size - 1):
        midpoint = 0.5 * (times[index] + times[index + 1])
        hamiltonian = model.hamiltonian(parameters.drive_difference(midpoint))
        eigenvalues, eigenvectors = eigh(hamiltonian, check_finite=True)
        coefficients = eigenvectors.conj().T @ states[index]
        states[index + 1] = eigenvectors @ (
            np.exp(-1j * step * eigenvalues) * coefficients
        )
        maximum_norm_defect = max(
            maximum_norm_defect,
            abs(float(np.vdot(states[index + 1], states[index + 1]).real) - 1.0),
        )
    return PurgMidpointTrajectory(
        times=times,
        states=states,
        norm_defect=maximum_norm_defect,
    )


def _orthogonal_component(vector: ComplexArray, reference: ComplexArray) -> ComplexArray:
    return vector - reference * np.vdot(reference, vector)


def _solve_shifted(
    factorization: object,
    operator: SparseOperator,
    vector: ComplexArray,
) -> tuple[ComplexArray, float]:
    solution = np.asarray(factorization.solve(np.asarray(vector, dtype=complex)))
    denominator = max(float(np.linalg.norm(vector)), 1e-30)
    residual = float(np.linalg.norm(operator @ solution - vector) / denominator)
    return solution, residual


def _peak_projection_residual(
    projection: PurgProjection,
    parameters: DimerParameters,
    trajectory: PurgMidpointTrajectory,
) -> tuple[float, float, ComplexArray]:
    norms = np.asarray(
        [
            projection.projection_residual_norm(
                state,
                drive_value=parameters.drive_difference(float(time)),
            )
            for time, state in zip(
                trajectory.times,
                trajectory.states,
                strict=True,
            )
        ],
        dtype=float,
    )
    peak_index = int(np.argmax(norms))
    peak_time = float(trajectory.times[peak_index])
    residual = projection.projection_residual(
        trajectory.states[peak_index],
        drive_value=parameters.drive_difference(peak_time),
    )
    return float(norms[peak_index]), peak_time, residual


def build_purg_construction(
    parameters: DimerParameters,
    *,
    phonon_cutoff: int,
    settings: PurgConstructionSettings | None = None,
) -> PurgConstruction:
    """Build nested deterministic PURG spaces without exact driven snapshots."""

    resolved = settings or PurgConstructionSettings()
    exact_model = _build_exact_dimer_model(
        parameters,
        phonon_cutoff=phonon_cutoff,
    )
    raw_basis = _build_raw_moment_basis_from_model(
        exact_model,
        phonon_cutoff=phonon_cutoff,
    )
    static_hamiltonian = exact_model.static_hamiltonian.tocsc()
    drive_hamiltonian = exact_model.drive_operator.tocsc()
    ground_energy, ground_state, ground_residual = _deterministic_ground_state(
        static_hamiltonian,
        tolerance=resolved.eigensolver_tolerance,
    )
    dimension = static_hamiltonian.shape[0]
    identity = eye(dimension, format="csc", dtype=complex)
    shifted = (
        static_hamiltonian
        - ground_energy * identity
        + resolved.shift * identity
    ).tocsc()
    shifted_factorization = splu(shifted)

    drive_seed = _orthogonal_component(
        drive_hamiltonian @ ground_state,
        ground_state,
    )
    initial_columns: list[ComplexArray] = [ground_state, drive_seed]
    for operator in raw_basis.observables:
        expectation = np.vdot(ground_state, operator @ ground_state).real
        centered_action = operator @ ground_state - expectation * ground_state
        initial_columns.append(_orthogonal_component(centered_action, ground_state))

    solve_residuals: list[float] = []
    inverse_drive, inverse_residual = _solve_shifted(
        shifted_factorization,
        shifted,
        drive_seed,
    )
    solve_residuals.append(inverse_residual)
    initial_columns.extend((inverse_drive, shifted @ drive_seed))
    initial_packet = np.column_stack(initial_columns)
    empty_basis = np.empty((dimension, 0), dtype=complex)
    initial_result = _rrqr_append(
        empty_basis,
        initial_packet,
        relative_tolerance=resolved.rrqr_relative_tolerance,
    )
    basis = initial_result.basis
    initial_rank = basis.shape[1]
    if initial_rank == 0:
        raise RuntimeError("PURG initial block has zero rank")

    records: list[PurgCapRecord] = []
    greedy_packets = 0
    deflated_columns = initial_result.deflated
    truncated_columns = initial_result.truncated
    cached_peak: tuple[float, float, ComplexArray] | None = None
    stalled = False

    for cap_label in resolved.caps:
        if initial_rank > cap_label:
            records.append(
                PurgCapRecord(
                    cap_label=cap_label,
                    actual_rank=initial_rank,
                    projection=None,
                    residual_peak=None,
                    residual_peak_time=None,
                    greedy_packets=greedy_packets,
                    deflated_columns=deflated_columns,
                    truncated_columns=truncated_columns,
                )
            )
            continue

        while basis.shape[1] < cap_label and not stalled:
            working_projection, _ = _make_projection(
                basis=basis,
                cap_label=cap_label,
                phonon_cutoff=phonon_cutoff,
                static_hamiltonian=static_hamiltonian,
                drive_hamiltonian=drive_hamiltonian,
                raw_observables=raw_basis.observables,
                reference_initial_state=ground_state,
            )
            if cached_peak is None:
                trajectory = propagate_purg_midpoint(
                    working_projection.model,
                    parameters,
                    final_time=resolved.final_time,
                    step=resolved.construction_step,
                )
                cached_peak = _peak_projection_residual(
                    working_projection,
                    parameters,
                    trajectory,
                )
            _, _, peak_residual = cached_peak
            inverse_residual_vector, solve_residual = _solve_shifted(
                shifted_factorization,
                shifted,
                peak_residual,
            )
            solve_residuals.append(solve_residual)
            packet = np.column_stack(
                (
                    peak_residual,
                    inverse_residual_vector,
                    shifted @ peak_residual,
                    drive_hamiltonian @ peak_residual,
                )
            )
            append_result = _rrqr_append(
                basis,
                packet,
                relative_tolerance=resolved.rrqr_relative_tolerance,
                maximum_new_columns=cap_label - basis.shape[1],
            )
            greedy_packets += 1
            deflated_columns += append_result.deflated
            truncated_columns += append_result.truncated
            if append_result.accepted == 0:
                stalled = True
                break
            basis = append_result.basis
            cached_peak = None

        if basis.shape[1] == cap_label:
            projection, _ = _make_projection(
                basis=basis,
                cap_label=cap_label,
                phonon_cutoff=phonon_cutoff,
                static_hamiltonian=static_hamiltonian,
                drive_hamiltonian=drive_hamiltonian,
                raw_observables=raw_basis.observables,
                reference_initial_state=ground_state,
            )
            trajectory = propagate_purg_midpoint(
                projection.model,
                parameters,
                final_time=resolved.final_time,
                step=resolved.construction_step,
            )
            cached_peak = _peak_projection_residual(
                projection,
                parameters,
                trajectory,
            )
            peak_value, peak_time, _ = cached_peak
        else:
            projection = None
            peak_value = None
            peak_time = None
        records.append(
            PurgCapRecord(
                cap_label=cap_label,
                actual_rank=basis.shape[1],
                projection=projection,
                residual_peak=peak_value,
                residual_peak_time=peak_time,
                greedy_packets=greedy_packets,
                deflated_columns=deflated_columns,
                truncated_columns=truncated_columns,
            )
        )

    maximum_solve_residual = max(solve_residuals, default=0.0)
    if maximum_solve_residual > resolved.solve_relative_tolerance:
        raise RuntimeError(
            "shifted solve residual exceeds construction tolerance: "
            f"{maximum_solve_residual:.3e}"
        )
    return PurgConstruction(
        parameters=parameters,
        phonon_cutoff=phonon_cutoff,
        settings=resolved,
        raw_basis=raw_basis,
        ground_energy=ground_energy,
        ground_state=ground_state,
        ground_residual=ground_residual,
        initial_rank=initial_rank,
        solve_relative_residuals=np.asarray(solve_residuals, dtype=float),
        records=tuple(records),
    )


def _hermitian_spectral_norm(operator: SparseOperator) -> float:
    matrix = 0.5 * (operator + operator.getH())
    matrix = matrix.tocsc()
    if matrix.nnz == 0:
        return 0.0
    dimension = matrix.shape[0]
    if dimension <= 96:
        return float(np.max(np.abs(np.linalg.eigvalsh(matrix.toarray()))))
    v0 = np.linspace(1.0, 2.0, dimension, dtype=float).astype(complex)
    v0 /= np.linalg.norm(v0)
    value = eigsh(
        matrix,
        k=1,
        which="LM",
        return_eigenvectors=False,
        tol=1e-10,
        v0=v0,
    )[0]
    return abs(float(value.real))


def build_purg_operator_bounds(projection: PurgProjection) -> PurgOperatorBounds:
    """Compute full-space spectral norms for the derivative certificate."""

    raw_norms: list[float] = []
    static_norms: list[float] = []
    drive_norms: list[float] = []
    for operator in projection.full_raw_observables:
        raw_norms.append(_hermitian_spectral_norm(operator))
        static_derivative = (
            1j
            * (
                projection.static_hamiltonian @ operator
                - operator @ projection.static_hamiltonian
            )
        ).tocsc()
        drive_derivative = (
            1j
            * (
                projection.drive_hamiltonian @ operator
                - operator @ projection.drive_hamiltonian
            )
        ).tocsc()
        static_norms.append(_hermitian_spectral_norm(static_derivative))
        drive_norms.append(_hermitian_spectral_norm(drive_derivative))
    return PurgOperatorBounds(
        raw=np.asarray(raw_norms, dtype=float),
        static_derivative=np.asarray(static_norms, dtype=float),
        drive_derivative=np.asarray(drive_norms, dtype=float),
    )


_GK15_ABSCISSA = np.asarray(
    [
        0.9914553711208126,
        0.9491079123427585,
        0.8648644233597691,
        0.7415311855993945,
        0.5860872354676911,
        0.4058451513773972,
        0.2077849550078985,
        0.0,
    ],
    dtype=float,
)
_GK15_WEIGHT = np.asarray(
    [
        0.02293532201052922,
        0.06309209262997855,
        0.1047900103222502,
        0.1406532597155259,
        0.1690047266392679,
        0.1903505780647854,
        0.2044329400752989,
        0.2094821410847278,
    ],
    dtype=float,
)
_G7_WEIGHT_BY_GK_INDEX = {
    1: 0.1294849661688697,
    3: 0.2797053914892767,
    5: 0.3818300505051189,
    7: 0.4179591836734694,
}


def _gk15(
    function: object,
    left: float,
    right: float,
) -> tuple[float, float]:
    center = 0.5 * (left + right)
    half = 0.5 * (right - left)
    kronrod = 0.0
    gauss = 0.0
    for index, abscissa in enumerate(_GK15_ABSCISSA):
        if abscissa == 0.0:
            value_sum = float(function(center))
        else:
            value_sum = float(function(center - half * abscissa)) + float(
                function(center + half * abscissa)
            )
        kronrod += _GK15_WEIGHT[index] * value_sum
        if index in _G7_WEIGHT_BY_GK_INDEX:
            gauss += _G7_WEIGHT_BY_GK_INDEX[index] * value_sum
    kronrod *= half
    gauss *= half
    return kronrod, abs(kronrod - gauss)


def _adaptive_gk15(
    function: object,
    left: float,
    right: float,
    *,
    absolute_tolerance: float,
    depth: int = 0,
) -> tuple[float, float]:
    value, error = _gk15(function, left, right)
    if error <= absolute_tolerance or depth >= 12:
        return value, error
    midpoint = 0.5 * (left + right)
    left_value, left_error = _adaptive_gk15(
        function,
        left,
        midpoint,
        absolute_tolerance=0.5 * absolute_tolerance,
        depth=depth + 1,
    )
    right_value, right_error = _adaptive_gk15(
        function,
        midpoint,
        right,
        absolute_tolerance=0.5 * absolute_tolerance,
        depth=depth + 1,
    )
    return left_value + right_value, left_error + right_error


def _centering_hessian() -> FloatArray:
    zero = np.zeros(_RAW_DIMENSION, dtype=float)
    base = raw_to_closed_jacobian(zero)
    hessian = np.empty((31, _RAW_DIMENSION, _RAW_DIMENSION), dtype=float)
    for column in range(_RAW_DIMENSION):
        direction = np.zeros(_RAW_DIMENSION, dtype=float)
        direction[column] = 1.0
        hessian[:, :, column] = raw_to_closed_jacobian(direction) - base
    return 0.5 * (hessian + np.swapaxes(hessian, 1, 2))


_CENTERING_HESSIAN = _centering_hessian()


def certify_purg_projection(
    projection: PurgProjection,
    parameters: DimerParameters,
    operator_bounds: PurgOperatorBounds,
    *,
    final_time: float,
    step: float,
    quadrature_absolute_tolerance: float = 1e-12,
) -> PurgCertificate:
    """Compute the unitary midpoint path and its Duhamel error certificate."""

    if quadrature_absolute_tolerance <= 0.0:
        raise ValueError("quadrature_absolute_tolerance must be positive")
    model = projection.model
    times = _time_grid(final_time, step)
    states = np.empty((times.size, model.dimension), dtype=complex)
    states[0] = model.initial_state
    projection_initial = projection.lift(states[0])
    initial_error = float(
        np.linalg.norm(projection.reference_initial_state - projection_initial)
    )
    cumulative = np.zeros(times.size, dtype=float)
    projection_norms = np.empty(times.size, dtype=float)
    total_defect_norms = np.empty(times.size, dtype=float)
    maximum_norm_defect = abs(float(np.vdot(states[0], states[0]).real) - 1.0)
    quadrature_error = 0.0
    local_tolerance = quadrature_absolute_tolerance / (times.size - 1)

    for index in range(times.size - 1):
        left = float(times[index])
        right = float(times[index + 1])
        midpoint = 0.5 * (left + right)
        midpoint_drive = parameters.drive_difference(midpoint)
        midpoint_hamiltonian = model.hamiltonian(midpoint_drive)
        eigenvalues, eigenvectors = eigh(midpoint_hamiltonian, check_finite=True)
        spectral_coordinates = eigenvectors.conj().T @ states[index]

        def continuous_state(time: float) -> ComplexArray:
            elapsed = time - left
            return eigenvectors @ (
                np.exp(-1j * elapsed * eigenvalues) * spectral_coordinates
            )

        def total_defect(time: float) -> float:
            nonlocal maximum_norm_defect
            state = continuous_state(time)
            maximum_norm_defect = max(
                maximum_norm_defect,
                abs(float(np.vdot(state, state).real) - 1.0),
            )
            drive_value = parameters.drive_difference(time)
            projection_norm = projection.projection_residual_norm(
                state,
                drive_value=drive_value,
            )
            integrator_defect = abs(midpoint_drive - drive_value) * float(
                np.linalg.norm(model.drive_hamiltonian @ state)
            )
            return sqrt(projection_norm**2 + integrator_defect**2)

        interval_value, interval_error = _adaptive_gk15(
            total_defect,
            left,
            right,
            absolute_tolerance=local_tolerance,
        )
        quadrature_error += interval_error
        cumulative[index + 1] = cumulative[index] + interval_value
        states[index + 1] = continuous_state(right)

        drive_left = parameters.drive_difference(left)
        projection_norms[index] = projection.projection_residual_norm(
            states[index],
            drive_value=drive_left,
        )
        integrator_left = abs(midpoint_drive - drive_left) * float(
            np.linalg.norm(model.drive_hamiltonian @ states[index])
        )
        total_defect_norms[index] = sqrt(
            projection_norms[index] ** 2 + integrator_left**2
        )

    final_drive = parameters.drive_difference(float(times[-1]))
    projection_norms[-1] = projection.projection_residual_norm(
        states[-1],
        drive_value=final_drive,
    )
    previous_midpoint = 0.5 * (times[-2] + times[-1])
    final_integrator = abs(
        parameters.drive_difference(float(previous_midpoint)) - final_drive
    ) * float(np.linalg.norm(model.drive_hamiltonian @ states[-1]))
    total_defect_norms[-1] = sqrt(
        projection_norms[-1] ** 2 + final_integrator**2
    )

    delta = initial_error + cumulative
    raw_derivative_bounds = np.empty((times.size, _RAW_DIMENSION), dtype=float)
    centered_derivative_bounds = np.empty((times.size, 31), dtype=float)
    for index, (time, state) in enumerate(zip(times, states, strict=True)):
        drive_value = parameters.drive_difference(float(time))
        raw = model.raw_coordinates(state)
        raw_velocity = model.raw_velocity(state, drive_value=drive_value)
        raw_value_bound = (
            operator_bounds.raw * delta[index] * (2.0 + delta[index])
        )
        derivative_operator_bound = (
            operator_bounds.static_derivative
            + abs(drive_value) * operator_bounds.drive_derivative
        )
        raw_derivative_bound = (
            derivative_operator_bound * delta[index] * (2.0 + delta[index])
            + 2.0 * operator_bounds.raw * projection_norms[index]
        )
        raw_derivative_bounds[index] = raw_derivative_bound
        jacobian = raw_to_closed_jacobian(raw)
        first = np.abs(jacobian) @ raw_derivative_bound
        second = np.einsum(
            "iab,a,b->i",
            np.abs(_CENTERING_HESSIAN),
            raw_value_bound,
            np.abs(raw_velocity),
        )
        third = np.einsum(
            "iab,a,b->i",
            np.abs(_CENTERING_HESSIAN),
            raw_value_bound,
            raw_derivative_bound,
        )
        centered_derivative_bounds[index] = first + second + third

    block_metrics: dict[str, dict[str, float]] = {}
    for name, block_slice in _CENTERED_BLOCKS.items():
        norms = np.linalg.norm(
            centered_derivative_bounds[:, block_slice],
            axis=1,
        )
        block_metrics[name] = {
            "rms_l2_bound": float(np.sqrt(np.mean(norms**2))),
            "max_l2_bound": float(np.max(norms)),
        }

    return PurgCertificate(
        times=times,
        states=states,
        projection_residual_norms=projection_norms,
        total_defect_norms=total_defect_norms,
        cumulative_defect=cumulative,
        initial_state_error=initial_error,
        state_error_bound=delta,
        raw_derivative_absolute_bounds=raw_derivative_bounds,
        centered_derivative_absolute_bounds=centered_derivative_bounds,
        block_derivative_metrics=block_metrics,
        quadrature_error_estimate=quadrature_error,
        continuous_norm_defect=maximum_norm_defect,
    )


def purg_gate_a_diagnostics(construction: PurgConstruction) -> dict[str, object]:
    """Return deterministic algebra and data-flow diagnostics for each cap."""

    rng = np.random.default_rng(2026080311)
    maximum_round_trip_residual = 0.0
    maximum_jacobian_residual = 0.0
    for _ in range(100):
        raw = rng.normal(scale=0.2, size=_RAW_DIMENSION)
        closed = raw_moments_to_closed_coordinates(raw)
        recovered = closed_coordinates_to_raw_moments(closed)
        maximum_round_trip_residual = max(
            maximum_round_trip_residual,
            float(np.linalg.norm(recovered - raw)),
        )
        direction = rng.normal(size=_RAW_DIMENSION)
        direction /= np.linalg.norm(direction)
        step = 0.125
        directional = (
            raw_moments_to_closed_coordinates(raw + step * direction)
            - raw_moments_to_closed_coordinates(raw - step * direction)
        ) / (2.0 * step)
        analytic = raw_to_closed_jacobian(raw) @ direction
        maximum_jacobian_residual = max(
            maximum_jacobian_residual,
            float(np.linalg.norm(directional - analytic)),
        )

    decoupled = build_krylov_closure_construction(
        replace(construction.parameters, lambda_ep=0.0),
        phonon_cutoff=construction.phonon_cutoff,
        shell_count=1,
        rank_tolerance=1e-10,
    )
    decoupled_force_norm = _operator_block_norm(
        decoupled.static_force,
        dimension=decoupled.hilbert_dimension,
    )
    if decoupled_force_norm <= 1e-12:
        decoupled_declared_rank = 0
    elif decoupled.force_singular_values.size:
        reference = float(decoupled.force_singular_values[0])
        decoupled_declared_rank = int(
            np.count_nonzero(
                decoupled.force_singular_values > 1e-10 * reference
            )
        )
    else:
        decoupled_declared_rank = 0

    allowed_online_fields = {
        "phonon_cutoff",
        "cap_label",
        "static_hamiltonian",
        "drive_hamiltonian",
        "raw_observables",
        "initial_state",
    }
    available_model = next(
        (
            record.projection.model
            for record in construction.records
            if record.projection is not None
        ),
        None,
    )
    online_fields = (
        set(available_model.__dataclass_fields__) if available_model else set()
    )

    diagnostics: dict[str, object] = {
        "ground_residual": construction.ground_residual,
        "initial_rank": construction.initial_rank,
        "maximum_shifted_solve_relative_residual": float(
            np.max(construction.solve_relative_residuals)
            if construction.solve_relative_residuals.size
            else 0.0
        ),
        "coordinate_map_round_trip_residual": maximum_round_trip_residual,
        "coordinate_jacobian_directional_residual": (
            maximum_jacobian_residual
        ),
        "decoupled_force": {
            "norm": decoupled_force_norm,
            "declared_rank": decoupled_declared_rank,
            "passed": bool(
                decoupled_force_norm <= 1e-12
                and decoupled_declared_rank == 0
                and decoupled.force_rank == 0
            ),
        },
        "online_dependency_audit": {
            "fields": sorted(online_fields),
            "passed": bool(available_model is not None and online_fields == allowed_online_fields),
        },
        "caps": {},
    }
    for record in construction.records:
        if record.projection is None:
            diagnostics["caps"][str(record.cap_label)] = {
                "available": False,
                "actual_rank": record.actual_rank,
            }
            continue
        projection = record.projection
        basis = projection.basis
        orthogonality = float(
            np.linalg.norm(basis.conj().T @ basis - np.eye(basis.shape[1]))
        )
        initial_containment = float(
            np.linalg.norm(
                projection.reference_initial_state
                - basis @ (basis.conj().T @ projection.reference_initial_state)
            )
        )
        drive_state = projection.drive_hamiltonian @ projection.reference_initial_state
        drive_containment = float(
            np.linalg.norm(drive_state - basis @ (basis.conj().T @ drive_state))
        )
        diagnostics["caps"][str(record.cap_label)] = {
            "available": True,
            "actual_rank": record.actual_rank,
            "orthogonality_residual": orthogonality,
            "initial_state_containment_residual": initial_containment,
            "initial_drive_direction_containment_residual": drive_containment,
            "static_hermitian_leakage": (
                projection.compression_hermitian_leakage[
                    "static_hamiltonian"
                ]
            ),
            "drive_hermitian_leakage": (
                projection.compression_hermitian_leakage[
                    "drive_hamiltonian"
                ]
            ),
            "raw_observable_hermitian_leakage": (
                projection.compression_hermitian_leakage["raw_observables"]
            ),
            "residual_peak": record.residual_peak,
            "residual_peak_time": record.residual_peak_time,
        }
    return diagnostics
