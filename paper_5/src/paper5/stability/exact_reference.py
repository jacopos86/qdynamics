"""Exact Holstein-dimer reference propagation and moment contractions.

The calculation works in the sector with one spin-up and one spin-down
electron.  It diagonalizes the two-site, two-local-phonon Hamiltonian from
Eqs. (16)--(19) of arXiv:2606.22233v1 and contracts the resulting state into
the matrix variables used by Eqs. (14a)--(14e).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations
from math import comb, factorial, sqrt
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import eigsh

from .cone_correction import closed_joint_cone_projected_rhs
from .hubbard_dimer import DimerParameters, RhsFunction
from .matrix_reference import (
    MatrixDimerState,
    closed_scalar_rhs,
    correlation_source_by_phonon_mode,
    matrix_dimer_rhs,
    matrix_state_to_closed_scalar_coordinates,
    same_spin_density_covariance,
)
from .moment_hierarchy import (
    FOURTH_ORDER_HIERARCHY,
    PAULI_LABELS,
    THIRD_ORDER_HIERARCHY,
    MomentHierarchy,
    MomentKey,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
ClosedProtocol = Literal[
    "archive",
    "residual_subtracted",
    "joint_barrier",
    "residual_subtracted_joint_barrier",
]
CLOSED_PROTOCOLS: tuple[ClosedProtocol, ...] = (
    "archive",
    "residual_subtracted",
    "joint_barrier",
    "residual_subtracted_joint_barrier",
)


@dataclass(frozen=True)
class ExactGroundState:
    """Ground-state energy and contracted equal-time matrix variables."""

    energy: float
    phonon_cutoff: int
    matrix_state: MatrixDimerState


@dataclass(frozen=True)
class ExactJointMomentInitialState:
    """One ground-state contraction shared by matrix and hierarchy charts."""

    energy: float
    phonon_cutoff: int
    matrix_state: MatrixDimerState
    hierarchy_coordinates: FloatArray


@dataclass(frozen=True)
class ExactDrivenTrajectory:
    """Exact wavefunction propagation sampled as retained matrix moments."""

    times: FloatArray
    matrix_states: tuple[MatrixDimerState, ...]
    matrix_derivatives: tuple[MatrixDimerState, ...]
    state_norms: FloatArray
    phonon_cutoff: int
    function_evaluations: int
    success: bool
    message: str


@dataclass(frozen=True)
class ExactCorrelationClosureTrajectory:
    """Exact audit of the two missing source blocks in the ``C`` equation.

    ``exact_mixed_moment`` and ``factorized_mixed_moment`` have indices
    ``[time, q, r, i, j]`` and represent, respectively,

    ``< [O_ij, n_{r up}] delta_X_r delta_b_q >``

    and the algebra-preserving product closure
    ``<[O_ij,n_r]> <delta_X_r delta_b_q>``.  Their difference is the genuine
    connected mixed-moment remainder denoted by ``K`` in the working note.
    ``archive_same_spin_covariance`` records the effective covariance left by
    archive Eq. (14d); comparing it with ``same_spin_covariance`` isolates the
    separate Pauli-algebra error.  The opposite-spin covariance has indices
    ``[time, q, i, j]``.
    """

    exact_trajectory: ExactDrivenTrajectory
    exact_mixed_moment: ComplexArray
    factorized_mixed_moment: ComplexArray
    same_spin_covariance: ComplexArray
    archive_same_spin_covariance: ComplexArray
    opposite_spin_covariance: ComplexArray
    archive_correlation_derivatives: ComplexArray
    mixed_moment_velocity_corrections: ComplexArray
    same_spin_pauli_velocity_corrections: ComplexArray
    opposite_spin_velocity_corrections: ComplexArray
    cutoff_velocity_remainders: ComplexArray

    @property
    def mixed_moment_remainder(self) -> ComplexArray:
        """Return ``K = Q_exact - Q_factorized`` at every sampled time."""

        return self.exact_mixed_moment - self.factorized_mixed_moment


@dataclass(frozen=True)
class ExactMomentHierarchyTrajectory:
    """Exact contractions into one symmetry-adapted moment hierarchy.

    The exact wavefunction supplies an offline oracle only.  ``coordinates``
    and ``coordinate_derivatives`` contain the same center-mode amplitude and
    spin-exchange-symmetric Hermitian Weyl moments used by the corresponding
    autonomous hierarchy.
    """

    maximum_degree: int
    times: FloatArray
    coordinates: FloatArray
    coordinate_derivatives: FloatArray
    state_norms: FloatArray
    phonon_cutoff: int
    function_evaluations: int
    success: bool
    message: str


@dataclass(frozen=True)
class ExactDiagnosticWavefunctionTrajectory:
    """Exact truncated states exposed only for offline representation gates.

    Autonomous closures and controllers must not consume this trajectory.
    Its purpose is to test whether a proposed finite representation can
    compress the exact state and reproduce its instantaneous velocity before
    that representation is propagated on its own.
    """

    times: FloatArray
    state_vectors: ComplexArray
    state_derivatives: ComplexArray
    phonon_cutoff: int
    function_evaluations: int
    success: bool
    message: str


ExactThirdCumulantTrajectory = ExactMomentHierarchyTrajectory
ExactFourthCumulantTrajectory = ExactMomentHierarchyTrajectory


@dataclass(frozen=True)
class ExactArchiveClosureComparison:
    """Exact contractions and the archive 31-coordinate trajectory."""

    times: FloatArray
    exact_coordinates: FloatArray
    archive_coordinates: FloatArray
    coordinate_errors: FloatArray
    block_names: tuple[str, ...]
    block_error_norms: FloatArray
    exact_trajectory: ExactDrivenTrajectory
    archive_function_evaluations: int
    success: bool
    message: str


@dataclass(frozen=True)
class ExactClosedProtocolComparison:
    """Exact contractions and one selected 31-coordinate trajectory."""

    protocol: ClosedProtocol
    times: FloatArray
    exact_coordinates: FloatArray
    closed_coordinates: FloatArray
    coordinate_errors: FloatArray
    block_names: tuple[str, ...]
    block_error_norms: FloatArray
    exact_trajectory: ExactDrivenTrajectory
    closed_function_evaluations: int
    success: bool
    message: str


@dataclass(frozen=True)
class _ExactDimerModel:
    static_hamiltonian: csc_matrix
    drive_operator: csc_matrix
    electron_observables: tuple[tuple[csc_matrix, ...], ...]
    phonon_annihilation: tuple[csc_matrix, ...]
    normal_phonon_observables: tuple[tuple[csc_matrix, ...], ...]
    anomalous_phonon_observables: tuple[tuple[csc_matrix, ...], ...]
    electron_phonon_observables: tuple[
        tuple[tuple[csc_matrix, ...], ...], ...
    ]
    spin_down_site_occupations: tuple[csc_matrix, ...]
    spin_pauli_observables: tuple[tuple[csc_matrix, ...], ...]
    center_phonon_annihilation: csc_matrix
    relative_phonon_annihilation: csc_matrix
    relative_position: csc_matrix
    relative_momentum: csc_matrix


@dataclass(frozen=True)
class _ExactWavefunctionTrajectory:
    """Private exact propagation retained only while contracting observables."""

    model: _ExactDimerModel
    times: FloatArray
    state_vectors: ComplexArray
    state_derivatives: ComplexArray
    phonon_cutoff: int
    function_evaluations: int
    success: bool
    message: str


def _expectation(
    state_vector: np.ndarray,
    operator: csc_matrix,
) -> complex:
    return complex(np.vdot(state_vector, operator @ state_vector))


def _expectation_derivative(
    state_vector: ComplexArray,
    state_derivative: ComplexArray,
    operator: csc_matrix,
) -> complex:
    """Differentiate ``<state|operator|state>`` without finite differences."""

    return complex(
        np.vdot(state_derivative, operator @ state_vector)
        + np.vdot(state_vector, operator @ state_derivative)
    )


def _build_exact_dimer_model(
    parameters: DimerParameters,
    *,
    phonon_cutoff: int,
) -> _ExactDimerModel:
    if phonon_cutoff < 1:
        raise ValueError("phonon_cutoff must be at least one")

    electron_identity = eye(4, format="csc", dtype=complex)
    site_identity = eye(2, format="csc", dtype=complex)
    sigma_x = csc_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex))
    sigma_y = csc_matrix(
        np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    )
    sigma_z = csc_matrix(np.diag([1.0, -1.0]).astype(complex))
    pauli_matrices = (site_identity, sigma_x, sigma_y, sigma_z)
    site_projectors = (
        csc_matrix(np.diag([1.0, 0.0]).astype(complex)),
        csc_matrix(np.diag([0.0, 1.0]).astype(complex)),
    )
    electron_hamiltonian = -parameters.hopping * (
        kron(sigma_x, site_identity, format="csc")
        + kron(site_identity, sigma_x, format="csc")
    )
    total_site_occupations = (
        kron(site_projectors[0], site_identity, format="csc")
        + kron(site_identity, site_projectors[0], format="csc"),
        kron(site_projectors[1], site_identity, format="csc")
        + kron(site_identity, site_projectors[1], format="csc"),
    )

    oscillator_dimension = phonon_cutoff + 1
    oscillator_identity = eye(
        oscillator_dimension, format="csc", dtype=complex
    )
    annihilation = diags(
        np.sqrt(np.arange(1, oscillator_dimension, dtype=float)),
        offsets=1,
        shape=(oscillator_dimension, oscillator_dimension),
        format="csc",
        dtype=complex,
    )
    phonon_identity = kron(
        oscillator_identity, oscillator_identity, format="csc"
    )
    local_annihilation = (
        kron(annihilation, oscillator_identity, format="csc"),
        kron(oscillator_identity, annihilation, format="csc"),
    )
    phonon_hamiltonian = parameters.omega_ph * sum(
        (
            operator.getH() @ operator
            for operator in local_annihilation
        ),
        start=csc_matrix(phonon_identity.shape, dtype=complex),
    )

    static_hamiltonian = (
        kron(electron_hamiltonian, phonon_identity, format="csc")
        + kron(electron_identity, phonon_hamiltonian, format="csc")
    )
    for site in range(2):
        displacement = (
            local_annihilation[site] + local_annihilation[site].getH()
        )
        static_hamiltonian += parameters.coupling * kron(
            total_site_occupations[site],
            displacement,
            format="csc",
        )

    # Eq. (19) has opposite site potentials (+v, -v).  The parameter helper
    # returns their difference, 2v, so the full Hamiltonian multiplies the
    # occupation imbalance by one half of that difference.
    drive_operator = 0.5 * kron(
        total_site_occupations[0] - total_site_occupations[1],
        phonon_identity,
        format="csc",
    )

    electron_operators: list[list[csc_matrix]] = []
    electron_observables: list[list[csc_matrix]] = []
    for alpha in range(2):
        operator_row: list[csc_matrix] = []
        observable_row: list[csc_matrix] = []
        for beta in range(2):
            one_body = np.zeros((2, 2), dtype=complex)
            one_body[beta, alpha] = 1.0
            operator = kron(
                csc_matrix(one_body),
                site_identity,
                format="csc",
            )
            operator_row.append(operator)
            observable_row.append(
                kron(operator, phonon_identity, format="csc")
            )
        electron_operators.append(operator_row)
        electron_observables.append(observable_row)

    full_annihilation = tuple(
        kron(electron_identity, operator, format="csc")
        for operator in local_annihilation
    )
    center_phonon_annihilation = (
        full_annihilation[0] + full_annihilation[1]
    ) / sqrt(2.0)
    relative_phonon_annihilation = (
        full_annihilation[0] - full_annihilation[1]
    ) / sqrt(2.0)
    relative_position = (
        relative_phonon_annihilation
        + relative_phonon_annihilation.getH()
    ) / sqrt(2.0)
    relative_momentum = (
        relative_phonon_annihilation
        - relative_phonon_annihilation.getH()
    ) / (1j * sqrt(2.0))
    spin_pauli_observables = tuple(
        tuple(
            kron(
                (
                    kron(pauli, site_identity, format="csc")
                    if spin == 0
                    else kron(site_identity, pauli, format="csc")
                ),
                phonon_identity,
                format="csc",
            )
            for pauli in pauli_matrices
        )
        for spin in range(2)
    )
    normal_observables = tuple(
        tuple(
            full_annihilation[q_prime].getH() @ full_annihilation[q]
            for q_prime in range(2)
        )
        for q in range(2)
    )
    anomalous_observables = tuple(
        tuple(
            full_annihilation[q_prime] @ full_annihilation[q]
            for q in range(2)
        )
        for q_prime in range(2)
    )
    electron_phonon_observables = tuple(
        tuple(
            tuple(
                kron(
                    electron_operators[alpha][beta],
                    local_annihilation[q],
                    format="csc",
                )
                for beta in range(2)
            )
            for alpha in range(2)
        )
        for q in range(2)
    )
    spin_down_site_occupations = tuple(
        kron(
            kron(
                site_identity,
                site_projectors[site],
                format="csc",
            ),
            phonon_identity,
            format="csc",
        )
        for site in range(2)
    )

    return _ExactDimerModel(
        static_hamiltonian=static_hamiltonian,
        drive_operator=drive_operator,
        electron_observables=tuple(
            tuple(row) for row in electron_observables
        ),
        phonon_annihilation=full_annihilation,
        normal_phonon_observables=normal_observables,
        anomalous_phonon_observables=anomalous_observables,
        electron_phonon_observables=electron_phonon_observables,
        spin_down_site_occupations=spin_down_site_occupations,
        spin_pauli_observables=spin_pauli_observables,
        center_phonon_annihilation=center_phonon_annihilation,
        relative_phonon_annihilation=relative_phonon_annihilation,
        relative_position=relative_position,
        relative_momentum=relative_momentum,
    )


def _ground_state(
    model: _ExactDimerModel,
    *,
    eigensolver_tolerance: float,
) -> tuple[float, ComplexArray]:
    if eigensolver_tolerance <= 0.0:
        raise ValueError("eigensolver_tolerance must be positive")

    dimension = model.static_hamiltonian.shape[0]
    starting_vector = np.linspace(1.0, 2.0, dimension, dtype=float)
    starting_vector /= np.linalg.norm(starting_vector)
    eigenvalues, eigenvectors = eigsh(
        model.static_hamiltonian,
        k=1,
        which="SA",
        tol=eigensolver_tolerance,
        v0=starting_vector,
    )
    energy = float(eigenvalues[0].real)
    state = np.asarray(eigenvectors[:, 0], dtype=complex)
    state /= sqrt(float(np.vdot(state, state).real))
    return energy, state


def _contract_matrix_state(
    model: _ExactDimerModel,
    state_vector: ComplexArray,
) -> MatrixDimerState:
    electron_density = np.empty((2, 2), dtype=complex)
    for alpha in range(2):
        for beta in range(2):
            electron_density[alpha, beta] = _expectation(
                state_vector,
                model.electron_observables[alpha][beta],
            )

    coherent_phonon = np.array(
        [
            _expectation(state_vector, operator)
            for operator in model.phonon_annihilation
        ],
        dtype=complex,
    )
    phonon_density = np.empty((2, 2), dtype=complex)
    anomalous_density = np.empty((2, 2), dtype=complex)
    for q in range(2):
        for q_prime in range(2):
            phonon_density[q, q_prime] = (
                _expectation(
                    state_vector,
                    model.normal_phonon_observables[q][q_prime],
                )
                - coherent_phonon[q] * coherent_phonon[q_prime].conjugate()
            )
            anomalous_density[q_prime, q] = (
                _expectation(
                    state_vector,
                    model.anomalous_phonon_observables[q_prime][q],
                )
                - coherent_phonon[q_prime] * coherent_phonon[q]
            )

    electron_phonon_correlation = np.empty((2, 2, 2), dtype=complex)
    for q in range(2):
        for alpha in range(2):
            for beta in range(2):
                electron_phonon_correlation[q, alpha, beta] = (
                    _expectation(
                        state_vector,
                        model.electron_phonon_observables[q][alpha][beta],
                    )
                    - electron_density[alpha, beta] * coherent_phonon[q]
                )

    return MatrixDimerState(
        electron_density=electron_density,
        coherent_phonon=coherent_phonon,
        phonon_density=phonon_density,
        anomalous_phonon_density=anomalous_density,
        electron_phonon_correlation=electron_phonon_correlation,
    )


def _contract_matrix_derivative(
    model: _ExactDimerModel,
    state_vector: ComplexArray,
    state_derivative: ComplexArray,
    matrix_state: MatrixDimerState,
) -> MatrixDimerState:
    """Contract the exact Schrödinger velocity into moment velocities."""

    electron_derivative = np.empty((2, 2), dtype=complex)
    for alpha in range(2):
        for beta in range(2):
            electron_derivative[alpha, beta] = _expectation_derivative(
                state_vector,
                state_derivative,
                model.electron_observables[alpha][beta],
            )

    coherent_derivative = np.array(
        [
            _expectation_derivative(
                state_vector,
                state_derivative,
                operator,
            )
            for operator in model.phonon_annihilation
        ],
        dtype=complex,
    )
    coherent = matrix_state.coherent_phonon

    phonon_derivative = np.empty((2, 2), dtype=complex)
    anomalous_derivative = np.empty((2, 2), dtype=complex)
    for q in range(2):
        for q_prime in range(2):
            phonon_derivative[q, q_prime] = (
                _expectation_derivative(
                    state_vector,
                    state_derivative,
                    model.normal_phonon_observables[q][q_prime],
                )
                - coherent_derivative[q] * coherent[q_prime].conjugate()
                - coherent[q] * coherent_derivative[q_prime].conjugate()
            )
            anomalous_derivative[q_prime, q] = (
                _expectation_derivative(
                    state_vector,
                    state_derivative,
                    model.anomalous_phonon_observables[q_prime][q],
                )
                - coherent_derivative[q_prime] * coherent[q]
                - coherent[q_prime] * coherent_derivative[q]
            )

    correlation_derivative = np.empty((2, 2, 2), dtype=complex)
    electron = matrix_state.electron_density
    for q in range(2):
        for alpha in range(2):
            for beta in range(2):
                correlation_derivative[q, alpha, beta] = (
                    _expectation_derivative(
                        state_vector,
                        state_derivative,
                        model.electron_phonon_observables[q][alpha][beta],
                    )
                    - electron_derivative[alpha, beta] * coherent[q]
                    - electron[alpha, beta] * coherent_derivative[q]
                )

    return MatrixDimerState(
        electron_density=electron_derivative,
        coherent_phonon=coherent_derivative,
        phonon_density=phonon_derivative,
        anomalous_phonon_density=anomalous_derivative,
        electron_phonon_correlation=correlation_derivative,
    )


def exact_holstein_ground_state(
    parameters: DimerParameters,
    *,
    phonon_cutoff: int = 16,
    eigensolver_tolerance: float = 1e-12,
) -> ExactGroundState:
    """Return exact ground-state contractions for the undriven dimer.

    ``phonon_cutoff`` is the largest allowed occupation of each local phonon
    mode, so each oscillator has dimension ``phonon_cutoff + 1``.
    """

    model = _build_exact_dimer_model(
        parameters,
        phonon_cutoff=phonon_cutoff,
    )
    energy, ground_state = _ground_state(
        model,
        eigensolver_tolerance=eigensolver_tolerance,
    )
    return ExactGroundState(
        energy=energy,
        phonon_cutoff=phonon_cutoff,
        matrix_state=_contract_matrix_state(model, ground_state),
    )


def exact_holstein_joint_moment_initial_state(
    parameters: DimerParameters,
    *,
    hierarchy: MomentHierarchy,
    phonon_cutoff: int = 16,
    eigensolver_tolerance: float = 1e-12,
    canonical_embedding: bool = False,
) -> ExactJointMomentInitialState:
    """Contract one ground-state vector into both requested representations.

    A single eigensolver call is essential when correlated matrix and hidden
    moment coordinates are combined: independently diagonalized nearly
    degenerate preparations need not select the same physical representative.
    ``canonical_embedding=True`` first normal-orders each requested Weyl word
    in the canonical CCR algebra and only then embeds that polynomial in the
    finite scorer space.  APCM initialization requires this ordering.
    """

    model = _build_exact_dimer_model(
        parameters,
        phonon_cutoff=phonon_cutoff,
    )
    energy, ground_state = _ground_state(
        model,
        eigensolver_tolerance=eigensolver_tolerance,
    )
    state_derivative = -1j * (model.static_hamiltonian @ ground_state)
    observables = _moment_hierarchy_observables(
        model,
        hierarchy,
        canonical_embedding=canonical_embedding,
    )
    hierarchy_coordinates, _ = _contract_moment_hierarchy_coordinates(
        model,
        ground_state,
        state_derivative,
        observables,
        hierarchy,
    )
    return ExactJointMomentInitialState(
        energy=energy,
        phonon_cutoff=phonon_cutoff,
        matrix_state=_contract_matrix_state(model, ground_state),
        hierarchy_coordinates=hierarchy_coordinates,
    )


def _propagate_exact_wavefunctions(
    parameters: DimerParameters,
    *,
    sample_times: FloatArray,
    phonon_cutoff: int,
    eigensolver_tolerance: float,
    relative_tolerance: float,
    absolute_tolerance: float,
    maximum_step: float,
) -> _ExactWavefunctionTrajectory:
    """Propagate the truncated wavefunction behind exact reporting adapters."""

    times = np.asarray(sample_times, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("sample_times must contain at least two times")
    if abs(float(times[0])) > 1e-15:
        raise ValueError("sample_times must start at zero")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("sample_times must be strictly increasing")
    if relative_tolerance <= 0.0 or absolute_tolerance <= 0.0:
        raise ValueError("integration tolerances must be positive")
    if maximum_step <= 0.0:
        raise ValueError("maximum_step must be positive")

    model = _build_exact_dimer_model(
        parameters,
        phonon_cutoff=phonon_cutoff,
    )
    _, initial_state = _ground_state(
        model,
        eigensolver_tolerance=eigensolver_tolerance,
    )

    def rhs(time: float, state: ComplexArray) -> ComplexArray:
        return -1j * (
            model.static_hamiltonian @ state
            + parameters.drive_difference(time)
            * (model.drive_operator @ state)
        )

    solution = solve_ivp(
        rhs,
        (float(times[0]), float(times[-1])),
        initial_state,
        method="DOP853",
        t_eval=times,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        max_step=maximum_step,
    )
    if not solution.success or solution.y.shape[1] != times.size:
        raise RuntimeError(f"exact driven propagation failed: {solution.message}")

    state_vectors = np.asarray(solution.y, dtype=complex)
    state_derivatives = np.column_stack(
        [
            rhs(float(solution.t[index]), state_vectors[:, index])
            for index in range(times.size)
        ]
    )
    return _ExactWavefunctionTrajectory(
        model=model,
        times=np.asarray(solution.t, dtype=float),
        state_vectors=state_vectors,
        state_derivatives=np.asarray(state_derivatives, dtype=complex),
        phonon_cutoff=phonon_cutoff,
        function_evaluations=int(solution.nfev),
        success=bool(solution.success),
        message=str(solution.message),
    )


def _contract_exact_driven_trajectory(
    wavefunctions: _ExactWavefunctionTrajectory,
) -> ExactDrivenTrajectory:
    """Contract a private wavefunction trajectory into retained moments."""

    sample_count = wavefunctions.times.size
    state_norms = np.asarray(
        [
            float(
                np.vdot(
                    wavefunctions.state_vectors[:, index],
                    wavefunctions.state_vectors[:, index],
                ).real
            )
            for index in range(sample_count)
        ],
        dtype=float,
    )
    matrix_states = tuple(
        _contract_matrix_state(
            wavefunctions.model,
            wavefunctions.state_vectors[:, index],
        )
        for index in range(sample_count)
    )
    matrix_derivatives = tuple(
        _contract_matrix_derivative(
            wavefunctions.model,
            wavefunctions.state_vectors[:, index],
            wavefunctions.state_derivatives[:, index],
            matrix_states[index],
        )
        for index in range(sample_count)
    )
    return ExactDrivenTrajectory(
        times=wavefunctions.times.copy(),
        matrix_states=matrix_states,
        matrix_derivatives=matrix_derivatives,
        state_norms=state_norms,
        phonon_cutoff=wavefunctions.phonon_cutoff,
        function_evaluations=wavefunctions.function_evaluations,
        success=wavefunctions.success,
        message=wavefunctions.message,
    )


def exact_holstein_driven_trajectory(
    parameters: DimerParameters,
    *,
    sample_times: FloatArray,
    phonon_cutoff: int = 16,
    eigensolver_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-11,
    maximum_step: float = 0.05,
) -> ExactDrivenTrajectory:
    """Propagate Eqs. (16)--(19) and contract each requested state.

    The initial wavefunction is the exact ground state of the zero-field
    truncated Hamiltonian.  ``phonon_cutoff`` is the maximum occupation of
    each local mode; it changes the exact Hilbert space, not the dimension of
    the 31-coordinate archive closure used for later comparison.
    """

    wavefunctions = _propagate_exact_wavefunctions(
        parameters,
        sample_times=sample_times,
        phonon_cutoff=phonon_cutoff,
        eigensolver_tolerance=eigensolver_tolerance,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step=maximum_step,
    )
    return _contract_exact_driven_trajectory(wavefunctions)


def exact_holstein_wavefunction_trajectory_for_diagnostics(
    parameters: DimerParameters,
    *,
    sample_times: FloatArray,
    phonon_cutoff: int = 16,
    eigensolver_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-11,
    maximum_step: float = 0.05,
) -> ExactDiagnosticWavefunctionTrajectory:
    """Return exact states for an offline compression or velocity gate.

    This reporting adapter deliberately omits the Hamiltonian model object so
    an autonomous right-hand side cannot accidentally use the exact reference
    as an online correction signal.
    """

    wavefunctions = _propagate_exact_wavefunctions(
        parameters,
        sample_times=sample_times,
        phonon_cutoff=phonon_cutoff,
        eigensolver_tolerance=eigensolver_tolerance,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step=maximum_step,
    )
    return ExactDiagnosticWavefunctionTrajectory(
        times=wavefunctions.times.copy(),
        state_vectors=wavefunctions.state_vectors.copy(),
        state_derivatives=wavefunctions.state_derivatives.copy(),
        phonon_cutoff=wavefunctions.phonon_cutoff,
        function_evaluations=wavefunctions.function_evaluations,
        success=wavefunctions.success,
        message=wavefunctions.message,
    )


def _relative_weyl_observables(
    model: _ExactDimerModel,
    moment_keys: tuple[MomentKey, ...],
) -> dict[tuple[int, int], csc_matrix]:
    """Build every fully symmetrized ``x^a p^b`` operator requested."""

    dimension = model.static_hamiltonian.shape[0]
    identity = eye(dimension, format="csc", dtype=complex)
    observables: dict[tuple[int, int], csc_matrix] = {(0, 0): identity}
    required_powers = {
        (key.x_power, key.p_power) for key in moment_keys
    }
    for x_power, p_power in sorted(required_powers):
        degree = x_power + p_power
        if degree == 0:
            continue
        accumulator = csc_matrix((dimension, dimension), dtype=complex)
        for x_positions in combinations(range(degree), x_power):
            x_position_set = set(x_positions)
            product = identity
            for position in range(degree):
                product = product @ (
                    model.relative_position
                    if position in x_position_set
                    else model.relative_momentum
                )
            accumulator += product
        observables[(x_power, p_power)] = accumulator / comb(
            degree, x_power
        )
    return observables


@lru_cache(maxsize=None)
def _normal_ordered_weyl_coefficients(
    x_power: int,
    p_power: int,
) -> tuple[tuple[int, int, complex], ...]:
    r"""Return the canonical normal ordering of one Weyl monomial.

    With ``x=(a+a^dagger)/sqrt(2)`` and
    ``p=i(a^dagger-a)/sqrt(2)``, the generating function is

    ``exp(s x + t p) = exp(alpha a^dagger) exp(beta a)``
    ``* exp((s^2+t^2)/4)``.

    Extracting the ``s**x_power * t**p_power`` coefficient therefore
    performs canonical-CCR reduction before any finite-cutoff matrices are
    multiplied.  This is the preparation embedding required by APCM; it is
    deliberately distinct from multiplying truncated quadrature matrices.
    """

    if x_power < 0 or p_power < 0:
        raise ValueError("Weyl powers must be nonnegative")
    total_degree = x_power + p_power
    derivative_factor = factorial(x_power) * factorial(p_power)
    coefficients: dict[tuple[int, int], complex] = {}
    root_two = sqrt(2.0)
    for creator_power in range(total_degree + 1):
        for annihilator_power in range(total_degree + 1 - creator_power):
            remaining = total_degree - creator_power - annihilator_power
            if remaining % 2:
                continue
            for gaussian_x_order in range(remaining // 2 + 1):
                gaussian_p_order = remaining // 2 - gaussian_x_order
                gaussian = 1.0 / (
                    4.0 ** (gaussian_x_order + gaussian_p_order)
                    * factorial(gaussian_x_order)
                    * factorial(gaussian_p_order)
                )
                for creator_t_power in range(creator_power + 1):
                    creator_s_power = creator_power - creator_t_power
                    creator = (
                        comb(creator_power, creator_t_power)
                        * (1.0j**creator_t_power)
                        / (root_two**creator_power * factorial(creator_power))
                    )
                    for annihilator_t_power in range(
                        annihilator_power + 1
                    ):
                        annihilator_s_power = (
                            annihilator_power - annihilator_t_power
                        )
                        if (
                            creator_s_power
                            + annihilator_s_power
                            + 2 * gaussian_x_order
                            != x_power
                            or creator_t_power
                            + annihilator_t_power
                            + 2 * gaussian_p_order
                            != p_power
                        ):
                            continue
                        annihilator = (
                            comb(annihilator_power, annihilator_t_power)
                            * ((-1.0j) ** annihilator_t_power)
                            / (
                                root_two**annihilator_power
                                * factorial(annihilator_power)
                            )
                        )
                        key = (creator_power, annihilator_power)
                        coefficients[key] = coefficients.get(key, 0.0j) + (
                            derivative_factor
                            * creator
                            * annihilator
                            * gaussian
                        )
    return tuple(
        (creator, annihilator, coefficient)
        for (creator, annihilator), coefficient in sorted(coefficients.items())
        if abs(coefficient) > 1e-15
    )


def _canonical_embedded_relative_weyl_observables(
    model: _ExactDimerModel,
    moment_keys: tuple[MomentKey, ...],
) -> dict[tuple[int, int], csc_matrix]:
    """Embed canonical-normal-ordered Weyl words in the cutoff space."""

    dimension = model.static_hamiltonian.shape[0]
    identity = eye(dimension, format="csc", dtype=complex)
    requested = {
        (key.x_power, key.p_power) for key in moment_keys
    }
    maximum_degree = max(
        (x_power + p_power for x_power, p_power in requested),
        default=0,
    )
    annihilation = model.relative_phonon_annihilation
    creation = annihilation.getH()
    annihilation_powers = {0: identity}
    creation_powers = {0: identity}
    for power in range(1, maximum_degree + 1):
        annihilation_powers[power] = (
            annihilation_powers[power - 1] @ annihilation
        ).tocsc()
        creation_powers[power] = (
            creation_powers[power - 1] @ creation
        ).tocsc()

    observables: dict[tuple[int, int], csc_matrix] = {}
    for powers in sorted(requested):
        accumulator = csc_matrix((dimension, dimension), dtype=complex)
        for creator_power, annihilator_power, coefficient in (
            _normal_ordered_weyl_coefficients(*powers)
        ):
            accumulator += coefficient * (
                creation_powers[creator_power]
                @ annihilation_powers[annihilator_power]
            )
        observables[powers] = accumulator.tocsc()
    return observables


def _moment_hierarchy_observables(
    model: _ExactDimerModel,
    hierarchy: MomentHierarchy,
    *,
    canonical_embedding: bool = False,
) -> tuple[csc_matrix, ...]:
    """Build exact Hermitian observables for one hierarchy basis."""

    label_index = {
        label: index for index, label in enumerate(PAULI_LABELS)
    }
    weyl = (
        _canonical_embedded_relative_weyl_observables(
            model,
            hierarchy.moment_keys,
        )
        if canonical_embedding
        else _relative_weyl_observables(model, hierarchy.moment_keys)
    )
    observables: list[csc_matrix] = []
    for key in hierarchy.moment_keys:
        up_index = label_index[key.spin_up]
        down_index = label_index[key.spin_down]
        oriented = (
            model.spin_pauli_observables[0][up_index]
            @ model.spin_pauli_observables[1][down_index]
        )
        if key.spin_up != key.spin_down:
            swapped = (
                model.spin_pauli_observables[0][down_index]
                @ model.spin_pauli_observables[1][up_index]
            )
            oriented = 0.5 * (oriented + swapped)
        observables.append(
            oriented @ weyl[(key.x_power, key.p_power)]
        )
    return tuple(observables)


def _contract_moment_hierarchy_coordinates(
    model: _ExactDimerModel,
    state_vector: ComplexArray,
    state_derivative: ComplexArray,
    observables: tuple[csc_matrix, ...],
    hierarchy: MomentHierarchy,
) -> tuple[FloatArray, FloatArray]:
    center = _expectation(state_vector, model.center_phonon_annihilation)
    center_derivative = _expectation_derivative(
        state_vector,
        state_derivative,
        model.center_phonon_annihilation,
    )
    moments: dict[MomentKey, float] = {}
    derivatives: dict[MomentKey, float] = {}
    for key, operator in zip(
        hierarchy.moment_keys,
        observables,
        strict=True,
    ):
        value = _expectation(state_vector, operator)
        derivative = _expectation_derivative(
            state_vector,
            state_derivative,
            operator,
        )
        if abs(value.imag) > 2e-9 or abs(derivative.imag) > 2e-8:
            raise FloatingPointError(
                f"exact Hermitian contraction became complex for {key}: "
                f"value={value}, derivative={derivative}"
            )
        moments[key] = float(value.real)
        derivatives[key] = float(derivative.real)
    return (
        hierarchy.pack(center, moments),
        hierarchy.pack(center_derivative, derivatives),
    )


def exact_holstein_moment_hierarchy_trajectory(
    parameters: DimerParameters,
    *,
    hierarchy: MomentHierarchy,
    sample_times: FloatArray,
    phonon_cutoff: int = 16,
    eigensolver_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-11,
    maximum_step: float = 0.05,
) -> ExactMomentHierarchyTrajectory:
    """Contract exact propagation into a declared moment hierarchy.

    This offline adapter is absent from every autonomous hierarchy RHS.
    """

    wavefunctions = _propagate_exact_wavefunctions(
        parameters,
        sample_times=sample_times,
        phonon_cutoff=phonon_cutoff,
        eigensolver_tolerance=eigensolver_tolerance,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step=maximum_step,
    )
    observables = _moment_hierarchy_observables(
        wavefunctions.model,
        hierarchy,
    )
    coordinates = np.empty(
        (wavefunctions.times.size, hierarchy.coordinate_count),
        dtype=float,
    )
    derivatives = np.empty_like(coordinates)
    state_norms = np.empty(wavefunctions.times.size, dtype=float)
    for index in range(wavefunctions.times.size):
        coordinates[index], derivatives[index] = (
            _contract_moment_hierarchy_coordinates(
                wavefunctions.model,
                wavefunctions.state_vectors[:, index],
                wavefunctions.state_derivatives[:, index],
                observables,
                hierarchy,
            )
        )
        state_norms[index] = float(
            np.vdot(
                wavefunctions.state_vectors[:, index],
                wavefunctions.state_vectors[:, index],
            ).real
        )
    return ExactMomentHierarchyTrajectory(
        maximum_degree=hierarchy.maximum_degree,
        times=wavefunctions.times.copy(),
        coordinates=coordinates,
        coordinate_derivatives=derivatives,
        state_norms=state_norms,
        phonon_cutoff=wavefunctions.phonon_cutoff,
        function_evaluations=wavefunctions.function_evaluations,
        success=wavefunctions.success,
        message=wavefunctions.message,
    )


def exact_holstein_third_cumulant_trajectory(
    parameters: DimerParameters,
    *,
    sample_times: FloatArray,
    phonon_cutoff: int = 16,
    eigensolver_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-11,
    maximum_step: float = 0.05,
) -> ExactThirdCumulantTrajectory:
    """Contract exact propagation into the 47-coordinate hierarchy."""

    return exact_holstein_moment_hierarchy_trajectory(
        parameters,
        hierarchy=THIRD_ORDER_HIERARCHY,
        sample_times=sample_times,
        phonon_cutoff=phonon_cutoff,
        eigensolver_tolerance=eigensolver_tolerance,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step=maximum_step,
    )


def exact_holstein_fourth_cumulant_trajectory(
    parameters: DimerParameters,
    *,
    sample_times: FloatArray,
    phonon_cutoff: int = 16,
    eigensolver_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-11,
    maximum_step: float = 0.05,
) -> ExactFourthCumulantTrajectory:
    """Contract exact propagation into the 82-coordinate hierarchy."""

    return exact_holstein_moment_hierarchy_trajectory(
        parameters,
        hierarchy=FOURTH_ORDER_HIERARCHY,
        sample_times=sample_times,
        phonon_cutoff=phonon_cutoff,
        eigensolver_tolerance=eigensolver_tolerance,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step=maximum_step,
    )


def exact_holstein_correlation_closure_trajectory(
    parameters: DimerParameters,
    *,
    sample_times: FloatArray,
    phonon_cutoff: int = 16,
    eigensolver_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-11,
    maximum_step: float = 0.05,
) -> ExactCorrelationClosureTrajectory:
    """Audit the mixed-moment and opposite-spin omissions in Eq. (14d).

    This is an exact-reference reporting adapter.  It does not define an
    online closure: exact wavefunctions are used only to measure the two
    source blocks absent from the autonomous 31-coordinate equations.

    With ``O_ij = c_j^dagger c_i``, the exact centered equation contains

    ``Q[q,r,i,j] = <[O_ij,n_{r up}] delta_X_r delta_b_q>``

    and ``Cov(O_ij,n_{q down})``.  The algebra-preserving factorization of
    ``Q`` is ``<[O_ij,n_r]> <delta_X_r delta_b_q>``.  Archive Eq. (14d) also
    factorizes the Pauli-blocking products separately; this does not preserve
    ``[n_i,n_r] = 0`` for a non-idempotent ``rho`` and is audited as its own
    velocity correction.  The opposite-spin covariance is set to zero.  The
    exact same-spin covariance requires no extra coordinate in this sector:
    the one-up-electron algebra reconstructs it from ``rho``.
    """

    coupling = parameters.coupling
    if coupling <= 0.0:
        raise ValueError(
            "the mixed-moment decomposition requires positive coupling"
        )

    wavefunctions = _propagate_exact_wavefunctions(
        parameters,
        sample_times=sample_times,
        phonon_cutoff=phonon_cutoff,
        eigensolver_tolerance=eigensolver_tolerance,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step=maximum_step,
    )
    exact_trajectory = _contract_exact_driven_trajectory(wavefunctions)
    sample_count = exact_trajectory.times.size

    exact_mixed_moment = np.empty(
        (sample_count, 2, 2, 2, 2),
        dtype=complex,
    )
    factorized_mixed_moment = np.empty_like(exact_mixed_moment)
    same_spin_covariance = np.empty(
        (sample_count, 2, 2, 2),
        dtype=complex,
    )
    archive_same_spin_covariance = np.empty_like(same_spin_covariance)
    opposite_spin_covariance = np.empty_like(same_spin_covariance)
    archive_correlation_derivatives = np.empty_like(same_spin_covariance)
    mixed_moment_velocity_corrections = np.empty_like(
        same_spin_covariance
    )
    same_spin_pauli_velocity_corrections = np.empty_like(
        same_spin_covariance
    )
    opposite_spin_velocity_corrections = np.empty_like(
        same_spin_covariance
    )
    cutoff_velocity_remainders = np.empty_like(same_spin_covariance)

    full_identity = eye(
        wavefunctions.model.static_hamiltonian.shape[0],
        format="csc",
        dtype=complex,
    )
    spin_up_site_occupations = tuple(
        wavefunctions.model.electron_observables[site][site]
        for site in range(2)
    )
    commutators = tuple(
        tuple(
            tuple(
                (
                    wavefunctions.model.electron_observables[i][j]
                    @ spin_up_site_occupations[site]
                    - spin_up_site_occupations[site]
                    @ wavefunctions.model.electron_observables[i][j]
                )
                for j in range(2)
            )
            for i in range(2)
        )
        for site in range(2)
    )

    for time_index, time in enumerate(exact_trajectory.times):
        state_vector = wavefunctions.state_vectors[:, time_index]
        matrix_state = exact_trajectory.matrix_states[time_index]
        rho = matrix_state.electron_density
        centered_annihilation = tuple(
            wavefunctions.model.phonon_annihilation[q]
            - matrix_state.coherent_phonon[q] * full_identity
            for q in range(2)
        )

        for q in range(2):
            for site in range(2):
                centered_displacement = (
                    centered_annihilation[site]
                    + centered_annihilation[site].getH()
                )
                displaced_state = centered_displacement @ (
                    centered_annihilation[q] @ state_vector
                )
                for i in range(2):
                    for j in range(2):
                        exact_mixed_moment[
                            time_index, q, site, i, j
                        ] = np.vdot(
                            state_vector,
                            commutators[site][i][j] @ displaced_state,
                        )

                        commutator_mean = (
                            (rho[site, j] if i == site else 0.0)
                            - (rho[i, site] if j == site else 0.0)
                        )
                        phonon_pair = (
                            matrix_state.anomalous_phonon_density[site, q]
                            + matrix_state.phonon_density[q, site]
                        )
                        factorized_mixed_moment[
                            time_index, q, site, i, j
                        ] = commutator_mean * phonon_pair

        same_spin_covariance[time_index] = same_spin_density_covariance(
            rho
        )
        for q in range(2):
            down_occupation = (
                wavefunctions.model.spin_down_site_occupations[q]
            )
            down_mean = _expectation(state_vector, down_occupation)
            for i in range(2):
                for j in range(2):
                    electron_operator = (
                        wavefunctions.model.electron_observables[i][j]
                    )
                    opposite_spin_covariance[
                        time_index, q, i, j
                    ] = (
                        _expectation(
                            state_vector,
                            electron_operator @ down_occupation,
                        )
                        - rho[i, j] * down_mean
                    )

        archive_source_by_mode = correlation_source_by_phonon_mode(
            matrix_state,
            parameters,
        )
        archive_source = np.sum(archive_source_by_mode, axis=1)
        factorized_mixed_velocity = -1j * coupling * np.sum(
            factorized_mixed_moment[time_index],
            axis=1,
        )
        archive_pauli_velocity = (
            archive_source - factorized_mixed_velocity
        )
        archive_same_spin_covariance[time_index] = (
            archive_pauli_velocity / (-1j * coupling)
        )

        archive_derivative = matrix_dimer_rhs(
            float(time),
            matrix_state,
            parameters,
        ).electron_phonon_correlation
        archive_correlation_derivatives[time_index] = archive_derivative
        mixed_correction = -1j * coupling * np.sum(
            exact_mixed_moment[time_index]
            - factorized_mixed_moment[time_index],
            axis=1,
        )
        same_spin_pauli_correction = (
            -1j * coupling * same_spin_covariance[time_index]
            - archive_pauli_velocity
        )
        opposite_spin_correction = (
            -1j * coupling * opposite_spin_covariance[time_index]
        )
        mixed_moment_velocity_corrections[time_index] = mixed_correction
        same_spin_pauli_velocity_corrections[
            time_index
        ] = same_spin_pauli_correction
        opposite_spin_velocity_corrections[
            time_index
        ] = opposite_spin_correction
        cutoff_velocity_remainders[time_index] = (
            exact_trajectory.matrix_derivatives[
                time_index
            ].electron_phonon_correlation
            - archive_derivative
            - mixed_correction
            - same_spin_pauli_correction
            - opposite_spin_correction
        )

    return ExactCorrelationClosureTrajectory(
        exact_trajectory=exact_trajectory,
        exact_mixed_moment=exact_mixed_moment,
        factorized_mixed_moment=factorized_mixed_moment,
        same_spin_covariance=same_spin_covariance,
        archive_same_spin_covariance=archive_same_spin_covariance,
        opposite_spin_covariance=opposite_spin_covariance,
        archive_correlation_derivatives=archive_correlation_derivatives,
        mixed_moment_velocity_corrections=(
            mixed_moment_velocity_corrections
        ),
        same_spin_pauli_velocity_corrections=(
            same_spin_pauli_velocity_corrections
        ),
        opposite_spin_velocity_corrections=(
            opposite_spin_velocity_corrections
        ),
        cutoff_velocity_remainders=cutoff_velocity_remainders,
    )


def _closed_protocol_rhs(
    protocol: ClosedProtocol,
    parameters: DimerParameters,
    initial_coordinates: FloatArray,
    *,
    activation_margin: float,
    target_flux: float,
    barrier_rate: float,
    energy_neutral: bool,
    require_correction_convergence: bool,
) -> RhsFunction:
    if protocol == "archive":
        return lambda time, state: closed_scalar_rhs(time, state, parameters)
    if protocol == "residual_subtracted":
        undriven = replace(parameters, drive_amplitude=0.0)
        residual = closed_scalar_rhs(
            0.0,
            initial_coordinates,
            undriven,
        )
        return lambda time, state: (
            closed_scalar_rhs(time, state, parameters) - residual
        )
    if protocol in (
        "joint_barrier",
        "residual_subtracted_joint_barrier",
    ):
        return closed_joint_cone_projected_rhs(
            parameters,
            initial_coordinates,
            activation_margin=activation_margin,
            target_flux=target_flux,
            barrier_rate=barrier_rate,
            energy_neutral=energy_neutral,
            subtract_initial_residual=(
                protocol == "residual_subtracted_joint_barrier"
            ),
            require_convergence=require_correction_convergence,
        )
    raise ValueError(
        f"unknown closed protocol {protocol!r}; expected one of "
        f"{CLOSED_PROTOCOLS}"
    )


def compare_exact_and_closed_protocols(
    parameters: DimerParameters,
    *,
    sample_times: FloatArray,
    phonon_cutoff: int = 16,
    protocols: tuple[ClosedProtocol, ...] = CLOSED_PROTOCOLS,
    eigensolver_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-11,
    maximum_step: float = 0.05,
    activation_margin: float = 0.0,
    target_flux: float = 0.0,
    barrier_rate: float = 5.0,
    energy_neutral: bool = True,
    require_correction_convergence: bool = True,
) -> dict[ClosedProtocol, ExactClosedProtocolComparison]:
    """Compare exact contractions with selected 31-coordinate protocols.

    Both trajectories begin from the same exact zero-field ground state.  The
    exact wavefunction follows Eqs. (16)--(19); the approximate moments follow
    the complete 31-coordinate representation of Eqs. (14a)--(14e), optionally
    with residual subtraction and/or the joint physicality barrier.
    """

    if not protocols:
        raise ValueError("protocols must contain at least one protocol")
    if len(set(protocols)) != len(protocols):
        raise ValueError("protocols must not contain duplicates")
    unknown = tuple(
        protocol for protocol in protocols if protocol not in CLOSED_PROTOCOLS
    )
    if unknown:
        raise ValueError(
            f"unknown closed protocols {unknown}; expected members of "
            f"{CLOSED_PROTOCOLS}"
        )

    exact = exact_holstein_driven_trajectory(
        parameters,
        sample_times=sample_times,
        phonon_cutoff=phonon_cutoff,
        eigensolver_tolerance=eigensolver_tolerance,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step=maximum_step,
    )
    exact_coordinates = np.asarray(
        [
            matrix_state_to_closed_scalar_coordinates(state)
            for state in exact.matrix_states
        ],
        dtype=float,
    )
    initial_coordinates = exact_coordinates[0].copy()
    block_names = (
        "electron_density",
        "coherent_phonon",
        "normal_phonon",
        "anomalous_phonon",
        "electron_phonon_correlation",
    )
    block_slices = (
        slice(0, 3),
        slice(3, 7),
        slice(7, 11),
        slice(11, 17),
        slice(17, 31),
    )

    comparisons: dict[ClosedProtocol, ExactClosedProtocolComparison] = {}
    for protocol in protocols:
        rhs = _closed_protocol_rhs(
            protocol,
            parameters,
            initial_coordinates,
            activation_margin=activation_margin,
            target_flux=target_flux,
            barrier_rate=barrier_rate,
            energy_neutral=energy_neutral,
            require_correction_convergence=require_correction_convergence,
        )
        solution = solve_ivp(
            rhs,
            (float(exact.times[0]), float(exact.times[-1])),
            initial_coordinates,
            method="DOP853",
            t_eval=exact.times,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            max_step=maximum_step,
        )
        if not solution.success or solution.y.shape[1] != exact.times.size:
            raise RuntimeError(
                f"{protocol} propagation failed: {solution.message}"
            )

        closed_coordinates = np.asarray(solution.y.T, dtype=float)
        coordinate_errors = closed_coordinates - exact_coordinates
        block_error_norms = np.column_stack(
            [
                np.linalg.norm(coordinate_errors[:, block], axis=1)
                for block in block_slices
            ]
        )
        comparisons[protocol] = ExactClosedProtocolComparison(
            protocol=protocol,
            times=exact.times.copy(),
            exact_coordinates=exact_coordinates,
            closed_coordinates=closed_coordinates,
            coordinate_errors=coordinate_errors,
            block_names=block_names,
            block_error_norms=np.asarray(block_error_norms, dtype=float),
            exact_trajectory=exact,
            closed_function_evaluations=int(solution.nfev),
            success=bool(solution.success),
            message=str(solution.message),
        )
    return comparisons


def compare_exact_and_archive_closure(
    parameters: DimerParameters,
    *,
    sample_times: FloatArray,
    phonon_cutoff: int = 16,
    eigensolver_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-11,
    maximum_step: float = 0.05,
) -> ExactArchiveClosureComparison:
    """Compare exact contractions with the unmodified archive closure."""

    comparison = compare_exact_and_closed_protocols(
        parameters,
        sample_times=sample_times,
        phonon_cutoff=phonon_cutoff,
        protocols=("archive",),
        eigensolver_tolerance=eigensolver_tolerance,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step=maximum_step,
    )["archive"]
    return ExactArchiveClosureComparison(
        times=comparison.times,
        exact_coordinates=comparison.exact_coordinates,
        archive_coordinates=comparison.closed_coordinates,
        coordinate_errors=comparison.coordinate_errors,
        block_names=comparison.block_names,
        block_error_norms=comparison.block_error_norms,
        exact_trajectory=comparison.exact_trajectory,
        archive_function_evaluations=comparison.closed_function_evaluations,
        success=comparison.success,
        message=comparison.message,
    )
