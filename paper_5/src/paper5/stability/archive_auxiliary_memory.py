"""Reciprocal operator frame and archive-section certificate.

This module implements the first algebraic gate of the state-adapted
auxiliary-memory construction.  It keeps the verified 29-coordinate raw
moment chart, projects one fixed Hilbert--Schmidt operator frame with the
static and drive Liouvillians, and asks whether the Pauli-repaired archive
velocity can be represented as an instantaneous section of that reciprocal
frame.

No trajectory data or exact state is used by :meth:`ArchiveAuxiliaryFrame.section`.
Exact or variational states enter only through the optional one-time hidden
coordinate contraction used to initialize a later autonomous model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .krylov_memory_closure import (
    KrylovClosureConstruction,
    OperatorBlock,
    RawMomentBasis,
    _apply_liouvillian,
    _real_block_gram,
    centered_jacobian_from_orthonormal,
    closed_coordinates_to_orthonormal,
    orthonormal_to_closed_coordinates,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
ArchiveField = Callable[[FloatArray, float], FloatArray]
ScalarTimeFunction = Callable[[float], float]


def _supported_pseudoinverse(
    matrix: FloatArray,
    *,
    relative_tolerance: float,
) -> tuple[FloatArray, FloatArray, int]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must lie between zero and one")
    left, singular_values, right_adjoint = np.linalg.svd(
        values,
        full_matrices=False,
    )
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return (
            np.zeros((values.shape[1], values.shape[0]), dtype=float),
            singular_values,
            0,
        )
    keep = singular_values >= relative_tolerance * singular_values[0]
    rank = int(np.count_nonzero(keep))
    inverse = np.zeros_like(singular_values)
    inverse[keep] = 1.0 / singular_values[keep]
    pseudoinverse = (right_adjoint.T * inverse[None, :]) @ left.T
    return pseudoinverse, singular_values, rank


def _relative_norm(residual: FloatArray, reference: FloatArray) -> float:
    denominator = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    return float(np.linalg.norm(residual) / denominator)


@dataclass(frozen=True)
class ReciprocalBlocks:
    """One assembled fixed-frame Liouvillian block matrix."""

    resolved_resolved: FloatArray
    resolved_hidden: FloatArray
    hidden_resolved: FloatArray
    hidden_hidden: FloatArray

    @property
    def resolved_skew_residual(self) -> float:
        matrix = self.resolved_resolved
        scale = max(float(np.linalg.norm(matrix)), 1.0)
        return float(np.linalg.norm(matrix + matrix.T) / scale)

    @property
    def hidden_skew_residual(self) -> float:
        matrix = self.hidden_hidden
        scale = max(float(np.linalg.norm(matrix)), 1.0)
        return float(np.linalg.norm(matrix + matrix.T) / scale)

    @property
    def reciprocity_residual(self) -> float:
        residual = self.resolved_hidden + self.hidden_resolved.T
        scale = max(
            float(np.linalg.norm(self.resolved_hidden)),
            float(np.linalg.norm(self.hidden_resolved)),
            1.0,
        )
        return float(np.linalg.norm(residual) / scale)


@dataclass(frozen=True)
class ArchiveSectionCertificate:
    """Minimum-norm archive section and its two compatibility residuals."""

    retained_coordinates: FloatArray
    centering_jacobian: FloatArray
    archive_velocity: FloatArray
    raw_archive_velocity: FloatArray
    raw_lift_residual: FloatArray
    raw_lift_singular_values: FloatArray
    raw_lift_rank: int
    in_span_velocity: FloatArray
    section_source: FloatArray
    hidden_section: FloatArray
    reconstructed_source: FloatArray
    incompatibility: FloatArray
    centered_incompatibility: FloatArray
    coupling_singular_values: FloatArray
    coupling_rank: int

    @property
    def raw_lift_relative_residual(self) -> float:
        return _relative_norm(self.raw_lift_residual, self.archive_velocity)

    @property
    def section_relative_residual(self) -> float:
        return _relative_norm(self.incompatibility, self.section_source)

    @property
    def centered_section_relative_residual(self) -> float:
        centered_source = self.centering_jacobian @ self.section_source
        return _relative_norm(self.centered_incompatibility, centered_source)


@dataclass(frozen=True)
class ArchiveMemoryVelocity:
    """One fixed-frame autonomous auxiliary-memory evaluation."""

    closed_coordinates: FloatArray
    centered_velocity: FloatArray
    retained_velocity: FloatArray
    memory_velocity: FloatArray
    physical_hidden_coordinates: FloatArray
    physical_hidden_velocity: FloatArray
    archive_section_derivative: FloatArray
    section: ArchiveSectionCertificate
    projected_norm_rate: float
    projected_norm_rate_from_incompatibility: float


@dataclass(frozen=True)
class ArchiveMemoryInitialCondition:
    """Preparation-initialized retained and archive-relative coordinates."""

    retained_coordinates: FloatArray
    memory_coordinates: FloatArray
    physical_hidden_coordinates: FloatArray
    section: ArchiveSectionCertificate


@dataclass(frozen=True)
class ArchiveMemoryTrajectory:
    """Stored fixed-step trajectory of the autonomous reciprocal model."""

    times: FloatArray
    retained_coordinates: FloatArray
    memory_coordinates: FloatArray
    closed_coordinates: FloatArray
    physical_hidden_norms: FloatArray
    centered_section_relative_residuals: FloatArray
    projected_norm_identity_residuals: FloatArray


@dataclass(frozen=True)
class ArchiveAuxiliaryFrame:
    """Fixed reciprocal frame over the verified 29-coordinate raw chart.

    The interface deliberately exposes only assembled blocks, the archive
    section certificate, and one-time hidden contractions.  Operator packing,
    component projection, and supported minimum-norm solves remain internal.
    """

    raw_basis: RawMomentBasis
    hidden_observables: OperatorBlock
    static_blocks: ReciprocalBlocks
    drive_blocks: ReciprocalBlocks

    def __post_init__(self) -> None:
        hidden_dimension = self.static_blocks.resolved_hidden.shape[1]
        expected_shapes = {
            "resolved_resolved": (29, 29),
            "resolved_hidden": (29, hidden_dimension),
            "hidden_resolved": (hidden_dimension, 29),
            "hidden_hidden": (hidden_dimension, hidden_dimension),
        }
        for label, blocks in (
            ("static", self.static_blocks),
            ("drive", self.drive_blocks),
        ):
            for name, expected in expected_shapes.items():
                actual = np.asarray(getattr(blocks, name)).shape
                if actual != expected:
                    raise ValueError(
                        f"{label} {name} must have shape {expected}, got {actual}"
                    )
        if len(self.hidden_observables) not in (0, hidden_dimension):
            raise ValueError(
                "hidden_observables must either contain the complete hidden "
                "operator frame or be empty for a numerical projection"
            )

    @property
    def hidden_dimension(self) -> int:
        return int(self.static_blocks.resolved_hidden.shape[1])

    @property
    def has_operator_frame(self) -> bool:
        """Whether hidden coordinates can be contracted from a state vector."""

        return len(self.hidden_observables) == self.hidden_dimension

    def blocks(self, *, drive_value: float) -> ReciprocalBlocks:
        """Assemble static plus instantaneous-drive component blocks."""

        if not np.isfinite(drive_value):
            raise ValueError("drive_value must be finite")
        return ReciprocalBlocks(
            resolved_resolved=(
                self.static_blocks.resolved_resolved
                + drive_value * self.drive_blocks.resolved_resolved
            ),
            resolved_hidden=(
                self.static_blocks.resolved_hidden
                + drive_value * self.drive_blocks.resolved_hidden
            ),
            hidden_resolved=(
                self.static_blocks.hidden_resolved
                + drive_value * self.drive_blocks.hidden_resolved
            ),
            hidden_hidden=(
                self.static_blocks.hidden_hidden
                + drive_value * self.drive_blocks.hidden_hidden
            ),
        )

    def prefix(self, hidden_dimension: int) -> ArchiveAuxiliaryFrame:
        """Return a nested prefix without recomputing component projections."""

        if not 0 < hidden_dimension <= self.hidden_dimension:
            raise ValueError(
                "hidden_dimension must be between one and "
                f"{self.hidden_dimension}"
            )

        def restrict(blocks: ReciprocalBlocks) -> ReciprocalBlocks:
            return ReciprocalBlocks(
                resolved_resolved=blocks.resolved_resolved,
                resolved_hidden=blocks.resolved_hidden[:, :hidden_dimension],
                hidden_resolved=blocks.hidden_resolved[:hidden_dimension, :],
                hidden_hidden=blocks.hidden_hidden[
                    :hidden_dimension,
                    :hidden_dimension,
                ],
            )

        hidden_observables = (
            self.hidden_observables[:hidden_dimension]
            if self.has_operator_frame
            else ()
        )
        return ArchiveAuxiliaryFrame(
            raw_basis=self.raw_basis,
            hidden_observables=hidden_observables,
            static_blocks=restrict(self.static_blocks),
            drive_blocks=restrict(self.drive_blocks),
        )

    def orthogonal_projection(
        self,
        hidden_basis: FloatArray,
        *,
        orthogonality_tolerance: float = 1e-10,
    ) -> ArchiveAuxiliaryFrame:
        """Project the hidden realization onto one orthonormal fixed frame.

        ``hidden_basis`` maps reduced hidden coordinates into this frame's
        hidden coordinates.  The same basis is applied on the trial and test
        sides, so skew-adjointness and reciprocal resolved--hidden coupling
        are preserved.  The returned object intentionally stores only the
        reduced numerical blocks; preparation coordinates must be projected
        from a contraction in the parent operator frame.
        """

        basis = np.asarray(hidden_basis, dtype=float)
        if basis.ndim != 2 or basis.shape[0] != self.hidden_dimension:
            raise ValueError(
                "hidden_basis must have shape "
                f"({self.hidden_dimension}, reduced_dimension)"
            )
        if basis.shape[1] == 0 or basis.shape[1] > self.hidden_dimension:
            raise ValueError("reduced hidden dimension is outside the frame")
        if not np.all(np.isfinite(basis)):
            raise ValueError("hidden_basis must be finite")
        orthogonality_error = float(
            np.linalg.norm(basis.T @ basis - np.eye(basis.shape[1]))
        )
        if orthogonality_error > orthogonality_tolerance:
            raise ValueError(
                "hidden_basis is not orthonormal: "
                f"residual {orthogonality_error:.3e}"
            )

        def project(blocks: ReciprocalBlocks) -> ReciprocalBlocks:
            return ReciprocalBlocks(
                resolved_resolved=blocks.resolved_resolved,
                resolved_hidden=blocks.resolved_hidden @ basis,
                hidden_resolved=basis.T @ blocks.hidden_resolved,
                hidden_hidden=basis.T @ blocks.hidden_hidden @ basis,
            )

        return ArchiveAuxiliaryFrame(
            raw_basis=self.raw_basis,
            hidden_observables=(),
            static_blocks=project(self.static_blocks),
            drive_blocks=project(self.drive_blocks),
        )

    def section(
        self,
        closed_coordinates: FloatArray,
        archive_velocity: FloatArray,
        *,
        drive_value: float,
        relative_tolerance: float = 1e-11,
    ) -> ArchiveSectionCertificate:
        """Lift and embed one archive velocity in the reciprocal frame.

        ``archive_velocity`` is the 31-coordinate Pauli-repaired archive EOM
        evaluated at ``closed_coordinates``.  The returned ``hidden_section``
        is the minimum-Euclidean-norm fixed-frame coordinate whose reciprocal
        feedback reconstructs the part of the raw archive field outside the
        in-span Liouvillian drift.
        """

        closed = np.asarray(closed_coordinates, dtype=float)
        velocity = np.asarray(archive_velocity, dtype=float)
        if closed.shape != (31,) or velocity.shape != (31,):
            raise ValueError("closed coordinates and velocity must have shape (31,)")

        retained = closed_coordinates_to_orthonormal(self.raw_basis, closed)
        jacobian = centered_jacobian_from_orthonormal(self.raw_basis, retained)
        jacobian_inverse, lift_values, lift_rank = _supported_pseudoinverse(
            jacobian,
            relative_tolerance=relative_tolerance,
        )
        if lift_rank != 29:
            raise RuntimeError(
                "raw-to-centered Jacobian lost full rank: "
                f"expected 29, got {lift_rank}"
            )
        raw_velocity = jacobian_inverse @ velocity
        raw_lift_residual = velocity - jacobian @ raw_velocity

        blocks = self.blocks(drive_value=drive_value)
        in_span = blocks.resolved_resolved @ retained
        source = raw_velocity - in_span
        coupling_inverse, coupling_values, coupling_rank = (
            _supported_pseudoinverse(
                blocks.resolved_hidden,
                relative_tolerance=relative_tolerance,
            )
        )
        hidden_section = coupling_inverse @ source
        reconstructed = blocks.resolved_hidden @ hidden_section
        incompatibility = source - reconstructed
        centered_incompatibility = jacobian @ incompatibility
        return ArchiveSectionCertificate(
            retained_coordinates=retained,
            centering_jacobian=jacobian,
            archive_velocity=velocity,
            raw_archive_velocity=raw_velocity,
            raw_lift_residual=raw_lift_residual,
            raw_lift_singular_values=lift_values,
            raw_lift_rank=lift_rank,
            in_span_velocity=in_span,
            section_source=source,
            hidden_section=hidden_section,
            reconstructed_source=reconstructed,
            incompatibility=incompatibility,
            centered_incompatibility=centered_incompatibility,
            coupling_singular_values=coupling_values,
            coupling_rank=coupling_rank,
        )

    def contract_hidden_state(self, state_vector: ComplexArray) -> FloatArray:
        """Contract one declared preparation into the fixed hidden frame."""

        if not self.has_operator_frame:
            raise RuntimeError(
                "this numerical projection has no materialized hidden "
                "operators; contract in its parent frame and project those "
                "coordinates"
            )
        state = np.asarray(state_vector, dtype=complex)
        if state.shape != (self.raw_basis.hilbert_dimension,):
            raise ValueError("state vector has incompatible Hilbert dimension")
        return np.asarray(
            [
                np.vdot(state, operator @ state).real
                for operator in self.hidden_observables
            ],
            dtype=float,
        )

    def initialize_memory(
        self,
        closed_coordinates: FloatArray,
        state_vector: ComplexArray,
        archive_field: ArchiveField,
        *,
        drive_value: float,
        relative_tolerance: float = 1e-11,
    ) -> ArchiveMemoryInitialCondition:
        """Initialize the physical hidden contraction and relative memory."""

        physical_hidden = self.contract_hidden_state(state_vector)
        return self.initialize_memory_from_hidden(
            closed_coordinates,
            physical_hidden,
            archive_field,
            drive_value=drive_value,
            relative_tolerance=relative_tolerance,
        )

    def initialize_memory_from_hidden(
        self,
        closed_coordinates: FloatArray,
        physical_hidden_coordinates: FloatArray,
        archive_field: ArchiveField,
        *,
        drive_value: float,
        relative_tolerance: float = 1e-11,
    ) -> ArchiveMemoryInitialCondition:
        """Initialize memory from already-contracted hidden coordinates."""

        closed = np.asarray(closed_coordinates, dtype=float)
        physical_hidden = np.asarray(physical_hidden_coordinates, dtype=float)
        if physical_hidden.shape != (self.hidden_dimension,):
            raise ValueError(
                "physical_hidden_coordinates must have shape "
                f"{(self.hidden_dimension,)}"
            )
        if not np.all(np.isfinite(physical_hidden)):
            raise ValueError("physical_hidden_coordinates must be finite")
        section = self.section(
            closed,
            archive_field(closed, drive_value),
            drive_value=drive_value,
            relative_tolerance=relative_tolerance,
        )
        return ArchiveMemoryInitialCondition(
            retained_coordinates=section.retained_coordinates,
            memory_coordinates=physical_hidden - section.hidden_section,
            physical_hidden_coordinates=physical_hidden,
            section=section,
        )

    def autonomous_velocity(
        self,
        retained_coordinates: FloatArray,
        memory_coordinates: FloatArray,
        archive_field: ArchiveField,
        *,
        drive_value: float,
        drive_rate: float,
        relative_tolerance: float = 1e-11,
        directional_step: float = 1e-6,
    ) -> ArchiveMemoryVelocity:
        """Evaluate the fixed-frame archive-relative auxiliary equations.

        The section derivative is taken along the just-computed retained
        velocity and the supplied drive rate.  This is the fixed-frame form of
        the projection-transport term; a future moving frame must add its
        separate connection and normal-transport contributions.
        """

        retained = np.asarray(retained_coordinates, dtype=float)
        memory = np.asarray(memory_coordinates, dtype=float)
        if retained.shape != (29,):
            raise ValueError("retained_coordinates must have shape (29,)")
        if memory.shape != (self.hidden_dimension,):
            raise ValueError(
                f"memory_coordinates must have shape {(self.hidden_dimension,)}"
            )
        if not np.isfinite(drive_value) or not np.isfinite(drive_rate):
            raise ValueError("drive_value and drive_rate must be finite")
        if directional_step <= 0.0:
            raise ValueError("directional_step must be positive")

        closed = orthonormal_to_closed_coordinates(self.raw_basis, retained)
        archive_velocity = np.asarray(
            archive_field(closed, drive_value),
            dtype=float,
        )
        section = self.section(
            closed,
            archive_velocity,
            drive_value=drive_value,
            relative_tolerance=relative_tolerance,
        )
        blocks = self.blocks(drive_value=drive_value)
        retained_velocity = (
            section.raw_archive_velocity
            + blocks.resolved_hidden @ memory
        )

        direction_scale = max(
            1.0,
            float(np.linalg.norm(retained_velocity)),
            abs(float(drive_rate)),
        )
        step = directional_step / direction_scale
        section_offsets: list[ArchiveSectionCertificate] = []
        for sign in (-1.0, 1.0):
            offset_retained = retained + sign * step * retained_velocity
            offset_drive = drive_value + sign * step * drive_rate
            offset_closed = orthonormal_to_closed_coordinates(
                self.raw_basis,
                offset_retained,
            )
            section_offsets.append(
                self.section(
                    offset_closed,
                    archive_field(offset_closed, offset_drive),
                    drive_value=offset_drive,
                    relative_tolerance=relative_tolerance,
                )
            )
        section_derivative = (
            section_offsets[1].hidden_section
            - section_offsets[0].hidden_section
        ) / (2.0 * step)

        physical_hidden = section.hidden_section + memory
        physical_hidden_velocity = (
            blocks.hidden_resolved @ retained
            + blocks.hidden_hidden @ physical_hidden
        )
        memory_velocity = physical_hidden_velocity - section_derivative
        centered_velocity = section.centering_jacobian @ retained_velocity
        projected_norm_rate = float(
            retained @ retained_velocity
            + physical_hidden @ physical_hidden_velocity
        )
        incompatibility_rate = float(retained @ section.incompatibility)
        return ArchiveMemoryVelocity(
            closed_coordinates=closed,
            centered_velocity=centered_velocity,
            retained_velocity=retained_velocity,
            memory_velocity=memory_velocity,
            physical_hidden_coordinates=physical_hidden,
            physical_hidden_velocity=physical_hidden_velocity,
            archive_section_derivative=section_derivative,
            section=section,
            projected_norm_rate=projected_norm_rate,
            projected_norm_rate_from_incompatibility=incompatibility_rate,
        )

    def lossless_exchange_rate(
        self,
        retained_coordinates: FloatArray,
        hidden_coordinates: FloatArray,
        *,
        drive_value: float,
    ) -> float:
        """Return the projected Hilbert--Schmidt norm derivative."""

        retained = np.asarray(retained_coordinates, dtype=float)
        hidden = np.asarray(hidden_coordinates, dtype=float)
        if retained.shape != (29,):
            raise ValueError("retained_coordinates must have shape (29,)")
        if hidden.shape != (self.hidden_dimension,):
            raise ValueError(
                "hidden_coordinates must have shape "
                f"{(self.hidden_dimension,)}"
            )
        blocks = self.blocks(drive_value=drive_value)
        retained_velocity = (
            blocks.resolved_resolved @ retained
            + blocks.resolved_hidden @ hidden
        )
        hidden_velocity = (
            blocks.hidden_resolved @ retained
            + blocks.hidden_hidden @ hidden
        )
        return float(
            retained @ retained_velocity + hidden @ hidden_velocity
        )


def _component_blocks(
    construction: KrylovClosureConstruction,
    hidden_observables: OperatorBlock,
    *,
    drive_component: bool,
) -> ReciprocalBlocks:
    retained = construction.raw_basis.orthonormal_observables
    hamiltonian = (
        construction.drive_hamiltonian
        if drive_component
        else construction.static_hamiltonian
    )
    retained_action = _apply_liouvillian(hamiltonian, retained)
    dimension = construction.hilbert_dimension
    resolved_hidden = np.empty(
        (len(retained), len(hidden_observables)),
        dtype=float,
    )
    hidden_hidden = np.empty(
        (len(hidden_observables), len(hidden_observables)),
        dtype=float,
    )
    for column, hidden_operator in enumerate(hidden_observables):
        # Stream one Liouvillian image at a time.  Retaining the complete
        # hidden-action block duplicates every dense preparation descendant
        # and can consume several gigabytes at cutoff 16.
        hidden_action = _apply_liouvillian(
            hamiltonian,
            (hidden_operator,),
        )
        resolved_hidden[:, column] = _real_block_gram(
            retained,
            hidden_action,
            dimension=dimension,
        )[:, 0]
        hidden_hidden[:, column] = _real_block_gram(
            hidden_observables,
            hidden_action,
            dimension=dimension,
        )[:, 0]
    return ReciprocalBlocks(
        resolved_resolved=_real_block_gram(
            retained,
            retained_action,
            dimension=dimension,
        ),
        resolved_hidden=resolved_hidden,
        hidden_resolved=-resolved_hidden.T,
        hidden_hidden=hidden_hidden,
    )


def build_archive_auxiliary_frame(
    construction: KrylovClosureConstruction,
    *,
    order: int,
) -> ArchiveAuxiliaryFrame:
    """Build a fixed-union Route-4 frame from existing operator shells.

    This is the first construction candidate, not an accepted closure.  Its
    section residual determines whether the shell union can contain the
    archive field reciprocally before state adaptation or autonomous rollout
    is attempted.
    """

    coefficients = construction.coefficients(order)
    return build_archive_auxiliary_frame_from_observables(
        construction,
        coefficients.auxiliary_observables,
    )


def build_archive_auxiliary_frame_from_observables(
    construction: KrylovClosureConstruction,
    hidden_observables: OperatorBlock,
    *,
    orthogonality_tolerance: float = 1e-10,
) -> ArchiveAuxiliaryFrame:
    """Build a reciprocal frame from a declared orthonormal hidden union."""

    hidden = tuple(hidden_observables)
    if not hidden:
        raise ValueError("hidden_observables must not be empty")
    dimension = construction.hilbert_dimension
    hidden_gram = _real_block_gram(hidden, hidden, dimension=dimension)
    hidden_error = float(np.linalg.norm(hidden_gram - np.eye(len(hidden))))
    cross = _real_block_gram(
        construction.raw_basis.orthonormal_observables,
        hidden,
        dimension=dimension,
    )
    cross_error = float(np.linalg.norm(cross))
    if hidden_error > orthogonality_tolerance or cross_error > orthogonality_tolerance:
        raise ValueError(
            "hidden frame is not Hilbert--Schmidt orthonormal to the raw frame: "
            f"hidden error {hidden_error:.3e}, cross error {cross_error:.3e}"
        )
    return ArchiveAuxiliaryFrame(
        raw_basis=construction.raw_basis,
        hidden_observables=hidden,
        static_blocks=_component_blocks(
            construction,
            hidden,
            drive_component=False,
        ),
        drive_blocks=_component_blocks(
            construction,
            hidden,
            drive_component=True,
        ),
    )


def propagate_archive_auxiliary_rk4(
    frame: ArchiveAuxiliaryFrame,
    initial_condition: ArchiveMemoryInitialCondition,
    archive_field: ArchiveField,
    drive_value: ScalarTimeFunction,
    drive_rate: ScalarTimeFunction,
    *,
    final_time: float,
    time_step: float,
    sample_step: float,
    relative_tolerance: float = 1e-11,
    directional_step: float = 1e-6,
) -> ArchiveMemoryTrajectory:
    """Propagate the fixed-frame autonomous model with stage-consistent RK4."""

    if final_time <= 0.0 or time_step <= 0.0 or sample_step <= 0.0:
        raise ValueError("final_time, time_step, and sample_step must be positive")
    step_count = int(round(final_time / time_step))
    sample_stride = int(round(sample_step / time_step))
    if abs(step_count * time_step - final_time) > 1e-12:
        raise ValueError("final_time must be an integer multiple of time_step")
    if sample_stride <= 0 or abs(sample_stride * time_step - sample_step) > 1e-12:
        raise ValueError("sample_step must be an integer multiple of time_step")
    if step_count % sample_stride:
        raise ValueError("final_time must be an integer multiple of sample_step")

    state = np.concatenate(
        (
            np.asarray(initial_condition.retained_coordinates, dtype=float),
            np.asarray(initial_condition.memory_coordinates, dtype=float),
        )
    )
    expected_size = 29 + frame.hidden_dimension
    if state.shape != (expected_size,):
        raise ValueError("initial condition is incompatible with the frame")

    def evaluate(time_value: float, augmented: FloatArray) -> ArchiveMemoryVelocity:
        return frame.autonomous_velocity(
            augmented[:29],
            augmented[29:],
            archive_field,
            drive_value=float(drive_value(time_value)),
            drive_rate=float(drive_rate(time_value)),
            relative_tolerance=relative_tolerance,
            directional_step=directional_step,
        )

    sample_count = step_count // sample_stride + 1
    times = np.linspace(0.0, final_time, sample_count)
    retained_history = np.empty((sample_count, 29), dtype=float)
    memory_history = np.empty(
        (sample_count, frame.hidden_dimension),
        dtype=float,
    )
    closed_history = np.empty((sample_count, 31), dtype=float)
    hidden_norms = np.empty(sample_count, dtype=float)
    section_residuals = np.empty(sample_count, dtype=float)
    identity_residuals = np.empty(sample_count, dtype=float)

    def record(index: int, time_value: float) -> None:
        evaluation = evaluate(time_value, state)
        retained_history[index] = state[:29]
        memory_history[index] = state[29:]
        closed_history[index] = evaluation.closed_coordinates
        hidden_norms[index] = float(
            np.linalg.norm(evaluation.physical_hidden_coordinates)
        )
        section_residuals[index] = (
            evaluation.section.centered_section_relative_residual
        )
        identity_residuals[index] = abs(
            evaluation.projected_norm_rate
            - evaluation.projected_norm_rate_from_incompatibility
        )

    record(0, 0.0)
    sample_index = 1
    for step_index in range(step_count):
        time_value = step_index * time_step

        first = evaluate(time_value, state)
        k1 = np.concatenate((first.retained_velocity, first.memory_velocity))
        second = evaluate(
            time_value + 0.5 * time_step,
            state + 0.5 * time_step * k1,
        )
        k2 = np.concatenate((second.retained_velocity, second.memory_velocity))
        third = evaluate(
            time_value + 0.5 * time_step,
            state + 0.5 * time_step * k2,
        )
        k3 = np.concatenate((third.retained_velocity, third.memory_velocity))
        fourth = evaluate(
            time_value + time_step,
            state + time_step * k3,
        )
        k4 = np.concatenate((fourth.retained_velocity, fourth.memory_velocity))
        state = state + (time_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(state)):
            raise FloatingPointError(
                f"auxiliary trajectory became nonfinite at t={time_value + time_step}"
            )
        if (step_index + 1) % sample_stride == 0:
            record(sample_index, time_value + time_step)
            sample_index += 1

    return ArchiveMemoryTrajectory(
        times=times,
        retained_coordinates=retained_history,
        memory_coordinates=memory_history,
        closed_coordinates=closed_history,
        physical_hidden_norms=hidden_norms,
        centered_section_relative_residuals=section_residuals,
        projected_norm_identity_residuals=identity_residuals,
    )


__all__ = [
    "ArchiveAuxiliaryFrame",
    "ArchiveField",
    "ArchiveMemoryInitialCondition",
    "ArchiveMemoryTrajectory",
    "ArchiveMemoryVelocity",
    "ArchiveSectionCertificate",
    "ReciprocalBlocks",
    "build_archive_auxiliary_frame",
    "build_archive_auxiliary_frame_from_observables",
    "propagate_archive_auxiliary_rk4",
]
