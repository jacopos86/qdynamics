"""Validation and comparison gates for reduced open-dynamics trajectories."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .contracts import (
    ANOMALOUS_PHONON_KEY,
    NORMAL_PHONON_KEY,
    ReducedTrajectory,
    TrajectoryDiagnostics,
)


class TrajectoryValidationError(ValueError):
    """Raised when a trajectory violates a structural data contract."""


class TrajectoryComparisonError(ValueError):
    """Raised when trajectories do not share a declared comparison frame."""


def _finite(array: np.ndarray) -> bool:
    values = np.asarray(array)
    return bool(
        np.all(np.isfinite(values.real))
        and np.all(np.isfinite(values.imag))
    )


def _max_abs(array: np.ndarray) -> float:
    values = np.asarray(array)
    return float(np.max(np.abs(values))) if values.size else 0.0


def _mapping_overlap(first: Mapping[str, object], second: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(sorted(set(first).intersection(second)))


def validate_trajectory(
    trajectory: ReducedTrajectory,
    *,
    atol: float = 1.0e-10,
    rtol: float = 1.0e-8,
) -> TrajectoryDiagnostics:
    """Return structural and physical diagnostics without mutating data.

    Structural findings prevent safe interpretation of array axes and therefore
    make the trajectory invalid.  Physicality findings remain diagnostics: the
    function never projects, deletes, clips, or otherwise repairs samples.
    """

    if atol < 0.0 or rtol < 0.0:
        raise ValueError("atol and rtol must be nonnegative")

    structure: list[str] = []
    physicality: list[str] = []
    metrics: dict[str, float] = {}

    time = np.asarray(trajectory.time)
    rho = np.asarray(trajectory.electron_1rdm)
    if time.ndim != 1 or time.size == 0:
        structure.append("time must be a nonempty one-dimensional array")
    elif not _finite(time):
        structure.append("time contains non-finite values")
    elif time.size > 1 and not np.all(np.diff(time) > 0.0):
        structure.append("time must be strictly increasing")

    expected_basis = len(trajectory.electronic_basis)
    if not trajectory.time_unit.strip():
        structure.append("time_unit must be nonempty")
    if not trajectory.electronic_basis_convention.strip():
        structure.append("electronic_basis_convention must be nonempty")
    if len(set(trajectory.electronic_basis)) != expected_basis:
        structure.append("electronic_basis labels must be unique")
    if rho.ndim != 3:
        structure.append("electron_1rdm must have axes (time,row,column)")
    elif rho.shape != (time.size, expected_basis, expected_basis):
        structure.append(
            "electron_1rdm shape must match time and electronic_basis"
        )
    elif not _finite(rho):
        structure.append("electron_1rdm contains non-finite values")

    if trajectory.electron_number is not None:
        number = float(trajectory.electron_number)
        if not np.isfinite(number) or number < 0.0:
            structure.append("electron_number must be finite and nonnegative")

    if trajectory.method.family == "exact_reference":
        if trajectory.method.reference_access != "offline_only":
            structure.append(
                "exact_reference methods must declare offline_only access"
            )
    overlap = _mapping_overlap(
        trajectory.provenance.paper_stated,
        trajectory.provenance.repository_choices,
    )
    if overlap:
        structure.append(
            "paper_stated and repository_choices reuse keys: "
            + ", ".join(overlap)
        )
    if not trajectory.provenance.citations:
        structure.append("provenance must include at least one citation")

    for key, series in trajectory.moments.items():
        values = np.asarray(series.values)
        if not key.strip():
            structure.append("moment keys must be nonempty")
        if values.ndim != len(series.axes):
            structure.append(f"moment {key!r} axis count does not match ndim")
            continue
        if not series.axes or series.axes[0] != "time":
            structure.append(f"moment {key!r} must declare time as its first axis")
        elif values.shape[0] != time.size:
            structure.append(f"moment {key!r} time axis does not match trajectory")
        if len(set(series.axes)) != len(series.axes):
            structure.append(f"moment {key!r} contains duplicate axis labels")
        if not series.unit.strip() or not series.convention.strip():
            structure.append(f"moment {key!r} must declare unit and convention")
        if not _finite(values):
            structure.append(f"moment {key!r} contains non-finite values")

    if not structure and rho.ndim == 3:
        hermitian_residual = _max_abs(
            rho - np.swapaxes(rho.conjugate(), -1, -2)
        )
        metrics["electron_hermiticity_max_abs"] = hermitian_residual
        scale = max(1.0, _max_abs(rho))
        if hermitian_residual > atol + rtol * scale:
            physicality.append("electron_1rdm is not Hermitian within tolerance")

        hermitian_rho = 0.5 * (
            rho + np.swapaxes(rho.conjugate(), -1, -2)
        )
        traces = np.trace(hermitian_rho, axis1=-2, axis2=-1).real
        metrics["electron_trace_min"] = float(np.min(traces))
        metrics["electron_trace_max"] = float(np.max(traces))
        if trajectory.electron_number is not None:
            target = float(trajectory.electron_number)
            trace_error = _max_abs(traces - target)
            metrics["electron_trace_max_abs_error"] = trace_error
            if trace_error > atol + rtol * max(1.0, abs(target)):
                physicality.append(
                    "electron_1rdm trace differs from declared electron_number"
                )

        eigenvalues = np.linalg.eigvalsh(hermitian_rho)
        minimum = float(np.min(eigenvalues))
        maximum = float(np.max(eigenvalues))
        metrics["electron_min_eigenvalue"] = minimum
        metrics["electron_max_eigenvalue"] = maximum
        if minimum < -(atol + rtol):
            physicality.append("electron_1rdm has a negative eigenvalue")
        if maximum > 1.0 + atol + rtol:
            physicality.append("electron_1rdm has an eigenvalue above one")

    normal = trajectory.moments.get(NORMAL_PHONON_KEY)
    if normal is not None:
        values = np.asarray(normal.values)
        if values.ndim >= 3 and values.shape[-1] == values.shape[-2]:
            residual = _max_abs(
                values - np.swapaxes(values.conjugate(), -1, -2)
            )
            metrics["normal_phonon_hermiticity_max_abs"] = residual
            if residual > atol + rtol * max(1.0, _max_abs(values)):
                physicality.append(
                    "normal phonon fluctuation is not Hermitian within tolerance"
                )
            hermitian = 0.5 * (
                values + np.swapaxes(values.conjugate(), -1, -2)
            )
            minimum = float(np.min(np.linalg.eigvalsh(hermitian)))
            metrics["normal_phonon_min_eigenvalue"] = minimum
            if minimum < -(atol + rtol):
                physicality.append(
                    "normal phonon fluctuation has a negative eigenvalue"
                )

    anomalous = trajectory.moments.get(ANOMALOUS_PHONON_KEY)
    if anomalous is not None:
        values = np.asarray(anomalous.values)
        if values.ndim >= 3 and values.shape[-1] == values.shape[-2]:
            residual = _max_abs(values - np.swapaxes(values, -1, -2))
            metrics["anomalous_phonon_symmetry_max_abs"] = residual
            if residual > atol + rtol * max(1.0, _max_abs(values)):
                physicality.append(
                    "anomalous phonon fluctuation is not transpose-symmetric"
                )

    return TrajectoryDiagnostics(
        structure_issues=tuple(structure),
        physicality_issues=tuple(physicality),
        metrics=metrics,
    )


def require_structurally_valid(
    trajectory: ReducedTrajectory,
    *,
    atol: float = 1.0e-10,
    rtol: float = 1.0e-8,
) -> None:
    diagnostics = validate_trajectory(trajectory, atol=atol, rtol=rtol)
    if diagnostics.structure_issues:
        raise TrajectoryValidationError("; ".join(diagnostics.structure_issues))


def require_comparable(
    *trajectories: ReducedTrajectory,
    time_atol: float = 1.0e-12,
    time_rtol: float = 1.0e-10,
    initial_atol: float = 1.0e-10,
) -> None:
    """Require a common sampled frame and common available initial moments."""

    if len(trajectories) < 2:
        raise ValueError("at least two trajectories are required")
    for trajectory in trajectories:
        require_structurally_valid(trajectory)

    anchor = trajectories[0]
    for candidate in trajectories[1:]:
        if candidate.time_unit != anchor.time_unit:
            raise TrajectoryComparisonError("time units differ")
        if candidate.electronic_basis != anchor.electronic_basis:
            raise TrajectoryComparisonError("electronic basis labels differ")
        if (
            candidate.electronic_basis_convention
            != anchor.electronic_basis_convention
        ):
            raise TrajectoryComparisonError("electronic basis conventions differ")
        if candidate.time.shape != anchor.time.shape or not np.allclose(
            candidate.time,
            anchor.time,
            atol=time_atol,
            rtol=time_rtol,
        ):
            raise TrajectoryComparisonError("sampled time grids differ")
        if candidate.electron_number != anchor.electron_number:
            raise TrajectoryComparisonError("declared electron numbers differ")
        if not np.allclose(
            candidate.electron_1rdm[0],
            anchor.electron_1rdm[0],
            atol=initial_atol,
            rtol=0.0,
        ):
            raise TrajectoryComparisonError("initial electronic 1-RDMs differ")

        shared = set(anchor.moments).intersection(candidate.moments)
        for key in sorted(shared):
            first = anchor.moments[key]
            second = candidate.moments[key]
            if first.axes != second.axes or first.unit != second.unit:
                raise TrajectoryComparisonError(
                    f"moment {key!r} axes or units differ"
                )
            if first.values.shape[1:] != second.values.shape[1:]:
                raise TrajectoryComparisonError(
                    f"moment {key!r} non-time shapes differ"
                )
            if not np.allclose(
                first.values[0],
                second.values[0],
                atol=initial_atol,
                rtol=0.0,
            ):
                raise TrajectoryComparisonError(
                    f"moment {key!r} initial values differ"
                )


__all__ = [
    "TrajectoryComparisonError",
    "TrajectoryValidationError",
    "require_comparable",
    "require_structurally_valid",
    "validate_trajectory",
]
