"""Solver-neutral contracts for reduced electron--phonon trajectories.

The contracts in this module carry data and declared conventions only.  They
do not import a solver, repair a trajectory, interpolate samples, or expose an
exact state vector.  Optional moments are capability-declared by their presence
in ``ReducedTrajectory.moments``; unavailable physics must never be represented
by undocumented zero arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, TypeAlias

import numpy as np
from numpy.typing import NDArray

JSONScalar: TypeAlias = str | int | float | bool | None
MethodFamily: TypeAlias = Literal[
    "exact_reference",
    "equal_time_open_dynamics",
    "selected_mode_unitary",
    "rt_tddft",
    "other",
]
ReferenceAccess: TypeAlias = Literal["none", "offline_only"]

COHERENT_PHONON_KEY = "phonon.coherent_amplitude"
NORMAL_PHONON_KEY = "phonon.normal_fluctuation"
ANOMALOUS_PHONON_KEY = "phonon.anomalous_fluctuation"
CONNECTED_EPH_KEY = "electron_phonon.connected"


@dataclass(frozen=True, slots=True)
class MethodIdentity:
    """Method identity and exact-reference data-flow declaration."""

    method_id: str
    family: MethodFamily
    approximation: str
    implementation: str
    reference_access: ReferenceAccess


@dataclass(frozen=True, slots=True)
class TrajectoryProvenance:
    """Separate source-stated facts from repository numerical choices."""

    citations: tuple[str, ...]
    paper_stated: Mapping[str, JSONScalar]
    repository_choices: Mapping[str, JSONScalar]
    code_revision: str | None = None


@dataclass(frozen=True, slots=True)
class MomentSeries:
    """One optional reduced-moment time series with explicit axes and units."""

    values: NDArray[np.generic]
    axes: tuple[str, ...]
    unit: str
    convention: str


@dataclass(frozen=True, slots=True)
class ReducedTrajectory:
    """Common reduced trajectory emitted by exact, classical, or quantum lanes."""

    time: NDArray[np.float64]
    time_unit: str
    electron_1rdm: NDArray[np.complex128]
    electronic_basis: tuple[str, ...]
    electronic_basis_convention: str
    electron_number: float | None
    method: MethodIdentity
    provenance: TrajectoryProvenance
    moments: Mapping[str, MomentSeries]


@dataclass(frozen=True, slots=True)
class TrajectoryDiagnostics:
    """Structural errors, physicality findings, and quantitative metrics."""

    structure_issues: tuple[str, ...]
    physicality_issues: tuple[str, ...]
    metrics: Mapping[str, float]

    @property
    def structurally_valid(self) -> bool:
        return not self.structure_issues


@dataclass(frozen=True, slots=True)
class SpectrumConvention:
    """Fully declared discrete Fourier-transform convention."""

    normalization: Literal["none", "forward", "ortho"] = "forward"
    detrend: Literal["none", "mean"] = "none"
    sided: Literal["one-sided", "two-sided"] = "one-sided"
    angular_frequency: bool = True
    window: Literal["hann_symmetric"] = "hann_symmetric"


@dataclass(frozen=True, slots=True)
class Spectrum:
    """Hann-apodized discrete spectrum and its declared transform."""

    frequency: NDArray[np.float64]
    transform: NDArray[np.complex128]
    magnitude: NDArray[np.float64]
    convention: SpectrumConvention


__all__ = [
    "ANOMALOUS_PHONON_KEY",
    "COHERENT_PHONON_KEY",
    "CONNECTED_EPH_KEY",
    "JSONScalar",
    "MethodFamily",
    "MethodIdentity",
    "MomentSeries",
    "NORMAL_PHONON_KEY",
    "ReducedTrajectory",
    "ReferenceAccess",
    "Spectrum",
    "SpectrumConvention",
    "TrajectoryDiagnostics",
    "TrajectoryProvenance",
]
