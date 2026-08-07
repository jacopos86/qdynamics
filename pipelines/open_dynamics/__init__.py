"""Common reduced trajectories for the repository's open-dynamics program."""

from .contracts import (
    ANOMALOUS_PHONON_KEY,
    COHERENT_PHONON_KEY,
    CONNECTED_EPH_KEY,
    MethodIdentity,
    MomentSeries,
    NORMAL_PHONON_KEY,
    ReducedTrajectory,
    Spectrum,
    SpectrumConvention,
    TrajectoryDiagnostics,
    TrajectoryProvenance,
)
from .observables import dimer_polarization, hann_spectrum
from .paper5_dimer import (
    DimerProtocol,
    Paper5DimerArrays,
    Paper5UnavailableError,
    Paper5VerticalSlice,
    arrays_from_matrix_states,
    from_paper5_dimer,
    load_dimer_protocol,
    run_paper5_vertical_slice,
)
from .validation import (
    TrajectoryComparisonError,
    TrajectoryValidationError,
    require_comparable,
    require_structurally_valid,
    validate_trajectory,
)

__all__ = [
    "ANOMALOUS_PHONON_KEY",
    "COHERENT_PHONON_KEY",
    "CONNECTED_EPH_KEY",
    "DimerProtocol",
    "MethodIdentity",
    "MomentSeries",
    "NORMAL_PHONON_KEY",
    "Paper5DimerArrays",
    "Paper5UnavailableError",
    "Paper5VerticalSlice",
    "ReducedTrajectory",
    "Spectrum",
    "SpectrumConvention",
    "TrajectoryComparisonError",
    "TrajectoryDiagnostics",
    "TrajectoryProvenance",
    "TrajectoryValidationError",
    "arrays_from_matrix_states",
    "dimer_polarization",
    "from_paper5_dimer",
    "hann_spectrum",
    "load_dimer_protocol",
    "require_comparable",
    "require_structurally_valid",
    "run_paper5_vertical_slice",
    "validate_trajectory",
]
