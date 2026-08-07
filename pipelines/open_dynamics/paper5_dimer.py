"""Adapter from the existing Paper V dimer solvers to common trajectories.

This module does not rederive or edit the dimer equations.  It imports the
existing tested Paper V implementation lazily, converts its public matrix-state
objects field by field, and exposes no exact wavefunction or Hamiltonian.

The public Riva--Simoni--Ping v1 archive omits the cited spin-reduced
supplement and numerical data.  Consequently this adapter supports a
source-anchored independent benchmark, not a reproduction of published curves.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, Mapping, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .contracts import (
    ANOMALOUS_PHONON_KEY,
    COHERENT_PHONON_KEY,
    CONNECTED_EPH_KEY,
    JSONScalar,
    MethodIdentity,
    MomentSeries,
    NORMAL_PHONON_KEY,
    ReducedTrajectory,
    TrajectoryProvenance,
)
from .validation import require_comparable, require_structurally_valid

Paper5Level: TypeAlias = Literal["exact", "coherent_only", "equal_time_full"]
PROTOCOL_SCHEMA_VERSION = "open_dynamics.riva_ping_dimer_protocol.v1"
_BASIS = ("site_1_spin_up", "site_2_spin_up")
_BASIS_CONVENTION = (
    "two-site spin-up one-particle density matrix; the spin-symmetric "
    "spin-down channel is not duplicated"
)


class Paper5UnavailableError(ImportError):
    """Raised when the separate ``paper5`` src-layout package is unavailable."""


@dataclass(frozen=True, slots=True)
class DimerProtocol:
    """Source statements and explicitly separate repository choices."""

    schema_version: str
    benchmark_id: str
    citations: tuple[str, ...]
    paper_stated: Mapping[str, JSONScalar]
    repository_choices: Mapping[str, JSONScalar]
    claim_boundary: str


@dataclass(frozen=True, slots=True)
class Paper5DimerArrays:
    """Field-wise Paper V dimer arrays before method identity is attached."""

    time: NDArray[np.float64]
    time_unit: str
    electron_1rdm: NDArray[np.complex128]
    electronic_basis: tuple[str, ...]
    electronic_basis_convention: str
    electron_number: float | None
    moments: Mapping[str, MomentSeries]


@dataclass(frozen=True, slots=True)
class Paper5VerticalSlice:
    """Common-frame exact, coherent, and equal-time benchmark trajectories."""

    protocol: DimerProtocol
    exact: ReducedTrajectory
    coherent_only: ReducedTrajectory
    equal_time_full: ReducedTrajectory
    full_coordinate_error: NDArray[np.float64]
    full_block_names: tuple[str, ...]
    full_block_error_norms: NDArray[np.float64]

    @property
    def trajectories(self) -> tuple[ReducedTrajectory, ...]:
        return (self.exact, self.coherent_only, self.equal_time_full)


def _is_json_scalar(value: object) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _validated_scalar_mapping(
    value: object,
    *,
    label: str,
) -> dict[str, JSONScalar]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a nonempty JSON object")
    result: dict[str, JSONScalar] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{label} keys must be nonempty strings")
        if not _is_json_scalar(item):
            raise ValueError(f"{label}[{key!r}] must be a JSON scalar")
        result[key] = item  # type: ignore[assignment]
    return result


def load_dimer_protocol(path: str | Path) -> DimerProtocol:
    """Load the checked protocol manifest without generating trajectory data."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("protocol manifest must contain a JSON object")
    expected = {
        "schema_version",
        "benchmark_id",
        "citations",
        "paper_stated",
        "repository_choices",
        "claim_boundary",
    }
    unknown = set(payload).difference(expected)
    missing = expected.difference(payload)
    if unknown or missing:
        raise ValueError(
            f"protocol keys mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    if payload["schema_version"] != PROTOCOL_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported protocol schema {payload['schema_version']!r}"
        )
    citations = payload["citations"]
    if (
        not isinstance(citations, list)
        or not citations
        or not all(isinstance(item, str) and item.strip() for item in citations)
    ):
        raise ValueError("citations must be a nonempty string list")
    paper_stated = _validated_scalar_mapping(
        payload["paper_stated"], label="paper_stated"
    )
    repository_choices = _validated_scalar_mapping(
        payload["repository_choices"], label="repository_choices"
    )
    overlap = set(paper_stated).intersection(repository_choices)
    if overlap:
        raise ValueError(
            "source statements and repository choices reuse keys: "
            + ", ".join(sorted(overlap))
        )
    benchmark_id = payload["benchmark_id"]
    claim_boundary = payload["claim_boundary"]
    if not isinstance(benchmark_id, str) or not benchmark_id.strip():
        raise ValueError("benchmark_id must be a nonempty string")
    if not isinstance(claim_boundary, str) or not claim_boundary.strip():
        raise ValueError("claim_boundary must be a nonempty string")
    return DimerProtocol(
        schema_version=PROTOCOL_SCHEMA_VERSION,
        benchmark_id=benchmark_id,
        citations=tuple(citations),
        paper_stated=paper_stated,
        repository_choices=repository_choices,
        claim_boundary=claim_boundary,
    )


def arrays_from_matrix_states(
    *,
    time: NDArray[np.float64],
    matrix_states: Sequence[object],
    include_fluctuation_moments: bool,
) -> Paper5DimerArrays:
    """Stack the five public ``MatrixDimerState`` fields without re-slicing."""

    samples = np.asarray(time, dtype=float)
    if len(matrix_states) != samples.size:
        raise ValueError("one matrix state is required per sampled time")
    if not matrix_states:
        raise ValueError("matrix_states must not be empty")

    def stack(field: str) -> np.ndarray:
        try:
            return np.asarray(
                [np.asarray(getattr(state, field)) for state in matrix_states],
                dtype=complex,
            )
        except AttributeError as exc:
            raise TypeError(f"matrix state is missing field {field!r}") from exc

    electron = stack("electron_density")
    coherent = stack("coherent_phonon")
    moments: dict[str, MomentSeries] = {
        COHERENT_PHONON_KEY: MomentSeries(
            values=coherent,
            axes=("time", "mode"),
            unit="dimensionless_boson_amplitude",
            convention="B_q=<b_q> in the Paper V local-mode ordering",
        )
    }
    if include_fluctuation_moments:
        moments.update(
            {
                NORMAL_PHONON_KEY: MomentSeries(
                    values=stack("phonon_density"),
                    axes=("time", "mode", "mode_prime"),
                    unit="dimensionless_boson_number",
                    convention=(
                        "connected normal fluctuation "
                        "<delta b_mode_prime^dagger delta b_mode>"
                    ),
                ),
                ANOMALOUS_PHONON_KEY: MomentSeries(
                    values=stack("anomalous_phonon_density"),
                    axes=("time", "mode", "mode_prime"),
                    unit="dimensionless_boson_pair_amplitude",
                    convention=(
                        "connected anomalous fluctuation "
                        "<delta b_mode delta b_mode_prime>"
                    ),
                ),
                CONNECTED_EPH_KEY: MomentSeries(
                    values=stack("electron_phonon_correlation"),
                    axes=("time", "mode", "row", "column"),
                    unit="dimensionless_connected_moment",
                    convention=(
                        "connected Paper V correlation C[mode,row,column]"
                    ),
                ),
            }
        )
    return Paper5DimerArrays(
        time=samples,
        time_unit="1/t_hop",
        electron_1rdm=np.asarray(electron, dtype=np.complex128),
        electronic_basis=_BASIS,
        electronic_basis_convention=_BASIS_CONVENTION,
        electron_number=1.0,
        moments=moments,
    )


def from_paper5_dimer(
    arrays: Paper5DimerArrays,
    *,
    level: Paper5Level,
    paper_stated: Mapping[str, JSONScalar],
    repository_choices: Mapping[str, JSONScalar],
    citations: tuple[str, ...],
    code_revision: str | None = None,
) -> ReducedTrajectory:
    """Attach a truthful method identity and capability set to dimer arrays."""

    if level == "exact":
        method = MethodIdentity(
            method_id="paper5.truncated_exact_holstein_dimer",
            family="exact_reference",
            approximation="finite local-phonon cutoff explicit supersystem",
            implementation="paper5.stability.exact_reference",
            reference_access="offline_only",
        )
        required = {
            COHERENT_PHONON_KEY,
            NORMAL_PHONON_KEY,
            ANOMALOUS_PHONON_KEY,
            CONNECTED_EPH_KEY,
        }
        moments = dict(arrays.moments)
    elif level == "coherent_only":
        method = MethodIdentity(
            method_id="paper5.coherent_only_holstein_dimer",
            family="equal_time_open_dynamics",
            approximation="coherent phonon field without connected feedback",
            implementation="paper5.stability.hubbard_dimer.ehrenfest_rhs",
            reference_access="none",
        )
        required = {COHERENT_PHONON_KEY}
        moments = {
            COHERENT_PHONON_KEY: arrays.moments[COHERENT_PHONON_KEY]
        }
    elif level == "equal_time_full":
        method = MethodIdentity(
            method_id="paper5.independent_31_coordinate_equal_time_dimer",
            family="equal_time_open_dynamics",
            approximation=(
                "independent complete 31-real-coordinate dimer specialization"
            ),
            implementation="paper5.stability.matrix_reference",
            reference_access="none",
        )
        required = {
            COHERENT_PHONON_KEY,
            NORMAL_PHONON_KEY,
            ANOMALOUS_PHONON_KEY,
            CONNECTED_EPH_KEY,
        }
        moments = dict(arrays.moments)
    else:
        raise ValueError(f"unknown Paper5 level {level!r}")

    missing = required.difference(arrays.moments)
    if missing:
        raise ValueError(f"{level} arrays are missing moments {sorted(missing)}")
    trajectory = ReducedTrajectory(
        time=np.asarray(arrays.time, dtype=float),
        time_unit=arrays.time_unit,
        electron_1rdm=np.asarray(arrays.electron_1rdm, dtype=np.complex128),
        electronic_basis=arrays.electronic_basis,
        electronic_basis_convention=arrays.electronic_basis_convention,
        electron_number=arrays.electron_number,
        method=method,
        provenance=TrajectoryProvenance(
            citations=citations,
            paper_stated=dict(paper_stated),
            repository_choices=dict(repository_choices),
            code_revision=code_revision,
        ),
        moments=moments,
    )
    require_structurally_valid(trajectory)
    return trajectory


def _paper5_imports() -> dict[str, object]:
    try:
        from paper5.stability.exact_reference import (
            compare_exact_and_closed_protocols,
        )
        from paper5.stability.hubbard_dimer import (
            DimerParameters,
            ehrenfest_rhs,
        )
        from paper5.stability.matrix_reference import (
            MatrixDimerState,
            closed_scalar_to_matrix_state,
            matrix_state_to_scalar_coordinates,
            scalar_to_matrix_state,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "paper5" or str(exc.name).startswith("paper5."):
            raise Paper5UnavailableError(
                "install the separate Paper V workspace or add "
                "paper_5/src to PYTHONPATH"
            ) from exc
        raise
    return {
        "DimerParameters": DimerParameters,
        "MatrixDimerState": MatrixDimerState,
        "closed_scalar_to_matrix_state": closed_scalar_to_matrix_state,
        "compare_exact_and_closed_protocols": compare_exact_and_closed_protocols,
        "ehrenfest_rhs": ehrenfest_rhs,
        "matrix_state_to_scalar_coordinates": matrix_state_to_scalar_coordinates,
        "scalar_to_matrix_state": scalar_to_matrix_state,
    }


def _choice_float(protocol: DimerProtocol, key: str) -> float:
    value = protocol.repository_choices.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"repository choice {key!r} must be numeric")
    return float(value)


def _paper_float(protocol: DimerProtocol, key: str) -> float:
    value = protocol.paper_stated.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"paper statement {key!r} must be numeric")
    return float(value)


def _sample_times(protocol: DimerProtocol) -> NDArray[np.float64]:
    start = _choice_float(protocol, "numerics.sample_start")
    stop = _choice_float(protocol, "numerics.sample_stop")
    step = _choice_float(protocol, "numerics.sample_step")
    if start != 0.0 or stop <= start or step <= 0.0:
        raise ValueError("repository sample grid must start at zero and increase")
    count = int(round((stop - start) / step))
    if count < 1 or not np.isclose(start + count * step, stop):
        raise ValueError("sample interval must be divisible by sample_step")
    return np.linspace(start, stop, count + 1, dtype=float)


def run_paper5_vertical_slice(
    protocol: DimerProtocol | str | Path,
    *,
    code_revision: str | None = None,
) -> Paper5VerticalSlice:
    """Run the short declared common-frame dimer benchmark.

    Exact wavefunctions remain private to the existing Paper V contraction
    routine.  This function returns reduced moments only and declares the exact
    trajectory ``offline_only``.
    """

    from scipy.integrate import solve_ivp

    resolved = load_dimer_protocol(protocol) if isinstance(protocol, (str, Path)) else protocol
    imports = _paper5_imports()
    DimerParameters = imports["DimerParameters"]
    MatrixDimerState = imports["MatrixDimerState"]
    compare = imports["compare_exact_and_closed_protocols"]
    closed_to_matrix = imports["closed_scalar_to_matrix_state"]
    ehrenfest_rhs = imports["ehrenfest_rhs"]
    matrix_to_scalar = imports["matrix_state_to_scalar_coordinates"]
    scalar_to_matrix = imports["scalar_to_matrix_state"]

    hopping = _paper_float(resolved, "model.t_hop")
    parameters = DimerParameters(  # type: ignore[operator]
        hopping=hopping,
        gamma=_paper_float(resolved, "model.gamma"),
        lambda_ep=_paper_float(resolved, "model.lambda"),
        drive_amplitude=_paper_float(resolved, "drive.v_over_t_hop") * hopping,
        pulse_width=_paper_float(resolved, "drive.Tp_times_t_hop") / hopping,
    )
    samples = _sample_times(resolved)
    phonon_cutoff = int(_choice_float(resolved, "numerics.phonon_cutoff"))
    relative_tolerance = _choice_float(resolved, "numerics.relative_tolerance")
    absolute_tolerance = _choice_float(resolved, "numerics.absolute_tolerance")
    maximum_step = _choice_float(resolved, "numerics.maximum_step")
    comparisons = compare(  # type: ignore[operator]
        parameters,
        sample_times=samples,
        phonon_cutoff=phonon_cutoff,
        protocols=("archive",),
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step=maximum_step,
    )
    comparison = comparisons["archive"]
    exact_native = comparison.exact_trajectory
    exact_arrays = arrays_from_matrix_states(
        time=samples,
        matrix_states=exact_native.matrix_states,
        include_fluctuation_moments=True,
    )

    full_states = tuple(
        closed_to_matrix(row)  # type: ignore[operator]
        for row in comparison.closed_coordinates
    )
    full_arrays = arrays_from_matrix_states(
        time=samples,
        matrix_states=full_states,
        include_fluctuation_moments=True,
    )

    exact_initial = exact_native.matrix_states[0]
    initial_scalar = np.asarray(
        matrix_to_scalar(exact_initial), dtype=float  # type: ignore[operator]
    )[:5]
    center_amplitude = complex(np.mean(exact_initial.coherent_phonon))
    coherent_solution = solve_ivp(
        lambda time, state: ehrenfest_rhs(  # type: ignore[operator]
            time, state, parameters
        ),
        (float(samples[0]), float(samples[-1])),
        initial_scalar,
        method="DOP853",
        t_eval=samples,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        max_step=maximum_step,
    )
    if not coherent_solution.success or coherent_solution.y.shape[1] != samples.size:
        raise RuntimeError(
            f"coherent-only propagation failed: {coherent_solution.message}"
        )
    coherent_states: list[object] = []
    for row in coherent_solution.y.T:
        scalar = np.zeros(13, dtype=float)
        scalar[:5] = row
        state = scalar_to_matrix(scalar)  # type: ignore[operator]
        shifted = MatrixDimerState(  # type: ignore[operator]
            electron_density=state.electron_density,
            coherent_phonon=state.coherent_phonon + center_amplitude,
            phonon_density=state.phonon_density,
            anomalous_phonon_density=state.anomalous_phonon_density,
            electron_phonon_correlation=state.electron_phonon_correlation,
        )
        coherent_states.append(shifted)
    coherent_arrays = arrays_from_matrix_states(
        time=samples,
        matrix_states=coherent_states,
        include_fluctuation_moments=False,
    )

    common = {
        "paper_stated": resolved.paper_stated,
        "repository_choices": resolved.repository_choices,
        "citations": resolved.citations,
        "code_revision": code_revision,
    }
    exact = from_paper5_dimer(exact_arrays, level="exact", **common)
    coherent = from_paper5_dimer(
        coherent_arrays, level="coherent_only", **common
    )
    full = from_paper5_dimer(full_arrays, level="equal_time_full", **common)
    require_comparable(exact, coherent, full)
    return Paper5VerticalSlice(
        protocol=resolved,
        exact=exact,
        coherent_only=coherent,
        equal_time_full=full,
        full_coordinate_error=np.asarray(comparison.coordinate_errors, dtype=float),
        full_block_names=tuple(comparison.block_names),
        full_block_error_norms=np.asarray(
            comparison.block_error_norms, dtype=float
        ),
    )


__all__ = [
    "DimerProtocol",
    "PROTOCOL_SCHEMA_VERSION",
    "Paper5DimerArrays",
    "Paper5Level",
    "Paper5UnavailableError",
    "Paper5VerticalSlice",
    "arrays_from_matrix_states",
    "from_paper5_dimer",
    "load_dimer_protocol",
    "run_paper5_vertical_slice",
]
