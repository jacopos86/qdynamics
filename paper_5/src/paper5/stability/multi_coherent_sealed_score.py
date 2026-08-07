"""Independent exact propagators and sealed scores for the packet holdout."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Mapping

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import eye, issparse
from scipy.sparse.linalg._expm_multiply import (  # type: ignore[attr-defined]
    LazyOperatorNormInfo,
    _exact_1_norm,
    _expm_multiply_simple_core,
    _fragment_3_1,
)

from .exact_reference import (
    _build_exact_dimer_model,
    _contract_matrix_state,
)
from .conditional_packets import (
    electron_relative_product_to_local_state,
    electron_relative_state,
)
from .hubbard_dimer import DimerParameters, GaussianSineDrive
from .matrix_reference import matrix_state_to_closed_scalar_coordinates
from .multi_coherent import multi_coherent_state
from .multi_coherent_holdout import (
    MultiCoherentHoldoutSettings,
    _SCORE_CONSUMPTION_RECEIPT,
    _frozen_source_hashes,
    _sha256,
    load_frozen_multi_coherent_model_batch,
)
from .multi_coherent_scores import (
    BoundedScoreCertificate,
    certify_bounded_score,
    closed_coordinate_distance,
    closed_coordinate_error_scores,
)


@dataclass(frozen=True)
class ExactHoldoutKetBatch:
    """Three exact ket trajectories on one frozen output grid."""

    times: np.ndarray
    state_vectors: np.ndarray
    maximum_norm_drift: float
    method: str
    function_evaluations: int


@dataclass(frozen=True)
class HoldoutScoreEvaluation:
    """Reference, resolution, and scientific outcomes in mandatory order."""

    reference_valid: bool
    numerically_resolved: bool
    scientific_passed: bool | None
    failures: tuple[str, ...]
    exact_maximum_mutual_infidelity: float
    exact_maximum_moment_disagreement: float
    certificates: dict[str, BoundedScoreCertificate]
    maximum_amplification_uncertainty: float
    robust_sensitivity_ratio: float


def _validate_exact_batch_inputs(
    initial_states: np.ndarray,
    sample_times: np.ndarray,
    *,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(initial_states, dtype=complex)
    times = np.asarray(sample_times, dtype=float)
    if states.ndim != 2 or states.shape[1] != dimension:
        raise ValueError("initial_states must have shape (members, dimension)")
    if states.shape[0] < 1:
        raise ValueError("at least one initial state is required")
    if times.ndim != 1 or times.size < 2 or not np.isclose(times[0], 0.0):
        raise ValueError("sample_times must increase from zero")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("sample_times must be strictly increasing")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(times)):
        raise ValueError("exact batch inputs must be finite")
    norms = np.linalg.norm(states, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("initial exact states must be normalized")
    return states, times


def propagate_exact_holdout_dop853(
    parameters: DimerParameters,
    initial_states: np.ndarray,
    sample_times: np.ndarray,
    *,
    drive_protocol: GaussianSineDrive,
    phonon_cutoff: int,
    relative_tolerance: float,
    absolute_tolerance: float,
    maximum_step: float,
) -> ExactHoldoutKetBatch:
    """Propagate a ket batch with adaptive DOP853 and no renormalization."""

    model = _build_exact_dimer_model(
        parameters,
        phonon_cutoff=phonon_cutoff,
    )
    states, times = _validate_exact_batch_inputs(
        initial_states,
        sample_times,
        dimension=model.static_hamiltonian.shape[0],
    )
    if min(relative_tolerance, absolute_tolerance, maximum_step) <= 0.0:
        raise ValueError("DOP853 controls must be positive")
    member_count, dimension = states.shape

    def rhs(time: float, flat_states: np.ndarray) -> np.ndarray:
        state_matrix = flat_states.reshape(member_count, dimension).T
        velocity = -1j * (
            model.static_hamiltonian @ state_matrix
            + drive_protocol.difference(float(time))
            * (model.drive_operator @ state_matrix)
        )
        return np.asarray(velocity.T).reshape(-1)

    solution = solve_ivp(
        rhs,
        (0.0, float(times[-1])),
        states.reshape(-1),
        method="DOP853",
        t_eval=times,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        max_step=maximum_step,
    )
    if not solution.success or solution.y.shape[1] != times.size:
        raise RuntimeError(f"exact DOP853 propagation failed: {solution.message}")
    trajectory = np.asarray(solution.y.T, dtype=complex).reshape(
        times.size,
        member_count,
        dimension,
    ).transpose(1, 0, 2)
    norm_drift = float(
        np.max(np.abs(np.linalg.norm(trajectory, axis=2) - 1.0))
    )
    return ExactHoldoutKetBatch(
        times=np.asarray(solution.t, dtype=float),
        state_vectors=trajectory,
        maximum_norm_drift=norm_drift,
        method="adaptive_DOP853",
        function_evaluations=int(solution.nfev),
    )


def _exponential_action_batch(
    generator,
    states: np.ndarray,
    *,
    step: float,
    relative_tolerance: float,
) -> np.ndarray:
    if step <= 0.0 or relative_tolerance <= 0.0:
        raise ValueError("exponential controls must be positive")
    operator = generator.tocsc() if issparse(generator) else np.asarray(generator)
    values = np.asarray(states, dtype=complex)
    if (
        operator.shape[0] != operator.shape[1]
        or values.ndim != 2
        or values.shape[0] != operator.shape[0]
    ):
        raise ValueError("generator and state-batch shapes are incompatible")
    dimension = operator.shape[0]
    trace = complex(operator.diagonal().sum())
    mean = trace / float(dimension)
    identity = (
        eye(dimension, format="csc", dtype=complex)
        if issparse(operator)
        else np.eye(dimension, dtype=complex)
    )
    shifted = operator - mean * identity
    one_norm = float(_exact_1_norm(shifted))
    if step * one_norm == 0.0:
        taylor_degree, scaling = 0, 1
    else:
        norm_info = LazyOperatorNormInfo(
            step * shifted,
            A_1_norm=step * one_norm,
            ell=2,
        )
        taylor_degree, scaling = _fragment_3_1(
            norm_info,
            values.shape[1],
            relative_tolerance,
            ell=2,
        )
    result = _expm_multiply_simple_core(
        shifted,
        values,
        step,
        mean,
        taylor_degree,
        scaling,
        relative_tolerance,
    )
    return np.asarray(result, dtype=complex)


def propagate_exact_holdout_midpoint(
    parameters: DimerParameters,
    initial_states: np.ndarray,
    sample_times: np.ndarray,
    *,
    drive_protocol: GaussianSineDrive,
    phonon_cutoff: int,
    integration_step: float,
    exponential_action_tolerance: float,
) -> ExactHoldoutKetBatch:
    """Propagate a ket batch with fixed unitary exponential-midpoint steps."""

    model = _build_exact_dimer_model(
        parameters,
        phonon_cutoff=phonon_cutoff,
    )
    states, times = _validate_exact_batch_inputs(
        initial_states,
        sample_times,
        dimension=model.static_hamiltonian.shape[0],
    )
    if min(integration_step, exponential_action_tolerance) <= 0.0:
        raise ValueError("midpoint controls must be positive")
    final_step = int(round(float(times[-1]) / integration_step))
    if not np.isclose(final_step * integration_step, times[-1], atol=1e-12):
        raise ValueError("integration_step must divide the final sample time")
    output_steps = np.rint(times / integration_step).astype(int)
    if not np.allclose(
        output_steps * integration_step,
        times,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("integration_step must divide every output time")
    output_lookup = {
        int(step_index): output_index
        for output_index, step_index in enumerate(output_steps)
    }
    state_matrix = states.T.copy()
    trajectory = np.empty(
        (states.shape[0], times.size, states.shape[1]),
        dtype=complex,
    )
    trajectory[:, 0, :] = states
    for step_index in range(1, final_step + 1):
        midpoint = (step_index - 0.5) * integration_step
        hamiltonian = (
            model.static_hamiltonian
            + drive_protocol.difference(float(midpoint)) * model.drive_operator
        )
        state_matrix = _exponential_action_batch(
            -1j * hamiltonian,
            state_matrix,
            step=integration_step,
            relative_tolerance=exponential_action_tolerance,
        )
        if step_index in output_lookup:
            trajectory[:, output_lookup[step_index], :] = state_matrix.T
    norm_drift = float(
        np.max(np.abs(np.linalg.norm(trajectory, axis=2) - 1.0))
    )
    return ExactHoldoutKetBatch(
        times=times.copy(),
        state_vectors=trajectory,
        maximum_norm_drift=norm_drift,
        method="unitary_exponential_midpoint",
        function_evaluations=final_step,
    )


def contract_exact_holdout_closed_coordinates(
    trajectory: ExactHoldoutKetBatch,
    parameters: DimerParameters,
    *,
    phonon_cutoff: int,
) -> np.ndarray:
    """Contract every exact ket to the established 31-coordinate output."""

    model = _build_exact_dimer_model(
        parameters,
        phonon_cutoff=phonon_cutoff,
    )
    return np.asarray(
        [
            [
                matrix_state_to_closed_scalar_coordinates(
                    _contract_matrix_state(model, state)
                )
                for state in member
            ]
            for member in trajectory.state_vectors
        ]
    )


def _validated_path_mapping(
    values: Mapping[str, np.ndarray],
    *,
    names: tuple[str, str],
    prefix: str,
    times: int,
    trailing_shape: tuple[int, ...],
    dtype,
) -> dict[str, np.ndarray]:
    if set(values) != set(names):
        raise ValueError(f"{prefix} must contain exactly {names}")
    result: dict[str, np.ndarray] = {}
    expected = (3, times, *trailing_shape)
    for name in names:
        array = np.asarray(values[name], dtype=dtype)
        if array.shape != expected or not np.all(np.isfinite(array)):
            raise ValueError(f"{prefix}[{name!r}] must have shape {expected}")
        result[name] = array
    return result


def _normalized_fidelity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    overlaps = np.sum(left.conj() * right, axis=2)
    norms = np.linalg.norm(left, axis=2) * np.linalg.norm(right, axis=2)
    return np.abs(overlaps / norms) ** 2


def evaluate_multi_coherent_holdout_scores(
    times: np.ndarray,
    *,
    model_closed: Mapping[str, np.ndarray],
    model_kets: Mapping[str, np.ndarray],
    model_normalized_work_residual: Mapping[str, np.ndarray],
    exact_closed: Mapping[str, np.ndarray],
    exact_kets: Mapping[str, np.ndarray],
    coordinate_scales: np.ndarray,
    initial_distance: float,
    settings: MultiCoherentHoldoutSettings,
) -> HoldoutScoreEvaluation:
    """Apply exact-reference, numerical, then scientific holdout gates."""

    time_array = np.asarray(times, dtype=float)
    if (
        time_array.ndim != 1
        or time_array.size < 2
        or not np.all(np.diff(time_array) > 0.0)
    ):
        raise ValueError("times must be strictly increasing")
    model_coordinate_paths = _validated_path_mapping(
        model_closed,
        names=("coarse", "fine"),
        prefix="model_closed",
        times=time_array.size,
        trailing_shape=(31,),
        dtype=float,
    )
    exact_coordinate_paths = _validated_path_mapping(
        exact_closed,
        names=("dop853", "midpoint"),
        prefix="exact_closed",
        times=time_array.size,
        trailing_shape=(31,),
        dtype=float,
    )
    ket_dimension = np.asarray(next(iter(model_kets.values()))).shape[-1]
    model_ket_paths = _validated_path_mapping(
        model_kets,
        names=("coarse", "fine"),
        prefix="model_kets",
        times=time_array.size,
        trailing_shape=(ket_dimension,),
        dtype=complex,
    )
    exact_ket_paths = _validated_path_mapping(
        exact_kets,
        names=("dop853", "midpoint"),
        prefix="exact_kets",
        times=time_array.size,
        trailing_shape=(ket_dimension,),
        dtype=complex,
    )
    work_paths = _validated_path_mapping(
        model_normalized_work_residual,
        names=("coarse", "fine"),
        prefix="model_normalized_work_residual",
        times=time_array.size,
        trailing_shape=(),
        dtype=float,
    )
    scales = np.asarray(coordinate_scales, dtype=float)
    if scales.shape != (31,) or np.any(scales <= 0.0):
        raise ValueError("coordinate_scales must be positive with shape (31,)")
    if initial_distance < 1e-6:
        raise ValueError("initial sensitivity distance is unresolved")

    exact_fidelity = _normalized_fidelity(
        exact_ket_paths["dop853"],
        exact_ket_paths["midpoint"],
    )
    exact_mutual_infidelity = float(np.max(1.0 - exact_fidelity))
    exact_moment_disagreement = float(
        max(
            closed_coordinate_distance(left, right, scales)
            for left, right in zip(
                exact_coordinate_paths["dop853"].reshape(-1, 31),
                exact_coordinate_paths["midpoint"].reshape(-1, 31),
                strict=True,
            )
        )
    )
    reference_failures: list[str] = []
    if exact_mutual_infidelity > settings.exact_mutual_infidelity_ceiling:
        reference_failures.append("exact_mutual_infidelity")
    if exact_moment_disagreement > settings.exact_moment_disagreement_ceiling:
        reference_failures.append("exact_moment_disagreement")
    if reference_failures:
        return HoldoutScoreEvaluation(
            reference_valid=False,
            numerically_resolved=False,
            scientific_passed=None,
            failures=tuple(reference_failures),
            exact_maximum_mutual_infidelity=exact_mutual_infidelity,
            exact_maximum_moment_disagreement=exact_moment_disagreement,
            certificates={},
            maximum_amplification_uncertainty=float("nan"),
            robust_sensitivity_ratio=float("nan"),
        )

    combination_scores: dict[str, dict[str, float]] = {}
    for model_name, model_path in model_coordinate_paths.items():
        for exact_name, exact_path in exact_coordinate_paths.items():
            key = f"{model_name}:{exact_name}"
            fidelity = _normalized_fidelity(
                model_ket_paths[model_name],
                exact_ket_paths[exact_name],
            )
            score_mask = (
                (time_array >= settings.score_interval[0])
                & (time_array <= settings.score_interval[1])
            )
            values: dict[str, float] = {
                "fidelity_deficit": float(1.0 - np.min(fidelity[:, score_mask]))
            }
            member_scores = [
                closed_coordinate_error_scores(
                    time_array,
                    model_path[member],
                    exact_path[member],
                    scales,
                    interval=settings.score_interval,
                )
                for member in range(3)
            ]
            values["electron_trace_distance"] = max(
                score.electron_trace_distance_maximum
                for score in member_scores
            )
            for block in ("B", "N", "A", "C"):
                values[f"{block}_rms"] = max(
                    score.block_rms[block] for score in member_scores
                )
                values[f"{block}_maximum"] = max(
                    score.block_maximum[block] for score in member_scores
                )
            combination_scores[key] = values

    ceilings = {
        "fidelity_deficit": 1.0 - settings.fidelity_floor,
        "electron_trace_distance": settings.electron_trace_distance_ceiling,
        **{
            f"{block}_rms": settings.block_rms_ceiling
            for block in ("B", "N", "A", "C")
        },
        **{
            f"{block}_maximum": settings.block_maximum_ceiling
            for block in ("B", "N", "A", "C")
        },
    }
    authoritative = combination_scores["fine:dop853"]
    certificates: dict[str, BoundedScoreCertificate] = {}
    for score_name, ceiling in ceilings.items():
        certificates[score_name] = certify_bounded_score(
            authoritative=authoritative[score_name],
            cross_combination_scores=np.asarray(
                [values[score_name] for values in combination_scores.values()]
            ),
            ceiling=ceiling,
            resolution_fraction=settings.numerical_resolution_fraction,
        )
    work_values = np.asarray(
        [
            np.max(np.abs(work_paths[resolution]))
            for resolution in ("coarse", "fine")
        ]
    )
    certificates["work_residual"] = certify_bounded_score(
        authoritative=float(np.max(np.abs(work_paths["fine"]))),
        cross_combination_scores=work_values,
        ceiling=settings.work_residual_ceiling,
        resolution_fraction=settings.numerical_resolution_fraction,
    )

    def amplification(path: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                closed_coordinate_distance(plus, minus, scales)
                / initial_distance
                for plus, minus in zip(path[1], path[2], strict=True)
            ]
        )

    model_amplification = {
        name: amplification(path)
        for name, path in model_coordinate_paths.items()
    }
    exact_amplification = {
        name: amplification(path)
        for name, path in exact_coordinate_paths.items()
    }
    amplification_uncertainty = np.maximum(
        np.abs(model_amplification["fine"] - model_amplification["coarse"]),
        np.abs(exact_amplification["dop853"] - exact_amplification["midpoint"]),
    )
    sensitivity_mask = (
        (time_array >= settings.sensitivity_interval[0])
        & (time_array <= settings.sensitivity_interval[1])
    )
    maximum_amplification_uncertainty = float(
        np.max(amplification_uncertainty[sensitivity_mask])
    )
    robust_ratio = (
        model_amplification["fine"] + amplification_uncertainty
    ) / np.maximum(
        1.0,
        exact_amplification["dop853"] - amplification_uncertainty,
    )
    robust_sensitivity_ratio = float(np.max(robust_ratio[sensitivity_mask]))
    numerically_resolved = (
        all(value.numerically_resolved for value in certificates.values())
        and maximum_amplification_uncertainty
        <= settings.amplification_uncertainty_ceiling
    )
    if not numerically_resolved:
        failures = [
            f"unresolved:{name}"
            for name, value in certificates.items()
            if not value.numerically_resolved
        ]
        if (
            maximum_amplification_uncertainty
            > settings.amplification_uncertainty_ceiling
        ):
            failures.append("unresolved:sensitivity_amplification")
        return HoldoutScoreEvaluation(
            reference_valid=True,
            numerically_resolved=False,
            scientific_passed=None,
            failures=tuple(failures),
            exact_maximum_mutual_infidelity=exact_mutual_infidelity,
            exact_maximum_moment_disagreement=exact_moment_disagreement,
            certificates=certificates,
            maximum_amplification_uncertainty=(
                maximum_amplification_uncertainty
            ),
            robust_sensitivity_ratio=robust_sensitivity_ratio,
        )
    scientific_failures = [
        f"scientific:{name}"
        for name, value in certificates.items()
        if not value.passes
    ]
    if robust_sensitivity_ratio > settings.sensitivity_ratio_ceiling:
        scientific_failures.append("scientific:sensitivity_ratio")
    return HoldoutScoreEvaluation(
        reference_valid=True,
        numerically_resolved=True,
        scientific_passed=not scientific_failures,
        failures=tuple(scientific_failures),
        exact_maximum_mutual_infidelity=exact_mutual_infidelity,
        exact_maximum_moment_disagreement=exact_moment_disagreement,
        certificates=certificates,
        maximum_amplification_uncertainty=maximum_amplification_uncertainty,
        robust_sensitivity_ratio=robust_sensitivity_ratio,
    )


def _load_blind_model_score_paths(batch) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, float],
]:
    prepared = batch.prepared
    cutoff = prepared.settings.phonon_cutoff
    relative_dimension = 2 * cutoff + 1
    center_state = electron_relative_state(
        prepared.exact_initial_state_vectors[0],
        phonon_cutoff=cutoff,
    ).center_state
    closed_paths: dict[str, np.ndarray] = {}
    ket_paths: dict[str, np.ndarray] = {}
    work_paths: dict[str, np.ndarray] = {}
    minimum_retained_norm: dict[str, float] = {}
    common_times: np.ndarray | None = None
    members = ("central", "plus", "minus")
    for resolution in ("coarse", "fine"):
        member_closed: list[np.ndarray] = []
        member_kets: list[np.ndarray] = []
        member_work: list[np.ndarray] = []
        retained: list[float] = []
        for member in members:
            path = batch.directory / f"{resolution}_{member}" / (
                "segmented_horizon.npz"
            )
            with np.load(path) as arrays:
                times = np.asarray(arrays["times"], dtype=float)
                if common_times is None:
                    common_times = times
                elif not np.array_equal(times, common_times):
                    raise ValueError("blind model output grids do not agree")
                closed = np.asarray(arrays["closed_coordinates"], dtype=float)
                work = np.asarray(
                    arrays["normalized_energy_work_residual"],
                    dtype=float,
                )
                parameter_path = np.asarray(
                    arrays["parameter_trajectory"],
                    dtype=float,
                )
                packet_counts = np.asarray(
                    arrays["packet_count_trajectory"],
                    dtype=int,
                )
            local_states: list[np.ndarray] = []
            for index, packet_count in enumerate(packet_counts):
                relative_state = multi_coherent_state(
                    parameter_path[index, : 16 * int(packet_count)],
                    relative_dimension=relative_dimension,
                )
                embedded = electron_relative_product_to_local_state(
                    relative_state,
                    center_state,
                    phonon_cutoff=cutoff,
                )
                local_states.append(embedded.state)
                retained.append(embedded.retained_norm)
            member_closed.append(closed)
            member_kets.append(np.asarray(local_states))
            member_work.append(work)
        closed_paths[resolution] = np.asarray(member_closed)
        ket_paths[resolution] = np.asarray(member_kets)
        work_paths[resolution] = np.asarray(member_work)
        minimum_retained_norm[resolution] = float(np.min(retained))
    if common_times is None:
        raise RuntimeError("blind model batch has no trajectories")
    return (
        common_times,
        closed_paths,
        ket_paths,
        work_paths,
        minimum_retained_norm,
    )


def _run_exact_score_attempt(batch, times: np.ndarray, *, refined: bool):
    prepared = batch.prepared
    settings = prepared.settings
    drive = settings.drive_protocol(prepared.parameters)
    if refined:
        midpoint_step = settings.exact_midpoint_refined_step
        dop_relative = settings.exact_dop853_refined_relative_tolerance
        dop_absolute = settings.exact_dop853_refined_absolute_tolerance
        dop_step = settings.exact_dop853_refined_maximum_step
    else:
        midpoint_step = settings.exact_midpoint_step
        dop_relative = settings.exact_dop853_relative_tolerance
        dop_absolute = settings.exact_dop853_absolute_tolerance
        dop_step = settings.exact_dop853_maximum_step
    dop853 = propagate_exact_holdout_dop853(
        prepared.parameters,
        prepared.exact_initial_state_vectors,
        times,
        drive_protocol=drive,
        phonon_cutoff=settings.phonon_cutoff,
        relative_tolerance=dop_relative,
        absolute_tolerance=dop_absolute,
        maximum_step=dop_step,
    )
    midpoint = propagate_exact_holdout_midpoint(
        prepared.parameters,
        prepared.exact_initial_state_vectors,
        times,
        drive_protocol=drive,
        phonon_cutoff=settings.phonon_cutoff,
        integration_step=midpoint_step,
        exponential_action_tolerance=(
            settings.exact_exponential_action_tolerance
        ),
    )
    exact_kets = {
        "dop853": dop853.state_vectors,
        "midpoint": midpoint.state_vectors,
    }
    exact_closed = {
        "dop853": contract_exact_holdout_closed_coordinates(
            dop853,
            prepared.parameters,
            phonon_cutoff=settings.phonon_cutoff,
        ),
        "midpoint": contract_exact_holdout_closed_coordinates(
            midpoint,
            prepared.parameters,
            phonon_cutoff=settings.phonon_cutoff,
        ),
    }
    return dop853, midpoint, exact_kets, exact_closed


def run_exact_holdout_cost_once(
    prepared_directory: Path,
    batch_directory: Path,
    output_directory: Path,
    *,
    method: str,
    refined: bool,
) -> dict:
    """Measure one direct central trajectory after the score receipt exists."""

    if method not in ("dop853", "midpoint"):
        raise ValueError("method must be dop853 or midpoint")
    batch = load_frozen_multi_coherent_model_batch(
        prepared_directory,
        batch_directory,
    )
    receipt_path = batch.directory / _SCORE_CONSUMPTION_RECEIPT
    if not receipt_path.is_file():
        raise RuntimeError("exact cost cannot run before scorer consumption")
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"exact cost output exists: {output}")
    output.mkdir(parents=True)
    with np.load(
        batch.directory / "fine_central" / "segmented_horizon.npz"
    ) as arrays:
        times = np.asarray(arrays["times"], dtype=float)
    prepared = batch.prepared
    settings = prepared.settings
    drive = settings.drive_protocol(prepared.parameters)
    initial = prepared.exact_initial_state_vectors[:1]
    wall_start = time.monotonic()
    if method == "dop853":
        trajectory = propagate_exact_holdout_dop853(
            prepared.parameters,
            initial,
            times,
            drive_protocol=drive,
            phonon_cutoff=settings.phonon_cutoff,
            relative_tolerance=(
                settings.exact_dop853_refined_relative_tolerance
                if refined
                else settings.exact_dop853_relative_tolerance
            ),
            absolute_tolerance=(
                settings.exact_dop853_refined_absolute_tolerance
                if refined
                else settings.exact_dop853_absolute_tolerance
            ),
            maximum_step=(
                settings.exact_dop853_refined_maximum_step
                if refined
                else settings.exact_dop853_maximum_step
            ),
        )
    else:
        trajectory = propagate_exact_holdout_midpoint(
            prepared.parameters,
            initial,
            times,
            drive_protocol=drive,
            phonon_cutoff=settings.phonon_cutoff,
            integration_step=(
                settings.exact_midpoint_refined_step
                if refined
                else settings.exact_midpoint_step
            ),
            exponential_action_tolerance=(
                settings.exact_exponential_action_tolerance
            ),
        )
    closed = contract_exact_holdout_closed_coordinates(
        trajectory,
        prepared.parameters,
        phonon_cutoff=settings.phonon_cutoff,
    )
    del closed
    wall_seconds = float(time.monotonic() - wall_start)
    maximum_resident_set = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    maximum_resident_set_bytes = int(
        maximum_resident_set
        if sys.platform == "darwin"
        else maximum_resident_set * 1024
    )
    result = {
        "schema": "paper5.multi_coherent.exact_cost_repeat.v1",
        "method": method,
        "refined": refined,
        "wall_seconds": wall_seconds,
        "maximum_resident_set_bytes": maximum_resident_set_bytes,
        "maximum_norm_drift": trajectory.maximum_norm_drift,
        "function_evaluations": trajectory.function_evaluations,
        "prepared_manifest_sha256": prepared.manifest_sha256,
        "model_batch_manifest_sha256": batch.manifest_sha256,
        "consumption_receipt_sha256": _sha256(receipt_path),
        "source_hashes": _frozen_source_hashes(),
    }
    (output / "cost.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _evaluate_cost_gate(
    prepared_directory: Path,
    batch_directory: Path,
    batch,
    output: Path,
    *,
    refined: bool,
) -> dict:
    settings = batch.prepared.settings
    cost_root = output / "exact_cost_repeats"
    cost_root.mkdir()
    direct_records: dict[str, list[dict]] = {}
    for method in ("dop853", "midpoint"):
        records: list[dict] = []
        for repeat in range(1, settings.cost_repeat_count + 1):
            repeat_directory = cost_root / f"{method}_{repeat}"
            command = [
                sys.executable,
                "-m",
                "paper5.stability.multi_coherent_holdout_cli",
                "exact-cost-once",
                "--prepared-directory",
                str(prepared_directory),
                "--batch-directory",
                str(batch_directory),
                "--output-directory",
                str(repeat_directory),
                "--method",
                method,
            ]
            if refined:
                command.append("--refined")
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            record = json.loads(
                (repeat_directory / "cost.json").read_text(encoding="utf-8")
            )
            records.append(record)
        direct_records[method] = records
    medians = {
        method: float(
            np.median([record["wall_seconds"] for record in records])
        )
        for method, records in direct_records.items()
    }
    selected_method = min(medians, key=medians.get)
    selected_records = direct_records[selected_method]
    direct_wall = medians[selected_method]
    direct_resident = max(
        int(record["maximum_resident_set_bytes"])
        for record in selected_records
    )
    model_cost_path = batch.directory / "model_cost_manifest.json"
    model_cost = json.loads(model_cost_path.read_text(encoding="utf-8"))
    model_wall = float(model_cost["median_wall_seconds"])
    model_resident = int(
        model_cost["maximum_resident_set_bytes_over_repeats"]
    )
    wall_ratio = model_wall / direct_wall
    resident_ratio = model_resident / direct_resident
    ceiling = settings.model_to_direct_cost_ratio_ceiling
    failures: list[str] = []
    if wall_ratio >= ceiling:
        failures.append("scientific:cost_wall_ratio")
    if resident_ratio >= ceiling:
        failures.append("scientific:cost_resident_ratio")
    return {
        "evaluated": True,
        "repeat_count": settings.cost_repeat_count,
        "selected_direct_method": selected_method,
        "direct_median_wall_seconds": direct_wall,
        "direct_maximum_resident_set_bytes": direct_resident,
        "model_median_wall_seconds": model_wall,
        "model_maximum_resident_set_bytes": model_resident,
        "model_to_direct_wall_ratio": wall_ratio,
        "model_to_direct_resident_ratio": resident_ratio,
        "ratio_ceiling": ceiling,
        "passed": not failures,
        "failures": failures,
        "records": direct_records,
    }


def score_frozen_multi_coherent_holdout(
    prepared_directory: Path,
    batch_directory: Path,
    output_directory: Path,
) -> dict:
    """Consume one frozen model batch, open the reference, and score once."""

    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"score output already exists: {output}")
    batch = load_frozen_multi_coherent_model_batch(
        prepared_directory,
        batch_directory,
        require_unconsumed=True,
    )
    output.mkdir(parents=True)
    receipt_path = batch.directory / _SCORE_CONSUMPTION_RECEIPT
    receipt = {
        "schema": "paper5.multi_coherent.score_consumption.v1",
        "status": "exact_holdout_opened_model_batch_consumed",
        "prepared_manifest_sha256": batch.prepared.manifest_sha256,
        "model_batch_manifest_sha256": batch.manifest_sha256,
        "source_hashes": _frozen_source_hashes(),
    }
    with receipt_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    (
        times,
        model_closed,
        model_kets,
        model_work,
        minimum_local_retained_norm,
    ) = _load_blind_model_score_paths(batch)
    attempt_records: list[dict] = []
    selected = None
    for refined in (False, True):
        dop853, midpoint, exact_kets, exact_closed = _run_exact_score_attempt(
            batch,
            times,
            refined=refined,
        )
        evaluation = evaluate_multi_coherent_holdout_scores(
            times,
            model_closed=model_closed,
            model_kets=model_kets,
            model_normalized_work_residual=model_work,
            exact_closed=exact_closed,
            exact_kets=exact_kets,
            coordinate_scales=batch.prepared.coordinate_scales,
            initial_distance=(
                batch.prepared.initial_conditions.initial_distance
            ),
            settings=batch.prepared.settings,
        )
        attempt_records.append(
            {
                "refined": refined,
                "reference_valid": evaluation.reference_valid,
                "numerically_resolved": evaluation.numerically_resolved,
                "exact_maximum_mutual_infidelity": (
                    evaluation.exact_maximum_mutual_infidelity
                ),
                "exact_maximum_moment_disagreement": (
                    evaluation.exact_maximum_moment_disagreement
                ),
                "dop853_function_evaluations": dop853.function_evaluations,
                "midpoint_steps": midpoint.function_evaluations,
            }
        )
        selected = (
            evaluation,
            dop853,
            midpoint,
            exact_kets,
            exact_closed,
            refined,
        )
        if evaluation.reference_valid and evaluation.numerically_resolved:
            break
    if selected is None:
        raise RuntimeError("exact score attempt was not executed")
    evaluation, dop853, midpoint, exact_kets, exact_closed, refined = selected
    cost_evaluation = {
        "evaluated": False,
        "repeat_count": batch.prepared.settings.cost_repeat_count,
        "passed": None,
        "failures": [],
    }
    if evaluation.reference_valid and evaluation.numerically_resolved:
        cost_evaluation = _evaluate_cost_gate(
            prepared_directory,
            batch_directory,
            batch,
            output,
            refined=refined,
        )
    overall_failures = [
        *evaluation.failures,
        *cost_evaluation.get("failures", []),
    ]
    scientific_passed = evaluation.scientific_passed
    if scientific_passed is not None and cost_evaluation["evaluated"]:
        scientific_passed = bool(
            scientific_passed and cost_evaluation["passed"]
        )
    if not evaluation.reference_valid:
        status = "indeterminate_reference_stop"
    elif not evaluation.numerically_resolved:
        status = "indeterminate_numerical_stop"
    elif scientific_passed:
        status = "scientific_pass"
    else:
        status = "scientific_failure"
    arrays_path = output / "score_arrays.npz"
    np.savez_compressed(
        arrays_path,
        times=times,
        coordinate_scales=batch.prepared.coordinate_scales,
        model_coarse_closed=model_closed["coarse"],
        model_fine_closed=model_closed["fine"],
        exact_dop853_closed=exact_closed["dop853"],
        exact_midpoint_closed=exact_closed["midpoint"],
        exact_dop853_state_vectors=exact_kets["dop853"],
        exact_midpoint_state_vectors=exact_kets["midpoint"],
    )
    summary = {
        "schema": "paper5.multi_coherent.sealed_score.v1",
        "status": status,
        "prepared_manifest_sha256": batch.prepared.manifest_sha256,
        "model_batch_manifest_sha256": batch.manifest_sha256,
        "refinement_used": refined,
        "attempts": attempt_records,
        "reference_valid": evaluation.reference_valid,
        "numerically_resolved": evaluation.numerically_resolved,
        "scientific_passed": scientific_passed,
        "failures": overall_failures,
        "cost_evaluation": cost_evaluation,
        "exact_maximum_mutual_infidelity": (
            evaluation.exact_maximum_mutual_infidelity
        ),
        "exact_maximum_moment_disagreement": (
            evaluation.exact_maximum_moment_disagreement
        ),
        "certificates": {
            name: asdict(value)
            for name, value in evaluation.certificates.items()
        },
        "maximum_amplification_uncertainty": (
            evaluation.maximum_amplification_uncertainty
        ),
        "robust_sensitivity_ratio": evaluation.robust_sensitivity_ratio,
        "minimum_model_local_cutoff_retained_norm": (
            minimum_local_retained_norm
        ),
        "exact_norm_drift": {
            "dop853": dop853.maximum_norm_drift,
            "midpoint": midpoint.maximum_norm_drift,
        },
    }
    summary_path = output / "score_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runtime_manifest = {
        "schema": "paper5.multi_coherent.sealed_score_runtime.v1",
        "status": status,
        "source_hashes": _frozen_source_hashes(),
        "input_hashes": {
            "prepared_manifest": batch.prepared.manifest_sha256,
            "model_batch_manifest": batch.manifest_sha256,
            "consumption_receipt": _sha256(receipt_path),
        },
        "artifact_hashes": {
            arrays_path.name: _sha256(arrays_path),
            summary_path.name: _sha256(summary_path),
        },
    }
    (output / "runtime_manifest.json").write_text(
        json.dumps(runtime_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "ExactHoldoutKetBatch",
    "HoldoutScoreEvaluation",
    "contract_exact_holdout_closed_coordinates",
    "evaluate_multi_coherent_holdout_scores",
    "propagate_exact_holdout_dop853",
    "propagate_exact_holdout_midpoint",
    "score_frozen_multi_coherent_holdout",
]
