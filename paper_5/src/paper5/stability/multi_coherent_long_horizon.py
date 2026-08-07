"""Checkpointed long-horizon propagation for the multi-coherent ansatz."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy.integrate import solve_ivp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .conditional_packets import electron_relative_state
from .exact_reference import (
    exact_holstein_moment_hierarchy_trajectory,
    exact_holstein_wavefunction_trajectory_for_diagnostics,
)
from .hubbard_dimer import DimerParameters, GaussianSineDrive
from .moment_hierarchy import moment_hierarchy
from .matrix_reference import matrix_state_to_closed_scalar_coordinates
from .multi_coherent import (
    multi_coherent_observables,
    multi_coherent_capacity,
    multi_coherent_rhs,
    multi_coherent_state,
    project_schrodinger_velocity,
    relative_holstein_hamiltonian,
    relative_state_closed_coordinates,
    relative_state_moment_coordinates,
    retract_multi_coherent_parameters,
    spawn_residual_coherent_packets,
)
from .multi_coherent_scores import energy_work_residual
from .multi_coherent_propagation import (
    _center_amplitude,
    _load_gate_initial_parameters,
    _uncertainty_margin,
)


class _SegmentTimeout(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _write_progress(
    path: Path,
    *,
    status: str,
    target_time: float,
    completed_times: list[float],
    segment_records: list[dict[str, Any]],
    spawn_records: list[dict[str, Any]] | None = None,
    spawn_attempt_records: list[dict[str, Any]] | None = None,
    failure: dict[str, Any] | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": status,
                "target_time": target_time,
                "last_completed_time": completed_times[-1],
                "completed_segment_count": len(segment_records),
                "segments": segment_records,
                "spawns": spawn_records or [],
                "spawn_attempts": spawn_attempt_records or [],
                "failure": failure,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _padded_parameter_trajectory(
    parameter_states: list[np.ndarray],
) -> np.ndarray:
    """Return variable-capacity parameter states in one NaN-padded array."""

    width = max(state.size for state in parameter_states)
    trajectory = np.full((len(parameter_states), width), np.nan)
    for index, state in enumerate(parameter_states):
        trajectory[index, : state.size] = state
    return trajectory


def _write_plot(
    path: Path,
    times: np.ndarray,
    segment_end_times: np.ndarray,
    segment_wall_seconds: np.ndarray,
    segment_function_evaluations: np.ndarray,
    norm: np.ndarray,
    minimum_electron_eigenvalue: np.ndarray,
    uncertainty_margin: np.ndarray,
    state_fidelity: np.ndarray | None,
    coordinate_relative_error: np.ndarray | None,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(10.7, 3.2))
    axes[0].plot(
        segment_end_times,
        segment_wall_seconds,
        color="#D28E2B",
        label="wall seconds",
    )
    secondary = axes[0].twinx()
    secondary.plot(
        segment_end_times,
        segment_function_evaluations,
        color="#6E3CBC",
        label="RHS evaluations",
    )
    axes[0].set_ylabel("segment wall seconds")
    secondary.set_ylabel("segment RHS evaluations")
    axes[0].set_xlabel(r"segment endpoint $t\,t_{\rm hop}$")
    axes[0].grid(alpha=0.22)

    if state_fidelity is not None and coordinate_relative_error is not None:
        axes[1].plot(
            times,
            state_fidelity,
            color="#2378B5",
            label="state fidelity",
        )
        axes[1].plot(
            times,
            1.0 - coordinate_relative_error,
            color="#44A047",
            label="1 - coordinate error",
        )
        axes[1].set_ylabel("offline exact agreement")
        axes[1].legend(frameon=False, fontsize=7.5)
    else:
        axes[1].semilogy(
            times,
            np.maximum(np.abs(norm - 1.0), 1e-14),
            color="#D28E2B",
        )
        axes[1].set_ylabel("norm drift")
    axes[1].set_xlabel(r"dimensionless time $t\,t_{\rm hop}$")
    axes[1].grid(alpha=0.22)

    axes[2].plot(
        times,
        minimum_electron_eigenvalue,
        color="#2378B5",
        label=r"$\lambda_{\min}(\rho)$",
    )
    axes[2].plot(
        times,
        uncertainty_margin,
        color="#44A047",
        label=r"$\det V-1/4$",
    )
    axes[2].axhline(0.0, color="#555555", linestyle="--", linewidth=0.8)
    axes[2].set_xlabel(r"dimensionless time $t\,t_{\rm hop}$")
    axes[2].set_ylabel("physicality margin")
    axes[2].legend(frameon=False, fontsize=7.5)
    axes[2].grid(alpha=0.22)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run_segmented_multi_coherent_horizon(
    run_directory: Path,
    *,
    gate_directory: Path | None,
    parameters: DimerParameters,
    final_time: float = 20.0,
    segment_length: float = 0.5,
    output_sample_step: float | None = None,
    segment_timeout_seconds: float = 30.0,
    maximum_step: float = 0.01,
    relative_tolerance: float = 1e-7,
    absolute_tolerance: float = 1e-9,
    phonon_cutoff: int = 20,
    hierarchy_degree: int = 4,
    packet_count: int = 4,
    tangent_singular_value_cutoff: float = 1e-2,
    tangent_regularization: str = "tikhonov",
    relative_damping: float = 3e-3,
    adaptive_capacity: bool = False,
    maximum_packet_count: int = 6,
    spawn_relative_residual_threshold: float = 5e-2,
    spawn_absolute_residual_threshold: float = 2e-2,
    spawn_fit_maximum_iterations: int = 40,
    spawn_fit_population_size: int = 6,
    spawn_seed: int = 260803,
    compare_exact: bool = True,
    drive_protocol: GaussianSineDrive | None = None,
    initial_parameters_override: np.ndarray | None = None,
) -> dict[str, Any]:
    """Advance in recoverable segments and stop at the first costly segment."""

    run_wall_start = time.monotonic()

    if final_time <= 0.0 or segment_length <= 0.0:
        raise ValueError("final_time and segment_length must be positive")
    segment_count = int(round(final_time / segment_length))
    if not np.isclose(segment_count * segment_length, final_time, atol=1e-12):
        raise ValueError("final_time must be divisible by segment_length")
    if segment_timeout_seconds <= 0.0:
        raise ValueError("segment_timeout_seconds must be positive")
    sample_step = (
        segment_length
        if output_sample_step is None
        else float(output_sample_step)
    )
    if sample_step <= 0.0:
        raise ValueError("output_sample_step must be positive")
    samples_per_segment = int(round(segment_length / sample_step))
    if samples_per_segment < 1 or not np.isclose(
        samples_per_segment * sample_step,
        segment_length,
        atol=1e-12,
    ):
        raise ValueError(
            "segment_length must be an integer multiple of output_sample_step"
        )
    if packet_count < 1:
        raise ValueError("packet_count must be positive")
    if maximum_packet_count < packet_count:
        raise ValueError("maximum_packet_count cannot be below packet_count")
    if (
        spawn_relative_residual_threshold < 0.0
        or spawn_absolute_residual_threshold < 0.0
    ):
        raise ValueError("spawn residual thresholds must be nonnegative")
    if compare_exact and drive_protocol is not None:
        default_drive = GaussianSineDrive.from_parameters(parameters)
        if drive_protocol != default_drive:
            raise ValueError(
                "nondefault drive exact comparison requires the sealed scorer"
            )
    run_directory.mkdir(parents=True, exist_ok=True)
    progress_path = run_directory / "progress.json"
    checkpoint_path = run_directory / "checkpoint.npz"
    gate_initial_parameters: np.ndarray | None = None
    gate_hashes: dict[str, str] = {}
    if gate_directory is not None:
        gate_initial_parameters, gate_hashes = _load_gate_initial_parameters(
            gate_directory,
            packet_count=packet_count,
        )
    relative_dimension = 2 * phonon_cutoff + 1
    if initial_parameters_override is None:
        if gate_initial_parameters is None:
            raise ValueError(
                "gate_directory is required without initial_parameters_override"
            )
        initial_parameters = gate_initial_parameters
        initialization_source = "gate_t0_fit"
    else:
        initial_parameters = np.asarray(
            initial_parameters_override,
            dtype=float,
        ).copy()
        expected_shape = (16 * packet_count,)
        if initial_parameters.shape != expected_shape:
            raise ValueError(
                "initial_parameters_override must match the declared packet count"
            )
        if (
            gate_initial_parameters is not None
            and initial_parameters.shape != gate_initial_parameters.shape
        ):
            raise ValueError(
                "initial_parameters_override must match the gate parameter shape"
            )
        if not np.all(np.isfinite(initial_parameters)):
            raise ValueError("initial_parameters_override must be finite")
        initialization_source = "explicit_model_chart"
    initial_parameters = retract_multi_coherent_parameters(
        initial_parameters,
        relative_dimension=relative_dimension,
    )
    initial_parameter_sha256 = _array_sha256(initial_parameters)
    hierarchy = moment_hierarchy(hierarchy_degree)
    completed_times = [0.0]
    parameter_states = [initial_parameters.copy()]
    packet_counts = [packet_count]
    segment_records: list[dict[str, Any]] = []
    spawn_records: list[dict[str, Any]] = []
    spawn_attempt_records: list[dict[str, Any]] = []

    def rhs(current_time: float, state: np.ndarray) -> np.ndarray:
        return multi_coherent_rhs(
            current_time,
            state,
            parameters,
            relative_dimension=relative_dimension,
            drive_protocol=drive_protocol,
            relative_singular_value_cutoff=tangent_singular_value_cutoff,
            regularization=tangent_regularization,
            relative_damping=relative_damping,
        )

    def timeout_handler(signum: int, frame: Any) -> None:
        del signum, frame
        raise _SegmentTimeout

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    failure: dict[str, Any] | None = None
    try:
        for segment_index in range(segment_count):
            start_time = segment_index * segment_length
            end_time = (segment_index + 1) * segment_length
            segment_sample_times = start_time + sample_step * np.arange(
                1,
                samples_per_segment + 1,
            )
            segment_sample_times[-1] = end_time
            segment_initial = parameter_states[-1]
            wall_start = time.monotonic()
            signal.setitimer(signal.ITIMER_REAL, segment_timeout_seconds)
            try:
                solution = solve_ivp(
                    rhs,
                    (start_time, end_time),
                    segment_initial,
                    method="DOP853",
                    t_eval=segment_sample_times,
                    rtol=relative_tolerance,
                    atol=absolute_tolerance,
                    max_step=maximum_step,
                )
            except _SegmentTimeout:
                failure = {
                    "kind": "segment_timeout",
                    "segment_start": start_time,
                    "segment_end": end_time,
                    "timeout_seconds": segment_timeout_seconds,
                }
                break
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
            wall_seconds = time.monotonic() - wall_start
            if (
                not solution.success
                or solution.y.shape[1] != samples_per_segment
            ):
                failure = {
                    "kind": "solver_failure",
                    "segment_start": start_time,
                    "segment_end": end_time,
                    "message": str(solution.message),
                }
                break
            sampled_parameters = [
                retract_multi_coherent_parameters(
                    np.asarray(solution.y[:, sample_index], dtype=float),
                    relative_dimension=relative_dimension,
                )
                for sample_index in range(samples_per_segment)
            ]
            endpoint = sampled_parameters[-1]
            endpoint_packet_count = endpoint.size // 16
            segment_record: dict[str, Any] = {
                "segment_start": start_time,
                "segment_end": end_time,
                "wall_seconds": wall_seconds,
                "function_evaluations": int(solution.nfev),
                "packet_count_start": int(segment_initial.size // 16),
                "packet_count_end": endpoint_packet_count,
                "spawned": False,
            }
            if (
                adaptive_capacity
                and end_time < final_time - 1e-12
                and endpoint_packet_count < maximum_packet_count
            ):
                projection = project_schrodinger_velocity(
                    endpoint,
                    relative_holstein_hamiltonian(
                        end_time,
                        parameters,
                        relative_dimension=relative_dimension,
                        drive_protocol=drive_protocol,
                    ),
                    relative_dimension=relative_dimension,
                    relative_singular_value_cutoff=(
                        tangent_singular_value_cutoff
                    ),
                    regularization=tangent_regularization,
                    relative_damping=relative_damping,
                )
                target_speed = float(np.linalg.norm(projection.target_velocity))
                segment_record.update(
                    {
                        "online_tangent_relative_residual": (
                            projection.relative_residual
                        ),
                        "online_tangent_absolute_residual": (
                            projection.absolute_residual
                        ),
                        "online_target_state_speed": target_speed,
                    }
                )
                should_spawn = (
                    projection.relative_residual
                    >= spawn_relative_residual_threshold
                    and projection.absolute_residual
                    >= spawn_absolute_residual_threshold
                )
                if should_spawn:
                    spawn = spawn_residual_coherent_packets(
                        endpoint,
                        projection.target_velocity
                        - projection.projected_velocity,
                        relative_dimension=relative_dimension,
                        maximum_iterations=spawn_fit_maximum_iterations,
                        population_size=spawn_fit_population_size,
                        seed=spawn_seed + 1000 * len(spawn_records),
                    )
                    if (
                        spawn.state_discontinuity > 1e-12
                        or spawn.norm_change > 1e-12
                    ):
                        raise RuntimeError(
                            "packet spawn did not preserve the represented state"
                        )
                    candidate_projection = project_schrodinger_velocity(
                        spawn.parameters,
                        relative_holstein_hamiltonian(
                            end_time,
                            parameters,
                            relative_dimension=relative_dimension,
                        ),
                        relative_dimension=relative_dimension,
                        relative_singular_value_cutoff=(
                            tangent_singular_value_cutoff
                        ),
                        regularization=tangent_regularization,
                        relative_damping=relative_damping,
                    )
                    accepted = (
                        candidate_projection.absolute_residual
                        < projection.absolute_residual
                    )
                    spawn_record: dict[str, Any] = {
                        "time": end_time,
                        "previous_packet_count": spawn.previous_packet_count,
                        "packet_count": spawn.packet_count,
                        "trigger_relative_residual": (
                            projection.relative_residual
                        ),
                        "trigger_absolute_residual": (
                            projection.absolute_residual
                        ),
                        "target_state_speed": target_speed,
                        "candidate_relative_residual": (
                            candidate_projection.relative_residual
                        ),
                        "candidate_absolute_residual": (
                            candidate_projection.absolute_residual
                        ),
                        "residual_reduction_fraction": float(
                            (
                                projection.absolute_residual
                                - candidate_projection.absolute_residual
                            )
                            / max(
                                projection.absolute_residual,
                                np.finfo(float).tiny,
                            )
                        ),
                        "accepted": accepted,
                        "parent_electronic_index": (
                            spawn.parent_electronic_index
                        ),
                        "centers": [
                            [float(center.real), float(center.imag)]
                            for center in spawn.centers
                        ],
                        "residual_block_norms": list(
                            spawn.residual_block_norms
                        ),
                        "fit_fidelities": list(spawn.fit_fidelities),
                        "fit_successes": list(spawn.fit_successes),
                        "fit_function_evaluations": (
                            spawn.function_evaluations
                        ),
                        "state_discontinuity": spawn.state_discontinuity,
                        "norm_change": spawn.norm_change,
                    }
                    spawn_attempt_records.append(spawn_record)
                    if accepted:
                        endpoint = spawn.parameters
                        endpoint_packet_count = spawn.packet_count
                        spawn_records.append(spawn_record)
                        segment_record["spawned"] = True
                        segment_record["packet_count_end"] = (
                            endpoint_packet_count
                        )
            sampled_parameters[-1] = endpoint
            completed_times.extend(
                float(value) for value in segment_sample_times
            )
            parameter_states.extend(sampled_parameters)
            packet_counts.extend(
                [int(segment_initial.size // 16)] * samples_per_segment
            )
            packet_counts[-1] = endpoint_packet_count
            segment_records.append(segment_record)
            np.savez_compressed(
                checkpoint_path,
                times=np.asarray(completed_times),
                parameter_trajectory=_padded_parameter_trajectory(
                    parameter_states
                ),
                packet_count_trajectory=np.asarray(packet_counts, dtype=int),
                segment_wall_seconds=np.asarray(
                    [record["wall_seconds"] for record in segment_records]
                ),
                segment_function_evaluations=np.asarray(
                    [
                        record["function_evaluations"]
                        for record in segment_records
                    ]
                ),
            )
            _write_progress(
                progress_path,
                status="running",
                target_time=final_time,
                completed_times=completed_times,
                segment_records=segment_records,
                spawn_records=spawn_records,
                spawn_attempt_records=spawn_attempt_records,
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)

    times = np.asarray(completed_times)
    trajectory = _padded_parameter_trajectory(parameter_states)
    norm = np.empty(times.size)
    minimum_electron_eigenvalue = np.empty(times.size)
    uncertainty_margin = np.empty(times.size)
    parameter_speed = np.empty(times.size)
    tangent_rank = np.empty(times.size, dtype=int)
    geometric_tangent_rank = np.empty(times.size, dtype=int)
    tangent_relative_residual = np.empty(times.size)
    tangent_absolute_residual = np.empty(times.size)
    coordinates = np.empty((times.size, hierarchy.coordinate_count))
    closed_coordinates = np.empty((times.size, 31))
    energy = np.empty(times.size)
    center_equilibrium = (
        -np.sqrt(2.0) * parameters.coupling / parameters.omega_ph
    )
    for index, current_time in enumerate(times):
        current_parameters = parameter_states[index]
        raw_state = multi_coherent_state(
            current_parameters,
            relative_dimension=relative_dimension,
        )
        norm[index] = float(np.vdot(raw_state, raw_state).real)
        normalized = raw_state / np.sqrt(norm[index])
        center = _center_amplitude(
            float(current_time),
            center_equilibrium,
            parameters,
        )
        coordinates[index] = relative_state_moment_coordinates(
            normalized,
            hierarchy,
            center_amplitude=center,
        )
        closed_coordinates[index] = relative_state_closed_coordinates(
            normalized,
            hierarchy,
            center_amplitude=center,
        )
        observables = multi_coherent_observables(
            float(current_time),
            current_parameters,
            parameters,
            relative_dimension=relative_dimension,
            drive_protocol=drive_protocol,
        )
        minimum_electron_eigenvalue[index] = float(
            np.min(np.linalg.eigvalsh(observables.electron_density))
        )
        energy[index] = observables.energy
        uncertainty_margin[index] = _uncertainty_margin(
            coordinates[index],
            hierarchy,
        )
        projection = project_schrodinger_velocity(
            current_parameters,
            relative_holstein_hamiltonian(
                float(current_time),
                parameters,
                relative_dimension=relative_dimension,
                drive_protocol=drive_protocol,
            ),
            relative_dimension=relative_dimension,
            relative_singular_value_cutoff=tangent_singular_value_cutoff,
            regularization=tangent_regularization,
            relative_damping=relative_damping,
        )
        parameter_speed[index] = projection.parameter_velocity_norm
        tangent_rank[index] = projection.tangent_rank
        geometric_tangent_rank[index] = projection.geometric_tangent_rank
        tangent_relative_residual[index] = projection.relative_residual
        tangent_absolute_residual[index] = projection.absolute_residual

    active_drive = (
        GaussianSineDrive.from_parameters(parameters)
        if drive_protocol is None
        else drive_protocol
    )
    drive_values = np.asarray(
        [active_drive.difference(float(value)) for value in times]
    )
    external_power = np.asarray(
        [active_drive.derivative(float(value)) for value in times]
    ) * closed_coordinates[:, 0]
    work_residual = energy_work_residual(times, energy, external_power)
    zero_drive = GaussianSineDrive(
        amplitude=0.0,
        pulse_width=parameters.pulse_width,
    )
    hamiltonian_zero = relative_holstein_hamiltonian(
        0.0,
        parameters,
        relative_dimension=relative_dimension,
        drive_protocol=zero_drive,
    )
    drive_operator_norm = 1.0
    work_scale = float(
        np.linalg.norm(hamiltonian_zero, ord=2)
        + np.max(np.abs(drive_values)) * drive_operator_norm
    )
    normalized_work_residual = np.abs(work_residual) / max(
        work_scale,
        np.finfo(float).tiny,
    )

    state_fidelity: np.ndarray | None = None
    coordinate_relative_error: np.ndarray | None = None
    exact_closed_coordinates: np.ndarray | None = None
    closed_coordinate_relative_error: np.ndarray | None = None
    exact_function_evaluations: dict[str, int] | None = None
    if compare_exact and times.size >= 2:
        exact_wavefunctions = (
            exact_holstein_wavefunction_trajectory_for_diagnostics(
                parameters,
                sample_times=times,
                phonon_cutoff=phonon_cutoff,
                eigensolver_tolerance=1e-12,
                relative_tolerance=1e-10,
                absolute_tolerance=1e-12,
                maximum_step=maximum_step,
            )
        )
        exact_moments = exact_holstein_moment_hierarchy_trajectory(
            parameters,
            hierarchy=hierarchy,
            sample_times=times,
            phonon_cutoff=phonon_cutoff,
            eigensolver_tolerance=1e-12,
            relative_tolerance=1e-10,
            absolute_tolerance=1e-12,
            maximum_step=maximum_step,
        )
        state_fidelity = np.empty(times.size)
        for index in range(times.size):
            exact_state = electron_relative_state(
                exact_wavefunctions.state_vectors[:, index],
                phonon_cutoff=phonon_cutoff,
            ).state
            autonomous = multi_coherent_state(
                parameter_states[index],
                relative_dimension=relative_dimension,
            )
            autonomous /= np.linalg.norm(autonomous)
            state_fidelity[index] = float(
                abs(np.vdot(exact_state, autonomous)) ** 2
            )
        coordinate_error = coordinates - exact_moments.coordinates
        coordinate_relative_error = np.linalg.norm(
            coordinate_error,
            axis=1,
        ) / np.maximum(
            np.linalg.norm(exact_moments.coordinates, axis=1),
            np.finfo(float).tiny,
        )
        exact_closed_coordinates = np.asarray(
            [
                matrix_state_to_closed_scalar_coordinates(
                    hierarchy.to_matrix_state(value)
                )
                for value in exact_moments.coordinates
            ]
        )
        closed_coordinate_relative_error = np.linalg.norm(
            closed_coordinates - exact_closed_coordinates,
            axis=1,
        ) / np.maximum(
            np.linalg.norm(exact_closed_coordinates, axis=1),
            np.finfo(float).tiny,
        )
        exact_function_evaluations = {
            "wavefunction": exact_wavefunctions.function_evaluations,
            "moments": exact_moments.function_evaluations,
        }

    status = "complete" if failure is None else "stopped"
    _write_progress(
        progress_path,
        status=status,
        target_time=final_time,
        completed_times=completed_times,
        segment_records=segment_records,
        spawn_records=spawn_records,
        spawn_attempt_records=spawn_attempt_records,
        failure=failure,
    )
    maximum_resident_set = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    maximum_resident_set_bytes = int(
        maximum_resident_set
        if sys.platform == "darwin"
        else maximum_resident_set * 1024
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "classification": "exploratory_local_not_promoted",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "resource_usage": {
            "wall_seconds": float(time.monotonic() - run_wall_start),
            "maximum_resident_set_bytes": maximum_resident_set_bytes,
        },
        "parameters": {
            **asdict(parameters),
            "coupling": parameters.coupling,
            "phonon_cutoff": phonon_cutoff,
            "packet_count": packet_count,
            "adaptive_capacity": adaptive_capacity,
            "maximum_packet_count": maximum_packet_count,
            "spawn_relative_residual_threshold": (
                spawn_relative_residual_threshold
            ),
            "spawn_absolute_residual_threshold": (
                spawn_absolute_residual_threshold
            ),
            "spawn_fit_maximum_iterations": spawn_fit_maximum_iterations,
            "spawn_fit_population_size": spawn_fit_population_size,
            "spawn_seed": spawn_seed,
            "target_final_time": final_time,
            "segment_length": segment_length,
            "output_sample_step": sample_step,
            "segment_timeout_seconds": segment_timeout_seconds,
            "maximum_step": maximum_step,
            "relative_tolerance": relative_tolerance,
            "absolute_tolerance": absolute_tolerance,
            "tangent_singular_value_cutoff": tangent_singular_value_cutoff,
            "tangent_regularization": tangent_regularization,
            "relative_damping": relative_damping,
            "drive_protocol": (
                None
                if drive_protocol is None
                else {
                    "amplitude": drive_protocol.amplitude,
                    "pulse_width": drive_protocol.pulse_width,
                    "delays": list(drive_protocol.delays),
                }
            ),
        },
        "initialization": {
            "source": initialization_source,
            "parameter_sha256": initial_parameter_sha256,
            "exact_reference_used_after_t0_by_model_rhs": False,
        },
        "progress": {
            "last_completed_time": float(times[-1]),
            "completed_segment_count": len(segment_records),
            "total_function_evaluations": int(
                sum(record["function_evaluations"] for record in segment_records)
            ),
            "maximum_segment_wall_seconds": float(
                max(
                    (record["wall_seconds"] for record in segment_records),
                    default=0.0,
                )
            ),
            "maximum_segment_function_evaluations": int(
                max(
                    (
                        record["function_evaluations"]
                        for record in segment_records
                    ),
                    default=0,
                )
            ),
            "failure": failure,
        },
        "physicality": {
            "maximum_norm_drift": float(np.max(np.abs(norm - 1.0))),
            "minimum_electron_density_eigenvalue": float(
                np.min(minimum_electron_eigenvalue)
            ),
            "minimum_relative_uncertainty_margin": float(
                np.min(uncertainty_margin)
            ),
        },
        "work_balance": {
            "maximum_absolute_residual": float(
                np.max(np.abs(work_residual))
            ),
            "normalization_scale": work_scale,
            "maximum_normalized_residual": float(
                np.max(normalized_work_residual)
            ),
        },
        "tangent_diagnostics": {
            "ranks": sorted({int(value) for value in tangent_rank}),
            "geometric_gram_relative_threshold": 1e-10,
            "geometric_ranks": sorted(
                {int(value) for value in geometric_tangent_rank}
            ),
            "maximum_parameter_speed": float(np.max(parameter_speed)),
            "maximum_relative_projection_residual": float(
                np.max(tangent_relative_residual)
            ),
            "maximum_absolute_projection_residual": float(
                np.max(tangent_absolute_residual)
            ),
        },
        "capacity": {
            "mode": (
                "adaptive_residual_spawn" if adaptive_capacity else "fixed"
            ),
            "online_exact_reference_used": False,
            "initial_packet_count": packet_count,
            "final_packet_count": int(packet_counts[-1]),
            "maximum_packet_count": maximum_packet_count,
            "initial_packets_per_electronic_branch": packet_count,
            "final_packets_per_electronic_branch": int(packet_counts[-1]),
            "maximum_packets_per_electronic_branch": maximum_packet_count,
            "initial_total_branch_packets": 4 * packet_count,
            "final_total_branch_packets": (
                multi_coherent_capacity(parameter_states[-1]).total_branch_packets
            ),
            "maximum_total_branch_packets": 4 * maximum_packet_count,
            "initial_raw_coordinate_count": int(initial_parameters.size),
            "final_raw_coordinate_count": int(parameter_states[-1].size),
            "spawn_count": len(spawn_records),
            "spawn_attempt_count": len(spawn_attempt_records),
            "rejected_spawn_count": (
                len(spawn_attempt_records) - len(spawn_records)
            ),
            "packet_count_trajectory": packet_counts,
            "spawns": spawn_records,
            "spawn_attempts": spawn_attempt_records,
        },
        "offline_exact_comparison": (
            None
            if state_fidelity is None or coordinate_relative_error is None
            else {
                "minimum_state_fidelity": float(np.min(state_fidelity)),
                "final_state_fidelity": float(state_fidelity[-1]),
                "maximum_coordinate_relative_error": float(
                    np.max(coordinate_relative_error)
                ),
                "final_coordinate_relative_error": float(
                    coordinate_relative_error[-1]
                ),
                "maximum_closed_coordinate_relative_error": float(
                    np.max(closed_coordinate_relative_error)
                ),
                "final_closed_coordinate_relative_error": float(
                    closed_coordinate_relative_error[-1]
                ),
                "function_evaluations": exact_function_evaluations,
            }
        ),
        "conclusion": (
            "target horizon completed"
            if failure is None
            else (
                "parameter-coordinate stiffness prevented the next segment; "
                "the last completed checkpoint is retained"
            )
        ),
    }

    final_path = run_directory / "segmented_horizon.npz"
    arrays: dict[str, np.ndarray] = {
        "times": times,
        "parameter_trajectory": trajectory,
        "packet_count_trajectory": np.asarray(packet_counts, dtype=int),
        "coordinates": coordinates,
        "closed_coordinates": closed_coordinates,
        "norm": norm,
        "energy": energy,
        "drive_difference": drive_values,
        "external_power": external_power,
        "energy_work_residual": work_residual,
        "normalized_energy_work_residual": normalized_work_residual,
        "minimum_electron_eigenvalue": minimum_electron_eigenvalue,
        "relative_uncertainty_margin": uncertainty_margin,
        "parameter_speed": parameter_speed,
        "tangent_rank": tangent_rank,
        "geometric_tangent_rank": geometric_tangent_rank,
        "tangent_relative_residual": tangent_relative_residual,
        "tangent_absolute_residual": tangent_absolute_residual,
        "segment_wall_seconds": np.asarray(
            [record["wall_seconds"] for record in segment_records]
        ),
        "segment_function_evaluations": np.asarray(
            [record["function_evaluations"] for record in segment_records]
        ),
    }
    if state_fidelity is not None and coordinate_relative_error is not None:
        arrays["state_fidelity"] = state_fidelity
        arrays["coordinate_relative_error"] = coordinate_relative_error
    if (
        exact_closed_coordinates is not None
        and closed_coordinate_relative_error is not None
    ):
        arrays["exact_closed_coordinates"] = exact_closed_coordinates
        arrays["closed_coordinate_relative_error"] = (
            closed_coordinate_relative_error
        )
    np.savez_compressed(final_path, **arrays)
    plot_path = run_directory / "segmented_horizon.png"
    _write_plot(
        plot_path,
        times,
        np.asarray(
            [record["segment_end"] for record in segment_records],
            dtype=float,
        ),
        arrays["segment_wall_seconds"],
        arrays["segment_function_evaluations"],
        norm,
        minimum_electron_eigenvalue,
        uncertainty_margin,
        state_fidelity,
        coordinate_relative_error,
    )
    summary_path = run_directory / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("multi_coherent.py"),
        Path(__file__).with_name("multi_coherent_propagation.py"),
        Path(__file__).with_name("conditional_packets.py"),
        Path(__file__).with_name("exact_reference.py"),
    )
    manifest = {
        "schema_version": 1,
        "status": status,
        "source_hashes": {
            str(path.resolve()): _sha256(path) for path in source_paths
        },
        "input_hashes": gate_hashes,
        "artifact_hashes": {
            path.name: _sha256(path)
            for path in (
                progress_path,
                checkpoint_path,
                final_path,
                plot_path,
                summary_path,
            )
            if path.is_file()
        },
    }
    (run_directory / "runtime_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--gate-directory", type=Path, required=True)
    parser.add_argument("--lambda-ep", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--drive", type=float, default=1.0)
    parser.add_argument("--phonon-cutoff", type=int, default=20)
    parser.add_argument("--final-time", type=float, default=20.0)
    parser.add_argument("--segment-length", type=float, default=0.5)
    parser.add_argument("--segment-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--maximum-step", type=float, default=0.01)
    parser.add_argument("--packet-count", type=int, default=4)
    parser.add_argument(
        "--tangent-regularization",
        choices=("truncated_svd", "tikhonov"),
        default="tikhonov",
    )
    parser.add_argument("--relative-damping", type=float, default=3e-3)
    parser.add_argument("--adaptive-capacity", action="store_true")
    parser.add_argument("--maximum-packet-count", type=int, default=6)
    parser.add_argument(
        "--spawn-relative-residual-threshold",
        type=float,
        default=5e-2,
    )
    parser.add_argument(
        "--spawn-absolute-residual-threshold",
        type=float,
        default=2e-2,
    )
    parser.add_argument("--spawn-fit-maximum-iterations", type=int, default=40)
    parser.add_argument("--spawn-fit-population-size", type=int, default=6)
    parser.add_argument("--spawn-seed", type=int, default=260803)
    parser.add_argument("--skip-exact-comparison", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = run_segmented_multi_coherent_horizon(
        args.run_directory,
        gate_directory=args.gate_directory,
        parameters=DimerParameters(
            lambda_ep=args.lambda_ep,
            gamma=args.gamma,
            drive_amplitude=args.drive,
        ),
        final_time=args.final_time,
        segment_length=args.segment_length,
        segment_timeout_seconds=args.segment_timeout_seconds,
        maximum_step=args.maximum_step,
        phonon_cutoff=args.phonon_cutoff,
        packet_count=args.packet_count,
        compare_exact=not args.skip_exact_comparison,
        tangent_regularization=args.tangent_regularization,
        relative_damping=args.relative_damping,
        adaptive_capacity=args.adaptive_capacity,
        maximum_packet_count=args.maximum_packet_count,
        spawn_relative_residual_threshold=(
            args.spawn_relative_residual_threshold
        ),
        spawn_absolute_residual_threshold=(
            args.spawn_absolute_residual_threshold
        ),
        spawn_fit_maximum_iterations=args.spawn_fit_maximum_iterations,
        spawn_fit_population_size=args.spawn_fit_population_size,
        spawn_seed=args.spawn_seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
