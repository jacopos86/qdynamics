"""Offline compression, tangent, moment, and cutoff gate for packet dynamics."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .conditional_packets import electron_relative_state
from .exact_reference import (
    exact_holstein_moment_hierarchy_trajectory,
    exact_holstein_wavefunction_trajectory_for_diagnostics,
)
from .hubbard_dimer import DimerParameters
from .moment_hierarchy import moment_hierarchy
from .multi_coherent import (
    fit_coherent_electron_relative_state,
    multi_coherent_state,
    project_schrodinger_velocity,
    relative_holstein_hamiltonian,
    relative_state_moment_coordinates,
    relative_state_moment_derivative,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_times(values: tuple[float, ...], *, name: str) -> np.ndarray:
    times = np.asarray(values, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError(f"{name} must contain at least two times")
    if abs(float(times[0])) > 1e-15:
        raise ValueError(f"{name} must start at zero")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return times


def _relative_rms(defect: np.ndarray, exact: np.ndarray) -> float:
    return float(
        np.sqrt(np.mean(np.sum(defect**2, axis=-1)))
        / max(
            float(np.sqrt(np.mean(np.sum(exact**2, axis=-1)))),
            np.finfo(float).tiny,
        )
    )


def _fit_and_gate(
    exact_states: tuple[np.ndarray, ...],
    exact_coordinates: np.ndarray,
    exact_derivatives: np.ndarray,
    times: np.ndarray,
    parameters: DimerParameters,
    *,
    packet_count: int,
    relative_dimension: int,
    hierarchy_degree: int,
    fit_maximum_iterations: int,
    fit_population_size: int,
    seed: int,
    tangent_singular_value_cutoff: float,
    tangent_regularization: str,
    relative_damping: float,
) -> dict[str, np.ndarray]:
    hierarchy = moment_hierarchy(hierarchy_degree)
    time_count = times.size
    state_fidelity = np.empty(time_count)
    exact_tangent_relative = np.empty(time_count)
    exact_tangent_absolute = np.empty(time_count)
    autonomous_tangent_relative = np.empty(time_count)
    autonomous_tangent_absolute = np.empty(time_count)
    autonomous_projected_speed = np.empty(time_count)
    parameter_speed = np.empty(time_count)
    tangent_rank = np.empty(time_count, dtype=int)
    smallest_retained_singular_value = np.empty(time_count)
    fitted_coordinates = np.empty_like(exact_coordinates)
    fitted_derivatives = np.empty_like(exact_derivatives)
    oracle_projected_derivatives = np.empty_like(exact_derivatives)
    fitted_parameters = np.empty((time_count, 16 * packet_count))
    function_evaluations = np.empty(time_count, dtype=int)

    for time_index, (time, exact_state) in enumerate(
        zip(times, exact_states, strict=True)
    ):
        fit = fit_coherent_electron_relative_state(
            exact_state,
            packet_count=packet_count,
            maximum_iterations=fit_maximum_iterations,
            population_size=fit_population_size,
            seed=seed + 101 * time_index,
        )
        fitted_parameters[time_index] = fit.parameters
        function_evaluations[time_index] = fit.function_evaluations
        state_fidelity[time_index] = fit.fidelity
        hamiltonian = relative_holstein_hamiltonian(
            float(time),
            parameters,
            relative_dimension=relative_dimension,
        )
        exact_projection = project_schrodinger_velocity(
            fit.parameters,
            hamiltonian,
            relative_dimension=relative_dimension,
            target_state=exact_state,
            relative_singular_value_cutoff=(
                tangent_singular_value_cutoff
            ),
            regularization=tangent_regularization,
            relative_damping=relative_damping,
        )
        autonomous_projection = project_schrodinger_velocity(
            fit.parameters,
            hamiltonian,
            relative_dimension=relative_dimension,
            relative_singular_value_cutoff=(
                tangent_singular_value_cutoff
            ),
            regularization=tangent_regularization,
            relative_damping=relative_damping,
        )
        exact_tangent_relative[time_index] = (
            exact_projection.relative_residual
        )
        exact_tangent_absolute[time_index] = (
            exact_projection.absolute_residual
        )
        autonomous_tangent_relative[time_index] = (
            autonomous_projection.relative_residual
        )
        autonomous_tangent_absolute[time_index] = (
            autonomous_projection.absolute_residual
        )
        autonomous_projected_speed[time_index] = float(
            np.linalg.norm(autonomous_projection.projected_velocity)
        )
        parameter_speed[time_index] = (
            autonomous_projection.parameter_velocity_norm
        )
        tangent_rank[time_index] = autonomous_projection.tangent_rank
        smallest_retained_singular_value[time_index] = (
            autonomous_projection.smallest_retained_singular_value
        )
        fitted_state = multi_coherent_state(
            fit.parameters,
            relative_dimension=relative_dimension,
        )
        center = complex(
            exact_coordinates[time_index, 0],
            exact_coordinates[time_index, 1],
        )
        fitted_coordinates[time_index] = relative_state_moment_coordinates(
            fitted_state,
            hierarchy,
            center_amplitude=center,
        )
        fitted_derivatives[time_index] = relative_state_moment_derivative(
            fitted_state,
            autonomous_projection.projected_velocity,
            hierarchy,
        )
        oracle_projected_derivatives[time_index] = (
            relative_state_moment_derivative(
                fitted_state,
                exact_projection.projected_velocity,
                hierarchy,
            )
        )
    return {
        "state_fidelity": state_fidelity,
        "exact_tangent_relative_residual": exact_tangent_relative,
        "exact_tangent_absolute_residual": exact_tangent_absolute,
        "autonomous_tangent_relative_residual": autonomous_tangent_relative,
        "autonomous_tangent_absolute_residual": autonomous_tangent_absolute,
        "autonomous_projected_state_speed": autonomous_projected_speed,
        "parameter_speed": parameter_speed,
        "tangent_rank": tangent_rank,
        "smallest_retained_singular_value": (
            smallest_retained_singular_value
        ),
        "fitted_coordinates": fitted_coordinates,
        "fitted_derivatives": fitted_derivatives,
        "oracle_projected_derivatives": oracle_projected_derivatives,
        "fitted_parameters": fitted_parameters,
        "function_evaluations": function_evaluations,
    }


def _summary_for_packet_count(
    result: dict[str, np.ndarray],
    exact_coordinates: np.ndarray,
    exact_derivatives: np.ndarray,
    times: np.ndarray,
    *,
    stationary_velocity_norm_threshold: float,
) -> dict[str, Any]:
    exact_velocity_norm = np.linalg.norm(exact_derivatives, axis=1)
    nonstationary = exact_velocity_norm > stationary_velocity_norm_threshold
    coordinate_defect = result["fitted_coordinates"] - exact_coordinates
    derivative_defect = result["fitted_derivatives"] - exact_derivatives
    oracle_derivative_defect = (
        result["oracle_projected_derivatives"] - exact_derivatives
    )
    return {
        "minimum_state_fidelity": float(np.min(result["state_fidelity"])),
        "maximum_exact_tangent_relative_residual_nonstationary": float(
            np.max(result["exact_tangent_relative_residual"][nonstationary])
        ),
        "maximum_autonomous_tangent_relative_residual_nonstationary": float(
            np.max(
                result["autonomous_tangent_relative_residual"][nonstationary]
            )
        ),
        "maximum_exact_tangent_absolute_residual": float(
            np.max(result["exact_tangent_absolute_residual"])
        ),
        "initial_autonomous_projected_state_speed": float(
            result["autonomous_projected_state_speed"][0]
        ),
        "maximum_parameter_speed": float(np.max(result["parameter_speed"])),
        "tangent_ranks": sorted(
            {int(value) for value in result["tangent_rank"]}
        ),
        "minimum_retained_singular_value": float(
            np.min(result["smallest_retained_singular_value"])
        ),
        "coordinate_relative_vector_rms": _relative_rms(
            coordinate_defect,
            exact_coordinates,
        ),
        "autonomous_derivative_relative_vector_rms": _relative_rms(
            derivative_defect,
            exact_derivatives,
        ),
        "oracle_tangent_derivative_relative_vector_rms": _relative_rms(
            oracle_derivative_defect,
            exact_derivatives,
        ),
        "time_resolved": [
            {
                "time": float(time),
                "state_fidelity": float(result["state_fidelity"][index]),
                "exact_tangent_relative_residual": float(
                    result["exact_tangent_relative_residual"][index]
                ),
                "exact_tangent_absolute_residual": float(
                    result["exact_tangent_absolute_residual"][index]
                ),
                "autonomous_tangent_relative_residual": float(
                    result["autonomous_tangent_relative_residual"][index]
                ),
                "autonomous_projected_state_speed": float(
                    result["autonomous_projected_state_speed"][index]
                ),
                "parameter_speed": float(result["parameter_speed"][index]),
                "tangent_rank": int(result["tangent_rank"][index]),
            }
            for index, time in enumerate(times)
        ],
    }


def _write_plot(
    path: Path,
    times: np.ndarray,
    results: dict[int, dict[str, np.ndarray]],
    exact_derivatives: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.25))
    colors = {2: "#D28E2B", 3: "#6E3CBC", 4: "#2378B5"}
    exact_norm = np.linalg.norm(exact_derivatives, axis=1)
    for packet_count, result in results.items():
        color = colors.get(packet_count, None)
        axes[0].semilogy(
            times,
            np.maximum(1.0 - result["state_fidelity"], 1e-12),
            label=f"K={packet_count}",
            color=color,
        )
        axes[1].plot(
            times[1:],
            result["exact_tangent_relative_residual"][1:],
            label=f"K={packet_count}",
            color=color,
        )
        derivative_defect = np.linalg.norm(
            result["fitted_derivatives"] - exact_derivatives,
            axis=1,
        )
        axes[2].plot(
            times[1:],
            derivative_defect[1:]
            / np.maximum(exact_norm[1:], np.finfo(float).tiny),
            label=f"K={packet_count}",
            color=color,
        )
    axes[0].set_ylabel("state infidelity")
    axes[1].set_ylabel("exact-velocity tangent defect")
    axes[2].set_ylabel("82-coordinate velocity defect")
    for axis in axes:
        axis.set_xlabel(r"dimensionless time $t\,t_{\rm hop}$")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=7.5)
    axes[1].axhline(0.1, color="#555555", linestyle="--", linewidth=0.8)
    axes[2].axhline(0.1, color="#555555", linestyle="--", linewidth=0.8)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run_multi_coherent_analysis(
    run_directory: Path,
    *,
    parameters: DimerParameters,
    sample_times: tuple[float, ...] = (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
    ),
    packet_counts: tuple[int, ...] = (2, 3, 4),
    selected_packet_count: int = 4,
    phonon_cutoff: int = 20,
    convergence_cutoffs: tuple[int, ...] = (12, 16, 20),
    convergence_times: tuple[float, ...] = (0.0, 2.0, 4.0),
    hierarchy_degree: int = 4,
    eigensolver_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-12,
    maximum_step: float = 0.01,
    fit_maximum_iterations: int = 120,
    fit_population_size: int = 8,
    tangent_singular_value_cutoff: float = 1e-2,
    tangent_regularization: str = "tikhonov",
    relative_damping: float = 3e-3,
    state_infidelity_threshold: float = 1e-3,
    tangent_relative_threshold: float = 0.1,
    moment_derivative_relative_threshold: float = 0.1,
    maximum_parameter_speed_threshold: float = 10.0,
    decoupled_derivative_relative_threshold: float = 1e-4,
    stationary_velocity_norm_threshold: float = 1e-5,
) -> dict[str, Any]:
    """Gate coherent packet counts before any autonomous propagation."""

    times = _validated_times(sample_times, name="sample_times")
    convergence_time_array = _validated_times(
        convergence_times,
        name="convergence_times",
    )
    if selected_packet_count not in packet_counts:
        raise ValueError("selected_packet_count must be in packet_counts")
    if phonon_cutoff not in convergence_cutoffs:
        raise ValueError("convergence_cutoffs must include phonon_cutoff")
    if not all(np.any(np.isclose(times, time)) for time in convergence_time_array):
        raise ValueError("convergence_times must be present in sample_times")
    run_directory.mkdir(parents=True, exist_ok=True)

    hierarchy = moment_hierarchy(hierarchy_degree)
    wavefunctions = exact_holstein_wavefunction_trajectory_for_diagnostics(
        parameters,
        sample_times=times,
        phonon_cutoff=phonon_cutoff,
        eigensolver_tolerance=eigensolver_tolerance,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step=maximum_step,
    )
    exact_moments = exact_holstein_moment_hierarchy_trajectory(
        parameters,
        hierarchy=hierarchy,
        sample_times=times,
        phonon_cutoff=phonon_cutoff,
        eigensolver_tolerance=eigensolver_tolerance,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step=maximum_step,
    )
    extracted = tuple(
        electron_relative_state(
            wavefunctions.state_vectors[:, index],
            phonon_cutoff=phonon_cutoff,
        )
        for index in range(times.size)
    )
    exact_states = tuple(value.state for value in extracted)
    results: dict[int, dict[str, np.ndarray]] = {}
    packet_summaries: dict[str, Any] = {}
    for packet_count in packet_counts:
        result = _fit_and_gate(
            exact_states,
            exact_moments.coordinates,
            exact_moments.coordinate_derivatives,
            times,
            parameters,
            packet_count=packet_count,
            relative_dimension=2 * phonon_cutoff + 1,
            hierarchy_degree=hierarchy_degree,
            fit_maximum_iterations=fit_maximum_iterations,
            fit_population_size=fit_population_size,
            seed=10000 * packet_count,
            tangent_singular_value_cutoff=tangent_singular_value_cutoff,
            tangent_regularization=tangent_regularization,
            relative_damping=relative_damping,
        )
        results[packet_count] = result
        packet_summaries[str(packet_count)] = _summary_for_packet_count(
            result,
            exact_moments.coordinates,
            exact_moments.coordinate_derivatives,
            times,
            stationary_velocity_norm_threshold=(
                stationary_velocity_norm_threshold
            ),
        )

    cutoff_summary: dict[str, Any] = {}
    for cutoff_index, cutoff in enumerate(convergence_cutoffs):
        if cutoff == phonon_cutoff:
            sample_indices = tuple(
                int(np.flatnonzero(np.isclose(times, time))[0])
                for time in convergence_time_array
            )
            cutoff_states = tuple(exact_states[index] for index in sample_indices)
            cutoff_coordinates = exact_moments.coordinates[list(sample_indices)]
            cutoff_derivatives = exact_moments.coordinate_derivatives[
                list(sample_indices)
            ]
        else:
            cutoff_wavefunctions = (
                exact_holstein_wavefunction_trajectory_for_diagnostics(
                    parameters,
                    sample_times=convergence_time_array,
                    phonon_cutoff=cutoff,
                    eigensolver_tolerance=eigensolver_tolerance,
                    relative_tolerance=relative_tolerance,
                    absolute_tolerance=absolute_tolerance,
                    maximum_step=maximum_step,
                )
            )
            cutoff_exact_moments = exact_holstein_moment_hierarchy_trajectory(
                parameters,
                hierarchy=hierarchy,
                sample_times=convergence_time_array,
                phonon_cutoff=cutoff,
                eigensolver_tolerance=eigensolver_tolerance,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                maximum_step=maximum_step,
            )
            cutoff_states = tuple(
                electron_relative_state(
                    cutoff_wavefunctions.state_vectors[:, index],
                    phonon_cutoff=cutoff,
                ).state
                for index in range(convergence_time_array.size)
            )
            cutoff_coordinates = cutoff_exact_moments.coordinates
            cutoff_derivatives = cutoff_exact_moments.coordinate_derivatives
        cutoff_result = _fit_and_gate(
            cutoff_states,
            cutoff_coordinates,
            cutoff_derivatives,
            convergence_time_array,
            parameters,
            packet_count=selected_packet_count,
            relative_dimension=2 * cutoff + 1,
            hierarchy_degree=hierarchy_degree,
            fit_maximum_iterations=fit_maximum_iterations,
            fit_population_size=fit_population_size,
            seed=700000 + 1000 * cutoff_index,
            tangent_singular_value_cutoff=tangent_singular_value_cutoff,
            tangent_regularization=tangent_regularization,
            relative_damping=relative_damping,
        )
        cutoff_summary[str(cutoff)] = _summary_for_packet_count(
            cutoff_result,
            cutoff_coordinates,
            cutoff_derivatives,
            convergence_time_array,
            stationary_velocity_norm_threshold=(
                stationary_velocity_norm_threshold
            ),
        )

    control_parameters = DimerParameters(
        hopping=parameters.hopping,
        gamma=parameters.gamma,
        lambda_ep=0.0,
        drive_amplitude=parameters.drive_amplitude,
        pulse_width=parameters.pulse_width,
    )
    control_times = np.asarray([0.0, 1.0, 2.0])
    control_wavefunctions = exact_holstein_wavefunction_trajectory_for_diagnostics(
        control_parameters,
        sample_times=control_times,
        phonon_cutoff=3,
        eigensolver_tolerance=eigensolver_tolerance,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step=maximum_step,
    )
    control_moments = exact_holstein_moment_hierarchy_trajectory(
        control_parameters,
        hierarchy=hierarchy,
        sample_times=control_times,
        phonon_cutoff=3,
        eigensolver_tolerance=eigensolver_tolerance,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step=maximum_step,
    )
    control_states = tuple(
        electron_relative_state(
            control_wavefunctions.state_vectors[:, index],
            phonon_cutoff=3,
        ).state
        for index in range(control_times.size)
    )
    control_result = _fit_and_gate(
        control_states,
        control_moments.coordinates,
        control_moments.coordinate_derivatives,
        control_times,
        control_parameters,
        packet_count=1,
        relative_dimension=7,
        hierarchy_degree=hierarchy_degree,
        fit_maximum_iterations=fit_maximum_iterations,
        fit_population_size=fit_population_size,
        seed=990000,
        tangent_singular_value_cutoff=tangent_singular_value_cutoff,
        tangent_regularization=tangent_regularization,
        relative_damping=relative_damping,
    )
    control_summary = _summary_for_packet_count(
        control_result,
        control_moments.coordinates,
        control_moments.coordinate_derivatives,
        control_times,
        stationary_velocity_norm_threshold=stationary_velocity_norm_threshold,
    )

    selected = packet_summaries[str(selected_packet_count)]
    compression_passed = (
        1.0 - selected["minimum_state_fidelity"]
        <= state_infidelity_threshold
    )
    tangent_passed = (
        selected["maximum_exact_tangent_relative_residual_nonstationary"]
        <= tangent_relative_threshold
    )
    moment_passed = (
        selected["autonomous_derivative_relative_vector_rms"]
        <= moment_derivative_relative_threshold
    )
    speed_passed = (
        selected["maximum_parameter_speed"]
        <= maximum_parameter_speed_threshold
    )
    control_passed = (
        control_summary["autonomous_derivative_relative_vector_rms"]
        <= decoupled_derivative_relative_threshold
    )
    gate_passed = all(
        (compression_passed, tangent_passed, moment_passed, speed_passed, control_passed)
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "classification": "exploratory_local_not_promoted",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            **asdict(parameters),
            "coupling": parameters.coupling,
            "phonon_cutoff": phonon_cutoff,
            "sample_times": times.tolist(),
            "hierarchy_degree": hierarchy_degree,
            "hierarchy_coordinate_count": hierarchy.coordinate_count,
            "tangent_singular_value_cutoff": tangent_singular_value_cutoff,
            "tangent_regularization": tangent_regularization,
            "relative_damping": relative_damping,
        },
        "representation": {
            "name": "electron-conditioned multi-coherent relative-mode ansatz",
            "construction": (
                "K normalized coherent packets per electronic site "
                "configuration; center mode removed by exact normal-mode "
                "factorization; real McLachlan tangent projection"
            ),
            "autonomous_rhs_uses_exact_reference": False,
        },
        "minimum_center_factorization": float(
            min(value.center_factorization for value in extracted)
        ),
        "packet_count_comparison": packet_summaries,
        "cutoff_convergence": {
            "times": convergence_time_array.tolist(),
            "selected_packet_count": selected_packet_count,
            "cutoffs": cutoff_summary,
        },
        "decoupled_one_packet_control": control_summary,
        "validation_gate": {
            "selected_packet_count": selected_packet_count,
            "state_infidelity_threshold": state_infidelity_threshold,
            "tangent_relative_threshold": tangent_relative_threshold,
            "moment_derivative_relative_threshold": (
                moment_derivative_relative_threshold
            ),
            "maximum_parameter_speed_threshold": (
                maximum_parameter_speed_threshold
            ),
            "decoupled_derivative_relative_threshold": (
                decoupled_derivative_relative_threshold
            ),
            "compression_passed": bool(compression_passed),
            "tangent_velocity_passed": bool(tangent_passed),
            "moment_derivative_passed": bool(moment_passed),
            "regularized_parameter_speed_passed": bool(speed_passed),
            "decoupled_control_passed": bool(control_passed),
            "all_gates_passed": bool(gate_passed),
            "autonomous_short_propagation_authorized": bool(gate_passed),
            "decision": (
                "run a short autonomous propagation and compare moments"
                if gate_passed
                else "do not propagate; revise the representation or metric"
            ),
        },
    }

    prefix = "multi_coherent_velocity_gate"
    arrays: dict[str, np.ndarray] = {
        "times": times,
        "exact_coordinates": exact_moments.coordinates,
        "exact_derivatives": exact_moments.coordinate_derivatives,
    }
    for packet_count, result in results.items():
        for name, values in result.items():
            arrays[f"k{packet_count}_{name}"] = values
    trajectory_path = run_directory / f"{prefix}.npz"
    np.savez_compressed(trajectory_path, **arrays)
    plot_path = run_directory / f"{prefix}.png"
    _write_plot(
        plot_path,
        times,
        results,
        exact_moments.coordinate_derivatives,
    )
    summary_path = run_directory / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("multi_coherent.py"),
        Path(__file__).with_name("conditional_packets.py"),
        Path(__file__).with_name("exact_reference.py"),
        Path(__file__).with_name("moment_hierarchy.py"),
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "source_hashes": {
            str(path.resolve()): _sha256(path) for path in source_paths
        },
        "artifact_hashes": {
            path.name: _sha256(path)
            for path in (summary_path, trajectory_path, plot_path)
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
    parser.add_argument("--lambda-ep", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--drive", type=float, default=1.0)
    parser.add_argument("--phonon-cutoff", type=int, default=20)
    parser.add_argument("--maximum-step", type=float, default=0.01)
    parser.add_argument("--fit-maximum-iterations", type=int, default=120)
    parser.add_argument("--fit-population-size", type=int, default=8)
    parser.add_argument(
        "--tangent-regularization",
        choices=("truncated_svd", "tikhonov"),
        default="tikhonov",
    )
    parser.add_argument("--relative-damping", type=float, default=3e-3)
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = run_multi_coherent_analysis(
        args.run_directory,
        parameters=DimerParameters(
            lambda_ep=args.lambda_ep,
            gamma=args.gamma,
            drive_amplitude=args.drive,
        ),
        phonon_cutoff=args.phonon_cutoff,
        maximum_step=args.maximum_step,
        fit_maximum_iterations=args.fit_maximum_iterations,
        fit_population_size=args.fit_population_size,
        tangent_regularization=args.tangent_regularization,
        relative_damping=args.relative_damping,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
