"""Autonomous propagation and exact offline audit of the packet ansatz."""

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
from scipy.integrate import solve_ivp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .conditional_packets import electron_relative_state
from .exact_reference import (
    exact_holstein_moment_hierarchy_trajectory,
    exact_holstein_wavefunction_trajectory_for_diagnostics,
)
from .hubbard_dimer import DimerParameters
from .moment_hierarchy import MomentKey, moment_hierarchy
from .multi_coherent import (
    fit_coherent_electron_relative_state,
    multi_coherent_observables,
    multi_coherent_rhs,
    multi_coherent_state,
    project_schrodinger_velocity,
    relative_holstein_hamiltonian,
    relative_state_moment_coordinates,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_times(final_time: float, sample_step: float) -> np.ndarray:
    if final_time <= 0.0 or sample_step <= 0.0:
        raise ValueError("final_time and sample_step must be positive")
    intervals = int(round(final_time / sample_step))
    if not np.isclose(intervals * sample_step, final_time, atol=1e-12):
        raise ValueError("final_time must be an integer multiple of sample_step")
    return np.linspace(0.0, final_time, intervals + 1)


def _load_gate_initial_parameters(
    gate_directory: Path,
    *,
    packet_count: int,
) -> tuple[np.ndarray, dict[str, str]]:
    summary_path = gate_directory / "summary.json"
    trajectory_path = gate_directory / "multi_coherent_velocity_gate.npz"
    if not summary_path.is_file() or not trajectory_path.is_file():
        raise FileNotFoundError("gate summary and trajectory artifacts are required")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gate = summary["validation_gate"]
    if not gate["all_gates_passed"]:
        raise ValueError("the supplied velocity gate did not pass")
    if int(gate["selected_packet_count"]) != packet_count:
        raise ValueError("packet_count does not match the supplied gate")
    with np.load(trajectory_path) as arrays:
        initial = np.asarray(
            arrays[f"k{packet_count}_fitted_parameters"][0],
            dtype=float,
        )
        if abs(float(arrays["times"][0])) > 1e-15:
            raise ValueError("gate trajectory does not begin at zero")
    return initial, {
        str(summary_path.resolve()): _sha256(summary_path),
        str(trajectory_path.resolve()): _sha256(trajectory_path),
    }


def _center_amplitude(
    time: float,
    initial: complex,
    parameters: DimerParameters,
) -> complex:
    equilibrium = -np.sqrt(2.0) * parameters.coupling / parameters.omega_ph
    return equilibrium + (initial - equilibrium) * np.exp(
        -1j * parameters.omega_ph * time
    )


def _uncertainty_margin(coordinates: np.ndarray, hierarchy: Any) -> float:
    mean_x = hierarchy.moment_value(
        coordinates,
        MomentKey("I", "I", 1, 0),
    )
    mean_p = hierarchy.moment_value(
        coordinates,
        MomentKey("I", "I", 0, 1),
    )
    second_x = hierarchy.moment_value(
        coordinates,
        MomentKey("I", "I", 2, 0),
    )
    second_xp = hierarchy.moment_value(
        coordinates,
        MomentKey("I", "I", 1, 1),
    )
    second_p = hierarchy.moment_value(
        coordinates,
        MomentKey("I", "I", 0, 2),
    )
    covariance = np.array(
        [
            [second_x - mean_x**2, second_xp - mean_x * mean_p],
            [second_xp - mean_x * mean_p, second_p - mean_p**2],
        ]
    )
    return float(np.linalg.det(covariance) - 0.25)


def _write_plot(
    path: Path,
    times: np.ndarray,
    state_fidelity: np.ndarray,
    coordinate_relative_error: np.ndarray,
    norm_error: np.ndarray,
    minimum_electron_eigenvalue: np.ndarray,
    uncertainty_margin: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(10.6, 3.2))
    axes[0].plot(times, state_fidelity, color="#2378B5")
    axes[0].set_ylabel("exact relative-state fidelity")
    axes[0].set_ylim(min(0.98, float(np.min(state_fidelity)) - 0.002), 1.0005)
    axes[1].semilogy(
        times,
        np.maximum(coordinate_relative_error, 1e-12),
        color="#6E3CBC",
        label="82-coordinate error",
    )
    axes[1].semilogy(
        times,
        np.maximum(norm_error, 1e-12),
        color="#D28E2B",
        label="norm error",
    )
    axes[1].set_ylabel("relative / absolute error")
    axes[1].legend(frameon=False, fontsize=7.5)
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
    axes[2].set_ylabel("physicality margin")
    axes[2].legend(frameon=False, fontsize=7.5)
    for axis in axes:
        axis.set_xlabel(r"dimensionless time $t\,t_{\rm hop}$")
        axis.grid(alpha=0.22)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run_multi_coherent_propagation(
    run_directory: Path,
    *,
    parameters: DimerParameters,
    final_time: float = 4.0,
    sample_step: float = 0.05,
    maximum_step: float = 0.01,
    relative_tolerance: float = 1e-7,
    absolute_tolerance: float = 1e-9,
    exact_relative_tolerance: float = 1e-10,
    exact_absolute_tolerance: float = 1e-12,
    phonon_cutoff: int = 20,
    hierarchy_degree: int = 4,
    packet_count: int = 4,
    tangent_singular_value_cutoff: float = 1e-2,
    tangent_regularization: str = "tikhonov",
    relative_damping: float = 3e-3,
    gate_directory: Path | None = None,
    fit_maximum_iterations: int = 120,
    fit_population_size: int = 8,
    final_state_fidelity_threshold: float = 0.99,
    coordinate_relative_rms_threshold: float = 0.05,
    norm_drift_threshold: float = 1e-3,
) -> dict[str, Any]:
    """Propagate the packet parameters without exact-reference feedback."""

    times = _sample_times(final_time, sample_step)
    run_directory.mkdir(parents=True, exist_ok=True)
    exact_wavefunctions = exact_holstein_wavefunction_trajectory_for_diagnostics(
        parameters,
        sample_times=times,
        phonon_cutoff=phonon_cutoff,
        relative_tolerance=exact_relative_tolerance,
        absolute_tolerance=exact_absolute_tolerance,
        maximum_step=maximum_step,
        eigensolver_tolerance=1e-12,
    )
    hierarchy = moment_hierarchy(hierarchy_degree)
    exact_moments = exact_holstein_moment_hierarchy_trajectory(
        parameters,
        hierarchy=hierarchy,
        sample_times=times,
        phonon_cutoff=phonon_cutoff,
        relative_tolerance=exact_relative_tolerance,
        absolute_tolerance=exact_absolute_tolerance,
        maximum_step=maximum_step,
        eigensolver_tolerance=1e-12,
    )
    exact_relative_states = tuple(
        electron_relative_state(
            exact_wavefunctions.state_vectors[:, index],
            phonon_cutoff=phonon_cutoff,
        ).state
        for index in range(times.size)
    )
    gate_hashes: dict[str, str] = {}
    if gate_directory is not None:
        initial_parameters, gate_hashes = _load_gate_initial_parameters(
            gate_directory,
            packet_count=packet_count,
        )
        initialization = "passed_velocity_gate_artifact"
    else:
        initial_fit = fit_coherent_electron_relative_state(
            exact_relative_states[0],
            packet_count=packet_count,
            maximum_iterations=fit_maximum_iterations,
            population_size=fit_population_size,
            seed=10000 * packet_count,
        )
        initial_parameters = initial_fit.parameters
        initialization = "offline_exact_t0_packet_fit"

    relative_dimension = 2 * phonon_cutoff + 1

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        return multi_coherent_rhs(
            time,
            state,
            parameters,
            relative_dimension=relative_dimension,
            relative_singular_value_cutoff=tangent_singular_value_cutoff,
            regularization=tangent_regularization,
            relative_damping=relative_damping,
        )

    solution = solve_ivp(
        rhs,
        (0.0, final_time),
        initial_parameters,
        method="DOP853",
        t_eval=times,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        max_step=maximum_step,
    )
    if not solution.success or solution.y.shape[1] != times.size:
        raise RuntimeError(f"multi-coherent propagation failed: {solution.message}")

    coordinate_trajectory = np.empty_like(exact_moments.coordinates)
    state_fidelity = np.empty(times.size)
    state_norm = np.empty(times.size)
    minimum_electron_eigenvalue = np.empty(times.size)
    maximum_electron_eigenvalue = np.empty(times.size)
    uncertainty_margin = np.empty(times.size)
    relative_population = np.empty(times.size)
    energy = np.empty(times.size)
    tangent_rank = np.empty(times.size, dtype=int)
    parameter_speed = np.empty(times.size)
    tangent_relative_residual = np.empty(times.size)
    initial_center = complex(
        exact_moments.coordinates[0, 0],
        exact_moments.coordinates[0, 1],
    )
    for index, time in enumerate(times):
        raw_state = multi_coherent_state(
            solution.y[:, index],
            relative_dimension=relative_dimension,
        )
        state_norm[index] = float(np.vdot(raw_state, raw_state).real)
        normalized_state = raw_state / np.sqrt(state_norm[index])
        state_fidelity[index] = float(
            abs(np.vdot(exact_relative_states[index], normalized_state)) ** 2
        )
        center = _center_amplitude(float(time), initial_center, parameters)
        coordinate_trajectory[index] = relative_state_moment_coordinates(
            normalized_state,
            hierarchy,
            center_amplitude=center,
        )
        observables = multi_coherent_observables(
            float(time),
            solution.y[:, index],
            parameters,
            relative_dimension=relative_dimension,
        )
        electron_eigenvalues = np.linalg.eigvalsh(observables.electron_density)
        minimum_electron_eigenvalue[index] = float(electron_eigenvalues[0])
        maximum_electron_eigenvalue[index] = float(electron_eigenvalues[-1])
        uncertainty_margin[index] = _uncertainty_margin(
            coordinate_trajectory[index],
            hierarchy,
        )
        relative_population[index] = observables.relative_population
        energy[index] = observables.energy
        projection = project_schrodinger_velocity(
            solution.y[:, index],
            relative_holstein_hamiltonian(
                float(time),
                parameters,
                relative_dimension=relative_dimension,
            ),
            relative_dimension=relative_dimension,
            relative_singular_value_cutoff=tangent_singular_value_cutoff,
            regularization=tangent_regularization,
            relative_damping=relative_damping,
        )
        tangent_rank[index] = projection.tangent_rank
        parameter_speed[index] = projection.parameter_velocity_norm
        tangent_relative_residual[index] = projection.relative_residual

    coordinate_error = coordinate_trajectory - exact_moments.coordinates
    coordinate_relative_error = np.linalg.norm(coordinate_error, axis=1) / np.maximum(
        np.linalg.norm(exact_moments.coordinates, axis=1),
        np.finfo(float).tiny,
    )
    coordinate_relative_rms = float(
        np.sqrt(np.mean(np.sum(coordinate_error**2, axis=1)))
        / np.sqrt(np.mean(np.sum(exact_moments.coordinates**2, axis=1)))
    )
    maximum_norm_drift = float(np.max(np.abs(state_norm - 1.0)))
    passed = all(
        (
            state_fidelity[-1] >= final_state_fidelity_threshold,
            coordinate_relative_rms <= coordinate_relative_rms_threshold,
            maximum_norm_drift <= norm_drift_threshold,
            np.min(minimum_electron_eigenvalue) >= -1e-10,
            np.min(uncertainty_margin) >= -1e-8,
        )
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "classification": "exploratory_local_not_promoted",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            **asdict(parameters),
            "coupling": parameters.coupling,
            "phonon_cutoff": phonon_cutoff,
            "packet_count": packet_count,
            "hierarchy_coordinate_count": hierarchy.coordinate_count,
            "final_time": final_time,
            "sample_step": sample_step,
            "maximum_step": maximum_step,
            "relative_tolerance": relative_tolerance,
            "absolute_tolerance": absolute_tolerance,
            "tangent_singular_value_cutoff": tangent_singular_value_cutoff,
            "tangent_regularization": tangent_regularization,
            "relative_damping": relative_damping,
        },
        "initialization": {
            "kind": initialization,
            "gate_directory": (
                str(gate_directory.resolve()) if gate_directory else None
            ),
            "exact_reference_used_after_t0_by_autonomous_rhs": False,
        },
        "integration": {
            "solver": "DOP853",
            "success": bool(solution.success),
            "function_evaluations": int(solution.nfev),
            "message": str(solution.message),
        },
        "comparison": {
            "initial_state_fidelity": float(state_fidelity[0]),
            "final_state_fidelity": float(state_fidelity[-1]),
            "minimum_state_fidelity": float(np.min(state_fidelity)),
            "coordinate_relative_vector_rms": coordinate_relative_rms,
            "maximum_coordinate_relative_error": float(
                np.max(coordinate_relative_error)
            ),
            "final_coordinate_relative_error": float(
                coordinate_relative_error[-1]
            ),
        },
        "physicality": {
            "maximum_norm_drift": maximum_norm_drift,
            "minimum_electron_density_eigenvalue": float(
                np.min(minimum_electron_eigenvalue)
            ),
            "maximum_electron_density_eigenvalue": float(
                np.max(maximum_electron_eigenvalue)
            ),
            "minimum_relative_uncertainty_margin": float(
                np.min(uncertainty_margin)
            ),
            "maximum_relative_population": float(np.max(relative_population)),
        },
        "tangent_diagnostics": {
            "ranks": sorted({int(value) for value in tangent_rank}),
            "maximum_parameter_speed": float(np.max(parameter_speed)),
            "maximum_relative_projection_residual": float(
                np.max(tangent_relative_residual)
            ),
        },
        "validation_gate": {
            "final_state_fidelity_threshold": final_state_fidelity_threshold,
            "coordinate_relative_rms_threshold": (
                coordinate_relative_rms_threshold
            ),
            "norm_drift_threshold": norm_drift_threshold,
            "passed": bool(passed),
            "longer_propagation_authorized": bool(passed),
            "decision": (
                "extend the autonomous horizon"
                if passed
                else "stop; accumulated trajectory error failed the short gate"
            ),
        },
    }

    prefix = "multi_coherent_autonomous_trajectory"
    trajectory_path = run_directory / f"{prefix}.npz"
    np.savez_compressed(
        trajectory_path,
        times=times,
        parameter_trajectory=solution.y.T,
        state_fidelity=state_fidelity,
        state_norm=state_norm,
        exact_coordinates=exact_moments.coordinates,
        autonomous_coordinates=coordinate_trajectory,
        coordinate_error=coordinate_error,
        coordinate_relative_error=coordinate_relative_error,
        minimum_electron_eigenvalue=minimum_electron_eigenvalue,
        maximum_electron_eigenvalue=maximum_electron_eigenvalue,
        relative_uncertainty_margin=uncertainty_margin,
        relative_population=relative_population,
        energy=energy,
        tangent_rank=tangent_rank,
        parameter_speed=parameter_speed,
        tangent_relative_residual=tangent_relative_residual,
    )
    plot_path = run_directory / f"{prefix}.png"
    _write_plot(
        plot_path,
        times,
        state_fidelity,
        coordinate_relative_error,
        np.abs(state_norm - 1.0),
        minimum_electron_eigenvalue,
        uncertainty_margin,
    )
    summary_path = run_directory / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("multi_coherent.py"),
        Path(__file__).with_name("multi_coherent_analysis.py"),
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
        "input_hashes": gate_hashes,
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
    parser.add_argument("--gate-directory", type=Path)
    parser.add_argument("--lambda-ep", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--drive", type=float, default=1.0)
    parser.add_argument("--phonon-cutoff", type=int, default=20)
    parser.add_argument("--final-time", type=float, default=4.0)
    parser.add_argument("--sample-step", type=float, default=0.05)
    parser.add_argument("--maximum-step", type=float, default=0.01)
    parser.add_argument(
        "--tangent-regularization",
        choices=("truncated_svd", "tikhonov"),
        default="tikhonov",
    )
    parser.add_argument("--relative-damping", type=float, default=3e-3)
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = run_multi_coherent_propagation(
        args.run_directory,
        parameters=DimerParameters(
            lambda_ep=args.lambda_ep,
            gamma=args.gamma,
            drive_amplitude=args.drive,
        ),
        final_time=args.final_time,
        sample_step=args.sample_step,
        maximum_step=args.maximum_step,
        phonon_cutoff=args.phonon_cutoff,
        gate_directory=args.gate_directory,
        tangent_regularization=args.tangent_regularization,
        relative_damping=args.relative_damping,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
