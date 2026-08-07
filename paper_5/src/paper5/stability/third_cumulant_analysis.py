"""Measure the derivative-level gate for the 47-coordinate closure.

The exact truncated Hamiltonian is used only as an offline oracle.  This
diagnostic first checks the lower retained equations, then measures the
terminal degree-three defect caused by setting connected fourth cumulants to
zero.  Raw approximate propagation is intentionally deferred when that
terminal gate fails.
"""

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

from .exact_reference import exact_holstein_third_cumulant_trajectory
from .hubbard_dimer import DimerParameters
from .matrix_reference import matrix_state_to_closed_scalar_coordinates
from .third_cumulant import (
    THIRD_CUMULANT_MOMENT_KEYS,
    THIRD_CUMULANT_STATE_NAMES,
    third_cumulant_matrix_derivative,
    third_cumulant_rhs,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_times(final_time: float, sample_step: float) -> np.ndarray:
    if final_time <= 0.0:
        raise ValueError("final_time must be positive")
    if sample_step <= 0.0:
        raise ValueError("sample_step must be positive")
    intervals = int(round(final_time / sample_step))
    if not np.isclose(intervals * sample_step, final_time, atol=1e-12):
        raise ValueError("final_time must be an integer multiple of sample_step")
    return np.linspace(0.0, final_time, intervals + 1)


def _degree_indices(degree: int) -> np.ndarray:
    return np.asarray(
        [
            index + 2
            for index, key in enumerate(THIRD_CUMULANT_MOMENT_KEYS)
            if key.degree == degree
        ],
        dtype=int,
    )


def _block_metrics(
    defect: np.ndarray,
    exact_derivative: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    selected_defect = defect[:, indices]
    selected_exact = exact_derivative[:, indices]
    defect_norms = np.linalg.norm(selected_defect, axis=1)
    exact_norms = np.linalg.norm(selected_exact, axis=1)
    vector_rms = float(np.sqrt(np.mean(defect_norms**2)))
    exact_vector_rms = float(np.sqrt(np.mean(exact_norms**2)))
    return {
        "coordinate_count": int(indices.size),
        "component_rms": float(np.sqrt(np.mean(selected_defect**2))),
        "vector_rms": vector_rms,
        "maximum_norm": float(np.max(defect_norms)),
        "exact_velocity_vector_rms": exact_vector_rms,
        "relative_vector_rms": float(
            vector_rms / max(exact_vector_rms, np.finfo(float).tiny)
        ),
    }


def _matrix_block_metrics(matrix_defect: np.ndarray) -> dict[str, Any]:
    blocks = {
        "rho": slice(0, 3),
        "B": slice(3, 7),
        "N": slice(7, 11),
        "A": slice(11, 17),
        "C": slice(17, 31),
    }
    metrics: dict[str, Any] = {}
    for name, block in blocks.items():
        norms = np.linalg.norm(matrix_defect[:, block], axis=1)
        metrics[name] = {
            "vector_rms": float(np.sqrt(np.mean(norms**2))),
            "maximum_norm": float(np.max(norms)),
        }
    return metrics


def _write_plot(
    path: Path,
    times: np.ndarray,
    defect: np.ndarray,
    exact_derivative: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.25))
    colors = {1: "#2378B5", 2: "#2A9D68", 3: "#C84A3A"}
    for degree in (1, 2, 3):
        indices = _degree_indices(degree)
        axes[0].plot(
            times,
            np.linalg.norm(defect[:, indices], axis=1),
            color=colors[degree],
            label=f"degree {degree}",
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"dimensionless time $t\,t_{\rm hop}$")
    axes[0].set_ylabel(r"exact-state derivative defect $\|\Delta\dot x\|_2$")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)

    terminal = _degree_indices(3)
    axes[1].plot(
        times,
        np.linalg.norm(exact_derivative[:, terminal], axis=1),
        color="#555555",
        label="exact degree-3 velocity",
    )
    axes[1].plot(
        times,
        np.linalg.norm(defect[:, terminal], axis=1),
        color=colors[3],
        label="fourth-cumulant closure defect",
    )
    axes[1].set_xlabel(r"dimensionless time $t\,t_{\rm hop}$")
    axes[1].set_ylabel(r"terminal-block $\ell_2$ norm")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run_analysis(
    run_directory: Path,
    *,
    parameters: DimerParameters,
    final_time: float = 4.0,
    sample_step: float = 0.01,
    phonon_cutoff: int = 20,
    convergence_cutoffs: tuple[int, ...] = (12, 16, 20),
    convergence_time: float = 0.5,
    eigensolver_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-12,
    maximum_step: float = 0.01,
    terminal_relative_defect_threshold: float = 0.1,
) -> dict[str, Any]:
    """Write the exact-state derivative audit and its pass/fail decision."""

    if phonon_cutoff not in convergence_cutoffs:
        raise ValueError("convergence_cutoffs must include phonon_cutoff")
    if not 0.0 < convergence_time <= final_time:
        raise ValueError("convergence_time must lie in (0, final_time]")
    if terminal_relative_defect_threshold <= 0.0:
        raise ValueError("terminal threshold must be positive")

    run_directory.mkdir(parents=True, exist_ok=True)
    times = _sample_times(final_time, sample_step)
    exact = exact_holstein_third_cumulant_trajectory(
        parameters,
        sample_times=times,
        phonon_cutoff=phonon_cutoff,
        eigensolver_tolerance=eigensolver_tolerance,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step=maximum_step,
    )
    approximate_derivatives = np.asarray(
        [
            third_cumulant_rhs(float(time), coordinates, parameters)
            for time, coordinates in zip(
                exact.times,
                exact.coordinates,
                strict=True,
            )
        ]
    )
    defect = approximate_derivatives - exact.coordinate_derivatives

    degree_metrics = {
        str(degree): _block_metrics(
            defect,
            exact.coordinate_derivatives,
            _degree_indices(degree),
        )
        for degree in (1, 2, 3)
    }
    center_metrics = _block_metrics(
        defect,
        exact.coordinate_derivatives,
        np.asarray([0, 1], dtype=int),
    )

    mapped_approximate = np.asarray(
        [
            matrix_state_to_closed_scalar_coordinates(
                third_cumulant_matrix_derivative(coordinates, derivative)
            )
            for coordinates, derivative in zip(
                exact.coordinates,
                approximate_derivatives,
                strict=True,
            )
        ]
    )
    mapped_exact = np.asarray(
        [
            matrix_state_to_closed_scalar_coordinates(
                third_cumulant_matrix_derivative(coordinates, derivative)
            )
            for coordinates, derivative in zip(
                exact.coordinates,
                exact.coordinate_derivatives,
                strict=True,
            )
        ]
    )
    matrix_defect = mapped_approximate - mapped_exact

    target_index = int(round(convergence_time / sample_step))
    cutoff_metrics: dict[str, Any] = {}
    for cutoff in convergence_cutoffs:
        if cutoff == phonon_cutoff:
            cutoff_coordinates = exact.coordinates[[0, target_index]]
            cutoff_derivatives = exact.coordinate_derivatives[
                [0, target_index]
            ]
            cutoff_times = exact.times[[0, target_index]]
        else:
            cutoff_exact = exact_holstein_third_cumulant_trajectory(
                parameters,
                sample_times=np.asarray([0.0, convergence_time]),
                phonon_cutoff=cutoff,
                eigensolver_tolerance=eigensolver_tolerance,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                maximum_step=maximum_step,
            )
            cutoff_coordinates = cutoff_exact.coordinates
            cutoff_derivatives = cutoff_exact.coordinate_derivatives
            cutoff_times = cutoff_exact.times
        cutoff_approximate = np.asarray(
            [
                third_cumulant_rhs(float(time), coordinates, parameters)
                for time, coordinates in zip(
                    cutoff_times,
                    cutoff_coordinates,
                    strict=True,
                )
            ]
        )
        cutoff_defect = cutoff_approximate - cutoff_derivatives
        cutoff_metrics[str(cutoff)] = {
            f"degree_{degree}_maximum_norm": float(
                np.max(
                    np.linalg.norm(
                        cutoff_defect[:, _degree_indices(degree)], axis=1
                    )
                )
            )
            for degree in (1, 2, 3)
        }
        cutoff_metrics[str(cutoff)]["center_maximum_norm"] = float(
            np.max(np.linalg.norm(cutoff_defect[:, :2], axis=1))
        )

    control_parameters = DimerParameters(
        hopping=parameters.hopping,
        gamma=parameters.gamma,
        lambda_ep=0.0,
        drive_amplitude=parameters.drive_amplitude,
        pulse_width=parameters.pulse_width,
    )
    control = exact_holstein_third_cumulant_trajectory(
        control_parameters,
        sample_times=np.linspace(0.0, 2.0, 9),
        phonon_cutoff=3,
        eigensolver_tolerance=eigensolver_tolerance,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step=maximum_step,
    )
    control_approximate = np.asarray(
        [
            third_cumulant_rhs(float(time), coordinates, control_parameters)
            for time, coordinates in zip(
                control.times,
                control.coordinates,
                strict=True,
            )
        ]
    )
    control_maximum_defect = float(
        np.max(np.abs(control_approximate - control.coordinate_derivatives))
    )

    terminal_indices = _degree_indices(3)
    terminal_component_rms = np.sqrt(
        np.mean(defect[:, terminal_indices] ** 2, axis=0)
    )
    top_terminal_components = []
    for local_index in np.argsort(terminal_component_rms)[::-1][:10]:
        coordinate_index = int(terminal_indices[local_index])
        top_terminal_components.append(
            {
                "name": THIRD_CUMULANT_STATE_NAMES[coordinate_index],
                "rms": float(terminal_component_rms[local_index]),
                "maximum_abs": float(
                    np.max(np.abs(defect[:, coordinate_index]))
                ),
            }
        )

    lower_gate_passed = (
        degree_metrics["1"]["relative_vector_rms"] < 1e-4
        and degree_metrics["2"]["relative_vector_rms"] < 1e-4
    )
    terminal_gate_passed = (
        degree_metrics["3"]["relative_vector_rms"]
        <= terminal_relative_defect_threshold
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "classification": "exploratory_local_not_promoted",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            **asdict(parameters),
            "coupling": parameters.coupling,
            "phonon_cutoff": phonon_cutoff,
            "final_time": final_time,
            "sample_step": sample_step,
            "maximum_step": maximum_step,
            "relative_tolerance": relative_tolerance,
            "absolute_tolerance": absolute_tolerance,
        },
        "basis": {
            "center_coordinates": 2,
            "degree_1_moments": int(_degree_indices(1).size),
            "degree_2_moments": int(_degree_indices(2).size),
            "degree_3_moments": int(_degree_indices(3).size),
            "total_coordinates": len(THIRD_CUMULANT_STATE_NAMES),
        },
        "center_metrics": center_metrics,
        "degree_metrics": degree_metrics,
        "matrix_projection_derivative_metrics": _matrix_block_metrics(
            matrix_defect
        ),
        "cutoff_convergence_at_t_0_and_t_0_5": cutoff_metrics,
        "decoupled_control_maximum_component_defect": (
            control_maximum_defect
        ),
        "undriven_initial_terminal_residual_norm": float(
            np.linalg.norm(defect[0, terminal_indices])
        ),
        "top_terminal_components": top_terminal_components,
        "validation_gate": {
            "lower_equations_passed": bool(lower_gate_passed),
            "terminal_relative_defect_threshold": (
                terminal_relative_defect_threshold
            ),
            "terminal_closure_passed": bool(terminal_gate_passed),
            "raw_propagation_authorized": bool(
                lower_gate_passed and terminal_gate_passed
            ),
            "decision": (
                "defer raw propagation and barrier adaptation; the complete "
                "third-order state repairs the lower C velocity, but the "
                "zero-fourth-cumulant terminal rule is not accurate enough"
                if not terminal_gate_passed
                else "proceed to raw autonomous propagation"
            ),
        },
    }

    trajectory_path = run_directory / "third_cumulant_derivative_gate.npz"
    np.savez_compressed(
        trajectory_path,
        times=exact.times,
        exact_coordinates=exact.coordinates,
        exact_coordinate_derivatives=exact.coordinate_derivatives,
        approximate_coordinate_derivatives=approximate_derivatives,
        coordinate_derivative_defects=defect,
        matrix_projection_derivative_defects=matrix_defect,
    )
    plot_path = run_directory / "third_cumulant_derivative_gate.png"
    _write_plot(
        plot_path,
        exact.times,
        defect,
        exact.coordinate_derivatives,
    )
    summary_path = run_directory / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    source_paths = (
        Path(__file__),
        Path(__file__).with_name("third_cumulant.py"),
        Path(__file__).with_name("exact_reference.py"),
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
    manifest_path = run_directory / "runtime_manifest.json"
    manifest_path.write_text(
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
    parser.add_argument("--final-time", type=float, default=4.0)
    parser.add_argument("--sample-step", type=float, default=0.01)
    parser.add_argument("--phonon-cutoff", type=int, default=20)
    parser.add_argument("--maximum-step", type=float, default=0.01)
    parser.add_argument(
        "--terminal-relative-defect-threshold",
        type=float,
        default=0.1,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = run_analysis(
        args.run_directory,
        parameters=DimerParameters(
            lambda_ep=args.lambda_ep,
            gamma=args.gamma,
            drive_amplitude=args.drive,
        ),
        final_time=args.final_time,
        sample_step=args.sample_step,
        phonon_cutoff=args.phonon_cutoff,
        maximum_step=args.maximum_step,
        terminal_relative_defect_threshold=(
            args.terminal_relative_defect_threshold
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
