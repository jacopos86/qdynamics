"""Exact-state derivative gate for an order-parameterized moment hierarchy."""

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

from .exact_reference import exact_holstein_moment_hierarchy_trajectory
from .hubbard_dimer import DimerParameters
from .matrix_reference import matrix_state_to_closed_scalar_coordinates
from .moment_hierarchy import MomentHierarchy, moment_hierarchy


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


def _degree_indices(
    hierarchy: MomentHierarchy,
    degree: int,
) -> np.ndarray:
    return np.asarray(
        [
            index + 2
            for index, key in enumerate(hierarchy.moment_keys)
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


def _matrix_metrics(matrix_defect: np.ndarray) -> dict[str, Any]:
    blocks = {
        "rho": slice(0, 3),
        "B": slice(3, 7),
        "N": slice(7, 11),
        "A": slice(11, 17),
        "C": slice(17, 31),
    }
    result: dict[str, Any] = {}
    for name, block in blocks.items():
        norms = np.linalg.norm(matrix_defect[:, block], axis=1)
        result[name] = {
            "vector_rms": float(np.sqrt(np.mean(norms**2))),
            "maximum_norm": float(np.max(norms)),
        }
    return result


def _write_plot(
    path: Path,
    hierarchy: MomentHierarchy,
    times: np.ndarray,
    defect: np.ndarray,
    exact_derivative: np.ndarray,
    previous_terminal_defect: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))
    colors = ("#2378B5", "#2A9D68", "#D28E2B", "#C84A3A")
    for degree in range(1, hierarchy.maximum_degree + 1):
        indices = _degree_indices(hierarchy, degree)
        axes[0].plot(
            times,
            np.linalg.norm(defect[:, indices], axis=1),
            color=colors[(degree - 1) % len(colors)],
            label=f"degree {degree}",
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"dimensionless time $t\,t_{\rm hop}$")
    axes[0].set_ylabel(r"exact-state derivative defect $\|\Delta\dot x\|_2$")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)

    terminal = _degree_indices(hierarchy, hierarchy.maximum_degree)
    axes[1].plot(
        times,
        np.linalg.norm(exact_derivative[:, terminal], axis=1),
        color="#555555",
        label=f"exact degree-{hierarchy.maximum_degree} velocity",
    )
    axes[1].plot(
        times,
        np.linalg.norm(defect[:, terminal], axis=1),
        color="#C84A3A",
        label=(
            f"order-{hierarchy.maximum_degree + 1} cumulant defect"
        ),
    )
    axes[1].plot(
        times,
        np.linalg.norm(previous_terminal_defect, axis=1),
        color="#D28E2B",
        linestyle="--",
        label=(
            f"previous order-{hierarchy.maximum_degree} cumulant defect"
        ),
    )
    axes[1].set_xlabel(r"dimensionless time $t\,t_{\rm hop}$")
    axes[1].set_ylabel(r"terminal-block $\ell_2$ norm")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=7.5)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run_hierarchy_analysis(
    run_directory: Path,
    *,
    hierarchy: MomentHierarchy,
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
    """Measure one hierarchy and write retrievable gate artifacts."""

    if hierarchy.maximum_degree < 3:
        raise ValueError("the derivative gate requires hierarchy order >= 3")
    if phonon_cutoff not in convergence_cutoffs:
        raise ValueError("convergence_cutoffs must include phonon_cutoff")
    if not 0.0 < convergence_time <= final_time:
        raise ValueError("convergence_time must lie in (0, final_time]")
    if terminal_relative_defect_threshold <= 0.0:
        raise ValueError("terminal threshold must be positive")

    run_directory.mkdir(parents=True, exist_ok=True)
    times = _sample_times(final_time, sample_step)
    exact = exact_holstein_moment_hierarchy_trajectory(
        parameters,
        hierarchy=hierarchy,
        sample_times=times,
        phonon_cutoff=phonon_cutoff,
        eigensolver_tolerance=eigensolver_tolerance,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step=maximum_step,
    )
    approximate = np.asarray(
        [
            hierarchy.rhs(float(time), coordinates, parameters)
            for time, coordinates in zip(
                exact.times,
                exact.coordinates,
                strict=True,
            )
        ]
    )
    defect = approximate - exact.coordinate_derivatives
    degree_metrics = {
        str(degree): _block_metrics(
            defect,
            exact.coordinate_derivatives,
            _degree_indices(hierarchy, degree),
        )
        for degree in range(1, hierarchy.maximum_degree + 1)
    }

    previous = moment_hierarchy(hierarchy.maximum_degree - 1)
    previous_coordinates = exact.coordinates[:, : previous.coordinate_count]
    previous_exact_derivatives = exact.coordinate_derivatives[
        :, : previous.coordinate_count
    ]
    previous_approximate = np.asarray(
        [
            previous.rhs(float(time), coordinates, parameters)
            for time, coordinates in zip(
                exact.times,
                previous_coordinates,
                strict=True,
            )
        ]
    )
    previous_defect = previous_approximate - previous_exact_derivatives
    previous_terminal_indices = _degree_indices(
        previous,
        previous.maximum_degree,
    )
    previous_terminal_metrics = _block_metrics(
        previous_defect,
        previous_exact_derivatives,
        previous_terminal_indices,
    )
    previous_terminal_defect = previous_defect[
        :, previous_terminal_indices
    ]

    mapped_approximate = np.asarray(
        [
            matrix_state_to_closed_scalar_coordinates(
                hierarchy.matrix_derivative(coordinates, derivative)
            )
            for coordinates, derivative in zip(
                exact.coordinates,
                approximate,
                strict=True,
            )
        ]
    )
    mapped_exact = np.asarray(
        [
            matrix_state_to_closed_scalar_coordinates(
                hierarchy.matrix_derivative(coordinates, derivative)
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
            cutoff_times = exact.times[[0, target_index]]
            cutoff_coordinates = exact.coordinates[[0, target_index]]
            cutoff_derivatives = exact.coordinate_derivatives[
                [0, target_index]
            ]
        else:
            cutoff_exact = exact_holstein_moment_hierarchy_trajectory(
                parameters,
                hierarchy=hierarchy,
                sample_times=np.asarray([0.0, convergence_time]),
                phonon_cutoff=cutoff,
                eigensolver_tolerance=eigensolver_tolerance,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                maximum_step=maximum_step,
            )
            cutoff_times = cutoff_exact.times
            cutoff_coordinates = cutoff_exact.coordinates
            cutoff_derivatives = cutoff_exact.coordinate_derivatives
        cutoff_approximate = np.asarray(
            [
                hierarchy.rhs(float(time), coordinates, parameters)
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
                        cutoff_defect[:, _degree_indices(hierarchy, degree)],
                        axis=1,
                    )
                )
            )
            for degree in range(1, hierarchy.maximum_degree + 1)
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
    control = exact_holstein_moment_hierarchy_trajectory(
        control_parameters,
        hierarchy=hierarchy,
        sample_times=np.linspace(0.0, 2.0, 9),
        phonon_cutoff=3,
        eigensolver_tolerance=eigensolver_tolerance,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step=maximum_step,
    )
    control_approximate = np.asarray(
        [
            hierarchy.rhs(float(time), coordinates, control_parameters)
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

    terminal_indices = _degree_indices(hierarchy, hierarchy.maximum_degree)
    terminal_component_rms = np.sqrt(
        np.mean(defect[:, terminal_indices] ** 2, axis=0)
    )
    top_terminal_components = [
        {
            "name": hierarchy.state_names[int(terminal_indices[index])],
            "rms": float(terminal_component_rms[index]),
            "maximum_abs": float(
                np.max(np.abs(defect[:, int(terminal_indices[index])]))
            ),
        }
        for index in np.argsort(terminal_component_rms)[::-1][:10]
    ]

    lower_gate_passed = all(
        degree_metrics[str(degree)]["relative_vector_rms"] < 1e-4
        for degree in range(1, hierarchy.maximum_degree)
    )
    terminal_metrics = degree_metrics[str(hierarchy.maximum_degree)]
    terminal_gate_passed = (
        terminal_metrics["relative_vector_rms"]
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
        "hierarchy": {
            "maximum_degree": hierarchy.maximum_degree,
            "terminal_cumulant_order": hierarchy.maximum_degree + 1,
            "center_coordinates": 2,
            "moments_by_degree": {
                str(degree): int(_degree_indices(hierarchy, degree).size)
                for degree in range(1, hierarchy.maximum_degree + 1)
            },
            "total_coordinates": hierarchy.coordinate_count,
        },
        "degree_metrics": degree_metrics,
        "previous_hierarchy_terminal_metrics": previous_terminal_metrics,
        "matrix_projection_derivative_metrics": _matrix_metrics(
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
                "proceed to raw autonomous propagation"
                if lower_gate_passed and terminal_gate_passed
                else (
                    "defer raw propagation and barrier adaptation; retained "
                    f"degree {hierarchy.maximum_degree} repairs the previous "
                    "terminal equations, but the new terminal cumulant rule "
                    "fails its derivative gate"
                )
            ),
        },
    }

    prefix = f"order_{hierarchy.maximum_degree}_cumulant_derivative_gate"
    trajectory_path = run_directory / f"{prefix}.npz"
    np.savez_compressed(
        trajectory_path,
        times=exact.times,
        exact_coordinates=exact.coordinates,
        exact_coordinate_derivatives=exact.coordinate_derivatives,
        approximate_coordinate_derivatives=approximate,
        coordinate_derivative_defects=defect,
        previous_terminal_derivative_defects=previous_terminal_defect,
        matrix_projection_derivative_defects=matrix_defect,
    )
    plot_path = run_directory / f"{prefix}.png"
    _write_plot(
        plot_path,
        hierarchy,
        exact.times,
        defect,
        exact.coordinate_derivatives,
        previous_terminal_defect,
    )
    summary_path = run_directory / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("moment_hierarchy.py"),
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
    (run_directory / "runtime_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--maximum-degree", type=int, default=4)
    parser.add_argument("--lambda-ep", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--drive", type=float, default=1.0)
    parser.add_argument("--final-time", type=float, default=4.0)
    parser.add_argument("--sample-step", type=float, default=0.01)
    parser.add_argument("--phonon-cutoff", type=int, default=20)
    parser.add_argument("--maximum-step", type=float, default=0.01)
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = run_hierarchy_analysis(
        args.run_directory,
        hierarchy=moment_hierarchy(args.maximum_degree),
        parameters=DimerParameters(
            lambda_ep=args.lambda_ep,
            gamma=args.gamma,
            drive_amplitude=args.drive,
        ),
        final_time=args.final_time,
        sample_step=args.sample_step,
        phonon_cutoff=args.phonon_cutoff,
        maximum_step=args.maximum_step,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
