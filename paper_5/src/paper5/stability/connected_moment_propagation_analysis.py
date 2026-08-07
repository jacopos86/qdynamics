"""Short matched propagation gate for the autonomous connected-moment source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import scipy

from .connected_moment_closure import (
    conditional_k_closed_scalar_rhs,
    conditional_k_pauli_repaired_closed_scalar_rhs,
)
from .electron_phonon_analysis import (
    FixedStepTrajectory,
    _correction_history_metrics,
    _protocol_metrics,
    integrate_closed_rk4,
)
from .exact_reference import exact_holstein_driven_trajectory
from .hubbard_dimer import DimerParameters, FloatArray
from .matrix_reference import (
    closed_scalar_rhs,
    closed_scalar_to_matrix_state,
    electron_phonon_moment_matrix,
    matrix_state_to_closed_scalar_coordinates,
    pauli_repaired_closed_scalar_rhs,
)

ScalarRhs = Callable[[float, FloatArray, DimerParameters], FloatArray]

LANE_RHS: dict[str, ScalarRhs] = {
    "controller": closed_scalar_rhs,
    "pauli_controller": pauli_repaired_closed_scalar_rhs,
    "conditional_k_controller": conditional_k_closed_scalar_rhs,
    "conditional_k_pauli_controller": (
        conditional_k_pauli_repaired_closed_scalar_rhs
    ),
}
LANE_LABELS = {
    "controller": "archive + controller",
    "pauli_controller": "Pauli + controller",
    "conditional_k_controller": "conditional K + controller",
    "conditional_k_pauli_controller": "conditional K + Pauli + controller",
}
LANE_COLORS = {
    "controller": "#376f9e",
    "pauli_controller": "#d28b16",
    "conditional_k_controller": "#6a51a3",
    "conditional_k_pauli_controller": "#2f7d4a",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sample_times(final_time: float, time_step: float) -> FloatArray:
    if final_time <= 0.0 or time_step <= 0.0:
        raise ValueError("final_time and time_step must be positive")
    steps = int(round(final_time / time_step))
    if not np.isclose(steps * time_step, final_time, atol=1e-12):
        raise ValueError("time_step must divide final_time")
    return np.linspace(0.0, final_time, steps + 1)


def _joint_minima(coordinates: FloatArray) -> FloatArray:
    return np.asarray(
        [
            np.linalg.eigvalsh(
                electron_phonon_moment_matrix(
                    closed_scalar_to_matrix_state(row)
                )
            )[0]
            for row in coordinates
        ],
        dtype=float,
    )


def _case_arrays(
    times: FloatArray,
    exact_coordinates: FloatArray,
    lanes: dict[str, FixedStepTrajectory],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "times": times,
        "exact_coordinates": exact_coordinates,
        "exact_joint_gram_minimum_eigenvalue": _joint_minima(
            exact_coordinates
        ),
    }
    for name, trajectory in lanes.items():
        arrays[f"{name}_coordinates"] = trajectory.coordinates
        arrays[f"{name}_correction_coordinates"] = (
            trajectory.correction_coordinates
        )
        arrays[f"{name}_joint_gram_minimum_eigenvalue"] = _joint_minima(
            trajectory.coordinates
        )
    return arrays


def _write_figure(path: Path, arrays: dict[str, np.ndarray]) -> None:
    times = arrays["times"]
    exact = arrays["exact_coordinates"]
    figure, axes = plt.subplots(3, 1, figsize=(8.0, 8.4), sharex=True)
    for name in LANE_RHS:
        coordinates = arrays[f"{name}_coordinates"]
        coordinate_error = np.linalg.norm(coordinates - exact, axis=1)
        correction_norm = np.linalg.norm(
            arrays[f"{name}_correction_coordinates"],
            axis=1,
        )
        axes[0].plot(
            times,
            np.maximum(coordinate_error, 1e-15),
            label=LANE_LABELS[name],
            color=LANE_COLORS[name],
        )
        axes[1].plot(
            times,
            arrays[f"{name}_joint_gram_minimum_eigenvalue"],
            label=LANE_LABELS[name],
            color=LANE_COLORS[name],
        )
        axes[2].plot(
            times,
            correction_norm,
            label=LANE_LABELS[name],
            color=LANE_COLORS[name],
        )
    axes[1].plot(
        times,
        arrays["exact_joint_gram_minimum_eigenvalue"],
        color="black",
        linestyle="--",
        label="exact cutoff reference",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\|x-x_{\rm ex}\|_2$")
    axes[0].set_title("Short-horizon trajectory error")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel(r"$\lambda_{\min}(\mathcal{G})$")
    axes[1].set_title("Joint representability margin")
    axes[2].set_ylabel("controller correction norm")
    axes[2].set_xlabel(r"time $t\,t_{\rm hop}$")
    axes[2].set_title("Minimum-norm controller action")
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run_analysis(
    run_directory: Path,
    *,
    parameters: DimerParameters,
    final_time: float = 4.0,
    time_step: float = 0.01,
    phonon_cutoff: int = 16,
    activation_margin: float = 1e-5,
    barrier_rate: float = 5.0,
    cone_tolerance: float = 1e-8,
    maximum_constraints: int = 128,
    exact_relative_tolerance: float = 1e-10,
    exact_absolute_tolerance: float = 1e-12,
    exact_maximum_step: float = 0.01,
) -> dict[str, Any]:
    """Run the authorized four-lane, controller-stabilized propagation gate."""

    run_directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    times = _sample_times(final_time, time_step)
    exact = exact_holstein_driven_trajectory(
        parameters,
        sample_times=times,
        phonon_cutoff=phonon_cutoff,
        relative_tolerance=exact_relative_tolerance,
        absolute_tolerance=exact_absolute_tolerance,
        maximum_step=min(exact_maximum_step, time_step),
    )
    exact_coordinates = np.asarray(
        [
            matrix_state_to_closed_scalar_coordinates(state)
            for state in exact.matrix_states
        ],
        dtype=float,
    )
    initial_state = exact_coordinates[0].copy()
    lanes: dict[str, FixedStepTrajectory] = {}
    for name, rhs in LANE_RHS.items():
        print(
            json.dumps(
                {"event": "lane_started", "lane": name, "time": _utc_now()},
                sort_keys=True,
            ),
            flush=True,
        )
        lanes[name] = integrate_closed_rk4(
            parameters,
            initial_state,
            final_time=final_time,
            time_step=time_step,
            corrected=True,
            rhs_override=rhs,
            activation_margin=activation_margin,
            barrier_rate=barrier_rate,
            cone_tolerance=cone_tolerance,
            maximum_constraints=maximum_constraints,
        )
        print(
            json.dumps(
                {"event": "lane_completed", "lane": name, "time": _utc_now()},
                sort_keys=True,
            ),
            flush=True,
        )

    lane_metrics: dict[str, Any] = {}
    for name, trajectory in lanes.items():
        metrics, _ = _protocol_metrics(
            times,
            exact_coordinates,
            trajectory,
            parameters,
        )
        lane_metrics[name] = {
            **metrics,
            "controller_history": _correction_history_metrics(trajectory),
        }

    parent = lane_metrics["pauli_controller"]
    candidate = lane_metrics["conditional_k_pauli_controller"]
    parent_c_error = parent["block_errors"]["C"]["rms_frobenius_error"]
    candidate_c_error = candidate["block_errors"]["C"]["rms_frobenius_error"]
    parent_total_error = parent["maximum_coordinate_l2_error"]
    candidate_total_error = candidate["maximum_coordinate_l2_error"]
    parent_effort = parent["controller_history"]["rms_correction_norm"]
    candidate_effort = candidate["controller_history"]["rms_correction_norm"]
    candidate_joint_minimum = candidate["certificates"][
        "minimum_joint_gram_eigenvalue"
    ]
    c_ratio = candidate_c_error / max(parent_c_error, 1e-30)
    total_ratio = candidate_total_error / max(parent_total_error, 1e-30)
    effort_ratio = candidate_effort / max(parent_effort, 1e-30)
    representability_passed = candidate_joint_minimum >= -cone_tolerance
    material_accuracy_passed = c_ratio <= 0.95 and total_ratio <= 1.05
    refinement_authorized = representability_passed and material_accuracy_passed

    summary: dict[str, Any] = {
        "schema_version": 1,
        "classification": "exploratory_local_not_promoted",
        "created_at_utc": _utc_now(),
        "scientific_question": (
            "Does the autonomous conditional-regression connected-moment "
            "source improve the controller-stabilized 31-coordinate "
            "trajectory, alone or with the same-spin Pauli repair?"
        ),
        "parameters": {
            "hopping": parameters.hopping,
            "gamma": parameters.gamma,
            "lambda_ep": parameters.lambda_ep,
            "drive_amplitude": parameters.drive_amplitude,
            "pulse_width": parameters.pulse_width,
            "coupling": parameters.coupling,
            "final_time": final_time,
            "time_step": time_step,
            "phonon_cutoff": phonon_cutoff,
            "activation_margin": activation_margin,
            "barrier_rate": barrier_rate,
            "cone_tolerance": cone_tolerance,
            "maximum_constraints": maximum_constraints,
        },
        "exact": {
            "hilbert_space_dimension": 4 * (phonon_cutoff + 1) ** 2,
            "function_evaluations": exact.function_evaluations,
            "maximum_norm_defect": float(
                np.max(np.abs(exact.state_norms - 1.0))
            ),
        },
        "lanes": lane_metrics,
        "gate": {
            "candidate_C_rms_error_ratio_over_Pauli_parent": c_ratio,
            "candidate_maximum_coordinate_error_ratio_over_Pauli_parent": (
                total_ratio
            ),
            "candidate_controller_rms_ratio_over_Pauli_parent": effort_ratio,
            "candidate_minimum_joint_gram_eigenvalue": candidate_joint_minimum,
            "representability_passed": representability_passed,
            "material_accuracy_passed": material_accuracy_passed,
            "step_refinement_authorized": refinement_authorized,
            "decision": (
                "repeat at half the RK4 step"
                if refinement_authorized
                else "reject or revise the autonomous K approximation"
            ),
        },
        "exact_reference_usage": (
            "common initial state and post-run scoring only; never queried by "
            "an integration RHS or controller"
        ),
        "wall_time_seconds": time.perf_counter() - started,
    }

    arrays = _case_arrays(times, exact_coordinates, lanes)
    arrays_path = run_directory / "trajectories.npz"
    plot_path = run_directory / "propagation_gate.png"
    summary_path = run_directory / "summary.json"
    manifest_path = run_directory / "runtime_manifest.json"
    _write_npz_atomic(arrays_path, arrays)
    _write_figure(plot_path, arrays)
    _write_json_atomic(summary_path, summary)
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("connected_moment_closure.py"),
        Path(__file__).with_name("electron_phonon_analysis.py"),
        Path(__file__).with_name("matrix_reference.py"),
        Path(__file__).with_name("exact_reference.py"),
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "command": "python -m paper5.stability.connected_moment_propagation_analysis",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "source_hashes": {
            str(path.resolve()): _sha256(path) for path in source_paths
        },
        "artifact_hashes": {
            path.name: _sha256(path)
            for path in (arrays_path, plot_path, summary_path)
        },
        "exact_reference_usage": summary["exact_reference_usage"],
    }
    _write_json_atomic(manifest_path, manifest)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--final-time", type=float, default=4.0)
    parser.add_argument("--time-step", type=float, default=0.01)
    parser.add_argument("--phonon-cutoff", type=int, default=16)
    parser.add_argument("--lambda-ep", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--drive", type=float, default=1.0)
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
        time_step=args.time_step,
        phonon_cutoff=args.phonon_cutoff,
    )
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
