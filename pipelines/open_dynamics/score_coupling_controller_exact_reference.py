"""Score a completed coupling-controller run against exact Hamiltonian dynamics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from paper5.stability.exact_reference import exact_holstein_driven_trajectory
from paper5.stability.hubbard_dimer import DimerParameters, GaussianSineDrive
from paper5.stability.initial_condition_sensitivity import physicality_diagnostics
from paper5.stability.matrix_reference import (
    matrix_state_to_closed_scalar_coordinates,
    matrix_total_energy,
)
from plot_high_coupling_controller_observables import (
    OBSERVABLE_NAMES,
    observable_trajectories,
)


class ProtocolParameters:
    """Delegate static dimer parameters while replacing the pulse."""

    def __init__(self, base: DimerParameters, drive: GaussianSineDrive) -> None:
        self._base = base
        self._drive = drive

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def drive_difference(self, time: float) -> float:
        return self._drive.difference(time)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _time_rms(times: np.ndarray, values: np.ndarray) -> float:
    duration = float(times[-1] - times[0])
    return float(np.sqrt(np.trapezoid(values**2, times) / duration))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phonon-cutoff", type=int, default=16)
    parser.add_argument("--relative-tolerance", type=float, default=1e-10)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--maximum-step", type=float, default=0.05)
    return parser


def main() -> int:
    args = _parser().parse_args()
    with (args.run_dir / "plan.json").open(encoding="utf-8") as handle:
        plan = json.load(handle)
    archive = np.load(args.run_dir / "trajectories.npz")
    times = np.asarray(archive["corrected_times"], dtype=float)
    raw_times = np.asarray(archive["raw_times"], dtype=float)
    if not np.array_equal(raw_times, times):
        raise SystemExit("raw and corrected sample grids must match exact scoring")

    values = plan["parameters"]
    base = DimerParameters(
        hopping=float(values["hopping"]),
        gamma=float(values["gamma"]),
        lambda_ep=float(values["lambda_ep"]),
        drive_amplitude=float(plan["drive"]["amplitude"]),
        pulse_width=float(plan["drive"]["pulse_width"]),
    )
    drive = GaussianSineDrive(
        amplitude=float(plan["drive"]["amplitude"]),
        pulse_width=float(plan["drive"]["pulse_width"]),
        delays=tuple(float(value) for value in plan["drive"]["delays"]),
    )
    parameters = ProtocolParameters(base, drive)
    exact = exact_holstein_driven_trajectory(
        parameters,
        sample_times=times,
        phonon_cutoff=args.phonon_cutoff,
        eigensolver_tolerance=1e-12,
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
        maximum_step=args.maximum_step,
    )
    exact_states = np.stack(
        [
            matrix_state_to_closed_scalar_coordinates(state)
            for state in exact.matrix_states
        ]
    )
    initial_difference = float(
        np.linalg.norm(exact_states[0] - np.asarray(archive["initial_state"]))
    )
    if initial_difference > 1e-9:
        raise RuntimeError(
            "exact reference does not reproduce the run initial moments: "
            f"{initial_difference}"
        )

    exact_margins = np.empty((times.size, 4), dtype=float)
    exact_traces = np.empty(times.size, dtype=float)
    exact_energies = np.empty(times.size, dtype=float)
    for index, (state, coordinates) in enumerate(
        zip(exact.matrix_states, exact_states, strict=True)
    ):
        exact_margins[index], exact_traces[index] = physicality_diagnostics(
            coordinates
        )
        exact_energies[index] = matrix_total_energy(state, parameters)

    exact_path = args.run_dir / "exact_trajectory.npz"
    np.savez_compressed(
        exact_path,
        times=times,
        states=exact_states,
        margins=exact_margins,
        trace_residuals=exact_traces,
        internal_energies=exact_energies,
        state_norms=exact.state_norms,
    )

    exact_observables = observable_trajectories(exact_states, parameters)
    comparisons: dict[str, object] = {}
    for lane in ("raw", "corrected"):
        lane_states = np.asarray(archive[f"{lane}_states"], dtype=float)
        lane_observables = observable_trajectories(lane_states, parameters)
        observable_errors = {}
        for index, name in enumerate(OBSERVABLE_NAMES):
            difference = lane_observables[:, index] - exact_observables[:, index]
            observable_errors[name] = {
                "time_rms": _time_rms(times, difference),
                "maximum_absolute": float(np.max(np.abs(difference))),
            }
        coordinate_difference = lane_states - exact_states
        coordinate_l2 = np.linalg.norm(coordinate_difference, axis=1)
        comparisons[lane] = {
            "coordinate_l2_time_rms": _time_rms(times, coordinate_l2),
            "coordinate_l2_maximum": float(np.max(coordinate_l2)),
            "observables": observable_errors,
        }

    metrics = {
        "schema": "paper5.coupling_controller_exact_score.v1",
        "classification": "diagnostic_exploratory_not_promoted",
        "reference": {
            "method": "cutoff-truncated full-wavefunction DOP853 propagation",
            "phonon_cutoff": args.phonon_cutoff,
            "relative_tolerance": args.relative_tolerance,
            "absolute_tolerance": args.absolute_tolerance,
            "maximum_step": args.maximum_step,
            "function_evaluations": exact.function_evaluations,
            "maximum_state_norm_error": float(
                np.max(np.abs(exact.state_norms - 1.0))
            ),
            "initial_coordinate_l2_difference": initial_difference,
            "minimum_margins": np.min(exact_margins, axis=0).tolist(),
            "maximum_correlation_trace_residual": float(np.max(exact_traces)),
        },
        "comparisons": comparisons,
    }
    metrics_path = args.run_dir / "exact_metrics.json"
    _atomic_json(metrics_path, metrics)
    _atomic_json(
        args.run_dir / "exact_runtime_manifest.json",
        {
            "schema": "paper5.coupling_controller_exact_runtime.v1",
            "artifact_hashes": {
                exact_path.name: _sha256(exact_path),
                metrics_path.name: _sha256(metrics_path),
            },
        },
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
