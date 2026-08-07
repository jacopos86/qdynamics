"""Stitch checkpointed carried-witness segments and score archive baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from .apcm_carried_witness_analysis import _accuracy_metrics
from .hubbard_dimer import DimerParameters
from .matrix_reference import (
    closed_scalar_rhs,
    closed_scalar_to_matrix_state,
    electron_phonon_moment_matrix,
    local_holstein_couplings,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--segment",
        type=Path,
        action="append",
        required=True,
        help="Chunk directory or trajectory.npz; repeat in any order.",
    )
    parser.add_argument("--time-step", type=float, default=0.0025)
    parser.add_argument("--phonon-cutoff", type=int, default=16)
    parser.add_argument("--lambda-ep", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--drive", type=float, default=1.0)
    return parser


def _trajectory_path(segment: Path) -> Path:
    resolved = segment.resolve()
    return resolved / "trajectory.npz" if resolved.is_dir() else resolved


def _load_and_stitch(
    segments: list[Path],
    *,
    time_step: float,
) -> tuple[dict[str, np.ndarray], list[str]]:
    keys = (
        "times",
        "carried_states",
        "approximate_archive_coordinates",
        "exact_archive_coordinates",
        "minimum_unshifted_eigenvalues",
        "minimum_shifted_lower_bounds",
        "completion_correction_norms",
        "critical_modes",
    )
    loaded: list[tuple[Path, dict[str, np.ndarray]]] = []
    for segment in segments:
        path = _trajectory_path(segment)
        arrays = np.load(path)
        missing = [key for key in keys if key not in arrays]
        if missing:
            raise ValueError(f"{path} is missing {missing}")
        loaded.append(
            (path, {key: np.asarray(arrays[key]) for key in keys})
        )
    loaded.sort(key=lambda item: float(item[1]["times"][0]))

    rows: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    sources: list[str] = []
    last_time: float | None = None
    last_state: np.ndarray | None = None
    for path, arrays in loaded:
        sources.append(str(path))
        for row_index, time_value in enumerate(arrays["times"]):
            time_float = float(time_value)
            state = arrays["carried_states"][row_index]
            if last_time is not None and np.isclose(
                time_float, last_time, atol=1e-12
            ):
                if not np.allclose(state, last_state, atol=2e-10, rtol=0.0):
                    raise ValueError(
                        f"duplicate state mismatch at t={time_float}"
                    )
                continue
            if last_time is not None and not np.isclose(
                time_float - last_time,
                time_step,
                atol=1e-12,
            ):
                raise ValueError(
                    f"trajectory gap {last_time} -> {time_float}"
                )
            for key in keys:
                rows[key].append(np.asarray(arrays[key][row_index]))
            last_time = time_float
            last_state = state
    stitched = {key: np.asarray(value) for key, value in rows.items()}
    return stitched, sources


def _raw_archive_trajectory(
    parameters: DimerParameters,
    times: np.ndarray,
    initial_coordinates: np.ndarray,
) -> tuple[np.ndarray, int]:
    solution = solve_ivp(
        lambda time, state: closed_scalar_rhs(time, state, parameters),
        (float(times[0]), float(times[-1])),
        initial_coordinates,
        method="DOP853",
        t_eval=times,
        rtol=1e-10,
        atol=1e-12,
    )
    if not solution.success or solution.y.shape[1] != times.size:
        raise RuntimeError(f"raw archive propagation failed: {solution.message}")
    return np.asarray(solution.y.T, dtype=float), int(solution.nfev)


def _minimum_joint_gram(coordinates: np.ndarray) -> float:
    return float(
        min(
            np.linalg.eigvalsh(
                electron_phonon_moment_matrix(
                    closed_scalar_to_matrix_state(row)
                )
            )[0]
            for row in coordinates
        )
    )


def _observables(
    coordinates: np.ndarray,
    parameters: DimerParameters,
) -> dict[str, np.ndarray]:
    coupling = local_holstein_couplings(parameters)
    bare_electron = np.asarray(
        [[0.0, -parameters.hopping], [-parameters.hopping, 0.0]],
        dtype=complex,
    )
    values = {
        "occupation": [],
        "electron_energy": [],
        "phonon_energy": [],
        "electron_phonon_energy": [],
        "total_energy": [],
    }
    for row in coordinates:
        state = closed_scalar_to_matrix_state(row)
        rho = np.asarray(state.electron_density, dtype=complex)
        coherent = np.asarray(state.coherent_phonon, dtype=complex)
        phonon = np.asarray(state.phonon_density, dtype=complex)
        correlation = np.asarray(
            state.electron_phonon_correlation, dtype=complex
        )
        electron = 2.0 * np.trace(bare_electron @ rho).real
        phonon_value = parameters.omega_ph * (
            np.vdot(coherent, coherent).real + np.trace(phonon).real
        )
        amplitude = 0.0j
        for q in range(2):
            for one in range(2):
                for two in range(2):
                    amplitude += coupling[q, one, two] * (
                        coherent[q] * rho[two, one]
                        + correlation[q, two, one]
                    )
        interaction = 4.0 * amplitude.real
        values["occupation"].append(float(rho[0, 0].real))
        values["electron_energy"].append(float(electron))
        values["phonon_energy"].append(float(phonon_value))
        values["electron_phonon_energy"].append(float(interaction))
        values["total_energy"].append(
            float(electron + phonon_value + interaction)
        )
    return {key: np.asarray(value) for key, value in values.items()}


def _plot_observables(
    output: Path,
    times: np.ndarray,
    exact: dict[str, np.ndarray],
    raw: dict[str, np.ndarray],
    carried: dict[str, np.ndarray],
) -> None:
    panels = (
        ("occupation", r"site-0 occupation"),
        ("electron_energy", r"electronic energy"),
        ("phonon_energy", r"phonon energy"),
        ("electron_phonon_energy", r"electron--phonon energy"),
        ("total_energy", r"total energy"),
    )
    figure, axes = plt.subplots(3, 2, figsize=(8.0, 8.6), sharex=True)
    for axis, (key, label) in zip(axes.flat, panels, strict=False):
        axis.plot(times, exact[key], color="black", linewidth=1.6, label="exact cutoff-16")
        axis.plot(times, raw[key], color="#b64926", linestyle="--", linewidth=1.2, label="raw archive EOM")
        axis.plot(times, carried[key], color="#2468a2", linewidth=1.3, label="carried-witness guard")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes.flat[len(panels)].axis("off")
    for axis in axes[-1]:
        if axis.axison:
            axis.set_xlabel(r"time ($t_{\mathrm{hop}}^{-1}$)")
    axes.flat[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> int:
    args = _parser().parse_args()
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=False)
    trajectory, sources = _load_and_stitch(
        args.segment,
        time_step=args.time_step,
    )
    parameters = DimerParameters(
        hopping=1.0,
        gamma=args.gamma,
        lambda_ep=args.lambda_ep,
        drive_amplitude=args.drive,
    )
    raw, raw_evaluations = _raw_archive_trajectory(
        parameters,
        trajectory["times"],
        trajectory["approximate_archive_coordinates"][0],
    )
    exact = trajectory["exact_archive_coordinates"]
    carried = trajectory["approximate_archive_coordinates"]
    carried_metrics = _accuracy_metrics(parameters, exact, carried)
    raw_metrics = _accuracy_metrics(parameters, exact, raw)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "classification": "exploratory_local_not_promoted",
        "model": "stitched_archive_kpd_carried_witness_moment_flow",
        "parameters": {
            "hopping": parameters.hopping,
            "gamma": parameters.gamma,
            "lambda_ep": parameters.lambda_ep,
            "coupling": parameters.coupling,
            "drive_amplitude": parameters.drive_amplitude,
            "phonon_cutoff": args.phonon_cutoff,
        },
        "integration": {
            "method": "checkpointed SSPRK2 with finite-step radial atoms",
            "initial_time": float(trajectory["times"][0]),
            "last_time": float(trajectory["times"][-1]),
            "time_step": args.time_step,
            "completed_steps": int(trajectory["times"].size - 1),
        },
        "feasibility": {
            "minimum_carried_shifted_lower_bound": float(
                np.min(trajectory["minimum_shifted_lower_bounds"])
            ),
            "minimum_carried_unshifted_eigenvalue": float(
                np.min(trajectory["minimum_unshifted_eigenvalues"])
            ),
            "minimum_carried_retained_joint_gram_eigenvalue": (
                _minimum_joint_gram(carried)
            ),
            "minimum_raw_archive_joint_gram_eigenvalue": (
                _minimum_joint_gram(raw)
            ),
        },
        "accuracy": {
            "carried_witness": carried_metrics,
            "raw_archive": raw_metrics,
        },
        "raw_archive": {
            "method": "DOP853, offline matched-initial-state baseline",
            "function_evaluations": raw_evaluations,
        },
        "sources": sources,
    }
    np.savez_compressed(
        output_directory / "trajectory.npz",
        **trajectory,
        raw_archive_coordinates=raw,
    )
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_observables(
        output_directory / "observable_comparison.png",
        trajectory["times"],
        _observables(exact, parameters),
        _observables(raw, parameters),
        _observables(carried, parameters),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
