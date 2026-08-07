#!/usr/bin/env python3
"""Run the first autonomous fixed-frame reciprocal archive-memory pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy.integrate import solve_ivp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from paper5.stability.archive_auxiliary_memory import (
    build_archive_auxiliary_frame_from_observables,
    propagate_archive_auxiliary_rk4,
)
from paper5.stability.exact_reference import (
    _build_exact_dimer_model,
    _contract_matrix_state,
    _ground_state,
)
from paper5.stability.hubbard_dimer import DimerParameters, GaussianSineDrive
from paper5.stability.matrix_reference import (
    boson_moment_matrix,
    closed_scalar_to_matrix_state,
    electron_phonon_moment_matrix,
    matrix_state_to_closed_scalar_coordinates,
    matrix_total_energy,
    pauli_repaired_closed_scalar_rhs,
)
from paper5.stability.multi_coherent_scores import (
    CLOSED_COORDINATE_BLOCKS,
    closed_coordinate_error_scores,
    development_coordinate_scales,
)
from paper5.stability.reachability_observability import (
    build_drive_aware_word_envelope,
)

RUN_ID = "paper_v_archive_auxiliary_autonomous_cutoff16_t4_20260804_v2"
DEFAULT_EXACT = Path(
    "output/local_runs/"
    "paper_v_exact_vs_31d_cutoff_convergence_t20_local_20260801_v1/"
    "trajectories_cutoff_16.npz"
)
DEFAULT_OUTPUT = Path("output/local_runs") / RUN_ID


class _FixedDriveParameters:
    def __init__(self, parameters: DimerParameters, drive_value: float) -> None:
        self._parameters = parameters
        self._drive_value = float(drive_value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._parameters, name)

    def drive_difference(self, time: float) -> float:
        del time
        return self._drive_value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _drive_rate(protocol: GaussianSineDrive, time_value: float) -> float:
    result = 0.0
    for delay in protocol.delays:
        local_time = time_value - delay
        if local_time < 0.0:
            continue
        phase = np.pi * local_time / 4.0
        envelope = np.exp(
            -0.5 * (local_time / protocol.pulse_width) ** 2
        )
        result += 2.0 * protocol.amplitude * envelope * (
            (np.pi / 4.0) * np.cos(phase)
            - (local_time / protocol.pulse_width**2) * np.sin(phase)
        )
    return float(result)


def _physical_diagnostics(
    coordinates: np.ndarray,
    parameters: DimerParameters,
) -> dict[str, np.ndarray]:
    electron_minimum = []
    boson_minimum = []
    joint_minimum = []
    energy = []
    for row in coordinates:
        state = closed_scalar_to_matrix_state(row)
        electron_minimum.append(
            float(np.linalg.eigvalsh(state.electron_density)[0])
        )
        boson_minimum.append(
            float(np.linalg.eigvalsh(boson_moment_matrix(state))[0])
        )
        joint_minimum.append(
            float(np.linalg.eigvalsh(electron_phonon_moment_matrix(state))[0])
        )
        energy.append(matrix_total_energy(state, parameters))
    return {
        "electron_minimum": np.asarray(electron_minimum),
        "boson_minimum": np.asarray(boson_minimum),
        "joint_minimum": np.asarray(joint_minimum),
        "energy": np.asarray(energy),
    }


def _score(
    times: np.ndarray,
    trajectory: np.ndarray,
    exact: np.ndarray,
    scales: np.ndarray,
    diagnostics: dict[str, np.ndarray],
    exact_diagnostics: dict[str, np.ndarray],
) -> dict[str, object]:
    difference = trajectory - exact
    block_scores = closed_coordinate_error_scores(
        times,
        trajectory,
        exact,
        scales,
        interval=(float(times[0]), float(times[-1])),
    )
    return {
        "coordinate_rms_error": float(np.sqrt(np.mean(difference**2))),
        "coordinate_maximum_error": float(np.max(np.abs(difference))),
        "electron_trace_distance_maximum": (
            block_scores.electron_trace_distance_maximum
        ),
        "scaled_block_rms": block_scores.block_rms,
        "scaled_block_maximum": block_scores.block_maximum,
        "minimum_electron_eigenvalue": float(
            np.min(diagnostics["electron_minimum"])
        ),
        "minimum_boson_moment_eigenvalue": float(
            np.min(diagnostics["boson_minimum"])
        ),
        "minimum_joint_gram_eigenvalue": float(
            np.min(diagnostics["joint_minimum"])
        ),
        "total_energy_rms_error": float(
            np.sqrt(
                np.mean(
                    (diagnostics["energy"] - exact_diagnostics["energy"]) ** 2
                )
            )
        ),
    }


def _make_plot(
    output_path: Path,
    times: np.ndarray,
    trajectories: dict[str, np.ndarray],
    diagnostics: dict[str, dict[str, np.ndarray]],
) -> None:
    colors = {
        "exact": "#111111",
        "archive": "#999999",
        "pauli_archive": "#e68310",
        "word_depth_0": "#3969ac",
        "word_depth_1": "#7f3c8d",
        "word_depth_2": "#11a579",
        "word_depth_3": "#e73f74",
    }
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    exact = trajectories["exact"]
    for name, trajectory in trajectories.items():
        axes[0, 0].plot(
            times,
            0.5 * (1.0 + trajectory[:, 0]),
            label=name.replace("_", " "),
            color=colors.get(name),
            linewidth=2.0 if name == "exact" else 1.2,
        )
        if name != "exact":
            axes[0, 1].semilogy(
                times,
                np.maximum(
                    np.sqrt(np.mean((trajectory - exact) ** 2, axis=1)),
                    1e-15,
                ),
                label=name.replace("_", " "),
                color=colors.get(name),
            )
        axes[1, 0].plot(
            times,
            diagnostics[name]["energy"],
            color=colors.get(name),
            linewidth=2.0 if name == "exact" else 1.2,
        )
        axes[1, 1].plot(
            times,
            diagnostics[name]["joint_minimum"],
            color=colors.get(name),
            linewidth=2.0 if name == "exact" else 1.2,
        )
    axes[0, 0].set_title("site-0 occupation")
    axes[0, 1].set_title("31-coordinate RMS error")
    axes[1, 0].set_title("internal energy")
    axes[1, 1].set_title("minimum joint-Gram eigenvalue")
    for axis in axes.reshape(-1):
        axis.set_xlabel("time")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[0, 1].legend(fontsize=7, ncol=2)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run(
    exact_path: Path,
    output_directory: Path,
    *,
    final_time: float = 4.0,
    time_step: float = 0.01,
    maximum_word_depth: int = 3,
    rank_tolerance: float = 1e-10,
) -> dict[str, object]:
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty output directory {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    parameters = DimerParameters(
        hopping=1.0,
        gamma=0.5,
        lambda_ep=1.5,
        drive_amplitude=1.0,
        pulse_width=1.0,
    )
    protocol = GaussianSineDrive.from_parameters(parameters)
    plan: dict[str, object] = {
        "schema": "paper_v_archive_auxiliary_autonomous_plan_v2",
        "run_id": output_directory.name,
        "classification": "autonomous_development_pilot",
        "evidence_status": "exploratory_local_not_promoted",
        "exact_reference": str(exact_path),
        "phonon_cutoff": 16,
        "lambda_ep": parameters.lambda_ep,
        "gamma": parameters.gamma,
        "drive_protocol": {
            "amplitude": protocol.amplitude,
            "pulse_width": protocol.pulse_width,
            "delays": list(protocol.delays),
        },
        "final_time": final_time,
        "time_step": time_step,
        "maximum_word_depth": maximum_word_depth,
        "rank_tolerance": rank_tolerance,
        "model_online_inputs": ["retained_coordinates", "memory", "V", "Vdot"],
        "preparation_residual_density_seeded": True,
        "online_exact_reference_used": False,
        "representability_controller_used": False,
    }
    _write_json_atomic(output_directory / "plan.json", plan)
    started = time.time()

    reference_payload = np.load(exact_path, allow_pickle=False)
    all_times = np.asarray(reference_payload["times"], dtype=float)
    selected = all_times <= final_time + 1e-12
    times = all_times[selected]
    exact = np.asarray(reference_payload["exact_coordinates"], dtype=float)[selected]
    archive = np.asarray(
        reference_payload["closed_coordinates__archive"],
        dtype=float,
    )[selected]
    sample_step = float(times[1] - times[0])
    if not np.isclose(times[-1], final_time, atol=1e-12):
        raise ValueError("exact artifact does not sample the requested final time")

    model = _build_exact_dimer_model(parameters, phonon_cutoff=16)
    _, ground_state = _ground_state(model, eigensolver_tolerance=1e-12)
    initial_closed = matrix_state_to_closed_scalar_coordinates(
        _contract_matrix_state(model, ground_state)
    )
    if np.max(np.abs(initial_closed - exact[0])) > 2e-10:
        raise RuntimeError("current ground-state contraction differs from exact artifact")

    def archive_field(state: np.ndarray, drive_value: float) -> np.ndarray:
        return pauli_repaired_closed_scalar_rhs(
            0.0,
            state,
            _FixedDriveParameters(parameters, drive_value),  # type: ignore[arg-type]
        )

    pauli_solution = solve_ivp(
        lambda time_value, state: archive_field(
            state,
            protocol.difference(float(time_value)),
        ),
        (0.0, final_time),
        initial_closed,
        method="DOP853",
        t_eval=times,
        rtol=1e-10,
        atol=1e-12,
        max_step=time_step,
    )
    if not pauli_solution.success:
        raise RuntimeError(pauli_solution.message)
    pauli_archive = np.asarray(pauli_solution.y.T, dtype=float)

    envelope = build_drive_aware_word_envelope(
        parameters,
        phonon_cutoff=16,
        maximum_word_depth=maximum_word_depth,
        rank_tolerance=rank_tolerance,
        preparation_state_vectors=(ground_state,),
    )
    full_frame = build_archive_auxiliary_frame_from_observables(
        envelope.construction,
        envelope.hidden_observables,
    )
    trajectories: dict[str, np.ndarray] = {
        "exact": exact,
        "archive": archive,
        "pauli_archive": pauli_archive,
    }
    auxiliary_diagnostics: dict[str, object] = {}
    for depth, hidden_dimension in enumerate(envelope.cumulative_dimensions):
        frame = full_frame.prefix(hidden_dimension)
        initial = frame.initialize_memory(
            initial_closed,
            ground_state,
            archive_field,
            drive_value=protocol.difference(0.0),
            relative_tolerance=rank_tolerance,
        )
        trajectory = propagate_archive_auxiliary_rk4(
            frame,
            initial,
            archive_field,
            protocol.difference,
            lambda time_value: _drive_rate(protocol, time_value),
            final_time=final_time,
            time_step=time_step,
            sample_step=sample_step,
            relative_tolerance=rank_tolerance,
            directional_step=3e-6,
        )
        name = f"word_depth_{depth}"
        trajectories[name] = trajectory.closed_coordinates
        auxiliary_diagnostics[name] = {
            "hidden_dimension": hidden_dimension,
            "maximum_section_relative_residual": float(
                np.max(trajectory.centered_section_relative_residuals)
            ),
            "maximum_lossless_identity_residual": float(
                np.max(trajectory.projected_norm_identity_residuals)
            ),
            "maximum_physical_hidden_norm": float(
                np.max(trajectory.physical_hidden_norms)
            ),
        }
        print(f"completed {name} with r={hidden_dimension}", flush=True)

    scales = development_coordinate_scales(exact, phonon_cutoff=16)
    diagnostics = {
        name: _physical_diagnostics(trajectory, parameters)
        for name, trajectory in trajectories.items()
    }
    scores = {
        name: _score(
            times,
            trajectory,
            exact,
            scales,
            diagnostics[name],
            diagnostics["exact"],
        )
        for name, trajectory in trajectories.items()
        if name != "exact"
    }
    _make_plot(
        output_directory / "archive_auxiliary_autonomous_pilot.png",
        times,
        trajectories,
        diagnostics,
    )
    np.savez_compressed(
        output_directory / "archive_auxiliary_autonomous_pilot.npz",
        times=times,
        coordinate_scales=scales,
        **{f"coordinates__{name}": value for name, value in trajectories.items()},
        **{
            f"joint_minimum__{name}": value["joint_minimum"]
            for name, value in diagnostics.items()
        },
        **{
            f"energy__{name}": value["energy"]
            for name, value in diagnostics.items()
        },
    )
    best_name = min(
        (name for name in scores if name.startswith("word_depth_")),
        key=lambda name: float(scores[name]["coordinate_rms_error"]),
    )
    summary: dict[str, object] = {
        "schema": "paper_v_archive_auxiliary_autonomous_summary_v2",
        "run_id": output_directory.name,
        "classification": "autonomous_development_pilot",
        "evidence_status": "exploratory_local_not_promoted",
        "status": "complete",
        "word_layer_dimensions": list(envelope.layer_dimensions),
        "word_cumulative_dimensions": list(envelope.cumulative_dimensions),
        "scores": scores,
        "auxiliary_diagnostics": auxiliary_diagnostics,
        "best_auxiliary_model": best_name,
        "interpretation": (
            "This is an autonomous preparation-seeded fixed-union pilot. The "
            "initial unresolved density and its component-Liouvillian words "
            "are retained, but the frame has not yet undergone finite-horizon "
            "reachable-observable selection or state adaptation."
        ),
        "online_exact_reference_used": False,
        "representability_controller_used": False,
        "elapsed_seconds": time.time() - started,
    }
    _write_json_atomic(output_directory / "summary.json", summary)

    repo_root = Path(__file__).resolve().parents[2]
    source_paths = (
        Path(__file__).resolve(),
        repo_root / "paper_5/src/paper5/stability/archive_auxiliary_memory.py",
        repo_root / "paper_5/src/paper5/stability/reachability_observability.py",
    )
    artifact_paths = (
        output_directory / "plan.json",
        output_directory / "summary.json",
        output_directory / "archive_auxiliary_autonomous_pilot.npz",
        output_directory / "archive_auxiliary_autonomous_pilot.png",
    )
    manifest: dict[str, object] = {
        "schema": "paper_v_archive_auxiliary_autonomous_manifest_v2",
        "run_id": output_directory.name,
        "status": "complete",
        "classification": "autonomous_development_pilot",
        "evidence_status": "exploratory_local_not_promoted",
        "python": sys.version,
        "platform": platform.platform(),
        "exact_reference_hash": _sha256(exact_path),
        "source_hashes": {
            str(path.relative_to(repo_root)): _sha256(path)
            for path in source_paths
        },
        "artifact_hashes": {
            path.name: _sha256(path) for path in artifact_paths
        },
    }
    _write_json_atomic(output_directory / "runtime_manifest.json", manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-reference", type=Path, default=DEFAULT_EXACT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--final-time", type=float, default=4.0)
    parser.add_argument("--time-step", type=float, default=0.01)
    parser.add_argument("--maximum-word-depth", type=int, default=3)
    parser.add_argument("--rank-tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    summary = run(
        args.exact_reference,
        args.output_directory,
        final_time=args.final_time,
        time_step=args.time_step,
        maximum_word_depth=args.maximum_word_depth,
        rank_tolerance=args.rank_tolerance,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
