"""Record matched long-horizon observables for the archive 31-coordinate EOM.

This reporting driver retains sampled states from three already-defined lanes:

* cutoff-truncated exact Hamiltonian propagation;
* the unmodified archive moment closure; and
* the archive closure with the joint electron--phonon physicality correction.

The moment lanes use fixed-step RK4.  The exact lane uses DOP853 while the
Gaussian pulse is active and sparse static-Hamiltonian propagation after the
drive is numerically negligible.  Every lane is checkpointed independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import expm_multiply

from paper5.stability import (
    DimerParameters,
    closed_boson_moment_eigenvalues,
    closed_electron_eigenvalues,
    closed_scalar_rhs,
    electron_phonon_moment_matrix,
    exact_ground_closed_scalar_coordinates,
    structured_closed_state_velocity_lift,
    structured_electron_phonon_barrier_correction,
)
from paper5.stability.exact_reference import (
    _build_exact_dimer_model,
    _contract_matrix_state,
    _ground_state,
)
from paper5.stability.matrix_reference import (
    MatrixDimerState,
    closed_scalar_to_matrix_state,
    local_holstein_couplings,
    matrix_state_to_closed_scalar_coordinates,
)


LANES = ("exact", "raw", "corrected")
ENERGY_NAMES = (
    "electronic",
    "phonon",
    "electron_phonon",
    "internal_total",
    "drive",
    "instantaneous_total",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_npz_atomic(path: Path, **arrays: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_progress(output_dir: Path, payload: dict[str, Any]) -> None:
    record = {"recorded_at_utc": _utc_now(), **payload}
    encoded = json.dumps(record, sort_keys=True)
    with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(encoded, flush=True)


def _integer_ratio(numerator: float, denominator: float, name: str) -> int:
    ratio = numerator / denominator
    rounded = int(round(ratio))
    if rounded < 1 or abs(ratio - rounded) > 1e-10:
        raise ValueError(f"{name} must be a positive integer multiple")
    return rounded


def _energy_components(
    state: MatrixDimerState,
    parameters: DimerParameters,
    time_value: float,
) -> np.ndarray:
    """Return the archive Eq. (22) decomposition and external-drive energy."""

    spin_degeneracy = 2.0
    rho = np.asarray(state.electron_density, dtype=complex)
    coherent = np.asarray(state.coherent_phonon, dtype=complex)
    phonon = np.asarray(state.phonon_density, dtype=complex)
    correlation = np.asarray(state.electron_phonon_correlation, dtype=complex)
    coupling = local_holstein_couplings(parameters)
    bare_electron = np.array(
        [[0.0, -parameters.hopping], [-parameters.hopping, 0.0]],
        dtype=complex,
    )

    electronic = spin_degeneracy * np.trace(bare_electron @ rho).real
    phonon_energy = parameters.omega_ph * (
        np.vdot(coherent, coherent).real + np.trace(phonon).real
    )
    interaction_amplitude = 0.0j
    for q in range(2):
        for one in range(2):
            for two in range(2):
                interaction_amplitude += coupling[q, one, two] * (
                    coherent[q] * rho[two, one]
                    + correlation[q, two, one]
                )
    electron_phonon = 2.0 * spin_degeneracy * interaction_amplitude.real
    internal_total = electronic + phonon_energy + electron_phonon
    drive = (
        0.5
        * spin_degeneracy
        * parameters.drive_difference(time_value)
        * (rho[0, 0] - rho[1, 1]).real
    )
    return np.asarray(
        (
            electronic,
            phonon_energy,
            electron_phonon,
            internal_total,
            drive,
            internal_total + drive,
        ),
        dtype=float,
    )


def _sample_state(
    state: MatrixDimerState,
    parameters: DimerParameters,
    time_value: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    coordinates = matrix_state_to_closed_scalar_coordinates(state)
    occupation = float(state.electron_density[0, 0].real)
    energies = _energy_components(state, parameters, time_value)
    return coordinates, occupation, energies


def _matrix_diagnostics(coordinates: np.ndarray) -> np.ndarray:
    matrix_state = closed_scalar_to_matrix_state(coordinates)
    electron = closed_electron_eigenvalues(coordinates)
    return np.asarray(
        (
            float(electron[0]),
            float(electron[-1]),
            float(closed_boson_moment_eigenvalues(coordinates)[0]),
            float(np.linalg.eigvalsh(electron_phonon_moment_matrix(matrix_state))[0]),
            float(np.max(np.abs(coordinates))),
            float(np.hypot(coordinates[17], coordinates[18])),
        ),
        dtype=float,
    )


def _save_completed_lane(
    output_dir: Path,
    lane: str,
    *,
    times: list[float],
    coordinates: list[np.ndarray],
    occupations: list[float],
    energies: list[np.ndarray],
    diagnostics: list[np.ndarray],
    metadata: dict[str, Any],
) -> None:
    result_path = output_dir / f"{lane}_trajectory.npz"
    _write_npz_atomic(
        result_path,
        times=np.asarray(times, dtype=float),
        coordinates=np.asarray(coordinates, dtype=float),
        site_occupation=np.asarray(occupations, dtype=float),
        energy_components=np.asarray(energies, dtype=float),
        diagnostics=np.asarray(diagnostics, dtype=float),
        energy_names=np.asarray(ENERGY_NAMES),
        diagnostic_names=np.asarray(
            (
                "electron_minimum_eigenvalue",
                "electron_maximum_eigenvalue",
                "boson_moment_minimum_eigenvalue",
                "joint_moment_minimum_eigenvalue",
                "maximum_absolute_coordinate",
                "correlation_trace_absolute_value",
            )
        ),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    checkpoint = output_dir / f"{lane}_checkpoint.npz"
    if checkpoint.exists():
        checkpoint.unlink()


def _run_exact(
    output_dir: Path,
    parameters: DimerParameters,
    *,
    final_time: float,
    sample_step: float,
    phonon_cutoff: int,
    drive_cutoff: float,
    exact_chunk: float,
) -> None:
    lane = "exact"
    result_path = output_dir / f"{lane}_trajectory.npz"
    checkpoint_path = output_dir / f"{lane}_checkpoint.npz"
    if result_path.exists():
        _append_progress(output_dir, {"event": "lane_skipped", "lane": lane})
        return

    _integer_ratio(exact_chunk, sample_step, "exact_chunk")
    model = _build_exact_dimer_model(parameters, phonon_cutoff=phonon_cutoff)
    static_generator = -1j * model.static_hamiltonian
    trace_generator = -1j * complex(model.static_hamiltonian.diagonal().sum())

    times: list[float]
    coordinates: list[np.ndarray]
    occupations: list[float]
    energies: list[np.ndarray]
    diagnostics: list[np.ndarray]
    state_norms: list[float]
    function_evaluations: int
    if checkpoint_path.exists():
        with np.load(checkpoint_path, allow_pickle=False) as payload:
            current_time = float(payload["current_time"])
            wavefunction = np.asarray(payload["wavefunction"], dtype=complex)
            times = np.asarray(payload["times"], dtype=float).tolist()
            coordinates = [row.copy() for row in payload["coordinates"]]
            occupations = np.asarray(payload["site_occupation"], dtype=float).tolist()
            energies = [row.copy() for row in payload["energy_components"]]
            diagnostics = [row.copy() for row in payload["diagnostics"]]
            state_norms = np.asarray(payload["state_norms"], dtype=float).tolist()
            function_evaluations = int(payload["function_evaluations"])
        _append_progress(
            output_dir,
            {"event": "lane_resumed", "lane": lane, "time": current_time},
        )
    else:
        _, initial_wavefunction = _ground_state(
            model,
            eigensolver_tolerance=1e-12,
        )

        def rhs(time_value: float, wavefunction_value: np.ndarray) -> np.ndarray:
            return -1j * (
                model.static_hamiltonian @ wavefunction_value
                + parameters.drive_difference(time_value)
                * (model.drive_operator @ wavefunction_value)
            )

        driven_end = min(final_time, drive_cutoff)
        driven_count = int(round(driven_end / sample_step))
        driven_times = np.linspace(0.0, driven_end, driven_count + 1)
        solution = solve_ivp(
            rhs,
            (0.0, driven_end),
            initial_wavefunction,
            method="DOP853",
            t_eval=driven_times,
            rtol=1e-10,
            atol=1e-12,
            max_step=min(0.02, sample_step),
        )
        if not solution.success or solution.y.shape[1] != driven_times.size:
            raise RuntimeError(f"exact driven propagation failed: {solution.message}")
        times = []
        coordinates = []
        occupations = []
        energies = []
        diagnostics = []
        state_norms = []
        for index, time_value in enumerate(solution.t):
            wavefunction_value = np.asarray(solution.y[:, index], dtype=complex)
            matrix_state = _contract_matrix_state(model, wavefunction_value)
            coordinate, occupation, energy = _sample_state(
                matrix_state,
                parameters,
                float(time_value),
            )
            times.append(float(time_value))
            coordinates.append(coordinate)
            occupations.append(occupation)
            energies.append(energy)
            diagnostics.append(_matrix_diagnostics(coordinate))
            state_norms.append(float(np.vdot(wavefunction_value, wavefunction_value).real))
        current_time = driven_end
        wavefunction = np.asarray(solution.y[:, -1], dtype=complex)
        function_evaluations = int(solution.nfev)
        _append_progress(
            output_dir,
            {
                "event": "exact_driven_segment_completed",
                "lane": lane,
                "time": current_time,
                "function_evaluations": function_evaluations,
            },
        )

    wall_start = time.monotonic()
    while current_time < final_time - 0.5 * sample_step:
        chunk_end = min(current_time + exact_chunk, final_time)
        chunk_samples = int(round((chunk_end - current_time) / sample_step))
        if chunk_samples < 1:
            break
        propagated = expm_multiply(
            static_generator,
            wavefunction,
            start=0.0,
            stop=chunk_end - current_time,
            num=chunk_samples + 1,
            endpoint=True,
            traceA=trace_generator,
        )
        for offset in range(1, chunk_samples + 1):
            time_value = current_time + offset * sample_step
            wavefunction_value = np.asarray(propagated[offset], dtype=complex)
            matrix_state = _contract_matrix_state(model, wavefunction_value)
            coordinate, occupation, energy = _sample_state(
                matrix_state,
                parameters,
                time_value,
            )
            times.append(time_value)
            coordinates.append(coordinate)
            occupations.append(occupation)
            energies.append(energy)
            diagnostics.append(_matrix_diagnostics(coordinate))
            state_norms.append(float(np.vdot(wavefunction_value, wavefunction_value).real))
        current_time = chunk_end
        wavefunction = np.asarray(propagated[-1], dtype=complex)
        _write_npz_atomic(
            checkpoint_path,
            current_time=np.asarray(current_time),
            wavefunction=wavefunction,
            times=np.asarray(times, dtype=float),
            coordinates=np.asarray(coordinates, dtype=float),
            site_occupation=np.asarray(occupations, dtype=float),
            energy_components=np.asarray(energies, dtype=float),
            diagnostics=np.asarray(diagnostics, dtype=float),
            state_norms=np.asarray(state_norms, dtype=float),
            function_evaluations=np.asarray(function_evaluations),
        )
        _append_progress(
            output_dir,
            {
                "event": "exact_static_checkpoint",
                "lane": lane,
                "time": current_time,
                "completion_fraction": current_time / final_time,
                "wall_elapsed_seconds": time.monotonic() - wall_start,
            },
        )

    drive_tail_bound = (
        2.0
        * parameters.drive_amplitude
        * np.exp(-0.5 * (drive_cutoff / parameters.pulse_width) ** 2)
    )
    metadata = {
        "lane": lane,
        "status": "complete",
        "final_time": float(times[-1]),
        "sample_step": sample_step,
        "phonon_cutoff": phonon_cutoff,
        "hilbert_space_dimension": 4 * (phonon_cutoff + 1) ** 2,
        "driven_method": "DOP853",
        "static_method": "scipy.sparse.linalg.expm_multiply",
        "drive_cutoff": drive_cutoff,
        "maximum_omitted_drive_amplitude_bound": float(drive_tail_bound),
        "driven_function_evaluations": function_evaluations,
        "maximum_state_norm_defect": float(
            np.max(np.abs(np.asarray(state_norms) - 1.0))
        ),
    }
    _save_completed_lane(
        output_dir,
        lane,
        times=times,
        coordinates=coordinates,
        occupations=occupations,
        energies=energies,
        diagnostics=diagnostics,
        metadata=metadata,
    )
    _append_progress(output_dir, {"event": "lane_completed", **metadata})


def _run_moment_lane(
    output_dir: Path,
    lane: str,
    parameters: DimerParameters,
    *,
    final_time: float,
    time_step: float,
    sample_step: float,
    checkpoint_interval: float,
    phonon_cutoff: int,
    raw_failure_threshold: float,
    correction_metric: str,
) -> None:
    if lane not in ("raw", "corrected"):
        raise ValueError(f"invalid moment lane {lane!r}")
    result_path = output_dir / f"{lane}_trajectory.npz"
    checkpoint_path = output_dir / f"{lane}_checkpoint.npz"
    if result_path.exists():
        _append_progress(output_dir, {"event": "lane_skipped", "lane": lane})
        return

    sample_stride = _integer_ratio(sample_step, time_step, "sample_step")
    checkpoint_stride = _integer_ratio(
        checkpoint_interval,
        time_step,
        "checkpoint_interval",
    )
    initial_state = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=phonon_cutoff,
    )
    stats: dict[str, Any]
    if checkpoint_path.exists():
        with np.load(checkpoint_path, allow_pickle=False) as payload:
            step_index = int(payload["step_index"])
            state = np.asarray(payload["state"], dtype=float)
            times = np.asarray(payload["times"], dtype=float).tolist()
            coordinates = [row.copy() for row in payload["coordinates"]]
            occupations = np.asarray(payload["site_occupation"], dtype=float).tolist()
            energies = [row.copy() for row in payload["energy_components"]]
            diagnostics = [row.copy() for row in payload["diagnostics"]]
            stats = json.loads(str(payload["stats_json"].item()))
        _append_progress(
            output_dir,
            {
                "event": "lane_resumed",
                "lane": lane,
                "time": step_index * time_step,
            },
        )
    else:
        step_index = 0
        state = np.asarray(initial_state, dtype=float).copy()
        matrix_state = closed_scalar_to_matrix_state(state)
        coordinate, occupation, energy = _sample_state(
            matrix_state,
            parameters,
            0.0,
        )
        times = [0.0]
        coordinates = [coordinate]
        occupations = [occupation]
        energies = [energy]
        diagnostics = [_matrix_diagnostics(coordinate)]
        stats = {
            "rhs_evaluations": 0,
            "steps_completed": 0,
            "maximum_absolute_coordinate": float(np.max(np.abs(state))),
            "maximum_correction_norm": 0.0,
            "maximum_correction_frobenius_norm": 0.0,
            "sum_squared_correction_norm": 0.0,
            "sum_squared_correction_frobenius_norm": 0.0,
            "active_correction_count": 0,
            "maximum_absolute_correction_energy_flux": 0.0,
            "correction_block_maximum_norms": {
                "rho": 0.0,
                "B": 0.0,
                "N": 0.0,
                "A": 0.0,
                "C": 0.0,
            },
            "correction_block_sum_squared_norms": {
                "rho": 0.0,
                "B": 0.0,
                "N": 0.0,
                "A": 0.0,
                "C": 0.0,
            },
            "maximum_constraint_count": 0,
            "minimum_corrected_joint_barrier_eigenvalue": None,
            "nonconverged_correction_count": 0,
            "failure_time": None,
            "failure_component": None,
        }

    activation_margin = 1e-5
    barrier_rate = 5.0
    cone_tolerance = 1e-8
    maximum_constraints = 128

    def evaluate(time_value: float, current_state: np.ndarray) -> np.ndarray:
        raw = closed_scalar_rhs(time_value, current_state, parameters)
        stats["rhs_evaluations"] += 1
        if lane == "raw":
            return raw
        correction = structured_electron_phonon_barrier_correction(
            current_state,
            raw,
            parameters,
            activation_margin=activation_margin,
            barrier_rate=barrier_rate,
            energy_neutral=True,
            preserve_correlation_trace=True,
            cone_tolerance=cone_tolerance,
            maximum_constraints=maximum_constraints,
            correction_metric=correction_metric,
        )
        euclidean_norm = correction.correction_norm
        frobenius_norm = correction.lifted_frobenius_norm
        stats["maximum_correction_norm"] = max(
            stats["maximum_correction_norm"],
            euclidean_norm,
        )
        stats["maximum_correction_frobenius_norm"] = max(
            stats["maximum_correction_frobenius_norm"],
            frobenius_norm,
        )
        stats["sum_squared_correction_norm"] += euclidean_norm**2
        stats["sum_squared_correction_frobenius_norm"] += frobenius_norm**2
        if euclidean_norm > 1e-14:
            stats["active_correction_count"] += 1
        stats["maximum_absolute_correction_energy_flux"] = max(
            stats["maximum_absolute_correction_energy_flux"],
            abs(correction.correction_energy_flux),
        )
        lifted = structured_closed_state_velocity_lift(
            correction.correction_coordinates
        )
        block_norms = {
            "rho": float(np.linalg.norm(lifted.electron_density)),
            "B": float(np.linalg.norm(lifted.coherent_phonon)),
            "N": float(np.linalg.norm(lifted.phonon_density)),
            "A": float(np.linalg.norm(lifted.anomalous_phonon_density)),
            "C": float(np.linalg.norm(lifted.electron_phonon_correlation)),
        }
        for block, block_norm in block_norms.items():
            stats["correction_block_maximum_norms"][block] = max(
                stats["correction_block_maximum_norms"][block],
                block_norm,
            )
            stats["correction_block_sum_squared_norms"][block] += (
                block_norm**2
            )
        stats["maximum_constraint_count"] = max(
            stats["maximum_constraint_count"],
            correction.constraint_count,
        )
        current_minimum = correction.corrected_joint_barrier_minimum_eigenvalue
        stored_minimum = stats["minimum_corrected_joint_barrier_eigenvalue"]
        stats["minimum_corrected_joint_barrier_eigenvalue"] = (
            current_minimum
            if stored_minimum is None
            else min(stored_minimum, current_minimum)
        )
        if not correction.converged:
            stats["nonconverged_correction_count"] += 1
            raise RuntimeError(
                "joint electron-phonon correction failed: "
                f"minimum={current_minimum}"
            )
        return raw + correction.correction_coordinates

    total_steps = int(round(final_time / time_step))
    wall_start = time.monotonic()
    while step_index < total_steps:
        time_value = step_index * time_step
        previous_state = state.copy()
        k1 = evaluate(time_value, state)
        k2 = evaluate(time_value + 0.5 * time_step, state + 0.5 * time_step * k1)
        k3 = evaluate(time_value + 0.5 * time_step, state + 0.5 * time_step * k2)
        k4 = evaluate(time_value + time_step, state + time_step * k3)
        state = state + (time_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        step_index += 1
        stats["steps_completed"] = step_index
        stats["maximum_absolute_coordinate"] = max(
            stats["maximum_absolute_coordinate"],
            float(np.max(np.abs(state))),
        )
        if not np.all(np.isfinite(state)):
            raise RuntimeError(f"non-finite {lane} state at t={step_index * time_step}")

        failed = lane == "raw" and np.max(np.abs(state)) >= raw_failure_threshold
        sampled = step_index % sample_stride == 0 or failed or step_index == total_steps
        if sampled:
            sampled_time = step_index * time_step
            matrix_state = closed_scalar_to_matrix_state(state)
            coordinate, occupation, energy = _sample_state(
                matrix_state,
                parameters,
                sampled_time,
            )
            times.append(sampled_time)
            coordinates.append(coordinate)
            occupations.append(occupation)
            energies.append(energy)
            diagnostics.append(_matrix_diagnostics(coordinate))

        if failed:
            component = int(np.argmax(np.abs(state)))
            previous_value = abs(float(previous_state[component]))
            current_value = abs(float(state[component]))
            fraction = (
                (raw_failure_threshold - previous_value)
                / (current_value - previous_value)
                if current_value > previous_value
                else 1.0
            )
            stats["failure_time"] = time_value + time_step * float(
                np.clip(fraction, 0.0, 1.0)
            )
            stats["failure_component"] = component
            break

        if step_index % checkpoint_stride == 0:
            _write_npz_atomic(
                checkpoint_path,
                step_index=np.asarray(step_index),
                state=state,
                times=np.asarray(times, dtype=float),
                coordinates=np.asarray(coordinates, dtype=float),
                site_occupation=np.asarray(occupations, dtype=float),
                energy_components=np.asarray(energies, dtype=float),
                diagnostics=np.asarray(diagnostics, dtype=float),
                stats_json=np.asarray(json.dumps(stats, sort_keys=True)),
            )
            _append_progress(
                output_dir,
                {
                    "event": "moment_checkpoint",
                    "lane": lane,
                    "time": step_index * time_step,
                    "completion_fraction": step_index / total_steps,
                    "wall_elapsed_seconds": time.monotonic() - wall_start,
                    "maximum_absolute_coordinate": stats[
                        "maximum_absolute_coordinate"
                    ],
                },
            )

    metadata = {
        "lane": lane,
        "status": "failed_threshold" if stats["failure_time"] is not None else "complete",
        "final_time": float(times[-1]),
        "integration_time": step_index * time_step,
        "time_step": time_step,
        "sample_step": sample_step,
        "phonon_cutoff_for_initial_state": phonon_cutoff,
        "integration_method": "fixed-step RK4",
        "raw_failure_threshold": raw_failure_threshold,
        "controller": (
            None
            if lane == "raw"
            else {
                "activation_margin": activation_margin,
                "barrier_rate": barrier_rate,
                "cone_tolerance": cone_tolerance,
                "maximum_constraints": maximum_constraints,
                "energy_neutral": True,
                "preserve_correlation_trace": True,
                "correction_metric": correction_metric,
            }
        ),
        "stats": stats,
    }
    _save_completed_lane(
        output_dir,
        lane,
        times=times,
        coordinates=coordinates,
        occupations=occupations,
        energies=energies,
        diagnostics=diagnostics,
        metadata=metadata,
    )
    _append_progress(output_dir, {"event": "lane_completed", **metadata})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--lane",
        choices=(*LANES, "all"),
        default="all",
    )
    parser.add_argument("--final-time", type=float, default=1000.0)
    parser.add_argument("--time-step", type=float, default=0.01)
    parser.add_argument("--sample-step", type=float, default=0.2)
    parser.add_argument("--checkpoint-interval", type=float, default=10.0)
    parser.add_argument("--phonon-cutoff", type=int, default=16)
    parser.add_argument("--drive-cutoff", type=float, default=10.0)
    parser.add_argument("--exact-chunk", type=float, default=10.0)
    parser.add_argument("--raw-failure-threshold", type=float, default=1e4)
    parser.add_argument(
        "--correction-metric",
        choices=("euclidean", "frobenius"),
        default="euclidean",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.final_time <= 0.0:
        raise ValueError("final_time must be positive")
    if args.time_step <= 0.0 or args.sample_step <= 0.0:
        raise ValueError("time_step and sample_step must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    parameters = DimerParameters(
        hopping=1.0,
        gamma=0.5,
        lambda_ep=1.5,
        drive_amplitude=1.0,
        pulse_width=1.0,
    )
    repo_root = Path(__file__).resolve().parents[2]
    source_paths = (
        Path(__file__).resolve(),
        repo_root / "paper_5/src/paper5/stability/exact_reference.py",
        repo_root / "paper_5/src/paper5/stability/matrix_reference.py",
        repo_root / "paper_5/src/paper5/stability/cone_correction.py",
    )
    manifest_path = args.output_dir / "runtime_manifest.json"
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": _utc_now(),
        "classification": "diagnostic",
        "evidence_status": "exploratory_local_not_promoted",
        "parameters": {
            "hopping": parameters.hopping,
            "gamma": parameters.gamma,
            "lambda_ep": parameters.lambda_ep,
            "drive_amplitude": parameters.drive_amplitude,
            "pulse_width": parameters.pulse_width,
            "phonon_cutoff": args.phonon_cutoff,
        },
        "integration": {
            "final_time": args.final_time,
            "time_step": args.time_step,
            "sample_step": args.sample_step,
            "checkpoint_interval": args.checkpoint_interval,
            "drive_cutoff": args.drive_cutoff,
            "exact_chunk": args.exact_chunk,
            "correction_metric": args.correction_metric,
        },
        "source_hashes": {
            str(path.relative_to(repo_root)): _sha256(path) for path in source_paths
        },
    }
    _write_json_atomic(manifest_path, manifest)

    lanes = LANES if args.lane == "all" else (args.lane,)
    try:
        for lane in lanes:
            _append_progress(
                args.output_dir,
                {"event": "lane_started", "lane": lane},
            )
            if lane == "exact":
                _run_exact(
                    args.output_dir,
                    parameters,
                    final_time=args.final_time,
                    sample_step=args.sample_step,
                    phonon_cutoff=args.phonon_cutoff,
                    drive_cutoff=args.drive_cutoff,
                    exact_chunk=args.exact_chunk,
                )
            else:
                _run_moment_lane(
                    args.output_dir,
                    lane,
                    parameters,
                    final_time=args.final_time,
                    time_step=args.time_step,
                    sample_step=args.sample_step,
                    checkpoint_interval=args.checkpoint_interval,
                    phonon_cutoff=args.phonon_cutoff,
                    raw_failure_threshold=args.raw_failure_threshold,
                    correction_metric=args.correction_metric,
                )
    except KeyboardInterrupt:
        manifest.update({"status": "interrupted", "finished_at_utc": _utc_now()})
        _write_json_atomic(manifest_path, manifest)
        return 130
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "finished_at_utc": _utc_now(),
                "error": repr(error),
            }
        )
        _write_json_atomic(manifest_path, manifest)
        raise

    manifest.update({"status": "complete", "finished_at_utc": _utc_now()})
    _write_json_atomic(manifest_path, manifest)
    _append_progress(args.output_dir, {"event": "run_completed"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
