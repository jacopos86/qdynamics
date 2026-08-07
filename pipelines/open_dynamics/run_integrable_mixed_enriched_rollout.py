#!/usr/bin/env python3
"""Run the first autonomous native-plus-integrable-mixed packet rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from paper5.stability.conditional_packets import (
    electron_relative_product_to_local_state,
    electron_relative_state,
)
from paper5.stability.exact_reference import (
    _build_exact_dimer_model,
    _contract_matrix_state,
)
from paper5.stability.hubbard_dimer import DimerParameters, GaussianSineDrive
from paper5.stability.matrix_reference import (
    closed_scalar_to_matrix_state,
    matrix_state_to_closed_scalar_coordinates,
)
from paper5.stability.mixed_enriched_propagation import (
    mixed_enriched_euler_step,
    mixed_enriched_midpoint_step,
    normalized_packet_state,
)
from pipelines.open_dynamics.run_archive_long_horizon_observables import (
    _energy_components,
)


RUN_ID = "paper_v_integrable_mixed_enriched_cutoff16_t4_dt001_20260805_v1"
DEFAULT_PARENT = Path(
    "output/local_runs/"
    "paper_v_multi_coherent_double_pulse_blind_model_cutoff16_20260804_v1/"
    "fine_central"
)
DEFAULT_EXACT = Path(
    "output/local_runs/"
    "paper_v_multi_coherent_double_pulse_sealed_score_cutoff16_20260804_v1/"
    "score_arrays.npz"
)
DEFAULT_OUTPUT = Path("output/local_runs") / RUN_ID
OBSERVABLE_NAMES = (
    "site_0_occupation",
    "site_1_occupation",
    "electronic_energy",
    "phonon_energy",
    "electron_phonon_energy",
    "internal_total_energy",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _integer_ratio(numerator: float, denominator: float, name: str) -> int:
    value = numerator / denominator
    rounded = int(round(value))
    if rounded < 1 or not np.isclose(value, rounded, atol=1e-12, rtol=0.0):
        raise ValueError(f"{name} must be a positive integer multiple")
    return rounded


def _matched_indices(source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    indices = np.asarray(
        [int(np.argmin(np.abs(source_times - value))) for value in target_times],
        dtype=int,
    )
    if not np.allclose(
        source_times[indices],
        target_times,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("stored trajectory does not contain requested samples")
    return indices


def _observables(
    times: np.ndarray,
    coordinates: np.ndarray,
    parameters: DimerParameters,
) -> np.ndarray:
    result = np.empty((times.size, len(OBSERVABLE_NAMES)), dtype=float)
    for index, (time_value, row) in enumerate(
        zip(times, coordinates, strict=True)
    ):
        state = closed_scalar_to_matrix_state(row)
        energy = _energy_components(state, parameters, float(time_value))
        result[index] = (
            2.0 * state.electron_density[0, 0].real,
            2.0 * state.electron_density[1, 1].real,
            energy[0],
            energy[1],
            energy[2],
            energy[3],
        )
    return result


def _time_rms(times: np.ndarray, values: np.ndarray) -> float:
    if times.size < 2:
        return float(np.sqrt(np.mean(np.asarray(values) ** 2)))
    return float(
        np.sqrt(np.trapezoid(np.asarray(values) ** 2, times) / times[-1])
    )


def _plot(
    path: Path,
    times: np.ndarray,
    exact: np.ndarray,
    parent: np.ndarray,
    enriched: np.ndarray,
) -> None:
    titles = (
        r"site 0 occupation, $n_0=2\rho_{00}$",
        r"site 1 occupation, $n_1=2\rho_{11}$",
        "electronic energy",
        "phonon energy",
        "electron-phonon energy",
        "total internal energy",
    )
    figure, axes = plt.subplots(3, 2, figsize=(8.2, 9.0), sharex=True)
    for index, axis in enumerate(axes.flat):
        axis.plot(times, exact[:, index], color="#171717", label="exact")
        axis.plot(
            times,
            parent[:, index],
            color="#4c78a8",
            linestyle="--",
            label="native packet",
        )
        axis.plot(
            times,
            enriched[:, index],
            color="#d62728",
            label="integrable mixed enrichment",
        )
        axis.set_title(titles[index])
        axis.grid(alpha=0.22)
        axis.set_xlabel(r"$t\,t_{\rm hop}$")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=190)
    plt.close(figure)


def run(
    output_directory: Path,
    *,
    parent_directory: Path = DEFAULT_PARENT,
    exact_arrays_path: Path = DEFAULT_EXACT,
    final_time: float = 4.0,
    time_step: float = 0.01,
    sample_step: float = 0.05,
    retraction_relative_tolerance: float = 1e-9,
    relative_damping: float | None = None,
    integrator: str = "midpoint",
) -> dict[str, object]:
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty directory {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    output_stride = _integer_ratio(sample_step, time_step, "sample_step")
    step_count = _integer_ratio(final_time, time_step, "final_time")
    sample_count = step_count // output_stride + 1
    if (sample_count - 1) * output_stride != step_count:
        raise ValueError("final_time must be a multiple of sample_step")
    step_functions = {
        "euler": mixed_enriched_euler_step,
        "midpoint": mixed_enriched_midpoint_step,
    }
    if integrator not in step_functions:
        raise ValueError("integrator must be 'euler' or 'midpoint'")
    step_function = step_functions[integrator]
    method_name = f"{integrator}_native_plus_integrable_mixed_retraction"

    parent_summary_path = parent_directory / "summary.json"
    parent_arrays_path = parent_directory / "segmented_horizon.npz"
    parent_summary = json.loads(
        parent_summary_path.read_text(encoding="utf-8")
    )
    settings = parent_summary["parameters"]
    cutoff = int(settings["phonon_cutoff"])
    relative_dimension = 2 * cutoff + 1
    parameters = DimerParameters(
        hopping=float(settings["hopping"]),
        gamma=float(settings["gamma"]),
        lambda_ep=float(settings["lambda_ep"]),
        drive_amplitude=float(settings["drive_amplitude"]),
        pulse_width=float(settings["pulse_width"]),
    )
    drive_data = settings["drive_protocol"]
    drive = GaussianSineDrive(
        amplitude=float(drive_data["amplitude"]),
        pulse_width=float(drive_data["pulse_width"]),
        delays=tuple(float(value) for value in drive_data["delays"]),
    )
    with np.load(parent_arrays_path, allow_pickle=False) as arrays:
        parent_times_all = np.asarray(arrays["times"], dtype=float)
        parent_parameters_all = np.asarray(
            arrays["parameter_trajectory"],
            dtype=float,
        )
        parent_counts_all = np.asarray(
            arrays["packet_count_trajectory"],
            dtype=int,
        )
        parent_closed_all = np.asarray(arrays["closed_coordinates"], dtype=float)
    times = np.linspace(0.0, final_time, sample_count)
    parent_indices = _matched_indices(parent_times_all, times)
    initial_count = int(parent_counts_all[0])
    packet_parameters = parent_parameters_all[0, : 16 * initial_count].copy()

    with np.load(exact_arrays_path, allow_pickle=False) as arrays:
        exact_times_all = np.asarray(arrays["times"], dtype=float)
        exact_closed_all = np.asarray(arrays["exact_dop853_closed"], dtype=float)[0]
        exact_states_all = np.asarray(
            arrays["exact_dop853_state_vectors"],
            dtype=complex,
        )[0]
        coordinate_scales = np.asarray(arrays["coordinate_scales"], dtype=float)
    exact_indices = _matched_indices(exact_times_all, times)
    exact_closed = exact_closed_all[exact_indices]
    exact_states = exact_states_all[exact_indices]
    center_state = electron_relative_state(
        exact_states[0],
        phonon_cutoff=cutoff,
    ).center_state
    exact_model = _build_exact_dimer_model(parameters, phonon_cutoff=cutoff)

    plan: dict[str, object] = {
        "schema": "paper_v_integrable_mixed_enriched_plan_v1",
        "run_id": output_directory.name,
        "classification": "autonomous_exploratory_local_not_promoted",
        "method": method_name,
        "final_time": final_time,
        "time_step": time_step,
        "sample_step": sample_step,
        "step_count": step_count,
        "retraction_relative_tolerance": retraction_relative_tolerance,
        "relative_damping": relative_damping,
        "exact_reference_used_by_online_step": False,
    }
    _write_json(output_directory / "plan.json", plan)

    sampled_parameters: list[np.ndarray] = [packet_parameters.copy()]
    sampled_counts = [initial_count]
    enriched_closed = np.empty((sample_count, 31), dtype=float)
    enriched_fidelity = np.empty(sample_count)
    retained_local_norm = np.empty(sample_count)
    step_packet_count = np.empty(step_count, dtype=int)
    step_retraction_error = np.empty(step_count)
    step_native_residual = np.empty(step_count)
    step_enriched_residual = np.empty(step_count)
    step_mixed_speed = np.empty(step_count)
    step_native_speed = np.empty(step_count)

    def sample(index: int, values: np.ndarray) -> None:
        relative_state = normalized_packet_state(
            values,
            relative_dimension=relative_dimension,
        )
        embedded = electron_relative_product_to_local_state(
            relative_state,
            center_state,
            phonon_cutoff=cutoff,
        )
        matrix_state = _contract_matrix_state(exact_model, embedded.state)
        enriched_closed[index] = matrix_state_to_closed_scalar_coordinates(
            matrix_state
        )
        enriched_fidelity[index] = float(
            abs(np.vdot(exact_states[index], embedded.state)) ** 2
        )
        retained_local_norm[index] = embedded.retained_norm

    sample(0, packet_parameters)
    started = time.time()
    output_index = 1
    progress_path = output_directory / "progress.jsonl"
    for step_index in range(step_count):
        time_value = step_index * time_step
        step = step_function(
            time_value,
            packet_parameters,
            time_step,
            parameters,
            relative_dimension=relative_dimension,
            drive_protocol=drive,
            relative_damping=relative_damping,
            retraction_relative_tolerance=retraction_relative_tolerance,
        )
        packet_parameters = step.parameters
        step_packet_count[step_index] = step.packet_count
        step_retraction_error[step_index] = step.retraction_state_error
        step_native_residual[step_index] = step.native_relative_residual
        step_enriched_residual[step_index] = step.enriched_relative_residual
        step_mixed_speed[step_index] = step.mixed_coordinate_speed
        step_native_speed[step_index] = step.native_parameter_speed
        if (step_index + 1) % output_stride == 0:
            sample(output_index, packet_parameters)
            sampled_parameters.append(packet_parameters.copy())
            sampled_counts.append(step.packet_count)
            output_index += 1
        if (step_index + 1) % max(1, 50) == 0 or step_index + 1 == step_count:
            progress = {
                "step": step_index + 1,
                "time": (step_index + 1) * time_step,
                "packet_count": step.packet_count,
                "maximum_packet_count": int(np.max(step_packet_count[: step_index + 1])),
                "maximum_retraction_error": float(
                    np.max(step_retraction_error[: step_index + 1])
                ),
                "elapsed_seconds": time.time() - started,
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(progress, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            print(json.dumps(progress, sort_keys=True), flush=True)

    maximum_count = max(sampled_counts)
    padded_parameters = np.zeros((sample_count, 16 * maximum_count))
    for index, values in enumerate(sampled_parameters):
        padded_parameters[index, : values.size] = values
    parent_closed = parent_closed_all[parent_indices]
    scaled_parent_error = (parent_closed - exact_closed) / coordinate_scales
    scaled_enriched_error = (enriched_closed - exact_closed) / coordinate_scales
    parent_observables = _observables(times, parent_closed, parameters)
    enriched_observables = _observables(times, enriched_closed, parameters)
    exact_observables = _observables(times, exact_closed, parameters)
    parent_observable_rms = {
        name: _time_rms(times, parent_observables[:, index] - exact_observables[:, index])
        for index, name in enumerate(OBSERVABLE_NAMES)
    }
    enriched_observable_rms = {
        name: _time_rms(times, enriched_observables[:, index] - exact_observables[:, index])
        for index, name in enumerate(OBSERVABLE_NAMES)
    }
    summary: dict[str, object] = {
        "schema": "paper_v_integrable_mixed_enriched_summary_v1",
        "run_id": output_directory.name,
        "classification": "autonomous_exploratory_local_not_promoted",
        "status": "complete",
        "parameters": {
            "lambda_ep": parameters.lambda_ep,
            "gamma": parameters.gamma,
            "coupling": parameters.coupling,
            "drive_protocol": drive_data,
            "phonon_cutoff": cutoff,
            "relative_dimension": relative_dimension,
            "final_time": final_time,
            "time_step": time_step,
            "sample_step": sample_step,
            "retraction_relative_tolerance": retraction_relative_tolerance,
            "relative_damping": relative_damping,
        },
        "integration": {
            "method": method_name,
            "step_count": step_count,
            "initial_packets_per_branch": initial_count,
            "final_packets_per_branch": int(step_packet_count[-1]),
            "maximum_packets_per_branch": int(np.max(step_packet_count)),
            "maximum_retraction_state_error": float(
                np.max(step_retraction_error)
            ),
            "maximum_native_parameter_speed": float(
                np.max(step_native_speed)
            ),
            "maximum_mixed_coordinate_speed": float(
                np.max(step_mixed_speed)
            ),
            "elapsed_seconds": time.time() - started,
            "online_exact_reference_used": False,
        },
        "comparison": {
            "parent_all31_scaled_rms": float(
                np.sqrt(np.mean(scaled_parent_error**2))
            ),
            "enriched_all31_scaled_rms": float(
                np.sqrt(np.mean(scaled_enriched_error**2))
            ),
            "parent_c_scaled_rms": float(
                np.sqrt(np.mean(scaled_parent_error[:, 17:31] ** 2))
            ),
            "enriched_c_scaled_rms": float(
                np.sqrt(np.mean(scaled_enriched_error[:, 17:31] ** 2))
            ),
            "minimum_enriched_exact_state_fidelity": float(
                np.min(enriched_fidelity)
            ),
            "final_enriched_exact_state_fidelity": float(
                enriched_fidelity[-1]
            ),
            "minimum_local_embedding_retained_norm": float(
                np.min(retained_local_norm)
            ),
            "parent_observable_time_rms": parent_observable_rms,
            "enriched_observable_time_rms": enriched_observable_rms,
        },
        "tangent": {
            "native_relative_residual_rms": float(
                np.sqrt(np.mean(step_native_residual**2))
            ),
            "enriched_relative_residual_rms": float(
                np.sqrt(np.mean(step_enriched_residual**2))
            ),
        },
        "interpretation": (
            "This is an autonomous construction pilot. Exact cutoff-16 data "
            "enter only after propagation for scoring. Matched step halving "
            "is required before attributing accuracy changes to the mixed "
            "layer rather than the manifold integrator."
        ),
    }
    arrays_path = output_directory / "mixed_enriched_rollout.npz"
    np.savez_compressed(
        arrays_path,
        times=times,
        packet_count=np.asarray(sampled_counts, dtype=int),
        parameter_trajectory=padded_parameters,
        exact_closed_coordinates=exact_closed,
        parent_closed_coordinates=parent_closed,
        enriched_closed_coordinates=enriched_closed,
        coordinate_scales=coordinate_scales,
        exact_observables=exact_observables,
        parent_observables=parent_observables,
        enriched_observables=enriched_observables,
        enriched_exact_state_fidelity=enriched_fidelity,
        local_embedding_retained_norm=retained_local_norm,
        step_packet_count=step_packet_count,
        step_retraction_state_error=step_retraction_error,
        step_native_relative_residual=step_native_residual,
        step_enriched_relative_residual=step_enriched_residual,
        step_native_parameter_speed=step_native_speed,
        step_mixed_coordinate_speed=step_mixed_speed,
    )
    plot_path = output_directory / "observable_comparison.png"
    _plot(
        plot_path,
        times,
        exact_observables,
        parent_observables,
        enriched_observables,
    )
    summary_path = output_directory / "summary.json"
    _write_json(summary_path, summary)
    source_files = (
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[2]
        / "paper_5/src/paper5/stability/mixed_enriched_propagation.py",
        Path(__file__).resolve().parents[2]
        / "paper_5/src/paper5/stability/mixed_exponential_layer.py",
    )
    artifacts = (
        output_directory / "plan.json",
        progress_path,
        arrays_path,
        plot_path,
        summary_path,
    )
    manifest = {
        "schema": "paper_v_integrable_mixed_enriched_manifest_v1",
        "status": "complete",
        "python": sys.version,
        "platform": platform.platform(),
        "input_hashes": {
            str(path): _sha256(path)
            for path in (
                parent_summary_path,
                parent_arrays_path,
                exact_arrays_path,
            )
        },
        "source_hashes": {
            str(path): _sha256(path) for path in source_files
        },
        "artifact_hashes": {
            path.name: _sha256(path) for path in artifacts
        },
    }
    _write_json(output_directory / "runtime_manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parent-directory", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--exact-arrays", type=Path, default=DEFAULT_EXACT)
    parser.add_argument("--final-time", type=float, default=4.0)
    parser.add_argument("--time-step", type=float, default=0.01)
    parser.add_argument("--sample-step", type=float, default=0.05)
    parser.add_argument("--retraction-relative-tolerance", type=float, default=1e-9)
    parser.add_argument(
        "--integrator",
        choices=("euler", "midpoint"),
        default="midpoint",
    )
    parser.add_argument(
        "--relative-damping",
        type=float,
        default=0.0,
        help="zero selects the supported geometric pseudoinverse",
    )
    arguments = parser.parse_args()
    run(
        arguments.output_directory,
        parent_directory=arguments.parent_directory,
        exact_arrays_path=arguments.exact_arrays,
        final_time=arguments.final_time,
        time_step=arguments.time_step,
        sample_step=arguments.sample_step,
        retraction_relative_tolerance=(
            arguments.retraction_relative_tolerance
        ),
        relative_damping=(
            None if arguments.relative_damping == 0.0 else arguments.relative_damping
        ),
        integrator=arguments.integrator,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
