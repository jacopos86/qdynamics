#!/usr/bin/env python3
"""Score a complete partial adaptive-packet checkpoint without propagating it."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

from paper5.stability.conditional_packets import (
    electron_relative_product_to_local_state,
    electron_relative_state,
)
from paper5.stability.exact_reference import _build_exact_dimer_model
from paper5.stability.hubbard_dimer import DimerParameters, GaussianSineDrive
from paper5.stability.matrix_reference import (
    boson_moment_matrix,
    closed_scalar_to_matrix_state,
    electron_phonon_moment_matrix,
)
from paper5.stability.mixed_enriched_propagation import normalized_packet_state
from paper5.stability.moment_hierarchy import moment_hierarchy
from paper5.stability.multi_coherent import (
    multi_coherent_state,
    relative_state_closed_coordinates,
)
from paper5.stability.multi_coherent_scores import CLOSED_COORDINATE_BLOCKS
from pipelines.open_dynamics.run_mixed_guided_packet_rollout import (
    DEFAULT_EXACT,
    DEFAULT_PARENT,
    _matched_indices,
    _observables,
    _plot,
    _sha256,
    _write_json,
)


def _time_rms(
    times: np.ndarray,
    values: np.ndarray,
    *,
    start: float,
    stop: float,
) -> float:
    selected = (times >= start - 1e-12) & (times <= stop + 1e-12)
    local_times = times[selected]
    return float(
        np.sqrt(np.trapezoid(values[selected] ** 2, local_times) / (stop - start))
    )


def finalize(
    partial_directory: Path,
    output_directory: Path,
    *,
    parent_directory: Path = DEFAULT_PARENT,
    exact_arrays_path: Path = DEFAULT_EXACT,
) -> dict[str, object]:
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    partial_arrays_path = partial_directory / "partial_trajectory.npz"
    partial_metadata_path = partial_directory / "partial_metadata.json"
    partial_plan_path = partial_directory / "plan.json"
    parent_summary_path = parent_directory / "summary.json"
    parent_arrays_path = parent_directory / "segmented_horizon.npz"
    required = (
        partial_arrays_path,
        partial_metadata_path,
        partial_plan_path,
        parent_summary_path,
        parent_arrays_path,
        exact_arrays_path,
    )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("partial or reference artifacts are incomplete")

    metadata = json.loads(partial_metadata_path.read_text(encoding="utf-8"))
    plan = json.loads(partial_plan_path.read_text(encoding="utf-8"))
    parent_summary = json.loads(parent_summary_path.read_text(encoding="utf-8"))
    settings = parent_summary["parameters"]
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
    cutoff = int(settings["phonon_cutoff"])
    relative_dimension = 2 * cutoff + 1
    scenario_index = int(plan["exact_scenario_index"])

    with np.load(partial_arrays_path, allow_pickle=False) as arrays:
        times = np.asarray(arrays["times"], dtype=float)
        padded_parameters = np.asarray(
            arrays["parameter_trajectory"], dtype=float
        )
        packet_counts = np.asarray(
            arrays["packet_count_trajectory"], dtype=int
        )
    if not np.isclose(times[-1], metadata["current_time"], atol=1e-12):
        raise ValueError("partial trajectory and metadata endpoints differ")
    final_time = float(times[-1])

    with np.load(parent_arrays_path, allow_pickle=False) as arrays:
        parent_times = np.asarray(arrays["times"], dtype=float)
        parent_closed_all = np.asarray(arrays["closed_coordinates"], dtype=float)
    with np.load(exact_arrays_path, allow_pickle=False) as arrays:
        exact_times = np.asarray(arrays["times"], dtype=float)
        scales = np.asarray(arrays["coordinate_scales"], dtype=float)
        exact_closed_all = np.asarray(
            arrays["exact_dop853_closed"], dtype=float
        )[scenario_index].copy()
        exact_states_all = np.asarray(
            arrays["exact_dop853_state_vectors"], dtype=complex
        )[scenario_index].copy()
    parent_indices = _matched_indices(parent_times, times)
    exact_indices = _matched_indices(exact_times, times)
    parent_closed = parent_closed_all[parent_indices]
    exact_closed = exact_closed_all[exact_indices]
    exact_states = exact_states_all[exact_indices]

    exact_initial_state = exact_states_all[0]
    center_state = electron_relative_state(
        exact_initial_state,
        phonon_cutoff=cutoff,
    ).center_state
    hierarchy = moment_hierarchy(4)
    center_amplitude = -np.sqrt(2.0) * parameters.coupling / parameters.omega_ph
    model_closed = np.empty((times.size, 31), dtype=float)
    fidelity = np.empty(times.size)
    retained_norm = np.empty(times.size)
    raw_norm = np.empty(times.size)
    normalized_norm = np.empty(times.size)
    electron_minimum = np.empty(times.size)
    boson_minimum = np.empty(times.size)
    joint_minimum = np.empty(times.size)
    trace_residual = np.empty(times.size)
    for index, count in enumerate(packet_counts):
        parameters_at_time = padded_parameters[index, : 16 * int(count)]
        raw_state = multi_coherent_state(
            parameters_at_time,
            relative_dimension=relative_dimension,
        )
        raw_norm[index] = float(np.linalg.norm(raw_state))
        state = normalized_packet_state(
            parameters_at_time,
            relative_dimension=relative_dimension,
        )
        normalized_norm[index] = float(np.linalg.norm(state))
        model_closed[index] = relative_state_closed_coordinates(
            state,
            hierarchy,
            center_amplitude=center_amplitude,
        )
        matrix_state = closed_scalar_to_matrix_state(model_closed[index])
        electron_minimum[index] = float(
            np.linalg.eigvalsh(matrix_state.electron_density)[0]
        )
        boson_minimum[index] = float(
            np.linalg.eigvalsh(boson_moment_matrix(matrix_state))[0]
        )
        joint_minimum[index] = float(
            np.linalg.eigvalsh(electron_phonon_moment_matrix(matrix_state))[0]
        )
        trace_residual[index] = float(
            max(
                abs(np.trace(matrix_state.electron_phonon_correlation[0])),
                abs(np.trace(matrix_state.electron_phonon_correlation[1])),
            )
        )
        embedded = electron_relative_product_to_local_state(
            state,
            center_state,
            phonon_cutoff=cutoff,
        )
        fidelity[index] = float(abs(np.vdot(exact_states[index], embedded.state)) ** 2)
        retained_norm[index] = embedded.retained_norm

    exact_observables = _observables(times, exact_closed, parameters)
    parent_observables = _observables(times, parent_closed, parameters)
    model_observables = _observables(times, model_closed, parameters)
    model_error = (model_closed - exact_closed) / scales
    parent_error = (parent_closed - exact_closed) / scales
    observable_names = (
        "site_0_occupation",
        "site_1_occupation",
        "electronic_energy",
        "phonon_energy",
        "electron_phonon_energy",
        "internal_total_energy",
    )

    def interval_score(start: float, stop: float) -> dict[str, object]:
        selected = (times >= start - 1e-12) & (times <= stop + 1e-12)
        return {
            "all31_scaled_rms": float(np.sqrt(np.mean(model_error[selected] ** 2))),
            "parent_all31_scaled_rms": float(
                np.sqrt(np.mean(parent_error[selected] ** 2))
            ),
            "block_scaled_rms": {
                name: float(np.sqrt(np.mean(model_error[selected, block] ** 2)))
                for name, block in CLOSED_COORDINATE_BLOCKS.items()
            },
            "observable_time_rms": {
                name: _time_rms(
                    times,
                    model_observables[:, column] - exact_observables[:, column],
                    start=start,
                    stop=stop,
                )
                for column, name in enumerate(observable_names)
            },
            "parent_observable_time_rms": {
                name: _time_rms(
                    times,
                    parent_observables[:, column] - exact_observables[:, column],
                    start=start,
                    stop=stop,
                )
                for column, name in enumerate(observable_names)
            },
            "minimum_fidelity": float(np.min(fidelity[selected])),
        }

    intervals = {
        "0_to_20": interval_score(0.0, 20.0),
        "0_to_40": interval_score(0.0, 40.0),
        f"0_to_{final_time:g}": interval_score(0.0, final_time),
        "20_to_40": interval_score(20.0, 40.0),
        f"40_to_{final_time:g}": interval_score(40.0, final_time),
    }
    full = intervals[f"0_to_{final_time:g}"]
    summary: dict[str, object] = {
        "schema": "paper_v_mixed_guided_packet_stopped_summary_v1",
        "run_id": output_directory.name,
        "classification": "autonomous_exploratory_local_not_promoted",
        "status": "complete_through_user_stop",
        "termination": {
            "reason": "user_declared_long_horizon_evidence_sufficient",
            "last_complete_trajectory_time": final_time,
            "requested_target_time": float(plan["final_time"]),
        },
        "parameters": {
            "exact_scenario": plan["exact_scenario"],
            "exact_scenario_index": scenario_index,
            "lambda_ep": parameters.lambda_ep,
            "gamma": parameters.gamma,
            "coupling": parameters.coupling,
            "drive_protocol": drive_data,
            "phonon_cutoff": cutoff,
            "initial_packets_per_branch": int(packet_counts[0]),
            "final_packets_per_branch": int(packet_counts[-1]),
            "final_time": final_time,
            "sample_step": float(plan["sample_step"]),
            "maximum_step": float(plan["maximum_step"]),
            "relative_damping": float(plan["relative_damping"]),
        },
        "admissions": metadata["admissions"],
        "adaptive_attempts": metadata["adaptive_attempts"],
        "integration": {
            "solver": "adaptive_DOP853",
            "segments": metadata["segments"],
            "function_evaluations": int(
                sum(item["function_evaluations"] for item in metadata["segments"])
            ),
            "continuation_function_evaluations": int(
                metadata["continuation_function_evaluations"]
            ),
            "continuation_elapsed_seconds": float(
                metadata["continuation_elapsed_seconds"]
            ),
            "peak_rss_mb": float(metadata["peak_rss_mb"]),
            "online_exact_reference_used": False,
        },
        "packet_capacity": {
            "meaning_of_K": "coherent packets per electronic branch",
            "final_K": int(packet_counts[-1]),
            "time_average_K": float(
                np.trapezoid(packet_counts.astype(float), times) / final_time
            ),
            "admission_times": [
                float(item["time"]) for item in metadata["admissions"]
            ],
        },
        "comparison": {
            "mixed_guided_all31_scaled_rms": full["all31_scaled_rms"],
            "mixed_guided_c_scaled_rms": full["block_scaled_rms"]["C"],
            "parent_all31_scaled_rms": full["parent_all31_scaled_rms"],
            "mixed_guided_observable_time_rms": full["observable_time_rms"],
            "parent_observable_time_rms": full["parent_observable_time_rms"],
            "minimum_exact_state_fidelity": float(np.min(fidelity)),
            "final_exact_state_fidelity": float(fidelity[-1]),
            "minimum_local_embedding_retained_norm": float(np.min(retained_norm)),
            "intervals": intervals,
            "representability": {
                "minimum_electron_density_eigenvalue": float(
                    np.min(electron_minimum)
                ),
                "minimum_boson_moment_eigenvalue": float(np.min(boson_minimum)),
                "minimum_joint_gram_eigenvalue": float(np.min(joint_minimum)),
                "maximum_correlation_trace_residual": float(
                    np.max(trace_residual)
                ),
            },
            "state_norms": {
                "maximum_normalized_physical_ket_norm_error": float(
                    np.max(np.abs(normalized_norm - 1.0))
                ),
                "minimum_raw_coefficient_gauge_state_norm": float(np.min(raw_norm)),
                "final_raw_coefficient_gauge_state_norm": float(raw_norm[-1]),
            },
        },
        "interpretation": (
            "The multi-coherent ket, not the archive moment EOM, generated the "
            "trajectory. Archive moments and Gram geometry were contracted from "
            "the ket and used for admission guidance. Exact cutoff-16 data were "
            "used only in this offline score."
        ),
    }
    arrays_path = output_directory / "mixed_guided_packet_rollout.npz"
    np.savez_compressed(
        arrays_path,
        times=times,
        parameter_trajectory=padded_parameters,
        packet_count_trajectory=packet_counts,
        exact_closed_coordinates=exact_closed,
        parent_closed_coordinates=parent_closed,
        mixed_guided_closed_coordinates=model_closed,
        coordinate_scales=scales,
        exact_observables=exact_observables,
        parent_observables=parent_observables,
        mixed_guided_observables=model_observables,
        exact_state_fidelity=fidelity,
        local_embedding_retained_norm=retained_norm,
        raw_coefficient_gauge_state_norm=raw_norm,
        normalized_physical_state_norm=normalized_norm,
        minimum_electron_density_eigenvalue=electron_minimum,
        minimum_boson_moment_eigenvalue=boson_minimum,
        minimum_joint_gram_eigenvalue=joint_minimum,
        correlation_trace_residual=trace_residual,
    )
    summary_path = output_directory / "summary.json"
    _write_json(summary_path, summary)
    plot_path = output_directory / "observable_comparison.png"
    _plot(
        plot_path,
        times,
        exact_observables,
        parent_observables,
        model_observables,
    )
    manifest = {
        "schema": "paper_v_mixed_guided_packet_stopped_manifest_v1",
        "status": "complete_through_user_stop",
        "python": sys.version,
        "platform": platform.platform(),
        "input_hashes": {str(path): _sha256(path) for path in required},
        "artifact_hashes": {
            path.name: _sha256(path)
            for path in (arrays_path, summary_path, plot_path)
        },
    }
    _write_json(output_directory / "runtime_manifest.json", manifest)
    print(json.dumps(summary["comparison"], indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--parent-directory", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--exact-arrays", type=Path, default=DEFAULT_EXACT)
    args = parser.parse_args()
    finalize(
        args.partial_directory,
        args.output_directory,
        parent_directory=args.parent_directory,
        exact_arrays_path=args.exact_arrays,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
