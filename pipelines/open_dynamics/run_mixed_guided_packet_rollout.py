#!/usr/bin/env python3
"""Propagate a state-continuous mixed-guided fixed-capacity packet model."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

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
    boson_moment_matrix,
    closed_scalar_to_matrix_state,
    electron_phonon_moment_matrix,
    matrix_state_to_closed_scalar_coordinates,
)
from paper5.stability.moment_hierarchy import moment_hierarchy
from paper5.stability.multi_coherent import (
    multi_coherent_state,
    relative_state_closed_coordinates,
)
from paper5.stability.multi_coherent_scores import CLOSED_COORDINATE_BLOCKS
from paper5.stability.mixed_enriched_propagation import (
    admit_mixed_guided_packets,
    archive_gram_admission_signals,
    normalized_packet_state,
)
from paper5.stability.multi_coherent import multi_coherent_rhs
from pipelines.open_dynamics.run_integrable_mixed_enriched_rollout import (
    DEFAULT_EXACT,
    DEFAULT_PARENT,
    _matched_indices,
    _observables,
    _plot,
    _sha256,
    _write_json,
)


RUN_ID = "paper_v_mixed_guided_packet_k6_cutoff16_t4_20260805_v1"
DEFAULT_OUTPUT = Path("output/local_runs") / RUN_ID
EXACT_SCENARIO_NAMES = ("central", "plus", "minus")


def _peak_rss_megabytes() -> float:
    """Return this process's peak resident memory in decimal megabytes."""

    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() != "Darwin":
        value *= 1024.0
    return value / 1e6


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _padded_parameter_trajectory(
    states: list[np.ndarray],
) -> np.ndarray:
    maximum_size = max(state.size for state in states)
    padded = np.zeros((len(states), maximum_size), dtype=float)
    for index, state in enumerate(states):
        padded[index, : state.size] = state
    return padded


def run(
    output_directory: Path,
    *,
    parent_directory: Path = DEFAULT_PARENT,
    exact_arrays_path: Path = DEFAULT_EXACT,
    resume_directory: Path | None = None,
    reference_coordinates_path: Path | None = None,
    exact_scenario_index: int = 0,
    admission_count: int = 2,
    final_time: float = 4.0,
    sample_step: float = 0.05,
    maximum_step: float = 0.01,
    relative_tolerance: float = 1e-8,
    absolute_tolerance: float = 1e-10,
    relative_damping: float = 3e-4,
    fit_maximum_iterations: int = 40,
    fit_population_size: int = 6,
    readmission_times: tuple[float, ...] = (),
    readmissions_per_time: int = 1,
    adaptive_readmission: bool = False,
    adaptive_segment_length: float = 0.5,
    spawn_relative_residual_threshold: float = 5e-2,
    spawn_absolute_residual_threshold: float = 2e-2,
    partial_trajectory_interval: float = 2.0,
) -> dict[str, object]:
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty directory {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    if admission_count < 1:
        raise ValueError("admission_count must be positive")
    if exact_scenario_index not in range(len(EXACT_SCENARIO_NAMES)):
        raise ValueError("exact_scenario_index must be 0, 1, or 2")
    if readmissions_per_time < 1:
        raise ValueError("readmissions_per_time must be positive")
    if adaptive_segment_length <= 0.0:
        raise ValueError("adaptive_segment_length must be positive")
    if (
        spawn_relative_residual_threshold < 0.0
        or spawn_absolute_residual_threshold < 0.0
    ):
        raise ValueError("spawn thresholds must be nonnegative")
    if adaptive_readmission and readmission_times:
        raise ValueError(
            "adaptive and declared readmissions are separate pilot modes"
        )
    if resume_directory is not None and not adaptive_readmission:
        raise ValueError("resume requires adaptive_readmission")
    if partial_trajectory_interval <= 0.0:
        raise ValueError("partial_trajectory_interval must be positive")
    sample_intervals = int(round(final_time / sample_step))
    if not np.isclose(
        sample_intervals * sample_step,
        final_time,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("final_time must be a multiple of sample_step")
    times = np.linspace(0.0, final_time, sample_intervals + 1)
    readmission_times = tuple(float(value) for value in readmission_times)
    if any(
        value <= 0.0 or value >= final_time for value in readmission_times
    ):
        raise ValueError("readmission times must lie strictly inside the horizon")
    if tuple(sorted(set(readmission_times))) != readmission_times:
        raise ValueError("readmission times must be unique and increasing")
    if any(
        not np.any(np.isclose(times, value, atol=1e-12, rtol=0.0))
        for value in readmission_times
    ):
        raise ValueError("readmission times must lie on the output sample grid")
    if adaptive_readmission:
        adaptive_intervals = int(round(final_time / adaptive_segment_length))
        if not np.isclose(
            adaptive_intervals * adaptive_segment_length,
            final_time,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                "final_time must be divisible by adaptive_segment_length"
            )
        if not np.isclose(
            round(adaptive_segment_length / sample_step) * sample_step,
            adaptive_segment_length,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                "adaptive_segment_length must lie on the output sample grid"
            )

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
    drive_data = settings.get("drive_protocol")
    if drive_data is None:
        drive = GaussianSineDrive.from_parameters(parameters)
        drive_data = {
            "amplitude": drive.amplitude,
            "pulse_width": drive.pulse_width,
            "delays": list(drive.delays),
        }
    else:
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
    with np.load(exact_arrays_path, allow_pickle=False) as arrays:
        coordinate_scales = np.asarray(arrays["coordinate_scales"], dtype=float)
        exact_initial_state = np.asarray(
            arrays["exact_dop853_state_vectors"],
            dtype=complex,
        )[exact_scenario_index, 0].copy()
    if coordinate_scales.shape != (31,) or np.any(coordinate_scales <= 0.0):
        raise ValueError("frozen coordinate scales must be positive")
    parent_indices = _matched_indices(parent_times_all, times)
    source_summary_path: Path | None = None
    source_arrays_path: Path | None = None
    source_manifest_path: Path | None = None
    source_plan_path: Path | None = None
    resume_summary: dict[str, object] | None = None
    resume_times = np.empty(0, dtype=float)
    resume_parameter_states: list[np.ndarray] = []
    resume_packet_counts = np.empty(0, dtype=int)
    resume_time = 0.0
    if resume_directory is not None:
        source_summary_path = resume_directory / "summary.json"
        source_arrays_path = resume_directory / "mixed_guided_packet_rollout.npz"
        source_manifest_path = resume_directory / "runtime_manifest.json"
        source_plan_path = resume_directory / "plan.json"
        if not all(
            path.is_file()
            for path in (
                source_summary_path,
                source_arrays_path,
                source_manifest_path,
                source_plan_path,
            )
        ):
            raise FileNotFoundError("resume rollout artifacts are incomplete")
        resume_summary = json.loads(
            source_summary_path.read_text(encoding="utf-8")
        )
        if resume_summary.get("status") != "complete":
            raise ValueError("resume rollout must be complete")
        resume_manifest = json.loads(
            source_manifest_path.read_text(encoding="utf-8")
        )
        if resume_manifest.get("status") != "complete":
            raise ValueError("resume manifest must certify a complete rollout")
        resume_plan = json.loads(source_plan_path.read_text(encoding="utf-8"))
        for key, expected in {
            "relative_tolerance": relative_tolerance,
            "absolute_tolerance": absolute_tolerance,
            "admission_count": admission_count,
        }.items():
            if resume_plan.get(key) != expected:
                raise ValueError(
                    f"resume plan {key!r} differs from the requested contract"
                )
        resume_settings = resume_summary["parameters"]
        expected_resume_settings = {
            "exact_scenario_index": exact_scenario_index,
            "lambda_ep": parameters.lambda_ep,
            "gamma": parameters.gamma,
            "phonon_cutoff": cutoff,
            "drive_protocol": drive_data,
            "sample_step": sample_step,
            "maximum_step": maximum_step,
            "relative_damping": relative_damping,
            "adaptive_readmission": True,
            "adaptive_segment_length": adaptive_segment_length,
            "spawn_relative_residual_threshold": (
                spawn_relative_residual_threshold
            ),
            "spawn_absolute_residual_threshold": (
                spawn_absolute_residual_threshold
            ),
        }
        for key, expected in expected_resume_settings.items():
            if resume_settings.get(key) != expected:
                raise ValueError(
                    f"resume setting {key!r} differs from the requested contract"
                )
        with np.load(source_arrays_path, allow_pickle=False) as arrays:
            resume_times = np.asarray(arrays["times"], dtype=float)
            padded = np.asarray(arrays["parameter_trajectory"], dtype=float)
            resume_packet_counts = np.asarray(
                arrays["packet_count_trajectory"], dtype=int
            )
        if (
            resume_times.ndim != 1
            or resume_times.size < 2
            or resume_packet_counts.shape != resume_times.shape
            or padded.shape[0] != resume_times.size
        ):
            raise ValueError("resume trajectory has incompatible dimensions")
        resume_time = float(resume_times[-1])
        if not 0.0 < resume_time < final_time:
            raise ValueError("resume endpoint must lie inside the requested horizon")
        if not np.array_equal(resume_times, times[: resume_times.size]):
            raise ValueError("resume samples are not a prefix of the target grid")
        resume_parameter_states = [
            padded[index, : 16 * int(count)].copy()
            for index, count in enumerate(resume_packet_counts)
        ]
        initial_count = int(resume_settings["initial_packets_per_branch"])
        packet_parameters = resume_parameter_states[-1].copy()
        admissions = list(resume_summary.get("admissions", []))
        adaptive_attempts = list(resume_summary.get("adaptive_attempts", []))
    else:
        initial_count = int(parent_counts_all[0])
        packet_parameters = parent_parameters_all[
            0, : 16 * initial_count
        ].copy()
        admissions = []
        adaptive_attempts = []

    def signal_payload(signals) -> dict[str, object]:
        return {
            "native_hilbert_residual_squared": (
                signals.native_hilbert_residual_squared
            ),
            "native_hilbert_relative_residual": (
                signals.native_hilbert_relative_residual
            ),
            "target_velocity_norm": signals.target_velocity_norm,
            "joint_gram_rate_defect_squared": (
                signals.joint_gram_rate_defect_squared
            ),
            "mixed_observable_impact_squared": (
                signals.mixed_observable_impact_squared
            ),
            "native_geometric_rank": signals.native_geometric_rank,
            "native_condition_number": signals.native_condition_number,
            "joint_gram_support_rank": signals.joint_gram_support_rank,
            "mixed_novel_rank": signals.mixed_novel_rank,
            "minimum_joint_gram_eigenvalue": (
                signals.minimum_joint_gram_eigenvalue
            ),
        }

    def admit_at_time(
        time_value: float,
        current_parameters: np.ndarray,
        count: int,
    ) -> np.ndarray:
        for _ in range(count):
            admission = admit_mixed_guided_packets(
                time_value,
                current_parameters,
                parameters,
                relative_dimension=relative_dimension,
                drive_protocol=drive,
                relative_damping=None,
                fit_maximum_iterations=fit_maximum_iterations,
                fit_population_size=fit_population_size,
                fit_seed=260805 + len(admissions),
            )
            current_parameters = admission.parameters
            admissions.append(
                {
                    "time": time_value,
                    "previous_packet_count": admission.previous_packet_count,
                    "packet_count": admission.packet_count,
                    "centers": [
                        [float(value.real), float(value.imag)]
                        for value in admission.fitted_centers
                    ],
                    "state_discontinuity": admission.state_discontinuity,
                    "native_relative_residual_before": (
                        admission.native_relative_residual_before
                    ),
                    "native_relative_residual_after": (
                        admission.native_relative_residual_after
                    ),
                    "mixed_gain_norm_before": admission.mixed_gain_norm_before,
                    "mixed_gain_norm_after": admission.mixed_gain_norm_after,
                    "function_evaluations": admission.function_evaluations,
                }
            )
        return current_parameters

    def adaptive_admit_at_time(
        time_value: float,
        current_parameters: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, object]]:
        before = archive_gram_admission_signals(
            time_value,
            current_parameters,
            parameters,
            relative_dimension=relative_dimension,
            coordinate_scales=coordinate_scales,
            drive_protocol=drive,
        )
        absolute_residual = np.sqrt(
            before.native_hilbert_residual_squared
        )
        triggered = bool(
            before.native_hilbert_relative_residual
            >= spawn_relative_residual_threshold
            and absolute_residual >= spawn_absolute_residual_threshold
        )
        record: dict[str, object] = {
            "time": time_value,
            "triggered": triggered,
            "accepted": False,
            "signals_before": signal_payload(before),
        }
        if not triggered:
            return current_parameters, record

        candidate_started = time.time()
        candidate = admit_mixed_guided_packets(
            time_value,
            current_parameters,
            parameters,
            relative_dimension=relative_dimension,
            drive_protocol=drive,
            relative_damping=None,
            fit_maximum_iterations=fit_maximum_iterations,
            fit_population_size=fit_population_size,
            fit_seed=260805 + len(admissions),
        )
        after = archive_gram_admission_signals(
            time_value,
            candidate.parameters,
            parameters,
            relative_dimension=relative_dimension,
            coordinate_scales=coordinate_scales,
            drive_protocol=drive,
        )
        objective_before = np.asarray(
            [
                before.native_hilbert_residual_squared,
                before.joint_gram_rate_defect_squared,
                before.mixed_observable_impact_squared,
            ],
            dtype=float,
        )
        objective_after = np.asarray(
            [
                after.native_hilbert_residual_squared,
                after.joint_gram_rate_defect_squared,
                after.mixed_observable_impact_squared,
            ],
            dtype=float,
        )
        active = objective_before > 100.0 * np.finfo(float).eps
        normalized_after = float(
            np.mean(objective_after[active] / objective_before[active])
        )
        objective_reduction = 1.0 - normalized_after
        rank_gain = (
            after.native_geometric_rank - before.native_geometric_rank
        )
        coordinate_gain = candidate.parameters.size - current_parameters.size
        measured_cost_seconds = time.time() - candidate_started
        condition_ratio = (
            before.native_condition_number / after.native_condition_number
        )
        admission_score = float(
            objective_reduction
            / max(measured_cost_seconds, np.finfo(float).tiny)
            * rank_gain
            / max(coordinate_gain, 1)
            * condition_ratio
        )
        accepted = bool(
            objective_reduction > 1e-8
            and rank_gain > 0
            and candidate.state_discontinuity <= 1e-12
        )
        record.update(
            {
                "accepted": accepted,
                "signals_after": signal_payload(after),
                "normalized_objective_after": normalized_after,
                "objective_reduction": objective_reduction,
                "native_rank_gain": rank_gain,
                "real_coordinate_gain": coordinate_gain,
                "condition_ratio": condition_ratio,
                "measured_candidate_cost_seconds": measured_cost_seconds,
                "admission_score": admission_score,
                "candidate_packet_count": candidate.packet_count,
                "state_discontinuity": candidate.state_discontinuity,
            }
        )
        if not accepted:
            return current_parameters, record
        admissions.append(
            {
                "time": time_value,
                "previous_packet_count": candidate.previous_packet_count,
                "packet_count": candidate.packet_count,
                "centers": [
                    [float(value.real), float(value.imag)]
                    for value in candidate.fitted_centers
                ],
                "state_discontinuity": candidate.state_discontinuity,
                "native_relative_residual_before": (
                    candidate.native_relative_residual_before
                ),
                "native_relative_residual_after": (
                    candidate.native_relative_residual_after
                ),
                "mixed_gain_norm_before": candidate.mixed_gain_norm_before,
                "mixed_gain_norm_after": candidate.mixed_gain_norm_after,
                "function_evaluations": candidate.function_evaluations,
                "adaptive_selection": record,
            }
        )
        return candidate.parameters, record

    if resume_summary is None:
        packet_parameters = admit_at_time(
            0.0,
            packet_parameters,
            admission_count,
        )
        admitted_count = initial_count + admission_count
    else:
        admitted_count = int(
            resume_summary["parameters"]["admitted_packets_per_branch"]
        )
    plan: dict[str, object] = {
        "schema": "paper_v_mixed_guided_packet_plan_v2",
        "run_id": output_directory.name,
        "classification": "autonomous_exploratory_local_not_promoted",
        "initial_packets_per_branch": initial_count,
        "admitted_packets_per_branch": admitted_count,
        "admission_count": admission_count,
        "readmission_times": list(readmission_times),
        "readmissions_per_time": readmissions_per_time,
        "adaptive_readmission": adaptive_readmission,
        "adaptive_segment_length": adaptive_segment_length,
        "spawn_relative_residual_threshold": (
            spawn_relative_residual_threshold
        ),
        "spawn_absolute_residual_threshold": (
            spawn_absolute_residual_threshold
        ),
        "coordinate_scales_source": str(exact_arrays_path),
        "coordinate_scales_are_frozen_construction_data": True,
        "reference_coordinates_source": (
            str(reference_coordinates_path)
            if reference_coordinates_path is not None
            else str(exact_arrays_path)
        ),
        "exact_scenario": EXACT_SCENARIO_NAMES[exact_scenario_index],
        "exact_scenario_index": exact_scenario_index,
        "final_time": final_time,
        "sample_step": sample_step,
        "maximum_step": maximum_step,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "relative_damping": relative_damping,
        "resume_source": (
            str(resume_directory) if resume_directory is not None else None
        ),
        "resume_time": resume_time,
        "terminal_resume_boundary_admission_checked": (
            resume_directory is not None
        ),
        "partial_trajectory_interval": partial_trajectory_interval,
        "online_exact_reference_used": False,
    }
    _write_json(output_directory / "plan.json", plan)

    def rhs(time_value: float, values: np.ndarray) -> np.ndarray:
        return multi_coherent_rhs(
            time_value,
            values,
            parameters,
            relative_dimension=relative_dimension,
            drive_protocol=drive,
            regularization="tikhonov",
            relative_damping=relative_damping,
        )

    started = time.time()
    parameter_states = list(resume_parameter_states)
    packet_counts = [int(value) for value in resume_packet_counts]
    segment_records = (
        list(resume_summary["integration"].get("segments", []))
        if resume_summary is not None
        else []
    )
    continuation_segment_records: list[dict[str, object]] = []

    def write_progress(
        current_time: float,
        *,
        force_trajectory: bool = False,
    ) -> None:
        current_count = packet_parameters.size // 16
        raw_norm = float(
            np.linalg.norm(
                multi_coherent_state(
                    packet_parameters,
                    relative_dimension=relative_dimension,
                )
            )
        )
        status = {
            "schema": "paper_v_mixed_guided_packet_progress_v1",
            "status": "running",
            "current_time": current_time,
            "target_time": final_time,
            "packets_per_electronic_branch": current_count,
            "accepted_admissions": len(admissions),
            "adaptive_attempts": len(adaptive_attempts),
            "continuation_segments": len(continuation_segment_records),
            "continuation_function_evaluations": int(
                sum(
                    record["function_evaluations"]
                    for record in continuation_segment_records
                )
            ),
            "continuation_elapsed_seconds": time.time() - started,
            "peak_rss_mb": _peak_rss_megabytes(),
            "raw_coefficient_gauge_state_norm": raw_norm,
        }
        _write_json(output_directory / "progress.json", status)
        _write_npz_atomic(
            output_directory / "continuation_checkpoint.npz",
            time=np.asarray([current_time], dtype=float),
            parameters=packet_parameters,
            packet_count=np.asarray([current_count], dtype=int),
        )
        relative_progress = current_time - resume_time
        write_trajectory = force_trajectory or np.isclose(
            relative_progress / partial_trajectory_interval,
            round(relative_progress / partial_trajectory_interval),
            atol=1e-10,
            rtol=0.0,
        )
        if write_trajectory:
            _write_npz_atomic(
                output_directory / "partial_trajectory.npz",
                times=times[: len(parameter_states)],
                parameter_trajectory=_padded_parameter_trajectory(
                    parameter_states
                ),
                packet_count_trajectory=np.asarray(packet_counts, dtype=int),
            )
            _write_json(
                output_directory / "partial_metadata.json",
                {
                    **status,
                    "admissions": admissions,
                    "adaptive_attempts": adaptive_attempts,
                    "segments": segment_records,
                },
            )
        print(
            "PROGRESS "
            f"t={current_time:.2f}/{final_time:.2f} "
            f"K={current_count} admissions={len(admissions)} "
            f"nfev={status['continuation_function_evaluations']} "
            f"elapsed={status['continuation_elapsed_seconds']:.1f}s "
            f"peak_rss={status['peak_rss_mb']:.1f}MB "
            f"raw_norm={raw_norm:.6g}",
            flush=True,
        )

    if resume_summary is not None:
        packet_parameters, boundary_record = adaptive_admit_at_time(
            resume_time,
            packet_parameters,
        )
        adaptive_attempts.append(boundary_record)
        parameter_states[-1] = packet_parameters.copy()
        packet_counts[-1] = packet_parameters.size // 16
        write_progress(resume_time, force_trajectory=True)
    if adaptive_readmission:
        adaptive_segment_count = int(
            round((final_time - resume_time) / adaptive_segment_length)
        )
        segment_boundaries = tuple(
            np.linspace(
                resume_time,
                final_time,
                adaptive_segment_count + 1,
            )
        )
    else:
        segment_boundaries = (0.0, *readmission_times, final_time)
    for segment_index, (start_time, end_time) in enumerate(
        zip(segment_boundaries[:-1], segment_boundaries[1:], strict=True)
    ):
        if segment_index > 0 and not adaptive_readmission:
            packet_parameters = admit_at_time(
                start_time,
                packet_parameters,
                readmissions_per_time,
            )
            parameter_states[-1] = packet_parameters.copy()
            packet_counts[-1] = packet_parameters.size // 16
        if segment_index == 0 and resume_summary is None:
            sample_mask = (times >= start_time) & (times <= end_time)
        else:
            sample_mask = (times > start_time) & (times <= end_time)
        segment_times = times[sample_mask]
        segment_started = time.time()
        solution = solve_ivp(
            rhs,
            (start_time, end_time),
            packet_parameters,
            method="DOP853",
            t_eval=segment_times,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            max_step=maximum_step,
        )
        if not solution.success or solution.y.shape[1] != segment_times.size:
            raise RuntimeError(
                f"mixed-guided segment failed: {solution.message}"
            )
        sampled_parameters = [
            np.asarray(solution.y[:, index], dtype=float).copy()
            for index in range(segment_times.size)
        ]
        parameter_states.extend(sampled_parameters)
        packet_counts.extend(
            [packet_parameters.size // 16] * segment_times.size
        )
        packet_parameters = sampled_parameters[-1]
        adaptive_record: dict[str, object] | None = None
        if adaptive_readmission and end_time < final_time - 1e-12:
            packet_parameters, adaptive_record = adaptive_admit_at_time(
                end_time,
                packet_parameters,
            )
            adaptive_attempts.append(adaptive_record)
            sampled_parameters[-1] = packet_parameters.copy()
            parameter_states[-1] = packet_parameters.copy()
            packet_counts[-1] = packet_parameters.size // 16
        segment_record = {
            "start_time": start_time,
            "end_time": end_time,
            "packet_count": packet_parameters.size // 16,
            "function_evaluations": int(solution.nfev),
            "elapsed_seconds": time.time() - segment_started,
            "adaptive_admission": adaptive_record,
        }
        segment_records.append(segment_record)
        continuation_segment_records.append(segment_record)
        write_progress(end_time, force_trajectory=end_time >= final_time - 1e-12)
    if len(parameter_states) != times.size:
        raise RuntimeError("segmented propagation did not fill the sample grid")

    if reference_coordinates_path is None:
        with np.load(exact_arrays_path, allow_pickle=False) as arrays:
            exact_times_all = np.asarray(arrays["times"], dtype=float)
            exact_closed_all = np.asarray(
                arrays["exact_dop853_closed"],
                dtype=float,
            )[exact_scenario_index].copy()
            exact_states_all: np.ndarray | None = np.asarray(
                arrays["exact_dop853_state_vectors"],
                dtype=complex,
            )[exact_scenario_index].copy()
            scoring_coordinate_scales = np.asarray(
                arrays["coordinate_scales"], dtype=float
            )
    else:
        with np.load(reference_coordinates_path, allow_pickle=False) as arrays:
            exact_times_all = np.asarray(arrays["times"], dtype=float)
            exact_closed_all = np.asarray(
                arrays["exact_closed_coordinates"],
                dtype=float,
            )
        exact_states_all = None
        scoring_coordinate_scales = coordinate_scales
    if not np.array_equal(scoring_coordinate_scales, coordinate_scales):
        raise RuntimeError("scoring and frozen construction scales differ")
    exact_indices = _matched_indices(exact_times_all, times)
    exact_closed = exact_closed_all[exact_indices]
    exact_states = (
        exact_states_all[exact_indices]
        if exact_states_all is not None
        else None
    )
    center_state = electron_relative_state(
        exact_initial_state,
        phonon_cutoff=cutoff,
    ).center_state
    exact_model = _build_exact_dimer_model(parameters, phonon_cutoff=cutoff)
    model_closed = np.empty((times.size, 31), dtype=float)
    model_projected_closed = np.empty((times.size, 31), dtype=float)
    model_fidelity = (
        np.empty(times.size) if exact_states is not None else None
    )
    retained_local_norm = np.empty(times.size)
    raw_coefficient_gauge_norm = np.empty(times.size)
    normalized_state_norm = np.empty(times.size)
    electron_minimum = np.empty(times.size)
    boson_minimum = np.empty(times.size)
    joint_minimum = np.empty(times.size)
    correlation_trace_residual = np.empty(times.size)
    hierarchy = moment_hierarchy(4)
    center_amplitude = (
        -np.sqrt(2.0) * parameters.coupling / parameters.omega_ph
    )
    for index, current_parameters in enumerate(parameter_states):
        raw_state = multi_coherent_state(
            current_parameters,
            relative_dimension=relative_dimension,
        )
        raw_coefficient_gauge_norm[index] = float(np.linalg.norm(raw_state))
        relative_state = normalized_packet_state(
            current_parameters,
            relative_dimension=relative_dimension,
        )
        normalized_state_norm[index] = float(np.linalg.norm(relative_state))
        model_closed[index] = relative_state_closed_coordinates(
            relative_state,
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
        correlation_trace_residual[index] = float(
            max(
                abs(np.trace(matrix_state.electron_phonon_correlation[0])),
                abs(np.trace(matrix_state.electron_phonon_correlation[1])),
            )
        )
        embedded = electron_relative_product_to_local_state(
            relative_state,
            center_state,
            phonon_cutoff=cutoff,
        )
        model_projected_closed[index] = matrix_state_to_closed_scalar_coordinates(
            _contract_matrix_state(exact_model, embedded.state)
        )
        if model_fidelity is not None and exact_states is not None:
            model_fidelity[index] = float(
                abs(np.vdot(exact_states[index], embedded.state)) ** 2
            )
        retained_local_norm[index] = embedded.retained_norm
    parent_closed = parent_closed_all[parent_indices]
    scaled_model_error = (model_closed - exact_closed) / coordinate_scales
    scaled_projected_error = (
        model_projected_closed - exact_closed
    ) / coordinate_scales
    scaled_parent_error = (parent_closed - exact_closed) / coordinate_scales
    exact_observables = _observables(times, exact_closed, parameters)
    parent_observables = _observables(times, parent_closed, parameters)
    model_observables = _observables(times, model_closed, parameters)

    def interval_mask(start: float, stop: float) -> np.ndarray:
        selected = (times >= start - 1e-12) & (times <= stop + 1e-12)
        if np.count_nonzero(selected) < 2:
            raise ValueError("score interval must contain at least two samples")
        return selected

    def observable_rms(
        path: np.ndarray,
        *,
        start: float = 0.0,
        stop: float = final_time,
    ) -> dict[str, float]:
        names = (
            "site_0_occupation",
            "site_1_occupation",
            "electronic_energy",
            "phonon_energy",
            "electron_phonon_energy",
            "internal_total_energy",
        )
        selected = interval_mask(start, stop)
        selected_times = times[selected]
        return {
            name: float(
                np.sqrt(
                    np.trapezoid(
                        (
                            path[selected, column]
                            - exact_observables[selected, column]
                        )
                        ** 2,
                        selected_times,
                    )
                    / (stop - start)
                )
            )
            for column, name in enumerate(names)
        }

    def coordinate_metrics(
        scaled_error: np.ndarray,
        *,
        start: float,
        stop: float,
    ) -> dict[str, object]:
        selected_error = scaled_error[interval_mask(start, stop)]
        return {
            "all31_scaled_rms": float(
                np.sqrt(np.mean(selected_error**2))
            ),
            "block_scaled_rms": {
                name: float(
                    np.sqrt(np.mean(selected_error[:, block] ** 2))
                )
                for name, block in CLOSED_COORDINATE_BLOCKS.items()
            },
        }

    horizon_stops = tuple(
        value for value in (20.0, 40.0, final_time) if value <= final_time
    )
    horizon_stops = tuple(dict.fromkeys(horizon_stops))
    cumulative_error_growth = {
        f"0_to_{stop:g}": {
            "packet": coordinate_metrics(
                scaled_model_error,
                start=0.0,
                stop=stop,
            ),
            "parent": coordinate_metrics(
                scaled_parent_error,
                start=0.0,
                stop=stop,
            ),
            "packet_observable_time_rms": observable_rms(
                model_observables,
                start=0.0,
                stop=stop,
            ),
            "parent_observable_time_rms": observable_rms(
                parent_observables,
                start=0.0,
                stop=stop,
            ),
            "minimum_fidelity": (
                float(np.min(model_fidelity[interval_mask(0.0, stop)]))
                if model_fidelity is not None
                else None
            ),
        }
        for stop in horizon_stops
    }
    windowed_error_growth = {}
    for start, stop in ((20.0, 40.0), (40.0, final_time)):
        if stop <= start or stop > final_time:
            continue
        windowed_error_growth[f"{start:g}_to_{stop:g}"] = {
            "packet": coordinate_metrics(
                scaled_model_error,
                start=start,
                stop=stop,
            ),
            "parent": coordinate_metrics(
                scaled_parent_error,
                start=start,
                stop=stop,
            ),
            "packet_observable_time_rms": observable_rms(
                model_observables,
                start=start,
                stop=stop,
            ),
            "parent_observable_time_rms": observable_rms(
                parent_observables,
                start=start,
                stop=stop,
            ),
            "minimum_fidelity": (
                float(np.min(model_fidelity[interval_mask(start, stop)]))
                if model_fidelity is not None
                else None
            ),
        }

    summary: dict[str, object] = {
        "schema": "paper_v_mixed_guided_packet_summary_v2",
        "run_id": output_directory.name,
        "classification": "autonomous_exploratory_local_not_promoted",
        "status": "complete",
        "parameters": {
            "exact_scenario": EXACT_SCENARIO_NAMES[exact_scenario_index],
            "exact_scenario_index": exact_scenario_index,
            "lambda_ep": parameters.lambda_ep,
            "gamma": parameters.gamma,
            "coupling": parameters.coupling,
            "drive_protocol": drive_data,
            "phonon_cutoff": cutoff,
            "initial_packets_per_branch": initial_count,
            "admitted_packets_per_branch": admitted_count,
            "final_packets_per_branch": packet_parameters.size // 16,
            "readmission_times": list(readmission_times),
            "readmissions_per_time": readmissions_per_time,
            "adaptive_readmission": adaptive_readmission,
            "adaptive_segment_length": adaptive_segment_length,
            "spawn_relative_residual_threshold": (
                spawn_relative_residual_threshold
            ),
            "spawn_absolute_residual_threshold": (
                spawn_absolute_residual_threshold
            ),
            "final_time": final_time,
            "sample_step": sample_step,
            "maximum_step": maximum_step,
            "relative_tolerance": relative_tolerance,
            "absolute_tolerance": absolute_tolerance,
            "relative_damping": relative_damping,
            "resume_source": (
                str(resume_directory) if resume_directory is not None else None
            ),
            "resume_time": resume_time,
        },
        "admissions": admissions,
        "adaptive_attempts": adaptive_attempts,
        "integration": {
            "solver": "adaptive_DOP853",
            "function_evaluations": int(
                sum(record["function_evaluations"] for record in segment_records)
            ),
            "continuation_function_evaluations": int(
                sum(
                    record["function_evaluations"]
                    for record in continuation_segment_records
                )
            ),
            "source_elapsed_seconds": (
                float(resume_summary["integration"]["elapsed_seconds"])
                if resume_summary is not None
                else 0.0
            ),
            "continuation_elapsed_seconds": time.time() - started,
            "elapsed_seconds": (
                time.time()
                - started
                + (
                    float(resume_summary["integration"]["elapsed_seconds"])
                    if resume_summary is not None
                    else 0.0
                )
            ),
            "peak_rss_mb_during_continuation_process": _peak_rss_megabytes(),
            "segments": segment_records,
            "success": True,
            "online_exact_reference_used": False,
        },
        "packet_capacity": {
            "meaning_of_K": "coherent packets per electronic branch",
            "minimum_K": int(np.min(packet_counts)),
            "maximum_K": int(np.max(packet_counts)),
            "final_K": int(packet_counts[-1]),
            "time_average_K": float(
                np.trapezoid(np.asarray(packet_counts, dtype=float), times)
                / final_time
            ),
            "admission_times": [float(item["time"]) for item in admissions],
        },
        "comparison": {
            "parent_all31_scaled_rms": float(
                np.sqrt(np.mean(scaled_parent_error**2))
            ),
            "mixed_guided_all31_scaled_rms": float(
                np.sqrt(np.mean(scaled_model_error**2))
            ),
            "parent_c_scaled_rms": float(
                np.sqrt(np.mean(scaled_parent_error[:, 17:31] ** 2))
            ),
            "mixed_guided_c_scaled_rms": float(
                np.sqrt(np.mean(scaled_model_error[:, 17:31] ** 2))
            ),
            "projected_embedding_all31_scaled_rms": float(
                np.sqrt(np.mean(scaled_projected_error**2))
            ),
            "projected_embedding_c_scaled_rms": float(
                np.sqrt(np.mean(scaled_projected_error[:, 17:31] ** 2))
            ),
            "minimum_exact_state_fidelity": (
                float(np.min(model_fidelity))
                if model_fidelity is not None
                else None
            ),
            "final_exact_state_fidelity": (
                float(model_fidelity[-1])
                if model_fidelity is not None
                else None
            ),
            "minimum_local_embedding_retained_norm": float(
                np.min(retained_local_norm)
            ),
            "parent_observable_time_rms": observable_rms(parent_observables),
            "mixed_guided_observable_time_rms": observable_rms(model_observables),
            "cumulative_error_growth": cumulative_error_growth,
            "windowed_error_growth": windowed_error_growth,
            "representability": {
                "minimum_electron_density_eigenvalue": float(
                    np.min(electron_minimum)
                ),
                "minimum_boson_moment_eigenvalue": float(
                    np.min(boson_minimum)
                ),
                "minimum_joint_gram_eigenvalue": float(np.min(joint_minimum)),
                "maximum_correlation_trace_residual": float(
                    np.max(correlation_trace_residual)
                ),
            },
            "state_norms": {
                "maximum_normalized_physical_ket_norm_error": float(
                    np.max(np.abs(normalized_state_norm - 1.0))
                ),
                "minimum_raw_coefficient_gauge_state_norm": float(
                    np.min(raw_coefficient_gauge_norm)
                ),
                "maximum_raw_coefficient_gauge_state_norm": float(
                    np.max(raw_coefficient_gauge_norm)
                ),
                "final_raw_coefficient_gauge_state_norm": float(
                    raw_coefficient_gauge_norm[-1]
                ),
            },
        },
        "interpretation": (
            "The mixed directions select zero-weight packets from the current "
            "model state, preserving that state exactly. The model propagates "
            "autonomously between admissions with the established packet "
            "McLachlan DOP853 right-hand side; exact data enter only afterward "
            "for scoring."
        ),
    }
    padded_parameter_trajectory = _padded_parameter_trajectory(parameter_states)
    arrays_path = output_directory / "mixed_guided_packet_rollout.npz"
    output_arrays: dict[str, np.ndarray] = {
        "times": times,
        "parameter_trajectory": padded_parameter_trajectory,
        "packet_count_trajectory": np.asarray(packet_counts, dtype=int),
        "exact_closed_coordinates": exact_closed,
        "parent_closed_coordinates": parent_closed,
        "mixed_guided_closed_coordinates": model_closed,
        "mixed_guided_projected_closed_coordinates": model_projected_closed,
        "coordinate_scales": coordinate_scales,
        "exact_observables": exact_observables,
        "parent_observables": parent_observables,
        "mixed_guided_observables": model_observables,
        "local_embedding_retained_norm": retained_local_norm,
        "raw_coefficient_gauge_state_norm": raw_coefficient_gauge_norm,
        "normalized_physical_state_norm": normalized_state_norm,
        "minimum_electron_density_eigenvalue": electron_minimum,
        "minimum_boson_moment_eigenvalue": boson_minimum,
        "minimum_joint_gram_eigenvalue": joint_minimum,
        "correlation_trace_residual": correlation_trace_residual,
    }
    if model_fidelity is not None:
        output_arrays["exact_state_fidelity"] = model_fidelity
    np.savez_compressed(
        arrays_path,
        **output_arrays,
    )
    plot_path = output_directory / "observable_comparison.png"
    _plot(
        plot_path,
        times,
        exact_observables,
        parent_observables,
        model_observables,
    )
    summary_path = output_directory / "summary.json"
    _write_json(summary_path, summary)
    source_files = (
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[2]
        / "paper_5/src/paper5/stability/mixed_enriched_propagation.py",
    )
    artifacts = (
        output_directory / "plan.json",
        arrays_path,
        plot_path,
        summary_path,
    )
    manifest = {
        "schema": "paper_v_mixed_guided_packet_manifest_v2",
        "status": "complete",
        "python": sys.version,
        "platform": platform.platform(),
        "input_hashes": {
            str(path): _sha256(path)
            for path in (
                parent_summary_path,
                parent_arrays_path,
                exact_arrays_path,
                *(
                    (reference_coordinates_path,)
                    if reference_coordinates_path is not None
                    else ()
                ),
                *(
                    (
                        source_summary_path,
                        source_arrays_path,
                        source_manifest_path,
                        source_plan_path,
                    )
                    if resume_directory is not None
                    else ()
                ),
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
    _write_json(
        output_directory / "progress.json",
        {
            "schema": "paper_v_mixed_guided_packet_progress_v1",
            "status": "complete",
            "current_time": final_time,
            "target_time": final_time,
            "packets_per_electronic_branch": int(packet_counts[-1]),
            "accepted_admissions": len(admissions),
            "adaptive_attempts": len(adaptive_attempts),
            "continuation_segments": len(continuation_segment_records),
            "continuation_function_evaluations": int(
                sum(
                    record["function_evaluations"]
                    for record in continuation_segment_records
                )
            ),
            "peak_rss_mb": _peak_rss_megabytes(),
        },
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "run_id": summary["run_id"],
                "final_time": final_time,
                "final_K": summary["packet_capacity"]["final_K"],
                "time_average_K": summary["packet_capacity"]["time_average_K"],
                "all31_scaled_rms": summary["comparison"][
                    "mixed_guided_all31_scaled_rms"
                ],
                "C_scaled_rms": summary["comparison"][
                    "mixed_guided_c_scaled_rms"
                ],
                "minimum_fidelity": summary["comparison"][
                    "minimum_exact_state_fidelity"
                ],
                "peak_rss_mb": summary["integration"][
                    "peak_rss_mb_during_continuation_process"
                ],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parent-directory", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--exact-arrays", type=Path, default=DEFAULT_EXACT)
    parser.add_argument(
        "--resume-directory",
        type=Path,
        help=(
            "Completed adaptive rollout whose final stored state is the "
            "continuation checkpoint."
        ),
    )
    parser.add_argument(
        "--reference-coordinates",
        type=Path,
        default=None,
        help=(
            "Optional NPZ with times and exact_closed_coordinates; use when "
            "exact state vectors are unavailable for observable-only scoring."
        ),
    )
    parser.add_argument(
        "--exact-scenario-index",
        type=int,
        choices=range(len(EXACT_SCENARIO_NAMES)),
        default=0,
        help="Sealed exact member: 0=central, 1=plus, 2=minus.",
    )
    parser.add_argument("--admission-count", type=int, default=2)
    parser.add_argument("--final-time", type=float, default=4.0)
    parser.add_argument("--sample-step", type=float, default=0.05)
    parser.add_argument("--maximum-step", type=float, default=0.01)
    parser.add_argument("--relative-damping", type=float, default=3e-4)
    parser.add_argument(
        "--readmission-time",
        action="append",
        type=float,
        default=[],
    )
    parser.add_argument("--readmissions-per-time", type=int, default=1)
    parser.add_argument("--adaptive-readmission", action="store_true")
    parser.add_argument("--adaptive-segment-length", type=float, default=0.5)
    parser.add_argument(
        "--spawn-relative-residual-threshold",
        type=float,
        default=5e-2,
    )
    parser.add_argument(
        "--spawn-absolute-residual-threshold",
        type=float,
        default=2e-2,
    )
    parser.add_argument(
        "--partial-trajectory-interval",
        type=float,
        default=2.0,
    )
    arguments = parser.parse_args()
    run(
        arguments.output_directory,
        parent_directory=arguments.parent_directory,
        exact_arrays_path=arguments.exact_arrays,
        resume_directory=arguments.resume_directory,
        reference_coordinates_path=arguments.reference_coordinates,
        exact_scenario_index=arguments.exact_scenario_index,
        admission_count=arguments.admission_count,
        final_time=arguments.final_time,
        sample_step=arguments.sample_step,
        maximum_step=arguments.maximum_step,
        relative_damping=arguments.relative_damping,
        readmission_times=tuple(arguments.readmission_time),
        readmissions_per_time=arguments.readmissions_per_time,
        adaptive_readmission=arguments.adaptive_readmission,
        adaptive_segment_length=arguments.adaptive_segment_length,
        spawn_relative_residual_threshold=(
            arguments.spawn_relative_residual_threshold
        ),
        spawn_absolute_residual_threshold=(
            arguments.spawn_absolute_residual_threshold
        ),
        partial_trajectory_interval=arguments.partial_trajectory_interval,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
