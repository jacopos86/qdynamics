#!/usr/bin/env python3
"""Verify the integrable mixed packet-union chart on stored packet states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from paper5.stability.archive_gram_tangent_pilot import (
    full_state_matrix_derivative,
    packet_archive_mixed_frames,
    project_real_tangent,
)
from paper5.stability.hubbard_dimer import DimerParameters, GaussianSineDrive
from paper5.stability.matrix_reference import matrix_derivative_to_closed_scalar
from paper5.stability.mixed_exponential_layer import (
    mixed_exponential_layer_state,
    mixed_exponential_origin_tangent,
    mixed_layer_centers,
)
from paper5.stability.multi_coherent import relative_holstein_hamiltonian

RUN_ID = "paper_v_integrable_mixed_layer_gate_cutoff16_20260805_v2"
DEFAULT_MEMBERS = {
    "K6": Path(
        "output/local_runs/"
        "paper_v_multi_coherent_double_pulse_blind_model_cutoff16_20260804_v1/"
        "fine_central"
    ),
    "K8": Path(
        "output/local_runs/"
        "paper_v_multi_coherent_capacity_k8_t40_20260804_v1/fine_central"
    ),
    "K10": Path(
        "output/local_runs/"
        "paper_v_multi_coherent_capacity_k10_t40_20260804_v1/fine_central"
    ),
    "K12": Path(
        "output/local_runs/"
        "paper_v_multi_coherent_capacity_k12_t40_20260804_v1/fine_central"
    ),
}
DEFAULT_OUTPUT = Path("output/local_runs") / RUN_ID
DEFAULT_SCALES = Path(
    "output/local_runs/"
    "paper_v_trajectory_closure_identifiability_cutoff16_20260804_v1/"
    "trajectory_closure_identifiability.npz"
)
LOCAL_SPACE_NAMES = (
    "packet_archive",
    "packet_archive_truncated_mixed",
    "packet_archive_analytic_mixed",
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


def _physical_contract(settings: dict[str, object]) -> dict[str, object]:
    drive = settings["drive_protocol"]
    if not isinstance(drive, dict):
        raise ValueError("drive_protocol must be a mapping")
    return {
        "hopping": float(settings["hopping"]),
        "gamma": float(settings["gamma"]),
        "lambda_ep": float(settings["lambda_ep"]),
        "drive_amplitude": float(settings["drive_amplitude"]),
        "pulse_width": float(settings["pulse_width"]),
        "phonon_cutoff": int(settings["phonon_cutoff"]),
        "drive_protocol": {
            "amplitude": float(drive["amplitude"]),
            "pulse_width": float(drive["pulse_width"]),
            "delays": tuple(float(value) for value in drive["delays"]),
        },
    }


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def _fractional_reduction(before: np.ndarray, after: np.ndarray) -> float:
    denominator = _rms(before)
    return float((denominator - _rms(after)) / denominator)


def _realify_frame(frame: np.ndarray) -> np.ndarray:
    return np.vstack((frame.real, frame.imag))


def _range_basis(frame: np.ndarray, relative_threshold: float) -> np.ndarray:
    left, singular_values, _ = np.linalg.svd(
        _realify_frame(frame),
        full_matrices=False,
    )
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return np.empty((2 * frame.shape[0], 0), dtype=float)
    threshold = np.sqrt(relative_threshold) * singular_values[0]
    return left[:, singular_values > threshold]


def _residualized_real_frame(
    frame: np.ndarray,
    base_basis: np.ndarray,
) -> np.ndarray:
    real = _realify_frame(frame)
    return real - base_basis @ (base_basis.T @ real)


def _projector(frame: np.ndarray, relative_threshold: float) -> tuple[np.ndarray, int]:
    left, singular_values, _ = np.linalg.svd(frame, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return np.zeros((frame.shape[0], frame.shape[0])), 0
    keep = singular_values > np.sqrt(relative_threshold) * singular_values[0]
    basis = left[:, keep]
    return basis @ basis.T, int(np.count_nonzero(keep))


def _closed_response(
    state: np.ndarray,
    tangent: np.ndarray,
    relative_dimension: int,
) -> np.ndarray:
    full_state = np.concatenate((state, np.zeros_like(state)))
    responses = []
    for column in range(tangent.shape[1]):
        full_tangent = np.concatenate(
            (tangent[:, column], np.zeros_like(state))
        )
        responses.append(
            matrix_derivative_to_closed_scalar(
                full_state_matrix_derivative(
                    full_state,
                    full_tangent,
                    relative_dimension=relative_dimension,
                )
            )
        )
    return np.asarray(responses, dtype=float).T


def run(
    member_directories: dict[str, Path],
    output_directory: Path,
    *,
    coordinate_scales_path: Path = DEFAULT_SCALES,
    maximum_time: float = 40.0,
    sample_step: float = 1.0,
    finite_difference_step: float = 2e-6,
    geometric_relative_threshold: float = 1e-10,
) -> dict[str, object]:
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty output directory {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    if maximum_time <= 0.0 or sample_step <= 0.0:
        raise ValueError("maximum_time and sample_step must be positive")
    if finite_difference_step <= 0.0:
        raise ValueError("finite_difference_step must be positive")
    summaries = {
        label: json.loads(
            (directory / "summary.json").read_text(encoding="utf-8")
        )
        for label, directory in member_directories.items()
    }
    contracts = {
        label: _physical_contract(summary["parameters"])
        for label, summary in summaries.items()
    }
    first_contract = next(iter(contracts.values()))
    if any(contract != first_contract for contract in contracts.values()):
        raise ValueError("stored members do not share one physical contract")
    dimer_parameters = DimerParameters(
        hopping=float(first_contract["hopping"]),
        gamma=float(first_contract["gamma"]),
        lambda_ep=float(first_contract["lambda_ep"]),
        drive_amplitude=float(first_contract["drive_amplitude"]),
        pulse_width=float(first_contract["pulse_width"]),
    )
    drive_data = first_contract["drive_protocol"]
    if not isinstance(drive_data, dict):
        raise ValueError("invalid drive protocol")
    drive = GaussianSineDrive(
        amplitude=float(drive_data["amplitude"]),
        pulse_width=float(drive_data["pulse_width"]),
        delays=tuple(float(value) for value in drive_data["delays"]),
    )
    with np.load(coordinate_scales_path, allow_pickle=False) as arrays:
        coordinate_scales = np.asarray(
            arrays["coordinate_scales"],
            dtype=float,
        )
    if coordinate_scales.shape != (31,) or np.any(coordinate_scales <= 0.0):
        raise ValueError("coordinate scales must be positive with shape (31,)")
    sample_times = np.arange(
        0.0,
        maximum_time + 0.5 * sample_step,
        sample_step,
    )
    plan: dict[str, object] = {
        "schema": "paper_v_integrable_mixed_layer_gate_plan_v2",
        "run_id": output_directory.name,
        "classification": "offline_stored_state_construction_gate",
        "evidence_status": "exploratory_local_not_promoted",
        "member_labels": list(member_directories),
        "maximum_time": maximum_time,
        "sample_step": sample_step,
        "sample_count_per_member": int(sample_times.size),
        "coordinate_scales_path": str(coordinate_scales_path),
        "finite_difference_step": finite_difference_step,
        "geometric_relative_threshold": geometric_relative_threshold,
        "online_exact_reference_used": False,
        "autonomous_rollout_executed": False,
    }
    _write_json(output_directory / "plan.json", plan)
    started = time.time()

    count = len(member_directories)
    shape = (count, sample_times.size)
    state_error = np.empty(shape)
    truncated_tangent_difference = np.empty(shape)
    truncated_tangent_relative_difference = np.empty(shape)
    finite_difference_relative_error = np.empty((*shape, 12))
    residualized_projector_error = np.empty(shape)
    residualized_gram_error = np.empty(shape)
    closed_response_error = np.empty(shape)
    residualized_rank = np.empty(shape, dtype=int)
    packet_count_samples = np.empty(shape, dtype=int)
    top_population = np.empty(shape)
    local_shape = (*shape, len(LOCAL_SPACE_NAMES))
    local_hilbert_residual = np.empty(local_shape)
    local_closed_error = np.empty(local_shape)
    local_correlation_error = np.empty(local_shape)
    local_rank = np.empty(local_shape, dtype=int)

    stored_paths: list[Path] = [coordinate_scales_path]
    relative_dimension: int | None = None
    for member_index, (label, directory) in enumerate(
        member_directories.items()
    ):
        arrays_path = directory / "segmented_horizon.npz"
        summary_path = directory / "summary.json"
        stored_paths.extend((arrays_path, summary_path))
        summary = summaries[label]
        cutoff = int(summary["parameters"]["phonon_cutoff"])
        member_dimension = 2 * cutoff + 1
        if relative_dimension is None:
            relative_dimension = member_dimension
        elif relative_dimension != member_dimension:
            raise ValueError("stored members use different phonon cutoffs")
        print(f"[mixed-layer] auditing {label}", flush=True)
        with np.load(arrays_path, allow_pickle=False) as arrays:
            stored_times = np.asarray(arrays["times"], dtype=float)
            parameter_trajectory = np.asarray(
                arrays["parameter_trajectory"],
                dtype=float,
            )
            packet_counts = np.asarray(
                arrays["packet_count_trajectory"],
                dtype=int,
            )
        indices = np.searchsorted(stored_times, sample_times)
        if np.any(indices >= stored_times.size) or not np.allclose(
            stored_times[indices],
            sample_times,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"{label} does not sample the requested times")

        for sample_index, source_index in enumerate(indices):
            packet_count = int(packet_counts[source_index])
            packet_count_samples[member_index, sample_index] = packet_count
            packet_parameters = parameter_trajectory[
                source_index,
                : 16 * packet_count,
            ]
            pilot = packet_archive_mixed_frames(
                packet_parameters,
                relative_dimension=member_dimension,
            )
            centers = mixed_layer_centers(
                packet_parameters,
                relative_dimension=member_dimension,
            )
            chart_state, chart_tangent = mixed_exponential_origin_tangent(
                packet_parameters,
                relative_dimension=member_dimension,
                centers=centers,
            )
            stored_tangent = pilot.relative_mixed_tangent[
                : 4 * member_dimension
            ]
            state_error[member_index, sample_index] = float(
                np.max(np.abs(chart_state - pilot.relative_state))
            )
            tangent_difference = chart_tangent - stored_tangent
            truncated_tangent_difference[member_index, sample_index] = float(
                np.max(np.abs(tangent_difference))
            )
            truncated_tangent_relative_difference[
                member_index,
                sample_index,
            ] = float(
                np.linalg.norm(tangent_difference)
                / max(
                    np.linalg.norm(stored_tangent),
                    np.finfo(float).tiny,
                )
            )

            for column in range(12):
                offset = np.zeros(12)
                offset[column] = finite_difference_step
                plus = mixed_exponential_layer_state(
                    packet_parameters,
                    offset,
                    relative_dimension=member_dimension,
                    centers=centers,
                ).state
                minus = mixed_exponential_layer_state(
                    packet_parameters,
                    -offset,
                    relative_dimension=member_dimension,
                    centers=centers,
                ).state
                derivative = (plus - minus) / (2.0 * finite_difference_step)
                derivative -= chart_state * np.vdot(chart_state, derivative)
                relative_error = float(
                    np.linalg.norm(derivative - chart_tangent[:, column])
                    / max(
                        np.linalg.norm(chart_tangent[:, column]),
                        np.finfo(float).tiny,
                    )
                )
                finite_difference_relative_error[
                    member_index,
                    sample_index,
                    column,
                ] = relative_error

            relative_base = np.column_stack(
                (
                    pilot.packet_tangent[: 4 * member_dimension],
                    pilot.archive_tangent[: 4 * member_dimension],
                )
            )
            base_basis = _range_basis(
                relative_base,
                geometric_relative_threshold,
            )
            chart_residual = _residualized_real_frame(
                chart_tangent,
                base_basis,
            )
            stored_residual = _residualized_real_frame(
                stored_tangent,
                base_basis,
            )
            chart_projector, chart_rank = _projector(
                chart_residual,
                geometric_relative_threshold,
            )
            stored_projector, stored_rank = _projector(
                stored_residual,
                geometric_relative_threshold,
            )
            if chart_rank != stored_rank:
                raise RuntimeError("chart and stored mixed supports have different rank")
            residualized_rank[member_index, sample_index] = chart_rank
            residualized_projector_error[member_index, sample_index] = float(
                np.linalg.norm(chart_projector - stored_projector, ord=2)
            )
            residualized_gram_error[member_index, sample_index] = float(
                np.max(
                    np.abs(
                        chart_residual.T @ chart_residual
                        - stored_residual.T @ stored_residual
                    )
                )
            )
            chart_response = _closed_response(
                chart_state,
                chart_tangent,
                member_dimension,
            )
            stored_response = _closed_response(
                pilot.relative_state,
                stored_tangent,
                member_dimension,
            )
            closed_response_error[member_index, sample_index] = float(
                np.max(np.abs(chart_response - stored_response))
            )
            top_population[member_index, sample_index] = (
                pilot.relative_top_population
            )

            hamiltonian = relative_holstein_hamiltonian(
                float(sample_times[sample_index]),
                dimer_parameters,
                relative_dimension=member_dimension,
                drive_protocol=drive,
            )
            energy = float(
                np.vdot(
                    pilot.relative_state,
                    hamiltonian @ pilot.relative_state,
                ).real
            )
            relative_target = -1j * (
                hamiltonian @ pilot.relative_state
                - energy * pilot.relative_state
            )
            full_target = np.concatenate(
                (relative_target, np.zeros_like(relative_target))
            )
            full_chart_tangent = np.vstack(
                (chart_tangent, np.zeros_like(chart_tangent))
            )
            full_base = np.column_stack(
                (pilot.packet_tangent, pilot.archive_tangent)
            )
            local_frames = (
                full_base,
                np.column_stack(
                    (full_base, pilot.relative_mixed_tangent)
                ),
                np.column_stack((full_base, full_chart_tangent)),
            )
            exact_closed_velocity = matrix_derivative_to_closed_scalar(
                full_state_matrix_derivative(
                    pilot.full_state,
                    full_target,
                    relative_dimension=member_dimension,
                )
            )
            for space_index, frame in enumerate(local_frames):
                projection = project_real_tangent(
                    full_target,
                    frame,
                    geometric_gram_relative_threshold=(
                        geometric_relative_threshold
                    ),
                )
                predicted_closed_velocity = (
                    matrix_derivative_to_closed_scalar(
                        full_state_matrix_derivative(
                            pilot.full_state,
                            projection.projected_velocity,
                            relative_dimension=member_dimension,
                        )
                    )
                )
                scaled_difference = (
                    predicted_closed_velocity - exact_closed_velocity
                ) / coordinate_scales
                local_hilbert_residual[
                    member_index,
                    sample_index,
                    space_index,
                ] = projection.relative_residual
                local_closed_error[
                    member_index,
                    sample_index,
                    space_index,
                ] = float(np.sqrt(np.mean(scaled_difference**2)))
                local_correlation_error[
                    member_index,
                    sample_index,
                    space_index,
                ] = float(
                    np.sqrt(np.mean(scaled_difference[17:31] ** 2))
                )
                local_rank[
                    member_index,
                    sample_index,
                    space_index,
                ] = projection.retained_rank

    if relative_dimension is None:
        raise RuntimeError("no stored members were supplied")
    np.savez_compressed(
        output_directory / "integrable_mixed_layer_gate.npz",
        member_labels=np.asarray(list(member_directories)),
        sample_times=sample_times,
        packet_count=packet_count_samples,
        state_max_abs_error=state_error,
        truncated_tangent_max_abs_difference=truncated_tangent_difference,
        truncated_tangent_relative_difference=(
            truncated_tangent_relative_difference
        ),
        finite_difference_relative_error=finite_difference_relative_error,
        residualized_projector_error=residualized_projector_error,
        residualized_gram_max_abs_error=residualized_gram_error,
        closed_response_max_abs_error=closed_response_error,
        residualized_rank=residualized_rank,
        relative_top_population=top_population,
        local_space_names=np.asarray(LOCAL_SPACE_NAMES),
        local_hilbert_relative_residual=local_hilbert_residual,
        local_closed_coordinate_scaled_rms=local_closed_error,
        local_correlation_scaled_rms=local_correlation_error,
        local_retained_rank=local_rank,
        coordinate_scales=coordinate_scales,
    )

    thresholds = {
        "state_max_abs_error": 1e-11,
        "finite_difference_relative_error": 1e-6,
    }
    maxima = {
        "state_max_abs_error": float(np.max(state_error)),
        "truncated_tangent_max_abs_difference": float(
            np.max(truncated_tangent_difference)
        ),
        "truncated_tangent_relative_difference": float(
            np.max(truncated_tangent_relative_difference)
        ),
        "finite_difference_relative_error": float(
            np.max(finite_difference_relative_error)
        ),
        "residualized_projector_error": float(
            np.max(residualized_projector_error)
        ),
        "residualized_gram_max_abs_error": float(
            np.max(residualized_gram_error)
        ),
        "closed_response_max_abs_error": float(
            np.max(closed_response_error)
        ),
        "relative_top_population": float(np.max(top_population)),
    }
    gate_passed = all(
        maxima[name] <= threshold for name, threshold in thresholds.items()
    )
    aggregate_local_scores = {}
    for space_index, name in enumerate(LOCAL_SPACE_NAMES):
        aggregate_local_scores[name] = {
            "hilbert_relative_residual_rms": _rms(
                local_hilbert_residual[..., space_index]
            ),
            "closed_coordinate_scaled_rms": _rms(
                local_closed_error[..., space_index]
            ),
            "correlation_scaled_rms": _rms(
                local_correlation_error[..., space_index]
            ),
            "retained_rank_minimum": int(
                np.min(local_rank[..., space_index])
            ),
            "retained_rank_maximum": int(
                np.max(local_rank[..., space_index])
            ),
        }
    base_index = LOCAL_SPACE_NAMES.index("packet_archive")
    truncated_index = LOCAL_SPACE_NAMES.index(
        "packet_archive_truncated_mixed"
    )
    analytic_index = LOCAL_SPACE_NAMES.index(
        "packet_archive_analytic_mixed"
    )
    aggregate_local_reductions = {
        "truncated_mixed_hilbert": _fractional_reduction(
            local_hilbert_residual[..., base_index],
            local_hilbert_residual[..., truncated_index],
        ),
        "analytic_mixed_hilbert": _fractional_reduction(
            local_hilbert_residual[..., base_index],
            local_hilbert_residual[..., analytic_index],
        ),
        "truncated_mixed_closed_coordinates": _fractional_reduction(
            local_closed_error[..., base_index],
            local_closed_error[..., truncated_index],
        ),
        "analytic_mixed_closed_coordinates": _fractional_reduction(
            local_closed_error[..., base_index],
            local_closed_error[..., analytic_index],
        ),
        "truncated_mixed_correlation": _fractional_reduction(
            local_correlation_error[..., base_index],
            local_correlation_error[..., truncated_index],
        ),
        "analytic_mixed_correlation": _fractional_reduction(
            local_correlation_error[..., base_index],
            local_correlation_error[..., analytic_index],
        ),
    }
    member_summaries = {}
    for index, label in enumerate(member_directories):
        local_scores = {}
        for space_index, name in enumerate(LOCAL_SPACE_NAMES):
            local_scores[name] = {
                "hilbert_relative_residual_rms": _rms(
                    local_hilbert_residual[index, :, space_index]
                ),
                "closed_coordinate_scaled_rms": _rms(
                    local_closed_error[index, :, space_index]
                ),
                "correlation_scaled_rms": _rms(
                    local_correlation_error[index, :, space_index]
                ),
                "retained_rank_minimum": int(
                    np.min(local_rank[index, :, space_index])
                ),
                "retained_rank_maximum": int(
                    np.max(local_rank[index, :, space_index])
                ),
            }
        member_summaries[label] = {
            "packet_count_minimum": int(np.min(packet_count_samples[index])),
            "packet_count_maximum": int(np.max(packet_count_samples[index])),
            "residualized_rank_values": sorted(
                {int(value) for value in residualized_rank[index]}
            ),
            "state_max_abs_error": float(np.max(state_error[index])),
            "truncated_tangent_max_abs_difference": float(
                np.max(truncated_tangent_difference[index])
            ),
            "truncated_tangent_relative_difference": float(
                np.max(truncated_tangent_relative_difference[index])
            ),
            "finite_difference_relative_error": float(
                np.max(finite_difference_relative_error[index])
            ),
            "closed_response_max_abs_error": float(
                np.max(closed_response_error[index])
            ),
            "local_scores": local_scores,
            "analytic_mixed_fractional_reductions": {
                "hilbert": _fractional_reduction(
                    local_hilbert_residual[index, :, base_index],
                    local_hilbert_residual[index, :, analytic_index],
                ),
                "closed_coordinates": _fractional_reduction(
                    local_closed_error[index, :, base_index],
                    local_closed_error[index, :, analytic_index],
                ),
                "correlation": _fractional_reduction(
                    local_correlation_error[index, :, base_index],
                    local_correlation_error[index, :, analytic_index],
                ),
            },
        }
    summary: dict[str, object] = {
        "schema": "paper_v_integrable_mixed_layer_gate_summary_v2",
        "run_id": output_directory.name,
        "classification": "offline_stored_state_construction_gate",
        "evidence_status": "exploratory_local_not_promoted",
        "status": "complete",
        "gate_passed": gate_passed,
        "numerical_verification_thresholds": thresholds,
        "global_maxima": maxima,
        "aggregate_local_scores": aggregate_local_scores,
        "aggregate_local_fractional_reductions": (
            aggregate_local_reductions
        ),
        "members": member_summaries,
        "relative_dimension": relative_dimension,
        "interpretation": (
            "Passing verifies the state origin and all twelve real origin "
            "derivatives of the normalized analytic-before-cutoff finite "
            "packet-union chart. The truncated-operator pilot is reported as "
            "a separate finite-cutoff comparator and is not required to "
            "coincide with the analytic chart. Local tangent projection scores "
            "measure whether the corrected analytic layer retains the stored "
            "same-state gain; they do not establish autonomous accuracy."
        ),
        "online_exact_reference_used": False,
        "autonomous_rollout_executed": False,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(output_directory / "summary.json", summary)

    repo_root = Path(__file__).resolve().parents[2]
    source_files = (
        Path(__file__).resolve(),
        repo_root / "paper_5/src/paper5/stability/mixed_exponential_layer.py",
        repo_root / "paper_5/src/paper5/stability/archive_gram_tangent_pilot.py",
    )
    artifacts = (
        output_directory / "plan.json",
        output_directory / "summary.json",
        output_directory / "integrable_mixed_layer_gate.npz",
    )
    manifest: dict[str, object] = {
        "schema": "paper_v_integrable_mixed_layer_gate_manifest_v2",
        "run_id": output_directory.name,
        "status": "complete",
        "python": sys.version,
        "platform": platform.platform(),
        "stored_input_hashes": {
            str(path): _sha256(path) for path in stored_paths
        },
        "source_hashes": {
            str(path.relative_to(repo_root)): _sha256(path)
            for path in source_files
        },
        "artifact_hashes": {path.name: _sha256(path) for path in artifacts},
    }
    _write_json(output_directory / "runtime_manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for label, path in DEFAULT_MEMBERS.items():
        parser.add_argument(f"--{label.lower()}-dir", type=Path, default=path)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--coordinate-scales",
        type=Path,
        default=DEFAULT_SCALES,
    )
    parser.add_argument("--maximum-time", type=float, default=40.0)
    parser.add_argument("--sample-step", type=float, default=1.0)
    parser.add_argument("--finite-difference-step", type=float, default=2e-6)
    parser.add_argument(
        "--geometric-relative-threshold",
        type=float,
        default=1e-10,
    )
    arguments = parser.parse_args()
    directories = {
        label: getattr(arguments, f"{label.lower()}_dir")
        for label in DEFAULT_MEMBERS
    }
    run(
        directories,
        arguments.output_directory,
        coordinate_scales_path=arguments.coordinate_scales,
        maximum_time=arguments.maximum_time,
        sample_step=arguments.sample_step,
        finite_difference_step=arguments.finite_difference_step,
        geometric_relative_threshold=arguments.geometric_relative_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
