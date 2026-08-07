"""Audit a packet-derived archive-closure source from stored trajectories."""

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

import numpy as np
from scipy.linalg import subspace_angles

from paper5.stability import DimerParameters, GaussianSineDrive
from paper5.stability.matrix_reference import pauli_repaired_closed_scalar_rhs
from paper5.stability.moment_hierarchy import moment_hierarchy
from paper5.stability.multi_coherent_scores import CLOSED_COORDINATE_BLOCKS
from paper5.stability.packet_derived_closure import (
    normalized_scaled_source_error,
    packet_closed_velocity_pair,
    reconstruct_frozen_source_subspace,
    scaled_source_fluctuation_rms,
)

RUN_ID = "paper_v_packet_derived_closure_source_cutoff16_20260804_v3"
MEMBERS = ("central", "plus", "minus")


class _ProtocolParameters:
    def __init__(self, base: DimerParameters, drive: GaussianSineDrive) -> None:
        self._base = base
        self._drive = drive

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def drive_difference(self, time_value: float) -> float:
        return self._drive.difference(time_value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _scaled_block_metrics(
    residual: np.ndarray,
    scales: np.ndarray,
) -> dict[str, dict[str, float]]:
    values = np.asarray(residual, dtype=float)
    scaled = values / scales
    squared_totals: dict[str, float] = {}
    metrics: dict[str, dict[str, float]] = {}
    for name, block in CLOSED_COORDINATE_BLOCKS.items():
        block_scaled = scaled[..., block]
        block_raw = values[..., block]
        squared = float(np.mean(np.sum(block_scaled**2, axis=-1)))
        squared_totals[name] = squared
        metrics[name] = {
            "raw_vector_rms": float(
                np.sqrt(np.mean(np.sum(block_raw**2, axis=-1)))
            ),
            "scaled_vector_rms": float(np.sqrt(squared)),
            "scaled_per_coordinate_rms": float(
                np.sqrt(np.mean(block_scaled**2))
            ),
        }
    total = max(sum(squared_totals.values()), np.finfo(float).tiny)
    for name in metrics:
        metrics[name]["scaled_squared_fraction"] = squared_totals[name] / total
    return metrics


def _rank_five_basis(source: np.ndarray, scales: np.ndarray) -> np.ndarray:
    values = np.asarray(source, dtype=float)
    centered = values - np.mean(values, axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered / scales, full_matrices=False)
    return right[:5]


def _window_source_metrics(
    mask: np.ndarray,
    projected_source: np.ndarray,
    schrodinger_source: np.ndarray,
    exact_source: np.ndarray,
    scales: np.ndarray,
    tangent_relative_residual: np.ndarray,
    packet_counts: np.ndarray,
) -> dict[str, float | int]:
    """Summarize one declared time window in its own exact-source scale."""

    exact_window = exact_source[:, mask]
    projected_window = projected_source[:, mask]
    schrodinger_window = schrodinger_source[:, mask]
    fluctuation = scaled_source_fluctuation_rms(exact_window, scales)
    return {
        "sample_count_per_member": int(np.count_nonzero(mask)),
        "exact_source_fluctuation_scale": fluctuation,
        "packet_projected_to_exact_path_nrms": (
            normalized_scaled_source_error(
                projected_window,
                exact_window,
                scales,
                reference_fluctuation_scale=fluctuation,
            )
        ),
        "packet_schrodinger_to_exact_path_nrms": (
            normalized_scaled_source_error(
                schrodinger_window,
                exact_window,
                scales,
                reference_fluctuation_scale=fluctuation,
            )
        ),
        "projected_to_same_state_schrodinger_nrms": (
            normalized_scaled_source_error(
                projected_window,
                schrodinger_window,
                scales,
                reference_fluctuation_scale=fluctuation,
            )
        ),
        "tangent_relative_residual_rms": float(
            np.sqrt(np.mean(tangent_relative_residual[:, mask] ** 2))
        ),
        "minimum_packet_count": int(np.min(packet_counts[:, mask])),
        "maximum_packet_count": int(np.max(packet_counts[:, mask])),
    }


def run(
    batch_directory: Path,
    exact_source_artifact: Path,
    output_directory: Path,
    *,
    members: tuple[str, ...] = MEMBERS,
    maximum_time: float | None = None,
    relative_damping_override: float | None = None,
    adaptive_directories: dict[str, Path] | None = None,
) -> dict[str, object]:
    if not members or len(set(members)) != len(members):
        raise ValueError("members must be a nonempty set of unique names")
    if any(member not in MEMBERS for member in members):
        raise ValueError(f"members must be selected from {MEMBERS}")
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty output directory {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    source_arrays_path = (
        exact_source_artifact / "trajectory_closure_identifiability.npz"
    )
    if not source_arrays_path.is_file():
        raise FileNotFoundError(source_arrays_path)
    adaptive_mode = adaptive_directories is not None
    if adaptive_mode:
        if adaptive_directories is None:
            raise RuntimeError("adaptive trajectory mapping was not initialized")
        if set(adaptive_directories) != set(members):
            raise ValueError(
                "adaptive_directories must provide exactly the selected members"
            )
        member_paths = tuple(
            adaptive_directories[member] / "mixed_guided_packet_rollout.npz"
            for member in members
        )
        summary_paths = tuple(
            adaptive_directories[member] / "summary.json"
            for member in members
        )
    else:
        member_paths = tuple(
            batch_directory / f"fine_{member}" / "segmented_horizon.npz"
            for member in members
        )
        summary_paths = tuple(
            batch_directory / f"fine_{member}" / "summary.json"
            for member in members
        )
    if not all(path.is_file() for path in (*member_paths, *summary_paths)):
        raise FileNotFoundError("stored fine packet trajectories are incomplete")

    summaries = [json.loads(path.read_text()) for path in summary_paths]
    settings = summaries[0]["parameters"]
    if adaptive_mode:
        common_keys = (
            "lambda_ep",
            "gamma",
            "phonon_cutoff",
            "drive_protocol",
            "relative_damping",
            "sample_step",
        )
        if any(
            any(
                summary["parameters"].get(key) != settings.get(key)
                for key in common_keys
            )
            for summary in summaries[1:]
        ):
            raise ValueError(
                "adaptive packet members do not share the physical settings contract"
            )
    elif any(summary["parameters"] != settings for summary in summaries[1:]):
        raise ValueError("packet members do not share one settings contract")
    drive_data = settings["drive_protocol"]
    parameters = DimerParameters(
        hopping=float(settings.get("hopping", 1.0)),
        gamma=float(settings["gamma"]),
        lambda_ep=float(settings["lambda_ep"]),
        drive_amplitude=float(
            settings.get("drive_amplitude", drive_data["amplitude"])
        ),
        pulse_width=float(
            settings.get("pulse_width", drive_data["pulse_width"])
        ),
    )
    drive = GaussianSineDrive(
        amplitude=float(drive_data["amplitude"]),
        pulse_width=float(drive_data["pulse_width"]),
        delays=tuple(float(value) for value in drive_data["delays"]),
    )
    protocol_parameters = _ProtocolParameters(parameters, drive)
    projection_relative_damping = (
        float(settings["relative_damping"])
        if relative_damping_override is None
        else float(relative_damping_override)
    )
    if projection_relative_damping <= 0.0:
        raise ValueError("relative_damping_override must be positive")
    phonon_cutoff = int(settings["phonon_cutoff"])
    relative_dimension = 2 * phonon_cutoff + 1
    hierarchy = moment_hierarchy(4)
    center_amplitude = (
        -np.sqrt(2.0) * parameters.coupling / parameters.omega_ph
    )
    tangent_singular_value_cutoff = float(
        settings.get(
            "tangent_singular_value_cutoff",
            1e-9 if adaptive_mode else 1e-2,
        )
    )
    tangent_regularization = str(
        settings.get("tangent_regularization", "tikhonov")
    )

    with np.load(source_arrays_path) as source_arrays:
        sample_indices = np.asarray(source_arrays["sample_indices"], dtype=int)
        times = np.asarray(source_arrays["times"], dtype=float)
        scales = np.asarray(source_arrays["coordinate_scales"], dtype=float)
        all_exact_source = np.asarray(
            source_arrays["dop853_target_source"], dtype=float
        )
        exact_basis = np.asarray(
            source_arrays["source_subspace_basis"], dtype=float
        )
        exact_center = np.asarray(
            source_arrays["source_subspace_center"], dtype=float
        )
    if all_exact_source.shape != (len(MEMBERS), times.size, 14):
        raise ValueError("exact source artifact has an unexpected shape")
    member_indices = np.asarray(
        [MEMBERS.index(member) for member in members],
        dtype=int,
    )
    with np.load(member_paths[0]) as first_member_arrays:
        packet_final_time = float(first_member_arrays["times"][-1])
    if maximum_time is not None:
        if maximum_time <= 0.0:
            raise ValueError("maximum_time must be positive")
        packet_final_time = min(packet_final_time, float(maximum_time))
    time_mask = times <= packet_final_time + 1e-12
    times = times[time_mask]
    sample_indices = sample_indices[time_mask]
    exact_source = all_exact_source[member_indices][:, time_mask]

    shape = (len(members), times.size, 31)
    stored_closed = np.empty(shape, dtype=float)
    reconstructed_closed = np.empty(shape, dtype=float)
    projected_derivative = np.empty(shape, dtype=float)
    schrodinger_derivative = np.empty(shape, dtype=float)
    archive_derivative = np.empty(shape, dtype=float)
    tangent_relative = np.empty(shape[:2], dtype=float)
    tangent_absolute = np.empty(shape[:2], dtype=float)
    tangent_rank = np.empty(shape[:2], dtype=int)
    geometric_rank = np.empty(shape[:2], dtype=int)
    parameter_speed = np.empty(shape[:2], dtype=float)
    hierarchy_coordinate_error = np.empty(shape[:2], dtype=float)
    sampled_packet_counts = np.empty(shape[:2], dtype=int)

    started = time.time()
    for member_index, (member, path) in enumerate(zip(members, member_paths)):
        print(f"contracting stored packet member: {member}", flush=True)
        with np.load(path) as arrays:
            all_times = np.asarray(arrays["times"], dtype=float)
            if not np.allclose(all_times[sample_indices], times, atol=1e-12, rtol=0.0):
                raise ValueError("packet and exact-source time grids differ")
            parameter_trajectory = np.asarray(
                arrays["parameter_trajectory"], dtype=float
            )
            packet_counts = np.asarray(
                arrays["packet_count_trajectory"], dtype=int
            )
            if adaptive_mode:
                hierarchy_trajectory = None
                stored_closed[member_index] = np.asarray(
                    arrays["mixed_guided_closed_coordinates"][sample_indices],
                    dtype=float,
                )
            else:
                hierarchy_trajectory = np.asarray(
                    arrays["coordinates"], dtype=float
                )
                stored_closed[member_index] = np.asarray(
                    arrays["closed_coordinates"][sample_indices], dtype=float
                )

        for local_index, source_index in enumerate(sample_indices):
            packet_count = int(packet_counts[source_index])
            sampled_packet_counts[member_index, local_index] = packet_count
            width = 16 * packet_count
            packed = parameter_trajectory[source_index, :width]
            result = packet_closed_velocity_pair(
                packed,
                time=float(times[local_index]),
                parameters=parameters,
                drive_protocol=drive,
                relative_dimension=relative_dimension,
                hierarchy=hierarchy,
                center_amplitude=complex(center_amplitude),
                center_derivative=0.0j,
                hierarchy_coordinates=(
                    None
                    if hierarchy_trajectory is None
                    else hierarchy_trajectory[source_index]
                ),
                tangent_singular_value_cutoff=tangent_singular_value_cutoff,
                tangent_regularization=tangent_regularization,
                relative_damping=projection_relative_damping,
            )
            reconstructed_closed[member_index, local_index] = (
                result.closed_coordinates
            )
            projected_derivative[member_index, local_index] = (
                result.projected_closed_velocity
            )
            schrodinger_derivative[member_index, local_index] = (
                result.schrodinger_closed_velocity
            )
            tangent_relative[member_index, local_index] = (
                result.tangent_relative_residual
            )
            tangent_absolute[member_index, local_index] = (
                result.tangent_absolute_residual
            )
            tangent_rank[member_index, local_index] = result.tangent_rank
            geometric_rank[member_index, local_index] = (
                result.geometric_tangent_rank
            )
            parameter_speed[member_index, local_index] = (
                result.parameter_velocity_norm
            )
            hierarchy_coordinate_error[member_index, local_index] = (
                result.hierarchy_coordinate_max_error
            )
            archive_derivative[member_index, local_index] = (
                pauli_repaired_closed_scalar_rhs(
                    float(times[local_index]),
                    stored_closed[member_index, local_index],
                    protocol_parameters,  # type: ignore[arg-type]
                )
            )

    reconstruction_error = float(
        np.max(np.abs(reconstructed_closed - stored_closed))
    )
    projected_residual = projected_derivative - archive_derivative
    schrodinger_residual = schrodinger_derivative - archive_derivative
    projected_c = projected_residual[..., 17:31]
    schrodinger_c = schrodinger_residual[..., 17:31]
    c_scales = scales[17:31]
    exact_fluctuation = scaled_source_fluctuation_rms(exact_source, c_scales)
    projected_fluctuation = scaled_source_fluctuation_rms(projected_c, c_scales)
    schrodinger_fluctuation = scaled_source_fluctuation_rms(
        schrodinger_c, c_scales
    )
    same_state_source_error_exact_scale = normalized_scaled_source_error(
        projected_c,
        schrodinger_c,
        c_scales,
        reference_fluctuation_scale=exact_fluctuation,
    )
    same_state_source_error_own_scale = normalized_scaled_source_error(
        projected_c,
        schrodinger_c,
        c_scales,
        reference_fluctuation_scale=schrodinger_fluctuation,
    )
    projected_to_exact_path = normalized_scaled_source_error(
        projected_c,
        exact_source,
        c_scales,
        reference_fluctuation_scale=exact_fluctuation,
    )
    schrodinger_to_exact_path = normalized_scaled_source_error(
        schrodinger_c,
        exact_source,
        c_scales,
        reference_fluctuation_scale=exact_fluctuation,
    )

    exact_q5 = exact_basis[:5]
    projected_q5_coefficients, projected_q5 = (
        reconstruct_frozen_source_subspace(
            projected_c, c_scales, exact_center, exact_q5
        )
    )
    schrodinger_q5_coefficients, schrodinger_q5 = (
        reconstruct_frozen_source_subspace(
            schrodinger_c, c_scales, exact_center, exact_q5
        )
    )
    exact_q5_coefficients, exact_q5_reconstruction = (
        reconstruct_frozen_source_subspace(
            exact_source, c_scales, exact_center, exact_q5
        )
    )
    q5_projected_error = normalized_scaled_source_error(
        projected_q5,
        projected_c,
        c_scales,
        reference_fluctuation_scale=exact_fluctuation,
    )
    q5_schrodinger_error = normalized_scaled_source_error(
        schrodinger_q5,
        schrodinger_c,
        c_scales,
        reference_fluctuation_scale=exact_fluctuation,
    )
    q5_exact_error = normalized_scaled_source_error(
        exact_q5_reconstruction,
        exact_source,
        c_scales,
        reference_fluctuation_scale=exact_fluctuation,
    )
    q5_projected_own_scale_error = normalized_scaled_source_error(
        projected_q5,
        projected_c,
        c_scales,
        reference_fluctuation_scale=projected_fluctuation,
    )
    q5_schrodinger_own_scale_error = normalized_scaled_source_error(
        schrodinger_q5,
        schrodinger_c,
        c_scales,
        reference_fluctuation_scale=schrodinger_fluctuation,
    )
    projected_basis = _rank_five_basis(projected_c[0], c_scales)
    schrodinger_basis = _rank_five_basis(schrodinger_c[0], c_scales)
    projected_angles = np.degrees(
        subspace_angles(exact_q5.T, projected_basis.T)
    )
    schrodinger_angles = np.degrees(
        subspace_angles(exact_q5.T, schrodinger_basis.T)
    )
    temporal_windows = {}
    declared_windows = {
        "first_pulse_0_to_8": (times >= 0.0) & (times <= 8.0),
        "both_pulses_0_to_20": (times >= 0.0) & (times <= 20.0),
        "post_pulse_20_to_100": (times > 20.0) & (times <= 100.0),
    }
    for window_name, window_mask in declared_windows.items():
        if not np.any(window_mask):
            continue
        temporal_windows[window_name] = _window_source_metrics(
            window_mask,
            projected_c,
            schrodinger_c,
            exact_source,
            c_scales,
            tangent_relative,
            sampled_packet_counts,
        )
    scaled_projection_defect_norm = np.linalg.norm(
        (projected_c - schrodinger_c) / c_scales,
        axis=-1,
    ).ravel()
    source_tangent_correlation = float(
        np.corrcoef(scaled_projection_defect_norm, tangent_relative.ravel())[0, 1]
    )

    per_member = {}
    for member_index, member in enumerate(members):
        per_member[member] = {
            "projected_to_same_state_schrodinger_source_nrms": (
                normalized_scaled_source_error(
                    projected_c[member_index],
                    schrodinger_c[member_index],
                    c_scales,
                    reference_fluctuation_scale=exact_fluctuation,
                )
            ),
            "projected_to_exact_path_source_nrms": (
                normalized_scaled_source_error(
                    projected_c[member_index],
                    exact_source[member_index],
                    c_scales,
                    reference_fluctuation_scale=exact_fluctuation,
                )
            ),
            "schrodinger_packet_to_exact_path_source_nrms": (
                normalized_scaled_source_error(
                    schrodinger_c[member_index],
                    exact_source[member_index],
                    c_scales,
                    reference_fluctuation_scale=exact_fluctuation,
                )
            ),
        }

    metrics: dict[str, object] = {
        "schema": "paper_v_packet_derived_closure_source_summary_v1",
        "run_id": output_directory.name,
        "classification": "diagnostic",
        "evidence_status": "exploratory_stored_data_not_promoted",
        "status": "complete",
        "baseline": "autonomous same-spin Pauli-repaired 31-coordinate EOM",
        "projection_settings": {
            "regularization": tangent_regularization,
            "tangent_singular_value_cutoff": tangent_singular_value_cutoff,
            "stored_trajectory_relative_damping": float(
                settings["relative_damping"]
            ),
            "evaluated_relative_damping": projection_relative_damping,
        },
        "members": list(members),
        "sample_count_per_member": int(times.size),
        "maximum_closed_reconstruction_error": reconstruction_error,
        "maximum_hierarchy_coordinate_reconstruction_error": float(
            np.max(hierarchy_coordinate_error)
        ),
        "source_fluctuation_scales": {
            "exact_path": exact_fluctuation,
            "packet_projected": projected_fluctuation,
            "packet_schrodinger": schrodinger_fluctuation,
        },
        "source_errors": {
            "projected_to_same_state_schrodinger_nrms_normalized_by_exact_path": (
                same_state_source_error_exact_scale
            ),
            "projected_to_same_state_schrodinger_nrms_normalized_by_packet_schrodinger": (
                same_state_source_error_own_scale
            ),
            "projected_to_exact_path_nrms": projected_to_exact_path,
            "schrodinger_packet_to_exact_path_nrms": schrodinger_to_exact_path,
            "per_member": per_member,
        },
        "temporal_source_errors": temporal_windows,
        "projection_defect_to_tangent_residual_pearson_correlation": (
            source_tangent_correlation
        ),
        "frozen_exact_q5": {
            "exact_path_reconstruction_nrms": q5_exact_error,
            "packet_projected_source_reconstruction_nrms_normalized_by_exact_path": (
                q5_projected_error
            ),
            "packet_projected_source_reconstruction_nrms_normalized_by_own_fluctuation": (
                q5_projected_own_scale_error
            ),
            "packet_schrodinger_source_reconstruction_nrms_normalized_by_exact_path": (
                q5_schrodinger_error
            ),
            "packet_schrodinger_source_reconstruction_nrms_normalized_by_own_fluctuation": (
                q5_schrodinger_own_scale_error
            ),
            "principal_angles_degrees_to_projected_packet_basis": (
                projected_angles.tolist()
            ),
            "principal_angles_degrees_to_schrodinger_packet_basis": (
                schrodinger_angles.tolist()
            ),
        },
        "full_residual_block_metrics": {
            "packet_projected": _scaled_block_metrics(
                projected_residual, scales
            ),
            "packet_schrodinger": _scaled_block_metrics(
                schrodinger_residual, scales
            ),
            "projection_defect": _scaled_block_metrics(
                projected_derivative - schrodinger_derivative,
                scales,
            ),
        },
        "tangent": {
            "maximum_relative_residual": float(np.max(tangent_relative)),
            "rms_relative_residual": float(
                np.sqrt(np.mean(tangent_relative**2))
            ),
            "maximum_absolute_residual": float(np.max(tangent_absolute)),
            "minimum_tangent_rank": int(np.min(tangent_rank)),
            "maximum_tangent_rank": int(np.max(tangent_rank)),
            "minimum_geometric_rank": int(np.min(geometric_rank)),
            "maximum_geometric_rank": int(np.max(geometric_rank)),
            "maximum_parameter_speed": float(np.max(parameter_speed)),
        },
        "scope": {
            "new_propagation": False,
            "new_timing": False,
            "sealed_rescore": False,
            "distinct_drive_tested": False,
            "trajectory_source": (
                "adaptive mixed-guided packet trajectories"
                if adaptive_mode
                else "fixed-capacity packet trajectories"
            ),
            "interpretation": (
                "matched stored preparation members under one double-pulse drive"
            ),
        },
        "elapsed_seconds": time.time() - started,
    }

    arrays_path = output_directory / "packet_derived_source_arrays.npz"
    np.savez_compressed(
        arrays_path,
        times=times,
        sample_indices=sample_indices,
        coordinate_scales=scales,
        stored_closed=stored_closed,
        reconstructed_closed=reconstructed_closed,
        projected_closed_derivative=projected_derivative,
        schrodinger_closed_derivative=schrodinger_derivative,
        archive_closed_derivative=archive_derivative,
        projected_residual=projected_residual,
        schrodinger_residual=schrodinger_residual,
        exact_path_source=exact_source,
        projected_q5_reconstruction=projected_q5,
        schrodinger_q5_reconstruction=schrodinger_q5,
        projected_q5_coefficients=projected_q5_coefficients,
        schrodinger_q5_coefficients=schrodinger_q5_coefficients,
        exact_q5_coefficients=exact_q5_coefficients,
        tangent_relative_residual=tangent_relative,
        tangent_absolute_residual=tangent_absolute,
        tangent_rank=tangent_rank,
        geometric_tangent_rank=geometric_rank,
        parameter_speed=parameter_speed,
        hierarchy_coordinate_reconstruction_error=hierarchy_coordinate_error,
        packet_count=sampled_packet_counts,
    )
    summary_path = output_directory / "summary.json"
    _write_json_atomic(summary_path, metrics)

    repo_root = Path(__file__).resolve().parents[2]
    source_paths = (
        Path(__file__).resolve(),
        repo_root / "paper_5/src/paper5/stability/packet_derived_closure.py",
        repo_root / "paper_5/src/paper5/stability/multi_coherent.py",
        repo_root / "paper_5/src/paper5/stability/matrix_reference.py",
    )
    input_paths = (*member_paths, *summary_paths, source_arrays_path)
    manifest: dict[str, object] = {
        "schema": "paper_v_packet_derived_closure_source_manifest_v1",
        "run_id": output_directory.name,
        "status": "complete",
        "classification": "diagnostic",
        "evidence_status": "exploratory_stored_data_not_promoted",
        "python": sys.version,
        "platform": platform.platform(),
        "source_hashes": {
            str(path.relative_to(repo_root)): _sha256(path)
            for path in source_paths
        },
        "input_hashes": {str(path): _sha256(path) for path in input_paths},
        "artifact_hashes": {
            arrays_path.name: _sha256(arrays_path),
            summary_path.name: _sha256(summary_path),
        },
    }
    _write_json_atomic(output_directory / "runtime_manifest.json", manifest)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-directory",
        type=Path,
        default=Path("output/local_runs")
        / "paper_v_multi_coherent_double_pulse_blind_model_cutoff16_20260804_v1",
    )
    parser.add_argument(
        "--exact-source-artifact",
        type=Path,
        default=Path("output/local_runs")
        / "paper_v_trajectory_closure_identifiability_cutoff16_20260804_v1",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("output/local_runs") / RUN_ID,
    )
    parser.add_argument(
        "--members",
        nargs="+",
        choices=MEMBERS,
        default=list(MEMBERS),
    )
    parser.add_argument(
        "--adaptive-member",
        action="append",
        default=[],
        metavar="MEMBER=DIRECTORY",
        help=(
            "Read an adaptive mixed-guided trajectory instead of the fixed "
            "batch member. Supply once for every selected member."
        ),
    )
    parser.add_argument("--maximum-time", type=float)
    parser.add_argument("--relative-damping-override", type=float)
    args = parser.parse_args()
    adaptive_directories: dict[str, Path] | None = None
    if args.adaptive_member:
        adaptive_directories = {}
        for specification in args.adaptive_member:
            member, separator, directory = specification.partition("=")
            if not separator or member not in MEMBERS or not directory:
                raise SystemExit(
                    "--adaptive-member must have the form "
                    "central=PATH, plus=PATH, or minus=PATH"
                )
            if member in adaptive_directories:
                raise SystemExit(f"duplicate adaptive member: {member}")
            adaptive_directories[member] = Path(directory)
    result = run(
        args.batch_directory,
        args.exact_source_artifact,
        args.output_directory,
        members=tuple(args.members),
        maximum_time=args.maximum_time,
        relative_damping_override=args.relative_damping_override,
        adaptive_directories=adaptive_directories,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
