"""Test whether six mixed-tangent coefficients admit a fixed C decoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from paper5.stability.hubbard_dimer import DimerParameters, GaussianSineDrive
from paper5.stability.mixed_tangent_closure_identifiability import (
    MIXED_LABELS,
    mixed_tangent_closure_point,
)

RUN_ID = "paper_v_mixed_tangent_closure_identifiability_cutoff16_t20_20260804_v2"


@dataclass(frozen=True)
class _PathContract:
    label: str
    directory: Path
    role: str


@dataclass(frozen=True)
class _LinearDecoder:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    source_mean: np.ndarray
    coefficient: np.ndarray
    alpha: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        standardized = (features - self.feature_mean) / self.feature_scale
        return self.source_mean + standardized @ self.coefficient


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _drive_from_summary(
    summary: dict[str, Any],
) -> tuple[DimerParameters, GaussianSineDrive]:
    settings = summary["parameters"]
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
    else:
        drive = GaussianSineDrive(
            amplitude=float(drive_data["amplitude"]),
            pulse_width=float(drive_data["pulse_width"]),
            delays=tuple(float(value) for value in drive_data["delays"]),
        )
    return parameters, drive


def _fit_decoder(
    features: np.ndarray,
    source: np.ndarray,
    source_scales: np.ndarray,
    *,
    alpha: float,
) -> _LinearDecoder:
    values = np.asarray(features, dtype=float)
    target = np.asarray(source, dtype=float) / source_scales
    feature_mean = np.mean(values, axis=0)
    feature_scale = np.std(values, axis=0)
    feature_scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)
    standardized = (values - feature_mean) / feature_scale
    source_mean = np.mean(target, axis=0)
    centered_source = target - source_mean
    gram = standardized.T @ standardized
    coefficient = np.linalg.solve(
        gram + float(alpha) * np.eye(gram.shape[0]),
        standardized.T @ centered_source,
    )
    return _LinearDecoder(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        source_mean=source_mean,
        coefficient=coefficient,
        alpha=float(alpha),
    )


def _source_nrms(
    prediction_scaled: np.ndarray,
    source: np.ndarray,
    source_scales: np.ndarray,
) -> float:
    reference = np.asarray(source, dtype=float) / source_scales
    prediction = np.asarray(prediction_scaled, dtype=float)
    centered = reference - np.mean(reference, axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    error = prediction - reference
    return float(
        np.sqrt(np.mean(np.sum(error * error, axis=1)))
        / max(scale, np.finfo(float).tiny)
    )


def _choose_alpha(
    features: np.ndarray,
    source: np.ndarray,
    source_scales: np.ndarray,
    groups: np.ndarray,
) -> tuple[float, dict[str, float]]:
    candidates = (1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    unique_groups = tuple(np.unique(groups))
    scores: dict[str, float] = {}
    for alpha in candidates:
        fold_scores = []
        for group in unique_groups:
            validation = groups == group
            model = _fit_decoder(
                features[~validation],
                source[~validation],
                source_scales,
                alpha=alpha,
            )
            fold_scores.append(
                _source_nrms(
                    model.predict(features[validation]),
                    source[validation],
                    source_scales,
                )
            )
        scores[f"{alpha:.0e}"] = float(np.mean(fold_scores))
    best = min(candidates, key=lambda value: scores[f"{value:.0e}"])
    return float(best), scores


def _flatten(paths: list[np.ndarray], indices: list[int]) -> np.ndarray:
    return np.concatenate([paths[index] for index in indices], axis=0)


def _variance_rank(values: np.ndarray, fractions: tuple[float, ...]) -> dict[str, int]:
    centered = values - np.mean(values, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = singular**2
    cumulative = np.cumsum(variance) / max(float(np.sum(variance)), np.finfo(float).tiny)
    return {
        f"rank_at_{fraction:.4f}": int(np.searchsorted(cumulative, fraction) + 1)
        for fraction in fractions
    }


def _response_block(
    archive_source: np.ndarray,
    mixed_response: np.ndarray,
    source_scales: np.ndarray,
) -> np.ndarray:
    archive_scaled = archive_source / source_scales
    response_scaled = mixed_response / source_scales[:, None]
    return np.concatenate(
        (archive_scaled, response_scaled.reshape(response_scaled.shape[0], -1)),
        axis=1,
    )


def _response_subspace_score(
    training_blocks: np.ndarray,
    evaluation_block: np.ndarray,
    evaluation_coefficients: np.ndarray,
    evaluation_source: np.ndarray,
    source_scales: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, object]:
    mean = np.mean(training_blocks, axis=0)
    _, singular_values, right = np.linalg.svd(
        training_blocks - mean,
        full_matrices=False,
    )
    numerical_rank = int(
        np.count_nonzero(
            singular_values
            > 1e-12 * max(float(singular_values[0]), 1.0)
        )
    )
    checkpoints = sorted(
        {
            value
            for value in (0, 1, 2, 3, 5, 8, 10, 12, 15, 20, 30, 40, 60, 80, 100, numerical_rank)
            if value <= numerical_rank
        }
    )
    selected = np.ones(evaluation_source.shape[0], dtype=bool) if mask is None else mask
    source_scores: list[float] = []
    block_errors: list[float] = []
    evaluation_centered = evaluation_block - mean
    evaluation_scale = float(
        np.sqrt(np.mean(np.sum(evaluation_centered[selected] ** 2, axis=1)))
    )
    for rank in range(numerical_rank + 1):
        if rank:
            basis = right[:rank]
            reconstructed = mean + (evaluation_centered @ basis.T) @ basis
        else:
            reconstructed = np.broadcast_to(mean, evaluation_block.shape)
        reconstructed_archive = reconstructed[:, :14]
        reconstructed_response = reconstructed[:, 14:].reshape(-1, 14, 12)
        predicted_scaled_source = reconstructed_archive + np.einsum(
            "nij,nj->ni",
            reconstructed_response,
            evaluation_coefficients,
        )
        source_scores.append(
            _source_nrms(
                predicted_scaled_source[selected],
                evaluation_source[selected],
                source_scales,
            )
        )
        difference = reconstructed[selected] - evaluation_block[selected]
        block_errors.append(
            float(
                np.sqrt(np.mean(np.sum(difference * difference, axis=1)))
                / max(evaluation_scale, np.finfo(float).tiny)
            )
        )
    threshold_ranks: dict[str, int | None] = {}
    for threshold in (0.2, 0.1, 0.05, 0.01):
        passing = np.flatnonzero(np.asarray(source_scores) <= threshold)
        threshold_ranks[f"source_nrms_le_{threshold:.2f}"] = (
            int(passing[0]) if passing.size else None
        )
    return {
        "training_numerical_rank": numerical_rank,
        "source_nrms_by_rank": {
            str(rank): source_scores[rank] for rank in checkpoints
        },
        "response_block_relative_error_by_rank": {
            str(rank): block_errors[rank] for rank in checkpoints
        },
        "minimum_rank": threshold_ranks,
        "full_training_subspace_source_nrms": source_scores[numerical_rank],
        "full_training_subspace_response_block_relative_error": (
            block_errors[numerical_rank]
        ),
    }


def _make_plot(
    path: Path,
    times: np.ndarray,
    labels: list[str],
    target_source: list[np.ndarray],
    archive_source: list[np.ndarray],
    mixed_source: list[np.ndarray],
    source_scales: np.ndarray,
    decoder_scores: dict[str, float],
    coefficients: list[np.ndarray],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), constrained_layout=True)
    central = 0
    axes[0, 0].plot(
        times,
        np.linalg.norm(target_source[central] / source_scales, axis=1),
        label="total missing source",
    )
    axes[0, 0].plot(
        times,
        np.linalg.norm(archive_source[central] / source_scales, axis=1),
        label="archive-frame part",
    )
    axes[0, 0].plot(
        times,
        np.linalg.norm(mixed_source[central] / source_scales, axis=1),
        label="mixed part",
    )
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel("scaled vector norm")
    axes[0, 0].legend(fontsize=8)

    for index, label in enumerate(labels[:4]):
        axes[0, 1].plot(
            times,
            np.linalg.norm(coefficients[index], axis=1),
            label=label,
        )
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel(r"$\|\eta\|_2$")
    axes[0, 1].legend(fontsize=7)

    names = list(decoder_scores)
    axes[1, 0].bar(np.arange(len(names)), [decoder_scores[name] for name in names])
    axes[1, 0].set_xticks(np.arange(len(names)), names, rotation=25, ha="right")
    axes[1, 0].set_ylabel("source NRMS")

    coefficient_values = np.concatenate(coefficients[:3], axis=0)
    singular = np.linalg.svd(
        coefficient_values - np.mean(coefficient_values, axis=0),
        compute_uv=False,
    )
    axes[1, 1].semilogy(np.arange(1, singular.size + 1), singular, "o-")
    axes[1, 1].set_xlabel("mixed-coefficient mode")
    axes[1, 1].set_ylabel("singular value")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(
    contracts: tuple[_PathContract, ...],
    coordinate_scale_artifact: Path,
    output_directory: Path,
    *,
    maximum_time: float,
    sample_step: float,
    geometric_gram_relative_threshold: float,
) -> dict[str, object]:
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    if maximum_time <= 0.0 or sample_step <= 0.0:
        raise ValueError("maximum_time and sample_step must be positive")
    with np.load(coordinate_scale_artifact) as arrays:
        coordinate_scales = np.asarray(arrays["coordinate_scales"], dtype=float)
    source_scales = coordinate_scales[17:31]
    times = np.arange(0.0, maximum_time + 0.5 * sample_step, sample_step)

    closed_paths: list[np.ndarray] = []
    coefficient_paths: list[np.ndarray] = []
    response_paths: list[np.ndarray] = []
    target_source_paths: list[np.ndarray] = []
    archive_source_paths: list[np.ndarray] = []
    mixed_source_paths: list[np.ndarray] = []
    unresolved_paths: list[np.ndarray] = []
    drive_paths: list[np.ndarray] = []
    ranks: list[np.ndarray] = []
    singular_ratios: list[np.ndarray] = []
    gram_errors: list[np.ndarray] = []
    top_populations: list[np.ndarray] = []
    packet_counts: list[np.ndarray] = []
    summaries: list[dict[str, Any]] = []
    input_paths: list[Path] = [coordinate_scale_artifact]
    started = time.time()

    for contract in contracts:
        print(f"factoring mixed response: {contract.label}", flush=True)
        summary_path = contract.directory / "summary.json"
        arrays_path = contract.directory / "segmented_horizon.npz"
        input_paths.extend((summary_path, arrays_path))
        summary = json.loads(summary_path.read_text())
        summaries.append(summary)
        parameters, drive = _drive_from_summary(summary)
        settings = summary["parameters"]
        relative_dimension = 2 * int(settings["phonon_cutoff"]) + 1
        with np.load(arrays_path) as arrays:
            stored_times = np.asarray(arrays["times"], dtype=float)
            parameter_trajectory = np.asarray(
                arrays["parameter_trajectory"], dtype=float
            )
            all_packet_counts = np.asarray(
                arrays["packet_count_trajectory"], dtype=int
            )
        indices = np.searchsorted(stored_times, times)
        if np.any(indices >= stored_times.size) or not np.allclose(
            stored_times[indices], times, atol=1e-12, rtol=0.0
        ):
            raise ValueError(f"{contract.label} does not contain requested times")

        closed = np.empty((times.size, 31))
        coefficients = np.empty((times.size, 12))
        responses = np.empty((times.size, 14, 12))
        target_source = np.empty((times.size, 14))
        archive_source = np.empty_like(target_source)
        mixed_source = np.empty_like(target_source)
        unresolved = np.empty_like(target_source)
        path_ranks = np.empty(times.size, dtype=int)
        path_ratios = np.empty(times.size)
        path_gram_errors = np.empty(times.size)
        path_top_population = np.empty(times.size)
        path_packet_counts = all_packet_counts[indices]
        for sample_index, source_index in enumerate(indices):
            count = int(all_packet_counts[source_index])
            point = mixed_tangent_closure_point(
                parameter_trajectory[source_index, : 16 * count],
                time=float(times[sample_index]),
                parameters=parameters,
                drive_protocol=drive,
                relative_dimension=relative_dimension,
                geometric_gram_relative_threshold=(
                    geometric_gram_relative_threshold
                ),
            )
            closed[sample_index] = point.closed_coordinates
            coefficients[sample_index] = point.mixed_coefficients
            responses[sample_index] = point.mixed_response
            target_source[sample_index] = point.target_source
            archive_source[sample_index] = point.archive_frame_source
            mixed_source[sample_index] = point.mixed_source
            unresolved[sample_index] = point.unresolved_source
            path_ranks[sample_index] = point.mixed_complement_rank
            path_ratios[sample_index] = (
                point.mixed_smallest_retained_singular_value
                / point.mixed_largest_singular_value
            )
            path_gram_errors[sample_index] = point.archive_gram_max_error
            path_top_population[sample_index] = point.relative_top_population
        closed_paths.append(closed)
        coefficient_paths.append(coefficients)
        response_paths.append(responses)
        target_source_paths.append(target_source)
        archive_source_paths.append(archive_source)
        mixed_source_paths.append(mixed_source)
        unresolved_paths.append(unresolved)
        drive_paths.append(
            np.asarray([drive.difference(float(value)) for value in times])[:, None]
        )
        ranks.append(path_ranks)
        singular_ratios.append(path_ratios)
        gram_errors.append(path_gram_errors)
        top_populations.append(path_top_population)
        packet_counts.append(path_packet_counts)

    labels = [contract.label for contract in contracts]
    roles = [contract.role for contract in contracts]
    training_indices = [index for index, role in enumerate(roles) if role == "train"]
    if len(training_indices) < 3:
        raise ValueError("at least three training paths are required")
    train_coefficients = _flatten(coefficient_paths, training_indices)
    train_source = _flatten(target_source_paths, training_indices)
    train_closed = _flatten(closed_paths, training_indices)
    train_drive = _flatten(drive_paths, training_indices)
    train_groups = np.concatenate(
        [np.full(times.size, index, dtype=int) for index in training_indices]
    )

    feature_sets = {
        "retained_state": np.column_stack((train_closed, train_drive)),
        "mixed_coefficients": train_coefficients,
        "state_plus_mixed": np.column_stack(
            (train_closed, train_drive, train_coefficients)
        ),
    }
    fitted_models: dict[str, _LinearDecoder] = {}
    alphas: dict[str, float] = {}
    cross_validation: dict[str, dict[str, float]] = {}
    for name, features in feature_sets.items():
        alpha, scores = _choose_alpha(
            features,
            train_source,
            source_scales,
            train_groups,
        )
        alphas[name] = alpha
        cross_validation[name] = scores
        fitted_models[name] = _fit_decoder(
            features,
            train_source,
            source_scales,
            alpha=alpha,
        )

    decoder_scores: dict[str, float] = {}
    per_path: dict[str, object] = {}
    for index, label in enumerate(labels):
        path_features = {
            "retained_state": np.column_stack(
                (closed_paths[index], drive_paths[index])
            ),
            "mixed_coefficients": coefficient_paths[index],
            "state_plus_mixed": np.column_stack(
                (
                    closed_paths[index],
                    drive_paths[index],
                    coefficient_paths[index],
                )
            ),
        }
        model_scores = {
            name: _source_nrms(
                fitted_models[name].predict(features),
                target_source_paths[index],
                source_scales,
            )
            for name, features in path_features.items()
        }
        window_scores: dict[str, dict[str, float]] = {}
        for window_name, window_mask in {
            "before_second_pulse_0_to_8": times <= 8.0,
            "after_second_pulse_8_to_20": times > 8.0,
            "late_10_to_20": times >= 10.0,
        }.items():
            window_scores[window_name] = {
                name: _source_nrms(
                    fitted_models[name].predict(features[window_mask]),
                    target_source_paths[index][window_mask],
                    source_scales,
                )
                for name, features in path_features.items()
            }
        if roles[index] != "train":
            for name, value in model_scores.items():
                decoder_scores[f"{label}:{name}"] = value
        target_scaled = target_source_paths[index] / source_scales
        archive_scaled = archive_source_paths[index] / source_scales
        mixed_scaled = mixed_source_paths[index] / source_scales
        unresolved_scaled = unresolved_paths[index] / source_scales
        target_rms = float(
            np.sqrt(np.mean(np.sum(target_scaled * target_scaled, axis=1)))
        )
        per_path[label] = {
            "role": roles[index],
            "source_factorization": {
                "target_scaled_vector_rms": target_rms,
                "archive_frame_part_relative_rms": float(
                    np.sqrt(np.mean(np.sum(archive_scaled**2, axis=1)))
                    / max(target_rms, np.finfo(float).tiny)
                ),
                "mixed_part_relative_rms": float(
                    np.sqrt(np.mean(np.sum(mixed_scaled**2, axis=1)))
                    / max(target_rms, np.finfo(float).tiny)
                ),
                "unresolved_relative_rms": float(
                    np.sqrt(np.mean(np.sum(unresolved_scaled**2, axis=1)))
                    / max(target_rms, np.finfo(float).tiny)
                ),
                "maximum_unresolved_absolute_coordinate": float(
                    np.max(np.abs(unresolved_paths[index]))
                ),
            },
            "fixed_decoder_source_nrms": model_scores,
            "fixed_decoder_window_source_nrms": window_scores,
            "mixed_complement_rank": sorted(set(ranks[index].tolist())),
            "minimum_mixed_singular_ratio": float(
                np.min(singular_ratios[index])
            ),
            "maximum_archive_gram_error": float(np.max(gram_errors[index])),
            "maximum_relative_top_population": float(
                np.max(top_populations[index])
            ),
            "packet_count_range": [
                int(np.min(packet_counts[index])),
                int(np.max(packet_counts[index])),
            ],
        }

    preparation_holdout_scores: dict[str, dict[str, float]] = {}
    for held_index in training_indices:
        fitting_indices = [
            index for index in training_indices if index != held_index
        ]
        preparation_holdout_scores[labels[held_index]] = {}
        held_features = {
            "retained_state": np.column_stack(
                (closed_paths[held_index], drive_paths[held_index])
            ),
            "mixed_coefficients": coefficient_paths[held_index],
            "state_plus_mixed": np.column_stack(
                (
                    closed_paths[held_index],
                    drive_paths[held_index],
                    coefficient_paths[held_index],
                )
            ),
        }
        fitting_features = {
            "retained_state": _flatten(
                [
                    np.column_stack((closed_paths[index], drive_paths[index]))
                    for index in range(len(contracts))
                ],
                fitting_indices,
            ),
            "mixed_coefficients": _flatten(
                coefficient_paths,
                fitting_indices,
            ),
            "state_plus_mixed": _flatten(
                [
                    np.column_stack(
                        (
                            closed_paths[index],
                            drive_paths[index],
                            coefficient_paths[index],
                        )
                    )
                    for index in range(len(contracts))
                ],
                fitting_indices,
            ),
        }
        fitting_source = _flatten(target_source_paths, fitting_indices)
        for name in fitted_models:
            fold_model = _fit_decoder(
                fitting_features[name],
                fitting_source,
                source_scales,
                alpha=alphas[name],
            )
            preparation_holdout_scores[labels[held_index]][name] = (
                _source_nrms(
                    fold_model.predict(held_features[name]),
                    target_source_paths[held_index],
                    source_scales,
                )
            )

    response_train = _flatten(
        [values.reshape(times.size, -1) for values in response_paths],
        training_indices,
    )
    response_blocks = [
        _response_block(
            archive_source_paths[index],
            response_paths[index],
            source_scales,
        )
        for index in range(len(contracts))
    ]
    double_training_blocks = _flatten(response_blocks, training_indices)
    multi_drive_indices = [*training_indices, labels.index("single_central")]
    multi_drive_training_blocks = _flatten(
        response_blocks,
        multi_drive_indices,
    )
    drive_holdout_index = labels.index("single_central")
    capacity_holdout_index = labels.index("double_central_k10")
    response_subspace_transfer = {
        "double_drive_to_single_drive": {
            "all_0_to_20": _response_subspace_score(
                double_training_blocks,
                response_blocks[drive_holdout_index],
                coefficient_paths[drive_holdout_index],
                target_source_paths[drive_holdout_index],
                source_scales,
            ),
            "before_second_pulse_0_to_8": _response_subspace_score(
                double_training_blocks,
                response_blocks[drive_holdout_index],
                coefficient_paths[drive_holdout_index],
                target_source_paths[drive_holdout_index],
                source_scales,
                mask=times <= 8.0,
            ),
            "after_second_pulse_8_to_20": _response_subspace_score(
                double_training_blocks,
                response_blocks[drive_holdout_index],
                coefficient_paths[drive_holdout_index],
                target_source_paths[drive_holdout_index],
                source_scales,
                mask=times > 8.0,
            ),
        },
        "multi_drive_to_k10_capacity": _response_subspace_score(
            multi_drive_training_blocks,
            response_blocks[capacity_holdout_index],
            coefficient_paths[capacity_holdout_index],
            target_source_paths[capacity_holdout_index],
            source_scales,
        ),
    }
    summary: dict[str, object] = {
        "schema": "paper_v.mixed_tangent_closure_identifiability.v1",
        "run_id": output_directory.name,
        "status": "complete",
        "classification": "offline_stored_state_identifiability_diagnostic",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_question": (
            "Do six physical relative-phonon--Pauli tangent coefficients admit "
            "a fixed transferable decoder into the missing C velocity, or does "
            "the state-dependent mixed response require additional hidden state?"
        ),
        "scope": {
            "online_propagation": False,
            "exact_reference_supplied_to_model": False,
            "same_state_schrodinger_teacher": True,
            "maximum_time": maximum_time,
            "sample_step": sample_step,
        },
        "mixed_direction_labels": list(MIXED_LABELS),
        "regularization": {
            "geometric_gram_relative_threshold": (
                geometric_gram_relative_threshold
            ),
            "selected_ridge_alpha": alphas,
            "leave_one_preparation_out_cv": cross_validation,
        },
        "variation_ranks": {
            "mixed_coefficients": _variance_rank(
                train_coefficients, (0.9, 0.99, 0.999)
            ),
            "mixed_response": _variance_rank(
                response_train, (0.9, 0.99, 0.999)
            ),
        },
        "leave_one_preparation_out_source_nrms": (
            preparation_holdout_scores
        ),
        "response_subspace_transfer": response_subspace_transfer,
        "paths": per_path,
        "interpretation_rule": (
            "Near-zero unresolved source verifies the local factorization. A "
            "mixed-coefficient decoder that transfers across preparation, drive, "
            "and packet-capacity holdouts would support those coefficients as a "
            "compact hidden state. Failure while the exact state-dependent "
            "factorization remains accurate means the response map itself carries "
            "additional omitted correlations."
        ),
        "elapsed_seconds": time.time() - started,
    }

    arrays_path = output_directory / "mixed_tangent_closure_identifiability.npz"
    np.savez_compressed(
        arrays_path,
        labels=np.asarray(labels),
        roles=np.asarray(roles),
        times=times,
        coordinate_scales=coordinate_scales,
        closed_coordinates=np.asarray(closed_paths),
        mixed_coefficients=np.asarray(coefficient_paths),
        mixed_response=np.asarray(response_paths),
        target_source=np.asarray(target_source_paths),
        archive_frame_source=np.asarray(archive_source_paths),
        mixed_source=np.asarray(mixed_source_paths),
        unresolved_source=np.asarray(unresolved_paths),
        drive_difference=np.asarray(drive_paths),
        mixed_complement_rank=np.asarray(ranks),
        mixed_singular_ratio=np.asarray(singular_ratios),
        packet_count=np.asarray(packet_counts),
    )
    summary_path = output_directory / "summary.json"
    _write_json(summary_path, summary)
    plot_path = output_directory / "mixed_tangent_closure_identifiability.png"
    _make_plot(
        plot_path,
        times,
        labels,
        target_source_paths,
        archive_source_paths,
        mixed_source_paths,
        source_scales,
        decoder_scores,
        coefficient_paths,
    )
    source_paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[2]
        / "paper_5/src/paper5/stability/mixed_tangent_closure_identifiability.py",
    )
    manifest = {
        "schema": "paper_v.mixed_tangent_closure_identifiability.runtime.v1",
        "run_id": output_directory.name,
        "status": "complete",
        "python": sys.version,
        "platform": platform.platform(),
        "inputs": {str(path): _sha256(path) for path in input_paths},
        "sources": {str(path): _sha256(path) for path in source_paths},
        "outputs": {
            path.name: _sha256(path)
            for path in (arrays_path, summary_path, plot_path)
        },
    }
    _write_json(output_directory / "runtime_manifest.json", manifest)
    print(json.dumps(summary["paths"], indent=2, sort_keys=True))
    return summary


def _parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    output_runs = repo_root / "output/local_runs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--double-pulse-batch",
        type=Path,
        default=output_runs
        / "paper_v_multi_coherent_double_pulse_blind_model_cutoff16_20260804_v1",
    )
    parser.add_argument(
        "--single-pulse-run",
        type=Path,
        default=output_runs
        / "paper_v_multi_coherent_horizontal_open_dev_cutoff16_t20_tikh3e4_dense_20260804_v1",
    )
    parser.add_argument(
        "--capacity-holdout-run",
        type=Path,
        default=output_runs
        / "paper_v_multi_coherent_capacity_k10_t40_20260804_v1/fine_central",
    )
    parser.add_argument(
        "--coordinate-scales",
        type=Path,
        default=output_runs
        / "paper_v_trajectory_closure_identifiability_cutoff16_20260804_v1"
        / "trajectory_closure_identifiability.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output_runs / RUN_ID,
    )
    parser.add_argument("--maximum-time", type=float, default=20.0)
    parser.add_argument("--sample-step", type=float, default=0.25)
    parser.add_argument(
        "--geometric-gram-relative-threshold",
        type=float,
        default=1e-10,
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    contracts = (
        _PathContract(
            "double_central",
            arguments.double_pulse_batch / "fine_central",
            "train",
        ),
        _PathContract(
            "double_plus",
            arguments.double_pulse_batch / "fine_plus",
            "train",
        ),
        _PathContract(
            "double_minus",
            arguments.double_pulse_batch / "fine_minus",
            "train",
        ),
        _PathContract("single_central", arguments.single_pulse_run, "drive_holdout"),
        _PathContract(
            "double_central_k10",
            arguments.capacity_holdout_run,
            "capacity_holdout",
        ),
    )
    run(
        contracts,
        arguments.coordinate_scales,
        arguments.output_dir,
        maximum_time=arguments.maximum_time,
        sample_step=arguments.sample_step,
        geometric_gram_relative_threshold=(
            arguments.geometric_gram_relative_threshold
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
