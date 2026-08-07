"""Derivative-first gate for a stable latent correction to the ``C`` EOM.

This command consumes only a stored exact-source audit.  It does not call an
integrator, reopen a scorer, or propagate the proposed latent closure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from paper5.stability.hubbard_dimer import GaussianSineDrive
from paper5.stability.latent_source_closure import (
    LatentSourceBasis,
    StableSecondOrderLatentSourceEvolutionModel,
    estimate_time_derivative,
    fit_latent_source_basis,
    fit_stable_second_order_latent_source_evolution,
    latent_homogeneous_eigenvalues,
    normalized_vector_rms_error,
    predict_stable_second_order_latent_source_evolution,
    project_missing_source,
    reconstruct_missing_source,
)


MEMBER_NAMES = ("central", "plus", "minus")
RIDGE_CANDIDATES = (1e-4, 1e-2, 1.0, 100.0, 1e4)
SOURCE_RANK = 5
STABILITY_MARGIN = 0.01
BLOCK_LENGTH = 5.0
FOLD_COUNT = 5
DERIVATIVE_ERROR_CEILING = 0.5
SOURCE_RECONSTRUCTION_CEILING = 0.05
DERIVATIVE_RESOLUTION_CEILING = 0.01


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _derivative_chain(
    times: np.ndarray,
    latent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rates = np.stack(
        [estimate_time_derivative(times, member) for member in latent]
    )
    accelerations = np.stack(
        [estimate_time_derivative(times, member) for member in rates]
    )
    return rates, accelerations


def _normalized_residual(
    reconstructed: np.ndarray,
    reference: np.ndarray,
    scales: np.ndarray,
    mask: np.ndarray,
) -> float:
    reconstructed_scaled = reconstructed[:, mask] / scales
    reference_scaled = reference[:, mask] / scales
    centered = reference_scaled - np.mean(
        reference_scaled,
        axis=(0, 1),
        keepdims=True,
    )
    scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=2))))
    residual = reconstructed_scaled - reference_scaled
    return float(np.sqrt(np.mean(np.sum(residual**2, axis=2))) / scale)


def _blocked_cv_score(
    times: np.ndarray,
    coordinates: np.ndarray,
    source: np.ndarray,
    rates: np.ndarray,
    accelerations: np.ndarray,
    drive: np.ndarray,
    scales: np.ndarray,
    *,
    ridge_penalty: float,
) -> dict[str, float]:
    eligible = (times >= 0.2) & (times <= times[-1] - 0.2)
    blocks = np.floor(times / BLOCK_LENGTH).astype(int)
    errors: list[np.ndarray] = []
    references: list[np.ndarray] = []
    maximum_real_parts: list[float] = []
    for fold in range(FOLD_COUNT):
        validation = eligible & (blocks % FOLD_COUNT == fold)
        training = eligible & ~validation
        model = fit_stable_second_order_latent_source_evolution(
            coordinates,
            source,
            rates,
            accelerations,
            drive,
            scales,
            ridge_penalty=ridge_penalty,
            stability_margin=STABILITY_MARGIN,
            training_mask=training,
        )
        predicted = predict_stable_second_order_latent_source_evolution(
            model,
            coordinates[validation],
            source[validation],
            rates[validation],
            drive[validation],
        )[:, SOURCE_RANK:]
        errors.append(predicted - accelerations[validation])
        references.append(accelerations[validation])
        maximum_real_parts.append(
            float(np.max(np.real(latent_homogeneous_eigenvalues(model))))
        )
    error = np.concatenate(errors, axis=0)
    reference = np.concatenate(references, axis=0)
    centered = reference - np.mean(reference, axis=0)
    scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    return {
        "ridge_penalty": ridge_penalty,
        "blocked_cv_acceleration_normalized_rms": float(
            np.sqrt(np.mean(np.sum(error**2, axis=1))) / scale
        ),
        "maximum_fold_homogeneous_real_part": float(
            np.max(maximum_real_parts)
        ),
    }


def _score_interval(
    times: np.ndarray,
    model: StableSecondOrderLatentSourceEvolutionModel,
    coordinates: np.ndarray,
    source: np.ndarray,
    rates: np.ndarray,
    accelerations: np.ndarray,
    drive: np.ndarray,
    interval: tuple[float, float],
) -> dict[str, float]:
    mask = (times >= interval[0]) & (times <= interval[1])
    predicted = predict_stable_second_order_latent_source_evolution(
        model,
        coordinates[mask],
        source[mask],
        rates[mask],
        drive[mask],
    )
    reference = np.concatenate((rates[mask], accelerations[mask]), axis=1)
    return {
        "acceleration_normalized_rms": normalized_vector_rms_error(
            predicted[:, SOURCE_RANK:],
            accelerations[mask],
        ),
        "full_latent_velocity_normalized_rms": normalized_vector_rms_error(
            predicted,
            reference,
        ),
    }


def _unshifted_eigenvalues(
    model: StableSecondOrderLatentSourceEvolutionModel,
) -> np.ndarray:
    rank = model.source_coefficients.shape[0]
    shift = model.stability_shift
    original_rate = model.rate_coefficients + 2.0 * shift * np.eye(rank)
    original_source = (
        model.source_coefficients
        - shift * original_rate
        + shift**2 * np.eye(rank)
    )
    return np.linalg.eigvals(
        np.block(
            [
                [np.zeros((rank, rank)), np.eye(rank)],
                [original_source, original_rate],
            ]
        )
    )


def _write_plot(
    path: Path,
    times: np.ndarray,
    source: np.ndarray,
    reconstructed_source: np.ndarray,
    acceleration: np.ndarray,
    predictions: np.ndarray,
    candidate_scores: list[dict[str, float]],
    model: StableSecondOrderLatentSourceEvolutionModel,
    basis: LatentSourceBasis,
) -> None:
    figure, axes_grid = plt.subplots(2, 2, figsize=(7.3, 4.7))
    axes = axes_grid.ravel()

    training_variance = basis.singular_values**2
    cumulative = np.cumsum(training_variance) / np.sum(training_variance)
    axis = axes[0]
    axis.plot(np.arange(1, 11), cumulative[:10], marker="o", markersize=3)
    axis.axvline(SOURCE_RANK, color="#777777", linestyle=":", linewidth=0.9)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("retained source modes")
    axis.set_ylabel("central variance explained")
    axis.set_title("(a) source compression", loc="left")

    axis = axes[1]
    ridge = np.asarray(
        [record["ridge_penalty"] for record in candidate_scores]
    )
    cv_error = np.asarray(
        [
            record["blocked_cv_acceleration_normalized_rms"]
            for record in candidate_scores
        ]
    )
    axis.plot(ridge, cv_error, marker="o", markersize=3)
    axis.axhline(
        DERIVATIVE_ERROR_CEILING,
        color="#a23b2a",
        linestyle="--",
        linewidth=0.8,
        label="derivative gate",
    )
    axis.set_xscale("log")
    axis.set_xlabel("ridge penalty")
    axis.set_ylabel("blocked-CV acceleration RMS")
    axis.set_title("(b) model selection", loc="left")
    axis.legend(frameon=False, fontsize=6.4)

    axis = axes[2]
    for member, name in enumerate(MEMBER_NAMES):
        centered = acceleration[member] - np.mean(acceleration[member], axis=0)
        scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
        point_error = np.linalg.norm(
            predictions[member] - acceleration[member],
            axis=1,
        ) / scale
        axis.plot(times, point_error, linewidth=0.7, label=name)
    axis.set_yscale("log")
    axis.set_xlabel(r"$t\,t_{\rm hop}$")
    axis.set_ylabel("normalized acceleration error")
    axis.set_title("(c) exact-state derivative score", loc="left")
    axis.legend(frameon=False, fontsize=6.4)

    axis = axes[3]
    before = _unshifted_eigenvalues(model)
    after = latent_homogeneous_eigenvalues(model)
    axis.scatter(before.real, before.imag, s=17, label="unconstrained fit")
    axis.scatter(after.real, after.imag, s=17, marker="x", label="stable shift")
    axis.axvline(0.0, color="#333333", linewidth=0.7)
    axis.axvline(
        -STABILITY_MARGIN,
        color="#777777",
        linestyle=":",
        linewidth=0.8,
    )
    axis.set_xlabel("real part")
    axis.set_ylabel("imaginary part")
    axis.set_title("(d) latent poles", loc="left")
    axis.legend(frameon=False, fontsize=6.4)

    for axis in axes:
        axis.grid(alpha=0.17, linewidth=0.5)
    source_error = _normalized_residual(
        reconstructed_source,
        source,
        basis.coordinate_scales,
        np.ones(times.size, dtype=bool),
    )
    figure.text(
        0.5,
        0.012,
        f"Stored exact states only; rank-{SOURCE_RANK} source residual "
        f"{source_error:.4f}; no candidate propagation.",
        ha="center",
        fontsize=6.4,
    )
    figure.subplots_adjust(
        left=0.1,
        right=0.985,
        bottom=0.12,
        top=0.94,
        hspace=0.43,
        wspace=0.36,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def run_audit(
    source_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    arrays_path = source_directory / "trajectory_closure_identifiability.npz"
    source_metrics_path = source_directory / "metrics.json"
    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    with np.load(arrays_path) as arrays:
        times = np.asarray(arrays["times"], dtype=float)
        scales = np.asarray(arrays["coordinate_scales"], dtype=float)
        coordinates = np.asarray(arrays["dop853_closed"], dtype=float)
        primary_source = np.asarray(
            arrays["dop853_target_source"],
            dtype=float,
        )
        independent_source = np.asarray(
            arrays["midpoint_target_source"],
            dtype=float,
        )
    if not np.isclose(times[1] - times[0], 0.05, atol=1e-12, rtol=0.0):
        raise ValueError("latent derivative audit requires the dense 0.05 grid")

    basis_training = times <= 8.0
    basis = fit_latent_source_basis(
        primary_source,
        scales[17:31],
        rank=SOURCE_RANK,
        training_member=0,
        training_mask=basis_training,
    )
    primary_latent = project_missing_source(primary_source, basis)
    independent_latent = project_missing_source(independent_source, basis)
    primary_rate, primary_acceleration = _derivative_chain(
        times,
        primary_latent,
    )
    independent_rate, independent_acceleration = _derivative_chain(
        times,
        independent_latent,
    )

    drive_data = source_metrics["drive_protocol"]
    drive_protocol = GaussianSineDrive(
        amplitude=float(drive_data["amplitude"]),
        pulse_width=float(drive_data["pulse_width"]),
        delays=tuple(float(value) for value in drive_data["delays"]),
    )
    drive = np.asarray(
        [drive_protocol.difference(float(time)) for time in times],
        dtype=float,
    )

    candidate_scores = [
        _blocked_cv_score(
            times,
            coordinates[0],
            primary_latent[0],
            primary_rate[0],
            primary_acceleration[0],
            drive,
            scales,
            ridge_penalty=penalty,
        )
        for penalty in RIDGE_CANDIDATES
    ]
    selected_candidate = min(
        candidate_scores,
        key=lambda record: record["blocked_cv_acceleration_normalized_rms"],
    )
    fit_mask = (times >= 0.2) & (times <= times[-1] - 0.2)
    model = fit_stable_second_order_latent_source_evolution(
        coordinates[0],
        primary_latent[0],
        primary_rate[0],
        primary_acceleration[0],
        drive,
        scales,
        ridge_penalty=selected_candidate["ridge_penalty"],
        stability_margin=STABILITY_MARGIN,
        training_mask=fit_mask,
    )
    predicted_acceleration = np.empty_like(primary_acceleration)
    interval_scores: dict[str, dict[str, dict[str, float]]] = {}
    for member, name in enumerate(MEMBER_NAMES):
        predicted = predict_stable_second_order_latent_source_evolution(
            model,
            coordinates[member],
            primary_latent[member],
            primary_rate[member],
            drive,
        )
        predicted_acceleration[member] = predicted[:, SOURCE_RANK:]
        interval_scores[name] = {
            "full": _score_interval(
                times,
                model,
                coordinates[member],
                primary_latent[member],
                primary_rate[member],
                primary_acceleration[member],
                drive,
                (0.2, times[-1] - 0.2),
            ),
            "second_pulse": _score_interval(
                times,
                model,
                coordinates[member],
                primary_latent[member],
                primary_rate[member],
                primary_acceleration[member],
                drive,
                (8.2, 20.0),
            ),
            "post_pulse": _score_interval(
                times,
                model,
                coordinates[member],
                primary_latent[member],
                primary_rate[member],
                primary_acceleration[member],
                drive,
                (20.0, times[-1] - 0.2),
            ),
        }

    coarse_times = times[::2]
    coarse_latent = primary_latent[:, ::2]
    coarse_rate, coarse_acceleration = _derivative_chain(
        coarse_times,
        coarse_latent,
    )
    common_mask = (coarse_times >= 0.4) & (
        coarse_times <= coarse_times[-1] - 0.4
    )
    dense_common_rate = primary_rate[:, ::2][:, common_mask]
    dense_common_acceleration = primary_acceleration[:, ::2][:, common_mask]
    derivative_resolution = {
        "two_exact_references_rate_normalized_rms": normalized_vector_rms_error(
            independent_rate.reshape(-1, SOURCE_RANK),
            primary_rate.reshape(-1, SOURCE_RANK),
        ),
        "two_exact_references_acceleration_normalized_rms": (
            normalized_vector_rms_error(
                independent_acceleration.reshape(-1, SOURCE_RANK),
                primary_acceleration.reshape(-1, SOURCE_RANK),
            )
        ),
        "step_0p05_vs_0p1_rate_normalized_rms": normalized_vector_rms_error(
            coarse_rate[:, common_mask].reshape(-1, SOURCE_RANK),
            dense_common_rate.reshape(-1, SOURCE_RANK),
        ),
        "step_0p05_vs_0p1_acceleration_normalized_rms": (
            normalized_vector_rms_error(
                coarse_acceleration[:, common_mask].reshape(-1, SOURCE_RANK),
                dense_common_acceleration.reshape(-1, SOURCE_RANK),
            )
        ),
    }

    reconstructed_source = reconstruct_missing_source(primary_latent, basis)
    source_reconstruction = {
        "full_normalized_rms": _normalized_residual(
            reconstructed_source,
            primary_source,
            basis.coordinate_scales,
            np.ones(times.size, dtype=bool),
        ),
        "second_pulse_normalized_rms": _normalized_residual(
            reconstructed_source,
            primary_source,
            basis.coordinate_scales,
            (times >= 8.0) & (times <= 20.0),
        ),
    }
    maximum_heldout = max(
        interval_scores[name]["full"]["acceleration_normalized_rms"]
        for name in ("plus", "minus")
    )
    maximum_real_after = float(
        np.max(np.real(latent_homogeneous_eigenvalues(model)))
    )
    gates = {
        "source_subspace": (
            source_reconstruction["full_normalized_rms"]
            <= SOURCE_RECONSTRUCTION_CEILING
        ),
        "derivative_resolution": (
            derivative_resolution["step_0p05_vs_0p1_acceleration_normalized_rms"]
            <= DERIVATIVE_RESOLUTION_CEILING
        ),
        "blocked_central_development": (
            selected_candidate["blocked_cv_acceleration_normalized_rms"]
            <= DERIVATIVE_ERROR_CEILING
        ),
        "heldout_preparations": maximum_heldout <= DERIVATIVE_ERROR_CEILING,
        "homogeneous_stability": maximum_real_after <= -STABILITY_MARGIN + 1e-10,
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    arrays_output = output_directory / "latent_source_closure.npz"
    np.savez_compressed(
        arrays_output,
        times=times,
        coordinate_scales=scales,
        source_basis=basis.basis,
        source_center=basis.center,
        source_coordinate_scales=basis.coordinate_scales,
        source_singular_values=basis.singular_values,
        primary_latent=primary_latent,
        independent_latent=independent_latent,
        primary_rate=primary_rate,
        independent_rate=independent_rate,
        primary_acceleration=primary_acceleration,
        independent_acceleration=independent_acceleration,
        predicted_acceleration=predicted_acceleration,
        initial_latent_source=primary_latent[:, 0],
        initial_latent_rate=primary_rate[:, 0],
        acceleration_intercept=model.acceleration_intercept,
        state_coefficients=model.state_coefficients,
        source_coefficients=model.source_coefficients,
        rate_coefficients=model.rate_coefficients,
        drive_coefficients=model.drive_coefficients,
        homogeneous_eigenvalues=latent_homogeneous_eigenvalues(model),
    )
    plot_path = output_directory / "latent_source_closure.pdf"
    _write_plot(
        plot_path,
        times,
        primary_source,
        reconstructed_source,
        primary_acceleration,
        predicted_acceleration,
        candidate_scores,
        model,
        basis,
    )
    metrics = {
        "schema": "paper5.latent_source_closure.derivative_gate.v1",
        "classification": "exploratory_stored_exact_states_not_promoted",
        "source_score_status": source_metrics["score_status"],
        "model": {
            "total_state_dimension": 41,
            "retained_moment_dimension": 31,
            "latent_source_dimension": 5,
            "latent_rate_dimension": 5,
            "equation": "dot(z)=p; dot(p)=c+B_x x+A_z z+A_p p+B_v V(t)",
            "basis_training_interval": [0.0, 8.0],
            "fit_member": "central",
            "selected_ridge_penalty": selected_candidate["ridge_penalty"],
            "stability_margin": STABILITY_MARGIN,
            "stability_shift": model.stability_shift,
            "maximum_real_part_before_shift": (
                model.maximum_real_part_before_shift
            ),
            "maximum_real_part_after_shift": maximum_real_after,
        },
        "candidate_scores": candidate_scores,
        "source_reconstruction": source_reconstruction,
        "derivative_resolution": derivative_resolution,
        "interval_scores": interval_scores,
        "gates": gates,
        "derivative_gate_passed": bool(all(gates.values())),
        "propagation_status": "not_run_derivative_gate_only",
        "limitations": [
            "The basis and coefficients use the opened central exact trajectory.",
            "Plus/minus holdouts are nearby preparations under the same drive.",
            "Derivative agreement does not establish stable coupled propagation.",
            "The consumed source score retained an indeterminate reference label.",
        ],
        "input_hashes": {
            str(arrays_path): _sha256(arrays_path),
            str(source_metrics_path): _sha256(source_metrics_path),
        },
    }
    metrics_path = output_directory / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runtime = {
        "schema": "paper5.latent_source_closure.runtime.v1",
        "input_hashes": metrics["input_hashes"],
        "artifact_hashes": {
            arrays_output.name: _sha256(arrays_output),
            metrics_path.name: _sha256(metrics_path),
            plot_path.name: _sha256(plot_path),
            plot_path.with_suffix(".png").name: _sha256(
                plot_path.with_suffix(".png")
            ),
        },
    }
    (output_directory / "runtime_manifest.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(args.source_directory, args.output_directory)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
