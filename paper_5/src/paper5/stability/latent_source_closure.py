"""Derivative-first reduced models for a missing correlation source.

The routines in this module are pure stored-array transformations.  They do
not propagate the archive EOM or query an exact reference during an online
right-hand-side evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LatentSourceBasis:
    """Scaled orthonormal basis for a missing 14-coordinate ``C`` source."""

    center: np.ndarray
    basis: np.ndarray
    coordinate_scales: np.ndarray
    singular_values: np.ndarray

    @property
    def rank(self) -> int:
        return int(self.basis.shape[0])


@dataclass(frozen=True)
class LatentSourceEvolutionModel:
    """Standardized ridge model for one latent-source velocity."""

    feature_family: str
    coefficients: np.ndarray
    feature_center: np.ndarray
    feature_scale: np.ndarray
    coordinate_scales: np.ndarray
    latent_rank: int
    ridge_penalty: float


@dataclass(frozen=True)
class SecondOrderLatentSourceEvolutionModel:
    """Latent source with exact ``dot(z)=p`` and a fitted acceleration."""

    acceleration_model: LatentSourceEvolutionModel
    source_rank: int


@dataclass(frozen=True)
class StableSecondOrderLatentSourceEvolutionModel:
    """Affine second-order source model with a stable homogeneous block."""

    acceleration_intercept: np.ndarray
    state_coefficients: np.ndarray
    source_coefficients: np.ndarray
    rate_coefficients: np.ndarray
    drive_coefficients: np.ndarray
    coordinate_scales: np.ndarray
    ridge_penalty: float
    stability_margin: float
    stability_shift: float
    maximum_real_part_before_shift: float


@dataclass(frozen=True)
class LatentEvolutionCandidateScore:
    """Development scores for one declared velocity-model candidate."""

    feature_family: str
    ridge_penalty: float
    feature_count: int
    training_normalized_rms: float
    validation_normalized_rms: float


@dataclass(frozen=True)
class LatentEvolutionSelection:
    """Selected model and the complete development candidate ledger."""

    model: LatentSourceEvolutionModel
    candidates: tuple[LatentEvolutionCandidateScore, ...]
    training_normalized_rms: float
    validation_normalized_rms: float


def estimate_time_derivative(
    times: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Differentiate uniformly sampled values with five-point formulas."""

    time_array = np.asarray(times, dtype=float)
    samples = np.asarray(values, dtype=float)
    if time_array.ndim != 1 or time_array.size < 5:
        raise ValueError("times must contain at least five samples")
    if samples.shape[0] != time_array.size:
        raise ValueError("values must have times on the leading axis")
    if not np.all(np.isfinite(samples)):
        raise ValueError("values must be finite")
    steps = np.diff(time_array)
    if not np.all(steps > 0.0):
        raise ValueError("times must be strictly increasing")
    step = float(steps[0])
    if not np.allclose(steps, step, atol=1e-12, rtol=1e-10):
        raise ValueError("five-point differentiation requires a uniform grid")

    derivative = np.empty_like(samples)
    derivative[0] = (
        -25.0 * samples[0]
        + 48.0 * samples[1]
        - 36.0 * samples[2]
        + 16.0 * samples[3]
        - 3.0 * samples[4]
    ) / (12.0 * step)
    derivative[1] = (
        -3.0 * samples[0]
        - 10.0 * samples[1]
        + 18.0 * samples[2]
        - 6.0 * samples[3]
        + samples[4]
    ) / (12.0 * step)
    derivative[2:-2] = (
        samples[:-4]
        - 8.0 * samples[1:-3]
        + 8.0 * samples[3:-1]
        - samples[4:]
    ) / (12.0 * step)
    derivative[-2] = (
        -samples[-5]
        + 6.0 * samples[-4]
        - 18.0 * samples[-3]
        + 10.0 * samples[-2]
        + 3.0 * samples[-1]
    ) / (12.0 * step)
    derivative[-1] = (
        3.0 * samples[-5]
        - 16.0 * samples[-4]
        + 36.0 * samples[-3]
        - 48.0 * samples[-2]
        + 25.0 * samples[-1]
    ) / (12.0 * step)
    return derivative


def fit_latent_source_basis(
    source: np.ndarray,
    coordinate_scales: np.ndarray,
    *,
    rank: int,
    training_member: int = 0,
    training_mask: np.ndarray | None = None,
) -> LatentSourceBasis:
    """Fit a scaled source basis on one declared development trajectory."""

    values = np.asarray(source, dtype=float)
    scales = np.asarray(coordinate_scales, dtype=float)
    if values.ndim != 3 or values.shape[-1] != 14:
        raise ValueError("source must have shape (members, times, 14)")
    if scales.shape != (14,) or np.any(scales <= 0.0):
        raise ValueError("coordinate_scales must be positive with shape (14,)")
    if not 0 <= training_member < values.shape[0]:
        raise ValueError("training_member is out of range")
    if not 1 <= rank <= 14:
        raise ValueError("rank must lie between one and fourteen")
    if training_mask is None:
        selected = np.ones(values.shape[1], dtype=bool)
    else:
        selected = np.asarray(training_mask, dtype=bool)
        if selected.shape != (values.shape[1],):
            raise ValueError("training_mask must match the time axis")
    if np.count_nonzero(selected) < rank:
        raise ValueError("training selection must contain at least rank samples")

    training = values[training_member, selected]
    center = np.mean(training, axis=0)
    scaled = (training - center) / scales
    _, singular_values, right_vectors = np.linalg.svd(
        scaled,
        full_matrices=False,
    )
    return LatentSourceBasis(
        center=center,
        basis=right_vectors[:rank],
        coordinate_scales=scales.copy(),
        singular_values=singular_values,
    )


def project_missing_source(
    source: np.ndarray,
    model: LatentSourceBasis,
) -> np.ndarray:
    """Project missing-source coordinates into the fitted latent basis."""

    values = np.asarray(source, dtype=float)
    if values.shape[-1] != 14:
        raise ValueError("source must have trailing dimension 14")
    scaled = (values - model.center) / model.coordinate_scales
    return scaled @ model.basis.T


def reconstruct_missing_source(
    latent: np.ndarray,
    model: LatentSourceBasis,
) -> np.ndarray:
    """Reconstruct missing-source coordinates from latent amplitudes."""

    amplitudes = np.asarray(latent, dtype=float)
    if amplitudes.shape[-1] != model.rank:
        raise ValueError("latent amplitudes do not match the basis rank")
    scaled = amplitudes @ model.basis
    return model.center + scaled * model.coordinate_scales


def _latent_evolution_features(
    coordinates: np.ndarray,
    latent: np.ndarray,
    drive: np.ndarray,
    coordinate_scales: np.ndarray,
    *,
    feature_family: str,
) -> np.ndarray:
    states = np.asarray(coordinates, dtype=float)
    amplitudes = np.asarray(latent, dtype=float)
    drive_values = np.asarray(drive, dtype=float)
    scales = np.asarray(coordinate_scales, dtype=float)
    if states.ndim != 2 or states.shape[1] != 31:
        raise ValueError("coordinates must have shape (samples, 31)")
    if amplitudes.ndim != 2 or amplitudes.shape[0] != states.shape[0]:
        raise ValueError("latent must have shape (samples, latent_rank)")
    if drive_values.shape != (states.shape[0],):
        raise ValueError("drive must have shape (samples,)")
    if scales.shape != (31,) or np.any(scales <= 0.0):
        raise ValueError("coordinate_scales must be positive with shape (31,)")
    normalized_states = states / scales
    drive_column = drive_values[:, None]
    if feature_family == "latent_affine":
        return np.concatenate((amplitudes, drive_column), axis=1)
    if feature_family == "state_latent_affine":
        return np.concatenate(
            (normalized_states, amplitudes, drive_column),
            axis=1,
        )
    if feature_family == "state_latent_drive_bilinear":
        return np.concatenate(
            (
                normalized_states,
                amplitudes,
                drive_column,
                normalized_states * drive_column,
                amplitudes * drive_column,
            ),
            axis=1,
        )
    raise ValueError(f"unsupported feature_family: {feature_family}")


def fit_latent_source_evolution(
    coordinates: np.ndarray,
    latent: np.ndarray,
    latent_derivative: np.ndarray,
    drive: np.ndarray,
    coordinate_scales: np.ndarray,
    *,
    feature_family: str,
    ridge_penalty: float,
    training_mask: np.ndarray | None = None,
) -> LatentSourceEvolutionModel:
    """Fit one declared latent-source velocity model on selected samples."""

    amplitudes = np.asarray(latent, dtype=float)
    derivatives = np.asarray(latent_derivative, dtype=float)
    if derivatives.ndim != 2 or derivatives.shape[0] != amplitudes.shape[0]:
        raise ValueError("latent_derivative must share the sample axis")
    if ridge_penalty < 0.0:
        raise ValueError("ridge_penalty must be nonnegative")
    features = _latent_evolution_features(
        coordinates,
        amplitudes,
        drive,
        coordinate_scales,
        feature_family=feature_family,
    )
    if training_mask is None:
        selected = np.ones(features.shape[0], dtype=bool)
    else:
        selected = np.asarray(training_mask, dtype=bool)
        if selected.shape != (features.shape[0],):
            raise ValueError("training_mask must match the sample axis")
    if np.count_nonzero(selected) < 2:
        raise ValueError("at least two training samples are required")

    training_features = features[selected]
    training_derivatives = derivatives[selected]
    center = np.mean(training_features, axis=0)
    scale = np.std(training_features, axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (training_features - center) / scale
    design = np.column_stack((np.ones(standardized.shape[0]), standardized))
    if ridge_penalty == 0.0:
        coefficients = np.linalg.lstsq(
            design,
            training_derivatives,
            rcond=None,
        )[0]
    else:
        penalty = np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + ridge_penalty * penalty,
            design.T @ training_derivatives,
        )
    return LatentSourceEvolutionModel(
        feature_family=feature_family,
        coefficients=coefficients,
        feature_center=center,
        feature_scale=scale,
        coordinate_scales=np.asarray(coordinate_scales, dtype=float).copy(),
        latent_rank=amplitudes.shape[1],
        ridge_penalty=float(ridge_penalty),
    )


def predict_latent_source_evolution(
    model: LatentSourceEvolutionModel,
    coordinates: np.ndarray,
    latent: np.ndarray,
    drive: np.ndarray,
) -> np.ndarray:
    """Evaluate a fitted latent-source velocity model."""

    amplitudes = np.asarray(latent, dtype=float)
    if amplitudes.ndim != 2 or amplitudes.shape[1] != model.latent_rank:
        raise ValueError("latent amplitudes do not match the fitted model")
    features = _latent_evolution_features(
        coordinates,
        amplitudes,
        drive,
        model.coordinate_scales,
        feature_family=model.feature_family,
    )
    standardized = (features - model.feature_center) / model.feature_scale
    design = np.column_stack((np.ones(standardized.shape[0]), standardized))
    return design @ model.coefficients


def fit_second_order_latent_source_evolution(
    coordinates: np.ndarray,
    source: np.ndarray,
    source_rate: np.ndarray,
    source_acceleration: np.ndarray,
    drive: np.ndarray,
    coordinate_scales: np.ndarray,
    *,
    feature_family: str,
    ridge_penalty: float,
    training_mask: np.ndarray | None = None,
) -> SecondOrderLatentSourceEvolutionModel:
    """Fit only the acceleration in a second-order latent-source model."""

    amplitudes = np.asarray(source, dtype=float)
    rates = np.asarray(source_rate, dtype=float)
    accelerations = np.asarray(source_acceleration, dtype=float)
    if amplitudes.ndim != 2 or rates.shape != amplitudes.shape:
        raise ValueError("source and source_rate must share a 2D shape")
    if accelerations.shape != amplitudes.shape:
        raise ValueError("source_acceleration must match source")
    latent_state = np.concatenate((amplitudes, rates), axis=1)
    acceleration_model = fit_latent_source_evolution(
        coordinates,
        latent_state,
        accelerations,
        drive,
        coordinate_scales,
        feature_family=feature_family,
        ridge_penalty=ridge_penalty,
        training_mask=training_mask,
    )
    return SecondOrderLatentSourceEvolutionModel(
        acceleration_model=acceleration_model,
        source_rank=amplitudes.shape[1],
    )


def predict_second_order_latent_source_evolution(
    model: SecondOrderLatentSourceEvolutionModel,
    coordinates: np.ndarray,
    source: np.ndarray,
    source_rate: np.ndarray,
    drive: np.ndarray,
) -> np.ndarray:
    """Return ``(dot(z), dot(p))`` with the kinematic block exact."""

    amplitudes = np.asarray(source, dtype=float)
    rates = np.asarray(source_rate, dtype=float)
    if amplitudes.ndim != 2 or amplitudes.shape[1] != model.source_rank:
        raise ValueError("source amplitudes do not match the fitted model")
    if rates.shape != amplitudes.shape:
        raise ValueError("source_rate must match source")
    latent_state = np.concatenate((amplitudes, rates), axis=1)
    acceleration = predict_latent_source_evolution(
        model.acceleration_model,
        coordinates,
        latent_state,
        drive,
    )
    return np.concatenate((rates, acceleration), axis=1)


def _second_order_homogeneous_matrix(
    source_coefficients: np.ndarray,
    rate_coefficients: np.ndarray,
) -> np.ndarray:
    rank = source_coefficients.shape[0]
    return np.block(
        [
            [np.zeros((rank, rank)), np.eye(rank)],
            [source_coefficients, rate_coefficients],
        ]
    )


def latent_homogeneous_eigenvalues(
    model: StableSecondOrderLatentSourceEvolutionModel,
) -> np.ndarray:
    """Return eigenvalues of the unforced latent oscillator block."""

    return np.linalg.eigvals(
        _second_order_homogeneous_matrix(
            model.source_coefficients,
            model.rate_coefficients,
        )
    )


def fit_stable_second_order_latent_source_evolution(
    coordinates: np.ndarray,
    source: np.ndarray,
    source_rate: np.ndarray,
    source_acceleration: np.ndarray,
    drive: np.ndarray,
    coordinate_scales: np.ndarray,
    *,
    ridge_penalty: float,
    stability_margin: float,
    training_mask: np.ndarray | None = None,
) -> StableSecondOrderLatentSourceEvolutionModel:
    """Fit an affine latent oscillator and shift its poles to the left."""

    if stability_margin < 0.0:
        raise ValueError("stability_margin must be nonnegative")
    states = np.asarray(coordinates, dtype=float)
    amplitudes = np.asarray(source, dtype=float)
    rates = np.asarray(source_rate, dtype=float)
    accelerations = np.asarray(source_acceleration, dtype=float)
    drive_values = np.asarray(drive, dtype=float)
    scales = np.asarray(coordinate_scales, dtype=float)
    if amplitudes.ndim != 2 or rates.shape != amplitudes.shape:
        raise ValueError("source and source_rate must share a 2D shape")
    if accelerations.shape != amplitudes.shape:
        raise ValueError("source_acceleration must match source")
    if training_mask is None:
        selected = np.ones(amplitudes.shape[0], dtype=bool)
    else:
        selected = np.asarray(training_mask, dtype=bool)
        if selected.shape != (amplitudes.shape[0],):
            raise ValueError("training_mask must match the sample axis")

    unconstrained = fit_second_order_latent_source_evolution(
        states,
        amplitudes,
        rates,
        accelerations,
        drive_values,
        scales,
        feature_family="state_latent_affine",
        ridge_penalty=ridge_penalty,
        training_mask=selected,
    ).acceleration_model
    raw_coefficients = (
        unconstrained.coefficients[1:] / unconstrained.feature_scale[:, None]
    )
    rank = amplitudes.shape[1]
    latent_start = 31
    latent_stop = latent_start + 2 * rank
    latent_coefficients = raw_coefficients[latent_start:latent_stop].T
    source_coefficients = latent_coefficients[:, :rank]
    rate_coefficients = latent_coefficients[:, rank:]
    before = _second_order_homogeneous_matrix(
        source_coefficients,
        rate_coefficients,
    )
    maximum_real_before = float(np.max(np.real(np.linalg.eigvals(before))))
    shift = max(0.0, maximum_real_before + stability_margin)
    stable_rate = rate_coefficients - 2.0 * shift * np.eye(rank)
    stable_source = (
        source_coefficients
        + shift * rate_coefficients
        - shift**2 * np.eye(rank)
    )

    forcing_design = np.column_stack(
        (
            np.ones(np.count_nonzero(selected)),
            states[selected] / scales,
            drive_values[selected],
        )
    )
    forcing_target = (
        accelerations[selected]
        - amplitudes[selected] @ stable_source.T
        - rates[selected] @ stable_rate.T
    )
    if ridge_penalty == 0.0:
        forcing_coefficients = np.linalg.lstsq(
            forcing_design,
            forcing_target,
            rcond=None,
        )[0]
    else:
        penalty = np.eye(forcing_design.shape[1])
        penalty[0, 0] = 0.0
        forcing_coefficients = np.linalg.solve(
            forcing_design.T @ forcing_design + ridge_penalty * penalty,
            forcing_design.T @ forcing_target,
        )
    return StableSecondOrderLatentSourceEvolutionModel(
        acceleration_intercept=forcing_coefficients[0],
        state_coefficients=forcing_coefficients[1:32].T,
        source_coefficients=stable_source,
        rate_coefficients=stable_rate,
        drive_coefficients=forcing_coefficients[32],
        coordinate_scales=scales.copy(),
        ridge_penalty=float(ridge_penalty),
        stability_margin=float(stability_margin),
        stability_shift=float(shift),
        maximum_real_part_before_shift=maximum_real_before,
    )


def predict_stable_second_order_latent_source_evolution(
    model: StableSecondOrderLatentSourceEvolutionModel,
    coordinates: np.ndarray,
    source: np.ndarray,
    source_rate: np.ndarray,
    drive: np.ndarray,
) -> np.ndarray:
    """Return the exact kinematics and stable fitted source acceleration."""

    states = np.asarray(coordinates, dtype=float)
    amplitudes = np.asarray(source, dtype=float)
    rates = np.asarray(source_rate, dtype=float)
    drive_values = np.asarray(drive, dtype=float)
    rank = model.source_coefficients.shape[0]
    if states.ndim != 2 or states.shape[1] != 31:
        raise ValueError("coordinates must have shape (samples, 31)")
    if amplitudes.shape != (states.shape[0], rank):
        raise ValueError("source amplitudes do not match the fitted model")
    if rates.shape != amplitudes.shape:
        raise ValueError("source_rate must match source")
    if drive_values.shape != (states.shape[0],):
        raise ValueError("drive must match the sample axis")
    acceleration = (
        model.acceleration_intercept
        + (states / model.coordinate_scales) @ model.state_coefficients.T
        + amplitudes @ model.source_coefficients.T
        + rates @ model.rate_coefficients.T
        + drive_values[:, None] * model.drive_coefficients
    )
    return np.concatenate((rates, acceleration), axis=1)


def normalized_vector_rms_error(
    predicted: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Return vector RMS error normalized by reference variation."""

    prediction = np.asarray(predicted, dtype=float)
    target = np.asarray(reference, dtype=float)
    if prediction.shape != target.shape or target.ndim != 2:
        raise ValueError("predicted and reference must share a 2D shape")
    centered = target - np.mean(target, axis=0)
    scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    if scale <= np.finfo(float).tiny:
        raise ValueError("reference has no resolved variation")
    error = prediction - target
    return float(np.sqrt(np.mean(np.sum(error**2, axis=1))) / scale)


def select_latent_source_evolution(
    coordinates: np.ndarray,
    latent: np.ndarray,
    latent_derivative: np.ndarray,
    drive: np.ndarray,
    coordinate_scales: np.ndarray,
    *,
    training_mask: np.ndarray,
    validation_mask: np.ndarray,
    feature_families: tuple[str, ...],
    ridge_penalties: tuple[float, ...],
    selection_relative_slack: float = 0.01,
) -> LatentEvolutionSelection:
    """Select the smallest near-best model on a declared validation split."""

    derivatives = np.asarray(latent_derivative, dtype=float)
    training = np.asarray(training_mask, dtype=bool)
    validation = np.asarray(validation_mask, dtype=bool)
    if derivatives.ndim != 2:
        raise ValueError("latent_derivative must be two-dimensional")
    if training.shape != (derivatives.shape[0],):
        raise ValueError("training_mask must match the sample axis")
    if validation.shape != training.shape:
        raise ValueError("validation_mask must match the sample axis")
    if np.any(training & validation):
        raise ValueError("training and validation masks must be disjoint")
    if not np.any(training) or not np.any(validation):
        raise ValueError("training and validation selections must be nonempty")
    if not feature_families or not ridge_penalties:
        raise ValueError("at least one candidate must be declared")
    if selection_relative_slack < 0.0:
        raise ValueError("selection_relative_slack must be nonnegative")

    candidates: list[
        tuple[
            LatentEvolutionCandidateScore,
            LatentSourceEvolutionModel,
        ]
    ] = []
    for feature_family in feature_families:
        for ridge_penalty in ridge_penalties:
            model = fit_latent_source_evolution(
                coordinates,
                latent,
                derivatives,
                drive,
                coordinate_scales,
                feature_family=feature_family,
                ridge_penalty=ridge_penalty,
                training_mask=training,
            )
            training_prediction = predict_latent_source_evolution(
                model,
                np.asarray(coordinates)[training],
                np.asarray(latent)[training],
                np.asarray(drive)[training],
            )
            validation_prediction = predict_latent_source_evolution(
                model,
                np.asarray(coordinates)[validation],
                np.asarray(latent)[validation],
                np.asarray(drive)[validation],
            )
            score = LatentEvolutionCandidateScore(
                feature_family=feature_family,
                ridge_penalty=float(ridge_penalty),
                feature_count=int(model.coefficients.shape[0] - 1),
                training_normalized_rms=normalized_vector_rms_error(
                    training_prediction,
                    derivatives[training],
                ),
                validation_normalized_rms=normalized_vector_rms_error(
                    validation_prediction,
                    derivatives[validation],
                ),
            )
            candidates.append((score, model))

    best_error = min(score.validation_normalized_rms for score, _ in candidates)
    eligible = [
        (score, model)
        for score, model in candidates
        if score.validation_normalized_rms
        <= best_error * (1.0 + selection_relative_slack) + 1e-15
    ]
    selected_score, selected_model = min(
        eligible,
        key=lambda item: (
            item[0].feature_count,
            item[0].validation_normalized_rms,
            item[0].ridge_penalty,
        ),
    )
    return LatentEvolutionSelection(
        model=selected_model,
        candidates=tuple(score for score, _ in candidates),
        training_normalized_rms=selected_score.training_normalized_rms,
        validation_normalized_rms=selected_score.validation_normalized_rms,
    )


__all__ = [
    "LatentSourceBasis",
    "LatentEvolutionCandidateScore",
    "LatentEvolutionSelection",
    "LatentSourceEvolutionModel",
    "SecondOrderLatentSourceEvolutionModel",
    "StableSecondOrderLatentSourceEvolutionModel",
    "estimate_time_derivative",
    "fit_latent_source_basis",
    "fit_latent_source_evolution",
    "fit_second_order_latent_source_evolution",
    "fit_stable_second_order_latent_source_evolution",
    "latent_homogeneous_eigenvalues",
    "normalized_vector_rms_error",
    "predict_latent_source_evolution",
    "predict_second_order_latent_source_evolution",
    "predict_stable_second_order_latent_source_evolution",
    "project_missing_source",
    "reconstruct_missing_source",
    "select_latent_source_evolution",
]
