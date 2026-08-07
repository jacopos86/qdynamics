"""Test trajectory-local closure predictability from a consumed exact score.

The command contracts analytic Schrödinger velocities from already stored
exact kets.  It performs no propagation, timing experiment, or holdout
rescore.  The target is the exact ``C`` velocity left after the autonomous
same-spin Pauli repair has been applied to the archive equation.
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

from paper5.stability.exact_reference import _build_exact_dimer_model
from paper5.stability.hubbard_dimer import (
    DimerParameters,
    GaussianSineDrive,
)
from paper5.stability.matrix_reference import (
    closed_scalar_to_matrix_state,
    pauli_repaired_matrix_dimer_rhs,
)
from paper5.stability.trajectory_closure_identifiability import (
    TrajectoryClosurePredictability,
    causal_history_metric_embedding,
    closed_state_metric_embedding,
    trajectory_closure_predictability,
    trajectory_source_predictability,
)


MEMBER_NAMES = ("central", "plus", "minus")


class _ProtocolParameters:
    """Delegate dimer parameters while replacing only the drive protocol."""

    def __init__(
        self,
        base: DimerParameters,
        drive: GaussianSineDrive,
    ) -> None:
        self._base = base
        self._drive = drive

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def drive_difference(self, time: float) -> float:
        return self._drive.difference(time)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _correlation_coordinates(correlation: np.ndarray) -> np.ndarray:
    """Pack one or more ``C`` matrices into the established 14 reals."""

    values = np.asarray(correlation, dtype=complex)
    if values.shape[-3:] != (2, 2, 2):
        raise ValueError("correlation must have trailing shape (2, 2, 2)")
    flat = values.reshape(-1, 2, 2, 2)
    result = np.empty((flat.shape[0], 14), dtype=float)
    shared_trace = 0.5 * (
        np.trace(flat[:, 0], axis1=-2, axis2=-1)
        + np.trace(flat[:, 1], axis1=-2, axis2=-1)
    )
    result[:, 0] = shared_trace.real
    result[:, 1] = shared_trace.imag
    offset = 2
    for mode in range(2):
        diagonal = flat[:, mode, 0, 0] - flat[:, mode, 1, 1]
        result[:, offset : offset + 6] = np.column_stack(
            (
                diagonal.real,
                diagonal.imag,
                flat[:, mode, 0, 1].real,
                flat[:, mode, 0, 1].imag,
                flat[:, mode, 1, 0].real,
                flat[:, mode, 1, 0].imag,
            )
        )
        offset += 6
    return result.reshape(values.shape[:-3] + (14,))


def _expectation_derivative_batch(
    states: np.ndarray,
    state_derivatives: np.ndarray,
    operator: Any,
) -> np.ndarray:
    state_columns = np.asarray(states, dtype=complex).T
    derivative_columns = np.asarray(state_derivatives, dtype=complex).T
    operator_state = operator @ state_columns
    operator_derivative = operator @ derivative_columns
    return np.sum(
        derivative_columns.conjugate() * operator_state
        + state_columns.conjugate() * operator_derivative,
        axis=0,
    )


def _exact_c_derivative_batch(
    model: Any,
    states: np.ndarray,
    times: np.ndarray,
    closed_coordinates: np.ndarray,
    drive: GaussianSineDrive,
) -> np.ndarray:
    """Contract analytic exact ket velocities into the centered ``dC``."""

    state_array = np.asarray(states, dtype=complex)
    time_array = np.asarray(times, dtype=float)
    closed = np.asarray(closed_coordinates, dtype=float)
    if state_array.ndim != 2 or state_array.shape[0] != time_array.size:
        raise ValueError("states must have shape (times, hilbert_dimension)")
    if closed.shape != (time_array.size, 31):
        raise ValueError("closed_coordinates must have shape (times, 31)")
    columns = state_array.T
    drive_values = np.asarray(
        [drive.difference(float(time)) for time in time_array],
        dtype=float,
    )
    derivative_columns = -1j * (
        model.static_hamiltonian @ columns
        + (model.drive_operator @ columns) * drive_values[None, :]
    )
    state_derivatives = derivative_columns.T

    electron_derivative = np.empty((time_array.size, 2, 2), dtype=complex)
    for row in range(2):
        for column in range(2):
            electron_derivative[:, row, column] = (
                _expectation_derivative_batch(
                    state_array,
                    state_derivatives,
                    model.electron_observables[row][column],
                )
            )
    coherent_derivative = np.column_stack(
        [
            _expectation_derivative_batch(
                state_array,
                state_derivatives,
                operator,
            )
            for operator in model.phonon_annihilation
        ]
    )
    mixed_derivative = np.empty(
        (time_array.size, 2, 2, 2),
        dtype=complex,
    )
    for mode in range(2):
        for row in range(2):
            for column in range(2):
                mixed_derivative[:, mode, row, column] = (
                    _expectation_derivative_batch(
                        state_array,
                        state_derivatives,
                        model.electron_phonon_observables[mode][row][column],
                    )
                )

    electron = np.empty_like(electron_derivative)
    electron[:, 0, 0] = 0.5 * (1.0 + closed[:, 0])
    electron[:, 1, 1] = 0.5 * (1.0 - closed[:, 0])
    electron[:, 0, 1] = closed[:, 1] + 1j * closed[:, 2]
    electron[:, 1, 0] = electron[:, 0, 1].conjugate()
    coherent = closed[:, 3:7:2] + 1j * closed[:, 4:7:2]
    centered_derivative = (
        mixed_derivative
        - coherent[:, :, None, None] * electron_derivative[:, None]
        - coherent_derivative[:, :, None, None] * electron[:, None]
    )
    return _correlation_coordinates(centered_derivative)


def _pauli_archive_c_derivative_batch(
    times: np.ndarray,
    closed_coordinates: np.ndarray,
    parameters: _ProtocolParameters,
) -> np.ndarray:
    derivatives = np.empty((len(times), 2, 2, 2), dtype=complex)
    for index, (time, coordinates) in enumerate(
        zip(times, closed_coordinates, strict=True)
    ):
        state = closed_scalar_to_matrix_state(coordinates)
        derivatives[index] = pauli_repaired_matrix_dimer_rhs(
            float(time),
            state,
            parameters,  # type: ignore[arg-type]
        ).electron_phonon_correlation
    return _correlation_coordinates(derivatives)


def _build_target_source(
    model: Any,
    times: np.ndarray,
    closed: np.ndarray,
    states: np.ndarray,
    parameters: DimerParameters,
    drive: GaussianSineDrive,
) -> tuple[np.ndarray, float]:
    protocol_parameters = _ProtocolParameters(parameters, drive)
    members = closed.shape[0]
    target = np.empty((members, times.size, 14), dtype=float)
    maximum_norm_error = 0.0
    for member in range(members):
        member_states = np.asarray(states[member], dtype=complex)
        norms = np.linalg.norm(member_states, axis=1)
        maximum_norm_error = max(
            maximum_norm_error,
            float(np.max(np.abs(norms - 1.0))),
        )
        exact_derivative = _exact_c_derivative_batch(
            model,
            member_states,
            times,
            closed[member],
            drive,
        )
        archive_derivative = _pauli_archive_c_derivative_batch(
            times,
            closed[member],
            protocol_parameters,
        )
        target[member] = exact_derivative - archive_derivative
    return target, maximum_norm_error


def _sample_identity(index: int, sample_count: int, times: np.ndarray) -> dict[str, Any]:
    member = index // sample_count
    time_index = index % sample_count
    return {
        "flat_index": int(index),
        "member": MEMBER_NAMES[member],
        "time_index": int(time_index),
        "time": float(times[time_index]),
    }


def _write_plot(
    path: Path,
    times: np.ndarray,
    target: np.ndarray,
    independent_target: np.ndarray,
    result: TrajectoryClosurePredictability,
    history_scan: list[dict[str, float | int]],
) -> None:
    target_scale = result.target_fluctuation_scale
    figure, axes_grid = plt.subplots(2, 2, figsize=(7.3, 4.65))
    axes = axes_grid.ravel()
    axis = axes[0]
    axis.scatter(
        result.nearest_state_distances,
        result.nearest_target_distances / target_scale,
        s=5,
        alpha=0.32,
        color="#6f3c8f",
        label="nearest cross-time recurrence",
    )
    axis.scatter(
        result.nearest_state_distances,
        result.nearest_reference_uncertainty_bounds / target_scale,
        s=4,
        alpha=0.22,
        color="#777777",
        label="two-reference bound",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("31-state distance")
    axis.set_ylabel("source difference / target RMS")
    axis.set_title("(a) recurrence consistency", loc="left")
    axis.legend(frameon=False, fontsize=6.2)

    flat_prediction = result.prediction_errors.reshape(target.shape[:2])
    flat_uncertainty = result.reference_uncertainties.reshape(target.shape[:2])
    axis = axes[1]
    for member, name in enumerate(MEMBER_NAMES):
        axis.plot(
            times,
            flat_prediction[member] / target_scale,
            linewidth=0.75,
            label=name,
        )
    axis.plot(
        times,
        np.max(flat_uncertainty, axis=0) / target_scale,
        color="#777777",
        linestyle=":",
        linewidth=0.9,
        label="exact-method spread",
    )
    axis.set_yscale("log")
    axis.set_xlabel(r"$t\,t_{\rm hop}$")
    axis.set_ylabel("normalized error")
    axis.set_title("(b) 8-neighbor prediction", loc="left")
    axis.legend(frameon=False, fontsize=6.2)

    axis = axes[2]
    for member, name in enumerate(MEMBER_NAMES):
        axis.plot(
            times,
            np.linalg.norm(target[member], axis=1),
            linewidth=0.75,
            label=name,
        )
    axis.plot(
        times,
        np.max(np.linalg.norm(target - independent_target, axis=2), axis=0),
        color="#777777",
        linestyle=":",
        linewidth=0.9,
        label="exact-method spread",
    )
    axis.set_yscale("log")
    axis.set_xlabel(r"$t\,t_{\rm hop}$")
    axis.set_ylabel(r"$\|R_C(t)\|_2$")
    axis.set_title("(c) missing source", loc="left")
    axis.legend(frameon=False, fontsize=6.2)

    axis = axes[3]
    lag_values = np.asarray(
        [record["lag_time"] for record in history_scan],
        dtype=float,
    )
    axis.plot(
        lag_values,
        [record["current_only_error"] for record in history_scan],
        marker="o",
        markersize=3,
        linewidth=0.9,
        label="current 31 moments",
    )
    axis.plot(
        lag_values,
        [record["causal_history_error"] for record in history_scan],
        marker="s",
        markersize=3,
        linewidth=0.9,
        label="current + lagged increment",
    )
    axis.plot(
        lag_values,
        [record["reference_error"] for record in history_scan],
        color="#777777",
        linestyle=":",
        linewidth=0.9,
        label="exact-method spread",
    )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel(r"history lag $\tau\,t_{\rm hop}$")
    axis.set_ylabel("normalized prediction RMS")
    axis.set_title("(d) causal-history scan", loc="left")
    axis.legend(frameon=False, fontsize=6.2)

    for axis in axes:
        axis.grid(alpha=0.17, linewidth=0.5)
    figure.text(
        0.5,
        0.012,
        "Stored exact states only; neighbors within four time units excluded.",
        ha="center",
        va="bottom",
        fontsize=6.3,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.115,
        top=0.93,
        hspace=0.42,
        wspace=0.38,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _causal_history_scan(
    times: np.ndarray,
    coordinates: np.ndarray,
    target: np.ndarray,
    independent_target: np.ndarray,
    scales: np.ndarray,
    *,
    minimum_time_separation: float,
) -> list[dict[str, float | int]]:
    sample_step = float(times[1] - times[0])
    base_embedding = closed_state_metric_embedding(coordinates, scales)
    records: list[dict[str, float | int]] = []
    for lag_time in (0.25, 0.5, 1.0, 2.0, 4.0):
        lag_steps = int(round(lag_time / sample_step))
        if lag_steps < 1 or not np.isclose(
            lag_steps * sample_step,
            lag_time,
            atol=1e-12,
            rtol=0.0,
        ):
            continue
        trimmed_times = times[lag_steps:]
        trimmed_target = target[:, lag_steps:]
        trimmed_independent = independent_target[:, lag_steps:]
        current_result = trajectory_source_predictability(
            trimmed_times,
            base_embedding[:, lag_steps:],
            trimmed_target,
            trimmed_independent,
            minimum_time_separation=minimum_time_separation,
            neighbor_count=8,
        )
        history_result = trajectory_source_predictability(
            trimmed_times,
            causal_history_metric_embedding(
                coordinates,
                scales,
                lag_steps=lag_steps,
            ),
            trimmed_target,
            trimmed_independent,
            minimum_time_separation=minimum_time_separation,
            neighbor_count=8,
        )
        records.append(
            {
                "lag_time": lag_time,
                "lag_steps": lag_steps,
                "current_only_error": current_result.normalized_prediction_rms,
                "causal_history_error": history_result.normalized_prediction_rms,
                "history_to_current_ratio": float(
                    history_result.normalized_prediction_rms
                    / current_result.normalized_prediction_rms
                ),
                "reference_error": history_result.normalized_reference_rms,
            }
        )
    if not records:
        raise ValueError("no declared history lag lies on the sampled grid")
    return records


def _source_subspace_scan(
    target: np.ndarray,
    scales: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Find a central-trajectory basis for the missing ``C`` source."""

    source = np.asarray(target, dtype=float)
    coordinate_scales = np.asarray(scales, dtype=float)[17:31]
    if source.ndim != 3 or source.shape[-1] != 14:
        raise ValueError("target must have shape (members, times, 14)")
    if coordinate_scales.shape != (14,) or np.any(coordinate_scales <= 0.0):
        raise ValueError("C-coordinate scales must be positive")
    center = np.mean(source[0], axis=0)
    scaled = (source - center) / coordinate_scales
    _, singular_values, right_vectors = np.linalg.svd(
        scaled[0],
        full_matrices=False,
    )
    centered_all = scaled - np.mean(scaled, axis=(0, 1), keepdims=True)
    fluctuation_scale = float(
        np.sqrt(np.mean(np.sum(centered_all**2, axis=2)))
    )
    total_training_variance = float(np.sum(singular_values**2))
    records: list[dict[str, float | int]] = []
    for rank in range(1, 11):
        basis = right_vectors[:rank]
        coefficients = np.einsum("mti,ri->mtr", scaled, basis)
        reconstruction = np.einsum("mtr,ri->mti", coefficients, basis)
        residual = reconstruction - scaled
        records.append(
            {
                "rank": rank,
                "central_training_variance_explained": float(
                    np.sum(singular_values[:rank] ** 2)
                    / total_training_variance
                ),
                "all_member_normalized_reconstruction_rms": float(
                    np.sqrt(np.mean(np.sum(residual**2, axis=2)))
                    / fluctuation_scale
                ),
            }
        )
    return (
        {
            "coordinate_metric": "development-scaled C coordinates",
            "basis_training_member": "central",
            "rank_scan": records,
            "rank_five": records[4],
        },
        right_vectors,
        center,
        singular_values,
    )


def run_audit(
    score_directory: Path,
    prepared_directory: Path,
    batch_directory: Path,
    output_directory: Path,
    *,
    sample_stride: int = 5,
    minimum_time_separation: float = 4.0,
) -> dict[str, Any]:
    if sample_stride < 1:
        raise ValueError("sample_stride must be positive")
    arrays_path = score_directory / "score_arrays.npz"
    score_summary_path = score_directory / "score_summary.json"
    prepared_path = prepared_directory / "pre_model_manifest.json"
    model_summary_path = batch_directory / "fine_central" / "summary.json"
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    model_summary = json.loads(model_summary_path.read_text(encoding="utf-8"))
    score_summary = json.loads(score_summary_path.read_text(encoding="utf-8"))
    parameters = DimerParameters(
        hopping=float(prepared["parameters"]["hopping"]),
        gamma=float(prepared["parameters"]["gamma"]),
        lambda_ep=float(prepared["parameters"]["lambda_ep"]),
        drive_amplitude=float(prepared["parameters"]["drive_amplitude"]),
        pulse_width=float(prepared["parameters"]["pulse_width"]),
    )
    drive_data = model_summary["parameters"]["drive_protocol"]
    drive = GaussianSineDrive(
        amplitude=float(drive_data["amplitude"]),
        pulse_width=float(drive_data["pulse_width"]),
        delays=tuple(float(value) for value in drive_data["delays"]),
    )
    with np.load(arrays_path) as arrays:
        all_times = np.asarray(arrays["times"], dtype=float)
        indices = np.arange(0, all_times.size, sample_stride, dtype=int)
        if indices[-1] != all_times.size - 1:
            indices = np.append(indices, all_times.size - 1)
        times = all_times[indices]
        scales = np.asarray(arrays["coordinate_scales"], dtype=float)
        dop_closed = np.asarray(
            arrays["exact_dop853_closed"][:, indices], dtype=float
        )
        midpoint_closed = np.asarray(
            arrays["exact_midpoint_closed"][:, indices], dtype=float
        )
        dop_states = np.asarray(
            arrays["exact_dop853_state_vectors"][:, indices], dtype=complex
        )
        midpoint_states = np.asarray(
            arrays["exact_midpoint_state_vectors"][:, indices], dtype=complex
        )

    model = _build_exact_dimer_model(
        parameters,
        phonon_cutoff=int(prepared["settings"]["phonon_cutoff"]),
    )
    dop_target, dop_norm_error = _build_target_source(
        model,
        times,
        dop_closed,
        dop_states,
        parameters,
        drive,
    )
    midpoint_target, midpoint_norm_error = _build_target_source(
        model,
        times,
        midpoint_closed,
        midpoint_states,
        parameters,
        drive,
    )
    result = trajectory_closure_predictability(
        times,
        dop_closed,
        dop_target,
        midpoint_target,
        scales,
        minimum_time_separation=minimum_time_separation,
        neighbor_count=8,
    )
    history_scan = _causal_history_scan(
        times,
        dop_closed,
        dop_target,
        midpoint_target,
        scales,
        minimum_time_separation=minimum_time_separation,
    )
    best_history = min(
        history_scan,
        key=lambda record: float(record["causal_history_error"]),
    )
    (
        source_subspace,
        source_basis,
        source_center,
        source_singular_values,
    ) = _source_subspace_scan(dop_target, scales)
    summary = result.summary()
    sample_count = times.size
    summary["tension_sample"] = _sample_identity(
        result.tension_sample_index,
        sample_count,
        times,
    )
    summary["tension_neighbor"] = _sample_identity(
        result.tension_neighbor_index,
        sample_count,
        times,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    arrays_output = output_directory / "trajectory_closure_identifiability.npz"
    np.savez_compressed(
        arrays_output,
        times=times,
        sample_indices=indices,
        coordinate_scales=scales,
        dop853_closed=dop_closed,
        midpoint_closed=midpoint_closed,
        dop853_target_source=dop_target,
        midpoint_target_source=midpoint_target,
        nearest_indices=result.nearest_indices,
        nearest_state_distances=result.nearest_state_distances,
        nearest_target_distances=result.nearest_target_distances,
        nearest_reference_uncertainty_bounds=(
            result.nearest_reference_uncertainty_bounds
        ),
        prediction_errors=result.prediction_errors,
        reference_uncertainties=result.reference_uncertainties,
        source_subspace_basis=source_basis,
        source_subspace_center=source_center,
        source_subspace_singular_values=source_singular_values,
    )
    plot_path = output_directory / "trajectory_closure_identifiability.pdf"
    _write_plot(
        plot_path,
        times,
        dop_target,
        midpoint_target,
        result,
        history_scan,
    )
    metrics = {
        "schema": "paper5.trajectory_closure_identifiability.v1",
        "classification": "exploratory_stored_exact_states_not_promoted",
        "score_status": score_summary["status"],
        "global_exact_instantaneous_31d_closure": (
            "already_ruled_out_by_finite_cutoff_witness"
        ),
        "question": (
            "Is the missing post-Pauli C source approximately predictable "
            "from the 31 moments on this visited trajectory family?"
        ),
        "sample_stride": sample_stride,
        "sample_step": float(times[1] - times[0]),
        "parameters": prepared["parameters"],
        "drive_protocol": drive_data,
        "maximum_exact_state_norm_error": {
            "dop853": dop_norm_error,
            "midpoint": midpoint_norm_error,
        },
        "predictability": summary,
        "causal_history_scan": history_scan,
        "best_causal_history_lag": best_history,
        "missing_source_subspace": source_subspace,
        "input_hashes": {
            str(arrays_path): _sha256(arrays_path),
            str(score_summary_path): _sha256(score_summary_path),
            str(prepared_path): _sha256(prepared_path),
            str(model_summary_path): _sha256(model_summary_path),
        },
    }
    metrics_path = output_directory / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runtime = {
        "schema": "paper5.trajectory_closure_identifiability_runtime.v1",
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
    parser.add_argument("--score-directory", type=Path, required=True)
    parser.add_argument("--prepared-directory", type=Path, required=True)
    parser.add_argument("--batch-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--minimum-time-separation", type=float, default=4.0)
    args = parser.parse_args()
    result = run_audit(
        args.score_directory,
        args.prepared_directory,
        args.batch_directory,
        args.output_directory,
        sample_stride=args.sample_stride,
        minimum_time_separation=args.minimum_time_separation,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
