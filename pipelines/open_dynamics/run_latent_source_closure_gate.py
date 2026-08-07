"""Propagate and score the frozen five-mode latent ``C``-source candidate.

The online right-hand sides use only the propagated states, frozen model
coefficients, and declared drive.  Apart from the declared initial preparation
and score grid, stored exact trajectories are opened only after both candidate
lanes have completed and are used solely for scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from paper5.stability.cone_correction import (
    structured_electron_phonon_barrier_correction,
)
from paper5.stability.hubbard_dimer import DimerParameters, GaussianSineDrive
from paper5.stability.initial_condition_sensitivity import (
    physicality_diagnostics,
)
from paper5.stability.latent_source_closure import (
    LatentSourceBasis,
    StableSecondOrderLatentSourceEvolutionModel,
    reconstruct_missing_source,
)
from paper5.stability.latent_source_propagation import (
    integrate_latent_augmented_rk4,
    latent_augmented_velocity,
)
from paper5.stability.matrix_reference import (
    closed_scalar_to_matrix_state,
    local_holstein_couplings,
    pauli_repaired_closed_scalar_rhs,
)
from paper5.stability.multi_coherent_scores import (
    closed_coordinate_distance,
    closed_coordinate_error_scores,
)


FINAL_TIME = 20.0
TIME_STEP = 0.01
SAMPLE_STEP = 0.05
ACTIVATION_MARGIN = 1e-5
BARRIER_RATE = 5.0
CONE_TOLERANCE = 1e-8
MAXIMUM_CONSTRAINTS = 128
PHYSICALITY_TOLERANCE = 1e-8
TRACE_TOLERANCE = 1e-8
ENERGY_NAMES = (
    "electronic",
    "phonon",
    "electron_phonon",
    "internal_total",
    "drive",
    "instantaneous_total",
)


class _ProtocolParameters:
    """Delegate static parameters while replacing the pulse protocol."""

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


@dataclass
class _ControllerAudit:
    parameters: Any
    evaluations: int = 0
    active_evaluations: int = 0
    nonconverged_evaluations: int = 0
    maximum_correction_norm: float = 0.0
    sum_squared_correction_norm: float = 0.0
    maximum_correction_energy_flux: float = 0.0
    minimum_corrected_joint_barrier: float = np.inf
    maximum_constraint_count: int = 0

    def __call__(
        self,
        _time: float,
        state: np.ndarray,
        proposed_velocity: np.ndarray,
    ) -> np.ndarray:
        result = structured_electron_phonon_barrier_correction(
            state,
            proposed_velocity,
            self.parameters,
            activation_margin=ACTIVATION_MARGIN,
            barrier_rate=BARRIER_RATE,
            energy_neutral=True,
            preserve_correlation_trace=True,
            cone_tolerance=CONE_TOLERANCE,
            maximum_constraints=MAXIMUM_CONSTRAINTS,
            correction_metric="euclidean",
        )
        self.evaluations += 1
        correction_norm = result.correction_norm
        self.maximum_correction_norm = max(
            self.maximum_correction_norm,
            correction_norm,
        )
        self.sum_squared_correction_norm += correction_norm**2
        self.maximum_correction_energy_flux = max(
            self.maximum_correction_energy_flux,
            abs(result.correction_energy_flux),
        )
        self.minimum_corrected_joint_barrier = min(
            self.minimum_corrected_joint_barrier,
            result.corrected_joint_barrier_minimum_eigenvalue,
        )
        self.maximum_constraint_count = max(
            self.maximum_constraint_count,
            result.constraint_count,
        )
        if correction_norm > 1e-14:
            self.active_evaluations += 1
        if not result.converged:
            self.nonconverged_evaluations += 1
            raise RuntimeError(
                "joint electron-phonon correction failed: "
                f"minimum={result.corrected_joint_barrier_minimum_eigenvalue}"
            )
        return np.asarray(result.correction_coordinates, dtype=float)

    def summary(self) -> dict[str, float | int]:
        return {
            "evaluations": self.evaluations,
            "active_evaluations": self.active_evaluations,
            "active_fraction": self.active_evaluations / self.evaluations,
            "nonconverged_evaluations": self.nonconverged_evaluations,
            "maximum_correction_norm": self.maximum_correction_norm,
            "rms_correction_norm": float(
                np.sqrt(self.sum_squared_correction_norm / self.evaluations)
            ),
            "maximum_correction_energy_flux": (
                self.maximum_correction_energy_flux
            ),
            "minimum_corrected_joint_barrier": (
                self.minimum_corrected_joint_barrier
            ),
            "maximum_constraint_count": self.maximum_constraint_count,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _time_rms(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    duration = float(times[-1] - times[0])
    return np.sqrt(np.trapezoid(values**2, times, axis=0) / duration)


def _energy_components(
    coordinates: np.ndarray,
    parameters: Any,
    times: np.ndarray,
) -> np.ndarray:
    coupling = local_holstein_couplings(parameters)
    bare_electron = np.array(
        [[0.0, -parameters.hopping], [-parameters.hopping, 0.0]],
        dtype=complex,
    )
    result = np.empty((times.size, len(ENERGY_NAMES)), dtype=float)
    for index, (time, row) in enumerate(zip(times, coordinates, strict=True)):
        state = closed_scalar_to_matrix_state(row)
        rho = np.asarray(state.electron_density, dtype=complex)
        coherent = np.asarray(state.coherent_phonon, dtype=complex)
        phonon = np.asarray(state.phonon_density, dtype=complex)
        correlation = np.asarray(
            state.electron_phonon_correlation,
            dtype=complex,
        )
        electronic = 2.0 * np.trace(bare_electron @ rho).real
        phonon_energy = parameters.omega_ph * (
            np.vdot(coherent, coherent).real + np.trace(phonon).real
        )
        interaction_amplitude = 0.0j
        for mode in range(2):
            for one in range(2):
                for two in range(2):
                    interaction_amplitude += coupling[mode, one, two] * (
                        coherent[mode] * rho[two, one]
                        + correlation[mode, two, one]
                    )
        electron_phonon = 4.0 * interaction_amplitude.real
        internal = electronic + phonon_energy + electron_phonon
        drive_energy = (
            parameters.drive_difference(float(time))
            * (rho[0, 0] - rho[1, 1]).real
        )
        result[index] = (
            electronic,
            phonon_energy,
            electron_phonon,
            internal,
            drive_energy,
            internal + drive_energy,
        )
    return result


def _sample_physicality(
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    margins = np.empty((coordinates.shape[0], 4), dtype=float)
    traces = np.empty(coordinates.shape[0], dtype=float)
    for index, row in enumerate(coordinates):
        margins[index], traces[index] = physicality_diagnostics(row)
    return margins, traces


def _source_normalized_rms(
    times: np.ndarray,
    candidate: np.ndarray,
    exact: np.ndarray,
    scales: np.ndarray,
) -> float:
    scaled_exact = exact / scales
    centered = scaled_exact - np.mean(scaled_exact, axis=0)
    fluctuation_scale = float(
        np.sqrt(np.mean(np.sum(centered**2, axis=1)))
    )
    scaled_error = (candidate - exact) / scales
    error_size = np.linalg.norm(scaled_error, axis=1)
    return float(_time_rms(times, error_size) / fluctuation_scale)


def _lane_score(
    times: np.ndarray,
    coordinates: np.ndarray,
    exact_coordinates: np.ndarray,
    coordinate_scales: np.ndarray,
    energies: np.ndarray,
    exact_energies: np.ndarray,
    margins: np.ndarray,
    traces: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    error_scores = closed_coordinate_error_scores(
        times,
        coordinates,
        exact_coordinates,
        coordinate_scales,
        interval=(float(times[0]), float(times[-1])),
    )
    distances = np.asarray(
        [
            closed_coordinate_distance(candidate, exact, coordinate_scales)
            for candidate, exact in zip(
                coordinates,
                exact_coordinates,
                strict=True,
            )
        ],
        dtype=float,
    )
    occupation = 0.5 * (1.0 + coordinates[:, 0])
    exact_occupation = 0.5 * (1.0 + exact_coordinates[:, 0])
    energy_errors = energies - exact_energies
    post_pulse = times >= 12.0 - 1e-12
    return (
        {
            "equal_five_block_time_rms_distance": float(
                _time_rms(times, distances)
            ),
            "equal_five_block_maximum_distance": float(np.max(distances)),
            "electron_trace_distance_maximum": (
                error_scores.electron_trace_distance_maximum
            ),
            "block_normalized_time_rms": error_scores.block_rms,
            "block_normalized_maximum": error_scores.block_maximum,
            "site_occupation_time_rms_error": float(
                _time_rms(times, occupation - exact_occupation)
            ),
            "site_occupation_maximum_error": float(
                np.max(np.abs(occupation - exact_occupation))
            ),
            "energy_time_rms_error": {
                name: float(value)
                for name, value in zip(
                    ENERGY_NAMES,
                    _time_rms(times, energy_errors),
                    strict=True,
                )
            },
            "energy_maximum_error": {
                name: float(np.max(np.abs(energy_errors[:, index])))
                for index, name in enumerate(ENERGY_NAMES)
            },
            "post_pulse_internal_energy_range": float(
                np.ptp(energies[post_pulse, ENERGY_NAMES.index("internal_total")])
            ),
            "physicality": {
                "minimum_electron_lower_margin": float(np.min(margins[:, 0])),
                "minimum_electron_upper_margin": float(np.min(margins[:, 1])),
                "minimum_boson_moment_eigenvalue": float(np.min(margins[:, 2])),
                "minimum_joint_moment_eigenvalue": float(np.min(margins[:, 3])),
                "maximum_correlation_trace_residual": float(np.max(traces)),
            },
        },
        distances,
    )


def _load_frozen_model(
    arrays: Any,
    metrics: dict[str, Any],
) -> tuple[LatentSourceBasis, StableSecondOrderLatentSourceEvolutionModel]:
    basis = LatentSourceBasis(
        center=np.asarray(arrays["source_center"], dtype=float),
        basis=np.asarray(arrays["source_basis"], dtype=float),
        coordinate_scales=np.asarray(
            arrays["source_coordinate_scales"],
            dtype=float,
        ),
        singular_values=np.asarray(arrays["source_singular_values"], dtype=float),
    )
    model_metrics = metrics["model"]
    model = StableSecondOrderLatentSourceEvolutionModel(
        acceleration_intercept=np.asarray(
            arrays["acceleration_intercept"],
            dtype=float,
        ),
        state_coefficients=np.asarray(arrays["state_coefficients"], dtype=float),
        source_coefficients=np.asarray(
            arrays["source_coefficients"],
            dtype=float,
        ),
        rate_coefficients=np.asarray(arrays["rate_coefficients"], dtype=float),
        drive_coefficients=np.asarray(arrays["drive_coefficients"], dtype=float),
        coordinate_scales=np.asarray(arrays["coordinate_scales"], dtype=float),
        ridge_penalty=float(model_metrics["selected_ridge_penalty"]),
        stability_margin=float(model_metrics["stability_margin"]),
        stability_shift=float(model_metrics["stability_shift"]),
        maximum_real_part_before_shift=float(
            model_metrics["maximum_real_part_before_shift"]
        ),
    )
    return basis, model


def run_gate(
    source_directory: Path,
    model_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    source_arrays_path = source_directory / "trajectory_closure_identifiability.npz"
    source_metrics_path = source_directory / "metrics.json"
    model_arrays_path = model_directory / "latent_source_closure.npz"
    model_metrics_path = model_directory / "metrics.json"
    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    model_metrics = json.loads(model_metrics_path.read_text(encoding="utf-8"))
    if not model_metrics["derivative_gate_passed"]:
        raise RuntimeError("the frozen derivative gate did not pass")

    with np.load(source_arrays_path) as source_arrays, np.load(
        model_arrays_path
    ) as model_arrays:
        source_times = np.asarray(source_arrays["times"], dtype=float)
        selected = source_times <= FINAL_TIME + 1e-12
        score_times = source_times[selected]
        initial_moments = np.asarray(
            source_arrays["dop853_closed"][0, 0],
            dtype=float,
        )
        model_times = np.asarray(model_arrays["times"], dtype=float)
        if not np.array_equal(model_times, source_times):
            raise RuntimeError("frozen model and exact-source grids differ")
        basis, model = _load_frozen_model(model_arrays, model_metrics)
        initial_latent = np.asarray(
            model_arrays["initial_latent_source"][0],
            dtype=float,
        )
        initial_rate = np.asarray(
            model_arrays["initial_latent_rate"][0],
            dtype=float,
        )

    if not np.allclose(
        np.diff(score_times),
        SAMPLE_STEP,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError("stored exact score grid does not match SAMPLE_STEP")
    parameter_data = source_metrics["parameters"]
    base_parameters = DimerParameters(
        hopping=float(parameter_data["hopping"]),
        gamma=float(parameter_data["gamma"]),
        lambda_ep=float(parameter_data["lambda_ep"]),
        drive_amplitude=float(parameter_data["drive_amplitude"]),
        pulse_width=float(parameter_data["pulse_width"]),
    )
    drive_data = source_metrics["drive_protocol"]
    drive = GaussianSineDrive(
        amplitude=float(drive_data["amplitude"]),
        pulse_width=float(drive_data["pulse_width"]),
        delays=tuple(float(value) for value in drive_data["delays"]),
    )
    parameters = _ProtocolParameters(base_parameters, drive)

    baseline_controller = _ControllerAudit(parameters)

    def moment_rhs(time: float, state: np.ndarray) -> np.ndarray:
        return pauli_repaired_closed_scalar_rhs(time, state, parameters)

    def baseline_rhs(time: float, state: np.ndarray) -> np.ndarray:
        proposed = moment_rhs(time, state)
        return proposed + baseline_controller(time, state, proposed)

    baseline = integrate_latent_augmented_rk4(
        baseline_rhs,
        initial_moments,
        final_time=FINAL_TIME,
        time_step=TIME_STEP,
        sample_step=SAMPLE_STEP,
    )

    latent_controller = _ControllerAudit(parameters)

    def latent_rhs(time: float, state: np.ndarray) -> np.ndarray:
        return latent_augmented_velocity(
            time,
            state,
            moment_rhs=moment_rhs,
            drive_difference=drive.difference,
            basis=basis,
            model=model,
            moment_correction=latent_controller,
        )

    latent = integrate_latent_augmented_rk4(
        latent_rhs,
        np.concatenate((initial_moments, initial_latent, initial_rate)),
        final_time=FINAL_TIME,
        time_step=TIME_STEP,
        sample_step=SAMPLE_STEP,
    )
    if not (
        np.allclose(baseline.times, score_times, atol=1e-12, rtol=0.0)
        and np.allclose(latent.times, score_times, atol=1e-12, rtol=0.0)
    ):
        raise RuntimeError("propagated and stored score grids differ")

    baseline_coordinates = baseline.states
    latent_coordinates = latent.states[:, :31]
    latent_source = reconstruct_missing_source(latent.states[:, 31:36], basis)

    # Full exact-reference access begins here, after both autonomous lanes complete.
    with np.load(source_arrays_path) as source_arrays:
        exact_coordinates = np.asarray(
            source_arrays["dop853_closed"][0, selected],
            dtype=float,
        )
        exact_source = np.asarray(
            source_arrays["dop853_target_source"][0, selected],
            dtype=float,
        )
        coordinate_scales = np.asarray(
            source_arrays["coordinate_scales"],
            dtype=float,
        )
    exact_energies = _energy_components(exact_coordinates, parameters, score_times)
    baseline_energies = _energy_components(
        baseline_coordinates,
        parameters,
        score_times,
    )
    latent_energies = _energy_components(
        latent_coordinates,
        parameters,
        score_times,
    )
    baseline_margins, baseline_traces = _sample_physicality(
        baseline_coordinates
    )
    latent_margins, latent_traces = _sample_physicality(latent_coordinates)
    baseline_score, baseline_distances = _lane_score(
        score_times,
        baseline_coordinates,
        exact_coordinates,
        coordinate_scales,
        baseline_energies,
        exact_energies,
        baseline_margins,
        baseline_traces,
    )
    latent_score, latent_distances = _lane_score(
        score_times,
        latent_coordinates,
        exact_coordinates,
        coordinate_scales,
        latent_energies,
        exact_energies,
        latent_margins,
        latent_traces,
    )
    baseline_source_error = _source_normalized_rms(
        score_times,
        np.zeros_like(exact_source),
        exact_source,
        basis.coordinate_scales,
    )
    latent_source_error = _source_normalized_rms(
        score_times,
        latent_source,
        exact_source,
        basis.coordinate_scales,
    )

    ratios = {
        "missing_source_rms": latent_source_error / baseline_source_error,
        "C_block_rms": (
            latent_score["block_normalized_time_rms"]["C"]
            / baseline_score["block_normalized_time_rms"]["C"]
        ),
        "equal_five_block_rms": (
            latent_score["equal_five_block_time_rms_distance"]
            / baseline_score["equal_five_block_time_rms_distance"]
        ),
        "site_occupation_rms": (
            latent_score["site_occupation_time_rms_error"]
            / baseline_score["site_occupation_time_rms_error"]
        ),
        "internal_energy_rms": (
            latent_score["energy_time_rms_error"]["internal_total"]
            / baseline_score["energy_time_rms_error"]["internal_total"]
        ),
    }
    latent_physicality = latent_score["physicality"]
    gates = {
        "all_latent_controller_solves_converged": (
            latent_controller.nonconverged_evaluations == 0
        ),
        "latent_sampled_physicality": min(
            latent_physicality["minimum_electron_lower_margin"],
            latent_physicality["minimum_electron_upper_margin"],
            latent_physicality["minimum_boson_moment_eigenvalue"],
            latent_physicality["minimum_joint_moment_eigenvalue"],
        )
        >= -PHYSICALITY_TOLERANCE,
        "latent_correlation_trace": (
            latent_physicality["maximum_correlation_trace_residual"]
            <= TRACE_TOLERANCE
        ),
        "missing_source_improved": ratios["missing_source_rms"] < 1.0,
        "C_block_improved": ratios["C_block_rms"] < 1.0,
        "combined_retained_state_improved": ratios["equal_five_block_rms"] < 1.0,
        "site_occupation_improved": ratios["site_occupation_rms"] < 1.0,
        "internal_energy_improved": ratios["internal_energy_rms"] < 1.0,
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    arrays_output = output_directory / "latent_source_closure_gate.npz"
    np.savez_compressed(
        arrays_output,
        times=score_times,
        exact_coordinates=exact_coordinates,
        baseline_coordinates=baseline_coordinates,
        latent_coordinates=latent_coordinates,
        latent_source_amplitudes=latent.states[:, 31:36],
        latent_source_rates=latent.states[:, 36:41],
        exact_missing_source=exact_source,
        autonomous_missing_source=latent_source,
        exact_energies=exact_energies,
        baseline_energies=baseline_energies,
        latent_energies=latent_energies,
        baseline_physicality_margins=baseline_margins,
        latent_physicality_margins=latent_margins,
        baseline_correlation_trace_residuals=baseline_traces,
        latent_correlation_trace_residuals=latent_traces,
        baseline_equal_five_block_distances=baseline_distances,
        latent_equal_five_block_distances=latent_distances,
    )
    metrics = {
        "schema": "paper5.latent_source_closure.propagation_gate.v1",
        "classification": "exploratory_frozen_model_not_promoted",
        "question": (
            "Does the frozen five-mode latent source improve the corrected "
            "archive trajectory through t=20 without exact online input?"
        ),
        "protocol": {
            "final_time": FINAL_TIME,
            "time_step": TIME_STEP,
            "sample_step": SAMPLE_STEP,
            "integrator": "fixed-step RK4",
            "drive": drive_data,
            "initial_moments": "central exact cutoff-16 contraction at t=0",
            "initial_latent_source_and_rate": (
                "central exact-source projection and five-point initial rate"
            ),
            "online_exact_reference_access": False,
            "controller": {
                "metric": "euclidean",
                "activation_margin": ACTIVATION_MARGIN,
                "barrier_rate": BARRIER_RATE,
                "cone_tolerance": CONE_TOLERANCE,
                "maximum_constraints": MAXIMUM_CONSTRAINTS,
                "energy_neutral": True,
                "preserve_correlation_trace": True,
            },
        },
        "baseline": {
            "description": "autonomous Pauli repair plus physicality controller",
            "rhs_evaluations": baseline.rhs_evaluations,
            "controller": baseline_controller.summary(),
            "score": baseline_score,
            "missing_source_normalized_rms": baseline_source_error,
        },
        "latent": {
            "description": (
                "baseline plus frozen autonomous five-mode C source and rates"
            ),
            "total_state_dimension": 41,
            "rhs_evaluations": latent.rhs_evaluations,
            "controller": latent_controller.summary(),
            "score": latent_score,
            "missing_source_normalized_rms": latent_source_error,
        },
        "latent_to_baseline_error_ratios": ratios,
        "gates": gates,
        "propagation_gate_passed": bool(all(gates.values())),
        "limitations": [
            "The source basis and latent coefficients were fitted to the opened central exact trajectory.",
            "The latent initial source and rate were initialized from exact preparation data.",
            "This is one central preparation and one double-pulse protocol through t=20.",
            "A passed physicality gate does not establish an accurate closure.",
        ],
        "input_hashes": {
            str(source_arrays_path): _sha256(source_arrays_path),
            str(source_metrics_path): _sha256(source_metrics_path),
            str(model_arrays_path): _sha256(model_arrays_path),
            str(model_metrics_path): _sha256(model_metrics_path),
        },
    }
    metrics_path = output_directory / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "paper5.latent_source_closure.propagation_runtime.v1",
        "input_hashes": metrics["input_hashes"],
        "artifact_hashes": {
            arrays_output.name: _sha256(arrays_output),
            metrics_path.name: _sha256(metrics_path),
        },
    }
    (output_directory / "runtime_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    result = run_gate(
        args.source_directory,
        args.model_directory,
        args.output_directory,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
