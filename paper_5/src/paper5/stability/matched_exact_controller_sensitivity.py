"""Matched nearby-pair comparison for exact and controlled moment dynamics.

The initial 31-coordinate pairs are contractions of nearby exact
wavefunctions.  The exact and representability-controlled trajectories then
start from precisely the same retained moments.  No Benettin rescaling is
performed.  Controller diagnostics separate raw-closure and correction
contributions to the instantaneous growth of the lifted Frobenius distance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cone_correction import (
    closed_state_lifted_frobenius_metric,
    closed_state_lifted_frobenius_norm,
    structured_electron_phonon_barrier_correction,
)
from .hubbard_dimer import DimerParameters, FloatArray
from .initial_condition_sensitivity import physicality_diagnostics
from .matrix_reference import CLOSED_SCALAR_STATE_NAMES, closed_scalar_rhs

SECTOR_NAMES = ("electron_lower", "electron_upper", "joint_gram")
STAGE_NAMES = ("k1", "k2", "k3", "k4")


@dataclass(frozen=True)
class MatchedControllerSensitivity:
    """Unrescaled controlled trajectories and controller diagnostics."""

    step_times: FloatArray
    sample_times: FloatArray
    labels: tuple[str, ...]
    sampled_states: FloatArray
    step_frobenius_distances: FloatArray
    sampled_euclidean_distances: FloatArray
    sampled_frobenius_distances: FloatArray
    sampled_margins: FloatArray
    sampled_trace_residuals: FloatArray
    correction_euclidean_norms: FloatArray
    correction_frobenius_norms: FloatArray
    correction_block_norms: FloatArray
    constraint_counts: np.ndarray
    raw_barrier_minima: FloatArray
    corrected_barrier_minima: FloatArray
    raw_violated_sectors: np.ndarray
    binding_sectors: np.ndarray
    converged: np.ndarray
    raw_growth_contributions: FloatArray
    correction_growth_contributions: FloatArray
    total_growth_contributions: FloatArray
    k1_correction_differences: FloatArray


def _block_norms(correction: FloatArray) -> FloatArray:
    blocks = ((0, 3), (3, 7), (7, 11), (11, 17), (17, 31))
    return np.asarray(
        [np.linalg.norm(correction[start:stop]) for start, stop in blocks],
        dtype=float,
    )


def matched_controller_sensitivity(
    parameters: DimerParameters,
    initial_base: FloatArray,
    initial_shadows: FloatArray,
    *,
    labels: tuple[str, ...],
    final_time: float,
    time_step: float,
    sample_step: float,
    activation_margin: float = 1e-5,
    barrier_rate: float = 5.0,
    cone_tolerance: float = 1e-8,
    projection_tolerance: float = 1e-12,
    correction_metric: str = "euclidean",
) -> MatchedControllerSensitivity:
    """Propagate one base state and exact-induced nearby shadows by RK4."""

    state_size = len(CLOSED_SCALAR_STATE_NAMES)
    base = np.asarray(initial_base, dtype=float)
    shadows = np.asarray(initial_shadows, dtype=float)
    if base.shape != (state_size,):
        raise ValueError(f"initial_base must have shape {(state_size,)}")
    if shadows.ndim != 2 or shadows.shape[1] != state_size:
        raise ValueError("initial_shadows must have shape (cases, 31)")
    if len(labels) != shadows.shape[0]:
        raise ValueError("labels must match the number of shadow states")
    if final_time <= 0.0 or time_step <= 0.0 or sample_step <= 0.0:
        raise ValueError("integration times must be positive")
    step_count = int(round(final_time / time_step))
    sample_stride = int(round(sample_step / time_step))
    if not np.isclose(step_count * time_step, final_time, atol=1e-12):
        raise ValueError("time_step must divide final_time")
    if sample_stride <= 0 or not np.isclose(
        sample_stride * time_step,
        sample_step,
        atol=1e-12,
    ):
        raise ValueError("time_step must divide sample_step")
    if step_count % sample_stride:
        raise ValueError("sample_step must divide final_time")

    states = np.vstack((base, shadows))
    trajectory_count = states.shape[0]
    case_count = shadows.shape[0]
    sample_count = step_count // sample_stride + 1
    step_times = np.linspace(0.0, final_time, step_count + 1)
    sample_times = step_times[::sample_stride]
    sampled_states = np.empty(
        (sample_count, trajectory_count, state_size),
        dtype=float,
    )
    sampled_margins = np.empty((sample_count, trajectory_count, 4), dtype=float)
    sampled_traces = np.empty((sample_count, trajectory_count), dtype=float)
    sampled_euclidean = np.empty((case_count, sample_count), dtype=float)
    sampled_frobenius = np.empty_like(sampled_euclidean)
    step_frobenius = np.empty((case_count, step_count + 1), dtype=float)

    diagnostic_shape = (step_count, 4, trajectory_count)
    correction_euclidean = np.empty(diagnostic_shape, dtype=float)
    correction_frobenius = np.empty(diagnostic_shape, dtype=float)
    correction_blocks = np.empty(diagnostic_shape + (5,), dtype=float)
    constraint_counts = np.empty(diagnostic_shape, dtype=int)
    raw_minima = np.empty(diagnostic_shape + (3,), dtype=float)
    corrected_minima = np.empty_like(raw_minima)
    raw_violated = np.empty(diagnostic_shape + (3,), dtype=bool)
    binding = np.empty_like(raw_violated)
    converged = np.empty(diagnostic_shape, dtype=bool)
    raw_growth = np.empty((case_count, step_count), dtype=float)
    correction_growth = np.empty_like(raw_growth)
    total_growth = np.empty_like(raw_growth)
    correction_differences = np.empty(
        (case_count, step_count, state_size),
        dtype=float,
    )
    metric = closed_state_lifted_frobenius_metric()
    binding_tolerance = 10.0 * cone_tolerance

    def record_sample(sample_index: int, current: FloatArray) -> None:
        sampled_states[sample_index] = current
        for trajectory_index, state in enumerate(current):
            (
                sampled_margins[sample_index, trajectory_index],
                sampled_traces[sample_index, trajectory_index],
            ) = physicality_diagnostics(state)
        for case_index in range(case_count):
            delta = current[case_index + 1] - current[0]
            sampled_euclidean[case_index, sample_index] = np.linalg.norm(delta)
            sampled_frobenius[case_index, sample_index] = (
                closed_state_lifted_frobenius_norm(delta)
            )

    record_sample(0, states)
    for case_index in range(case_count):
        step_frobenius[case_index, 0] = sampled_frobenius[case_index, 0]

    def evaluate(
        time_value: float,
        state: FloatArray,
        step_index: int,
        stage_index: int,
        trajectory_index: int,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        raw = closed_scalar_rhs(time_value, state, parameters)
        result = structured_electron_phonon_barrier_correction(
            state,
            raw,
            parameters,
            activation_margin=activation_margin,
            target_flux=0.0,
            barrier_rate=barrier_rate,
            energy_neutral=True,
            preserve_correlation_trace=True,
            cone_tolerance=cone_tolerance,
            projection_tolerance=projection_tolerance,
            correction_metric=correction_metric,
        )
        if not result.converged:
            raise RuntimeError(
                "matched controller solve failed at "
                f"t={time_value}, stage={STAGE_NAMES[stage_index]}, "
                f"trajectory={trajectory_index}"
            )
        correction = result.correction_coordinates
        index = (step_index, stage_index, trajectory_index)
        correction_euclidean[index] = result.correction_norm
        correction_frobenius[index] = result.lifted_frobenius_norm
        correction_blocks[index] = _block_norms(correction)
        constraint_counts[index] = result.constraint_count
        raw_minima[index] = (
            result.raw_electron_lower_barrier_minimum_eigenvalue,
            result.raw_electron_upper_barrier_minimum_eigenvalue,
            result.raw_joint_barrier_minimum_eigenvalue,
        )
        corrected_minima[index] = (
            result.corrected_electron_lower_barrier_minimum_eigenvalue,
            result.corrected_electron_upper_barrier_minimum_eigenvalue,
            result.corrected_joint_barrier_minimum_eigenvalue,
        )
        raw_violated[index] = raw_minima[index] < -cone_tolerance
        binding[index] = corrected_minima[index] <= binding_tolerance
        converged[index] = result.converged
        return raw + correction, raw, correction

    for step_index in range(step_count):
        time_value = step_times[step_index]
        next_states = np.empty_like(states)
        k1_raw = np.empty_like(states)
        k1_correction = np.empty_like(states)
        for trajectory_index, state in enumerate(states):
            k1, raw1, correction1 = evaluate(
                time_value,
                state,
                step_index,
                0,
                trajectory_index,
            )
            k2, _, _ = evaluate(
                time_value + 0.5 * time_step,
                state + 0.5 * time_step * k1,
                step_index,
                1,
                trajectory_index,
            )
            k3, _, _ = evaluate(
                time_value + 0.5 * time_step,
                state + 0.5 * time_step * k2,
                step_index,
                2,
                trajectory_index,
            )
            k4, _, _ = evaluate(
                time_value + time_step,
                state + time_step * k3,
                step_index,
                3,
                trajectory_index,
            )
            next_states[trajectory_index] = state + (time_step / 6.0) * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            )
            k1_raw[trajectory_index] = raw1
            k1_correction[trajectory_index] = correction1

        for case_index in range(case_count):
            delta = states[case_index + 1] - states[0]
            denominator = float(delta @ metric @ delta)
            if denominator <= 0.0:
                raise RuntimeError("matched pair collapsed in the selected metric")
            raw_delta = k1_raw[case_index + 1] - k1_raw[0]
            correction_delta = (
                k1_correction[case_index + 1] - k1_correction[0]
            )
            correction_differences[case_index, step_index] = correction_delta
            raw_growth[case_index, step_index] = float(
                delta @ metric @ raw_delta / denominator
            )
            correction_growth[case_index, step_index] = float(
                delta @ metric @ correction_delta / denominator
            )
            total_growth[case_index, step_index] = (
                raw_growth[case_index, step_index]
                + correction_growth[case_index, step_index]
            )

        states = next_states
        for case_index in range(case_count):
            step_frobenius[case_index, step_index + 1] = (
                closed_state_lifted_frobenius_norm(
                    states[case_index + 1] - states[0]
                )
            )
        if (step_index + 1) % sample_stride == 0:
            record_sample((step_index + 1) // sample_stride, states)

    return MatchedControllerSensitivity(
        step_times=step_times,
        sample_times=sample_times,
        labels=labels,
        sampled_states=sampled_states,
        step_frobenius_distances=step_frobenius,
        sampled_euclidean_distances=sampled_euclidean,
        sampled_frobenius_distances=sampled_frobenius,
        sampled_margins=sampled_margins,
        sampled_trace_residuals=sampled_traces,
        correction_euclidean_norms=correction_euclidean,
        correction_frobenius_norms=correction_frobenius,
        correction_block_norms=correction_blocks,
        constraint_counts=constraint_counts,
        raw_barrier_minima=raw_minima,
        corrected_barrier_minima=corrected_minima,
        raw_violated_sectors=raw_violated,
        binding_sectors=binding,
        converged=converged,
        raw_growth_contributions=raw_growth,
        correction_growth_contributions=correction_growth,
        total_growth_contributions=total_growth,
        k1_correction_differences=correction_differences,
    )


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


def _transition_count(signatures: np.ndarray) -> int:
    flattened = np.asarray(signatures, dtype=bool).reshape(signatures.shape[0], -1)
    return int(np.sum(np.any(flattened[1:] != flattened[:-1], axis=1)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--amplitude-index", type=int, default=0)
    parser.add_argument("--final-time", type=float, default=100.0)
    parser.add_argument("--time-step", type=float, default=0.02)
    parser.add_argument("--sample-step", type=float, default=0.1)
    parser.add_argument("--activation-margin", type=float, default=1e-5)
    parser.add_argument("--barrier-rate", type=float, default=5.0)
    parser.add_argument("--cone-tolerance", type=float, default=1e-8)
    parser.add_argument("--projection-tolerance", type=float, default=1e-12)
    parser.add_argument(
        "--correction-metric",
        choices=("euclidean", "frobenius"),
        default="euclidean",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    exact = np.load(args.exact_trajectory)
    exact_times = np.asarray(exact["times"], dtype=float)
    direction_names = tuple(str(value) for value in exact["direction_names"])
    amplitudes = np.asarray(exact["perturbation_amplitudes"], dtype=float)
    if args.amplitude_index < 0 or args.amplitude_index >= amplitudes.size:
        raise SystemExit("amplitude-index is out of range")
    if not np.isclose(exact_times[-1], args.final_time, atol=1e-12):
        raise SystemExit("exact trajectory and requested final time differ")
    initial_base = np.asarray(exact["base_coordinates"][0], dtype=float)
    initial_shadows = np.asarray(
        exact["shadow_coordinates"][:, args.amplitude_index, 0],
        dtype=float,
    )
    exact_distances = np.asarray(
        exact["coordinate_frobenius_distances"][:, args.amplitude_index],
        dtype=float,
    )
    amplitude = float(amplitudes[args.amplitude_index])
    labels = tuple(f"{name}_epsilon_{amplitude:.8g}" for name in direction_names)
    parameters = DimerParameters(
        hopping=1.0,
        gamma=0.5,
        lambda_ep=1.5,
        drive_amplitude=1.0,
        pulse_width=1.0,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema": "paper_v_matched_exact_controller_sensitivity_plan_v1",
        "classification": "diagnostic",
        "evidence_status": "exploratory_local_not_promoted",
        "scientific_question": (
            "Does the controlled 31-coordinate flow amplify the same nearby "
            "physical pair that remains bounded under exact unitary evolution?"
        ),
        "exact_input": {
            "path": str(args.exact_trajectory),
            "sha256": _sha256(args.exact_trajectory),
            "direction_names": list(direction_names),
            "wavefunction_perturbation_amplitude": amplitude,
        },
        "parameters": {
            "hopping": parameters.hopping,
            "gamma": parameters.gamma,
            "lambda_ep": parameters.lambda_ep,
            "drive_amplitude": parameters.drive_amplitude,
            "pulse_width": parameters.pulse_width,
        },
        "integration": {
            "method": "fixed_step_RK4",
            "final_time": args.final_time,
            "time_step": args.time_step,
            "sample_step": args.sample_step,
            "trajectory_rescaling": "none",
        },
        "controller": {
            "activation_margin": args.activation_margin,
            "barrier_rate": args.barrier_rate,
            "cone_tolerance": args.cone_tolerance,
            "projection_tolerance": args.projection_tolerance,
            "correction_metric": args.correction_metric,
            "energy_neutral": True,
            "preserve_correlation_trace": True,
        },
        "source_sha256": {
            "analysis": _sha256(Path(__file__).resolve()),
            "cone_correction": _sha256(
                Path(__file__).resolve().parent / "cone_correction.py"
            ),
        },
    }
    _write_json_atomic(args.output_dir / "plan.json", plan)
    result = matched_controller_sensitivity(
        parameters,
        initial_base,
        initial_shadows,
        labels=labels,
        final_time=args.final_time,
        time_step=args.time_step,
        sample_step=args.sample_step,
        activation_margin=args.activation_margin,
        barrier_rate=args.barrier_rate,
        cone_tolerance=args.cone_tolerance,
        projection_tolerance=args.projection_tolerance,
        correction_metric=args.correction_metric,
    )
    np.savez_compressed(
        args.output_dir / "trajectory.npz",
        step_times=result.step_times,
        sample_times=result.sample_times,
        labels=np.asarray(result.labels),
        exact_times=exact_times,
        exact_frobenius_distances=exact_distances,
        sampled_states=result.sampled_states,
        step_frobenius_distances=result.step_frobenius_distances,
        sampled_euclidean_distances=result.sampled_euclidean_distances,
        sampled_frobenius_distances=result.sampled_frobenius_distances,
        sampled_margins=result.sampled_margins,
        sampled_trace_residuals=result.sampled_trace_residuals,
        correction_euclidean_norms=result.correction_euclidean_norms,
        correction_frobenius_norms=result.correction_frobenius_norms,
        correction_block_norms=result.correction_block_norms,
        constraint_counts=result.constraint_counts,
        raw_barrier_minima=result.raw_barrier_minima,
        corrected_barrier_minima=result.corrected_barrier_minima,
        raw_violated_sectors=result.raw_violated_sectors,
        binding_sectors=result.binding_sectors,
        converged=result.converged,
        raw_growth_contributions=result.raw_growth_contributions,
        correction_growth_contributions=(
            result.correction_growth_contributions
        ),
        total_growth_contributions=result.total_growth_contributions,
        k1_correction_differences=result.k1_correction_differences,
    )

    late_step_mask = result.step_times[:-1] >= min(40.0, 0.4 * args.final_time)
    cases: list[dict[str, object]] = []
    for case_index, name in enumerate(direction_names):
        corrected_distance = result.step_frobenius_distances[case_index]
        exact_distance = exact_distances[case_index]
        initial_distance = float(corrected_distance[0])
        cases.append(
            {
                "direction": name,
                "wavefunction_perturbation_amplitude": amplitude,
                "exact_maximum_amplification": float(
                    np.max(exact_distance) / exact_distance[0]
                ),
                "exact_final_amplification": float(
                    exact_distance[-1] / exact_distance[0]
                ),
                "corrected_maximum_amplification": float(
                    np.max(corrected_distance) / initial_distance
                ),
                "corrected_final_amplification": float(
                    corrected_distance[-1] / initial_distance
                ),
                "corrected_final_endpoint_log_rate": float(
                    np.log(corrected_distance[-1] / initial_distance)
                    / args.final_time
                ),
                "late_mean_raw_growth_contribution": float(
                    np.mean(result.raw_growth_contributions[case_index, late_step_mask])
                ),
                "late_mean_correction_growth_contribution": float(
                    np.mean(
                        result.correction_growth_contributions[
                            case_index,
                            late_step_mask,
                        ]
                    )
                ),
                "late_mean_total_growth_contribution": float(
                    np.mean(
                        result.total_growth_contributions[
                            case_index,
                            late_step_mask,
                        ]
                    )
                ),
                "maximum_k1_correction_difference_norm": float(
                    np.max(
                        np.linalg.norm(
                            result.k1_correction_differences[case_index],
                            axis=1,
                        )
                    )
                ),
                "base_binding_transition_count": _transition_count(
                    result.binding_sectors[:, :, 0]
                ),
                "shadow_binding_transition_count": _transition_count(
                    result.binding_sectors[:, :, case_index + 1]
                ),
                "base_joint_binding_fraction": float(
                    np.mean(result.binding_sectors[:, :, 0, 2])
                ),
                "shadow_joint_binding_fraction": float(
                    np.mean(
                        result.binding_sectors[:, :, case_index + 1, 2]
                    )
                ),
            }
        )
    summary = {
        "schema": "paper_v_matched_exact_controller_sensitivity_summary_v1",
        "status": "complete",
        "classification": "diagnostic",
        "evidence_status": "exploratory_local_not_promoted",
        "all_controller_evaluations_converged": bool(np.all(result.converged)),
        "minimum_sampled_physicality_margin": float(
            np.min(result.sampled_margins)
        ),
        "maximum_sampled_correlation_trace_residual": float(
            np.max(result.sampled_trace_residuals)
        ),
        "cases": cases,
        "interpretation_contract": (
            "Matched initial moments and no rescaling isolate differences "
            "between exact and controlled vector fields. Instantaneous growth "
            "is decomposed in the lifted Frobenius metric into raw-closure and "
            "controller-correction contributions."
        ),
    }
    _write_json_atomic(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
