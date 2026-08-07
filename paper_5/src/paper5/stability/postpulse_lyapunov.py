"""Long-horizon Lyapunov screen for the autonomous corrected moment EOM.

The driven corrected trajectory is first advanced to a declared pulse end.
The drive is then set exactly to zero, making the subsequent vector field
autonomous.  Two exact-wavefunction-induced perturbations are propagated with
the Benettin rescaling algorithm and projected onto the local energy- and
correlation-trace tangent space after every rescaling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from .cone_correction import (
    closed_state_correction_energy_gradient,
    closed_state_lifted_frobenius_metric,
    closed_state_lifted_frobenius_norm,
    structured_electron_phonon_barrier_correction,
)
from .hubbard_dimer import DimerParameters, FloatArray
from .initial_condition_sensitivity import physicality_diagnostics
from .matched_exact_controller_sensitivity import matched_controller_sensitivity
from .matrix_reference import (
    CLOSED_SCALAR_STATE_NAMES,
    closed_scalar_rhs,
    closed_scalar_to_matrix_state,
    matrix_total_energy,
)

ProgressFunction = Callable[[dict[str, object]], None]
SECTOR_NAMES = ("electron_lower", "electron_upper", "joint_gram")


@dataclass(frozen=True)
class PostPulseLyapunovEstimate:
    """Two-direction post-pulse Benettin estimate and physicality record."""

    times: FloatArray
    labels: tuple[str, ...]
    local_exponents: FloatArray
    cumulative_exponents: FloatArray
    start_distances: FloatArray
    end_distances: FloatArray
    projection_retention: FloatArray
    base_margins: FloatArray
    shadow_margins: FloatArray
    base_trace_residuals: FloatArray
    shadow_trace_residuals: FloatArray
    base_energies: FloatArray
    shadow_energies: FloatArray
    base_correction_norms: FloatArray
    shadow_correction_norms: FloatArray
    base_binding_sectors: np.ndarray
    shadow_binding_sectors: np.ndarray
    maximum_absolute_coordinates: FloatArray
    final_base: FloatArray
    final_shadows: FloatArray

    def window_exponent(self, start: float, stop: float) -> FloatArray:
        """Return time-weighted exponents over a global-time window."""

        segment_starts = self.times[:-1]
        segment_stops = self.times[1:]
        mask = (segment_starts >= start - 1e-12) & (
            segment_stops <= stop + 1e-12
        )
        if not np.any(mask):
            raise ValueError("window contains no complete segments")
        durations = segment_stops[mask] - segment_starts[mask]
        return np.sum(self.local_exponents[:, mask] * durations, axis=1) / np.sum(
            durations
        )


def _tangent_projection(
    direction: FloatArray,
    base_state: FloatArray,
    parameters: DimerParameters,
) -> tuple[FloatArray, float]:
    """Project into the equality tangent space in the lifted matrix metric."""

    vector = np.asarray(direction, dtype=float)
    expected = (len(CLOSED_SCALAR_STATE_NAMES),)
    if vector.shape != expected:
        raise ValueError(f"direction must have shape {expected}")
    rows = np.zeros((3, vector.size), dtype=float)
    rows[0] = closed_state_correction_energy_gradient(base_state, parameters)
    rows[1, 17] = 1.0
    rows[2, 18] = 1.0
    metric = closed_state_lifted_frobenius_metric()
    metric_dual = np.linalg.solve(metric, rows.T)
    constraint_gram = rows @ metric_dual
    if np.linalg.matrix_rank(constraint_gram, tol=1e-12) != rows.shape[0]:
        raise RuntimeError("post-pulse tangent equalities are rank deficient")
    multipliers = np.linalg.solve(constraint_gram, rows @ vector)
    projected = vector - metric_dual @ multipliers
    original_size = closed_state_lifted_frobenius_norm(vector)
    size = closed_state_lifted_frobenius_norm(projected)
    if not np.isfinite(size) or size <= 1e-15:
        raise RuntimeError("tangent projection removed the perturbation")
    if np.max(np.abs(rows @ projected)) > 2e-11:
        raise RuntimeError("projected direction violates tangent equalities")
    retention = size / original_size if original_size > 0.0 else 0.0
    return np.asarray(projected, dtype=float), float(retention)


def tangent_projected_direction(
    direction: FloatArray,
    base_state: FloatArray,
    parameters: DimerParameters,
) -> FloatArray:
    """Return a unit direction tangent to energy and correlation trace."""

    projected, _ = _tangent_projection(direction, base_state, parameters)
    return projected / closed_state_lifted_frobenius_norm(projected)


def _controller_velocity(
    time_value: float,
    state: FloatArray,
    parameters: DimerParameters,
    *,
    activation_margin: float,
    barrier_rate: float,
    cone_tolerance: float,
    projection_tolerance: float,
    correction_metric: str,
) -> tuple[FloatArray, object]:
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
            "post-pulse controller solve failed: "
            f"lower={result.corrected_electron_lower_barrier_minimum_eigenvalue}, "
            f"upper={result.corrected_electron_upper_barrier_minimum_eigenvalue}, "
            f"joint={result.corrected_joint_barrier_minimum_eigenvalue}"
        )
    return raw + result.correction_coordinates, result


def _rk4_bundle_step(
    states: FloatArray,
    time_value: float,
    time_step: float,
    parameters: DimerParameters,
    *,
    activation_margin: float,
    barrier_rate: float,
    cone_tolerance: float,
    projection_tolerance: float,
    correction_metric: str,
) -> FloatArray:
    next_states = np.empty_like(states)
    for index, state in enumerate(states):
        k1, _ = _controller_velocity(
            time_value,
            state,
            parameters,
            activation_margin=activation_margin,
            barrier_rate=barrier_rate,
            cone_tolerance=cone_tolerance,
            projection_tolerance=projection_tolerance,
            correction_metric=correction_metric,
        )
        k2, _ = _controller_velocity(
            time_value + 0.5 * time_step,
            state + 0.5 * time_step * k1,
            parameters,
            activation_margin=activation_margin,
            barrier_rate=barrier_rate,
            cone_tolerance=cone_tolerance,
            projection_tolerance=projection_tolerance,
            correction_metric=correction_metric,
        )
        k3, _ = _controller_velocity(
            time_value + 0.5 * time_step,
            state + 0.5 * time_step * k2,
            parameters,
            activation_margin=activation_margin,
            barrier_rate=barrier_rate,
            cone_tolerance=cone_tolerance,
            projection_tolerance=projection_tolerance,
            correction_metric=correction_metric,
        )
        k4, _ = _controller_velocity(
            time_value + time_step,
            state + time_step * k3,
            parameters,
            activation_margin=activation_margin,
            barrier_rate=barrier_rate,
            cone_tolerance=cone_tolerance,
            projection_tolerance=projection_tolerance,
            correction_metric=correction_metric,
        )
        next_states[index] = state + (time_step / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )
    return next_states


def postpulse_lyapunov_estimate(
    parameters: DimerParameters,
    initial_base: FloatArray,
    initial_directions: FloatArray,
    *,
    labels: tuple[str, ...],
    initial_time: float = 4.0,
    final_time: float = 1000.0,
    time_step: float = 0.02,
    renormalization_interval: float = 0.5,
    perturbation_size: float = 1e-5,
    activation_margin: float = 1e-5,
    barrier_rate: float = 5.0,
    cone_tolerance: float = 1e-8,
    projection_tolerance: float = 1e-12,
    correction_metric: str = "euclidean",
    progress: ProgressFunction | None = None,
) -> PostPulseLyapunovEstimate:
    """Estimate the largest exponent from two tangent-space directions."""

    base = np.asarray(initial_base, dtype=float).copy()
    directions = np.asarray(initial_directions, dtype=float)
    expected = (len(CLOSED_SCALAR_STATE_NAMES),)
    if base.shape != expected:
        raise ValueError(f"initial_base must have shape {expected}")
    if directions.ndim != 2 or directions.shape[1:] != expected:
        raise ValueError("initial_directions must have shape (directions, 31)")
    if directions.shape[0] != len(labels):
        raise ValueError("labels must match initial_directions")
    if final_time <= initial_time:
        raise ValueError("final_time must exceed initial_time")
    if time_step <= 0.0 or renormalization_interval <= 0.0:
        raise ValueError("integration intervals must be positive")
    if perturbation_size <= 0.0:
        raise ValueError("perturbation_size must be positive")
    steps_per_segment = int(round(renormalization_interval / time_step))
    segment_count = int(round((final_time - initial_time) / renormalization_interval))
    if steps_per_segment <= 0 or not np.isclose(
        steps_per_segment * time_step,
        renormalization_interval,
        atol=1e-12,
    ):
        raise ValueError("time_step must divide renormalization_interval")
    if not np.isclose(
        initial_time + segment_count * renormalization_interval,
        final_time,
        atol=1e-12,
    ):
        raise ValueError("renormalization_interval must divide the horizon")

    normalized_directions = np.asarray(
        [
            tangent_projected_direction(direction, base, parameters)
            for direction in directions
        ],
        dtype=float,
    )
    shadows = base[None, :] + perturbation_size * normalized_directions
    direction_count = directions.shape[0]
    times = np.linspace(initial_time, final_time, segment_count + 1)
    local = np.empty((direction_count, segment_count), dtype=float)
    cumulative = np.empty_like(local)
    starts = np.empty_like(local)
    ends = np.empty_like(local)
    retention = np.empty_like(local)
    base_margins = np.empty((segment_count + 1, 4), dtype=float)
    shadow_margins = np.empty((direction_count, segment_count + 1, 4), dtype=float)
    base_traces = np.empty(segment_count + 1, dtype=float)
    shadow_traces = np.empty((direction_count, segment_count + 1), dtype=float)
    base_energies = np.empty(segment_count + 1, dtype=float)
    shadow_energies = np.empty((direction_count, segment_count + 1), dtype=float)
    base_corrections = np.empty(segment_count + 1, dtype=float)
    shadow_corrections = np.empty((direction_count, segment_count + 1), dtype=float)
    base_binding = np.empty((segment_count + 1, 3), dtype=bool)
    shadow_binding = np.empty((direction_count, segment_count + 1, 3), dtype=bool)
    maximum_coordinates = np.empty((direction_count + 1, segment_count + 1), dtype=float)
    accumulated_logs = np.zeros(direction_count, dtype=float)
    binding_tolerance = 10.0 * cone_tolerance

    def record(index: int) -> None:
        base_margins[index], base_traces[index] = physicality_diagnostics(base)
        base_energies[index] = matrix_total_energy(
            closed_scalar_to_matrix_state(base),
            parameters,
        )
        _, base_result = _controller_velocity(
            times[index],
            base,
            parameters,
            activation_margin=activation_margin,
            barrier_rate=barrier_rate,
            cone_tolerance=cone_tolerance,
            projection_tolerance=projection_tolerance,
            correction_metric=correction_metric,
        )
        base_corrections[index] = base_result.lifted_frobenius_norm
        base_binding[index] = np.asarray(
            [
                base_result.corrected_electron_lower_barrier_minimum_eigenvalue,
                base_result.corrected_electron_upper_barrier_minimum_eigenvalue,
                base_result.corrected_joint_barrier_minimum_eigenvalue,
            ]
        ) <= binding_tolerance
        maximum_coordinates[0, index] = np.max(np.abs(base))
        for direction_index, shadow in enumerate(shadows):
            shadow_margins[direction_index, index], shadow_traces[
                direction_index, index
            ] = physicality_diagnostics(shadow)
            shadow_energies[direction_index, index] = matrix_total_energy(
                closed_scalar_to_matrix_state(shadow),
                parameters,
            )
            _, shadow_result = _controller_velocity(
                times[index],
                shadow,
                parameters,
                activation_margin=activation_margin,
                barrier_rate=barrier_rate,
                cone_tolerance=cone_tolerance,
                projection_tolerance=projection_tolerance,
                correction_metric=correction_metric,
            )
            shadow_corrections[direction_index, index] = (
                shadow_result.lifted_frobenius_norm
            )
            shadow_binding[direction_index, index] = np.asarray(
                [
                    shadow_result.corrected_electron_lower_barrier_minimum_eigenvalue,
                    shadow_result.corrected_electron_upper_barrier_minimum_eigenvalue,
                    shadow_result.corrected_joint_barrier_minimum_eigenvalue,
                ]
            ) <= binding_tolerance
            maximum_coordinates[direction_index + 1, index] = np.max(
                np.abs(shadow)
            )

    record(0)
    time_value = initial_time
    for segment in range(segment_count):
        for direction_index in range(direction_count):
            starts[direction_index, segment] = closed_state_lifted_frobenius_norm(
                shadows[direction_index] - base
            )
        states = np.vstack((base, shadows))
        for _ in range(steps_per_segment):
            states = _rk4_bundle_step(
                states,
                time_value,
                time_step,
                parameters,
                activation_margin=activation_margin,
                barrier_rate=barrier_rate,
                cone_tolerance=cone_tolerance,
                projection_tolerance=projection_tolerance,
                correction_metric=correction_metric,
            )
            time_value += time_step
        base = states[0]
        evolved_shadows = states[1:]
        for direction_index, evolved_shadow in enumerate(evolved_shadows):
            delta = evolved_shadow - base
            end_distance = closed_state_lifted_frobenius_norm(delta)
            ends[direction_index, segment] = end_distance
            log_growth = np.log(
                end_distance / starts[direction_index, segment]
            )
            local[direction_index, segment] = (
                log_growth / renormalization_interval
            )
            accumulated_logs[direction_index] += log_growth
            cumulative[direction_index, segment] = (
                accumulated_logs[direction_index]
                / (time_value - initial_time)
            )
            projected, retained_fraction = _tangent_projection(
                delta,
                base,
                parameters,
            )
            retention[direction_index, segment] = retained_fraction
            projected /= closed_state_lifted_frobenius_norm(projected)
            shadows[direction_index] = base + perturbation_size * projected
        record(segment + 1)
        if progress is not None and (
            segment == 0 or (segment + 1) % 20 == 0 or segment + 1 == segment_count
        ):
            progress(
                {
                    "segment": segment + 1,
                    "segments": segment_count,
                    "time": float(time_value),
                    "cumulative_exponents": cumulative[:, segment].tolist(),
                    "minimum_margin": float(
                        min(
                            np.min(base_margins[segment + 1]),
                            np.min(shadow_margins[:, segment + 1]),
                        )
                    ),
                    "maximum_absolute_coordinate": float(
                        np.max(maximum_coordinates[:, segment + 1])
                    ),
                }
            )

    return PostPulseLyapunovEstimate(
        times=times,
        labels=labels,
        local_exponents=local,
        cumulative_exponents=cumulative,
        start_distances=starts,
        end_distances=ends,
        projection_retention=retention,
        base_margins=base_margins,
        shadow_margins=shadow_margins,
        base_trace_residuals=base_traces,
        shadow_trace_residuals=shadow_traces,
        base_energies=base_energies,
        shadow_energies=shadow_energies,
        base_correction_norms=base_corrections,
        shadow_correction_norms=shadow_corrections,
        base_binding_sectors=base_binding,
        shadow_binding_sectors=shadow_binding,
        maximum_absolute_coordinates=maximum_coordinates,
        final_base=base,
        final_shadows=shadows,
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--amplitude-index", type=int, default=0)
    parser.add_argument("--pulse-end", type=float, default=4.0)
    parser.add_argument("--final-time", type=float, default=1000.0)
    parser.add_argument("--time-step", type=float, default=0.02)
    parser.add_argument("--renormalization-interval", type=float, default=0.5)
    parser.add_argument("--perturbation-size", type=float, default=1e-5)
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
    amplitudes = np.asarray(exact["perturbation_amplitudes"], dtype=float)
    if args.amplitude_index < 0 or args.amplitude_index >= amplitudes.size:
        raise SystemExit("amplitude-index is out of range")
    direction_names = tuple(str(value) for value in exact["direction_names"])
    initial_base = np.asarray(exact["base_coordinates"][0], dtype=float)
    initial_shadows = np.asarray(
        exact["shadow_coordinates"][:, args.amplitude_index, 0],
        dtype=float,
    )
    driven_parameters = DimerParameters(
        hopping=1.0,
        gamma=0.5,
        lambda_ep=1.5,
        drive_amplitude=1.0,
        pulse_width=1.0,
    )
    pre_pulse = matched_controller_sensitivity(
        driven_parameters,
        initial_base,
        initial_shadows,
        labels=direction_names,
        final_time=args.pulse_end,
        time_step=args.time_step,
        sample_step=args.renormalization_interval,
        activation_margin=args.activation_margin,
        barrier_rate=args.barrier_rate,
        cone_tolerance=args.cone_tolerance,
        projection_tolerance=args.projection_tolerance,
        correction_metric=args.correction_metric,
    )
    postpulse_base = pre_pulse.sampled_states[-1, 0]
    postpulse_directions = (
        pre_pulse.sampled_states[-1, 1:] - postpulse_base[None, :]
    )
    undriven_parameters = DimerParameters(
        hopping=driven_parameters.hopping,
        gamma=driven_parameters.gamma,
        lambda_ep=driven_parameters.lambda_ep,
        drive_amplitude=0.0,
        pulse_width=driven_parameters.pulse_width,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema": "paper_v_postpulse_lyapunov_plan_v1",
        "classification": "diagnostic",
        "evidence_status": "exploratory_local_not_promoted",
        "scientific_question": (
            "Does the autonomous post-pulse corrected moment EOM retain a "
            "positive largest Lyapunov exponent through t=1000?"
        ),
        "exact_input": {
            "path": str(args.exact_trajectory),
            "sha256": _sha256(args.exact_trajectory),
            "directions": list(direction_names),
            "wavefunction_perturbation_amplitude": float(
                amplitudes[args.amplitude_index]
            ),
        },
        "protocol": {
            "driven_until": args.pulse_end,
            "drive_exactly_zero_after": args.pulse_end,
            "final_time": args.final_time,
            "time_step": args.time_step,
            "renormalization_interval": args.renormalization_interval,
            "perturbation_size": args.perturbation_size,
            "tangent_constraints": [
                "zero first-order post-pulse energy change",
                "zero real correlation-trace change",
                "zero imaginary correlation-trace change",
            ],
        },
        "controller": {
            "activation_margin": args.activation_margin,
            "barrier_rate": args.barrier_rate,
            "cone_tolerance": args.cone_tolerance,
            "projection_tolerance": args.projection_tolerance,
            "correction_metric": args.correction_metric,
        },
        "source_sha256": {
            "analysis": _sha256(Path(__file__).resolve()),
            "cone_correction": _sha256(
                Path(__file__).resolve().parent / "cone_correction.py"
            ),
        },
    }
    _write_json_atomic(args.output_dir / "plan.json", plan)
    progress_path = args.output_dir / "progress.jsonl"

    def progress(payload: dict[str, object]) -> None:
        record = {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
        print(json.dumps(record, sort_keys=True), flush=True)

    estimate = postpulse_lyapunov_estimate(
        undriven_parameters,
        postpulse_base,
        postpulse_directions,
        labels=direction_names,
        initial_time=args.pulse_end,
        final_time=args.final_time,
        time_step=args.time_step,
        renormalization_interval=args.renormalization_interval,
        perturbation_size=args.perturbation_size,
        activation_margin=args.activation_margin,
        barrier_rate=args.barrier_rate,
        cone_tolerance=args.cone_tolerance,
        projection_tolerance=args.projection_tolerance,
        correction_metric=args.correction_metric,
        progress=progress,
    )
    np.savez_compressed(
        args.output_dir / "trajectory.npz",
        times=estimate.times,
        labels=np.asarray(estimate.labels),
        local_exponents=estimate.local_exponents,
        cumulative_exponents=estimate.cumulative_exponents,
        start_distances=estimate.start_distances,
        end_distances=estimate.end_distances,
        projection_retention=estimate.projection_retention,
        base_margins=estimate.base_margins,
        shadow_margins=estimate.shadow_margins,
        base_trace_residuals=estimate.base_trace_residuals,
        shadow_trace_residuals=estimate.shadow_trace_residuals,
        base_energies=estimate.base_energies,
        shadow_energies=estimate.shadow_energies,
        base_correction_norms=estimate.base_correction_norms,
        shadow_correction_norms=estimate.shadow_correction_norms,
        base_binding_sectors=estimate.base_binding_sectors,
        shadow_binding_sectors=estimate.shadow_binding_sectors,
        maximum_absolute_coordinates=estimate.maximum_absolute_coordinates,
        final_base=estimate.final_base,
        final_shadows=estimate.final_shadows,
    )
    window_boundaries = [
        args.pulse_end,
        *(
            boundary
            for boundary in (100.0, 250.0, 500.0, 750.0)
            if args.pulse_end < boundary < args.final_time
        ),
        args.final_time,
    ]
    windows = tuple(zip(window_boundaries[:-1], window_boundaries[1:]))
    window_exponents = {
        f"{start:g}_to_{stop:g}": estimate.window_exponent(start, stop).tolist()
        for start, stop in windows
    }
    all_margins = np.concatenate(
        [estimate.base_margins[None, :, :], estimate.shadow_margins],
        axis=0,
    )
    all_traces = np.concatenate(
        [estimate.base_trace_residuals[None, :], estimate.shadow_trace_residuals],
        axis=0,
    )
    summary = {
        "schema": "paper_v_postpulse_lyapunov_summary_v1",
        "status": "complete",
        "classification": "diagnostic",
        "evidence_status": "exploratory_local_not_promoted",
        "directions": list(estimate.labels),
        "final_cumulative_exponents": estimate.cumulative_exponents[:, -1].tolist(),
        "window_exponents": window_exponents,
        "minimum_physicality_margin": float(np.min(all_margins)),
        "maximum_correlation_trace_residual": float(np.max(all_traces)),
        "maximum_absolute_coordinate": float(
            np.max(estimate.maximum_absolute_coordinates)
        ),
        "base_energy_drift": float(
            np.max(np.abs(estimate.base_energies - estimate.base_energies[0]))
        ),
        "maximum_shadow_energy_drift": float(
            np.max(
                np.abs(
                    estimate.shadow_energies
                    - estimate.shadow_energies[:, :1]
                )
            )
        ),
        "base_joint_binding_fraction": float(
            np.mean(estimate.base_binding_sectors[:, 2])
        ),
        "shadow_joint_binding_fractions": np.mean(
            estimate.shadow_binding_sectors[:, :, 2],
            axis=1,
        ).tolist(),
        "interpretation_contract": (
            "A persistent positive direction-independent late-window exponent "
            "on a bounded physical autonomous trajectory is evidence for "
            "chaotic sensitivity of the corrected reduced EOM, not exact "
            "quantum chaos."
        ),
    }
    _write_json_atomic(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
