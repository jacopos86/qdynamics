"""Nearby-trajectory sensitivity diagnostics for the 31-coordinate dimer.

This module estimates finite-time growth with the two-trajectory Benettin
algorithm.  It deliberately distinguishes the raw archive closure from the
representability-controlled vector field: a positive exponent after the raw
trajectory leaves the physical cone is not evidence for physical chaos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from .cone_correction import (
    closed_electron_phonon_cone_projected_rhs,
    closed_state_correction_energy_gradient,
    closed_state_lifted_frobenius_norm,
)
from .hubbard_dimer import DimerParameters, FloatArray, RhsFunction
from .initial_conditions import exact_ground_closed_scalar_coordinates
from .matrix_reference import (
    CLOSED_SCALAR_STATE_NAMES,
    boson_moment_matrix,
    closed_scalar_rhs,
    closed_scalar_to_matrix_state,
    electron_phonon_moment_matrix,
)

Protocol = Literal["archive", "joint_barrier"]
Metric = Literal["euclidean", "frobenius"]
ProgressFunction = Callable[[dict[str, float]], None]

MARGIN_NAMES = (
    "electron_lower",
    "electron_upper",
    "boson_moment",
    "joint_gram",
)


@dataclass(frozen=True)
class BenettinEstimate:
    """Finite-time nearby-trajectory growth and physicality diagnostics."""

    times: FloatArray
    local_euclidean_exponents: FloatArray
    local_frobenius_exponents: FloatArray
    cumulative_euclidean_exponents: FloatArray
    cumulative_frobenius_exponents: FloatArray
    start_euclidean_distances: FloatArray
    end_euclidean_distances: FloatArray
    start_frobenius_distances: FloatArray
    end_frobenius_distances: FloatArray
    base_margins: FloatArray
    shadow_margins: FloatArray
    base_trace_residuals: FloatArray
    shadow_trace_residuals: FloatArray
    selected_metric: Metric

    def cumulative_exponents(self, metric: Metric) -> FloatArray:
        """Return the cumulative estimate in the requested norm."""

        if metric == "euclidean":
            return self.cumulative_euclidean_exponents
        if metric == "frobenius":
            return self.cumulative_frobenius_exponents
        raise ValueError(f"unknown metric {metric!r}")

    def post_time_exponent(self, start_time: float, metric: Metric) -> float:
        """Average segment growth whose intervals start at ``start_time``."""

        segment_starts = self.times[:-1]
        mask = segment_starts >= start_time - 1e-12
        if not np.any(mask):
            raise ValueError("start_time leaves no complete segments")
        if metric == "euclidean":
            local = self.local_euclidean_exponents
        elif metric == "frobenius":
            local = self.local_frobenius_exponents
        else:
            raise ValueError(f"unknown metric {metric!r}")
        durations = np.diff(self.times)[mask]
        return float(np.sum(local[mask] * durations) / np.sum(durations))


def _norm(direction: FloatArray, metric: Metric) -> float:
    if metric == "euclidean":
        return float(np.linalg.norm(direction))
    if metric == "frobenius":
        return closed_state_lifted_frobenius_norm(direction)
    raise ValueError(f"unknown metric {metric!r}")


def constrained_initial_direction(
    initial_state: FloatArray,
    parameters: DimerParameters,
    *,
    seed: int,
    metric: Metric = "frobenius",
) -> FloatArray:
    """Return a deterministic random direction tangent to three equalities.

    The direction has zero first-order energy change and zero changes in the
    real and imaginary shared correlation traces.  The 31-coordinate lift
    already preserves Hermiticity, unit electronic trace, and covariance
    symmetry.
    """

    state = np.asarray(initial_state, dtype=float)
    expected = (len(CLOSED_SCALAR_STATE_NAMES),)
    if state.shape != expected:
        raise ValueError(f"expected initial state shape {expected}, got {state.shape}")

    rows = np.zeros((3, state.size), dtype=float)
    rows[0] = closed_state_correction_energy_gradient(state, parameters)
    rows[1, 17] = 1.0
    rows[2, 18] = 1.0
    orthonormal_rows, triangular = np.linalg.qr(rows.T, mode="reduced")
    rank = int(
        np.sum(
            np.abs(np.diag(triangular))
            > 1e-12 * max(1.0, float(np.linalg.norm(rows)))
        )
    )
    if rank != rows.shape[0]:
        raise RuntimeError("energy and correlation-trace constraints are rank deficient")

    direction = np.random.default_rng(seed).normal(size=state.size)
    basis = orthonormal_rows[:, :rank]
    direction = direction - basis @ (basis.T @ direction)
    size = _norm(direction, metric)
    if not np.isfinite(size) or size <= 0.0:
        raise RuntimeError("failed to construct a nonzero perturbation direction")
    direction = direction / size
    if np.max(np.abs(rows @ direction)) > 2e-12:
        raise RuntimeError("constructed direction is not tangent to the equalities")
    return np.asarray(direction, dtype=float)


def physicality_diagnostics(state: FloatArray) -> tuple[FloatArray, float]:
    """Return cone margins and the largest correlation-trace residual."""

    matrix_state = closed_scalar_to_matrix_state(state)
    electron = 0.5 * (
        matrix_state.electron_density
        + matrix_state.electron_density.conjugate().T
    )
    electron_eigenvalues = np.linalg.eigvalsh(electron)
    boson = boson_moment_matrix(matrix_state)
    joint = electron_phonon_moment_matrix(matrix_state)
    margins = np.asarray(
        [
            electron_eigenvalues[0],
            1.0 - electron_eigenvalues[-1],
            np.linalg.eigvalsh(boson)[0],
            np.linalg.eigvalsh(joint)[0],
        ],
        dtype=float,
    )
    trace_residual = float(
        max(
            abs(np.trace(matrix_state.electron_phonon_correlation[q]))
            for q in range(2)
        )
    )
    return margins, trace_residual


def _rk4_step(
    rhs: RhsFunction,
    time_value: float,
    state: FloatArray,
    time_step: float,
) -> FloatArray:
    k1 = rhs(time_value, state)
    k2 = rhs(
        time_value + 0.5 * time_step,
        state + 0.5 * time_step * k1,
    )
    k3 = rhs(
        time_value + 0.5 * time_step,
        state + 0.5 * time_step * k2,
    )
    k4 = rhs(time_value + time_step, state + time_step * k3)
    return np.asarray(
        state + (time_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
        dtype=float,
    )


def benettin_nearby_trajectory(
    rhs: RhsFunction,
    initial_state: FloatArray,
    initial_direction: FloatArray,
    *,
    final_time: float,
    time_step: float,
    renormalization_interval: float,
    perturbation_size: float,
    selected_metric: Metric = "frobenius",
    record_physicality: bool = True,
    progress: ProgressFunction | None = None,
) -> BenettinEstimate:
    """Estimate finite-time growth by repeated nearby-trajectory rescaling."""

    if final_time <= 0.0 or time_step <= 0.0:
        raise ValueError("final_time and time_step must be positive")
    if renormalization_interval <= 0.0 or perturbation_size <= 0.0:
        raise ValueError(
            "renormalization_interval and perturbation_size must be positive"
        )
    steps_per_segment = int(round(renormalization_interval / time_step))
    segment_count = int(round(final_time / renormalization_interval))
    if steps_per_segment <= 0:
        raise ValueError("renormalization_interval must be at least one time step")
    if not np.isclose(
        steps_per_segment * time_step,
        renormalization_interval,
        rtol=0.0,
        atol=1e-13,
    ):
        raise ValueError("time_step must divide renormalization_interval")
    if not np.isclose(
        segment_count * renormalization_interval,
        final_time,
        rtol=0.0,
        atol=1e-13,
    ):
        raise ValueError("renormalization_interval must divide final_time")

    base = np.asarray(initial_state, dtype=float).copy()
    direction = np.asarray(initial_direction, dtype=float).copy()
    if base.shape != direction.shape or base.ndim != 1:
        raise ValueError("initial_state and initial_direction must be matching vectors")
    direction_norm = _norm(direction, selected_metric)
    if direction_norm <= 0.0:
        raise ValueError("initial_direction must be nonzero")
    direction = direction / direction_norm
    shadow = base + perturbation_size * direction

    times = np.linspace(0.0, final_time, segment_count + 1)
    start_euclidean = np.empty(segment_count, dtype=float)
    end_euclidean = np.empty(segment_count, dtype=float)
    start_frobenius = np.empty(segment_count, dtype=float)
    end_frobenius = np.empty(segment_count, dtype=float)
    local_euclidean = np.empty(segment_count, dtype=float)
    local_frobenius = np.empty(segment_count, dtype=float)
    cumulative_euclidean = np.empty(segment_count, dtype=float)
    cumulative_frobenius = np.empty(segment_count, dtype=float)
    base_margins = np.full((segment_count + 1, len(MARGIN_NAMES)), np.nan)
    shadow_margins = np.full_like(base_margins, np.nan)
    base_traces = np.full(segment_count + 1, np.nan)
    shadow_traces = np.full(segment_count + 1, np.nan)
    if record_physicality:
        base_margins[0], base_traces[0] = physicality_diagnostics(base)
        shadow_margins[0], shadow_traces[0] = physicality_diagnostics(shadow)

    accumulated_euclidean = 0.0
    accumulated_frobenius = 0.0
    time_value = 0.0
    for segment in range(segment_count):
        initial_delta = shadow - base
        start_euclidean[segment] = float(np.linalg.norm(initial_delta))
        start_frobenius[segment] = closed_state_lifted_frobenius_norm(
            initial_delta
        )
        for _ in range(steps_per_segment):
            base = _rk4_step(rhs, time_value, base, time_step)
            shadow = _rk4_step(rhs, time_value, shadow, time_step)
            time_value += time_step
        if not np.all(np.isfinite(base)) or not np.all(np.isfinite(shadow)):
            raise RuntimeError(f"non-finite nearby trajectory at t={time_value}")

        final_delta = shadow - base
        end_euclidean[segment] = float(np.linalg.norm(final_delta))
        end_frobenius[segment] = closed_state_lifted_frobenius_norm(final_delta)
        if end_euclidean[segment] <= 0.0 or end_frobenius[segment] <= 0.0:
            raise RuntimeError(f"nearby trajectories collapsed at t={time_value}")
        euclidean_log = float(
            np.log(end_euclidean[segment] / start_euclidean[segment])
        )
        frobenius_log = float(
            np.log(end_frobenius[segment] / start_frobenius[segment])
        )
        local_euclidean[segment] = euclidean_log / renormalization_interval
        local_frobenius[segment] = frobenius_log / renormalization_interval
        accumulated_euclidean += euclidean_log
        accumulated_frobenius += frobenius_log
        cumulative_euclidean[segment] = accumulated_euclidean / time_value
        cumulative_frobenius[segment] = accumulated_frobenius / time_value

        selected_distance = (
            end_euclidean[segment]
            if selected_metric == "euclidean"
            else end_frobenius[segment]
        )
        shadow = base + (perturbation_size / selected_distance) * final_delta
        if record_physicality:
            base_margins[segment + 1], base_traces[segment + 1] = (
                physicality_diagnostics(base)
            )
            shadow_margins[segment + 1], shadow_traces[segment + 1] = (
                physicality_diagnostics(shadow)
            )
        if progress is not None:
            progress(
                {
                    "segment": float(segment + 1),
                    "time": float(time_value),
                    "cumulative_euclidean_exponent": float(
                        cumulative_euclidean[segment]
                    ),
                    "cumulative_frobenius_exponent": float(
                        cumulative_frobenius[segment]
                    ),
                    "base_joint_margin": float(base_margins[segment + 1, 3]),
                    "shadow_joint_margin": float(
                        shadow_margins[segment + 1, 3]
                    ),
                }
            )

    return BenettinEstimate(
        times=times,
        local_euclidean_exponents=local_euclidean,
        local_frobenius_exponents=local_frobenius,
        cumulative_euclidean_exponents=cumulative_euclidean,
        cumulative_frobenius_exponents=cumulative_frobenius,
        start_euclidean_distances=start_euclidean,
        end_euclidean_distances=end_euclidean,
        start_frobenius_distances=start_frobenius,
        end_frobenius_distances=end_frobenius,
        base_margins=base_margins,
        shadow_margins=shadow_margins,
        base_trace_residuals=base_traces,
        shadow_trace_residuals=shadow_traces,
        selected_metric=selected_metric,
    )


def _rhs_for_protocol(
    protocol: Protocol,
    parameters: DimerParameters,
    initial_state: FloatArray,
    *,
    activation_margin: float,
    barrier_rate: float,
    cone_tolerance: float,
    projection_tolerance: float,
    correction_metric: Metric,
) -> RhsFunction:
    if protocol == "archive":
        return lambda time, state: closed_scalar_rhs(time, state, parameters)
    if protocol == "joint_barrier":
        return closed_electron_phonon_cone_projected_rhs(
            parameters,
            initial_state,
            activation_margin=activation_margin,
            target_flux=0.0,
            barrier_rate=barrier_rate,
            energy_neutral=True,
            preserve_correlation_trace=True,
            subtract_initial_residual=False,
            require_convergence=True,
            cone_tolerance=cone_tolerance,
            projection_tolerance=projection_tolerance,
            correction_metric=correction_metric,
        )
    raise ValueError(f"unknown protocol {protocol!r}")


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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol", choices=("archive", "joint_barrier"), required=True
    )
    parser.add_argument("--final-time", type=float, default=20.0)
    parser.add_argument("--time-step", type=float, default=0.01)
    parser.add_argument("--renormalization-interval", type=float, default=0.5)
    parser.add_argument("--perturbation-size", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--phonon-cutoff", type=int, default=16)
    parser.add_argument("--activation-margin", type=float, default=1e-5)
    parser.add_argument("--barrier-rate", type=float, default=5.0)
    parser.add_argument("--cone-tolerance", type=float, default=1e-8)
    parser.add_argument("--projection-tolerance", type=float, default=1e-12)
    parser.add_argument(
        "--perturbation-metric",
        choices=("euclidean", "frobenius"),
        default="frobenius",
    )
    parser.add_argument(
        "--correction-metric",
        choices=("euclidean", "frobenius"),
        default="euclidean",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parameters = DimerParameters(
        hopping=1.0,
        gamma=0.5,
        lambda_ep=1.5,
        drive_amplitude=1.0,
        pulse_width=1.0,
    )
    initial_state = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=args.phonon_cutoff,
    )
    direction = constrained_initial_direction(
        initial_state,
        parameters,
        seed=args.seed,
        metric=args.perturbation_metric,
    )
    rhs = _rhs_for_protocol(
        args.protocol,
        parameters,
        initial_state,
        activation_margin=args.activation_margin,
        barrier_rate=args.barrier_rate,
        cone_tolerance=args.cone_tolerance,
        projection_tolerance=args.projection_tolerance,
        correction_metric=args.correction_metric,
    )

    source_dir = Path(__file__).resolve().parent
    plan = {
        "schema": "paper_v_nearby_trajectory_plan_v1",
        "classification": "diagnostic",
        "evidence_status": "exploratory_local_not_promoted",
        "scientific_question": (
            "Do nearby admissible 31-coordinate trajectories exhibit "
            "converged exponential separation?"
        ),
        "protocol": args.protocol,
        "parameters": {
            "hopping": parameters.hopping,
            "gamma": parameters.gamma,
            "lambda_ep": parameters.lambda_ep,
            "drive_amplitude": parameters.drive_amplitude,
            "pulse_width": parameters.pulse_width,
            "phonon_cutoff": args.phonon_cutoff,
        },
        "initial_condition": "exact_ground_state_contractions",
        "perturbation": {
            "seed": args.seed,
            "size": args.perturbation_size,
            "renormalization_metric": args.perturbation_metric,
            "constraints": [
                "zero first-order energy change",
                "zero real correlation-trace change",
                "zero imaginary correlation-trace change",
            ],
        },
        "integration": {
            "method": "fixed_step_RK4",
            "final_time": args.final_time,
            "time_step": args.time_step,
            "renormalization_interval": args.renormalization_interval,
        },
        "controller": {
            "enabled": args.protocol == "joint_barrier",
            "activation_margin": args.activation_margin,
            "barrier_rate": args.barrier_rate,
            "cone_tolerance": args.cone_tolerance,
            "projection_tolerance": args.projection_tolerance,
            "correction_metric": args.correction_metric,
            "energy_neutral": True,
            "preserve_correlation_trace": True,
            "subtract_initial_residual": False,
        },
        "interpretation_contract": {
            "archive_after_cone_exit": "mathematical_ode_only",
            "positive_finite_time_exponent": "screen_not_chaos_proof",
            "required_checks": [
                "perturbation-size convergence above the controller tolerance floor",
                "time-step convergence",
                "direction convergence",
                "bounded physical trajectory",
            ],
        },
        "source_sha256": {
            "analysis": _sha256(Path(__file__).resolve()),
            "matrix_reference": _sha256(source_dir / "matrix_reference.py"),
            "cone_correction": _sha256(source_dir / "cone_correction.py"),
        },
    }
    _write_json_atomic(args.output_dir / "plan.json", plan)
    progress_path = args.output_dir / "progress.jsonl"

    def progress(payload: dict[str, float]) -> None:
        record = {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
        print(json.dumps(record, sort_keys=True), flush=True)

    estimate = benettin_nearby_trajectory(
        rhs,
        initial_state,
        direction,
        final_time=args.final_time,
        time_step=args.time_step,
        renormalization_interval=args.renormalization_interval,
        perturbation_size=args.perturbation_size,
        selected_metric=args.perturbation_metric,
        record_physicality=True,
        progress=progress,
    )
    np.savez_compressed(
        args.output_dir / "trajectory.npz",
        times=estimate.times,
        local_euclidean_exponents=estimate.local_euclidean_exponents,
        local_frobenius_exponents=estimate.local_frobenius_exponents,
        cumulative_euclidean_exponents=estimate.cumulative_euclidean_exponents,
        cumulative_frobenius_exponents=estimate.cumulative_frobenius_exponents,
        start_euclidean_distances=estimate.start_euclidean_distances,
        end_euclidean_distances=estimate.end_euclidean_distances,
        start_frobenius_distances=estimate.start_frobenius_distances,
        end_frobenius_distances=estimate.end_frobenius_distances,
        base_margins=estimate.base_margins,
        shadow_margins=estimate.shadow_margins,
        base_trace_residuals=estimate.base_trace_residuals,
        shadow_trace_residuals=estimate.shadow_trace_residuals,
        initial_direction=direction,
    )
    all_margins = np.concatenate(
        [estimate.base_margins, estimate.shadow_margins],
        axis=0,
    )
    summary = {
        "schema": "paper_v_nearby_trajectory_summary_v1",
        "status": "complete",
        "protocol": args.protocol,
        "final_time": args.final_time,
        "final_euclidean_ftle": float(
            estimate.cumulative_euclidean_exponents[-1]
        ),
        "final_frobenius_ftle": float(
            estimate.cumulative_frobenius_exponents[-1]
        ),
        "post_t4_euclidean_ftle": (
            estimate.post_time_exponent(4.0, "euclidean")
            if args.final_time > 4.0
            else None
        ),
        "post_t4_frobenius_ftle": (
            estimate.post_time_exponent(4.0, "frobenius")
            if args.final_time > 4.0
            else None
        ),
        "minimum_margins": {
            name: float(np.nanmin(all_margins[:, index]))
            for index, name in enumerate(MARGIN_NAMES)
        },
        "maximum_correlation_trace_residual": float(
            np.nanmax(
                np.concatenate(
                    [
                        estimate.base_trace_residuals,
                        estimate.shadow_trace_residuals,
                    ]
                )
            )
        ),
        "perturbation_to_cone_tolerance_ratio": float(
            args.perturbation_size / args.cone_tolerance
        ),
        "physicality_retained_at_sampled_resets": bool(
            np.nanmin(all_margins) >= -args.cone_tolerance
            and np.nanmax(
                np.concatenate(
                    [
                        estimate.base_trace_residuals,
                        estimate.shadow_trace_residuals,
                    ]
                )
            )
            <= args.cone_tolerance
        ),
        "interpretation": (
            "finite-time sensitivity screen only; a chaos claim requires "
            "convergence across the declared checks"
        ),
    }
    _write_json_atomic(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
