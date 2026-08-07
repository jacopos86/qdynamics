"""Compare raw and representability-corrected 31-moment EOMs at fixed coupling.

The initial moment state and its interpretation are explicit command inputs.
The run tests the archive vector field and controller; it is not an accuracy
comparison unless a matched exact trajectory is supplied separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from paper5.stability.cone_correction import (
    structured_electron_phonon_barrier_correction,
)
from paper5.stability.hubbard_dimer import DimerParameters, GaussianSineDrive
from paper5.stability.initial_condition_sensitivity import physicality_diagnostics
from paper5.stability.matrix_reference import (
    closed_scalar_rhs,
    closed_scalar_to_matrix_state,
    matrix_total_energy,
)


Vector = np.ndarray
Rhs = Callable[[float, Vector], Vector]
MARGIN_NAMES = (
    "electron_lower",
    "electron_upper",
    "boson_moment",
    "joint_gram",
)


class ProtocolParameters:
    """Delegate static dimer parameters while replacing the pulse."""

    def __init__(self, base: DimerParameters, drive: GaussianSineDrive) -> None:
        self._base = base
        self._drive = drive

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def drive_difference(self, time: float) -> float:
        return self._drive.difference(time)


@dataclass
class ControllerAudit:
    parameters: Any
    evaluations: int = 0
    active_evaluations: int = 0
    nonconverged_evaluations: int = 0
    maximum_correction_norm: float = 0.0
    sum_squared_correction_norm: float = 0.0
    maximum_correction_energy_flux: float = 0.0
    minimum_corrected_joint_barrier: float = np.inf

    def corrected_velocity(
        self,
        time: float,
        state: Vector,
        proposed: Vector,
    ) -> Vector:
        del time
        result = structured_electron_phonon_barrier_correction(
            state,
            proposed,
            self.parameters,
            activation_margin=1e-5,
            target_flux=0.0,
            barrier_rate=5.0,
            energy_neutral=True,
            preserve_correlation_trace=True,
            cone_tolerance=1e-8,
            projection_tolerance=1e-12,
            maximum_constraints=128,
            correction_metric="euclidean",
        )
        self.evaluations += 1
        norm = float(result.correction_norm)
        if norm > 1e-14:
            self.active_evaluations += 1
        self.maximum_correction_norm = max(self.maximum_correction_norm, norm)
        self.sum_squared_correction_norm += norm**2
        self.maximum_correction_energy_flux = max(
            self.maximum_correction_energy_flux,
            abs(float(result.correction_energy_flux)),
        )
        self.minimum_corrected_joint_barrier = min(
            self.minimum_corrected_joint_barrier,
            float(result.corrected_joint_barrier_minimum_eigenvalue),
        )
        if not result.converged:
            self.nonconverged_evaluations += 1
            raise RuntimeError("representability correction did not converge")
        return proposed + np.asarray(result.correction_coordinates, dtype=float)

    def summary(self) -> dict[str, float | int]:
        rms = (
            float(np.sqrt(self.sum_squared_correction_norm / self.evaluations))
            if self.evaluations
            else 0.0
        )
        active_fraction = (
            self.active_evaluations / self.evaluations if self.evaluations else 0.0
        )
        return {
            "evaluations": self.evaluations,
            "active_evaluations": self.active_evaluations,
            "active_fraction": active_fraction,
            "nonconverged_evaluations": self.nonconverged_evaluations,
            "maximum_correction_norm": self.maximum_correction_norm,
            "rms_correction_norm": rms,
            "maximum_correction_energy_flux": self.maximum_correction_energy_flux,
            "minimum_corrected_joint_barrier": (
                self.minimum_corrected_joint_barrier
                if np.isfinite(self.minimum_corrected_joint_barrier)
                else None
            ),
        }


@dataclass(frozen=True)
class LaneResult:
    times: np.ndarray
    states: np.ndarray
    margins: np.ndarray
    trace_residuals: np.ndarray
    internal_energies: np.ndarray
    rhs_evaluations: int
    failure_time: float | None
    failure_reason: str | None


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _diagnostics(state: Vector, parameters: Any) -> tuple[np.ndarray, float, float]:
    margins, trace = physicality_diagnostics(state)
    energy = matrix_total_energy(closed_scalar_to_matrix_state(state), parameters)
    return margins, trace, energy


def _integrate(
    rhs: Rhs,
    initial_state: Vector,
    parameters: Any,
    *,
    lane: str,
    final_time: float,
    time_step: float,
    sample_step: float,
    failure_threshold: float,
    progress_path: Path,
) -> LaneResult:
    total_steps = int(round(final_time / time_step))
    sample_stride = int(round(sample_step / time_step))
    if not np.isclose(total_steps * time_step, final_time, atol=1e-12, rtol=0.0):
        raise ValueError("time_step must divide final_time")
    if not np.isclose(sample_stride * time_step, sample_step, atol=1e-12, rtol=0.0):
        raise ValueError("time_step must divide sample_step")

    state = np.asarray(initial_state, dtype=float).copy()
    times = [0.0]
    states = [state.copy()]
    margin, trace, energy = _diagnostics(state, parameters)
    margins = [margin]
    traces = [trace]
    energies = [energy]
    rhs_evaluations = 0
    failure_time: float | None = None
    failure_reason: str | None = None

    def progress(time: float) -> None:
        record = {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "lane": lane,
            "time": time,
            "maximum_absolute_coordinate": float(np.max(np.abs(state))),
            "minimum_joint_gram_eigenvalue": float(margins[-1][3]),
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
        print(json.dumps(record, sort_keys=True), flush=True)

    for step_index in range(total_steps):
        time = step_index * time_step
        try:
            k1 = np.asarray(rhs(time, state), dtype=float)
            k2 = np.asarray(
                rhs(time + 0.5 * time_step, state + 0.5 * time_step * k1),
                dtype=float,
            )
            k3 = np.asarray(
                rhs(time + 0.5 * time_step, state + 0.5 * time_step * k2),
                dtype=float,
            )
            k4 = np.asarray(
                rhs(time + time_step, state + time_step * k3),
                dtype=float,
            )
        except Exception as error:  # diagnostic must preserve partial results
            failure_time = float(time)
            failure_reason = f"rhs_error:{type(error).__name__}:{error}"
            break
        rhs_evaluations += 4
        state = state + (time_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        next_time = (step_index + 1) * time_step
        maximum = float(np.max(np.abs(state)))
        if not np.all(np.isfinite(state)):
            failure_time = float(next_time)
            failure_reason = "nonfinite_state"
        elif maximum >= failure_threshold:
            failure_time = float(next_time)
            failure_reason = "coordinate_threshold"

        if (step_index + 1) % sample_stride == 0 or failure_reason is not None:
            times.append(next_time)
            states.append(state.copy())
            if np.all(np.isfinite(state)):
                margin, trace, energy = _diagnostics(state, parameters)
            else:
                margin = np.full(4, np.nan)
                trace = np.nan
                energy = np.nan
            margins.append(margin)
            traces.append(trace)
            energies.append(energy)
            if abs(next_time - round(next_time)) <= 0.5 * time_step:
                progress(next_time)
        if failure_reason is not None:
            break

    return LaneResult(
        times=np.asarray(times, dtype=float),
        states=np.asarray(states, dtype=float),
        margins=np.asarray(margins, dtype=float),
        trace_residuals=np.asarray(traces, dtype=float),
        internal_energies=np.asarray(energies, dtype=float),
        rhs_evaluations=rhs_evaluations,
        failure_time=failure_time,
        failure_reason=failure_reason,
    )


def _lane_summary(result: LaneResult, final_time: float) -> dict[str, object]:
    finite_margins = np.where(np.isfinite(result.margins), result.margins, np.inf)
    minima = np.min(finite_margins, axis=0)
    minima[~np.isfinite(minima)] = np.nan
    post_mask = result.times >= 12.0
    post_energy_range = (
        float(np.ptp(result.internal_energies[post_mask]))
        if np.count_nonzero(post_mask) >= 2
        else None
    )
    return {
        "completed": bool(result.failure_time is None and result.times[-1] >= final_time),
        "last_time": float(result.times[-1]),
        "failure_time": result.failure_time,
        "failure_reason": result.failure_reason,
        "rhs_evaluations": result.rhs_evaluations,
        "maximum_absolute_coordinate": float(np.nanmax(np.abs(result.states))),
        "minimum_margins": {
            name: float(value) for name, value in zip(MARGIN_NAMES, minima, strict=True)
        },
        "maximum_correlation_trace_residual": float(
            np.nanmax(result.trace_residuals)
        ),
        "post_second_pulse_internal_energy_range": post_energy_range,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coupling", type=float, default=2.0)
    parser.add_argument("--final-time", type=float, default=20.0)
    parser.add_argument("--time-step", type=float, default=0.01)
    parser.add_argument("--sample-step", type=float, default=0.05)
    parser.add_argument("--failure-threshold", type=float, default=1e4)
    parser.add_argument(
        "--pulse-delays",
        type=float,
        nargs="+",
        default=(0.0, 8.0),
        help="Causal pulse delays; pass '--pulse-delays 0' for one pulse.",
    )
    parser.add_argument(
        "--initial-source",
        type=Path,
        default=Path(
            "output/local_runs/"
            "paper_v_trajectory_closure_identifiability_dense_cutoff16_20260804_v1/"
            "trajectory_closure_identifiability.npz"
        ),
    )
    parser.add_argument("--initial-key", default="dop853_closed")
    parser.add_argument(
        "--initial-description",
        default=(
            "central correlated cutoff-16 moment contraction from the "
            "validated lambda=1.5 preparation"
        ),
    )
    parser.add_argument(
        "--initial-interpretation",
        default="sudden coupling quench; not the equilibrium state at this coupling",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.coupling <= 0.0:
        raise SystemExit("coupling must be positive")
    pulse_delays = tuple(float(value) for value in args.pulse_delays)
    if not pulse_delays or any(value < 0.0 for value in pulse_delays):
        raise SystemExit("pulse delays must be nonnegative")
    if tuple(sorted(pulse_delays)) != pulse_delays:
        raise SystemExit("pulse delays must be sorted")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = np.load(args.initial_source)
    if args.initial_key not in source:
        raise SystemExit(f"initial key not found: {args.initial_key}")
    initial_rows = np.asarray(source[args.initial_key], dtype=float)
    if initial_rows.shape[-1] != 31:
        raise SystemExit(
            f"initial array must end in 31 coordinates, got {initial_rows.shape}"
        )
    initial_state = initial_rows.reshape(-1, 31)[0].copy()

    hopping = 1.0
    gamma = 0.5
    omega = hopping * gamma
    lambda_ep = 2.0 * args.coupling**2 / (hopping * omega)
    base = DimerParameters(
        hopping=hopping,
        gamma=gamma,
        lambda_ep=lambda_ep,
        drive_amplitude=1.0,
        pulse_width=1.0,
    )
    if not np.isclose(base.coupling, args.coupling, atol=1e-14, rtol=1e-14):
        raise RuntimeError("coupling conversion failed")
    drive = GaussianSineDrive(
        amplitude=1.0,
        pulse_width=1.0,
        delays=pulse_delays,
    )
    parameters = ProtocolParameters(base, drive)

    plan = {
        "schema": "paper5.coupling_controller_comparison.plan.v2",
        "classification": "diagnostic_exploratory_not_promoted",
        "execution_authorized": True,
        "question": (
            "Does the minimum-norm representability controller keep the "
            f"31-coordinate archive EOM physical and bounded at g={args.coupling:g}?"
        ),
        "parameters": {
            "hopping": hopping,
            "omega_ph": omega,
            "gamma": gamma,
            "coupling_g": args.coupling,
            "lambda_ep": lambda_ep,
        },
        "drive": {
            "amplitude": 1.0,
            "pulse_width": 1.0,
            "delays": list(pulse_delays),
        },
        "initial_condition": {
            "description": args.initial_description,
            "interpretation": args.initial_interpretation,
            "source_key": args.initial_key,
            "source_sha256": _sha256(args.initial_source),
        },
        "integration": {
            "method": "fixed_step_RK4",
            "final_time": args.final_time,
            "time_step": args.time_step,
            "sample_step": args.sample_step,
            "failure_threshold": args.failure_threshold,
        },
        "lanes": {
            "raw": "archive 31-coordinate EOM",
            "corrected": (
                "same archive EOM plus Euclidean minimum-norm joint-Gram, "
                "electronic, trace, and energy-neutral velocity correction"
            ),
        },
    }
    _atomic_json(args.output_dir / "plan.json", plan)
    progress_path = args.output_dir / "progress.jsonl"

    raw_rhs = lambda time, state: closed_scalar_rhs(time, state, parameters)
    raw = _integrate(
        raw_rhs,
        initial_state,
        parameters,
        lane="raw",
        final_time=args.final_time,
        time_step=args.time_step,
        sample_step=args.sample_step,
        failure_threshold=args.failure_threshold,
        progress_path=progress_path,
    )

    audit = ControllerAudit(parameters)

    def corrected_rhs(time: float, state: Vector) -> Vector:
        proposed = closed_scalar_rhs(time, state, parameters)
        return audit.corrected_velocity(time, state, proposed)

    corrected = _integrate(
        corrected_rhs,
        initial_state,
        parameters,
        lane="corrected",
        final_time=args.final_time,
        time_step=args.time_step,
        sample_step=args.sample_step,
        failure_threshold=args.failure_threshold,
        progress_path=progress_path,
    )

    np.savez_compressed(
        args.output_dir / "trajectories.npz",
        initial_state=initial_state,
        raw_times=raw.times,
        raw_states=raw.states,
        raw_margins=raw.margins,
        raw_trace_residuals=raw.trace_residuals,
        raw_internal_energies=raw.internal_energies,
        corrected_times=corrected.times,
        corrected_states=corrected.states,
        corrected_margins=corrected.margins,
        corrected_trace_residuals=corrected.trace_residuals,
        corrected_internal_energies=corrected.internal_energies,
    )
    metrics = {
        "schema": "paper5.coupling_controller_comparison.metrics.v2",
        "classification": "diagnostic_exploratory_not_promoted",
        "parameters": plan["parameters"],
        "initial_condition": plan["initial_condition"],
        "raw": _lane_summary(raw, args.final_time),
        "corrected": {
            **_lane_summary(corrected, args.final_time),
            "controller": audit.summary(),
        },
        "interpretation_limits": [
            plan["initial_condition"]["interpretation"],
            (
                "No matched exact trajectory is scored, so this run tests "
                "representability containment rather than physical accuracy."
            ),
        ],
    }
    _atomic_json(args.output_dir / "metrics.json", metrics)
    _atomic_json(
        args.output_dir / "runtime_manifest.json",
        {
            "schema": "paper5.coupling_controller_comparison.runtime.v2",
            "artifact_hashes": {
                name: _sha256(args.output_dir / name)
                for name in ("plan.json", "metrics.json", "trajectories.npz")
            },
        },
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
