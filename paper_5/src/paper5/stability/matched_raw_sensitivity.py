"""Matched nearby-state propagation for the raw 31-coordinate archive EOM."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .cone_correction import closed_state_lifted_frobenius_norm
from .hubbard_dimer import DimerParameters, FloatArray
from .initial_condition_sensitivity import physicality_diagnostics
from .matrix_reference import closed_scalar_rhs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rk4_step(
    time_value: float,
    states: FloatArray,
    time_step: float,
    parameters: DimerParameters,
) -> FloatArray:
    def evaluate(time_point: float, values: FloatArray) -> FloatArray:
        return np.stack(
            [closed_scalar_rhs(time_point, state, parameters) for state in values]
        )

    k1 = evaluate(time_value, states)
    k2 = evaluate(time_value + 0.5 * time_step, states + 0.5 * time_step * k1)
    k3 = evaluate(time_value + 0.5 * time_step, states + 0.5 * time_step * k2)
    k4 = evaluate(time_value + time_step, states + time_step * k3)
    return np.asarray(
        states + (time_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
        dtype=float,
    )


def _first_crossing(
    times: FloatArray,
    values: FloatArray,
    predicate: np.ndarray,
) -> float | None:
    indices = np.flatnonzero(predicate)
    return None if indices.size == 0 else float(times[int(indices[0])])


def run(
    source_trajectory: Path,
    output_directory: Path,
    *,
    final_time: float = 100.0,
    time_step: float = 0.02,
    sample_step: float = 0.1,
) -> dict[str, object]:
    """Propagate the matched initial states with the uncorrected archive EOM."""

    with np.load(source_trajectory, allow_pickle=False) as source:
        states = np.asarray(source["sampled_states"][0], dtype=float).copy()
        labels = np.asarray(source["labels"])
        exact_times = np.asarray(source["exact_times"], dtype=float)
        exact_distances = np.asarray(
            source["exact_frobenius_distances"], dtype=float
        )

    step_count = int(round(final_time / time_step))
    sample_stride = int(round(sample_step / time_step))
    if not np.isclose(step_count * time_step, final_time, atol=1e-12):
        raise ValueError("time_step must divide final_time")
    if sample_stride <= 0 or not np.isclose(
        sample_stride * time_step, sample_step, atol=1e-12
    ):
        raise ValueError("time_step must divide sample_step")
    if step_count % sample_stride:
        raise ValueError("sample_step must divide final_time")

    sample_times = np.linspace(0.0, final_time, step_count // sample_stride + 1)
    sampled_states = np.empty((sample_times.size, *states.shape), dtype=float)
    distances = np.empty((states.shape[0] - 1, sample_times.size), dtype=float)
    margins = np.empty((sample_times.size, states.shape[0], 4), dtype=float)
    trace_residuals = np.empty((sample_times.size, states.shape[0]), dtype=float)

    def record(index: int) -> None:
        sampled_states[index] = states
        for trajectory_index, state in enumerate(states):
            margins[index, trajectory_index], trace_residuals[
                index, trajectory_index
            ] = physicality_diagnostics(state)
        for case_index in range(states.shape[0] - 1):
            distances[case_index, index] = closed_state_lifted_frobenius_norm(
                states[case_index + 1] - states[0]
            )

    record(0)
    started = time.perf_counter()
    sample_index = 0
    for step_index in range(step_count):
        time_value = step_index * time_step
        states = _rk4_step(time_value, states, time_step, DimerParameters(
            hopping=1.0,
            gamma=0.5,
            lambda_ep=1.5,
            drive_amplitude=1.0,
        ))
        if (step_index + 1) % sample_stride == 0:
            sample_index += 1
            record(sample_index)
        if (step_index + 1) % max(1, step_count // 10) == 0:
            print(
                json.dumps(
                    {
                        "time": (step_index + 1) * time_step,
                        "fraction": (step_index + 1) / step_count,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )

    elapsed = time.perf_counter() - started
    amplification = distances / distances[:, :1]
    minimum_base_margins = np.min(margins[:, 0], axis=0)
    first_joint_violation = _first_crossing(
        sample_times,
        margins[:, 0, 3],
        margins[:, 0, 3] < -1e-8,
    )
    first_trace_violation = _first_crossing(
        sample_times,
        trace_residuals[:, 0],
        trace_residuals[:, 0] > 1e-8,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_directory / "trajectory.npz",
        sample_times=sample_times,
        labels=labels,
        raw_sampled_states=sampled_states,
        raw_frobenius_distances=distances,
        raw_amplification=amplification,
        raw_margins=margins,
        raw_trace_residuals=trace_residuals,
        exact_times=exact_times,
        exact_frobenius_distances=exact_distances,
    )
    summary: dict[str, object] = {
        "schema": "paper_v_matched_raw_sensitivity_v1",
        "classification": "exploratory_local_not_promoted",
        "model": "uncorrected_31_coordinate_archive_eom",
        "parameters": {
            "hopping": 1.0,
            "gamma": 0.5,
            "lambda_ep": 1.5,
            "drive_amplitude": 1.0,
            "final_time": final_time,
            "time_step": time_step,
            "sample_step": sample_step,
        },
        "raw_final_amplification": amplification[:, -1].tolist(),
        "raw_maximum_amplification": np.max(amplification, axis=1).tolist(),
        "minimum_base_margins": minimum_base_margins.tolist(),
        "maximum_base_correlation_trace_residual": float(
            np.max(trace_residuals[:, 0])
        ),
        "first_sampled_joint_gram_violation": first_joint_violation,
        "first_sampled_correlation_trace_violation": first_trace_violation,
        "elapsed_seconds": elapsed,
        "source": str(source_trajectory.resolve()),
        "source_sha256": _sha256(source_trajectory),
        "interpretation": (
            "The matched raw mathematical ODE may be continued after it leaves "
            "the representable moment domain, but later separation then does not "
            "establish physical initial-condition sensitivity."
        ),
    }
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-trajectory",
        type=Path,
        default=Path(
            "output/local_runs/"
            "paper_v_matched_exact_controller_sensitivity_t100_dt002_20260803_v1/"
            "trajectory.npz"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            "output/local_runs/"
            "paper_v_matched_exact_raw_sensitivity_t100_dt002_20260812_v1"
        ),
    )
    parser.add_argument("--final-time", type=float, default=100.0)
    parser.add_argument("--time-step", type=float, default=0.02)
    parser.add_argument("--sample-step", type=float, default=0.1)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.source_trajectory,
                args.output_directory,
                final_time=args.final_time,
                time_step=args.time_step,
                sample_step=args.sample_step,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
