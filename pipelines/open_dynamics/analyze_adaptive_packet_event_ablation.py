#!/usr/bin/env python3
"""Ablate one zero-weight adaptive packet admission from the same ket."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from paper5.stability.conditional_packets import (
    electron_relative_product_to_local_state,
    electron_relative_state,
)
from paper5.stability.exact_reference import (
    _build_exact_dimer_model,
    _contract_matrix_state,
)
from paper5.stability.hubbard_dimer import DimerParameters, GaussianSineDrive
from paper5.stability.matrix_reference import (
    matrix_state_to_closed_scalar_coordinates,
)
from paper5.stability.mixed_enriched_propagation import normalized_packet_state
from paper5.stability.multi_coherent import (
    multi_coherent_rhs,
    pack_multi_coherent_parameters,
    unpack_multi_coherent_parameters,
)
from pipelines.open_dynamics.run_integrable_mixed_enriched_rollout import (
    DEFAULT_EXACT,
    _matched_indices,
    _observables,
    _write_json,
)


DEFAULT_RUN = Path(
    "output/local_runs/"
    "paper_v_archive_gram_adaptive_packet_cutoff16_t20_20260805_v1"
)
DEFAULT_OUTPUT = Path(
    "output/local_runs/"
    "paper_v_archive_gram_adaptive_packet_t19_ablation_cutoff16_20260805_v1"
)


def _remove_last_zero_weight_packet(parameters: np.ndarray) -> np.ndarray:
    coefficients, displacements = unpack_multi_coherent_parameters(parameters)
    if coefficients.shape[1] < 2:
        raise ValueError("cannot remove the only packet")
    if np.max(np.abs(coefficients[:, -1])) > 1e-14:
        raise ValueError("last packet is not the zero-weight admission")
    return pack_multi_coherent_parameters(
        coefficients[:, :-1],
        displacements[:, :-1],
    )


def _propagate_segmented(
    initial: np.ndarray,
    *,
    start_time: float,
    final_time: float,
    sample_step: float,
    segment_length: float,
    maximum_step: float,
    relative_damping: float,
    parameters: DimerParameters,
    drive: GaussianSineDrive,
    relative_dimension: int,
) -> tuple[np.ndarray, list[np.ndarray], int]:
    sample_count = int(round((final_time - start_time) / sample_step))
    times = np.linspace(start_time, final_time, sample_count + 1)
    values = np.asarray(initial, dtype=float).copy()
    states: list[np.ndarray] = [values.copy()]
    total_evaluations = 0

    def rhs(time_value: float, current: np.ndarray) -> np.ndarray:
        return multi_coherent_rhs(
            time_value,
            current,
            parameters,
            relative_dimension=relative_dimension,
            drive_protocol=drive,
            regularization="tikhonov",
            relative_damping=relative_damping,
        )

    segment_count = int(round((final_time - start_time) / segment_length))
    boundaries = np.linspace(start_time, final_time, segment_count + 1)
    for segment_index, (left, right) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        del segment_index
        selected = times[(times > left) & (times <= right + 1e-12)]
        selected[-1] = right
        solution = solve_ivp(
            rhs,
            (float(left), float(right)),
            values,
            method="DOP853",
            t_eval=selected,
            rtol=1e-8,
            atol=1e-10,
            max_step=maximum_step,
        )
        if not solution.success or solution.y.shape[1] != selected.size:
            raise RuntimeError(f"event ablation failed: {solution.message}")
        states.extend(
            np.asarray(solution.y[:, index], dtype=float).copy()
            for index in range(selected.size)
        )
        values = states[-1]
        total_evaluations += int(solution.nfev)
    if len(states) != times.size:
        raise RuntimeError("event ablation did not fill the sample grid")
    return times, states, total_evaluations


def run(
    run_directory: Path,
    exact_arrays_path: Path,
    output_directory: Path,
    *,
    event_time: float = 19.0,
) -> dict[str, object]:
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run_directory / "summary.json").read_text())
    settings = summary["parameters"]
    parameters = DimerParameters(
        gamma=float(settings["gamma"]),
        lambda_ep=float(settings["lambda_ep"]),
        drive_amplitude=float(settings["drive_protocol"]["amplitude"]),
        pulse_width=float(settings["drive_protocol"]["pulse_width"]),
    )
    drive = GaussianSineDrive(
        amplitude=float(settings["drive_protocol"]["amplitude"]),
        pulse_width=float(settings["drive_protocol"]["pulse_width"]),
        delays=tuple(float(value) for value in settings["drive_protocol"]["delays"]),
    )
    cutoff = int(settings["phonon_cutoff"])
    relative_dimension = 2 * cutoff + 1
    with np.load(
        run_directory / "mixed_guided_packet_rollout.npz",
        allow_pickle=False,
    ) as arrays:
        full_times = np.asarray(arrays["times"], dtype=float)
        event_index = int(np.argmin(np.abs(full_times - event_time)))
        if not np.isclose(full_times[event_index], event_time, atol=1e-12):
            raise ValueError("event time is absent from the stored grid")
        packet_count = int(arrays["packet_count_trajectory"][event_index])
        admitted = np.asarray(
            arrays["parameter_trajectory"][event_index, : 16 * packet_count],
            dtype=float,
        )
        coordinate_scales = np.asarray(arrays["coordinate_scales"], dtype=float)
    ablated = _remove_last_zero_weight_packet(admitted)
    admitted_state = normalized_packet_state(
        admitted,
        relative_dimension=relative_dimension,
    )
    ablated_state = normalized_packet_state(
        ablated,
        relative_dimension=relative_dimension,
    )
    state_discontinuity = float(np.linalg.norm(admitted_state - ablated_state))

    paths: dict[str, list[np.ndarray]] = {}
    evaluation_counts: dict[str, int] = {}
    times: np.ndarray | None = None
    for name, initial in (("ablated_k7", ablated), ("admitted_k8", admitted)):
        local_times, states, evaluations = _propagate_segmented(
            initial,
            start_time=event_time,
            final_time=float(settings["final_time"]),
            sample_step=float(settings["sample_step"]),
            segment_length=float(settings["adaptive_segment_length"]),
            maximum_step=float(settings["maximum_step"]),
            relative_damping=float(settings["relative_damping"]),
            parameters=parameters,
            drive=drive,
            relative_dimension=relative_dimension,
        )
        if times is None:
            times = local_times
        elif not np.array_equal(times, local_times):
            raise RuntimeError("ablation paths use different grids")
        paths[name] = states
        evaluation_counts[name] = evaluations
    assert times is not None

    with np.load(exact_arrays_path, allow_pickle=False) as arrays:
        exact_times = np.asarray(arrays["times"], dtype=float)
        exact_indices = _matched_indices(exact_times, times)
        exact_closed = np.asarray(
            arrays["exact_dop853_closed"], dtype=float
        )[0, exact_indices]
        all_exact_states = np.asarray(
            arrays["exact_dop853_state_vectors"], dtype=complex
        )[0]
        exact_states = all_exact_states[exact_indices]
    center_state = electron_relative_state(
        all_exact_states[0],
        phonon_cutoff=cutoff,
    ).center_state
    exact_model = _build_exact_dimer_model(parameters, phonon_cutoff=cutoff)
    exact_observables = _observables(times, exact_closed, parameters)
    closed_paths: dict[str, np.ndarray] = {}
    observable_paths: dict[str, np.ndarray] = {}
    fidelities: dict[str, np.ndarray] = {}
    metrics: dict[str, object] = {}
    for name, states in paths.items():
        closed = np.empty((times.size, 31), dtype=float)
        fidelity = np.empty(times.size)
        for index, packet_values in enumerate(states):
            relative_state = normalized_packet_state(
                packet_values,
                relative_dimension=relative_dimension,
            )
            embedded = electron_relative_product_to_local_state(
                relative_state,
                center_state,
                phonon_cutoff=cutoff,
            ).state
            closed[index] = matrix_state_to_closed_scalar_coordinates(
                _contract_matrix_state(exact_model, embedded)
            )
            fidelity[index] = float(abs(np.vdot(exact_states[index], embedded)) ** 2)
        observables = _observables(times, closed, parameters)
        scaled_error = (closed - exact_closed) / coordinate_scales
        observable_error = observables - exact_observables
        metrics[name] = {
            "packet_count": states[0].size // 16,
            "function_evaluations": evaluation_counts[name],
            "all31_scaled_rms": float(np.sqrt(np.mean(scaled_error**2))),
            "c_scaled_rms": float(np.sqrt(np.mean(scaled_error[:, 17:31] ** 2))),
            "observable_rms": [
                float(np.sqrt(np.mean(observable_error[:, column] ** 2)))
                for column in range(observable_error.shape[1])
            ],
            "final_exact_state_fidelity": float(fidelity[-1]),
        }
        closed_paths[name] = closed
        observable_paths[name] = observables
        fidelities[name] = fidelity
    result: dict[str, object] = {
        "schema": "paper_v_adaptive_packet_event_ablation_v1",
        "status": "complete",
        "event_time": event_time,
        "final_time": float(settings["final_time"]),
        "state_discontinuity": state_discontinuity,
        "online_exact_reference_used": False,
        "metrics": metrics,
    }
    np.savez_compressed(
        output_directory / "event_ablation.npz",
        times=times,
        exact_closed_coordinates=exact_closed,
        exact_observables=exact_observables,
        ablated_closed_coordinates=closed_paths["ablated_k7"],
        admitted_closed_coordinates=closed_paths["admitted_k8"],
        ablated_observables=observable_paths["ablated_k7"],
        admitted_observables=observable_paths["admitted_k8"],
        ablated_fidelity=fidelities["ablated_k7"],
        admitted_fidelity=fidelities["admitted_k8"],
    )
    _write_json(output_directory / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--exact-arrays", type=Path, default=DEFAULT_EXACT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--event-time", type=float, default=19.0)
    arguments = parser.parse_args()
    run(
        arguments.run_directory,
        arguments.exact_arrays,
        arguments.output_directory,
        event_time=arguments.event_time,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
