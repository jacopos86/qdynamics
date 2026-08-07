#!/usr/bin/env python3
"""Audit whether split-local reciprocal frames form a usable moving atlas."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from paper5.stability.archive_auxiliary_memory import (
    build_archive_auxiliary_frame_from_observables,
)
from paper5.stability.exact_reference import (
    _build_exact_dimer_model,
    _contract_matrix_state,
)
from paper5.stability.finite_horizon_auxiliary import (
    FiniteHorizonScenario,
    finite_horizon_reachable_observable_audit,
)
from paper5.stability.hubbard_dimer import DimerParameters, GaussianSineDrive
from paper5.stability.matrix_reference import (
    matrix_state_to_closed_scalar_coordinates,
    pauli_repaired_closed_scalar_rhs,
)
from paper5.stability.moving_frame_viability import (
    moving_frame_viability_audit,
)
from paper5.stability.multi_coherent_scores import development_coordinate_scales
from paper5.stability.reachability_observability import (
    build_drive_aware_word_envelope,
)

RUN_ID = (
    "paper_v_archive_auxiliary_moving_frame_viability_"
    "cutoff16_t20_20260805_v2"
)
DEFAULT_SINGLE = Path(
    "output/local_runs/"
    "paper_v_exact_vs_31d_cutoff_convergence_t20_local_20260801_v1/"
    "trajectories_cutoff_16.npz"
)
DEFAULT_DOUBLE = Path(
    "output/local_runs/"
    "paper_v_multi_coherent_double_pulse_sealed_score_cutoff16_20260804_v1/"
    "score_arrays.npz"
)
DEFAULT_PREPARATIONS = Path(
    "output/local_runs/"
    "paper_v_multi_coherent_double_pulse_prepared_cutoff16_20260804_v1/"
    "frozen_initial_conditions.npz"
)
DEFAULT_OUTPUT = Path("output/local_runs") / RUN_ID
MEMBERS = ("central", "plus", "minus")


class _FixedDriveParameters:
    def __init__(self, parameters: DimerParameters, drive_value: float) -> None:
        self._parameters = parameters
        self._drive_value = float(drive_value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._parameters, name)

    def drive_difference(self, time_value: float) -> float:
        del time_value
        return self._drive_value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _json_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _order_record(order) -> dict[str, object]:
    record = asdict(order)
    return {
        key: _json_number(value) if isinstance(value, float) else value
        for key, value in record.items()
    }


def _first_order_below(
    orders,
    threshold: float,
) -> dict[str, int] | None:
    for order in orders:
        if max(
            order.worst_reachability_residual,
            order.worst_observability_residual,
        ) <= threshold:
            return {
                "pair_count": order.pair_count,
                "minimum_local_order": order.minimum_local_order,
                "maximum_local_order": order.maximum_local_order,
            }
    return None


def run(
    single_path: Path,
    double_path: Path,
    preparations_path: Path,
    output_directory: Path,
    *,
    final_time: float = 20.0,
    maximum_word_depth: int = 2,
    split_times: tuple[float, ...] = (0.5, 2.0, 4.0, 8.5, 10.0, 12.0, 16.0, 19.0),
    rank_tolerance: float = 1e-9,
) -> dict[str, object]:
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty output directory {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    parameters = DimerParameters(
        hopping=1.0,
        gamma=0.5,
        lambda_ep=1.5,
        drive_amplitude=1.0,
        pulse_width=1.0,
    )
    single_protocol = GaussianSineDrive.from_parameters(parameters)
    double_protocol = GaussianSineDrive.from_parameters(
        parameters,
        delays=(0.0, 8.0),
    )
    plan: dict[str, object] = {
        "schema": "paper_v_archive_auxiliary_moving_frame_viability_plan_v1",
        "run_id": output_directory.name,
        "classification": "offline_development_construction_diagnostic",
        "evidence_status": "exploratory_local_not_promoted",
        "phonon_cutoff": 16,
        "lambda_ep": parameters.lambda_ep,
        "gamma": parameters.gamma,
        "final_time": final_time,
        "maximum_word_depth": maximum_word_depth,
        "split_times": list(split_times),
        "rank_tolerance": rank_tolerance,
        "preparation_weight": 1.0,
        "scenario_labels": [
            "single_central",
            "double_central",
            "double_plus",
            "double_minus",
        ],
        "double_pulse_reference_use": (
            "offline construction only; its prior refined DOP853-versus-"
            "midpoint moment disagreement was 1.0550496242626838e-6, just "
            "above the declared 1e-6 validation ceiling"
        ),
        "autonomous_rollout_executed": False,
        "online_exact_reference_used": False,
        "scientific_dimension_cap": None,
        "pair_curve_stop": "all split-local frames saturated or local Hankel support exhausted",
    }
    _write_json(output_directory / "plan.json", plan)
    started = time.time()

    print("[moving-frame] loading compatible development paths", flush=True)
    with np.load(single_path, allow_pickle=False) as payload:
        single_times_all = np.asarray(payload["times"], dtype=float)
        single_selected = single_times_all <= final_time + 1e-12
        single_times = single_times_all[single_selected]
        single_closed = np.asarray(payload["exact_coordinates"], dtype=float)[
            single_selected
        ]
    with np.load(double_path, allow_pickle=False) as payload:
        double_times_all = np.asarray(payload["times"], dtype=float)
        double_selected = double_times_all <= final_time + 1e-12
        double_times = double_times_all[double_selected]
        double_closed = np.asarray(payload["exact_dop853_closed"], dtype=float)[
            :, double_selected
        ]
    with np.load(preparations_path, allow_pickle=False) as payload:
        initial_states = np.asarray(
            payload["exact_initial_state_vectors"],
            dtype=complex,
        )
        declared_initial_closed = np.asarray(
            payload["exact_initial_closed_coordinates"],
            dtype=float,
        )
    if not np.isclose(single_times[-1], final_time, atol=1e-12):
        raise ValueError("single-pulse path does not sample final_time")
    if not np.isclose(double_times[-1], final_time, atol=1e-12):
        raise ValueError("double-pulse path does not sample final_time")
    if not np.allclose(single_times, double_times, rtol=0.0, atol=1e-12):
        raise ValueError("single- and double-pulse development grids differ")
    double_times = single_times.copy()
    if initial_states.shape != (3, 1156):
        raise ValueError("frozen exact preparations have an unexpected shape")

    model = _build_exact_dimer_model(parameters, phonon_cutoff=16)
    contracted_initial = np.asarray(
        [
            matrix_state_to_closed_scalar_coordinates(
                _contract_matrix_state(model, state)
            )
            for state in initial_states
        ]
    )
    contraction_residual = float(
        np.max(np.abs(contracted_initial - declared_initial_closed))
    )
    path_initial_residual = float(
        max(
            np.max(np.abs(single_closed[0] - contracted_initial[0])),
            np.max(np.abs(double_closed[:, 0] - contracted_initial)),
        )
    )
    if contraction_residual > 2e-10 or path_initial_residual > 2e-10:
        raise RuntimeError("frozen preparation contractions do not match paths")

    coordinate_scales = development_coordinate_scales(
        np.concatenate((single_closed, *tuple(double_closed)), axis=0),
        phonon_cutoff=16,
    )
    print("[moving-frame] building common preparation-aware word envelope", flush=True)
    envelope = build_drive_aware_word_envelope(
        parameters,
        phonon_cutoff=16,
        maximum_word_depth=maximum_word_depth,
        rank_tolerance=rank_tolerance,
        preparation_state_vectors=tuple(initial_states),
    )
    frame = build_archive_auxiliary_frame_from_observables(
        envelope.construction,
        envelope.hidden_observables,
    )

    def archive_field(state: np.ndarray, drive_value: float) -> np.ndarray:
        return pauli_repaired_closed_scalar_rhs(
            0.0,
            state,
            _FixedDriveParameters(parameters, drive_value),  # type: ignore[arg-type]
        )

    initial_memory = []
    for closed, state in zip(contracted_initial, initial_states, strict=True):
        initial_memory.append(
            frame.initialize_memory(
                closed,
                state,
                archive_field,
                drive_value=single_protocol.difference(0.0),
                relative_tolerance=rank_tolerance,
            ).memory_coordinates
        )
    single_drives = np.asarray(
        [single_protocol.difference(float(value)) for value in single_times]
    )
    double_drives = np.asarray(
        [double_protocol.difference(float(value)) for value in double_times]
    )
    scenarios = (
        FiniteHorizonScenario(
            label="single_central",
            times=single_times,
            closed_coordinates=single_closed,
            drive_values=single_drives,
            initial_memory_coordinates=initial_memory[0],
        ),
        *tuple(
            FiniteHorizonScenario(
                label=f"double_{member}",
                times=double_times,
                closed_coordinates=double_closed[index],
                drive_values=double_drives,
                initial_memory_coordinates=initial_memory[index],
            )
            for index, member in enumerate(MEMBERS)
        ),
    )

    mandatory_dimension = envelope.layer_dimensions[0]
    print(
        "[moving-frame] constructing split Gramians "
        f"(hidden={frame.hidden_dimension}, mandatory={mandatory_dimension})",
        flush=True,
    )
    finite_horizon = finite_horizon_reachable_observable_audit(
        frame,
        scenarios,
        coordinate_scales,
        split_times=split_times,
        mandatory_dimension=mandatory_dimension,
        relative_tolerance=rank_tolerance,
        preparation_weight=1.0,
    )
    print("[moving-frame] auditing local order, geometry, and transport", flush=True)
    audit = moving_frame_viability_audit(
        frame,
        finite_horizon,
        scenarios,
        coordinate_scales,
        archive_field,
    )

    split_count = len(finite_horizon.split_audits)
    hidden_dimension = frame.hidden_dimension
    padded_values = np.zeros((split_count, hidden_dimension), dtype=float)
    padded_primal = np.zeros(
        (split_count, hidden_dimension, hidden_dimension),
        dtype=float,
    )
    padded_dual = np.zeros_like(padded_primal)
    supported_counts = np.empty(split_count, dtype=int)
    for index, split in enumerate(finite_horizon.split_audits):
        count = split.hankel_singular_values.size
        supported_counts[index] = count
        padded_values[index, :count] = split.hankel_singular_values
        padded_primal[index, :, :count] = split.primal_directions
        padded_dual[index, :, :count] = split.dual_directions

    metric_names = tuple(asdict(audit.orders[0]).keys()) if audit.orders else ()
    numeric_metrics = {
        name: np.asarray([getattr(order, name) for order in audit.orders])
        for name in metric_names
    }
    np.savez_compressed(
        output_directory / "moving_frame_viability_audit.npz",
        coordinate_scales=coordinate_scales,
        scenario_labels=np.asarray([scenario.label for scenario in scenarios]),
        split_scenario_labels=np.asarray(
            [split.scenario_label for split in finite_horizon.split_audits]
        ),
        split_times=np.asarray(
            [split.split_time for split in finite_horizon.split_audits]
        ),
        split_supported_counts=supported_counts,
        split_hankel_singular_values=padded_values,
        split_primal_directions=padded_primal,
        split_dual_directions=padded_dual,
        split_reachability_gramians=np.asarray(
            [split.reachability_gramian for split in finite_horizon.split_audits]
        ),
        split_observability_gramians=np.asarray(
            [split.observability_gramian for split in finite_horizon.split_audits]
        ),
        **numeric_metrics,
    )

    order_records = [_order_record(order) for order in audit.orders]
    summary: dict[str, object] = {
        "schema": "paper_v_archive_auxiliary_moving_frame_viability_summary_v1",
        "run_id": output_directory.name,
        "classification": "offline_development_construction_diagnostic",
        "evidence_status": "exploratory_local_not_promoted",
        "status": "complete",
        "hidden_dimension": frame.hidden_dimension,
        "mandatory_entrance_dimension": mandatory_dimension,
        "word_layer_dimensions": list(envelope.layer_dimensions),
        "word_cumulative_dimensions": list(envelope.cumulative_dimensions),
        "scenario_count": len(scenarios),
        "split_count": split_count,
        "first_full_pair_count": audit.first_full_pair_count,
        "first_orders_by_actual_capture_residual": {
            f"at_most_{threshold:g}": _first_order_below(audit.orders, threshold)
            for threshold in (0.1, 0.05, 0.01)
        },
        "orders": order_records,
        "initial_contraction_max_abs_residual": contraction_residual,
        "path_initial_max_abs_residual": path_initial_residual,
        "interpretation": (
            "This is a moving-frame viability audit, not an autonomous model "
            "score. Local reciprocal frames are judged by actual Gramian "
            "capture, subspace geometry, archive-section compatibility, "
            "resolved-input leakage, and the normal transport defect "
            "(I-P)(A U-Udot). No scientific dimension cap or exact online "
            "trajectory input was used."
        ),
        "autonomous_rollout_executed": False,
        "online_exact_reference_used": False,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(output_directory / "summary.json", summary)

    repo_root = Path(__file__).resolve().parents[2]
    source_paths = (
        Path(__file__).resolve(),
        repo_root / "paper_5/src/paper5/stability/moving_frame_viability.py",
        repo_root / "paper_5/src/paper5/stability/finite_horizon_auxiliary.py",
        repo_root / "paper_5/src/paper5/stability/archive_auxiliary_memory.py",
        repo_root / "paper_5/src/paper5/stability/reachability_observability.py",
    )
    artifact_paths = (
        output_directory / "plan.json",
        output_directory / "summary.json",
        output_directory / "moving_frame_viability_audit.npz",
    )
    manifest: dict[str, object] = {
        "schema": "paper_v_archive_auxiliary_moving_frame_viability_manifest_v1",
        "run_id": output_directory.name,
        "status": "complete",
        "classification": "offline_development_construction_diagnostic",
        "evidence_status": "exploratory_local_not_promoted",
        "python": sys.version,
        "platform": platform.platform(),
        "input_hashes": {
            str(path): _sha256(path)
            for path in (single_path, double_path, preparations_path)
        },
        "source_hashes": {
            str(path.relative_to(repo_root)): _sha256(path)
            for path in source_paths
        },
        "artifact_hashes": {
            path.name: _sha256(path) for path in artifact_paths
        },
    }
    _write_json(output_directory / "runtime_manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return summary


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(","))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-reference", type=Path, default=DEFAULT_SINGLE)
    parser.add_argument("--double-reference", type=Path, default=DEFAULT_DOUBLE)
    parser.add_argument("--preparations", type=Path, default=DEFAULT_PREPARATIONS)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--final-time", type=float, default=20.0)
    parser.add_argument("--maximum-word-depth", type=int, default=2)
    parser.add_argument(
        "--split-times",
        type=_parse_floats,
        default=(0.5, 2.0, 4.0, 8.5, 10.0, 12.0, 16.0, 19.0),
    )
    parser.add_argument("--rank-tolerance", type=float, default=1e-9)
    arguments = parser.parse_args()
    run(
        arguments.single_reference,
        arguments.double_reference,
        arguments.preparations,
        arguments.output_directory,
        final_time=arguments.final_time,
        maximum_word_depth=arguments.maximum_word_depth,
        split_times=arguments.split_times,
        rank_tolerance=arguments.rank_tolerance,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
