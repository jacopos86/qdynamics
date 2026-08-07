#!/usr/bin/env python3
"""Construct the first finite-horizon reachable--observable Route-4 audit."""

from __future__ import annotations

import argparse
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
    _ground_state,
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
from paper5.stability.multi_coherent_scores import development_coordinate_scales
from paper5.stability.reachability_observability import (
    build_drive_aware_word_envelope,
)

RUN_ID = "paper_v_archive_auxiliary_finite_horizon_cutoff16_t4_20260805_v1"
DEFAULT_EXACT = Path(
    "output/local_runs/"
    "paper_v_exact_vs_31d_cutoff_convergence_t20_local_20260801_v1/"
    "trajectories_cutoff_16.npz"
)
DEFAULT_OUTPUT = Path("output/local_runs") / RUN_ID


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
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _supported_ranks(values: np.ndarray) -> dict[str, int]:
    if values.size == 0 or values[0] == 0.0:
        return {f"relative_{threshold:.0e}": 0 for threshold in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10)}
    relative = values / values[0]
    return {
        f"relative_{threshold:.0e}": int(np.count_nonzero(relative >= threshold))
        for threshold in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10)
    }


def _largest_resolved_gaps(values: np.ndarray, count: int = 10) -> list[dict[str, float | int]]:
    if values.size < 2 or values[0] == 0.0:
        return []
    relative = values / values[0]
    eligible = np.flatnonzero(relative[1:] >= 1e-10)
    records = []
    for index in eligible:
        ratio = float(values[index] / values[index + 1])
        records.append(
            {
                "pair_count_before_gap": int(index + 1),
                "gap_ratio": ratio,
                "left_relative_value": float(relative[index]),
                "right_relative_value": float(relative[index + 1]),
            }
        )
    return sorted(records, key=lambda item: float(item["gap_ratio"]), reverse=True)[:count]


def run(
    exact_path: Path,
    output_directory: Path,
    *,
    final_time: float = 4.0,
    maximum_word_depth: int = 2,
    split_times: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5),
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
    protocol = GaussianSineDrive.from_parameters(parameters)
    plan: dict[str, object] = {
        "schema": "paper_v_archive_auxiliary_finite_horizon_plan_v1",
        "run_id": output_directory.name,
        "classification": "offline_development_construction_diagnostic",
        "evidence_status": "exploratory_local_not_promoted",
        "exact_development_input": str(exact_path),
        "phonon_cutoff": 16,
        "lambda_ep": parameters.lambda_ep,
        "gamma": parameters.gamma,
        "drive_protocol": {
            "amplitude": protocol.amplitude,
            "pulse_width": protocol.pulse_width,
            "delays": list(protocol.delays),
        },
        "final_time": final_time,
        "maximum_word_depth": maximum_word_depth,
        "split_times": list(split_times),
        "rank_tolerance": rank_tolerance,
        "preparation_weight": 1.0,
        "output_metric": "31_coordinate_development_scales",
        "autonomous_rollout_executed": False,
        "online_exact_reference_used": False,
    }
    _write_json(output_directory / "plan.json", plan)
    started = time.time()

    with np.load(exact_path, allow_pickle=False) as payload:
        all_times = np.asarray(payload["times"], dtype=float)
        selected = all_times <= final_time + 1e-12
        times = all_times[selected]
        exact_coordinates = np.asarray(
            payload["exact_coordinates"],
            dtype=float,
        )[selected]
    if not np.isclose(times[-1], final_time, atol=1e-12):
        raise ValueError("development trajectory does not sample final_time")
    coordinate_scales = development_coordinate_scales(
        exact_coordinates,
        phonon_cutoff=16,
    )

    model = _build_exact_dimer_model(parameters, phonon_cutoff=16)
    _, ground_state = _ground_state(model, eigensolver_tolerance=1e-12)
    initial_closed = matrix_state_to_closed_scalar_coordinates(
        _contract_matrix_state(model, ground_state)
    )
    if np.max(np.abs(initial_closed - exact_coordinates[0])) > 2e-10:
        raise RuntimeError("ground-state contraction differs from development input")

    envelope = build_drive_aware_word_envelope(
        parameters,
        phonon_cutoff=16,
        maximum_word_depth=maximum_word_depth,
        rank_tolerance=rank_tolerance,
        preparation_state_vectors=(ground_state,),
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

    initial = frame.initialize_memory(
        initial_closed,
        ground_state,
        archive_field,
        drive_value=protocol.difference(0.0),
        relative_tolerance=rank_tolerance,
    )
    scenario = FiniteHorizonScenario(
        label="exact_correlated_single_pulse_development",
        times=times,
        closed_coordinates=exact_coordinates,
        drive_values=np.asarray(
            [protocol.difference(float(value)) for value in times],
            dtype=float,
        ),
        initial_memory_coordinates=initial.memory_coordinates,
    )
    audit = finite_horizon_reachable_observable_audit(
        frame,
        (scenario,),
        coordinate_scales,
        split_times=split_times,
        mandatory_dimension=envelope.layer_dimensions[0],
        relative_tolerance=rank_tolerance,
        preparation_weight=1.0,
    )

    pair_tail_defects = np.asarray(
        [
            audit.worst_optimal_relative_defect(pair_count)
            for pair_count in range(audit.supported_pair_count + 1)
        ],
        dtype=float,
    )
    actual_orders = audit.actual_order_curve()
    aggregate_values = audit.aggregate_hankel_singular_values
    padded_split_values = np.zeros(
        (len(audit.split_audits), frame.hidden_dimension),
        dtype=float,
    )
    padded_split_primal = np.zeros(
        (
            len(audit.split_audits),
            frame.hidden_dimension,
            frame.hidden_dimension,
        ),
        dtype=float,
    )
    padded_split_dual = np.zeros_like(padded_split_primal)
    split_supported_counts = np.empty(len(audit.split_audits), dtype=int)
    for index, split in enumerate(audit.split_audits):
        count = split.hankel_singular_values.size
        padded_split_values[index, :count] = split.hankel_singular_values
        padded_split_primal[index, :, :count] = split.primal_directions
        padded_split_dual[index, :, :count] = split.dual_directions
        split_supported_counts[index] = count

    np.savez_compressed(
        output_directory / "finite_horizon_auxiliary_audit.npz",
        times=times,
        coordinate_scales=coordinate_scales,
        split_times=np.asarray(split_times, dtype=float),
        split_hankel_singular_values=padded_split_values,
        split_primal_directions=padded_split_primal,
        split_dual_directions=padded_split_dual,
        split_reachability_gramians=np.asarray(
            [split.reachability_gramian for split in audit.split_audits]
        ),
        split_observability_gramians=np.asarray(
            [split.observability_gramian for split in audit.split_audits]
        ),
        split_supported_counts=split_supported_counts,
        aggregate_hankel_singular_values=aggregate_values,
        aggregate_primal_directions=audit.aggregate_primal_directions,
        aggregate_dual_directions=audit.aggregate_dual_directions,
        worst_optimal_relative_defect_by_pair_count=pair_tail_defects,
        actual_order_curve=actual_orders,
    )

    split_summaries = [
        {
            "split_time": split.split_time,
            "reachability_rank": split.reachability_rank,
            "observability_rank": split.observability_rank,
            "supported_hankel_rank": int(split.hankel_singular_values.size),
            "relative_ranks": _supported_ranks(split.hankel_singular_values),
        }
        for split in audit.split_audits
    ]
    summary: dict[str, object] = {
        "schema": "paper_v_archive_auxiliary_finite_horizon_summary_v1",
        "run_id": output_directory.name,
        "classification": "offline_development_construction_diagnostic",
        "evidence_status": "exploratory_local_not_promoted",
        "status": "complete",
        "hidden_dimension": frame.hidden_dimension,
        "mandatory_entrance_dimension": envelope.layer_dimensions[0],
        "word_layer_dimensions": list(envelope.layer_dimensions),
        "word_cumulative_dimensions": list(envelope.cumulative_dimensions),
        "aggregate_supported_hankel_rank": audit.supported_pair_count,
        "aggregate_relative_ranks": _supported_ranks(aggregate_values),
        "largest_resolved_gaps": _largest_resolved_gaps(aggregate_values),
        "actual_order_at_all_supported_pairs": int(actual_orders[-1]),
        "worst_ideal_defect_at_zero_balanced_pairs": float(
            pair_tail_defects[0]
        ),
        "split_audits": split_summaries,
        "interpretation": (
            "This audit measures finite-horizon reachable-observable structure "
            "on one declared development path. It proposes reciprocal "
            "orthogonal frames but does not score an autonomous model or "
            "constitute held-out validation."
        ),
        "autonomous_rollout_executed": False,
        "online_exact_reference_used": False,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(output_directory / "summary.json", summary)

    repo_root = Path(__file__).resolve().parents[2]
    source_paths = (
        Path(__file__).resolve(),
        repo_root / "paper_5/src/paper5/stability/finite_horizon_auxiliary.py",
        repo_root / "paper_5/src/paper5/stability/archive_auxiliary_memory.py",
        repo_root / "paper_5/src/paper5/stability/reachability_observability.py",
    )
    artifact_paths = (
        output_directory / "plan.json",
        output_directory / "summary.json",
        output_directory / "finite_horizon_auxiliary_audit.npz",
    )
    manifest: dict[str, object] = {
        "schema": "paper_v_archive_auxiliary_finite_horizon_manifest_v1",
        "run_id": output_directory.name,
        "status": "complete",
        "classification": "offline_development_construction_diagnostic",
        "evidence_status": "exploratory_local_not_promoted",
        "python": sys.version,
        "platform": platform.platform(),
        "input_hash": _sha256(exact_path),
        "source_hashes": {
            str(path.relative_to(repo_root)): _sha256(path)
            for path in source_paths
        },
        "artifact_hashes": {
            path.name: _sha256(path) for path in artifact_paths
        },
    }
    _write_json(output_directory / "runtime_manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(","))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-reference", type=Path, default=DEFAULT_EXACT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--final-time", type=float, default=4.0)
    parser.add_argument("--maximum-word-depth", type=int, default=2)
    parser.add_argument(
        "--split-times",
        type=_parse_floats,
        default=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5),
    )
    parser.add_argument("--rank-tolerance", type=float, default=1e-9)
    arguments = parser.parse_args()
    run(
        arguments.exact_reference,
        arguments.output_directory,
        final_time=arguments.final_time,
        maximum_word_depth=arguments.maximum_word_depth,
        split_times=arguments.split_times,
        rank_tolerance=arguments.rank_tolerance,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
