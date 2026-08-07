#!/usr/bin/env python3
"""Run the autonomous finite-horizon reciprocal-frame order curve."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from paper5.stability.archive_auxiliary_memory import (
    build_archive_auxiliary_frame_from_observables,
    propagate_archive_auxiliary_rk4,
)
from paper5.stability.exact_reference import (
    _build_exact_dimer_model,
    _contract_matrix_state,
    _ground_state,
)
from paper5.stability.finite_horizon_auxiliary import (
    FiniteHorizonAuxiliaryAudit,
)
from paper5.stability.hubbard_dimer import DimerParameters, GaussianSineDrive
from paper5.stability.matrix_reference import (
    matrix_state_to_closed_scalar_coordinates,
    pauli_repaired_closed_scalar_rhs,
)
from paper5.stability.reachability_observability import (
    build_drive_aware_word_envelope,
)
from analyze_archive_auxiliary_autonomous_pilot import (
    _FixedDriveParameters,
    _drive_rate,
    _physical_diagnostics,
    _score,
)

RUN_ID = (
    "paper_v_archive_auxiliary_finite_horizon_rollout_"
    "cutoff16_t4_20260805_v1"
)
DEFAULT_EXACT = Path(
    "output/local_runs/"
    "paper_v_exact_vs_31d_cutoff_convergence_t20_local_20260801_v1/"
    "trajectories_cutoff_16.npz"
)
DEFAULT_AUDIT = Path(
    "output/local_runs/"
    "paper_v_archive_auxiliary_finite_horizon_cutoff16_t4_20260805_v2/"
    "finite_horizon_auxiliary_audit.npz"
)
DEFAULT_OUTPUT = Path("output/local_runs") / RUN_ID


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


def _candidate_pair_counts(audit: FiniteHorizonAuxiliaryAudit) -> tuple[int, ...]:
    """Return every pair count that produces a new orthogonal order."""

    counts: list[int] = []
    previous_order = -1
    for pair_count in range(audit.supported_pair_count + 1):
        order = audit.orthogonal_frame(pair_count).shape[1]
        if order != previous_order:
            counts.append(pair_count)
            previous_order = order
    return tuple(counts)


def _load_audit(path: Path, mandatory_dimension: int) -> FiniteHorizonAuxiliaryAudit:
    with np.load(path, allow_pickle=False) as payload:
        values = np.asarray(
            payload["aggregate_hankel_singular_values"],
            dtype=float,
        )
        primal = np.asarray(payload["aggregate_primal_directions"], dtype=float)
        dual = np.asarray(payload["aggregate_dual_directions"], dtype=float)
    if primal.shape != dual.shape or primal.shape[1] != values.size:
        raise ValueError("finite-horizon audit arrays have incompatible shapes")
    return FiniteHorizonAuxiliaryAudit(
        hidden_dimension=primal.shape[0],
        mandatory_dimension=mandatory_dimension,
        relative_tolerance=1e-9,
        split_audits=(),
        aggregate_hankel_singular_values=values,
        aggregate_primal_directions=primal,
        aggregate_dual_directions=dual,
    )


def run(
    exact_path: Path,
    audit_path: Path,
    output_directory: Path,
    *,
    final_time: float = 4.0,
    time_step: float = 0.01,
    maximum_word_depth: int = 2,
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
        "schema": "paper_v_archive_auxiliary_finite_horizon_rollout_plan_v1",
        "run_id": output_directory.name,
        "classification": "autonomous_development_order_curve",
        "evidence_status": "exploratory_local_not_promoted",
        "exact_scoring_input": str(exact_path),
        "finite_horizon_construction_input": str(audit_path),
        "phonon_cutoff": 16,
        "lambda_ep": parameters.lambda_ep,
        "gamma": parameters.gamma,
        "final_time": final_time,
        "time_step": time_step,
        "maximum_word_depth": maximum_word_depth,
        "rank_tolerance": rank_tolerance,
        "candidate_rule": (
            "every pair count that produces a new numerical rank in the "
            "orthogonal mandatory-plus-primal-plus-dual union"
        ),
        "trial_test_frames_identical": True,
        "model_online_inputs": ["retained_coordinates", "memory", "V", "Vdot"],
        "online_exact_reference_used": False,
        "representability_controller_used": False,
    }
    _write_json(output_directory / "plan.json", plan)
    started = time.time()

    with np.load(exact_path, allow_pickle=False) as payload:
        all_times = np.asarray(payload["times"], dtype=float)
        selected = all_times <= final_time + 1e-12
        times = all_times[selected]
        exact = np.asarray(payload["exact_coordinates"], dtype=float)[selected]
        archive = np.asarray(
            payload["closed_coordinates__archive"],
            dtype=float,
        )[selected]
    if not np.isclose(times[-1], final_time, atol=1e-12):
        raise ValueError("exact artifact does not sample final_time")
    sample_step = float(times[1] - times[0])

    model = _build_exact_dimer_model(parameters, phonon_cutoff=16)
    _, ground_state = _ground_state(model, eigensolver_tolerance=1e-12)
    initial_closed = matrix_state_to_closed_scalar_coordinates(
        _contract_matrix_state(model, ground_state)
    )
    if np.max(np.abs(initial_closed - exact[0])) > 2e-10:
        raise RuntimeError("ground-state contraction differs from exact input")

    def archive_field(state: np.ndarray, drive_value: float) -> np.ndarray:
        return pauli_repaired_closed_scalar_rhs(
            0.0,
            state,
            _FixedDriveParameters(parameters, drive_value),  # type: ignore[arg-type]
        )

    pauli_solution = solve_ivp(
        lambda time_value, state: archive_field(
            state,
            protocol.difference(float(time_value)),
        ),
        (0.0, final_time),
        initial_closed,
        method="DOP853",
        t_eval=times,
        rtol=1e-10,
        atol=1e-12,
        max_step=time_step,
    )
    if not pauli_solution.success:
        raise RuntimeError(pauli_solution.message)
    pauli_archive = np.asarray(pauli_solution.y.T, dtype=float)

    envelope = build_drive_aware_word_envelope(
        parameters,
        phonon_cutoff=16,
        maximum_word_depth=maximum_word_depth,
        rank_tolerance=rank_tolerance,
        preparation_state_vectors=(ground_state,),
    )
    full_frame = build_archive_auxiliary_frame_from_observables(
        envelope.construction,
        envelope.hidden_observables,
    )
    audit = _load_audit(audit_path, envelope.layer_dimensions[0])
    if audit.hidden_dimension != full_frame.hidden_dimension:
        raise RuntimeError("audit and reconstructed hidden envelope differ")
    pair_counts = _candidate_pair_counts(audit)
    full_physical_hidden = full_frame.contract_hidden_state(ground_state)
    plan["sample_step"] = sample_step

    trajectories: dict[str, np.ndarray] = {
        "exact": exact,
        "archive": archive,
        "pauli_archive": pauli_archive,
    }
    auxiliary_diagnostics: dict[str, object] = {}
    candidate_records: list[dict[str, int | str]] = []
    for pair_count in pair_counts:
        basis = audit.orthogonal_frame(pair_count)
        frame = full_frame.orthogonal_projection(basis)
        initial = frame.initialize_memory_from_hidden(
            initial_closed,
            basis.T @ full_physical_hidden,
            archive_field,
            drive_value=protocol.difference(0.0),
            relative_tolerance=rank_tolerance,
        )
        trajectory = propagate_archive_auxiliary_rk4(
            frame,
            initial,
            archive_field,
            protocol.difference,
            lambda time_value: _drive_rate(protocol, time_value),
            final_time=final_time,
            time_step=time_step,
            sample_step=sample_step,
            relative_tolerance=rank_tolerance,
            directional_step=3e-6,
        )
        name = f"balanced_union_p{pair_count}_r{frame.hidden_dimension}"
        trajectories[name] = trajectory.closed_coordinates
        auxiliary_diagnostics[name] = {
            "pair_count": pair_count,
            "hidden_dimension": frame.hidden_dimension,
            "maximum_section_relative_residual": float(
                np.max(trajectory.centered_section_relative_residuals)
            ),
            "maximum_lossless_identity_residual": float(
                np.max(trajectory.projected_norm_identity_residuals)
            ),
            "maximum_physical_hidden_norm": float(
                np.max(trajectory.physical_hidden_norms)
            ),
        }
        candidate_records.append(
            {
                "name": name,
                "pair_count": pair_count,
                "hidden_dimension": frame.hidden_dimension,
            }
        )
        print(
            f"completed balanced union p={pair_count}, "
            f"r={frame.hidden_dimension}",
            flush=True,
        )

    with np.load(audit_path, allow_pickle=False) as payload:
        scales = np.asarray(payload["coordinate_scales"], dtype=float)
    diagnostics = {
        name: _physical_diagnostics(trajectory, parameters)
        for name, trajectory in trajectories.items()
    }
    scores = {
        name: _score(
            times,
            trajectory,
            exact,
            scales,
            diagnostics[name],
            diagnostics["exact"],
        )
        for name, trajectory in trajectories.items()
        if name != "exact"
    }
    candidate_names = [str(record["name"]) for record in candidate_records]
    best_coordinate = min(
        candidate_names,
        key=lambda name: float(scores[name]["coordinate_rms_error"]),
    )
    best_energy = min(
        candidate_names,
        key=lambda name: float(scores[name]["total_energy_rms_error"]),
    )

    plan["candidate_records"] = candidate_records
    _write_json(output_directory / "plan.json", plan)
    np.savez_compressed(
        output_directory / "finite_horizon_order_curve.npz",
        times=times,
        coordinate_scales=scales,
        **{f"coordinates__{name}": value for name, value in trajectories.items()},
        **{
            f"joint_minimum__{name}": value["joint_minimum"]
            for name, value in diagnostics.items()
        },
        **{
            f"energy__{name}": value["energy"]
            for name, value in diagnostics.items()
        },
    )
    summary: dict[str, object] = {
        "schema": "paper_v_archive_auxiliary_finite_horizon_rollout_summary_v1",
        "run_id": output_directory.name,
        "classification": "autonomous_development_order_curve",
        "evidence_status": "exploratory_local_not_promoted",
        "status": "complete",
        "candidate_records": candidate_records,
        "scores": scores,
        "auxiliary_diagnostics": auxiliary_diagnostics,
        "best_coordinate_model": best_coordinate,
        "best_energy_model": best_energy,
        "interpretation": (
            "Every distinct order in the finite-horizon orthogonal union was "
            "propagated autonomously. Exact data entered only after rollout "
            "for development scoring."
        ),
        "online_exact_reference_used": False,
        "representability_controller_used": False,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(output_directory / "summary.json", summary)

    repo_root = Path(__file__).resolve().parents[2]
    source_paths = (
        Path(__file__).resolve(),
        repo_root / "paper_5/src/paper5/stability/archive_auxiliary_memory.py",
        repo_root / "paper_5/src/paper5/stability/finite_horizon_auxiliary.py",
        repo_root / "paper_5/src/paper5/stability/reachability_observability.py",
    )
    artifact_paths = (
        output_directory / "plan.json",
        output_directory / "summary.json",
        output_directory / "finite_horizon_order_curve.npz",
    )
    manifest: dict[str, object] = {
        "schema": "paper_v_archive_auxiliary_finite_horizon_rollout_manifest_v1",
        "run_id": output_directory.name,
        "status": "complete",
        "classification": "autonomous_development_order_curve",
        "evidence_status": "exploratory_local_not_promoted",
        "python": sys.version,
        "platform": platform.platform(),
        "input_hashes": {
            str(exact_path): _sha256(exact_path),
            str(audit_path): _sha256(audit_path),
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
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-reference", type=Path, default=DEFAULT_EXACT)
    parser.add_argument("--finite-horizon-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--final-time", type=float, default=4.0)
    parser.add_argument("--time-step", type=float, default=0.01)
    parser.add_argument("--maximum-word-depth", type=int, default=2)
    parser.add_argument("--rank-tolerance", type=float, default=1e-9)
    arguments = parser.parse_args()
    run(
        arguments.exact_reference,
        arguments.finite_horizon_audit,
        arguments.output_directory,
        final_time=arguments.final_time,
        time_step=arguments.time_step,
        maximum_word_depth=arguments.maximum_word_depth,
        rank_tolerance=arguments.rank_tolerance,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
