"""Run a source-locked exact-versus-closure Holstein-dimer diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import scipy

from .exact_reference import (
    CLOSED_PROTOCOLS,
    ClosedProtocol,
    ExactClosedProtocolComparison,
    compare_exact_and_closed_protocols,
)
from .hubbard_dimer import DimerParameters
from .initial_conditions import (
    closed_boson_moment_eigenvalues,
    closed_electron_eigenvalues,
    closed_phonon_eigenvalues,
)
from .matrix_reference import (
    closed_scalar_to_matrix_state,
    matrix_total_energy,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_progress(path: Path, payload: dict[str, Any]) -> None:
    record = {"recorded_at_utc": _utc_now(), **payload}
    encoded = json.dumps(record, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(encoded, flush=True)


def _matrix_block_errors(
    reference_coordinates: np.ndarray,
    candidate_coordinates: np.ndarray,
) -> dict[str, np.ndarray]:
    if reference_coordinates.shape != candidate_coordinates.shape:
        raise ValueError("coordinate trajectories must have matching shapes")

    block_names = (
        "electron_density",
        "coherent_phonon",
        "normal_phonon",
        "anomalous_phonon",
        "electron_phonon_correlation",
    )
    result = {
        name: np.empty(reference_coordinates.shape[0], dtype=float)
        for name in block_names
    }
    for index, (reference_row, candidate_row) in enumerate(
        zip(reference_coordinates, candidate_coordinates, strict=True)
    ):
        reference = closed_scalar_to_matrix_state(reference_row)
        candidate = closed_scalar_to_matrix_state(candidate_row)
        result["electron_density"][index] = np.linalg.norm(
            candidate.electron_density - reference.electron_density
        )
        result["coherent_phonon"][index] = np.linalg.norm(
            candidate.coherent_phonon - reference.coherent_phonon
        )
        result["normal_phonon"][index] = np.linalg.norm(
            candidate.phonon_density - reference.phonon_density
        )
        result["anomalous_phonon"][index] = np.linalg.norm(
            candidate.anomalous_phonon_density
            - reference.anomalous_phonon_density
        )
        result["electron_phonon_correlation"][index] = np.linalg.norm(
            candidate.electron_phonon_correlation
            - reference.electron_phonon_correlation
        )
    return result


def _trajectory_physicality(coordinates: np.ndarray) -> dict[str, float]:
    electron_eigenvalues = np.asarray(
        [closed_electron_eigenvalues(row) for row in coordinates],
        dtype=float,
    )
    phonon_eigenvalues = np.asarray(
        [closed_phonon_eigenvalues(row) for row in coordinates],
        dtype=float,
    )
    boson_minima = np.asarray(
        [closed_boson_moment_eigenvalues(row)[0] for row in coordinates],
        dtype=float,
    )
    return {
        "minimum_electron_eigenvalue": float(
            np.min(electron_eigenvalues[:, 0])
        ),
        "maximum_electron_eigenvalue": float(
            np.max(electron_eigenvalues[:, -1])
        ),
        "minimum_normal_phonon_eigenvalue": float(
            np.min(phonon_eigenvalues[:, 0])
        ),
        "minimum_boson_moment_eigenvalue": float(np.min(boson_minima)),
        "maximum_absolute_coordinate": float(np.max(np.abs(coordinates))),
    }


def _static_energies(
    coordinates: np.ndarray,
    parameters: DimerParameters,
) -> np.ndarray:
    return np.asarray(
        [
            matrix_total_energy(
                closed_scalar_to_matrix_state(row),
                parameters,
            )
            for row in coordinates
        ],
        dtype=float,
    )


def _protocol_metrics(
    comparison: ExactClosedProtocolComparison,
    parameters: DimerParameters,
) -> dict[str, Any]:
    coordinate_l2_errors = np.linalg.norm(
        comparison.coordinate_errors,
        axis=1,
    )
    matrix_errors = _matrix_block_errors(
        comparison.exact_coordinates,
        comparison.closed_coordinates,
    )
    exact_energies = _static_energies(
        comparison.exact_coordinates,
        parameters,
    )
    closed_energies = _static_energies(
        comparison.closed_coordinates,
        parameters,
    )
    energy_errors = closed_energies - exact_energies
    return {
        "closed_function_evaluations": (
            comparison.closed_function_evaluations
        ),
        "maximum_absolute_coordinate_error": float(
            np.max(np.abs(comparison.coordinate_errors))
        ),
        "maximum_coordinate_l2_error": float(
            np.max(coordinate_l2_errors)
        ),
        "final_coordinate_l2_error": float(coordinate_l2_errors[-1]),
        "maximum_static_energy_error": float(np.max(np.abs(energy_errors))),
        "final_static_energy_error": float(energy_errors[-1]),
        "maximum_block_frobenius_error": {
            name: float(np.max(values))
            for name, values in matrix_errors.items()
        },
        "final_block_frobenius_error": {
            name: float(values[-1])
            for name, values in matrix_errors.items()
        },
        "physicality": _trajectory_physicality(
            comparison.closed_coordinates
        ),
    }


def _cutoff_pair_metrics(
    lower_coordinates: np.ndarray,
    upper_coordinates: np.ndarray,
) -> dict[str, Any]:
    differences = lower_coordinates - upper_coordinates
    coordinate_l2 = np.linalg.norm(differences, axis=1)
    matrix_errors = _matrix_block_errors(
        upper_coordinates,
        lower_coordinates,
    )
    return {
        "maximum_absolute_coordinate_difference": float(
            np.max(np.abs(differences))
        ),
        "maximum_coordinate_l2_difference": float(np.max(coordinate_l2)),
        "final_coordinate_l2_difference": float(coordinate_l2[-1]),
        "maximum_block_frobenius_difference": {
            name: float(np.max(values))
            for name, values in matrix_errors.items()
        },
        "final_block_frobenius_difference": {
            name: float(values[-1])
            for name, values in matrix_errors.items()
        },
    }


def _sample_times(integration: dict[str, Any]) -> np.ndarray:
    initial_time = float(integration["initial_time"])
    final_time = float(integration["final_time"])
    sample_step = float(integration["sample_step"])
    if abs(initial_time) > 1e-15:
        raise ValueError("the exact comparison currently requires t0=0")
    if final_time <= initial_time or sample_step <= 0.0:
        raise ValueError("final_time and sample_step must be positive")
    intervals = int(round((final_time - initial_time) / sample_step))
    if not np.isclose(
        initial_time + intervals * sample_step,
        final_time,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("sample_step must divide the integration interval")
    return np.linspace(initial_time, final_time, intervals + 1)


def _validate_run_contract(
    run_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    plan_path = run_directory / "plan.json"
    authorization_path = run_directory / "authorization.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    authorization = json.loads(
        authorization_path.read_text(encoding="utf-8")
    )
    if plan["execution_authorized"]:
        raise RuntimeError("the immutable plan must remain unauthorized")
    if plan["classification"] != "diagnostic":
        raise RuntimeError("this runner accepts diagnostic plans only")
    if plan["evidence_status"] != "exploratory_local_not_promoted":
        raise RuntimeError("diagnostic output must remain unpromoted")
    if not authorization["authorized"]:
        raise RuntimeError("current user authorization is required")
    if authorization["run_id"] != plan["run_id"]:
        raise RuntimeError("authorization and plan run IDs do not match")

    repository_root = Path(__file__).resolve().parents[4]
    for relative_path, expected_hash in plan["source_hashes"].items():
        source_path = repository_root / relative_path
        actual_hash = _sha256(source_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"source hash mismatch for {relative_path}: "
                f"{actual_hash} != {expected_hash}"
            )
    return plan, authorization, repository_root


def run_diagnostic(run_directory: Path) -> dict[str, Any]:
    """Execute one immutable exact-versus-closure diagnostic plan."""

    run_directory = run_directory.resolve()
    plan, _authorization, _repository_root = _validate_run_contract(
        run_directory
    )
    runtime_manifest_path = run_directory / "runtime_manifest.json"
    progress_path = run_directory / "progress.jsonl"
    summary_path = run_directory / "summary.json"
    partial_summary_path = run_directory / "summary.partial.json"
    if runtime_manifest_path.exists() or summary_path.exists():
        raise RuntimeError(
            "this immutable run directory already contains runtime output"
        )

    started_at = _utc_now()
    started_clock = time.perf_counter()
    runtime_manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": plan["run_id"],
        "classification": plan["classification"],
        "evidence_status": plan["evidence_status"],
        "status": "running",
        "started_at_utc": started_at,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "plan_sha256": _sha256(run_directory / "plan.json"),
        "authorization_sha256": _sha256(
            run_directory / "authorization.json"
        ),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "source_hashes": plan["source_hashes"],
    }
    parameters = DimerParameters(**plan["parameters"])
    integration = plan["integration"]
    correction = plan["correction"]
    times = _sample_times(integration)
    protocols = tuple(plan["protocols"])
    if any(protocol not in CLOSED_PROTOCOLS for protocol in protocols):
        raise ValueError(f"protocols must be selected from {CLOSED_PROTOCOLS}")
    typed_protocols = cast(tuple[ClosedProtocol, ...], tuple(protocols))
    cutoffs = tuple(int(value) for value in plan["cutoff_execution_order"])

    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": plan["run_id"],
        "status": "running",
        "scientific_question": plan["scientific_question"],
        "times": {
            "initial": float(times[0]),
            "final": float(times[-1]),
            "sample_count": int(times.size),
        },
        "cutoffs": {},
        "cutoff_convergence": {},
    }
    exact_by_cutoff: dict[int, np.ndarray] = {}
    _write_json_atomic(runtime_manifest_path, runtime_manifest)

    try:
        for cutoff in cutoffs:
            cutoff_clock = time.perf_counter()
            _append_progress(
                progress_path,
                {
                    "event": "cutoff_started",
                    "phonon_cutoff": cutoff,
                    "protocols": protocols,
                },
            )
            comparisons = compare_exact_and_closed_protocols(
                parameters,
                sample_times=times,
                phonon_cutoff=cutoff,
                protocols=typed_protocols,
                eigensolver_tolerance=float(
                    integration["eigensolver_tolerance"]
                ),
                relative_tolerance=float(
                    integration["relative_tolerance"]
                ),
                absolute_tolerance=float(
                    integration["absolute_tolerance"]
                ),
                maximum_step=float(integration["maximum_step"]),
                activation_margin=float(correction["activation_margin"]),
                target_flux=float(correction["target_flux"]),
                barrier_rate=float(correction["barrier_rate"]),
                energy_neutral=bool(correction["energy_neutral"]),
                require_correction_convergence=bool(
                    correction["require_convergence"]
                ),
            )
            first = comparisons[typed_protocols[0]]
            exact_coordinates = first.exact_coordinates
            exact_by_cutoff[cutoff] = exact_coordinates
            exact_energies = _static_energies(exact_coordinates, parameters)
            cutoff_summary = {
                "exact": {
                    "phonon_cutoff": cutoff,
                    "hilbert_space_dimension": int(
                        4 * (cutoff + 1) ** 2
                    ),
                    "function_evaluations": (
                        first.exact_trajectory.function_evaluations
                    ),
                    "maximum_state_norm_defect": float(
                        np.max(
                            np.abs(first.exact_trajectory.state_norms - 1.0)
                        )
                    ),
                    "initial_static_energy": float(exact_energies[0]),
                    "final_static_energy": float(exact_energies[-1]),
                    "physicality": _trajectory_physicality(
                        exact_coordinates
                    ),
                },
                "protocols": {
                    protocol: _protocol_metrics(comparison, parameters)
                    for protocol, comparison in comparisons.items()
                },
                "wall_elapsed_seconds": float(
                    time.perf_counter() - cutoff_clock
                ),
            }
            summary["cutoffs"][str(cutoff)] = cutoff_summary

            arrays = {
                "times": times,
                "exact_coordinates": exact_coordinates,
                "exact_state_norms": first.exact_trajectory.state_norms,
            }
            arrays.update(
                {
                    f"closed_coordinates__{protocol}": (
                        comparison.closed_coordinates
                    )
                    for protocol, comparison in comparisons.items()
                }
            )
            _write_npz_atomic(
                run_directory / f"trajectories_cutoff_{cutoff}.npz",
                arrays,
            )
            _write_json_atomic(partial_summary_path, summary)
            _append_progress(
                progress_path,
                {
                    "event": "cutoff_completed",
                    "phonon_cutoff": cutoff,
                    "wall_elapsed_seconds": cutoff_summary[
                        "wall_elapsed_seconds"
                    ],
                    "maximum_coordinate_errors": {
                        protocol: metrics[
                            "maximum_absolute_coordinate_error"
                        ]
                        for protocol, metrics in cutoff_summary[
                            "protocols"
                        ].items()
                    },
                },
            )

        sorted_cutoffs = tuple(sorted(cutoffs))
        adjacent_pairs = tuple(zip(sorted_cutoffs[:-1], sorted_cutoffs[1:]))
        summary["cutoff_convergence"] = {
            f"{lower}_vs_{upper}": _cutoff_pair_metrics(
                exact_by_cutoff[lower],
                exact_by_cutoff[upper],
            )
            for lower, upper in adjacent_pairs
        }
        summary["status"] = "complete"
        summary["wall_elapsed_seconds"] = float(
            time.perf_counter() - started_clock
        )
        _write_json_atomic(summary_path, summary)
        _write_json_atomic(partial_summary_path, summary)

        _append_progress(
            progress_path,
            {
                "event": "run_completed",
                "wall_elapsed_seconds": summary["wall_elapsed_seconds"],
            },
        )

        artifact_paths = (
            summary_path,
            partial_summary_path,
            progress_path,
            *(
                run_directory / f"trajectories_cutoff_{cutoff}.npz"
                for cutoff in cutoffs
            ),
        )
        runtime_manifest.update(
            {
                "status": "complete",
                "finished_at_utc": _utc_now(),
                "wall_elapsed_seconds": summary["wall_elapsed_seconds"],
                "artifact_hashes": {
                    path.name: _sha256(path) for path in artifact_paths
                },
            }
        )
        _write_json_atomic(runtime_manifest_path, runtime_manifest)
        return summary
    except BaseException as error:
        runtime_manifest.update(
            {
                "status": (
                    "interrupted"
                    if isinstance(error, KeyboardInterrupt)
                    else "failed"
                ),
                "finished_at_utc": _utc_now(),
                "wall_elapsed_seconds": float(
                    time.perf_counter() - started_clock
                ),
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
        )
        _write_json_atomic(runtime_manifest_path, runtime_manifest)
        _append_progress(
            progress_path,
            {
                "event": runtime_manifest["status"],
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            },
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-directory",
        type=Path,
        required=True,
        help="Directory containing immutable plan.json and authorization.json.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    run_diagnostic(args.run_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
