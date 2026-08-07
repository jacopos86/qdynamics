"""Reproduce the exact Eq. (14d) missing-moment attribution.

The command compares the archive correlation velocity with exact Holstein-
dimer contractions and separates three model terms: the connected mixed moment
``K``, the fixed-sector Pauli-algebra repair, and the opposite-spin covariance.
Exact data remain reporting-only and are never queried by an autonomous ODE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .exact_reference import (
    ExactCorrelationClosureTrajectory,
    exact_holstein_correlation_closure_trajectory,
)
from .hubbard_dimer import DimerParameters


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_times(final_time: float, sample_step: float) -> np.ndarray:
    if final_time <= 0.0:
        raise ValueError("final_time must be positive")
    if sample_step <= 0.0:
        raise ValueError("sample_step must be positive")
    count = int(round(final_time / sample_step))
    if not np.isclose(count * sample_step, final_time, atol=1e-12):
        raise ValueError("final_time must be an integer multiple of sample_step")
    return np.linspace(0.0, final_time, count + 1)


def _correlation_coordinates(correlation: np.ndarray) -> np.ndarray:
    values = np.asarray(correlation, dtype=complex)
    if values.ndim != 4 or values.shape[1:] != (2, 2, 2):
        raise ValueError("correlation must have shape (samples, 2, 2, 2)")

    coordinates = np.empty((values.shape[0], 14), dtype=float)
    shared_trace = 0.5 * (
        np.trace(values[:, 0], axis1=1, axis2=2)
        + np.trace(values[:, 1], axis1=1, axis2=2)
    )
    coordinates[:, 0] = shared_trace.real
    coordinates[:, 1] = shared_trace.imag
    for q in range(2):
        offset = 2 + 6 * q
        diagonal_difference = values[:, q, 0, 0] - values[:, q, 1, 1]
        coordinates[:, offset : offset + 6] = np.column_stack(
            [
                diagonal_difference.real,
                diagonal_difference.imag,
                values[:, q, 0, 1].real,
                values[:, q, 0, 1].imag,
                values[:, q, 1, 0].real,
                values[:, q, 1, 0].imag,
            ]
        )
    return coordinates


def _residual_subtracted_coordinates(error: np.ndarray) -> np.ndarray:
    values = np.asarray(error, dtype=complex)
    return _correlation_coordinates(values - values[:1])


def _coordinate_metrics(coordinates: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(coordinates, axis=1)
    return {
        "component_rms": float(np.sqrt(np.mean(coordinates**2))),
        "vector_rms": float(np.sqrt(np.mean(norms**2))),
        "maximum_norm": float(np.max(norms)),
        "final_norm": float(norms[-1]),
    }


def _reduced_k_coordinates(remainder: np.ndarray) -> np.ndarray:
    """Project ``K`` onto the relative-mode four-real-coordinate block."""

    values = np.asarray(remainder, dtype=complex)
    relative = 0.25 * (
        values[:, 0, 0]
        + values[:, 0, 1]
        - values[:, 1, 0]
        - values[:, 1, 1]
    )
    return np.column_stack(
        [
            relative[:, 0, 1].real,
            relative[:, 0, 1].imag,
            relative[:, 1, 0].real,
            relative[:, 1, 0].imag,
        ]
    )


def _reduced_opposite_spin_coordinates(covariance: np.ndarray) -> np.ndarray:
    """Project the traceless Hermitian ``D[0] = -D[1]`` block to three reals."""

    values = np.asarray(covariance, dtype=complex)
    first_mode = 0.5 * (values[:, 0] - values[:, 1])
    return np.column_stack(
        [
            first_mode[:, 0, 0].real,
            first_mode[:, 0, 1].real,
            first_mode[:, 0, 1].imag,
        ]
    )


def _audit_arrays(
    audit: ExactCorrelationClosureTrajectory,
) -> dict[str, np.ndarray]:
    exact_derivatives = np.asarray(
        [
            derivative.electron_phonon_correlation
            for derivative in audit.exact_trajectory.matrix_derivatives
        ]
    )
    archive_error = audit.archive_correlation_derivatives - exact_derivatives
    errors = {
        "archive": archive_error,
        "archive_plus_k": (
            archive_error + audit.mixed_moment_velocity_corrections
        ),
        "archive_plus_k_plus_pauli": (
            archive_error
            + audit.mixed_moment_velocity_corrections
            + audit.same_spin_pauli_velocity_corrections
        ),
        "archive_plus_k_plus_pauli_plus_opposite_spin": (
            archive_error
            + audit.mixed_moment_velocity_corrections
            + audit.same_spin_pauli_velocity_corrections
            + audit.opposite_spin_velocity_corrections
        ),
    }
    arrays = {
        "times": audit.exact_trajectory.times,
        "exact_mixed_moment": audit.exact_mixed_moment,
        "factorized_mixed_moment": audit.factorized_mixed_moment,
        "mixed_moment_remainder": audit.mixed_moment_remainder,
        "same_spin_covariance": audit.same_spin_covariance,
        "archive_same_spin_covariance": (
            audit.archive_same_spin_covariance
        ),
        "opposite_spin_covariance": audit.opposite_spin_covariance,
        "mixed_moment_velocity_correction": (
            audit.mixed_moment_velocity_corrections
        ),
        "same_spin_pauli_velocity_correction": (
            audit.same_spin_pauli_velocity_corrections
        ),
        "opposite_spin_velocity_correction": (
            audit.opposite_spin_velocity_corrections
        ),
        "cutoff_velocity_remainder": audit.cutoff_velocity_remainders,
        "reduced_k_coordinates": _reduced_k_coordinates(
            audit.mixed_moment_remainder
        ),
        "reduced_opposite_spin_coordinates": (
            _reduced_opposite_spin_coordinates(
                audit.opposite_spin_covariance
            )
        ),
    }
    for name, error in errors.items():
        arrays[f"{name}_residual_subtracted_coordinates"] = (
            _residual_subtracted_coordinates(error)
        )
    return arrays


def _symmetry_metrics(
    audit: ExactCorrelationClosureTrajectory,
) -> dict[str, float]:
    remainder = audit.mixed_moment_remainder
    opposite = audit.opposite_spin_covariance
    return {
        "k_diagonal_max_abs": float(
            np.max(
                np.abs(
                    np.diagonal(remainder, axis1=-2, axis2=-1)
                )
            )
        ),
        "k_q_antisymmetry_max_abs": float(
            np.max(np.abs(remainder[:, 0] + remainder[:, 1]))
        ),
        "k_r_equality_max_abs": float(
            np.max(np.abs(remainder[:, :, 0] - remainder[:, :, 1]))
        ),
        "opposite_spin_antihermitian_max_abs": float(
            np.max(
                np.abs(opposite - opposite.conjugate().swapaxes(-1, -2))
            )
        ),
        "opposite_spin_trace_max_abs": float(
            np.max(np.abs(np.trace(opposite, axis1=-2, axis2=-1)))
        ),
        "opposite_spin_mode_sum_max_abs": float(
            np.max(np.abs(opposite[:, 0] + opposite[:, 1]))
        ),
    }


def _write_plot(
    path: Path,
    times: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> None:
    figure, axis = plt.subplots(figsize=(6.6, 3.8))
    labels = {
        "archive": "archive",
        "archive_plus_k": "+ K",
        "archive_plus_k_plus_pauli": "+ K + Pauli repair",
        "archive_plus_k_plus_pauli_plus_opposite_spin": (
            "+ K + Pauli + opposite spin"
        ),
    }
    for name, label in labels.items():
        coordinates = arrays[f"{name}_residual_subtracted_coordinates"]
        axis.plot(times, np.linalg.norm(coordinates, axis=1), label=label)
    axis.set_xlabel(r"dimensionless time $t\,t_{\rm hop}$")
    axis.set_ylabel(r"residual-subtracted $\|\Delta\dot C\|_2$")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def run_analysis(
    run_directory: Path,
    *,
    parameters: DimerParameters,
    final_time: float = 4.0,
    sample_step: float = 0.01,
    phonon_cutoff: int = 16,
    convergence_cutoffs: tuple[int, ...] = (12, 16, 20),
    convergence_time: float = 0.5,
    eigensolver_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-12,
    maximum_step: float = 0.01,
) -> dict[str, Any]:
    """Run the exact attribution and write retrievable exploratory artifacts."""

    if phonon_cutoff not in convergence_cutoffs:
        raise ValueError("convergence_cutoffs must include phonon_cutoff")
    run_directory.mkdir(parents=True, exist_ok=True)
    times = _sample_times(final_time, sample_step)
    audit = exact_holstein_correlation_closure_trajectory(
        parameters,
        sample_times=times,
        phonon_cutoff=phonon_cutoff,
        eigensolver_tolerance=eigensolver_tolerance,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step=maximum_step,
    )
    arrays = _audit_arrays(audit)

    variant_metrics = {
        name: _coordinate_metrics(
            arrays[f"{name}_residual_subtracted_coordinates"]
        )
        for name in (
            "archive",
            "archive_plus_k",
            "archive_plus_k_plus_pauli",
            "archive_plus_k_plus_pauli_plus_opposite_spin",
        )
    }
    convergence: dict[str, dict[str, float]] = {}
    for cutoff in convergence_cutoffs:
        if cutoff == phonon_cutoff:
            index = int(round(convergence_time / sample_step))
            short_audit = audit
            selected = slice(0, index + 1, index)
            cutoff_remainder = short_audit.mixed_moment_remainder[selected]
            cutoff_edge = short_audit.cutoff_velocity_remainders[selected]
        else:
            short_audit = exact_holstein_correlation_closure_trajectory(
                parameters,
                sample_times=np.array([0.0, convergence_time]),
                phonon_cutoff=cutoff,
                eigensolver_tolerance=eigensolver_tolerance,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                maximum_step=maximum_step,
            )
            cutoff_remainder = short_audit.mixed_moment_remainder
            cutoff_edge = short_audit.cutoff_velocity_remainders
        convergence[str(cutoff)] = {
            "k_q_antisymmetry_max_abs": float(
                np.max(np.abs(cutoff_remainder[:, 0] + cutoff_remainder[:, 1]))
            ),
            "k_r_equality_max_abs": float(
                np.max(
                    np.abs(
                        cutoff_remainder[:, :, 0]
                        - cutoff_remainder[:, :, 1]
                    )
                )
            ),
            "cutoff_velocity_change_frobenius": float(
                np.linalg.norm(cutoff_edge[-1] - cutoff_edge[0])
            ),
        }

    target_index = int(round(convergence_time / sample_step))
    source_changes = {
        "mixed_moment_k": float(
            np.linalg.norm(
                _correlation_coordinates(
                    audit.mixed_moment_velocity_corrections[
                        [target_index]
                    ]
                    - audit.mixed_moment_velocity_corrections[[0]]
                )[0]
            )
        ),
        "same_spin_pauli": float(
            np.linalg.norm(
                _correlation_coordinates(
                    audit.same_spin_pauli_velocity_corrections[
                        [target_index]
                    ]
                    - audit.same_spin_pauli_velocity_corrections[[0]]
                )[0]
            )
        ),
        "opposite_spin": float(
            np.linalg.norm(
                _correlation_coordinates(
                    audit.opposite_spin_velocity_corrections[
                        [target_index]
                    ]
                    - audit.opposite_spin_velocity_corrections[[0]]
                )[0]
            )
        ),
        "finite_cutoff": float(
            np.linalg.norm(
                _correlation_coordinates(
                    audit.cutoff_velocity_remainders[[target_index]]
                    - audit.cutoff_velocity_remainders[[0]]
                )[0]
            )
        ),
    }

    reduced_k = arrays["reduced_k_coordinates"]
    singular_values = np.linalg.svd(
        reduced_k - reduced_k[:1],
        full_matrices=False,
        compute_uv=False,
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "classification": "exploratory_local_not_promoted",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            **asdict(parameters),
            "coupling": parameters.coupling,
            "phonon_cutoff": phonon_cutoff,
            "final_time": final_time,
            "sample_step": sample_step,
            "maximum_step": maximum_step,
            "relative_tolerance": relative_tolerance,
            "absolute_tolerance": absolute_tolerance,
        },
        "variant_metrics": variant_metrics,
        "source_velocity_change_at_t_0_5": source_changes,
        "symmetry_metrics": _symmetry_metrics(audit),
        "cutoff_convergence": convergence,
        "reduced_coordinate_count": {
            "mixed_moment_k": 4,
            "sampled_opposite_spin_covariance": 3,
        },
        "reduced_k_singular_values": singular_values.tolist(),
        "closure_result": {
            "exact_terms_collapse_defect": True,
            "autonomous_k_plus_d_is_closed": False,
            "reason": (
                "The D equation rotates into the full spin-covariance tensor "
                "and mixed two-electron/one-phonon moments; the K equation "
                "introduces higher mixed phonon cumulants."
            ),
            "recommended_next_model": (
                "symmetry-adapted third-cumulant closure with the full set of "
                "same-order moments, followed by the joint Gram barrier"
            ),
        },
    }

    trajectory_path = run_directory / "mixed_moment_trajectory.npz"
    np.savez_compressed(trajectory_path, **arrays)
    plot_path = run_directory / "closure_defect_attribution.png"
    _write_plot(plot_path, times, arrays)
    summary_path = run_directory / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    source_paths = (
        Path(__file__),
        Path(__file__).with_name("exact_reference.py"),
        Path(__file__).with_name("matrix_reference.py"),
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "source_hashes": {
            str(path.resolve()): _sha256(path) for path in source_paths
        },
        "artifact_hashes": {
            path.name: _sha256(path)
            for path in (summary_path, trajectory_path, plot_path)
        },
    }
    manifest_path = run_directory / "runtime_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--lambda-ep", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--drive", type=float, default=1.0)
    parser.add_argument("--final-time", type=float, default=4.0)
    parser.add_argument("--sample-step", type=float, default=0.01)
    parser.add_argument("--phonon-cutoff", type=int, default=16)
    parser.add_argument("--maximum-step", type=float, default=0.01)
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = run_analysis(
        args.run_directory,
        parameters=DimerParameters(
            lambda_ep=args.lambda_ep,
            gamma=args.gamma,
            drive_amplitude=args.drive,
        ),
        final_time=args.final_time,
        sample_step=args.sample_step,
        phonon_cutoff=args.phonon_cutoff,
        maximum_step=args.maximum_step,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
