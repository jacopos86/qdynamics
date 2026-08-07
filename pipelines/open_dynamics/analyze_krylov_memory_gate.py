"""Run the preregistered teacher-forced Gate B for the HS Krylov closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper5.stability import (
    DimerParameters,
    build_krylov_closure_construction,
    exact_holstein_wavefunction_trajectory_for_diagnostics,
    teacher_forced_krylov_gate,
)


RUN_ID = "paper_v_hs_krylov_gate_b_cutoff16_20260803_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _source_paths(repo_root: Path) -> tuple[Path, ...]:
    return (
        repo_root / "paper_5/src/paper5/stability/krylov_memory_closure.py",
        repo_root / "paper_5/src/paper5/stability/krylov_memory_analysis.py",
        repo_root / "paper_5/src/paper5/stability/exact_reference.py",
        repo_root / "paper_5/src/paper5/stability/matrix_reference.py",
        repo_root / "pipelines/open_dynamics/analyze_krylov_memory_gate.py",
    )


def _plot_gate(result, output_path: Path) -> None:
    times = result.times
    exact_source = (
        result.exact_closed_derivatives[:, 17:]
        - result.archive_derivatives[:, 17:]
    )
    figure, axes = plt.subplots(3, 1, figsize=(7.2, 8.2), sharex=True)

    archive_error = np.linalg.norm(
        result.archive_derivatives[:, 17:]
        - result.exact_closed_derivatives[:, 17:],
        axis=1,
    )
    axes[0].plot(times, archive_error, color="#202020", label="archive")
    colors = {2: "#2A6FBB", 3: "#D17C18", 4: "#8C4AA8"}
    for order, order_result in sorted(result.orders.items()):
        error = np.linalg.norm(
            order_result.modeled_derivatives[:, 17:]
            - result.exact_closed_derivatives[:, 17:],
            axis=1,
        )
        axes[0].plot(
            times,
            error,
            color=colors[order],
            label=f"Krylov order {order}",
        )
    axes[0].set_ylabel(r"$\|\dot C-\dot C_{\rm ex}\|_2$")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, ncol=2)

    exact_source_norm = np.linalg.norm(exact_source, axis=1)
    axes[1].plot(
        times,
        exact_source_norm,
        color="#202020",
        label="exact missing source",
    )
    for order, order_result in sorted(result.orders.items()):
        axes[1].plot(
            times,
            np.linalg.norm(order_result.modeled_missing_c_source, axis=1),
            color=colors[order],
            label=f"order {order}",
        )
    axes[1].set_ylabel("missing-source norm")
    axes[1].set_yscale("log")

    for order, order_result in sorted(result.orders.items()):
        axes[2].plot(
            times,
            order_result.total_residual_norms,
            color=colors[order],
            label=f"order {order}",
        )
    axes[2].axhline(0.0, color="#202020", linewidth=0.7)
    axes[2].set_ylabel("Galerkin residual norm")
    axes[2].set_xlabel(r"time ($t_{\rm hop}^{-1}$)")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def run(output_directory: Path, *, phonon_cutoff: int = 16) -> dict[str, Any]:
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty output directory {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    parameters = DimerParameters(
        hopping=1.0,
        gamma=0.5,
        lambda_ep=1.5,
        drive_amplitude=1.0,
        pulse_width=1.0,
    )
    times = np.linspace(0.0, 4.0, 401)
    started = time.time()
    print("building five sparse operator shells", flush=True)
    construction = build_krylov_closure_construction(
        parameters,
        phonon_cutoff=phonon_cutoff,
        shell_count=5,
        rank_tolerance=1e-12,
    )
    print(
        "constructed force rank "
        f"{construction.force_rank} and shells {construction.shell_dimensions}",
        flush=True,
    )
    print("propagating the offline exact scorer", flush=True)
    exact = exact_holstein_wavefunction_trajectory_for_diagnostics(
        parameters,
        sample_times=times,
        phonon_cutoff=phonon_cutoff,
        relative_tolerance=1e-11,
        absolute_tolerance=1e-13,
        maximum_step=0.01,
    )
    print("integrating teacher-forced orders 2, 3, and 4", flush=True)
    result = teacher_forced_krylov_gate(
        parameters,
        phonon_cutoff=phonon_cutoff,
        final_time=4.0,
        sample_step=0.01,
        orders=(2, 3, 4),
        rank_tolerance=1e-12,
        construction=construction,
        exact_trajectory=exact,
    )

    order_three = result.orders[3].metrics
    c_rms = float(
        order_three["block_residual_subtracted_rms_l2"]["C"]
    )
    terminal = float(order_three["terminal_relative_rms"])
    eta = float(order_three["integrated_residual_ratio_eta"])
    order_difference = float(result.order_3_to_4_source_difference)
    criteria = {
        "cutoff_16_c_residual_subtracted_rms_at_most_0_05": c_rms <= 0.05,
        "beats_exact_k_diagnostic_0_089345": c_rms < 0.089345,
        "beats_archive_0_166690": c_rms < 0.166690,
        "terminal_relative_rms_at_most_0_1": terminal <= 0.1,
        "order_3_to_4_source_difference_at_most_0_1": (
            order_difference <= 0.1
        ),
        "integrated_residual_ratio_at_most_0_1": eta <= 0.1,
    }

    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "classification": "diagnostic",
        "evidence_status": "exploratory_local_not_promoted",
        "status": "complete",
        "decision": "reject_static_hs_krylov_before_rollout",
        "gate_passed": all(criteria.values()),
        "criteria": criteria,
        "parameters": {
            "hopping": parameters.hopping,
            "gamma": parameters.gamma,
            "lambda_ep": parameters.lambda_ep,
            "drive_amplitude": parameters.drive_amplitude,
            "pulse_width": parameters.pulse_width,
            "phonon_cutoff": phonon_cutoff,
            "sample_step": 0.01,
            "final_time": 4.0,
            "rank_tolerance": 1e-12,
        },
        "construction": {
            "hilbert_dimension": construction.hilbert_dimension,
            "force_rank": construction.force_rank,
            "shell_dimensions": construction.shell_dimensions,
            "force_singular_values": construction.force_singular_values.tolist(),
            "retained_symmetric_leakage": (
                construction.retained_symmetric_leakage
            ),
        },
        "archive_metrics": result.archive_metrics,
        "orders": {
            str(order): order_result.metrics
            for order, order_result in result.orders.items()
        },
        "order_3_to_4_source_difference": order_difference,
        "exact_function_evaluations": result.exact_function_evaluations,
        "elapsed_seconds": time.time() - started,
    }

    arrays: dict[str, np.ndarray] = {
        "times": result.times,
        "exact_closed_coordinates": result.exact_closed_coordinates,
        "exact_closed_derivatives": result.exact_closed_derivatives,
        "archive_derivatives": result.archive_derivatives,
    }
    for order, order_result in result.orders.items():
        prefix = f"order_{order}"
        arrays[f"{prefix}_auxiliary_coordinates"] = (
            order_result.auxiliary_coordinates
        )
        arrays[f"{prefix}_modeled_derivatives"] = (
            order_result.modeled_derivatives
        )
        arrays[f"{prefix}_modeled_missing_c_source"] = (
            order_result.modeled_missing_c_source
        )
        arrays[f"{prefix}_static_residual_norms"] = (
            order_result.static_residual_norms
        )
        arrays[f"{prefix}_drive_residual_norms"] = (
            order_result.drive_residual_norms
        )
        arrays[f"{prefix}_total_residual_norms"] = (
            order_result.total_residual_norms
        )
        coefficients = order_result.coefficients
        arrays[f"{prefix}_retained_static"] = coefficients.retained_static
        arrays[f"{prefix}_retained_drive"] = coefficients.retained_drive
        arrays[f"{prefix}_retained_to_auxiliary"] = (
            coefficients.retained_to_auxiliary
        )
        arrays[f"{prefix}_auxiliary_static"] = coefficients.auxiliary_static
        arrays[f"{prefix}_auxiliary_drive"] = coefficients.auxiliary_drive
    np.savez_compressed(output_directory / "gate_b_arrays.npz", **arrays)
    _plot_gate(result, output_directory / "gate_b_residuals.pdf")
    _write_json(output_directory / "summary.json", summary)

    source_hashes = {
        str(path.relative_to(repo_root)): _sha256(path)
        for path in _source_paths(repo_root)
    }
    artifact_paths = (
        output_directory / "summary.json",
        output_directory / "gate_b_arrays.npz",
        output_directory / "gate_b_residuals.pdf",
    )
    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "complete",
        "classification": "diagnostic",
        "evidence_status": "exploratory_local_not_promoted",
        "python": sys.version,
        "platform": platform.platform(),
        "source_hashes": source_hashes,
        "artifact_hashes": {
            path.name: _sha256(path) for path in artifact_paths
        },
    }
    _write_json(output_directory / "runtime_manifest.json", manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("output/local_runs") / RUN_ID,
    )
    parser.add_argument("--phonon-cutoff", type=int, default=16)
    args = parser.parse_args()
    summary = run(args.output_directory, phonon_cutoff=args.phonon_cutoff)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
