"""Matched four-lane test of the autonomous same-spin Pauli repair."""

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
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import scipy

from .electron_phonon_analysis import (
    PauliRepairAblation,
    analyze_pauli_repair_ablation,
)
from .hubbard_dimer import DimerParameters
from .matrix_reference import (
    closed_scalar_to_matrix_state,
    electron_phonon_moment_matrix,
)


LANES = (
    "raw",
    "pauli_repaired",
    "controller",
    "pauli_repaired_controller",
)
LANE_LABELS = {
    "raw": "archive EOM",
    "pauli_repaired": "Pauli repair",
    "controller": "controller",
    "pauli_repaired_controller": "Pauli repair + controller",
}
LANE_COLORS = {
    "raw": "#a33a2b",
    "pauli_repaired": "#d28b16",
    "controller": "#376f9e",
    "pauli_repaired_controller": "#2f7d4a",
}


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


def _joint_minima(coordinates: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.linalg.eigvalsh(
                electron_phonon_moment_matrix(
                    closed_scalar_to_matrix_state(row)
                )
            )[0]
            for row in coordinates
        ],
        dtype=float,
    )


def _case_arrays(case: PauliRepairAblation) -> dict[str, np.ndarray]:
    arrays = {
        "times": case.exact.times,
        "exact_coordinates": case.exact_coordinates,
        "exact_joint_gram_minimum_eigenvalue": _joint_minima(
            case.exact_coordinates
        ),
    }
    for name in LANES:
        trajectory = getattr(case, name)
        arrays[f"{name}_coordinates"] = trajectory.coordinates
        arrays[f"{name}_correction_coordinates"] = (
            trajectory.correction_coordinates
        )
        arrays[f"{name}_joint_gram_minimum_eigenvalue"] = _joint_minima(
            trajectory.coordinates
        )
    return arrays


def _write_figure(
    run_directory: Path,
    case: PauliRepairAblation,
    arrays: dict[str, np.ndarray],
) -> None:
    times = arrays["times"]
    figure, axes = plt.subplots(3, 1, figsize=(8.2, 8.7), sharex=True)
    for name in LANES:
        coordinate_error = np.linalg.norm(
            arrays[f"{name}_coordinates"] - arrays["exact_coordinates"],
            axis=1,
        )
        axes[0].plot(
            times,
            np.maximum(coordinate_error, 1e-15),
            label=LANE_LABELS[name],
            color=LANE_COLORS[name],
        )
        axes[1].plot(
            times,
            arrays[f"{name}_joint_gram_minimum_eigenvalue"],
            label=LANE_LABELS[name],
            color=LANE_COLORS[name],
        )
    axes[1].plot(
        times,
        arrays["exact_joint_gram_minimum_eigenvalue"],
        label="exact cutoff reference",
        color="black",
        linestyle="--",
    )
    for name in ("controller", "pauli_repaired_controller"):
        correction_norm = np.linalg.norm(
            arrays[f"{name}_correction_coordinates"],
            axis=1,
        )
        axes[2].plot(
            times,
            correction_norm,
            label=LANE_LABELS[name],
            color=LANE_COLORS[name],
        )

    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\|x-x_{\rm ex}\|_2$")
    axes[0].set_title("Trajectory error")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel(r"$\lambda_{\min}(\mathcal{G})$")
    axes[1].set_title("Joint representability margin")
    axes[2].set_ylabel("controller correction norm")
    axes[2].set_xlabel(r"time $t\,t_{\rm hop}$")
    axes[2].set_title("Minimum-norm controller action")
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            run_directory / f"pauli_repair_ablation.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def _decision_summary(case: PauliRepairAblation) -> dict[str, Any]:
    defects = case.metrics["exact_sample_C_derivative_defect"]
    raw_absolute_defect = defects["raw"]["absolute_time_rms_l2"]
    repaired_absolute_defect = defects["pauli_repaired"][
        "absolute_time_rms_l2"
    ]
    raw_dynamic_defect = defects["raw"]["residual_subtracted_time_rms_l2"]
    repaired_dynamic_defect = defects["pauli_repaired"][
        "residual_subtracted_time_rms_l2"
    ]
    raw_c_error = case.metrics["lanes"]["raw"]["block_errors"]["C"][
        "rms_frobenius_error"
    ]
    repaired_c_error = case.metrics["lanes"]["pauli_repaired"][
        "block_errors"
    ]["C"]["rms_frobenius_error"]
    controller_c_error = case.metrics["lanes"]["controller"]["block_errors"][
        "C"
    ]["rms_frobenius_error"]
    combined_c_error = case.metrics["lanes"]["pauli_repaired_controller"][
        "block_errors"
    ]["C"]["rms_frobenius_error"]
    controller_coordinate_error = case.metrics["lanes"]["controller"][
        "maximum_coordinate_l2_error"
    ]
    combined_coordinate_error = case.metrics["lanes"][
        "pauli_repaired_controller"
    ]["maximum_coordinate_l2_error"]
    controller_rms = case.metrics["controller_history"]["rms_correction_norm"]
    combined_rms = case.metrics["pauli_repaired_controller_history"][
        "rms_correction_norm"
    ]
    return {
        "Pauli_repair_reduces_absolute_exact_sample_C_derivative_defect": (
            repaired_absolute_defect < raw_absolute_defect
        ),
        "absolute_C_derivative_defect_ratio_repaired_over_raw": (
            repaired_absolute_defect / max(raw_absolute_defect, 1e-30)
        ),
        "Pauli_repair_reduces_residual_subtracted_C_derivative_defect": (
            repaired_dynamic_defect < raw_dynamic_defect
        ),
        "residual_subtracted_C_derivative_defect_ratio_repaired_over_raw": (
            repaired_dynamic_defect / max(raw_dynamic_defect, 1e-30)
        ),
        "Pauli_repair_reduces_C_trajectory_error": repaired_c_error < raw_c_error,
        "C_trajectory_rms_error_ratio_repaired_over_raw": (
            repaired_c_error / max(raw_c_error, 1e-30)
        ),
        "combined_C_trajectory_rms_error_ratio_over_controller_only": (
            combined_c_error / max(controller_c_error, 1e-30)
        ),
        "combined_maximum_coordinate_error_ratio_over_controller_only": (
            combined_coordinate_error
            / max(controller_coordinate_error, 1e-30)
        ),
        "combined_controller_rms_ratio_over_controller_only": (
            combined_rms / max(controller_rms, 1e-30)
        ),
        "interpretation": (
            "A lower absolute defect and lower propagated error establish a "
            "useful partial repair. A larger residual-subtracted defect means "
            "that it does not repair the remaining time-dependent closure error; "
            "representability and controller effort remain separate tests."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default="paper_v_autonomous_pauli_repair_ablation_20260803_v1",
    )
    parser.add_argument("--output-root", type=Path, default=Path("output/local_runs"))
    parser.add_argument("--final-time", type=float, default=4.0)
    parser.add_argument("--time-step", type=float, default=0.01)
    parser.add_argument("--phonon-cutoff", type=int, default=16)
    parser.add_argument("--lambda-ep", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--drive", type=float, default=1.0)
    parser.add_argument("--activation-margin", type=float, default=1e-5)
    parser.add_argument("--barrier-rate", type=float, default=5.0)
    parser.add_argument("--cone-tolerance", type=float, default=1e-8)
    parser.add_argument("--maximum-constraints", type=int, default=128)
    return parser


def main() -> int:
    args = _parser().parse_args()
    run_directory = args.output_root / args.run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    parameters = DimerParameters(
        lambda_ep=args.lambda_ep,
        gamma=args.gamma,
        drive_amplitude=args.drive,
    )
    print(
        json.dumps(
            {
                "event": "four_lane_ablation_started",
                "run_id": args.run_id,
                "recorded_at_utc": _utc_now(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    case = analyze_pauli_repair_ablation(
        parameters,
        final_time=args.final_time,
        time_step=args.time_step,
        phonon_cutoff=args.phonon_cutoff,
        activation_margin=args.activation_margin,
        barrier_rate=args.barrier_rate,
        cone_tolerance=args.cone_tolerance,
        maximum_constraints=args.maximum_constraints,
    )
    arrays = _case_arrays(case)
    summary = {
        "schema_version": 1,
        "run_id": args.run_id,
        "classification": "exploratory_local_not_promoted",
        "created_at_utc": _utc_now(),
        "scientific_question": (
            "Does the autonomous fixed-sector same-spin Pauli replacement "
            "improve the 31-coordinate archive EOM alone and with the joint "
            "representability controller?"
        ),
        **case.metrics,
        "decision_summary": _decision_summary(case),
        "wall_time_seconds": time.perf_counter() - started,
    }
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("matrix_reference.py"),
        Path(__file__).with_name("electron_phonon_analysis.py"),
        Path(__file__).with_name("exact_reference.py"),
    )
    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "created_at_utc": summary["created_at_utc"],
        "command": "python -m paper5.stability.pauli_repair_analysis",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "source_hashes": {
            str(path): _sha256(path) for path in source_paths
        },
        "exact_reference_usage": summary["exact_reference_usage"],
    }
    _write_npz_atomic(run_directory / "trajectories.npz", arrays)
    _write_figure(run_directory, case, arrays)
    _write_json_atomic(run_directory / "summary.json", summary)
    _write_json_atomic(run_directory / "runtime_manifest.json", manifest)
    print(
        json.dumps(
            {
                "event": "four_lane_ablation_completed",
                "run_id": args.run_id,
                "recorded_at_utc": _utc_now(),
                "decision_summary": summary["decision_summary"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
