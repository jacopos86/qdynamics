"""Run a memory-safe, chunked carried-witness campaign and merge its evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .apcm_carried_witness_analysis import _accuracy_metrics
from .hubbard_dimer import DimerParameters


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--initial-state-file", type=Path)
    parser.add_argument("--final-time", type=float, required=True)
    parser.add_argument("--time-step", type=float, default=0.0025)
    parser.add_argument("--chunk-duration", type=float, default=0.025)
    parser.add_argument("--phonon-cutoff", type=int, default=16)
    parser.add_argument("--lambda-ep", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--drive", type=float, default=1.0)
    parser.add_argument("--maximum-critical-modes", type=int)
    return parser


def _format_time(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _concatenate(
    chunks: list[Path],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    trajectory_keys = (
        "times",
        "carried_states",
        "approximate_archive_coordinates",
        "exact_archive_coordinates",
        "minimum_unshifted_eigenvalues",
        "minimum_shifted_lower_bounds",
        "completion_correction_norms",
        "critical_modes",
    )
    autonomous_keys = (
        "maximum_atom_seconds",
        "correction_iterations",
        "readable_rate_residuals",
        "velocity_margins",
    )
    trajectory_parts: dict[str, list[np.ndarray]] = {
        key: [] for key in trajectory_keys
    }
    autonomous_parts: dict[str, list[np.ndarray]] = {
        key: [] for key in autonomous_keys
    }
    for chunk_index, chunk in enumerate(chunks):
        trajectory = np.load(chunk / "trajectory.npz")
        autonomous = np.load(chunk / "autonomous_trajectory.npz")
        row = slice(None) if chunk_index == 0 else slice(1, None)
        for key in trajectory_keys:
            trajectory_parts[key].append(np.asarray(trajectory[key])[row])
        for key in autonomous_keys:
            autonomous_parts[key].append(np.asarray(autonomous[key])[row])
    return (
        {
            key: np.concatenate(parts, axis=0)
            for key, parts in trajectory_parts.items()
        },
        {
            key: np.concatenate(parts, axis=0)
            for key, parts in autonomous_parts.items()
        },
    )


def _merged_summary(
    args: argparse.Namespace,
    chunks: list[Path],
    trajectory: dict[str, np.ndarray],
    autonomous: dict[str, np.ndarray],
) -> dict[str, Any]:
    chunk_summaries = [
        json.loads((chunk / "summary.json").read_text(encoding="utf-8"))
        for chunk in chunks
    ]
    parameters = DimerParameters(
        hopping=1.0,
        gamma=args.gamma,
        lambda_ep=args.lambda_ep,
        drive_amplitude=args.drive,
    )
    metrics = _accuracy_metrics(
        parameters,
        trajectory["exact_archive_coordinates"],
        trajectory["approximate_archive_coordinates"],
    )
    return {
        "schema_version": 1,
        "classification": "exploratory_local_not_promoted",
        "model": "chunked_carried_witness_radial_moment_flow",
        "parameters": chunk_summaries[0]["parameters"],
        "state": chunk_summaries[0]["state"],
        "integration": {
            "method": "chunked SSPRK2 with finite-step radial atoms",
            "initial_time": float(trajectory["times"][0]),
            "last_time": float(trajectory["times"][-1]),
            "time_step": args.time_step,
            "chunk_duration": args.chunk_duration,
            "chunk_count": len(chunks),
            "completed_steps": int(trajectory["times"].size - 1),
            "maximum_critical_modes": args.maximum_critical_modes,
            "critical_mode_selection": (
                "adaptive_from_gram_spectrum"
                if args.maximum_critical_modes is None
                else "spectrum_selected_with_explicit_cap"
            ),
            "maximum_active_critical_modes": int(
                np.max(trajectory["critical_modes"])
            ),
            "mean_active_critical_modes": float(
                np.mean(trajectory["critical_modes"])
            ),
            "autonomous_wall_seconds": float(
                sum(
                    summary["integration"]["autonomous_wall_seconds"]
                    for summary in chunk_summaries
                )
            ),
            "exact_reference_wall_seconds": float(
                sum(
                    summary["integration"]["exact_reference_wall_seconds"]
                    for summary in chunk_summaries
                )
            ),
            "maximum_chunk_resident_bytes": int(
                max(
                    summary["integration"]["peak_resident_bytes"]
                    for summary in chunk_summaries
                )
            ),
        },
        "feasibility": {
            "minimum_carried_unshifted_eigenvalue": float(
                np.min(trajectory["minimum_unshifted_eigenvalues"])
            ),
            "minimum_carried_shifted_lower_bound": float(
                np.min(trajectory["minimum_shifted_lower_bounds"])
            ),
            "minimum_retained_joint_gram_eigenvalue": float(
                min(
                    summary["feasibility"][
                        "minimum_retained_joint_gram_eigenvalue"
                    ]
                    for summary in chunk_summaries
                )
            ),
            "maximum_readable_rate_residual": float(
                np.max(autonomous["readable_rate_residuals"])
            ),
            "maximum_completion_correction_norm": float(
                np.max(trajectory["completion_correction_norms"])
            ),
            "minimum_velocity_margin": float(
                np.min(autonomous["velocity_margins"])
            ),
        },
        "accuracy": metrics,
        "chunks": [str(chunk) for chunk in chunks],
    }


def main() -> int:
    args = _parser().parse_args()
    if args.final_time <= 0.0:
        raise ValueError("final_time must be positive")
    if args.time_step <= 0.0 or args.chunk_duration <= 0.0:
        raise ValueError("step and chunk durations must be positive")
    if not np.isclose(
        round(args.chunk_duration / args.time_step) * args.time_step,
        args.chunk_duration,
        atol=1e-12,
    ):
        raise ValueError("chunk_duration must be a multiple of time_step")

    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=False)
    chunks_directory = output_directory / "chunks"
    chunks_directory.mkdir()
    package_root = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()
    python_path = [str(package_root / "src"), str(package_root)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)

    completed_chunks: list[Path] = []
    current_time = 0.0
    previous_checkpoint: Path | None = None
    if args.initial_state_file is not None:
        previous_checkpoint = args.initial_state_file.resolve()
        checkpoint = np.load(previous_checkpoint)
        if "time" not in checkpoint or "state" not in checkpoint:
            raise ValueError("initial-state-file must be a campaign checkpoint")
        current_time = float(np.asarray(checkpoint["time"]))
        if current_time >= args.final_time:
            raise ValueError("checkpoint time must precede final_time")
    chunk_index = 0
    while current_time < args.final_time - 1e-12:
        chunk_index += 1
        next_time = min(args.final_time, current_time + args.chunk_duration)
        chunk = chunks_directory / (
            f"chunk_{chunk_index:03d}_{_format_time(current_time)}_"
            f"{_format_time(next_time)}"
        )
        command = [
            sys.executable,
            "-m",
            "paper5.stability.apcm_carried_witness_analysis",
            "--output-directory",
            str(chunk),
            "--final-time",
            str(next_time),
            "--time-step",
            str(args.time_step),
            "--phonon-cutoff",
            str(args.phonon_cutoff),
            "--lambda-ep",
            str(args.lambda_ep),
            "--gamma",
            str(args.gamma),
            "--drive",
            str(args.drive),
            "--compact-output",
        ]
        if previous_checkpoint is not None:
            command.extend(
                ["--initial-state-file", str(previous_checkpoint)]
            )
        if args.maximum_critical_modes is not None:
            command.extend(
                [
                    "--maximum-critical-modes",
                    str(args.maximum_critical_modes),
                ]
            )
        print(
            f"campaign chunk={chunk_index} t={current_time:.6f}->{next_time:.6f}",
            flush=True,
        )
        result = subprocess.run(
            command,
            cwd=package_root,
            env=environment,
            check=False,
        )
        if result.returncode != 0:
            print(
                f"campaign stopped in chunk {chunk_index}; artifacts preserved",
                flush=True,
            )
            return result.returncode
        completed_chunks.append(chunk)
        previous_checkpoint = chunk / "checkpoint.npz"
        current_time = next_time

    trajectory, autonomous = _concatenate(completed_chunks)
    np.savez_compressed(output_directory / "trajectory.npz", **trajectory)
    np.savez_compressed(
        output_directory / "autonomous_diagnostics.npz", **autonomous
    )
    summary = _merged_summary(
        args, completed_chunks, trajectory, autonomous
    )
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
