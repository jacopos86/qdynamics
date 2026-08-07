"""Extend a stored multi-coherent model with a larger exploratory packet cap."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from paper5.stability import DimerParameters, GaussianSineDrive
from paper5.stability.multi_coherent_long_horizon import (
    run_segmented_multi_coherent_horizon,
)

MEMBERS = ("central", "plus", "minus")
DEFAULT_RUN_ID = "paper_v_multi_coherent_capacity_k8_t40_20260804_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(
    source_batch: Path,
    output_directory: Path,
    *,
    members: tuple[str, ...],
    final_time: float,
    maximum_packet_count: int,
) -> dict[str, object]:
    """Run selected members with one enlarged adaptive-capacity setting."""

    if not members or len(set(members)) != len(members):
        raise ValueError("members must be a nonempty set of unique names")
    if any(member not in MEMBERS for member in members):
        raise ValueError(f"members must be selected from {MEMBERS}")
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)

    summaries: list[dict[str, Any]] = []
    source_arrays: list[Path] = []
    source_summaries: list[Path] = []
    initial_parameters: dict[str, np.ndarray] = {}
    for member in members:
        source_run = source_batch / f"fine_{member}"
        arrays_path = source_run / "segmented_horizon.npz"
        summary_path = source_run / "summary.json"
        if not arrays_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"missing stored source member {member}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        with np.load(arrays_path) as arrays:
            count = int(arrays["packet_count_trajectory"][0])
            initial_parameters[member] = np.asarray(
                arrays["parameter_trajectory"][0, : 16 * count],
                dtype=float,
            )
        summaries.append(summary)
        source_arrays.append(arrays_path)
        source_summaries.append(summary_path)

    settings = summaries[0]["parameters"]
    if any(summary["parameters"] != settings for summary in summaries[1:]):
        raise ValueError("stored source members have different settings")
    initial_packet_count = int(settings["packet_count"])
    if maximum_packet_count <= int(settings["maximum_packet_count"]):
        raise ValueError("capacity extension must exceed the stored packet cap")
    if maximum_packet_count < initial_packet_count:
        raise ValueError("maximum packet count is below the initial count")

    parameters = DimerParameters(
        hopping=float(settings["hopping"]),
        gamma=float(settings["gamma"]),
        lambda_ep=float(settings["lambda_ep"]),
        drive_amplitude=float(settings["drive_amplitude"]),
        pulse_width=float(settings["pulse_width"]),
    )
    drive_data = settings["drive_protocol"]
    drive = GaussianSineDrive(
        amplitude=float(drive_data["amplitude"]),
        pulse_width=float(drive_data["pulse_width"]),
        delays=tuple(float(value) for value in drive_data["delays"]),
    )

    results: dict[str, dict[str, object]] = {}
    for member in members:
        print(
            f"running {member} with maximum packet count {maximum_packet_count}",
            flush=True,
        )
        target = output_directory / f"fine_{member}"
        result = run_segmented_multi_coherent_horizon(
            target,
            gate_directory=None,
            parameters=parameters,
            final_time=final_time,
            segment_length=float(settings["segment_length"]),
            output_sample_step=float(settings["output_sample_step"]),
            segment_timeout_seconds=240.0,
            maximum_step=float(settings["maximum_step"]),
            relative_tolerance=float(settings["relative_tolerance"]),
            absolute_tolerance=float(settings["absolute_tolerance"]),
            phonon_cutoff=int(settings["phonon_cutoff"]),
            packet_count=initial_packet_count,
            tangent_singular_value_cutoff=float(
                settings["tangent_singular_value_cutoff"]
            ),
            tangent_regularization=str(settings["tangent_regularization"]),
            relative_damping=float(settings["relative_damping"]),
            adaptive_capacity=True,
            maximum_packet_count=maximum_packet_count,
            spawn_relative_residual_threshold=float(
                settings["spawn_relative_residual_threshold"]
            ),
            spawn_absolute_residual_threshold=float(
                settings["spawn_absolute_residual_threshold"]
            ),
            spawn_fit_maximum_iterations=int(
                settings["spawn_fit_maximum_iterations"]
            ),
            spawn_fit_population_size=int(
                settings["spawn_fit_population_size"]
            ),
            spawn_seed=int(settings["spawn_seed"]),
            compare_exact=False,
            drive_protocol=drive,
            initial_parameters_override=initial_parameters[member],
        )
        results[member] = {
            "status": result["status"],
            "final_time": result["progress"]["last_completed_time"],
            "final_packet_count": result["capacity"]["final_packet_count"],
            "spawn_count": result["capacity"]["spawn_count"],
            "wall_seconds": result["resource_usage"]["wall_seconds"],
        }

    repo_root = Path(__file__).resolve().parents[2]
    source_files = (
        Path(__file__).resolve(),
        repo_root / "paper_5/src/paper5/stability/multi_coherent_long_horizon.py",
        repo_root / "paper_5/src/paper5/stability/multi_coherent.py",
    )
    manifest: dict[str, object] = {
        "schema": "paper5.multi_coherent.capacity_extension.v1",
        "status": "complete",
        "classification": "exploratory_capacity_continuation",
        "run_id": output_directory.name,
        "members": list(members),
        "final_time": final_time,
        "stored_packet_cap": int(settings["maximum_packet_count"]),
        "extended_packet_cap": maximum_packet_count,
        "interpretation": (
            "one controlled capacity continuation; the cap is not a "
            "scientific rejection threshold"
        ),
        "results": results,
        "input_hashes": {
            str(path): _sha256(path)
            for path in (*source_arrays, *source_summaries)
        },
        "source_hashes": {
            str(path.relative_to(repo_root)): _sha256(path)
            for path in source_files
        },
    }
    manifest_path = output_directory / "capacity_extension_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-batch",
        type=Path,
        default=Path("output/local_runs")
        / "paper_v_multi_coherent_double_pulse_blind_model_cutoff16_20260804_v1",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("output/local_runs") / DEFAULT_RUN_ID,
    )
    parser.add_argument("--members", nargs="+", choices=MEMBERS, default=["central"])
    parser.add_argument("--final-time", type=float, default=40.0)
    parser.add_argument("--maximum-packet-count", type=int, default=8)
    args = parser.parse_args()
    result = run(
        args.source_batch,
        args.output_directory,
        members=tuple(args.members),
        final_time=args.final_time,
        maximum_packet_count=args.maximum_packet_count,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
