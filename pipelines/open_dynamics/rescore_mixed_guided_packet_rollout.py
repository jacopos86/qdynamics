#!/usr/bin/env python3
"""Rescore a stored mixed-guided trajectory in the native moment chart.

The propagation is not repeated.  The stored packet parameters are contracted
with the same center/relative moment map used by the ordinary and fixed-
capacity packet trajectories.  Projection into the local cutoff is retained
only as a separate embedding diagnostic and for any already-stored fidelity.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

from paper5.stability.hubbard_dimer import DimerParameters
from paper5.stability.mixed_enriched_propagation import normalized_packet_state
from paper5.stability.moment_hierarchy import moment_hierarchy
from paper5.stability.multi_coherent import relative_state_closed_coordinates
from pipelines.open_dynamics.run_mixed_guided_packet_rollout import (
    _observables,
    _plot,
    _sha256,
    _write_json,
)


def _observable_rms(
    times: np.ndarray,
    path: np.ndarray,
    exact: np.ndarray,
) -> dict[str, float]:
    names = (
        "site_0_occupation",
        "site_1_occupation",
        "electronic_energy",
        "phonon_energy",
        "electron_phonon_energy",
        "internal_total_energy",
    )
    duration = float(times[-1] - times[0])
    return {
        name: float(
            np.sqrt(
                np.trapezoid((path[:, column] - exact[:, column]) ** 2, times)
                / duration
            )
        )
        for column, name in enumerate(names)
    }


def rescore(source_directory: Path, output_directory: Path) -> dict[str, object]:
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty directory {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    source_summary_path = source_directory / "summary.json"
    source_arrays_path = source_directory / "mixed_guided_packet_rollout.npz"
    source_manifest_path = source_directory / "runtime_manifest.json"
    if not all(
        path.is_file()
        for path in (
            source_summary_path,
            source_arrays_path,
            source_manifest_path,
        )
    ):
        raise FileNotFoundError("source rollout artifacts are incomplete")

    summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    settings = summary["parameters"]
    parameters = DimerParameters(
        hopping=1.0,
        gamma=float(settings["gamma"]),
        lambda_ep=float(settings["lambda_ep"]),
        drive_amplitude=float(settings["drive_protocol"]["amplitude"]),
        pulse_width=float(settings["drive_protocol"]["pulse_width"]),
    )
    with np.load(source_arrays_path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    times = np.asarray(arrays["times"], dtype=float)
    packet_parameters = np.asarray(arrays["parameter_trajectory"], dtype=float)
    if "packet_count_trajectory" in arrays:
        packet_counts = np.asarray(
            arrays["packet_count_trajectory"],
            dtype=int,
        )
    else:
        packet_counts = np.full(
            times.size,
            packet_parameters.shape[1] // 16,
            dtype=int,
        )
    exact_closed = np.asarray(arrays["exact_closed_coordinates"], dtype=float)
    parent_closed = np.asarray(arrays["parent_closed_coordinates"], dtype=float)
    projected_closed = np.asarray(
        arrays["mixed_guided_closed_coordinates"],
        dtype=float,
    )
    scales = np.asarray(arrays["coordinate_scales"], dtype=float)
    exact_observables = np.asarray(arrays["exact_observables"], dtype=float)
    parent_observables = np.asarray(arrays["parent_observables"], dtype=float)
    if packet_parameters.shape[0] != times.size or packet_counts.shape != times.shape:
        raise ValueError("stored packet trajectory has incompatible dimensions")

    relative_dimension = 2 * int(settings["phonon_cutoff"]) + 1
    hierarchy = moment_hierarchy(4)
    center_amplitude = (
        -np.sqrt(2.0) * parameters.coupling / parameters.omega_ph
    )
    direct_closed = np.empty_like(exact_closed)
    for index, packet_count in enumerate(packet_counts):
        state = normalized_packet_state(
            packet_parameters[index, : 16 * int(packet_count)],
            relative_dimension=relative_dimension,
        )
        direct_closed[index] = relative_state_closed_coordinates(
            state,
            hierarchy,
            center_amplitude=center_amplitude,
        )

    direct_observables = _observables(times, direct_closed, parameters)
    projected_observables = _observables(times, projected_closed, parameters)
    direct_error = (direct_closed - exact_closed) / scales
    projected_error = (projected_closed - exact_closed) / scales
    parent_error = (parent_closed - exact_closed) / scales
    comparison = {
        "parent_all31_scaled_rms": float(np.sqrt(np.mean(parent_error**2))),
        "mixed_guided_all31_scaled_rms": float(
            np.sqrt(np.mean(direct_error**2))
        ),
        "parent_c_scaled_rms": float(
            np.sqrt(np.mean(parent_error[:, 17:31] ** 2))
        ),
        "mixed_guided_c_scaled_rms": float(
            np.sqrt(np.mean(direct_error[:, 17:31] ** 2))
        ),
        "projected_embedding_all31_scaled_rms": float(
            np.sqrt(np.mean(projected_error**2))
        ),
        "projected_embedding_c_scaled_rms": float(
            np.sqrt(np.mean(projected_error[:, 17:31] ** 2))
        ),
        "minimum_exact_state_fidelity": summary["comparison"].get(
            "minimum_exact_state_fidelity"
        ),
        "final_exact_state_fidelity": summary["comparison"].get(
            "final_exact_state_fidelity"
        ),
        "minimum_local_embedding_retained_norm": summary["comparison"].get(
            "minimum_local_embedding_retained_norm"
        ),
        "parent_observable_time_rms": _observable_rms(
            times,
            parent_observables,
            exact_observables,
        ),
        "mixed_guided_observable_time_rms": _observable_rms(
            times,
            direct_observables,
            exact_observables,
        ),
        "projected_embedding_observable_time_rms": _observable_rms(
            times,
            projected_observables,
            exact_observables,
        ),
    }
    rescored_summary = {
        **summary,
        "schema": "paper_v_mixed_guided_packet_summary_v2",
        "run_id": output_directory.name,
        "comparison": comparison,
        "scoring_representation": {
            "primary": "native_center_relative_moment_contraction",
            "matches_parent_and_fixed_capacity_packet_scoring": True,
            "local_cutoff_projection_role": (
                "separate embedding and exact-state-fidelity diagnostic"
            ),
            "source_run": str(source_directory),
        },
        "interpretation": (
            "The stored autonomous packet trajectory is scored through the "
            "same center/relative moment contraction as the ordinary and "
            "fixed-capacity packet comparators. Local-cutoff projection is "
            "reported separately and is not mixed into the primary moment "
            "or energy comparison."
        ),
    }

    arrays.update(
        {
            "mixed_guided_closed_coordinates": direct_closed,
            "mixed_guided_projected_closed_coordinates": projected_closed,
            "mixed_guided_observables": direct_observables,
            "mixed_guided_projected_observables": projected_observables,
        }
    )
    arrays_path = output_directory / "mixed_guided_packet_rollout.npz"
    np.savez_compressed(arrays_path, **arrays)
    plot_path = output_directory / "observable_comparison.png"
    _plot(
        plot_path,
        times,
        exact_observables,
        parent_observables,
        direct_observables,
    )
    summary_path = output_directory / "summary.json"
    _write_json(summary_path, rescored_summary)
    plan_path = output_directory / "plan.json"
    _write_json(
        plan_path,
        {
            "schema": "paper_v_mixed_guided_packet_rescore_plan_v1",
            "source_directory": str(source_directory),
            "source_arrays_sha256": _sha256(source_arrays_path),
            "dynamics_repeated": False,
            "primary_scoring_representation": (
                "native_center_relative_moment_contraction"
            ),
        },
    )
    artifacts = (plan_path, arrays_path, plot_path, summary_path)
    manifest = {
        "schema": "paper_v_mixed_guided_packet_rescore_manifest_v1",
        "status": "complete",
        "python": sys.version,
        "platform": platform.platform(),
        "input_hashes": {
            str(path): _sha256(path)
            for path in (
                source_summary_path,
                source_arrays_path,
                source_manifest_path,
            )
        },
        "source_hashes": {
            str(Path(__file__).resolve()): _sha256(Path(__file__).resolve()),
        },
        "artifact_hashes": {
            path.name: _sha256(path) for path in artifacts
        },
    }
    _write_json(output_directory / "runtime_manifest.json", manifest)
    print(json.dumps(rescored_summary, indent=2, sort_keys=True))
    return rescored_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    rescore(arguments.source_directory, arguments.output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
