#!/usr/bin/env python3
"""Score archive, mixed, packet, and augmented tangents on stored states."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from paper5.stability.archive_gram_tangent_pilot import (
    MIXED_LABELS,
    SPACE_NAMES,
    archive_mixed_tangent_pilot_point,
)
from paper5.stability.hubbard_dimer import (
    DimerParameters,
    GaussianSineDrive,
)


DEFAULT_MEMBERS = {
    "K6": Path(
        "output/local_runs/"
        "paper_v_multi_coherent_double_pulse_blind_model_cutoff16_20260804_v1/"
        "fine_central"
    ),
    "K8": Path(
        "output/local_runs/"
        "paper_v_multi_coherent_capacity_k8_t40_20260804_v1/fine_central"
    ),
    "K10": Path(
        "output/local_runs/"
        "paper_v_multi_coherent_capacity_k10_t40_20260804_v1/fine_central"
    ),
    "K12": Path(
        "output/local_runs/"
        "paper_v_multi_coherent_capacity_k12_t40_20260804_v1/fine_central"
    ),
}
DEFAULT_SCALES = Path(
    "output/local_runs/"
    "paper_v_trajectory_closure_identifiability_cutoff16_20260804_v1/"
    "trajectory_closure_identifiability.npz"
)
DEFAULT_OUTPUT = Path(
    "output/local_runs/"
    "paper_v_archive_mixed_tangent_pilot_cutoff16_20260804_v3"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for label, path in DEFAULT_MEMBERS.items():
        parser.add_argument(
            f"--{label.lower()}-dir",
            type=Path,
            default=path,
        )
    parser.add_argument("--coordinate-scales", type=Path, default=DEFAULT_SCALES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-time", type=float, default=40.0)
    parser.add_argument("--sample-step", type=float, default=1.0)
    parser.add_argument(
        "--geometric-gram-relative-threshold",
        type=float,
        default=1e-10,
    )
    parser.add_argument("--relative-damping", type=float, default=3e-4)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _physical_contract(settings: dict[str, object]) -> dict[str, object]:
    drive = settings["drive_protocol"]
    if not isinstance(drive, dict):
        raise ValueError("drive_protocol must be a mapping")
    return {
        "hopping": float(settings["hopping"]),
        "gamma": float(settings["gamma"]),
        "lambda_ep": float(settings["lambda_ep"]),
        "drive_amplitude": float(settings["drive_amplitude"]),
        "pulse_width": float(settings["pulse_width"]),
        "phonon_cutoff": int(settings["phonon_cutoff"]),
        "drive_protocol": {
            "amplitude": float(drive["amplitude"]),
            "pulse_width": float(drive["pulse_width"]),
            "delays": [float(value) for value in drive["delays"]],
        },
    }


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def _space_summary(
    residuals: np.ndarray,
    closed_errors: np.ndarray,
    correlation_errors: np.ndarray,
    retained_ranks: np.ndarray,
    geometric_ranks: np.ndarray,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for index, name in enumerate(SPACE_NAMES):
        result[name] = {
            "hilbert_relative_residual_rms": _rms(residuals[:, index]),
            "closed_coordinate_scaled_rms": _rms(closed_errors[:, index]),
            "correlation_scaled_rms": _rms(correlation_errors[:, index]),
            "retained_rank_minimum": int(np.min(retained_ranks[:, index])),
            "retained_rank_median": float(
                np.median(retained_ranks[:, index])
            ),
            "retained_rank_maximum": int(np.max(retained_ranks[:, index])),
            "geometric_rank_minimum": int(
                np.min(geometric_ranks[:, index])
            ),
            "geometric_rank_maximum": int(
                np.max(geometric_ranks[:, index])
            ),
        }
    return result


def _fractional_reduction(before: np.ndarray, after: np.ndarray) -> float:
    before_rms = _rms(before)
    return float((before_rms - _rms(after)) / before_rms)


def _make_plot(
    output: Path,
    times: np.ndarray,
    residuals: np.ndarray,
    correlation_errors: np.ndarray,
    candidate_scores: np.ndarray,
) -> None:
    names = {name: index for index, name in enumerate(SPACE_NAMES)}
    k10 = 2
    selected = (
        "archive",
        "archive_relative_mixed",
        "packet_geometric",
        "packet_relative_mixed",
    )
    colors = ("#7f3c8d", "#11a579", "#3969ac", "#e73f74")
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), constrained_layout=True)
    for name, color in zip(selected, colors, strict=True):
        index = names[name]
        axes[0, 0].plot(
            times,
            residuals[k10, :, index],
            label=name.replace("_", " "),
            color=color,
        )
        axes[0, 1].semilogy(
            times,
            np.maximum(correlation_errors[k10, :, index], 1e-16),
            label=name.replace("_", " "),
            color=color,
        )
    axes[0, 0].set_title("K=10 same-state Hilbert residual")
    axes[0, 0].set_ylabel("relative residual")
    axes[0, 1].set_title("K=10 induced C-velocity error")
    axes[0, 1].set_ylabel("scaled RMS")
    for axis in axes[0]:
        axis.set_xlabel("time")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    caps = np.asarray([6, 8, 10, 12])
    for name, color in zip(
        (
            "packet_geometric",
            "packet_archive",
            "packet_relative_mixed",
        ),
        ("#3969ac", "#f2b701", "#e73f74"),
        strict=True,
    ):
        index = names[name]
        axes[1, 0].plot(
            caps,
            [_rms(residuals[member, :, index]) for member in range(4)],
            marker="o",
            label=name.replace("_", " "),
            color=color,
        )
    axes[1, 0].set_title("Capacity dependence")
    axes[1, 0].set_xlabel("maximum packets per electronic branch")
    axes[1, 0].set_ylabel("time-RMS relative residual")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    mean_scores = np.mean(candidate_scores[k10], axis=0)
    order = np.argsort(mean_scores)[-8:]
    axes[1, 1].barh(
        np.arange(order.size),
        mean_scores[order],
        color="#11a579",
    )
    axes[1, 1].set_yticks(
        np.arange(order.size),
        [MIXED_LABELS[index] for index in order],
        fontsize=8,
    )
    axes[1, 1].set_title("K=10 individual mixed-direction scores")
    axes[1, 1].set_xlabel("fraction of archive residual norm squared")
    axes[1, 1].grid(axis="x", alpha=0.25)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    args = _parser().parse_args()
    if args.maximum_time <= 0.0 or args.sample_step <= 0.0:
        raise SystemExit("maximum-time and sample-step must be positive")
    member_directories = {
        label: getattr(args, f"{label.lower()}_dir")
        for label in DEFAULT_MEMBERS
    }
    source_paths: list[Path] = [args.coordinate_scales]
    summaries: dict[str, dict[str, object]] = {}
    for label, directory in member_directories.items():
        arrays_path = directory / "segmented_horizon.npz"
        summary_path = directory / "summary.json"
        if not arrays_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"missing stored trajectory for {label}")
        source_paths.extend((arrays_path, summary_path))
        summaries[label] = json.loads(summary_path.read_text(encoding="utf-8"))

    contracts = {
        label: _physical_contract(summary["parameters"])  # type: ignore[arg-type]
        for label, summary in summaries.items()
    }
    first_contract = contracts["K6"]
    if any(contract != first_contract for contract in contracts.values()):
        raise ValueError("stored members do not share one physical contract")
    parameters = DimerParameters(
        hopping=float(first_contract["hopping"]),
        gamma=float(first_contract["gamma"]),
        lambda_ep=float(first_contract["lambda_ep"]),
        drive_amplitude=float(first_contract["drive_amplitude"]),
        pulse_width=float(first_contract["pulse_width"]),
    )
    drive_data = first_contract["drive_protocol"]
    if not isinstance(drive_data, dict):
        raise ValueError("invalid drive protocol")
    drive = GaussianSineDrive(
        amplitude=float(drive_data["amplitude"]),
        pulse_width=float(drive_data["pulse_width"]),
        delays=tuple(float(value) for value in drive_data["delays"]),
    )
    phonon_cutoff = int(first_contract["phonon_cutoff"])
    relative_dimension = 2 * phonon_cutoff + 1
    with np.load(args.coordinate_scales) as scale_arrays:
        coordinate_scales = np.asarray(
            scale_arrays["coordinate_scales"],
            dtype=float,
        )

    sample_times = np.arange(
        0.0,
        args.maximum_time + 0.5 * args.sample_step,
        args.sample_step,
    )
    member_count = len(member_directories)
    sample_count = sample_times.size
    space_count = len(SPACE_NAMES)
    residuals = np.empty((member_count, sample_count, space_count))
    closed_errors = np.empty_like(residuals)
    correlation_errors = np.empty_like(residuals)
    retained_ranks = np.empty_like(residuals, dtype=int)
    geometric_ranks = np.empty_like(residuals, dtype=int)
    target_norms = np.empty((member_count, sample_count))
    spin_swap_fidelity = np.empty_like(target_norms)
    symmetry_projection_fidelity = np.empty_like(target_norms)
    relative_top_population = np.empty_like(target_norms)
    gram_errors = np.empty_like(target_norms)
    archive_novelty = np.empty_like(target_norms)
    mixed_novelty = np.empty_like(target_norms)
    full_mixed_novelty = np.empty_like(target_norms)
    archive_spectrum = np.empty((member_count, sample_count, 14))
    mixed_spectrum = np.empty((member_count, sample_count, 12))
    full_mixed_spectrum = np.empty((member_count, sample_count, 24))
    candidate_scores = np.empty(
        (member_count, sample_count, len(MIXED_LABELS))
    )
    sampled_packet_counts = np.empty((member_count, sample_count), dtype=int)

    started = time.time()
    for member_index, (label, directory) in enumerate(
        member_directories.items()
    ):
        print(f"scoring stored tangent frames: {label}", flush=True)
        with np.load(directory / "segmented_horizon.npz") as arrays:
            stored_times = np.asarray(arrays["times"], dtype=float)
            parameters_by_time = np.asarray(
                arrays["parameter_trajectory"],
                dtype=float,
            )
            packet_counts = np.asarray(
                arrays["packet_count_trajectory"],
                dtype=int,
            )
        indices = np.searchsorted(stored_times, sample_times)
        if np.any(indices >= stored_times.size) or not np.allclose(
            stored_times[indices],
            sample_times,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(f"{label} does not sample the requested times")
        for sample_index, source_index in enumerate(indices):
            packet_count = int(packet_counts[source_index])
            sampled_packet_counts[member_index, sample_index] = packet_count
            packed = parameters_by_time[
                source_index,
                : 16 * packet_count,
            ]
            result = archive_mixed_tangent_pilot_point(
                packed,
                time=float(sample_times[sample_index]),
                parameters=parameters,
                drive_protocol=drive,
                relative_dimension=relative_dimension,
                coordinate_scales=coordinate_scales,
                geometric_gram_relative_threshold=(
                    args.geometric_gram_relative_threshold
                ),
                relative_damping=args.relative_damping,
            )
            residuals[member_index, sample_index] = (
                result.hilbert_relative_residual
            )
            closed_errors[member_index, sample_index] = (
                result.closed_coordinate_scaled_rms
            )
            correlation_errors[member_index, sample_index] = (
                result.correlation_scaled_rms
            )
            retained_ranks[member_index, sample_index] = result.retained_rank
            geometric_ranks[member_index, sample_index] = result.geometric_rank
            target_norms[member_index, sample_index] = result.target_velocity_norm
            spin_swap_fidelity[member_index, sample_index] = (
                result.spin_swap_fidelity
            )
            symmetry_projection_fidelity[member_index, sample_index] = (
                result.symmetry_projection_fidelity
            )
            relative_top_population[member_index, sample_index] = (
                result.relative_top_population
            )
            gram_errors[member_index, sample_index] = (
                result.archive_gram_max_error
            )
            archive_novelty[member_index, sample_index] = (
                result.archive_novelty_fraction
            )
            mixed_novelty[member_index, sample_index] = (
                result.mixed_novelty_fraction
            )
            full_mixed_novelty[member_index, sample_index] = (
                result.full_mixed_novelty_fraction
            )
            archive_spectrum[member_index, sample_index] = (
                result.archive_novelty_eigenvalues
            )
            mixed_spectrum[member_index, sample_index] = (
                result.mixed_novelty_eigenvalues
            )
            full_mixed_spectrum[member_index, sample_index] = (
                result.full_mixed_novelty_eigenvalues
            )
            candidate_scores[member_index, sample_index] = (
                result.mixed_candidate_residual_reduction
            )

    output = args.output_dir
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    arrays_path = output / "archive_mixed_tangent_pilot.npz"
    np.savez_compressed(
        arrays_path,
        member_labels=np.asarray(tuple(member_directories)),
        space_names=np.asarray(SPACE_NAMES),
        mixed_labels=np.asarray(MIXED_LABELS),
        times=sample_times,
        sampled_packet_counts=sampled_packet_counts,
        coordinate_scales=coordinate_scales,
        hilbert_relative_residual=residuals,
        closed_coordinate_scaled_rms=closed_errors,
        correlation_scaled_rms=correlation_errors,
        retained_rank=retained_ranks,
        geometric_rank=geometric_ranks,
        target_velocity_norm=target_norms,
        spin_swap_fidelity=spin_swap_fidelity,
        symmetry_projection_fidelity=symmetry_projection_fidelity,
        relative_top_population=relative_top_population,
        archive_gram_max_error=gram_errors,
        archive_novelty_fraction=archive_novelty,
        mixed_novelty_fraction=mixed_novelty,
        full_mixed_novelty_fraction=full_mixed_novelty,
        archive_novelty_eigenvalues=archive_spectrum,
        mixed_novelty_eigenvalues=mixed_spectrum,
        full_mixed_novelty_eigenvalues=full_mixed_spectrum,
        mixed_candidate_residual_reduction=candidate_scores,
    )

    per_member: dict[str, object] = {}
    indices = {name: index for index, name in enumerate(SPACE_NAMES)}
    for member_index, label in enumerate(member_directories):
        mean_scores = np.mean(candidate_scores[member_index], axis=0)
        order = np.argsort(mean_scores)[::-1]
        per_member[label] = {
            "packet_count_minimum": int(
                np.min(sampled_packet_counts[member_index])
            ),
            "packet_count_maximum": int(
                np.max(sampled_packet_counts[member_index])
            ),
            "spin_swap_fidelity_minimum": float(
                np.min(spin_swap_fidelity[member_index])
            ),
            "symmetry_projection_fidelity_minimum": float(
                np.min(symmetry_projection_fidelity[member_index])
            ),
            "relative_top_population_maximum": float(
                np.max(relative_top_population[member_index])
            ),
            "archive_gram_max_error": float(
                np.max(gram_errors[member_index])
            ),
            "archive_gram_cutoff_identity_max_residual": float(
                np.max(
                    np.abs(
                        gram_errors[member_index]
                        - 0.5
                        * relative_dimension
                        * relative_top_population[member_index]
                    )
                )
            ),
            "archive_novelty_fraction_mean": float(
                np.mean(archive_novelty[member_index])
            ),
            "mixed_novelty_fraction_mean": float(
                np.mean(mixed_novelty[member_index])
            ),
            "full_local_mixed_novelty_fraction_mean": float(
                np.mean(full_mixed_novelty[member_index])
            ),
            "spaces": _space_summary(
                residuals[member_index],
                closed_errors[member_index],
                correlation_errors[member_index],
                retained_ranks[member_index],
                geometric_ranks[member_index],
            ),
            "fractional_hilbert_residual_reduction": {
                "archive_to_archive_relative_mixed": _fractional_reduction(
                    residuals[member_index, :, indices["archive"]],
                    residuals[
                        member_index,
                        :,
                        indices["archive_relative_mixed"],
                    ],
                ),
                "archive_to_archive_full_local_mixed": _fractional_reduction(
                    residuals[member_index, :, indices["archive"]],
                    residuals[member_index, :, indices["archive_mixed"]],
                ),
                "packet_to_packet_archive": _fractional_reduction(
                    residuals[member_index, :, indices["packet_geometric"]],
                    residuals[member_index, :, indices["packet_archive"]],
                ),
                "packet_to_packet_relative_mixed": _fractional_reduction(
                    residuals[member_index, :, indices["packet_geometric"]],
                    residuals[
                        member_index,
                        :,
                        indices["packet_relative_mixed"],
                    ],
                ),
                "packet_to_packet_full_local_mixed": _fractional_reduction(
                    residuals[member_index, :, indices["packet_geometric"]],
                    residuals[member_index, :, indices["packet_mixed"]],
                ),
            },
            "relative_vs_full_mixed_hilbert_residual_max_difference": {
                "archive": float(
                    np.max(
                        np.abs(
                            residuals[
                                member_index,
                                :,
                                indices["archive_relative_mixed"],
                            ]
                            - residuals[
                                member_index,
                                :,
                                indices["archive_mixed"],
                            ]
                        )
                    )
                ),
                "packet": float(
                    np.max(
                        np.abs(
                            residuals[
                                member_index,
                                :,
                                indices["packet_relative_mixed"],
                            ]
                            - residuals[
                                member_index,
                                :,
                                indices["packet_mixed"],
                            ]
                        )
                    )
                ),
            },
            "leading_mixed_candidates": [
                {
                    "label": MIXED_LABELS[index],
                    "mean_individual_residual_reduction": float(
                        mean_scores[index]
                    ),
                }
                for index in order[:6]
            ],
        }

    summary = {
        "schema_version": 1,
        "status": "complete",
        "classification": "offline_stored_state_tangent_pilot",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_scope": (
            "Same-state local projection only; no online propagation and no "
            "exact trajectory supplied to a model velocity."
        ),
        "symmetry_treatment": (
            "Each packet ket and tangent is projected into the spin-exchange-"
            "symmetric dimer sector before the archive frame is constructed."
        ),
        "center_mode_treatment": (
            "The factored center oscillator is restored as a centered vacuum; "
            "the stored relative-mode state supplies the interacting factor."
        ),
        "archive_gram_cutoff_audit": (
            "For relative_dimension D, the maximum difference between the "
            "canonical archive Gram and the explicit cutoff-space operator "
            "Gram equals (D/2) times the highest-level relative-mode "
            "population. This is the truncated boson-commutator boundary term."
        ),
        "space_relation": (
            "archive is nested in both mixed frames; packet_archive and the "
            "packet-mixed frames are nested augmentations of packet_geometric. "
            "The standalone archive and packet spaces are not assumed nested."
        ),
        "parameters": {
            **first_contract,
            "coupling": parameters.coupling,
            "relative_dimension": relative_dimension,
            "maximum_time": args.maximum_time,
            "sample_step": args.sample_step,
            "geometric_gram_relative_threshold": (
                args.geometric_gram_relative_threshold
            ),
            "relative_damping": args.relative_damping,
        },
        "members": per_member,
        "interpretation_rule": (
            "A large archive-to-relative-mixed reduction identifies conditional "
            "electron-phonon products as useful retained-observable tangents; "
            "a small packet-to-packet_mixed reduction means the packet chart "
            "already realizes most of them. This pilot does not establish an "
            "autonomous 31-coordinate closure."
        ),
        "wall_seconds": time.time() - started,
    }
    summary_path = output / "summary.json"
    _write_json(summary_path, summary)
    plot_path = output / "archive_mixed_tangent_pilot.png"
    _make_plot(
        plot_path,
        sample_times,
        residuals,
        correlation_errors,
        candidate_scores,
    )
    runtime_manifest = {
        "schema_version": 1,
        "status": "complete",
        "command": sys.argv,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            str(path.resolve()): _sha256(path)
            for path in source_paths
        },
        "outputs": {
            path.name: _sha256(path)
            for path in (arrays_path, summary_path, plot_path)
        },
    }
    _write_json(output / "runtime_manifest.json", runtime_manifest)
    print(json.dumps(summary["members"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
