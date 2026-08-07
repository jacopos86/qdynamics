"""Offline exact-state gate for a physically adapted packet representation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .conditional_packets import (
    ConditionalPacketMetrics,
    ConditionalRelativeState,
    analyze_conditional_packet,
    conditional_relative_state,
    husimi_peaks,
)
from .exact_reference import (
    exact_holstein_wavefunction_trajectory_for_diagnostics,
)
from .hubbard_dimer import DimerParameters

ELECTRON_CONFIGURATION_LABELS = (
    "up_site_0_down_site_0",
    "up_site_0_down_site_1",
    "up_site_1_down_site_0",
    "up_site_1_down_site_1",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_times(values: tuple[float, ...], *, name: str) -> np.ndarray:
    times = np.asarray(values, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError(f"{name} must contain at least two times")
    if abs(float(times[0])) > 1e-15:
        raise ValueError(f"{name} must start at zero")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return times


def _packet_geometry(metrics: ConditionalPacketMetrics) -> dict[str, Any]:
    parameters = metrics.two_coherent_fit.parameters
    alpha_0 = complex(parameters[0], parameters[1])
    alpha_1 = complex(parameters[2], parameters[3])
    separation = abs(alpha_0 - alpha_1)
    return {
        "centers": [
            [float(alpha_0.real), float(alpha_0.imag)],
            [float(alpha_1.real), float(alpha_1.imag)],
        ],
        "center_separation": float(separation),
        "coherent_packet_overlap_magnitude": float(
            np.exp(-0.5 * separation**2)
        ),
    }


def _metrics_record(
    metrics: ConditionalPacketMetrics,
    *,
    fit_reused_from_index: int | None,
) -> dict[str, Any]:
    record = asdict(metrics)
    record["configuration"] = ELECTRON_CONFIGURATION_LABELS[
        metrics.electronic_index
    ]
    record["fit_reused_from_index"] = fit_reused_from_index
    record["two_packet_geometry"] = _packet_geometry(metrics)
    record["two_packet_fidelity_gain"] = float(
        metrics.two_coherent_fit.fidelity
        - metrics.single_gaussian_fit.fidelity
    )
    return record


def _analyze_wavefunction(
    state_vector: np.ndarray,
    *,
    phonon_cutoff: int,
    seed: int,
    single_gaussian_random_starts: int,
    single_gaussian_maximum_iterations: int,
    two_packet_maximum_iterations: int,
    two_packet_population_size: int,
    husimi_grid_points: int,
    spin_exchange_reuse_tolerance: float,
) -> tuple[
    tuple[ConditionalRelativeState, ...],
    tuple[ConditionalPacketMetrics, ...],
    tuple[int | None, ...],
    dict[str, float],
]:
    conditional = tuple(
        conditional_relative_state(
            state_vector,
            electronic_index=index,
            phonon_cutoff=phonon_cutoff,
        )
        for index in range(4)
    )
    exchange_probability_difference = abs(
        conditional[1].probability - conditional[2].probability
    )
    exchange_density_difference = float(
        np.linalg.norm(
            conditional[1].density_matrix - conditional[2].density_matrix
        )
    )
    reuse_exchange_fit = (
        exchange_probability_difference <= spin_exchange_reuse_tolerance
        and exchange_density_difference <= spin_exchange_reuse_tolerance
    )

    metrics: list[ConditionalPacketMetrics] = []
    reused: list[int | None] = []
    for electronic_index, block in enumerate(conditional):
        if electronic_index == 2 and reuse_exchange_fit:
            source = metrics[1]
            metrics.append(
                replace(
                    source,
                    electronic_index=2,
                    probability=block.probability,
                    center_relative_factorization=(
                        block.center_relative_factorization
                    ),
                    relative_purity=float(
                        np.trace(
                            block.density_matrix @ block.density_matrix
                        ).real
                    ),
                )
            )
            reused.append(1)
            continue
        metrics.append(
            analyze_conditional_packet(
                block,
                single_gaussian_random_starts=(
                    single_gaussian_random_starts
                ),
                single_gaussian_maximum_iterations=(
                    single_gaussian_maximum_iterations
                ),
                two_packet_maximum_iterations=(
                    two_packet_maximum_iterations
                ),
                two_packet_population_size=two_packet_population_size,
                seed=seed + 17 * electronic_index,
                husimi_grid_points=husimi_grid_points,
            )
        )
        reused.append(None)
    return (
        conditional,
        tuple(metrics),
        tuple(reused),
        {
            "probability_difference": float(
                exchange_probability_difference
            ),
            "density_frobenius_difference": exchange_density_difference,
            "fit_reused": bool(reuse_exchange_fit),
        },
    )


def _metric_arrays(
    metrics_by_time: tuple[tuple[ConditionalPacketMetrics, ...], ...],
) -> dict[str, np.ndarray]:
    extractors = {
        "probability": lambda value: value.probability,
        "center_relative_factorization": (
            lambda value: value.center_relative_factorization
        ),
        "relative_purity": lambda value: value.relative_purity,
        "gaussian_non_gaussianity": (
            lambda value: value.gaussian_non_gaussianity
        ),
        "husimi_peak_count": lambda value: value.husimi_peak_count,
        "husimi_second_peak_ratio": (
            lambda value: value.husimi_second_peak_ratio
        ),
        "single_gaussian_fidelity": (
            lambda value: value.single_gaussian_fit.fidelity
        ),
        "two_coherent_fidelity": (
            lambda value: value.two_coherent_fit.fidelity
        ),
        "two_packet_center_separation": (
            lambda value: _packet_geometry(value)["center_separation"]
        ),
    }
    return {
        name: np.asarray(
            [
                [extractor(value) for value in row]
                for row in metrics_by_time
            ]
        )
        for name, extractor in extractors.items()
    }


def _padded_fidelity(left: np.ndarray, right: np.ndarray) -> float:
    dimension = max(left.size, right.size)
    padded_left = np.zeros(dimension, dtype=complex)
    padded_right = np.zeros(dimension, dtype=complex)
    padded_left[: left.size] = left
    padded_right[: right.size] = right
    return float(abs(np.vdot(padded_left, padded_right)) ** 2)


def _write_plot(
    path: Path,
    times: np.ndarray,
    metric_arrays: dict[str, np.ndarray],
    worst_conditional: ConditionalRelativeState,
    worst_metrics: ConditionalPacketMetrics,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.35))
    colors = ("#6E3CBC", "#2378B5", "#44A047", "#D97824")
    for electronic_index, label in enumerate(ELECTRON_CONFIGURATION_LABELS):
        short_label = label.replace("_site_", "").replace("_", " ")
        axes[0].plot(
            times,
            metric_arrays["single_gaussian_fidelity"][:, electronic_index],
            color=colors[electronic_index],
            linestyle="--",
            alpha=0.8,
        )
        axes[0].plot(
            times,
            metric_arrays["two_coherent_fidelity"][:, electronic_index],
            color=colors[electronic_index],
            label=short_label,
        )
        axes[1].plot(
            times,
            metric_arrays["gaussian_non_gaussianity"][
                :, electronic_index
            ],
            color=colors[electronic_index],
            label=short_label,
        )
    axes[0].plot([], [], color="#333333", linestyle="--", label="one Gaussian")
    axes[0].plot([], [], color="#333333", label="two coherent packets")
    axes[0].set_ylim(0.90, 1.002)
    axes[0].set_xlabel(r"dimensionless time $t\,t_{\rm hop}$")
    axes[0].set_ylabel("conditional-state fidelity")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=6.4, ncol=2)

    axes[1].set_xlabel(r"dimensionless time $t\,t_{\rm hop}$")
    axes[1].set_ylabel("relative-entropy non-Gaussianity")
    axes[1].grid(alpha=0.22)

    _, _, grid, q_values = husimi_peaks(
        worst_conditional.dominant_state,
        grid_points=121,
    )
    image = axes[2].contourf(grid, grid, q_values, levels=28, cmap="magma")
    geometry = _packet_geometry(worst_metrics)
    centers = np.asarray(geometry["centers"])
    axes[2].scatter(
        centers[:, 0],
        centers[:, 1],
        marker="x",
        color="#36E1F2",
        linewidths=1.5,
        label="fitted coherent centers",
    )
    axes[2].set_aspect("equal")
    axes[2].set_xlabel(r"$\operatorname{Re}\alpha_-$")
    axes[2].set_ylabel(r"$\operatorname{Im}\alpha_-$")
    axes[2].set_title(
        "worst one-Gaussian case\n"
        f"electron block {worst_metrics.electronic_index}",
        fontsize=8,
    )
    legend = axes[2].legend(
        frameon=False,
        fontsize=6.5,
        loc="upper right",
    )
    for label in legend.get_texts():
        label.set_color("white")
    figure.colorbar(image, ax=axes[2], fraction=0.047, pad=0.03, label="Husimi Q")
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def run_conditional_packet_analysis(
    run_directory: Path,
    *,
    parameters: DimerParameters,
    sample_times: tuple[float, ...] = (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
    ),
    phonon_cutoff: int = 20,
    convergence_cutoffs: tuple[int, ...] = (12, 16, 20),
    convergence_times: tuple[float, ...] = (0.0, 2.0, 4.0),
    eigensolver_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-12,
    maximum_step: float = 0.01,
    single_gaussian_random_starts: int = 2,
    single_gaussian_maximum_iterations: int = 350,
    two_packet_maximum_iterations: int = 100,
    two_packet_population_size: int = 9,
    husimi_grid_points: int = 81,
    spin_exchange_reuse_tolerance: float = 1e-8,
    packet_infidelity_threshold: float = 0.02,
    cutoff_metric_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Test one-Gaussian and two-coherent compression of exact blocks."""

    times = _validated_times(sample_times, name="sample_times")
    convergence_time_array = _validated_times(
        convergence_times,
        name="convergence_times",
    )
    if phonon_cutoff not in convergence_cutoffs:
        raise ValueError("convergence_cutoffs must include phonon_cutoff")
    if not np.all(
        [np.any(np.isclose(times, time)) for time in convergence_time_array]
    ):
        raise ValueError("convergence_times must be present in sample_times")
    if packet_infidelity_threshold <= 0.0:
        raise ValueError("packet_infidelity_threshold must be positive")
    if cutoff_metric_tolerance <= 0.0:
        raise ValueError("cutoff_metric_tolerance must be positive")
    run_directory.mkdir(parents=True, exist_ok=True)

    baseline = exact_holstein_wavefunction_trajectory_for_diagnostics(
        parameters,
        sample_times=times,
        phonon_cutoff=phonon_cutoff,
        eigensolver_tolerance=eigensolver_tolerance,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_step=maximum_step,
    )
    baseline_conditionals: list[tuple[ConditionalRelativeState, ...]] = []
    baseline_metrics: list[tuple[ConditionalPacketMetrics, ...]] = []
    baseline_records: list[dict[str, Any]] = []
    exchange_diagnostics: list[dict[str, Any]] = []
    for time_index, time in enumerate(baseline.times):
        conditional, metrics, reused, exchange = _analyze_wavefunction(
            baseline.state_vectors[:, time_index],
            phonon_cutoff=phonon_cutoff,
            seed=101 + 101 * time_index,
            single_gaussian_random_starts=single_gaussian_random_starts,
            single_gaussian_maximum_iterations=(
                single_gaussian_maximum_iterations
            ),
            two_packet_maximum_iterations=two_packet_maximum_iterations,
            two_packet_population_size=two_packet_population_size,
            husimi_grid_points=husimi_grid_points,
            spin_exchange_reuse_tolerance=spin_exchange_reuse_tolerance,
        )
        baseline_conditionals.append(conditional)
        baseline_metrics.append(metrics)
        baseline_records.append(
            {
                "time": float(time),
                "blocks": [
                    _metrics_record(
                        value,
                        fit_reused_from_index=reused[index],
                    )
                    for index, value in enumerate(metrics)
                ],
            }
        )
        exchange_diagnostics.append(
            {"time": float(time), **exchange}
        )
    metrics_tuple = tuple(baseline_metrics)
    baseline_arrays = _metric_arrays(metrics_tuple)

    convergence_data: dict[
        int,
        tuple[
            tuple[tuple[ConditionalRelativeState, ...], ...],
            tuple[tuple[ConditionalPacketMetrics, ...], ...],
        ],
    ] = {}
    for cutoff_index, cutoff in enumerate(convergence_cutoffs):
        if cutoff == phonon_cutoff:
            sample_indices = tuple(
                int(np.flatnonzero(np.isclose(times, time))[0])
                for time in convergence_time_array
            )
            cutoff_conditionals = tuple(
                baseline_conditionals[index] for index in sample_indices
            )
            cutoff_metrics = tuple(
                baseline_metrics[index] for index in sample_indices
            )
        else:
            trajectory = (
                exact_holstein_wavefunction_trajectory_for_diagnostics(
                    parameters,
                    sample_times=convergence_time_array,
                    phonon_cutoff=cutoff,
                    eigensolver_tolerance=eigensolver_tolerance,
                    relative_tolerance=relative_tolerance,
                    absolute_tolerance=absolute_tolerance,
                    maximum_step=maximum_step,
                )
            )
            conditional_rows: list[tuple[ConditionalRelativeState, ...]] = []
            metric_rows: list[tuple[ConditionalPacketMetrics, ...]] = []
            for time_index in range(convergence_time_array.size):
                conditional, metrics, _, _ = _analyze_wavefunction(
                    trajectory.state_vectors[:, time_index],
                    phonon_cutoff=cutoff,
                    seed=10007 * (cutoff_index + 1) + 101 * time_index,
                    single_gaussian_random_starts=(
                        single_gaussian_random_starts
                    ),
                    single_gaussian_maximum_iterations=(
                        single_gaussian_maximum_iterations
                    ),
                    two_packet_maximum_iterations=(
                        two_packet_maximum_iterations
                    ),
                    two_packet_population_size=two_packet_population_size,
                    husimi_grid_points=husimi_grid_points,
                    spin_exchange_reuse_tolerance=(
                        spin_exchange_reuse_tolerance
                    ),
                )
                conditional_rows.append(conditional)
                metric_rows.append(metrics)
            cutoff_conditionals = tuple(conditional_rows)
            cutoff_metrics = tuple(metric_rows)
        convergence_data[cutoff] = (cutoff_conditionals, cutoff_metrics)

    reference_conditionals, reference_metrics = convergence_data[phonon_cutoff]
    reference_arrays = _metric_arrays(reference_metrics)
    convergence_summary: dict[str, Any] = {}
    convergence_arrays: dict[str, np.ndarray] = {}
    maximum_nonreference_metric_delta = 0.0
    for cutoff in convergence_cutoffs:
        cutoff_conditionals, cutoff_metrics = convergence_data[cutoff]
        arrays = _metric_arrays(cutoff_metrics)
        fidelity_to_reference = np.asarray(
            [
                [
                    _padded_fidelity(
                        cutoff_conditionals[time_index][electronic_index]
                        .dominant_state,
                        reference_conditionals[time_index][electronic_index]
                        .dominant_state,
                    )
                    for electronic_index in range(4)
                ]
                for time_index in range(convergence_time_array.size)
            ]
        )
        metric_deltas = {
            name: float(np.max(np.abs(arrays[name] - reference_arrays[name])))
            for name in (
                "probability",
                "gaussian_non_gaussianity",
                "single_gaussian_fidelity",
                "two_coherent_fidelity",
            )
        }
        if cutoff != phonon_cutoff:
            maximum_nonreference_metric_delta = max(
                maximum_nonreference_metric_delta,
                *metric_deltas.values(),
            )
        convergence_summary[str(cutoff)] = {
            "minimum_conditional_state_fidelity_to_reference": float(
                np.min(fidelity_to_reference)
            ),
            "maximum_metric_deltas_to_reference": metric_deltas,
        }
        convergence_arrays[f"cutoff_{cutoff}_state_fidelity"] = (
            fidelity_to_reference
        )
        for name, values in arrays.items():
            convergence_arrays[f"cutoff_{cutoff}_{name}"] = values

    single_fidelity = baseline_arrays["single_gaussian_fidelity"]
    two_fidelity = baseline_arrays["two_coherent_fidelity"]
    worst_single_flat = int(np.argmin(single_fidelity))
    worst_time_index, worst_electronic_index = np.unravel_index(
        worst_single_flat,
        single_fidelity.shape,
    )
    worst_single_infidelity = float(1.0 - np.min(single_fidelity))
    worst_two_infidelity = float(1.0 - np.min(two_fidelity))
    maximum_fidelity_gain = float(np.max(two_fidelity - single_fidelity))
    maximum_non_gaussianity = float(
        np.max(baseline_arrays["gaussian_non_gaussianity"])
    )
    maximum_peak_count = int(np.max(baseline_arrays["husimi_peak_count"]))
    minimum_factorization = float(
        np.min(baseline_arrays["center_relative_factorization"])
    )
    minimum_purity = float(np.min(baseline_arrays["relative_purity"]))
    two_packet_passed = (
        worst_two_infidelity <= packet_infidelity_threshold
    )
    nonreference_cutoffs = sorted(
        cutoff for cutoff in convergence_cutoffs if cutoff != phonon_cutoff
    )
    convergence_decision_cutoff = (
        nonreference_cutoffs[-1] if nonreference_cutoffs else phonon_cutoff
    )
    decision_metric_delta = max(
        convergence_summary[str(convergence_decision_cutoff)][
            "maximum_metric_deltas_to_reference"
        ].values()
    )
    cutoff_passed = decision_metric_delta <= cutoff_metric_tolerance
    single_gaussian_sufficient = (
        worst_single_infidelity <= packet_infidelity_threshold
    )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "classification": "exploratory_local_not_promoted",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            **asdict(parameters),
            "coupling": parameters.coupling,
            "phonon_cutoff": phonon_cutoff,
            "sample_times": times.tolist(),
            "maximum_step": maximum_step,
            "relative_tolerance": relative_tolerance,
            "absolute_tolerance": absolute_tolerance,
        },
        "representation_tested": {
            "normal_mode_factorization": (
                "exact center mode times an electron-conditioned relative mode"
            ),
            "one_packet_family": "one displaced-squeezed Gaussian",
            "two_packet_family": (
                "optimal linear span of two normalized coherent states"
            ),
            "exact_reference_usage": (
                "offline compression and cutoff gate only; not an autonomous RHS"
            ),
        },
        "time_resolved_metrics": baseline_records,
        "spin_exchange_diagnostics": exchange_diagnostics,
        "aggregate_metrics": {
            "minimum_center_relative_factorization": minimum_factorization,
            "minimum_conditional_relative_purity": minimum_purity,
            "maximum_gaussian_non_gaussianity": maximum_non_gaussianity,
            "worst_single_gaussian_infidelity": worst_single_infidelity,
            "worst_two_coherent_infidelity": worst_two_infidelity,
            "maximum_two_packet_fidelity_gain": maximum_fidelity_gain,
            "maximum_resolved_husimi_peak_count": maximum_peak_count,
            "worst_single_gaussian_case": {
                "time": float(times[worst_time_index]),
                "electronic_index": int(worst_electronic_index),
                "configuration": ELECTRON_CONFIGURATION_LABELS[
                    worst_electronic_index
                ],
            },
        },
        "cutoff_convergence": {
            "reference_cutoff": phonon_cutoff,
            "times": convergence_time_array.tolist(),
            "cutoffs": convergence_summary,
            "maximum_nonreference_metric_delta": (
                maximum_nonreference_metric_delta
            ),
            "decision_cutoff": convergence_decision_cutoff,
            "decision_metric_delta": decision_metric_delta,
            "decision_rule": (
                "the highest available nonreference cutoff must agree with "
                "the reference; lower cutoffs remain coarse controls"
            ),
        },
        "validation_gate": {
            "packet_infidelity_threshold": packet_infidelity_threshold,
            "cutoff_metric_tolerance": cutoff_metric_tolerance,
            "single_gaussian_sufficient": bool(single_gaussian_sufficient),
            "two_coherent_compression_passed": bool(two_packet_passed),
            "cutoff_convergence_passed": bool(cutoff_passed),
            "multiple_resolved_husimi_peaks_observed": bool(
                maximum_peak_count > 1
            ),
            "autonomous_propagation_authorized": False,
            "decision": (
                "test an autonomous two-coherent electron-relative tangent "
                "representation against exact instantaneous velocities"
                if two_packet_passed and cutoff_passed
                else (
                    "do not promote a two-coherent representation; improve "
                    "the packet family or cutoff evidence first"
                )
            ),
        },
    }

    prefix = "conditional_relative_packet_gate"
    trajectory_path = run_directory / f"{prefix}.npz"
    np.savez_compressed(
        trajectory_path,
        times=times,
        **baseline_arrays,
        **convergence_arrays,
    )
    plot_path = run_directory / f"{prefix}.png"
    _write_plot(
        plot_path,
        times,
        baseline_arrays,
        baseline_conditionals[worst_time_index][worst_electronic_index],
        baseline_metrics[worst_time_index][worst_electronic_index],
    )
    summary_path = run_directory / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("conditional_packets.py"),
        Path(__file__).with_name("exact_reference.py"),
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
    (run_directory / "runtime_manifest.json").write_text(
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
    parser.add_argument("--phonon-cutoff", type=int, default=20)
    parser.add_argument("--maximum-step", type=float, default=0.01)
    parser.add_argument("--two-packet-maximum-iterations", type=int, default=100)
    parser.add_argument("--two-packet-population-size", type=int, default=9)
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = run_conditional_packet_analysis(
        args.run_directory,
        parameters=DimerParameters(
            lambda_ep=args.lambda_ep,
            gamma=args.gamma,
            drive_amplitude=args.drive,
        ),
        phonon_cutoff=args.phonon_cutoff,
        maximum_step=args.maximum_step,
        two_packet_maximum_iterations=(
            args.two_packet_maximum_iterations
        ),
        two_packet_population_size=args.two_packet_population_size,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
