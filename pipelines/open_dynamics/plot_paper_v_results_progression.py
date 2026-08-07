"""Build matched observable figures for the Paper V results progression.

The command reads completed trajectory and source-audit files only.  It does
not propagate a state or alter any existing numerical result.  The exact
overlay is the stored cutoff-16 DOP853 trajectory; the midpoint reference is
used only to report the remaining exact-solver numerical discrepancy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy.integrate import solve_ivp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from paper5.stability.adaptive_positive_moment import (
    RAW_MOMENT_COORDINATE_NAMES,
    raw_moment_coordinates_to_matrix_state,
)
from paper5.stability.hubbard_dimer import DimerParameters, GaussianSineDrive
from paper5.stability.matrix_reference import (
    closed_scalar_rhs,
    closed_scalar_to_matrix_state,
    matrix_state_to_closed_scalar_coordinates,
)
from pipelines.open_dynamics.run_archive_long_horizon_observables import (
    _energy_components,
)


ROOT = Path(__file__).resolve().parents[2]
SCORE_ARRAYS = ROOT / (
    "output/local_runs/"
    "paper_v_multi_coherent_double_pulse_sealed_score_cutoff16_20260804_v1/"
    "score_arrays.npz"
)
TRAJECTORIES = {
    6: ROOT / (
        "output/local_runs/"
        "paper_v_multi_coherent_double_pulse_blind_model_cutoff16_20260804_v1/"
        "fine_central/segmented_horizon.npz"
    ),
    8: ROOT / (
        "output/local_runs/"
        "paper_v_multi_coherent_capacity_k8_t40_20260804_v1/"
        "fine_central/segmented_horizon.npz"
    ),
    10: ROOT / (
        "output/local_runs/"
        "paper_v_multi_coherent_capacity_k10_t40_20260804_v1/"
        "fine_central/segmented_horizon.npz"
    ),
    12: ROOT / (
        "output/local_runs/"
        "paper_v_multi_coherent_capacity_k12_t40_20260804_v1/"
        "fine_central/segmented_horizon.npz"
    ),
}
TRAJECTORY_SUMMARIES = {
    packet_count: path.with_name("summary.json")
    for packet_count, path in TRAJECTORIES.items()
}
SOURCE_SUMMARIES = {
    6: ROOT / (
        "output/local_runs/"
        "paper_v_packet_derived_closure_source_k6_central_t40_20260804_v2/"
        "summary.json"
    ),
    8: ROOT / (
        "output/local_runs/"
        "paper_v_packet_derived_closure_source_k8_central_t40_20260804_v3/"
        "summary.json"
    ),
    10: ROOT / (
        "output/local_runs/"
        "paper_v_packet_derived_closure_source_k10_central_t40_20260804_v2/"
        "summary.json"
    ),
    12: ROOT / (
        "output/local_runs/"
        "paper_v_packet_derived_closure_source_k12_central_t40_20260804_v2/"
        "summary.json"
    ),
}
MIXED_GUIDED_RUNS = {
    "guided K6 at t=0": ROOT / (
        "output/local_runs/"
        "paper_v_mixed_guided_packet_k6_cutoff16_t20_direct_20260805_v2/"
        "mixed_guided_packet_rollout.npz"
    ),
    "readmit K7 at t=8": ROOT / (
        "output/local_runs/"
        "paper_v_mixed_guided_packet_k7_readmit_t8_cutoff16_t20_direct_20260805_v2/"
        "mixed_guided_packet_rollout.npz"
    ),
    "readmit K8 at t=8": ROOT / (
        "output/local_runs/"
        "paper_v_mixed_guided_packet_k8_readmit_t8_cutoff16_t20_direct_20260805_v2/"
        "mixed_guided_packet_rollout.npz"
    ),
    "adaptive K6->7->8": ROOT / (
        "output/local_runs/"
        "paper_v_archive_gram_adaptive_packet_cutoff16_t20_direct_20260805_v2/"
        "mixed_guided_packet_rollout.npz"
    ),
}
TRANSFER_RUNS = {
    "plus preparation, double pulse": ROOT / (
        "output/local_runs/"
        "paper_v_archive_gram_adaptive_packet_plus_cutoff16_t20_direct_20260805_v2/"
        "mixed_guided_packet_rollout.npz"
    ),
    "minus preparation, double pulse": ROOT / (
        "output/local_runs/"
        "paper_v_archive_gram_adaptive_packet_minus_cutoff16_t20_direct_20260805_v2/"
        "mixed_guided_packet_rollout.npz"
    ),
    "central preparation, single pulse": ROOT / (
        "output/local_runs/"
        "paper_v_archive_gram_adaptive_packet_single_pulse_cutoff16_t20_direct_20260805_v2/"
        "mixed_guided_packet_rollout.npz"
    ),
}
ADAPTIVE_T40 = ROOT / (
    "output/local_runs/"
    "paper_v_archive_gram_adaptive_packet_cutoff16_t40_direct_20260805_v2/"
    "mixed_guided_packet_rollout.npz"
)
ADAPTIVE_T98_DIRECTORY = ROOT / (
    "output/local_runs/"
    "paper_v_archive_gram_adaptive_packet_cutoff16_t98_user_stopped_20260805_v1"
)
ADAPTIVE_T98 = ADAPTIVE_T98_DIRECTORY / "mixed_guided_packet_rollout.npz"
ADAPTIVE_T98_SUMMARY = ADAPTIVE_T98_DIRECTORY / "summary.json"
APCM_MATCHED_RUNS = {
    "higher-moment: no cones": ROOT / (
        "output/local_runs/"
        "paper_v_apcm_ablation_C_prior_no_controller_t4_h0025_20260805_v2/"
        "trajectory.npz"
    ),
    "higher-moment + retained cone": ROOT / (
        "output/local_runs/"
        "paper_v_apcm_ablation_B_prior_controller_t4_h0025_20260805_v2/"
        "trajectory.npz"
    ),
    "higher-moment + M4 cone": ROOT / (
        "output/local_runs/"
        "paper_v_apcm_ablation_D_positive_no_controller_t4_h0025_20260805_v1/"
        "trajectory.npz"
    ),
    "higher-moment + both cones": ROOT / (
        "output/local_runs/"
        "paper_v_apcm_strong_t4_h0025_20260805_v1/trajectory.npz"
    ),
}
APCM_T20_RUNS = {
    "higher moment + both cones": ROOT / (
        "output/local_runs/"
        "paper_v_apcm_spin_exchange_blocks_controller_t20_h0025_20260805_v1/"
        "trajectory.npz"
    ),
    "higher moment + M4 cone only": ROOT / (
        "output/local_runs/"
        "paper_v_apcm_spin_exchange_blocks_no_controller_t20_h0025_20260805_v1/"
        "trajectory.npz"
    ),
}
APCM_T31_CHECKPOINT = ROOT / (
    "output/local_runs/"
    "paper_v_apcm_spin_exchange_blocks_controller_t240_from_t20_h0025_20260805_v1/"
    "checkpoint.npz"
)
APCM_T31_STORED_REFERENCE = ROOT / (
    "output/plots/paper_v_results_progression_20260804/"
    "apcm_t31_checkpoint_score.npz"
)
ARCHIVE_LONG_RAW_REFINED = ROOT / (
    "output/local_runs/"
    "paper_v_archive_observable_trajectories_t1000_20260803_v1/"
    "raw_refined_rk4_dt005_trajectory.npz"
)
ARCHIVE_ABLATION = ROOT / (
    "output/local_runs/"
    "paper_v_autonomous_pauli_repair_ablation_20260803_v2/trajectories.npz"
)

OBSERVABLE_NAMES = (
    "site_0_occupation",
    "site_1_occupation",
    "electronic_energy",
    "phonon_energy",
    "electron_phonon_energy",
    "internal_total_energy",
)
PANEL_TITLES = (
    r"site 0 occupation, $n_0=2\rho_{00}$",
    r"site 1 occupation, $n_1=2\rho_{11}$",
    "electronic energy",
    "phonon energy",
    "electron-phonon energy",
    "total internal energy",
)
COLORS = {
    "exact": "#171717",
    6: "#4c78a8",
    8: "#f58518",
    10: "#54a24b",
    12: "#b279a2",
}
LINESTYLES = {"exact": "-", 6: ":", 8: "--", 10: "-", 12: "-."}
RAW_ARCHIVE_COLOR = "#9c3b2e"
RAW_ARCHIVE_STYLE = (0, (5, 2))
RAW_ARCHIVE_ALPHA = 0.58
RAW_ARCHIVE_LINEWIDTH = 0.75


class _DrivenParameters:
    """Use the common dimer parameters with a declared pulse sequence."""

    def __init__(
        self,
        parameters: DimerParameters,
        delays: tuple[float, ...],
    ) -> None:
        self._parameters = parameters
        self._drive = GaussianSineDrive.from_parameters(
            parameters,
            delays=delays,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._parameters, name)

    def drive_difference(self, time: float) -> float:
        return self._drive.difference(time)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _observables(
    times: np.ndarray,
    coordinates: np.ndarray,
    parameters: DimerParameters,
) -> np.ndarray:
    path = np.asarray(coordinates, dtype=float)
    if path.shape != (times.size, 31):
        raise ValueError("closed coordinates must have shape (times, 31)")
    result = np.empty((times.size, len(OBSERVABLE_NAMES)), dtype=float)
    for index, (time_value, row) in enumerate(
        zip(times, path, strict=True)
    ):
        state = closed_scalar_to_matrix_state(row)
        energy = _energy_components(state, parameters, float(time_value))
        result[index] = (
            2.0 * float(state.electron_density[0, 0].real),
            2.0 * float(state.electron_density[1, 1].real),
            float(energy[0]),
            float(energy[1]),
            float(energy[2]),
            float(energy[3]),
        )
    return result


def _raw_archive_observables(
    times: np.ndarray,
    initial_coordinates: np.ndarray,
    *,
    pulse_delays: tuple[float, ...],
) -> tuple[np.ndarray, float | None]:
    """Propagate the matched raw 31-coordinate archive EOM for plotting."""

    grid = np.asarray(times, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or not np.isclose(grid[0], 0.0):
        raise ValueError("raw archive reference requires a time grid from zero")
    parameters = DimerParameters(
        hopping=1.0,
        gamma=0.5,
        lambda_ep=1.5,
        drive_amplitude=1.0,
        pulse_width=1.0,
    )
    driven = _DrivenParameters(parameters, pulse_delays)

    def threshold_event(time: float, state: np.ndarray) -> float:
        del time
        return 1.0e4 - float(np.max(np.abs(state)))

    threshold_event.terminal = True  # type: ignore[attr-defined]
    threshold_event.direction = -1.0  # type: ignore[attr-defined]
    solution = solve_ivp(
        lambda time, state: closed_scalar_rhs(
            float(time), state, driven  # type: ignore[arg-type]
        ),
        (0.0, float(grid[-1])),
        np.asarray(initial_coordinates, dtype=float),
        method="DOP853",
        t_eval=grid,
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=min(0.05, float(np.min(np.diff(grid)))),
        events=threshold_event,
    )
    if not solution.success:
        raise RuntimeError(f"raw archive reference failed: {solution.message}")
    coordinates = np.full((grid.size, 31), np.nan, dtype=float)
    coordinates[: solution.t.size] = np.asarray(solution.y.T, dtype=float)
    values = np.full((grid.size, len(OBSERVABLE_NAMES)), np.nan, dtype=float)
    values[: solution.t.size] = _observables(
        grid[: solution.t.size],
        coordinates[: solution.t.size],
        parameters,
    )
    failure_time = (
        None
        if not solution.t_events or solution.t_events[0].size == 0
        else float(solution.t_events[0][0])
    )
    return values, failure_time


def _time_rms(times: np.ndarray, values: np.ndarray) -> float:
    duration = float(times[-1] - times[0])
    return float(np.sqrt(np.trapezoid(values**2, times) / duration))


def _closed_coordinate_error_metrics(
    exact_coordinates: np.ndarray,
    approximate_coordinates: np.ndarray,
) -> dict[str, float]:
    """Return the registered scalar and correlation-block trajectory errors."""

    difference = np.asarray(approximate_coordinates, dtype=float) - np.asarray(
        exact_coordinates, dtype=float
    )
    correlation_errors = []
    for exact_row, approximate_row in zip(
        exact_coordinates, approximate_coordinates, strict=True
    ):
        exact_state = closed_scalar_to_matrix_state(exact_row)
        approximate_state = closed_scalar_to_matrix_state(approximate_row)
        correlation_errors.append(
            np.linalg.norm(
                approximate_state.electron_phonon_correlation
                - exact_state.electron_phonon_correlation
            )
        )
    return {
        "all_coordinate_scalar_rms": float(np.sqrt(np.mean(difference**2))),
        "C_block_time_rms_frobenius": float(
            np.sqrt(np.mean(np.asarray(correlation_errors, dtype=float) ** 2))
        ),
    }


def _cumulative_rms(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    squared = np.asarray(values, dtype=float) ** 2
    increments = 0.5 * (squared[:-1] + squared[1:]) * np.diff(times)
    integral = np.concatenate(([0.0], np.cumsum(increments)))
    result = np.zeros_like(times)
    result[1:] = np.sqrt(integral[1:] / (times[1:] - times[0]))
    return result


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.titlesize": 8.4,
            "axes.labelsize": 8.0,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.linewidth": 0.75,
            "lines.solid_capstyle": "round",
            "savefig.facecolor": "white",
        }
    )


def _set_primary_limits(
    axis: plt.Axes,
    values: list[np.ndarray],
) -> None:
    """Keep comparison trajectories readable when the archive baseline blows up."""

    finite = np.concatenate(
        [np.asarray(value, dtype=float)[np.isfinite(value)] for value in values]
    )
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    span = upper - lower
    padding = 0.08 * span if span > 0.0 else 0.08 * max(1.0, abs(lower))
    axis.set_ylim(lower - padding, upper + padding)


def _decorate_axis(axis: plt.Axes, panel: int) -> None:
    axis.set_title(f"({chr(97 + panel)}) {PANEL_TITLES[panel]}", loc="left")
    axis.set_xlim(0.0, 40.0)
    axis.grid(alpha=0.18, linewidth=0.5)
    for start in (0.0, 8.0):
        axis.axvspan(start, start + 4.0, color="#d8e8f1", alpha=0.32, lw=0.0)
    if panel in (0, 1):
        axis.set_ylabel("occupation")
    else:
        axis.set_ylabel(r"energy / $t_{\rm hop}$")
    if panel >= 3:
        axis.set_xlabel(r"$t\,t_{\rm hop}$")


def _save_figure(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(figure)


def _plot_observables(
    times: np.ndarray,
    exact: np.ndarray,
    raw_archive: np.ndarray,
    packet: dict[int, np.ndarray],
    output_directory: Path,
) -> None:
    _style()
    figure, axes = plt.subplots(2, 3, figsize=(7.25, 4.65))
    for panel, axis in enumerate(axes.flat):
        axis.plot(
            times,
            exact[:, panel],
            color=COLORS["exact"],
            linestyle=LINESTYLES["exact"],
            linewidth=1.45,
            label="exact cutoff-16",
            zorder=5,
        )
        axis.plot(
            times,
            raw_archive[:, panel],
            color=RAW_ARCHIVE_COLOR,
            linestyle=RAW_ARCHIVE_STYLE,
            linewidth=RAW_ARCHIVE_LINEWIDTH,
            alpha=RAW_ARCHIVE_ALPHA,
            label="raw archive EOM",
            zorder=1,
        )
        for packet_count in sorted(packet):
            axis.plot(
                times,
                packet[packet_count][:, panel],
                color=COLORS[packet_count],
                linestyle=LINESTYLES[packet_count],
                linewidth=1.0,
                alpha=0.96,
                label=rf"$K_{{\max}}={packet_count}$",
            )
        _set_primary_limits(
            axis,
            [exact[:, panel], *[values[:, panel] for values in packet.values()]],
        )
        _decorate_axis(axis, panel)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.05,
        handlelength=2.3,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.10,
        top=0.89,
        hspace=0.42,
        wspace=0.34,
    )
    _save_figure(figure, output_directory / "packet_capacity_observables")


def _plot_apcm_matched_observables(
    output_directory: Path,
) -> dict[str, Any]:
    """Plot the matched cone ablation and original archive baselines."""

    parameters = DimerParameters(
        hopping=1.0,
        gamma=0.5,
        lambda_ep=1.5,
        drive_amplitude=1.0,
    )
    times: np.ndarray | None = None
    exact_coordinates: np.ndarray | None = None
    paths: dict[str, np.ndarray] = {}
    apcm_summaries: dict[str, dict[str, Any]] = {}
    completion_minima: dict[str, float] = {}
    joint_gram_minima: dict[str, float] = {}
    maximum_exact_disagreement = 0.0
    for label, path in APCM_MATCHED_RUNS.items():
        apcm_summaries[label] = json.loads(
            path.with_name("summary.json").read_text(encoding="utf-8")
        )
        with np.load(path, allow_pickle=False) as arrays:
            local_times = np.asarray(arrays["times"], dtype=float)
            local_exact = np.asarray(
                arrays["exact_archive_coordinates"], dtype=float
            )
            if times is None:
                times = local_times
                exact_coordinates = local_exact
            else:
                if not np.array_equal(local_times, times):
                    raise ValueError(f"APCM time grid mismatch for {label}")
                maximum_exact_disagreement = max(
                    maximum_exact_disagreement,
                    float(np.max(np.abs(local_exact - exact_coordinates))),
                )
            paths[label] = np.asarray(
                arrays["approximate_archive_coordinates"], dtype=float
            )
            completion_minima[label] = float(
                np.nanmin(arrays["completion_minimum_eigenvalues"])
            )
            joint_gram_minima[label] = float(
                np.min(arrays["joint_gram_minimum_eigenvalues"])
            )
    if times is None or exact_coordinates is None:
        raise RuntimeError("APCM matched comparison has no trajectories")

    with np.load(ARCHIVE_ABLATION, allow_pickle=False) as arrays:
        archive_times = np.asarray(arrays["times"], dtype=float)
        archive_exact = np.asarray(arrays["exact_coordinates"], dtype=float)
        archive_paths = {
            "31-coordinate raw EOM": np.asarray(
                arrays["raw_coordinates"], dtype=float
            ),
            "31-coordinate + retained cone": np.asarray(
                arrays["controller_coordinates"], dtype=float
            ),
        }
        archive_joint_gram_minima = {
            "31-coordinate raw EOM": float(
                np.min(arrays["raw_joint_gram_minimum_eigenvalue"])
            ),
            "31-coordinate + retained cone": float(
                np.min(arrays["controller_joint_gram_minimum_eigenvalue"])
            ),
        }
    archive_exact_observables = _observables(
        archive_times, archive_exact, parameters
    )
    archive_native_observables = {
        label: _observables(archive_times, coordinates, parameters)
        for label, coordinates in archive_paths.items()
    }
    archive_native_trajectory_errors = {
        label: _closed_coordinate_error_metrics(archive_exact, coordinates)
        for label, coordinates in archive_paths.items()
    }
    for label, values in archive_native_observables.items():
        archive_native_trajectory_errors[label][
            "site_0_occupation_time_rms"
        ] = _time_rms(
            archive_times, values[:, 0] - archive_exact_observables[:, 0]
        )
        archive_native_trajectory_errors[label][
            "internal_energy_time_rms"
        ] = _time_rms(
            archive_times, values[:, 5] - archive_exact_observables[:, 5]
        )
    if not np.isclose(archive_times[0], times[0]) or not np.isclose(
        archive_times[-1], times[-1]
    ):
        raise ValueError("archive and APCM comparisons cover different intervals")
    if not np.allclose(
        archive_exact[0], exact_coordinates[0], rtol=0.0, atol=3e-13
    ):
        raise ValueError("archive and APCM comparisons use different initial states")
    shared_indices = np.searchsorted(times, archive_times)
    if not np.array_equal(times[shared_indices], archive_times):
        raise ValueError("archive grid is not nested in the APCM grid")
    maximum_exact_disagreement = max(
        maximum_exact_disagreement,
        float(
            np.max(
                np.abs(
                    archive_exact - exact_coordinates[shared_indices]
                )
            )
        ),
    )
    interpolated_archive_paths = {
        label: np.column_stack(
            [
                np.interp(times, archive_times, coordinates[:, column])
                for column in range(coordinates.shape[1])
            ]
        )
        for label, coordinates in archive_paths.items()
    }
    paths = {**interpolated_archive_paths, **paths}
    joint_gram_minima = {
        "exact cutoff-16 Hamiltonian": apcm_summaries[
            "higher-moment + both cones"
        ][
            "feasibility"
        ]["exact_minimum_joint_gram_eigenvalue"],
        **archive_joint_gram_minima,
        **joint_gram_minima,
    }

    exact = _observables(times, exact_coordinates, parameters)
    approximate = {
        label: _observables(times, coordinates, parameters)
        for label, coordinates in paths.items()
    }
    colors = {
        "31-coordinate raw EOM": "#9c3b2e",
        "31-coordinate + retained cone": "#7f7f7f",
        "higher-moment: no cones": "#e45756",
        "higher-moment + retained cone": "#f58518",
        "higher-moment + M4 cone": "#54a24b",
        "higher-moment + both cones": "#4c78a8",
    }
    styles = {
        "31-coordinate raw EOM": (0, (5, 2)),
        "31-coordinate + retained cone": "-.",
        "higher-moment: no cones": ":",
        "higher-moment + retained cone": "--",
        "higher-moment + M4 cone": (0, (3, 1, 1, 1)),
        "higher-moment + both cones": "-",
    }

    _style()
    figure, axes = plt.subplots(2, 3, figsize=(7.25, 4.65))
    for panel, axis in enumerate(axes.flat):
        axis.plot(
            times,
            exact[:, panel],
            color="#171717",
            linewidth=1.5,
            label="exact cutoff-16 Hamiltonian",
            zorder=5,
        )
        for label, values in approximate.items():
            axis.plot(
                times,
                values[:, panel],
                color=colors[label],
                linestyle=styles[label],
                linewidth=0.9 if label.startswith("31-coordinate") else 1.05,
                alpha=0.78 if label.startswith("31-coordinate") else 0.96,
                label=label,
            )
        _set_primary_limits(
            axis,
            [
                exact[:, panel],
                *[values[:, panel] for values in approximate.values()],
            ],
        )
        axis.set_title(
            f"({chr(97 + panel)}) {PANEL_TITLES[panel]}", loc="left"
        )
        axis.set_xlim(0.0, 4.0)
        axis.grid(alpha=0.18, linewidth=0.5)
        if panel in (0, 1):
            axis.set_ylabel("occupation")
        else:
            axis.set_ylabel(r"energy / $t_{\rm hop}$")
        if panel >= 3:
            axis.set_xlabel(r"$t\,t_{\rm hop}$")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=0.75,
        handlelength=2.0,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.10,
        top=0.81,
        hspace=0.42,
        wspace=0.34,
    )
    _save_figure(figure, output_directory / "apcm_t4_matched_observables")

    trajectory_errors = {
        label: _closed_coordinate_error_metrics(exact_coordinates, coordinates)
        for label, coordinates in paths.items()
    }
    for label, values in approximate.items():
        trajectory_errors[label]["site_0_occupation_time_rms"] = _time_rms(
            times, values[:, 0] - exact[:, 0]
        )
        trajectory_errors[label]["internal_energy_time_rms"] = _time_rms(
            times, values[:, 5] - exact[:, 5]
        )
    trajectory_errors.update(archive_native_trajectory_errors)
    observable_errors = {
        label: {
            name: {
                "time_rms": _time_rms(
                    times, values[:, index] - exact[:, index]
                ),
                "maximum_absolute": float(
                    np.max(np.abs(values[:, index] - exact[:, index]))
                ),
            }
            for index, name in enumerate(OBSERVABLE_NAMES)
        }
        for label, values in approximate.items()
    }
    for label, values in archive_native_observables.items():
        observable_errors[label] = {
            name: {
                "time_rms": _time_rms(
                    archive_times,
                    values[:, index] - archive_exact_observables[:, index],
                ),
                "maximum_absolute": float(
                    np.max(
                        np.abs(
                            values[:, index]
                            - archive_exact_observables[:, index]
                        )
                    )
                ),
            }
            for index, name in enumerate(OBSERVABLE_NAMES)
        }

    return {
        "maximum_exact_coordinate_disagreement_between_runs": (
            maximum_exact_disagreement
        ),
        "joint_gram_minimum_eigenvalues": joint_gram_minima,
        "M4_minimum_eigenvalues": completion_minima,
        "trajectory_errors": trajectory_errors,
        "autonomous_wall_seconds": {
            label: summary["integration"]["autonomous_wall_seconds"]
            for label, summary in apcm_summaries.items()
        },
        "grid_notes": {
            "APCM_time_step": 0.0025,
            "stored_31_moment_time_step": 0.01,
            "31_moment_curves_interpolated_to_APCM_grid_for_display": True,
            "31_moment_native_grid_nested_exactly_in_APCM_grid": True,
        },
        "observable_errors": observable_errors,
    }


def _plot_apcm_t20_comparison(output_directory: Path) -> dict[str, Any]:
    """Plot the accelerated higher-moment cone routes through t=20."""

    parameters = DimerParameters(
        hopping=1.0,
        gamma=0.5,
        lambda_ep=1.5,
        drive_amplitude=1.0,
    )
    times: np.ndarray | None = None
    exact_coordinates: np.ndarray | None = None
    paths: dict[str, np.ndarray] = {}
    joint_minima: dict[str, np.ndarray] = {}
    completion_minima: dict[str, np.ndarray] = {}
    correction_norms: dict[str, np.ndarray] = {}
    retraction_norms: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, Any]] = {}
    maximum_exact_disagreement = 0.0

    for label, path in APCM_T20_RUNS.items():
        summaries[label] = json.loads(
            path.with_name("summary.json").read_text(encoding="utf-8")
        )
        with np.load(path, allow_pickle=False) as arrays:
            local_times = np.asarray(arrays["times"], dtype=float)
            local_exact = np.asarray(
                arrays["exact_archive_coordinates"], dtype=float
            )
            if times is None:
                times = local_times
                exact_coordinates = local_exact
            else:
                if not np.array_equal(local_times, times):
                    raise ValueError(f"APCM t=20 time grid mismatch for {label}")
                maximum_exact_disagreement = max(
                    maximum_exact_disagreement,
                    float(np.max(np.abs(local_exact - exact_coordinates))),
                )
            paths[label] = np.asarray(
                arrays["approximate_archive_coordinates"], dtype=float
            )
            joint_minima[label] = np.asarray(
                arrays["joint_gram_minimum_eigenvalues"], dtype=float
            )
            completion_minima[label] = np.asarray(
                arrays["completion_minimum_eigenvalues"], dtype=float
            )
            correction_norms[label] = np.asarray(
                arrays["correction_norms"], dtype=float
            )
            retraction_norms[label] = np.asarray(
                arrays["hidden_retraction_norms"], dtype=float
            )
    if times is None or exact_coordinates is None:
        raise RuntimeError("APCM t=20 comparison has no trajectories")

    with np.load(ARCHIVE_LONG_RAW_REFINED, allow_pickle=False) as arrays:
        stored_raw_times = np.asarray(arrays["times"], dtype=float)
        stored_raw_coordinates = np.asarray(arrays["coordinates"], dtype=float)
    raw_selected = stored_raw_times <= float(times[-1]) + 1e-12
    raw_times = stored_raw_times[raw_selected]
    raw_coordinates = stored_raw_coordinates[raw_selected]
    if not np.isclose(raw_times[-1], times[-1]):
        raise ValueError("stored raw archive reference does not reach t=20")
    if not np.allclose(
        raw_coordinates[0], exact_coordinates[0], rtol=0.0, atol=3e-12
    ):
        raise ValueError("stored raw archive reference has a different initial state")

    exact_observables = _observables(times, exact_coordinates, parameters)
    raw_observables = _observables(raw_times, raw_coordinates, parameters)
    approximate_observables = {
        label: _observables(times, coordinates, parameters)
        for label, coordinates in paths.items()
    }
    colors = {
        "higher moment + both cones": "#1f77b4",
        "higher moment + M4 cone only": "#2ca02c",
    }
    styles = {
        "higher moment + both cones": (0, (6, 2)),
        "higher moment + M4 cone only": ":",
    }

    _style()
    figure, axes = plt.subplots(2, 3, figsize=(7.25, 4.20))
    for panel, axis in enumerate(axes.flat):
        axis.plot(
            times,
            exact_observables[:, panel],
            color="#000000",
            linewidth=1.55,
            label="exact cutoff-16 Hamiltonian",
            zorder=5,
        )
        axis.plot(
            raw_times,
            raw_observables[:, panel],
            color="#d62728",
            linestyle="-.",
            linewidth=1.10,
            label="uncorrected archive EOM",
            zorder=1,
        )
        for label, values in approximate_observables.items():
            axis.plot(
                times,
                values[:, panel],
                color=colors[label],
                linestyle=styles[label],
                linewidth=1.25,
                label=label,
            )
        _set_primary_limits(
            axis,
            [
                exact_observables[:, panel],
                raw_observables[:, panel],
                *[
                    values[:, panel]
                    for values in approximate_observables.values()
                ],
            ],
        )
        axis.set_title(
            f"({chr(97 + panel)}) {PANEL_TITLES[panel]}", loc="left"
        )
        axis.set_xlim(0.0, 20.0)
        axis.grid(alpha=0.18, linewidth=0.5)
        if panel in (0, 1):
            axis.set_ylabel("occupation")
        else:
            axis.set_ylabel(r"energy / $t_{\rm hop}$")
        if panel >= 3:
            axis.set_xlabel(r"$t\,t_{\rm hop}$")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=0.9,
        handlelength=2.2,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.10,
        top=0.82,
        hspace=0.42,
        wspace=0.34,
    )
    _save_figure(figure, output_directory / "apcm_t20_observables")

    scalar_error_paths = {
        label: np.sqrt(np.mean((coordinates - exact_coordinates) ** 2, axis=1))
        for label, coordinates in paths.items()
    }
    correlation_error_paths: dict[str, np.ndarray] = {}
    for label, coordinates in paths.items():
        values = np.empty(times.size, dtype=float)
        for index, (exact_row, approximate_row) in enumerate(
            zip(exact_coordinates, coordinates, strict=True)
        ):
            exact_state = closed_scalar_to_matrix_state(exact_row)
            approximate_state = closed_scalar_to_matrix_state(approximate_row)
            values[index] = np.linalg.norm(
                approximate_state.electron_phonon_correlation
                - exact_state.electron_phonon_correlation
            )
        correlation_error_paths[label] = values

    figure, axes = plt.subplots(2, 3, figsize=(7.25, 3.45))
    diagnostics = (
        (scalar_error_paths, "31-coordinate RMS error", None),
        (correlation_error_paths, r"$C$ Frobenius error", None),
        (joint_minima, r"$\lambda_{\min}(\mathcal{G})$", None),
        (completion_minima, r"$\lambda_{\min}(M_4)$", "symlog"),
        (correction_norms, "retained-cone correction", "symlog"),
        (retraction_norms, "hidden-stage retraction", "symlog"),
    )
    for panel, (series, title, scale) in enumerate(diagnostics):
        axis = axes.flat[panel]
        for label, values in series.items():
            axis.plot(
                times,
                values,
                color=colors[label],
                linestyle=styles[label],
                linewidth=1.15,
                label=label,
            )
        if panel in (2, 3):
            axis.axhline(0.0, color="#000000", linewidth=0.65, alpha=0.55)
        if scale == "symlog":
            axis.set_yscale("symlog", linthresh=1e-7)
        axis.set_title(f"({chr(103 + panel)}) {title}", loc="left")
        axis.set_xlim(0.0, 20.0)
        axis.set_xlabel(r"$t\,t_{\rm hop}$")
        axis.grid(alpha=0.18, linewidth=0.5)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.0,
        handlelength=2.4,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.12,
        top=0.86,
        hspace=0.52,
        wspace=0.34,
    )
    _save_figure(figure, output_directory / "apcm_t20_diagnostics")

    trajectory_errors = {
        label: _closed_coordinate_error_metrics(exact_coordinates, coordinates)
        for label, coordinates in paths.items()
    }
    for label, values in approximate_observables.items():
        trajectory_errors[label]["site_0_occupation_time_rms"] = _time_rms(
            times, values[:, 0] - exact_observables[:, 0]
        )
        trajectory_errors[label]["internal_energy_time_rms"] = _time_rms(
            times, values[:, 5] - exact_observables[:, 5]
        )

    return {
        "maximum_exact_coordinate_disagreement_between_runs": (
            maximum_exact_disagreement
        ),
        "trajectory_errors": trajectory_errors,
        "minimum_joint_gram_eigenvalues": {
            label: float(np.min(values)) for label, values in joint_minima.items()
        },
        "minimum_M4_eigenvalues": {
            label: float(np.min(values))
            for label, values in completion_minima.items()
        },
        "maximum_correction_norms": {
            label: float(np.max(values))
            for label, values in correction_norms.items()
        },
        "maximum_retraction_norms": {
            label: float(np.max(values))
            for label, values in retraction_norms.items()
        },
        "autonomous_wall_seconds": {
            label: summary["integration"]["autonomous_wall_seconds"]
            for label, summary in summaries.items()
        },
    }


def _plot_apcm_t31_checkpoint(output_directory: Path) -> dict[str, Any]:
    """Plot the valid endpoint of the interrupted t=20 to t=240 run."""

    parameters = DimerParameters(
        hopping=1.0,
        gamma=0.5,
        lambda_ep=1.5,
        drive_amplitude=1.0,
    )
    with np.load(
        APCM_T20_RUNS["higher moment + both cones"], allow_pickle=False
    ) as arrays:
        completed_times = np.asarray(arrays["times"], dtype=float)
        completed_coordinates = np.asarray(
            arrays["approximate_archive_coordinates"], dtype=float
        )
    with np.load(APCM_T31_CHECKPOINT, allow_pickle=False) as arrays:
        checkpoint_time = float(arrays["time"])
        checkpoint_state = np.asarray(arrays["state"], dtype=float)
        checkpoint_step = int(arrays["step"])
    checkpoint_matrix_state = raw_moment_coordinates_to_matrix_state(
        checkpoint_state[: len(RAW_MOMENT_COORDINATE_NAMES)]
    )
    checkpoint_coordinates = matrix_state_to_closed_scalar_coordinates(
        checkpoint_matrix_state
    )

    with np.load(APCM_T31_STORED_REFERENCE, allow_pickle=False) as arrays:
        exact_times = np.asarray(arrays["exact_times"], dtype=float)
        exact_coordinates = np.asarray(arrays["exact_coordinates"], dtype=float)
        stored_checkpoint_time = float(arrays["checkpoint_time"])
        stored_checkpoint_coordinates = np.asarray(
            arrays["checkpoint_coordinates"], dtype=float
        )
    if not np.isclose(stored_checkpoint_time, checkpoint_time):
        raise ValueError("stored exact reference has a different checkpoint time")
    if not np.allclose(
        stored_checkpoint_coordinates,
        checkpoint_coordinates,
        rtol=0.0,
        atol=2e-13,
    ):
        raise ValueError("stored score has a different checkpoint state")
    if not np.isclose(exact_times[-1], checkpoint_time):
        raise ValueError("stored exact reference does not reach the checkpoint")

    with np.load(ARCHIVE_LONG_RAW_REFINED, allow_pickle=False) as arrays:
        stored_raw_times = np.asarray(arrays["times"], dtype=float)
        stored_raw_coordinates = np.asarray(arrays["coordinates"], dtype=float)
        stored_raw_diagnostics = np.asarray(arrays["diagnostics"], dtype=float)
        stored_raw_diagnostic_names = tuple(
            str(value) for value in arrays["diagnostic_names"]
        )
    raw_selected = stored_raw_times <= checkpoint_time + 1e-12
    raw_times = stored_raw_times[raw_selected]
    raw_coordinates = stored_raw_coordinates[raw_selected]
    raw_diagnostics = stored_raw_diagnostics[raw_selected]
    if not np.isclose(raw_times[-1], checkpoint_time):
        raise ValueError("stored raw archive reference does not reach the checkpoint")
    if not np.allclose(
        raw_coordinates[0], exact_coordinates[0], rtol=0.0, atol=3e-12
    ):
        raise ValueError("stored raw archive reference has a different initial state")

    exact_observables = _observables(exact_times, exact_coordinates, parameters)
    raw_observables = _observables(raw_times, raw_coordinates, parameters)
    completed_observables = _observables(
        completed_times, completed_coordinates, parameters
    )
    checkpoint_observables = _observables(
        np.asarray([checkpoint_time]),
        checkpoint_coordinates[None, :],
        parameters,
    )[0]

    _style()
    figure, axes = plt.subplots(2, 3, figsize=(7.25, 4.20))
    for panel, axis in enumerate(axes.flat):
        axis.plot(
            exact_times,
            exact_observables[:, panel],
            color="#000000",
            linewidth=1.55,
            label="exact cutoff-16 Hamiltonian",
            zorder=4,
        )
        axis.plot(
            raw_times,
            raw_observables[:, panel],
            color="#d62728",
            linestyle="-.",
            linewidth=1.10,
            label="uncorrected archive EOM",
            zorder=1,
        )
        axis.plot(
            completed_times,
            completed_observables[:, panel],
            color="#1f77b4",
            linestyle=(0, (6, 2)),
            linewidth=1.25,
            label="both cones: stored trajectory to t=20",
        )
        axis.plot(
            [checkpoint_time],
            [checkpoint_observables[panel]],
            color="#1f77b4",
            marker="D",
            markersize=4.4,
            linestyle="none",
            label="valid interrupted checkpoint at t=31",
            zorder=6,
        )
        _set_primary_limits(
            axis,
            [
                exact_observables[:, panel],
                raw_observables[:, panel],
                completed_observables[:, panel],
                np.asarray([checkpoint_observables[panel]]),
            ],
        )
        axis.set_title(
            f"({chr(97 + panel)}) {PANEL_TITLES[panel]}", loc="left"
        )
        axis.set_xlim(0.0, checkpoint_time)
        axis.grid(alpha=0.18, linewidth=0.5)
        if panel in (0, 1):
            axis.set_ylabel("occupation")
        else:
            axis.set_ylabel(r"energy / $t_{\rm hop}$")
        if panel >= 3:
            axis.set_xlabel(r"$t\,t_{\rm hop}$")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=0.8,
        handlelength=2.1,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.10,
        top=0.82,
        hspace=0.42,
        wspace=0.34,
    )
    _save_figure(figure, output_directory / "apcm_t31_checkpoint_observables")

    exact_checkpoint = exact_coordinates[-1]
    difference = checkpoint_coordinates - exact_checkpoint
    raw_difference = raw_coordinates[-1] - exact_checkpoint
    exact_checkpoint_state = closed_scalar_to_matrix_state(exact_checkpoint)
    correlation_error = float(
        np.linalg.norm(
            checkpoint_matrix_state.electron_phonon_correlation
            - exact_checkpoint_state.electron_phonon_correlation
        )
    )
    raw_checkpoint_state = closed_scalar_to_matrix_state(raw_coordinates[-1])
    raw_correlation_error = float(
        np.linalg.norm(
            raw_checkpoint_state.electron_phonon_correlation
            - exact_checkpoint_state.electron_phonon_correlation
        )
    )
    raw_joint_gram_index = stored_raw_diagnostic_names.index(
        "joint_moment_minimum_eigenvalue"
    )
    endpoint_metrics = {
        "checkpoint_time": checkpoint_time,
        "checkpoint_step": checkpoint_step,
        "state_is_finite": bool(np.all(np.isfinite(checkpoint_state))),
        "maximum_absolute_state_coordinate": float(
            np.max(np.abs(checkpoint_state))
        ),
        "all31_scalar_rms_error": float(np.sqrt(np.mean(difference**2))),
        "C_block_frobenius_error": correlation_error,
        "total_site_0_occupation_absolute_error": float(
            abs(checkpoint_observables[0] - exact_observables[-1, 0])
        ),
        "internal_energy_absolute_error": float(
            abs(checkpoint_observables[5] - exact_observables[-1, 5])
        ),
        "raw_archive_all31_scalar_rms_error": float(
            np.sqrt(np.mean(raw_difference**2))
        ),
        "raw_archive_C_block_frobenius_error": raw_correlation_error,
        "raw_archive_joint_gram_minimum_eigenvalue_at_checkpoint": float(
            raw_diagnostics[-1, raw_joint_gram_index]
        ),
        "endpoint_observable_absolute_errors": {
            "uncorrected_archive_EOM": {
                name: float(value)
                for name, value in zip(
                    OBSERVABLE_NAMES,
                    np.abs(raw_observables[-1] - exact_observables[-1]),
                    strict=True,
                )
            },
            "higher_moment_both_cones": {
                name: float(value)
                for name, value in zip(
                    OBSERVABLE_NAMES,
                    np.abs(checkpoint_observables - exact_observables[-1]),
                    strict=True,
                )
            },
        },
        "references_reused_without_propagation": True,
        "intermediate_samples_available": False,
    }
    return endpoint_metrics


def _plot_mixed_guided_readmission_observables(
    output_directory: Path,
) -> dict[str, Any]:
    """Compare state-derived packet admissions through the second pulse."""

    paths: dict[str, np.ndarray] = {}
    times: np.ndarray | None = None
    exact: np.ndarray | None = None
    exact_coordinates: np.ndarray | None = None
    parent: np.ndarray | None = None
    for label, path in MIXED_GUIDED_RUNS.items():
        with np.load(path, allow_pickle=False) as arrays:
            local_times = np.asarray(arrays["times"], dtype=float)
            if times is None:
                times = local_times
                exact = np.asarray(arrays["exact_observables"], dtype=float)
                exact_coordinates = np.asarray(
                    arrays["exact_closed_coordinates"], dtype=float
                )
                parent = np.asarray(arrays["parent_observables"], dtype=float)
            elif not np.array_equal(local_times, times):
                raise ValueError(f"time grid mismatch for {label}")
            paths[label] = np.asarray(
                arrays["mixed_guided_observables"],
                dtype=float,
            )
    if (
        times is None
        or exact is None
        or exact_coordinates is None
        or parent is None
    ):
        raise RuntimeError("mixed-guided comparison has no trajectories")

    raw_archive, _ = _raw_archive_observables(
        times,
        exact_coordinates[0],
        pulse_delays=(0.0, 8.0),
    )

    _style()
    colors = {
        "ordinary K6": "#7f7f7f",
        "guided K6 at t=0": "#4c78a8",
        "readmit K7 at t=8": "#e45756",
        "readmit K8 at t=8": "#54a24b",
        "adaptive K6->7->8": "#b279a2",
    }
    styles = {
        "ordinary K6": "--",
        "guided K6 at t=0": ":",
        "readmit K7 at t=8": "-",
        "readmit K8 at t=8": "-.",
        "adaptive K6->7->8": (0, (3, 1, 1, 1)),
    }
    colors["raw archive EOM"] = RAW_ARCHIVE_COLOR
    styles["raw archive EOM"] = RAW_ARCHIVE_STYLE
    comparison = {
        "raw archive EOM": raw_archive,
        "ordinary K6": parent,
        **paths,
    }
    figure, axes = plt.subplots(2, 3, figsize=(7.25, 4.65))
    for panel, axis in enumerate(axes.flat):
        axis.plot(
            times,
            exact[:, panel],
            color="#171717",
            linewidth=1.45,
            label="exact cutoff-16",
            zorder=5,
        )
        for label, values in comparison.items():
            axis.plot(
                times,
                values[:, panel],
                color=colors[label],
                linestyle=styles[label],
                linewidth=(
                    RAW_ARCHIVE_LINEWIDTH
                    if label == "raw archive EOM"
                    else 1.0
                ),
                alpha=(
                    RAW_ARCHIVE_ALPHA
                    if label == "raw archive EOM"
                    else 1.0
                ),
                label=label,
                zorder=1 if label == "raw archive EOM" else 2,
            )
        _set_primary_limits(
            axis,
            [
                exact[:, panel],
                *[
                    values[:, panel]
                    for name, values in comparison.items()
                    if name != "raw archive EOM"
                ],
            ],
        )
        axis.set_title(
            f"({chr(97 + panel)}) {PANEL_TITLES[panel]}",
            loc="left",
        )
        axis.set_xlim(0.0, 20.0)
        axis.grid(alpha=0.18, linewidth=0.5)
        for start in (0.0, 8.0):
            axis.axvspan(
                start,
                start + 4.0,
                color="#d8e8f1",
                alpha=0.32,
                lw=0.0,
            )
        axis.set_ylabel(
            "occupation" if panel in (0, 1) else r"energy / $t_{\rm hop}$"
        )
        if panel >= 3:
            axis.set_xlabel(r"$t\,t_{\rm hop}$")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=0.8,
        handlelength=2.1,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.10,
        top=0.84,
        hspace=0.42,
        wspace=0.34,
    )
    _save_figure(
        figure,
        output_directory / "mixed_guided_readmission_observables",
    )
    return {
        label: {
            name: _time_rms(times, values[:, index] - exact[:, index])
            for index, name in enumerate(OBSERVABLE_NAMES)
        }
        for label, values in comparison.items()
    }


def _plot_adaptive_transfer_observables(
    output_directory: Path,
) -> dict[str, Any]:
    """Plot the frozen adaptive gate on preparation and drive holdouts."""

    _style()
    selected_observables = (
        (0, r"site-0 occupation, $n_0$"),
        (3, r"phonon energy, $E_{\rm ph}$"),
        (4, r"electron--phonon energy, $E_{\rm e-ph}$"),
    )
    figure, axes = plt.subplots(3, 3, figsize=(7.25, 5.65), sharex=True)
    metrics: dict[str, Any] = {}
    for row, (label, path) in enumerate(TRANSFER_RUNS.items()):
        with np.load(path, allow_pickle=False) as arrays:
            times = np.asarray(arrays["times"], dtype=float)
            exact_coordinates = np.asarray(
                arrays["exact_closed_coordinates"], dtype=float
            )
            exact = np.asarray(arrays["exact_observables"], dtype=float)
            parent = np.asarray(arrays["parent_observables"], dtype=float)
            adaptive = np.asarray(
                arrays["mixed_guided_observables"],
                dtype=float,
            )
        pulse_delays = (
            (0.0,) if "single pulse" in label else (0.0, 8.0)
        )
        raw_archive, raw_failure_time = _raw_archive_observables(
            times,
            exact_coordinates[0],
            pulse_delays=pulse_delays,
        )
        metrics[label] = {
            "raw_archive_amplitude_threshold_time": raw_failure_time,
            "ordinary_parent": {
                name: _time_rms(times, parent[:, index] - exact[:, index])
                for index, name in enumerate(OBSERVABLE_NAMES)
            },
            "adaptive_gate": {
                name: _time_rms(times, adaptive[:, index] - exact[:, index])
                for index, name in enumerate(OBSERVABLE_NAMES)
            },
        }
        pulse_starts = pulse_delays
        for column, (observable, title) in enumerate(selected_observables):
            axis = axes[row, column]
            axis.plot(
                times,
                exact[:, observable],
                color="#171717",
                linewidth=1.35,
                label="exact cutoff-16",
                zorder=5,
            )
            axis.plot(
                times,
                raw_archive[:, observable],
                color=RAW_ARCHIVE_COLOR,
                linestyle=RAW_ARCHIVE_STYLE,
                linewidth=RAW_ARCHIVE_LINEWIDTH,
                alpha=RAW_ARCHIVE_ALPHA,
                label="raw archive EOM",
                zorder=1,
            )
            axis.plot(
                times,
                parent[:, observable],
                color="#7f7f7f",
                linestyle="--",
                linewidth=1.0,
                label="ordinary parent",
            )
            axis.plot(
                times,
                adaptive[:, observable],
                color="#b279a2",
                linewidth=1.1,
                label="adaptive gate",
            )
            _set_primary_limits(
                axis,
                [
                    exact[:, observable],
                    parent[:, observable],
                    adaptive[:, observable],
                ],
            )
            for start in pulse_starts:
                axis.axvspan(
                    start,
                    start + 4.0,
                    color="#d8e8f1",
                    alpha=0.32,
                    lw=0.0,
                )
            axis.set_xlim(0.0, 20.0)
            axis.grid(alpha=0.18, linewidth=0.5)
            if row == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(label.replace(", ", "\n"))
            if row == 2:
                axis.set_xlabel(r"$t\,t_{\rm hop}$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
    )
    figure.subplots_adjust(
        left=0.13,
        right=0.985,
        bottom=0.08,
        top=0.91,
        hspace=0.26,
        wspace=0.27,
    )
    _save_figure(figure, output_directory / "adaptive_transfer_observables")
    return metrics


def _plot_adaptive_t40_observables(
    times: np.ndarray,
    exact: np.ndarray,
    raw_archive: np.ndarray,
    fixed_k10: np.ndarray,
    output_directory: Path,
) -> dict[str, Any]:
    """Compare the uncapped gate with matched fixed-capacity trajectories."""

    with np.load(ADAPTIVE_T40, allow_pickle=False) as arrays:
        adaptive_times = np.asarray(arrays["times"], dtype=float)
        parent = np.asarray(arrays["parent_observables"], dtype=float)
        adaptive = np.asarray(
            arrays["mixed_guided_observables"],
            dtype=float,
        )
    if adaptive_times.shape != times.shape or not np.allclose(
        adaptive_times,
        times,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("adaptive t=40 time grid does not match exact grid")

    _style()
    paths = {
        "raw archive EOM": raw_archive,
        "ordinary K6": parent,
        "fixed K10": fixed_k10,
        "adaptive to K11": adaptive,
    }
    colors = {
        "raw archive EOM": RAW_ARCHIVE_COLOR,
        "ordinary K6": "#7f7f7f",
        "fixed K10": "#54a24b",
        "adaptive to K11": "#b279a2",
    }
    styles = {
        "raw archive EOM": RAW_ARCHIVE_STYLE,
        "ordinary K6": "--",
        "fixed K10": "-.",
        "adaptive to K11": "-",
    }
    figure, axes = plt.subplots(2, 3, figsize=(7.25, 4.65))
    for panel, axis in enumerate(axes.flat):
        axis.plot(
            times,
            exact[:, panel],
            color="#171717",
            linewidth=1.45,
            label="exact cutoff-16",
            zorder=5,
        )
        for label, values in paths.items():
            axis.plot(
                times,
                values[:, panel],
                color=colors[label],
                linestyle=styles[label],
                linewidth=(
                    RAW_ARCHIVE_LINEWIDTH
                    if label == "raw archive EOM"
                    else 1.05
                ),
                alpha=(
                    RAW_ARCHIVE_ALPHA
                    if label == "raw archive EOM"
                    else 1.0
                ),
                label=label,
                zorder=1 if label == "raw archive EOM" else 2,
            )
        _set_primary_limits(
            axis,
            [
                exact[:, panel],
                *[
                    values[:, panel]
                    for name, values in paths.items()
                    if name != "raw archive EOM"
                ],
            ],
        )
        _decorate_axis(axis, panel)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.10,
        top=0.89,
        hspace=0.42,
        wspace=0.34,
    )
    _save_figure(figure, output_directory / "adaptive_t40_observables")
    return {
        label: {
            name: _time_rms(times, values[:, index] - exact[:, index])
            for index, name in enumerate(OBSERVABLE_NAMES)
        }
        for label, values in paths.items()
    }


def _plot_adaptive_t98_observables(
    output_directory: Path,
) -> dict[str, Any]:
    """Plot the resumed adaptive trajectory against its matched reference."""

    with np.load(ADAPTIVE_T98, allow_pickle=False) as arrays:
        times = np.asarray(arrays["times"], dtype=float)
        exact_coordinates = np.asarray(
            arrays["exact_closed_coordinates"], dtype=float
        )
        exact = np.asarray(arrays["exact_observables"], dtype=float)
        parent = np.asarray(arrays["parent_observables"], dtype=float)
        adaptive = np.asarray(
            arrays["mixed_guided_observables"],
            dtype=float,
        )
    raw_archive, raw_failure_time = _raw_archive_observables(
        times,
        exact_coordinates[0],
        pulse_delays=(0.0, 8.0),
    )
    summary = json.loads(ADAPTIVE_T98_SUMMARY.read_text(encoding="utf-8"))
    adaptive_label = f"adaptive to K{summary['packet_capacity']['final_K']}"
    paths = {
        "raw archive EOM": raw_archive,
        "ordinary K6": parent,
        adaptive_label: adaptive,
    }
    colors = {
        "raw archive EOM": RAW_ARCHIVE_COLOR,
        "ordinary K6": "#7f7f7f",
        adaptive_label: "#b279a2",
    }
    styles = {
        "raw archive EOM": RAW_ARCHIVE_STYLE,
        "ordinary K6": "--",
        adaptive_label: "-",
    }

    _style()
    figure, axes = plt.subplots(2, 3, figsize=(7.25, 4.65))
    for panel, axis in enumerate(axes.flat):
        axis.plot(
            times,
            exact[:, panel],
            color="#171717",
            linewidth=1.45,
            label="exact cutoff-16",
            zorder=5,
        )
        for label, values in paths.items():
            axis.plot(
                times,
                values[:, panel],
                color=colors[label],
                linestyle=styles[label],
                linewidth=(
                    RAW_ARCHIVE_LINEWIDTH
                    if label == "raw archive EOM"
                    else 1.05
                ),
                alpha=(
                    RAW_ARCHIVE_ALPHA
                    if label == "raw archive EOM"
                    else 1.0
                ),
                label=label,
                zorder=1 if label == "raw archive EOM" else 2,
            )
        _set_primary_limits(
            axis,
            [
                exact[:, panel],
                *[
                    values[:, panel]
                    for name, values in paths.items()
                    if name != "raw archive EOM"
                ],
            ],
        )
        axis.set_title(
            f"({chr(97 + panel)}) {PANEL_TITLES[panel]}",
            loc="left",
        )
        axis.set_xlim(0.0, float(times[-1]))
        axis.grid(alpha=0.18, linewidth=0.5)
        for start in (0.0, 8.0):
            axis.axvspan(
                start,
                start + 4.0,
                color="#d8e8f1",
                alpha=0.32,
                lw=0.0,
            )
        axis.set_ylabel(
            "occupation" if panel in (0, 1) else r"energy / $t_{\rm hop}$"
        )
        if panel >= 3:
            axis.set_xlabel(r"$t\,t_{\rm hop}$")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.10,
        top=0.89,
        hspace=0.42,
        wspace=0.34,
    )
    _save_figure(figure, output_directory / "adaptive_t98_observables")
    return {
        "summary": summary["comparison"],
        "packet_capacity": summary["packet_capacity"],
        "raw_archive_amplitude_threshold_time": raw_failure_time,
    }


def _plot_cumulative_errors(
    times: np.ndarray,
    exact: np.ndarray,
    raw_archive: np.ndarray,
    packet: dict[int, np.ndarray],
    output_directory: Path,
) -> None:
    _style()
    figure, axes = plt.subplots(2, 3, figsize=(7.25, 4.65))
    for panel, axis in enumerate(axes.flat):
        raw_error = raw_archive[:, panel] - exact[:, panel]
        finite = np.isfinite(raw_error)
        axis.plot(
            times[finite],
            _cumulative_rms(times[finite], raw_error[finite]),
            color=RAW_ARCHIVE_COLOR,
            linestyle=RAW_ARCHIVE_STYLE,
            linewidth=RAW_ARCHIVE_LINEWIDTH,
            alpha=RAW_ARCHIVE_ALPHA,
            label="raw archive EOM",
        )
        for packet_count in sorted(packet):
            error = packet[packet_count][:, panel] - exact[:, panel]
            axis.plot(
                times,
                _cumulative_rms(times, error),
                color=COLORS[packet_count],
                linestyle=LINESTYLES[packet_count],
                linewidth=1.15,
                label=rf"$K_{{\max}}={packet_count}$",
            )
        axis.set_title(
            f"({chr(97 + panel)}) {PANEL_TITLES[panel]}", loc="left"
        )
        axis.set_xlim(0.0, 40.0)
        axis.grid(alpha=0.18, linewidth=0.5)
        if panel in (0, 1):
            axis.set_ylabel("cumulative RMS error")
        else:
            axis.set_ylabel(r"cumulative RMS error / $t_{\rm hop}$")
        if panel >= 3:
            axis.set_xlabel(r"$t\,t_{\rm hop}$")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.2,
        handlelength=2.4,
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.10,
        top=0.89,
        hspace=0.42,
        wspace=0.36,
    )
    _save_figure(
        figure,
        output_directory / "packet_capacity_cumulative_observable_errors",
    )


def _plot_capacity_diagnostics(
    times: np.ndarray,
    trajectory_arrays: dict[int, dict[str, np.ndarray]],
    source_summaries: dict[int, dict[str, Any]],
    output_directory: Path,
) -> None:
    _style()
    packet_counts = np.asarray(sorted(source_summaries), dtype=int)
    projected_exact = np.asarray(
        [
            source_summaries[count]["source_errors"]
            ["projected_to_exact_path_nrms"]
            for count in packet_counts
        ]
    )
    projection = np.asarray(
        [
            source_summaries[count]["source_errors"]
            ["projected_to_same_state_schrodinger_nrms_normalized_by_exact_path"]
            for count in packet_counts
        ]
    )
    state_path = np.asarray(
        [
            source_summaries[count]["source_errors"]
            ["schrodinger_packet_to_exact_path_nrms"]
            for count in packet_counts
        ]
    )
    tangent = np.asarray(
        [source_summaries[count]["tangent"]["rms_relative_residual"] for count in packet_counts]
    )

    figure, axes = plt.subplots(1, 3, figsize=(7.25, 2.45))
    axes[0].plot(packet_counts, projected_exact, "o-", color="#4c78a8", label="projected source vs exact path")
    axes[0].plot(packet_counts, state_path, "s--", color="#e45756", label="same-state exact source vs exact path")
    axes[0].plot(packet_counts, projection, "^--", color="#54a24b", label="projection defect")
    axes[0].set_title("(a) correlation-source error", loc="left")
    axes[0].set_xlabel(r"maximum packets per branch, $K_{\max}$")
    axes[0].set_ylabel("normalized RMS")
    axes[0].set_xticks(packet_counts)
    axes[0].legend(frameon=False, fontsize=6.1, loc="best")

    axes[1].plot(packet_counts, tangent, "o-", color="#b279a2")
    axes[1].set_title("(b) tangent-projection residual", loc="left")
    axes[1].set_xlabel(r"maximum packets per branch, $K_{\max}$")
    axes[1].set_ylabel("relative RMS")
    axes[1].set_xticks(packet_counts)

    for packet_count in (10, 12):
        arrays = trajectory_arrays[packet_count]
        deficit = arrays["geometric_tangent_rank"] - arrays["tangent_rank"]
        axes[2].plot(
            times,
            deficit,
            color=COLORS[packet_count],
            linestyle=LINESTYLES[packet_count],
            linewidth=1.1,
            label=rf"$K_{{\max}}={packet_count}$",
        )
    axes[2].set_title("(c) discarded tangent directions", loc="left")
    axes[2].set_xlabel(r"$t\,t_{\rm hop}$")
    axes[2].set_ylabel("geometric rank - retained rank")
    axes[2].set_xlim(0.0, 40.0)
    axes[2].legend(frameon=False)

    for axis in axes:
        axis.grid(alpha=0.18, linewidth=0.5)
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.22, top=0.86, wspace=0.36)
    _save_figure(figure, output_directory / "packet_capacity_diagnostics")


def build(output_directory: Path) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    parameters = DimerParameters(
        hopping=1.0,
        gamma=0.5,
        lambda_ep=1.5,
        drive_amplitude=1.0,
        pulse_width=1.0,
    )
    with np.load(SCORE_ARRAYS) as score:
        score_times = np.asarray(score["times"], dtype=float)
        selected = score_times <= 40.0 + 1e-12
        times = score_times[selected]
        exact_coordinates = np.asarray(
            score["exact_dop853_closed"][0, selected], dtype=float
        )
        midpoint_coordinates = np.asarray(
            score["exact_midpoint_closed"][0, selected], dtype=float
        )

    trajectory_arrays: dict[int, dict[str, np.ndarray]] = {}
    packet_coordinates: dict[int, np.ndarray] = {}
    parameter_hashes: dict[int, str] = {}
    for packet_count, path in TRAJECTORIES.items():
        with np.load(path) as arrays:
            local_times = np.asarray(arrays["times"], dtype=float)
            keep = local_times <= 40.0 + 1e-12
            if not np.array_equal(local_times[keep], times):
                raise ValueError(f"time grid mismatch for Kmax={packet_count}")
            trajectory_arrays[packet_count] = {
                "packet_count": np.asarray(
                    arrays["packet_count_trajectory"][keep], dtype=int
                ),
                "tangent_rank": np.asarray(arrays["tangent_rank"][keep], dtype=int),
                "geometric_tangent_rank": np.asarray(
                    arrays["geometric_tangent_rank"][keep], dtype=int
                ),
            }
            packet_coordinates[packet_count] = np.asarray(
                arrays["closed_coordinates"][keep], dtype=float
            )
        summary = json.loads(
            TRAJECTORY_SUMMARIES[packet_count].read_text(encoding="utf-8")
        )
        parameter_hashes[packet_count] = summary["initialization"][
            "parameter_sha256"
        ]
    if len(set(parameter_hashes.values())) != 1:
        raise ValueError("packet-capacity trajectories do not share initialization")

    exact = _observables(times, exact_coordinates, parameters)
    midpoint = _observables(times, midpoint_coordinates, parameters)
    raw_archive, raw_archive_failure_time = _raw_archive_observables(
        times,
        exact_coordinates[0],
        pulse_delays=(0.0, 8.0),
    )
    packet = {
        count: _observables(times, path, parameters)
        for count, path in packet_coordinates.items()
    }
    source_summaries = {
        count: json.loads(path.read_text(encoding="utf-8"))
        for count, path in SOURCE_SUMMARIES.items()
    }

    _plot_observables(times, exact, raw_archive, packet, output_directory)
    _plot_cumulative_errors(
        times,
        exact,
        raw_archive,
        packet,
        output_directory,
    )
    _plot_capacity_diagnostics(
        times,
        trajectory_arrays,
        source_summaries,
        output_directory,
    )
    mixed_guided_metrics = _plot_mixed_guided_readmission_observables(
        output_directory
    )
    adaptive_transfer_metrics = _plot_adaptive_transfer_observables(
        output_directory
    )
    adaptive_t40_metrics = _plot_adaptive_t40_observables(
        times,
        exact,
        raw_archive,
        packet[10],
        output_directory,
    )
    adaptive_t98_metrics = _plot_adaptive_t98_observables(output_directory)
    apcm_matched_metrics = _plot_apcm_matched_observables(output_directory)
    apcm_t20_metrics = _plot_apcm_t20_comparison(output_directory)
    apcm_t31_metrics = _plot_apcm_t31_checkpoint(output_directory)

    observable_metrics: dict[str, Any] = {}
    finite_raw = np.all(np.isfinite(raw_archive), axis=1)
    observable_metrics["raw archive EOM"] = {
        name: {
            "time_rms": _time_rms(
                times[finite_raw],
                raw_archive[finite_raw, index] - exact[finite_raw, index],
            ),
            "maximum_absolute": float(
                np.max(
                    np.abs(
                        raw_archive[finite_raw, index]
                        - exact[finite_raw, index]
                    )
                )
            ),
        }
        for index, name in enumerate(OBSERVABLE_NAMES)
    }
    for packet_count, values in packet.items():
        difference = values - exact
        observable_metrics[str(packet_count)] = {
            name: {
                "time_rms": _time_rms(times, difference[:, index]),
                "maximum_absolute": float(np.max(np.abs(difference[:, index]))),
            }
            for index, name in enumerate(OBSERVABLE_NAMES)
        }
    exact_difference = exact - midpoint
    exact_solver_metrics = {
        name: {
            "time_rms": _time_rms(times, exact_difference[:, index]),
            "maximum_absolute": float(
                np.max(np.abs(exact_difference[:, index]))
            ),
        }
        for index, name in enumerate(OBSERVABLE_NAMES)
    }
    initial_observable_mismatch = {
        str(count): {
            name: float(values[0, index] - exact[0, index])
            for index, name in enumerate(OBSERVABLE_NAMES)
        }
        for count, values in packet.items()
    }
    source_metrics = {
        str(count): {
            "projected_source_vs_exact_path_nrms": summary["source_errors"]
            ["projected_to_exact_path_nrms"],
            "same_state_projection_nrms": summary["source_errors"]
            ["projected_to_same_state_schrodinger_nrms_normalized_by_exact_path"],
            "same_state_source_vs_exact_path_nrms": summary["source_errors"]
            ["schrodinger_packet_to_exact_path_nrms"],
            "tangent_relative_residual_rms": summary["tangent"]
            ["rms_relative_residual"],
            "minimum_tangent_rank": summary["tangent"]["minimum_tangent_rank"],
            "maximum_tangent_rank": summary["tangent"]["maximum_tangent_rank"],
            "maximum_geometric_tangent_rank": summary["tangent"]
            ["maximum_geometric_rank"],
        }
        for count, summary in source_summaries.items()
    }
    input_paths = [
        SCORE_ARRAYS,
        *TRAJECTORIES.values(),
        *SOURCE_SUMMARIES.values(),
        *MIXED_GUIDED_RUNS.values(),
        *TRANSFER_RUNS.values(),
        ADAPTIVE_T40,
        ADAPTIVE_T98,
        ADAPTIVE_T98_SUMMARY,
        ARCHIVE_ABLATION,
        *APCM_MATCHED_RUNS.values(),
        *APCM_T20_RUNS.values(),
        APCM_T31_CHECKPOINT,
        APCM_T31_STORED_REFERENCE,
        ARCHIVE_LONG_RAW_REFINED,
    ]
    metrics = {
        "schema": "paper5.results_progression.packet_capacity.v1",
        "classification": "exploratory_completed_results_summary",
        "time_interval": [0.0, 40.0],
        "parameters": {
            "hopping": 1.0,
            "gamma": 0.5,
            "lambda_ep": 1.5,
            "coupling": parameters.coupling,
            "drive_amplitude": 1.0,
            "pulse_delays": [0.0, 8.0],
            "pulse_width": 1.0,
            "phonon_cutoff": 16,
            "output_sample_step": 0.05,
            "packet_initialization_sha256": next(iter(parameter_hashes.values())),
        },
        "reference": {
            "displayed": "cutoff-16 DOP853 wavefunction propagation",
            "midpoint_used_online": False,
            "exact_solver_observable_disagreement": exact_solver_metrics,
            "raw_archive_EOM_amplitude_threshold_time": (
                raw_archive_failure_time
            ),
        },
        "initial_packet_minus_exact_observable_difference": initial_observable_mismatch,
        "observable_errors_vs_dop853": observable_metrics,
        "source_and_tangent_metrics": source_metrics,
        "mixed_guided_readmission_observable_errors": mixed_guided_metrics,
        "adaptive_transfer_observable_errors": adaptive_transfer_metrics,
        "adaptive_t40_observable_errors": adaptive_t40_metrics,
        "adaptive_t98_result": adaptive_t98_metrics,
        "apcm_t4_matched_ablation": apcm_matched_metrics,
        "apcm_t20_accelerated_comparison": apcm_t20_metrics,
        "apcm_t31_interrupted_checkpoint": apcm_t31_metrics,
        "input_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in input_paths},
    }
    metrics_path = output_directory / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "output/plots/paper_v_results_progression_20260804",
    )
    args = parser.parse_args()
    result = build(args.output_directory)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
