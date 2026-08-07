"""Render advisor-facing Paper V figures from completed trajectory data."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from paper5.stability.initial_conditions import (
    exact_ground_closed_scalar_coordinates,
)
from paper5.stability.matrix_reference import (
    closed_scalar_rhs,
    closed_scalar_to_matrix_state,
    electron_phonon_moment_matrix,
)
from paper5.stability.hubbard_dimer import DimerParameters


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "paper_v"
TRAJECTORY_PATH = (
    REPO_ROOT
    / "output/local_runs/paper_v_electron_phonon_analysis_20260801_v3/"
    "baseline_trajectories.npz"
)
ATTRIBUTION_PATH = (
    REPO_ROOT
    / "output/local_runs/paper_v_mixed_moment_attribution_20260801_v1/"
    "mixed_moment_trajectory.npz"
)


def _thirty_one_coordinate_divergence(
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the audited raw 31-coordinate trajectory through its threshold."""

    parameters = DimerParameters(
        gamma=0.5,
        lambda_ep=1.5,
        drive_amplitude=1.0,
    )
    initial_state = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=16,
    )

    def threshold_event(_time: float, state: np.ndarray) -> float:
        return float(np.max(np.abs(state)) - 1e4)

    threshold_event.terminal = True
    threshold_event.direction = 1.0
    sample_times = np.arange(0.0, 150.0 + 0.02, 0.02)
    solution = solve_ivp(
        lambda time, state: closed_scalar_rhs(time, state, parameters),
        (0.0, 150.0),
        initial_state,
        method="DOP853",
        t_eval=sample_times,
        events=threshold_event,
        rtol=1e-9,
        atol=1e-11,
        max_step=0.02,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    if not solution.t_events[0].size:
        raise RuntimeError("31-coordinate trajectory did not reach 1e4")

    crossing_time = float(solution.t_events[0][0])
    crossing_state = np.asarray(solution.y_events[0][0], dtype=float)
    times = np.append(np.asarray(solution.t, dtype=float), crossing_time)
    coordinates = np.column_stack(
        [np.asarray(solution.y, dtype=float), crossing_state]
    )
    maxima = np.max(np.abs(coordinates), axis=0)
    return times, maxima, crossing_time


def _build_divergence_figure() -> None:
    times, maxima, crossing_time = _thirty_one_coordinate_divergence()
    figure, axis = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    axis.semilogy(
        times,
        maxima,
        color="#A33A2B",
        linewidth=1.8,
        label="uncorrected 31-coordinate trajectory",
    )
    axis.axhline(
        1e4,
        color="#4A4F57",
        linestyle="--",
        linewidth=1.0,
        label="declared failure threshold",
    )
    axis.axvline(
        crossing_time,
        color="#A33A2B",
        linestyle=":",
        linewidth=1.0,
    )
    axis.plot(crossing_time, 1e4, "o", color="#A33A2B", markersize=3.5)
    axis.annotate(
        rf"$t={crossing_time:.2f}$",
        xy=(crossing_time, 1e4),
        xytext=(-48, -21),
        textcoords="offset points",
        color="#7B2C21",
        fontsize=8,
    )
    axis.set_xlim(0.0, 145.0)
    axis.set_ylim(1e-1, 3e4)
    axis.set_xlabel("time (inverse-hopping units)")
    axis.set_ylabel("largest absolute coordinate")
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(loc="upper left", frameon=False, fontsize=8)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_DIRECTORY / "paper_v_divergence_reproduction.pdf",
        bbox_inches="tight",
    )
    figure.savefig(
        OUTPUT_DIRECTORY / "paper_v_divergence_reproduction.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    multiscale, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.4),
        constrained_layout=True,
    )
    full, early = axes
    full.semilogy(
        times,
        maxima,
        color="#A33A2B",
        linewidth=1.8,
        label="uncorrected 31-coordinate trajectory",
    )
    full.axhline(
        1e4,
        color="#4A4F57",
        linestyle="--",
        linewidth=1.0,
        label="declared failure threshold",
    )
    full.axvline(
        crossing_time,
        color="#A33A2B",
        linestyle=":",
        linewidth=1.0,
    )
    full.plot(crossing_time, 1e4, "o", color="#A33A2B", markersize=3.5)
    full.annotate(
        rf"$t={crossing_time:.2f}$",
        xy=(crossing_time, 1e4),
        xytext=(-48, -21),
        textcoords="offset points",
        color="#7B2C21",
        fontsize=8,
    )
    full.set_xlim(0.0, 145.0)
    full.set_ylim(1e-1, 3e4)
    full.set_title("(a) Full trajectory through the threshold")
    full.set_ylabel("largest absolute coordinate")
    full.grid(True, which="both", alpha=0.22)
    full.legend(loc="upper left", frameon=False, fontsize=8)

    early_mask = times <= 40.0
    early.plot(
        times[early_mask],
        maxima[early_mask],
        color="#A33A2B",
        linewidth=1.8,
    )
    early.set_xlim(0.0, 40.0)
    early.set_ylim(1.15, 2.2)
    early.set_title(r"(b) Early-time zoom, $0\leq t\leq40$")
    early.set_xlabel("time (inverse-hopping units)")
    early.set_ylabel("largest absolute coordinate")
    early.grid(True, alpha=0.22)

    multiscale.savefig(
        OUTPUT_DIRECTORY / "paper_v_divergence_reproduction_multiscale.pdf",
        bbox_inches="tight",
    )
    multiscale.savefig(
        OUTPUT_DIRECTORY / "paper_v_divergence_reproduction_multiscale.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(multiscale)


def _joint_minimum(coordinates: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.linalg.eigvalsh(
                electron_phonon_moment_matrix(
                    closed_scalar_to_matrix_state(state)
                )
            )[0]
            for state in coordinates
        ]
    )


def _build_physicality_figure() -> None:
    with np.load(TRAJECTORY_PATH) as arrays:
        times = arrays["baseline__times"]
        exact = arrays["baseline__exact"]
        uncorrected = arrays["baseline__raw"]
        corrected = arrays["baseline__corrected"]

    exact_minimum = _joint_minimum(exact)
    uncorrected_minimum = _joint_minimum(uncorrected)
    corrected_minimum = _joint_minimum(corrected)

    figure, first = plt.subplots(
        figsize=(3.55, 2.3),
        constrained_layout=True,
    )
    first.axhline(0.0, color="black", linewidth=0.65)
    first.plot(
        times,
        uncorrected_minimum,
        color="#9B2C2C",
        label="uncorrected moment equations",
    )
    first.plot(
        times,
        exact_minimum,
        color="#2B6CB0",
        label="exact Hamiltonian",
    )
    first.plot(
        times,
        corrected_minimum,
        color="#2F6B4F",
        label="physicality-corrected equations",
    )
    first.axvline(0.1607116782, color="#9B2C2C", linestyle=":", linewidth=0.9)
    first.set_title("Physicality fails before the divergence")
    first.set_xlabel("time (inverse-hopping units)")
    first.set_ylabel("smallest joint-matrix eigenvalue")
    first.legend(frameon=False, loc="best")
    first.grid(alpha=0.18, linewidth=0.5)

    figure.savefig(
        OUTPUT_DIRECTORY / "paper_v_physicality_correction.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def _build_missing_physics_figure() -> None:
    with np.load(ATTRIBUTION_PATH) as arrays:
        times = arrays["times"]
        series = (
            (
                "archive_residual_subtracted_coordinates",
                "second-order moment equations",
            ),
            (
                "archive_plus_k_residual_subtracted_coordinates",
                "+ omitted mixed correlation",
            ),
            (
                "archive_plus_k_plus_pauli_residual_subtracted_coordinates",
                "+ exact fermionic identity",
            ),
            (
                "archive_plus_k_plus_pauli_plus_opposite_spin_residual_subtracted_coordinates",
                "+ opposite-spin correlation",
            ),
        )
        values = [(label, np.linalg.norm(arrays[key], axis=1)) for key, label in series]

    figure, axis = plt.subplots(figsize=(6.0, 3.45), constrained_layout=True)
    for label, norms in values:
        axis.plot(times, norms, label=label)
    axis.set_title("Origin of the remaining correlation error")
    axis.set_xlabel("time (inverse-hopping units)")
    axis.set_ylabel("electron-phonon derivative error")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    figure.savefig(
        OUTPUT_DIRECTORY / "paper_v_missing_physics.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 7.4,
            "axes.labelsize": 7.4,
            "axes.titlesize": 8.0,
            "legend.fontsize": 6.4,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "lines.linewidth": 1.15,
        }
    )
    _build_divergence_figure()
    _build_physicality_figure()
    _build_missing_physics_figure()


if __name__ == "__main__":
    main()
