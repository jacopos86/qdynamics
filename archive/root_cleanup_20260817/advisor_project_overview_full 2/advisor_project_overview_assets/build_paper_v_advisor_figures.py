"""Render advisor-facing Paper V figures from completed trajectory data."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper5.stability.matrix_reference import (
    closed_scalar_to_matrix_state,
    electron_phonon_moment_matrix,
)
from paper5.stability.hubbard_dimer import (
    DimerParameters,
    fan_migdal_rhs,
    hartree_fock_zero_correlation_state,
)


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


def _maximum_component_trace(coupling_strength: float) -> tuple[np.ndarray, np.ndarray]:
    parameters = DimerParameters(
        gamma=0.5,
        lambda_ep=coupling_strength,
        drive_amplitude=1.0,
    )
    state = hartree_fock_zero_correlation_state()
    time = 0.0
    times = [time]
    maxima = [float(np.max(np.abs(state)))]
    while time < 140.0:
        step = min(0.01, 140.0 - time)
        first = fan_migdal_rhs(time, state, parameters)
        second = fan_migdal_rhs(
            time + 0.5 * step,
            state + 0.5 * step * first,
            parameters,
        )
        third = fan_migdal_rhs(
            time + 0.5 * step,
            state + 0.5 * step * second,
            parameters,
        )
        fourth = fan_migdal_rhs(
            time + step,
            state + step * third,
            parameters,
        )
        state = state + (step / 6.0) * (
            first + 2.0 * second + 2.0 * third + fourth
        )
        time += step
        times.append(time)
        maxima.append(float(np.max(np.abs(state))))
        if maxima[-1] > 1e4:
            break
    return np.asarray(times), np.asarray(maxima)


def _build_divergence_figure() -> None:
    strong_times, strong_maxima = _maximum_component_trace(1.5)
    weak_times, weak_maxima = _maximum_component_trace(0.5)
    figure, axis = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    axis.semilogy(
        strong_times,
        strong_maxima,
        color="#A33A2B",
        linewidth=1.8,
        label="strong coupling",
    )
    axis.semilogy(
        weak_times,
        weak_maxima,
        color="#236C5B",
        linewidth=1.8,
        label="weak-coupling comparison",
    )
    axis.axhline(
        1e4,
        color="#4A4F57",
        linestyle="--",
        linewidth=1.0,
        label="declared failure threshold",
    )
    axis.set_xlim(0.0, 140.0)
    axis.set_ylim(1e-1, 3e4)
    axis.set_xlabel("time (inverse-hopping units)")
    axis.set_ylabel("largest absolute component")
    axis.grid(True, which="both", alpha=0.22)
    axis.legend(loc="upper left", frameon=False, fontsize=8)
    figure.savefig(
        OUTPUT_DIRECTORY / "paper_v_divergence_reproduction.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


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
