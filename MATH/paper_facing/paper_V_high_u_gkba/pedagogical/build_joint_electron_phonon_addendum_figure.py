"""Build the addendum figure for the joint electron--phonon correction."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from paper5.stability import (
    DimerParameters,
    closed_electron_phonon_cone_projected_rhs,
    closed_scalar_rhs,
    electron_phonon_moment_matrix,
    exact_ground_closed_scalar_coordinates,
    exact_holstein_driven_trajectory,
)
from paper5.stability.matrix_reference import closed_scalar_to_matrix_state


REPO_ROOT = Path(__file__).resolve().parents[4]
RUN_DIR = (
    REPO_ROOT
    / "output/local_runs/"
    "paper_v_joint_moment_barrier_t1000_local_20260801_v1"
)
ANALYSIS_RUN_DIR = (
    REPO_ROOT
    / "output/local_runs/"
    "paper_v_electron_phonon_analysis_20260801_v3"
)
FIGURE_DIR = Path(__file__).resolve().parent / "figures"
PDF_PATH = FIGURE_DIR / "joint_electron_phonon_addendum_evidence.pdf"
PNG_PATH = FIGURE_DIR / "joint_electron_phonon_addendum_evidence.png"
ACCURACY_PDF_PATH = (
    FIGURE_DIR / "joint_electron_phonon_accuracy_evidence.pdf"
)
ACCURACY_PNG_PATH = (
    FIGURE_DIR / "joint_electron_phonon_accuracy_evidence.png"
)
MECHANISM_PDF_PATH = (
    FIGURE_DIR / "joint_electron_phonon_mechanism_evidence.pdf"
)
MECHANISM_PNG_PATH = (
    FIGURE_DIR / "joint_electron_phonon_mechanism_evidence.png"
)

COORDINATE_BLOCKS = {
    r"$\rho$": slice(0, 3),
    r"$B$": slice(3, 7),
    r"$N$": slice(7, 11),
    r"$A$": slice(11, 17),
    r"$C$": slice(17, 31),
}


def joint_minimum(state: np.ndarray) -> float:
    matrix_state = closed_scalar_to_matrix_state(state)
    return float(
        np.linalg.eigvalsh(electron_phonon_moment_matrix(matrix_state))[0]
    )


def fixed_rk4(
    rhs,
    initial: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    state = np.asarray(initial, dtype=float).copy()
    states = [state.copy()]
    for left, right in zip(times[:-1], times[1:], strict=True):
        step = float(right - left)
        k1 = rhs(float(left), state)
        k2 = rhs(float(left + 0.5 * step), state + 0.5 * step * k1)
        k3 = rhs(float(left + 0.5 * step), state + 0.5 * step * k2)
        k4 = rhs(float(right), state + step * k3)
        state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        states.append(state.copy())
    return np.asarray(states)


def load_progress() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    records: list[dict[str, object]] = []
    with (RUN_DIR / "progress.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") in {"started", "resumed", "checkpoint", "complete"}:
                records.append(record)

    # A resumed run can repeat a checkpoint.  Keep the last record at each time.
    by_time = {
        float(record.get("time", record["final_time"])): record
        for record in records
    }
    ordered = [by_time[time] for time in sorted(by_time)]
    times = np.asarray(
        [float(record.get("time", record["final_time"])) for record in ordered]
    )
    keys = (
        "minimum_joint_moment_eigenvalue",
        "minimum_boson_moment_eigenvalue",
        "minimum_electron_eigenvalue",
        "maximum_electron_eigenvalue",
        "maximum_correlation_trace_absolute_value",
        "post_pulse_maximum_drift_from_t4",
    )
    series = {
        key: np.asarray([float(record["stats"][key]) for record in ordered])
        for key in keys
    }
    return times, series


def correlation_error(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """Return the row-wise Frobenius error of the reconstructed C block."""

    return np.asarray(
        [
            np.linalg.norm(
                closed_scalar_to_matrix_state(
                    left
                ).electron_phonon_correlation
                - closed_scalar_to_matrix_state(
                    right
                ).electron_phonon_correlation
            )
            for left, right in zip(first, second, strict=True)
        ]
    )


def build_analysis_figures() -> None:
    """Build compact figures from the completed exact/raw/corrected analysis."""

    with np.load(ANALYSIS_RUN_DIR / "baseline_trajectories.npz") as arrays:
        times = arrays["baseline__times"]
        exact = arrays["baseline__exact"]
        raw = arrays["baseline__raw"]
        corrected = arrays["baseline__corrected"]
        correction = arrays["baseline__correction"]
        equality = arrays["baseline__equality_only_correction"]
        mode_weights = arrays["baseline__joint_mode_weights"]

    raw_c_error = correlation_error(raw, exact)
    corrected_c_error = correlation_error(corrected, exact)
    exact_joint = np.asarray([joint_minimum(row) for row in exact])
    raw_joint = np.asarray([joint_minimum(row) for row in raw])
    corrected_joint = np.asarray([joint_minimum(row) for row in corrected])

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(3.45, 4.25),
        constrained_layout=True,
    )
    axes[0].plot(times, raw_c_error, color="#9B2C2C", label="raw closure")
    axes[0].plot(
        times,
        corrected_c_error,
        color="#17365D",
        label="joint barrier",
    )
    axes[0].set_title(r"$C$-sector error against exact contractions")
    axes[0].set_ylabel(r"$\|C-C_{\rm ex}\|_{\rm F}$")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.18, linewidth=0.5)

    axes[1].plot(times, exact_joint, color="#2F6B4F", label="exact")
    axes[1].plot(times, raw_joint, color="#9B2C2C", label="raw closure")
    axes[1].plot(
        times,
        corrected_joint,
        color="#17365D",
        label="joint barrier",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.65)
    axes[1].set_title("Joint representability certificate")
    axes[1].set_xlabel(r"time $t\,t_{\mathrm{hop}}$")
    axes[1].set_ylabel(r"$\lambda_{\min}(\mathcal{G})$")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.18, linewidth=0.5)
    figure.savefig(ACCURACY_PDF_PATH, bbox_inches="tight")
    figure.savefig(ACCURACY_PNG_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)

    cone_action = correction - equality
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(3.45, 4.25),
        constrained_layout=True,
    )
    for name, block in COORDINATE_BLOCKS.items():
        axes[0].plot(
            times,
            np.linalg.norm(cone_action[:, block], axis=1),
            label=name,
        )
    axes[0].set_title("Additional cone action by coordinate block")
    axes[0].set_ylabel(r"$\|w_{\rm g}\|_2$")
    axes[0].legend(ncol=3, frameon=False)
    axes[0].grid(alpha=0.18, linewidth=0.5)

    axes[1].stackplot(
        times,
        mode_weights[:, :4].sum(axis=1),
        mode_weights[:, 4:].sum(axis=1),
        labels=("bosonic entries", "electronic entries"),
        colors=("#D4A017", "#3F7CAC"),
        alpha=0.82,
    )
    axes[1].set_title("Composition of the weakest joint mode")
    axes[1].set_xlabel(r"time $t\,t_{\mathrm{hop}}$")
    axes[1].set_ylabel("mode weight")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.18, linewidth=0.5)
    figure.savefig(MECHANISM_PDF_PATH, bbox_inches="tight")
    figure.savefig(MECHANISM_PNG_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parameters = DimerParameters(
        hopping=1.0,
        gamma=0.5,
        lambda_ep=1.5,
        drive_amplitude=1.0,
        pulse_width=1.0,
    )
    initial = exact_ground_closed_scalar_coordinates(
        parameters,
        phonon_cutoff=16,
    )
    short_times = np.linspace(0.0, 0.2, 101)

    raw = solve_ivp(
        lambda time, state: closed_scalar_rhs(time, state, parameters),
        (0.0, 0.2),
        initial,
        method="DOP853",
        t_eval=short_times,
        rtol=1e-11,
        atol=1e-13,
        max_step=0.002,
    )
    if not raw.success:
        raise RuntimeError(raw.message)

    exact = exact_holstein_driven_trajectory(
        parameters,
        sample_times=short_times,
        phonon_cutoff=16,
        relative_tolerance=1e-11,
        absolute_tolerance=1e-13,
        maximum_step=0.002,
    )
    corrected_rhs = closed_electron_phonon_cone_projected_rhs(
        parameters,
        initial,
        activation_margin=1e-5,
        barrier_rate=5.0,
        energy_neutral=True,
        preserve_correlation_trace=True,
        cone_tolerance=1e-8,
    )
    corrected = fixed_rk4(corrected_rhs, initial, short_times)

    raw_joint = np.asarray(
        [joint_minimum(raw.y[:, index]) for index in range(raw.y.shape[1])]
    )
    exact_joint = np.asarray(
        [
            float(np.linalg.eigvalsh(electron_phonon_moment_matrix(state))[0])
            for state in exact.matrix_states
        ]
    )
    corrected_joint = np.asarray([joint_minimum(state) for state in corrected])

    long_times, long = load_progress()
    upper_slack = 1.0 - long["maximum_electron_eigenvalue"]

    plt.rcParams.update(
        {
            "font.size": 7.4,
            "axes.labelsize": 7.4,
            "axes.titlesize": 8.0,
            "legend.fontsize": 6.6,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "lines.linewidth": 1.15,
        }
    )
    figure, axes = plt.subplots(2, 1, figsize=(3.45, 4.25), constrained_layout=True)

    first = axes[0]
    first.axhline(0.0, color="black", linewidth=0.65)
    first.plot(short_times, raw_joint, color="#9B2C2C", label="raw 31D closure")
    first.plot(short_times, exact_joint, color="#2B6CB0", label="exact truncated state")
    first.plot(
        short_times,
        corrected_joint,
        color="#2F6B4F",
        label="joint-barrier closure",
    )
    first.axvline(0.1607116782, color="#9B2C2C", linestyle=":", linewidth=0.9)
    first.set_title("Earliest mixed representability event")
    first.set_xlabel(r"time $t\,t_{\mathrm{hop}}$")
    first.set_ylabel(r"$\lambda_{\min}(\mathcal{G})$")
    first.legend(frameon=False, loc="best")
    first.grid(alpha=0.18, linewidth=0.5)

    second = axes[1]
    second.plot(
        long_times,
        long["minimum_joint_moment_eigenvalue"],
        color="#17365D",
        label=r"running min $\lambda_{\min}(\mathcal{G})$",
    )
    second.plot(
        long_times,
        long["minimum_boson_moment_eigenvalue"],
        color="#2B6CB0",
        linestyle="--",
        label=r"running min $\lambda_{\min}(M_{\mathrm{B}})$",
    )
    second.plot(
        long_times,
        long["minimum_electron_eigenvalue"],
        color="#2F6B4F",
        linestyle="-.",
        label=r"running min $\lambda_{\min}(\rho)$",
    )
    second.plot(
        long_times,
        upper_slack,
        color="#805AD5",
        linestyle=":",
        label=r"running min $1-\lambda_{\max}(\rho)$",
    )
    second.axhline(1e-5, color="black", linewidth=0.65, alpha=0.65)
    second.set_yscale("log")
    second.set_title("Corrected long-horizon certificates")
    second.set_xlabel(r"time $t\,t_{\mathrm{hop}}$")
    second.set_ylabel("smallest retained margin")
    second.legend(frameon=False, loc="best")
    second.grid(alpha=0.18, linewidth=0.5, which="both")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(PDF_PATH, bbox_inches="tight")
    figure.savefig(PNG_PATH, dpi=220, bbox_inches="tight")
    plt.close(figure)
    build_analysis_figures()


if __name__ == "__main__":
    main()
