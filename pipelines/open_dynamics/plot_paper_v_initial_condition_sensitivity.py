"""Plot the stored Paper V initial-condition sensitivity diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper5.stability.hubbard_dimer import DimerParameters
from paper5.stability.matrix_reference import closed_scalar_to_matrix_state
from pipelines.open_dynamics.run_archive_long_horizon_observables import (
    _energy_components,
)


ROOT = Path(__file__).resolve().parents[2]
MATCHED_TRAJECTORY = ROOT / (
    "output/local_runs/"
    "paper_v_matched_exact_controller_sensitivity_t100_dt002_20260803_v1/"
    "trajectory.npz"
)
MATCHED_SUMMARY = MATCHED_TRAJECTORY.with_name("summary.json")
LYAPUNOV_TRAJECTORY = ROOT / (
    "output/local_runs/"
    "paper_v_postpulse_lyapunov_t1000_dt002_20260804_v1/trajectory.npz"
)
LYAPUNOV_SUMMARY = LYAPUNOV_TRAJECTORY.with_name("summary.json")
RAW_TRAJECTORY = ROOT / (
    "output/local_runs/"
    "paper_v_matched_exact_raw_sensitivity_t100_dt002_20260812_v1/"
    "trajectory.npz"
)
RAW_SUMMARY = RAW_TRAJECTORY.with_name("summary.json")
EXACT_SENSITIVITY_TRAJECTORY = ROOT / (
    "output/local_runs/"
    "paper_v_exact_reference_sensitivity_t100_cutoff12_16_20_20260803_v1/"
    "trajectory_cutoff_16.npz"
)
DEFAULT_OUTPUT = (
    ROOT / "output/plots/paper_v_results_progression_20260804"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def _matched_amplification(output_directory: Path) -> dict[str, object]:
    with np.load(MATCHED_TRAJECTORY, allow_pickle=False) as arrays:
        times = np.asarray(arrays["sample_times"], dtype=float)
        exact = np.asarray(arrays["exact_frobenius_distances"], dtype=float)
        corrected = np.asarray(
            arrays["sampled_frobenius_distances"], dtype=float
        )

    exact_amplification = exact / exact[:, :1]
    corrected_amplification = corrected / corrected[:, :1]
    direction_names = ("electronic drive", "relative phonon position")

    figure, axes = plt.subplots(1, 2, figsize=(10.4, 3.75), sharey=True)
    for index, (axis, direction) in enumerate(zip(axes, direction_names, strict=True)):
        axis.plot(
            times,
            exact_amplification[index],
            color="black",
            linewidth=2.1,
            label="exact cutoff-16 Hamiltonian",
        )
        axis.plot(
            times,
            corrected_amplification[index],
            color="#c43c39",
            linewidth=1.9,
            linestyle="--",
            label="corrected 31-coordinate EOM",
        )
        axis.axhline(1.0, color="#777777", linewidth=0.9, linestyle=":")
        axis.set_yscale("log")
        axis.set_xlim(0.0, 100.0)
        axis.set_ylim(0.2, 1000.0)
        axis.set_title(f"({chr(97 + index)}) {direction}", loc="left")
        axis.set_xlabel(r"$t\,t_{\rm hop}$")
        axis.grid(alpha=0.2, linewidth=0.5, which="both")
        axis.text(
            0.03,
            0.93,
            (
                f"final: {corrected_amplification[index, -1]:.1f}x corrected\n"
                f"{exact_amplification[index, -1]:.3f}x exact"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
    axes[0].set_ylabel(
        r"amplification $\|\delta x(t)\|_{\rm lift}/\|\delta x(0)\|_{\rm lift}$"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.17,
        top=0.82,
        wspace=0.10,
    )
    _save(figure, output_directory / "initial_condition_matched_amplification")

    return {
        "corrected_final_amplification": corrected_amplification[:, -1].tolist(),
        "corrected_maximum_amplification": np.max(
            corrected_amplification, axis=1
        ).tolist(),
        "exact_final_amplification": exact_amplification[:, -1].tolist(),
        "exact_maximum_amplification": np.max(exact_amplification, axis=1).tolist(),
    }


def _raw_matched_sensitivity(output_directory: Path) -> dict[str, object]:
    with np.load(RAW_TRAJECTORY, allow_pickle=False) as arrays:
        times = np.asarray(arrays["sample_times"], dtype=float)
        raw_amplification = np.asarray(arrays["raw_amplification"], dtype=float)
        exact_times = np.asarray(arrays["exact_times"], dtype=float)
        exact_distances = np.asarray(
            arrays["exact_frobenius_distances"], dtype=float
        )
        margins = np.asarray(arrays["raw_margins"][:, 0], dtype=float)
        trace_residual = np.asarray(
            arrays["raw_trace_residuals"][:, 0], dtype=float
        )

    exact_amplification = exact_distances / exact_distances[:, :1]
    direction_names = ("electronic drive", "relative phonon position")
    first_trace_violation = 0.1
    first_joint_violation = 0.2

    figure, axes = plt.subplots(1, 2, figsize=(10.4, 3.75), sharey=True)
    for index, (axis, direction) in enumerate(zip(axes, direction_names, strict=True)):
        axis.axvspan(
            first_trace_violation,
            times[-1],
            color="#b0b0b0",
            alpha=0.15,
            linewidth=0.0,
            label="raw state violates a retained constraint",
        )
        axis.plot(
            exact_times,
            exact_amplification[index],
            color="black",
            linewidth=2.1,
            label="exact cutoff-16 Hamiltonian",
        )
        axis.plot(
            times,
            raw_amplification[index],
            color="#d07a00",
            linewidth=1.9,
            linestyle="--",
            label="uncorrected 31-coordinate EOM",
        )
        axis.axhline(1.0, color="#777777", linewidth=0.9, linestyle=":")
        axis.set_yscale("log")
        axis.set_xlim(0.0, 100.0)
        axis.set_ylim(0.2, 1.0e5)
        axis.set_title(f"({chr(97 + index)}) {direction}", loc="left")
        axis.set_xlabel(r"$t\,t_{\rm hop}$")
        axis.grid(alpha=0.2, linewidth=0.5, which="both")
        axis.text(
            0.03,
            0.93,
            (
                f"final: {raw_amplification[index, -1]:.2e} raw\n"
                f"{exact_amplification[index, -1]:.3f} exact"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
    axes[0].set_ylabel(
        r"amplification $\|\delta x(t)\|_{\rm lift}/\|\delta x(0)\|_{\rm lift}$"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.17,
        top=0.82,
        wspace=0.10,
    )
    _save(figure, output_directory / "initial_condition_raw_matched_amplification")

    early = times <= 4.0 + 1e-12
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 3.75))
    margin_labels = (
        "electron lower",
        "electron upper",
        "boson moment",
        "joint Gram",
    )
    margin_styles = ("-", ":", "-.", "--")
    margin_colors = ("#6b4c9a", "#6b4c9a", "#198f8f", "#c43c39")
    for index, label in enumerate(margin_labels):
        axes[0].plot(
            times[early],
            margins[early, index],
            color=margin_colors[index],
            linestyle=margin_styles[index],
            linewidth=1.7,
            label=label,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.9)
    axes[0].set_yscale("symlog", linthresh=1e-5)
    axes[0].set_xlim(0.0, 4.0)
    axes[0].set_title("(a) raw-EOM cone margins", loc="left")
    axes[0].set_xlabel(r"$t\,t_{\rm hop}$")
    axes[0].set_ylabel("minimum eigenvalue")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[0].grid(alpha=0.2, linewidth=0.5, which="both")

    positive_trace = np.maximum(trace_residual[early], np.finfo(float).tiny)
    axes[1].plot(
        times[early],
        positive_trace,
        color="#d07a00",
        linewidth=1.9,
    )
    axes[1].axhline(
        1e-8,
        color="black",
        linewidth=0.9,
        linestyle=":",
        label=r"$10^{-8}$ diagnostic floor",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlim(0.0, 4.0)
    axes[1].set_title("(b) correlation-trace violation", loc="left")
    axes[1].set_xlabel(r"$t\,t_{\rm hop}$")
    axes[1].set_ylabel(r"$\max_q|\operatorname{Tr}C^q|$")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.2, linewidth=0.5, which="both")
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.17,
        top=0.90,
        wspace=0.25,
    )
    _save(figure, output_directory / "initial_condition_raw_physicality")

    return {
        "raw_final_amplification": raw_amplification[:, -1].tolist(),
        "raw_maximum_amplification": np.max(raw_amplification, axis=1).tolist(),
        "exact_final_amplification": exact_amplification[:, -1].tolist(),
        "exact_maximum_amplification": np.max(exact_amplification, axis=1).tolist(),
        "first_sampled_correlation_trace_violation": first_trace_violation,
        "first_sampled_joint_gram_violation": first_joint_violation,
        "minimum_margins": np.min(margins, axis=0).tolist(),
        "maximum_correlation_trace_residual": float(np.max(trace_residual)),
    }


def _observable_series(times: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    parameters = DimerParameters(
        hopping=1.0,
        gamma=0.5,
        lambda_ep=1.5,
        drive_amplitude=1.0,
    )
    result = np.empty((times.size, 6), dtype=float)
    for index, (time_value, row) in enumerate(
        zip(times, coordinates, strict=True)
    ):
        state = closed_scalar_to_matrix_state(row)
        energies = _energy_components(state, parameters, float(time_value))
        result[index] = (
            2.0 * float(state.electron_density[0, 0].real),
            2.0 * float(state.electron_density[1, 1].real),
            float(energies[0]),
            float(energies[1]),
            float(energies[2]),
            float(energies[3]),
        )
    return result


def _raw_observable_separation(output_directory: Path) -> dict[str, object]:
    with np.load(RAW_TRAJECTORY, allow_pickle=False) as raw_arrays:
        raw_times = np.asarray(raw_arrays["sample_times"], dtype=float)
        raw_states = np.asarray(raw_arrays["raw_sampled_states"], dtype=float)
    with np.load(EXACT_SENSITIVITY_TRAJECTORY, allow_pickle=False) as exact_arrays:
        exact_times = np.asarray(exact_arrays["times"], dtype=float)
        exact_base_coordinates = np.asarray(
            exact_arrays["base_coordinates"], dtype=float
        )
        exact_shadow_coordinates = np.asarray(
            exact_arrays["shadow_coordinates"][:, 0], dtype=float
        )

    exact_base = _observable_series(exact_times, exact_base_coordinates)
    raw_base = _observable_series(raw_times, raw_states[:, 0])
    observable_names = (
        r"site 0 occupation, $n_0$",
        r"site 1 occupation, $n_1$",
        "electronic energy",
        "phonon energy",
        "electron-phonon energy",
        "total internal energy",
    )
    direction_names = ("electronic-drive", "relative-phonon-position")
    metrics: dict[str, object] = {}

    for direction_index, direction_name in enumerate(direction_names):
        exact_shadow = _observable_series(
            exact_times,
            exact_shadow_coordinates[direction_index],
        )
        raw_shadow = _observable_series(
            raw_times,
            raw_states[:, direction_index + 1],
        )
        figure, axes = plt.subplots(2, 3, figsize=(10.4, 6.2), sharex=True)
        for panel, axis in enumerate(axes.flat):
            axis.plot(
                exact_times,
                exact_base[:, panel],
                color="black",
                linewidth=2.0,
                label="exact base",
            )
            axis.plot(
                exact_times,
                exact_shadow[:, panel],
                color="#777777",
                linewidth=1.5,
                linestyle=":",
                label="exact perturbed",
            )
            axis.plot(
                raw_times,
                raw_base[:, panel],
                color="#d07a00",
                linewidth=1.7,
                linestyle="--",
                label="raw-EOM base",
            )
            axis.plot(
                raw_times,
                raw_shadow[:, panel],
                color="#b23a48",
                linewidth=1.5,
                linestyle="-.",
                label="raw-EOM perturbed",
            )
            axis.set_xlim(0.0, 100.0)
            axis.set_title(f"({chr(97 + panel)}) {observable_names[panel]}", loc="left")
            axis.set_ylabel(
                "occupation" if panel < 2 else r"energy / $t_{\rm hop}$"
            )
            axis.grid(alpha=0.2, linewidth=0.5)
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
            left=0.08,
            right=0.985,
            bottom=0.10,
            top=0.89,
            hspace=0.38,
            wspace=0.32,
        )
        _save(
            figure,
            output_directory / f"initial_condition_raw_{direction_name}_observables",
        )

        raw_difference = raw_shadow - raw_base
        exact_difference = exact_shadow - exact_base
        figure, axes = plt.subplots(2, 3, figsize=(10.4, 6.2), sharex=True)
        for panel, axis in enumerate(axes.flat):
            axis.plot(
                exact_times,
                np.maximum(np.abs(exact_difference[:, panel]), 1e-16),
                color="black",
                linewidth=2.0,
                label=r"exact $|O_{\rm pert}-O_{\rm base}|$",
            )
            axis.plot(
                raw_times,
                np.maximum(np.abs(raw_difference[:, panel]), 1e-16),
                color="#d07a00",
                linewidth=1.8,
                linestyle="--",
                label=r"raw EOM $|O_{\rm pert}-O_{\rm base}|$",
            )
            axis.set_yscale("log")
            axis.set_xlim(0.0, 100.0)
            axis.set_title(f"({chr(97 + panel)}) {observable_names[panel]}", loc="left")
            axis.set_ylabel(
                "absolute occupation difference"
                if panel < 2
                else r"absolute energy difference / $t_{\rm hop}$"
            )
            axis.grid(alpha=0.2, linewidth=0.5, which="both")
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
        )
        figure.subplots_adjust(
            left=0.09,
            right=0.985,
            bottom=0.10,
            top=0.89,
            hspace=0.38,
            wspace=0.35,
        )
        _save(
            figure,
            output_directory
            / f"initial_condition_raw_{direction_name}_observable_separation",
        )

        metrics[direction_name] = {
            "raw_maximum_absolute_observable_difference": np.max(
                np.abs(raw_difference), axis=0
            ).tolist(),
            "raw_final_observable_difference": raw_difference[-1].tolist(),
            "exact_maximum_absolute_observable_difference": np.max(
                np.abs(exact_difference), axis=0
            ).tolist(),
            "exact_final_observable_difference": exact_difference[-1].tolist(),
        }
    return metrics


def _rolling_weighted_mean(
    values: np.ndarray,
    interval_widths: np.ndarray,
    window_intervals: int,
) -> np.ndarray:
    weighted = values * interval_widths[None, :]
    kernel = np.ones(window_intervals, dtype=float)
    numerator = np.stack(
        [np.convolve(row, kernel, mode="valid") for row in weighted]
    )
    denominator = np.convolve(interval_widths, kernel, mode="valid")
    return numerator / denominator[None, :]


def _lyapunov_convergence(output_directory: Path) -> dict[str, object]:
    with np.load(LYAPUNOV_TRAJECTORY, allow_pickle=False) as arrays:
        times = np.asarray(arrays["times"], dtype=float)
        local = np.asarray(arrays["local_exponents"], dtype=float)
        cumulative = np.asarray(arrays["cumulative_exponents"], dtype=float)

    interval_widths = np.diff(times)
    end_times = times[1:]
    nominal_window = 250.0
    window_intervals = int(round(nominal_window / float(np.median(interval_widths))))
    rolling = _rolling_weighted_mean(local, interval_widths, window_intervals)
    rolling_times = end_times[window_intervals - 1 :]
    direction_names = ("electronic drive", "relative phonon position")
    colors = ("#6b4c9a", "#198f8f")
    line_styles = ("-", "--")

    figure, axes = plt.subplots(1, 2, figsize=(10.4, 3.75))
    for index, (direction, color, line_style) in enumerate(
        zip(direction_names, colors, line_styles, strict=True)
    ):
        axes[0].plot(
            end_times,
            cumulative[index],
            color=color,
            linewidth=1.8,
            linestyle=line_style,
            label=direction,
        )
        axes[1].plot(
            rolling_times,
            rolling[index],
            color=color,
            linewidth=1.8,
            linestyle=line_style,
            label=direction,
        )
    for axis in axes:
        axis.axhline(0.0, color="black", linewidth=0.9, linestyle=":")
        axis.set_xlim(4.0, 1000.0)
        axis.set_xlabel(r"$t\,t_{\rm hop}$")
        axis.set_ylabel(r"finite-time exponent / $t_{\rm hop}$")
        axis.grid(alpha=0.2, linewidth=0.5)
    axes[0].set_title("(a) cumulative exponent after the pulse", loc="left")
    axes[1].set_title("(b) trailing 250-unit exponent", loc="left")
    axes[0].text(
        0.97,
        0.08,
        (
            f"final: {cumulative[0, -1]:.5f}\n"
            f"{cumulative[1, -1]:.5f}"
        ),
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.17,
        top=0.82,
        wspace=0.25,
    )
    _save(figure, output_directory / "initial_condition_lyapunov_convergence")

    return {
        "cumulative_final": cumulative[:, -1].tolist(),
        "trailing_250_final": rolling[:, -1].tolist(),
        "window_width": nominal_window,
    }


def main() -> None:
    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    matched = _matched_amplification(DEFAULT_OUTPUT)
    lyapunov = _lyapunov_convergence(DEFAULT_OUTPUT)
    raw = _raw_matched_sensitivity(DEFAULT_OUTPUT)
    raw_observables = _raw_observable_separation(DEFAULT_OUTPUT)
    summary = {
        "classification": "stored_initial_condition_sensitivity_visualization",
        "matched_t100": matched,
        "postpulse_t1000": lyapunov,
        "raw_matched_t100": raw,
        "raw_matched_observables": raw_observables,
        "input_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                MATCHED_TRAJECTORY,
                MATCHED_SUMMARY,
                LYAPUNOV_TRAJECTORY,
                LYAPUNOV_SUMMARY,
                RAW_TRAJECTORY,
                RAW_SUMMARY,
                EXACT_SENSITIVITY_TRAJECTORY,
            )
        },
    }
    output_path = DEFAULT_OUTPUT / "initial_condition_sensitivity_metrics.json"
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
