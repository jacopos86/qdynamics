"""Plot observables from the matched raw/corrected high-coupling EOM run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper5.stability.hubbard_dimer import DimerParameters
from paper5.stability.matrix_reference import (
    closed_scalar_to_matrix_state,
    local_holstein_couplings,
)


OBSERVABLE_NAMES = (
    "site_0_occupation",
    "site_1_occupation",
    "electronic_energy",
    "phonon_energy",
    "electron_phonon_energy",
    "internal_energy",
)


def observable_trajectories(
    coordinates: np.ndarray,
    parameters: Any,
) -> np.ndarray:
    couplings = local_holstein_couplings(parameters)
    bare_electron = np.array(
        [[0.0, -parameters.hopping], [-parameters.hopping, 0.0]],
        dtype=complex,
    )
    values = np.empty((coordinates.shape[0], len(OBSERVABLE_NAMES)), dtype=float)
    for index, row in enumerate(coordinates):
        state = closed_scalar_to_matrix_state(row)
        rho = np.asarray(state.electron_density, dtype=complex)
        coherent = np.asarray(state.coherent_phonon, dtype=complex)
        normal = np.asarray(state.phonon_density, dtype=complex)
        correlation = np.asarray(
            state.electron_phonon_correlation,
            dtype=complex,
        )

        electronic = 2.0 * np.trace(bare_electron @ rho).real
        phonon = parameters.omega_ph * (
            np.vdot(coherent, coherent).real + np.trace(normal).real
        )
        interaction_amplitude = 0.0j
        for mode in range(2):
            for one in range(2):
                for two in range(2):
                    interaction_amplitude += couplings[mode, one, two] * (
                        coherent[mode] * rho[two, one]
                        + correlation[mode, two, one]
                    )
        electron_phonon = 4.0 * interaction_amplitude.real
        values[index] = (
            2.0 * rho[0, 0].real,
            2.0 * rho[1, 1].real,
            electronic,
            phonon,
            electron_phonon,
            electronic + phonon + electron_phonon,
        )
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output or args.run_dir / "observables_raw_vs_corrected.png"
    with (args.run_dir / "plan.json").open(encoding="utf-8") as handle:
        plan = json.load(handle)
    with (args.run_dir / "metrics.json").open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    archive = np.load(args.run_dir / "trajectories.npz")

    parameter_values = plan["parameters"]
    parameters = DimerParameters(
        hopping=float(parameter_values["hopping"]),
        gamma=float(parameter_values["gamma"]),
        lambda_ep=float(parameter_values["lambda_ep"]),
        drive_amplitude=float(plan["drive"]["amplitude"]),
        pulse_width=float(plan["drive"]["pulse_width"]),
    )
    lanes: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "raw": (
            np.asarray(archive["raw_times"], dtype=float),
            observable_trajectories(
                np.asarray(archive["raw_states"], dtype=float),
                parameters,
            ),
        ),
        "corrected": (
            np.asarray(archive["corrected_times"], dtype=float),
            observable_trajectories(
                np.asarray(archive["corrected_states"], dtype=float),
                parameters,
            ),
        ),
    }
    exact_path = args.run_dir / "exact_trajectory.npz"
    if exact_path.exists():
        exact_archive = np.load(exact_path)
        lanes = {
            "exact": (
                np.asarray(exact_archive["times"], dtype=float),
                observable_trajectories(
                    np.asarray(exact_archive["states"], dtype=float),
                    parameters,
                ),
            ),
            **lanes,
        }
    np.testing.assert_allclose(
        lanes["raw"][1][:, 5],
        archive["raw_internal_energies"],
        atol=2e-12,
        rtol=2e-12,
    )
    np.testing.assert_allclose(
        lanes["corrected"][1][:, 5],
        archive["corrected_internal_energies"],
        atol=2e-12,
        rtol=2e-12,
    )

    plt.rcParams.update(
        {
            "font.size": 8.2,
            "axes.titlesize": 8.7,
            "axes.labelsize": 8.2,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(9.2, 5.5), sharex=True)
    panel_specs = (
        (0, r"site 0 occupation, $n_0=2\rho_{00}$", r"$n_0$"),
        (1, r"site 1 occupation, $n_1=2\rho_{11}$", r"$n_1$"),
        (2, "electronic energy", r"$E_{\mathrm{e}}/t_{\mathrm{hop}}$"),
        (3, "phonon energy", r"$E_{\mathrm{ph}}/t_{\mathrm{hop}}$"),
        (4, "electron--phonon energy", r"$E_{\mathrm{e-ph}}/t_{\mathrm{hop}}$"),
        (5, "total internal energy", r"$E_{\mathrm{int}}/t_{\mathrm{hop}}$"),
    )
    styles = {
        "exact": ("exact cutoff-16 Hamiltonian", "#171717", 1.75),
        "raw": ("uncorrected 31-moment EOM", "#b33b2e", 1.45),
        "corrected": ("representability-corrected EOM", "#1768ac", 1.65),
    }
    pulse_width = 4.0
    pulse_delays = tuple(float(value) for value in plan["drive"]["delays"])
    raw_failure_time = metrics["raw"]["failure_time"]
    final_time = float(lanes["corrected"][0][-1])

    for panel, (observable, title, ylabel) in enumerate(panel_specs):
        axis = axes.flat[panel]
        for delay in pulse_delays:
            axis.axvspan(
                delay,
                delay + pulse_width,
                color="#b9d7e8",
                alpha=0.20,
                linewidth=0.0,
                zorder=0,
            )
        for lane, (times, values) in lanes.items():
            label, color, width = styles[lane]
            axis.plot(
                times,
                values[:, observable],
                color=color,
                linewidth=width,
                label=label,
                zorder=2,
            )
        if raw_failure_time is not None:
            axis.axvline(
                float(raw_failure_time),
                color="#7d2520",
                linestyle="--",
                linewidth=0.9,
                alpha=0.8,
                zorder=1,
            )
        if observable in (0, 1):
            axis.axhspan(0.0, 2.0, color="#6aa56a", alpha=0.08, zorder=0)
            axis.axhline(0.0, color="#555555", linewidth=0.55, alpha=0.55)
            axis.axhline(2.0, color="#555555", linewidth=0.55, alpha=0.55)
        axis.set_xlim(0.0, final_time)
        axis.set_title(f"({chr(97 + panel)}) {title}", loc="left")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.18, linewidth=0.5)
        if panel >= 3:
            axis.set_xlabel(r"$t\,t_{\rm hop}$")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend_handles = list(handles)
    legend_labels = list(labels)
    if raw_failure_time is not None:
        failure_handle = plt.Line2D(
            [0],
            [0],
            color="#7d2520",
            linestyle="--",
            linewidth=0.9,
            label=rf"raw threshold stop, $t={float(raw_failure_time):g}$",
        )
        legend_handles.append(failure_handle)
        legend_labels.append(failure_handle.get_label())
    figure.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        ncol=len(legend_handles),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
    )
    coupling = float(parameter_values["coupling_g"])
    lambda_ep = float(parameter_values["lambda_ep"])
    comparison_label = (
        "exact, uncorrected, and corrected dynamics"
        if "exact" in lanes
        else "uncorrected and corrected 31-moment EOMs"
    )
    figure.suptitle(
        rf"$g={coupling:.4g}$, $\lambda={lambda_ep:.4g}$: {comparison_label}",
        fontsize=10.2,
        y=1.025,
    )
    footer = (
        "Matched exact reference: cutoff-16 wavefunction propagated with DOP853."
        if "exact" in lanes
        else "No matched exact trajectory is scored."
    )
    figure.text(
        0.5,
        0.012,
        (
            f"{plan['initial_condition']['description']}; "
            f"{footer}"
        ),
        ha="center",
        va="bottom",
        fontsize=7.2,
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.12,
        top=0.88,
        hspace=0.32,
        wspace=0.30,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
