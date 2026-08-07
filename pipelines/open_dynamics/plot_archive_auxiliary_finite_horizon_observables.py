#!/usr/bin/env python3
"""Plot observables from the finite-horizon auxiliary-memory order curve."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from paper5.stability.hubbard_dimer import DimerParameters
from plot_high_coupling_controller_observables import observable_trajectories


DEFAULT_RUN = Path(
    "output/local_runs/"
    "paper_v_archive_auxiliary_finite_horizon_rollout_"
    "cutoff16_t4_20260805_v1/finite_horizon_order_curve.npz"
)
DEFAULT_OUTPUT = Path(
    "output/plots/paper_v_results_progression_20260804/"
    "finite_horizon_auxiliary_observables"
)

LANES = (
    ("exact", "exact cutoff-16", "#171717", "-", 1.7),
    ("archive", "raw archive EOM", "#b33b2e", "--", 1.25),
    ("pauli_archive", "Pauli-repaired archive EOM", "#1768ac", "-.", 1.35),
    ("balanced_union_p29_r78", r"finite-horizon $r=78$", "#178f68", "-", 1.35),
    ("balanced_union_p31_r81", r"full envelope $r=81$", "#7651a3", ":", 1.45),
)


def run(input_path: Path, output_stem: Path) -> None:
    parameters = DimerParameters(
        hopping=1.0,
        gamma=0.5,
        lambda_ep=1.5,
        drive_amplitude=1.0,
        pulse_width=1.0,
    )
    with np.load(input_path, allow_pickle=False) as payload:
        times = np.asarray(payload["times"], dtype=float)
        observables = {
            key: observable_trajectories(
                np.asarray(payload[f"coordinates__{key}"], dtype=float),
                parameters,
            )
            for key, _, _, _, _ in LANES
        }

    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "legend.fontsize": 7.3,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(9.25, 5.35), sharex=True)
    panels = (
        (0, r"site 0 occupation, $n_0=2\rho_{00}$", r"$n_0$"),
        (1, r"site 1 occupation, $n_1=2\rho_{11}$", r"$n_1$"),
        (2, "electronic energy", r"$E_{\rm e}/t_{\rm hop}$"),
        (3, "phonon energy", r"$E_{\rm ph}/t_{\rm hop}$"),
        (4, "electron--phonon energy", r"$E_{\rm e-ph}/t_{\rm hop}$"),
        (5, "total internal energy", r"$E_{\rm int}/t_{\rm hop}$"),
    )
    for panel, (observable, title, ylabel) in enumerate(panels):
        axis = axes.flat[panel]
        for key, label, color, linestyle, linewidth in LANES:
            axis.plot(
                times,
                observables[key][:, observable],
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
        axis.set_title(f"({chr(97 + panel)}) {title}", loc="left")
        axis.set_ylabel(ylabel)
        axis.set_xlim(float(times[0]), float(times[-1]))
        axis.grid(alpha=0.18, linewidth=0.5)
        if panel >= 3:
            axis.set_xlabel(r"$t\,t_{\rm hop}$")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=0.85,
        handlelength=2.1,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.10,
        top=0.875,
        hspace=0.34,
        wspace=0.30,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    run(arguments.input, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
