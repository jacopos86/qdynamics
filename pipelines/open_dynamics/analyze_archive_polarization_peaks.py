"""Measure polarization peak positions and finite-window broadening.

This is postprocessing of completed trajectories.  Exact data are never used
online.  The weak and strong comparisons use the same post-pulse interval,
sampling, mean subtraction, and Hann window.  Each figure shows both the
common-scale response power and the independently normalized spectral shape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipelines.open_dynamics.analyze_archive_observable_spectra import (
    Spectrum,
    _band_power_ratio,
    _band_peak,
    _hellinger,
    _load_lane,
    _power_ratio,
    _reference_scaled_power_density,
    _spectrum,
    _time_grid,
)


LANES = ("exact", "raw", "corrected")
COLORS = {
    "exact": "#171717",
    "raw": "#c4513c",
    "corrected": "#3569a8",
}
LINESTYLES = {"exact": "-", "raw": "-.", "corrected": "--"}
LABELS = {
    "exact": "exact cutoff-16",
    "raw": "uncorrected archive EOM",
    "corrected": "regular EOM correction (31D joint-Gram)",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _interpolate(
    source_times: np.ndarray,
    source_values: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    tolerance = 1e-10
    if (
        target_times[0] < source_times[0] - tolerance
        or target_times[-1] > source_times[-1] + tolerance
    ):
        raise ValueError("the requested spectral interval is not recorded")
    return np.interp(target_times, source_times, source_values)


def _strong_polarization(
    run_dir: Path,
    target_times: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Path]]:
    files = {
        "exact": run_dir / "exact_trajectory.npz",
        "raw": run_dir / "raw_refined_rk4_dt005_trajectory.npz",
        "corrected": run_dir / "corrected_trajectory.npz",
    }
    values: dict[str, np.ndarray] = {}
    for lane, path in files.items():
        payload = _load_lane(path)
        occupation = np.asarray(payload["site_occupation"], dtype=float)
        # P=(rho_11-rho_00)/2 differs from rho_00 only by sign and a
        # constant under unit trace; both disappear from normalized power.
        polarization = 0.5 - occupation
        values[lane] = _interpolate(
            np.asarray(payload["times"], dtype=float),
            polarization,
            target_times,
        )
    return values, files


def _weak_polarization(
    run_dir: Path,
    target_times: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Path]]:
    moment_path = run_dir / "trajectories.npz"
    exact_path = run_dir / "exact_trajectory.npz"
    with np.load(moment_path, allow_pickle=False) as payload:
        raw_times = np.asarray(payload["raw_times"], dtype=float)
        raw_delta_n = np.asarray(payload["raw_states"], dtype=float)[:, 0]
        corrected_times = np.asarray(payload["corrected_times"], dtype=float)
        corrected_delta_n = np.asarray(
            payload["corrected_states"], dtype=float
        )[:, 0]
    with np.load(exact_path, allow_pickle=False) as payload:
        exact_times = np.asarray(payload["times"], dtype=float)
        exact_delta_n = np.asarray(payload["states"], dtype=float)[:, 0]

    # The first coordinate is Delta n=rho_00-rho_11, hence P=-Delta n/2.
    values = {
        "exact": _interpolate(
            exact_times, -0.5 * exact_delta_n, target_times
        ),
        "raw": _interpolate(raw_times, -0.5 * raw_delta_n, target_times),
        "corrected": _interpolate(
            corrected_times, -0.5 * corrected_delta_n, target_times
        ),
    }
    return values, {"moments": moment_path, "exact": exact_path}


def _metrics(spectra: dict[str, Spectrum]) -> dict[str, Any]:
    exact = spectra["exact"]
    result: dict[str, Any] = {}
    for lane, spectrum in spectra.items():
        result[lane] = {
            "hellinger_distance_to_exact": _hellinger(spectrum, exact),
            "total_hann_power_ratio_to_exact": _power_ratio(spectrum, exact),
            "oscillation_rms_ratio_to_exact": (
                spectrum.oscillation_rms / exact.oscillation_rms
            ),
            "electronic_band_hann_power_ratio_to_exact": _band_power_ratio(
                spectrum,
                exact,
                minimum_angular_frequency=1.5,
                maximum_angular_frequency=3.5,
            ),
            "low_frequency_peak": asdict(
                _band_peak(
                    spectrum,
                    minimum_angular_frequency=0.05,
                    maximum_angular_frequency=1.0,
                )
            ),
            "electronic_band_peak": asdict(
                _band_peak(
                    spectrum,
                    minimum_angular_frequency=1.5,
                    maximum_angular_frequency=3.5,
                )
            ),
        }
    return result


def _plot(
    spectra_by_regime: dict[str, dict[str, Spectrum]],
    metrics_by_regime: dict[str, dict[str, Any]],
    output_stem: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 4.35), sharex=True)
    titles = {
        "weak": r"$g/t_{\rm hop}=0.3536$ ($\lambda=0.5$)",
        "strong": r"$g/t_{\rm hop}=0.6124$ ($\lambda=1.5$)",
    }
    column_titles = (
        "common scale (exact total power = 1)",
        "shape only (each spectrum has unit area)",
    )
    for row, regime in enumerate(("weak", "strong")):
        exact = spectra_by_regime[regime]["exact"]
        for column, axis in enumerate(axes[row]):
            for lane in LANES:
                spectrum = spectra_by_regime[regime][lane]
                frequency = spectrum.angular_frequency
                spacing = float(frequency[1] - frequency[0])
                if column == 0:
                    density = _reference_scaled_power_density(spectrum, exact)
                    peak_scale = metrics_by_regime[regime][lane][
                        "total_hann_power_ratio_to_exact"
                    ]
                else:
                    density = spectrum.normalized_power / spacing
                    peak_scale = 1.0
                visible = (frequency > 0.0) & (frequency <= 4.0)
                axis.semilogy(
                    frequency[visible],
                    density[visible],
                    color=COLORS[lane],
                    linestyle=LINESTYLES[lane],
                    linewidth=1.0,
                    label=LABELS[lane],
                )
                peak = metrics_by_regime[regime][lane]["electronic_band_peak"]
                axis.plot(
                    peak["angular_frequency"],
                    peak["normalized_power_density"] * peak_scale,
                    marker="o",
                    markersize=2.8,
                    color=COLORS[lane],
                )
            axis.axvspan(
                1.5, 3.5, color="#777777", alpha=0.055, linewidth=0
            )
            axis.set_title(
                f"({chr(97 + 2 * row + column)}) {titles[regime]}\n"
                f"{column_titles[column]}",
                loc="left",
            )
            axis.set_xlim(0.0, 4.0)
            axis.set_ylim(1e-5, 20.0)
            axis.grid(alpha=0.18, linewidth=0.5)
            if row == 1:
                axis.set_xlabel(r"angular frequency $\omega/t_{\rm hop}$")
        axes[row, 0].set_ylabel("polarization power density")
    handles, labels = axes[0, 0].get_legend_handles_labels()
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
        bottom=0.105,
        top=0.86,
        wspace=0.22,
        hspace=0.42,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def analyze(
    strong_run_dir: Path,
    weak_run_dir: Path,
    output_dir: Path,
    *,
    prefix: str,
) -> dict[str, Any]:
    start = 10.0
    stop = 100.0
    sample_step = 0.2
    target_times = _time_grid(start, stop, sample_step)
    weak_values, weak_files = _weak_polarization(weak_run_dir, target_times)
    strong_values, strong_files = _strong_polarization(
        strong_run_dir, target_times
    )
    spectra_by_regime = {
        "weak": {
            lane: _spectrum(values, sample_step)
            for lane, values in weak_values.items()
        },
        "strong": {
            lane: _spectrum(values, sample_step)
            for lane, values in strong_values.items()
        },
    }
    metrics_by_regime = {
        regime: _metrics(spectra)
        for regime, spectra in spectra_by_regime.items()
    }

    output_stem = output_dir / prefix
    _plot(spectra_by_regime, metrics_by_regime, output_stem)
    summary = {
        "schema_version": 2,
        "classification": "diagnostic_postprocessing",
        "question": (
            "How do archive and representability-corrected polarization "
            "peak positions, strengths, and observed widths compare with "
            "exact cutoff-16 dynamics as g increases?"
        ),
        "method": {
            "interval": [start, stop],
            "sample_step": sample_step,
            "mean_subtracted": True,
            "window": "Hann",
            "power_displays": {
                "common_scale": (
                    "Hann-windowed power divided by the exact total power "
                    "for the same coupling; area preserves relative power"
                ),
                "shape_only": "unit sum over positive frequencies per lane",
            },
            "electronic_peak_band": [1.5, 3.5],
            "low_frequency_peak_band": [0.05, 1.0],
            "width_definition": (
                "observed half-maximum width of the finite-window spectrum; "
                "no lifetime deconvolution"
            ),
        },
        "parameters": {
            "weak": {"coupling_g": 0.3535533905932738, "lambda_ep": 0.5},
            "strong": {"coupling_g": 0.6123724356957945, "lambda_ep": 1.5},
            "hopping": 1.0,
            "omega_ph": 0.5,
            "gamma": 0.5,
            "drive_amplitude": 1.0,
            "phonon_cutoff": 16,
        },
        "metrics": metrics_by_regime,
        "source_files": {
            "weak": {
                key: {"path": str(path), "sha256": _sha256(path)}
                for key, path in weak_files.items()
            },
            "strong": {
                key: {"path": str(path), "sha256": _sha256(path)}
                for key, path in strong_files.items()
            },
        },
        "artifacts": {
            "figure_pdf": str(output_stem.with_suffix(".pdf")),
            "figure_png": str(output_stem.with_suffix(".png")),
        },
    }
    output_stem.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strong_run_dir", type=Path)
    parser.add_argument("weak_run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--prefix", default="paper_v_archive_polarization_peaks"
    )
    args = parser.parse_args()
    summary = analyze(
        args.strong_run_dir,
        args.weak_run_dir,
        args.output_dir,
        prefix=args.prefix,
    )
    print(json.dumps(summary["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
