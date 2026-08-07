"""Compare the four distinct strong-coupling polarization routes.

The common comparison ends at t=20 because the implemented APCM entrance-layer
M4 trajectory ends there.  Spectra use only the post-pulse interval 4 <= t <=
20, so the reported FWHM values are finite-window observed widths rather than
deconvolved lifetimes.
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
    _power_ratio,
    _reference_scaled_power_density,
    _spectrum,
    _time_grid,
)


ROUTE_ORDER = (
    "exact_cutoff16",
    "archive_eom",
    "regular_eom_correction",
    "apcm_m4_prototype",
)
ROUTE_DOCUMENTATION = {
    "exact_cutoff16": {
        "display_label": "exact cutoff-16 Hamiltonian",
        "propagated_state": "full cutoff-16 wavefunction",
        "online_rule": "DOP853 Schrodinger propagation",
        "correction": "none",
    },
    "archive_eom": {
        "display_label": "archive EOM (uncorrected)",
        "propagated_state": "31 independent real archive moments",
        "online_rule": "archive moment equations of motion",
        "correction": "none",
    },
    "regular_eom_correction": {
        "display_label": "regular EOM correction (31D joint-Gram)",
        "propagated_state": "31 independent real archive moments",
        "online_rule": "archive moment equations of motion",
        "correction": (
            "minimum-Euclidean-norm 31-coordinate velocity correction for "
            "retained electronic/joint-Gram positivity, correlation trace, "
            "and zero correction-induced energy flux"
        ),
    },
    "apcm_m4_prototype": {
        "display_label": "McLachlan-type M4 correction (APCM prototype)",
        "propagated_state": (
            "60 real coordinates: raw archive chart plus preparation-dependent "
            "relative-mode moments"
        ),
        "online_rule": (
            "archive-backed entrance-layer commutator-moment equations with "
            "SSPRK(3,3)"
        ),
        "correction": (
            "positive fourth-moment M4 completion with hidden-stage retraction "
            "and the retained joint-Gram controller"
        ),
        "implementation_limit": (
            "entrance-layer prototype only; the full proposed adaptive "
            "moment-metric McLachlan projection is not implemented"
        ),
    },
}
COLORS = {
    "exact_cutoff16": "#171717",
    "archive_eom": "#c4513c",
    "regular_eom_correction": "#3569a8",
    "apcm_m4_prototype": "#2f8f5b",
}
LINESTYLES = {
    "exact_cutoff16": "-",
    "archive_eom": "-.",
    "regular_eom_correction": "--",
    "apcm_m4_prototype": ":",
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
        raise ValueError("the requested interval is not recorded")
    return np.interp(target_times, source_times, source_values)


def _load_archive_coordinates(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return (
            np.asarray(payload["times"], dtype=float),
            np.asarray(payload["coordinates"], dtype=float),
        )


def _load_route_polarizations(
    archive_run_dir: Path,
    m4_run_dir: Path,
    target_times: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Path], dict[str, float]]:
    files = {
        "exact_cutoff16": archive_run_dir / "exact_trajectory.npz",
        "archive_eom": (
            archive_run_dir / "raw_refined_rk4_dt005_trajectory.npz"
        ),
        "regular_eom_correction": (
            archive_run_dir / "corrected_trajectory.npz"
        ),
        "apcm_m4_prototype": m4_run_dir / "trajectory.npz",
    }
    exact_times, exact_coordinates = _load_archive_coordinates(
        files["exact_cutoff16"]
    )
    raw_times, raw_coordinates = _load_archive_coordinates(
        files["archive_eom"]
    )
    corrected_times, corrected_coordinates = _load_archive_coordinates(
        files["regular_eom_correction"]
    )
    with np.load(files["apcm_m4_prototype"], allow_pickle=False) as payload:
        m4_times = np.asarray(payload["times"], dtype=float)
        m4_coordinates = np.asarray(
            payload["approximate_archive_coordinates"], dtype=float
        )
        m4_exact_coordinates = np.asarray(
            payload["exact_archive_coordinates"], dtype=float
        )

    coordinate_sets = {
        "exact_cutoff16": exact_coordinates,
        "archive_eom": raw_coordinates,
        "regular_eom_correction": corrected_coordinates,
        "apcm_m4_prototype": m4_coordinates,
        "m4_exact": m4_exact_coordinates,
    }
    initial_reference = exact_coordinates[0]
    initial_maximum_difference = max(
        float(np.max(np.abs(values[0] - initial_reference)))
        for values in coordinate_sets.values()
    )
    if initial_maximum_difference > 3e-12:
        raise ValueError(
            "the four routes do not share the same initial archive moments"
        )

    exact_on_target = np.column_stack(
        [
            _interpolate(exact_times, exact_coordinates[:, index], target_times)
            for index in range(exact_coordinates.shape[1])
        ]
    )
    m4_exact_on_target = np.column_stack(
        [
            _interpolate(
                m4_times,
                m4_exact_coordinates[:, index],
                target_times,
            )
            for index in range(m4_exact_coordinates.shape[1])
        ]
    )
    exact_reference_difference = float(
        np.max(np.abs(exact_on_target - m4_exact_on_target))
    )
    if exact_reference_difference > 5e-11:
        raise ValueError("the archive and M4 exact references do not agree")

    # Coordinate zero is Delta n=rho_00-rho_11; P=-Delta n/2.
    polarizations = {
        "exact_cutoff16": -0.5
        * _interpolate(
            exact_times,
            exact_coordinates[:, 0],
            target_times,
        ),
        "archive_eom": -0.5
        * _interpolate(raw_times, raw_coordinates[:, 0], target_times),
        "regular_eom_correction": -0.5
        * _interpolate(
            corrected_times,
            corrected_coordinates[:, 0],
            target_times,
        ),
        "apcm_m4_prototype": -0.5
        * _interpolate(m4_times, m4_coordinates[:, 0], target_times),
    }
    validation = {
        "maximum_initial_coordinate_difference": initial_maximum_difference,
        "maximum_exact_reference_difference_on_analysis_grid": (
            exact_reference_difference
        ),
    }
    return polarizations, files, validation


def _route_metrics(
    spectra: dict[str, Spectrum],
) -> dict[str, dict[str, Any]]:
    exact = spectra["exact_cutoff16"]
    result: dict[str, dict[str, Any]] = {}
    for route, spectrum in spectra.items():
        result[route] = {
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
    spectra: dict[str, Spectrum],
    metrics: dict[str, dict[str, Any]],
    output_stem: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "legend.fontsize": 7.1,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharex=True)
    exact = spectra["exact_cutoff16"]
    titles = (
        "(a) common scale (exact total power = 1)",
        "(b) shape only (each spectrum has unit area)",
    )
    for column, axis in enumerate(axes):
        for route in ROUTE_ORDER:
            spectrum = spectra[route]
            frequency = spectrum.angular_frequency
            spacing = float(frequency[1] - frequency[0])
            if column == 0:
                density = _reference_scaled_power_density(spectrum, exact)
                peak_scale = metrics[route][
                    "total_hann_power_ratio_to_exact"
                ]
            else:
                density = spectrum.normalized_power / spacing
                peak_scale = 1.0
            visible = (frequency > 0.0) & (frequency <= 4.0)
            axis.semilogy(
                frequency[visible],
                density[visible],
                color=COLORS[route],
                linestyle=LINESTYLES[route],
                linewidth=1.15,
                label=ROUTE_DOCUMENTATION[route]["display_label"],
            )
            peak = metrics[route]["electronic_band_peak"]
            axis.plot(
                peak["angular_frequency"],
                peak["normalized_power_density"] * peak_scale,
                marker="o",
                markersize=3.0,
                color=COLORS[route],
            )
        axis.axvspan(1.5, 3.5, color="#777777", alpha=0.055, linewidth=0)
        axis.set_title(titles[column], loc="left")
        axis.set_xlabel(r"angular frequency $\omega/t_{\rm hop}$")
        axis.set_xlim(0.0, 4.0)
        axis.set_ylim(1e-5, 20.0)
        axis.grid(alpha=0.18, linewidth=0.5)
    axes[0].set_ylabel("polarization power density")
    figure.legend(
        *axes[0].get_legend_handles_labels(),
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.1,
    )
    figure.subplots_adjust(
        left=0.085, right=0.985, bottom=0.16, top=0.78, wspace=0.22
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def analyze(
    archive_run_dir: Path,
    m4_run_dir: Path,
    output_dir: Path,
    *,
    prefix: str,
) -> dict[str, Any]:
    start = 4.0
    stop = 20.0
    sample_step = 0.2
    target_times = _time_grid(start, stop, sample_step)
    polarizations, source_files, validation = _load_route_polarizations(
        archive_run_dir,
        m4_run_dir,
        target_times,
    )
    spectra = {
        route: _spectrum(values, sample_step)
        for route, values in polarizations.items()
    }
    metrics = _route_metrics(spectra)
    output_stem = output_dir / prefix
    _plot(spectra, metrics, output_stem)

    summary = {
        "schema_version": 2,
        "classification": "diagnostic_postprocessing",
        "comparison_id": "strong_coupling_archive_m4_four_route_spectrum",
        "models": ROUTE_DOCUMENTATION,
        "method": {
            "interval": [start, stop],
            "sample_step": sample_step,
            "mean_subtracted": True,
            "window": "Hann",
            "power_displays": {
                "common_scale": (
                    "Hann-windowed power divided by exact total power; "
                    "area preserves relative power"
                ),
                "shape_only": "unit sum over positive frequencies per route",
            },
            "electronic_peak_band": [1.5, 3.5],
            "width_definition": (
                "observed finite-window FWHM; no lifetime deconvolution"
            ),
        },
        "parameters": {
            "hopping": 1.0,
            "omega_ph": 0.5,
            "gamma": 0.5,
            "coupling_g": 0.6123724356957945,
            "lambda_ep": 1.5,
            "drive_amplitude": 1.0,
            "phonon_cutoff": 16,
        },
        "validation": validation,
        "metrics": metrics,
        "source_files": {
            route: {"path": str(path), "sha256": _sha256(path)}
            for route, path in source_files.items()
        },
        "interpretation_limits": [
            (
                "The common M4 horizon gives frequency resolution near "
                "0.388 t_hop; reported widths are resolution dominated."
            ),
            (
                "The APCM curve is the implemented entrance-layer prototype, "
                "not the full proposed adaptive moment-metric projection."
            ),
            "Exact data enter only in offline scoring and plotting.",
        ],
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
    parser.add_argument("archive_run_dir", type=Path)
    parser.add_argument("m4_run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--prefix",
        default="archive_m4_four_route_polarization_spectrum",
    )
    args = parser.parse_args()
    summary = analyze(
        args.archive_run_dir,
        args.m4_run_dir,
        args.output_dir,
        prefix=args.prefix,
    )
    print(json.dumps(summary["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
