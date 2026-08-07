"""Compare spectra of exact, raw, and physicality-corrected observables.

The primary comparison uses the common pre-divergence interval 10 <= t <= 120.
An exact-versus-corrected comparison on 10 <= t <= 1000 supplies the higher
frequency-resolution long-horizon check.  All spectra use matched sampling,
mean subtraction and one Hann window.  Absolute-response comparisons retain
the common power scale, while shape comparisons normalize each spectrum to
unit positive-frequency power.

When matched Euclidean and lifted-Frobenius correction trajectories are
provided, a separate four-lane comparison uses their complete common interval
0 <= t <= 20.  Keeping this short-horizon comparison separate avoids mixing
spectra computed from unequal time windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "exact": "#171717",
    "raw": "#c4513c",
    "corrected": "#6f3c8f",
    "euclidean": "#6f3c8f",
    "frobenius": "#16827A",
}
LINESTYLES = {
    "exact": "-",
    "raw": "--",
    "corrected": "-",
    "euclidean": "-",
    "frobenius": "-.",
}
LABELS = {
    "exact": "exact cutoff-16",
    "raw": "archive EOM",
    "corrected": "physicality-corrected",
    "euclidean": "Euclidean correction",
    "frobenius": "lifted-Frobenius correction",
}
DISTANCE_SYMBOLS = {
    "raw": "r",
    "corrected": "c",
    "euclidean": "E",
    "frobenius": "F",
}
LINEWIDTHS = {
    "exact": 1.05,
    "raw": 0.85,
    "corrected": 0.95,
    "euclidean": 0.95,
    "frobenius": 0.95,
}
ENERGY_INDEX = {
    "electronic_energy": 0,
    "phonon_energy": 1,
    "electron_phonon_energy": 2,
}
OBSERVABLE_LABELS = {
    "site_occupation": r"site occupation $\rho_{11}$",
    "electronic_energy": "electronic energy",
    "phonon_energy": "phonon energy",
    "electron_phonon_energy": "electron-phonon energy",
}


@dataclass(frozen=True)
class Spectrum:
    angular_frequency: np.ndarray
    power: np.ndarray
    normalized_power: np.ndarray
    total_power: float
    oscillation_rms: float
    dominant_angular_frequency: float


@dataclass(frozen=True)
class SpectralPeak:
    """Peak center and observed width on one finite-window spectrum."""

    angular_frequency: float
    fwhm: float
    normalized_power_density: float
    band_power: float
    frequency_resolution: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_lane(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        result = {key: payload[key].copy() for key in payload.files}
    result["metadata"] = json.loads(str(result.pop("metadata_json").item()))
    return result


def _observable(payload: dict[str, Any], name: str) -> np.ndarray:
    if name == "site_occupation":
        return np.asarray(payload["site_occupation"], dtype=float)
    return np.asarray(payload["energy_components"], dtype=float)[
        :, ENERGY_INDEX[name]
    ]


def _matched_values(
    payload: dict[str, Any],
    name: str,
    target_times: np.ndarray,
) -> np.ndarray:
    source_times = np.asarray(payload["times"], dtype=float)
    tolerance = 1e-10
    if (
        target_times[0] < source_times[0] - tolerance
        or target_times[-1] > source_times[-1] + tolerance
    ):
        raise ValueError(
            f"{name} target interval lies outside the recorded trajectory"
        )
    return np.interp(target_times, source_times, _observable(payload, name))


def _spectrum(values: np.ndarray, sample_step: float) -> Spectrum:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    oscillation_rms = float(np.sqrt(np.mean(centered**2)))
    if oscillation_rms <= 1e-14:
        raise ValueError("the selected observable has no resolvable oscillation")
    windowed = centered * np.hanning(centered.size)
    transform = np.fft.rfft(windowed)
    power = np.abs(transform) ** 2
    power[0] = 0.0
    total_power = float(np.sum(power))
    if total_power <= 0.0:
        raise ValueError("the positive-frequency power is zero")
    normalized_power = power / total_power
    angular_frequency = 2.0 * np.pi * np.fft.rfftfreq(
        centered.size,
        d=sample_step,
    )
    dominant_index = 1 + int(np.argmax(power[1:]))
    return Spectrum(
        angular_frequency=angular_frequency,
        power=power,
        normalized_power=normalized_power,
        total_power=total_power,
        oscillation_rms=oscillation_rms,
        dominant_angular_frequency=float(angular_frequency[dominant_index]),
    )


def _power_ratio(spectrum: Spectrum, reference: Spectrum) -> float:
    """Return total Hann-windowed positive-frequency power / reference."""

    np.testing.assert_allclose(
        spectrum.angular_frequency,
        reference.angular_frequency,
        atol=0.0,
        rtol=0.0,
    )
    return float(spectrum.total_power / reference.total_power)


def _band_power_ratio(
    spectrum: Spectrum,
    reference: Spectrum,
    *,
    minimum_angular_frequency: float,
    maximum_angular_frequency: float,
) -> float:
    """Return common-scale Hann power in one band relative to reference."""

    np.testing.assert_allclose(
        spectrum.angular_frequency,
        reference.angular_frequency,
        atol=0.0,
        rtol=0.0,
    )
    frequency = np.asarray(spectrum.angular_frequency, dtype=float)
    band = (frequency >= minimum_angular_frequency) & (
        frequency <= maximum_angular_frequency
    )
    reference_band_power = float(np.sum(reference.power[band]))
    if reference_band_power <= 0.0:
        raise ValueError("the reference has zero power in the selected band")
    return float(np.sum(spectrum.power[band]) / reference_band_power)


def _reference_scaled_power_density(
    spectrum: Spectrum,
    reference: Spectrum,
) -> np.ndarray:
    """Density whose area is total power relative to the exact reference."""

    frequency = np.asarray(spectrum.angular_frequency, dtype=float)
    np.testing.assert_allclose(
        frequency,
        reference.angular_frequency,
        atol=0.0,
        rtol=0.0,
    )
    if frequency.size < 2:
        raise ValueError("the frequency grid has no positive bin")
    spacing = float(frequency[1] - frequency[0])
    return spectrum.power / (reference.total_power * spacing)


def _hellinger(left: Spectrum, right: Spectrum) -> float:
    np.testing.assert_allclose(
        left.angular_frequency,
        right.angular_frequency,
        atol=0.0,
        rtol=0.0,
    )
    return float(
        np.sqrt(
            0.5
            * np.sum(
                (
                    np.sqrt(left.normalized_power)
                    - np.sqrt(right.normalized_power)
                )
                ** 2
            )
        )
    )


def _half_maximum_crossing(
    frequency: np.ndarray,
    density: np.ndarray,
    *,
    first_index: int,
    second_index: int,
    half_height: float,
) -> float:
    first_value = float(density[first_index])
    second_value = float(density[second_index])
    if second_value == first_value:
        return float(0.5 * (frequency[first_index] + frequency[second_index]))
    fraction = (half_height - first_value) / (second_value - first_value)
    return float(
        frequency[first_index]
        + fraction * (frequency[second_index] - frequency[first_index])
    )


def _band_peak(
    spectrum: Spectrum,
    *,
    minimum_angular_frequency: float,
    maximum_angular_frequency: float,
) -> SpectralPeak:
    """Return the strongest peak and its observed FWHM inside one band.

    The width is measured directly on the Hann-windowed spectrum.  It is
    therefore a finite-window observed width, not a deconvolved lifetime.
    """

    frequency = np.asarray(spectrum.angular_frequency, dtype=float)
    power = np.asarray(spectrum.normalized_power, dtype=float)
    if frequency.ndim != 1 or power.shape != frequency.shape:
        raise ValueError("spectrum arrays must be same-length vectors")
    if frequency.size < 3:
        raise ValueError("at least three frequency bins are required")
    spacing = float(frequency[1] - frequency[0])
    if spacing <= 0.0 or not np.allclose(
        np.diff(frequency), spacing, atol=1e-13, rtol=1e-12
    ):
        raise ValueError("frequency bins must be uniformly increasing")
    if minimum_angular_frequency >= maximum_angular_frequency:
        raise ValueError("the peak band must have positive width")

    density = power / spacing
    band = np.flatnonzero(
        (frequency >= minimum_angular_frequency)
        & (frequency <= maximum_angular_frequency)
    )
    if band.size == 0:
        raise ValueError("the requested peak band contains no frequency bins")
    peak_index = int(band[int(np.argmax(density[band]))])
    if peak_index == 0 or peak_index == frequency.size - 1:
        raise ValueError("the selected peak touches the spectrum boundary")

    peak_frequency = float(frequency[peak_index])
    peak_density = float(density[peak_index])
    left_value = float(density[peak_index - 1])
    right_value = float(density[peak_index + 1])
    curvature = left_value - 2.0 * peak_density + right_value
    if curvature < 0.0:
        offset = 0.5 * (left_value - right_value) / curvature
        if abs(offset) <= 1.0:
            peak_frequency += float(offset * spacing)
            peak_density -= float(
                0.25 * (left_value - right_value) * offset
            )

    half_height = 0.5 * peak_density
    left_index = peak_index
    while left_index > 0 and density[left_index] > half_height:
        left_index -= 1
    right_index = peak_index
    while right_index < density.size - 1 and density[right_index] > half_height:
        right_index += 1
    if left_index == 0 and density[left_index] > half_height:
        raise ValueError("the selected peak has no left half-maximum crossing")
    if right_index == density.size - 1 and density[right_index] > half_height:
        raise ValueError("the selected peak has no right half-maximum crossing")
    left_crossing = _half_maximum_crossing(
        frequency,
        density,
        first_index=left_index,
        second_index=left_index + 1,
        half_height=half_height,
    )
    right_crossing = _half_maximum_crossing(
        frequency,
        density,
        first_index=right_index - 1,
        second_index=right_index,
        half_height=half_height,
    )
    return SpectralPeak(
        angular_frequency=peak_frequency,
        fwhm=right_crossing - left_crossing,
        normalized_power_density=peak_density,
        band_power=float(np.sum(power[band])),
        frequency_resolution=spacing,
    )


def _time_grid(start: float, stop: float, step: float) -> np.ndarray:
    count = int(round((stop - start) / step))
    if count < 2 or abs(start + count * step - stop) > 1e-10:
        raise ValueError("the spectrum interval must contain whole sample steps")
    return start + step * np.arange(count + 1, dtype=float)


def _window_analysis(
    lanes: dict[str, dict[str, Any]],
    *,
    start: float,
    stop: float,
    sample_step: float,
    lane_names: tuple[str, ...],
) -> tuple[dict[str, dict[str, Spectrum]], dict[str, Any]]:
    times = _time_grid(start, stop, sample_step)
    spectra: dict[str, dict[str, Spectrum]] = {}
    metrics: dict[str, Any] = {}
    for observable_name in OBSERVABLE_LABELS:
        spectra[observable_name] = {
            lane: _spectrum(
                _matched_values(lanes[lane], observable_name, times),
                sample_step,
            )
            for lane in lane_names
        }
        exact = spectra[observable_name]["exact"]
        metrics[observable_name] = {
            lane: {
                "hellinger_distance": _hellinger(
                    spectra[observable_name][lane],
                    exact,
                ),
                "oscillation_rms_ratio_to_exact": (
                    spectra[observable_name][lane].oscillation_rms
                    / exact.oscillation_rms
                ),
                "dominant_angular_frequency": (
                    spectra[observable_name][lane].dominant_angular_frequency
                ),
            }
            for lane in lane_names
        }
    return spectra, metrics


def _plot_spectra(
    spectra: dict[str, dict[str, Spectrum]],
    metrics: dict[str, Any],
    output_stem: Path,
    *,
    lane_names: tuple[str, ...],
    distance_lanes: tuple[str, ...],
    maximum_angular_frequency: float,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 7.2,
            "axes.titlesize": 7.7,
            "axes.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.15, 4.45), sharex=True)
    for panel_index, observable_name in enumerate(OBSERVABLE_LABELS):
        axis = axes.flat[panel_index]
        for lane in lane_names:
            spectrum = spectra[observable_name][lane]
            frequency = spectrum.angular_frequency
            if frequency.size < 2:
                raise RuntimeError("the frequency grid has no positive bin")
            spacing = float(frequency[1] - frequency[0])
            density = spectrum.normalized_power / spacing
            visible = (
                (frequency > 0.0)
                & (frequency <= maximum_angular_frequency)
                & (density > 1e-7)
            )
            axis.semilogy(
                frequency[visible],
                density[visible],
                color=COLORS[lane],
                linestyle=LINESTYLES[lane],
                linewidth=LINEWIDTHS[lane],
                label=LABELS[lane],
            )
        distance_text = r",\ ".join(
            rf"H_{{\rm {DISTANCE_SYMBOLS[lane]}}}="
            rf"{metrics[observable_name][lane]['hellinger_distance']:.3f}"
            for lane in distance_lanes
        )
        axis.set_title(
            f"({chr(97 + panel_index)}) {OBSERVABLE_LABELS[observable_name]}",
            loc="left",
        )
        axis.text(
            0.98,
            0.94,
            f"${distance_text}$",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=6.2 if len(distance_lanes) > 2 else 6.6,
        )
        axis.set_xlim(0.0, maximum_angular_frequency)
        axis.set_ylim(1e-6, 30.0)
        axis.grid(alpha=0.18, linewidth=0.5)
        if panel_index >= 2:
            axis.set_xlabel(r"angular frequency $\omega/t_{\rm hop}$")
        if panel_index % 2 == 0:
            axis.set_ylabel("normalized power density")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=len(lane_names),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.2,
        handlelength=2.2,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.11,
        top=0.88,
        hspace=0.37,
        wspace=0.25,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def analyze(
    trajectory_dir: Path,
    output_dir: Path,
    *,
    raw_filename: str,
    prefix: str,
) -> dict[str, Any]:
    files = {
        "exact": trajectory_dir / "exact_trajectory.npz",
        "raw": trajectory_dir / raw_filename,
        "corrected": trajectory_dir / "corrected_trajectory.npz",
    }
    lanes = {lane: _load_lane(path) for lane, path in files.items()}
    primary_spectra, primary_metrics = _window_analysis(
        lanes,
        start=10.0,
        stop=120.0,
        sample_step=0.2,
        lane_names=("exact", "raw", "corrected"),
    )
    _, long_metrics = _window_analysis(
        lanes,
        start=10.0,
        stop=1000.0,
        sample_step=0.2,
        lane_names=("exact", "corrected"),
    )
    sensitivity: dict[str, Any] = {}
    for stop in (80.0, 100.0, 120.0):
        _, metrics = _window_analysis(
            lanes,
            start=10.0,
            stop=stop,
            sample_step=0.2,
            lane_names=("exact", "raw", "corrected"),
        )
        sensitivity[f"10_to_{int(stop)}"] = {
            observable: {
                lane: values[lane]["hellinger_distance"]
                for lane in ("raw", "corrected")
            }
            for observable, values in metrics.items()
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / prefix
    _plot_spectra(
        primary_spectra,
        primary_metrics,
        output_stem,
        lane_names=("exact", "raw", "corrected"),
        distance_lanes=("raw", "corrected"),
        maximum_angular_frequency=4.0,
    )
    summary = {
        "schema_version": 1,
        "classification": "diagnostic_postprocessing",
        "evidence_status": "paper_facing_analysis_of_existing_trajectory",
        "question": (
            "Does the physicality correction improve post-pulse observable "
            "frequency content relative to the raw archive closure?"
        ),
        "source_files": {
            lane: {"path": str(path), "sha256": _sha256(path)}
            for lane, path in files.items()
        },
        "method": {
            "sample_step": 0.2,
            "mean_subtracted": True,
            "window": "Hann",
            "zero_frequency_excluded": True,
            "power_normalization": "unit sum over positive frequencies",
            "spectral_distance": "Hellinger distance; zero is exact",
            "primary_interval": [10.0, 120.0],
            "long_interval": [10.0, 1000.0],
            "endpoint_sensitivity": [80.0, 100.0, 120.0],
            "plot_maximum_angular_frequency": 4.0,
        },
        "primary_metrics": primary_metrics,
        "long_metrics": long_metrics,
        "endpoint_sensitivity": sensitivity,
        "artifacts": {
            "figure_pdf": str(output_stem.with_suffix(".pdf")),
            "figure_png": str(output_stem.with_suffix(".png")),
        },
    }
    summary_path = output_stem.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def analyze_metric_ablation(
    source_dir: Path,
    euclidean_trajectory: Path,
    frobenius_trajectory: Path,
    output_dir: Path,
    *,
    raw_filename: str,
    prefix: str,
) -> dict[str, Any]:
    """Compare both correction metrics on their matched recorded horizon."""

    files = {
        "exact": source_dir / "exact_trajectory.npz",
        "raw": source_dir / raw_filename,
        "euclidean": euclidean_trajectory,
        "frobenius": frobenius_trajectory,
    }
    lanes = {lane: _load_lane(path) for lane, path in files.items()}
    lane_names = ("exact", "raw", "euclidean", "frobenius")
    spectra, metrics = _window_analysis(
        lanes,
        start=0.0,
        stop=20.0,
        sample_step=0.2,
        lane_names=lane_names,
    )
    _, late_window_metrics = _window_analysis(
        lanes,
        start=4.0,
        stop=20.0,
        sample_step=0.2,
        lane_names=lane_names,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / prefix
    _plot_spectra(
        spectra,
        metrics,
        output_stem,
        lane_names=lane_names,
        distance_lanes=("raw", "euclidean", "frobenius"),
        maximum_angular_frequency=4.0,
    )
    summary = {
        "schema_version": 1,
        "classification": "diagnostic_postprocessing",
        "evidence_status": "paper_facing_analysis_of_existing_trajectories",
        "question": (
            "How does the lifted-Frobenius correction change finite-horizon "
            "observable frequency content relative to the Euclidean "
            "correction, raw archive closure, and exact reference?"
        ),
        "source_files": {
            lane: {"path": str(path), "sha256": _sha256(path)}
            for lane, path in files.items()
        },
        "method": {
            "sample_step": 0.2,
            "mean_subtracted": True,
            "window": "Hann",
            "zero_frequency_excluded": True,
            "power_normalization": "unit sum over positive frequencies",
            "spectral_distance": "Hellinger distance; zero is exact",
            "primary_interval": [0.0, 20.0],
            "late_window_check": [4.0, 20.0],
            "plot_maximum_angular_frequency": 4.0,
        },
        "primary_metrics": metrics,
        "late_window_metrics": late_window_metrics,
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
    parser.add_argument("trajectory_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--raw-filename",
        default="raw_refined_rk4_dt005_trajectory.npz",
    )
    parser.add_argument(
        "--prefix",
        default="paper_v_archive_observable_spectra",
    )
    parser.add_argument("--metric-euclidean-trajectory", type=Path)
    parser.add_argument("--metric-frobenius-trajectory", type=Path)
    parser.add_argument(
        "--metric-raw-filename",
        default="raw_trajectory.npz",
    )
    parser.add_argument(
        "--metric-prefix",
        default="paper_v_archive_observable_spectra_metric_t20",
    )
    args = parser.parse_args()
    summary = analyze(
        args.trajectory_dir,
        args.output_dir,
        raw_filename=args.raw_filename,
        prefix=args.prefix,
    )
    print(json.dumps(summary["primary_metrics"], indent=2, sort_keys=True))
    metric_paths = (
        args.metric_euclidean_trajectory,
        args.metric_frobenius_trajectory,
    )
    if any(path is not None for path in metric_paths):
        if not all(path is not None for path in metric_paths):
            parser.error(
                "both metric trajectories are required for the metric ablation"
            )
        metric_summary = analyze_metric_ablation(
            args.trajectory_dir,
            args.metric_euclidean_trajectory,
            args.metric_frobenius_trajectory,
            args.output_dir,
            raw_filename=args.metric_raw_filename,
            prefix=args.metric_prefix,
        )
        print(
            json.dumps(
                metric_summary["primary_metrics"],
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
