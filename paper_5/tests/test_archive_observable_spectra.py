from __future__ import annotations

import numpy as np

from pipelines.open_dynamics.analyze_archive_m4_polarization_spectra import (
    ROUTE_DOCUMENTATION,
    ROUTE_ORDER,
    _load_route_polarizations,
)
from pipelines.open_dynamics.analyze_archive_observable_spectra import (
    Spectrum,
    _band_power_ratio,
    _band_peak,
    _hellinger,
    _power_ratio,
    _reference_scaled_power_density,
    _spectrum,
    _time_grid,
    _window_analysis,
)


def test_spectrum_recovers_an_exact_fft_bin_and_normalizes_power() -> None:
    sample_step = 0.2
    times = sample_step * np.arange(501, dtype=float)
    target_bin = 13
    angular_frequency = 2.0 * np.pi * target_bin / (
        times.size * sample_step
    )
    values = 1.7 + 0.4 * np.cos(angular_frequency * times)

    spectrum = _spectrum(values, sample_step)

    assert abs(spectrum.dominant_angular_frequency - angular_frequency) < 1e-14
    assert abs(float(np.sum(spectrum.normalized_power)) - 1.0) < 5e-16
    assert abs(float(np.sum(spectrum.power)) - spectrum.total_power) < 5e-16
    assert spectrum.normalized_power[0] == 0.0
    assert spectrum.power[0] == 0.0


def test_common_scale_power_preserves_amplitude_information() -> None:
    sample_step = 0.2
    times = sample_step * np.arange(501, dtype=float)
    exact = _spectrum(np.cos(0.7 * times), sample_step)
    doubled = _spectrum(2.0 * np.cos(0.7 * times), sample_step)

    assert abs(_power_ratio(doubled, exact) - 4.0) < 1e-14
    assert abs(
        _band_power_ratio(
            doubled,
            exact,
            minimum_angular_frequency=0.5,
            maximum_angular_frequency=0.9,
        )
        - 4.0
    ) < 1e-14
    spacing = exact.angular_frequency[1] - exact.angular_frequency[0]
    exact_density = _reference_scaled_power_density(exact, exact)
    doubled_density = _reference_scaled_power_density(doubled, exact)
    assert abs(float(np.sum(exact_density) * spacing) - 1.0) < 1e-14
    assert abs(float(np.sum(doubled_density) * spacing) - 4.0) < 1e-14


def test_hellinger_distance_is_zero_only_for_the_same_spectrum() -> None:
    sample_step = 0.2
    times = sample_step * np.arange(501, dtype=float)
    first = _spectrum(np.cos(0.5 * times), sample_step)
    second = _spectrum(np.cos(1.5 * times), sample_step)

    assert _hellinger(first, first) == 0.0
    assert 0.0 < _hellinger(first, second) <= 1.0


def test_time_grid_includes_both_declared_endpoints() -> None:
    times = _time_grid(10.0, 120.0, 0.2)

    assert times.size == 551
    assert times[0] == 10.0
    assert times[-1] == 120.0
    np.testing.assert_allclose(np.diff(times), 0.2, atol=2e-14, rtol=0.0)


def test_band_peak_recovers_gaussian_center_and_fwhm() -> None:
    frequency = np.linspace(0.0, 4.0, 4001)
    center = 2.15
    sigma = 0.08
    power = np.exp(-0.5 * ((frequency - center) / sigma) ** 2)
    power /= np.sum(power)
    spectrum = Spectrum(
        angular_frequency=frequency,
        power=power,
        normalized_power=power,
        total_power=1.0,
        oscillation_rms=1.0,
        dominant_angular_frequency=center,
    )

    peak = _band_peak(
        spectrum,
        minimum_angular_frequency=1.5,
        maximum_angular_frequency=3.0,
    )

    expected_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    assert abs(peak.angular_frequency - center) < 1e-12
    assert abs(peak.fwhm - expected_fwhm) < 2e-6
    assert peak.frequency_resolution == 0.001
    assert 0.999 < peak.band_power <= 1.0


def test_window_analysis_accepts_both_correction_metrics() -> None:
    sample_step = 0.2
    times = sample_step * np.arange(101, dtype=float)

    def payload(frequency: float) -> dict[str, np.ndarray]:
        signal = np.cos(frequency * times)
        return {
            "times": times,
            "site_occupation": signal,
            "energy_components": np.column_stack(
                (signal, 0.7 * signal, -0.4 * signal)
            ),
        }

    lanes = {
        "exact": payload(0.5),
        "raw": payload(0.7),
        "euclidean": payload(0.55),
        "frobenius": payload(0.52),
    }

    _, metrics = _window_analysis(
        lanes,
        start=0.0,
        stop=20.0,
        sample_step=sample_step,
        lane_names=("exact", "raw", "euclidean", "frobenius"),
    )

    assert set(metrics["site_occupation"]) == set(lanes)
    assert metrics["site_occupation"]["exact"]["hellinger_distance"] == 0.0
    assert (
        metrics["site_occupation"]["frobenius"]["hellinger_distance"]
        < metrics["site_occupation"]["euclidean"]["hellinger_distance"]
    )


def test_four_route_loader_keeps_corrections_distinct(tmp_path) -> None:
    archive_directory = tmp_path / "archive"
    m4_directory = tmp_path / "m4"
    archive_directory.mkdir()
    m4_directory.mkdir()
    times = np.array([0.0, 1.0, 2.0])
    exact = np.zeros((3, 31))
    exact[:, 0] = [0.0, 0.2, 0.4]
    raw = exact.copy()
    raw[1:, 0] += 0.1
    corrected = exact.copy()
    corrected[1:, 0] += 0.2
    m4 = exact.copy()
    m4[1:, 0] += 0.3
    np.savez(
        archive_directory / "exact_trajectory.npz",
        times=times,
        coordinates=exact,
    )
    np.savez(
        archive_directory / "raw_refined_rk4_dt005_trajectory.npz",
        times=times,
        coordinates=raw,
    )
    np.savez(
        archive_directory / "corrected_trajectory.npz",
        times=times,
        coordinates=corrected,
    )
    np.savez(
        m4_directory / "trajectory.npz",
        times=times,
        approximate_archive_coordinates=m4,
        exact_archive_coordinates=exact,
    )

    values, _, validation = _load_route_polarizations(
        archive_directory,
        m4_directory,
        np.array([1.0, 2.0]),
    )

    assert tuple(values) == ROUTE_ORDER
    np.testing.assert_allclose(values["exact_cutoff16"], [-0.1, -0.2])
    np.testing.assert_allclose(values["archive_eom"], [-0.15, -0.25])
    np.testing.assert_allclose(
        values["regular_eom_correction"], [-0.2, -0.3]
    )
    np.testing.assert_allclose(values["apcm_m4_prototype"], [-0.25, -0.35])
    assert validation["maximum_initial_coordinate_difference"] == 0.0
    assert set(ROUTE_DOCUMENTATION) == set(ROUTE_ORDER)
    assert "not implemented" in ROUTE_DOCUMENTATION[
        "apcm_m4_prototype"
    ]["implementation_limit"]
