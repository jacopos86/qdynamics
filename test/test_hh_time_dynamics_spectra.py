from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pipelines.hardcoded.hh_time_dynamics_spectra import (
    analyze_payload,
    build_pair_difference_signal,
    build_site_fluctuation_signals,
    compute_one_sided_amplitude_spectrum,
    load_trajectory_payload,
    main,
)


def _write_synthetic_controller_json(path: Path, *, omega: float = 1.5) -> None:
    times = np.linspace(0.0, 20.0, 401)
    n0 = 1.0 + 0.2 * np.sin(omega * times)
    n1 = 1.0 - 0.2 * np.sin(omega * times)
    staggered = 0.5 * (n0 - n1)
    energy = 0.2 + 0.05 * np.cos(omega * times)
    payload = {
        "run_tag": "synthetic_spectrum_case",
        "drive": {
            "enabled": True,
            "drive_A": 1.5,
            "drive_omega": omega,
        },
        "summary": {},
        "trajectory": [
            {
                "time": float(t),
                "physical_time": float(t),
                "site_occupations": [float(a), float(b)],
                "site_occupations_exact": [float(a), float(b)],
                "energy_total": float(e),
                "energy_total_exact": float(e),
                "staggered": float(m),
                "staggered_exact": float(m),
                "doublon": 0.1,
                "doublon_exact": 0.1,
            }
            for t, a, b, m, e in zip(times, n0, n1, staggered, energy)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_site_fluctuations_and_pair_difference_agree_for_two_sites() -> None:
    site_occ = np.asarray(
        [
            [1.2, 0.8],
            [0.9, 1.1],
        ],
        dtype=float,
    )
    delta = build_site_fluctuation_signals(site_occ)
    pair = build_pair_difference_signal(site_occ, pair=(0, 1))
    assert np.allclose(delta[:, 0], 0.5 * pair)
    assert np.allclose(delta[:, 1], -0.5 * pair)


def test_compute_one_sided_amplitude_spectrum_finds_drive_peak() -> None:
    omega = 1.75
    times = np.linspace(0.0, 40.0, 1601)
    signal = np.sin(omega * times)
    spectrum = compute_one_sided_amplitude_spectrum(
        times,
        signal,
        detrend="constant",
        window="hann",
        max_peaks=3,
        drive_omega=omega,
        max_harmonic=2,
    )
    strongest = spectrum.top_peaks[0]
    assert abs(float(strongest["omega"]) - float(omega)) < 0.2
    assert spectrum.harmonic_fit
    assert abs(float(spectrum.harmonic_fit[0]["amplitude"]) - 1.0) < 0.1


def test_compute_one_sided_amplitude_spectrum_handles_short_traces() -> None:
    times = np.asarray([0.0, 1.0], dtype=float)
    signal = np.asarray([1.0, -1.0], dtype=float)
    spectrum = compute_one_sided_amplitude_spectrum(
        times,
        signal,
        detrend="constant",
        window="hann",
        max_peaks=2,
        drive_omega=None,
        max_harmonic=1,
    )
    assert np.all(np.isfinite(spectrum.amplitude))


def test_compute_one_sided_amplitude_spectrum_doubles_odd_length_positive_bins() -> None:
    times = np.asarray([0.0, 1.0, 2.0], dtype=float)
    signal = np.asarray([0.0, 1.0, 0.0], dtype=float)
    spectrum = compute_one_sided_amplitude_spectrum(
        times,
        signal,
        detrend="constant",
        window="none",
        max_peaks=2,
        drive_omega=None,
        max_harmonic=1,
    )
    centered = signal - np.mean(signal)
    fft_vals = np.fft.rfft(centered)
    expected = np.abs(fft_vals) / 3.0
    expected[1:] *= 2.0
    assert np.allclose(spectrum.amplitude, expected)


def test_main_writes_json_and_png_for_controller_payload(tmp_path: Path) -> None:
    input_json = tmp_path / "synthetic_controller.json"
    _write_synthetic_controller_json(input_json, omega=1.5)

    output_json = tmp_path / "synthetic_controller_spectra.json"
    output_png = tmp_path / "synthetic_controller_spectra.png"

    rc = main(
        [
            "--input-json",
            str(input_json),
            "--output-json",
            str(output_json),
            "--output-png",
            str(output_png),
            "--pair",
            "0,1",
        ]
    )
    assert rc == 0
    assert output_json.exists()
    assert output_png.exists()

    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["metadata"]["num_sites"] == 2
    assert "staggered" in data["spectra"]
    assert "pair_difference_0_1" in data["spectra"]
    assert "site_occupation_0" in data["spectra"]
    assert "energy_total" in data["spectra"]
    assert "energy_total_error" in data["spectra"]
    strongest = data["spectra"]["staggered"]["top_peaks"][0]
    assert abs(float(strongest["omega"]) - 1.5) < 0.2


def test_load_and_analyze_payload_from_synthetic_json(tmp_path: Path) -> None:
    input_json = tmp_path / "synthetic_controller.json"
    _write_synthetic_controller_json(input_json, omega=2.0)
    payload = load_trajectory_payload(input_json, time_key="time")
    analysis = analyze_payload(
        payload,
        pair=(0, 1),
        detrend="constant",
        window="hann",
        max_peaks=4,
        max_harmonic=3,
    )
    assert analysis["metadata"]["drive_omega"] == 2.0
    assert analysis["metadata"]["pair_difference"] == [0, 1]
    assert len(analysis["spectra"]["site_fluctuation_0"]["omega"]) > 10
    assert "energy_total" in analysis["spectra"]
    assert "staggered_error" in analysis["spectra"]


def test_load_nested_staged_payload_uses_reference_drive_profile(tmp_path: Path) -> None:
    times = np.linspace(0.0, 4.0, 81)
    omega = 1.25
    rows = []
    for t in times:
        n0 = 1.0 + 0.1 * np.sin(omega * t)
        n1 = 1.0 - 0.1 * np.sin(omega * t)
        rows.append(
            {
                "time": float(t),
                "site_occupations": [float(n0), float(n1)],
                "site_occupations_exact": [float(n0), float(n1)],
                "staggered": float(0.5 * (n0 - n1)),
                "staggered_exact": float(0.5 * (n0 - n1)),
            }
        )
    payload = {
        "adaptive_realtime_checkpoint": {
            "reference": {
                "drive_profile": {
                    "omega": omega,
                    "A": 1.5,
                }
            },
            "trajectory": rows,
        }
    }
    path = tmp_path / "nested.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_trajectory_payload(path, time_key="auto")
    assert loaded.source_schema == "staged_adaptive_realtime_checkpoint"
    assert loaded.time_key == "time"
    assert loaded.drive_omega == 1.25
    assert loaded.drive_amplitude == 1.5
