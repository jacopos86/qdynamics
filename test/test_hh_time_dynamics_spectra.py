from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from pipelines.hardcoded.hh_time_dynamics_spectra import (
    _resolve_panel_plot_max_omega,
    _set_wrapped_suptitle,
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


def test_resolve_panel_plot_max_omega_uses_specific_overrides() -> None:
    resolved = _resolve_panel_plot_max_omega(common=30.0, primary=10.0, error=20.0)
    assert resolved["energy"] == 30.0
    assert resolved["primary"] == 10.0
    assert resolved["error"] == 20.0


def test_spectrum_page_title_wraps_long_run_names() -> None:
    fig = plt.figure(figsize=(14.0, 13.0))
    try:
        long_title = (
            "HH time-dynamics spectra | "
            "20260424_hh_strict_qpu_driven_t8_ideal_obs_v1_with_a_deliberately_long_filename_"
            "that_would_previously_clip_page_two_headers_result.json | hann window"
        )
        top = _set_wrapped_suptitle(fig, long_title, width=72)
        assert fig._suptitle is not None
        assert "\n" in fig._suptitle.get_text()
        assert top < 0.955
    finally:
        plt.close(fig)


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
            "--plot-max-omega-primary",
            "10",
            "--plot-max-omega-error",
            "20",
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


def test_spectra_loader_treats_null_exact_series_as_unavailable(tmp_path: Path) -> None:
    input_json = tmp_path / "strict_measured_controller.json"
    _write_synthetic_controller_json(input_json, omega=2.0)
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    for row in payload["trajectory"]:
        row["energy_total_exact"] = None
        row["site_occupations_exact"] = None
        row["staggered_exact"] = None
        row["doublon_exact"] = None
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_trajectory_payload(input_json, time_key="time")
    analysis = analyze_payload(
        loaded,
        pair=(0, 1),
        detrend="constant",
        window="hann",
        max_peaks=4,
        max_harmonic=3,
    )

    assert loaded.energy_total is not None
    assert loaded.energy_total_exact is None
    assert loaded.site_occupations_exact is None
    assert "energy_total" in analysis["spectra"]
    assert "energy_total_error" not in analysis["spectra"]
    assert "staggered_error" not in analysis["spectra"]


def test_spectra_filters_duplicate_time_repair_event_rows(tmp_path: Path) -> None:
    input_json = tmp_path / "repair_rows_controller.json"
    _write_synthetic_controller_json(input_json, omega=1.5)
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    repair_row = dict(payload["trajectory"][0])
    repair_row.update(
        {
            "action_kind": "repair_miss",
            "trajectory_sample_kind": "repair_event",
            "advances_time": False,
            "repair_retry_next": True,
        }
    )
    payload["trajectory"].insert(1, repair_row)
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_trajectory_payload(input_json, time_key="time")
    analysis = analyze_payload(
        loaded,
        pair=(0, 1),
        detrend="constant",
        window="hann",
        max_peaks=4,
        max_harmonic=3,
    )

    assert loaded.raw_trajectory_row_count == 402
    assert loaded.repair_event_row_count == 1
    assert loaded.trajectory_state_sample_count == 401
    assert loaded.times.size == 401
    assert np.all(np.diff(loaded.times) > 0.0)
    assert analysis["metadata"]["raw_trajectory_row_count"] == 402
    assert analysis["metadata"]["repair_event_row_count"] == 1
    assert analysis["metadata"]["trajectory_state_sample_count"] == 401


def test_spectra_repair_only_payload_falls_back_to_raw_rows(tmp_path: Path) -> None:
    input_json = tmp_path / "repair_only_controller.json"
    payload = {
        "run_tag": "repair_only",
        "summary": {"status": "stopped_early"},
        "trajectory": [
            {
                "time": 0.0,
                "physical_time": 0.0,
                "action_kind": "repair_miss",
                "trajectory_sample_kind": "repair_event",
                "advances_time": False,
                "site_occupations": [1.0, 1.0],
                "site_occupations_exact": [1.0, 1.0],
                "energy_total": 0.2,
                "energy_total_exact": 0.2,
                "staggered": 0.0,
                "staggered_exact": 0.0,
                "doublon": 0.1,
                "doublon_exact": 0.1,
            }
        ],
    }
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_trajectory_payload(input_json, time_key="time")
    analysis = analyze_payload(
        loaded,
        pair=(0, 1),
        detrend="constant",
        window="hann",
        max_peaks=4,
        max_harmonic=3,
        allow_short=True,
    )

    assert loaded.raw_trajectory_row_count == 1
    assert loaded.repair_event_row_count == 1
    assert loaded.trajectory_state_sample_count == 0
    assert loaded.times.tolist() == [0.0]
    assert analysis["metadata"]["analysis_status"] == "spectra_unavailable"


def test_spectra_manifest_discovers_realtime_compile_audit_fields(tmp_path: Path) -> None:
    input_json = tmp_path / "audited_controller.json"
    artifact_json = tmp_path / "adapt_artifact.json"
    artifact_json.write_text(
        json.dumps(
            {
                "settings": {
                    "L": 2,
                    "t": 1.0,
                    "u": 4.0,
                    "dv": 0.0,
                    "problem": "hh",
                    "omega0": 1.0,
                    "g_ep": 0.5,
                    "n_ph_max": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    _write_synthetic_controller_json(input_json, omega=1.5)
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    payload["artifact_json"] = str(artifact_json)
    payload["summary"] = {
        "oracle_compile_observation": {
            "compiled_count_2q": 42,
            "compiled_depth": 99,
            "compiled_size": 123,
            "compiled_num_qubits": 6,
        },
        "oracle_backend_snapshot": {"backend_name": "FakeMarrakesh"},
        "oracle_compile_request": {
            "transpile_seed": 7,
            "transpile_optimization_level": 2,
        },
    }
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_trajectory_payload(input_json, time_key="time")
    analysis = analyze_payload(
        loaded,
        pair=(0, 1),
        detrend="constant",
        window="hann",
        max_peaks=4,
        max_harmonic=3,
    )
    manifest = analysis["metadata"]["parameter_manifest"]

    assert manifest["model_family_name"] == "Hubbard-Holstein"
    assert manifest["L"] == 2
    assert manifest["t"] == 1.0
    assert manifest["U"] == 4.0
    assert manifest["dv"] == 0.0
    assert manifest["omega0"] == 1.0
    assert manifest["g_ep"] == 0.5
    assert manifest["n_ph_max"] == 1
    assert manifest["time_final"] == 20.0
    assert manifest["time_samples"] == 401
    assert manifest["compiled_count_2q"] == 42
    assert manifest["compiled_depth"] == 99
    assert manifest["compiled_size"] == 123
    assert manifest["compile_backend"] == "FakeMarrakesh"
    assert manifest["transpile_seed"] == 7
    assert manifest["transpile_optimization_level"] == 2
    assert manifest["compile_note"] == "recorded in payload"


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
