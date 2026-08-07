from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_analytic_noise_calibration import fit_hybrid_proxy_calibration


def test_fit_hybrid_proxy_calibration_builds_expected_payload() -> None:
    payload = {
        "oracle_config": {
            "backend_name": "FakeMarrakesh",
            "noise_mode": "backend_scheduled",
            "shots": 4096,
            "oracle_repeats": 2,
            "mitigation": {
                "local_gate_twirling": True,
                "local_gate_twirling_scope": "2q_only",
            },
        },
        "summary": {
            "final_runtime_parameter_count": 41,
        },
        "trajectory": [
            {
                "checkpoint_index": 0,
                "time": 0.5,
                "abs_energy_total_error": 0.12,
                "abs_doublon_error": 0.03,
                "abs_pair_difference_error": 0.02,
                "groups_new": 3,
                "runtime_parameter_count": 18,
                "energy_total": -0.82,
                "energy_total_exact": -0.94,
                "doublon": 0.46,
                "doublon_exact": 0.43,
                "pair_difference": 0.18,
                "pair_difference_exact": 0.16,
            },
            {
                "checkpoint_index": 1,
                "time": 2.0,
                "abs_energy_total_error": 0.25,
                "abs_doublon_error": 0.09,
                "abs_pair_difference_error": 0.07,
                "groups_new": 5,
                "runtime_parameter_count": 28,
                "energy_total": -0.74,
                "energy_total_exact": -0.99,
                "doublon": 0.49,
                "doublon_exact": 0.40,
                "pair_difference": 0.32,
                "pair_difference_exact": 0.25,
            },
            {
                "checkpoint_index": 2,
                "time": 4.0,
                "abs_energy_total_error": 0.18,
                "abs_doublon_error": 0.06,
                "abs_pair_difference_error": 0.05,
                "groups_new": 4,
                "runtime_parameter_count": 36,
                "energy_total": -0.81,
                "energy_total_exact": -0.99,
                "doublon": 0.47,
                "doublon_exact": 0.41,
                "pair_difference": 0.29,
                "pair_difference_exact": 0.24,
            },
        ],
    }

    result = fit_hybrid_proxy_calibration(
        payload,
        sample_times=(0.5, 2.0, 4.0),
        max_samples=3,
    )

    assert str(result["model"]) == "hybrid_qpu_proxy_v1"
    assert len(result["sampled_checkpoints"]) == 3
    assert str(result["source"]["backend_name"]) == "FakeMarrakesh"
    assert int(result["source"]["shots"]) == 4096
    assert int(result["source"]["oracle_repeats"]) == 2
    coeffs = dict(result["coefficients"])
    assert str(coeffs["analytic_noise_model"]) == "hybrid_qpu_proxy_v1"
    assert int(coeffs["analytic_noise_nominal_shots"]) == 4096
    assert int(coeffs["analytic_noise_nominal_repeats"]) == 2
    assert float(coeffs["analytic_noise_std"]) > 0.0
    assert float(coeffs["analytic_noise_two_qubit_depth_scale"]) >= 0.0
    assert float(coeffs["analytic_noise_groups_new_scale"]) >= 0.0
    assert 0.0 <= float(coeffs["analytic_noise_time_corr"]) <= 0.95
    assert bool(coeffs["analytic_noise_force_psd"]) is True
    residuals = dict(result["fit_residual_summary"])
    assert float(residuals["median_scale"]) > 0.0


def test_fit_hybrid_proxy_calibration_requires_trajectory_rows() -> None:
    with pytest.raises(ValueError, match="trajectory rows"):
        fit_hybrid_proxy_calibration({"trajectory": []})
