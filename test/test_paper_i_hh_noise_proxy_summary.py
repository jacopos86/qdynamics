#!/usr/bin/env python3
"""Tests for HH scalar-noise to S-proxy diagnostic summaries."""

from __future__ import annotations

import json
import math

from pipelines.reporting.paper_i_hh_noise_proxy_summary import (
    build_summary,
    extract_value_noise_contract,
    summarize_payload,
)


def _runtime_payload(*, n_eff: float | None = None, stop_reason: str = "benchmark_abs_delta_e_target") -> dict:
    adapt = {
        "ansatz_depth": 2,
        "stop_reason": stop_reason,
        "benchmark_target_hit_success": stop_reason == "benchmark_abs_delta_e_target",
        "energy": -1.0,
        "abs_delta_e": 1e-4,
        "benchmark_target_abs_delta_e_current": 1e-4,
        "benchmark_target_abs_delta_e": 2e-4,
        "controller_measurement_work_summary": {
            "schema": "controller_measurement_work_proxy_v1",
            "source": "native_controller_live_decision_work_v1",
            "source_kind": "native_controller_work",
            "legacy_fallback_used": False,
            "by_phase": {
                "phase1": {"records_with_group_keys": 6, "shots_new": 6},
                "phase2": {"records_with_group_keys": 2, "shots_new": 2},
                "phase3": {"records_with_group_keys": 3, "shots_new": 3},
            },
        },
        "history": [{"nfev_opt": 4}, {"nfev_opt": 5}],
        "resume_boundary_refit": {"nfev": 1},
        "final_full_refit": {"nfev": 2},
        "nfev_total": 20,
    }
    if n_eff is not None:
        adapt["continuation"] = {
            "oracle_inner_exact_structure_value_noise": {
                "last_draw": {
                    "n_eff": n_eff,
                    "sigma0_abs": 1e-3,
                    "noise_delta": 0.0,
                    "post_expectation_value_noise_not_physical_shots": True,
                }
            },
            "oracle_inner_objective_mode": "exact_structure_plus_value_noise_v1",
        }
    return {"adapt_vqe": adapt}


def test_extract_value_noise_contract_labels_neff_as_scalar_model() -> None:
    payload = _runtime_payload(n_eff=1e18)

    contract = extract_value_noise_contract(payload)

    assert contract["status"] == "ok"
    assert contract["N_eff"] == 1e18
    assert contract["sigma0_abs"] == 1e-3
    assert contract["std_abs"] == 1e-12
    assert contract["post_expectation_value_noise_not_physical_shots"] is True


def test_summarize_payload_reconstructs_s_norm_and_model_units(tmp_path) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_runtime_payload(n_eff=1e18)), encoding="utf-8")

    summary = summarize_payload(path, role="noise_result", baseline_s_norm=31.0)

    assert summary["measurement_work"]["S_norm_status"] == "ok"
    assert summary["measurement_work"]["S_norm"] == 31.0
    assert summary["measurement_work"]["S_norm_components"] == {
        "N_H_outer_eval": 8.0,
        "N_grad": 6.0,
        "N_metric": 5.0,
        "N_H_refit_eval": 12.0,
    }
    derived = summary["derived_mapping"]
    assert derived["S_norm_factor_vs_baseline_S_norm"] == 1.0
    assert derived["scalar_noise_model_units_per_event"] == 1e18
    assert derived["scalar_noise_model_units_total"] == 31.0e18
    assert derived["scalar_noise_model_units_factor_vs_baseline_S_norm"] == 1e18
    assert math.isclose(derived["target_abs_delta_e_over_std_abs"], 2e8)
    assert math.isclose(derived["target_abs_delta_e_over_std_abs_squared"], 4e16)
    assert "not physical shots" in derived["interpretation"]


def test_build_summary_keeps_baseline_and_diagnostic_current_separate(tmp_path) -> None:
    baseline = tmp_path / "baseline_result.json"
    noise = tmp_path / "noise_result.json"
    current = tmp_path / "current.json"
    baseline.write_text(json.dumps(_runtime_payload()), encoding="utf-8")
    noise.write_text(json.dumps(_runtime_payload(n_eff=1e18)), encoding="utf-8")
    current.write_text(json.dumps(_runtime_payload(n_eff=3.16227766016837952e17, stop_reason="")), encoding="utf-8")

    summary = build_summary(baseline=baseline, noise_results=[noise], diagnostic_currents=[current])

    assert summary["schema"] == "paper_i_hh_noise_proxy_summary_v1"
    assert summary["baseline"]["role"] == "baseline"
    assert summary["baseline"]["value_noise"]["status"] == "off_or_missing"
    assert len(summary["noise_results"]) == 1
    assert summary["noise_results"][0]["role"] == "noise_result"
    assert len(summary["diagnostic_currents"]) == 1
    assert summary["diagnostic_currents"][0]["artifact_kind"] == "current"
    assert summary["conventions"]["N_eff"].endswith("It is not a hardware shot count.")
