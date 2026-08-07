#!/usr/bin/env python3
"""Tests for benchmark decision/evaluation-time noise helpers."""

from __future__ import annotations

import inspect

import pytest

from pipelines.exact_bench import benchmark_decision_noise as bdn


def test_benchmark_decision_noise_config_from_env_values_validates_and_derives_seed() -> None:
    disabled = bdn.config_from_env_values({}, family="hh", case_id="hh_L2", algorithm_id="static_hea_qiskit_vqe")
    assert disabled.enabled is False
    assert disabled.model == "off"
    assert disabled.std == 0.0
    assert disabled.seed is None
    assert disabled.semantic == "benchmark_decision_value_noise_not_physical_shots_v1"

    cfg_a = bdn.config_from_env_values(
        {
            "benchmark_decision_noise_model": "gaussian_iid_v1",
            "benchmark_decision_noise_std": "1e-3",
        },
        family="hh",
        case_id="hh_L2",
        algorithm_id="static_hea_qiskit_vqe",
    )
    cfg_b = bdn.config_from_env_values(
        {
            "GENERIC_STATIC_TABLE_BENCHMARK_DECISION_NOISE_MODEL": "gaussian_iid_v1",
            "GENERIC_STATIC_TABLE_BENCHMARK_DECISION_NOISE_STD": "1e-3",
        },
        family="hh",
        case_id="hh_L2",
        algorithm_id="static_hea_qiskit_vqe",
    )
    assert cfg_a.enabled is True
    assert cfg_a.seed_source == "derived_stable_hash_v1"
    assert cfg_a.seed == cfg_b.seed

    explicit = bdn.config_from_env_values(
        {
            "benchmark_decision_noise_model": "gaussian_iid_v1",
            "benchmark_decision_noise_std": "1e-3",
            "benchmark_decision_noise_seed": "20260515",
        },
        family="hh",
        case_id="hh_L2",
        algorithm_id="static_hea_qiskit_vqe",
    )
    assert explicit.seed == 20260515
    assert explicit.seed_source == "env"


@pytest.mark.parametrize(
    ("values", "match"),
    (
        ({"benchmark_decision_noise_seed": "1"}, "seed requires"),
        ({"benchmark_decision_noise_std": "1e-3"}, "model='off'"),
        ({"benchmark_decision_noise_model": "bad"}, "must be one of"),
        ({"benchmark_decision_noise_model": "gaussian_iid_v1"}, "std > 0"),
        ({"benchmark_decision_noise_model": "gaussian_iid_v1", "benchmark_decision_noise_std": "0"}, "std > 0"),
    ),
)
def test_benchmark_decision_noise_config_rejects_invalid_env_values(values: dict[str, str], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        bdn.config_from_env_values(values, family="hh", case_id="hh_L2", algorithm_id="static_hea_qiskit_vqe")


def test_benchmark_decision_noise_recorder_is_deterministic_and_reports_summary() -> None:
    cfg = bdn.config_from_env_values(
        {
            "benchmark_decision_noise_model": "gaussian_iid_v1",
            "benchmark_decision_noise_std": "0.5",
            "benchmark_decision_noise_seed": "17",
        },
        family="hh",
        case_id="hh_L2",
        algorithm_id="static_hea_qiskit_vqe",
    )
    scope = {"family": "hh", "case_id": "hh_L2", "algorithm_id": "static_hea_qiskit_vqe"}
    rec_a = bdn.BenchmarkDecisionNoiseRecorder(cfg, base_scope=scope)
    rec_b = bdn.BenchmarkDecisionNoiseRecorder(cfg, base_scope=scope)

    first_a = rec_a.apply(1.0, surface="objective", value_kind="energy", phase="optimizer")
    first_b = rec_b.apply(1.0, surface="objective", value_kind="energy", phase="optimizer")
    second_a = rec_a.apply(1.0, surface="objective", value_kind="energy", phase="optimizer")

    assert first_a == pytest.approx(first_b)
    assert second_a != pytest.approx(first_a)
    summary = rec_a.summary()
    assert summary["status"] == "ok"
    assert summary["supported"] is True
    assert summary["applied"] is True
    assert summary["draw_count_total"] == 2
    assert summary["draw_count_by_surface"] == {"objective": 2}
    assert summary["surfaces_affected"] == ["objective"]
    assert summary["physical_shots_unchanged"] is True
    assert summary["algorithmic_measurement_work_schema"] == "algorithmic_measurement_work_v1"
    assert summary["algorithmic_measurement_work_unchanged"] is True


def test_benchmark_decision_noise_unsupported_metadata_is_fail_closed() -> None:
    cfg = bdn.config_from_env_values(
        {
            "benchmark_decision_noise_model": "gaussian_iid_v1",
            "benchmark_decision_noise_std": "1e-3",
            "benchmark_decision_noise_seed": "5",
        },
        family="hh",
        case_id="hh_L2",
        algorithm_id="static_qiskit_adapt_vqe",
    )

    meta = bdn.unsupported_metadata(
        cfg,
        family="hh",
        case_id="hh_L2",
        algorithm_id="static_qiskit_adapt_vqe",
        dispatch="generic_static_qiskit_adapt_vqe",
        reason="hidden decision seam",
    )

    assert meta["status"] == "unsupported"
    assert meta["supported"] is False
    assert meta["applied"] is False
    assert meta["fail_closed"] is True
    assert meta["reason"] == "hidden decision seam"
    assert meta["semantic"] == "benchmark_decision_value_noise_not_physical_shots_v1"
    assert meta["draw_count_total"] == 0


def test_benchmark_decision_noise_helper_has_no_qiskit_imports() -> None:
    source = inspect.getsource(bdn).lower()
    assert "import qiskit" not in source
    assert "from qiskit" not in source
