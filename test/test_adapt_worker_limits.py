from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.static_adapt import adapt_worker_limits as limits


def _clear_worker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in limits._ALLOCATED_CPU_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(limits._AUTO_WORKER_CAP_ENV, raising=False)
    monkeypatch.delenv("_CONDOR_JOB_AD", raising=False)
    monkeypatch.delenv("_CONDOR_MACHINE_AD", raising=False)


def test_positive_int_parsing_preserves_legacy_float_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    assert limits._positive_int_from_text(None) is None
    assert limits._positive_int_from_text("") is None
    assert limits._positive_int_from_text("0") is None
    assert limits._positive_int_from_text("-2") is None
    assert limits._positive_int_from_text("not-an-int") is None
    assert limits._positive_int_from_text("3.9") == 3

    monkeypatch.setenv("UNIT_CPUS", "6.2")
    assert limits._positive_int_env("UNIT_CPUS") == 6


def test_allocated_cpu_count_prefers_env_order_and_ignores_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_worker_env(monkeypatch)
    monkeypatch.setenv("STATIC_ADAPT_ALLOCATED_CPUS", "bad")
    monkeypatch.setenv("CONDOR_REQUEST_CPUS", "7")
    monkeypatch.setenv("REQUEST_CPUS", "11")

    assert limits._allocated_cpu_count() == 7


def test_allocated_cpu_count_reads_condor_ad_after_envs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _clear_worker_env(monkeypatch)
    ad_path = tmp_path / "job.ad"
    ad_path.write_text("Other = 1\nRequestCpus = 5.8\nCpus = 9\n", encoding="utf-8")
    monkeypatch.setenv("_CONDOR_JOB_AD", str(ad_path))

    assert limits._condor_ad_cpu_count() == 5
    assert limits._allocated_cpu_count() == 5


def test_resolve_worker_limit_uses_allocated_cpu_env_and_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_worker_env(monkeypatch)
    monkeypatch.setenv("STATIC_ADAPT_ALLOCATED_CPUS", "12")
    monkeypatch.setenv("STATIC_ADAPT_AUTO_WORKER_CAP", "8")

    resolved, meta = limits._resolve_adapt_worker_limit(0, name="adapt_parallel_gradient_workers")
    assert resolved == 8
    assert meta == {
        "requested": 0,
        "resolved": 8,
        "source": "auto_allocated_cpu_count",
        "allocated_cpus": 12,
        "configured_cap": 8,
        "cap_env": "STATIC_ADAPT_AUTO_WORKER_CAP",
        "cap_env_value": 8,
    }

    explicit, explicit_meta = limits._resolve_adapt_worker_limit(20, name="adapt_parallel_gradient_workers")
    assert explicit == 12
    assert explicit_meta["source"] == "explicit"
    assert explicit_meta["configured_cap"] == 20
    assert explicit_meta["cap_env_value"] == 8

    with pytest.raises(ValueError, match="0=auto"):
        limits._resolve_adapt_worker_limit(-1, name="adapt_parallel_gradient_workers")


def test_cap_worker_limit_for_items_preserves_minimum_one() -> None:
    assert limits._cap_worker_limit_for_items(8, 3) == 3
    assert limits._cap_worker_limit_for_items(0, 3) == 1
    assert limits._cap_worker_limit_for_items(8, 0) == 1
    assert limits._cap_worker_limit_for_items(8, -2) == 1
