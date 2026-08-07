#!/usr/bin/env python3
"""Cheap regressions for Paper-I HH SNAKE/source-lock behavior."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RESOLVER_PATH = REPO_ROOT / "agent_guidance" / "skills" / "shared" / "scripts" / "resolve_visible_settings.py"
NATIVE200_CONTRACT = (
    REPO_ROOT
    / "MATH"
    / "paper_facing"
    / "paper_I_static_scaffold"
    / "paper_i_hh_native200_rerun_contract_20260621.json"
)
TABLEIII_SOURCE_MAP = (
    REPO_ROOT
    / "MATH"
    / "paper_facing"
    / "paper_I_static_scaffold"
    / "hh_tableiii_convergence_sources.json"
)
HH_MINUS_HVA_FILTER = "agent_guidance/static-adapt/hh_full_meta_minus_hva_class_filter.json"
ACTIVE_REGIMES = {
    "weak_weak",
    "intermediate_weak",
    "strong_weak",
    "weak_strong",
    "intermediate_strong",
    "strong_strong",
}
ACTIVE_METHOD_KEYS = {"snake", "geo", "append"}
ACTIVE_METHOD_LABELS = {"SNAKE", "Geo-ADAPT", "Append-ADAPT"}


_resolver_spec = importlib.util.spec_from_file_location("resolve_visible_settings_test", RESOLVER_PATH)
assert _resolver_spec is not None and _resolver_spec.loader is not None
resolver = importlib.util.module_from_spec(_resolver_spec)
_resolver_spec.loader.exec_module(resolver)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_source_map(
    path: Path,
    *,
    source_json: Path,
    source_sha256: str | None,
    entry_extra: dict[str, Any] | None = None,
    target_extra: dict[str, Any] | None = None,
) -> Path:
    entry: dict[str, Any] = {
        "visible_value": "1.0e-3",
        "source_json": str(source_json),
    }
    if source_sha256 is not None:
        entry["source_sha256"] = source_sha256
    entry.update(entry_extra or {})
    target: dict[str, Any] = {"methods": {"Snake": entry}}
    target.update(target_extra or {})
    return _write_json(
        path,
        {
            "schema": "unit_visible_source_map_v1",
            "figure_label": "fig:unit",
            "regimes": {"weak-weak": target},
        },
    )


def _run_resolver_cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> tuple[int, dict[str, Any], str]:
    monkeypatch.setattr(sys, "argv", [str(RESOLVER_PATH), *args])
    exit_code = resolver.main()
    captured = capsys.readouterr()
    return exit_code, json.loads(captured.out), captured.err


def test_resolver_resolves_tiny_source_map_with_aliases_and_reusable_settings(tmp_path: Path) -> None:
    source_json = _write_json(
        tmp_path / "source.json",
        {
            "settings": {"adapt_pool": "full_meta", "adapt_max_depth": 30},
            "route_contract": {"static_route_id": "route_a", "meta_feature_profile": "paper_i_production_v1"},
            "algorithm_id": "static_family_native_adapt_phase3",
        },
    )
    source_map = _tiny_source_map(
        tmp_path / "source_map.json",
        source_json=source_json,
        source_sha256=_sha256(source_json),
        target_extra={"cutoff_contract": {"n_ph_work": 2, "n_ph_ref": 5}},
        entry_extra={"pool_contract": {"hh_adaptive_pool_profile": "full_meta_minus_hva"}},
    )

    trace, problems = resolver.build_trace(
        SimpleNamespace(
            source_map=str(source_map),
            target_axis=None,
            target_key=None,
            regime="WEAK_WEAK",
            case=None,
            method="snake",
        )
    )

    assert problems == []
    assert trace["status"] if "status" in trace else True
    assert trace["regime_or_case"] == "weak-weak"
    assert trace["method"] == "Snake"
    assert trace["visible_value"] == "1.0e-3"
    assert trace["source_sha256_match"] is True
    assert trace["settings_reused"]["settings"]["adapt_max_depth"] == 30
    assert trace["settings_reused"]["route_contract"]["static_route_id"] == "route_a"
    assert trace["settings_reused"]["pool_contract"]["hh_adaptive_pool_profile"] == "full_meta_minus_hva"
    assert trace["settings_reused"]["cutoff_contract"] == {"n_ph_work": 2, "n_ph_ref": 5}
    assert trace["run_note_template"]["settings_changed"] == "list only the user-requested changes"


def test_resolver_fails_closed_for_missing_source_json_unless_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_map = _tiny_source_map(
        tmp_path / "source_map.json",
        source_json=tmp_path / "missing_source.json",
        source_sha256="0" * 64,
        entry_extra={"pool_contract": {"hh_adaptive_pool_profile": "full_meta_minus_hva"}},
    )

    exit_code, payload, stderr = _run_resolver_cli(
        monkeypatch,
        capsys,
        "--source-map",
        str(source_map),
        "--regime",
        "weak_weak",
        "--method",
        "SNAKE",
    )
    assert exit_code == 2
    assert payload["status"] == "blocked"
    assert "source JSON is missing locally" in payload["problems"]
    assert "FAIL CLOSED" in stderr

    exit_code, payload, stderr = _run_resolver_cli(
        monkeypatch,
        capsys,
        "--source-map",
        str(source_map),
        "--regime",
        "weak_weak",
        "--method",
        "SNAKE",
        "--allow-missing-source-json",
    )
    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["source_json_exists_locally"] is False
    assert payload["settings_reused"]["pool_contract"]["hh_adaptive_pool_profile"] == "full_meta_minus_hva"
    assert stderr == ""


def test_resolver_fails_closed_for_sha_mismatch_unless_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_json = _write_json(tmp_path / "source.json", {"settings": {"adapt_pool": "full_meta"}})
    source_map = _tiny_source_map(
        tmp_path / "source_map.json",
        source_json=source_json,
        source_sha256="f" * 64,
    )

    exit_code, payload, stderr = _run_resolver_cli(
        monkeypatch,
        capsys,
        "--source-map",
        str(source_map),
        "--regime",
        "weak_weak",
        "--method",
        "SNAKE",
    )
    assert exit_code == 2
    assert payload["status"] == "blocked"
    assert payload["source_sha256_match"] is False
    assert "source JSON SHA-256 does not match source map" in payload["problems"]
    assert "FAIL CLOSED" in stderr

    exit_code, payload, stderr = _run_resolver_cli(
        monkeypatch,
        capsys,
        "--source-map",
        str(source_map),
        "--regime",
        "weak_weak",
        "--method",
        "SNAKE",
        "--allow-sha-mismatch",
    )
    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["source_sha256_match"] is False
    assert payload["settings_reused"]["settings"] == {"adapt_pool": "full_meta"}
    assert stderr == ""


def test_resolver_fails_closed_for_empty_reusable_settings_unless_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_json = _write_json(tmp_path / "source.json", {"only_metrics": {"delta_e": 0.1}})
    source_map = _tiny_source_map(
        tmp_path / "source_map.json",
        source_json=source_json,
        source_sha256=_sha256(source_json),
    )

    exit_code, payload, stderr = _run_resolver_cli(
        monkeypatch,
        capsys,
        "--source-map",
        str(source_map),
        "--regime",
        "weak_weak",
        "--method",
        "SNAKE",
    )
    assert exit_code == 2
    assert payload["status"] == "blocked"
    assert "no reusable settings/contracts found in source JSON" in payload["problems"]
    assert "FAIL CLOSED" in stderr

    exit_code, payload, stderr = _run_resolver_cli(
        monkeypatch,
        capsys,
        "--source-map",
        str(source_map),
        "--regime",
        "weak_weak",
        "--method",
        "SNAKE",
        "--allow-empty-settings",
    )
    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["settings_reused"] == {}
    assert stderr == ""


def test_native200_rerun_contract_freezes_active_three_by_six_shape() -> None:
    payload = json.loads(NATIVE200_CONTRACT.read_text(encoding="utf-8"))
    rows = payload["rows"]

    assert payload["schema"] == "paper_i_hh_native200_rerun_contract_v1"
    assert len(rows) == 18
    assert {row["regime_key"] for row in rows} == ACTIVE_REGIMES
    assert {row["method_key"] for row in rows} == ACTIVE_METHOD_KEYS
    assert {row["method"] for row in rows} == ACTIVE_METHOD_LABELS

    matrix = {(row["regime_key"], row["method_key"]) for row in rows}
    assert matrix == {(regime, method) for regime in ACTIVE_REGIMES for method in ACTIVE_METHOD_KEYS}
    assert all(row["source_sha256_verified"] is True for row in rows)


def test_native200_append_geo_rows_are_locked_to_full_meta_minus_hva() -> None:
    rows = json.loads(NATIVE200_CONTRACT.read_text(encoding="utf-8"))["rows"]
    adaptive_comparators = [row for row in rows if row["method_key"] in {"append", "geo"}]

    assert len(adaptive_comparators) == 12
    for row in adaptive_comparators:
        assert row["rerun_contract_status"] == "sufficient_source_locked_wrapper_payload"
        assert row["algorithm_id"] in {"static_full_meta_append_adapt_vqe", "static_geo_adapt_vqe"}
        assert row["guardrails"]["pool_name"] == "full_meta"
        assert row["guardrails"]["uses_exact_for_decision"] is False
        assert row["guardrails"]["uses_reference_for_decision"] is False
        pool = row["pool_contract"]
        assert pool["base_pool_name"] == "full_meta"
        assert pool["hh_adaptive_pool_profile"] == "full_meta_minus_hva"
        assert pool["hh_full_meta_class_filter_json"] == HH_MINUS_HVA_FILTER
        assert pool["hh_full_meta_class_filter_classifier_version"] == "hh_full_meta_v4"
        assert pool["hh_full_meta_class_filter_dropped_classes"] == ["hva_layer"]
        assert "hva_layer" not in set(pool["hh_full_meta_class_filter_keep_classes"])
        assert pool["hh_pool_cache_mode"] == "disk"
        assert pool["hh_pool_cache_scope"] == "paper_i_holstein_sector"


def test_native200_snake_rows_preserve_literal_source_lock_minus_hva_conflict() -> None:
    payload = json.loads(NATIVE200_CONTRACT.read_text(encoding="utf-8"))
    rows = [row for row in payload["rows"] if row["method_key"] == "snake"]

    assert len(rows) == 6
    assert "literal source-locked rerun" in payload["answer_summary"]["snake_exact_rerun_status"]
    assert "contain hva_layer records" in payload["answer_summary"]["snake_exact_rerun_status"]
    assert payload["answer_summary"]["snake_minus_hva_contract_status"].startswith(
        "not proven by current SNAKE source settings"
    )

    for row in rows:
        assert row["algorithm_id"] == "static_family_native_adapt_phase3"
        assert row["settings_payload"] == "top-level settings object in source_json"
        assert (
            row["rerun_contract_status"]
            == "source_locked_literal_settings_sufficient_but_pool_contract_conflicts_with_minus_hva_table_contract"
        )
        assert row["snake_route_settings"]["adapt_pool"] == "full_meta"
        assert row["snake_route_settings"]["adapt_pool_requested"] == "full_meta"
        assert row["snake_route_settings"]["phase2_novelty_mode"] == "collective_span_v1"
        assert row["snake_route_settings"]["phase3_selector_geometry_mode"] == "reduced"
        assert row["snake_route_settings"]["phase3_runtime_split_mode"] == "shortlist_pauli_children_v1"
        pool = row["pool_contract"]
        assert pool["literal_source_adapt_pool"] == "full_meta"
        assert pool["literal_source_adapt_pool_class_filter_json"] is None
        assert pool["literal_source_adapt_pool_class_filter_keep_classes"] is None
        assert pool["literal_source_contains_hva_layer_records"] is True
        assert pool["table_contract_expected_filter_json"] == HH_MINUS_HVA_FILTER
        assert pool["table_contract_expected_dropped_classes"] == ["hva_layer"]
        assert "exact historical rerun" in pool["rerun_instruction"]
        assert "corrected minus-HVA Table-III rerun" in pool["rerun_instruction"]
        assert row["trajectory_contract"]["trajectory_present"] is True
        assert row["trajectory_contract"]["terminal_strict_replay_fields_present"] is True


def test_tableiii_source_map_resolver_smoke_resolves_snake_without_generated_artifact_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(REPO_ROOT)

    trace, problems = resolver.build_trace(
        SimpleNamespace(
            source_map=str(TABLEIII_SOURCE_MAP),
            target_axis=None,
            target_key=None,
            regime="strong-strong",
            case=None,
            method="snake",
        )
    )

    assert trace["source_map"] == str(TABLEIII_SOURCE_MAP.relative_to(REPO_ROOT))
    assert trace["target_axis"] == "regimes"
    assert trace["regime_or_case"] == "strong_strong"
    assert trace["method"] == "SNAKE"
    assert trace["source_entry"]["source_json"]
    assert trace["source_entry"]["source_sha256"]
    assert trace["source_entry"]["source_kind"]
    assert trace["visible_value"] is not None
    # This smoke intentionally does not require raw_outputs/generated artifacts to exist.
    assert set(problems).issubset(
        {
            "source JSON is missing locally",
            "source JSON SHA-256 does not match source map",
            "no reusable settings/contracts found in source JSON",
        }
    )
