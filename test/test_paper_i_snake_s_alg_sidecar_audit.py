#!/usr/bin/env python3
"""Tests for Paper-I SNAKE terminal S_alg sidecar auditing."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from pipelines.exact_bench.snake_table_i_measurement_work import (
    BEAM_AGGREGATE_RUN_SCOPE,
    BEAM_TERMINAL_ROW_POLICY,
    SNAKE_TERMINAL_WORK_SEMANTICS_VERSION,
)
from pipelines.reporting.build_paper_i_hh_fullmeta_singleton_symmetry_matrix_pdf import _snake_s_work


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = (
    REPO_ROOT
    / "agent_guidance"
    / "skills"
    / "paper-i-results"
    / "scripts"
    / "audit_paper_i_snake_s_alg_sidecars.py"
)


def _load_audit_module() -> Any:
    spec = importlib.util.spec_from_file_location("audit_paper_i_snake_s_alg_sidecars", AUDIT_SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_sidecar(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "paper_i_terminal_qiskit_cost.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _winner_beam_sidecar() -> dict[str, Any]:
    return {
        "schema": "paper_i_hh_child_fairness_snake_terminal_qiskit_cost_v2",
        "work_semantics_version": SNAKE_TERMINAL_WORK_SEMANTICS_VERSION,
        "status": "done",
        "S_alg_status": "ok",
        "S_alg": 26,
        "S_alg_N_H_outer_eval": 3,
        "S_alg_N_grad_probe": 6,
        "S_alg_N_metric_probe": 5,
        "S_alg_N_H_refit_eval": 12,
        "S_alg_work_scope": "winner_lineage_terminal",
        "S_alg_row_policy": BEAM_TERMINAL_ROW_POLICY,
        "S_beam_search_total": 1150,
        "S_beam_search_total_status": "ok",
        "S_beam_search_scope": BEAM_AGGREGATE_RUN_SCOPE,
        "work_reconstruction": {
            "beam_search_total_reconstruction": {
                "promoted_to_row_s_alg": False,
            }
        },
    }


def test_snake_s_alg_sidecar_audit_accepts_winner_lineage_beam_sidecar(tmp_path: Path) -> None:
    audit_module = _load_audit_module()
    sidecar = _write_sidecar(tmp_path, _winner_beam_sidecar())

    row = audit_module.audit_sidecar(sidecar)
    report = audit_module.build_audit([tmp_path])

    assert row["status"] == "ok"
    assert row["issue_count"] == 0
    assert row["S_alg"] == 26
    assert row["S_beam_search_total"] == 1150
    assert report["status"] == "ok"
    assert report["row_count"] == 1


def test_snake_s_alg_sidecar_audit_rejects_stale_aggregate_beam_sidecar(tmp_path: Path) -> None:
    audit_module = _load_audit_module()
    stale = {
        "schema": "paper_i_hh_child_fairness_snake_terminal_qiskit_cost_v1",
        "status": "done",
        "S_alg_status": "ok",
        "S_alg": 1150,
        "S_alg_N_H_outer_eval": 100,
        "S_alg_N_grad_probe": 500,
        "S_alg_N_metric_probe": 300,
        "S_alg_N_H_refit_eval": 250,
        "S_beam_search_total": 1150,
        "S_beam_search_total_status": "ok",
        "S_beam_search_scope": BEAM_AGGREGATE_RUN_SCOPE,
    }
    sidecar = _write_sidecar(tmp_path, stale)

    row = audit_module.audit_sidecar(sidecar)

    assert row["status"] == "blocked:unsafe_or_stale_s_alg_semantics"
    assert "missing_or_old_work_semantics_version" in row["issues"]
    assert "beam_row_s_alg_scope_not_winner_lineage:missing" in row["issues"]
    assert "beam_row_policy_missing_or_wrong:missing" in row["issues"]


def test_fullmeta_report_s_work_blocks_current_version_with_missing_beam_row_policy() -> None:
    unsafe = _winner_beam_sidecar()
    unsafe.pop("S_alg_work_scope")
    unsafe.pop("S_alg_row_policy")

    result = _snake_s_work({}, unsafe)

    assert result["s_alg"] is None
    assert result["s_work_status"] == "blocked:beam_row_s_alg_scope_not_winner_lineage"
