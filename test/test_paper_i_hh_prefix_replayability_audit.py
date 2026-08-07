#!/usr/bin/env python3
"""Regression checks for Paper-I HH Table-III exact-prefix replayability audit."""

from __future__ import annotations

import json
from pathlib import Path

from pipelines.reporting.audit_paper_i_hh_prefix_replayability import build_audit, main


SOURCE_MAP = Path("MATH/paper_facing/paper_I_static_scaffold/hh_tableiii_convergence_sources.json")


def _rows_by_key(audit: dict) -> dict[tuple[str, str], dict]:
    return {(row["regime"], row["method"]): row for row in audit["rows"]}


def test_prefix_replayability_audit_classifies_current_visible_sources() -> None:
    audit = build_audit(SOURCE_MAP)
    rows = _rows_by_key(audit)

    assert audit["schema"] == "paper_i_hh_tableiii_prefix_replayability_audit_v1"
    assert audit["row_count"] == 16
    assert audit["status_counts"] == {
        "exact-prefix-replay-ready": 12,
        "needs-richer-history": 1,
        "stdout-only-blocked": 3,
    }

    for regime in ("weak_weak", "strong_weak", "weak_strong", "strong_strong"):
        for method in ("Append-ADAPT", "TETRIS-ADAPT", "Geo-ADAPT"):
            row = rows[(regime, method)]
            assert row["classification"] == "exact-prefix-replay-ready"
            assert row["blockers"] == []
            assert row["primary_source"]["history_len"] == row["last_history_len_from_source_map"]

    for regime in ("strong_weak", "weak_strong", "strong_strong"):
        row = rows[(regime, "SNAKE")]
        assert row["classification"] == "stdout-only-blocked"
        assert "visible_source_is_stdout_or_ai_log_derived" in row["blockers"]

    weak_weak_snake = rows[("weak_weak", "SNAKE")]
    assert weak_weak_snake["classification"] == "needs-richer-history"
    assert "missing_runtime_seed_or_strict_resume_state" in weak_weak_snake["blockers"]


def test_prefix_replayability_audit_records_tetris_batch_semantics() -> None:
    audit = build_audit(SOURCE_MAP)
    rows = _rows_by_key(audit)

    tetris = rows[("weak_weak", "TETRIS-ADAPT")]
    assert tetris["classification"] == "exact-prefix-replay-ready"
    primary = next(candidate for candidate in tetris["candidates"] if candidate["primary_visible_source"])
    assert primary["history"]["history_len"] == 20
    assert primary["history"]["batch_size_histogram"]
    assert any(int(size) > 1 for size in primary["history"]["batch_size_histogram"])
    assert primary["recoverable_pauli_source"]["final_selected_operator_pauli_labels_exyz"] is True


def test_prefix_replayability_audit_cli_writes_json(tmp_path: Path) -> None:
    output_json = tmp_path / "audit.json"
    exit_code = main(["--source-map", str(SOURCE_MAP), "--output-json", str(output_json)])

    assert exit_code == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema"] == "paper_i_hh_tableiii_prefix_replayability_audit_v1"
    assert payload["row_count"] == 16
