from __future__ import annotations

import json
from pathlib import Path

from pipelines.reporting.update_paper_i_hh_joint_response_pareto_ledger import (
    BEGIN_MARKER,
    END_MARKER,
    update_ledger,
)


def _row(regime: str, method: str, base: int) -> dict[str, object]:
    return {
        "regime": regime,
        "method": method,
        "k_pl": base,
        "ansatz_depth": base + 2,
        "abs_delta_e": 1.0 / base,
        "N2q": base + 10,
        "D2q": base + 20,
        "Dc": base + 30,
        "S": base + 40,
    }


def test_ledger_update_is_idempotent_and_preserves_existing_evidence(tmp_path: Path) -> None:
    report_json = tmp_path / "report.json"
    ledger = tmp_path / "ledger.md"
    rows = []
    for regime_index, regime in enumerate(
        ("weak-weak", "intermediate-weak", "strong-weak"),
        start=1,
    ):
        rows.extend(
            (
                _row(regime, "joint_response_snake", 10 * regime_index),
                _row(regime, "snake", 10 * regime_index + 1),
                _row(regime, "geo", 10 * regime_index + 2),
                _row(regime, "append", 10 * regime_index + 3),
            )
        )
    report_json.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    ledger.write_text(
        "# Ledger\n\nExisting evidence remains.\n\n## Candidate policies\n\nKeep me.\n",
        encoding="utf-8",
    )

    update_ledger(report_json=report_json, ledger_md=ledger)
    first = ledger.read_text(encoding="utf-8")
    update_ledger(report_json=report_json, ledger_md=ledger)
    second = ledger.read_text(encoding="utf-8")

    assert first == second
    assert first.count(BEGIN_MARKER) == 1
    assert first.count(END_MARKER) == 1
    assert "Existing evidence remains." in first
    assert "## Candidate policies" in first
    assert "Keep me." in first
    assert "Selected round / ansatz depth" in first
    assert "winning-branch `S_alg`" in first
