"""Anti-drift contract for the final exact-estimator telemetry reruns."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.time_dynamics import paper_ii_runs as runs
from pipelines.time_dynamics.campaigns import (
    paper_ii_final_estimator_telemetry_v1 as telemetry,
)
from pipelines.time_dynamics.campaigns import (
    paper_ii_final_selector_comparison_v1 as selector,
)
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
    _build_parser,
)


@pytest.fixture(scope="module")
def lock() -> selector.MatchedCellLock:
    return selector.load_lock()


def _pairs(argv: list[str]) -> set[tuple[str, str]]:
    return set(zip(argv, argv[1:]))


def test_builds_exactly_three_apm_routes_plus_fresh_avqds_per_cell(
    lock: selector.MatchedCellLock, tmp_path: Path
) -> None:
    rows = telemetry.build_actions(lock, output_root=tmp_path)
    assert len(rows) == telemetry.EXPECTED_ACTION_COUNT == 60
    assert sum(row["method"] == "apm" for row in rows) == 45
    assert sum(row["method"] == "avqds" for row in rows) == 15
    assert all(row["precursor"] is False for row in rows)
    assert {row["cell_id"] for row in rows} == {cell.cell_id for cell in lock.cells}
    assert len({row["action_id"] for row in rows}) == 60
    assert len({row["run_id"] for row in rows}) == 60

    by_cell: dict[str, list[dict]] = {}
    for row in rows:
        by_cell.setdefault(row["cell_id"], []).append(row)
    assert set(map(len, by_cell.values())) == {4}
    for cell_rows in by_cell.values():
        assert sum(row["method"] == "apm" for row in cell_rows) == 3
        assert sum(row["method"] == "avqds" for row in cell_rows) == 1


def test_all_actions_are_fresh_locked_t20_runs_with_exact_telemetry_contract(
    lock: selector.MatchedCellLock, tmp_path: Path
) -> None:
    parser = _build_parser()
    cells = {cell.cell_id: cell for cell in lock.cells}
    rows = telemetry.build_actions(
        lock,
        output_root=tmp_path,
        source_package_sha256="a" * 64,
        compile_workers=8,
    )
    for row in rows:
        argv = list(row["runner_argv"])
        parsed = parser.parse_args(argv)
        cell = cells[row["cell_id"]]
        assert row["schema"] == telemetry.ACTIONS_SCHEMA
        assert row["campaign_id"] == telemetry.CAMPAIGN_ID
        assert row["estimator_telemetry_contract"] == telemetry.TELEMETRY_CONTRACT
        assert row["source_package_sha256"] == "a" * 64
        assert row["matched_cell_lock_sha256"] == lock.sha256
        assert row["seed_sha256"] == lock.seed["sha256"]
        assert row["trajectory_json"] == str(
            tmp_path / telemetry.CAMPAIGN_ID / row["action_id"] / "run.json"
        )
        assert parsed.output_json == row["trajectory_json"]
        assert parsed.artifact_json == runs.REGIMES["hh_snake_nph1"].seed_path
        assert parsed.t_final == 20.0
        assert parsed.num_times == 501
        assert parsed.record_statevector is True
        assert parsed.resume_from_run_json is None
        assert "--resume-from-run-json" not in argv

        if row["method"] == "apm":
            assert parsed.insertion_l2_cut == pytest.approx(
                cell.apm_cut, rel=1.0e-12, abs=0.0
            )
            assert parsed.exchange_route_bundle == row["exchange_route_bundle"]
            assert ("--compile-workers", "8") in _pairs(argv)
            assert ("--support-patch-scoring-workers", "4") in _pairs(argv)
            assert row["comparator"]["reuse"] == telemetry.COMPARATOR_MODE
            assert row["comparator"]["action"] == f"{cell.cell_id}__avqds"
        else:
            assert parsed.dynamics_policy == "avqds"
            assert parsed.avqds_l2_cut == pytest.approx(
                cell.avqds_cut, rel=1.0e-12, abs=0.0
            )
            assert row["exchange_route_bundle"] is None
            assert row["exchange_route_contract"] is None
            assert row["exchange_route_contract_sha256"] is None
            assert "--exchange-route-bundle" not in argv
            assert "--insertion-gate-mode" not in argv
            assert "--compile-workers" not in argv
            assert set(row["paired_apm_actions"]) == {
                f"{cell.cell_id}__{arm}"
                for arm in selector.bind_selector_routes(lock.selector_routes).values()
            }


def test_prepare_writes_reproducible_60_action_summary(
    lock: selector.MatchedCellLock, tmp_path: Path
) -> None:
    actions = tmp_path / "input" / "actions.jsonl"
    summary = telemetry.prepare(
        output_root="raw_outputs",
        actions_jsonl=actions,
        source_package_sha256="b" * 64,
        compile_workers=8,
    )
    assert summary["status"] == "PREPARED_NOT_SUBMITTED"
    assert summary["campaign_id"] == telemetry.CAMPAIGN_ID
    assert summary["telemetry_contract"] == telemetry.TELEMETRY_CONTRACT
    assert summary["cell_count"] == 15
    assert summary["action_count"] == 60
    assert summary["apm_action_count"] == 45
    assert summary["avqds_action_count"] == 15
    assert summary["precursor_action_count"] == 0
    assert summary["comparator_mode"] == "fresh_exact_telemetry"
    assert summary["compile_workers"] == 8
    assert summary["blocks"] == []
    assert summary["lock"]["sha256"] == lock.sha256
    assert summary["lock"]["source_campaign_id"] == selector.CAMPAIGN_ID
    rows = [json.loads(line) for line in actions.read_text().splitlines() if line]
    assert len(rows) == 60
    assert [row["action_id"] for row in rows[:3]] == [
        f"{lock.cells[0].cell_id}__{arm}"
        for arm in selector.bind_selector_routes(lock.selector_routes).values()
    ]
    assert rows[45]["action_id"] == f"{lock.cells[0].cell_id}__avqds"


def test_invalid_compile_worker_or_source_digest_fails_closed(
    lock: selector.MatchedCellLock, tmp_path: Path
) -> None:
    with pytest.raises(selector.MatchedCellLockError, match="positive"):
        telemetry.build_actions(lock, output_root=tmp_path, compile_workers=0)
    with pytest.raises(selector.MatchedCellLockError, match="invalid sha256"):
        telemetry.build_actions(
            lock,
            output_root=tmp_path,
            source_package_sha256="not-a-digest",
        )
