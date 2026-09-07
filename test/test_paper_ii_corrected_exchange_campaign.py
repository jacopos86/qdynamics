"""Contract for packaging the corrected exchange campaign from its own lock.

The fifteen-cell final campaign searched under a profile that never
enumerated a true-exchange patch.  This lock re-runs the eleven surviving
cells on the single q-compiled route under ``paper_declared_exchange_v1``,
whose declared insertion frontier is what makes the exchange face reachable.
The packager must therefore accept a lock other than its module default
without loosening any check that is not the shape the lock declares.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.time_dynamics.campaigns import (
    paper_ii_final_estimator_telemetry_v1 as telemetry,
)
from pipelines.time_dynamics.campaigns import (
    paper_ii_final_selector_comparison_v1 as selector,
)
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import _build_parser

LOCK_JSON = (
    Path(selector.__file__).with_name("paper_ii_corrected_exchange_cells_v1.json")
)
CAMPAIGN_ID = "paper_ii_corrected_exchange_v1"
ROUTE_ID = "paper_ii_exchange_q_compiled_v1"
ARM_ID = "exchange_q_compiled"
SEARCH_PROFILE = "paper_declared_exchange_v1"
FRONTIER = "8"


@pytest.fixture(scope="module")
def lock() -> selector.MatchedCellLock:
    return selector.load_lock(LOCK_JSON, **selector.lock_parameters(LOCK_JSON))


def _pairs(argv: list[str]) -> set[tuple[str, str]]:
    return set(zip(argv, argv[1:]))


def test_corrected_lock_loads_as_eleven_cells_on_one_declared_exchange_route(
    lock: selector.MatchedCellLock,
) -> None:
    assert lock.campaign_id == CAMPAIGN_ID
    assert len(lock.cells) == 11
    assert len({cell.cell_id for cell in lock.cells}) == 11
    assert lock.selector_routes == {ROUTE_ID: ARM_ID}
    assert lock.registry_bindings["search_profile"] == SEARCH_PROFILE
    assert lock.registry_bindings["apm_arm"] == "exchange"
    assert lock.payload["blocks"]["active"] == []
    # The loader's partition check is kept, so the eleven cells and the seven
    # exclusions must still cover the whole drive x controller grid exactly.
    pairs = {(cell.drive, cell.controller) for cell in lock.cells}
    assert len(lock.exclusions) == 7
    assert pairs | set(lock.exclusions) == set(lock.grid)
    assert not pairs & set(lock.exclusions)


def test_the_loader_still_fails_closed_on_the_module_default_identity() -> None:
    with pytest.raises(selector.MatchedCellLockError, match="campaign_id mismatch"):
        selector.load_lock(LOCK_JSON)


def test_packager_yields_eleven_apm_actions_carrying_the_declared_frontier(
    lock: selector.MatchedCellLock, tmp_path: Path
) -> None:
    rows = telemetry.build_actions(
        lock,
        output_root=tmp_path,
        comparator_mode=telemetry.COMPARATOR_MODE_REUSE,
    )
    assert len(rows) == 11
    assert sum(row["method"] == "apm" for row in rows) == 11
    assert sum(row["method"] == "avqds" for row in rows) == 0
    assert {row["cell_id"] for row in rows} == {cell.cell_id for cell in lock.cells}

    parser = _build_parser()
    cuts = {cell.cell_id: cell.apm_cut for cell in lock.cells}
    for row in rows:
        argv = list(row["runner_argv"])
        pairs = _pairs(argv)
        assert ("--exchange-insertion-frontier", FRONTIER) in pairs
        assert ("--exchange-route-bundle", ROUTE_ID) in pairs
        parsed = parser.parse_args(argv)
        assert parsed.exchange_insertion_frontier == 8
        assert parsed.exchange_route_bundle == ROUTE_ID
        assert parsed.dynamics_policy == "exchange"
        assert parsed.insertion_l2_cut == pytest.approx(
            cuts[row["cell_id"]], rel=1.0e-12, abs=0.0
        )
        assert row["campaign_id"] == CAMPAIGN_ID
        assert row["arm"] == ARM_ID
        assert row["exchange_route_bundle"] == ROUTE_ID
        assert row["trajectory_json"] == str(
            tmp_path / CAMPAIGN_ID / row["action_id"] / "run.json"
        )
        assert parsed.output_json == row["trajectory_json"]
        comparator = row["comparator"]
        assert comparator["reuse"] == telemetry.COMPARATOR_MODE_REUSE
        assert comparator["action"] == f"{row['cell_id']}__avqds"
        assert (
            comparator["source_campaign_id"]
            == telemetry.COMPARATOR_SOURCE_CAMPAIGN_ID
        )


def test_only_the_comparator_switch_suppresses_the_avqds_reruns(
    lock: selector.MatchedCellLock, tmp_path: Path
) -> None:
    fresh = telemetry.build_actions(lock, output_root=tmp_path)
    assert sum(row["method"] == "avqds" for row in fresh) == 11
    with pytest.raises(selector.MatchedCellLockError, match="unknown comparator_mode"):
        telemetry.build_actions(lock, output_root=tmp_path, comparator_mode="off")


def test_prepare_writes_an_eleven_action_summary_under_the_corrected_campaign(
    lock: selector.MatchedCellLock, tmp_path: Path
) -> None:
    actions = tmp_path / "input" / "actions.jsonl"
    summary = telemetry.prepare(
        output_root="raw_outputs",
        actions_jsonl=actions,
        lock_json=LOCK_JSON,
        source_package_sha256="c" * 64,
        compile_workers=4,
        comparator_mode=telemetry.COMPARATOR_MODE_REUSE,
    )
    assert summary["campaign_id"] == CAMPAIGN_ID
    assert summary["lock"]["source_campaign_id"] == CAMPAIGN_ID
    assert summary["status"] == "PREPARED_NOT_SUBMITTED"
    assert summary["cell_count"] == 11
    assert summary["excluded_cell_count"] == 7
    assert summary["route_count"] == 1
    assert summary["action_count"] == 11
    assert summary["apm_action_count"] == 11
    assert summary["avqds_action_count"] == 0
    assert summary["comparator_mode"] == telemetry.COMPARATOR_MODE_REUSE
    assert (
        summary["comparator_source_campaign_id"]
        == telemetry.COMPARATOR_SOURCE_CAMPAIGN_ID
    )
    assert summary["registry_bindings"]["search_profile"] == SEARCH_PROFILE
    assert summary["selector_routes"] == {ROUTE_ID: ARM_ID}
    assert summary["blocks"] == []
    rows = [json.loads(line) for line in actions.read_text().splitlines() if line]
    assert len(rows) == 11
    assert all(
        row["trajectory_json"].startswith(f"raw_outputs/{CAMPAIGN_ID}/") for row in rows
    )
    assert all(row["source_package_sha256"] == "c" * 64 for row in rows)


def test_the_fifteen_cell_campaign_keeps_its_own_identity() -> None:
    historical = selector.load_lock()
    assert historical.campaign_id == selector.CAMPAIGN_ID
    assert telemetry.campaign_id_for(historical) == telemetry.CAMPAIGN_ID
