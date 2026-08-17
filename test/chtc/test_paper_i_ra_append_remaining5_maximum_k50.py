from __future__ import annotations

import importlib


MODULE = (
    "chtc.paper_i_ra_adapt_repair_20260727."
    "run_local_paper_i_ra_append_remaining5_maximum_k50_20260817"
)


def _runner():
    return importlib.import_module(MODULE)


def test_campaign_contains_only_the_five_remaining_append_cells() -> None:
    runner = _runner()

    assert runner.CAMPAIGN_ID == (
        "paper_i_ra_allphase_adaptive_append_remaining5_maximum_k50_"
        "20260817_v3"
    )
    assert [cell.regime_id for cell in runner.CELL_SPECS] == [
        "intermediate_weak",
        "strong_weak_u8",
        "weak_strong",
        "intermediate_strong",
        "strong_strong_u8",
    ]
    assert [cell.nph for cell in runner.CELL_SPECS] == [3, 3, 7, 7, 7]
    assert all(cell.block == "append" for cell in runner.CELL_SPECS)
    assert all(cell.insertion_policy == "append_only" for cell in runner.CELL_SPECS)
    assert all(cell.horizon == 50 for cell in runner.CELL_SPECS)
    assert all("natural_terminal" in cell.route_variant for cell in runner.CELL_SPECS)
    assert not any("weak_weak" in cell.execution_id for cell in runner.CELL_SPECS)
    assert len({cell.execution_id for cell in runner.CELL_SPECS}) == 5


def test_plan_authorizes_no_unrequested_science(monkeypatch) -> None:
    runner = _runner()
    inventory = {"sha256": "a" * 64, "source_count": 7}
    monkeypatch.setattr(runner, "_source_inventory", lambda: inventory)
    monkeypatch.setattr(
        runner,
        "_protocol_binding",
        lambda cell: {
            "execution_id": cell.execution_id,
            "source_implementation_inventory_sha256": inventory["sha256"],
            "protocol_sha256": (str(cell.ordinal) * 64)[:64],
        },
    )
    monkeypatch.setattr(
        runner,
        "file_binding",
        lambda path: {"path": str(path), "size_bytes": 1, "sha256": "b" * 64},
    )

    plan = runner.build_plan()

    assert plan["execution_authorized"] is False
    assert plan["archive_rotation_authorized"] is False
    assert plan["submission_authorized"] is False
    assert plan["paper_adoption_authorized"] is False
    assert plan["paper_evidence_adoption_authorized"] is False
    assert plan["maximum_controller_rounds"] == 50
    assert plan["allowed_cell_completions"] == [
        runner.MAXIMUM_COMPLETION_KIND,
        runner.NATURAL_COMPLETION_KIND,
    ]
    assert plan["insertion_policies"] == ["append_only"]
    assert plan["execution_ids"] == [cell.execution_id for cell in runner.CELL_SPECS]
    assert len(plan["protocol_bindings"]) == 5
    assert plan["source_implementation_inventory_sha256"] == inventory["sha256"]


def test_scheduler_falls_back_to_serial_when_pair_capacity_is_absent() -> None:
    runner = _runner()

    assert runner.choose_concurrency(
        available_memory_bytes=6 * 1024**3,
        free_disk_bytes=32 * 1024**3,
    ) == 1
    assert runner.choose_concurrency(
        available_memory_bytes=16 * 1024**3,
        free_disk_bytes=32 * 1024**3,
    ) == 2
