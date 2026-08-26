"""Paper-II factorial campaign preparation and anti-drift invariants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.time_dynamics import paper_ii_runs as runs
from pipelines.time_dynamics.campaign import (
    CAMPAIGN_SCHEMA_V2,
    PAPER_I_FAKE_MARRAKESH,
    CampaignSpec,
    SeedSpec,
    audit_completed_campaign,
    plan_thresholds_from_prior,
    prepare_chtc_campaign,
    uniform_threshold_plan,
    write_campaign_manifest,
)
from pipelines.time_dynamics.campaigns.paper_ii_factorial_euler_v1 import (
    CONTROLLER_IDS,
    METHOD_IDS,
    build_spec,
)
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
    _build_parser,
)


def _write_seed(path: Path) -> SeedSpec:
    path.write_text('{"seed": true}\n', encoding="utf-8")
    return SeedSpec("seed", str(path), "hh", 1, "calibration")


def _spec(seed: SeedSpec, *, drives=("fastweak",)) -> CampaignSpec:
    plan = uniform_threshold_plan(
        method_ids=METHOD_IDS,
        controller_ids=CONTROLLER_IDS,
        drive_ids=drives,
        thresholds=(1.0e-3, 3.0e-4),
    )
    return CampaignSpec(
        campaign_id="test_factorial",
        seeds=(seed,),
        method_ids=METHOD_IDS,
        controller_ids=CONTROLLER_IDS,
        drive_ids=tuple(drives),
        horizon_ids=("smoke",),
        threshold_plan=plan,
        numerics_id=runs.EULER_RIDGE1E6_NUMERICS.numerics_id,
        output_root="raw_outputs",
    )


def test_binary_aligned_cutoffs_are_enforced() -> None:
    for good in (1, 3, 7):
        SeedSpec("s", "seed.json", "hh", good, "r")
    for bad in (0, 2, 4, 5, 6, 8):
        with pytest.raises(ValueError, match="binary phonon register"):
            SeedSpec("s", "seed.json", "hh", bad, "r")


def test_factorial_matrix_expands_method_controller_drive_threshold_product(
    tmp_path: Path,
) -> None:
    spec = _spec(_write_seed(tmp_path / "seed.json"), drives=("fastweak", "weakslow"))
    assert spec.cell_count() == 2 * 3 * 2 * 2
    cells = list(spec.cells())
    assert len({cell.cell_id for cell in cells}) == len(cells)
    assert {
        (cell.method_id, cell.controller_id)
        for cell in cells
    } == set(__import__("itertools").product(METHOD_IDS, CONTROLLER_IDS))


def test_every_cell_uses_shared_euler_ridge_and_method_specific_cut(
    tmp_path: Path,
) -> None:
    parser = _build_parser()
    spec = _spec(_write_seed(tmp_path / "seed.json"))
    for cell in spec.cells():
        parsed = parser.parse_args(cell.runner_argv())
        assert parsed.integrator == "euler"
        assert parsed.ridge_lambda == pytest.approx(1.0e-6)
        if cell.method_id == "avqds":
            assert parsed.avqds_l2_cut == pytest.approx(cell.activation_cut)
        else:
            assert parsed.insertion_l2_cut == pytest.approx(cell.activation_cut)


def test_three_controller_protocols_resolve_to_distinct_live_settings(
    tmp_path: Path,
) -> None:
    parser = _build_parser()
    spec = _spec(_write_seed(tmp_path / "seed.json"))
    values = {}
    for cell in spec.cells():
        if cell.method_id != "exchange" or cell.activation_cut != 1.0e-3:
            continue
        parsed = parser.parse_args(cell.runner_argv())
        values[cell.controller_id] = (
            parsed.solve_repair,
            parsed.avqds_delta_theta_max,
            parsed.solve_repair_state_motion_l2_step_max,
            parsed.solve_repair_parameter_step_max,
        )
    assert values["delta_theta_5e-3"][0:2] == (
        False,
        pytest.approx(5.0e-3),
    )
    assert values["state_motion_1e-2"][0] is True
    assert values["state_motion_1e-2"][2] == pytest.approx(1.0e-2)
    assert values["state_motion_1e-2_plus_parameter_5e-3"][2:] == (
        pytest.approx(1.0e-2),
        pytest.approx(5.0e-3),
    )


def test_paper_i_compile_profile_is_locked() -> None:
    assert PAPER_I_FAKE_MARRAKESH.backend_name == "FakeMarrakesh"
    assert PAPER_I_FAKE_MARRAKESH.native_basis == ("sx", "rz", "x", "cz", "id")
    assert PAPER_I_FAKE_MARRAKESH.optimization_level == 1
    assert PAPER_I_FAKE_MARRAKESH.seed_transpiler == 7


def test_prior_planner_anchors_target_hits_and_extends_target_misses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior = tmp_path / "prior"
    prior.mkdir()
    monkeypatch.setattr(
        "pipelines.time_dynamics.accuracy_target_report.collect",
        lambda _root: {
            ("fastweak", "append_only"): [
                (1.0e-4, 3.0e-4, 30),
                (3.0e-5, 8.0e-5, 44),
                (1.0e-5, 6.0e-6, 48),
            ],
            ("fastweak", "avqds"): [(3.0e-6, 2.0e-4, 50)],
        },
    )
    plan = plan_thresholds_from_prior(
        prior_root=prior,
        method_ids=METHOD_IDS,
        controller_ids=CONTROLLER_IDS,
        drive_ids=("fastweak", "weakslow"),
    )
    fast = plan.thresholds_by_configuration[
        "exchange__state_motion_1e-2__fastweak"
    ]
    assert 3.0e-5 in fast  # resource-cheapest old target hit
    assert 1.0e-6 in fast  # next rung after the old AVQDS target miss
    assert plan.thresholds_by_configuration[
        "avqds__delta_theta_5e-3__weakslow"
    ] == (
        1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5, 3.0e-6
    )


def test_prior_planner_deduplicates_overlapping_neighbor_rungs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior = tmp_path / "prior"
    prior.mkdir()
    monkeypatch.setattr(
        "pipelines.time_dynamics.accuracy_target_report.collect",
        lambda _root: {
            ("fastweak", "append_only"): [(3.0e-5, 8.0e-5, 44)],
            ("fastweak", "avqds"): [(1.0e-5, 8.0e-5, 46)],
        },
    )
    plan = plan_thresholds_from_prior(
        prior_root=prior,
        method_ids=METHOD_IDS,
        controller_ids=CONTROLLER_IDS,
        drive_ids=("fastweak",),
    )
    cuts = plan.thresholds_by_configuration[
        "exchange__state_motion_1e-2__fastweak"
    ]
    assert cuts == (1.0e-4, 3.0e-5, 1.0e-5, 3.0e-6)
    assert len(cuts) == len(set(cuts))


def test_manifest_records_factorial_terms_threshold_provenance_and_compile_lock(
    tmp_path: Path,
) -> None:
    spec = _spec(_write_seed(tmp_path / "seed.json"))
    path = write_campaign_manifest(spec, tmp_path / "manifest.json")
    payload = json.loads(path.read_text())
    assert payload["schema"] == CAMPAIGN_SCHEMA_V2
    assert payload["cell_count"] == 12
    assert payload["compile_profile"]["backend_name"] == "FakeMarrakesh"
    assert payload["threshold_plan"]["target_semantics"].startswith("offline")
    assert all("algorithmic_method" in row["factors"] for row in payload["cells"])


def test_prepare_package_is_fail_closed_and_uses_json_argv(tmp_path: Path) -> None:
    missing = SeedSpec("missing", str(tmp_path / "missing.json"), "hh", 1, "r")
    with pytest.raises(FileNotFoundError, match="missing campaign seed"):
        prepare_chtc_campaign(_spec(missing), tmp_path / "bad")

    prepared = prepare_chtc_campaign(
        _spec(_write_seed(tmp_path / "seed.json")), tmp_path / "package"
    )
    audit = json.loads(prepared.audit.read_text())
    assert audit["status"] == "PASS"
    assert audit["submission_status"] == "NOT_SUBMITTED"
    records = [json.loads(line) for line in prepared.cells_jsonl.read_text().splitlines()]
    assert len(records) == 12
    assert all(isinstance(record["runner_argv"], list) for record in records)
    assert all(
        record["compile_profile"] == PAPER_I_FAKE_MARRAKESH.to_json_dict()
        for record in records
    )
    assert all(record["trajectory_json"].endswith("/run.json") for record in records)
    run_cell_source = prepared.run_cell.read_text()
    assert "$ARGV" not in run_cell_source
    assert "ap_terminal_qiskit_cost" in run_cell_source
    assert "terminal_qiskit_cost.json" in run_cell_source


def test_result_audit_fails_closed_when_any_declared_output_is_missing(
    tmp_path: Path,
) -> None:
    prepared = prepare_chtc_campaign(
        _spec(_write_seed(tmp_path / "seed.json")), tmp_path / "package"
    )
    with pytest.raises(ValueError, match="missing output"):
        audit_completed_campaign(
            prepared.manifest,
            attach_exact_energy_error=False,
            require_qiskit_cost=False,
        )
    payload = json.loads((prepared.manifest.parent / "result_audit.json").read_text())
    assert payload["status"] == "FAIL"
    assert payload["submission_recommendation"] == "STOP"


def test_canonical_smoke_is_six_configurations_and_not_submitted() -> None:
    spec = build_spec(mode="smoke", prior_root="output/frontier")
    assert spec.cell_count() == 6
    assert spec.method_ids == ("exchange", "avqds")
    assert len(spec.controller_ids) == 3
    assert spec.numerics_id == "euler_ridge1e-6"
