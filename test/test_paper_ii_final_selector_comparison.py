"""Anti-drift and anti-fabrication contract for the final selector comparison."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pipelines.time_dynamics import paper_ii_runs as runs
from pipelines.time_dynamics.campaign import _canonical_cut
from pipelines.time_dynamics.campaigns import (
    paper_ii_final_selector_comparison_v1 as final,
)
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
    _build_parser,
)

ROOT = Path(__file__).resolve().parents[1]
SEED_SHA256 = "7a36a2da06bba447c6d65d95692c8a45bc3aa64933d02f1f641e2178f4ea8f91"
ROUTES = {
    "paper_ii_exchange_legacy_proxy_fixed_v1": "exchange_legacy_proxy_fixed",
    "paper_ii_exchange_q_compiled_v1": "exchange_q_compiled",
    "paper_ii_exchange_gamma_compiled_v1": "exchange_gamma_compiled",
}
EXCLUDED = {
    ("midres", "state_motion_1e-3_plus_parameter_5e-5"),
    ("slowstrong", "state_motion_1e-3_plus_parameter_5e-5"),
    ("strongfast", "state_motion_5e-5"),
}


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(final.LOCK_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def lock() -> final.MatchedCellLock:
    return final.load_lock()


def _leaves(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from _leaves(value)
    elif isinstance(node, list):
        for value in node:
            yield from _leaves(value)
    else:
        yield node


# --- lock identity -----------------------------------------------------------


def test_lock_identity_and_status(payload: dict) -> None:
    assert payload["schema"] == "paper_ii_final_matched_cells_v1"
    assert payload["status"] == "PREPARED_NOT_SUBMITTED"
    assert payload["campaign_id"] == final.CAMPAIGN_ID == (
        "paper_ii_final_selector_comparison_v1"
    )
    assert "code_revision" not in payload


def test_lock_is_exactly_fifteen_unique_cells(lock: final.MatchedCellLock) -> None:
    assert len(lock.cells) == 15
    ids = [cell.cell_id for cell in lock.cells]
    assert len(set(ids)) == 15
    pairs = {(cell.drive, cell.controller) for cell in lock.cells}
    assert len(pairs) == 15


def test_drives_and_controllers_key_into_the_registries(
    lock: final.MatchedCellLock,
) -> None:
    for cell in lock.cells:
        assert cell.drive in runs.DRIVES
        assert cell.controller in runs.STEP_CONTROLS
        assert cell.adoption_status in final.ADOPTION_STATUSES
    bindings = lock.registry_bindings
    assert bindings["regime"] == "hh_snake_nph1"
    assert bindings["horizon"] == "t20"
    assert bindings["numerics"] == runs.EULER_RIDGE1E6_NUMERICS.numerics_id
    assert bindings["gate"] == runs.MCLACHLAN_L2_GATE.gate_id
    assert bindings["search_profile"] == runs.PAPER_DECLARED_SEARCH.profile_id
    assert bindings["apm_arm"] == "exchange"
    assert bindings["comparator_arm"] == "avqds"
    assert runs.ARMS["avqds"].is_comparator


def test_cells_and_exclusions_reconstruct_the_six_by_three_grid(
    lock: final.MatchedCellLock, payload: dict
) -> None:
    grid = set(lock.grid)
    assert len(grid) == 18
    assert len(payload["grid"]["drives"]) == 6
    assert len(payload["grid"]["controllers"]) == 3
    present = {(cell.drive, cell.controller) for cell in lock.cells}
    excluded = set(lock.exclusions)
    assert excluded == EXCLUDED
    assert present | excluded == grid
    assert not (present & excluded)
    for row in payload["exclusions"]:
        assert row["reason"] == final.EXCLUSION_REASON == "not_in_final_matched_set"


def test_cuts_are_canonical_and_labels_unique_per_cell(
    lock: final.MatchedCellLock, payload: dict
) -> None:
    # Canonical form is ``.12e`` (13 significant digits); the run-id label is
    # ``.12g`` (12), so the label is lossy on the finest ladder rungs and must
    # be tested for injectivity over the locked cut set, not for round-trip.
    labels_by_cell: dict[str, tuple[str, str]] = {}
    for raw, cell in zip(payload["cells"], lock.cells):
        for value in (raw["apm_cut"], raw["avqds_cut"]):
            assert _canonical_cut(value) == value
            assert _canonical_cut(_canonical_cut(value)) == value
        labels_by_cell[cell.cell_id] = (cell.apm_cut_label, cell.avqds_cut_label)
        assert cell.apm_cut_label == f"{cell.apm_cut:.12g}"
        assert float(cell.apm_cut_label) == pytest.approx(cell.apm_cut, rel=1.0e-11, abs=0.0)
        assert float(cell.avqds_cut_label) == pytest.approx(cell.avqds_cut, rel=1.0e-11, abs=0.0)
    assert len(labels_by_cell) == 15
    apm_keys = {(cell_id, labels[0]) for cell_id, labels in labels_by_cell.items()}
    assert len(apm_keys) == 15
    distinct_apm = {cell.apm_cut for cell in lock.cells}
    assert len({f"{cut:.12g}" for cut in distinct_apm}) == len(distinct_apm)
    distinct_avqds = {cell.avqds_cut for cell in lock.cells}
    assert len({f"{cut:.12g}" for cut in distinct_avqds}) == len(distinct_avqds)


def test_declared_cuts_and_adoption_match_the_handoff_table(
    lock: final.MatchedCellLock,
) -> None:
    expected = {
        ("weakslow", "delta_theta_5e-5"): (4.572473708273e-11, 1.524157902758e-11, "pair_matched_comparison"),
        ("weakslow", "state_motion_1e-3_plus_parameter_5e-5"): (1.524157902758e-11, 1.524157902758e-11, "pair_matched_comparison"),
        ("weakslow", "state_motion_5e-5"): (1.371742112482e-10, 1.693508780842e-12, "pair_matched_comparison"),
        ("fastweak", "delta_theta_5e-5"): (3e-7, 3e-6, "individual_response_target"),
        ("fastweak", "state_motion_1e-3_plus_parameter_5e-5"): (9e-6, 2.7e-5, "pair_matched_comparison"),
        ("fastweak", "state_motion_5e-5"): (3e-6, 3e-6, "individual_response_target"),
        ("midres", "delta_theta_5e-5"): (3e-6, 3e-6, "individual_response_target"),
        ("midres", "state_motion_5e-5"): (3e-6, 3e-6, "individual_response_target"),
        ("slowstrong", "delta_theta_5e-5"): (3e-6, 3e-6, "individual_response_target"),
        ("slowstrong", "state_motion_5e-5"): (5.080526342527e-12, 3e-6, "pair_matched_comparison"),
        ("strongfast", "delta_theta_5e-5"): (3e-6, 1.371742112482e-10, "pair_matched_comparison"),
        ("strongfast", "state_motion_1e-3_plus_parameter_5e-5"): (9e-6, 9e-6, "pair_matched_comparison"),
        ("fastfast", "delta_theta_5e-5"): (3e-6, 1e-6, "individual_response_target"),
        ("fastfast", "state_motion_1e-3_plus_parameter_5e-5"): (9e-6, 9e-6, "individual_response_target"),
        ("fastfast", "state_motion_5e-5"): (3e-6, 3e-7, "individual_response_target"),
    }
    observed = {
        (cell.drive, cell.controller): (cell.apm_cut, cell.avqds_cut, cell.adoption_status)
        for cell in lock.cells
    }
    assert set(observed) == set(expected)
    for key, (apm, avqds, status) in expected.items():
        assert observed[key][0] == pytest.approx(apm, rel=1.0e-12, abs=0.0)
        assert observed[key][1] == pytest.approx(avqds, rel=1.0e-12, abs=0.0)
        assert observed[key][2] == status
    assert sum(s == "pair_matched_comparison" for _, _, s in observed.values()) == 7


# --- anti-fabrication --------------------------------------------------------


def test_seed_is_pinned_to_the_regime_artifact(payload: dict) -> None:
    seed = payload["seed"]
    assert seed["seed_id"] == "hh_snake_nph1"
    assert seed["artifact_json"] == runs.REGIMES["hh_snake_nph1"].seed_path
    assert seed["sha256"] == SEED_SHA256
    assert seed["n_ph_max"] == 1


def test_per_cell_evidence_is_fully_hydrated(payload: dict) -> None:
    for row in payload["cells"]:
        assert len(row["tuple_id"]) == 64
        assert set(row["metrics"]) == {"apm", "avqds"}
        assert row["paper_status"]["apm"] == "qualified"
        assert row["paper_status"]["avqds"] == "qualified"
        assert row["paper_status"]["qualification_basis"] == row["adoption_status"]
        assert row["source_artifacts"]["apm"]["stitched_point_count"] == 501
        assert row["source_artifacts"]["avqds"]["stitched_point_count"] == 501
        assert row["source_artifacts"]["apm"]["stitched_time_span"] == [0.0, 20.0]
        assert row["source_artifacts"]["avqds"]["stitched_time_span"] == [0.0, 20.0]
        assert all(value is not None for value in _leaves(row["source_artifacts"]))
        assert all(value is not None for value in _leaves(row["run_lock"]))


def test_external_artifacts_are_sha_bound_and_resolved(payload: dict) -> None:
    external = payload["external_artifacts"]
    preview = external["promotion_preview"]
    assert preview["sha256"] == "40eb31245d369ed2fa335860f832d578e69bc9151c73c91b9457321d4b474fdf"
    assert preview["resolution_status"] == "RESOLVED"
    ledger = external["ledger"]
    assert ledger["sha256"] == "830370e43b26a818f92d36d571d1aae4cb7a5475b8df61652dba9e86a5b1b9e6"
    assert ledger["record_count"] == 371
    assert ledger["resolution_status"] == "RESOLVED"
    pdf = external["results_pdf"]
    assert pdf["resolution_status"] == "RESOLVED"
    assert pdf["currency_claim"] == "current_through_coverage_cluster_9718010"
    assert pdf["sha256"] == (
        "f5e1666a567a7abe3acea4609ce57a3930038f01749cef02ff411dc65151a2e6"
    )
    assert pdf["path"].endswith("paper_ii_full_factorial_results_refresh_20260901.pdf")
    assert external["final_ingestion_receipt"]["resolution_status"] == "RESOLVED"
    assert external["report_json"]["resolution_status"] == "RESOLVED"
    assert external["report_refresh_resolution"]["resolution_status"] == "RESOLVED"


def test_report_refresh_block_is_resolved(payload: dict) -> None:
    blocks = payload["blocks"]
    assert blocks["active"] == []
    assert blocks["superseded_sha256"] == (
        "84c67ea56fd6267b1fb772178f96d0f5025556ad3b64412bb65edc673fc29a11"
    )
    assert not (ROOT / blocks["supersedes"]).exists()
    assert "verified 371-row post-coverage ledger" in payload["adoption_note"]


# --- route binding -----------------------------------------------------------


def test_route_binding_resolves_all_three_routes(lock: final.MatchedCellLock) -> None:
    assert lock.selector_routes == ROUTES
    bound = final.bind_selector_routes(lock.selector_routes)
    assert bound == ROUTES
    for route_id, arm_id in bound.items():
        flags = runs.ARMS[arm_id].flags
        assert flags[: len(runs.EXCHANGE.flags)] == runs.EXCHANGE.flags
        assert ("--exchange-route-bundle", route_id) in tuple(zip(flags, flags[1:]))
        assert not runs.ARMS[arm_id].is_comparator


def test_route_binding_fails_closed_on_an_unknown_route() -> None:
    with pytest.raises(final.UnboundSelectorRouteError):
        final.bind_selector_routes({"paper_ii_exchange_not_a_route_v9": "exchange"})
    with pytest.raises(final.UnboundSelectorRouteError):
        final.bind_selector_routes({"paper_ii_exchange_q_compiled_v1": "exchange"})
    with pytest.raises(final.UnboundSelectorRouteError):
        final.bind_selector_routes({"paper_ii_exchange_q_compiled_v1": "no_such_arm"})
    with pytest.raises(final.UnboundSelectorRouteError):
        final.bind_selector_routes(
            {"paper_ii_exchange_q_compiled_v1": "exchange_gamma_compiled"}
        )


# --- actions -----------------------------------------------------------------


def test_build_actions_yields_forty_five_apm_rows_and_no_comparators(
    lock: final.MatchedCellLock, tmp_path: Path
) -> None:
    rows = final.build_actions(lock, output_root=tmp_path)
    assert len(rows) == 45
    assert all(row["method"] == "apm" for row in rows)
    assert sum(row["arm"] == "avqds" for row in rows) == 0
    assert sum("--dynamics-policy" in row["runner_argv"] for row in rows) == 0
    ids = [row["action_id"] for row in rows]
    assert len(set(ids)) == 45
    assert len({row["run_id"] for row in rows}) == 45
    by_cell: dict[str, set[str]] = {}
    for row in rows:
        by_cell.setdefault(row["cell_id"], set()).add(row["exchange_route_bundle"])
    assert len(by_cell) == 15
    assert all(routes == set(ROUTES) for routes in by_cell.values())
    for row in rows:
        assert row["comparator"]["reuse"] == "locked_historical_avqds_15_cells"
        assert row["comparator"]["action"] is None
        assert row["comparator"]["arm"] == "avqds"


def test_every_action_parses_with_the_live_runner_parser(
    lock: final.MatchedCellLock, tmp_path: Path
) -> None:
    parser = _build_parser()
    cells = {cell.cell_id: cell for cell in lock.cells}
    for row in final.build_actions(lock, output_root=tmp_path):
        argv = list(row["runner_argv"])
        parsed = parser.parse_args(argv)
        cell = cells[row["cell_id"]]
        assert parsed.insertion_l2_cut == pytest.approx(cell.apm_cut, rel=1.0e-12, abs=0.0)
        assert parsed.exchange_route_bundle == row["exchange_route_bundle"]
        assert "--resume-from-run-json" not in argv
        assert parsed.resume_from_run_json is None
        pairs = set(zip(argv, argv[1:]))
        assert ("--t-final", "20.0") in pairs
        assert ("--num-times", "501") in pairs
        assert parsed.record_statevector is True
        assert parsed.artifact_json == runs.REGIMES["hh_snake_nph1"].seed_path
        assert parsed.output_json == row["trajectory_json"]
        assert parsed.code_revision == row["code_revision"]
        assert parsed.support_patch_scoring_workers == final.CAMPAIGN_SCORING_WORKERS == 4
        assert argv.count("--support-patch-scoring-workers") == 1
        assert len(row["exchange_route_contract_sha256"]) == 64
        assert row["exchange_route_contract"]["bundle_id"] == row["exchange_route_bundle"]
        assert row["matched_cell_lock_sha256"] == lock.sha256
        assert row["seed_sha256"] == SEED_SHA256
        assert row["source_package_sha256"] is None
        assert row["run_id"].endswith(f"_cut{cell.apm_cut_label}")
        assert row["runner_module"] == runs.RUNNER_MODULE


def test_main_writes_the_jsonl_and_prints_a_prepared_summary(
    tmp_path: Path, capsys
) -> None:
    actions = tmp_path / "pkg" / "actions.jsonl"
    code = final.main(
        [
            "--output-root", str(tmp_path / "raw"),
            "--actions-jsonl", str(actions),
        ]
    )
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "PREPARED_NOT_SUBMITTED"
    assert summary["campaign_id"] == "paper_ii_final_selector_comparison_v1"
    assert summary["action_count"] == 45
    assert summary["avqds_action_count"] == 0
    assert summary["comparator_actions"] == 0
    assert summary["comparator_reuse"] == "locked_historical_avqds_15_cells"
    assert summary["lock"]["path"] == str(final.LOCK_JSON)
    assert len(summary["lock"]["sha256"]) == 64
    assert summary["code_revision"] not in ("", None)
    assert summary["seed"]["sha256"] == SEED_SHA256
    assert summary["selector_routes"] == ROUTES
    assert summary["blocks"] == []
    assert Path(summary["actions_jsonl"]["path"]) == actions
    rows = [json.loads(line) for line in actions.read_text().splitlines() if line]
    assert len(rows) == 45
    assert len({row["action_id"] for row in rows}) == 45
    assert all(row["trajectory_json"].startswith(str(tmp_path / "raw")) for row in rows)


def test_default_actions_path_sits_under_the_output_root(
    tmp_path: Path, capsys
) -> None:
    assert final.main(["--output-root", str(tmp_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    expected = tmp_path / final.CAMPAIGN_ID / "actions.jsonl"
    assert Path(summary["actions_jsonl"]["path"]) == expected
    assert expected.is_file()


def test_lock_drift_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(final.LOCK_JSON.read_text(encoding="utf-8"))
    short = dict(payload, cells=payload["cells"][:-1])
    target = tmp_path / "short.json"
    target.write_text(json.dumps(short), encoding="utf-8")
    with pytest.raises(final.MatchedCellLockError):
        final.load_lock(target)
    blocked = copy.deepcopy(payload)
    blocked["blocks"]["active"] = ["further_campaign_submission"]
    target.write_text(json.dumps(blocked), encoding="utf-8")
    with pytest.raises(final.MatchedCellLockError, match="active submission block"):
        final.load_lock(target)
    unresolved = copy.deepcopy(payload)
    unresolved["external_artifacts"]["promotion_preview"]["resolution_status"] = "PENDING"
    target.write_text(json.dumps(unresolved), encoding="utf-8")
    with pytest.raises(final.MatchedCellLockError, match="promotion_preview: unresolved"):
        final.load_lock(target)
    duplicated = dict(payload, cells=payload["cells"][:-1] + [payload["cells"][0]])
    target.write_text(json.dumps(duplicated), encoding="utf-8")
    with pytest.raises(final.MatchedCellLockError):
        final.load_lock(target)
    pinned = dict(payload, code_revision="deadbeef")
    target.write_text(json.dumps(pinned), encoding="utf-8")
    with pytest.raises(final.MatchedCellLockError):
        final.load_lock(target)


@pytest.mark.parametrize(
    "injected",
    [
        ["--artifact-json", "other.json"],
        ["--t-final", "5"],
        ["--num-times", "126"],
        ["--times", "0,1"],
        ["--t-initial", "1"],
        ["--insertion-gate-mode", "residual_ratio"],
        ["--debt-policy", "insertion_only"],
        ["--no-exchange-deletions"],
        ["--code-revision", "fake-sha"],
        ["--integrator", "rk4"],
        ["--drive-A", "9"],
        ["--avqds-delta-theta-max", "1"],
    ],
)
def test_locked_campaign_rejects_all_generic_extra_flag_overrides(
    tmp_path: Path, payload: dict, injected: list[str]
) -> None:
    mutated = copy.deepcopy(payload)
    mutated["registry_bindings"]["extra_flags"] = [
        "--record-statevector",
        *injected,
    ]
    target = tmp_path / "mutated.json"
    target.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(final.MatchedCellLockError, match="permits only"):
        final.load_lock(target)
