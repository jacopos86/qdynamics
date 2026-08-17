from __future__ import annotations

import copy
from contextlib import nullcontext
import importlib.util
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.static_adapt.ra_adapt.adaptive_phase_shortlist import (
    AdaptivePhaseCandidateScore,
    adaptive_phase_record_id,
    select_adaptive_phase_shortlist,
)
RUNNER_PATH = (
    REPO_ROOT
    / "chtc/paper_i_ra_adapt_repair_20260727/"
    "run_local_paper_i_ra_all6_adaptive_shortlist_append_then_plateau_k50_"
    "overnight_20260816.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "paper_i_ra_all6_adaptive_overnight", RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pair_launch_observation(runner, batch):
    contract = runner.pair_launch_capacity_contract(batch)
    return {
        **contract,
        "available_memory_bytes": contract["required_available_memory_bytes"],
        "free_disk_bytes": contract["required_free_disk_bytes"],
        "launch_available_memory_bytes": contract["required_available_memory_bytes"],
        "launch_free_disk_bytes": contract["required_free_disk_bytes"],
        "launch_ready": True,
        "elapsed_wait_seconds": 0.0,
        "status": "ready",
        "ready_after_bound": False,
        "capacity_kind": "fresh_pair_launch_recheck",
        "physical_memory_bytes": contract["required_physical_memory_bytes"],
    }


def _serial_launch_observation(runner, cell):
    floor = runner.strict_archive.regime_launch_capacity_floor(
        regime_id=cell.regime_id,
        nph=cell.nph,
    )
    required_disk = max(
        runner.LAUNCH_FREE_DISK_BYTES,
        int(floor["minimum_free_bytes"]),
    )
    return {
        "available_memory_bytes": runner.LAUNCH_AVAILABLE_MEMORY_BYTES,
        "free_disk_bytes": required_disk,
        "launch_available_memory_bytes": runner.LAUNCH_AVAILABLE_MEMORY_BYTES,
        "launch_free_disk_bytes": required_disk,
        "launch_ready": True,
        "elapsed_wait_seconds": 0.0,
        "status": "ready",
        "ready_after_bound": False,
        "capacity_kind": "per_regime_cell_launch",
        "execution_id": cell.execution_id,
        "regime_capacity_floor": floor,
    }


def _monitor_worker(runner, cell, **overrides):
    payload = {
        "schema": runner.WORKER_SCHEMA,
        "status": "passed_k50",
        "campaign_id": runner.CAMPAIGN_ID,
        "execution_id": cell.execution_id,
        "manifest_sha256": "1" * 64,
        "artifact_inventory": [
            {
                "path": f"runs/{cell.execution_id}/execution_manifest.json",
                "sha256": "2" * 64,
                "size_bytes": 1,
            }
        ],
        "submission_authorized": False,
        "paper_adoption_authorized": False,
        "paper_evidence_adoption_authorized": False,
    }
    payload.update(overrides)
    return runner.digested(payload)


def test_direct_runner_has_no_standalone_canary_or_conditional_grant_surface() -> None:
    runner = _load_runner()
    assert not hasattr(runner, "CONDITIONAL_GRANT_PATH")
    assert "canary_runner" not in inspect.signature(
        runner.validate_terminal_matrix
    ).parameters
    assert "canary_runner" not in inspect.signature(runner.run_campaign).parameters
    parser_source = inspect.getsource(runner.main)
    assert "--prepare-conditional-grant" not in parser_source
    assert "--authorize" in parser_source


def test_exact_append_block_precedes_the_exact_plateau_block() -> None:
    runner = _load_runner()
    assert runner.CAMPAIGN_ID.endswith("_v2")
    assert runner.AUTHORITY_DIR.name.endswith("_v2_authority")
    assert runner.RUNTIME_ROOT.name.endswith("_v2")
    expected_regimes = [
        ("weak_weak", 3),
        ("intermediate_weak", 3),
        ("strong_weak_u8", 3),
        ("weak_strong", 7),
        ("intermediate_strong", 7),
        ("strong_strong_u8", 7),
    ]

    assert len(runner.CELL_SPECS) == 12
    assert [
        (cell.regime_id, cell.nph, cell.insertion_policy)
        for cell in runner.CELL_SPECS[:6]
    ] == [
        (regime, nph, "append_only") for regime, nph in expected_regimes
    ]
    assert [
        (cell.regime_id, cell.nph, cell.insertion_policy)
        for cell in runner.CELL_SPECS[6:]
    ] == [
        (regime, nph, "plateau_commutation")
        for regime, nph in expected_regimes
    ]
    assert [cell.ordinal for cell in runner.CELL_SPECS] == list(range(1, 13))
    assert all(cell.horizon == 50 for cell in runner.CELL_SPECS)


def test_capacity_batches_preserve_exact_canonical_order_within_each_block() -> None:
    runner = _load_runner()
    expected = [
        ("append", ("weak_weak", "intermediate_weak")),
        ("append", ("strong_weak_u8", "weak_strong")),
        ("append", ("intermediate_strong", "strong_strong_u8")),
        ("plateau", ("weak_weak", "intermediate_weak")),
        ("plateau", ("strong_weak_u8", "weak_strong")),
        ("plateau", ("intermediate_strong", "strong_strong_u8")),
    ]

    assert runner.MAXIMUM_CONCURRENCY == 2
    assert [
        (
            batch.block,
            tuple(
                runner._cell_by_execution_id(execution_id).regime_id
                for execution_id in batch.execution_ids
            ),
        )
        for batch in runner.BATCH_SPECS
    ] == expected


def test_missing_all_phase_adaptive_core_seam_fails_closed() -> None:
    runner = _load_runner()

    if not runner.core_interface_available():
        with pytest.raises(runner.RunnerError, match="all-phase adaptive core"):
            runner.build_plan()


def test_runner_binds_exact_gradient_only_all_phase_route_identity() -> None:
    runner = _load_runner()
    expected = (
        "position_records_gradient_only_adaptive_shortlist_phase123_adaptive_v1"
    )
    assert runner.EXPECTED_CORE_ROUTE_VARIANT == expected
    assert getattr(runner._load_core_module(), runner.CORE_ROUTE_CONSTANT) == expected
    assert runner.CORE_ROUTE_CONSTANT == (
        "PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1"
    )
    assert runner.CORE_REQUEST_BUILDER == (
        "build_paper_i_ra_all_phase_position_adaptive_request"
    )
    assert {cell.route_variant for cell in runner.CELL_SPECS} == {expected}


def test_runner_materializes_exact_position_record_phase0_contract() -> None:
    runner = _load_runner()
    module = runner._load_core_module()
    cell = runner.CELL_SPECS[0]
    problem = getattr(module, runner.CORE_PROBLEM_BUILDER)(cell.regime_id)
    request = getattr(module, runner.CORE_REQUEST_BUILDER)(
        insertion_policy=cell.insertion_policy,
        maximum_controller_rounds=cell.horizon,
    )
    protocol = module.materialize_paper_i_ra_semantic_protocol(problem, request)
    native = protocol.route_contract["native_semantic_contract"]
    assert native["route_variant"] == runner.EXPECTED_CORE_ROUTE_VARIANT
    assert native["phase0_policy"] == {
        "population": "current_commutation_reduced_candidate_position_records_v1",
        "benefit": "absolute_position_record_gradient_v1",
        "fubini_study_metric": "off",
        "qiskit_compile": "off",
        "graph_proxy_cost": "off",
        "score": "absolute_position_record_gradient_v1",
        "shortlist": "phase0_active_score_effective_competition_shortlist_v2",
        "adaptive_shadow_receipt": False,
        "placement_activation": (
            "append_record_when_closed_full_commutation_reduced_records_when_open_v1"
        ),
        "generator_level_reexpansion_after_phase0": False,
    }
    assert runner._protocol_binding(cell)["route_variant"] == (
        runner.EXPECTED_CORE_ROUTE_VARIANT
    )


def test_runner_reporting_phase0_gate_accepts_position_and_rejects_generator_first() -> None:
    runner = _load_runner()
    module = runner._load_core_module()
    position = module.build_semantic_position_phase0_receipt(
        [
            {
                "domain_record_id": "g0@0",
                "generator_id": "g0",
                "pool_index": 0,
                "pool_label": "G0",
                "insertion_position": 0,
                "position_class": "append",
                "gradient_signed": 0.5,
                "graph_proxy_denominator": 1.0,
            },
            {
                "domain_record_id": "g1@1",
                "generator_id": "g1",
                "pool_index": 1,
                "pool_label": "G1",
                "insertion_position": 1,
                "position_class": "interior",
                "gradient_signed": 0.49,
                "graph_proxy_denominator": 1.0,
            },
        ],
        estimator_event_ids=["gradient:0", "gradient:1"],
        route_variant=module.PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_V1,
    )
    generator_first = module.build_semantic_gradient_adaptive_phase0_receipt(
        available_indices=[0],
        gradients=[0.5],
        pool_labels=["G0"],
        estimator_event_ids=["gradient:0"],
        route_variant=module.PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1,
    )
    projection_fields = (
        "domain_record_id",
        "generator_id",
        "pool_index",
        "pool_label",
        "insertion_position",
        "position_class",
    )

    def project(row):
        return {field: row[field] for field in projection_fields}

    population = [project(row) for row in position["population"]]
    shortlist = [project(row) for row in position["retained_records"]]
    scored_population = {
        "phase0_gradient_screen": {
            "schema": "paper_i_scored_gradient_phase0_population_v1",
            "population_count": len(population),
            "population": population,
            "ordered_population_sha256": runner.canonical_sha256(population),
            "shortlist_count": len(shortlist),
            "shortlist": shortlist,
            "ordered_shortlist_sha256": runner.canonical_sha256(shortlist),
        }
    }
    assert runner.validate_reporting_phase0_receipt(
        position,
        scored_population=scored_population,
    ) == position
    retained_ids = [
        adaptive_phase_record_id(
            generator_id=row["generator_id"],
            pool_index=row["pool_index"],
            insertion_position=row["insertion_position"],
        )
        for row in position["retained_records"]
    ]
    assert len(retained_ids) == 2
    regrouped_phase_i_ids = list(reversed(retained_ids))
    regrouped_link = runner.canonical_sha256(
        {
            "phase0_retained_record_ids": retained_ids,
            "phase_i_population_record_ids": regrouped_phase_i_ids,
        }
    )
    assert runner.validate_reporting_phase0_phase_i_link(
        position,
        {
            "phase_i": {
                "population_record_ids": regrouped_phase_i_ids,
            }
        },
        closure_round={
            "phase0_phase_i_direct_population_link_sha256": regrouped_link,
        },
    ) == regrouped_link
    detached = copy.deepcopy(scored_population)
    detached["phase0_gradient_screen"]["shortlist"] = []
    detached["phase0_gradient_screen"]["shortlist_count"] = 0
    detached["phase0_gradient_screen"]["ordered_shortlist_sha256"] = (
        runner.canonical_sha256([])
    )
    with pytest.raises(runner.RunnerError, match="position-record"):
        runner.validate_reporting_phase0_receipt(
            position,
            scored_population=detached,
        )
    with pytest.raises(runner.RunnerError, match="position-record"):
        runner.validate_reporting_phase0_receipt(
            generator_first,
            scored_population=scored_population,
        )


def test_reporting_phase0_phase_i_link_rejects_detached_live_population() -> None:
    runner = _load_runner()
    retained = [
        {
            "generator_id": "g0",
            "pool_index": 0,
            "insertion_position": 0,
        },
        {
            "generator_id": "g1",
            "pool_index": 1,
            "insertion_position": 1,
        },
    ]
    phase0 = {"retained_records": retained}
    phase0_ids = [
        adaptive_phase_record_id(
            generator_id=row["generator_id"],
            pool_index=row["pool_index"],
            insertion_position=row["insertion_position"],
        )
        for row in retained
    ]
    link_sha256 = runner.canonical_sha256(
        {
            "phase0_retained_record_ids": phase0_ids,
            "phase_i_population_record_ids": phase0_ids,
        }
    )
    phase_evidence = {
        "phase_i": {"population_record_ids": list(phase0_ids)}
    }
    closure_round = {
        "phase0_phase_i_direct_population_link_sha256": link_sha256
    }
    assert runner.validate_reporting_phase0_phase_i_link(
        phase0,
        phase_evidence,
        closure_round=closure_round,
    ) == link_sha256

    regrouped = copy.deepcopy(phase_evidence)
    regrouped["phase_i"]["population_record_ids"] = list(reversed(phase0_ids))
    regrouped_link = {
        "phase0_phase_i_direct_population_link_sha256": runner.canonical_sha256(
            {
                "phase0_retained_record_ids": phase0_ids,
                "phase_i_population_record_ids": list(reversed(phase0_ids)),
            }
        )
    }
    assert runner.validate_reporting_phase0_phase_i_link(
        phase0,
        regrouped,
        closure_round=regrouped_link,
    ) == regrouped_link["phase0_phase_i_direct_population_link_sha256"]

    invalid_phase_i_populations = (
        phase0_ids[:1],
        [*phase0_ids, "extra-record"],
        [phase0_ids[0], phase0_ids[0]],
    )
    for invalid_ids in invalid_phase_i_populations:
        with pytest.raises(runner.RunnerError, match="passed directly"):
            runner.validate_reporting_phase0_phase_i_link(
                phase0,
                {"phase_i": {"population_record_ids": invalid_ids}},
                closure_round={
                    "phase0_phase_i_direct_population_link_sha256": (
                        runner.canonical_sha256(
                            {
                                "phase0_retained_record_ids": phase0_ids,
                                "phase_i_population_record_ids": invalid_ids,
                            }
                        )
                    )
                },
            )


def test_reporting_plateau_state_requires_authenticated_boolean_for_plateau_cell() -> None:
    runner = _load_runner()
    append_cell = runner.CELL_SPECS[0]
    plateau_cell = runner.CELL_SPECS[6]
    assert runner.reporting_plateau_state(append_cell, {}) == "closed"
    assert runner.reporting_plateau_state(
        plateau_cell,
        {"insertion_commutation_plateau": {"domain_open": True}},
    ) == "open"
    assert runner.reporting_plateau_state(
        plateau_cell,
        {"insertion_commutation_plateau": {"domain_open": False}},
    ) == "closed"
    for malformed in (
        {},
        {"insertion_commutation_plateau": {}},
        {"insertion_commutation_plateau": {"domain_open": 1}},
    ):
        with pytest.raises(runner.RunnerError, match="plateau state"):
            runner.reporting_plateau_state(plateau_cell, malformed)


def test_post_science_identity_revalidates_authority_and_exact_protocol(
    monkeypatch,
) -> None:
    runner = _load_runner()
    cell = runner.CELL_SPECS[0]
    expected = {"execution_id": cell.execution_id, "protocol_sha256": "4" * 64}
    plan = {
        "sha256": "1" * 64,
        "protocol_bindings": [expected],
    }
    authorization = {"sha256": "2" * 64}
    monkeypatch.setattr(
        runner,
        "validate_authority",
        lambda **_kwargs: (plan, authorization),
    )
    monkeypatch.setattr(runner, "_protocol_binding", lambda _cell: expected)
    runner.validate_post_science_batch_identity(
        (cell,), plan=plan, authorization=authorization
    )
    monkeypatch.setattr(
        runner,
        "_protocol_binding",
        lambda _cell: {**expected, "protocol_sha256": "5" * 64},
    )
    with pytest.raises(runner.RunnerError, match="Post-science protocol"):
        runner.validate_post_science_batch_identity(
            (cell,), plan=plan, authorization=authorization
        )


def test_bounded_serial_capacity_returns_blocked_instead_of_idling() -> None:
    runner = _load_runner()
    ticks = iter((0.0, 0.0, 300.0))

    observed = runner.wait_for_capacity(
        maximum_wait_seconds=300.0,
        clock=lambda: next(ticks),
        sleeper=lambda _seconds: None,
        memory_supplier=lambda: 0,
        disk_supplier=lambda: 0,
    )

    assert observed["status"] == "blocked_capacity"
    assert observed["elapsed_wait_seconds"] == 300.0


def test_pair_capacity_uses_empirical_ram_and_disk_anchors_at_exact_boundaries() -> None:
    runner = _load_runner()
    batch = runner.BATCH_SPECS[2]
    contract = runner.pair_launch_capacity_contract(batch)
    cells = [
        runner._cell_by_execution_id(execution_id)
        for execution_id in batch.execution_ids
    ]
    expected_memory = (
        sum(runner.empirical_peak_rss_anchor(cell)["peak_rss_bytes"] for cell in cells)
        + runner.AVAILABLE_MEMORY_FLOOR_BYTES
    )
    guarded_disk = sum(
        (
            5
            * runner.strict_archive.regime_launch_capacity_floor(
                regime_id=cell.regime_id,
                nph=cell.nph,
            )["observed_working_disk_bytes"]
            + 3
        )
        // 4
        for cell in cells
    )
    expected_disk = (
        guarded_disk
        + runner.strict_archive.campaign_default_archive_limits()
        .archive_start_free_floor_bytes
    )
    assert contract["required_available_memory_bytes"] == expected_memory
    assert contract["required_physical_memory_bytes"] == expected_memory
    assert contract["required_free_disk_bytes"] == expected_disk

    ready = runner.wait_for_batch_capacity(
        batch,
        maximum_wait_seconds=300.0,
        clock=lambda: 0.0,
        sleeper=lambda _seconds: None,
        memory_supplier=lambda: expected_memory,
        physical_memory_supplier=lambda: expected_memory,
        disk_supplier=lambda: expected_disk,
    )
    assert ready["status"] == "ready_pair"
    assert ready["scheduling_mode"] == "pair"


@pytest.mark.parametrize("shortfall", ["memory", "disk", "physical"])
def test_pair_capacity_shortfall_is_an_explicit_serial_fallback(
    shortfall: str,
) -> None:
    runner = _load_runner()
    batch = runner.BATCH_SPECS[2]
    contract = runner.pair_launch_capacity_contract(batch)
    required_memory = contract["required_available_memory_bytes"]
    required_disk = contract["required_free_disk_bytes"]
    ticks = iter((0.0, 0.0, 300.0))
    observed = runner.wait_for_batch_capacity(
        batch,
        maximum_wait_seconds=300.0,
        clock=lambda: next(ticks),
        sleeper=lambda _seconds: None,
        memory_supplier=lambda: (
            required_memory - 1 if shortfall == "memory" else required_memory
        ),
        physical_memory_supplier=lambda: (
            required_memory - 1 if shortfall == "physical" else required_memory
        ),
        disk_supplier=lambda: (
            required_disk - 1 if shortfall == "disk" else required_disk
        ),
    )
    assert observed["status"] == "serial_capacity_fallback"
    assert observed["scheduling_mode"] == "serial_capacity_fallback"
    assert shortfall in observed["fallback_reasons"]
    if shortfall == "physical":
        assert observed["elapsed_wait_seconds"] == 0.0
    else:
        assert observed["elapsed_wait_seconds"] == 300.0


def test_pair_capacity_ready_only_after_bound_is_a_valid_audited_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "SCHEDULER_ROOT", tmp_path / "scheduler")
    batch = runner.BATCH_SPECS[0]
    contract = runner.pair_launch_capacity_contract(batch)
    memory = iter(
        (
            contract["required_available_memory_bytes"] - 1,
            contract["required_available_memory_bytes"],
        )
    )
    ticks = iter((0.0, 0.0, 301.0))
    observation = runner.wait_for_batch_capacity(
        batch,
        maximum_wait_seconds=300.0,
        clock=lambda: next(ticks),
        sleeper=lambda _seconds: None,
        memory_supplier=lambda: next(memory),
        physical_memory_supplier=lambda: contract["required_physical_memory_bytes"],
        disk_supplier=lambda: contract["required_free_disk_bytes"],
    )
    assert observation["status"] == "serial_capacity_fallback"
    assert observation["fallback_reasons"] == ["wait_bound"]
    plan = {
        "sha256": "1" * 64,
        "source_implementation_inventory_sha256": "2" * 64,
    }
    authorization = {"sha256": "3" * 64}
    decision = runner.select_batch_schedule(
        batch,
        plan=plan,
        authorization=authorization,
        capacity_waiter=lambda _batch: observation,
    )
    assert decision["scheduling_mode"] == "serial_capacity_fallback"


@pytest.mark.parametrize("mode", ["pair", "serial_capacity_fallback"])
def test_scheduler_decision_is_immutable_restart_safe_and_capacity_bound(
    monkeypatch, tmp_path: Path, mode: str
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "SCHEDULER_ROOT", tmp_path / "scheduler")
    batch = runner.BATCH_SPECS[0]
    plan = {
        "sha256": "1" * 64,
        "source_implementation_inventory_sha256": "2" * 64,
    }
    authorization = {"sha256": "3" * 64}
    contract = runner.pair_launch_capacity_contract(batch)
    observed = {
        **contract,
        "status": "ready_pair" if mode == "pair" else "serial_capacity_fallback",
        "scheduling_mode": mode,
        "available_memory_bytes": contract["required_available_memory_bytes"],
        "free_disk_bytes": contract["required_free_disk_bytes"],
        "physical_memory_bytes": (
            contract["required_physical_memory_bytes"]
            if mode == "pair"
            else contract["required_physical_memory_bytes"] - 1
        ),
        "elapsed_wait_seconds": 0.0,
        "fallback_reasons": [] if mode == "pair" else ["physical"],
    }
    if mode == "pair":
        observed.update(
            {
                "launch_available_memory_bytes": contract[
                    "required_available_memory_bytes"
                ],
                "launch_free_disk_bytes": contract["required_free_disk_bytes"],
                "launch_ready": True,
            }
        )
    calls = []

    def capacity_waiter(selected_batch):
        calls.append(selected_batch.ordinal)
        return observed

    first = runner.select_batch_schedule(
        batch,
        plan=plan,
        authorization=authorization,
        capacity_waiter=capacity_waiter,
    )
    second = runner.select_batch_schedule(
        batch,
        plan=plan,
        authorization=authorization,
        capacity_waiter=lambda _batch: (_ for _ in ()).throw(
            AssertionError("immutable scheduler decision was recomputed")
        ),
    )

    assert first == second
    assert calls == [batch.ordinal]
    assert first["scheduling_mode"] == mode
    assert first["capacity_contract"] == contract
    assert first["capacity_observation_sha256"] == runner.canonical_sha256(
        observed
    )
    assert first["plan_sha256"] == plan["sha256"]
    assert first["authorization_sha256"] == authorization["sha256"]
    extra_payload = {key: value for key, value in first.items() if key != "sha256"}
    extra_payload["unexpected_authority_claim"] = True
    with pytest.raises(runner.RunnerError, match="shape drifted"):
        runner.validate_scheduler_decision(
            batch,
            runner.digested(extra_payload),
            plan=plan,
            authorization=authorization,
        )


def test_empirical_child_rss_guard_supersedes_eight_gib_only_for_nph7_plateau() -> None:
    runner = _load_runner()
    for cell in runner.CELL_SPECS:
        expected = (
            10 * 1024**3
            if cell.nph == 7 and cell.block == "plateau"
            else 8 * 1024**3
        )
        assert runner.child_rss_limit_bytes(cell) == expected


def test_launch_capacity_observation_is_deeply_bound_to_pair_and_serial_modes() -> None:
    runner = _load_runner()
    batch = runner.BATCH_SPECS[0]
    pair_cells = tuple(
        runner._cell_by_execution_id(value) for value in batch.execution_ids
    )
    pair = _pair_launch_observation(runner, batch)
    assert runner.validate_launch_capacity_observation(
        pair_cells, batch=batch, scheduling_mode="pair", observation=pair
    ) == pair
    with pytest.raises(runner.RunnerError, match="mode contract"):
        runner.validate_launch_capacity_observation(
            pair_cells,
            batch=batch,
            scheduling_mode="pair",
            observation={
                **pair,
                "available_memory_bytes": pair["available_memory_bytes"] - 1,
            },
        )

    cell = pair_cells[0]
    floor = runner.strict_archive.regime_launch_capacity_floor(
        regime_id=cell.regime_id, nph=cell.nph
    )
    required_disk = max(
        runner.LAUNCH_FREE_DISK_BYTES, int(floor["minimum_free_bytes"])
    )
    serial = {
        "available_memory_bytes": runner.LAUNCH_AVAILABLE_MEMORY_BYTES,
        "free_disk_bytes": required_disk,
        "launch_available_memory_bytes": runner.LAUNCH_AVAILABLE_MEMORY_BYTES,
        "launch_free_disk_bytes": required_disk,
        "launch_ready": True,
        "elapsed_wait_seconds": 0.0,
        "status": "ready",
        "ready_after_bound": False,
        "capacity_kind": "per_regime_cell_launch",
        "execution_id": cell.execution_id,
        "regime_capacity_floor": floor,
    }
    assert runner.validate_launch_capacity_observation(
        (cell,),
        batch=batch,
        scheduling_mode="serial_capacity_fallback",
        observation=serial,
    ) == serial
    with pytest.raises(runner.RunnerError, match="mode contract"):
        runner.validate_launch_capacity_observation(
            (cell,),
            batch=batch,
            scheduling_mode="serial_capacity_fallback",
            observation={**serial, "free_disk_bytes": required_disk - 1},
        )


def test_pair_monitor_launches_both_children_before_poll_and_keeps_sibling_guarded(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "CELL_LOG_ROOT", tmp_path / "logs")
    cells = tuple(
        runner._cell_by_execution_id(execution_id)
        for execution_id in runner.BATCH_SPECS[2].execution_ids
    )
    launched = []
    paths = {}
    for index, cell in enumerate(cells):
        cell_paths = (
            tmp_path / f"run-{index}",
            tmp_path / f"stage-{index}",
            tmp_path / f"worker-{index}.json",
            tmp_path / f"guard-{index}.json",
        )
        paths[cell.execution_id] = cell_paths
        runner.write_json_exclusive(cell_paths[2], _monitor_worker(runner, cell))
    monkeypatch.setattr(
        runner, "cell_paths", lambda cell: paths[cell.execution_id]
    )

    class FakeChild:
        def __init__(self, pid: int, polls):
            self.pid = pid
            self._polls = list(polls)
            self.returncode = None
            self.terminated = False

        def poll(self):
            assert len(launched) == 2
            if self.returncode is not None:
                return self.returncode
            value = self._polls.pop(0)
            if value is not None:
                self.returncode = value
            return value

        def wait(self, timeout=None):
            del timeout
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def kill(self):
            self.terminated = True
            self.returncode = -9

    children = [FakeChild(101, [0]), FakeChild(102, [None, 0])]

    def popen_factory(*_args, **kwargs):
        assert kwargs["start_new_session"] is True
        assert kwargs["env"]["STATIC_ADAPT_ALLOCATED_CPUS"] == "1"
        assert kwargs["env"]["QISKIT_NUM_PROCS"] == "1"
        assert kwargs["env"]["QISKIT_PARALLEL"] == "FALSE"
        assert kwargs["env"]["RAYON_NUM_THREADS"] == "1"
        assert kwargs["stdout"] is not None
        assert kwargs["stderr"] is runner.subprocess.STDOUT
        child = children[len(launched)]
        launched.append(child)
        return child

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def memory_info(self):
            return SimpleNamespace(rss=1024)

        def children(self, recursive=True):
            assert recursive is True
            return []

    guards = runner.monitor_cells(
        cells,
        {"sha256": "a" * 64},
        batch=runner.BATCH_SPECS[2],
        scheduling_mode="pair",
        scheduler_decision_sha256="9" * 64,
        launch_capacity_observation=_pair_launch_observation(
            runner, runner.BATCH_SPECS[2]
        ),
        popen_factory=popen_factory,
        process_factory=FakeProcess,
        memory_supplier=lambda: 16 * 1024**3,
        disk_supplier=lambda: 100 * 1024**3,
        sleeper=lambda _seconds: None,
        status_writer=lambda _payload: None,
    )

    assert len(launched) == 2
    assert [guard["status"] for guard in guards] == ["passed", "passed"]
    assert {
        guard["scheduler_decision_sha256"] for guard in guards
    } == {"9" * 64}
    assert len({guard["log_file_binding"]["path"] for guard in guards}) == 2
    assert all((tmp_path / "logs" / f"{cell.execution_id}.log").is_file() for cell in cells)


def test_each_science_child_disables_internal_phase0_threadpool_concurrency() -> None:
    runner = _load_runner()
    assert runner.EXPECTED_ENV["STATIC_ADAPT_ALLOCATED_CPUS"] == "1"
    assert runner.EXPECTED_ENV["QISKIT_NUM_PROCS"] == "1"
    assert runner.EXPECTED_ENV["QISKIT_PARALLEL"] == "FALSE"
    assert runner.EXPECTED_ENV["RAYON_NUM_THREADS"] == "1"
    assert runner.PHASE_FRONTIER_RATIOS == {
        "phase_i": 0.9,
        "phase_ii": 0.9,
        "phase_iii": 0.9,
    }


def test_pair_monitor_failure_terminates_sibling_and_preserves_both_guards(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "CELL_LOG_ROOT", tmp_path / "logs")
    cells = tuple(
        runner._cell_by_execution_id(execution_id)
        for execution_id in runner.BATCH_SPECS[2].execution_ids
    )
    paths = {
        cell.execution_id: (
            tmp_path / f"run-{index}",
            tmp_path / f"stage-{index}",
            tmp_path / f"worker-{index}.json",
            tmp_path / f"guard-{index}.json",
        )
        for index, cell in enumerate(cells)
    }
    monkeypatch.setattr(
        runner, "cell_paths", lambda cell: paths[cell.execution_id]
    )
    launched = []

    class FakeChild:
        def __init__(self, pid: int, rc):
            self.pid = pid
            self.returncode = rc
            self.terminated = False

        def poll(self):
            assert len(launched) == 2
            return self.returncode

        def wait(self, timeout=None):
            del timeout
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def kill(self):
            self.terminated = True
            self.returncode = -9

    children = [FakeChild(201, 7), FakeChild(202, None)]

    def popen_factory(*_args, **_kwargs):
        child = children[len(launched)]
        launched.append(child)
        return child

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def memory_info(self):
            return SimpleNamespace(rss=1024)

        def children(self, recursive=True):
            return []

    with pytest.raises(runner.RunnerError, match="failed"):
        runner.monitor_cells(
            cells,
            {"sha256": "b" * 64},
            batch=runner.BATCH_SPECS[2],
            scheduling_mode="pair",
            scheduler_decision_sha256="8" * 64,
            launch_capacity_observation=_pair_launch_observation(
                runner, runner.BATCH_SPECS[2]
            ),
            popen_factory=popen_factory,
            process_factory=FakeProcess,
            memory_supplier=lambda: 16 * 1024**3,
            disk_supplier=lambda: 100 * 1024**3,
            sleeper=lambda _seconds: None,
            terminate_process_group=lambda child: child.terminate(),
            status_writer=lambda _payload: None,
        )
    assert children[1].terminated is True
    guards = [
        runner.load_digested(paths[cell.execution_id][3], schema=runner.GUARD_SCHEMA)
        for cell in cells
    ]
    assert guards[0]["stop_reason"] == "returncode_nonzero"
    assert guards[1]["stop_reason"] == "sibling_failed"
    assert all(guard["status"] == "failed" for guard in guards)


def test_serial_monitor_failure_never_publishes_transient_pair_status(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "CELL_LOG_ROOT", tmp_path / "logs")
    batch = runner.BATCH_SPECS[0]
    cell = runner._cell_by_execution_id(batch.execution_ids[0])
    paths = (
        tmp_path / "run",
        tmp_path / "stage",
        tmp_path / "worker.json",
        tmp_path / "guard.json",
    )
    monkeypatch.setattr(runner, "cell_paths", lambda _cell: paths)

    class FailedChild:
        pid = 211
        returncode = 7

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            del timeout
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def memory_info(self):
            return SimpleNamespace(rss=1024)

        def children(self, recursive=True):
            del recursive
            return []

    statuses = []
    with pytest.raises(runner.BatchExecutionFailed) as caught:
        runner.monitor_cells(
            (cell,),
            {"sha256": "b" * 64},
            batch=batch,
            scheduling_mode="serial_capacity_fallback",
            scheduler_decision_sha256="8" * 64,
            launch_capacity_observation=_serial_launch_observation(runner, cell),
            popen_factory=lambda *_args, **_kwargs: FailedChild(),
            process_factory=FakeProcess,
            memory_supplier=lambda: 16 * 1024**3,
            disk_supplier=lambda: 100 * 1024**3,
            sleeper=lambda _seconds: None,
            terminate_process_group=lambda child: child.terminate(),
            status_writer=lambda payload: statuses.append(payload["status"]),
        )
    assert caught.value.failure_status == "failed_campaign"
    assert "failed_pair" not in statuses
    assert statuses[-1] == "failed_campaign"


@pytest.mark.parametrize(
    "worker_failure", ["missing", "malformed", "wrong_status", "wrong_campaign"]
)
def test_pair_monitor_worker_closure_failure_contains_sibling_and_preserves_guards(
    monkeypatch, tmp_path: Path, worker_failure: str
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "CELL_LOG_ROOT", tmp_path / "logs")
    batch = runner.BATCH_SPECS[0]
    cells = tuple(
        runner._cell_by_execution_id(execution_id)
        for execution_id in batch.execution_ids
    )
    paths = {
        cell.execution_id: (
            tmp_path / f"run-{index}",
            tmp_path / f"stage-{index}",
            tmp_path / f"worker-{index}.json",
            tmp_path / f"guard-{index}.json",
        )
        for index, cell in enumerate(cells)
    }
    if worker_failure == "malformed":
        paths[cells[0].execution_id][2].write_text("{}\n", encoding="utf-8")
    elif worker_failure == "wrong_status":
        runner.write_json_exclusive(
            paths[cells[0].execution_id][2],
            _monitor_worker(runner, cells[0], status="wrong"),
        )
    elif worker_failure == "wrong_campaign":
        runner.write_json_exclusive(
            paths[cells[0].execution_id][2],
            _monitor_worker(runner, cells[0], campaign_id="wrong"),
        )
    runner.write_json_exclusive(
        paths[cells[1].execution_id][2],
        runner.digested(
            {"schema": runner.WORKER_SCHEMA, "execution_id": cells[1].execution_id}
        ),
    )
    monkeypatch.setattr(runner, "cell_paths", lambda cell: paths[cell.execution_id])
    launched = []

    class FakeChild:
        def __init__(self, pid: int, polls):
            self.pid = pid
            self._polls = list(polls)
            self.returncode = None
            self.terminated = False

        def poll(self):
            if self.returncode is not None:
                return self.returncode
            if not self._polls:
                return None
            value = self._polls.pop(0)
            if value is not None:
                self.returncode = value
            return value

        def wait(self, timeout=None):
            del timeout
            if self.returncode is None:
                self.returncode = -15 if self.terminated else 0
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def kill(self):
            self.terminated = True
            self.returncode = -9

    children = [FakeChild(301, [0]), FakeChild(302, [None])]

    def popen_factory(*_args, **_kwargs):
        child = children[len(launched)]
        launched.append(child)
        return child

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def memory_info(self):
            return SimpleNamespace(rss=1024)

        def children(self, recursive=True):
            return []

    with pytest.raises(runner.RunnerError, match="worker receipt"):
        runner.monitor_cells(
            cells,
            {"sha256": "7" * 64},
            batch=batch,
            scheduling_mode="pair",
            scheduler_decision_sha256="6" * 64,
            launch_capacity_observation=_pair_launch_observation(runner, batch),
            popen_factory=popen_factory,
            process_factory=FakeProcess,
            memory_supplier=lambda: 16 * 1024**3,
            disk_supplier=lambda: 100 * 1024**3,
            sleeper=lambda _seconds: None,
            terminate_process_group=lambda child: child.terminate(),
            status_writer=lambda _payload: None,
        )
    assert children[1].terminated is True
    guards = [
        runner.load_digested(paths[cell.execution_id][3], schema=runner.GUARD_SCHEMA)
        for cell in cells
    ]
    assert guards[0]["status"] == "failed"
    expected_reason = (
        f"{worker_failure}_worker_receipt"
        if worker_failure in {"missing", "malformed"}
        else "malformed_worker_receipt"
    )
    assert guards[0]["stop_reason"] == expected_reason
    assert guards[1]["status"] == "failed"
    assert guards[1]["stop_reason"] == "sibling_closure_validation_failed"


def test_pair_monitor_second_launch_failure_contains_first_child(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "CELL_LOG_ROOT", tmp_path / "logs")
    batch = runner.BATCH_SPECS[0]
    cells = tuple(
        runner._cell_by_execution_id(execution_id)
        for execution_id in batch.execution_ids
    )
    paths = {
        cell.execution_id: (
            tmp_path / f"run-{index}",
            tmp_path / f"stage-{index}",
            tmp_path / f"worker-{index}.json",
            tmp_path / f"guard-{index}.json",
        )
        for index, cell in enumerate(cells)
    }
    monkeypatch.setattr(runner, "cell_paths", lambda cell: paths[cell.execution_id])

    class FirstChild:
        pid = 401
        returncode = None
        terminated = False

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            del timeout
            if self.returncode is None:
                self.returncode = -15
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def kill(self):
            self.terminated = True
            self.returncode = -9

    first = FirstChild()
    launches = 0

    def popen_factory(*_args, **_kwargs):
        nonlocal launches
        launches += 1
        if launches == 2:
            raise OSError("second launch failed")
        return first

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

    with pytest.raises(runner.RunnerError, match="launch"):
        runner.monitor_cells(
            cells,
            {"sha256": "5" * 64},
            batch=batch,
            scheduling_mode="pair",
            scheduler_decision_sha256="4" * 64,
            launch_capacity_observation=_pair_launch_observation(runner, batch),
            popen_factory=popen_factory,
            process_factory=FakeProcess,
            memory_supplier=lambda: 16 * 1024**3,
            disk_supplier=lambda: 100 * 1024**3,
            sleeper=lambda _seconds: None,
            terminate_process_group=lambda child: child.terminate(),
            status_writer=lambda _payload: None,
        )
    assert first.terminated is True
    guard = runner.load_digested(paths[cells[0].execution_id][3], schema=runner.GUARD_SCHEMA)
    assert guard["status"] == "failed"
    assert guard["stop_reason"] == "batch_launch_failed"


def _patch_overnight_authority(monkeypatch, runner, tmp_path: Path, plan) -> None:
    authority = tmp_path / "authority"
    authority.mkdir()
    monkeypatch.setattr(runner, "AUTHORITY_DIR", authority)
    monkeypatch.setattr(runner, "PLAN_PATH", authority / "plan.json")
    monkeypatch.setattr(runner, "AUTHORIZATION_PATH", authority / "authorization.json")
    runner.write_json_exclusive(runner.PLAN_PATH, plan)
    monkeypatch.setattr(runner, "validate_plan", lambda **_kwargs: plan)


def test_plan_and_direct_execution_authorization_are_separate(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    inventory = "a" * 64
    plan = runner.digested(
        {
            "schema": runner.PLAN_SCHEMA,
            "campaign_id": runner.CAMPAIGN_ID,
            "source_implementation_inventory_sha256": inventory,
            "runner": {"sha256": "b" * 64},
            "canonical_cell_order": [cell.execution_id for cell in runner.CELL_SPECS],
            "execution_path_canary": {
                "execution_id": runner.CELL_SPECS[0].execution_id,
                "accepted_round": 1,
                "continues_same_trajectory_to_k50": True,
                "separate_scientific_trajectory": False,
            },
        }
    )
    _patch_overnight_authority(monkeypatch, runner, tmp_path, plan)
    assert not runner.AUTHORIZATION_PATH.exists()

    authorization = runner.authorize()

    assert authorization["plan_sha256"] == plan["sha256"]
    assert authorization["execution_authorized"] is True
    assert authorization["execution_path_canary"] == plan["execution_path_canary"]
    with pytest.raises(runner.RunnerError, match="already exists"):
        runner.authorize()


def test_direct_authority_rejects_inventory_or_canary_observation_drift(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    inventory = "b" * 64
    plan = runner.digested(
        {
            "schema": runner.PLAN_SCHEMA,
            "campaign_id": runner.CAMPAIGN_ID,
            "source_implementation_inventory_sha256": inventory,
            "runner": {"sha256": "c" * 64},
            "canonical_cell_order": [cell.execution_id for cell in runner.CELL_SPECS],
            "execution_path_canary": {
                "execution_id": runner.CELL_SPECS[0].execution_id,
                "accepted_round": 1,
                "continues_same_trajectory_to_k50": True,
                "separate_scientific_trajectory": False,
            },
        }
    )
    _patch_overnight_authority(monkeypatch, runner, tmp_path, plan)
    authorization = runner.authorize()
    monkeypatch.setattr(
        runner,
        "load_digested",
        lambda *_args, **_kwargs: {**authorization, "source_implementation_inventory_sha256": "d" * 64},
    )
    with pytest.raises(runner.RunnerError, match="authorization drifted"):
        runner.validate_authority()
    extra = runner.digested(
        {
            **{
                key: value
                for key, value in authorization.items()
                if key != "sha256"
            },
            "unexpected_authority_claim": True,
        }
    )
    monkeypatch.setattr(runner, "load_digested", lambda *_args, **_kwargs: extra)
    with pytest.raises(runner.RunnerError, match="shape drifted"):
        runner.validate_authority()


def test_plan_validator_rejects_rehashed_unknown_fields(
    monkeypatch,
) -> None:
    runner = _load_runner()
    inventory = {"sha256": "a" * 64, "source_count": 1}
    expected_capacity = {
        "maximum_wait_seconds": runner.CAPACITY_WAIT_SECONDS,
        "launch_available_memory_bytes": runner.LAUNCH_AVAILABLE_MEMORY_BYTES,
        "launch_free_disk_bytes": runner.LAUNCH_FREE_DISK_BYTES,
        "child_rss_limit_bytes": runner.CHILD_RSS_LIMIT_BYTES,
        "runtime_available_memory_floor_bytes": runner.AVAILABLE_MEMORY_FLOOR_BYTES,
        "runtime_free_disk_floor_bytes": runner.FREE_DISK_FLOOR_BYTES,
        "nph7_plateau_child_rss_limit_bytes": runner.NPH7_PLATEAU_CHILD_RSS_LIMIT_BYTES,
        "host_physical_memory_evidence": dict(runner.HOST_PHYSICAL_MEMORY_EVIDENCE),
        "pair_launch_capacity_contracts": [
            runner.pair_launch_capacity_contract(batch) for batch in runner.BATCH_SPECS
        ],
    }
    plan = runner.digested(
        {
            "schema": runner.PLAN_SCHEMA,
            "created_at": "2026-08-17T00:00:00Z",
            "campaign_id": runner.CAMPAIGN_ID,
            "run_class": "local_diagnostic_non_adopted",
            "target_horizon": runner.TARGET_HORIZON,
            "block_order": ["append", "plateau"],
            "canonical_cell_order": [cell.execution_id for cell in runner.CELL_SPECS],
            "deterministic_launch_order": [
                value for batch in runner.BATCH_SPECS for value in batch.execution_ids
            ],
            "append_block_execution_ids": [
                cell.execution_id for cell in runner.CELL_SPECS[:6]
            ],
            "plateau_block_execution_ids": [
                cell.execution_id for cell in runner.CELL_SPECS[6:]
            ],
            "deterministic_batches": [
                runner._batch_payload(batch) for batch in runner.BATCH_SPECS
            ],
            "cells": [runner.asdict(cell) for cell in runner.CELL_SPECS],
            "protocol_bindings": [],
            "source_implementation_inventory_sha256": inventory["sha256"],
            "source_implementation_file_count": inventory["source_count"],
            "runner": runner.file_binding(runner.RUNNER_PATH),
            "runner_runtime_dependencies": runner._runtime_dependencies(),
            "optimizer": {
                "name": "powell", "xtol": 1.0e-4, "ftol": 1.0e-8,
                "maxiter": 200, "maxfev": None,
            },
            "seeds": {"adapt": 7, "transpiler": 7},
            "frontier_ratios": dict(runner.PHASE_FRONTIER_RATIOS),
            "shortlist_maxima": dict(runner.SHORTLIST_MAXIMA),
            "maximum_concurrency": runner.MAXIMUM_CONCURRENCY,
            "serial_capacity_fallback_authorized": True,
            "silent_serial_fallback_authorized": False,
            "append_block_must_close_before_plateau": True,
            "execution_path_canary": {
                "execution_id": runner.CELL_SPECS[0].execution_id,
                "accepted_round": 1,
                "continues_same_trajectory_to_k50": True,
                "separate_scientific_trajectory": False,
            },
            "capacity": expected_capacity,
            "per_cell_storage_lifecycle": runner._storage_lifecycle_contract(),
            "runtime_environment": dict(runner.EXPECTED_ENV),
            "execution_authorized": False,
            "archive_rotation_authorized": False,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    monkeypatch.setattr(runner, "core_interface_available", lambda: True)
    monkeypatch.setattr(runner, "_cells", lambda: runner.CELL_SPECS)
    monkeypatch.setattr(
        runner,
        "_load_core_module",
        lambda: SimpleNamespace(
            semantic_closure_source_implementation_inventory=lambda: inventory
        ),
    )
    monkeypatch.setattr(runner, "load_digested", lambda *_args, **_kwargs: plan)
    assert runner.validate_plan(recompute_protocols=False) == plan
    bad = runner.digested(
        {
            **{key: value for key, value in plan.items() if key != "sha256"},
            "unexpected_authority_claim": True,
        }
    )
    monkeypatch.setattr(runner, "load_digested", lambda *_args, **_kwargs: bad)
    with pytest.raises(runner.RunnerError, match="shape drifted"):
        runner.validate_plan(recompute_protocols=False)


def test_plateau_block_gate_requires_six_validated_append_closures() -> None:
    runner = _load_runner()
    append_ids = [
        execution_id
        for batch in runner.BATCH_SPECS[:3]
        for execution_id in batch.execution_ids
    ]

    with pytest.raises(runner.RunnerError, match="append block"):
        runner.assert_append_block_closed(append_ids[:5])
    runner.assert_append_block_closed(append_ids)


def test_cell_output_classifier_skips_only_complete_closures_and_rejects_partial(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    cell = runner.CELL_SPECS[0]
    paths = (
        tmp_path / "run",
        tmp_path / "stage",
        tmp_path / "worker.json",
        tmp_path / "guard.json",
    )
    monkeypatch.setattr(runner, "cell_paths", lambda _cell: paths)
    assert runner.classify_cell_output(cell) == "pristine"

    paths[0].mkdir()
    paths[2].touch()
    paths[3].touch()
    assert runner.classify_cell_output(cell) == "closed"

    paths[1].mkdir()
    with pytest.raises(runner.RunnerError, match="partial output"):
        runner.classify_cell_output(cell)


def test_terminal_matrix_recomputes_json_csv_and_markdown(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "REPORT_JSON", tmp_path / "comparison.json")
    monkeypatch.setattr(runner, "REPORT_CSV", tmp_path / "comparison.csv")
    monkeypatch.setattr(runner, "REPORT_MD", tmp_path / "comparison.md")
    monkeypatch.setattr(runner, "TERMINAL_PATH", tmp_path / "terminal.json")
    plan = {
        "sha256": "1" * 64,
        "source_implementation_inventory_sha256": "2" * 64,
    }
    authorization = {
        "sha256": "3" * 64,
    }
    canary_observation = {
        "execution_id": runner.CELL_SPECS[0].execution_id,
        "accepted_round": 1,
        "round_row_sha256": "4" * 64,
        "continues_same_trajectory_to_k50": True,
        "separate_scientific_trajectory": False,
    }
    comparison = runner.digested(
        {
            "schema": runner.REPORT_SCHEMA,
            "status": "test",
            "rows": [],
        }
    )
    csv_text = "cell,k\n"
    markdown = "# comparison\n"
    initial_capacity = {"sha256": "7" * 64}
    schedulers = [
        {"sha256": f"{index:x}" * 64, "scheduling_mode": "pair"}
        for index in range(1, 7)
    ]
    batch_receipts = [{"sha256": f"{index:x}" * 64} for index in range(7, 13)]
    closures = {
        cell.execution_id: runner.ArchivedCellClosure(
            cell=cell,
            rows=(),
            worker_receipt_sha256="5" * 64,
            guard_receipt_sha256="6" * 64,
            compact_receipt_sha256="a" * 64,
            archive_backed_closure_sha256="b" * 64,
            archive_closure_receipt_sha256="d" * 64,
            archived_cell_receipt_sha256="c" * 64,
        )
        for cell in runner.CELL_SPECS
    }
    monkeypatch.setattr(
        runner,
        "load_initial_campaign_capacity",
        lambda **_kwargs: initial_capacity,
    )
    monkeypatch.setattr(
        runner,
        "ensure_initial_campaign_capacity",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("terminal validation attempted capacity publication")
        ),
    )
    original_load = runner.load_digested

    def load_for_terminal(path, *, schema):
        if schema == runner.SCHEDULER_SCHEMA:
            ordinal = int(Path(path).stem.split("_")[-1])
            return schedulers[ordinal - 1]
        if schema == runner.BATCH_RECEIPT_SCHEMA:
            ordinal = int(Path(path).stem.split("_")[-1])
            return batch_receipts[ordinal - 1]
        return original_load(path, schema=schema)

    monkeypatch.setattr(runner, "load_digested", load_for_terminal)
    monkeypatch.setattr(
        runner,
        "validate_scheduler_decision",
        lambda batch, decision, **_kwargs: schedulers[batch.ordinal - 1],
    )
    monkeypatch.setattr(
        runner,
        "load_archived_cell",
        lambda cell, **_kwargs: closures[cell.execution_id],
    )
    monkeypatch.setattr(
        runner,
        "archive_and_load_cell",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("terminal validation attempted archive rotation")
        ),
    )
    monkeypatch.setattr(
        runner,
        "validate_batch_receipt",
        lambda batch, _receipt, **_kwargs: batch_receipts[batch.ordinal - 1],
    )
    monkeypatch.setattr(
        runner,
        "archive_paths",
        lambda cell: SimpleNamespace(source_root=tmp_path / f"absent-{cell.ordinal}"),
    )
    monkeypatch.setattr(
        runner,
        "build_comparison",
        lambda _closed: (comparison, csv_text, markdown),
    )
    monkeypatch.setattr(
        runner,
        "execution_path_canary_observation",
        lambda _comparison: canary_observation,
    )
    runner.write_json_exclusive(runner.REPORT_JSON, comparison)
    runner.write_text_exclusive(runner.REPORT_CSV, csv_text)
    runner.write_text_exclusive(runner.REPORT_MD, markdown)
    terminal = runner.digested(
        {
            "schema": runner.TERMINAL_SCHEMA,
            "status": "passed_all6_append_then_plateau_k50",
            "campaign_id": runner.CAMPAIGN_ID,
            "plan_sha256": plan["sha256"],
            "authorization_sha256": authorization["sha256"],
            "execution_path_canary_observation": canary_observation,
            "source_implementation_inventory_sha256": plan[
                "source_implementation_inventory_sha256"
            ],
            "canonical_cell_order": [
                cell.execution_id for cell in runner.CELL_SPECS
            ],
            "deterministic_launch_order": [
                execution_id
                for batch in runner.BATCH_SPECS
                for execution_id in batch.execution_ids
            ],
            "maximum_concurrency": runner.MAXIMUM_CONCURRENCY,
            "serial_capacity_fallback_authorized": (
                runner.SERIAL_CAPACITY_FALLBACK_AUTHORIZED
            ),
            "initial_capacity_receipt_sha256": initial_capacity["sha256"],
            "scheduler_decision_sha256s": [row["sha256"] for row in schedulers],
            "batch_receipt_sha256s": [row["sha256"] for row in batch_receipts],
            "archived_cell_receipt_sha256s": [
                closures[cell.execution_id].archived_cell_receipt_sha256
                for cell in runner.CELL_SPECS
            ],
            "archive_backed_closure_sha256s": [
                closures[cell.execution_id].archive_backed_closure_sha256
                for cell in runner.CELL_SPECS
            ],
            "archive_closure_receipt_sha256s": [
                closures[cell.execution_id].archive_closure_receipt_sha256
                for cell in runner.CELL_SPECS
            ],
            "compact_receipt_sha256s": [
                closures[cell.execution_id].compact_receipt_sha256
                for cell in runner.CELL_SPECS
            ],
            "archive_backed_cell_count": 12,
            "direct_run_tree_count": 0,
            "comparison_row_count": 0,
            "append_block_closed_before_plateau": True,
            "comparison_sha256": comparison["sha256"],
            "comparison_csv_sha256": runner.sha256_file(runner.REPORT_CSV),
            "comparison_markdown_sha256": runner.sha256_file(runner.REPORT_MD),
            "controller_rounds_completed_by_cell": {
                cell.execution_id: runner.TARGET_HORIZON
                for cell in runner.CELL_SPECS
            },
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    runner.write_json_exclusive(runner.TERMINAL_PATH, terminal)

    assert runner.validate_terminal_matrix(
        plan=plan, authorization=authorization
    ) == terminal
    runner.REPORT_MD.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(runner.RunnerError, match="Markdown"):
        runner.validate_terminal_matrix(plan=plan, authorization=authorization)


def test_report_gate_requires_typed_adaptive_receipts_for_all_three_phases() -> None:
    runner = _load_runner()
    caps = {"phase_i": 24, "phase_ii": 12, "phase_iii": 12}
    score_keys = {
        "phase_i": "phase1_active_score",
        "phase_ii": "phase2_raw_score",
        "phase_iii": "full_v2_score",
    }
    phases = []
    for index, (phase, cap) in enumerate(caps.items(), start=1):
        identities = tuple(
            adaptive_phase_record_id(
                generator_id=f"generator-{position}",
                pool_index=position - 1,
                insertion_position=0,
            )
            for position in (1, 2)
        )
        decision = select_adaptive_phase_shortlist(
            (
                AdaptivePhaseCandidateScore(
                    record_id=identities[0],
                    pool_index=0,
                    insertion_position=0,
                    active_score=2.0,
                    tie_break_score=1.0,
                ),
                AdaptivePhaseCandidateScore(
                    record_id=identities[1],
                    pool_index=1,
                    insertion_position=0,
                    active_score=1.9,
                    tie_break_score=0.9,
                ),
            ),
            phase=phase,
            score_key=score_keys[phase],
            hard_cap=cap,
            threshold=0.0,
            frontier_ratio=0.9,
        )
        receipt = decision.receipt.to_dict()
        records = [
            {
                "adaptive_record_id": score["record_id"],
                "generator_id": f"generator-{position}",
                "pool_index": position - 1,
                "insertion_position": 0,
            }
            for position, score in enumerate(receipt["input_scores"], start=1)
        ]
        shortlist = records if phase != "phase_iii" else records[:1]
        phases.append(
            {
                "phase": phase,
                "population_count": 2,
                "records": records,
                "ordered_population_sha256": runner.canonical_sha256(records),
                "shortlist_count": len(shortlist),
                "shortlist_records": shortlist,
                "adaptive_population_scores": receipt["input_scores"],
                "ordered_adaptive_population_scores_sha256": (
                    runner.canonical_sha256(receipt["input_scores"])
                ),
                "adaptive_shortlist": receipt,
                "final_admission_record_id": (
                    shortlist[0]["adaptive_record_id"]
                    if phase == "phase_iii"
                    else None
                ),
            }
        )
    round_receipt = {
        "scored_insertion_position_population": {"phases": phases}
    }

    assert runner._adaptive_phase_counts(round_receipt) == {
        "phase_i": (2, 2),
        "phase_ii": (2, 2),
        "phase_iii": (2, 2),
    }
    phase_iii = runner._adaptive_phase_evidence(round_receipt)["phase_iii"]
    assert phase_iii["adaptive_retained_count"] == 2
    assert phase_iii["final_singleton_count"] == 1
    assert phase_iii["final_record_id"] == phases[-1][
        "final_admission_record_id"
    ]
    with pytest.raises(runner.RunnerError, match="exact, unique, and ordered"):
        runner._adaptive_phase_evidence(
            {
                "scored_insertion_position_population": {
                    "phases": [phases[1], phases[0], phases[2]]
                }
            }
        )
    phases[0]["shortlist_records"][0]["adaptive_record_id"] = "detached"
    with pytest.raises(runner.RunnerError, match="deep mapping"):
        runner._adaptive_phase_counts(round_receipt)


def test_report_gate_rejects_a_self_digesting_receipt_detached_from_live_scores() -> None:
    runner = _load_runner()
    identities = tuple(
        adaptive_phase_record_id(
            generator_id=f"g{index}",
            pool_index=index,
            insertion_position=0,
        )
        for index in range(2)
    )
    decision = select_adaptive_phase_shortlist(
        (
            AdaptivePhaseCandidateScore(identities[0], 0, 0, 2.0, 1.0),
            AdaptivePhaseCandidateScore(identities[1], 1, 0, 1.9, 0.9),
        ),
        phase="phase_i",
        score_key="phase1_active_score",
        hard_cap=24,
        threshold=0.0,
        frontier_ratio=0.9,
    )
    receipt = decision.receipt.to_dict()
    records = [
        {
            "adaptive_record_id": row["record_id"],
            "generator_id": f"g{index}",
            "pool_index": index,
            "insertion_position": 0,
        }
        for index, row in enumerate(receipt["input_scores"])
    ]
    phase_i = {
        "phase": "phase_i",
        "population_count": 2,
        "records": records,
        "ordered_population_sha256": runner.canonical_sha256(records),
        "shortlist_count": 2,
        "shortlist_records": list(records),
        "adaptive_population_scores": list(reversed(receipt["input_scores"])),
        "ordered_adaptive_population_scores_sha256": runner.canonical_sha256(
            list(reversed(receipt["input_scores"]))
        ),
        "adaptive_shortlist": receipt,
    }
    with pytest.raises(runner.RunnerError, match="exact, unique, and ordered"):
        runner._adaptive_phase_evidence(
            {"scored_insertion_position_population": {"phases": [phase_i]}}
        )


def test_compact_cell_receipt_preserves_exact_50_row_reporting_projection(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "CELL_LOG_ROOT", tmp_path / "logs")
    cell = runner.CELL_SPECS[0]
    runner.cell_log_path(cell).parent.mkdir(parents=True)
    runner.cell_log_path(cell).write_text("cell output\n", encoding="utf-8")
    rows = []
    for controller_round in range(1, runner.TARGET_HORIZON + 1):
        rows.append(
            {
                "execution_id": cell.execution_id,
                "cell_ordinal": cell.ordinal,
                "block": cell.block,
                "regime_id": cell.regime_id,
                "nph": cell.nph,
                "insertion_policy": cell.insertion_policy,
                "controller_round": controller_round,
                "energy": -1.0,
                "absolute_delta_e": 0.1,
                "plateau_state": "closed",
                "phase0_population_count": 10,
                "phase0_retained_count": 5,
                "phase_i_input_count": 5,
                "phase_i_retained_count": 4,
                "phase_ii_input_count": 4,
                "phase_ii_retained_count": 3,
                "phase_iii_input_count": 3,
                "phase_iii_adaptive_retained_count": 2,
                "phase_iii_final_singleton_count": 1,
                "phase_iii_final_record_id": "record-1",
                "selected_generator": "generator-1",
                "selected_operator": "operator-1",
                "selected_position": 0,
                "s_alg": controller_round,
                "n2q": 1,
                "d2q": 1,
                "dc": 1,
                "checkpoint_sha256": "a" * 64,
            }
        )
    plan = {
        "sha256": "b" * 64,
        "source_implementation_inventory_sha256": "c" * 64,
        "protocol_bindings": [
            {"execution_id": cell.execution_id, "protocol_sha256": "d" * 64}
        ],
    }
    authorization = {"sha256": "e" * 64}
    artifact_bindings = {
        role: {"path": f"{role}.json", "sha256": "5" * 64, "size_bytes": 1}
        for role in ("checkpoint", "estimator_ledger", "result", "summary")
    }
    manifest_file_binding = {
        "path": "execution_manifest.json",
        "sha256": "7" * 64,
        "size_bytes": 2,
    }
    receipt = runner.build_compact_cell_receipt(
        cell,
        rows=rows,
        plan=plan,
        authorization=authorization,
        manifest_sha256="1" * 64,
        manifest_file_binding=manifest_file_binding,
        worker_receipt_sha256="2" * 64,
        guard_receipt_sha256="3" * 64,
        scheduler_decision_sha256="4" * 64,
        artifact_bindings=artifact_bindings,
    )

    assert runner.validate_compact_cell_receipt(
        cell,
        receipt,
        plan=plan,
        authorization=authorization,
        manifest_sha256="1" * 64,
        manifest_file_binding=manifest_file_binding,
        worker_receipt_sha256="2" * 64,
        guard_receipt_sha256="3" * 64,
        scheduler_decision_sha256="4" * 64,
        artifact_bindings=artifact_bindings,
    ) == receipt
    assert len(receipt["rows"]) == 50
    assert receipt["rows_sha256"] == runner.canonical_sha256(rows)

    tampered = {**receipt, "rows": [dict(row) for row in receipt["rows"]]}
    tampered["rows"][0]["phase_iii_final_singleton_count"] = 2
    tampered = runner.digested(tampered)
    with pytest.raises(runner.RunnerError, match="compact"):
        runner.validate_compact_cell_receipt(
            cell,
            tampered,
            plan=plan,
            authorization=authorization,
            manifest_sha256="1" * 64,
            manifest_file_binding=manifest_file_binding,
            worker_receipt_sha256="2" * 64,
            guard_receipt_sha256="3" * 64,
            scheduler_decision_sha256="4" * 64,
            artifact_bindings=artifact_bindings,
        )

    detached = runner.digested({**receipt, "manifest_sha256": "6" * 64})
    with pytest.raises(runner.RunnerError, match="compact"):
        runner.validate_compact_cell_receipt(
            cell,
            detached,
            plan=plan,
            authorization=authorization,
            manifest_sha256="1" * 64,
            manifest_file_binding=manifest_file_binding,
            worker_receipt_sha256="2" * 64,
            guard_receipt_sha256="3" * 64,
            scheduler_decision_sha256="4" * 64,
            artifact_bindings=artifact_bindings,
        )


def _valid_compact_rows(runner, cell):
    return [
        {
            "execution_id": cell.execution_id,
            "cell_ordinal": cell.ordinal,
            "block": cell.block,
            "regime_id": cell.regime_id,
            "nph": cell.nph,
            "insertion_policy": cell.insertion_policy,
            "controller_round": controller_round,
            "energy": -1.0,
            "absolute_delta_e": 0.1,
            "plateau_state": "closed",
            "phase0_population_count": 10,
            "phase0_retained_count": 5,
            "phase_i_input_count": 5,
            "phase_i_retained_count": 4,
            "phase_ii_input_count": 4,
            "phase_ii_retained_count": 3,
            "phase_iii_input_count": 3,
            "phase_iii_adaptive_retained_count": 2,
            "phase_iii_final_singleton_count": 1,
            "phase_iii_final_record_id": "record-1",
            "selected_generator": "generator-1",
            "selected_operator": "operator-1",
            "selected_position": 0,
            "s_alg": controller_round,
            "n2q": 1,
            "d2q": 1,
            "dc": 1,
            "checkpoint_sha256": "a" * 64,
        }
        for controller_round in range(1, runner.TARGET_HORIZON + 1)
    ]


def _archive_cell_fixture(monkeypatch, tmp_path: Path, runner):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(runner, "RUNTIME_ROOT", runtime)
    monkeypatch.setattr(runner, "RUNS_ROOT", runtime / "runs")
    monkeypatch.setattr(runner, "STAGING_ROOT", runtime / "in_progress")
    monkeypatch.setattr(runner, "RECEIPTS_ROOT", runtime / "worker_receipts")
    monkeypatch.setattr(runner, "GUARD_ROOT", runtime / "guard_receipts")
    monkeypatch.setattr(runner, "CELL_LOG_ROOT", runtime / "cell_logs")
    monkeypatch.setattr(runner, "COMPACT_ROOT", runtime / "compact_cell_receipts")
    monkeypatch.setattr(
        runner, "ARCHIVED_RECEIPTS_ROOT", runtime / "archived_cell_receipts"
    )
    monkeypatch.setattr(runner, "SCHEDULER_ROOT", runtime / "scheduler_receipts")
    cell = runner.CELL_SPECS[0]
    batch = runner._batch_for_cell(cell)
    plan = {
        "sha256": "b" * 64,
        "runner": {"sha256": "c" * 64},
        "source_implementation_inventory_sha256": "d" * 64,
        "protocol_bindings": [
            {"execution_id": cell.execution_id, "protocol_sha256": "e" * 64}
        ],
    }
    authorization = {
        "sha256": "f" * 64,
        "execution_authorized": True,
        "archive_rotation_authorized": True,
    }
    contract = runner.pair_launch_capacity_contract(batch)
    observation = {
        **contract,
        "status": "ready_pair",
        "scheduling_mode": "pair",
        "available_memory_bytes": contract["required_available_memory_bytes"],
        "free_disk_bytes": contract["required_free_disk_bytes"],
        "physical_memory_bytes": contract["required_physical_memory_bytes"],
        "elapsed_wait_seconds": 0.0,
        "fallback_reasons": [],
        "launch_available_memory_bytes": contract[
            "required_available_memory_bytes"
        ],
        "launch_free_disk_bytes": contract["required_free_disk_bytes"],
        "launch_ready": True,
    }
    scheduler = runner.select_batch_schedule(
        batch,
        plan=plan,
        authorization=authorization,
        capacity_waiter=lambda _batch: observation,
    )
    run_dir = runner.RUNS_ROOT / cell.execution_id
    artifact_paths = {
        "checkpoint": run_dir / "checkpoints/current.json",
        "estimator_ledger": run_dir / "result/estimator_ledger.json",
        "result": run_dir / "result/result.json",
        "summary": run_dir / "summary/summary.json",
    }
    for path in artifact_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    artifacts = {
        role: runner._artifact_binding(path, run_dir)
        for role, path in artifact_paths.items()
    }
    manifest = runner.digested(
        {
            "schema": runner.MANIFEST_SCHEMA,
            "status": "passed_k50",
            "campaign_id": runner.CAMPAIGN_ID,
            "execution_id": cell.execution_id,
            "cell": runner.asdict(cell),
            "plan_sha256": plan["sha256"],
            "authorization_sha256": authorization["sha256"],
            "source_implementation_inventory_sha256": plan[
                "source_implementation_inventory_sha256"
            ],
            "protocol_binding": plan["protocol_bindings"][0],
            "controller_rounds_completed": runner.TARGET_HORIZON,
            "execution_path_canary": {
                "is_canary_cell": True,
                "accepted_round": 1,
                "continued_same_trajectory_to_k50": True,
            },
            "artifacts": artifacts,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    runner.write_json_exclusive(run_dir / "execution_manifest.json", manifest)
    worker = runner.digested(
        {
            "schema": runner.WORKER_SCHEMA,
            "status": "passed_k50",
            "campaign_id": runner.CAMPAIGN_ID,
            "execution_id": cell.execution_id,
            "manifest_sha256": manifest["sha256"],
            "artifact_inventory": [
                runner._artifact_binding(path, runtime)
                for path in sorted(run_dir.rglob("*"))
                if path.is_file()
            ],
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    runner.write_json_exclusive(runner.cell_paths(cell)[2], worker)
    runner.cell_log_path(cell).parent.mkdir(parents=True)
    runner.cell_log_path(cell).write_text("authenticated cell output\n", encoding="utf-8")
    launch_observation = _pair_launch_observation(runner, batch)
    guard = runner.digested(
        {
            "schema": runner.GUARD_SCHEMA,
            "status": "passed",
            "campaign_id": runner.CAMPAIGN_ID,
            "execution_id": cell.execution_id,
            "batch_ordinal": batch.ordinal,
            "scheduling_mode": "pair",
            "scheduler_decision_sha256": scheduler["sha256"],
            "launch_capacity_observation": launch_observation,
            "launch_capacity_observation_sha256": runner.canonical_sha256(
                launch_observation
            ),
            "returncode": 0,
            "stop_reason": None,
            "elapsed_seconds": 1.0,
            "peak_rss_bytes": 1,
            "rss_limit_bytes": runner.child_rss_limit_bytes(cell),
            "minimum_available_memory_bytes": 16 * 1024**3,
            "minimum_free_disk_bytes": 100 * 1024**3,
            "worker_receipt_sha256": worker["sha256"],
            "log_file_binding": runner._log_file_binding(cell),
            "attempt_inventory": runner._attempt_inventory(cell),
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    runner.write_json_exclusive(runner.cell_paths(cell)[3], guard)
    monkeypatch.setattr(
        runner,
        "report_rows",
        lambda selected, _result, _summary: _valid_compact_rows(runner, selected),
    )
    return cell, plan, authorization, scheduler


def test_direct_cell_rotates_to_restart_safe_archive_without_extraction(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    cell, plan, authorization, scheduler = _archive_cell_fixture(
        monkeypatch, tmp_path, runner
    )
    closure = runner.archive_and_load_cell(
        cell,
        plan=plan,
        authorization=authorization,
        scheduler_decision=scheduler,
        archive_capacity_waiter=lambda: {"status": "ready"},
    )
    paths = runner.archive_paths(cell)
    assert len(closure.rows) == 50
    assert runner.strict_archive.inspect_rotation_state(paths)["state"] == "archived_closed"
    assert not paths.source_root.exists()
    assert runner.compact_cell_receipt_path(cell).is_file()
    assert runner.archived_cell_receipt_path(cell).is_file()
    compact = runner.load_digested(
        runner.compact_cell_receipt_path(cell), schema=runner.COMPACT_CELL_SCHEMA
    )
    assert compact["log_file_binding"] == runner._log_file_binding(cell)
    archive_manifest = runner.load_digested(
        paths.archive_manifest_path, schema=runner.strict_archive.ARCHIVE_SCHEMA
    )
    assert "evidence/cell.log" in {
        row["path"] for row in archive_manifest["external_members"]
    }

    monkeypatch.setattr(
        runner,
        "publish_compact_cell_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("archived restart attempted direct reconstruction")
        ),
    )
    restarted = runner.archive_and_load_cell(
        cell,
        plan=plan,
        authorization=authorization,
        scheduler_decision=scheduler,
    )
    assert restarted == closure


def test_passed_guard_cannot_rehash_a_peak_above_its_rss_limit(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    cell, plan, authorization, scheduler = _archive_cell_fixture(
        monkeypatch, tmp_path, runner
    )
    guard_path = runner.cell_paths(cell)[3]
    guard = runner.load_digested(guard_path, schema=runner.GUARD_SCHEMA)
    payload = {key: value for key, value in guard.items() if key != "sha256"}
    payload["peak_rss_bytes"] = int(payload["rss_limit_bytes"]) + 1
    guard_path.write_text(
        runner.json.dumps(runner.digested(payload), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(runner.RunnerError, match="guard closure drifted"):
        runner.load_closed_cell(
            cell,
            plan=plan,
            authorization=authorization,
            scheduler_decision=scheduler,
        )


def test_archived_restart_rejects_rehashed_external_compact_detachment(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    cell, plan, authorization, scheduler = _archive_cell_fixture(
        monkeypatch, tmp_path, runner
    )
    runner.archive_and_load_cell(
        cell,
        plan=plan,
        authorization=authorization,
        scheduler_decision=scheduler,
        archive_capacity_waiter=lambda: {"status": "ready"},
    )
    compact_path = runner.compact_cell_receipt_path(cell)
    compact = runner.load_digested(compact_path, schema=runner.COMPACT_CELL_SCHEMA)
    detached = runner.digested({**compact, "manifest_sha256": "0" * 64})
    compact_path.write_text(
        runner.json.dumps(detached, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(runner.RunnerError):
        runner.archive_and_load_cell(
            cell,
            plan=plan,
            authorization=authorization,
            scheduler_decision=scheduler,
        )


def test_missing_initial_capacity_receipt_cannot_be_minted_after_science_evidence(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "INITIAL_CAPACITY_PATH", tmp_path / "initial.json")
    monkeypatch.setattr(runner, "SCHEDULER_ROOT", tmp_path / "scheduler")
    runner.write_json_exclusive(
        runner.scheduler_decision_path(runner.BATCH_SPECS[0]),
        runner.digested({"schema": runner.SCHEDULER_SCHEMA}),
    )
    with pytest.raises(runner.RunnerError, match="capacity receipt is missing"):
        runner.ensure_initial_campaign_capacity(
            plan={
                "sha256": "1" * 64,
                "source_implementation_inventory_sha256": "2" * 64,
            },
            authorization={"sha256": "3" * 64},
            capacity_waiter=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("retroactive gate attempted a live capacity wait")
            ),
        )


def test_absent_runtime_root_is_a_pristine_initial_capacity_state(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    runtime = tmp_path / "absent-runtime"
    monkeypatch.setattr(runner, "RUNTIME_ROOT", runtime)
    monkeypatch.setattr(runner, "REPORT_JSON", runtime / "comparison.json")
    monkeypatch.setattr(runner, "REPORT_CSV", runtime / "comparison.csv")
    monkeypatch.setattr(runner, "REPORT_MD", runtime / "comparison.md")
    monkeypatch.setattr(runner, "TERMINAL_PATH", runtime / "terminal.json")
    monkeypatch.setattr(
        runner,
        "scheduler_decision_path",
        lambda batch: runtime / "scheduler" / f"{batch.ordinal}.json",
    )
    monkeypatch.setattr(
        runner,
        "batch_receipt_path",
        lambda batch: runtime / "batch" / f"{batch.ordinal}.json",
    )
    monkeypatch.setattr(
        runner,
        "cell_paths",
        lambda cell: (
            runtime / "runs" / cell.execution_id,
            runtime / "staging" / cell.execution_id,
            runtime / "worker" / f"{cell.execution_id}.json",
            runtime / "guard" / f"{cell.execution_id}.json",
        ),
    )
    monkeypatch.setattr(
        runner,
        "cell_log_path",
        lambda cell: runtime / "logs" / f"{cell.execution_id}.log",
    )
    monkeypatch.setattr(
        runner,
        "compact_cell_receipt_path",
        lambda cell: runtime / "compact" / f"{cell.execution_id}.json",
    )
    monkeypatch.setattr(
        runner,
        "archived_cell_receipt_path",
        lambda cell: runtime / "archived" / f"{cell.execution_id}.json",
    )
    runner._assert_campaign_pristine_before_initial_capacity()


def test_initial_capacity_receipt_deeply_validates_both_observations(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    path = tmp_path / "initial.json"
    monkeypatch.setattr(runner, "INITIAL_CAPACITY_PATH", path)
    floor = runner.strict_archive.campaign_capacity_floor()
    required_disk = int(floor["campaign_minimum_free_bytes"])
    bounded = {
        "available_memory_bytes": runner.LAUNCH_AVAILABLE_MEMORY_BYTES,
        "free_disk_bytes": required_disk,
        "launch_available_memory_bytes": runner.LAUNCH_AVAILABLE_MEMORY_BYTES,
        "launch_free_disk_bytes": required_disk,
        "launch_ready": True,
        "elapsed_wait_seconds": 0.0,
        "status": "ready",
        "ready_after_bound": False,
    }
    strict = {
        **floor,
        "status": "passed_campaign_capacity_floor",
        "observed_free_bytes": required_disk,
        "headroom_bytes": 0,
    }
    plan = {
        "sha256": "1" * 64,
        "source_implementation_inventory_sha256": "2" * 64,
    }
    authorization = {"sha256": "3" * 64}
    receipt = runner.digested(
        {
            "schema": runner.INITIAL_CAPACITY_SCHEMA,
            "status": "passed_one_time_initial_campaign_capacity",
            "created_at": "2026-08-17T00:00:00Z",
            "campaign_id": runner.CAMPAIGN_ID,
            "plan_sha256": plan["sha256"],
            "authorization_sha256": authorization["sha256"],
            "source_implementation_inventory_sha256": plan[
                "source_implementation_inventory_sha256"
            ],
            "campaign_capacity_floor": floor,
            "bounded_wait_observation": bounded,
            "strict_capacity_observation": strict,
            "one_time_gate_not_reimposed_on_restart": True,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    runner.write_json_exclusive(path, receipt)
    assert runner.load_initial_campaign_capacity(
        plan=plan, authorization=authorization
    ) == receipt
    bad_payload = {key: value for key, value in receipt.items() if key != "sha256"}
    bad_payload["bounded_wait_observation"] = {
        **bounded,
        "available_memory_bytes": runner.LAUNCH_AVAILABLE_MEMORY_BYTES - 1,
    }
    path.write_text(
        runner.json.dumps(runner.digested(bad_payload), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(runner.RunnerError, match="capacity receipt drifted"):
        runner.load_initial_campaign_capacity(plan=plan, authorization=authorization)


@pytest.mark.parametrize(
    ("mode", "expected_events"),
    [
        ("pair", ["monitor:1,2", "archive:1", "archive:2"]),
        (
            "serial_capacity_fallback",
            ["monitor:1", "archive:1", "monitor:2", "archive:2"],
        ),
    ],
)
def test_archived_batch_dispatches_pair_or_serial_and_archives_before_progress(
    monkeypatch, mode: str, expected_events
) -> None:
    runner = _load_runner()
    batch = runner.BATCH_SPECS[0]
    cells = [runner._cell_by_execution_id(value) for value in batch.execution_ids]
    plan = {"sha256": "1" * 64}
    authorization = {"sha256": "2" * 64}
    scheduler = {"sha256": "3" * 64, "scheduling_mode": mode}
    events = []
    monkeypatch.setattr(runner, "_cell_lifecycle_state", lambda _cell: "pristine")
    monkeypatch.setattr(
        runner,
        "validate_scheduler_decision",
        lambda _batch, _decision, **_kwargs: scheduler,
    )
    monkeypatch.setattr(
        runner,
        "publish_batch_receipt",
        lambda *_args, **_kwargs: {"sha256": "4" * 64},
    )
    monkeypatch.setattr(
        runner,
        "load_closed_cell",
        lambda cell, **_kwargs: events.append(f"preflight:{cell.ordinal}"),
    )

    def monitor(selected, *_args, **_kwargs):
        events.append(
            "monitor:" + ",".join(str(cell.ordinal) for cell in selected)
        )
        return []

    def archive_loader(cell, **_kwargs):
        events.append(f"archive:{cell.ordinal}")
        return runner.ArchivedCellClosure(
            cell=cell,
            rows=tuple(_valid_compact_rows(runner, cell)),
            worker_receipt_sha256="5" * 64,
            guard_receipt_sha256="6" * 64,
            compact_receipt_sha256="7" * 64,
            archive_backed_closure_sha256="8" * 64,
            archive_closure_receipt_sha256="b" * 64,
            archived_cell_receipt_sha256="9" * 64,
        )

    closures, _receipt = runner.run_archived_batch(
        batch,
        plan=plan,
        authorization=authorization,
        initial_capacity_receipt={"sha256": "a" * 64},
        schedule_selector=lambda *_args, **_kwargs: scheduler,
        pair_capacity_waiter=lambda _batch: {"status": "ready"},
        cell_capacity_waiter=lambda _cell: {"status": "ready"},
        monitor=monitor,
        archive_loader=archive_loader,
        post_science_validator=lambda *_args, **_kwargs: None,
    )
    expected_with_preflight = []
    for event in expected_events:
        expected_with_preflight.append(event)
        if event.startswith("monitor:"):
            expected_with_preflight.extend(
                f"preflight:{ordinal}" for ordinal in event.split(":", 1)[1].split(",")
            )
    assert events == expected_with_preflight
    assert [row.cell for row in closures] == cells


def test_pair_post_science_identity_failure_is_failed_pair_before_archive(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    batch = runner.BATCH_SPECS[0]
    scheduler = {"sha256": "3" * 64, "scheduling_mode": "pair"}
    archive_calls = []
    monkeypatch.setattr(runner, "_cell_lifecycle_state", lambda _cell: "pristine")
    monkeypatch.setattr(
        runner,
        "scheduler_decision_path",
        lambda _batch: tmp_path / "scheduler.json",
    )
    monkeypatch.setattr(
        runner,
        "batch_receipt_path",
        lambda _batch: tmp_path / "batch.json",
    )
    monkeypatch.setattr(
        runner,
        "validate_scheduler_decision",
        lambda _batch, _decision, **_kwargs: scheduler,
    )

    def fail_identity(*_args, **_kwargs):
        raise runner.RunnerError("Post-science protocol identity drifted.")

    with pytest.raises(runner.BatchExecutionFailed) as caught:
        runner.run_archived_batch(
            batch,
            plan={"sha256": "1" * 64},
            authorization={"sha256": "2" * 64},
            initial_capacity_receipt={"sha256": "a" * 64},
            schedule_selector=lambda *_args, **_kwargs: scheduler,
            pair_capacity_waiter=lambda _batch: {"status": "ready"},
            monitor=lambda *_args, **_kwargs: [],
            archive_loader=lambda cell, **_kwargs: archive_calls.append(cell),
            post_science_validator=fail_identity,
        )
    assert caught.value.failure_status == "failed_pair"
    assert archive_calls == []


@pytest.mark.parametrize(
    ("mode", "states", "message"),
    [
        ("pair", ("direct_unarchived", "pristine"), "one pristine"),
        (
            "pair",
            ("direct_unarchived", "closure_published_pending_intent"),
            "archive order",
        ),
        (
            "serial_capacity_fallback",
            ("pristine", "archived_closed"),
            "canonical cell order",
        ),
        (
            "serial_capacity_fallback",
            ("direct_unarchived", "direct_unarchived"),
            "canonical cell order",
        ),
    ],
)
def test_batch_restart_rejects_illegal_state_order_before_archive_mutation(
    monkeypatch, tmp_path: Path, mode: str, states, message: str
) -> None:
    runner = _load_runner()
    batch = runner.BATCH_SPECS[0]
    by_execution = dict(zip(batch.execution_ids, states, strict=True))
    scheduler = {"sha256": "3" * 64, "scheduling_mode": mode}
    archive_calls = []
    scheduler_path = tmp_path / "scheduler.json"
    scheduler_path.touch()
    monkeypatch.setattr(
        runner, "scheduler_decision_path", lambda _batch: scheduler_path
    )
    monkeypatch.setattr(
        runner,
        "_cell_lifecycle_state",
        lambda cell: by_execution[cell.execution_id],
    )
    monkeypatch.setattr(
        runner,
        "validate_scheduler_decision",
        lambda _batch, _decision, **_kwargs: scheduler,
    )
    with pytest.raises(runner.RunnerError, match=message):
        runner.run_archived_batch(
            batch,
            plan={"sha256": "1" * 64},
            authorization={"sha256": "2" * 64},
            initial_capacity_receipt={"sha256": "4" * 64},
            schedule_selector=lambda *_args, **_kwargs: scheduler,
            archive_loader=lambda cell, **_kwargs: archive_calls.append(cell),
        )
    assert archive_calls == []


def test_progressed_batch_without_scheduler_receipt_fails_before_mutation(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    batch = runner.BATCH_SPECS[0]
    states = iter(("direct_unarchived", "pristine"))
    archive_calls = []
    monkeypatch.setattr(runner, "_cell_lifecycle_state", lambda _cell: next(states))
    monkeypatch.setattr(
        runner,
        "scheduler_decision_path",
        lambda _batch: tmp_path / "absent-scheduler.json",
    )
    with pytest.raises(runner.RunnerError, match="Scheduler receipt is missing"):
        runner.run_archived_batch(
            batch,
            plan={"sha256": "1" * 64},
            authorization={"sha256": "2" * 64},
            initial_capacity_receipt={"sha256": "3" * 64},
            schedule_selector=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("scheduler was minted after science")
            ),
            archive_loader=lambda cell, **_kwargs: archive_calls.append(cell),
        )
    assert archive_calls == []


def test_batch_deep_preflight_validates_both_cells_before_first_archive(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    batch = runner.BATCH_SPECS[0]
    scheduler_path = tmp_path / "scheduler.json"
    scheduler_path.touch()
    scheduler = {"sha256": "3" * 64, "scheduling_mode": "pair"}
    calls = []
    monkeypatch.setattr(runner, "scheduler_decision_path", lambda _batch: scheduler_path)
    monkeypatch.setattr(runner, "_cell_lifecycle_state", lambda _cell: "direct_unarchived")
    monkeypatch.setattr(
        runner, "validate_scheduler_decision", lambda *_args, **_kwargs: scheduler
    )

    def preflight(cell, *_args, **_kwargs):
        calls.append(f"preflight:{cell.ordinal}")
        if cell == runner._cell_by_execution_id(batch.execution_ids[1]):
            raise runner.RunnerError("tampered second guard")

    monkeypatch.setattr(runner, "_preflight_nonpristine_cell", preflight)
    with pytest.raises(runner.RunnerError, match="tampered second guard"):
        runner.run_archived_batch(
            batch,
            plan={"sha256": "1" * 64},
            authorization={"sha256": "2" * 64},
            initial_capacity_receipt={"sha256": "4" * 64},
            schedule_selector=lambda *_args, **_kwargs: scheduler,
            archive_loader=lambda cell, **_kwargs: calls.append(f"archive:{cell.ordinal}"),
        )
    assert calls == ["preflight:1", "preflight:2"]


def test_campaign_preflight_rejects_future_batch_evidence_before_validation_or_mutation(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "INITIAL_CAPACITY_PATH", tmp_path / "initial.json")
    monkeypatch.setattr(runner, "_cell_lifecycle_state", lambda _cell: "pristine")
    scheduler_paths = {
        batch.ordinal: tmp_path / f"scheduler-{batch.ordinal}.json"
        for batch in runner.BATCH_SPECS
    }
    scheduler_paths[2].write_text("future\n", encoding="utf-8")
    monkeypatch.setattr(
        runner, "scheduler_decision_path", lambda batch: scheduler_paths[batch.ordinal]
    )
    monkeypatch.setattr(
        runner, "batch_receipt_path", lambda batch: tmp_path / f"batch-{batch.ordinal}.json"
    )
    monkeypatch.setattr(runner, "REPORT_JSON", tmp_path / "comparison.json")
    monkeypatch.setattr(runner, "REPORT_CSV", tmp_path / "comparison.csv")
    monkeypatch.setattr(runner, "REPORT_MD", tmp_path / "comparison.md")
    with pytest.raises(runner.RunnerError, match="canonical batch prefix"):
        runner.preflight_campaign_lifecycle(
            plan={"sha256": "1" * 64}, authorization={"sha256": "2" * 64}
        )


def test_campaign_preflight_allows_valid_partial_final_report_recovery(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    plan = {"sha256": "1" * 64}
    authorization = {"sha256": "2" * 64}
    initial = {"sha256": "3" * 64}
    initial_path = tmp_path / "initial.json"
    initial_path.touch()
    monkeypatch.setattr(runner, "INITIAL_CAPACITY_PATH", initial_path)
    monkeypatch.setattr(
        runner, "load_initial_campaign_capacity", lambda **_kwargs: initial
    )
    monkeypatch.setattr(runner, "_cell_lifecycle_state", lambda _cell: "archived_closed")
    closures = {
        cell.execution_id: runner.ArchivedCellClosure(
            cell=cell,
            rows=(),
            worker_receipt_sha256="4" * 64,
            guard_receipt_sha256="5" * 64,
            compact_receipt_sha256="6" * 64,
            archive_backed_closure_sha256="7" * 64,
            archive_closure_receipt_sha256="8" * 64,
            archived_cell_receipt_sha256="9" * 64,
        )
        for cell in runner.CELL_SPECS
    }
    monkeypatch.setattr(
        runner,
        "_preflight_nonpristine_cell",
        lambda cell, *_args, **_kwargs: closures[cell.execution_id],
    )
    scheduler_paths = {}
    receipt_paths = {}
    for batch in runner.BATCH_SPECS:
        scheduler_paths[batch.ordinal] = tmp_path / f"scheduler-{batch.ordinal}.json"
        receipt_paths[batch.ordinal] = tmp_path / f"batch-{batch.ordinal}.json"
        scheduler_paths[batch.ordinal].touch()
        receipt_paths[batch.ordinal].touch()
    monkeypatch.setattr(
        runner, "scheduler_decision_path", lambda batch: scheduler_paths[batch.ordinal]
    )
    monkeypatch.setattr(
        runner, "batch_receipt_path", lambda batch: receipt_paths[batch.ordinal]
    )
    monkeypatch.setattr(
        runner,
        "validate_scheduler_decision",
        lambda batch, _receipt, **_kwargs: {
            "sha256": f"{batch.ordinal:x}" * 64,
            "scheduling_mode": "pair",
        },
    )
    monkeypatch.setattr(
        runner,
        "validate_batch_receipt",
        lambda batch, _receipt, **_kwargs: {"sha256": f"{batch.ordinal:x}" * 64},
    )
    comparison = {"schema": runner.REPORT_SCHEMA, "sha256": "a" * 64}
    monkeypatch.setattr(
        runner,
        "build_comparison",
        lambda _closures: (comparison, "csv\n", "markdown\n"),
    )
    monkeypatch.setattr(runner, "REPORT_JSON", tmp_path / "comparison.json")
    monkeypatch.setattr(runner, "REPORT_CSV", tmp_path / "comparison.csv")
    monkeypatch.setattr(runner, "REPORT_MD", tmp_path / "comparison.md")
    monkeypatch.setattr(runner, "TERMINAL_PATH", tmp_path / "terminal.json")
    runner.REPORT_JSON.touch()

    def load_receipt(path, *, schema):
        if schema == runner.REPORT_SCHEMA:
            return comparison
        return {"schema": schema, "sha256": "b" * 64}

    monkeypatch.setattr(runner, "load_digested", load_receipt)
    observed = runner.preflight_campaign_lifecycle(
        plan=plan, authorization=authorization
    )
    assert observed["completed_batch_count"] == 6
    assert observed["report_set_present"] is False


def test_campaign_orchestrates_six_archived_batches_and_writes_600_row_terminal(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(runner, "RUNTIME_ROOT", runtime)
    monkeypatch.setattr(runner, "LOCK_PATH", runtime / "campaign.lock")
    monkeypatch.setattr(runner, "REPORT_JSON", runtime / "comparison.json")
    monkeypatch.setattr(runner, "REPORT_CSV", runtime / "comparison.csv")
    monkeypatch.setattr(runner, "REPORT_MD", runtime / "comparison.md")
    monkeypatch.setattr(runner, "TERMINAL_PATH", runtime / "terminal.json")
    plan = {
        "sha256": "1" * 64,
        "source_implementation_inventory_sha256": "2" * 64,
    }
    authorization = {"sha256": "3" * 64}
    initial = {"sha256": "4" * 64}
    observed_batches = []
    all_closures = {
        cell.execution_id: runner.ArchivedCellClosure(
            cell=cell,
            rows=tuple(_valid_compact_rows(runner, cell)),
            worker_receipt_sha256="5" * 64,
            guard_receipt_sha256="6" * 64,
            compact_receipt_sha256="7" * 64,
            archive_backed_closure_sha256="8" * 64,
            archive_closure_receipt_sha256="b" * 64,
            archived_cell_receipt_sha256="9" * 64,
        )
        for cell in runner.CELL_SPECS
    }
    batch_receipts = {
        batch.ordinal: {"sha256": f"{batch.ordinal:x}" * 64}
        for batch in runner.BATCH_SPECS
    }

    def run_batch(batch, **_kwargs):
        observed_batches.append(batch.ordinal)
        return (
            [all_closures[value] for value in batch.execution_ids],
            batch_receipts[batch.ordinal],
        )

    comparison = runner.digested(
        {
            "schema": runner.REPORT_SCHEMA,
            "status": "test",
            "rows": [{} for _ in range(600)],
        }
    )
    monkeypatch.setattr(
        runner, "validate_authority", lambda **_kwargs: (plan, authorization)
    )
    monkeypatch.setattr(
        runner, "preflight_campaign_lifecycle", lambda **_kwargs: {"status": "passed"}
    )
    monkeypatch.setattr(
        runner, "ensure_initial_campaign_capacity", lambda **_kwargs: initial
    )
    monkeypatch.setattr(runner, "run_archived_batch", run_batch)
    monkeypatch.setattr(
        runner,
        "build_comparison",
        lambda closures: (
            comparison,
            "rows\n",
            "# rows\n",
        ),
    )
    monkeypatch.setattr(
        runner,
        "execution_path_canary_observation",
        lambda _comparison: {"accepted_round": 1},
    )
    original_load = runner.load_digested

    def load_scheduler(path, *, schema):
        if schema == runner.SCHEDULER_SCHEMA:
            return {"sha256": "a" * 64}
        return original_load(path, schema=schema)

    monkeypatch.setattr(runner, "load_digested", load_scheduler)
    monkeypatch.setattr(
        runner,
        "archive_paths",
        lambda cell: SimpleNamespace(source_root=tmp_path / f"absent-{cell.ordinal}"),
    )
    terminal_events = []
    monkeypatch.setattr(
        runner,
        "write_status",
        lambda payload: terminal_events.append(("status", payload["status"])),
    )

    def validate_terminal(**_kwargs):
        terminal_events.append(("validate", None))
        return runner.load_digested(
            runner.TERMINAL_PATH, schema=runner.TERMINAL_SCHEMA
        )

    monkeypatch.setattr(
        runner,
        "validate_terminal_matrix",
        validate_terminal,
    )

    assert runner.run_campaign() == 0
    assert observed_batches == [1, 2, 3, 4, 5, 6]
    terminal = original_load(runner.TERMINAL_PATH, schema=runner.TERMINAL_SCHEMA)
    assert terminal["archive_backed_cell_count"] == 12
    assert terminal["comparison_row_count"] == 600
    assert terminal["direct_run_tree_count"] == 0
    assert terminal_events[-2:] == [
        ("validate", None),
        ("status", "passed_all6_append_then_plateau_k50"),
    ]
    assert not hasattr(runner, "monitor_cell")


@pytest.mark.parametrize(
    ("message", "expected_status"),
    [
        ("Batch 2 failed at cell; sibling execution stopped.", "failed_pair"),
        ("terminal recomputation drifted", "failed_campaign"),
    ],
)
def test_campaign_wrapper_publishes_truthful_failure_status(
    monkeypatch, message: str, expected_status: str
) -> None:
    runner = _load_runner()
    observed = []
    monkeypatch.setattr(
        runner,
        "_run_campaign_impl",
        lambda: (_ for _ in ()).throw(runner.RunnerError(message)),
    )
    monkeypatch.setattr(runner, "write_status", lambda payload: observed.append(payload))
    with pytest.raises(runner.RunnerError, match=message.split(";")[0]):
        runner.run_campaign()
    assert observed[-1]["status"] == expected_status


def test_existing_valid_terminal_restores_canonical_pass_status(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _load_runner()
    runtime = tmp_path / "runtime"
    terminal_path = runtime / "terminal.json"
    terminal_path.parent.mkdir(parents=True)
    terminal_path.write_text("{}\n", encoding="utf-8")
    plan = {"sha256": "1" * 64}
    authorization = {"sha256": "2" * 64}
    terminal = {"sha256": "3" * 64}
    statuses = []
    monkeypatch.setattr(runner, "RUNTIME_ROOT", runtime)
    monkeypatch.setattr(runner, "LOCK_PATH", runtime / "campaign.lock")
    monkeypatch.setattr(runner, "TERMINAL_PATH", terminal_path)
    monkeypatch.setattr(
        runner, "validate_authority", lambda **_kwargs: (plan, authorization)
    )
    monkeypatch.setattr(
        runner,
        "preflight_campaign_lifecycle",
        lambda **_kwargs: {"status": "passed"},
    )
    monkeypatch.setattr(
        runner.serial_runtime,
        "exclusive_campaign_lock",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        runner,
        "validate_terminal_matrix",
        lambda **_kwargs: terminal,
    )
    monkeypatch.setattr(runner, "write_status", lambda payload: statuses.append(payload))

    assert runner._run_campaign_impl() == 0
    assert statuses == [
        {
            "status": "passed_all6_append_then_plateau_k50",
            "terminal_sha256": terminal["sha256"],
        }
    ]

@pytest.mark.parametrize(
    ("state_name", "source", "retiring", "durable", "expected_action"),
    [
        ("empty", False, False, (0, 0, 0, 0, 0), "launch"),
        ("direct_unarchived", True, False, (0, 0, 0, 0, 0), "prepare_archive"),
        (
            "archive_published_pending_manifest",
            True,
            False,
            (1, 0, 0, 0, 0),
            "resume_archive",
        ),
        (
            "manifest_published_pending_closure",
            True,
            False,
            (1, 1, 0, 0, 0),
            "resume_archive",
        ),
        (
            "closure_published_pending_intent",
            True,
            False,
            (1, 1, 1, 0, 0),
            "publish_rotation_intent",
        ),
        (
            "intent_published_pending_rename",
            True,
            False,
            (1, 1, 1, 1, 0),
            "complete_rotation",
        ),
        (
            "retiring_pending_removal",
            False,
            True,
            (1, 1, 1, 1, 0),
            "complete_rotation",
        ),
        (
            "cleanup_receipt_pending",
            False,
            False,
            (1, 1, 1, 1, 0),
            "complete_rotation",
        ),
        (
            "archived_closed",
            False,
            False,
            (1, 1, 1, 1, 1),
            "validate_archived",
        ),
    ],
)
def test_all_nine_archive_restart_states_dispatch_fail_closed(
    tmp_path: Path,
    state_name: str,
    source: bool,
    retiring: bool,
    durable,
    expected_action: str,
) -> None:
    runner = _load_runner()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    paths = runner.strict_archive.CellArchivePaths(
        runtime_root=runtime,
        execution_id="cell",
    )
    if source:
        paths.source_root.mkdir(parents=True)
    if retiring:
        paths.retiring_root.mkdir(parents=True)
    durable_paths = (
        paths.archive_path,
        paths.archive_manifest_path,
        paths.archive_closure_path,
        paths.rotation_intent_path,
        paths.cleanup_receipt_path,
    )
    for present, path in zip(durable, durable_paths, strict=True):
        if present:
            path.parent.mkdir()
            path.touch()
    observed = runner.strict_archive.inspect_rotation_state(paths)
    assert observed["state"] == state_name
    assert runner.archive_restart_action(
        observed,
        archive_rotation_authorized=True,
    ) == expected_action
    if expected_action == "complete_rotation":
        assert runner.archive_restart_action(
            observed,
            archive_rotation_authorized=False,
        ) == "blocked_missing_rotation_authority"
