"""The durable outer state machine for Paper-II and SIAM acquisition."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

import pipelines.time_dynamics.acquisition_supervisor as supervisor_module

from pipelines.artifacts.ingestion import TransactionSpec, run_ingestion_transaction
from pipelines.time_dynamics.acquisition_supervisor import (
    LANE_SCIENTIFIC_AUTHORITY_SCHEMA_V1,
    RAW_SCHEDULER_EVIDENCE_SCHEMA_V1,
    RECONCILIATION_INDETERMINATE,
    SCOPE_AUTHORITY_SCHEMA_V1,
    CampaignRegistry,
    CycleGuardFailed,
    ArtifactBinding,
    LaneHandlers,
    LaneRegistration,
    ProductionOperationsBinding,
    REGISTRY_SCHEMA_V1,
    ReconciliationResult,
    SNAPSHOT_SCHEMA_V1,
    SchedulerScope,
    SchedulerSnapshot,
    SubmissionPlan,
    SubmissionUnit,
    SupervisorLock,
    SupervisorLocked,
    main,
    load_operations_factory,
    run_acquisition_cycle,
    run_production_cycle,
    run_replay_cycle,
)


def _lane(tmp_path: Path, *, enabled: bool = True, submit: bool = True) -> LaneRegistration:
    return LaneRegistration(
        lane_id="paper",
        kind="paper_ii_staged_accuracy",
        monitor_enabled=enabled,
        submission_enabled=submit,
        attempts_root=tmp_path / "attempts",
        state_root=tmp_path / "paper_state",
        scientific_state=tmp_path / "ledger.json",
        report_contract=tmp_path / "report.json",
        planner_authority="StagedCampaign.next_actions",
        blocked_by=() if enabled else ("author_decision",),
    )


def _registry(tmp_path: Path, lane: LaneRegistration | None = None) -> CampaignRegistry:
    registry = CampaignRegistry(
        registry_id="unit_registry",
        repository_root=tmp_path,
        supervisor_state_root=tmp_path / "supervisor",
        lanes=(lane or _lane(tmp_path),),
    )
    for item in registry.lanes:
        for path in (item.scientific_state, item.report_contract):
            if path is not None and not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n")
    return registry


def _production_registry(
    tmp_path: Path,
    *,
    scopes: tuple[SchedulerScope, ...] = (
        SchedulerScope(lane_id="paper", cluster_id=42, process_ids=(0,)),
    ),
    lane: LaneRegistration | None = None,
    adapter_id: str = "test_operations_v1",
    factory: str = "test_acquisition_operations_factory:build_operations",
) -> CampaignRegistry:
    source = tmp_path / "operations_adapter.py"
    source.write_text("# immutable test operations adapter\n")
    authority = tmp_path / "scheduler_scope_authority.json"
    authority.write_text(
        json.dumps(
            {
                "schema": SCOPE_AUTHORITY_SCHEMA_V1,
                "registry_id": "unit_registry",
                "scopes": [scope.to_json_dict() for scope in scopes],
            },
            sort_keys=True,
        )
        + "\n"
    )
    registry = _registry(tmp_path, lane)
    return CampaignRegistry(
        registry_id=registry.registry_id,
        repository_root=registry.repository_root,
        supervisor_state_root=registry.supervisor_state_root,
        lanes=registry.lanes,
        production_operations=ProductionOperationsBinding(
            adapter_id=adapter_id,
            factory=factory,
            source_path=source,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            bootstrap_scopes_path=authority,
            bootstrap_scopes_sha256=hashlib.sha256(authority.read_bytes()).hexdigest(),
        ),
    )


def _snapshot(*, state: str = "succeeded", exit_code: int | None = 0) -> SchedulerSnapshot:
    return SchedulerSnapshot.from_mapping(
        {
            "schema": SNAPSHOT_SCHEMA_V1,
            "captured_at": "2026-08-28T12:00:00Z",
            "source": {
                "queue_query_succeeded": True,
                "history_query_succeeded": True,
                "scope_complete": True,
                "expected_processes": [
                    {"lane_id": "paper", "cluster_id": 42, "process_ids": [0]}
                ],
            },
            "jobs": [
                {
                    "lane_id": "paper",
                    "cluster_id": 42,
                    "process_id": 0,
                    "state": state,
                    "exit_code": exit_code,
                    "action_id": "action-0",
                }
            ],
        }
    )


def _snapshot_rows(
    rows: list[dict], *, captured_at: str = "2026-08-28T12:00:00Z"
) -> SchedulerSnapshot:
    scopes: dict[tuple[str, int], list[int]] = {}
    for row in rows:
        scopes.setdefault((row["lane_id"], int(row["cluster_id"])), []).append(
            int(row["process_id"])
        )
    return SchedulerSnapshot.from_mapping(
        {
            "schema": SNAPSHOT_SCHEMA_V1,
            "captured_at": captured_at,
            "source": {
                "queue_query_succeeded": True,
                "history_query_succeeded": True,
                "scope_complete": True,
                "expected_processes": [
                    {
                        "lane_id": lane_id,
                        "cluster_id": cluster_id,
                        "process_ids": sorted(process_ids),
                    }
                    for (lane_id, cluster_id), process_ids in sorted(scopes.items())
                ],
            },
            "jobs": rows,
        }
    )


def _production_snapshot(
    *,
    repository_root: Path | None = None,
    adapter_id: str = "test_operations_v1",
    process_id: int = 0,
    captured_at: str = "2026-08-28T12:00:00Z",
    extra_jobs: tuple[dict, ...] = (),
) -> SchedulerSnapshot:
    evidence: dict[str, dict[str, str]] = {}
    for query, fallback in (("queue", "a" * 64), ("history", "b" * 64)):
        reference = Path("scheduler") / f"{query}.raw"
        if repository_root is None:
            digest = fallback
        else:
            target = repository_root / reference
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"raw {query} evidence\n")
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
        evidence[query] = {"reference": str(reference), "sha256": digest}
    jobs = [
        {
            "lane_id": "paper",
            "cluster_id": 42,
            "process_id": process_id,
            "state": "running",
            "exit_code": None,
            "action_id": f"action-{process_id}",
        },
        *extra_jobs,
    ]
    scopes: dict[tuple[str, int], list[int]] = {}
    for row in jobs:
        scopes.setdefault((row["lane_id"], int(row["cluster_id"])), []).append(
            int(row["process_id"])
        )
    return SchedulerSnapshot.from_mapping(
        {
            "schema": SNAPSHOT_SCHEMA_V1,
            "captured_at": captured_at,
            "source": {
                "queue_query_succeeded": True,
                "history_query_succeeded": True,
                "scope_complete": True,
                "adapter_id": adapter_id,
                "expected_processes": [
                    {
                        "lane_id": lane_id,
                        "cluster_id": cluster_id,
                        "process_ids": sorted(process_ids),
                    }
                    for (lane_id, cluster_id), process_ids in sorted(scopes.items())
                ],
                "raw_scheduler_evidence": {
                    "schema": RAW_SCHEDULER_EVIDENCE_SCHEMA_V1,
                    **evidence,
                },
            },
            "jobs": jobs,
        }
    )


def _write_increment_receipt(lane: LaneRegistration) -> Path:
    path = (
        lane.attempts_root
        / "cluster_42/increments/proc0/evidence/increment_receipt.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    run = path.parents[1] / "extracted/raw_outputs/action-0/run.json"
    run.parent.mkdir(parents=True, exist_ok=True)
    run.write_text('{"action_id":"action-0"}\n')
    run_sha = hashlib.sha256(run.read_bytes()).hexdigest()
    path.write_text(
        json.dumps(
            {
                "cluster_id": 42,
                "process_ids": [0],
                "status": "INGESTED",
                "action_dispositions": [
                    {
                        "process_id": 0,
                        "action_id": "action-0",
                        "run_path": str(run),
                        "run_sha256": run_sha,
                        "status": "recorded",
                    }
                ],
            }
        )
    )
    return path


def _binding(path: Path) -> ArtifactBinding:
    return ArtifactBinding(
        path=path, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )


def _refresh(lane: LaneRegistration, *, status: str = "refreshed") -> dict:
    assert lane.report_contract is not None
    return {
        "status": status,
        "report_contract": _binding(lane.report_contract).to_json_dict(),
    }


def _plan(
    lane: LaneRegistration, units: tuple[SubmissionUnit, ...] | None = None
) -> SubmissionPlan:
    assert lane.scientific_state is not None
    assert lane.report_contract is not None
    return SubmissionPlan(
        units=(_unit(),) if units is None else units,
        planner_authority=lane.planner_authority,
        scientific_state=_binding(lane.scientific_state),
        report_contract=_binding(lane.report_contract),
    )


def _unit() -> SubmissionUnit:
    return SubmissionUnit(
        unit_id="frontier-1", action_ids=("next-action",), payload={"epoch": "e1"}
    )


def test_atomic_json_write_fsyncs_the_containing_directory(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "state" / "receipt.json"
    real_open = supervisor_module.os.open
    real_fsync = supervisor_module.os.fsync
    directory_fds: set[int] = set()
    fsynced: list[int] = []

    def open_spy(path, flags, *args):
        descriptor = real_open(path, flags, *args)
        if Path(path) == target.parent:
            directory_fds.add(descriptor)
        return descriptor

    def fsync_spy(descriptor):
        fsynced.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch.setattr(supervisor_module.os, "open", open_spy)
    monkeypatch.setattr(supervisor_module.os, "fsync", fsync_spy)
    supervisor_module._atomic_write_json(target, {"status": "durable"})

    assert json.loads(target.read_text()) == {"status": "durable"}
    assert directory_fds
    assert directory_fds.issubset(fsynced)


def test_registry_file_freezes_active_and_disabled_lanes(tmp_path: Path) -> None:
    path = tmp_path / "pipelines/time_dynamics/campaigns/registry.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema": REGISTRY_SCHEMA_V1,
                "registry_id": "r",
                "supervisor_state_root": "output/supervisor",
                "lanes": [
                    {
                        "lane_id": "active",
                        "kind": "paper",
                        "monitor_enabled": True,
                        "submission_enabled": True,
                        "attempts_root": "output/a/attempts",
                        "state_root": "output/a/state",
                        "scientific_state": "output/a/ledger.json",
                        "report_contract": "output/a/report.json",
                        "planner_authority": "planner.next_actions",
                        "blocked_by": [],
                    },
                    {
                        "lane_id": "future",
                        "kind": "nph3",
                        "monitor_enabled": False,
                        "submission_enabled": False,
                        "attempts_root": "output/b/attempts",
                        "state_root": "output/b/state",
                        "blocked_by": ["author_matrix"],
                    },
                ],
            }
        )
    )
    registry = CampaignRegistry.read(path, repository_root=tmp_path)
    assert registry.lane("active").submission_enabled is True
    assert registry.lane("future").monitor_enabled is False
    assert registry.lane("future").blocked_by == ("author_matrix",)


def _portable_registry_payload() -> dict:
    return {
        "schema": REGISTRY_SCHEMA_V1,
        "registry_id": "portable",
        "supervisor_state_root": "output/supervisor",
        "production_operations": None,
        "lanes": [
            {
                "lane_id": "future",
                "kind": "paper",
                "monitor_enabled": False,
                "submission_enabled": False,
                "attempts_root": "output/future/attempts",
                "state_root": "output/future/state",
                "scientific_state": "output/future/scientific.json",
                "report_contract": "output/future/report.json",
                "planner_authority": "planner.next_actions",
                "blocked_by": ["transport_binding"],
            }
        ],
        "scientific_contracts": {
            "schema": "paper_ii_acquisition_contract_bindings_v1",
            "matrix": {"raw_sha256": "a" * 64},
        },
    }


def _scientific_authority_registry_payload() -> dict:
    payload = _portable_registry_payload()
    lane = payload["lanes"][0]
    lane["report_contract"] = "output/future/operational-report.json"
    lane["metadata"] = {
        "planned_report_contract_file_sha256": "e" * 64,
        "planned_report_contract_sha256": "f" * 64,
        "matrix_raw_sha256": "a" * 64,
        "matrix_payload_sha256": "b" * 64,
        "matrix_expanded_semantic_sha256": "c" * 64,
        "matrix_accuracy_policy_sha256": "d" * 64,
    }
    payload["scientific_contracts"] = {
        "schema": "paper_ii_acquisition_contract_bindings_v1",
        "matrix": {
            "raw_sha256": "a" * 64,
            "payload_sha256": "b" * 64,
            "expanded_semantic_sha256": "c" * 64,
            "accuracy_policy_spec_sha256": "d" * 64,
        },
        "planned_report_contracts": {
            "future": {
                "path": "output/future/planned-matrix-report.json",
                "file_sha256": "e" * 64,
                "report_contract_sha256": "f" * 64,
                "tuple_structure_sha256": "1" * 64,
            }
        },
    }
    return payload


def _read_registry_payload(tmp_path: Path, payload: dict) -> CampaignRegistry:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload))
    return CampaignRegistry.read(path, repository_root=tmp_path)


def test_lane_scientific_authority_retains_operational_and_planned_paths(
    tmp_path: Path,
) -> None:
    registry = _read_registry_payload(
        tmp_path, _scientific_authority_registry_payload()
    )
    authority = registry.lane_scientific_authority("future")
    row = authority.to_json_dict()

    assert row["schema"] == LANE_SCIENTIFIC_AUTHORITY_SCHEMA_V1
    assert authority.registry_id == "portable"
    assert authority.lane_id == "future"
    assert authority.planner_authority == "planner.next_actions"
    assert authority.scientific_state_path == (
        "output/future/scientific.json"
    )
    assert authority.report_contract_path == (
        "output/future/operational-report.json"
    )
    assert authority.planned_report_contract_path == (
        "output/future/planned-matrix-report.json"
    )
    assert (
        authority.report_contract_path
        != authority.planned_report_contract_path
    )
    assert authority.planned_report_contract_file_sha256 == "e" * 64
    assert authority.planned_report_contract_sha256 == "f" * 64
    assert authority.planned_report_tuple_structure_sha256 == "1" * 64
    assert authority.matrix_raw_sha256 == "a" * 64
    assert authority.matrix_payload_sha256 == "b" * 64
    assert authority.matrix_expanded_semantic_sha256 == "c" * 64
    assert authority.matrix_accuracy_policy_sha256 == "d" * 64

    unsigned = dict(row)
    declared = unsigned.pop("authority_sha256")
    assert declared == hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    assert {
        "monitor_enabled",
        "submission_enabled",
        "blocked_by",
        "production_operations",
    }.isdisjoint(row)


def test_lane_scientific_authority_excludes_mutable_control_plane(
    tmp_path: Path,
) -> None:
    baseline_payload = _scientific_authority_registry_payload()
    baseline = _read_registry_payload(tmp_path / "baseline", baseline_payload)

    activated_payload = deepcopy(baseline_payload)
    activated_lane = activated_payload["lanes"][0]
    activated_lane["monitor_enabled"] = True
    activated_lane["blocked_by"] = []
    activated_payload["production_operations_blocked"] = {
        "status": "blocked",
        "adapter_id": "different-control-adapter",
        "factory": "different.module:build_operations",
        "source": {
            "path": "pipelines/different_adapter.py",
            "sha256": "9" * 64,
        },
        "blocked_by": ["transport"],
    }
    activated = _read_registry_payload(tmp_path / "activated", activated_payload)

    assert baseline.digest != activated.digest
    assert (
        baseline.lane_scientific_authority("future").authority_sha256
        == activated.lane_scientific_authority("future").authority_sha256
    )


def test_canonical_lane_authorities_keep_siam_report_paths_distinct() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = CampaignRegistry.read(
        root
        / "pipelines/time_dynamics/campaigns/"
        "time_dynamics_acquisition_registry_v1.json",
        repository_root=root,
    )

    siam = registry.lane_scientific_authority("siam_nb4_pilot_v1")
    assert siam.report_contract_path == (
        "output/siam_screening/nb4_pilot_v1/acquisition_supervisor/"
        "report_contract_current.json"
    )
    assert siam.planned_report_contract_path == (
        "output/siam_screening/nb4_pilot_v1/acquisition_supervisor/"
        "planned_matrix_report_contract_v1.json"
    )
    assert siam.report_contract_path != siam.planned_report_contract_path

    for lane_id in (
        "paper_ii_hh_product_v1",
        "paper_ii_hubbard_l4_scale_v1",
    ):
        authority = registry.lane_scientific_authority(lane_id)
        assert (
            authority.report_contract_path
            == authority.planned_report_contract_path
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda payload: payload["lanes"][0].pop("scientific_state"),
            "scientific state",
        ),
        (
            lambda payload: payload["lanes"][0].pop("report_contract"),
            "report contract",
        ),
        (
            lambda payload: payload["lanes"][0].update(
                {"planner_authority": ""}
            ),
            "registry/planner identity",
        ),
        (
            lambda payload: payload["scientific_contracts"][
                "planned_report_contracts"
            ].pop("future"),
            "planned report contract",
        ),
        (
            lambda payload: payload["scientific_contracts"]["matrix"].update(
                {"raw_sha256": "not-a-digest"}
            ),
            "matrix raw digest",
        ),
        (
            lambda payload: payload["scientific_contracts"][
                "planned_report_contracts"
            ]["future"].update({"lane_id": "another-lane"}),
            "crosses lanes",
        ),
        (
            lambda payload: payload["scientific_contracts"][
                "planned_report_contracts"
            ]["future"].update({"path": "../outside/report.json"}),
            "planned report contract path",
        ),
    ),
)
def test_lane_scientific_authority_rejects_incomplete_or_cross_lane_contracts(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    payload = _scientific_authority_registry_payload()
    mutation(payload)
    registry = _read_registry_payload(tmp_path, payload)
    with pytest.raises(ValueError, match=message):
        registry.lane_scientific_authority("future")


def test_lane_scientific_authority_rejects_lane_path_escaping_root(
    tmp_path: Path,
) -> None:
    payload = _scientific_authority_registry_payload()
    parsed = _read_registry_payload(tmp_path, payload)
    lane = parsed.lane("future")
    outside_lane = LaneRegistration(
        lane_id=lane.lane_id,
        kind=lane.kind,
        monitor_enabled=lane.monitor_enabled,
        submission_enabled=lane.submission_enabled,
        attempts_root=lane.attempts_root,
        state_root=lane.state_root,
        scientific_state=tmp_path.parent / "outside-scientific-state.json",
        report_contract=lane.report_contract,
        planner_authority=lane.planner_authority,
        blocked_by=lane.blocked_by,
        metadata=lane.metadata,
    )
    registry = CampaignRegistry(
        registry_id=parsed.registry_id,
        repository_root=parsed.repository_root,
        supervisor_state_root=parsed.supervisor_state_root,
        lanes=(outside_lane,),
        scientific_contracts=parsed.scientific_contracts,
    )
    with pytest.raises(ValueError, match="outside repository_root"):
        registry.lane_scientific_authority("future")


def test_registry_parses_scientific_contracts_into_its_semantic_digest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registry.json"
    payload = _portable_registry_payload()
    path.write_text(json.dumps(payload))
    first = CampaignRegistry.read(path, repository_root=tmp_path)
    assert first.scientific_contracts == payload["scientific_contracts"]

    payload["scientific_contracts"]["matrix"]["raw_sha256"] = "b" * 64
    path.write_text(json.dumps(payload))
    second = CampaignRegistry.read(path, repository_root=tmp_path)
    assert first.digest != second.digest


def test_registry_digest_is_checkout_independent(tmp_path: Path) -> None:
    payload = _portable_registry_payload()
    roots = (tmp_path / "checkout-a", tmp_path / "checkout-b")
    registries = []
    for root in roots:
        root.mkdir()
        path = root / "registry.json"
        path.write_text(json.dumps(payload))
        registries.append(CampaignRegistry.read(path, repository_root=root))
    assert registries[0].digest == registries[1].digest


def test_registry_rejects_nonobject_scientific_contracts(tmp_path: Path) -> None:
    payload = _portable_registry_payload()
    payload["scientific_contracts"] = []
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="scientific_contracts"):
        CampaignRegistry.read(path, repository_root=tmp_path)


def test_registry_digest_rejects_path_outside_repository(tmp_path: Path) -> None:
    registry = CampaignRegistry(
        registry_id="outside",
        repository_root=tmp_path,
        supervisor_state_root=tmp_path.parent / "outside-supervisor",
        lanes=(_lane(tmp_path, enabled=False, submit=False),),
    )
    with pytest.raises(ValueError, match="outside repository_root"):
        _ = registry.digest


def test_registry_rejects_launch_while_scientific_blockers_remain(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "schema": REGISTRY_SCHEMA_V1,
                "registry_id": "r",
                "supervisor_state_root": "output/supervisor",
                "lanes": [
                    {
                        "lane_id": "unsafe",
                        "kind": "nph3",
                        "monitor_enabled": True,
                        "submission_enabled": True,
                        "attempts_root": "output/a",
                        "state_root": "output/b",
                        "blocked_by": ["missing_seed"],
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="blocked_by"):
        CampaignRegistry.read(path, repository_root=tmp_path)


def test_one_cycle_ingests_reports_and_submits_exactly_once(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    lane = registry.lane("paper")
    calls = {"ingest": 0, "refresh": 0, "reconcile": 0, "submit": 0}

    def ingest(_lane, cluster, processes, jobs):
        calls["ingest"] += 1
        assert cluster == 42 and processes == (0,)
        assert jobs[0].action_id == "action-0"
        receipt = _write_increment_receipt(lane)
        return {"status": "ingested", "receipt": str(receipt)}

    def refresh(_lane, _snapshot, context):
        calls["refresh"] += 1
        if context["phase"] == "evidence_refresh":
            assert context["ingestion"][0]["status"] == "ingested"
        else:
            assert context["submissions"][0]["status"] == "submitted"
        return _refresh(lane)

    def reconcile(_lane, _unit, _snapshot):
        calls["reconcile"] += 1
        return ReconciliationResult.verified_absent()

    def submit(_lane, _unit):
        calls["submit"] += 1
        return {"cluster_id": 43, "action_ids": list(_unit.action_ids)}

    handlers = {
        "paper": LaneHandlers(
            ingest_delta=ingest,
            refresh_status=refresh,
            plan_submissions=lambda _lane, _snapshot, _context: _plan(_lane),
            reconcile_submission=reconcile,
            submit_unit=submit,
        )
    }
    first = run_acquisition_cycle(
        registry=registry, snapshot=_snapshot(), handlers=handlers
    )
    assert first.status == "complete"
    assert calls == {"ingest": 1, "refresh": 2, "reconcile": 1, "submit": 1}
    assert first.receipt_path.is_file()

    second = run_acquisition_cycle(
        registry=registry, snapshot=_snapshot(), handlers=handlers
    )
    assert second.status == "noop"
    assert calls == {"ingest": 1, "refresh": 2, "reconcile": 1, "submit": 1}
    assert second.receipt_path != first.receipt_path


def test_blocked_ingestion_resumes_without_repeating_completed_work(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    lane = registry.lane("paper")
    calls = {"ingest": 0, "refresh": 0, "submit": 0}

    def ingest(_lane, _cluster, _processes, _jobs):
        calls["ingest"] += 1
        if calls["ingest"] == 1:
            raise CycleGuardFailed("free_space_floor", "8 GiB < 10 GiB")
        _write_increment_receipt(lane)
        return {"status": "ingested"}

    def refresh(_lane, _snapshot, context):
        calls["refresh"] += 1
        if context["phase"] == "evidence_refresh":
            assert context["gate_pending"] is (calls["ingest"] == 1)
        return _refresh(lane)

    def submit(_lane, _unit):
        calls["submit"] += 1
        return {"cluster_id": 99, "action_ids": list(_unit.action_ids)}

    handlers = {
        "paper": LaneHandlers(
            ingest_delta=ingest,
            refresh_status=refresh,
            plan_submissions=lambda _lane, _snapshot, _context: _plan(_lane),
            reconcile_submission=lambda _lane, _unit, _snapshot: (
                ReconciliationResult.verified_absent()
            ),
            submit_unit=submit,
        )
    }
    first = run_acquisition_cycle(
        registry=registry, snapshot=_snapshot(), handlers=handlers
    )
    assert first.status == "blocked"
    assert calls == {"ingest": 1, "refresh": 1, "submit": 0}

    second = run_acquisition_cycle(
        registry=registry, snapshot=_snapshot(), handlers=handlers
    )
    assert second.status == "complete"
    assert calls == {"ingest": 2, "refresh": 3, "submit": 1}


def test_report_failure_is_retried_after_ingestion_receipt_exists(tmp_path: Path) -> None:
    registry = _registry(tmp_path, _lane(tmp_path, submit=False))
    lane = registry.lane("paper")
    calls = {"ingest": 0, "refresh": 0}

    def ingest(_lane, _cluster, _processes, _jobs):
        calls["ingest"] += 1
        _write_increment_receipt(lane)
        return {"status": "ingested"}

    def refresh(_lane, _snapshot, _context):
        calls["refresh"] += 1
        if calls["refresh"] == 1:
            raise CycleGuardFailed("latex_build_failed")
        return _refresh(lane)

    handlers = {
        "paper": LaneHandlers(ingest_delta=ingest, refresh_status=refresh)
    }
    first = run_acquisition_cycle(
        registry=registry, snapshot=_snapshot(), handlers=handlers
    )
    second = run_acquisition_cycle(
        registry=registry, snapshot=_snapshot(), handlers=handlers
    )
    assert first.status == "blocked"
    assert second.status == "complete"
    assert calls == {"ingest": 1, "refresh": 2}


def test_external_receipt_appearance_forces_same_snapshot_report_refresh(
    tmp_path: Path,
) -> None:
    lane = _lane(tmp_path, submit=False)
    registry = _registry(tmp_path, lane)
    calls = {"refresh": 0}

    def refresh(item, *_args):
        calls["refresh"] += 1
        return _refresh(item)

    first = run_acquisition_cycle(
        registry=registry,
        snapshot=_snapshot(),
        handlers={
            "paper": LaneHandlers(
                ingest_delta=lambda *_args: (_ for _ in ()).throw(
                    CycleGuardFailed("transfer_pending")
                ),
                refresh_status=refresh,
            )
        },
    )
    assert first.status == "blocked"
    assert calls["refresh"] == 1

    _write_increment_receipt(lane)

    def must_not_ingest(*_args):
        pytest.fail("externally recorded coverage must not be re-ingested")

    second = run_acquisition_cycle(
        registry=registry,
        snapshot=_snapshot(),
        handlers={
            "paper": LaneHandlers(
                ingest_delta=must_not_ingest,
                refresh_status=refresh,
            )
        },
    )
    assert second.status == "complete"
    assert calls["refresh"] == 2
    lane_result = second.lane_results["paper"]
    assert lane_result["durable_coverage_changed"] is True
    assert lane_result["report_refresh"]["status"] == "refreshed"


def test_scheduler_accepted_submission_is_recovered_after_local_write_failure(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    lane = registry.lane("paper")
    _write_increment_receipt(lane)
    accepted: dict[str, object] = {}
    calls = {"submit": 0, "reconcile": 0}

    def reconcile(_lane, unit, _snapshot):
        calls["reconcile"] += 1
        remote = accepted.get(unit.unit_id)
        return (
            ReconciliationResult.found(remote)  # type: ignore[arg-type]
            if remote is not None
            else ReconciliationResult.verified_absent()
        )

    def submit_then_lose_receipt(_lane, unit):
        calls["submit"] += 1
        accepted[unit.unit_id] = {"cluster_id": 77, "action_ids": list(unit.action_ids)}
        raise CycleGuardFailed("local_submission_receipt_write_interrupted")

    first_handlers = {
        "paper": LaneHandlers(
            refresh_status=lambda lane, *_args: _refresh(lane),
            plan_submissions=lambda lane, _snapshot, _context: _plan(lane),
            reconcile_submission=reconcile,
            submit_unit=submit_then_lose_receipt,
        )
    }
    first = run_acquisition_cycle(
        registry=registry, snapshot=_snapshot(), handlers=first_handlers
    )
    assert first.status == "blocked"
    assert calls == {"submit": 1, "reconcile": 1}

    def must_not_resubmit(_lane, _unit):
        pytest.fail("an accepted submission must be reconciled, not resubmitted")

    replay_handlers = {
        "paper": LaneHandlers(
            refresh_status=lambda lane, *_args: _refresh(lane),
            plan_submissions=lambda lane, _snapshot, _context: _plan(
                lane, units=()
            ),
            reconcile_submission=reconcile,
            submit_unit=must_not_resubmit,
        )
    }
    replay = run_acquisition_cycle(
        registry=registry, snapshot=_snapshot(), handlers=replay_handlers
    )
    assert replay.status == "complete"
    assert calls == {"submit": 1, "reconcile": 2}
    submission = replay.lane_results["paper"]["submissions"][0]
    assert submission["status"] == "recovered"


def test_preexisting_unresolved_intent_is_never_resubmitted_on_empty_lookup(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    lane = registry.lane("paper")
    _write_increment_receipt(lane)
    calls = {"submit": 0}

    def submit_then_lose_receipt(_lane, _unit):
        calls["submit"] += 1
        raise CycleGuardFailed("local_submission_receipt_write_interrupted")

    first = run_acquisition_cycle(
        registry=registry,
        snapshot=_snapshot(),
        handlers={
            "paper": LaneHandlers(
                refresh_status=lambda item, *_args: _refresh(item),
                plan_submissions=lambda item, _snapshot, _context: _plan(item),
                reconcile_submission=lambda *_args: (
                    ReconciliationResult.verified_absent()
                ),
                submit_unit=submit_then_lose_receipt,
            )
        },
    )
    assert first.status == "blocked"
    assert calls["submit"] == 1

    def must_not_resubmit(*_args):
        pytest.fail("an unresolved pre-existing intent must not be resubmitted")

    replay = run_acquisition_cycle(
        registry=registry,
        snapshot=_snapshot(),
        handlers={
            "paper": LaneHandlers(
                refresh_status=lambda item, *_args: _refresh(item),
                plan_submissions=lambda item, _snapshot, _context: _plan(item),
                reconcile_submission=lambda *_args: ReconciliationResult.indeterminate(
                    "queue/history lookup was inconclusive"
                ),
                submit_unit=must_not_resubmit,
            )
        },
    )
    assert replay.status == "blocked"
    assert calls["submit"] == 1
    assert replay.lane_results["paper"]["submissions"][0]["guard"] == (
        "submission_identity_unresolved"
    )


def test_indeterminate_reconciliation_never_reaches_submission(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    lane = registry.lane("paper")
    _write_increment_receipt(lane)
    submitted = {"count": 0}

    result = run_acquisition_cycle(
        registry=registry,
        snapshot=_snapshot(),
        handlers={
            "paper": LaneHandlers(
                refresh_status=lambda item, *_args: _refresh(item),
                plan_submissions=lambda item, _snapshot, _context: _plan(item),
                reconcile_submission=lambda _lane, _unit, snapshot: (
                    ReconciliationResult.indeterminate(
                        f"snapshot {snapshot.digest} did not prove absence"
                    )
                ),
                submit_unit=lambda *_args: submitted.__setitem__("count", 1),
            )
        },
    )

    assert result.status == "blocked"
    assert submitted["count"] == 0
    guard = result.lane_results["paper"]["submissions"][0]
    assert guard["guard"] == "submission_identity_unresolved"
    assert RECONCILIATION_INDETERMINATE in guard["detail"]


def test_siam_singular_process_receipt_is_validated_as_covered(
    tmp_path: Path,
) -> None:
    lane = _lane(tmp_path, submit=False)
    registry = _registry(tmp_path, lane)
    receipt = (
        lane.attempts_root
        / "cluster_42/increments/proc0/evidence/increment_receipt.json"
    )
    receipt.parent.mkdir(parents=True)
    run = receipt.parents[1] / "extracted/raw_outputs/action-0/run.json"
    run.parent.mkdir(parents=True)
    run.write_text('{"action_id":"action-0"}\n')
    receipt.write_text(
        json.dumps(
            {
                "cluster_id": 42,
                "process_id": 0,
                "action_id": "action-0",
                "status": "INGESTED_RESULTS_REFRESHED",
                "run": {
                    "path": str(run),
                    "sha256": hashlib.sha256(run.read_bytes()).hexdigest(),
                },
            }
        )
    )

    def must_not_ingest(*_args):
        pytest.fail("verified SIAM coverage must not be re-ingested")

    result = run_acquisition_cycle(
        registry=registry,
        snapshot=_snapshot(),
        handlers={
            "paper": LaneHandlers(
                ingest_delta=must_not_ingest,
                refresh_status=lambda item, *_args: _refresh(item),
            )
        },
    )
    assert result.status == "complete"
    assert len(result.lane_results["paper"]["durable_coverage"]) == 1


def test_real_engine_v3_paper_receipt_replays_as_verified_coverage(
    tmp_path: Path,
) -> None:
    lane = _lane(tmp_path, submit=False)
    registry = _registry(tmp_path, lane)
    assert lane.scientific_state is not None
    spec = TransactionSpec(
        campaign_id="paper",
        cluster_id=42,
        process_ids=(0,),
        increment_root=lane.attempts_root / "cluster_42/increments/proc0",
        owner="test",
    )
    run_path = (
        spec.evidence_dir / "extracted/raw_outputs/action-0/run.json"
    )
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text('{"action_id":"action-0"}\n', encoding="utf-8")
    run_sha256 = hashlib.sha256(run_path.read_bytes()).hexdigest()
    ledger_sha256 = hashlib.sha256(lane.scientific_state.read_bytes()).hexdigest()
    transaction = run_ingestion_transaction(
        spec,
        {
            "ledger_append": lambda _context: {
                "action_dispositions": [
                    {
                        "process_id": 0,
                        "action_id": "action-0",
                        "run_path": str(run_path),
                        "run_sha256": run_sha256,
                        "status": "recorded",
                    }
                ],
                "scientific_state_after": {
                    "path": str(lane.scientific_state),
                    "sha256": ledger_sha256,
                },
                "ledger_after_sha256": ledger_sha256,
            }
        },
    )
    assert transaction.status == "ingested"
    called = {"ingest": 0}

    result = run_acquisition_cycle(
        registry=registry,
        snapshot=_snapshot(),
        handlers={
            "paper": LaneHandlers(
                ingest_delta=lambda *_args: called.__setitem__("ingest", 1),
                refresh_status=lambda item, *_args: _refresh(item),
            )
        },
    )

    assert result.status == "complete"
    assert called["ingest"] == 0
    assert result.lane_results["paper"]["durable_coverage"][0] == {
        "receipt": str(transaction.receipt_path),
        "action_id": "action-0",
        "run_path": str(run_path),
        "run_sha256": run_sha256,
    }

    lane.scientific_state.write_text('{"generation":2}\n', encoding="utf-8")
    second_state_sha256 = hashlib.sha256(
        lane.scientific_state.read_bytes()
    ).hexdigest()
    second_spec = TransactionSpec(
        campaign_id="paper",
        cluster_id=43,
        process_ids=(0,),
        increment_root=lane.attempts_root / "cluster_43/increments/proc0",
        owner="test",
    )
    second_run = (
        second_spec.evidence_dir / "extracted/raw_outputs/action-1/run.json"
    )
    second_run.parent.mkdir(parents=True, exist_ok=True)
    second_run.write_text('{"action_id":"action-1"}\n', encoding="utf-8")
    second_run_sha256 = hashlib.sha256(second_run.read_bytes()).hexdigest()
    second_transaction = run_ingestion_transaction(
        second_spec,
        {
            "ledger_append": lambda _context: {
                "action_dispositions": [
                    {
                        "process_id": 0,
                        "action_id": "action-1",
                        "run_path": str(second_run),
                        "run_sha256": second_run_sha256,
                        "status": "recorded",
                    }
                ],
                "scientific_state_after": {
                    "path": str(lane.scientific_state),
                    "sha256": second_state_sha256,
                },
                "ledger_after_sha256": second_state_sha256,
            }
        },
    )
    assert second_transaction.status == "ingested"
    later_snapshot = _snapshot_rows(
        [
            {
                "lane_id": "paper",
                "cluster_id": 42,
                "process_id": 0,
                "state": "succeeded",
                "exit_code": 0,
                "action_id": "action-0",
            },
            {
                "lane_id": "paper",
                "cluster_id": 43,
                "process_id": 0,
                "state": "succeeded",
                "exit_code": 0,
                "action_id": "action-1",
            },
        ],
        captured_at="2026-08-28T12:01:00Z",
    )

    historical = run_acquisition_cycle(
        registry=registry,
        snapshot=later_snapshot,
        handlers={
            "paper": LaneHandlers(
                ingest_delta=lambda *_args: called.__setitem__("ingest", 1),
                refresh_status=lambda item, *_args: _refresh(item),
            )
        },
    )
    assert historical.status == "complete"
    assert called["ingest"] == 0

    corrupted = json.loads(transaction.receipt_path.read_text(encoding="utf-8"))
    corrupted["ledger_append"]["ledger_after_sha256"] = "0" * 64
    transaction.receipt_path.write_text(
        json.dumps(corrupted), encoding="utf-8"
    )
    replay_registry = CampaignRegistry(
        registry_id=registry.registry_id,
        repository_root=registry.repository_root,
        supervisor_state_root=tmp_path / "supervisor_replay",
        lanes=registry.lanes,
    )
    rejected = run_acquisition_cycle(
        registry=replay_registry,
        snapshot=later_snapshot,
        handlers={
            "paper": LaneHandlers(
                ingest_delta=lambda *_args: called.__setitem__("ingest", 1),
                refresh_status=lambda item, *_args: _refresh(item),
            )
        },
    )
    assert rejected.status == "blocked"
    assert called["ingest"] == 0
    assert rejected.lane_results["paper"]["ingestion"][0]["guard"] == (
        "receipt_scientific_state_digest_mismatch"
    )


def test_receipt_action_mismatch_blocks_instead_of_reingesting(tmp_path: Path) -> None:
    lane = _lane(tmp_path, submit=False)
    registry = _registry(tmp_path, lane)
    receipt = _write_increment_receipt(lane)
    payload = json.loads(receipt.read_text())
    payload["action_dispositions"][0]["action_id"] = "different-action"
    receipt.write_text(json.dumps(payload))
    called = {"ingest": 0}

    result = run_acquisition_cycle(
        registry=registry,
        snapshot=_snapshot(),
        handlers={
            "paper": LaneHandlers(
                ingest_delta=lambda *_args: called.__setitem__("ingest", 1),
                refresh_status=lambda item, *_args: _refresh(item),
            )
        },
    )
    assert result.status == "blocked"
    assert called["ingest"] == 0
    assert result.lane_results["paper"]["ingestion"][0]["guard"] == (
        "receipt_action_identity_mismatch"
    )


def test_ingest_success_without_exact_receipt_is_rejected(tmp_path: Path) -> None:
    lane = _lane(tmp_path, submit=False)
    registry = _registry(tmp_path, lane)
    result = run_acquisition_cycle(
        registry=registry,
        snapshot=_snapshot(),
        handlers={
            "paper": LaneHandlers(
                ingest_delta=lambda *_args: {"status": "ingested"},
                refresh_status=lambda item, *_args: _refresh(item),
            )
        },
    )
    assert result.status == "blocked"
    assert result.lane_results["paper"]["ingestion"][0]["guard"] == (
        "ingestion_receipt_missing_or_mismatched"
    )


def test_snapshot_rejects_partial_scope() -> None:
    payload = _snapshot().to_json_dict()
    payload["source"]["expected_processes"][0]["process_ids"].append(1)
    with pytest.raises(ValueError, match="scope-complete"):
        SchedulerSnapshot.from_mapping(payload)


def test_active_job_disappearance_is_durably_blocked(tmp_path: Path) -> None:
    lane = _lane(tmp_path, submit=False)
    registry = _registry(tmp_path, lane)
    handlers = {
        "paper": LaneHandlers(refresh_status=lambda item, *_args: _refresh(item))
    }
    running = _snapshot(state="running", exit_code=None)
    first = run_acquisition_cycle(
        registry=registry, snapshot=running, handlers=handlers
    )
    assert first.status == "complete"
    empty = _snapshot_rows([], captured_at="2026-08-28T12:01:00Z")
    second = run_acquisition_cycle(
        registry=registry, snapshot=empty, handlers=handlers
    )
    assert second.status == "blocked"
    guard = second.lane_results["paper"]["snapshot_validation"]
    assert guard["guard"] == "active_scheduler_jobs_disappeared"
    assert second.receipt_path.is_file()


def test_report_binding_is_required_before_submission(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    submitted = {"count": 0}
    result = run_acquisition_cycle(
        registry=registry,
        snapshot=_snapshot(state="running", exit_code=None),
        handlers={
            "paper": LaneHandlers(
                refresh_status=lambda *_args: {"status": "refreshed"},
                plan_submissions=lambda lane, _snapshot, _context: _plan(lane),
                reconcile_submission=lambda *_args: ReconciliationResult.indeterminate(),
                submit_unit=lambda *_args: submitted.__setitem__("count", 1),
            )
        },
    )
    assert result.status == "blocked"
    assert submitted["count"] == 0
    assert result.lane_results["paper"]["report_refresh"]["guard"] == (
        "report_contract_binding_missing"
    )


def test_untyped_scheduler_reconciliation_blocks_submission(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    lane = registry.lane("paper")
    _write_increment_receipt(lane)
    submitted = {"count": 0}

    def submit(_lane, _unit):
        submitted["count"] += 1
        return {"cluster_id": 1, "action_ids": list(_unit.action_ids)}

    handlers = {
        "paper": LaneHandlers(
            refresh_status=lambda lane, *_args: _refresh(lane),
            plan_submissions=lambda lane, _snapshot, _context: _plan(lane),
            reconcile_submission=lambda _lane, _unit, _snapshot: {},  # type: ignore[arg-type]
            submit_unit=submit,
        )
    }
    result = run_acquisition_cycle(
        registry=registry, snapshot=_snapshot(), handlers=handlers
    )
    assert result.status == "blocked"
    assert submitted["count"] == 0
    assert result.lane_results["paper"]["submissions"][0]["guard"] == (
        "submission_reconciliation_untyped"
    )


def test_submitted_action_cannot_be_repackaged_under_a_new_unit_id(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    lane = registry.lane("paper")
    _write_increment_receipt(lane)
    original = run_acquisition_cycle(
        registry=registry,
        snapshot=_snapshot(),
        handlers={
            "paper": LaneHandlers(
                refresh_status=lambda item, *_args: _refresh(item),
                plan_submissions=lambda item, _snapshot, _context: _plan(item),
                reconcile_submission=lambda *_args: (
                    ReconciliationResult.verified_absent()
                ),
                submit_unit=lambda _lane, unit: {
                    "cluster_id": 77,
                    "action_ids": list(unit.action_ids),
                },
            )
        },
    )
    assert original.status == "complete"
    submitted = {"count": 0}
    replacement = SubmissionUnit(
        unit_id="frontier-repacked",
        action_ids=_unit().action_ids,
        payload={"epoch": "e2"},
    )

    result = run_acquisition_cycle(
        registry=registry,
        snapshot=_snapshot(),
        handlers={
            "paper": LaneHandlers(
                refresh_status=lambda item, *_args: _refresh(item),
                plan_submissions=lambda item, _snapshot, _context: _plan(
                    item, units=(replacement,)
                ),
                reconcile_submission=lambda *_args: ReconciliationResult.indeterminate(),
                submit_unit=lambda *_args: submitted.__setitem__("count", 1),
            )
        },
    )
    assert result.status == "blocked"
    assert submitted["count"] == 0
    assert result.lane_results["paper"]["planning"]["guard"] == (
        "action_already_owned_by_other_unit"
    )


def test_terminal_failure_is_durable_and_refreshes_only_once(tmp_path: Path) -> None:
    registry = _registry(tmp_path, _lane(tmp_path, submit=False))
    calls = {"refresh": 0}

    def refresh(_lane, _snapshot, context):
        calls["refresh"] += 1
        assert len(context["failure_dispositions"]) == 1
        return _refresh(_lane)

    handlers = {"paper": LaneHandlers(refresh_status=refresh)}
    snapshot = _snapshot(state="failed", exit_code=3)
    first = run_acquisition_cycle(registry=registry, snapshot=snapshot, handlers=handlers)
    second = run_acquisition_cycle(registry=registry, snapshot=snapshot, handlers=handlers)
    assert first.status == "complete"
    assert second.status == "noop"
    assert calls["refresh"] == 1
    dispositions = tuple((tmp_path / "paper_state/scheduler_dispositions").rglob("*.json"))
    assert len(dispositions) == 1
    payload = json.loads(dispositions[0].read_text())
    assert payload["exit_code"] == 3
    assert payload["process_id"] == 0


def test_disabled_lane_with_scheduler_rows_fails_closed(tmp_path: Path) -> None:
    lane = _lane(tmp_path, enabled=False, submit=False)
    registry = _registry(tmp_path, lane)
    result = run_acquisition_cycle(
        registry=registry, snapshot=_snapshot(state="running", exit_code=None), handlers={}
    )
    assert result.status == "blocked"
    assert result.lane_results["paper"]["monitoring_guard"]["guard"] == (
        "unmonitored_lane_has_scheduler_rows"
    )


def test_enabled_lane_fails_closed_when_operational_handler_is_missing(
    tmp_path: Path,
) -> None:
    result = run_acquisition_cycle(
        registry=_registry(tmp_path), snapshot=_snapshot(), handlers={}
    )
    assert result.status == "blocked"
    ingestion = result.lane_results["paper"]["ingestion"][0]
    assert ingestion["guard"] == "missing_ingestion_handler"
    assert result.lane_results["paper"]["report_refresh"]["guard"] == (
        "missing_report_refresh_handler"
    )


def test_campaign_wide_lock_serializes_different_cluster_deltas(tmp_path: Path) -> None:
    path = tmp_path / "supervisor/acquisition.lock"
    with SupervisorLock(path):
        with pytest.raises(SupervisorLocked):
            with SupervisorLock(path):
                pass


def test_production_operations_capture_one_bound_snapshot_and_can_submit(
    tmp_path: Path,
) -> None:
    registry = _production_registry(tmp_path)
    lane = registry.lane("paper")
    calls = {"capture": 0, "handlers": 0, "submit": 0}
    scope = SchedulerScope(lane_id="paper", cluster_id=42, process_ids=(0,))

    class Operations:
        adapter_id = "test_operations_v1"

        def capture_snapshot(self, supplied_registry, *, expected_scopes):
            calls["capture"] += 1
            assert supplied_registry is registry
            assert expected_scopes == (scope,)
            return _production_snapshot(repository_root=tmp_path)

        def handlers_for(self, supplied_registry):
            calls["handlers"] += 1
            assert supplied_registry is registry

            def submit(_lane, unit):
                calls["submit"] += 1
                return {"cluster_id": 43, "action_ids": list(unit.action_ids)}

            return {
                "paper": LaneHandlers(
                    refresh_status=lambda item, *_args: _refresh(item),
                    plan_submissions=lambda item, _snapshot, _context: _plan(item),
                    reconcile_submission=lambda *_args: (
                        ReconciliationResult.verified_absent()
                    ),
                    submit_unit=submit,
                )
            }

    result = run_production_cycle(registry=registry, operations=Operations())
    assert result.status == "complete"
    assert calls == {"capture": 1, "handlers": 1, "submit": 1}
    assert result.lane_results["paper"]["submission_execution_enabled"] is True


def test_production_cycle_requires_registry_operations_binding(tmp_path: Path) -> None:
    registry = _registry(tmp_path)

    class Operations:
        adapter_id = "test_operations_v1"

        def capture_snapshot(self, *_args, **_kwargs):
            raise AssertionError("capture must not run before registry binding")

        def handlers_for(self, _registry):
            return {}

    with pytest.raises(CycleGuardFailed, match="production_operations_binding_missing"):
        run_production_cycle(registry=registry, operations=Operations())


def test_production_boundary_revalidates_canonical_registry_source(
    tmp_path: Path,
) -> None:
    from pipelines.time_dynamics.acquisition_registry_contract import (
        build_registry_payload,
    )

    repository_root = Path(__file__).resolve().parents[1]
    target = tmp_path / "canonical-registry.json"
    payload = build_registry_payload(repository_root)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    registry = CampaignRegistry.read(target, repository_root=repository_root)
    payload["scientific_contracts"]["matrix"]["raw_sha256"] = "0" * 64
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    class Operations:
        adapter_id = "must-not-run"

        def capture_snapshot(self, *_args, **_kwargs):
            raise AssertionError("capture must not run after registry drift")

        def handlers_for(self, _registry):
            raise AssertionError("handlers must not load after registry drift")

    with pytest.raises(CycleGuardFailed, match="acquisition_registry_contract_invalid"):
        run_production_cycle(registry=registry, operations=Operations())
    with pytest.raises(CycleGuardFailed, match="acquisition_registry_contract_invalid"):
        load_operations_factory("must.not.import:build", registry)


def test_operations_factory_must_match_registry_binding(tmp_path: Path) -> None:
    registry = _production_registry(tmp_path)
    with pytest.raises(CycleGuardFailed, match="operations_factory_binding_mismatch"):
        load_operations_factory("wrong.module:build", registry)


def test_production_scopes_come_from_registry_bound_durable_authority(
    tmp_path: Path,
) -> None:
    registry = _production_registry(tmp_path, lane=_lane(tmp_path, submit=False))
    expected = (SchedulerScope(lane_id="paper", cluster_id=42, process_ids=(0,)),)
    calls = {"capture": 0}

    class Operations:
        adapter_id = "test_operations_v1"

        def capture_snapshot(self, supplied_registry, *, expected_scopes):
            calls["capture"] += 1
            assert supplied_registry is registry
            assert expected_scopes == expected
            return _production_snapshot(repository_root=tmp_path)

        def handlers_for(self, _registry):
            return {
                "paper": LaneHandlers(
                    refresh_status=lambda item, *_args: _refresh(item)
                )
            }

    result = run_production_cycle(registry=registry, operations=Operations())
    assert result.status == "complete"
    assert calls == {"capture": 1}


def test_durable_submission_receipt_extends_next_production_scope(
    tmp_path: Path,
) -> None:
    registry = _production_registry(tmp_path)
    bootstrap = SchedulerScope(lane_id="paper", cluster_id=42, process_ids=(0,))
    submitted = SchedulerScope(lane_id="paper", cluster_id=43, process_ids=(0,))
    calls = {"capture": 0, "submit": 0}

    class Operations:
        adapter_id = "test_operations_v1"

        def capture_snapshot(self, _registry, *, expected_scopes):
            calls["capture"] += 1
            if calls["capture"] == 1:
                assert expected_scopes == (bootstrap,)
                return _production_snapshot(repository_root=tmp_path)
            assert expected_scopes == (bootstrap, submitted)
            return _production_snapshot(
                repository_root=tmp_path,
                captured_at="2026-08-28T12:01:00Z",
                extra_jobs=(
                    {
                        "lane_id": "paper",
                        "cluster_id": 43,
                        "process_id": 0,
                        "state": "running",
                        "exit_code": None,
                        "action_id": "next-action",
                    },
                ),
            )

        def handlers_for(self, _registry):
            def submit(_lane, unit):
                calls["submit"] += 1
                return {"cluster_id": 43, "action_ids": list(unit.action_ids)}

            return {
                "paper": LaneHandlers(
                    refresh_status=lambda item, *_args: _refresh(item),
                    plan_submissions=lambda item, _snapshot, _context: _plan(item),
                    reconcile_submission=lambda *_args: (
                        ReconciliationResult.verified_absent()
                    ),
                    submit_unit=submit,
                )
            }

    first = run_production_cycle(registry=registry, operations=Operations())
    second = run_production_cycle(registry=registry, operations=Operations())
    assert first.status == "complete"
    assert second.status == "complete"
    assert calls == {"capture": 2, "submit": 1}


def test_production_rejects_empty_durable_scope_for_monitored_lane(
    tmp_path: Path,
) -> None:
    registry = _production_registry(
        tmp_path,
        scopes=(),
        lane=_lane(tmp_path, submit=False),
    )

    class Operations:
        adapter_id = "test_operations_v1"

        def capture_snapshot(self, *_args, **_kwargs):
            raise AssertionError("capture must not run with empty durable scopes")

        def handlers_for(self, _registry):
            return {}

    with pytest.raises(CycleGuardFailed, match="expected_scheduler_scopes_empty"):
        run_production_cycle(registry=registry, operations=Operations())


def test_snapshot_replay_never_calls_submission_handlers(tmp_path: Path) -> None:
    registry = _registry(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("submission code must be inert during replay")

    result = run_replay_cycle(
        registry=registry,
        snapshot=_snapshot(state="running", exit_code=None),
        handlers={
            "paper": LaneHandlers(
                refresh_status=lambda item, *_args: _refresh(item),
                plan_submissions=forbidden,
                reconcile_submission=forbidden,
                submit_unit=forbidden,
            )
        },
    )
    assert result.status == "complete"
    lane = result.lane_results["paper"]
    assert lane["submission_enabled"] is True
    assert lane["submission_execution_enabled"] is False
    assert lane["submissions"] == []
    assert "planning" not in lane


def test_production_operations_require_raw_scheduler_evidence(
    tmp_path: Path,
) -> None:
    registry = _production_registry(tmp_path)
    scope = SchedulerScope(lane_id="paper", cluster_id=42, process_ids=(0,))
    payload = _production_snapshot().to_json_dict()
    payload["source"].pop("raw_scheduler_evidence")
    snapshot = SchedulerSnapshot.from_mapping(payload)
    handlers_called = {"value": False}

    class Operations:
        adapter_id = "test_operations_v1"

        def capture_snapshot(self, _registry, *, expected_scopes):
            assert expected_scopes == (scope,)
            return snapshot

        def handlers_for(self, _registry):
            handlers_called["value"] = True
            return {}

    with pytest.raises(CycleGuardFailed, match="raw_scheduler_evidence_missing"):
        run_production_cycle(registry=registry, operations=Operations())
    assert handlers_called["value"] is False


def test_production_operations_verify_raw_scheduler_evidence_bytes(
    tmp_path: Path,
) -> None:
    registry = _production_registry(
        tmp_path, lane=_lane(tmp_path, submit=False)
    )

    class Operations:
        adapter_id = "test_operations_v1"

        def capture_snapshot(self, _registry, *, expected_scopes):
            assert expected_scopes
            return _production_snapshot()

        def handlers_for(self, _registry):
            raise AssertionError("handlers must not load before evidence verification")

    with pytest.raises(CycleGuardFailed, match="raw_scheduler_evidence_unverifiable"):
        run_production_cycle(registry=registry, operations=Operations())


def test_production_operations_reject_changed_raw_scheduler_evidence(
    tmp_path: Path,
) -> None:
    registry = _production_registry(
        tmp_path, lane=_lane(tmp_path, submit=False)
    )
    snapshot = _production_snapshot(repository_root=tmp_path)
    (tmp_path / "scheduler/queue.raw").write_text("mutated after capture\n")

    class Operations:
        adapter_id = "test_operations_v1"

        def capture_snapshot(self, _registry, *, expected_scopes):
            assert expected_scopes
            return snapshot

        def handlers_for(self, _registry):
            raise AssertionError("handlers must not load after evidence changed")

    with pytest.raises(CycleGuardFailed, match="raw_scheduler_evidence_unverifiable"):
        run_production_cycle(registry=registry, operations=Operations())


def test_production_operations_reject_changed_registered_adapter_source(
    tmp_path: Path,
) -> None:
    registry = _production_registry(tmp_path)
    assert registry.production_operations is not None
    registry.production_operations.source_path.write_text("# changed adapter\n")

    class Operations:
        adapter_id = "test_operations_v1"

        def capture_snapshot(self, *_args, **_kwargs):
            raise AssertionError("capture must not run after adapter source changed")

        def handlers_for(self, _registry):
            return {}

    with pytest.raises(CycleGuardFailed, match="registry_artifact_digest_mismatch"):
        run_production_cycle(registry=registry, operations=Operations())


def test_production_operations_require_adapter_identity(tmp_path: Path) -> None:
    registry = _production_registry(tmp_path)
    scope = SchedulerScope(lane_id="paper", cluster_id=42, process_ids=(0,))

    class Operations:
        adapter_id = ""

        def capture_snapshot(self, _registry, *, expected_scopes):
            assert expected_scopes == (scope,)
            return _production_snapshot(adapter_id="")

        def handlers_for(self, _registry):
            return {}

    with pytest.raises(CycleGuardFailed, match="operations_adapter_identity_missing"):
        run_production_cycle(registry=registry, operations=Operations())


def test_production_operations_bind_snapshot_to_independent_expected_scopes(
    tmp_path: Path,
) -> None:
    registry = _production_registry(tmp_path)
    declared = SchedulerScope(lane_id="paper", cluster_id=42, process_ids=(0,))

    class Operations:
        adapter_id = "test_operations_v1"

        def capture_snapshot(self, _registry, *, expected_scopes):
            assert expected_scopes == (declared,)
            return _production_snapshot(repository_root=tmp_path, process_id=1)

        def handlers_for(self, _registry):
            return {}

    with pytest.raises(CycleGuardFailed, match="scheduler_scope_binding_mismatch"):
        run_production_cycle(registry=registry, operations=Operations())


def test_production_cli_loads_one_explicit_operations_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = tmp_path / "scheduler/queue.raw"
    history = tmp_path / "scheduler/history.raw"
    queue.parent.mkdir(parents=True)
    queue.write_text("queue\n")
    history.write_text("history\n")
    queue_sha256 = hashlib.sha256(queue.read_bytes()).hexdigest()
    history_sha256 = hashlib.sha256(history.read_bytes()).hexdigest()
    source = tmp_path / "test_acquisition_operations_factory.py"
    source.write_text(
        "from pipelines.time_dynamics.acquisition_supervisor import "
        "RAW_SCHEDULER_EVIDENCE_SCHEMA_V1, SNAPSHOT_SCHEMA_V1, SchedulerSnapshot\n"
        "class Operations:\n"
        "    adapter_id = 'cli_operations_v1'\n"
        "    def capture_snapshot(self, registry, *, expected_scopes):\n"
        "        assert expected_scopes == ()\n"
        "        return SchedulerSnapshot.from_mapping({\n"
        "            'schema': SNAPSHOT_SCHEMA_V1,\n"
        "            'captured_at': '2026-08-28T12:00:00Z',\n"
        "            'source': {\n"
        "                'queue_query_succeeded': True,\n"
        "                'history_query_succeeded': True,\n"
        "                'scope_complete': True,\n"
        "                'adapter_id': self.adapter_id,\n"
        "                'expected_processes': [],\n"
        "                'raw_scheduler_evidence': {\n"
        "                    'schema': RAW_SCHEDULER_EVIDENCE_SCHEMA_V1,\n"
        "                    'queue': {'reference': 'scheduler/queue.raw', "
        f"'sha256': '{queue_sha256}'}},\n"
        "                    'history': {'reference': 'scheduler/history.raw', "
        f"'sha256': '{history_sha256}'}},\n"
        "                },\n"
        "            },\n"
        "            'jobs': [],\n"
        "        })\n"
        "    def handlers_for(self, registry):\n"
        "        return {}\n"
        "def build_operations(registry):\n"
        "    return Operations()\n"
    )
    authority = tmp_path / "scheduler_scope_authority.json"
    authority.write_text(
        json.dumps(
            {
                "schema": SCOPE_AUTHORITY_SCHEMA_V1,
                "registry_id": "cli_registry",
                "scopes": [],
            },
            sort_keys=True,
        )
        + "\n"
    )
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema": REGISTRY_SCHEMA_V1,
                "registry_id": "cli_registry",
                "supervisor_state_root": "output/supervisor",
                "production_operations": {
                        "adapter_id": "cli_operations_v1",
                        "factory": (
                            "test_acquisition_operations_factory:build_operations"
                        ),
                        "source": {
                            "path": source.name,
                            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        },
                        "bootstrap_scopes": {
                            "path": authority.name,
                            "sha256": hashlib.sha256(
                                authority.read_bytes()
                            ).hexdigest(),
                        },
                },
                "lanes": [
                    {
                        "lane_id": "future",
                        "kind": "future",
                        "monitor_enabled": False,
                        "submission_enabled": False,
                        "attempts_root": "output/future/attempts",
                        "state_root": "output/future/state",
                        "blocked_by": ["not_activated"],
                    }
                ],
            }
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    assert (
        main(
            [
                "--registry",
                str(registry_path),
                "--repository-root",
                str(tmp_path),
                "--operations-factory",
                    "test_acquisition_operations_factory:build_operations",
            ]
        )
        == 0
    )
