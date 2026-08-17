from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "run_local_page12_strong_holstein_sector5_20260814.py"
)


def _load_runner():
    name = "paper_i_page12_strong_sector5_local_runner_test"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_scope_is_exactly_the_five_held_strong_holstein_cells() -> None:
    runner = _load_runner()

    assert runner.TARGET_PROCS == (5, 11, 4, 10, 3)
    assert len(runner.TARGET_EXECUTION_IDS) == 5
    assert len(set(runner.TARGET_EXECUTION_IDS)) == 5
    assert all("__nph7__" in row for row in runner.TARGET_EXECUTION_IDS)
    assert all("strong" in row for row in runner.TARGET_EXECUTION_IDS)
    assert not any(
        "weak_strong" in row and row.endswith("append_only")
        for row in runner.TARGET_EXECUTION_IDS
    )
    assert sorted(runner.TARGET_PROCS) == [3, 4, 5, 10, 11]
    assert runner.MAXIMUM_CONCURRENCY == 1
    assert runner.TARGET_HORIZON == 50


def test_streaming_json_matches_the_canonical_byte_contract(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    payload = {
        "z": [3, {"unicode": "Holstein–Hubbard"}],
        "a": {"float": 0.790569415042, "bool": True},
    }
    path = tmp_path / "payload.json"

    runner._write_json_streaming(path, payload)

    assert path.read_bytes() == runner._canonical_json_bytes(payload) + b"\n"


def test_checkpoint_wrapper_forces_only_tail_retention_to_one() -> None:
    runner = _load_runner()
    observed: dict = {}

    def original(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return "constructed"

    wrapped = runner._compact_checkpoint_observation_factory(original)
    result = wrapped(
        "sentinel",
        path=Path("current.json"),
        every_controller_rounds=1,
        keep_history_tail=100,
    )

    assert result == "constructed"
    assert observed["args"] == ("sentinel",)
    assert observed["kwargs"] == {
        "path": Path("current.json"),
        "every_controller_rounds": 1,
        "keep_history_tail": 1,
    }


def _prepare_test_planning(
    runner, tmp_path: Path, monkeypatch
) -> tuple[Path, Path, dict]:
    planning_dir = tmp_path / "planning"
    runtime_dir = tmp_path / "runtime"
    passed_capacity = runner._digested(
        {
            "schema": "test_capacity_v1",
            "status": "passed",
            "blockers": [],
            "scientific_execution_performed": False,
        }
    )
    monkeypatch.setattr(runner, "_capacity", lambda _path: passed_capacity)
    hold_rows = [
        {
            "proc": cell.proc,
            "execution_id": cell.execution_id,
            "job_status": 5,
            "hold_reason": runner.HOLD_REASON,
        }
        for cell in sorted(runner.TARGET_CELLS, key=lambda item: item.proc)
    ]
    hold_receipt = runner._digested(
        {
            "schema": runner.REMOTE_HOLD_SCHEMA,
            "status": "passed_authenticated_exact_remote_holds",
            "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "scheduler": "chtc_condor",
            "cluster_id": 9647385,
            "authentication_kind": "interactive_ssh_duo_condor_hold_query_v1",
            "authenticated_remote_query": True,
            "held_procs": sorted(runner.TARGET_PROCS),
            "held_execution_ids": [row["execution_id"] for row in hold_rows],
            "remote_active_execution_ids": [],
            "late_materialization_factory_active": False,
            "rows": hold_rows,
            "remote_rows_sha256": runner._canonical_sha256({"rows": hold_rows}),
            "scientific_execution_performed": False,
        }
    )
    hold_path = tmp_path / "remote_hold.json"
    runner._write_json_exclusive(hold_path, hold_receipt)
    planning = runner.prepare_planning(
        planning_dir=planning_dir,
        runtime_dir=runtime_dir,
        remote_hold_receipt=hold_path,
    )
    return planning_dir, runtime_dir, planning


def _write_fake_parity(
    runner,
    *,
    activation_dir: Path,
    plan: dict,
    authorization: dict,
    **_kwargs,
) -> dict:
    witness = {
        "final_state": {"energy": runner.PARITY_BASELINE_ENERGY},
        "accepted_transition": {"controller_round": 1},
    }
    branch_paths = []
    for variant, source_sha, name in (
        (
            "sealed_baseline",
            runner.SEALED_CURRENT_CHECKPOINT_SHA256,
            "parity_sealed_baseline_branch.json",
        ),
        (
            "operational_candidate",
            runner.CANDIDATE_CURRENT_CHECKPOINT_SHA256,
            "parity_operational_candidate_branch.json",
        ),
    ):
        branch = runner._digested(
            {
                "schema": runner.PARITY_BRANCH_SCHEMA,
                "status": "passed_guarded_authorized_diagnostic_branch",
                "variant": variant,
                "execution_id": runner.TARGET_EXECUTION_IDS[0],
                "controller_rounds_completed": runner.PARITY_MAXIMUM_ROUNDS,
                "job_spec_sha256": runner.PARITY_JOB_SPEC_SHA256,
                "protocol_sha256": runner.PARITY_PROTOCOL_SHA256,
                "route_contract_sha256": runner.PARITY_ROUTE_SHA256,
                "base_source_archive_sha256": runner.SOURCE_ARCHIVE_SHA256,
                "current_checkpoint_source_sha256": source_sha,
                "execution_plan_sha256": plan["sha256"],
                "execution_authorization_sha256": authorization["sha256"],
                "child_payload_sha256": "0" * 64,
                "scientific_projection_sha256": "1" * 64,
                "checkpoint_observation_sha256": "2" * 64,
                "estimator_ledger_sha256": "3" * 64,
                "checkpoint_file_sha256": "4" * 64,
                "checkpoint_size_bytes": 1,
                "witness": witness,
                "resource_guard": {
                    "guard_stop_reason": None,
                    "child_returncode": 0,
                },
                "scientific_execution_performed": True,
                "diagnostic_only": True,
                "campaign_cell_progress_credited": False,
                "scientific_artifacts_retained": False,
                "paper_evidence_adoption_authorized": False,
            }
        )
        path = activation_dir / name
        runner._write_json_exclusive(path, branch)
        branch_paths.append(path)
    parity = runner._digested(
        {
            "schema": runner.PARITY_SCHEMA,
            "status": "passed_exact_scientific_parity",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "execution_id": runner.TARGET_EXECUTION_IDS[0],
            "maximum_controller_rounds": runner.PARITY_MAXIMUM_ROUNDS,
            "parity_canary_spec_sha256": plan[
                "scientific_parity_canary_spec"
            ]["sha256"],
            "execution_plan_sha256": plan["sha256"],
            "execution_authorization_sha256": authorization["sha256"],
            "sealed_baseline_branch": runner._binding(
                branch_paths[0], root=activation_dir, canonical=True
            ),
            "operational_candidate_branch": runner._binding(
                branch_paths[1], root=activation_dir, canonical=True
            ),
            "scientific_projection_sha256": "1" * 64,
            "estimator_ledger_sha256": "3" * 64,
            "one_round_energy": runner.PARITY_BASELINE_ENERGY,
            "exact_canonical_projection_equal": True,
            "estimator_ledger_equal": True,
            "scientific_execution_performed": True,
            "diagnostic_only": True,
            "campaign_cell_progress_credited": False,
            "scientific_artifacts_retained": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    runner._write_json_exclusive(
        activation_dir / "scientific_parity_canary.json", parity
    )
    return parity


def test_prepare_materializes_planning_without_authority(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    planning_dir, _runtime_dir, planning = _prepare_test_planning(
        runner, tmp_path, monkeypatch
    )
    plan = runner._load_digested(
        planning_dir / "execution_plan.json", label="test execution plan"
    )
    host = runner._load_digested(
        planning_dir / "host_preflight.json", label="test host preflight"
    )

    assert planning["execution_authorized"] is False
    assert sorted(path.name for path in planning_dir.iterdir()) == [
        "execution_plan.json",
        "host_preflight.json",
        "planning_manifest.json",
        "remote_hold_receipt.json",
    ]
    assert plan["execution_authorized"] is False
    assert plan["run_order"] == list(runner.TARGET_EXECUTION_IDS)
    assert [row["proc"] for row in plan["cells"]] == [5, 11, 4, 10, 3]
    assert all(row["nph"] == 7 for row in plan["cells"])
    assert all(row["target_horizon"] == 50 for row in plan["cells"])
    assert plan["source_authorizations_reused"] is False
    assert plan["remote_overlap_control"]["held_procs"] == [3, 4, 5, 10, 11]
    assert plan["remote_overlap_control"]["authenticated_remote_query"] is True
    assert plan["scientific_parity_canary_spec"]["proc"] == 5
    assert plan["scientific_parity_canary_spec"][
        "maximum_controller_rounds"
    ] == 1
    repairs = plan["operational_repairs"]
    assert repairs["checkpoint_keep_history_tail"] == 1
    assert repairs["scientific_protocol_settings_changed"] is False
    assert repairs["route_contracts_changed"] is False
    assert [row["path"] for row in repairs["source_overlay_files"]] == [
        path.as_posix() for path in runner.OVERLAY_RELATIVE_PATHS
    ]
    assert host["scientific_execution_performed"] is False
    assert len(host["sealed_worker_preflights"]) == 5


def test_authorize_binds_unchanged_planning_and_parity(
    tmp_path: Path, monkeypatch
) -> None:
    runner = _load_runner()
    planning_dir, runtime_dir, planning = _prepare_test_planning(
        runner, tmp_path, monkeypatch
    )
    activation_dir = tmp_path / "activation"
    planning_bytes = {
        path.name: path.read_bytes() for path in planning_dir.iterdir()
    }
    monkeypatch.setattr(
        runner,
        "_materialize_scientific_parity_canary",
        lambda **kwargs: _write_fake_parity(runner, **kwargs),
    )

    activation = runner.authorize_activation(
        planning_dir=planning_dir,
        activation_dir=activation_dir,
        runtime_dir=runtime_dir,
        authorization_basis="test explicit user authority",
    )
    plan = runner._load_digested(
        activation_dir / "execution_plan.json", label="activated plan"
    )
    authority = runner._load_digested(
        activation_dir / "execution_authorization.json",
        label="test authorization",
    )

    assert activation["execution_authorized"] is True
    assert planning["execution_authorized"] is False
    assert authority["execution_plan_sha256"] == plan["sha256"]
    assert authority["scientific_parity_canary_authorized"] is True
    assert activation["scientific_parity_canary"]["canonical_sha256"]
    assert {
        path.name: path.read_bytes() for path in planning_dir.iterdir()
    } == planning_bytes
    assert not (planning_dir / "execution_authorization.json").exists()
    manifest, _rows = runner._closed_inputs(runner._load_worker())
    validated = runner._validate_activation(
        activation_dir, manifest=manifest, require_fresh_hold=True
    )
    assert validated[0] == activation


def test_scientific_projection_ignores_only_checkpoint_transport() -> None:
    runner = _load_runner()
    payload = {
        "schema": "result",
        "protocol": {"sha256": "p"},
        "selector_identity": "selector",
        "parent_inventory": {"count": 1},
        "executable_pool": {"count": 1},
        "policy": {"kind": "always"},
        "run": {
            "final_state": {"energy": 1.0},
            "accepted_trajectory": [{"energy": 1.0}],
            "accepted_transitions": [{"operator": "X"}],
            "problem": {"id": "hh"},
            "route": {"id": "ra"},
            "stop": {"reason": "max"},
            "scientific_replay": [{"operator": "X"}],
            "estimator_accounting": {"s_alg": 1},
            "canonical_reporting": {"work": 1},
            "paper_i_summary": {
                "energy": 1.0,
                "append_matched": {
                    "failure": {
                            "message": (
                                "[Errno 2] No such file or directory: "
                                "'/private/var/tmp/"
                                "paper-i-strong5-parity-sealed_baseline.aaa/"
                                "paper-i-strong5-sealed_baseline.bbb/"
                                "append.json'"
                        )
                    }
                },
            },
            "observation": {"checkpoint_sha256": "old"},
        },
        "numerical_physical_integrity": {"status": "passed"},
        "scientific_receipts": {
            "controller_replay_evidence": {"checkpoint": "old"},
            "controller_replay_evidence_sha256": "old",
            "route_contract": {"id": "ra"},
        },
    }
    changed_transport = json.loads(json.dumps(payload))
    changed_transport["run"]["observation"] = {"checkpoint_sha256": "new"}
    changed_transport["scientific_receipts"][
        "controller_replay_evidence"
    ] = {"checkpoint": "new"}
    changed_transport["scientific_receipts"][
        "controller_replay_evidence_sha256"
    ] = "new"
    changed_transport["run"]["paper_i_summary"]["append_matched"][
        "failure"
    ]["message"] = (
        "[Errno 2] No such file or directory: '/private/var/tmp/"
        "paper-i-strong5-parity-operational_candidate.ccc/"
        "paper-i-strong5-operational_candidate.ddd/append.json'"
    )

    assert runner._scientific_parity_projection(
        payload
    ) == runner._scientific_parity_projection(changed_transport)
    changed_science = json.loads(json.dumps(payload))
    changed_science["run"]["final_state"]["energy"] = 2.0
    assert runner._scientific_parity_projection(
        payload
    ) != runner._scientific_parity_projection(changed_science)


def test_atomic_noreplace_never_overwrites_terminal(tmp_path: Path) -> None:
    runner = _load_runner()
    path = tmp_path / "terminal_receipt.json"
    first = runner._digested({"schema": "terminal", "status": "passed"})
    second = runner._digested({"schema": "terminal", "status": "changed"})

    runner._write_json_atomic_noreplace(path, first)
    with pytest.raises(FileExistsError):
        runner._write_json_atomic_noreplace(path, second)

    assert json.loads(path.read_text()) == first
