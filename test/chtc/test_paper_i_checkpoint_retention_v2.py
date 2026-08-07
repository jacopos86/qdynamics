from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from types import ModuleType
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
BUILDER_PATH = BASE / "build_checkpoint_retention_v2.py"
CHECKPOINT_MEMBER = "pipelines/static_adapt/current_checkpoint.py"
OLD_CHECKPOINT_SHA256 = (
    "16ffddfdbf20674c50af7b797131efa40478c5281d16f4f034d7db49b8249cb8"
)
REPAIRED_CHECKPOINT_SHA256 = (
    "87e032010e009261de415101b717ff38fdb3d9b894b18d1939e6b219d94219f3"
)
SINGLETON_V2_ARCHIVE_SHA256 = (
    "4c79f8de78c1700120f2018b098d361c37c3b054e261be7214cb5fb74d862dd8"
)


def _load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_checkpoint_retention_v2", BUILDER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


@pytest.fixture(scope="module")
def builder() -> ModuleType:
    return _load_builder()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _tree_file_hashes(path: Path) -> dict[str, str]:
    return {
        member.relative_to(path).as_posix(): _sha256_file(member)
        for member in sorted(path.rglob("*"))
        if member.is_file()
    }


def _archive_members(
    path: Path,
) -> tuple[list[str], dict[str, tuple[bytes, int]]]:
    order: list[str] = []
    members: dict[str, tuple[bytes, int]] = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            assert member.isfile()
            assert not member.issym()
            assert not member.islnk()
            stream = archive.extractfile(member)
            assert stream is not None
            order.append(member.name)
            members[member.name] = (stream.read(), int(member.mode))
    assert len(order) == len(members)
    return order, members


def _campaign_paths(
    campaign: Any,
) -> tuple[Path, Path, Path, Path]:
    parent_package = BASE / campaign.parent_package_dirname
    package = BASE / campaign.package_dirname
    activation = BASE / campaign.activation_dirname
    return (
        parent_package,
        package,
        activation,
        BASE / campaign.parent_activation_dirname,
    )


def test_materialized_v2_campaigns_validate(builder: ModuleType) -> None:
    observed = [
        builder.validate_campaign(campaign, output_root=BASE)
        for campaign in builder.CAMPAIGNS
    ]

    assert [row["status"] for row in observed] == ["passed", "passed"]
    assert [row["parent_row_count"] for row in observed] == [48, 12]
    assert [row["completed_v1_excluded_count"] for row in observed] == [3, 1]
    assert [row["v2_queue_count"] for row in observed] == [45, 11]
    assert all(
        row["changed_source_members"] == [CHECKPOINT_MEMBER]
        for row in observed
    )
    assert all(row["scientific_settings_changed"] == [] for row in observed)


@pytest.mark.parametrize("campaign_index", (0, 1))
def test_v2_source_is_exactly_one_member_delta_with_modes_preserved(
    builder: ModuleType,
    campaign_index: int,
) -> None:
    campaign = builder.CAMPAIGNS[campaign_index]
    parent_package, package, _activation, _parent_activation = (
        _campaign_paths(campaign)
    )
    parent_order, parent_members = _archive_members(
        parent_package / "source_locked.tar.gz"
    )
    repaired_order, repaired_members = _archive_members(
        package / "source_locked.tar.gz"
    )

    changed = [
        name
        for name in parent_order
        if parent_members[name][0] != repaired_members[name][0]
    ]
    assert repaired_order == parent_order
    assert changed == [CHECKPOINT_MEMBER]
    assert {
        name: row[1] for name, row in repaired_members.items()
    } == {
        name: row[1] for name, row in parent_members.items()
    }
    assert (
        hashlib.sha256(parent_members[CHECKPOINT_MEMBER][0]).hexdigest()
        == OLD_CHECKPOINT_SHA256
    )
    assert (
        hashlib.sha256(repaired_members[CHECKPOINT_MEMBER][0]).hexdigest()
        == REPAIRED_CHECKPOINT_SHA256
    )
    protocol_members = [
        name
        for name in parent_order
        if "/protocols/" in name and name.endswith(".json")
    ]
    assert protocol_members
    assert all(
        repaired_members[name] == parent_members[name]
        for name in protocol_members
    )
    if campaign.key == "global_singleton":
        assert (
            _sha256_file(package / "source_locked.tar.gz")
            == SINGLETON_V2_ARCHIVE_SHA256
        )


@pytest.mark.parametrize("campaign_index", (0, 1))
def test_subset_is_exact_parent_order_minus_preserved_completions(
    builder: ModuleType,
    campaign_index: int,
) -> None:
    campaign = builder.CAMPAIGNS[campaign_index]
    parent_package, package, activation, parent_activation = _campaign_paths(
        campaign
    )
    parent_manifest = _load_json(
        parent_activation / "activation_manifest.json"
    )
    parent_ids = [
        row["execution_id"] for row in parent_manifest["executions"]
    ]
    expected_ids = [
        execution_id
        for proc_id, execution_id in enumerate(parent_ids)
        if proc_id not in set(campaign.completed_proc_ids)
    ]
    plan = _load_json(package / "execution_plan.json")
    supersession = _load_json(package / "supersession_map.json")
    activation_manifest = _load_json(
        activation / "activation_manifest.json"
    )

    assert plan["execution_ids"] == expected_ids
    assert [
        row["execution_id"] for row in activation_manifest["executions"]
    ] == expected_ids
    assert len(supersession["rows"]) == campaign.campaign_cell_count
    assert {
        row["parent_proc_id"] for row in supersession["rows"]
    } == set(range(campaign.campaign_cell_count))
    assert [
        row["v2_queue_index"]
        for row in supersession["rows"]
        if row["state"] == "uncompleted_v1_superseded_by_v2"
    ] == list(range(len(expected_ids)))
    completed_rows = [
        row
        for row in supersession["rows"]
        if row["state"] == "completed_v1_preserved_excluded_from_v2"
    ]
    assert [
        row["parent_proc_id"] for row in completed_rows
    ] == list(campaign.completed_proc_ids)
    assert all(
        row["completion_evidence"]["snapshot_row"][
            "local_verification"
        ]["status"]
        == "passed"
        and row["completion_evidence"]["snapshot_row"][
            "local_verification"
        ][
            "gzip_and_full_tar_scan_passed"
        ]
        is True
        and row["completion_evidence"]["snapshot_row"][
            "local_verification"
        ][
            "worker_exit_status"
        ]
        == 0
        for row in completed_rows
    )
    assert "rolling_receipt" not in json.dumps(supersession)
    queued_ids = {
        line.split("\t", 1)[0]
        for line in (activation / "queue.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    assert queued_ids == set(expected_ids)
    assert not queued_ids.intersection(
        row["execution_id"] for row in completed_rows
    )
    assert parent_package.is_dir()


@pytest.mark.parametrize("campaign_index", (0, 1))
def test_jobs_and_runtime_controls_are_byte_identical_to_v1(
    builder: ModuleType,
    campaign_index: int,
) -> None:
    campaign = builder.CAMPAIGNS[campaign_index]
    parent_package, package, activation, parent_activation = _campaign_paths(
        campaign
    )
    manifest = _load_json(package / "package_manifest.json")
    for row in manifest["jobs"]:
        copied = package / row["job"]["path"]
        parent = REPO_ROOT / row["parent_job"]["path"]
        assert copied.read_bytes() == parent.read_bytes()
    for name in ("run_cell.py", "package_contract.py"):
        assert (package / name).read_bytes() == (
            parent_package / name
        ).read_bytes()
    for name in ("execute_authorized_job.sh", "build_attempt_archive.py"):
        assert (activation / name).read_bytes() == (
            parent_activation / name
        ).read_bytes()


@pytest.mark.parametrize("campaign_index", (0, 1))
def test_v2_authorization_is_accepted_by_byte_identical_v1_runner(
    builder: ModuleType,
    campaign_index: int,
) -> None:
    campaign = builder.CAMPAIGNS[campaign_index]
    _parent_package, package, activation, _parent_activation = (
        _campaign_paths(campaign)
    )
    activation_manifest = _load_json(
        activation / "activation_manifest.json"
    )
    row = activation_manifest["executions"][0]
    job_path = REPO_ROOT / row["job"]["path"]
    authorization_path = REPO_ROOT / row["authorization"]["path"]
    job = _load_json(job_path)
    probe = """
import importlib.util
from pathlib import Path
import sys
sys.dont_write_bytecode = True
package = Path(sys.argv[1])
sys.path.insert(0, str(package))
spec = importlib.util.spec_from_file_location("v2_run_cell", package / "run_cell.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module._validate_authorization(
    Path(sys.argv[2]),
    execution_id=sys.argv[3],
    job_sha256=sys.argv[4],
)
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            probe,
            str(package),
            str(authorization_path),
            str(row["execution_id"]),
            str(job["sha256"]),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("campaign_index", (0, 1))
def test_submit_is_frozen_and_successes_leave_the_factory(
    builder: ModuleType,
    campaign_index: int,
) -> None:
    campaign = builder.CAMPAIGNS[campaign_index]
    _parent_package, _package, activation, _parent_activation = (
        _campaign_paths(campaign)
    )
    text = (activation / "submit.sub").read_text(encoding="utf-8")
    assignments = builder._submit_assignments(text)
    manifest = _load_json(activation / "activation_manifest.json")

    assert assignments["max_materialize"] == [
        str(campaign.max_materialize)
    ]
    assert assignments["max_idle"] == ["0"]
    assert assignments["leave_in_queue"] == [
        "(JobStatus == 4) && (ExitCode =!= 0)"
    ]
    assert manifest["pre_execution_gates"] == {
        "completed_predecessor_local_preservation_verified": True,
        "execution_thaw_authorized": False,
        "execution_thaw_blockers": [
            "parent_cluster_retirement_not_verified_by_local_builder"
        ],
        "parent_cluster_retirement_verified": False,
        "submission_must_remain_frozen_until_gates_pass": True,
    }


def _checkpoint_payload(depth: int) -> dict[str, Any]:
    return {
        "adapt_vqe": {
            "route_family": "singleton_response_snake",
            "history": [
                {
                    "depth": index,
                    "pool_index": index - 1,
                    "selected_batch_labels": [f"X{index}"],
                    "selected_pool_indices": [index - 1],
                    "selected_logical_size": 1,
                    "selected_feature_rows": [
                        {
                            "controller_snapshot": {
                                "controller_round": index
                            }
                        }
                    ],
                }
                for index in range(1, depth + 1)
            ],
            "history_count": depth,
            "pool_size": 8,
            "operators": [f"X{index}" for index in range(1, depth + 1)],
            "estimator_call_accounting": {
                "S_alg": depth,
                "S_unique": depth,
            },
            "terminal_active_prefix_checkpoint": {
                "schema": "test_active_prefix_checkpoint_v1",
                "depth": depth,
            },
        }
    }


def _ledger_payload(depth: int) -> dict[str, Any]:
    return {
        "schema": "estimator_call_ledger_v1",
        "ledger_fingerprint": f"ledger-at-depth-{depth}",
        "summary": {
            "unique_primitive_count": depth,
            "S_unique": depth,
        },
        "occurrence_summary": {"total_call_occurrences": depth},
    }


class _FixedDateTime(datetime):
    @classmethod
    def now(cls, tz: Any = None) -> "_FixedDateTime":
        return cls(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)


def _module_from_bytes(name: str, payload: bytes) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = f"<{name}>"
    exec(compile(payload, module.__file__, "exec"), module.__dict__)
    module.datetime = _FixedDateTime
    return module


def _published_projection(current_path: Path) -> dict[str, bytes]:
    current = json.loads(current_path.read_text(encoding="utf-8"))
    adapt = current["adapt_vqe"]
    paths = [
        current_path,
        current_path.with_name(
            adapt["estimator_call_ledger_checkpoint"]["path"]
        ),
        current_path.with_name(
            adapt["verified_singleton_resume_sidecar"]["path"]
        ),
    ]
    return {path.name: path.read_bytes() for path in paths}


def _publish_two_rounds(module: ModuleType, current_path: Path) -> None:
    for depth in (1, 2):
        module._publish_active_cli_current_checkpoint(
            _checkpoint_payload(depth),
            ledger_payload=_ledger_payload(depth),
            path=current_path,
            keep_history_tail=1,
        )


def test_archived_repair_preserves_payload_and_retires_only_predecessors(
    builder: ModuleType,
    tmp_path: Path,
) -> None:
    campaign = builder.CAMPAIGNS[0]
    parent_package, package, _activation, _parent_activation = (
        _campaign_paths(campaign)
    )
    _old_order, old_members = _archive_members(
        parent_package / "source_locked.tar.gz"
    )
    _new_order, new_members = _archive_members(
        package / "source_locked.tar.gz"
    )
    old_module = _module_from_bytes(
        "sealed_v1_current_checkpoint",
        old_members[CHECKPOINT_MEMBER][0],
    )
    repaired_module = _module_from_bytes(
        "sealed_v2_current_checkpoint",
        new_members[CHECKPOINT_MEMBER][0],
    )
    current_path = tmp_path / "current.json"

    _publish_two_rounds(old_module, current_path)
    old_projection = _published_projection(current_path)
    old_sidecars = {
        path
        for path in tmp_path.iterdir()
        if path.name != current_path.name
    }
    assert len(old_sidecars) == 4

    for path in list(tmp_path.iterdir()):
        path.unlink()
    _publish_two_rounds(repaired_module, current_path)
    repaired_projection = _published_projection(current_path)
    repaired_sidecars = {
        path
        for path in tmp_path.iterdir()
        if path.name != current_path.name
    }

    assert repaired_projection == old_projection
    assert len(repaired_sidecars) == 2


def test_retirement_receipt_requires_exact_empty_live_and_factory_queries(
    builder: ModuleType,
) -> None:
    with tempfile.TemporaryDirectory(
        prefix=".checkpoint-retirement-test.",
        dir=BASE,
    ) as temporary_name:
        temporary = Path(temporary_name)
        removal = temporary / "condor_rm.stdout"
        live = temporary / "condor_q.json"
        factory = temporary / "condor_q_factory.json"
        receipt_path = temporary / "retirement_receipt.json"
        removal.write_text(
            "All jobs in cluster 9395481 have been marked for removal\\n"
            "All jobs in cluster 9395482 have been marked for removal\\n",
            encoding="utf-8",
        )
        live.write_text("[]\n", encoding="utf-8")
        factory.write_text("[]\n", encoding="utf-8")

        receipt = builder.seal_retirement_receipt(
            output_path=receipt_path,
            retired_utc="2026-07-30T18:00:00Z",
            verified_utc="2026-07-30T18:01:00Z",
            schedd="submit.example.invalid",
            owner="jsstrobel",
            condor_rm_stdout=removal,
            post_condor_q_json=live,
            post_condor_q_factory_json=factory,
            condor_rm_exit_status=0,
            post_condor_q_exit_status=0,
            post_condor_q_factory_exit_status=0,
        )

        assert builder.validate_retirement_receipt(receipt_path) == receipt
        assert receipt["target_cluster_ids"] == [9395481, 9395482]
        assert receipt["all_target_clusters_absent_from_queue"] is True
        assert receipt["unrelated_clusters_touched"] is False

        bad_live = temporary / "bad_condor_q.json"
        bad_live.write_text(
            '[{"ClusterId":9395481,"ProcId":2}]\n',
            encoding="utf-8",
        )
        with pytest.raises(
            builder.RepairPackageError,
            match="still have live or factory ads",
        ):
            builder.seal_retirement_receipt(
                output_path=temporary / "bad_receipt.json",
                retired_utc="2026-07-30T18:00:00Z",
                verified_utc="2026-07-30T18:01:00Z",
                schedd="submit.example.invalid",
                owner="jsstrobel",
                condor_rm_stdout=removal,
                post_condor_q_json=bad_live,
                post_condor_q_factory_json=factory,
                condor_rm_exit_status=0,
                post_condor_q_exit_status=0,
                post_condor_q_factory_exit_status=0,
            )


def test_immutable_preservation_snapshot_is_bound_not_the_rolling_guard(
    builder: ModuleType,
) -> None:
    snapshot_path = REPO_ROOT / builder.PRESERVATION_SNAPSHOT_RELATIVE
    snapshot = builder.validate_preservation_snapshot(snapshot_path)
    expected_binding = builder._json_binding(
        snapshot_path, relative_to=REPO_ROOT
    )

    assert snapshot["rolling_guard_receipt_bound"] is False
    assert "rolling_receipt" not in json.dumps(snapshot)
    assert snapshot["completed_archive_count"] == 4
    for campaign in builder.CAMPAIGNS:
        package = BASE / campaign.package_dirname
        activation = BASE / campaign.activation_dirname
        for path, field in (
            (
                package / "package_manifest.json",
                "completed_v1_preservation_snapshot",
            ),
            (
                package / "execution_plan.json",
                "completed_v1_preservation_snapshot",
            ),
            (
                package / "supersession_map.json",
                "completed_v1_preservation_snapshot",
            ),
            (
                activation / "activation_manifest.json",
                "completed_v1_preservation_snapshot",
            ),
        ):
            assert _load_json(path)[field] == expected_binding

    with tempfile.TemporaryDirectory(
        prefix=".checkpoint-preservation-tamper.",
        dir=BASE,
    ) as temporary_name:
        tampered_path = Path(temporary_name) / snapshot_path.name
        tampered = dict(snapshot)
        tampered["status"] = "tampered"
        tampered_path.write_text(
            json.dumps(tampered, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(
            builder.RepairPackageError,
            match="self digest drifted",
        ):
            builder.validate_preservation_snapshot(tampered_path)


def test_post_retirement_release_activation_reseals_only_authority_layer(
    builder: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(
        prefix=".checkpoint-release-test.",
        dir=REPO_ROOT,
    ) as temporary_name:
        mini_root = Path(temporary_name)
        mini_base = (
            mini_root / "chtc/paper_i_ra_adapt_repair_20260727"
        )
        mini_base.mkdir(parents=True)
        for campaign in builder.CAMPAIGNS:
            for dirname in (
                campaign.parent_package_dirname,
                campaign.parent_activation_dirname,
                campaign.package_dirname,
                campaign.activation_dirname,
            ):
                shutil.copytree(BASE / dirname, mini_base / dirname)
            shutil.copy2(
                BASE / campaign.parent_submission_receipt_name,
                mini_base / campaign.parent_submission_receipt_name,
            )
        snapshot_relative = builder.PRESERVATION_SNAPSHOT_RELATIVE
        snapshot_source = REPO_ROOT / snapshot_relative
        snapshot_destination = mini_root / snapshot_relative
        snapshot_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot_source, snapshot_destination)

        monkeypatch.setattr(builder, "REPO_ROOT", mini_root)
        monkeypatch.setattr(builder, "BASE", mini_base)

        evidence_dir = mini_base / "retirement_evidence"
        evidence_dir.mkdir()
        removal = evidence_dir / "condor_rm.stdout"
        live = evidence_dir / "condor_q.json"
        factory = evidence_dir / "condor_q_factory.json"
        removal.write_text(
            "exact clusters marked for removal\n", encoding="utf-8"
        )
        live.write_text("[]\n", encoding="utf-8")
        factory.write_text("[]\n", encoding="utf-8")
        retirement_path = (
            mini_base / "parent_cluster_retirement_receipt.json"
        )
        builder.seal_retirement_receipt(
            output_path=retirement_path,
            retired_utc="2026-07-30T18:00:00Z",
            verified_utc="2026-07-30T18:01:00Z",
            schedd="submit.example.invalid",
            owner="jsstrobel",
            condor_rm_stdout=removal,
            post_condor_q_json=live,
            post_condor_q_factory_json=factory,
            condor_rm_exit_status=0,
            post_condor_q_exit_status=0,
            post_condor_q_factory_exit_status=0,
        )

        for campaign in builder.CAMPAIGNS:
            builder.materialize_release_activation(
                campaign,
                output_root=mini_base,
                retirement_receipt_path=retirement_path,
                released_utc="2026-07-30T18:02:00Z",
            )
            result = builder.validate_release_activation(
                campaign,
                output_root=mini_base,
                retirement_receipt_path=retirement_path,
            )
            assert result["status"] == "passed"
            assert result["parent_cluster_retirement_verified"] is True
            assert result["execution_thaw_authorized"] is True
            assert result["submission_initial_state"] == (
                "frozen_max_idle_zero"
            )
            frozen = mini_base / campaign.activation_dirname
            released = mini_base / campaign.release_activation_dirname
            for name in (
                "execute_authorized_job.sh",
                "build_attempt_archive.py",
            ):
                assert (released / name).read_bytes() == (
                    frozen / name
                ).read_bytes()

            package = mini_base / campaign.package_dirname
            frozen_hashes = _tree_file_hashes(frozen)
            package_hashes = _tree_file_hashes(package)
            release_v1_hashes = _tree_file_hashes(released)
            builder.materialize_ordinary_held_release_activation(
                campaign,
                output_root=mini_base,
                retirement_receipt_path=retirement_path,
                released_utc="2026-07-30T18:30:00Z",
            )
            held_result = (
                builder.validate_ordinary_held_release_activation(
                    campaign,
                    output_root=mini_base,
                    retirement_receipt_path=retirement_path,
                )
            )
            assert held_result == {
                "campaign": campaign.key,
                "status": "passed",
                "release_activation_id": (
                    campaign.ordinary_held_release_activation_id
                ),
                "direct_execution_count": (
                    campaign.campaign_cell_count
                    - len(campaign.completed_proc_ids)
                ),
                "parent_cluster_ids": [9395481, 9395482],
                "parent_cluster_retirement_verified": True,
                "execution_thaw_authorized": True,
                "submission_initial_state": (
                    "ordinary_all_procs_held_before_start"
                ),
            }
            held = (
                mini_base
                / campaign.ordinary_held_release_activation_dirname
            )
            held_manifest = _load_json(
                held / "activation_manifest.json"
            )
            held_submit = (held / "submit.sub").read_text(
                encoding="utf-8"
            )
            held_assignments = builder._submit_assignments(held_submit)
            assert "max_materialize" not in held_assignments
            assert "max_idle" not in held_assignments
            assert held_assignments["hold"] == ["True"]
            assert held_assignments["periodic_release"] == ["False"]
            assert held_assignments["+holsteinlifecyclemode"] == [
                '"ordinary_held_exact_proc_release_v1"'
            ]
            assert held_assignments["+jobbatchname"] == [
                f'"{campaign.ordinary_held_release_batch_name}"'
            ]
            assert held_assignments["leave_in_queue"] == [
                "(JobStatus == 4) && (ExitCode =!= 0)"
            ]
            assert held_manifest["scheduler_contract"] == {
                "mode": "ordinary_held_exact_proc_release_v1",
                "late_materialization": False,
                "expected_proc_count": len(
                    held_manifest["executions"]
                ),
                "submit_hold": True,
                "automatic_release": False,
                "post_submit_verification_required_before_release": {
                    "exact_cluster_proc_count": len(
                        held_manifest["executions"]
                    ),
                    "all_job_status": 5,
                    "all_num_job_starts": 0,
                },
                "release_scope": "exact_cluster_proc_only",
                "cluster_wide_release_forbidden": True,
                "owner_wide_release_forbidden": True,
                "constraint_wide_release_forbidden": True,
                "one_proc_per_quota_cycle": True,
            }
            frozen_ids = [
                row["execution_id"]
                for row in _load_json(
                    frozen / "activation_manifest.json"
                )["executions"]
            ]
            held_ids = [
                row["execution_id"]
                for row in held_manifest["executions"]
            ]
            assert held_ids == frozen_ids
            assert [
                line.split("\t", 1)[0]
                for line in (held / "queue.tsv")
                .read_text(encoding="utf-8")
                .splitlines()
            ] == frozen_ids
            assert _tree_file_hashes(frozen) == frozen_hashes
            assert _tree_file_hashes(package) == package_hashes
            assert _tree_file_hashes(released) == release_v1_hashes
            for name in (
                "execute_authorized_job.sh",
                "build_attempt_archive.py",
            ):
                assert (held / name).read_bytes() == (
                    frozen / name
                ).read_bytes()
            release_v1_manifest = _load_json(
                released / "activation_manifest.json"
            )
            for held_row, release_v1_row in zip(
                held_manifest["executions"],
                release_v1_manifest["executions"],
                strict=True,
            ):
                assert held_row["execution_id"] == (
                    release_v1_row["execution_id"]
                )
                assert held_row["job"] == release_v1_row["job"]
                assert held_row["resources"] == (
                    release_v1_row["resources"]
                )
                job_path = mini_root / held_row["job"]["path"]
                predecessor_job_path = (
                    mini_root / release_v1_row["job"]["path"]
                )
                assert job_path.read_bytes() == (
                    predecessor_job_path.read_bytes()
                )
