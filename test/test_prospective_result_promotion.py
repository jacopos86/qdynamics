"""Focused contracts for offline prospective-result promotion."""

from __future__ import annotations

import hashlib
import gzip
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import tarfile

import pytest

from pipelines.time_dynamics.guarded_archive_transfer import (
    DEFAULT_FREE_FLOOR_BYTES,
    finalize_archive_transfer,
)
from pipelines.time_dynamics.combined_scheduler_snapshot import (
    CAPTURE_SPEC_V1,
    HH_ACQUISITION_LANE_ID,
    HUBBARD_ACQUISITION_LANE_ID,
    REQUIRED_PROJECTED_ATTRIBUTES,
    build_combined_snapshot,
    write_immutable_snapshot,
)
from pipelines.time_dynamics.prospective_bootstrap_activation import (
    authorize_from_snapshot,
    load_activation_manifest,
    load_submission_intent,
    prepare_bootstrap_activation,
    snapshot_intent_binding,
)
from pipelines.time_dynamics.prospective_campaign_bootstrap import LANE_HUBBARD
from pipelines.time_dynamics.prospective_execution_bindings import (
    ProspectiveExecutionError,
)
from pipelines.time_dynamics.prospective_seed_chtc_package import (
    write_runtime_image_identity,
)
import pipelines.time_dynamics.prospective_exact_reference_worker as exact_worker
import pipelines.time_dynamics.runners.build_prospective_exact_reference as exact_runner
from pipelines.time_dynamics import prospective_result_promotion as promotion
from pipelines.time_dynamics.prospective_result_promotion import (
    MODE_AVQDS_PREFIX,
    MODE_AVQDS_TRAJECTORY,
    MODE_EXACT_REFERENCE,
    PROMOTION_PLAN_V1,
    PROMOTION_RECEIPT_V1,
    REMOTE_ARCHIVE_IDENTITY_V1,
    ResultPromotionError,
    build_archive_transfer_spec,
    build_promotion_plan,
    load_promotion_plan,
    promote_result,
    validate_finalized_archive,
    validate_remote_archive_identity,
)


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _load_test_module(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load test helper {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def prepared_bootstrap_repository(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, object]:
    root = tmp_path_factory.mktemp("promotion-bootstrap") / "repository"
    root.mkdir()
    scout_tests = _load_test_module(
        "_promotion_scout_fixture", "test_prospective_avqds_scout_worker.py"
    )
    exact_tests = _load_test_module(
        "_promotion_exact_fixture", "test_prospective_exact_reference_worker.py"
    )
    source, seed, scout_request = scout_tests._receipts_and_request(root)
    overlays = {
        "pipelines/time_dynamics/prospective_exact_reference_worker.py": Path(
            exact_worker.__file__
        ),
        "pipelines/time_dynamics/runners/build_prospective_exact_reference.py": Path(
            exact_runner.__file__
        ),
    }
    for relative, source_file in overlays.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target)
    subprocess.run(
        ("git", "add", *overlays), cwd=root, check=True, capture_output=True
    )
    subprocess.run(
        ("git", "commit", "-qm", "exact-reference executor overlay"),
        cwd=root,
        check=True,
        capture_output=True,
    )
    source_tar = subprocess.run(
        (
            "git",
            "archive",
            "--format=tar",
            "--prefix=source/",
            str(source.source_epoch.source_revision),
            "--",
            "pipelines",
            "src",
            "requirements.txt",
        ),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    compressed = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=compressed, compresslevel=9, mtime=0
    ) as handle:
        handle.write(source_tar)
    exact_archive = root / "runtime/exact_source.tar.gz"
    exact_archive.parent.mkdir(parents=True, exist_ok=True)
    exact_archive.write_bytes(compressed.getvalue())
    scout_order = scout_tests.build_work_order(
        scout_request,
        source_epoch_receipt=source,
        seed_receipt=seed,
        repository_root=root,
    )
    avqds_archive = root / "runtime/avqds_exact_blind_source.tar.gz"
    scout_tests.materialize_avqds_runtime_source_archive(
        avqds_archive,
        repository_root=root,
        profile=scout_tests.SCOUT_PROFILE,
        base_revision=str(source.source_epoch.source_revision),
        base_full_archive_sha256=str(source.source_epoch.source_archive_sha256),
        overlay_revision=scout_order.executor_overlay.overlay_revision,
        overlay_files=scout_order.executor_overlay.files,
    )
    image = root / "runtime/paper_ii_runtime.sif"
    image.write_bytes(b"promotion integration runtime image\n")
    image_identity = root / "runtime/paper_ii_runtime.identity.json"
    write_runtime_image_identity(
        image_identity, image_path=image, repository_root=root
    )
    manifest = prepare_bootstrap_activation(
        repository_root=root,
        output_dir="prepared/bootstrap_activation",
        seed_receipt_paths=[seed.receipt_path],
        source_epoch_receipt_paths=[source.receipt_path],
        exact_source_archive_paths={
            LANE_HUBBARD: exact_archive.relative_to(root).as_posix()
        },
        avqds_source_archive_paths={
            LANE_HUBBARD: avqds_archive.relative_to(root).as_posix()
        },
        runtime_image_identity_path=image_identity.relative_to(root).as_posix(),
        free_bytes=30 * 1024**3,
        recorded_at="2026-08-29T02:00:00-05:00",
    )
    return {
        "root": root,
        "manifest": manifest,
        "scout_tests": scout_tests,
        "exact_tests": exact_tests,
    }


@pytest.fixture
def bootstrap_repository(
    tmp_path: Path, prepared_bootstrap_repository: dict[str, object]
) -> tuple[Path, dict[str, object], object, object]:
    root = tmp_path / "repository"
    shutil.copytree(Path(prepared_bootstrap_repository["root"]), root)
    manifest = load_activation_manifest(
        "prepared/bootstrap_activation/activation_manifest.json",
        repository_root=root,
    )
    return (
        root,
        manifest,
        prepared_bootstrap_repository["scout_tests"],
        prepared_bootstrap_repository["exact_tests"],
    )


def _success_snapshot(
    root: Path,
    *,
    intent_path: str,
    name: str,
    cluster_id: int,
    acquisition_unit_id: str | None = None,
    acquisition_lane_id: str | None = None,
    paper_lane_id: str | None = None,
    omit_acquisition_unit_id: bool = False,
    omit_acquisition_lane_id: bool = False,
) -> tuple[Path, Path, Path]:
    intent = load_submission_intent(intent_path, repository_root=root)
    unit_id = intent.unit.unit_id if acquisition_unit_id is None else acquisition_unit_id
    lane_id = intent.lane_id if acquisition_lane_id is None else acquisition_lane_id
    paper_lane = intent.lane_id if paper_lane_id is None else paper_lane_id
    row = {
        "ClusterId": cluster_id,
        "ProcId": 0,
        "JobStatus": 4,
        "ExitCode": 0,
        "HoldReason": "",
        "JobBatchName": intent.package_manifest.job_batch_name,
        "AcquisitionUnitId": unit_id,
        "AcquisitionLaneId": lane_id,
        "PaperIILaneId": paper_lane,
        "PaperIIActionId": intent.request.action_id,
        "PaperIIExecutionRequestSha256": intent.request.digest,
        "PaperIISourceEpochId": intent.source_epoch.source_epoch.source_epoch_id,
        "EnteredCurrentStatus": 2,
        "CompletionDate": 2,
        "QDate": 1,
    }
    if omit_acquisition_unit_id:
        row.pop("AcquisitionUnitId")
    if omit_acquisition_lane_id:
        row.pop("AcquisitionLaneId")
    condition = (
        f'(ClusterId == {cluster_id}) || '
        f'(JobBatchName == "{intent.package_manifest.job_batch_name}")'
    )
    lane_key = "hh" if intent.lane_id == HH_ACQUISITION_LANE_ID else "hubbard"
    lane_scopes = {
        "hh": {
            "cluster_ids": [],
            "job_batch_names": [],
            "acquisition_lane_ids": [HH_ACQUISITION_LANE_ID],
        },
        "hubbard": {
            "cluster_ids": [],
            "job_batch_names": [],
            "acquisition_lane_ids": [HUBBARD_ACQUISITION_LANE_ID],
        },
        "siam": {
            "cluster_ids": [],
            "job_batch_names": [],
            "acquisition_lane_ids": [],
        },
    }
    lane_scopes[lane_key]["cluster_ids"] = [cluster_id]
    lane_scopes[lane_key]["job_batch_names"] = [
        intent.package_manifest.job_batch_name
    ]
    attributes = ",".join(
        sorted(set(REQUIRED_PROJECTED_ATTRIBUTES) | {"PaperIILaneId"})
    )
    directory = root / "snapshots" / name
    directory.mkdir(parents=True)
    queue = directory / "queue.json"
    history = directory / "history.json"
    queue.write_text("[]\n", encoding="utf-8")
    history.write_text(json.dumps([row]) + "\n", encoding="utf-8")
    queries = {
        query: {
            "argv": [
                "condor_q" if query == "queue" else "condor_history",
                "jsstrobel",
                "-constraint",
                condition,
                "-json",
                "-attributes",
                attributes,
            ],
            "returncode": 0,
            "stderr": "",
        }
        for query in ("queue", "history")
    }
    snapshot = build_combined_snapshot(
        queue_stdout=queue,
        history_stdout=history,
        capture_spec={
            "schema": CAPTURE_SPEC_V1,
            "captured_at": "2026-08-29T03:00:00-05:00",
            "captured_at_unix_ns": 1787990400000000000 + cluster_id,
            "single_combined_snapshot": True,
            "polling_performed": False,
            "capture_counts": {"queue": 1, "history": 1},
            "reconciliation_intents": {
                intent.package_manifest.job_batch_name: snapshot_intent_binding(intent)
            },
            "scope": {"condition": condition, "lanes": lane_scopes},
            "queries": queries,
        },
    )
    path = directory / "combined.json"
    write_immutable_snapshot(path, snapshot)
    return path, queue, history


def _canonical_receipt(
    root: Path, *, intent_path: str, cluster_id: int
) -> tuple[str, Path]:
    snapshot, queue, history = _success_snapshot(
        root,
        intent_path=intent_path,
        name=f"success-{cluster_id}",
        cluster_id=cluster_id,
    )
    result = authorize_from_snapshot(
        repository_root=root,
        intent_path=intent_path,
        snapshot_path=snapshot,
        queue_stdout_path=queue,
        history_stdout_path=history,
    )
    return str(result["submission_receipt"]), snapshot


def _arm_paths(root: Path, arm: dict[str, object]) -> tuple[str, str]:
    intent = load_submission_intent(
        str(arm["intent"]["path"]), repository_root=root
    )
    preflight = str(intent.unit.payload["package_preflight"]["path"])
    work_order = Path(str(intent.unit.payload["execution_request"]["path"])).with_name(
        "work_order.json"
    )
    return preflight, work_order.as_posix()


def _materialize_worker_raw(
    root: Path,
    package: object,
    *,
    scout_tests: object,
    exact_tests: object,
    output: Path,
) -> dict[str, bytes]:
    output.mkdir(parents=True)
    order = package.order
    if package.mode == MODE_EXACT_REFERENCE:
        exact_worker.materialize_exact_reference(
            order,
            repository_root=root,
            compute=lambda current: exact_tests._exact_computation(current),
        )
        artifact = root / str(package.request.inputs["artifact_path"])
        sources = {
            "manifest.json": artifact,
            "reference_data.json": artifact.with_name("reference_data.json"),
            "exact_reference_receipt.json": artifact.with_name(
                "exact_reference_receipt.json"
            ),
            "exact_reference_work_order.json": root / package.work_order_path,
            "execution_request.json": Path(root / package.work_order_path).with_name(
                "execution_request.json"
            ),
            "source_epoch_receipt.json": root
            / package.source_receipt.receipt_path,
            "seed_qualification_receipt.json": root
            / package.seed_receipt.receipt_path,
            "seed.json": root / package.seed_receipt.artifact.path,
        }
    else:
        def execute(actual_order, repository: Path) -> None:
            scout_tests._write(
                repository / actual_order.run_part_path,
                scout_tests._fake_raw(
                    actual_order,
                    package.seed_receipt,
                    package.source_receipt.source_epoch,
                ),
            )

        scout_tests.materialize_scout(
            order, repository_root=root, execute=execute
        )
        overlay = output / "executor_overlay_identity.json"
        overlay.write_text(
            json.dumps(order.executor_overlay.to_json_dict(), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        sources = {
            "run.json": root / order.run_path,
            "completion_receipt.json": root / order.completion_receipt_path,
            "executor_overlay_identity.json": overlay,
        }
        (output / "BASE_SOURCE_VERIFIED_BEFORE_OVERLAY").write_bytes(
            promotion._BASE_SOURCE_MARKER
        )
    for name, source in sources.items():
        target = output / name
        if source != target:
            shutil.copyfile(source, target)
    (output / "REMOTE_RAW_PRESERVED").write_bytes(promotion._PRESERVATION_MARKER)
    (output / "worker_stderr.log").write_bytes(b"")
    (output / "worker_stdout.json").write_text(
        json.dumps({"status": "completed_unscored"}) + "\n", encoding="utf-8"
    )
    assert {path.name for path in output.iterdir()} == set(
        promotion._INVENTORIES[package.mode]
    )
    return {path.name: path.read_bytes() for path in output.iterdir()}


def _plan(tmp_path: Path, mode: str) -> dict[str, object]:
    lane = "paper_ii_hh_product_v1"
    request = "a" * 64
    token = request[:20]
    increment = (
        Path("output")
        / lane
        / "attempts/cluster_8123/increments"
        / f"proc0_{mode}_{request[:16]}"
    )
    archive = increment / "archives" / f"{token}.tar.gz"
    inventory = list(promotion._INVENTORIES[mode])
    core: dict[str, object] = {
        "schema": PROMOTION_PLAN_V1,
        "status": "ready_for_verified_local_archive",
        "mode": mode,
        "lane_id": lane,
        "cell_id": "cell-a",
        "action_id": f"action::{mode}",
        "execution_request_sha256": request,
        "work_order_sha256": "b" * 64,
        "source_epoch_id": "source-a",
        "source_epoch_identity_sha256": "c" * 64,
        "scheduler_identity": {
            "cluster_id": 8123,
            "process_id": 0,
            "job_batch_name": f"batch-{mode}",
        },
        "package_evidence": {
            "preflight": {
                "path": "output/package_preflight.json",
                "sha256": "1" * 64,
            },
            "work_order": {
                "path": "output/work_order.json",
                "sha256": "2" * 64,
            },
        },
        "submission_evidence": {},
        "success_evidence": {
            "snapshot": {
                "path": "output/success.json",
                "sha256": "d" * 64,
                "snapshot_sha256": "e" * 64,
            },
            "captured_at": "2026-08-29T03:00:00-05:00",
            "query": "history",
            "scheduler_row_sha256": "f" * 64,
        },
        "remote_packaging": {
            "raw_subtree": f"/home/test/raw_outputs/{mode}/{token}",
            "archive_path": f"/home/test/transfers/{mode}-{token}.tar.gz",
            "archive_temporary_path": (
                f"/home/test/transfers/{mode}-{token}.tar.gz.building.part"
            ),
            "expected_inventory": inventory,
            "expected_member_count": len(inventory),
            "regular_files_only": True,
            "forbid_links": True,
            "preserve_source": True,
            "remote_raw_evidence_policy": "preserve",
        },
        "local_transfer": {
            "final_path": archive.as_posix(),
            "part_path": f"{archive.as_posix()}.part",
            "quarantine_root": (archive.parent / "quarantine").as_posix(),
            "recovery_receipt_path": (
                archive.parent / "transfer_recovery_receipt.json"
            ).as_posix(),
            "finalization_intent_path": (
                archive.parent / "transfer_finalization_intent.json"
            ).as_posix(),
            "finalization_receipt_path": (
                archive.parent / "transfer_finalization_receipt.json"
            ).as_posix(),
            "free_floor_bytes": DEFAULT_FREE_FLOOR_BYTES,
            "ram_guard_limit_mb": 8000,
            "remote_prefix_proof_required_for_resume": True,
            "one_archive_at_a_time": True,
        },
        "promotion": {
            "increment_root": increment.as_posix(),
            "raw_output_root": (
                increment / "raw" / f"paper_ii_{mode}"
            ).as_posix(),
            "intent_path": (increment / "evidence/promotion_intent.json").as_posix(),
            "receipt_path": (increment / "evidence/promotion_receipt.json").as_posix(),
            "receipt_last_publication": True,
            "immutable_no_overwrite": True,
        },
        "scheduler_contacted": False,
        "submission_authorized": False,
        "production_submission_enabled": False,
        "remote_raw_evidence_policy": "preserve",
        "remote_cleanup_authorized": False,
        "qiskit_compilation": False,
    }
    return {**core, "promotion_plan_sha256": _canonical_sha(core)}


def _raw_files(mode: str) -> dict[str, bytes]:
    files = {
        name: f"fixture:{mode}:{name}\n".encode()
        for name in promotion._INVENTORIES[mode]
    }
    files["REMOTE_RAW_PRESERVED"] = promotion._PRESERVATION_MARKER
    if mode == MODE_AVQDS_PREFIX:
        files["BASE_SOURCE_VERIFIED_BEFORE_OVERLAY"] = (
            promotion._BASE_SOURCE_MARKER
        )
    return files


def _archive(
    path: Path,
    files: dict[str, bytes],
    *,
    symlink_name: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as bundle:
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.mtime = 0
            if name == symlink_name:
                info.type = tarfile.SYMTYPE
                info.linkname = "../outside"
                bundle.addfile(info)
            else:
                info.size = len(files[name])
                bundle.addfile(info, io.BytesIO(files[name]))


def _identity(
    plan: dict[str, object], archive: Path, files: dict[str, bytes]
) -> dict[str, object]:
    core: dict[str, object] = {
        "schema": REMOTE_ARCHIVE_IDENTITY_V1,
        "promotion_plan_sha256": plan["promotion_plan_sha256"],
        "remote_path": plan["remote_packaging"]["archive_path"],
        "source_raw_subtree": plan["remote_packaging"]["raw_subtree"],
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "tar_member_count": len(files),
        "gzip_integrity": True,
        "tar_integrity": True,
        "remote_raw_preserved": True,
        "remote_cleanup_performed": False,
        "members": {
            name: {
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "type": "regular",
            }
            for name, content in sorted(files.items())
        },
    }
    return {**core, "archive_identity_sha256": _canonical_sha(core)}


def _finalize(
    tmp_path: Path,
    plan: dict[str, object],
    identity: dict[str, object],
    source: Path,
):
    spec = build_archive_transfer_spec(
        repository_root=tmp_path,
        plan=plan,
        remote_identity=identity,
    )
    spec.part_path.parent.mkdir(parents=True, exist_ok=True)
    spec.part_path.write_bytes(source.read_bytes())
    finalize_archive_transfer(spec)
    return spec


@pytest.mark.parametrize(
    "mode",
    [MODE_EXACT_REFERENCE, MODE_AVQDS_PREFIX, MODE_AVQDS_TRAJECTORY],
)
def test_all_result_modes_validate_guarded_exact_inventory(
    tmp_path: Path, mode: str
) -> None:
    plan = _plan(tmp_path, mode)
    files = _raw_files(mode)
    source = tmp_path / f"source-{mode}.tar.gz"
    _archive(source, files)
    identity = _identity(plan, source, files)
    _finalize(tmp_path, plan, identity, source)

    normalized = validate_remote_archive_identity(plan, identity)
    result = validate_finalized_archive(
        repository_root=tmp_path,
        plan=plan,
        remote_identity=identity,
    )

    assert normalized["archive_sha256"] == identity["archive_sha256"]
    assert result["archive_sha256"] == identity["archive_sha256"]
    assert tuple(result["members"]) == promotion._INVENTORIES[mode]
    assert result["remote_raw_preserved"] is True


@pytest.mark.parametrize("mode", [MODE_EXACT_REFERENCE, MODE_AVQDS_PREFIX])
def test_receipt_last_promotion_replays_without_overwrite(
    tmp_path: Path,
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path, mode)
    plan_path = tmp_path / "output/promotion_plan.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    files = _raw_files(mode)
    source = tmp_path / f"source-{mode}.tar.gz"
    _archive(source, files)
    identity = _identity(plan, source, files)
    identity_path = tmp_path / "output/archive_identity.json"
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    _finalize(tmp_path, plan, identity, source)

    monkeypatch.setattr(promotion, "load_promotion_plan", lambda *_a, **_k: plan)
    monkeypatch.setattr(
        promotion,
        "_load_package_context",
        lambda *_a, **_k: SimpleNamespace(mode=mode),
    )
    monkeypatch.setattr(promotion, "enforce_local_storage_floor", lambda *_a, **_k: None)

    def extract(**kwargs):
        return promotion._extract_verified_archive(
            repository_root=kwargs["repository_root"],
            plan=plan,
            identity=identity,
        )

    monkeypatch.setattr(promotion, "_extract_under_ram_guard", extract)
    canonical_dir = Path("output/canonical") / mode
    canonical = (
        (
            "run.json"
            if "run.json" in files
            else "reference_data.json",
            (canonical_dir / "run.json").as_posix(),
        ),
        (
            "completion_receipt.json"
            if "completion_receipt.json" in files
            else "exact_reference_receipt.json",
            (canonical_dir / "completion_receipt.json").as_posix(),
        ),
    )

    def validate_worker(_root, _package, raw_root):
        mappings = tuple((raw_root / name, destination) for name, destination in canonical)
        return (
            {destination: hashlib.sha256(source.read_bytes()).hexdigest()
             for source, destination in mappings},
            mappings,
        )

    monkeypatch.setattr(promotion, "_validate_worker_outputs", validate_worker)
    monkeypatch.setattr(promotion, "_validate_canonical_outputs", lambda *_a, **_k: object())

    first = promote_result(
        repository_root=tmp_path,
        plan_path=plan_path,
        remote_identity_path=identity_path,
    )
    repeated = promote_result(
        repository_root=tmp_path,
        plan_path=plan_path,
        remote_identity_path=identity_path,
    )

    assert first == repeated
    assert first["schema"] == PROMOTION_RECEIPT_V1
    assert first["status"] == "promoted_completed_unscored"
    assert first["remote_raw_preserved"] is True
    assert first["production_submission_enabled"] is False
    for _name, destination in canonical:
        assert (tmp_path / destination).is_file()
    drifted = tmp_path / canonical[0][1]
    drifted.write_bytes(drifted.read_bytes() + b"\ncoherent-drift\n")
    with pytest.raises(ResultPromotionError, match="canonical artifact.*digest changed"):
        promote_result(
            repository_root=tmp_path,
            plan_path=plan_path,
            remote_identity_path=identity_path,
        )


def test_stale_member_digest_and_link_archive_fail_closed(tmp_path: Path) -> None:
    mode = MODE_AVQDS_TRAJECTORY
    plan = _plan(tmp_path, mode)
    files = _raw_files(mode)
    source = tmp_path / "source.tar.gz"
    _archive(source, files)
    identity = _identity(plan, source, files)
    stale = json.loads(json.dumps(identity))
    stale["members"]["run.json"]["sha256"] = "0" * 64
    unsigned = {
        key: value
        for key, value in stale.items()
        if key != "archive_identity_sha256"
    }
    stale["archive_identity_sha256"] = _canonical_sha(unsigned)
    _finalize(tmp_path, plan, identity, source)
    with pytest.raises(ResultPromotionError, match="member digest changed"):
        validate_finalized_archive(
            repository_root=tmp_path,
            plan=plan,
            remote_identity=stale,
        )

    other = _plan(tmp_path, MODE_AVQDS_PREFIX)
    other_files = _raw_files(MODE_AVQDS_PREFIX)
    linked = tmp_path / "linked.tar.gz"
    _archive(linked, other_files, symlink_name="run.json")
    linked_identity = _identity(other, linked, other_files)
    _finalize(tmp_path, other, linked_identity, linked)
    with pytest.raises(ResultPromotionError, match="regular files"):
        validate_finalized_archive(
            repository_root=tmp_path,
            plan=other,
            remote_identity=linked_identity,
        )


def test_path_traversal_inventory_is_rejected_before_extraction(tmp_path: Path) -> None:
    plan = _plan(tmp_path, MODE_EXACT_REFERENCE)
    files = _raw_files(MODE_EXACT_REFERENCE)
    source = tmp_path / "traversal.tar.gz"
    bad = dict(files)
    bad["../manifest.json"] = bad.pop("manifest.json")
    _archive(source, bad)
    identity = _identity(plan, source, bad)

    with pytest.raises(ResultPromotionError, match="inventory"):
        validate_remote_archive_identity(plan, identity)


def test_real_bootstrap_exact_and_scout_chains_promote_with_typed_workers(
    bootstrap_repository: tuple[Path, dict[str, object], object, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest, scout_tests, exact_tests = bootstrap_repository
    pair = manifest["pairs"][0]
    monkeypatch.setattr(
        promotion, "enforce_local_storage_floor", lambda *_args, **_kwargs: None
    )

    for index, (mode, arm_name) in enumerate(
        (
            (MODE_EXACT_REFERENCE, "exact_reference"),
            (MODE_AVQDS_PREFIX, "avqds_scout"),
        )
    ):
        arm = pair[arm_name]
        intent_path = str(arm["intent"]["path"])
        receipt_path, success_path = _canonical_receipt(
            root, intent_path=intent_path, cluster_id=9300 + index
        )
        preflight, work_order = _arm_paths(root, arm)
        plan_path = root / f"output/promotion/{mode}/plan.json"
        plan = build_promotion_plan(
            repository_root=root,
            mode=mode,
            package_preflight_path=preflight,
            work_order_path=work_order,
            submission_receipt_path=receipt_path,
            success_snapshot_path=success_path,
            output_path=plan_path,
        )
        intent = load_submission_intent(intent_path, repository_root=root)
        assert plan["submission_evidence"]["intent"] == {
            "path": intent.path,
            "sha256": intent.intent_sha256,
            "file_sha256": intent.file_sha256,
        }
        assert plan["remote_packaging"]["raw_subtree"].startswith(
            f"/home/jsstrobel/{intent.unit.payload['remote_workdir']}/"
        )
        package = promotion._load_package_context(
            root,
            mode=mode,
            preflight_path=preflight,
            work_order_path=work_order,
        )
        raw_source = root / f"raw-source/{mode}"
        files = _materialize_worker_raw(
            root,
            package,
            scout_tests=scout_tests,
            exact_tests=exact_tests,
            output=raw_source,
        )
        source_archive = root / f"archives/{mode}.tar.gz"
        _archive(source_archive, files)
        identity = _identity(plan, source_archive, files)
        identity_path = root / f"output/promotion/{mode}/archive_identity.json"
        identity_path.write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _finalize(root, plan, identity, source_archive)

        def extract(**kwargs):
            loaded_plan = load_promotion_plan(
                kwargs["plan_path"], repository_root=kwargs["repository_root"]
            )
            return promotion._extract_verified_archive(
                repository_root=kwargs["repository_root"],
                plan=loaded_plan,
                identity=identity,
            )

        monkeypatch.setattr(promotion, "_extract_under_ram_guard", extract)
        result = promote_result(
            repository_root=root,
            plan_path=plan_path,
            remote_identity_path=identity_path,
        )
        assert result["status"] == "promoted_completed_unscored"
        assert result["mode"] == mode
        if mode == MODE_AVQDS_PREFIX:
            run_path = root / package.order.run_path
            run_payload = json.loads(run_path.read_text(encoding="utf-8"))
            run_payload["promotion_replay_coherent_drift"] = True
            run_path.write_text(
                json.dumps(run_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            completion_path = root / package.order.completion_receipt_path
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            completion["run"]["sha256"] = hashlib.sha256(
                run_path.read_bytes()
            ).hexdigest()
            completion.pop("completion_receipt_sha256")
            completion["completion_receipt_sha256"] = _canonical_sha(completion)
            completion_path.write_text(
                json.dumps(completion, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            # The worker's typed validator accepts this internally coherent
            # pair, but immutable promotion provenance must still reject it.
            promotion._validate_canonical_outputs(root, package)
            with pytest.raises(
                ResultPromotionError, match="canonical artifact.*digest changed"
            ):
                promote_result(
                    repository_root=root,
                    plan_path=plan_path,
                    remote_identity_path=identity_path,
                )


def test_avqds_package_context_reloads_frozen_work_order_after_overlay_commit(
    bootstrap_repository: tuple[Path, dict[str, object], object, object],
) -> None:
    root, manifest, scout_tests, _exact_tests = bootstrap_repository
    arm = manifest["pairs"][0]["avqds_scout"]
    preflight, work_order = _arm_paths(root, arm)
    original = scout_tests.load_work_order(work_order, repository_root=root)[0]

    overlay_relative = scout_tests.EXECUTOR_OVERLAY_PATHS[2]
    overlay_path = root / overlay_relative
    overlay_path.write_text(
        overlay_path.read_text(encoding="utf-8")
        + "\n# post-submission committed overlay evolution\n",
        encoding="utf-8",
    )
    subprocess.run(
        ("git", "add", overlay_relative), cwd=root, check=True, capture_output=True
    )
    subprocess.run(
        ("git", "commit", "-qm", "post-submission overlay evolution"),
        cwd=root,
        check=True,
        capture_output=True,
    )

    with pytest.raises(ProspectiveExecutionError, match="overlay bytes changed"):
        scout_tests.load_work_order(work_order, repository_root=root)

    package = promotion._load_package_context(
        root,
        mode=MODE_AVQDS_PREFIX,
        preflight_path=preflight,
        work_order_path=work_order,
    )
    assert package.order.work_order_sha256 == original.work_order_sha256
    assert package.order.executor_overlay == original.executor_overlay


def test_canonical_intent_digest_and_validator_fail_closed(
    bootstrap_repository: tuple[Path, dict[str, object], object, object],
) -> None:
    root, manifest, _scout_tests, _exact_tests = bootstrap_repository
    arm = manifest["pairs"][0]["exact_reference"]
    intent_path = str(arm["intent"]["path"])
    receipt_path, success_path = _canonical_receipt(
        root, intent_path=intent_path, cluster_id=9400
    )
    receipt_file = root / receipt_path
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    intent = load_submission_intent(intent_path, repository_root=root)
    receipt["remote"]["recovery_evidence"]["intent"]["sha256"] = (
        intent.file_sha256
    )
    receipt_file.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    preflight, work_order = _arm_paths(root, arm)
    with pytest.raises(ResultPromotionError, match="canonical bootstrap submission"):
        build_promotion_plan(
            repository_root=root,
            mode=MODE_EXACT_REFERENCE,
            package_preflight_path=preflight,
            work_order_path=work_order,
            submission_receipt_path=receipt_path,
            success_snapshot_path=success_path,
            output_path="output/promotion/rejected.json",
        )


@pytest.mark.parametrize(
    "value",
    ["/absolute/workdir", "safe/../escape", "safe//noncanonical", "safe/$bad"],
)
def test_bootstrap_remote_workdir_rejects_noncanonical_home_paths(value: str) -> None:
    with pytest.raises(ResultPromotionError, match="relative to the remote home"):
        promotion._home_relative_remote(value, label="remote workdir")


@pytest.mark.parametrize(
    ("unit_id", "acquisition_lane", "paper_lane"),
    [
        ("", None, None),
        (None, "", None),
        (None, None, ""),
        (None, "wrong-lane", None),
        ("wrong-unit", None, None),
        (None, None, "wrong-lane"),
    ],
)
def test_success_snapshot_rejects_present_mismatched_unit_or_lane(
    bootstrap_repository: tuple[Path, dict[str, object], object, object],
    unit_id: str | None,
    acquisition_lane: str | None,
    paper_lane: str | None,
) -> None:
    root, manifest, _scout_tests, _exact_tests = bootstrap_repository
    arm = manifest["pairs"][0]["avqds_scout"]
    intent_path = str(arm["intent"]["path"])
    receipt_path, _good_success = _canonical_receipt(
        root, intent_path=intent_path, cluster_id=9500
    )
    bad_success, _queue, _history = _success_snapshot(
        root,
        intent_path=intent_path,
        name=f"bad-success-{unit_id}-{acquisition_lane}-{paper_lane}",
        cluster_id=9500,
        acquisition_unit_id=unit_id,
        acquisition_lane_id=acquisition_lane,
        paper_lane_id=paper_lane,
    )
    preflight, work_order = _arm_paths(root, arm)
    with pytest.raises(ResultPromotionError, match="acquisition-(unit|lane)"):
        build_promotion_plan(
            repository_root=root,
            mode=MODE_AVQDS_PREFIX,
            package_preflight_path=preflight,
            work_order_path=work_order,
            submission_receipt_path=receipt_path,
            success_snapshot_path=bad_success,
            output_path="output/promotion/rejected-success.json",
        )


def test_paper_lane_authenticates_rows_with_absent_acquisition_identity(
    bootstrap_repository: tuple[Path, dict[str, object], object, object],
) -> None:
    root, manifest, _scout_tests, _exact_tests = bootstrap_repository
    arm = manifest["pairs"][0]["avqds_scout"]
    intent_path = str(arm["intent"]["path"])
    receipt_path, _good_success = _canonical_receipt(
        root, intent_path=intent_path, cluster_id=9510
    )
    sparse_success, _queue, _history = _success_snapshot(
        root,
        intent_path=intent_path,
        name="absent-acquisition-identity-with-matching-paper-lane",
        cluster_id=9510,
        paper_lane_id=load_submission_intent(
            intent_path, repository_root=root
        ).lane_id,
        omit_acquisition_unit_id=True,
        omit_acquisition_lane_id=True,
    )
    preflight, work_order = _arm_paths(root, arm)
    plan = build_promotion_plan(
        repository_root=root,
        mode=MODE_AVQDS_PREFIX,
        package_preflight_path=preflight,
        work_order_path=work_order,
        submission_receipt_path=receipt_path,
        success_snapshot_path=sparse_success,
        output_path="output/promotion/accepted-absent-acquisition-identity.json",
    )
    assert plan["scheduler_identity"] == {
        "cluster_id": 9510,
        "process_id": 0,
        "job_batch_name": load_submission_intent(
            intent_path, repository_root=root
        ).package_manifest.job_batch_name,
    }
