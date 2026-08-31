"""Contracts for offline prospective seed-delta acquisition coordination."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

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
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    LANE_HH,
    LANE_HUBBARD,
)
from pipelines.time_dynamics.prospective_seed_delta_acquisition import (
    ARCHIVE_IDENTITY_V1,
    PLAN_V1,
    CoveredSeedIncrement,
    SeedDeltaAcquisitionError,
    SeedSubmissionBinding,
    assemble_seed_delta_acquisition_plan,
    build_archive_transfer_spec,
    build_delta_plan_item,
    discover_seed_deltas,
    extract_verified_seed_archive,
    load_hh_submission_binding,
    load_covered_seed_increments,
    load_seed_delta_plan,
    main,
    validate_seed_delta_plan,
    validate_remote_archive_identity,
    validate_verified_archive,
    write_seed_delta_plan,
)
from pipelines.time_dynamics import prospective_seed_delta_acquisition as coordinator
from pipelines.time_dynamics.prospective_hh_seed_ingestion import (
    PRESERVATION_MARKER,
    RAW_FILENAMES,
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


def _binding(
    *,
    lane_id: str = LANE_HH,
    cluster_id: int = 8100,
    token: str = "a",
    slug: str = "nph1__crossover",
) -> SeedSubmissionBinding:
    request_sha = token * 64
    source_sha = ("b" if token != "b" else "c") * 64
    unit_prefix = "hh-seed" if lane_id == LANE_HH else "hubbard-seed"
    return SeedSubmissionBinding(
        lane_id=lane_id,
        seed_id=f"{lane_id}::seed::{slug}",
        seed_slug=slug,
        unit_id=f"{unit_prefix}-{request_sha[:16]}",
        action_id=f"build_seed::{lane_id}::seed::{slug}::{token * 16}",
        execution_request_sha256=request_sha,
        source_epoch_id=f"{lane_id}-source-{token * 12}",
        source_epoch_identity_sha256=source_sha,
        job_batch_name=(
            f"paper-ii-seed-{lane_id.replace('_', '-')}-{request_sha[:16]}"
        ),
        cluster_id=cluster_id,
        process_id=0,
        remote_workspace=f"/home/jsstrobel/{lane_id}-{token}/source",
        execution_request_path=(
            f"output/{lane_id}/source_epochs/bootstrap_{token * 8}/"
            f"prepared_seed_actions/{slug}/execution_request.json"
        ),
        execution_request_file_sha256=("d" if token != "d" else "e") * 64,
        source_epoch_receipt_path=(
            f"output/{lane_id}/source_epochs/bootstrap_{token * 8}/"
            "source_epoch_receipt.json"
        ),
        source_epoch_receipt_file_sha256=("e" if token != "e" else "f") * 64,
        submission_receipt_path=(
            f"output/{lane_id}/acquisition_supervisor/submission_receipts/"
            f"{unit_prefix}-{request_sha[:16]}.json"
        ),
        submission_receipt_file_sha256=("f" if token != "f" else "0") * 64,
        intent_path=(
            f"output/{lane_id}/acquisition_supervisor/submission_intents/"
            f"{unit_prefix}-{request_sha[:16]}.json"
        ),
        intent_file_sha256=("1" if token != "1" else "2") * 64,
        provisional_result_path=(
            f"output/{lane_id}/acquisition_supervisor/submission_results/"
            f"{unit_prefix}-{request_sha[:16]}.json"
        ),
        provisional_result_file_sha256=("2" if token != "2" else "3") * 64,
    )


def _success_row(binding: SeedSubmissionBinding) -> dict[str, object]:
    return {
        "ClusterId": binding.cluster_id,
        "ProcId": binding.process_id,
        "JobStatus": 4,
        "ExitCode": 0,
        "JobBatchName": binding.job_batch_name,
        "AcquisitionUnitId": binding.unit_id,
        "AcquisitionLaneId": binding.lane_id,
        "PaperIIActionId": binding.action_id,
        "PaperIIExecutionRequestSha256": binding.execution_request_sha256,
        "PaperIISourceEpochId": binding.source_epoch_id,
        "CompletionDate": 100,
    }


def _snapshot(*rows: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "paper_ii_combined_scoped_scheduler_snapshot_v1",
        "snapshot_sha256": "9" * 64,
        "hh_rows": [
            row for row in rows if row.get("AcquisitionLaneId") == LANE_HH
        ],
        "hubbard_rows": [
            row
            for row in rows
            if row.get("AcquisitionLaneId") == LANE_HUBBARD
        ],
        "siam_rows": [
            {
                "ClusterId": 9999,
                "ProcId": 0,
                "JobStatus": 2,
                "AcquisitionLaneId": "siam_nb4_pilot_v1",
            }
        ],
    }


def test_discovers_new_successes_minus_durable_coverage_without_serializing_lanes(
) -> None:
    first = _binding(cluster_id=8100, token="a")
    covered = _binding(cluster_id=8101, token="b", slug="nph1__mott_dominated")
    hubbard = _binding(
        lane_id=LANE_HUBBARD,
        cluster_id=9100,
        token="c",
        slug="l4__u4__half_filled",
    )
    running = {**_success_row(hubbard), "JobStatus": 2}

    discovery = discover_seed_deltas(
        _snapshot(_success_row(first), _success_row(covered), running),
        submissions=(first, covered, hubbard),
        covered=(
            CoveredSeedIncrement(
                lane_id=covered.lane_id,
                cluster_id=covered.cluster_id,
                process_id=0,
                action_id=covered.action_id,
                execution_request_sha256=covered.execution_request_sha256,
                source_epoch_identity_sha256=(
                    covered.source_epoch_identity_sha256
                ),
                receipt_path="output/covered.json",
                receipt_file_sha256="8" * 64,
            ),
        ),
    )

    assert discovery.deltas == (first,)
    assert [row.disposition for row in discovery.observations] == [
        "new_success",
        "covered_success",
        "nonterminal",
    ]


@pytest.mark.parametrize(
    ("status", "exit_code", "disposition"),
    [
        (1, None, "nonterminal"),
        (2, None, "nonterminal"),
        (5, None, "held"),
        (4, 7, "terminal_failure"),
    ],
)
def test_non_successful_scheduler_rows_never_become_deltas(
    status: int, exit_code: int | None, disposition: str
) -> None:
    binding = _binding()
    row = {**_success_row(binding), "JobStatus": status}
    if exit_code is None:
        row.pop("ExitCode", None)
    else:
        row["ExitCode"] = exit_code

    discovery = discover_seed_deltas(
        _snapshot(row), submissions=(binding,), covered=()
    )

    assert discovery.deltas == ()
    assert discovery.observations[0].disposition == disposition


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("PaperIIExecutionRequestSha256", "0" * 64),
        ("PaperIIActionId", "wrong-action"),
        ("AcquisitionUnitId", "wrong-unit"),
        ("JobBatchName", "wrong-batch"),
        ("PaperIISourceEpochId", "wrong-source"),
        ("AcquisitionLaneId", LANE_HUBBARD),
    ],
)
def test_scheduler_cross_identity_rows_fail_closed(
    field: str, bad_value: str
) -> None:
    binding = _binding()
    row = {**_success_row(binding), field: bad_value}
    snapshot = _snapshot(_success_row(binding))
    # Keep the row in the intended lane list even when the lane value is stale.
    snapshot["hh_rows"] = [row]
    snapshot["hubbard_rows"] = []

    with pytest.raises(SeedDeltaAcquisitionError, match="identity"):
        discover_seed_deltas(snapshot, submissions=(binding,), covered=())


def test_scheduler_row_accepts_absent_optional_unit_and_lane_classads() -> None:
    binding = _binding()
    row = _success_row(binding)
    row.pop("AcquisitionUnitId")
    row.pop("AcquisitionLaneId")
    snapshot = _snapshot(_success_row(binding))
    snapshot["hh_rows"] = [row]
    snapshot["hubbard_rows"] = []

    discovery = discover_seed_deltas(
        snapshot, submissions=(binding,), covered=()
    )

    assert discovery.deltas == (binding,)
    assert discovery.observations[0].disposition == "new_success"


def test_duplicate_scheduler_identity_fails_closed() -> None:
    binding = _binding()
    row = _success_row(binding)
    snapshot = _snapshot(row)
    snapshot["hh_rows"] = [row, dict(row)]

    with pytest.raises(SeedDeltaAcquisitionError, match="duplicate"):
        discover_seed_deltas(snapshot, submissions=(binding,), covered=())


def test_delta_plan_derives_exact_remote_subtree_and_registered_part(
    tmp_path: Path,
) -> None:
    binding = _binding()

    item = build_delta_plan_item(tmp_path, binding)

    token = f"{LANE_HH.replace('_', '-')}-{binding.execution_request_sha256[:20]}"
    packaging = item["remote_packaging"]
    transfer = item["local_transfer"]
    assert packaging["raw_subtree"] == (
        f"{binding.remote_workspace}/raw_outputs/paper_ii_seed/{token}"
    )
    assert packaging["archive_path"].endswith(f"/{token}.tar.gz")
    assert packaging["preserve_source"] is True
    assert packaging["expected_inventory"] == sorted(RAW_FILENAMES)
    assert transfer["part_path"] == f"{transfer['final_path']}.part"
    assert transfer["free_floor_bytes"] == DEFAULT_FREE_FLOOR_BYTES
    assert transfer["ram_guard_limit_mb"] == 8000
    assert transfer["one_archive_at_a_time"] is True
    assert item["plan_item_sha256"] == _canonical_sha(
        {key: value for key, value in item.items() if key != "plan_item_sha256"}
    )


def _archive(path: Path, files: dict[str, bytes], *, symlink: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.mtime = 0
            if symlink and name == "seed.json":
                info.type = tarfile.SYMTYPE
                info.linkname = "../outside"
                archive.addfile(info)
            else:
                info.size = len(files[name])
                archive.addfile(info, io.BytesIO(files[name]))


def _raw_files() -> dict[str, bytes]:
    return {
        "REMOTE_RAW_PRESERVED": PRESERVATION_MARKER,
        "seed.json": b'{"seed": 1}\n',
        "seed_qualification_receipt.json": b'{"receipt": 1}\n',
        "worker_stderr.log": b"",
        "worker_stdout.json": b'{"status": "qualified"}\n',
    }


def _identity(item: dict[str, object], archive: Path, files: dict[str, bytes]) -> dict:
    core = {
        "schema": ARCHIVE_IDENTITY_V1,
        "plan_item_sha256": item["plan_item_sha256"],
        "remote_path": item["remote_packaging"]["archive_path"],
        "source_raw_subtree": item["remote_packaging"]["raw_subtree"],
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "tar_member_count": len(files),
        "gzip_integrity": True,
        "tar_integrity": True,
        "remote_raw_preserved": True,
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


def _finalized_spec(
    tmp_path: Path,
    item: dict[str, object],
    source_archive: Path,
    identity: dict,
):
    spec = build_archive_transfer_spec(
        repository_root=tmp_path,
        plan_item=item,
        remote_identity=identity,
    )
    spec.part_path.parent.mkdir(parents=True, exist_ok=True)
    spec.part_path.write_bytes(source_archive.read_bytes())
    finalize_archive_transfer(spec)
    return spec


def test_verified_archive_extracts_exact_inventory_and_replays_idempotently(
    tmp_path: Path,
) -> None:
    binding = _binding()
    item = build_delta_plan_item(tmp_path, binding)
    files = _raw_files()
    source_archive = tmp_path / "source.tar.gz"
    _archive(source_archive, files)
    identity = _identity(item, source_archive, files)
    spec = _finalized_spec(tmp_path, item, source_archive, identity)

    validate_remote_archive_identity(item, identity)
    validated = validate_verified_archive(spec, item, identity)
    first = extract_verified_seed_archive(spec, item, identity)
    repeated = extract_verified_seed_archive(spec, item, identity)

    assert validated["archive_sha256"] == identity["archive_sha256"]
    assert first == repeated
    raw_root = tmp_path / item["ingestion"]["raw_output_root"]
    assert {path.name for path in raw_root.iterdir()} == set(RAW_FILENAMES)


@pytest.mark.parametrize("kind", ["traversal", "symlink"])
def test_archive_traversal_and_links_fail_closed(tmp_path: Path, kind: str) -> None:
    binding = _binding()
    item = build_delta_plan_item(tmp_path, binding)
    files = _raw_files()
    if kind == "traversal":
        files = dict(files)
        files["../outside"] = files.pop("seed.json")
    source_archive = tmp_path / f"{kind}.tar.gz"
    _archive(source_archive, files, symlink=kind == "symlink")
    identity = _identity(item, source_archive, files)
    # The remote identity itself is rejected for the traversal inventory; a
    # symlink-shaped archive is rejected after an otherwise canonical identity.
    if kind == "traversal":
        with pytest.raises(SeedDeltaAcquisitionError, match="inventory"):
            validate_remote_archive_identity(item, identity)
    else:
        identity = _identity(item, source_archive, _raw_files())
        spec = _finalized_spec(tmp_path, item, source_archive, identity)
        with pytest.raises(SeedDeltaAcquisitionError, match="regular"):
            validate_verified_archive(spec, item, identity)


def test_wrong_inventory_or_digest_fails_closed(tmp_path: Path) -> None:
    binding = _binding()
    item = build_delta_plan_item(tmp_path, binding)
    files = _raw_files()
    source_archive = tmp_path / "source.tar.gz"
    _archive(source_archive, files)
    identity = _identity(item, source_archive, files)

    wrong_inventory = json.loads(json.dumps(identity))
    wrong_inventory["members"].pop("worker_stderr.log")
    unsigned = {
        key: value
        for key, value in wrong_inventory.items()
        if key != "archive_identity_sha256"
    }
    wrong_inventory["archive_identity_sha256"] = _canonical_sha(unsigned)
    with pytest.raises(SeedDeltaAcquisitionError, match="inventory"):
        validate_remote_archive_identity(item, wrong_inventory)

    wrong_digest = json.loads(json.dumps(identity))
    wrong_digest["archive_sha256"] = "0" * 64
    unsigned = {
        key: value
        for key, value in wrong_digest.items()
        if key != "archive_identity_sha256"
    }
    wrong_digest["archive_identity_sha256"] = _canonical_sha(unsigned)
    spec = build_archive_transfer_spec(
        repository_root=tmp_path,
        plan_item=item,
        remote_identity=wrong_digest,
    )
    spec.local_final_path.parent.mkdir(parents=True, exist_ok=True)
    spec.local_final_path.write_bytes(source_archive.read_bytes())
    with pytest.raises(Exception, match="binding mismatch"):
        validate_verified_archive(spec, item, wrong_digest)


def test_plan_schema_constant_is_stable() -> None:
    assert PLAN_V1 == "paper_ii_prospective_seed_delta_acquisition_plan_v1"


def _materialize_binding_files(
    root: Path, binding: SeedSubmissionBinding
) -> SeedSubmissionBinding:
    replacements: dict[str, str] = {}
    for path_field, digest_field in (
        ("execution_request_path", "execution_request_file_sha256"),
        ("source_epoch_receipt_path", "source_epoch_receipt_file_sha256"),
        ("submission_receipt_path", "submission_receipt_file_sha256"),
        ("intent_path", "intent_file_sha256"),
        ("provisional_result_path", "provisional_result_file_sha256"),
    ):
        path = root / getattr(binding, path_field)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f"{path_field}:{binding.unit_id}\n".encode()
        path.write_bytes(content)
        replacements[digest_field] = hashlib.sha256(content).hexdigest()
    return replace(binding, **replacements)


def _combined_snapshot(
    root: Path, submissions: tuple[SeedSubmissionBinding, ...]
) -> Path:
    condition = " || ".join(
        f"ClusterId == {row.cluster_id}" for row in submissions
    )
    attributes = ",".join(sorted(REQUIRED_PROJECTED_ATTRIBUTES))
    rows = [_success_row(row) for row in submissions]
    queue = root / "scheduler/queue.json"
    history = root / "scheduler/history.json"
    queue.parent.mkdir(parents=True)
    queue.write_text("[]\n", encoding="utf-8")
    history.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    spec = {
        "schema": CAPTURE_SPEC_V1,
        "captured_at": "2026-08-29T02:00:00-05:00",
        "captured_at_unix_ns": 1787986800000000000,
        "single_combined_snapshot": True,
        "polling_performed": False,
        "capture_counts": {"queue": 1, "history": 1},
        "scope": {
            "condition": condition,
            "lanes": {
                "hh": {
                    "cluster_ids": [
                        row.cluster_id
                        for row in submissions
                        if row.lane_id == LANE_HH
                    ],
                    "job_batch_names": [],
                    "acquisition_lane_ids": [HH_ACQUISITION_LANE_ID],
                },
                "hubbard": {
                    "cluster_ids": [
                        row.cluster_id
                        for row in submissions
                        if row.lane_id == LANE_HUBBARD
                    ],
                    "job_batch_names": [],
                    "acquisition_lane_ids": [HUBBARD_ACQUISITION_LANE_ID],
                },
                "siam": {
                    "cluster_ids": [],
                    "job_batch_names": [],
                    "acquisition_lane_ids": [],
                },
            },
        },
        "queries": {
            name: {
                "argv": [
                    "condor_q" if name == "queue" else "condor_history",
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
            for name in ("queue", "history")
        },
    }
    snapshot = build_combined_snapshot(
        queue_stdout=queue, history_stdout=history, capture_spec=spec
    )
    path = root / "scheduler/combined.json"
    write_immutable_snapshot(path, snapshot)
    return path


def _seven_seed_plan(
    root: Path, *, cluster_base: int = 8300
) -> tuple[dict[str, object], tuple[SeedSubmissionBinding, ...]]:
    hh_slugs = (
        "nph1__crossover",
        "nph1__mott_dominated",
        "nph1__phonon_dominated",
        "nph3__crossover",
        "nph3__mott_dominated",
        "nph3__phonon_dominated",
    )
    bindings = tuple(
        _materialize_binding_files(
            root,
            _binding(cluster_id=cluster_base + index, token=token, slug=slug),
        )
        for index, (token, slug) in enumerate(zip("abcdef", hh_slugs))
    ) + (
        _materialize_binding_files(
            root,
            _binding(
                lane_id=LANE_HUBBARD,
                cluster_id=cluster_base + 100,
                token="7",
                slug="l4__u4__half_filled",
            ),
        ),
    )
    snapshot = _combined_snapshot(root, bindings)
    return (
        assemble_seed_delta_acquisition_plan(
            repository_root=root,
            snapshot_path=snapshot,
            submissions=bindings,
            covered=(),
        ),
        bindings,
    )


def _reseal(payload: dict[str, object], digest_field: str) -> None:
    unsigned = {key: value for key, value in payload.items() if key != digest_field}
    payload[digest_field] = _canonical_sha(unsigned)


def test_persists_and_replays_a_valid_seven_seed_plan(tmp_path: Path) -> None:
    hh_slugs = (
        "nph1__crossover",
        "nph1__mott_dominated",
        "nph1__phonon_dominated",
        "nph3__crossover",
        "nph3__mott_dominated",
        "nph3__phonon_dominated",
    )
    bindings = tuple(
        _materialize_binding_files(
            tmp_path,
            _binding(cluster_id=8100 + index, token=token, slug=slug),
        )
        for index, (token, slug) in enumerate(zip("abcdef", hh_slugs))
    ) + (
        _materialize_binding_files(
            tmp_path,
            _binding(
                lane_id=LANE_HUBBARD,
                cluster_id=9100,
                token="7",
                slug="l4__u4__half_filled",
            ),
        ),
    )
    snapshot = _combined_snapshot(tmp_path, bindings)
    plan = assemble_seed_delta_acquisition_plan(
        repository_root=tmp_path,
        snapshot_path=snapshot,
        submissions=bindings,
        covered=(),
    )
    output = tmp_path / "output/plans/seed_delta_plan.json"

    assert write_seed_delta_plan(output, plan, repository_root=tmp_path) == "created"
    assert write_seed_delta_plan(output, plan, repository_root=tmp_path) == "unchanged"
    assert load_seed_delta_plan(output, repository_root=tmp_path) == plan
    assert plan["delta_count"] == 7


def test_plan_validation_recomputes_deltas_from_bound_snapshot(
    tmp_path: Path,
) -> None:
    plan, _ = _seven_seed_plan(tmp_path)
    item = json.loads(json.dumps(plan["deltas"][0]))
    item["scheduler_identity"]["cluster_id"] = 999_999
    item["ingestion"]["cluster_id"] = 999_999
    item["local_transfer"]["archive_transfer_key"] = [
        item["lane_id"],
        999_999,
        [0],
    ]
    _reseal(item, "plan_item_sha256")
    plan["deltas"][0] = item
    _reseal(plan, "plan_sha256")

    with pytest.raises(SeedDeltaAcquisitionError, match="scheduler snapshot"):
        validate_seed_delta_plan(plan, repository_root=tmp_path)


@pytest.mark.parametrize(
    "field",
    [
        "lane_id",
        "unit_id",
        "expected_action_id",
        "expected_execution_request_sha256",
        "source_epoch_id",
        "expected_source_epoch_identity_sha256",
    ],
)
def test_plan_item_cross_checks_submission_and_ingestion_identity(
    tmp_path: Path, field: str
) -> None:
    item = build_delta_plan_item(tmp_path, _binding())
    item["ingestion"][field] = (
        "0" * 64 if field.endswith("sha256") else "different-identity"
    )
    _reseal(item, "plan_item_sha256")

    with pytest.raises(SeedDeltaAcquisitionError, match="policy"):
        coordinator._validate_plan_item(item)


def _minimal_covered_receipt(root: Path) -> Path:
    increment = (
        root
        / f"output/{LANE_HH}/attempts/cluster_8400/increments/proc0"
    )
    raw = increment / "raw/paper_ii_seed"
    raw.mkdir(parents=True)
    request = root / "output/evidence/request.json"
    source = root / "output/evidence/source.json"
    request.parent.mkdir(parents=True)
    request.write_text("{}\n", encoding="utf-8")
    source.write_text("{}\n", encoding="utf-8")
    artifact = {"path": "output/seeds/seed.json", "sha256": "3" * 64}
    qualification = {
        "path": "output/seeds/seed_qualification_receipt.json",
        "sha256": "4" * 64,
        "seed_receipt_sha256": "5" * 64,
    }
    core = {
        "schema": coordinator.HH_SEED_INCREMENT_RECEIPT_V1,
        "status": "complete",
        "lane_id": LANE_HH,
        "cluster_id": 8400,
        "process_ids": [0],
        "action_id": "build_seed::paper_ii_hh_product_v1::seed::test",
        "execution_request": {
            "path": request.relative_to(root).as_posix(),
            "execution_request_sha256": "1" * 64,
        },
        "source_epoch_receipt": {
            "path": source.relative_to(root).as_posix(),
            "source_epoch_identity_sha256": "2" * 64,
        },
        "seed": {
            "artifact": artifact,
            "qualification_receipt": qualification,
        },
        "raw_evidence": {
            "root": raw.relative_to(root).as_posix(),
            "remote_raw_preserved": True,
        },
        "canonical_pair_promoted_atomically": True,
        "qiskit_compilation_used": False,
    }
    payload = {**core, "increment_receipt_sha256": _canonical_sha(core)}
    path = increment / "evidence/increment_receipt.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def test_covered_receipt_replays_raw_and_canonical_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    receipt_path = _minimal_covered_receipt(tmp_path)
    calls: list[dict] = []

    def replay(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            increment_receipt_path=receipt_path.relative_to(tmp_path).as_posix(),
            increment_receipt_sha256=hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest(),
            seed_receipt=SimpleNamespace(
                artifact=SimpleNamespace(
                    to_json_dict=lambda: {
                        "path": "output/seeds/seed.json",
                        "sha256": "3" * 64,
                    }
                ),
                receipt_path="output/seeds/seed_qualification_receipt.json",
                receipt_sha256="4" * 64,
            ),
        )

    monkeypatch.setattr(coordinator, "ingest_hh_seed_delta", replay)
    covered = load_covered_seed_increments(repository_root=tmp_path)

    assert len(covered) == 1
    assert len(calls) == 1
    assert calls[0]["raw_output_root"].endswith("raw/paper_ii_seed")


def test_covered_receipt_rejects_failed_canonical_replay(
    tmp_path: Path, monkeypatch
) -> None:
    _minimal_covered_receipt(tmp_path)

    def reject(**_kwargs):
        raise coordinator.HHSeedIngestionError("canonical artifact changed")

    monkeypatch.setattr(coordinator, "ingest_hh_seed_delta", reject)
    with pytest.raises(SeedDeltaAcquisitionError, match="typed replay"):
        load_covered_seed_increments(repository_root=tmp_path)


def test_covered_receipt_rejects_symlinked_parent(tmp_path: Path) -> None:
    target = tmp_path / "redirected_attempts"
    receipt = (
        target / "cluster_8400/increments/proc0/evidence/increment_receipt.json"
    )
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}\n", encoding="utf-8")
    lane = tmp_path / f"output/{LANE_HH}"
    lane.mkdir(parents=True)
    (lane / "attempts").symlink_to(target, target_is_directory=True)

    with pytest.raises(SeedDeltaAcquisitionError, match="symlink"):
        load_covered_seed_increments(repository_root=tmp_path)


def test_loads_both_checked_in_canonical_hh_receipt_epochs() -> None:
    root = Path(__file__).resolve().parents[1]
    receipts = sorted(
        (
            root
            / "output/paper_ii_hh_product_v1/acquisition_supervisor/"
            "submission_receipts"
        ).glob("hh-seed-*.json")
    )

    loaded = tuple(
        load_hh_submission_binding(path, repository_root=root) for path in receipts
    )

    assert len(loaded) == 12
    assert {row.seed_slug for row in loaded} == {
        "nph1__crossover",
        "nph1__mott_dominated",
        "nph1__phonon_dominated",
        "nph3__crossover",
        "nph3__mott_dominated",
        "nph3__phonon_dominated",
    }
    assert {row.cluster_id for row in loaded} == {
        *range(9713875, 9713881),
        *range(9714490, 9714496),
    }
    assert {row.source_epoch_id for row in loaded} == {
        "paper-ii-hh-product-v1-source-f99b702fb9fd-9764399e09d4",
        "paper-ii-hh-product-v1-source-267516eab3b3-ecc181dd4bce",
    }


def test_ingest_and_replay_cli_modes_enter_the_ram_guard(monkeypatch) -> None:
    calls: list[str] = []

    def guarded(args: SimpleNamespace) -> int:
        calls.append(args.command)
        return 0

    monkeypatch.setattr(coordinator, "_run_ram_guarded", guarded)
    common = [
        "--repo-root",
        ".",
        "--plan",
        "plan.json",
        "--item-sha256",
        "a" * 64,
        "--remote-identity",
        "identity.json",
    ]

    assert main(["ingest-verified-archive", *common]) == 0
    assert main(["replay", *common]) == 0
    assert calls == ["ingest-verified-archive", "replay"]


def test_private_ingestion_worker_rejects_direct_unguarded_call(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv(
        "PAPER_II_SEED_DELTA_RAM_GUARD_LIMIT_MB", raising=False
    )

    with pytest.raises(SeedDeltaAcquisitionError, match="RAM guard"):
        coordinator._ingest_verified_seed_archive_under_guard(
            repository_root=tmp_path,
            plan_path="missing-plan.json",
            plan_item_sha256="a" * 64,
            remote_identity_path="missing-identity.json",
        )
    assert not hasattr(coordinator, "ingest_verified_seed_archive")


def test_verified_ingestion_dispatch_replays_idempotently(
    tmp_path: Path, monkeypatch
) -> None:
    hh_slugs = (
        "nph1__crossover",
        "nph1__mott_dominated",
        "nph1__phonon_dominated",
        "nph3__crossover",
        "nph3__mott_dominated",
        "nph3__phonon_dominated",
    )
    bindings = tuple(
        _materialize_binding_files(
            tmp_path,
            _binding(cluster_id=8200 + index, token=token, slug=slug),
        )
        for index, (token, slug) in enumerate(zip("abcdef", hh_slugs))
    ) + (
        _materialize_binding_files(
            tmp_path,
            _binding(
                lane_id=LANE_HUBBARD,
                cluster_id=9200,
                token="7",
                slug="l4__u4__half_filled",
            ),
        ),
    )
    snapshot = _combined_snapshot(tmp_path, bindings)
    plan = assemble_seed_delta_acquisition_plan(
        repository_root=tmp_path,
        snapshot_path=snapshot,
        submissions=bindings,
        covered=(),
    )
    plan_path = tmp_path / "output/plans/ingest_plan.json"
    write_seed_delta_plan(plan_path, plan, repository_root=tmp_path)
    item = plan["deltas"][0]
    files = _raw_files()
    source_archive = tmp_path / "ingest_source.tar.gz"
    _archive(source_archive, files)
    identity = _identity(item, source_archive, files)
    identity_path = tmp_path / "output/plans/remote_identity.json"
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    _finalized_spec(tmp_path, item, source_archive, identity)
    calls: list[dict] = []

    def fake_ingest(**kwargs):
        calls.append(kwargs)
        receipt_path = (
            tmp_path
            / kwargs["increment_root"]
            / "evidence/increment_receipt.json"
        )
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_bytes(b"stable increment receipt\n")
        return SimpleNamespace(
            increment_receipt_path=receipt_path.relative_to(tmp_path).as_posix(),
            increment_receipt_sha256=hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest(),
            seed_receipt=SimpleNamespace(
                receipt_path="output/seeds/seed_qualification_receipt.json",
                receipt_sha256="3" * 64,
            ),
            collection_status="pending",
        )

    monkeypatch.setattr(coordinator, "ingest_hh_seed_delta", fake_ingest)
    monkeypatch.setenv("PAPER_II_SEED_DELTA_RAM_GUARD_LIMIT_MB", "8000")
    first = coordinator._ingest_verified_seed_archive_under_guard(
        repository_root=tmp_path,
        plan_path=plan_path,
        plan_item_sha256=item["plan_item_sha256"],
        remote_identity_path=identity_path,
    )
    repeated = coordinator._ingest_verified_seed_archive_under_guard(
        repository_root=tmp_path,
        plan_path=plan_path,
        plan_item_sha256=item["plan_item_sha256"],
        remote_identity_path=identity_path,
    )

    assert repeated == first
    assert len(calls) == 2
    assert first["status"] == "verified_and_ingested"
    assert first["remote_raw_evidence_policy"] == "preserve"
