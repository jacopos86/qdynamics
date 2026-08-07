from __future__ import annotations

import copy
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = (
    REPO_ROOT
    / "chtc/paper_i_ra_adapt_repair_20260727/"
    "stationary_core_full48_r50_20260728_v8_chtc"
)
V6_PACKAGE_DIR = (
    REPO_ROOT
    / "chtc/paper_i_ra_adapt_repair_20260727/"
    "stationary_core_full48_r50_20260728_v6_chtc"
)
sys.dont_write_bytecode = True
sys.path.insert(0, str(PACKAGE_DIR))
sys.path.insert(1, str(REPO_ROOT))

import package_contract as contract  # noqa: E402
import build_attempt_selection as attempt_selection  # noqa: E402
import build_package as package_builder  # noqa: E402
import run_cell as cell_runner  # noqa: E402
import validate_fetched as fetched_validation  # noqa: E402
import validate_package as package_validation  # noqa: E402


def test_selected_core_is_exact_direct_48_without_append_dedupe() -> None:
    rows = list(contract.direct_execution_rows())
    assert len(rows) == 48
    assert len({row["execution_id"] for row in rows}) == 48
    assert {
        (row["regime_id"], row["nph"]) for row in rows
    } == set(contract.REGIME_CUTOFF_PAIRS)
    assert {row["route_id"] for row in rows} == set(contract.ROUTE_IDS)
    for regime, cutoff in contract.REGIME_CUTOFF_PAIRS:
        selected = [
            row
            for row in rows
            if row["regime_id"] == regime and row["nph"] == cutoff
        ]
        assert len(selected) == 8
        assert {row["route_id"] for row in selected} == set(
            contract.ROUTE_IDS
        )
    append_rows = [
        row for row in rows if row["route_id"] in contract.APPEND_ROUTES
    ]
    assert len(append_rows) == 12
    assert len({row["execution_id"] for row in append_rows}) == 12
    assert all(
        row["execution_entrypoint"] == "run_append_adapt"
        for row in append_rows
    )


def test_resource_envelopes_are_route_representation_and_cutoff_specific() -> None:
    rows = contract.direct_execution_rows()
    assert all(
        row["resources"]["max_runtime_seconds"] == 259_200
        for row in rows
    )
    assert {
        (
            row["execution_entrypoint"],
            row["candidate_representation"],
            row["nph"],
            row["resources"]["request_cpus"],
            row["resources"]["request_memory_mb"],
            row["resources"]["request_disk_mb"],
        )
        for row in rows
    } == {
        ("run_ra_adapt", "macro_generator_v1", 3, 4, 49_152, 61_440),
        (
            "run_ra_adapt",
            "single_pauli_word_v1",
            3,
            4,
            57_344,
            61_440,
        ),
        ("run_ra_adapt", "macro_generator_v1", 7, 4, 65_536, 81_920),
        (
            "run_ra_adapt",
            "single_pauli_word_v1",
            7,
            4,
            90_112,
            98_304,
        ),
        ("run_append_adapt", "macro_generator_v1", 3, 1, 32_768, 20_480),
        (
            "run_append_adapt",
            "single_pauli_word_v1",
            3,
            1,
            32_768,
            20_480,
        ),
        ("run_append_adapt", "macro_generator_v1", 7, 1, 65_536, 40_960),
        (
            "run_append_adapt",
            "single_pauli_word_v1",
            7,
            1,
            65_536,
            40_960,
        ),
    }


def test_submit_is_on_exit_attempt_safe_and_keeps_runtime_outside_package() -> None:
    submit = (PACKAGE_DIR / "submit.sub").read_text(encoding="utf-8")
    wrapper = (PACKAGE_DIR / "execute_source_locked_job.sh").read_text(
        encoding="utf-8"
    )
    assert "when_to_transfer_output = ON_EXIT" in submit
    assert "ON_EXIT_OR_EVICT" not in submit
    assert "$(NumJobStarts)" not in submit
    assert "stationary_core_full48_r50_20260728_v8_chtc_runtime" in submit
    assert (
        "stationary_core_full48_r50_20260728_v8_chtc/fetched" not in submit
    )
    assert (
        "stationary_core_full48_r50_20260728_v8_chtc/logs" not in submit
    )
    assert "$(ClusterId)" in submit
    assert "$(ProcId)" in submit
    assert "_CONDOR_JOB_AD" in wrapper
    assert "NumJobStarts" in wrapper
    assert "scheduler_attempt_ordinal.txt" in wrapper
    assert "__$(ClusterId)__$(ProcId).tar.gz" in submit
    assert "attempt packaging failed" in wrapper
    assert '[[ ! -s "$output_archive" ]]' in wrapper


def test_g11_ra_diagnostic_binds_only_six_canonical_artifacts_with_sidecars(
    tmp_path: Path,
) -> None:
    root = tmp_path / "g11_diagnostic"
    root.mkdir()
    canonical_names = (
        "independent.checkpoint.json",
        "independent.ledger.json",
        "resume_prefix.checkpoint.json",
        "resume_prefix.ledger.json",
        "resumed.checkpoint.json",
        "resumed.ledger.json",
    )
    for name in canonical_names:
        (root / name).write_text(
            json.dumps({"artifact": name}) + "\n",
            encoding="utf-8",
        )
    for stem, sidecars_per_family in (
        ("independent", 3),
        ("resume_prefix", 2),
        ("resumed", 2),
    ):
        for family in (
            "estimator_call_ledger_checkpoint",
            "verified_singleton_resume",
        ):
            for index in range(sidecars_per_family):
                name = (
                    f"{stem}.checkpoint.{family}."
                    f"{index:016x}.json"
                )
                (root / name).write_text(
                    json.dumps({"sidecar": name}) + "\n",
                    encoding="utf-8",
                )

    assert len(list(root.glob("*.json"))) == 20
    paths = cell_runner._g11_ra_diagnostic_artifact_paths(root)
    assert tuple(path.name for path in paths) == canonical_names
    bindings = cell_runner._diagnostic_artifact_bindings(root, paths)
    assert tuple(binding["path"] for binding in bindings) == tuple(
        f"g11_diagnostic/{name}" for name in canonical_names
    )


def test_wrapper_maps_num_job_starts_to_one_based_attempts_and_packages_failures(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "package"
    job_spec = package_dir / "jobs/test_execution.json"
    job_spec.parent.mkdir(parents=True)
    job_spec.write_text("{}\n", encoding="utf-8")
    job_ad = tmp_path / "job.ad"
    output_archive = (
        tmp_path / "transfer/test_execution__9391801__0.tar.gz"
    )
    wrapper = PACKAGE_DIR / "execute_source_locked_job.sh"

    for num_job_starts, expected_ordinal in ((0, 1), (1, 2)):
        job_ad.write_text(
            (
                f"NumJobStarts = {num_job_starts}\n"
                "ClusterId = 9391801\n"
                "ProcId = 0\n"
            ),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                "bash",
                str(wrapper),
                "package",
                "package/jobs/test_execution.json",
                "0" * 64,
                "1" * 64,
                "image.sif",
                "2" * 64,
                (
                    "transfer/"
                    "test_execution__9391801__0.tar.gz"
                ),
            ],
            cwd=tmp_path,
            env={"_CONDOR_JOB_AD": str(job_ad), "PATH": "/usr/bin:/bin"},
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 65, completed.stderr
        assert output_archive.is_file()
        with tarfile.open(output_archive, "r:gz") as archive:
            assert archive.extractfile(
                "worker_outputs/worker_exit_status.txt"
            ).read() == b"65\n"
            assert archive.extractfile(
                "worker_outputs/scheduler_attempt_ordinal.txt"
            ).read() == f"{expected_ordinal}\n".encode("ascii")


def test_wrapper_compacts_superseded_ledger_sidecars_and_retains_failure_evidence(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "package"
    fixture_root = tmp_path / "fixture"
    fake_bin = tmp_path / "bin"
    job_spec = package_dir / "jobs/test_execution.json"
    output_archive = (
        tmp_path / "transfer/test_execution__9391900__0.tar.gz"
    )
    job_ad = tmp_path / "job.ad"
    image = tmp_path / "image.sif"
    source_archive = package_dir / "source_locked.tar.gz"
    authorization = (
        package_dir / "authority/submission_authorization_receipt.json"
    )
    for directory in (
        job_spec.parent,
        authorization.parent,
        fixture_root / "g11_diagnostic",
        fake_bin,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    job_spec.write_text("{}\n", encoding="utf-8")
    source_archive.write_bytes(b"source archive fixture\n")
    image.write_bytes(b"image fixture\n")
    (package_dir / "package_manifest.json").write_text(
        "{}\n", encoding="utf-8"
    )
    authorization.write_text("{}\n", encoding="utf-8")

    retained_bytes = b'{"ledger":"terminal"}\n'
    retained_sha256 = hashlib.sha256(retained_bytes).hexdigest()
    retained_name = (
        "checkpoint.estimator_call_ledger_checkpoint."
        f"{retained_sha256[:16]}.json"
    )
    superseded_names = (
        "checkpoint.estimator_call_ledger_checkpoint."
        + hashlib.sha256(b"round one").hexdigest()[:16]
        + ".json",
        "checkpoint.estimator_call_ledger_checkpoint."
        + hashlib.sha256(b"round two").hexdigest()[:16]
        + ".json",
    )
    pointer = {
        "path": retained_name,
        "sha256": retained_sha256,
    }
    (fixture_root / "checkpoint.json").write_text(
        json.dumps(
            {
                "checkpoint": {
                    "estimator_call_ledger_checkpoint": pointer,
                },
                "adapt_vqe": {
                    "estimator_call_ledger_checkpoint": pointer,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (fixture_root / retained_name).write_bytes(retained_bytes)
    for index, name in enumerate(superseded_names, start=1):
        (fixture_root / name).write_text(
            json.dumps({"superseded_round": index}) + "\n",
            encoding="utf-8",
        )
    required_files = {
        "execution_manifest.json": b'{"manifest":true}\n',
        "estimator_ledger.json": b'{"ledger":"terminal-copy"}\n',
        "result.json": b'{"result":true}\n',
        "summary.json": b'{"summary":true}\n',
        "worker_receipt.json": b'{"receipt":true}\n',
        "failure_diagnostic.json": b'{"failure":"preserved"}\n',
        "checkpoint.verified_singleton_resume.0123456789abcdef.json": (
            b'{"resume":"preserved"}\n'
        ),
    }
    for name, data in required_files.items():
        (fixture_root / name).write_bytes(data)
    diagnostic_name = "g11_diagnostic/independent_primary.checkpoint.json"
    (fixture_root / diagnostic_name).write_text(
        '{"diagnostic":"preserved"}\n', encoding="utf-8"
    )

    fake_apptainer = fake_bin / "apptainer"
    fake_apptainer.write_text(
        """#!/bin/bash
set -euo pipefail
args=("$@")
for ((index = 0; index < ${#args[@]}; index++)); do
  if [[ "${args[$index]}" == */build_transfer_archive.py ]]; then
    exec /usr/bin/python3 "${args[@]:$index}"
  fi
done
/bin/cp -R "${FIXTURE_ROOT}/." worker_outputs/
exit "${FAKE_WORKER_STATUS}"
""",
        encoding="utf-8",
    )
    fake_apptainer.chmod(0o755)
    job_ad.write_text(
        "NumJobStarts = 0\nClusterId = 9391900\nProcId = 0\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "bash",
            str(PACKAGE_DIR / "execute_source_locked_job.sh"),
            "package",
            "package/jobs/test_execution.json",
            hashlib.sha256(job_spec.read_bytes()).hexdigest(),
            hashlib.sha256(source_archive.read_bytes()).hexdigest(),
            "image.sif",
            hashlib.sha256(image.read_bytes()).hexdigest(),
            "transfer/test_execution__9391900__0.tar.gz",
        ],
        cwd=tmp_path,
        env={
            "_CONDOR_JOB_AD": str(job_ad),
            "FAKE_WORKER_STATUS": "42",
            "FIXTURE_ROOT": str(fixture_root),
            "PATH": f"{fake_bin}:/usr/bin:/bin",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 42, completed.stderr
    with tarfile.open(output_archive, "r:gz") as archive:
        members = {
            member.name
            for member in archive
            if member.isfile()
        }
    assert f"worker_outputs/{retained_name}" in members
    assert not {
        f"worker_outputs/{name}" for name in superseded_names
    } & members
    for name in (
        "checkpoint.json",
        "execution_manifest.json",
        "estimator_ledger.json",
        "result.json",
        "summary.json",
        "worker_receipt.json",
        "failure_diagnostic.json",
        "checkpoint.verified_singleton_resume.0123456789abcdef.json",
        "attempt_identity.tsv",
        "worker_exit_status.txt",
        "scheduler_attempt_ordinal.txt",
        diagnostic_name,
    ):
        assert f"worker_outputs/{name}" in members
    assert "package/jobs/test_execution.json" in members

    stale_worker_file = tmp_path / "worker_outputs/stale_prior_attempt.txt"
    stale_source_file = (
        tmp_path / "source_locked_checkout/stale_prior_attempt.txt"
    )
    stale_worker_file.write_text("stale\n", encoding="utf-8")
    stale_source_file.parent.mkdir()
    stale_source_file.write_text("stale\n", encoding="utf-8")
    job_ad.write_text(
        "NumJobStarts = 1\nClusterId = 9391900\nProcId = 0\n",
        encoding="utf-8",
    )
    retried = subprocess.run(
        [
            "bash",
            str(PACKAGE_DIR / "execute_source_locked_job.sh"),
            "package",
            "package/jobs/test_execution.json",
            hashlib.sha256(job_spec.read_bytes()).hexdigest(),
            hashlib.sha256(source_archive.read_bytes()).hexdigest(),
            "image.sif",
            hashlib.sha256(image.read_bytes()).hexdigest(),
            "transfer/test_execution__9391900__0.tar.gz",
        ],
        cwd=tmp_path,
        env={
            "_CONDOR_JOB_AD": str(job_ad),
            "FAKE_WORKER_STATUS": "0",
            "FIXTURE_ROOT": str(fixture_root),
            "PATH": f"{fake_bin}:/usr/bin:/bin",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert retried.returncode == 0, retried.stderr
    assert not stale_worker_file.exists()
    assert not stale_source_file.parent.exists()
    with tarfile.open(output_archive, "r:gz") as archive:
        retry_members = {
            member.name
            for member in archive
            if member.isfile()
        }
        assert archive.extractfile(
            "worker_outputs/scheduler_attempt_ordinal.txt"
        ).read() == b"2\n"
    assert "worker_outputs/stale_prior_attempt.txt" not in retry_members
    assert f"worker_outputs/{retained_name}" in retry_members
    assert not {
        f"worker_outputs/{name}" for name in superseded_names
    } & retry_members

    checkpoint = json.loads(
        (fixture_root / "checkpoint.json").read_text(encoding="utf-8")
    )
    checkpoint["checkpoint"]["estimator_call_ledger_checkpoint"][
        "sha256"
    ] = "0" * 64
    checkpoint["adapt_vqe"]["estimator_call_ledger_checkpoint"][
        "sha256"
    ] = "0" * 64
    (fixture_root / "checkpoint.json").write_text(
        json.dumps(checkpoint, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    job_ad.write_text(
        "NumJobStarts = 2\nClusterId = 9391900\nProcId = 0\n",
        encoding="utf-8",
    )
    integrity_failure = subprocess.run(
        [
            "bash",
            str(PACKAGE_DIR / "execute_source_locked_job.sh"),
            "package",
            "package/jobs/test_execution.json",
            hashlib.sha256(job_spec.read_bytes()).hexdigest(),
            hashlib.sha256(source_archive.read_bytes()).hexdigest(),
            "image.sif",
            hashlib.sha256(image.read_bytes()).hexdigest(),
            "transfer/test_execution__9391900__0.tar.gz",
        ],
        cwd=tmp_path,
        env={
            "_CONDOR_JOB_AD": str(job_ad),
            "FAKE_WORKER_STATUS": "0",
            "FIXTURE_ROOT": str(fixture_root),
            "PATH": f"{fake_bin}:/usr/bin:/bin",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert integrity_failure.returncode == 71
    assert "attempt packaging failed" in integrity_failure.stderr
    assert not output_archive.exists()


def test_failed_attempt_validator_accepts_compactor_retained_worker_evidence(
    tmp_path: Path,
) -> None:
    execution_id = contract.direct_execution_ids()[0]
    name = (
        f"{execution_id}__cluster_9392023__proc_0.tar.gz"
    )
    archive_path = tmp_path / name

    def add_bytes(
        archive: tarfile.TarFile, member_name: str, data: bytes
    ) -> None:
        info = tarfile.TarInfo(member_name)
        info.size = len(data)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(data))

    with tarfile.open(archive_path, "w:gz") as archive:
        directory = tarfile.TarInfo("worker_outputs")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        archive.addfile(directory)
        for member_name, data in {
            "worker_outputs/worker_exit_status.txt": b"2\n",
            "worker_outputs/scheduler_attempt_ordinal.txt": b"1\n",
            "worker_outputs/result.json": b'{"result":true}\n',
            "worker_outputs/summary.json": b'{"summary":true}\n',
            "worker_outputs/checkpoint.json": b'{"checkpoint":true}\n',
            "worker_outputs/estimator_ledger.json": b'{"ledger":true}\n',
            "worker_outputs/execution_manifest.json": b'{"manifest":true}\n',
            "worker_outputs/worker_receipt.json": b'{"receipt":true}\n',
            "worker_outputs/failure_diagnostic.json": (
                b'{"failure":"preserved"}\n'
            ),
            (
                "worker_outputs/checkpoint."
                "estimator_call_ledger_checkpoint."
                "0123456789abcdef.json"
            ): b'{"ledger":"pointed"}\n',
            (
                "worker_outputs/checkpoint."
                "verified_singleton_resume."
                "fedcba9876543210.json"
            ): b'{"resume":"preserved"}\n',
            (
                "worker_outputs/g11_diagnostic/"
                "independent.checkpoint.json"
            ): b'{"diagnostic":"preserved"}\n',
            (
                f"{contract.PACKAGE_RELATIVE_ROOT}/jobs/"
                f"{execution_id}.json"
            ): b"{}\n",
        }.items():
            add_bytes(archive, member_name, data)

    validated = fetched_validation.validate_attempt(
        archive_path,
        relative_path=name,
    )
    assert validated["status"] == "failed_attempt_retained"
    assert validated["worker_exit_status"] == 2


def test_terminal_archive_name_ordinal_validation_and_explicit_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_fixture = tmp_path / "package"
    fetched_dir = tmp_path / "fetched"
    authority_dir = package_fixture / "authority"
    authority_dir.mkdir(parents=True)
    fetched_dir.mkdir()

    def json_bytes(payload: dict[str, object]) -> bytes:
        return contract.canonical_json_bytes(payload) + b"\n"

    manifest = contract.digested({"fixture": "package"})
    plan = contract.digested(
        {"source_archive": {"sha256": "a" * 64}}
    )
    authorization = contract.digested({"fixture": "authorization"})
    manifest_bytes = json_bytes(manifest)
    plan_bytes = json_bytes(plan)
    authorization_bytes = json_bytes(authorization)
    (package_fixture / "package_manifest.json").write_bytes(manifest_bytes)
    (package_fixture / "execution_plan.json").write_bytes(plan_bytes)
    (
        authority_dir / "submission_authorization_receipt.json"
    ).write_bytes(authorization_bytes)
    monkeypatch.setattr(
        fetched_validation, "PACKAGE_DIR", package_fixture
    )

    def add_bytes(
        archive: tarfile.TarFile, name: str, data: bytes
    ) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(data))

    def build_attempt(
        execution_id: str,
        *,
        cluster_id: int,
        proc_id: int,
        ordinal: int,
    ) -> Path:
        artifact_paths = {
            role: f"runs/{execution_id}/{role}.json"
            for role in contract.EXPECTED_ARTIFACT_ROLES
        }
        job = contract.digested(
            {
                "schema": contract.JOB_SPEC_SCHEMA,
                "execution_id": execution_id,
                "artifact_paths": artifact_paths,
            }
        )
        job_bytes = json_bytes(job)
        bindings: list[dict[str, object]] = []
        artifact_bytes: dict[str, bytes] = {}
        for role in contract.EXPECTED_ARTIFACT_ROLES:
            local_name = (
                "execution_manifest.json"
                if role == "execution_manifest"
                else f"{role}.json"
            )
            data = json_bytes(
                {"execution_id": execution_id, "role": role}
            )
            artifact_bytes[local_name] = data
            bindings.append(
                {
                    "role": role,
                    "path": local_name,
                    "declared_canonical_path": artifact_paths[role],
                    "mapping_kind": (
                        "worker_archive_copy_of_declared_output_v1"
                    ),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                }
            )
        worker_receipt = contract.digested(
            {
                "schema": contract.WORKER_RECEIPT_SCHEMA,
                "package_id": contract.PACKAGE_ID,
                "execution_id": execution_id,
                "scheduler_attempt_ordinal": ordinal,
                "job_spec_path": f"jobs/{execution_id}.json",
                "job_spec_sha256": job["sha256"],
                "job_spec_file_sha256": hashlib.sha256(
                    job_bytes
                ).hexdigest(),
                "package_manifest_sha256": manifest["sha256"],
                "package_manifest_file_sha256": hashlib.sha256(
                    manifest_bytes
                ).hexdigest(),
                "execution_plan_sha256": plan["sha256"],
                "execution_plan_file_sha256": hashlib.sha256(
                    plan_bytes
                ).hexdigest(),
                "submission_authorization_sha256": authorization[
                    "sha256"
                ],
                "submission_authorization_file_sha256": hashlib.sha256(
                    authorization_bytes
                ).hexdigest(),
                "source_archive_sha256": "a" * 64,
                "artifact_bindings": bindings,
                "status": "passed",
            }
        )
        name = (
            f"{execution_id}__cluster_{cluster_id}"
            f"__proc_{proc_id}.tar.gz"
        )
        destination = fetched_dir / name
        with tarfile.open(destination, "w:gz") as archive:
            directory = tarfile.TarInfo("worker_outputs")
            directory.type = tarfile.DIRTYPE
            directory.mode = 0o755
            archive.addfile(directory)
            add_bytes(
                archive,
                "worker_outputs/worker_exit_status.txt",
                b"0\n",
            )
            add_bytes(
                archive,
                "worker_outputs/scheduler_attempt_ordinal.txt",
                f"{ordinal}\n".encode("ascii"),
            )
            add_bytes(
                archive,
                "worker_outputs/worker_receipt.json",
                json_bytes(worker_receipt),
            )
            for local_name, data in artifact_bytes.items():
                add_bytes(
                    archive, f"worker_outputs/{local_name}", data
                )
            add_bytes(
                archive,
                (
                    "worker_outputs/checkpoint."
                    "verified_singleton_resume."
                    "0123456789abcdef.json"
                ),
                b'{"resume":"preserved"}\n',
            )
            add_bytes(
                archive,
                (
                    "worker_outputs/g11_diagnostic/"
                    "independent.checkpoint.json"
                ),
                b'{"diagnostic":"preserved"}\n',
            )
            add_bytes(
                archive,
                (
                    f"{contract.PACKAGE_RELATIVE_ROOT}/jobs/"
                    f"{execution_id}.json"
                ),
                job_bytes,
            )
        return destination

    attempts_by_execution: dict[str, list[Path]] = {}
    for proc_id, execution_id in enumerate(
        contract.direct_execution_ids()
    ):
        attempts_by_execution[execution_id] = [
            build_attempt(
                execution_id,
                cluster_id=9001,
                proc_id=proc_id,
                ordinal=1,
            )
        ]
    first_id = contract.direct_execution_ids()[0]
    attempts_by_execution[first_id].append(
        build_attempt(
            first_id,
            cluster_id=9002,
            proc_id=0,
            ordinal=2,
        )
    )

    validation_output = tmp_path / "fetched_validation.json"
    validated = fetched_validation.validate_fetched(
        fetched_dir=fetched_dir,
        output=validation_output,
    )
    assert validated["attempt_count"] == 49
    first_attempts = [
        row
        for row in validated["attempts"]
        if row["execution_id"] == first_id
    ]
    assert {
        (row["cluster_id"], row["proc_id"], row["attempt_ordinal"])
        for row in first_attempts
    } == {(9001, 0, 1), (9002, 0, 2)}

    chosen_rows = {
        row["execution_id"]: row
        for row in validated["attempts"]
        if row["status"] == "passed"
    }
    assert chosen_rows[first_id]["cluster_id"] == 9002
    choices = contract.digested(
        {
            "schema": (
                "paper_i_ra_adapt_stationary_core_"
                "explicit_attempt_choices_v1"
            ),
            "package_id": contract.PACKAGE_ID,
            "selection_authorized_by_user": True,
            "paper_evidence_adoption_authorized": False,
            "choices": {
                execution_id: {
                    "attempt_path": chosen_rows[execution_id]["path"],
                    "attempt_sha256": chosen_rows[execution_id]["sha256"],
                }
                for execution_id in contract.direct_execution_ids()
            },
        }
    )
    choices_path = tmp_path / "choices.json"
    choices_path.write_bytes(json_bytes(choices))
    selection_output = tmp_path / "selection.json"
    selection = attempt_selection.build_selection(
        validation_path=validation_output,
        choices_path=choices_path,
        output=selection_output,
    )
    assert selection["selected_count"] == 48
    selected_first = next(
        row
        for row in selection["selected_attempts"]
        if row["execution_id"] == first_id
    )
    assert "__cluster_9002__proc_0.tar.gz" in selected_first[
        "attempt_path"
    ]


def test_cache_and_bytecode_outputs_are_hard_disabled() -> None:
    for name in (
        "build_package.py",
        "run_semantic_preflight.py",
        "validate_package.py",
        "run_cell.py",
        "build_attempt_selection.py",
        "validate_fetched.py",
    ):
        source = (PACKAGE_DIR / name).read_text(encoding="utf-8")
        assert "sys.dont_write_bytecode = True" in source
    for name in ("run_semantic_preflight.py", "run_cell.py"):
        source = (PACKAGE_DIR / name).read_text(encoding="utf-8")
        assert 'os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"' in source
        assert (
            'os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"'
            in source
        )
    wrapper = (PACKAGE_DIR / "execute_source_locked_job.sh").read_text(
        encoding="utf-8"
    )
    assert "export PYTHONDONTWRITEBYTECODE=1" in wrapper
    assert "export STATIC_ADAPT_HH_POOL_CACHE=off" in wrapper
    assert "export STATIC_ADAPT_CANDIDATE_RECORD_CACHE=off" in wrapper
    assert not list(PACKAGE_DIR.rglob("__pycache__"))
    assert not list(PACKAGE_DIR.rglob("*.pyc"))


def test_source_activation_accepts_isolated_pipelines_namespace(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    concrete = source_root / "pipelines/static_adapt/ra_adapt/__init__.py"
    concrete.parent.mkdir(parents=True)
    concrete.write_text("SOURCE_LOCK_TEST = True\n", encoding="utf-8")
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(PACKAGE_DIR)!r})\n"
        "import run_cell\n"
        f"root = run_cell.Path({str(source_root)!r})\n"
        "run_cell._activate_source_root(root)\n"
        "run_cell._assert_source_locked_imports(root)\n"
        "namespace = sys.modules['pipelines']\n"
        "assert namespace.__file__ is None\n"
        "assert sys.modules['pipelines.static_adapt.ra_adapt']."
        "SOURCE_LOCK_TEST is True\n"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_authorized_worker_activates_locked_source_before_package_validation(
    tmp_path: Path,
) -> None:
    execution_id = "core__strong_weak_u8__nph3__ra_macro_append_only"
    job_path = PACKAGE_DIR / f"jobs/{execution_id}.json"
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path[:] = [\n"
        "    item for item in sys.path\n"
        "    if not (Path(item or '.').resolve() / 'pipelines').exists()\n"
        "    and not (Path(item or '.').resolve() / 'src').exists()\n"
        "]\n"
        f"sys.path.insert(0, {str(PACKAGE_DIR)!r})\n"
        "import run_cell\n"
        "import validate_package\n"
        "class ReachedPackageValidation(Exception):\n"
        "    pass\n"
        f"source_root = Path({str(tmp_path / 'source_locked_checkout')!r})\n"
        f"output_root = Path({str(tmp_path / 'worker_outputs')!r})\n"
        f"job_path = Path({str(job_path)!r})\n"
        "def stop_at_package_validation(**kwargs):\n"
        "    run_cell._assert_source_locked_imports(source_root)\n"
        "    raise ReachedPackageValidation\n"
        "validate_package.validate_package = stop_at_package_validation\n"
        "try:\n"
        "    run_cell.run_authorized_job(\n"
        "        source_root=source_root,\n"
        "        job_path=job_path,\n"
        "        expected_job_sha256=run_cell.sha256_file(job_path),\n"
        "        output_root=output_root,\n"
        "        scheduler_attempt_ordinal=1,\n"
        "        scheduler_cluster_id=1,\n"
        "        scheduler_proc_id=0,\n"
        "        verified_image_path=run_cell.REMOTE_IMAGE_PATH,\n"
        "        verified_image_sha256=run_cell.REMOTE_IMAGE_SHA256,\n"
        "    )\n"
        "except ReachedPackageValidation:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('worker did not reach package validation')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-E", "-B", "-c", script],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_p3_entrypoint_activates_explicit_repo_root_from_isolated_cwd(
    tmp_path: Path,
) -> None:
    script = (
        "import site, sys\n"
        "from pathlib import Path\n"
        "sys.path.append(site.getusersitepackages())\n"
        f"sys.path.insert(0, {str(PACKAGE_DIR)!r})\n"
        "import run_semantic_preflight as preflight\n"
        f"root = Path({str(REPO_ROOT)!r})\n"
        "preflight._activate_repo_source(root)\n"
        "preflight._assert_repo_imports(root)\n"
        "module = sys.modules['pipelines.static_adapt.ra_adapt']\n"
        "Path(module.__file__).resolve().relative_to(root.resolve())\n"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_atomic_publication_refuses_existing_and_raced_destinations(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "receipt.json"
    contract.atomic_write_json(destination, {"status": "first"})
    with pytest.raises(contract.PackageContractError, match="overwrite"):
        contract.atomic_write_json(destination, {"status": "second"})

    temporary = tmp_path / "source.tmp"
    raced = tmp_path / "raced.json"
    temporary.write_bytes(b"source")
    raced.write_bytes(b"raced")
    with pytest.raises(contract.PackageContractError, match="raced"):
        contract.atomic_publish_noreplace(temporary, raced)
    assert temporary.read_bytes() == b"source"
    assert raced.read_bytes() == b"raced"


def _g3_projection(label: str, count: int) -> dict[str, object]:
    return {
        "count": count,
        "ordered_labels_sha256": hashlib.sha256(
            f"{label}:labels".encode()
        ).hexdigest(),
        "ordered_pool_sha256": hashlib.sha256(
            f"{label}:pool".encode()
        ).hexdigest(),
    }


@pytest.mark.parametrize(
    ("entrypoint", "append"),
    (("run_ra_adapt", False), ("run_append_adapt", True)),
)
def test_g3_ra_and_append_gate_execute_and_reject_pool_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    append: bool,
) -> None:
    parent = _g3_projection("parent", 3)
    macro = _g3_projection("macro", 2)
    singleton = _g3_projection("singleton-parent", 3)
    global_children = _g3_projection("singleton-global", 7)
    construction = contract.digested(
        {
            "schema": "test_singleton_construction_equivalence_v1",
            "status": "passed",
        }
    )
    proof_row = contract.digested(
        {
            "schema": "test_regime_pool_construction_proof_v1",
            "regime_id": "weak_weak",
            "nph": 3,
            "parent_inventory": parent,
            "macro_coefficient_pool": macro,
            "singleton_parent_inventory": singleton,
            "singleton_append_global_pool": global_children,
            "singleton_construction_equivalence": construction,
            "ra_append_macro_pool_equal": True,
            "ra_append_singleton_parent_equal": True,
            "status": "passed",
        }
    )
    proof = contract.digested(
        {
            "schema": "test_six_regime_pool_construction_proof_v1",
            "rows": [proof_row],
            "status": "passed",
        }
    )
    p2 = contract.digested(
        {
            "schema": contract.P2_RECEIPT_SCHEMA,
            "six_regime_pool_construction_proof": proof,
            "six_regime_pool_construction_proof_sha256": proof["sha256"],
            "status": "passed",
        }
    )
    package = tmp_path / "package"
    p2_path = package / contract.P2_RECEIPT_RELATIVE
    p2_path.parent.mkdir(parents=True)
    p2_path.write_bytes(contract.canonical_json_bytes(p2) + b"\n")
    monkeypatch.setattr(cell_runner, "PACKAGE_DIR", package)
    job = {
        "execution_id": f"test__{entrypoint}",
        "execution_entrypoint": entrypoint,
        "regime_id": "weak_weak",
        "nph": 3,
        "candidate_representation": "single_pauli_word_v1",
    }
    expected_pool = global_children if append else singleton
    validated = cell_runner._validate_g3_pool_construction_gate(
        job=job,
        result_parent=singleton,
        result_pool=expected_pool,
        append=append,
    )
    assert validated["pool_proof_row"]["sha256"] == proof_row["sha256"]

    tampered_pool = dict(expected_pool)
    tampered_pool["ordered_pool_sha256"] = "f" * 64
    with pytest.raises(contract.PackageContractError, match="G3"):
        cell_runner._validate_g3_pool_construction_gate(
            job=job,
            result_parent=singleton,
            result_pool=tampered_pool,
            append=append,
        )


@pytest.mark.parametrize(
    ("regime_id", "nph"), contract.REGIME_CUTOFF_PAIRS
)
def test_verified_same_cutoff_ed_source_covers_all_six_regimes(
    regime_id: str,
    nph: int,
) -> None:
    authority = contract.validate_core_authority(REPO_ROOT)
    job = next(
        row
        for row in contract.direct_execution_rows()
        if row["regime_id"] == regime_id
        and row["nph"] == nph
        and row["route_id"] == "ra_macro_append_only"
    )
    source_lock = authority["source_lock_cells"][job["source_lock_id"]]
    receipt = cell_runner._verified_same_cutoff_ed_reference(
        job=job,
        source_lock=source_lock,
        source_root=REPO_ROOT,
        authority=authority,
    )
    assert receipt["regime_name"] == contract.ED_REGIME_NAME_BY_ID[
        regime_id
    ]
    assert receipt["n_ph_work"] == receipt["n_ph_reference"] == nph
    assert receipt["status"] == "passed"


def test_verified_same_cutoff_ed_source_rejects_parsed_value_tamper(
    tmp_path: Path,
) -> None:
    authority = contract.validate_core_authority(REPO_ROOT)
    job = next(
        row
        for row in contract.direct_execution_rows()
        if row["regime_id"] == "strong_weak_u8"
        and row["route_id"] == "append_singleton"
    )
    source_lock = copy.deepcopy(
        authority["source_lock_cells"][job["source_lock_id"]]
    )
    declared = source_lock["resolver_trace"][
        "same_cutoff_ed_reference"
    ]
    relative = Path(declared["path"])
    source = REPO_ROOT / relative
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    regime = next(
        row
        for row in payload["regimes"]
        if row["name"] == declared["regime_name"]
    )
    cell = next(row for row in regime["cells"] if row["M"] == job["nph"])
    cell["E_ED"] = float(cell["E_ED"]) + 0.125
    tampered_bytes = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    target.write_bytes(tampered_bytes)
    tampered_sha = hashlib.sha256(tampered_bytes).hexdigest()
    declared["sha256"] = tampered_sha
    source_lock = contract.digested(
        {
            key: value
            for key, value in source_lock.items()
            if key != "sha256"
        }
    )
    tampered_authority = copy.deepcopy(authority)
    tampered_authority["global_source_locks"]["ed_cutoff_reference"][
        "sha256"
    ] = tampered_sha
    tampered_authority["global_source_files"]["ed_cutoff_reference"].update(
        {
            "sha256": tampered_sha,
            "size_bytes": len(tampered_bytes),
        }
    )
    with pytest.raises(
        contract.PackageContractError, match="regime/cutoff/value"
    ):
        cell_runner._verified_same_cutoff_ed_reference(
            job=job,
            source_lock=source_lock,
            source_root=tmp_path,
            authority=tampered_authority,
        )


def test_source_archive_members_include_exact_global_source_bytes() -> None:
    authority = contract.validate_core_authority(REPO_ROOT)
    user_selection = contract.validate_user_selection_authority(REPO_ROOT)
    members = package_builder._source_members(
        repo_root=REPO_ROOT,
        authority=authority,
        user_selection=user_selection,
    )
    by_path = {row["path"]: row for row in members}
    for binding in authority["global_source_files"].values():
        archived = by_path[binding["path"]]
        assert archived["source_kind"] == "verified_global_source_locks"
        assert archived["sha256"] == binding["sha256"]
        assert archived["size_bytes"] == binding["size_bytes"]


def test_source_archive_members_include_append_runtime_hash_dependency() -> None:
    authority = contract.validate_core_authority(REPO_ROOT)
    user_selection = contract.validate_user_selection_authority(REPO_ROOT)
    members = package_builder._source_members(
        repo_root=REPO_ROOT,
        authority=authority,
        user_selection=user_selection,
    )
    dependency = contract.APPEND_RUNTIME_SOURCE_DEPENDENCIES[0]
    archived = {row["path"]: row for row in members}[dependency["path"]]
    assert archived == {
        **dependency,
        "source_kind": "append_runtime_hash_dependency",
    }


def test_sealed_archive_satisfies_append_source_lock_dependency_access(
    tmp_path: Path,
) -> None:
    execution_id = "core__weak_weak__nph3__append_macro"
    job_path = PACKAGE_DIR / f"jobs/{execution_id}.json"
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path[:] = [\n"
        "    item for item in sys.path\n"
        "    if not (Path(item or '.').resolve() / 'pipelines').exists()\n"
        "    and not (Path(item or '.').resolve() / 'src').exists()\n"
        "]\n"
        f"sys.path.insert(0, {str(PACKAGE_DIR)!r})\n"
        "import run_cell\n"
        f"source_root = Path({str(tmp_path / 'source_locked_checkout')!r})\n"
        "run_cell._safe_extract_source_archive(source_root)\n"
        "run_cell._activate_source_root(source_root)\n"
        f"job_path = Path({str(job_path)!r})\n"
        "job = run_cell.load_json_object(job_path, label='Append job')\n"
        "protocol_path = source_root / run_cell.safe_relative_path(\n"
        "    job['protocol']['path'], label='job protocol'\n"
        ")\n"
        "assert run_cell.sha256_file(protocol_path) == "
        "job['protocol']['sha256']\n"
        "assert protocol_path.stat().st_size == "
        "int(job['protocol']['size_bytes'])\n"
        "from pipelines.static_adapt.ra_adapt.bundles import "
        "load_validated_bundle_protocol\n"
        "from pipelines.static_adapt.ra_adapt import append\n"
        "protocol = load_validated_bundle_protocol(protocol_path)\n"
        "problem = run_cell._problem_from_protocol(protocol)\n"
        "receipts = append._source_lock_receipts(problem)\n"
        "expected = dict(protocol.source_locks)\n"
        "assert receipts\n"
        "assert all(expected[key] == value "
        "for key, value in receipts.items())\n"
        "run_cell._assert_source_locked_imports(source_root)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-E", "-B", "-c", script],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_p4_revalidates_verified_same_cutoff_ed_receipt_tamper() -> None:
    authority = contract.validate_core_authority(REPO_ROOT)
    job = next(
        row
        for row in contract.direct_execution_rows()
        if row["regime_id"] == "strong_weak_u8"
        and row["route_id"] == "ra_macro_append_only"
    )
    source_lock = authority["source_lock_cells"][job["source_lock_id"]]
    receipt = cell_runner._verified_same_cutoff_ed_reference(
        job=job,
        source_lock=source_lock,
        source_root=REPO_ROOT,
        authority=authority,
    )
    smoke = {
        "verified_same_cutoff_ed_reference": receipt,
        "verified_same_cutoff_ed_reference_sha256": receipt["sha256"],
    }
    assert (
        package_validation._validate_p4_verified_ed_reference(
            smoke=smoke,
            smoke_job=job,
            authority=authority,
        )
        == receipt["sha256"]
    )
    tampered = copy.deepcopy(smoke)
    tampered["verified_same_cutoff_ed_reference_sha256"] = "0" * 64
    with pytest.raises(contract.PackageContractError, match="P4 verified"):
        package_validation._validate_p4_verified_ed_reference(
            smoke=tampered,
            smoke_job=job,
            authority=authority,
        )


def test_g8_ra_reporting_projection_accepts_serialized_parameter_delta_only(
) -> None:
    """The retained v6 P4 bytes expose the exact production projection seam."""

    p4 = json.loads(
        (
            V6_PACKAGE_DIR
            / "authority/p4_packaged_dispatch_receipt.json"
        ).read_text(encoding="utf-8")
    )
    smoke = p4["smoke_result"]["canonical_payload"]
    summary = next(
        row["canonical_payload"]
        for row in smoke["artifact_bindings"]
        if row["role"] == "summary"
    )
    source_exact = float(
        smoke["verified_same_cutoff_ed_reference"]["E_ED"]
    )
    summary_exact = float(
        summary["provenance"]["exact_same_cutoff_energy"]
    )
    assert summary_exact != source_exact
    assert abs(summary_exact - source_exact) < 1.0e-12

    projection = cell_runner._validate_g8_ra_reporting_projection(
        summary=summary,
        source_locked_exact_energy=source_exact,
    )
    assert projection == {
        "source": "paper_i_run_summary_v1.provenance",
        "controller_decision_influence": False,
        "reporting_only": True,
        "typed_summary_exact_same_cutoff_energy": summary_exact,
        "source_locked_exact_same_cutoff_energy": source_exact,
        "absolute_delta": abs(summary_exact - source_exact),
        "relative_tolerance": 0.0,
        "absolute_tolerance": 1.0e-12,
        "matched_within_tolerance": True,
        "serialized_parameter_limitation": (
            "typed_summary_ed_uses_serialized_protocol_g_ep_while_"
            "locked_reference_ed_uses_full_precision_regime_g_ep_v1"
        ),
    }

    tampered = copy.deepcopy(summary)
    tampered["provenance"]["exact_same_cutoff_energy"] = (
        source_exact + 1.0e-10
    )
    with pytest.raises(
        contract.PackageContractError,
        match="G8 RA reporting projection drifted",
    ):
        cell_runner._validate_g8_ra_reporting_projection(
            summary=tampered,
            source_locked_exact_energy=source_exact,
        )


def _valid_p3() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    hashes = {
        name: f"{index:064x}"
        for index, name in enumerate(
            (
                "final_canonical",
                "final_file",
                "implementation",
                "generator",
                "parent3",
                "macro3",
                "singleton3",
                "poolobs3",
                "parent7",
                "macro7",
                "singleton7",
                "poolobs7",
                "fixture",
                "construction",
                "result",
                "trajectory",
                "checkpoint",
                "comparison",
                "population",
                "receipt_file",
            ),
            start=1,
        )
    }
    authority = contract.validate_core_authority(REPO_ROOT)
    protocol_bindings = authority["protocol_bindings"]
    control = contract.control_plane_receipt(PACKAGE_DIR)
    generator = next(
        row
        for row in control["files"]
        if row["path"] == "run_semantic_preflight.py"
    )
    pool_proof_rows = []
    for regime_id, nph in contract.REGIME_CUTOFF_PAIRS:
        cell_ids = {
            "ra_macro": contract.core_cell_id(
                regime_id, nph, "ra_macro_append_only"
            ),
            "append_macro": contract.core_cell_id(
                regime_id, nph, "append_macro"
            ),
            "ra_singleton": contract.core_cell_id(
                regime_id, nph, "ra_singleton_append_only"
            ),
            "append_singleton": contract.core_cell_id(
                regime_id, nph, "append_singleton"
            ),
        }
        protocols = {
            name: contract.load_json_object(
                Path(authority["bundle_root"])
                / "protocols"
                / f"{cell_id}.json",
                label=f"test P3 {name} protocol",
            )
            for name, cell_id in cell_ids.items()
        }

        def projection(value: dict[str, object]) -> dict[str, object]:
            return {
                "count": value["count"],
                "ordered_labels_sha256": value[
                    "ordered_labels_sha256"
                ],
                "ordered_pool_sha256": value["ordered_pool_sha256"],
            }

        parent = projection(protocols["ra_macro"]["parent_inventory"])
        macro = projection(protocols["ra_macro"]["executable_pool"])
        singleton_parent = projection(
            protocols["ra_singleton"]["parent_inventory"]
        )
        global_children = projection(
            protocols["append_singleton"]["executable_pool"]
        )
        construction = contract.digested(
            {
                "schema": (
                    "paper_i_stationary_core_singleton_"
                    "construction_equivalence_v1"
                ),
                "regime_id": regime_id,
                "nph": nph,
                "ra_staged_child_pool": global_children,
                "append_global_child_pool": global_children,
                "ordered_child_manifest_sha256": hashes["construction"],
                "canonical_unit_pauli_representatives": True,
                "hard_guarded": True,
                "construction_equivalent_for_identical_parent_supply": (
                    True
                ),
                "status": "passed",
            }
        )
        pool_proof_rows.append(
            contract.digested(
                {
                    "schema": (
                        "paper_i_stationary_core_regime_"
                        "pool_construction_proof_v1"
                    ),
                    "regime_id": regime_id,
                    "nph": nph,
                    "problem_receipt_sha256": contract.canonical_sha256(
                        protocols["ra_macro"]["problem"]
                    ),
                    "protocol_sha256s": {
                        name: protocol_bindings[cell_id][
                            "canonical_sha256"
                        ]
                        for name, cell_id in cell_ids.items()
                    },
                    "parent_inventory": parent,
                    "macro_coefficient_pool": macro,
                    "singleton_parent_inventory": singleton_parent,
                    "singleton_append_global_pool": global_children,
                    "singleton_construction_equivalence": construction,
                    "ra_append_macro_pool_equal": True,
                    "ra_append_singleton_parent_equal": True,
                    "status": "passed",
                }
            )
        )
    pool_proof = contract.digested(
        {
            "schema": (
                "paper_i_stationary_core_six_regime_"
                "pool_construction_proof_v1"
            ),
            "regime_count": 6,
            "regime_cutoff_pairs": [
                [regime_id, nph]
                for regime_id, nph in contract.REGIME_CUTOFF_PAIRS
            ],
            "rows": pool_proof_rows,
            "macro_ra_append_equality_all_regimes": True,
            "singleton_construction_equivalence_all_regimes": True,
            "status": "passed",
        }
    )
    route_rows = []
    for route in contract.ROUTE_IDS:
        entrypoint = (
            "run_append_adapt"
            if route in contract.APPEND_ROUTES
            else "run_ra_adapt"
        )
        purposes = (
            ["fresh_execution", "independent_reconstruction"]
            if route in contract.APPEND_ROUTES
            else [
                "independent_primary",
                "fresh_resume_prefix",
                "authenticated_resume",
            ]
        )
        if route in contract.INSERTION_CAPABLE_ROUTES:
            purposes.append("g5_scored_position_witness")
        row = {
            "route_id": route,
            "candidate_representation": contract.representation_for_route(
                route
            ),
            "fixture_identity": (
                contract.P3_FIXTURE_ID
            ),
            "fixture_regime_id": contract.P3_REGIME_ID,
            "fixture_nph": contract.P3_NPH,
            "fixture_problem_receipt": contract.load_json_object(
                Path(authority["bundle_root"])
                / "protocols"
                / (
                    f"{contract.core_cell_id(contract.P3_REGIME_ID, contract.P3_NPH, route)}.json"
                ),
                label="test P3 final protocol",
            )["problem"],
            "bounded_protocol_mode": (
                "final_bundle_problem_and_source_authority_bounded_protocol_v1"
                if route in contract.APPEND_ROUTES
                else "exact_final_bundle_protocol_with_operational_round_cap_v1"
            ),
            "ordinary_smoke_controller_rounds": contract.P3_SHORT_ROUNDS,
            "final_protocol_nph": contract.P3_NPH,
            "final_protocol_cell_id": contract.core_cell_id(
                contract.P3_REGIME_ID, contract.P3_NPH, route
            ),
            "protocol_sha256": protocol_bindings[
                contract.core_cell_id(
                    contract.P3_REGIME_ID, contract.P3_NPH, route
                )
            ]["canonical_sha256"],
            "fixture_protocol_sha256": hashes["fixture"],
            "fixture_construction_sha256": hashes["construction"],
            "run_class": "smoke",
            "paper_facing_result_allowed": False,
            "maximum_controller_rounds_executed": (
                contract.P3_PLATEAU_G5_ROUNDS
                if route.endswith("plateau")
                else contract.P3_ALWAYS_G5_ROUNDS
                if route.endswith("always")
                else contract.P3_SHORT_ROUNDS
            ),
            "facade_invocations": [
                {
                    "entrypoint": entrypoint,
                    "purpose": purpose,
                    "maximum_controller_rounds": (
                        contract.P3_PLATEAU_G5_ROUNDS
                        if purpose == "g5_scored_position_witness"
                        and route.endswith("plateau")
                        else contract.P3_ALWAYS_G5_ROUNDS
                        if purpose == "g5_scored_position_witness"
                        else 1
                        if purpose == "fresh_resume_prefix"
                        else contract.P3_SHORT_ROUNDS
                    ),
                }
                for purpose in purposes
            ],
            "fresh_execution": {
                "status": "passed",
                "result_sha256": hashes["result"],
                "trajectory_sha256": hashes["trajectory"],
            },
            "independent_replay": {
                "status": "passed",
                "matched": True,
                "result_sha256": hashes["result"],
                "trajectory_sha256": hashes["trajectory"],
            },
            "status": "passed",
        }
        if route in contract.RA_ROUTES:
            row["authenticated_resume"] = {
                "status": "passed",
                "authenticated": True,
                "trajectory_prefix_matched": True,
                "checkpoint_file_sha256": hashes["checkpoint"],
                "resumed_result_sha256": hashes["result"],
                "comparison_sha256": hashes["comparison"],
            }
        else:
            row["reconstruction_boundary"] = {
                "status": "authenticated_reconstruction_only_verified",
                "public_resume_execution_supported": False,
                "reconstruction_fields_complete": True,
            }
        if route in contract.INSERTION_CAPABLE_ROUTES:
            always_route = route.endswith("always")
            row["g5_scored_position_witness"] = {
                "status": "passed",
                "aggregate_g5_passed": True,
                "execution_mode": (
                    "independent_fresh_exact_final_nph3_protocol_v1"
                ),
                "trajectory_prefix_matched": True,
                "authenticated_prefix_controller_rounds": (
                    contract.P3_SHORT_ROUNDS
                ),
                "witness_controller_rounds": (
                    contract.P3_PLATEAU_G5_ROUNDS
                    if route.endswith("plateau")
                    else contract.P3_ALWAYS_G5_ROUNDS
                ),
                "first_interior_controller_round": (
                    None if always_route else 2
                ),
                "scored_position_count": 10,
                "interior_scored_count": 0 if always_route else 1,
                "interior_witness_status": (
                    "not_serialized_by_v9_selected_phase_"
                    "population_projection"
                    if always_route
                    else "observed"
                ),
                "full_insertion_policy_verified": (
                    True if always_route else None
                ),
                **(
                    {
                        "limitation": (
                            "immutable_v9_full_insertion_population_"
                            "receipt_retains_selected_phase_records_"
                            "but_not_the_exhaustive_domain_population"
                        )
                    }
                    if always_route
                    else {}
                ),
                "population_receipt_sha256": hashes["population"],
            }
        route_rows.append(row)
    receipt = contract.digested(
        {
            "schema": contract.P3_RECEIPT_SCHEMA,
            "package_id": contract.PACKAGE_ID,
            "campaign_id": contract.CAMPAIGN_ID,
            "generator": {
                "path": "run_semantic_preflight.py",
                "sha256": generator["sha256"],
                "size_bytes": generator["size_bytes"],
            },
            "core_final_receipt_canonical_sha256": authority[
                "final_receipt_binding"
            ]["canonical_sha256"],
            "core_final_receipt_file_sha256": authority[
                "final_receipt_binding"
            ]["sha256"],
            "implementation_source_inventory_sha256": authority[
                "implementation_inventory_sha256"
            ],
            "active_gradient_policy": "stationary_source_response_v1",
            "resource_weighting_scope": "late_resource_weighting_v1",
            "execution_mode": "bounded_non_paper_semantic_preflight_v1",
            "governing_plan_p3_alignment": {
                "regime_id": contract.P3_REGIME_ID,
                "nph": contract.P3_NPH,
                "ordinary_smoke_controller_rounds": (
                    contract.P3_SHORT_ROUNDS
                ),
                "route_coverage": "all_eight_selected_routes_v1",
                "ra_protocol_authority": (
                    "exact_final_stationary_core_protocol_v1"
                ),
                "append_protocol_authority": (
                    "exact_final_problem_and_source_authority_bounded_v1"
                ),
                "g5_execution_boundary": (
                    "separate_independent_fresh_witness_v1"
                ),
                "plateau_g5_round_cap": (
                    contract.P3_PLATEAU_G5_ROUNDS
                ),
                "always_g5_round_cap": (
                    contract.P3_ALWAYS_G5_ROUNDS
                ),
            },
            "full_horizon_executed": False,
            "paper_facing_result_allowed": False,
            "cutoff_pool_observations": [
                {
                    "nph": cutoff,
                    "parent_count": 1,
                    "macro_count": 1,
                    "singleton_count": 1,
                    "parent_pool_sha256": hashes[f"parent{cutoff}"],
                    "macro_pool_sha256": hashes[f"macro{cutoff}"],
                    "singleton_pool_sha256": hashes[
                        f"singleton{cutoff}"
                    ],
                    "observation_sha256": hashes[f"poolobs{cutoff}"],
                    "status": "passed",
                }
                for cutoff in (3, 7)
            ],
            "p2_pool_construction_proof": pool_proof,
            "p2_pool_construction_proof_sha256": pool_proof["sha256"],
            "route_observations": route_rows,
            "semantic_coverage": {
                "route_families": list(contract.ROUTE_IDS),
                "candidate_representations": [
                    "macro_generator_v1",
                    "single_pauli_word_v1",
                ],
                "pool_construction_regime_count": 6,
                "cutoff_pool_coverage": [3, 7],
                "ra_fresh_resume_replay_routes": sorted(
                    contract.RA_ROUTES
                ),
                "append_fresh_reconstruction_routes": sorted(
                    contract.APPEND_ROUTES
                ),
                "nonvacuous_g5_routes": sorted(
                    contract.INSERTION_CAPABLE_ROUTES
                ),
            },
            "status": "passed",
            "p3_passed": True,
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_state": "not_submitted",
        }
    )
    return receipt, authority, control


def test_p3_requires_per_route_evidence_not_magic_call_counts() -> None:
    receipt, authority, control = _valid_p3()
    binding = contract.validate_p3_receipt(
        receipt,
        receipt_file_sha256="f" * 64,
        authority=authority,
        control_plane=control,
    )
    assert binding["canonical_sha256"] == receipt["sha256"]
    assert len(receipt["route_observations"]) == 8
    assert all(
        "actual_facade_execution_count" not in row
        for row in receipt["route_observations"]
    )

    tampered = copy.deepcopy(receipt)
    tampered["route_observations"][0]["independent_replay"][
        "matched"
    ] = False
    tampered = contract.digested(tampered)
    with pytest.raises(contract.PackageContractError):
        contract.validate_p3_receipt(
            tampered,
            receipt_file_sha256="f" * 64,
            authority=authority,
            control_plane=control,
        )


def test_p4_and_worker_sources_close_five_roles_and_retry_seam() -> None:
    validator = (PACKAGE_DIR / "validate_package.py").read_text(
        encoding="utf-8"
    )
    worker = (PACKAGE_DIR / "run_cell.py").read_text(encoding="utf-8")
    assert "embedded_complete_canonical_payload_v1" in validator
    assert "allow_partial_p4=p4_exists" in validator
    assert "canonical_payload" in validator
    assert "artifact_roles != list(EXPECTED_ARTIFACT_ROLES)" in validator
    assert validator.count("_validate_p4_smoke_payload(") >= 3
    assert "worker_archive_copy_of_declared_output_v1" in worker
    assert "bounded_smoke_shadow_not_fulfillment_v1" in worker
    assert "paper_i_append_adapt_reconstruction_checkpoint_v1" in worker
    assert "set(artifact_paths) != set(EXPECTED_ARTIFACT_ROLES)" in worker


def _g6_ra_round_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    row: dict[str, Any] = {
        "retained_support": {
            "rank_relative_tolerance": 1.0e-6,
            "metric_regularization": 0.0,
        },
        "phase3_stabilization": {
            "kappa_stabilization_shift": 0.25,
            "trust_boundary_multiplier_lambda": 0.5,
            "total_metric_multiplier_mu": 0.75,
            "metric_whitening_active": False,
            "metric_inverse_sqrt_constructed": False,
            "trust_boundary_active": True,
        },
        "source_gram_no_overlap_trust": {
            "supported_metric_whitening_active": False,
            "supported_metric_inverse_sqrt_constructed": False,
            "endpoint_overlap_query_charge": 0,
        },
    }
    replay: dict[str, Any] = {
        "accepted_refit": {
            "supported_metric": {
                "metric_regularization": 1.0e-9,
            },
        },
        "trust_solve": {
            "policy": "source_metric_inverse_sqrt_no_overlap_v1",
            "update_reason": "model_agreement_radius_hold",
            "endpoint_overlap_query_charge": 0,
            "transaction_complete": True,
            "supported_rank": 1,
            "supported_metric_whitening_active": False,
            "supported_metric_inverse_sqrt_constructed": False,
        },
    }
    return row, replay


_G6_GEOMETRY_LIMITATION = {
    "policy": "source_metric_inverse_sqrt_no_overlap_v1",
    "update_reason": (
        "geometry_expansion_no_coordinate_prediction_no_overlap_hold"
    ),
    "endpoint_overlap_query_charge": 0,
    "transaction_failure": (
        "not_applicable_geometry_expansion_without_coordinate_prediction"
    ),
}


def test_g6_ordinary_round_retains_full_source_gram_validation() -> None:
    row, replay = _g6_ra_round_fixture()
    cell_runner._validate_g6_ra_round(
        job_id="ordinary-test",
        index=1,
        raw=row,
        raw_replay=replay,
    )

    tampered = copy.deepcopy(row)
    tampered["source_gram_no_overlap_trust"][
        "supported_metric_whitening_active"
    ] = True
    with pytest.raises(
        contract.PackageContractError,
        match="G6 Phase-III integrity",
    ):
        cell_runner._validate_g6_ra_round(
            job_id="ordinary-test",
            index=1,
            raw=tampered,
            raw_replay=replay,
        )

    missing = copy.deepcopy(row)
    missing["source_gram_no_overlap_trust"] = None
    with pytest.raises(
        contract.PackageContractError,
        match="G6 source-Gram trust limitation",
    ):
        cell_runner._validate_g6_ra_round(
            job_id="ordinary-test",
            index=1,
            raw=missing,
            raw_replay=replay,
        )


def test_g6_accepts_only_exact_geometry_expansion_limitation() -> None:
    row, replay = _g6_ra_round_fixture()
    row["source_gram_no_overlap_trust"] = None
    replay["trust_solve"] = copy.deepcopy(_G6_GEOMETRY_LIMITATION)
    assert (
        cell_runner.GEOMETRY_EXPANSION_TRUST_SOLVE_LIMITATION
        == _G6_GEOMETRY_LIMITATION
    )
    cell_runner._validate_g6_ra_round(
        job_id="geometry-test",
        index=19,
        raw=row,
        raw_replay=replay,
    )

    missing_key = copy.deepcopy(row)
    missing_key.pop("source_gram_no_overlap_trust")
    with pytest.raises(
        contract.PackageContractError,
        match="G6 source-Gram trust limitation",
    ):
        cell_runner._validate_g6_ra_round(
            job_id="geometry-test",
            index=19,
            raw=missing_key,
            raw_replay=replay,
        )


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("limitation", "transaction_failure", "different-limitation"),
        ("limitation", "unexpected_field", False),
        ("limitation", "endpoint_overlap_query_charge", False),
        ("limitation", "endpoint_overlap_query_charge", 0.0),
        ("support", "rank_relative_tolerance", 1.0e-5),
        ("stabilization", "metric_whitening_active", True),
        ("support", "metric_regularization", 1.0e-12),
    ],
)
def test_g6_geometry_expansion_rejects_tamper_and_keeps_common_checks(
    target: str,
    field: str,
    value: object,
) -> None:
    row, replay = _g6_ra_round_fixture()
    row["source_gram_no_overlap_trust"] = None
    replay["trust_solve"] = copy.deepcopy(_G6_GEOMETRY_LIMITATION)
    if target == "limitation":
        replay["trust_solve"][field] = value
    elif target == "support":
        row["retained_support"][field] = value
    elif target == "stabilization":
        row["phase3_stabilization"][field] = value
    else:
        raise AssertionError(f"unexpected G6 tamper target: {target}")

    with pytest.raises(contract.PackageContractError):
        cell_runner._validate_g6_ra_round(
            job_id="geometry-test",
            index=19,
            raw=row,
            raw_replay=replay,
        )


def _stationary_active_gradient_payload_fixture() -> dict[str, object]:
    first_receipt = {
        "schema": "phase3_active_gradient_query_accounting_v1",
        "active_gradient_policy": "stationary_source_response_v1",
        "active_coordinate_count": 3,
        "active_gradient_indices_acquired": [],
        "new_unique_gradients_charged": 0,
        "deduplicated_or_ledger_disabled_count": 0,
        "primitive_ids": [],
        "component": "N_grad",
        "consumer_scope": "phase3_macro_active_gradient",
        "status": "not_acquired_stationary_source_protocol",
    }
    second_receipt = {
        **first_receipt,
        "active_coordinate_count": 4,
        "consumer_scope": "phase3_singleton_active_gradient",
    }
    return {
        "policy": {
            "active_gradient_policy": "stationary_source_response_v1",
            "resource_weighting_scope": "late_resource_weighting_v1",
            "active_gradient_indices_acquired": [],
            "active_gradient_charge": 0,
        },
        "scientific_receipts": {
            "policy": {
                "active_gradient_policy": (
                    "stationary_source_response_v1"
                ),
                "resource_weighting_scope": (
                    "late_resource_weighting_v1"
                ),
                "active_gradient_indices_acquired": [],
                "active_gradient_charge": 0,
            },
        },
        "run": {
            "accepted_transitions": [
                {
                    "controller_round": 1,
                    "terminal": True,
                    "active_gradient_query_accounting": first_receipt,
                },
                {
                    "controller_round": 2,
                    "terminal": False,
                    "nonterminal_phase3_record": {
                        "active_gradient_query_accounting": second_receipt,
                    },
                },
            ],
            "estimator_call_ledger": {
                "schema": "estimator_call_ledger_v1",
                "occurrences": [
                    {
                        "sequence": 1,
                        "primitive_id": "candidate-gradient-1",
                        "component": "N_grad",
                        "consumer_scope": "phase2_candidate_gradient",
                        "branch_id": None,
                        "charged": True,
                    }
                ],
                "entries": [
                    {
                        "primitive_id": "candidate-gradient-1",
                        "consumers": [
                            {
                                "component": "N_grad",
                                "scope": "phase2_candidate_gradient",
                                "branch_id": None,
                                "occurrence_count": 1,
                            }
                        ],
                    }
                ],
                "summary": {
                    "unique_primitive_count_by_consumer_scope": {
                        "phase2_candidate_gradient": 1,
                    }
                },
                "occurrence_summary": {
                    "occurrence_count_by_consumer_scope": {
                        "phase2_candidate_gradient": 1,
                    }
                },
            },
        },
    }


def test_stationary_active_gradient_gate_collects_nested_nonterminal_receipts() -> None:
    payload = _stationary_active_gradient_payload_fixture()
    closure = cell_runner._validate_stationary_active_gradient_payload(
        payload
    )
    assert closure["status"] == "passed"
    assert (
        closure[
            "phase3_active_gradient_accounting_occurrence_count"
        ]
        == 2
    )
    assert len(
        closure[
            "phase3_active_gradient_accounting_ordered_sha256"
        ]
    ) == 64
    assert closure["estimator_ledger_count_checked"] == 1
    assert closure["active_gradient_estimator_ledger_occurrence_count"] == 0
    assert (
        closure["per_round_phase3_accounting_coverage"]
        == "not_serialized_by_v9_result_contract"
    )
    assert "immutable_v9" in closure["limitation"]
    assert closure == (
        cell_runner._validate_stationary_active_gradient_payload(
            copy.deepcopy(payload)
        )
    )


@pytest.mark.parametrize(
    ("tamper_path", "tampered_value"),
    [
        (
            (
                "run",
                "accepted_transitions",
                1,
                "nonterminal_phase3_record",
                "active_gradient_query_accounting",
                "primitive_ids",
            ),
            ["smuggled-active-gradient-primitive"],
        ),
        (
            (
                "run",
                "accepted_transitions",
                1,
                "nonterminal_phase3_record",
                "active_gradient_query_accounting",
                "new_unique_gradients_charged",
            ),
            1,
        ),
        (
            ("policy", "active_gradient_charge"),
            1,
        ),
    ],
)
def test_stationary_active_gradient_gate_rejects_receipt_or_policy_tamper(
    tamper_path: tuple[str | int, ...],
    tampered_value: object,
) -> None:
    payload = _stationary_active_gradient_payload_fixture()
    target: object = payload
    for token in tamper_path[:-1]:
        target = target[token]  # type: ignore[index]
    target[tamper_path[-1]] = tampered_value  # type: ignore[index]
    with pytest.raises(contract.PackageContractError):
        cell_runner._validate_stationary_active_gradient_payload(payload)


def test_stationary_active_gradient_gate_rejects_ledger_smuggling() -> None:
    payload = _stationary_active_gradient_payload_fixture()
    ledger = payload["run"]["estimator_call_ledger"]  # type: ignore[index]
    ledger["occurrences"][0]["consumer_scope"] = (  # type: ignore[index]
        "phase3_singleton_active_gradient"
    )
    ledger["occurrences"][0]["charged"] = True  # type: ignore[index]
    with pytest.raises(
        contract.PackageContractError,
        match="estimator-ledger consumer occurrence",
    ):
        cell_runner._validate_stationary_active_gradient_payload(payload)


def test_stationary_active_gradient_gate_accepts_honest_v9_ra_limitation() -> None:
    payload = _stationary_active_gradient_payload_fixture()
    payload["run"]["accepted_transitions"] = []  # type: ignore[index]
    payload["run"].pop("estimator_call_ledger")  # type: ignore[index]
    closure = cell_runner._validate_stationary_active_gradient_payload(
        payload,
        append=False,
    )
    assert closure["phase3_active_gradient_accounting_occurrence_count"] == 0
    assert closure["estimator_ledger_count_checked"] == 0
    assert (
        closure["per_round_phase3_accounting_coverage"]
        == "not_serialized_by_v9_result_contract"
    )


def test_stationary_active_gradient_gate_accepts_inert_append_echo() -> None:
    payload = _stationary_active_gradient_payload_fixture()
    payload["run"]["accepted_transitions"] = []  # type: ignore[index]
    payload["run"].pop("estimator_call_ledger")  # type: ignore[index]
    payload["scientific_receipts"].update(  # type: ignore[union-attr]
        {
            "phase3_solver_invoked": False,
            "trust_transaction_invoked": False,
        }
    )
    closure = cell_runner._validate_stationary_active_gradient_payload(
        payload,
        append=True,
    )
    assert closure["phase3_active_gradient_accounting_occurrence_count"] == 0

    payload = _stationary_active_gradient_payload_fixture()
    payload["scientific_receipts"].update(  # type: ignore[union-attr]
        {
            "phase3_solver_invoked": False,
            "trust_transaction_invoked": False,
        }
    )
    with pytest.raises(
        contract.PackageContractError,
        match="serialized a Phase-III",
    ):
        cell_runner._validate_stationary_active_gradient_payload(
            payload,
            append=True,
        )
