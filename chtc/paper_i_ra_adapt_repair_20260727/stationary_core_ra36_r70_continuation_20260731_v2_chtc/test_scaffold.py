from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from typing import Any, Mapping

import pytest


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[2]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from build_activation import (  # noqa: E402
    FINAL_AUTHORIZATIONS_DIR,
    FINAL_CONTROL_PLANE_SCHEMA,
    FINAL_JOBS_DIR,
    FINAL_MANIFEST_SCHEMA,
    FINAL_PACKAGE_ID,
    LIFECYCLE_MODE,
    SUBMIT_NAME,
    FinalizationContext,
    _build_activation_from_context,
    _file_inventory_digest,
    _prepare,
    _resource_bucket,
    assert_row_sharded_descriptor,
    render_submit_descriptor,
)
from scaffold_contract import (  # noqa: E402
    ACTIVATION_AUTHORIZATION_SCHEMA,
    ACTIVATION_INPUT_SCHEMA,
    ACTIVE_GRADIENT_POLICY,
    CELL_COUNT,
    CONTROLLED_CYCLE_VALIDATOR_BINDING,
    CONTROLLED_CYCLE_VALIDATOR_PATH,
    IMAGE_VERIFICATION_SCHEMA,
    INHERITED_RESUME_COUNT,
    PACKAGE_ID,
    PENDING_PREDECESSORS,
    PENDING_RESUME_COUNT,
    PREDECESSOR_BINDING_SCHEMA,
    RESOURCE_EVIDENCE_SCHEMA,
    RESOURCE_OBSERVATION_SCHEMA,
    RESOURCE_WEIGHTING_SCOPE,
    RUNTIME_BUNDLE_MANIFEST_SCHEMA,
    SCAFFOLD_MANIFEST_NAME,
    SCHEDULER_TERMINAL_RECEIPT_SCHEMA,
    SEALED_PARENT_MANIFEST_CANONICAL_SHA256,
    SEALED_PARENT_MANIFEST_FILE_SHA256,
    ScaffoldContractError,
    canonical_json_bytes,
    canonical_sha256,
    digested,
    load_json,
    sha256_file,
    transfer_path_is_regular_file,
    validate_predecessor_binding,
    validate_resume_input_contents,
    verify_controlled_cycle_dependency,
)
from validate_controlled_cycle_archive import (  # noqa: E402
    COMPLETION_RECEIPT_SCHEMA,
    ExpectedAttempt,
    validate_attempt_archive,
)
from validate_scaffold import validate_scaffold  # noqa: E402


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _write_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(value) if "sha256" in value else digested(value)
    path.write_bytes(_json_bytes(payload))
    return payload


def _binding(path: Path, *, canonical: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if canonical:
        result["canonical_sha256"] = json.loads(path.read_text())["sha256"]
    return result


def _tar(path: Path, members: Mapping[str, tuple[bytes, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name, (payload, mode) in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = mode
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))


def _member(path: str, role: str, payload: bytes, source: str) -> dict[str, Any]:
    return {
        "path": path,
        "role": role,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "source_member": source,
    }


@dataclass
class SyntheticCase:
    root: Path
    context: FinalizationContext
    intent_path: Path
    full_path: Path
    output_dir: Path
    output_relative: str
    input_payload: dict[str, Any]
    predecessor_paths: dict[str, Path]
    completion_paths: dict[str, Path]
    compact_paths: dict[str, Path]
    compact_members: dict[str, dict[str, tuple[bytes, int]]]
    authorization_paths: dict[str, Path]
    observation_paths: dict[str, Path]
    resource_evidence_path: Path
    runtime_bundle_path: Path
    image_receipt_path: Path

    def rewrite_input(self) -> None:
        self.input_payload.pop("sha256", None)
        self.input_payload = _write_json(self.full_path, self.input_payload)

    def assert_no_publish(self) -> None:
        assert not self.output_dir.exists()
        assert not list(self.output_dir.parent.glob(f".{self.output_dir.name}.staging.*"))


@pytest.fixture
def repo_tmp() -> Any:
    parent = REPO_ROOT / "chtc" / "paper_i_ra_adapt_repair_20260727"
    with tempfile.TemporaryDirectory(prefix=".ra-r70-v3-test-", dir=parent) as name:
        yield Path(name)


def _controlled_binding(
    *,
    case_root: Path,
    execution_id: str,
    source_execution_id: str,
    proc_id: int,
    requirement: Mapping[str, Any],
    source_sha256: str,
    image_sha256: str,
) -> tuple[Path, Path, Path, dict[str, tuple[bytes, int]]]:
    authority_root = case_root / "controlled" / execution_id
    job_path = authority_root / "job.json"
    authorization_path = authority_root / "execution_authorization.json"
    activation_path = authority_root / "activation_manifest.json"
    route_sha = requirement["scientific_anchor"]["route_contract_sha256"]
    job = _write_json(
        job_path,
        {
            "schema": "synthetic_controlled_predecessor_job_v1",
            "execution_id": source_execution_id,
            "horizon": 50,
            "active_gradient_policy": ACTIVE_GRADIENT_POLICY,
            "resource_weighting_scope": RESOURCE_WEIGHTING_SCOPE,
            "phase1_cost_term": "disabled_for_phase1_only",
            "protocol": {
                "sha256": requirement["scientific_anchor"]["source_protocol_sha256"],
                "canonical_sha256": requirement["scientific_anchor"][
                    "source_protocol_canonical_sha256"
                ],
                "route_contract_sha256": route_sha,
            },
            "remote_image": {
                "path": "chtc/phase3_optuna/image.sif",
                "sha256": image_sha256,
            },
        },
    )
    control_sha = hashlib.sha256(f"control-{proc_id}".encode()).hexdigest()
    authorization = _write_json(
        authorization_path,
        {
            "schema": "synthetic_controlled_execution_authorization_v1",
            "execution_id": source_execution_id,
            "job_file_sha256": sha256_file(job_path),
            "job_sha256": job["sha256"],
            "source_archive_sha256": source_sha256,
            "remote_image_sha256": image_sha256,
            "activation_id": f"synthetic-controlled-{proc_id}",
            "activation_control_plane_sha256": control_sha,
            "execution_authorized": True,
            "submission_authorized": True,
        },
    )
    activation = _write_json(
        activation_path,
        {
            "schema": "synthetic_controlled_activation_v1",
            "activation_id": f"synthetic-controlled-{proc_id}",
            "activation_control_plane_sha256": control_sha,
            "source_archive_sha256": source_sha256,
            "remote_image": {"sha256": image_sha256},
            "executions": [
                {
                    "execution_id": source_execution_id,
                    "job": _binding(job_path, canonical=True),
                    "authorization": _binding(authorization_path, canonical=True),
                }
            ],
        },
    )

    ledger_source = f"runs/{source_execution_id}/checkpoints/ledger.json"
    sidecar_source = f"runs/{source_execution_id}/checkpoints/resume.json"
    checkpoint_source = f"runs/{source_execution_id}/checkpoints/current.json"
    ledger_bytes = b'{"ledger":"complete"}\n'
    sidecar_bytes = b'{"resume":"verified"}\n'
    checkpoint = {
        "checkpoint": {"depth": 50},
        "adapt_vqe": {
            "history_count": 50,
            "history_checkpoint_complete": True,
            "strict_replay": {"passed": True},
            "sr_route_profile_contract_sha256": route_sha,
            "active_prefix_checkpoints": [{} for _ in range(50)],
            "estimator_call_ledger_checkpoint": {
                "path": ledger_source,
                "sha256": hashlib.sha256(ledger_bytes).hexdigest(),
                "status": "complete",
            },
            "verified_singleton_resume_sidecar": {
                "path": sidecar_source,
                "sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
                "status": "complete",
                "enabled": True,
            },
        },
    }
    checkpoint_bytes = canonical_json_bytes(checkpoint) + b"\n"
    worker: dict[str, bytes] = {
        "attempt_identity.tsv": (
            f"{source_execution_id}\t9397758\t{proc_id}\t1\n"
        ).encode(),
        "worker_exit_status.txt": b"0\n",
        "result.json": b'{"status":"passed"}\n',
        checkpoint_source: checkpoint_bytes,
        ledger_source: ledger_bytes,
        sidecar_source: sidecar_bytes,
    }
    attempt_receipt = digested(
        {
            "schema": "paper_i_ra_always_factorial_worker_attempt_v1",
            "execution_id": source_execution_id,
            "cluster_id": 9397758,
            "proc_id": proc_id,
            "attempt_ordinal": 1,
            "worker_exit_status": 0,
            "job_file_sha256": sha256_file(job_path),
            "authorization_file_sha256": sha256_file(authorization_path),
            "activation_manifest_file_sha256": sha256_file(activation_path),
            "source_archive_sha256": source_sha256,
            "image_sha256": image_sha256,
            "worker_files": [
                {
                    "path": name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
                for name, payload in worker.items()
            ],
        }
    )
    attempt_path = authority_root / "attempt.tar.gz"
    attempt_members: dict[str, tuple[bytes, int]] = {
        "authority/job.json": (job_path.read_bytes(), 0o644),
        "authority/execution_authorization.json": (
            authorization_path.read_bytes(),
            0o644,
        ),
        "authority/activation_manifest.json": (activation_path.read_bytes(), 0o644),
        "worker_attempt_receipt.json": (_json_bytes(attempt_receipt), 0o644),
        **{f"worker_outputs/{name}": (payload, 0o644) for name, payload in worker.items()},
    }
    _tar(attempt_path, attempt_members)
    expected = ExpectedAttempt(
        execution_id=source_execution_id,
        cluster_id=9397758,
        proc_id=proc_id,
        job_path=job_path,
        authorization_path=authorization_path,
        activation_manifest_path=activation_path,
        source_archive_sha256=source_sha256,
        image_sha256=image_sha256,
    )
    validation = validate_attempt_archive(attempt_path, expected)
    completion_path = authority_root / "retrieval_completion_receipt.json"
    completion = _write_json(
        completion_path,
        {
            "schema": COMPLETION_RECEIPT_SCHEMA,
            "status": "passed",
            "completion_classification": "worker_exit_zero_archive_fully_authenticated",
            "execution": {
                "execution_id": source_execution_id,
                "cluster_id": 9397758,
                "proc_id": proc_id,
                "attempt_ordinal": 1,
            },
            "retrieval": {
                "remote_archive_sha256": validation["archive"]["sha256"],
                "remote_archive_size_bytes": validation["archive"]["size_bytes"],
                "local_archive": validation["archive"],
                "remote_local_hash_size_match": True,
            },
            "release": {
                "target": f"9397758.{proc_id}",
                "scope": "exact_cluster_proc_only",
                "exit_code": 0,
            },
            "bindings": validation["bindings"],
            "archive_validation": {
                "worker_attempt_receipt": validation["worker_attempt_receipt"],
                **validation["member_validation"],
            },
        },
    )
    scheduler_path = authority_root / "scheduler_terminal_receipt.json"
    _write_json(
        scheduler_path,
        {
            "schema": SCHEDULER_TERMINAL_RECEIPT_SCHEMA,
            "status": "passed",
            "execution_id": source_execution_id,
            "cluster_id": 9397758,
            "proc_id": proc_id,
            "job_status": 4,
            "exit_code": 0,
            "num_job_starts": 1,
            "completion_epoch": 1785513600 + proc_id,
            "source": "condor_history_exact_cluster_proc",
        },
    )

    compact_path = REPO_ROOT / requirement["resume_archive_path"]
    compact_members = {
        "checkpoint/01.checkpoint.json": (checkpoint_bytes, 0o644),
        "checkpoint/ledger.json": (ledger_bytes, 0o644),
        "checkpoint/resume.json": (sidecar_bytes, 0o644),
    }
    _tar(compact_path, compact_members)
    resume_members = [
        _member(
            "checkpoint/01.checkpoint.json",
            "checkpoint",
            checkpoint_bytes,
            f"worker_outputs/{checkpoint_source}",
        ),
        _member(
            "checkpoint/ledger.json",
            "estimator_ledger_checkpoint",
            ledger_bytes,
            f"worker_outputs/{ledger_source}",
        ),
        _member(
            "checkpoint/resume.json",
            "verified_resume_sidecar",
            sidecar_bytes,
            f"worker_outputs/{sidecar_source}",
        ),
    ]
    predecessor_path = REPO_ROOT / requirement["binding_path"]
    _write_json(
        predecessor_path,
        {
            "schema": PREDECESSOR_BINDING_SCHEMA,
            "package_id": PACKAGE_ID,
            "execution_id": execution_id,
            "source_horizon": 50,
            "target_horizon": 70,
            "status": "passed",
            "scientific_anchor": requirement["scientific_anchor"],
            "scheduler_terminal_receipt": _binding(scheduler_path, canonical=True),
            "retrieval_completion_receipt": _binding(completion_path, canonical=True),
            "attempt_archive": _binding(attempt_path),
            "resume_input": {
                "archive": _binding(compact_path),
                "checkpoint_path": "checkpoint/01.checkpoint.json",
                "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
                "member_count": 3,
                "members": resume_members,
                "pointer_closed": True,
                "superseded_sidecars_retained": False,
            },
        },
    )
    return predecessor_path, completion_path, compact_path, compact_members


def _build_case(case_root: Path) -> SyntheticCase:
    inert = case_root / "inert_v2"
    evidence = case_root / "evidence"
    inert.mkdir(parents=True)
    evidence.mkdir(parents=True)
    source_path = evidence / "source.tar.gz"
    source_path.write_bytes(b"synthetic-source-archive\n")
    source_binding = _binding(source_path)
    source_manifest_path = evidence / "source_manifest.json"
    _write_json(source_manifest_path, {"schema": "synthetic_source_manifest_v1"})
    source_delta_path = evidence / "source_delta.json"
    _write_json(source_delta_path, {"schema": "synthetic_source_delta_v1"})
    source_manifest_binding = _binding(source_manifest_path, canonical=True)
    source_delta_binding = _binding(source_delta_path, canonical=True)

    image_binding = {
        "host": "ap2001.chtc.wisc.edu",
        "remote_root": "/home/jsstrobel/Holstein_phase3_optuna_chtc",
        "path": "chtc/phase3_optuna/image.sif",
        "sha256": hashlib.sha256(b"synthetic-sif-image\n").hexdigest(),
        "size_bytes": len(b"synthetic-sif-image\n"),
    }
    image_receipt_path = evidence / "image_verification.json"
    _write_json(
        image_receipt_path,
        {
            "schema": IMAGE_VERIFICATION_SCHEMA,
            "package_id": FINAL_PACKAGE_ID,
            "status": "passed",
            "remote_image": {
                **image_binding,
                "verified_utc": "2026-07-31T18:00:00Z",
            },
            "verification": {
                "remote_regular_file": True,
                "remote_sha256_verified": True,
                "remote_size_verified": True,
            },
        },
    )

    bootstrap_path = evidence / "run_row.sh"
    bootstrap_bytes = b"#!/usr/bin/env bash\nexit 0\n"
    bootstrap_path.write_bytes(bootstrap_bytes)
    bootstrap_path.chmod(0o755)
    runtime_bundle_path = evidence / "runtime.tar.gz"
    _tar(runtime_bundle_path, {"bin/run_row.sh": (bootstrap_bytes, 0o755)})
    runtime_manifest_path = evidence / "runtime_manifest.json"
    _write_json(
        runtime_manifest_path,
        {
            "schema": RUNTIME_BUNDLE_MANIFEST_SCHEMA,
            "package_id": FINAL_PACKAGE_ID,
            "status": "passed",
            "archive": _binding(runtime_bundle_path),
            "entrypoint_member": "bin/run_row.sh",
            "member_count": 1,
            "members": [
                {
                    "path": "bin/run_row.sh",
                    "role": "row_bootstrap",
                    "sha256": hashlib.sha256(bootstrap_bytes).hexdigest(),
                    "size_bytes": len(bootstrap_bytes),
                    "mode": 0o755,
                }
            ],
        },
    )

    shapes = [
        ("macro_generator_v1", 3),
        ("macro_generator_v1", 3),
        ("macro_generator_v1", 3),
        ("macro_generator_v1", 7),
        ("macro_generator_v1", 7),
        ("single_pauli_word_v1", 3),
        ("single_pauli_word_v1", 3),
        ("single_pauli_word_v1", 7),
        ("single_pauli_word_v1", 7),
    ]
    jobs: dict[str, Mapping[str, Any]] = {}
    job_bindings: dict[str, Mapping[str, Any]] = {}
    requirements: dict[str, Mapping[str, Any]] = {}
    transfer_rows: list[dict[str, Any]] = []
    for proc_id, (representation, nph) in enumerate(shapes):
        execution_id = f"synthetic__cell_{proc_id}__r70"
        source_execution_id = f"synthetic__source_{proc_id}"
        route_sha = hashlib.sha256(f"route-{proc_id}".encode()).hexdigest()
        scientific = {"schema": "synthetic_scientific_v1", "execution_id": execution_id}
        scientific_sha = canonical_sha256(scientific)
        requirement = digested(
            {
                "schema": "synthetic_predecessor_requirement_v1",
                "package_id": PACKAGE_ID,
                "execution_id": execution_id,
                "source_horizon": 50,
                "target_horizon": 70,
                "predecessor": {
                    "cluster_id": 9397758,
                    "proc_id": proc_id,
                    "source_execution_id": source_execution_id,
                },
                "binding_path": (
                    evidence / "predecessor_bindings" / f"{execution_id}.json"
                ).relative_to(REPO_ROOT).as_posix(),
                "resume_archive_path": (
                    evidence / "resume_inputs" / f"{execution_id}.tar.gz"
                ).relative_to(REPO_ROOT).as_posix(),
                "scientific_anchor": {
                    "route_contract_sha256": route_sha,
                    "scientific_settings_sha256": scientific_sha,
                    "source_protocol_sha256": hashlib.sha256(
                        f"protocol-file-{proc_id}".encode()
                    ).hexdigest(),
                    "source_protocol_canonical_sha256": hashlib.sha256(
                        f"protocol-canonical-{proc_id}".encode()
                    ).hexdigest(),
                },
                "status": "missing_fail_closed",
            }
        )
        requirements[execution_id] = requirement
        job = digested(
            {
                "schema": "synthetic_inert_r70_job_v2",
                "package_id": PACKAGE_ID,
                "campaign_id": "synthetic_v2",
                "execution_id": execution_id,
                "candidate_representation": representation,
                "nph": nph,
                "scientific_settings": scientific,
                "scientific_settings_sha256": scientific_sha,
                "source_protocol": {"route_contract_sha256": route_sha},
                "effective_sources": {
                    "source_archive": source_binding,
                    "source_manifest": source_manifest_binding,
                    "source_delta_receipt": source_delta_binding,
                },
                "resources": {
                    "request_cpus": 1,
                    "request_memory_mb": 64,
                    "request_disk_mb": 64,
                    "max_runtime_seconds": 3600,
                    "source": "synthetic_v2",
                    "r70_demonstration_status": "not_demonstrated",
                },
                "resume_input": None,
                "resume_source": None,
                "predecessor_binding_sha256": None,
                "execution_authorized": False,
                "submission_authorized": False,
                "submission_ready": False,
                "submitted": False,
            }
        )
        job_path = inert / "jobs" / f"{execution_id}.json"
        _write_json(job_path, job)
        jobs[execution_id] = job
        job_bindings[execution_id] = _binding(job_path, canonical=True)
        transfer_rows.append(
            {
                "proc_id": proc_id,
                "execution_id": execution_id,
                "transfer_inputs": [
                    {"role": "source_archive", **source_binding, "status": "ready"},
                    {"role": "source_manifest", **source_manifest_binding, "status": "ready"},
                    {"role": "source_delta_receipt", **source_delta_binding, "status": "ready"},
                ],
            }
        )

    scaffold_manifest_path = inert / "scaffold_manifest.json"
    scaffold_manifest = _write_json(
        scaffold_manifest_path,
        {
            "schema": "synthetic_inert_scaffold_v2",
            "package_id": PACKAGE_ID,
            "cell_count": len(jobs),
            "status": "passed_inert_scaffold_missing_9_predecessors",
        },
    )
    context = FinalizationContext(
        repo_root=REPO_ROOT,
        package_dir=inert,
        package_relative_root=inert.relative_to(REPO_ROOT).as_posix(),
        scaffold_manifest=scaffold_manifest,
        scaffold_manifest_binding=_binding(scaffold_manifest_path, canonical=True),
        jobs=jobs,
        job_bindings=job_bindings,
        transfer=digested({"rows": transfer_rows}),
        requirements=requirements,
        expected_pending_ids=frozenset(jobs),
        expected_cell_count=len(jobs),
        inert_inventory_sha256=_file_inventory_digest(inert),
    )

    predecessor_paths: dict[str, Path] = {}
    completion_paths: dict[str, Path] = {}
    compact_paths: dict[str, Path] = {}
    compact_members: dict[str, dict[str, tuple[bytes, int]]] = {}
    for proc_id, execution_id in enumerate(jobs):
        predecessor, completion, compact, members = _controlled_binding(
            case_root=case_root,
            execution_id=execution_id,
            source_execution_id=requirements[execution_id]["predecessor"][
                "source_execution_id"
            ],
            proc_id=proc_id,
            requirement=requirements[execution_id],
            source_sha256=source_binding["sha256"],
            image_sha256=image_binding["sha256"],
        )
        predecessor_paths[execution_id] = predecessor
        completion_paths[execution_id] = completion
        compact_paths[execution_id] = compact
        compact_members[execution_id] = members

    observation_paths: dict[str, Path] = {}
    observation_bindings: dict[str, Mapping[str, Any]] = {}
    for bucket in sorted({_resource_bucket(job) for job in jobs.values()}):
        target_execution_id = next(
            execution_id
            for execution_id, job in jobs.items()
            if _resource_bucket(job) == bucket
        )
        representation, nph_label = bucket.split(":", 1)
        nph = int(nph_label.removeprefix("nph"))
        source_execution_id = f"synthetic__resource_source__{bucket.replace(':', '__')}"
        source_protocol_sha = hashlib.sha256(
            f"resource-protocol-{bucket}".encode()
        ).hexdigest()
        source_job_path = evidence / "resource_source_jobs" / f"{bucket.replace(':', '__')}.json"
        _write_json(
            source_job_path,
            {
                "schema": "synthetic_r50_resource_source_job_v1",
                "execution_id": source_execution_id,
                "horizon": 50,
                "candidate_representation": representation,
                "nph": nph,
                "protocol": {"sha256": source_protocol_sha},
                "resources": {
                    "request_cpus": 1,
                    "request_memory_mb": 128,
                    "request_disk_mb": 128,
                    "max_runtime_seconds": 7200,
                },
            },
        )
        terminal_path = evidence / "resource_terminals" / f"{bucket.replace(':', '__')}.json"
        _write_json(
            terminal_path,
            {
                "schema": SCHEDULER_TERMINAL_RECEIPT_SCHEMA,
                "status": "passed",
                "execution_id": source_execution_id,
                "cluster_id": 9000000 + len(observation_paths),
                "proc_id": 0,
                "job_status": 4,
                "exit_code": 0,
                "num_job_starts": 1,
                "completion_epoch": 1785517200 + len(observation_paths),
                "source": "condor_history_exact_cluster_proc",
            },
        )
        path = evidence / "resource_observations" / f"{bucket.replace(':', '__')}.json"
        _write_json(
            path,
            {
                "schema": RESOURCE_OBSERVATION_SCHEMA,
                "package_id": FINAL_PACKAGE_ID,
                "bucket_id": bucket,
                "status": "passed",
                "approval_policy": "r50_history_plus_conservative_r70_headroom_v1",
                "horizon": 50,
                "evidence_role": "r50_history_envelope_basis",
                "target_execution_id": target_execution_id,
                "target_scientific_settings_sha256": jobs[target_execution_id][
                    "scientific_settings_sha256"
                ],
                "source_execution_id": source_execution_id,
                "source_job": _binding(source_job_path, canonical=True),
                "source_protocol_sha256": source_protocol_sha,
                "scheduler_terminal_receipt": _binding(
                    terminal_path, canonical=True
                ),
                "requested": {
                    "request_cpus": 1,
                    "request_memory_mb": 128,
                    "request_disk_mb": 128,
                    "max_runtime_seconds": 7200,
                },
                "observed": {
                    "peak_memory_mb": 32,
                    "peak_disk_mb": 16,
                    "wall_seconds": 60,
                    "output_archive_bytes": 1024,
                },
                "approved_envelope": {
                    "request_cpus": 1,
                    "request_memory_mb": 128,
                    "request_disk_mb": 128,
                    "max_runtime_seconds": 7200,
                },
            },
        )
        observation_paths[bucket] = path
        observation_bindings[bucket] = _binding(path, canonical=True)
    resource_evidence_path = evidence / "resource_evidence.json"
    _write_json(
        resource_evidence_path,
        {
            "schema": RESOURCE_EVIDENCE_SCHEMA,
            "package_id": FINAL_PACKAGE_ID,
            "status": "passed_for_held_submission_r70_pilot_pending",
            "policy": "r50_history_plus_conservative_r70_headroom_v1",
            "observations": observation_bindings,
            "r70_worst_bucket_pilot": {
                "bucket_id": "single_pauli_word_v1:nph7",
                "execution_id": next(
                    execution_id
                    for execution_id, job in jobs.items()
                    if _resource_bucket(job) == "single_pauli_word_v1:nph7"
                ),
                "status": "planned_not_executed",
                "initial_state": "held",
                "broad_release_authorized": False,
                "pilot_release_requires_separate_exact_proc_authorization": True,
            },
        },
    )

    base_inputs = {
        "schema": ACTIVATION_INPUT_SCHEMA,
        "inert_package_id": PACKAGE_ID,
        "final_package_id": FINAL_PACKAGE_ID,
        "status": "evidence_complete_authorizations_pending",
        "predecessor_bindings": {
            execution_id: _binding(path, canonical=True)
            for execution_id, path in predecessor_paths.items()
        },
        "runtime_bundle": _binding(runtime_bundle_path),
        "runtime_bundle_manifest": _binding(runtime_manifest_path, canonical=True),
        "row_bootstrap": _binding(bootstrap_path),
        "image": image_binding,
        "image_verification": _binding(image_receipt_path, canonical=True),
        "resource_evidence": _binding(resource_evidence_path, canonical=True),
        "authorizations": {},
        "execution_authorized": False,
        "submission_authorized": False,
        "submitted": False,
    }
    intent_path = evidence / "activation_inputs_intent.json"
    _write_json(intent_path, base_inputs)
    output_dir = case_root / "v3"
    output_relative = output_dir.relative_to(REPO_ROOT).as_posix()
    prepared = _prepare(
        context=context,
        activation_inputs_path=intent_path,
        output_relative_root=output_relative,
        require_authorizations=False,
    )
    assert prepared["control_plane"]["schema"] == FINAL_CONTROL_PLANE_SCHEMA

    authorization_paths: dict[str, Path] = {}
    authorization_bindings: dict[str, Mapping[str, Any]] = {}
    lifecycle = {
        "mode": LIFECYCLE_MODE,
        "initial_state": "held",
        "automatic_release": False,
        "release_scope": "exact_cluster_proc_only",
    }
    for execution_id, job in prepared["final_jobs"].items():
        bucket = _resource_bucket(job)
        receipt = {
            "schema": ACTIVATION_AUTHORIZATION_SCHEMA,
            "status": "passed",
            "authorization_id": f"{FINAL_PACKAGE_ID}::{execution_id}",
            "authorized_utc": "2026-07-31T18:30:00Z",
            "package_id": FINAL_PACKAGE_ID,
            "inert_package_id": PACKAGE_ID,
            "execution_id": execution_id,
            "job": prepared["job_bindings"][execution_id],
            "scientific_settings_sha256": job["scientific_settings_sha256"],
            "resume_archive": prepared["resumes"][execution_id]["archive"],
            "source_archive_sha256": prepared["sources"][execution_id][
                "source_archive"
            ]["sha256"],
            "runtime_bundle": prepared["runtime"]["bundle"],
            "runtime_bundle_manifest": prepared["runtime"]["manifest"],
            "row_bootstrap": prepared["runtime"]["bootstrap"],
            "image": prepared["image"]["image"],
            "image_verification": prepared["image"]["verification"],
            "resource_evidence": prepared["resources"]["binding"],
            "resource_observation": prepared["resources"]["buckets"][bucket][
                "binding"
            ],
            "activation_control_plane_sha256": prepared["control_plane"]["sha256"],
            "lifecycle": lifecycle,
            "execution_authorized": True,
            "submission_authorized": True,
            "release_authorized": False,
            "submission_state": "authorized_not_submitted",
            "remote_stage": False,
            "condor_submit": False,
            "submitted": False,
        }
        path = evidence / "authorizations" / f"{execution_id}.json"
        _write_json(path, receipt)
        authorization_paths[execution_id] = path
        authorization_bindings[execution_id] = _binding(path, canonical=True)

    full_payload = dict(base_inputs)
    full_payload.update(
        {
            "status": "evidence_and_authorizations_complete",
            "authorizations": authorization_bindings,
            "execution_authorized": True,
            "submission_authorized": True,
        }
    )
    full_path = evidence / "activation_inputs_full.json"
    full_payload = _write_json(full_path, full_payload)
    return SyntheticCase(
        root=case_root,
        context=context,
        intent_path=intent_path,
        full_path=full_path,
        output_dir=output_dir,
        output_relative=output_relative,
        input_payload=full_payload,
        predecessor_paths=predecessor_paths,
        completion_paths=completion_paths,
        compact_paths=compact_paths,
        compact_members=compact_members,
        authorization_paths=authorization_paths,
        observation_paths=observation_paths,
        resource_evidence_path=resource_evidence_path,
        runtime_bundle_path=runtime_bundle_path,
        image_receipt_path=image_receipt_path,
    )


def test_checked_in_scaffold_is_exact_27_plus_9_inert_state() -> None:
    result = validate_scaffold(rehash_existing_resumes=False)
    assert result["status"] == "passed_inert_scaffold_missing_9_predecessors"
    assert result["ready_authenticated_resume_count"] == INHERITED_RESUME_COUNT
    assert result["missing_authenticated_resume_count"] == PENDING_RESUME_COUNT
    assert result["scientific_settings_exact_parent_count"] == CELL_COUNT
    assert result["sealed_parent_manifest_file_sha256"] == (
        SEALED_PARENT_MANIFEST_FILE_SHA256
    )
    assert result["sealed_parent_manifest_canonical_sha256"] == (
        SEALED_PARENT_MANIFEST_CANONICAL_SHA256
    )
    assert result["submission_ready"] is False
    manifest = load_json(PACKAGE_DIR / SCAFFOLD_MANIFEST_NAME, label="manifest")
    assert manifest["external_control_dependencies"] == [
        CONTROLLED_CYCLE_VALIDATOR_BINDING
    ]


def test_controlled_cycle_validator_dependency_byte_drift_is_rejected(
    repo_tmp: Path,
) -> None:
    candidate = repo_tmp / "validate_controlled_cycle_archive.py"
    candidate.write_bytes(CONTROLLED_CYCLE_VALIDATOR_PATH.read_bytes())
    assert verify_controlled_cycle_dependency(candidate) == (
        CONTROLLED_CYCLE_VALIDATOR_BINDING
    )
    payload = bytearray(candidate.read_bytes())
    payload[-1] ^= 1
    candidate.write_bytes(payload)
    with pytest.raises(ScaffoldContractError, match="dependency bytes drifted"):
        verify_controlled_cycle_dependency(candidate)


def test_nine_placeholders_bind_external_evidence_not_mutable_v2() -> None:
    placeholders = {
        path.stem: json.loads(path.read_text())
        for path in (PACKAGE_DIR / "predecessor_placeholders").glob("*.json")
    }
    assert set(placeholders) == set(PENDING_PREDECESSORS)
    for row in placeholders.values():
        assert row["binding_path"].startswith(
            "chtc/paper_i_ra_adapt_repair_20260727/"
            "stationary_core_ra36_r70_continuation_20260731_input_evidence_v1/"
        )
        assert row["resume_archive_path"].startswith(
            "chtc/paper_i_ra_adapt_repair_20260727/"
            "stationary_core_ra36_r70_continuation_20260731_input_evidence_v1/"
        )
        assert PACKAGE_ID not in row["binding_path"]
    assert not (PACKAGE_DIR / "resume_inputs_new").exists()
    assert not (PACKAGE_DIR / "predecessor_bindings").exists()


def test_one_real_inherited_resume_gets_full_member_and_pointer_validation() -> None:
    path = PACKAGE_DIR / "jobs/core__weak_weak__nph3__ra_macro_append_only__r70.json"
    job = json.loads(path.read_text())
    result = validate_resume_input_contents(
        resume=job["resume_input"],
        repo_root=REPO_ROOT,
        expected_route_contract_sha256=job["source_protocol"][
            "route_contract_sha256"
        ],
        expected_depth=50,
    )
    assert result["metadata"]["active_prefix_checkpoint_count"] == 50


def test_submit_renderer_is_row_sharded_ordinary_held_and_not_factory() -> None:
    descriptor = render_submit_descriptor(
        queue_path="chtc/example/v3/queue.tsv",
        batch_name="paper-i-ra-r70-test",
        runtime_root="chtc/example/v3_runtime",
    )
    assert_row_sharded_descriptor(descriptor)
    assert "hold = True" in descriptor
    assert "periodic_release = False" in descriptor
    assert "max_materialize" not in descriptor.lower()
    assert "resume_inputs/" not in descriptor.split("transfer_input_files", 1)[1].splitlines()[0]


@pytest.mark.parametrize(
    "path,expected",
    [
        ("resume_inputs", False),
        ("resume_inputs_new", False),
        ("resume_inputs/cell.tar.gz", True),
        ("jobs/cell.json", True),
    ],
)
def test_transfer_path_requires_an_exact_file(path: str, expected: bool) -> None:
    assert transfer_path_is_regular_file(path) is expected


def test_nine_valid_external_bindings_reach_atomic_v3_held_descriptor(
    repo_tmp: Path,
) -> None:
    case = _build_case(repo_tmp)
    before = _file_inventory_digest(case.context.package_dir)
    result = _build_activation_from_context(
        activation_inputs_path=case.full_path,
        context=case.context,
        output_dir=case.output_dir,
        output_relative_root=case.output_relative,
    )
    assert result["status"] == "materialized_atomic_v3_ordinary_held_not_submitted"
    assert result["new_predecessor_binding_count"] == 9
    assert result["row_count"] == 9
    assert result["submitted"] is False
    assert _file_inventory_digest(case.context.package_dir) == before
    manifest = json.loads((case.output_dir / "activation_manifest.json").read_text())
    assert manifest["schema"] == FINAL_MANIFEST_SCHEMA
    assert manifest["status"] == "passed_atomic_v3_ordinary_held_not_submitted"
    assert manifest["initial_state"] == "all_rows_held"
    assert manifest["aggregate_resume_directory_transferred"] is False
    assert manifest["resource_release_gate"]["broad_release_authorized"] is False
    assert manifest["resource_release_gate"]["r70_worst_bucket_pilot"][
        "status"
    ] == "planned_not_executed"
    assert len(list((case.output_dir / FINAL_JOBS_DIR).glob("*.json"))) == 9
    assert len(list((case.output_dir / FINAL_AUTHORIZATIONS_DIR).glob("*.json"))) == 9
    descriptor = (case.output_dir / SUBMIT_NAME).read_text()
    assert_row_sharded_descriptor(descriptor)
    assert len((case.output_dir / "queue.tsv").read_text().splitlines()) == 9
    final_job = json.loads(next((case.output_dir / FINAL_JOBS_DIR).glob("*.json")).read_text())
    assert final_job["resources"]["r70_demonstration_status"] == (
        "worst_bucket_pilot_pending"
    )
    assert final_job["resources"]["broad_release_authorized"] is False


def test_actual_runtime_hash_drift_leaves_v3_absent(
    repo_tmp: Path,
) -> None:
    case = _build_case(repo_tmp)
    path = case.runtime_bundle_path
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ScaffoldContractError, match="bytes drifted"):
        _build_activation_from_context(
            activation_inputs_path=case.full_path,
            context=case.context,
            output_dir=case.output_dir,
            output_relative_root=case.output_relative,
        )
    case.assert_no_publish()


def test_remote_image_receipt_hash_relation_drift_leaves_v3_absent(
    repo_tmp: Path,
) -> None:
    case = _build_case(repo_tmp)
    receipt = json.loads(case.image_receipt_path.read_text())
    receipt.pop("sha256")
    receipt["remote_image"]["sha256"] = "0" * 64
    _write_json(case.image_receipt_path, receipt)
    case.input_payload["image_verification"] = _binding(
        case.image_receipt_path, canonical=True
    )
    case.rewrite_input()
    with pytest.raises(ScaffoldContractError, match="Image verification relation"):
        _build_activation_from_context(
            activation_inputs_path=case.full_path,
            context=case.context,
            output_dir=case.output_dir,
            output_relative_root=case.output_relative,
        )
    case.assert_no_publish()


def test_completion_receipt_relation_drift_leaves_v3_absent(repo_tmp: Path) -> None:
    case = _build_case(repo_tmp)
    execution_id = next(iter(case.completion_paths))
    completion_path = case.completion_paths[execution_id]
    completion = json.loads(completion_path.read_text())
    completion.pop("sha256")
    completion["release"]["target"] = "9397758.999"
    _write_json(completion_path, completion)
    predecessor_path = case.predecessor_paths[execution_id]
    predecessor = json.loads(predecessor_path.read_text())
    predecessor.pop("sha256")
    predecessor["retrieval_completion_receipt"] = _binding(
        completion_path, canonical=True
    )
    _write_json(predecessor_path, predecessor)
    case.input_payload["predecessor_bindings"][execution_id] = _binding(
        predecessor_path, canonical=True
    )
    case.rewrite_input()
    with pytest.raises(ScaffoldContractError, match="completion receipt relation"):
        _build_activation_from_context(
            activation_inputs_path=case.full_path,
            context=case.context,
            output_dir=case.output_dir,
            output_relative_root=case.output_relative,
        )
    case.assert_no_publish()


def test_compact_resume_missing_member_leaves_v3_absent(repo_tmp: Path) -> None:
    case = _build_case(repo_tmp)
    execution_id = next(iter(case.compact_paths))
    compact_path = case.compact_paths[execution_id]
    members = dict(case.compact_members[execution_id])
    members.pop("checkpoint/resume.json")
    _tar(compact_path, members)
    predecessor_path = case.predecessor_paths[execution_id]
    predecessor = json.loads(predecessor_path.read_text())
    predecessor.pop("sha256")
    predecessor["resume_input"]["archive"] = _binding(compact_path)
    _write_json(predecessor_path, predecessor)
    case.input_payload["predecessor_bindings"][execution_id] = _binding(
        predecessor_path, canonical=True
    )
    case.rewrite_input()
    with pytest.raises(ScaffoldContractError, match="member closure is incomplete"):
        _build_activation_from_context(
            activation_inputs_path=case.full_path,
            context=case.context,
            output_dir=case.output_dir,
            output_relative_root=case.output_relative,
        )
    case.assert_no_publish()


def test_resource_terminal_failure_leaves_v3_absent(repo_tmp: Path) -> None:
    case = _build_case(repo_tmp)
    bucket, observation_path = next(iter(case.observation_paths.items()))
    observation = json.loads(observation_path.read_text())
    observation.pop("sha256")
    terminal_path = REPO_ROOT / observation["scheduler_terminal_receipt"]["path"]
    terminal = json.loads(terminal_path.read_text())
    terminal.pop("sha256")
    terminal["exit_code"] = 7
    _write_json(terminal_path, terminal)
    observation["scheduler_terminal_receipt"] = _binding(
        terminal_path, canonical=True
    )
    _write_json(observation_path, observation)
    evidence = json.loads(case.resource_evidence_path.read_text())
    evidence.pop("sha256")
    evidence["observations"][bucket] = _binding(observation_path, canonical=True)
    _write_json(case.resource_evidence_path, evidence)
    case.input_payload["resource_evidence"] = _binding(
        case.resource_evidence_path, canonical=True
    )
    case.rewrite_input()
    with pytest.raises(ScaffoldContractError, match="resource observation failed"):
        _build_activation_from_context(
            activation_inputs_path=case.full_path,
            context=case.context,
            output_dir=case.output_dir,
            output_relative_root=case.output_relative,
        )
    case.assert_no_publish()


def test_authorization_job_relation_drift_leaves_v3_absent(repo_tmp: Path) -> None:
    case = _build_case(repo_tmp)
    execution_id, authorization_path = next(iter(case.authorization_paths.items()))
    authorization = json.loads(authorization_path.read_text())
    authorization.pop("sha256")
    authorization["job"]["sha256"] = "0" * 64
    _write_json(authorization_path, authorization)
    case.input_payload["authorizations"][execution_id] = _binding(
        authorization_path, canonical=True
    )
    case.rewrite_input()
    with pytest.raises(ScaffoldContractError, match="authorization relation drifted at job"):
        _build_activation_from_context(
            activation_inputs_path=case.full_path,
            context=case.context,
            output_dir=case.output_dir,
            output_relative_root=case.output_relative,
        )
    case.assert_no_publish()


def test_existing_v3_target_is_never_overwritten(repo_tmp: Path) -> None:
    case = _build_case(repo_tmp)
    case.output_dir.mkdir()
    sentinel = case.output_dir / "sentinel"
    sentinel.write_text("preserve\n")
    with pytest.raises(ScaffoldContractError, match="Refusing to overwrite"):
        _build_activation_from_context(
            activation_inputs_path=case.full_path,
            context=case.context,
            output_dir=case.output_dir,
            output_relative_root=case.output_relative,
        )
    assert sentinel.read_text() == "preserve\n"
