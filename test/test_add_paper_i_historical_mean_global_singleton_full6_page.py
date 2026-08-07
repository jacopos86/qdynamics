from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
from typing import Any

import pytest

from pipelines.reporting import (
    add_paper_i_historical_mean_global_singleton_full6_page as subject,
)
from pipelines.reporting import (
    build_paper_i_historical_mean_global_singleton_ra_projection as projection_cli,
)


ACTIVATION_DIRS = {
    3: subject.REPO_ROOT
    / (
        "chtc/paper_i_ra_adapt_repair_20260727/"
        "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
        "nph3_r50_20260802_v3_chtc_activation_ordinary_v1"
    ),
    7: subject.REPO_ROOT
    / (
        "chtc/paper_i_ra_adapt_repair_20260727/"
        "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
        "nph7_r50_20260802_v2_chtc_activation_ordinary_v1"
    ),
}


def _json_bytes(value: Any) -> bytes:
    return subject.canonical_json_bytes(value) + b"\n"


def _work(s_alg: int) -> dict[str, Any]:
    return {
        "components": {
            "n_h_outer": s_alg,
            "n_h_refit": 0,
            "n_grad": 0,
            "n_metric": 0,
        },
        "s_alg": s_alg,
    }


def _summary(*, exact: float) -> dict[str, Any]:
    points = [
        {
            "controller_round": round_index,
            "active_ansatz_depth": round_index,
            "accepted_energy": exact + 0.4 / (round_index + 1),
            "exact_same_cutoff_energy": exact,
            "absolute_energy_error": 0.4 / (round_index + 1),
            "projective_state_fingerprint": f"state-{round_index}",
            "checkpoint_sha256": f"{round_index:064x}"[-64:],
        }
        for round_index in range(1, 51)
    ]
    marker = subject._effective_plateau(
        [
            {
                "round": 0,
                "energy": exact + 1.0,
                "delta_e": 1.0,
            },
            *[
                {
                    "round": row["controller_round"],
                    "energy": row["accepted_energy"],
                    "delta_e": row["absolute_energy_error"],
                }
                for row in points
            ],
        ],
        label="fixture",
    )
    return {
        "schema": "paper_i_run_summary_v1",
        "available_controller_rounds": 50,
        "accepted_error_trace": points,
        "effective_plateau": {
            "policy": marker["policy"],
            "controller_round": marker["round"],
            "available_horizon_controller_rounds": 50,
            "absolute_energy_error": marker["delta_e"],
        },
        "requested_rounds": [
            {
                "controller_round": 50,
                "status": "available",
                "algorithmic_work": _work(50),
                "resources": {
                    "compile_convention": "table_i_basis_gate_transpile_v1"
                },
            }
        ],
        "canonical_all_work": _work(50),
        "provenance": {
            "candidate_representation": "single_pauli_word_v1",
            "route_contract_sha256": subject.ROUTE_CONTRACT_SHA256,
            "exact_same_cutoff_energy": exact,
        },
    }


def _result(*, exact: float, route_profile: str) -> dict[str, Any]:
    energies = [exact + 0.4 / (round_index + 1) for round_index in range(1, 51)]
    return {
        "schema": "paper_i_ra_adapt_result_v1",
        "run": {
            "stop": {"completed_controller_rounds": 50},
            "route": {
                "profile": route_profile,
                "contract_sha256": subject.ROUTE_CONTRACT_SHA256,
            },
            "problem": {"problem_request_sha256": "p" * 64},
            "accepted_transitions": [
                {
                    "controller_round": round_index,
                    "energy_before": (
                        exact + 1.0 if round_index == 1 else energies[round_index - 2]
                    ),
                    "energy_after": energies[round_index - 1],
                    "cumulative_s_alg": round_index,
                }
                for round_index in range(1, 51)
            ],
            "accepted_trajectory": [
                {"controller_round": round_index, "energy": energies[round_index - 1]}
                for round_index in range(1, 51)
            ],
            "canonical_reporting": {
                "exact_same_cutoff_energy": exact,
                "candidate_representation": "single_pauli_word_v1",
                "accepted_prefix_work": [_work(k) for k in range(1, 51)],
            },
            "estimator_accounting": {"all_work": _work(50)},
        },
    }


def _ledger() -> dict[str, Any]:
    return {
        "schema": "paper_i_estimator_call_ledger_sidecar_v2",
        "accounting": {
            "complete": True,
            "status": "resolved_from_live_state_keyed_instrumentation",
            "components": {
                "N_H_outer": 50,
                "N_H_refit": 0,
                "N_grad": 0,
                "N_metric": 0,
            },
            "S_alg": 50,
        },
        "ledger": {
            "schema": "estimator_call_ledger_v1",
            "occurrence_summary": {"S_alg": 50},
        },
        "adapt_success": True,
        "adapt_error": None,
    }


def _attempt_archive(
    root: Path,
    *,
    regime: str,
    omit_summary: bool = False,
    drift_outer_binding: bool = False,
) -> Path:
    nph = subject.NPH_BY_REGIME[regime]
    spec = subject.PACKAGE_SPECS[nph]
    execution_id = subject.expected_execution_id(regime)
    package_dir = Path(spec["package_dir"])
    job_path = package_dir / "jobs" / f"{execution_id}.json"
    activation_dir = ACTIVATION_DIRS[nph]
    authorization_path = activation_dir / "authorizations" / f"{execution_id}.json"
    activation_path = activation_dir / "activation_manifest.json"
    job_bytes = job_path.read_bytes()
    authorization_bytes = authorization_path.read_bytes()
    activation_bytes = activation_path.read_bytes()
    job = json.loads(job_bytes)
    authorization = json.loads(authorization_bytes)
    activation = json.loads(activation_bytes)
    exact = float(job["exact_same_cutoff_energy"])

    artifacts: dict[str, bytes] = {
        "checkpoint.json": _json_bytes({"schema": "fixture_checkpoint_v1"}),
        "estimator_ledger.json": _json_bytes(_ledger()),
        "paper_i_summary.json": _json_bytes(_summary(exact=exact)),
        "result.json": _json_bytes(
            _result(exact=exact, route_profile=str(job.get("route_profile", "route")))
        ),
    }
    if omit_summary:
        artifacts.pop("paper_i_summary.json")
    preliminary = {
        name: {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        for name, payload in artifacts.items()
    }
    execution = subject.digested(
        {
            "schema": spec["execution_schema"],
            "status": "passed",
            "package_id": spec["package_id"],
            "campaign_id": spec["campaign_id"],
            "execution_id": execution_id,
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": authorization["sha256"],
            "protocol_sha256": job["protocol_sha256"],
            "target_horizon": 50,
            "controller_rounds_completed": 50,
            "fresh_start": True,
            "source_checkpoint_consumed": False,
            "output_payloads": preliminary,
        }
    )
    artifacts["execution_manifest.json"] = _json_bytes(execution)
    worker = subject.digested(
        {
            "schema": spec["worker_schema"],
            "status": "passed",
            "package_id": spec["package_id"],
            "campaign_id": spec["campaign_id"],
            "execution_id": execution_id,
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": authorization["sha256"],
            "execution_manifest_sha256": execution["sha256"],
            "controller_rounds_completed": 50,
            "fresh_start": True,
            "artifacts": [
                {
                    "path": name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
                for name, payload in sorted(artifacts.items())
            ],
        }
    )
    cluster = 12345 + nph
    proc = next(
        int(row["queue_index"])
        for row in activation["executions"]
        if row["execution_id"] == execution_id
    )
    worker_files: dict[str, bytes] = {
        "attempt_identity.tsv": (
            f"{execution_id}\t{cluster}\t{proc}\t1\n".encode("ascii")
        ),
        "worker_exit_status.txt": b"0\n",
        "worker_receipt.json": _json_bytes(worker),
        **{f"artifacts/{name}": payload for name, payload in artifacts.items()},
    }
    rows = [
        {
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        for relative, payload in sorted(worker_files.items())
    ]
    if drift_outer_binding:
        rows[0]["sha256"] = "f" * 64
    outer = subject.digested(
        {
            "schema": spec["attempt_schema"],
            "execution_id": execution_id,
            "cluster_id": cluster,
            "proc_id": proc,
            "attempt_ordinal": 1,
            "worker_exit_status": 0,
            "job_file_sha256": hashlib.sha256(job_bytes).hexdigest(),
            "authorization_file_sha256": hashlib.sha256(
                authorization_bytes
            ).hexdigest(),
            "activation_manifest_file_sha256": hashlib.sha256(
                activation_bytes
            ).hexdigest(),
            "source_archive_sha256": subject.SOURCE_ARCHIVE_SHA256,
            "image_sha256": authorization["remote_image_sha256"],
            "worker_files": rows,
        }
    )
    members = {
        **{f"worker_outputs/{relative}": payload for relative, payload in worker_files.items()},
        "authority/job.json": job_bytes,
        "authority/execution_authorization.json": authorization_bytes,
        "authority/activation_manifest.json": activation_bytes,
        "worker_attempt_receipt.json": _json_bytes(outer),
    }
    output = root / f"{regime}.tar.gz"
    with tarfile.open(output, "w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return output


def _resume_attempt_archive(root: Path, *, regime: str) -> Path:
    spec = subject.RESUME_SPEC
    execution_id = subject.expected_resume_execution_id(regime)
    package_dir = Path(spec["package_dir"])
    activation_dir = Path(spec["activation_dir"])
    job_path = package_dir / "jobs" / f"{execution_id}.json"
    authorization_path = activation_dir / "authorizations" / f"{execution_id}.json"
    activation_path = activation_dir / "activation_manifest.json"
    job_bytes = job_path.read_bytes()
    authorization_bytes = authorization_path.read_bytes()
    activation_bytes = activation_path.read_bytes()
    job = json.loads(job_bytes)
    authorization = json.loads(authorization_bytes)
    activation = json.loads(activation_bytes)
    source_job_path = subject.REPO_ROOT / job["source_job"]["path"]
    source_job = json.loads(source_job_path.read_text(encoding="utf-8"))
    exact = float(source_job["exact_same_cutoff_energy"])

    artifacts: dict[str, bytes] = {
        "checkpoint.json": _json_bytes({"schema": "fixture_checkpoint_v1"}),
        "checkpoint.estimator_call_ledger_checkpoint.fixture.json": _json_bytes(
            {"schema": "fixture_ledger_checkpoint_v1"}
        ),
        "checkpoint.verified_singleton_resume.fixture.json": _json_bytes(
            {"schema": "fixture_verified_resume_v1"}
        ),
        "estimator_ledger.json": _json_bytes(_ledger()),
        "paper_i_summary.json": _json_bytes(_summary(exact=exact)),
        "result.json": _json_bytes(
            _result(exact=exact, route_profile=str(job["route_profile"]))
        ),
    }
    preliminary = {
        name: {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        for name, payload in artifacts.items()
    }
    resume_round = int(job["resume_input"]["resume_controller_round"])
    execution = subject.digested(
        {
            "schema": spec["execution_schema"],
            "status": "passed",
            "package_id": spec["package_id"],
            "campaign_id": spec["campaign_id"],
            "execution_id": execution_id,
            "source_execution_id": job["source_execution_id"],
            "job_sha256": job["sha256"],
            "authorization_sha256": authorization["sha256"],
            "scientific_protocol_sha256": job["scientific_protocol_sha256"],
            "scientific_protocol_changed": False,
            "scientific_settings_changed": [],
            "source_checkpoint_sha256": job["resume_input"]["checkpoint_sha256"],
            "resume_controller_round": resume_round,
            "controller_rounds_completed": 50,
            "target_horizon": 50,
            "source_held_job_preserved": True,
            "output_payloads": preliminary,
        }
    )
    artifacts["execution_manifest.json"] = _json_bytes(execution)
    worker = subject.digested(
        {
            "schema": spec["worker_schema"],
            "status": "passed",
            "package_id": spec["package_id"],
            "campaign_id": spec["campaign_id"],
            "execution_id": execution_id,
            "job_sha256": job["sha256"],
            "authorization_sha256": authorization["sha256"],
            "execution_manifest_sha256": execution["sha256"],
            "controller_rounds_completed": 50,
            "source_checkpoint_consumed": True,
            "source_checkpoint_sha256": job["resume_input"]["checkpoint_sha256"],
            "source_held_job_preserved": True,
            "artifacts": [
                {
                    "path": name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
                for name, payload in sorted(artifacts.items())
            ],
        }
    )
    cluster = 22345
    activation_execution = next(
        row for row in activation["executions"] if row["execution_id"] == execution_id
    )
    proc = int(activation_execution["queue_index"])
    worker_files: dict[str, bytes] = {
        "attempt_identity.tsv": (
            f"{execution_id}\t{cluster}\t{proc}\t1\n".encode("ascii")
        ),
        "worker_exit_status.txt": b"0\n",
        "worker_receipt.json": _json_bytes(worker),
        **{f"artifacts/{name}": payload for name, payload in artifacts.items()},
    }
    rows = [
        {
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        for relative, payload in sorted(worker_files.items())
    ]

    def worker_binding(relative: str) -> dict[str, Any]:
        payload = worker_files[relative]
        return {
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    outer = subject.digested(
        {
            "schema": spec["attempt_schema"],
            "execution_id": execution_id,
            "cluster_id": cluster,
            "proc_id": proc,
            "attempt_ordinal": 1,
            "worker_exit_status": 0,
            "job_file_sha256": hashlib.sha256(job_bytes).hexdigest(),
            "authorization_file_sha256": hashlib.sha256(
                authorization_bytes
            ).hexdigest(),
            "activation_manifest_file_sha256": hashlib.sha256(
                activation_bytes
            ).hexdigest(),
            "source_archive_sha256": subject.SOURCE_ARCHIVE_SHA256,
            "resume_archive_sha256": job["resume_input"]["archive"]["sha256"],
            "image_sha256": activation["remote_image"]["sha256"],
            "worker_files": rows,
            "resumable_checkpoint_triplets": [
                {
                    "checkpoint": worker_binding("artifacts/checkpoint.json"),
                    "estimator_ledger_checkpoint": worker_binding(
                        "artifacts/checkpoint.estimator_call_ledger_checkpoint.fixture.json"
                    ),
                    "verified_resume_sidecar": worker_binding(
                        "artifacts/checkpoint.verified_singleton_resume.fixture.json"
                    ),
                    "pointer_closed_by_sibling_identity": True,
                }
            ],
            "resumable_checkpoint_triplet_count": 1,
            "failure_safe_checkpoint_transfer": True,
        }
    )
    members = {
        **{
            f"worker_outputs/{relative}": payload
            for relative, payload in worker_files.items()
        },
        "authority/job.json": job_bytes,
        "authority/execution_authorization.json": authorization_bytes,
        "authority/activation_manifest.json": activation_bytes,
        "worker_attempt_receipt.json": _json_bytes(outer),
    }
    output = root / f"{regime}-resume.tar.gz"
    with tarfile.open(output, "w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return output


def _append_adapter(path: Path) -> Path:
    cells: list[dict[str, Any]] = []
    for regime in subject.REGIME_ORDER:
        execution_id = subject.expected_execution_id(regime)
        nph = subject.NPH_BY_REGIME[regime]
        job = json.loads(
            (
                Path(subject.PACKAGE_SPECS[nph]["package_dir"])
                / "jobs"
                / f"{execution_id}.json"
            ).read_text(encoding="utf-8")
        )
        exact = float(job["exact_same_cutoff_energy"])
        points = [
            {
                "round": round_index,
                "energy": exact + 0.5 / (round_index + 1),
                "delta_e": 0.5 / (round_index + 1),
            }
            for round_index in range(71)
        ]
        cells.append(
            {
                "regime_id": regime,
                "display_name": subject.REGIME_LABELS[regime],
                "execution_id": f"append-{regime}",
                "nph": nph,
                "exact_same_cutoff_energy": exact,
                "points": points,
                "endpoints": {
                    f"round_{round_index}": {
                        **copy.deepcopy(points[round_index]),
                        "costs": {
                            "N2q": round_index,
                            "D2q": round_index,
                            "Dc": round_index,
                            "W1q": round_index,
                            "S_alg": round_index,
                        },
                        "checkpoint_sha256": f"{round_index:064x}"[-64:],
                        "compile": {
                            "compile_convention": "table_i_basis_gate_transpile_v1"
                        },
                    }
                    for round_index in (50, 70)
                },
                "source": {"fixture": True},
            }
        )
    adapter = subject.digested(
        {
            "schema": subject.APPEND_ADAPTER_SCHEMA,
            "status": "passed",
            "regime_order": list(subject.REGIME_ORDER),
            "completed_regimes": list(subject.REGIME_ORDER),
            "pending_regimes": [],
            "same_cutoff_reference": {
                "path": "fixture.json",
                "sha256": "a" * 64,
            },
            "cells": cells,
        }
    )
    path.write_bytes(_json_bytes(adapter))
    return path


def _cost_provider(
    method: str,
    cell: dict[str, Any],
    controller_round: int,
    error: float,
) -> dict[str, Any]:
    offset = 100 if method == "ra" else 200
    return {
        "round": controller_round,
        "delta_e": error,
        "costs": {
            "N2q": offset + controller_round,
            "D2q": offset + controller_round,
            "Dc": offset + controller_round,
            "W1q": offset + controller_round,
            "S_alg": controller_round,
        },
        "checkpoint_sha256": f"{controller_round:064x}"[-64:],
        "compile": {
            "compile_convention": "table_i_basis_gate_transpile_v1",
            "source": "authenticated_prefix_shared_locked_compiler_v1",
        },
    }


def _attempt_group(tmp_path: Path, regimes: tuple[str, ...]) -> dict[str, Path]:
    return {
        regime: _attempt_archive(tmp_path, regime=regime) for regime in regimes
    }


def _archive_validation(
    path: Path, *, archive: Path, regime: str
) -> Path:
    cell = subject.validate_attempt_archive(archive, regime=regime)
    source = cell["source"]
    archive_binding = subject.file_binding(archive)
    validation = subject.digested(
        {
            "schema": subject.NPH3_ARCHIVE_VALIDATION_SCHEMA,
            "status": "passed",
            "execution_id": cell["execution_id"],
            "cluster_id": source["cluster_id"],
            "proc_id": source["proc_id"],
            "attempt_ordinal": source["attempt_ordinal"],
            "controller_rounds_completed": 50,
            "archive": archive_binding,
            "worker_attempt_receipt": {
                "schema": subject.PACKAGE_SPECS[3]["attempt_schema"],
                "canonical_sha256": source[
                    "attempt_receipt_canonical_sha256"
                ],
                "file_sha256": "1" * 64,
                "worker_exit_status": 0,
            },
            "worker_receipt": {
                "schema": subject.PACKAGE_SPECS[3]["worker_schema"],
                "canonical_sha256": source[
                    "worker_receipt_canonical_sha256"
                ],
                "file_sha256": "2" * 64,
                "controller_rounds_completed": 50,
            },
            "execution_manifest": {
                "schema": subject.PACKAGE_SPECS[3]["execution_schema"],
                "canonical_sha256": source[
                    "execution_manifest_canonical_sha256"
                ],
                "file_sha256": "3" * 64,
                "controller_rounds_completed": 50,
            },
            "member_validation": {
                "gzip_and_full_tar_scan_passed": True,
                "compressed_hash_size_stream_closure_passed": True,
                "safe_unique_regular_only_member_closure_passed": True,
                "worker_inventory_hash_size_closure_passed": True,
                "nested_artifact_inventory_closure_passed": True,
                "authority_byte_identity_passed": True,
                "fifty_round_success_closure_passed": True,
                "member_count": 13,
                "worker_file_count": 9,
            },
            "bindings": {
                "job": {
                    "canonical_sha256": source["job_canonical_sha256"]
                },
                "authorization": {
                    "canonical_sha256": source[
                        "authorization_canonical_sha256"
                    ]
                },
                "source_archive_sha256": subject.SOURCE_ARCHIVE_SHA256,
                "image_sha256": "4" * 64,
            },
        }
    )
    path.write_bytes(_json_bytes(validation))
    return path


def test_compact_projection_builds_without_a_local_attempt_archive(
    tmp_path: Path,
) -> None:
    regime = "intermediate_weak"
    append_path = _append_adapter(tmp_path / "append.json")
    archive = _attempt_archive(tmp_path, regime=regime)
    validation = _archive_validation(
        tmp_path / "archive-validation.json", archive=archive, regime=regime
    )
    archive_binding = subject.file_binding(archive)
    projection_path = tmp_path / "projection.json"
    projection = subject.build_compact_ra_projection(
        append_adapter_path=append_path,
        regime=regime,
        archive_path=archive,
        archive_validation_path=validation,
        remote_archive_path=(
            "/home/jsstrobel/Holstein_phase3_optuna_chtc/fetched/"
            "intermediate_weak.tar.gz"
        ),
        remote_archive_sha256=archive_binding["sha256"],
        remote_archive_size_bytes=archive_binding["size_bytes"],
        output=projection_path,
        prefix_cost_provider=_cost_provider,
    )
    archive.unlink()

    adapter_path = tmp_path / "adapter.json"
    adapter = subject.build_adapter(
        append_adapter_path=append_path,
        ra_attempts={},
        ra_projections={regime: projection_path},
        output=adapter_path,
        prefix_cost_provider=_cost_provider,
    )

    assert projection["source_archive"]["state"] == (
        "preserved_remote_not_fetched"
    )
    assert adapter["completed_regimes"] == [regime]
    cell = next(
        row for row in adapter["cells"] if row["regime_id"] == regime
    )
    assert cell["ra"]["source"]["archive"] == projection["source_archive"]
    assert cell["ra"]["source"]["compact_projection"][
        "full_archive_preserved_remote"
    ] is True
    assert subject.validate_adapter(adapter_path)["sha256"] == adapter["sha256"]


def test_compact_projection_rejects_rehashed_trajectory_drift(
    tmp_path: Path,
) -> None:
    regime = "intermediate_weak"
    append_path = _append_adapter(tmp_path / "append.json")
    archive = _attempt_archive(tmp_path, regime=regime)
    validation = _archive_validation(
        tmp_path / "archive-validation.json", archive=archive, regime=regime
    )
    archive_binding = subject.file_binding(archive)
    projection_path = tmp_path / "projection.json"
    projection = subject.build_compact_ra_projection(
        append_adapter_path=append_path,
        regime=regime,
        archive_path=archive,
        archive_validation_path=validation,
        remote_archive_path="/home/jsstrobel/fetched/intermediate_weak.tar.gz",
        remote_archive_sha256=archive_binding["sha256"],
        remote_archive_size_bytes=archive_binding["size_bytes"],
        output=projection_path,
        prefix_cost_provider=_cost_provider,
    )
    unsigned = copy.deepcopy(projection)
    unsigned.pop("sha256")
    unsigned["cell"]["points"][10]["delta_e"] += 1.0e-6
    projection_path.write_bytes(_json_bytes(subject.digested(unsigned)))
    append = subject.validate_append_adapter(append_path)

    with pytest.raises(subject.Page7InputError, match="trajectory drifted"):
        subject.validate_compact_ra_projection(
            projection_path, regime=regime, append=append
        )


def test_compact_projection_accepts_observed_roundoff_but_not_larger_drift(
    tmp_path: Path,
) -> None:
    regime = "intermediate_weak"
    append_path = _append_adapter(tmp_path / "append.json")
    archive = _attempt_archive(tmp_path, regime=regime)
    validation = _archive_validation(
        tmp_path / "archive-validation.json", archive=archive, regime=regime
    )
    archive_binding = subject.file_binding(archive)
    projection_path = tmp_path / "projection.json"
    projection = subject.build_compact_ra_projection(
        append_adapter_path=append_path,
        regime=regime,
        archive_path=archive,
        archive_validation_path=validation,
        remote_archive_path="/home/jsstrobel/fetched/intermediate_weak.tar.gz",
        remote_archive_sha256=archive_binding["sha256"],
        remote_archive_size_bytes=archive_binding["size_bytes"],
        output=projection_path,
        prefix_cost_provider=_cost_provider,
    )
    append = subject.validate_append_adapter(append_path)

    for observed_offset in (
        3.735900477863652e-14,
        6.417089082333405e-14,
    ):
        compatible = copy.deepcopy(projection)
        compatible.pop("sha256")
        compatible["source_code"]["updater"] = copy.deepcopy(
            subject.PRIOR_COMPACT_PROJECTION_UPDATER_BINDING
        )
        compatible["cell"]["exact_same_cutoff_energy"] += observed_offset
        projection_path.write_bytes(_json_bytes(subject.digested(compatible)))

        validated = subject.validate_compact_ra_projection(
            projection_path, regime=regime, append=append
        )
        assert validated["cell"]["regime_id"] == regime

    incompatible = copy.deepcopy(projection)
    incompatible.pop("sha256")
    incompatible["source_code"]["updater"] = copy.deepcopy(
        subject.PRIOR_COMPACT_PROJECTION_UPDATER_BINDING
    )
    incompatible["cell"]["exact_same_cutoff_energy"] += (
        10.0 * subject.COMPACT_PROJECTION_DELTA_E_ABS_TOL
    )
    projection_path.write_bytes(_json_bytes(subject.digested(incompatible)))

    with pytest.raises(subject.Page7InputError, match="trajectory drifted"):
        subject.validate_compact_ra_projection(
            projection_path, regime=regime, append=append
        )


def test_compact_projection_rejects_rehashed_source_binding_drift(
    tmp_path: Path,
) -> None:
    regime = "intermediate_weak"
    append_path = _append_adapter(tmp_path / "append.json")
    archive = _attempt_archive(tmp_path, regime=regime)
    validation = _archive_validation(
        tmp_path / "archive-validation.json", archive=archive, regime=regime
    )
    archive_binding = subject.file_binding(archive)
    projection_path = tmp_path / "projection.json"
    projection = subject.build_compact_ra_projection(
        append_adapter_path=append_path,
        regime=regime,
        archive_path=archive,
        archive_validation_path=validation,
        remote_archive_path="/home/jsstrobel/fetched/intermediate_weak.tar.gz",
        remote_archive_sha256=archive_binding["sha256"],
        remote_archive_size_bytes=archive_binding["size_bytes"],
        output=projection_path,
        prefix_cost_provider=_cost_provider,
    )
    unsigned = copy.deepcopy(projection)
    unsigned.pop("sha256")
    unsigned["source_code"]["updater"]["sha256"] = "f" * 64
    projection_path.write_bytes(_json_bytes(subject.digested(unsigned)))
    append = subject.validate_append_adapter(append_path)

    with pytest.raises(subject.Page7InputError, match="updater byte binding drifted"):
        subject.validate_compact_ra_projection(
            projection_path, regime=regime, append=append
        )


def test_projection_builder_cli_help_and_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as help_exit:
        projection_cli.parser().parse_args(["--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert "--archive-validation" in help_text
    assert "--remote-archive-sha256" in help_text

    captured: dict[str, Any] = {}

    def fake_build(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "regime_id": "intermediate_weak",
            "execution_id": subject.expected_execution_id("intermediate_weak"),
            "sha256": "a" * 64,
            "source_archive": {
                "path": "/home/jsstrobel/fetched/result.tar.gz",
                "sha256": "b" * 64,
                "size_bytes": 123,
                "state": "preserved_remote_not_fetched",
            },
        }

    monkeypatch.setattr(
        projection_cli.page7, "build_compact_ra_projection", fake_build
    )
    status = projection_cli.main(
        [
            "--append-adapter",
            str(tmp_path / "append.json"),
            "--regime",
            "intermediate_weak",
            "--archive",
            str(tmp_path / "result.tar.gz"),
            "--archive-validation",
            str(tmp_path / "validation.json"),
            "--remote-archive-path",
            "/home/jsstrobel/fetched/result.tar.gz",
            "--remote-archive-sha256",
            "b" * 64,
            "--remote-archive-size-bytes",
            "123",
            "--output",
            str(tmp_path / "projection.json"),
        ]
    )

    assert status == 0
    assert captured["regime"] == "intermediate_weak"
    assert captured["remote_archive_size_bytes"] == 123
    output = json.loads(capsys.readouterr().out)
    assert output["projection_sha256"] == "a" * 64


def test_build_partial_adapter_accepts_complete_nph3_and_marks_nph7_pending(
    tmp_path: Path,
) -> None:
    append_path = _append_adapter(tmp_path / "append.json")
    attempts = _attempt_group(tmp_path, subject.REGIME_ORDER[:3])
    output = tmp_path / "adapter.json"

    adapter = subject.build_adapter(
        append_adapter_path=append_path,
        ra_attempts=attempts,
        output=output,
        prefix_cost_provider=_cost_provider,
    )

    assert adapter["completed_regimes"] == list(subject.REGIME_ORDER[:3])
    assert adapter["pending_regimes"] == list(subject.REGIME_ORDER[3:])
    cells = {cell["regime_id"]: cell for cell in adapter["cells"]}
    assert [point["round"] for point in cells["weak_weak"]["ra"]["points"]] == list(
        range(51)
    )
    assert cells["weak_weak"]["ra"]["terminal"]["costs"]["S_alg"] == 50
    for regime in subject.REGIME_ORDER[:3]:
        append = cells[regime]["append"]
        assert append["display_terminal_round"] == 50
        assert [point["round"] for point in append["points"]] == list(range(51))
        assert append["terminal"]["round"] == 50
        assert cells[regime]["common_accuracy"]["policy"] == (
            "display_horizon_equal_attainable_error_v2"
        )
        assert cells[regime]["common_accuracy"]["append_horizon"] == 50
    for regime in subject.REGIME_ORDER[3:]:
        append = cells[regime]["append"]
        assert append["display_terminal_round"] == 70
        assert [point["round"] for point in append["points"]] == list(range(71))
        assert append["terminal"]["round"] == 70
    assert cells["weak_strong"]["ra"] is None
    assert subject.validate_adapter(output)["sha256"] == adapter["sha256"]


def test_adapter_accepts_zero_then_one_completed_archive(tmp_path: Path) -> None:
    append_path = _append_adapter(tmp_path / "append.json")
    output = tmp_path / "adapter.json"
    pending = subject.build_adapter(
        append_adapter_path=append_path,
        ra_attempts={},
        output=output,
        prefix_cost_provider=_cost_provider,
    )
    assert pending["status"] == "passed_pending"
    assert pending["completed_regimes"] == []
    assert pending["pending_regimes"] == list(subject.REGIME_ORDER)

    archive = _attempt_archive(tmp_path, regime="weak_weak")
    partial = subject.build_adapter(
        append_adapter_path=append_path,
        ra_attempts={"weak_weak": archive},
        output=output,
        prefix_cost_provider=_cost_provider,
    )
    assert partial["completed_regimes"] == ["weak_weak"]
    assert partial["pending_regimes"] == list(subject.REGIME_ORDER[1:])


def test_adapter_rejects_rehashed_mixed_horizon_policy_drift(
    tmp_path: Path,
) -> None:
    append_path = _append_adapter(tmp_path / "append.json")
    output = tmp_path / "adapter.json"
    adapter = subject.build_adapter(
        append_adapter_path=append_path,
        ra_attempts={},
        output=output,
        prefix_cost_provider=_cost_provider,
    )
    unsigned = copy.deepcopy(adapter)
    unsigned.pop("sha256")
    unsigned["display_rounds_by_regime"]["weak_weak"]["maximum"] = 70
    output.write_bytes(_json_bytes(subject.digested(unsigned)))

    with pytest.raises(subject.Page7InputError, match="reporting policy drifted"):
        subject.validate_adapter(output)


def test_adapter_rejects_rehashed_single_marker_drift(tmp_path: Path) -> None:
    append_path = _append_adapter(tmp_path / "append.json")
    output = tmp_path / "adapter.json"
    adapter = subject.build_adapter(
        append_adapter_path=append_path,
        ra_attempts={},
        output=output,
        prefix_cost_provider=_cost_provider,
    )
    unsigned = copy.deepcopy(adapter)
    unsigned.pop("sha256")
    weak = next(
        cell for cell in unsigned["cells"] if cell["regime_id"] == "weak_weak"
    )
    weak["append"]["effective_plateau"]["round"] -= 1
    output.write_bytes(_json_bytes(subject.digested(unsigned)))

    with pytest.raises(subject.Page7InputError, match="Append marker drifted"):
        subject.validate_adapter(output)


def test_page_tex_labels_weak_append_50_and_strong_append_70(
    tmp_path: Path,
) -> None:
    append_path = _append_adapter(tmp_path / "append.json")
    adapter = subject.build_adapter(
        append_adapter_path=append_path,
        ra_attempts={},
        output=tmp_path / "adapter.json",
        prefix_cost_provider=_cost_provider,
    )
    tex_path = tmp_path / "page.tex"
    subject.write_page_tex(
        adapter,
        plot_pdf=tmp_path / "plot.pdf",
        tex_path=tex_path,
    )
    tex = tex_path.read_text(encoding="utf-8")

    assert "$k_A$" in tex
    assert tex.count(" & 50 &") == 3
    assert tex.count(" & 70 &") == 3
    assert "$k_A=50$ for weak-Holstein" in tex
    assert "$k_A=70$ for" in tex


def test_attempt_rejects_missing_completed_artifact(tmp_path: Path) -> None:
    archive = _attempt_archive(
        tmp_path, regime="weak_weak", omit_summary=True
    )
    with pytest.raises(subject.Page7InputError, match="members missing"):
        subject.validate_attempt_archive(archive, regime="weak_weak")


def test_attempt_rejects_rehashed_false_worker_binding(tmp_path: Path) -> None:
    archive = _attempt_archive(
        tmp_path, regime="weak_weak", drift_outer_binding=True
    )
    with pytest.raises(subject.Page7InputError, match="byte binding drifted"):
        subject.validate_attempt_archive(archive, regime="weak_weak")


def test_attempt_accepts_authenticated_nph7_resume_with_extra_sidecars(
    tmp_path: Path,
) -> None:
    archive = _resume_attempt_archive(tmp_path, regime="weak_strong")

    cell = subject.validate_attempt_archive(archive, regime="weak_strong")

    assert cell["execution_id"] == subject.expected_resume_execution_id(
        "weak_strong"
    )
    assert [point["round"] for point in cell["points"]] == list(range(51))
    assert cell["source"]["resume_controller_round"] == 35
    assert cell["source"]["large_members_extracted"] is False


def test_monotone_adapter_rejects_append_drift_in_pending_cell(
    tmp_path: Path,
) -> None:
    append_path = _append_adapter(tmp_path / "append.json")
    existing_path = tmp_path / "existing.json"
    subject.build_adapter(
        append_adapter_path=append_path,
        ra_attempts={},
        output=existing_path,
        prefix_cost_provider=_cost_provider,
    )
    archive = _attempt_archive(tmp_path, regime="weak_weak")
    candidate_path = tmp_path / "candidate.json"
    candidate = subject.build_adapter(
        append_adapter_path=append_path,
        ra_attempts={"weak_weak": archive},
        output=candidate_path,
        prefix_cost_provider=_cost_provider,
    )
    candidate.pop("sha256")
    pending = next(
        cell
        for cell in candidate["cells"]
        if cell["regime_id"] == "weak_strong"
    )
    pending["append"]["points"][1]["delta_e"] += 1.0e-6
    drifted = subject.digested(candidate)

    with pytest.raises(subject.Page7InputError, match="Append cell drifted"):
        subject._write_monotone_adapter(existing_path, drifted)


def _page_asset(tmp_path: Path, width: float) -> dict[str, Path]:
    from pypdf import PdfWriter

    page_pdf = tmp_path / f"page-{width}.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=width, height=200)
    with page_pdf.open("wb") as stream:
        writer.write(stream)
    plot_png = tmp_path / "plot.png"
    plot_pdf = tmp_path / "plot.pdf"
    page_tex = tmp_path / "page.tex"
    plot_png.write_bytes(b"png")
    plot_pdf.write_bytes(b"pdf")
    page_tex.write_text("tex", encoding="utf-8")
    return {
        "plot_png": plot_png,
        "plot_pdf": plot_pdf,
        "page_tex": page_tex,
        "page_pdf": page_pdf,
    }


def _legacy_adapter_from_mixed(
    path: Path,
    *,
    mixed: dict[str, Any],
    append_path: Path,
) -> dict[str, Any]:
    source = json.loads(append_path.read_text(encoding="utf-8"))
    source_cells = {cell["regime_id"]: cell for cell in source["cells"]}
    legacy = copy.deepcopy(mixed)
    legacy.pop("sha256")
    legacy["schema"] = subject.LEGACY_ADAPTER_SCHEMA
    legacy.pop("display_rounds_by_regime")
    legacy.pop("append_rounds_by_regime")
    legacy["display_rounds"] = {"minimum": 0, "maximum": 70}
    legacy["append_rounds"] = {"minimum": 0, "maximum": 70}
    legacy["cost_policy"] = {
        "tuple_fields": list(subject.COST_FIELDS),
        "terminal": {"ra_round": 50, "append_round": 70},
        "matched": "full_horizon_equal_attainable_error_v1",
        "compile_convention": "table_i_basis_gate_transpile_v1",
        "optimization_level": 0,
        "seed_transpiler": 7,
        "reference_state_included": True,
    }
    legacy["limitations"] = [subject.LEGACY_LIMITATION]
    for cell in legacy["cells"]:
        regime = cell["regime_id"]
        source_cell = source_cells[regime]
        cell["append"] = {
            "execution_id": source_cell["execution_id"],
            "exact_same_cutoff_energy": source_cell["exact_same_cutoff_energy"],
            "points": copy.deepcopy(source_cell["points"]),
            "effective_plateau": subject._effective_plateau(
                source_cell["points"], label=f"legacy fixture {regime}"
            ),
            "terminal": copy.deepcopy(source_cell["endpoints"]["round_70"]),
            "source": copy.deepcopy(source_cell["source"]),
        }
        if cell["status"] == "complete":
            # The production predecessor and successor have the same RA crossing
            # rounds. Preserve the successor's authenticated RA observation here
            # so this fixture exercises that exact migration invariant.
            mixed_common = next(
                row["common_accuracy"]
                for row in mixed["cells"]
                if row["regime_id"] == regime
            )
            selection = subject._common_accuracy_selection(
                cell["ra"]["points"], source_cell["points"], regime=regime
            )
            selection["ra_round"] = mixed_common["ra_round"]
            selection["ra_delta_e"] = mixed_common["ra_delta_e"]
            append_round = selection["append_round"]
            append_observation = source_cell["endpoints"].get(
                f"round_{append_round}"
            ) or _cost_provider(
                "append",
                source_cell,
                append_round,
                selection["append_delta_e"],
            )
            cell["common_accuracy"] = {
                **selection,
                "ra": copy.deepcopy(mixed_common["ra"]),
                "append": copy.deepcopy(append_observation),
            }
    result = subject.digested(legacy)
    path.write_bytes(_json_bytes(result))
    return result


def test_page7_migrates_pinned_r70_policy_to_mixed_horizons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pypdf import PdfReader, PdfWriter

    append_path = _append_adapter(tmp_path / "append.json")
    attempts = _attempt_group(tmp_path, subject.REGIME_ORDER[:3])
    mixed_path = tmp_path / "mixed-v2.json"
    mixed = subject.build_adapter(
        append_adapter_path=append_path,
        ra_attempts=attempts,
        output=mixed_path,
        prefix_cost_provider=_cost_provider,
    )
    legacy_path = tmp_path / "legacy-v1.json"
    legacy = _legacy_adapter_from_mixed(
        legacy_path,
        mixed=mixed,
        append_path=append_path,
    )
    legacy_file = subject.file_binding(legacy_path)
    monkeypatch.setattr(
        subject,
        "LEGACY_PAGE7_ADAPTER_BINDING",
        {
            "canonical_sha256": legacy["sha256"],
            "sha256": legacy_file["sha256"],
            "size_bytes": legacy_file["size_bytes"],
        },
    )

    target_pdf = tmp_path / "report.pdf"
    writer = PdfWriter()
    for index in range(7):
        writer.add_blank_page(width=100 + index, height=200)
    with target_pdf.open("wb") as stream:
        writer.write(stream)
    before = subject.legacy_page._page_content_hashes(target_pdf)
    legacy_report = {
        "schema": subject.LEGACY_PAGE_ID,
        "page_id": subject.LEGACY_PAGE_ID,
        "classification": "supplemental_diagnostic_not_adopted_evidence",
        "paper_evidence_adopted": False,
        "adapter": {
            **legacy_file,
            "canonical_sha256": legacy["sha256"],
        },
        "completed_regimes": copy.deepcopy(legacy["completed_regimes"]),
        "pending_regimes": copy.deepcopy(legacy["pending_regimes"]),
    }
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "layout": {
                    "page_count": 7,
                    "page_6": subject.EXPECTED_BASE_PAGE_6,
                    "page_7": subject.LEGACY_PAGE_ID,
                },
                "outputs": {
                    "partial_progress_pdf": subject.file_binding(target_pdf)
                },
                "limitations": [subject.LEGACY_LIMITATION],
                subject.LEGACY_REPORT_KEY: legacy_report,
            }
        ),
        encoding="utf-8",
    )
    assets = _page_asset(tmp_path, 900)
    monkeypatch.setattr(subject, "build_assets", lambda *_args, **_kwargs: assets)

    migrated = subject.update_page7(
        target_pdf=target_pdf,
        target_provenance=provenance_path,
        adapter_path=mixed_path,
        asset_dir=tmp_path,
        asset_stem="mixed-v2",
    )

    assert migrated["status"] == "migrated_page_7_append_horizon_policy"
    assert subject.legacy_page._page_content_hashes(target_pdf)[:6] == before[:6]
    assert float(PdfReader(str(target_pdf)).pages[6].mediabox.width) == 900
    updated = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert updated[subject.LEGACY_REPORT_KEY] == legacy_report
    assert updated["layout"]["page_7"] == subject.PAGE_ID
    assert updated[subject.REPORT_KEY]["reporting_policy_migration"][
        "scientific_ra_evidence_changed"
    ] is False
    assert subject.LEGACY_LIMITATION not in updated["limitations"]
    assert subject.LIMITATION in updated["limitations"]


def test_page7_append_then_completion_replace_preserves_pages_one_to_six(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pypdf import PdfReader, PdfWriter

    append_path = _append_adapter(tmp_path / "append.json")
    adapter_path = tmp_path / "adapter.json"
    partial_attempts = _attempt_group(tmp_path, subject.REGIME_ORDER[:3])
    partial = subject.build_adapter(
        append_adapter_path=append_path,
        ra_attempts=partial_attempts,
        output=adapter_path,
        prefix_cost_provider=_cost_provider,
    )
    target_pdf = tmp_path / "report.pdf"
    writer = PdfWriter()
    for index in range(6):
        writer.add_blank_page(width=100 + index, height=200)
    with target_pdf.open("wb") as stream:
        writer.write(stream)
    before = subject.legacy_page._page_content_hashes(target_pdf)
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "layout": {
                    "page_count": 6,
                    "page_6": subject.EXPECTED_BASE_PAGE_6,
                },
                "outputs": {
                    "partial_progress_pdf": subject.file_binding(target_pdf)
                },
                "limitations": [],
            }
        ),
        encoding="utf-8",
    )
    assets = _page_asset(tmp_path, 700)
    monkeypatch.setattr(subject, "build_assets", lambda *_args, **_kwargs: assets)

    appended = subject.update_page7(
        target_pdf=target_pdf,
        target_provenance=provenance_path,
        adapter_path=adapter_path,
        asset_dir=tmp_path,
        asset_stem="page7",
    )
    assert appended["status"] == "appended_page_7"
    assert len(PdfReader(str(target_pdf)).pages) == 7
    assert subject.legacy_page._page_content_hashes(target_pdf)[:6] == before

    all_attempts = {
        **partial_attempts,
        **_attempt_group(tmp_path, subject.REGIME_ORDER[3:]),
    }
    complete = subject.build_adapter(
        append_adapter_path=append_path,
        ra_attempts=all_attempts,
        output=adapter_path,
        prefix_cost_provider=_cost_provider,
    )
    assert set(partial["completed_regimes"]) < set(complete["completed_regimes"])
    replacement_assets = _page_asset(tmp_path, 900)
    monkeypatch.setattr(
        subject,
        "build_assets",
        lambda *_args, **_kwargs: replacement_assets,
    )
    replaced = subject.update_page7(
        target_pdf=target_pdf,
        target_provenance=provenance_path,
        adapter_path=adapter_path,
        asset_dir=tmp_path,
        asset_stem="page7",
    )
    assert replaced["status"] == "replaced_page_7"
    pages = PdfReader(str(target_pdf)).pages
    assert len(pages) == 7
    assert [float(page.mediabox.width) for page in pages[:6]] == [
        100,
        101,
        102,
        103,
        104,
        105,
    ]
    assert float(pages[6].mediabox.width) == 900
    assert subject.legacy_page._page_content_hashes(target_pdf)[:6] == before
    updated = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert updated["layout"]["page_count"] == 7
    assert updated["layout"]["page_7"] == subject.PAGE_ID
    assert updated[subject.REPORT_KEY]["pending_regimes"] == []


def test_adapter_rejects_completion_demotion(tmp_path: Path) -> None:
    append_path = _append_adapter(tmp_path / "append.json")
    adapter_path = tmp_path / "adapter.json"
    attempts = _attempt_group(tmp_path, subject.REGIME_ORDER)
    subject.build_adapter(
        append_adapter_path=append_path,
        ra_attempts=attempts,
        output=adapter_path,
        prefix_cost_provider=_cost_provider,
    )

    with pytest.raises(subject.Page7InputError, match="strict completion superset"):
        subject.build_adapter(
            append_adapter_path=append_path,
            ra_attempts={regime: attempts[regime] for regime in subject.REGIME_ORDER[:3]},
            output=adapter_path,
            prefix_cost_provider=_cost_provider,
        )
