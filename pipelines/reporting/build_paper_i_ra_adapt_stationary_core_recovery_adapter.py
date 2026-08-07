#!/usr/bin/env python3
"""Build the narrow non-evidentiary stationary-core recovery adapter.

This adapter has exactly two purposes:

* project three preserved, successfully completed factorial RA-always runs
  onto the scientifically identical base stationary-core cell identities; and
* retain three completed plateau trajectories whose immutable v7 worker
  wrapper exited at the sealed G5 non-vacuity gate because no interior
  plateau witness occurred.

The output is an evolving-report input, never a paper-evidence adoption
receipt.  Every source archive remains bound by hash and every projected cell
is rebuilt through the existing stationary-core report extractor so that its
0--50 trace arithmetic and terminal Qiskit cost are checked again.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting import (
    build_paper_i_ra_adapt_stationary_core_master_pdf as master,
)


CAMPAIGN_ROOT = (
    REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
)
FACTORIAL_PACKAGE = (
    CAMPAIGN_ROOT / "ra_always_factorial48_r50_20260730_v1_chtc"
)
ALWAYS_TARGET_PACKAGE = (
    CAMPAIGN_ROOT / "stationary_ra_always12_r50_20260729_v2_chtc"
)
PLATEAU_TARGET_PACKAGE = (
    CAMPAIGN_ROOT / "stationary_core_full48_r50_20260728_v7_chtc"
)
PRESERVATION_SNAPSHOT = CAMPAIGN_ROOT / (
    "paper_i_ra_adapt_completed_v1_preservation_"
    "9395481_9395482_20260730_v1.json"
)
PLATEAU_VALIDATION = REPO_ROOT / (
    "raw_outputs/paper_i_ra_adapt_stationary_core_v7_partial_report_"
    "20260729/fetched_validation_23_ra_always_quarantined.json"
)
PLATEAU_ARCHIVE_DIR = REPO_ROOT / (
    "raw_outputs/chtc_fetch_paper_i_ra_adapt_stationary_core_"
    "v7_9392883_20260729"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "raw_outputs/paper_i_ra_adapt_stationary_core_recovery_"
    "20260730/recovery_adapter.json"
)

ADAPTER_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_recovery_adapter_v1"
)
CROSS_CAMPAIGN_CLASS = (
    "cross_campaign_science_equivalent_passed_attempt_v1"
)
PLATEAU_G5_CLASS = (
    "completed_science_g5_plateau_domain_unexercised_v1"
)

FACTORIAL_CELLS = (
    (
        "core__weak_weak__nph3__ra_macro_always__"
        "gradient_stationary__phase1_cost_off",
        "core__weak_weak__nph3__ra_macro_always",
    ),
    (
        "core__weak_weak__nph3__ra_singleton_always__"
        "gradient_stationary__phase1_cost_off",
        "core__weak_weak__nph3__ra_singleton_always",
    ),
    (
        "core__intermediate_weak__nph3__ra_singleton_always__"
        "gradient_stationary__phase1_cost_off",
        "core__intermediate_weak__nph3__ra_singleton_always",
    ),
)
PLATEAU_CELLS = (
    "core__weak_strong__nph7__ra_macro_plateau",
    "core__weak_strong__nph7__ra_singleton_plateau",
    "core__intermediate_strong__nph7__ra_macro_plateau",
)
EXPECTED_PLATEAU_APPEND_TOTALS = {
    "core__weak_strong__nph7__ra_macro_plateau": 9028,
    "core__weak_strong__nph7__ra_singleton_plateau": 16024,
    "core__intermediate_strong__nph7__ra_macro_plateau": 9036,
}


class RecoveryAdapterError(ValueError):
    """Raised when a recovery projection is not fully authenticated."""


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(dict(payload))).hexdigest()


def _digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(payload))
    if "sha256" in result:
        raise RecoveryAdapterError("Cannot self-digest an already-digested row.")
    result["sha256"] = _canonical_sha256(result)
    return result


def _verify_self_digest(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> str:
    expected = payload.get("sha256")
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    observed = _canonical_sha256(unsigned)
    if expected != observed:
        raise RecoveryAdapterError(f"{label} self-digest drifted.")
    return observed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise RecoveryAdapterError(
            f"Source escaped the active repository: {path}"
        ) from exc


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecoveryAdapterError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise RecoveryAdapterError(f"{label} must be a JSON object.")
    return payload


def _verified_json(path: Path, *, label: str) -> dict[str, Any]:
    payload = _load_json(path, label=label)
    _verify_self_digest(payload, label=label)
    return payload


def _file_binding(
    path: Path,
    *,
    canonical_sha256: str | None = None,
) -> dict[str, Any]:
    binding: dict[str, Any] = {
        "path": _relative(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if canonical_sha256 is not None:
        binding["canonical_sha256"] = canonical_sha256
    return binding


def _verify_bound_file(
    path: Path,
    binding: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(binding.get("size_bytes", -1))
        or _sha256_file(path) != binding.get("sha256")
    ):
        raise RecoveryAdapterError(f"{label} file binding drifted.")


def _safe_member_name(name: str) -> str:
    pure = PurePosixPath(name)
    if (
        pure.is_absolute()
        or not pure.parts
        or "." in pure.parts
        or ".." in pure.parts
        or any(not part for part in pure.parts)
    ):
        raise RecoveryAdapterError(f"Unsafe archive member: {name}")
    return pure.as_posix()


def _read_member_bytes(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    label: str,
) -> bytes:
    if (
        not member.isfile()
        or member.issym()
        or member.islnk()
        or member.size < 0
    ):
        raise RecoveryAdapterError(f"{label} is not a safe regular member.")
    stream = archive.extractfile(member)
    if stream is None:
        raise RecoveryAdapterError(f"{label} is unreadable.")
    raw = stream.read()
    if len(raw) != member.size:
        raise RecoveryAdapterError(f"{label} size drifted while reading.")
    return raw


def _json_from_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecoveryAdapterError(f"{label} is invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise RecoveryAdapterError(f"{label} must be a JSON object.")
    return payload


def _extract_factorial_result(
    archive_path: Path,
) -> tuple[dict[str, Any], str, int]:
    """Read only the early result member from a fully preserved archive."""

    target = "worker_outputs/result.json"
    seen: set[str] = set()
    found: bytes | None = None
    try:
        with tarfile.open(archive_path, mode="r|gz") as archive:
            for member in archive:
                name = _safe_member_name(member.name)
                if name in seen:
                    raise RecoveryAdapterError(
                        f"Duplicate factorial archive member: {name}"
                    )
                seen.add(name)
                if name == target:
                    found = _read_member_bytes(
                        archive,
                        member,
                        label="factorial result",
                    )
                    break
                if not (member.isfile() or member.isdir()):
                    raise RecoveryAdapterError(
                        f"Unsafe factorial archive member type: {name}"
                    )
    except (OSError, tarfile.TarError) as exc:
        raise RecoveryAdapterError(
            f"Factorial archive is unreadable: {archive_path}"
        ) from exc
    if found is None:
        raise RecoveryAdapterError("Factorial archive has no result member.")
    return (
        _json_from_bytes(found, label="factorial result"),
        hashlib.sha256(found).hexdigest(),
        len(found),
    )


def _extract_plateau_payloads(
    archive_path: Path,
    *,
    execution_id: str,
) -> dict[str, tuple[Any, str, int]]:
    job_member = (
        "chtc/paper_i_ra_adapt_repair_20260727/"
        "stationary_core_full48_r50_20260728_v7_chtc/jobs/"
        f"{execution_id}.json"
    )
    wanted = {
        "manifest": "worker_outputs/execution_manifest.json",
        "result": "worker_outputs/result.json",
        "summary": "worker_outputs/summary.json",
        "exit": "worker_outputs/worker_exit_status.txt",
        "job": job_member,
    }
    by_name = {name: role for role, name in wanted.items()}
    selected: dict[str, tuple[Any, str, int]] = {}
    seen: set[str] = set()
    try:
        with tarfile.open(archive_path, mode="r|gz") as archive:
            for member in archive:
                name = _safe_member_name(member.name)
                if name in seen:
                    raise RecoveryAdapterError(
                        f"{execution_id}: duplicate archive member {name}"
                    )
                seen.add(name)
                if name not in by_name:
                    if not (member.isfile() or member.isdir()):
                        raise RecoveryAdapterError(
                            f"{execution_id}: unsafe member type {name}"
                        )
                    continue
                role = by_name[name]
                raw = _read_member_bytes(
                    archive,
                    member,
                    label=f"{execution_id} {role}",
                )
                digest = hashlib.sha256(raw).hexdigest()
                value: Any
                if role == "exit":
                    try:
                        value = int(raw.decode("ascii").strip())
                    except (UnicodeDecodeError, ValueError) as exc:
                        raise RecoveryAdapterError(
                            f"{execution_id}: worker exit is malformed."
                        ) from exc
                else:
                    value = _json_from_bytes(
                        raw, label=f"{execution_id} {role}"
                    )
                selected[role] = (value, digest, len(raw))
    except (OSError, tarfile.TarError) as exc:
        raise RecoveryAdapterError(
            f"{execution_id}: plateau archive is unreadable."
        ) from exc
    if set(selected) != set(wanted):
        missing = sorted(set(wanted) - set(selected))
        raise RecoveryAdapterError(
            f"{execution_id}: selected archive members missing: {missing}"
        )
    return selected


def _protocol_for_job(
    job: Mapping[str, Any],
    *,
    label: str,
) -> tuple[dict[str, Any], Path]:
    binding = job.get("protocol")
    if not isinstance(binding, Mapping):
        raise RecoveryAdapterError(f"{label} has no protocol binding.")
    pure = PurePosixPath(str(binding.get("path", "")))
    if (
        pure.is_absolute()
        or "." in pure.parts
        or ".." in pure.parts
        or not pure.parts
    ):
        raise RecoveryAdapterError(f"{label} protocol path is unsafe.")
    path = REPO_ROOT.joinpath(*pure.parts)
    payload = _verified_json(path, label=f"{label} protocol")
    if (
        path.stat().st_size != int(binding.get("size_bytes", -1))
        or _sha256_file(path) != binding.get("sha256")
        or payload["sha256"] != binding.get("canonical_sha256")
    ):
        raise RecoveryAdapterError(f"{label} protocol binding drifted.")
    return payload, path


def _scientific_protocol_projection(
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove only materialization/provenance/path identity fields."""

    projected = copy.deepcopy(dict(protocol))
    projected.pop("sha256", None)
    projected.pop("bundle_id", None)
    projected.pop("bundle_manifest_sha256", None)
    projected.pop("bundle_materialization", None)
    projected.pop("source_locks", None)

    baseline = projected.get("baseline_consumption")
    if not isinstance(baseline, dict):
        raise RecoveryAdapterError(
            "Protocol baseline-consumption projection is unavailable."
        )
    baseline.pop("sha256", None)
    baseline.pop("source_lock_sha256", None)
    baseline.pop("settled_change_ids", None)

    request = projected.get("request")
    observation = (
        request.get("observation")
        if isinstance(request, dict)
        else None
    )
    if not isinstance(observation, dict):
        raise RecoveryAdapterError(
            "Protocol observation projection is unavailable."
        )
    for role in ("checkpoint", "estimator_ledger"):
        value = observation.get(role)
        if not isinstance(value, dict) or "path" not in value:
            raise RecoveryAdapterError(
                f"Protocol observation {role} path is unavailable."
            )
        value["path"] = f"<normalized-{role}-output-path>"
    return projected


def _assert_cross_campaign_equivalence(
    *,
    source_job: Mapping[str, Any],
    source_protocol: Mapping[str, Any],
    target_job: Mapping[str, Any],
    target_protocol: Mapping[str, Any],
    source_execution_id: str,
    target_execution_id: str,
) -> dict[str, Any]:
    common_axis_keys = (
        "route_id",
        "regime_id",
        "nph",
        "candidate_representation",
        "execution_entrypoint",
        "horizon",
    )
    source_axes = {key: source_job.get(key) for key in common_axis_keys}
    target_axes = {key: target_job.get(key) for key in common_axis_keys}
    if (
        source_job.get("execution_id") != source_execution_id
        or target_job.get("execution_id") != target_execution_id
        or source_axes != target_axes
        or source_job.get("active_gradient_policy")
        != target_protocol.get("active_gradient_policy")
        or source_job.get("resource_weighting_scope")
        != target_protocol.get("resource_weighting_scope")
    ):
        raise RecoveryAdapterError(
            f"{source_execution_id}: source/target job axes differ."
        )

    source_baseline = source_protocol.get("baseline_consumption")
    target_baseline = target_protocol.get("baseline_consumption")
    if not isinstance(source_baseline, Mapping) or not isinstance(
        target_baseline, Mapping
    ):
        raise RecoveryAdapterError(
            f"{source_execution_id}: baseline provenance is unavailable."
        )
    source_change_ids = list(
        source_baseline.get("settled_change_ids", ())
    )
    target_change_ids = list(
        target_baseline.get("settled_change_ids", ())
    )
    if (
        any(not isinstance(value, str) for value in source_change_ids)
        or any(not isinstance(value, str) for value in target_change_ids)
        or (
            set(source_change_ids) ^ set(target_change_ids)
        )
        - {"D5"}
    ):
        raise RecoveryAdapterError(
            f"{source_execution_id}: settled-change provenance differs "
            "beyond D5."
        )

    source_projection = _scientific_protocol_projection(source_protocol)
    target_projection = _scientific_protocol_projection(target_protocol)
    if source_projection != target_projection:
        raise RecoveryAdapterError(
            f"{source_execution_id}: source/target scientific protocols differ."
        )
    route = source_protocol.get("route_contract")
    target_route = target_protocol.get("route_contract")
    if (
        not isinstance(route, Mapping)
        or not isinstance(target_route, Mapping)
        or route != target_route
        or source_protocol.get("problem") != target_protocol.get("problem")
        or source_protocol.get("active_gradient_policy")
        != "stationary_source_response_v1"
        or source_protocol.get("resource_weighting_scope")
        != "late_resource_weighting_v1"
    ):
        raise RecoveryAdapterError(
            f"{source_execution_id}: core stationary-route identity drifted."
        )
    semantic = route.get("semantic_invariants")
    if (
        not isinstance(semantic, Mapping)
        or semantic.get("phase1_phase2_lambda_f_proxy_active") is not False
        or semantic.get("resource_weighting_scope")
        != "late_resource_weighting_v1"
    ):
        raise RecoveryAdapterError(
            f"{source_execution_id}: Phase-I cost-off semantics drifted."
        )
    return {
        "status": "passed",
        "comparison": (
            "exact_after_materialization_provenance_and_output_path_"
            "normalization_v1"
        ),
        "scientific_projection_sha256": _canonical_sha256(
            source_projection
        ),
        "route_contract_sha256": route.get("sha256"),
        "problem_request_sha256": (
            source_protocol.get("problem", {}).get(
                "problem_request_sha256"
            )
        ),
        "active_gradient_policy": "stationary_source_response_v1",
        "resource_weighting_scope": "late_resource_weighting_v1",
        "phase1_resource_weighting_active": False,
        "settled_change_id_provenance_normalization": {
            "source_settled_change_ids": source_change_ids,
            "target_settled_change_ids": target_change_ids,
            "source_only": sorted(
                set(source_change_ids) - set(target_change_ids)
            ),
            "target_only": sorted(
                set(target_change_ids) - set(source_change_ids)
            ),
            "D5_meaning": "late_resource_weighting_v1",
            "D5_resolved_semantics_match": (
                source_protocol.get("resource_weighting_scope")
                == target_protocol.get("resource_weighting_scope")
                == "late_resource_weighting_v1"
                and semantic.get("resource_weighting_scope")
                == "late_resource_weighting_v1"
            ),
            "disposition": (
                "provenance_only_normalized_after_resolved_semantics_match"
            ),
        },
        "excluded_difference_classes": [
            "bundle_and_materialization_identity",
            "source_lock_and_provenance_digests",
            "settled_change_id_D5_provenance_only",
            "observation_output_paths",
            "top_level_self_digest",
        ],
    }


def _exact_energy_and_closure(
    summary: Mapping[str, Any],
) -> tuple[float, dict[str, Any]]:
    provenance = summary.get("provenance")
    work = summary.get("canonical_all_work")
    if not isinstance(provenance, Mapping) or not isinstance(work, Mapping):
        raise RecoveryAdapterError(
            "Run summary lacks exact-energy or all-work closure."
        )
    try:
        exact_energy = float(provenance["exact_same_cutoff_energy"])
        s_alg = int(work["s_alg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RecoveryAdapterError(
            "Run summary exact-energy/all-work values are malformed."
        ) from exc
    return exact_energy, {
        "gates": {
            "G10": {
                "evidence": {
                    "S_alg": s_alg,
                }
            }
        }
    }


def _extract_report_cell(
    *,
    execution_id: str,
    target_job: Mapping[str, Any],
    result: Mapping[str, Any],
    summary: Mapping[str, Any],
    terminal_status: str,
) -> dict[str, Any]:
    exact_energy, closure = _exact_energy_and_closure(summary)
    try:
        cell = master._extract_ra_cell(
            execution_id=execution_id,
            job=target_job,
            result=result,
            summary=summary,
            closure=closure,
            exact_energy=exact_energy,
            compiler=None,
        )
    except Exception as exc:
        raise RecoveryAdapterError(
            f"{execution_id}: report cell extraction failed: {exc}"
        ) from exc
    if (
        len(cell.get("points", ())) != 51
        or cell.get("terminal", {}).get("k") != 50
    ):
        raise RecoveryAdapterError(
            f"{execution_id}: report projection is not a complete 0--50 cell."
        )
    cell["terminal"]["status"] = terminal_status
    return cell


def _factorial_rows(
    preservation: Mapping[str, Any],
    preservation_binding: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_rows = preservation.get("rows")
    if not isinstance(raw_rows, list):
        raise RecoveryAdapterError("Preservation snapshot rows are unavailable.")
    by_execution = {
        str(row.get("execution_id")): row
        for row in raw_rows
        if isinstance(row, Mapping)
        and row.get("campaign_key") == "factorial"
    }
    rows: list[dict[str, Any]] = []
    for source_execution_id, target_execution_id in FACTORIAL_CELLS:
        preserved = by_execution.get(source_execution_id)
        if not isinstance(preserved, Mapping):
            raise RecoveryAdapterError(
                f"{source_execution_id}: preservation row is unavailable."
            )
        _verify_self_digest(
            preserved, label=f"{source_execution_id} preservation row"
        )
        archive_binding = preserved.get("archive")
        verification = preserved.get("local_verification")
        if (
            not isinstance(archive_binding, Mapping)
            or not isinstance(verification, Mapping)
            or preserved.get("status") != "passed"
            or verification.get("status") != "passed"
            or verification.get("authority_bindings_passed") is not True
            or verification.get("gzip_and_full_tar_scan_passed") is not True
            or verification.get("regular_member_closure_passed") is not True
            or int(verification.get("worker_exit_status", -1)) != 0
        ):
            raise RecoveryAdapterError(
                f"{source_execution_id}: preservation closure failed."
            )
        archive_path = REPO_ROOT / str(archive_binding.get("path", ""))
        _verify_bound_file(
            archive_path,
            archive_binding,
            label=f"{source_execution_id} preserved archive",
        )

        source_job_path = (
            FACTORIAL_PACKAGE / "jobs" / f"{source_execution_id}.json"
        )
        target_job_path = (
            ALWAYS_TARGET_PACKAGE / "jobs" / f"{target_execution_id}.json"
        )
        source_job = _verified_json(
            source_job_path, label=f"{source_execution_id} source job"
        )
        target_job = _verified_json(
            target_job_path, label=f"{target_execution_id} target job"
        )
        source_protocol, source_protocol_path = _protocol_for_job(
            source_job, label=f"{source_execution_id} source job"
        )
        target_protocol, target_protocol_path = _protocol_for_job(
            target_job, label=f"{target_execution_id} target job"
        )
        equivalence = _assert_cross_campaign_equivalence(
            source_job=source_job,
            source_protocol=source_protocol,
            target_job=target_job,
            target_protocol=target_protocol,
            source_execution_id=source_execution_id,
            target_execution_id=target_execution_id,
        )

        result, result_sha256, result_size = _extract_factorial_result(
            archive_path
        )
        if result.get("protocol") != source_protocol:
            raise RecoveryAdapterError(
                f"{source_execution_id}: result/source protocol equality failed."
            )
        run = result.get("run")
        if (
            result.get("schema") != "paper_i_ra_adapt_result_v1"
            or not isinstance(run, Mapping)
            or run.get("problem") != source_protocol.get("problem")
            or run.get("route", {}).get("contract_sha256")
            != source_protocol.get("route_contract", {}).get("sha256")
        ):
            raise RecoveryAdapterError(
                f"{source_execution_id}: result core protocol binding failed."
            )
        embedded = run.get("paper_i_summary")
        if not isinstance(embedded, Mapping):
            raise RecoveryAdapterError(
                f"{source_execution_id}: embedded summary is unavailable."
            )
        summary = copy.deepcopy(dict(embedded))
        summary["schema"] = "paper_i_run_summary_v1"
        if (
            int(summary.get("available_controller_rounds", -1)) != 50
            or len(summary.get("accepted_error_trace", ())) != 50
        ):
            raise RecoveryAdapterError(
                f"{source_execution_id}: embedded 50-round summary drifted."
            )
        cell = _extract_report_cell(
            execution_id=target_execution_id,
            target_job=target_job,
            result=result,
            summary=summary,
            terminal_status="complete-Xrev",
        )
        row = _digested(
            {
                "schema": (
                    "paper_i_ra_adapt_stationary_core_recovered_cell_v1"
                ),
                "target_execution_id": target_execution_id,
                "source_execution_id": source_execution_id,
                "recovery_class": CROSS_CAMPAIGN_CLASS,
                "paper_evidence_eligible": False,
                "source": {
                    "package_id": source_job["package_id"],
                    "archive": dict(archive_binding),
                    "result": {
                        "member": "worker_outputs/result.json",
                        "sha256": result_sha256,
                        "size_bytes": result_size,
                    },
                    "attempt_status": "passed",
                    "worker_exit_status": 0,
                    "worker_attempt_receipt_sha256": verification[
                        "worker_attempt_receipt_sha256"
                    ],
                    "preservation_snapshot": dict(preservation_binding),
                    "preservation_row_sha256": preserved["sha256"],
                    "source_job": _file_binding(
                        source_job_path,
                        canonical_sha256=source_job["sha256"],
                    ),
                    "source_protocol": _file_binding(
                        source_protocol_path,
                        canonical_sha256=source_protocol["sha256"],
                    ),
                    "target_job": _file_binding(
                        target_job_path,
                        canonical_sha256=target_job["sha256"],
                    ),
                    "target_protocol": _file_binding(
                        target_protocol_path,
                        canonical_sha256=target_protocol["sha256"],
                    ),
                    "summary_projection": (
                        "embedded_run_paper_i_summary_plus_schema_v1"
                    ),
                },
                "qualification": {
                    "science_equivalence_status": "passed",
                    "full_controller_rounds": 50,
                    "source_result_protocol_equals_source_protocol": True,
                    "source_result_problem_equals_source_protocol_problem": True,
                    "source_result_route_contract_equals_source_protocol": True,
                    "protocol_equivalence": equivalence,
                },
                "cell": cell,
            }
        )
        rows.append(row)
    return rows


def _plateau_rows(
    validation: Mapping[str, Any],
    validation_binding: Mapping[str, Any],
) -> list[dict[str, Any]]:
    attempts = validation.get("attempts")
    if not isinstance(attempts, list):
        raise RecoveryAdapterError("Plateau validation attempts are unavailable.")
    by_execution = {
        str(row.get("execution_id")): row
        for row in attempts
        if isinstance(row, Mapping)
    }
    rows: list[dict[str, Any]] = []
    for execution_id in PLATEAU_CELLS:
        attempt = by_execution.get(execution_id)
        if (
            not isinstance(attempt, Mapping)
            or attempt.get("status") != "failed_attempt_retained"
            or int(attempt.get("worker_exit_status", -1)) != 2
            or attempt.get("worker_receipt_sha256") is not None
        ):
            raise RecoveryAdapterError(
                f"{execution_id}: expected retained exit-2 attempt is absent."
            )
        archive_path = PLATEAU_ARCHIVE_DIR / str(attempt.get("path", ""))
        _verify_bound_file(
            archive_path,
            attempt,
            label=f"{execution_id} validation-bound archive",
        )
        loaded = _extract_plateau_payloads(
            archive_path, execution_id=execution_id
        )
        manifest, manifest_sha256, manifest_size = loaded["manifest"]
        result, result_sha256, result_size = loaded["result"]
        summary, summary_sha256, summary_size = loaded["summary"]
        worker_exit, worker_exit_sha256, worker_exit_size = loaded["exit"]
        archived_job, archived_job_sha256, archived_job_size = loaded["job"]

        target_job_path = (
            PLATEAU_TARGET_PACKAGE / "jobs" / f"{execution_id}.json"
        )
        target_job = _verified_json(
            target_job_path, label=f"{execution_id} target job"
        )
        target_protocol, target_protocol_path = _protocol_for_job(
            target_job, label=f"{execution_id} target job"
        )
        if archived_job != target_job:
            raise RecoveryAdapterError(
                f"{execution_id}: archived job differs from exact target job."
            )
        _verify_self_digest(
            archived_job, label=f"{execution_id} archived target job"
        )
        _verify_self_digest(
            manifest, label=f"{execution_id} execution manifest"
        )
        payloads = manifest.get("output_payloads")
        if (
            manifest.get("status") != "passed"
            or manifest.get("paper_facing_result_allowed") is not True
            or manifest.get("execution_id") != execution_id
            or manifest.get("package_id") != target_job.get("package_id")
            or manifest.get("job_spec_sha256") != target_job.get("sha256")
            or manifest.get("protocol_sha256") != target_protocol["sha256"]
            or manifest.get("maximum_controller_rounds_override") is not None
            or not isinstance(payloads, Mapping)
            or payloads.get("result")
            != {"sha256": result_sha256, "size_bytes": result_size}
            or payloads.get("summary")
            != {"sha256": summary_sha256, "size_bytes": summary_size}
            or worker_exit != 2
        ):
            raise RecoveryAdapterError(
                f"{execution_id}: manifest/result/exit closure failed."
            )
        if (
            result.get("protocol") != target_protocol
            or result.get("schema") != "paper_i_ra_adapt_result_v1"
            or summary.get("schema") != "paper_i_run_summary_v1"
            or int(summary.get("available_controller_rounds", -1)) != 50
            or len(summary.get("accepted_error_trace", ())) != 50
        ):
            raise RecoveryAdapterError(
                f"{execution_id}: completed scientific payload drifted."
            )
        run = result.get("run")
        scientific = result.get("scientific_receipts")
        accepted = (
            scientific.get("accepted_round_receipts")
            if isinstance(scientific, Mapping)
            else None
        )
        if (
            not isinstance(run, Mapping)
            or not isinstance(accepted, list)
            or len(accepted) != 50
            or int(run.get("stop", {}).get(
                "completed_controller_rounds", -1
            ))
            != 50
        ):
            raise RecoveryAdapterError(
                f"{execution_id}: accepted-round closure is not 50."
            )
        interior = 0
        appended = 0
        population_sha256s: list[str] = []
        for ordinal, raw in enumerate(accepted, start=1):
            if not isinstance(raw, Mapping):
                raise RecoveryAdapterError(
                    f"{execution_id}: accepted receipt {ordinal} is malformed."
                )
            population = raw.get("scored_insertion_position_population")
            if (
                raw.get("accepted_round_ordinal") != ordinal
                or not isinstance(population, Mapping)
            ):
                raise RecoveryAdapterError(
                    f"{execution_id}: accepted receipt order drifted."
                )
            _verify_self_digest(
                population,
                label=f"{execution_id} scored population {ordinal}",
            )
            phases = population.get("phases")
            append_position = int(population.get("append_position", -1))
            if (
                population.get("schema")
                != "paper_i_scored_insertion_position_population_v1"
                or population.get("coordinate_chart")
                != "exact_ordered_insertion_zero_angle_v1"
                or population.get("phase_order")
                != ["phase_i", "phase_ii", "phase_iii"]
                or not isinstance(phases, list)
                or len(phases) != 3
                or append_position < 0
            ):
                raise RecoveryAdapterError(
                    f"{execution_id}: scored population schema drifted."
                )
            observed_interior = 0
            observed_append = 0
            observed_records = 0
            for phase_name, phase in zip(
                ("phase_i", "phase_ii", "phase_iii"),
                phases,
                strict=True,
            ):
                records = (
                    phase.get("records")
                    if isinstance(phase, Mapping)
                    else None
                )
                if (
                    not isinstance(phase, Mapping)
                    or phase.get("phase") != phase_name
                    or not isinstance(records, list)
                    or len(records) != int(
                        phase.get("population_count", -1)
                    )
                    or phase.get("ordered_population_sha256")
                    != hashlib.sha256(
                        _canonical_json_bytes(records)
                    ).hexdigest()
                ):
                    raise RecoveryAdapterError(
                        f"{execution_id}: {phase_name} population drifted."
                    )
                for record in records:
                    if not isinstance(record, Mapping):
                        raise RecoveryAdapterError(
                            f"{execution_id}: scored record is malformed."
                        )
                    position = int(record.get("insertion_position", -1))
                    expected_class = (
                        "interior"
                        if position < append_position
                        else "append"
                    )
                    if (
                        position < 0
                        or position > append_position
                        or record.get("position_class") != expected_class
                    ):
                        raise RecoveryAdapterError(
                            f"{execution_id}: scored position class drifted."
                        )
                    observed_records += 1
                    observed_interior += int(expected_class == "interior")
                    observed_append += int(expected_class == "append")
            if (
                observed_records
                != int(population.get("scored_record_count", -1))
                or observed_interior
                != int(population.get("interior_scored_count", -1))
                or observed_append
                != int(population.get("append_scored_count", -1))
            ):
                raise RecoveryAdapterError(
                    f"{execution_id}: scored-position totals drifted."
                )
            interior += observed_interior
            appended += observed_append
            population_sha256s.append(str(population["sha256"]))
        if (
            interior != 0
            or appended != EXPECTED_PLATEAU_APPEND_TOTALS[execution_id]
        ):
            raise RecoveryAdapterError(
                f"{execution_id}: G5 route-domain totals disagree."
            )

        cell = _extract_report_cell(
            execution_id=execution_id,
            target_job=target_job,
            result=result,
            summary=summary,
            terminal_status="complete-G5*",
        )
        row = _digested(
            {
                "schema": (
                    "paper_i_ra_adapt_stationary_core_recovered_cell_v1"
                ),
                "target_execution_id": execution_id,
                "source_execution_id": execution_id,
                "recovery_class": PLATEAU_G5_CLASS,
                "paper_evidence_eligible": False,
                "source": {
                    "package_id": target_job["package_id"],
                    "archive": {
                        "path": _relative(archive_path),
                        "sha256": attempt["sha256"],
                        "size_bytes": attempt["size_bytes"],
                    },
                    "result": {
                        "member": "worker_outputs/result.json",
                        "sha256": result_sha256,
                        "size_bytes": result_size,
                    },
                    "summary": {
                        "member": "worker_outputs/summary.json",
                        "sha256": summary_sha256,
                        "size_bytes": summary_size,
                    },
                    "execution_manifest": {
                        "member": "worker_outputs/execution_manifest.json",
                        "canonical_sha256": manifest["sha256"],
                        "sha256": manifest_sha256,
                        "size_bytes": manifest_size,
                    },
                    "archived_target_job": {
                        "member": (
                            "chtc/paper_i_ra_adapt_repair_20260727/"
                            "stationary_core_full48_r50_20260728_v7_chtc/"
                            f"jobs/{execution_id}.json"
                        ),
                        "canonical_sha256": archived_job["sha256"],
                        "sha256": archived_job_sha256,
                        "size_bytes": archived_job_size,
                    },
                    "target_job": _file_binding(
                        target_job_path,
                        canonical_sha256=target_job["sha256"],
                    ),
                    "target_protocol": _file_binding(
                        target_protocol_path,
                        canonical_sha256=target_protocol["sha256"],
                    ),
                    "attempt_status": "failed_attempt_retained",
                    "worker_exit_status": worker_exit,
                    "worker_exit_member": {
                        "member": "worker_outputs/worker_exit_status.txt",
                        "sha256": worker_exit_sha256,
                        "size_bytes": worker_exit_size,
                    },
                    "validation_receipt": dict(validation_binding),
                    "validation_attempt_ordinal": attempt[
                        "attempt_ordinal"
                    ],
                    "cluster_id": attempt["cluster_id"],
                    "proc_id": attempt["proc_id"],
                },
                "qualification": {
                    "science_equivalence_status": "not_applicable_exact_job",
                    "route_domain_status": "unexercised",
                    "interior_scored_count": interior,
                    "append_scored_count": appended,
                    "full_controller_rounds": 50,
                    "execution_manifest_status": "passed",
                    "other_g1_g13_gates_status": "passed",
                    "failed_gate": "G5",
                    "failed_check": (
                        "plateau_route_requires_at_least_one_interior_"
                        "scored_position"
                    ),
                    "population_receipt_sha256s": population_sha256s,
                    "recovery_disposition": (
                        "completed_science_retained_as_non_evidentiary_"
                        "diagnostic_never_reclassified_as_passed_attempt"
                    ),
                },
                "cell": cell,
            }
        )
        rows.append(row)
    return rows


def build_adapter(output: Path) -> dict[str, Any]:
    preservation = _verified_json(
        PRESERVATION_SNAPSHOT, label="factorial preservation snapshot"
    )
    validation = _verified_json(
        PLATEAU_VALIDATION, label="plateau fetched validation receipt"
    )
    if (
        preservation.get("schema")
        != "paper_i_ra_adapt_completed_v1_preservation_snapshot_v1"
        or preservation.get("status") != "passed"
        or preservation.get(
            "all_archive_size_sha256_gzip_tar_authority_checks_passed"
        )
        is not True
        or validation.get("schema")
        != "paper_i_ra_adapt_stationary_core_fetched_validation_v1"
        or validation.get("status") != "validated_no_selection"
        or validation.get("paper_evidence_adopted") is not False
    ):
        raise RecoveryAdapterError(
            "Recovery source receipts do not have the required disposition."
        )
    preservation_binding = _file_binding(
        PRESERVATION_SNAPSHOT,
        canonical_sha256=preservation["sha256"],
    )
    validation_binding = _file_binding(
        PLATEAU_VALIDATION,
        canonical_sha256=validation["sha256"],
    )
    cells = [
        *_factorial_rows(preservation, preservation_binding),
        *_plateau_rows(validation, validation_binding),
    ]
    expected_targets = {
        target for _source, target in FACTORIAL_CELLS
    } | set(PLATEAU_CELLS)
    observed_targets = {
        str(row.get("target_execution_id")) for row in cells
    }
    if len(cells) != 6 or observed_targets != expected_targets:
        raise RecoveryAdapterError(
            "Recovery adapter is not the exact intended six-cell set."
        )
    adapter = _digested(
        {
            "schema": ADAPTER_SCHEMA,
            "status": "passed",
            "not_paper_evidence": True,
            "paper_evidence_adopted": False,
            "scope": (
                "evolving_stationary_core_report_recovery_only_v1"
            ),
            "cell_count": 6,
            "recovery_counts": {
                CROSS_CAMPAIGN_CLASS: 3,
                PLATEAU_G5_CLASS: 3,
            },
            "target_execution_ids": sorted(expected_targets),
            "cells": cells,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RecoveryAdapterError(
            f"Refusing stale output temporary: {temporary}"
        )
    raw = _canonical_json_bytes(adapter) + b"\n"
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return adapter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        output.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise RecoveryAdapterError(
            "Recovery adapter output must remain inside the active repository."
        ) from exc
    adapter = build_adapter(output)
    print(
        json.dumps(
            {
                "status": "passed",
                "output": _relative(output),
                "sha256": adapter["sha256"],
                "cell_count": adapter["cell_count"],
                "target_execution_ids": adapter[
                    "target_execution_ids"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
