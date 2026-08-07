#!/usr/bin/env python3
"""Fail-closed contract for the inert stationary-core RA r70 package.

The package is a diagnostic continuation plan.  It contains 27 authenticated
round-50 resume inputs and nine fresh always-insertion plans.  The latter are
intentionally blocked while the exact round-50 predecessor jobs in cluster
9397758 remain live.  Nothing in this module authorizes execution or contacts
HTCondor.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


sys.dont_write_bytecode = True

PACKAGE_ID = (
    "paper_i_ra_adapt_stationary_core_ra36_r70_"
    "continuation_20260731_v1_chtc"
)
PACKAGE_RELATIVE_ROOT = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    f"{PACKAGE_ID.removeprefix('paper_i_ra_adapt_')}"
)
CAMPAIGN_ID = (
    "paper_i_ra_adapt_stationary_core_r70_continuation_v1"
)
RUN_CLASS = "diagnostic"
EXECUTION_TARGET = "chtc"
SOURCE_HORIZON = 50
TARGET_HORIZON = 70
RESUME_COUNT = 27
FRESH_COUNT = 9
CELL_COUNT = RESUME_COUNT + FRESH_COUNT

ACTIVE_GRADIENT_POLICY = "stationary_source_response_v1"
RESOURCE_WEIGHTING_SCOPE = "late_resource_weighting_v1"
OPTIMIZER = "powell"
OPTIMIZER_MAXITER = 200
SEED = 7

SOURCE_REPORT_RELATIVE = (
    "output/pdf/"
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving/"
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_"
    "evolving_partial_progress_provenance.json"
)
COLLISION_QUEUE_RELATIVE = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "ra_always_factorial48_r50_20260730_v2_chtc_activation_"
    "release_v2/queue.tsv"
)
COLLISION_SUBMISSION_RECEIPT_RELATIVE = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "ra_always_factorial48_r50_20260730_v2_chtc_activation_"
    "release_v2_submission_receipt.json"
)
COLLISION_STATE_SNAPSHOT_RELATIVE = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "v2_submission_evidence_20260730T1830Z/factorial_current.json"
)
COLLISION_CLUSTER_ID = 9397758
COLLISION_PROC_IDS = tuple(range(9))

CORE_SOURCE_PACKAGE_RELATIVE = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "stationary_core_full48_r50_20260728_v8_chtc"
)
ALWAYS_V1_SOURCE_PACKAGE_RELATIVE = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "ra_always_factorial48_r50_20260730_v1_chtc"
)
ALWAYS_V2_SOURCE_PACKAGE_RELATIVE = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "ra_always_factorial48_r50_20260730_v2_chtc"
)
CORE_PROTOCOL_ROOT_RELATIVE = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "bundles/materializations/ra_adapt_stationary_late_core_v11/"
    "ra_repair_stationary_late_core_v1/protocols"
)
ALWAYS_PROTOCOL_ROOT_RELATIVE = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "bundles/materializations/ra_adapt_always_factorial48_v1/"
    "ra_repair_always_factorial_stationary_late_v1/protocols"
)
ALWAYS_SOURCE_SUFFIX = (
    "__gradient_stationary__phase1_cost_off"
)

SOURCE_FAMILIES = {
    "stationary_core_v11": {
        "source_package_root": CORE_SOURCE_PACKAGE_RELATIVE,
        "packaged_root": "source_archives/stationary_core_v11",
    },
    "always_factorial_v1": {
        "source_package_root": ALWAYS_V1_SOURCE_PACKAGE_RELATIVE,
        "packaged_root": "source_archives/always_factorial_v1",
    },
    "always_factorial_v2": {
        "source_package_root": ALWAYS_V2_SOURCE_PACKAGE_RELATIVE,
        "packaged_root": "source_archives/always_factorial_v2",
    },
}

REGIME_CUTOFF_PAIRS = (
    ("weak_weak", 3),
    ("intermediate_weak", 3),
    ("strong_weak_u8", 3),
    ("weak_strong", 7),
    ("intermediate_strong", 7),
    ("strong_strong_u8", 7),
)
ROUTE_IDS = (
    "ra_macro_append_only",
    "ra_macro_plateau",
    "ra_macro_always",
    "ra_singleton_append_only",
    "ra_singleton_plateau",
    "ra_singleton_always",
)
INSERTION_KIND_BY_ROUTE = {
    "ra_macro_append_only": "append_only",
    "ra_macro_plateau": "plateau_commutation",
    "ra_macro_always": "always_commutation_reduced",
    "ra_singleton_append_only": "append_only",
    "ra_singleton_plateau": "plateau_commutation",
    "ra_singleton_always": "always_commutation_reduced",
}
RESOURCE_ENVELOPES = {
    ("macro_generator_v1", 3): (4, 49_152, 61_440),
    ("single_pauli_word_v1", 3): (4, 57_344, 61_440),
    ("macro_generator_v1", 7): (4, 65_536, 81_920),
    ("single_pauli_word_v1", 7): (4, 90_112, 98_304),
}
MAX_RUNTIME_SECONDS = 72 * 60 * 60

PACKAGE_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_r70_continuation_package_v1"
)
EXECUTION_PLAN_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_r70_continuation_plan_v1"
)
SOURCE_ARCHIVES_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_r70_source_archives_v1"
)
RESUME_INPUTS_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_r70_resume_inputs_v1"
)
COLLISION_STATUS_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_r70_collision_status_v1"
)
JOB_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_r70_job_v1"
)
AUTHORIZATION_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_r70_execution_authorization_v1"
)

SOURCE_ARCHIVES_NAME = "source_archives.json"
RESUME_INPUTS_NAME = "resume_inputs_manifest.json"
SOURCE_LOCK_AUDIT_NAME = "source_lock_audit.json"
COLLISION_STATUS_NAME = "collision_status.json"
EXECUTION_PLAN_NAME = "execution_plan.json"
QUEUE_NAME = "queue.tsv"
PACKAGE_MANIFEST_NAME = "package_manifest.json"

CONTROL_FILES = (
    "package_contract.py",
    "build_package.py",
    "validate_package.py",
    "run_cell.py",
)
GENERATED_FILES = (
    SOURCE_ARCHIVES_NAME,
    RESUME_INPUTS_NAME,
    SOURCE_LOCK_AUDIT_NAME,
    COLLISION_STATUS_NAME,
    EXECUTION_PLAN_NAME,
    QUEUE_NAME,
    PACKAGE_MANIFEST_NAME,
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PackageContractError(ValueError):
    """Raised when source, resume, collision, or package closure drifts."""


def repo_root_from_script(script_path: str | Path) -> Path:
    return Path(script_path).resolve().parents[3]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value.pop("sha256", None)
    value["sha256"] = canonical_sha256(value)
    return value


def require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise PackageContractError(f"{label} is not a lowercase SHA-256.")
    return value


def load_json(path: str | Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        raise PackageContractError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackageContractError(f"{label} is not valid JSON.") from exc
    if not isinstance(value, dict):
        raise PackageContractError(f"{label} must be a JSON object.")
    return value


def verify_self_digest(
    payload: Mapping[str, Any], *, label: str
) -> str:
    unsigned = dict(payload)
    observed = unsigned.pop("sha256", None)
    require_sha256(observed, label=f"{label} digest")
    if canonical_sha256(unsigned) != observed:
        raise PackageContractError(f"{label} self digest drifted.")
    return str(observed)


def safe_relative_path(value: Any, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise PackageContractError(f"{label} must be a nonempty path.")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or any(not part for part in path.parts)
    ):
        raise PackageContractError(f"{label} is unsafe: {value!r}.")
    return path


def file_binding(path: Path, *, repo_root: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PackageContractError(f"Binding input is missing: {path}")
    return {
        "path": path.relative_to(repo_root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def base_execution_id(
    *, regime_id: str, nph: int, route_id: str
) -> str:
    return f"core__{regime_id}__nph{nph}__{route_id}"


def parse_execution_id(execution_id: str) -> tuple[str, int, str]:
    parts = execution_id.split("__")
    if (
        len(parts) != 4
        or parts[0] != "core"
        or not parts[2].startswith("nph")
    ):
        raise PackageContractError(
            f"Malformed stationary-core execution id: {execution_id}"
        )
    try:
        nph = int(parts[2][3:])
    except ValueError as exc:
        raise PackageContractError(
            f"Malformed cutoff in execution id: {execution_id}"
        ) from exc
    regime_id, route_id = parts[1], parts[3]
    if (
        (regime_id, nph) not in REGIME_CUTOFF_PAIRS
        or route_id not in ROUTE_IDS
    ):
        raise PackageContractError(
            f"Execution id is outside the RA36 matrix: {execution_id}"
        )
    return regime_id, nph, route_id


def candidate_representation(route_id: str) -> str:
    return (
        "macro_generator_v1"
        if "_macro_" in route_id
        else "single_pauli_word_v1"
    )


def source_execution_id(base_id: str, route_id: str) -> str:
    return (
        f"{base_id}{ALWAYS_SOURCE_SUFFIX}"
        if route_id.endswith("_always")
        else base_id
    )


def source_protocol_relative(base_id: str, route_id: str) -> str:
    source_id = source_execution_id(base_id, route_id)
    root = (
        ALWAYS_PROTOCOL_ROOT_RELATIVE
        if route_id.endswith("_always")
        else CORE_PROTOCOL_ROOT_RELATIVE
    )
    return f"{root}/{source_id}.json"


def source_family_for(
    *, route_id: str, execution_mode: str
) -> str:
    if not route_id.endswith("_always"):
        return "stationary_core_v11"
    if execution_mode == "authenticated_resume_50_to_70":
        return "always_factorial_v1"
    return "always_factorial_v2"


def resolve_attempt_path(
    *,
    repo_root: Path,
    source_record: Mapping[str, Any],
    attempt_path: str,
) -> Path:
    if attempt_path.startswith(("raw_outputs/", "chtc/")):
        return repo_root / safe_relative_path(
            attempt_path, label="attempt path"
        )
    fetched_dir = Path(str(source_record.get("fetched_dir", "")))
    if not fetched_dir.is_absolute():
        raise PackageContractError(
            "Relative attempt lacks an absolute fetched directory."
        )
    return fetched_dir / safe_relative_path(
        attempt_path, label="attempt basename"
    )


def selected_resume_sources(
    repo_root: Path,
    provenance: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    records = provenance.get("source_records")
    included = provenance.get("included_sources")
    if not isinstance(records, list) or not isinstance(included, list):
        raise PackageContractError(
            "Stationary-core provenance lacks source records."
        )
    records_by_index = {
        int(row["source_receipt_index"]): row
        for row in records
        if isinstance(row, Mapping)
    }
    selected: dict[str, dict[str, Any]] = {}
    for raw in included:
        if not isinstance(raw, Mapping):
            continue
        execution_id = str(raw.get("execution_id", ""))
        try:
            _regime, _nph, route_id = parse_execution_id(execution_id)
        except PackageContractError:
            continue
        source_index = int(raw.get("source_receipt_index", -1))
        source_record = records_by_index.get(source_index)
        if source_record is None:
            raise PackageContractError(
                f"No source record for {execution_id}."
            )
        attempt_text = str(raw.get("attempt_path", ""))
        attempt = resolve_attempt_path(
            repo_root=repo_root,
            source_record=source_record,
            attempt_path=attempt_text,
        )
        attempt_sha256 = require_sha256(
            raw.get("attempt_sha256"),
            label=f"{execution_id} attempt digest",
        )
        selected[execution_id] = {
            "execution_id": execution_id,
            "route_id": route_id,
            "source_receipt_index": source_index,
            "source_package_id": str(
                raw.get("package_id", "")
            ),
            "attempt_report_path": attempt_text,
            "attempt_resolved_path": attempt.as_posix(),
            "attempt_sha256": attempt_sha256,
            "attempt_size_bytes": (
                attempt.stat().st_size if attempt.is_file() else -1
            ),
        }
    expected_routes = {
        base_execution_id(
            regime_id=regime,
            nph=nph,
            route_id=route,
        )
        for regime, nph in REGIME_CUTOFF_PAIRS
        for route in ROUTE_IDS
    }
    missing = set(map(str, provenance.get("missing_execution_ids", ())))
    if (
        len(selected) != RESUME_COUNT
        or len(missing) != FRESH_COUNT
        or set(selected).intersection(missing)
        or set(selected).union(missing) != expected_routes
        or any(not item.endswith("_always") for item in missing)
    ):
        raise PackageContractError(
            "The provenance no longer resolves 27 RA resumes plus nine "
            "missing always-insertion rows."
        )
    return selected


def collision_map(repo_root: Path) -> dict[str, dict[str, Any]]:
    queue_path = repo_root / COLLISION_QUEUE_RELATIVE
    receipt_path = repo_root / COLLISION_SUBMISSION_RECEIPT_RELATIVE
    state_path = repo_root / COLLISION_STATE_SNAPSHOT_RELATIVE
    receipt = load_json(
        receipt_path, label="collision submission receipt"
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        receipt.get("cluster_id") != COLLISION_CLUSTER_ID
        or not isinstance(state, list)
    ):
        raise PackageContractError(
            "Collision evidence no longer binds cluster 9397758."
        )
    state_by_proc = {
        int(row["ProcId"]): row
        for row in state
        if isinstance(row, Mapping)
        and int(row.get("ClusterId", -1)) == COLLISION_CLUSTER_ID
    }
    lines = queue_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < FRESH_COUNT:
        raise PackageContractError(
            "Collision queue has fewer than nine rows."
        )
    mapped: dict[str, dict[str, Any]] = {}
    for proc_id in COLLISION_PROC_IDS:
        fields = lines[proc_id].split("\t")
        if len(fields) != 9:
            raise PackageContractError(
                f"Collision queue row {proc_id} is malformed."
            )
        source_id = fields[0]
        if not source_id.endswith(ALWAYS_SOURCE_SUFFIX):
            raise PackageContractError(
                f"Collision row {proc_id} is not stationary/late."
            )
        base_id = source_id[: -len(ALWAYS_SOURCE_SUFFIX)]
        parse_execution_id(base_id)
        observed = state_by_proc.get(proc_id)
        if (
            observed is None
            or int(observed.get("JobStatus", -1)) != 5
            or int(observed.get("NumJobStarts", -1)) != 0
        ):
            raise PackageContractError(
                f"Local collision snapshot does not show "
                f"{COLLISION_CLUSTER_ID}.{proc_id} held/unstarted."
            )
        mapped[base_id] = {
            "cluster_id": COLLISION_CLUSTER_ID,
            "proc_id": proc_id,
            "source_execution_id": source_id,
            "observed_job_status": 5,
            "observed_num_job_starts": 0,
            "observed_hold_reason": str(
                observed.get("HoldReason", "")
            ),
            "status": "blocked_live_r50_predecessor",
        }
    if len(mapped) != FRESH_COUNT:
        raise PackageContractError(
            "Collision map does not contain exactly nine rows."
        )
    return mapped


def planned_rows(
    *,
    repo_root: Path,
    provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    resume_sources = selected_resume_sources(repo_root, provenance)
    collisions = collision_map(repo_root)
    missing = set(map(str, provenance["missing_execution_ids"]))
    if set(collisions) != missing:
        raise PackageContractError(
            "The nine held predecessors do not exactly match missing RA "
            "always rows."
        )
    rows: list[dict[str, Any]] = []
    for regime_id, nph in REGIME_CUTOFF_PAIRS:
        for route_id in ROUTE_IDS:
            base_id = base_execution_id(
                regime_id=regime_id,
                nph=nph,
                route_id=route_id,
            )
            mode = (
                "authenticated_resume_50_to_70"
                if base_id in resume_sources
                else "fresh_0_to_70"
            )
            representation = candidate_representation(route_id)
            cpus, memory_mb, disk_mb = RESOURCE_ENVELOPES[
                (representation, nph)
            ]
            row: dict[str, Any] = {
                "execution_id": f"{base_id}__r70",
                "base_execution_id": base_id,
                "source_execution_id": source_execution_id(
                    base_id, route_id
                ),
                "regime_id": regime_id,
                "nph": nph,
                "route_id": route_id,
                "candidate_representation": representation,
                "insertion_policy": INSERTION_KIND_BY_ROUTE[
                    route_id
                ],
                "active_gradient_policy": ACTIVE_GRADIENT_POLICY,
                "resource_weighting_scope": (
                    RESOURCE_WEIGHTING_SCOPE
                ),
                "optimizer": OPTIMIZER,
                "optimizer_maxiter": OPTIMIZER_MAXITER,
                "seed": SEED,
                "source_horizon": SOURCE_HORIZON,
                "target_horizon": TARGET_HORIZON,
                "execution_mode": mode,
                "source_family": source_family_for(
                    route_id=route_id,
                    execution_mode=mode,
                ),
                "source_protocol_path": source_protocol_relative(
                    base_id, route_id
                ),
                "resources": {
                    "request_cpus": cpus,
                    "request_memory_mb": memory_mb,
                    "request_disk_mb": disk_mb,
                    "max_runtime_seconds": MAX_RUNTIME_SECONDS,
                    "source": "inherited_exact_r50_envelope",
                    "r70_demonstration_status": "not_demonstrated",
                },
                "execution_authorized": False,
                "submission_authorized": False,
                "submitted": False,
            }
            if mode == "authenticated_resume_50_to_70":
                row["resume_source"] = dict(
                    resume_sources[base_id]
                )
                row["collision_status"] = "none"
            else:
                row["resume_source"] = None
                row["collision"] = dict(collisions[base_id])
                row["collision_status"] = (
                    "blocked_live_r50_predecessor"
                )
            rows.append(row)
    if (
        len(rows) != CELL_COUNT
        or sum(
            row["execution_mode"]
            == "authenticated_resume_50_to_70"
            for row in rows
        )
        != RESUME_COUNT
        or sum(row["execution_mode"] == "fresh_0_to_70" for row in rows)
        != FRESH_COUNT
    ):
        raise AssertionError("RA36 continuation matrix drifted.")
    return tuple(rows)


def validate_source_protocol(
    *,
    repo_root: Path,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    relative = safe_relative_path(
        row["source_protocol_path"], label="source protocol path"
    ).as_posix()
    path = repo_root / relative
    protocol = load_json(path, label=f"{row['execution_id']} protocol")
    verify_self_digest(protocol, label=f"{row['execution_id']} protocol")
    request = protocol.get("request")
    method = request.get("method") if isinstance(request, Mapping) else None
    execution = (
        request.get("execution") if isinstance(request, Mapping) else None
    )
    stop = (
        execution.get("stop")
        if isinstance(execution, Mapping)
        else None
    )
    adapter = (
        request.get("adapter") if isinstance(request, Mapping) else None
    )
    insertion = (
        method.get("insertion")
        if isinstance(method, Mapping)
        else None
    )
    route_contract = protocol.get("route_contract")
    if (
        protocol.get("schema")
        != "paper_i_ra_adapt_resolved_protocol_v1"
        or int(protocol.get("horizon", -1)) != SOURCE_HORIZON
        or not isinstance(stop, Mapping)
        or int(stop.get("maximum_controller_rounds", -1))
        != SOURCE_HORIZON
        or protocol.get("active_gradient_policy")
        != ACTIVE_GRADIENT_POLICY
        or protocol.get("resource_weighting_scope")
        != RESOURCE_WEIGHTING_SCOPE
        or protocol.get("optimizer") != OPTIMIZER
        or int(protocol.get("optimizer_maxiter", -1))
        != OPTIMIZER_MAXITER
        or not isinstance(adapter, Mapping)
        or adapter.get("candidate_representation_id")
        != row["candidate_representation"]
        or not isinstance(insertion, Mapping)
        or insertion.get("kind") != row["insertion_policy"]
        or not isinstance(route_contract, Mapping)
    ):
        raise PackageContractError(
            f"Source protocol drifted for {row['execution_id']}."
        )
    route_digest = require_sha256(
        route_contract.get("sha256"),
        label=f"{row['execution_id']} route digest",
    )
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "canonical_sha256": str(protocol["sha256"]),
        "size_bytes": path.stat().st_size,
        "route_profile": str(route_contract.get("route_profile", "")),
        "route_contract_sha256": route_digest,
        "source_locks": dict(protocol.get("source_locks", {})),
        "request": dict(request),
    }

