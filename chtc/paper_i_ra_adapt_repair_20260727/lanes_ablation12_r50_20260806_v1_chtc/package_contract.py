#!/usr/bin/env python3
"""Fail-closed contract for the repaired 12-cell RA-always package.

The package is intentionally inert: it contains no authority overlay and this
module never authorizes execution, contacts CHTC, or invokes HTCondor.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

PACKAGE_ID = (
    "paper_i_ra_adapt_lanes_ablation12_r50_20260806_v1_chtc"
)
PACKAGE_RELATIVE_ROOT = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "lanes_ablation12_r50_20260806_v1_chtc"
)
CORE_MATERIALIZATION_RELATIVE_ROOT = (
    "chtc/paper_i_ra_adapt_repair_20260727/bundles/materializations/"
    "ra_adapt_macro_always_lanes_ablation_r50_v5"
)
CORE_BUNDLE_ID = "ra_adapt_macro_always_lanes_ablation_r50_v1"
CAMPAIGN_ID = "paper_i_ra_adapt_macro_always_lanes_ablation_r50_v1"
RUN_CLASS = "diagnostic"
EXECUTION_TARGET = "chtc"
HORIZON = 50
OPTIMIZER = "powell"
OPTIMIZER_MAXITER = 200
SEED = 7
ALWAYS_INSERTION_KIND = "always_commutation_reduced"
ALWAYS_INSERTION_MODE = "full_commutation_reduced"
INSERTION_EQUIVALENCE_POLICY = (
    "termwise_cross_component_commutation_earliest_representative_v1"
)
INSERTION_POSITION_SCOPE = (
    "full_logical_ansatz_commutation_classes_every_depth_v2"
)
REGIME_CUTOFF_PAIRS = (
    ("weak_weak", 3),
    ("intermediate_weak", 3),
    ("strong_weak_u8", 3),
    ("weak_strong", 7),
    ("intermediate_strong", 7),
    ("strong_strong_u8", 7),
)
ROUTE_IDS = ("ra_macro_always",)
ARMS = (
    (
        "lanes_on",
        "paper_i_ra_adapt_macro_always_insertion_qiskit_transpile_cost_v1",
    ),
    (
        "lanes_off",
        "paper_i_ra_adapt_macro_always_insertion_no_lanes_"
        "qiskit_transpile_cost_v1",
    ),
)
DIRECT_EXECUTION_COUNT = 12
SMOKE_ROUNDS = 2
SMOKE_EXECUTION_IDS = (
    "lanes_ablation__lanes_on__weak_weak__nph3__ra_macro_always",
    "lanes_ablation__lanes_off__weak_weak__nph3__ra_macro_always",
)
RESOURCE_ENVELOPES = {
    ("ra_macro_always", 3): (4, 49_152, 61_440),
    ("ra_macro_always", 7): (4, 65_536, 81_920),
}
MAX_RUNTIME_SECONDS = 72 * 60 * 60
REMOTE_IMAGE_PATH = "chtc/phase3_optuna/image.sif"
REMOTE_IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)

SMOKE_RECEIPT_NAME = "two_round_smoke_receipt.json"
SOURCE_ARCHIVE_NAME = "source_locked.tar.gz"
SOURCE_ARCHIVE_MANIFEST_NAME = "source_archive_manifest.json"
EXECUTION_PLAN_NAME = "execution_plan.json"
QUEUE_NAME = "queue.tsv"
PACKAGE_MANIFEST_NAME = "package_manifest.json"
JOB_SCHEMA = "paper_i_ra_lanes_ablation_job_v1"
SMOKE_SCHEMA = "paper_i_ra_lanes_ablation_smoke_v1"
PLAN_SCHEMA = "paper_i_ra_lanes_ablation_execution_plan_v1"
MANIFEST_SCHEMA = "paper_i_ra_lanes_ablation_package_v1"
ARCHIVE_SCHEMA = "paper_i_ra_lanes_ablation_source_archive_v1"

CONTROL_FILES = (
    "package_contract.py",
    "run_semantic_preflight.py",
    "build_package.py",
    "validate_package.py",
    "run_cell.py",
    "execute_source_locked_job.sh",
    "submit.sub",
)
GENERATED_FILES = (
    SMOKE_RECEIPT_NAME,
    SOURCE_ARCHIVE_NAME,
    SOURCE_ARCHIVE_MANIFEST_NAME,
    EXECUTION_PLAN_NAME,
    QUEUE_NAME,
    PACKAGE_MANIFEST_NAME,
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PackageContractError(ValueError):
    """Raised when the repaired package contract does not close."""


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
    result = dict(payload)
    result.pop("sha256", None)
    result["sha256"] = canonical_sha256(result)
    return result


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
    if (
        not isinstance(observed, str)
        or SHA256_RE.fullmatch(observed) is None
        or canonical_sha256(unsigned) != observed
    ):
        raise PackageContractError(f"{label} self digest drifted.")
    return observed


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


def direct_execution_rows() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for arm, algorithm_id in ARMS:
        for regime_id, nph in REGIME_CUTOFF_PAIRS:
            route_id = "ra_macro_always"
            cpus, memory_mb, disk_mb = RESOURCE_ENVELOPES[(route_id, nph)]
            execution_id = (
                f"lanes_ablation__{arm}__{regime_id}__nph{nph}__{route_id}"
            )
            rows.append(
                {
                    "execution_id": execution_id,
                    "cell_id": execution_id,
                    "arm": arm,
                    "algorithm_id": algorithm_id,
                    "regime_id": regime_id,
                    "nph": nph,
                    "route_id": route_id,
                    "candidate_representation": "macro_generator_v1",
                    "execution_entrypoint": "run_ra_adapt",
                    "resources": {
                        "request_cpus": cpus,
                        "request_memory_mb": memory_mb,
                        "request_disk_mb": disk_mb,
                        "max_runtime_seconds": MAX_RUNTIME_SECONDS,
                    },
                }
            )
    return tuple(rows)


def _validate_protocol(
    protocol: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
) -> None:
    verify_self_digest(protocol, label=f"{row['cell_id']} protocol")
    method = (
        protocol.get("request", {}).get("method", {})
        if isinstance(protocol.get("request"), Mapping)
        else {}
    )
    insertion = (
        method.get("insertion") if isinstance(method, Mapping) else None
    )
    route_contract = protocol.get("route_contract")
    execution_settings = (
        route_contract.get("execution_settings")
        if isinstance(route_contract, Mapping)
        else None
    )
    invariants = (
        route_contract.get("semantic_invariants")
        if isinstance(route_contract, Mapping)
        else None
    )
    materialization = protocol.get("bundle_materialization")
    if (
        not isinstance(materialization, Mapping)
        or materialization.get("cell_id") != row["cell_id"]
        or protocol.get("candidate_representation")
        != row["candidate_representation"]
        or int(protocol.get("horizon", -1)) != HORIZON
        or str(protocol.get("optimizer", "")).lower() != OPTIMIZER
        or int(protocol.get("optimizer_maxiter", -1))
        != OPTIMIZER_MAXITER
        or protocol.get("seeds")
        != {"adapt": SEED, "transpiler": SEED}
        or not isinstance(insertion, Mapping)
        or insertion.get("kind") != ALWAYS_INSERTION_KIND
        or not isinstance(execution_settings, Mapping)
        or execution_settings.get("adapt_insertion_mode")
        != ALWAYS_INSERTION_MODE
        or not isinstance(invariants, Mapping)
        or invariants.get("insertion_position_scope")
        != INSERTION_POSITION_SCOPE
        or invariants.get("insertion_equivalence_policy")
        != INSERTION_EQUIVALENCE_POLICY
        or (
            invariants.get("canonical_insertion_policy") is not None
            and invariants.get("canonical_insertion_policy")
            != ALWAYS_INSERTION_KIND
        )
        or protocol.get("execution_authorized") is not False
    ):
        raise PackageContractError(
            f"Repaired always protocol drifted: {row['cell_id']}."
        )


def validate_core_authority(repo_root: str | Path) -> dict[str, Any]:
    """Validate the inert lanes-ablation bundle this package ships."""

    root = Path(repo_root).resolve()
    core_root = root / CORE_MATERIALIZATION_RELATIVE_ROOT
    bundle_root = core_root / CORE_BUNDLE_ID
    manifest = load_json(
        bundle_root / "bundle_manifest.json",
        label="lanes ablation bundle manifest",
    )
    verify_self_digest(manifest, label="lanes ablation bundle manifest")
    if (
        manifest.get("bundle_id") != CORE_BUNDLE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("run_class") != RUN_CLASS
        or manifest.get("execution_target") != EXECUTION_TARGET
        or manifest.get("lanes_ablation_cell_count") != DIRECT_EXECUTION_COUNT
    ):
        raise PackageContractError("Lanes-ablation bundle manifest drifted.")
    validation = load_json(
        bundle_root / "validation_report.json",
        label="lanes ablation validation report",
    )
    verify_self_digest(validation, label="lanes ablation validation report")
    source_locks = load_json(
        bundle_root / "source_locks.json",
        label="lanes ablation source locks",
    )
    verify_self_digest(source_locks, label="lanes ablation source locks")
    return {
        "bundle_manifest": manifest,
        "validation_report": validation,
        "source_locks": source_locks,
        "implementation_inventory": source_locks["implementation_sources"],
        "final_binding": {
            "path": (
                bundle_root / "bundle_manifest.json"
            ).relative_to(root).as_posix(),
            "sha256": sha256_file(bundle_root / "bundle_manifest.json"),
            "canonical_sha256": manifest["sha256"],
        },
        "protocol_bindings": {
            path.stem: {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "canonical_sha256": canonical_sha256(
                    {
                        key: value
                        for key, value in load_json(
                            path, label="cell protocol"
                        ).items()
                        if key != "sha256"
                    }
                ),
            }
            for path in sorted((bundle_root / "protocols").glob("*.json"))
        },
        "core_root": core_root,
        "bundle_root": bundle_root,
    }


def _validate_candidate_plan(
    plan: Mapping[str, Any],
    *,
    requested: Sequence[int],
) -> tuple[bool, bool]:
    requested_positions = [int(x) for x in plan.get("requested_positions", ())]
    representatives = [
        int(x) for x in plan.get("representative_positions", ())
    ]
    members = plan.get("members_by_representative")
    representative_by_position = plan.get("representative_by_position")
    if (
        plan.get("schema")
        != "commutation_reduced_insertion_positions_v1"
        or requested_positions != list(requested)
        or not representatives
        or not isinstance(members, Mapping)
        or not isinstance(representative_by_position, Mapping)
    ):
        raise PackageContractError(
            "Smoke candidate plan is not a reduced full-domain plan."
        )
    normalized_members = {
        int(key): [int(x) for x in value]
        for key, value in members.items()
        if isinstance(value, list)
    }
    if (
        sorted(normalized_members) != representatives
        or any(
            not values or representative != min(values)
            for representative, values in normalized_members.items()
        )
        or sorted(
            position
            for values in normalized_members.values()
            for position in values
        )
        != requested_positions
        or {
            int(key): int(value)
            for key, value in representative_by_position.items()
        }
        != {
            position: representative
            for representative, values in normalized_members.items()
            for position in values
        }
        or int(plan.get("collapsed_position_count", -1))
        != len(requested_positions) - len(representatives)
    ):
        raise PackageContractError(
            "Smoke commutation-equivalence membership does not close."
        )
    crossings = plan.get("commuting_crossings")
    if (
        not isinstance(crossings, list)
        or len(crossings) != max(requested_positions)
        or any(not isinstance(value, bool) for value in crossings)
    ):
        raise PackageContractError(
            "Smoke commuting-crossing certificate drifted."
        )
    class_start_by_position: dict[int, int] = {0: 0}
    class_start = 0
    for crossing_index, crossing in enumerate(crossings):
        if not crossing:
            class_start = crossing_index + 1
        class_start_by_position[crossing_index + 1] = class_start
    requested_by_class: dict[int, list[int]] = {}
    for position in requested_positions:
        requested_by_class.setdefault(
            class_start_by_position[position], []
        ).append(position)
    expected_members = {
        min(values): values for values in requested_by_class.values()
    }
    if normalized_members != expected_members:
        raise PackageContractError(
            "Smoke equivalence classes disagree with their "
            "commuting-crossing certificate."
        )
    collapsed = len(representatives) < len(requested_positions)
    uncollapsed = len(representatives) == len(requested_positions)
    return collapsed, uncollapsed


def validate_smoke_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    verify_self_digest(receipt, label="two-round smoke receipt")
    observations = receipt.get("route_observations")
    if (
        receipt.get("schema") != SMOKE_SCHEMA
        or receipt.get("status") != "passed"
        or receipt.get("package_id") != PACKAGE_ID
        or receipt.get("maximum_controller_rounds") != SMOKE_ROUNDS
        or receipt.get("paper_evidence_allowed") is not False
        or receipt.get("execution_authorized") is not False
        or receipt.get("submission_authorized") is not False
        or not isinstance(observations, list)
        or [row.get("arm") for row in observations]
        != [arm for arm, _ in ARMS]
    ):
        raise PackageContractError("Two-round smoke envelope drifted.")

    for observation in observations:
        rounds = observation.get("accepted_round_reduction_receipts")
        if (
            not isinstance(rounds, list)
            or len(rounds) != SMOKE_ROUNDS
            or observation.get("controller_round_count")
            != SMOKE_ROUNDS
        ):
            raise PackageContractError(
                "Two-round smoke trajectory length drifted."
            )
        for round_index, reduction in enumerate(rounds, start=1):
            expected_requested = list(range(round_index))
            plans = reduction.get("candidate_position_plans")
            candidate_count = int(reduction.get("candidate_count", -1))
            collapsed_count = int(
                reduction.get("collapsed_position_count", -1)
            )
            retained_count = int(
                reduction.get("retained_representative_count", -1)
            )
            if (
                reduction.get("schema")
                != "commutation_reduced_insertion_domain_receipt_v1"
                or reduction.get("policy") != ALWAYS_INSERTION_KIND
                or reduction.get("domain_open") is not True
                or reduction.get("domain_state") != "open"
                or reduction.get("effective_insertion_mode")
                != ALWAYS_INSERTION_MODE
                or reduction.get("requested_positions")
                != expected_requested
                or int(reduction.get("requested_position_count", -1))
                != len(expected_requested)
                or not isinstance(plans, list)
                or len(plans) != candidate_count
                or candidate_count <= 0
                or retained_count + collapsed_count
                != candidate_count * len(expected_requested)
            ):
                raise PackageContractError(
                    f"Round-{round_index} reduced domain count closure "
                    "failed."
                )
            for plan in plans:
                if not isinstance(plan, Mapping):
                    raise PackageContractError(
                        "Smoke candidate plan is malformed."
                    )
                _validate_candidate_plan(
                    plan, requested=expected_requested
                )
        second = rounds[1]
        plans = second.get("candidate_position_plans")
        collapsed_count = int(
            second.get("collapsed_position_count", -1)
        )
        if not isinstance(plans, list) or collapsed_count <= 0:
            raise PackageContractError(
                "Second-round reduced domain count closure failed."
            )
        collapsed = False
        uncollapsed = False
        for plan in plans:
            if not isinstance(plan, Mapping):
                raise PackageContractError(
                    "Smoke candidate plan is malformed."
                )
            plan_collapsed, plan_uncollapsed = _validate_candidate_plan(
                plan, requested=(0, 1)
            )
            collapsed = collapsed or plan_collapsed
            uncollapsed = uncollapsed or plan_uncollapsed
        if not collapsed or not uncollapsed:
            raise PackageContractError(
                "Smoke lacks both collapsed and uncollapsed candidates."
            )
    return dict(receipt)


__all__ = [
    "ALWAYS_INSERTION_KIND",
    "ALWAYS_INSERTION_MODE",
    "ARCHIVE_SCHEMA",
    "CAMPAIGN_ID",
    "CONTROL_FILES",
    "DIRECT_EXECUTION_COUNT",
    "EXECUTION_PLAN_NAME",
    "EXECUTION_TARGET",
    "GENERATED_FILES",
    "HORIZON",
    "JOB_SCHEMA",
    "MANIFEST_SCHEMA",
    "PACKAGE_ID",
    "PACKAGE_MANIFEST_NAME",
    "PLAN_SCHEMA",
    "PackageContractError",
    "QUEUE_NAME",
    "REMOTE_IMAGE_PATH",
    "REMOTE_IMAGE_SHA256",
    "ROUTE_IDS",
    "RUN_CLASS",
    "SMOKE_EXECUTION_IDS",
    "SMOKE_RECEIPT_NAME",
    "SMOKE_ROUNDS",
    "SMOKE_SCHEMA",
    "SOURCE_ARCHIVE_MANIFEST_NAME",
    "SOURCE_ARCHIVE_NAME",
    "canonical_json_bytes",
    "canonical_sha256",
    "digested",
    "direct_execution_rows",
    "load_json",
    "repo_root_from_script",
    "safe_relative_path",
    "sha256_file",
    "validate_core_authority",
    "validate_smoke_receipt",
    "verify_self_digest",
]
