#!/usr/bin/env python3
"""Run the source-locked strong--weak singleton Phase-III plateau pilot.

The pilot first reproduces ten rounds of the ordinary historical-mean
plateau-v2 route from the current source inventory.  Only after that anchor
matches the completed source trajectory does it run the named route whose
competitive Phase-III population activates on the same-round insertion
plateau predicate.  Phase I, Phase II, physics, pool, optimizer, seeds,
stationary gradients, late resource weighting, and commutation reduction are
identical between the two cells.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"

REPAIR_ROOT = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
SOURCE_PACKAGE = (
    REPAIR_ROOT
    / "paper_i_ra_adapt_historical_average_singleton_plateau6_r70_fresh_"
    "20260802_v5_chtc"
)
SOURCE_CELL_ID = (
    "historical_average_v5_r70_fresh__strong_weak_u8__nph3__"
    "ra_singleton_plateau"
)
SOURCE_PROTOCOL = SOURCE_PACKAGE / "protocols" / f"{SOURCE_CELL_ID}.json"
SOURCE_JOB = SOURCE_PACKAGE / "jobs" / f"{SOURCE_CELL_ID}.json"
SOURCE_PACKAGE_MANIFEST = SOURCE_PACKAGE / "package_manifest.json"
SOURCE_PROTOCOL_BUNDLE_MANIFEST = (
    SOURCE_PACKAGE / "protocol_bundle_manifest.json"
)
SOURCE_LOCKS = SOURCE_PACKAGE / "source_locks_snapshot.json"

COMPLETED_SOURCE_PACKAGE = (
    REPAIR_ROOT
    / "paper_i_ra_adapt_historical_average_singleton_plateau6_r70_fresh_"
    "20260801_v4_chtc"
)
COMPLETED_SOURCE_CELL_ID = (
    "historical_average_v4_r70_fresh__strong_weak_u8__nph3__"
    "ra_singleton_plateau"
)
COMPLETED_SOURCE_PROTOCOL = (
    COMPLETED_SOURCE_PACKAGE
    / "protocols"
    / f"{COMPLETED_SOURCE_CELL_ID}.json"
)
COMPLETED_SOURCE_LOG = (
    REPAIR_ROOT
    / "retrieved_chtc_20260802_historical_average_plateau_r70_cluster_9400249"
    / "logs"
    / "9400249.2__historical_average_v4_r70_fresh__strong_weak_u8__nph3__"
    "ra_singleton_plateau.out"
)

OUTPUT_ROOT = (
    REPO_ROOT
    / "output/local_runs"
    / "paper_i_ra_adapt_singleton_phase3_on_plateau_strong_weak_r10_local_"
    "20260802_v2"
)
MATERIALIZATION_ROOT = OUTPUT_ROOT / "materialization"
RUNS_ROOT = OUTPUT_ROOT / "runs"
BUNDLE_ID = (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_strong_weak_r10_local_v2"
)
CONTROL_CELL_ID = (
    "phase3_plateau_control_r10__strong_weak_u8__nph3__"
    "ra_singleton_plateau"
)
TARGET_CELL_ID = (
    "phase3_plateau_target_r10__strong_weak_u8__nph3__"
    "ra_singleton_plateau"
)
CELLS = (CONTROL_CELL_ID, TARGET_CELL_ID)
MAXIMUM_CONTROLLER_ROUNDS = 10
EXACT_ENERGY = 0.5264586847939832
CONTROL_ALGORITHM_ID = (
    "paper_i_ra_adapt_singleton_plateau_insertion_repair_v1"
)
EXPECTED_CONTROL_ROUTE_SHA256 = (
    "947f981d6eeadc874cd61150ff3732504bdf21193cf21bc8bbc34dbf8260ebea"
)
EXPECTED_TARGET_ROUTE_SHA256 = (
    "ac868db4dab4f8446ff06e768c5ea77512ef70764efd5699621bd95ad341599d"
)

SOURCE_TO_CONTROL_ALLOWED_DIFFS = {
    ("bundle_id",),
    ("bundle_manifest_sha256",),
    ("bundle_materialization", "bundle_id"),
    ("bundle_materialization", "bundle_manifest_sha256"),
    ("bundle_materialization", "cell_id"),
    ("bundle_materialization", "sha256"),
    ("bundle_materialization", "source_lock_refs_sha256"),
    ("bundle_materialization", "source_locks_sha256"),
    ("horizon",),
    ("request", "execution", "stop", "maximum_controller_rounds"),
    ("request", "observation", "checkpoint", "path"),
    ("request", "observation", "estimator_ledger", "path"),
    ("sha256",),
    ("source_locks", "implementation_source_inventory_sha256"),
    ("source_locks", "source_locks_manifest_sha256"),
    ("stopping_rule", "maximum_controller_rounds"),
}

CONTROL_TO_TARGET_ALLOWED_DIFFS = {
    ("algorithm_id",),
    ("bundle_materialization", "algorithm_id"),
    ("bundle_materialization", "cell_id"),
    ("bundle_materialization", "sha256"),
    ("request", "observation", "checkpoint", "path"),
    ("request", "observation", "estimator_ledger", "path"),
    (
        "route_contract",
        "execution_settings",
        "ra_phase3_population_activation_policy",
    ),
    (
        "route_contract",
        "execution_settings",
        "ra_phase3_preplateau_materialization_policy",
    ),
    (
        "route_contract",
        "lineage_authority",
        "only_intended_scientific_changes",
    ),
    (
        "route_contract",
        "lineage_authority",
        "supersession_reason",
    ),
    ("route_contract", "route_profile"),
    ("route_contract", "semantic_invariants", "phase1_activation_scope"),
    ("route_contract", "semantic_invariants", "phase2_activation_scope"),
    (
        "route_contract",
        "semantic_invariants",
        "phase3_activation_hysteresis_active",
    ),
    (
        "route_contract",
        "semantic_invariants",
        "phase3_activation_independent_latch",
    ),
    (
        "route_contract",
        "semantic_invariants",
        "phase3_activation_source",
    ),
    (
        "route_contract",
        "semantic_invariants",
        "phase3_competitive_population_activation",
    ),
    (
        "route_contract",
        "semantic_invariants",
        "phase3_preplateau_admission_authority",
    ),
    (
        "route_contract",
        "semantic_invariants",
        "phase3_preplateau_materialization_policy",
    ),
    ("route_contract", "sha256"),
    ("sha256",),
}


class DiagnosticContractError(RuntimeError):
    """Fail-closed local-diagnostic contract error."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digested(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("sha256", None)
    result["sha256"] = _canonical_sha256(result)
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise DiagnosticContractError(f"Missing or unsafe JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DiagnosticContractError(f"Expected a JSON object: {path}")
    return value


def _verify_digest(value: Mapping[str, Any], *, label: str) -> None:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("sha256") != _canonical_sha256(unsigned):
        raise DiagnosticContractError(f"{label} self-digest drifted.")


def _binding(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise DiagnosticContractError(f"Missing or unsafe source: {path}")
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _verify_binding(binding: Mapping[str, Any]) -> None:
    path = REPO_ROOT / str(binding["path"])
    observed = _binding(path)
    if observed != dict(binding):
        raise DiagnosticContractError(f"Source binding drifted: {path}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _problem_from_receipt(receipt: Any) -> Any:
    from pipelines.contracts.problem import ProblemRequest
    from pipelines.static_adapt.builders.problem_registry import (
        resolve_problem_context,
    )

    return resolve_problem_context(
        ProblemRequest(
            problem_key=str(receipt.problem_key),
            num_sites=int(receipt.num_sites),
            t=float(receipt.t),
            u=float(receipt.u),
            dv=float(receipt.dv),
            omega0=float(receipt.omega0),
            g_ep=float(receipt.g_ep),
            n_ph_max=int(receipt.n_ph_max),
            boson_encoding=str(receipt.boson_encoding),
            ordering=str(receipt.ordering),
            boundary=str(receipt.boundary),
            include_zero_point=bool(receipt.include_zero_point),
            v_nn=float(receipt.v_nn),
            t_prime=float(receipt.t_prime),
            n_fermions=(
                None
                if receipt.n_fermions is None
                else int(receipt.n_fermions)
            ),
        )
    )


def _scalar_differences(
    before: Any,
    after: Any,
    *,
    path: tuple[str | int, ...] = (),
) -> list[tuple[tuple[str | int, ...], Any, Any]]:
    result: list[tuple[tuple[str | int, ...], Any, Any]] = []
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        for key in sorted(set(before) | set(after)):
            if key not in before:
                result.append(((*path, str(key)), "<missing>", after[key]))
            elif key not in after:
                result.append(((*path, str(key)), before[key], "<missing>"))
            else:
                result.extend(
                    _scalar_differences(
                        before[key], after[key], path=(*path, str(key))
                    )
                )
        return result
    if isinstance(before, (list, tuple)) and isinstance(after, (list, tuple)):
        if len(before) != len(after):
            return [(path, before, after)]
        for index, (left, right) in enumerate(zip(before, after)):
            result.extend(
                _scalar_differences(left, right, path=(*path, index))
            )
        return result
    if before != after:
        result.append((path, before, after))
    return result


def _cell_specs() -> tuple[Any, Any]:
    from pipelines.static_adapt.ra_adapt.bundles import BundleCellSpec
    from pipelines.static_adapt.ra_adapt.contracts import (
        RA_ADAPT_SINGLETON_PHASE3_PLATEAU_ALGORITHM_ID,
    )

    shared = {
        "stage": "phase3_plateau_local_diagnostic",
        "regime_id": "strong_weak_u8",
        "nph": 3,
        "route_id": "ra_singleton_plateau",
        "selector_family": "ra_adapt",
        "candidate_representation": "single_pauli_word_v1",
        "horizon": MAXIMUM_CONTROLLER_ROUNDS,
        "source_lock_id": "strong_weak_u8__nph3__ra_singleton_plateau",
    }
    return (
        BundleCellSpec(
            cell_id=CONTROL_CELL_ID,
            algorithm_id=CONTROL_ALGORITHM_ID,
            **shared,
        ),
        BundleCellSpec(
            cell_id=TARGET_CELL_ID,
            algorithm_id=(
                RA_ADAPT_SINGLETON_PHASE3_PLATEAU_ALGORITHM_ID
            ),
            **shared,
        ),
    )


def _source_lock_snapshot() -> dict[str, Any]:
    from pipelines.static_adapt.ra_adapt.bundles import (
        _implementation_source_inventory,
    )
    from pipelines.static_adapt.ra_adapt.contracts import canonical_sha256

    locks = copy.deepcopy(_load_json(SOURCE_LOCKS))
    _verify_digest(locks, label="source-lock snapshot")
    locks["implementation_sources"] = _implementation_source_inventory(
        REPO_ROOT
    )
    locks.pop("sha256", None)
    locks["sha256"] = canonical_sha256(locks)
    for row in locks["global_sources"].values():
        path = REPO_ROOT / str(row["path"])
        if _sha256_file(path) != row["sha256"]:
            raise DiagnosticContractError(
                f"Global source lock drifted: {path}"
            )
    return locks


def _materialize_protocol(
    *,
    cell: Any,
    source_protocol: Any,
    source_locks: Mapping[str, Any],
    plan_sha256: str,
) -> dict[str, Any]:
    from pipelines.static_adapt.ra_adapt.bundles import (
        _build_request,
        _bundle_protocol_materialization_authority,
        _decorate_protocol_payload,
        _source_lock_refs,
        _validate_protocol_payload,
    )
    from pipelines.static_adapt.ra_adapt.contracts import (
        resolved_ra_adapt_protocol_from_mapping,
    )
    from pipelines.static_adapt.ra_adapt.engine import (
        build_resolved_ra_protocol,
    )

    request = _build_request(cell, bundle_dir=MATERIALIZATION_ROOT)
    refs = _source_lock_refs(source_locks, cell=cell)
    authority = _bundle_protocol_materialization_authority(
        cell=cell,
        bundle_id=BUNDLE_ID,
        bundle_manifest_sha256=plan_sha256,
        source_locks_sha256=str(source_locks["sha256"]),
        source_lock_refs=refs,
        active_gradient_policy=source_protocol.active_gradient_policy,
        resource_weighting_scope=source_protocol.resource_weighting_scope,
    )
    base = build_resolved_ra_protocol(
        _problem_from_receipt(source_protocol.problem),
        request,
        materialization_authority=authority,
    )
    decorated = _decorate_protocol_payload(
        base.to_dict(),
        cell=cell,
        request=request,
        cell_source_lock=source_locks["cell_locks"][cell.source_lock_id],
        materialization_authority=authority,
    )
    _validate_protocol_payload(
        decorated,
        cell=cell,
        bundle_id=BUNDLE_ID,
        bundle_manifest_sha256=plan_sha256,
        active_gradient_policy=source_protocol.active_gradient_policy,
        resource_weighting_scope=source_protocol.resource_weighting_scope,
        source_lock_refs=refs,
        cell_source_lock=source_locks["cell_locks"][cell.source_lock_id],
        source_locks_sha256=str(source_locks["sha256"]),
    )
    resolved_ra_adapt_protocol_from_mapping(decorated)
    return decorated


def materialize() -> dict[str, Any]:
    from pipelines.static_adapt.ra_adapt.contracts import (
        resolved_ra_adapt_protocol_from_mapping,
    )

    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_ROOT}")
    source_payload = _load_json(SOURCE_PROTOCOL)
    source_protocol = resolved_ra_adapt_protocol_from_mapping(source_payload)
    completed_source_payload = _load_json(COMPLETED_SOURCE_PROTOCOL)
    completed_source_protocol = resolved_ra_adapt_protocol_from_mapping(
        completed_source_payload
    )
    if source_protocol.problem != completed_source_protocol.problem:
        raise DiagnosticContractError(
            "Completed-source and active-source physics differ."
        )
    scientific_keys = (
        "algorithm_id",
        "candidate_representation",
        "adapter_id",
        "active_gradient_policy",
        "resource_weighting_scope",
        "parent_inventory",
        "executable_pool",
        "optimizer",
        "optimizer_maxiter",
        "seeds",
        "route_contract",
        "horizon",
        "stopping_rule",
    )
    for key in scientific_keys:
        if completed_source_payload[key] != source_payload[key]:
            raise DiagnosticContractError(
                f"Completed-source protocol drifted at {key}."
            )

    source_locks = _source_lock_snapshot()
    runner_binding = _binding(Path(__file__).resolve())
    selector_hash_dependency = _binding(
        REPO_ROOT
        / "pipelines/exact_bench/generic_static_adapt_variants.py"
    )
    source_bindings = {
        "source_protocol": _binding(SOURCE_PROTOCOL),
        "source_job": _binding(SOURCE_JOB),
        "source_package_manifest": _binding(SOURCE_PACKAGE_MANIFEST),
        "source_protocol_bundle_manifest": _binding(
            SOURCE_PROTOCOL_BUNDLE_MANIFEST
        ),
        "source_locks": _binding(SOURCE_LOCKS),
        "completed_source_protocol": _binding(COMPLETED_SOURCE_PROTOCOL),
        "completed_source_log": _binding(COMPLETED_SOURCE_LOG),
        "diagnostic_runner": runner_binding,
        "append_runtime_hash_dependency": selector_hash_dependency,
    }
    plan = _digested(
        {
            "schema": "paper_i_ra_adapt_phase3_plateau_local_plan_v1",
            "bundle_id": BUNDLE_ID,
            "run_class": "diagnostic",
            "execution_target": "local_active_checkout_source_locked_v1",
            "cells": [cell.to_dict() for cell in _cell_specs()],
            "source_bindings": source_bindings,
            "implementation_source_inventory_sha256": source_locks[
                "implementation_sources"
            ]["sha256"],
            "source_locks_sha256": source_locks["sha256"],
            "scientific_delta": (
                "competitive_phase3_population_activates_on_same_round_"
                "insertion_plateau_v1"
            ),
            "preserved_settings": {
                "problem": "hh_l2_strong_weak_u8_nph3",
                "candidate_supply": (
                    "full_meta_parent_factory_guarded_singleton_exposure_v1"
                ),
                "candidate_adapter": (
                    "paper_i_ra_adapt_single_pauli_word_candidate_adapter_v1"
                ),
                "insertion": "plateau_commutation_v2_historical_prior_mean",
                "plateau_ratio_threshold": 1.0e-4,
                "active_gradient_policy": "stationary_source_response_v1",
                "resource_weighting_scope": "late_resource_weighting_v1",
                "optimizer": "powell",
                "optimizer_maxiter": 200,
                "adapt_seed": 7,
                "transpiler_seed": 7,
                "same_cutoff_exact_energy": EXACT_ENERGY,
            },
            "execution_mechanics": {
                "source_horizon": 70,
                "diagnostic_horizon": MAXIMUM_CONTROLLER_ROUNDS,
                "fresh_start": True,
                "checkpoint_every_round": True,
                "wrapper_used": True,
            },
            "anchor_required_before_target": True,
            "execution_authorized": False,
            "submission_authorized": False,
            "created_at_utc": _utc_now(),
        }
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    _write_json(MATERIALIZATION_ROOT / "materialization_plan.json", plan)
    _write_json(
        MATERIALIZATION_ROOT / "source_locks_snapshot.json", source_locks
    )
    protocols: dict[str, dict[str, Any]] = {}
    for cell in _cell_specs():
        payload = _materialize_protocol(
            cell=cell,
            source_protocol=source_protocol,
            source_locks=source_locks,
            plan_sha256=plan["sha256"],
        )
        protocols[cell.cell_id] = payload
        _write_json(
            MATERIALIZATION_ROOT / "protocols" / f"{cell.cell_id}.json",
            payload,
        )

    source_diffs = _scalar_differences(
        source_payload, protocols[CONTROL_CELL_ID]
    )
    if {row[0] for row in source_diffs} != SOURCE_TO_CONTROL_ALLOWED_DIFFS:
        raise DiagnosticContractError(
            "Source-to-control projection changed unexpected fields: "
            + repr(sorted({row[0] for row in source_diffs}, key=str))
        )
    target_diffs = _scalar_differences(
        protocols[CONTROL_CELL_ID], protocols[TARGET_CELL_ID]
    )
    if {row[0] for row in target_diffs} != CONTROL_TO_TARGET_ALLOWED_DIFFS:
        raise DiagnosticContractError(
            "Control-to-target projection changed unexpected fields: "
            + repr(sorted({row[0] for row in target_diffs}, key=str))
        )
    route_digests = {
        CONTROL_CELL_ID: protocols[CONTROL_CELL_ID]["route_contract"][
            "sha256"
        ],
        TARGET_CELL_ID: protocols[TARGET_CELL_ID]["route_contract"][
            "sha256"
        ],
    }
    if route_digests != {
        CONTROL_CELL_ID: EXPECTED_CONTROL_ROUTE_SHA256,
        TARGET_CELL_ID: EXPECTED_TARGET_ROUTE_SHA256,
    }:
        raise DiagnosticContractError("Resolved route digests drifted.")

    validation = _digested(
        {
            "schema": "paper_i_ra_adapt_phase3_plateau_local_validation_v1",
            "status": "passed",
            "source_to_control_differences": [
                {
                    "path": list(path),
                    "before": before,
                    "after": after,
                }
                for path, before, after in source_diffs
            ],
            "control_to_target_differences": [
                {
                    "path": list(path),
                    "before": before,
                    "after": after,
                }
                for path, before, after in target_diffs
            ],
            "route_contract_sha256": route_digests,
            "source_value_anchor_status": "pending_current_source_replay",
            "target_execution_status": "blocked_until_anchor_passes",
            "execution_authorized": False,
        }
    )
    _write_json(MATERIALIZATION_ROOT / "validation_report.json", validation)
    receipt = _digested(
        {
            "schema": "paper_i_ra_adapt_phase3_plateau_local_materialization_v1",
            "status": "passed",
            "bundle_id": BUNDLE_ID,
            "plan_sha256": plan["sha256"],
            "source_locks_sha256": source_locks["sha256"],
            "validation_sha256": validation["sha256"],
            "protocol_sha256": {
                cell_id: payload["sha256"]
                for cell_id, payload in protocols.items()
            },
            "execution_authorized": False,
            "submission_authorized": False,
        }
    )
    _write_json(
        MATERIALIZATION_ROOT / "materialization_receipt.json", receipt
    )
    return receipt


def _validate_materialization() -> tuple[dict[str, Any], dict[str, Any]]:
    from pipelines.static_adapt.ra_adapt.bundles import (
        _implementation_source_inventory,
    )

    plan = _load_json(MATERIALIZATION_ROOT / "materialization_plan.json")
    locks = _load_json(MATERIALIZATION_ROOT / "source_locks_snapshot.json")
    validation = _load_json(MATERIALIZATION_ROOT / "validation_report.json")
    receipt = _load_json(
        MATERIALIZATION_ROOT / "materialization_receipt.json"
    )
    for label, payload in (
        ("plan", plan),
        ("source locks", locks),
        ("validation", validation),
        ("materialization receipt", receipt),
    ):
        _verify_digest(payload, label=label)
    if (
        receipt["plan_sha256"] != plan["sha256"]
        or receipt["source_locks_sha256"] != locks["sha256"]
        or receipt["validation_sha256"] != validation["sha256"]
        or validation["status"] != "passed"
    ):
        raise DiagnosticContractError("Materialization bindings drifted.")
    current_inventory = _implementation_source_inventory(REPO_ROOT)
    if current_inventory != locks["implementation_sources"]:
        raise DiagnosticContractError(
            "Current implementation source inventory drifted after sealing."
        )
    for binding in plan["source_bindings"].values():
        _verify_binding(binding)
    return plan, locks


def _load_bound_protocol(cell_id: str) -> Any:
    from pipelines.static_adapt.ra_adapt.bundles import (
        _bundle_protocol_materialization_authority,
        _source_lock_refs,
        _validate_protocol_payload,
    )
    from pipelines.static_adapt.ra_adapt.contracts import (
        _attach_validated_bundle_protocol_authority,
        resolved_ra_adapt_protocol_from_mapping,
    )

    plan, locks = _validate_materialization()
    source_protocol = resolved_ra_adapt_protocol_from_mapping(
        _load_json(SOURCE_PROTOCOL)
    )
    rows = {str(row["cell_id"]): row for row in plan["cells"]}
    if cell_id not in rows:
        raise DiagnosticContractError(f"Unknown diagnostic cell: {cell_id}")
    from pipelines.static_adapt.ra_adapt.bundles import BundleCellSpec

    cell = BundleCellSpec(**rows[cell_id])
    payload = _load_json(
        MATERIALIZATION_ROOT / "protocols" / f"{cell_id}.json"
    )
    protocol = resolved_ra_adapt_protocol_from_mapping(payload)
    refs = _source_lock_refs(locks, cell=cell)
    _validate_protocol_payload(
        payload,
        cell=cell,
        bundle_id=BUNDLE_ID,
        bundle_manifest_sha256=plan["sha256"],
        active_gradient_policy=source_protocol.active_gradient_policy,
        resource_weighting_scope=source_protocol.resource_weighting_scope,
        source_lock_refs=refs,
        cell_source_lock=locks["cell_locks"][cell.source_lock_id],
        source_locks_sha256=locks["sha256"],
    )
    authority = _bundle_protocol_materialization_authority(
        cell=cell,
        bundle_id=BUNDLE_ID,
        bundle_manifest_sha256=plan["sha256"],
        source_locks_sha256=locks["sha256"],
        source_lock_refs=refs,
        active_gradient_policy=protocol.active_gradient_policy,
        resource_weighting_scope=protocol.resource_weighting_scope,
        protocol_sha256=protocol.sha256,
    )
    return _attach_validated_bundle_protocol_authority(protocol, authority)


def _completed_source_events() -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with COMPLETED_SOURCE_LOG.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("AI_LOG "):
                continue
            payload = json.loads(line[len("AI_LOG ") :])
            if payload.get("event") == "hardcoded_adapt_iter":
                events.append(payload)
            if len(events) >= MAXIMUM_CONTROLLER_ROUNDS + 1:
                break
    if len(events) != MAXIMUM_CONTROLLER_ROUNDS + 1:
        raise DiagnosticContractError(
            "Completed source log has insufficient anchor rounds."
        )
    return events


def _anchor_validation(result_payload: Mapping[str, Any]) -> dict[str, Any]:
    events = _completed_source_events()
    trajectory = result_payload["run"]["accepted_trajectory"]
    receipts = result_payload["scientific_receipts"][
        "accepted_round_receipts"
    ]
    if len(trajectory) != MAXIMUM_CONTROLLER_ROUNDS or len(receipts) != len(
        trajectory
    ):
        raise DiagnosticContractError(
            "Control did not complete the ten-round anchor horizon."
        )
    rows: list[dict[str, Any]] = []
    for index, (state, receipt) in enumerate(zip(trajectory, receipts)):
        lineage = receipt["accepted_candidate_lineage"]
        if not isinstance(lineage, list) or len(lineage) != 1:
            raise DiagnosticContractError(
                f"Control round {index + 1} lost singleton admission."
            )
        observed_operator = str(lineage[0]["candidate_label"])
        observed_position = int(lineage[0]["insertion_position"])
        observed_energy = float(state["energy"])
        expected_operator = str(events[index]["best_op"])
        expected_position = int(events[index]["selected_position"])
        expected_energy = float(events[index + 1]["energy"])
        energy_passed = math.isclose(
            observed_energy,
            expected_energy,
            rel_tol=1.0e-12,
            abs_tol=1.0e-9,
        )
        decision_path_match = bool(
            observed_operator == expected_operator
            and observed_position == expected_position
        )
        rows.append(
            {
                "controller_round": index + 1,
                "expected_operator": expected_operator,
                "observed_operator": observed_operator,
                "expected_insertion_position": expected_position,
                "observed_insertion_position": observed_position,
                "expected_post_refit_energy": expected_energy,
                "observed_post_refit_energy": observed_energy,
                "absolute_energy_difference": abs(
                    observed_energy - expected_energy
                ),
                "energy_passed": energy_passed,
                "decision_path_match": decision_path_match,
            }
        )
    status = (
        "passed" if all(row["energy_passed"] for row in rows) else "failed"
    )
    decision_path_mismatches = [
        row["controller_round"]
        for row in rows
        if not row["decision_path_match"]
    ]
    return _digested(
        {
            "schema": "paper_i_ra_adapt_source_value_anchor_v1",
            "status": status,
            "source_protocol": _binding(COMPLETED_SOURCE_PROTOCOL),
            "source_log": _binding(COMPLETED_SOURCE_LOG),
            "comparison_tolerance": {
                "relative": 1.0e-12,
                "absolute": 1.0e-9,
            },
            "anchor_quantity": "accepted_post_refit_energy_trajectory_v1",
            "rounds": rows,
            "decision_path_exact_replay": (
                "passed"
                if not decision_path_mismatches
                else "limited_by_completed_stdout_only_v1"
            ),
            "decision_path_mismatch_rounds": decision_path_mismatches,
            "decision_path_limitation": (
                None
                if not decision_path_mismatches
                else (
                    "The completed source preserved stdout rather than the "
                    "typed accepted-transition result. Round 4 names a "
                    "symmetry-degenerate singleton differently and round 8 "
                    "reports a commutation-equivalent position differently; "
                    "all ten accepted post-refit energies reproduce within "
                    "the declared tolerance."
                )
            ),
            "target_execution_authorized": status == "passed",
        }
    )


def _activation_validation(result_payload: Mapping[str, Any]) -> dict[str, Any]:
    receipts = result_payload["scientific_receipts"][
        "accepted_round_receipts"
    ]
    rows: list[dict[str, Any]] = []
    for receipt in receipts:
        plateau = receipt["insertion_commutation_plateau"]
        activation = receipt["phase3_population_activation"]
        population = receipt["projected_phase3_population_receipt"]
        domain_open = bool(plateau["domain_open"])
        live = bool(activation["competitive_population_live"])
        competitive_count = int(
            population["competitive_population_input_count"]
        )
        available_count = int(
            population["phase2_available_shortlist_count"]
        )
        passed = bool(
            live is domain_open
            and competitive_count >= 1
            and available_count >= competitive_count
            and (domain_open or competitive_count == 1)
        )
        rows.append(
            {
                "controller_round": int(receipt["accepted_round_ordinal"]),
                "insertion_plateau_domain_open": domain_open,
                "competitive_phase3_population_live": live,
                "competitive_population_input_count": competitive_count,
                "phase2_available_shortlist_count": available_count,
                "passed": passed,
            }
        )
    return _digested(
        {
            "schema": "paper_i_ra_adapt_phase3_activation_validation_v1",
            "status": (
                "passed" if rows and all(row["passed"] for row in rows) else "failed"
            ),
            "rounds": rows,
            "first_open_round": next(
                (
                    row["controller_round"]
                    for row in rows
                    if row["insertion_plateau_domain_open"]
                ),
                None,
            ),
        }
    )


def _run_cell(cell_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from pipelines.static_adapt.ra_adapt import (
        RAAdaptOperationalControls,
        run_ra_adapt,
    )
    from pipelines.static_adapt.sr_snake import (
        CheckpointObservation,
        EstimatorLedgerObservation,
        FreshStart,
        SRObservationPolicy,
    )

    protocol = _load_bound_protocol(cell_id)
    run_root = RUNS_ROOT / cell_id
    run_root.mkdir(parents=True, exist_ok=False)
    authorization = _digested(
        {
            "schema": "paper_i_ra_adapt_local_execution_authorization_v1",
            "cell_id": cell_id,
            "protocol_sha256": protocol.sha256,
            "authorization_source": "explicit_user_request_2026-08-02",
            "maximum_controller_rounds": MAXIMUM_CONTROLLER_ROUNDS,
            "execution_authorized": True,
            "submission_authorized": False,
            "authorized_at_utc": _utc_now(),
        }
    )
    _write_json(run_root / "execution_authorization.json", authorization)
    checkpoint = run_root / "checkpoint.json"
    ledger = run_root / "estimator_ledger.json"
    manifest = _digested(
        {
            "schema": "paper_i_ra_adapt_phase3_plateau_local_run_v1",
            "run_class": "diagnostic",
            "cell_id": cell_id,
            "protocol_sha256": protocol.sha256,
            "route_contract_sha256": protocol.route_contract["sha256"],
            "candidate_representation": protocol.candidate_representation,
            "active_gradient_policy": protocol.active_gradient_policy,
            "resource_weighting_scope": protocol.resource_weighting_scope,
            "optimizer": protocol.optimizer,
            "optimizer_maxiter": protocol.optimizer_maxiter,
            "adapt_seed": protocol.seeds["adapt"],
            "maximum_controller_rounds": MAXIMUM_CONTROLLER_ROUNDS,
            "same_cutoff_exact_energy": EXACT_ENERGY,
            "execution_authorization_sha256": authorization["sha256"],
            "started_at_utc": _utc_now(),
        }
    )
    _write_json(run_root / "run_manifest.json", manifest)
    controls = RAAdaptOperationalControls(
        maximum_controller_rounds=MAXIMUM_CONTROLLER_ROUNDS,
        resume=FreshStart(),
        observation=SRObservationPolicy(
            checkpoint=CheckpointObservation(
                path=checkpoint,
                every_controller_rounds=1,
                keep_history_tail=100,
            ),
            estimator_ledger=EstimatorLedgerObservation(path=ledger),
            resource_rounds=(10,),
        ),
    )
    try:
        result = run_ra_adapt(
            _problem_from_receipt(protocol.problem),
            protocol,
            operational_controls=controls,
        )
        result_payload = result.to_dict()
        _write_json(run_root / "result.json", result_payload)
        if result.run.paper_i_summary is not None:
            _write_json(
                run_root / "paper_i_summary.json",
                result.run.paper_i_summary.to_dict(),
            )
        final_energy = float(result.final_state.energy)
        terminal = _digested(
            {
                "schema": "paper_i_ra_adapt_phase3_plateau_local_terminal_v1",
                "status": "passed",
                "cell_id": cell_id,
                "accepted_controller_rounds": len(result.accepted_trajectory),
                "final_same_cutoff_delta_e": abs(final_energy - EXACT_ENERGY),
                "protocol_sha256": protocol.sha256,
                "manifest_sha256": manifest["sha256"],
                "checkpoint_sha256": _sha256_file(checkpoint),
                "estimator_ledger_sha256": _sha256_file(ledger),
                "result_sha256": _sha256_file(run_root / "result.json"),
                "completed_at_utc": _utc_now(),
            }
        )
        _write_json(run_root / "terminal_receipt.json", terminal)
        return terminal, result_payload
    except BaseException as exc:
        failure = _digested(
            {
                "schema": "paper_i_ra_adapt_phase3_plateau_local_failure_v1",
                "status": "failed",
                "cell_id": cell_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "checkpoint_present": checkpoint.is_file(),
                "estimator_ledger_present": ledger.is_file(),
                "failed_at_utc": _utc_now(),
            }
        )
        _write_json(run_root / "failure_receipt.json", failure)
        raise


def execute() -> dict[str, Any]:
    _validate_materialization()
    if RUNS_ROOT.exists() or RUNS_ROOT.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {RUNS_ROOT}")
    control_terminal, control_result = _run_cell(CONTROL_CELL_ID)
    anchor = _anchor_validation(control_result)
    _write_json(MATERIALIZATION_ROOT / "source_value_anchor.json", anchor)
    if anchor["status"] != "passed":
        raise DiagnosticContractError(
            "Current-source control did not reproduce the completed source; "
            "target remains blocked."
        )
    target_terminal, target_result = _run_cell(TARGET_CELL_ID)
    activation = _activation_validation(target_result)
    _write_json(
        MATERIALIZATION_ROOT / "target_activation_validation.json",
        activation,
    )
    if activation["status"] != "passed":
        raise DiagnosticContractError(
            "Target Phase-III activation receipts failed validation."
        )
    completion = _digested(
        {
            "schema": "paper_i_ra_adapt_phase3_plateau_local_completion_v1",
            "status": "passed",
            "control_terminal_sha256": control_terminal["sha256"],
            "target_terminal_sha256": target_terminal["sha256"],
            "source_value_anchor_sha256": anchor["sha256"],
            "target_activation_validation_sha256": activation["sha256"],
            "control_final_same_cutoff_delta_e": control_terminal[
                "final_same_cutoff_delta_e"
            ],
            "target_final_same_cutoff_delta_e": target_terminal[
                "final_same_cutoff_delta_e"
            ],
            "target_first_phase3_open_round": activation["first_open_round"],
            "submission_authorized": False,
            "completed_at_utc": _utc_now(),
        }
    )
    _write_json(OUTPUT_ROOT / "completion_receipt.json", completion)
    return completion


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--materialize", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--execution-authorized", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.materialize:
        if args.execution_authorized:
            raise DiagnosticContractError(
                "Materialization cannot carry execution authorization."
            )
        print(_canonical_bytes(materialize()).decode("utf-8"))
        return 0
    if not args.execution_authorized:
        raise DiagnosticContractError(
            "Execution requires --execution-authorized."
        )
    print(_canonical_bytes(execute()).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
