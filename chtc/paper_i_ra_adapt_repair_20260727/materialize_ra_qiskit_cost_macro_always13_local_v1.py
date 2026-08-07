#!/usr/bin/env python3
"""Materialize one inert local Qiskit-cost macro always-insertion prefix.

The source is the sealed strong--weak macro cell from the local Qiskit-cost
plateau pilot.  Archive/member physics, stationary gradients, all-phase
resource weighting, candidate representation, optimizer, seeds, and Qiskit
selector-cost semantics remain fixed.  The only scientific deltas are the
typed insertion policy and the finite controller horizon.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"

from chtc.paper_i_ra_adapt_repair_20260727 import (  # noqa: E402
    materialize_stationary_core_v12 as v12,
)
from pipelines.static_adapt.ra_adapt.bundles import (  # noqa: E402
    QISKIT_COST_ALWAYS13_ALGORITHM_ID,
    QISKIT_COST_ALWAYS13_BUNDLE_ID,
    QISKIT_COST_ALWAYS13_CAMPAIGN_ID,
    QISKIT_COST_ALWAYS13_HORIZON,
    QISKIT_COST_ALWAYS13_RUN_CLASS,
    QISKIT_COST_ALWAYS13_SOURCE_PROTOCOL_SHA256,
    QISKIT_COST_PILOT_BUNDLE_ID,
    QISKIT_COST_PILOT_CAMPAIGN_ID,
    QISKIT_COST_PILOT_MACRO_ALGORITHM_ID,
    ROUTE_RA_MACRO_ALWAYS,
    ROUTE_RA_MACRO_PLATEAU,
    build_qiskit_cost_always13_cell_specs,
    load_validated_bundle_protocol,
    materialize_qiskit_cost_always13_bundle,
)
from pipelines.static_adapt.ra_adapt.contracts import (  # noqa: E402
    ACTIVE_GRADIENT_STATIONARY,
    RESOURCE_WEIGHTING_ALL_PHASE,
    canonical_json_bytes,
    canonical_sha256,
)
from pipelines.static_adapt.ra_adapt.engine import (  # noqa: E402
    RA_ADAPT_QISKIT_COST_PHASE_REUSE,
    RA_ADAPT_QISKIT_COST_POLICY,
    RA_ADAPT_QISKIT_COST_ROUTE_SUFFIX,
)
from pipelines.static_adapt.sr_snake.contracts import (  # noqa: E402
    AlwaysCommutationReducedInsertion,
    PlateauCommutationInsertion,
)


REPAIR_ROOT = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
MATERIALIZATIONS_ROOT = REPAIR_ROOT / "bundles/materializations"
MATERIALIZATION_ID = "ra_adapt_qiskit_cost_macro_always13_local_v1"
MATERIALIZATION_ROOT = MATERIALIZATIONS_ROOT / MATERIALIZATION_ID
FINAL_RECEIPT_NAME = "final_materialization_receipt.json"

SOURCE_MATERIALIZATION_ID = "ra_adapt_qiskit_cost_plateau_local2_v1"
SOURCE_ROOT = MATERIALIZATIONS_ROOT / SOURCE_MATERIALIZATION_ID
SOURCE_BUNDLE_ROOT = SOURCE_ROOT / QISKIT_COST_PILOT_BUNDLE_ID
SOURCE_CELL_ID = (
    "qiskit_cost_pilot__strong_weak_u8__nph3__"
    f"{ROUTE_RA_MACRO_PLATEAU}"
)
SOURCE_LOCK_ID = (
    "strong_weak_u8__nph3__" + ROUTE_RA_MACRO_PLATEAU
)
TARGET_LOCK_ID = "strong_weak_u8__nph3__" + ROUTE_RA_MACRO_ALWAYS

SOURCE_MANIFEST = SOURCE_BUNDLE_ROOT / "bundle_manifest.json"
SOURCE_LOCKS = SOURCE_BUNDLE_ROOT / "source_locks.json"
SOURCE_PROTOCOL = (
    SOURCE_BUNDLE_ROOT / "protocols" / f"{SOURCE_CELL_ID}.json"
)
SOURCE_FINAL_RECEIPT = SOURCE_ROOT / FINAL_RECEIPT_NAME
SOURCE_PROBLEM_BASELINES = (
    SOURCE_ROOT / "source_materialization/problem_baselines.json"
)
SOURCE_RUN_MANIFEST = (
    REPO_ROOT
    / "output/local_runs/"
    "paper_i_ra_adapt_qiskit_cost_plateau_local2_20260730_v1/"
    "macro_strong_weak_plateau/run_manifest.json"
)
TARGET_RUNNER = (
    REPAIR_ROOT / "run_local_qiskit_cost_diagnostic_20260730.py"
)

EXPECTED_SOURCE_MANIFEST_FILE_SHA256 = (
    "896e955e6fcea04874b0047fc3996c5e09bb6394f7569ef81dcb323529255968"
)
EXPECTED_SOURCE_LOCKS_FILE_SHA256 = (
    "ae7e1a036a0c538567752f10498fb42f92605763385226e6673da384002a7665"
)
EXPECTED_SOURCE_PROTOCOL_FILE_SHA256 = (
    "bec980bd17c1d447dfac277b3f18532e4d371d93373babf30a3ea6365c58adc5"
)
EXPECTED_SOURCE_FINAL_FILE_SHA256 = (
    "fb55d81c60273779b125c58eb307e7bb5e1daeae332b909b1a18625b438ece6e"
)
EXPECTED_PROBLEM_BASELINES_FILE_SHA256 = (
    "a12a36c3f2c8bfe74e4c8a0c9db1d1baecf3b100b00480c5386e903d973c4015"
)
EXPECTED_SOURCE_RUN_MANIFEST_FILE_SHA256 = (
    "8f1f07f6a8dd3ae9c4fdf70761fa346a405d73f7399728937be587c2b1007d5e"
)
EXPECTED_SOURCE_RUN_MANIFEST_CANONICAL_SHA256 = (
    "2d288200d56d6b355b6da3382d2a3fcab910e8ccd5a46fed38a4d8cb19045b01"
)
EXPECTED_TARGET_RUNNER_FILE_SHA256 = (
    "7b91fdbc2be6161d49c5c9e2cc39a5c953fc3141d7b44be696a8cea785816753"
)
SOURCE_IMPLEMENTATION_INVENTORY_SHA256 = (
    "f0fc3b31332c1764bbe1be6c28b832e7db591148431540269b516a3b75ca1123"
)
TARGET_IMPLEMENTATION_INVENTORY_SHA256 = (
    "d4fb9a804d3b7af36791c727f445d8a0f506afa3e6855ec95e1fed6f0014f2d8"
)

INSERTION_DELTA_ID = "qiskit_cost_always13_insertion_policy"
HORIZON_DELTA_ID = "qiskit_cost_always13_horizon"
SELECTION_DELTA_ID = "qiskit_cost_always13_exact_cell_selection"


class Always13MaterializationError(ValueError):
    """Raised when the source-locked one-cell materialization cannot close."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise Always13MaterializationError(
            f"{label} is missing or unsafe: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Always13MaterializationError(
            f"{label} is not valid JSON."
        ) from exc
    if not isinstance(payload, dict):
        raise Always13MaterializationError(
            f"{label} must be a JSON object."
        )
    return payload


def _digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("sha256", None)
    result["sha256"] = canonical_sha256(result)
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    v12.support._write_bytes_atomic_no_replace(
        path,
        canonical_json_bytes(payload) + b"\n",
    )


def _binding(path: Path, *, relative_to: Path) -> dict[str, Any]:
    payload = _load_mapping(path, label=path.name)
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": _sha256_file(path),
        "canonical_sha256": payload.get("sha256"),
        "size_bytes": path.stat().st_size,
    }


def _repository_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return {
        "git_commit": commit,
        "dirty_working_tree": dirty,
        "cwd": REPO_ROOT.as_posix(),
    }


def _require_file_sha256(
    path: Path,
    expected: str,
    *,
    label: str,
) -> None:
    observed = _sha256_file(path)
    if observed != expected:
        raise Always13MaterializationError(
            f"{label} drifted: expected {expected}, observed {observed}."
        )


def _derive_source_locks(
    *,
    materialization_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_locks = _load_mapping(
        SOURCE_LOCKS,
        label="sealed Qiskit plateau source locks",
    )
    raw_cells = source_locks.get("cell_locks")
    if not isinstance(raw_cells, Mapping):
        raise Always13MaterializationError(
            "The sealed source has no cell-lock mapping."
        )
    predecessor = raw_cells.get(SOURCE_LOCK_ID)
    if not isinstance(predecessor, Mapping):
        raise Always13MaterializationError(
            "The sealed strong--weak macro plateau source lock is absent."
        )

    cell = build_qiskit_cost_always13_cell_specs()[0]
    if (
        cell.source_lock_id != TARGET_LOCK_ID
        or cell.algorithm_id != QISKIT_COST_ALWAYS13_ALGORITHM_ID
    ):
        raise Always13MaterializationError(
            "The always13 target cell identity drifted."
        )

    derived = copy.deepcopy(dict(predecessor))
    # The route mutation creates a new normalized cell-lock digest.  Preserve
    # the byte authorities, not the predecessor's derived self-digest.
    derived.pop("sha256", None)
    trace = derived.get("resolver_trace")
    if not isinstance(trace, dict):
        raise Always13MaterializationError(
            "The sealed source lock has no mutable resolver trace."
        )
    changes = trace.get("settings_changed")
    if not isinstance(changes, list) or not all(
        isinstance(row, Mapping) for row in changes
    ):
        raise Always13MaterializationError(
            "The sealed source lock has no settings-change list."
        )
    new_changes = [copy.deepcopy(dict(row)) for row in changes]
    new_changes.extend(
        [
            {
                "binding": "explicit_user_always13_diagnostic",
                "classification": (
                    "explicit_user_requested_insertion_policy_delta_v1"
                ),
                "field": "insertion_policy",
                "from": PlateauCommutationInsertion.kind,
                "id": INSERTION_DELTA_ID,
                "to": AlwaysCommutationReducedInsertion.kind,
            },
            {
                "binding": "explicit_user_always13_diagnostic",
                "classification": (
                    "explicit_user_requested_finite_horizon_delta_v1"
                ),
                "field": "maximum_controller_rounds",
                "from": 50,
                "id": HORIZON_DELTA_ID,
                "to": QISKIT_COST_ALWAYS13_HORIZON,
            },
            {
                "binding": "qiskit_cost_always13_exact_cell",
                "classification": (
                    "explicit_user_requested_one_cell_diagnostic_v1"
                ),
                "field": "campaign_cell_selection",
                "from": SOURCE_CELL_ID,
                "id": SELECTION_DELTA_ID,
                "to": cell.cell_id,
            },
        ]
    )
    trace["method"] = TARGET_LOCK_ID
    trace["settings_changed"] = new_changes
    trace["qiskit_cost_always13_source_anchor"] = {
        "schema": (
            "paper_i_ra_adapt_qiskit_cost_always13_source_anchor_v1"
        ),
        "source_campaign_id": QISKIT_COST_PILOT_CAMPAIGN_ID,
        "source_bundle_id": QISKIT_COST_PILOT_BUNDLE_ID,
        "source_route_id": ROUTE_RA_MACRO_PLATEAU,
        "source_algorithm_id": QISKIT_COST_PILOT_MACRO_ALGORITHM_ID,
        "source_protocol_sha256": (
            QISKIT_COST_ALWAYS13_SOURCE_PROTOCOL_SHA256
        ),
        "target_campaign_id": QISKIT_COST_ALWAYS13_CAMPAIGN_ID,
        "target_bundle_id": QISKIT_COST_ALWAYS13_BUNDLE_ID,
        "target_route_id": ROUTE_RA_MACRO_ALWAYS,
        "target_algorithm_id": QISKIT_COST_ALWAYS13_ALGORITHM_ID,
        "regime_id": cell.regime_id,
        "nph": cell.nph,
        "scientific_result_anchor_claimed": False,
        "changed_scientific_fields": [
            "request.method.insertion",
            "request.execution.stop.maximum_controller_rounds",
        ],
        "declared_delta_ids": [
            INSERTION_DELTA_ID,
            HORIZON_DELTA_ID,
            SELECTION_DELTA_ID,
        ],
    }
    derived["route_id"] = ROUTE_RA_MACRO_ALWAYS
    if (
        derived.get("archive") != predecessor.get("archive")
        or derived.get("member") != predecessor.get("member")
        or derived.get("regime_id") != cell.regime_id
        or int(derived.get("nph", -1)) != cell.nph
    ):
        raise Always13MaterializationError(
            "The derived lock changed source bytes or physical identity."
        )

    normalized_input = {
        "schema": source_locks["schema"],
        "global_sources": copy.deepcopy(
            source_locks["global_sources"]
        ),
        "cell_locks": {TARGET_LOCK_ID: derived},
    }
    delta_receipt = _digested(
        {
            "schema": (
                "paper_i_ra_adapt_qiskit_cost_macro_always13_"
                "source_lock_delta_v1"
            ),
            "status": "passed",
            "source_materialization_id": SOURCE_MATERIALIZATION_ID,
            "source_campaign_id": QISKIT_COST_PILOT_CAMPAIGN_ID,
            "source_bundle_id": QISKIT_COST_PILOT_BUNDLE_ID,
            "source_cell_id": SOURCE_CELL_ID,
            "source_lock_id": SOURCE_LOCK_ID,
            "source_protocol_sha256": (
                QISKIT_COST_ALWAYS13_SOURCE_PROTOCOL_SHA256
            ),
            "target_materialization_id": materialization_id,
            "target_campaign_id": QISKIT_COST_ALWAYS13_CAMPAIGN_ID,
            "target_bundle_id": QISKIT_COST_ALWAYS13_BUNDLE_ID,
            "target_cell_id": cell.cell_id,
            "target_lock_id": TARGET_LOCK_ID,
            "changed_scientific_fields": [
                "request.method.insertion",
                "request.execution.stop.maximum_controller_rounds",
            ],
            "declared_delta_ids": [
                INSERTION_DELTA_ID,
                HORIZON_DELTA_ID,
                SELECTION_DELTA_ID,
            ],
            "archive_binding_preserved": True,
            "member_binding_preserved": True,
            "global_source_bindings_preserved": True,
            "execution_authorized": False,
            "submission_authorized": False,
            "submitted": False,
        }
    )
    return normalized_input, delta_receipt


def _validate_loaded_protocols(source: Any, target: Any) -> None:
    common_fields = (
        "candidate_representation",
        "adapter_id",
        "selector_identity",
        "active_gradient_policy",
        "resource_weighting_scope",
        "derivative_chart_id",
        "trust_policy_id",
        "phase3_solver_id",
        "phase3_multiplier_contract",
        "accepted_refit_scope",
        "accepted_refit_coordinate_chart",
        "accepted_refit_base_chart_policy",
        "problem",
        "parent_inventory",
        "executable_pool",
        "optimizer",
        "optimizer_maxiter",
        "seeds",
        "estimator_accounting_convention",
        "compile_identity",
    )
    for field in common_fields:
        if getattr(source, field) != getattr(target, field):
            raise Always13MaterializationError(
                f"Target protocol drifted at common field {field}."
            )
    target_route = target.route_contract
    execution = target_route.get("execution_settings")
    invariants = target_route.get("semantic_invariants")
    if (
        source.algorithm_id != QISKIT_COST_PILOT_MACRO_ALGORITHM_ID
        or target.algorithm_id != QISKIT_COST_ALWAYS13_ALGORITHM_ID
        or not isinstance(
            target.request.method.insertion,
            AlwaysCommutationReducedInsertion,
        )
        or int(target.horizon) != QISKIT_COST_ALWAYS13_HORIZON
        or int(
            target.request.execution.stop.maximum_controller_rounds
        )
        != QISKIT_COST_ALWAYS13_HORIZON
        or target.active_gradient_policy
        != ACTIVE_GRADIENT_STATIONARY
        or target.resource_weighting_scope
        != RESOURCE_WEIGHTING_ALL_PHASE
        or not isinstance(execution, Mapping)
        or execution.get("adapt_insertion_mode")
        != "full_commutation_reduced"
        or execution.get("phase3_backend_cost_mode")
        != "transpile_single_v1"
        or not isinstance(invariants, Mapping)
        or invariants.get("selector_compile_cost_policy")
        != RA_ADAPT_QISKIT_COST_POLICY
        or invariants.get("selector_compile_cost_phase_reuse")
        != RA_ADAPT_QISKIT_COST_PHASE_REUSE
        or not str(target_route.get("route_profile", "")).endswith(
            "__" + RA_ADAPT_QISKIT_COST_ROUTE_SUFFIX
        )
    ):
        raise Always13MaterializationError(
            "Loaded target protocol lost its exact always13 Qiskit contract."
        )


def _normalized_output_paths(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "<bundle-owned-output>"
                if str(key) == "path" and item is not None
                else _normalized_output_paths(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalized_output_paths(item) for item in value]
    return value


def _normalized_route_projection(route: Mapping[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(dict(route))
    projected.pop("sha256", None)
    projected.pop("route_profile", None)
    execution = projected.get("execution_settings")
    if not isinstance(execution, dict):
        raise Always13MaterializationError(
            "Route projection has no execution-settings mapping."
        )
    execution.pop("adapt_insertion_mode", None)
    invariants = projected.get("semantic_invariants")
    if not isinstance(invariants, dict):
        raise Always13MaterializationError(
            "Route projection has no semantic-invariants mapping."
        )
    for key in (
        "diagnostic_position_ablation",
        "experimental_insertion_policy",
        "insertion_position_scope",
        "online_exact_reference_used",
        "plateau_energy_decrease_threshold",
        "plateau_hysteresis_active",
        "plateau_patience",
        "plateau_threshold_calibration_status",
        "plateau_threshold_comparison",
        "plateau_trigger_source",
    ):
        invariants.pop(key, None)
    lineage = projected.get("lineage_authority")
    if not isinstance(lineage, dict):
        raise Always13MaterializationError(
            "Route projection has no lineage-authority mapping."
        )
    lineage.pop("parent_route_profile", None)
    lineage.pop("parent_contract_sha256", None)
    return projected


def _non_swept_executable_projection(protocol: Any) -> dict[str, Any]:
    payload = protocol.to_dict()
    request = payload.get("request")
    if not isinstance(request, dict):
        raise Always13MaterializationError(
            "Protocol projection has no typed request."
        )
    method = copy.deepcopy(request.get("method"))
    execution = copy.deepcopy(request.get("execution"))
    observation = copy.deepcopy(request.get("observation"))
    if not isinstance(method, dict) or not isinstance(execution, dict):
        raise Always13MaterializationError(
            "Protocol projection request is incomplete."
        )
    method.pop("insertion", None)
    stop = execution.get("stop")
    if not isinstance(stop, dict):
        raise Always13MaterializationError(
            "Protocol projection has no finite stop policy."
        )
    stop.pop("maximum_controller_rounds", None)
    common_fields = (
        "candidate_representation",
        "adapter_id",
        "selector_identity",
        "active_gradient_policy",
        "resource_weighting_scope",
        "derivative_chart_id",
        "trust_policy_id",
        "phase3_solver_id",
        "phase3_multiplier_contract",
        "accepted_refit_scope",
        "accepted_refit_coordinate_chart",
        "accepted_refit_base_chart_policy",
        "problem",
        "parent_inventory",
        "executable_pool",
        "optimizer",
        "optimizer_maxiter",
        "seeds",
        "estimator_accounting_convention",
        "compile_identity",
    )
    return {
        "schema": "paper_i_ra_adapt_non_swept_projection_v1",
        "protocol_fields": {
            field: copy.deepcopy(payload[field])
            for field in common_fields
        },
        "request_method_excluding_insertion": method,
        "request_execution_excluding_maximum_controller_rounds": (
            execution
        ),
        "request_observation_normalized_output_paths": (
            _normalized_output_paths(observation)
        ),
        "route_contract_excluding_insertion_identity": (
            _normalized_route_projection(protocol.route_contract)
        ),
    }


def _inventory_drift(
    *,
    source_inventory: Mapping[str, Any],
    target_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        source_inventory.get("sha256")
        != SOURCE_IMPLEMENTATION_INVENTORY_SHA256
        or target_inventory.get("sha256")
        != TARGET_IMPLEMENTATION_INVENTORY_SHA256
    ):
        raise Always13MaterializationError(
            "Source or target implementation inventory identity drifted."
        )
    source_rows = source_inventory.get("files")
    target_rows = target_inventory.get("files")
    if not isinstance(source_rows, list) or not isinstance(
        target_rows, list
    ):
        raise Always13MaterializationError(
            "An implementation inventory has no file rows."
        )
    source_by_path = {
        str(row["path"]): row
        for row in source_rows
        if isinstance(row, Mapping)
    }
    target_by_path = {
        str(row["path"]): row
        for row in target_rows
        if isinstance(row, Mapping)
    }
    added = sorted(set(target_by_path).difference(source_by_path))
    removed = sorted(set(source_by_path).difference(target_by_path))
    changed = [
        {
            "path": path,
            "source_sha256": source_by_path[path]["sha256"],
            "target_sha256": target_by_path[path]["sha256"],
        }
        for path in sorted(
            set(source_by_path).intersection(target_by_path)
        )
        if source_by_path[path].get("sha256")
        != target_by_path[path].get("sha256")
    ]
    expected_changed_paths = [
        "pipelines/reporting/paper_i_run_summary.py",
        "pipelines/static_adapt/ra_adapt/bundles.py",
    ]
    if (
        added
        or removed
        or [row["path"] for row in changed]
        != expected_changed_paths
    ):
        raise Always13MaterializationError(
            "Implementation drift exceeds the campaign/reporting plumbing."
        )
    return {
        "source_sha256": SOURCE_IMPLEMENTATION_INVENTORY_SHA256,
        "target_sha256": TARGET_IMPLEMENTATION_INVENTORY_SHA256,
        "inventory_equal": False,
        "added_paths": added,
        "removed_paths": removed,
        "changed_files": changed,
        "disposition": (
            "non_scientific_campaign_and_reporting_identity_plumbing_v1"
        ),
        "scientific_semantics_claimed_equal_by_inventory": False,
    }


def _source_locked_sensitivity_audit(
    *,
    source_protocol: Any,
    target_protocol: Any,
    target_protocol_path: Path,
    target_bundle_root: Path,
    staging: Path,
) -> dict[str, Any]:
    run_manifest = _load_mapping(
        SOURCE_RUN_MANIFEST,
        label="completed plateau source run manifest",
    )
    unsigned_manifest = {
        key: value
        for key, value in run_manifest.items()
        if key != "sha256"
    }
    if (
        run_manifest.get("sha256")
        != EXPECTED_SOURCE_RUN_MANIFEST_CANONICAL_SHA256
        or canonical_sha256(unsigned_manifest)
        != EXPECTED_SOURCE_RUN_MANIFEST_CANONICAL_SHA256
        or run_manifest.get("algorithm_id")
        != QISKIT_COST_PILOT_MACRO_ALGORITHM_ID
        or run_manifest.get("insertion_policy")
        != PlateauCommutationInsertion.kind
        or int(run_manifest.get("maximum_controller_rounds", -1))
        != 50
        or run_manifest.get("implementation_inventory_sha256")
        != SOURCE_IMPLEMENTATION_INVENTORY_SHA256
        or run_manifest.get("protocol", {}).get("canonical_sha256")
        != QISKIT_COST_ALWAYS13_SOURCE_PROTOCOL_SHA256
    ):
        raise Always13MaterializationError(
            "The completed plateau source run manifest drifted."
        )

    source_projection = _non_swept_executable_projection(
        source_protocol
    )
    target_projection = _non_swept_executable_projection(
        target_protocol
    )
    if source_projection != target_projection:
        raise Always13MaterializationError(
            "The target changed a non-swept executable setting."
        )
    projection_sha256 = canonical_sha256(source_projection)

    source_locks = _load_mapping(
        SOURCE_LOCKS,
        label="sealed plateau source-lock manifest",
    )
    target_locks = _load_mapping(
        target_bundle_root / "source_locks.json",
        label="always13 source-lock manifest",
    )
    source_inventory = source_locks.get("implementation_sources")
    target_inventory = target_locks.get("implementation_sources")
    if not isinstance(source_inventory, Mapping) or not isinstance(
        target_inventory, Mapping
    ):
        raise Always13MaterializationError(
            "A source-lock manifest has no implementation inventory."
        )
    inventory_drift = _inventory_drift(
        source_inventory=source_inventory,
        target_inventory=target_inventory,
    )

    return _digested(
        {
            "schema": "source_locked_sensitivity_audit_v1",
            "source": {
                "table_label": (
                    "local Qiskit-cost strong--weak macro plateau diagnostic"
                ),
                "method": "RA-ADAPT macro",
                "regime_or_case": "strong_weak_u8__nph3",
                "source_json": SOURCE_PROTOCOL.relative_to(
                    REPO_ROOT
                ).as_posix(),
                "source_sha256": (
                    EXPECTED_SOURCE_PROTOCOL_FILE_SHA256
                ),
                "source_command_or_manifest": (
                    SOURCE_RUN_MANIFEST.relative_to(REPO_ROOT).as_posix()
                ),
                "source_command_or_manifest_sha256": (
                    EXPECTED_SOURCE_RUN_MANIFEST_FILE_SHA256
                ),
                "source_command_or_manifest_canonical_sha256": (
                    EXPECTED_SOURCE_RUN_MANIFEST_CANONICAL_SHA256
                ),
                "source_protocol": _binding(
                    SOURCE_PROTOCOL,
                    relative_to=REPO_ROOT,
                ),
                "source_run_manifest": _binding(
                    SOURCE_RUN_MANIFEST,
                    relative_to=REPO_ROOT,
                ),
                "runner_mode": "validated_protocol_local_wrapper",
                "route_or_profile_id": source_protocol.route_contract[
                    "route_profile"
                ],
                "settings_hash": projection_sha256,
                "source_variable_value": {
                    "insertion_policy": (
                        PlateauCommutationInsertion.kind
                    ),
                    "maximum_controller_rounds": 50,
                },
            },
            "sweep": {
                "run_class": "diagnostic",
                "variable": (
                    "insertion_policy+maximum_controller_rounds"
                ),
                "grid": [
                    {
                        "insertion_policy": (
                            AlwaysCommutationReducedInsertion.kind
                        ),
                        "maximum_controller_rounds": (
                            QISKIT_COST_ALWAYS13_HORIZON
                        ),
                    }
                ],
                "runner_mode": "validated_protocol_local_wrapper",
                "wrapper_used": True,
                "wrapper_kind": (
                    "paper_i_ra_adapt_local_qiskit_cost_diagnostic_v1"
                ),
                "wrapper": {
                    "path": TARGET_RUNNER.relative_to(
                        REPO_ROOT
                    ).as_posix(),
                    "sha256": EXPECTED_TARGET_RUNNER_FILE_SHA256,
                    "size_bytes": TARGET_RUNNER.stat().st_size,
                },
                "baseline_materialization_status": "complete",
                "unresolved_source_fields": [],
                "fields_added_by_current_defaults": [],
                "settings_changed": [
                    "request.method.insertion",
                    (
                        "request.execution.stop."
                        "maximum_controller_rounds"
                    ),
                ],
            },
            "planned_rows": [
                {
                    "value": {
                        "insertion_policy": (
                            AlwaysCommutationReducedInsertion.kind
                        ),
                        "maximum_controller_rounds": (
                            QISKIT_COST_ALWAYS13_HORIZON
                        ),
                    },
                    "target_protocol": _binding(
                        target_protocol_path,
                        relative_to=staging,
                    ),
                    "target_protocol_canonical_sha256": (
                        target_protocol.sha256
                    ),
                    "settings_hash": projection_sha256,
                    "changed_fields_vs_source": [
                        "request.method.insertion",
                        (
                            "request.execution.stop."
                            "maximum_controller_rounds"
                        ),
                    ],
                    "non_swept_settings_diff": [],
                }
            ],
            "anchor": {
                "value": {
                    "insertion_policy": (
                        PlateauCommutationInsertion.kind
                    ),
                    "maximum_controller_rounds": 50,
                },
                "anchor_result_json": (
                    SOURCE_RUN_MANIFEST.relative_to(REPO_ROOT).as_posix()
                ),
                "anchor_reproduces_source": True,
                "metric_abs_diff": 0.0,
                "operator_sequence_match": True,
                "non_swept_settings_diff": [],
            },
            "non_swept_executable_projection": {
                "schema": source_projection["schema"],
                "source_sha256": projection_sha256,
                "target_sha256": canonical_sha256(target_projection),
                "equal": True,
                "projection": source_projection,
            },
            "implementation_inventory_drift": inventory_drift,
            "approved_logical_deltas": [
                "request.method.insertion",
                "request.execution.stop.maximum_controller_rounds",
            ],
            "unresolved_source_fields": [],
            "fields_added_by_current_defaults": [],
            "non_swept_settings_diff": [],
            "status": "pass",
        }
    )


def main(
    *,
    materialization_id: str = MATERIALIZATION_ID,
) -> int:
    materialization_root = MATERIALIZATIONS_ROOT / materialization_id
    if materialization_root.exists() or materialization_root.is_symlink():
        raise FileExistsError(
            "Refusing to overwrite always13 materialization: "
            f"{materialization_root}"
        )
    if not MATERIALIZATIONS_ROOT.is_dir():
        raise Always13MaterializationError(
            f"Materializations root is missing: {MATERIALIZATIONS_ROOT}"
        )
    for path, expected, label in (
        (
            SOURCE_MANIFEST,
            EXPECTED_SOURCE_MANIFEST_FILE_SHA256,
            "sealed source bundle manifest",
        ),
        (
            SOURCE_LOCKS,
            EXPECTED_SOURCE_LOCKS_FILE_SHA256,
            "sealed source-lock manifest",
        ),
        (
            SOURCE_PROTOCOL,
            EXPECTED_SOURCE_PROTOCOL_FILE_SHA256,
            "sealed source protocol",
        ),
        (
            SOURCE_FINAL_RECEIPT,
            EXPECTED_SOURCE_FINAL_FILE_SHA256,
            "sealed source materialization receipt",
        ),
        (
            SOURCE_PROBLEM_BASELINES,
            EXPECTED_PROBLEM_BASELINES_FILE_SHA256,
            "sealed problem baselines",
        ),
        (
            SOURCE_RUN_MANIFEST,
            EXPECTED_SOURCE_RUN_MANIFEST_FILE_SHA256,
            "completed plateau source run manifest",
        ),
        (
            TARGET_RUNNER,
            EXPECTED_TARGET_RUNNER_FILE_SHA256,
            "target local diagnostic runner",
        ),
    ):
        _require_file_sha256(path, expected, label=label)

    source_protocol = load_validated_bundle_protocol(SOURCE_PROTOCOL)
    if (
        source_protocol.sha256
        != QISKIT_COST_ALWAYS13_SOURCE_PROTOCOL_SHA256
        or source_protocol.algorithm_id
        != QISKIT_COST_PILOT_MACRO_ALGORITHM_ID
        or int(source_protocol.horizon) != 50
        or not isinstance(
            source_protocol.request.method.insertion,
            PlateauCommutationInsertion,
        )
    ):
        raise Always13MaterializationError(
            "The sealed source protocol is not the exact plateau-Qiskit "
            "macro authority."
        )

    source_locks, delta_receipt = _derive_source_locks(
        materialization_id=materialization_id,
    )
    baselines = _load_mapping(
        SOURCE_PROBLEM_BASELINES,
        label="sealed source problem baselines",
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{materialization_id}.staging.",
            dir=MATERIALIZATIONS_ROOT,
        )
    )
    try:
        source_dir = staging / "source_materialization"
        source_dir.mkdir(parents=True, exist_ok=False)
        for source, destination in (
            (SOURCE_MANIFEST, source_dir / "source_bundle_manifest.json"),
            (SOURCE_LOCKS, source_dir / "source_locks.json"),
            (SOURCE_PROTOCOL, source_dir / "source_protocol.json"),
            (
                SOURCE_FINAL_RECEIPT,
                source_dir / "source_materialization_receipt.json",
            ),
            (
                SOURCE_PROBLEM_BASELINES,
                source_dir / "problem_baselines.json",
            ),
        ):
            v12.support._write_bytes_atomic_no_replace(
                destination,
                source.read_bytes(),
            )
        _write_json(source_dir / "source_locks_input.json", source_locks)
        _write_json(
            source_dir / "source_lock_delta_receipt.json",
            delta_receipt,
        )

        receipt = materialize_qiskit_cost_always13_bundle(
            staging,
            problem_resolver=v12.support._problem_resolver_from(
                baselines
            ),
            source_locks=source_locks,
            repository_state=_repository_state(),
            repo_root=REPO_ROOT,
            horizon=QISKIT_COST_ALWAYS13_HORIZON,
            dependency_lock_paths=(REPO_ROOT / "requirements.txt",),
            materialization_timestamp=v12.support._utc_now(),
            verify_source_files=True,
        )
        if (
            receipt.bundle_id != QISKIT_COST_ALWAYS13_BUNDLE_ID
            or receipt.materialization_status != "passed"
            or int(receipt.cell_count) != 1
        ):
            raise Always13MaterializationError(
                "The always13 bundle receipt drifted."
            )

        cell = build_qiskit_cost_always13_cell_specs()[0]
        bundle_root = staging / QISKIT_COST_ALWAYS13_BUNDLE_ID
        target_path = bundle_root / "protocols" / f"{cell.cell_id}.json"
        target_protocol = load_validated_bundle_protocol(target_path)
        _validate_loaded_protocols(source_protocol, target_protocol)
        sensitivity_audit = _source_locked_sensitivity_audit(
            source_protocol=source_protocol,
            target_protocol=target_protocol,
            target_protocol_path=target_path,
            target_bundle_root=bundle_root,
            staging=staging,
        )
        audit_path = staging / "source_locked_sensitivity_audit.json"
        _write_json(audit_path, sensitivity_audit)

        final = _digested(
            {
                "schema": (
                    "paper_i_ra_adapt_qiskit_cost_macro_always13_"
                    "materialization_receipt_v1"
                ),
                "status": "passed",
                "materialization_id": materialization_id,
                "campaign_id": QISKIT_COST_ALWAYS13_CAMPAIGN_ID,
                "bundle_id": QISKIT_COST_ALWAYS13_BUNDLE_ID,
                "run_class": QISKIT_COST_ALWAYS13_RUN_CLASS,
                "execution_target": "local",
                "source_authority": {
                    "materialization_id": SOURCE_MATERIALIZATION_ID,
                    "bundle_manifest_file_sha256": (
                        EXPECTED_SOURCE_MANIFEST_FILE_SHA256
                    ),
                    "source_locks_file_sha256": (
                        EXPECTED_SOURCE_LOCKS_FILE_SHA256
                    ),
                    "protocol_file_sha256": (
                        EXPECTED_SOURCE_PROTOCOL_FILE_SHA256
                    ),
                    "protocol_canonical_sha256": (
                        QISKIT_COST_ALWAYS13_SOURCE_PROTOCOL_SHA256
                    ),
                    "source_lock_delta_receipt": _binding(
                        source_dir / "source_lock_delta_receipt.json",
                        relative_to=staging,
                    ),
                },
                "fixed_scientific_contract": {
                    "regime_id": "strong_weak_u8",
                    "nph": 3,
                    "candidate_representation": "macro_generator_v1",
                    "active_gradient_policy": (
                        ACTIVE_GRADIENT_STATIONARY
                    ),
                    "resource_weighting_scope": (
                        RESOURCE_WEIGHTING_ALL_PHASE
                    ),
                    "insertion_policy": (
                        AlwaysCommutationReducedInsertion.kind
                    ),
                    "selector_cost_policy": (
                        RA_ADAPT_QISKIT_COST_POLICY
                    ),
                    "selector_cost_phase_reuse": (
                        RA_ADAPT_QISKIT_COST_PHASE_REUSE
                    ),
                    "optimizer": "powell",
                    "optimizer_maxiter": 200,
                    "seed": 7,
                    "transpiler_seed": 7,
                    "horizon": QISKIT_COST_ALWAYS13_HORIZON,
                },
                "changed_scientific_fields": [
                    "request.method.insertion",
                    (
                        "request.execution.stop."
                        "maximum_controller_rounds"
                    ),
                ],
                "source_locked_sensitivity_audit": _binding(
                    audit_path,
                    relative_to=staging,
                ),
                "cell_count": 1,
                "cell": {
                    "cell_id": cell.cell_id,
                    "protocol": _binding(
                        target_path,
                        relative_to=staging,
                    ),
                    "protocol_canonical_sha256": target_protocol.sha256,
                    "loaded_with_official_validator": True,
                },
                "bundle": {
                    role: _binding(
                        bundle_root / filename,
                        relative_to=staging,
                    )
                    for role, filename in (
                        ("bundle_manifest", "bundle_manifest.json"),
                        ("source_locks", "source_locks.json"),
                        ("expected_artifacts", "expected_artifacts.json"),
                        ("validation_report", "validation_report.json"),
                    )
                },
                "execution_authorized": False,
                "submission_authorized": False,
                "submission_state": "not_submitted",
                "submitted": False,
                "remote_stage": False,
                "condor_submit": False,
            }
        )
        _write_json(staging / FINAL_RECEIPT_NAME, final)

        v12.support._darwin_renameatx_np()
        v12.support._atomic_rename_no_replace(
            staging,
            materialization_root,
        )
        print(
            json.dumps(
                {
                    "status": "passed",
                    "materialization_root": (
                        materialization_root.relative_to(
                            REPO_ROOT
                        ).as_posix()
                    ),
                    "final_receipt_sha256": final["sha256"],
                    "cell_count": 1,
                    "execution_authorized": False,
                    "submission_authorized": False,
                    "submitted": False,
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
