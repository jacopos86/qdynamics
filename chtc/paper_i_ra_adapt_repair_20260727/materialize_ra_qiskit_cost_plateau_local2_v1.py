#!/usr/bin/env python3
"""Materialize the inert two-cell local Qiskit-cost plateau pilot.

The pilot is deliberately narrow: one strong--weak macro plateau cell and one
strong--strong global-singleton plateau cell.  It derives both cells from
already sealed Paper-I authorities, changes only the declared selector-cost
oracle (and, for the macro cell, its application scope), and leaves execution
authorization to the separate local runner.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
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
    CORE_BUNDLE_ID,
    CORE_CAMPAIGN_ID,
    GLOBAL_SINGLETON_BUNDLE_ID,
    GLOBAL_SINGLETON_CAMPAIGN_ID,
    QISKIT_COST_PILOT_BUNDLE_ID,
    QISKIT_COST_PILOT_CAMPAIGN_ID,
    QISKIT_COST_PILOT_GLOBAL_SINGLETON_ALGORITHM_ID,
    QISKIT_COST_PILOT_MACRO_ALGORITHM_ID,
    QISKIT_COST_PILOT_RUN_CLASS,
    ROUTE_RA_GLOBAL_SINGLETON_PLATEAU,
    ROUTE_RA_MACRO_PLATEAU,
    build_qiskit_cost_plateau_pilot_cell_specs,
    load_validated_bundle_protocol,
    materialize_qiskit_cost_plateau_pilot_bundle,
)
from pipelines.static_adapt.ra_adapt.contracts import (  # noqa: E402
    ACTIVE_GRADIENT_STATIONARY,
    RESOURCE_WEIGHTING_ALL_PHASE,
    RESOURCE_WEIGHTING_LATE,
    canonical_json_bytes,
    canonical_sha256,
)
from pipelines.static_adapt.ra_adapt.engine import (  # noqa: E402
    RA_ADAPT_QISKIT_COST_PHASE_REUSE,
    RA_ADAPT_QISKIT_COST_POLICY,
)
from pipelines.static_adapt.sr_snake.contracts import (  # noqa: E402
    PlateauCommutationInsertion,
)


REPAIR_ROOT = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
MATERIALIZATIONS_ROOT = REPAIR_ROOT / "bundles/materializations"
MATERIALIZATION_ID = "ra_adapt_qiskit_cost_plateau_local2_v1"
MATERIALIZATION_ROOT = MATERIALIZATIONS_ROOT / MATERIALIZATION_ID
FINAL_RECEIPT_NAME = "final_materialization_receipt.json"

V13_MATERIALIZATION_ID = "ra_adapt_stationary_late_core_v13"
V13_ROOT = MATERIALIZATIONS_ROOT / V13_MATERIALIZATION_ID
V13_SOURCE_LOCKS = (
    V13_ROOT / "source_materialization/source_locks_input.json"
)
V13_PROBLEM_BASELINES = (
    V13_ROOT / "source_materialization/problem_baselines.json"
)
V13_FINAL_RECEIPT = V13_ROOT / "final_publication_receipt.json"

GLOBAL_SINGLETON_MATERIALIZATION_ID = (
    "ra_adapt_global_singleton_insertion12_v1"
)
GLOBAL_SINGLETON_ROOT = (
    MATERIALIZATIONS_ROOT / GLOBAL_SINGLETON_MATERIALIZATION_ID
)
GLOBAL_SINGLETON_SOURCE_LOCKS = (
    GLOBAL_SINGLETON_ROOT
    / "source_materialization/source_locks_input.json"
)
GLOBAL_SINGLETON_FINAL_RECEIPT = (
    GLOBAL_SINGLETON_ROOT
    / "global_singleton_insertion12_materialization_receipt.json"
)

EXPECTED_V13_SOURCE_LOCKS_FILE_SHA256 = (
    "c361f55eb5551e9c11251ea01fce0059521c48d9fa0398d865b02a6bd29df656"
)
EXPECTED_GLOBAL_SINGLETON_SOURCE_LOCKS_FILE_SHA256 = (
    "9780195c3041dd16a45f2dedc3955b1b3932fce8daa8dd026ac69d1e1d1df666"
)
EXPECTED_PROBLEM_BASELINES_FILE_SHA256 = (
    "a12a36c3f2c8bfe74e4c8a0c9db1d1baecf3b100b00480c5386e903d973c4015"
)
EXPECTED_V13_FINAL_FILE_SHA256 = (
    "d9219d94db6f75d65842b828642e833d3c17eef0534516d0ef6dc2de08f6b415"
)
EXPECTED_GLOBAL_SINGLETON_FINAL_FILE_SHA256 = (
    "2b0763e0d9552be6322a9869adee3217a742b8612a75996a2b143f307e54a6dd"
)

MACRO_SOURCE_LOCK_ID = "strong_weak_u8__nph3__ra_macro_plateau"
GLOBAL_SINGLETON_SOURCE_LOCK_ID = (
    "strong_strong_u8__nph7__"
    "ra_global_singleton_plateau_commutation"
)
BASELINE_MACRO_ALGORITHM_ID = (
    "paper_i_ra_adapt_macro_plateau_insertion_repair_v1"
)
BASELINE_GLOBAL_SINGLETON_ALGORITHM_ID = (
    "paper_i_ra_adapt_global_singleton_plateau_commutation_v1"
)

QISKIT_ORACLE_DELTA_ID = "qiskit_selector_cost_oracle"
QISKIT_EXACT_CELL_DELTA_ID = "qiskit_cost_pilot_exact_cell_selection"
QISKIT_ALL_PHASE_DELTA_ID = "qiskit_cost_all_phase_scope"


class QiskitCostPilotMaterializationError(ValueError):
    """Raised when the two-cell pilot authority cannot close."""


def _sha256_file(path: Path) -> str:
    return v12.support._hash_file(path)


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise QiskitCostPilotMaterializationError(
            f"{label} is missing or unsafe: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QiskitCostPilotMaterializationError(
            f"{label} is not valid JSON."
        ) from exc
    if not isinstance(payload, dict):
        raise QiskitCostPilotMaterializationError(
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
        raise QiskitCostPilotMaterializationError(
            f"{label} drifted: expected {expected}, observed {observed}."
        )


def _settings_changes(lock: dict[str, Any]) -> list[dict[str, Any]]:
    trace = lock.get("resolver_trace")
    if not isinstance(trace, dict):
        raise QiskitCostPilotMaterializationError(
            "Selected source lock has no mutable resolver trace."
        )
    changes = trace.get("settings_changed")
    if not isinstance(changes, list) or not all(
        isinstance(row, Mapping) for row in changes
    ):
        raise QiskitCostPilotMaterializationError(
            "Selected source lock has no settings-change list."
        )
    return [copy.deepcopy(dict(row)) for row in changes]


def _without_change_id(
    rows: list[dict[str, Any]],
    change_id: str,
    *,
    expected_count: int,
) -> list[dict[str, Any]]:
    observed = sum(row.get("id") == change_id for row in rows)
    if observed != expected_count:
        raise QiskitCostPilotMaterializationError(
            f"Expected {expected_count} {change_id!r} rows, observed "
            f"{observed}."
        )
    return [row for row in rows if row.get("id") != change_id]


def _source_anchor(
    *,
    source_campaign_id: str,
    source_bundle_id: str,
    source_route_id: str,
    source_algorithm_id: str,
    target_algorithm_id: str,
    regime_id: str,
    nph: int,
    declared_delta_ids: list[str],
) -> dict[str, Any]:
    return {
        "schema": (
            "paper_i_ra_adapt_qiskit_cost_plateau_pilot_source_anchor_v1"
        ),
        "source_campaign_id": source_campaign_id,
        "source_bundle_id": source_bundle_id,
        "source_route_id": source_route_id,
        "source_algorithm_id": source_algorithm_id,
        "target_campaign_id": QISKIT_COST_PILOT_CAMPAIGN_ID,
        "target_bundle_id": QISKIT_COST_PILOT_BUNDLE_ID,
        "target_algorithm_id": target_algorithm_id,
        "regime_id": regime_id,
        "nph": int(nph),
        "scientific_result_anchor_claimed": False,
        "declared_delta_ids": declared_delta_ids,
    }


def _derive_source_locks() -> tuple[dict[str, Any], dict[str, Any]]:
    _require_file_sha256(
        V13_SOURCE_LOCKS,
        EXPECTED_V13_SOURCE_LOCKS_FILE_SHA256,
        label="sealed v13 source-lock input",
    )
    _require_file_sha256(
        GLOBAL_SINGLETON_SOURCE_LOCKS,
        EXPECTED_GLOBAL_SINGLETON_SOURCE_LOCKS_FILE_SHA256,
        label="sealed global-singleton source-lock input",
    )
    macro_source = _load_mapping(
        V13_SOURCE_LOCKS,
        label="sealed v13 source locks",
    )
    singleton_source = _load_mapping(
        GLOBAL_SINGLETON_SOURCE_LOCKS,
        label="sealed global-singleton source locks",
    )
    if macro_source.get("schema") != singleton_source.get("schema"):
        raise QiskitCostPilotMaterializationError(
            "The two source-lock schemas differ."
        )
    if macro_source.get("global_sources") != singleton_source.get(
        "global_sources"
    ):
        raise QiskitCostPilotMaterializationError(
            "The two source authorities disagree on global source locks."
        )
    macro_cells = macro_source.get("cell_locks")
    singleton_cells = singleton_source.get("cell_locks")
    if not isinstance(macro_cells, Mapping) or not isinstance(
        singleton_cells,
        Mapping,
    ):
        raise QiskitCostPilotMaterializationError(
            "A source authority has no cell-lock map."
        )
    macro_predecessor = macro_cells.get(MACRO_SOURCE_LOCK_ID)
    singleton_predecessor = singleton_cells.get(
        GLOBAL_SINGLETON_SOURCE_LOCK_ID
    )
    if not isinstance(macro_predecessor, Mapping) or not isinstance(
        singleton_predecessor,
        Mapping,
    ):
        raise QiskitCostPilotMaterializationError(
            "An exact source predecessor is absent."
        )

    cells = build_qiskit_cost_plateau_pilot_cell_specs()
    if len(cells) != 2:
        raise QiskitCostPilotMaterializationError(
            "The Qiskit-cost pilot no longer has exactly two cells."
        )
    derived_cells: dict[str, Any] = {}
    delta_receipt_rows: list[dict[str, Any]] = []
    for cell in cells:
        macro = cell.route_id == ROUTE_RA_MACRO_PLATEAU
        if macro:
            predecessor_id = MACRO_SOURCE_LOCK_ID
            predecessor = macro_predecessor
            source_campaign_id = CORE_CAMPAIGN_ID
            source_bundle_id = CORE_BUNDLE_ID
            source_algorithm_id = BASELINE_MACRO_ALGORITHM_ID
            target_algorithm_id = QISKIT_COST_PILOT_MACRO_ALGORITHM_ID
        elif cell.route_id == ROUTE_RA_GLOBAL_SINGLETON_PLATEAU:
            predecessor_id = GLOBAL_SINGLETON_SOURCE_LOCK_ID
            predecessor = singleton_predecessor
            source_campaign_id = GLOBAL_SINGLETON_CAMPAIGN_ID
            source_bundle_id = GLOBAL_SINGLETON_BUNDLE_ID
            source_algorithm_id = BASELINE_GLOBAL_SINGLETON_ALGORITHM_ID
            target_algorithm_id = (
                QISKIT_COST_PILOT_GLOBAL_SINGLETON_ALGORITHM_ID
            )
        else:
            raise QiskitCostPilotMaterializationError(
                f"Unexpected pilot route {cell.route_id!r}."
            )
        if (
            cell.source_lock_id != predecessor_id
            or cell.algorithm_id != target_algorithm_id
        ):
            raise QiskitCostPilotMaterializationError(
                f"Pilot cell identity drifted: {cell.cell_id}."
            )

        derived = copy.deepcopy(dict(predecessor))
        rows = _settings_changes(derived)
        declared_delta_ids = [
            QISKIT_ORACLE_DELTA_ID,
            QISKIT_EXACT_CELL_DELTA_ID,
        ]
        if macro:
            rows = _without_change_id(rows, "D5", expected_count=1)
            rows.append(
                {
                    "binding": "qiskit_cost_pilot_fixed_policy",
                    "classification": (
                        "explicit_user_requested_all_phase_cost_scope_v1"
                    ),
                    "field": "resource_weighting_scope",
                    "from": RESOURCE_WEIGHTING_LATE,
                    "id": QISKIT_ALL_PHASE_DELTA_ID,
                    "to": RESOURCE_WEIGHTING_ALL_PHASE,
                }
            )
            declared_delta_ids.append(QISKIT_ALL_PHASE_DELTA_ID)
        rows.extend(
            [
                {
                    "binding": "qiskit_cost_pilot_selector_cost",
                    "classification": (
                        "explicit_user_requested_qiskit_cost_ablation_v1"
                    ),
                    "field": "selector_cost_policy",
                    "from": "marrakesh_graph_span_v1",
                    "id": QISKIT_ORACLE_DELTA_ID,
                    "to": RA_ADAPT_QISKIT_COST_POLICY,
                },
                {
                    "binding": "qiskit_cost_pilot_exact_cell",
                    "classification": (
                        "explicit_user_requested_local_two_cell_pilot_v1"
                    ),
                    "field": "campaign_cell_selection",
                    "from": predecessor_id,
                    "id": QISKIT_EXACT_CELL_DELTA_ID,
                    "to": cell.cell_id,
                },
            ]
        )
        trace = derived["resolver_trace"]
        trace["settings_changed"] = rows
        trace["qiskit_cost_pilot_source_anchor"] = _source_anchor(
            source_campaign_id=source_campaign_id,
            source_bundle_id=source_bundle_id,
            source_route_id=cell.route_id,
            source_algorithm_id=source_algorithm_id,
            target_algorithm_id=target_algorithm_id,
            regime_id=cell.regime_id,
            nph=cell.nph,
            declared_delta_ids=declared_delta_ids,
        )
        if (
            derived.get("archive") != predecessor.get("archive")
            or derived.get("member") != predecessor.get("member")
            or derived.get("route_id") != cell.route_id
            or derived.get("regime_id") != cell.regime_id
            or int(derived.get("nph", -1)) != int(cell.nph)
        ):
            raise QiskitCostPilotMaterializationError(
                f"Source-byte or cell identity drifted for {cell.cell_id}."
            )
        derived_cells[cell.source_lock_id] = derived
        delta_receipt_rows.append(
            {
                "cell_id": cell.cell_id,
                "source_lock_id": cell.source_lock_id,
                "predecessor_source_lock_id": predecessor_id,
                "predecessor_source_lock_canonical_sha256": (
                    canonical_sha256(predecessor)
                ),
                "source_campaign_id": source_campaign_id,
                "source_bundle_id": source_bundle_id,
                "source_algorithm_id": source_algorithm_id,
                "target_algorithm_id": target_algorithm_id,
                "declared_delta_ids": declared_delta_ids,
                "archive_binding_preserved": True,
                "member_binding_preserved": True,
                "scientific_result_anchor_claimed": False,
            }
        )

    source_locks = {
        "schema": macro_source["schema"],
        "global_sources": copy.deepcopy(macro_source["global_sources"]),
        "cell_locks": derived_cells,
    }
    delta_receipt = _digested(
        {
            "schema": (
                "paper_i_ra_adapt_qiskit_cost_plateau_local2_"
                "source_lock_delta_v1"
            ),
            "status": "passed",
            "source_authorities": {
                "macro": {
                    "materialization_id": V13_MATERIALIZATION_ID,
                    "source_locks_file_sha256": (
                        EXPECTED_V13_SOURCE_LOCKS_FILE_SHA256
                    ),
                    "source_lock_id": MACRO_SOURCE_LOCK_ID,
                },
                "global_singleton": {
                    "materialization_id": (
                        GLOBAL_SINGLETON_MATERIALIZATION_ID
                    ),
                    "source_locks_file_sha256": (
                        EXPECTED_GLOBAL_SINGLETON_SOURCE_LOCKS_FILE_SHA256
                    ),
                    "source_lock_id": (
                        GLOBAL_SINGLETON_SOURCE_LOCK_ID
                    ),
                },
            },
            "fixed_scientific_contract": {
                "active_gradient_policy": ACTIVE_GRADIENT_STATIONARY,
                "resource_weighting_scope": RESOURCE_WEIGHTING_ALL_PHASE,
                "insertion_policy": PlateauCommutationInsertion.kind,
                "selector_cost_policy": RA_ADAPT_QISKIT_COST_POLICY,
                "selector_cost_phase_reuse": (
                    RA_ADAPT_QISKIT_COST_PHASE_REUSE
                ),
                "horizon": 50,
                "optimizer": "powell",
                "optimizer_maxiter": 200,
                "seed": 7,
                "transpiler_seed": 7,
            },
            "rows": delta_receipt_rows,
            "derived_cell_count": 2,
            "all_archive_bindings_preserved": True,
            "all_member_bindings_preserved": True,
            "all_global_source_bindings_preserved": True,
            "execution_authorized": False,
            "submission_authorized": False,
            "submitted": False,
        }
    )
    return source_locks, delta_receipt


def main() -> int:
    if MATERIALIZATION_ROOT.exists() or MATERIALIZATION_ROOT.is_symlink():
        raise FileExistsError(
            "Refusing to overwrite Qiskit-cost pilot materialization: "
            f"{MATERIALIZATION_ROOT}"
        )
    if not MATERIALIZATIONS_ROOT.is_dir():
        raise QiskitCostPilotMaterializationError(
            f"Materializations root is missing: {MATERIALIZATIONS_ROOT}"
        )
    for path, expected, label in (
        (
            V13_PROBLEM_BASELINES,
            EXPECTED_PROBLEM_BASELINES_FILE_SHA256,
            "sealed problem baselines",
        ),
        (
            V13_FINAL_RECEIPT,
            EXPECTED_V13_FINAL_FILE_SHA256,
            "sealed v13 final receipt",
        ),
        (
            GLOBAL_SINGLETON_FINAL_RECEIPT,
            EXPECTED_GLOBAL_SINGLETON_FINAL_FILE_SHA256,
            "sealed global-singleton final receipt",
        ),
    ):
        _require_file_sha256(path, expected, label=label)

    source_locks, delta_receipt = _derive_source_locks()
    baselines = _load_mapping(
        V13_PROBLEM_BASELINES,
        label="sealed v13 problem baselines",
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{MATERIALIZATION_ID}.staging.",
            dir=MATERIALIZATIONS_ROOT,
        )
    )
    try:
        source_dir = staging / "source_materialization"
        source_dir.mkdir(parents=True, exist_ok=False)
        v12.support._write_bytes_atomic_no_replace(
            source_dir / "v13_source_locks.json",
            V13_SOURCE_LOCKS.read_bytes(),
        )
        v12.support._write_bytes_atomic_no_replace(
            source_dir / "global_singleton_source_locks.json",
            GLOBAL_SINGLETON_SOURCE_LOCKS.read_bytes(),
        )
        _write_json(
            source_dir / "source_locks_input.json",
            source_locks,
        )
        _write_json(
            source_dir / "source_lock_delta_receipt.json",
            delta_receipt,
        )
        v12.support._write_bytes_atomic_no_replace(
            source_dir / "problem_baselines.json",
            V13_PROBLEM_BASELINES.read_bytes(),
        )

        receipt = materialize_qiskit_cost_plateau_pilot_bundle(
            staging,
            problem_resolver=v12.support._problem_resolver_from(
                baselines
            ),
            source_locks=source_locks,
            repository_state=_repository_state(),
            repo_root=REPO_ROOT,
            horizon=50,
            dependency_lock_paths=(REPO_ROOT / "requirements.txt",),
            materialization_timestamp=v12.support._utc_now(),
            verify_source_files=True,
        )
        if (
            receipt.bundle_id != QISKIT_COST_PILOT_BUNDLE_ID
            or receipt.materialization_status != "passed"
            or int(receipt.cell_count) != 2
        ):
            raise QiskitCostPilotMaterializationError(
                "Qiskit-cost pilot bundle receipt drifted."
            )

        bundle_root = staging / QISKIT_COST_PILOT_BUNDLE_ID
        # The official loader expects the final destination's stable paths.
        loaded_receipts: list[dict[str, Any]] = []
        expected = {
            cell.cell_id: cell
            for cell in build_qiskit_cost_plateau_pilot_cell_specs()
        }
        for cell_id, cell in expected.items():
            protocol_path = (
                bundle_root / "protocols" / f"{cell_id}.json"
            )
            protocol = load_validated_bundle_protocol(protocol_path)
            route_cost = protocol.route_contract.get(
                "execution_settings"
            )
            if not isinstance(route_cost, Mapping):
                raise QiskitCostPilotMaterializationError(
                    f"Loaded pilot route contract is malformed: {cell_id}."
                )
            if (
                protocol.algorithm_id != cell.algorithm_id
                or protocol.active_gradient_policy
                != ACTIVE_GRADIENT_STATIONARY
                or protocol.resource_weighting_scope
                != RESOURCE_WEIGHTING_ALL_PHASE
                or int(protocol.horizon) != 50
                or int(protocol.optimizer_maxiter) != 200
                or int(protocol.seeds.get("adapt", -1)) != 7
                or int(protocol.seeds.get("transpiler", -1)) != 7
                or route_cost.get("phase3_backend_cost_mode")
                != "transpile_single_v1"
            ):
                raise QiskitCostPilotMaterializationError(
                    f"Loaded pilot protocol drifted: {cell_id}."
                )
            loaded_receipts.append(
                {
                    "cell_id": cell_id,
                    "protocol": _binding(
                        protocol_path,
                        relative_to=staging,
                    ),
                    "protocol_canonical_sha256": protocol.sha256,
                    "algorithm_id": protocol.algorithm_id,
                    "candidate_representation": (
                        protocol.candidate_representation
                    ),
                    "loaded_with_official_validator": True,
                }
            )

        final = _digested(
            {
                "schema": (
                    "paper_i_ra_adapt_qiskit_cost_plateau_local2_"
                    "materialization_receipt_v1"
                ),
                "status": "passed",
                "materialization_id": MATERIALIZATION_ID,
                "campaign_id": QISKIT_COST_PILOT_CAMPAIGN_ID,
                "bundle_id": QISKIT_COST_PILOT_BUNDLE_ID,
                "run_class": QISKIT_COST_PILOT_RUN_CLASS,
                "execution_target": "local",
                "source_authorities": {
                    "v13_source_locks_file_sha256": (
                        EXPECTED_V13_SOURCE_LOCKS_FILE_SHA256
                    ),
                    "global_singleton_source_locks_file_sha256": (
                        EXPECTED_GLOBAL_SINGLETON_SOURCE_LOCKS_FILE_SHA256
                    ),
                    "problem_baselines_file_sha256": (
                        EXPECTED_PROBLEM_BASELINES_FILE_SHA256
                    ),
                    "source_lock_delta_receipt": _binding(
                        source_dir / "source_lock_delta_receipt.json",
                        relative_to=staging,
                    ),
                },
                "fixed_scientific_contract": {
                    "active_gradient_policy": (
                        ACTIVE_GRADIENT_STATIONARY
                    ),
                    "resource_weighting_scope": (
                        RESOURCE_WEIGHTING_ALL_PHASE
                    ),
                    "insertion_policy": (
                        PlateauCommutationInsertion.kind
                    ),
                    "selector_cost_policy": (
                        RA_ADAPT_QISKIT_COST_POLICY
                    ),
                    "selector_cost_phase_reuse": (
                        RA_ADAPT_QISKIT_COST_PHASE_REUSE
                    ),
                    "phase1_shortlist_size": 24,
                    "phase2_shortlist_size": 12,
                    "phase3_admission_cardinality": 1,
                    "optimizer": "powell",
                    "optimizer_maxiter": 200,
                    "seed": 7,
                    "transpiler_seed": 7,
                    "horizon": 50,
                },
                "cell_count": 2,
                "cells": loaded_receipts,
                "bundle": {
                    "bundle_manifest": _binding(
                        bundle_root / "bundle_manifest.json",
                        relative_to=staging,
                    ),
                    "source_locks": _binding(
                        bundle_root / "source_locks.json",
                        relative_to=staging,
                    ),
                    "expected_artifacts": _binding(
                        bundle_root / "expected_artifacts.json",
                        relative_to=staging,
                    ),
                    "validation_report": _binding(
                        bundle_root / "validation_report.json",
                        relative_to=staging,
                    ),
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
            MATERIALIZATION_ROOT,
        )
        print(
            json.dumps(
                {
                    "status": "passed",
                    "materialization_root": (
                        MATERIALIZATION_ROOT.relative_to(
                            REPO_ROOT
                        ).as_posix()
                    ),
                    "final_receipt_sha256": final["sha256"],
                    "cell_count": 2,
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
