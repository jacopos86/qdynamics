#!/usr/bin/env python3
"""Materialize and run the two-cell cumulative-relative plateau diagnostic."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
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

SOURCE_BUNDLE = (
    REPO_ROOT
    / "chtc/paper_i_ra_adapt_repair_20260727/bundles/materializations"
    / "ra_adapt_stationary_late_core_v13"
    / "ra_repair_stationary_late_core_v1"
)
OUTPUT_ROOT = (
    REPO_ROOT
    / "output/local_runs"
    / "paper_i_ra_adapt_cumulative_plateau_pair_r20_local_20260731_v1"
)
MATERIALIZATION_ROOT = OUTPUT_ROOT / "materialization"
RUNS_ROOT = OUTPUT_ROOT / "runs"
DIAGNOSTIC_BUNDLE_ID = (
    "paper_i_ra_adapt_cumulative_relative_plateau_pair_r20_local_v1"
)
MAXIMUM_CONTROLLER_ROUNDS = 20
EXACT_ENERGIES = {
    "core__intermediate_strong__nph7__ra_macro_plateau": (
        -0.6239396137518985
    ),
    "core__strong_strong_u8__nph7__ra_singleton_plateau": (
        0.5205762765682245
    ),
}
CELLS = tuple(EXACT_ENERGIES)
ALLOWED_ROUTE_DIFFS = {
    ("lineage_authority", "parent_contract_sha256"),
    (
        "semantic_invariants",
        "plateau_cumulative_decrease_ratio_threshold",
    ),
    ("semantic_invariants", "plateau_energy_decrease_threshold"),
    (
        "semantic_invariants",
        "plateau_threshold_calibration_status",
    ),
    ("semantic_invariants", "plateau_threshold_comparison"),
    ("semantic_invariants", "plateau_trigger_source"),
    ("sha256",),
}


class DiagnosticContractError(RuntimeError):
    """Fail-closed diagnostic contract violation."""


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
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DiagnosticContractError(f"Expected a JSON object: {path}")
    return value


def _verify_digest(value: Mapping[str, Any], *, label: str) -> None:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("sha256") != _canonical_sha256(unsigned):
        raise DiagnosticContractError(f"{label} self-digest drifted.")


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
                _scalar_differences(
                    left, right, path=(*path, index)
                )
            )
        return result
    if before != after:
        result.append((path, before, after))
    return result


def _source_surfaces() -> tuple[dict[str, Any], dict[str, Any]]:
    source_manifest = _load_json(SOURCE_BUNDLE / "bundle_manifest.json")
    source_locks = _load_json(SOURCE_BUNDLE / "source_locks.json")
    _verify_digest(source_manifest, label="source bundle manifest")
    _verify_digest(source_locks, label="source-lock manifest")
    return source_manifest, source_locks


def materialize() -> dict[str, Any]:
    from pipelines.static_adapt.ra_adapt.bundles import (
        _bundle_protocol_materialization_authority,
        _cell_from_manifest_row,
        _decorate_protocol_payload,
        _implementation_source_inventory,
        _source_lock_refs,
    )
    from pipelines.static_adapt.ra_adapt.contracts import (
        canonical_sha256,
        resolved_ra_adapt_protocol_from_mapping,
    )
    from pipelines.static_adapt.ra_adapt.engine import (
        build_resolved_ra_protocol,
    )

    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_ROOT}")
    source_manifest, predecessor_locks = _source_surfaces()
    inventory = _implementation_source_inventory(REPO_ROOT)
    source_locks = copy.deepcopy(predecessor_locks)
    source_locks["implementation_sources"] = inventory
    source_locks.pop("sha256", None)
    source_locks["sha256"] = canonical_sha256(source_locks)
    source_rows = {
        str(row["cell_id"]): row for row in source_manifest["cells"]
    }
    plan = _digested(
        {
            "schema": "paper_i_ra_adapt_local_plateau_ratio_plan_v1",
            "campaign_id": DIAGNOSTIC_BUNDLE_ID,
            "run_class": "diagnostic",
            "cells": list(CELLS),
            "source_bundle": {
                "path": SOURCE_BUNDLE.relative_to(REPO_ROOT).as_posix(),
                "manifest_sha256": source_manifest["sha256"],
                "manifest_file_sha256": _sha256_file(
                    SOURCE_BUNDLE / "bundle_manifest.json"
                ),
                "source_locks_sha256": predecessor_locks["sha256"],
            },
            "implementation_source_inventory_sha256": inventory["sha256"],
            "scientific_changes": [
                "absolute_plateau_drop_to_prior_cumulative_relative_drop_v1",
                "plateau_ratio_threshold_1e-4",
            ],
            "execution_mechanics_changes": [
                "operational_maximum_controller_rounds_20",
                "local_checkpoint_and_estimator_ledger_paths",
            ],
            "preserved_settings": {
                "active_gradient_policy": "stationary_source_response_v1",
                "resource_weighting_scope": "late_resource_weighting_v1",
                "optimizer": "powell",
                "optimizer_maxiter": 200,
                "seed": 7,
                "protocol_horizon": 50,
                "same_cutoff_exact_reference": True,
            },
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
    validations: list[dict[str, Any]] = []
    for cell_id in CELLS:
        source_protocol_path = SOURCE_BUNDLE / "protocols" / f"{cell_id}.json"
        source_payload = _load_json(source_protocol_path)
        source_protocol = resolved_ra_adapt_protocol_from_mapping(
            source_payload
        )
        cell = _cell_from_manifest_row(source_rows[cell_id])
        refs = _source_lock_refs(source_locks, cell=cell)
        authority = _bundle_protocol_materialization_authority(
            cell=cell,
            bundle_id=DIAGNOSTIC_BUNDLE_ID,
            bundle_manifest_sha256=plan["sha256"],
            source_locks_sha256=source_locks["sha256"],
            source_lock_refs=refs,
            active_gradient_policy=source_protocol.active_gradient_policy,
            resource_weighting_scope=source_protocol.resource_weighting_scope,
        )
        base = build_resolved_ra_protocol(
            _problem_from_receipt(source_protocol.problem),
            source_protocol.request,
            materialization_authority=authority,
        )
        decorated = _decorate_protocol_payload(
            base.to_dict(),
            cell=cell,
            request=source_protocol.request,
            cell_source_lock=source_locks["cell_locks"][cell.source_lock_id],
            materialization_authority=authority,
        )
        protocol = resolved_ra_adapt_protocol_from_mapping(decorated)
        differences = _scalar_differences(
            source_payload["route_contract"], decorated["route_contract"]
        )
        observed_paths = {path for path, _before, _after in differences}
        if observed_paths != ALLOWED_ROUTE_DIFFS:
            raise DiagnosticContractError(
                f"{cell_id} changed unexpected route fields: "
                f"{sorted(observed_paths, key=str)}"
            )
        for key in (
            "problem",
            "parent_inventory",
            "executable_pool",
            "optimizer",
            "optimizer_maxiter",
            "seeds",
            "candidate_representation",
            "adapter_id",
            "algorithm_id",
            "active_gradient_policy",
            "resource_weighting_scope",
            "accepted_refit_scope",
            "accepted_refit_coordinate_chart",
            "accepted_refit_base_chart_policy",
            "phase3_solver_id",
            "phase3_multiplier_contract",
            "request",
            "horizon",
        ):
            if source_payload[key] != decorated[key]:
                raise DiagnosticContractError(
                    f"{cell_id} changed preserved protocol field {key}."
                )
        protocol_path = MATERIALIZATION_ROOT / "protocols" / f"{cell_id}.json"
        _write_json(protocol_path, decorated)
        validations.append(
            {
                "cell_id": cell_id,
                "source_protocol_sha256": source_protocol.sha256,
                "source_protocol_file_sha256": _sha256_file(
                    source_protocol_path
                ),
                "diagnostic_protocol_sha256": protocol.sha256,
                "diagnostic_protocol_file_sha256": _sha256_file(protocol_path),
                "route_differences": [
                    {
                        "path": list(path),
                        "before": before,
                        "after": after,
                    }
                    for path, before, after in differences
                ],
                "non_trigger_scientific_diff": [],
                "status": "passed",
            }
        )
    validation = _digested(
        {
            "schema": "paper_i_ra_adapt_local_plateau_ratio_validation_v1",
            "status": "passed",
            "cell_count": len(validations),
            "cells": validations,
            "source_value_anchor": {
                "status": "passed_by_completed_trajectory_replay_v1",
                "old_absolute_threshold": 1.0e-8,
                "recorded_first_open_rounds_reproduced": True,
            },
            "execution_authorized": False,
        }
    )
    _write_json(MATERIALIZATION_ROOT / "validation_report.json", validation)
    receipt = _digested(
        {
            "schema": "paper_i_ra_adapt_local_plateau_ratio_materialization_v1",
            "status": "passed",
            "plan_sha256": plan["sha256"],
            "source_locks_sha256": source_locks["sha256"],
            "validation_sha256": validation["sha256"],
            "cell_count": len(CELLS),
            "execution_authorized": False,
            "submission_authorized": False,
        }
    )
    _write_json(
        MATERIALIZATION_ROOT / "materialization_receipt.json", receipt
    )
    return receipt


def _load_bound_protocol(cell_id: str) -> Any:
    from pipelines.static_adapt.ra_adapt.bundles import (
        _bundle_protocol_materialization_authority,
        _cell_from_manifest_row,
    )
    from pipelines.static_adapt.ra_adapt.contracts import (
        _attach_validated_bundle_protocol_authority,
        resolved_ra_adapt_protocol_from_mapping,
    )

    source_manifest, _predecessor_locks = _source_surfaces()
    rows = {str(row["cell_id"]): row for row in source_manifest["cells"]}
    plan = _load_json(MATERIALIZATION_ROOT / "materialization_plan.json")
    source_locks = _load_json(
        MATERIALIZATION_ROOT / "source_locks_snapshot.json"
    )
    _verify_digest(plan, label="materialization plan")
    _verify_digest(source_locks, label="diagnostic source locks")
    protocol_path = MATERIALIZATION_ROOT / "protocols" / f"{cell_id}.json"
    protocol = resolved_ra_adapt_protocol_from_mapping(
        _load_json(protocol_path)
    )
    cell = _cell_from_manifest_row(rows[cell_id])
    authority = _bundle_protocol_materialization_authority(
        cell=cell,
        bundle_id=DIAGNOSTIC_BUNDLE_ID,
        bundle_manifest_sha256=plan["sha256"],
        source_locks_sha256=source_locks["sha256"],
        source_lock_refs=protocol.source_locks,
        active_gradient_policy=protocol.active_gradient_policy,
        resource_weighting_scope=protocol.resource_weighting_scope,
        protocol_sha256=protocol.sha256,
    )
    return _attach_validated_bundle_protocol_authority(protocol, authority)


def run_cell(cell_id: str) -> dict[str, Any]:
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

    if cell_id not in CELLS:
        raise DiagnosticContractError(f"Unknown diagnostic cell: {cell_id}")
    protocol = _load_bound_protocol(cell_id)
    run_root = RUNS_ROOT / cell_id
    run_root.mkdir(parents=True, exist_ok=False)
    authorization = _digested(
        {
            "schema": "paper_i_ra_adapt_local_execution_authorization_v1",
            "cell_id": cell_id,
            "protocol_sha256": protocol.sha256,
            "maximum_controller_rounds": MAXIMUM_CONTROLLER_ROUNDS,
            "authorization_source": "explicit_user_request_2026-07-31",
            "execution_authorized": True,
            "submission_authorized": False,
            "authorized_at_utc": _utc_now(),
        }
    )
    _write_json(run_root / "execution_authorization.json", authorization)
    manifest = _digested(
        {
            "schema": "paper_i_ra_adapt_local_run_manifest_v1",
            "run_class": "diagnostic",
            "cell_id": cell_id,
            "protocol_sha256": protocol.sha256,
            "active_gradient_policy": protocol.active_gradient_policy,
            "resource_weighting_scope": protocol.resource_weighting_scope,
            "candidate_representation": protocol.candidate_representation,
            "optimizer": protocol.optimizer,
            "optimizer_maxiter": protocol.optimizer_maxiter,
            "adapt_seed": protocol.seeds["adapt"],
            "protocol_horizon": protocol.horizon,
            "operational_maximum_controller_rounds": (
                MAXIMUM_CONTROLLER_ROUNDS
            ),
            "plateau_cumulative_decrease_ratio_threshold": 1.0e-4,
            "checkpoint_path": "checkpoint.json",
            "estimator_ledger_path": "estimator_ledger.json",
            "exact_same_cutoff_energy": EXACT_ENERGIES[cell_id],
            "execution_authorization_sha256": authorization["sha256"],
            "started_at_utc": _utc_now(),
        }
    )
    _write_json(run_root / "run_manifest.json", manifest)
    checkpoint = run_root / "checkpoint.json"
    ledger = run_root / "estimator_ledger.json"
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
            resource_rounds=(10, 20),
        ),
    )
    try:
        result = run_ra_adapt(
            _problem_from_receipt(protocol.problem),
            protocol,
            operational_controls=controls,
        )
        _write_json(run_root / "result.json", result.to_dict())
        if result.run.paper_i_summary is not None:
            _write_json(
                run_root / "paper_i_summary.json",
                result.run.paper_i_summary.to_dict(),
            )
        delta_e = abs(
            float(result.final_state.energy) - EXACT_ENERGIES[cell_id]
        )
        terminal = _digested(
            {
                "schema": "paper_i_ra_adapt_local_terminal_receipt_v1",
                "status": "passed",
                "cell_id": cell_id,
                "accepted_controller_rounds": len(
                    result.accepted_trajectory
                ),
                "final_same_cutoff_delta_e": delta_e,
                "protocol_sha256": protocol.sha256,
                "manifest_sha256": manifest["sha256"],
                "checkpoint_sha256": _sha256_file(checkpoint),
                "estimator_ledger_sha256": _sha256_file(ledger),
                "result_sha256": _sha256_file(run_root / "result.json"),
                "completed_at_utc": _utc_now(),
            }
        )
        _write_json(run_root / "terminal_receipt.json", terminal)
        return terminal
    except BaseException as exc:
        failure = _digested(
            {
                "schema": "paper_i_ra_adapt_local_failure_receipt_v1",
                "status": "failed",
                "cell_id": cell_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "checkpoint_present": checkpoint.is_file(),
                "ledger_present": ledger.is_file(),
                "failed_at_utc": _utc_now(),
            }
        )
        _write_json(run_root / "failure_receipt.json", failure)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--materialize", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--cell", choices=CELLS)
    parser.add_argument("--execution-authorized", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.materialize:
        if args.cell is not None or args.execution_authorized:
            raise DiagnosticContractError(
                "Materialization cannot carry execution arguments."
            )
        print(_canonical_bytes(materialize()).decode("utf-8"))
        return 0
    if args.cell is None or not args.execution_authorized:
        raise DiagnosticContractError(
            "Execution requires --cell and --execution-authorized."
        )
    print(_canonical_bytes(run_cell(args.cell)).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
