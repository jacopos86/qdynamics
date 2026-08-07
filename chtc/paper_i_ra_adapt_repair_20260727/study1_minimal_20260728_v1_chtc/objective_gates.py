#!/usr/bin/env python3
"""Fail-closed Study-1 objective-gate validation.

The module has no execution or submission authority.  It validates compact,
self-digested materialization receipts and derives cell/matrix receipts from
the five narrow scientific artifacts.  Missing evidence is an error; a
favorable-looking boolean is never substituted for the underlying receipt.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from package_contract import (
    BUNDLE_IDS,
    PACKAGE_ID,
    STATIONARY_BUNDLE_ID,
    VALIDATION_REGIMES,
    PackageContractError,
    canonical_sha256,
    digested,
    load_json_object,
    logical_cell_keys,
    require_sha256,
    sha256_file,
    validation_cell_id,
    verify_exact_key_set,
    verify_self_digest,
)


OBJECTIVE_GATE_AUTHORITY_SCHEMA = (
    "paper_i_ra_adapt_study1_objective_gate_authority_v3"
)
SOURCE_LOCK_CELL_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_study1_source_lock_cell_receipt_v2"
)
POOL_CONSTRUCTION_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_study1_pool_construction_receipt_v2"
)
SINGLETON_CONSTRUCTION_EQUIVALENCE_SCHEMA = (
    "paper_i_ra_adapt_study1_singleton_construction_equivalence_v1"
)
TRUSTED_EXECUTION_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_study1_trusted_execution_receipt_v2"
)
T13_CHARACTERIZATION_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_t13_characterization_receipt_v2"
)
CELL_GATE_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_study1_cell_objective_gate_receipt_v2"
)
MATRIX_GATE_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_study1_objective_gate_matrix_v2"
)
G5_EVIDENCE_SCHEMA = "paper_i_scored_insertion_position_population_v1"
G8_EVIDENCE_SCHEMA = (
    "paper_i_ra_adapt_exact_reference_isolation_receipt_v1"
)
G9_EVIDENCE_SCHEMA = "paper_i_numerical_physical_integrity_v1"
G11_EVIDENCE_SCHEMA = "paper_i_controller_replay_evidence_v1"
G11_DIAGNOSTIC_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_study1_g11_replay_diagnostic_v2"
)
G11_COMPARISON_SCHEMA = (
    "paper_i_bounded_deterministic_replay_comparison_v1"
)
RA_SIGNED_PREFIX_SCHEMA = "paper_i_signed_active_prefix_checkpoint_v1"
APPEND_SIGNED_PREFIX_SCHEMA = (
    "paper_i_signed_append_active_prefix_checkpoint_v1"
)
SIGNED_CONTROLLER_PREFIX_SCHEMA = (
    "paper_i_signed_controller_round_prefix_v1"
)
ESTIMATOR_LEDGER_SCHEMA = "estimator_call_ledger_v1"
RA_ESTIMATOR_LEDGER_SIDECAR_SCHEMA = (
    "paper_i_estimator_call_ledger_sidecar_v2"
)
RA_ESTIMATOR_ACCOUNTING_SCHEMA = "paper_i_current_s_alg_accounting_v2"

OBJECTIVE_GATE_IDS = tuple(f"G{index}" for index in range(1, 15))
COMPLETION_STATES = (
    "done",
    "failed",
    "missing",
    "blocked",
    "superseded",
)

EXACT_INSERTION_CHART = "exact_ordered_insertion_zero_angle_v1"
RA_REFIT_CHART = "supported_fs_whitened_fixed_v1"
APPEND_REFIT_CHART = "native_v1"
TABLE_I_COMPILE_ID = "table_i_basis_gate_transpile_v1"
MACRO_COUNT_NPH3 = 102
MACRO_ORDERED_LABELS_SHA256_NPH3 = (
    "a8831528590e870a09ce08492b6f61da4a4d377e63fa8983b30ca9698af5d3d9"
)
SINGLETON_PARENT_COUNT_NPH3 = 123
SINGLETON_PARENT_LABELS_SHA256_NPH3 = (
    "17cc97b744f8e6b50b686b24edd28426ca2c055bc2c31054fd353ddfa10efbe3"
)

# T13 remains a generic route characterization at its own physics.  These
# pins prevent a Study-1 U=8 cell from being compared to the fixture.
T13_PROBLEM_REQUEST_SHA256 = (
    "b7299ce9e978abc1f5c2db8b11328dbe2df4f679be1abc4d6aa4da3fc0159c53"
)
T13_FIXTURE_FILE_SHA256 = (
    "722ff9e3503c46a577d18ef9206b5914b8ad7a5b965a6510e42e69ed645ac220"
)
T13_FIXTURE_CANONICAL_SHA256 = (
    "df5e73b94f0abea0e3d37b0b8c1c00eff15ecdd357dde670c72b9d4f2ca1bd67"
)

_COMPONENT_NAMES = ("N_H_outer", "N_H_refit", "N_grad", "N_metric")
_LOWER_COMPONENT_NAMES = {
    "N_H_outer": "n_h_outer",
    "N_H_refit": "n_h_refit",
    "N_grad": "n_grad",
    "N_metric": "n_metric",
}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PackageContractError(f"{label} must be a JSON object.")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise PackageContractError(f"{label} must be a JSON array.")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise PackageContractError(f"{label} must be an integer.")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise PackageContractError(f"{label} must be an integer.") from exc
    if resolved != value or resolved < minimum:
        raise PackageContractError(
            f"{label} must be an integer >= {minimum}."
        )
    return resolved


def _finite(value: Any, *, label: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise PackageContractError(f"{label} must be finite.") from exc
    if not math.isfinite(resolved):
        raise PackageContractError(f"{label} must be finite.")
    return resolved


def _bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise PackageContractError(f"{label} must be a boolean.")
    return value


def _digested_mapping(
    value: Any,
    *,
    label: str,
    schema: str | None = None,
) -> Mapping[str, Any]:
    payload = _mapping(value, label=label)
    verify_self_digest(payload, label=label)
    if schema is not None and payload.get("schema") != schema:
        raise PackageContractError(
            f"{label} schema drifted: {payload.get('schema')!r}."
        )
    return payload


def _logical_protocol_path(
    revision_root: Path,
    *,
    bundle_id: str,
    cell_id: str,
) -> Path:
    return revision_root / bundle_id / "protocols" / f"{cell_id}.json"


def _load_protocol(
    revision_root: Path,
    *,
    bundle_id: str,
    cell_id: str,
) -> dict[str, Any]:
    path = _logical_protocol_path(
        revision_root,
        bundle_id=bundle_id,
        cell_id=cell_id,
    )
    payload = load_json_object(path, label=f"{bundle_id}::{cell_id} protocol")
    verify_self_digest(payload, label=f"{bundle_id}::{cell_id} protocol")
    return payload


def _source_lock_row(
    revision_root: Path,
    *,
    bundle_id: str,
    source_lock_id: str,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    source_locks = load_json_object(
        revision_root / bundle_id / "source_locks.json",
        label=f"{bundle_id} source locks",
    )
    verify_self_digest(source_locks, label=f"{bundle_id} source locks")
    rows = _mapping(
        source_locks.get("cell_locks"),
        label=f"{bundle_id} cell locks",
    )
    row = _mapping(
        rows.get(source_lock_id),
        label=f"{bundle_id} source lock {source_lock_id}",
    )
    verify_self_digest(
        row,
        label=f"{bundle_id} source lock {source_lock_id}",
    )
    return source_locks, row


def _validate_cell_authority(
    raw: Any,
    *,
    revision_root: Path,
) -> dict[str, Any]:
    receipt = dict(
        _digested_mapping(
            raw,
            label="G1/G2 per-cell authority receipt",
            schema=SOURCE_LOCK_CELL_RECEIPT_SCHEMA,
        )
    )
    logical_key = str(receipt.get("logical_key", ""))
    if logical_key not in logical_cell_keys():
        raise PackageContractError(
            f"G1/G2 receipt has an unknown logical key: {logical_key!r}."
        )
    bundle_id, cell_id = logical_key.split("::", maxsplit=1)
    if (
        receipt.get("bundle_id") != bundle_id
        or receipt.get("cell_id") != cell_id
    ):
        raise PackageContractError(
            f"G1/G2 receipt identity drifted: {logical_key}."
        )
    protocol = _load_protocol(
        revision_root,
        bundle_id=bundle_id,
        cell_id=cell_id,
    )
    problem = _mapping(protocol.get("problem"), label=f"{logical_key} problem")
    source_refs = _mapping(
        protocol.get("source_locks"),
        label=f"{logical_key} source-lock refs",
    )
    source_lock_id = str(source_refs.get("cell_source_lock_id", ""))
    source_locks, source_row = _source_lock_row(
        revision_root,
        bundle_id=bundle_id,
        source_lock_id=source_lock_id,
    )
    archive = _mapping(
        source_row.get("archive"),
        label=f"{logical_key} historical source archive",
    )
    member = _mapping(
        source_row.get("member"),
        label=f"{logical_key} historical source member",
    )
    expected = {
        "protocol_sha256": protocol["sha256"],
        "problem_request_sha256": problem.get("problem_request_sha256"),
        "source_locks_manifest_sha256": source_locks["sha256"],
        "cell_source_lock_id": source_lock_id,
        "cell_source_lock_sha256": source_row["sha256"],
        "source_archive_sha256": archive.get("sha256"),
        "source_member_sha256": member.get("sha256"),
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise PackageContractError(
                f"G1 authority drifted at {logical_key}.{field}."
            )
        if field.endswith("_sha256"):
            require_sha256(value, label=f"{logical_key}.{field}")
    if (
        receipt.get("source_archive_rehashed_at_materialization") is not True
        or receipt.get("source_member_rehashed_at_materialization") is not True
    ):
        raise PackageContractError(
            f"G1 compact source bytes were not rehashed for {logical_key}."
        )

    same_cutoff = _mapping(
        receipt.get("same_cutoff_reference"),
        label=f"{logical_key} same-cutoff receipt",
    )
    trace = _mapping(
        source_row.get("resolver_trace"),
        label=f"{logical_key} resolver trace",
    )
    source_ed = _mapping(
        trace.get("same_cutoff_ed_reference"),
        label=f"{logical_key} source ED receipt",
    )
    nph_work = _integer(
        same_cutoff.get("n_ph_work"),
        label=f"{logical_key} n_ph_work",
    )
    nph_reference = _integer(
        same_cutoff.get("n_ph_reference"),
        label=f"{logical_key} n_ph_reference",
    )
    if (
        nph_work != nph_reference
        or nph_work != _integer(
            problem.get("n_ph_max"),
            label=f"{logical_key} problem cutoff",
        )
        or nph_reference
        != _integer(
            source_ed.get("nph"),
            label=f"{logical_key} ED cutoff",
        )
        or same_cutoff.get("exact_target_label")
        != problem.get("exact_target_label")
        or same_cutoff.get("ed_receipt_path") != source_ed.get("path")
        or same_cutoff.get("ed_receipt_sha256") != source_ed.get("sha256")
        or same_cutoff.get("reference_role")
        != "same_cutoff_reporting_reference"
        or source_ed.get("reference_role")
        != "same_cutoff_reporting_reference"
    ):
        raise PackageContractError(
            f"G2 same-cutoff/ED identity drifted for {logical_key}."
        )
    require_sha256(
        same_cutoff.get("ed_receipt_sha256"),
        label=f"{logical_key} ED receipt SHA-256",
    )
    receipt["_protocol"] = protocol
    return receipt


def _pool_projection(pool: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "count": _integer(pool.get("count"), label="pool count"),
        "ordered_labels_sha256": require_sha256(
            pool.get("ordered_labels_sha256"),
            label="pool ordered-label SHA-256",
        ),
        "ordered_pool_sha256": require_sha256(
            pool.get("ordered_pool_sha256"),
            label="pool full SHA-256",
        ),
    }


def _validate_pool_authority(
    raw: Any,
    *,
    cells: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    receipt = dict(
        _digested_mapping(
            raw,
            label="G3 pool-construction receipt",
            schema=POOL_CONSTRUCTION_RECEIPT_SCHEMA,
        )
    )
    groups = _sequence(receipt.get("regime_groups"), label="G3 regime groups")
    by_regime: dict[str, Mapping[str, Any]] = {}
    for raw_group in groups:
        group = _mapping(raw_group, label="G3 regime group")
        regime_id = str(group.get("regime_id", ""))
        if regime_id in by_regime:
            raise PackageContractError(
                f"Duplicate G3 regime group: {regime_id}."
            )
        by_regime[regime_id] = group
    verify_exact_key_set(
        by_regime,
        VALIDATION_REGIMES,
        label="G3 regime groups",
    )

    for regime_id in VALIDATION_REGIMES:
        group = by_regime[regime_id]
        if _integer(group.get("nph"), label=f"{regime_id} G3 cutoff") != 3:
            raise PackageContractError(f"G3 cutoff drifted for {regime_id}.")
        macro = _pool_projection(
            _mapping(group.get("macro"), label=f"{regime_id} macro pool")
        )
        if (
            macro["count"] != MACRO_COUNT_NPH3
            or macro["ordered_labels_sha256"]
            != MACRO_ORDERED_LABELS_SHA256_NPH3
        ):
            raise PackageContractError(
                f"G3 stable macro membership drifted for {regime_id}."
            )
        macro_protocols = [
            cell["_protocol"]
            for cell in cells.values()
            if (
                cell["_protocol"]["problem"]["problem_request_sha256"]
                == group.get("problem_request_sha256")
                and cell["_protocol"]["candidate_representation"]
                == "macro_generator_v1"
            )
        ]
        if not macro_protocols or any(
            _pool_projection(
                _mapping(
                    protocol.get("executable_pool"),
                    label=f"{regime_id} executable macro pool",
                )
            )
            != macro
            for protocol in macro_protocols
        ):
            raise PackageContractError(
                f"G3 per-regime RA/Append macro full hashes drifted for "
                f"{regime_id}."
            )

        singleton = _mapping(
            group.get("singleton_construction"),
            label=f"{regime_id} singleton construction",
        )
        ra_parent = _pool_projection(
            _mapping(
                singleton.get("ra_parent"),
                label=f"{regime_id} singleton RA parent",
            )
        )
        append_parent = _pool_projection(
            _mapping(
                singleton.get("append_parent"),
                label=f"{regime_id} singleton Append parent",
            )
        )
        child = _pool_projection(
            _mapping(
                singleton.get("append_guarded_child"),
                label=f"{regime_id} guarded singleton child",
            )
        )
        if (
            ra_parent != append_parent
            or ra_parent["count"] != SINGLETON_PARENT_COUNT_NPH3
            or ra_parent["ordered_labels_sha256"]
            != SINGLETON_PARENT_LABELS_SHA256_NPH3
            or singleton.get("source_parent_ordered_labels_sha256")
            != ra_parent["ordered_labels_sha256"]
            or not child["count"]
        ):
            raise PackageContractError(
                f"G3 singleton construction equivalence drifted for "
                f"{regime_id}."
            )
        equivalence = _digested_mapping(
            singleton.get("construction_equivalence_receipt"),
            label=f"{regime_id} singleton construction equivalence",
            schema=SINGLETON_CONSTRUCTION_EQUIVALENCE_SCHEMA,
        )
        verify_exact_key_set(
            equivalence,
            (
                "schema",
                "regime_id",
                "problem_request_sha256",
                "ra_parent",
                "append_parent",
                "append_guarded_child",
                "ra_exposure_mode",
                "append_exposure_mode",
                "ra_staged_funnel_invoked",
                "append_ra_staged_funnel_invoked",
                "parent_identity_equal",
                "guarded_child_construction_passed",
                "guarded_child_inventory_receipt_sha256",
                "sha256",
            ),
            label=f"{regime_id} singleton construction equivalence",
        )
        guarded_child_receipt_sha256 = require_sha256(
            equivalence.get("guarded_child_inventory_receipt_sha256"),
            label=f"{regime_id} guarded-child inventory receipt",
        )
        if (
            equivalence.get("regime_id") != regime_id
            or equivalence.get("problem_request_sha256")
            != group.get("problem_request_sha256")
            or _pool_projection(
                _mapping(
                    equivalence.get("ra_parent"),
                    label=f"{regime_id} equivalence RA parent",
                )
            )
            != ra_parent
            or _pool_projection(
                _mapping(
                    equivalence.get("append_parent"),
                    label=f"{regime_id} equivalence Append parent",
                )
            )
            != append_parent
            or _pool_projection(
                _mapping(
                    equivalence.get("append_guarded_child"),
                    label=f"{regime_id} equivalence guarded child",
                )
            )
            != child
            or equivalence.get("ra_exposure_mode")
            != "staged_child_exposure_v1"
            or equivalence.get("append_exposure_mode")
            != "global_guarded_child_pool_v1"
            or equivalence.get("ra_staged_funnel_invoked") is not True
            or equivalence.get("append_ra_staged_funnel_invoked") is not False
            or equivalence.get("parent_identity_equal") is not True
            or equivalence.get("guarded_child_construction_passed") is not True
            or singleton.get("construction_receipt_sha256")
            != equivalence["sha256"]
            or guarded_child_receipt_sha256
            != equivalence["guarded_child_inventory_receipt_sha256"]
        ):
            raise PackageContractError(
                f"G3 singleton construction receipt drifted for "
                f"{regime_id}."
            )
        require_sha256(
            singleton.get("construction_receipt_sha256"),
            label=f"{regime_id} singleton construction receipt SHA-256",
        )
        singleton_protocols = [
            cell["_protocol"]
            for cell in cells.values()
            if (
                cell["_protocol"]["problem"]["problem_request_sha256"]
                == group.get("problem_request_sha256")
                and cell["_protocol"]["candidate_representation"]
                == "single_pauli_word_v1"
            )
        ]
        if len(singleton_protocols) != 2 or any(
            _pool_projection(
                _mapping(
                    protocol.get("parent_inventory"),
                    label=f"{regime_id} singleton protocol parent",
                )
            )
            != ra_parent
            for protocol in singleton_protocols
        ):
            raise PackageContractError(
                f"G3 singleton Study-1 parents drifted for {regime_id}."
            )
    receipt["_by_regime"] = by_regime
    return receipt


def _validate_trusted_authority(raw: Any) -> dict[str, Any]:
    receipt = dict(
        _digested_mapping(
            raw,
            label="G8 trusted-execution authority",
            schema=TRUSTED_EXECUTION_RECEIPT_SCHEMA,
        )
    )
    try:
        from pipelines.static_adapt.ra_adapt.exact_reference_isolation import (
            validate_study1_trusted_execution_receipt,
        )

        validated = validate_study1_trusted_execution_receipt(receipt)
    except (TypeError, ValueError) as exc:
        raise PackageContractError(
            f"G8 trusted source/dataflow validation failed: {exc}"
        ) from exc
    if (
        validated != receipt
        or receipt.get("controller_exact_reference_policy")
        != "reporting_only_after_controller_finalization_v1"
        or receipt.get("controller_exact_reference_inputs") != []
        or receipt.get("study1_protocol_requirement")
        != "request.execution.stop.exact_ed_target_is_none_v1"
        or receipt.get("source_dataflow_regression_passed") is not True
        or receipt.get("source_dataflow_regression_test_id")
        != (
            "test_study1_reporting_reference_differential_preserves_"
            "controller_trajectory_and_replay_v1"
        )
    ):
        raise PackageContractError(
            "G8 trusted execution authority permits exact-reference control."
        )
    for field in (
        "controller_instrumentation_sha256",
        "reporting_boundary_sha256",
        "source_dataflow_regression_receipt_sha256",
    ):
        require_sha256(receipt.get(field), label=f"G8 {field}")
    return receipt


def _validate_t13_authority(raw: Any) -> dict[str, Any]:
    receipt = dict(
        _digested_mapping(
            raw,
            label="generic T13 characterization receipt",
            schema=T13_CHARACTERIZATION_RECEIPT_SCHEMA,
        )
    )
    expected = {
        "fixture_contract_id": "historical_singleton_plateau_route_t13_v1",
        "problem_request_sha256": T13_PROBLEM_REQUEST_SHA256,
        "fixture_file_sha256": T13_FIXTURE_FILE_SHA256,
        "fixture_canonical_sha256": T13_FIXTURE_CANONICAL_SHA256,
        "status": "passed",
        "study1_problem_comparison_performed": False,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise PackageContractError(
            "Generic T13 characterization authority drifted."
        )
    require_sha256(
        receipt.get("route_contract_sha256"),
        label="T13 route-contract SHA-256",
    )
    return receipt


def validate_objective_gate_authority(
    *,
    receipt_path: Path,
    revision_root: Path,
    final_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate v8's compact pre-execution G1/G2/G3/G8/G13 authority."""

    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise PackageContractError(
            f"Missing regular objective-gate authority receipt: {receipt_path}"
        )
    payload = dict(
        _digested_mapping(
            load_json_object(
                receipt_path,
                label="Study-1 objective-gate authority",
            ),
            label="Study-1 objective-gate authority",
            schema=OBJECTIVE_GATE_AUTHORITY_SCHEMA,
        )
    )
    final_binding = _mapping(
        final_receipt.get("study1_objective_gate_authority"),
        label="v8 final objective-gate binding",
    )
    if (
        final_binding.get("path") != "study1_objective_gate_authority_receipt.json"
        or final_binding.get("canonical_sha256") != payload["sha256"]
        or final_binding.get("file_sha256") != sha256_file(receipt_path)
        or payload.get("package_id") != PACKAGE_ID
        or payload.get("materialization_revision") != "v8"
    ):
        raise PackageContractError(
            "v8 final receipt does not authenticate the objective-gate authority."
        )
    raw_cells = _sequence(
        payload.get("g1_g2_cell_receipts"),
        label="G1/G2 cell receipts",
    )
    cells: dict[str, dict[str, Any]] = {}
    for raw in raw_cells:
        cell = _validate_cell_authority(raw, revision_root=revision_root)
        logical_key = str(cell["logical_key"])
        if logical_key in cells:
            raise PackageContractError(
                f"Duplicate G1/G2 authority cell: {logical_key}."
            )
        cells[logical_key] = cell
    verify_exact_key_set(
        cells,
        logical_cell_keys(),
        label="G1/G2 authority logical cells",
    )
    pools = _validate_pool_authority(
        payload.get("g3_pool_construction_receipt"),
        cells=cells,
    )
    trusted = _validate_trusted_authority(
        payload.get("g8_trusted_execution_receipt")
    )
    t13 = _validate_t13_authority(
        payload.get("g13_t13_characterization_receipt")
    )
    return {
        "payload": payload,
        "sha256": payload["sha256"],
        "file_sha256": sha256_file(receipt_path),
        "cells": cells,
        "pools": pools,
        "trusted": trusted,
        "t13": t13,
    }


def _result_protocol(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(result.get("protocol"), label="result protocol")


def _result_scientific(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(
        result.get("scientific_receipts"),
        label="result scientific receipts",
    )


def _append_payload(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(
        result.get("result_payload"),
        label="Append result payload",
    )


def _ra_run(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(result.get("run"), label="RA run payload")


def _policy_receipt(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(result.get("policy"), label="result policy receipt")


def _gate_status(gate_id: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "status": "passed",
        "evidence": dict(evidence),
    }


def _validate_g4(
    *,
    job: Mapping[str, Any],
    protocol: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    append = job["execution_entrypoint"] == "run_append_adapt"
    expected = APPEND_REFIT_CHART if append else RA_REFIT_CHART
    scientific = _result_scientific(result)
    observed = scientific.get("accepted_refit_coordinate_chart")
    if (
        protocol.get("accepted_refit_coordinate_chart") != expected
        or observed != expected
    ):
        raise PackageContractError(
            f"G4 refit chart drifted for {job['execution_id']}."
        )
    return {"coordinate_chart": expected}


def _validate_g5(
    *,
    job: Mapping[str, Any],
    protocol: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    append = job["execution_entrypoint"] == "run_append_adapt"
    if protocol.get("derivative_chart_id") != EXACT_INSERTION_CHART:
        raise PackageContractError(
            f"G5 protocol chart drifted for {job['execution_id']}."
        )
    if append:
        history = _sequence(
            _append_payload(result).get("history"),
            label="Append accepted history",
        )
        for index, raw in enumerate(history):
            row = _mapping(raw, label=f"Append history[{index}]")
            if (
                _integer(
                    row.get("insertion_position"),
                    label=f"Append history[{index}] position",
                )
                != index
            ):
                raise PackageContractError(
                    f"G5 Append endpoint domain drifted for "
                    f"{job['execution_id']}."
                )
        return {
            "domain": "endpoint_only",
            "serialized_position_count": len(history),
            "interior_scored_count": 0,
        }

    accepted_rounds = _sequence(
        _result_scientific(result).get("accepted_round_receipts"),
        label=f"{job['execution_id']} accepted-round receipts",
    )
    if not accepted_rounds:
        raise PackageContractError(
            f"G5 has no accepted-round population receipts for "
            f"{job['execution_id']}."
        )
    interior = 0
    append_count = 0
    receipt_hashes: list[str] = []
    for round_index, raw_round in enumerate(accepted_rounds, start=1):
        accepted_round = _mapping(
            raw_round,
            label=f"{job['execution_id']} accepted round {round_index}",
        )
        if (
            _integer(
                accepted_round.get("accepted_round_ordinal"),
                label=f"G5 accepted round {round_index} ordinal",
                minimum=1,
            )
            != round_index
        ):
            raise PackageContractError(
                f"G5 accepted-round order drifted for {job['execution_id']}."
            )
        population = _digested_mapping(
            accepted_round.get("scored_insertion_position_population"),
            label=(
                f"{job['execution_id']} scored insertion population "
                f"{round_index}"
            ),
            schema=G5_EVIDENCE_SCHEMA,
        )
        if (
            population.get("coordinate_chart") != EXACT_INSERTION_CHART
            or population.get("phase_order")
            != ["phase_i", "phase_ii", "phase_iii"]
        ):
            raise PackageContractError(
                f"G5 scored-position chart drifted for "
                f"{job['execution_id']}."
            )
        append_position = _integer(
            population.get("append_position"),
            label=f"G5 round {round_index} append position",
        )
        phases = _sequence(
            population.get("phases"),
            label=f"G5 round {round_index} phase populations",
        )
        if len(phases) != 3:
            raise PackageContractError(
                f"G5 requires all three scored phases for "
                f"{job['execution_id']}."
            )
        observed_records = 0
        observed_interior = 0
        observed_append = 0
        phase_iii_identities: set[tuple[str, int]] = set()
        for phase_index, raw_phase in enumerate(phases):
            phase = _mapping(
                raw_phase,
                label=f"G5 round {round_index} phase {phase_index + 1}",
            )
            expected_phase = ("phase_i", "phase_ii", "phase_iii")[
                phase_index
            ]
            records = _sequence(
                phase.get("records"),
                label=f"G5 round {round_index} {expected_phase} records",
            )
            if (
                phase.get("phase") != expected_phase
                or not records
                or _integer(
                    phase.get("population_count"),
                    label=f"G5 {expected_phase} population count",
                )
                != len(records)
                or phase.get("ordered_population_sha256")
                != canonical_sha256(records)
            ):
                raise PackageContractError(
                    f"G5 scored {expected_phase} population drifted for "
                    f"{job['execution_id']}."
                )
            identities: set[tuple[str, str]] = set()
            for raw_record in records:
                record = _mapping(
                    raw_record,
                    label=f"G5 {expected_phase} scored record",
                )
                domain_id = str(record.get("domain_record_id", "")).strip()
                generator_id = str(record.get("generator_id", "")).strip()
                pool_label = str(record.get("pool_label", "")).strip()
                _integer(record.get("pool_index"), label="G5 pool index")
                position = _integer(
                    record.get("insertion_position"),
                    label="G5 insertion position",
                )
                expected_class = (
                    "interior" if position < append_position else "append"
                )
                if (
                    not domain_id
                    or not generator_id
                    or not pool_label
                    or position > append_position
                    or record.get("position_class") != expected_class
                    or (domain_id, generator_id) in identities
                ):
                    raise PackageContractError(
                        f"G5 scored record drifted for "
                        f"{job['execution_id']}."
                    )
                identities.add((domain_id, generator_id))
                if expected_phase == "phase_iii":
                    phase_iii_identities.add((generator_id, position))
                observed_records += 1
                observed_interior += int(expected_class == "interior")
                observed_append += int(expected_class == "append")
        if (
            observed_records
            != _integer(
                population.get("scored_record_count"),
                label="G5 scored-record total",
            )
            or observed_interior
            != _integer(
                population.get("interior_scored_count"),
                label="G5 interior-scored total",
            )
            or observed_append
            != _integer(
                population.get("append_scored_count"),
                label="G5 append-scored total",
            )
        ):
            raise PackageContractError(
                f"G5 scored population totals drifted for "
                f"{job['execution_id']}."
            )
        lineage = _sequence(
            accepted_round.get("accepted_candidate_lineage"),
            label=f"G5 round {round_index} accepted lineage",
        )
        if not lineage:
            raise PackageContractError(
                f"G5 accepted round lacks admitted lineage for "
                f"{job['execution_id']}."
            )
        for raw_lineage in lineage:
            admitted = _digested_mapping(
                raw_lineage,
                label=f"G5 round {round_index} accepted candidate lineage",
            )
            identity = (
                str(admitted.get("generator_identity", "")),
                _integer(
                    admitted.get("insertion_position"),
                    label="G5 accepted insertion position",
                ),
            )
            if not identity[0] or identity not in phase_iii_identities:
                raise PackageContractError(
                    f"G5 accepted admission was not in the scored Phase-III "
                    f"population for {job['execution_id']}."
                )
        interior += observed_interior
        append_count += observed_append
        receipt_hashes.append(str(population["sha256"]))
    route_id = str(job["route_id"])
    if route_id in {"ra_macro_plateau", "ra_macro_always", "singleton_plateau"}:
        if interior < 1:
            raise PackageContractError(
                f"G5 requires an interior scored receipt for "
                f"{job['execution_id']}."
            )
    elif route_id == "ra_macro_append_only" and interior != 0:
        raise PackageContractError(
            f"G5 append-only RA domain drifted for {job['execution_id']}."
        )
    return {
        "domain": (
            "endpoint_only"
            if route_id == "ra_macro_append_only"
            else "full_commutation_or_plateau"
        ),
        "accepted_round_population_count": len(accepted_rounds),
        "serialized_position_count": interior + append_count,
        "interior_scored_count": interior,
        "population_receipt_sha256s": receipt_hashes,
    }


def _validate_g6(
    *,
    job: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    if job["execution_entrypoint"] == "run_append_adapt":
        scientific = _result_scientific(result)
        if (
            scientific.get("phase3_solver_invoked") is not False
            or scientific.get("trust_transaction_invoked") is not False
        ):
            raise PackageContractError(
                f"G6 Append invoked RA Phase III for {job['execution_id']}."
            )
        return {"scope": "not_applicable_to_append"}
    rounds = _sequence(
        _result_scientific(result).get("accepted_round_receipts"),
        label=f"{job['execution_id']} RA accepted-round receipts",
    )
    if not rounds:
        raise PackageContractError(
            f"G6 has no accepted-round receipts for {job['execution_id']}."
        )
    replay_rows = _sequence(
        _ra_run(result).get("scientific_replay"),
        label=f"{job['execution_id']} scientific replay",
    )
    if len(replay_rows) != len(rounds):
        raise PackageContractError(
            f"G6 replay/accepted-round counts drifted for "
            f"{job['execution_id']}."
        )
    for index, (raw, raw_replay) in enumerate(
        zip(rounds, replay_rows, strict=True)
    ):
        row = _mapping(raw, label=f"G6 accepted round {index}")
        support = _mapping(row.get("retained_support"), label="G6 support")
        stabilization = _mapping(
            row.get("phase3_stabilization"),
            label="G6 stabilization",
        )
        trust = _mapping(
            row.get("source_gram_no_overlap_trust"),
            label="G6 trust",
        )
        replay = _mapping(raw_replay, label=f"G6 replay {index}")
        accepted_refit = _mapping(
            replay.get("accepted_refit"),
            label=f"G6 replay {index} accepted refit",
        )
        supported_metric = _mapping(
            accepted_refit.get("supported_metric"),
            label=f"G6 replay {index} supported metric",
        )
        kappa = _finite(
            stabilization.get("kappa_stabilization_shift"),
            label="G6 kappa",
        )
        boundary = _finite(
            stabilization.get("trust_boundary_multiplier_lambda"),
            label="G6 lambda",
        )
        total = _finite(
            stabilization.get("total_metric_multiplier_mu"),
            label="G6 mu",
        )
        if (
            _finite(
                support.get("rank_relative_tolerance"),
                label="G6 support threshold",
            )
            != 1.0e-6
            or stabilization.get("metric_whitening_active") is not False
            or stabilization.get("metric_inverse_sqrt_constructed")
            is not False
            or _finite(
                supported_metric.get("metric_regularization"),
                label="G6 metric ridge",
            )
            != 0.0
            or not math.isclose(
                total,
                kappa + boundary,
                rel_tol=0.0,
                abs_tol=128.0 * math.ulp(max(1.0, abs(total))),
            )
            or bool(stabilization.get("trust_boundary_active"))
            != bool(boundary > 0.0)
            or trust.get("supported_metric_whitening_active") is not False
            or trust.get("supported_metric_inverse_sqrt_constructed")
            is not False
            or _integer(
                trust.get("endpoint_overlap_query_charge"),
                label="G6 endpoint-overlap charge",
            )
            != 0
        ):
            raise PackageContractError(
                f"G6 Phase-III receipt drifted for {job['execution_id']}."
            )
    return {"accepted_round_receipt_count": len(rounds)}


def _validate_g7(
    *,
    job: Mapping[str, Any],
    protocol: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    policy = _policy_receipt(result)
    for field in ("active_gradient_policy", "resource_weighting_scope"):
        if policy.get(field) != protocol.get(field):
            raise PackageContractError(
                f"G7 policy echo drifted at {job['execution_id']}.{field}."
            )
    indices = _sequence(
        policy.get("active_gradient_indices_acquired"),
        label=f"{job['execution_id']} active-gradient indices",
    )
    charge = _integer(
        policy.get("active_gradient_charge"),
        label=f"{job['execution_id']} active-gradient charge",
    )
    if (
        protocol.get("active_gradient_policy")
        == "stationary_source_response_v1"
        and (indices or charge)
    ):
        raise PackageContractError(
            f"G7 stationary cell acquired active gradients: "
            f"{job['execution_id']}."
        )
    return {
        "active_gradient_policy": protocol["active_gradient_policy"],
        "resource_weighting_scope": protocol["resource_weighting_scope"],
        "active_gradient_indices_acquired": list(indices),
        "active_gradient_charge": charge,
    }


def _validate_g8(
    *,
    job: Mapping[str, Any],
    protocol: Mapping[str, Any],
    result: Mapping[str, Any],
    trusted: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _digested_mapping(
        _result_scientific(result).get(
            "study1_g8_exact_reference_isolation"
        ),
        label=f"{job['execution_id']} G8 evidence",
        schema=G8_EVIDENCE_SCHEMA,
    )
    try:
        from pipelines.static_adapt.ra_adapt.contracts import (
            resolved_ra_adapt_protocol_from_mapping,
        )
        from pipelines.static_adapt.ra_adapt.exact_reference_isolation import (
            validate_study1_exact_reference_isolation_receipt,
        )

        typed_protocol = resolved_ra_adapt_protocol_from_mapping(protocol)
        validated = validate_study1_exact_reference_isolation_receipt(
            receipt,
            protocol=typed_protocol,
            trusted_execution_receipt=trusted,
        )
    except (TypeError, ValueError) as exc:
        raise PackageContractError(
            f"G8 runtime attestation validation failed for "
            f"{job['execution_id']}: {exc}"
        ) from exc
    if (
        validated != receipt
        or receipt.get("protocol_sha256") != protocol.get("sha256")
        or receipt.get("controller_consumed_exact_reference") is not False
        or receipt.get("reference_usage")
        != "reporting_only_after_controller_finalization_v1"
        or receipt.get("controller_instrumentation_sha256")
        != trusted["controller_instrumentation_sha256"]
        or receipt.get("reporting_boundary_sha256")
        != trusted["reporting_boundary_sha256"]
    ):
        raise PackageContractError(
            f"G8 exact-reference isolation drifted for {job['execution_id']}."
        )
    events = _sequence(
        receipt.get("exact_reference_events"),
        label=f"{job['execution_id']} exact-reference events",
    )
    method = (
        "append_adapt"
        if job["execution_entrypoint"] == "run_append_adapt"
        else "ra_adapt"
    )
    if not events or any(
        not isinstance(event, Mapping)
        or event.get("phase") != "reporting_after_controller_finalization"
        or event.get("event_id")
        != "same_cutoff_exact_energy_reporting_projection_v1"
        or event.get("method") != method
        or _integer(
            event.get("finalized_controller_rounds"),
            label="G8 finalized controller rounds",
        )
        < 0
        or require_sha256(
            event.get("exact_reference_value_sha256"),
            label="G8 exact-reference value SHA-256",
        )
        != event.get("exact_reference_value_sha256")
        for event in events
    ):
        raise PackageContractError(
            f"G8 exact-reference event phase drifted for "
            f"{job['execution_id']}."
        )
    return {
        "evidence_sha256": receipt["sha256"],
        "exact_reference_event_count": len(events),
    }


def _g9_receipt(
    result: Mapping[str, Any],
    *,
    execution_id: str,
) -> Mapping[str, Any]:
    top = _mapping(
        result.get("numerical_physical_integrity"),
        label=f"{execution_id} top-level G9 evidence",
    )
    scientific = _result_scientific(result)
    copied = _mapping(
        scientific.get("numerical_physical_integrity"),
        label=f"{execution_id} scientific G9 evidence",
    )
    if top != copied:
        raise PackageContractError(
            f"G9 authenticated result copies disagree for {execution_id}."
        )
    if top.get("schema") != G9_EVIDENCE_SCHEMA:
        raise PackageContractError(
            f"G9 evidence schema drifted for {execution_id}."
        )
    expected_sha = require_sha256(
        scientific.get("numerical_physical_integrity_sha256"),
        label=f"{execution_id} G9 evidence SHA-256",
    )
    if canonical_sha256(top) != expected_sha:
        raise PackageContractError(
            f"G9 evidence digest drifted for {execution_id}."
        )
    if "result_payload" in result and (
        _mapping(
            _append_payload(result).get("numerical_physical_integrity"),
            label=f"{execution_id} Append G9 evidence",
        )
        != top
    ):
        raise PackageContractError(
            f"G9 Append evidence copies disagree for {execution_id}."
        )
    return top


def _validate_g9(
    *,
    job: Mapping[str, Any],
    protocol: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _g9_receipt(result, execution_id=str(job["execution_id"]))
    try:
        from pipelines.static_adapt.numerical_physical_integrity import (
            numerical_physical_integrity_from_mapping,
        )

        numerical_physical_integrity_from_mapping(receipt)
    except (TypeError, ValueError) as exc:
        raise PackageContractError(
            f"G9 typed integrity validation failed for "
            f"{job['execution_id']}: {exc}"
        ) from exc
    expected_method = (
        "append_adapt"
        if job["execution_entrypoint"] == "run_append_adapt"
        else "ra_adapt"
    )
    fixed_probability = _finite(
        receipt.get("fixed_count_sector_probability"),
        label="G9 fixed-count probability",
    )
    fixed_leak = _finite(
        receipt.get("fixed_count_sector_leak_probability"),
        label="G9 fixed-count leakage",
    )
    boson_legal = _finite(
        receipt.get("boson_legal_probability_min"),
        label="G9 boson legal probability",
    )
    boson_illegal = _finite(
        receipt.get("boson_illegal_probability_max"),
        label="G9 boson illegal probability",
    )
    if (
        receipt.get("method") != expected_method
        or receipt.get("reporting_only") is not True
        or receipt.get("controller_decision_influence") is not False
        or receipt.get("finite_values_passed") is not True
        or receipt.get("nonfinite_value_paths") != []
        or _integer(
            receipt.get("checked_energy_value_count"),
            label="G9 checked energy count",
            minimum=1,
        )
        < 1
        or _integer(
            receipt.get("checked_parameter_value_count"),
            label="G9 checked parameter count",
        )
        < 0
        or not str(receipt.get("state_fingerprint", "")).strip()
        or _finite(
            receipt.get("sector_leak_threshold"),
            label="G9 sector leak threshold",
        )
        != 1.0e-8
        or any(
            probability < 0.0 or probability > 1.0
            for probability in (
                fixed_probability,
                fixed_leak,
                boson_legal,
                boson_illegal,
            )
        )
        or not math.isclose(
            fixed_probability + fixed_leak,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        )
        or not math.isclose(
            boson_legal + boson_illegal,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        )
        or receipt.get("sector_leak_flag") is not False
        or receipt.get("boson_truncation_leak_flag") is not False
        or receipt.get("accepted_energy_integrity_passed") is not True
        or receipt.get("integrity_passed") is not True
    ):
        raise PackageContractError(
            f"G9 leakage diagnostics failed for {job['execution_id']}."
        )
    transitions = _sequence(
        receipt.get("accepted_energy_transitions"),
        label="G9 accepted-transition checks",
    )
    completed = _sequence(
        (
            _append_payload(result).get("history")
            if expected_method == "append_adapt"
            else _ra_run(result).get("accepted_transitions")
        ),
        label=f"{job['execution_id']} accepted transitions",
    )
    if len(transitions) != len(completed):
        raise PackageContractError(
            f"G9 accepted-transition count drifted for "
            f"{job['execution_id']}."
        )
    for index, raw in enumerate(transitions, start=1):
        row = _mapping(raw, label=f"G9 transition {index}")
        if (
            row.get("schema")
            != "paper_i_accepted_energy_transition_integrity_v1"
            or row.get("gate_passed") is not True
        ):
            raise PackageContractError(
                f"G9 transition receipt drifted for {job['execution_id']}."
            )
        before = _finite(row.get("energy_before"), label="G9 energy before")
        after = _finite(row.get("energy_after"), label="G9 energy after")
        tolerance = _finite(
            row.get("absolute_tolerance"),
            label="G9 non-worsening tolerance",
        )
        rollback = row.get("typed_rollback_receipt")
        nonincrease = after <= before + tolerance
        if (
            _integer(
                row.get("controller_round"),
                label="G9 transition round",
                minimum=1,
            )
            != index
            or tolerance != 0.0
            or row.get("nonincrease_passed") is not nonincrease
            or row.get("gate_passed")
            is not bool(nonincrease or isinstance(rollback, Mapping))
        ):
            raise PackageContractError(
                f"G9 transition semantics drifted for "
                f"{job['execution_id']}."
            )
        if not nonincrease and not isinstance(rollback, Mapping):
            raise PackageContractError(
                f"G9 worsening transition lacks rollback for "
                f"{job['execution_id']}."
            )
        if isinstance(rollback, Mapping):
            if not rollback:
                raise PackageContractError(
                    f"G9 transition {index} has an empty typed rollback."
                )
    return {
        "evidence_sha256": canonical_sha256(receipt),
        "accepted_transition_check_count": len(transitions),
        "sector_leak_flag": False,
        "boson_truncation_leak_flag": False,
    }


def _components(accounting: Mapping[str, Any]) -> dict[str, int]:
    raw = accounting.get("components", accounting)
    values = _mapping(raw, label="estimator-accounting components")
    resolved: dict[str, int] = {}
    for name in _COMPONENT_NAMES:
        value = values.get(name, values.get(_LOWER_COMPONENT_NAMES[name]))
        resolved[name] = _integer(value, label=f"accounting {name}")
    return resolved


def _accounting(result: Mapping[str, Any], *, append: bool) -> Mapping[str, Any]:
    if append:
        return _mapping(
            _append_payload(result).get("estimator_accounting"),
            label="Append estimator accounting",
        )
    return _mapping(
        _ra_run(result).get("estimator_accounting"),
        label="RA estimator accounting",
    )


def _g10_ledger_payload(
    ledger: Mapping[str, Any],
    *,
    append: bool,
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
    if append:
        if ledger.get("schema") != ESTIMATOR_LEDGER_SCHEMA:
            raise PackageContractError(
                "G10 Append estimator ledger schema drifted."
            )
        return ledger, None

    if ledger.get("schema") != RA_ESTIMATOR_LEDGER_SIDECAR_SCHEMA:
        raise PackageContractError(
            "G10 RA estimator-ledger sidecar schema drifted."
        )
    if ledger.get("adapt_success") is not True or ledger.get(
        "adapt_error"
    ) not in {None, ""}:
        raise PackageContractError(
            "G10 RA estimator-ledger sidecar is not complete."
        )
    sidecar_accounting = _mapping(
        ledger.get("accounting"),
        label="RA estimator-ledger sidecar accounting",
    )
    if (
        sidecar_accounting.get("schema")
        != RA_ESTIMATOR_ACCOUNTING_SCHEMA
        or sidecar_accounting.get("enabled") is not True
        or sidecar_accounting.get("complete") is not True
        or sidecar_accounting.get("status")
        != "resolved_from_live_state_keyed_instrumentation"
        or sidecar_accounting.get("exact_blockers") != []
    ):
        raise PackageContractError(
            "G10 RA estimator-ledger sidecar accounting is incomplete."
        )
    payload = _mapping(
        ledger.get("ledger"),
        label="RA estimator-ledger sidecar payload",
    )
    if payload.get("schema") != ESTIMATOR_LEDGER_SCHEMA:
        raise PackageContractError(
            "G10 RA nested estimator ledger schema drifted."
        )
    return payload, sidecar_accounting


def _validate_g10(
    *,
    job: Mapping[str, Any],
    result: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    append = job["execution_entrypoint"] == "run_append_adapt"
    accounting = _accounting(result, append=append)
    work = (
        accounting
        if append
        else _mapping(
            accounting.get("all_work"),
            label="RA all-executed estimator work",
        )
    )
    components = _components(work)
    s_alg = _integer(
        work.get("S_alg", work.get("s_alg")),
        label="accounting S_alg",
    )
    if s_alg != sum(components.values()):
        raise PackageContractError(
            f"G10 component closure failed for {job['execution_id']}."
        )
    if append:
        if (
            accounting.get("closed_occurrence_reconciliation") is not True
            or components["N_metric"] != 0
        ):
            raise PackageContractError(
                f"G10 Append closure/chart charges failed for "
                f"{job['execution_id']}."
            )
    elif (
        accounting.get("complete") is not True
        or accounting.get("status")
        != "resolved_from_live_state_keyed_instrumentation"
        or accounting.get("exact_blockers") != []
        or accounting.get("prefix_closure_passed") is not True
        or accounting.get("prefix_closure_status") != "complete"
        or _integer(
            accounting.get("raw_occurrence_total"),
            label="RA raw occurrence total",
        )
        != s_alg
        or _components(
            _mapping(
                accounting.get("raw_occurrences"),
                label="RA raw occurrence components",
            )
        )
        != components
    ):
        raise PackageContractError(
            f"G10 RA accounting is incomplete for {job['execution_id']}."
        )
    ledger_payload, sidecar_accounting = _g10_ledger_payload(
        ledger,
        append=append,
    )
    if sidecar_accounting is not None:
        sidecar_components = _components(sidecar_accounting)
        sidecar_s_alg = _integer(
            sidecar_accounting.get("S_alg"),
            label="RA sidecar-accounting S_alg",
        )
        if (
            sidecar_components != components
            or sidecar_s_alg != s_alg
            or sidecar_s_alg != sum(sidecar_components.values())
        ):
            raise PackageContractError(
                f"G10 RA sidecar accounting closure failed for "
                f"{job['execution_id']}."
            )
    occurrence = _mapping(
        ledger_payload.get("occurrence_summary"),
        label="estimator-ledger occurrence summary",
    )
    for name, expected in (*components.items(), ("S_alg", s_alg)):
        if _integer(
            occurrence.get(name),
            label=f"ledger occurrence {name}",
        ) != expected:
            raise PackageContractError(
                f"G10 ledger occurrence closure failed for "
                f"{job['execution_id']}."
            )
    if append:
        forbidden = [
            raw
            for raw in _sequence(
                ledger_payload.get("occurrences"),
                label="Append ledger occurrences",
            )
            if isinstance(raw, Mapping)
            and (
                "whiten" in str(raw.get("event_kind", "")).lower()
                or "whiten" in str(raw.get("provenance", "")).lower()
            )
        ]
        if forbidden:
            raise PackageContractError(
                f"G10 Append ledger contains whitening charges for "
                f"{job['execution_id']}."
            )
    return {
        "components": components,
        "S_alg": s_alg,
        "ledger_canonical_sha256": canonical_sha256(ledger),
    }


def _g11_receipt(
    result: Mapping[str, Any],
    *,
    execution_id: str,
) -> Mapping[str, Any]:
    scientific = _result_scientific(result)
    evidence = _mapping(
        scientific.get("controller_replay_evidence"),
        label=f"{execution_id} G11 evidence",
    )
    if evidence.get("schema") != G11_EVIDENCE_SCHEMA:
        raise PackageContractError(
            f"G11 evidence schema drifted for {execution_id}."
        )
    try:
        from pipelines.static_adapt.ra_adapt.replay_evidence import (
            validate_controller_replay_evidence,
        )

        validated = validate_controller_replay_evidence(evidence)
    except (TypeError, ValueError) as exc:
        raise PackageContractError(
            f"G11 evidence validation failed for {execution_id}: {exc}"
        ) from exc
    expected_sha = require_sha256(
        scientific.get("controller_replay_evidence_sha256"),
        label=f"{execution_id} G11 evidence SHA-256",
    )
    if validated["sha256"] != expected_sha:
        raise PackageContractError(
            f"G11 evidence digest drifted for {execution_id}."
        )
    if "result_payload" in result and (
        _mapping(
            _append_payload(result).get("controller_replay_evidence"),
            label=f"{execution_id} Append G11 evidence",
        )
        != validated
    ):
        raise PackageContractError(
            f"G11 Append evidence copies disagree for {execution_id}."
        )
    return validated


def _validated_replay_evidence(
    value: Any,
    *,
    label: str,
) -> Mapping[str, Any]:
    try:
        from pipelines.static_adapt.ra_adapt.replay_evidence import (
            validate_controller_replay_evidence,
        )

        return validate_controller_replay_evidence(
            _mapping(value, label=label)
        )
    except (TypeError, ValueError) as exc:
        raise PackageContractError(
            f"{label} failed replay-evidence validation: {exc}"
        ) from exc


def _replay_comparison(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    controller_round: int,
    label: str,
) -> Mapping[str, Any]:
    try:
        from pipelines.static_adapt.ra_adapt.replay_evidence import (
            compare_bounded_controller_replays,
        )

        comparison = compare_bounded_controller_replays(
            first,
            second,
            controller_round=controller_round,
        )
    except (TypeError, ValueError) as exc:
        raise PackageContractError(
            f"{label} failed bounded replay comparison: {exc}"
        ) from exc
    if comparison.get("schema") != G11_COMPARISON_SCHEMA:
        raise PackageContractError(f"{label} comparison schema drifted.")
    return comparison


def build_g11_replay_diagnostic(
    *,
    job: Mapping[str, Any],
    primary_result: Mapping[str, Any],
    secondary_result: Mapping[str, Any] | None,
    resumed_result: Mapping[str, Any] | None,
    resume_checkpoint_file_sha256: str | None,
) -> dict[str, Any]:
    """Build the worker-owned proof of independent replay and RA continuation."""

    contract = _mapping(
        job.get("objective_gate_diagnostics"),
        label=f"{job['execution_id']} G11 diagnostic contract",
    )
    selected = _bool(
        contract.get("selected"),
        label=f"{job['execution_id']} G11 selected",
    )
    primary = _g11_receipt(
        primary_result,
        execution_id=str(job["execution_id"]),
    )
    base: dict[str, Any] = {
        "schema": G11_DIAGNOSTIC_RECEIPT_SCHEMA,
        "execution_id": str(job["execution_id"]),
        "diagnostic_contract_sha256": canonical_sha256(contract),
        "selected": selected,
        "method_family": str(primary["method_family"]),
        "primary_controller_replay_evidence_sha256": primary["sha256"],
    }
    if not selected:
        if (
            secondary_result is not None
            or resumed_result is not None
            or resume_checkpoint_file_sha256 is not None
        ):
            raise PackageContractError(
                "An unselected G11 job performed an unauthorized diagnostic."
            )
        return digested(
            {
                **base,
                "status": "not_selected_by_fixed_contract",
                "total_facade_execution_count": 1,
            }
        )

    round_index = _integer(
        contract.get("bounded_controller_round"),
        label=f"{job['execution_id']} bounded replay round",
        minimum=1,
    )
    if secondary_result is None:
        raise PackageContractError(
            f"Selected G11 job lacks an independent replay: "
            f"{job['execution_id']}."
        )
    secondary = _g11_receipt(
        secondary_result,
        execution_id=f"{job['execution_id']} secondary",
    )
    comparison = _replay_comparison(
        primary,
        secondary,
        controller_round=round_index,
        label=f"{job['execution_id']} independent replay",
    )
    primary_trajectory = _trajectory_projection(primary_result)
    secondary_trajectory = _trajectory_projection(secondary_result)
    primary_prefix = primary_trajectory[:round_index]
    secondary_prefix = secondary_trajectory[:round_index]
    bounded_prefix_equal = (
        len(primary_prefix) == round_index
        and len(secondary_prefix) == round_index
        and primary_prefix == secondary_prefix
    )
    if (
        comparison.get("matched") is not True
        or not bounded_prefix_equal
        or contract.get("independent_fresh_execution_required") is not True
    ):
        raise PackageContractError(
            f"Selected G11 independent replay failed for "
            f"{job['execution_id']}."
        )
    method = str(primary["method_family"])
    base.update(
        {
            "status": "passed",
            "bounded_controller_round": round_index,
            "independent_fresh_execution_performed": True,
            "independent_execution_isolation": (
                "separate_temporary_observation_root_v1"
            ),
            "secondary_controller_replay_evidence": dict(secondary),
            "bounded_replay_comparison": dict(comparison),
            "primary_trajectory_sha256": canonical_sha256(
                primary_trajectory
            ),
            "secondary_trajectory": secondary_trajectory,
            "secondary_trajectory_sha256": canonical_sha256(
                secondary_trajectory
            ),
            "bounded_trajectory_prefix_sha256": canonical_sha256(
                primary_prefix
            ),
            "bounded_trajectory_prefix_identity_equal": True,
            "full_trajectory_identity_equal": (
                primary_trajectory == secondary_trajectory
            ),
        }
    )
    if method == "ra_adapt":
        fresh_leg_rounds = _integer(
            contract.get("ra_fresh_leg_maximum_controller_rounds"),
            label=f"{job['execution_id']} fresh-leg horizon",
            minimum=1,
        )
        resumed_rounds = _integer(
            contract.get("ra_resumed_maximum_controller_rounds"),
            label=f"{job['execution_id']} resumed horizon",
            minimum=1,
        )
        if (
            contract.get("authenticated_ra_continuation_required")
            is not True
            or fresh_leg_rounds != 2
            or resumed_rounds != 3
            or resumed_result is None
            or resume_checkpoint_file_sha256 is None
        ):
            raise PackageContractError(
                f"Selected RA G11 job lacks an authenticated continuation: "
                f"{job['execution_id']}."
            )
        resumed = _g11_receipt(
            resumed_result,
            execution_id=f"{job['execution_id']} resumed",
        )
        resume_comparison = _replay_comparison(
            secondary,
            resumed,
            controller_round=fresh_leg_rounds,
            label=f"{job['execution_id']} authenticated continuation",
        )
        require_sha256(
            resume_checkpoint_file_sha256,
            label=f"{job['execution_id']} resume checkpoint file SHA-256",
        )
        resumed_trajectory = _trajectory_projection(resumed_result)
        secondary_round_count = len(
            _sequence(
                secondary.get("signed_controller_round_prefixes"),
                label="G11 fresh-leg signed prefixes",
            )
        )
        resumed_round_count = len(
            _sequence(
                resumed.get("signed_controller_round_prefixes"),
                label="G11 resumed signed prefixes",
            )
        )
        post_resume_round_count = resumed_round_count - secondary_round_count
        if (
            resume_comparison.get("matched") is not True
            or secondary_round_count != fresh_leg_rounds
            or resumed_round_count != resumed_rounds
            or post_resume_round_count < 1
            or len(secondary_trajectory) != fresh_leg_rounds
            or len(resumed_trajectory) != resumed_rounds
            or resumed_trajectory[:fresh_leg_rounds]
            != secondary_trajectory
            or primary_trajectory[:resumed_rounds] != resumed_trajectory
        ):
            raise PackageContractError(
                f"Selected RA authenticated continuation drifted for "
                f"{job['execution_id']}."
            )
        base.update(
            {
                "total_facade_execution_count": 3,
                "authenticated_ra_continuation_performed": True,
                "resume_checkpoint_file_sha256": (
                    resume_checkpoint_file_sha256
                ),
                "resumed_controller_replay_evidence": dict(resumed),
                "resume_bounded_replay_comparison": dict(
                    resume_comparison
                ),
                "resumed_from_round": fresh_leg_rounds,
                "resumed_final_round": resumed_rounds,
                "post_resume_controller_round_count": (
                    post_resume_round_count
                ),
                "resumed_trajectory": resumed_trajectory,
                "resumed_trajectory_sha256": canonical_sha256(
                    resumed_trajectory
                ),
                "resumed_prefix_matches_fresh_leg": True,
                "resumed_trajectory_matches_primary_prefix": True,
                "ra_resume_round_trip_status": (
                    "authenticated_continuation_identity_verified"
                ),
                "append_resume_boundary_status": "not_applicable",
            }
        )
    elif method == "append_adapt":
        resume = _mapping(
            primary.get("resume_sidecar_closure"),
            label=f"{job['execution_id']} Append resume boundary",
        )
        if (
            resumed_result is not None
            or resume_checkpoint_file_sha256 is not None
            or contract.get("authenticated_ra_continuation_required")
            is not False
            or contract.get("append_resume_boundary")
            != "authenticated_reconstruction_only_v1"
            or resume.get("resume_mode")
            != "authenticated_reconstruction_only_v1"
            or resume.get("public_resume_execution_supported") is not False
            or resume.get("reconstruction_fields_complete") is not True
            or primary_trajectory != secondary_trajectory
        ):
            raise PackageContractError(
                f"Selected Append replay crossed its non-resume boundary: "
                f"{job['execution_id']}."
            )
        base.update(
            {
                "total_facade_execution_count": 2,
                "authenticated_ra_continuation_performed": False,
                "ra_resume_round_trip_status": "not_applicable",
                "append_resume_boundary_status": (
                    "authenticated_reconstruction_only_verified"
                ),
            }
        )
    else:
        raise PackageContractError(
            f"Unknown G11 diagnostic method family: {method}."
        )
    return digested(base)


def _validate_g11_replay_diagnostic(
    *,
    job: Mapping[str, Any],
    primary_result: Mapping[str, Any],
    replay_diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    diagnostic = _digested_mapping(
        replay_diagnostic,
        label=f"{job['execution_id']} G11 replay diagnostic",
        schema=G11_DIAGNOSTIC_RECEIPT_SCHEMA,
    )
    contract = _mapping(
        job.get("objective_gate_diagnostics"),
        label=f"{job['execution_id']} G11 diagnostic contract",
    )
    selected = _bool(
        contract.get("selected"),
        label=f"{job['execution_id']} G11 selected",
    )
    primary = _g11_receipt(
        primary_result,
        execution_id=str(job["execution_id"]),
    )
    method = str(primary["method_family"])
    if (
        diagnostic.get("execution_id") != job["execution_id"]
        or diagnostic.get("diagnostic_contract_sha256")
        != canonical_sha256(contract)
        or diagnostic.get("selected") is not selected
        or diagnostic.get("method_family") != method
        or diagnostic.get("primary_controller_replay_evidence_sha256")
        != primary["sha256"]
    ):
        raise PackageContractError(
            f"G11 diagnostic authority drifted for {job['execution_id']}."
        )
    if not selected:
        if (
            diagnostic.get("status") != "not_selected_by_fixed_contract"
            or _integer(
                diagnostic.get("total_facade_execution_count"),
                label="G11 unselected execution count",
                minimum=1,
            )
            != 1
        ):
            raise PackageContractError(
                f"G11 unselected diagnostic drifted for "
                f"{job['execution_id']}."
            )
        return {
            "selected_for_method_regime_replay": False,
            "method_family": method,
            "bounded_replay_status": "not_selected_by_fixed_contract",
            "ra_resume_round_trip_status": "not_selected",
            "append_resume_boundary_status": "not_selected",
            "replay_diagnostic_sha256": diagnostic["sha256"],
        }

    round_index = _integer(
        contract.get("bounded_controller_round"),
        label="G11 bounded controller round",
        minimum=1,
    )
    secondary = _validated_replay_evidence(
        diagnostic.get("secondary_controller_replay_evidence"),
        label=f"{job['execution_id']} secondary replay",
    )
    comparison = _replay_comparison(
        primary,
        secondary,
        controller_round=round_index,
        label=f"{job['execution_id']} fetched independent replay",
    )
    primary_trajectory = _trajectory_projection(primary_result)
    secondary_trajectory = list(
        _sequence(
            diagnostic.get("secondary_trajectory"),
            label="G11 secondary trajectory",
        )
    )
    bounded_prefix = primary_trajectory[:round_index]
    bounded_prefix_equal = (
        len(bounded_prefix) == round_index
        and secondary_trajectory[:round_index] == bounded_prefix
    )
    full_trajectory_equal = (
        primary_trajectory == secondary_trajectory
    )
    if (
        diagnostic.get("status") != "passed"
        or diagnostic.get("independent_fresh_execution_performed")
        is not True
        or diagnostic.get("independent_execution_isolation")
        != "separate_temporary_observation_root_v1"
        or diagnostic.get("bounded_replay_comparison") != comparison
        or diagnostic.get("primary_trajectory_sha256")
        != canonical_sha256(primary_trajectory)
        or diagnostic.get("secondary_trajectory_sha256")
        != canonical_sha256(secondary_trajectory)
        or diagnostic.get("bounded_trajectory_prefix_sha256")
        != canonical_sha256(bounded_prefix)
        or diagnostic.get("bounded_trajectory_prefix_identity_equal")
        is not True
        or not bounded_prefix_equal
        or diagnostic.get("full_trajectory_identity_equal")
        is not full_trajectory_equal
    ):
        raise PackageContractError(
            f"G11 independent replay diagnostic drifted for "
            f"{job['execution_id']}."
        )
    ra_status = str(diagnostic.get("ra_resume_round_trip_status", ""))
    append_status = str(
        diagnostic.get("append_resume_boundary_status", "")
    )
    if method == "ra_adapt":
        fresh_leg_rounds = _integer(
            contract.get("ra_fresh_leg_maximum_controller_rounds"),
            label="G11 fresh-leg horizon",
            minimum=1,
        )
        resumed_rounds = _integer(
            contract.get("ra_resumed_maximum_controller_rounds"),
            label="G11 resumed horizon",
            minimum=1,
        )
        resumed = _validated_replay_evidence(
            diagnostic.get("resumed_controller_replay_evidence"),
            label=f"{job['execution_id']} resumed replay",
        )
        resume_comparison = _replay_comparison(
            secondary,
            resumed,
            controller_round=fresh_leg_rounds,
            label=f"{job['execution_id']} fetched resumed replay",
        )
        resumed_trajectory = list(
            _sequence(
                diagnostic.get("resumed_trajectory"),
                label="G11 resumed trajectory",
            )
        )
        secondary_round_count = len(
            _sequence(
                secondary.get("signed_controller_round_prefixes"),
                label="G11 fresh-leg signed prefixes",
            )
        )
        resumed_round_count = len(
            _sequence(
                resumed.get("signed_controller_round_prefixes"),
                label="G11 resumed signed prefixes",
            )
        )
        post_resume_round_count = resumed_round_count - secondary_round_count
        if (
            _integer(
                diagnostic.get("total_facade_execution_count"),
                label="G11 RA execution count",
                minimum=1,
            )
            != 3
            or diagnostic.get("authenticated_ra_continuation_performed")
            is not True
            or diagnostic.get("resume_bounded_replay_comparison")
            != resume_comparison
            or fresh_leg_rounds != 2
            or resumed_rounds != 3
            or secondary_round_count != fresh_leg_rounds
            or resumed_round_count != resumed_rounds
            or diagnostic.get("resumed_from_round") != fresh_leg_rounds
            or diagnostic.get("resumed_final_round") != resumed_rounds
            or diagnostic.get("post_resume_controller_round_count")
            != post_resume_round_count
            or post_resume_round_count < 1
            or require_sha256(
                diagnostic.get("resume_checkpoint_file_sha256"),
                label="G11 resume checkpoint SHA-256",
            )
            != diagnostic.get("resume_checkpoint_file_sha256")
            or len(secondary_trajectory) != fresh_leg_rounds
            or len(resumed_trajectory) != resumed_rounds
            or resumed_trajectory[:fresh_leg_rounds]
            != secondary_trajectory
            or primary_trajectory[:resumed_rounds] != resumed_trajectory
            or diagnostic.get("resumed_trajectory_sha256")
            != canonical_sha256(resumed_trajectory)
            or diagnostic.get("resumed_prefix_matches_fresh_leg")
            is not True
            or diagnostic.get(
                "resumed_trajectory_matches_primary_prefix"
            )
            is not True
            or ra_status
            != "authenticated_continuation_identity_verified"
            or append_status != "not_applicable"
        ):
            raise PackageContractError(
                f"G11 RA continuation diagnostic drifted for "
                f"{job['execution_id']}."
            )
    elif method == "append_adapt":
        if (
            _integer(
                diagnostic.get("total_facade_execution_count"),
                label="G11 Append execution count",
                minimum=1,
            )
            != 2
            or diagnostic.get("authenticated_ra_continuation_performed")
            is not False
            or diagnostic.get("full_trajectory_identity_equal") is not True
            or ra_status != "not_applicable"
            or append_status
            != "authenticated_reconstruction_only_verified"
        ):
            raise PackageContractError(
                f"G11 Append reconstruction boundary drifted for "
                f"{job['execution_id']}."
            )
    return {
        "selected_for_method_regime_replay": True,
        "method_family": method,
        "bounded_replay_status": "identity_verified",
        "ra_resume_round_trip_status": ra_status,
        "append_resume_boundary_status": append_status,
        "replay_diagnostic_sha256": diagnostic["sha256"],
        "bounded_trajectory_prefix_identity_equal": True,
        "full_trajectory_identity_equal": full_trajectory_equal,
    }


def _validate_g11(
    *,
    job: Mapping[str, Any],
    protocol: Mapping[str, Any],
    result: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_file_sha256: str,
    checkpoint_size_bytes: int,
    ledger_file_sha256: str,
    ledger_size_bytes: int,
    replay_diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _g11_receipt(result, execution_id=str(job["execution_id"]))
    method = (
        "append_adapt"
        if job["execution_entrypoint"] == "run_append_adapt"
        else "ra_adapt"
    )
    if (
        receipt.get("protocol_sha256") != protocol.get("sha256")
        or receipt.get("problem_request_sha256")
        != protocol["problem"]["problem_request_sha256"]
        or receipt.get("method_family") != method
    ):
        raise PackageContractError(
            f"G11 resume authentication drifted for {job['execution_id']}."
        )
    prefixes = _sequence(
        receipt.get("signed_controller_round_prefixes"),
        label=f"{job['execution_id']} signed prefixes",
    )
    completed = len(
        _sequence(
            (
                _append_payload(result).get("history")
                if method == "append_adapt"
                else _ra_run(result).get("accepted_trajectory")
            ),
            label=f"{job['execution_id']} completed trajectory",
        )
    )
    if len(prefixes) != completed:
        raise PackageContractError(
            f"G11 signed-prefix count drifted for {job['execution_id']}."
        )
    previous_digest: str | None = None
    for index, raw in enumerate(prefixes, start=1):
        prefix = _digested_mapping(
            raw,
            label=f"G11 signed prefix {index}",
            schema=SIGNED_CONTROLLER_PREFIX_SCHEMA,
        )
        prefix_checkpoint = _mapping(
            prefix.get("active_prefix_checkpoint"),
            label=f"G11 active-prefix checkpoint {index}",
        )
        expected_checkpoint_schema = (
            APPEND_SIGNED_PREFIX_SCHEMA
            if method == "append_adapt"
            else RA_SIGNED_PREFIX_SCHEMA
        )
        if prefix_checkpoint.get("schema") != expected_checkpoint_schema:
            raise PackageContractError(
                f"G11 signed-prefix schema drifted for {job['execution_id']}."
            )
        checkpoint_digest = require_sha256(
            prefix_checkpoint.get("checkpoint_sha256"),
            label=f"G11 checkpoint {index} digest",
        )
        checkpoint_unsigned = dict(prefix_checkpoint)
        del checkpoint_unsigned["checkpoint_sha256"]
        if (
            canonical_sha256(checkpoint_unsigned) != checkpoint_digest
            or _integer(
                prefix.get("controller_round"),
                label=f"G11 prefix {index} round",
            )
            != index
            or prefix.get("preceding_signed_prefix_sha256")
            != previous_digest
            or prefix.get("source_checkpoint_sha256") != checkpoint_digest
        ):
            raise PackageContractError(
                f"G11 signed-prefix chain drifted for {job['execution_id']}."
            )
        previous_digest = str(prefix["sha256"])
    replay = _mapping(
        receipt.get("bounded_replay_identity"),
        label=f"{job['execution_id']} bounded replay",
    )
    resume = _mapping(
        receipt.get("resume_sidecar_closure"),
        label=f"{job['execution_id']} resume round trip",
    )
    require_sha256(
        checkpoint_file_sha256,
        label=f"{job['execution_id']} transported checkpoint SHA-256",
    )
    require_sha256(
        ledger_file_sha256,
        label=f"{job['execution_id']} transported ledger SHA-256",
    )
    _integer(
        checkpoint_size_bytes,
        label=f"{job['execution_id']} transported checkpoint size",
        minimum=1,
    )
    _integer(
        ledger_size_bytes,
        label=f"{job['execution_id']} transported ledger size",
        minimum=1,
    )
    require_sha256(
        replay.get("scientific_input_sha256"),
        label=f"{job['execution_id']} G11 scientific input",
    )
    if method == "ra_adapt":
        checkpoint_artifact = _mapping(
            resume.get("checkpoint_artifact"),
            label=f"{job['execution_id']} checkpoint artifact",
        )
        estimator_artifact = _mapping(
            resume.get("estimator_ledger_artifact"),
            label=f"{job['execution_id']} estimator-ledger artifact",
        )
        adapt_checkpoint = _mapping(
            checkpoint.get("adapt_vqe"),
            label=f"{job['execution_id']} transported RA checkpoint",
        )
        if (
            resume.get("resume_mode")
            != "canonical_accepted_state_resume_v1"
            or resume.get("public_resume_execution_supported") is not True
            or resume.get("authentication_binding_complete") is not True
            or resume.get("checkpoint_artifact_available") is not True
            or resume.get("estimator_ledger_artifact_available") is not True
            or checkpoint_artifact.get("every_controller_rounds") != 1
            or checkpoint_artifact.get("sha256")
            != checkpoint_file_sha256
            or _integer(
                checkpoint_artifact.get("size_bytes"),
                label="G11 checkpoint artifact size",
                minimum=1,
            )
            != checkpoint_size_bytes
            or estimator_artifact.get("sha256") != ledger_file_sha256
            or _integer(
                estimator_artifact.get("size_bytes"),
                label="G11 estimator artifact size",
                minimum=1,
            )
            != ledger_size_bytes
            or adapt_checkpoint.get("terminal_active_prefix_checkpoint")
            != resume.get("terminal_signed_prefix_checkpoint")
            or _mapping(
                resume.get("estimator_prefix_closure"),
                label=f"{job['execution_id']} estimator-prefix closure",
            ).get("passed")
            is not True
        ):
            raise PackageContractError(
                f"G11 RA resume closure failed for "
                f"{job['execution_id']}."
            )
    if method == "append_adapt" and (
        resume.get("resume_mode") != "authenticated_reconstruction_only_v1"
        or resume.get("public_resume_execution_supported") is not False
        or resume.get("reconstruction_fields_complete") is not True
        or resume.get("continuation_execution_status")
        != "not_authorized_append_resume_contract"
        or checkpoint.get("schema") != "paper_i_append_adapt_checkpoint_v1"
        or checkpoint.get("protocol_sha256") != protocol.get("sha256")
        or checkpoint.get("controller_rounds_completed") != completed
        or checkpoint.get("controller_replay_evidence") != receipt
    ):
        raise PackageContractError(
            f"G11 Append resume boundary drifted for {job['execution_id']}."
        )
    diagnostic = _validate_g11_replay_diagnostic(
        job=job,
        primary_result=result,
        replay_diagnostic=replay_diagnostic,
    )
    return {
        "evidence_sha256": receipt["sha256"],
        "signed_prefix_count": len(prefixes),
        "method_family": method,
        "scientific_input_sha256": replay["scientific_input_sha256"],
        "resume_mode": resume["resume_mode"],
        "transported_checkpoint_file_sha256": checkpoint_file_sha256,
        "transported_checkpoint_size_bytes": checkpoint_size_bytes,
        "transported_ledger_file_sha256": ledger_file_sha256,
        "transported_ledger_size_bytes": ledger_size_bytes,
        **diagnostic,
    }


def _resource_counts(resources: Mapping[str, Any]) -> dict[str, int]:
    aliases = {
        "N2q": ("compiled_two_qubit_count", "compiled_two_qubit_gate_count"),
        "D2q": ("compiled_two_qubit_depth",),
        "Dc": ("compiled_total_depth", "compiled_circuit_depth"),
    }
    resolved: dict[str, int] = {}
    for label, names in aliases.items():
        found = next((resources.get(name) for name in names if name in resources), None)
        resolved[label] = _integer(found, label=f"G12 {label}")
    return resolved


def _validate_compile_identity(
    compile_identity: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if (
        compile_identity.get("policy") != TABLE_I_COMPILE_ID
        or _integer(
            compile_identity.get("optimization_level"),
            label=f"{label} optimization level",
        )
        != 0
        or _integer(
            compile_identity.get("transpiler_seed"),
            label=f"{label} transpiler seed",
        )
        != 7
        or compile_identity.get("coupling_map") is not None
        or compile_identity.get("reference_preparation_included") is not True
    ):
        raise PackageContractError(f"{label} compile identity drifted.")


def _validate_g12(
    *,
    job: Mapping[str, Any],
    protocol: Mapping[str, Any],
    result: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_compile_identity(
        _mapping(protocol.get("compile_identity"), label="protocol compile identity"),
        label=f"{job['execution_id']} protocol",
    )
    if job["execution_entrypoint"] == "run_append_adapt":
        _validate_compile_identity(
            _mapping(
                summary.get("compile_identity"),
                label="Append summary compile identity",
            ),
            label=f"{job['execution_id']} summary",
        )
        resources = _mapping(
            _mapping(
                summary.get("resources"),
                label="Append summary resources",
            ).get("terminal_compiled_resources"),
            label="Append terminal compiled resources",
        )
        if (
            resources.get("compiled_circuit_stats_status") != "ok"
            or resources.get("compile_convention") != TABLE_I_COMPILE_ID
        ):
            raise PackageContractError(
                f"G12 Append compilation is unavailable for "
                f"{job['execution_id']}."
            )
    else:
        plateau = _mapping(
            summary.get("effective_plateau"),
            label="RA effective-plateau summary",
        )
        resources = _mapping(
            plateau.get("resources"),
            label="RA reporting-prefix resources",
        )
        if resources.get("compile_convention") != TABLE_I_COMPILE_ID:
            raise PackageContractError(
                f"G12 RA compilation drifted for {job['execution_id']}."
            )
    return {
        "compile_identity": TABLE_I_COMPILE_ID,
        "reporting_prefix_resources": _resource_counts(resources),
    }


def _trajectory_projection(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "run" in result:
        rows = _sequence(
            _ra_run(result).get("accepted_trajectory"),
            label="RA accepted trajectory",
        )
        return [
            {
                "controller_round": _integer(
                    row.get("controller_round"),
                    label="trajectory round",
                    minimum=1,
                ),
                "energy": _finite(row.get("energy"), label="trajectory energy"),
                "state_fingerprint": str(
                    row.get("projective_state_fingerprint", "")
                ),
                "operator_labels": list(
                    _sequence(
                        row.get("operators"),
                        label="trajectory operators",
                    )
                ),
                "logical_parameters": [
                    _finite(value, label="trajectory logical parameter")
                    for value in _sequence(
                        row.get("logical_parameters"),
                        label="trajectory logical parameters",
                    )
                ],
            }
            for row in rows
            if isinstance(row, Mapping)
        ]
    payload = _append_payload(result)
    return [
        {
            "controller_round": _integer(
                row.get("controller_round"),
                label="Append trajectory round",
                minimum=1,
            ),
            "energy": _finite(
                row.get("energy_after"),
                label="Append trajectory energy",
            ),
            "state_fingerprint": None,
            "operator_labels": [str(row.get("selected_label", ""))],
            "logical_parameters": [],
        }
        for row in _sequence(payload.get("history"), label="Append history")
        if isinstance(row, Mapping)
    ]


def _validate_g13(
    *,
    job: Mapping[str, Any],
    protocol: Mapping[str, Any],
    result: Mapping[str, Any],
    g7: Mapping[str, Any],
    g11: Mapping[str, Any],
    replay_diagnostic: Mapping[str, Any],
    t13: Mapping[str, Any],
) -> dict[str, Any]:
    if job["route_id"] != "singleton_plateau":
        return {"scope": "not_a_preservation_cell"}
    problem_sha = protocol["problem"]["problem_request_sha256"]
    if (
        problem_sha == t13["problem_request_sha256"]
        or t13.get("study1_problem_comparison_performed") is not False
    ):
        raise PackageContractError(
            f"G13 attempted a wrong-physics T13 comparison for "
            f"{job['execution_id']}."
        )
    same_cell = _digested_mapping(
        replay_diagnostic,
        label=f"{job['execution_id']} same-cell replay",
        schema=G11_DIAGNOSTIC_RECEIPT_SCHEMA,
    )
    trajectory_sha256 = canonical_sha256(_trajectory_projection(result))
    if (
        same_cell.get("status") != "passed"
        or same_cell.get("selected") is not True
        or same_cell.get("primary_trajectory_sha256")
        != trajectory_sha256
        or same_cell.get("bounded_trajectory_prefix_identity_equal")
        is not True
        or g11.get("bounded_trajectory_prefix_identity_equal") is not True
        or g11.get("bounded_replay_status") != "identity_verified"
    ):
        raise PackageContractError(
            f"G13 same-physics replay failed for {job['execution_id']}."
        )
    if (
        protocol.get("active_gradient_policy")
        == "stationary_source_response_v1"
        and (
            g7["active_gradient_indices_acquired"]
            or g7["active_gradient_charge"]
        )
    ):
        raise PackageContractError(
            f"G13 stationary acquisition failed for {job['execution_id']}."
        )
    return {
        "generic_t13_characterization_sha256": t13["sha256"],
        "generic_t13_problem_request_sha256": T13_PROBLEM_REQUEST_SHA256,
        "study1_problem_request_sha256": problem_sha,
        "cross_physics_trajectory_comparison_performed": False,
        "same_cell_replay_status": "bounded_identity_verified",
        "same_cell_bounded_controller_round": same_cell[
            "bounded_controller_round"
        ],
        "same_cell_trajectory_sha256": trajectory_sha256,
        "same_cell_replay_diagnostic_sha256": same_cell["sha256"],
        "paired_policy_deviation_disposition": "report_neutrally_at_matrix_level",
    }


def validate_cell_objective_gates(
    *,
    job: Mapping[str, Any],
    protocol: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_file_sha256: str,
    checkpoint_size_bytes: int,
    ledger: Mapping[str, Any],
    ledger_file_sha256: str,
    ledger_size_bytes: int,
    result: Mapping[str, Any],
    summary: Mapping[str, Any],
    objective_authority: Mapping[str, Any],
    replay_diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate G1-G13 for one direct cell and return a signed projection."""

    execution_id = str(job["execution_id"])
    logical_key = str(job["logical_key"])
    cell_authority = _mapping(
        objective_authority["cells"].get(logical_key),
        label=f"{execution_id} objective cell authority",
    )
    if (
        protocol.get("sha256") != cell_authority.get("protocol_sha256")
        or _result_protocol(result) != protocol
        or protocol["problem"]["problem_request_sha256"]
        != cell_authority.get("problem_request_sha256")
    ):
        raise PackageContractError(
            f"G1 result/protocol authority drifted for {execution_id}."
        )
    same_cutoff = _mapping(
        cell_authority.get("same_cutoff_reference"),
        label=f"{execution_id} same-cutoff authority",
    )
    g1 = _gate_status(
        "G1",
        {
            "cell_source_lock_sha256": cell_authority[
                "cell_source_lock_sha256"
            ],
            "source_archive_sha256": cell_authority[
                "source_archive_sha256"
            ],
            "source_member_sha256": cell_authority[
                "source_member_sha256"
            ],
            "compact_materialization_receipt_sha256": cell_authority["sha256"],
        },
    )
    g2 = _gate_status(
        "G2",
        {
            "n_ph_work": same_cutoff["n_ph_work"],
            "n_ph_reference": same_cutoff["n_ph_reference"],
            "exact_target_label": same_cutoff["exact_target_label"],
            "ed_receipt_sha256": same_cutoff["ed_receipt_sha256"],
            "same_cutoff_reference": True,
        },
    )
    pool = _pool_projection(
        _mapping(
            protocol.get("executable_pool"),
            label=f"{execution_id} executable pool",
        )
    )
    g3 = _gate_status(
        "G3",
        {
            "count": pool["count"],
            "ordered_labels_sha256": pool["ordered_labels_sha256"],
            "ordered_pool_sha256": pool["ordered_pool_sha256"],
            "per_regime_cross_method_check": "deferred_to_matrix",
            "singleton_construction_receipt_sha256": (
                objective_authority["pools"]["sha256"]
            ),
        },
    )
    g4_evidence = _validate_g4(job=job, protocol=protocol, result=result)
    g5_evidence = _validate_g5(job=job, protocol=protocol, result=result)
    g6_evidence = _validate_g6(job=job, result=result)
    g7_evidence = _validate_g7(job=job, protocol=protocol, result=result)
    g8_evidence = _validate_g8(
        job=job,
        protocol=protocol,
        result=result,
        trusted=objective_authority["trusted"],
    )
    g9_evidence = _validate_g9(
        job=job,
        protocol=protocol,
        result=result,
    )
    g10_evidence = _validate_g10(job=job, result=result, ledger=ledger)
    g11_evidence = _validate_g11(
        job=job,
        protocol=protocol,
        result=result,
        checkpoint=checkpoint,
        checkpoint_file_sha256=checkpoint_file_sha256,
        checkpoint_size_bytes=checkpoint_size_bytes,
        ledger_file_sha256=ledger_file_sha256,
        ledger_size_bytes=ledger_size_bytes,
        replay_diagnostic=replay_diagnostic,
    )
    g12_evidence = _validate_g12(
        job=job,
        protocol=protocol,
        result=result,
        summary=summary,
    )
    g13_evidence = _validate_g13(
        job=job,
        protocol=protocol,
        result=result,
        g7=g7_evidence,
        g11=g11_evidence,
        replay_diagnostic=replay_diagnostic,
        t13=objective_authority["t13"],
    )
    gates = [
        g1,
        g2,
        g3,
        _gate_status("G4", g4_evidence),
        _gate_status("G5", g5_evidence),
        _gate_status("G6", g6_evidence),
        _gate_status("G7", g7_evidence),
        _gate_status("G8", g8_evidence),
        _gate_status("G9", g9_evidence),
        _gate_status("G10", g10_evidence),
        _gate_status("G11", g11_evidence),
        _gate_status("G12", g12_evidence),
        _gate_status("G13", g13_evidence),
    ]
    return digested(
        {
            "schema": CELL_GATE_RECEIPT_SCHEMA,
            "package_id": PACKAGE_ID,
            "execution_id": execution_id,
            "logical_key": logical_key,
            "protocol_sha256": protocol["sha256"],
            "objective_gate_authority_sha256": objective_authority["sha256"],
            "gate_ids": [row["gate_id"] for row in gates],
            "gates": gates,
            "status": "passed",
        }
    )


def _gate_by_id(
    cell_receipt: Mapping[str, Any],
    gate_id: str,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in _sequence(
            cell_receipt.get("gates"),
            label="cell objective gates",
        )
        if isinstance(row, Mapping) and row.get("gate_id") == gate_id
    ]
    if len(matches) != 1 or matches[0].get("status") != "passed":
        raise PackageContractError(
            f"Cell objective receipt lacks passed {gate_id}."
        )
    return matches[0]


def _trajectory_deviation(
    measured: Sequence[Mapping[str, Any]],
    stationary: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    shared = min(len(measured), len(stationary))
    energy_deltas = [
        float(stationary[index]["energy"]) - float(measured[index]["energy"])
        for index in range(shared)
    ]
    return {
        "measured_round_count": len(measured),
        "stationary_round_count": len(stationary),
        "shared_round_count": shared,
        "signed_stationary_minus_measured_energy_deltas": energy_deltas,
        "maximum_absolute_energy_delta": (
            None if not energy_deltas else max(abs(value) for value in energy_deltas)
        ),
        "trajectory_identity_equal": (
            list(measured) == list(stationary)
        ),
        "disposition": "neutral_observation_not_a_pass_condition",
    }


_G13_MATCHED_PROTOCOL_FIELDS = (
    "algorithm_id",
    "candidate_representation",
    "adapter_id",
    "selector_identity",
    "selector_scope",
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
    "stopping_rule",
    "horizon",
    "seeds",
    "estimator_accounting_convention",
    "compile_identity",
    "route_contract",
    "resource_weighting_scope",
)


def _g13_matched_protocol_projection(
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        field: protocol.get(field)
        for field in _G13_MATCHED_PROTOCOL_FIELDS
    }


def validate_objective_gate_matrix(
    *,
    plan: Mapping[str, Any],
    jobs: Mapping[str, Mapping[str, Any]],
    cell_records: Sequence[Mapping[str, Any]],
    shared_receipts: Sequence[Mapping[str, Any]],
    objective_authority: Mapping[str, Any],
    completion_states: Mapping[str, str],
) -> dict[str, Any]:
    """Close cross-cell G3/G11/G13/G14 without ranking either policy."""

    by_execution = {
        str(row["execution_id"]): row for row in cell_records
    }
    verify_exact_key_set(
        by_execution,
        tuple(jobs),
        label="objective-gate direct cells",
    )
    for execution_id, row in by_execution.items():
        receipt = _digested_mapping(
            row.get("objective_gates"),
            label=f"{execution_id} cell objective gates",
            schema=CELL_GATE_RECEIPT_SCHEMA,
        )
        if (
            receipt.get("status") != "passed"
            or receipt.get("objective_gate_authority_sha256")
            != objective_authority["sha256"]
        ):
            raise PackageContractError(
                f"Cell objective gates failed for {execution_id}."
            )
        for gate_id in OBJECTIVE_GATE_IDS[:-1]:
            _gate_by_id(receipt, gate_id)

    g3_groups: list[dict[str, Any]] = []
    for regime_id in VALIDATION_REGIMES:
        macro_rows = [
            row
            for execution_id, row in by_execution.items()
            if (
                jobs[execution_id]["regime_id"] == regime_id
                and jobs[execution_id]["route_id"] != "singleton_plateau"
            )
        ]
        identities = {
            (
                _gate_by_id(row["objective_gates"], "G3")["evidence"]["count"],
                _gate_by_id(row["objective_gates"], "G3")["evidence"][
                    "ordered_labels_sha256"
                ],
                _gate_by_id(row["objective_gates"], "G3")["evidence"][
                    "ordered_pool_sha256"
                ],
            )
            for row in macro_rows
        }
        if len(identities) != 1:
            raise PackageContractError(
                f"G3 per-regime RA/Append full-pool equality failed for "
                f"{regime_id}."
            )
        count, labels_sha, full_sha = identities.pop()
        g3_groups.append(
            {
                "regime_id": regime_id,
                "count": count,
                "ordered_labels_sha256": labels_sha,
                "ordered_pool_sha256": full_sha,
                "direct_cell_count": len(macro_rows),
                "ra_append_equal_within_problem": True,
            }
        )

    method_regime_coverage = {
        (
            "append"
            if job["execution_entrypoint"] == "run_append_adapt"
            else "ra",
            str(job["regime_id"]),
        ): False
        for job in jobs.values()
    }
    for execution_id, row in by_execution.items():
        evidence = _gate_by_id(row["objective_gates"], "G11")["evidence"]
        key = (
            "append"
            if jobs[execution_id]["execution_entrypoint"]
            == "run_append_adapt"
            else "ra",
            str(jobs[execution_id]["regime_id"]),
        )
        if evidence["selected_for_method_regime_replay"]:
            method = evidence["method_family"]
            method_specific_resume_passed = (
                evidence["ra_resume_round_trip_status"]
                == "authenticated_continuation_identity_verified"
                and evidence["append_resume_boundary_status"]
                == "not_applicable"
                if method == "ra_adapt"
                else evidence["ra_resume_round_trip_status"]
                == "not_applicable"
                and evidence["append_resume_boundary_status"]
                == "authenticated_reconstruction_only_verified"
                if method == "append_adapt"
                else False
            )
            if (
                evidence["bounded_replay_status"] != "identity_verified"
                or evidence.get(
                    "bounded_trajectory_prefix_identity_equal"
                )
                is not True
                or not method_specific_resume_passed
            ):
                raise PackageContractError(
                    f"G11 selected coverage failed for {execution_id}."
                )
            method_regime_coverage[key] = True
    if not all(method_regime_coverage.values()):
        missing = sorted(
            f"{method}:{regime}"
            for (method, regime), covered in method_regime_coverage.items()
            if not covered
        )
        raise PackageContractError(
            f"G11 method/regime replay coverage is incomplete: {missing}."
        )

    deviations: list[dict[str, Any]] = []
    for regime_id in VALIDATION_REGIMES:
        measured_id = (
            "ra_repair_measured_late_v1__"
            f"{validation_cell_id(regime_id, 'singleton_plateau')}"
        )
        stationary_id = (
            f"{STATIONARY_BUNDLE_ID}__"
            f"{validation_cell_id(regime_id, 'singleton_plateau')}"
        )
        measured = by_execution[measured_id]
        stationary = by_execution[stationary_id]
        measured_result = _mapping(
            measured.get("result_payload"),
            label=f"{measured_id} result payload",
        )
        stationary_result = _mapping(
            stationary.get("result_payload"),
            label=f"{stationary_id} result payload",
        )
        measured_protocol = _result_protocol(measured_result)
        stationary_protocol = _result_protocol(stationary_result)
        matched_projection = _g13_matched_protocol_projection(
            measured_protocol
        )
        if (
            matched_projection
            != _g13_matched_protocol_projection(stationary_protocol)
            or measured_protocol.get("active_gradient_policy")
            != "measured_residual_response_v1"
            or stationary_protocol.get("active_gradient_policy")
            != "stationary_source_response_v1"
        ):
            raise PackageContractError(
                f"G13 measured/stationary protocol pairing drifted for "
                f"{regime_id}."
            )
        deviations.append(
            {
                "regime_id": regime_id,
                "measured_execution_id": measured_id,
                "stationary_execution_id": stationary_id,
                "matched_scientific_protocol_projection_sha256": (
                    canonical_sha256(matched_projection)
                ),
                "only_active_gradient_policy_differs": True,
                **_trajectory_deviation(
                    _trajectory_projection(measured_result),
                    _trajectory_projection(stationary_result),
                ),
            }
        )

    expected_states = set(logical_cell_keys())
    verify_exact_key_set(
        completion_states,
        tuple(expected_states),
        label="G14 logical completion states",
    )
    invalid_states = {
        key: state
        for key, state in completion_states.items()
        if state not in COMPLETION_STATES
    }
    if invalid_states:
        raise PackageContractError(
            f"G14 contains invalid completion states: {invalid_states}."
        )
    if any(state != "done" for state in completion_states.values()):
        raise PackageContractError(
            "G14 cannot pass until every exact Study-1 logical cell is done."
        )
    if len(shared_receipts) != 2:
        raise PackageContractError(
            "G14 requires exactly two authenticated shared Append references."
        )

    gates = [
        _gate_status(
            "G3",
            {
                "regime_groups": g3_groups,
                "singleton_construction_receipt_sha256": (
                    objective_authority["pools"]["sha256"]
                ),
            },
        ),
        _gate_status(
            "G11",
            {
                "method_regime_coverage": [
                    {
                        "method": method,
                        "regime_id": regime,
                        "covered": covered,
                    }
                    for (method, regime), covered in sorted(
                        method_regime_coverage.items()
                    )
                ]
            },
        ),
        _gate_status(
            "G13",
            {
                "generic_t13_characterization_sha256": (
                    objective_authority["t13"]["sha256"]
                ),
                "cross_physics_trajectory_comparison_performed": False,
                "paired_policy_deviations": deviations,
            },
        ),
        _gate_status(
            "G14",
            {
                "logical_cell_count": 20,
                "direct_execution_count": 18,
                "shared_reference_count": 2,
                "state_vocabulary": list(COMPLETION_STATES),
                "states": dict(sorted(completion_states.items())),
            },
        ),
    ]
    return digested(
        {
            "schema": MATRIX_GATE_RECEIPT_SCHEMA,
            "package_id": PACKAGE_ID,
            "execution_plan_sha256": plan["sha256"],
            "objective_gate_authority_sha256": objective_authority["sha256"],
            "gate_ids": [row["gate_id"] for row in gates],
            "gates": gates,
            "scientific_outcome_used_as_pass_condition": False,
            "status": "passed",
        }
    )


__all__ = [
    "CELL_GATE_RECEIPT_SCHEMA",
    "COMPLETION_STATES",
    "G11_EVIDENCE_SCHEMA",
    "G5_EVIDENCE_SCHEMA",
    "G8_EVIDENCE_SCHEMA",
    "G9_EVIDENCE_SCHEMA",
    "MATRIX_GATE_RECEIPT_SCHEMA",
    "OBJECTIVE_GATE_AUTHORITY_SCHEMA",
    "OBJECTIVE_GATE_IDS",
    "POOL_CONSTRUCTION_RECEIPT_SCHEMA",
    "SOURCE_LOCK_CELL_RECEIPT_SCHEMA",
    "T13_CHARACTERIZATION_RECEIPT_SCHEMA",
    "TRUSTED_EXECUTION_RECEIPT_SCHEMA",
    "validate_cell_objective_gates",
    "validate_objective_gate_authority",
    "validate_objective_gate_matrix",
]
