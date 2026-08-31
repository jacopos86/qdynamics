#!/usr/bin/env python3
"""Build and validate immutable planned-slot report contracts.

The parallel benchmark matrix owns the tuple identities.  This module expands
that matrix into one report contract per future lane and keeps result state
mutable only *inside* the predeclared exact-reference and method slots.  An
ingestion adapter can therefore replace a pending slot with verified evidence
without rediscovering cells or changing the report's tuple structure.

The contracts deliberately contain logical/graph resource placeholders only.
Qiskit compilation is outside this campaign and is rejected by validation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics.acquisition_supervisor import CampaignRegistry
from pipelines.time_dynamics.parallel_benchmark_matrix import (
    DEFAULT_MATRIX_PATH,
    ParallelBenchmarkMatrix,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    ACTION_REGISTRY_ENTRY_V1,
    KIND_UPDATE_REPORT_SLOT,
    ExactReferenceReceipt,
    ProspectiveExecutionError,
    SafeguardQualificationReceipt,
    ScoreQualificationReceipt,
    SeedQualificationReceipt,
    ScientificPreSubmitAudit,
    SourceEpochReceipt,
    VerifiedFile,
    binding_registry_payload,
    load_action_registry_entry,
)
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    ExactReferenceIdentity,
    SeedIdentity,
    SourceEpochIdentity,
    build_identity_registry,
)
from pipelines.time_dynamics.prospective_scientific_planner import (
    ARM_APM,
    ARM_AVQDS,
    HORIZON_FULL,
    ProspectivePlannerError,
    ProspectiveScientificBindings,
)
from pipelines.time_dynamics.staged_campaign import AccuracyPolicy


PLANNED_REPORT_CONTRACT_V1 = "paper_ii_planned_report_contract_v1"
REPORT_CONTRACT_SELF_DIGEST_KEY = "report_contract_sha256"
TUPLE_STRUCTURE_DIGEST_KEY = "tuple_structure_sha256"
REPORT_SLOT_EVIDENCE_V1 = "paper_ii_report_slot_evidence_v1"
EXACT_SLOT_EVIDENCE_V1 = "paper_ii_exact_slot_evidence_v1"

METHODS = ("apm", "avqds")
REPORT_METHOD_TO_ARM = {"apm": ARM_APM, "avqds": ARM_AVQDS}
METHOD_STATES = frozenset({"pending", "candidate", "qualified", "unresolved"})
EXACT_REFERENCE_STATES = frozenset({"pending", "verified"})

DEFAULT_REPORT_PATHS: Mapping[str, Path] = {
    "paper_ii_hh_product_v1": Path(
        "output/pdf/paper_ii_hh_product_v1/report_contract_current.json"
    ),
    # The live SIAM stable report contract already binds acquired evidence.
    # Keep the full planned tuple skeleton beside it instead of overwriting it.
    "siam_nb4_pilot_v1": Path(
        "output/siam_screening/nb4_pilot_v1/acquisition_supervisor/"
        "planned_matrix_report_contract_v1.json"
    ),
    "paper_ii_hubbard_l4_scale_v1": Path(
        "output/pdf/paper_ii_hubbard_l4_scale_v1/report_contract_current.json"
    ),
}


class PlannedReportContractError(ValueError):
    """A planned report contract does not match its frozen matrix."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _numeric_equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-14)


def _json_normalized(value: Any) -> Any:
    """Return the exact JSON data model used by persisted contracts."""

    return json.loads(_canonical_bytes(value).decode("utf-8"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PlannedReportContractError(f"{label} must be a JSON object")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise PlannedReportContractError(f"{label} must be a JSON array")
    return value


def _relative_path(repository_root: Path, path: Path) -> str:
    root = repository_root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise PlannedReportContractError(
            f"matrix path is outside repository root: {path}"
        ) from exc


def _matrix_binding(
    repository_root: Path,
    matrix_path: Path,
    matrix: ParallelBenchmarkMatrix,
) -> dict[str, Any]:
    return {
        "path": _relative_path(repository_root, matrix_path),
        "raw_sha256": _file_sha256(matrix_path),
        "payload_sha256": matrix.payload_sha256,
        "expanded_semantic_sha256": matrix.expanded_semantic_sha256,
        "matrix_id": matrix.matrix_id,
    }


def _slot_id(cell_id: str, method: str) -> str:
    return f"{cell_id}::{method}"


def _pending_method_slot(cell_id: str, method: str) -> dict[str, Any]:
    return {
        "slot_id": _slot_id(cell_id, method),
        "method": method,
        "state": "pending",
        "values": None,
        "provenance": None,
    }


def _pending_exact_reference(
    cell_id: str,
    *,
    reference_id: str,
    reference_identity_sha256: str,
) -> dict[str, Any]:
    return {
        "slot_id": f"{cell_id}::exact",
        "reference_id": reference_id,
        "reference_identity_sha256": reference_identity_sha256,
        "state": "pending",
        "artifact": None,
    }


def _structure_projection(contract: Mapping[str, Any]) -> dict[str, Any]:
    cells = []
    for raw_cell in _sequence(contract.get("cells"), label="cells"):
        cell = _mapping(raw_cell, label="cell")
        methods = []
        for raw_slot in _sequence(cell.get("method_slots"), label="method_slots"):
            slot = _mapping(raw_slot, label="method slot")
            methods.append(
                {
                    "slot_id": slot.get("slot_id"),
                    "method": slot.get("method"),
                }
            )
        exact = _mapping(cell.get("exact_reference"), label="exact_reference")
        cells.append(
            {
                "lane_id": cell.get("lane_id"),
                "cell_id": cell.get("cell_id"),
                "model": cell.get("model"),
                "drive": cell.get("drive"),
                "controller": cell.get("controller"),
                "horizon": cell.get("horizon"),
                "grid": cell.get("grid"),
                "physics": _json_normalized(cell.get("physics")),
                "exact_slot_id": exact.get("slot_id"),
                "exact_reference_id": exact.get("reference_id"),
                "exact_reference_identity_sha256": exact.get(
                    "reference_identity_sha256"
                ),
                "method_slots": methods,
            }
        )
    return {
        "lane_id": contract.get("lane_id"),
        "controller": contract.get("controller"),
        "methods": contract.get("methods"),
        "cells": cells,
    }


def _seal(contract: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(contract))
    result.pop(REPORT_CONTRACT_SELF_DIGEST_KEY, None)
    result[TUPLE_STRUCTURE_DIGEST_KEY] = _canonical_sha256(
        _structure_projection(result)
    )
    result[REPORT_CONTRACT_SELF_DIGEST_KEY] = _canonical_sha256(result)
    return result


def build_planned_report_contract(
    *,
    lane_id: str,
    repository_root: str | Path = ".",
    matrix_path: str | Path = DEFAULT_MATRIX_PATH,
) -> dict[str, Any]:
    """Return the all-pending report contract for one frozen matrix lane."""

    root = Path(repository_root).resolve()
    source = Path(matrix_path)
    if not source.is_absolute():
        source = root / source
    source = source.resolve(strict=True)
    matrix = ParallelBenchmarkMatrix.read(source)
    identities = build_identity_registry(matrix)
    references = {row.cell_id: row for row in identities.references}
    cells = matrix.cells_for(lane_id)
    if not cells:
        raise PlannedReportContractError(f"matrix has no lane {lane_id!r}")

    common = _mapping(matrix.payload.get("common"), label="matrix common")
    grid = _mapping(common.get("grid"), label="matrix grid")
    methods = tuple(str(value) for value in common.get("methods") or ())
    if methods != METHODS:
        raise PlannedReportContractError("matrix method order is not apm,avqds")
    if common.get("qiskit_compilation") is not False:
        raise PlannedReportContractError("planned campaign must disable Qiskit")

    planned_cells = []
    for cell in cells:
        reference = references[cell.cell_id]
        planned_cells.append(
            {
                "lane_id": lane_id,
                "cell_id": cell.cell_id,
                "model": cell.model,
                "drive": cell.drive,
                "controller": cell.controller,
                "horizon": cell.horizon,
                "grid": {
                    "start": float(grid["start"]),
                    "stop": cell.horizon,
                    "points": int(grid["points"]),
                },
                "physics": dict(cell.physics),
                "exact_reference": _pending_exact_reference(
                    cell.cell_id,
                    reference_id=reference.reference_id,
                    reference_identity_sha256=reference.identity_sha256,
                ),
                "method_slots": [
                    _pending_method_slot(cell.cell_id, method)
                    for method in methods
                ],
            }
        )

    contract = {
        "schema": PLANNED_REPORT_CONTRACT_V1,
        "contract_id": f"{matrix.matrix_id}:{lane_id}",
        "lane_id": lane_id,
        "matrix": _matrix_binding(root, source, matrix),
        "controller": str(common["controller"]),
        "methods": list(methods),
        "cost_authority": str(common["cost_authority"]),
        "qiskit_compilation": {
            "requested": False,
            "used": False,
            "canonical": False,
        },
        "slot_state_contract": {
            "method_states": [
                "pending",
                "candidate",
                "qualified",
                "unresolved",
            ],
            "pending_means": "no_valid_completed_trajectory",
            "completed_values_are_never_hidden_by_a_failed_gate": True,
        },
        "counts": {
            "planned_cells": len(planned_cells),
            "planned_exact_references": len(planned_cells),
            "planned_method_slots": len(planned_cells) * len(methods),
            "pending_exact_references": len(planned_cells),
            "pending_method_slots": len(planned_cells) * len(methods),
            "candidate_method_slots": 0,
            "qualified_method_slots": 0,
            "unresolved_method_slots": 0,
        },
        "cells": planned_cells,
    }
    sealed = _seal(contract)
    validate_planned_report_contract(
        sealed,
        repository_root=root,
        matrix_path=source,
    )
    return sealed


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_file_binding(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise PlannedReportContractError(f"{label} is not a typed file binding")
    path = value.get("path")
    if not isinstance(path, str) or not path.strip() or not _is_sha256(
        value.get("sha256")
    ):
        raise PlannedReportContractError(f"{label} is not a valid file binding")


def _validate_slot_payload(slot: Mapping[str, Any]) -> None:
    state = str(slot.get("state") or "")
    if state not in METHOD_STATES:
        raise PlannedReportContractError(f"unsupported method-slot state {state!r}")
    values = slot.get("values")
    provenance = slot.get("provenance")
    if state == "pending":
        if values is not None or provenance is not None:
            raise PlannedReportContractError("pending method slot contains evidence")
        return
    if state in {"candidate", "qualified"} and not isinstance(values, Mapping):
        raise PlannedReportContractError(
            f"{state} method slot must contain numerical values"
        )
    if not isinstance(provenance, Mapping):
        raise PlannedReportContractError(
            f"{state} method slot must contain provenance"
        )
    if provenance.get("schema") != REPORT_SLOT_EVIDENCE_V1:
        raise PlannedReportContractError(
            f"{state} method slot has no typed evidence binding"
        )
    required = {
        "action_registry",
        "score_receipt",
        "run_receipt",
        "run",
        "metrics",
        "planner_receipt",
        "planner_sequence",
        "selected_action_id",
        "exact_reference_id",
        "exact_reference_identity_sha256",
        "exact_reference_receipt",
        "scientific_pre_submit_audit",
        "audit_warnings",
        "paired_score_receipt",
        "paired_action_registry",
        "paired_run_receipt",
        "accuracy_policy_sha256",
        "matrix_accuracy_policy_sha256",
    }
    if not required.issubset(provenance):
        raise PlannedReportContractError(
            f"{state} method slot has incomplete evidence provenance"
        )
    planner_sequence = provenance.get("planner_sequence")
    if (
        not isinstance(planner_sequence, int)
        or isinstance(planner_sequence, bool)
        or planner_sequence < 0
    ):
        raise PlannedReportContractError(
            f"{state} method slot has an invalid planner sequence"
        )
    for field in (
        "action_registry",
        "score_receipt",
        "run_receipt",
        "run",
        "metrics",
        "planner_receipt",
        "exact_reference_receipt",
        "scientific_pre_submit_audit",
    ):
        _validate_file_binding(
            provenance.get(field),
            label=f"{state} method-slot {field}",
        )
    for field in ("accuracy_policy_sha256", "matrix_accuracy_policy_sha256"):
        if not _is_sha256(provenance.get(field)):
            raise PlannedReportContractError(
                f"{state} method slot has an invalid {field}"
            )
    warnings = provenance.get("audit_warnings")
    if (
        not isinstance(warnings, list)
        or any(not isinstance(row, str) or not row.strip() for row in warnings)
        or len(set(warnings)) != len(warnings)
    ):
        raise PlannedReportContractError(
            f"{state} method slot has invalid audit warnings"
        )
    paired_fields = (
        "paired_score_receipt",
        "paired_action_registry",
        "paired_run_receipt",
    )
    if state == "qualified":
        for field in paired_fields:
            _validate_file_binding(
                provenance.get(field),
                label=f"qualified method-slot {field}",
            )
    elif any(provenance.get(field) is not None for field in paired_fields):
        raise PlannedReportContractError(
            f"{state} method slot contains unqualified peer evidence"
        )
    safeguard = provenance.get("safeguard_receipt")
    if state == "unresolved" and not isinstance(safeguard, Mapping):
        raise PlannedReportContractError(
            "unresolved method slot has no terminal safeguard receipt"
        )
    if state != "unresolved" and safeguard is not None:
        raise PlannedReportContractError(
            "nonterminal method slot contains a safeguard receipt"
        )
    history = provenance.get("superseded_candidates", [])
    if not isinstance(history, list) or any(
        not isinstance(row, Mapping) for row in history
    ):
        raise PlannedReportContractError(
            f"{state} method slot has malformed superseded history"
        )
    previous_sequence = -1
    for row in history:
        if "superseded_candidates" in row:
            raise PlannedReportContractError(
                "superseded candidate history must not be nested"
            )
        _validate_slot_payload(
            {
                "state": "candidate",
                "values": {},
                "provenance": row,
            }
        )
        history_sequence = row.get("planner_sequence")
        assert isinstance(history_sequence, int)
        if history_sequence <= previous_sequence:
            raise PlannedReportContractError(
                "superseded candidate planner sequence is not strictly increasing"
            )
        previous_sequence = history_sequence
    if history and previous_sequence >= planner_sequence:
        raise PlannedReportContractError(
            "superseded candidate planner sequence does not precede the current one"
        )


def _validate_exact_payload(exact: Mapping[str, Any]) -> None:
    if not str(exact.get("slot_id") or "").endswith("::exact"):
        raise PlannedReportContractError("exact-reference slot identity is invalid")
    reference_id = str(exact.get("reference_id") or "")
    if "::reference::" not in reference_id:
        raise PlannedReportContractError(
            "exact-reference evidence does not use the canonical reference ID"
        )
    identity = str(exact.get("reference_identity_sha256") or "")
    if len(identity) != 64 or any(ch not in "0123456789abcdef" for ch in identity):
        raise PlannedReportContractError(
            "exact-reference identity digest is invalid"
        )
    state = str(exact.get("state") or "")
    if state not in EXACT_REFERENCE_STATES:
        raise PlannedReportContractError(
            f"unsupported exact-reference state {state!r}"
        )
    artifact = exact.get("artifact")
    if state == "pending" and artifact is not None:
        raise PlannedReportContractError("pending exact reference contains an artifact")
    if state == "verified" and not isinstance(artifact, Mapping):
        raise PlannedReportContractError(
            "verified exact reference must contain an artifact binding"
        )
    if state == "verified":
        assert isinstance(artifact, Mapping)
        if artifact.get("schema") != EXACT_SLOT_EVIDENCE_V1:
            raise PlannedReportContractError(
                "verified exact reference has no typed evidence binding"
            )
        if not {
            "reference_id",
            "reference_identity_sha256",
            "manifest",
            "data",
            "reference_receipt",
            "seed_receipt",
            "source_epoch_receipt",
            "scientific_pre_submit_audit",
            "audit_warnings",
        }.issubset(artifact):
            raise PlannedReportContractError(
                "verified exact reference has incomplete evidence provenance"
            )
        warnings = artifact.get("audit_warnings")
        if (
            not isinstance(warnings, list)
            or any(
                not isinstance(row, str) or not row.strip()
                for row in warnings
            )
            or len(set(warnings)) != len(warnings)
        ):
            raise PlannedReportContractError(
                "verified exact reference has invalid audit warnings"
            )
        if (
            artifact.get("reference_id") != exact.get("reference_id")
            or artifact.get("reference_identity_sha256")
            != exact.get("reference_identity_sha256")
        ):
            raise PlannedReportContractError(
                "verified exact-reference artifact crosses slot identity"
            )


def _state_counts(contract: Mapping[str, Any]) -> dict[str, int]:
    method = {state: 0 for state in METHOD_STATES}
    exact = {state: 0 for state in EXACT_REFERENCE_STATES}
    for raw_cell in _sequence(contract.get("cells"), label="cells"):
        cell = _mapping(raw_cell, label="cell")
        exact_state = str(
            _mapping(cell.get("exact_reference"), label="exact_reference").get(
                "state"
            )
        )
        exact[exact_state] = exact.get(exact_state, 0) + 1
        for raw_slot in _sequence(cell.get("method_slots"), label="method_slots"):
            state = str(_mapping(raw_slot, label="method slot").get("state"))
            method[state] = method.get(state, 0) + 1
    return {
        "planned_cells": len(_sequence(contract.get("cells"), label="cells")),
        "planned_exact_references": sum(exact.values()),
        "planned_method_slots": sum(method.values()),
        "pending_exact_references": exact.get("pending", 0),
        "pending_method_slots": method.get("pending", 0),
        "candidate_method_slots": method.get("candidate", 0),
        "qualified_method_slots": method.get("qualified", 0),
        "unresolved_method_slots": method.get("unresolved", 0),
    }


def validate_planned_report_contract(
    contract: Mapping[str, Any],
    *,
    repository_root: str | Path = ".",
    matrix_path: str | Path = DEFAULT_MATRIX_PATH,
) -> None:
    """Validate matrix binding, immutable tuple structure, and mutable slots."""

    root = Path(repository_root).resolve()
    source = Path(matrix_path)
    if not source.is_absolute():
        source = root / source
    source = source.resolve(strict=True)
    row = _mapping(contract, label="report contract")
    if row.get("schema") != PLANNED_REPORT_CONTRACT_V1:
        raise PlannedReportContractError("unsupported report-contract schema")
    lane_id = str(row.get("lane_id") or "")
    if lane_id not in DEFAULT_REPORT_PATHS:
        raise PlannedReportContractError(f"unsupported report lane {lane_id!r}")

    matrix = ParallelBenchmarkMatrix.read(source)
    identities = build_identity_registry(matrix)
    references = {row.cell_id: row for row in identities.references}
    common = _mapping(matrix.payload.get("common"), label="matrix common")
    cells = matrix.cells_for(lane_id)
    if not cells:
        raise PlannedReportContractError(f"matrix has no lane {lane_id!r}")
    grid = _mapping(common.get("grid"), label="matrix grid")
    expected_cells = []
    for cell in cells:
        reference = references[cell.cell_id]
        expected_cells.append(
            {
                "lane_id": lane_id,
                "cell_id": cell.cell_id,
                "model": cell.model,
                "drive": cell.drive,
                "controller": cell.controller,
                "horizon": cell.horizon,
                "grid": {
                    "start": float(grid["start"]),
                    "stop": cell.horizon,
                    "points": int(grid["points"]),
                },
                "physics": dict(cell.physics),
                "exact_reference": _pending_exact_reference(
                    cell.cell_id,
                    reference_id=reference.reference_id,
                    reference_identity_sha256=reference.identity_sha256,
                ),
                "method_slots": [
                    _pending_method_slot(cell.cell_id, method)
                    for method in METHODS
                ],
            }
        )
    expected = _seal(
        {
            "lane_id": lane_id,
            "controller": str(common["controller"]),
            "methods": list(METHODS),
            "cells": expected_cells,
        }
    )

    matrix = ParallelBenchmarkMatrix.read(source)
    expected_binding = _matrix_binding(root, source, matrix)
    if row.get("matrix") != expected_binding:
        raise PlannedReportContractError("report contract matrix binding is stale")
    qiskit = _mapping(row.get("qiskit_compilation"), label="qiskit_compilation")
    if qiskit != {"requested": False, "used": False, "canonical": False}:
        raise PlannedReportContractError("Qiskit compilation is not canonical")
    if row.get("controller") != "state_motion_5e-5":
        raise PlannedReportContractError("report contract changed its controller")
    if tuple(row.get("methods") or ()) != METHODS:
        raise PlannedReportContractError("report contract changed its methods")

    projection = _structure_projection(row)
    structure_digest = _canonical_sha256(projection)
    if row.get(TUPLE_STRUCTURE_DIGEST_KEY) != structure_digest:
        raise PlannedReportContractError("tuple structure digest is invalid")
    expected_projection = _structure_projection(expected)
    if projection != expected_projection:
        raise PlannedReportContractError("tuple structure differs from the matrix")

    for raw_cell in _sequence(row.get("cells"), label="cells"):
        cell = _mapping(raw_cell, label="cell")
        _validate_exact_payload(
            _mapping(cell.get("exact_reference"), label="exact_reference")
        )
        for raw_slot in _sequence(cell.get("method_slots"), label="method_slots"):
            _validate_slot_payload(_mapping(raw_slot, label="method slot"))

    counts = _state_counts(row)
    if row.get("counts") != counts:
        raise PlannedReportContractError("report-contract state counts are stale")
    declared_self_digest = row.get(REPORT_CONTRACT_SELF_DIGEST_KEY)
    unsigned = deepcopy(dict(row))
    unsigned.pop(REPORT_CONTRACT_SELF_DIGEST_KEY, None)
    if declared_self_digest != _canonical_sha256(unsigned):
        raise PlannedReportContractError("report-contract self digest is invalid")


def _registered_rows(
    matrix: ParallelBenchmarkMatrix, cell_id: str
) -> tuple[SeedIdentity, ExactReferenceIdentity]:
    identities = build_identity_registry(matrix)
    references = [row for row in identities.references if row.cell_id == cell_id]
    if len(references) != 1:
        raise PlannedReportContractError(
            f"matrix does not register exactly one reference for {cell_id}"
        )
    reference = references[0]
    seeds = [row for row in identities.seeds if row.seed_id == reference.seed_id]
    if len(seeds) != 1:
        raise PlannedReportContractError(
            f"matrix does not register exactly one seed for {cell_id}"
        )
    return seeds[0], reference


def _action_context(
    *,
    root: Path,
    action_registry: VerifiedFile,
    matrix: ParallelBenchmarkMatrix,
    cell_id: str,
) -> tuple[
    ProspectiveScientificBindings,
    SeedIdentity,
    ExactReferenceIdentity,
    SourceEpochIdentity,
]:
    try:
        action_registry.verify(root)
        raw = _mapping(
            json.loads((root / action_registry.path).read_text(encoding="utf-8")),
            label="action registry entry",
        )
        request = _mapping(raw.get("execution_request"), label="execution request")
        inputs = _mapping(request.get("inputs"), label="execution inputs")
        bindings = ProspectiveScientificBindings.from_json_dict(inputs)
        receipt_rows = _sequence(raw.get("source_receipts"), label="source receipts")
        source_paths = []
        for raw_receipt in receipt_rows:
            binding = _mapping(raw_receipt, label="source receipt binding")
            file = VerifiedFile(
                str(binding.get("path") or ""), str(binding.get("sha256") or "")
            )
            file.verify(root)
            payload = _mapping(
                json.loads((root / file.path).read_text(encoding="utf-8")),
                label="source receipt",
            )
            if payload.get("schema") == "paper_ii_source_epoch_receipt_v1":
                source_paths.append(file.path)
        if len(source_paths) != 1:
            raise PlannedReportContractError(
                "action registry has no unique source-epoch receipt"
            )
        source = SourceEpochReceipt.load(
            root / source_paths[0], repository_root=root
        )
    except (
        OSError,
        json.JSONDecodeError,
        ProspectiveExecutionError,
        ProspectivePlannerError,
    ) as exc:
        raise PlannedReportContractError(
            "action registry evidence is not digest-verifiable"
        ) from exc
    seed, reference = _registered_rows(matrix, cell_id)
    try:
        load_action_registry_entry(
            action_registry,
            repository_root=root,
            bindings=bindings,
            expected_seed=seed,
            expected_reference=reference,
            expected_source_epoch=source.source_epoch,
        )
    except ProspectiveExecutionError as exc:
        raise PlannedReportContractError(
            "action registry does not join the registered cell evidence"
        ) from exc
    return bindings, seed, reference, source.source_epoch


def _validate_score_action_registry(
    score: ScoreQualificationReceipt,
    action_registry: VerifiedFile,
    *,
    root: Path,
    bindings: ProspectiveScientificBindings,
    seed_identity: SeedIdentity,
    reference_identity: ExactReferenceIdentity,
    source_epoch: SourceEpochIdentity,
    label: str,
) -> None:
    try:
        action = load_action_registry_entry(
            action_registry,
            repository_root=root,
            bindings=bindings,
            expected_seed=seed_identity,
            expected_reference=reference_identity,
            expected_source_epoch=source_epoch,
        )
    except ProspectiveExecutionError as exc:
        raise PlannedReportContractError(
            f"{label} action registry does not bind the current matrix cell"
        ) from exc
    if (
        action.request.action_id != score.scientific_action_id
        or action.request.execution_id != score.execution_id
        or action.request.attempt != score.attempt
        or action.request.planner_sequence != score.planner_sequence
        or action.request.requested_memory_gib != score.requested_memory_gib
    ):
        raise PlannedReportContractError(
            f"{label} score does not bind its registered execution request"
        )


def _verify_report_audit(
    audit: ScientificPreSubmitAudit,
    *,
    root: Path,
    campaign_registry: CampaignRegistry,
    contract: Mapping[str, Any] | None,
    matrix: ParallelBenchmarkMatrix,
    lane_id: str,
    source_epoch: SourceEpochIdentity,
    seed_receipt_sha256: str,
    exact_reference_receipt_sha256: str,
    primary_score: ScoreQualificationReceipt | None = None,
    peer_score: ScoreQualificationReceipt | None = None,
    safeguard: SafeguardQualificationReceipt | None = None,
    source_epoch_receipt: SourceEpochReceipt | None = None,
    seed_receipt: SeedQualificationReceipt | None = None,
    exact_reference_receipt: ExactReferenceReceipt | None = None,
    require_current_registry: bool = True,
) -> ScientificPreSubmitAudit:
    try:
        current = audit.reload(root)
        if require_current_registry:
            current.verify_current_registry(
                campaign_registry,
                repository_root=root,
            )
        if current.audit_scope != KIND_UPDATE_REPORT_SLOT:
            raise ProspectiveExecutionError(
                "scientific pre-submit audit is not report-slot scoped"
            )
        if contract is not None:
            identities = build_identity_registry(matrix)
            current.verify_contracts(
                lane_id=lane_id,
                matrix_raw_sha256=matrix.raw_sha256,
                matrix_payload_sha256=matrix.payload_sha256,
                matrix_expanded_semantic_sha256=matrix.expanded_semantic_sha256,
                identity_registry_sha256=identities.digest,
                executor_registry_sha256=binding_registry_payload()["bindings_sha256"],
                planned_report_contract_sha256=str(
                    contract[REPORT_CONTRACT_SELF_DIGEST_KEY]
                ),
            )
        current.verify_evidence(
            source_epoch_identity_sha256=source_epoch.identity_sha256,
            seed_receipt_sha256=seed_receipt_sha256,
            exact_reference_receipt_sha256=exact_reference_receipt_sha256,
        )
        for label, audited_score in (
            ("primary", primary_score),
            ("peer", peer_score),
        ):
            if audited_score is None:
                continue
            score_file = VerifiedFile(
                audited_score.receipt_path,
                audited_score.receipt_sha256,
            )
            if (
                score_file not in current.score_receipts
                or audited_score.action_registry not in current.action_registries
                or audited_score.run_receipt not in current.run_lifecycle_tips
            ):
                raise ProspectiveExecutionError(
                    f"scientific pre-submit audit omits {label} report evidence"
                )
            required_files = {
                score_file,
                audited_score.action_registry,
                audited_score.run_receipt,
                audited_score.run,
                audited_score.exact_reference_receipt,
                audited_score.metrics,
                audited_score.planner_receipt,
            }
            if not required_files.issubset(set(current.evidence_files)):
                raise ProspectiveExecutionError(
                    "scientific pre-submit audit inventory omits "
                    f"{label} report evidence"
                )
        if safeguard is not None:
            safeguard_file = VerifiedFile(
                safeguard.receipt_path,
                safeguard.receipt_sha256,
            )
            if (
                safeguard_file not in current.safeguard_receipts
                or safeguard.action_registry not in current.action_registries
                or not set(safeguard.subject_run_receipts).issubset(
                    set(current.run_lifecycle_tips)
                )
                or (
                    safeguard.subject_score_receipt is not None
                    and safeguard.subject_score_receipt
                    not in current.score_receipts
                )
            ):
                raise ProspectiveExecutionError(
                    "scientific pre-submit audit omits safeguard report evidence"
                )
            safeguard_files = {
                safeguard_file,
                safeguard.action_registry,
                *safeguard.subject_run_receipts,
                safeguard.planner_receipt,
            }
            if safeguard.subject_score_receipt is not None:
                safeguard_files.add(safeguard.subject_score_receipt)
            if not safeguard_files.issubset(set(current.evidence_files)):
                raise ProspectiveExecutionError(
                    "scientific pre-submit audit inventory omits safeguard report evidence"
                )
        bootstrap = (
            source_epoch_receipt,
            seed_receipt,
            exact_reference_receipt,
        )
        if any(row is not None for row in bootstrap):
            if any(row is None for row in bootstrap):
                raise ProspectiveExecutionError(
                    "exact-reference report evidence is incomplete"
                )
            assert source_epoch_receipt is not None
            assert seed_receipt is not None
            assert exact_reference_receipt is not None
            source_file = VerifiedFile(
                source_epoch_receipt.receipt_path,
                source_epoch_receipt.receipt_sha256,
            )
            seed_file = VerifiedFile(
                seed_receipt.receipt_path,
                seed_receipt.receipt_sha256,
            )
            reference_file = VerifiedFile(
                exact_reference_receipt.receipt_path,
                exact_reference_receipt.receipt_sha256,
            )
            if (
                source_file not in current.source_epoch_receipts
                or seed_file not in current.seed_receipts
                or reference_file not in current.exact_reference_receipts
            ):
                raise ProspectiveExecutionError(
                    "scientific pre-submit audit omits exact-reference report evidence"
                )
            required_bootstrap_files = {
                source_file,
                source_epoch_receipt.source_tree_manifest,
                seed_file,
                seed_receipt.artifact,
                reference_file,
                exact_reference_receipt.manifest,
                exact_reference_receipt.data,
            }
            if not required_bootstrap_files.issubset(
                set(current.evidence_files)
            ):
                raise ProspectiveExecutionError(
                    "scientific pre-submit audit inventory omits exact-reference report evidence"
                )
    except ProspectiveExecutionError as exc:
        raise PlannedReportContractError(
            "scientific pre-submit audit does not bind this promotion: "
            f"{exc}"
        ) from exc
    return current


def _planner_payload(root: Path, score: ScoreQualificationReceipt) -> Mapping[str, Any]:
    try:
        current = score.reload(root)
        current.planner_receipt.verify(root)
        payload = _mapping(
            json.loads(
                (root / current.planner_receipt.path).read_text(encoding="utf-8")
            ),
            label="scientific planner receipt",
        )
    except (OSError, json.JSONDecodeError, ProspectiveExecutionError) as exc:
        raise PlannedReportContractError(
            "planner receipt is not digest-verifiable"
        ) from exc
    unsigned = dict(payload)
    declared = unsigned.pop("planner_receipt_sha256", None)
    if declared != _canonical_sha256(unsigned):
        raise PlannedReportContractError("planner receipt self digest is invalid")
    if payload.get("bindings_sha256") != score.bindings_sha256:
        raise PlannedReportContractError("planner receipt crosses scientific bindings")
    return payload


def _canonical_method_provenance(
    *,
    score: ScoreQualificationReceipt,
    action_registry: VerifiedFile,
    audit: ScientificPreSubmitAudit,
    bindings: ProspectiveScientificBindings,
    peer: ScoreQualificationReceipt | None = None,
    safeguard: SafeguardQualificationReceipt | None = None,
) -> dict[str, Any]:
    return {
        "schema": REPORT_SLOT_EVIDENCE_V1,
        "action_registry": action_registry.to_json_dict(),
        "score_receipt": {
            "path": score.receipt_path,
            "sha256": score.receipt_sha256,
        },
        "run_receipt": score.run_receipt.to_json_dict(),
        "run": score.run.to_json_dict(),
        "metrics": score.metrics.to_json_dict(),
        "planner_receipt": score.planner_receipt.to_json_dict(),
        "planner_sequence": score.planner_sequence,
        "selected_action_id": score.scientific_action_id,
        "exact_reference_id": score.exact_reference_id,
        "exact_reference_identity_sha256": (
            score.exact_reference_identity_sha256
        ),
        "exact_reference_receipt": (
            score.exact_reference_receipt.to_json_dict()
        ),
        "exact_manifest_sha256": score.exact_manifest_sha256,
        "exact_data_sha256": score.exact_data_sha256,
        "accuracy_policy_sha256": score.accuracy_policy_sha256,
        "matrix_accuracy_policy_sha256": (
            bindings.matrix_accuracy_policy_sha256
        ),
        "scientific_pre_submit_audit": {
            "path": audit.receipt_path,
            "sha256": audit.receipt_sha256,
        },
        "audit_warnings": list(audit.warnings),
        "paired_score_receipt": (
            None
            if peer is None
            else {"path": peer.receipt_path, "sha256": peer.receipt_sha256}
        ),
        "paired_action_registry": (
            None if peer is None else peer.action_registry.to_json_dict()
        ),
        "paired_run_receipt": (
            None if peer is None else peer.run_receipt.to_json_dict()
        ),
        "safeguard_receipt": (
            None
            if safeguard is None
            else {
                "path": safeguard.receipt_path,
                "sha256": safeguard.receipt_sha256,
                "outcome": safeguard.outcome,
            }
        ),
    }


def _verified_file_from_binding(value: Any, *, label: str) -> VerifiedFile:
    _validate_file_binding(value, label=label)
    assert isinstance(value, Mapping)
    return VerifiedFile(str(value["path"]), str(value["sha256"]))


def _validate_candidate_provenance_replay(
    provenance: Mapping[str, Any],
    *,
    root: Path,
    campaign_registry: CampaignRegistry,
    matrix: ParallelBenchmarkMatrix,
    cell_id: str,
    method: str,
    label: str,
) -> int:
    """Deep-replay one history row as an independently audited candidate."""

    if "superseded_candidates" in provenance:
        raise PlannedReportContractError(
            f"{label} must not contain nested superseded history"
        )
    try:
        action_registry = _verified_file_from_binding(
            provenance.get("action_registry"),
            label=f"{label} action registry",
        )
        score_file = _verified_file_from_binding(
            provenance.get("score_receipt"),
            label=f"{label} score receipt",
        )
        score_file.verify(root)
        score = ScoreQualificationReceipt.load(
            root / score_file.path,
            repository_root=root,
        )
        if (
            VerifiedFile(score.receipt_path, score.receipt_sha256)
            != score_file
            or score.action_registry != action_registry
        ):
            raise PlannedReportContractError(
                f"{label} score binding is stale"
            )
        bindings, seed_identity, reference_identity, source_epoch = (
            _action_context(
                root=root,
                action_registry=action_registry,
                matrix=matrix,
                cell_id=cell_id,
            )
        )
        score.validate_bindings(root, bindings)
        arm = REPORT_METHOD_TO_ARM[method]
        if (
            score.cell_id != cell_id
            or score.method != arm
            or score.horizon != HORIZON_FULL
        ):
            raise PlannedReportContractError(
                f"{label} does not identify this T20 method slot"
            )
        _validate_score_action_registry(
            score,
            action_registry,
            root=root,
            bindings=bindings,
            seed_identity=seed_identity,
            reference_identity=reference_identity,
            source_epoch=source_epoch,
            label=label,
        )
        planner = _planner_payload(root, score)
        selected = _mapping(
            planner.get("selected_full_action_ids"),
            label=f"{label} selected full actions",
        )
        if selected.get(arm) != score.scientific_action_id:
            raise PlannedReportContractError(
                f"{label} was not selected by its planner"
            )
        if (
            planner.get("unresolved") is True
            or planner.get("cost_comparison_admissible") is True
        ):
            raise PlannedReportContractError(
                f"{label} is not candidate-state planner evidence"
            )
        audit_file = _verified_file_from_binding(
            provenance.get("scientific_pre_submit_audit"),
            label=f"{label} scientific pre-submit audit",
        )
        audit_file.verify(root)
        audit = ScientificPreSubmitAudit.load(
            root / audit_file.path,
            repository_root=root,
        )
        if VerifiedFile(audit.receipt_path, audit.receipt_sha256) != audit_file:
            raise PlannedReportContractError(
                f"{label} audit binding is stale"
            )
        replay_audit = _verify_report_audit(
            audit,
            root=root,
            campaign_registry=campaign_registry,
            contract=None,
            matrix=matrix,
            lane_id=bindings.lane_id,
            source_epoch=source_epoch,
            seed_receipt_sha256=bindings.seed_receipt_sha256,
            exact_reference_receipt_sha256=(
                bindings.reference_receipt_sha256
            ),
            primary_score=score,
            # Historical candidate audits bind earlier immutable report/state
            # snapshots.  The supplied audit is checked against the current
            # registry before this replay; requiring an old audit to equal the
            # newly rebound authority would make a second increment impossible.
            require_current_registry=False,
        )
        if provenance.get("audit_warnings") != list(replay_audit.warnings):
            raise PlannedReportContractError(
                f"{label} audit warnings differ from its authorizing audit"
            )
        expected = _canonical_method_provenance(
            score=score,
            action_registry=action_registry,
            audit=replay_audit,
            bindings=bindings,
        )
        if dict(provenance) != expected:
            raise PlannedReportContractError(
                f"{label} differs from its replayed evidence"
            )
        return score.planner_sequence
    except (
        OSError,
        json.JSONDecodeError,
        ProspectiveExecutionError,
        ValueError,
    ) as exc:
        raise PlannedReportContractError(
            f"{label} is not digest-replayable: {exc}"
        ) from exc


def _validate_superseded_candidate_chain(
    provenance: Mapping[str, Any],
    *,
    current_state: str,
    root: Path,
    campaign_registry: CampaignRegistry,
    matrix: ParallelBenchmarkMatrix,
    cell_id: str,
    method: str,
) -> None:
    history = provenance.get("superseded_candidates", [])
    assert isinstance(history, list)
    for index, row in enumerate(history):
        assert isinstance(row, Mapping)
        _validate_candidate_provenance_replay(
            row,
            root=root,
            campaign_registry=campaign_registry,
            matrix=matrix,
            cell_id=cell_id,
            method=method,
            label=f"superseded candidate history item {index}",
        )
    if current_state == "candidate":
        current = deepcopy(dict(provenance))
        current.pop("superseded_candidates", None)
        _validate_candidate_provenance_replay(
            current,
            root=root,
            campaign_registry=campaign_registry,
            matrix=matrix,
            cell_id=cell_id,
            method=method,
            label="current candidate provenance",
        )


def _current_method_slot(
    contract: Mapping[str, Any], *, cell_id: str, method: str
) -> Mapping[str, Any]:
    matches = [
        _mapping(raw_slot, label="method slot")
        for raw_cell in _sequence(contract.get("cells"), label="cells")
        for cell in (_mapping(raw_cell, label="cell"),)
        if cell.get("cell_id") == cell_id
        for raw_slot in _sequence(cell.get("method_slots"), label="method slots")
        if _mapping(raw_slot, label="method slot").get("method") == method
    ]
    if len(matches) != 1:
        raise PlannedReportContractError(
            f"expected one predeclared slot for {cell_id}/{method}, found {len(matches)}"
        )
    return matches[0]


def update_method_slot(
    contract: Mapping[str, Any],
    *,
    cell_id: str,
    method: str,
    state: str | None = None,
    action_registry: VerifiedFile,
    score_receipt: ScoreQualificationReceipt,
    paired_score_receipt: ScoreQualificationReceipt | None = None,
    safeguard_receipt: SafeguardQualificationReceipt | None = None,
    scientific_pre_submit_audit: ScientificPreSubmitAudit,
    campaign_registry: CampaignRegistry,
    repository_root: str | Path = ".",
    matrix_path: str | Path = DEFAULT_MATRIX_PATH,
) -> dict[str, Any]:
    """Promote one T20 slot from planner-selected, independently verified evidence."""

    root = Path(repository_root).resolve()
    validate_planned_report_contract(
        contract,
        repository_root=repository_root,
        matrix_path=matrix_path,
    )
    if method not in METHODS:
        raise PlannedReportContractError(f"unsupported method {method!r}")
    arm = REPORT_METHOD_TO_ARM[method]
    if state is not None and state not in METHOD_STATES:
        raise PlannedReportContractError(f"unsupported method-slot state {state!r}")
    if state == "pending":
        raise PlannedReportContractError(
            "pending slots cannot be produced by an evidence update"
        )
    source = Path(matrix_path)
    if not source.is_absolute():
        source = root / source
    matrix = ParallelBenchmarkMatrix.read(source.resolve(strict=True))
    try:
        score = score_receipt.reload(root)
    except ProspectiveExecutionError as exc:
        raise PlannedReportContractError(
            "score evidence is not digest-verifiable"
        ) from exc
    if score.action_registry != action_registry:
        raise PlannedReportContractError(
            "score action registry differs from the supplied registry"
        )
    bindings, seed_identity, reference_identity, source_epoch = _action_context(
        root=root,
        action_registry=action_registry,
        matrix=matrix,
        cell_id=cell_id,
    )
    try:
        score.validate_bindings(root, bindings)
        metrics = _mapping(
            json.loads(
                (root / score.metrics.path).read_text(encoding="utf-8")
            ),
            label="score metrics",
        )
    except (OSError, json.JSONDecodeError, ProspectiveExecutionError) as exc:
        raise PlannedReportContractError(
            "score evidence is not digest-verifiable"
        ) from exc
    if (
        score.cell_id != cell_id
        or score.method != arm
        or score.horizon != HORIZON_FULL
    ):
        raise PlannedReportContractError(
            "report promotion requires the requested cell/method at T20"
        )
    _validate_score_action_registry(
        score,
        action_registry,
        root=root,
        bindings=bindings,
        seed_identity=seed_identity,
        reference_identity=reference_identity,
        source_epoch=source_epoch,
        label="selected",
    )
    existing_slot = _current_method_slot(
        contract, cell_id=cell_id, method=method
    )
    try:
        replay_peer = (
            None
            if paired_score_receipt is None
            else paired_score_receipt.reload(root)
        )
        if replay_peer is not None:
            replay_peer.validate_bindings(root, bindings)
            _validate_score_action_registry(
                replay_peer,
                replay_peer.action_registry,
                root=root,
                bindings=bindings,
                seed_identity=seed_identity,
                reference_identity=reference_identity,
                source_epoch=source_epoch,
                label="qualified peer",
            )
        replay_safeguard = (
            None
            if safeguard_receipt is None
            else safeguard_receipt.reload(root)
        )
        if replay_safeguard is not None:
            replay_safeguard.validate_bindings(root, bindings)
    except ProspectiveExecutionError as exc:
        raise PlannedReportContractError(
            "replayed report evidence is not digest-verifiable"
        ) from exc
    replay_audit = _verify_report_audit(
        scientific_pre_submit_audit,
        root=root,
        campaign_registry=campaign_registry,
        contract=None,
        matrix=matrix,
        lane_id=bindings.lane_id,
        source_epoch=source_epoch,
        seed_receipt_sha256=bindings.seed_receipt_sha256,
        exact_reference_receipt_sha256=bindings.reference_receipt_sha256,
        primary_score=score,
        peer_score=replay_peer,
        safeguard=replay_safeguard,
    )
    planner = _planner_payload(root, score)
    selected = _mapping(
        planner.get("selected_full_action_ids"), label="selected full actions"
    )
    if selected.get(arm) != score.scientific_action_id:
        raise PlannedReportContractError(
            "planner did not select this T20 action for reporting"
        )

    derived_state = "candidate"
    peer: ScoreQualificationReceipt | None = None
    if safeguard_receipt is not None:
        safeguard = safeguard_receipt.reload(root)
        if (
            safeguard.cell_id != cell_id
            or safeguard.method != arm
            or safeguard.scientific_action_id != score.scientific_action_id
            or safeguard.score_receipt_sha256 != score.receipt_sha256
            or safeguard.planner_receipt != score.planner_receipt
            or safeguard.bindings_sha256 != bindings.digest
            or planner.get("complete") is not True
            or planner.get("unresolved") is not True
            or planner.get("cost_comparison_admissible") is not False
        ):
            raise PlannedReportContractError(
                "terminal safeguard does not bind the selected unresolved result"
            )
        derived_state = "unresolved"
    elif (
        planner.get("complete") is True
        and planner.get("unresolved") is False
        and planner.get("cost_comparison_admissible") is True
    ):
        if paired_score_receipt is None:
            raise PlannedReportContractError(
                "qualified promotion requires the selected peer score"
            )
        try:
            peer = paired_score_receipt.reload(root)
            peer.validate_bindings(root, bindings)
        except ProspectiveExecutionError as exc:
            raise PlannedReportContractError(
                "qualified peer score is not digest-verifiable"
            ) from exc
        other = ARM_AVQDS if arm == ARM_APM else ARM_APM
        if (
            peer.method != other
            or peer.cell_id != cell_id
            or peer.horizon != HORIZON_FULL
            or selected.get(other) != peer.scientific_action_id
            or peer.planner_receipt != score.planner_receipt
            or peer.accuracy_policy_sha256 != score.accuracy_policy_sha256
        ):
            raise PlannedReportContractError(
                "planner-selected qualified pair does not join"
            )
        policy = score.accuracy_policy(root)
        left, right = (
            (score, peer) if arm == ARM_APM else (peer, score)
        )
        left_metrics = left.accuracy_metrics(root)
        right_metrics = right.accuracy_metrics(root)
        if (
            not policy.comparison_eligible(
                left_metrics, key=bindings.key(ARM_APM), horizon=HORIZON_FULL
            )
            or not policy.comparison_eligible(
                right_metrics,
                key=bindings.key(ARM_AVQDS),
                horizon=HORIZON_FULL,
            )
            or not policy.pair_matches(
                left_metrics,
                left_key=bindings.key(ARM_APM),
                right_metrics=right_metrics,
                right_key=bindings.key(ARM_AVQDS),
                horizon=HORIZON_FULL,
            )
        ):
            raise PlannedReportContractError(
                "planner qualification fails T20 guards or factor-two matching"
            )
        derived_state = "qualified"
    elif planner.get("unresolved") is True or planner.get("cost_comparison_admissible") is True:
        raise PlannedReportContractError(
            "planner terminal flags are inconsistent with candidate evidence"
        )

    if state is not None and state != derived_state:
        raise PlannedReportContractError(
            "requested report state differs from planner-derived state"
        )
    raw_values = metrics.get("report_values")
    if not isinstance(raw_values, Mapping) or not raw_values:
        raise PlannedReportContractError(
            "score metrics have no nonempty report_values mapping"
        )
    values = dict(raw_values)
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in values.values()
    ):
        raise PlannedReportContractError("report values must be numerical")
    scored_values = score.accuracy_metrics(root).to_json_dict()
    for name, value in scored_values.items():
        if value is not None and (
            name not in values
            or not isinstance(values[name], (int, float))
            or not _numeric_equal(float(values[name]), float(value))
        ):
            raise PlannedReportContractError(
                "report values diverge from exact-scored accuracy metrics"
            )
    provenance = _canonical_method_provenance(
        score=score,
        action_registry=action_registry,
        audit=replay_audit,
        bindings=bindings,
        peer=peer,
        safeguard=replay_safeguard,
    )
    existing_state = str(existing_slot.get("state") or "")
    existing_provenance = existing_slot.get("provenance")
    if isinstance(existing_provenance, Mapping):
        _validate_superseded_candidate_chain(
            existing_provenance,
            current_state=existing_state,
            root=root,
            campaign_registry=campaign_registry,
            matrix=matrix,
            cell_id=cell_id,
            method=method,
        )
    comparable_existing = (
        None
        if not isinstance(existing_provenance, Mapping)
        else deepcopy(dict(existing_provenance))
    )
    if comparable_existing is not None:
        comparable_existing.pop("superseded_candidates", None)
    if (
        existing_state == derived_state
        and comparable_existing == provenance
        and existing_slot.get("values") == values
    ):
        return deepcopy(dict(contract))
    if existing_state in {"qualified", "unresolved"}:
        raise PlannedReportContractError(
            f"terminal {existing_state} slot is immutable"
        )

    # New evidence and candidate supersession must be authorized against the
    # exact pre-update contract.  Canonical replay above instead authenticates
    # the original audit and its full evidence inventory without pretending
    # that the audit was written against the already-updated self digest.
    _verify_report_audit(
        scientific_pre_submit_audit,
        root=root,
        campaign_registry=campaign_registry,
        contract=contract,
        matrix=matrix,
        lane_id=bindings.lane_id,
        source_epoch=source_epoch,
        seed_receipt_sha256=bindings.seed_receipt_sha256,
        exact_reference_receipt_sha256=bindings.reference_receipt_sha256,
        primary_score=score,
        peer_score=replay_peer,
        safeguard=replay_safeguard,
    )
    result = deepcopy(dict(contract))
    before_structure = result[TUPLE_STRUCTURE_DIGEST_KEY]
    matches = 0
    for cell in result["cells"]:
        if cell["cell_id"] != cell_id:
            continue
        for slot in cell["method_slots"]:
            if slot["method"] != method:
                continue
            existing_state = str(slot["state"])
            existing_provenance = slot.get("provenance")
            if existing_state in {"qualified", "unresolved"}:
                if (
                    existing_state == derived_state
                    and existing_provenance == provenance
                    and slot.get("values") == values
                ):
                    return deepcopy(dict(contract))
                raise PlannedReportContractError(
                    f"terminal {existing_state} slot is immutable"
                )
            if existing_state == "candidate":
                if existing_provenance == provenance and slot.get("values") == values:
                    return deepcopy(dict(contract))
                old_sequence = int(
                    _mapping(existing_provenance, label="candidate provenance").get(
                        "planner_sequence", -1
                    )
                )
                if score.planner_sequence <= old_sequence:
                    raise PlannedReportContractError(
                        "candidate supersession requires a newer selected planner action"
                    )
                old_without_history = deepcopy(dict(existing_provenance))
                prior = list(old_without_history.pop("superseded_candidates", []))
                prior.append(old_without_history)
                provenance["superseded_candidates"] = prior
            slot["state"] = derived_state
            slot["values"] = values
            slot["provenance"] = provenance
            matches += 1
    if matches != 1:
        raise PlannedReportContractError(
            f"expected one predeclared slot for {cell_id}/{method}, found {matches}"
        )
    result["counts"] = _state_counts(result)
    sealed = _seal(result)
    if sealed[TUPLE_STRUCTURE_DIGEST_KEY] != before_structure:
        raise PlannedReportContractError("method update changed tuple structure")
    validate_planned_report_contract(
        sealed,
        repository_root=repository_root,
        matrix_path=matrix_path,
    )
    return sealed


def update_exact_reference(
    contract: Mapping[str, Any],
    *,
    cell_id: str,
    reference_receipt: ExactReferenceReceipt,
    seed_receipt: SeedQualificationReceipt,
    source_epoch_receipt: SourceEpochReceipt,
    scientific_pre_submit_audit: ScientificPreSubmitAudit,
    campaign_registry: CampaignRegistry,
    repository_root: str | Path = ".",
    matrix_path: str | Path = DEFAULT_MATRIX_PATH,
) -> dict[str, Any]:
    """Update one exact slot from its verified manifest/data/receipt tuple."""

    root = Path(repository_root).resolve()
    validate_planned_report_contract(
        contract,
        repository_root=repository_root,
        matrix_path=matrix_path,
    )
    source = Path(matrix_path)
    if not source.is_absolute():
        source = root / source
    matrix = ParallelBenchmarkMatrix.read(source.resolve(strict=True))
    seed_identity, reference_identity = _registered_rows(matrix, cell_id)
    try:
        source_epoch = SourceEpochReceipt.load(
            root / source_epoch_receipt.receipt_path,
            repository_root=root,
            expected_epoch=source_epoch_receipt.source_epoch,
        )
        seed = SeedQualificationReceipt.load(
            root / seed_receipt.receipt_path,
            repository_root=root,
            expected_seed=seed_identity,
            expected_source_epoch=source_epoch.source_epoch,
        )
        reference = ExactReferenceReceipt.load(
            root / reference_receipt.receipt_path,
            repository_root=root,
            expected_reference=reference_identity,
            expected_seed_identity=seed_identity,
            expected_seed_receipt=seed,
            expected_source_epoch=source_epoch.source_epoch,
        )
    except ProspectiveExecutionError as exc:
        raise PlannedReportContractError(
            "exact-reference evidence is not digest-verifiable"
        ) from exc
    if (
        source_epoch.receipt_sha256 != source_epoch_receipt.receipt_sha256
        or seed.receipt_sha256 != seed_receipt.receipt_sha256
        or reference.receipt_sha256 != reference_receipt.receipt_sha256
    ):
        raise PlannedReportContractError("exact-reference evidence changed after load")
    replay_audit = _verify_report_audit(
        scientific_pre_submit_audit,
        root=root,
        campaign_registry=campaign_registry,
        contract=None,
        matrix=matrix,
        lane_id=reference_identity.lane_id,
        source_epoch=source_epoch.source_epoch,
        seed_receipt_sha256=seed.receipt_sha256,
        exact_reference_receipt_sha256=reference.receipt_sha256,
        source_epoch_receipt=source_epoch,
        seed_receipt=seed,
        exact_reference_receipt=reference,
    )
    artifact = {
        "schema": EXACT_SLOT_EVIDENCE_V1,
        "reference_id": reference.reference_id,
        "reference_identity_sha256": reference.reference_identity_sha256,
        "manifest": reference.manifest.to_json_dict(),
        "data": reference.data.to_json_dict(),
        "reference_receipt": {
            "path": reference.receipt_path,
            "sha256": reference.receipt_sha256,
        },
        "seed_receipt": {
            "path": seed.receipt_path,
            "sha256": seed.receipt_sha256,
        },
        "source_epoch_receipt": {
            "path": source_epoch.receipt_path,
            "sha256": source_epoch.receipt_sha256,
        },
        "seed_artifact": seed.artifact.to_json_dict(),
        "source_epoch_identity_sha256": (
            reference.source_epoch_identity_sha256
        ),
        "scientific_pre_submit_audit": {
            "path": replay_audit.receipt_path,
            "sha256": replay_audit.receipt_sha256,
        },
        "audit_warnings": list(replay_audit.warnings),
    }
    current_exact = [
        _mapping(cell.get("exact_reference"), label="exact_reference")
        for raw_cell in _sequence(contract.get("cells"), label="cells")
        for cell in (_mapping(raw_cell, label="cell"),)
        if cell.get("cell_id") == cell_id
    ]
    if len(current_exact) != 1:
        raise PlannedReportContractError(
            f"expected one predeclared exact reference for {cell_id}, "
            f"found {len(current_exact)}"
        )
    if (
        current_exact[0].get("state") == "verified"
        and current_exact[0].get("artifact") == artifact
    ):
        return deepcopy(dict(contract))
    _verify_report_audit(
        scientific_pre_submit_audit,
        root=root,
        campaign_registry=campaign_registry,
        contract=contract,
        matrix=matrix,
        lane_id=reference_identity.lane_id,
        source_epoch=source_epoch.source_epoch,
        seed_receipt_sha256=seed.receipt_sha256,
        exact_reference_receipt_sha256=reference.receipt_sha256,
        source_epoch_receipt=source_epoch,
        seed_receipt=seed,
        exact_reference_receipt=reference,
    )
    result = deepcopy(dict(contract))
    before_structure = result[TUPLE_STRUCTURE_DIGEST_KEY]
    matches = 0
    for cell in result["cells"]:
        if cell["cell_id"] != cell_id:
            continue
        slot = cell["exact_reference"]
        if (
            slot.get("reference_id") != reference.reference_id
            or slot.get("reference_identity_sha256")
            != reference.reference_identity_sha256
        ):
            raise PlannedReportContractError(
                "exact slot identity diverges from the registered reference"
            )
        if slot.get("state") == "verified":
            if slot.get("artifact") == artifact:
                return deepcopy(dict(contract))
            raise PlannedReportContractError(
                "verified exact-reference slot is immutable"
            )
        cell["exact_reference"]["state"] = "verified"
        cell["exact_reference"]["artifact"] = artifact
        matches += 1
    if matches != 1:
        raise PlannedReportContractError(
            f"expected one predeclared exact reference for {cell_id}, found {matches}"
        )
    result["counts"] = _state_counts(result)
    sealed = _seal(result)
    if sealed[TUPLE_STRUCTURE_DIGEST_KEY] != before_structure:
        raise PlannedReportContractError("exact update changed tuple structure")
    validate_planned_report_contract(
        sealed,
        repository_root=repository_root,
        matrix_path=matrix_path,
    )
    return sealed


def _pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_default_planned_report_contracts(
    *,
    repository_root: str | Path = ".",
    matrix_path: str | Path = DEFAULT_MATRIX_PATH,
    check_only: bool = False,
) -> dict[str, dict[str, Any]]:
    """Write or verify every stable future-lane report-contract path."""

    root = Path(repository_root).resolve()
    results: dict[str, dict[str, Any]] = {}
    for lane_id, relative_path in DEFAULT_REPORT_PATHS.items():
        contract = build_planned_report_contract(
            lane_id=lane_id,
            repository_root=root,
            matrix_path=matrix_path,
        )
        path = root / relative_path
        content = _pretty_json_bytes(contract)
        if check_only:
            if not path.is_file() or path.read_bytes() != content:
                raise PlannedReportContractError(
                    f"planned report contract is absent or stale: {relative_path}"
                )
        else:
            _atomic_write(path, content)
        results[lane_id] = {
            "path": relative_path.as_posix(),
            "sha256": hashlib.sha256(content).hexdigest(),
            "planned_cells": contract["counts"]["planned_cells"],
            "planned_method_slots": contract["counts"]["planned_method_slots"],
            "pending_method_slots": contract["counts"]["pending_method_slots"],
            "tuple_structure_sha256": contract[TUPLE_STRUCTURE_DIGEST_KEY],
        }
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX_PATH))
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that all stable contracts exist byte-for-byte",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary = write_default_planned_report_contracts(
        repository_root=args.repo_root,
        matrix_path=args.matrix,
        check_only=args.check,
    )
    print(json.dumps({"ok": True, "contracts": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_REPORT_PATHS",
    "EXACT_REFERENCE_STATES",
    "METHOD_STATES",
    "PLANNED_REPORT_CONTRACT_V1",
    "PlannedReportContractError",
    "build_planned_report_contract",
    "update_exact_reference",
    "update_method_slot",
    "validate_planned_report_contract",
    "write_default_planned_report_contracts",
]
