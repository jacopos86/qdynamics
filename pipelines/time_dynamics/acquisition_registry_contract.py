#!/usr/bin/env python3
"""Build and validate the disabled Paper-II acquisition registry contract.

This module binds the frozen 39-cell matrix to its seed/reference identities,
importable executor registry, planned report slots, SIAM stable manifests, and
prospective scientific planner.  It is intentionally not an activation tool:
production remains null, every lane remains submission-disabled, and missing
transport/source/package/audit evidence remains an explicit blocker.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from pipelines.time_dynamics.acquisition_supervisor import (
    CampaignRegistry,
    REGISTRY_SCHEMA_V1,
)
from pipelines.time_dynamics.parallel_benchmark_matrix import (
    DEFAULT_MATRIX_PATH,
    ParallelBenchmarkMatrix,
)
from pipelines.time_dynamics.planned_report_contracts import (
    DEFAULT_REPORT_PATHS,
    REPORT_CONTRACT_SELF_DIGEST_KEY,
    TUPLE_STRUCTURE_DIGEST_KEY,
    validate_planned_report_contract,
)
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    LANE_HH,
    LANE_HUBBARD,
    LANE_SIAM,
    build_identity_registry,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    binding_registry_payload,
)


REGISTRY_CONTRACT_SCHEMA_V1 = "paper_ii_acquisition_contract_bindings_v1"
CANONICAL_REGISTRY_ID = "paper_ii_parallel_benchmarks_authorized_v2"
CANONICAL_SUPERVISOR_STATE_ROOT = Path("output/time_dynamics_acquisition")
LEGACY_LANE_ID = "paper_ii_hh_nph1"
LEGACY_SCIENTIFIC_STATE = Path("output/paper_ii_staged/ledger_current.json")
LEGACY_REPORT_CONTRACT = Path(
    "output/pdf/paper_ii_staged_accuracy_campaign/paper_ii_staged_report.json"
)
LEGACY_COVERAGE_INDEX = Path(
    "output/paper_ii_staged/acquisition_supervisor/legacy_coverage_index_v1.json"
)
DECISION_HANDOFF_SHA256 = (
    "420911c88b75516f0fc0218e447b686aca2299e4923b4d99249c6f337a6f03b0"
)
DEFAULT_REGISTRY_PATH = Path(
    "pipelines/time_dynamics/campaigns/time_dynamics_acquisition_registry_v1.json"
)
BOOTSTRAP_SOURCE = Path(
    "pipelines/time_dynamics/prospective_campaign_bootstrap.py"
)
EXECUTOR_SOURCE = Path(
    "pipelines/time_dynamics/prospective_execution_bindings.py"
)
PLANNER_SOURCE = Path(
    "pipelines/time_dynamics/prospective_scientific_planner.py"
)
ADAPTER_SOURCE = Path(
    "pipelines/time_dynamics/chtc_acquisition_operations.py"
)
SIAM_SCIENTIFIC_STATE = Path(
    "output/siam_screening/nb4_pilot_v1/acquisition_supervisor/"
    "scientific_state_current.json"
)
SIAM_REPORT_CONTRACT = Path(
    "output/siam_screening/nb4_pilot_v1/acquisition_supervisor/"
    "report_contract_current.json"
)

FUTURE_LANES = (LANE_HH, LANE_HUBBARD)
CANONICAL_LANE_IDS = (LEGACY_LANE_ID, LANE_HH, LANE_SIAM, LANE_HUBBARD)
FUTURE_REQUIRED_BLOCKERS = (
    "seed_manifest_binding",
    "exact_reference_manifest_binding",
    "action_registry_binding",
    "scout_anchor_planner_v2_binding",
    "report_slots_binding",
    "package_builder_binding",
    "source_epoch_binding",
    "production_operations_adapter_binding",
    "authenticated_transport_binding",
    "remote_workdir_binding",
    "lane_mutation_handler_binding",
    "production_service_preflight_binding",
    "scientific_pre_submit_audit_binding",
    "apm_exchange_family_cost_normalization_binding",
    "apm_exchange_rejection_continuation_binding",
)
PRODUCTION_REQUIRED_BLOCKERS = (
    "authenticated_transport_binding",
    "bootstrap_action_identity_binding",
    "lane_mutation_handler_binding",
    "remote_workdir_binding",
    "package_builder_binding",
    "production_service_preflight_binding",
    "scientific_pre_submit_audit_binding",
)


class AcquisitionRegistryContractError(ValueError):
    """The registry does not match its frozen local scientific contracts."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repository_relative(root: Path, path: Path, *, label: str) -> Path:
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise AcquisitionRegistryContractError(
            f"{label} is outside repository root: {path}"
        ) from exc
    if not relative.parts or ".." in relative.parts:
        raise AcquisitionRegistryContractError(
            f"{label} is not a repository-relative file: {path}"
        )
    return relative


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise AcquisitionRegistryContractError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcquisitionRegistryContractError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise AcquisitionRegistryContractError(f"{label} must be a JSON object")
    return value


def _module_callable(value: str, *, expected_source: Path) -> None:
    module, separator, name = value.partition(":")
    imported = importlib.import_module(module) if separator else None
    function = None if imported is None else getattr(imported, name, None)
    module_path = None if imported is None else getattr(imported, "__file__", None)
    function_path = None if not callable(function) else inspect.getsourcefile(function)
    if (
        not separator
        or not callable(function)
        or module_path is None
        or function_path is None
        or Path(module_path).resolve() != expected_source.resolve()
        or Path(function_path).resolve() != expected_source.resolve()
    ):
        raise AcquisitionRegistryContractError(
            f"registered builder is not bound to its source: {value}"
        )


def _source_binding(root: Path, path: Path, builder: str) -> dict[str, Any]:
    relative = _repository_relative(root, path, label="bound source")
    source = root / relative
    if not source.is_file():
        raise AcquisitionRegistryContractError(f"bound source is missing: {path}")
    _module_callable(builder, expected_source=source)
    return {
        "builder": builder,
        "source": {"path": relative.as_posix(), "sha256": _file_sha(source)},
    }


def _self_digest(payload: Mapping[str, Any], key: str) -> str:
    unsigned = dict(payload)
    declared = unsigned.pop(key, None)
    actual = _canonical_sha(unsigned)
    if declared != actual:
        raise AcquisitionRegistryContractError(
            f"document self digest is invalid for {key}"
        )
    return actual


def _planned_contract_binding(
    root: Path,
    lane_id: str,
    path: Path,
) -> tuple[dict[str, Any], set[str]]:
    relative = _repository_relative(
        root, path, label=f"{lane_id} planned report contract"
    )
    target = root / relative
    payload = _read_object(target, label=f"{lane_id} planned report contract")
    validate_planned_report_contract(
        payload, repository_root=root, matrix_path=DEFAULT_MATRIX_PATH
    )
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise AcquisitionRegistryContractError("planned contract has no cells")
    counts = payload.get("counts")
    if not isinstance(counts, Mapping):
        raise AcquisitionRegistryContractError("planned contract has no counts")
    return (
        {
            "path": relative.as_posix(),
            "file_sha256": _file_sha(target),
            "report_contract_sha256": str(
                payload[REPORT_CONTRACT_SELF_DIGEST_KEY]
            ),
            "tuple_structure_sha256": str(payload[TUPLE_STRUCTURE_DIGEST_KEY]),
            "planned_cells": int(counts["planned_cells"]),
            "planned_exact_references": int(
                counts["planned_exact_references"]
            ),
            "planned_method_slots": int(counts["planned_method_slots"]),
        },
        {str(row["cell_id"]) for row in cells},
    )


def build_scientific_contracts(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    matrix_relative = _repository_relative(
        root, DEFAULT_MATRIX_PATH, label="parallel benchmark matrix"
    )
    matrix_path = root / matrix_relative
    matrix = ParallelBenchmarkMatrix.read(matrix_path)
    matrix_digests = {
        matrix.raw_sha256,
        matrix.payload_sha256,
        matrix.expanded_semantic_sha256,
    }
    if len(matrix.cells) != 39 or len(matrix_digests) != 3:
        raise AcquisitionRegistryContractError(
            "matrix must bind 39 cells and three distinct digest vocabularies"
        )
    identity = build_identity_registry(matrix)
    if len(identity.seeds) != 8 or len(identity.references) != 39:
        raise AcquisitionRegistryContractError(
            "identity registry must bind exactly 8 seeds and 39 references"
        )
    reference_cells = {row.cell_id for row in identity.references}
    matrix_cells = {row.cell_id for row in matrix.cells}
    if reference_cells != matrix_cells:
        raise AcquisitionRegistryContractError(
            "identity references do not cover the exact matrix cell set"
        )
    lane_counts: dict[str, dict[str, int]] = {}
    for lane_id in (LANE_HH, LANE_SIAM, LANE_HUBBARD):
        lane_counts[lane_id] = {
            "seeds": sum(row.lane_id == lane_id for row in identity.seeds),
            "references": sum(
                row.lane_id == lane_id for row in identity.references
            ),
        }

    executor = binding_registry_payload()
    bindings = executor.get("bindings")
    if (
        not isinstance(bindings, list)
        or len(bindings) != 16
        or executor.get("qiskit_compilation") is not False
        or executor.get("production_submission_enabled") is not False
    ):
        raise AcquisitionRegistryContractError("executor registry is not fail-closed")

    planned: dict[str, Any] = {}
    planned_cells: set[str] = set()
    for lane_id, path in DEFAULT_REPORT_PATHS.items():
        binding, cells = _planned_contract_binding(root, lane_id, path)
        planned[lane_id] = binding
        if planned_cells.intersection(cells):
            raise AcquisitionRegistryContractError(
                "planned contracts repeat a physical cell"
            )
        planned_cells.update(cells)
    if planned_cells != matrix_cells:
        raise AcquisitionRegistryContractError(
            "planned report contracts do not cover the exact matrix cell set"
        )
    if sum(row["planned_exact_references"] for row in planned.values()) != 39:
        raise AcquisitionRegistryContractError("planned exact-reference count is not 39")
    if sum(row["planned_method_slots"] for row in planned.values()) != 78:
        raise AcquisitionRegistryContractError("planned method-slot count is not 78")

    siam_state_path = root / SIAM_SCIENTIFIC_STATE
    siam_report_path = root / SIAM_REPORT_CONTRACT
    siam_state = _read_object(siam_state_path, label="SIAM stable scientific state")
    siam_report = _read_object(siam_report_path, label="SIAM stable report contract")
    state_self = _self_digest(siam_state, "scientific_state_sha256")
    report_self = _self_digest(siam_report, "report_contract_sha256")
    if siam_state.get("source_epoch") != siam_report.get("source_epoch"):
        raise AcquisitionRegistryContractError("SIAM stable source epochs diverge")
    if siam_report.get("qiskit_compilation_used") not in (None, False):
        raise AcquisitionRegistryContractError("SIAM stable report used Qiskit")

    contracts = {
        "schema": REGISTRY_CONTRACT_SCHEMA_V1,
        "matrix": {
            "path": matrix_relative.as_posix(),
            "matrix_id": matrix.matrix_id,
            "raw_sha256": matrix.raw_sha256,
            "payload_sha256": matrix.payload_sha256,
            "expanded_semantic_sha256": matrix.expanded_semantic_sha256,
            "accuracy_policy_spec_sha256": (
                matrix.accuracy_policy_spec_sha256
            ),
            "physical_cell_count": len(matrix.cells),
        },
        "identity_registry": {
            **_source_binding(
                root,
                BOOTSTRAP_SOURCE,
                "pipelines.time_dynamics.prospective_campaign_bootstrap:build_identity_registry",
            ),
            "registry_sha256": identity.digest,
            "seed_count": len(identity.seeds),
            "reference_count": len(identity.references),
            "lane_counts": lane_counts,
        },
        "executor_registry": {
            **_source_binding(
                root,
                EXECUTOR_SOURCE,
                "pipelines.time_dynamics.prospective_execution_bindings:binding_registry_payload",
            ),
            "bindings_sha256": str(executor["bindings_sha256"]),
            "binding_count": len(bindings),
            "handlers_importable": True,
            "qiskit_compilation": False,
            "production_submission_enabled": False,
        },
        "planned_report_contracts": planned,
        "siam_stable": {
            "scientific_state": {
                "path": SIAM_SCIENTIFIC_STATE.as_posix(),
                "file_sha256": _file_sha(siam_state_path),
                "scientific_state_sha256": state_self,
            },
            "report_contract": {
                "path": SIAM_REPORT_CONTRACT.as_posix(),
                "file_sha256": _file_sha(siam_report_path),
                "report_contract_sha256": report_self,
            },
            "source_epoch": dict(siam_state["source_epoch"]),
        },
        "scientific_planner": {
            **_source_binding(
                root,
                PLANNER_SOURCE,
                "pipelines.time_dynamics.prospective_scientific_planner:plan_prospective_cell",
            ),
        },
        "scientific_pre_submit_audits": {
            lane_id: {"required": True, "status": "pending", "receipt": None}
            for lane_id in FUTURE_LANES
        },
    }
    return contracts


def _canonical_lane_payloads(
    root: Path, contracts: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Derive the exact four-lane registry without consulting its target file."""

    matrix = contracts["matrix"]
    planned = contracts["planned_report_contracts"]
    siam_stable = contracts["siam_stable"]
    coverage_path = root / LEGACY_COVERAGE_INDEX
    coverage = _read_object(coverage_path, label="legacy coverage index")
    if (
        coverage.get("lane_id") != LEGACY_LANE_ID
        or coverage.get("decision_handoff_sha256") != DECISION_HANDOFF_SHA256
        or coverage.get("scientific_state") != LEGACY_SCIENTIFIC_STATE.as_posix()
    ):
        raise AcquisitionRegistryContractError(
            "legacy coverage index is not bound to the canonical baseline"
        )
    covered = int(coverage.get("covered_process_count", -1))
    receipts = int(coverage.get("normalized_receipt_count", -1))
    rows = coverage.get("rows")
    if not isinstance(rows, list) or covered != len(rows) or covered < 1 or receipts < 1:
        raise AcquisitionRegistryContractError("legacy coverage counts are invalid")

    def future_lane(
        *,
        lane_id: str,
        kind: str,
        attempts_root: str,
        state_root: str,
        scientific_state: str,
    ) -> dict[str, Any]:
        report = planned[lane_id]
        return {
            "lane_id": lane_id,
            "kind": kind,
            "monitor_enabled": False,
            "submission_enabled": False,
            "planner_authority": (
                "pipelines.time_dynamics.prospective_scientific_planner:"
                "plan_prospective_cell"
            ),
            "blocked_by": list(FUTURE_REQUIRED_BLOCKERS),
            "attempts_root": attempts_root,
            "state_root": state_root,
            "scientific_state": scientific_state,
            "report_contract": report["path"],
            "metadata": {
                "controller": "state_motion_5e-5",
                "matrix_manifest": matrix["path"],
                "matrix_raw_sha256": matrix["raw_sha256"],
                "matrix_payload_sha256": matrix["payload_sha256"],
                "matrix_expanded_semantic_sha256": matrix[
                    "expanded_semantic_sha256"
                ],
                "matrix_accuracy_policy_sha256": matrix[
                    "accuracy_policy_spec_sha256"
                ],
                "physical_cell_count": report["planned_cells"],
                "method_slot_count": report["planned_method_slots"],
                "planned_report_contract_file_sha256": report["file_sha256"],
                "planned_report_contract_sha256": report[
                    "report_contract_sha256"
                ],
                "qiskit_compilation_requested": False,
            },
        }

    source_epoch = siam_stable["source_epoch"]
    lanes = [
        {
            "lane_id": LEGACY_LANE_ID,
            "kind": "paper_ii_hh_controller_baseline_v2",
            "monitor_enabled": True,
            "submission_enabled": False,
            "attempts_root": "output/paper_ii_staged/attempts",
            "state_root": "output/paper_ii_staged/acquisition_supervisor",
            "scientific_state": LEGACY_SCIENTIFIC_STATE.as_posix(),
            "report_contract": LEGACY_REPORT_CONTRACT.as_posix(),
            "planner_authority": (
                "pipelines.time_dynamics.staged_campaign."
                "StagedCampaign.next_actions"
            ),
            "blocked_by": ["production_operations_adapter_binding"],
            "metadata": {
                "decision_handoff_sha256": DECISION_HANDOFF_SHA256,
                "downstream_controller": "state_motion_5e-5",
                "legacy_coverage_index": LEGACY_COVERAGE_INDEX.as_posix(),
                "legacy_coverage_index_sha256": _file_sha(coverage_path),
                "legacy_coverage_process_count": covered,
                "legacy_coverage_receipt_count": receipts,
                "physical_identity": {
                    "U": 4.0,
                    "g": 0.5,
                    "lattice_sites": 2,
                    "n_ph_max": 1,
                },
                "preserved_controllers": [
                    "delta_theta_5e-5",
                    "state_motion_1e-3_plus_parameter_5e-5",
                    "state_motion_5e-5",
                ],
                "prior_root": {"state": "absent"},
                "qiskit_compilation_requested": False,
                "reporting_cost_authority": "logical_and_graph_proxies",
                "role": "canonical_controller_comparison_baseline",
            },
        },
        future_lane(
            lane_id=LANE_HH,
            kind="paper_ii_hh_cutoff_interaction_product",
            attempts_root="output/paper_ii_hh_product_v1/attempts",
            state_root="output/paper_ii_hh_product_v1/acquisition_supervisor",
            scientific_state=(
                "output/paper_ii_hh_product_v1/scientific_state_current.json"
            ),
        ),
        {
            "lane_id": LANE_SIAM,
            "kind": "siam_nb4_comparator_first",
            "monitor_enabled": True,
            "submission_enabled": False,
            "attempts_root": "output/siam_screening/nb4_pilot_v1/attempts",
            "state_root": (
                "output/siam_screening/nb4_pilot_v1/acquisition_supervisor"
            ),
            "scientific_state": siam_stable["scientific_state"]["path"],
            "report_contract": siam_stable["report_contract"]["path"],
            "planner_authority": (
                "pipelines.time_dynamics.siam_nb4_ladder.plan_siam_nb4_ladder"
            ),
            "blocked_by": ["production_operations_adapter_binding"],
            "metadata": {
                "controller": "state_motion_5e-5",
                "decision_handoff_sha256": DECISION_HANDOFF_SHA256,
                "n_bath": 4,
                "qiskit_compilation_requested": False,
                "reporting_cost_authority": "logical_and_graph_proxies",
                "scientific_state_file_sha256": siam_stable[
                    "scientific_state"
                ]["file_sha256"],
                "scientific_state_sha256": siam_stable["scientific_state"][
                    "scientific_state_sha256"
                ],
                "report_contract_file_sha256": siam_stable["report_contract"][
                    "file_sha256"
                ],
                "report_contract_sha256": siam_stable["report_contract"][
                    "report_contract_sha256"
                ],
                "source_epoch": source_epoch["source_epoch_id"],
                "source_revision": source_epoch["source_revision"],
            },
        },
        future_lane(
            lane_id=LANE_HUBBARD,
            kind="paper_ii_hubbard_scale_pilot",
            attempts_root="output/paper_ii_hubbard_l4_scale_v1/attempts",
            state_root=(
                "output/paper_ii_hubbard_l4_scale_v1/acquisition_supervisor"
            ),
            scientific_state=(
                "output/paper_ii_hubbard_l4_scale_v1/scientific_state_current.json"
            ),
        ),
    ]
    if tuple(lane["lane_id"] for lane in lanes) != CANONICAL_LANE_IDS:
        raise AcquisitionRegistryContractError("canonical lane derivation is incomplete")
    return lanes


def build_registry_payload(
    repository_root: str | Path,
    template_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    # ``template_path`` is retained for API compatibility only.  The expected
    # document must never be derived from the file under test.
    del template_path
    contracts = build_scientific_contracts(root)
    blocked = {
        "status": "blocked",
        "adapter_id": "chtc_authenticated_controlmaster_v1",
        "factory": (
            "pipelines.time_dynamics.chtc_acquisition_operations:build_operations"
        ),
        "source": {
            "path": ADAPTER_SOURCE.as_posix(),
            "sha256": _file_sha(root / ADAPTER_SOURCE),
        },
        "blocked_by": list(PRODUCTION_REQUIRED_BLOCKERS),
    }
    return {
        "schema": REGISTRY_SCHEMA_V1,
        "registry_id": CANONICAL_REGISTRY_ID,
        "supervisor_state_root": CANONICAL_SUPERVISOR_STATE_ROOT.as_posix(),
        "lanes": _canonical_lane_payloads(root, contracts),
        "scientific_contracts": contracts,
        "production_operations": None,
        "production_operations_blocked": blocked,
    }


def validate_registry_payload(
    payload: Mapping[str, Any],
    repository_root: str | Path,
) -> None:
    root = Path(repository_root).resolve()
    row = dict(payload)
    if row.get("schema") != REGISTRY_SCHEMA_V1:
        raise AcquisitionRegistryContractError("unsupported acquisition registry schema")
    if row.get("production_operations") is not None:
        raise AcquisitionRegistryContractError("production operations must remain null")
    blocked = row.get("production_operations_blocked")
    if not isinstance(blocked, Mapping) or not set(PRODUCTION_REQUIRED_BLOCKERS).issubset(
        set(blocked.get("blocked_by") or ())
    ):
        raise AcquisitionRegistryContractError(
            "production adapter blockers are incomplete"
        )
    if blocked.get("source") != {
        "path": ADAPTER_SOURCE.as_posix(),
        "sha256": _file_sha(root / ADAPTER_SOURCE),
    }:
        raise AcquisitionRegistryContractError("production adapter source is stale")
    lanes = row.get("lanes")
    if not isinstance(lanes, list):
        raise AcquisitionRegistryContractError("registry lanes must be an array")
    by_lane = {
        str(lane.get("lane_id")): lane
        for lane in lanes
        if isinstance(lane, Mapping)
    }
    if len(by_lane) != len(lanes):
        raise AcquisitionRegistryContractError("registry lanes are malformed")
    if tuple(by_lane) != CANONICAL_LANE_IDS:
        raise AcquisitionRegistryContractError(
            "registry does not contain the exact canonical lane set/order"
        )
    if any(lane.get("submission_enabled") is not False for lane in by_lane.values()):
        raise AcquisitionRegistryContractError("all submissions must remain disabled")
    for lane_id in FUTURE_LANES:
        lane = by_lane.get(lane_id)
        if lane is None or not set(FUTURE_REQUIRED_BLOCKERS).issubset(
            set(lane.get("blocked_by") or ())
        ):
            raise AcquisitionRegistryContractError(
                f"future lane blockers are incomplete: {lane_id}"
            )
        if lane.get("planner_authority") != (
            "pipelines.time_dynamics.prospective_scientific_planner:"
            "plan_prospective_cell"
        ):
            raise AcquisitionRegistryContractError(
                f"future lane planner binding is stale: {lane_id}"
            )
    expected_contracts = build_scientific_contracts(root)
    if row.get("scientific_contracts") != expected_contracts:
        raise AcquisitionRegistryContractError("scientific contract bindings are stale")
    expected = build_registry_payload(root)
    expected_lanes = {
        lane["lane_id"]: lane for lane in expected["lanes"]
    }
    for lane_id in CANONICAL_LANE_IDS:
        if by_lane[lane_id] != expected_lanes[lane_id]:
            raise AcquisitionRegistryContractError(
                f"canonical lane paths/planner/roots/digests are stale: {lane_id}"
            )
    for key in (
        "registry_id",
        "supervisor_state_root",
        "production_operations",
        "production_operations_blocked",
    ):
        if row.get(key) != expected[key]:
            raise AcquisitionRegistryContractError(
                f"canonical registry binding is stale: {key}"
            )
    if row != expected:
        raise AcquisitionRegistryContractError(
            "registry contains noncanonical or unbound fields"
        )
    descriptor, temporary = tempfile.mkstemp(
        prefix="paper-ii-registry-validation-", suffix=".json"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(row))
        registry = CampaignRegistry.read(temporary, repository_root=root)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    if registry.production_operations is not None:
        raise AcquisitionRegistryContractError("parsed registry activated production")


def validate_registry_for_activation(registry: CampaignRegistry) -> None:
    """Strictly revalidate a canonical registry immediately before activation.

    ``CampaignRegistry.read`` intentionally remains a permissive parser for
    legacy/test registries.  A registry that claims the canonical ID or
    scientific-contract schema must, however, retain its source file and pass
    the independent canonical derivation before any production factory or
    cycle can be entered.
    """

    lane_ids = tuple(lane.lane_id for lane in registry.lanes)
    canonical_shape = (
        lane_ids == CANONICAL_LANE_IDS
        or bool(registry.scientific_contracts)
        or registry.supervisor_state_root == CANONICAL_SUPERVISOR_STATE_ROOT
    )
    claims_contract = (
        registry.registry_id == CANONICAL_REGISTRY_ID
        or registry.scientific_contracts.get("schema")
        == REGISTRY_CONTRACT_SCHEMA_V1
        or canonical_shape
    )
    if not claims_contract:
        return
    if (
        registry.registry_id != CANONICAL_REGISTRY_ID
        or registry.scientific_contracts.get("schema")
        != REGISTRY_CONTRACT_SCHEMA_V1
    ):
        raise AcquisitionRegistryContractError(
            "canonical-shaped registry was declassified before activation"
        )
    source_path = registry.registry_path
    if source_path is None:
        raise AcquisitionRegistryContractError(
            "canonical activation requires a source registry artifact"
        )
    payload = _read_object(source_path, label="activation acquisition registry")
    validate_registry_payload(payload, registry.repository_root)
    reparsed = CampaignRegistry.read(
        source_path, repository_root=registry.repository_root
    )
    if reparsed.digest != registry.digest:
        raise AcquisitionRegistryContractError(
            "activation registry object differs from its verified source artifact"
        )


def write_or_check_registry(
    *,
    repository_root: str | Path,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    check_only: bool = False,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    target = Path(registry_path)
    if not target.is_absolute():
        target = root / target
    expected = build_registry_payload(root)
    validate_registry_payload(expected, root)
    content = _canonical_bytes(expected)
    if check_only:
        if not target.is_file() or target.read_bytes() != content:
            raise AcquisitionRegistryContractError(
                "acquisition registry is not the canonical generated payload"
            )
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            temporary = ""
        finally:
            if temporary:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
    return {
        "path": target.resolve().relative_to(root).as_posix(),
        "file_sha256": hashlib.sha256(content).hexdigest(),
        "registry_digest": CampaignRegistry.read(
            target, repository_root=root
        ).digest,
        "physical_cell_count": 39,
        "seed_count": 8,
        "reference_count": 39,
        "production_enabled": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--check", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = write_or_check_registry(
        repository_root=args.repo_root,
        registry_path=args.registry,
        check_only=args.check,
    )
    print(json.dumps({"ok": True, **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AcquisitionRegistryContractError",
    "CANONICAL_LANE_IDS",
    "CANONICAL_REGISTRY_ID",
    "DEFAULT_REGISTRY_PATH",
    "FUTURE_REQUIRED_BLOCKERS",
    "PRODUCTION_REQUIRED_BLOCKERS",
    "REGISTRY_CONTRACT_SCHEMA_V1",
    "build_registry_payload",
    "build_scientific_contracts",
    "validate_registry_payload",
    "validate_registry_for_activation",
    "write_or_check_registry",
]
