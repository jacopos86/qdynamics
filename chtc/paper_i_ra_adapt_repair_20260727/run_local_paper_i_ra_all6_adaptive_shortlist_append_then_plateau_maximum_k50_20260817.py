#!/usr/bin/env python3
"""Capacity-audited maximum-k50 all-six Paper-I RA campaign.

Cells close at either fifty accepted controller rounds or the authenticated
natural Phase-III terminal of the opt-in V2 route.  Historical exact-k50
campaigns and their evidence are intentionally outside this namespace.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from dataclasses import asdict
import io
import importlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import psutil


_BOOTSTRAP_RUNNER_PATH = Path(__file__).resolve()
_BOOTSTRAP_REPO_ROOT = _BOOTSTRAP_RUNNER_PATH.parents[2]
if str(_BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_REPO_ROOT))

from chtc.paper_i_ra_adapt_repair_20260727 import (
    paper_i_ra_serial_campaign_runtime_20260816 as serial_runtime,
)
from chtc.paper_i_ra_adapt_repair_20260727 import (
    paper_i_matched_singleton12_archive_20260815 as strict_archive,
)
from chtc.paper_i_ra_adapt_repair_20260727 import (
    paper_i_ra_all6_maximum_k50_reporting_20260817 as ragged_reporting,
)
from chtc.paper_i_ra_adapt_repair_20260727 import (
    paper_i_ra_all6_maximum_k50_archive_20260817 as maximum_archive,
)
from pipelines.static_adapt.ra_adapt.contracts import RAAdaptOperationalControls
from pipelines.static_adapt.ra_adapt.contracts import canonical_sha256
from pipelines.static_adapt.ra_adapt.adaptive_phase_shortlist import (
    adaptive_phase_record_id,
    adaptive_phase_selection_receipt_from_mapping,
)
from pipelines.static_adapt.ra_adapt.engine import run_ra_adapt
from pipelines.static_adapt.ra_adapt.replay_evidence import (
    validate_controller_replay_evidence,
)
from pipelines.static_adapt.ra_adapt.semantic_closure_routes import (
    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
    build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request,
    build_paper_i_ra_hh_regime_problem,
    materialize_paper_i_ra_semantic_protocol,
    validate_semantic_phase3_natural_terminal_route_contract,
    validate_semantic_position_phase0_receipt,
)
from pipelines.static_adapt.adaptive_phase_contracts import (
    ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1,
)
from pipelines.static_adapt.sr_snake._resume import (
    CanonicalResumeError,
    load_canonical_accepted_state_resume,
)
from pipelines.static_adapt.sr_snake.contracts import (
    AcceptedStateResume,
    CheckpointObservation,
    EstimatorLedgerObservation,
    FreshStart,
    SRObservationPolicy,
)


RUNNER_PATH = Path(__file__).resolve()
REPO_ROOT = RUNNER_PATH.parents[2]
CAMPAIGN_ID = (
    "paper_i_ra_all6_adaptive_shortlist_append_then_plateau_maximum_k50_"
    "20260817_v1"
)
TARGET_HORIZON = 50

AUTHORITY_DIR = RUNNER_PATH.parent / f"{CAMPAIGN_ID}_authority"
PLAN_PATH = AUTHORITY_DIR / "plan.json"
AUTHORIZATION_PATH = AUTHORITY_DIR / "authorization.json"
RUNTIME_ROOT = REPO_ROOT / "output/local_runs" / CAMPAIGN_ID
RUNS_ROOT = RUNTIME_ROOT / "runs"
STAGING_ROOT = RUNTIME_ROOT / "in_progress"
RECEIPTS_ROOT = RUNTIME_ROOT / "worker_receipts"
GUARD_ROOT = RUNTIME_ROOT / "guard_receipts"
CELL_LOG_ROOT = RUNTIME_ROOT / "cell_logs"
COMPACT_ROOT = RUNTIME_ROOT / "compact_cell_receipts"
ARCHIVED_RECEIPTS_ROOT = RUNTIME_ROOT / "archived_cell_receipts"
STATUS_PATH = RUNTIME_ROOT / "status.json"
LOCK_PATH = RUNTIME_ROOT / "campaign.lock"
SCHEDULER_ROOT = RUNTIME_ROOT / "scheduler_receipts"
BATCH_RECEIPTS_ROOT = RUNTIME_ROOT / "batch_receipts"
INITIAL_CAPACITY_PATH = RUNTIME_ROOT / "initial_campaign_capacity_receipt.json"
REPORT_JSON = RUNTIME_ROOT / "comparison.json"
REPORT_CSV = RUNTIME_ROOT / "accepted_rows.csv"
REPORT_TERMINAL_CSV = RUNTIME_ROOT / "terminal_attempts.csv"
REPORT_OUTCOMES_CSV = RUNTIME_ROOT / "cell_outcomes.csv"
REPORT_SHARED_PREFIX_CSV = RUNTIME_ROOT / "shared_prefix_pairs.csv"
REPORT_ENDPOINTS_CSV = RUNTIME_ROOT / "regime_endpoint_pairs.csv"
REPORT_MD = RUNTIME_ROOT / "comparison.md"
TERMINAL_PATH = RUNTIME_ROOT / "terminal_matrix_receipt.json"

SHORTLIST_MAXIMA = {"phase_0": 24, "phase_i": 24, "phase_ii": 12, "phase_iii": 12}
PHASE_FRONTIER_RATIOS = {"phase_i": 0.9, "phase_ii": 0.9, "phase_iii": 0.9}
LAUNCH_AVAILABLE_MEMORY_BYTES = 5 * 1024**3
LAUNCH_FREE_DISK_BYTES = 10 * 1024**3
MAXIMUM_CONCURRENCY = 2
SERIAL_CAPACITY_FALLBACK_AUTHORIZED = True
CAPACITY_WAIT_SECONDS = 5 * 60
CHILD_RSS_LIMIT_BYTES = 8 * 1024**3
NPH7_PLATEAU_CHILD_RSS_LIMIT_BYTES = 10 * 1024**3
AVAILABLE_MEMORY_FLOOR_BYTES = 2 * 1024**3
FREE_DISK_FLOOR_BYTES = 2 * 1024**3
POLL_SECONDS = 1.0
CAPACITY_POLL_SECONDS = 10.0
CHILD_TOKEN_ENV = "PAPER_I_RA_ALL6_ADAPTIVE_OVERNIGHT_CHILD_TOKEN"

PLAN_SCHEMA = "paper_i_ra_all6_adaptive_maximum_k50_plan_v1"
AUTH_SCHEMA = "paper_i_ra_all6_adaptive_maximum_k50_authorization_v1"
MANIFEST_SCHEMA = "paper_i_ra_all6_adaptive_maximum_k50_manifest_v1"
WORKER_SCHEMA = "paper_i_ra_all6_adaptive_maximum_k50_worker_v1"
GUARD_SCHEMA = "paper_i_ra_all6_adaptive_maximum_k50_guard_v1"
REPORT_SCHEMA = "paper_i_ra_all6_adaptive_maximum_k50_comparison_v1"
TERMINAL_SCHEMA = "paper_i_ra_all6_adaptive_maximum_k50_terminal_matrix_v1"
STATUS_SCHEMA = "paper_i_ra_all6_adaptive_maximum_k50_status_v1"
SCHEDULER_SCHEMA = "paper_i_ra_all6_adaptive_maximum_k50_scheduler_v1"
BATCH_RECEIPT_SCHEMA = "paper_i_ra_all6_adaptive_maximum_k50_batch_v1"
INITIAL_CAPACITY_SCHEMA = "paper_i_ra_all6_adaptive_maximum_k50_capacity_v1"
COMPACT_CELL_SCHEMA = "paper_i_ra_all6_adaptive_maximum_k50_compact_cell_v1"
ARCHIVED_CELL_SCHEMA = "paper_i_ra_all6_adaptive_maximum_k50_archived_cell_v1"
CELL_COMPLETION_SCHEMA = (
    "paper_i_ra_all6_adaptive_maximum_k50_cell_completion_v1"
)
MAXIMUM_COMPLETION_KIND = "reached_maximum_controller_rounds_v1"
NATURAL_COMPLETION_KIND = (
    "authenticated_phase3_no_positive_natural_terminal_v1"
)
NATURAL_STOP_REASON = "phase_iii_no_positive_feasible_candidate"

REPORT_ROW_FIELDS = ragged_reporting.ACCEPTED_ROW_FIELDS

EXPECTED_ENV = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
    "CUDA_VISIBLE_DEVICES": "",
    "STATIC_ADAPT_HH_POOL_CACHE": "off",
    "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
    "STATIC_ADAPT_ALLOCATED_CPUS": "1",
    "QISKIT_NUM_PROCS": "1",
    "QISKIT_PARALLEL": "FALSE",
    "RAYON_NUM_THREADS": "1",
}

CORE_ROUTE_VARIANT_PLACEHOLDER = "__unresolved_maximum_k50_core_route__"
CORE_ROUTE_CONSTANT = (
    "PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2"
)
EXPECTED_CORE_ROUTE_VARIANT = (
    "position_records_gradient_only_adaptive_shortlist_phase123_adaptive_"
    "natural_terminal_v2"
)
CORE_REGIME_CONSTANT = "PAPER_I_RA_CANONICAL_REGIME_IDS"
CORE_PROBLEM_BUILDER = "build_paper_i_ra_hh_regime_problem"
CORE_REQUEST_BUILDER = (
    "build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request"
)

# Exact byte anchors are copied from authenticated local guard receipts.  When
# a matching full-horizon route is absent, the table deliberately substitutes
# the largest observed peak in the same nph class.  The decision receipt keeps
# both the selected anchor and its provenance visible.
EMPIRICAL_RSS_EVIDENCE: Mapping[str, Mapping[str, Any]] = {
    "nph3_append_weak_weak": {
        "peak_rss_bytes": 3_523_837_952,
        "source": (
            "output/local_runs/paper_i_page12_weak_holstein_ra6_priority_"
            "20260815_v1/guard_receipts/global_singleton_gradient_phase0_"
            "phase23_qiskit_no_lanes__weak_weak__nph3__ra_global_singleton_"
            "gradient_phase0_phase123_qiskit_phase23_append_only.json"
        ),
        "source_file_sha256": (
            "d0d24b6615267734253e1c0855ffa8cfe56592f19cfa7a3cdef279acf3929c96"
        ),
    },
    "nph3_append_intermediate_weak": {
        "peak_rss_bytes": 2_186_018_816,
        "source": (
            "output/local_runs/paper_i_page12_weak_holstein_ra6_priority_"
            "20260815_v1/guard_receipts/global_singleton_gradient_phase0_"
            "phase23_qiskit_no_lanes__intermediate_weak__nph3__ra_global_"
            "singleton_gradient_phase0_phase123_qiskit_phase23_append_only.json"
        ),
        "source_file_sha256": (
            "f408ce3b2957a523294922073bfa1f4c1cd812e4ed650cceb2175c5a9e1aac91"
        ),
    },
    "nph3_always_strong_weak_u8_k15": {
        "peak_rss_bytes": 1_944_420_352,
        "source": (
            "output/local_runs/paper_i_position_aware_phase0_sw_always_k15_"
            "20260816_v1/guard_receipts/position_aware_phase0__strong_weak_"
            "u8__nph3__ra_always_commutation_reduced__k15.json"
        ),
        "source_file_sha256": (
            "638d6198da7792ffe30db5de2a62f71b688615ed979468fd0f404508ecfe8fe8"
        ),
    },
    "nph7_append_intermediate_strong": {
        "peak_rss_bytes": 6_474_907_648,
        "source": (
            "output/local_runs/paper_i_page12_strong_holstein_sector5_local_"
            "repair_20260814_v1/guard_receipts/global_singleton_gradient_"
            "phase0_phase23_qiskit_no_lanes__intermediate_strong__nph7__ra_"
            "global_singleton_gradient_phase0_phase123_qiskit_phase23_"
            "append_only.json"
        ),
        "source_file_sha256": (
            "52c19453a4598cceb245c5d77977bb79160144a0c00219c13a91f107043af990"
        ),
    },
    "nph7_append_strong_strong_u8": {
        "peak_rss_bytes": 6_762_299_392,
        "source": (
            "output/local_runs/paper_i_page12_strong_holstein_sector5_local_"
            "repair_20260814_v1/guard_receipts/global_singleton_gradient_"
            "phase0_phase23_qiskit_no_lanes__strong_strong_u8__nph7__ra_"
            "global_singleton_gradient_phase0_phase123_qiskit_phase23_"
            "append_only.json"
        ),
        "source_file_sha256": (
            "c8619341247237baff4761312493637cffd6451c01c26414000f92c993c4e76f"
        ),
    },
    "nph7_plateau_weak_strong": {
        "peak_rss_bytes": 9_564_340_224,
        "source": (
            "output/local_runs/paper_i_page12_strong_holstein_sector5_local_"
            "repair_20260814_v1/guard_receipts/global_singleton_gradient_"
            "phase0_phase23_qiskit_no_lanes__weak_strong__nph7__ra_global_"
            "singleton_gradient_phase0_phase123_qiskit_phase23_always_"
            "commutation_reduced.json"
        ),
        "source_file_sha256": (
            "80d40e6b7be41d1db4b758da83226fed1ffa6957eba1fff46508d2fea3db3039"
        ),
    },
    "nph7_plateau_intermediate_strong": {
        "peak_rss_bytes": 9_270_280_192,
        "source": (
            "output/local_runs/paper_i_page12_strong_holstein_sector5_local_"
            "repair_20260814_v1/guard_receipts/global_singleton_gradient_"
            "phase0_phase23_qiskit_no_lanes__intermediate_strong__nph7__ra_"
            "global_singleton_gradient_phase0_phase123_qiskit_phase23_"
            "always_commutation_reduced.json"
        ),
        "source_file_sha256": (
            "77bebd919bd988f5a0860a0aa8fd6d73041e022fb84299dd30fd90f141fea667"
        ),
    },
    "nph7_plateau_strong_strong_u8": {
        "peak_rss_bytes": 9_346_334_720,
        "source": (
            "output/local_runs/paper_i_page12_strong_holstein_sector5_local_"
            "repair_20260814_v1/guard_receipts/global_singleton_gradient_"
            "phase0_phase23_qiskit_no_lanes__strong_strong_u8__nph7__ra_"
            "global_singleton_gradient_phase0_phase123_qiskit_phase23_"
            "always_commutation_reduced.json"
        ),
        "source_file_sha256": (
            "4eb0dbc98a4ad0b51499a749edb65adc339706d8d4cd0fc99e14816cfb2dc5bb"
        ),
    },
}

HOST_PHYSICAL_MEMORY_EVIDENCE = {
    "physical_memory_bytes": 17_179_869_184,
    "source": (
        "chtc/paper_i_ra_adapt_repair_20260727/"
        "macro_gradient_phase0_macro_phase123_proxy_no_lanes_local_20260810_"
        "v1/activation/host_preflight.json"
    ),
    "source_file_sha256": (
        "4a2e9aea79207c67b5c37087945e908a149f446106b9b9c46e567d56844b857a"
    ),
}


class RunnerError(RuntimeError):
    """Raised when the overnight campaign cannot preserve its contract."""


class ArchiveCapacityBlocked(RunnerError):
    """Raised when a validated cell cannot yet be archived safely."""

    def __init__(self, snapshot: Mapping[str, Any]):
        super().__init__("Per-cell archive capacity is unavailable.")
        self.snapshot = dict(snapshot)


class BatchExecutionFailed(RunnerError):
    """Raised after guarded child failure with an exact scheduler-mode status."""

    def __init__(self, message: str, *, scheduling_mode: str):
        super().__init__(message)
        self.failure_status = (
            "failed_pair" if scheduling_mode == "pair" else "failed_campaign"
        )


@dataclass(frozen=True, slots=True)
class CellSpec:
    ordinal: int
    block: str
    regime_id: str
    nph: int
    insertion_policy: str
    horizon: int
    route_variant: str
    execution_id: str


@dataclass(frozen=True, slots=True)
class BatchSpec:
    ordinal: int
    block: str
    execution_ids: tuple[str, str]


@dataclass(frozen=True, slots=True)
class ArchivedCellClosure:
    cell: CellSpec
    rows: tuple[Mapping[str, Any], ...]
    worker_receipt_sha256: str
    guard_receipt_sha256: str
    compact_receipt_sha256: str
    archive_backed_closure_sha256: str
    archive_closure_receipt_sha256: str
    archived_cell_receipt_sha256: str


PLAN_RECEIPT_KEYS = frozenset(
    {
        "schema", "created_at", "campaign_id", "run_class",
        "maximum_controller_rounds", "allowed_cell_completions",
        "phase3_no_positive_policy", "controller_horizon_policy",
        "block_order", "canonical_cell_order", "deterministic_launch_order",
        "append_block_execution_ids", "plateau_block_execution_ids",
        "deterministic_batches", "cells", "protocol_bindings",
        "source_implementation_inventory_sha256",
        "source_implementation_file_count", "runner",
        "runner_runtime_dependencies", "optimizer", "seeds", "frontier_ratios",
        "shortlist_maxima", "maximum_concurrency",
        "serial_capacity_fallback_authorized",
        "silent_serial_fallback_authorized",
        "append_block_must_close_before_plateau", "execution_path_canary",
        "capacity", "per_cell_storage_lifecycle", "runtime_environment",
        "execution_authorized", "archive_rotation_authorized",
        "submission_authorized", "paper_adoption_authorized",
        "paper_evidence_adoption_authorized", "sha256",
    }
)
AUTH_RECEIPT_KEYS = frozenset(
    {
        "schema", "created_at", "campaign_id", "authorization_basis",
        "plan_sha256", "runner_sha256",
        "source_implementation_inventory_sha256", "execution_ids",
        "execution_path_canary", "execution_authorized",
        "archive_rotation_authorized", "submission_authorized",
        "paper_adoption_authorized", "paper_evidence_adoption_authorized",
        "sha256",
    }
)
SCHEDULER_RECEIPT_KEYS = frozenset(
    {
        "schema", "created_at", "campaign_id", "batch", "execution_ids",
        "plan_sha256", "authorization_sha256",
        "source_implementation_inventory_sha256", "scheduling_mode",
        "maximum_concurrency", "capacity_contract", "capacity_observation",
        "capacity_observation_sha256", "serial_capacity_fallback_audited",
        "silent_serial_fallback_authorized", "submission_authorized",
        "paper_adoption_authorized", "paper_evidence_adoption_authorized",
        "sha256",
    }
)
WORKER_RECEIPT_KEYS = frozenset(
    {
        "schema", "status", "campaign_id", "execution_id",
        "manifest_sha256", "artifact_inventory", "submission_authorized",
        "paper_adoption_authorized", "paper_evidence_adoption_authorized",
        "sha256",
    }
)
GUARD_RECEIPT_KEYS = frozenset(
    {
        "schema", "status", "campaign_id", "execution_id", "batch_ordinal",
        "scheduling_mode", "scheduler_decision_sha256",
        "launch_capacity_observation", "launch_capacity_observation_sha256",
        "returncode", "stop_reason", "elapsed_seconds", "peak_rss_bytes",
        "rss_limit_bytes", "minimum_available_memory_bytes",
        "minimum_free_disk_bytes", "worker_receipt_sha256",
        "log_file_binding", "attempt_inventory", "submission_authorized",
        "paper_adoption_authorized", "paper_evidence_adoption_authorized",
        "sha256",
    }
)
INITIAL_CAPACITY_RECEIPT_KEYS = frozenset(
    {
        "schema", "status", "created_at", "campaign_id", "plan_sha256",
        "authorization_sha256", "source_implementation_inventory_sha256",
        "campaign_capacity_floor", "bounded_wait_observation",
        "strict_capacity_observation",
        "one_time_gate_not_reimposed_on_restart", "submission_authorized",
        "paper_adoption_authorized", "paper_evidence_adoption_authorized",
        "sha256",
    }
)
COMPACT_RECEIPT_KEYS = frozenset(
    {
        "schema", "status", "created_at", "campaign_id", "execution_id",
        "cell", "target_horizon", "plan_sha256", "authorization_sha256",
        "source_implementation_inventory_sha256", "protocol_binding",
        "manifest_sha256", "manifest_file_binding", "worker_receipt_sha256",
        "guard_receipt_sha256", "scheduler_decision_sha256",
        "log_file_binding", "artifact_bindings", "rows", "rows_sha256",
        "submission_authorized", "paper_adoption_authorized",
        "paper_evidence_adoption_authorized", "sha256",
    }
)


_REGIMES = (
    ("weak_weak", 3),
    ("intermediate_weak", 3),
    ("strong_weak_u8", 3),
    ("weak_strong", 7),
    ("intermediate_strong", 7),
    ("strong_strong_u8", 7),
)


def _load_core_module() -> Any:
    return importlib.import_module(
        "pipelines.static_adapt.ra_adapt.semantic_closure_routes"
    )


def _route_variant_if_available() -> str:
    value = getattr(_load_core_module(), CORE_ROUTE_CONSTANT, None)
    return str(value) if isinstance(value, str) and value else CORE_ROUTE_VARIANT_PLACEHOLDER


def _cells() -> tuple[CellSpec, ...]:
    rows: list[CellSpec] = []
    route_variant = _route_variant_if_available()
    for block, insertion in (
        ("append", "append_only"),
        ("plateau", "plateau_commutation"),
    ):
        for regime_id, nph in _REGIMES:
            ordinal = len(rows) + 1
            rows.append(
                CellSpec(
                    ordinal=ordinal,
                    block=block,
                    regime_id=regime_id,
                    nph=nph,
                    insertion_policy=insertion,
                    horizon=TARGET_HORIZON,
                    route_variant=route_variant,
                    execution_id=(
                        "all_phase_adaptive_natural_terminal__"
                        f"{regime_id}__nph{nph}__{insertion}__maximum_k50"
                    ),
                )
            )
    return tuple(rows)


CELL_SPECS = _cells()


def _batches() -> tuple[BatchSpec, ...]:
    # Preserve the handoff's exact canonical within-block order.  Capacity is
    # handled by the audited pair/serial decision, never by reordering cells.
    regime_pairs = (
        ("weak_weak", "intermediate_weak"),
        ("strong_weak_u8", "weak_strong"),
        ("intermediate_strong", "strong_strong_u8"),
    )
    batches: list[BatchSpec] = []
    for block in ("append", "plateau"):
        for regime_pair in regime_pairs:
            cells = tuple(
                next(
                    cell
                    for cell in CELL_SPECS
                    if cell.block == block and cell.regime_id == regime_id
                )
                for regime_id in regime_pair
            )
            batches.append(
                BatchSpec(
                    ordinal=len(batches) + 1,
                    block=block,
                    execution_ids=(cells[0].execution_id, cells[1].execution_id),
                )
            )
    return tuple(batches)


BATCH_SPECS = _batches()


def _batch_payload(batch: BatchSpec) -> dict[str, Any]:
    return {
        "ordinal": batch.ordinal,
        "block": batch.block,
        "execution_ids": list(batch.execution_ids),
    }


def core_interface_available() -> bool:
    module = _load_core_module()
    return all(
        hasattr(module, name)
        for name in (
            CORE_ROUTE_CONSTANT,
            CORE_REGIME_CONSTANT,
            CORE_PROBLEM_BUILDER,
            CORE_REQUEST_BUILDER,
        )
    ) and (
        str(getattr(module, CORE_ROUTE_CONSTANT)) == EXPECTED_CORE_ROUTE_VARIANT
        and tuple(getattr(module, CORE_REGIME_CONSTANT))
        == tuple(regime_id for regime_id, _nph in _REGIMES)
    )


digested = serial_runtime.digested
canonical_sha256 = serial_runtime.canonical_sha256
sha256_file = serial_runtime.sha256_file
write_json_exclusive = serial_runtime.write_json_exclusive
write_text_exclusive = serial_runtime.write_text_exclusive


def load_digested(path: Path, *, schema: str) -> dict[str, Any]:
    return serial_runtime.load_digested(
        path,
        schema=schema,
        error_type=RunnerError,
    )


def _require_exact_keys(
    payload: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    if set(payload) != expected:
        raise RunnerError(f"{label} receipt shape drifted.")


def _valid_created_at(payload: Mapping[str, Any]) -> bool:
    return isinstance(payload.get("created_at"), str) and bool(payload["created_at"])


def file_binding(path: Path) -> dict[str, Any]:
    return serial_runtime.file_binding(path)


def _runtime_dependencies() -> list[dict[str, Any]]:
    return [
        serial_runtime.file_binding(Path(serial_runtime.__file__).resolve()),
        serial_runtime.file_binding(Path(strict_archive.__file__).resolve()),
        serial_runtime.file_binding(Path(ragged_reporting.__file__).resolve()),
        serial_runtime.file_binding(Path(maximum_archive.__file__).resolve()),
    ]


def _storage_lifecycle_contract() -> dict[str, Any]:
    limits = strict_archive.campaign_default_archive_limits()
    return {
        "policy": "validate_compact_strict_archive_rotate_before_next_cell_v1",
        "archive_limits": limits.as_dict(),
        "archive_start_free_floor_bytes": limits.archive_start_free_floor_bytes,
        "preserve_compact_reporting_receipt": True,
        "preserve_worker_and_guard_receipts": True,
        "preserve_summary_and_ledger_in_authenticated_archive": True,
        "direct_run_tree_absent_before_next_cell": True,
    }


def _protocol_binding(cell: CellSpec) -> dict[str, Any]:
    if not core_interface_available():
        raise RunnerError(
            "The versioned all-phase adaptive core interface is absent; "
            "protocol materialization is forbidden."
        )
    module = _load_core_module()
    route_variant = str(getattr(module, CORE_ROUTE_CONSTANT))
    if cell.route_variant != route_variant:
        raise RunnerError("Overnight cell route identity is stale.")
    problem = getattr(module, CORE_PROBLEM_BUILDER)(cell.regime_id)
    request = getattr(module, CORE_REQUEST_BUILDER)(
        insertion_policy=cell.insertion_policy,
        maximum_controller_rounds=cell.horizon,
    )
    protocol = module.materialize_paper_i_ra_semantic_protocol(problem, request)
    identity = module.semantic_closure_route_identity(route_variant)
    inventory = module.semantic_closure_source_implementation_inventory()
    if protocol.route_contract is None or protocol.bundle_materialization is None:
        raise RunnerError("Overnight semantic materialization is incomplete.")
    native = protocol.route_contract["native_semantic_contract"]
    execution = protocol.route_contract["execution_settings"]
    invariants = protocol.route_contract["semantic_invariants"]
    expected_phase0 = {
        "population": "current_commutation_reduced_candidate_position_records_v1",
        "benefit": "absolute_position_record_gradient_v1",
        "fubini_study_metric": "off",
        "qiskit_compile": "off",
        "graph_proxy_cost": "off",
        "score": "absolute_position_record_gradient_v1",
        "shortlist": "phase0_active_score_effective_competition_shortlist_v2",
        "adaptive_shadow_receipt": False,
        "placement_activation": (
            "append_record_when_closed_full_commutation_reduced_records_when_open_v1"
        ),
        "generator_level_reexpansion_after_phase0": False,
    }
    if (
        route_variant != EXPECTED_CORE_ROUTE_VARIANT
        or protocol.horizon != TARGET_HORIZON
        or protocol.optimizer != "powell"
        or protocol.optimizer_maxiter != 200
        or protocol.seeds != {"adapt": 7, "transpiler": 7}
        or execution.get("adapt_inner_optimizer") != "POWELL"
        or execution.get("adapt_maxiter") != 200
        or execution.get("adapt_scipy_maxfev") != 0
        or native.get("optimizer_options")
        != {"xtol": 1.0e-4, "ftol": 1.0e-8, "maxfev": None}
        or native.get("phase_shortlist_maxima")
        != {"phase_i": 24, "phase_ii": 12, "phase_iii": 12}
        or native.get("phase_frontier_ratios")
        != {"phase_i": 0.9, "phase_ii": 0.9, "phase_iii": 0.9}
        or native.get("phase_frontier_ratio_role") != "eligibility_only"
        or native.get("phase0_policy") != expected_phase0
        or native.get("qiskit_active_phases")
        != ["phase_i", "phase_ii", "phase_iii"]
        or native.get("phase123_shortlist_policy")
        != "phase123_active_score_inverse_simpson_adaptive_shortlist_v1"
        or invariants.get("phase0_fubini_metric_active") is not False
        or invariants.get("phase0_compile_cost_active") is not False
        or invariants.get("phase0_structural_proxy_cost_active") is not False
        or invariants.get("phase0_resource_cost_active") is not False
        or invariants.get("phase0_estimator_components") != ["N_grad"]
        or native.get("phase3_no_positive_policy")
        != "typed_natural_terminal_v1"
        or native.get("controller_horizon_policy")
        != "maximum_accepted_controller_rounds_v1"
        or execution.get("ra_phase3_no_positive_policy")
        != "typed_natural_terminal_v1"
        or execution.get("ra_controller_horizon_policy")
        != "maximum_accepted_controller_rounds_v1"
        or invariants.get("phase3_no_positive_policy")
        != "typed_natural_terminal_v1"
        or invariants.get("controller_horizon_policy")
        != "maximum_accepted_controller_rounds_v1"
        or execution.get("ra_phase123_shortlist_policy")
        != "phase123_active_score_inverse_simpson_adaptive_shortlist_v1"
        or protocol.execution_authorized is not False
        or protocol.source_locks.get("implementation_source_inventory_sha256")
        != inventory["sha256"]
    ):
        raise RunnerError("Overnight physics or adaptive shortlist contract drifted.")
    return {
        "execution_id": cell.execution_id,
        "route_variant": route_variant,
        "algorithm_id": identity.algorithm_id,
        "route_id": identity.route_id,
        "semantic_implementation_version": identity.semantic_implementation_version,
        "route_contract_sha256": protocol.route_contract["sha256"],
        "protocol_sha256": protocol.sha256,
        "bundle_id": protocol.bundle_id,
        "bundle_manifest_sha256": protocol.bundle_manifest_sha256,
        "materialization_receipt_sha256": protocol.bundle_materialization.sha256,
        "source_locks": dict(protocol.source_locks),
        "execution_authorized_in_serialized_protocol": protocol.execution_authorized,
    }


def build_plan() -> dict[str, Any]:
    if not core_interface_available():
        raise RunnerError(
            "The versioned all-phase adaptive core interface is absent; "
            "no plan or execution authority may be created."
        )
    module = _load_core_module()
    inventory = module.semantic_closure_source_implementation_inventory()
    protocols = [_protocol_binding(cell) for cell in CELL_SPECS]
    inventory_hashes = {
        row["source_locks"]["implementation_source_inventory_sha256"]
        for row in protocols
    }
    if inventory_hashes != {inventory["sha256"]}:
        raise RunnerError("The 12 cells do not share one source inventory.")
    return digested(
        {
            "schema": PLAN_SCHEMA,
            "created_at": serial_runtime.utc_now(),
            "campaign_id": CAMPAIGN_ID,
            "run_class": "local_diagnostic_non_adopted",
            "maximum_controller_rounds": TARGET_HORIZON,
            "allowed_cell_completions": [
                MAXIMUM_COMPLETION_KIND,
                NATURAL_COMPLETION_KIND,
            ],
            "phase3_no_positive_policy": "typed_natural_terminal_v1",
            "controller_horizon_policy": (
                "maximum_accepted_controller_rounds_v1"
            ),
            "block_order": ["append", "plateau"],
            "canonical_cell_order": [cell.execution_id for cell in CELL_SPECS],
            "deterministic_launch_order": [
                execution_id
                for batch in BATCH_SPECS
                for execution_id in batch.execution_ids
            ],
            "append_block_execution_ids": [
                cell.execution_id for cell in CELL_SPECS[:6]
            ],
            "plateau_block_execution_ids": [
                cell.execution_id for cell in CELL_SPECS[6:]
            ],
            "deterministic_batches": [
                _batch_payload(batch) for batch in BATCH_SPECS
            ],
            "cells": [asdict(cell) for cell in CELL_SPECS],
            "protocol_bindings": protocols,
            "source_implementation_inventory_sha256": inventory["sha256"],
            "source_implementation_file_count": inventory["source_count"],
            "runner": file_binding(RUNNER_PATH),
            "runner_runtime_dependencies": _runtime_dependencies(),
            "optimizer": {
                "name": "powell",
                "xtol": 1.0e-4,
                "ftol": 1.0e-8,
                "maxiter": 200,
                "maxfev": None,
            },
            "seeds": {"adapt": 7, "transpiler": 7},
            "frontier_ratios": dict(PHASE_FRONTIER_RATIOS),
            "shortlist_maxima": dict(SHORTLIST_MAXIMA),
            "maximum_concurrency": MAXIMUM_CONCURRENCY,
            "serial_capacity_fallback_authorized": (
                SERIAL_CAPACITY_FALLBACK_AUTHORIZED
            ),
            "silent_serial_fallback_authorized": False,
            "append_block_must_close_before_plateau": True,
            "execution_path_canary": {
                "execution_id": CELL_SPECS[0].execution_id,
                "completion_witness": (
                    "accepted_round_1_or_authenticated_round_zero_terminal_v1"
                ),
                "continues_same_trajectory_to_completion": True,
                "separate_scientific_trajectory": False,
            },
            "capacity": {
                "maximum_wait_seconds": CAPACITY_WAIT_SECONDS,
                "launch_available_memory_bytes": LAUNCH_AVAILABLE_MEMORY_BYTES,
                "launch_free_disk_bytes": LAUNCH_FREE_DISK_BYTES,
                "child_rss_limit_bytes": CHILD_RSS_LIMIT_BYTES,
                "runtime_available_memory_floor_bytes": AVAILABLE_MEMORY_FLOOR_BYTES,
                "runtime_free_disk_floor_bytes": FREE_DISK_FLOOR_BYTES,
                "nph7_plateau_child_rss_limit_bytes": (
                    NPH7_PLATEAU_CHILD_RSS_LIMIT_BYTES
                ),
                "host_physical_memory_evidence": dict(
                    HOST_PHYSICAL_MEMORY_EVIDENCE
                ),
                "pair_launch_capacity_contracts": [
                    pair_launch_capacity_contract(batch) for batch in BATCH_SPECS
                ],
            },
            "per_cell_storage_lifecycle": _storage_lifecycle_contract(),
            "runtime_environment": dict(EXPECTED_ENV),
            "execution_authorized": False,
            "archive_rotation_authorized": False,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )


def prepare_plan() -> dict[str, Any]:
    plan = build_plan()
    serial_runtime.prepare_authority_directory(
        AUTHORITY_DIR,
        files={"plan.json": plan},
        error_type=RunnerError,
    )
    return plan


def validate_plan(*, recompute_protocols: bool) -> dict[str, Any]:
    plan = load_digested(PLAN_PATH, schema=PLAN_SCHEMA)
    _require_exact_keys(plan, PLAN_RECEIPT_KEYS, label="Plan")
    if not core_interface_available():
        raise RunnerError("The all-phase adaptive core interface disappeared.")
    module = _load_core_module()
    inventory = module.semantic_closure_source_implementation_inventory()
    expected_cells = _cells()
    expected_capacity = {
        "maximum_wait_seconds": CAPACITY_WAIT_SECONDS,
        "launch_available_memory_bytes": LAUNCH_AVAILABLE_MEMORY_BYTES,
        "launch_free_disk_bytes": LAUNCH_FREE_DISK_BYTES,
        "child_rss_limit_bytes": CHILD_RSS_LIMIT_BYTES,
        "runtime_available_memory_floor_bytes": AVAILABLE_MEMORY_FLOOR_BYTES,
        "runtime_free_disk_floor_bytes": FREE_DISK_FLOOR_BYTES,
        "nph7_plateau_child_rss_limit_bytes": NPH7_PLATEAU_CHILD_RSS_LIMIT_BYTES,
        "host_physical_memory_evidence": dict(HOST_PHYSICAL_MEMORY_EVIDENCE),
        "pair_launch_capacity_contracts": [
            pair_launch_capacity_contract(batch) for batch in BATCH_SPECS
        ],
    }
    if (
        expected_cells != CELL_SPECS
        or not _valid_created_at(plan)
        or plan.get("campaign_id") != CAMPAIGN_ID
        or plan.get("run_class") != "local_diagnostic_non_adopted"
        or plan.get("maximum_controller_rounds") != TARGET_HORIZON
        or plan.get("allowed_cell_completions")
        != [MAXIMUM_COMPLETION_KIND, NATURAL_COMPLETION_KIND]
        or plan.get("phase3_no_positive_policy")
        != "typed_natural_terminal_v1"
        or plan.get("controller_horizon_policy")
        != "maximum_accepted_controller_rounds_v1"
        or plan.get("block_order") != ["append", "plateau"]
        or plan.get("cells") != [asdict(cell) for cell in CELL_SPECS]
        or plan.get("canonical_cell_order")
        != [cell.execution_id for cell in CELL_SPECS]
        or plan.get("deterministic_launch_order")
        != [
            execution_id
            for batch in BATCH_SPECS
            for execution_id in batch.execution_ids
        ]
        or plan.get("append_block_execution_ids")
        != [cell.execution_id for cell in CELL_SPECS[:6]]
        or plan.get("plateau_block_execution_ids")
        != [cell.execution_id for cell in CELL_SPECS[6:]]
        or plan.get("source_implementation_inventory_sha256")
        != inventory["sha256"]
        or plan.get("source_implementation_file_count") != inventory["source_count"]
        or plan.get("runner") != file_binding(RUNNER_PATH)
        or plan.get("runner_runtime_dependencies") != _runtime_dependencies()
        or plan.get("deterministic_batches")
        != [_batch_payload(batch) for batch in BATCH_SPECS]
        or plan.get("maximum_concurrency") != MAXIMUM_CONCURRENCY
        or plan.get("frontier_ratios") != PHASE_FRONTIER_RATIOS
        or plan.get("optimizer")
        != {
            "name": "powell", "xtol": 1.0e-4, "ftol": 1.0e-8,
            "maxiter": 200, "maxfev": None,
        }
        or plan.get("seeds") != {"adapt": 7, "transpiler": 7}
        or plan.get("shortlist_maxima") != SHORTLIST_MAXIMA
        or plan.get("runtime_environment") != EXPECTED_ENV
        or plan.get("serial_capacity_fallback_authorized") is not True
        or plan.get("silent_serial_fallback_authorized") is not False
        or plan.get("capacity") != expected_capacity
        or plan.get("append_block_must_close_before_plateau") is not True
        or plan.get("per_cell_storage_lifecycle")
        != _storage_lifecycle_contract()
        or plan.get("execution_path_canary")
        != {
            "execution_id": CELL_SPECS[0].execution_id,
            "completion_witness": (
                "accepted_round_1_or_authenticated_round_zero_terminal_v1"
            ),
            "continues_same_trajectory_to_completion": True,
            "separate_scientific_trajectory": False,
        }
        or plan.get("execution_authorized") is not False
        or plan.get("archive_rotation_authorized") is not False
        or plan.get("submission_authorized") is not False
        or plan.get("paper_adoption_authorized") is not False
        or plan.get("paper_evidence_adoption_authorized") is not False
    ):
        raise RunnerError("Overnight plan drifted.")
    if recompute_protocols and plan.get("protocol_bindings") != [
        _protocol_binding(cell) for cell in CELL_SPECS
    ]:
        raise RunnerError("Overnight protocol bindings drifted.")
    return plan


def authorize() -> dict[str, Any]:
    """Mint the separate direct execution receipt authorized by the user."""

    plan = validate_plan(recompute_protocols=True)
    if AUTHORIZATION_PATH.exists() or AUTHORIZATION_PATH.is_symlink():
        raise RunnerError("Direct execution authorization already exists.")
    authorization = digested(
        {
            "schema": AUTH_SCHEMA,
            "created_at": serial_runtime.utc_now(),
            "campaign_id": CAMPAIGN_ID,
            "authorization_basis": (
                "explicit_current_user_maximum_k50_natural_terminal_request"
            ),
            "plan_sha256": plan["sha256"],
            "runner_sha256": plan["runner"]["sha256"],
            "source_implementation_inventory_sha256": plan[
                "source_implementation_inventory_sha256"
            ],
            "execution_ids": [cell.execution_id for cell in CELL_SPECS],
            "execution_path_canary": plan["execution_path_canary"],
            "execution_authorized": True,
            "archive_rotation_authorized": True,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    write_json_exclusive(AUTHORIZATION_PATH, authorization)
    return authorization


def validate_authority(
    *, recompute_protocols: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = validate_plan(recompute_protocols=recompute_protocols)
    authorization = load_digested(AUTHORIZATION_PATH, schema=AUTH_SCHEMA)
    _require_exact_keys(authorization, AUTH_RECEIPT_KEYS, label="Authorization")
    if (
        not _valid_created_at(authorization)
        or authorization.get("campaign_id") != CAMPAIGN_ID
        or authorization.get("authorization_basis")
        != "explicit_current_user_maximum_k50_natural_terminal_request"
        or authorization.get("plan_sha256") != plan["sha256"]
        or authorization.get("runner_sha256") != plan["runner"]["sha256"]
        or authorization.get("source_implementation_inventory_sha256")
        != plan["source_implementation_inventory_sha256"]
        or authorization.get("execution_ids")
        != [cell.execution_id for cell in CELL_SPECS]
        or authorization.get("execution_path_canary")
        != plan["execution_path_canary"]
        or authorization.get("execution_authorized") is not True
        or authorization.get("archive_rotation_authorized") is not True
        or authorization.get("submission_authorized") is not False
        or authorization.get("paper_adoption_authorized") is not False
        or authorization.get("paper_evidence_adoption_authorized") is not False
    ):
        raise RunnerError("Direct all6 execution authorization drifted.")
    return plan, authorization


def validate_post_science_batch_identity(
    cells: Sequence[CellSpec],
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> None:
    """Repin source and route identity after science, before cell closure."""

    current_plan, current_authorization = validate_authority(
        recompute_protocols=True
    )
    if (
        current_plan != dict(plan)
        or current_authorization != dict(authorization)
    ):
        raise RunnerError("Post-science plan or authorization identity drifted.")
    for cell in cells:
        if _protocol_binding(cell) != _protocol_binding_for_cell(current_plan, cell):
            raise RunnerError(
                f"Post-science protocol/source identity drifted: {cell.execution_id}"
            )


def wait_for_capacity(
    *,
    maximum_wait_seconds: float = CAPACITY_WAIT_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    memory_supplier: Callable[[], int] | None = None,
    disk_supplier: Callable[[], int] | None = None,
) -> dict[str, Any]:
    return serial_runtime.wait_for_capacity(
        repo_root=REPO_ROOT,
        launch_memory_bytes=LAUNCH_AVAILABLE_MEMORY_BYTES,
        launch_disk_bytes=LAUNCH_FREE_DISK_BYTES,
        maximum_wait_seconds=maximum_wait_seconds,
        poll_seconds=CAPACITY_POLL_SECONDS,
        clock=clock,
        sleeper=sleeper,
        memory_supplier=memory_supplier,
        disk_supplier=disk_supplier,
    )


def _bounded_capacity_wait(
    *,
    launch_memory_bytes: int,
    launch_disk_bytes: int,
    maximum_wait_seconds: float = CAPACITY_WAIT_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    memory_supplier: Callable[[], int] | None = None,
    disk_supplier: Callable[[], int] | None = None,
) -> dict[str, Any]:
    observed = serial_runtime.wait_for_capacity(
        repo_root=REPO_ROOT,
        launch_memory_bytes=launch_memory_bytes,
        launch_disk_bytes=launch_disk_bytes,
        maximum_wait_seconds=maximum_wait_seconds,
        poll_seconds=CAPACITY_POLL_SECONDS,
        clock=clock,
        sleeper=sleeper,
        memory_supplier=memory_supplier,
        disk_supplier=disk_supplier,
    )
    if float(observed["elapsed_wait_seconds"]) > maximum_wait_seconds:
        return {
            **observed,
            "status": "blocked_capacity",
            "ready_after_bound": observed.get("status") == "ready",
        }
    return {**observed, "ready_after_bound": False}


def wait_for_cell_launch_capacity(
    cell: CellSpec,
    **wait_kwargs: Any,
) -> dict[str, Any]:
    floor = strict_archive.regime_launch_capacity_floor(
        regime_id=cell.regime_id,
        nph=cell.nph,
    )
    observed = _bounded_capacity_wait(
        launch_memory_bytes=LAUNCH_AVAILABLE_MEMORY_BYTES,
        launch_disk_bytes=max(
            LAUNCH_FREE_DISK_BYTES, int(floor["minimum_free_bytes"])
        ),
        **wait_kwargs,
    )
    return {
        **observed,
        "capacity_kind": "per_regime_cell_launch",
        "execution_id": cell.execution_id,
        "regime_capacity_floor": floor,
    }


def wait_for_archive_capacity(**wait_kwargs: Any) -> dict[str, Any]:
    limits = strict_archive.campaign_default_archive_limits()
    return {
        **_bounded_capacity_wait(
            launch_memory_bytes=AVAILABLE_MEMORY_FLOOR_BYTES,
            launch_disk_bytes=limits.archive_start_free_floor_bytes,
            **wait_kwargs,
        ),
        "capacity_kind": "per_cell_archive_build",
        "archive_limits": limits.as_dict(),
    }


def ensure_initial_campaign_capacity(
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    capacity_waiter: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    floor = strict_archive.campaign_capacity_floor()
    if INITIAL_CAPACITY_PATH.is_file():
        return load_initial_campaign_capacity(
            plan=plan, authorization=authorization
        )
    _assert_campaign_pristine_before_initial_capacity()
    capacity_waiter = capacity_waiter or _bounded_capacity_wait
    observation = dict(
        capacity_waiter(
            launch_memory_bytes=LAUNCH_AVAILABLE_MEMORY_BYTES,
            launch_disk_bytes=int(floor["campaign_minimum_free_bytes"]),
        )
    )
    if observation.get("status") != "ready":
        raise ArchiveCapacityBlocked(
            {**observation, "capacity_kind": "initial_campaign"}
        )
    strict_observation = strict_archive.require_campaign_capacity(REPO_ROOT)
    receipt = digested(
        {
            "schema": INITIAL_CAPACITY_SCHEMA,
            "status": "passed_one_time_initial_campaign_capacity",
            "created_at": serial_runtime.utc_now(),
            "campaign_id": CAMPAIGN_ID,
            "plan_sha256": plan["sha256"],
            "authorization_sha256": authorization["sha256"],
            "source_implementation_inventory_sha256": plan[
                "source_implementation_inventory_sha256"
            ],
            "campaign_capacity_floor": floor,
            "bounded_wait_observation": observation,
            "strict_capacity_observation": strict_observation,
            "one_time_gate_not_reimposed_on_restart": True,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    write_json_exclusive(INITIAL_CAPACITY_PATH, receipt)
    return receipt


def _assert_campaign_pristine_before_initial_capacity() -> None:
    evidence_paths = [REPORT_JSON, REPORT_CSV, REPORT_MD, TERMINAL_PATH]
    evidence_paths.extend(scheduler_decision_path(batch) for batch in BATCH_SPECS)
    evidence_paths.extend(batch_receipt_path(batch) for batch in BATCH_SPECS)
    if any(path.exists() or path.is_symlink() for path in evidence_paths):
        raise RunnerError(
            "Initial capacity receipt is missing after campaign evidence exists."
        )
    runtime_present = RUNTIME_ROOT.exists() or RUNTIME_ROOT.is_symlink()
    if RUNTIME_ROOT.is_symlink() or (runtime_present and not RUNTIME_ROOT.is_dir()):
        raise RunnerError("Initial capacity runtime root is unsafe.")
    for cell in CELL_SPECS:
        evidence_paths.extend(cell_paths(cell))
        evidence_paths.extend(
            (
                cell_log_path(cell), compact_cell_receipt_path(cell),
                archived_cell_receipt_path(cell),
            )
        )
        paths = archive_paths(cell)
        if runtime_present:
            try:
                state = strict_archive.inspect_rotation_state(paths)
            except strict_archive.Singleton12ArchiveError as exc:
                raise RunnerError("Initial archive evidence is malformed.") from exc
            if state.get("state") != "empty" or state.get(
                "stale_archive_temporaries"
            ):
                raise RunnerError(
                    "Initial capacity receipt is missing after archive evidence exists."
                )
        evidence_paths.extend(
            (
                paths.retiring_root,
                paths.archive_path,
                paths.archive_manifest_path,
                paths.archive_closure_path,
                paths.rotation_intent_path,
                paths.cleanup_receipt_path,
            )
        )
    observed = sorted(
        path.as_posix()
        for path in evidence_paths
        if path.exists() or path.is_symlink()
    )
    if observed:
        raise RunnerError(
            "Initial capacity receipt is missing after campaign evidence exists."
        )


def load_initial_campaign_capacity(
    *, plan: Mapping[str, Any], authorization: Mapping[str, Any]
) -> dict[str, Any]:
    """Purely validate the already-published one-time capacity receipt."""

    receipt = load_digested(INITIAL_CAPACITY_PATH, schema=INITIAL_CAPACITY_SCHEMA)
    _require_exact_keys(
        receipt, INITIAL_CAPACITY_RECEIPT_KEYS, label="Initial capacity"
    )
    floor = strict_archive.campaign_capacity_floor()
    bounded = receipt.get("bounded_wait_observation")
    strict = receipt.get("strict_capacity_observation")
    bounded_keys = {
        "available_memory_bytes", "free_disk_bytes",
        "launch_available_memory_bytes", "launch_free_disk_bytes",
        "launch_ready", "elapsed_wait_seconds", "status", "ready_after_bound",
    }
    strict_keys = set(floor) | {"status", "observed_free_bytes", "headroom_bytes"}
    valid_nested = isinstance(bounded, Mapping) and isinstance(strict, Mapping)
    if valid_nested:
        try:
            bounded_elapsed = float(bounded.get("elapsed_wait_seconds", -1.0))
            bounded_memory = int(bounded.get("available_memory_bytes", -1))
            bounded_disk = int(bounded.get("free_disk_bytes", -1))
            strict_free = int(strict.get("observed_free_bytes", -1))
            strict_headroom = int(strict.get("headroom_bytes", -1))
        except (TypeError, ValueError):
            valid_nested = False
        else:
            valid_nested = (
            set(bounded) == bounded_keys
            and bounded.get("status") == "ready"
            and bounded.get("launch_ready") is True
            and bounded.get("ready_after_bound") is False
            and bounded.get("launch_available_memory_bytes")
            == LAUNCH_AVAILABLE_MEMORY_BYTES
            and bounded.get("launch_free_disk_bytes")
            == int(floor["campaign_minimum_free_bytes"])
            and bounded_memory >= LAUNCH_AVAILABLE_MEMORY_BYTES
            and bounded_disk >= int(floor["campaign_minimum_free_bytes"])
            and 0.0 <= bounded_elapsed <= CAPACITY_WAIT_SECONDS
            and set(strict) == strict_keys
            and all(strict.get(key) == value for key, value in floor.items())
            and strict.get("status") == "passed_campaign_capacity_floor"
            and strict_free >= int(floor["campaign_minimum_free_bytes"])
            and strict_headroom == strict_free
            - int(floor["campaign_minimum_free_bytes"])
            )
    if (
        not _valid_created_at(receipt)
        or not valid_nested
        or receipt.get("status") != "passed_one_time_initial_campaign_capacity"
        or receipt.get("campaign_id") != CAMPAIGN_ID
        or receipt.get("plan_sha256") != plan["sha256"]
        or receipt.get("authorization_sha256") != authorization["sha256"]
        or receipt.get("source_implementation_inventory_sha256")
        != plan["source_implementation_inventory_sha256"]
        or receipt.get("campaign_capacity_floor") != floor
        or receipt.get("one_time_gate_not_reimposed_on_restart") is not True
        or receipt.get("submission_authorized") is not False
        or receipt.get("paper_adoption_authorized") is not False
        or receipt.get("paper_evidence_adoption_authorized") is not False
    ):
        raise RunnerError("Initial campaign capacity receipt drifted.")
    return receipt


def validate_launch_capacity_observation(
    cells: Sequence[CellSpec],
    *,
    batch: BatchSpec,
    scheduling_mode: str,
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Deeply bind a launch observation to the exact child scheduling contract."""

    cells = tuple(cells)
    value = dict(observation)
    base_keys = {
        "available_memory_bytes", "free_disk_bytes",
        "launch_available_memory_bytes", "launch_free_disk_bytes",
        "launch_ready", "elapsed_wait_seconds", "status", "ready_after_bound",
    }
    try:
        elapsed = float(value.get("elapsed_wait_seconds", -1.0))
        available = int(value.get("available_memory_bytes", -1))
        free_disk = int(value.get("free_disk_bytes", -1))
    except (TypeError, ValueError) as exc:
        raise RunnerError("Launch-capacity observation is malformed.") from exc
    if scheduling_mode == "pair":
        contract = pair_launch_capacity_contract(batch)
        expected_keys = base_keys | set(contract) | {
            "capacity_kind", "physical_memory_bytes"
        }
        try:
            physical = int(value.get("physical_memory_bytes", -1))
        except (TypeError, ValueError) as exc:
            raise RunnerError("Pair launch physical memory is malformed.") from exc
        valid = (
            len(cells) == 2
            and tuple(cell.execution_id for cell in cells) == batch.execution_ids
            and set(value) == expected_keys
            and all(value.get(key) == item for key, item in contract.items())
            and value.get("capacity_kind") == "fresh_pair_launch_recheck"
            and value.get("status") == "ready"
            and value.get("launch_ready") is True
            and value.get("ready_after_bound") is False
            and value.get("launch_available_memory_bytes")
            == int(contract["required_available_memory_bytes"])
            and value.get("launch_free_disk_bytes")
            == int(contract["required_free_disk_bytes"])
            and available >= int(contract["required_available_memory_bytes"])
            and free_disk >= int(contract["required_free_disk_bytes"])
            and physical >= int(contract["required_physical_memory_bytes"])
            and 0.0 <= elapsed <= CAPACITY_WAIT_SECONDS
        )
    elif scheduling_mode == "serial_capacity_fallback":
        if len(cells) != 1 or cells[0].execution_id not in batch.execution_ids:
            valid = False
        else:
            cell = cells[0]
            floor = strict_archive.regime_launch_capacity_floor(
                regime_id=cell.regime_id, nph=cell.nph
            )
            required_disk = max(
                LAUNCH_FREE_DISK_BYTES, int(floor["minimum_free_bytes"])
            )
            expected_keys = base_keys | {
                "capacity_kind", "execution_id", "regime_capacity_floor"
            }
            valid = (
                set(value) == expected_keys
                and value.get("capacity_kind") == "per_regime_cell_launch"
                and value.get("execution_id") == cell.execution_id
                and value.get("regime_capacity_floor") == floor
                and value.get("status") == "ready"
                and value.get("launch_ready") is True
                and value.get("ready_after_bound") is False
                and value.get("launch_available_memory_bytes")
                == LAUNCH_AVAILABLE_MEMORY_BYTES
                and value.get("launch_free_disk_bytes") == required_disk
                and available >= LAUNCH_AVAILABLE_MEMORY_BYTES
                and free_disk >= required_disk
                and 0.0 <= elapsed <= CAPACITY_WAIT_SECONDS
            )
    else:
        valid = False
    if not valid:
        raise RunnerError("Launch-capacity observation drifted from its mode contract.")
    return value


def empirical_peak_rss_anchor(cell: CellSpec) -> dict[str, Any]:
    """Return the conservative checked-in RAM anchor for one cell."""

    substituted = False
    if cell.nph == 3:
        if cell.block == "append" and cell.regime_id == "intermediate_weak":
            key = "nph3_append_intermediate_weak"
        elif cell.block == "append" and cell.regime_id == "weak_weak":
            key = "nph3_append_weak_weak"
        else:
            # The largest measured nph3 peak dominates the shorter strong-weak
            # always-open canary and is used for unmeasured nph3 full horizons.
            key = "nph3_append_weak_weak"
            substituted = True
    elif cell.block == "append":
        if cell.regime_id == "intermediate_strong":
            key = "nph7_append_intermediate_strong"
        elif cell.regime_id == "strong_strong_u8":
            key = "nph7_append_strong_strong_u8"
        else:
            key = "nph7_append_strong_strong_u8"
            substituted = True
    else:
        key = f"nph7_plateau_{cell.regime_id}"
    evidence = EMPIRICAL_RSS_EVIDENCE[key]
    return {
        "execution_id": cell.execution_id,
        "regime_id": cell.regime_id,
        "nph": cell.nph,
        "block": cell.block,
        "peak_rss_bytes": int(evidence["peak_rss_bytes"]),
        "evidence_key": key,
        "evidence_source": str(evidence["source"]),
        "evidence_source_file_sha256": str(evidence["source_file_sha256"]),
        "conservative_same_nph_substitution": substituted,
    }


def child_rss_limit_bytes(cell: CellSpec) -> int:
    if cell.nph == 7 and cell.block == "plateau":
        return NPH7_PLATEAU_CHILD_RSS_LIMIT_BYTES
    return CHILD_RSS_LIMIT_BYTES


def archive_restart_action(
    observed_state: Mapping[str, Any],
    *,
    archive_rotation_authorized: bool,
) -> str:
    """Map one validated strict archive state to its sole safe next action."""

    stale = observed_state.get("stale_archive_temporaries")
    if not isinstance(stale, list) or any(not isinstance(row, str) for row in stale):
        raise RunnerError("Strict archive state has malformed temporary evidence.")
    if stale:
        raise RunnerError("Strict archive state contains unresolved temporaries.")
    state = observed_state.get("state")
    actions = {
        "empty": "launch",
        "direct_unarchived": "prepare_archive",
        "archive_published_pending_manifest": "resume_archive",
        "manifest_published_pending_closure": "resume_archive",
        "closure_published_pending_intent": "publish_rotation_intent",
        "intent_published_pending_rename": "complete_rotation",
        "retiring_pending_removal": "complete_rotation",
        "cleanup_receipt_pending": "complete_rotation",
        "archived_closed": "validate_archived",
    }
    try:
        action = actions[str(state)]
    except KeyError as exc:
        raise RunnerError("Unknown strict archive restart state.") from exc
    if action in {"publish_rotation_intent", "complete_rotation"} and not (
        archive_rotation_authorized
    ):
        return "blocked_missing_rotation_authority"
    return action


def _guarded_working_disk_bytes(cell: CellSpec) -> tuple[int, dict[str, Any]]:
    evidence = strict_archive.regime_launch_capacity_floor(
        regime_id=cell.regime_id,
        nph=cell.nph,
    )
    raw = int(evidence["observed_working_disk_bytes"])
    factor = evidence["working_space_safety_factor"]
    numerator = int(factor["numerator"])
    denominator = int(factor["denominator"])
    guarded = (numerator * raw + denominator - 1) // denominator
    return guarded, evidence


def pair_launch_capacity_contract(batch: BatchSpec) -> dict[str, Any]:
    cells = tuple(_cell_by_execution_id(value) for value in batch.execution_ids)
    if len(cells) != MAXIMUM_CONCURRENCY or any(
        cell.block != batch.block for cell in cells
    ):
        raise RunnerError("Capacity batch identity drifted.")
    memory_anchors = [empirical_peak_rss_anchor(cell) for cell in cells]
    disk_rows: list[dict[str, Any]] = []
    guarded_disk_total = 0
    for cell in cells:
        guarded, evidence = _guarded_working_disk_bytes(cell)
        guarded_disk_total += guarded
        disk_rows.append(
            {
                "execution_id": cell.execution_id,
                "regime_id": cell.regime_id,
                "nph": cell.nph,
                "observed_working_disk_bytes": evidence[
                    "observed_working_disk_bytes"
                ],
                "guarded_working_disk_bytes": guarded,
                "capacity_evidence": evidence["capacity_evidence"],
            }
        )
    archive_floor = (
        strict_archive.campaign_default_archive_limits()
        .archive_start_free_floor_bytes
    )
    required_memory = (
        sum(int(row["peak_rss_bytes"]) for row in memory_anchors)
        + AVAILABLE_MEMORY_FLOOR_BYTES
    )
    return {
        "schema": "paper_i_ra_all6_pair_launch_capacity_contract_v1",
        "batch": _batch_payload(batch),
        "maximum_concurrency": MAXIMUM_CONCURRENCY,
        "required_available_memory_bytes": required_memory,
        "required_physical_memory_bytes": required_memory,
        "required_free_disk_bytes": guarded_disk_total + archive_floor,
        "runtime_available_memory_reserve_bytes": AVAILABLE_MEMORY_FLOOR_BYTES,
        "archive_start_free_floor_bytes": archive_floor,
        "memory_anchors": memory_anchors,
        "disk_anchors": disk_rows,
        "memory_formula": "sum_empirical_peak_rss_bytes_plus_2gib_reserve",
        "disk_formula": (
            "sum_ceil_5_over_4_observed_working_bytes_plus_archive_start_floor"
        ),
        "host_physical_memory_evidence": dict(HOST_PHYSICAL_MEMORY_EVIDENCE),
    }


def wait_for_batch_capacity(
    batch: BatchSpec,
    *,
    maximum_wait_seconds: float = CAPACITY_WAIT_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    memory_supplier: Callable[[], int] | None = None,
    physical_memory_supplier: Callable[[], int] | None = None,
    disk_supplier: Callable[[], int] | None = None,
) -> dict[str, Any]:
    """Attempt pair capacity, then return an explicit serial fallback."""

    contract = pair_launch_capacity_contract(batch)
    memory_supplier = memory_supplier or (
        lambda: int(psutil.virtual_memory().available)
    )
    disk_supplier = disk_supplier or (
        lambda: int(shutil.disk_usage(REPO_ROOT).free)
    )
    physical_memory_supplier = physical_memory_supplier or (
        lambda: int(psutil.virtual_memory().total)
    )
    physical = int(physical_memory_supplier())
    required_physical = int(contract["required_physical_memory_bytes"])
    if physical < required_physical:
        available = int(memory_supplier())
        free_disk = int(disk_supplier())
        reasons = ["physical"]
        if available < int(contract["required_available_memory_bytes"]):
            reasons.append("memory")
        if free_disk < int(contract["required_free_disk_bytes"]):
            reasons.append("disk")
        return {
            "status": "serial_capacity_fallback",
            "scheduling_mode": "serial_capacity_fallback",
            "fallback_reasons": reasons,
            "physical_memory_bytes": physical,
            "available_memory_bytes": available,
            "free_disk_bytes": free_disk,
            "elapsed_wait_seconds": 0.0,
            **contract,
        }
    observed = serial_runtime.wait_for_capacity(
        repo_root=REPO_ROOT,
        launch_memory_bytes=int(contract["required_available_memory_bytes"]),
        launch_disk_bytes=int(contract["required_free_disk_bytes"]),
        maximum_wait_seconds=maximum_wait_seconds,
        poll_seconds=CAPACITY_POLL_SECONDS,
        clock=clock,
        sleeper=sleeper,
        memory_supplier=memory_supplier,
        disk_supplier=disk_supplier,
    )
    elapsed = float(observed["elapsed_wait_seconds"])
    ready_within_bound = observed["status"] == "ready" and elapsed <= maximum_wait_seconds
    if ready_within_bound:
        return {
            **contract,
            **observed,
            "status": "ready_pair",
            "scheduling_mode": "pair",
            "physical_memory_bytes": physical,
            "fallback_reasons": [],
        }
    reasons: list[str] = []
    if int(observed["available_memory_bytes"]) < int(
        contract["required_available_memory_bytes"]
    ):
        reasons.append("memory")
    if int(observed["free_disk_bytes"]) < int(
        contract["required_free_disk_bytes"]
    ):
        reasons.append("disk")
    if elapsed >= maximum_wait_seconds and not reasons:
        reasons.append("wait_bound")
    return {
        **contract,
        **observed,
        "status": "serial_capacity_fallback",
        "scheduling_mode": "serial_capacity_fallback",
        "physical_memory_bytes": physical,
        "fallback_reasons": reasons,
    }


def scheduler_decision_path(batch: BatchSpec) -> Path:
    return SCHEDULER_ROOT / f"batch_{batch.ordinal:02d}.json"


def validate_scheduler_decision(
    batch: BatchSpec,
    decision: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    _require_exact_keys(decision, SCHEDULER_RECEIPT_KEYS, label="Scheduler")
    contract = pair_launch_capacity_contract(batch)
    observation = decision.get("capacity_observation")
    if not isinstance(observation, Mapping):
        raise RunnerError("Scheduler capacity observation is absent.")
    mode = str(decision.get("scheduling_mode"))
    reasons = observation.get("fallback_reasons")
    if not isinstance(reasons, list) or any(
        reason not in {"memory", "disk", "physical", "wait_bound"}
        for reason in reasons
    ):
        raise RunnerError("Scheduler fallback reasons are invalid.")
    required_memory = int(contract["required_available_memory_bytes"])
    required_disk = int(contract["required_free_disk_bytes"])
    available = int(observation.get("available_memory_bytes", -1))
    free_disk = int(observation.get("free_disk_bytes", -1))
    physical = int(observation.get("physical_memory_bytes", -1))
    elapsed = float(observation.get("elapsed_wait_seconds", -1.0))
    pair_ready = (
        available >= required_memory
        and free_disk >= required_disk
        and physical >= int(contract["required_physical_memory_bytes"])
    )
    expected_fallback_reasons = {
        reason
        for reason, failed in {
            "memory": available < required_memory,
            "disk": free_disk < required_disk,
            "physical": physical
            < int(contract["required_physical_memory_bytes"]),
        }.items()
        if failed
    }
    if elapsed >= CAPACITY_WAIT_SECONDS and not expected_fallback_reasons:
        expected_fallback_reasons.add("wait_bound")
    valid_fallback_reasons = set(reasons) == expected_fallback_reasons
    if (
        not _valid_created_at(decision)
        or decision.get("schema") != SCHEDULER_SCHEMA
        or decision.get("campaign_id") != CAMPAIGN_ID
        or decision.get("batch") != _batch_payload(batch)
        or decision.get("execution_ids") != list(batch.execution_ids)
        or decision.get("plan_sha256") != plan["sha256"]
        or decision.get("authorization_sha256") != authorization["sha256"]
        or decision.get("source_implementation_inventory_sha256")
        != plan["source_implementation_inventory_sha256"]
        or decision.get("capacity_contract") != contract
        or decision.get("capacity_observation_sha256")
        != canonical_sha256(observation)
        or decision.get("maximum_concurrency") != MAXIMUM_CONCURRENCY
        or decision.get("serial_capacity_fallback_audited")
        is not (mode == "serial_capacity_fallback")
        or decision.get("silent_serial_fallback_authorized") is not False
        or decision.get("submission_authorized") is not False
        or decision.get("paper_adoption_authorized") is not False
        or decision.get("paper_evidence_adoption_authorized") is not False
        or elapsed < 0.0
        or any(observation.get(key) != value for key, value in contract.items())
        or (
            mode == "pair"
            and (
                observation.get("status") != "ready_pair"
                or observation.get("scheduling_mode") != "pair"
                or observation.get("launch_ready") is not True
                or observation.get("launch_available_memory_bytes")
                != required_memory
                or observation.get("launch_free_disk_bytes") != required_disk
                or elapsed > CAPACITY_WAIT_SECONDS
                or not pair_ready
                or reasons
            )
        )
        or (
            mode == "serial_capacity_fallback"
            and (
                observation.get("status") != "serial_capacity_fallback"
                or observation.get("scheduling_mode")
                != "serial_capacity_fallback"
                or (pair_ready and "wait_bound" not in reasons)
                or not reasons
                or not valid_fallback_reasons
                or (
                    "physical" not in reasons
                    and elapsed < CAPACITY_WAIT_SECONDS
                )
            )
        )
        or mode not in {"pair", "serial_capacity_fallback"}
    ):
        raise RunnerError("Immutable batch scheduler decision drifted.")
    return dict(decision)


def select_batch_schedule(
    batch: BatchSpec,
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    capacity_waiter: Callable[[BatchSpec], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    path = scheduler_decision_path(batch)
    if path.is_file():
        return validate_scheduler_decision(
            batch,
            load_digested(path, schema=SCHEDULER_SCHEMA),
            plan=plan,
            authorization=authorization,
        )
    capacity_waiter = capacity_waiter or wait_for_batch_capacity
    observation = dict(capacity_waiter(batch))
    mode = str(observation.get("scheduling_mode"))
    decision = digested(
        {
            "schema": SCHEDULER_SCHEMA,
            "created_at": serial_runtime.utc_now(),
            "campaign_id": CAMPAIGN_ID,
            "batch": _batch_payload(batch),
            "execution_ids": list(batch.execution_ids),
            "plan_sha256": plan["sha256"],
            "authorization_sha256": authorization["sha256"],
            "source_implementation_inventory_sha256": plan[
                "source_implementation_inventory_sha256"
            ],
            "scheduling_mode": mode,
            "maximum_concurrency": MAXIMUM_CONCURRENCY,
            "capacity_contract": pair_launch_capacity_contract(batch),
            "capacity_observation": observation,
            "capacity_observation_sha256": canonical_sha256(observation),
            "serial_capacity_fallback_audited": (
                mode == "serial_capacity_fallback"
            ),
            "silent_serial_fallback_authorized": False,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    validate_scheduler_decision(
        batch,
        decision,
        plan=plan,
        authorization=authorization,
    )
    write_json_exclusive(path, decision)
    return decision


def assert_append_block_closed(completed_execution_ids: Sequence[str]) -> None:
    expected = [
        execution_id
        for batch in BATCH_SPECS[:3]
        for execution_id in batch.execution_ids
    ]
    if list(completed_execution_ids) != expected:
        raise RunnerError(
            "The append block must have six validated closures in fixed order "
            "before any plateau cell may start."
        )


def write_status(payload: Mapping[str, Any]) -> None:
    serial_runtime.write_json_atomic(
        STATUS_PATH,
        digested(
            {
                "schema": STATUS_SCHEMA,
                "campaign_id": CAMPAIGN_ID,
                "updated_at": serial_runtime.utc_now(),
                **dict(payload),
            }
        ),
    )


def assert_environment() -> None:
    drift = {
        key: {"expected": value, "observed": os.environ.get(key)}
        for key, value in EXPECTED_ENV.items()
        if os.environ.get(key) != value
    }
    if drift:
        raise RunnerError(f"Numerical environment drifted: {drift}")


def cell_paths(cell: CellSpec) -> tuple[Path, Path, Path, Path]:
    return (
        RUNS_ROOT / cell.execution_id,
        STAGING_ROOT / cell.execution_id,
        RECEIPTS_ROOT / f"{cell.execution_id}.json",
        GUARD_ROOT / f"{cell.execution_id}.json",
    )


def cell_log_path(cell: CellSpec) -> Path:
    return CELL_LOG_ROOT / f"{cell.execution_id}.log"


def compact_cell_receipt_path(cell: CellSpec) -> Path:
    return COMPACT_ROOT / f"{cell.execution_id}.json"


def archived_cell_receipt_path(cell: CellSpec) -> Path:
    return ARCHIVED_RECEIPTS_ROOT / f"{cell.execution_id}.json"


def batch_receipt_path(batch: BatchSpec) -> Path:
    return BATCH_RECEIPTS_ROOT / f"batch_{batch.ordinal:02d}.json"


def archive_paths(cell: CellSpec) -> Any:
    return strict_archive.CellArchivePaths(
        runtime_root=RUNTIME_ROOT,
        execution_id=cell.execution_id,
    )


def _batch_for_cell(cell: CellSpec) -> BatchSpec:
    matches = [
        batch for batch in BATCH_SPECS if cell.execution_id in batch.execution_ids
    ]
    if len(matches) != 1:
        raise RunnerError("Cell does not belong to exactly one capacity batch.")
    return matches[0]


def classify_cell_output(cell: CellSpec) -> str:
    state = tuple(
        path.exists() or path.is_symlink() for path in cell_paths(cell)
    )
    if state == (False, False, False, False):
        return "pristine"
    if state == (True, False, True, True):
        return "closed"
    raise RunnerError(f"Cell has partial output: {cell.execution_id}")


def child_token(authorization_sha256: str, cell: CellSpec) -> str:
    return canonical_sha256(
        {
            "campaign_id": CAMPAIGN_ID,
            "authorization_sha256": authorization_sha256,
            "execution_id": cell.execution_id,
            "route_variant": cell.route_variant,
        }
    )


def _cell_by_execution_id(execution_id: str) -> CellSpec:
    matches = [cell for cell in CELL_SPECS if cell.execution_id == execution_id]
    if len(matches) != 1:
        raise RunnerError("Unknown all6 overnight execution ID.")
    return matches[0]


def _artifact_binding(path: Path, root: Path) -> dict[str, Any]:
    try:
        return serial_runtime.artifact_binding(path, root)
    except serial_runtime.RuntimeContractError as exc:
        raise RunnerError(str(exc)) from exc


def _materialize_cell(cell: CellSpec):
    module = _load_core_module()
    problem = getattr(module, CORE_PROBLEM_BUILDER)(cell.regime_id)
    request = getattr(module, CORE_REQUEST_BUILDER)(
        insertion_policy=cell.insertion_policy,
        maximum_controller_rounds=cell.horizon,
    )
    protocol = module.materialize_paper_i_ra_semantic_protocol(problem, request)
    return problem, protocol


def _completion_mapping(value: Any, *, owner: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RunnerError(f"{owner} must be a mapping.")
    return copy.deepcopy(dict(value))


def _completion_sequence(value: Any, *, owner: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise RunnerError(f"{owner} must be a sequence.")
    return copy.deepcopy(list(value))


def _signed_completion_sha(value: Any, *, owner: str) -> str:
    payload = _completion_mapping(value, owner=owner)
    observed = payload.pop("sha256", None)
    expected = canonical_sha256(payload)
    if observed != expected:
        raise RunnerError(f"{owner} digest drifted.")
    return str(observed)


def _validate_selector_closure(
    scientific: Mapping[str, Any],
    *,
    accepted_rounds: int,
    terminal_outcome: str | None,
    terminal_receipt_sha256: str | None,
) -> str:
    closure = _completion_mapping(
        scientific.get("semantic_selector_accounting_closure"),
        owner="semantic selector accounting closure",
    )
    closure_sha256 = _signed_completion_sha(
        closure, owner="semantic selector accounting closure"
    )
    rounds = closure.get("rounds")
    if (
        closure.get("schema")
        != "paper_i_ra_semantic_final_selector_accounting_closure_v1"
        or closure.get("route_variant") != EXPECTED_CORE_ROUTE_VARIANT
        or closure.get("validated_round_count") != accepted_rounds
        or not isinstance(rounds, list)
        or [row.get("accepted_round") for row in rounds]
        != list(range(1, accepted_rounds + 1))
    ):
        raise RunnerError("Semantic selector accounting closure drifted.")
    if terminal_outcome == ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1:
        if (
            closure.get("terminal_controller_outcome") != terminal_outcome
            or closure.get("terminal_accepted_controller_round")
            != accepted_rounds
            or closure.get("terminal_attempted_controller_round")
            != accepted_rounds + 1
            or closure.get("terminal_phase3_selection_receipt_sha256")
            != terminal_receipt_sha256
            or closure.get("terminal_final_admission_record_id") is not None
        ):
            raise RunnerError("Natural terminal selector closure drifted.")
    elif any(
        key.startswith("terminal_") for key in closure if key != "terminal_unused"
    ):
        raise RunnerError("Exact-target selector closure smuggled a terminal.")
    return closure_sha256


def _authenticate_checkpoint(
    *,
    checkpoint_path: Path,
    problem: Any,
    run_route: Mapping[str, Any],
    final_state: Mapping[str, Any],
    natural_terminal: bool,
) -> tuple[str, str | None]:
    checkpoint = Path(checkpoint_path)
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise RunnerError("Cell checkpoint must be a regular, non-symlink file.")
    checkpoint_sha256 = sha256_file(checkpoint)
    profile = run_route.get("profile")
    route_contract_sha256 = run_route.get("contract_sha256")
    if (
        not isinstance(profile, str)
        or not profile
        or not isinstance(route_contract_sha256, str)
        or len(route_contract_sha256) != 64
    ):
        raise RunnerError("Checkpoint route binding is invalid.")
    try:
        hydration = load_canonical_accepted_state_resume(
            AcceptedStateResume(
                checkpoint_path=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
            ),
            expected_problem=problem,
            expected_route_profile=profile,
            expected_route_contract_sha256=route_contract_sha256,
        )
    except CanonicalResumeError as exc:
        accepted_terminal_messages = {
            "Authenticated Phase-III natural terminal is complete and "
            "non-resumable.",
            "Authenticated Phase-III natural terminal is complete and "
            "non-resumable on the same route.",
        }
        if not natural_terminal or str(exc) not in accepted_terminal_messages:
            raise RunnerError("Cell checkpoint authentication failed.") from exc
        hydration = None
    if natural_terminal:
        if hydration is not None:
            raise RunnerError(
                "Natural-terminal checkpoint incorrectly advertised resumability."
            )
        return checkpoint_sha256, None
    if hydration is None or (
        hydration.controller_round != TARGET_HORIZON
        or hydration.accepted_state_fingerprint
        != final_state.get("projective_state_fingerprint")
        or not math.isclose(
            float(hydration.accepted_energy),
            float(final_state.get("energy")),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise RunnerError("Exact-target checkpoint state drifted.")
    return checkpoint_sha256, canonical_sha256(
        hydration.mutable_terminal_signed_checkpoint()
    )


def validate_cell_completion(
    cell: CellSpec,
    *,
    result: Mapping[str, Any],
    summary: Mapping[str, Any] | None,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Authenticate one maximum-k50 cell without publishing or resuming it."""

    if not isinstance(cell, CellSpec) or cell not in CELL_SPECS:
        raise RunnerError("Unknown maximum-k50 campaign cell.")
    problem, protocol = _materialize_cell(cell)
    raw_result = _completion_mapping(result, owner="cell result")
    if raw_result.get("protocol") != protocol.to_dict():
        raise RunnerError("Cell result protocol drifted from the frozen cell.")

    run = _completion_mapping(raw_result.get("run"), owner="cell run")
    run_route = _completion_mapping(run.get("route"), owner="cell run route")
    final_state = _completion_mapping(
        run.get("final_state"), owner="final accepted state"
    )
    trajectory = _completion_sequence(
        run.get("accepted_trajectory"), owner="accepted trajectory"
    )
    transitions = _completion_sequence(
        run.get("accepted_transitions"), owner="accepted transitions"
    )
    scientific_replay = _completion_sequence(
        run.get("scientific_replay"), owner="scientific replay"
    )
    accepted_rounds = len(trajectory)
    if (
        accepted_rounds > TARGET_HORIZON
        or len(transitions) != accepted_rounds
        or len(scientific_replay) != accepted_rounds
        or [row.get("controller_round") for row in trajectory]
        != list(range(1, accepted_rounds + 1))
        or (
            accepted_rounds > 0
            and final_state != trajectory[-1]
        )
        or (
            accepted_rounds == 0
            and (
                final_state.get("controller_round") != 0
                or final_state.get("operators") != []
            )
        )
    ):
        raise RunnerError("Accepted cell trajectory cardinality drifted.")
    stop = _completion_mapping(run.get("stop"), owner="cell stop receipt")
    final_operators = _completion_sequence(
        final_state.get("operators"), owner="final accepted operators"
    )
    if (
        stop.get("completed_controller_rounds") != accepted_rounds
        or stop.get("accepted_operator_count") != len(final_operators)
    ):
        raise RunnerError("Cell stop receipt drifted from its accepted state.")

    scientific = _completion_mapping(
        raw_result.get("scientific_receipts"), owner="scientific receipts"
    )
    resolved_route = _completion_mapping(
        scientific.get("resolved_route_contract"),
        owner="resolved semantic route contract",
    )
    expected_route = _completion_mapping(
        protocol.to_dict().get("route_contract"),
        owner="expected semantic route contract",
    )
    route_contract_sha256 = str(expected_route.pop("sha256", ""))
    comparable_route = dict(resolved_route)
    embedded_route_sha256 = comparable_route.pop("sha256", None)
    if (
        comparable_route != expected_route
        or embedded_route_sha256 not in {None, route_contract_sha256}
    ):
        raise RunnerError("Resolved semantic route contract drifted.")
    try:
        validate_semantic_phase3_natural_terminal_route_contract(
            resolved_route,
            expected_route_contract_sha256=route_contract_sha256,
        )
        replay_evidence = validate_controller_replay_evidence(
            scientific.get("controller_replay_evidence")
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RunnerError(
            "Cell semantic route or controller replay evidence is invalid."
        ) from exc
    if (
        replay_evidence.get("protocol_sha256") != protocol.sha256
        or scientific.get("controller_replay_evidence_sha256")
        != replay_evidence.get("sha256")
        or len(replay_evidence.get("signed_controller_round_prefixes", ()))
        != accepted_rounds
    ):
        raise RunnerError("Controller replay evidence binding drifted.")

    terminal_outcome = stop.get("terminal_controller_outcome")
    terminal_receipt_sha256: str | None
    terminal_checkpoint_sha256: str | None
    if terminal_outcome == ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1:
        if (
            accepted_rounds >= TARGET_HORIZON
            or stop.get("primary_reason") != NATURAL_STOP_REASON
        ):
            raise RunnerError("Natural-terminal stop boundary drifted.")
        terminal_receipt = _completion_mapping(
            scientific.get("terminal_phase3_selection_receipt"),
            owner="Phase-III terminal selection receipt",
        )
        terminal_receipt_sha256 = _signed_completion_sha(
            terminal_receipt, owner="Phase-III terminal selection receipt"
        )
        replay_terminal = _completion_mapping(
            replay_evidence.get("phase3_no_positive_terminal"),
            owner="controller replay natural terminal",
        )
        if (
            terminal_receipt.get("accepted_controller_round")
            != accepted_rounds
            or terminal_receipt.get("attempted_controller_round")
            != accepted_rounds + 1
            or terminal_receipt.get("accepted_state_fingerprint")
            != final_state.get("projective_state_fingerprint")
            or terminal_receipt.get("accepted_operator_count")
            != len(final_operators)
            or terminal_receipt.get("accepted_state_unchanged") is not True
            or terminal_receipt.get("final_admission_record_id") is not None
            or replay_evidence.get("terminal_controller_outcome")
            != terminal_outcome
            or replay_terminal.get(
                "terminal_phase3_selection_receipt_sha256"
            )
            != terminal_receipt_sha256
            or replay_terminal.get("accepted_state_sha256")
            != canonical_sha256(final_state)
            or (
                accepted_rounds == 0
                and replay_terminal.get("round_zero_accepted_state")
                != final_state
            )
        ):
            raise RunnerError("Natural-terminal state binding drifted.")
        checkpoint_sha256, _unused = _authenticate_checkpoint(
            checkpoint_path=checkpoint_path,
            problem=problem,
            run_route=run_route,
            final_state=final_state,
            natural_terminal=True,
        )
        terminal_checkpoint_sha256 = str(
            terminal_receipt.get("terminal_active_prefix_checkpoint_sha256")
        )
        if accepted_rounds == 0:
            if summary is not None or run.get("paper_i_summary") is not None:
                raise RunnerError("Round-zero terminal cannot carry a summary.")
            summary_status = "not_applicable_round_zero"
            summary_sha256 = None
        else:
            summary_mapping = _completion_mapping(
                summary, owner="Paper-I summary"
            )
            if (
                run.get("paper_i_summary") != summary_mapping
                or summary_mapping.get("available_controller_rounds")
                != accepted_rounds
            ):
                raise RunnerError("Natural-terminal summary drifted.")
            summary_status = "present"
            summary_sha256 = canonical_sha256(summary_mapping)
        completion_kind = NATURAL_COMPLETION_KIND
        terminal_attempted_round = accepted_rounds + 1
    elif terminal_outcome is None:
        if (
            accepted_rounds != TARGET_HORIZON
            or stop.get("primary_reason") != "maximum_controller_rounds"
            or scientific.get("terminal_phase3_selection_receipt") is not None
            or replay_evidence.get("phase3_no_positive_terminal") is not None
        ):
            raise RunnerError(
                "Nonterminal completion requires an unpadded exact k50 result."
            )
        summary_mapping = _completion_mapping(summary, owner="Paper-I summary")
        if (
            run.get("paper_i_summary") != summary_mapping
            or summary_mapping.get("available_controller_rounds")
            != TARGET_HORIZON
        ):
            raise RunnerError("Maximum-k50 summary drifted.")
        checkpoint_sha256, terminal_checkpoint_sha256 = (
            _authenticate_checkpoint(
                checkpoint_path=checkpoint_path,
                problem=problem,
                run_route=run_route,
                final_state=final_state,
                natural_terminal=False,
            )
        )
        terminal_receipt_sha256 = None
        summary_status = "present"
        summary_sha256 = canonical_sha256(summary_mapping)
        completion_kind = MAXIMUM_COMPLETION_KIND
        terminal_attempted_round = None
    else:
        raise RunnerError("Cell ended under an unauthorized controller outcome.")

    selector_closure_sha256 = _validate_selector_closure(
        scientific,
        accepted_rounds=accepted_rounds,
        terminal_outcome=terminal_outcome,
        terminal_receipt_sha256=terminal_receipt_sha256,
    )
    all_work = _completion_mapping(
        run.get("estimator_accounting", {}).get("all_work"),
        owner="all-executed estimator work",
    )
    accepted_prefix_work = _completion_sequence(
        run.get("canonical_reporting", {}).get("accepted_prefix_work"),
        owner="accepted-prefix work",
    )
    if len(accepted_prefix_work) != accepted_rounds:
        raise RunnerError("Accepted-prefix work cardinality drifted.")
    payload = {
        "schema": CELL_COMPLETION_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "execution_id": cell.execution_id,
        "cell_ordinal": cell.ordinal,
        "completion_kind": completion_kind,
        "maximum_controller_rounds": TARGET_HORIZON,
        "accepted_controller_rounds": accepted_rounds,
        "terminal_attempted_controller_round": terminal_attempted_round,
        "terminal_controller_outcome": terminal_outcome,
        "route_variant": cell.route_variant,
        "route_contract_sha256": route_contract_sha256,
        "protocol_sha256": protocol.sha256,
        "final_state_sha256": canonical_sha256(final_state),
        "final_state_fingerprint": final_state.get(
            "projective_state_fingerprint"
        ),
        "final_energy": float(final_state["energy"]),
        "accepted_trajectory_sha256": canonical_sha256(trajectory),
        "controller_replay_evidence_sha256": replay_evidence["sha256"],
        "selector_accounting_closure_sha256": selector_closure_sha256,
        "terminal_phase3_selection_receipt_sha256": (
            terminal_receipt_sha256
        ),
        "terminal_active_prefix_checkpoint_sha256": (
            terminal_checkpoint_sha256
        ),
        "checkpoint_file_sha256": checkpoint_sha256,
        "paper_i_summary_sha256": summary_sha256,
        "summary_artifact_status": summary_status,
        "all_executed_estimator_work": all_work,
        "accepted_prefix_work_sha256": canonical_sha256(
            accepted_prefix_work
        ),
    }
    return digested(payload)


def run_child(execution_id: str) -> int:
    cell = _cell_by_execution_id(execution_id)
    plan, authorization = validate_authority()
    assert_environment()
    if os.environ.get(CHILD_TOKEN_ENV) != child_token(
        authorization["sha256"], cell
    ):
        raise RunnerError("Child capability is invalid.")
    run_dir, staging, receipt_path, _guard_path = cell_paths(cell)
    if any(
        path.exists() or path.is_symlink()
        for path in (run_dir, staging, receipt_path)
    ):
        raise RunnerError("Cell output is not pristine.")
    staging.mkdir(parents=True)
    checkpoint_path = staging / "checkpoints/current.json"
    ledger_path = staging / "result/estimator_ledger.json"
    observation = SRObservationPolicy(
        checkpoint=CheckpointObservation(
            path=checkpoint_path,
            every_controller_rounds=1,
            keep_history_tail=TARGET_HORIZON,
        ),
        estimator_ledger=EstimatorLedgerObservation(path=ledger_path),
        resource_rounds=tuple(range(1, TARGET_HORIZON + 1)),
    )
    problem, protocol = _materialize_cell(cell)
    expected_binding = next(
        row
        for row in plan["protocol_bindings"]
        if row["execution_id"] == execution_id
    )
    if _protocol_binding(cell) != expected_binding:
        raise RunnerError("Cell protocol drifted immediately before execution.")
    result = run_ra_adapt(
        problem,
        protocol,
        operational_controls=RAAdaptOperationalControls(
            maximum_controller_rounds=TARGET_HORIZON,
            resume=FreshStart(),
            observation=observation,
        ),
    )
    result_mapping = result.to_dict()
    summary_mapping = (
        result.run.paper_i_summary.to_dict()
        if result.run.paper_i_summary is not None
        else None
    )
    completion = validate_cell_completion(
        cell,
        result=result_mapping,
        summary=summary_mapping,
        checkpoint_path=checkpoint_path,
    )
    if _protocol_binding(cell) != expected_binding:
        raise RunnerError("Cell protocol drifted during scientific execution.")
    rounds = int(completion["accepted_controller_rounds"])
    result_path = staging / "result/result.json"
    summary_path = staging / "summary/summary.json"
    write_json_exclusive(result_path, result_mapping)
    if summary_mapping is not None:
        write_json_exclusive(summary_path, summary_mapping)
    artifact_paths = {
        "checkpoint": checkpoint_path,
        "estimator_ledger": ledger_path,
        "result": result_path,
        **({"summary": summary_path} if summary_mapping is not None else {}),
    }
    artifacts = {
        role: _artifact_binding(path, staging)
        for role, path in artifact_paths.items()
    }
    manifest = digested(
        {
            "schema": MANIFEST_SCHEMA,
            "status": "passed_maximum_k50",
            "campaign_id": CAMPAIGN_ID,
            "execution_id": execution_id,
            "cell": asdict(cell),
            "plan_sha256": plan["sha256"],
            "authorization_sha256": authorization["sha256"],
            "source_implementation_inventory_sha256": plan[
                "source_implementation_inventory_sha256"
            ],
            "protocol_binding": expected_binding,
            "maximum_controller_rounds": TARGET_HORIZON,
            "controller_rounds_completed": rounds,
            "cell_completion": completion,
            "execution_path_canary": {
                "is_canary_cell": cell.ordinal == 1,
                "accepted_round": (
                    1 if cell.ordinal == 1 and rounds >= 1 else None
                ),
                "round_zero_terminal": (
                    cell.ordinal == 1 and rounds == 0
                ),
                "continued_same_trajectory_to_completion": (
                    cell.ordinal == 1
                ),
            },
            "artifacts": artifacts,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    write_json_exclusive(staging / "execution_manifest.json", manifest)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staging, run_dir)
    worker = digested(
        {
            "schema": WORKER_SCHEMA,
            "status": "passed_maximum_k50",
            "campaign_id": CAMPAIGN_ID,
            "execution_id": execution_id,
            "manifest_sha256": manifest["sha256"],
            "artifact_inventory": [
                _artifact_binding(path, RUNTIME_ROOT)
                for path in sorted(run_dir.rglob("*"))
                if path.is_file()
            ],
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    write_json_exclusive(receipt_path, worker)
    return 0


def total_rss(process: psutil.Process) -> int:
    total = 0
    try:
        candidates = (process, *process.children(recursive=True))
    except psutil.Error:
        candidates = (process,)
    for candidate in candidates:
        try:
            total += int(candidate.memory_info().rss)
        except psutil.Error:
            pass
    return total


def terminate_process_group(child: subprocess.Popen[Any]) -> None:
    """Gracefully stop the isolated child process group."""

    try:
        os.killpg(os.getpgid(child.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    except (AttributeError, OSError):
        child.terminate()


def validate_worker_receipt(
    cell: CellSpec, worker: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the complete immutable identity of a successful worker."""

    _require_exact_keys(worker, WORKER_RECEIPT_KEYS, label="Worker")
    manifest_sha = worker.get("manifest_sha256")
    inventory = worker.get("artifact_inventory")
    valid_inventory = isinstance(inventory, list) and bool(inventory)
    if valid_inventory:
        paths: list[str] = []
        for row in inventory:
            if (
                not isinstance(row, Mapping)
                or set(row) != {"path", "sha256", "size_bytes"}
                or not isinstance(row.get("path"), str)
                or not str(row["path"]).startswith(
                    f"runs/{cell.execution_id}/"
                )
                or not isinstance(row.get("size_bytes"), int)
                or isinstance(row.get("size_bytes"), bool)
                or int(row["size_bytes"]) < 0
                or not isinstance(row.get("sha256"), str)
                or len(str(row["sha256"])) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in str(row["sha256"])
                )
            ):
                valid_inventory = False
                break
            paths.append(str(row["path"]))
        valid_inventory = valid_inventory and len(paths) == len(set(paths))
    if (
        worker.get("schema") != WORKER_SCHEMA
        or worker.get("status") != "passed_maximum_k50"
        or worker.get("campaign_id") != CAMPAIGN_ID
        or worker.get("execution_id") != cell.execution_id
        or not isinstance(manifest_sha, str)
        or len(manifest_sha) != 64
        or any(character not in "0123456789abcdef" for character in manifest_sha)
        or not valid_inventory
        or worker.get("submission_authorized") is not False
        or worker.get("paper_adoption_authorized") is not False
        or worker.get("paper_evidence_adoption_authorized") is not False
    ):
        raise RunnerError(f"Worker receipt identity drifted: {cell.execution_id}")
    return dict(worker)


def _log_file_binding(cell: CellSpec) -> dict[str, Any]:
    path = cell_log_path(cell)
    observed = path.lstat()
    if not stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        raise RunnerError(f"Cell log is not a plain file: {path}")
    return {
        "path": f"cell_logs/{cell.execution_id}.log",
        "sha256": sha256_file(path),
        "size_bytes": observed.st_size,
    }


def _attempt_inventory(cell: CellSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope, root in zip(("run", "staging"), cell_paths(cell)[:2], strict=True):
        if not root.is_dir() or root.is_symlink():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.is_symlink():
                rows.append(
                    {
                        "scope": scope,
                        "path": path.relative_to(root).as_posix(),
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
    return rows


def _close_cell_log(state: Mapping[str, Any]) -> None:
    stream = state.get("log_stream")
    if stream is None or stream.closed:
        return
    stream.flush()
    os.fsync(stream.fileno())
    stream.close()


def validate_passed_guard_receipt(
    cell: CellSpec,
    guard: Mapping[str, Any],
    *,
    batch: BatchSpec,
    scheduler_decision: Mapping[str, Any],
    worker: Mapping[str, Any],
    require_live_attempt_inventory: bool = True,
) -> dict[str, Any]:
    _require_exact_keys(guard, GUARD_RECEIPT_KEYS, label="Guard")
    mode = str(scheduler_decision["scheduling_mode"])
    observation = guard.get("launch_capacity_observation")
    if not isinstance(observation, Mapping):
        raise RunnerError("Guard launch-capacity observation is absent.")
    validate_launch_capacity_observation(
        (cell,),
        batch=batch,
        scheduling_mode=mode,
        observation=observation,
    ) if mode == "serial_capacity_fallback" else validate_launch_capacity_observation(
        tuple(_cell_by_execution_id(value) for value in batch.execution_ids),
        batch=batch,
        scheduling_mode=mode,
        observation=observation,
    )
    numeric = (
        guard.get("elapsed_seconds"), guard.get("peak_rss_bytes"),
        guard.get("rss_limit_bytes"),
        guard.get("minimum_available_memory_bytes"),
        guard.get("minimum_free_disk_bytes"),
    )
    attempt = guard.get("attempt_inventory")
    valid_attempt = isinstance(attempt, list)
    if valid_attempt:
        identities: list[tuple[str, str]] = []
        for row in attempt:
            if (
                not isinstance(row, Mapping)
                or set(row) != {"scope", "path", "sha256", "size_bytes"}
                or row.get("scope") not in {"run", "staging"}
                or not isinstance(row.get("path"), str)
                or not row.get("path")
                or not isinstance(row.get("sha256"), str)
                or len(str(row["sha256"])) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in str(row["sha256"])
                )
                or not isinstance(row.get("size_bytes"), int)
                or isinstance(row.get("size_bytes"), bool)
                or int(row["size_bytes"]) < 0
            ):
                valid_attempt = False
                break
            identities.append((str(row["scope"]), str(row["path"])))
        valid_attempt = valid_attempt and len(identities) == len(set(identities))
    if (
        guard.get("schema") != GUARD_SCHEMA
        or guard.get("status") != "passed"
        or guard.get("campaign_id") != CAMPAIGN_ID
        or guard.get("execution_id") != cell.execution_id
        or guard.get("batch_ordinal") != batch.ordinal
        or mode not in {"pair", "serial_capacity_fallback"}
        or guard.get("scheduling_mode") != mode
        or guard.get("scheduler_decision_sha256") != scheduler_decision["sha256"]
        or guard.get("launch_capacity_observation_sha256")
        != canonical_sha256(observation)
        or guard.get("returncode") != 0
        or guard.get("stop_reason") is not None
        or guard.get("worker_receipt_sha256") != worker["sha256"]
        or guard.get("log_file_binding") != _log_file_binding(cell)
        or not valid_attempt
        or (
            require_live_attempt_inventory
            and guard.get("attempt_inventory") != _attempt_inventory(cell)
        )
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
            for value in numeric
        )
        or guard.get("rss_limit_bytes") != child_rss_limit_bytes(cell)
        or guard.get("peak_rss_bytes") > guard.get("rss_limit_bytes")
        or guard.get("minimum_available_memory_bytes")
        < AVAILABLE_MEMORY_FLOOR_BYTES
        or guard.get("minimum_free_disk_bytes") < FREE_DISK_FLOOR_BYTES
        or guard.get("submission_authorized") is not False
        or guard.get("paper_adoption_authorized") is not False
        or guard.get("paper_evidence_adoption_authorized") is not False
    ):
        raise RunnerError(f"Cell guard closure drifted: {cell.execution_id}")
    return dict(guard)


def monitor_cells(
    cells: Sequence[CellSpec],
    authorization: Mapping[str, Any],
    *,
    batch: BatchSpec,
    scheduling_mode: str,
    scheduler_decision_sha256: str,
    launch_capacity_observation: Mapping[str, Any],
    popen_factory: Callable[..., Any] = subprocess.Popen,
    process_factory: Callable[[int], Any] = psutil.Process,
    memory_supplier: Callable[[], int] | None = None,
    disk_supplier: Callable[[], int] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    terminate_process_group: Callable[[Any], None] = terminate_process_group,
    status_writer: Callable[[Mapping[str, Any]], None] = write_status,
) -> list[dict[str, Any]]:
    """Run one or two isolated children under independent RSS guards."""

    cells = tuple(cells)
    if (
        not cells
        or len(cells) > MAXIMUM_CONCURRENCY
        or any(cell.execution_id not in batch.execution_ids for cell in cells)
        or scheduling_mode not in {"pair", "serial_capacity_fallback"}
        or (scheduling_mode == "pair" and len(cells) != MAXIMUM_CONCURRENCY)
        or (scheduling_mode == "serial_capacity_fallback" and len(cells) != 1)
        or len(scheduler_decision_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in scheduler_decision_sha256
        )
    ):
        raise RunnerError("Guarded child scheduling request drifted.")
    memory_supplier = memory_supplier or (
        lambda: int(psutil.virtual_memory().available)
    )
    disk_supplier = disk_supplier or (
        lambda: int(shutil.disk_usage(REPO_ROOT).free)
    )
    environment = {**os.environ, **EXPECTED_ENV}
    launch_capacity = validate_launch_capacity_observation(
        cells,
        batch=batch,
        scheduling_mode=scheduling_mode,
        observation=launch_capacity_observation,
    )
    launch_capacity_sha256 = canonical_sha256(launch_capacity)
    states: list[dict[str, Any]] = []
    initial_memory = int(memory_supplier())
    initial_disk = int(disk_supplier())
    def finalize(
        state: dict[str, Any], *, forced_stop_reason: str | None = None
    ) -> dict[str, Any]:
        child = state["child"]
        try:
            returncode = child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(child.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                child.kill()
            returncode = child.wait(timeout=30)
        cell = state["cell"]
        stop_reason = forced_stop_reason or state["stop_reason"]
        _close_cell_log(state)
        worker_path = cell_paths(cell)[2]
        worker_sha: str | None = None
        if returncode == 0 and stop_reason is None:
            if not worker_path.is_file() or worker_path.is_symlink():
                stop_reason = "missing_worker_receipt"
            else:
                try:
                    worker = load_digested(worker_path, schema=WORKER_SCHEMA)
                except RunnerError:
                    stop_reason = "malformed_worker_receipt"
                else:
                    try:
                        validate_worker_receipt(cell, worker)
                    except RunnerError:
                        stop_reason = "malformed_worker_receipt"
                    else:
                        worker_sha = str(worker["sha256"])
        passed = returncode == 0 and stop_reason is None and worker_sha is not None
        guard = digested(
            {
                "schema": GUARD_SCHEMA,
                "status": "passed" if passed else "failed",
                "campaign_id": CAMPAIGN_ID,
                "execution_id": cell.execution_id,
                "batch_ordinal": batch.ordinal,
                "scheduling_mode": scheduling_mode,
                "scheduler_decision_sha256": scheduler_decision_sha256,
                "launch_capacity_observation": launch_capacity,
                "launch_capacity_observation_sha256": launch_capacity_sha256,
                "returncode": returncode,
                "stop_reason": stop_reason,
                "elapsed_seconds": clock() - state["started"],
                "peak_rss_bytes": state["peak_rss"],
                "rss_limit_bytes": child_rss_limit_bytes(cell),
                "minimum_available_memory_bytes": state["minimum_memory"],
                "minimum_free_disk_bytes": state["minimum_disk"],
                "worker_receipt_sha256": worker_sha,
                "log_file_binding": _log_file_binding(cell),
                "attempt_inventory": _attempt_inventory(cell),
                "submission_authorized": False,
                "paper_adoption_authorized": False,
                "paper_evidence_adoption_authorized": False,
            }
        )
        write_json_exclusive(cell_paths(cell)[3], guard)
        state["guard"] = guard
        return guard

    def terminate_and_finalize(
        targets: Sequence[dict[str, Any]], *, reason: str
    ) -> None:
        for state in targets:
            if state["guard"] is not None:
                continue
            if state["child"].poll() is None:
                terminate_process_group(state["child"])
        for state in targets:
            if state["guard"] is not None:
                continue
            try:
                finalize(state, forced_stop_reason=state["stop_reason"] or reason)
            except BaseException:
                # Preserve the original exception while still attempting to reap
                # every remaining child in the outer containment block.
                pass

    # Launch is transactional.  A failure during any Popen/process wrapper
    # construction contains and reaps every process group already created.
    try:
        for cell in cells:
            child_environment = dict(environment)
            child_environment[CHILD_TOKEN_ENV] = child_token(
                authorization["sha256"], cell
            )
            command = [
                sys.executable,
                "-u",
                "-B",
                str(RUNNER_PATH),
                "--child",
                cell.execution_id,
            ]
            log_path = cell_log_path(cell)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_stream = log_path.open("xb")
            try:
                child = popen_factory(
                    command,
                    cwd=REPO_ROOT,
                    env=child_environment,
                    start_new_session=True,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                )
            except BaseException:
                log_stream.flush()
                os.fsync(log_stream.fileno())
                log_stream.close()
                raise
            state = {
                "cell": cell,
                "child": child,
                "process": None,
                "started": clock(),
                "peak_rss": 0,
                "minimum_memory": initial_memory,
                "minimum_disk": initial_disk,
                "stop_reason": None,
                "guard": None,
                "log_stream": log_stream,
            }
            states.append(state)
            state["process"] = process_factory(child.pid)
    except BaseException as exc:
        terminate_and_finalize(states, reason="batch_launch_failed")
        raise BatchExecutionFailed(
            "Guarded batch launch failed and was contained.",
            scheduling_mode=scheduling_mode,
        ) from exc

    active = list(states)
    completed_guards: list[dict[str, Any]] = []
    try:
        while active:
            memory = int(memory_supplier())
            disk = int(disk_supplier())
            finished: list[dict[str, Any]] = []
            failure: dict[str, Any] | None = None
            for state in active:
                child = state["child"]
                cell = state["cell"]
                returncode = child.poll()
                rss = total_rss(state["process"]) if returncode is None else 0
                state["peak_rss"] = max(state["peak_rss"], rss)
                state["minimum_memory"] = min(state["minimum_memory"], memory)
                state["minimum_disk"] = min(state["minimum_disk"], disk)
                if returncode is not None:
                    if returncode != 0:
                        state["stop_reason"] = "returncode_nonzero"
                        failure = state
                    finished.append(state)
                    continue
                if rss > child_rss_limit_bytes(cell):
                    state["stop_reason"] = "rss_limit_breached"
                    failure = state
                elif memory < AVAILABLE_MEMORY_FLOOR_BYTES:
                    state["stop_reason"] = "available_memory_floor_breached"
                    failure = state
                elif disk < FREE_DISK_FLOOR_BYTES:
                    state["stop_reason"] = "free_disk_floor_breached"
                    failure = state

            status_writer(
                {
                    "status": (
                        "running_batch"
                        if failure is None
                        else (
                            "failed_pair"
                            if scheduling_mode == "pair"
                            else "failed_campaign"
                        )
                    ),
                    "batch_ordinal": batch.ordinal,
                    "block": batch.block,
                    "scheduling_mode": scheduling_mode,
                    "scheduler_decision_sha256": scheduler_decision_sha256,
                    "active_execution_ids": [
                        state["cell"].execution_id for state in active
                    ],
                    "children": [
                        {
                            "execution_id": state["cell"].execution_id,
                            "pid": state["child"].pid,
                            "peak_rss_bytes": state["peak_rss"],
                            "rss_limit_bytes": child_rss_limit_bytes(state["cell"]),
                            "stop_reason": state["stop_reason"],
                        }
                        for state in active
                    ],
                    "available_memory_bytes": memory,
                    "free_disk_bytes": disk,
                }
            )
            if failure is not None:
                for state in active:
                    if (
                        state is not failure
                        and state not in finished
                        and state["guard"] is None
                    ):
                        state["stop_reason"] = "sibling_failed"
                for state in finished:
                    if state is failure or state["guard"] is not None:
                        continue
                    finalize(state)
                terminate_and_finalize(active, reason="sibling_failed")
                raise BatchExecutionFailed(
                    f"Batch {batch.ordinal} failed at "
                    f"{failure['cell'].execution_id}; sibling execution stopped.",
                    scheduling_mode=scheduling_mode,
                )
            for state in finished:
                guard = finalize(state)
                if guard["status"] != "passed":
                    for sibling in active:
                        if sibling is not state and sibling["guard"] is None:
                            sibling["stop_reason"] = (
                                "sibling_closure_validation_failed"
                            )
                    terminate_and_finalize(
                        active, reason="sibling_closure_validation_failed"
                    )
                    reason = str(guard["stop_reason"]).replace("_", " ")
                    raise BatchExecutionFailed(
                        f"Cell {state['cell'].execution_id} failed {reason}.",
                        scheduling_mode=scheduling_mode,
                    )
                completed_guards.append(guard)
                active.remove(state)
            if active:
                sleeper(POLL_SECONDS)
    except BaseException:
        terminate_and_finalize(active, reason="supervisor_exception")
        raise
    by_execution = {guard["execution_id"]: guard for guard in completed_guards}
    return [by_execution[cell.execution_id] for cell in cells]


def _adaptive_phase_evidence(
    round_receipt: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    try:
        phases = round_receipt["scored_insertion_position_population"]["phases"]
    except (KeyError, TypeError) as exc:
        raise RunnerError("Accepted round lacks typed Phase-I/II/III receipts.") from exc
    expected_caps = {"phase_i": 24, "phase_ii": 12, "phase_iii": 12}
    expected_score_keys = {
        "phase_i": "phase1_active_score",
        "phase_ii": "phase2_raw_score",
        "phase_iii": "full_v2_score",
    }
    if (
        not isinstance(phases, list)
        or [str(row.get("phase")) for row in phases if isinstance(row, Mapping)]
        != ["phase_i", "phase_ii", "phase_iii"]
    ):
        raise RunnerError(
            "Adaptive Phase-I/II/III receipts are not exact, unique, and ordered."
        )
    observed: dict[str, dict[str, Any]] = {}
    for phase_row in phases:
        phase = str(phase_row.get("phase"))
        if phase not in expected_caps:
            raise RunnerError(
                "Accepted round lacks an adaptive shortlist receipt for every phase."
            )
        try:
            mapped = adaptive_phase_selection_receipt_from_mapping(
                phase_row,
                expected_phase=phase,
                expected_score_key=expected_score_keys[phase],
                expected_hard_cap=expected_caps[phase],
                expected_frontier_ratio=0.9,
            )
        except ValueError as exc:
            raise RunnerError(
                f"Adaptive live selection failed deep mapping for {phase}."
            ) from exc
        shortlist_records = phase_row.get("shortlist_records")
        if not isinstance(shortlist_records, list):
            raise RunnerError(f"Adaptive live shortlist evidence is absent for {phase}.")
        input_count = int(mapped.population_count)
        retained_count = int(mapped.adaptive_retained_count)
        final_record_id = mapped.final_admission_record_id
        if (
            mapped.phase != phase
            or input_count != len(mapped.population_record_ids)
            or not 0 < retained_count <= expected_caps[phase]
            or (
                phase == "phase_iii"
                and (
                    mapped.final_singleton_count != 1
                    or len(mapped.shortlist_record_ids) != 1
                    or str(final_record_id) != mapped.shortlist_record_ids[0]
                )
            )
            or (phase != "phase_iii" and final_record_id is not None)
        ):
            raise RunnerError(f"Adaptive shortlist receipt drifted for {phase}.")
        observed[phase] = {
            "input_count": input_count,
            "population_record_ids": list(mapped.population_record_ids),
            "adaptive_retained_count": retained_count,
            "final_singleton_count": mapped.final_singleton_count,
            "final_record_id": (
                str(final_record_id) if phase == "phase_iii" else None
            ),
            "final_generator_id": (
                str(shortlist_records[0]["generator_id"])
                if phase == "phase_iii"
                else None
            ),
            "final_insertion_position": (
                int(shortlist_records[0]["insertion_position"])
                if phase == "phase_iii"
                else None
            ),
            "adaptive_receipt_sha256": mapped.adaptive_shortlist.sha256,
            "selection_mapping_sha256": mapped.mapping_sha256,
        }
    if set(observed) != set(expected_caps):
        raise RunnerError("Adaptive Phase-I/II/III receipt set is incomplete.")
    return observed


def validate_reporting_phase0_receipt(
    receipt: Mapping[str, Any],
    *,
    scored_population: Mapping[str, Any],
) -> dict[str, Any]:
    """Accept only the position-record Phase-0 receipt used by this campaign."""

    try:
        validated = validate_semantic_position_phase0_receipt(
            receipt,
            scored_population=scored_population,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RunnerError(
            "Reporting Phase-0 evidence is not a deep-valid position-record receipt."
        ) from exc
    if (
        validated.get("route_variant") != EXPECTED_CORE_ROUTE_VARIANT
        or validated.get("position_aware_gradient_surface") is not True
        or validated.get("generator_level_reexpansion_after_phase0") is not False
    ):
        raise RunnerError("Reporting position-record Phase-0 identity drifted.")
    return validated


def validate_reporting_phase0_phase_i_link(
    phase0: Mapping[str, Any],
    phase_evidence: Mapping[str, Mapping[str, Any]],
    *,
    closure_round: Mapping[str, Any],
) -> str:
    """Prove exact ordered position-record pass-through into Phase I."""

    retained = phase0.get("retained_records")
    phase_i = phase_evidence.get("phase_i")
    if (
        not isinstance(retained, list)
        or not retained
        or any(not isinstance(row, Mapping) for row in retained)
        or not isinstance(phase_i, Mapping)
        or not isinstance(phase_i.get("population_record_ids"), list)
    ):
        raise RunnerError("Position-record Phase-0/Phase-I link is absent.")
    try:
        phase0_ids = [
            adaptive_phase_record_id(
                generator_id=str(row["generator_id"]),
                pool_index=int(row["pool_index"]),
                insertion_position=int(row["insertion_position"]),
            )
            for row in retained
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise RunnerError("Position-record Phase-0 identity is malformed.") from exc
    phase_i_ids = [str(value) for value in phase_i["population_record_ids"]]
    if (
        len(set(phase0_ids)) != len(phase0_ids)
        or len(set(phase_i_ids)) != len(phase_i_ids)
        or len(phase_i_ids) != len(phase0_ids)
        or set(phase_i_ids) != set(phase0_ids)
    ):
        raise RunnerError(
            "Position-record Phase-0 was not passed directly to Phase I."
        )
    link_sha256 = canonical_sha256(
        {
            "phase0_retained_record_ids": phase0_ids,
            "phase_i_population_record_ids": phase_i_ids,
        }
    )
    if closure_round.get(
        "phase0_phase_i_direct_population_link_sha256"
    ) != link_sha256:
        raise RunnerError(
            "Stored Phase-0/Phase-I direct-population link digest drifted."
        )
    return link_sha256


def _reporting_selector_closure_rounds(
    result: Mapping[str, Any],
) -> dict[int, Mapping[str, Any]]:
    """Validate the self-digesting final selector closure used by reporting."""

    try:
        closure = dict(
            result["scientific_receipts"][
                "semantic_selector_accounting_closure"
            ]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RunnerError("Semantic selector accounting closure is absent.") from exc
    observed_sha = closure.pop("sha256", None)
    rounds = closure.get("rounds")
    accepted_count = len(
        result.get("scientific_receipts", {}).get(
            "accepted_round_receipts", ()
        )
    )
    if (
        observed_sha != canonical_sha256(closure)
        or closure.get("schema")
        != "paper_i_ra_semantic_final_selector_accounting_closure_v1"
        or closure.get("route_variant") != EXPECTED_CORE_ROUTE_VARIANT
        or closure.get("validated_round_count") != accepted_count
        or not isinstance(rounds, list)
        or len(rounds) != accepted_count
        or any(not isinstance(row, Mapping) for row in rounds)
        or [row.get("accepted_round") for row in rounds]
        != list(range(1, accepted_count + 1))
    ):
        raise RunnerError("Semantic selector accounting closure drifted.")
    return {int(row["accepted_round"]): row for row in rounds}


def reporting_plateau_state(
    cell: CellSpec,
    round_receipt: Mapping[str, Any],
) -> str:
    """Project plateau activation without silently coercing malformed evidence."""

    if cell.insertion_policy != "plateau_commutation":
        return "append_only"
    plateau = round_receipt.get("insertion_commutation_plateau")
    domain_open = plateau.get("domain_open") if isinstance(plateau, Mapping) else None
    if type(domain_open) is not bool:
        raise RunnerError("Authenticated plateau state is missing or malformed.")
    return "open" if domain_open else "closed"


def _adaptive_phase_counts(
    round_receipt: Mapping[str, Any]
) -> dict[str, tuple[int, int]]:
    evidence = _adaptive_phase_evidence(round_receipt)
    return {
        phase: (
            int(row["input_count"]),
            int(row["adaptive_retained_count"]),
        )
        for phase, row in evidence.items()
    }


def report_rows(
    cell: CellSpec,
    result: Mapping[str, Any],
    summary: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    accepted = result["scientific_receipts"]["accepted_round_receipts"]
    replay = result["run"]["scientific_replay"]
    accepted_count = len(accepted)
    if accepted_count == 0:
        if (
            summary is not None
            or replay
            or result["run"]["accepted_transitions"]
        ):
            raise RunnerError("Round-zero report inputs are not empty.")
        _reporting_selector_closure_rounds(result)
        return []
    if not isinstance(summary, Mapping):
        raise RunnerError("Accepted cell report requires a Paper-I summary.")
    errors = summary["accepted_error_trace"]
    requested = {
        int(row["controller_round"]): row for row in summary["requested_rounds"]
    }
    transitions = {
        int(row["controller_round"]): row
        for row in result["run"]["accepted_transitions"]
    }
    selector_closure_rounds = _reporting_selector_closure_rounds(result)
    if not (
        len(accepted)
        == len(replay)
        == len(errors)
        == accepted_count
    ):
        raise RunnerError("Cell report inputs do not close at accepted depth.")
    rows: list[dict[str, Any]] = []
    for ordinal, (round_receipt, replay_row, error_row) in enumerate(
        zip(accepted, replay, errors, strict=True), start=1
    ):
        phase0 = round_receipt["ra_gradient_phase0_shortlist"]
        phase0 = validate_reporting_phase0_receipt(
            phase0,
            scored_population=round_receipt[
                "scored_insertion_position_population"
            ],
        )
        phase_evidence = _adaptive_phase_evidence(round_receipt)
        validate_reporting_phase0_phase_i_link(
            phase0,
            phase_evidence,
            closure_round=selector_closure_rounds[ordinal],
        )
        prefix = requested[ordinal]
        resources = prefix.get("resources") or {}
        transition = transitions[ordinal]
        if int(transition["controller_round"]) != ordinal:
            raise RunnerError("Accepted transition order drifted.")
        work = prefix["algorithmic_work"]
        phase_iii = phase_evidence["phase_iii"]
        if (
            phase_iii["final_generator_id"] != str(replay_row["generator_id"])
            or phase_iii["final_insertion_position"]
            != int(replay_row["selected_position"])
        ):
            raise RunnerError(
                "Phase-III final singleton diverged from the accepted replay winner."
            )
        rows.append(
            {
                "execution_id": cell.execution_id,
                "cell_ordinal": cell.ordinal,
                "block": cell.block,
                "regime_id": cell.regime_id,
                "nph": cell.nph,
                "insertion_policy": cell.insertion_policy,
                "controller_round": ordinal,
                "energy": float(error_row["accepted_energy"]),
                "absolute_delta_e": float(error_row["absolute_energy_error"]),
                "plateau_state": reporting_plateau_state(cell, round_receipt),
                "phase0_population_count": int(phase0["input_candidate_count"]),
                "phase0_retained_count": int(phase0["retained_candidate_count"]),
                "phase_i_input_count": phase_evidence["phase_i"]["input_count"],
                "phase_i_retained_count": phase_evidence["phase_i"][
                    "adaptive_retained_count"
                ],
                "phase_ii_input_count": phase_evidence["phase_ii"]["input_count"],
                "phase_ii_retained_count": phase_evidence["phase_ii"][
                    "adaptive_retained_count"
                ],
                "phase_iii_input_count": phase_iii["input_count"],
                "phase_iii_adaptive_retained_count": phase_iii[
                    "adaptive_retained_count"
                ],
                "phase_iii_final_singleton_count": phase_iii[
                    "final_singleton_count"
                ],
                "phase_iii_final_record_id": phase_iii["final_record_id"],
                "selected_generator": str(replay_row["generator_id"]),
                "selected_operator": str(replay_row["selected_operator"]),
                "selected_position": int(replay_row["selected_position"]),
                "s_alg": int(work["s_alg"]),
                "n2q": int(resources["compiled_two_qubit_count"]),
                "d2q": int(resources["compiled_two_qubit_depth"]),
                "dc": int(resources["compiled_total_depth"]),
                "checkpoint_sha256": str(
                    replay_row["checkpoint"]["checkpoint_sha256"]
                ),
            }
        )
    return rows


def _protocol_binding_for_cell(
    plan: Mapping[str, Any], cell: CellSpec
) -> Mapping[str, Any]:
    bindings = plan.get("protocol_bindings")
    if not isinstance(bindings, list):
        raise RunnerError("Plan protocol bindings are absent.")
    matches = [
        row
        for row in bindings
        if isinstance(row, Mapping)
        and row.get("execution_id") == cell.execution_id
    ]
    if len(matches) != 1:
        raise RunnerError("Cell protocol binding is not unique.")
    return dict(matches[0])


def _validate_compact_rows(
    cell: CellSpec, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    normalized = [dict(row) for row in rows]
    if (
        len(normalized) != TARGET_HORIZON
        or [row.get("controller_round") for row in normalized]
        != list(range(1, TARGET_HORIZON + 1))
    ):
        raise RunnerError("Cell compact report is not exact ordered k=1..50.")
    expected_keys = set(REPORT_ROW_FIELDS)
    for row in normalized:
        integer_counts = (
            "phase0_population_count",
            "phase0_retained_count",
            "phase_i_input_count",
            "phase_i_retained_count",
            "phase_ii_input_count",
            "phase_ii_retained_count",
            "phase_iii_input_count",
            "phase_iii_adaptive_retained_count",
            "s_alg",
            "n2q",
            "d2q",
            "dc",
        )
        checkpoint = str(row.get("checkpoint_sha256", ""))
        if (
            set(row) != expected_keys
            or row.get("execution_id") != cell.execution_id
            or row.get("cell_ordinal") != cell.ordinal
            or row.get("block") != cell.block
            or row.get("regime_id") != cell.regime_id
            or row.get("nph") != cell.nph
            or row.get("insertion_policy") != cell.insertion_policy
            or row.get("plateau_state") not in {"open", "closed"}
            or row.get("phase_iii_final_singleton_count") != 1
            or any(
                isinstance(row.get(key), bool)
                or not isinstance(row.get(key), int)
                or int(row[key]) < 0
                for key in integer_counts
            )
            or not isinstance(row.get("selected_position"), int)
            or len(checkpoint) != 64
            or any(character not in "0123456789abcdef" for character in checkpoint)
        ):
            raise RunnerError("Cell compact report row drifted.")
    return normalized


def build_compact_cell_receipt(
    cell: CellSpec,
    *,
    rows: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    manifest_sha256: str,
    manifest_file_binding: Mapping[str, Any],
    worker_receipt_sha256: str,
    guard_receipt_sha256: str,
    scheduler_decision_sha256: str,
    artifact_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    normalized_rows = _validate_compact_rows(cell, rows)
    receipt = digested(
        {
            "schema": COMPACT_CELL_SCHEMA,
            "status": "passed_exact_k50_reporting_projection",
            "created_at": serial_runtime.utc_now(),
            "campaign_id": CAMPAIGN_ID,
            "execution_id": cell.execution_id,
            "cell": asdict(cell),
            "target_horizon": TARGET_HORIZON,
            "plan_sha256": plan["sha256"],
            "authorization_sha256": authorization["sha256"],
            "source_implementation_inventory_sha256": plan[
                "source_implementation_inventory_sha256"
            ],
            "protocol_binding": _protocol_binding_for_cell(plan, cell),
            "manifest_sha256": str(manifest_sha256),
            "manifest_file_binding": dict(manifest_file_binding),
            "worker_receipt_sha256": str(worker_receipt_sha256),
            "guard_receipt_sha256": str(guard_receipt_sha256),
            "scheduler_decision_sha256": str(scheduler_decision_sha256),
            "log_file_binding": _log_file_binding(cell),
            "artifact_bindings": {
                key: dict(value) for key, value in artifact_bindings.items()
            },
            "rows": normalized_rows,
            "rows_sha256": canonical_sha256(normalized_rows),
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    return validate_compact_cell_receipt(
        cell,
        receipt,
        plan=plan,
        authorization=authorization,
        manifest_sha256=manifest_sha256,
        manifest_file_binding=manifest_file_binding,
        worker_receipt_sha256=worker_receipt_sha256,
        guard_receipt_sha256=guard_receipt_sha256,
        scheduler_decision_sha256=scheduler_decision_sha256,
        artifact_bindings=artifact_bindings,
    )


def validate_compact_cell_receipt(
    cell: CellSpec,
    receipt: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    manifest_sha256: str,
    manifest_file_binding: Mapping[str, Any],
    worker_receipt_sha256: str,
    guard_receipt_sha256: str,
    scheduler_decision_sha256: str,
    artifact_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    value = dict(receipt)
    _require_exact_keys(value, COMPACT_RECEIPT_KEYS, label="Compact")
    observed_sha = value.pop("sha256", None)
    rows = value.get("rows")
    artifacts = value.get("artifact_bindings")
    if not isinstance(rows, list) or not isinstance(artifacts, Mapping):
        raise RunnerError("Cell compact receipt payload is incomplete.")
    normalized_rows = _validate_compact_rows(cell, rows)
    expected_artifacts = {
        key: dict(binding) for key, binding in artifact_bindings.items()
    }
    required_roles = {"checkpoint", "estimator_ledger", "result", "summary"}
    valid_artifacts = set(artifacts) == required_roles and artifacts == expected_artifacts
    if (
        observed_sha != canonical_sha256(value)
        or not _valid_created_at(value)
        or value.get("schema") != COMPACT_CELL_SCHEMA
        or value.get("status") != "passed_exact_k50_reporting_projection"
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("execution_id") != cell.execution_id
        or value.get("cell") != asdict(cell)
        or value.get("target_horizon") != TARGET_HORIZON
        or value.get("plan_sha256") != plan["sha256"]
        or value.get("authorization_sha256") != authorization["sha256"]
        or value.get("source_implementation_inventory_sha256")
        != plan["source_implementation_inventory_sha256"]
        or value.get("protocol_binding")
        != _protocol_binding_for_cell(plan, cell)
        or value.get("manifest_sha256") != manifest_sha256
        or value.get("manifest_file_binding") != dict(manifest_file_binding)
        or value.get("worker_receipt_sha256") != worker_receipt_sha256
        or value.get("guard_receipt_sha256") != guard_receipt_sha256
        or value.get("scheduler_decision_sha256")
        != scheduler_decision_sha256
        or value.get("log_file_binding") != _log_file_binding(cell)
        or value.get("rows_sha256") != canonical_sha256(normalized_rows)
        or not valid_artifacts
        or value.get("submission_authorized") is not False
        or value.get("paper_adoption_authorized") is not False
        or value.get("paper_evidence_adoption_authorized") is not False
    ):
        raise RunnerError("Cell compact receipt drifted.")
    return dict(receipt)


def load_closed_cell(
    cell: CellSpec,
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    scheduler_decision: Mapping[str, Any] | None = None,
) -> tuple[CellSpec, Mapping[str, Any], Mapping[str, Any], str, str]:
    run_dir, staging, worker_path, guard_path = cell_paths(cell)
    if staging.exists() or staging.is_symlink():
        raise RunnerError(f"Cell has a preserved partial attempt: {cell.execution_id}")
    manifest_path = run_dir / "execution_manifest.json"
    manifest = load_digested(manifest_path, schema=MANIFEST_SCHEMA)
    worker = load_digested(worker_path, schema=WORKER_SCHEMA)
    guard = load_digested(guard_path, schema=GUARD_SCHEMA)
    if scheduler_decision is None:
        batch = _batch_for_cell(cell)
        scheduler_decision = validate_scheduler_decision(
            batch,
            load_digested(scheduler_decision_path(batch), schema=SCHEDULER_SCHEMA),
            plan=plan,
            authorization=authorization,
        )
    else:
        batch = _batch_for_cell(cell)
        scheduler_decision = validate_scheduler_decision(
            batch,
            scheduler_decision,
            plan=plan,
            authorization=authorization,
        )
    expected_binding = next(
        row
        for row in plan["protocol_bindings"]
        if row["execution_id"] == cell.execution_id
    )
    if (
        manifest.get("status") != "passed_maximum_k50"
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("execution_id") != cell.execution_id
        or manifest.get("cell") != asdict(cell)
        or manifest.get("plan_sha256") != plan["sha256"]
        or manifest.get("authorization_sha256") != authorization["sha256"]
        or manifest.get("source_implementation_inventory_sha256")
        != plan["source_implementation_inventory_sha256"]
        or manifest.get("protocol_binding") != expected_binding
        or manifest.get("maximum_controller_rounds") != TARGET_HORIZON
        or manifest.get("submission_authorized") is not False
        or manifest.get("paper_adoption_authorized") is not False
        or manifest.get("paper_evidence_adoption_authorized") is not False
    ):
        raise RunnerError(f"Cell manifest drifted: {cell.execution_id}")
    manifest_artifacts = manifest.get("artifacts")
    if not isinstance(manifest_artifacts, Mapping):
        raise RunnerError(f"Cell artifact inventory is absent: {cell.execution_id}")
    for binding in manifest_artifacts.values():
        artifact = run_dir / str(binding["path"])
        if binding != _artifact_binding(artifact, run_dir):
            raise RunnerError(f"Cell artifact drifted: {artifact}")
    expected_inventory = [
        _artifact_binding(path, RUNTIME_ROOT)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
    ]
    validate_worker_receipt(cell, worker)
    if (
        worker.get("manifest_sha256") != manifest["sha256"]
        or worker.get("artifact_inventory") != expected_inventory
    ):
        raise RunnerError(f"Cell worker closure drifted: {cell.execution_id}")
    validate_passed_guard_receipt(
        cell,
        guard,
        batch=batch,
        scheduler_decision=scheduler_decision,
        worker=worker,
    )
    with (run_dir / "result/result.json").open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    summary_binding = manifest_artifacts.get("summary")
    if summary_binding is None:
        summary = None
    else:
        with (run_dir / "summary/summary.json").open(
            "r", encoding="utf-8"
        ) as stream:
            summary = json.load(stream)
    completion = validate_cell_completion(
        cell,
        result=result,
        summary=summary,
        checkpoint_path=run_dir / str(manifest_artifacts["checkpoint"]["path"]),
    )
    accepted_rounds = int(completion["accepted_controller_rounds"])
    expected_roles = {"checkpoint", "estimator_ledger", "result"}
    if accepted_rounds > 0:
        expected_roles.add("summary")
    if (
        manifest.get("cell_completion") != completion
        or manifest.get("controller_rounds_completed") != accepted_rounds
        or set(manifest_artifacts) != expected_roles
        or manifest.get("execution_path_canary")
        != {
            "is_canary_cell": cell.ordinal == 1,
            "accepted_round": (
                1 if cell.ordinal == 1 and accepted_rounds >= 1 else None
            ),
            "round_zero_terminal": (
                cell.ordinal == 1 and accepted_rounds == 0
            ),
            "continued_same_trajectory_to_completion": cell.ordinal == 1,
        }
    ):
        raise RunnerError(f"Cell completion manifest drifted: {cell.execution_id}")
    if len(report_rows(cell, result, summary)) != accepted_rounds:
        raise RunnerError(
            f"Cell report does not close at accepted depth: {cell.execution_id}"
        )
    return cell, result, summary, worker["sha256"], guard["sha256"]


def _external_archive_row(name: str, path: Path) -> dict[str, Any]:
    observed = path.lstat()
    if not stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        raise RunnerError(f"Archive evidence member is not a plain file: {path}")
    return {
        "mode": stat.S_IMODE(observed.st_mode),
        "path": name,
        "sha256": sha256_file(path),
        "size_bytes": observed.st_size,
    }


def _persistent_cell_evidence(
    cell: CellSpec,
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    scheduler_decision: Mapping[str, Any],
) -> dict[str, Any]:
    batch = _batch_for_cell(cell)
    scheduler = validate_scheduler_decision(
        batch,
        scheduler_decision,
        plan=plan,
        authorization=authorization,
    )
    worker_path = cell_paths(cell)[2]
    guard_path = cell_paths(cell)[3]
    compact_path = compact_cell_receipt_path(cell)
    worker = load_digested(worker_path, schema=WORKER_SCHEMA)
    guard = load_digested(guard_path, schema=GUARD_SCHEMA)
    compact = load_digested(compact_path, schema=COMPACT_CELL_SCHEMA)
    validate_worker_receipt(cell, worker)
    validate_passed_guard_receipt(
        cell,
        guard,
        batch=batch,
        scheduler_decision=scheduler,
        worker=worker,
        require_live_attempt_inventory=False,
    )
    artifact_bindings = compact.get("artifact_bindings")
    if not isinstance(artifact_bindings, Mapping):
        raise RunnerError("Persistent compact artifact bindings are absent.")
    compact = validate_compact_cell_receipt(
        cell,
        compact,
        plan=plan,
        authorization=authorization,
        manifest_sha256=str(compact.get("manifest_sha256", "")),
        manifest_file_binding=dict(compact.get("manifest_file_binding", {})),
        worker_receipt_sha256=worker["sha256"],
        guard_receipt_sha256=guard["sha256"],
        scheduler_decision_sha256=scheduler["sha256"],
        artifact_bindings={
            str(role): dict(binding)
            for role, binding in artifact_bindings.items()
            if isinstance(binding, Mapping)
        },
    )
    return {
        "worker": worker,
        "guard": guard,
        "compact": compact,
        "paths": {
            "worker": worker_path,
            "guard": guard_path,
            "compact": compact_path,
        },
    }


def publish_compact_cell_receipt(
    cell: CellSpec,
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    scheduler_decision: Mapping[str, Any],
) -> dict[str, Any]:
    batch = _batch_for_cell(cell)
    scheduler = validate_scheduler_decision(
        batch,
        scheduler_decision,
        plan=plan,
        authorization=authorization,
    )
    _cell, result, summary, worker_sha, guard_sha = load_closed_cell(
        cell,
        plan=plan,
        authorization=authorization,
        scheduler_decision=scheduler,
    )
    run_dir = cell_paths(cell)[0]
    manifest = load_digested(
        run_dir / "execution_manifest.json", schema=MANIFEST_SCHEMA
    )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RunnerError("Direct manifest artifact bindings are absent.")
    artifact_bindings = {
        str(role): dict(binding)
        for role, binding in artifacts.items()
        if isinstance(binding, Mapping)
    }
    manifest_file_binding = _artifact_binding(
        run_dir / "execution_manifest.json", run_dir
    )
    path = compact_cell_receipt_path(cell)
    if path.is_file():
        return validate_compact_cell_receipt(
            cell,
            load_digested(path, schema=COMPACT_CELL_SCHEMA),
            plan=plan,
            authorization=authorization,
            manifest_sha256=manifest["sha256"],
            manifest_file_binding=manifest_file_binding,
            worker_receipt_sha256=worker_sha,
            guard_receipt_sha256=guard_sha,
            scheduler_decision_sha256=scheduler["sha256"],
            artifact_bindings=artifact_bindings,
        )
    compact = build_compact_cell_receipt(
        cell,
        rows=report_rows(cell, result, summary),
        plan=plan,
        authorization=authorization,
        manifest_sha256=manifest["sha256"],
        manifest_file_binding=manifest_file_binding,
        worker_receipt_sha256=worker_sha,
        guard_receipt_sha256=guard_sha,
        scheduler_decision_sha256=scheduler["sha256"],
        artifact_bindings=artifact_bindings,
    )
    write_json_exclusive(path, compact)
    return compact


def _archive_metadata(
    cell: CellSpec,
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    scheduler_decision: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    worker = evidence["worker"]
    guard = evidence["guard"]
    compact = evidence["compact"]
    authority = {
        "campaign_id": CAMPAIGN_ID,
        "plan_sha256": plan["sha256"],
        "authorization_sha256": authorization["sha256"],
        "runner_sha256": plan["runner"]["sha256"],
        "source_implementation_inventory_sha256": plan[
            "source_implementation_inventory_sha256"
        ],
        "execution_authorized": True,
        "submission_authorized": False,
        "paper_adoption_authorized": False,
        "paper_evidence_adoption_authorized": False,
    }
    metadata = {
        "execution_id": cell.execution_id,
        "cell": asdict(cell),
        "target_horizon": TARGET_HORIZON,
        "protocol_binding": _protocol_binding_for_cell(plan, cell),
        "manifest_sha256": compact["manifest_sha256"],
        "manifest_file_binding": compact["manifest_file_binding"],
        "worker_receipt_sha256": worker["sha256"],
        "guard_receipt_sha256": guard["sha256"],
        "compact_receipt_sha256": compact["sha256"],
        "compact_rows_sha256": compact["rows_sha256"],
        "scheduler_decision_sha256": scheduler_decision["sha256"],
        "log_file_binding": compact["log_file_binding"],
        "artifact_bindings": compact["artifact_bindings"],
    }
    paths = evidence["paths"]
    external = {
        "evidence/worker_receipt.json": Path(paths["worker"]),
        "evidence/guard_receipt.json": Path(paths["guard"]),
        "evidence/compact_reporting_receipt.json": Path(paths["compact"]),
        "evidence/cell.log": cell_log_path(cell),
    }
    return authority, metadata, external


def _validate_archive_manifest_crossbindings(
    cell: CellSpec,
    *,
    paths: Any,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    archive_manifest = load_digested(
        paths.archive_manifest_path, schema=strict_archive.ARCHIVE_SCHEMA
    )
    worker = evidence["worker"]
    compact = evidence["compact"]
    external_paths = evidence["paths"]
    expected_external = sorted(
        (
            _external_archive_row("evidence/worker_receipt.json", external_paths["worker"]),
            _external_archive_row("evidence/guard_receipt.json", external_paths["guard"]),
            _external_archive_row(
                "evidence/compact_reporting_receipt.json", external_paths["compact"]
            ),
            _external_archive_row("evidence/cell.log", cell_log_path(cell)),
        ),
        key=lambda row: str(row["path"]),
    )
    if archive_manifest.get("external_members") != expected_external:
        raise RunnerError("Archived external evidence bindings drifted.")
    source_tree = archive_manifest.get("source_tree")
    inventory = worker.get("artifact_inventory")
    if not isinstance(source_tree, Mapping) or not isinstance(inventory, list):
        raise RunnerError("Archived source inventory evidence is absent.")
    tree_rows = source_tree.get("files")
    if not isinstance(tree_rows, list):
        raise RunnerError("Archived source file inventory is absent.")
    tree_by_path = {
        str(row.get("path")): dict(row)
        for row in tree_rows
        if isinstance(row, Mapping)
    }
    runtime_prefix = f"runs/{cell.execution_id}/"
    worker_by_path: dict[str, dict[str, Any]] = {}
    for raw in inventory:
        if not isinstance(raw, Mapping):
            raise RunnerError("Worker artifact inventory row is malformed.")
        runtime_path = str(raw.get("path", ""))
        if not runtime_path.startswith(runtime_prefix):
            raise RunnerError("Worker artifact inventory escaped the cell tree.")
        relative = runtime_path[len(runtime_prefix) :]
        if relative in worker_by_path:
            raise RunnerError("Worker artifact inventory is not unique.")
        worker_by_path[relative] = dict(raw)
    attempt_rows = evidence["guard"].get("attempt_inventory")
    if not isinstance(attempt_rows, list):
        raise RunnerError("Guard attempt inventory is absent from archive evidence.")
    attempt_run = {
        str(row["path"]): dict(row)
        for row in attempt_rows
        if isinstance(row, Mapping) and row.get("scope") == "run"
    }
    if set(attempt_run) != set(worker_by_path) or any(
        {
            "sha256": attempt_run[path].get("sha256"),
            "size_bytes": attempt_run[path].get("size_bytes"),
        }
        != {
            "sha256": worker_by_path[path].get("sha256"),
            "size_bytes": worker_by_path[path].get("size_bytes"),
        }
        for path in worker_by_path
    ):
        raise RunnerError("Guard attempt inventory detached from worker evidence.")
    if set(worker_by_path) != set(tree_by_path):
        raise RunnerError("Worker and archived source inventories differ.")
    for relative, worker_row in worker_by_path.items():
        tree_row = tree_by_path[relative]
        if (
            tree_row.get("archive_path")
            != f"runs/{cell.execution_id}/{relative}"
            or tree_row.get("sha256") != worker_row.get("sha256")
            or tree_row.get("size_bytes") != worker_row.get("size_bytes")
        ):
            raise RunnerError("Archived source member detached from worker evidence.")
    manifest_row = worker_by_path.get("execution_manifest.json")
    manifest_file_binding = compact.get("manifest_file_binding")
    if (
        manifest_row is None
        or not isinstance(manifest_file_binding, Mapping)
        or {
            key: manifest_row.get(key) for key in ("path", "sha256", "size_bytes")
        }
        != {
            "path": f"runs/{cell.execution_id}/{manifest_file_binding.get('path')}",
            "sha256": manifest_file_binding.get("sha256"),
            "size_bytes": manifest_file_binding.get("size_bytes"),
        }
    ):
        raise RunnerError("Archived execution manifest digest drifted.")
    for binding in compact["artifact_bindings"].values():
        observed = worker_by_path.get(str(binding["path"]))
        if observed is None or {
            key: observed.get(key) for key in ("path", "sha256", "size_bytes")
        } != {
            "path": f"runs/{cell.execution_id}/{binding['path']}",
            "sha256": binding["sha256"],
            "size_bytes": binding["size_bytes"],
        }:
            raise RunnerError("Archived scientific artifact binding drifted.")
    return archive_manifest


def validate_archived_cell_receipt(
    cell: CellSpec,
    receipt: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    scheduler_decision: Mapping[str, Any],
    evidence: Mapping[str, Any],
    archive_backed_closure: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(receipt)
    observed_sha = value.pop("sha256", None)
    expected = {
        "schema": ARCHIVED_CELL_SCHEMA,
        "status": "passed_archive_backed_exact_k50",
        "campaign_id": CAMPAIGN_ID,
        "execution_id": cell.execution_id,
        "cell": asdict(cell),
        "plan_sha256": plan["sha256"],
        "authorization_sha256": authorization["sha256"],
        "source_implementation_inventory_sha256": plan[
            "source_implementation_inventory_sha256"
        ],
        "scheduler_decision_sha256": scheduler_decision["sha256"],
        "worker_receipt_sha256": evidence["worker"]["sha256"],
        "guard_receipt_sha256": evidence["guard"]["sha256"],
        "compact_receipt_sha256": evidence["compact"]["sha256"],
        "compact_rows_sha256": evidence["compact"]["rows_sha256"],
        "archive_backed_closure_sha256": archive_backed_closure["sha256"],
        "archive_closure_receipt_sha256": archive_backed_closure[
            "archive_closure"
        ]["canonical_sha256"],
        "archive": archive_backed_closure["archive"],
        "archive_manifest": archive_backed_closure["archive_manifest"],
        "archive_closure": archive_backed_closure["archive_closure"],
        "rotation_intent": archive_backed_closure["rotation_intent"],
        "cleanup_receipt": archive_backed_closure["cleanup_receipt"],
        "direct_run_tree_absent": True,
        "retiring_tree_absent": True,
        "submission_authorized": False,
        "paper_adoption_authorized": False,
        "paper_evidence_adoption_authorized": False,
    }
    created_at = value.pop("created_at", None)
    if (
        not isinstance(created_at, str)
        or not created_at
        or observed_sha != canonical_sha256({**value, "created_at": created_at})
        or value != expected
    ):
        raise RunnerError("Archived cell receipt drifted.")
    return dict(receipt)


def _publish_archived_cell_receipt(
    cell: CellSpec,
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    scheduler_decision: Mapping[str, Any],
    evidence: Mapping[str, Any],
    archive_backed_closure: Mapping[str, Any],
) -> dict[str, Any]:
    path = archived_cell_receipt_path(cell)
    if path.is_file():
        return validate_archived_cell_receipt(
            cell,
            load_digested(path, schema=ARCHIVED_CELL_SCHEMA),
            plan=plan,
            authorization=authorization,
            scheduler_decision=scheduler_decision,
            evidence=evidence,
            archive_backed_closure=archive_backed_closure,
        )
    payload = digested(
        {
            "schema": ARCHIVED_CELL_SCHEMA,
            "status": "passed_archive_backed_exact_k50",
            "created_at": serial_runtime.utc_now(),
            "campaign_id": CAMPAIGN_ID,
            "execution_id": cell.execution_id,
            "cell": asdict(cell),
            "plan_sha256": plan["sha256"],
            "authorization_sha256": authorization["sha256"],
            "source_implementation_inventory_sha256": plan[
                "source_implementation_inventory_sha256"
            ],
            "scheduler_decision_sha256": scheduler_decision["sha256"],
            "worker_receipt_sha256": evidence["worker"]["sha256"],
            "guard_receipt_sha256": evidence["guard"]["sha256"],
            "compact_receipt_sha256": evidence["compact"]["sha256"],
            "compact_rows_sha256": evidence["compact"]["rows_sha256"],
            "archive_backed_closure_sha256": archive_backed_closure["sha256"],
            "archive_closure_receipt_sha256": archive_backed_closure[
                "archive_closure"
            ]["canonical_sha256"],
            "archive": archive_backed_closure["archive"],
            "archive_manifest": archive_backed_closure["archive_manifest"],
            "archive_closure": archive_backed_closure["archive_closure"],
            "rotation_intent": archive_backed_closure["rotation_intent"],
            "cleanup_receipt": archive_backed_closure["cleanup_receipt"],
            "direct_run_tree_absent": True,
            "retiring_tree_absent": True,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    validate_archived_cell_receipt(
        cell,
        payload,
        plan=plan,
        authorization=authorization,
        scheduler_decision=scheduler_decision,
        evidence=evidence,
        archive_backed_closure=archive_backed_closure,
    )
    write_json_exclusive(path, payload)
    return payload


def archive_and_load_cell(
    cell: CellSpec,
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    scheduler_decision: Mapping[str, Any],
    archive_capacity_waiter: Callable[..., Mapping[str, Any]] | None = None,
) -> ArchivedCellClosure:
    paths = archive_paths(cell)
    state = strict_archive.inspect_rotation_state(paths)
    stale = state.get("stale_archive_temporaries")
    if not isinstance(stale, list):
        raise RunnerError("Strict archive temporary evidence is malformed.")
    if stale:
        if state.get("state") not in {
            "direct_unarchived",
            "archive_published_pending_manifest",
            "manifest_published_pending_closure",
        } or state.get("source_present") is not True:
            raise RunnerError("Strict archive temporaries are unsafe in this state.")
        removed = strict_archive.discard_stale_archive_temporaries(paths)
        if sorted(removed) != sorted(str(name) for name in stale):
            raise RunnerError("Strict archive temporary cleanup drifted.")
        state = strict_archive.inspect_rotation_state(paths)
    state_name = str(state.get("state"))
    if state_name == "empty":
        raise RunnerError("Cannot archive a cell with no completed direct tree.")
    if cell_paths(cell)[1].exists() or cell_paths(cell)[1].is_symlink():
        raise RunnerError("Cannot archive a cell with a preserved staging attempt.")
    if state_name == "direct_unarchived":
        publish_compact_cell_receipt(
            cell,
            plan=plan,
            authorization=authorization,
            scheduler_decision=scheduler_decision,
        )
    evidence = _persistent_cell_evidence(
        cell,
        plan=plan,
        authorization=authorization,
        scheduler_decision=scheduler_decision,
    )
    authority, metadata, external = _archive_metadata(
        cell,
        plan=plan,
        authorization=authorization,
        scheduler_decision=scheduler_decision,
        evidence=evidence,
    )
    prefix = f"runs/{cell.execution_id}"
    limits = strict_archive.campaign_default_archive_limits()
    try:
        if state_name == "direct_unarchived":
            archive_capacity_waiter = archive_capacity_waiter or wait_for_archive_capacity
            capacity = dict(archive_capacity_waiter())
            if capacity.get("status") != "ready":
                raise ArchiveCapacityBlocked(capacity)
        if state_name in {
            "direct_unarchived",
            "archive_published_pending_manifest",
            "manifest_published_pending_closure",
            "closure_published_pending_intent",
            "intent_published_pending_rename",
        }:
            strict_archive.build_cell_archive(
                paths=paths,
                source_member_prefix=prefix,
                external_members=external,
                authority_metadata=authority,
                cell_metadata=metadata,
                limits=limits,
            )
            strict_archive.publish_archive_closure(
                paths=paths,
                source_member_prefix=prefix,
                authority_metadata=authority,
                cell_metadata=metadata,
                limits=limits,
            )
            strict_archive.publish_rotation_intent(
                paths=paths,
                source_member_prefix=prefix,
                authority_metadata=authority,
                cell_metadata=metadata,
                rotation_authority=authorization,
                limits=limits,
            )
            strict_archive.complete_safe_tree_rotation(
                paths=paths,
                source_member_prefix=prefix,
                authority_metadata=authority,
                cell_metadata=metadata,
                rotation_authority=authorization,
                limits=limits,
            )
        elif state_name in {"retiring_pending_removal", "cleanup_receipt_pending"}:
            strict_archive.complete_safe_tree_rotation(
                paths=paths,
                source_member_prefix=prefix,
                authority_metadata=authority,
                cell_metadata=metadata,
                rotation_authority=authorization,
                limits=limits,
            )
        elif state_name != "archived_closed":
            raise RunnerError(f"Unsupported archive restart state: {state_name}")
        closure = strict_archive.validate_archive_backed_closure(
            paths=paths,
            source_member_prefix=prefix,
            expected_authority_metadata=authority,
            expected_cell_metadata=metadata,
            limits=limits,
            expected_rotation_authority=authorization,
            require_cleanup=True,
        )
    except ArchiveCapacityBlocked:
        raise
    except strict_archive.Singleton12ArchiveError as exc:
        raise RunnerError(f"Strict cell archive closure failed: {cell.execution_id}") from exc
    _validate_archive_manifest_crossbindings(cell, paths=paths, evidence=evidence)
    archived = _publish_archived_cell_receipt(
        cell,
        plan=plan,
        authorization=authorization,
        scheduler_decision=scheduler_decision,
        evidence=evidence,
        archive_backed_closure=closure,
    )
    return ArchivedCellClosure(
        cell=cell,
        rows=tuple(dict(row) for row in evidence["compact"]["rows"]),
        worker_receipt_sha256=evidence["worker"]["sha256"],
        guard_receipt_sha256=evidence["guard"]["sha256"],
        compact_receipt_sha256=evidence["compact"]["sha256"],
        archive_backed_closure_sha256=closure["sha256"],
        archive_closure_receipt_sha256=closure["archive_closure"][
            "canonical_sha256"
        ],
        archived_cell_receipt_sha256=archived["sha256"],
    )


def load_archived_cell(
    cell: CellSpec,
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    scheduler_decision: Mapping[str, Any],
) -> ArchivedCellClosure:
    """Purely validate an archive-backed cell without publishing or rotating."""

    paths = archive_paths(cell)
    state = strict_archive.inspect_rotation_state(paths)
    if state.get("state") != "archived_closed":
        raise RunnerError(f"Cell is not archive-closed: {cell.execution_id}")
    if cell_paths(cell)[1].exists() or cell_paths(cell)[1].is_symlink():
        raise RunnerError("Archived cell retains a staging attempt.")
    scheduler = validate_scheduler_decision(
        _batch_for_cell(cell),
        scheduler_decision,
        plan=plan,
        authorization=authorization,
    )
    evidence = _persistent_cell_evidence(
        cell,
        plan=plan,
        authorization=authorization,
        scheduler_decision=scheduler,
    )
    authority, metadata, _external = _archive_metadata(
        cell,
        plan=plan,
        authorization=authorization,
        scheduler_decision=scheduler,
        evidence=evidence,
    )
    try:
        closure = strict_archive.validate_archive_backed_closure(
            paths=paths,
            source_member_prefix=f"runs/{cell.execution_id}",
            expected_authority_metadata=authority,
            expected_cell_metadata=metadata,
            limits=strict_archive.campaign_default_archive_limits(),
            expected_rotation_authority=authorization,
            require_cleanup=True,
        )
    except strict_archive.Singleton12ArchiveError as exc:
        raise RunnerError(f"Archived cell closure failed: {cell.execution_id}") from exc
    _validate_archive_manifest_crossbindings(cell, paths=paths, evidence=evidence)
    receipt_path = archived_cell_receipt_path(cell)
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise RunnerError("Archived cell receipt is absent.")
    archived = validate_archived_cell_receipt(
        cell,
        load_digested(receipt_path, schema=ARCHIVED_CELL_SCHEMA),
        plan=plan,
        authorization=authorization,
        scheduler_decision=scheduler,
        evidence=evidence,
        archive_backed_closure=closure,
    )
    return ArchivedCellClosure(
        cell=cell,
        rows=tuple(dict(row) for row in evidence["compact"]["rows"]),
        worker_receipt_sha256=evidence["worker"]["sha256"],
        guard_receipt_sha256=evidence["guard"]["sha256"],
        compact_receipt_sha256=evidence["compact"]["sha256"],
        archive_backed_closure_sha256=closure["sha256"],
        archive_closure_receipt_sha256=closure["archive_closure"][
            "canonical_sha256"
        ],
        archived_cell_receipt_sha256=archived["sha256"],
    )


def _cell_lifecycle_state(cell: CellSpec) -> str:
    paths = archive_paths(cell)
    observed = strict_archive.inspect_rotation_state(paths)
    state = str(observed["state"])
    _run, staging, worker, guard = cell_paths(cell)
    compact = compact_cell_receipt_path(cell)
    archived = archived_cell_receipt_path(cell)
    log = cell_log_path(cell)
    if staging.exists() or staging.is_symlink():
        raise RunnerError(f"Cell has a preserved partial staging attempt: {cell.execution_id}")
    if state == "empty":
        if any(
            path.exists() or path.is_symlink()
            for path in (worker, guard, compact, archived, log)
        ):
            raise RunnerError(f"Cell has detached external evidence: {cell.execution_id}")
        return "pristine"
    if compact.is_symlink():
        raise RunnerError(f"Cell compact evidence is a symlink: {cell.execution_id}")
    if archived.exists() or archived.is_symlink():
        if state != "archived_closed" or archived.is_symlink() or not archived.is_file():
            raise RunnerError(
                f"Cell has a premature or unsafe archived receipt: {cell.execution_id}"
            )
    if not log.is_file() or log.is_symlink():
        raise RunnerError(f"Cell lifecycle lacks its immutable log: {cell.execution_id}")
    if not worker.is_file() or worker.is_symlink() or not guard.is_file() or guard.is_symlink():
        raise RunnerError(
            f"Cell direct/archive state lacks an authenticated worker/guard closure: {cell.execution_id}"
        )
    if state != "direct_unarchived" and (
        not compact.is_file() or compact.is_symlink()
    ):
        raise RunnerError(f"Archive state lacks compact reporting evidence: {cell.execution_id}")
    return state


def fresh_pair_launch_capacity(
    batch: BatchSpec,
    **wait_kwargs: Any,
) -> dict[str, Any]:
    contract = pair_launch_capacity_contract(batch)
    observation = _bounded_capacity_wait(
        launch_memory_bytes=int(contract["required_available_memory_bytes"]),
        launch_disk_bytes=int(contract["required_free_disk_bytes"]),
        **wait_kwargs,
    )
    physical = int(psutil.virtual_memory().total)
    ready = (
        observation.get("status") == "ready"
        and physical >= int(contract["required_physical_memory_bytes"])
    )
    return {
        **contract,
        **observation,
        "status": "ready" if ready else "blocked_capacity",
        "capacity_kind": "fresh_pair_launch_recheck",
        "physical_memory_bytes": physical,
    }


def _preflight_nonpristine_cell(
    cell: CellSpec,
    state: str,
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    scheduler_decision: Mapping[str, Any],
) -> ArchivedCellClosure | None:
    """Purely validate all durable prerequisites before any batch mutation."""

    if state == "direct_unarchived":
        load_closed_cell(
            cell,
            plan=plan,
            authorization=authorization,
            scheduler_decision=scheduler_decision,
        )
        return None
    if state == "archived_closed":
        receipt = archived_cell_receipt_path(cell)
        if receipt.is_file() and not receipt.is_symlink():
            return load_archived_cell(
                cell,
                plan=plan,
                authorization=authorization,
                scheduler_decision=scheduler_decision,
            )
        # Rotation can durably finish immediately before the small archived
        # receipt is published.  Validate the complete closure without writing;
        # archive_and_load_cell will idempotently publish the receipt later.
        evidence = _persistent_cell_evidence(
            cell,
            plan=plan,
            authorization=authorization,
            scheduler_decision=scheduler_decision,
        )
        authority, metadata, _external = _archive_metadata(
            cell,
            plan=plan,
            authorization=authorization,
            scheduler_decision=scheduler_decision,
            evidence=evidence,
        )
        paths = archive_paths(cell)
        try:
            strict_archive.validate_archive_backed_closure(
                paths=paths,
                source_member_prefix=f"runs/{cell.execution_id}",
                expected_authority_metadata=authority,
                expected_cell_metadata=metadata,
                limits=strict_archive.campaign_default_archive_limits(),
                expected_rotation_authority=authorization,
                require_cleanup=True,
            )
        except strict_archive.Singleton12ArchiveError as exc:
            raise RunnerError(
                f"Archived restart prerequisite failed: {cell.execution_id}"
            ) from exc
        _validate_archive_manifest_crossbindings(cell, paths=paths, evidence=evidence)
        return None
    evidence = _persistent_cell_evidence(
        cell,
        plan=plan,
        authorization=authorization,
        scheduler_decision=scheduler_decision,
    )
    authority, metadata, _external = _archive_metadata(
        cell,
        plan=plan,
        authorization=authorization,
        scheduler_decision=scheduler_decision,
        evidence=evidence,
    )
    paths = archive_paths(cell)
    prefix = f"runs/{cell.execution_id}"
    limits = strict_archive.campaign_default_archive_limits()
    try:
        strict_archive.validate_cell_archive(
            paths.archive_path,
            expected_execution_id=cell.execution_id,
            expected_source_member_prefix=prefix,
            expected_authority_metadata=authority,
            expected_cell_metadata=metadata,
            limits=limits,
        )
        if state != "archive_published_pending_manifest":
            _validate_archive_manifest_crossbindings(
                cell, paths=paths, evidence=evidence
            )
        if state in {
            "closure_published_pending_intent",
            "intent_published_pending_rename",
            "retiring_pending_removal",
        }:
            strict_archive.validate_archive_backed_closure(
                paths=paths,
                source_member_prefix=prefix,
                expected_authority_metadata=authority,
                expected_cell_metadata=metadata,
                limits=limits,
                expected_rotation_authority=authorization,
                require_cleanup=False,
            )
    except strict_archive.Singleton12ArchiveError as exc:
        raise RunnerError(
            f"Archive restart prerequisite failed: {cell.execution_id}"
        ) from exc
    return None


def validate_batch_receipt(
    batch: BatchSpec,
    receipt: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    scheduler_decision: Mapping[str, Any],
    closures: Sequence[ArchivedCellClosure],
    initial_capacity_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = list(closures)
    if [row.cell.execution_id for row in ordered] != list(batch.execution_ids):
        raise RunnerError("Batch archived-cell order drifted.")
    value = dict(receipt)
    observed_sha = value.pop("sha256", None)
    created_at = value.pop("created_at", None)
    cell_rows = [
        {
            "execution_id": row.cell.execution_id,
            "worker_receipt_sha256": row.worker_receipt_sha256,
            "guard_receipt_sha256": row.guard_receipt_sha256,
            "compact_receipt_sha256": row.compact_receipt_sha256,
            "archive_backed_closure_sha256": row.archive_backed_closure_sha256,
            "archive_closure_receipt_sha256": row.archive_closure_receipt_sha256,
            "archived_cell_receipt_sha256": row.archived_cell_receipt_sha256,
        }
        for row in ordered
    ]
    expected = {
        "schema": BATCH_RECEIPT_SCHEMA,
        "status": "passed_two_archived_cells_before_next_batch",
        "campaign_id": CAMPAIGN_ID,
        "batch": _batch_payload(batch),
        "execution_ids": list(batch.execution_ids),
        "plan_sha256": plan["sha256"],
        "authorization_sha256": authorization["sha256"],
        "source_implementation_inventory_sha256": plan[
            "source_implementation_inventory_sha256"
        ],
        "initial_capacity_receipt_sha256": initial_capacity_receipt["sha256"],
        "scheduler_decision_sha256": scheduler_decision["sha256"],
        "scheduling_mode": scheduler_decision["scheduling_mode"],
        "cell_archive_closures": cell_rows,
        "direct_run_trees_absent": True,
        "submission_authorized": False,
        "paper_adoption_authorized": False,
        "paper_evidence_adoption_authorized": False,
    }
    if (
        not isinstance(created_at, str)
        or not created_at
        or observed_sha != canonical_sha256({**value, "created_at": created_at})
        or value != expected
        or any(archive_paths(row.cell).source_root.exists() for row in ordered)
    ):
        raise RunnerError("Archived batch closure receipt drifted.")
    return dict(receipt)


def publish_batch_receipt(
    batch: BatchSpec,
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    scheduler_decision: Mapping[str, Any],
    closures: Sequence[ArchivedCellClosure],
    initial_capacity_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    path = batch_receipt_path(batch)
    if path.is_file():
        return validate_batch_receipt(
            batch,
            load_digested(path, schema=BATCH_RECEIPT_SCHEMA),
            plan=plan,
            authorization=authorization,
            scheduler_decision=scheduler_decision,
            closures=closures,
            initial_capacity_receipt=initial_capacity_receipt,
        )
    payload = digested(
        {
            "schema": BATCH_RECEIPT_SCHEMA,
            "status": "passed_two_archived_cells_before_next_batch",
            "created_at": serial_runtime.utc_now(),
            "campaign_id": CAMPAIGN_ID,
            "batch": _batch_payload(batch),
            "execution_ids": list(batch.execution_ids),
            "plan_sha256": plan["sha256"],
            "authorization_sha256": authorization["sha256"],
            "source_implementation_inventory_sha256": plan[
                "source_implementation_inventory_sha256"
            ],
            "initial_capacity_receipt_sha256": initial_capacity_receipt["sha256"],
            "scheduler_decision_sha256": scheduler_decision["sha256"],
            "scheduling_mode": scheduler_decision["scheduling_mode"],
            "cell_archive_closures": [
                {
                    "execution_id": row.cell.execution_id,
                    "worker_receipt_sha256": row.worker_receipt_sha256,
                    "guard_receipt_sha256": row.guard_receipt_sha256,
                    "compact_receipt_sha256": row.compact_receipt_sha256,
                    "archive_backed_closure_sha256": (
                        row.archive_backed_closure_sha256
                    ),
                    "archive_closure_receipt_sha256": (
                        row.archive_closure_receipt_sha256
                    ),
                    "archived_cell_receipt_sha256": row.archived_cell_receipt_sha256,
                }
                for row in closures
            ],
            "direct_run_trees_absent": True,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    validate_batch_receipt(
        batch,
        payload,
        plan=plan,
        authorization=authorization,
        scheduler_decision=scheduler_decision,
        closures=closures,
        initial_capacity_receipt=initial_capacity_receipt,
    )
    write_json_exclusive(path, payload)
    return payload


def run_archived_batch(
    batch: BatchSpec,
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    initial_capacity_receipt: Mapping[str, Any],
    schedule_selector: Callable[..., Mapping[str, Any]] = select_batch_schedule,
    pair_capacity_waiter: Callable[[BatchSpec], Mapping[str, Any]] = fresh_pair_launch_capacity,
    cell_capacity_waiter: Callable[[CellSpec], Mapping[str, Any]] = wait_for_cell_launch_capacity,
    monitor: Callable[..., Sequence[Mapping[str, Any]]] = monitor_cells,
    archive_loader: Callable[..., ArchivedCellClosure] = archive_and_load_cell,
    post_science_validator: Callable[..., None] = (
        validate_post_science_batch_identity
    ),
) -> tuple[list[ArchivedCellClosure], dict[str, Any]]:
    cells = tuple(_cell_by_execution_id(value) for value in batch.execution_ids)
    states = [_cell_lifecycle_state(cell) for cell in cells]
    scheduler_path = scheduler_decision_path(batch)
    if scheduler_path.is_symlink():
        raise RunnerError("Scheduler receipt is an unsafe symlink.")
    if any(state != "pristine" for state in states) and not scheduler_path.is_file():
        raise RunnerError(
            "Scheduler receipt is missing after batch science evidence exists."
        )
    scheduler = dict(
        schedule_selector(
            batch,
            plan=plan,
            authorization=authorization,
        )
    )
    scheduler = validate_scheduler_decision(
        batch, scheduler, plan=plan, authorization=authorization
    )
    mode = str(scheduler["scheduling_mode"])
    pristine_count = sum(state == "pristine" for state in states)
    if mode == "pair" and pristine_count not in {0, len(cells)}:
        raise RunnerError("Pair-mode restart has only one pristine cell.")
    if (
        mode == "pair"
        and states[1] not in {"pristine", "direct_unarchived"}
        and states[0] != "archived_closed"
    ):
        raise RunnerError("Pair restart violates canonical archive order.")
    if (
        mode == "serial_capacity_fallback"
        and states[1] != "pristine"
        and states[0] != "archived_closed"
    ):
        raise RunnerError("Serial restart state violates canonical cell order.")
    preflight_archived: dict[str, ArchivedCellClosure] = {}
    for cell, state in zip(cells, states, strict=True):
        if state == "pristine":
            continue
        closure = _preflight_nonpristine_cell(
            cell,
            state,
            plan=plan,
            authorization=authorization,
            scheduler_decision=scheduler,
        )
        if closure is not None:
            preflight_archived[cell.execution_id] = closure
    existing_batch_receipt = batch_receipt_path(batch)
    if existing_batch_receipt.exists() or existing_batch_receipt.is_symlink():
        if states != ["archived_closed", "archived_closed"]:
            raise RunnerError("Batch receipt exists before both cells are archived.")
        validate_batch_receipt(
            batch,
            load_digested(existing_batch_receipt, schema=BATCH_RECEIPT_SCHEMA),
            plan=plan,
            authorization=authorization,
            scheduler_decision=scheduler,
            closures=[
                preflight_archived[execution_id]
                for execution_id in batch.execution_ids
            ],
            initial_capacity_receipt=initial_capacity_receipt,
        )
    closures: dict[str, ArchivedCellClosure] = {}
    for cell, state in zip(cells, states, strict=True):
        if state != "pristine":
            closures[cell.execution_id] = archive_loader(
                cell,
                plan=plan,
                authorization=authorization,
                scheduler_decision=scheduler,
            )
    pristine = [cell for cell in cells if cell.execution_id not in closures]
    if mode == "pair" and pristine:
        capacity = dict(pair_capacity_waiter(batch))
        if capacity.get("status") != "ready":
            raise ArchiveCapacityBlocked(capacity)
        monitor(
            pristine,
            authorization,
            batch=batch,
            scheduling_mode=mode,
            scheduler_decision_sha256=scheduler["sha256"],
            launch_capacity_observation=capacity,
        )
        try:
            post_science_validator(
                cells,
                plan=plan,
                authorization=authorization,
            )
            for cell in cells:
                load_closed_cell(
                    cell,
                    plan=plan,
                    authorization=authorization,
                    scheduler_decision=scheduler,
                )
        except BaseException as exc:
            raise BatchExecutionFailed(
                f"Batch {batch.ordinal} post-science closure validation failed.",
                scheduling_mode=mode,
            ) from exc
        # Both paired children are terminal before either heavy archive build.
        for cell in cells:
            closures[cell.execution_id] = archive_loader(
                cell,
                plan=plan,
                authorization=authorization,
                scheduler_decision=scheduler,
            )
    elif mode == "serial_capacity_fallback":
        for cell in cells:
            if cell.execution_id in closures:
                continue
            capacity = dict(cell_capacity_waiter(cell))
            if capacity.get("status") != "ready":
                raise ArchiveCapacityBlocked(capacity)
            monitor(
                (cell,),
                authorization,
                batch=batch,
                scheduling_mode=mode,
                scheduler_decision_sha256=scheduler["sha256"],
                launch_capacity_observation=capacity,
            )
            try:
                post_science_validator(
                    (cell,),
                    plan=plan,
                    authorization=authorization,
                )
                load_closed_cell(
                    cell,
                    plan=plan,
                    authorization=authorization,
                    scheduler_decision=scheduler,
                )
            except BaseException as exc:
                raise BatchExecutionFailed(
                    f"Batch {batch.ordinal} post-science closure validation "
                    f"failed for {cell.execution_id}.",
                    scheduling_mode=mode,
                ) from exc
            closures[cell.execution_id] = archive_loader(
                cell,
                plan=plan,
                authorization=authorization,
                scheduler_decision=scheduler,
            )
    ordered = [closures[execution_id] for execution_id in batch.execution_ids]
    receipt = publish_batch_receipt(
        batch,
        plan=plan,
        authorization=authorization,
        scheduler_decision=scheduler,
        closures=ordered,
        initial_capacity_receipt=initial_capacity_receipt,
    )
    return ordered, receipt


def preflight_campaign_lifecycle(
    *, plan: Mapping[str, Any], authorization: Mapping[str, Any]
) -> dict[str, Any]:
    """Purely prove that all campaign evidence is one reachable prefix."""

    states_by_execution = {
        cell.execution_id: _cell_lifecycle_state(cell) for cell in CELL_SPECS
    }
    initial_present = INITIAL_CAPACITY_PATH.is_file()
    if INITIAL_CAPACITY_PATH.is_symlink() or (
        INITIAL_CAPACITY_PATH.exists() and not initial_present
    ):
        raise RunnerError("Initial capacity evidence is not a plain file.")
    initial = (
        load_initial_campaign_capacity(plan=plan, authorization=authorization)
        if initial_present
        else None
    )
    frontier_seen = False
    completed_closures: list[ArchivedCellClosure] = []
    batch_receipts: list[dict[str, Any]] = []
    for batch in BATCH_SPECS:
        cells = tuple(_cell_by_execution_id(value) for value in batch.execution_ids)
        states = [states_by_execution[cell.execution_id] for cell in cells]
        scheduler_path = scheduler_decision_path(batch)
        receipt_path = batch_receipt_path(batch)
        scheduler_present = scheduler_path.is_file()
        receipt_present = receipt_path.is_file()
        if (
            scheduler_path.is_symlink()
            or receipt_path.is_symlink()
            or (scheduler_path.exists() and not scheduler_present)
            or (receipt_path.exists() and not receipt_present)
        ):
            raise RunnerError("Campaign lifecycle evidence is not a plain file.")
        has_evidence = (
            scheduler_present
            or receipt_present
            or any(state != "pristine" for state in states)
        )
        if frontier_seen and has_evidence:
            raise RunnerError("Campaign evidence is not a canonical batch prefix.")
        if not has_evidence:
            frontier_seen = True
            continue
        if initial is None:
            raise RunnerError("Campaign evidence predates its initial capacity receipt.")
        if not scheduler_present:
            raise RunnerError("Progressed batch lacks its immutable scheduler receipt.")
        scheduler = validate_scheduler_decision(
            batch,
            load_digested(scheduler_path, schema=SCHEDULER_SCHEMA),
            plan=plan,
            authorization=authorization,
        )
        mode = str(scheduler["scheduling_mode"])
        if mode == "pair":
            if sum(state == "pristine" for state in states) not in {0, 2}:
                raise RunnerError("Pair campaign history has only one pristine cell.")
            if (
                states[1] not in {"pristine", "direct_unarchived"}
                and states[0] != "archived_closed"
            ):
                raise RunnerError("Pair campaign archive order is unreachable.")
        elif mode == "serial_capacity_fallback":
            if states[1] != "pristine" and states[0] != "archived_closed":
                raise RunnerError("Serial campaign cell order is unreachable.")
        else:
            raise RunnerError("Campaign scheduler mode is invalid.")
        closures: list[ArchivedCellClosure] = []
        for cell, state in zip(cells, states, strict=True):
            if state == "pristine":
                continue
            closure = _preflight_nonpristine_cell(
                cell,
                state,
                plan=plan,
                authorization=authorization,
                scheduler_decision=scheduler,
            )
            if closure is not None:
                closures.append(closure)
        if receipt_present:
            if states != ["archived_closed", "archived_closed"] or len(closures) != 2:
                raise RunnerError("Batch receipt precedes complete archived cells.")
            receipt = validate_batch_receipt(
                batch,
                load_digested(receipt_path, schema=BATCH_RECEIPT_SCHEMA),
                plan=plan,
                authorization=authorization,
                scheduler_decision=scheduler,
                closures=closures,
                initial_capacity_receipt=initial,
            )
            completed_closures.extend(closures)
            batch_receipts.append(receipt)
        else:
            frontier_seen = True
    report_paths = (REPORT_JSON, REPORT_CSV, REPORT_MD)
    report_presence = [path.is_file() for path in report_paths]
    if any(
        path.is_symlink() or (path.exists() and not path.is_file())
        for path in report_paths
    ):
        raise RunnerError("Campaign reporting evidence is not a plain file.")
    if any(report_presence):
        if len(batch_receipts) != len(BATCH_SPECS):
            raise RunnerError("Campaign reporting evidence is premature or partial.")
        comparison, csv_text, markdown = build_comparison(completed_closures)
        valid_present = (
            (not report_presence[0] or load_digested(REPORT_JSON, schema=REPORT_SCHEMA) == comparison)
            and (not report_presence[1] or REPORT_CSV.read_text(encoding="utf-8") == csv_text)
            and (not report_presence[2] or REPORT_MD.read_text(encoding="utf-8") == markdown)
        )
        if not valid_present:
            raise RunnerError("Campaign reporting evidence failed pure recomputation.")
    return {
        "states": states_by_execution,
        "completed_batch_count": len(batch_receipts),
        "report_set_present": all(report_presence),
    }


def build_comparison(
    cells: Sequence[ArchivedCellClosure],
) -> tuple[dict[str, Any], str, str]:
    by_execution = {closure.cell.execution_id: closure for closure in cells}
    if set(by_execution) != {cell.execution_id for cell in CELL_SPECS}:
        raise RunnerError("Comparison archived-cell closure set drifted.")
    ordered = [by_execution[cell.execution_id] for cell in CELL_SPECS]
    rows = [dict(row) for closure in ordered for row in closure.rows]
    if len(rows) != len(CELL_SPECS) * TARGET_HORIZON:
        raise RunnerError("Terminal comparison is not the exact 12-by-50 matrix.")
    by_key = {
        (row["block"], row["regime_id"], row["controller_round"]): row
        for row in rows
    }
    placement_activated = any(
        row["block"] == "plateau" and row["plateau_state"] == "open"
        for row in rows
    )
    metrics = ("energy", "absolute_delta_e", "s_alg", "n2q", "d2q", "dc")
    terminal_differences: list[dict[str, Any]] = []
    selected_agreement: list[dict[str, Any]] = []
    for regime_id, _nph in _REGIMES:
        append = by_key[("append", regime_id, TARGET_HORIZON)]
        plateau = by_key[("plateau", regime_id, TARGET_HORIZON)]
        regime_placement_status = (
            "activated"
            if any(
                row["block"] == "plateau"
                and row["regime_id"] == regime_id
                and row["plateau_state"] == "open"
                for row in rows
            )
            else "not activated"
        )
        terminal_differences.append(
            {
                "regime_id": regime_id,
                "placement_factor_status": regime_placement_status,
                "plateau_minus_append": {
                    metric: float(plateau[metric]) - float(append[metric])
                    for metric in metrics
                },
            }
        )
        for round_index in range(1, TARGET_HORIZON + 1):
            append_round = by_key[("append", regime_id, round_index)]
            plateau_round = by_key[("plateau", regime_id, round_index)]
            selected_agreement.append(
                {
                    "regime_id": regime_id,
                    "controller_round": round_index,
                    "placement_factor_status": regime_placement_status,
                    "status": (
                        "agree"
                        if (
                            append_round["selected_generator"],
                            append_round["selected_position"],
                        )
                        == (
                            plateau_round["selected_generator"],
                            plateau_round["selected_position"],
                        )
                        else "diverge"
                    ),
                }
            )
    payload = digested(
        {
            "schema": REPORT_SCHEMA,
            "status": "passed_all6_append_then_plateau_k50",
            "campaign_id": CAMPAIGN_ID,
            "placement_factor_status": (
                "activated" if placement_activated else "not activated"
            ),
            "rows": rows,
            "terminal_plateau_minus_append": terminal_differences,
            "selected_record_agreement": selected_agreement,
            "cell_receipts": [
                {
                    "execution_id": cell.execution_id,
                    "worker_receipt_sha256": closure.worker_receipt_sha256,
                    "guard_receipt_sha256": closure.guard_receipt_sha256,
                    "compact_receipt_sha256": closure.compact_receipt_sha256,
                    "archive_backed_closure_sha256": (
                        closure.archive_backed_closure_sha256
                    ),
                    "archive_closure_receipt_sha256": (
                        closure.archive_closure_receipt_sha256
                    ),
                    "archived_cell_receipt_sha256": (
                        closure.archived_cell_receipt_sha256
                    ),
                }
                for closure in ordered
                for cell in (closure.cell,)
            ],
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    columns = list(rows[0])
    csv_stream = io.StringIO(newline="")
    writer = csv.DictWriter(csv_stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    markdown = [
        "# Paper-I RA all-six adaptive append then plateau diagnostic",
        "",
        "Diagnostic only; no manuscript or evidence adoption.",
        "",
        f"Placement factor: **{payload['placement_factor_status']}**.",
        "",
        "| cell | k | E | |ΔE| | plateau | P0 | PI | PII | PIII adaptive/final | selected | S_alg | N2q | D2q | Dc |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        selected = f"{row['selected_generator']}@{row['selected_position']}"
        markdown.append(
            f"| {row['block']}/{row['regime_id']} | {row['controller_round']} | "
            f"{row['energy']:.12g} | {row['absolute_delta_e']:.4e} | "
            f"{row['plateau_state']} | "
            f"{row['phase0_population_count']}/{row['phase0_retained_count']} | "
            f"{row['phase_i_input_count']}/{row['phase_i_retained_count']} | "
            f"{row['phase_ii_input_count']}/{row['phase_ii_retained_count']} | "
            f"{row['phase_iii_input_count']}/{row['phase_iii_adaptive_retained_count']}/"
            f"{row['phase_iii_final_singleton_count']} | "
            f"{selected} | {row['s_alg']} | {row['n2q']} | {row['d2q']} | {row['dc']} |"
        )
    markdown.extend(
        [
            "",
            "Paired terminal differences and categorical selected-record comparisons are authoritative in `comparison.json`.",
            "",
        ]
    )
    return payload, csv_stream.getvalue(), "\n".join(markdown)


def _publish_or_validate_json(
    path: Path, payload: Mapping[str, Any], *, schema: str
) -> None:
    serial_runtime.publish_or_validate_json(
        path,
        payload,
        schema=schema,
        error_type=RunnerError,
    )


def _publish_or_validate_text(path: Path, body: str) -> None:
    serial_runtime.publish_or_validate_text(
        path,
        body,
        error_type=RunnerError,
    )


def execution_path_canary_observation(
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind cell 1 round 1 from the same trajectory that continued to k=50."""

    rows = comparison.get("rows")
    if not isinstance(rows, list):
        raise RunnerError("Comparison rows are unavailable for the execution check.")
    expected_cell = CELL_SPECS[0]
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("block") == expected_cell.block
        and row.get("regime_id") == expected_cell.regime_id
        and row.get("controller_round") == 1
    ]
    same_cell_rows = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("block") == expected_cell.block
        and row.get("regime_id") == expected_cell.regime_id
    ]
    if len(matches) != 1 or len(same_cell_rows) != TARGET_HORIZON:
        raise RunnerError("Cell 1 same-trajectory execution check is incomplete.")
    if [row.get("controller_round") for row in same_cell_rows] != list(
        range(1, TARGET_HORIZON + 1)
    ):
        raise RunnerError("Cell 1 trajectory is not the exact ordered k=1..50 path.")
    return {
        "execution_id": expected_cell.execution_id,
        "accepted_round": 1,
        "round_row_sha256": serial_runtime.canonical_sha256(matches[0]),
        "continues_same_trajectory_to_k50": True,
        "separate_scientific_trajectory": False,
    }


def validate_terminal_matrix(
    *,
    plan: Mapping[str, Any] | None = None,
    authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if plan is None or authorization is None:
        plan, authorization = validate_authority(recompute_protocols=True)
    initial_capacity = load_initial_campaign_capacity(
        plan=plan, authorization=authorization
    )
    batch_receipts: list[dict[str, Any]] = []
    closed_by_execution: dict[str, ArchivedCellClosure] = {}
    scheduler_rows: list[dict[str, Any]] = []
    for batch in BATCH_SPECS:
        scheduler = validate_scheduler_decision(
            batch,
            load_digested(scheduler_decision_path(batch), schema=SCHEDULER_SCHEMA),
            plan=plan,
            authorization=authorization,
        )
        scheduler_rows.append(scheduler)
        closures = [
            load_archived_cell(
                _cell_by_execution_id(execution_id),
                plan=plan,
                authorization=authorization,
                scheduler_decision=scheduler,
            )
            for execution_id in batch.execution_ids
        ]
        for closure in closures:
            closed_by_execution[closure.cell.execution_id] = closure
        batch_receipts.append(
            validate_batch_receipt(
                batch,
                load_digested(batch_receipt_path(batch), schema=BATCH_RECEIPT_SCHEMA),
                plan=plan,
                authorization=authorization,
                scheduler_decision=scheduler,
                closures=closures,
                initial_capacity_receipt=initial_capacity,
            )
        )
    closed = [closed_by_execution[cell.execution_id] for cell in CELL_SPECS]
    comparison, csv_text, markdown = build_comparison(closed)
    if load_digested(REPORT_JSON, schema=REPORT_SCHEMA) != comparison:
        raise RunnerError("Terminal comparison JSON failed recomputation.")
    if REPORT_CSV.read_text(encoding="utf-8") != csv_text:
        raise RunnerError("Terminal comparison CSV failed recomputation.")
    if REPORT_MD.read_text(encoding="utf-8") != markdown:
        raise RunnerError("Terminal comparison Markdown failed recomputation.")
    terminal = load_digested(TERMINAL_PATH, schema=TERMINAL_SCHEMA)
    expected = {
        "schema": TERMINAL_SCHEMA,
        "status": "passed_all6_append_then_plateau_k50",
        "campaign_id": CAMPAIGN_ID,
        "plan_sha256": plan["sha256"],
        "authorization_sha256": authorization["sha256"],
        "execution_path_canary_observation": (
            execution_path_canary_observation(comparison)
        ),
        "source_implementation_inventory_sha256": plan[
            "source_implementation_inventory_sha256"
        ],
        "canonical_cell_order": [cell.execution_id for cell in CELL_SPECS],
        "deterministic_launch_order": [
            execution_id
            for batch in BATCH_SPECS
            for execution_id in batch.execution_ids
        ],
        "maximum_concurrency": MAXIMUM_CONCURRENCY,
        "serial_capacity_fallback_authorized": (
            SERIAL_CAPACITY_FALLBACK_AUTHORIZED
        ),
        "initial_capacity_receipt_sha256": initial_capacity["sha256"],
        "scheduler_decision_sha256s": [row["sha256"] for row in scheduler_rows],
        "batch_receipt_sha256s": [row["sha256"] for row in batch_receipts],
        "archived_cell_receipt_sha256s": [
            row.archived_cell_receipt_sha256 for row in closed
        ],
        "archive_backed_closure_sha256s": [
            row.archive_backed_closure_sha256 for row in closed
        ],
        "archive_closure_receipt_sha256s": [
            row.archive_closure_receipt_sha256 for row in closed
        ],
        "compact_receipt_sha256s": [row.compact_receipt_sha256 for row in closed],
        "archive_backed_cell_count": len(closed),
        "direct_run_tree_count": sum(
            int(archive_paths(cell).source_root.exists()) for cell in CELL_SPECS
        ),
        "comparison_row_count": len(comparison["rows"]),
        "append_block_closed_before_plateau": True,
        "comparison_sha256": comparison["sha256"],
        "comparison_csv_sha256": sha256_file(REPORT_CSV),
        "comparison_markdown_sha256": sha256_file(REPORT_MD),
        "controller_rounds_completed_by_cell": {
            cell.execution_id: TARGET_HORIZON for cell in CELL_SPECS
        },
        "submission_authorized": False,
        "paper_adoption_authorized": False,
        "paper_evidence_adoption_authorized": False,
    }
    if set(terminal) != set(expected) | {"sha256"} or {
        key: terminal.get(key) for key in expected
    } != expected:
        raise RunnerError("Terminal all6 matrix receipt failed deep validation.")
    return terminal


def _run_campaign_impl() -> int:
    plan, authorization = validate_authority(recompute_protocols=True)
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    with serial_runtime.exclusive_campaign_lock(
        LOCK_PATH,
        label="All6 adaptive overnight campaign",
        error_type=RunnerError,
    ):
        preflight_campaign_lifecycle(plan=plan, authorization=authorization)
        if TERMINAL_PATH.is_file():
            terminal = validate_terminal_matrix(
                plan=plan,
                authorization=authorization,
            )
            write_status(
                {
                    "status": "passed_all6_append_then_plateau_k50",
                    "terminal_sha256": terminal["sha256"],
                }
            )
            return 0
        try:
            initial_capacity = ensure_initial_campaign_capacity(
                plan=plan, authorization=authorization
            )
        except ArchiveCapacityBlocked as exc:
            write_status({**exc.snapshot, "status": "blocked_capacity"})
            return 2
        completed: list[ArchivedCellClosure] = []
        batch_receipts: list[dict[str, Any]] = []
        for batch in BATCH_SPECS:
            if batch.block == "plateau":
                assert_append_block_closed(
                    [row.cell.execution_id for row in completed[:6]]
                )
            try:
                closures, batch_receipt = run_archived_batch(
                    batch,
                    plan=plan,
                    authorization=authorization,
                    initial_capacity_receipt=initial_capacity,
                )
            except ArchiveCapacityBlocked as exc:
                kind = str(exc.snapshot.get("capacity_kind", ""))
                write_status(
                    {
                        **exc.snapshot,
                        "status": (
                            "blocked_archive_capacity"
                            if kind == "per_cell_archive_build"
                            else "blocked_capacity"
                        ),
                        "batch_ordinal": batch.ordinal,
                    }
                )
                return 2
            completed.extend(closures)
            batch_receipts.append(batch_receipt)
        assert_append_block_closed(
            [row.cell.execution_id for row in completed[:6]]
        )
        comparison, csv_text, markdown = build_comparison(completed)
        _publish_or_validate_json(
            REPORT_JSON,
            comparison,
            schema=REPORT_SCHEMA,
        )
        _publish_or_validate_text(REPORT_CSV, csv_text)
        _publish_or_validate_text(REPORT_MD, markdown)
        terminal = digested(
            {
                "schema": TERMINAL_SCHEMA,
                "status": "passed_all6_append_then_plateau_k50",
                "campaign_id": CAMPAIGN_ID,
                "plan_sha256": plan["sha256"],
                "authorization_sha256": authorization["sha256"],
                "execution_path_canary_observation": (
                    execution_path_canary_observation(comparison)
                ),
                "source_implementation_inventory_sha256": plan[
                    "source_implementation_inventory_sha256"
                ],
                "canonical_cell_order": [
                    cell.execution_id for cell in CELL_SPECS
                ],
                "deterministic_launch_order": [
                    execution_id
                    for batch in BATCH_SPECS
                    for execution_id in batch.execution_ids
                ],
                "maximum_concurrency": MAXIMUM_CONCURRENCY,
                "serial_capacity_fallback_authorized": (
                    SERIAL_CAPACITY_FALLBACK_AUTHORIZED
                ),
                "initial_capacity_receipt_sha256": initial_capacity["sha256"],
                "scheduler_decision_sha256s": [
                    load_digested(
                        scheduler_decision_path(batch), schema=SCHEDULER_SCHEMA
                    )["sha256"]
                    for batch in BATCH_SPECS
                ],
                "batch_receipt_sha256s": [row["sha256"] for row in batch_receipts],
                "archived_cell_receipt_sha256s": [
                    row.archived_cell_receipt_sha256 for row in completed
                ],
                "archive_backed_closure_sha256s": [
                    row.archive_backed_closure_sha256 for row in completed
                ],
                "archive_closure_receipt_sha256s": [
                    row.archive_closure_receipt_sha256 for row in completed
                ],
                "compact_receipt_sha256s": [
                    row.compact_receipt_sha256 for row in completed
                ],
                "archive_backed_cell_count": len(completed),
                "direct_run_tree_count": sum(
                    int(archive_paths(cell).source_root.exists())
                    for cell in CELL_SPECS
                ),
                "comparison_row_count": len(comparison["rows"]),
                "append_block_closed_before_plateau": True,
                "comparison_sha256": comparison["sha256"],
                "comparison_csv_sha256": sha256_file(REPORT_CSV),
                "comparison_markdown_sha256": sha256_file(REPORT_MD),
                "controller_rounds_completed_by_cell": {
                    cell.execution_id: TARGET_HORIZON for cell in CELL_SPECS
                },
                "submission_authorized": False,
                "paper_adoption_authorized": False,
                "paper_evidence_adoption_authorized": False,
            }
        )
        write_json_exclusive(TERMINAL_PATH, terminal)
        validate_terminal_matrix(plan=plan, authorization=authorization)
        write_status(
            {
                "status": "passed_all6_append_then_plateau_k50",
                "terminal_sha256": terminal["sha256"],
            }
        )
        return 0


def run_campaign() -> int:
    try:
        return _run_campaign_impl()
    except BaseException as exc:
        message = str(exc)
        failed_status = (
            exc.failure_status
            if isinstance(exc, BatchExecutionFailed)
            else (
                "failed_pair"
                if isinstance(exc, RunnerError)
                and (message.startswith("Batch ") or "sibling" in message)
                else "failed_campaign"
            )
        )
        try:
            write_status(
                {
                    "status": failed_status,
                    "error_type": type(exc).__name__,
                    "error": message,
                }
            )
        except BaseException:
            pass
        raise


def preflight() -> dict[str, Any]:
    plan = build_plan() if not PLAN_PATH.is_file() else validate_plan(
        recompute_protocols=True
    )
    return {
        "campaign_id": CAMPAIGN_ID,
        "cell_count": len(CELL_SPECS),
        "canonical_cell_order": [cell.execution_id for cell in CELL_SPECS],
        "source_implementation_inventory_sha256": plan[
            "source_implementation_inventory_sha256"
        ],
        "plan_present": PLAN_PATH.is_file(),
        "authorization_present": AUTHORIZATION_PATH.is_file(),
        "runtime_present": RUNTIME_ROOT.exists(),
        "scientific_execution_performed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--prepare-plan", action="store_true")
    actions.add_argument("--authorize", action="store_true")
    actions.add_argument("--run-campaign", action="store_true")
    actions.add_argument("--child")
    args = parser.parse_args(argv)
    if args.preflight:
        print(json.dumps(preflight(), indent=2, sort_keys=True))
        return 0
    if args.prepare_plan:
        print(json.dumps(prepare_plan(), indent=2, sort_keys=True))
        return 0
    if args.authorize:
        print(json.dumps(authorize(), indent=2, sort_keys=True))
        return 0
    if args.run_campaign:
        return run_campaign()
    return run_child(str(args.child))


if __name__ == "__main__":
    raise SystemExit(main())
