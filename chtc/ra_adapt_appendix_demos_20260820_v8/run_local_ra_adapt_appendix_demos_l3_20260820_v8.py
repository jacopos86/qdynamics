#!/usr/bin/env python3
"""Run the fixed L=3 RA-ADAPT appendix escalation campaign locally.

The command-line surface is deliberately empty.  The campaign is an exact
four-cell allowlist: minimal RA, RA+prune, RA+batch, and RA+beam on the pure
Hubbard ``L=3`` point, each through controller round five, on the append-RA
route with the sealed explicit-depth compile cost and no warm start.

The ``L=3`` candidate supply is the sector-guarded composite generator pool.
The guarded single-Pauli exposure used at ``L=2`` is sector-degenerate here:
its nine survivors are diagonal ``Z`` words whose Hartree--Fock gradients and
tangent self-metrics vanish identically, so the controller's supported
response rank is zero.  The campaign measures and records that response
before executing any cell.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
import csv
from decimal import Decimal, ROUND_FLOOR
import fcntl
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping


RUNNER_PATH = Path(__file__).resolve()
REPO_ROOT = RUNNER_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CAMPAIGN_ID = "ra_adapt_appendix_demos_l3_20260820_v8"
REMOTE_JOB_ID = "ra-adapt-appendix-demos-l3-20260820-v8"
RUNTIME_RELATIVE_PATH = "output/local_runs/ra_adapt_appendix_demos_l3_20260820_v8"
REPORT_RELATIVE_PATH = "chtc/ra_adapt_appendix_demos_20260820/results_l3_v8"
MEMORY_PRESSURE_COMMAND = ("/usr/bin/memory_pressure", "-Q")
# The v1--v5 campaigns are preserved unchanged; this campaign neither reads
# nor rewrites their runtime roots, reports, or receipts.
PRESERVED_CAMPAIGN_IDS = (
    "ra_adapt_appendix_demos_20260820_v1",
    "ra_adapt_appendix_demos_20260820_v2",
    "ra_adapt_appendix_demos_20260820_v3",
    "ra_adapt_appendix_demos_20260820_v4",
    "ra_adapt_appendix_demos_20260820_v5",
    "ra_adapt_appendix_demos_l3_20260820_v6",
    "ra_adapt_appendix_demos_l3_20260820_v7",
)
PRESERVED_V5_REPORT_ROOT = (
    "chtc/ra_adapt_appendix_demos_20260820/results_v5"
)
PRESERVED_V6_REPORT_ROOT = (
    "chtc/ra_adapt_appendix_demos_20260820/results_l3_v6"
)
PRESERVED_V7_REPORT_ROOT = (
    "chtc/ra_adapt_appendix_demos_20260820/results_l3_v7"
)
PRESERVED_V5_MANIFEST = (
    "chtc/ra_adapt_appendix_demos_20260820/results_v5/campaign_manifest.json"
)
REQUIRED_APPENDIX_LANE_ANCESTOR = "fd18c46804db2df8fb195a89406ca4db4ee94782"
REQUIRED_SCIENCE_ANCESTOR = "eecd9425e980949793c43514ed5a2ff7baa8dfd0"
ACCURACY_THRESHOLDS = (1.0e-2, 1.0e-4, 1.0e-6)
GENERATOR_REPRESENTATION = "macro_generator_v1"
# Round fifteen.  Round ten already reaches machine level, but the prune arm
# cannot act before then: a freshly admitted generator is protection-locked
# for phase1_prune_protect_steps=2 rounds, so the first accepted deletion
# lands at round eleven.  Fifteen is the shortest horizon that shows the
# prune and beam arms doing anything at all.
CAMPAIGN_HORIZON = 15
# Six arms: the four canonical ones plus the two tuned variants.  The gated
# prune screens nominees by angle so no futile delete-and-refit is paid for,
# and the cost-ranked beam ranks siblings by Phase-III compiled burden rather
# than estimator queries.
RA_ADAPT_APPENDIX_L3_V8_ARMS = (
    "baseline",
    "metric_ablation",
    "metric_ablation_gated",
    "batching",
    "beam",
    "beam_cost",
)
COMPILE_COST_WEIGHTS = {
    "delta_N2q": 0.30,
    "delta_D2q": 0.30,
    "delta_Dc": 0.25,
    "delta_N1q": 0.0,
    "delta_size": 0.0,
}
GIB = 1024**3
LAUNCH_MINIMUM_AVAILABLE_MEMORY_BYTES = 6 * GIB
LAUNCH_MINIMUM_FREE_DISK_BYTES = 20 * GIB
RUNTIME_MINIMUM_AVAILABLE_MEMORY_BYTES = 2 * GIB
RUNTIME_MINIMUM_FREE_DISK_BYTES = 10 * GIB
MAXIMUM_CHILD_RSS_BYTES = 8 * GIB
_MEMORY_PRESSURE_PERCENT_RE = re.compile(
    r"^System-wide memory free percentage:\s*"
    r"(?P<percent>(?:100(?:\.0+)?)|(?:\d{1,2}(?:\.\d+)?))%\s*$",
    re.MULTILINE,
)


class CapacityBlocked(RuntimeError):
    """The fixed campaign lacks the resources required for safe execution."""


class ChildMemoryExceeded(RuntimeError):
    """The one authorized child exceeded its fixed RSS ceiling."""


class CampaignAlreadyRunning(RuntimeError):
    """Another process owns the campaign's nonblocking lock."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256_without_field(
    value: Mapping[str, Any],
    *,
    field: str = "sha256",
) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop(field, None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _science_cell(*, num_sites: int, arm: str, horizon: int) -> dict[str, Any]:
    return {
        "cell_id": f"l{num_sites}_{arm}_h{horizon}",
        "stage": f"l{num_sites}",
        "num_sites": num_sites,
        "u_over_t": 1.25,
        "arm": arm,
        "noise_level": None,
        "horizon": horizon,
        "execution_entrypoint": "run_ra_adapt",
    }


def build_campaign_plan() -> dict[str, Any]:
    cells = [
        _science_cell(num_sites=3, arm=arm, horizon=CAMPAIGN_HORIZON)
        for arm in RA_ADAPT_APPENDIX_L3_V8_ARMS
    ]
    payload = {
        "schema": "ra_adapt_appendix_l3_campaign_plan_v3",
        "campaign_id": CAMPAIGN_ID,
        "required_science_ancestor": REQUIRED_SCIENCE_ANCESTOR,
        "accuracy_thresholds": list(ACCURACY_THRESHOLDS),
        "compile_cost_weights": dict(COMPILE_COST_WEIGHTS),
        "generator_representation": GENERATOR_REPRESENTATION,
        "warm_start": "off",
        "insertion": "append_only",
        "inner_optimizer": "powell",
        "preserved_campaign_ids": list(PRESERVED_CAMPAIGN_IDS),
        "cells": cells,
    }
    return {**payload, "sha256": canonical_sha256_without_field(payload)}


def _threshold_hit(summary: Mapping[str, Any], threshold: float) -> int | None:
    rounds = summary.get("accepted_rounds")
    if not isinstance(rounds, list):
        raise ValueError("Cell summary accepted_rounds must be a list.")
    for row in rounds:
        if not isinstance(row, Mapping):
            raise ValueError("Accepted-round rows must be mappings.")
        if float(row["same_cutoff_absolute_error"]) <= threshold:
            return int(row["controller_round"])
    return None


def build_l3_pool_preflight() -> dict[str, Any]:
    """Measure the L=3 candidate response before any cell executes.

    This records both constructions: the degenerate guarded single-Pauli
    exposure that produced supported response rank zero, and the
    sector-guarded composite pool the campaign actually runs.  It fails
    closed if the executed pool is not responsive, and equally if the
    degenerate pool has silently stopped being degenerate -- either would
    mean the campaign is no longer testing what it claims.
    """

    from pipelines.static_adapt.ra_adapt.appendix_demos import (
        build_ra_adapt_appendix_pure_hubbard_problem,
        ra_adapt_appendix_generator_representation,
    )
    from pipelines.static_adapt.ra_adapt.appendix_l3_pool_diagnostics import (
        appendix_pool_response_diagnostics,
        require_responsive_appendix_pool,
    )
    from pipelines.static_adapt.ra_adapt.pools import (
        build_ra_adapt_appendix_pure_hubbard_guarded_single_pauli_pool,
        build_ra_adapt_appendix_pure_hubbard_sector_guarded_macro_pool,
    )

    problem = build_ra_adapt_appendix_pure_hubbard_problem(num_sites=3)
    if (
        ra_adapt_appendix_generator_representation(3)
        != GENERATOR_REPRESENTATION
    ):
        raise ValueError("The L=3 generator representation drifted.")
    executed_pool = (
        build_ra_adapt_appendix_pure_hubbard_sector_guarded_macro_pool(problem)
    )
    executed = require_responsive_appendix_pool(problem, executed_pool)
    degenerate_pool = (
        build_ra_adapt_appendix_pure_hubbard_guarded_single_pauli_pool(problem)
    )
    degenerate = appendix_pool_response_diagnostics(problem, degenerate_pool)
    if not (
        degenerate["all_generators_diagonal"] is True
        and int(degenerate["supported_response_rank"]) == 0
        and int(degenerate["nonzero_gradient_count"]) == 0
        and int(degenerate["nonzero_self_metric_count"]) == 0
    ):
        raise ValueError(
            "The recorded degenerate L=3 single-Pauli construction is no "
            "longer degenerate; the campaign premise must be re-derived."
        )
    if int(executed["supported_response_rank"]) <= 0:
        raise ValueError("The executed L=3 pool has no supported response.")
    payload = {
        "schema": "ra_adapt_appendix_l3_pool_preflight_v1",
        "campaign_id": CAMPAIGN_ID,
        "executed_pool": executed,
        "superseded_degenerate_pool": degenerate,
        "executed_generator_count": int(executed["generator_count"]),
        "executed_supported_response_rank": int(
            executed["supported_response_rank"]
        ),
        "superseded_generator_count": int(degenerate["generator_count"]),
        "superseded_supported_response_rank": int(
            degenerate["supported_response_rank"]
        ),
        "rank_validator_weakened": False,
    }
    return {**payload, "sha256": canonical_sha256_without_field(payload)}


def select_execution_cells(
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return the exact fixed six-cell L=3 execution allowlist."""

    if dict(plan) != build_campaign_plan():
        raise ValueError("Only the exact fixed campaign plan may execute.")
    cells = plan.get("cells")
    if not isinstance(cells, list) or len(cells) != len(
        RA_ADAPT_APPENDIX_L3_V8_ARMS
    ):
        raise ValueError("Campaign plan must contain exactly six cells.")
    if len({str(cell["cell_id"]) for cell in cells}) != len(cells):
        raise ValueError("Campaign plan contains repeated cell identities.")
    if preflight is not None:
        if canonical_sha256_without_field(preflight) != preflight.get("sha256"):
            raise ValueError("L=3 pool preflight digest is invalid.")
        if (
            int(preflight.get("executed_supported_response_rank", 0)) <= 0
            or int(preflight.get("superseded_supported_response_rank", -1)) != 0
            or preflight.get("rank_validator_weakened") is not False
        ):
            raise ValueError(
                "The L=3 pool preflight does not authorize execution."
            )
    return [dict(cell) for cell in cells]


def validate_source_facts(facts: Mapping[str, Any]) -> dict[str, Any]:
    """Validate already-measured Git facts at the pure decision seam."""

    value = dict(facts)
    root = Path(str(value.get("repo_root", ""))).resolve(strict=False)
    authorized_root = Path("/Users/jakestrobel/local_repos").resolve(strict=False)
    try:
        root.relative_to(authorized_root)
    except ValueError as exc:
        raise ValueError(
            "Campaign checkout must be under /Users/jakestrobel/local_repos."
        ) from exc
    head = str(value.get("head", ""))
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise ValueError("Source HEAD must be a full lowercase Git SHA-1.")
    if value.get("required_ancestor") != REQUIRED_SCIENCE_ANCESTOR:
        raise ValueError("Source facts name the wrong required science ancestor.")
    if value.get("required_ancestor_present") is not True:
        raise ValueError("Current source does not contain the required science ancestor.")
    if (
        value.get("required_appendix_lane_ancestor")
        != REQUIRED_APPENDIX_LANE_ANCESTOR
        or value.get("required_appendix_lane_ancestor_present") is not True
    ):
        raise ValueError("Current source lost the corrected appendix lane ancestor.")
    if value.get("worktree_clean") is not True:
        raise ValueError("Campaign source must be clean before launch.")
    return value


def _capacity_thresholds(phase: str) -> tuple[int, int]:
    if phase == "launch":
        return (
            LAUNCH_MINIMUM_AVAILABLE_MEMORY_BYTES,
            LAUNCH_MINIMUM_FREE_DISK_BYTES,
        )
    if phase == "runtime":
        return (
            RUNTIME_MINIMUM_AVAILABLE_MEMORY_BYTES,
            RUNTIME_MINIMUM_FREE_DISK_BYTES,
        )
    raise ValueError("Capacity phase must be launch or runtime.")


def build_capacity_receipt(
    *,
    psutil_available_memory_bytes: int,
    total_physical_memory_bytes: int,
    free_disk_bytes: int,
    platform_name: str,
    memory_pressure_output: str | None,
    memory_pressure_error: str | None,
    native_probe_not_needed: bool = False,
    native_probe_phase: str | None = None,
) -> dict[str, Any]:
    """Seal conservative or native macOS effective-availability evidence."""

    psutil_available = int(psutil_available_memory_bytes)
    total_physical = int(total_physical_memory_bytes)
    free_disk = int(free_disk_bytes)
    if psutil_available < 0 or free_disk < 0 or total_physical <= 0:
        raise ValueError(
            "Available memory and disk must be non-negative and physical "
            "memory must be positive."
        )
    if psutil_available > total_physical:
        raise ValueError("psutil available memory cannot exceed physical memory.")
    system = str(platform_name)
    native_status = "not_applicable"
    native_percent: float | None = None
    native_available: int | None = None
    memory_source = "psutil_available_conservative"
    effective_available = psutil_available
    output = None if memory_pressure_output is None else str(memory_pressure_output)
    error = None if memory_pressure_error is None else str(memory_pressure_error)
    if system == "Darwin":
        if native_probe_not_needed:
            if output is not None or error is not None:
                raise ValueError(
                    "A skipped native probe cannot carry output or an error."
                )
            native_status = "not_needed_psutil_satisfied"
        elif error is not None or output is None:
            native_status = "unavailable"
        else:
            matches = list(_MEMORY_PRESSURE_PERCENT_RE.finditer(output))
            if len(matches) != 1:
                native_status = "malformed"
            else:
                percent_text = matches[0].group("percent")
                percent_decimal = Decimal(percent_text)
                native_percent = float(percent_decimal)
                native_available = int(
                    (
                        Decimal(total_physical)
                        * percent_decimal
                        / Decimal(100)
                    ).to_integral_value(rounding=ROUND_FLOOR)
                )
                native_status = "available"
                if native_available > psutil_available:
                    memory_source = "macos_memory_pressure_q"
                    effective_available = native_available
    payload = {
        "schema": "ra_adapt_appendix_capacity_receipt_v4",
        "campaign_id": CAMPAIGN_ID,
        "platform_name": system,
        "psutil_available_memory_bytes": psutil_available,
        "total_physical_memory_bytes": total_physical,
        "free_disk_bytes": free_disk,
        "native_probe_command": list(MEMORY_PRESSURE_COMMAND),
        "native_probe_phase": native_probe_phase,
        "native_probe_status": native_status,
        "native_probe_output": output,
        "native_probe_error": error,
        "macos_memory_pressure_free_percent": native_percent,
        "native_effective_available_memory_bytes": native_available,
        "memory_source": memory_source,
        "effective_available_memory_bytes": effective_available,
        # Compatibility alias for report consumers; the decision below uses
        # only the explicitly named effective field.
        "available_memory_bytes": effective_available,
    }
    return {**payload, "sha256": canonical_sha256_without_field(payload)}


def build_capacity_decision(
    receipt: Mapping[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    """Seal the fixed-threshold decision derived from one capacity receipt."""

    capacity = _require_signed_mapping(receipt, owner="capacity receipt")
    if (
        capacity.get("schema") != "ra_adapt_appendix_capacity_receipt_v4"
        or capacity.get("campaign_id") != CAMPAIGN_ID
    ):
        raise ValueError("Capacity receipt identity drifted from campaign v4.")
    minimum_memory, minimum_disk = _capacity_thresholds(phase)
    available_memory = int(capacity["effective_available_memory_bytes"])
    free_disk = int(capacity["free_disk_bytes"])
    reason = "capacity_satisfied"
    if available_memory < minimum_memory:
        reason = (
            f"{phase} available memory is below {minimum_memory // GIB} GiB."
        )
    elif free_disk < minimum_disk:
        reason = f"{phase} free disk is below {minimum_disk // GIB} GiB."
    payload = {
        "schema": "ra_adapt_appendix_capacity_decision_v4",
        "campaign_id": CAMPAIGN_ID,
        "phase": phase,
        "capacity_receipt_sha256": str(capacity["sha256"]),
        "memory_source": str(capacity["memory_source"]),
        "effective_available_memory_bytes": available_memory,
        "free_disk_bytes": free_disk,
        "minimum_available_memory_bytes": minimum_memory,
        "minimum_free_disk_bytes": minimum_disk,
        "allowed": reason == "capacity_satisfied",
        "reason": reason,
    }
    return {**payload, "sha256": canonical_sha256_without_field(payload)}


def require_capacity(snapshot: Mapping[str, Any], *, phase: str) -> None:
    """Enforce the unchanged launch or between-cell floor from signed evidence."""

    decision = build_capacity_decision(snapshot, phase=phase)
    if decision["allowed"] is not True:
        raise CapacityBlocked(str(decision["reason"]))


def require_child_rss(rss_bytes: int) -> None:
    if int(rss_bytes) > MAXIMUM_CHILD_RSS_BYTES:
        raise ChildMemoryExceeded("Campaign child exceeded the fixed 8 GiB RSS ceiling.")


@contextmanager
def campaign_lock(path: Path):
    """Own one persistent lock file without waiting for another campaign."""

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignAlreadyRunning(
                "The appendix campaign lock is already held."
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _require_signed_mapping(value: Any, *, owner: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{owner} must be a mapping.")
    payload = dict(value)
    if payload.get("sha256") != canonical_sha256_without_field(payload):
        raise ValueError(f"{owner} digest is invalid.")
    return payload


def _transition_pool_indices(transition: Mapping[str, Any]) -> list[int]:
    if "pool_index" in transition:
        return [int(transition["pool_index"])]
    if "selected_pool_indices" in transition:
        return [int(value) for value in transition["selected_pool_indices"]]
    admission = transition.get("admission")
    if isinstance(admission, Mapping) and isinstance(admission.get("members"), list):
        return [int(member["pool_index"]) for member in admission["members"]]
    raise ValueError("Accepted transition lost its admitted pool indices.")


def _beam_branch_sequences(value: Any) -> list[list[list[int]]]:
    if value in (None, {}):
        return []
    if not isinstance(value, Mapping) or not isinstance(value.get("rounds"), list):
        raise ValueError("Beam diagnostics must contain round rows.")
    lineages: dict[str | None, list[list[int]]] = {None: []}
    observed: list[list[list[int]]] = []
    for raw_round in value["rounds"]:
        if not isinstance(raw_round, Mapping) or not isinstance(
            raw_round.get("children"), list
        ):
            raise ValueError("Beam diagnostic round is invalid.")
        for child in raw_round["children"]:
            if not isinstance(child, Mapping):
                raise ValueError("Beam diagnostic child is invalid.")
            parent = child.get("parent_branch_id")
            if parent not in lineages:
                raise ValueError("Beam child refers to an unknown parent branch.")
            branch_id = str(child.get("branch_id", ""))
            if not branch_id or branch_id in lineages:
                raise ValueError("Beam child identity is empty or repeated.")
            selected = [int(value) for value in child["selected_pool_indices"]]
            lineage = [*lineages[parent], selected]
            lineages[branch_id] = lineage
            observed.append(copy.deepcopy(lineage))
    return observed


def summarize_cell_result(
    *,
    cell: Mapping[str, Any],
    source_contract: Mapping[str, Any],
    result_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a validated RA result onto campaign reporting quantities."""

    plan_cells = {
        str(value["cell_id"]): value for value in build_campaign_plan()["cells"]
    }
    cell_id = str(cell.get("cell_id", ""))
    if cell_id not in plan_cells or dict(cell) != plan_cells[cell_id]:
        raise ValueError("Cell escaped or drifted from the fixed allowlist.")
    source = _require_signed_mapping(source_contract, owner="application source")
    if not isinstance(result_payload, Mapping):
        raise ValueError("RA result payload must be a mapping.")
    result = dict(result_payload)
    protocol = result.get("protocol")
    run = result.get("run")
    receipts = result.get("scientific_receipts")
    if not all(isinstance(value, Mapping) for value in (protocol, run, receipts)):
        raise ValueError("RA result lost protocol, run, or scientific receipts.")
    evidence = _require_signed_mapping(
        receipts.get("accepted_run_evidence"),
        owner="accepted-run evidence",
    )
    if (
        protocol.get("sha256") != source.get("protocol_sha256")
        or evidence.get("protocol_sha256") != source.get("protocol_sha256")
        or evidence.get("route_contract_sha256")
        != source.get("route_contract_sha256")
        or receipts.get("accepted_run_evidence_sha256") != evidence.get("sha256")
    ):
        raise ValueError("Cell source, protocol, and accepted evidence disagree.")
    exact_reference = source.get("exact_reference")
    if not isinstance(exact_reference, Mapping) or exact_reference.get(
        "controller_input"
    ) is not False:
        raise ValueError("Cell exact reference is not reporting-only.")
    exact_energy = float(exact_reference["energy"])
    raw_rounds = evidence.get("accepted_rounds")
    transitions = run.get("accepted_transitions")
    trajectory = run.get("accepted_trajectory")
    if not all(isinstance(value, list) for value in (raw_rounds, transitions, trajectory)):
        raise ValueError("Cell accepted trajectory evidence must be lists.")
    if len(raw_rounds) != len(transitions) or len(raw_rounds) != len(trajectory):
        raise ValueError("Cell accepted rounds, transitions, and states do not align.")
    if transitions:
        first = transitions[0]
        if not isinstance(first, Mapping):
            raise ValueError("Accepted transition must be a mapping.")
        initial_energy = float(first["energy_before"])
    else:
        final_state = run.get("final_state")
        if not isinstance(final_state, Mapping):
            raise ValueError("A no-admission cell requires its final round-zero state.")
        initial_energy = float(final_state["energy"])
    accepted_record_sequence: list[list[int]] = []
    projected_rounds: list[dict[str, Any]] = []
    nominations = 0
    deletions = 0
    prune_trials: list[dict[str, Any]] = []
    for raw_row, raw_transition, raw_state in zip(
        raw_rounds, transitions, trajectory, strict=True
    ):
        if not all(
            isinstance(value, Mapping)
            for value in (raw_row, raw_transition, raw_state)
        ):
            raise ValueError("Accepted cell rows must be mappings.")
        controller_round = int(raw_row["controller_round"])
        if (
            int(raw_transition["controller_round"]) != controller_round
            or int(raw_state["controller_round"]) != controller_round
            or float(raw_state["energy"]) != float(raw_row["accepted_energy"])
        ):
            raise ValueError("Accepted cell round identities disagree.")
        compile_row = raw_row.get("compile")
        prune_row = raw_row.get("pruning")
        if not isinstance(compile_row, Mapping) or not isinstance(prune_row, Mapping):
            raise ValueError("Accepted row lost compile or pruning evidence.")
        cost_tuple = {
            name: int(compile_row[name]) for name in ("N2q", "D2q", "Dc")
        }
        if min(cost_tuple.values()) < 0 or cost_tuple["Dc"] < cost_tuple["D2q"]:
            raise ValueError("Accepted compile tuple is invalid.")
        nomination_count = int(prune_row.get("nomination_count", -1))
        deletion_count = int(prune_row.get("accepted_deletion_count", -1))
        if nomination_count < 0 or deletion_count not in {0, 1}:
            raise ValueError("Accepted pruning counts are invalid.")
        nominations += nomination_count
        deletions += deletion_count
        typed = prune_row.get("typed_receipt")
        if isinstance(typed, Mapping) and nomination_count:
            prune_trials.append(
                {
                    "controller_round": controller_round,
                    "status": str(prune_row.get("status")),
                    "nomination_label": prune_row.get("nomination_label"),
                    "deleted_labels": list(prune_row.get("deleted_labels", ())),
                    "measured_energy_before": typed.get("measured_energy_before"),
                    "measured_energy_after": typed.get("measured_energy_after"),
                    "predicted_energy_change": typed.get("predicted_energy_change"),
                }
            )
        accepted_record_sequence.append(_transition_pool_indices(raw_transition))
        accepted_energy = float(raw_row["accepted_energy"])
        projected_rounds.append(
            {
                "controller_round": controller_round,
                "accepted_energy": accepted_energy,
                "same_cutoff_absolute_error": abs(accepted_energy - exact_energy),
                "active_ansatz_depth": int(raw_row["active_ansatz_depth"]),
                "compile": cost_tuple,
            }
        )
    rejected_margins = [
        float(row["measured_energy_after"]) - float(row["measured_energy_before"])
        for row in prune_trials
        if row["status"] == "rejected"
        and row["measured_energy_before"] is not None
        and row["measured_energy_after"] is not None
    ]
    payload = {
        "schema": "ra_adapt_appendix_cell_summary_v1",
        "cell_id": cell_id,
        "num_sites": int(cell["num_sites"]),
        "u_over_t": float(cell["u_over_t"]),
        "arm": str(cell["arm"]),
        "noise_level": cell.get("noise_level"),
        "horizon": int(cell["horizon"]),
        "protocol_sha256": str(source["protocol_sha256"]),
        "route_contract_sha256": str(source["route_contract_sha256"]),
        "application_source_sha256": str(source["sha256"]),
        "accepted_run_evidence_sha256": str(evidence["sha256"]),
        "accepted_run_identity_sha256": str(
            evidence["accepted_run_identity_sha256"]
        ),
        "initial_energy": initial_energy,
        "exact_same_cutoff_energy": exact_energy,
        "accepted_rounds": projected_rounds,
        "accepted_record_sequence": accepted_record_sequence,
        "pruning_nomination_count": nominations,
        "accepted_deletion_count": deletions,
        "pruning_trials": prune_trials,
        "closest_miss_energy_margin": (
            None if not rejected_margins else min(rejected_margins)
        ),
        "explored_branch_sequences": _beam_branch_sequences(
            evidence.get("beam_search_diagnostics")
        ),
        "completed_controller_rounds": int(
            run.get("stop", {}).get("completed_controller_rounds", -1)
        ),
        "terminal_controller_outcome": run.get("stop", {}).get(
            "terminal_controller_outcome"
        ),
    }
    return {**payload, "sha256": canonical_sha256_without_field(payload)}


def _matched_observation(
    summary: Mapping[str, Any],
    threshold: float,
) -> dict[str, Any] | None:
    for row in summary.get("accepted_rounds", ()):
        if float(row["same_cutoff_absolute_error"]) <= threshold:
            cost = {name: int(row["compile"][name]) for name in ("N2q", "D2q", "Dc")}
            weighted = (
                COMPILE_COST_WEIGHTS["delta_N2q"] * cost["N2q"]
                + COMPILE_COST_WEIGHTS["delta_D2q"] * cost["D2q"]
                + COMPILE_COST_WEIGHTS["delta_Dc"] * cost["Dc"]
            )
            decrease = float(summary["initial_energy"]) - float(row["accepted_energy"])
            return {
                "cell_id": str(summary["cell_id"]),
                "controller_round": int(row["controller_round"]),
                "accepted_energy": float(row["accepted_energy"]),
                "same_cutoff_absolute_error": float(
                    row["same_cutoff_absolute_error"]
                ),
                "cost_tuple": cost,
                "weighted_compile_cost": weighted,
                "energy_decrease": decrease,
                "weighted_cost_per_energy_decrease": (
                    None if decrease <= 0.0 else weighted / decrease
                ),
            }
    return None


def build_matched_accuracy_rows(
    baseline: Mapping[str, Any],
    feature: Mapping[str, Any],
    *,
    thresholds: tuple[float, ...] = ACCURACY_THRESHOLDS,
) -> list[dict[str, Any]]:
    """Report the first post-prune compiled prefix reaching each fixed target."""

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        baseline_hit = _matched_observation(baseline, float(threshold))
        feature_hit = _matched_observation(feature, float(threshold))
        if baseline_hit is None and feature_hit is None:
            status = "neither_reached"
        elif baseline_hit is None:
            status = "feature_only_baseline_censored"
        elif feature_hit is None:
            status = "baseline_only_feature_censored"
        else:
            status = "reached_by_both"
        rows.append(
            {
                "threshold": float(threshold),
                "status": status,
                "baseline": baseline_hit,
                "feature": feature_hit,
                "accepted_round_gap": (
                    None
                    if baseline_hit is None or feature_hit is None
                    else int(baseline_hit["controller_round"])
                    - int(feature_hit["controller_round"])
                ),
            }
        )
    return rows


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_atomic_noreplace(path: Path, payload: Any) -> None:
    """Write one JSON file durably and refuse replacement.

    Campaign files are first written inside an unpublished staging directory,
    so exclusive creation plus the later directory rename is the atomic
    publication boundary.  A partial file is intentionally retained on an I/O
    failure for post-mortem inspection.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=False) as stream:
            json.dump(
                payload,
                stream,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(target.parent)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    base = Path(root)
    rows: list[dict[str, Any]] = []
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            raise ValueError("Cell staging artifacts cannot contain symlinks.")
        if not path.is_file() or path.name == "artifact_receipt.json":
            continue
        relative = path.relative_to(base).as_posix()
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    if not rows:
        raise ValueError("Cell staging directory contains no artifacts.")
    return rows


def publish_staged_cell(
    staging_dir: Path,
    final_dir: Path,
    *,
    cell_id: str,
) -> dict[str, Any]:
    """Hash, fsync, and atomically rename one completed cell directory."""

    staging = Path(staging_dir)
    final = Path(final_dir)
    if not staging.is_dir() or staging.is_symlink():
        raise ValueError("Cell staging path must be a real directory.")
    if final.exists():
        raise FileExistsError(f"Published cell already exists: {final}")
    if (staging / "artifact_receipt.json").exists():
        raise FileExistsError("Staging artifact receipt already exists.")
    expected = {
        str(value["cell_id"]) for value in build_campaign_plan()["cells"]
    }
    if cell_id not in expected:
        raise ValueError("Published cell escaped the fixed allowlist.")
    files = _artifact_inventory(staging)
    for path in staging.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    payload = {
        "schema": "ra_adapt_appendix_cell_artifact_receipt_v1",
        "campaign_id": CAMPAIGN_ID,
        "cell_id": cell_id,
        "files": files,
        "files_sha256": hashlib.sha256(_canonical_bytes(files)).hexdigest(),
    }
    receipt = {**payload, "sha256": canonical_sha256_without_field(payload)}
    write_json_atomic_noreplace(staging / "artifact_receipt.json", receipt)
    _fsync_directory(staging)
    final.parent.mkdir(parents=True, exist_ok=True)
    if staging.stat().st_dev != final.parent.stat().st_dev:
        raise ValueError("Cell staging and final directories must share a filesystem.")
    os.rename(staging, final)
    _fsync_directory(final.parent)
    return validate_published_cell(final)


def validate_published_cell(final_dir: Path) -> dict[str, Any]:
    """Rehash one immutable published cell before reuse or aggregation."""

    final = Path(final_dir)
    if not final.is_dir() or final.is_symlink():
        raise ValueError("Published cell path must be a real directory.")
    receipt_path = final / "artifact_receipt.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("Published cell lost its artifact receipt.") from exc
    receipt = _require_signed_mapping(receipt, owner="cell artifact receipt")
    if (
        receipt.get("schema") != "ra_adapt_appendix_cell_artifact_receipt_v1"
        or receipt.get("campaign_id") != CAMPAIGN_ID
        or receipt.get("cell_id") != final.name
    ):
        raise ValueError("Published cell receipt identity drifted.")
    files = _artifact_inventory(final)
    if (
        files != receipt.get("files")
        or hashlib.sha256(_canonical_bytes(files)).hexdigest()
        != receipt.get("files_sha256")
    ):
        raise ValueError("Published cell file inventory or hash drifted.")
    return receipt


def materialize_execution_plan(
    plan: Mapping[str, Any],
    *,
    runtime_root: Path | None = None,
) -> dict[str, Any]:
    """Resolve all four L=3 cell contracts through the named appendix API."""

    expected = build_campaign_plan()
    if dict(plan) != expected:
        raise ValueError(
            "Only the exact fixed campaign plan can be materialized."
        )

    cells: list[dict[str, Any]] = []
    for raw_cell in expected["cells"]:
        cell = dict(raw_cell)
        observation_root = (
            Path("/__ra_adapt_appendix_contract__") / cell["cell_id"]
            if runtime_root is None
            else Path(runtime_root)
            / "staging"
            / f"{cell['cell_id']}__l3_v8"
            / "observations"
        )
        # One construction seam: the child re-derives the identical
        # application from the same helper and compares digests.
        _problem, _request, source = build_cell_application(
            cell, output_dir=observation_root
        )
        if source.get("sha256") != canonical_sha256_without_field(source):
            raise ValueError(
                "Application source digest changed during materialization."
            )
        payload = {**cell, "application_source": source}
        cells.append({**payload, "sha256": canonical_sha256_without_field(payload)})
    payload = {
        "schema": "ra_adapt_appendix_l3_materialized_plan_v3",
        "campaign_id": CAMPAIGN_ID,
        "campaign_plan_sha256": expected["sha256"],
        "cells": cells,
    }
    return {**payload, "sha256": canonical_sha256_without_field(payload)}


def _csv_text(fieldnames: list[str], rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def build_report_documents(
    *,
    source_facts: Mapping[str, Any],
    campaign_plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
    summaries: Mapping[str, Mapping[str, Any]],
    cell_receipts: Mapping[str, Mapping[str, Any]],
    capacity_start: Mapping[str, Any],
    capacity_end: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete tracked L=3 report bundle from published summaries."""

    selected = select_execution_cells(campaign_plan, preflight)
    selected_ids = [str(cell["cell_id"]) for cell in selected]
    if set(summaries) != set(selected_ids) or set(cell_receipts) != set(
        selected_ids
    ):
        raise ValueError(
            "Reporting inputs do not match the selected execution cells."
        )
    for cell_id, summary in summaries.items():
        signed = _require_signed_mapping(summary, owner=f"summary {cell_id}")
        if signed.get("cell_id") != cell_id:
            raise ValueError("Reporting summary identity drifted.")

    baseline_cell_id = f"l3_baseline_h{CAMPAIGN_HORIZON}"
    baseline = summaries[baseline_cell_id]
    thresholds = {
        cell_id: {
            f"{threshold:.0e}": _threshold_hit(summaries[cell_id], threshold)
            for threshold in ACCURACY_THRESHOLDS
        }
        for cell_id in selected_ids
    }
    comparisons: list[dict[str, Any]] = []
    matched_csv_rows: list[dict[str, Any]] = []
    for demonstration, arm in (
        ("metric_ablation", "metric_ablation"),
        ("metric_ablation_gated", "metric_ablation_gated"),
        ("batching", "batching"),
        ("beam", "beam"),
        ("beam_cost", "beam_cost"),
    ):
        feature_cell_id = f"l3_{arm}_h{CAMPAIGN_HORIZON}"
        feature = summaries[feature_cell_id]
        rows = build_matched_accuracy_rows(baseline, feature)
        baseline_first = thresholds[baseline_cell_id]
        feature_first = thresholds[feature_cell_id]
        beats_before_round_three = bool(
            any(
                feature_first[key] is not None
                and feature_first[key] < 3
                and (
                    baseline_first[key] is None
                    or feature_first[key] < baseline_first[key]
                )
                for key in feature_first
            )
        )
        comparisons.append(
            {
                "demonstration": demonstration,
                "num_sites": 3,
                "baseline_cell_id": baseline["cell_id"],
                "feature_cell_id": feature["cell_id"],
                "baseline_first_round_by_threshold": dict(baseline_first),
                "feature_first_round_by_threshold": dict(feature_first),
                "beats_minimal_ra_before_round_three": (
                    beats_before_round_three
                ),
                "beats_minimal_ra_at_any_threshold": bool(
                    any(
                        feature_first[key] is not None
                        and (
                            baseline_first[key] is None
                            or feature_first[key] < baseline_first[key]
                        )
                        for key in feature_first
                    )
                ),
                "rounds_saved_by_threshold": {
                    key: (
                        None
                        if feature_first[key] is None
                        or baseline_first[key] is None
                        else baseline_first[key] - feature_first[key]
                    )
                    for key in feature_first
                },
                "baseline_accepted_operator_count": int(
                    baseline["accepted_rounds"][-1]["active_ansatz_depth"]
                )
                if baseline["accepted_rounds"]
                else 0,
                "feature_accepted_operator_count": int(
                    feature["accepted_rounds"][-1]["active_ansatz_depth"]
                )
                if feature["accepted_rounds"]
                else 0,
                "rows": rows,
                "feature_events": (
                    {
                        "pruning_nomination_count": int(
                            feature.get("pruning_nomination_count", 0)
                        ),
                        "accepted_deletion_count": int(
                            feature.get("accepted_deletion_count", 0)
                        ),
                        "pruning_trials": feature.get("pruning_trials", []),
                    }
                    if demonstration == "metric_ablation"
                    else {
                        "branches_explored": len(
                            feature.get("explored_branch_sequences", [])
                        ),
                        "explored_branch_sequences": feature.get(
                            "explored_branch_sequences", []
                        ),
                    }
                    if demonstration == "beam"
                    else {
                        "accepted_record_sequence": feature.get(
                            "accepted_record_sequence", []
                        )
                    }
                ),
            }
        )
        for row in rows:
            for role in ("baseline", "feature"):
                observation = row[role]
                matched_csv_rows.append(
                    {
                        "demonstration": demonstration,
                        "num_sites": 3,
                        "threshold": row["threshold"],
                        "status": row["status"],
                        "role": role,
                        "cell_id": ""
                        if observation is None
                        else observation["cell_id"],
                        "controller_round": ""
                        if observation is None
                        else observation["controller_round"],
                        "N2q": ""
                        if observation is None
                        else observation["cost_tuple"]["N2q"],
                        "D2q": ""
                        if observation is None
                        else observation["cost_tuple"]["D2q"],
                        "Dc": ""
                        if observation is None
                        else observation["cost_tuple"]["Dc"],
                        "weighted_cost_per_energy_decrease": ""
                        if observation is None
                        else observation["weighted_cost_per_energy_decrease"],
                    }
                )
    matched_payload = {
        "schema": "ra_adapt_appendix_l3_matched_accuracy_v1",
        "campaign_id": CAMPAIGN_ID,
        "accuracy_thresholds": list(ACCURACY_THRESHOLDS),
        "compile_cost_weights": dict(COMPILE_COST_WEIGHTS),
        "comparisons": comparisons,
    }
    matched = {
        **matched_payload,
        "sha256": canonical_sha256_without_field(matched_payload),
    }

    trajectory_rows: list[dict[str, Any]] = []
    for cell_id in selected_ids:
        summary = summaries[cell_id]
        for row in summary["accepted_rounds"]:
            trajectory_rows.append(
                {
                    "cell_id": cell_id,
                    "arm": str(summary["arm"]),
                    "controller_round": int(row["controller_round"]),
                    "accepted_energy": float(row["accepted_energy"]),
                    "same_cutoff_absolute_error": float(
                        row["same_cutoff_absolute_error"]
                    ),
                    "active_ansatz_depth": int(row["active_ansatz_depth"]),
                    "N2q": int(row["compile"]["N2q"]),
                    "D2q": int(row["compile"]["D2q"]),
                    "Dc": int(row["compile"]["Dc"]),
                }
            )
    trajectory_payload = {
        "schema": "ra_adapt_appendix_l3_trajectories_v1",
        "campaign_id": CAMPAIGN_ID,
        "exact_same_cutoff_energy": float(
            baseline["exact_same_cutoff_energy"]
        ),
        "first_round_by_threshold": thresholds,
        "rows": trajectory_rows,
    }
    trajectories = {
        **trajectory_payload,
        "sha256": canonical_sha256_without_field(trajectory_payload),
    }

    pool_labels = [
        str(row["label"])
        for row in preflight["executed_pool"]["generators"]
    ]
    arm_events: dict[str, Any] = {}
    for cell_id in selected_ids:
        summary = summaries[cell_id]
        admitted = []
        counts: dict[str, int] = {}
        sequence = summary.get("accepted_record_sequence", [])
        for index, indices in enumerate(sequence, start=1):
            labels = [pool_labels[int(value)] for value in indices]
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
            admitted.append(
                {
                    "controller_round": index,
                    "admitted_count": len(labels),
                    "pool_indices": [int(value) for value in indices],
                    "admitted_labels": labels,
                }
            )
        explored = summary.get("explored_branch_sequences", []) or []
        arm_events[cell_id] = {
            "arm": str(summary["arm"]),
            "admitted_by_round": admitted,
            "admitted_generator_multiplicity": dict(
                sorted(counts.items())
            ),
            "distinct_admitted_generator_count": len(counts),
            "total_admitted_generator_count": sum(counts.values()),
            "pruning": {
                "nomination_count": int(
                    summary.get("pruning_nomination_count", 0)
                ),
                "accepted_deletion_count": int(
                    summary.get("accepted_deletion_count", 0)
                ),
                "deleted_labels": [
                    label
                    for trial in summary.get("pruning_trials", [])
                    for label in trial.get("deleted_labels", [])
                ],
                "trials": summary.get("pruning_trials", []),
            },
            "beam": {
                "branches_explored": len(explored),
                "explored_branch_sequences": explored,
            },
        }
    events_payload = {
        "schema": "ra_adapt_appendix_l3_generator_events_v1",
        "campaign_id": CAMPAIGN_ID,
        "pool_ordered_labels": pool_labels,
        "arms": arm_events,
    }
    generator_events = {
        **events_payload,
        "sha256": canonical_sha256_without_field(events_payload),
    }

    manifest_payload = {
        "schema": "ra_adapt_appendix_l3_campaign_manifest_v3",
        "campaign_id": CAMPAIGN_ID,
        "source": dict(source_facts),
        "campaign_plan_sha256": str(campaign_plan["sha256"]),
        "l3_pool_preflight_sha256": str(preflight["sha256"]),
        "executed_cell_count": len(selected_ids),
        "executed_cell_ids": selected_ids,
        "cell_artifact_receipts": {
            cell_id: dict(cell_receipts[cell_id]) for cell_id in selected_ids
        },
        "capacity_start": dict(capacity_start),
        "capacity_end": dict(capacity_end),
        "preserved_campaign_ids": list(PRESERVED_CAMPAIGN_IDS),
        "preserved_v5_report_root": PRESERVED_V5_REPORT_ROOT,
        "preserved_v6_report_root": PRESERVED_V6_REPORT_ROOT,
        "preserved_v7_report_root": PRESERVED_V7_REPORT_ROOT,
        "manuscript_repository_mutated": False,
        "chtc_submitted": False,
    }
    manifest = {
        **manifest_payload,
        "sha256": canonical_sha256_without_field(manifest_payload),
    }

    def _first(cell_id: str, key: str) -> str:
        value = thresholds[cell_id][key]
        return "not reached" if value is None else f"round {value}"

    report_lines = [
        "# RA-ADAPT appendix L=3 escalation",
        "",
        f"Source commit: `{source_facts.get('head', '')}`.",
        "",
        "Route: append-RA, no warm start, Powell inner optimizer, "
        "append-only insertion, selection cost "
        "K = 0.30 dN2q + 0.30 dD2q + 0.25 dDc.",
        "",
        "## Candidate supply",
        "",
        "The L=2 guarded single-Pauli exposure is sector-degenerate at L=3: "
        f"its {preflight['superseded_generator_count']} survivors are diagonal "
        "Z words with zero Hartree--Fock gradient and zero tangent "
        "self-metric, giving supported response rank "
        f"{preflight['superseded_supported_response_rank']}.",
        "",
        "This campaign runs the sector-guarded composite generator pool: "
        f"{preflight['executed_generator_count']} generators, supported "
        f"response rank {preflight['executed_supported_response_rank']}. "
        "The controller rank validator was not weakened.",
        "",
        "## Energy error by iteration",
        "",
        "| cell | "
        + " | ".join(f"r{index}" for index in range(1, CAMPAIGN_HORIZON + 1))
        + " |",
        "| --- | " + " | ".join("---" for _ in range(CAMPAIGN_HORIZON)) + " |",
    ]
    for cell_id in selected_ids:
        errors = {
            int(row["controller_round"]): float(
                row["same_cutoff_absolute_error"]
            )
            for row in summaries[cell_id]["accepted_rounds"]
        }
        cells_text = " | ".join(
            "-" if index not in errors else f"{errors[index]:.3e}"
            for index in range(1, CAMPAIGN_HORIZON + 1)
        )
        report_lines.append(f"| {cell_id} | {cells_text} |")
    report_lines += [
        "",
        "## First iteration reaching each accuracy",
        "",
        "| cell | 1e-2 | 1e-4 | 1e-6 |",
        "| --- | --- | --- | --- |",
    ]
    for cell_id in selected_ids:
        report_lines.append(
            f"| {cell_id} | {_first(cell_id, '1e-02')} | "
            f"{_first(cell_id, '1e-04')} | {_first(cell_id, '1e-06')} |"
        )
    report_lines += [
        "",
        "## Admitted and pruned generators",
        "",
        "| cell | admissions | distinct generators | prune nominations | "
        "accepted deletions |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cell_id in selected_ids:
        events = arm_events[cell_id]
        report_lines.append(
            f"| {cell_id} | {events['total_admitted_generator_count']} | "
            f"{events['distinct_admitted_generator_count']} | "
            f"{events['pruning']['nomination_count']} | "
            f"{events['pruning']['accepted_deletion_count']} |"
        )
    batch_cell = f"l3_batching_h{CAMPAIGN_HORIZON}"
    report_lines += [
        "",
        "### Batch membership",
        "",
        "| round | size | admitted generators |",
        "| --- | --- | --- |",
    ]
    for row in arm_events[batch_cell]["admitted_by_round"]:
        report_lines.append(
            f"| {row['controller_round']} | {row['admitted_count']} | "
            + ", ".join(f"`{label}`" for label in row["admitted_labels"])
            + " |"
        )
    beam_cell = f"l3_beam_h{CAMPAIGN_HORIZON}"
    report_lines += [
        "",
        "### Beam child and survivor accounting",
        "",
        f"Explored branch sequences: "
        f"{arm_events[beam_cell]['beam']['branches_explored']}. "
        "Survivors are the accepted lineage recorded round by round in "
        "`generator_events.json`.",
        "",
        "## Does each extension beat minimal RA?",
        "",
        "| extension | before round 3 | at any threshold | rounds saved "
        "to 1e-6 |",
        "| --- | --- | --- | --- |",
    ]
    for comparison in comparisons:
        saved = comparison["rounds_saved_by_threshold"].get("1e-06")
        report_lines.append(
            f"| {comparison['demonstration']} | "
            f"{comparison['beats_minimal_ra_before_round_three']} | "
            f"{comparison['beats_minimal_ra_at_any_threshold']} | "
            f"{'-' if saved is None else saved} |"
        )
    report_lines += [
        "",
        "The v1--v5 campaign evidence is preserved unchanged. No CHTC "
        "submission and no manuscript edit was made by this campaign.",
        "",
    ]
    report = "\n".join(report_lines)

    return {
        "campaign_manifest.json": manifest,
        "l3_pool_preflight.json": copy.deepcopy(dict(preflight)),
        "generator_events.json": generator_events,
        "matched_accuracy.json": matched,
        "matched_accuracy.csv": _csv_text(
            [
                "demonstration",
                "num_sites",
                "threshold",
                "status",
                "role",
                "cell_id",
                "controller_round",
                "N2q",
                "D2q",
                "Dc",
                "weighted_cost_per_energy_decrease",
            ],
            matched_csv_rows,
        ),
        "l3_trajectories.json": trajectories,
        "l3_trajectories.csv": _csv_text(
            [
                "cell_id",
                "arm",
                "controller_round",
                "accepted_energy",
                "same_cutoff_absolute_error",
                "active_ansatz_depth",
                "N2q",
                "D2q",
                "Dc",
            ],
            trajectory_rows,
        ),
        "appendix_l3_escalation_report.md": report,
    }


def inspect_source(repo_root: Path) -> dict[str, Any]:
    """Measure the exact clean Git source without invoking a shell."""

    root = Path(repo_root).resolve()

    def git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=check,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    head = git("rev-parse", "HEAD").stdout.strip()
    clean = git("status", "--porcelain", "--untracked-files=all").stdout == ""
    science = git(
        "merge-base", "--is-ancestor", REQUIRED_SCIENCE_ANCESTOR, head, check=False
    ).returncode == 0
    lane = git(
        "merge-base", "--is-ancestor", REQUIRED_APPENDIX_LANE_ANCESTOR, head, check=False
    ).returncode == 0
    facts = {
        "repo_root": str(root),
        "head": head,
        "required_ancestor": REQUIRED_SCIENCE_ANCESTOR,
        "required_ancestor_present": science,
        "required_appendix_lane_ancestor": REQUIRED_APPENDIX_LANE_ANCESTOR,
        "required_appendix_lane_ancestor_present": lane,
        "worktree_clean": clean,
        "runner_path": str(RUNNER_PATH.relative_to(root)),
        "runner_sha256": _file_sha256(RUNNER_PATH),
    }
    return validate_source_facts(facts)


def build_runner_contract(source_facts: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize the post-commit runner lock without self-reference."""

    payload = {
        "schema": "ra_adapt_appendix_l3_campaign_runner_contract_v3",
        "campaign_id": CAMPAIGN_ID,
        "source_head": str(source_facts["head"]),
        "runner_sha256": str(source_facts["runner_sha256"]),
        "argv": [],
        "maximum_simultaneous_children": 1,
        "launch_minimum_available_memory_bytes": (
            LAUNCH_MINIMUM_AVAILABLE_MEMORY_BYTES
        ),
        "launch_minimum_free_disk_bytes": LAUNCH_MINIMUM_FREE_DISK_BYTES,
        "runtime_minimum_available_memory_bytes": (
            RUNTIME_MINIMUM_AVAILABLE_MEMORY_BYTES
        ),
        "runtime_minimum_free_disk_bytes": RUNTIME_MINIMUM_FREE_DISK_BYTES,
        "maximum_child_rss_bytes": MAXIMUM_CHILD_RSS_BYTES,
        "remote_job_id": REMOTE_JOB_ID,
        "runtime_root": RUNTIME_RELATIVE_PATH,
        "report_root": REPORT_RELATIVE_PATH,
        "preserved_campaign_ids": list(PRESERVED_CAMPAIGN_IDS),
        "preserved_v5_report_root": PRESERVED_V5_REPORT_ROOT,
        "preserved_v5_manifest": PRESERVED_V5_MANIFEST,
        "preserved_v6_report_root": PRESERVED_V6_REPORT_ROOT,
        "preserved_v7_report_root": PRESERVED_V7_REPORT_ROOT,
        "generator_representation": GENERATOR_REPRESENTATION,
        "migration_policy": (
            "preserve_v1_v5_and_l3_v6_v7_evidence_and_run_six_fresh_h15_cells_v8"
        ),
        "capacity_native_probe_command": list(MEMORY_PRESSURE_COMMAND),
    }
    return {**payload, "sha256": canonical_sha256_without_field(payload)}


def capacity_snapshot(
    path: Path,
    *,
    phase: str,
    virtual_memory_reader: Any = None,
    disk_usage_reader: Any = None,
    native_probe_runner: Any = None,
    platform_reader: Any = None,
) -> dict[str, Any]:
    import psutil

    memory_reader = (
        psutil.virtual_memory
        if virtual_memory_reader is None
        else virtual_memory_reader
    )
    disk_reader = (
        shutil.disk_usage if disk_usage_reader is None else disk_usage_reader
    )
    probe_runner = (
        subprocess.run if native_probe_runner is None else native_probe_runner
    )
    system_reader = platform.system if platform_reader is None else platform_reader
    virtual_memory = memory_reader()
    system = system_reader()
    minimum_memory, _ = _capacity_thresholds(phase)
    native_probe_not_needed = bool(
        system == "Darwin"
        and int(virtual_memory.available) >= minimum_memory
    )
    output: str | None = None
    error: str | None = None
    if system == "Darwin" and not native_probe_not_needed:
        try:
            probe = probe_runner(
                list(MEMORY_PRESSURE_COMMAND),
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5.0,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            error = f"{type(exc).__name__}: {exc}"
        else:
            output = probe.stdout
            if probe.returncode != 0:
                error = (
                    f"memory_pressure exited {probe.returncode}: "
                    f"{probe.stderr.strip()}"
                )
    return build_capacity_receipt(
        psutil_available_memory_bytes=int(virtual_memory.available),
        total_physical_memory_bytes=int(virtual_memory.total),
        free_disk_bytes=int(disk_reader(path).free),
        platform_name=system,
        memory_pressure_output=output,
        memory_pressure_error=error,
        native_probe_not_needed=native_probe_not_needed,
        native_probe_phase=phase,
    )


def configure_science_environment() -> None:
    fixed = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
        "STATIC_ADAPT_ALLOCATED_CPUS": "1",
        "QISKIT_NUM_PROCS": "1",
        "QISKIT_PARALLEL": "FALSE",
        "RAYON_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
        "STATIC_ADAPT_HH_POOL_CACHE": "off",
        "STATIC_ADAPT_L3_ROUTE_POOL_CACHE": "off",
    }
    os.environ.update(fixed)


def _write_or_validate_json(path: Path, payload: Any) -> None:
    target = Path(path)
    if target.exists():
        try:
            observed = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Existing control file is invalid: {target}") from exc
        if observed != payload:
            raise ValueError(f"Existing control file drifted: {target}")
        return
    write_json_atomic_noreplace(target, payload)


def publish_status(
    *,
    runtime_root: Path,
    report_root: Path,
    phase: str,
    completed_cell_ids: list[str],
) -> None:
    """Publish one replaceable status pointer locally and to remote artifacts."""

    payload = {
        "schema": "ra_adapt_appendix_campaign_status_v1",
        "campaign_id": CAMPAIGN_ID,
        "phase": str(phase),
        "completed_cell_ids": list(completed_cell_ids),
        "completed_cell_count": len(completed_cell_ids),
        "runtime_root": str(runtime_root),
        "tracked_report_root": str(report_root),
        "process_id": os.getpid(),
    }

    def replace(path: Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.parent / f".{target.name}.tmp.{os.getpid()}.{time.time_ns()}"
        write_json_atomic_noreplace(temporary, payload)
        os.replace(temporary, target)
        _fsync_directory(target.parent)

    replace(runtime_root / "status.json")
    remote = os.environ.get("REMOTE_ARTIFACT_DIR")
    if remote:
        remote_root = Path(remote)
        if not remote_root.is_absolute():
            raise ValueError("REMOTE_ARTIFACT_DIR must be absolute when provided.")
        replace(remote_root / "status.json")


def build_cell_application(cell: Mapping[str, Any], *, output_dir: Path):
    from pipelines.static_adapt.ra_adapt.appendix_demos import (
        build_ra_adapt_appendix_pure_hubbard_problem,
        build_ra_adapt_appendix_request,
        ra_adapt_appendix_application_source_contract,
    )

    if cell["stage"] != "l3" or int(cell["num_sites"]) != 3:
        raise ValueError("The L=3 escalation campaign executes only L=3 cells.")
    if cell.get("noise_level") is not None:
        raise ValueError("The L=3 escalation campaign runs no noisy cells.")
    problem = build_ra_adapt_appendix_pure_hubbard_problem(num_sites=3)
    request = build_ra_adapt_appendix_request(
        arm=cell["arm"],
        maximum_controller_rounds=cell["horizon"],
        num_sites=3,
        output_dir=output_dir,
    )
    if str(request.adapter.candidate_representation_id) != (
        GENERATOR_REPRESENTATION
    ):
        raise ValueError("The L=3 cell lost its composite generator supply.")
    source = ra_adapt_appendix_application_source_contract(problem, request)
    return problem, request, source


def execute_cell_child(
    repo_root_text: str,
    runtime_root_text: str,
    materialized_cell: Mapping[str, Any],
) -> None:
    """Execute exactly one authenticated allowlisted cell in a direct child."""

    configure_science_environment()
    repo_root = Path(repo_root_text).resolve()
    runtime_root = Path(runtime_root_text).resolve()
    cell_contract = _require_signed_mapping(materialized_cell, owner="materialized cell")
    cell_id = str(cell_contract["cell_id"])
    allowed = {str(cell["cell_id"]): cell for cell in build_campaign_plan()["cells"]}
    raw_cell = {key: cell_contract[key] for key in allowed[cell_id]}
    if raw_cell != allowed.get(cell_id):
        raise ValueError("Child cell drifted from the fixed allowlist.")
    attempt = f"{cell_id}__l3_v8"
    staging = runtime_root / "staging" / attempt
    staging.mkdir(parents=True, exist_ok=False)
    try:
        problem, request, source = build_cell_application(
            raw_cell, output_dir=staging / "observations"
        )
        if source != cell_contract["application_source"]:
            raise ValueError("Child application source drifted from materialized plan.")
        from pipelines.static_adapt.ra_adapt import run_ra_adapt

        result = run_ra_adapt(problem, request)
        result_payload = {
            "protocol": result.protocol.to_dict(),
            "run": {
                "final_state": result.run.final_state.to_dict(),
                "accepted_trajectory": [
                    row.to_dict() for row in result.run.accepted_trajectory
                ],
                "accepted_transitions": [
                    row.to_dict() for row in result.run.accepted_transitions
                ],
                "stop": result.run.stop.to_dict(),
            },
            "scientific_receipts": copy.deepcopy(dict(result.scientific_receipts)),
        }
        summary = summarize_cell_result(
            cell=raw_cell,
            source_contract=source,
            result_payload=result_payload,
        )
        evidence = result.scientific_receipts["accepted_run_evidence"]
        replay = result.scientific_receipts["controller_replay_evidence"]
        identity = {
            "schema": "ra_adapt_appendix_cell_result_identity_v1",
            "cell_id": cell_id,
            "result_schema": result.schema,
            "protocol_sha256": result.protocol.sha256,
            "accepted_run_evidence_sha256": evidence["sha256"],
            "controller_replay_evidence_sha256": replay["sha256"],
            "stop": result.run.stop.to_dict(),
            "final_state": result.run.final_state.to_dict(),
            "numerical_physical_integrity": result.numerical_physical_integrity.to_dict(),
        }
        identity = {**identity, "sha256": canonical_sha256_without_field(identity)}
        write_json_atomic_noreplace(staging / "application_source.json", source)
        write_json_atomic_noreplace(staging / "accepted_run_evidence.json", evidence)
        write_json_atomic_noreplace(staging / "controller_replay_evidence.json", replay)
        write_json_atomic_noreplace(staging / "result_identity.json", identity)
        write_json_atomic_noreplace(staging / "cell_summary.json", summary)
        publish_staged_cell(
            staging,
            runtime_root / "cells" / cell_id,
            cell_id=cell_id,
        )
    except BaseException as exc:
        failure = {
            "schema": "ra_adapt_appendix_cell_failure_v1",
            "cell_id": cell_id,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        failure = {**failure, "sha256": canonical_sha256_without_field(failure)}
        try:
            write_json_atomic_noreplace(staging / "failure.json", failure)
            failure_root = runtime_root / "failures"
            failure_root.mkdir(parents=True, exist_ok=True)
            os.rename(
                staging,
                failure_root / f"{attempt}__failure__{time.time_ns()}",
            )
            _fsync_directory(failure_root)
        finally:
            raise


def process_tree_rss_bytes(root_process: Any) -> int:
    """Sum RSS for the direct child and every recursive descendant."""

    processes = [root_process, *list(root_process.children(recursive=True))]
    total = 0
    for process in processes:
        try:
            total += int(process.memory_info().rss)
        except Exception as exc:
            if type(exc).__name__ not in {"NoSuchProcess", "ZombieProcess"}:
                raise
    return total


def terminate_process_tree(
    root_process: Any | None,
    process: Any,
    *,
    wait_processes: Any,
) -> None:
    """Stop descendants first, then the supervised direct child."""

    try:
        descendants = (
            [] if root_process is None else list(root_process.children(recursive=True))
        )
    except Exception as exc:
        if type(exc).__name__ not in {"NoSuchProcess", "ZombieProcess"}:
            raise
        descendants = []
    for descendant in reversed(descendants):
        try:
            descendant.terminate()
        except Exception as exc:
            if type(exc).__name__ not in {"NoSuchProcess", "ZombieProcess"}:
                raise
    _, alive = wait_processes(descendants, 5.0)
    for descendant in alive:
        try:
            descendant.kill()
        except Exception as exc:
            if type(exc).__name__ not in {"NoSuchProcess", "ZombieProcess"}:
                raise
    if alive:
        wait_processes(alive, 5.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=30.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=5.0)


def monitor_child_process(
    process: Any,
    *,
    repo_root: Path,
    capacity_reader: Any = capacity_snapshot,
    process_reader: Any = None,
    wait_processes: Any = None,
    poll_seconds: float = 0.5,
) -> None:
    """Poll capacity and total process-tree RSS until the sole child exits."""

    import psutil

    reader = psutil.Process if process_reader is None else process_reader
    waiter = psutil.wait_procs if wait_processes is None else wait_processes
    root_process = None

    def read_capacity() -> Mapping[str, Any]:
        try:
            parameters = inspect.signature(capacity_reader).parameters.values()
        except (TypeError, ValueError):
            return capacity_reader(repo_root, phase="runtime")
        supports_phase = any(
            parameter.name == "phase"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        if supports_phase:
            return capacity_reader(repo_root, phase="runtime")
        return capacity_reader(repo_root)

    try:
        while process.is_alive():
            try:
                root_process = reader(process.pid)
            except psutil.NoSuchProcess:
                process.join(timeout=poll_seconds)
                continue
            require_capacity(read_capacity(), phase="runtime")
            require_child_rss(process_tree_rss_bytes(root_process))
            process.join(timeout=poll_seconds)
    except BaseException:
        terminate_process_tree(
            root_process,
            process,
            wait_processes=waiter,
        )
        raise


def archive_interrupted_staging(runtime_root: Path, *, cell_id: str) -> None:
    """Move an interrupted deterministic staging tree aside without deletion."""

    staging = runtime_root / "staging" / f"{cell_id}__l3_v8"
    if not staging.exists():
        return
    failure_root = runtime_root / "failures"
    failure_root.mkdir(parents=True, exist_ok=True)
    os.rename(
        staging,
        failure_root / f"{cell_id}__l3_v8__interrupted__{time.time_ns()}",
    )
    _fsync_directory(failure_root)


def run_cell_process(
    *,
    repo_root: Path,
    runtime_root: Path,
    materialized_cell: Mapping[str, Any],
) -> dict[str, Any]:
    """Run or reuse one cell while supervising the sole direct child RSS."""

    cell_id = str(materialized_cell["cell_id"])
    final = runtime_root / "cells" / cell_id
    if final.exists():
        return validate_published_cell(final)
    require_capacity(
        capacity_snapshot(repo_root, phase="runtime"),
        phase="runtime",
    )
    import multiprocessing
    process = multiprocessing.get_context("spawn").Process(
        target=execute_cell_child,
        args=(str(repo_root), str(runtime_root), dict(materialized_cell)),
        name=f"ra-appendix-{cell_id}",
    )
    process.start()
    try:
        monitor_child_process(process, repo_root=repo_root)
    except BaseException:
        archive_interrupted_staging(runtime_root, cell_id=cell_id)
        raise
    if process.exitcode != 0:
        archive_interrupted_staging(runtime_root, cell_id=cell_id)
        raise RuntimeError(f"Appendix cell {cell_id} failed with exit {process.exitcode}.")
    return validate_published_cell(final)


def _load_published_summary(cell_dir: Path) -> dict[str, Any]:
    validate_published_cell(cell_dir)
    value = json.loads((cell_dir / "cell_summary.json").read_text(encoding="utf-8"))
    return _require_signed_mapping(value, owner="published cell summary")


def _publish_report_documents(
    *, runtime_root: Path, report_root: Path, documents: Mapping[str, Any]
) -> dict[str, Any]:
    if report_root.exists():
        raise FileExistsError(f"Tracked report directory already exists: {report_root}")
    staging = runtime_root / "report_staging" / f"reports__{time.time_ns()}"
    staging.mkdir(parents=True, exist_ok=False)
    for name, value in documents.items():
        path = staging / name
        if isinstance(value, str):
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", closefd=False) as stream:
                    stream.write(value)
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                os.close(descriptor)
        else:
            write_json_atomic_noreplace(path, value)
    files = _artifact_inventory(staging)
    report_root.parent.mkdir(parents=True, exist_ok=True)
    if staging.stat().st_dev != report_root.parent.stat().st_dev:
        raise ValueError("Tracked reports and staging must share a filesystem.")
    os.rename(staging, report_root)
    _fsync_directory(report_root.parent)
    payload = {
        "schema": "ra_adapt_appendix_report_bundle_receipt_v1",
        "campaign_id": CAMPAIGN_ID,
        "files": files,
        "files_sha256": hashlib.sha256(_canonical_bytes(files)).hexdigest(),
    }
    receipt = {**payload, "sha256": canonical_sha256_without_field(payload)}
    write_json_atomic_noreplace(runtime_root / "report_bundle_receipt.json", receipt)
    return receipt


def run_fixed_campaign() -> int:
    configure_science_environment()
    repo_root = REPO_ROOT
    runtime_root = repo_root / RUNTIME_RELATIVE_PATH
    report_root = repo_root / REPORT_RELATIVE_PATH
    source = inspect_source(repo_root)
    runner_contract = build_runner_contract(source)
    runtime_root.mkdir(parents=True, exist_ok=True)
    with campaign_lock(runtime_root / "campaign.lock"):
        start = capacity_snapshot(repo_root, phase="launch")
        start_decision = build_capacity_decision(start, phase="launch")
        if start_decision["allowed"] is not True:
            blocked = {
                "schema": "ra_adapt_appendix_l3_capacity_block_v3",
                "campaign_id": CAMPAIGN_ID,
                "capacity": start,
                "capacity_decision": start_decision,
                "preserved_campaign_ids": list(PRESERVED_CAMPAIGN_IDS),
                "reason": str(start_decision["reason"]),
            }
            blocked = {
                **blocked,
                "sha256": canonical_sha256_without_field(blocked),
            }
            write_json_atomic_noreplace(
                runtime_root / f"capacity_block__{time.time_ns()}.json", blocked
            )
            raise CapacityBlocked(str(start_decision["reason"]))
        plan = build_campaign_plan()
        preflight = build_l3_pool_preflight()
        materialized = materialize_execution_plan(
            plan, runtime_root=runtime_root
        )
        _write_or_validate_json(runtime_root / "source_facts.json", source)
        _write_or_validate_json(
            runtime_root / "runner_contract.json", runner_contract
        )
        _write_or_validate_json(
            runtime_root / "campaign_plan.json", materialized
        )
        _write_or_validate_json(
            runtime_root / "l3_pool_preflight.json", preflight
        )
        materialized_by_id = {
            str(cell["cell_id"]): cell for cell in materialized["cells"]
        }
        selected = select_execution_cells(plan, preflight)
        receipts: dict[str, dict[str, Any]] = {}
        summaries: dict[str, dict[str, Any]] = {}
        publish_status(
            runtime_root=runtime_root,
            report_root=report_root,
            phase="running_l3",
            completed_cell_ids=[],
        )
        for cell in selected:
            cell_id = str(cell["cell_id"])
            receipts[cell_id] = run_cell_process(
                repo_root=repo_root,
                runtime_root=runtime_root,
                materialized_cell=materialized_by_id[cell_id],
            )
            summaries[cell_id] = _load_published_summary(
                runtime_root / "cells" / cell_id
            )
            publish_status(
                runtime_root=runtime_root,
                report_root=report_root,
                phase="running_l3",
                completed_cell_ids=list(summaries),
            )
        end = capacity_snapshot(repo_root, phase="runtime")
        documents = build_report_documents(
            source_facts=source,
            campaign_plan=plan,
            preflight=preflight,
            summaries=summaries,
            cell_receipts=receipts,
            capacity_start=start,
            capacity_end=end,
        )
        _publish_report_documents(
            runtime_root=runtime_root,
            report_root=report_root,
            documents=documents,
        )
        publish_status(
            runtime_root=runtime_root,
            report_root=report_root,
            phase="complete",
            completed_cell_ids=list(summaries),
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        print(
            "The fixed appendix campaign takes no arguments; its cell allowlist "
            "and completion gates are sealed.",
            file=sys.stderr,
        )
        return 2
    try:
        return run_fixed_campaign()
    except (CapacityBlocked, CampaignAlreadyRunning) as exc:
        print(str(exc), file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
