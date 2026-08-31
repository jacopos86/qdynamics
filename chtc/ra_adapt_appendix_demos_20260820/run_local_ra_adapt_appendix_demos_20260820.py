#!/usr/bin/env python3
"""Run the fixed corrected-route RA-ADAPT appendix campaign locally.

The command-line surface is deliberately empty.  The campaign is a fixed
allowlist; its L=3 cells are selected only from signed L=2 evidence and its
noise horizon is selected only from those promotion decisions.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
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

CAMPAIGN_ID = "ra_adapt_appendix_demos_20260820_v1"
REQUIRED_APPENDIX_LANE_ANCESTOR = "fd18c46804db2df8fb195a89406ca4db4ee94782"
REQUIRED_SCIENCE_ANCESTOR = "5495bfa6dfb475cf65421b7ffa76936eac15f3a6"
ACCURACY_THRESHOLDS = (1.0e-2, 1.0e-4, 1.0e-6)
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


def _noise_cell(*, u_over_t: float, level: str, horizon: int) -> dict[str, Any]:
    u_slug = str(u_over_t).replace(".", "p")
    return {
        "cell_id": f"noise_u{u_slug}_{level}_h{horizon}",
        "stage": "noise",
        "num_sites": 2,
        "u_over_t": u_over_t,
        "arm": "baseline",
        "noise_level": level,
        "horizon": horizon,
        "execution_entrypoint": "run_ra_adapt",
    }


def build_campaign_plan() -> dict[str, Any]:
    cells = [
        *(
            _science_cell(num_sites=2, arm=arm, horizon=15)
            for arm in ("baseline", "batching", "metric_ablation", "beam")
        ),
        *(
            _science_cell(num_sites=3, arm=arm, horizon=20)
            for arm in ("baseline", "batching", "metric_ablation", "beam")
        ),
        *(
            _noise_cell(u_over_t=u, level=level, horizon=horizon)
            for u in (1.5, 8.0)
            for level in ("low", "high", "extreme")
            for horizon in (10, 15)
        ),
    ]
    payload = {
        "schema": "ra_adapt_appendix_campaign_plan_v1",
        "campaign_id": CAMPAIGN_ID,
        "required_science_ancestor": REQUIRED_SCIENCE_ANCESTOR,
        "accuracy_thresholds": list(ACCURACY_THRESHOLDS),
        "compile_cost_weights": dict(COMPILE_COST_WEIGHTS),
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


def decide_l2_escalations(
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the three fixed L=2 promotion predicates without tuning."""

    required = {
        f"l2_{arm}_h15"
        for arm in ("baseline", "batching", "metric_ablation", "beam")
    }
    if set(summaries) != required:
        raise ValueError("L=2 decisions require exactly the four fixed L=2 cells.")
    for cell_id, summary in summaries.items():
        if summary.get("cell_id") != cell_id:
            raise ValueError("L=2 cell summary identity drifted.")

    baseline = summaries["l2_baseline_h15"]
    batching = summaries["l2_batching_h15"]
    primary = ACCURACY_THRESHOLDS[0]
    baseline_hit = _threshold_hit(baseline, primary)
    batching_hit = _threshold_hit(batching, primary)
    identical_records = (
        list(baseline.get("accepted_record_sequence", ()))
        == list(batching.get("accepted_record_sequence", ()))
    )
    batch_reasons: list[str] = []
    if baseline_hit is None or batching_hit is None:
        batch_reasons.append("primary_fixed_accuracy_threshold_censored")
    elif baseline_hit - batching_hit < 1:
        batch_reasons.append("accepted_round_gap_under_one")
    if identical_records:
        batch_reasons.append("identical_admission_records_every_round")

    pruning = summaries["l2_metric_ablation_h15"]
    nomination_count = int(pruning.get("pruning_nomination_count", -1))
    deletion_count = int(pruning.get("accepted_deletion_count", -1))
    if nomination_count < 0 or deletion_count < 0 or deletion_count > nomination_count:
        raise ValueError("Metric-ablation counts are invalid.")
    pruning_reasons: list[str] = []
    if nomination_count == 0:
        pruning_reasons.append("no_metric_pruning_nomination")
    elif deletion_count == 0:
        pruning_reasons.append("no_accepted_delete_refit")

    beam = summaries["l2_beam_h15"]
    baseline_sequence = list(baseline.get("accepted_record_sequence", ()))
    explored = beam.get("explored_branch_sequences")
    if not isinstance(explored, list):
        raise ValueError("Beam explored_branch_sequences must be a list.")
    divergence_offsets: list[int] = []
    for sequence in explored:
        branch = list(sequence)
        for offset, record in enumerate(branch, start=1):
            if offset > len(baseline_sequence) or record != baseline_sequence[offset - 1]:
                divergence_offsets.append(offset)
                break
    beam_reasons = (
        []
        if divergence_offsets
        else ["all_beam_branches_collapse_to_default_sequence"]
    )

    demonstrations = {
        "batching": {
            "promote_to_l3": bool(batch_reasons),
            "reasons": batch_reasons,
            "primary_threshold": primary,
            "baseline_round": baseline_hit,
            "feature_round": batching_hit,
            "accepted_round_gap": (
                None
                if baseline_hit is None or batching_hit is None
                else baseline_hit - batching_hit
            ),
            "identical_records_every_round": identical_records,
        },
        "metric_ablation": {
            "promote_to_l3": bool(pruning_reasons),
            "reasons": pruning_reasons,
            "nomination_count": nomination_count,
            "accepted_deletion_count": deletion_count,
        },
        "beam": {
            "promote_to_l3": bool(beam_reasons),
            "reasons": beam_reasons,
            "branches_explored": len(explored),
            "branch_divergence_round": (
                None if not divergence_offsets else min(divergence_offsets)
            ),
        },
    }
    promoted = [
        name
        for name in ("batching", "metric_ablation", "beam")
        if demonstrations[name]["promote_to_l3"]
    ]
    payload = {
        "schema": "ra_adapt_appendix_escalation_decisions_v1",
        "demonstrations": demonstrations,
        "promoted_demonstrations": promoted,
        "noise_horizon": 15 if promoted else 10,
    }
    return {**payload, "sha256": canonical_sha256_without_field(payload)}


def select_execution_cells(
    plan: Mapping[str, Any],
    decisions: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Resolve the ten-to-fourteen executed cells from the fixed allowlist."""

    if canonical_sha256_without_field(plan) != plan.get("sha256"):
        raise ValueError("Campaign plan digest is invalid.")
    if canonical_sha256_without_field(decisions) != decisions.get("sha256"):
        raise ValueError("Escalation decision digest is invalid.")
    cells = plan.get("cells")
    if not isinstance(cells, list):
        raise ValueError("Campaign plan cells must be a list.")
    by_id = {str(cell["cell_id"]): dict(cell) for cell in cells}
    if len(by_id) != len(cells):
        raise ValueError("Campaign plan contains repeated cell identities.")
    selected_ids = [
        "l2_baseline_h15",
        "l2_batching_h15",
        "l2_metric_ablation_h15",
        "l2_beam_h15",
    ]
    promoted = list(decisions.get("promoted_demonstrations", ()))
    if any(value not in {"batching", "metric_ablation", "beam"} for value in promoted):
        raise ValueError("Escalation decisions contain an unknown demonstration.")
    if promoted:
        selected_ids.append("l3_baseline_h20")
        selected_ids.extend(f"l3_{arm}_h20" for arm in promoted)
    noise_horizon = int(decisions.get("noise_horizon", -1))
    if noise_horizon != (15 if promoted else 10):
        raise ValueError("Noise horizon does not follow the promotion decision.")
    selected_ids.extend(
        f"noise_u{str(u).replace('.', 'p')}_{level}_h{noise_horizon}"
        for u in (1.5, 8.0)
        for level in ("low", "high", "extreme")
    )
    try:
        return [by_id[cell_id] for cell_id in selected_ids]
    except KeyError as exc:
        raise ValueError("Selected cell escaped the fixed campaign allowlist.") from exc


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


def require_capacity(snapshot: Mapping[str, Any], *, phase: str) -> None:
    """Enforce the launch or between-cell capacity floor exactly."""

    if phase == "launch":
        minimum_memory = LAUNCH_MINIMUM_AVAILABLE_MEMORY_BYTES
        minimum_disk = LAUNCH_MINIMUM_FREE_DISK_BYTES
    elif phase == "runtime":
        minimum_memory = RUNTIME_MINIMUM_AVAILABLE_MEMORY_BYTES
        minimum_disk = RUNTIME_MINIMUM_FREE_DISK_BYTES
    else:
        raise ValueError("Capacity phase must be launch or runtime.")
    available_memory = int(snapshot.get("available_memory_bytes", -1))
    free_disk = int(snapshot.get("free_disk_bytes", -1))
    if available_memory < minimum_memory:
        raise CapacityBlocked(
            f"{phase} available memory is below {minimum_memory // GIB} GiB."
        )
    if free_disk < minimum_disk:
        raise CapacityBlocked(
            f"{phase} free disk is below {minimum_disk // GIB} GiB."
        )


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
        if baseline_hit is None or feature_hit is None:
            rows.append(
                {
                    "threshold": float(threshold),
                    "status": "censored",
                    "baseline": None,
                    "feature": None,
                    "accepted_round_gap": None,
                }
            )
        else:
            rows.append(
                {
                    "threshold": float(threshold),
                    "status": "reached_by_both",
                    "baseline": baseline_hit,
                    "feature": feature_hit,
                    "accepted_round_gap": int(baseline_hit["controller_round"])
                    - int(feature_hit["controller_round"]),
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
    """Resolve all twenty cell contracts through the named appendix API."""

    expected = build_campaign_plan()
    if dict(plan) != expected:
        raise ValueError("Only the exact fixed campaign plan can be materialized.")
    from pipelines.static_adapt.ra_adapt.appendix_demos import (
        build_ra_adapt_appendix_noise_problem,
        build_ra_adapt_appendix_noise_request,
        build_ra_adapt_appendix_pure_hubbard_problem,
        build_ra_adapt_appendix_request,
        ra_adapt_appendix_application_source_contract,
    )

    cells: list[dict[str, Any]] = []
    for raw_cell in expected["cells"]:
        cell = dict(raw_cell)
        observation_root = (
            Path("/__ra_adapt_appendix_contract__") / cell["cell_id"]
            if runtime_root is None
            else Path(runtime_root)
            / "staging"
            / f"{cell['cell_id']}__v1"
            / "observations"
        )
        if cell["stage"] == "noise":
            problem = build_ra_adapt_appendix_noise_problem(u=cell["u_over_t"])
            request = build_ra_adapt_appendix_noise_request(
                noise_level=cell["noise_level"],
                maximum_controller_rounds=cell["horizon"],
                output_dir=observation_root,
            )
        else:
            problem = build_ra_adapt_appendix_pure_hubbard_problem(
                num_sites=cell["num_sites"]
            )
            request = build_ra_adapt_appendix_request(
                arm=cell["arm"],
                maximum_controller_rounds=cell["horizon"],
                output_dir=observation_root,
            )
        source = ra_adapt_appendix_application_source_contract(problem, request)
        if source.get("sha256") != canonical_sha256_without_field(source):
            raise ValueError("Application source digest changed during materialization.")
        payload = {**cell, "application_source": source}
        cells.append({**payload, "sha256": canonical_sha256_without_field(payload)})
    payload = {
        "schema": "ra_adapt_appendix_materialized_plan_v1",
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
    decisions: Mapping[str, Any],
    summaries: Mapping[str, Mapping[str, Any]],
    cell_receipts: Mapping[str, Mapping[str, Any]],
    capacity_start: Mapping[str, Any],
    capacity_end: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete tracked report bundle from published cell summaries."""

    selected = select_execution_cells(campaign_plan, decisions)
    selected_ids = [str(cell["cell_id"]) for cell in selected]
    if set(summaries) != set(selected_ids) or set(cell_receipts) != set(selected_ids):
        raise ValueError("Reporting inputs do not match the selected execution cells.")
    for cell_id, summary in summaries.items():
        signed = _require_signed_mapping(summary, owner=f"summary {cell_id}")
        if signed.get("cell_id") != cell_id:
            raise ValueError("Reporting summary identity drifted.")
    promoted = set(decisions["promoted_demonstrations"])
    comparisons: list[dict[str, Any]] = []
    matched_csv_rows: list[dict[str, Any]] = []
    for demonstration, arm in (
        ("batching", "batching"),
        ("metric_ablation", "metric_ablation"),
        ("beam", "beam"),
    ):
        sites = 3 if demonstration in promoted else 2
        horizon = 20 if sites == 3 else 15
        baseline = summaries[f"l{sites}_baseline_h{horizon}"]
        feature = summaries[f"l{sites}_{arm}_h{horizon}"]
        rows = build_matched_accuracy_rows(baseline, feature)
        comparison = {
            "demonstration": demonstration,
            "num_sites": sites,
            "baseline_cell_id": baseline["cell_id"],
            "feature_cell_id": feature["cell_id"],
            "promoted_from_l2": demonstration in promoted,
            "rows": rows,
            "feature_events": (
                feature.get("pruning_trials", [])
                if demonstration == "metric_ablation"
                else {
                    "branches_explored": len(
                        feature.get("explored_branch_sequences", [])
                    )
                }
                if demonstration == "beam"
                else None
            ),
        }
        comparisons.append(comparison)
        for row in rows:
            for role in ("baseline", "feature"):
                observation = row[role]
                matched_csv_rows.append(
                    {
                        "demonstration": demonstration,
                        "num_sites": sites,
                        "threshold": row["threshold"],
                        "status": row["status"],
                        "role": role,
                        "cell_id": "" if observation is None else observation["cell_id"],
                        "controller_round": "" if observation is None else observation["controller_round"],
                        "N2q": "" if observation is None else observation["cost_tuple"]["N2q"],
                        "D2q": "" if observation is None else observation["cost_tuple"]["D2q"],
                        "Dc": "" if observation is None else observation["cost_tuple"]["Dc"],
                        "weighted_cost_per_energy_decrease": ""
                        if observation is None
                        else observation["weighted_cost_per_energy_decrease"],
                    }
                )
    matched_payload = {
        "schema": "ra_adapt_appendix_matched_accuracy_v1",
        "campaign_id": CAMPAIGN_ID,
        "accuracy_thresholds": list(ACCURACY_THRESHOLDS),
        "compile_cost_weights": dict(COMPILE_COST_WEIGHTS),
        "comparisons": comparisons,
    }
    matched = {
        **matched_payload,
        "sha256": canonical_sha256_without_field(matched_payload),
    }
    noise_rows: list[dict[str, Any]] = []
    for cell in selected:
        if cell["stage"] != "noise":
            continue
        summary = summaries[cell["cell_id"]]
        accepted = list(summary["accepted_rounds"])
        terminal = accepted[-1] if accepted else None
        noise_rows.append(
            {
                "u_over_t": float(cell["u_over_t"]),
                "noise_level": str(cell["noise_level"]),
                "horizon": int(cell["horizon"]),
                "cell_id": str(cell["cell_id"]),
                "completed_controller_rounds": int(
                    summary["completed_controller_rounds"]
                ),
                "accepted_energy": None if terminal is None else terminal["accepted_energy"],
                "same_cutoff_absolute_error": None
                if terminal is None
                else terminal["same_cutoff_absolute_error"],
                "N2q": None if terminal is None else terminal["compile"]["N2q"],
                "D2q": None if terminal is None else terminal["compile"]["D2q"],
                "Dc": None if terminal is None else terminal["compile"]["Dc"],
            }
        )
    noise_payload = {
        "schema": "ra_adapt_appendix_noise_same_cutoff_v1",
        "campaign_id": CAMPAIGN_ID,
        "horizon": int(decisions["noise_horizon"]),
        "rows": noise_rows,
    }
    noise = {
        **noise_payload,
        "sha256": canonical_sha256_without_field(noise_payload),
    }
    manifest_payload = {
        "schema": "ra_adapt_appendix_campaign_manifest_v1",
        "campaign_id": CAMPAIGN_ID,
        "source": dict(source_facts),
        "campaign_plan_sha256": str(campaign_plan["sha256"]),
        "escalation_decisions_sha256": str(decisions["sha256"]),
        "executed_cell_count": len(selected_ids),
        "executed_cell_ids": selected_ids,
        "cell_artifact_receipts": {
            cell_id: dict(cell_receipts[cell_id]) for cell_id in selected_ids
        },
        "capacity_start": dict(capacity_start),
        "capacity_end": dict(capacity_end),
        "manuscript_repository_mutated": False,
    }
    manifest = {
        **manifest_payload,
        "sha256": canonical_sha256_without_field(manifest_payload),
    }
    report = "\n".join(
        [
            "# Corrected-route RA-ADAPT appendix demonstrations",
            "",
            f"Source commit: `{source_facts.get('head', '')}`. Cost: "
            "0.30 delta N2q + 0.30 delta D2q + 0.25 delta Dc.",
            "",
            "## Increment 1 — Batching",
            "",
            f"Escalation: {decisions['demonstrations']['batching']}.",
            "",
            "## Increment 2 — Generator ablation",
            "",
            f"Escalation: {decisions['demonstrations']['metric_ablation']}.",
            "",
            "## Increment 3 — Beam continuation",
            "",
            f"Escalation: {decisions['demonstrations']['beam']}.",
            "",
            "## Increment 4 — Noise diagnostic",
            "",
            f"Executed horizon: {decisions['noise_horizon']}; six L=2 cells.",
            "",
            "The Overleaf-synced manuscript repository was read-only for this campaign.",
            "",
        ]
    )
    return {
        "campaign_manifest.json": manifest,
        "escalation_decisions.json": copy.deepcopy(dict(decisions)),
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
        "noise_same_cutoff.json": noise,
        "noise_same_cutoff.csv": _csv_text(
            [
                "u_over_t",
                "noise_level",
                "horizon",
                "cell_id",
                "completed_controller_rounds",
                "accepted_energy",
                "same_cutoff_absolute_error",
                "N2q",
                "D2q",
                "Dc",
            ],
            noise_rows,
        ),
        "appendix_demonstrations_report.md": report,
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
        "schema": "ra_adapt_appendix_campaign_runner_contract_v1",
        "campaign_id": CAMPAIGN_ID,
        "source_head": str(source_facts["head"]),
        "runner_sha256": str(source_facts["runner_sha256"]),
        "argv": [],
        "maximum_simultaneous_children": 1,
        "launch_minimum_available_memory_bytes": LAUNCH_MINIMUM_AVAILABLE_MEMORY_BYTES,
        "launch_minimum_free_disk_bytes": LAUNCH_MINIMUM_FREE_DISK_BYTES,
        "runtime_minimum_available_memory_bytes": RUNTIME_MINIMUM_AVAILABLE_MEMORY_BYTES,
        "runtime_minimum_free_disk_bytes": RUNTIME_MINIMUM_FREE_DISK_BYTES,
        "maximum_child_rss_bytes": MAXIMUM_CHILD_RSS_BYTES,
        "runtime_root": f"output/local_runs/{CAMPAIGN_ID}",
        "report_root": "chtc/ra_adapt_appendix_demos_20260820/results",
    }
    return {**payload, "sha256": canonical_sha256_without_field(payload)}


def capacity_snapshot(path: Path) -> dict[str, int]:
    import psutil

    return {
        "available_memory_bytes": int(psutil.virtual_memory().available),
        "free_disk_bytes": int(shutil.disk_usage(path).free),
    }


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
        build_ra_adapt_appendix_noise_problem,
        build_ra_adapt_appendix_noise_request,
        build_ra_adapt_appendix_pure_hubbard_problem,
        build_ra_adapt_appendix_request,
        ra_adapt_appendix_application_source_contract,
    )

    if cell["stage"] == "noise":
        problem = build_ra_adapt_appendix_noise_problem(u=cell["u_over_t"])
        request = build_ra_adapt_appendix_noise_request(
            noise_level=cell["noise_level"],
            maximum_controller_rounds=cell["horizon"],
            output_dir=output_dir,
        )
    else:
        problem = build_ra_adapt_appendix_pure_hubbard_problem(
            num_sites=cell["num_sites"]
        )
        request = build_ra_adapt_appendix_request(
            arm=cell["arm"],
            maximum_controller_rounds=cell["horizon"],
            output_dir=output_dir,
        )
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
    attempt = f"{cell_id}__v1"
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
    try:
        while process.is_alive():
            try:
                root_process = reader(process.pid)
            except psutil.NoSuchProcess:
                process.join(timeout=poll_seconds)
                continue
            require_capacity(capacity_reader(repo_root), phase="runtime")
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

    staging = runtime_root / "staging" / f"{cell_id}__v1"
    if not staging.exists():
        return
    failure_root = runtime_root / "failures"
    failure_root.mkdir(parents=True, exist_ok=True)
    os.rename(
        staging,
        failure_root / f"{cell_id}__v1__interrupted__{time.time_ns()}",
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
    require_capacity(capacity_snapshot(repo_root), phase="runtime")
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
    runtime_root = repo_root / "output" / "local_runs" / CAMPAIGN_ID
    report_root = repo_root / "chtc" / "ra_adapt_appendix_demos_20260820" / "results"
    source = inspect_source(repo_root)
    runner_contract = build_runner_contract(source)
    runtime_root.mkdir(parents=True, exist_ok=True)
    with campaign_lock(runtime_root / "campaign.lock"):
        start = capacity_snapshot(repo_root)
        try:
            require_capacity(start, phase="launch")
        except CapacityBlocked as exc:
            blocked = {
                "schema": "ra_adapt_appendix_capacity_block_v1",
                "capacity": start,
                "reason": str(exc),
            }
            blocked = {**blocked, "sha256": canonical_sha256_without_field(blocked)}
            write_json_atomic_noreplace(
                runtime_root / f"capacity_block__{time.time_ns()}.json", blocked
            )
            raise
        plan = build_campaign_plan()
        materialized = materialize_execution_plan(plan, runtime_root=runtime_root)
        _write_or_validate_json(runtime_root / "source_facts.json", source)
        _write_or_validate_json(runtime_root / "runner_contract.json", runner_contract)
        _write_or_validate_json(runtime_root / "campaign_plan.json", materialized)
        materialized_by_id = {
            str(cell["cell_id"]): cell for cell in materialized["cells"]
        }
        l2_ids = [
            "l2_baseline_h15",
            "l2_batching_h15",
            "l2_metric_ablation_h15",
            "l2_beam_h15",
        ]
        receipts: dict[str, dict[str, Any]] = {}
        summaries: dict[str, dict[str, Any]] = {}
        publish_status(
            runtime_root=runtime_root,
            report_root=report_root,
            phase="running_l2",
            completed_cell_ids=[],
        )
        for cell_id in l2_ids:
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
                phase="running_l2",
                completed_cell_ids=list(summaries),
            )
        decisions = decide_l2_escalations(summaries)
        _write_or_validate_json(runtime_root / "escalation_decisions.json", decisions)
        selected = select_execution_cells(plan, decisions)
        publish_status(
            runtime_root=runtime_root,
            report_root=report_root,
            phase="running_escalations_and_noise",
            completed_cell_ids=list(summaries),
        )
        for cell in selected[4:]:
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
                phase="running_escalations_and_noise",
                completed_cell_ids=list(summaries),
            )
        end = capacity_snapshot(repo_root)
        documents = build_report_documents(
            source_facts=source,
            campaign_plan=plan,
            decisions=decisions,
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
            "and escalation policy are sealed.",
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
