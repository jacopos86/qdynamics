#!/usr/bin/env python3
"""Build a provenance-bound Paper-II selector comparison.

The comparison is deliberately downstream of trajectory generation.  It
re-evaluates every available final t=20 trajectory on the locked exact grid,
keeps every same-cell route separate, and reports terminal circuit, complete
logical-estimator accounting, and support-motion burdens as separate coordinates.

The APM route identities are the historical unit-cost proxy, the repaired
analytical proxy, the realized-Gamma(v-star) compiled selector, and the
ideal-Gamma(v)-ranked compiled selector.  A fresh direct AVQDS rerun supplies
the fifth human-facing column when available.  The legacy internal route keys
remain unchanged so archived actions and receipts stay addressable.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from pipelines.time_dynamics.campaigns import paper_ii_staged_accuracy_v1
from pipelines.time_dynamics.diagnostics.exact_reference_store import (
    ExactReferenceArtifact,
    ExactReferenceStore,
)
from pipelines.time_dynamics.diagnostics.physical_errors import (
    combine_resumed_rows,
    physical_errors,
)
from pipelines.time_dynamics.run_lock import assert_comparable
from pipelines.time_dynamics.staged_campaign import AccuracyMetrics, CellKey


SCHEMA = "paper_ii_selector_comparison_v3"
ROUTE_ORDER = (
    "q_compiled",
    "gamma_compiled",
    "repaired_proxy",
    "old_proxy",
    "avqds",
)
ROUTE_LABELS = {
    "q_compiled": "realized Gamma(v*) compiled",
    "gamma_compiled": "ideal Gamma(v)-ranked compiled",
    "repaired_proxy": "repaired proxy",
    "old_proxy": "old proxy (historical stitched)",
    "avqds": "AVQDS",
}
ROUTE_BUNDLES = {
    "q_compiled": "paper_ii_exchange_q_compiled_v1",
    "gamma_compiled": "paper_ii_exchange_gamma_compiled_v1",
    "repaired_proxy": "paper_ii_exchange_legacy_proxy_fixed_v1",
    "old_proxy": "paper_ii_generalized_exchange_v2",
    "avqds": "avqds_adaptive_append_v1",
}
ACTION_SUFFIXES = {
    "q_compiled": "__exchange_q_compiled",
    "gamma_compiled": "__exchange_gamma_compiled",
    "repaired_proxy": "__exchange_legacy_proxy_fixed",
    "avqds": "__avqds",
}

HISTORICAL_STITCHED_ROUTES = frozenset({"old_proxy"})
APM_ROUTES = tuple(route for route in ROUTE_ORDER if route != "avqds")

_INSERTION_MARKER_RE = re.compile(r"::(?:insr|avqds)[0-9]+c[0-9]+o[0-9]+")
_OCCURRENCE_RE = re.compile(r"::ap_occ(?:urrence\[)?[0-9]+\]?")


@dataclass(frozen=True)
class RunSource:
    route: str
    cell_id: str
    run_paths: tuple[Path, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return dict(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _resolve_path(value: str | Path, *, repository_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return repository_root / path


def _resolve_existing_path(value: str | Path, *, roots: Sequence[Path]) -> Path:
    """Resolve a locked absolute path or a repository-relative rescued path."""

    path = Path(value).expanduser()
    if path.is_absolute():
        if path.is_file():
            return path
        raise FileNotFoundError(path)
    candidates = [root / path for root in roots]
    present = [candidate for candidate in candidates if candidate.is_file()]
    if not present:
        raise FileNotFoundError(
            f"locked source is absent below all artifact roots: {path}"
        )
    digests = {_sha256(candidate) for candidate in present}
    if len(digests) != 1:
        raise ValueError(f"rescued copies disagree for locked source: {path}")
    return present[0]


def _route_and_cell(action_id: str) -> tuple[str, str] | None:
    for route, suffix in ACTION_SUFFIXES.items():
        if action_id.endswith(suffix):
            return route, action_id[: -len(suffix)]
    return None


def discover_direct_runs(roots: Sequence[Path]) -> dict[tuple[str, str], RunSource]:
    """Find direct t=20 APM and AVQDS products below explicit narrow roots."""

    result: dict[tuple[str, str], RunSource] = {}
    for root in roots:
        if not root.exists():
            continue
        for run_path in sorted(root.rglob("run.json")):
            parsed = _route_and_cell(run_path.parent.name)
            if parsed is None:
                continue
            route, cell_id = parsed
            key = (cell_id, route)
            source = RunSource(route=route, cell_id=cell_id, run_paths=(run_path,))
            previous = result.get(key)
            if previous is not None:
                if _sha256(previous.run_paths[0]) != _sha256(run_path):
                    raise ValueError(f"conflicting direct runs for {cell_id}/{route}")
                continue
            result[key] = source
    return result


def _locked_reference(manifest_path: Path) -> ExactReferenceArtifact:
    manifest = _read_json(manifest_path)
    identity = {
        name: manifest[name]
        for name in (
            "schema",
            "propagator_schema",
            "seed_sha256",
            "drive_profile",
            "times",
            "substeps",
        )
    }
    data_path = manifest_path.with_name(str(manifest["data_file"]))
    store = ExactReferenceStore(manifest_path.parent.parent)
    return store._load_verified(  # noqa: SLF001 - locked read-only load
        key=str(manifest["key"]),
        identity=identity,
        manifest_path=manifest_path,
        data_path=data_path,
    )


def _validate_direct_source(source: RunSource) -> dict[str, Any]:
    run_path = source.run_paths[0]
    completion_path = run_path.with_name("completion_receipt.json")
    cost_path = run_path.with_name("terminal_qiskit_cost.json")
    completion = _read_json(completion_path)
    status = str(completion.get("status") or "")
    if status not in {"SUCCESS", "SUCCESS_EXACT"}:
        raise ValueError(f"direct run is not successful: {run_path}")
    if source.route == "avqds":
        if completion.get("route") not in {None, "avqds"}:
            raise ValueError(f"route mismatch for {run_path}")
    else:
        expected_bundle = ROUTE_BUNDLES[source.route]
        if completion.get("exchange_route_bundle") != expected_bundle:
            raise ValueError(f"route-bundle mismatch for {run_path}")
    if completion.get("trajectory_sha256") != _sha256(run_path):
        raise ValueError(f"trajectory digest mismatch for {run_path}")
    if completion.get("terminal_cost_sha256") != _sha256(cost_path):
        raise ValueError(f"worker terminal-cost digest mismatch for {run_path}")
    exact_work = None
    if status == "SUCCESS_EXACT":
        work_path = run_path.with_name("exact_estimator_work.json")
        exact_work = _read_json(work_path)
        if (
            completion.get("schema")
            != "paper_ii_final_estimator_telemetry_completion_v1"
            or completion.get("route") != source.route
            or completion.get("terminal_cost_status") != "SUCCESS"
            or completion.get("estimator_work_sha256") != _sha256(work_path)
            or exact_work.get("status") != "EXACT"
        ):
            raise ValueError(f"exact telemetry receipt mismatch for {run_path}")
    return {
        "completion_receipt": {
            "path": str(completion_path),
            "sha256": _sha256(completion_path),
        },
        "worker_terminal_cost": {
            "path": str(cost_path),
            "sha256": _sha256(cost_path),
        },
        "code_revision": str(completion.get("code_revision")),
        "route_contract_sha256": str(completion.get("exchange_route_contract_sha256")),
        "completion_status": status,
        "estimator_work_status": (
            None if exact_work is None else str(exact_work.get("status"))
        ),
    }


def _base_coordinate_id(label: str) -> str:
    """Collapse insertion-instance labels onto a stable generator identity."""

    value = str(label)
    marker = _INSERTION_MARKER_RE.search(value)
    if marker is not None:
        value = value[: marker.start()]
    value = _OCCURRENCE_RE.sub("", value)
    return value


def _counter_delta(
    before: Counter[str], after: Counter[str]
) -> tuple[Counter[str], Counter[str]]:
    return after - before, before - after


def _short_cycle_count(
    events: Sequence[Mapping[str, Counter[str]]],
    *,
    source_key: str,
    target_key: str,
    window: int,
) -> int:
    """Greedily match source events to a later inverse event within ``window``."""

    waiting: dict[str, deque[int]] = defaultdict(deque)
    count = 0
    for event_index, event in enumerate(events):
        for base, multiplicity in event[source_key].items():
            waiting[base].extend([event_index] * int(multiplicity))
        for base, multiplicity in event[target_key].items():
            queue = waiting[base]
            while queue and event_index - queue[0] > int(window):
                queue.popleft()
            for _ in range(int(multiplicity)):
                while queue and event_index - queue[0] > int(window):
                    queue.popleft()
                if not queue:
                    break
                queue.pop()
                count += 1
    return int(count)


def support_motion(
    runs: Sequence[Mapping[str, Any]],
    *,
    short_cycle_window: int = 2,
) -> dict[str, Any]:
    """Measure persisted net support motion and short generator recurrence."""

    if not runs:
        raise ValueError("structural churn needs at least one run segment")
    initial = tuple(
        str(value)
        for value in (runs[0].get("initial_state") or {}).get(
            "runtime_coordinate_labels", ()
        )
    )
    if not initial:
        raise ValueError("run omits initial runtime-coordinate labels")
    current = Counter(initial)
    additions = Counter()
    deletions = Counter()
    base_events: list[dict[str, Counter[str]]] = []
    support_change_count = 0
    candidate_count_sum = 0
    scored_count_sum = 0
    scored_not_selected_lower_bound = 0
    reason_counts: Counter[str] = Counter()
    multi_commit_checkpoint_count = 0

    for run in runs:
        points = ((run.get("trajectory") or {}).get("points") or ())
        for point in points:
            decision = point.get("patch_decision") or {}
            metadata = decision.get("metadata") or {}
            if int(metadata.get("insertion_rounds") or 0) > 1:
                multi_commit_checkpoint_count += 1
            candidate_count_sum += int(decision.get("candidate_count") or 0)
            scored = int(decision.get("scored_count") or 0)
            scored_count_sum += scored
            accepted = bool(decision.get("accepted"))
            scored_not_selected_lower_bound += max(scored - int(accepted), 0)
            reason_counts[str(decision.get("reason") or "undeclared")] += 1

            fixed = point.get("fixed_step") or {}
            labels = tuple(str(value) for value in fixed.get("support_labels") or ())
            if not labels:
                raise ValueError("trajectory point omits fixed-step support labels")
            after = Counter(labels)
            added, deleted = _counter_delta(current, after)
            if added or deleted:
                support_change_count += 1
                additions.update(added)
                deletions.update(deleted)
                added_base = Counter()
                deleted_base = Counter()
                for label, multiplicity in added.items():
                    added_base[_base_coordinate_id(label)] += int(multiplicity)
                for label, multiplicity in deleted.items():
                    deleted_base[_base_coordinate_id(label)] += int(multiplicity)
                base_events.append({"added": added_base, "deleted": deleted_base})
            current = after

    added_count = int(sum(additions.values()))
    deleted_count = int(sum(deletions.values()))
    initial_count = len(initial)
    final_count = int(sum(current.values()))
    same_checkpoint_replacements = sum(
        sum((event["added"] & event["deleted"]).values()) for event in base_events
    )
    delete_to_readd = _short_cycle_count(
        base_events,
        source_key="deleted",
        target_key="added",
        window=short_cycle_window,
    )
    add_to_redelete = _short_cycle_count(
        base_events,
        source_key="added",
        target_key="deleted",
        window=short_cycle_window,
    )
    return {
        "initial_coordinate_count": initial_count,
        "final_coordinate_count": final_count,
        "support_changing_checkpoint_count": support_change_count,
        "added_coordinate_count": added_count,
        "deleted_coordinate_count": deleted_count,
        "short_cycle_window_support_changes": int(short_cycle_window),
        "short_delete_to_readd_count": delete_to_readd,
        "short_add_to_redelete_count": add_to_redelete,
        "same_checkpoint_base_replacement_count": int(same_checkpoint_replacements),
        "persisted_candidate_count_sum": candidate_count_sum,
        "persisted_scored_count_sum": scored_count_sum,
        "persisted_scored_not_selected_lower_bound": (
            scored_not_selected_lower_bound
        ),
        "decision_reason_counts": dict(sorted(reason_counts.items())),
        "multi_commit_checkpoint_count": int(multi_commit_checkpoint_count),
        "support_event_ledger_complete": bool(
            multi_commit_checkpoint_count == 0
        ),
    }


# Compatibility for callers of the first report prototype.  New artifacts and
# reader-facing text use ``support_motion`` and do not emit gross/excess churn.
structural_churn = support_motion


def deletion_gate_outcomes(
    runs: Sequence[Mapping[str, Any]],
    *,
    applicable: bool,
) -> dict[str, Any]:
    """Count persisted deletion-bearing finalist outcomes.

    The selector records every materialized finalist under
    ``patch_decision.metadata.attempts``.  Older payloads cap this list at the
    last 16 attempts per checkpoint, so the result explicitly says whether the
    counts are exact or lower bounds.  Rejections before materialization are
    kept separate from failures of the post-delete certification/realized
    authority gates.
    """

    if not applicable:
        return {
            "applicable": False,
            "attempt_record_complete": True,
            "deletion_bearing_finalist_attempt_count": None,
            "pre_certification_rejection_count": None,
            "post_delete_gate_failure_count": None,
            "certified_deletion_bearing_finalist_count": None,
            "committed_deletion_patch_count": None,
            "post_delete_failure_reason_counts": {},
            "pre_certification_failure_reason_counts": {},
            "multi_commit_checkpoint_count": 0,
        }

    complete = True
    multi_commit_checkpoint_count = 0
    deletion_attempts = 0
    pre_certification_rejections = 0
    post_delete_failures = 0
    certified_deletion_finalists = 0
    committed_deletion_patches = 0
    post_reasons: Counter[str] = Counter()
    pre_reasons: Counter[str] = Counter()
    pre_certification_prefixes = (
        "insertion_resolution_failed:",
        "compiled_word_identity_mismatch",
    )

    for run in runs:
        for point in ((run.get("trajectory") or {}).get("points") or ()):
            decision = point.get("patch_decision") or {}
            if bool(decision.get("accepted")) and str(decision.get("patch_kind")) in {
                "delete",
                "exchange",
            }:
                committed_deletion_patches += 1
            metadata = decision.get("metadata") or {}
            if int(metadata.get("insertion_rounds") or 0) > 1:
                # The point retains only the final selector invocation after
                # an in-checkpoint exchange loop.  Earlier accepted rounds
                # and their failed finalists are not present in ``attempts``.
                complete = False
                multi_commit_checkpoint_count += 1
            attempts = tuple(metadata.get("attempts") or ())
            declared = int(metadata.get("attempt_count", len(attempts)) or 0)
            if declared != len(attempts):
                complete = False
            for attempt in attempts:
                if not tuple(attempt.get("removed_runtime_indices") or ()):
                    continue
                deletion_attempts += 1
                reason = str(attempt.get("reason") or "undeclared")
                if reason == "certified":
                    certified_deletion_finalists += 1
                elif reason.startswith(pre_certification_prefixes):
                    pre_certification_rejections += 1
                    pre_reasons[reason] += 1
                else:
                    post_delete_failures += 1
                    post_reasons[reason] += 1

    return {
        "applicable": True,
        "attempt_record_complete": bool(complete),
        "deletion_bearing_finalist_attempt_count": int(deletion_attempts),
        "pre_certification_rejection_count": int(pre_certification_rejections),
        "post_delete_gate_failure_count": int(post_delete_failures),
        "certified_deletion_bearing_finalist_count": int(
            certified_deletion_finalists
        ),
        "committed_deletion_patch_count": int(committed_deletion_patches),
        "post_delete_failure_reason_counts": dict(sorted(post_reasons.items())),
        "pre_certification_failure_reason_counts": dict(sorted(pre_reasons.items())),
        "multi_commit_checkpoint_count": int(multi_commit_checkpoint_count),
    }


def integration_geometry_work(runs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Return a logical scalar-estimator proxy for integration RHS work.

    One RHS geometry at support size p is charged p(p+1)/2 symmetric metric
    entries, p force entries, and two scalar observables.  This is not a
    hardware shot count. Checkpoint, candidate-acquisition, certification, and
    local-refit work are accounted separately by :func:`complete_estimator_work`.
    """

    rhs_count = 0
    logical_scalar_estimators = 0
    for run in runs:
        for row in run.get("plot_rows") or ():
            parameter_rhs = int(row.get("parameter_control_rhs_evaluation_count") or 0)
            integration_rhs = int(row.get("integration_rhs_evaluation_count") or 0)
            if parameter_rhs and integration_rhs:
                raise ValueError("one interval declares two active RHS channels")
            evaluations = parameter_rhs if parameter_rhs else integration_rhs
            parameters = int(row["runtime_parameter_count"])
            if parameters < 0 or evaluations < 0:
                raise ValueError("negative estimator-work coordinate")
            rhs_count += evaluations
            per_rhs = parameters * (parameters + 1) // 2 + parameters + 2
            logical_scalar_estimators += evaluations * per_rhs
    summary_rhs = sum(int((run.get("summary") or {})["rhs_evaluation_count"]) for run in runs)
    if rhs_count != summary_rhs:
        raise ValueError(
            f"RHS telemetry mismatch: row sum {rhs_count} != summary {summary_rhs}"
        )
    return {
        "rhs_geometry_evaluation_count": int(rhs_count),
        "logical_integration_geometry_work": int(logical_scalar_estimators),
    }


def _geometry_entry_count(parameter_count: int) -> int:
    """Logical entries in one full McLachlan geometry at support size ``p``."""

    p = int(parameter_count)
    if p < 0:
        raise ValueError("negative geometry dimension")
    return p * (p + 1) // 2 + p + 2


def _apm_positioned_column_count(
    decision: Mapping[str, Any],
) -> tuple[int, bool, str]:
    """Infer the positioned insertion-cache width for one persisted call.

    Current compiled selectors persist the exact cache width as
    ``placements_chosen``.  Proxy selectors expose the singleton/no-deletion
    population, whose size is the same width.  Older artifacts may expose
    only the deduplicated atom pool; because every atom has at least one legal
    retained cut, that is a conservative lower bound on positioned columns.
    """

    metadata = decision.get("metadata") or {}
    if metadata.get("insertions_enabled") is False:
        return 0, True, "insertions_disabled"

    selector_route = metadata.get("selector_route") or {}
    compile_telemetry = selector_route.get("compile") or {}
    placements = compile_telemetry.get("placements_chosen")
    if placements is not None:
        value = int(placements)
        if value < 0:
            raise ValueError("negative positioned-candidate count")
        return value, True, "compiled_placements_chosen"

    levels = tuple(selector_route.get("levels") or ())
    for level in levels:
        insert_size = level.get("insert_population_size")
        delete_size = level.get("delete_population_size")
        if insert_size is not None and int(delete_size or 0) == 0:
            value = int(insert_size)
            if value < 0:
                raise ValueError("negative singleton insertion population")
            return value, True, "singleton_d0_population"

    work_guard = metadata.get("work_guard") or {}
    admitted = tuple(str(value) for value in work_guard.get("admitted_families") or ())
    if admitted and admitted[0] == "singleton_d0":
        scored = int(work_guard.get("scored_count") or 0)
        if scored > 0:
            # The raw singleton family includes stay followed by all M
            # positioned insertion columns.
            return scored - 1, True, "singleton_d0_work_guard"

    pool = int(metadata.get("candidate_pool_deduplicated") or 0)
    if pool < 0:
        raise ValueError("negative candidate-pool size")
    if pool > 0:
        return pool, False, "deduplicated_atom_pool_lower_bound"
    return 0, True, "empty_candidate_pool"


def _final_selector_pre_support(
    decision: Mapping[str, Any],
    *,
    checkpoint_pre: int,
    checkpoint_post: int,
) -> tuple[int, bool]:
    """Recover support size entering the final persisted selector call."""

    if not bool(decision.get("accepted")):
        return int(checkpoint_post), True
    metadata = decision.get("metadata") or {}
    committed = metadata.get("committed") or {}
    if committed:
        removed = len(tuple(committed.get("removed_runtime_indices") or ()))
        inserted = len(tuple(committed.get("inserted_selection") or ()))
        value = int(checkpoint_post) + removed - inserted
        if value < 0:
            raise ValueError("negative reconstructed selector support")
        return value, True
    # A legacy accepted payload without a committed record does not expose the
    # last patch cardinality.  The smaller endpoint is a safe dimension floor.
    return min(int(checkpoint_pre), int(checkpoint_post)), False


def _attempt_reached_full_geometry(attempt: Mapping[str, Any]) -> bool:
    """Whether a persisted finalist certainly reached full patched geometry."""

    reason = str(attempt.get("reason") or "")
    excluded_prefixes = (
        "insertion_resolution_failed:",
        "compiled_word_identity_mismatch",
        "materialization_failed:",
        # Geometry and solve share this exception label, so it cannot certify
        # that the geometry evaluation completed.
        "patched_solve_failed:",
    )
    return bool(reason) and not reason.startswith(excluded_prefixes)


def complete_estimator_work(
    runs: Sequence[Mapping[str, Any]],
    *,
    route: str,
) -> dict[str, Any]:
    """Account for propagation, checkpoint, candidate, and finalist geometry.

    Values are logical scalar-geometry entries under the same convention as
    :func:`integration_geometry_work`.  They are not hardware shots.  New
    artifacts carry exact per-round selector and refit ledgers; legacy or
    interrupted artifacts retain explicit lower-bound statuses.
    """

    result: dict[str, Any] = dict(integration_geometry_work(runs))
    checkpoint_count = 0
    checkpoint_work = 0
    candidate_evaluations = 0
    candidate_work = 0
    certification_evaluations = 0
    certification_work = 0
    refit_evaluations = 0
    refit_work = 0
    selector_call_count = 0
    candidate_exact = True
    certification_exact = True
    refit_exact = True
    selection_round_ledger_complete = True
    inference_counts: Counter[str] = Counter()

    for run in runs:
        initial_labels = tuple(
            str(value)
            for value in (run.get("initial_state") or {}).get(
                "runtime_coordinate_labels", ()
            )
        )
        if not initial_labels:
            raise ValueError("run omits initial runtime-coordinate labels")
        checkpoint_pre = len(initial_labels)
        for point in ((run.get("trajectory") or {}).get("points") or ()):
            fixed = point.get("fixed_step") or {}
            labels = tuple(str(value) for value in fixed.get("support_labels") or ())
            if not labels:
                raise ValueError("trajectory point omits fixed-step support labels")
            checkpoint_post = len(labels)
            checkpoint_count += 1
            checkpoint_work += _geometry_entry_count(checkpoint_pre)
            decision = point.get("patch_decision") or {}
            metadata = decision.get("metadata") or {}

            if route == "avqds":
                exact_work = metadata.get("estimator_work") or {}
                geometry_rounds = tuple(
                    metadata.get("candidate_geometry_rounds") or ()
                )
                if (
                    exact_work.get("schema")
                    == "paper_ii_avqds_estimator_work_v1"
                ):
                    declared_rounds = int(
                        exact_work.get(
                            "selector_round_count", len(geometry_rounds)
                        )
                    )
                    attempted = int(
                        exact_work.get(
                            "full_geometry_evaluation_attempt_count", 0
                        )
                    )
                    completed = int(
                        exact_work.get("full_geometry_evaluation_count", 0)
                    )
                    entries = int(
                        exact_work.get("logical_candidate_geometry_entries", 0)
                    )
                    rounds_consistent = all(
                        int(row.get("round_index", -1)) == expected_index
                        and int(row.get("base_runtime_parameter_count", -1)) >= 0
                        and int(row.get("candidate_runtime_parameter_count", -1))
                        == int(row.get("base_runtime_parameter_count", -1)) + 1
                        and int(row.get("materialization_failure_count", 0))
                        + int(row.get("geometry_evaluation_attempt_count", 0))
                        == int(row.get("candidate_pool_count", -1))
                        and 0
                        <= int(row.get("geometry_evaluation_count", -1))
                        <= int(row.get("geometry_evaluation_attempt_count", -1))
                        and int(row.get("finite_candidate_count", 0))
                        + int(row.get("nonfinite_candidate_count", 0))
                        == int(row.get("geometry_evaluation_count", -1))
                        and int(row.get("logical_geometry_entry_count", -1))
                        == int(row.get("geometry_evaluation_count", -1))
                        * _geometry_entry_count(
                            int(row.get("candidate_runtime_parameter_count", -1))
                        )
                        for expected_index, row in enumerate(geometry_rounds)
                    )
                    receipt_consistent = bool(
                        declared_rounds == len(geometry_rounds)
                        and rounds_consistent
                        and attempted
                        == sum(
                            int(
                                row.get(
                                    "geometry_evaluation_attempt_count", 0
                                )
                            )
                            for row in geometry_rounds
                        )
                        and completed
                        == sum(
                            int(row.get("geometry_evaluation_count", 0))
                            for row in geometry_rounds
                        )
                        and entries
                        == sum(
                            int(row.get("logical_geometry_entry_count", 0))
                            for row in geometry_rounds
                        )
                    )
                    selector_call_count += declared_rounds
                    candidate_evaluations += completed
                    candidate_work += entries
                    candidate_exact = bool(
                        candidate_exact
                        and receipt_consistent
                        and exact_work.get("complete", False)
                    )
                    selection_round_ledger_complete = bool(
                        selection_round_ledger_complete and receipt_consistent
                    )
                    checkpoint_pre = checkpoint_post
                    continue

                scored = int(
                    metadata.get("candidates_scored")
                    if metadata.get("candidates_scored") is not None
                    else decision.get("scored_count") or 0
                )
                if scored < 0:
                    raise ValueError("negative AVQDS candidate count")
                candidate_evaluations += scored
                if scored:
                    # Per-round dimensions are not persisted.  Every tested
                    # append has at least checkpoint_pre + 1 coordinates.
                    candidate_work += scored * _geometry_entry_count(
                        checkpoint_pre + 1
                    )
                    candidate_exact = False
                    selection_round_ledger_complete = False
                checkpoint_pre = checkpoint_post
                continue

            exact_rounds = metadata.get("selection_round_work")
            if (
                metadata.get("selection_round_work_schema")
                == "paper_ii_exchange_selection_round_work_v1"
                and isinstance(exact_rounds, list)
            ):
                declared_calls = int(
                    metadata.get("selector_call_count", len(exact_rounds))
                )
                selector_call_count += declared_calls
                declared_accepted = int(
                    metadata.get(
                        "accepted_selector_count",
                        sum(bool(row.get("accepted", False)) for row in exact_rounds),
                    )
                )
                ledger_consistent = bool(
                    declared_calls == len(exact_rounds)
                    and declared_accepted
                    == sum(bool(row.get("accepted", False)) for row in exact_rounds)
                )
                for expected_index, round_row in enumerate(exact_rounds):
                    if int(
                        round_row.get("selector_call_index", -1)
                    ) != expected_index:
                        ledger_consistent = False
                    work = round_row.get("estimator_work") or {}
                    if (
                        work.get("schema")
                        != "paper_ii_generalized_exchange_estimator_work_v1"
                    ):
                        ledger_consistent = False

                    selector_passes = tuple(work.get("selector_passes") or ())
                    declared_passes = int(
                        work.get("selector_pass_count", len(selector_passes))
                    )
                    cache_entries = int(
                        work.get("logical_shared_cache_entry_count", 0)
                    )
                    pass_rows_consistent = all(
                        int(row.get("pass_index", -1)) == pass_index
                        and int(row.get("pre_runtime_parameter_count", -1)) >= 0
                        and int(row.get("positioned_candidate_column_count", -1))
                        >= 0
                        and int(row.get("logical_shared_cache_entry_count", -1))
                        == int(row.get("pre_runtime_parameter_count", -1))
                        * int(row.get("positioned_candidate_column_count", -1))
                        + int(row.get("positioned_candidate_column_count", -1))
                        * (
                            int(row.get("positioned_candidate_column_count", -1))
                            + 3
                        )
                        // 2
                        for pass_index, row in enumerate(selector_passes)
                    )
                    pass_consistent = bool(
                        declared_passes == len(selector_passes)
                        and pass_rows_consistent
                        and int(
                            work.get("positioned_candidate_column_count", 0)
                        )
                        == sum(
                            int(row.get("positioned_candidate_column_count", 0))
                            for row in selector_passes
                        )
                        and cache_entries
                        == sum(
                            int(
                                row.get(
                                    "logical_shared_cache_entry_count", 0
                                )
                            )
                            for row in selector_passes
                        )
                        and all(
                            bool(row.get("complete", False))
                            for row in selector_passes
                        )
                    )
                    candidate_work += cache_entries
                    inference_counts["persisted_exact"] += len(selector_passes)
                    candidate_exact = bool(candidate_exact and pass_consistent)

                    attempts = tuple(work.get("certification_attempts") or ())
                    declared_attempts = int(
                        work.get("certification_attempt_count", len(attempts))
                    )
                    full_evaluations = int(
                        work.get("full_geometry_evaluation_count", 0)
                    )
                    full_entries = int(
                        work.get("logical_finalist_geometry_entry_count", 0)
                    )
                    refit_count = int(
                        work.get("refit_objective_evaluation_count", 0)
                    )
                    refit_entries = int(
                        work.get("logical_refit_entry_count", 0)
                    )
                    certification_rows_consistent = True
                    for attempt in attempts:
                        patched_count = attempt.get(
                            "patched_runtime_parameter_count"
                        )
                        full_attempted = int(
                            attempt.get("full_geometry_evaluation_attempt_count", 0)
                        )
                        full_completed = int(
                            attempt.get("full_geometry_evaluation_count", 0)
                        )
                        refit_attempted = int(
                            attempt.get(
                                "refit_objective_evaluation_attempt_count", 0
                            )
                        )
                        refit_completed = int(
                            attempt.get("refit_objective_evaluation_count", 0)
                        )
                        expected_full_entries = 0
                        expected_refit_entries = 0
                        if patched_count is not None:
                            patched_count = int(patched_count)
                            if patched_count >= 0:
                                expected_full_entries = (
                                    full_completed
                                    * _geometry_entry_count(patched_count)
                                )
                                expected_refit_entries = refit_completed * (
                                    patched_count + 1
                                )
                            else:
                                certification_rows_consistent = False
                        certification_rows_consistent = bool(
                            certification_rows_consistent
                            and attempt.get("schema")
                            == "paper_ii_exchange_certification_estimator_work_v1"
                            and 0 <= full_completed <= full_attempted
                            and 0 <= refit_completed <= refit_attempted
                            and (
                                patched_count is not None
                                or (
                                    full_attempted == 0
                                    and full_completed == 0
                                    and refit_attempted == 0
                                    and refit_completed == 0
                                )
                            )
                            and int(
                                attempt.get("logical_full_geometry_entry_count", -1)
                            )
                            == expected_full_entries
                            and int(attempt.get("logical_refit_entry_count", -1))
                            == expected_refit_entries
                            and int(
                                attempt.get("logical_total_estimator_entries", -1)
                            )
                            == expected_full_entries + expected_refit_entries
                        )
                    attempt_consistent = bool(
                        declared_attempts == len(attempts)
                        and certification_rows_consistent
                        and int(
                            work.get("full_geometry_evaluation_attempt_count", 0)
                        )
                        == sum(
                            int(
                                row.get(
                                    "full_geometry_evaluation_attempt_count", 0
                                )
                            )
                            for row in attempts
                        )
                        and full_evaluations
                        == sum(
                            int(
                                row.get("full_geometry_evaluation_count", 0)
                            )
                            for row in attempts
                        )
                        and full_entries
                        == sum(
                            int(
                                row.get(
                                    "logical_full_geometry_entry_count", 0
                                )
                            )
                            for row in attempts
                        )
                        and refit_count
                        == sum(
                            int(
                                row.get(
                                    "refit_objective_evaluation_count", 0
                                )
                            )
                            for row in attempts
                        )
                        and refit_entries
                        == sum(
                            int(row.get("logical_refit_entry_count", 0))
                            for row in attempts
                        )
                        and int(
                            work.get(
                                "refit_objective_evaluation_attempt_count", 0
                            )
                        )
                        == sum(
                            int(
                                row.get(
                                    "refit_objective_evaluation_attempt_count", 0
                                )
                            )
                            for row in attempts
                        )
                        and int(
                            work.get("logical_total_selector_estimator_entries", -1)
                        )
                        == cache_entries + full_entries + refit_entries
                        and int(work.get("attempt_record_count", 0))
                        == sum(
                            int(value)
                            for value in (
                                work.get("attempt_reason_counts") or {}
                            ).values()
                        )
                    )
                    certification_evaluations += full_evaluations
                    certification_work += full_entries
                    refit_evaluations += refit_count
                    refit_work += refit_entries
                    certification_exact = bool(
                        certification_exact
                        and attempt_consistent
                        and all(
                            bool(row.get("complete", False))
                            for row in attempts
                        )
                    )
                    refit_exact = bool(
                        refit_exact
                        and attempt_consistent
                        and all(
                            bool(row.get("refit_counter_available", False))
                            and int(
                                row.get(
                                    "refit_objective_evaluation_attempt_count",
                                    0,
                                )
                            )
                            == int(
                                row.get(
                                    "refit_objective_evaluation_count", 0
                                )
                            )
                            for row in attempts
                            if bool(row.get("refit_requested", False))
                        )
                    )
                    ledger_consistent = bool(
                        ledger_consistent
                        and pass_consistent
                        and attempt_consistent
                        and bool(round_row.get("complete", False))
                        == bool(work.get("complete", False))
                    )
                selection_round_ledger_complete = bool(
                    selection_round_ledger_complete and ledger_consistent
                )
                if not ledger_consistent:
                    candidate_exact = False
                    certification_exact = False
                    refit_exact = False
                checkpoint_pre = checkpoint_post
                continue

            rounds = int(metadata.get("insertion_rounds") or 0)
            if rounds > 1:
                selection_round_ledger_complete = False
                candidate_exact = False
                certification_exact = False

            selector_pre, selector_pre_exact = _final_selector_pre_support(
                decision,
                checkpoint_pre=checkpoint_pre,
                checkpoint_post=checkpoint_post,
            )
            columns, columns_exact, inference = _apm_positioned_column_count(decision)
            inference_counts[inference] += 1
            if columns:
                candidate_work += (
                    selector_pre * columns + columns * (columns + 3) // 2
                )
            if not (columns_exact and selector_pre_exact):
                candidate_exact = False

            attempts = tuple(metadata.get("attempts") or ())
            declared = int(metadata.get("attempt_count", len(attempts)) or 0)
            if declared != len(attempts):
                certification_exact = False
            for attempt in attempts:
                if not _attempt_reached_full_geometry(attempt):
                    if str(attempt.get("reason") or "").startswith(
                        "patched_solve_failed:"
                    ):
                        certification_exact = False
                    continue
                removed = len(tuple(attempt.get("removed_runtime_indices") or ()))
                inserted = len(tuple(attempt.get("inserted_selection") or ()))
                patched_support = selector_pre - removed + inserted
                if patched_support < 0:
                    raise ValueError("negative finalist support dimension")
                certification_evaluations += 1
                certification_work += _geometry_entry_count(patched_support)
            if attempts:
                # Optional local refit objective evaluations are not persisted.
                certification_exact = False
                refit_exact = False
            checkpoint_pre = checkpoint_post

    if route == "avqds":
        certification_status = "not_applicable"
        certification_value: int | None = None
        certification_evaluation_value: int | None = None
        refit_status = "not_applicable"
        refit_value: int | None = None
        refit_evaluation_value: int | None = None
    else:
        certification_status = "exact" if certification_exact else "lower_bound"
        certification_value = int(certification_work)
        certification_evaluation_value = int(certification_evaluations)
        refit_status = "exact" if refit_exact else "lower_bound"
        refit_value = int(refit_work)
        refit_evaluation_value = int(refit_evaluations)

    candidate_status = "exact" if candidate_exact else "lower_bound"
    total = (
        checkpoint_work
        + candidate_work
        + certification_work
        + refit_work
        + int(result["logical_integration_geometry_work"])
    )
    total_status = (
        "exact"
        if candidate_status == "exact"
        and certification_status in {"exact", "not_applicable"}
        and refit_status in {"exact", "not_applicable"}
        and selection_round_ledger_complete
        else "lower_bound"
    )
    result.update(
        {
            "logical_checkpoint_base_geometry_work": int(checkpoint_work),
            "checkpoint_base_geometry_evaluation_count": int(checkpoint_count),
            "checkpoint_base_status": "exact",
            "candidate_full_geometry_evaluation_count": int(candidate_evaluations),
            "logical_candidate_acquisition_work": int(candidate_work),
            "candidate_acquisition_status": candidate_status,
            "certification_full_geometry_evaluation_count": (
                certification_evaluation_value
            ),
            "logical_certification_geometry_work": certification_value,
            "certification_geometry_status": certification_status,
            "refit_objective_evaluation_count": refit_evaluation_value,
            "logical_refit_estimator_work": refit_value,
            "refit_estimator_status": refit_status,
            "selector_call_count": int(selector_call_count),
            "logical_total_estimator_work": int(total),
            "logical_total_estimator_work_status": total_status,
            "selection_round_ledger_complete": bool(
                selection_round_ledger_complete
            ),
            "candidate_column_inference_counts": dict(
                sorted(inference_counts.items())
            ),
            "physical_shots": None,
            "physical_shots_status": "unavailable_statevector_no_shot_ledger",
        }
    )
    return result


def _accuracy_record(
    metrics: AccuracyMetrics,
    *,
    drive: str,
    controller: str,
    method: str,
    policy: Any,
) -> dict[str, Any]:
    key = CellKey(
        regime="hh_snake_nph1",
        drive=drive,
        method=str(method),
        controller=controller,
    )
    components = policy.response_components(metrics, key=key, horizon="t20")
    return {
        **metrics.to_json_dict(),
        "mean_abs_charge_imbalance_error": (
            2.0 * float(metrics.mean_max_abs_site_occupation_error)
        ),
        "qualifies_individually": bool(
            policy.qualifies(metrics, key=key, horizon="t20")
        ),
        "comparison_eligible": bool(
            policy.comparison_eligible(metrics, key=key, horizon="t20")
        ),
        "accuracy_burden": float(
            policy.accuracy_burden(metrics, key=key, horizon="t20")
        ),
        "response_components": {
            name: float(value) for name, value in components.items()
        },
    }


def _pair_match(left: Mapping[str, Any], right: Mapping[str, Any], policy: Any) -> dict[str, Any]:
    if not left["comparison_eligible"] or not right["comparison_eligible"]:
        return {
            "passes": False,
            "burden_ratio": None,
            "worst_component_ratio": None,
        }
    floor = float(policy.response_scale_floor)
    factor = float(policy.matched_accuracy_factor)
    deadband = float(policy.matched_component_deadband)
    left_burden = max(float(left["accuracy_burden"]), floor)
    right_burden = max(float(right["accuracy_burden"]), floor)
    burden_ratio = max(left_burden / right_burden, right_burden / left_burden)
    component_ratios = []
    for name in left["response_components"]:
        lval = max(float(left["response_components"][name]), deadband)
        rval = max(float(right["response_components"][name]), deadband)
        component_ratios.append(max(lval / rval, rval / lval))
    worst = max(component_ratios)
    return {
        "passes": bool(burden_ratio <= factor and worst <= factor),
        "burden_ratio": float(burden_ratio),
        "worst_component_ratio": float(worst),
    }


def _summarize_source(
    source: RunSource,
    *,
    drive: str,
    controller: str,
    reference: ExactReferenceArtifact,
    policy: Any,
    artifact_roots: Sequence[Path],
    expected_hashes: Sequence[str] | None = None,
) -> dict[str, Any]:
    runs = [_read_json(path) for path in source.run_paths]
    if expected_hashes is not None:
        if len(expected_hashes) != len(source.run_paths):
            raise ValueError("locked source/hash cardinality mismatch")
        for path, expected in zip(source.run_paths, expected_hashes):
            if _sha256(path) != expected:
                raise ValueError(f"locked trajectory digest mismatch: {path}")
    if len(runs) == 1:
        observed = tuple(runs[0].get("plot_rows") or ())
    elif len(runs) == 2:
        observed = combine_resumed_rows(
            tuple(runs[0].get("plot_rows") or ()),
            tuple(runs[1].get("plot_rows") or ()),
        )
    else:
        raise ValueError("only direct or one-boundary stitched runs are supported")
    if len(observed) != 501:
        raise ValueError(f"{source.cell_id}/{source.route} is not a 501-point t=20 run")
    metrics = physical_errors(observed, reference.rows)
    steps = sum(
        int((run.get("summary") or {})["accepted_internal_substep_count"])
        for run in runs
    )
    declared_runtime_source = runs[-1].get("source_artifact_json")
    runtime_source_record = None
    if declared_runtime_source is not None:
        try:
            resolved_runtime_source = _resolve_existing_path(
                str(declared_runtime_source),
                roots=artifact_roots,
            )
        except FileNotFoundError:
            runtime_source_record = {
                "declared_path": str(declared_runtime_source),
                "resolved_path": None,
                "sha256": None,
            }
        else:
            runtime_source_record = {
                "declared_path": str(declared_runtime_source),
                "resolved_path": str(resolved_runtime_source),
                "sha256": _sha256(resolved_runtime_source),
            }
    record = {
        "route": source.route,
        "route_label": ROUTE_LABELS[source.route],
        "route_bundle": ROUTE_BUNDLES[source.route],
        "cell_id": source.cell_id,
        "drive": drive,
        "controller": controller,
        "trajectory_kind": "historical_stitched" if len(runs) == 2 else "direct_t0_t20",
        "run_sources": [
            {"path": str(path), "sha256": _sha256(path)} for path in source.run_paths
        ],
        "terminal_runtime_source_artifact": runtime_source_record,
        "time_step_count": int(steps),
        "n_theta": int((runs[-1].get("summary") or {})["runtime_parameter_count_final"]),
        "accuracy": _accuracy_record(
            metrics,
            drive=drive,
            controller=controller,
            method="avqds" if source.route == "avqds" else "exchange",
            policy=policy,
        ),
        "estimator_work": complete_estimator_work(runs, route=source.route),
        "support_motion": support_motion(runs),
        "deletion_gate_outcomes": deletion_gate_outcomes(
            runs,
            applicable=source.route != "avqds",
        ),
        "run_locks": [run.get("run_lock") for run in runs],
    }
    if source.route not in HISTORICAL_STITCHED_ROUTES:
        record["completion"] = _validate_direct_source(source)
    return record


def build_comparison(
    *,
    repository_root: Path,
    lock_path: Path,
    direct_roots: Sequence[Path],
    artifact_roots: Sequence[Path],
) -> dict[str, Any]:
    lock = _read_json(lock_path)
    if lock.get("schema") != "paper_ii_final_matched_cells_v1":
        raise ValueError("unexpected final matched-cell lock schema")
    direct = discover_direct_runs(direct_roots)
    policy = paper_ii_staged_accuracy_v1.build_campaign(prior_root=None).accuracy_policy
    cells: list[dict[str, Any]] = []

    for locked_cell in lock.get("cells") or ():
        cell_id = str(locked_cell["cell_id"])
        drive = str(locked_cell["drive"])
        controller = str(locked_cell["controller"])
        exact_source = locked_cell["source_artifacts"]["exact_reference"]
        reference_path = _resolve_path(
            str(exact_source["path"]), repository_root=repository_root
        )
        reference = _locked_reference(reference_path)
        route_records: dict[str, Any] = {}
        for route, source_name in (("old_proxy", "apm"),):
            locked_sources = locked_cell["source_artifacts"][source_name]
            source = RunSource(
                route=route,
                cell_id=cell_id,
                run_paths=(
                    _resolve_existing_path(
                        str(locked_sources["screen_run_json"]["path"]),
                        roots=artifact_roots,
                    ),
                    _resolve_existing_path(
                        str(locked_sources["continuation_run_json"]["path"]),
                        roots=artifact_roots,
                    ),
                ),
            )
            route_records[route] = _summarize_source(
                source,
                drive=drive,
                controller=controller,
                reference=reference,
                policy=policy,
                artifact_roots=artifact_roots,
                expected_hashes=(
                    str(locked_sources["screen_run_json"]["sha256"]),
                    str(locked_sources["continuation_run_json"]["sha256"]),
                ),
            )
        for route in ("repaired_proxy", "q_compiled", "gamma_compiled", "avqds"):
            source = direct.get((cell_id, route))
            if source is None:
                continue
            route_records[route] = _summarize_source(
                source,
                drive=drive,
                controller=controller,
                reference=reference,
                policy=policy,
                artifact_roots=artifact_roots,
            )

        fresh_locks = [
            record["run_locks"][0]
            for route, record in route_records.items()
            if route not in HISTORICAL_STITCHED_ROUTES
        ]
        comparison_fingerprint = None
        if len(fresh_locks) >= 2:
            comparison_fingerprint = assert_comparable(fresh_locks)

        pairwise: dict[str, Any] = {}
        for left_route, right_route in itertools.combinations(ROUTE_ORDER, 2):
            if left_route not in route_records or right_route not in route_records:
                continue
            pairwise[f"{left_route}__{right_route}"] = _pair_match(
                route_records[left_route]["accuracy"],
                route_records[right_route]["accuracy"],
                policy,
            )
        complete = all(route in route_records for route in ROUTE_ORDER)
        clique = bool(
            complete
            and len(pairwise) == math.comb(len(ROUTE_ORDER), 2)
            and all(pair["passes"] for pair in pairwise.values())
        )
        apm_complete = all(route in route_records for route in APM_ROUTES)
        apm_pair_names = {
            f"{left}__{right}"
            for left, right in itertools.combinations(APM_ROUTES, 2)
        }
        apm_clique = bool(
            apm_complete
            and all(pairwise[name]["passes"] for name in apm_pair_names)
        )
        available_clique = bool(
            len(route_records) >= 2
            and all(pair["passes"] for pair in pairwise.values())
        )
        all_individual = bool(
            complete
            and all(
                route_records[route]["accuracy"]["qualifies_individually"]
                for route in ROUTE_ORDER
            )
        )
        reportable = bool(
            {"old_proxy", "repaired_proxy"}.issubset(route_records)
            and ({"q_compiled", "gamma_compiled"} & set(route_records))
        )
        tier = (
            "individual_target"
            if clique and all_individual
            else "comparison_guard"
            if clique
            else "same_cell_route_best"
            if reportable
            else "unmatched"
        )
        achieved_route = max(
            route_records,
            key=lambda route: float(
                route_records[route]["accuracy"]["accuracy_burden"]
            ),
        )
        cells.append(
            {
                "cell_id": cell_id,
                "drive": drive,
                "controller": controller,
                "apm_cut": float(locked_cell["apm_cut"]),
                "reference": {
                    "manifest_path": str(reference.manifest_path),
                    "manifest_sha256": _sha256(reference.manifest_path),
                    "data_sha256": reference.data_sha256,
                },
                "fresh_comparison_fingerprint": comparison_fingerprint,
                "routes": route_records,
                "missing_routes": [route for route in ROUTE_ORDER if route not in route_records],
                "pairwise_accuracy": pairwise,
                "all_routes_complete": complete,
                "all_routes_clique": clique,
                "available_routes_clique": available_clique,
                "apm_four_route_complete": apm_complete,
                "apm_four_route_clique": apm_clique,
                "four_way_complete": apm_complete,
                "four_way_clique": apm_clique,
                "reportable_current_best": reportable,
                "achieved_accuracy_floor": {
                    "route": achieved_route,
                    "accuracy_burden": float(
                        route_records[achieved_route]["accuracy"]["accuracy_burden"]
                    ),
                },
                "comparison_tier": tier,
            }
        )

    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "analysis_only_not_adopted",
        "route_order": list(ROUTE_ORDER),
        "route_labels": ROUTE_LABELS,
        "route_bundles": ROUTE_BUNDLES,
        "matched_cell_lock": {"path": str(lock_path), "sha256": _sha256(lock_path)},
        "policy": {
            "response_relative_target": float(policy.response_relative_target),
            "response_relative_guard": float(policy.response_relative_guard),
            "matched_accuracy_factor": float(policy.matched_accuracy_factor),
            "matched_component_deadband": float(policy.matched_component_deadband),
            "infidelity_ceiling": float(policy.infidelity_ceiling),
        },
        "estimator_work_definition": {
            "geometry_formula": "g(p) = p*(p+1)/2 + p + 2",
            "integration": "sum_i r_i * g(p_i); exact from RHS telemetry",
            "checkpoint_base": "sum_k g(p_k); exact from pre-edit support",
            "generalized_exchange_candidate_acquisition": (
                "sum_k [p_k*M_k + M_k*(M_k+3)/2] over every persisted "
                "shared positioned-tangent cache; exact for complete new receipts"
            ),
            "avqds_candidate_acquisition": (
                "sum_r c_r*g(p_r+1) over every persisted greedy round; "
                "exact for complete new receipts and a lower bound for legacy ones"
            ),
            "certification": (
                "sum_a g(p_a) over every completed finalist geometry; exact "
                "for complete new receipts and a lower bound for legacy ones"
            ),
            "local_refit": (
                "sum_a r_a*(p_a+1) over instrumented finalist-refit objective "
                "callbacks; reported separately from full geometry"
            ),
            "total": (
                "integration + checkpoint base + candidate acquisition + "
                "certification + local refit; status records exact versus lower bound"
            ),
            "meaning": "logical scalar-geometry entries, not physical shots",
            "physical_shots": (
                "unavailable: these runs use dense statevector evaluation and "
                "persist no estimator circuits, commuting groups, precision, "
                "or shot allocation"
            ),
        },
        "support_motion_definition": {
            "coordinate_identity": "exact runtime-coordinate label multiset",
            "recurrence_identity": (
                "insertion-instance and append-occurrence suffixes removed"
            ),
            "short_cycle_window_support_changes": 2,
        },
        "deletion_gate_definition": {
            "attempt_unit": "one deletion-bearing finalist materialization",
            "post_delete_failure": (
                "a deletion-bearing finalist rejected by certification or the "
                "realized-authority gate after pre-certification resolution"
            ),
            "completeness": (
                "exact only when every checkpoint has one selector invocation and "
                "its persisted attempt list equals declared attempt_count; the "
                "current multi-round loop retains only the final invocation"
            ),
            "avqds": "not applicable because the locked AVQDS route is append-only",
        },
        "summary": {
            "locked_cell_count": len(cells),
            "all_routes_complete_count": sum(cell["all_routes_complete"] for cell in cells),
            "all_routes_clique_count": sum(cell["all_routes_clique"] for cell in cells),
            "apm_four_route_complete_count": sum(
                cell["apm_four_route_complete"] for cell in cells
            ),
            "apm_four_route_clique_count": sum(
                cell["apm_four_route_clique"] for cell in cells
            ),
            "reportable_current_best_count": sum(
                cell["reportable_current_best"] for cell in cells
            ),
            "individual_target_count": sum(
                cell["comparison_tier"] == "individual_target" for cell in cells
            ),
            "comparison_guard_count": sum(
                cell["comparison_tier"] == "comparison_guard" for cell in cells
            ),
            "route_available_counts": {
                route: sum(route in cell["routes"] for cell in cells)
                for route in ROUTE_ORDER
            },
        },
        "cells": cells,
    }


def build_terminal_compile_plan(
    comparison: Mapping[str, Any],
    *,
    output_root: Path,
    include_comparison_guard: bool = False,
    include_complete_target: bool = False,
    include_reportable_current_best: bool = True,
) -> dict[str, Any]:
    """Plan identical fresh terminal compiles for reader-facing cells."""

    allowed_tiers = {"individual_target"}
    if include_comparison_guard:
        allowed_tiers.add("comparison_guard")
    jobs: list[dict[str, Any]] = []
    for cell in comparison.get("cells") or ():
        complete_target = bool(
            cell.get("all_routes_complete")
            and all(
                (cell.get("routes") or {}).get(route, {})
                .get("accuracy", {})
                .get("qualifies_individually")
                is True
                for route in ROUTE_ORDER
            )
        )
        if (
            cell.get("comparison_tier") not in allowed_tiers
            and not (include_complete_target and complete_target)
            and not (
                include_reportable_current_best
                and cell.get("reportable_current_best") is True
            )
        ):
            continue
        for route in ROUTE_ORDER:
            if route not in cell["routes"]:
                continue
            route_record = cell["routes"][route]
            terminal_source = route_record["run_sources"][-1]
            runtime_source = route_record.get("terminal_runtime_source_artifact") or {}
            if runtime_source.get("resolved_path") is None:
                raise FileNotFoundError(
                    "cannot plan terminal compilation without the runtime source "
                    f"artifact for {cell['cell_id']}/{route}: "
                    f"{runtime_source.get('declared_path')}"
                )
            jobs.append(
                {
                    "cell_id": str(cell["cell_id"]),
                    "route": route,
                    "trajectory_json": str(terminal_source["path"]),
                    "trajectory_sha256": str(terminal_source["sha256"]),
                    "source_artifact_json": str(runtime_source["resolved_path"]),
                    "source_artifact_sha256": str(runtime_source["sha256"]),
                    "output_cost_table_json": str(
                        output_root / str(cell["cell_id"]) / f"{route}.json"
                    ),
                    "label": f"{cell['cell_id']}__{route}",
                }
            )
    return {
        "schema": "paper_ii_selector_terminal_compile_plan_v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_sha256": hashlib.sha256(
            json.dumps(
                comparison,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "selection_tiers": sorted(allowed_tiers),
        "include_complete_target": bool(include_complete_target),
        "include_reportable_current_best": bool(include_reportable_current_best),
        "profile": {
            "backend_name": "FakeMarrakesh",
            "seed_transpiler": 7,
            "optimization_level": 1,
            "local_fake_only": True,
            "memory_limit_mb": 8000,
            "concurrency": 1,
        },
        "job_count": len(jobs),
        "jobs": jobs,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--matched-cell-lock",
        type=Path,
        default=Path(
            "pipelines/time_dynamics/campaigns/paper_ii_final_matched_cells_v1.json"
        ),
    )
    parser.add_argument(
        "--direct-root",
        type=Path,
        action="append",
        required=True,
        help="Narrow root containing extracted repaired/Q/Gamma final runs; repeatable.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Root used to resolve repository-relative locked historical sources; "
            "repeatable. The repository root is always searched first."
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--terminal-plan-json",
        type=Path,
        help="Optional fresh terminal-compile plan for matched individual-target cells.",
    )
    parser.add_argument(
        "--terminal-cost-root",
        type=Path,
        help="Output root recorded in the optional terminal-compile plan.",
    )
    parser.add_argument(
        "--include-complete-target",
        action="store_true",
        help=(
            "Also plan fresh terminal compiles for complete four-route cells in "
            "which every route passes the individual target, even when the "
            "strict pairwise factor match fails."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repository_root = args.repository_root.resolve()
    lock_path = _resolve_path(args.matched_cell_lock, repository_root=repository_root)
    direct_roots = [path.expanduser().resolve() for path in args.direct_root]
    artifact_roots = [
        repository_root,
        *(path.expanduser().resolve() for path in args.artifact_root),
    ]
    comparison = build_comparison(
        repository_root=repository_root,
        lock_path=lock_path,
        direct_roots=direct_roots,
        artifact_roots=artifact_roots,
    )
    _write_json(args.output_json.expanduser().resolve(), comparison)
    if args.terminal_plan_json is not None:
        if args.terminal_cost_root is None:
            raise ValueError("--terminal-plan-json requires --terminal-cost-root")
        plan = build_terminal_compile_plan(
            comparison,
            output_root=args.terminal_cost_root.expanduser().resolve(),
            include_complete_target=args.include_complete_target,
        )
        _write_json(args.terminal_plan_json.expanduser().resolve(), plan)
    print(json.dumps(comparison["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
