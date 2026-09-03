#!/usr/bin/env python3
"""Build a provenance-bound four-route Paper-II selector comparison.

The comparison is deliberately downstream of trajectory generation.  It
re-evaluates every available final t=20 trajectory on the locked exact grid,
requires a same-cell four-way accuracy clique, and reports terminal circuit,
integration-estimator, and support-churn burdens as separate coordinates.

The four route identities are the historical unit-cost proxy, the repaired
analytical proxy, the realized-Q compiled selector, and the ideal-Gamma
compiled selector.  ``S`` is a ranking score shared by the compiled routes;
it is not a fifth trajectory route.
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


SCHEMA = "paper_ii_four_route_comparison_v1"
ROUTE_ORDER = (
    "q_compiled",
    "gamma_compiled",
    "repaired_proxy",
    "old_proxy",
)
ROUTE_LABELS = {
    "q_compiled": "Q-compiled",
    "gamma_compiled": "Gamma-compiled (support)",
    "repaired_proxy": "repaired proxy",
    "old_proxy": "old proxy (historical stitched)",
}
ROUTE_BUNDLES = {
    "q_compiled": "paper_ii_exchange_q_compiled_v1",
    "gamma_compiled": "paper_ii_exchange_gamma_compiled_v1",
    "repaired_proxy": "paper_ii_exchange_legacy_proxy_fixed_v1",
    "old_proxy": "paper_ii_generalized_exchange_v2",
}
ACTION_SUFFIXES = {
    "q_compiled": "__exchange_q_compiled",
    "gamma_compiled": "__exchange_gamma_compiled",
    "repaired_proxy": "__exchange_legacy_proxy_fixed",
}

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
    """Find the three direct t=20 route products below explicit narrow roots."""

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
    if completion.get("status") != "SUCCESS":
        raise ValueError(f"direct run is not successful: {run_path}")
    expected_bundle = ROUTE_BUNDLES[source.route]
    if completion.get("exchange_route_bundle") != expected_bundle:
        raise ValueError(f"route-bundle mismatch for {run_path}")
    if completion.get("trajectory_sha256") != _sha256(run_path):
        raise ValueError(f"trajectory digest mismatch for {run_path}")
    if completion.get("terminal_cost_sha256") != _sha256(cost_path):
        raise ValueError(f"worker terminal-cost digest mismatch for {run_path}")
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


def structural_churn(
    runs: Sequence[Mapping[str, Any]],
    *,
    short_cycle_window: int = 2,
) -> dict[str, Any]:
    """Measure exact support motion and short generator recurrence."""

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

    for run in runs:
        points = ((run.get("trajectory") or {}).get("points") or ())
        for point in points:
            decision = point.get("patch_decision") or {}
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
    gross = added_count + deleted_count
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
        "gross_churn": gross,
        "excess_churn": gross - abs(final_count - initial_count),
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
    }


def integration_geometry_work(runs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Return a logical scalar-estimator proxy for integration RHS work.

    One RHS geometry at support size p is charged p(p+1)/2 symmetric metric
    entries, p force entries, and two scalar observables.  This is not a
    hardware shot count and excludes checkpoint candidate acquisition and
    certification work that the current artifacts do not ledger completely.
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


def _accuracy_record(
    metrics: AccuracyMetrics,
    *,
    drive: str,
    controller: str,
    policy: Any,
) -> dict[str, Any]:
    key = CellKey(
        regime="hh_snake_nph1",
        drive=drive,
        method="exchange",
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
        "time_step_count": int(steps),
        "n_theta": int((runs[-1].get("summary") or {})["runtime_parameter_count_final"]),
        "accuracy": _accuracy_record(
            metrics,
            drive=drive,
            controller=controller,
            policy=policy,
        ),
        "estimator_work": integration_geometry_work(runs),
        "structural_churn": structural_churn(runs),
        "run_locks": [run.get("run_lock") for run in runs],
    }
    if source.route != "old_proxy":
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
        old_sources = locked_cell["source_artifacts"]["apm"]
        old = RunSource(
            route="old_proxy",
            cell_id=cell_id,
            run_paths=(
                _resolve_existing_path(
                    str(old_sources["screen_run_json"]["path"]),
                    roots=artifact_roots,
                ),
                _resolve_existing_path(
                    str(old_sources["continuation_run_json"]["path"]),
                    roots=artifact_roots,
                ),
            ),
        )
        route_records: dict[str, Any] = {
            "old_proxy": _summarize_source(
                old,
                drive=drive,
                controller=controller,
                reference=reference,
                policy=policy,
                expected_hashes=(
                    str(old_sources["screen_run_json"]["sha256"]),
                    str(old_sources["continuation_run_json"]["sha256"]),
                ),
            )
        }
        for route in ("repaired_proxy", "q_compiled", "gamma_compiled"):
            source = direct.get((cell_id, route))
            if source is None:
                continue
            route_records[route] = _summarize_source(
                source,
                drive=drive,
                controller=controller,
                reference=reference,
                policy=policy,
            )

        fresh_locks = [
            record["run_locks"][0]
            for route, record in route_records.items()
            if route != "old_proxy"
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
            and len(pairwise) == 6
            and all(pair["passes"] for pair in pairwise.values())
        )
        all_individual = bool(
            complete
            and all(
                route_records[route]["accuracy"]["qualifies_individually"]
                for route in ROUTE_ORDER
            )
        )
        tier = (
            "individual_target"
            if clique and all_individual
            else "comparison_guard"
            if clique
            else "unmatched"
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
                "four_way_complete": complete,
                "four_way_clique": clique,
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
            "name": "logical_integration_geometry_work",
            "formula": "sum_i r_i * (p_i*(p_i+1)/2 + p_i + 2)",
            "meaning": "logical scalar-estimator proxy; not physical shots",
            "excludes": [
                "checkpoint base geometry",
                "candidate tangent acquisition",
                "candidate certification",
                "shot allocation and commuting-group structure",
            ],
        },
        "structural_churn_definition": {
            "coordinate_identity": "exact runtime-coordinate label multiset",
            "recurrence_identity": (
                "insertion-instance and append-occurrence suffixes removed"
            ),
            "short_cycle_window_support_changes": 2,
        },
        "summary": {
            "locked_cell_count": len(cells),
            "four_way_complete_count": sum(cell["four_way_complete"] for cell in cells),
            "four_way_clique_count": sum(cell["four_way_clique"] for cell in cells),
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
) -> dict[str, Any]:
    """Plan identical fresh terminal compiles for matched four-route cells."""

    allowed_tiers = {"individual_target"}
    if include_comparison_guard:
        allowed_tiers.add("comparison_guard")
    jobs: list[dict[str, Any]] = []
    for cell in comparison.get("cells") or ():
        complete_target = bool(
            cell.get("four_way_complete")
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
        ):
            continue
        for route in ROUTE_ORDER:
            route_record = cell["routes"][route]
            terminal_source = route_record["run_sources"][-1]
            jobs.append(
                {
                    "cell_id": str(cell["cell_id"]),
                    "route": route,
                    "trajectory_json": str(terminal_source["path"]),
                    "trajectory_sha256": str(terminal_source["sha256"]),
                    "output_cost_table_json": str(
                        output_root / str(cell["cell_id"]) / f"{route}.json"
                    ),
                    "label": f"{cell['cell_id']}__{route}",
                }
            )
    return {
        "schema": "paper_ii_four_route_terminal_compile_plan_v1",
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
