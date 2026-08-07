#!/usr/bin/env python3
"""Compile exact first-crossing resources for the Paper-I HH target error.

The selected prefix is the earliest completed stored history row whose
same-cutoff absolute energy error is at or below the fixed target.  SNAKE uses
signed active-prefix checkpoints (and never admission-history slicing after an
accepted prune); comparator rows use embedded active-prefix checkpoints and
closed cumulative estimator receipts.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from pipelines.exact_bench.paper_i_s_alg_accounting import (
    PAPER_I_S_ALG_ACCOUNTING_SCHEMA,
    PAPER_I_S_ALG_CONTRACT,
)
from pipelines.reporting.build_paper_i_hh_comparator_tracking_summary import (
    _iter_named_json_array,
    _tar_array_item,
    _tar_json_member,
)
from pipelines.reporting.build_paper_i_hh_tracking_plateau_costs import (
    DEFAULT_TRACKER_JSON,
    LEGACY_OFF_SW_PREFIX_SOURCE,
    REPO_ROOT,
    TABLE_I_QISKIT_COMPILE_CONVENTION,
    COST_ARM_ROUTE_IDS,
    PAPER_I_CLEAN_S_ALG_SNAKE_REPRESENTATIONS,
    _comparator_prefix,
    _cost_arm_terminal_prefix,
    _history_errors,
    _read_source_result,
    _sha256_path,
    _snake_prefix,
    _write_json_atomic,
)


SCHEMA = "paper_i_hh_tracking_target_energy_prefix_costs_v1"
RULE_ID = "first_prefix_at_or_below_fixed_same_cutoff_error_v1"
TARGET_ABS_ERROR = 2.0e-4
DEFAULT_OUTPUT_JSON = DEFAULT_TRACKER_JSON.parent / "target_energy_prefix_costs.json"

COMPARATOR_ROUTE_IDS = {
    "geo_adapt_macro_nph3_7",
    "append_adapt_macro_nph3_7",
    "geo_adapt_projected_singleton_nph3_7",
    "append_adapt_projected_singleton_nph3_7",
}
PUBLIC_TERMINAL_COMPARATOR_ROUTE_IDS = {
    "geo_adapt_macro_nph3_7",
    "append_adapt_projected_singleton_nph3_7",
}


def _clean_s_alg_receipt_closes(payload: Mapping[str, Any]) -> bool:
    receipt = payload.get("S_alg_receipt")
    if not isinstance(receipt, Mapping):
        return False
    components = receipt.get("components")
    if not isinstance(components, Mapping):
        return False
    try:
        component_total = sum(
            int(components[key])
            for key in ("N_H_outer", "N_H_refit", "N_grad", "N_metric")
        )
        receipt_total = int(receipt.get("S_alg"))
        payload_total = int(payload.get("S_alg"))
        accepted_prefix_length = int(
            receipt.get("accepted_prefix_length")
        )
    except (KeyError, TypeError, ValueError):
        return False
    expected_prefix = payload.get(
        "k_target",
        payload.get("k_pl", payload.get("active_depth")),
    )
    return bool(
        receipt.get("schema") == PAPER_I_S_ALG_ACCOUNTING_SCHEMA
        and receipt.get("contract") == PAPER_I_S_ALG_CONTRACT
        and receipt.get("representation") not in {None, ""}
        and component_total == receipt_total == payload_total
        and (
            expected_prefix is None
            or accepted_prefix_length == int(expected_prefix)
        )
    )


def _cost_arm_target_selection(result: Mapping[str, Any]) -> dict[str, Any] | None:
    trajectory_role = str(
        result.get("trajectory_role") or "selected_terminal_path_v1"
    )
    if trajectory_role == "controller_frontier_non_selected_v1":
        trajectory = result.get("selected_winner_history")
        expected_horizon = int(result.get("selected_terminal", {}).get("round") or 0)
    else:
        trajectory = result.get("trajectory")
        expected_horizon = 50
    if not isinstance(trajectory, list) or len(trajectory) != expected_horizon:
        raise ValueError("cost-arm tracker row lacks its validated selected history")
    selected_index: int | None = None
    errors: list[float] = []
    for index, point in enumerate(trajectory, start=1):
        if not isinstance(point, Mapping) or int(point.get("round") or 0) != index:
            raise ValueError("cost-arm compact trajectory round order drift")
        error = float(point.get("error", math.nan))
        if not math.isfinite(error):
            raise ValueError("cost-arm compact trajectory contains nonfinite error")
        errors.append(abs(error))
        if selected_index is None and abs(error) <= TARGET_ABS_ERROR:
            selected_index = index - 1
    if selected_index is None:
        return None
    point = trajectory[selected_index]
    assert isinstance(point, Mapping)
    selected_round = selected_index + 1
    return {
        "history_position": selected_round,
        "k_target": selected_round,
        "k_pl": selected_round,
        "outer_iteration": selected_round,
        "horizon": len(trajectory),
        "trajectory_scope": (
            "selected_terminal_winner_history"
            if trajectory_role == "controller_frontier_non_selected_v1"
            else "selected_terminal_path"
        ),
        "error": errors[selected_index],
        "best_observed_error": min(errors),
        "threshold": TARGET_ABS_ERROR,
    }


def _comparator_target_prefix_streaming(
    *,
    source: Mapping[str, Any],
    result: Mapping[str, Any],
    route_id: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    """Recover a comparator first hit or, for a miss, its terminal prefix."""

    trajectory = result.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("completed comparator row lacks a trajectory")
    selected = next(
        (
            point
            for point in trajectory
            if isinstance(point, Mapping)
            and float(point.get("error", math.inf)) <= TARGET_ABS_ERROR
        ),
        None,
    )
    path = Path(str(source.get("path") or ""))
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_sha = str(source.get("sha256") or "")
    observed_sha = _sha256_path(path)
    if expected_sha and observed_sha != expected_sha:
        raise ValueError(
            f"source SHA-256 drift for {path}: expected={expected_sha}, observed={observed_sha}"
        )
    member_name = str(source.get("member") or "")
    if not member_name:
        raise ValueError("streaming comparator source lacks a result member")
    source_receipt = {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": observed_sha,
        "result_member": member_name,
        "streaming_bounded_memory": True,
    }
    target_reached = selected is not None
    selected = selected if selected is not None else trajectory[-1]
    if not isinstance(selected, Mapping):
        raise ValueError("comparator terminal trajectory point is malformed")
    k_target = int(selected["round"])
    if k_target <= 0:
        raise ValueError(f"invalid comparator target round: {k_target}")
    history_row = _tar_array_item(
        path,
        member_name=member_name,
        array_key="adapt_history",
        zero_index=k_target - 1,
    )
    receipt_row = _tar_array_item(
        path,
        member_name=member_name,
        array_key="estimator_call_round_receipts",
        zero_index=k_target - 1,
    )
    seed_member = member_name.rsplit("/", 1)[0] + "/runtime_seed.json"
    runtime_seed = _tar_json_member(path, member_name=seed_member)
    source_receipt["runtime_seed_member"] = seed_member

    source_error = history_row.get("abs_delta_e_same_cutoff_after")
    if source_error is None:
        source_error = history_row.get("abs_delta_e_after")
    if source_error is None or not math.isclose(
        float(source_error),
        float(selected["error"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("streamed comparator target row disagrees with tracker trajectory")
    minimal_payload = {
        "status": "completed",
        "result": {
            "adapt_history": [{} for _ in range(k_target - 1)] + [history_row],
            "estimator_call_round_receipts": [{} for _ in range(k_target - 1)]
            + [receipt_row],
        },
    }
    selection = {
        "history_position": k_target,
        "k_target": k_target,
        "k_pl": k_target,
        "outer_iteration": k_target,
        "horizon": len(trajectory),
        "error": float(selected["error"]),
        "best_observed_error": min(float(point["error"]) for point in trajectory),
        "threshold": TARGET_ABS_ERROR,
    }
    prefix = _comparator_prefix(
        minimal_payload,
        runtime_seed=runtime_seed,
        selection=selection,
        representation=(
            "intact_macro"
            if "macro" in str(route_id).lower()
            else "projected_singleton"
        ),
        source_kind="paper_i_hh_comparator_exact_active_target_prefix",
    )
    resolved = {**selection, **prefix}
    return (
        resolved if target_reached else None,
        None if target_reached else resolved,
        source_receipt,
    )


def select_target_prefix(
    payload: Mapping[str, Any],
    *,
    method: str,
    target_abs_error: float = TARGET_ABS_ERROR,
) -> dict[str, Any] | None:
    target = float(target_abs_error)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError(f"target_abs_error must be finite and positive: {target!r}")
    history, errors = _history_errors(payload, method=method)
    zero_index = next((index for index, error in enumerate(errors) if error <= target), None)
    if zero_index is None:
        return None
    row = history[zero_index]
    history_position = zero_index + 1
    outer_iteration = row.get("outer_iteration")
    if outer_iteration is None:
        iteration = row.get("iteration")
        outer_iteration = history_position if iteration is None else int(iteration) + 1
    return {
        "history_position": history_position,
        "k_target": history_position,
        # Keep k_pl for the shared exact-prefix compilers.  The public receipt
        # remains unambiguous through k_target and the target rule below.
        "k_pl": history_position,
        "outer_iteration": int(outer_iteration),
        "horizon": len(history),
        "error": float(errors[zero_index]),
        "best_observed_error": float(min(errors)),
        "threshold": target,
    }


def _cache(output_json: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    if not output_json.is_file():
        return {}, {}
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    rule = payload.get("rule")
    if (
        payload.get("schema") != SCHEMA
        or not isinstance(rule, Mapping)
        or rule.get("id") != RULE_ID
        or not math.isclose(float(rule.get("target_abs_error", math.nan)), TARGET_ABS_ERROR)
    ):
        return {}, {}
    rows = {
        (str(row.get("route_id")), str(row.get("regime"))): dict(row)
        for row in payload.get("rows", [])
        if isinstance(row, Mapping)
    }
    unresolved = {
        (str(row.get("route_id")), str(row.get("regime"))): dict(row)
        for row in payload.get("unresolved", [])
        if isinstance(row, Mapping)
    }
    return rows, unresolved


def _payload(
    *,
    tracker_path: Path,
    tracker_schema: Any,
    rows: list[dict[str, Any]],
    unresolved: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tracker": {
            "path": str(tracker_path.relative_to(REPO_ROOT)),
            "sha256": _sha256_path(tracker_path),
            "schema": tracker_schema,
        },
        "rule": {
            "id": RULE_ID,
            "target_abs_error": TARGET_ABS_ERROR,
            "definition": (
                "earliest completed stored history prefix with same-cutoff "
                "absolute energy error <= 2e-4; no interpolation"
            ),
            "physical_definition": "E_T = 1e-4 * L * E0 with L=2 and E0=1",
            "reporting_only": True,
        },
        "compile_policy": {
            "identity": TABLE_I_QISKIT_COMPILE_CONVENTION,
            "basis_gate_family": "Paper-I backend-free Table-I",
            "optimization_level": 0,
            "seed_transpiler": 7,
            "reference_state_included": True,
            "snake_synthesis": "historical structural Pauli-label-group convention",
            "comparator_synthesis": "execution-aware coefficient-bearing convention",
        },
        "rows": sorted(rows, key=lambda row: (row["route_id"], row["regime"])),
        "unresolved": sorted(
            unresolved,
            key=lambda row: (row["route_id"], row["regime"]),
        ),
        "summary": {
            "status": status,
            "complete_prefix_count": len(rows),
            "unresolved_count": len(unresolved),
            "threshold_not_reached_count": sum(
                row.get("status") == "threshold_not_reached" for row in unresolved
            ),
        },
    }


def build_target_costs(*, tracker_json: Path, output_json: Path) -> dict[str, Any]:
    tracker_path = tracker_json.resolve()
    tracker = json.loads(tracker_path.read_text(encoding="utf-8"))
    routes = tracker.get("routes")
    if not isinstance(routes, list):
        raise TypeError("tracker JSON has no routes list")
    cached_rows, cached_unresolved = _cache(output_json)
    rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for route in routes:
        if not isinstance(route, Mapping):
            continue
        route_id = str(route.get("id"))
        results = route.get("results")
        if not isinstance(results, Mapping):
            raise TypeError(f"route {route_id} has no results mapping")
        for regime, result in results.items():
            key = (route_id, str(regime))
            if not isinstance(result, Mapping) or not result.get("trajectory"):
                unresolved.append(
                    {
                        "route_id": route_id,
                        "regime": str(regime),
                        "status": str(result.get("status") if isinstance(result, Mapping) else "missing"),
                        "reason": "no completed validated trajectory in tracker",
                    }
                )
                continue
            source = result.get("source")
            if not isinstance(source, Mapping):
                raise ValueError(f"completed row {route_id}/{regime} has no source")
            declared_path = str(source.get("path") or "")
            declared_sha = str(source.get("sha256") or "")
            declared_trajectory_sha = source.get("trajectory_receipt_sha256")
            cached = cached_rows.get(key) or cached_unresolved.get(key)
            cached_source = cached.get("source") if isinstance(cached, Mapping) else None
            cached_rule = cached.get("rule") if isinstance(cached, Mapping) else None
            cached_is_current = True
            if (
                route_id in PAPER_I_CLEAN_S_ALG_SNAKE_REPRESENTATIONS
                and isinstance(cached, Mapping)
            ):
                prefix_receipt = cached.get("prefix_receipt")
                if (
                    isinstance(prefix_receipt, Mapping)
                    and prefix_receipt.get("mode") == "signed_checkpoint"
                ):
                    cached_is_current = (
                        cached.get("S_alg_reconstruction_status")
                        == "clean_algorithm_recount_closed_signed_prefix"
                    )
            clean_receipt_required = bool(
                route_id in PAPER_I_CLEAN_S_ALG_SNAKE_REPRESENTATIONS
                or route_id.startswith("append_adapt_")
            )
            if (
                route_id in PUBLIC_TERMINAL_COMPARATOR_ROUTE_IDS
                and isinstance(cached, Mapping)
            ):
                if cached.get("status") == "threshold_not_reached":
                    cached_is_current = isinstance(cached.get("terminal"), Mapping)
            if (
                isinstance(cached, Mapping)
                and cached_is_current
                and isinstance(cached_source, Mapping)
                and isinstance(cached_rule, Mapping)
                and cached_rule.get("id") == RULE_ID
                and math.isclose(float(cached_rule.get("target_abs_error", math.nan)), TARGET_ABS_ERROR)
                and str(cached_source.get("path")) == declared_path
                and str(cached_source.get("sha256")) == declared_sha
                and cached_source.get("trajectory_receipt_sha256")
                == declared_trajectory_sha
                and (
                    not clean_receipt_required
                    or (
                        (
                            cached.get("status") == "complete"
                            and _clean_s_alg_receipt_closes(cached)
                        )
                        or (
                            cached.get("status") == "threshold_not_reached"
                            and isinstance(cached.get("terminal"), Mapping)
                            and _clean_s_alg_receipt_closes(
                                cached.get("terminal", {})
                            )
                        )
                    )
                )
            ):
                destination = rows if cached.get("status") == "complete" else unresolved
                destination.append(dict(cached))
                print(f"reuse {route_id}/{regime} status={cached.get('status')}", flush=True)
                continue

            method = "comparator" if route_id in COMPARATOR_ROUTE_IDS else "snake"
            rule = {"id": RULE_ID, "target_abs_error": TARGET_ABS_ERROR, "reporting_only": True}
            if method == "comparator":
                streamed, terminal, source_receipt = _comparator_target_prefix_streaming(
                    source=source,
                    result=result,
                    route_id=route_id,
                )
                if streamed is None:
                    trajectory = result.get("trajectory") or []
                    errors = [
                        float(point["error"])
                        for point in trajectory
                        if isinstance(point, Mapping)
                    ]
                    unresolved.append(
                        {
                            "route_id": route_id,
                            "regime": str(regime),
                            "status": "threshold_not_reached",
                            "reason": "no completed stored prefix reached the fixed target",
                            "best_observed_error": float(min(errors)),
                            "horizon": len(errors),
                            "terminal": terminal,
                            "rule": rule,
                            "source": source_receipt,
                        }
                    )
                    print(f"unreached {route_id}/{regime} best={min(errors):.8e}", flush=True)
                else:
                    rows.append(
                        {
                            "route_id": route_id,
                            "regime": str(regime),
                            "status": "complete",
                            "rule": rule,
                            **streamed,
                            "source": source_receipt,
                            "prefix_source": source_receipt,
                        }
                    )
                    print(
                        f"compile {route_id}/{regime} k={streamed['k_target']} "
                        f"of {streamed['horizon']} (bounded-memory)",
                        flush=True,
                    )
                _write_json_atomic(
                    output_json,
                    _payload(
                        tracker_path=tracker_path,
                        tracker_schema=tracker.get("schema"),
                        rows=rows,
                        unresolved=unresolved,
                        status="in_progress",
                    ),
                )
                gc.collect()
                continue

            if route_id in COST_ARM_ROUTE_IDS:
                trajectory = (
                    result.get("selected_winner_history")
                    if result.get("trajectory_role")
                    == "controller_frontier_non_selected_v1"
                    else result.get("trajectory")
                )
                assert isinstance(trajectory, list) and trajectory
                terminal_point = trajectory[-1]
                if not isinstance(terminal_point, Mapping):
                    raise ValueError("cost-arm terminal trajectory row is malformed")
                terminal_selection = {
                    "history_position": len(trajectory),
                    "k_target": len(trajectory),
                    "k_pl": len(trajectory),
                    "outer_iteration": int(terminal_point.get("round") or 0),
                    "horizon": len(trajectory),
                    "error": float(terminal_point.get("error", math.nan)),
                    "best_observed_error": min(
                        float(point["error"])
                        for point in trajectory
                        if isinstance(point, Mapping)
                    ),
                    "threshold": TARGET_ABS_ERROR,
                }
                terminal_prefix, source_receipt = _cost_arm_terminal_prefix(
                    source=source,
                    result=result,
                    selection=terminal_selection,
                )
                selection = _cost_arm_target_selection(result)
                if selection is None:
                    unresolved.append(
                        {
                            "route_id": route_id,
                            "regime": str(regime),
                            "status": "threshold_not_reached",
                            "reason": "no completed stored prefix reached the fixed target",
                            "best_observed_error": terminal_selection[
                                "best_observed_error"
                            ],
                            "horizon": len(trajectory),
                            "rule": rule,
                            "source": source_receipt,
                        }
                    )
                    print(
                        f"unreached {route_id}/{regime} "
                        f"best={terminal_selection['best_observed_error']:.8e}",
                        flush=True,
                    )
                elif int(selection["outer_iteration"]) != int(
                    terminal_selection["outer_iteration"]
                ):
                    unresolved.append(
                        {
                            "route_id": route_id,
                            "regime": str(regime),
                            "status": "exact_prefix_unavailable",
                            "reason": (
                                "target crossing predates the validated terminal "
                                "checkpoint; raw beam archive remains closed"
                            ),
                            "rule": rule,
                            **selection,
                            "source": source_receipt,
                        }
                    )
                    print(
                        f"defer {route_id}/{regime} k={selection['k_target']}: "
                        "only the terminal checkpoint is executable",
                        flush=True,
                    )
                else:
                    rows.append(
                        {
                            "route_id": route_id,
                            "regime": str(regime),
                            "status": "complete",
                            "rule": rule,
                            **selection,
                            **terminal_prefix,
                            "source": source_receipt,
                            "prefix_source": source_receipt,
                        }
                    )
                    print(
                        f"compile {route_id}/{regime} k={selection['k_target']} "
                        "from validated terminal checkpoint",
                        flush=True,
                    )
                _write_json_atomic(
                    output_json,
                    _payload(
                        tracker_path=tracker_path,
                        tracker_schema=tracker.get("schema"),
                        rows=rows,
                        unresolved=unresolved,
                        status="in_progress",
                    ),
                )
                gc.collect()
                continue

            payload, runtime_seed, source_receipt = _read_source_result(
                source,
                need_runtime_seed=False,
            )
            selection = select_target_prefix(payload, method=method)
            if selection is None:
                _history, errors = _history_errors(payload, method=method)
                unresolved.append(
                    {
                        "route_id": route_id,
                        "regime": str(regime),
                        "status": "threshold_not_reached",
                        "reason": "no completed stored prefix reached the fixed target",
                        "best_observed_error": float(min(errors)),
                        "horizon": len(errors),
                        "rule": rule,
                        "source": source_receipt,
                    }
                )
                print(f"unreached {route_id}/{regime} best={min(errors):.8e}", flush=True)
            else:
                print(
                    f"compile {route_id}/{regime} k={selection['k_target']} "
                    f"of {selection['horizon']}",
                    flush=True,
                )
                prefix_source_receipt = source_receipt
                if method == "snake":
                    prefix_payload = payload
                    if (
                        route_id == "legacy_no_ordinary_novelty_nph2_4"
                        and str(regime) == "strong_weak_u8"
                        and int(selection["k_target"]) <= 30
                    ):
                        if not LEGACY_OFF_SW_PREFIX_SOURCE.is_file():
                            raise FileNotFoundError(LEGACY_OFF_SW_PREFIX_SOURCE)
                        prefix_payload, _seed, prefix_source_receipt = _read_source_result(
                            {
                                "path": str(LEGACY_OFF_SW_PREFIX_SOURCE.relative_to(REPO_ROOT)),
                                "sha256": _sha256_path(LEGACY_OFF_SW_PREFIX_SOURCE),
                            },
                            need_runtime_seed=False,
                        )
                        prefix_errors = _history_errors(prefix_payload, method="snake")[1]
                        selected_error = prefix_errors[int(selection["k_target"]) - 1]
                        if not math.isclose(
                            selected_error,
                            float(selection["error"]),
                            rel_tol=0.0,
                            abs_tol=1.0e-12,
                        ):
                            raise ValueError(
                                "legacy strong-weak authenticated prefix disagrees "
                                "with the displayed continuation trajectory"
                            )
                    prefix = _snake_prefix(
                        prefix_payload,
                        selection=selection,
                        source=prefix_source_receipt,
                        route_id=route_id,
                        fallback_source_kind="paper_i_hh_nonpruned_history_target_prefix",
                    )
                else:
                    if runtime_seed is None:
                        raise RuntimeError(f"comparator runtime seed missing for {route_id}/{regime}")
                    prefix = _comparator_prefix(
                        payload,
                        runtime_seed=runtime_seed,
                        selection=selection,
                        representation=(
                            "intact_macro"
                            if "macro" in str(route_id).lower()
                            else "projected_singleton"
                        ),
                        source_kind="paper_i_hh_comparator_exact_active_target_prefix",
                    )
                rows.append(
                    {
                        "route_id": route_id,
                        "regime": str(regime),
                        "status": "complete",
                        "rule": rule,
                        **selection,
                        **prefix,
                        "source": source_receipt,
                        "prefix_source": prefix_source_receipt,
                    }
                )
                del prefix
                if method == "snake":
                    del prefix_payload
            _write_json_atomic(
                output_json,
                _payload(
                    tracker_path=tracker_path,
                    tracker_schema=tracker.get("schema"),
                    rows=rows,
                    unresolved=unresolved,
                    status="in_progress",
                ),
            )
            del payload, runtime_seed
            gc.collect()

    final = _payload(
        tracker_path=tracker_path,
        tracker_schema=tracker.get("schema"),
        rows=rows,
        unresolved=unresolved,
        status="complete",
    )
    _write_json_atomic(output_json, final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker-json", type=Path, default=DEFAULT_TRACKER_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()
    payload = build_target_costs(
        tracker_json=args.tracker_json,
        output_json=args.output_json,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
