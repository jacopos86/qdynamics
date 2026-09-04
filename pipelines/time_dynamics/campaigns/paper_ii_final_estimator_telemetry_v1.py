"""Prepare the final Paper-II exact-estimator telemetry reruns.

This campaign preserves the fifteen locked matched-accuracy cells and the
three active APM selector routes from
``paper_ii_final_selector_comparison_v1``.  It additionally reruns each
cell's locked AVQDS comparator because exact candidate-acquisition work
cannot be reconstructed from the historical trajectory artifacts.

The resulting matrix is exactly ``15 cells x (3 APM + 1 AVQDS)`` fresh
``t=20`` trajectories.  It contains no screen, continuation, or resume
actions.  The module prepares actions; it does not submit them.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics import paper_ii_runs as runs
from pipelines.time_dynamics.campaigns import (
    paper_ii_final_selector_comparison_v1 as selector,
)
from pipelines.time_dynamics.run_lock import file_sha256, resolve_code_revision


CAMPAIGN_ID = "paper_ii_final_estimator_telemetry_v1"
ACTIONS_SCHEMA = "paper_ii_final_estimator_telemetry_actions_v1"
SUMMARY_SCHEMA = "paper_ii_final_estimator_telemetry_prepare_v1"
TELEMETRY_CONTRACT = "paper_ii_exact_logical_estimator_telemetry_20260903"
COMPARATOR_MODE = "fresh_exact_telemetry"
EXPECTED_APM_ACTION_COUNT = 45
EXPECTED_AVQDS_ACTION_COUNT = 15
EXPECTED_ACTION_COUNT = EXPECTED_APM_ACTION_COUNT + EXPECTED_AVQDS_ACTION_COUNT
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _replace_flag_value(argv: Sequence[str], flag: str, value: str) -> tuple[str, ...]:
    values = list(argv)
    if values.count(flag) != 1:
        raise selector.MatchedCellLockError(
            f"expected exactly one {flag} in runner argv; found {values.count(flag)}"
        )
    position = values.index(flag) + 1
    if position >= len(values):
        raise selector.MatchedCellLockError(f"{flag} has no value")
    values[position] = value
    return tuple(values)


def _avqds_action_id(cell: selector.MatchedCell, comparator_arm: str) -> str:
    return f"{cell.cell_id}__{comparator_arm}"


def _trajectory_path(output_root: str | Path, action_id: str) -> Path:
    return Path(output_root) / CAMPAIGN_ID / action_id / "run.json"


def _assert_common_fresh_run(
    argv: Sequence[str],
    *,
    trajectory: Path,
    lock: selector.MatchedCellLock,
    code_revision: str,
) -> Any:
    from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
        _build_parser,
        _resolve_times,
    )

    values = tuple(argv)
    parsed = _build_parser().parse_args(list(values))
    if "--resume-from-run-json" in values or parsed.resume_from_run_json is not None:
        raise selector.MatchedCellLockError("exact telemetry reruns must be fresh")
    if str(parsed.output_json) != str(trajectory):
        raise selector.MatchedCellLockError("trajectory path drifted")
    if str(parsed.artifact_json) != str(lock.seed["artifact_json"]):
        raise selector.MatchedCellLockError("seed artifact drifted")
    if str(parsed.code_revision) != code_revision:
        raise selector.MatchedCellLockError("code revision drifted")
    if not bool(parsed.record_statevector):
        raise selector.MatchedCellLockError("--record-statevector is required")

    horizon = runs.HORIZONS[str(lock.registry_bindings["horizon"])]
    times = _resolve_times(parsed)
    if (
        parsed.times is not None
        or parsed.t_initial is not None
        or not math.isclose(float(parsed.t_final), float(horizon.t_final), rel_tol=0.0)
        or int(parsed.num_times) != int(horizon.num_times)
        or len(times) != int(horizon.num_times)
        or not math.isclose(float(times[0]), 0.0, rel_tol=0.0, abs_tol=0.0)
        or not math.isclose(
            float(times[-1]), float(horizon.t_final), rel_tol=0.0, abs_tol=0.0
        )
    ):
        raise selector.MatchedCellLockError("time grid is not the locked t=0..20 grid")
    return parsed


def _rehome_apm_action(
    row: Mapping[str, Any],
    *,
    output_root: str | Path,
    lock: selector.MatchedCellLock,
) -> dict[str, Any]:
    action = dict(row)
    action_id = str(action["action_id"])
    trajectory = _trajectory_path(output_root, action_id)
    argv = _replace_flag_value(
        action["runner_argv"], "--output-json", str(trajectory)
    )
    _assert_common_fresh_run(
        argv,
        trajectory=trajectory,
        lock=lock,
        code_revision=str(action["code_revision"]),
    )
    comparator = dict(action["comparator"])
    comparator["reuse"] = COMPARATOR_MODE
    comparator["action"] = _avqds_action_id(
        next(cell for cell in lock.cells if cell.cell_id == action["cell_id"]),
        str(lock.registry_bindings["comparator_arm"]),
    )
    action.update(
        {
            "schema": ACTIONS_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "runner_argv": list(argv),
            "trajectory_json": str(trajectory),
            "comparator": comparator,
            "estimator_telemetry_contract": TELEMETRY_CONTRACT,
            "precursor": False,
        }
    )
    return action


def _build_avqds_action(
    cell: selector.MatchedCell,
    *,
    lock: selector.MatchedCellLock,
    output_root: str | Path,
    source_package_sha256: str | None,
    code_revision: str,
) -> dict[str, Any]:
    bindings = lock.registry_bindings
    arm_id = str(bindings["comparator_arm"])
    action_id = _avqds_action_id(cell, arm_id)
    trajectory = _trajectory_path(output_root, action_id)
    command = runs.build_run(
        regime=str(bindings["regime"]),
        arm=arm_id,
        gate=str(bindings["gate"]),
        drive=cell.drive,
        horizon=str(bindings["horizon"]),
        numerics=str(bindings["numerics"]),
        step_control=cell.controller,
        search_profile=str(bindings["search_profile"]),
        activation_cut=cell.avqds_cut,
        output_json=str(trajectory),
        extra_flags=("--record-statevector",),
        require_seed=True,
    )
    argv = (*command.argv(), "--code-revision", code_revision)
    parsed = _assert_common_fresh_run(
        argv,
        trajectory=trajectory,
        lock=lock,
        code_revision=code_revision,
    )
    if str(parsed.dynamics_policy) != "avqds":
        raise selector.MatchedCellLockError(f"{action_id}: comparator is not AVQDS")
    if not math.isclose(
        float(parsed.avqds_l2_cut), float(cell.avqds_cut), rel_tol=1.0e-12, abs_tol=0.0
    ):
        raise selector.MatchedCellLockError(f"{action_id}: AVQDS cut drifted")
    forbidden = (
        "--exchange-route-bundle",
        "--insertion-gate-mode",
        "--compile-workers",
    )
    if any(flag in argv for flag in forbidden):
        raise selector.MatchedCellLockError(
            f"{action_id}: AVQDS carries an APM-only execution or policy flag"
        )
    paired_actions = [
        f"{cell.cell_id}__{arm_id}"
        for arm_id in selector.bind_selector_routes(lock.selector_routes).values()
    ]
    return {
        "schema": ACTIONS_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "action_id": action_id,
        "cell_id": cell.cell_id,
        "drive": cell.drive,
        "controller": cell.controller,
        "method": "avqds",
        "arm": arm_id,
        "exchange_route_bundle": None,
        "exchange_route_contract": None,
        "exchange_route_contract_sha256": None,
        "avqds_cut": float(cell.avqds_cut),
        "avqds_cut_label": cell.avqds_cut_label,
        "apm_cut": float(cell.apm_cut),
        "apm_cut_label": cell.apm_cut_label,
        "adoption_status": cell.adoption_status,
        "paired_apm_actions": paired_actions,
        "runner_module": runs.RUNNER_MODULE,
        "runner_argv": list(argv),
        "trajectory_json": str(trajectory),
        "run_id": command.run_id,
        "code_revision": code_revision,
        "matched_cell_lock_sha256": str(lock.sha256),
        "seed_sha256": str(lock.seed["sha256"]),
        "source_package_sha256": source_package_sha256,
        "estimator_telemetry_contract": TELEMETRY_CONTRACT,
        "precursor": False,
    }


def build_actions(
    lock: selector.MatchedCellLock,
    *,
    output_root: str | Path,
    source_package_sha256: str | None = None,
    compile_workers: int = 8,
) -> tuple[dict[str, Any], ...]:
    """Build 45 APM and 15 AVQDS exact-telemetry rerun actions."""

    if int(compile_workers) < 1:
        raise selector.MatchedCellLockError("compile_workers must be positive")
    if source_package_sha256 is not None:
        source_package_sha256 = selector._bound_sha256(
            source_package_sha256,
            where="source_package_sha256",
        )
    code_revision = resolve_code_revision(_REPO_ROOT)
    if code_revision in ("", "unknown"):
        raise selector.MatchedCellLockError("could not resolve a code revision")

    base_apm = selector.build_actions(
        lock,
        output_root=output_root,
        source_package_sha256=source_package_sha256,
        execution_flags=("--compile-workers", str(int(compile_workers))),
    )
    rows = [
        _rehome_apm_action(row, output_root=output_root, lock=lock)
        for row in base_apm
    ]
    rows.extend(
        _build_avqds_action(
            cell,
            lock=lock,
            output_root=output_root,
            source_package_sha256=source_package_sha256,
            code_revision=str(code_revision),
        )
        for cell in lock.cells
    )

    apm = [row for row in rows if row["method"] == "apm"]
    avqds = [row for row in rows if row["method"] == "avqds"]
    if len(apm) != EXPECTED_APM_ACTION_COUNT or len(avqds) != EXPECTED_AVQDS_ACTION_COUNT:
        raise selector.MatchedCellLockError(
            f"expected {EXPECTED_APM_ACTION_COUNT} APM + "
            f"{EXPECTED_AVQDS_ACTION_COUNT} AVQDS actions; got "
            f"{len(apm)} + {len(avqds)}"
        )
    identifiers = [str(row["action_id"]) for row in rows]
    run_ids = [str(row["run_id"]) for row in rows]
    if len(set(identifiers)) != EXPECTED_ACTION_COUNT:
        raise selector.MatchedCellLockError("action ids are not unique")
    if len(set(run_ids)) != EXPECTED_ACTION_COUNT:
        raise selector.MatchedCellLockError("run ids are not unique")
    if any(bool(row.get("precursor")) for row in rows):
        raise selector.MatchedCellLockError("precursor action entered final telemetry matrix")
    return tuple(rows)


def prepare(
    *,
    output_root: str | Path,
    actions_jsonl: str | Path,
    lock_json: str | Path = selector.LOCK_JSON,
    source_package_sha256: str | None = None,
    compile_workers: int = 8,
) -> dict[str, Any]:
    lock = selector.load_lock(lock_json)
    seed_sha256 = selector._verify_seed(lock)
    rows = build_actions(
        lock,
        output_root=output_root,
        source_package_sha256=source_package_sha256,
        compile_workers=compile_workers,
    )
    written = selector.write_actions_jsonl(rows, actions_jsonl)
    return {
        "schema": SUMMARY_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "status": selector.LOCK_STATUS,
        "telemetry_contract": TELEMETRY_CONTRACT,
        "lock": {
            "path": str(lock.path),
            "sha256": lock.sha256,
            "schema": selector.LOCK_SCHEMA,
            "source_campaign_id": selector.CAMPAIGN_ID,
        },
        "code_revision": str(rows[0]["code_revision"]),
        "seed": {"path": str(lock.seed["artifact_json"]), "sha256": seed_sha256},
        "registry_bindings": dict(lock.registry_bindings),
        "selector_routes": selector.bind_selector_routes(lock.selector_routes),
        "cell_count": len(lock.cells),
        "excluded_cell_count": len(lock.exclusions),
        "route_count": len(lock.selector_routes),
        "action_count": len(rows),
        "apm_action_count": sum(row["method"] == "apm" for row in rows),
        "avqds_action_count": sum(row["method"] == "avqds" for row in rows),
        "precursor_action_count": sum(bool(row.get("precursor")) for row in rows),
        "comparator_mode": COMPARATOR_MODE,
        "actions_jsonl": {"path": str(written), "sha256": file_sha256(written)},
        "compile_workers": int(compile_workers),
        "output_root": str(Path(output_root)),
        "source_package_sha256": source_package_sha256,
        "blocks": list(lock.payload["blocks"]["active"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path, default=Path("raw_outputs"))
    parser.add_argument("--lock-json", type=Path, default=selector.LOCK_JSON)
    parser.add_argument("--actions-jsonl", type=Path, required=True)
    parser.add_argument("--source-package-sha256", default=None)
    parser.add_argument("--compile-workers", type=int, default=8)
    args = parser.parse_args(argv)
    summary = prepare(
        output_root=args.output_root,
        actions_jsonl=args.actions_jsonl,
        lock_json=args.lock_json,
        source_package_sha256=args.source_package_sha256,
        compile_workers=args.compile_workers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTIONS_SCHEMA",
    "CAMPAIGN_ID",
    "COMPARATOR_MODE",
    "EXPECTED_ACTION_COUNT",
    "EXPECTED_APM_ACTION_COUNT",
    "EXPECTED_AVQDS_ACTION_COUNT",
    "SUMMARY_SCHEMA",
    "TELEMETRY_CONTRACT",
    "build_actions",
    "main",
    "prepare",
]
