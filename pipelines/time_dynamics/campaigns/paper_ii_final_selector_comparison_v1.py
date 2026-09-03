"""Prepare the Paper-II final three-route selector comparison.

The scientific matrix is the locked ``paper_ii_final_matched_cells_v1.json``:
fifteen drive x controller cells, each carrying the APM activation cut that
was matched against one locked historical AVQDS comparator.  This module
re-runs only the APM side of every cell, once per closed selector route
(``--exchange-route-bundle``), and never re-runs a comparator.

Every action resolves through :mod:`pipelines.time_dynamics.paper_ii_runs`
and is re-parsed with the live runner parser before it is written.  The
module prepares; it does not submit.

Example
-------

::

    python -m pipelines.time_dynamics.campaigns.paper_ii_final_selector_comparison_v1 \
        --output-root output/paper_ii_final_selector \
        --actions-jsonl output/paper_ii_final_selector/actions.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics import paper_ii_runs as runs
from pipelines.time_dynamics.campaign import _canonical_cut
from pipelines.time_dynamics.run_lock import file_sha256, resolve_code_revision

CAMPAIGN_ID = "paper_ii_final_selector_comparison_v1"
LOCK_SCHEMA = "paper_ii_final_matched_cells_v1"
LOCK_STATUS = "PREPARED_NOT_SUBMITTED"
LOCK_JSON = Path(__file__).with_name("paper_ii_final_matched_cells_v1.json")
EXPECTED_CELL_COUNT = 15
EXPECTED_ROUTE_COUNT = 3
EXCLUSION_REASON = "not_in_final_matched_set"
ADOPTION_STATUSES = ("pair_matched_comparison", "individual_response_target")
ROUTE_BUNDLE_FLAG = "--exchange-route-bundle"
COMPARATOR_REUSE = "locked_historical_avqds_15_cells"
ACTIONS_SCHEMA = "paper_ii_final_selector_comparison_actions_v1"
SUMMARY_SCHEMA = "paper_ii_final_selector_comparison_prepare_v1"
_REPO_ROOT = Path(__file__).resolve().parents[3]


class UnboundSelectorRouteError(ValueError):
    """A lock route id is not carried by the arm the lock binds it to."""


class MatchedCellLockError(ValueError):
    """The lock file violates its own declared schema."""


@dataclass(frozen=True)
class MatchedCell:
    """One drive x controller cell with its matched APM/AVQDS cuts."""

    cell_id: str
    drive: str
    controller: str
    apm_cut: float
    avqds_cut: float
    adoption_status: str

    @property
    def apm_cut_label(self) -> str:
        return f"{float(self.apm_cut):.12g}"

    @property
    def avqds_cut_label(self) -> str:
        return f"{float(self.avqds_cut):.12g}"


@dataclass(frozen=True)
class MatchedCellLock:
    """The parsed, validated lock."""

    path: Path
    sha256: str
    payload: Mapping[str, Any]
    cells: tuple[MatchedCell, ...]
    exclusions: tuple[tuple[str, str], ...]
    registry_bindings: Mapping[str, Any]
    seed: Mapping[str, Any]
    selector_routes: Mapping[str, str]

    @property
    def grid(self) -> tuple[tuple[str, str], ...]:
        drives = tuple(str(d) for d in self.payload["grid"]["drives"])
        controllers = tuple(str(c) for c in self.payload["grid"]["controllers"])
        return tuple((d, c) for d in drives for c in controllers)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatchedCellLockError(message)


def _cut(value: Any, *, where: str) -> float:
    try:
        cut = float(value)
    except (TypeError, ValueError) as exc:
        raise MatchedCellLockError(f"{where}: cut is not a float: {value!r}") from exc
    _require(math.isfinite(cut) and cut > 0.0, f"{where}: cut must be finite and positive")
    _require(
        _canonical_cut(cut) == cut,
        f"{where}: cut {cut!r} is not canonical ({_canonical_cut(cut)!r})",
    )
    return cut


def load_lock(path: str | Path = LOCK_JSON) -> MatchedCellLock:
    """Read and validate the matched-cell lock, failing closed on drift."""

    lock_path = Path(path)
    if not lock_path.is_file():
        raise FileNotFoundError(f"missing matched-cell lock: {lock_path}")
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    _require(payload.get("schema") == LOCK_SCHEMA, "lock schema mismatch")
    _require(payload.get("status") == LOCK_STATUS, "lock status mismatch")
    _require(payload.get("campaign_id") == CAMPAIGN_ID, "lock campaign_id mismatch")
    _require("code_revision" not in payload, "lock must not pin a code_revision")

    raw_cells = payload.get("cells")
    _require(isinstance(raw_cells, list), "lock cells must be a list")
    _require(
        len(raw_cells) == EXPECTED_CELL_COUNT,
        f"lock must carry exactly {EXPECTED_CELL_COUNT} cells; got {len(raw_cells)}",
    )
    cells: list[MatchedCell] = []
    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for row in raw_cells:
        cell_id = str(row["cell_id"])
        _require(cell_id not in seen_ids, f"duplicate cell id: {cell_id}")
        seen_ids.add(cell_id)
        drive = str(row["drive"])
        controller = str(row["controller"])
        _require(drive in runs.DRIVES, f"{cell_id}: unknown drive {drive!r}")
        _require(
            controller in runs.STEP_CONTROLS,
            f"{cell_id}: unknown controller {controller!r}",
        )
        _require((drive, controller) not in seen_pairs, f"{cell_id}: duplicate cell")
        seen_pairs.add((drive, controller))
        status = str(row["adoption_status"])
        _require(
            status in ADOPTION_STATUSES,
            f"{cell_id}: adoption_status {status!r} not in {ADOPTION_STATUSES}",
        )
        cells.append(
            MatchedCell(
                cell_id=cell_id,
                drive=drive,
                controller=controller,
                apm_cut=_cut(row["apm_cut"], where=f"{cell_id}.apm_cut"),
                avqds_cut=_cut(row["avqds_cut"], where=f"{cell_id}.avqds_cut"),
                adoption_status=status,
            )
        )

    exclusions: list[tuple[str, str]] = []
    for row in payload.get("exclusions") or ():
        _require(
            str(row.get("reason")) == EXCLUSION_REASON,
            f"exclusion reason must be {EXCLUSION_REASON!r}",
        )
        exclusions.append((str(row["drive"]), str(row["controller"])))

    drives = tuple(str(d) for d in payload["grid"]["drives"])
    controllers = tuple(str(c) for c in payload["grid"]["controllers"])
    grid = {(d, c) for d in drives for c in controllers}
    _require(
        seen_pairs | set(exclusions) == grid and not (seen_pairs & set(exclusions)),
        "cells + exclusions must partition the drive x controller grid",
    )

    bindings = payload["registry_bindings"]
    _require(bindings["regime"] in runs.REGIMES, "unknown regime binding")
    _require(bindings["horizon"] in runs.HORIZONS, "unknown horizon binding")
    _require(bindings["numerics"] in runs.NUMERICS, "unknown numerics binding")
    _require(bindings["gate"] in runs.GATES, "unknown gate binding")
    _require(
        bindings["search_profile"] in runs.SEARCH_PROFILES,
        "unknown search-profile binding",
    )
    _require(bindings["apm_arm"] in runs.ARMS, "unknown APM arm binding")
    _require(
        runs.ARMS[bindings["comparator_arm"]].is_comparator,
        "comparator_arm must be a comparator arm",
    )

    seed = payload["seed"]
    _require(
        str(seed["artifact_json"]) == runs.REGIMES[bindings["regime"]].seed_path,
        "seed artifact path does not match the bound regime",
    )

    routes = payload["selector_routes"]
    _require(
        len(routes) == EXPECTED_ROUTE_COUNT,
        f"lock must bind exactly {EXPECTED_ROUTE_COUNT} selector routes",
    )
    selector_routes = {str(route_id): str(spec["arm"]) for route_id, spec in routes.items()}

    return MatchedCellLock(
        path=lock_path,
        sha256=str(file_sha256(lock_path)),
        payload=payload,
        cells=tuple(cells),
        exclusions=tuple(exclusions),
        registry_bindings=dict(bindings),
        seed=dict(seed),
        selector_routes=selector_routes,
    )


def bind_selector_routes(
    selector_routes: Mapping[str, str],
) -> dict[str, str]:
    """Return ``{route_id: arm_id}`` after proving each arm carries its route.

    An arm counts as bound only when its registered flags contain the exact
    pair ``("--exchange-route-bundle", route_id)``.  Anything else fails
    closed: a lock must never name a route the registry would silently
    replace with the historical default bundle.
    """

    bound: dict[str, str] = {}
    for route_id, arm_id in selector_routes.items():
        arm = runs.ARMS.get(arm_id)
        if arm is None:
            raise UnboundSelectorRouteError(
                f"route {route_id!r} names unregistered arm {arm_id!r}"
            )
        flags = tuple(arm.flags)
        pairs = tuple(zip(flags, flags[1:]))
        if (ROUTE_BUNDLE_FLAG, route_id) not in pairs:
            raise UnboundSelectorRouteError(
                f"arm {arm_id!r} does not carry {ROUTE_BUNDLE_FLAG} {route_id!r}; "
                f"flags: {flags}"
            )
        if arm.is_comparator:
            raise UnboundSelectorRouteError(
                f"route {route_id!r} is bound to comparator arm {arm_id!r}"
            )
        bound[route_id] = arm_id
    return bound


def _verify_seed(lock: MatchedCellLock) -> str:
    seed_path = _REPO_ROOT / str(lock.seed["artifact_json"])
    observed = file_sha256(seed_path)
    declared = str(lock.seed["sha256"]).lower()
    if observed != declared:
        raise MatchedCellLockError(
            f"seed {seed_path} sha256 {observed} != locked {declared}"
        )
    return declared


def _action_id(cell: MatchedCell, arm_id: str) -> str:
    return f"{cell.cell_id}__{arm_id}"


def _parse_and_assert(
    argv: Sequence[str],
    *,
    cell: MatchedCell,
    route_id: str,
    lock: MatchedCellLock,
) -> None:
    from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
        _build_parser,
    )

    argv = tuple(argv)
    parsed = _build_parser().parse_args(list(argv))
    label = _action_id(cell, lock.selector_routes[route_id])
    if not math.isclose(
        float(parsed.insertion_l2_cut), float(cell.apm_cut), rel_tol=1.0e-12, abs_tol=0.0
    ):
        raise MatchedCellLockError(
            f"{label}: --insertion-l2-cut {parsed.insertion_l2_cut} != {cell.apm_cut}"
        )
    if str(parsed.exchange_route_bundle) != route_id:
        raise MatchedCellLockError(
            f"{label}: route bundle {parsed.exchange_route_bundle!r} != {route_id!r}"
        )
    if "--resume-from-run-json" in argv or parsed.resume_from_run_json is not None:
        raise MatchedCellLockError(f"{label}: a fresh trajectory must not resume")
    horizon = runs.HORIZONS[str(lock.registry_bindings["horizon"])]
    pairs = set(zip(argv, argv[1:]))
    if ("--t-final", repr(float(horizon.t_final))) not in pairs:
        raise MatchedCellLockError(f"{label}: --t-final {horizon.t_final!r} absent")
    if ("--num-times", str(int(horizon.num_times))) not in pairs:
        raise MatchedCellLockError(f"{label}: --num-times {horizon.num_times} absent")
    if not bool(parsed.record_statevector):
        raise MatchedCellLockError(f"{label}: --record-statevector absent")
    if str(parsed.dynamics_policy) == "avqds":
        raise MatchedCellLockError(f"{label}: an APM action resolved to the comparator")


def build_actions(
    lock: MatchedCellLock,
    *,
    output_root: str | Path,
) -> tuple[dict[str, Any], ...]:
    """Build exactly ``15 cells x 3 routes`` APM actions and zero comparators."""

    routes = bind_selector_routes(lock.selector_routes)
    bindings = lock.registry_bindings
    extra_flags = tuple(str(flag) for flag in bindings.get("extra_flags") or ())
    if "--record-statevector" not in extra_flags:
        extra_flags = extra_flags + ("--record-statevector",)
    root = Path(output_root)
    rows: list[dict[str, Any]] = []
    for cell in lock.cells:
        for route_id, arm_id in routes.items():
            action_id = _action_id(cell, arm_id)
            output_json = root / CAMPAIGN_ID / action_id / "run.json"
            command = runs.build_run(
                regime=str(bindings["regime"]),
                arm=arm_id,
                gate=str(bindings["gate"]),
                drive=cell.drive,
                horizon=str(bindings["horizon"]),
                numerics=str(bindings["numerics"]),
                step_control=cell.controller,
                search_profile=str(bindings["search_profile"]),
                activation_cut=cell.apm_cut,
                output_json=str(output_json),
                extra_flags=extra_flags,
                require_seed=True,
            )
            argv = tuple(command.argv())
            _parse_and_assert(argv, cell=cell, route_id=route_id, lock=lock)
            rows.append(
                {
                    "schema": ACTIONS_SCHEMA,
                    "campaign_id": CAMPAIGN_ID,
                    "action_id": action_id,
                    "cell_id": cell.cell_id,
                    "drive": cell.drive,
                    "controller": cell.controller,
                    "method": "apm",
                    "arm": arm_id,
                    "exchange_route_bundle": route_id,
                    "apm_cut": float(cell.apm_cut),
                    "apm_cut_label": cell.apm_cut_label,
                    "adoption_status": cell.adoption_status,
                    "comparator": {
                        "arm": str(bindings["comparator_arm"]),
                        "avqds_cut": float(cell.avqds_cut),
                        "avqds_cut_label": cell.avqds_cut_label,
                        "reuse": COMPARATOR_REUSE,
                        "action": None,
                    },
                    "runner_module": runs.RUNNER_MODULE,
                    "runner_argv": list(argv),
                    "trajectory_json": str(output_json),
                    "run_id": command.run_id,
                }
            )
    if len(rows) != EXPECTED_CELL_COUNT * EXPECTED_ROUTE_COUNT:
        raise MatchedCellLockError(
            f"expected {EXPECTED_CELL_COUNT * EXPECTED_ROUTE_COUNT} actions; got {len(rows)}"
        )
    ids = [row["action_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise MatchedCellLockError("action ids are not unique")
    run_ids = [row["run_id"] for row in rows]
    if len(set(run_ids)) != len(run_ids):
        raise MatchedCellLockError("run ids are not unique")
    return tuple(rows)


def write_actions_jsonl(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def prepare(
    *,
    output_root: str | Path,
    actions_jsonl: str | Path,
    lock_json: str | Path = LOCK_JSON,
) -> dict[str, Any]:
    """Validate the lock, build the actions, write the JSONL, and summarize."""

    lock = load_lock(lock_json)
    seed_sha = _verify_seed(lock)
    routes = bind_selector_routes(lock.selector_routes)
    rows = build_actions(lock, output_root=output_root)
    written = write_actions_jsonl(rows, actions_jsonl)
    avqds_rows = [row for row in rows if row["method"] != "apm"]
    return {
        "schema": SUMMARY_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "status": LOCK_STATUS,
        "lock": {
            "path": str(lock.path),
            "sha256": lock.sha256,
            "schema": LOCK_SCHEMA,
        },
        "code_revision": resolve_code_revision(_REPO_ROOT),
        "seed": {"path": str(lock.seed["artifact_json"]), "sha256": seed_sha},
        "registry_bindings": dict(lock.registry_bindings),
        "selector_routes": routes,
        "cell_count": len(lock.cells),
        "excluded_cell_count": len(lock.exclusions),
        "route_count": len(routes),
        "action_count": len(rows),
        "avqds_action_count": len(avqds_rows),
        "comparator_actions": 0,
        "comparator_reuse": COMPARATOR_REUSE,
        "comparator_note": (
            "AVQDS comparators are reused locked comparators; no comparator "
            "trajectory is re-run by this campaign."
        ),
        "actions_jsonl": {"path": str(written), "sha256": file_sha256(written)},
        "output_root": str(Path(output_root)),
        "blocks": list(lock.payload["blocks"]["active"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/paper_ii_final_selector"),
        help="root under which each action's run.json is placed",
    )
    parser.add_argument(
        "--lock-json",
        type=Path,
        default=LOCK_JSON,
        help="matched-cell lock to prepare from",
    )
    parser.add_argument(
        "--actions-jsonl",
        type=Path,
        default=None,
        help="where to write the prepared actions (default: <output-root>/<campaign>/actions.jsonl)",
    )
    args = parser.parse_args(argv)
    actions_jsonl = (
        args.actions_jsonl
        if args.actions_jsonl is not None
        else args.output_root / CAMPAIGN_ID / "actions.jsonl"
    )
    summary = prepare(
        output_root=args.output_root,
        actions_jsonl=actions_jsonl,
        lock_json=args.lock_json,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADOPTION_STATUSES",
    "CAMPAIGN_ID",
    "COMPARATOR_REUSE",
    "EXCLUSION_REASON",
    "EXPECTED_CELL_COUNT",
    "EXPECTED_ROUTE_COUNT",
    "LOCK_JSON",
    "LOCK_SCHEMA",
    "MatchedCell",
    "MatchedCellLock",
    "MatchedCellLockError",
    "UnboundSelectorRouteError",
    "bind_selector_routes",
    "build_actions",
    "load_lock",
    "main",
    "prepare",
    "write_actions_jsonl",
]
