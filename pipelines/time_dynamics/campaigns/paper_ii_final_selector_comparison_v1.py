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
from dataclasses import asdict
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics import paper_ii_runs as runs
from pipelines.time_dynamics.ap_mclachlan.exchange_config import resolve_route_bundle
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
PERMISSION_RAY_FLAG = "--exchange-permission-ray-distance-max"
COMPARATOR_REUSE = "locked_historical_avqds_15_cells"
ACTIONS_SCHEMA = "paper_ii_final_selector_comparison_actions_v1"
SUMMARY_SCHEMA = "paper_ii_final_selector_comparison_prepare_v1"
CAMPAIGN_SCORING_WORKERS = 4
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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
    campaign_id: str
    payload: Mapping[str, Any]
    cells: tuple[MatchedCell, ...]
    exclusions: tuple[tuple[str, str], ...]
    registry_bindings: Mapping[str, Any]
    seed: Mapping[str, Any]
    selector_routes: Mapping[str, str]
    # Declared threshold for the pre-permission ray bound on an unrefit
    # deletion, carried as the lock's own argv literal.  None is the
    # historical coupling to the post-refit ray tolerance.
    permission_ray_distance_max: str | None = None

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


def _bound_sha256(value: Any, *, where: str) -> str:
    digest = str(value or "").lower()
    _require(bool(_SHA256_RE.fullmatch(digest)), f"{where}: invalid sha256")
    return digest


def _verify_repo_artifact(entry: Mapping[str, Any], *, where: str) -> Path:
    _require(entry.get("resolution_status") == "RESOLVED", f"{where}: unresolved")
    relative = Path(str(entry.get("path") or ""))
    _require(bool(str(relative)) and not relative.is_absolute(), f"{where}: path must be repo-relative")
    target = (_REPO_ROOT / relative).resolve()
    _require(target.is_relative_to(_REPO_ROOT), f"{where}: path escapes the repository")
    _require(target.is_file(), f"{where}: missing artifact {relative}")
    expected = _bound_sha256(entry.get("sha256"), where=f"{where}.sha256")
    observed = str(file_sha256(target))
    _require(observed == expected, f"{where}: sha256 drift ({observed})")
    return target


def _verify_submission_evidence(payload: Mapping[str, Any]) -> None:
    blocks = payload.get("blocks")
    _require(isinstance(blocks, Mapping), "lock blocks must be an object")
    _require(list(blocks.get("active") or ()) == [], "lock carries an active submission block")
    superseded_path = Path(str(blocks.get("supersedes") or ""))
    _require(
        not (_REPO_ROOT / superseded_path).is_file(),
        "superseded report_refresh_required.json is still present",
    )
    _bound_sha256(blocks.get("superseded_sha256"), where="blocks.superseded_sha256")

    external = payload.get("external_artifacts")
    _require(isinstance(external, Mapping), "lock external_artifacts must be an object")
    required = (
        "promotion_preview",
        "ledger",
        "final_ingestion_receipt",
        "report_json",
        "report_refresh_resolution",
    )
    resolved: dict[str, Path] = {}
    for key in required:
        entry = external.get(key)
        _require(isinstance(entry, Mapping), f"external_artifacts.{key}: absent")
        resolved[key] = _verify_repo_artifact(entry, where=f"external_artifacts.{key}")

    results_pdf = external.get("results_pdf")
    _require(isinstance(results_pdf, Mapping), "external_artifacts.results_pdf: absent")
    _require(results_pdf.get("resolution_status") == "RESOLVED", "results PDF unresolved")
    _require(
        results_pdf.get("currency_claim") == "current_through_coverage_cluster_9718010",
        "results PDF currency is not bound to the final coverage cluster",
    )
    _bound_sha256(results_pdf.get("sha256"), where="external_artifacts.results_pdf.sha256")

    resolution = json.loads(resolved["report_refresh_resolution"].read_text(encoding="utf-8"))
    _require(
        resolution.get("schema") == "paper_ii_report_refresh_resolution_v1",
        "report-refresh resolution schema mismatch",
    )
    decision = resolution.get("resolution") or {}
    _require(decision.get("status") == "RESOLVED", "report-refresh resolution is not final")
    _require(list(decision.get("active_blocks") or ()) == [], "resolution retains active blocks")
    _require(
        decision.get("authorizes_new_campaign_submission") is True,
        "resolution does not authorize new campaign submission",
    )
    calibration = (resolution.get("resolution_evidence") or {}).get("calibration_receipt") or {}
    calibration_path = _verify_repo_artifact(
        {**calibration, "resolution_status": "RESOLVED"},
        where="resolution.calibration_receipt",
    )
    receipt = json.loads(calibration_path.read_text(encoding="utf-8"))
    _require(int((receipt.get("counts") or {}).get("error", -1)) == 0, "calibration has ERROR checks")
    _require(
        receipt.get("status") in ("OK", "INCOMPLETE"),
        "calibration status is neither OK nor no-error INCOMPLETE",
    )


def lock_parameters(path: str | Path) -> dict[str, Any]:
    """Return a lock's own declared identity and shape, without validating it.

    A packager that must accept more than one lock reads its parameters here
    and hands them straight back to :func:`load_lock`, so the loader still
    fails closed on every check that is not the shape the lock declares about
    itself.  Callers that intend a single fixed campaign pass nothing and get
    the module constants.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    routes = payload.get("selector_routes")
    cells = payload.get("cells")
    return {
        "campaign_id": str(payload.get("campaign_id") or ""),
        "expected_cell_count": len(cells) if isinstance(cells, list) else 0,
        "expected_route_count": len(routes) if isinstance(routes, Mapping) else 0,
    }


def load_lock(
    path: str | Path = LOCK_JSON,
    *,
    campaign_id: str = CAMPAIGN_ID,
    expected_cell_count: int = EXPECTED_CELL_COUNT,
    expected_route_count: int = EXPECTED_ROUTE_COUNT,
) -> MatchedCellLock:
    """Read and validate the matched-cell lock, failing closed on drift."""

    lock_path = Path(path)
    if not lock_path.is_file():
        raise FileNotFoundError(f"missing matched-cell lock: {lock_path}")
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    _require(payload.get("schema") == LOCK_SCHEMA, "lock schema mismatch")
    _require(payload.get("status") == LOCK_STATUS, "lock status mismatch")
    _require(payload.get("campaign_id") == campaign_id, "lock campaign_id mismatch")
    _require("code_revision" not in payload, "lock must not pin a code_revision")
    _verify_submission_evidence(payload)

    raw_cells = payload.get("cells")
    _require(isinstance(raw_cells, list), "lock cells must be a list")
    _require(
        len(raw_cells) == expected_cell_count,
        f"lock must carry exactly {expected_cell_count} cells; got {len(raw_cells)}",
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
        _bound_sha256(row.get("tuple_id"), where=f"{cell_id}.tuple_id")
        _require(isinstance(row.get("metrics"), Mapping), f"{cell_id}: metrics unresolved")
        _require(isinstance(row.get("paper_status"), Mapping), f"{cell_id}: paper status unresolved")
        _require(isinstance(row.get("run_lock"), Mapping), f"{cell_id}: run lock unresolved")
        sources = row.get("source_artifacts")
        _require(isinstance(sources, Mapping), f"{cell_id}: source artifacts unresolved")
        for method in ("apm", "avqds"):
            method_sources = sources.get(method)
            _require(isinstance(method_sources, Mapping), f"{cell_id}: {method} sources absent")
            for source_key in ("screen_run_json", "continuation_run_json", "terminal_cost_json"):
                source = method_sources.get(source_key)
                _require(isinstance(source, Mapping), f"{cell_id}: {method} {source_key} absent")
                _require(bool(source.get("path")), f"{cell_id}: {method} {source_key} path absent")
                _bound_sha256(source.get("sha256"), where=f"{cell_id}.{method}.{source_key}")
        exact = sources.get("exact_reference")
        _require(isinstance(exact, Mapping), f"{cell_id}: exact-reference source absent")
        _require(bool(exact.get("path")), f"{cell_id}: exact-reference path absent")
        _bound_sha256(exact.get("sha256"), where=f"{cell_id}.exact_reference")
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
    extra = tuple(str(flag) for flag in bindings.get("extra_flags") or ())
    permission_ray_distance_max: str | None = None
    if extra == ("--record-statevector",):
        pass
    elif (
        len(extra) == 3
        and extra[:2] == ("--record-statevector", PERMISSION_RAY_FLAG)
    ):
        try:
            declared = float(extra[2])
        except ValueError:
            declared = math.nan
        _require(
            math.isfinite(declared) and declared > 0.0,
            f"{PERMISSION_RAY_FLAG} must be a positive finite literal",
        )
        permission_ray_distance_max = str(extra[2])
    else:
        _require(
            False,
            "locked campaign permits only --record-statevector, optionally "
            f"followed by {PERMISSION_RAY_FLAG} <value>, as extra flags",
        )

    seed = payload["seed"]
    _require(
        str(seed["artifact_json"]) == runs.REGIMES[bindings["regime"]].seed_path,
        "seed artifact path does not match the bound regime",
    )

    routes = payload["selector_routes"]
    _require(
        len(routes) == expected_route_count,
        f"lock must bind exactly {expected_route_count} selector routes",
    )
    selector_routes = {str(route_id): str(spec["arm"]) for route_id, spec in routes.items()}

    return MatchedCellLock(
        path=lock_path,
        sha256=str(file_sha256(lock_path)),
        campaign_id=str(campaign_id),
        payload=payload,
        cells=tuple(cells),
        exclusions=tuple(exclusions),
        registry_bindings=dict(bindings),
        seed=dict(seed),
        selector_routes=selector_routes,
        permission_ray_distance_max=permission_ray_distance_max,
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
    code_revision: str,
) -> None:
    from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
        _build_parser,
        _resolve_times,
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
    if not math.isclose(float(parsed.t_final), float(horizon.t_final), rel_tol=0.0):
        raise MatchedCellLockError(f"{label}: effective --t-final drifted")
    if int(parsed.num_times) != int(horizon.num_times):
        raise MatchedCellLockError(f"{label}: effective --num-times drifted")
    if parsed.times is not None or parsed.t_initial is not None:
        raise MatchedCellLockError(f"{label}: explicit time-grid override is forbidden")
    resolved_times = _resolve_times(parsed)
    if (
        len(resolved_times) != int(horizon.num_times)
        or not math.isclose(float(resolved_times[0]), 0.0, rel_tol=0.0, abs_tol=0.0)
        or not math.isclose(
            float(resolved_times[-1]), float(horizon.t_final), rel_tol=0.0, abs_tol=0.0
        )
    ):
        raise MatchedCellLockError(f"{label}: resolved time grid is not locked t=0..20")
    if str(parsed.artifact_json) != str(lock.seed["artifact_json"]):
        raise MatchedCellLockError(f"{label}: effective seed artifact drifted")
    if str(parsed.code_revision) != str(code_revision):
        raise MatchedCellLockError(f"{label}: effective code revision drifted")
    if not bool(parsed.record_statevector):
        raise MatchedCellLockError(f"{label}: --record-statevector absent")
    if str(parsed.dynamics_policy) != "exchange":
        raise MatchedCellLockError(f"{label}: APM action is not full exchange")
    if str(parsed.insertion_gate_mode) != "mclachlan_l2":
        raise MatchedCellLockError(f"{label}: insertion gate is not realized L2")
    if str(parsed.debt_policy) != "drift_ranked" or not bool(parsed.exchange_deletions):
        raise MatchedCellLockError(f"{label}: generalized-exchange policy drifted")
    if int(parsed.support_patch_scoring_workers) != CAMPAIGN_SCORING_WORKERS:
        raise MatchedCellLockError(f"{label}: scoring-worker contract drifted")
    assert_permission_ray_distance_max(parsed, lock=lock, label=label)


def assert_permission_ray_distance_max(
    parsed: Any, *, lock: MatchedCellLock, label: str
) -> None:
    """Require the effective pre-permission threshold to be the lock's."""

    effective = parsed.exchange_permission_ray_distance_max
    declared = lock.permission_ray_distance_max
    if declared is None:
        if effective is not None:
            raise MatchedCellLockError(
                f"{label}: {PERMISSION_RAY_FLAG} {effective} is not declared by the lock"
            )
        return
    if effective is None or not math.isclose(
        float(effective), float(declared), rel_tol=1.0e-12, abs_tol=0.0
    ):
        raise MatchedCellLockError(
            f"{label}: {PERMISSION_RAY_FLAG} {effective} != locked {declared}"
        )


def build_actions(
    lock: MatchedCellLock,
    *,
    output_root: str | Path,
    source_package_sha256: str | None = None,
    execution_flags: Sequence[str] = (),
) -> tuple[dict[str, Any], ...]:
    """Build exactly ``cells x routes`` APM actions and zero comparators."""

    if source_package_sha256 is not None:
        source_package_sha256 = _bound_sha256(
            source_package_sha256,
            where="source_package_sha256",
        )
    routes = bind_selector_routes(lock.selector_routes)
    bindings = lock.registry_bindings
    # This is an execution-only knob: scoring and compilation replay in a
    # canonical order and the worker-count parity contract is tested at
    # 1/2/4 workers.  Pin four explicitly so CHTC requests and run receipts
    # cannot silently inherit a different parser default.
    extra_flags = (
        "--record-statevector",
        "--support-patch-scoring-workers",
        str(CAMPAIGN_SCORING_WORKERS),
    )
    code_revision = resolve_code_revision(_REPO_ROOT)
    if code_revision in ("", "unknown"):
        raise MatchedCellLockError("could not resolve a code revision for the actions")
    # Execution-resource flags (compile pool size) are not scientific
    # settings and are not part of the lock; they are appended per package so
    # the slot sizing travels with the argv and its provenance.
    extra_flags = extra_flags + tuple(str(flag) for flag in execution_flags)
    if lock.permission_ray_distance_max is not None:
        # A scientific setting declared by the lock, emitted as its literal so
        # the argv, the run id, and the receipt carry exactly what was locked.
        extra_flags = extra_flags + (
            PERMISSION_RAY_FLAG,
            str(lock.permission_ray_distance_max),
        )
    root = Path(output_root)
    rows: list[dict[str, Any]] = []
    for cell in lock.cells:
        for route_id, arm_id in routes.items():
            route_contract = asdict(resolve_route_bundle(route_id))
            route_contract_sha256 = hashlib.sha256(
                json.dumps(
                    route_contract,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            action_id = _action_id(cell, arm_id)
            output_json = root / lock.campaign_id / action_id / "run.json"
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
            argv = (*command.argv(), "--code-revision", str(code_revision))
            _parse_and_assert(
                argv,
                cell=cell,
                route_id=route_id,
                lock=lock,
                code_revision=code_revision,
            )
            rows.append(
                {
                    "schema": ACTIONS_SCHEMA,
                    "campaign_id": lock.campaign_id,
                    "action_id": action_id,
                    "cell_id": cell.cell_id,
                    "drive": cell.drive,
                    "controller": cell.controller,
                    "method": "apm",
                    "arm": arm_id,
                    "exchange_route_bundle": route_id,
                    "exchange_route_contract": route_contract,
                    "exchange_route_contract_sha256": route_contract_sha256,
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
                    "code_revision": str(code_revision),
                    "matched_cell_lock_sha256": str(lock.sha256),
                    "seed_sha256": str(lock.seed["sha256"]),
                    "source_package_sha256": source_package_sha256,
                }
            )
    expected = len(lock.cells) * len(routes)
    if len(rows) != expected:
        raise MatchedCellLockError(f"expected {expected} actions; got {len(rows)}")
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
    source_package_sha256: str | None = None,
    execution_flags: Sequence[str] = (),
    campaign_id: str = CAMPAIGN_ID,
    expected_cell_count: int = EXPECTED_CELL_COUNT,
    expected_route_count: int = EXPECTED_ROUTE_COUNT,
) -> dict[str, Any]:
    """Validate the lock, build the actions, write the JSONL, and summarize."""

    lock = load_lock(
        lock_json,
        campaign_id=campaign_id,
        expected_cell_count=expected_cell_count,
        expected_route_count=expected_route_count,
    )
    seed_sha = _verify_seed(lock)
    routes = bind_selector_routes(lock.selector_routes)
    rows = build_actions(
        lock,
        output_root=output_root,
        source_package_sha256=source_package_sha256,
        execution_flags=execution_flags,
    )
    written = write_actions_jsonl(rows, actions_jsonl)
    avqds_rows = [row for row in rows if row["method"] != "apm"]
    return {
        "schema": SUMMARY_SCHEMA,
        "campaign_id": lock.campaign_id,
        "status": LOCK_STATUS,
        "lock": {
            "path": str(lock.path),
            "sha256": lock.sha256,
            "schema": LOCK_SCHEMA,
        },
        "code_revision": str(rows[0]["code_revision"]),
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
        "execution_flags": [str(flag) for flag in execution_flags],
        "output_root": str(Path(output_root)),
        "source_package_sha256": source_package_sha256,
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
    parser.add_argument(
        "--source-package-sha256",
        default=None,
        help="optional SHA-256 of the immutable source archive carried by every action",
    )
    parser.add_argument(
        "--compile-workers",
        type=int,
        default=None,
        help=(
            "compile process-pool size appended to every action argv "
            "(execution only; size to the slot: request_cpus >= compile_workers + 1)"
        ),
    )
    args = parser.parse_args(argv)
    execution_flags: list[str] = []
    if args.compile_workers is not None:
        execution_flags += ["--compile-workers", str(int(args.compile_workers))]
    actions_jsonl = (
        args.actions_jsonl
        if args.actions_jsonl is not None
        else args.output_root / CAMPAIGN_ID / "actions.jsonl"
    )
    summary = prepare(
        output_root=args.output_root,
        actions_jsonl=actions_jsonl,
        lock_json=args.lock_json,
        source_package_sha256=args.source_package_sha256,
        execution_flags=tuple(execution_flags),
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
    "PERMISSION_RAY_FLAG",
    "UnboundSelectorRouteError",
    "assert_permission_ray_distance_max",
    "bind_selector_routes",
    "build_actions",
    "load_lock",
    "lock_parameters",
    "main",
    "prepare",
    "write_actions_jsonl",
]
