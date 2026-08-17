#!/usr/bin/env python3
"""Source-locked v2 supervisor for the native Phase-0 eight-arm canary.

This is a new campaign identity.  It never imports, mutates, or adopts the
blocked-capacity v1 campaign's authority or runtime artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


RUNNER_PATH = Path(__file__).resolve()
REPO_ROOT = RUNNER_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.static_adapt.ra_adapt.semantic_closure_routes import (
    PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2,
    PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2,
    PAPER_I_RA_PHASE0_POSITION_GRADIENT_ADAPTIVE_V1,
    PAPER_I_RA_PHASE0_POSITION_GRADIENT_FIXED24_V1,
    PAPER_I_RA_PHASE0_POSITION_PROXY_ADAPTIVE_V1,
    PAPER_I_RA_PHASE0_POSITION_PROXY_FIXED24_V1,
    PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2,
    PAPER_I_RA_PHASE0_PROXY_FIXED24_V2,
)
from chtc.paper_i_ra_adapt_repair_20260727 import (
    paper_i_ra_serial_campaign_runtime_20260816 as serial_runtime,
)
from chtc.paper_i_ra_adapt_repair_20260727 import (
    run_local_paper_i_native_phase0_plateau_eight_arm_k5_20260816 as _v1,
)


LEGACY_V1_RUNNER_PATH = RUNNER_PATH.with_name(
    "run_local_paper_i_native_phase0_plateau_eight_arm_k5_20260816.py"
)
CAMPAIGN_ID = "paper_i_native_phase0_plateau_eight_arm_k5_20260816_v2"
TARGET_HORIZON = 5
AUTHORITY_DIR = RUNNER_PATH.parent / f"{CAMPAIGN_ID}_authority"
PLAN_PATH = AUTHORITY_DIR / "plan.json"
AUTHORIZATION_PATH = AUTHORITY_DIR / "authorization.json"
RUNTIME_ROOT = REPO_ROOT / "output/local_runs" / CAMPAIGN_ID
RUNS_ROOT = RUNTIME_ROOT / "runs"
STAGING_ROOT = RUNTIME_ROOT / "in_progress"
RECEIPTS_ROOT = RUNTIME_ROOT / "worker_receipts"
GUARD_ROOT = RUNTIME_ROOT / "guard_receipts"
STATUS_PATH = RUNTIME_ROOT / "status.json"
LOCK_PATH = RUNTIME_ROOT / "campaign.lock"
REPORT_JSON = RUNTIME_ROOT / "comparison.json"
REPORT_CSV = RUNTIME_ROOT / "comparison.csv"
REPORT_MD = RUNTIME_ROOT / "comparison.md"
TERMINAL_PATH = RUNTIME_ROOT / "terminal_matrix_receipt.json"

CHILD_TOKEN_ENV = "PAPER_I_NATIVE_PHASE0_EIGHT_ARM_V2_CHILD_TOKEN"
PLAN_SCHEMA = "paper_i_native_phase0_eight_arm_plateau_plan_v2"
AUTH_SCHEMA = "paper_i_native_phase0_eight_arm_plateau_authorization_v2"
MANIFEST_SCHEMA = "paper_i_native_phase0_eight_arm_execution_manifest_v2"
WORKER_SCHEMA = "paper_i_native_phase0_eight_arm_worker_receipt_v2"
GUARD_SCHEMA = "paper_i_native_phase0_eight_arm_guard_receipt_v2"
REPORT_SCHEMA = "paper_i_native_phase0_eight_arm_comparison_v2"
TERMINAL_SCHEMA = "paper_i_native_phase0_eight_arm_terminal_matrix_v2"
STATUS_SCHEMA = "paper_i_native_phase0_eight_arm_status_v2"

RunnerError = _v1.RunnerError


@dataclass(frozen=True, slots=True)
class CellSpec:
    ordinal: int
    placement: str
    score: str
    cardinality: str
    route_variant: str
    execution_id: str
    insertion_policy: str = "plateau_commutation"
    horizon: int = TARGET_HORIZON
    regime_id: str = "strong_weak_u8"
    nph: int = 3


def _cell(
    ordinal: int,
    placement: str,
    score: str,
    cardinality: str,
    route_variant: str,
) -> CellSpec:
    return CellSpec(
        ordinal=ordinal,
        placement=placement,
        score=score,
        cardinality=cardinality,
        route_variant=route_variant,
        execution_id=(
            "native_phase0_plateau_k5_v2__strong_weak_u8__nph3__"
            f"{placement}__{score}__{cardinality}"
        ),
    )


CELL_SPECS = (
    _cell(1, "generator_first", "gradient", "fixed24", PAPER_I_RA_PHASE0_GRADIENT_FIXED24_V2),
    _cell(2, "position_aware", "gradient", "fixed24", PAPER_I_RA_PHASE0_POSITION_GRADIENT_FIXED24_V1),
    _cell(3, "generator_first", "gradient", "adaptive", PAPER_I_RA_PHASE0_GRADIENT_ADAPTIVE_V2),
    _cell(4, "position_aware", "gradient", "adaptive", PAPER_I_RA_PHASE0_POSITION_GRADIENT_ADAPTIVE_V1),
    _cell(5, "generator_first", "proxy", "fixed24", PAPER_I_RA_PHASE0_PROXY_FIXED24_V2),
    _cell(6, "position_aware", "proxy", "fixed24", PAPER_I_RA_PHASE0_POSITION_PROXY_FIXED24_V1),
    _cell(7, "generator_first", "proxy", "adaptive", PAPER_I_RA_PHASE0_PROXY_ADAPTIVE_V2),
    _cell(8, "position_aware", "proxy", "adaptive", PAPER_I_RA_PHASE0_POSITION_PROXY_ADAPTIVE_V1),
)


def _sync_v1_runtime() -> None:
    """Point the proven v1 supervisor mechanics at the isolated v2 campaign.

    The v1 source is a read-only runtime dependency.  No v1 authority or output
    path is consulted, and its file binding is sealed into every v2 plan.
    """

    values = {
        "RUNNER_PATH": RUNNER_PATH,
        "REPO_ROOT": REPO_ROOT,
        "CAMPAIGN_ID": CAMPAIGN_ID,
        "AUTHORITY_DIR": AUTHORITY_DIR,
        "PLAN_PATH": PLAN_PATH,
        "AUTHORIZATION_PATH": AUTHORIZATION_PATH,
        "RUNTIME_ROOT": RUNTIME_ROOT,
        "RUNS_ROOT": RUNS_ROOT,
        "STAGING_ROOT": STAGING_ROOT,
        "RECEIPTS_ROOT": RECEIPTS_ROOT,
        "GUARD_ROOT": GUARD_ROOT,
        "STATUS_PATH": STATUS_PATH,
        "LOCK_PATH": LOCK_PATH,
        "REPORT_JSON": REPORT_JSON,
        "REPORT_CSV": REPORT_CSV,
        "REPORT_MD": REPORT_MD,
        "TERMINAL_PATH": TERMINAL_PATH,
        "CHILD_TOKEN_ENV": CHILD_TOKEN_ENV,
        "PLAN_SCHEMA": PLAN_SCHEMA,
        "AUTH_SCHEMA": AUTH_SCHEMA,
        "MANIFEST_SCHEMA": MANIFEST_SCHEMA,
        "WORKER_SCHEMA": WORKER_SCHEMA,
        "GUARD_SCHEMA": GUARD_SCHEMA,
        "REPORT_SCHEMA": REPORT_SCHEMA,
        "TERMINAL_SCHEMA": TERMINAL_SCHEMA,
        "STATUS_SCHEMA": STATUS_SCHEMA,
        "CELL_SPECS": CELL_SPECS,
        "TARGET_HORIZON": TARGET_HORIZON,
    }
    for name, value in values.items():
        setattr(_v1, name, value)


def _runtime_dependencies() -> list[dict[str, Any]]:
    return [
        serial_runtime.file_binding(LEGACY_V1_RUNNER_PATH),
        serial_runtime.file_binding(Path(serial_runtime.__file__).resolve()),
    ]


def build_plan() -> dict[str, Any]:
    _sync_v1_runtime()
    payload = dict(_v1.build_plan())
    payload.pop("sha256", None)
    payload["runner_runtime_dependencies"] = _runtime_dependencies()
    return serial_runtime.digested(payload)


def prepare_plan() -> dict[str, Any]:
    plan = build_plan()
    serial_runtime.prepare_authority_directory(
        AUTHORITY_DIR,
        files={"plan.json": plan},
        error_type=RunnerError,
    )
    return plan


def validate_plan(*, recompute_protocols: bool) -> dict[str, Any]:
    _sync_v1_runtime()
    plan = _v1.validate_plan(recompute_protocols=recompute_protocols)
    if plan.get("runner_runtime_dependencies") != _runtime_dependencies():
        raise RunnerError("V2 canary runtime dependency binding drifted.")
    return plan


def authorize() -> dict[str, Any]:
    _sync_v1_runtime()
    return _v1.authorize()


def validate_authority(
    *, recompute_protocols: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    _sync_v1_runtime()
    plan, authorization = _v1.validate_authority(
        recompute_protocols=recompute_protocols
    )
    if plan.get("runner_runtime_dependencies") != _runtime_dependencies():
        raise RunnerError("V2 canary runtime dependency binding drifted.")
    return plan, authorization


def wait_for_capacity(**kwargs: Any) -> dict[str, Any]:
    _sync_v1_runtime()
    return _v1.wait_for_capacity(**kwargs)


def cell_paths(cell: CellSpec):
    _sync_v1_runtime()
    return _v1.cell_paths(cell)


def load_closed_cell(
    cell: CellSpec,
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
):
    _sync_v1_runtime()
    return _v1.load_closed_cell(cell, plan=plan, authorization=authorization)


def build_comparison(cells):
    _sync_v1_runtime()
    return _v1.build_comparison(cells)


def validate_terminal_matrix(
    *,
    plan: Mapping[str, Any] | None = None,
    authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _sync_v1_runtime()
    if plan is None or authorization is None:
        plan, authorization = validate_authority(recompute_protocols=True)
    return _v1.validate_terminal_matrix(plan=plan, authorization=authorization)


def run_child(execution_id: str) -> int:
    _sync_v1_runtime()
    validate_authority(recompute_protocols=False)
    return _v1.run_child(execution_id)


def run_campaign() -> int:
    _sync_v1_runtime()
    validate_authority(recompute_protocols=True)
    return _v1.run_campaign()


def preflight() -> dict[str, Any]:
    plan = build_plan() if not PLAN_PATH.is_file() else validate_plan(
        recompute_protocols=True
    )
    return {
        "campaign_id": CAMPAIGN_ID,
        "cell_count": len(CELL_SPECS),
        "fixed_serial_order": [cell.execution_id for cell in CELL_SPECS],
        "source_implementation_inventory_sha256": plan[
            "source_implementation_inventory_sha256"
        ],
        "plan_present": PLAN_PATH.is_file(),
        "authorization_present": AUTHORIZATION_PATH.is_file(),
        "runtime_present": RUNTIME_ROOT.exists(),
        "legacy_v1_runtime_dependency_only": True,
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
