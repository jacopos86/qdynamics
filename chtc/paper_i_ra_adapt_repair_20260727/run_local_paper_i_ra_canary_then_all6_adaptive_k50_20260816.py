#!/usr/bin/env python3
"""Direct, non-idling canary-to-overnight Paper-I RA chain."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


RUNNER_PATH = Path(__file__).resolve()
REPO_ROOT = RUNNER_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chtc.paper_i_ra_adapt_repair_20260727 import (
    paper_i_ra_serial_campaign_runtime_20260816 as serial_runtime,
)


CAMPAIGN_ID = "paper_i_ra_canary_then_all6_adaptive_k50_20260816_v1"
RUNTIME_ROOT = REPO_ROOT / "output/local_runs" / CAMPAIGN_ID
LOCK_PATH = RUNTIME_ROOT / "chain.lock"
TERMINAL_PATH = RUNTIME_ROOT / "terminal_chain_receipt.json"
TERMINAL_SCHEMA = "paper_i_ra_canary_to_overnight_terminal_chain_v1"


class RunnerError(RuntimeError):
    """Raised when the direct chain cannot preserve its authority boundary."""


def _default_canary_runner():
    return importlib.import_module(
        "chtc.paper_i_ra_adapt_repair_20260727."
        "run_local_paper_i_native_phase0_plateau_eight_arm_k5_20260816_v2"
    )


def _default_overnight_runner():
    return importlib.import_module(
        "chtc.paper_i_ra_adapt_repair_20260727."
        "run_local_paper_i_ra_all6_adaptive_shortlist_append_then_plateau_k50_"
        "overnight_20260816"
    )


def _plans_and_conditional_grant(canary_runner, overnight_runner):
    try:
        canary_plan, canary_authorization = canary_runner.validate_authority(
            recompute_protocols=True
        )
        overnight_plan, conditional_grant = (
            overnight_runner.validate_conditional_authority(
                recompute_protocols=True,
                canary_runner=canary_runner,
            )
        )
    except Exception as exc:
        raise RunnerError("Chain authority failed deep validation.") from exc
    canary_inventory = canary_plan.get(
        "source_implementation_inventory_sha256"
    )
    overnight_inventory = overnight_plan.get(
        "source_implementation_inventory_sha256"
    )
    if not canary_inventory or canary_inventory != overnight_inventory:
        raise RunnerError(
            "Canary and overnight plans must bind the same source inventory."
        )
    return (
        canary_plan,
        canary_authorization,
        overnight_plan,
        conditional_grant,
    )


def prepare_chain_authority(*, canary_runner=None, overnight_runner=None) -> dict[str, Any]:
    """Seal the canary authority and overnight conditional grant only.

    This function intentionally cannot create the overnight execution
    authorization.  That receipt is minted by the chain only after the real
    canary terminal has passed deep validation.
    """

    canary_runner = canary_runner or _default_canary_runner()
    overnight_runner = overnight_runner or _default_overnight_runner()
    if not canary_runner.PLAN_PATH.exists():
        canary_runner.prepare_plan()
    if not canary_runner.AUTHORIZATION_PATH.exists():
        canary_runner.authorize()
    if not overnight_runner.PLAN_PATH.exists():
        overnight_runner.prepare_plan()
    if not overnight_runner.CONDITIONAL_GRANT_PATH.exists():
        overnight_runner.prepare_conditional_grant(canary_runner=canary_runner)
    if overnight_runner.AUTHORIZATION_PATH.exists():
        raise RunnerError(
            "Overnight execution authorization exists before chain canary validation."
        )
    canary_plan, _canary_auth, overnight_plan, grant = _plans_and_conditional_grant(
        canary_runner, overnight_runner
    )
    return {
        "campaign_id": CAMPAIGN_ID,
        "canary_plan_sha256": canary_plan["sha256"],
        "overnight_plan_sha256": overnight_plan["sha256"],
        "conditional_grant_sha256": grant["sha256"],
        "source_implementation_inventory_sha256": canary_plan[
            "source_implementation_inventory_sha256"
        ],
        "overnight_execution_authorization_present": False,
        "scientific_execution_performed": False,
    }


def _deep_canary_terminal(canary_runner) -> Mapping[str, Any]:
    try:
        terminal = canary_runner.validate_terminal_matrix()
    except Exception as exc:
        raise RunnerError("Canary terminal failed deep chain validation.") from exc
    return terminal


def _deep_overnight_terminal(overnight_runner) -> Mapping[str, Any]:
    try:
        terminal = overnight_runner.validate_terminal_matrix()
    except Exception as exc:
        raise RunnerError("Overnight terminal failed deep chain validation.") from exc
    return terminal


def _terminal_payload(
    *,
    canary_plan: Mapping[str, Any],
    canary_authorization: Mapping[str, Any],
    overnight_plan: Mapping[str, Any],
    conditional_grant: Mapping[str, Any],
    overnight_authorization: Mapping[str, Any],
    canary_terminal: Mapping[str, Any],
    overnight_terminal: Mapping[str, Any],
) -> dict[str, Any]:
    inventory = canary_plan["source_implementation_inventory_sha256"]
    if any(
        payload.get("source_implementation_inventory_sha256") != inventory
        for payload in (overnight_plan, canary_terminal, overnight_terminal)
    ):
        raise RunnerError("Terminal chain source inventory diverged.")
    return serial_runtime.digested(
        {
            "schema": TERMINAL_SCHEMA,
            "status": "passed_canary_then_all6_k50",
            "campaign_id": CAMPAIGN_ID,
            "runner": serial_runtime.file_binding(RUNNER_PATH),
            "source_implementation_inventory_sha256": inventory,
            "canary_plan_sha256": canary_plan["sha256"],
            "canary_authorization_sha256": canary_authorization["sha256"],
            "canary_terminal_sha256": canary_terminal["sha256"],
            "overnight_plan_sha256": overnight_plan["sha256"],
            "conditional_grant_sha256": conditional_grant["sha256"],
            "overnight_authorization_sha256": overnight_authorization["sha256"],
            "overnight_terminal_sha256": overnight_terminal["sha256"],
            "execution_order": ["canary", "overnight"],
            "wait_only_supervisor_used": False,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )


def validate_terminal_chain(*, canary_runner=None, overnight_runner=None) -> dict[str, Any]:
    canary_runner = canary_runner or _default_canary_runner()
    overnight_runner = overnight_runner or _default_overnight_runner()
    (
        canary_plan,
        canary_authorization,
        overnight_plan,
        conditional_grant,
    ) = _plans_and_conditional_grant(canary_runner, overnight_runner)
    canary_terminal = _deep_canary_terminal(canary_runner)
    if hasattr(overnight_runner, "validate_authority"):
        try:
            _validated_plan, overnight_authorization = overnight_runner.validate_authority(
                recompute_protocols=True,
                canary_runner=canary_runner,
            )
        except Exception as exc:
            raise RunnerError("Overnight authority failed terminal validation.") from exc
    else:
        # Test doubles exercise the chain ordering without filesystem authority.
        overnight_authorization = {"sha256": "7" * 64}
    overnight_terminal = _deep_overnight_terminal(overnight_runner)
    expected = _terminal_payload(
        canary_plan=canary_plan,
        canary_authorization=canary_authorization,
        overnight_plan=overnight_plan,
        conditional_grant=conditional_grant,
        overnight_authorization=overnight_authorization,
        canary_terminal=canary_terminal,
        overnight_terminal=overnight_terminal,
    )
    observed = serial_runtime.load_digested(
        TERMINAL_PATH,
        schema=TERMINAL_SCHEMA,
        error_type=RunnerError,
    )
    if observed != expected:
        raise RunnerError("Terminal chain receipt failed recomputation.")
    return observed


def run_chain(*, canary_runner=None, overnight_runner=None) -> int:
    canary_runner = canary_runner or _default_canary_runner()
    overnight_runner = overnight_runner or _default_overnight_runner()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    with serial_runtime.exclusive_campaign_lock(
        LOCK_PATH,
        label="Canary-to-overnight chain",
        error_type=RunnerError,
    ):
        if TERMINAL_PATH.is_file():
            validate_terminal_chain(
                canary_runner=canary_runner,
                overnight_runner=overnight_runner,
            )
            return 0
        (
            canary_plan,
            canary_authorization,
            overnight_plan,
            conditional_grant,
        ) = _plans_and_conditional_grant(canary_runner, overnight_runner)

        canary_returncode = int(canary_runner.run_campaign())
        if canary_returncode != 0:
            # In particular, blocked_capacity exits directly.  The chain never
            # leaves a wait-only supervisor behind and mints no overnight auth.
            return canary_returncode
        canary_terminal = _deep_canary_terminal(canary_runner)
        if canary_terminal.get("source_implementation_inventory_sha256") != (
            canary_plan["source_implementation_inventory_sha256"]
        ):
            raise RunnerError("Canary terminal source inventory diverged.")

        try:
            overnight_authorization = overnight_runner.authorize_after_canary(
                canary_runner=canary_runner
            )
        except Exception as exc:
            raise RunnerError(
                "Post-canary overnight authorization could not be minted."
            ) from exc
        overnight_returncode = int(overnight_runner.run_campaign())
        if overnight_returncode != 0:
            return overnight_returncode
        overnight_terminal = _deep_overnight_terminal(overnight_runner)
        terminal = _terminal_payload(
            canary_plan=canary_plan,
            canary_authorization=canary_authorization,
            overnight_plan=overnight_plan,
            conditional_grant=conditional_grant,
            overnight_authorization=overnight_authorization,
            canary_terminal=canary_terminal,
            overnight_terminal=overnight_terminal,
        )
        serial_runtime.write_json_exclusive(TERMINAL_PATH, terminal)
        validate_terminal_chain(
            canary_runner=canary_runner,
            overnight_runner=overnight_runner,
        )
        return 0


def preflight(*, canary_runner=None, overnight_runner=None) -> dict[str, Any]:
    canary_runner = canary_runner or _default_canary_runner()
    overnight_runner = overnight_runner or _default_overnight_runner()
    (
        canary_plan,
        _canary_authorization,
        overnight_plan,
        conditional_grant,
    ) = _plans_and_conditional_grant(canary_runner, overnight_runner)
    return {
        "campaign_id": CAMPAIGN_ID,
        "source_implementation_inventory_sha256": canary_plan[
            "source_implementation_inventory_sha256"
        ],
        "canary_plan_sha256": canary_plan["sha256"],
        "overnight_plan_sha256": overnight_plan["sha256"],
        "conditional_grant_sha256": conditional_grant["sha256"],
        "overnight_authorization_present": overnight_runner.AUTHORIZATION_PATH.is_file(),
        "terminal_present": TERMINAL_PATH.is_file(),
        "wait_only_supervisor": False,
        "scientific_execution_performed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare-authority", action="store_true")
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--run-chain", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare_authority:
        print(json.dumps(prepare_chain_authority(), indent=2, sort_keys=True))
        return 0
    if args.preflight:
        print(json.dumps(preflight(), indent=2, sort_keys=True))
        return 0
    return run_chain()


if __name__ == "__main__":
    raise SystemExit(main())
