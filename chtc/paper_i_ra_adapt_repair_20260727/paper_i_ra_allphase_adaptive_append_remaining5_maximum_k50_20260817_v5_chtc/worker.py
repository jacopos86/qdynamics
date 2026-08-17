#!/usr/bin/env python3
"""Execute and authenticate one source-locked three-arm CHTC cell."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    AUTH_SCHEMA,
    CAMPAIGN_ID,
    JOB_SCHEMA,
    PLAN_SCHEMA,
    RESULT_MANIFEST_SCHEMA,
    TARGET_HORIZON,
    WORKER_SCHEMA,
    canonical_sha256,
    digested,
    file_sha256,
    read_json,
    verify_digested,
)

from pipelines.static_adapt.adaptive_phase_contracts import (  # noqa: E402
    ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1,
)
from pipelines.static_adapt.ra_adapt import (  # noqa: E402
    RAAdaptOperationalControls,
    build_paper_i_ra_hh_regime_problem,
    materialize_paper_i_ra_semantic_protocol,
    run_ra_adapt,
    semantic_closure_source_implementation_inventory,
    validate_semantic_phase3_natural_terminal_route_contract,
)
from pipelines.static_adapt.ra_adapt.contracts import (  # noqa: E402
    canonical_sha256 as engine_canonical_sha256,
)
from pipelines.static_adapt.ra_adapt.replay_evidence import (  # noqa: E402
    validate_controller_replay_evidence,
)
from pipelines.static_adapt.sr_snake._resume import (  # noqa: E402
    CanonicalResumeError,
    load_canonical_accepted_state_resume,
)
from pipelines.static_adapt.sr_snake.contracts import (  # noqa: E402
    AcceptedStateResume,
    CheckpointObservation,
    EstimatorLedgerObservation,
    FreshStart,
    SRObservationPolicy,
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"artifact is not a regular file: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _completion(
    *,
    result: Mapping[str, Any],
    summary: Mapping[str, Any] | None,
    checkpoint: Path,
    problem: Any,
    protocol: Any,
) -> dict[str, Any]:
    run = result.get("run")
    scientific = result.get("scientific_receipts")
    if not isinstance(run, Mapping) or not isinstance(scientific, Mapping):
        raise RuntimeError("result lacks typed run/scientific evidence")
    trajectory = run.get("accepted_trajectory")
    transitions = run.get("accepted_transitions")
    replay_rows = run.get("scientific_replay")
    final_state = run.get("final_state")
    stop = run.get("stop")
    if not all(isinstance(value, list) for value in (trajectory, transitions, replay_rows)):
        raise RuntimeError("accepted trajectory evidence is malformed")
    if not isinstance(final_state, Mapping) or not isinstance(stop, Mapping):
        raise RuntimeError("final state or stop receipt is malformed")
    accepted = len(trajectory)
    if (
        len(transitions) != accepted
        or len(replay_rows) != accepted
        or [row.get("controller_round") for row in trajectory]
        != list(range(1, accepted + 1))
        or stop.get("completed_controller_rounds") != accepted
    ):
        raise RuntimeError("accepted trajectory cardinality drifted")

    route = protocol.to_dict()["route_contract"]
    route_sha = str(route["sha256"])
    validate_semantic_phase3_natural_terminal_route_contract(
        route,
        expected_route_contract_sha256=route_sha,
    )
    replay = validate_controller_replay_evidence(
        scientific.get("controller_replay_evidence")
    )
    if len(replay.get("signed_controller_round_prefixes", [])) != accepted:
        raise RuntimeError("controller replay prefix cardinality drifted")

    terminal = stop.get("terminal_controller_outcome")
    terminal_receipt_sha: str | None = None
    if terminal == ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1:
        if accepted >= TARGET_HORIZON:
            raise RuntimeError("natural terminal is outside the maximum horizon")
        receipt = scientific.get("terminal_phase3_selection_receipt")
        if not isinstance(receipt, Mapping):
            raise RuntimeError("natural terminal receipt is absent")
        terminal_receipt_sha = str(receipt.get("sha256", ""))
        if (
            # The engine digests receipts with its own canonical convention
            # (no trailing newline); the package convention appends one.
            terminal_receipt_sha != engine_canonical_sha256(
                {key: value for key, value in receipt.items() if key != "sha256"}
            )
            or receipt.get("accepted_controller_round") != accepted
            or receipt.get("attempted_controller_round") != accepted + 1
            or receipt.get("accepted_state_unchanged") is not True
            or receipt.get("final_admission_record_id") is not None
            or replay.get("terminal_controller_outcome") != terminal
        ):
            raise RuntimeError("natural terminal state binding drifted")
        if accepted == 0:
            if summary is not None:
                raise RuntimeError("round-zero terminal must not have a summary")
        elif not isinstance(summary, Mapping) or summary.get("available_controller_rounds") != accepted:
            raise RuntimeError("natural terminal summary drifted")
        completion_kind = "authenticated_phase3_no_positive_natural_terminal_v1"
        attempted = accepted + 1
        natural = True
    elif terminal is None:
        if (
            accepted != TARGET_HORIZON
            or stop.get("primary_reason") != "maximum_controller_rounds"
            or not isinstance(summary, Mapping)
            or summary.get("available_controller_rounds") != TARGET_HORIZON
            or scientific.get("terminal_phase3_selection_receipt") is not None
        ):
            raise RuntimeError("nonterminal completion is not exact k50")
        completion_kind = "reached_maximum_controller_rounds_v1"
        attempted = None
        natural = False
    else:
        raise RuntimeError("unauthorized terminal controller outcome")

    checkpoint_sha = file_sha256(checkpoint)
    route_row = run.get("route")
    if not isinstance(route_row, Mapping):
        raise RuntimeError("run route binding is absent")
    try:
        hydration = load_canonical_accepted_state_resume(
            AcceptedStateResume(
                checkpoint_path=checkpoint,
                checkpoint_sha256=checkpoint_sha,
            ),
            expected_problem=problem,
            expected_route_profile=str(route_row.get("profile", "")),
            expected_route_contract_sha256=str(route_row.get("contract_sha256", "")),
        )
    except CanonicalResumeError as exc:
        if not natural or "natural terminal" not in str(exc) or "non-resumable" not in str(exc):
            raise RuntimeError("checkpoint authentication failed") from exc
        hydration = None
    if natural and hydration is not None:
        raise RuntimeError("natural terminal advertised resumability")
    if not natural and (hydration is None or hydration.controller_round != TARGET_HORIZON):
        raise RuntimeError("exact-k50 checkpoint depth drifted")

    return digested(
        {
            "schema": "paper_i_ra_allphase_adaptive_append_remaining5_cell_completion_v1",
            "completion_kind": completion_kind,
            "maximum_controller_rounds": TARGET_HORIZON,
            "accepted_controller_rounds": accepted,
            "terminal_attempted_controller_round": attempted,
            "terminal_controller_outcome": terminal,
            "terminal_phase3_selection_receipt_sha256": terminal_receipt_sha,
            "protocol_sha256": protocol.sha256,
            "route_contract_sha256": route_sha,
            "final_state_sha256": canonical_sha256(dict(final_state)),
            "checkpoint_file_sha256": checkpoint_sha,
            "controller_replay_evidence_sha256": replay["sha256"],
        }
    )


def run(job_path: Path, protocol_path: Path, authorization_path: Path, output: Path) -> int:
    job = verify_digested(read_json(job_path), schema=JOB_SCHEMA)
    authorization = verify_digested(read_json(authorization_path), schema=AUTH_SCHEMA)
    plan_path = authorization_path.parent / "execution_plan.json"
    plan = verify_digested(read_json(plan_path), schema=PLAN_SCHEMA)
    if (
        authorization.get("execution_authorized") is not True
        or authorization.get("submission_authorized") is not True
        or authorization.get("plan_sha256") != plan["sha256"]
        or job.get("execution_id") not in authorization.get("execution_ids", [])
        or job.get("campaign_id") != CAMPAIGN_ID
        or job.get("source_inventory_sha256")
        != semantic_closure_source_implementation_inventory()["sha256"]
        or job.get("protocol_file_sha256") != file_sha256(protocol_path)
    ):
        raise RuntimeError("job authority or source binding drifted")

    import pipelines.static_adapt.ra_adapt as public_ra

    builder = getattr(public_ra, str(job["builder"]))
    expected_route = getattr(public_ra, str(job["route_constant"]))
    problem = build_paper_i_ra_hh_regime_problem(str(job["regime_id"]))
    request = builder(
        insertion_policy=str(job["insertion_policy"]),
        maximum_controller_rounds=TARGET_HORIZON,
    )
    protocol = materialize_paper_i_ra_semantic_protocol(problem, request)
    protocol_mapping = protocol.to_dict()
    if (
        read_json(protocol_path) != protocol_mapping
        or protocol.sha256 != job.get("protocol_sha256")
        or protocol_mapping["route_contract"]["native_semantic_contract"]["route_variant"]
        != expected_route
    ):
        raise RuntimeError("protocol rematerialization drifted")

    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "checkpoints/current.json"
    ledger = output / "result/estimator_ledger.json"
    result_obj = run_ra_adapt(
        problem,
        protocol,
        operational_controls=RAAdaptOperationalControls(
            maximum_controller_rounds=TARGET_HORIZON,
            resume=FreshStart(),
            observation=SRObservationPolicy(
                checkpoint=CheckpointObservation(
                    path=checkpoint,
                    every_controller_rounds=1,
                    keep_history_tail=TARGET_HORIZON,
                ),
                estimator_ledger=EstimatorLedgerObservation(path=ledger),
                resource_rounds=tuple(range(1, TARGET_HORIZON + 1)),
            ),
        ),
    )
    result = result_obj.to_dict()
    summary = (
        result_obj.run.paper_i_summary.to_dict()
        if result_obj.run.paper_i_summary is not None
        else None
    )
    completion = _completion(
        result=result,
        summary=summary,
        checkpoint=checkpoint,
        problem=problem,
        protocol=protocol,
    )
    # Recompute after science to reject any lazy source drift.
    if materialize_paper_i_ra_semantic_protocol(problem, request).to_dict() != protocol_mapping:
        raise RuntimeError("protocol drifted during execution")
    result_path = output / "result/result.json"
    _write_json(result_path, result)
    summary_path: Path | None = None
    if summary is not None:
        summary_path = output / "summary/summary.json"
        _write_json(summary_path, summary)
    paths = [checkpoint, ledger, result_path]
    if summary_path is not None:
        paths.append(summary_path)
    manifest = digested(
        {
            "schema": RESULT_MANIFEST_SCHEMA,
            "status": "passed_authenticated_maximum_k50",
            "campaign_id": CAMPAIGN_ID,
            "execution_id": job["execution_id"],
            "job_sha256": job["sha256"],
            "authorization_sha256": authorization["sha256"],
            "cell_completion": completion,
            "artifacts": [_artifact(path, output) for path in paths],
        }
    )
    _write_json(output / "execution_manifest.json", manifest)
    worker = digested(
        {
            "schema": WORKER_SCHEMA,
            "status": "passed_authenticated_maximum_k50",
            "campaign_id": CAMPAIGN_ID,
            "execution_id": job["execution_id"],
            "manifest_sha256": manifest["sha256"],
        }
    )
    _write_json(output / "worker_receipt.json", worker)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(args.job, args.protocol, args.authorization, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
