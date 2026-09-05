"""Operate the crossed-method staged Paper-II accuracy campaign.

This command is orchestration only.  It prints registered runner commands and
records returned artifacts in an immutable ledger; it never implements a
trajectory method, exact propagator, or resource-cost rule.

Examples
--------

Create/inspect the crossed APM/AVQDS t=5 worklist::

    python -m pipelines.time_dynamics.campaigns.paper_ii_staged_accuracy_v1 \
        next --ledger output/paper_ii_staged/ledger.json \
        --prior-root output/frontier \
        --output-root output/paper_ii_staged/raw_outputs

Record one completed action and replay the ledger::

    python -m pipelines.time_dynamics.campaigns.paper_ii_staged_accuracy_v1 \
        record --ledger output/paper_ii_staged/ledger.json \
        --prior-root output/frontier --action-json action.json \
        --run-json output/.../run.json --seed-json chtc/.../hh_snake_nph1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics import paper_ii_runs as runs
from pipelines.time_dynamics.campaign import (
    DEFAULT_THRESHOLD_LADDER,
    plan_thresholds_from_prior,
)
from pipelines.time_dynamics.campaigns.paper_ii_factorial_euler_v1 import (
    CONTROLLER_IDS,
    DRIVE_IDS,
    METHOD_IDS,
    NUMERICS_ID,
    SEED,
)
from pipelines.time_dynamics.response_scales import (
    load_verified_response_scale_index,
)
from pipelines.time_dynamics.staged_campaign import (
    ACTION_RECORD_SAFEGUARD,
    METHOD_APM,
    AccuracyPolicy,
    CampaignLedger,
    CellAction,
    CellKey,
    StagedCampaign,
    ThresholdSchedule,
    factorial_cells,
)
from pipelines.time_dynamics.staged_campaign_artifacts import (
    record_prepared_action_outputs,
    record_run_outcome,
    record_safeguard_action,
)
from pipelines.time_dynamics.staged_campaign_chtc import (
    prepare_staged_chtc_campaign,
)


CAMPAIGN_ID = "paper_ii_staged_accuracy_euler_v1"
# The locked calibration campaign runs the seed every measurement was taken on
# to the t20 promotion horizon. A regime survey reuses the same staged machinery
# for another registered regime and a shorter promotion horizon; its campaign id
# and output root are suffixed so it can never be confused with the locked
# ledger, and the canonical response-normalized policy stays bound to the
# calibration seed's verified exact-response sidecar.
DEFAULT_REGIME = "hh_snake_nph1"
DEFAULT_PROMOTION_HORIZON = "t20"


DEFAULT_APM_ARM = METHOD_APM
REGIME_RESPONSE_SCALE_REGISTRY = Path(__file__).with_name(
    "paper_ii_exact_response_scales_by_regime.json"
)
REGIME_RESPONSE_SCALE_REGISTRY_SCHEMA = "paper_ii_exact_response_scales_by_regime_v1"


def campaign_id_for(
    regime: str, promotion_horizon: str, apm_arm: str = DEFAULT_APM_ARM
) -> str:
    if (
        regime == DEFAULT_REGIME
        and promotion_horizon == DEFAULT_PROMOTION_HORIZON
        and apm_arm == DEFAULT_APM_ARM
    ):
        return CAMPAIGN_ID
    suffix = f"{CAMPAIGN_ID}__{regime}__{promotion_horizon}"
    return suffix if apm_arm == DEFAULT_APM_ARM else f"{suffix}__{apm_arm}"


def _regime_response_scales(regime: str) -> Mapping[str, Any]:
    """Load a survey regime's verified exact-response index from the registry.

    Stage 1.5 writes the registry row with
    ``pipelines.time_dynamics.diagnostics.build_regime_response_scales``; the
    loader re-derives every scale from the content-addressed exact reference,
    so a stale or edited sidecar fails closed here rather than at scoring time.
    """

    repository_root = Path(__file__).resolve().parents[3]
    registry_path = REGIME_RESPONSE_SCALE_REGISTRY
    producer = (
        "PYTHONPATH=. python -m pipelines.time_dynamics.diagnostics."
        f"build_regime_response_scales --regime {regime}"
    )
    if not registry_path.is_file():
        raise ValueError(
            f"no exact-response scales are registered for regime {regime!r}: "
            f"{registry_path.name} is absent. Build them first with `{producer}`."
        )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("schema") != REGIME_RESPONSE_SCALE_REGISTRY_SCHEMA:
        raise ValueError(f"unsupported response-scale registry: {registry_path}")
    row = (registry.get("regimes") or {}).get(regime)
    if not isinstance(row, Mapping):
        raise ValueError(
            f"no exact-response scales are registered for regime {regime!r}. "
            f"Build them first with `{producer}`."
        )
    def _resolve(value: str) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else repository_root / path
    seed_path = _resolve(runs.REGIMES[regime].seed_path)
    return load_verified_response_scale_index(
        _resolve(row["sidecar"]),
        expected_sidecar_sha256=str(row["sha256"]),
        exact_reference_root=_resolve(row["exact_reference_root"]),
        seed_path=seed_path,
        regime_id=regime,
        family_key=SEED.family_key,
        controller_ids=CONTROLLER_IDS,
        expected_drives={
            drive_id: {
                "amplitude": runs.DRIVES[drive_id].amplitude,
                "omega": runs.DRIVES[drive_id].omega,
            }
            for drive_id in DRIVE_IDS
        },
    )
PRIOR_ENERGY_TARGET = 1.0e-4
ACCURACY_CEILING = 1.0e-4
PREFERRED_ACCURACY_FLOOR = 1.0e-5
DEFAULT_EXTENSION_SEED = (3.0e-6, 1.0e-6, 3.0e-7, 1.0e-7)
RESPONSE_RELATIVE_TARGET = 0.05
RESPONSE_RELATIVE_GUARD = 0.15
MATCHED_ACCURACY_FACTOR = 2.0
ABSOLUTE_ENERGY_ERROR_CEILING = 1.0e-2
ABSOLUTE_DOUBLON_ERROR_CEILING = 5.0e-3
ABSOLUTE_SITE_OCCUPATION_ERROR_CEILING = 5.0e-3
EXACT_RESPONSE_SCALES = Path(__file__).with_name(
    "paper_ii_exact_response_scales_v1.json"
)
# Rebaselined 2026-09-01 onto the regenerated exact-reference store.  Per the
# same-day pin-migration ruling the sidecar pins each reference's
# data_sha256 -- the reference.npz float payload -- instead of the manifest's
# byte digest, which was never a function of the physics (exact_reference_store
# writes manifest.json without sort_keys, so its digest also encodes the JSON
# key order of whichever driver regenerated the store).  The store itself is now
# tracked under output/paper_ii_staged/exact_references/, so a fresh checkout
# verifies without regenerating.  The sidecar's "rebaseline" block carries the
# full history: superseded manifest pins, the causal correction, and the
# physics-agreement evidence.
# Pin history: ee9d1a3d... (lost cluster-built store)
#           -> f7bb1dfd... (2026-09-01 rebaseline)
#           -> 4fc1b732... (2026-09-01 causal correction)
EXACT_RESPONSE_SCALES_SHA256 = (
    "3e6d42a6241e7ee954ce9f20d7245aa67ac82307d25ed2943126f1ead7e778df"
)


def _response_scales() -> Mapping[str, Any]:
    repository_root = Path(__file__).resolve().parents[3]
    return load_verified_response_scale_index(
        EXACT_RESPONSE_SCALES,
        expected_sidecar_sha256=EXACT_RESPONSE_SCALES_SHA256,
        exact_reference_root=(
            repository_root / "output/paper_ii_staged/exact_references"
        ),
        seed_path=repository_root / SEED.artifact_json,
        regime_id=SEED.seed_id,
        family_key=SEED.family_key,
        controller_ids=CONTROLLER_IDS,
        expected_drives={
            drive_id: {
                "amplitude": runs.DRIVES[drive_id].amplitude,
                "omega": runs.DRIVES[drive_id].omega,
            }
            for drive_id in DRIVE_IDS
        },
    )


def _configuration_id(key: CellKey) -> str:
    return f"{key.method}__{key.controller}__{key.drive}"


def _thresholds_from_prior(
    cells: Sequence[CellKey],
    prior_root: str | Path | None,
) -> Mapping[CellKey, tuple[float, ...]]:
    if prior_root is None:
        return {key: DEFAULT_EXTENSION_SEED for key in cells}
    root = Path(prior_root)
    if not root.is_dir():
        raise FileNotFoundError(f"prior threshold root does not exist: {root}")
    plan = plan_thresholds_from_prior(
        prior_root=root,
        method_ids=METHOD_IDS,
        controller_ids=CONTROLLER_IDS,
        drive_ids=DRIVE_IDS,
        target_mean_abs_energy_error=PRIOR_ENERGY_TARGET,
        fallback_ladder=DEFAULT_THRESHOLD_LADDER,
    )
    return {
        key: tuple(
            float(value)
            for value in plan.thresholds_by_configuration[_configuration_id(key)]
        )
        for key in cells
    }


def build_campaign(
    *,
    prior_root: str | Path | None,
    submission_policy_replay: str = "canonical",
    regime: str = DEFAULT_REGIME,
    promotion_horizon: str = DEFAULT_PROMOTION_HORIZON,
    apm_arm: str = DEFAULT_APM_ARM,
) -> StagedCampaign:
    if regime not in runs.REGIMES:
        raise KeyError(f"unknown regime {regime!r}; known: {sorted(runs.REGIMES)}")
    if promotion_horizon not in runs.HORIZONS:
        raise KeyError(
            f"unknown promotion horizon {promotion_horizon!r}; "
            f"known: {sorted(runs.HORIZONS)}"
        )
    cells = factorial_cells(
        regime=regime,
        drives=DRIVE_IDS,
        controllers=CONTROLLER_IDS,
        methods=METHOD_IDS,
    )
    if submission_policy_replay == "canonical":
        response_scales = (
            _response_scales()
            if regime == DEFAULT_REGIME
            else _regime_response_scales(regime)
        )
        accuracy_policy = AccuracyPolicy.response_normalized_matched_pair(
            response_scales=response_scales,
            response_relative_target=RESPONSE_RELATIVE_TARGET,
            response_relative_guard=RESPONSE_RELATIVE_GUARD,
            matched_accuracy_factor=MATCHED_ACCURACY_FACTOR,
            infidelity_ceiling=ACCURACY_CEILING,
            absolute_energy_error_ceiling=ABSOLUTE_ENERGY_ERROR_CEILING,
            absolute_doublon_error_ceiling=ABSOLUTE_DOUBLON_ERROR_CEILING,
            absolute_site_occupation_error_ceiling=(
                ABSOLUTE_SITE_OCCUPATION_ERROR_CEILING
            ),
        )
    elif submission_policy_replay == "componentwise_1e-4":
        accuracy_policy = AccuracyPolicy(
            ceiling=ACCURACY_CEILING,
            preferred_floor=PREFERRED_ACCURACY_FLOOR,
            require_infidelity=True,
        )
    else:
        raise ValueError(
            "unsupported submission-policy replay: "
            f"{submission_policy_replay!r}"
        )
    return StagedCampaign(
        cells=cells,
        accuracy_policy=accuracy_policy,
        threshold_schedule=ThresholdSchedule(
            _thresholds_from_prior(cells, prior_root),
            extension_ratio=3.0,
            minimum_threshold=1.0e-12,
            maximum_rungs_per_horizon=24,
        ),
        promotion_horizon=promotion_horizon,
        comparator_first=True,
        apm_arm=apm_arm,
    )


def _load_ledger(path: Path) -> CampaignLedger:
    return CampaignLedger.read(path) if path.is_file() else CampaignLedger()


def _action_output_path(
    output_root: Path, action: CellAction, *, campaign_id: str = CAMPAIGN_ID
) -> Path:
    return output_root / campaign_id / action.action_id / "run.json"


def _resolved_action(
    campaign: StagedCampaign,
    ledger: CampaignLedger,
    payload: Mapping[str, Any],
) -> CellAction:
    declared = CellAction.from_json_dict(payload)
    current = {action.action_id: action for action in campaign.next_actions(ledger)}
    if declared.action_id not in current:
        raise ValueError(
            f"action {declared.action_id!r} is not current for this ledger"
        )
    if current[declared.action_id] != declared:
        raise ValueError("action payload differs from the ledger replay")
    return declared


def _next_payload(
    *,
    campaign: StagedCampaign,
    ledger: CampaignLedger,
    output_root: Path,
    search_profile: str,
    campaign_id: str = CAMPAIGN_ID,
    seed_path: str = SEED.artifact_json,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for action in campaign.next_actions(ledger):
        row = action.to_json_dict()
        if action.action != ACTION_RECORD_SAFEGUARD:
            output = _action_output_path(output_root, action, campaign_id=campaign_id)
            command = campaign.build_run_command(
                action,
                seed_path=seed_path,
                output_json=str(output),
                numerics=NUMERICS_ID,
                gate=runs.MCLACHLAN_L2_GATE.gate_id,
                search_profile=search_profile,
            )
            row.update(
                {
                    "output_json": str(output),
                    "runner_argv": list(command.argv()),
                    "runner_shell": command.shell(),
                    "run_id": command.run_id,
                }
            )
        rows.append(row)
    return {
        "campaign_id": campaign_id,
        "stage": campaign.stage(ledger),
        "ledger_record_count": len(ledger.records),
        "action_count": len(rows),
        "actions": rows,
    }


def _write_action_files(payload: Mapping[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for row in payload.get("actions") or ():
        action_id = str(row["action_id"])
        target = directory / f"{action_id}.json"
        target.write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _selected_action_rows(
    actions_jsonl: Path, action_ids: Sequence[str]
) -> tuple[dict[str, Any], ...]:
    """Read a prepared package and retain an exact process-level increment."""

    rows = tuple(
        json.loads(line)
        for line in actions_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    selected_ids = tuple(str(value) for value in action_ids)
    if not selected_ids:
        return rows
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("record-package action selection contains duplicates")
    by_id = {str(row.get("action_id") or ""): row for row in rows}
    missing = [action_id for action_id in selected_ids if action_id not in by_id]
    if missing:
        raise ValueError(
            "record-package action selection is absent from the prepared package: "
            + ", ".join(missing)
        )
    return tuple(by_id[action_id] for action_id in selected_ids)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command",
        choices=(
            "next",
            "prepare-chtc",
            "record",
            "record-package",
            "record-safeguard",
            "reclassify",
        ),
    )
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--prior-root", type=Path)
    parser.add_argument(
        "--regime",
        choices=tuple(runs.REGIMES),
        default=DEFAULT_REGIME,
        help=(
            "registered regime whose seed every cell starts from; anything but "
            f"{DEFAULT_REGIME} is a survey campaign with a suffixed campaign id"
        ),
    )
    parser.add_argument(
        "--promotion-horizon",
        choices=tuple(runs.HORIZONS),
        default=DEFAULT_PROMOTION_HORIZON,
        help="continuation horizon after the t5 screen",
    )
    parser.add_argument(
        "--apm-arm",
        choices=tuple(
            arm_id for arm_id, arm in runs.ARMS.items() if not arm.is_comparator
        ),
        default=DEFAULT_APM_ARM,
        help=(
            "registered arm that realizes the APM method (the manuscript's "
            "realized Gamma(v*) route is exchange_q_compiled where registered)"
        ),
    )
    parser.add_argument(
        "--submission-policy-replay",
        choices=("canonical", "componentwise_1e-4"),
        default="canonical",
        help=(
            "replay an already-submitted package under its bound policy; the "
            "legacy componentwise choice is ingestion-only"
        ),
    )
    parser.add_argument(
        "--search-profile",
        choices=tuple(runs.SEARCH_PROFILES),
        default=runs.PAPER_DECLARED_SEARCH.profile_id,
    )
    parser.add_argument("--output-root", type=Path, default=Path("output/paper_ii_staged"))
    parser.add_argument("--action-dir", type=Path)
    parser.add_argument("--worklist-json", type=Path)
    parser.add_argument("--action-json", type=Path)
    parser.add_argument("--run-json", type=Path)
    parser.add_argument("--actions-jsonl", type=Path)
    parser.add_argument(
        "--action-id",
        action="append",
        default=[],
        help="record only this exact prepared action; repeat for an increment",
    )
    parser.add_argument(
        "--completed-actions-jsonl",
        type=Path,
        help="durable normalized copy of the selected increment",
    )
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--seed-json", type=Path)
    parser.add_argument("--reference-substeps", type=int, default=40)
    parser.add_argument(
        "--reference-store-root",
        type=Path,
        default=Path("output/paper_ii_staged/exact_references"),
    )
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument("--image-path", default="chtc/time_dynamics_optuna/image.sif")
    parser.add_argument("--code-revision")
    parser.add_argument(
        "--in-flight-action-id",
        action="append",
        default=[],
        help="current action already running elsewhere; exclude it and record a receipt",
    )
    args = parser.parse_args(argv)

    if (
        args.submission_policy_replay != "canonical"
        and args.regime == DEFAULT_REGIME
        and args.command not in ("record", "record-package", "record-safeguard")
    ):
        parser.error(
            "a legacy --submission-policy-replay is permitted only for "
            "ingesting already-submitted evidence of the locked campaign"
        )
    campaign = build_campaign(
        prior_root=args.prior_root,
        submission_policy_replay=str(args.submission_policy_replay),
        regime=str(args.regime),
        promotion_horizon=str(args.promotion_horizon),
        apm_arm=str(args.apm_arm),
    )
    campaign_id = campaign_id_for(
        str(args.regime), str(args.promotion_horizon), str(args.apm_arm)
    )
    default_seed_path = runs.REGIMES[str(args.regime)].seed_path
    ledger = _load_ledger(args.ledger)
    declared_policy = campaign.accuracy_policy.to_json_dict()
    if args.command == "reclassify":
        before = len(ledger.records)
        reclassified = ledger.reclassified(campaign.accuracy_policy)
        reclassified.write(args.ledger)
        counts: dict[str, int] = {}
        for record in reclassified.records:
            counts[record.status] = counts.get(record.status, 0) + 1
        print(
            json.dumps(
                {
                    "campaign_id": campaign_id,
                    "ledger": str(args.ledger),
                    "record_count": before,
                    "qualification_policy": declared_policy,
                    "status_counts": counts,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if ledger.records and not ledger.qualification_policy:
        raise ValueError(
            "legacy ledger has no qualification-policy binding; run reclassify first"
        )
    if ledger.qualification_policy and dict(ledger.qualification_policy) != declared_policy:
        raise ValueError(
            "ledger qualification policy differs from the canonical campaign policy"
        )
    if not ledger.qualification_policy:
        ledger = CampaignLedger(
            ledger.records,
            qualification_policy=declared_policy,
        )
    if args.command == "next":
        payload = _next_payload(
            campaign=campaign,
            ledger=ledger,
            output_root=args.output_root,
            search_profile=str(args.search_profile),
            campaign_id=campaign_id,
            seed_path=default_seed_path,
        )
        if args.action_dir is not None:
            _write_action_files(payload, args.action_dir)
        if args.worklist_json is not None:
            args.worklist_json.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.worklist_json.with_name(
                f".{args.worklist_json.name}.tmp"
            )
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(args.worklist_json)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "prepare-chtc":
        if args.package_dir is None:
            parser.error("prepare-chtc requires --package-dir")
        prepared = prepare_staged_chtc_campaign(
            campaign_id=campaign_id,
            campaign=campaign,
            ledger=ledger,
            package_dir=args.package_dir,
            seed_path=default_seed_path if args.seed_json is None else args.seed_json,
            numerics=NUMERICS_ID,
            gate=runs.MCLACHLAN_L2_GATE.gate_id,
            search_profile=str(args.search_profile),
            repo_root=Path(__file__).resolve().parents[3],
            image_path=str(args.image_path),
            code_revision=args.code_revision,
            in_flight_action_ids=tuple(args.in_flight_action_id),
        )
        print(
            json.dumps(
                {
                    "campaign_id": campaign_id,
                    "stage": campaign.stage(ledger),
                    "status": "PREPARED_NOT_SUBMITTED",
                    **prepared.to_json_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "record-package":
        if args.actions_jsonl is None or args.artifact_root is None:
            parser.error(
                "record-package requires --actions-jsonl and --artifact-root"
            )
        if args.seed_json is None:
            parser.error("record-package requires --seed-json")
        rows = _selected_action_rows(
            args.actions_jsonl,
            tuple(args.action_id),
        )
        if args.completed_actions_jsonl is not None:
            args.completed_actions_jsonl.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.completed_actions_jsonl.with_name(
                f".{args.completed_actions_jsonl.name}.tmp"
            )
            temporary.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            temporary.replace(args.completed_actions_jsonl)
        ledger, receipts = record_prepared_action_outputs(
            campaign=campaign,
            ledger=ledger,
            action_rows=rows,
            artifact_root=args.artifact_root,
            seed_path=args.seed_json,
            numerics=NUMERICS_ID,
            gate=runs.MCLACHLAN_L2_GATE.gate_id,
            search_profile=str(args.search_profile),
            reference_substeps=int(args.reference_substeps),
            reference_store_root=args.reference_store_root,
        )
        ledger.write(args.ledger)
        print(
            json.dumps(
                {
                    "campaign_id": campaign_id,
                    "ledger": str(args.ledger),
                    "ledger_record_count": len(ledger.records),
                    "next_stage": campaign.stage(ledger),
                    "artifact_root": str(args.artifact_root),
                    "receipts": list(receipts),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.action_json is None:
        parser.error(f"{args.command} requires --action-json")
    action_payload = json.loads(args.action_json.read_text(encoding="utf-8"))
    action = _resolved_action(campaign, ledger, action_payload)
    if args.command == "record-safeguard":
        ledger = record_safeguard_action(ledger=ledger, action=action)
    else:
        if args.run_json is None or args.seed_json is None:
            parser.error("record requires --run-json and --seed-json")
        command = campaign.build_run_command(
            action,
            seed_path=str(args.seed_json),
            output_json=str(args.run_json),
            numerics=NUMERICS_ID,
            gate=runs.MCLACHLAN_L2_GATE.gate_id,
            search_profile=str(args.search_profile),
        )
        ledger = record_run_outcome(
            campaign=campaign,
            ledger=ledger,
            action=action,
            command=command,
            run_path=args.run_json,
            seed_path=args.seed_json,
            reference_substeps=int(args.reference_substeps),
            reference_store_root=args.reference_store_root,
        )
    ledger.write(args.ledger)
    print(
        json.dumps(
            {
                "campaign_id": campaign_id,
                "ledger": str(args.ledger),
                "ledger_record_count": len(ledger.records),
                "next_stage": campaign.stage(ledger),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAMPAIGN_ID",
    "DEFAULT_APM_ARM",
    "DEFAULT_PROMOTION_HORIZON",
    "DEFAULT_REGIME",
    "REGIME_RESPONSE_SCALE_REGISTRY",
    "build_campaign",
    "campaign_id_for",
    "main",
]
