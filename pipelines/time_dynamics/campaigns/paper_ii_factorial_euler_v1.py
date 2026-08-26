"""Prepare the Paper-II APM/AVQDS x step-controller factorial campaign.

This module declares the scientific matrix only.  Campaign expansion, runner
resolution, seed staging, argv validation, and CHTC packaging live behind
``prepare_chtc_campaign``.

Preparation never submits jobs.

Examples
--------

Prepare the six-cell short-horizon gate::

    python -m pipelines.time_dynamics.campaigns.paper_ii_factorial_euler_v1 \
        --mode smoke

Prepare the prior-informed production worklist after the smoke gate passes::

    python -m pipelines.time_dynamics.campaigns.paper_ii_factorial_euler_v1 \
        --mode production --prior-root output/frontier
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipelines.time_dynamics import paper_ii_runs as runs
from pipelines.time_dynamics.campaign import (
    CampaignSpec,
    SeedSpec,
    plan_thresholds_from_prior,
    prepare_chtc_campaign,
    uniform_threshold_plan,
)

METHOD_IDS = ("exchange", "avqds")
CONTROLLER_IDS = (
    "state_motion_1e-2",
    "delta_theta_5e-3",
    "state_motion_1e-2_plus_parameter_5e-3",
)
DRIVE_IDS = (
    "fastweak",
    "slowstrong",
    "weakslow",
    "strongfast",
    "midres",
    "fastfast",
)
NUMERICS_ID = runs.EULER_RIDGE1E6_NUMERICS.numerics_id
TARGET_MEAN_ABS_ENERGY_ERROR = 1.0e-4
SMOKE_DRIVE = "strongfast"
SMOKE_THRESHOLD = 3.0e-6

SEED = SeedSpec(
    seed_id="hh_snake_nph1",
    artifact_json=runs.REGIMES["hh_snake_nph1"].seed_path,
    family_key="hubbard_holstein",
    n_ph_max=1,
    regime="calibration",
    note="Current Paper-II calibration seed; one hash is shared by all cells.",
)


def build_spec(*, mode: str, prior_root: str | Path) -> CampaignSpec:
    if mode == "smoke":
        drives = (SMOKE_DRIVE,)
        plan = uniform_threshold_plan(
            method_ids=METHOD_IDS,
            controller_ids=CONTROLLER_IDS,
            drive_ids=drives,
            thresholds=(SMOKE_THRESHOLD,),
            target_mean_abs_energy_error=TARGET_MEAN_ABS_ENERGY_ERROR,
            rationale=(
                "short-horizon execution gate at the prior strongfast target "
                "neighborhood; not paper evidence"
            ),
        )
        return CampaignSpec(
            campaign_id="paper_ii_factorial_euler_smoke_20260824",
            seeds=(SEED,),
            method_ids=METHOD_IDS,
            controller_ids=CONTROLLER_IDS,
            drive_ids=drives,
            horizon_ids=("smoke",),
            threshold_plan=plan,
            numerics_id=NUMERICS_ID,
        )
    if mode != "production":
        raise ValueError(f"unknown mode {mode!r}")
    plan = plan_thresholds_from_prior(
        prior_root=prior_root,
        method_ids=METHOD_IDS,
        controller_ids=CONTROLLER_IDS,
        drive_ids=DRIVE_IDS,
        target_mean_abs_energy_error=TARGET_MEAN_ABS_ENERGY_ERROR,
    )
    return CampaignSpec(
        campaign_id="paper_ii_factorial_euler_t10_20260824",
        seeds=(SEED,),
        method_ids=METHOD_IDS,
        controller_ids=CONTROLLER_IDS,
        drive_ids=DRIVE_IDS,
        horizon_ids=("t10",),
        threshold_plan=plan,
        numerics_id=NUMERICS_ID,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("smoke", "production"), default="smoke")
    parser.add_argument("--prior-root", default="output/frontier")
    parser.add_argument("--package-dir", default=None)
    args = parser.parse_args(argv)

    spec = build_spec(mode=str(args.mode), prior_root=str(args.prior_root))
    package_dir = (
        Path(args.package_dir)
        if args.package_dir is not None
        else Path("chtc") / spec.campaign_id
    )
    prepared = prepare_chtc_campaign(spec, package_dir)
    print(
        json.dumps(
            {
                "campaign_id": spec.campaign_id,
                "mode": str(args.mode),
                "cell_count": spec.cell_count(),
                "status": "PREPARED_NOT_SUBMITTED",
                **prepared.to_json_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONTROLLER_IDS",
    "DRIVE_IDS",
    "METHOD_IDS",
    "NUMERICS_ID",
    "SEED",
    "SMOKE_DRIVE",
    "SMOKE_THRESHOLD",
    "TARGET_MEAN_ABS_ENERGY_ERROR",
    "build_spec",
    "main",
]
