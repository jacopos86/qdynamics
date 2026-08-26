"""Drive sweep on the Holstein and Hubbard seeds already in hand.

The production matrix establishes the removal effect across Hamiltonian
families at two drive conditions. This sweep asks the narrower question the
main result rests on: does the effect survive across drive amplitude and
frequency on the two families the paper centers, rather than holding at one
operating point.

Six drive conditions crossed with the three Holstein and Hubbard seeds and the
three structural policies. The regime axis is covered separately once the
six-regime seed build returns; this varies the drive at fixed physics.
"""

from __future__ import annotations

from pipelines.time_dynamics.campaign import (
    CampaignSpec,
    SeedSpec,
    uniform_threshold_plan,
)
from pipelines.time_dynamics import paper_ii_runs as runs

SEED_ROOT = "chtc/generic_time_dynamics_table/input/seed_artifacts_paper_ii_seed_tracks_v2"

SEEDS = (
    SeedSpec(
        seed_id="hh_fixedvqe_nph3",
        artifact_json="chtc/paper_ii_production_v1/input/seeds/hh_fixedvqe_nph3.json",
        family_key="hh", n_ph_max=3, regime="intermediate_weak",
        note="fixed-structure VQE ansatz, 620 coordinates",
    ),
    SeedSpec(
        seed_id="hh_snake_nph1",
        artifact_json=f"{SEED_ROOT}/hh_snake_seed.json",
        family_key="hh", n_ph_max=1, regime="hh_L2",
        note="adaptive Holstein seed",
    ),
    SeedSpec(
        seed_id="hubbard_snake_nph1",
        artifact_json=f"{SEED_ROOT}/hubbard_snake_seed.json",
        family_key="hubbard", n_ph_max=1, regime="hubbard_L2",
        note="purely fermionic",
    ),
)

# Amplitude and frequency are varied separately so a dependence on either can
# be read off rather than inferred from a diagonal sweep. The first two repeat
# the production conditions so this sweep and that matrix share anchors.
DRIVE_IDS = (
    "fastweak", "slowstrong", "weakslow", "strongfast", "midres", "fastfast",
)

HORIZON_IDS = ("t20",)

METHOD_IDS = ("append_only", "exchange", "avqds")
CONTROLLER_IDS = ("state_motion_1e-2",)
THRESHOLD_PLAN = uniform_threshold_plan(
    method_ids=METHOD_IDS,
    controller_ids=CONTROLLER_IDS,
    drive_ids=DRIVE_IDS,
    thresholds=(1.0e-3,),
)

SPEC = CampaignSpec(
    campaign_id="paper_ii_drive_sweep_v1",
    seeds=SEEDS,
    method_ids=METHOD_IDS,
    controller_ids=CONTROLLER_IDS,
    drive_ids=DRIVE_IDS,
    horizon_ids=HORIZON_IDS,
    threshold_plan=THRESHOLD_PLAN,
    numerics_id=runs.SHARED_NUMERICS.numerics_id,
    output_root="raw_outputs",
)


if __name__ == "__main__":
    from pipelines.time_dynamics.campaign import write_chtc_package

    written = write_chtc_package(
        SPEC, "chtc/paper_ii_drive_sweep_v1", max_runtime_seconds=86400
    )
    print(f"cells: {SPEC.cell_count()}  ({len(SEEDS)} seeds x {len(DRIVE_IDS)} drives x {len(METHOD_IDS)} methods)")
    for name, path in written.to_json_dict().items():
        print(f"  {name:10s} {path}")
