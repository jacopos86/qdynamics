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
    DriveSpec,
    HorizonSpec,
    SeedSpec,
    append_only_arm,
    avqds_arm,
    exchange_arm,
)

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
DRIVES = (
    DriveSpec("fastweak", True, 0.6, 3.0),
    DriveSpec("slowstrong", True, 1.2, 1.0),
    DriveSpec("weakslow", True, 0.3, 1.0),
    DriveSpec("strongfast", True, 2.4, 3.0),
    DriveSpec("midres", True, 1.2, 2.0),
    DriveSpec("fastfast", True, 0.6, 6.0),
)

HORIZONS = (HorizonSpec(horizon_id="t20", t_final=20.0, num_times=501),)

ARMS = (append_only_arm(), exchange_arm(2.0e-3), avqds_arm(1.0e-3))

SPEC = CampaignSpec(
    campaign_id="paper_ii_drive_sweep_v1",
    seeds=SEEDS,
    drives=DRIVES,
    horizons=HORIZONS,
    arms=ARMS,
    output_root="raw_outputs",
)


if __name__ == "__main__":
    from pipelines.time_dynamics.campaign import write_chtc_package

    written = write_chtc_package(
        SPEC, "chtc/paper_ii_drive_sweep_v1", max_runtime_seconds=86400
    )
    print(f"cells: {SPEC.cell_count()}  ({len(SEEDS)} seeds x {len(DRIVES)} drives x {len(ARMS)} arms)")
    for name, path in written.items():
        print(f"  {name:10s} {path}")
