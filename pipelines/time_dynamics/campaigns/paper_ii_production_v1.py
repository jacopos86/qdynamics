"""Paper II production matrix: cost-accuracy evidence for prune/exchange.

The claim is reduced circuit cost at matched or better accuracy for ansaetze
carrying redundant support, so every cell places one policy on the
cost-accuracy plane against comparators that share its physics.

Seeds span the redundancy axis, which is the axis the claim lives on:

* ``hh_fixedvqe_nph3`` - fixed-structure VQE ansatz, 616 coordinates. Redundant
  by construction of the ansatz family, not by arrangement, and the regime the
  claim targets.
* ``hh_snake_nph1`` - compact adaptive seed, 28 coordinates. The boundary case,
  included so the trade is quantified rather than avoided.
* ``hubbard_snake_nph1`` - purely fermionic, so removal cannot be read as a
  phonon-sector artifact.

Arms: append-only (our own ablation), exchange (the contribution), and adaptive
append (Yao et al.) at its published McLachlan-distance convention with no
per-step cap.
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
        artifact_json=(
            "chtc/paper_ii_exchange_hook_calibration_20260818/input/seed_intweak_nph3.json"
        ),
        family_key="hh", n_ph_max=3, regime="intermediate_weak",
        note="fixed-structure VQE ansatz, 616 coordinates; redundancy-carrying",
    ),
    SeedSpec(
        seed_id="hh_snake_nph1",
        artifact_json=f"{SEED_ROOT}/hh_snake_seed.json",
        family_key="hh", n_ph_max=1, regime="compact",
        note="adaptive seed, 28 coordinates, static error 7.0e-5; boundary case",
    ),
    SeedSpec(
        seed_id="hubbard_snake_nph1",
        artifact_json=f"{SEED_ROOT}/hubbard_snake_seed.json",
        family_key="hubbard", n_ph_max=1, regime="compact",
        note="purely fermionic control",
    ),
)

# Driven: a static Hamiltonian on a ground-state seed produces no dynamics, so
# the drive is what makes support maintenance meaningful at all.
DRIVES = (DriveSpec(drive_id="driven", enabled=True, amplitude=0.6, omega=3.0),)

# Production horizon: t<=1 barely exposes accumulated error or staleness.
HORIZONS = (HorizonSpec(horizon_id="t5", t_final=5.0, num_times=126),)

ARMS = (
    append_only_arm(),
    exchange_arm(2.0e-3),
    # Published convention (L^2 = 2(||b||^2 - Q)); the cut is absolute, and the
    # append rule runs uncapped as the source specifies.
    avqds_arm(1.0e-3),
)

SPEC = CampaignSpec(
    campaign_id="paper_ii_production_v1",
    seeds=SEEDS,
    drives=DRIVES,
    horizons=HORIZONS,
    arms=ARMS,
    output_root="raw_outputs",
)


if __name__ == "__main__":
    from pipelines.time_dynamics.campaign import write_chtc_package

    written = write_chtc_package(
        SPEC, "chtc/paper_ii_production_v1", max_runtime_seconds=86400
    )
    print(f"cells: {SPEC.cell_count()}")
    for name, path in written.items():
        print(f"  {name:10s} {path}")
