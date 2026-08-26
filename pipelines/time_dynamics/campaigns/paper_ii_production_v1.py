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
    SeedSpec,
    uniform_threshold_plan,
)
from pipelines.time_dynamics import paper_ii_runs as runs

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
DRIVE_IDS = ("fastweak",)

# Production horizon: t<=1 barely exposes accumulated error or staleness.
HORIZON_IDS = ("t5",)

METHOD_IDS = ("append_only", "exchange", "avqds")
CONTROLLER_IDS = ("state_motion_1e-2",)
THRESHOLD_PLAN = uniform_threshold_plan(
    method_ids=METHOD_IDS,
    controller_ids=CONTROLLER_IDS,
    drive_ids=DRIVE_IDS,
    thresholds=(1.0e-3,),
)

SPEC = CampaignSpec(
    campaign_id="paper_ii_production_v1",
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
        SPEC, "chtc/paper_ii_production_v1", max_runtime_seconds=86400
    )
    print(f"cells: {SPEC.cell_count()}")
    for name, path in written.to_json_dict().items():
        print(f"  {name:10s} {path}")
