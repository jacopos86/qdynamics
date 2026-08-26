"""Paper II wide production matrix: support removal across families and drives.

One batch is a pilot; a claim about support removal needs it to hold across
Hamiltonian families, seed construction tracks, and drive conditions. This
matrix crosses every seed in the Paper-II seed ledger that satisfies the
binary-aligned phonon-cutoff policy with two drive conditions and the three
structural policies, at a horizon long enough for support dynamics to develop.

The redundant fixed-structure VQE seed is included alongside the ledger seeds
because it is where the claim is sharpest; the ledger seeds establish that the
effect is not a property of that one ansatz.
"""

from __future__ import annotations

import json
import os

from pipelines.time_dynamics.campaign import (
    CampaignSpec,
    SeedSpec,
    uniform_threshold_plan,
)
from pipelines.time_dynamics import paper_ii_runs as runs

LEDGER = "chtc/generic_time_dynamics_table/input/paper_ii_seed_tracks_seed_ledger_v2.json"


def _seed_loads(path: str) -> bool:
    """Can the dynamics route actually start from this artifact?

    The ledger contains seeds from two construction tracks, and the
    position-geometric track serializes a legacy parameterization the AP state
    builder cannot reconstruct. Probing here keeps an unusable seed out of the
    matrix instead of turning it into a failed cluster job.
    """

    try:
        from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input
        from pipelines.time_dynamics.ap_mclachlan.state import (
            state_from_scaffold_runtime_input,
        )

        state_from_scaffold_runtime_input(load_scaffold_runtime_input(path))
        return True
    except Exception:
        return False


def _ledger_seeds() -> tuple[SeedSpec, ...]:
    """Every ledger seed the route can start from, at an allowed cutoff.

    ``SeedSpec`` rejects cutoffs that do not fill the binary phonon register
    (one spin-boson seed at nph=2), and ``_seed_loads`` rejects artifacts the
    state builder cannot reconstruct; both are skipped rather than failing the
    whole matrix.
    """

    with open(LEDGER, encoding="utf-8") as handle:
        ledger = json.load(handle)
    out: list[SeedSpec] = []
    for key, entry in sorted(ledger.items()):
        family, case, nph_field, track = key.split("|")
        nph = int(nph_field.split("=")[1])
        if nph not in (1, 3, 7):
            continue
        path = entry.get("normalized_seed_artifact_json")
        if not path or not os.path.exists(path) or not _seed_loads(path):
            continue
        out.append(
            SeedSpec(
                seed_id=f"{family}_{track.split('=')[1]}_nph{nph}",
                artifact_json=path,
                family_key=family,
                n_ph_max=nph,
                regime=case.split("=")[1],
                note=f"ledger seed, static error {entry['abs_delta_e']:.1e}",
            )
        )
    return tuple(out)


REDUNDANT = SeedSpec(
    seed_id="hh_fixedvqe_nph3",
    artifact_json="chtc/paper_ii_production_v1/input/seeds/hh_fixedvqe_nph3.json",
    family_key="hh", n_ph_max=3, regime="intermediate_weak",
    note="fixed-structure VQE ansatz, 620 coordinates; redundancy-carrying",
)

SEEDS = (REDUNDANT,) + _ledger_seeds()

# Two drive conditions: a fast weak drive and a slow strong one. A claim that
# holds under only one drive is a claim about that drive.
DRIVE_IDS = ("fastweak", "slowstrong")

# t=20 at the dt=0.04 grid the shorter runs used, so accuracy is comparable
# across horizons rather than confounded by step size.
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
    campaign_id="paper_ii_production_v2",
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
        SPEC, "chtc/paper_ii_production_v2", max_runtime_seconds=86400
    )
    print(f"seeds: {len(SEEDS)}  drives: {len(DRIVE_IDS)}  methods: {len(METHOD_IDS)}")
    print(f"cells: {SPEC.cell_count()}")
    for name, path in written.to_json_dict().items():
        print(f"  {name:10s} {path}")
