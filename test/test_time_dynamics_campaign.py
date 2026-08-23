"""Campaign specification: matrix expansion, provenance, and invariants."""

from __future__ import annotations

import json

import pytest

from pipelines.time_dynamics.campaign import (
    CAMPAIGN_SCHEMA_V1,
    CampaignSpec,
    DriveSpec,
    HorizonSpec,
    SeedSpec,
    append_only_arm,
    avqds_arm,
    exchange_arm,
    write_campaign_manifest,
)


def _seed(seed_id="hh_nph1", nph=1):
    return SeedSpec(
        seed_id=seed_id, artifact_json=f"seeds/{seed_id}.json",
        family_key="hh", n_ph_max=nph, regime="intermediate_weak",
    )


def test_binary_aligned_cutoffs_are_enforced() -> None:
    for good in (1, 3, 7):
        _seed(nph=good)
    for bad in (0, 2, 4, 5, 6, 8):
        with pytest.raises(ValueError, match="binary phonon register"):
            _seed(nph=bad)


def test_matrix_expands_as_the_full_product() -> None:
    spec = CampaignSpec(
        campaign_id="c1",
        seeds=(_seed("a"), _seed("b", nph=3)),
        drives=(DriveSpec(), DriveSpec("driven", True, 0.6, 3.0)),
        horizons=(HorizonSpec("short", 1.0, 26), HorizonSpec("long", 5.0, 101)),
        arms=(exchange_arm(), append_only_arm(), avqds_arm(1.0e-5)),
    )
    assert spec.cell_count() == 2 * 2 * 2 * 3
    cells = list(spec.cells())
    assert len(cells) == spec.cell_count()
    assert len({c.cell_id for c in cells}) == spec.cell_count()


def test_cell_argv_carries_numerics_guards_drive_and_arm() -> None:
    spec = CampaignSpec(
        campaign_id="c2", seeds=(_seed(),),
        drives=(DriveSpec("driven", True, 0.6, 3.0),),
        horizons=(HorizonSpec("short", 1.0, 26),),
        arms=(exchange_arm(2.0e-3),),
    )
    argv = next(iter(spec.cells())).runner_argv()
    joined = " ".join(argv)
    # canonical numerics
    assert "--integrator euler" in joined
    assert "--solve-repair" in argv
    assert "--solve-repair-profile minimal" in joined
    # guards
    assert "--max-joint-patch-evaluations 50000" in joined
    assert "--max-certification-attempts-per-deletion-branch 2" in joined
    # physics and policy
    assert "--enable-drive" in argv
    assert "--prune-target-policy all_active" in joined
    assert "--prune-ray-distance-tol 0.002" in joined


def test_comparator_arm_is_flagged_and_uses_avqds_policy() -> None:
    arm = avqds_arm(1.0e-5, max_appends=2)
    assert arm.is_comparator
    assert "--dynamics-policy" in arm.flags and "avqds" in arm.flags


def test_manifest_records_every_cell_with_seed_binding(tmp_path) -> None:
    spec = CampaignSpec(
        campaign_id="c3", seeds=(_seed(), _seed("hh_nph3", 3)),
        drives=(DriveSpec(),), horizons=(HorizonSpec("short", 1.0, 26),),
        arms=(exchange_arm(), append_only_arm()),
    )
    path = write_campaign_manifest(spec, tmp_path / "manifest.json")
    payload = json.loads(path.read_text())
    assert payload["schema"] == CAMPAIGN_SCHEMA_V1
    assert payload["cell_count"] == 4 and len(payload["cells"]) == 4
    for cell in payload["cells"]:
        assert cell["seed"]["n_ph_max"] in (1, 3, 7)
        assert cell["seed"]["sha256"]  # 'missing' when the artifact is absent
        assert cell["runner_argv"]
        assert cell["arm"]["arm_id"]


def test_output_dirs_are_unique_per_cell() -> None:
    spec = CampaignSpec(
        campaign_id="c4", seeds=(_seed(),), drives=(DriveSpec(),),
        horizons=(HorizonSpec("short", 1.0, 26), HorizonSpec("long", 5.0, 101)),
        arms=(exchange_arm(2.0e-3), exchange_arm(5.0e-4)),
    )
    dirs = [c.output_dir for c in spec.cells()]
    assert len(set(dirs)) == len(dirs) == 4
