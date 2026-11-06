"""Per-regime exact-response-scale sidecars reproduce the locked store identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipelines.time_dynamics import paper_ii_runs as runs
from pipelines.time_dynamics.diagnostics import build_regime_response_scales as scales
from pipelines.time_dynamics.diagnostics.exact_reference_store import (
    PROPAGATOR_SCHEMA,
    SCHEMA as STORE_SCHEMA,
    _canonical_bytes,
)
from pipelines.time_dynamics.response_scales import response_scale_row

REPO = Path(__file__).resolve().parents[1]
CALIBRATION_SEED = REPO / runs.REGIMES["hh_snake_nph1"].seed_path
V1_SIDECAR = REPO / "pipelines/time_dynamics/campaigns/paper_ii_exact_response_scales_v1.json"


def _store_key(seed: Path, profile: dict, times: list[float], substeps: int) -> str:
    identity = {
        "schema": STORE_SCHEMA,
        "propagator_schema": PROPAGATOR_SCHEMA,
        "seed_sha256": hashlib.sha256(seed.read_bytes()).hexdigest(),
        "drive_profile": profile,
        "times": times,
        "substeps": int(substeps),
    }
    return hashlib.sha256(_canonical_bytes(identity)).hexdigest()


def test_locked_grid_is_the_501_point_0p04_grid() -> None:
    grid = scales.locked_grid()
    assert len(grid) == 501 and grid[0] == 0.0 and grid[-1] == pytest.approx(20.0)
    assert grid[125] == pytest.approx(5.0)


def test_runner_drive_profile_reproduces_every_locked_reference_key() -> None:
    """The producer's store keys equal the v1 sidecar's keys for the calibration seed."""

    sidecar = json.loads(V1_SIDECAR.read_text())
    grid = scales.locked_grid()
    for drive_id, horizons in sidecar["drives"].items():
        profile = scales.drive_profile_for(CALIBRATION_SEED, drive_id)
        assert profile["A"] == runs.DRIVES[drive_id].amplitude
        assert profile["omega"] == runs.DRIVES[drive_id].omega
        assert (profile["tbar"], profile["phi"], profile["t0"]) == (1.0, 0.0, 0.0)
        assert profile["pattern"] == "staggered" and profile["spatial_weights"] == [1.0, -1.0]
        assert _store_key(CALIBRATION_SEED, profile, grid, 40) == horizons["t20"]["exact_reference_key"]
        assert horizons["t5"]["exact_reference_key"] == horizons["t20"]["exact_reference_key"]


def test_sidecar_builds_and_round_trips_through_the_campaign_loader(tmp_path: Path) -> None:
    root = tmp_path / "output" / "paper_ii_regime_survey_hh_snake_nph1" / "exact_references"
    payload = scales.build_sidecar(
        regime="hh_snake_nph1",
        seed_path=CALIBRATION_SEED,
        exact_reference_root=root,
        substeps=1,
        repository_root=tmp_path,
    )
    assert payload["schema"] == scales.SIDECAR_SCHEMA_V1
    assert set(payload["drives"]) == set(scales.DEFAULT_DRIVES)
    sidecar_path = tmp_path / "paper_ii_exact_response_scales_hh_snake_nph1.json"
    sha = scales.write_sidecar(payload, sidecar_path)
    index = scales.verify_sidecar(
        sidecar_path=sidecar_path,
        sidecar_sha256=sha,
        exact_reference_root=root,
        seed_path=CALIBRATION_SEED,
        regime="hh_snake_nph1",
    )
    row = response_scale_row(
        index, regime="hh_snake_nph1", controller="state_motion_5e-5",
        drive="fastweak", horizon="t20",
    )
    assert row["energy_rms_excursion"] > 0.0 and row["doublon_rms_excursion"] > 0.0
    assert row["max_site_occupation_rms_excursion"] > 0.0
    assert row["exact_reference_key"] == payload["drives"]["fastweak"]["t20"]["exact_reference_key"]
    # Every drive materialized exactly one reference directory in the store.
    assert len([p for p in root.iterdir() if p.is_dir()]) == len(scales.DEFAULT_DRIVES)
    registry = scales.update_registry(
        tmp_path / "registry.json",
        regime="hh_snake_nph1", sidecar_path=str(sidecar_path), sidecar_sha256=sha,
        seed_path=str(CALIBRATION_SEED), seed_sha256=payload["seed"]["sha256"],
        exact_reference_root=str(root),
    )
    assert registry["regimes"]["hh_snake_nph1"]["sha256"] == sha
    # Tampering with a sidecar value fails closed in the loader.
    tampered = json.loads(sidecar_path.read_text())
    tampered["drives"]["fastweak"]["t20"]["energy_rms_excursion"] *= 1.01
    sidecar_path.write_text(json.dumps(tampered, sort_keys=True))
    with pytest.raises(Exception, match="disagrees with exact data|SHA-256 mismatch"):
        scales.verify_sidecar(
            sidecar_path=sidecar_path, sidecar_sha256=sha,
            exact_reference_root=root, seed_path=CALIBRATION_SEED, regime="hh_snake_nph1",
        )
