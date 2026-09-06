"""The threshold-bundle tool reproduces the committed one-cell pilot locks and yields eleven-cell locks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.time_dynamics.campaigns import (
    paper_ii_final_selector_comparison_v1 as selector,
    paper_ii_permission_threshold_bundle as bundle,
)


def test_token_and_campaign_ids():
    assert bundle.token_for("5e-3") == "5em03"
    assert bundle.token_for("2e-2") == "2em02"
    assert bundle.token_for("1e-1") == "1em01"
    assert bundle.campaign_id_for("5e-3", "strongfast_dtheta") == "paper_ii_permission_threshold_5em03_v1"
    assert bundle.campaign_id_for("5e-3", "eleven") == "paper_ii_corrected_exchange_tau_5em03_v1"
    with pytest.raises(ValueError):
        bundle.token_for("0.005")


def test_one_cell_lock_matches_the_committed_pilot_lock(tmp_path: Path):
    out = bundle.write_lock(literal="5e-3", scope="strongfast_dtheta", output=tmp_path / "lock.json", created="2026-09-06")
    committed = Path(selector.__file__).with_name("paper_ii_permission_threshold_5em03_cells_v1.json")
    a, b = json.loads(out.read_text()), json.loads(committed.read_text())
    for key in ("campaign_id", "lock_id", "cells", "grid", "exclusions", "registry_bindings", "selector_routes", "seed", "parent_matched_cell_lock_sha256"):
        assert a[key] == b[key], key


def test_eleven_cell_lock_loads_and_declares_the_literal(tmp_path: Path):
    out = bundle.write_lock(literal="1e-2", scope="eleven", output=tmp_path / "lock.json")
    lock = selector.load_lock(out, **selector.lock_parameters(out))
    assert len(lock.cells) == 11 and len(lock.exclusions) == 7  # 6 drives x 3 controllers
    assert lock.permission_ray_distance_max == "1e-2"
    assert lock.campaign_id == "paper_ii_corrected_exchange_tau_1em02_v1"


def test_cells_scope_with_cut_overrides_and_no_threshold(tmp_path: Path):
    out = bundle.write_lock(
        literal=None, scope="cells", cell_ids=["fastweak__delta_theta_5e-5", "slowstrong__delta_theta_5e-5"],
        apm_cut_overrides={"fastweak__delta_theta_5e-5": 1e-7, "slowstrong__delta_theta_5e-5": 1e-6},
        campaign_id="paper_ii_corrected_exchange_next_rung_v1", purpose="next rung", output=tmp_path / "lock.json",
    )
    lock = selector.load_lock(out, **selector.lock_parameters(out))
    assert lock.permission_ray_distance_max is None
    assert {c.cell_id: c.apm_cut for c in lock.cells} == {
        "fastweak__delta_theta_5e-5": 1e-7, "slowstrong__delta_theta_5e-5": 1e-6,
    }
    assert lock.campaign_id == "paper_ii_corrected_exchange_next_rung_v1"
    assert json.loads(out.read_text())["registry_bindings"]["extra_flags"] == ["--record-statevector"]
