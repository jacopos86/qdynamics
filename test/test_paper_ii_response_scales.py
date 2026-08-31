"""Integrity and physical-identity tests for Paper-II response scales."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from pipelines.time_dynamics import paper_ii_runs as runs
from pipelines.time_dynamics.campaigns.paper_ii_factorial_euler_v1 import (
    CONTROLLER_IDS,
    DRIVE_IDS,
    SEED,
)
from pipelines.time_dynamics.campaigns.paper_ii_staged_accuracy_v1 import (
    EXACT_RESPONSE_SCALES,
    EXACT_RESPONSE_SCALES_SHA256,
    _response_scales,
)
from pipelines.time_dynamics.response_scales import (
    RESPONSE_SCALE_INDEX_SCHEMA_V2,
    ResponseScaleIntegrityError,
    load_verified_response_scale_index,
    response_scale_cell_identity,
    response_scale_row,
)
from pipelines.time_dynamics.staged_campaign import (
    AccuracyMetrics,
    AccuracyPolicy,
    CellKey,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXACT_REFERENCE_ROOT = REPOSITORY_ROOT / "output/paper_ii_staged/exact_references"


def _loader(path: Path, *, expected_sha256: str):
    return load_verified_response_scale_index(
        path,
        expected_sidecar_sha256=expected_sha256,
        exact_reference_root=EXACT_REFERENCE_ROOT,
        seed_path=REPOSITORY_ROOT / SEED.artifact_json,
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


def test_baseline_sidecar_loads_as_full_physical_cell_index() -> None:
    index = _response_scales()
    assert index["schema"] == RESPONSE_SCALE_INDEX_SCHEMA_V2
    assert index["source"]["sha256"] == EXACT_RESPONSE_SCALES_SHA256

    identity = response_scale_cell_identity(
        index,
        regime="hh_snake_nph1",
        controller="state_motion_5e-5",
        drive="strongfast",
        horizon="t20",
    )
    assert identity == {
        "model": "hh",
        "family_key": "hubbard_holstein",
        "regime": "hh_snake_nph1",
        "seed_sha256": (
            "7a36a2da06bba447c6d65d95692c8a45bc3aa64933d02f1f641e2178f4ea8f91"
        ),
        "lattice_sites": 2,
        "n_ph_max": 1,
        "boundary": "open",
        "boson_encoding": "binary",
        "orbital_ordering": "blocked",
        "hopping": 1.0,
        "next_nearest_hopping": 0.0,
        "staggered_potential": 0.25,
        "phonon_frequency": 1.0,
        "U": 4.0,
        "g": 0.5,
        "nearest_neighbor_interaction": 0.0,
        "n_fermions": None,
        "initial_state_source": "adapt_vqe",
        "controller": "state_motion_5e-5",
        "drive": "strongfast",
        "drive_amplitude": 2.4,
        "drive_omega": 3.0,
        "drive_profile": {
            "family_key": "hh",
            "operator_label": "hh_spinful_onsite_density",
            "pattern": "staggered",
            "amplitude": 2.4,
            "omega": 3.0,
            "phase": 0.0,
            "start_time": 0.0,
            "time_scale": 1.0,
            "include_identity": False,
            "spatial_weights": [1.0, -1.0],
            "custom_weights": None,
            "time_sampling": "midpoint",
            "reference_method_name": "exponential_midpoint_magnus2_order2",
        },
        "horizon": "t20",
        "horizon_final_time": 20.0,
        "horizon_sample_count": 501,
        "horizon_time_step": 0.04,
    }
    row = response_scale_row(
        index,
        regime="hh_snake_nph1",
        controller="state_motion_5e-5",
        drive="strongfast",
        horizon="t20",
    )
    assert row["energy_rms_excursion"] == pytest.approx(2.50207998889209)
    assert row["cell_identity"] == identity


def test_accuracy_policy_uses_controller_aware_index_and_legacy_map() -> None:
    index = _response_scales()
    policy = AccuracyPolicy.response_normalized_matched_pair(
        response_scales=index
    )
    policy = AccuracyPolicy.from_json_dict(policy.to_json_dict())
    key = CellKey(
        "hh_snake_nph1", "weakslow", "avqds", "delta_theta_5e-5"
    )
    metrics = AccuracyMetrics(
        mean_abs_energy_error=1.1335122334496122e-3 * 0.05,
        mean_abs_doublon_error=2.1686891125251777e-3 * 0.05,
        mean_max_abs_site_occupation_error=1.2561874240715799e-2 * 0.05,
        mean_infidelity=1.0e-5,
    )
    assert policy.response_components(metrics, key=key, horizon="t5") == pytest.approx(
        {
            "mean_abs_energy_error": 1.0,
            "mean_abs_doublon_error": 1.0,
            "mean_max_abs_site_occupation_error": 1.0,
            "mean_infidelity": 0.1,
        }
    )

    legacy = AccuracyPolicy.response_normalized_matched_pair(
        response_scales={
            "weakslow": {
                horizon: {
                    "energy_rms_excursion": 0.1,
                    "doublon_rms_excursion": 0.1,
                    "max_site_occupation_rms_excursion": 0.1,
                }
                for horizon in ("t5", "t20")
            }
        }
    )
    assert legacy.response_components(
        AccuracyMetrics(0.005, 0.005, 0.005, 1.0e-5),
        key=key,
        horizon="t20",
    )["mean_abs_energy_error"] == pytest.approx(1.0)


def test_sidecar_digest_and_exact_manifest_bindings_fail_closed(
    tmp_path: Path,
) -> None:
    payload = json.loads(EXACT_RESPONSE_SCALES.read_text(encoding="utf-8"))
    payload["drives"]["weakslow"]["t5"]["energy_rms_excursion"] *= 2.0
    tampered = tmp_path / "response_scales.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ResponseScaleIntegrityError, match="sidecar SHA-256"):
        _loader(tampered, expected_sha256=EXACT_RESPONSE_SCALES_SHA256)

    payload = json.loads(EXACT_RESPONSE_SCALES.read_text(encoding="utf-8"))
    payload["drives"]["weakslow"]["t5"][
        "exact_reference_manifest_sha256"
    ] = "0" * 64
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    tampered_sha256 = hashlib.sha256(tampered.read_bytes()).hexdigest()
    with pytest.raises(ResponseScaleIntegrityError, match="manifest SHA-256"):
        _loader(tampered, expected_sha256=tampered_sha256)


def test_sidecar_values_and_drive_reference_are_recomputed_not_trusted(
    tmp_path: Path,
) -> None:
    payload = json.loads(EXACT_RESPONSE_SCALES.read_text(encoding="utf-8"))
    payload["drives"]["weakslow"]["t5"]["energy_rms_excursion"] *= 2.0
    tampered = tmp_path / "wrong_scale.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ResponseScaleIntegrityError, match="disagrees with exact data"):
        _loader(
            tampered,
            expected_sha256=hashlib.sha256(tampered.read_bytes()).hexdigest(),
        )

    payload = json.loads(EXACT_RESPONSE_SCALES.read_text(encoding="utf-8"))
    payload["drives"]["weakslow"] = copy.deepcopy(payload["drives"]["fastfast"])
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ResponseScaleIntegrityError, match="drive mismatch"):
        _loader(
            tampered,
            expected_sha256=hashlib.sha256(tampered.read_bytes()).hexdigest(),
        )


def test_full_identity_prevents_cross_regime_or_controller_reuse() -> None:
    index = _response_scales()
    with pytest.raises(ResponseScaleIntegrityError, match="no response scale"):
        response_scale_row(
            index,
            regime="hh_snake_nph3",
            controller="state_motion_5e-5",
            drive="weakslow",
            horizon="t20",
        )

    corrupted = copy.deepcopy(index)
    corrupted["regimes"]["hh_snake_nph1"]["controllers"][
        "state_motion_5e-5"
    ]["drives"]["weakslow"]["t20"]["cell_identity"][
        "horizon_sample_count"
    ] = 500
    with pytest.raises(ResponseScaleIntegrityError, match="cell identity mismatch"):
        AccuracyPolicy.response_normalized_matched_pair(response_scales=corrupted)
    with pytest.raises(ResponseScaleIntegrityError, match="no response scale"):
        response_scale_row(
            index,
            regime="hh_snake_nph1",
            controller="unregistered_controller",
            drive="weakslow",
            horizon="t20",
        )
