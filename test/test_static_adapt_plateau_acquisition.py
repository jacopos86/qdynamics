from __future__ import annotations

from pathlib import Path
import inspect
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.static_adapt.plateau_acquisition import (
    PLATEAU_ACQUISITION_SCORE_LOG_VOLUME_V1,
    PLATEAU_ACQUISITION_MODE_NOVELTY_COST_V1,
    PLATEAU_DUPLICATE_POLICY_ALLOW_EXACT_POSITION_REPLAY,
    PLATEAU_DUPLICATE_POLICY_BLOCK_EXACT_POSITION_V1,
    PlateauAcquisitionError,
    PlateauAcquisitionState,
    admit_failed_unlock_dormant,
    candidate_key_from_parts,
    candidate_key_from_record,
    duplicate_status,
    failed_family_backoff_status,
    normalize_plateau_acquisition_config,
    plateau_state_from_payload,
    remap_logical_indices_after_insertion,
)
from pipelines.static_adapt.route_c_plateau import (
    ROUTE_C_PLATEAU_HISTORICAL_NOVELTY_EPS,
    route_c_plateau_active_dormant_novelty_payload,
    run_route_c_sp_qngd_trial_optimizer,
)


def test_historical_route_c_ridge_is_local_to_the_quarantined_route() -> None:
    assert ROUTE_C_PLATEAU_HISTORICAL_NOVELTY_EPS == pytest.approx(1.0e-6)
    source = inspect.getsource(route_c_plateau_active_dormant_novelty_payload)
    assert "score_config.novelty_eps" not in source
    assert "ROUTE_C_PLATEAU_HISTORICAL_NOVELTY_EPS" in source


def test_sp_qngd_energy_observer_reconciles_exactly_with_nfev(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_energy_via_one_apply(
        state: np.ndarray,
        _compiled: object,
    ) -> tuple[float, np.ndarray]:
        psi = np.asarray(state, dtype=complex).reshape(-1)
        hpsi = np.asarray([0.0j, psi[1]], dtype=complex)
        return float(np.real(np.vdot(psi, hpsi))), hpsi

    monkeypatch.setattr(
        "pipelines.static_adapt.route_c_plateau.energy_via_one_apply",
        _fake_energy_via_one_apply,
    )
    observed: list[tuple[np.ndarray, dict[str, object]]] = []

    result = run_route_c_sp_qngd_trial_optimizer(
        x0=np.asarray([0.4], dtype=float),
        theta_template=np.asarray([0.4], dtype=float),
        active_runtime_indices=[0],
        maxiter_value=2,
        trial_qngd_maxiter=2,
        metric_floor=1.0e-8,
        h_compiled=object(),  # type: ignore[arg-type]
        prepare_state_for_theta=lambda theta: np.asarray(
            [np.cos(float(theta[0])), np.sin(float(theta[0]))],
            dtype=complex,
        ),
        canonicalize_runtime_theta=lambda theta: np.asarray(
            theta,
            dtype=float,
        ),
        oracle_inner_objective_enabled=False,
        analytic_noise_enabled=False,
        energy_evaluation_observer=lambda state, metadata: observed.append(
            (
                np.asarray(state, dtype=complex).copy(),
                dict(metadata),
            )
        ),
    )

    assert len(observed) == int(result.nfev)
    assert int(result.qngd_info["qngd_energy_observer_call_count"]) == int(
        result.nfev
    )
    assert [int(metadata["evaluation_index"]) for _, metadata in observed] == list(
        range(1, int(result.nfev) + 1)
    )
    assert observed[0][1]["evaluation_stage"] == "initial"
    assert all(
        metadata["evaluation_stage"] in {"initial", "line_search"}
        for _, metadata in observed
    )
    for state, metadata in observed:
        expected_energy = float(abs(state[1]) ** 2)
        assert float(metadata["energy"]) == pytest.approx(expected_energy)


def test_plateau_config_validates_mode_margin_and_duplicate_policy() -> None:
    cfg = normalize_plateau_acquisition_config(
        mode="plateau",
        unlock_margin="1e-7",
        duplicate_policy="block_exact_position",
    )

    assert cfg.enabled is True
    assert cfg.mode == PLATEAU_ACQUISITION_MODE_NOVELTY_COST_V1
    assert cfg.acquisition_score == PLATEAU_ACQUISITION_SCORE_LOG_VOLUME_V1
    assert cfg.unlock_margin == pytest.approx(1e-7)
    assert cfg.duplicate_policy == PLATEAU_DUPLICATE_POLICY_BLOCK_EXACT_POSITION_V1
    assert cfg.lambda_vol == pytest.approx(1e-8)
    assert cfg.sigma_min == pytest.approx(0.0)
    assert cfg.nu_min == pytest.approx(0.0)
    assert cfg.volume_min == pytest.approx(0.0)
    assert cfg.failed_family_patience == 0
    assert cfg.trial_optimizer == "inherit"
    assert cfg.trial_qngd_maxiter == 64
    assert cfg.as_dict()["score_formula"] == "log(1 + sigma_perp_lambda / lambda_vol) / (1 + K3)"
    assert cfg.as_dict()["trial_optimizer"] == "inherit"

    sp_qngd_cfg = normalize_plateau_acquisition_config(
        trial_optimizer="qngd",
        trial_qngd_maxiter="12",
    )
    assert sp_qngd_cfg.trial_optimizer == "sp_qngd"
    assert sp_qngd_cfg.trial_qngd_maxiter == 12

    with pytest.raises(PlateauAcquisitionError, match="unlock_margin"):
        normalize_plateau_acquisition_config(unlock_margin=-1.0)
    with pytest.raises(PlateauAcquisitionError, match="acquisition_score"):
        normalize_plateau_acquisition_config(acquisition_score="unknown")
    with pytest.raises(PlateauAcquisitionError, match="duplicate_policy"):
        normalize_plateau_acquisition_config(duplicate_policy="unknown")
    with pytest.raises(PlateauAcquisitionError, match="lambda_vol"):
        normalize_plateau_acquisition_config(lambda_vol=0.0)
    with pytest.raises(PlateauAcquisitionError, match="failed_family_patience"):
        normalize_plateau_acquisition_config(failed_family_patience=-1)
    with pytest.raises(PlateauAcquisitionError, match="trial_optimizer"):
        normalize_plateau_acquisition_config(trial_optimizer="unknown")
    with pytest.raises(PlateauAcquisitionError, match="trial_qngd_maxiter"):
        normalize_plateau_acquisition_config(trial_qngd_maxiter=-1)


def test_candidate_key_requires_identity_and_integer_position() -> None:
    with pytest.raises(PlateauAcquisitionError, match="candidate identity"):
        candidate_key_from_parts(candidate_identity="", position_id=0)
    with pytest.raises(PlateauAcquisitionError, match="position_id must be an integer"):
        candidate_key_from_parts(candidate_identity="g0", position_id=None)


def test_candidate_duplicate_key_blocks_same_position_but_allows_other_position() -> None:
    first_key = candidate_key_from_parts(candidate_identity="G0", position_id=1)
    state = admit_failed_unlock_dormant(
        PlateauAcquisitionState(),
        candidate_key=first_key,
        insertion_position=1,
        candidate_label="G0@1",
        generator_id="G0",
    )

    same_position = candidate_key_from_parts(candidate_identity="G0", position_id=1)
    other_position = candidate_key_from_parts(candidate_identity="G0", position_id=2)

    same_status = duplicate_status(state, same_position)
    other_status = duplicate_status(state, other_position)
    assert same_status["duplicate"] is True
    assert same_status["blocked"] is True
    assert other_status["duplicate"] is False
    assert other_status["blocked"] is False

    replay_status = duplicate_status(
        state,
        same_position,
        duplicate_policy=PLATEAU_DUPLICATE_POLICY_ALLOW_EXACT_POSITION_REPLAY,
    )
    assert replay_status["duplicate"] is True
    assert replay_status["blocked"] is False


def test_admitting_failed_unlock_remaps_existing_dormant_indices() -> None:
    first = candidate_key_from_parts(candidate_identity="G_old", position_id=3)
    state = admit_failed_unlock_dormant(
        PlateauAcquisitionState(),
        candidate_key=first,
        insertion_position=3,
    )
    assert state.dormant_logical_indices() == (3,)

    second = candidate_key_from_parts(candidate_identity="G_new", position_id=1)
    state = admit_failed_unlock_dormant(
        state,
        candidate_key=second,
        insertion_position=1,
    )

    assert state.active_episode is True
    assert state.failed_unlock_count == 2
    assert state.dormant_logical_indices() == (4, 1)
    assert remap_logical_indices_after_insertion([0, 3, 5], 3) == (0, 4, 6)


def test_plateau_state_payload_roundtrip_is_json_safe() -> None:
    key = candidate_key_from_record({"generator_id": "G", "candidate_label": "ignored", "position_id": 0})
    state = admit_failed_unlock_dormant(
        PlateauAcquisitionState(),
        candidate_key=key,
        insertion_position=0,
        event_payload={"trial_energy": -1.25, "tuple_payload": (1, 2)},
    )

    payload = state.as_dict()
    assert payload["dormant_records"][0]["candidate_key"]["candidate_identity"] == "G"
    assert payload["last_event"]["tuple_payload"] == [1, 2]
    assert plateau_state_from_payload(payload).as_dict() == payload


def test_exact_duplicate_admission_raises_under_default_policy() -> None:
    key = candidate_key_from_parts(candidate_identity="G", position_id=0)
    state = admit_failed_unlock_dormant(PlateauAcquisitionState(), candidate_key=key, insertion_position=0)

    with pytest.raises(PlateauAcquisitionError, match="duplicate"):
        admit_failed_unlock_dormant(state, candidate_key=key, insertion_position=1)


def test_failed_family_backoff_counts_same_identity_until_success() -> None:
    key = candidate_key_from_parts(candidate_identity="G", position_id=3)
    events = [
        {"event": "failed_unlock_dormant_admission", "candidate_key": {"candidate_identity": "G", "position_id": 0}},
        {"event": "failed_unlock_dormant_admission", "candidate_key": {"candidate_identity": "H", "position_id": 1}},
        {"event": "failed_unlock_dormant_admission", "candidate_key": {"candidate_identity": "G", "position_id": 2}},
    ]

    status = failed_family_backoff_status(events, key, patience=2)

    assert status["failed_family_count"] == 2
    assert status["blocked"] is True
    assert status["block_reason"] == "failed_family_backoff"

    reset_status = failed_family_backoff_status(
        [*events, {"event": "successful_unlock"}, events[0]],
        key,
        patience=2,
    )
    assert reset_status["failed_family_count"] == 1
    assert reset_status["blocked"] is False
