"""Exact-store and offline-scoring contracts for the SIAM proof pair."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from pipelines.time_dynamics.diagnostics import siam_offline_scoring as scoring_module
from pipelines.time_dynamics.diagnostics.siam_exact_reference_store import (
    SIAM_EXACT_PROPAGATOR_V1,
    SIAM_EXACT_REFERENCE_STORE_V1,
    SIAM_EXACT_REFERENCE_STORE_V2,
    SIAMExactReferenceStore,
)
from pipelines.time_dynamics.diagnostics.siam_offline_scoring import (
    RELATIVE_ENERGY_DRIFT_CEILING,
    SIAM_MATCHED_PAIR_V1,
    SIAM_MATCHED_PAIR_V2,
    SIAM_MATCHED_PAIR_V3,
    SIAM_OFFLINE_SCORE_V1,
    SIAM_OFFLINE_SCORE_V2,
    SIAM_OFFLINE_SCORE_V3,
    SIAM_OFFLINE_SCORE_V4,
    SIAM_ROUTE_HISTORICAL_UNRECORDED,
    SIAM_ROUTE_NOT_APPLICABLE,
    SIAM_SIGMA_V2_TRANSITION_V1,
    _aggregate_accuracy_burden,
    _relative_energy_guard,
    _verified_score,
    build_siam_sigma_v2_transition_receipt,
    classify_siam_matched_pair,
    score_siam_raw_run,
    write_siam_matched_pair,
    write_siam_run_score,
)
from pipelines.time_dynamics.paper_ii_runs import (
    SIAM_NB4_CONTINUATION_TIMES,
    build_siam_nb2_proof_run,
)
from pipelines.time_dynamics.run_lock import build_run_lock
from pipelines.time_dynamics.resume import (
    CONTROLLER_RESUME_SCHEMA_V1,
    RESUME_SCHEMA_V2,
)
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import main
from pipelines.time_dynamics.siam_screening import PILOT_NB4, SIAMScreeningConfig
from pipelines.time_dynamics.siam_seed import write_siam_seed_manifest
from src.quantum.siam_model import embed_sector_vector
from src.quantum.siam_observables import SIAM_OBSERVABLES_V2
from src.quantum.siam_reference import SIAM_REFERENCE_V2, sector_hamiltonian


TIMES = (0.0, 0.04, 0.08)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _redigest_score(score: dict) -> dict:
    payload = json.loads(json.dumps(score))
    payload.pop("score_sha256", None)
    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {**payload, "score_sha256": digest}


def _with_accuracy_profile(
    score: dict,
    *,
    normalized_observable_errors: dict[str, float],
    mean_infidelity: float,
) -> dict:
    payload = json.loads(json.dumps(score))
    assert set(payload["primitive_response_metrics"]) == set(
        normalized_observable_errors
    )
    for name, value in normalized_observable_errors.items():
        payload["primitive_response_metrics"][name][
            "response_normalized_mean_absolute_error"
        ] = float(value)
    components = {
        name: float(value) / 0.05
        for name, value in normalized_observable_errors.items()
    }
    components["fidelity"] = float(mean_infidelity) / 1.0e-4
    summary = payload["summary"]
    summary["mean_infidelity"] = float(mean_infidelity)
    summary["max_primitive_response_normalized_error"] = max(
        normalized_observable_errors.values()
    )
    summary["normalized_accuracy_components"] = components
    summary["pair_normalized_accuracy_components"] = {
        **components,
        "energy": payload["summary"]["pair_normalized_accuracy_components"][
            "energy"
        ],
    }
    summary["aggregate_accuracy_burden"] = math.sqrt(
        0.25
        * (
            components["m_d"] ** 2
            + components["D_d"] ** 2
            + np.mean([components["C_1"] ** 2, components["C_2"] ** 2])
            + components["fidelity"] ** 2
        )
    )
    summary["fidelity_pass"] = bool(mean_infidelity <= 1.0e-4)
    summary["individual_target_pass"] = bool(
        max(normalized_observable_errors.values()) <= 0.05
        and summary["fidelity_pass"]
        and summary["energy_conservation_pass"]
    )
    summary["broad_response_guard_pass"] = bool(
        max(normalized_observable_errors.values()) <= 0.15
        and summary["fidelity_pass"]
        and summary["energy_conservation_pass"]
    )
    summary["classification"] = (
        "qualified"
        if summary["individual_target_pass"]
        else "candidate_broad_guard"
        if summary["broad_response_guard_pass"]
        else "candidate_outside_broad_guard"
    )
    return _redigest_score(payload)


def _seed(path: Path) -> None:
    write_siam_seed_manifest(
        path,
        SIAMScreeningConfig(
            n_bath=2,
            u=1.5,
            t_final=1.0,
            num_times=26,
            label="offline_scoring_test",
        ),
    )


def _lock(
    seed: Path,
    *,
    policy: str,
    ridge: float = 1.0e-6,
    code_revision: str = "siam-offline-score-test",
    times: tuple[float, ...] = TIMES,
) -> dict:
    return build_run_lock(
        seed_artifact_json=str(seed.resolve()),
        verified_seed_sha256=_sha256(seed),
        family_key="siam",
        n_ph_max=None,
        times=times,
        drive_profile=None,
        integrator_method="euler",
        inverse_policy={"ridge_lambda": ridge},
        solve_repair={"policy": "minimal"},
        structural_policy=policy,
        guards={"activation_l2_cut": 1.0e-3},
        time_step_controller={"state_motion_l2_step_max": 5.0e-5},
        candidate_pool={"schema": "siam_pool_v1", "completeness": "complete"},
        exact_reference_json=None,
        code_revision=code_revision,
    )


def _raw_run(
    path: Path,
    *,
    seed: Path,
    artifact,
    policy: str,
    times: tuple[float, ...] = TIMES,
    ridge: float = 1.0e-6,
    perturbation: float = 0.0,
    perturb_initial: bool = False,
    code_revision: str = "siam-offline-score-test",
    terminal_success: bool | None = None,
) -> None:
    h_sector = sector_hamiltonian(2, u=1.5, v=1.0 / np.sqrt(2.0), hopping=1.0)
    rows = []
    for index, (time, exact_state) in enumerate(zip(times, artifact.sector_states)):
        state = np.asarray(exact_state, dtype=complex).copy()
        if perturbation and (index > 0 or perturb_initial):
            direction = np.zeros_like(state)
            direction[(index + 1) % direction.size] = 1.0
            direction -= np.vdot(state, direction) * state
            direction /= np.linalg.norm(direction)
            state = state + float(perturbation) * direction
            state /= np.linalg.norm(state)
        full = embed_sector_vector(state, 2)
        rows.append(
            {
                "index": index,
                "time": time,
                "energy_expectation": float(np.real(np.vdot(state, h_sector @ state))),
                "psi_real": [float(value) for value in full.real],
                "psi_imag": [float(value) for value in full.imag],
            }
        )
    payload = {
        "schema": "ap_append_from_adapt_artifact_v1",
        "run_lock": _lock(
            seed,
            policy=policy,
            ridge=ridge,
            code_revision=code_revision,
            times=times,
        ),
        "resume_state": {
            "schema": RESUME_SCHEMA_V2,
            "time": times[-1],
            "seed_artifact_json": str(seed.resolve()),
            "problem_family": "siam",
            "parameterization_mode": "logical_shared",
            "layout": {
                "term_order": "sorted",
                "ignore_identity": True,
                "coefficient_tolerance": 1.0e-12,
            },
            "coordinates": [],
            "theta_runtime": [],
            "theta_logical": [],
            "controller_state": {
                "schema": CONTROLLER_RESUME_SCHEMA_V1,
                "checkpoint_index": len(rows) - 1,
                "prune_runtime_state": {},
                "repair_hold_state": None,
                "previous_accepted_theta_dot": None,
            },
        },
        "summary": {"uncured_local_subdivision_count": 0},
        "decision_data_flow": {
            "uses_reference_for_decision": False,
            "uses_exact_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
        },
        "plot_rows": rows,
    }
    if terminal_success is not None:
        payload["terminal_success"] = terminal_success
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path):
    seed = tmp_path / "seed.json"
    _seed(seed)
    store = SIAMExactReferenceStore(tmp_path / "exact")
    artifact = store.get_or_build(seed_path=seed, times=TIMES)
    return seed, store, artifact


def test_exact_reference_is_content_addressed_and_array_bound(tmp_path: Path) -> None:
    seed, store, artifact = _fixture(tmp_path)
    repeated = store.get_or_build(seed_path=seed, times=TIMES)

    assert repeated.key == artifact.key
    assert repeated.manifest_path == artifact.manifest_path
    assert SIAM_EXACT_REFERENCE_STORE_V1 != SIAM_EXACT_REFERENCE_STORE_V2
    assert artifact.manifest["schema"] == SIAM_EXACT_REFERENCE_STORE_V2
    assert artifact.manifest["propagator_schema"] == SIAM_EXACT_PROPAGATOR_V1
    assert artifact.manifest["seed_sha256"] == _sha256(seed)
    assert artifact.manifest["physical"]["n_bath"] == 2
    assert artifact.manifest["times"] == list(TIMES)
    assert set(artifact.manifest["arrays"]) == set(artifact.arrays)
    assert artifact.sector_states.shape == (3, 9)
    observable_names = {
        name.removeprefix("observable__")
        for name in artifact.arrays
        if name.startswith("observable__")
    }
    assert observable_names == {
        "m_d",
        "D_d",
        "N_d",
        "S_d2",
        "C_1",
        "C_2",
        "Sigma_1",
        "Sigma_2",
        "I_d",
    }

    independent = SIAMExactReferenceStore(tmp_path / "exact_independent").get_or_build(
        seed_path=seed, times=TIMES
    )
    assert independent.key == artifact.key
    assert independent.data_sha256 == artifact.data_sha256

    other = store.get_or_build(seed_path=seed, times=(0.0, 0.02, 0.08))
    assert other.key != artifact.key


def test_sigma_v2_rescore_is_version_bound_and_transitionable(tmp_path: Path) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    run = tmp_path / "run.json"
    _raw_run(run, seed=seed, artifact=artifact, policy="exchange")

    score = score_siam_raw_run(
        run_path=run,
        seed_path=seed,
        reference_store_root=tmp_path / "exact",
    )
    assert score["schema"] == SIAM_OFFLINE_SCORE_V4
    assert score["observable_contract"]["schema"] == SIAM_OBSERVABLES_V2
    assert score["reference"]["store_schema"] == SIAM_EXACT_REFERENCE_STORE_V2
    assert score["reference"]["reference_schema"] == SIAM_REFERENCE_V2
    assert score["reference"]["observable_schema"] == SIAM_OBSERVABLES_V2
    assert set(score["primitive_response_metrics"]) == {
        "m_d",
        "D_d",
        "C_1",
        "C_2",
    }
    for side in ("observed", "exact"):
        series = score["series"][side]
        n_d = np.asarray(series["N_d"])
        d_d = np.asarray(series["D_d"])
        s_d2 = np.asarray(series["S_d2"])
        assert np.allclose(s_d2, 0.75 * (n_d - 2.0 * d_d), atol=1.0e-12)
        assert np.allclose(
            series["Sigma_2"],
            s_d2 + np.asarray(series["C_1"]) + np.asarray(series["C_2"]),
            atol=1.0e-12,
        )

    historical = json.loads(json.dumps(score))
    historical["schema"] = SIAM_OFFLINE_SCORE_V2
    historical.pop("observable_contract")
    for field in ("store_schema", "reference_schema", "observable_schema"):
        historical["reference"].pop(field)
    historical = _redigest_score(historical)
    with pytest.raises(ValueError, match="Sigma-v2"):
        _verified_score(historical)

    transition = build_siam_sigma_v2_transition_receipt(historical, score)
    assert transition["schema"] == SIAM_SIGMA_V2_TRANSITION_V1
    assert transition["trajectory_rerun_required"] is False
    assert transition["primitive_gate_policy_changed"] is False


def test_score_verification_recomputes_ordered_time_grid_digest(
    tmp_path: Path,
) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    run = tmp_path / "run.json"
    _raw_run(run, seed=seed, artifact=artifact, policy="exchange")
    score = score_siam_raw_run(
        run_path=run,
        seed_path=seed,
        reference_store_root=tmp_path / "exact",
    )
    changed = json.loads(json.dumps(score))
    changed["grid"]["times"][1] = 0.041
    changed = _redigest_score(changed)
    destination = tmp_path / "changed-score.json"

    with pytest.raises(ValueError, match="ordered time-grid digest mismatch"):
        write_siam_run_score(destination, changed)

    assert not destination.exists()


def test_sigma_v2_transition_rejects_changed_full_time_grid(
    tmp_path: Path,
) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    run = tmp_path / "run.json"
    _raw_run(run, seed=seed, artifact=artifact, policy="exchange")
    score = score_siam_raw_run(
        run_path=run,
        seed_path=seed,
        reference_store_root=tmp_path / "exact",
    )
    historical = json.loads(json.dumps(score))
    historical["schema"] = SIAM_OFFLINE_SCORE_V2
    historical.pop("observable_contract")
    for field in ("store_schema", "reference_schema", "observable_schema"):
        historical["reference"].pop(field)
    historical = _redigest_score(historical)
    changed = json.loads(json.dumps(score))
    changed["grid"]["times"][1] = 0.041
    changed["grid"]["ordered_values_sha256"] = hashlib.sha256(
        json.dumps(
            [float(value) for value in changed["grid"]["times"]],
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    changed = _redigest_score(changed)

    with pytest.raises(ValueError, match="changes the full time grid"):
        build_siam_sigma_v2_transition_receipt(historical, changed)


def test_sigma_v2_transition_rejects_changed_initial_boundary(
    tmp_path: Path,
) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    run = tmp_path / "run.json"
    _raw_run(run, seed=seed, artifact=artifact, policy="exchange")
    score = score_siam_raw_run(
        run_path=run,
        seed_path=seed,
        reference_store_root=tmp_path / "exact",
    )
    historical = json.loads(json.dumps(score))
    historical["schema"] = SIAM_OFFLINE_SCORE_V2
    historical.pop("observable_contract")
    for field in ("store_schema", "reference_schema", "observable_schema"):
        historical["reference"].pop(field)
    historical = _redigest_score(historical)
    changed = json.loads(json.dumps(score))
    changed["diagnostics"]["initial_boundary"]["state_infidelity"] = 1.0e-13
    changed = _redigest_score(changed)

    with pytest.raises(ValueError, match="initial-boundary diagnostics differ"):
        build_siam_sigma_v2_transition_receipt(historical, changed)


def test_exact_reference_detects_data_tampering(tmp_path: Path) -> None:
    seed, store, artifact = _fixture(tmp_path)
    content = bytearray(artifact.data_path.read_bytes())
    content[len(content) // 2] ^= 1
    artifact.data_path.write_bytes(bytes(content))

    with pytest.raises(ValueError, match="data digest mismatch"):
        store.get_or_build(seed_path=seed, times=TIMES)


def test_explicit_prefix_reference_fails_closed_on_grid_or_receipt_mismatch(
    tmp_path: Path,
) -> None:
    seed, store, artifact = _fixture(tmp_path)
    receipt_path = tmp_path / "prefix.json"
    prefix = store.prefix_view(
        artifact,
        times=TIMES[:2],
        receipt_path=receipt_path,
    )
    run = tmp_path / "run.json"
    _raw_run(run, seed=seed, artifact=artifact, policy="exchange")

    with pytest.raises(ValueError, match="raw-run grid"):
        score_siam_raw_run(
            run_path=run,
            seed_path=seed,
            reference_store_root=tmp_path / "exact",
            verified_reference=prefix,
        )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["receipt_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt digest mismatch"):
        score_siam_raw_run(
            run_path=run,
            seed_path=seed,
            reference_store_root=tmp_path / "exact",
            verified_reference=prefix,
        )


def test_explicit_full_artifact_and_prefix_view_score_without_new_npz(
    tmp_path: Path,
) -> None:
    seed, store, artifact = _fixture(tmp_path)
    full_run = tmp_path / "full_run.json"
    _raw_run(full_run, seed=seed, artifact=artifact, policy="exchange")
    full_score = score_siam_raw_run(
        run_path=full_run,
        seed_path=seed,
        reference_store_root=tmp_path / "exact",
        verified_reference=artifact,
    )
    assert full_score["reference"]["key"] == artifact.key
    assert "prefix_view" not in full_score["reference"]

    receipt_path = tmp_path / "prefix_receipt.json"
    prefix = store.prefix_view(
        artifact,
        times=TIMES[:2],
        receipt_path=receipt_path,
    )
    prefix_run = tmp_path / "prefix_run.json"
    _raw_run(
        prefix_run,
        seed=seed,
        artifact=prefix,
        policy="exchange",
        times=TIMES[:2],
    )
    prefix_score = score_siam_raw_run(
        run_path=prefix_run,
        seed_path=seed,
        reference_store_root=tmp_path / "exact",
        verified_reference=store.load_prefix_view(receipt_path),
    )
    assert prefix_score["reference"]["key"] == artifact.key
    assert prefix_score["reference"]["prefix_view"]["receipt_sha256"] == (
        prefix.receipt["receipt_sha256"]
    )
    assert tuple((tmp_path / "exact").rglob("reference.npz")) == (
        artifact.data_path,
    )


def test_pilot_nb4_prefix_scoring_reuses_one_full_reference_npz(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = tmp_path / "nb4_seed.json"
    write_siam_seed_manifest(seed, PILOT_NB4)
    full_times = tuple(float(value) for value in PILOT_NB4.times)
    prefix_times = tuple(value for value in full_times if value <= 1.0)
    store_root = tmp_path / "exact"
    store = SIAMExactReferenceStore(store_root)

    def synthetic_arrays(*, physical, times):
        count = len(times)
        dimension = int(physical["sector_dimension"])
        state_real = np.zeros((count, dimension), dtype=np.dtype("<f8"))
        state_real[:, 0] = 1.0
        arrays = {
            "time": np.asarray(times, dtype=np.dtype("<f8")),
            "state_real": state_real,
            "state_imag": np.zeros_like(state_real),
            "energy": np.zeros(count, dtype=np.dtype("<f8")),
            "observable__m_d": np.zeros(count, dtype=np.dtype("<f8")),
            "observable__D_d": np.zeros(count, dtype=np.dtype("<f8")),
            "observable__N_d": np.zeros(count, dtype=np.dtype("<f8")),
            "observable__S_d2": np.zeros(count, dtype=np.dtype("<f8")),
            "observable__I_d": np.zeros(count, dtype=np.dtype("<f8")),
        }
        for site in range(1, int(physical["n_bath"]) + 1):
            arrays[f"observable__C_{site}"] = np.zeros(
                count, dtype=np.dtype("<f8")
            )
            arrays[f"observable__Sigma_{site}"] = np.zeros(
                count, dtype=np.dtype("<f8")
            )
        return arrays

    monkeypatch.setattr(
        SIAMExactReferenceStore,
        "_build_arrays",
        staticmethod(synthetic_arrays),
    )
    full = store.get_or_build(seed_path=seed, times=full_times)
    receipt_path = tmp_path / "nb4_prefix_receipt.json"
    prefix = store.prefix_view(
        full,
        times=prefix_times,
        receipt_path=receipt_path,
    )
    loaded_prefix = store.load_prefix_view(receipt_path)
    reference_files_before = tuple(store_root.rglob("reference.npz"))
    assert reference_files_before == (full.data_path,)

    run = tmp_path / "nb4_prefix_raw.json"
    run.write_text(
        json.dumps(
            {
                "run_lock": _lock(
                    seed,
                    policy="exchange",
                    times=prefix_times,
                ),
                "summary": {"uncured_local_subdivision_count": 0},
                "decision_data_flow": {
                    "uses_reference_for_decision": False,
                    "uses_exact_reference_for_decision": False,
                    "uses_future_exact_forecast_for_decision": False,
                },
                "plot_rows": [{"time": value} for value in prefix_times],
            }
        ),
        encoding="utf-8",
    )

    def reject_get_or_build(*args, **kwargs):
        raise AssertionError("prefix scoring must not call get_or_build")

    monkeypatch.setattr(
        SIAMExactReferenceStore,
        "get_or_build",
        reject_get_or_build,
    )

    def proof_core(**kwargs):
        reference = kwargs["reference"]
        return {
            "reference": {
                "key": reference.key,
                "manifest_sha256": reference.manifest_sha256,
                "data_sha256": reference.data_sha256,
            },
            "grid": {
                "times": list(kwargs["times"]),
            },
        }

    monkeypatch.setattr(scoring_module, "_score_core", proof_core)
    score = score_siam_raw_run(
        run_path=run,
        seed_path=seed,
        reference_store_root=store_root,
        verified_reference=loaded_prefix,
    )

    assert score["reference"]["key"] == full.key
    assert score["reference"]["prefix_view"]["receipt_sha256"] == (
        prefix.receipt["receipt_sha256"]
    )
    assert score["reference"]["prefix_view"]["point_count"] == len(prefix_times)
    assert tuple(store_root.rglob("reference.npz")) == reference_files_before

    continuation_receipt_path = tmp_path / "nb4_slice_receipt.json"
    continuation_view = store.slice_view(
        full,
        times=SIAM_NB4_CONTINUATION_TIMES,
        receipt_path=continuation_receipt_path,
    )
    continuation_run = tmp_path / "nb4_continuation_raw.json"
    continuation_run.write_text(
        json.dumps(
            {
                "run_lock": _lock(
                    seed,
                    policy="exchange",
                    times=SIAM_NB4_CONTINUATION_TIMES,
                ),
                "summary": {"uncured_local_subdivision_count": 0},
                "decision_data_flow": {
                    "uses_reference_for_decision": False,
                    "uses_exact_reference_for_decision": False,
                    "uses_future_exact_forecast_for_decision": False,
                },
                "plot_rows": [
                    {"time": value} for value in SIAM_NB4_CONTINUATION_TIMES
                ],
            }
        ),
        encoding="utf-8",
    )
    continuation_score = score_siam_raw_run(
        run_path=continuation_run,
        seed_path=seed,
        reference_store_root=store_root,
        verified_reference=store.load_slice_view(continuation_receipt_path),
    )
    assert continuation_score["reference"]["key"] == full.key
    assert continuation_score["reference"]["slice_view"]["receipt_sha256"] == (
        continuation_view.receipt["receipt_sha256"]
    )
    assert continuation_score["reference"]["slice_view"]["start_index"] == 100
    assert continuation_score["reference"]["slice_view"][
        "stop_index_exclusive"
    ] == 501
    assert tuple(store_root.rglob("reference.npz")) == reference_files_before


def test_pilot_nb4_segment_scoring_rejects_implicit_short_reference(
    tmp_path: Path,
) -> None:
    seed = tmp_path / "nb4_seed.json"
    write_siam_seed_manifest(seed, PILOT_NB4)
    run = tmp_path / "nb4_screen_raw.json"
    run.write_text(
        json.dumps(
            {
                "run_lock": _lock(
                    seed,
                    policy="exchange",
                    times=tuple(PILOT_NB4.times[:101]),
                ),
                "summary": {"uncured_local_subdivision_count": 0},
                "decision_data_flow": {
                    "uses_reference_for_decision": False,
                    "uses_exact_reference_for_decision": False,
                    "uses_future_exact_forecast_for_decision": False,
                },
                "plot_rows": [
                    {"time": value} for value in PILOT_NB4.times[:101]
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="one canonical full-T5"):
        score_siam_raw_run(
            run_path=run,
            seed_path=seed,
            reference_store_root=tmp_path / "exact",
        )
    assert tuple((tmp_path / "exact").rglob("reference.npz")) == ()


def test_scorer_reports_primitive_derived_and_safety_roles(tmp_path: Path) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    run = tmp_path / "exchange.json"
    _raw_run(run, seed=seed, artifact=artifact, policy="exchange")

    score = score_siam_raw_run(
        run_path=run,
        seed_path=seed,
        reference_store_root=tmp_path / "exact",
    )

    assert score["schema"] == SIAM_OFFLINE_SCORE_V4
    assert len(score["score_sha256"]) == 64
    assert set(score["primitive_response_metrics"]) == {"m_d", "D_d", "C_1", "C_2"}
    assert set(score["derived_screening_metrics"]) == {"Sigma_1", "Sigma_2"}
    assert "response_normalized_mean_absolute_error" not in score["diagnostics"]["I_d"]
    assert "response_scale" not in score["diagnostics"]["energy"]
    assert score["diagnostics"]["I_d"]["gate_role"] == "diagnostic_only_symmetry_identity"
    assert "sector_spectral_width_normalized" in (
        score["diagnostics"]["energy"]["gate_role"]
    )
    assert score["diagnostics"]["energy"]["sector_spectral_width"] > 0.0
    assert score["diagnostics"]["energy"]["g_E"] == (
        pytest.approx(0.0, abs=1.0e-14)
    )
    assert score["accuracy_policy"]["energy_in_aggregate_burden"] is False
    assert score["accuracy_policy"]["energy_in_componentwise_matching"] is True
    assert score["summary"]["individual_target_pass"] is True
    assert score["summary"]["broad_response_guard_pass"] is True
    json.dumps(score, allow_nan=False)


def test_scorer_rejects_family_seed_grid_and_missing_state_mismatches(
    tmp_path: Path,
) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    run = tmp_path / "run.json"
    _raw_run(run, seed=seed, artifact=artifact, policy="exchange")
    base = json.loads(run.read_text(encoding="utf-8"))

    cases = (
        (lambda doc: doc["run_lock"]["physics"].__setitem__("family_key", "hh"), "family_key"),
        (
            lambda doc: doc["run_lock"]["physics"].__setitem__(
                "seed_sha256", "0" * 64
            ),
            "seed digest",
        ),
        (
            lambda doc: doc["run_lock"]["physics"]["time_grid"].__setitem__(
                "ordered_values_sha256", "0" * 64
            ),
            "time-grid digest",
        ),
        (lambda doc: doc["plot_rows"][1].pop("psi_real"), "record its statevector"),
    )
    for index, (mutate, message) in enumerate(cases):
        changed = json.loads(json.dumps(base))
        mutate(changed)
        candidate = tmp_path / f"bad_{index}.json"
        candidate.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            score_siam_raw_run(
                run_path=candidate,
                seed_path=seed,
                reference_store_root=tmp_path / "exact",
            )


@pytest.mark.parametrize(
    "field",
    (
        "uses_reference_for_decision",
        "uses_exact_reference_for_decision",
        "uses_future_exact_forecast_for_decision",
    ),
)
def test_scorer_rejects_any_online_reference_decision_channel(
    tmp_path: Path,
    field: str,
) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    run = tmp_path / "run.json"
    _raw_run(run, seed=seed, artifact=artifact, policy="exchange")
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["decision_data_flow"][field] = True
    run.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exact-blind"):
        score_siam_raw_run(
            run_path=run,
            seed_path=seed,
            reference_store_root=tmp_path / "exact",
        )


@pytest.mark.parametrize(
    "field",
    (
        "uses_reference_for_decision",
        "uses_exact_reference_for_decision",
        "uses_future_exact_forecast_for_decision",
    ),
)
def test_scorer_requires_every_exact_blind_decision_flag_explicitly_false(
    tmp_path: Path,
    field: str,
) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    run = tmp_path / "run.json"
    _raw_run(run, seed=seed, artifact=artifact, policy="exchange")
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["decision_data_flow"].pop(field)
    run.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        score_siam_raw_run(
            run_path=run,
            seed_path=seed,
            reference_store_root=tmp_path / "exact",
        )


def test_scorer_rejects_conflicting_trajectory_exact_decision_metadata(
    tmp_path: Path,
) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    run = tmp_path / "run.json"
    _raw_run(run, seed=seed, artifact=artifact, policy="exchange")
    payload = json.loads(run.read_text(encoding="utf-8"))
    payload["trajectory"] = {
        "metadata": {
            "uses_reference_for_decision": False,
            "uses_exact_reference_for_decision": True,
            "uses_future_exact_forecast_for_decision": False,
        }
    }
    run.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="trajectory.metadata"):
        score_siam_raw_run(
            run_path=run,
            seed_path=seed,
            reference_store_root=tmp_path / "exact",
        )


def test_scorer_binds_parsed_run_and_digest_to_one_byte_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    run = tmp_path / "run.json"
    _raw_run(run, seed=seed, artifact=artifact, policy="exchange")
    original_bytes = run.read_bytes()
    original_core = scoring_module._score_core

    def mutate_path_after_parse(**kwargs):
        run.write_text("{}\n", encoding="utf-8")
        return original_core(**kwargs)

    monkeypatch.setattr(scoring_module, "_score_core", mutate_path_after_parse)
    score = score_siam_raw_run(
        run_path=run,
        seed_path=seed,
        reference_store_root=tmp_path / "exact",
    )

    assert score["source"]["run_sha256"] == hashlib.sha256(
        original_bytes
    ).hexdigest()
    assert score["source"]["run_sha256"] != _sha256(run)


def test_matched_pair_accepts_equal_burden_and_writes_immutably(tmp_path: Path) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    scores = {}
    for method in ("exchange", "avqds"):
        run = tmp_path / f"{method}.json"
        _raw_run(
            run,
            seed=seed,
            artifact=artifact,
            policy=method,
            perturbation=1.0e-8,
        )
        scores[method] = score_siam_raw_run(
            run_path=run,
            seed_path=seed,
            reference_store_root=tmp_path / "exact",
        )
    pair = classify_siam_matched_pair(scores["exchange"], scores["avqds"])

    assert pair["schema"] == SIAM_MATCHED_PAIR_V3
    assert pair["matched_accuracy"]["burden_ratio"] == pytest.approx(1.0)
    assert pair["matched_accuracy"]["burdens_matched"] is True
    assert pair["matched_accuracy"]["cost_comparison_admissible"] is True
    assert pair["matched_accuracy"]["classification"] == "qualified_pair"
    assert len(pair["pair_sha256"]) == 64

    score_path = tmp_path / "score.json"
    pair_path = tmp_path / "pair.json"
    write_siam_run_score(score_path, scores["exchange"])
    write_siam_run_score(score_path, scores["exchange"])
    write_siam_matched_pair(pair_path, pair)
    write_siam_matched_pair(pair_path, pair)
    assert json.loads(score_path.read_text())["score_sha256"] == scores["exchange"]["score_sha256"]
    assert json.loads(pair_path.read_text())["pair_sha256"] == pair["pair_sha256"]

    altered = dict(scores["exchange"])
    altered["score_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        write_siam_run_score(tmp_path / "bad_score.json", altered)


def test_matched_pair_rejects_incomparable_locks(tmp_path: Path) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    scores = []
    for method, ridge in (("exchange", 1.0e-6), ("avqds", 2.0e-6)):
        run = tmp_path / f"{method}.json"
        _raw_run(
            run,
            seed=seed,
            artifact=artifact,
            policy=method,
            ridge=ridge,
        )
        scores.append(
            score_siam_raw_run(
                run_path=run,
                seed_path=seed,
                reference_store_root=tmp_path / "exact",
            )
        )

    with pytest.raises(ValueError, match="comparison_fingerprint"):
        classify_siam_matched_pair(*scores)


def test_matched_pair_rejects_factor_greater_than_two_inside_broad_guard(
    tmp_path: Path,
) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    scores = {}
    for method, perturbation in (("exchange", 3.0e-5), ("avqds", 1.0e-4)):
        run = tmp_path / f"{method}.json"
        _raw_run(
            run,
            seed=seed,
            artifact=artifact,
            policy=method,
            perturbation=perturbation,
        )
        scores[method] = score_siam_raw_run(
            run_path=run,
            seed_path=seed,
            reference_store_root=tmp_path / "exact",
        )
        assert scores[method]["summary"]["broad_response_guard_pass"] is True

    pair = classify_siam_matched_pair(scores["exchange"], scores["avqds"])

    assert pair["matched_accuracy"]["burden_ratio"] > 2.0
    assert pair["matched_accuracy"]["burdens_matched"] is False
    assert pair["matched_accuracy"]["cost_comparison_admissible"] is False
    assert (
        pair["matched_accuracy"]["classification"]
        == "rejected_accuracy_confounded_pair"
    )


@pytest.mark.parametrize(
    ("changed_field", "changed_value", "message"),
    (
        ("broad_response_guard_pass", False, "broad_response_guard_pass"),
        ("energy_conservation_pass", False, "energy_conservation_pass"),
        ("fidelity_pass", False, "fidelity_pass"),
        ("individual_target_pass", False, "individual_target_pass"),
        ("classification", "candidate_outside_broad_guard", "classification"),
    ),
)
def test_matched_pair_rejects_redigested_derived_gate_or_classification_tampering(
    tmp_path: Path,
    changed_field: str,
    changed_value: object,
    message: str,
) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    scores = {}
    for method in ("exchange", "avqds"):
        run = tmp_path / f"{method}.json"
        _raw_run(
            run,
            seed=seed,
            artifact=artifact,
            policy=method,
            perturbation=1.0e-8,
        )
        scores[method] = score_siam_raw_run(
            run_path=run,
            seed_path=seed,
            reference_store_root=tmp_path / "exact",
        )
    changed = json.loads(json.dumps(scores["exchange"]))
    changed["summary"][changed_field] = changed_value
    changed = _redigest_score(changed)

    with pytest.raises(ValueError, match=message):
        classify_siam_matched_pair(changed, scores["avqds"])


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda score: score["summary"].__setitem__(
                "max_primitive_response_normalized_error", 99.0
            ),
            "max_primitive_response_normalized_error",
        ),
        (
            lambda score: score["diagnostics"]["energy"].__setitem__(
                "pass", False
            ),
            "diagnostics.energy.pass",
        ),
    ),
)
def test_matched_pair_rejects_redigested_metric_gate_inconsistency(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    scores = {}
    for method in ("exchange", "avqds"):
        run = tmp_path / f"{method}.json"
        _raw_run(run, seed=seed, artifact=artifact, policy=method)
        scores[method] = score_siam_raw_run(
            run_path=run,
            seed_path=seed,
            reference_store_root=tmp_path / "exact",
        )
    changed = json.loads(json.dumps(scores["exchange"]))
    mutate(changed)
    changed = _redigest_score(changed)

    with pytest.raises(ValueError, match=message):
        classify_siam_matched_pair(changed, scores["avqds"])


@pytest.mark.parametrize(
    "components",
    (
        {"m_d": 1.0, "D_d": 0.0, "C_1": 0.0, "C_2": 0.0, "fidelity": 0.0},
        {"m_d": 0.0, "D_d": 1.0, "C_1": 0.0, "C_2": 0.0, "fidelity": 0.0},
        {"m_d": 0.0, "D_d": 0.0, "C_1": 1.0, "C_2": 1.0, "fidelity": 0.0},
        {"m_d": 0.0, "D_d": 0.0, "C_1": 0.0, "C_2": 0.0, "fidelity": 1.0},
    ),
)
def test_aggregate_burden_weights_each_target_normalized_block_equally(
    components: dict[str, float],
) -> None:
    # q=1 is exactly the declared 5% observable boundary or 1e-4 fidelity
    # boundary. Each of the four bracketed blocks therefore contributes 1/4.
    assert _aggregate_accuracy_burden(components) == pytest.approx(0.5)


def test_matched_pair_rejects_complementary_errors_with_equal_aggregate_burden(
    tmp_path: Path,
) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    base_scores = {}
    for method in ("exchange", "avqds"):
        run = tmp_path / f"{method}.json"
        _raw_run(run, seed=seed, artifact=artifact, policy=method)
        base_scores[method] = score_siam_raw_run(
            run_path=run,
            seed_path=seed,
            reference_store_root=tmp_path / "exact",
        )
    apm = _with_accuracy_profile(
        base_scores["exchange"],
        normalized_observable_errors={
            "m_d": 0.05,
            "D_d": 0.005,
            "C_1": 0.005,
            "C_2": 0.005,
        },
        mean_infidelity=1.0e-5,
    )
    avqds = _with_accuracy_profile(
        base_scores["avqds"],
        normalized_observable_errors={
            "m_d": 0.005,
            "D_d": 0.05,
            "C_1": 0.005,
            "C_2": 0.005,
        },
        mean_infidelity=1.0e-5,
    )

    pair = classify_siam_matched_pair(apm, avqds)

    assert pair["matched_accuracy"]["burden_ratio"] == pytest.approx(1.0)
    assert pair["matched_accuracy"]["burdens_matched"] is True
    assert pair["matched_accuracy"]["componentwise_ratios"]["m_d"] == pytest.approx(
        10.0
    )
    assert pair["matched_accuracy"]["componentwise_ratios"]["D_d"] == pytest.approx(
        10.0
    )
    assert pair["matched_accuracy"]["components_matched"] is False
    assert pair["matched_accuracy"]["cost_comparison_admissible"] is False


def test_relative_energy_guard_is_shift_and_rescaling_invariant() -> None:
    baseline = _relative_energy_guard(
        (2.0, 2.002, 1.997),
        spectral_minimum=-3.0,
        spectral_maximum=7.0,
    )
    shifted = _relative_energy_guard(
        (13.0, 13.002, 12.997),
        spectral_minimum=8.0,
        spectral_maximum=18.0,
    )
    rescaled = _relative_energy_guard(
        (8.0, 8.008, 7.988),
        spectral_minimum=-12.0,
        spectral_maximum=28.0,
    )

    assert shifted["g_E"] == pytest.approx(
        baseline["g_E"]
    )
    assert rescaled["g_E"] == pytest.approx(
        baseline["g_E"]
    )


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    (
        (1.0, 1.0),
        (2.0, 1.0),
        (float("nan"), 1.0),
        (0.0, float("inf")),
    ),
)
def test_relative_energy_guard_rejects_invalid_sector_width(
    minimum: float,
    maximum: float,
) -> None:
    with pytest.raises(ValueError, match="spectral (bounds|width)"):
        _relative_energy_guard(
            (0.0, 1.0e-4),
            spectral_minimum=minimum,
            spectral_maximum=maximum,
        )


def test_relative_energy_guard_threshold_is_inclusive() -> None:
    boundary = _relative_energy_guard(
        (0.0, 0.01),
        spectral_minimum=-5.0,
        spectral_maximum=5.0,
    )
    outside = _relative_energy_guard(
        (0.0, 0.010001),
        spectral_minimum=-5.0,
        spectral_maximum=5.0,
    )

    assert boundary["g_E"] == pytest.approx(
        RELATIVE_ENERGY_DRIFT_CEILING
    )
    assert boundary["pass"] is True
    assert outside["g_E"] > RELATIVE_ENERGY_DRIFT_CEILING
    assert outside["pass"] is False


def test_matched_pair_rejects_componentwise_energy_burden_mismatch(
    tmp_path: Path,
) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    scores = {}
    for method in ("exchange", "avqds"):
        run = tmp_path / f"{method}.json"
        _raw_run(run, seed=seed, artifact=artifact, policy=method)
        scores[method] = score_siam_raw_run(
            run_path=run,
            seed_path=seed,
            reference_store_root=tmp_path / "exact",
        )

    for method, relative_drift in (("exchange", 1.0e-5), ("avqds", 5.0e-4)):
        payload = json.loads(json.dumps(scores[method]))
        energy = payload["diagnostics"]["energy"]
        width = float(energy["sector_spectral_width"])
        energy["g_E"] = relative_drift
        energy["max_absolute_drift"] = relative_drift * width
        energy["mean_absolute_drift"] = 0.5 * relative_drift * width
        energy["pass"] = True
        payload["summary"]["energy_conservation_pass"] = True
        payload["summary"]["pair_normalized_accuracy_components"]["energy"] = (
            relative_drift / RELATIVE_ENERGY_DRIFT_CEILING
        )
        scores[method] = _redigest_score(payload)

    pair = classify_siam_matched_pair(scores["exchange"], scores["avqds"])

    assert pair["matched_accuracy"]["burden_ratio"] == pytest.approx(1.0)
    assert pair["matched_accuracy"]["burdens_matched"] is True
    assert pair["matched_accuracy"]["componentwise_ratios"]["energy"] == (
        pytest.approx(5.0)
    )
    assert pair["matched_accuracy"]["components_matched"] is False
    assert pair["matched_accuracy"]["cost_comparison_admissible"] is False


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    (
        (
            "thresholds",
            "individual_response_target",
            0.051,
            "exact policy-v2 thresholds",
        ),
        (
            "thresholds",
            "unexpected_threshold",
            1.0,
            "exact policy-v2 thresholds",
        ),
        (
            "accuracy_policy",
            "energy_in_componentwise_matching",
            False,
            "exact policy-v2 identity",
        ),
    ),
)
def test_matched_pair_rejects_redigested_policy_v2_drift(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    scores = {}
    for method in ("exchange", "avqds"):
        run = tmp_path / f"{method}.json"
        _raw_run(run, seed=seed, artifact=artifact, policy=method)
        scores[method] = score_siam_raw_run(
            run_path=run,
            seed_path=seed,
            reference_store_root=tmp_path / "exact",
        )
    changed = json.loads(json.dumps(scores["exchange"]))
    changed[section][field] = value
    scores["exchange"] = _redigest_score(changed)

    with pytest.raises(ValueError, match=message):
        classify_siam_matched_pair(scores["exchange"], scores["avqds"])


@pytest.mark.parametrize("uncured", (None, 1))
def test_scorer_requires_explicit_zero_uncured_subdivision_count(
    tmp_path: Path,
    uncured: int | None,
) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    run = tmp_path / "run.json"
    _raw_run(run, seed=seed, artifact=artifact, policy="exchange")
    payload = json.loads(run.read_text(encoding="utf-8"))
    if uncured is None:
        payload["summary"].pop("uncured_local_subdivision_count")
    else:
        payload["summary"]["uncured_local_subdivision_count"] = uncured
    run.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="uncured_local_subdivision_count"):
        score_siam_raw_run(
            run_path=run,
            seed_path=seed,
            reference_store_root=tmp_path / "exact",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda resume: resume.__setitem__("schema", "paper_ii_resume_state_v1"), "resume schema"),
        (lambda resume: resume.pop("controller_state"), "controller resume state"),
        (lambda resume: resume.__setitem__("time", 0.04), "resume time"),
    ),
)
def test_scorer_requires_reconstructible_terminal_siam_resume_v2(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    run = tmp_path / "run.json"
    _raw_run(run, seed=seed, artifact=artifact, policy="exchange")
    payload = json.loads(run.read_text(encoding="utf-8"))
    mutation(payload["resume_state"])
    run.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        score_siam_raw_run(
            run_path=run,
            seed_path=seed,
            reference_store_root=tmp_path / "exact",
        )


def test_scorer_validates_present_terminal_success_marker(tmp_path: Path) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    failed = tmp_path / "failed.json"
    _raw_run(
        failed,
        seed=seed,
        artifact=artifact,
        policy="exchange",
        terminal_success=False,
    )
    with pytest.raises(ValueError, match="terminal-success marker"):
        score_siam_raw_run(
            run_path=failed,
            seed_path=seed,
            reference_store_root=tmp_path / "exact",
        )

    succeeded = tmp_path / "succeeded.json"
    _raw_run(
        succeeded,
        seed=seed,
        artifact=artifact,
        policy="exchange",
        terminal_success=True,
    )
    score = score_siam_raw_run(
        run_path=succeeded,
        seed_path=seed,
        reference_store_root=tmp_path / "exact",
    )
    assert score["diagnostics"]["terminal_success"] == {
        "runner_marker_provided": True,
        "validated_sources": ["run.terminal_success"],
        "implicit_boundary_checks_required": False,
    }
    assert score["diagnostics"]["resume"]["reconstruction_pass"] is True


def test_scorer_rejects_t0_state_not_equal_to_seed(tmp_path: Path) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    run = tmp_path / "run.json"
    _raw_run(
        run,
        seed=seed,
        artifact=artifact,
        policy="exchange",
        perturbation=1.0e-4,
        perturb_initial=True,
    )

    with pytest.raises(ValueError, match="t=0 state"):
        score_siam_raw_run(
            run_path=run,
            seed_path=seed,
            reference_store_root=tmp_path / "exact",
        )


def test_scorer_accepts_approximate_nonzero_resume_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = tmp_path / "seed.json"
    _seed(seed)
    continuation_times = (1.0, 1.04, 1.08)
    store = SIAMExactReferenceStore(tmp_path / "exact")
    full_reference = store.get_or_build(
        seed_path=seed,
        times=(0.0, *continuation_times),
    )

    class ContinuationReference:
        def __init__(self) -> None:
            self.key = full_reference.key
            self.manifest = full_reference.manifest
            self.manifest_path = full_reference.manifest_path
            self.manifest_sha256 = full_reference.manifest_sha256
            self.data_path = full_reference.data_path
            self.data_sha256 = full_reference.data_sha256
            self.times = continuation_times
            self.arrays = {
                name: np.asarray(values)[1:]
                for name, values in full_reference.arrays.items()
            }

        @property
        def sector_states(self):
            return np.asarray(full_reference.sector_states)[1:]

        def observable(self, name: str):
            return np.asarray(full_reference.observable(name))[1:]

    continuation_reference = ContinuationReference()
    run = tmp_path / "continuation.json"
    _raw_run(
        run,
        seed=seed,
        artifact=continuation_reference,
        policy="exchange",
        times=continuation_times,
        perturbation=1.0e-4,
        perturb_initial=True,
    )
    monkeypatch.setattr(
        SIAMExactReferenceStore,
        "get_or_build",
        lambda *args, **kwargs: continuation_reference,
    )

    score = score_siam_raw_run(
        run_path=run,
        seed_path=seed,
        reference_store_root=tmp_path / "exact",
    )

    boundary = score["diagnostics"]["initial_boundary"]
    assert boundary["boundary_time"] == pytest.approx(1.0)
    assert boundary["boundary_role"] == "approximate_resume_boundary"
    assert boundary["exact_seed_equality_required"] is False
    assert boundary["pass"] is None
    assert boundary["state_infidelity"] > 1.0e-12


def test_scorer_energy_ceiling_is_not_a_caller_override(tmp_path: Path) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    run = tmp_path / "run.json"
    _raw_run(run, seed=seed, artifact=artifact, policy="exchange")

    with pytest.raises(TypeError, match="energy_drift_ceiling"):
        score_siam_raw_run(
            run_path=run,
            seed_path=seed,
            reference_store_root=tmp_path / "exact",
            energy_drift_ceiling=1.0,
        )


def test_matched_pair_requires_equal_code_revisions(tmp_path: Path) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    scores = []
    for method, revision in (("exchange", "revision-a"), ("avqds", "revision-b")):
        run = tmp_path / f"{method}.json"
        _raw_run(
            run,
            seed=seed,
            artifact=artifact,
            policy=method,
            code_revision=revision,
        )
        scores.append(
            score_siam_raw_run(
                run_path=run,
                seed_path=seed,
                reference_store_root=tmp_path / "exact",
            )
        )

    with pytest.raises(ValueError, match="method.code_revision"):
        classify_siam_matched_pair(*scores)


def test_actual_short_siam_runner_output_crosses_offline_proof_boundary(
    tmp_path: Path,
) -> None:
    seed = tmp_path / "seed.json"
    output = tmp_path / "raw.json"
    _seed(seed)
    command = build_siam_nb2_proof_run(
        seed_path=str(seed),
        arm="exchange",
        output_json=str(output),
        code_revision="actual-short-proof-revision",
        extra_flags=("--t-final", "0.08", "--num-times", "3"),
    )
    assert main(list(command.argv())) == 0

    score = score_siam_raw_run(
        run_path=output,
        seed_path=seed,
        reference_store_root=tmp_path / "exact",
    )

    assert score["method"]["code_revision"] == "actual-short-proof-revision"
    assert score["diagnostics"]["initial_boundary"]["pass"] is True
    assert score["diagnostics"]["resume"]["schema"] == RESUME_SCHEMA_V2
    assert score["diagnostics"]["resume"]["reconstruction_pass"] is True
    assert score["schema"] == SIAM_OFFLINE_SCORE_V4
    assert score["thresholds"]["relative_energy_drift_ceiling"] == 1.0e-3


def test_immutable_writers_preserve_historical_v1_receipts(tmp_path: Path) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    run = tmp_path / "exchange.json"
    _raw_run(run, seed=seed, artifact=artifact, policy="exchange")
    current = score_siam_raw_run(
        run_path=run,
        seed_path=seed,
        reference_store_root=tmp_path / "exact",
    )
    legacy_score_core = json.loads(json.dumps(current))
    legacy_score_core.pop("score_sha256")
    legacy_score_core["schema"] = SIAM_OFFLINE_SCORE_V1
    legacy_score_core.pop("accuracy_policy")
    legacy_score_core["summary"].pop("pair_normalized_accuracy_components")
    legacy_score = _redigest_score(legacy_score_core)
    legacy_pair_core = {
        "schema": SIAM_MATCHED_PAIR_V1,
        "historical_policy": "absolute_energy_guard_v1",
    }
    legacy_pair = {
        **legacy_pair_core,
        "pair_sha256": hashlib.sha256(
            json.dumps(
                legacy_pair_core,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }

    score_path = tmp_path / "legacy_score.json"
    pair_path = tmp_path / "legacy_pair.json"
    write_siam_run_score(score_path, legacy_score)
    write_siam_run_score(score_path, legacy_score)
    write_siam_matched_pair(pair_path, legacy_pair)
    write_siam_matched_pair(pair_path, legacy_pair)

    assert json.loads(score_path.read_text()) == legacy_score
    assert json.loads(pair_path.read_text()) == legacy_pair


Q_COMPILED_ROUTE = "paper_ii_exchange_q_compiled_v1"


def _inject_route_guards(run: Path, *, bundle_id: str, digest: str) -> None:
    """Stamp one exchange route contract into an already-written raw run."""

    payload = json.loads(run.read_text(encoding="utf-8"))
    guards = dict(payload["run_lock"]["policy"]["guards"])
    guards["exchange_route_contract"] = {
        "bundle_id": bundle_id,
        "objective_id": "realized_q",
        "cost_model_id": "compiled_population",
        "kernel_id": "shifted_paper_i",
        "compiler": None,
    }
    guards["exchange_route_contract_sha256"] = digest
    payload["run_lock"]["policy"]["guards"] = guards
    run.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_v4_score_records_the_exchange_route_identity(tmp_path: Path) -> None:
    seed, _store, artifact = _fixture(tmp_path)

    historical = tmp_path / "historical.json"
    _raw_run(historical, seed=seed, artifact=artifact, policy="exchange")
    historical_score = score_siam_raw_run(
        run_path=historical,
        seed_path=seed,
        reference_store_root=tmp_path / "exact",
    )
    assert historical_score["schema"] == SIAM_OFFLINE_SCORE_V4
    assert historical_score["method"]["exchange_route_bundle"] == (
        SIAM_ROUTE_HISTORICAL_UNRECORDED
    )
    assert historical_score["method"]["exchange_route_contract_sha256"] is None

    comparator = tmp_path / "avqds.json"
    _raw_run(comparator, seed=seed, artifact=artifact, policy="avqds")
    comparator_score = score_siam_raw_run(
        run_path=comparator,
        seed_path=seed,
        reference_store_root=tmp_path / "exact",
    )
    assert comparator_score["method"]["exchange_route_bundle"] == (
        SIAM_ROUTE_NOT_APPLICABLE
    )
    assert comparator_score["method"]["exchange_route_contract_sha256"] is None

    compiled = tmp_path / "compiled.json"
    _raw_run(compiled, seed=seed, artifact=artifact, policy="exchange")
    _inject_route_guards(compiled, bundle_id=Q_COMPILED_ROUTE, digest="9" * 64)
    compiled_score = score_siam_raw_run(
        run_path=compiled,
        seed_path=seed,
        reference_store_root=tmp_path / "exact",
    )
    assert compiled_score["method"]["exchange_route_bundle"] == Q_COMPILED_ROUTE
    assert compiled_score["method"]["exchange_route_contract_sha256"] == "9" * 64
    # The comparison fingerprint deliberately excludes ``policy``: the two
    # structural arms must share the shared protocol, not the exchange route.
    assert (
        compiled_score["method"]["comparison_fingerprint"]
        == historical_score["method"]["comparison_fingerprint"]
    )


def test_pre_route_v3_score_reads_as_historical_unrecorded(tmp_path: Path) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    run = tmp_path / "run.json"
    _raw_run(run, seed=seed, artifact=artifact, policy="exchange")
    score = score_siam_raw_run(
        run_path=run,
        seed_path=seed,
        reference_store_root=tmp_path / "exact",
    )

    legacy = json.loads(json.dumps(score))
    legacy["schema"] = SIAM_OFFLINE_SCORE_V3
    legacy["method"].pop("exchange_route_bundle")
    legacy["method"].pop("exchange_route_contract_sha256")
    legacy = _redigest_score(legacy)

    verified = _verified_score(legacy)
    assert verified["schema"] == SIAM_OFFLINE_SCORE_V3
    assert "exchange_route_bundle" not in verified["method"]
    assert scoring_module._score_route(verified) == SIAM_ROUTE_HISTORICAL_UNRECORDED


def test_matched_pair_binds_the_declared_exchange_route(tmp_path: Path) -> None:
    seed, _store, artifact = _fixture(tmp_path)
    scores = {}
    for method in ("exchange", "avqds"):
        run = tmp_path / f"{method}.json"
        _raw_run(
            run,
            seed=seed,
            artifact=artifact,
            policy=method,
            perturbation=1.0e-8,
        )
        if method == "exchange":
            _inject_route_guards(
                run, bundle_id=Q_COMPILED_ROUTE, digest="7" * 64
            )
        scores[method] = score_siam_raw_run(
            run_path=run,
            seed_path=seed,
            reference_store_root=tmp_path / "exact",
        )

    pair = classify_siam_matched_pair(
        scores["exchange"],
        scores["avqds"],
        expected_apm_route=Q_COMPILED_ROUTE,
    )
    assert pair["matched_accuracy"]["apm_route"] == Q_COMPILED_ROUTE
    assert pair["matched_accuracy"]["avqds_route"] == SIAM_ROUTE_NOT_APPLICABLE

    with pytest.raises(ValueError, match="route differs from the campaign route"):
        classify_siam_matched_pair(
            scores["exchange"],
            scores["avqds"],
            expected_apm_route=SIAM_ROUTE_HISTORICAL_UNRECORDED,
        )

    # An unlabelled historical pilot can never satisfy a compiled-Q campaign.
    unlabelled = classify_siam_matched_pair(scores["exchange"], scores["avqds"])
    assert unlabelled["matched_accuracy"]["apm_route"] == Q_COMPILED_ROUTE
