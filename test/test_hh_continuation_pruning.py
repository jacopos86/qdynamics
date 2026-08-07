from __future__ import annotations

import pytest
import numpy as np

from pipelines.scaffold.hh_continuation_pruning import (
    PRUNE_METRIC_SCHUR_SOLVE_GRADIENT_CORRECTED_V1,
    PRUNE_METRIC_SCHUR_SOLVE_STATIONARY_GW_ZERO_V1,
    PRUNE_POLICY_RECOVERABILITY_LADDER_V1,
    PRUNE_SCHUR_ROUTE_METRIC_REGULARIZED_V1,
    PRUNE_TOLERANCE_ADAPTIVE_V1,
    PRUNE_TOLERANCE_AUTO,
    PRUNE_TOLERANCE_FIXED,
    compute_prune_regression_tolerance,
    evaluate_prune_permission,
    evaluate_recoverability_curvature_guard,
    build_metric_regularized_prune_surrogate_scores,
    build_static_prune_surrogate_scores,
    cheap_prune_score,
    initialize_static_prune_curvature_cache,
    rank_prune_candidates,
    recoverability_prune_ladder,
    resolve_prune_tolerance_mode,
    metric_regularized_prune_schur_surrogate_ladder,
    static_prune_schur_surrogate_ladder,
    update_static_prune_curvature_cache,
)


def test_recoverability_prune_permission_opens_after_accepted_admission_not_low_snr() -> None:
    telemetry = evaluate_prune_permission(
        policy=PRUNE_POLICY_RECOVERABILITY_LADDER_V1,
        mode_enabled=True,
        has_min_scaffold=True,
        stable_refit=True,
        mature_open=True,
        accepted_admission=True,
        snr_low_enough=False,
        snr_adm=10.0,
    )
    assert telemetry["permission_open"] is True
    assert telemetry["permission_reason"] == "accepted_admission"
    assert telemetry["permission_triggers"]["accepted_admission"] is True

    low_snr_only = evaluate_prune_permission(
        policy=PRUNE_POLICY_RECOVERABILITY_LADDER_V1,
        mode_enabled=True,
        has_min_scaffold=True,
        stable_refit=True,
        mature_open=True,
        accepted_admission=False,
        snr_low_enough=True,
        snr_adm=0.1,
    )
    assert low_snr_only["permission_open"] is False
    assert low_snr_only["permission_reason"] == "awaiting_recoverability_cadence"
    assert low_snr_only["pressure_signals"]["snr_low_enough"] is True


def test_recoverability_curvature_guard_is_inactive_for_flat_rungs() -> None:
    guard = evaluate_recoverability_curvature_guard(
        rung_index=1,
        rung_kind="comm_refit",
        confidence_upper_regression=9e-7,
        regression_threshold=1e-6,
        mode="conservative_v1",
    )
    assert guard["curvature_guard_active"] is False
    assert guard["curvature_guard_ok"] is True


def test_recoverability_curvature_guard_rejects_loose_noncommuting_acceptance() -> None:
    guard = evaluate_recoverability_curvature_guard(
        rung_index=3,
        rung_kind="comm_corr_nc_refit",
        confidence_upper_regression=9e-7,
        regression_threshold=1e-6,
        mode="conservative_v1",
        context={"gamma_curv": 0.25, "terminal_full": False, "compression_mode": False},
        admitted_gain=0.0,
        retained_gain=1.0,
        retained_gain_ratio=0.5,
    )
    assert guard["curvature_guard_active"] is True
    assert guard["curvature_guard_ok"] is False
    assert guard["strong_retained_gain_ok"] is False


def test_recoverability_curvature_guard_accepts_strict_noncommuting_regression() -> None:
    guard = evaluate_recoverability_curvature_guard(
        rung_index=3,
        rung_kind="comm_corr_nc_refit",
        confidence_upper_regression=2e-7,
        regression_threshold=1e-6,
        mode="conservative_v1",
        context={"gamma_curv": 0.25},
    )
    assert guard["curvature_guard_active"] is True
    assert guard["curvature_guard_ok"] is True
    assert guard["curvature_guard_reason"] == "strict_regression"


def test_recoverability_curvature_guard_accepts_compression_mode() -> None:
    guard = evaluate_recoverability_curvature_guard(
        rung_index=3,
        rung_kind="comm_corr_nc_refit",
        confidence_upper_regression=9e-7,
        regression_threshold=1e-6,
        mode="conservative_v1",
        context={
            "gamma_curv": 0.25,
            "compression_mode": True,
            "active_window_fraction": 0.0,
        },
        admitted_gain=0.0,
        retained_gain=1.0,
        retained_gain_ratio=0.5,
    )
    assert guard["curvature_guard_active"] is True
    assert guard["curvature_guard_ok"] is True
    assert guard["curvature_guard_reason"] == "compression_mode"


def test_recoverability_curvature_guard_accepts_terminal_rung_s4() -> None:
    guard = evaluate_recoverability_curvature_guard(
        rung_index=4,
        rung_kind="terminal_refit",
        confidence_upper_regression=9e-7,
        regression_threshold=1e-6,
        mode="conservative_v1",
        context={
            "gamma_curv": 0.25,
            "compression_mode": False,
            "active_window_fraction": 0.0,
        },
    )
    assert guard["curvature_guard_active"] is True
    assert guard["curvature_guard_ok"] is True
    assert guard["curvature_guard_reason"] == "terminal_rung_s4"
    assert guard["terminal_rung_s4"] is True


def test_prune_tolerance_auto_resolves_by_policy() -> None:
    assert (
        resolve_prune_tolerance_mode(
            mode=PRUNE_TOLERANCE_AUTO,
            prune_policy=PRUNE_POLICY_RECOVERABILITY_LADDER_V1,
        )
        == PRUNE_TOLERANCE_ADAPTIVE_V1
    )
    assert (
        resolve_prune_tolerance_mode(
            mode=PRUNE_TOLERANCE_FIXED,
            prune_policy=PRUNE_POLICY_RECOVERABILITY_LADDER_V1,
        )
        == PRUNE_TOLERANCE_FIXED
    )
    with pytest.raises(ValueError, match="Unsupported prune policy"):
        resolve_prune_tolerance_mode(
            mode=PRUNE_TOLERANCE_AUTO,
            prune_policy="unknown_policy",
        )


def test_prune_tolerance_fixed_ignores_adaptive_terms() -> None:
    telemetry = compute_prune_regression_tolerance(
        delta_num=1e-10,
        mode=PRUNE_TOLERANCE_FIXED,
        sigma_e=1.0,
        delta_scr=1.0,
        delta_chem=1e-3,
        current_energy=-1.0,
        target_energy=-2.0,
        c_shot=1.0,
        c_scr=1.0,
        c_rel=1.0,
    )
    assert telemetry["effective_tolerance"] == pytest.approx(1e-10)
    assert telemetry["used_component"] == "delta_num"
    assert telemetry["dropped_components"]["screen"] == "fixed_mode"


def test_prune_tolerance_adaptive_uses_largest_available_component() -> None:
    telemetry = compute_prune_regression_tolerance(
        delta_num=1e-10,
        mode=PRUNE_TOLERANCE_ADAPTIVE_V1,
        sigma_e=2e-4,
        delta_scr=2e-2,
        delta_chem=7e-7,
        current_energy=-1.1,
        target_energy=-1.0,
        c_shot=0.1,
        c_scr=0.01,
        c_rel=1e-3,
    )
    assert telemetry["effective_tolerance"] == pytest.approx(2e-4)
    assert telemetry["used_component"] == "screen"
    assert telemetry["components"]["shot"]["value"] == pytest.approx(2e-5)
    assert telemetry["components"]["relative_target"]["value"] == pytest.approx(1e-4)


def test_prune_tolerance_adaptive_drops_unavailable_terms() -> None:
    telemetry = compute_prune_regression_tolerance(
        delta_num=3e-8,
        mode=PRUNE_TOLERANCE_ADAPTIVE_V1,
        sigma_e=float("nan"),
        delta_scr=-1.0,
        delta_chem=0.0,
        current_energy=-1.0,
        target_energy=None,
        c_shot=1.0,
        c_scr=1.0,
        c_rel=1.0,
    )
    assert telemetry["effective_tolerance"] == pytest.approx(3e-8)
    assert telemetry["used_component"] == "delta_num"
    assert telemetry["dropped_components"]["shot"] == "raw_value_unavailable"
    assert telemetry["dropped_components"]["screen"] == "raw_value_unavailable"
    assert telemetry["dropped_components"]["relative_target"] == "target_energy_unavailable"


def test_recoverability_prune_ladder_uses_effective_regression_threshold() -> None:
    def _eval(idx_remove: int, theta_cur: np.ndarray, labels_cur: list[str], *_args: object) -> tuple[float, np.ndarray]:
        return -0.99995, np.delete(np.asarray(theta_cur, dtype=float), [idx_remove])

    theta_out, labels_out, decisions, energy_out, ladder_rows = recoverability_prune_ladder(
        theta=np.array([0.1, 0.2], dtype=float),
        labels=["keep", "remove"],
        candidate_indices=[1],
        rung_windows_by_index={1: [("local_refit", [0])]},
        eval_with_removal_window=_eval,
        energy_before=-1.0,
        max_regression=1e-4,
    )
    assert labels_out == ["keep"]
    assert theta_out.tolist() == pytest.approx([0.1])
    assert energy_out == pytest.approx(-0.99995)
    assert decisions[0].accepted is True
    assert decisions[0].regression_threshold == pytest.approx(1e-4)
    assert ladder_rows[0]["regression_threshold"] == pytest.approx(1e-4)
    assert ladder_rows[0]["acceptance_source"] == "remove_refit_energy_safety"
    assert ladder_rows[0]["surrogate_used_for_acceptance"] is False


def test_recoverability_prune_ladder_rejects_above_effective_threshold() -> None:
    def _eval(idx_remove: int, theta_cur: np.ndarray, labels_cur: list[str], *_args: object) -> tuple[float, np.ndarray]:
        return -0.99995, np.delete(np.asarray(theta_cur, dtype=float), [idx_remove])

    theta_out, labels_out, decisions, energy_out, ladder_rows = recoverability_prune_ladder(
        theta=np.array([0.1, 0.2], dtype=float),
        labels=["keep", "remove"],
        candidate_indices=[1],
        rung_windows_by_index={1: [("local_refit", [0])]},
        eval_with_removal_window=_eval,
        energy_before=-1.0,
        max_regression=1e-6,
    )
    assert labels_out == ["keep", "remove"]
    assert theta_out.tolist() == pytest.approx([0.1, 0.2])
    assert energy_out == pytest.approx(-1.0)
    assert decisions[0].accepted is False
    assert decisions[0].reason == "safe_regression_exceeded"
    assert ladder_rows[0]["safe_regression_ok"] is False


def test_recoverability_prune_ladder_curvature_guard_can_reject_high_rung() -> None:
    def _eval(idx_remove: int, theta_cur: np.ndarray, labels_cur: list[str], *_args: object) -> tuple[float, np.ndarray]:
        return -0.9999991, np.delete(np.asarray(theta_cur, dtype=float), [idx_remove])

    theta_out, labels_out, decisions, energy_out, ladder_rows = recoverability_prune_ladder(
        theta=np.array([0.1, 0.2], dtype=float),
        labels=["keep", "remove"],
        candidate_indices=[1],
        rung_windows_by_index={1: [("comm_corr_nc_refit", [0])]},
        eval_with_removal_window=_eval,
        energy_before=-1.0,
        max_regression=1e-6,
        curvature_guard_mode="conservative_v1",
        curvature_guard_context={
            "gamma_curv": 0.25,
            "terminal_full": False,
            "active_window_fraction": 0.0,
        },
    )
    assert labels_out == ["keep", "remove"]
    assert theta_out.tolist() == pytest.approx([0.1, 0.2])
    assert energy_out == pytest.approx(-1.0)
    assert decisions[0].accepted is False
    assert decisions[0].curvature_guard_active is True
    assert decisions[0].curvature_guard_ok is False
    assert "curvature_compensated_guard_failed" in decisions[0].reason
    assert ladder_rows[0]["confidence_upper_regression"] == pytest.approx(9e-7)


def test_recoverability_prune_ladder_retained_gain_guard_still_rejects() -> None:
    def _eval(idx_remove: int, theta_cur: np.ndarray, labels_cur: list[str], *_args: object) -> tuple[float, np.ndarray]:
        return 0.9, np.delete(np.asarray(theta_cur, dtype=float), [idx_remove])

    _theta_out, labels_out, decisions, _energy_out, ladder_rows = recoverability_prune_ladder(
        theta=np.array([0.1, 0.2], dtype=float),
        labels=["keep", "remove"],
        candidate_indices=[1],
        rung_windows_by_index={1: [("local_refit", [0])]},
        eval_with_removal_window=_eval,
        energy_before=0.0,
        max_regression=1.0,
        retained_reference_energy=1.0,
        admitted_gain=1.0,
        retained_gain_ratio=0.5,
        retained_gain_activation=1e-12,
    )
    assert labels_out == ["keep", "remove"]
    assert decisions[0].safe_regression_ok is True
    assert decisions[0].retained_gain_ok is False
    assert ladder_rows[0]["retained_guard_active"] is True


def test_rank_prune_candidates_uses_proxy_benefit_without_metadata() -> None:
    theta = np.array([0.8, 0.01, 0.02, 0.5], dtype=float)
    labels = ["a", "b", "c", "d"]
    idx = rank_prune_candidates(
        theta=theta,
        labels=labels,
        marginal_proxy_benefit=[0.3, 0.4, 0.1, 0.2],
        max_candidates=3,
        min_candidates=2,
        fraction_candidates=0.5,
    )
    assert idx == [2, 3]


def test_rank_prune_candidates_uses_proxy_benefit_as_tiebreak_without_metadata() -> None:
    theta = np.array([0.01, 0.01, 0.5], dtype=float)
    labels = ["a", "b", "c"]
    idx = rank_prune_candidates(
        theta=theta,
        labels=labels,
        marginal_proxy_benefit=[0.3, 0.1, 1.0],
        max_candidates=2,
        min_candidates=2,
        fraction_candidates=0.5,
    )
    assert idx == [1, 0]


def test_rank_prune_candidates_metadata_path_applies_protection_and_cooldown() -> None:
    theta = np.array([0.004, 0.03, 0.001, 0.002], dtype=float)
    labels = ["a", "b", "c", "d"]
    idx = rank_prune_candidates(
        theta=theta,
        labels=labels,
        marginal_proxy_benefit=[0.3, 0.2, 0.1, 0.4],
        max_candidates=4,
        min_candidates=1,
        fraction_candidates=0.5,
        selector_burden=[0.0, 0.0, 0.0, 0.0],
        admission_steps=[1, 4, 1, 1],
        cooldown_remaining=[0, 0, 0, 2],
        current_step=5,
        protect_steps=2,
    )
    assert idx == [2, 0]


def test_recoverability_policy_does_not_require_small_angle_or_stale_gate() -> None:
    theta = np.array([0.4, 0.03, 0.5], dtype=float)
    labels = ["compensable", "small", "other"]
    idx = rank_prune_candidates(
        theta=theta,
        labels=labels,
        marginal_proxy_benefit=[0.5, 0.2, 0.4],
        max_candidates=2,
        min_candidates=1,
        fraction_candidates=0.5,
        selector_burden=[0.0, 0.0, 0.0],
        admission_steps=[1, 1, 1],
        cooldown_remaining=[0, 0, 0],
        current_step=10,
        protect_steps=2,
        policy=PRUNE_POLICY_RECOVERABILITY_LADDER_V1,
        surrogate_scores={0: {"score": 1e-5}, 1: {"score": 1e-2}},
    )
    assert idx[0] == 0
    assert 1 in idx


def test_recoverability_policy_gates_and_caps_schur_surrogate_candidates() -> None:
    theta = np.array([0.4, 0.03, 0.5], dtype=float)
    labels = ["good_schur", "bad_schur", "also_good"]
    idx = rank_prune_candidates(
        theta=theta,
        labels=labels,
        marginal_proxy_benefit=[0.5, 0.2, 0.4],
        max_candidates=3,
        min_candidates=1,
        fraction_candidates=1.0,
        selector_burden=[0.0, 0.0, 0.0],
        admission_steps=[1, 1, 1],
        cooldown_remaining=[0, 0, 0],
        current_step=10,
        protect_steps=2,
        policy=PRUNE_POLICY_RECOVERABILITY_LADDER_V1,
        surrogate_scores={
            0: {"score": 1e-5},
            1: {"score": 1e-2},
            2: {"score": 5e-5},
        },
        surrogate_score_threshold=1e-4,
        surrogate_candidate_cap=1,
    )
    assert idx == [0]


def test_static_prune_surrogate_zero_window_uses_full_survivor_window() -> None:
    rows = build_static_prune_surrogate_scores(
        theta=np.array([0.1, 0.2, 0.3], dtype=float),
        labels=["a", "b", "c"],
        hessian=np.eye(3),
        local_window_size=0,
    )

    for idx, row in rows.items():
        schur_rows = list(row["schur_rows"])
        assert len(schur_rows) == 3
        survivor_count = 2
        assert len(schur_rows[1]["window_indices"]) == survivor_count
        assert len(schur_rows[2]["window_indices"]) == survivor_count


def test_static_prune_surrogate_bounded_recovery_limits_ideal_compensation() -> None:
    ladder = static_prune_schur_surrogate_ladder(
        theta=np.array([1.0, 0.0], dtype=float),
        hessian=np.array([[1.0, 0.9], [0.9, 1.0]], dtype=float),
        block_indices=[0],
        windows=[[1]],
        ridge=0.0,
        recovery_trust_radius=0.1,
    )

    row = ladder["rows"][0]
    assert row["bounded_recovery_active"] is True
    assert row["bounded_recovery_clipped"] is True
    assert row["compensation_norm"] == pytest.approx(0.1)
    assert row["schur_value"] == pytest.approx(0.095)
    assert row["bounded_value"] == pytest.approx(0.415)

    unbounded = build_static_prune_surrogate_scores(
        theta=np.array([1.0, 0.0], dtype=float),
        labels=["delete_me", "survivor"],
        hessian=np.array([[1.0, 0.9], [0.9, 1.0]], dtype=float),
        local_window_size=0,
        ridge=0.0,
    )
    bounded = build_static_prune_surrogate_scores(
        theta=np.array([1.0, 0.0], dtype=float),
        labels=["delete_me", "survivor"],
        hessian=np.array([[1.0, 0.9], [0.9, 1.0]], dtype=float),
        local_window_size=0,
        ridge=0.0,
        recovery_trust_radius=0.1,
    )
    assert unbounded[0]["score"] == pytest.approx(0.095)
    assert bounded[0]["schur_min"] == pytest.approx(0.095)
    assert bounded[0]["score"] == pytest.approx(0.415)


def test_metric_regularized_prune_schur_mu_zero_matches_hessian_schur() -> None:
    theta = np.array([1.0, 0.0], dtype=float)
    hessian = np.array([[1.0, 0.9], [0.9, 1.0]], dtype=float)
    old = static_prune_schur_surrogate_ladder(
        theta=theta,
        hessian=hessian,
        block_indices=[0],
        windows=[[1]],
        ridge=0.0,
    )
    new = metric_regularized_prune_schur_surrogate_ladder(
        theta=theta,
        hessian=hessian,
        metric=np.eye(2),
        block_indices=[0],
        windows=[[1]],
        ridge=0.0,
        metric_mu=0.0,
        solve_mode=PRUNE_METRIC_SCHUR_SOLVE_STATIONARY_GW_ZERO_V1,
    )

    assert old["values"][0] == pytest.approx(0.095)
    assert new["values"][0] == pytest.approx(old["values"][0])
    assert new["rows"][0]["schur_model"] == PRUNE_SCHUR_ROUTE_METRIC_REGULARIZED_V1
    assert new["rows"][0]["warm_start_compensation_solve"] == pytest.approx([0.9])


def test_metric_regularized_prune_schur_gradient_corrected_subtracts_gw() -> None:
    theta = np.array([1.0, 0.0], dtype=float)
    hessian = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=float)
    metric = np.zeros((2, 2), dtype=float)
    stationary = metric_regularized_prune_schur_surrogate_ladder(
        theta=theta,
        hessian=hessian,
        metric=metric,
        block_indices=[0],
        windows=[[1]],
        ridge=0.0,
        metric_mu=0.0,
        solve_mode=PRUNE_METRIC_SCHUR_SOLVE_STATIONARY_GW_ZERO_V1,
        gradient=np.array([0.0, 0.25], dtype=float),
    )
    corrected = metric_regularized_prune_schur_surrogate_ladder(
        theta=theta,
        hessian=hessian,
        metric=metric,
        block_indices=[0],
        windows=[[1]],
        ridge=0.0,
        metric_mu=0.0,
        solve_mode=PRUNE_METRIC_SCHUR_SOLVE_GRADIENT_CORRECTED_V1,
        gradient=np.array([0.0, 0.25], dtype=float),
    )

    assert stationary["values"][0] == pytest.approx(0.375)
    assert corrected["values"][0] == pytest.approx(0.46875)
    assert corrected["rows"][0]["rhs"] == pytest.approx([0.25])
    assert corrected["rows"][0]["warm_start_compensation_solve"] == pytest.approx([0.25])


def test_metric_regularized_prune_score_divides_by_entry_cost_denominator() -> None:
    scores = build_metric_regularized_prune_surrogate_scores(
        theta=np.array([1.0, 1.0], dtype=float),
        labels=["costly", "cheap"],
        hessian=np.diag([2.0, 2.0]),
        metric=np.zeros((2, 2), dtype=float),
        local_window_size=0,
        ridge=0.0,
        metric_mu=0.0,
        entry_cost_denominators=[2.0, 1.0],
    )

    assert scores[0]["unweighted_score"] == pytest.approx(1.0)
    assert scores[0]["score"] == pytest.approx(0.5)
    assert scores[1]["score"] == pytest.approx(1.0)


def test_recoverability_primary_surrogate_rank_omits_legacy_tiebreak_ladder() -> None:
    idx = rank_prune_candidates(
        theta=np.array([0.01, 0.01], dtype=float),
        labels=["b_label", "a_label"],
        marginal_proxy_benefit=[0.0, -10.0],
        max_candidates=2,
        min_candidates=1,
        fraction_candidates=1.0,
        selector_burden=[0.0, 0.0],
        admission_steps=[0, 0],
        cooldown_remaining=[0, 0],
        current_step=10,
        protect_steps=0,
        policy=PRUNE_POLICY_RECOVERABILITY_LADDER_V1,
        surrogate_scores={0: {"score": 1.0}, 1: {"score": 1.0}},
        surrogate_score_primary_only=True,
    )

    assert idx == [1, 0]


def test_recoverability_ladder_honors_exact_trial_budget() -> None:
    calls: list[tuple[int, str]] = []

    def _eval(
        idx_remove: int,
        theta_cur: np.ndarray,
        labels_cur: list[str],
        active_indices: list[int],
        rung_kind: str,
    ) -> tuple[float, np.ndarray]:
        calls.append((int(idx_remove), str(rung_kind)))
        return 1.0, np.delete(np.asarray(theta_cur, dtype=float), [int(idx_remove)])

    theta_out, labels_out, decisions, energy_out, ladder_rows = recoverability_prune_ladder(
        theta=np.array([0.1, 0.2], dtype=float),
        labels=["a", "b"],
        candidate_indices=[0, 1],
        rung_windows_by_index={
            0: [("local_refit", [0]), ("full_refit", [0])],
            1: [("local_refit", [0])],
        },
        eval_with_removal_window=_eval,
        energy_before=0.0,
        max_regression=1e-6,
        max_trial_evaluations=1,
    )

    assert calls == [(0, "local_refit")]
    assert len(decisions) == 1
    assert len(ladder_rows) == 1
    assert labels_out == ["a", "b"]
    assert np.allclose(theta_out, np.array([0.1, 0.2]))
    assert energy_out == 0.0


def test_static_prune_curvature_cache_bfgs_update_is_psd_and_non_authoritative() -> None:
    cache = initialize_static_prune_curvature_cache(labels=["a", "b"], ridge=1e-4)
    cache, first = update_static_prune_curvature_cache(
        cache,
        labels=["a", "b"],
        theta=np.array([0.0, 0.0]),
        gradient=np.array([0.0, 0.0]),
        ridge=1e-4,
    )
    assert first["reason"] == "seeded_initial_pair"
    cache, diag = update_static_prune_curvature_cache(
        cache,
        labels=["a", "b"],
        theta=np.array([0.2, -0.1]),
        gradient=np.array([0.4, -0.2]),
        ridge=1e-4,
    )
    assert diag["updated"] is True
    assert cache.surrogate_authority == "rank_window_diag_only"
    assert np.allclose(cache.hessian, cache.hessian.T)
    assert np.min(np.linalg.eigvalsh(cache.hessian)) >= -1e-12


def test_static_prune_curvature_cache_bad_secant_is_damped_or_skipped() -> None:
    cache = initialize_static_prune_curvature_cache(labels=["a"], ridge=1e-3)
    cache, _ = update_static_prune_curvature_cache(
        cache,
        labels=["a"],
        theta=np.array([0.0]),
        gradient=np.array([0.0]),
    )
    cache, diag = update_static_prune_curvature_cache(
        cache,
        labels=["a"],
        theta=np.array([1.0]),
        gradient=np.array([-1.0]),
    )
    assert diag["surrogate_authority"] == "rank_window_diag_only"
    assert cache.health in {"healthy", "skipped_secant"}
    assert np.min(np.linalg.eigvalsh(cache.hessian)) >= -1e-12


def test_static_prune_schur_ladder_is_monotone_for_psd_hessian() -> None:
    theta = np.array([0.3, 0.1, -0.2], dtype=float)
    H = np.array(
        [
            [3.0, 0.6, 0.2],
            [0.6, 2.0, 0.1],
            [0.2, 0.1, 1.5],
        ],
        dtype=float,
    )
    ladder = static_prune_schur_surrogate_ladder(
        theta=theta,
        hessian=H,
        block_indices=[0],
        windows=[[], [1], [1, 2]],
        ridge=1e-9,
    )
    vals = ladder["values"]
    assert ladder["monotone"] is True
    assert vals[0] >= vals[1] >= vals[2]
    assert ladder["used_for_acceptance"] is False


def test_static_prune_surrogate_scores_are_ranking_evidence_only() -> None:
    theta = np.array([0.4, 0.1], dtype=float)
    H = np.array([[2.0, 1.5], [1.5, 2.0]], dtype=float)
    rows = build_static_prune_surrogate_scores(
        theta=theta,
        labels=["a", "b"],
        hessian=H,
        local_window_size=1,
    )
    assert set(rows) == {0, 1}
    assert all(row["surrogate_authority"] == "rank_window_diag_only" for row in rows.values())
    assert all(row["used_for_acceptance"] is False for row in rows.values())


def test_recoverability_ladder_escalates_from_bad_local_to_safe_full_refit() -> None:
    theta = np.array([0.2, 0.1, -0.1], dtype=float)
    labels = ["drop", "a", "b"]
    calls: list[str] = []

    def _eval(idx_remove, theta_cur, labels_cur, active_indices, rung_kind):
        calls.append(str(rung_kind))
        theta_new = np.delete(theta_cur, idx_remove)
        if str(rung_kind) == "local_refit":
            return 1.10, theta_new
        return 1.000000001, theta_new * 0.0

    theta_out, labels_out, decisions, energy_out, rows = recoverability_prune_ladder(
        theta=theta,
        labels=labels,
        candidate_indices=[0],
        rung_windows_by_index={0: [("local_refit", [0]), ("full_refit", [0, 1])]},
        eval_with_removal_window=_eval,
        energy_before=1.0,
        max_regression=1e-8,
        retained_reference_energy=1.0,
        admitted_gain=0.0,
        retained_gain_ratio=0.5,
        retained_gain_activation=1e-8,
    )
    assert calls == ["local_refit", "full_refit"]
    assert labels_out == ["a", "b"]
    assert energy_out == pytest.approx(1.000000001)
    assert np.allclose(theta_out, [0.0, 0.0])
    assert [d.accepted for d in decisions] == [False, True]
    assert rows[0]["accepted"] is False
    assert rows[1]["accepted"] is True
    assert rows[1]["acceptance_source"] == "remove_refit_energy_safety"
    assert rows[1]["surrogate_used_for_acceptance"] is False


def test_recoverability_ladder_rejects_when_all_refit_rungs_regress() -> None:
    theta = np.array([0.2, 0.1], dtype=float)
    labels = ["drop", "keep"]

    def _eval(idx_remove, theta_cur, labels_cur, active_indices, rung_kind):
        return 1.2, np.delete(theta_cur, idx_remove)

    theta_out, labels_out, decisions, energy_out, rows = recoverability_prune_ladder(
        theta=theta,
        labels=labels,
        candidate_indices=[0],
        rung_windows_by_index={0: [("local_refit", [0]), ("full_refit", [0])]},
        eval_with_removal_window=_eval,
        energy_before=1.0,
        max_regression=1e-8,
    )
    assert np.allclose(theta_out, theta)
    assert labels_out == labels
    assert energy_out == pytest.approx(1.0)
    assert len(decisions) == 2
    assert all(not d.accepted for d in decisions)
    assert all(row["surrogate_authority"] == "rank_window_diag_only" for row in rows)


def test_recoverability_ladder_retained_gain_guard_bypasses_when_admitted_gain_tiny() -> None:
    theta = np.array([0.2, 0.1], dtype=float)
    labels = ["drop", "keep"]

    def _eval(idx_remove, theta_cur, labels_cur, active_indices, rung_kind):
        return 1.0, np.delete(theta_cur, idx_remove)

    _theta_out, labels_out, decisions, _energy_out, rows = recoverability_prune_ladder(
        theta=theta,
        labels=labels,
        candidate_indices=[0],
        rung_windows_by_index={0: [("local_refit", [0])]},
        eval_with_removal_window=_eval,
        energy_before=1.0,
        max_regression=1e-8,
        retained_reference_energy=1.0,
        admitted_gain=1e-12,
        retained_gain_ratio=0.9,
        retained_gain_activation=1e-8,
    )
    assert labels_out == ["keep"]
    assert decisions[0].accepted is True
    assert decisions[0].retained_gain_threshold is None
    assert rows[0]["retained_guard_active"] is False


def test_cheap_prune_score_divides_frozen_regression_by_selector_burden() -> None:
    assert cheap_prune_score(frozen_regression=0.3, selector_burden=2.0) == pytest.approx(0.1)
    assert cheap_prune_score(frozen_regression=-1.0, selector_burden=2.0) == pytest.approx(0.0)
