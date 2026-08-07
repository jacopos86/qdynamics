from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest

from pipelines.scaffold.hh_continuation_pruning import (
    recoverability_prune_ladder,
)
from pipelines.static_adapt import adapt_pipeline
from pipelines.static_adapt.cli_config import _build_adapt_arg_parser
from pipelines.static_adapt.sr_snake import RecoverabilityPruneReceipt


RETIRED_PRUNE_KEYS = frozenset(
    {
        "phase1_prune_stale_age",
        "phase1_prune_stagnation_threshold",
        "phase1_prune_small_theta_abs",
        "phase1_prune_small_theta_relative",
        "phase1_prune_amplitude_witness_required",
        "phase1_prune_collapse_peak_abs_min",
        "phase1_prune_collapse_current_abs_max",
        "phase1_prune_collapse_ratio",
        "phase1_prune_collapse_min_abs_drop",
        "phase1_prune_collapse_min_observations",
        "small_angle_pool_indices",
        "amplitude_witness_required",
        "amplitude_witness_ok",
        "amplitude_witness_reason",
        "amplitude_witness",
    }
)


def _keys_in(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return {
            *(str(key) for key in value),
            *(
                nested_key
                for nested in value.values()
                for nested_key in _keys_in(nested)
            ),
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return {
            nested_key
            for nested in value
            for nested_key in _keys_in(nested)
        }
    return set()


def _no_prune_kwargs() -> dict[str, Any]:
    return {
        "phase1_prune_policy": "recoverability_ladder_v1",
        "phase1_prune_max_candidates": 1,
        "phase1_prune_min_candidates": 1,
        "phase1_prune_fraction": 1.0,
        "phase1_prune_max_regression": 1.0e-8,
        "phase1_prune_tolerance_mode": "auto",
        "phase1_prune_tolerance_shot_coeff": 0.0,
        "phase1_prune_tolerance_screen_coeff": 0.01,
        "phase1_prune_tolerance_chem": 0.0,
        "phase1_prune_tolerance_rel_coeff": 0.0,
        "phase1_prune_tolerance_target_energy": None,
        "phase1_prune_retained_gain_ratio": 0.5,
        "phase1_prune_protect_steps": 2,
        "phase1_prune_cooldown_steps": 2,
        "phase1_prune_local_window_size": 0,
        "phase1_prune_recovery_trust_radius": 0.125,
        "phase1_prune_schur_nomination_route": (
            "full_logical_fs_trust_delete_refit_v1"
        ),
        "phase1_prune_metric_schur_mu": 0.0,
        "phase1_prune_metric_schur_solve_mode": (
            "affine_deletion_global_trust_v1"
        ),
        "phase1_prune_metric_schur_cost_weighting": "off",
        "phase1_prune_old_fraction": 0.25,
        "phase3_selector_policy": "algebraic_nested_v1",
        "phase1_prune_mode": "both",
        "phase1_prune_maturity_threshold": 0.5,
        "phase1_prune_checkpoint_period": 3,
        "phase1_prune_live_min_depth": 0,
        "phase1_prune_snr_threshold": 1.0,
        "phase1_prune_prefilter_policy": "off",
        "phase1_prune_risk_threshold": 0.0,
        "phase1_prune_prefilter_max_candidates": 1,
        "phase1_prune_trust_update_policy": (
            "modeled_local_fs_conservative_v1"
        ),
        "phase1_prune_metric_mu_update_policy": "off",
        "phase1_prune_endpoint_overlap_policy": "off",
    }


def test_cli_and_no_prune_receipt_exclude_retired_amplitude_surface() -> None:
    args = _build_adapt_arg_parser(
        adapt_gradient_parity_rtol=1.0e-7
    ).parse_args([])
    assert args.phase1_prune_policy == "recoverability_ladder_v1"
    assert all(not hasattr(args, key) for key in RETIRED_PRUNE_KEYS)

    receipt = adapt_pipeline._default_no_prune_prune_summary_template(
        kwargs=_no_prune_kwargs(),
        parameterization_mode="logical_shared",
    )
    assert RETIRED_PRUNE_KEYS.isdisjoint(_keys_in(receipt))


@pytest.mark.parametrize(
    "nomination_policy",
    (
        "metric_regularized_v1",
        "full_logical_fs_trust_delete_refit_v1",
    ),
)
def test_metric_and_trust_prune_receipts_exclude_retired_keys(
    nomination_policy: str,
) -> None:
    receipt = RecoverabilityPruneReceipt(
        status="not_executed",
        reason="no_mature_old_coordinate",
        policy="recoverability_ladder_v1",
        nomination_policy=nomination_policy,
        source_state_fingerprint="source",
        trust_radius_before=0.125,
        trust_radius_after=0.125,
        metric_damping=0.0,
        endpoint_overlap_query_charge=0,
        terminal_prune_active=False,
        final_state_fingerprint="source",
    ).to_dict()

    assert RETIRED_PRUNE_KEYS.isdisjoint(_keys_in(receipt))


def test_measured_deletion_runs_one_complete_survivor_refit() -> None:
    calls: list[tuple[int, tuple[int, ...], str]] = []

    def _measured_refit(
        removal_index: int,
        theta: np.ndarray,
        _labels: list[str],
        active_indices: list[int],
        rung_kind: str,
    ) -> tuple[float, np.ndarray]:
        calls.append(
            (
                int(removal_index),
                tuple(int(index) for index in active_indices),
                str(rung_kind),
            )
        )
        return -1.01, np.delete(theta, removal_index)

    theta, labels, decisions, energy, rows = recoverability_prune_ladder(
        theta=np.asarray([0.2, 0.1, 0.05], dtype=float),
        labels=["drop", "keep-a", "keep-b"],
        candidate_indices=[0],
        rung_windows_by_index={
            0: [("full_survivor_refit", [0, 1])]
        },
        eval_with_removal_window=_measured_refit,
        energy_before=-1.0,
        max_regression=0.02,
        max_trial_evaluations=1,
    )

    assert calls == [(0, (0, 1), "full_survivor_refit")]
    assert labels == ["keep-a", "keep-b"]
    assert theta.tolist() == pytest.approx([0.1, 0.05])
    assert energy == pytest.approx(-1.01)
    assert len(decisions) == len(rows) == 1
    assert decisions[0].accepted is True
    assert RETIRED_PRUNE_KEYS.isdisjoint(
        _keys_in([decisions[0].__dict__, rows[0]])
    )
