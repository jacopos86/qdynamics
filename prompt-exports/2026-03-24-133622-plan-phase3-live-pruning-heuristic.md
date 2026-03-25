<file_map>
/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3
├── pipelines
│   ├── hardcoded
│   │   ├── hh_realtime_vqs
│   │   ├── adapt_pipeline.py * +
│   │   ├── hh_continuation_pruning.py * +
│   │   ├── hh_continuation_scoring.py * +
│   │   ├── hh_continuation_stage_control.py * +
│   │   └── hh_continuation_types.py * +
│   ├── exact_bench
│   └── shell
├── test
│   ├── test_adapt_vqe_integration.py * +
│   └── test_hh_continuation_pruning.py * +
├── .claude
├── MATH
├── artifacts
│   ├── agent_runs
│   │   ├── corrective_g0p5_strongwarm_depth400_v1
│   │   ├── corrective_g0p5_strongwarm_depth400_v1_rerun1
│   │   ├── l2_nph2_beam_depth_search
│   │   │   ├── json
│   │   │   ├── logs
│   │   │   └── recovery
│   │   └── overnight_shallow_g0p5_depth400
│   ├── csv
│   ├── json
│   │   ├── artifacts
│   │   │   └── json
│   │   ├── cfqm4_ps2_t4_budget_scout_runs
│   │   │   ├── cfqm4_S1
│   │   │   ├── cfqm4_S10
│   │   │   ├── cfqm4_S12
│   │   │   ├── cfqm4_S14
│   │   │   ├── cfqm4_S16
│   │   │   ├── cfqm4_S2
│   │   │   ├── cfqm4_S3
│   │   │   ├── cfqm4_S4
│   │   │   ├── cfqm4_S5
│   │   │   ├── cfqm4_S6
│   │   │   └── cfqm4_S8
│   │   ├── cfqm4_ps2_t4_lowest3caps_runs
│   │   │   ├── cfqm4_S1
│   │   │   ├── cfqm4_S2
│   │   │   └── cfqm4_S3
│   │   ├── cfqm4_t4_budget100_cmp_runs
│   │   │   ├── cfqm4_S4
│   │   │   ├── cfqm4_S6
│   │   │   └── cfqm4_S8
│   │   ├── feasibility_t5_cx200_runs
│   │   │   ├── cfqm4_S1
│   │   │   └── suzuki2_S1
│   │   ├── hh_fixed_seed_budgeted_projected_dynamics_runs
│   │   │   ├── K01
│   │   │   ├── K02
│   │   │   ├── K03
│   │   │   ├── K04
│   │   │   ├── K05
│   │   │   ├── K06
│   │   │   ├── K07
│   │   │   ├── K08
│   │   │   ├── K09
│   │   │   ├── K10
│   │   │   ├── K11
│   │   │   ├── K12
│   │   │   ├── K13
│   │   │   ├── K14
│   │   │   └── K15
│   │   ├── hh_fixed_seed_local_checkpoint_fit_runs
│   │   │   ├── local_y_xx-zz_reps1
│   │   │   ├── local_y_xx-zz_reps2
│   │   │   └── local_y_xx-zz_reps3
│   │   ├── hh_fixed_seed_local_checkpoint_fit_t4_runs
│   │   │   ├── local_y_xx-zz_reps1
│   │   │   ├── local_y_xx-zz_reps2
│   │   │   └── local_y_xx-zz_reps3
│   │   ├── hh_fixed_seed_qpu_prep_selected_runs
│   │   │   └── suzuki2_S64
│   │   ├── hh_fixed_seed_qpu_prep_snapshot_t4_runs
│   │   │   ├── cfqm4_S1
│   │   │   ├── cfqm4_S2
│   │   │   ├── cfqm4_S4
│   │   │   ├── suzuki2_S2
│   │   │   ├── suzuki2_S4
│   │   │   └── suzuki2_S8
│   │   ├── hh_fixed_seed_qpu_prep_suzuki_only_runs
│   │   │   ├── suzuki2_S128
│   │   │   ├── suzuki2_S64
│   │   │   └── suzuki2_S96
│   │   ├── hh_l2_diag_sentinel_p0_p1_p2_runs
│   │   │   ├── hh_l2_diag_sentinel_p0_p1_p2_U4_g0p5_w1_seed01
│   │   │   └── hh_l2_diag_sentinel_p0_p1_p2_U4_g0p5_w1_seed02
│   │   ├── hh_l2_diag_sentinel_p0_p1_p2_runs.partial-20260313-153327
│   │   │   └── hh_l2_diag_sentinel_p0_p1_p2_U4_g0p5_w1_seed01
│   │   ├── hh_l2_diag_spine_gsweep_runs
│   │   │   └── hh_l2_diag_spine_gsweep_U4_g0p5_w1_seed01
│   │   ├── hh_l2_fullmeta_legacy_heavy_runs
│   │   │   └── hh_l2_fullmeta_legacy_heavy_U4_g1_w1_seed01
│   │   ├── hh_l2_heavy_prune_run
│   │   │   └── l2_hh_open_heavy_prune_phase3_paoplf_u4_g1_w1
│   │   ├── hh_l2_live_scaffold_patch_benchmark_run
│   │   │   ├── 20260318_hh_l2_promoted_live_patch_main_only_v1
│   │   │   │   └── promoted_live_bridge_fixed
│   │   │   ├── 20260318_hh_l2_promoted_live_patch_main_only_v2
│   │   │   │   └── promoted_live_bridge_fixed
│   │   │   ├── 20260318_hh_l2_promoted_live_patch_main_supervised_live__attempt1
│   │   │   │   └── promoted_live_bridge_fixed
│   │   │   ├── 20260318_hh_l2_promoted_live_patch_v1
│   │   │   │   └── promoted_live_bridge_fixed
│   │   │   ├── 20260319_hh_l2_no_replay_multipatch_controls_v1__attempt1
│   │   │   │   ├── hva_live_direct
│   │   │   │   ├── promoted_live_bridge_fixed
│   │   │   │   ├── ptw_live_direct
│   │   │   │   └── ptw_prepared_legacy
│   │   │   ├── 20260319_hh_l2_promoted_live_patch_main_launchd_v1__attempt1
│   │   │   │   └── promoted_live_bridge_fixed
│   │   │   ├── 20260319_hh_l2_promoted_live_patch_main_launchd_v1__attempt2
│   │   │   │   └── promoted_live_bridge_fixed
│   │   │   ├── 20260319_hh_l2_promoted_live_patch_parallel_controls_v1__attempt1
│   │   │   │   ├── hva_live_direct
│   │   │   │   ├── promoted_live_bridge_fixed
│   │   │   │   ├── ptw_live_direct
│   │   │   │   └── ptw_prepared_legacy
│   │   │   ├── 20260319_hh_l2_promoted_live_patch_parallel_controls_v1__attempt2
│   │   │   │   ├── hva_live_direct
│   │   │   │   ├── promoted_live_bridge_fixed
│   │   │   │   ├── ptw_live_direct
│   │   │   │   └── ptw_prepared_legacy
│   │   │   ├── 20260319_hh_l2_promoted_live_patch_parallel_controls_v1__attempt3
│   │   │   │   ├── hva_live_direct
│   │   │   │   ├── promoted_live_bridge_fixed
│   │   │   │   ├── ptw_live_direct
│   │   │   │   └── ptw_prepared_legacy
│   │   │   ├── 20260319_hh_l2_ptw_live_core_pool_compare_v1__attempt1
│   │   │   │   ├── ptw_live_core_paop_lf_std
│   │   │   │   └── ptw_live_core_uccsd_otimes_paop_lf_std
│   │   │   ├── 20260319_hh_l2_ptw_warm_core_beam_replay_compare_v1__attempt1
│   │   │   │   ├── ptw_live_core_paop_lf_std_beam_off
│   │   │   │   ├── ptw_live_core_paop_lf_std_beam_on
│   │   │   │   ├── ptw_live_core_uccsd_otimes_paop_lf_std_beam_off
│   │   │   │   └── ptw_live_core_uccsd_otimes_paop_lf_std_beam_on
│   │   │   ├── 20260319_hh_l2_ptw_warm_core_beam_replay_compare_v1__attempt2
│   │   │   │   ├── ptw_live_core_paop_lf_std_beam_off
│   │   │   │   ├── ptw_live_core_paop_lf_std_beam_on
│   │   │   │   ├── ptw_live_core_uccsd_otimes_paop_lf_std_beam_off
│   │   │   │   └── ptw_live_core_uccsd_otimes_paop_lf_std_beam_on
│   │   │   ├── 20260319_hh_l2_ptw_warm_core_beam_replay_compare_v2__attempt1
│   │   │   │   ├── ptw_live_core_paop_lf_std_beam_off
│   │   │   │   ├── ptw_live_core_paop_lf_std_beam_on
│   │   │   │   ├── ptw_live_core_uccsd_otimes_paop_lf_std_beam_off
│   │   │   │   └── ptw_live_core_uccsd_otimes_paop_lf_std_beam_on
│   │   │   ├── 20260319_hh_l2_ptw_warm_core_beam_replay_compare_v2__attempt2
│   │   │   │   ├── ptw_live_core_paop_lf_std_beam_off
│   │   │   │   ├── ptw_live_core_paop_lf_std_beam_on
│   │   │   │   ├── ptw_live_core_uccsd_otimes_paop_lf_std_beam_off
│   │   │   │   └── ptw_live_core_uccsd_otimes_paop_lf_std_beam_on
│   │   │   ├── 20260320_hh_l2_historical_d6_pair_compare_v1__attempt1
│   │   │   │   ├── historical_d6_beam_off
│   │   │   │   └── historical_d6_beam_on
│   │   │   ├── 20260320_hh_l2_old_good_d6_factorial_compare_v1__attempt1
│   │   │   │   ├── old_good_d6_seedrefine_off_beam_off
│   │   │   │   ├── old_good_d6_seedrefine_off_beam_on
│   │   │   │   ├── old_good_d6_seedrefine_on_beam_off
│   │   │   │   │   └── state_export
│   │   │   │   └── old_good_d6_seedrefine_on_beam_on
│   │   │   │       └── state_export
│   │   │   ├── 20260320_hh_l2_old_good_replay_compare_v1__attempt1
│   │   │   │   ├── old_good_ptw_prepared_beam_d6_powell
│   │   │   │   └── old_good_ptw_prepared_seed_refine_powell_b1
│   │   │   │       └── state_export
│   │   │   └── 20260320_hh_l2_old_good_replay_compare_v1_probe__attempt1
│   │   │       ├── old_good_ptw_prepared_beam_d6_powell
│   │   │       └── old_good_ptw_prepared_seed_refine_powell_b1
│   │   ├── hh_l2_live_scaffold_patch_benchmark_supervised
│   │   │   ├── 20260318_hh_l2_promoted_live_patch_main_supervised_live
│   │   │   │   └── attempt_1
│   │   │   ├── 20260318_hh_l2_promoted_live_patch_main_supervised_v1
│   │   │   ├── 20260319_hh_l2_no_replay_multipatch_controls_v1
│   │   │   │   └── attempt_1
│   │   │   ├── 20260319_hh_l2_promoted_live_patch_main_launchd_v1
│   │   │   │   ├── attempt_1
│   │   │   │   └── attempt_2
│   │   │   ├── 20260319_hh_l2_promoted_live_patch_parallel_controls_v1
│   │   │   │   ├── attempt_1
│   │   │   │   ├── attempt_2
│   │   │   │   └── attempt_3
│   │   │   ├── 20260319_hh_l2_ptw_live_core_pool_compare_v1
│   │   │   │   └── attempt_1
│   │   │   ├── 20260319_hh_l2_ptw_warm_core_beam_replay_compare_v1
│   │   │   │   ├── attempt_1
│   │   │   │   └── attempt_2
│   │   │   ├── 20260319_hh_l2_ptw_warm_core_beam_replay_compare_v2
│   │   │   │   ├── attempt_1
│   │   │   │   └── attempt_2
│   │   │   ├── 20260320_hh_l2_historical_d6_pair_compare_v1
│   │   │   │   └── attempt_1
│   │   │   ├── 20260320_hh_l2_old_good_d6_factorial_compare_v1
│   │   │   │   └── attempt_1
│   │   │   ├── 20260320_hh_l2_old_good_replay_compare_v1
│   │   │   │   └── attempt_1
│   │   │   └── 20260320_hh_l2_old_good_replay_compare_v1_probe
│   │   │       └── attempt_1
│   │   ├── hh_l2_logical_screen_runs
│   │   │   ├── hh_l2_logical_screen_U4_g1_w1_seed01
│   │   │   ├── hh_l2_logical_screen_U4_g1_w1_seed02
│   │   │   └── hh_l2_logical_screen_U4_g1_w1_seed03
│   │   ├── hh_l2_logical_screen_runs_heavy_d30_a3000_f2500
│   │   │   ├── hh_l2_logical_screen_heavy_d30_a3000_f2500_U4_g1_w1_seed01
│   │   │   ├── hh_l2_logical_screen_heavy_d30_a3000_f2500_U4_g1_w1_seed02
│   │   │   └── hh_l2_logical_screen_heavy_d30_a3000_f2500_U4_g1_w1_seed03
│   │   ├── hh_l2_logical_screen_runs_v2full
│   │   │   └── hh_l2_logical_screen_v2full_U4_g1_w1_seed01
│   │   ├── hh_l2_prune_saved_parent
│   │   │   └── l2_hh_open_prune_parent97_phase3_allhhmeta_u4_g05_nph2_pass1
│   │   │       └── l2_hh_open_prune_parent97_phase3_allhhmeta_u4_g05_nph2_pass1
│   │   ├── hh_optimal_exp_qualifier
│   │   ├── matched_cx_t5_runs
│   │   │   ├── cfqm4_S1
│   │   │   ├── cfqm4_S2
│   │   │   ├── cfqm4_S3
│   │   │   ├── suzuki2_S1
│   │   │   ├── suzuki2_S2
│   │   │   ├── suzuki2_S3
│   │   │   ├── suzuki2_S4
│   │   │   ├── suzuki2_S5
│   │   │   ├── suzuki2_S6
│   │   │   └── suzuki2_S8
│   │   ├── noise_l2_pdf
│   │   ├── noise_l2_test
│   │   ├── phase_a_local_rehearsal_20260317_134642
│   │   └── suzuki2_t4_seed_cmp_runs
│   │       ├── suzuki2_S1
│   │       ├── suzuki2_S10
│   │       ├── suzuki2_S12
│   │       ├── suzuki2_S14
│   │       ├── suzuki2_S16
│   │       ├── suzuki2_S2
│   │       ├── suzuki2_S3
│   │       ├── suzuki2_S4
│   │       ├── suzuki2_S5
│   │       ├── suzuki2_S6
│   │       └── suzuki2_S8
│   ├── logs
│   │   ├── live_qpu
│   │   └── noise_l2_test
│   ├── pdf
│   │   └── noise_l2_pdf
│   ├── phase3_window_ab_sweep_proxy1_20260320_162749
│   │   ├── json
│   │   └── logs
│   ├── phase3_window_ab_sweep_shots2048_20260320_162749
│   │   ├── json
│   │   └── logs
│   ├── phase3_windowed_cost_sweep_20260320_143517
│   │   ├── json
│   │   └── logs
│   ├── phase3_windowed_cost_sweep_g1p0_20260320_151337
│   │   ├── json
│   │   └── logs
│   ├── reports
│   ├── runstate
│   ├── useful
│   │   ├── L2
│   │   └── L3
│   └── user_runs
│       ├── 20260309_hh_l2_noiseless
│       │   ├── json
│       │   ├── logs
│       │   └── staged_logs
│       └── 20260309_hh_l2_v3_plateau_only
│           └── logs
├── docs
│   └── reports
├── prompt-exports
└── src
    └── quantum
        ├── operator_pools
        └── time_propagation


(* denotes selected files)
(+ denotes code-map available)
Config: directory-only view; selected files shown.
</file_map>
<file_contents>
File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/hardcoded/hh_continuation_stage_control.py
```py
#!/usr/bin/env python3
"""Stage and position-probe policy for HH continuation Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class StageControllerConfig:
    plateau_patience: int = 2
    weak_drop_threshold: float = 1e-9
    probe_margin_ratio: float = 1.0
    max_probe_positions: int = 6
    append_admit_threshold: float = 0.05
    family_repeat_patience: int = 2


@dataclass(frozen=True)
class PositionProbeDecision:
    should_probe: bool
    reason: str
    positions: list[int]


def allowed_positions(
    *,
    n_params: int,
    append_position: int,
    active_window_indices: Iterable[int],
    max_positions: int,
) -> list[int]:
    positions = [int(append_position)]
    if int(n_params) <= 0:
        return [0]

    positions.append(0)
    for idx in active_window_indices:
        positions.append(int(idx))

    out: list[int] = []
    for p in positions:
        p_clamped = max(0, min(int(append_position), int(p)))
        if p_clamped not in out:
            out.append(p_clamped)
        if len(out) >= int(max_positions):
            break
    return out


def detect_trough(
    *,
    append_score: float,
    best_non_append_score: float,
    best_non_append_g_lcb: float,
    margin_ratio: float,
    append_admit_threshold: float,
) -> bool:
    if float(best_non_append_g_lcb) <= 0.0:
        return False
    if float(best_non_append_score) >= float(margin_ratio) * float(append_score):
        return True
    return (
        float(append_score) < float(append_admit_threshold)
        and float(best_non_append_score) >= float(append_admit_threshold)
    )


def should_probe_positions(
    *,
    stage_name: str,
    drop_plateau_hits: int,
    max_grad: float,
    eps_grad: float,
    append_score: float,
    finite_angle_flat: bool,
    repeated_family_flat: bool,
    cfg: StageControllerConfig,
) -> tuple[bool, str]:
    if str(stage_name) == "residual":
        return False, "residual_stage"
    if int(drop_plateau_hits) >= int(cfg.plateau_patience):
        return True, "drop_plateau"
    if float(max_grad) < float(eps_grad) and bool(finite_angle_flat):
        return True, "eps_grad_flat"
    if bool(repeated_family_flat):
        return True, "family_repeat_flat"
    return False, "default_append_only"


class StageController:
    def __init__(self, cfg: StageControllerConfig) -> None:
        self.cfg = cfg
        self._stage = "core"

    @property
    def stage_name(self) -> str:
        return str(self._stage)

    def start_with_seed(self) -> None:
        self._stage = "seed"

    def begin_core(self) -> None:
        self._stage = "core"

    def resolve_stage_transition(
        self,
        *,
        drop_plateau_hits: int,
        trough_detected: bool,
        residual_opened: bool,
    ) -> tuple[str, str]:
        if self._stage == "seed":
            self._stage = "core"
            return self._stage, "seed_complete"
        if self._stage == "core":
            if int(drop_plateau_hits) >= int(self.cfg.plateau_patience) and (not bool(trough_detected)):
                self._stage = "residual"
                return self._stage, "plateau_without_trough"
            return self._stage, "stay_core"
        if self._stage == "residual":
            if bool(residual_opened):
                return self._stage, "stay_residual"
            return self._stage, "residual_closed"
        return self._stage, "unknown_stage"

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/test/test_hh_continuation_pruning.py
```py
from __future__ import annotations

import numpy as np

from pipelines.hardcoded.hh_continuation_pruning import (
    apply_pruning,
    post_prune_refit,
    rank_prune_candidates,
)


def test_rank_prune_candidates_prefers_small_theta() -> None:
    theta = np.array([0.8, 0.01, 0.02, 0.5], dtype=float)
    labels = ["a", "b", "c", "d"]
    idx = rank_prune_candidates(
        theta=theta,
        labels=labels,
        marginal_proxy_benefit=None,
        max_candidates=3,
        min_candidates=2,
        fraction_candidates=0.5,
    )
    assert idx[0] == 1
    assert 2 in idx


def test_rank_prune_candidates_uses_proxy_benefit_as_tiebreak() -> None:
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


def test_apply_pruning_accepts_when_regression_small() -> None:
    theta = np.array([0.2, 0.1, 0.05], dtype=float)
    labels = ["a", "b", "c"]

    def _eval(idx_remove: int, theta_cur: np.ndarray) -> tuple[float, np.ndarray]:
        theta_new = np.delete(theta_cur, idx_remove)
        return float(np.sum(theta_new**2)), theta_new

    theta_out, labels_out, decisions, energy_out = apply_pruning(
        theta=theta,
        labels=labels,
        candidate_indices=[2],
        eval_with_removal=_eval,
        energy_before=float(np.sum(theta**2)),
        max_regression=1e-3,
    )
    assert len(theta_out) == 2
    assert labels_out == ["a", "b"]
    assert decisions[0].accepted is True
    assert energy_out <= float(np.sum(theta**2))


def test_post_prune_refit_returns_callback_result() -> None:
    theta = np.array([0.1, 0.2], dtype=float)
    theta_new, e = post_prune_refit(
        theta=theta,
        refit_fn=lambda x: (x * 0.5, float(np.sum((x * 0.5) ** 2))),
    )
    assert np.allclose(theta_new, [0.05, 0.1])
    assert e > 0.0

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/hardcoded/hh_continuation_pruning.py
```py
#!/usr/bin/env python3
"""Prune-before-replay helpers for HH continuation Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pipelines.hardcoded.hh_continuation_types import PruneDecision


@dataclass(frozen=True)
class PruneConfig:
    max_candidates: int = 6
    min_candidates: int = 2
    fraction_candidates: float = 0.25
    max_regression: float = 1e-8


def rank_prune_candidates(
    *,
    theta: np.ndarray,
    labels: list[str],
    marginal_proxy_benefit: list[float] | None,
    max_candidates: int,
    min_candidates: int,
    fraction_candidates: float,
) -> list[int]:
    n = int(theta.size)
    if n <= 0:
        return []
    target = int(np.ceil(float(fraction_candidates) * float(n)))
    target = max(int(min_candidates), target)
    target = min(int(max_candidates), target, n)
    benefits = list(marginal_proxy_benefit) if marginal_proxy_benefit is not None else []

    def _benefit_key(i: int) -> float:
        if i >= len(benefits):
            return float("inf")
        val = float(benefits[i])
        if not np.isfinite(val):
            return float("inf")
        return float(val)

    order = sorted(
        range(n),
        key=lambda i: (
            abs(float(theta[i])),
            _benefit_key(int(i)),
            str(labels[i]),
        ),
    )
    return [int(i) for i in order[:target]]


def apply_pruning(
    *,
    theta: np.ndarray,
    labels: list[str],
    candidate_indices: list[int],
    eval_with_removal: Callable[..., tuple[float, np.ndarray]],
    energy_before: float,
    max_regression: float,
) -> tuple[np.ndarray, list[str], list[PruneDecision], float]:
    cur_theta = np.asarray(theta, dtype=float).copy()
    cur_labels = list(labels)
    decisions: list[PruneDecision] = []
    cur_energy = float(energy_before)
    removed_so_far = 0

    for idx0 in candidate_indices:
        idx = int(idx0) - int(removed_so_far)
        if idx < 0 or idx >= len(cur_labels):
            continue
        try:
            trial_energy, trial_theta = eval_with_removal(idx, cur_theta, list(cur_labels))
        except TypeError:
            trial_energy, trial_theta = eval_with_removal(idx, cur_theta)
        regression = float(trial_energy - cur_energy)
        accepted = bool(regression <= float(max_regression))
        reason = "accepted" if accepted else "regression_exceeded"
        label = str(cur_labels[idx])
        decisions.append(
            PruneDecision(
                index=int(idx),
                label=label,
                accepted=bool(accepted),
                energy_before=float(cur_energy),
                energy_after=float(trial_energy),
                regression=float(regression),
                reason=str(reason),
            )
        )
        if accepted:
            cur_theta = np.asarray(trial_theta, dtype=float).copy()
            del cur_labels[idx]
            cur_energy = float(trial_energy)
            removed_so_far += 1

    return cur_theta, cur_labels, decisions, float(cur_energy)


def post_prune_refit(
    *,
    theta: np.ndarray,
    refit_fn: Callable[[np.ndarray], tuple[np.ndarray, float]],
) -> tuple[np.ndarray, float]:
    return refit_fn(np.asarray(theta, dtype=float))

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/hardcoded/hh_continuation_scoring.py
(lines 1-72: full-score config including lifetime and remaining-evaluations knobs)
```py
#!/usr/bin/env python3
"""Scoring and proxy accounting for HH continuation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from pipelines.hardcoded.hh_continuation_types import (
    CandidateFeatures,
    CompileCostEstimate,
    CurvatureOracle,
    MeasurementCacheStats,
    MeasurementPlan,
    NoveltyOracle,
)
from pipelines.hardcoded.hh_continuation_motifs import motif_bonus_for_generator
from src.quantum.compiled_polynomial import (
    CompiledPolynomialAction,
    apply_compiled_polynomial,
    compile_polynomial_action,
)


@dataclass(frozen=True)
class SimpleScoreConfig:
    lambda_F: float = 1.0
    lambda_compile: float = 0.05
    lambda_measure: float = 0.02
    lambda_leak: float = 0.0
    z_alpha: float = 0.0
    score_version: str = "simple_v1"


@dataclass(frozen=True)
class FullScoreConfig:
    z_alpha: float = 0.0
    lambda_F: float = 1.0
    lambda_H: float = 1e-6
    rho: float = 0.25
    eta_L: float = 0.0
    gamma_N: float = 1.0
    wD: float = 0.2
    wG: float = 0.15
    wC: float = 0.15
    wP: float = 0.1
    wc: float = 0.1
    depth_ref: float = 1.0
    group_ref: float = 1.0
    shot_ref: float = 1.0
    optdim_ref: float = 1.0
    reuse_ref: float = 1.0
    novelty_eps: float = 1e-6
    shortlist_fraction: float = 0.2
    shortlist_size: int = 12
    batch_target_size: int = 2
    batch_size_cap: int = 3
    batch_near_degenerate_ratio: float = 0.9
    compat_overlap_weight: float = 0.4
    compat_comm_weight: float = 0.2
    compat_curv_weight: float = 0.2
    compat_sched_weight: float = 0.2
    compat_measure_weight: float = 0.2
    leakage_cap: float = 1e6
    lifetime_cost_mode: str = "off"
    remaining_evaluations_proxy_mode: str = "none"
    lifetime_weight: float = 0.05
    motif_bonus_weight: float = 0.05
    score_version: str = "full_v2"

```

(lines 355-444: current remaining_evaluations_proxy and lifetime/full_v2 scoring behavior)
```py
def remaining_evaluations_proxy(
    *,
    current_depth: int | None,
    max_depth: int | None,
    mode: str,
) -> float:
    mode_key = str(mode).strip().lower()
    if mode_key == "none":
        return 0.0
    depth_now = 0 if current_depth is None else int(max(0, current_depth))
    depth_cap = depth_now if max_depth is None else int(max(depth_now, max_depth))
    if mode_key == "remaining_depth":
        return float(max(1, depth_cap - depth_now + 1))
    raise ValueError("remaining_evaluations_proxy_mode must be 'none' or 'remaining_depth'")


def lifetime_weight_components(
    feat: CandidateFeatures,
    cfg: FullScoreConfig,
) -> dict[str, float]:
    if str(cfg.lifetime_cost_mode).strip().lower() == "off":
        return {
            "remaining_evaluations_proxy": float(feat.remaining_evaluations_proxy),
            "compiled": 0.0,
            "measurement": 0.0,
            "optimizer_dim": 0.0,
            "total": 0.0,
        }
    rem = float(max(0.0, feat.remaining_evaluations_proxy))
    compiled = rem * normalize(float(feat.depth_cost), float(cfg.depth_ref))
    measurement = rem * (
        normalize(float(feat.new_group_cost), float(cfg.group_ref))
        + normalize(float(feat.new_shot_cost), float(cfg.shot_ref))
        + normalize(float(feat.reuse_count_cost), float(cfg.reuse_ref))
    )
    optimizer_dim = rem * normalize(float(feat.opt_dim_cost), float(cfg.optdim_ref))
    total = compiled + measurement + optimizer_dim
    return {
        "remaining_evaluations_proxy": float(rem),
        "compiled": float(compiled),
        "measurement": float(measurement),
        "optimizer_dim": float(optimizer_dim),
        "total": float(total),
    }


def full_v2_score(
    feat: CandidateFeatures,
    cfg: FullScoreConfig,
) -> tuple[float, str]:
    if (not bool(feat.stage_gate_open)) or (not bool(feat.leakage_gate_open)):
        return float("-inf"), "blocked_stage_or_leakage_gate"
    if not bool(feat.compile_gate_open):
        return float("-inf"), "compile_gate_closed"
    if float(feat.leakage_penalty) > float(cfg.leakage_cap):
        return float("-inf"), "leakage_cap"

    g_lcb = max(float(feat.g_abs) - float(cfg.z_alpha) * float(max(0.0, feat.sigma_hat)), 0.0)
    if g_lcb <= 0.0 or float(feat.F_metric) <= 0.0:
        return 0.0, "nonpositive_gradient_or_metric"

    novelty = 1.0 if feat.novelty is None else min(max(float(feat.novelty), 0.0), 1.0)
    fallback_mode = "full"
    h_eff = float(cfg.lambda_F) * float(feat.F_metric)
    if feat.h_hat is None:
        fallback_mode = "lambda_F_metric_only"
    elif feat.b_hat is None or feat.H_window is None:
        h_eff = float(max(0.0, feat.h_hat))
        fallback_mode = "self_curvature_only"
    else:
        try:
            b_vec = np.asarray(feat.b_hat, dtype=float).reshape(-1)
            H_mat = np.asarray(feat.H_window, dtype=float)
            if H_mat.ndim != 2 or H_mat.shape[0] != H_mat.shape[1] or H_mat.shape[0] != b_vec.size:
                raise ValueError("invalid_shape")
            H_reg = H_mat + float(cfg.lambda_H) * np.eye(H_mat.shape[0], dtype=float)
            correction = float(b_vec.T @ np.linalg.solve(H_reg, b_vec))
            h_eff = float(max(0.0, float(feat.h_hat) - correction))
        except Exception:
            h_eff = float(max(0.0, feat.h_hat))
            fallback_mode = "curvature_solve_failed"

    delta_e = trust_region_drop(g_lcb, h_eff, float(feat.F_metric), float(cfg.rho))
    if delta_e <= 0.0:
        return 0.0, fallback_mode

    K = (
        1.0
        + float(cfg.wD) * normalize(float(feat.depth_cost), float(cfg.depth_ref))
        + float(cfg.wG) * normalize(float(feat.new_group_cost), float(cfg.group_ref))

```

(lines 1016-1075: candidate feature construction and current remaining_eval_proxy wiring)
```py
    motif_source: str = "none",
    current_depth: int | None = None,
    max_depth: int | None = None,
    lifetime_cost_mode: str = "off",
    remaining_evaluations_proxy_mode: str = "none",
) -> CandidateFeatures:
    """Built-in math expression:
    g_lcb = max(|g| - z_alpha * sigma_hat, 0)
    """
    g_abs = float(abs(float(gradient_signed)))
    g_lcb = max(g_abs - float(cfg.z_alpha) * float(max(0.0, sigma_hat)), 0.0)
    remaining_eval_proxy = remaining_evaluations_proxy(
        current_depth=current_depth,
        max_depth=max_depth,
        mode=str(remaining_evaluations_proxy_mode),
    )
    proxy_cost = (
        dict(compile_cost.proxy_baseline)
        if isinstance(compile_cost.proxy_baseline, Mapping)
        else {
            "new_pauli_actions": float(compile_cost.new_pauli_actions),
            "new_rotation_steps": float(compile_cost.new_rotation_steps),
            "position_shift_span": float(compile_cost.position_shift_span),
            "refit_active_count": float(compile_cost.refit_active_count),
            "cx_proxy_total": float(compile_cost.cx_proxy_total),
            "sq_proxy_total": float(compile_cost.sq_proxy_total),
            "gate_proxy_total": float(compile_cost.gate_proxy_total),
            "max_pauli_weight": float(compile_cost.max_pauli_weight),
            "proxy_total": float(compile_cost.proxy_total),
        }
    )
    compile_cost_total = (
        float(compile_cost.penalty_total)
        if compile_cost.penalty_total is not None
        else float(compile_cost.proxy_total)
    )
    depth_cost_value = (
        float(compile_cost.depth_surrogate)
        if compile_cost.depth_surrogate is not None
        else float(
            (
                float(proxy_cost.get("gate_proxy_total", 0.0))
                if float(proxy_cost.get("gate_proxy_total", 0.0)) > 0.0
                else float(proxy_cost.get("new_rotation_steps", 0.0))
            )
            + float(proxy_cost.get("position_shift_span", 0.0))
        )
    )
    feat = CandidateFeatures(
        stage_name=str(stage_name),
        candidate_label=str(candidate_label),
        candidate_family=str(candidate_family),
        candidate_pool_index=int(candidate_pool_index),
        position_id=int(position_id),
        append_position=int(append_position),
        positions_considered=[int(x) for x in positions_considered],
        g_signed=float(gradient_signed),
        g_abs=float(g_abs),
        g_lcb=float(g_lcb),
        sigma_hat=float(max(0.0, sigma_hat)),

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/test/test_adapt_vqe_integration.py
(lines 1437-1569: eps-energy gate and patience behavior that should inform remaining-work estimation)
```py
class TestAdaptEnergyStopGate:
    """eps_energy stop must honor min-extra-depth and patience gates."""

    def test_eps_energy_defaults_wait_for_L_gate_and_L_patience(self, monkeypatch: pytest.MonkeyPatch):
        h_poly = build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        events: list[tuple[str, dict[str, object]]] = []
        original_ai_log = _adapt_mod._ai_log

        def _capture(event: str, **fields: object) -> None:
            events.append((str(event), dict(fields)))

        monkeypatch.setattr(_adapt_mod, "_ai_log", _capture)
        try:
            payload, _ = _run_hardcoded_adapt_vqe(
                h_poly=h_poly,
                num_sites=2,
                ordering="blocked",
                problem="hubbard",
                adapt_pool="uccsd",
                t=1.0,
                u=4.0,
                dv=0.0,
                boundary="periodic",
                omega0=0.0,
                g_ep=0.0,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=4,
                eps_grad=-1.0,
                eps_energy=1e9,
                maxiter=20,
                seed=19,
                adapt_inner_optimizer="SPSA",
                adapt_spsa_callback_every=10,
                adapt_spsa_progress_every_s=999.0,
                allow_repeats=True,
                finite_angle_fallback=False,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
            )
            assert payload["success"] is True
            assert str(payload["stop_reason"]) == "eps_energy"
            assert bool(payload["eps_energy_termination_enabled"]) is True
            assert int(payload["eps_energy_min_extra_depth_effective"]) == 2
            assert int(payload["eps_energy_patience_effective"]) == 2
            assert int(payload["ansatz_depth"]) >= 3

            iter_done_events = [ev[1] for ev in events if ev[0] == "hardcoded_adapt_iter_done"]
            by_depth = {int(ev["depth"]): ev for ev in iter_done_events}
            assert bool(by_depth[1]["eps_energy_gate_open"]) is False
            assert bool(by_depth[2]["eps_energy_gate_open"]) is True
            assert int(by_depth[2]["eps_energy_low_streak"]) == 1
            assert int(by_depth[3]["eps_energy_low_streak"]) >= 2
            assert bool(by_depth[3]["eps_energy_termination_enabled"]) is True

            gate_wait_events = [ev for ev in events if ev[0] == "hardcoded_adapt_energy_convergence_gate_wait"]
            assert len(gate_wait_events) >= 1
        finally:
            monkeypatch.setattr(_adapt_mod, "_ai_log", original_ai_log)

    def test_eps_energy_gate_override_is_respected(self, monkeypatch: pytest.MonkeyPatch):
        h_poly = build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        events: list[tuple[str, dict[str, object]]] = []
        original_ai_log = _adapt_mod._ai_log

        def _capture(event: str, **fields: object) -> None:
            events.append((str(event), dict(fields)))

        monkeypatch.setattr(_adapt_mod, "_ai_log", _capture)
        try:
            payload, _ = _run_hardcoded_adapt_vqe(
                h_poly=h_poly,
                num_sites=2,
                ordering="blocked",
                problem="hubbard",
                adapt_pool="uccsd",
                t=1.0,
                u=4.0,
                dv=0.0,
                boundary="periodic",
                omega0=0.0,
                g_ep=0.0,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=5,
                eps_grad=-1.0,
                eps_energy=1e9,
                maxiter=20,
                seed=21,
                adapt_inner_optimizer="SPSA",
                adapt_spsa_callback_every=10,
                adapt_spsa_progress_every_s=999.0,
                allow_repeats=True,
                finite_angle_fallback=False,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_eps_energy_min_extra_depth=3,
                adapt_eps_energy_patience=2,
            )
            assert payload["success"] is True
            assert str(payload["stop_reason"]) == "eps_energy"
            assert bool(payload["eps_energy_termination_enabled"]) is True
            assert int(payload["eps_energy_min_extra_depth_effective"]) == 3
            assert int(payload["eps_energy_patience_effective"]) == 2
            assert int(payload["ansatz_depth"]) >= 4

            iter_done_events = [ev[1] for ev in events if ev[0] == "hardcoded_adapt_iter_done"]
            by_depth = {int(ev["depth"]): ev for ev in iter_done_events}
            assert bool(by_depth[2]["eps_energy_gate_open"]) is False
            assert bool(by_depth[3]["eps_energy_gate_open"]) is True
            assert int(by_depth[3]["eps_energy_low_streak"]) == 1
            assert int(by_depth[4]["eps_energy_low_streak"]) >= 2
            assert bool(by_depth[4]["eps_energy_termination_enabled"]) is True

            converged_energy = [ev[1] for ev in events if ev[0] == "hardcoded_adapt_converged_energy"]
            assert len(converged_energy) == 1
            assert int(converged_energy[0]["eps_energy_min_extra_depth"]) == 3
            assert int(converged_energy[0]["eps_energy_patience"]) == 2
        finally:
            monkeypatch.setattr(_adapt_mod, "_ai_log", original_ai_log)


# ============================================================================
# ADAPT re-optimization policy tests
# ============================================================================

class TestAdaptReoptPolicyAppendOnly:

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/hardcoded/adapt_pipeline.py
(lines 1737-1818: resolved stop-policy defaults and staged-HH telemetry signals)
```py
def _resolve_adapt_stop_policy(
    *,
    problem: str,
    continuation_mode: str,
    adapt_drop_floor: float | None,
    adapt_drop_patience: int | None,
    adapt_drop_min_depth: int | None,
    adapt_grad_floor: float | None,
) -> ResolvedAdaptStopPolicy:
    staged_hh = bool(
        str(problem).strip().lower() == "hh"
        and str(continuation_mode).strip().lower() in _HH_STAGED_CONTINUATION_MODES
    )

    def _resolve_float(raw: float | None, *, staged_value: float, default_value: float) -> tuple[float, str]:
        if raw is None:
            if staged_hh:
                return float(staged_value), "auto_hh_staged"
            return float(default_value), "default_off"
        return float(raw), "explicit"

    def _resolve_int(raw: int | None, *, staged_value: int, default_value: int) -> tuple[int, str]:
        if raw is None:
            if staged_hh:
                return int(staged_value), "auto_hh_staged"
            return int(default_value), "default_off"
        return int(raw), "explicit"

    drop_floor_resolved, drop_floor_source = _resolve_float(
        adapt_drop_floor,
        staged_value=5e-4,
        default_value=-1.0,
    )
    drop_patience_resolved, drop_patience_source = _resolve_int(
        adapt_drop_patience,
        staged_value=3,
        default_value=0,
    )
    drop_min_depth_resolved, drop_min_depth_source = _resolve_int(
        adapt_drop_min_depth,
        staged_value=12,
        default_value=0,
    )
    grad_floor_resolved, grad_floor_source = _resolve_float(
        adapt_grad_floor,
        staged_value=2e-2,
        default_value=-1.0,
    )
    drop_policy_enabled = bool(drop_floor_resolved >= 0.0 and drop_patience_resolved > 0)
    if staged_hh and all(src == "auto_hh_staged" for src in (
        drop_floor_source,
        drop_patience_source,
        drop_min_depth_source,
        grad_floor_source,
    )):
        drop_policy_source = "auto_hh_staged"
    elif any(src == "explicit" for src in (
        drop_floor_source,
        drop_patience_source,
        drop_min_depth_source,
        grad_floor_source,
    )):
        drop_policy_source = "explicit"
    else:
        drop_policy_source = "default_off"

    return ResolvedAdaptStopPolicy(
        adapt_drop_floor=float(drop_floor_resolved),
        adapt_drop_patience=int(drop_patience_resolved),
        adapt_drop_min_depth=int(drop_min_depth_resolved),
        adapt_grad_floor=float(grad_floor_resolved),
        adapt_drop_floor_source=str(drop_floor_source),
        adapt_drop_patience_source=str(drop_patience_source),
        adapt_drop_min_depth_source=str(drop_min_depth_source),
        adapt_grad_floor_source=str(grad_floor_source),
        drop_policy_enabled=bool(drop_policy_enabled),
        drop_policy_source=str(drop_policy_source),
        eps_energy_termination_enabled=(not staged_hh),
        eps_grad_termination_enabled=(not staged_hh),
    )



```

(lines 5413-5592: current post-loop prune pass, candidate ranking, removal callback, and optimizer-memory remap)
```py
        def _reconstruct_phase1_proxy_benefits() -> list[float]:
            benefits: list[float] = []
            for row in history:
                if not isinstance(row, dict):
                    continue
                if row.get("continuation_mode") not in {"phase1_v1", "phase2_v1", "phase3_v1"}:
                    continue
                pos = int(row.get("selected_position", len(benefits)))
                pos = max(0, min(len(benefits), pos))
                benefit = row.get("full_v2_score", row.get("simple_score", None))
                if benefit is None:
                    benefit = row.get("metric_proxy", row.get("selected_grad_abs", float("inf")))
                benefit_f = float(benefit)
                if not math.isfinite(benefit_f):
                    benefit_f = float("inf")
                benefits.insert(pos, benefit_f)
            while len(benefits) < int(len(selected_ops)):
                benefits.append(float("inf"))
            return [float(x) for x in benefits[: int(len(selected_ops))]]

        pre_prune_ops = list(selected_ops)
        pre_prune_layout = _build_selected_layout(pre_prune_ops)
        pre_prune_theta = np.asarray(theta, dtype=float).copy()
        pre_prune_energy = float(energy_current)
        pre_prune_memory = dict(phase2_optimizer_memory) if phase2_enabled else None
        pre_prune_generator_meta = (
            selected_generator_metadata_for_labels(
                [str(op.label) for op in pre_prune_ops],
                pool_generator_registry,
            )
            if phase3_enabled
            else []
        )
        prune_proxy_benefit = _reconstruct_phase1_proxy_benefits()
        candidate_indices = rank_prune_candidates(
            theta=np.asarray(_logical_theta_alias(theta, selected_layout), dtype=float),
            labels=[str(op.label) for op in selected_ops],
            marginal_proxy_benefit=list(prune_proxy_benefit),
            max_candidates=int(prune_cfg.max_candidates),
            min_candidates=int(prune_cfg.min_candidates),
            fraction_candidates=float(prune_cfg.fraction_candidates),
        )
        prune_summary["candidate_count"] = int(len(candidate_indices))
        prune_summary["marginal_proxy_benefit"] = [float(x) for x in prune_proxy_benefit]

        def _refit_given_ops(ops_refit: list[AnsatzTerm], theta0: np.ndarray) -> tuple[np.ndarray, float]:
            nonlocal phase2_optimizer_memory
            if len(ops_refit) == 0:
                return np.zeros(0, dtype=float), float(energy_current)
            executor_refit = _build_compiled_executor(ops_refit) if adapt_state_backend_key == "compiled" else None

            def _obj_prune(x: np.ndarray) -> float:
                if executor_refit is not None:
                    psi_obj = executor_refit.prepare_state(np.asarray(x, dtype=float), psi_ref)
                    e_obj, _ = energy_via_one_apply(psi_obj, h_compiled)
                    return float(e_obj)
                return float(
                    _adapt_energy_fn(
                        h_poly,
                        psi_ref,
                        ops_refit,
                        x,
                        h_compiled=h_compiled,
                        parameter_layout=_build_selected_layout(ops_refit),
                    )
                )

            x0 = np.asarray(theta0, dtype=float).reshape(-1)
            if adapt_inner_optimizer_key == "SPSA":
                refit_memory = None
                if phase2_enabled:
                    refit_memory = phase2_memory_adapter.select_active(
                        phase2_optimizer_memory,
                        active_indices=list(range(int(x0.size))),
                        source="adapt.post_prune_refit.active_subset",
                    )
                res = spsa_minimize(
                    fun=_obj_prune,
                    x0=x0,
                    maxiter=int(max(25, min(int(maxiter), 120))),
                    seed=int(seed) + 700000 + int(len(ops_refit)),
                    a=float(adapt_spsa_a),
                    c=float(adapt_spsa_c),
                    alpha=float(adapt_spsa_alpha),
                    gamma=float(adapt_spsa_gamma),
                    A=float(adapt_spsa_A),
                    bounds=None,
                    project="none",
                    eval_repeats=int(adapt_spsa_eval_repeats),
                    eval_agg=str(adapt_spsa_eval_agg_key),
                    avg_last=int(adapt_spsa_avg_last),
                    memory=(dict(refit_memory) if isinstance(refit_memory, Mapping) else None),
                    refresh_every=0,
                    precondition_mode=("diag_rms_grad" if phase2_enabled else "none"),
                )
                if phase2_enabled:
                    phase2_optimizer_memory = phase2_memory_adapter.merge_active(
                        phase2_optimizer_memory,
                        active_indices=list(range(int(x0.size))),
                        active_state=phase2_memory_adapter.from_result(
                            res,
                            method=str(adapt_inner_optimizer_key),
                            parameter_count=int(x0.size),
                            source="adapt.post_prune_refit.result",
                        ),
                        source="adapt.post_prune_refit.merge",
                    )
                return np.asarray(res.x, dtype=float), float(res.fun)
            if scipy_minimize is None:
                raise RuntimeError("SciPy minimize is unavailable for prune refit.")
            res = scipy_minimize(
                _obj_prune,
                x0,
                method="COBYLA",
                options={"maxiter": int(max(25, min(int(maxiter), 120))), "rhobeg": 0.3},
            )
            return np.asarray(res.x, dtype=float), float(res.fun)

        def _ops_from_labels(labels_cur: list[str]) -> list[AnsatzTerm]:
            buckets: dict[str, list[AnsatzTerm]] = {}
            for op_ref in selected_ops:
                key_ref = str(op_ref.label)
                buckets.setdefault(key_ref, []).append(op_ref)
            rebuilt: list[AnsatzTerm] = []
            for lbl in labels_cur:
                key_lbl = str(lbl)
                if key_lbl not in buckets or len(buckets[key_lbl]) == 0:
                    continue
                rebuilt.append(buckets[key_lbl].pop(0))
            return rebuilt

        def _eval_with_removal(
            idx_remove: int,
            theta_cur: np.ndarray,
            labels_cur: list[str],
        ) -> tuple[float, np.ndarray]:
            ops_current = _ops_from_labels(list(labels_cur))
            layout_current = _build_selected_layout(ops_current)
            runtime_remove_indices = runtime_indices_for_logical_indices(layout_current, [int(idx_remove)])
            ops_trial = list(ops_current)
            del ops_trial[int(idx_remove)]
            theta_trial0 = np.delete(np.asarray(theta_cur, dtype=float), runtime_remove_indices)
            theta_trial_opt, e_trial = _refit_given_ops(ops_trial, theta_trial0)
            return float(e_trial), np.asarray(theta_trial_opt, dtype=float)

        theta_pruned, labels_pruned, prune_decisions, energy_after_prune = apply_pruning(
            theta=np.asarray(theta, dtype=float),
            labels=[str(op.label) for op in selected_ops],
            candidate_indices=[int(i) for i in candidate_indices],
            eval_with_removal=_eval_with_removal,
            energy_before=float(energy_current),
            max_regression=float(prune_cfg.max_regression),
        )
        accepted_count = int(sum(1 for d in prune_decisions if bool(d.accepted)))
        if accepted_count > 0:
            accepted_remove_indices = [int(d.index) for d in prune_decisions if bool(d.accepted)]
            accepted_runtime_remove_indices = runtime_indices_for_logical_indices(
                pre_prune_layout,
                accepted_remove_indices,
            )
            if phase2_enabled:
                phase2_optimizer_memory = phase2_memory_adapter.remap_remove(
                    phase2_optimizer_memory,
                    indices=list(accepted_runtime_remove_indices),
                )
            label_to_ops: dict[str, list[AnsatzTerm]] = {}
            for op in selected_ops:
                key = str(op.label)
                label_to_ops.setdefault(key, []).append(op)
            rebuilt_ops: list[AnsatzTerm] = []
            for lbl in labels_pruned:
                key = str(lbl)
                bucket = label_to_ops.get(key, [])
                if not bucket:
                    continue
                rebuilt_ops.append(bucket.pop(0))
            selected_ops = list(rebuilt_ops)
            selected_layout = _build_selected_layout(selected_ops)
            theta = np.asarray(theta_pruned, dtype=float)
            energy_current = float(energy_after_prune)

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/hardcoded/hh_continuation_types.py
(lines 1-110: candidate feature schema including runtime split, symmetry, and lifetime fields)
```py
#!/usr/bin/env python3
"""Shared continuation datamodel for HH ADAPT -> replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class CandidateFeatures:
    stage_name: str
    candidate_label: str
    candidate_family: str
    candidate_pool_index: int
    position_id: int
    append_position: int
    positions_considered: list[int]
    g_signed: float
    g_abs: float
    g_lcb: float
    sigma_hat: float
    F_metric: float
    metric_proxy: float
    novelty: float | None
    curvature_mode: str
    novelty_mode: str
    refit_window_indices: list[int]
    compiled_position_cost_proxy: dict[str, float]
    measurement_cache_stats: dict[str, float]
    leakage_penalty: float
    stage_gate_open: bool
    leakage_gate_open: bool
    trough_probe_triggered: bool
    trough_detected: bool
    simple_score: float | None
    score_version: str
    h_hat: float | None = None
    b_hat: list[float] | None = None
    H_window: list[list[float]] | None = None
    depth_cost: float = 0.0
    new_group_cost: float = 0.0
    new_shot_cost: float = 0.0
    opt_dim_cost: float = 0.0
    reuse_count_cost: float = 0.0
    full_v2_score: float | None = None
    shortlist_rank: int | None = None
    shortlist_size: int | None = None
    actual_fallback_mode: str = "simple_v1_only"
    compatibility_penalty_total: float = 0.0
    generator_id: str | None = None
    template_id: str | None = None
    is_macro_generator: bool = False
    parent_generator_id: str | None = None
    runtime_split_mode: str = "off"
    runtime_split_parent_label: str | None = None
    runtime_split_child_index: int | None = None
    runtime_split_child_count: int | None = None
    runtime_split_chosen_representation: str = "parent"
    runtime_split_child_indices: list[int] = field(default_factory=list)
    runtime_split_child_labels: list[str] = field(default_factory=list)
    runtime_split_child_generator_ids: list[str] = field(default_factory=list)
    generator_metadata: dict[str, Any] | None = None
    symmetry_spec: dict[str, Any] | None = None
    symmetry_mode: str = "none"
    symmetry_mitigation_mode: str = "off"
    motif_metadata: dict[str, Any] | None = None
    motif_bonus: float = 0.0
    motif_source: str = "none"
    remaining_evaluations_proxy: float = 0.0
    remaining_evaluations_proxy_mode: str = "none"
    lifetime_cost_mode: str = "off"
    lifetime_weight_components: dict[str, float] = field(default_factory=dict)
    placeholder_hooks: dict[str, bool] = field(default_factory=dict)
    compile_cost_source: str = "proxy"
    compile_cost_total: float = 0.0
    compile_gate_open: bool = True
    compile_failure_reason: str | None = None
    compiled_position_cost_backend: dict[str, Any] | None = None


@dataclass(frozen=True)
class MeasurementPlan:
    plan_version: str
    group_keys: list[str]
    nominal_shots_per_group: int
    grouping_mode: str


@dataclass(frozen=True)
class MeasurementCacheStats:
    groups_total: int
    groups_reused: int
    groups_new: int
    shots_reused: float
    shots_new: float
    reuse_count_cost: float


@dataclass(frozen=True)
class CompileCostEstimate:
    new_pauli_actions: float
    new_rotation_steps: float
    position_shift_span: float
    refit_active_count: float
    proxy_total: float
    cx_proxy_total: float = 0.0
    sq_proxy_total: float = 0.0
    gate_proxy_total: float = 0.0
    max_pauli_weight: float = 0.0

```

(lines 145-158: prune decision schema)
```py
    generator_family: str
    continuation_mode: str
    compiled_pauli_cache_size: int
    measurement_plan_version: str
    post_prune: bool
    split_event_count: int = 0
    motif_record_ids: list[str] = field(default_factory=list)
    compile_cost_mode: str = "proxy"
    backend_target_names: list[str] = field(default_factory=list)
    backend_reduction_mode: str = "none"


@dataclass(frozen=True)
class PruneDecision:

```
</file_contents>
<meta prompt 1 = "[Architect]">
You are producing an implementation-ready technical plan. The implementer will work from your plan without asking clarifying questions, so every design decision must be resolved, every touched component must be identified, and every behavioral change must be specified precisely.

Your job:
1. Analyze the requested change against the provided code — identify the relevant architecture, constraints, data flow, and extension points.
2. Decide whether this is best solved by a targeted change or a broader refactor, and justify that decision.
3. Produce a plan detailed enough that an engineer can implement it file-by-file without making design decisions of their own.

Hard constraints:
- Do not write production code, patches, diffs, or copy-paste-ready implementations.
- Stay in analysis and architecture mode only.
- Use illustrative snippets, interface shapes, sample signatures, state/data shapes, or pseudocode when they communicate the design more precisely than prose. Keep them partial — enough to remove ambiguity, not enough to copy-paste.

─── ANALYSIS ───

Current-state analysis (always include):
- Map the existing responsibilities, type relationships, ownership, data flow, and mutation points relevant to the request.
- Identify existing code that should be reused or extended — never duplicate what already exists without justification.
- Note hard constraints: API contracts, protocol conformances, state ownership rules, thread/actor isolation, persistence schemas, UI update mechanisms.
- When multiple subsystems interact, trace the call chain end-to-end and identify each transformation boundary.

─── DESIGN ───

Design standards — apply uniformly to every aspect of the plan:

1. New and modified components/types: For each, specify:
   - The name, kind (for example: class, interface, enum, record, service, module, controller), and why that kind fits the codebase and language.
   - The fields/properties/state it owns, including data shape, mutability, and ownership/lifecycle semantics.
   - Key callable interfaces or signatures, including inputs, outputs, and whether execution is synchronous/asynchronous or can fail.
   - Contracts it implements, extends, composes with, or depends on.
   - For closed sets of variants (for example enums, tagged unions, discriminated unions): all cases/variants and any attached data.
   - Where the component lives (file path) and who creates/owns its instances.

2. State and data flow: For each state change the plan introduces or modifies:
   - What triggers the change (user action, callback, notification, timer, stream event).
   - The exact path the data travels: source → transformations → destination.
   - Thread/actor/queue context at each step.
   - How downstream consumers observe the change (published property, delegate, notification, binding, callback).
   - What happens if the change arrives out of order, is duplicated, or is dropped.

3. API and interface changes: For each modified public/internal interface:
   - The before and after signatures (or new signature if additive).
   - Every call site that must be updated, grouped by file.
   - Backward-compatibility strategy if the interface is used by external consumers or persisted data.

4. Persistence and serialization: When the plan touches stored data:
   - Schema changes with exact field names, types, and defaults.
   - Migration strategy: how existing data is read, transformed, and re-persisted.
   - What happens when new code reads old data and when old code reads new data (if rollback is possible).

5. Concurrency and lifecycle:
   - Specify the execution model and safety boundaries for each new/modified component: thread affinity, event-loop/runtime constraints, isolation boundaries, queue/worker discipline, or thread-safety expectations as applicable.
   - Identify potential races, leaked references/resources, or lifecycle mismatches introduced by the change.
   - When operations are asynchronous, specify cancellation/abort behavior and what state remains after interruption.

6. Error handling and edge cases:
   - For each operation that can fail, specify what failures are possible and how they propagate.
   - Describe degraded-mode behavior: what the user sees, what state is preserved, what recovery is available.
   - Identify boundary conditions: empty collections, missing/null/optional values, first-run states, interrupted operations.

7. Algorithmic and logic-heavy work (include whenever the change involves non-trivial control flow, state machines, data transformations, or performance-sensitive paths):
   - Describe the algorithm step-by-step: inputs, outputs, invariants, and data structures.
   - Cover edge cases, failure modes, and performance characteristics (time/space complexity if relevant).
   - Explain why this approach over the most plausible alternatives.

8. Avoid unnecessary complexity:
   - Do not add layers, abstractions, or indirection without a concrete benefit identified in the plan.
   - Do not create parallel code paths — unify where possible.
   - Reuse existing patterns unless those patterns are themselves the problem.

─── OUTPUT ───

Structure your response as:

1. **Summary** — One paragraph: what changes, why, and the high-level approach.

2. **Current-state analysis** — How the relevant code works today. Trace the data/control flow end-to-end. Identify what is reusable and what is blocking.

3. **Design** — The core of the plan. Apply every applicable standard from above. Organize by logical component or subsystem, not by standard number. Each component section should cover types, state flow, interfaces, persistence, concurrency, and error handling as relevant to that component.

4. **File-by-file impact** — For every file that changes, list:
   - What changes (added/modified/removed types, methods, properties).
   - Why (which design decision drives this change).
   - Dependencies on other changes in this plan (ordering constraints).

5. **Trade-offs and alternatives** — What was considered and rejected, and why. Include the cost/benefit of the chosen approach vs. the runner-up.

6. **Risks and migration** — Breaking changes, rollback concerns, data migration, feature flags, and incremental delivery strategy if the change is large.

7. **Implementation order** — A numbered sequence of steps. Each step should be independently compilable and testable where possible. Call out steps that must be atomic (landed together).

Response discipline:
- Be specific to the provided code — reference actual type names, file paths, method names, and property names.
- Make every assumption explicit.
- Flag unknowns that must be validated during implementation, with a suggested validation approach.
- When a design decision has a non-obvious rationale, explain it in one sentence.
- Do not pad with generic advice. Every sentence should convey information the implementer needs.

Please proceed with your analysis based on the following <user instructions>
</meta prompt 1>
<user_instructions>
# Phase3 live-pruning design question

You are helping with one narrow design change inside this repo.

## Scope guardrail
Do **not** answer this as if the task is to redesign or stabilize the entire staged HH ADAPT continuation system. That broader system already exists. Do **not** propose a broad refactor, class rewrite, runtime-parameterization overhaul, or general architecture migration unless a tiny local change is impossible.

Focus only on this question:

**How should ADAPT-VQE `phase3` perform live / on-the-fly pruning of already-admitted operators while the run is still in progress?**

## Current situation
- The repo already has staged HH continuation, phase1/phase2/phase3 selection, stop-policy telemetry, generator metadata, symmetry metadata, rescue logic, and a current prune helper.
- The current prune helper is effectively **post-loop** pruning, not live phase3 pruning.
- `rank_prune_candidates(...)` currently ranks mainly by small `|theta|`, with marginal proxy benefit only as a tiebreak.
- `remaining_evaluations_proxy(...)` currently only supports `none` and `remaining_depth`, i.e. a simple function of current depth vs max depth.
- Candidate/history features already carry useful signals: `full_v2_score`, `remaining_evaluations_proxy`, generator metadata, runtime-split metadata, symmetry fields, and stop-telemetry-related facts.
- In staged HH, hard `eps_energy` / `eps_grad` termination is suppressed, but telemetry around improvement gates and low-streak behavior still exists and is tested.

## What I actually want
I want a **phase3 live-pruning heuristic** that can remove previously admitted operators when:
1. the operator's optimized `theta` is very small, **and**
2. the operator's direction in state space appears structurally insignificant / stale.

But the pruning pressure should be **context-aware**:
- If the run still appears to need many more operators, pruning should be near-zero or very conservative.
- If the run looks close to saturation / plateau, pruning should become more willing to act.

### Important nuance
I do **not** want "expected remaining depth" to be estimated from `max_depth - current_depth` alone.
I want it inferred from the repo's existing improvement signals, such as:
- drop-floor / patience / min-depth logic
- recent absolute-energy-drop trend
- eps-energy low-streak or gate-open telemetry if useful
- shortlist flatness / weak improvement behavior
- plateau-style patience windows already conceptually present

### Mild age bias idea
I am considering a **weak** scaffold-age prior:
- earlier scaffold generators slightly less likely to be pruned
- later scaffold generators slightly more likely to be pruned

But this should be a soft regularizer, not a dominant rule, because later operators may still be structurally important. I want your opinion on whether this age bias is actually useful, and if so how weak it should be.

## Constraints
- Preserve the cached production gradient path in `pipelines/hardcoded/adapt_pipeline.py`.
- Prefer a local extension of existing helper modules over a broad refactor.
- Do not change operator-core conventions or JW ordering rules.
- Do not introduce Qiskit into the core ADAPT / VQE path.
- Treat the selected files as the current source of truth for where this heuristic would hook in.

## What I want from you
Please give a **narrow, implementation-oriented design** for this live-pruning feature.

Specifically answer:

1. **Heuristic formulation**
   - Propose a concrete prune score / prune probability / prune gate.
   - Show how to combine:
     - small `|theta|`
     - structural insignificance / stale-direction evidence
     - expected remaining work / expected remaining useful terms
     - optional weak age bias
   - If possible, write the heuristic as a simple formula or pseudocode.

2. **Structural insignificance proxy**
   - Given the current repo signals, what is the best cheap proxy for "this operator's direction is no longer structurally important"?
   - For example: low marginal proxy benefit, low novelty, low recent contribution, weak reopt sensitivity, repeated-family stagnation, or something else.
   - Prefer proxies that fit the current code shape and do not require expensive new inner-loop machinery.

3. **Remaining-work estimator**
   - Propose a better replacement or extension for `remaining_evaluations_proxy(mode="remaining_depth")`.
   - It should use improvement / plateau / patience-window signals rather than only `max_depth`.
   - Suggest whether this should be a scalar estimator, a normalized pressure term, or a few discrete regimes.

4. **Where it should live**
   - Which parts belong in:
     - `hh_continuation_pruning.py`
     - `hh_continuation_scoring.py`
     - `hh_continuation_types.py`
     - `adapt_pipeline.py`
   - Keep the answer local to the current architecture.

5. **When pruning should happen inside phase3**
   - After each accepted insertion?
   - Only after weak-improvement steps?
   - Only after patience-window evidence accumulates?
   - Should pruning be single-op only, or permit a tiny candidate set?

6. **Safeguards**
   - What guardrails should prevent over-pruning?
   - Examples: cooldown after insertion, minimum operator age, energy-regression cap, symmetry verification, no-prune regime while remaining-work estimate is high, etc.

7. **Age-bias critique**
   - Should early-vs-late scaffold position be used at all?
   - If yes, should it be a tiny additive prior, a multiplicative dampener, or only a tie-break?
   - Please be opinionated here.

8. **Validation plan**
   - What tests should be added or updated?
   - Please focus on targeted tests for:
     - no live prune when remaining-work pressure is high
     - more willingness to prune when improvement plateaus
     - age bias stays weak
     - pruning does not break runtime/logical index consistency
     - pruning preserves existing post-prune safety expectations

## Preferred answer shape
Please answer in this structure:
1. Short thesis: 3-6 bullets
2. Recommended heuristic
3. Why this is better than small-`theta` pruning alone
4. Integration points by file/function
5. Safeguards / failure modes
6. Validation plan
7. Explicit opinion on the age-bias idea

Again: keep this tightly focused on **phase3 live pruning of already-admitted operators**, not a whole-system redesign.
</user_instructions>
