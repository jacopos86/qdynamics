<file_map>
/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3
├── pipelines
│   ├── hardcoded
│   │   ├── hh_realtime_vqs
│   │   ├── adapt_pipeline.py * +
│   │   ├── hh_continuation_generators.py * +
│   │   ├── hh_continuation_pruning.py * +
│   │   ├── hh_continuation_rescue.py * +
│   │   ├── hh_continuation_scoring.py * +
│   │   ├── hh_continuation_stage_control.py * +
│   │   ├── hh_continuation_symmetry.py * +
│   │   └── hh_continuation_types.py * +
│   ├── exact_bench
│   └── shell
├── src
│   └── quantum
│       ├── operator_pools
│       ├── time_propagation
│       └── ansatz_parameterization.py * +
├── test
│   ├── test_adapt_vqe_integration.py * +
│   ├── test_hh_continuation_generators.py * +
│   ├── test_hh_continuation_pruning.py * +
│   ├── test_hh_continuation_rescue.py * +
│   ├── test_hh_continuation_scoring.py * +
│   ├── test_hh_continuation_stage_control.py * +
│   └── test_hh_continuation_symmetry.py * +
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
└── AGENTS.md *


(* denotes selected files)
(+ denotes code-map available)
Config: directory-only view; selected files shown.

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/src/quantum/compiled_ansatz.py
Imports:
  - from dataclasses import dataclass
  - from typing import TYPE_CHECKING, Any, Literal, Sequence
  - import numpy as np
  - from src.quantum.ansatz_parameterization import (
    AnsatzParameterLayout,
    RotationTermSpec,
    build_parameter_layout,
    iter_runtime_rotation_terms,
)
  - from src.quantum.pauli_actions import (
    CompiledPauliAction,
    apply_compiled_pauli,
    apply_exp_term,
    compile_pauli_action_exyz,
)
  - from src.quantum.vqe_latex_python_pairs import AnsatzTerm
---
Classes:
  - CompiledRotationStep
    Properties:
      - coeff_real
      - action
  - CompiledPolynomialRotationPlan
    Methods:
      - L45: def runtime_count(self) -> int:
      - L49: def runtime_stop(self) -> int:
    Properties:
      - nq
      - label
      - steps
      - runtime_start
  - CompiledAnsatzExecutor
    Methods:
      - L61: def __init__(
        self,
        terms: Sequence["AnsatzTerm"],
        *,
        coefficient_tolerance: float = 1e-12,
        ignore_identity: bool = True,
        sort_terms: bool = True,
        pauli_action_cache: dict[str, CompiledPauliAction] | None = None,
        parameterization_mode: Literal["logical_shared", "per_pauli_term"] = "logical_shared",
        parameterization_layout: AnsatzParameterLayout | None = None,
    ):
      - L129: def _compile_rotation_specs(
        self,
        specs: Sequence[RotationTermSpec],
        *,
        poly: Any,
        label: str,
        runtime_start: int,
    ) -> CompiledPolynomialRotationPlan:
      - L164: def _compile_polynomial_plan(
        self,
        poly: Any,
        *,
        label: str,
        runtime_start: int,
    ) -> CompiledPolynomialRotationPlan:
      - L185: def _validate_parameterization_layout(
        self,
        layout: AnsatzParameterLayout,
    ) -> None:
      - L239: def prepare_state(self, theta: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
      - L297: def prepare_state_with_runtime_tangents(
        self,
        theta: np.ndarray,
        psi_ref: np.ndarray,
        *,
        runtime_indices: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    Properties:
      - _MATH_INIT
      - _MATH_COMPILE_POLYNOMIAL_PLAN
      - _MATH_PREPARE_STATE
      - _MATH_PREPARE_STATE_WITH_RUNTIME_TANGENTS

Global vars:
  - __all__
---

</file_map>
<file_contents>
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

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/AGENTS.md
```md
## Default working scope

Default focus for coding tasks:
- `src/`
- `pipelines/hardcoded/`
- `test/`
- `pipelines/run_guide.md`
- `AGENTS.md`
- `README.md`

Ignore by default unless the user explicitly asks for them:
- `.obsidian/`
- `.vscode-extensions/`
- `.vscode-home/`
- `.vscode-userdata/`
- `archive/`
- `docs/` except `docs/reports/` when PDF/report output is in scope
- `claude_code_adapt_wave/`

Use only when the task explicitly requires them:
- `MATH/`
- `docs/reports/`
- `pipelines/exact_bench/`
- `pipelines/shell/`

If the user narrows scope further (for example, “focus only on X and Y”), obey the user’s narrower scope over this default.

```markdown
<!-- agents.md -->

# Agent Instructions (Repo Rules of Engagement)

This file defines how automated coding agents (and humans) should modify this repository.

The priority is **correctness and consistency of operator conventions**, not “cleverness”.

---

## 1) Non-negotiable conventions

### Runbook authority for operational workflows
- For HH staging, run presets, and the new `ecut_1`/`ecut_2` interpretation, agents must consult `pipelines/run_guide.md` before editing pipeline invocation defaults, scaling tables, or manual run plans.
- Treat this as the canonical source for execution contracts that are not operator-level invariants (e.g., thresholds, gating policy, and recommended run ladders).
- Canonical doc order for agent decisions: `AGENTS.md` -> `pipelines/run_guide.md` -> `README.md` -> task-specific `MATH/` notes.
- Root-level supporting docs for agents: `README.md` and task-specific `MATH/` notes.
- Math naming contract for agents:
  - `MATH/Math.md` is the canonical symbolic math manuscript and is the source corresponding to `MATH/Math.pdf`.
  - `MATH/archaic_repo_math.md` is archival only; do not treat it as the default math file unless the user explicitly asks for the archaic repo-oriented notes.
- Agents should ignore `docs/` unless the user explicitly asks for material from that folder or PDF/report output is in scope.
- Agents may use `docs/reports/` when repairing or extending PDF/report output.

### Policy-vs-code conflict rule (mandatory)
- If AGENTS policy and current code/CLI behavior diverge, agents must **stop and ask the user before proceeding**.
- In docs, when such a mismatch exists, present:
  - `AGENTS target`
  - `Current code behavior`
  - `Required action: ask user before proceeding`

### Terminology contract (agent-run commands)
- When the user says **"conventional VQE"**, interpret it as the **non-ADAPT VQE** path.
- In this repo, **"conventional VQE"** maps to hardcoded non-ADAPT VQE flows (for example, the VQE stage in `pipelines/hardcoded/hubbard_pipeline.py` and non-ADAPT replay paths).
- **"ADAPT"** / **"ADAPT-VQE"** refers specifically to `pipelines/hardcoded/adapt_pipeline.py` and ADAPT stages.
- The phrase **"hardcoded pipeline"** in repo history/agent direction should be interpreted as the conventional (**non-ADAPT**) path unless ADAPT is explicitly named.

### Pauli symbols
- Use `e/x/y/z` internally (`e` = identity)
- If you need I/X/Y/Z output for reports, convert at the boundaries only.

### Pauli-string qubit ordering
- Pauli word string is ordered:
  - **left-to-right = q_(n-1) ... q_0**
  - **qubit 0 is rightmost character**
- All statevector bit indexing must match this.

### JW mapping source of truth
Do not re-derive JW mapping ad-hoc in new files. Use:
- `fermion_plus_operator(repr_mode="JW", nq, j)`
- `fermion_minus_operator(repr_mode="JW", nq, j)`
from `pauli_polynomial_class.py`

If you need number operators, implement:
- `n_p = (I - Z_p)/2`
in a way consistent with the Pauli-string convention above.

### PDF artifact parameter manifest (mandatory)
Every generated PDF artifact must include a **clear, list-style parameter manifest at the start of the document** (first page or first text-summary page), not just a raw command dump.

Required fields:
- Model family/name (for this repo: `Hubbard` unless/ until additional models are added)
- Ansatz type(s) used
- Whether drive is enabled (`--enable-drive` true/false)
- Core physical parameters: `t`, `U`, `dv`
- Any other run-defining parameters needed to reproduce the physics for that PDF

This rule applies to all PDF outputs (single-pipeline PDFs, compare PDFs, bundle PDFs, amplitude-comparison PDFs, and future report PDFs).

---

## 2) Keep the operator layer clean

### Operator layer responsibilities
The following modules are “operator algebra core”:
- `pauli_letters_module.py` (Symbol Product Map + PauliLetter)
- `qubitization_module.py` (PauliTerm)
- `pauli_polynomial_class.py` (PauliPolynomial + JW ladder operators)

### PauliTerm canonical source (mandatory)
Canonical `PauliTerm` source:
- `src.quantum.qubitization_module.PauliTerm`

Compatibility aliases (same class, not separate definitions):
- `src.quantum.pauli_words.PauliTerm`
- `pydephasing.quantum.pauli_words.PauliTerm`

Rules:
- Core package code **must** import `PauliTerm` from `qubitization_module.py`.
- Compatibility scripts may import `pauli_words.PauliTerm` only when required by existing interfaces; they must not introduce a new `PauliTerm` implementation.
- Base operator files **must remain unchanged**: `pauli_letters_module.py`, `pauli_words.py`, `qubitization_module.py`.
- Repo integration changes **must** be implemented with wrappers/shims around base files.

---

## 3) VQE implementation rules

### No Qiskit in the core VQE path
Qiskit is allowed only for:
- validation scripts/notebooks
- reference data generation/comparison

The production VQE path must be:
- numpy statevector backend
- minimal optimizer dependencies (SciPy optional; provide fallback if absent)

### VQE structure (required)
Implement VQE using the notation:

- Hamiltonian:
  `H = Σ_j h_j P_j`
- Energy:
  `E(θ) = Σ_j h_j ⟨ψ(θ)|P_j|ψ(θ)⟩`
- Ansatz:
  `|ψ(θ)⟩ = U_p(θ_p)…U_1(θ_1)|ψ_ref⟩`

### Ansatz selection
Default ansatz should be compatible with future time evolution:
- prefer “term-wise” or “Hamiltonian-variational” style layers
- each unitary should be representable as exp(-i θ * (PauliPolynomial))

Do not hardcode an ansatz that cannot be decomposed into Pauli exponentials.

### ADAPT gradient cache invariant
- The production ADAPT gradient path in `pipelines/hardcoded/adapt_pipeline.py` must keep compiled Pauli-action caching enabled for repeated operator evaluations.
- Do not replace the cached production gradient path with uncached per-term `apply_pauli_string` loops.
- If refactoring this area, preserve cached-vs-uncached numerical parity and keep regression tests for that parity.

---

## 4) Time-dynamics readiness (Suzuki–Trotter / QPE)

When implementing primitives, favor ones reusable for time evolution:
- Implement "apply exp(-i θ * PauliTerm)" and "apply exp(-i θ * PauliPolynomial)" as first-class utilities.
- Keep functions that return **term lists** (coeff, pauli_string) available for later grouping/ordering.
- Avoid architectures that require opaque circuit objects.

If adding higher-order Suzuki–Trotter later:
- do it by composition on top of the same primitive exp(PauliTerm) backend.

## 4a) Time-dependent drive implementation rules

The repo supports a **time-dependent onsite density drive** with a Gaussian-envelope sinusoidal waveform:

```
v(t) = A · sin(ω t + φ) · exp(-(t - t₀)² / (2 t̄²))
```

### Drive architecture
- The drive waveform and spatial patterns are defined in `src/quantum/drives_time_potential.py` (if present) or inline in the pipeline files.
- The compare pipeline forwards drive flags verbatim to both sub-pipelines via `_build_drive_args()` and `_build_drive_args_with_amplitude()`.
- All drive parameters are pass-through CLI flags; the compare pipeline does **not** interpret drive physics, only routes them.

### Drive reference propagator
- When drive is enabled, the **reference (exact) propagator** uses `scipy.sparse.linalg.expm_multiply` with piecewise-constant H(t) at each time step.
- The `--exact-steps-multiplier` flag controls refinement: `N_ref = multiplier × trotter_steps`.
- The `reference_method_name` in JSON output records which method was used (`expm_multiply_sparse_timedep` vs `eigendecomposition`).
- When drive is disabled, the static reference uses exact eigendecomposition — no changes.

### Spatial patterns
Three built-in patterns (`--drive-pattern`):
| Pattern | Weights s_j per site j |
|---------|------------------------|
| `staggered` | `(-1)^j` alternating sign |
| `dimer_bias` | `[+1, -1, +1, -1, ...]` (same as staggered for even L) |
| `custom` | User-supplied JSON array via `--drive-custom-s` |

### Rules for agents modifying drive code
- Do **not** add new drive parameters without also updating: (1) both pipeline `parse_args()`, (2) the compare pipeline's `_build_drive_args()` and `_build_drive_args_with_amplitude()`, (3) `pipelines/run_guide.md`.
- Drive must be **opt-in** (`--enable-drive`). Default behaviour (no flag) must be bit-for-bit identical to the static case.
- All drive-related CLI args must have the `--drive-` prefix (except `--enable-drive` and `--exact-steps-multiplier`).
- The safe-test (`_safe_test_check`) must remain: A=0 drive must produce trajectories identical to the no-drive case within `_SAFE_TEST_THRESHOLD = 1e-10`.

## 4b) Drive amplitude comparison PDF

The compare pipeline supports `--with-drive-amplitude-comparison-pdf` which:
1. Runs both pipelines 3× per L: drive-disabled, A0-enabled, A1-enabled (6 sub-runs total per L).
2. Generates a multi-page physics-facing PDF per L with scoreboard tables, drive waveform, response deltas, and a combined HC/QK overlay.
3. Writes `json/amp_cmp_hubbard_{tag}_metrics.json` with `safe_test`, `delta_vqe_hc_minus_qk_at_A0`, `delta_vqe_hc_minus_qk_at_A1`.

### Rules for agents modifying amplitude comparison
- All artifacts go to `json/` or `pdf/` subdirectories. Filenames use the tag convention `L{L}_{drive|static}_t{t}_U{u}_S{steps}`.
- Intermediate JSON files use the `amp_hc_hubbard_` / `amp_qk_hubbard_` prefix: `json/amp_hc_hubbard_L2_static_t1.0_U4.0_S32_disabled.json`, `json/amp_qk_hubbard_L2_static_t1.0_U4.0_S32_A0.json`, etc.
- Safe-test scalar metrics must always be reported on the scoreboard table. The full safe-test timeseries page is conditional (fail, near-threshold, or `--report-verbose`).
- VQE delta is defined as `ΔE = VQE_hardcoded − VQE_qiskit` (the sector-filtered energy, not full-Hilbert).
- New amplitude comparison CLI args: `--drive-amplitudes A0,A1`, `--with-drive-amplitude-comparison-pdf`, `--report-verbose`, and `--safe-test-near-threshold-factor`.

## 4c) User shorthand run convention (`run L`)

When the user requests a shorthand run like:
- "run L=4"
- "run a number L"
- "run L 5"

interpret it with the following **default contract**:

1. The run is **drive-enabled, never static**.
2. **Default Hard Gate (final conventional VQE):**
   `abs(vqe.energy - ground_state.exact_energy_filtered) < 1e-4`.
3. Use **L-scaled heaviness** (stronger settings for larger L), not one-size-fits-all settings.
4. Pre-VQE stages (warm-start HVA / ADAPT) are **diagnostic** by default and are not hard-fail gates unless explicitly requested by the user.

Implementation rule:
- Prefer `pipelines/shell/run_drive_accurate.sh --L <L>` when available.
- If this script is unavailable, emulate its semantics manually:
  - drive enabled with scaling profile defaults,
  - per-L parameter table,
  - enforce at least the shorthand contract gate (`< 1e-4`),
  - treat `< 1e-7` as **optional strict mode**, not the default hard stop.

## 4d) Mandatory minimum VQE / Trotter parameters per L

**Agents must never run a pipeline with settings weaker than the table below.**
Under-parameterised runs waste wall-clock time and produce unconverged results
that are useless for diagnostics — you cannot tell whether a failure is a code
bug or just insufficient optimiser effort.

If in doubt, **round up** to the next row.

### Hubbard (pure) — minimum settings

| L | `--trotter-steps` | `--exact-steps-multiplier` | `--num-times` | `--vqe-reps` | `--vqe-restarts` | `--vqe-maxiter` | optimizer | `--t-final` |
|---|---|---|---|---|---|---|---|---|
| 2 | 64 | 2 | 201 | 2 | 2 | 600 | COBYLA | 10.0 |
| 3 | 128 | 2 | 201 | 2 | 3 | 1200 | COBYLA | 15.0 |
| 4 | 256 | 3 | 241 | 3 | 4 | 6000 | SLSQP | 20.0 |
| 5 | 384 | 3 | 301 | 3 | 5 | 8000 | SLSQP | 20.0 |
| 6 | 512 | 4 | 361 | 4 | 6 | 10000 | SLSQP | 20.0 |

### Hubbard-Holstein (HH) — minimum settings

HH requires heavier settings than pure Hubbard at the same L due to
the enlarged Hilbert space (phonon modes).

| L | `--n-ph-max` | `--trotter-steps` | `--vqe-reps` | `--vqe-restarts` | `--vqe-maxiter` | optimizer |
|---|---|---|---|---|---|---|
| 2 | 1 | 64 | 2 | 3 | 800 | SPSA |
| 2 | 2 | 128 | 3 | 4 | 1500 | SPSA |
| 3 | 1 | 192 | 2 | 4 | 2400 | SPSA |

### Rules

1. **Never use L=2 defaults for L≥3.** The Hilbert space grows as $2^{2L}$
   (Hubbard) or $2^{2L} \cdot (n_{ph}+1)^L$ (HH). Parameters that converge
   at L=2 are catastrophically insufficient at L=3+.
2. If the user says "run L=3" without specifying parameters, use this table
   (or `pipelines/shell/run_drive_accurate.sh`) — do not invent lighter settings.
3. For validation / smoke-test runs that intentionally use weak settings,
   add an explicit comment: `# SMOKE TEST — intentionally weak settings`.
4. When writing tests, light settings (e.g., `maxiter=40`) are acceptable
   because tests verify implementation correctness, not convergence quality.
   But pipeline runs and demo artifacts must meet the table above.

## 4e) Cross-check suite (`pipelines/exact_bench/cross_check_suite.py`)

The cross-check suite compares **all available ansätze × VQE modes** against
exact ED for a given L, with Trotter dynamics and multi-page PDF output.

### Usage

```bash
# Pure Hubbard, auto-scaled parameters from §4d:
python pipelines/exact_bench/cross_check_suite.py --L 2 --problem hubbard

# Hubbard-Holstein:
python pipelines/exact_bench/cross_check_suite.py --L 2 --problem hh --omega0 1.0 --g-ep 0.5

# Override auto-scaled params (e.g. for smoke tests):
python pipelines/exact_bench/cross_check_suite.py --L 2 --problem hubbard \
  --vqe-reps 1 --vqe-restarts 1 --vqe-maxiter 40 --trotter-steps 8
```

### Shorthand convention

When the user says "run cross-check L=3" or "cross-check L 4":
1. Run `cross_check_suite.py --L <L> --problem hubbard` (auto-scaled from §4d).
2. If the user says "cross-check HH L=2", add `--problem hh`.
3. Do **not** override `--vqe-maxiter` or `--trotter-steps` below §4d minimums.

### Trial matrix

| Problem | Ansätze |
|---------|---------|
| `hubbard` | HVA-Layerwise, UCCSD-Layerwise, ADAPT(UCCSD), ADAPT(full_H) |
| `hh` | HH-Termwise, HH-Layerwise, ADAPT(full_H) |

### Output

- JSON: `<output-dir>/xchk_L{L}_{problem}_t{t}_U{U}.json`
- PDF: same path with `.pdf` — parameter manifest, scoreboard table, per-ansatz 3-panel trajectory plots (fidelity, energy, occupation), fidelity/energy/doublon overlay pages, command page.

## 4f) CFQM propagation rules (`hubbard_pipeline.py`)

CFQM support is available in the hardcoded pipeline via:
- `--propagator cfqm4`
- `--propagator cfqm6`

### CFQM semantics (must preserve)
- CFQM node sampling is fixed by scheme nodes `c_j`; it does **not** use midpoint/left/right `--drive-time-sampling`.
- `--exact-steps-multiplier` is reference-only (piecewise/reference refinement) and must not alter CFQM macro-step count.
- Default behavior remains unchanged unless `--propagator` is switched from `suzuki2`.

### Required warning strings (exact text)
- `CFQM ignores midpoint/left/right sampling; uses fixed scheme nodes c_j.`
- `Inner Suzuki-2 makes overall method 2nd order; use expm_multiply_sparse/dense_expm for true CFQM order.`

### Invariants and guardrails
- Keep A=0 safe-test invariant: drive-enabled run with `A=0` must match no-drive within `<= 1e-10`.
- Keep zero-increment insertion guard in CFQM stage accumulation (do not insert new keys for exact-zero increments).
- Keep deterministic stage assembly keyed by `ordered_labels`.
- Drive labels not present in `ordered_labels` must not be inserted.
- Current default policy: warn once per unknown label for nontrivial coefficients, then ignore; tiny coefficients (`abs(coeff) <= 1e-14`) are silently ignored.
- Keep fail-fast validation for:
  - `dt > 0`
  - `n_steps >= 1`
  - finite drive coefficients (NaN/inf -> explicit error with label/time)
  - CFQM scheme validation (`validate_scheme`)

### Backend rule
- For `--cfqm-stage-exp expm_multiply_sparse`, prefer sparse-native stage assembly + `scipy.sparse.linalg.expm_multiply`.
- Avoid dense intermediate stage matrices in the sparse backend path.
- Shared Pauli action primitives live in `src/quantum/pauli_actions.py`; do not reintroduce `src/quantum` -> `pipelines/*` import dependency.

### Normalization rule
- No renormalization by default; renormalize only when `--cfqm-normalize` is explicitly enabled.

### Benchmarking rule (CFQM vs Suzuki)
- For propagator comparison reports, use hardware-oriented proxy budgets (at minimum 2-qubit/depth-style proxies such as `cx_proxy_total`) as the primary axis.
- Local machine wall-clock is secondary and must not be the only headline comparison metric.
- For efficiency reports generated by `pipelines/exact_bench/cfqm_vs_suzuki_efficiency_suite.py`, keep main apple-to-apple tables to exact-cost ties (`delta=0`) only.
- Do not mix nearest-neighbor fallback matches into the same main comparison table; fallback rows belong in an appendix/diagnostic section only.
- `S` in benchmark tables means macro-step count (`trotter_steps`) and must not be treated as fairness axis.

### Minimal post-edit verification commands

```bash
# CFQM unit/acceptance tests
pytest -q test/test_cfqm_schemes.py test/test_cfqm_propagator.py test/test_cfqm_acceptance.py

# Help/flag sanity
python pipelines/hardcoded/hubbard_pipeline.py --help | rg -n "propagator|cfqm-stage-exp|cfqm-coeff-drop-abs-tol|cfqm-normalize"
```

## 4g) Codex-run HH warm cutoff + state handoff (no manual keypresses)

For Codex-agent runs, do **not** rely on interactive `Ctrl+C` behavior as part
of the normal workflow. Use exported state bundles as the active handoff
contract, and treat warm-cutoff orchestration scripts as archived examples.

Canonical active contract:
- `pipelines/hardcoded/handoff_state_bundle.py`
- `pipelines/hardcoded/hubbard_pipeline.py --initial-state-source adapt_json`

Archived workflow examples:
- `archive/handoff/l4_hh_warmstart_uccsd_paop_hva_seq_probe.py`

Required conventions for agent runs:
1. If warm-stage runtime must be bounded by convergence trend in an archived
   sequential workflow, enable:
   - `--warm-auto-cutoff`
   - slope/window knobs (`--warm-cutoff-*`)
2. Always set state export paths (`--state-export-dir`, `--state-export-prefix`)
   so warm and ADAPT checkpoints are persisted, then write reusable handoff
   bundles with `pipelines/hardcoded/handoff_state_bundle.py`.
3. Use exported `*_A_probe_state.json` / `*_A_medium_state.json` as
   `adapt_json` handoff into `pipelines/hardcoded/hubbard_pipeline.py` when
   running conventional VQE+trotter trajectories from a saved state.
4. For “UCCSD + PAOP only” handoff, use **A-arm** exports; do not use B-arm
   files (`B_*` includes HVA in pool construction).
5. ADAPT stop/handoff decisions must be **energy-error-drop first**:
   - Primary signal: per-depth `ΔE_abs` improvement (`drop = ΔE_abs(d-1)-ΔE_abs(d)`).
   - Use patience over completed depths (`M` consecutive low-drop depths) with
     a minimum depth guard (`d_min`) before stopping.
   - Gradient floors (`max|g|`) are secondary diagnostics/safety checks only;
     they must not be the sole stop reason in agent-run HH workflows.
   - Do not interpret `max|g|` as “the energy-error drop per depth.”

---

## 5) Style and maintainability

### Clean/simple code
- Prefer pure functions where possible (no hidden global state).
- Keep modules small, with a single responsibility.
- Use explicit types for public function signatures.
- Prefer explicit errors (`log.error(...)` or raising) over silent coercions.

### Built Math-Symbols/Description above Python pairing
When adding new modules :
- include the Built-in math symbolic expression in a string right above the function that implements it
- keep the math and code aligned 1:1

### Regression/validation
Whenever you modify:
- Hubbard Hamiltonian construction
- indexing conventions
- JW mapping / number operator

You must update or re-run reference checks against:
- `hubbard_jw_*.json`

Qiskit baseline scripts may be used to sanity check, but they are not the core test oracle.

---

## 6) What an agent should NOT do
- Do not change Pauli-string ordering conventions.
- Do not introduce Qiskit into core/'hardcoded' algorithm modules.
- Do not add heavy dependencies without a strong reason.
- Do not "optimize" by rewriting algebra rules unless correctness is proven with regression tests.
- Do not add new drive parameters without updating all three pipelines' `parse_args()`, `_build_drive_args()`, `_build_drive_args_with_amplitude()`, and `pipelines/run_guide.md`.
- Do not break the safe-test invariant (A=0 drive must equal no-drive to machine precision).
- Do not stop a run because you think it is taking up too much run-time. The only acceptable reason to stop/interrupt an already active run/script is for debugging.
- **Do not run a pipeline with parameters below the §4d minimum table.** If the user does not specify parameters, look up the table — never guess or use L=2 defaults for larger L.



--Note -- Take your time coding! Be safe, and do not rush. The user has a lot of time and does not need things quickly.

## Plans

- Make the plan extremely consise. Sacrifice grammar for the sake of concision.
- Near the end of each plan, give me a list of unresolved questions to answer/problems, if any, and the files you will edit.
- At the end of each plan, state all files intended to alter, and functions and classes to be altered. If none, write 'Files to edit: None'.

## 4h) Legacy noiseless-estimator parity rule (HH anchor)

When a user asks to verify that the new noiseless-estimator path is equivalent to the pre-noise HH pipeline:

1. Use the locked L=2 baseline artifact:
- `artifacts/json/hc_hh_L2_static_t1.0_U2.0_g1.0_nph1.json`
2. Run a full-match parity case (do not downscale run knobs for this check).
3. Enforce strict gate: `max_abs_delta <= 1e-10` on selected observables, with exact time-grid match required.
4. Record parity fields in JSON/PDF (`legacy_parity.*`) and emit the comparison plot if requested.

This is a validation exception to “very light” preference: parity verdicts must use full baseline-matched settings.

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/test/test_hh_continuation_rescue.py
```py
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_continuation_rescue import (
    RescueConfig,
    rank_rescue_candidates,
    should_trigger_rescue,
)


def test_rescue_is_off_by_default() -> None:
    enabled, reason = should_trigger_rescue(
        enabled=False,
        exact_state_available=True,
        residual_opened=True,
        trough_detected=True,
        history=[{"delta_abs_drop_from_prev": 0.0}, {"delta_abs_drop_from_prev": 0.0}],
        shortlist_records=[{"full_v2_score": 1.0}, {"full_v2_score": 0.99}],
        cfg=RescueConfig(),
    )
    assert enabled is False
    assert reason == "disabled"


def test_rescue_requires_exact_state_and_flat_diagnostics() -> None:
    enabled, reason = should_trigger_rescue(
        enabled=True,
        exact_state_available=True,
        residual_opened=True,
        trough_detected=False,
        history=[{"delta_abs_drop_from_prev": 0.0}, {"delta_abs_drop_from_prev": 0.0}],
        shortlist_records=[{"full_v2_score": 1.0}, {"full_v2_score": 0.98}],
        cfg=RescueConfig(),
    )
    assert enabled is True
    assert reason == "flat_drop_and_shortlist"


def test_rank_rescue_candidates_is_deterministic_and_requires_gain() -> None:
    cfg = RescueConfig(min_overlap_gain=1e-4)
    best, meta = rank_rescue_candidates(
        records=[
            {"candidate_pool_index": 1, "position_id": 0, "full_v2_score": 0.8, "simple_score": 0.8},
            {"candidate_pool_index": 0, "position_id": 1, "full_v2_score": 0.8, "simple_score": 0.8},
        ],
        overlap_gain_fn=lambda rec: 0.2 if int(rec["candidate_pool_index"]) == 0 else 0.1,
        cfg=cfg,
    )
    assert meta["executed"] is True
    assert best is not None
    assert int(best["candidate_pool_index"]) == 0


def test_rank_rescue_candidates_skips_when_gain_too_small() -> None:
    best, meta = rank_rescue_candidates(
        records=[{"candidate_pool_index": 0, "position_id": 0, "full_v2_score": 0.5, "simple_score": 0.5}],
        overlap_gain_fn=lambda _rec: 0.0,
        cfg=RescueConfig(min_overlap_gain=1e-4),
    )
    assert best is None
    assert meta["reason"] == "insufficient_overlap_gain"

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/hardcoded/hh_continuation_generators.py
```py
#!/usr/bin/env python3
"""Generator metadata helpers for HH continuation."""

from __future__ import annotations

from dataclasses import asdict
import itertools
import hashlib
from typing import Any, Mapping, Sequence

from pipelines.hardcoded.hh_continuation_types import GeneratorMetadata, GeneratorSplitEvent
from src.quantum.hubbard_latex_python_pairs import SPIN_DN, SPIN_UP, mode_index
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


def _polynomial_signature(poly: Any, *, tol: float = 1e-12) -> tuple[tuple[str, float], ...]:
    items: list[tuple[str, float]] = []
    for term in poly.return_polynomial():
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"Non-negligible imaginary coefficient in generator polynomial: {coeff}")
        items.append((str(term.pw2strng()), float(round(coeff.real, 12))))
    items.sort()
    return tuple(items)


def _support_qubits(poly: Any) -> list[int]:
    support: set[int] = set()
    for term in poly.return_polynomial():
        word = str(term.pw2strng())
        nq = int(term.nqubit())
        for idx, ch in enumerate(word):
            if ch == "e":
                continue
            support.add(int(nq - 1 - idx))
    return sorted(int(q) for q in support)


def _qubit_to_site(
    qubit: int,
    *,
    num_sites: int,
    ordering: str,
    qpb: int,
) -> int:
    q = int(qubit)
    n_sites = int(num_sites)
    fermion_qubits = 2 * n_sites
    if q < fermion_qubits:
        ordering_key = str(ordering).strip().lower()
        if ordering_key == "interleaved":
            return int(q // 2)
        return int(q % n_sites)
    return int((q - fermion_qubits) // int(max(1, qpb)))


def _support_sites(
    support_qubits: Sequence[int],
    *,
    num_sites: int,
    ordering: str,
    qpb: int,
) -> list[int]:
    out = {
        _qubit_to_site(int(q), num_sites=int(num_sites), ordering=str(ordering), qpb=int(qpb))
        for q in support_qubits
    }
    return sorted(int(x) for x in out)


def _relative_site_offsets(sites: Sequence[int]) -> list[int]:
    if not sites:
        return []
    site_min = min(int(x) for x in sites)
    return [int(int(x) - site_min) for x in sites]


def _template_id(
    *,
    family_id: str,
    support_site_offsets: Sequence[int],
    n_poly_terms: int,
    has_boson_support: bool,
    has_fermion_support: bool,
    is_macro_generator: bool,
) -> str:
    parts = [
        str(family_id),
        "macro" if bool(is_macro_generator) else "atomic",
        f"terms{int(n_poly_terms)}",
        f"sites{','.join(str(int(x)) for x in support_site_offsets)}",
        f"bos{int(bool(has_boson_support))}",
        f"ferm{int(bool(has_fermion_support))}",
    ]
    return "|".join(parts)


def _serialize_polynomial_terms(poly: Any, *, tol: float = 1e-12) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for term in poly.return_polynomial():
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        out.append(
            {
                "pauli_exyz": str(term.pw2strng()),
                "coeff_re": float(coeff.real),
                "coeff_im": float(coeff.imag),
                "nq": int(term.nqubit()),
            }
        )
    return out


def _build_number_operator(*, nq: int, qubit: int) -> PauliPolynomial:
    z_word = ["e"] * int(nq)
    z_word[int(nq - 1 - int(qubit))] = "z"
    return PauliPolynomial(
        "JW",
        [
            PauliTerm(int(nq), ps=("e" * int(nq)), pc=0.5),
            PauliTerm(int(nq), ps=("".join(z_word)), pc=-0.5),
        ],
    )


def _fermion_number_operators(
    *,
    nq: int,
    num_sites: int,
    ordering: str,
) -> tuple[PauliPolynomial, PauliPolynomial]:
    n_up = PauliPolynomial("JW", [])
    n_dn = PauliPolynomial("JW", [])
    for site in range(int(num_sites)):
        n_up += _build_number_operator(
            nq=int(nq),
            qubit=int(mode_index(int(site), SPIN_UP, indexing=str(ordering), n_sites=int(num_sites))),
        )
        n_dn += _build_number_operator(
            nq=int(nq),
            qubit=int(mode_index(int(site), SPIN_DN, indexing=str(ordering), n_sites=int(num_sites))),
        )
    return n_up, n_dn


def _commutator_l1_norm(lhs: PauliPolynomial, rhs: PauliPolynomial) -> float:
    comm = lhs * rhs - rhs * lhs
    return float(sum(abs(complex(term.p_coeff)) for term in comm.return_polynomial()))


def _operator_symmetry_gate(
    *,
    polynomial: Any,
    num_sites: int,
    ordering: str,
    symmetry_spec: Mapping[str, Any] | None,
    tol: float = 1e-10,
) -> dict[str, Any]:
    terms = list(polynomial.return_polynomial())
    nq = int(terms[0].nqubit()) if terms else 0
    if nq <= 0 or int(num_sites) <= 0:
        return {
            "checked": False,
            "passed": True,
            "particle_number_preserving": True,
            "spin_sector_preserving": True,
            "commutator_l1_total": 0.0,
            "commutator_l1_up": 0.0,
            "commutator_l1_dn": 0.0,
        }
    require_particle = bool(
        not isinstance(symmetry_spec, Mapping)
        or str(symmetry_spec.get("particle_number_mode", "preserving")) == "preserving"
    )
    require_spin = bool(
        not isinstance(symmetry_spec, Mapping)
        or str(symmetry_spec.get("spin_sector_mode", "preserving")) == "preserving"
    )
    n_up, n_dn = _fermion_number_operators(
        nq=int(nq),
        num_sites=int(num_sites),
        ordering=str(ordering),
    )
    comm_up = _commutator_l1_norm(n_up, polynomial)
    comm_dn = _commutator_l1_norm(n_dn, polynomial)
    comm_total = _commutator_l1_norm(n_up + n_dn, polynomial)
    particle_ok = bool((not require_particle) or comm_total <= float(tol))
    spin_ok = bool((not require_spin) or (comm_up <= float(tol) and comm_dn <= float(tol)))
    return {
        "checked": True,
        "passed": bool(particle_ok and spin_ok),
        "particle_number_preserving": bool(comm_total <= float(tol)),
        "spin_sector_preserving": bool(comm_up <= float(tol) and comm_dn <= float(tol)),
        "commutator_l1_total": float(comm_total),
        "commutator_l1_up": float(comm_up),
        "commutator_l1_dn": float(comm_dn),
        "required_particle_number": bool(require_particle),
        "required_spin_sector": bool(require_spin),
    }


def _runtime_split_symmetry_gate(
    *,
    polynomial: Any,
    num_sites: int,
    ordering: str,
    symmetry_spec: Mapping[str, Any] | None,
    tol: float = 1e-10,
) -> dict[str, Any]:
    return _operator_symmetry_gate(
        polynomial=polynomial,
        num_sites=int(num_sites),
        ordering=str(ordering),
        symmetry_spec=symmetry_spec,
        tol=float(tol),
    )


def _symmetry_spec_with_gate(
    *,
    base_spec: Mapping[str, Any] | None,
    gate: Mapping[str, Any],
    checked_tag: str,
    rejected_tag: str,
) -> dict[str, Any] | None:
    if not isinstance(base_spec, Mapping):
        return None
    out = dict(base_spec)
    raw_tags = out.get("tags", [])
    tags = (
        [str(tag) for tag in raw_tags]
        if isinstance(raw_tags, Sequence) and not isinstance(raw_tags, (str, bytes))
        else []
    )
    if str(checked_tag) not in tags:
        tags.append(str(checked_tag))
    particle_ok = bool(gate.get("particle_number_preserving", True))
    spin_ok = bool(gate.get("spin_sector_preserving", True))
    out["particle_number_mode"] = "preserving" if particle_ok else "violating"
    out["spin_sector_mode"] = "preserving" if spin_ok else "violating"
    if not bool(gate.get("passed", True)):
        out["leakage_risk"] = float(max(float(out.get("leakage_risk", 0.0)), 1.0))
        out["hard_guard"] = True
        if str(rejected_tag) not in tags:
            tags.append(str(rejected_tag))
    else:
        out["leakage_risk"] = 0.0
    out["tags"] = tags
    return out


def _symmetry_spec_with_runtime_gate(
    *,
    base_spec: Mapping[str, Any] | None,
    gate: Mapping[str, Any],
) -> dict[str, Any] | None:
    return _symmetry_spec_with_gate(
        base_spec=base_spec,
        gate=gate,
        checked_tag="runtime_split_checked",
        rejected_tag="runtime_split_rejected",
    )


def rebuild_polynomial_from_serialized_terms(
    serialized_terms: Sequence[Mapping[str, Any]],
) -> PauliPolynomial:
    nq_expected: int | None = None
    coeffs_by_label: dict[str, complex] = {}
    label_order: list[str] = []
    for raw in serialized_terms:
        if not isinstance(raw, Mapping):
            continue
        nq = int(raw.get("nq", 0))
        label = str(raw.get("pauli_exyz", ""))
        coeff = complex(float(raw.get("coeff_re", 0.0)), float(raw.get("coeff_im", 0.0)))
        if nq <= 0 or label == "":
            continue
        if nq_expected is None:
            nq_expected = int(nq)
        elif int(nq) != int(nq_expected):
            raise ValueError("Serialized runtime-split terms use inconsistent nq values.")
        if label not in coeffs_by_label:
            label_order.append(label)
            coeffs_by_label[label] = complex(0.0)
        coeffs_by_label[label] += coeff
    if nq_expected is None or not label_order:
        raise ValueError("Serialized runtime-split terms are missing or invalid.")

    poly = PauliPolynomial("JW")
    for label in label_order:
        coeff = complex(coeffs_by_label[label])
        if abs(coeff) < 1.0e-7:
            continue
        poly.add_term(PauliTerm(int(nq_expected), ps=label, pc=coeff))
    if int(poly.count_number_terms()) <= 0:
        raise ValueError("Serialized runtime-split terms cancel below tolerance.")
    return poly


def build_generator_metadata(
    *,
    label: str,
    polynomial: Any,
    family_id: str,
    num_sites: int,
    ordering: str,
    qpb: int,
    split_policy: str = "preserve",
    parent_generator_id: str | None = None,
    symmetry_spec: Mapping[str, Any] | None = None,
) -> GeneratorMetadata:
    signature = _polynomial_signature(polynomial)
    support_qubits = _support_qubits(polynomial)
    support_sites = _support_sites(
        support_qubits,
        num_sites=int(num_sites),
        ordering=str(ordering),
        qpb=int(qpb),
    )
    support_site_offsets = _relative_site_offsets(support_sites)
    has_fermion_support = any(int(q) < 2 * int(num_sites) for q in support_qubits)
    has_boson_support = any(int(q) >= 2 * int(num_sites) for q in support_qubits)
    n_poly_terms = int(len(list(polynomial.return_polynomial())))
    is_macro = bool(n_poly_terms > 1 and str(split_policy) != "deliberate_split")
    template_id = _template_id(
        family_id=str(family_id),
        support_site_offsets=support_site_offsets,
        n_poly_terms=int(n_poly_terms),
        has_boson_support=bool(has_boson_support),
        has_fermion_support=bool(has_fermion_support),
        is_macro_generator=bool(is_macro),
    )
    digest = hashlib.sha1(
        (
            f"{family_id}|{template_id}|{signature}|{split_policy}|{parent_generator_id or ''}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    compile_metadata: dict[str, Any] = {
        "num_polynomial_terms": int(n_poly_terms),
        "signature_size": int(len(signature)),
        "has_boson_support": bool(has_boson_support),
        "has_fermion_support": bool(has_fermion_support),
        "support_size": int(len(support_qubits)),
        "serialized_terms_exyz": _serialize_polynomial_terms(polynomial),
    }
    symmetry_spec_out = (dict(symmetry_spec) if isinstance(symmetry_spec, Mapping) else None)
    if symmetry_spec_out is not None:
        symmetry_gate = _operator_symmetry_gate(
            polynomial=polynomial,
            num_sites=int(num_sites),
            ordering=str(ordering),
            symmetry_spec=symmetry_spec_out,
        )
        compile_metadata["symmetry_intent"] = dict(symmetry_spec_out)
        compile_metadata["symmetry_gate"] = dict(symmetry_gate)
        symmetry_spec_out = _symmetry_spec_with_gate(
            base_spec=symmetry_spec_out,
            gate=symmetry_gate,
            checked_tag="operator_symmetry_checked",
            rejected_tag="operator_symmetry_rejected",
        )
    return GeneratorMetadata(
        generator_id=f"gen:{digest}",
        family_id=str(family_id),
        template_id=str(template_id),
        candidate_label=str(label),
        support_qubits=[int(x) for x in support_qubits],
        support_sites=[int(x) for x in support_sites],
        support_site_offsets=[int(x) for x in support_site_offsets],
        is_macro_generator=bool(is_macro),
        split_policy=str(split_policy),
        parent_generator_id=(str(parent_generator_id) if parent_generator_id is not None else None),
        symmetry_spec=symmetry_spec_out,
        compile_metadata=compile_metadata,
    )


def build_pool_generator_registry(
    *,
    terms: Sequence[Any],
    family_ids: Sequence[str],
    num_sites: int,
    ordering: str,
    qpb: int,
    symmetry_specs: Sequence[Mapping[str, Any] | None] | None = None,
    split_policy: str = "preserve",
) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    sym_specs = list(symmetry_specs) if symmetry_specs is not None else [None] * len(list(terms))
    for idx, term in enumerate(terms):
        family_id = str(family_ids[idx] if idx < len(family_ids) else "unknown")
        symmetry_spec = sym_specs[idx] if idx < len(sym_specs) else None
        meta = build_generator_metadata(
            label=str(term.label),
            polynomial=term.polynomial,
            family_id=str(family_id),
            num_sites=int(num_sites),
            ordering=str(ordering),
            qpb=int(qpb),
            split_policy=str(split_policy),
            symmetry_spec=symmetry_spec,
        )
        registry[str(term.label)] = asdict(meta)
    return registry


def build_runtime_split_children(
    *,
    parent_label: str,
    polynomial: Any,
    family_id: str,
    num_sites: int,
    ordering: str,
    qpb: int,
    split_mode: str,
    parent_generator_metadata: Mapping[str, Any] | None = None,
    symmetry_spec: Mapping[str, Any] | None = None,
    max_children: int | None = None,
    tol: float = 1e-12,
) -> list[dict[str, Any]]:
    serialized = _serialize_polynomial_terms(polynomial, tol=float(tol))
    total_children = int(len(serialized))
    if total_children <= 1:
        return []
    out: list[dict[str, Any]] = []
    parent_generator_id = None
    if isinstance(parent_generator_metadata, Mapping) and parent_generator_metadata.get("generator_id") is not None:
        parent_generator_id = str(parent_generator_metadata.get("generator_id"))
    child_limit = total_children if max_children is None or int(max_children) <= 0 else min(total_children, int(max_children))
    for child_index, term_info in enumerate(serialized[:child_limit]):
        child_poly = rebuild_polynomial_from_serialized_terms([term_info])
        symmetry_gate = _runtime_split_symmetry_gate(
            polynomial=child_poly,
            num_sites=int(num_sites),
            ordering=str(ordering),
            symmetry_spec=symmetry_spec,
        )
        child_label = (
            f"{str(parent_label)}::split[{int(child_index)}]::{str(term_info.get('pauli_exyz', ''))}"
        )
        child_meta = asdict(
            build_generator_metadata(
                label=str(child_label),
                polynomial=child_poly,
                family_id=str(family_id),
                num_sites=int(num_sites),
                ordering=str(ordering),
                qpb=int(qpb),
                split_policy="deliberate_split",
                parent_generator_id=parent_generator_id,
                symmetry_spec=_symmetry_spec_with_runtime_gate(
                    base_spec=symmetry_spec,
                    gate=symmetry_gate,
                ),
            )
        )
        compile_metadata = dict(child_meta.get("compile_metadata", {}))
        compile_metadata["runtime_split"] = {
            "mode": str(split_mode),
            "parent_label": str(parent_label),
            "child_index": int(child_index),
            "child_count": int(total_children),
            "representation": "child_atom",
            "symmetry_gate": dict(symmetry_gate),
        }
        compile_metadata["serialized_terms_exyz"] = [dict(term_info)]
        child_meta["compile_metadata"] = compile_metadata
        out.append(
            {
                "child_label": str(child_label),
                "child_polynomial": child_poly,
                "child_generator_metadata": dict(child_meta),
                "child_index": int(child_index),
                "child_count": int(total_children),
                "parent_label": str(parent_label),
                "symmetry_gate": dict(symmetry_gate),
            }
        )
    return out


def build_runtime_split_child_sets(
    *,
    parent_label: str,
    family_id: str,
    num_sites: int,
    ordering: str,
    qpb: int,
    split_mode: str,
    children: Sequence[Mapping[str, Any]],
    parent_generator_metadata: Mapping[str, Any] | None = None,
    symmetry_spec: Mapping[str, Any] | None = None,
    max_subset_size: int = 3,
    tol: float = 1e-12,
) -> list[dict[str, Any]]:
    parent_generator_id = None
    if isinstance(parent_generator_metadata, Mapping) and parent_generator_metadata.get("generator_id") is not None:
        parent_generator_id = str(parent_generator_metadata.get("generator_id"))
    parent_signature = None
    if isinstance(parent_generator_metadata, Mapping):
        compile_meta = parent_generator_metadata.get("compile_metadata")
        if isinstance(compile_meta, Mapping):
            serialized_parent = compile_meta.get("serialized_terms_exyz")
            if isinstance(serialized_parent, Sequence):
                try:
                    parent_signature = _polynomial_signature(
                        rebuild_polynomial_from_serialized_terms(serialized_parent),
                        tol=float(tol),
                    )
                except Exception:
                    parent_signature = None
    child_rows = [dict(row) for row in children if isinstance(row, Mapping)]
    if len(child_rows) <= 1:
        return []
    subset_cap = int(max(1, min(int(max_subset_size), len(child_rows))))
    out: list[dict[str, Any]] = []
    seen_signatures: set[tuple[tuple[str, float], ...]] = set()
    for subset_size in range(1, subset_cap + 1):
        for subset in itertools.combinations(child_rows, subset_size):
            serialized_subset: list[dict[str, Any]] = []
            child_labels: list[str] = []
            child_indices: list[int] = []
            child_generator_ids: list[str] = []
            for child in subset:
                child_labels.append(str(child.get("child_label")))
                if child.get("child_index") is not None:
                    child_indices.append(int(child.get("child_index")))
                child_meta = child.get("child_generator_metadata")
                if isinstance(child_meta, Mapping) and child_meta.get("generator_id") is not None:
                    child_generator_ids.append(str(child_meta.get("generator_id")))
                compile_meta = child_meta.get("compile_metadata", {}) if isinstance(child_meta, Mapping) else {}
                serialized_terms = compile_meta.get("serialized_terms_exyz", []) if isinstance(compile_meta, Mapping) else []
                for term_info in serialized_terms:
                    if isinstance(term_info, Mapping):
                        serialized_subset.append(dict(term_info))
            if not serialized_subset:
                continue
            subset_poly = rebuild_polynomial_from_serialized_terms(serialized_subset)
            subset_signature = _polynomial_signature(subset_poly, tol=float(tol))
            if parent_signature is not None and subset_signature == parent_signature:
                continue
            if subset_signature in seen_signatures:
                continue
            symmetry_gate = _runtime_split_symmetry_gate(
                polynomial=subset_poly,
                num_sites=int(num_sites),
                ordering=str(ordering),
                symmetry_spec=symmetry_spec,
            )
            if not bool(symmetry_gate.get("passed", True)):
                continue
            child_index_tag = ",".join(str(int(idx)) for idx in child_indices)
            subset_label = f"{str(parent_label)}::child_set[{child_index_tag}]"
            subset_meta = asdict(
                build_generator_metadata(
                    label=str(subset_label),
                    polynomial=subset_poly,
                    family_id=str(family_id),
                    num_sites=int(num_sites),
                    ordering=str(ordering),
                    qpb=int(qpb),
                    split_policy="runtime_split_child_set",
                    parent_generator_id=parent_generator_id,
                    symmetry_spec=_symmetry_spec_with_runtime_gate(
                        base_spec=symmetry_spec,
                        gate=symmetry_gate,
                    ),
                )
            )
            compile_metadata = dict(subset_meta.get("compile_metadata", {}))
            compile_metadata["runtime_split"] = {
                "mode": str(split_mode),
                "parent_label": str(parent_label),
                "child_indices": [int(idx) for idx in child_indices],
                "child_labels": [str(label) for label in child_labels],
                "child_generator_ids": [str(x) for x in child_generator_ids],
                "child_count": int(len(child_rows)),
                "representation": "child_set",
                "symmetry_gate": dict(symmetry_gate),
            }
            compile_metadata["serialized_terms_exyz"] = [dict(term) for term in serialized_subset]
            subset_meta["compile_metadata"] = compile_metadata
            out.append(
                {
                    "candidate_label": str(subset_label),
                    "candidate_polynomial": subset_poly,
                    "candidate_generator_metadata": dict(subset_meta),
                    "child_indices": [int(idx) for idx in child_indices],
                    "child_labels": [str(label) for label in child_labels],
                    "child_generator_ids": [str(x) for x in child_generator_ids],
                    "symmetry_gate": dict(symmetry_gate),
                }
            )
            seen_signatures.add(subset_signature)
    return out


def selected_generator_metadata_for_labels(
    labels: Sequence[str],
    registry: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label in labels:
        meta = registry.get(str(label))
        if isinstance(meta, Mapping):
            out.append(dict(meta))
    return out


def build_split_event(
    *,
    parent_generator_id: str,
    child_generator_ids: Sequence[str],
    reason: str,
    split_mode: str,
    probe_trigger: str | None = None,
    choice_reason: str | None = None,
    parent_score: float | None = None,
    child_scores: Mapping[str, float] | None = None,
    admissible_child_subsets: Sequence[Sequence[str]] | None = None,
    chosen_representation: str = "parent",
    chosen_child_ids: Sequence[str] | None = None,
    split_margin: float | None = None,
    symmetry_gate_results: Mapping[str, Any] | None = None,
    compiled_cost_parent: float | None = None,
    compiled_cost_children: float | None = None,
    insertion_positions: Sequence[int] | None = None,
) -> dict[str, Any]:
    event = GeneratorSplitEvent(
        parent_generator_id=str(parent_generator_id),
        child_generator_ids=[str(x) for x in child_generator_ids],
        reason=str(reason),
        split_mode=str(split_mode),
        probe_trigger=(str(probe_trigger) if probe_trigger is not None else None),
        choice_reason=(str(choice_reason) if choice_reason is not None else None),
        parent_score=(float(parent_score) if parent_score is not None else None),
        child_scores=(
            {str(key): float(val) for key, val in child_scores.items()}
            if isinstance(child_scores, Mapping)
            else {}
        ),
        admissible_child_subsets=(
            [[str(x) for x in subset] for subset in admissible_child_subsets]
            if admissible_child_subsets is not None
            else []
        ),
        chosen_representation=str(chosen_representation),
        chosen_child_ids=([str(x) for x in chosen_child_ids] if chosen_child_ids is not None else []),
        split_margin=(float(split_margin) if split_margin is not None else None),
        symmetry_gate_results=(
            dict(symmetry_gate_results) if isinstance(symmetry_gate_results, Mapping) else {}
        ),
        compiled_cost_parent=(float(compiled_cost_parent) if compiled_cost_parent is not None else None),
        compiled_cost_children=(float(compiled_cost_children) if compiled_cost_children is not None else None),
        insertion_positions=([int(x) for x in insertion_positions] if insertion_positions is not None else []),
    )
    return asdict(event)

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/hardcoded/hh_continuation_rescue.py
```py
#!/usr/bin/env python3
"""Simulator-only overlap-guided rescue helpers for HH continuation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class RescueConfig:
    enabled: bool = False
    simulator_only: bool = True
    recent_drop_patience: int = 2
    weak_drop_threshold: float = 1e-6
    shortlist_flat_ratio: float = 0.95
    max_candidates: int = 6
    min_overlap_gain: float = 1e-7


def should_trigger_rescue(
    *,
    enabled: bool,
    exact_state_available: bool,
    residual_opened: bool,
    trough_detected: bool,
    history: Sequence[Mapping[str, Any]],
    shortlist_records: Sequence[Mapping[str, Any]],
    cfg: RescueConfig,
) -> tuple[bool, str]:
    if not bool(enabled):
        return False, "disabled"
    if not bool(exact_state_available):
        return False, "exact_state_unavailable"
    if not (bool(residual_opened) or bool(trough_detected)):
        return False, "residual_not_open_or_no_trough"
    need = int(max(1, cfg.recent_drop_patience))
    recent = [row for row in history if isinstance(row, Mapping)][-need:]
    if len(recent) < need:
        return False, "insufficient_history"
    if any(float(row.get("delta_abs_drop_from_prev", 1.0)) > float(cfg.weak_drop_threshold) for row in recent):
        return False, "drop_not_flat"
    if len(shortlist_records) < 2:
        return False, "shortlist_too_small"
    top = float(shortlist_records[0].get("full_v2_score", shortlist_records[0].get("simple_score", 0.0)))
    second = float(shortlist_records[1].get("full_v2_score", shortlist_records[1].get("simple_score", 0.0)))
    if top <= 0.0:
        return False, "nonpositive_shortlist"
    if second < float(cfg.shortlist_flat_ratio) * top:
        return False, "shortlist_not_flat"
    return True, "flat_drop_and_shortlist"


def rank_rescue_candidates(
    *,
    records: Sequence[Mapping[str, Any]],
    overlap_gain_fn: Callable[[Mapping[str, Any]], float],
    cfg: RescueConfig,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for rec in list(records)[: int(max(1, cfg.max_candidates))]:
        gain = float(overlap_gain_fn(rec))
        ranked.append(
            {
                **dict(rec),
                "overlap_gain": float(gain),
            }
        )
    ranked = sorted(
        ranked,
        key=lambda rec: (
            -float(rec.get("overlap_gain", 0.0)),
            -float(rec.get("full_v2_score", rec.get("simple_score", float("-inf")))),
            -float(rec.get("simple_score", float("-inf"))),
            int(rec.get("candidate_pool_index", -1)),
            int(rec.get("position_id", -1)),
        ),
    )
    if not ranked:
        return None, {"executed": False, "reason": "no_candidates", "ranked": []}
    best = dict(ranked[0])
    if float(best.get("overlap_gain", 0.0)) <= float(cfg.min_overlap_gain):
        return None, {
            "executed": True,
            "reason": "insufficient_overlap_gain",
            "ranked": [dict(x) for x in ranked],
        }
    return best, {
        "executed": True,
        "reason": "selected",
        "ranked": [dict(x) for x in ranked],
    }

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/hardcoded/hh_continuation_scoring.py
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


class Phase1CompileCostOracle:
    """Built-in math expression:
    D_proxy = gate_proxy + shift_span + active_count
    """

    @staticmethod
    def _pauli_weight(label_exyz: str) -> int:
        return int(sum(1 for ch in str(label_exyz) if ch in {"x", "y", "z"}))

    @staticmethod
    def _pauli_xy_count(label_exyz: str) -> int:
        return int(sum(1 for ch in str(label_exyz) if ch in {"x", "y"}))

    @classmethod
    def _cx_proxy_term(cls, label_exyz: str) -> int:
        return int(2 * max(cls._pauli_weight(label_exyz) - 1, 0))

    @classmethod
    def _sq_proxy_term(cls, label_exyz: str) -> int:
        weight = cls._pauli_weight(label_exyz)
        if weight <= 0:
            return 0
        return int(2 * cls._pauli_xy_count(label_exyz) + 1)

    def estimate(
        self,
        *,
        candidate_term_count: int,
        position_id: int,
        append_position: int,
        refit_active_count: int,
        candidate_term: Any | None = None,
    ) -> CompileCostEstimate:
        candidate_labels = _pauli_labels_from_term(candidate_term)
        active_labels = [str(lbl) for lbl in candidate_labels if _pauli_weight_exyz(str(lbl)) > 0]
        if active_labels:
            new_pauli_actions = float(len(active_labels))
            new_rotation_steps = float(len(active_labels))
            cx_proxy_total = float(sum(self._cx_proxy_term(lbl) for lbl in active_labels))
            sq_proxy_total = float(sum(self._sq_proxy_term(lbl) for lbl in active_labels))
            gate_proxy_total = float(cx_proxy_total + 0.5 * sq_proxy_total)
            max_pauli_weight = float(max(self._pauli_weight(lbl) for lbl in active_labels))
        else:
            fallback_count = float(max(1, int(candidate_term_count)))
            new_pauli_actions = fallback_count
            new_rotation_steps = fallback_count
            cx_proxy_total = fallback_count
            sq_proxy_total = fallback_count
            gate_proxy_total = fallback_count
            max_pauli_weight = 0.0
        position_shift_span = float(abs(int(append_position) - int(position_id)))
        refit_active = float(max(0, int(refit_active_count)))
        total = float(gate_proxy_total + position_shift_span + refit_active)
        return CompileCostEstimate(
            new_pauli_actions=new_pauli_actions,
            new_rotation_steps=new_rotation_steps,
            position_shift_span=position_shift_span,
            refit_active_count=refit_active,
            proxy_total=total,
            cx_proxy_total=cx_proxy_total,
            sq_proxy_total=sq_proxy_total,
            gate_proxy_total=gate_proxy_total,
            max_pauli_weight=max_pauli_weight,
        )


class MeasurementCacheAudit:
    """Phase 1 accounting-only grouped reuse tracker."""

    def __init__(
        self,
        nominal_shots_per_group: int = 1,
        *,
        plan_version: str = "phase1_qwc_basis_cover_reuse",
        grouping_mode: str = "qwc_basis_cover_reuse",
    ) -> None:
        self._seen_groups: set[str] = set()
        self._nominal_shots = int(max(1, nominal_shots_per_group))
        self._plan_version = str(plan_version)
        self._grouping_mode = str(grouping_mode)

    def plan_for(self, group_keys: Iterable[str]) -> MeasurementPlan:
        unique_keys = _compress_measurement_group_keys([str(k) for k in group_keys if str(k) != ""])
        return MeasurementPlan(
            plan_version=str(self._plan_version),
            group_keys=list(unique_keys),
            nominal_shots_per_group=int(self._nominal_shots),
            grouping_mode=str(self._grouping_mode),
        )

    def estimate(self, group_keys: Iterable[str]) -> MeasurementCacheStats:
        plan = self.plan_for(group_keys)
        unique_keys = list(plan.group_keys)

        groups_total = int(len(unique_keys))
        groups_reused = 0
        seen_keys = list(self._seen_groups)
        for key in unique_keys:
            if any(_measurement_basis_key_covers(str(key), str(seen)) for seen in seen_keys):
                groups_reused += 1
        groups_new = int(groups_total - groups_reused)
        shots_reused = float(groups_reused * self._nominal_shots)
        shots_new = float(groups_new * self._nominal_shots)
        reuse_count_cost = float(groups_new)
        return MeasurementCacheStats(
            groups_total=groups_total,
            groups_reused=int(groups_reused),
            groups_new=int(groups_new),
            shots_reused=shots_reused,
            shots_new=shots_new,
            reuse_count_cost=reuse_count_cost,
        )

    def commit(self, group_keys: Iterable[str]) -> None:
        for key in _compress_measurement_group_keys(group_keys):
            key_s = str(key)
            if key_s == "":
                continue
            if any(_measurement_basis_key_covers(key_s, seen) for seen in self._seen_groups):
                continue
            covered = {seen for seen in self._seen_groups if _measurement_basis_key_covers(seen, key_s)}
            if covered:
                self._seen_groups -= covered
            self._seen_groups.add(key_s)

    def summary(self) -> dict[str, float]:
        return {
            "groups_known": float(len(self._seen_groups)),
            "nominal_shots_per_group": float(self._nominal_shots),
            "plan_version": str(self._plan_version),
            "grouping_mode": str(self._grouping_mode),
        }


def _replace_feature(feat: CandidateFeatures, **updates: Any) -> CandidateFeatures:
    return CandidateFeatures(**{**feat.__dict__, **updates})


def _pauli_weight_exyz(label_exyz: str) -> int:
    return int(sum(1 for ch in str(label_exyz) if ch in {"x", "y", "z"}))


def _measurement_basis_key_covers(required_key: str, seen_key: str) -> bool:
    req = str(required_key)
    seen = str(seen_key)
    if len(req) != len(seen):
        return False
    return all((r == "e") or (r == s) for r, s in zip(req, seen))


def _measurement_basis_key_merge(lhs_key: str, rhs_key: str) -> str | None:
    lhs = str(lhs_key)
    rhs = str(rhs_key)
    if len(lhs) != len(rhs):
        return None
    merged: list[str] = []
    for lhs_ch, rhs_ch in zip(lhs, rhs):
        if lhs_ch == "e":
            merged.append(rhs_ch)
            continue
        if rhs_ch in {"e", lhs_ch}:
            merged.append(lhs_ch)
            continue
        return None
    return "".join(merged)


def _compress_measurement_group_keys(group_keys: Iterable[str]) -> list[str]:
    ordered = sorted(
        {str(key) for key in group_keys if str(key) != ""},
        key=lambda key: (-_pauli_weight_exyz(str(key)), str(key)),
    )
    kept: list[str] = []
    for key in ordered:
        if any(_measurement_basis_key_covers(str(key), existing) for existing in kept):
            continue
        kept = [existing for existing in kept if not _measurement_basis_key_covers(existing, str(key))]
        kept.append(str(key))
    return kept


def _measurement_group_keys_from_labels(labels: Sequence[str]) -> list[str]:
    active_labels = sorted(
        {str(lbl) for lbl in labels if _pauli_weight_exyz(str(lbl)) > 0},
        key=lambda lbl: (-_pauli_weight_exyz(lbl), lbl),
    )
    groups: list[str] = []
    for label in active_labels:
        best_idx: int | None = None
        best_key: str | None = None
        best_delta: tuple[int, int] | None = None
        for idx, group_key in enumerate(groups):
            merged = _measurement_basis_key_merge(str(group_key), str(label))
            if merged is None:
                continue
            delta = (_pauli_weight_exyz(merged) - _pauli_weight_exyz(str(group_key)), idx)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_idx = int(idx)
                best_key = str(merged)
        if best_idx is None or best_key is None:
            groups.append(str(label))
        else:
            groups[best_idx] = str(best_key)
    return _compress_measurement_group_keys(groups)


def measurement_group_keys_for_term(term: Any) -> list[str]:
    return _measurement_group_keys_from_labels(_pauli_labels_from_term(term))


def _measurement_group_overlap_score(keys_a: Sequence[str], keys_b: Sequence[str]) -> float:
    groups_a = _compress_measurement_group_keys(keys_a)
    groups_b = _compress_measurement_group_keys(keys_b)
    if not groups_a or not groups_b:
        return 1.0

    def _directional(required_groups: Sequence[str], seen_groups: Sequence[str]) -> float:
        if not required_groups:
            return 1.0
        covered = 0
        for req in required_groups:
            if any(_measurement_basis_key_covers(str(req), str(seen)) for seen in seen_groups):
                covered += 1
        return float(covered / len(required_groups))

    return float(
        0.5 * (
            _directional(groups_a, groups_b)
            + _directional(groups_b, groups_a)
        )
    )


def simple_v1_score(
    feat: CandidateFeatures,
    cfg: SimpleScoreConfig,
) -> float:
    if not bool(feat.stage_gate_open):
        return float("-inf")
    if not bool(feat.leakage_gate_open):
        return float("-inf")
    if not bool(feat.compile_gate_open):
        return float("-inf")

    compile_proxy = float(feat.compile_cost_total)
    groups_new = float(feat.measurement_cache_stats.get("groups_new", 0.0))
    shots_new = float(feat.measurement_cache_stats.get("shots_new", 0.0))
    reuse_count_cost = float(feat.measurement_cache_stats.get("reuse_count_cost", 0.0))
    leakage_penalty = float(feat.leakage_penalty)

    score = (
        float(feat.g_abs) + float(cfg.lambda_F) * float(feat.metric_proxy)
        - float(cfg.lambda_compile) * compile_proxy
        - float(cfg.lambda_measure) * (groups_new + shots_new + reuse_count_cost)
        - float(cfg.lambda_leak) * leakage_penalty
    )
    return float(score)


def normalize(value: float, ref: float) -> float:
    denom = float(ref)
    if not math.isfinite(denom) or denom <= 0.0:
        return float(max(0.0, value))
    return float(max(0.0, value) / denom)


def trust_region_drop(g_lcb: float, h_eff: float, F: float, rho: float) -> float:
    if float(g_lcb) <= 0.0 or float(F) <= 0.0:
        return 0.0
    h_eff_pos = float(max(0.0, h_eff))
    alpha_max = float(rho) / float(math.sqrt(float(F)))
    if h_eff_pos > 0.0:
        alpha_newton = float(g_lcb) / h_eff_pos
        if alpha_newton <= alpha_max:
            return float(0.5 * float(g_lcb) * float(g_lcb) / h_eff_pos)
    alpha = float(alpha_max)
    return float(float(g_lcb) * alpha - 0.5 * h_eff_pos * alpha * alpha)


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
        + float(cfg.wC) * normalize(float(feat.new_shot_cost), float(cfg.shot_ref))
        + float(cfg.wP) * normalize(float(feat.opt_dim_cost), float(cfg.optdim_ref))
        + float(cfg.wc) * normalize(float(feat.reuse_count_cost), float(cfg.reuse_ref))
    )
    lifetime_components = lifetime_weight_components(feat, cfg)
    K = float(K + float(cfg.lifetime_weight) * float(lifetime_components.get("total", 0.0)))
    score = (
        math.exp(-float(cfg.eta_L) * float(feat.leakage_penalty))
        * (float(novelty) ** float(cfg.gamma_N))
        * float(delta_e)
        / float(K)
    )
    score += float(cfg.motif_bonus_weight) * float(max(0.0, feat.motif_bonus))
    return float(score), str(fallback_mode)


def shortlist_records(
    records: Sequence[Mapping[str, Any]],
    *,
    cfg: FullScoreConfig,
    score_key: str = "simple_score",
) -> list[dict[str, Any]]:
    ranked = sorted(
        [dict(rec) for rec in records],
        key=lambda rec: (
            -float(rec.get(score_key, float("-inf"))),
            -float(rec.get("simple_score", float("-inf"))),
            int(rec.get("candidate_pool_index", -1)),
            int(rec.get("position_id", -1)),
        ),
    )
    if not ranked:
        return []
    total = int(len(ranked))
    target = int(max(1, min(total, cfg.shortlist_size, math.ceil(float(cfg.shortlist_fraction) * total))))
    out: list[dict[str, Any]] = []
    for idx, rec in enumerate(ranked[:target], start=1):
        updated = dict(rec)
        feat = updated.get("feature", None)
        if isinstance(feat, CandidateFeatures):
            updated["feature"] = _replace_feature(
                feat,
                shortlist_rank=int(idx),
                shortlist_size=int(target),
            )
        out.append(updated)
    return out


def _compiled_for_label(
    *,
    label: str,
    polynomial: Any,
    compiled_cache: dict[str, CompiledPolynomialAction] | None,
    pauli_action_cache: dict[str, Any] | None,
) -> CompiledPolynomialAction:
    cache = compiled_cache if compiled_cache is not None else {}
    key = str(label)
    compiled = cache.get(key)
    if compiled is None:
        compiled = compile_polynomial_action(
            polynomial,
            tol=1e-12,
            pauli_action_cache=pauli_action_cache,
        )
        cache[key] = compiled
    return compiled


def _tangent_data(
    *,
    psi_state: np.ndarray,
    label: str,
    polynomial: Any,
    compiled_cache: dict[str, CompiledPolynomialAction] | None,
    pauli_action_cache: dict[str, Any] | None,
) -> tuple[np.ndarray, float]:
    psi = np.asarray(psi_state, dtype=complex).reshape(-1)
    compiled = _compiled_for_label(
        label=str(label),
        polynomial=polynomial,
        compiled_cache=compiled_cache,
        pauli_action_cache=pauli_action_cache,
    )
    apsi = apply_compiled_polynomial(psi, compiled)
    mean = complex(np.vdot(psi, apsi))
    centered = np.asarray(apsi - mean * psi, dtype=complex)
    F = float(max(0.0, np.real(np.vdot(centered, centered))))
    return centered, F


def _tangent_overlap_matrix(tangents: Sequence[np.ndarray]) -> np.ndarray:
    n = int(len(tangents))
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            val = float(np.real(np.vdot(tangents[i], tangents[j])))
            out[i, j] = val
            out[j, i] = val
    return out


class Phase2NoveltyOracle:
    """Shortlist-only tangent novelty using the current statevector."""

    def estimate(
        self,
        *,
        psi_state: np.ndarray,
        candidate_label: str,
        candidate_term: Any,
        window_terms: Sequence[Any],
        window_labels: Sequence[str],
        compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
        pauli_action_cache: dict[str, Any] | None = None,
        novelty_eps: float = 1e-6,
    ) -> Mapping[str, Any]:
        candidate_tangent, F_metric = _tangent_data(
            psi_state=psi_state,
            label=str(candidate_label),
            polynomial=candidate_term.polynomial,
            compiled_cache=compiled_cache,
            pauli_action_cache=pauli_action_cache,
        )
        if F_metric <= 0.0:
            return {
                "novelty": 0.0,
                "novelty_mode": "exact_statevector_tangent_window_proxy",
                "candidate_tangent": candidate_tangent,
                "window_tangents": [],
                "window_overlap": [],
                "window_gram": [],
                "F_metric": float(F_metric),
            }
        if not window_terms:
            return {
                "novelty": 1.0,
                "novelty_mode": "exact_statevector_tangent_window_proxy",
                "candidate_tangent": candidate_tangent,
                "window_tangents": [],
                "window_overlap": [],
                "window_gram": [],
                "F_metric": float(F_metric),
            }
        window_tangents = []
        for lbl, term in zip(window_labels, window_terms):
            tangent_j, _ = _tangent_data(
                psi_state=psi_state,
                label=str(lbl),
                polynomial=term.polynomial,
                compiled_cache=compiled_cache,
                pauli_action_cache=pauli_action_cache,
            )
            window_tangents.append(tangent_j)
        overlap = np.asarray(
            [float(np.real(np.vdot(tj, candidate_tangent))) for tj in window_tangents],
            dtype=float,
        )
        gram = _tangent_overlap_matrix(window_tangents)
        try:
            novelty_raw = 1.0 - float(
                overlap.T
                @ np.linalg.solve(gram + float(novelty_eps) * np.eye(gram.shape[0], dtype=float), overlap)
                / float(F_metric)
            )
            novelty_val = float(min(1.0, max(0.0, novelty_raw)))
        except Exception:
            novelty_val = 1.0
        return {
            "novelty": float(novelty_val),
            "novelty_mode": "exact_statevector_tangent_window_proxy",
            "candidate_tangent": candidate_tangent,
            "window_tangents": list(window_tangents),
            "window_overlap": [float(x) for x in overlap.tolist()],
            "window_gram": [[float(x) for x in row] for row in gram.tolist()],
            "F_metric": float(F_metric),
        }


class Phase2CurvatureOracle:
    """Shortlist-only Schur-style metric/curvature proxy."""

    def estimate(
        self,
        *,
        base_feature: CandidateFeatures,
        novelty_info: Mapping[str, Any],
        optimizer_memory: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        F_metric = float(max(0.0, novelty_info.get("F_metric", base_feature.F_metric)))
        overlaps = [float(x) for x in novelty_info.get("window_overlap", [])]
        gram_rows = [
            [float(x) for x in row]
            for row in novelty_info.get("window_gram", [])
            if isinstance(row, Sequence)
        ]
        if not overlaps or not gram_rows:
            return {
                "h_hat": float(F_metric),
                "b_hat": None,
                "H_window": None,
                "curvature_mode": "lambda_F_metric_proxy_only",
            }

        H_window = np.asarray(gram_rows, dtype=float)
        if H_window.ndim != 2 or H_window.shape[0] != H_window.shape[1]:
            return {
                "h_hat": float(F_metric),
                "b_hat": None,
                "H_window": None,
                "curvature_mode": "lambda_F_metric_proxy_only",
            }
        if isinstance(optimizer_memory, Mapping):
            raw_diag = list(optimizer_memory.get("preconditioner_diag", []))
            if raw_diag:
                mem_diag = []
                for local_idx, _global_idx in enumerate(base_feature.refit_window_indices):
                    if local_idx < len(raw_diag):
                        denom = max(float(raw_diag[local_idx]), 1e-8)
                        mem_diag.append(1.0 / denom)
                    else:
                        mem_diag.append(0.0)
                if len(mem_diag) == H_window.shape[0]:
                    H_window = H_window + np.diag(np.asarray(mem_diag, dtype=float))
                    mode = "schur_metric_proxy_with_memory_diag"
                else:
                    mode = "schur_metric_proxy"
            else:
                mode = "schur_metric_proxy"
        else:
            mode = "schur_metric_proxy"
        return {
            "h_hat": float(F_metric),
            "b_hat": [float(x) for x in overlaps],
            "H_window": [[float(x) for x in row] for row in H_window.tolist()],
            "curvature_mode": str(mode),
        }


def _pauli_labels_from_term(term: Any) -> list[str]:
    labels: list[str] = []
    if term is None or not hasattr(term, "polynomial"):
        return labels
    for poly_term in term.polynomial.return_polynomial():
        labels.append(str(poly_term.pw2strng()))
    return labels


def _support_set(term: Any) -> set[int]:
    support: set[int] = set()
    labels = _pauli_labels_from_term(term)
    for label in labels:
        for idx, ch in enumerate(str(label)):
            if ch != "e":
                support.add(int(idx))
    return support


def _pauli_strings_commute(lhs: str, rhs: str) -> bool:
    anticomm = 0
    for a, b in zip(str(lhs), str(rhs)):
        if a == "e" or b == "e" or a == b:
            continue
        anticomm += 1
    return bool((anticomm % 2) == 0)


def _polynomials_commute(term_a: Any, term_b: Any) -> bool:
    labels_a = _pauli_labels_from_term(term_a)
    labels_b = _pauli_labels_from_term(term_b)
    if not labels_a or not labels_b:
        return True
    for lhs in labels_a:
        for rhs in labels_b:
            if not _pauli_strings_commute(lhs, rhs):
                return False
    return True


def build_full_candidate_features(
    *,
    base_feature: CandidateFeatures,
    psi_state: np.ndarray,
    candidate_term: Any,
    window_terms: Sequence[Any],
    window_labels: Sequence[str],
    cfg: FullScoreConfig,
    novelty_oracle: NoveltyOracle,
    curvature_oracle: CurvatureOracle,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
    optimizer_memory: Mapping[str, Any] | None = None,
    motif_library: Mapping[str, Any] | None = None,
    target_num_sites: int | None = None,
) -> CandidateFeatures:
    novelty_info = novelty_oracle.estimate(
        psi_state=psi_state,
        candidate_label=str(base_feature.candidate_label),
        candidate_term=candidate_term,
        window_terms=list(window_terms),
        window_labels=list(window_labels),
        compiled_cache=compiled_cache,
        pauli_action_cache=pauli_action_cache,
        novelty_eps=float(cfg.novelty_eps),
    )
    curvature_info = curvature_oracle.estimate(
        base_feature=base_feature,
        novelty_info=novelty_info,
        optimizer_memory=optimizer_memory,
    )
    feat = _replace_feature(
        base_feature,
        novelty=(
            None
            if novelty_info.get("novelty") is None
            else float(novelty_info.get("novelty", 1.0))
        ),
        novelty_mode=str(novelty_info.get("novelty_mode", base_feature.novelty_mode)),
        F_metric=float(max(0.0, novelty_info.get("F_metric", base_feature.F_metric))),
        metric_proxy=float(max(0.0, novelty_info.get("F_metric", base_feature.metric_proxy))),
        h_hat=(
            None
            if curvature_info.get("h_hat") is None
            else float(curvature_info.get("h_hat", 0.0))
        ),
        b_hat=(
            None
            if curvature_info.get("b_hat") is None
            else [float(x) for x in curvature_info.get("b_hat", [])]
        ),
        H_window=(
            None
            if curvature_info.get("H_window") is None
            else [[float(x) for x in row] for row in curvature_info.get("H_window", [])]
        ),
        curvature_mode=str(curvature_info.get("curvature_mode", base_feature.curvature_mode)),
        score_version=str(cfg.score_version),
        motif_bonus=float(base_feature.motif_bonus),
        placeholder_hooks={
            **dict(base_feature.placeholder_hooks),
            "novelty_oracle": True,
            "curvature_oracle": True,
            "full_v2_score": True,
        },
    )
    if isinstance(base_feature.generator_metadata, Mapping) and isinstance(motif_library, Mapping):
        motif_bonus, motif_meta = motif_bonus_for_generator(
            generator_metadata=base_feature.generator_metadata,
            motif_library=motif_library,
            target_num_sites=int(max(0, target_num_sites or 0)),
        )
        feat = _replace_feature(
            feat,
            motif_bonus=float(motif_bonus),
            motif_source=(
                str(motif_library.get("source_tag", "payload"))
                if bool(motif_bonus) else str(feat.motif_source)
            ),
            motif_metadata=(dict(motif_meta) if isinstance(motif_meta, Mapping) else feat.motif_metadata),
        )
    feat = _replace_feature(
        feat,
        lifetime_weight_components=dict(lifetime_weight_components(feat, cfg)),
        lifetime_cost_mode=str(cfg.lifetime_cost_mode),
        remaining_evaluations_proxy_mode=str(cfg.remaining_evaluations_proxy_mode),
    )
    score, fallback_mode = full_v2_score(feat, cfg)
    return _replace_feature(
        feat,
        full_v2_score=float(score),
        actual_fallback_mode=str(fallback_mode),
    )


def compatibility_penalty(
    *,
    record_a: Mapping[str, Any],
    record_b: Mapping[str, Any],
    cfg: FullScoreConfig,
    psi_state: np.ndarray | None = None,
    compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
) -> dict[str, float]:
    feat_a = record_a.get("feature")
    feat_b = record_b.get("feature")
    term_a = record_a.get("candidate_term")
    term_b = record_b.get("candidate_term")
    if not isinstance(feat_a, CandidateFeatures) or not isinstance(feat_b, CandidateFeatures):
        return {
            "support_overlap": 0.0,
            "noncommutation": 0.0,
            "cross_curvature": 0.0,
            "schedule": 0.0,
            "measurement_mismatch": 0.0,
            "total": 0.0,
        }

    supp_a = _support_set(term_a)
    supp_b = _support_set(term_b)
    union = len(supp_a | supp_b)
    support_overlap = 0.0 if union == 0 else float(len(supp_a & supp_b) / union)
    noncomm = 0.0 if _polynomials_commute(term_a, term_b) else 1.0

    cross_curv = 0.0
    if psi_state is not None and term_a is not None and term_b is not None:
        try:
            tang_a, F_a = _tangent_data(
                psi_state=np.asarray(psi_state, dtype=complex),
                label=str(feat_a.candidate_label),
                polynomial=term_a.polynomial,
                compiled_cache=compiled_cache,
                pauli_action_cache=pauli_action_cache,
            )
            tang_b, F_b = _tangent_data(
                psi_state=np.asarray(psi_state, dtype=complex),
                label=str(feat_b.candidate_label),
                polynomial=term_b.polynomial,
                compiled_cache=compiled_cache,
                pauli_action_cache=pauli_action_cache,
            )
            denom = math.sqrt(max(F_a, 0.0) * max(F_b, 0.0))
            if denom > 0.0:
                cross_curv = float(min(1.0, abs(float(np.real(np.vdot(tang_a, tang_b)))) / denom))
        except Exception:
            cross_curv = float(support_overlap)
    elif feat_a.b_hat is not None and feat_b.b_hat is not None:
        vec_a = np.asarray(feat_a.b_hat, dtype=float)
        vec_b = np.asarray(feat_b.b_hat, dtype=float)
        denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        if denom > 0.0:
            cross_curv = float(min(1.0, abs(float(vec_a @ vec_b)) / denom))

    win_a = set(int(i) for i in feat_a.refit_window_indices)
    win_b = set(int(i) for i in feat_b.refit_window_indices)
    union_w = len(win_a | win_b)
    schedule = 0.0 if union_w == 0 else float(len(win_a & win_b) / union_w)
    measurement_overlap = _measurement_group_overlap_score(
        measurement_group_keys_for_term(term_a),
        measurement_group_keys_for_term(term_b),
    )
    measurement_mismatch = float(1.0 - measurement_overlap)
    total = (
        float(cfg.compat_overlap_weight) * float(support_overlap)
        + float(cfg.compat_comm_weight) * float(noncomm)
        + float(cfg.compat_curv_weight) * float(cross_curv)
        + float(cfg.compat_sched_weight) * float(schedule)
        + float(cfg.compat_measure_weight) * float(measurement_mismatch)
    )
    return {
        "support_overlap": float(support_overlap),
        "noncommutation": float(noncomm),
        "cross_curvature": float(cross_curv),
        "schedule": float(schedule),
        "measurement_mismatch": float(measurement_mismatch),
        "total": float(total),
    }


class CompatibilityPenaltyOracle:
    def __init__(
        self,
        *,
        cfg: FullScoreConfig,
        psi_state: np.ndarray | None = None,
        compiled_cache: dict[str, CompiledPolynomialAction] | None = None,
        pauli_action_cache: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.psi_state = None if psi_state is None else np.asarray(psi_state, dtype=complex)
        self.compiled_cache = compiled_cache
        self.pauli_action_cache = pauli_action_cache

    def penalty(self, record_a: Mapping[str, Any], record_b: Mapping[str, Any]) -> dict[str, float]:
        return compatibility_penalty(
            record_a=record_a,
            record_b=record_b,
            cfg=self.cfg,
            psi_state=self.psi_state,
            compiled_cache=self.compiled_cache,
            pauli_action_cache=self.pauli_action_cache,
        )


def greedy_batch_select(
    ranked_records: Sequence[Mapping[str, Any]],
    compat_oracle: CompatibilityPenaltyOracle,
    cfg: FullScoreConfig,
) -> tuple[list[dict[str, Any]], float]:
    ranked = sorted(
        [dict(rec) for rec in ranked_records],
        key=lambda rec: (
            -float(rec.get("full_v2_score", float("-inf"))),
            -float(rec.get("simple_score", float("-inf"))),
            int(rec.get("candidate_pool_index", -1)),
            int(rec.get("position_id", -1)),
        ),
    )
    if not ranked:
        return [], 0.0

    batch: list[dict[str, Any]] = []
    total_penalty = 0.0
    top_score = float(ranked[0].get("full_v2_score", float("-inf")))
    for rec in ranked:
        if len(batch) >= int(max(1, cfg.batch_size_cap)):
            break
        rec_score = float(rec.get("full_v2_score", float("-inf")))
        if not math.isfinite(rec_score) or rec_score <= 0.0:
            continue
        if batch and rec_score < float(cfg.batch_near_degenerate_ratio) * float(top_score):
            continue
        penalty_total = 0.0
        penalty_breakdown = {
            "support_overlap": 0.0,
            "noncommutation": 0.0,
            "cross_curvature": 0.0,
            "schedule": 0.0,
            "measurement_mismatch": 0.0,
        }
        for existing in batch:
            breakdown = compat_oracle.penalty(rec, existing)
            penalty_total += float(breakdown.get("total", 0.0))
            for key in penalty_breakdown:
                penalty_breakdown[key] += float(breakdown.get(key, 0.0))
        if float(rec_score) - float(penalty_total) <= 0.0 and batch:
            continue
        feat = rec.get("feature")
        updated = dict(rec)
        if isinstance(feat, CandidateFeatures):
            updated["feature"] = _replace_feature(
                feat,
                compatibility_penalty_total=float(penalty_total),
            )
        updated["compatibility_penalty"] = {
            **penalty_breakdown,
            "total": float(penalty_total),
        }
        batch.append(updated)
        total_penalty += float(penalty_total)
        if len(batch) >= int(max(1, cfg.batch_target_size)):
            break
    return batch if batch else [dict(ranked[0])], float(total_penalty)


def build_candidate_features(
    *,
    stage_name: str,
    candidate_label: str,
    candidate_family: str,
    candidate_pool_index: int,
    position_id: int,
    append_position: int,
    positions_considered: list[int],
    gradient_signed: float,
    metric_proxy: float,
    sigma_hat: float,
    refit_window_indices: list[int],
    compile_cost: CompileCostEstimate,
    measurement_stats: MeasurementCacheStats,
    leakage_penalty: float,
    stage_gate_open: bool,
    leakage_gate_open: bool,
    trough_probe_triggered: bool,
    trough_detected: bool,
    cfg: SimpleScoreConfig,
    generator_metadata: Mapping[str, Any] | None = None,
    symmetry_spec: Mapping[str, Any] | None = None,
    symmetry_mode: str = "none",
    symmetry_mitigation_mode: str = "off",
    motif_metadata: Mapping[str, Any] | None = None,
    motif_bonus: float = 0.0,
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
        F_metric=float(max(0.0, metric_proxy)),
        metric_proxy=float(max(0.0, metric_proxy)),
        novelty=None,
        curvature_mode="lambda_F_metric_proxy_only",
        novelty_mode="none",
        refit_window_indices=[int(i) for i in refit_window_indices],
        compiled_position_cost_proxy={str(k): float(v) for k, v in proxy_cost.items()},
        measurement_cache_stats={
            "groups_total": float(measurement_stats.groups_total),
            "groups_reused": float(measurement_stats.groups_reused),
            "groups_new": float(measurement_stats.groups_new),
            "shots_reused": float(measurement_stats.shots_reused),
            "shots_new": float(measurement_stats.shots_new),
            "reuse_count_cost": float(measurement_stats.reuse_count_cost),
        },
        leakage_penalty=float(max(0.0, leakage_penalty)),
        stage_gate_open=bool(stage_gate_open),
        leakage_gate_open=bool(leakage_gate_open),
        trough_probe_triggered=bool(trough_probe_triggered),
        trough_detected=bool(trough_detected),
        simple_score=None,
        score_version=str(cfg.score_version),
        depth_cost=float(depth_cost_value),
        new_group_cost=float(measurement_stats.groups_new),
        new_shot_cost=float(measurement_stats.shots_new),
        opt_dim_cost=float(len(refit_window_indices)),
        reuse_count_cost=float(measurement_stats.reuse_count_cost),
        generator_id=(
            str(generator_metadata.get("generator_id"))
            if isinstance(generator_metadata, Mapping) and generator_metadata.get("generator_id") is not None
            else None
        ),
        template_id=(
            str(generator_metadata.get("template_id"))
            if isinstance(generator_metadata, Mapping) and generator_metadata.get("template_id") is not None
            else None
        ),
        is_macro_generator=bool(generator_metadata.get("is_macro_generator", False)) if isinstance(generator_metadata, Mapping) else False,
        parent_generator_id=(
            str(generator_metadata.get("parent_generator_id"))
            if isinstance(generator_metadata, Mapping) and generator_metadata.get("parent_generator_id") is not None
            else None
        ),
        generator_metadata=(dict(generator_metadata) if isinstance(generator_metadata, Mapping) else None),
        symmetry_spec=(dict(symmetry_spec) if isinstance(symmetry_spec, Mapping) else None),
        symmetry_mode=str(symmetry_mode),
        symmetry_mitigation_mode=str(symmetry_mitigation_mode),
        motif_metadata=(dict(motif_metadata) if isinstance(motif_metadata, Mapping) else None),
        motif_bonus=float(max(0.0, motif_bonus)),
        motif_source=str(motif_source),
        remaining_evaluations_proxy=float(remaining_eval_proxy),
        remaining_evaluations_proxy_mode=str(remaining_evaluations_proxy_mode),
        lifetime_cost_mode=str(lifetime_cost_mode),
        lifetime_weight_components={
            "remaining_evaluations_proxy": float(remaining_eval_proxy),
        },
        placeholder_hooks={
            "novelty_oracle": False,
            "curvature_oracle": False,
            "full_v2_score": False,
            "qn_spsa_refresh": False,
            "motif_metadata": False,
            "symmetry_metadata": bool(isinstance(symmetry_spec, Mapping)),
            "backend_compile_oracle": bool(str(compile_cost.source_mode) != "proxy"),
        },
        compile_cost_source=str(compile_cost.source_mode),
        compile_cost_total=float(compile_cost_total),
        compile_gate_open=bool(compile_cost.compile_gate_open),
        compile_failure_reason=(
            None if compile_cost.failure_reason is None else str(compile_cost.failure_reason)
        ),
        compiled_position_cost_backend=(
            None
            if str(compile_cost.source_mode) == "proxy"
            else {
                "selected_backend_name": compile_cost.selected_backend_name,
                "selected_resolution_kind": compile_cost.selected_resolution_kind,
                "aggregation_mode": str(compile_cost.aggregation_mode),
                "target_backend_names": [str(x) for x in compile_cost.target_backend_names],
                "successful_target_count": int(compile_cost.successful_target_count),
                "failed_target_count": int(compile_cost.failed_target_count),
                "raw_delta_compiled_count_2q": compile_cost.raw_delta_compiled_count_2q,
                "delta_compiled_count_2q": compile_cost.delta_compiled_count_2q,
                "raw_delta_compiled_depth": compile_cost.raw_delta_compiled_depth,
                "delta_compiled_depth": compile_cost.delta_compiled_depth,
                "raw_delta_compiled_size": compile_cost.raw_delta_compiled_size,
                "delta_compiled_size": compile_cost.delta_compiled_size,
                "delta_compiled_cx_count": compile_cost.delta_compiled_cx_count,
                "delta_compiled_ecr_count": compile_cost.delta_compiled_ecr_count,
                "base_compiled_count_2q": compile_cost.base_compiled_count_2q,
                "base_compiled_depth": compile_cost.base_compiled_depth,
                "base_compiled_size": compile_cost.base_compiled_size,
                "trial_compiled_count_2q": compile_cost.trial_compiled_count_2q,
                "trial_compiled_depth": compile_cost.trial_compiled_depth,
                "trial_compiled_size": compile_cost.trial_compiled_size,
            }
        ),
    )
    score = simple_v1_score(feat, cfg)
    return _replace_feature(feat, simple_score=float(score))

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/hardcoded/hh_continuation_types.py
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
    source_mode: str = "proxy"
    penalty_total: float | None = None
    depth_surrogate: float | None = None
    compile_gate_open: bool = True
    failure_reason: str | None = None
    selected_backend_name: str | None = None
    selected_resolution_kind: str | None = None
    aggregation_mode: str = "proxy"
    target_backend_names: list[str] = field(default_factory=list)
    successful_target_count: int = 0
    failed_target_count: int = 0
    raw_delta_compiled_count_2q: float | None = None
    delta_compiled_count_2q: float | None = None
    raw_delta_compiled_depth: float | None = None
    delta_compiled_depth: float | None = None
    raw_delta_compiled_size: float | None = None
    delta_compiled_size: float | None = None
    delta_compiled_cx_count: float | None = None
    delta_compiled_ecr_count: float | None = None
    base_compiled_count_2q: float | None = None
    base_compiled_depth: float | None = None
    base_compiled_size: float | None = None
    trial_compiled_count_2q: float | None = None
    trial_compiled_depth: float | None = None
    trial_compiled_size: float | None = None
    proxy_baseline: dict[str, float] | None = None
    selected_backend_row: dict[str, Any] | None = None


@dataclass(frozen=True)
class ScaffoldFingerprintLite:
    selected_operator_labels: list[str]
    selected_generator_ids: list[str]
    num_parameters: int
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
    index: int
    label: str
    accepted: bool
    energy_before: float
    energy_after: float
    regression: float
    reason: str


@dataclass(frozen=True)
class ReplayPlan:
    continuation_mode: str
    seed_policy_resolved: str
    handoff_state_kind: str
    freeze_scaffold_steps: int
    unfreeze_steps: int
    full_replay_steps: int
    trust_radius_initial: float
    trust_radius_growth: float
    trust_radius_max: float
    scaffold_block_indices: list[int]
    residual_block_indices: list[int]
    qn_spsa_refresh_every: int
    trust_radius_schedule: list[float]
    optimizer_memory_source: str = "unavailable"
    optimizer_memory_reused: bool = False
    refresh_mode: str = "disabled"
    symmetry_mitigation_mode: str = "off"
    generator_ids: list[str] = field(default_factory=list)
    motif_reference_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReplayPhaseTelemetry:
    phase_name: str
    nfev: int
    nit: int
    success: bool
    energy_before: float
    energy_after: float
    delta_abs_before: float | None
    delta_abs_after: float | None
    active_count: int
    frozen_count: int
    optimizer_memory_reused: bool = False
    optimizer_memory_source: str = "unavailable"
    qn_spsa_refresh_points: list[int] = field(default_factory=list)
    residual_zero_initialized: bool = True


class NoveltyOracle(Protocol):
    def estimate(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:  # pragma: no cover - interface
        ...


class CurvatureOracle(Protocol):
    def estimate(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:  # pragma: no cover - interface
        ...


class OptimizerMemoryAdapter(Protocol):
    def unavailable(self, *, method: str, parameter_count: int, reason: str) -> dict[str, Any]:
        ...

    def from_result(
        self,
        result: Any,
        *,
        method: str,
        parameter_count: int,
        source: str,
    ) -> dict[str, Any]:
        ...

    def remap_insert(
        self,
        state: Mapping[str, Any] | None,
        *,
        position_id: int,
        count: int = 1,
    ) -> dict[str, Any]:
        ...

    def remap_remove(
        self,
        state: Mapping[str, Any] | None,
        *,
        indices: Sequence[int],
    ) -> dict[str, Any]:
        ...

    def select_active(
        self,
        state: Mapping[str, Any] | None,
        *,
        active_indices: Sequence[int],
        source: str,
    ) -> dict[str, Any]:
        ...

    def merge_active(
        self,
        base_state: Mapping[str, Any] | None,
        *,
        active_indices: Sequence[int],
        active_state: Mapping[str, Any] | None,
        source: str,
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class QNSPSARefreshPlan:
    enabled: bool = False
    refresh_every: int = 0
    mode: str = "disabled"
    skip_reason: str = ""
    refresh_points: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class MotifMetadata:
    enabled: bool = False
    motif_tags: list[str] = field(default_factory=list)
    motif_ids: list[str] = field(default_factory=list)
    motif_source: str = "none"
    tiled_from_num_sites: int | None = None
    target_num_sites: int | None = None
    boundary_behavior: str | None = None
    transfer_mode: str = "exact_match_v1"


@dataclass(frozen=True)
class SymmetrySpec:
    spec_version: str = "phase3_symmetry_v1"
    particle_number_mode: str = "preserving"
    spin_sector_mode: str = "preserving"
    phonon_number_mode: str = "not_conserved"
    leakage_risk: float = 0.0
    mitigation_eligible: bool = False
    grouping_eligible: bool = True
    hard_guard: bool = False
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GeneratorMetadata:
    generator_id: str
    family_id: str
    template_id: str
    candidate_label: str
    support_qubits: list[int]
    support_sites: list[int]
    support_site_offsets: list[int]
    is_macro_generator: bool
    split_policy: str
    parent_generator_id: str | None = None
    symmetry_spec: dict[str, Any] | None = None
    compile_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratorSplitEvent:
    parent_generator_id: str
    child_generator_ids: list[str]
    reason: str
    split_mode: str
    probe_trigger: str | None = None
    choice_reason: str | None = None
    parent_score: float | None = None
    child_scores: dict[str, float] = field(default_factory=dict)
    admissible_child_subsets: list[list[str]] = field(default_factory=list)
    chosen_representation: str = "parent"
    chosen_child_ids: list[str] = field(default_factory=list)
    split_margin: float | None = None
    symmetry_gate_results: dict[str, Any] = field(default_factory=dict)
    compiled_cost_parent: float | None = None
    compiled_cost_children: float | None = None
    insertion_positions: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class MotifRecord:
    motif_id: str
    family_id: str
    template_id: str
    source_num_sites: int
    relative_order: int
    support_site_offsets: list[int]
    mean_theta: float
    mean_abs_theta: float
    sign_hint: int
    generator_ids: list[str]
    symmetry_spec: dict[str, Any] | None = None
    boundary_behavior: str = "interior_only"
    source_tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MotifLibrary:
    library_version: str
    source_tag: str
    source_num_sites: int
    ordering: str
    boson_encoding: str
    source_tags: list[str] = field(default_factory=list)
    records: list[MotifRecord] = field(default_factory=list)


@dataclass(frozen=True)
class RescueDiagnostic:
    enabled: bool = False
    triggered: bool = False
    reason: str = "disabled"
    shortlisted_labels: list[str] = field(default_factory=list)
    selected_label: str | None = None
    selected_position: int | None = None
    overlap_gain: float = 0.0


class Phase2OptimizerMemoryAdapter:
    """Deterministic remapping adapter for persisted optimizer memory."""

    _VECTOR_KEYS = (
        "preconditioner_diag",
        "grad_sq_ema",
    )

    def unavailable(self, *, method: str, parameter_count: int, reason: str) -> dict[str, Any]:
        return {
            "version": "phase2_optimizer_memory_v1",
            "optimizer": str(method),
            "parameter_count": int(max(0, parameter_count)),
            "available": False,
            "reason": str(reason),
            "source": "unavailable",
            "reused": False,
            "preconditioner_diag": [1.0] * int(max(0, parameter_count)),
            "grad_sq_ema": [0.0] * int(max(0, parameter_count)),
            "history_tail": [],
            "refresh_points": [],
            "remap_events": [],
        }

    def from_result(
        self,
        result: Any,
        *,
        method: str,
        parameter_count: int,
        source: str,
    ) -> dict[str, Any]:
        raw = getattr(result, "optimizer_memory", None)
        if not isinstance(raw, Mapping):
            return self.unavailable(
                method=str(method),
                parameter_count=int(parameter_count),
                reason="optimizer_memory_missing",
            )
        state = self._normalize(raw, parameter_count=int(parameter_count))
        state["source"] = str(source)
        return state

    def remap_insert(
        self,
        state: Mapping[str, Any] | None,
        *,
        position_id: int,
        count: int = 1,
    ) -> dict[str, Any]:
        base = self._normalize(state, parameter_count=self._parameter_count(state))
        n = int(base["parameter_count"])
        pos = max(0, min(int(position_id), n))
        add_n = int(max(0, count))
        for key, default in (("preconditioner_diag", 1.0), ("grad_sq_ema", 0.0)):
            vec = list(base.get(key, []))
            base[key] = vec[:pos] + ([float(default)] * add_n) + vec[pos:]
        base["parameter_count"] = int(n + add_n)
        self._append_remap_event(base, {"op": "insert", "position_id": int(pos), "count": int(add_n)})
        return base

    def remap_remove(
        self,
        state: Mapping[str, Any] | None,
        *,
        indices: Sequence[int],
    ) -> dict[str, Any]:
        base = self._normalize(state, parameter_count=self._parameter_count(state))
        n = int(base["parameter_count"])
        drop = sorted({int(i) for i in indices if 0 <= int(i) < n})
        keep = [i for i in range(n) if i not in set(drop)]
        for key in self._VECTOR_KEYS:
            vec = list(base.get(key, []))
            base[key] = [float(vec[i]) for i in keep]
        base["parameter_count"] = int(len(keep))
        self._append_remap_event(base, {"op": "remove", "indices": [int(i) for i in drop]})
        return base

    def select_active(
        self,
        state: Mapping[str, Any] | None,
        *,
        active_indices: Sequence[int],
        source: str,
    ) -> dict[str, Any]:
        base = self._normalize(state, parameter_count=self._parameter_count(state))
        n = int(base["parameter_count"])
        active = [int(i) for i in active_indices if 0 <= int(i) < n]
        out = {
            **base,
            "parameter_count": int(len(active)),
            "preconditioner_diag": [float(base["preconditioner_diag"][i]) for i in active],
            "grad_sq_ema": [float(base["grad_sq_ema"][i]) for i in active],
            "source": str(source),
            "reused": bool(base.get("available", False) and len(active) > 0),
            "active_indices": [int(i) for i in active],
        }
        self._append_remap_event(out, {"op": "select_active", "active_indices": [int(i) for i in active]})
        return out

    def merge_active(
        self,
        base_state: Mapping[str, Any] | None,
        *,
        active_indices: Sequence[int],
        active_state: Mapping[str, Any] | None,
        source: str,
    ) -> dict[str, Any]:
        base = self._normalize(base_state, parameter_count=self._parameter_count(base_state))
        active_norm = self._normalize(active_state, parameter_count=len(list(active_indices)))
        n = int(base["parameter_count"])
        active = [int(i) for i in active_indices if 0 <= int(i) < n]
        for key, default in (("preconditioner_diag", 1.0), ("grad_sq_ema", 0.0)):
            vec = list(base.get(key, [float(default)] * n))
            active_vec = list(active_norm.get(key, []))
            for k, idx in enumerate(active):
                if k < len(active_vec):
                    vec[idx] = float(active_vec[k])
            base[key] = vec
        base["source"] = str(source)
        base["available"] = bool(base.get("available", False) or active_norm.get("available", False))
        base["reused"] = bool(active_norm.get("reused", False))
        refresh = list(base.get("refresh_points", []))
        refresh.extend(int(x) for x in active_norm.get("refresh_points", []) if int(x) not in refresh)
        base["refresh_points"] = refresh
        self._append_remap_event(base, {"op": "merge_active", "active_indices": [int(i) for i in active]})
        return base

    def _parameter_count(self, state: Mapping[str, Any] | None) -> int:
        if isinstance(state, Mapping) and state.get("parameter_count") is not None:
            return int(max(0, int(state.get("parameter_count", 0))))
        if isinstance(state, Mapping):
            for key in self._VECTOR_KEYS:
                raw = state.get(key, None)
                if isinstance(raw, Sequence):
                    return int(len(list(raw)))
        return 0

    def _normalize(self, state: Mapping[str, Any] | None, *, parameter_count: int) -> dict[str, Any]:
        n = int(max(0, parameter_count))
        if not isinstance(state, Mapping):
            return self.unavailable(method="unknown", parameter_count=n, reason="missing_state")
        out = {
            "version": str(state.get("version", "phase2_optimizer_memory_v1")),
            "optimizer": str(state.get("optimizer", "unknown")),
            "parameter_count": int(n),
            "available": bool(state.get("available", False)),
            "reason": str(state.get("reason", "")),
            "source": str(state.get("source", "")),
            "reused": bool(state.get("reused", False)),
            "history_tail": [dict(x) for x in state.get("history_tail", []) if isinstance(x, Mapping)][-32:],
            "refresh_points": [int(x) for x in state.get("refresh_points", [])],
            "remap_events": [dict(x) for x in state.get("remap_events", []) if isinstance(x, Mapping)][-32:],
        }
        for key, default in (("preconditioner_diag", 1.0), ("grad_sq_ema", 0.0)):
            raw = list(state.get(key, [])) if isinstance(state.get(key, []), Sequence) else []
            vec = [float(default)] * n
            for i in range(min(n, len(raw))):
                vec[i] = float(raw[i])
            out[key] = vec
        return out

    def _append_remap_event(self, state: dict[str, Any], event: Mapping[str, Any]) -> None:
        events = [dict(x) for x in state.get("remap_events", []) if isinstance(x, Mapping)]
        events.append({str(k): v for k, v in event.items()})
        state["remap_events"] = events[-32:]

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/test/test_hh_continuation_generators.py
```py
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_continuation_generators import (
    build_generator_metadata,
    build_pool_generator_registry,
    build_runtime_split_child_sets,
    build_runtime_split_children,
    build_split_event,
    rebuild_polynomial_from_serialized_terms,
)
from pipelines.hardcoded.hh_continuation_symmetry import build_symmetry_spec
from src.quantum.pauli_polynomial_class import (
    PauliPolynomial,
    fermion_minus_operator,
    fermion_plus_operator,
)
from src.quantum.pauli_words import PauliTerm


def _term(label: str, poly: PauliPolynomial):
    return type("_DummyAnsatzTerm", (), {"label": str(label), "polynomial": poly})()


def _macro_poly() -> PauliPolynomial:
    return PauliPolynomial(
        "JW",
        [
            PauliTerm(6, ps="eyeexy", pc=1.0),
            PauliTerm(6, ps="eyeeyx", pc=-1.0),
        ],
    )


def _number_preserving_macro_poly() -> PauliPolynomial:
    return (-1j) * (
        fermion_plus_operator("JW", 4, 1) * fermion_minus_operator("JW", 4, 0)
        - fermion_plus_operator("JW", 4, 0) * fermion_minus_operator("JW", 4, 1)
    )


def _mixed_macro_poly() -> PauliPolynomial:
    return _number_preserving_macro_poly() + PauliPolynomial(
        "JW",
        [PauliTerm(4, ps="zeee", pc=0.25)],
    )


def test_build_generator_metadata_is_stable_for_same_structure() -> None:
    sym = build_symmetry_spec(family_id="paop_lf_std", mitigation_mode="verify_only")
    first = build_generator_metadata(
        label="cand",
        polynomial=_macro_poly(),
        family_id="paop_lf_std",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_spec=sym.__dict__,
    )
    second = build_generator_metadata(
        label="cand",
        polynomial=_macro_poly(),
        family_id="paop_lf_std",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_spec=sym.__dict__,
    )
    assert first.generator_id == second.generator_id
    assert first.template_id == second.template_id
    assert first.support_site_offsets == [0, 1]
    assert first.is_macro_generator is True


def test_pool_registry_carries_symmetry_metadata() -> None:
    sym = build_symmetry_spec(family_id="paop_lf_std", mitigation_mode="verify_only")
    registry = build_pool_generator_registry(
        terms=[_term("macro", _macro_poly())],
        family_ids=["paop_lf_std"],
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_specs=[sym.__dict__],
    )
    meta = registry["macro"]
    assert meta["family_id"] == "paop_lf_std"
    assert meta["is_macro_generator"] is True
    assert meta["symmetry_spec"]["mitigation_eligible"] is True
    assert meta["symmetry_spec"]["particle_number_mode"] == "preserving"
    assert meta["compile_metadata"]["symmetry_gate"]["passed"] is True
    assert "operator_symmetry_checked" in meta["symmetry_spec"]["tags"]


def test_build_generator_metadata_hard_guards_base_terms_that_break_required_symmetry() -> None:
    sym = build_symmetry_spec(family_id="uccsd", mitigation_mode="verify_only")
    bad_term = _number_preserving_macro_poly().return_polynomial()[0]
    bad_poly = PauliPolynomial("JW", [bad_term])
    meta = build_generator_metadata(
        label="bad_base_term",
        polynomial=bad_poly,
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_spec=sym.__dict__,
    )
    assert meta.symmetry_spec is not None
    assert meta.symmetry_spec["particle_number_mode"] == "violating"
    assert meta.symmetry_spec["spin_sector_mode"] == "violating"
    assert meta.symmetry_spec["hard_guard"] is True
    assert "operator_symmetry_checked" in meta.symmetry_spec["tags"]
    assert "operator_symmetry_rejected" in meta.symmetry_spec["tags"]
    assert meta.compile_metadata["symmetry_intent"]["particle_number_mode"] == "preserving"
    assert meta.compile_metadata["symmetry_gate"]["passed"] is False


def test_deliberate_split_marks_child_metadata() -> None:
    meta = build_generator_metadata(
        label="child",
        polynomial=_macro_poly(),
        family_id="paop_lf_std",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        split_policy="deliberate_split",
        parent_generator_id="gen:parent",
    )
    assert meta.is_macro_generator is False
    assert meta.parent_generator_id == "gen:parent"
    assert meta.split_policy == "deliberate_split"


def test_build_split_event_keeps_parent_child_provenance() -> None:
    event = build_split_event(
        parent_generator_id="gen:parent",
        child_generator_ids=["gen:c1", "gen:c2"],
        reason="compiled_depth_cap",
        split_mode="selective",
    )
    assert event["parent_generator_id"] == "gen:parent"
    assert event["child_generator_ids"] == ["gen:c1", "gen:c2"]
    assert event["reason"] == "compiled_depth_cap"


def test_build_runtime_split_children_marks_atomic_terms_that_break_required_symmetry() -> None:
    sym = build_symmetry_spec(family_id="uccsd", mitigation_mode="verify_only")
    parent_meta = build_generator_metadata(
        label="macro",
        polynomial=_number_preserving_macro_poly(),
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_spec=sym.__dict__,
    )
    children = build_runtime_split_children(
        parent_label="macro",
        polynomial=_number_preserving_macro_poly(),
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        split_mode="shortlist_pauli_children_v1",
        parent_generator_metadata=parent_meta.__dict__,
        symmetry_spec=sym.__dict__,
    )
    assert len(children) == 2
    assert children[0]["child_label"].startswith("macro::split[0]::")
    assert children[1]["child_label"].startswith("macro::split[1]::")
    for idx, child in enumerate(children):
        meta = child["child_generator_metadata"]
        compile_meta = meta["compile_metadata"]
        assert meta["parent_generator_id"] == parent_meta.generator_id
        assert meta["split_policy"] == "deliberate_split"
        assert meta["is_macro_generator"] is False
        assert compile_meta["runtime_split"]["mode"] == "shortlist_pauli_children_v1"
        assert compile_meta["runtime_split"]["parent_label"] == "macro"
        assert compile_meta["runtime_split"]["child_index"] == idx
        assert compile_meta["runtime_split"]["child_count"] == 2
        assert compile_meta["runtime_split"]["representation"] == "child_atom"
        assert compile_meta["runtime_split"]["symmetry_gate"]["passed"] is False
        assert meta["symmetry_spec"]["particle_number_mode"] == "violating"
        assert meta["symmetry_spec"]["hard_guard"] is True
        assert len(compile_meta["serialized_terms_exyz"]) == 1


def test_build_runtime_split_child_sets_only_returns_symmetry_safe_combinations() -> None:
    sym = build_symmetry_spec(family_id="uccsd", mitigation_mode="verify_only")
    parent_meta = build_generator_metadata(
        label="macro",
        polynomial=_mixed_macro_poly(),
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_spec=sym.__dict__,
    )
    children = build_runtime_split_children(
        parent_label="macro",
        polynomial=_mixed_macro_poly(),
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        split_mode="shortlist_pauli_children_v1",
        parent_generator_metadata=parent_meta.__dict__,
        symmetry_spec=sym.__dict__,
    )
    child_sets = build_runtime_split_child_sets(
        parent_label="macro",
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        split_mode="shortlist_pauli_children_v1",
        children=children,
        parent_generator_metadata=parent_meta.__dict__,
        symmetry_spec=sym.__dict__,
        max_subset_size=3,
    )
    labels = {row["candidate_label"] for row in child_sets}
    assert labels == {"macro::child_set[0]", "macro::child_set[1,2]"}
    by_label = {row["candidate_label"]: row for row in child_sets}
    pair_meta = by_label["macro::child_set[1,2]"]["candidate_generator_metadata"]["compile_metadata"]
    singleton_meta = by_label["macro::child_set[0]"]["candidate_generator_metadata"]["compile_metadata"]
    assert pair_meta["runtime_split"]["representation"] == "child_set"
    assert pair_meta["runtime_split"]["child_indices"] == [1, 2]
    assert pair_meta["runtime_split"]["symmetry_gate"]["passed"] is True
    assert singleton_meta["runtime_split"]["child_indices"] == [0]
    assert by_label["macro::child_set[1,2]"]["candidate_generator_metadata"]["symmetry_spec"]["particle_number_mode"] == "preserving"
    assert len(pair_meta["serialized_terms_exyz"]) == 2


def test_build_split_event_records_probe_choice_details() -> None:
    event = build_split_event(
        parent_generator_id="gen:parent",
        child_generator_ids=["gen:c1", "gen:c2"],
        reason="depth4_shortlist_probe",
        split_mode="shortlist_pauli_children_v1",
        probe_trigger="phase2_shortlist",
        choice_reason="parent_actual_score_better",
        parent_score=1.25,
        child_scores={"c1": 0.8, "c2": 0.7},
        admissible_child_subsets=[["c1", "c2"]],
        chosen_representation="parent",
        chosen_child_ids=[],
        split_margin=-0.1,
        symmetry_gate_results={"passed": True},
        compiled_cost_parent=2.0,
        compiled_cost_children=2.4,
        insertion_positions=[3],
    )
    assert event["probe_trigger"] == "phase2_shortlist"
    assert event["choice_reason"] == "parent_actual_score_better"
    assert event["child_scores"] == {"c1": 0.8, "c2": 0.7}
    assert event["admissible_child_subsets"] == [["c1", "c2"]]
    assert event["chosen_representation"] == "parent"
    assert event["compiled_cost_parent"] == 2.0
    assert event["insertion_positions"] == [3]


def test_rebuild_polynomial_from_serialized_terms_preserves_serialized_order() -> None:
    poly = rebuild_polynomial_from_serialized_terms(
        [
            {"pauli_exyz": "eyezee", "coeff_re": 1.0, "coeff_im": 0.0, "nq": 6},
            {"pauli_exyz": "eyeeez", "coeff_re": -1.0, "coeff_im": 0.0, "nq": 6},
        ]
    )
    assert [term.pw2strng() for term in poly.return_polynomial()] == ["eyezee", "eyeeez"]

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/hardcoded/hh_continuation_symmetry.py
```py
#!/usr/bin/env python3
"""Shared symmetry metadata and verify-only mitigation hooks for HH continuation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

from pipelines.hardcoded.hh_continuation_types import SymmetrySpec


_LOW_RISK_FAMILIES = {
    "paop",
    "paop_min",
    "paop_std",
    "paop_full",
    "paop_lf",
    "paop_lf_std",
    "paop_lf2_std",
    "paop_lf_full",
    "uccsd_paop_lf_full",
    "uccsd",
    "hva",
    "core",
}

_ALLOWED_SYMMETRY_MODES = {"off", "verify_only", "postselect_diag_v1", "projector_renorm_v1"}


def normalize_phase3_symmetry_mitigation_mode(mode: str | None) -> str:
    mode_key = "off" if mode is None else str(mode).strip().lower()
    if mode_key == "":
        return "off"
    if mode_key not in _ALLOWED_SYMMETRY_MODES:
        raise ValueError(
            "phase3_symmetry_mitigation_mode must be one of "
            f"{sorted(_ALLOWED_SYMMETRY_MODES)}."
        )
    return str(mode_key)


def build_symmetry_spec(
    *,
    family_id: str,
    mitigation_mode: str = "off",
) -> SymmetrySpec:
    """Return baseline HH symmetry intent before operator-level auditing."""
    family_key = str(family_id).strip().lower()
    mitigation_mode_key = normalize_phase3_symmetry_mitigation_mode(mitigation_mode)
    if family_key in _LOW_RISK_FAMILIES:
        leakage_risk = 0.0
    elif family_key in {"residual", "full_meta", "pareto_lean", "pareto_lean_l2", "full_hamiltonian"}:
        leakage_risk = 0.1
    else:
        leakage_risk = 0.2
    tags = ["fermion_number", "spin_sector"]
    if mitigation_mode_key in {"postselect_diag_v1", "projector_renorm_v1"}:
        tags.append("active_symmetry_requested")
    return SymmetrySpec(
        particle_number_mode="preserving",
        spin_sector_mode="preserving",
        phonon_number_mode="not_conserved",
        leakage_risk=float(leakage_risk),
        mitigation_eligible=bool(mitigation_mode_key != "off"),
        grouping_eligible=bool(leakage_risk <= 0.2),
        hard_guard=False,
        tags=list(tags),
    )


def symmetry_spec_to_dict(spec: SymmetrySpec | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if isinstance(spec, SymmetrySpec):
        return asdict(spec)
    if isinstance(spec, Mapping):
        return dict(spec)
    return None


def leakage_penalty_from_spec(spec: SymmetrySpec | Mapping[str, Any] | None) -> float:
    if isinstance(spec, SymmetrySpec):
        return float(spec.leakage_risk)
    if isinstance(spec, Mapping):
        return float(spec.get("leakage_risk", 0.0))
    return 0.0


def verify_symmetry_sequence(
    *,
    generator_metadata: Sequence[Mapping[str, Any]],
    mitigation_mode: str,
) -> dict[str, Any]:
    mode = normalize_phase3_symmetry_mitigation_mode(mitigation_mode)
    if mode == "off":
        return {
            "mode": "off",
            "executed": False,
            "active": False,
            "passed": True,
            "high_risk_count": 0,
            "max_leakage_risk": 0.0,
        }
    leakage_values = []
    for meta in generator_metadata:
        spec = meta.get("symmetry_spec", None) if isinstance(meta, Mapping) else None
        leakage_values.append(float(leakage_penalty_from_spec(spec)))
    max_risk = float(max(leakage_values) if leakage_values else 0.0)
    high_risk_count = int(sum(1 for val in leakage_values if float(val) > 0.5))
    return {
        "mode": str(mode),
        "executed": True,
        "active": bool(mode in {"postselect_diag_v1", "projector_renorm_v1"}),
        "passed": bool(high_risk_count == 0),
        "high_risk_count": int(high_risk_count),
        "max_leakage_risk": float(max_risk),
    }

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/test/test_hh_continuation_symmetry.py
```py
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_continuation_symmetry import (
    build_symmetry_spec,
    leakage_penalty_from_spec,
    verify_symmetry_sequence,
)


def test_build_symmetry_spec_shares_metadata_for_gating_and_verification() -> None:
    spec = build_symmetry_spec(family_id="paop_lf_std", mitigation_mode="verify_only")
    leak = leakage_penalty_from_spec(spec)
    verify = verify_symmetry_sequence(
        generator_metadata=[{"symmetry_spec": spec.__dict__}],
        mitigation_mode="verify_only",
    )
    assert leak == 0.0
    assert verify["executed"] is True
    assert verify["passed"] is True
    assert verify["max_leakage_risk"] == 0.0


def test_verify_symmetry_sequence_fails_for_high_risk_metadata() -> None:
    verify = verify_symmetry_sequence(
        generator_metadata=[{"symmetry_spec": {"leakage_risk": 0.9}}],
        mitigation_mode="verify_only",
    )
    assert verify["executed"] is True
    assert verify["passed"] is False
    assert verify["high_risk_count"] == 1


def test_off_mode_preserves_legacy_behavior() -> None:
    verify = verify_symmetry_sequence(
        generator_metadata=[{"symmetry_spec": {"leakage_risk": 0.9}}],
        mitigation_mode="off",
    )
    assert verify["executed"] is False
    assert verify["passed"] is True
    assert verify["mode"] == "off"

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/test/test_adapt_vqe_integration.py
```py
"""Integration tests for the hardcoded ADAPT-VQE pipeline.

Tests cover:
  - L=2 Hubbard UCCSD pool (basic ADAPT-VQE convergence)
  - L=2 HH HVA pool (sector-filtered HH ground energy)
  - L=2 HH PAOP pool (polaron-adapted operators)
  - Pool builder sanity checks (non-empty, correct types)
  - Sector filtering correctness (HH uses fermion-only filtering)
  - PAOP module importability
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.hubbard_latex_python_pairs import (
    build_hubbard_hamiltonian,
    build_hubbard_holstein_hamiltonian,
    boson_qubits_per_site,
)
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    exact_ground_energy_sector,
    exact_ground_energy_sector_hh,
    half_filled_num_particles,
)

# Import ADAPT pipeline internals
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "hardcoded_adapt_pipeline",
    str(REPO_ROOT / "pipelines" / "hardcoded" / "adapt_pipeline.py"),
)
_adapt_mod = importlib.util.module_from_spec(_spec)
sys.modules["hardcoded_adapt_pipeline"] = _adapt_mod
_spec.loader.exec_module(_adapt_mod)

from pipelines.hardcoded.hh_continuation_types import CompileCostEstimate

_run_hardcoded_adapt_vqe = _adapt_mod._run_hardcoded_adapt_vqe
_build_uccsd_pool = _adapt_mod._build_uccsd_pool
_build_cse_pool = _adapt_mod._build_cse_pool
_build_full_hamiltonian_pool = _adapt_mod._build_full_hamiltonian_pool
_build_hva_pool = _adapt_mod._build_hva_pool
_build_paop_pool = _adapt_mod._build_paop_pool
_build_hh_termwise_augmented_pool = _adapt_mod._build_hh_termwise_augmented_pool
_build_hh_uccsd_fermion_lifted_pool = _adapt_mod._build_hh_uccsd_fermion_lifted_pool
_build_hh_pareto_lean_pool = _adapt_mod._build_hh_pareto_lean_pool
_build_hh_pareto_lean_l2_pool = _adapt_mod._build_hh_pareto_lean_l2_pool
_deduplicate_pool_terms = _adapt_mod._deduplicate_pool_terms
_exact_gs_energy_for_problem = _adapt_mod._exact_gs_energy_for_problem
_compile_polynomial_action = _adapt_mod._compile_polynomial_action
_apply_compiled_polynomial = _adapt_mod._apply_compiled_polynomial
_apply_pauli_polynomial_uncached = _adapt_mod._apply_pauli_polynomial_uncached
_commutator_gradient = _adapt_mod._commutator_gradient
_resolve_reopt_active_indices = _adapt_mod._resolve_reopt_active_indices
_make_reduced_objective = _adapt_mod._make_reduced_objective
_VALID_REOPT_POLICIES = _adapt_mod._VALID_REOPT_POLICIES


def _fermion_sector_weights(
    psi: np.ndarray,
    *,
    num_sites: int,
    ordering: str,
) -> dict[tuple[int, int], float]:
    if str(ordering) == "blocked":
        alpha = list(range(int(num_sites)))
        beta = list(range(int(num_sites), 2 * int(num_sites)))
    else:
        alpha = list(range(0, 2 * int(num_sites), 2))
        beta = list(range(1, 2 * int(num_sites), 2))
    out: dict[tuple[int, int], float] = {}
    for idx, amp in enumerate(np.asarray(psi, dtype=complex).reshape(-1)):
        prob = float(abs(amp) ** 2)
        if prob <= 1e-14:
            continue
        n_up = int(sum((idx >> int(q)) & 1 for q in alpha))
        n_dn = int(sum((idx >> int(q)) & 1 for q in beta))
        out[(n_up, n_dn)] = float(out.get((n_up, n_dn), 0.0) + prob)
    return out


class TestCompiledPauliCache:
    """Parity and performance checks for cached compiled Pauli actions."""

    @staticmethod
    def _random_state(nq: int, seed: int = 13) -> np.ndarray:
        rng = np.random.default_rng(int(seed))
        psi = rng.normal(size=1 << int(nq)) + 1j * rng.normal(size=1 << int(nq))
        psi = np.asarray(psi, dtype=complex)
        return psi / np.linalg.norm(psi)

    def test_compiled_apply_matches_uncached(self):
        h_poly = build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.3,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        psi = self._random_state(4, seed=101)
        compiled = _compile_polynomial_action(h_poly)
        uncached = _apply_pauli_polynomial_uncached(psi, h_poly)
        cached = _apply_compiled_polynomial(psi, compiled)
        assert np.max(np.abs(cached - uncached)) < 1e-12

    def test_commutator_gradient_matches_uncached(self):
        h_poly = build_hubbard_hamiltonian(
            dims=3, t=1.0, U=4.0, v=0.1,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        num_particles = half_filled_num_particles(3)
        pool = _build_uccsd_pool(3, num_particles, "blocked")
        assert len(pool) > 0
        op = pool[0]
        psi = self._random_state(6, seed=202)

        grad_uncached = _commutator_gradient(h_poly, op, psi)
        grad_cached = _commutator_gradient(
            h_poly,
            op,
            psi,
            h_compiled=_compile_polynomial_action(h_poly),
            pool_compiled=_compile_polynomial_action(op.polynomial),
        )
        assert abs(grad_cached - grad_uncached) < 1e-12

    def test_gradient_cached_speedup(self):
        h_poly = build_hubbard_hamiltonian(
            dims=3, t=1.0, U=4.0, v=0.1,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        num_particles = half_filled_num_particles(3)
        pool = _build_cse_pool(3, "blocked", 1.0, 4.0, 0.1, "periodic")
        assert len(pool) > 0
        op = pool[0]
        psi = self._random_state(6, seed=303)

        h_compiled = _compile_polynomial_action(h_poly)
        op_compiled = _compile_polynomial_action(op.polynomial)

        # Warm up to avoid one-time dispatch effects dominating timings.
        _commutator_gradient(h_poly, op, psi)
        _commutator_gradient(h_poly, op, psi, h_compiled=h_compiled, pool_compiled=op_compiled)

        def _bench_uncached(num_iter: int) -> float:
            t0 = time.perf_counter()
            for _ in range(int(num_iter)):
                _commutator_gradient(h_poly, op, psi)
            return float(time.perf_counter() - t0)

        def _bench_cached(num_iter: int) -> float:
            t0 = time.perf_counter()
            for _ in range(int(num_iter)):
                _commutator_gradient(
                    h_poly,
                    op,
                    psi,
                    h_compiled=h_compiled,
                    pool_compiled=op_compiled,
                )
            return float(time.perf_counter() - t0)

        num_iter = 8
        uncached_elapsed = _bench_uncached(num_iter)
        while uncached_elapsed < 0.15 and num_iter < 4096:
            num_iter *= 2
            uncached_elapsed = _bench_uncached(num_iter)
        cached_elapsed = _bench_cached(num_iter)
        speedup = uncached_elapsed / cached_elapsed if cached_elapsed > 0.0 else float("inf")
        assert speedup > 1.5, (
            f"Expected cached gradient speedup > 1.5x, got {speedup:.2f}x "
            f"(uncached={uncached_elapsed:.4f}s, cached={cached_elapsed:.4f}s, iters={num_iter})"
        )


class TestAdaptCompiledStateBackendParity:
    """Compiled ansatz execution must preserve ADAPT selection/energy parity."""

    def test_compiled_state_backend_matches_legacy_sequence(self):
        h_poly = build_hubbard_hamiltonian(
            dims=2,
            t=1.0,
            U=4.0,
            v=0.0,
            repr_mode="JW",
            indexing="blocked",
            pbc=True,
        )
        common_kwargs = dict(
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
            max_depth=6,
            eps_grad=1e-6,
            eps_energy=1e-10,
            maxiter=120,
            seed=17,
            allow_repeats=False,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
        )

        payload_legacy, _psi_legacy = _run_hardcoded_adapt_vqe(
            **common_kwargs,
            adapt_state_backend="legacy",
        )
        payload_compiled, _psi_compiled = _run_hardcoded_adapt_vqe(
            **common_kwargs,
            adapt_state_backend="compiled",
        )

        seq_legacy = [int(row["pool_index"]) for row in payload_legacy.get("history", [])]
        seq_compiled = [int(row["pool_index"]) for row in payload_compiled.get("history", [])]
        labels_legacy = [str(row["selected_op"]) for row in payload_legacy.get("history", [])]
        labels_compiled = [str(row["selected_op"]) for row in payload_compiled.get("history", [])]

        n_check = min(5, len(seq_legacy), len(seq_compiled))
        assert n_check > 0
        assert seq_compiled[:n_check] == seq_legacy[:n_check]
        assert labels_compiled[:n_check] == labels_legacy[:n_check]
        assert abs(float(payload_compiled["energy"]) - float(payload_legacy["energy"])) < 1e-8


# ============================================================================
# Pool builder tests
# ============================================================================

class TestAdaptCLIParsing:
    """CLI parsing includes newly supported ADAPT pool options."""

    def test_parse_accepts_uccsd_paop_lf_full_pool(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--adapt-pool", "uccsd_paop_lf_full"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_pool) == "uccsd_paop_lf_full"

    def test_parse_accepts_full_meta_pool(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--problem", "hh", "--adapt-pool", "full_meta"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_pool) == "full_meta"

    def test_parse_accepts_pareto_lean_pool(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--problem", "hh", "--adapt-pool", "pareto_lean"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_pool) == "pareto_lean"

    def test_parse_accepts_pareto_lean_l2_pool(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--problem", "hh", "--adapt-pool", "pareto_lean_l2"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_pool) == "pareto_lean_l2"

    def test_parse_accepts_adapt_state_backend_legacy(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--adapt-state-backend", "legacy"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_state_backend) == "legacy"

    def test_parse_accepts_phase1_continuation_mode(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--adapt-continuation-mode", "phase1_v1"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_continuation_mode) == "phase1_v1"

    def test_parse_accepts_phase2_continuation_mode(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--adapt-continuation-mode", "phase2_v1"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_continuation_mode) == "phase2_v1"

    def test_parse_accepts_phase3_continuation_mode(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--adapt-continuation-mode", "phase3_v1"],
        )
        args = _adapt_mod.parse_args()
        assert str(args.adapt_continuation_mode) == "phase3_v1"

    def test_parse_rejects_auto_continuation_mode(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["adapt_pipeline.py", "--adapt-continuation-mode", "auto"],
        )
        with pytest.raises(SystemExit):
            _adapt_mod.parse_args()

    def test_parse_defaults_eps_energy_gate_knobs(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sys, "argv", ["adapt_pipeline.py"])
        args = _adapt_mod.parse_args()
        assert int(args.adapt_eps_energy_min_extra_depth) == -1
        assert int(args.adapt_eps_energy_patience) == -1

    def test_parse_defaults_drop_knobs_to_auto(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sys, "argv", ["adapt_pipeline.py"])
        args = _adapt_mod.parse_args()
        assert args.adapt_drop_floor is None
        assert args.adapt_drop_patience is None
        assert args.adapt_drop_min_depth is None
        assert args.adapt_grad_floor is None

    def test_parse_accepts_eps_energy_gate_knobs(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--adapt-eps-energy-min-extra-depth", "6",
                "--adapt-eps-energy-patience", "4",
            ],
        )
        args = _adapt_mod.parse_args()
        assert int(args.adapt_eps_energy_min_extra_depth) == 6
        assert int(args.adapt_eps_energy_patience) == 4

class TestPoolBuilders:
    """Verify pool builders return non-empty pools of AnsatzTerm."""

    def test_uccsd_pool_L2(self):
        num_particles = half_filled_num_particles(2)
        pool = _build_uccsd_pool(2, num_particles, "blocked")
        assert len(pool) > 0, "UCCSD pool must be non-empty for L=2"
        for op in pool:
            assert isinstance(op, AnsatzTerm)

    def test_cse_pool_L2(self):
        pool = _build_cse_pool(2, "blocked", 1.0, 4.0, 0.0, "periodic")
        assert len(pool) > 0, "CSE pool must be non-empty for L=2"
        for op in pool:
            assert isinstance(op, AnsatzTerm)

    def test_full_hamiltonian_pool_L2(self):
        h_poly = build_hubbard_hamiltonian(dims=2, t=1.0, U=4.0, v=0.0,
                                            repr_mode="JW", indexing="blocked",
                                            pbc=True)
        pool = _build_full_hamiltonian_pool(h_poly)
        assert len(pool) > 0
        for op in pool:
            assert isinstance(op, AnsatzTerm)

    def test_hva_pool_L2_hh(self):
        pool = _build_hva_pool(
            num_sites=2, t=1.0, u=4.0, omega0=1.0, g_ep=0.5, dv=0.0,
            n_ph_max=1, boson_encoding="binary", ordering="blocked",
            boundary="periodic",
        )
        assert len(pool) > 0, "HVA pool must be non-empty for L=2 HH"
        for op in pool:
            assert isinstance(op, AnsatzTerm)
        labels = [str(op.label) for op in pool]
        lifted_pool = _build_hh_uccsd_fermion_lifted_pool(
            num_sites=2,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="periodic",
            num_particles=half_filled_num_particles(2),
        )
        lifted_labels = {str(op.label) for op in lifted_pool}
        assert lifted_labels.issubset(set(labels))
        assert any(label.startswith("uccsd_ferm_lifted::") for label in labels)
        assert not any(
            label.startswith("uccsd_sing(") or label.startswith("uccsd_dbl(")
            for label in labels
        )

    def test_hh_termwise_augmented_pool_L2(self):
        h_poly = build_hubbard_holstein_hamiltonian(
            dims=2, J=1.0, U=4.0, omega0=1.0, g=0.5,
            n_ph_max=1, boson_encoding="binary",
            repr_mode="JW", indexing="blocked", pbc=True,
            include_zero_point=True,
        )
        pool = _build_hh_termwise_augmented_pool(h_poly)
        assert len(pool) > 0
        # Must contain at least some quadrature partners
        quad_ops = [op for op in pool if "quadrature" in op.label]
        assert len(quad_ops) > 0, "HH termwise augmented pool should have quadrature partners"


class TestPAOPPoolBuilder:
    """Verify PAOP pool builder returns non-empty pools."""

    def test_paop_min_L2(self):
        num_particles = half_filled_num_particles(2)
        pool = _build_paop_pool(
            num_sites=2, n_ph_max=1, boson_encoding="binary",
            ordering="blocked", boundary="periodic",
            pool_key="paop_min", paop_r=1,
            paop_split_paulis=False, paop_prune_eps=0.0,
            paop_normalization="none", num_particles=num_particles,
        )
        assert len(pool) > 0, "paop_min must produce operators for L=2"

    def test_paop_std_L2(self):
        num_particles = half_filled_num_particles(2)
        pool = _build_paop_pool(
            num_sites=2, n_ph_max=1, boson_encoding="binary",
            ordering="blocked", boundary="periodic",
            pool_key="paop_std", paop_r=1,
            paop_split_paulis=False, paop_prune_eps=0.0,
            paop_normalization="none", num_particles=num_particles,
        )
        assert len(pool) > 0
        # paop_std includes hopdrag so should be larger than paop_min
        pool_min = _build_paop_pool(
            num_sites=2, n_ph_max=1, boson_encoding="binary",
            ordering="blocked", boundary="periodic",
            pool_key="paop_min", paop_r=1,
            paop_split_paulis=False, paop_prune_eps=0.0,
            paop_normalization="none", num_particles=num_particles,
        )
        assert len(pool) >= len(pool_min)

    def test_paop_full_L2(self):
        num_particles = half_filled_num_particles(2)
        pool = _build_paop_pool(
            num_sites=2, n_ph_max=1, boson_encoding="binary",
            ordering="blocked", boundary="periodic",
            pool_key="paop_full", paop_r=1,
            paop_split_paulis=False, paop_prune_eps=0.0,
            paop_normalization="none", num_particles=num_particles,
        )
        assert len(pool) > 0

    def test_paop_lf_std_L2(self):
        num_particles = half_filled_num_particles(2)
        pool_lf = _build_paop_pool(
            num_sites=2, n_ph_max=1, boson_encoding="binary",
            ordering="blocked", boundary="periodic",
            pool_key="paop_lf_std", paop_r=1,
            paop_split_paulis=False, paop_prune_eps=0.0,
            paop_normalization="none", num_particles=num_particles,
        )
        pool_std = _build_paop_pool(
            num_sites=2, n_ph_max=1, boson_encoding="binary",
            ordering="blocked", boundary="periodic",
            pool_key="paop_std", paop_r=1,
            paop_split_paulis=False, paop_prune_eps=0.0,
            paop_normalization="none", num_particles=num_particles,
        )
        assert len(pool_lf) >= len(pool_std)

    def test_paop_lf2_std_L2(self):
        num_particles = half_filled_num_particles(2)
        pool_lf = _build_paop_pool(
            num_sites=2, n_ph_max=1, boson_encoding="binary",
            ordering="blocked", boundary="periodic",
            pool_key="paop_lf_std", paop_r=1,
            paop_split_paulis=False, paop_prune_eps=0.0,
            paop_normalization="none", num_particles=num_particles,
        )
        pool_lf2 = _build_paop_pool(
            num_sites=2, n_ph_max=1, boson_encoding="binary",
            ordering="blocked", boundary="periodic",
            pool_key="paop_lf2_std", paop_r=1,
            paop_split_paulis=False, paop_prune_eps=0.0,
            paop_normalization="none", num_particles=num_particles,
        )
        assert len(pool_lf2) >= len(pool_lf)

    def test_paop_lf_full_L2(self):
        num_particles = half_filled_num_particles(2)
        pool = _build_paop_pool(
            num_sites=2, n_ph_max=1, boson_encoding="binary",
            ordering="blocked", boundary="periodic",
            pool_key="paop_lf_full", paop_r=1,
            paop_split_paulis=False, paop_prune_eps=0.0,
            paop_normalization="none", num_particles=num_particles,
        )
        assert len(pool) > 0

    def test_paop_lf_alias_matches_lf_std(self):
        num_particles = half_filled_num_particles(2)
        pool_alias = _build_paop_pool(
            num_sites=2, n_ph_max=1, boson_encoding="binary",
            ordering="blocked", boundary="periodic",
            pool_key="paop_lf", paop_r=1,
            paop_split_paulis=False, paop_prune_eps=0.0,
            paop_normalization="none", num_particles=num_particles,
        )
        pool_std = _build_paop_pool(
            num_sites=2, n_ph_max=1, boson_encoding="binary",
            ordering="blocked", boundary="periodic",
            pool_key="paop_lf_std", paop_r=1,
            paop_split_paulis=False, paop_prune_eps=0.0,
            paop_normalization="none", num_particles=num_particles,
        )
        assert len(pool_alias) == len(pool_std)

    def test_paop_curdrag_L2_open_blocked_signature(self):
        num_particles = half_filled_num_particles(2)
        pool = _build_paop_pool(
            num_sites=2, n_ph_max=1, boson_encoding="binary",
            ordering="blocked", boundary="open",
            pool_key="paop_lf_std", paop_r=1,
            paop_split_paulis=False, paop_prune_eps=0.0,
            paop_normalization="none", num_particles=num_particles,
        )
        curdrag = None
        for op in pool:
            if "paop_curdrag(0,1)" in op.label:
                curdrag = op
                break
        assert curdrag is not None, "Expected paop_curdrag(0,1) in paop_lf_std for L=2 open chain."

        coeff_map: dict[str, float] = {}
        for term in curdrag.polynomial.return_polynomial():
            coeff = complex(term.p_coeff)
            if abs(coeff) <= 1e-12:
                continue
            assert abs(coeff.imag) <= 1e-10
            coeff_map[str(term.pw2strng())] = float(round(coeff.real, 12))

        expected = {
            "eyeexy": 0.5,
            "eyeeyx": -0.5,
            "eyxyee": 0.5,
            "eyyxee": -0.5,
            "yeeexy": -0.5,
            "yeeeyx": 0.5,
            "yexyee": -0.5,
            "yeyxee": 0.5,
        }
        assert set(coeff_map.keys()) == set(expected.keys())
        same_sign = all(abs(coeff_map[key] - expected[key]) <= 1e-10 for key in expected)
        flipped_sign = all(abs(coeff_map[key] + expected[key]) <= 1e-10 for key in expected)
        assert same_sign or flipped_sign

    def test_paop_lf_coefficients_are_real_after_cleaning(self):
        num_particles = half_filled_num_particles(2)
        for pool_key in ("paop_lf_std", "paop_lf2_std", "paop_lf_full"):
            pool = _build_paop_pool(
                num_sites=2, n_ph_max=1, boson_encoding="binary",
                ordering="blocked", boundary="periodic",
                pool_key=pool_key, paop_r=1,
                paop_split_paulis=False, paop_prune_eps=0.0,
                paop_normalization="none", num_particles=num_particles,
            )
            assert len(pool) > 0
            for op in pool:
                for term in op.polynomial.return_polynomial():
                    assert abs(complex(term.p_coeff).imag) <= 1e-10

    def test_paop_module_importable(self):
        """Verify the operator_pools module can be imported directly."""
        from src.quantum.operator_pools import make_pool
        assert callable(make_pool)


class TestHHUCCSDPAOPCompositePoolBuilder:
    """Verify HH composite UCCSD+PAOP(lf_full) pool semantics."""

    def test_uccsd_lift_has_boson_identity_prefix(self):
        n_sites = 2
        n_ph_max = 1
        boson_encoding = "binary"
        boson_bits = n_sites * int(boson_qubits_per_site(n_ph_max, boson_encoding))
        pool = _build_hh_uccsd_fermion_lifted_pool(
            num_sites=n_sites,
            n_ph_max=n_ph_max,
            boson_encoding=boson_encoding,
            ordering="blocked",
            boundary="periodic",
            num_particles=half_filled_num_particles(n_sites),
        )
        assert len(pool) > 0
        boson_identity = "e" * boson_bits
        nq_total = 2 * n_sites + boson_bits
        for op in pool:
            has_nontrivial_fermion_support = False
            for term in op.polynomial.return_polynomial():
                coeff = complex(term.p_coeff)
                if abs(coeff) <= 1e-15:
                    continue
                ps = str(term.pw2strng())
                assert len(ps) == nq_total
                assert ps[:boson_bits] == boson_identity
                if any(ch != "e" for ch in ps[boson_bits:]):
                    has_nontrivial_fermion_support = True
            assert has_nontrivial_fermion_support

    def test_composite_pool_is_non_empty_and_deduplicated(self):
        n_sites = 2
        num_particles = half_filled_num_particles(n_sites)
        uccsd_pool = _build_hh_uccsd_fermion_lifted_pool(
            num_sites=n_sites,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="periodic",
            num_particles=num_particles,
        )
        paop_pool = _build_paop_pool(
            num_sites=n_sites,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="periodic",
            pool_key="paop_lf_full",
            paop_r=1,
            paop_split_paulis=False,
            paop_prune_eps=0.0,
            paop_normalization="none",
            num_particles=num_particles,
        )
        dedup_pool = _deduplicate_pool_terms(list(uccsd_pool) + list(paop_pool))
        assert len(uccsd_pool) > 0
        assert len(paop_pool) > 0
        assert len(dedup_pool) > 0
        assert len(dedup_pool) <= len(uccsd_pool) + len(paop_pool)

    def test_pareto_lean_pool_keeps_only_scaffold_supported_families(self):
        n_sites = 2
        num_particles = half_filled_num_particles(n_sites)
        h_poly = build_hubbard_holstein_hamiltonian(
            dims=n_sites,
            J=1.0,
            U=4.0,
            omega0=1.0,
            g=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            repr_mode="JW",
            indexing="blocked",
            pbc=True,
            include_zero_point=True,
        )
        pool, meta = _build_hh_pareto_lean_pool(
            h_poly=h_poly,
            num_sites=n_sites,
            t=1.0,
            u=4.0,
            omega0=1.0,
            g_ep=0.5,
            dv=0.0,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="periodic",
            paop_r=1,
            paop_split_paulis=False,
            paop_prune_eps=0.0,
            paop_normalization="none",
            num_particles=num_particles,
        )
        labels = [str(op.label) for op in pool]

        assert len(pool) > 0
        assert int(meta["raw_total"]) > 0
        assert any(label.startswith("uccsd_ferm_lifted::uccsd_sing(") for label in labels)
        assert any(label.startswith("uccsd_ferm_lifted::uccsd_dbl(") for label in labels)
        assert any(label.startswith("hh_termwise_ham_quadrature_term(") for label in labels)
        assert any(label.startswith("paop_full:paop_cloud_p(") for label in labels)
        assert any(label.startswith("paop_full:paop_disp(") for label in labels)
        assert any(label.startswith("paop_full:paop_hopdrag(") for label in labels)
        assert any(label.startswith("paop_lf_full:paop_dbl_p(") for label in labels)

        assert not any(label in {"hop_layer", "onsite_layer", "phonon_layer", "eph_layer"} for label in labels)
        assert not any(label.startswith("hh_termwise_ham_unit_term(") for label in labels)
        assert not any(label.startswith("paop_full:paop_dbl(") for label in labels)
        assert not any(label.startswith("paop_full:paop_cloud_x(") for label in labels)
        assert not any(label.startswith("paop_lf_full:paop_dbl_x(") for label in labels)
        assert not any(label.startswith("paop_lf_full:paop_curdrag(") for label in labels)
        assert not any(label.startswith("paop_lf_full:paop_hop2(") for label in labels)

    def test_pareto_lean_l2_pool_is_nonempty_for_l2_nph1(self):
        n_sites = 2
        num_particles = half_filled_num_particles(n_sites)
        h_poly = build_hubbard_holstein_hamiltonian(
            dims=n_sites,
            J=1.0,
            U=4.0,
            omega0=1.0,
            g=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            repr_mode="JW",
            indexing="blocked",
            pbc=False,
            include_zero_point=True,
        )
        pool, meta = _build_hh_pareto_lean_l2_pool(
            h_poly=h_poly,
            num_sites=n_sites,
            t=1.0,
            u=4.0,
            omega0=1.0,
            g_ep=0.5,
            dv=0.0,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            paop_r=1,
            paop_split_paulis=False,
            paop_prune_eps=0.0,
            paop_normalization="none",
            num_particles=num_particles,
        )
        assert len(pool) > 0
        assert int(meta["raw_total"]) > 0

    def test_pareto_lean_l2_pool_rejects_non_l2(self):
        h_poly = build_hubbard_holstein_hamiltonian(
            dims=3,
            J=1.0,
            U=4.0,
            omega0=1.0,
            g=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            repr_mode="JW",
            indexing="blocked",
            pbc=False,
            include_zero_point=True,
        )
        with pytest.raises(ValueError, match="only valid for L=2"):
            _build_hh_pareto_lean_l2_pool(
                h_poly=h_poly,
                num_sites=3,
                t=1.0,
                u=4.0,
                omega0=1.0,
                g_ep=0.5,
                dv=0.0,
                n_ph_max=1,
                boson_encoding="binary",
                ordering="blocked",
                boundary="open",
                paop_r=1,
                paop_split_paulis=False,
                paop_prune_eps=0.0,
                paop_normalization="none",
                num_particles=half_filled_num_particles(3),
            )

    def test_pareto_lean_l2_pool_rejects_nphmax_not_1(self):
        h_poly = build_hubbard_holstein_hamiltonian(
            dims=2,
            J=1.0,
            U=4.0,
            omega0=1.0,
            g=0.5,
            n_ph_max=2,
            boson_encoding="binary",
            repr_mode="JW",
            indexing="blocked",
            pbc=False,
            include_zero_point=True,
        )
        with pytest.raises(ValueError, match="only valid for n_ph_max=1"):
            _build_hh_pareto_lean_l2_pool(
                h_poly=h_poly,
                num_sites=2,
                t=1.0,
                u=4.0,
                omega0=1.0,
                g_ep=0.5,
                dv=0.0,
                n_ph_max=2,
                boson_encoding="binary",
                ordering="blocked",
                boundary="open",
                paop_r=1,
                paop_split_paulis=False,
                paop_prune_eps=0.0,
                paop_normalization="none",
                num_particles=half_filled_num_particles(2),
            )


# ============================================================================
# Sector filtering dispatch
# ============================================================================

class TestSectorFilteringDispatch:
    """Verify _exact_gs_energy_for_problem dispatches correctly."""

    def test_hubbard_dispatch(self):
        h_poly = build_hubbard_hamiltonian(dims=2, t=1.0, U=4.0, v=0.0,
                                            repr_mode="JW", indexing="blocked", pbc=True)
        num_particles = half_filled_num_particles(2)
        e_dispatch = _exact_gs_energy_for_problem(
            h_poly, problem="hubbard", num_sites=2,
            num_particles=num_particles, indexing="blocked",
        )
        e_direct = exact_ground_energy_sector(
            h_poly, num_sites=2, num_particles=num_particles, indexing="blocked",
        )
        assert abs(e_dispatch - e_direct) < 1e-12

    def test_hh_dispatch_uses_fermion_only(self):
        """HH dispatch must use exact_ground_energy_sector_hh (fermion-only filtering)."""
        h_poly = build_hubbard_holstein_hamiltonian(
            dims=2, J=1.0, U=4.0, omega0=1.0, g=0.5,
            n_ph_max=1, boson_encoding="binary",
            repr_mode="JW", indexing="blocked", pbc=True,
            include_zero_point=True,
        )
        num_particles = half_filled_num_particles(2)
        e_dispatch = _exact_gs_energy_for_problem(
            h_poly, problem="hh", num_sites=2,
            num_particles=num_particles, indexing="blocked",
            n_ph_max=1, boson_encoding="binary",
        )
        e_direct = exact_ground_energy_sector_hh(
            h_poly, num_sites=2, num_particles=num_particles,
            n_ph_max=1, boson_encoding="binary", indexing="blocked",
        )
        assert abs(e_dispatch - e_direct) < 1e-12


# ============================================================================
# End-to-end ADAPT-VQE smoke tests
# ============================================================================

class TestAdaptVQEHubbardUCCSD:
    """L=2 Hubbard UCCSD ADAPT-VQE must converge to near-exact energy."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.L = 2
        self.t = 1.0
        self.u = 4.0
        self.h_poly = build_hubbard_hamiltonian(
            dims=self.L, t=self.t, U=self.u, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        self.num_particles = half_filled_num_particles(self.L)
        self.exact_gs = exact_ground_energy_sector(
            self.h_poly, num_sites=self.L,
            num_particles=self.num_particles, indexing="blocked",
        )

    def test_adapt_uccsd_converges(self):
        payload, psi = _run_hardcoded_adapt_vqe(
            h_poly=self.h_poly,
            num_sites=self.L,
            ordering="blocked",
            problem="hubbard",
            adapt_pool="uccsd",
            t=self.t, u=self.u, dv=0.0,
            boundary="periodic",
            omega0=0.0, g_ep=0.0,
            n_ph_max=1, boson_encoding="binary",
            max_depth=15,
            eps_grad=1e-6,
            eps_energy=1e-10,
            maxiter=300,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
        )
        assert payload["success"] is True
        assert payload["energy"] is not None
        # UCCSD pool for L=2 half-filling is small (3 ops: 2 singles + 1 double).
        # The ADAPT greedy loop may not select the double (zero gradient at HF),
        # so the energy may not reach the exact GS. Verify it at least improves
        # significantly from the HF energy and returns a physically valid result.
        hf_energy = 4.0  # known for L=2 periodic t=1 U=4 half-filled
        assert payload["energy"] < hf_energy - 1.0, \
            f"ADAPT UCCSD must improve on HF: E={payload['energy']:.4f} vs HF={hf_energy}"
        assert payload["exact_gs_energy"] is not None

    def test_adapt_cse_converges(self):
        payload, psi = _run_hardcoded_adapt_vqe(
            h_poly=self.h_poly,
            num_sites=self.L,
            ordering="blocked",
            problem="hubbard",
            adapt_pool="cse",
            t=self.t, u=self.u, dv=0.0,
            boundary="periodic",
            omega0=0.0, g_ep=0.0,
            n_ph_max=1, boson_encoding="binary",
            max_depth=15,
            eps_grad=1e-6,
            eps_energy=1e-10,
            maxiter=300,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
        )
        assert payload["success"] is True
        # CSE pool for L=2 has only 4 Hamiltonian-term generators (hopping + onsite).
        # With such a small pool ADAPT may not reach exact GS, but should improve on HF.
        hf_energy = 4.0
        assert payload["energy"] < hf_energy - 1.0, \
            f"ADAPT CSE must improve on HF: E={payload['energy']:.4f} vs HF={hf_energy}"

    def test_adapt_full_hamiltonian_converges(self):
        """full_hamiltonian pool should converge well for L=2."""
        payload, psi = _run_hardcoded_adapt_vqe(
            h_poly=self.h_poly,
            num_sites=self.L,
            ordering="blocked",
            problem="hubbard",
            adapt_pool="full_hamiltonian",
            t=self.t, u=self.u, dv=0.0,
            boundary="periodic",
            omega0=0.0, g_ep=0.0,
            n_ph_max=1, boson_encoding="binary",
            max_depth=20,
            eps_grad=1e-6,
            eps_energy=1e-10,
            maxiter=300,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
        )
        assert payload["success"] is True
        # full_hamiltonian pool for L=2 periodic Hubbard: ADAPT can get trapped
        # at E≈0 (a degenerate eigenvalue) because the greedy gradient selection
        # cannot escape this local minimum with only 10 Hamiltonian-term generators.
        # Verify significant improvement over HF reference energy.
        hf_energy = 4.0
        assert payload["energy"] < hf_energy - 1.0, \
            f"ADAPT full_hamiltonian must improve on HF: E={payload['energy']:.4f} vs HF={hf_energy}"


class TestAdaptVQEHolsteinHVA:
    """L=2 HH HVA ADAPT-VQE must converge to near-exact HH energy."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.L = 2
        self.t = 1.0
        self.u = 4.0
        self.omega0 = 1.0
        self.g_ep = 0.5
        self.n_ph_max = 1
        self.h_poly = build_hubbard_holstein_hamiltonian(
            dims=self.L, J=self.t, U=self.u,
            omega0=self.omega0, g=self.g_ep,
            n_ph_max=self.n_ph_max, boson_encoding="binary",
            repr_mode="JW", indexing="blocked", pbc=True,
            include_zero_point=True,
        )
        self.num_particles = half_filled_num_particles(self.L)
        self.exact_gs = exact_ground_energy_sector_hh(
            self.h_poly, num_sites=self.L,
            num_particles=self.num_particles,
            n_ph_max=self.n_ph_max, boson_encoding="binary",
            indexing="blocked",
        )

    def test_adapt_hva_hh_preserves_sector_and_is_variational(self):
        payload, psi = _run_hardcoded_adapt_vqe(
            h_poly=self.h_poly,
            num_sites=self.L,
            ordering="blocked",
            problem="hh",
            adapt_pool="hva",
            t=self.t, u=self.u, dv=0.0,
            boundary="periodic",
            omega0=self.omega0, g_ep=self.g_ep,
            n_ph_max=self.n_ph_max, boson_encoding="binary",
            max_depth=30,
            eps_grad=1e-5,
            eps_energy=1e-10,
            maxiter=600,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="full",  # convergence test — needs full re-opt
        )
        assert payload["success"] is True
        assert payload["energy"] is not None
        # HH exact_gs in payload should match our computed value
        assert abs(payload["exact_gs_energy"] - self.exact_gs) < 1e-10
        sector_weights = _fermion_sector_weights(psi, num_sites=self.L, ordering="blocked")
        assert sector_weights.get(tuple(self.num_particles), 0.0) > 1.0 - 1e-10
        assert payload["energy"] >= self.exact_gs - 1e-10

    def test_adapt_hh_uses_fermion_only_sector(self):
        """Verify the payload exact_gs matches fermion-only sector filtering."""
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self.h_poly,
            num_sites=self.L,
            ordering="blocked",
            problem="hh",
            adapt_pool="full_hamiltonian",
            t=self.t, u=self.u, dv=0.0,
            boundary="periodic",
            omega0=self.omega0, g_ep=self.g_ep,
            n_ph_max=self.n_ph_max, boson_encoding="binary",
            max_depth=5,
            eps_grad=1e-2,
            eps_energy=1e-6,
            maxiter=100,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
        )
        assert abs(payload["exact_gs_energy"] - self.exact_gs) < 1e-10, \
            "HH ADAPT must use fermion-only sector filtering"


class TestAdaptVQEHolsteinPAOP:
    """L=2 HH PAOP ADAPT-VQE smoke test."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.L = 2
        self.t = 1.0
        self.u = 4.0
        self.omega0 = 1.0
        self.g_ep = 0.5
        self.n_ph_max = 1
        self.h_poly = build_hubbard_holstein_hamiltonian(
            dims=self.L, J=self.t, U=self.u,
            omega0=self.omega0, g=self.g_ep,
            n_ph_max=self.n_ph_max, boson_encoding="binary",
            repr_mode="JW", indexing="blocked", pbc=True,
            include_zero_point=True,
        )
        self.num_particles = half_filled_num_particles(self.L)
        self.exact_gs = exact_ground_energy_sector_hh(
            self.h_poly, num_sites=self.L,
            num_particles=self.num_particles,
            n_ph_max=self.n_ph_max, boson_encoding="binary",
            indexing="blocked",
        )

    def test_adapt_paop_std_runs(self):
        """PAOP std pool should run without error and produce a valid energy."""
        payload, psi = _run_hardcoded_adapt_vqe(
            h_poly=self.h_poly,
            num_sites=self.L,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_std",
            t=self.t, u=self.u, dv=0.0,
            boundary="periodic",
            omega0=self.omega0, g_ep=self.g_ep,
            n_ph_max=self.n_ph_max, boson_encoding="binary",
            max_depth=15,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=300,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            paop_r=1,
            paop_split_paulis=False,
            paop_prune_eps=0.0,
            paop_normalization="none",
            adapt_reopt_policy="full",  # convergence test — needs full re-opt
            adapt_continuation_mode="legacy",
        )
        assert payload["success"] is True
        assert str(payload["pool_type"]) == "paop_std"
        assert str(payload["method"]) == "hardcoded_adapt_vqe_paop_std"
        assert payload["energy"] is not None
        # Energy should be finite and not NaN
        assert np.isfinite(payload["energy"])
        # Should be lower than reference state energy (some improvement)
        assert payload["energy"] <= payload["exact_gs_energy"] + 0.5

    def test_adapt_paop_min_runs(self):
        """PAOP min pool (displacement only) should run."""
        payload, psi = _run_hardcoded_adapt_vqe(
            h_poly=self.h_poly,
            num_sites=self.L,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_min",
            t=self.t, u=self.u, dv=0.0,
            boundary="periodic",
            omega0=self.omega0, g_ep=self.g_ep,
            n_ph_max=self.n_ph_max, boson_encoding="binary",
            max_depth=10,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=200,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            paop_r=1,
            paop_split_paulis=False,
            paop_prune_eps=0.0,
            paop_normalization="none",
            adapt_continuation_mode="legacy",
        )
        assert payload["success"] is True
        assert np.isfinite(payload["energy"])

    def test_adapt_uccsd_paop_lf_full_runs(self):
        """Composite HH pool should run and report composite pool_type."""
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self.h_poly,
            num_sites=self.L,
            ordering="blocked",
            problem="hh",
            adapt_pool="uccsd_paop_lf_full",
            t=self.t,
            u=self.u,
            dv=0.0,
            boundary="periodic",
            omega0=self.omega0,
            g_ep=self.g_ep,
            n_ph_max=self.n_ph_max,
            boson_encoding="binary",
            max_depth=6,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=200,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            paop_r=1,
            paop_split_paulis=False,
            paop_prune_eps=0.0,
            paop_normalization="none",
            adapt_continuation_mode="legacy",
        )
        assert payload["success"] is True
        assert str(payload["pool_type"]) == "uccsd_paop_lf_full"
        assert int(payload["pool_size"]) > 0

    def test_adapt_full_meta_runs(self):
        """Full HH meta-pool should run and report full_meta pool type."""
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self.h_poly,
            num_sites=self.L,
            ordering="blocked",
            problem="hh",
            adapt_pool="full_meta",
            t=self.t,
            u=self.u,
            dv=0.0,
            boundary="periodic",
            omega0=self.omega0,
            g_ep=self.g_ep,
            n_ph_max=self.n_ph_max,
            boson_encoding="binary",
            max_depth=4,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=120,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            paop_r=1,
            paop_split_paulis=False,
            paop_prune_eps=0.0,
            paop_normalization="none",
            adapt_continuation_mode="legacy",
        )
        assert payload["success"] is True
        assert str(payload["pool_type"]) == "full_meta"
        assert int(payload["pool_size"]) > 0

    def test_adapt_pareto_lean_runs(self):
        """Pareto-lean HH pool should run and report pareto_lean pool type."""
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self.h_poly,
            num_sites=self.L,
            ordering="blocked",
            problem="hh",
            adapt_pool="pareto_lean",
            t=self.t,
            u=self.u,
            dv=0.0,
            boundary="periodic",
            omega0=self.omega0,
            g_ep=self.g_ep,
            n_ph_max=self.n_ph_max,
            boson_encoding="binary",
            max_depth=4,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=120,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=True,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            paop_r=1,
            paop_split_paulis=False,
            paop_prune_eps=0.0,
            paop_normalization="none",
            adapt_continuation_mode="legacy",
        )
        assert payload["success"] is True
        assert str(payload["pool_type"]) == "pareto_lean"
        assert int(payload["pool_size"]) > 0


class TestAdaptSPSAHeartbeats:
    """SPSA inner optimizer should emit progress heartbeats for ADAPT."""

    def test_spsa_heartbeat_event_is_emitted(self, monkeypatch: pytest.MonkeyPatch):
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
                max_depth=2,
                eps_grad=1e-6,
                eps_energy=1e-10,
                maxiter=40,
                seed=11,
                adapt_inner_optimizer="SPSA",
                adapt_spsa_callback_every=1,
                adapt_spsa_progress_every_s=0.0,
                allow_repeats=False,
                finite_angle_fallback=True,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
            )
            assert payload["success"] is True
            heartbeat_events = [ev for ev in events if ev[0] == "hardcoded_adapt_spsa_heartbeat"]
            assert len(heartbeat_events) > 0
            assert any(str(ev[1].get("stage", "")).startswith("depth_") for ev in heartbeat_events)
        finally:
            monkeypatch.setattr(_adapt_mod, "_ai_log", original_ai_log)


class TestAdaptDepthRollbackGuard:
    """ADAPT must not accept a depth that regresses energy.

    Regression test: before the rollback guard, the ADAPT loop would
    unconditionally accept the optimizer result.  If SPSA (or COBYLA)
    returned an energy worse than entry, the regression was permanently
    committed.  Now iter_done should never show positive delta_e.
    """

    def test_spsa_depth_never_regresses_energy(self, monkeypatch: pytest.MonkeyPatch):
        """Every iter_done event must have delta_e <= 0 (or depth_rollback=True with delta_e==0)."""
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
                eps_grad=1e-6,
                eps_energy=1e-10,
                maxiter=40,
                seed=11,
                adapt_inner_optimizer="SPSA",
                adapt_spsa_callback_every=10,
                adapt_spsa_progress_every_s=999.0,
                allow_repeats=True,
                finite_angle_fallback=True,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
            )
            iter_done_events = [
                ev for ev in events if ev[0] == "hardcoded_adapt_iter_done"
            ]
            assert len(iter_done_events) > 0, "No iter_done events emitted"
            for ev_name, ev_fields in iter_done_events:
                delta_e = float(ev_fields["delta_e"])
                # After rollback guard: accepted delta_e must be <= 0.
                # Rolled-back depths have delta_e == 0.0 exactly.
                assert delta_e <= 0.0 + 1e-14, (
                    f"depth {ev_fields.get('depth')} accepted a regression: "
                    f"delta_e={delta_e}"
                )
        finally:
            monkeypatch.setattr(_adapt_mod, "_ai_log", original_ai_log)

    def test_rollback_event_is_logged(self, monkeypatch: pytest.MonkeyPatch):
        """If rollback fires, the hardcoded_adapt_depth_rollback event must be emitted."""
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
            _run_hardcoded_adapt_vqe(
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
                eps_grad=1e-6,
                eps_energy=1e-10,
                maxiter=40,
                seed=11,
                adapt_inner_optimizer="SPSA",
                adapt_spsa_callback_every=10,
                adapt_spsa_progress_every_s=999.0,
                allow_repeats=True,
                finite_angle_fallback=True,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
            )
            rollback_events = [
                ev for ev in events if ev[0] == "hardcoded_adapt_depth_rollback"
            ]
            iter_done_events = [
                ev for ev in events if ev[0] == "hardcoded_adapt_iter_done"
            ]
            # Verify that any iter_done with depth_rollback=True has a
            # corresponding rollback log event
            rollback_depths_from_iter = {
                int(ev[1]["depth"])
                for ev in iter_done_events
                if ev[1].get("depth_rollback") is True
            }
            rollback_depths_from_event = {
                int(ev[1]["depth"])
                for ev in rollback_events
            }
            assert rollback_depths_from_iter == rollback_depths_from_event, (
                f"Mismatch: iter_done rollback depths={rollback_depths_from_iter} "
                f"vs rollback events={rollback_depths_from_event}"
            )
        finally:
            monkeypatch.setattr(_adapt_mod, "_ai_log", original_ai_log)


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
    """append_only policy must freeze the theta prefix and only optimize the newest param."""

    def test_prefix_preserved_across_depths(self, monkeypatch: pytest.MonkeyPatch):
        """After depth k, theta[:k] must be identical before and after depth k+1 optimization."""
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
                max_depth=3,
                eps_grad=1e-6,
                eps_energy=1e-10,
                maxiter=40,
                seed=11,
                adapt_inner_optimizer="COBYLA",
                allow_repeats=True,
                finite_angle_fallback=True,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_reopt_policy="append_only",
            )
            assert payload["success"] is True
            assert int(payload["ansatz_depth"]) >= 2, "Need at least 2 depths to check prefix"
            assert str(payload.get("adapt_reopt_policy", "")) == "append_only"

            # Extract the optimal_point (full theta) from payload.
            # History rows record depth-by-depth results.
            history = payload.get("history", [])
            assert len(history) >= 2

            # For append_only: at each depth k (0-indexed), the prefix
            # theta[:k] must be exactly what it was after depth k-1.
            # We verify this by checking that optimal_point[:k] from
            # depth k's row matches optimal_point[:k] constructed from
            # previous depths.
            #
            # Since the payload only gives us the final optimal_point,
            # we verify via the invariant: after the run, each history
            # row's "energy_before_opt" and "energy_after_opt" are
            # computed consistently with frozen prefixes.
            # More directly: re-run with full policy and confirm the
            # prefix DOES change there (see full_legacy test below).
            final_theta = np.array(payload["optimal_point"], dtype=float)
            logical_theta = np.array(payload["logical_optimal_point"], dtype=float)
            depth = int(payload["ansatz_depth"])
            assert int(payload["logical_num_parameters"]) == depth
            assert logical_theta.size == depth
            assert final_theta.size >= depth
            assert int(payload["num_parameters"]) == int(final_theta.size)
            assert payload.get("parameterization", {}).get("mode") == "per_pauli_term_v1"
        finally:
            monkeypatch.setattr(_adapt_mod, "_ai_log", original_ai_log)

    def test_append_only_vs_full_prefix_differs(self, monkeypatch: pytest.MonkeyPatch):
        """Running append_only vs full should produce different prefix values,
        proving append_only actually freezes and full actually changes them."""
        h_poly = build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )

        def _run_with_policy(policy: str) -> dict:
            original_ai_log = _adapt_mod._ai_log
            monkeypatch.setattr(_adapt_mod, "_ai_log", lambda event, **kw: None)
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
                    max_depth=3,
                    eps_grad=1e-6,
                    eps_energy=1e-10,
                    maxiter=80,
                    seed=7,
                    adapt_inner_optimizer="COBYLA",
                    allow_repeats=True,
                    finite_angle_fallback=True,
                    finite_angle=0.1,
                    finite_angle_min_improvement=1e-12,
                    adapt_reopt_policy=policy,
                )
                return payload
            finally:
                monkeypatch.setattr(_adapt_mod, "_ai_log", original_ai_log)

        payload_ao = _run_with_policy("append_only")
        payload_full = _run_with_policy("full")

        assert payload_ao["success"] is True
        assert payload_full["success"] is True

        theta_ao = np.array(payload_ao["optimal_point"], dtype=float)
        theta_full = np.array(payload_full["optimal_point"], dtype=float)

        # Both should produce valid results
        assert theta_ao.size >= 2
        assert theta_full.size >= 2

        # If both have at least 2 params, the first param should differ
        # (full re-optimizes it, append_only doesn't)
        min_len = min(theta_ao.size, theta_full.size)
        if min_len >= 2:
            # At least one prefix entry should differ between policies
            prefix_ao = theta_ao[:min_len - 1]
            prefix_full = theta_full[:min_len - 1]
            # They won't be exactly equal if full actually changes the prefix
            assert not np.allclose(prefix_ao, prefix_full, atol=1e-14), (
                "append_only and full produced identical prefix — "
                "policy difference is not effective"
            )


class TestAdaptReoptPolicyFull:
    """Full (legacy) re-optimization policy must allow all parameters to change."""

    def test_full_policy_allows_prefix_change(self, monkeypatch: pytest.MonkeyPatch):
        """With full policy, theta[:k] can change after appending depth k+1."""
        h_poly = build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )

        original_ai_log = _adapt_mod._ai_log
        monkeypatch.setattr(_adapt_mod, "_ai_log", lambda event, **kw: None)
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
                max_depth=3,
                eps_grad=1e-6,
                eps_energy=1e-10,
                maxiter=80,
                seed=7,
                adapt_inner_optimizer="COBYLA",
                allow_repeats=True,
                finite_angle_fallback=True,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_reopt_policy="full",
            )
            assert payload["success"] is True
            assert str(payload.get("adapt_reopt_policy", "")) == "full"
            assert int(payload["ansatz_depth"]) >= 2
        finally:
            monkeypatch.setattr(_adapt_mod, "_ai_log", original_ai_log)

    def test_invalid_policy_raises(self):
        """Invalid reopt policy must raise ValueError."""
        h_poly = build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        with pytest.raises(ValueError, match="adapt_reopt_policy"):
            _run_hardcoded_adapt_vqe(
                h_poly=h_poly,
                num_sites=2,
                ordering="blocked",
                problem="hubbard",
                adapt_pool="uccsd",
                t=1.0, u=4.0, dv=0.0,
                boundary="periodic",
                omega0=0.0, g_ep=0.0,
                n_ph_max=1, boson_encoding="binary",
                max_depth=3, eps_grad=1e-6, eps_energy=1e-10,
                maxiter=40, seed=7,
                allow_repeats=True,
                finite_angle_fallback=True,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_reopt_policy="bogus_policy",
            )


class TestAdaptReoptPolicyWrapperPassthrough:
    """hubbard_pipeline._run_internal_adapt_paop must accept and forward adapt_reopt_policy."""

    def test_wrapper_signature_accepts_reopt_policy(self):
        """The wrapper function signature must include adapt_reopt_policy."""
        import inspect
        from pipelines.hardcoded import hubbard_pipeline as hp_mod
        sig = inspect.signature(hp_mod._run_internal_adapt_paop)
        assert "adapt_reopt_policy" in sig.parameters, (
            "_run_internal_adapt_paop is missing adapt_reopt_policy parameter"
        )
        param = sig.parameters["adapt_reopt_policy"]
        assert param.default == "append_only", (
            f"Expected default='append_only', got default={param.default!r}"
        )


# ============================================================================
# Edge cases
# ============================================================================

class TestAdaptEdgeCases:
    """Edge case and error handling tests."""

    def test_hubbard_pool_hva_raises(self):
        """Using pool='hva' with problem='hubbard' should raise ValueError."""
        h_poly = build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        with pytest.raises(ValueError, match="pool='hva' is not valid"):
            _run_hardcoded_adapt_vqe(
                h_poly=h_poly,
                num_sites=2, ordering="blocked",
                problem="hubbard", adapt_pool="hva",
                t=1.0, u=4.0, dv=0.0, boundary="periodic",
                omega0=0.0, g_ep=0.0, n_ph_max=1, boson_encoding="binary",
                max_depth=5, eps_grad=1e-2, eps_energy=1e-6,
                maxiter=50, seed=7,
                allow_repeats=True, finite_angle_fallback=False,
                finite_angle=0.1, finite_angle_min_improvement=1e-12,
            )

    def test_invalid_pool_raises(self):
        h_poly = build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        with pytest.raises(ValueError, match="Unsupported adapt pool"):
            _run_hardcoded_adapt_vqe(
                h_poly=h_poly,
                num_sites=2, ordering="blocked",
                problem="hubbard", adapt_pool="nonexistent_pool",
                t=1.0, u=4.0, dv=0.0, boundary="periodic",
                omega0=0.0, g_ep=0.0, n_ph_max=1, boson_encoding="binary",
                max_depth=5, eps_grad=1e-2, eps_energy=1e-6,
                maxiter=50, seed=7,
                allow_repeats=True, finite_angle_fallback=False,
                finite_angle=0.1, finite_angle_min_improvement=1e-12,
            )

    def test_hubbard_pool_uccsd_paop_lf_full_raises(self):
        """Composite HH-only pool must reject pure Hubbard runs."""
        h_poly = build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        with pytest.raises(ValueError, match="only valid for problem='hh'"):
            _run_hardcoded_adapt_vqe(
                h_poly=h_poly,
                num_sites=2, ordering="blocked",
                problem="hubbard", adapt_pool="uccsd_paop_lf_full",
                t=1.0, u=4.0, dv=0.0, boundary="periodic",
                omega0=0.0, g_ep=0.0, n_ph_max=1, boson_encoding="binary",
                max_depth=5, eps_grad=1e-2, eps_energy=1e-6,
                maxiter=50, seed=7,
                allow_repeats=True, finite_angle_fallback=False,
                finite_angle=0.1, finite_angle_min_improvement=1e-12,
            )

    def test_hubbard_pool_full_meta_raises(self):
        """full_meta is HH-only and must reject pure Hubbard runs."""
        h_poly = build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
        with pytest.raises(ValueError, match="only valid for problem='hh'"):
            _run_hardcoded_adapt_vqe(
                h_poly=h_poly,
                num_sites=2, ordering="blocked",
                problem="hubbard", adapt_pool="full_meta",
                t=1.0, u=4.0, dv=0.0, boundary="periodic",
                omega0=0.0, g_ep=0.0, n_ph_max=1, boson_encoding="binary",
                max_depth=5, eps_grad=1e-2, eps_energy=1e-6,
                maxiter=50, seed=7,
                allow_repeats=True, finite_angle_fallback=False,
                finite_angle=0.1, finite_angle_min_improvement=1e-12,
            )

    def test_hh_phase1_allows_explicit_depth0_full_meta_override(self):
        h_poly = build_hubbard_holstein_hamiltonian(
            dims=2,
            J=1.0,
            U=2.0,
            omega0=1.0,
            g=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            v_t=None,
            v0=0.0,
            t_eval=None,
            repr_mode="JW",
            indexing="blocked",
            pbc=True,
            include_zero_point=True,
        )
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=h_poly,
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="full_meta",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=2,
            eps_grad=1e-2,
            eps_energy=1e-6,
            maxiter=30,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_continuation_mode="phase1_v1",
        )
        assert payload["phase1_depth0_full_meta_override"] is True
        assert payload["pool_type"] == "phase1_v1"


class TestHHPhase1Continuation:
    def _hh_h(self):
        return build_hubbard_holstein_hamiltonian(
            dims=2,
            J=1.0,
            U=2.0,
            omega0=1.0,
            g=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            v_t=None,
            v0=0.0,
            t_eval=None,
            repr_mode="JW",
            indexing="blocked",
            pbc=True,
            include_zero_point=True,
        )

    def test_legacy_history_omits_phase1_fields(self):
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=2,
            eps_grad=1e-2,
            eps_energy=1e-6,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_continuation_mode="legacy",
        )
        assert "continuation" not in payload
        assert "measurement_cache_summary" not in payload
        for row in payload.get("history", []):
            assert "candidate_family" not in row
            assert "refit_window_indices" not in row
            assert "simple_score" not in row

    def test_phase1_refit_window_matches_actual_window(self):
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=2,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase1_v1",
        )
        assert payload["continuation"]["mode"] == "phase1_v1"
        assert "stage_events" in payload["continuation"]
        for row in payload.get("history", []):
            assert row["refit_window_indices"] == row["reopt_active_indices"]


class TestHHPhase2Continuation:
    def _hh_h(self):
        return build_hubbard_holstein_hamiltonian(
            dims=2,
            J=1.0,
            U=2.0,
            omega0=1.0,
            g=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            v_t=None,
            v0=0.0,
            t_eval=None,
            repr_mode="JW",
            indexing="blocked",
            pbc=True,
            include_zero_point=True,
        )

    def test_phase2_emits_full_v2_and_memory_fields(self):
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=2,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase2_v1",
        )
        assert payload["continuation"]["mode"] == "phase2_v1"
        assert "optimizer_memory" in payload["continuation"]
        assert payload["continuation"]["optimizer_memory"]["parameter_count"] == payload["num_parameters"]
        for row in payload.get("history", []):
            assert row["refit_window_indices"] == row["reopt_active_indices"]
            assert "full_v2_score" in row
            assert "shortlisted_records" in row
            assert "optimizer_memory_source" in row


class TestHHPhase3Continuation:
    def _hh_h(self):
        return build_hubbard_holstein_hamiltonian(
            dims=2,
            J=1.0,
            U=2.0,
            omega0=1.0,
            g=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            v_t=None,
            v0=0.0,
            t_eval=None,
            repr_mode="JW",
            indexing="blocked",
            pbc=True,
            include_zero_point=True,
        )

    def test_phase3_emits_generator_motif_symmetry_and_lifetime_fields(self):
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=2,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=False,
            phase3_lifetime_cost_mode="phase3_v1",
        )
        continuation = payload["continuation"]
        assert continuation["mode"] == "phase3_v1"
        assert continuation["selected_generator_metadata"]
        assert "motif_library" in continuation
        assert continuation["symmetry_mitigation"]["mode"] == "verify_only"
        assert "rescue_history" in continuation
        assert payload["scaffold_fingerprint_lite"]["selected_generator_ids"]
        assert payload["compile_cost_proxy_summary"]["version"] == "phase3_v1_proxy"
        for row in payload.get("history", []):
            assert row["refit_window_indices"] == row["reopt_active_indices"]
            assert "generator_id" in row
            assert "symmetry_mode" in row
            assert "lifetime_cost_mode" in row
            assert "remaining_evaluations_proxy" in row

    def test_phase3_backend_cost_mode_rejects_non_hh_problem(self):
        h_poly = build_hubbard_hamiltonian(
            dims=2,
            t=1.0,
            U=4.0,
            v=0.0,
            repr_mode="JW",
            indexing="blocked",
            pbc=True,
        )
        with pytest.raises(ValueError, match="phase3_backend_cost_mode is only valid for problem='hh'"):
            _run_hardcoded_adapt_vqe(
                h_poly=h_poly,
                num_sites=2,
                ordering="blocked",
                problem="hubbard",
                adapt_pool="uccsd",
                t=1.0,
                u=4.0,
                dv=0.0,
                boundary="periodic",
                omega0=1.0,
                g_ep=0.5,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=1,
                eps_grad=1e-3,
                eps_energy=1e-8,
                maxiter=5,
                seed=7,
                allow_repeats=True,
                finite_angle_fallback=False,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_reopt_policy="windowed",
                adapt_window_size=1,
                adapt_window_topk=0,
                adapt_continuation_mode="phase3_v1",
                phase3_backend_cost_mode="transpile_single_v1",
                phase3_backend_name="ibm_boston",
            )

    def test_phase3_backend_cost_mode_emits_backend_compile_summary(self, monkeypatch: pytest.MonkeyPatch):
        class _StubBackendCompileOracle:
            def __init__(self, *, config, num_qubits, ref_state):
                self.config = config
                self.num_qubits = num_qubits
                self.ref_state = ref_state
                self.targets = ("FakeNighthawk",)
                self.resolution_audit = [
                    {
                        "requested_name": "ibm_boston",
                        "resolved_name": "FakeNighthawk",
                        "success": True,
                        "resolution_kind": "fake_exact",
                        "using_fake_backend": True,
                    }
                ]

            def snapshot_base(self, ops):
                return {"ops": [str(op.label) for op in ops]}

            def estimate_insertion(self, snapshot, *, candidate_term, position_id, proxy_baseline=None):
                return CompileCostEstimate(
                    new_pauli_actions=(0.0 if proxy_baseline is None else float(proxy_baseline.new_pauli_actions)),
                    new_rotation_steps=(0.0 if proxy_baseline is None else float(proxy_baseline.new_rotation_steps)),
                    position_shift_span=(0.0 if proxy_baseline is None else float(proxy_baseline.position_shift_span)),
                    refit_active_count=(0.0 if proxy_baseline is None else float(proxy_baseline.refit_active_count)),
                    proxy_total=(0.0 if proxy_baseline is None else float(proxy_baseline.proxy_total)),
                    cx_proxy_total=(0.0 if proxy_baseline is None else float(proxy_baseline.cx_proxy_total)),
                    sq_proxy_total=(0.0 if proxy_baseline is None else float(proxy_baseline.sq_proxy_total)),
                    gate_proxy_total=(0.0 if proxy_baseline is None else float(proxy_baseline.gate_proxy_total)),
                    max_pauli_weight=(0.0 if proxy_baseline is None else float(proxy_baseline.max_pauli_weight)),
                    source_mode="backend_transpile_v1",
                    penalty_total=4.5,
                    depth_surrogate=4.5,
                    compile_gate_open=True,
                    failure_reason=None,
                    selected_backend_name="FakeNighthawk",
                    selected_resolution_kind="fake_exact",
                    aggregation_mode="single_backend",
                    target_backend_names=["FakeNighthawk"],
                    successful_target_count=1,
                    failed_target_count=0,
                    raw_delta_compiled_count_2q=2.0,
                    delta_compiled_count_2q=2.0,
                    raw_delta_compiled_depth=3.0,
                    delta_compiled_depth=3.0,
                    raw_delta_compiled_size=5.0,
                    delta_compiled_size=5.0,
                    delta_compiled_cx_count=2.0,
                    delta_compiled_ecr_count=0.0,
                    base_compiled_count_2q=10.0,
                    base_compiled_depth=12.0,
                    base_compiled_size=20.0,
                    trial_compiled_count_2q=12.0,
                    trial_compiled_depth=15.0,
                    trial_compiled_size=25.0,
                    proxy_baseline=(
                        None
                        if proxy_baseline is None
                        else {
                            "new_pauli_actions": float(proxy_baseline.new_pauli_actions),
                            "new_rotation_steps": float(proxy_baseline.new_rotation_steps),
                            "position_shift_span": float(proxy_baseline.position_shift_span),
                            "refit_active_count": float(proxy_baseline.refit_active_count),
                            "proxy_total": float(proxy_baseline.proxy_total),
                            "cx_proxy_total": float(proxy_baseline.cx_proxy_total),
                            "sq_proxy_total": float(proxy_baseline.sq_proxy_total),
                            "gate_proxy_total": float(proxy_baseline.gate_proxy_total),
                            "max_pauli_weight": float(proxy_baseline.max_pauli_weight),
                        }
                    ),
                    selected_backend_row={
                        "transpile_backend": "FakeNighthawk",
                        "resolution_kind": "fake_exact",
                        "compiled_count_2q": 12,
                        "compiled_depth": 15,
                        "compiled_size": 25,
                    },
                )

            def final_scaffold_summary(self, ops):
                return {
                    "rows": [
                        {
                            "transpile_backend": "FakeNighthawk",
                            "resolution_kind": "fake_exact",
                            "transpile_status": "ok",
                            "compiled_count_2q": 18,
                            "compiled_depth": 21,
                            "compiled_size": 33,
                            "compiled_op_counts": {"swap": 1, "cx": 18},
                            "absolute_burden_score_v1": 20.43,
                        }
                    ],
                    "selected_backend": {
                        "transpile_backend": "FakeNighthawk",
                        "resolution_kind": "fake_exact",
                        "transpile_status": "ok",
                        "compiled_count_2q": 18,
                        "compiled_depth": 21,
                        "compiled_size": 33,
                        "compiled_op_counts": {"swap": 1, "cx": 18},
                        "absolute_burden_score_v1": 20.43,
                    },
                }

            def cache_summary(self):
                return {"row_hits": 2, "row_misses": 1, "compile_failures": 0, "cache_entries": 3}

        monkeypatch.setattr(_adapt_mod, "BackendCompileOracle", _StubBackendCompileOracle)

        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=2,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode="phase3_v1",
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=False,
            phase3_lifetime_cost_mode="phase3_v1",
            phase3_backend_cost_mode="transpile_single_v1",
            phase3_backend_name="ibm_boston",
        )

        assert payload["compile_cost_mode"] == "transpile_single_v1"
        assert payload["backend_compile_cost_summary"]["selected_backend"]["transpile_backend"] == "FakeNighthawk"
        assert payload["continuation"]["backend_compile_cost_summary"]["cache_summary"]["cache_entries"] == 3
        assert payload["scaffold_fingerprint_lite"]["compile_cost_mode"] == "transpile_single_v1"
        assert payload["scaffold_fingerprint_lite"]["backend_target_names"] == ["FakeNighthawk"]
        assert any(row["compile_cost_mode"] == "transpile_single_v1" for row in payload["history"])
        assert any(row["compile_cost_source"] == "backend_transpile_v1" for row in payload["history"])
        assert any(
            isinstance(row.get("compile_cost_backend"), dict)
            and row["compile_cost_backend"].get("selected_backend_name") == "FakeNighthawk"
            for row in payload["history"]
        )

    def test_phase3_eps_energy_is_telemetry_only_without_drop_policy(self, monkeypatch: pytest.MonkeyPatch):
        events: list[tuple[str, dict[str, object]]] = []
        original_ai_log = _adapt_mod._ai_log

        def _capture(event: str, **fields: object) -> None:
            events.append((str(event), dict(fields)))

        monkeypatch.setattr(_adapt_mod, "_ai_log", _capture)
        try:
            payload, _ = _run_hardcoded_adapt_vqe(
                h_poly=self._hh_h(),
                num_sites=2,
                ordering="blocked",
                problem="hh",
                adapt_pool="paop_lf_std",
                t=1.0,
                u=2.0,
                dv=0.0,
                boundary="periodic",
                omega0=1.0,
                g_ep=0.5,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=3,
                eps_grad=-1.0,
                eps_energy=1e9,
                maxiter=20,
                seed=17,
                adapt_inner_optimizer="SPSA",
                adapt_spsa_callback_every=10,
                adapt_spsa_progress_every_s=999.0,
                allow_repeats=True,
                finite_angle_fallback=False,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_reopt_policy="windowed",
                adapt_window_size=1,
                adapt_window_topk=0,
                adapt_continuation_mode="phase3_v1",
                phase3_symmetry_mitigation_mode="off",
                phase3_enable_rescue=False,
                phase3_lifetime_cost_mode="phase3_v1",
            )
            assert payload["success"] is True
            assert bool(payload["eps_energy_termination_enabled"]) is False
            assert bool(payload["eps_grad_termination_enabled"]) is False
            assert bool(payload["adapt_drop_policy_enabled"]) is True
            assert payload["adapt_drop_floor_resolved"] == pytest.approx(5e-4)
            assert int(payload["adapt_drop_patience_resolved"]) == 3
            assert int(payload["adapt_drop_min_depth_resolved"]) == 12
            assert payload["adapt_grad_floor_resolved"] == pytest.approx(2e-2)
            assert payload["adapt_drop_policy_source"] == "auto_hh_staged"
            assert payload["adapt_drop_floor_source"] == "auto_hh_staged"
            assert payload["adapt_drop_patience_source"] == "auto_hh_staged"
            assert payload["adapt_drop_min_depth_source"] == "auto_hh_staged"
            assert payload["adapt_grad_floor_source"] == "auto_hh_staged"
            assert str(payload["stop_reason"]) in {"max_depth", "pool_exhausted"}
            assert str(payload["stop_reason"]) != "eps_energy"
            assert all(bool(row["eps_energy_termination_enabled"]) is False for row in payload.get("history", []))
            assert all(bool(row["eps_grad_termination_enabled"]) is False for row in payload.get("history", []))
            assert any(int(row["eps_energy_low_streak"]) >= 2 for row in payload.get("history", []))

            suppressed = [ev[1] for ev in events if ev[0] == "hardcoded_adapt_eps_energy_termination_suppressed"]
            assert len(suppressed) >= 1
            converged_energy = [ev for ev in events if ev[0] == "hardcoded_adapt_converged_energy"]
            assert len(converged_energy) == 0
        finally:
            monkeypatch.setattr(_adapt_mod, "_ai_log", original_ai_log)

    def test_phase3_low_gradient_no_longer_terminates(self, monkeypatch: pytest.MonkeyPatch):
        events: list[tuple[str, dict[str, object]]] = []
        original_ai_log = _adapt_mod._ai_log

        def _capture(event: str, **fields: object) -> None:
            events.append((str(event), dict(fields)))

        monkeypatch.setattr(_adapt_mod, "_ai_log", _capture)
        try:
            payload, _ = _run_hardcoded_adapt_vqe(
                h_poly=self._hh_h(),
                num_sites=2,
                ordering="blocked",
                problem="hh",
                adapt_pool="paop_lf_std",
                t=1.0,
                u=2.0,
                dv=0.0,
                boundary="periodic",
                omega0=1.0,
                g_ep=0.5,
                n_ph_max=1,
                boson_encoding="binary",
                max_depth=3,
                eps_grad=1e9,
                eps_energy=1e-9,
                maxiter=20,
                seed=23,
                adapt_inner_optimizer="SPSA",
                adapt_spsa_callback_every=10,
                adapt_spsa_progress_every_s=999.0,
                allow_repeats=True,
                finite_angle_fallback=False,
                finite_angle=0.1,
                finite_angle_min_improvement=1e-12,
                adapt_reopt_policy="windowed",
                adapt_window_size=1,
                adapt_window_topk=0,
                adapt_continuation_mode="phase3_v1",
                phase3_symmetry_mitigation_mode="off",
                phase3_enable_rescue=False,
                phase3_lifetime_cost_mode="phase3_v1",
            )
            assert payload["success"] is True
            assert bool(payload["eps_grad_termination_enabled"]) is False
            assert str(payload["stop_reason"]) in {"max_depth", "pool_exhausted"}
            assert str(payload["stop_reason"]) != "eps_grad"
            assert any(bool(row["eps_grad_threshold_hit"]) is True for row in payload.get("history", []))

            suppressed = [ev for ev in events if ev[0] == "hardcoded_adapt_eps_grad_termination_suppressed"]
            assert len(suppressed) >= 1
            converged_grad = [ev for ev in events if ev[0] == "hardcoded_adapt_converged_grad"]
            assert len(converged_grad) == 0
        finally:
            monkeypatch.setattr(_adapt_mod, "_ai_log", original_ai_log)

    def test_phase3_drop_plateau_preempts_eps_energy_hard_stop(self, monkeypatch: pytest.MonkeyPatch):
        events: list[tuple[str, dict[str, object]]] = []
        original_ai_log = _adapt_mod._ai_log

        def _capture(event: str, **fields: object) -> None:
            events.append((str(event), dict(fields)))

        monkeypatch.setattr(_adapt_mod, "_ai_log", _capture)
        try:
            payload, _ = _run_hardcoded_adapt_vqe(
                h_poly=self._hh_h(),
                num_sites=2,
                ordering="blocked",
                problem="hh",
                adapt_pool="paop_lf_std",
                t=1.0,
                u=2.0,
                dv=0.0,
                boundary="periodic",
                omega0=1.0,
                g_ep=0.5,
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
                adapt_reopt_policy="windowed",
                adapt_window_size=1,
                adapt_window_topk=0,
                adapt_continuation_mode="phase3_v1",
                adapt_drop_floor=1e9,
                adapt_drop_patience=1,
                adapt_drop_min_depth=1,
                adapt_grad_floor=-1.0,
                phase3_symmetry_mitigation_mode="off",
                phase3_enable_rescue=False,
                phase3_lifetime_cost_mode="phase3_v1",
            )
            assert payload["success"] is True
            assert bool(payload["eps_energy_termination_enabled"]) is False
            assert payload["adapt_drop_floor_resolved"] == pytest.approx(1e9)
            assert int(payload["adapt_drop_patience_resolved"]) == 1
            assert int(payload["adapt_drop_min_depth_resolved"]) == 1
            assert payload["adapt_drop_floor_source"] == "explicit"
            assert payload["adapt_drop_patience_source"] == "explicit"
            assert payload["adapt_drop_min_depth_source"] == "explicit"
            assert str(payload["stop_reason"]) == "drop_plateau"
            assert str(payload["stop_reason"]) != "eps_energy"
            assert all(bool(row["eps_energy_termination_enabled"]) is False for row in payload.get("history", []))

            residual_opened = [ev for ev in events if ev[0] == "hardcoded_adapt_phase1_residual_opened_on_plateau"]
            assert len(residual_opened) >= 1
            converged_drop = [ev for ev in events if ev[0] == "hardcoded_adapt_converged_drop_plateau"]
            assert len(converged_drop) == 1
            converged_energy = [ev for ev in events if ev[0] == "hardcoded_adapt_converged_energy"]
            assert len(converged_energy) == 0
        finally:
            monkeypatch.setattr(_adapt_mod, "_ai_log", original_ai_log)

    def test_hubbard_legacy_still_allows_eps_grad_stop(self):
        h_poly = build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )
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
            eps_grad=1e9,
            eps_energy=1e-12,
            maxiter=20,
            seed=29,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_continuation_mode="legacy",
        )
        assert payload["success"] is True
        assert bool(payload["eps_grad_termination_enabled"]) is True
        assert str(payload["stop_reason"]) == "eps_grad"
        assert bool(payload["adapt_drop_policy_enabled"]) is False
        assert payload["adapt_drop_policy_source"] == "default_off"


class TestHHContinuationModeGatingNegative:
    def _hh_h(self):
        return build_hubbard_holstein_hamiltonian(
            dims=2,
            J=1.0,
            U=2.0,
            omega0=1.0,
            g=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            v_t=None,
            v0=0.0,
            t_eval=None,
            repr_mode="JW",
            indexing="blocked",
            pbc=True,
            include_zero_point=True,
        )

    @pytest.mark.parametrize("mode", ["legacy", "phase1_v1", "phase2_v1"])
    def test_phase3_knobs_do_not_leak_into_older_modes(self, mode: str):
        payload, _ = _run_hardcoded_adapt_vqe(
            h_poly=self._hh_h(),
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="paop_lf_std",
            t=1.0,
            u=2.0,
            dv=0.0,
            boundary="periodic",
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=2,
            eps_grad=1e-3,
            eps_energy=1e-8,
            maxiter=20,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1e-12,
            adapt_reopt_policy="windowed",
            adapt_window_size=1,
            adapt_window_topk=0,
            adapt_continuation_mode=mode,
            phase3_symmetry_mitigation_mode="verify_only",
            phase3_enable_rescue=True,
            phase3_lifetime_cost_mode="phase3_v1",
        )

        if mode == "legacy":
            assert "continuation" not in payload
            for row in payload.get("history", []):
                assert "full_v2_score" not in row
                assert "shortlisted_records" not in row
                assert "optimizer_memory_source" not in row
                assert "generator_id" not in row
                assert "symmetry_mode" not in row
                assert "lifetime_cost_mode" not in row
                assert "remaining_evaluations_proxy" not in row
            return

        continuation = payload["continuation"]
        assert continuation["mode"] == mode

        if mode == "phase1_v1":
            assert "optimizer_memory" not in continuation
            assert "selected_generator_metadata" not in continuation
            assert "motif_library" not in continuation
            assert "symmetry_mitigation" not in continuation
            assert "rescue_history" not in continuation
            for row in payload.get("history", []):
                assert "full_v2_score" not in row
                assert "shortlisted_records" not in row
                assert "optimizer_memory_source" not in row
                assert "generator_id" not in row
                assert "symmetry_mode" not in row
                assert "lifetime_cost_mode" not in row
                assert "remaining_evaluations_proxy" not in row
            return

        assert "optimizer_memory" in continuation
        assert "selected_generator_metadata" not in continuation
        assert "motif_library" not in continuation
        assert "symmetry_mitigation" not in continuation
        assert "rescue_history" not in continuation
        for row in payload.get("history", []):
            assert "full_v2_score" in row
            assert "shortlisted_records" in row
            assert "optimizer_memory_source" in row
            assert "generator_id" not in row
            assert "symmetry_mode" not in row
            assert "lifetime_cost_mode" not in row
            assert "remaining_evaluations_proxy" not in row


# ────────────────────────────────────────────────────────────────────
#  P2 — windowed reopt pure helpers
# ────────────────────────────────────────────────────────────────────

class TestResolveReoptActiveIndices:
    """Tests for _resolve_reopt_active_indices (pure deterministic helper)."""

    def test_append_only_returns_last(self):
        theta = np.array([0.1, 0.2, 0.3, 0.4])
        idx, name = _resolve_reopt_active_indices(
            policy="append_only", n=4, theta=theta,
            window_size=3, window_topk=0, periodic_full_refit_triggered=False,
        )
        assert idx == [3]
        assert name == "append_only"

    def test_full_returns_all(self):
        theta = np.array([0.1, 0.2, 0.3, 0.4])
        idx, name = _resolve_reopt_active_indices(
            policy="full", n=4, theta=theta,
            window_size=3, window_topk=0, periodic_full_refit_triggered=False,
        )
        assert idx == [0, 1, 2, 3]
        assert name == "full"

    def test_windowed_newest_window(self):
        theta = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        idx, name = _resolve_reopt_active_indices(
            policy="windowed", n=5, theta=theta,
            window_size=2, window_topk=0, periodic_full_refit_triggered=False,
        )
        assert idx == [3, 4]
        assert name == "windowed"

    def test_windowed_topk_selection(self):
        theta = np.array([0.9, 0.01, 0.02, 0.5, 0.6])
        idx, _name = _resolve_reopt_active_indices(
            policy="windowed", n=5, theta=theta,
            window_size=2, window_topk=1, periodic_full_refit_triggered=False,
        )
        # newest = [3,4]; older by |theta| desc: [0(0.9), 2(0.02), 1(0.01)]
        # topk=1 -> pick [0]
        assert 0 in idx
        assert 3 in idx
        assert 4 in idx

    def test_windowed_topk_tiebreak_ascending(self):
        theta = np.array([0.5, 0.5, 0.3, 0.4])
        idx, _name = _resolve_reopt_active_indices(
            policy="windowed", n=4, theta=theta,
            window_size=1, window_topk=1, periodic_full_refit_triggered=False,
        )
        # newest = [3]; older by |theta| desc = [0(0.5),1(0.5),2(0.3)]
        # tie at 0.5: ascending index -> pick 0
        assert idx == [0, 3]

    def test_windowed_sorted_ascending(self):
        theta = np.array([0.9, 0.01, 0.02, 0.5, 0.6])
        idx, _name = _resolve_reopt_active_indices(
            policy="windowed", n=5, theta=theta,
            window_size=2, window_topk=2, periodic_full_refit_triggered=False,
        )
        assert idx == sorted(idx)

    def test_windowed_window_larger_than_n(self):
        theta = np.array([0.1, 0.2])
        idx, _name = _resolve_reopt_active_indices(
            policy="windowed", n=2, theta=theta,
            window_size=10, window_topk=5, periodic_full_refit_triggered=False,
        )
        assert idx == [0, 1]

    def test_periodic_full_override(self):
        theta = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        idx, name = _resolve_reopt_active_indices(
            policy="windowed", n=5, theta=theta,
            window_size=2, window_topk=0, periodic_full_refit_triggered=True,
        )
        assert idx == [0, 1, 2, 3, 4]
        assert name == "windowed_periodic_full"

    def test_append_only_ignores_periodic_full(self):
        """append_only does not honour periodic_full — only windowed does."""
        theta = np.array([0.1, 0.2, 0.3])
        idx, name = _resolve_reopt_active_indices(
            policy="append_only", n=3, theta=theta,
            window_size=3, window_topk=0, periodic_full_refit_triggered=True,
        )
        assert idx == [2]
        assert name == "append_only"

    def test_n_zero_returns_empty(self):
        """n=0 is a degenerate case — returns empty list."""
        idx, _name = _resolve_reopt_active_indices(
            policy="windowed", n=0, theta=np.array([]),
            window_size=2, window_topk=0, periodic_full_refit_triggered=False,
        )
        assert idx == []

    def test_invalid_policy_raises(self):
        with pytest.raises(ValueError, match="Unknown reopt policy"):
            _resolve_reopt_active_indices(
                policy="bogus", n=1, theta=np.array([0.1]),
                window_size=2, window_topk=0, periodic_full_refit_triggered=False,
            )

    def test_n_equals_1(self):
        theta = np.array([0.42])
        idx, _name = _resolve_reopt_active_indices(
            policy="windowed", n=1, theta=theta,
            window_size=3, window_topk=2, periodic_full_refit_triggered=False,
        )
        assert idx == [0]


class TestMakeReducedObjective:
    """Tests for _make_reduced_objective (pure mapping helper)."""

    def test_full_prefix_passthrough(self):
        theta_full = np.array([0.1, 0.2, 0.3])
        active = [0, 1, 2]
        calls = []

        def fake_obj(t):
            calls.append(t.copy())
            return float(np.sum(t))

        obj_r, x0 = _make_reduced_objective(theta_full, active, fake_obj)
        np.testing.assert_array_equal(x0, theta_full)
        val = obj_r(x0)
        assert val == pytest.approx(0.6)
        np.testing.assert_array_equal(calls[-1], theta_full)

    def test_subset_freezes_inactive(self):
        theta_full = np.array([10.0, 0.2, 0.3, 20.0])
        active = [1, 2]
        calls = []

        def fake_obj(t):
            calls.append(t.copy())
            return float(np.sum(t))

        obj_r, x0 = _make_reduced_objective(theta_full, active, fake_obj)
        np.testing.assert_array_equal(x0, np.array([0.2, 0.3]))
        val = obj_r(np.array([0.5, 0.6]))
        expected_full = np.array([10.0, 0.5, 0.6, 20.0])
        np.testing.assert_array_equal(calls[-1], expected_full)
        assert val == pytest.approx(expected_full.sum())

    def test_multiple_active_indices(self):
        theta_full = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        active = [0, 2, 4]
        log = []

        def fake_obj(t):
            log.append(t.copy())
            return float(t[0] + t[2] + t[4])

        obj_r, x0 = _make_reduced_objective(theta_full, active, fake_obj)
        assert len(x0) == 3
        np.testing.assert_array_equal(x0, np.array([1.0, 3.0, 5.0]))


class TestValidReoptPoliciesSet:
    """Smoke test: constant matches spec."""

    def test_members(self):
        assert _VALID_REOPT_POLICIES == {"append_only", "full", "windowed"}


class TestAdaptCLIParsingWindowed:
    """CLI arg-parsing tests for windowed knobs."""

    def test_accepts_windowed(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["adapt_pipeline.py", "--adapt-reopt-policy", "windowed"],
        )
        args = _adapt_mod.parse_args()
        assert args.adapt_reopt_policy == "windowed"

    def test_defaults(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["adapt_pipeline.py"])
        args = _adapt_mod.parse_args()
        assert args.adapt_reopt_policy == "append_only"
        assert args.adapt_window_size == 3
        assert args.adapt_window_topk == 0
        assert args.adapt_full_refit_every == 0
        assert args.adapt_final_full_refit == "true"

    def test_overrides(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            [
                "adapt_pipeline.py",
                "--adapt-reopt-policy", "windowed",
                "--adapt-window-size", "5",
                "--adapt-window-topk", "2",
                "--adapt-full-refit-every", "4",
                "--adapt-final-full-refit", "false",
            ],
        )
        args = _adapt_mod.parse_args()
        assert args.adapt_window_size == 5
        assert args.adapt_window_topk == 2
        assert args.adapt_full_refit_every == 4
        assert args.adapt_final_full_refit == "false"


class TestWindowedReoptValidation:
    """Validation guard tests (called via _run_hardcoded_adapt_vqe)."""

    @pytest.fixture()
    def tiny_h(self):
        return build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )

    def _call(self, h, **overrides):
        defaults = dict(
            h_poly=h, num_sites=2, ordering="blocked",
            problem="hubbard", adapt_pool="uccsd",
            t=1.0, u=4.0, dv=0.0, boundary="periodic",
            omega0=0.0, g_ep=0.0, n_ph_max=1, boson_encoding="binary",
            max_depth=1, eps_grad=1e-2, eps_energy=1e-6,
            maxiter=5, seed=7,
            allow_repeats=True, finite_angle_fallback=False,
            finite_angle=0.1, finite_angle_min_improvement=1e-12,
        )
        defaults.update(overrides)
        return _run_hardcoded_adapt_vqe(**defaults)

    def test_window_size_lt1_raises(self, tiny_h):
        with pytest.raises(ValueError, match="adapt_window_size"):
            self._call(tiny_h, adapt_reopt_policy="windowed",
                       adapt_window_size=0)

    def test_topk_negative_raises(self, tiny_h):
        with pytest.raises(ValueError, match="adapt_window_topk"):
            self._call(tiny_h, adapt_reopt_policy="windowed",
                       adapt_window_topk=-1)

    def test_refit_every_negative_raises(self, tiny_h):
        with pytest.raises(ValueError, match="adapt_full_refit_every"):
            self._call(tiny_h, adapt_reopt_policy="windowed",
                       adapt_full_refit_every=-1)

    def test_invalid_policy_raises(self, tiny_h):
        with pytest.raises(ValueError, match="adapt_reopt_policy"):
            self._call(tiny_h, adapt_reopt_policy="bogus")


class TestWindowedReoptIntegration:
    """End-to-end integration tests for windowed reopt."""

    @pytest.fixture()
    def tiny_h(self):
        return build_hubbard_hamiltonian(
            dims=2, t=1.0, U=4.0, v=0.0,
            repr_mode="JW", indexing="blocked", pbc=True,
        )

    def _run(self, h, **overrides):
        defaults = dict(
            h_poly=h, num_sites=2, ordering="blocked",
            problem="hubbard", adapt_pool="uccsd",
            t=1.0, u=4.0, dv=0.0, boundary="periodic",
            omega0=0.0, g_ep=0.0, n_ph_max=1, boson_encoding="binary",
            max_depth=3, eps_grad=1e-2, eps_energy=1e-6,
            maxiter=40, seed=7,
            allow_repeats=True, finite_angle_fallback=False,
            finite_angle=0.1, finite_angle_min_improvement=1e-12,
        )
        defaults.update(overrides)
        payload, _psi = _run_hardcoded_adapt_vqe(**defaults)
        return payload

    # -- payload schema --

    def test_windowed_payload_valid(self, tiny_h):
        res = self._run(tiny_h, adapt_reopt_policy="windowed",
                        adapt_window_size=2, adapt_window_topk=0)
        assert "adapt_window_size" in res
        assert "adapt_window_topk" in res
        assert "adapt_full_refit_every" in res
        assert "adapt_final_full_refit" in res
        assert "final_full_refit" in res

    def test_history_row_metadata(self, tiny_h):
        res = self._run(tiny_h, adapt_reopt_policy="windowed",
                        adapt_window_size=2, adapt_window_topk=0)
        for row in res.get("history", []):
            assert "reopt_policy_effective" in row
            assert "reopt_active_indices" in row
            assert "reopt_active_count" in row

    def test_active_count_bounded(self, tiny_h):
        res = self._run(tiny_h, adapt_reopt_policy="windowed",
                        adapt_window_size=1, adapt_window_topk=0)
        for row in res.get("history", []):
            assert row["reopt_active_count"] <= row.get("depth", 999)

    def test_periodic_trigger(self, tiny_h):
        res = self._run(tiny_h, adapt_reopt_policy="windowed",
                        adapt_window_size=1, adapt_window_topk=0,
                        adapt_full_refit_every=2, max_depth=4)
        triggered = [r["reopt_periodic_full_refit_triggered"]
                     for r in res.get("history", [])]
        # at least one True expected at some cumulative-depth % 2 == 0
        assert any(triggered) or len(triggered) < 2

    def test_final_refit_metadata(self, tiny_h):
        res = self._run(tiny_h, adapt_reopt_policy="windowed",
                        adapt_window_size=1, adapt_window_topk=0,
                        adapt_final_full_refit=True)
        ffr = res.get("final_full_refit", {})
        assert "executed" in ffr

    def test_final_refit_false_skips(self, tiny_h):
        res = self._run(tiny_h, adapt_reopt_policy="windowed",
                        adapt_window_size=1, adapt_window_topk=0,
                        adapt_final_full_refit=False)
        ffr = res.get("final_full_refit", {})
        assert ffr.get("executed") is False or ffr.get("skipped_reason") is not None

    def test_knobs_recorded(self, tiny_h):
        res = self._run(tiny_h, adapt_reopt_policy="windowed",
                        adapt_window_size=5, adapt_window_topk=2,
                        adapt_full_refit_every=3)
        assert res["adapt_window_size"] == 5
        assert res["adapt_window_topk"] == 2
        assert res["adapt_full_refit_every"] == 3

    # -- regression: existing policies unchanged --

    def test_append_only_regression(self, tiny_h):
        res = self._run(tiny_h, adapt_reopt_policy="append_only")
        for row in res.get("history", []):
            assert row["reopt_active_count"] == 1

    def test_full_regression(self, tiny_h):
        res = self._run(tiny_h, adapt_reopt_policy="full")
        for row in res.get("history", []):
            d = row.get("depth", 1)
            assert row["reopt_active_count"] == d

    def test_topk_carry(self, tiny_h):
        res = self._run(tiny_h, adapt_reopt_policy="windowed",
                        adapt_window_size=1, adapt_window_topk=1,
                        max_depth=3)
        for row in res.get("history", []):
            d = row.get("depth", 1)
            expected_max = min(1 + 1, d)  # window + topk, capped by depth
            assert row["reopt_active_count"] <= expected_max

    def test_replay_compat(self, tiny_h):
        """Windowed run must still emit replay-compatible fields."""
        res = self._run(tiny_h, adapt_reopt_policy="windowed",
                        adapt_window_size=2, adapt_window_topk=0)
        assert "operators" in res
        assert "optimal_point" in res


class TestPeriodicFullRefitCadence:
    """Edge cases for periodic full-refit triggering."""

    def test_periodic_full_returns_all(self):
        theta = np.array([0.1, 0.2, 0.3])
        # periodic_full_refit_triggered=True should override to full prefix
        idx, name = _resolve_reopt_active_indices(
            policy="windowed", n=3, theta=theta,
            window_size=1, window_topk=0, periodic_full_refit_triggered=True,
        )
        assert idx == [0, 1, 2]
        assert name == "windowed_periodic_full"

    def test_disabled_when_not_triggered(self):
        # periodic_full_refit_triggered=False with windowed stays windowed.
        theta = np.array([0.1, 0.2, 0.3, 0.4])
        idx, name = _resolve_reopt_active_indices(
            policy="windowed", n=4, theta=theta,
            window_size=2, window_topk=0, periodic_full_refit_triggered=False,
        )
        assert idx == [2, 3]
        assert name == "windowed"


class TestAdaptRefExactEnergyReuse:
    @staticmethod
    def _hh_nq_total() -> int:
        return int(2 * 2 + 2 * boson_qubits_per_site(1, "binary"))

    @classmethod
    def _ref_payload(
        cls,
        *,
        t: float = 1.0,
        include_exact_energy: bool = True,
        exact_energy: float = 0.15866790412572704,
    ) -> dict[str, object]:
        nq_total = cls._hh_nq_total()
        payload: dict[str, object] = {
            "settings": {
                "L": 2,
                "problem": "hh",
                "ordering": "blocked",
                "boundary": "open",
                "t": float(t),
                "u": 4.0,
                "dv": 0.0,
                "omega0": 1.0,
                "g_ep": 0.5,
                "n_ph_max": 1,
                "boson_encoding": "binary",
            },
            "initial_state": {
                "source": "adapt_vqe",
                "amplitudes_qn_to_q0": {
                    format(0, f"0{nq_total}b"): {"re": 1.0, "im": 0.0},
                },
            },
            "adapt_vqe": {
                "ansatz_depth": 2,
            },
        }
        if include_exact_energy:
            payload["ground_state"] = {
                "exact_energy_filtered": float(exact_energy),
            }
        return payload

    @classmethod
    def _run_main_with_ref(
        cls,
        *,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        ref_payload: dict[str, object],
        exact_impl,
    ) -> tuple[dict[str, object], dict[str, float | None]]:
        ref_json = tmp_path / "warm_ref.json"
        out_json = tmp_path / "adapt_out.json"
        ref_json.write_text(json.dumps(ref_payload), encoding="utf-8")

        captured: dict[str, float | None] = {"exact_gs_override": None}
        dim = 1 << cls._hh_nq_total()

        def _fake_run_hardcoded_adapt_vqe(**kwargs):
            captured["exact_gs_override"] = kwargs.get("exact_gs_override")
            psi = np.zeros(dim, dtype=complex)
            psi[0] = 1.0
            return {
                "success": True,
                "method": "mock_adapt",
                "energy": float(kwargs.get("exact_gs_override")),
                "pool_type": str(kwargs.get("adapt_pool")),
                "ansatz_depth": 1,
                "num_parameters": 1,
            }, psi

        def _fake_simulate_trajectory(**kwargs):
            return ([{"time": 0.0, "fidelity": 1.0}], [])

        monkeypatch.setattr(_adapt_mod, "_exact_gs_energy_for_problem", exact_impl)
        monkeypatch.setattr(_adapt_mod, "_run_hardcoded_adapt_vqe", _fake_run_hardcoded_adapt_vqe)
        monkeypatch.setattr(_adapt_mod, "_simulate_trajectory", _fake_simulate_trajectory)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_pipeline.py",
                "--L", "2",
                "--problem", "hh",
                "--t", "1.0",
                "--u", "4.0",
                "--dv", "0.0",
                "--omega0", "1.0",
                "--g-ep", "0.5",
                "--n-ph-max", "1",
                "--boson-encoding", "binary",
                "--boundary", "open",
                "--ordering", "blocked",
                "--adapt-pool", "paop_lf_std",
                "--adapt-continuation-mode", "phase3_v1",
                "--adapt-ref-json", str(ref_json),
                "--skip-pdf",
                "--output-json", str(out_json),
            ],
        )

        _adapt_mod.main()
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        return payload, captured

    def test_main_reuses_exact_energy_from_metadata_compatible_ref(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        ref_payload = self._ref_payload(include_exact_energy=True)

        def _fail_exact(*args, **kwargs):
            raise AssertionError("_exact_gs_energy_for_problem should not run when warm exact energy is reusable")

        payload, captured = self._run_main_with_ref(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            ref_payload=ref_payload,
            exact_impl=_fail_exact,
        )

        assert payload["ground_state"]["exact_energy_source"] == "adapt_ref_json"
        assert payload["ground_state"]["exact_energy"] == pytest.approx(0.15866790412572704)
        assert captured["exact_gs_override"] == pytest.approx(0.15866790412572704)
        assert bool(payload["adapt_ref_import"]["exact_energy_reused"]) is True
        assert payload["adapt_ref_import"]["exact_energy_reuse_mismatches"] == []
        assert bool(payload["adapt_ref_import"]["ansatz_input_state_persisted"]) is True
        assert payload["adapt_ref_import"]["initial_state_handoff_state_kind"] is None
        assert payload["ansatz_input_state"]["source"] == "adapt_vqe"
        assert payload["ansatz_input_state"]["nq_total"] == self._hh_nq_total()

    def test_main_falls_back_when_ref_lacks_exact_energy(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        sentinel = 0.777
        ref_payload = self._ref_payload(include_exact_energy=False)

        def _fake_exact(*args, **kwargs):
            return float(sentinel)

        payload, captured = self._run_main_with_ref(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            ref_payload=ref_payload,
            exact_impl=_fake_exact,
        )

        assert payload["ground_state"]["exact_energy_source"] == "computed"
        assert payload["ground_state"]["exact_energy"] == pytest.approx(sentinel)
        assert captured["exact_gs_override"] == pytest.approx(sentinel)
        assert bool(payload["adapt_ref_import"]["exact_energy_reused"]) is False
        assert payload["adapt_ref_import"]["exact_energy_reuse_mismatches"] == []

    def test_main_falls_back_when_ref_metadata_mismatches(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        sentinel = 0.666
        ref_payload = self._ref_payload(include_exact_energy=True, t=0.75)

        def _fake_exact(*args, **kwargs):
            return float(sentinel)

        payload, captured = self._run_main_with_ref(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            ref_payload=ref_payload,
            exact_impl=_fake_exact,
        )

        assert payload["ground_state"]["exact_energy_source"] == "computed"
        assert payload["ground_state"]["exact_energy"] == pytest.approx(sentinel)
        assert captured["exact_gs_override"] == pytest.approx(sentinel)
        assert bool(payload["adapt_ref_import"]["exact_energy_reused"]) is False
        mismatches = payload["adapt_ref_import"]["exact_energy_reuse_mismatches"]
        assert isinstance(mismatches, list)
        assert any(str(msg).startswith("t:") for msg in mismatches)

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/test/test_hh_continuation_scoring.py
```py
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_continuation_scoring import (
    FullScoreConfig,
    MeasurementCacheAudit,
    Phase2CurvatureOracle,
    Phase2NoveltyOracle,
    Phase1CompileCostOracle,
    SimpleScoreConfig,
    build_full_candidate_features,
    build_candidate_features,
    compatibility_penalty,
    full_v2_score,
    lifetime_weight_components,
    measurement_group_keys_for_term,
    remaining_evaluations_proxy,
    shortlist_records,
    trust_region_drop,
)
from pipelines.hardcoded.hh_continuation_generators import build_generator_metadata
from pipelines.hardcoded.hh_continuation_types import CompileCostEstimate
from pipelines.hardcoded.hh_continuation_symmetry import build_symmetry_spec
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.pauli_words import PauliTerm


def test_simple_v1_prefers_higher_gradient_with_equal_costs() -> None:
    cfg = SimpleScoreConfig(lambda_F=1.0, lambda_compile=0.0, lambda_measure=0.0, lambda_leak=0.0, z_alpha=0.0)
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    cost = oracle.estimate(candidate_term_count=1, position_id=2, append_position=2, refit_active_count=1)
    mstats = meas.estimate(["x"])
    feat_a = build_candidate_features(
        stage_name="core",
        candidate_label="a",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=2,
        append_position=2,
        positions_considered=[2],
        gradient_signed=0.4,
        metric_proxy=0.4,
        sigma_hat=0.0,
        refit_window_indices=[2],
        compile_cost=cost,
        measurement_stats=mstats,
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=cfg,
    )
    feat_b = build_candidate_features(
        stage_name="core",
        candidate_label="b",
        candidate_family="core",
        candidate_pool_index=1,
        position_id=2,
        append_position=2,
        positions_considered=[2],
        gradient_signed=0.2,
        metric_proxy=0.2,
        sigma_hat=0.0,
        refit_window_indices=[2],
        compile_cost=cost,
        measurement_stats=mstats,
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=cfg,
    )
    assert float(feat_a.simple_score or 0.0) > float(feat_b.simple_score or 0.0)


def test_stage_gate_blocks_score() -> None:
    cfg = SimpleScoreConfig()
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    cost = oracle.estimate(candidate_term_count=1, position_id=0, append_position=1, refit_active_count=1)
    mstats = meas.estimate(["x"])
    feat = build_candidate_features(
        stage_name="core",
        candidate_label="resid",
        candidate_family="residual",
        candidate_pool_index=0,
        position_id=0,
        append_position=1,
        positions_considered=[0, 1],
        gradient_signed=1.0,
        metric_proxy=1.0,
        sigma_hat=0.0,
        refit_window_indices=[0, 1],
        compile_cost=cost,
        measurement_stats=mstats,
        leakage_penalty=0.0,
        stage_gate_open=False,
        leakage_gate_open=True,
        trough_probe_triggered=True,
        trough_detected=True,
        cfg=cfg,
    )
    assert feat.simple_score == float("-inf")


def test_backend_compile_cost_replaces_proxy_term_in_simple_score() -> None:
    cfg = SimpleScoreConfig(lambda_F=0.0, lambda_compile=1.0, lambda_measure=0.0, lambda_leak=0.0, z_alpha=0.0)
    meas = MeasurementCacheAudit()
    cost = CompileCostEstimate(
        new_pauli_actions=3.0,
        new_rotation_steps=2.0,
        position_shift_span=1.0,
        refit_active_count=4.0,
        proxy_total=99.0,
        cx_proxy_total=11.0,
        sq_proxy_total=22.0,
        gate_proxy_total=33.0,
        max_pauli_weight=2.0,
        source_mode="backend_transpile_v1",
        penalty_total=7.5,
        depth_surrogate=5.0,
        compile_gate_open=True,
        selected_backend_name="FakeNighthawk",
        proxy_baseline={
            "new_pauli_actions": 3.0,
            "new_rotation_steps": 2.0,
            "position_shift_span": 1.0,
            "refit_active_count": 4.0,
            "proxy_total": 99.0,
            "cx_proxy_total": 11.0,
            "sq_proxy_total": 22.0,
            "gate_proxy_total": 33.0,
            "max_pauli_weight": 2.0,
        },
        selected_backend_row={"transpile_backend": "FakeNighthawk", "compiled_count_2q": 18},
    )
    feat = build_candidate_features(
        stage_name="core",
        candidate_label="backend",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=1,
        append_position=1,
        positions_considered=[1],
        gradient_signed=0.0,
        metric_proxy=0.0,
        sigma_hat=0.0,
        refit_window_indices=[0, 1],
        compile_cost=cost,
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=cfg,
    )
    assert feat.compile_cost_source == "backend_transpile_v1"
    assert float(feat.compile_cost_total) == pytest.approx(7.5)
    assert float(feat.simple_score or 0.0) == pytest.approx(-7.5)
    assert feat.compiled_position_cost_proxy["proxy_total"] == pytest.approx(99.0)
    assert feat.compiled_position_cost_backend is not None
    assert feat.compiled_position_cost_backend["selected_backend_name"] == "FakeNighthawk"


def test_backend_compile_gate_closed_blocks_simple_and_full_scores() -> None:
    cfg = SimpleScoreConfig()
    full_cfg = FullScoreConfig()
    meas = MeasurementCacheAudit()
    cost = CompileCostEstimate(
        new_pauli_actions=0.0,
        new_rotation_steps=0.0,
        position_shift_span=0.0,
        refit_active_count=1.0,
        proxy_total=1.0,
        source_mode="backend_transpile_v1",
        penalty_total=float("inf"),
        depth_surrogate=float("inf"),
        compile_gate_open=False,
        failure_reason="all_targets_failed",
    )
    feat = build_candidate_features(
        stage_name="core",
        candidate_label="blocked",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=1.0,
        metric_proxy=1.0,
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=cost,
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=cfg,
    )
    assert feat.simple_score == float("-inf")
    score, fallback = full_v2_score(feat, full_cfg)
    assert score == float("-inf")
    assert fallback == "compile_gate_closed"


def test_simple_v1_uses_g_abs_not_g_lcb_for_ranking() -> None:
    cfg = SimpleScoreConfig(lambda_F=0.0, lambda_compile=0.0, lambda_measure=0.0, lambda_leak=0.0, z_alpha=10.0)
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    cost = oracle.estimate(candidate_term_count=1, position_id=0, append_position=0, refit_active_count=1)
    mstats = meas.estimate(["x"])
    feat = build_candidate_features(
        stage_name="core",
        candidate_label="a",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=0.4,
        metric_proxy=0.0,
        sigma_hat=0.03,
        refit_window_indices=[0],
        compile_cost=cost,
        measurement_stats=mstats,
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=cfg,
    )
    assert float(feat.g_lcb) == pytest.approx(0.1)
    assert float(feat.simple_score or 0.0) == pytest.approx(0.4)


def test_measurement_cache_reuse_accounting() -> None:
    cache = MeasurementCacheAudit(nominal_shots_per_group=10)
    first = cache.estimate(["a", "b"])
    assert first.groups_new == 2
    cache.commit(["a", "b"])
    second = cache.estimate(["a", "b", "c"])
    assert second.groups_reused == 2
    assert second.groups_new == 1
    summary = cache.summary()
    assert str(summary["plan_version"]) == "phase1_qwc_basis_cover_reuse"


def test_measurement_cache_reuses_more_specific_seen_basis_keys() -> None:
    cache = MeasurementCacheAudit(nominal_shots_per_group=10)
    cache.commit(["xz"])
    reused = cache.estimate(["ez"])
    assert reused.groups_reused == 1
    assert reused.groups_new == 0


def test_measurement_group_keys_for_term_merges_qwc_compatible_labels() -> None:
    poly = PauliPolynomial(
        "JW",
        [
            PauliTerm(2, ps="xe", pc=1.0),
            PauliTerm(2, ps="xz", pc=1.0),
            PauliTerm(2, ps="ez", pc=1.0),
        ],
    )
    term = type("_DummyAnsatzTerm", (), {"label": "macro", "polynomial": poly})()
    assert measurement_group_keys_for_term(term) == ["xz"]


def _term(label: str) -> object:
    return type(
        "_DummyAnsatzTerm",
        (),
        {
            "label": str(label),
            "polynomial": PauliPolynomial("JW", [PauliTerm(len(str(label)), ps=str(label), pc=1.0)]),
        },
    )()


def test_trust_region_drop_matches_newton_branch() -> None:
    got = trust_region_drop(0.4, 2.0, 1.0, 1.0)
    assert got == pytest.approx(0.04)


def test_full_v2_score_falls_back_safely_without_window_curvature() -> None:
    cfg = FullScoreConfig(z_alpha=0.0, lambda_F=1.0, rho=0.5, gamma_N=1.0, wD=0.0, wG=0.0, wC=0.0, wP=0.0, wc=0.0)
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    feat = build_candidate_features(
        stage_name="core",
        candidate_label="x",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=0.5,
        metric_proxy=0.5,
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=oracle.estimate(candidate_term_count=1, position_id=0, append_position=0, refit_active_count=1),
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=SimpleScoreConfig(),
    )
    feat = type(feat)(**{**feat.__dict__, "h_hat": 0.5, "curvature_mode": "self_only"})
    score, fallback = full_v2_score(feat, cfg)
    assert score > 0.0
    assert fallback == "self_curvature_only"


def test_build_full_candidate_features_clips_novelty_and_preserves_window() -> None:
    psi_ref = np.zeros(2, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    base = build_candidate_features(
        stage_name="core",
        candidate_label="x",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=1,
        append_position=1,
        positions_considered=[1],
        gradient_signed=0.3,
        metric_proxy=0.3,
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=oracle.estimate(candidate_term_count=1, position_id=1, append_position=1, refit_active_count=1),
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=SimpleScoreConfig(),
    )
    feat = build_full_candidate_features(
        base_feature=base,
        psi_state=psi_ref,
        candidate_term=_term("x"),
        window_terms=[_term("x")],
        window_labels=["x"],
        cfg=FullScoreConfig(shortlist_size=2),
        novelty_oracle=Phase2NoveltyOracle(),
        curvature_oracle=Phase2CurvatureOracle(),
        compiled_cache={},
        pauli_action_cache={},
        optimizer_memory=None,
    )
    assert 0.0 <= float(feat.novelty or 0.0) <= 1.0
    assert feat.refit_window_indices == [0]
    assert feat.full_v2_score is not None


def test_phase1_compile_cost_oracle_penalizes_heavier_pauli_structure() -> None:
    oracle = Phase1CompileCostOracle()
    light_term = type(
        "_DummyAnsatzTerm",
        (),
        {"label": "light", "polynomial": PauliPolynomial("JW", [PauliTerm(2, ps="xe", pc=1.0)])},
    )()
    heavy_term = type(
        "_DummyAnsatzTerm",
        (),
        {"label": "heavy", "polynomial": PauliPolynomial("JW", [PauliTerm(2, ps="xx", pc=1.0)])},
    )()
    light = oracle.estimate(
        candidate_term_count=1,
        position_id=0,
        append_position=0,
        refit_active_count=1,
        candidate_term=light_term,
    )
    heavy = oracle.estimate(
        candidate_term_count=1,
        position_id=0,
        append_position=0,
        refit_active_count=1,
        candidate_term=heavy_term,
    )
    assert heavy.gate_proxy_total > light.gate_proxy_total
    assert heavy.proxy_total > light.proxy_total


def test_compatibility_penalty_uses_measurement_mismatch_signal() -> None:
    cfg = FullScoreConfig(
        compat_overlap_weight=0.0,
        compat_comm_weight=0.0,
        compat_curv_weight=0.0,
        compat_sched_weight=0.0,
        compat_measure_weight=1.0,
    )
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()

    def _feat_and_record(label: str, term: object) -> dict[str, object]:
        feat = build_candidate_features(
            stage_name="core",
            candidate_label=str(label),
            candidate_family="core",
            candidate_pool_index=0,
            position_id=0,
            append_position=0,
            positions_considered=[0],
            gradient_signed=0.5,
            metric_proxy=0.5,
            sigma_hat=0.0,
            refit_window_indices=[0],
            compile_cost=oracle.estimate(
                candidate_term_count=1,
                position_id=0,
                append_position=0,
                refit_active_count=1,
                candidate_term=term,
            ),
            measurement_stats=meas.estimate(measurement_group_keys_for_term(term)),
            leakage_penalty=0.0,
            stage_gate_open=True,
            leakage_gate_open=True,
            trough_probe_triggered=False,
            trough_detected=False,
            cfg=SimpleScoreConfig(lambda_compile=0.0, lambda_measure=0.0),
        )
        return {"feature": feat, "candidate_term": term}

    rec_xz = _feat_and_record("xz", _term("xz"))
    rec_ez = _feat_and_record("ez", _term("ez"))
    rec_yy = _feat_and_record("yy", _term("yy"))

    close_penalty = compatibility_penalty(record_a=rec_xz, record_b=rec_ez, cfg=cfg)
    far_penalty = compatibility_penalty(record_a=rec_xz, record_b=rec_yy, cfg=cfg)

    assert close_penalty["measurement_mismatch"] < far_penalty["measurement_mismatch"]
    assert close_penalty["total"] < far_penalty["total"]


def test_shortlist_only_expensive_scoring_calls_oracles_for_shortlist() -> None:
    class _CountingNovelty(Phase2NoveltyOracle):
        def __init__(self) -> None:
            self.calls = 0

        def estimate(self, *args, **kwargs):
            self.calls += 1
            return super().estimate(*args, **kwargs)

    psi_ref = np.zeros(2, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    cheap_records = []
    for idx, grad in enumerate([0.9, 0.8, 0.3, 0.2]):
        feat = build_candidate_features(
            stage_name="core",
            candidate_label=f"x{idx}",
            candidate_family="core",
            candidate_pool_index=idx,
            position_id=0,
            append_position=0,
            positions_considered=[0],
            gradient_signed=float(grad),
            metric_proxy=float(grad),
            sigma_hat=0.0,
            refit_window_indices=[0],
            compile_cost=oracle.estimate(candidate_term_count=1, position_id=0, append_position=0, refit_active_count=1),
            measurement_stats=meas.estimate([f"x{idx}"]),
            leakage_penalty=0.0,
            stage_gate_open=True,
            leakage_gate_open=True,
            trough_probe_triggered=False,
            trough_detected=False,
            cfg=SimpleScoreConfig(lambda_compile=0.0, lambda_measure=0.0),
        )
        cheap_records.append(
            {
                "feature": feat,
                "simple_score": float(feat.simple_score or 0.0),
                "candidate_pool_index": idx,
                "position_id": 0,
                "candidate_term": _term("x"),
                "window_terms": [],
                "window_labels": [],
            }
        )
    shortlisted = shortlist_records(cheap_records, cfg=FullScoreConfig(shortlist_fraction=0.5, shortlist_size=2))
    novelty = _CountingNovelty()
    for rec in shortlisted:
        build_full_candidate_features(
            base_feature=rec["feature"],
            psi_state=psi_ref,
            candidate_term=rec["candidate_term"],
            window_terms=[],
            window_labels=[],
            cfg=FullScoreConfig(shortlist_fraction=0.5, shortlist_size=2),
            novelty_oracle=novelty,
            curvature_oracle=Phase2CurvatureOracle(),
            compiled_cache={},
            pauli_action_cache={},
            optimizer_memory=None,
        )
    assert len(shortlisted) == 2
    assert novelty.calls == 2


def test_remaining_evaluations_proxy_uses_remaining_depth_mode() -> None:
    got = remaining_evaluations_proxy(current_depth=2, max_depth=6, mode="remaining_depth")
    assert got == pytest.approx(5.0)


def test_lifetime_weight_components_are_zero_when_mode_off() -> None:
    cfg = FullScoreConfig(lifetime_cost_mode="off", lifetime_weight=1.0)
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    feat = build_candidate_features(
        stage_name="core",
        candidate_label="x",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=0.5,
        metric_proxy=0.5,
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=oracle.estimate(candidate_term_count=1, position_id=0, append_position=0, refit_active_count=1),
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=SimpleScoreConfig(),
        current_depth=2,
        max_depth=6,
        lifetime_cost_mode="off",
        remaining_evaluations_proxy_mode="remaining_depth",
    )
    comps = lifetime_weight_components(feat, cfg)
    assert comps["remaining_evaluations_proxy"] == pytest.approx(5.0)
    assert comps["total"] == pytest.approx(0.0)


def test_full_v2_motif_bonus_and_lifetime_weighting_are_deterministic() -> None:
    cfg = FullScoreConfig(
        z_alpha=0.0,
        lambda_F=1.0,
        rho=0.5,
        gamma_N=1.0,
        wD=0.0,
        wG=0.0,
        wC=0.0,
        wP=0.0,
        wc=0.0,
        lifetime_cost_mode="phase3_v1",
        remaining_evaluations_proxy_mode="remaining_depth",
        lifetime_weight=0.1,
        motif_bonus_weight=1.0,
    )
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    feat = build_candidate_features(
        stage_name="core",
        candidate_label="x",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=0.5,
        metric_proxy=0.5,
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=oracle.estimate(candidate_term_count=1, position_id=0, append_position=0, refit_active_count=1),
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=SimpleScoreConfig(),
        current_depth=1,
        max_depth=4,
        motif_bonus=0.2,
        lifetime_cost_mode="phase3_v1",
        remaining_evaluations_proxy_mode="remaining_depth",
    )
    feat = type(feat)(
        **{
            **feat.__dict__,
            "h_hat": 0.5,
            "curvature_mode": "self_only",
        }
    )
    score_with_bonus, _ = full_v2_score(feat, cfg)
    score_without_bonus, _ = full_v2_score(
        type(feat)(**{**feat.__dict__, "motif_bonus": 0.0}),
        cfg,
    )
    assert score_with_bonus > score_without_bonus


def test_build_candidate_features_carries_generator_and_symmetry_metadata() -> None:
    poly = PauliPolynomial(
        "JW",
        [
            PauliTerm(6, ps="eyeexy", pc=1.0),
            PauliTerm(6, ps="eyeeyx", pc=-1.0),
        ],
    )
    sym = build_symmetry_spec(family_id="paop_lf_std", mitigation_mode="verify_only")
    meta = build_generator_metadata(
        label="macro_candidate",
        polynomial=poly,
        family_id="paop_lf_std",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_spec=sym.__dict__,
    )
    oracle = Phase1CompileCostOracle()
    meas = MeasurementCacheAudit()
    feat = build_candidate_features(
        stage_name="core",
        candidate_label="macro_candidate",
        candidate_family="paop_lf_std",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=0.4,
        metric_proxy=0.4,
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=oracle.estimate(candidate_term_count=2, position_id=0, append_position=0, refit_active_count=1),
        measurement_stats=meas.estimate(["macro"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=SimpleScoreConfig(),
        generator_metadata=meta.__dict__,
        symmetry_spec=sym.__dict__,
        symmetry_mode="phase3_shared_spec",
        symmetry_mitigation_mode="verify_only",
        current_depth=0,
        max_depth=3,
        lifetime_cost_mode="phase3_v1",
        remaining_evaluations_proxy_mode="remaining_depth",
    )
    assert feat.generator_id == meta.generator_id
    assert feat.template_id == meta.template_id
    assert feat.is_macro_generator is True
    assert feat.symmetry_mode == "phase3_shared_spec"
    assert feat.symmetry_mitigation_mode == "verify_only"
    assert feat.remaining_evaluations_proxy == pytest.approx(4.0)

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/pipelines/hardcoded/adapt_pipeline.py
```py
#!/usr/bin/env python3
"""Hardcoded ADAPT-VQE end-to-end Hubbard / Hubbard-Holstein pipeline.

Flow:
1) Build Hubbard (or HH) Hamiltonian (JW) from repo source-of-truth helpers.
2) Build operator pool (UCCSD, CSE, full_hamiltonian, HVA, or PAOP variants).
3) Run standard ADAPT-VQE: commutator gradients, one operator per
   iteration, COBYLA/SPSA inner optimizer, optional repeats.
4) Run Suzuki-2 Trotter dynamics + exact dynamics from the ADAPT ground state.
5) Emit JSON + compact PDF artifact.

Uses the *same* src/quantum/ primitives as the regular VQE hardcoded pipeline.
No dependency on Qiskit in the core path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — this file lives at pipelines/hardcoded/adapt_pipeline.py
# REPO_ROOT is the top-level Holstein_test directory.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # Holstein_test/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from docs.reports.pdf_utils import (
    HAS_MATPLOTLIB,
    require_matplotlib,
    get_plt,
    get_PdfPages,
    render_command_page,
    render_text_page,
    current_command_string,
)
from docs.reports.report_pages import (
    render_executive_summary_page,
    render_manifest_overview_page,
    render_section_divider_page,
)

# Module-level aliases used by the plotting body
plt = get_plt() if HAS_MATPLOTLIB else None  # type: ignore[assignment]
PdfPages = get_PdfPages() if HAS_MATPLOTLIB else type("PdfPages", (), {})  # type: ignore[misc]

# ---------------------------------------------------------------------------
# Imports from the active repo quantum modules (no pydephasing fallback).
# ---------------------------------------------------------------------------
from src.quantum.hubbard_latex_python_pairs import (
    build_hubbard_hamiltonian,
    build_hubbard_holstein_hamiltonian,
    boson_qubits_per_site,
)
from src.quantum.hartree_fock_reference_state import hartree_fock_statevector
from src.quantum.ansatz_parameterization import (
    AnsatzParameterLayout,
    build_parameter_layout,
    project_runtime_theta_block_mean,
    runtime_indices_for_logical_indices,
    runtime_insert_position,
    serialize_layout,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import (
    CompiledPolynomialAction,
    adapt_commutator_grad_from_hpsi,
    apply_compiled_polynomial as _apply_compiled_polynomial_shared,
    compile_polynomial_action as _compile_polynomial_action_shared,
    energy_via_one_apply,
)
from src.quantum.pauli_actions import (
    CompiledPauliAction,
    apply_compiled_pauli as _apply_compiled_pauli_shared,
    apply_exp_term as _apply_exp_term_shared,
    compile_pauli_action_exyz as _compile_pauli_action_exyz_shared,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.pauli_words import PauliTerm
from src.quantum.spsa_optimizer import spsa_minimize
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    HardcodedUCCSDAnsatz,
    HubbardHolsteinLayerwiseAnsatz,
    HubbardTermwiseAnsatz,
    apply_exp_pauli_polynomial,
    apply_exp_pauli_polynomial_termwise,
    apply_pauli_string,
    basis_state,
    exact_ground_energy_sector,
    exact_ground_energy_sector_hh,
    expval_pauli_polynomial,
    half_filled_num_particles,
    hamiltonian_matrix,
    hartree_fock_bitstring,
    hubbard_holstein_reference_state,
)
from pipelines.hardcoded.hh_continuation_types import (
    CandidateFeatures,
    Phase2OptimizerMemoryAdapter,
    ScaffoldFingerprintLite,
)
from pipelines.hardcoded.hh_continuation_generators import (
    build_pool_generator_registry,
    build_runtime_split_child_sets,
    build_runtime_split_children,
    build_split_event,
    selected_generator_metadata_for_labels,
)
from pipelines.hardcoded.handoff_state_bundle import build_statevector_manifest
from pipelines.hardcoded.hh_continuation_motifs import (
    extract_motif_library,
    load_motif_library_from_json,
    select_tiled_generators_from_library,
)
from pipelines.hardcoded.hh_continuation_symmetry import (
    build_symmetry_spec,
    leakage_penalty_from_spec,
    verify_symmetry_sequence,
)
from pipelines.hardcoded.hh_continuation_rescue import (
    RescueConfig,
    rank_rescue_candidates,
    should_trigger_rescue,
)
from pipelines.hardcoded.hh_continuation_stage_control import (
    StageController,
    StageControllerConfig,
    allowed_positions,
    detect_trough,
    should_probe_positions,
)
from pipelines.hardcoded.hh_continuation_scoring import (
    CompatibilityPenaltyOracle,
    FullScoreConfig,
    MeasurementCacheAudit,
    Phase2CurvatureOracle,
    Phase1CompileCostOracle,
    Phase2NoveltyOracle,
    SimpleScoreConfig,
    build_candidate_features,
    build_full_candidate_features,
    greedy_batch_select,
    measurement_group_keys_for_term,
    shortlist_records,
)
from pipelines.hardcoded.hh_continuation_pruning import (
    PruneConfig,
    apply_pruning,
    post_prune_refit,
    rank_prune_candidates,
)
from pipelines.hardcoded.hh_backend_compile_oracle import (
    BackendCompileConfig,
    BackendCompileOracle,
)

try:
    from src.quantum.operator_pools import make_pool as make_paop_pool
except Exception as exc:  # pragma: no cover - defensive fallback
    make_paop_pool = None
    _PAOP_IMPORT_ERROR = str(exc)
else:
    _PAOP_IMPORT_ERROR = ""

EXACT_LABEL = "Exact_Hardcode"
EXACT_METHOD = "python_matrix_eigendecomposition"
_ADAPT_GRADIENT_PARITY_RTOL = 1e-8

PAULI_MATS = {
    "e": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def _ai_log(event: str, **fields: Any) -> None:
    payload = {
        "event": str(event),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    print(f"AI_LOG {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


# ---------------------------------------------------------------------------
# Utility helpers (mirror hardcoded VQE pipeline)
# ---------------------------------------------------------------------------

def _to_ixyz(label_exyz: str) -> str:
    return str(label_exyz).replace("e", "I").upper()


def _normalize_state(psi: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(psi))
    if nrm <= 0.0:
        raise ValueError("Encountered zero-norm state.")
    return psi / nrm


def _collect_hardcoded_terms_exyz(poly: Any, tol: float = 1e-12) -> tuple[list[str], dict[str, complex]]:
    coeff_map: dict[str, complex] = {}
    order: list[str] = []
    for term in poly.return_polynomial():
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) <= tol:
            continue
        if label not in coeff_map:
            coeff_map[label] = 0.0 + 0.0j
            order.append(label)
        coeff_map[label] += coeff
    cleaned_order = [lbl for lbl in order if abs(coeff_map[lbl]) > tol]
    cleaned_map = {lbl: coeff_map[lbl] for lbl in cleaned_order}
    return cleaned_order, cleaned_map


def _pauli_matrix_exyz(label: str) -> np.ndarray:
    mats = [PAULI_MATS[ch] for ch in label]
    out = mats[0]
    for mat in mats[1:]:
        out = np.kron(out, mat)
    return out


def _build_hamiltonian_matrix(coeff_map_exyz: dict[str, complex]) -> np.ndarray:
    if not coeff_map_exyz:
        return np.zeros((1, 1), dtype=complex)
    nq = len(next(iter(coeff_map_exyz)))
    dim = 1 << nq
    hmat = np.zeros((dim, dim), dtype=complex)
    for label, coeff in coeff_map_exyz.items():
        hmat += coeff * _pauli_matrix_exyz(label)
    return hmat


# ---------------------------------------------------------------------------
# Compiled Pauli + Trotter (identical to VQE pipeline)
# ---------------------------------------------------------------------------

def _compile_pauli_action(label_exyz: str, nq: int) -> CompiledPauliAction:
    return _compile_pauli_action_exyz_shared(label_exyz=label_exyz, nq=nq)


def _apply_compiled_pauli(psi: np.ndarray, action: CompiledPauliAction) -> np.ndarray:
    return _apply_compiled_pauli_shared(psi=psi, action=action)


def _compile_polynomial_action(
    poly: Any,
    tol: float = 1e-15,
    *,
    pauli_action_cache: dict[str, CompiledPauliAction] | None = None,
) -> CompiledPolynomialAction:
    """Compile a PauliPolynomial into reusable Pauli actions for repeated apply."""
    terms = poly.return_polynomial()
    if not terms:
        return CompiledPolynomialAction(nq=0, terms=tuple())
    return _compile_polynomial_action_shared(
        poly,
        tol=float(tol),
        pauli_action_cache=pauli_action_cache,
    )


def _apply_compiled_polynomial(state: np.ndarray, compiled_poly: CompiledPolynomialAction) -> np.ndarray:
    """Apply a compiled PauliPolynomial action to a statevector."""
    if int(getattr(compiled_poly, "nq", 0)) == 0 and len(compiled_poly.terms) == 0:
        return np.zeros_like(state)
    return _apply_compiled_polynomial_shared(state, compiled_poly)


def _apply_exp_term(
    psi: np.ndarray, action: CompiledPauliAction, coeff: complex, alpha: float, tol: float = 1e-12,
) -> np.ndarray:
    return _apply_exp_term_shared(
        psi=psi,
        action=action,
        coeff=complex(coeff),
        dt=float(alpha),
        tol=float(tol),
    )


def _evolve_trotter_suzuki2_absolute(
    psi0, ordered_labels, coeff_map, compiled_actions, time_value, trotter_steps,
) -> np.ndarray:
    psi = np.array(psi0, copy=True)
    if abs(time_value) <= 1e-15:
        return psi
    dt = float(time_value) / float(trotter_steps)
    half = 0.5 * dt
    for _ in range(trotter_steps):
        for label in ordered_labels:
            psi = _apply_exp_term(psi, compiled_actions[label], coeff_map[label], half)
        for label in reversed(ordered_labels):
            psi = _apply_exp_term(psi, compiled_actions[label], coeff_map[label], half)
    return _normalize_state(psi)


def _expectation_hamiltonian(psi: np.ndarray, hmat: np.ndarray) -> float:
    return float(np.real(np.vdot(psi, hmat @ psi)))


# ---------------------------------------------------------------------------
# Observables (identical to VQE pipeline)
# ---------------------------------------------------------------------------

def _occupation_site0(psi: np.ndarray, num_sites: int) -> tuple[float, float]:
    probs = np.abs(psi) ** 2
    n_up = 0.0
    n_dn = 0.0
    for idx, prob in enumerate(probs):
        n_up += float((idx >> 0) & 1) * float(prob)
        n_dn += float((idx >> num_sites) & 1) * float(prob)
    return float(n_up), float(n_dn)


def _doublon_total(psi: np.ndarray, num_sites: int) -> float:
    probs = np.abs(psi) ** 2
    out = 0.0
    for idx, prob in enumerate(probs):
        count = 0
        for site in range(num_sites):
            up = (idx >> site) & 1
            dn = (idx >> (num_sites + site)) & 1
            count += int(up * dn)
        out += float(count) * float(prob)
    return float(out)


def _state_to_amplitudes_qn_to_q0(psi: np.ndarray, cutoff: float = 1e-12) -> dict[str, dict[str, float]]:
    nq = int(round(math.log2(psi.size)))
    out: dict[str, dict[str, float]] = {}
    for idx, amp in enumerate(psi):
        if abs(amp) < cutoff:
            continue
        bit = format(idx, f"0{nq}b")
        out[bit] = {"re": float(np.real(amp)), "im": float(np.imag(amp))}
    return out


def _state_from_amplitudes_qn_to_q0(
    amplitudes_qn_to_q0: dict[str, Any],
    nq_total: int,
) -> np.ndarray:
    if not isinstance(amplitudes_qn_to_q0, dict) or len(amplitudes_qn_to_q0) == 0:
        raise ValueError("Missing or empty initial_state.amplitudes_qn_to_q0 in ADAPT JSON.")
    dim = 1 << int(nq_total)
    psi = np.zeros(dim, dtype=complex)
    for bitstr, comp in amplitudes_qn_to_q0.items():
        if not isinstance(bitstr, str) or len(bitstr) != int(nq_total) or any(ch not in "01" for ch in bitstr):
            raise ValueError(f"Invalid bitstring key in ADAPT amplitudes: {bitstr!r}")
        if not isinstance(comp, dict):
            raise ValueError(f"Amplitude payload for bitstring {bitstr!r} must be a dict.")
        re_val = float(comp.get("re", 0.0))
        im_val = float(comp.get("im", 0.0))
        idx = int(bitstr, 2)
        psi[idx] = complex(re_val, im_val)
    return _normalize_state(psi)


def _load_adapt_initial_state(
    adapt_json_path: Path,
    nq_total: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not adapt_json_path.exists():
        raise FileNotFoundError(f"ADAPT input JSON not found: {adapt_json_path}")
    raw = json.loads(adapt_json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("ADAPT input JSON must be a top-level object.")
    initial_state = raw.get("initial_state")
    if not isinstance(initial_state, dict):
        raise ValueError("ADAPT input JSON missing object key: initial_state")
    stored_nq_total_raw = initial_state.get("nq_total", None)
    if stored_nq_total_raw is not None and int(stored_nq_total_raw) != int(nq_total):
        raise ValueError(
            f"ADAPT input JSON initial_state.nq_total={int(stored_nq_total_raw)} does not match expected nq_total={int(nq_total)}."
        )
    amplitudes = initial_state.get("amplitudes_qn_to_q0")
    psi = _state_from_amplitudes_qn_to_q0(amplitudes, int(nq_total))
    meta = {
        "settings": raw.get("settings", {}),
        "adapt_vqe": raw.get("adapt_vqe", {}),
        "ground_state": raw.get("ground_state", {}),
        "vqe": raw.get("vqe", {}),
        "initial_state_source": initial_state.get("source"),
        "initial_state_handoff_state_kind": initial_state.get("handoff_state_kind"),
    }
    return psi, meta


def _default_adapt_input_state(
    *,
    problem: str,
    num_sites: int,
    ordering: str,
    n_ph_max: int,
    boson_encoding: str,
) -> tuple[np.ndarray, str, str]:
    problem_key = str(problem).strip().lower()
    num_particles = half_filled_num_particles(int(num_sites))
    if problem_key == "hh":
        psi = _normalize_state(
            np.asarray(
                hubbard_holstein_reference_state(
                    dims=int(num_sites),
                    num_particles=num_particles,
                    n_ph_max=int(n_ph_max),
                    boson_encoding=str(boson_encoding),
                    indexing=str(ordering),
                ),
                dtype=complex,
            ).reshape(-1)
        )
    else:
        psi = _normalize_state(
            np.asarray(
                hartree_fock_statevector(
                    int(num_sites),
                    num_particles,
                    indexing=str(ordering),
                ),
                dtype=complex,
            ).reshape(-1)
        )
    return psi, "hf", "reference_state"


_HH_STAGED_CONTINUATION_MODES = frozenset({"phase1_v1", "phase2_v1", "phase3_v1"})


def _extract_nested(payload: Mapping[str, Any], *keys: str) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _resolve_exact_energy_from_payload(payload: Mapping[str, Any]) -> float | None:
    candidates = (
        _extract_nested(payload, "ground_state", "exact_energy_filtered"),
        _extract_nested(payload, "ground_state", "exact_energy"),
        _extract_nested(payload, "adapt_vqe", "exact_gs_energy"),
        _extract_nested(payload, "vqe", "exact_energy"),
    )
    for raw in candidates:
        if raw is None:
            continue
        try:
            value = float(raw)
        except Exception:
            continue
        if np.isfinite(value):
            return float(value)
    return None


def _validate_adapt_ref_metadata_for_exact_reuse(
    *,
    adapt_settings: Mapping[str, Any],
    args: argparse.Namespace,
    is_hh: bool,
    float_tol: float = 1e-10,
) -> list[str]:
    if not isinstance(adapt_settings, Mapping):
        return ["settings missing from adapt_ref_json"]

    mismatches: list[str] = []

    def _cmp_scalar(field: str, expected: Any, actual: Any) -> None:
        if actual != expected:
            mismatches.append(f"{field}: expected={expected!r} adapt_ref_json={actual!r}")

    def _cmp_float(field: str, expected: float, actual_raw: Any) -> None:
        try:
            actual = float(actual_raw)
        except Exception:
            mismatches.append(f"{field}: expected={expected!r} adapt_ref_json={actual_raw!r}")
            return
        if abs(float(expected) - actual) > float(float_tol):
            mismatches.append(f"{field}: expected={float(expected)!r} adapt_ref_json={actual!r}")

    _cmp_scalar("L", int(args.L), adapt_settings.get("L"))
    _cmp_scalar("problem", str(args.problem).strip().lower(), str(adapt_settings.get("problem", "")).strip().lower())
    _cmp_scalar("ordering", str(args.ordering), adapt_settings.get("ordering"))
    _cmp_scalar("boundary", str(args.boundary), adapt_settings.get("boundary"))
    _cmp_float("t", float(args.t), adapt_settings.get("t"))
    _cmp_float("u", float(args.u), adapt_settings.get("u"))
    _cmp_float("dv", float(args.dv), adapt_settings.get("dv"))

    if bool(is_hh):
        _cmp_float("omega0", float(args.omega0), adapt_settings.get("omega0"))
        _cmp_float("g_ep", float(args.g_ep), adapt_settings.get("g_ep"))
        _cmp_scalar("n_ph_max", int(args.n_ph_max), adapt_settings.get("n_ph_max"))
        _cmp_scalar("boson_encoding", str(args.boson_encoding), adapt_settings.get("boson_encoding"))

    return mismatches


def _resolve_exact_energy_override_from_adapt_ref(
    *,
    adapt_ref_meta: Mapping[str, Any] | None,
    args: argparse.Namespace,
    problem: str,
    continuation_mode: str | None,
) -> tuple[float | None, str, list[str]]:
    if not isinstance(adapt_ref_meta, Mapping):
        return None, "computed", []
    if str(problem).strip().lower() != "hh":
        return None, "computed", []
    mode_key = str(continuation_mode if continuation_mode is not None else "legacy").strip().lower()
    if mode_key not in _HH_STAGED_CONTINUATION_MODES:
        return None, "computed", []

    mismatches = _validate_adapt_ref_metadata_for_exact_reuse(
        adapt_settings=adapt_ref_meta.get("settings", {}),
        args=args,
        is_hh=True,
    )
    if len(mismatches) > 0:
        return None, "computed", mismatches

    exact_energy = _resolve_exact_energy_from_payload(adapt_ref_meta)
    if exact_energy is None:
        return None, "computed", []
    return float(exact_energy), "adapt_ref_json", []


# ============================================================================
# ADAPT-VQE core — standard algorithm, no meta-learner
# ============================================================================

@dataclass
class AdaptVQEResult:
    """Result container for the hardcoded ADAPT-VQE run."""
    energy: float
    theta: np.ndarray
    selected_ops: list[AnsatzTerm]
    history: list[dict[str, Any]]
    stop_reason: str
    nfev_total: int


def _build_uccsd_pool(
    num_sites: int,
    num_particles: tuple[int, int],
    ordering: str,
) -> list[AnsatzTerm]:
    """Build the UCCSD operator pool using HardcodedUCCSDAnsatz.base_terms.

    This reuses the exact same excitation generators the VQE pipeline uses,
    ensuring apples-to-apples comparison with the Qiskit UCCSD pool.
    """
    dummy_ansatz = HardcodedUCCSDAnsatz(
        dims=int(num_sites),
        num_particles=num_particles,
        reps=1,
        repr_mode="JW",
        indexing=str(ordering),
        include_singles=True,
        include_doubles=True,
    )
    return list(dummy_ansatz.base_terms)


def _build_cse_pool(
    num_sites: int,
    ordering: str,
    t: float,
    u: float,
    dv: float,
    boundary: str,
) -> list[AnsatzTerm]:
    """Build a CSE-style pool from the term-wise Hubbard ansatz base terms."""
    dummy_ansatz = HubbardTermwiseAnsatz(
        dims=int(num_sites),
        t=float(t),
        U=float(u),
        v=float(dv),
        reps=1,
        repr_mode="JW",
        indexing=str(ordering),
        pbc=(str(boundary).strip().lower() == "periodic"),
        include_potential_terms=True,
    )
    return list(dummy_ansatz.base_terms)


def _build_full_hamiltonian_pool(
    h_poly: Any,
    tol: float = 1e-12,
    normalize_coeff: bool = False,
) -> list[AnsatzTerm]:
    """Build a pool with one generator per non-identity Hamiltonian Pauli term."""
    pool: list[AnsatzTerm] = []
    terms = h_poly.return_polynomial()
    if not terms:
        return pool
    nq = int(terms[0].nqubit())
    id_label = "e" * nq

    for term in terms:
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if label == id_label:
            continue
        if abs(coeff) <= tol:
            continue
        if abs(coeff.imag) > tol:
            raise ValueError(
                f"Non-negligible imaginary Hamiltonian coefficient for term {label}: {coeff}"
            )
        generator = PauliPolynomial("JW")
        term_coeff = 1.0 if bool(normalize_coeff) else float(coeff.real)
        label_prefix = "ham_unit_term" if bool(normalize_coeff) else "ham_term"
        generator.add_term(PauliTerm(nq, ps=label, pc=float(term_coeff)))
        pool.append(AnsatzTerm(label=f"{label_prefix}({label})", polynomial=generator))
    return pool


def _polynomial_signature(poly: Any, tol: float = 1e-12) -> tuple[tuple[str, float], ...]:
    """Canonical real-valued signature for deduplicating PauliPolynomial generators."""
    items: list[tuple[str, float]] = []
    for term in poly.return_polynomial():
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) <= tol:
            continue
        if abs(coeff.imag) > tol:
            raise ValueError(f"Non-negligible imaginary coefficient in pool polynomial: {coeff} ({label})")
        items.append((label, round(float(coeff.real), 12)))
    items.sort()
    return tuple(items)


def _build_hh_termwise_augmented_pool(h_poly: Any, tol: float = 1e-12) -> list[AnsatzTerm]:
    """HH-only termwise pool: unit-normalized Hamiltonian terms + x->y quadrature partners."""
    base_pool = _build_full_hamiltonian_pool(h_poly, tol=tol, normalize_coeff=True)
    if not base_pool:
        return []

    terms = h_poly.return_polynomial()
    nq = int(terms[0].nqubit())
    id_label = "e" * nq

    seen_labels: set[str] = set()
    for op in base_pool:
        op_terms = op.polynomial.return_polynomial()
        if not op_terms:
            continue
        seen_labels.add(str(op_terms[0].pw2strng()))

    aug_pool = list(base_pool)
    for term in terms:
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if label == id_label or abs(coeff) <= tol:
            continue
        if "x" not in label:
            continue
        y_label = label.replace("x", "y")
        if y_label in seen_labels:
            continue
        gen = PauliPolynomial("JW")
        # Keep quadrature partners physically scaled to avoid over-dominating early ADAPT steps.
        y_coeff = abs(float(coeff.real))
        if y_coeff <= tol:
            y_coeff = 1.0
        gen.add_term(PauliTerm(nq, ps=y_label, pc=y_coeff))
        aug_pool.append(AnsatzTerm(label=f"ham_quadrature_term({y_label})", polynomial=gen))
        seen_labels.add(y_label)
    return aug_pool


def _build_hva_pool(
    num_sites: int,
    t: float,
    u: float,
    omega0: float,
    g_ep: float,
    dv: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
) -> list[AnsatzTerm]:
    # 1) Preserve the original HH layerwise generators used by the HVA form.
    layerwise = HubbardHolsteinLayerwiseAnsatz(
        dims=int(num_sites),
        J=float(t),
        U=float(u),
        omega0=float(omega0),
        g=float(g_ep),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        v=None,
        v_t=None,
        v0=float(dv),
        t_eval=None,
        reps=1,
        repr_mode="JW",
        indexing=str(ordering),
        pbc=(str(boundary).strip().lower() == "periodic"),
        include_zero_point=True,
    )
    pool: list[AnsatzTerm] = list(layerwise.base_terms)

    # 2) Augment with the same lifted HH UCCSD macro generators used by the
    # warm-start path. Do not expose individual Pauli fragments here: they can
    # break the fermion sector even when the parent macro generator preserves it.
    n_sites = int(num_sites)
    pool.extend(
        _build_hh_uccsd_fermion_lifted_pool(
            int(num_sites),
            int(n_ph_max),
            str(boson_encoding),
            str(ordering),
            str(boundary),
            num_particles=tuple(half_filled_num_particles(n_sites)),
        )
    )

    return pool


def _build_hh_uccsd_fermion_lifted_pool(
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    num_particles: tuple[int, int] | None = None,
) -> list[AnsatzTerm]:
    """HH-only UCCSD pool lifted into full HH register with boson identity prefix."""
    n_sites = int(num_sites)
    num_particles_eff = tuple(num_particles) if num_particles is not None else tuple(half_filled_num_particles(n_sites))
    ferm_nq = 2 * n_sites
    boson_bits = n_sites * int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    nq_total = ferm_nq + boson_bits

    uccsd_kwargs = {
        "dims": n_sites,
        "num_particles": num_particles_eff,
        "include_singles": True,
        "include_doubles": True,
        "repr_mode": "JW",
        "indexing": str(ordering),
    }
    if str(boundary).strip().lower() == "periodic":
        try:
            uccsd_kwargs["pbc"] = True
            uccsd = HardcodedUCCSDAnsatz(**uccsd_kwargs)
        except TypeError as exc:
            if "pbc" not in str(exc):
                raise
            uccsd_kwargs.pop("pbc", None)
            uccsd = HardcodedUCCSDAnsatz(**uccsd_kwargs)
    else:
        uccsd = HardcodedUCCSDAnsatz(**uccsd_kwargs)

    lifted_pool: list[AnsatzTerm] = []
    for op in uccsd.base_terms:
        lifted = PauliPolynomial("JW")
        for term in op.polynomial.return_polynomial():
            coeff = complex(term.p_coeff)
            if abs(coeff) <= 1e-15:
                continue
            if abs(coeff.imag) > 1e-12:
                raise ValueError(f"Non-negligible imaginary UCCSD coefficient in {op.label}: {coeff}")
            ferm_ps = str(term.pw2strng())
            if len(ferm_ps) != ferm_nq:
                raise ValueError(
                    f"Unexpected fermion Pauli length {len(ferm_ps)} != {ferm_nq} for UCCSD operator {op.label}"
                )
            full_ps = ("e" * boson_bits) + ferm_ps
            lifted.add_term(PauliTerm(nq_total, ps=full_ps, pc=float(coeff.real)))
        if len(lifted.return_polynomial()) == 0:
            continue
        lifted_pool.append(AnsatzTerm(label=f"uccsd_ferm_lifted::{op.label}", polynomial=lifted))
    return lifted_pool


def _build_paop_pool(
    num_sites: int,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    pool_key: str,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    num_particles: tuple[int, int],
) -> list[AnsatzTerm]:
    if make_paop_pool is None:
        raise RuntimeError(f"PAOP pool requested but operator_pools module unavailable: {_PAOP_IMPORT_ERROR}")

    pool_specs = make_paop_pool(
        pool_key,
        num_sites=int(num_sites),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        ordering=str(ordering),
        boundary=str(boundary),
        paop_r=int(paop_r),
        paop_split_paulis=bool(paop_split_paulis),
        paop_prune_eps=float(paop_prune_eps),
        paop_normalization=str(paop_normalization),
        num_particles=tuple(num_particles),
    )
    return [AnsatzTerm(label=label, polynomial=poly) for label, poly in pool_specs]


def _deduplicate_pool_terms(pool: list[AnsatzTerm]) -> list[AnsatzTerm]:
    """Deduplicate pool operators by canonical polynomial signature."""
    seen: set[tuple[tuple[str, float], ...]] = set()
    dedup_pool: list[AnsatzTerm] = []
    for term in pool:
        sig = _polynomial_signature(term.polynomial)
        if sig in seen:
            continue
        seen.add(sig)
        dedup_pool.append(term)
    return dedup_pool


def _polynomial_signature_digest(poly: Any, tol: float = 1e-12) -> str:
    """Low-memory ordered polynomial signature digest."""
    h = hashlib.sha1()
    for term in poly.return_polynomial():
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"Non-negligible imaginary coefficient in pool term: {coeff}")
        label = str(term.pw2strng())
        coeff_real = round(float(coeff.real), 12)
        h.update(label.encode("ascii", errors="ignore"))
        h.update(b":")
        h.update(f"{coeff_real:+.12e}".encode("ascii"))
        h.update(b";")
    return h.hexdigest()


def _deduplicate_pool_terms_lightweight(pool: list[AnsatzTerm]) -> list[AnsatzTerm]:
    """Deduplicate pool operators with a streaming digest to reduce peak memory."""
    seen: set[str] = set()
    dedup_pool: list[AnsatzTerm] = []
    for term in pool:
        sig = _polynomial_signature_digest(term.polynomial)
        if sig in seen:
            continue
        seen.add(sig)
        dedup_pool.append(term)
    return dedup_pool


def _build_hh_full_meta_pool(
    *,
    h_poly: Any,
    num_sites: int,
    t: float,
    u: float,
    omega0: float,
    g_ep: float,
    dv: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    num_particles: tuple[int, int],
) -> tuple[list[AnsatzTerm], dict[str, int]]:
    """Build HH full meta-pool: uccsd_lifted + hva + paop_full + paop_lf_full."""
    uccsd_lifted_pool = _build_hh_uccsd_fermion_lifted_pool(
        int(num_sites),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
        num_particles=num_particles,
    )
    hva_pool = _build_hva_pool(
        int(num_sites),
        float(t),
        float(u),
        float(omega0),
        float(g_ep),
        float(dv),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
    )
    termwise_aug: list[AnsatzTerm] = []
    if abs(float(g_ep)) > 1e-15:
        termwise_aug = [
            AnsatzTerm(label=f"hh_termwise_{term.label}", polynomial=term.polynomial)
            for term in _build_hh_termwise_augmented_pool(h_poly)
        ]
    paop_full_pool = _build_paop_pool(
        int(num_sites),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
        "paop_full",
        int(paop_r),
        bool(paop_split_paulis),
        float(paop_prune_eps),
        str(paop_normalization),
        num_particles,
    )
    paop_lf_full_pool = _build_paop_pool(
        int(num_sites),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
        "paop_lf_full",
        int(paop_r),
        bool(paop_split_paulis),
        float(paop_prune_eps),
        str(paop_normalization),
        num_particles,
    )
    merged = (
        list(uccsd_lifted_pool)
        + list(hva_pool)
        + list(termwise_aug)
        + list(paop_full_pool)
        + list(paop_lf_full_pool)
    )
    meta = {
        "raw_uccsd_lifted": int(len(uccsd_lifted_pool)),
        "raw_hva": int(len(hva_pool)),
        "raw_hh_termwise_augmented": int(len(termwise_aug)),
        "raw_paop_full": int(len(paop_full_pool)),
        "raw_paop_lf_full": int(len(paop_lf_full_pool)),
        "raw_total": int(len(merged)),
    }
    # n_ph_max>=2 can create very large term signatures; use streaming digest
    # dedup to avoid high transient memory spikes from tuple materialization.
    if int(n_ph_max) >= 2:
        dedup_pool = _deduplicate_pool_terms_lightweight(merged)
    else:
        dedup_pool = _deduplicate_pool_terms(merged)
    return dedup_pool, meta


# ---------------------------------------------------------------------------
# Pareto-lean pool: pruned subset of full_meta informed by motif analysis.
# Retains only operator classes that contributed energy improvement in the
# best-yet heavy scaffold run.  See artifacts/reports/hh_heavy_scaffold_best_yet_20260321.md
# ---------------------------------------------------------------------------

_PARETO_LEAN_PAOP_FULL_KEEP = {"paop_cloud_p", "paop_disp", "paop_hopdrag"}
_PARETO_LEAN_PAOP_LF_KEEP = {"paop_dbl_p"}


def _pareto_lean_paop_match(label: str, allowed: set[str]) -> bool:
    """True if the PAOP label's family name is in *allowed*."""
    colon_idx = label.find(":")
    if colon_idx < 0:
        return False
    after_colon = label[colon_idx + 1:]
    for family in allowed:
        if after_colon.startswith(family + "("):
            return True
    return False


def _build_hh_pareto_lean_pool(
    *,
    h_poly: Any,
    num_sites: int,
    t: float,
    u: float,
    omega0: float,
    g_ep: float,
    dv: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    num_particles: tuple[int, int],
) -> tuple[list[AnsatzTerm], dict[str, int]]:
    """Build the Pareto-lean HH pool: only classes that survived the heavy scaffold.

    Kept classes:
      - uccsd_sing, uccsd_dbl  (all lifted UCCSD)
      - hh_termwise_quadrature (y-partner quadrature terms only, no unit terms)
      - paop_cloud_p, paop_disp, paop_hopdrag  (from paop_full)
      - paop_dbl_p  (from paop_lf_full)

    Dropped classes:
      - HVA layerwise macros  (hop_layer, onsite_layer, phonon_layer, eph_layer)
      - hh_termwise_unit       (diagonal Ham terms, never selected)
      - paop_cloud_x, paop_dbl, paop_lf_dbl_x, paop_lf_curdrag, paop_lf_hop2
    """
    # 1. Lifted UCCSD (all singles + doubles)
    uccsd_lifted_pool = _build_hh_uccsd_fermion_lifted_pool(
        int(num_sites),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
        num_particles=num_particles,
    )

    # 2. Termwise quadrature only (skip unit terms)
    quadrature_pool: list[AnsatzTerm] = []
    if abs(float(g_ep)) > 1e-15:
        termwise_aug = _build_hh_termwise_augmented_pool(h_poly)
        quadrature_pool = [
            AnsatzTerm(label=f"hh_termwise_{term.label}", polynomial=term.polynomial)
            for term in termwise_aug
            if "quadrature" in term.label
        ]

    # 3. PAOP full, filtered to cloud_p + disp + hopdrag
    paop_full_raw = _build_paop_pool(
        int(num_sites),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
        "paop_full",
        int(paop_r),
        bool(paop_split_paulis),
        float(paop_prune_eps),
        str(paop_normalization),
        num_particles,
    )
    paop_full_kept = [
        t for t in paop_full_raw
        if _pareto_lean_paop_match(t.label, _PARETO_LEAN_PAOP_FULL_KEEP)
    ]

    # 4. PAOP lf_full, filtered to dbl_p only
    paop_lf_raw = _build_paop_pool(
        int(num_sites),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
        "paop_lf_full",
        int(paop_r),
        bool(paop_split_paulis),
        float(paop_prune_eps),
        str(paop_normalization),
        num_particles,
    )
    paop_lf_kept = [
        t for t in paop_lf_raw
        if _pareto_lean_paop_match(t.label, _PARETO_LEAN_PAOP_LF_KEEP)
    ]

    merged = (
        list(uccsd_lifted_pool)
        + list(quadrature_pool)
        + list(paop_full_kept)
        + list(paop_lf_kept)
    )
    meta = {
        "raw_uccsd_lifted": int(len(uccsd_lifted_pool)),
        "raw_hh_termwise_quadrature": int(len(quadrature_pool)),
        "raw_paop_full_kept": int(len(paop_full_kept)),
        "raw_paop_full_dropped": int(len(paop_full_raw) - len(paop_full_kept)),
        "raw_paop_lf_kept": int(len(paop_lf_kept)),
        "raw_paop_lf_dropped": int(len(paop_lf_raw) - len(paop_lf_kept)),
        "raw_total": int(len(merged)),
    }
    if int(n_ph_max) >= 2:
        dedup_pool = _deduplicate_pool_terms_lightweight(merged)
    else:
        dedup_pool = _deduplicate_pool_terms(merged)
    return dedup_pool, meta


# ---------------------------------------------------------------------------
# Pareto-lean L2 pool: tighter pruning from L=2 n_ph_max=1 motif analysis.
# Drops disp, uccsd_dbl, and unused dbl_p site-phonon pairings.
# See artifacts/reports/hh_L2_ecut1_scaffold_motif_analysis.md
# ---------------------------------------------------------------------------

_PARETO_LEAN_L2_PAOP_FULL_KEEP = {"paop_cloud_p", "paop_hopdrag"}
_PARETO_LEAN_L2_PAOP_LF_KEEP = {"paop_dbl_p"}

# Only the site->phonon pairings that were actually selected at L=2.
# site=0->phonon=1 and site=1->phonon=0 were used; the others were not.
_PARETO_LEAN_L2_DPL_P_KEEP_SUFFIXES = {"site=0->phonon=1", "site=1->phonon=0"}


def _build_hh_pareto_lean_l2_pool(
    *,
    h_poly: Any,
    num_sites: int,
    t: float,
    u: float,
    omega0: float,
    g_ep: float,
    dv: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    num_particles: tuple[int, int],
) -> tuple[list[AnsatzTerm], dict[str, int]]:
    """Build the L=2-specific Pareto-lean pool (11 operators).

    Kept: quadrature(4), uccsd_sing(all), cloud_p(all), hopdrag(all),
          dbl_p(site=0->ph=1, site=1->ph=0 only).
    Dropped: uccsd_dbl, disp, dbl (bare), all x-type, curdrag, hop2,
             HVA layers, unit terms, unused dbl_p variants.
    """
    if int(num_sites) != 2:
        raise ValueError("adapt_pool='pareto_lean_l2' is only valid for L=2.")
    if int(n_ph_max) != 1:
        raise ValueError("adapt_pool='pareto_lean_l2' is only valid for n_ph_max=1.")
    # 1. UCCSD lifted — singles only
    all_uccsd = _build_hh_uccsd_fermion_lifted_pool(
        int(num_sites),
        int(n_ph_max),
        str(boson_encoding),
        str(ordering),
        str(boundary),
        num_particles=num_particles,
    )
    uccsd_singles = [t for t in all_uccsd if "uccsd_sing" in t.label]

    # 2. Quadrature only
    quadrature_pool: list[AnsatzTerm] = []
    if abs(float(g_ep)) > 1e-15:
        termwise_aug = _build_hh_termwise_augmented_pool(h_poly)
        quadrature_pool = [
            AnsatzTerm(label=f"hh_termwise_{term.label}", polynomial=term.polynomial)
            for term in termwise_aug
            if "quadrature" in term.label
        ]

    # 3. PAOP full: cloud_p + hopdrag only (no disp)
    paop_full_raw = _build_paop_pool(
        int(num_sites), int(n_ph_max), str(boson_encoding), str(ordering),
        str(boundary), "paop_full", int(paop_r), bool(paop_split_paulis),
        float(paop_prune_eps), str(paop_normalization), num_particles,
    )
    paop_full_kept = [
        t for t in paop_full_raw
        if _pareto_lean_paop_match(t.label, _PARETO_LEAN_L2_PAOP_FULL_KEEP)
    ]

    # 4. PAOP lf_full: dbl_p, filtered to used site->phonon pairings
    paop_lf_raw = _build_paop_pool(
        int(num_sites), int(n_ph_max), str(boson_encoding), str(ordering),
        str(boundary), "paop_lf_full", int(paop_r), bool(paop_split_paulis),
        float(paop_prune_eps), str(paop_normalization), num_particles,
    )
    paop_lf_kept = []
    for t in paop_lf_raw:
        if not _pareto_lean_paop_match(t.label, _PARETO_LEAN_L2_PAOP_LF_KEEP):
            continue
        # Further filter to only the used site->phonon pairings
        if any(suffix in t.label for suffix in _PARETO_LEAN_L2_DPL_P_KEEP_SUFFIXES):
            paop_lf_kept.append(t)

    merged = (
        list(uccsd_singles)
        + list(quadrature_pool)
        + list(paop_full_kept)
        + list(paop_lf_kept)
    )
    meta = {
        "raw_uccsd_singles": int(len(uccsd_singles)),
        "raw_hh_termwise_quadrature": int(len(quadrature_pool)),
        "raw_paop_full_kept": int(len(paop_full_kept)),
        "raw_paop_full_dropped": int(len(paop_full_raw) - len(paop_full_kept)),
        "raw_paop_lf_kept": int(len(paop_lf_kept)),
        "raw_paop_lf_dropped": int(len(paop_lf_raw) - len(paop_lf_kept)),
        "raw_total": int(len(merged)),
    }
    if int(n_ph_max) >= 2:
        dedup_pool = _deduplicate_pool_terms_lightweight(merged)
    else:
        dedup_pool = _deduplicate_pool_terms(merged)
    return dedup_pool, meta


# ---------------------------------------------------------------------------
# Gate-pruned pool: informed by term-level leave-one-out analysis.
# For operators where one Pauli term suffices, replace the multi-term
# polynomial with the single dominant term.  This halves the 2Q gate count
# for uccsd_sing and paop_hopdrag at zero energy regression.
# See artifacts/reports/hh_prune_nighthawk_pareto_menu_20260322.md
# ---------------------------------------------------------------------------

# Mapping: operator label pattern -> list of Pauli strings to KEEP.
# If an operator label matches, its polynomial is replaced with only the
# listed Pauli terms (preserving original coefficients).
# None means keep the original (no pruning).
_GATE_PRUNE_TERM_KEEP: dict[str, list[str] | None] = {
    "uccsd_sing(alpha:0->1)": ["eeeexy"],   # drop eeeeyx (reg ~0)
    "uccsd_sing(beta:2->3)":  ["eeyxee"],   # drop eexyee (reg ~0)
    "paop_hopdrag":           ["yeyyee"],    # drop yexxee (reg ~0)
}


def _gate_prune_polynomial(
    label: str,
    poly: Any,
) -> Any:
    """If the operator matches a gate-prune rule, return a trimmed polynomial."""
    for pattern, keep_paulis in _GATE_PRUNE_TERM_KEEP.items():
        if keep_paulis is None:
            continue
        if pattern in label:
            terms = poly.return_polynomial()
            if not terms:
                return poly
            nq = int(terms[0].nqubit())
            keep_set = set(keep_paulis)
            kept_terms = [t for t in terms if str(t.pw2strng()) in keep_set]
            if not kept_terms:
                return poly  # safety: if nothing matches, keep original
            pruned = PauliPolynomial("JW", [
                PauliTerm(nq, ps=str(t.pw2strng()), pc=float(t.p_coeff))
                for t in kept_terms
            ])
            return pruned
    return poly


def _build_hh_pareto_lean_gate_pruned_pool(
    *,
    h_poly: Any,
    num_sites: int,
    t: float,
    u: float,
    omega0: float,
    g_ep: float,
    dv: float,
    n_ph_max: int,
    boson_encoding: str,
    ordering: str,
    boundary: str,
    paop_r: int,
    paop_split_paulis: bool,
    paop_prune_eps: float,
    paop_normalization: str,
    num_particles: tuple[int, int],
) -> tuple[list[AnsatzTerm], dict[str, int]]:
    """Pareto-lean pool with gate-level term pruning.

    Starts from pareto_lean, then replaces multi-term operators with single-term
    variants where term-level leave-one-out showed zero regression:
      - uccsd_sing(alpha:0->1): keep only eeeexy (drop eeeeyx)
      - uccsd_sing(beta:2->3):  keep only eeyxee (drop eexyee)
      - paop_hopdrag(*):        keep only yeyyee (drop yexxee)

    All other operators (paop_dbl_p, paop_disp, quadrature) are unchanged.
    """
    # Build the base pareto_lean pool
    base_pool, base_meta = _build_hh_pareto_lean_pool(
        h_poly=h_poly,
        num_sites=int(num_sites),
        t=float(t),
        u=float(u),
        omega0=float(omega0),
        g_ep=float(g_ep),
        dv=float(dv),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        ordering=str(ordering),
        boundary=str(boundary),
        paop_r=int(paop_r),
        paop_split_paulis=bool(paop_split_paulis),
        paop_prune_eps=float(paop_prune_eps),
        paop_normalization=str(paop_normalization),
        num_particles=num_particles,
    )

    # Apply gate-level term pruning
    pruned_pool: list[AnsatzTerm] = []
    n_pruned = 0
    for term in base_pool:
        pruned_poly = _gate_prune_polynomial(term.label, term.polynomial)
        if pruned_poly is not term.polynomial:
            n_pruned += 1
        pruned_pool.append(AnsatzTerm(label=term.label, polynomial=pruned_poly))

    meta = dict(base_meta)
    meta["gate_pruned_operators"] = int(n_pruned)
    meta["gate_prune_rules"] = {k: v for k, v in _GATE_PRUNE_TERM_KEEP.items() if v is not None}
    return pruned_pool, meta


def _apply_pauli_polynomial_uncached(state: np.ndarray, poly: Any) -> np.ndarray:
    r"""Compute G|psi> where G is a PauliPolynomial (sum of weighted Pauli strings).

    G = \sum_j c_j P_j   =>   G|psi> = \sum_j c_j P_j|psi>
    """
    terms = poly.return_polynomial()
    if not terms:
        return np.zeros_like(state)
    nq = int(terms[0].nqubit())
    id_str = "e" * nq
    result = np.zeros_like(state)
    for term in terms:
        ps = term.pw2strng()
        coeff = complex(term.p_coeff)
        if abs(coeff) < 1e-15:
            continue
        if ps == id_str:
            result += coeff * state
        else:
            result += coeff * apply_pauli_string(state, ps)
    return result


def _apply_pauli_polynomial(
    state: np.ndarray,
    poly: Any,
    *,
    compiled: CompiledPolynomialAction | None = None,
) -> np.ndarray:
    if compiled is not None:
        return _apply_compiled_polynomial(state, compiled)
    return _apply_pauli_polynomial_uncached(state, poly)


def _commutator_gradient(
    h_poly: Any,
    pool_op: AnsatzTerm,
    psi_current: np.ndarray,
    *,
    h_compiled: CompiledPolynomialAction | None = None,
    pool_compiled: CompiledPolynomialAction | None = None,
    hpsi_precomputed: np.ndarray | None = None,
) -> float:
    r"""Compute dE/dtheta at theta=0 for appending pool_op to the current state.

    E(theta) = <psi|exp(+i theta G) H exp(-i theta G)|psi>

    The analytic gradient at theta=0 is:
        dE/dtheta|_0 = i <psi|[H, G]|psi> = 2 Im(<psi|H G|psi>)

    Since H is Hermitian: <psi|H G|psi> = <H psi | G psi>.

    This is exact and works for multi-term PauliPolynomial generators
    (unlike the parameter-shift rule which requires single-Pauli generators).
    """
    g_psi = _apply_pauli_polynomial(psi_current, pool_op.polynomial, compiled=pool_compiled)
    h_psi = (
        np.asarray(hpsi_precomputed, dtype=complex)
        if hpsi_precomputed is not None
        else _apply_pauli_polynomial(psi_current, h_poly, compiled=h_compiled)
    )
    return adapt_commutator_grad_from_hpsi(h_psi, g_psi)


def _prepare_adapt_state(
    psi_ref: np.ndarray,
    selected_ops: list[AnsatzTerm],
    theta: np.ndarray,
    *,
    parameter_layout: AnsatzParameterLayout | None = None,
) -> np.ndarray:
    """Apply the current ADAPT ansatz.

    Supports both legacy logical-shared theta and per-Pauli runtime theta.
    """
    psi = np.array(psi_ref, copy=True)
    if not selected_ops:
        return psi
    theta_arr = np.asarray(theta, dtype=float).reshape(-1)
    layout = (
        parameter_layout
        if parameter_layout is not None
        else build_parameter_layout(selected_ops, ignore_identity=True, coefficient_tolerance=1e-12, sort_terms=True)
    )
    if int(theta_arr.size) == int(layout.runtime_parameter_count):
        for block, op in zip(layout.blocks, selected_ops):
            if int(block.runtime_count) <= 0:
                continue
            psi = apply_exp_pauli_polynomial_termwise(
                psi,
                op.polynomial,
                theta_arr[block.runtime_start:block.runtime_stop],
                ignore_identity=bool(layout.ignore_identity),
                coefficient_tolerance=float(layout.coefficient_tolerance),
                sort_terms=(str(layout.term_order) == "sorted"),
            )
        return psi
    if int(theta_arr.size) == int(len(selected_ops)):
        for k, op in enumerate(selected_ops):
            psi = apply_exp_pauli_polynomial(psi, op.polynomial, float(theta_arr[k]))
        return psi
    raise ValueError(
        f"ADAPT theta length mismatch: got {theta_arr.size}, expected {layout.runtime_parameter_count} (runtime) or {len(selected_ops)} (logical)."
    )


def _logical_theta_alias(theta: np.ndarray, layout: AnsatzParameterLayout) -> np.ndarray:
    theta_arr = np.asarray(theta, dtype=float).reshape(-1)
    if int(theta_arr.size) == int(layout.runtime_parameter_count):
        return np.asarray(project_runtime_theta_block_mean(theta_arr, layout), dtype=float)
    if int(theta_arr.size) == int(layout.logical_parameter_count):
        return np.asarray(theta_arr, dtype=float)
    raise ValueError(
        f"Cannot project theta of length {theta_arr.size} onto logical blocks of size {layout.logical_parameter_count}."
    )


def _adapt_energy_fn(
    h_poly: Any,
    psi_ref: np.ndarray,
    selected_ops: list[AnsatzTerm],
    theta: np.ndarray,
    *,
    h_compiled: CompiledPolynomialAction | None = None,
    parameter_layout: AnsatzParameterLayout | None = None,
) -> float:
    """Energy of the current ADAPT ansatz at parameters theta."""
    psi = _prepare_adapt_state(psi_ref, selected_ops, theta, parameter_layout=parameter_layout)
    if h_compiled is not None:
        energy, _hpsi = energy_via_one_apply(psi, h_compiled)
        return float(energy)
    return float(expval_pauli_polynomial(psi, h_poly))


def _exact_gs_energy_for_problem(
    h_poly: Any,
    *,
    problem: str,
    num_sites: int,
    num_particles: tuple[int, int],
    indexing: str,
    n_ph_max: int = 1,
    boson_encoding: str = "binary",
    t: float | None = None,
    u: float | None = None,
    dv: float | None = None,
    omega0: float | None = None,
    g_ep: float | None = None,
    boundary: str = "open",
) -> float:
    """Dispatch to the correct sector-filtered exact ground energy.

    For problem='hh', use fermion-only sector filtering (phonon qubits free).
    For problem='hubbard', use standard full-register sector filtering.
    """
    if str(problem).strip().lower() == "hh":
        if (
            t is not None
            and u is not None
            and dv is not None
            and omega0 is not None
            and g_ep is not None
        ):
            try:
                from src.quantum.ed_hubbard_holstein import build_hh_sector_hamiltonian_ed

                h_sector = build_hh_sector_hamiltonian_ed(
                    dims=int(num_sites),
                    J=float(t),
                    U=float(u),
                    omega0=float(omega0),
                    g=float(g_ep),
                    n_ph_max=int(n_ph_max),
                    num_particles=tuple(num_particles),
                    indexing=str(indexing),
                    boson_encoding=str(boson_encoding),
                    pbc=(str(boundary).strip().lower() == "periodic"),
                    delta_v=float(dv),
                    include_zero_point=True,
                    sparse=True,
                    return_basis=False,
                )
                try:
                    from scipy.sparse import spmatrix as _spmatrix
                    from scipy.sparse.linalg import eigsh as _eigsh

                    if isinstance(h_sector, _spmatrix):
                        eval0 = _eigsh(
                            h_sector,
                            k=1,
                            which="SA",
                            return_eigenvectors=False,
                            tol=1e-10,
                            maxiter=max(1000, 10 * int(h_sector.shape[0])),
                        )
                        return float(np.real(eval0[0]))
                except Exception:
                    pass

                h_dense = np.asarray(
                    h_sector.toarray() if hasattr(h_sector, "toarray") else h_sector,
                    dtype=complex,
                )
                evals = np.linalg.eigvalsh(h_dense)
                return float(np.min(np.real(evals)))
            except Exception as exc:
                _ai_log(
                    "hardcoded_adapt_hh_exact_sparse_fallback",
                    status="failed",
                    error=str(exc),
                )
        return exact_ground_energy_sector_hh(
            h_poly,
            num_sites=int(num_sites),
            num_particles=num_particles,
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            indexing=str(indexing),
        )
    else:
        return exact_ground_energy_sector(
            h_poly,
            num_sites=int(num_sites),
            num_particles=num_particles,
            indexing=str(indexing),
        )


def _exact_reference_state_for_hh(
    *,
    num_sites: int,
    num_particles: tuple[int, int],
    indexing: str,
    n_ph_max: int,
    boson_encoding: str,
    t: float,
    u: float,
    dv: float,
    omega0: float,
    g_ep: float,
    boundary: str,
) -> np.ndarray | None:
    try:
        from src.quantum.ed_hubbard_holstein import build_hh_sector_hamiltonian_ed
        from scipy.sparse import spmatrix as _spmatrix
        from scipy.sparse.linalg import eigsh as _eigsh

        h_sector, basis = build_hh_sector_hamiltonian_ed(
            dims=int(num_sites),
            J=float(t),
            U=float(u),
            omega0=float(omega0),
            g=float(g_ep),
            n_ph_max=int(n_ph_max),
            num_particles=tuple(num_particles),
            indexing=str(indexing),
            boson_encoding=str(boson_encoding),
            pbc=(str(boundary).strip().lower() == "periodic"),
            delta_v=float(dv),
            include_zero_point=True,
            sparse=True,
            return_basis=True,
        )
        if isinstance(h_sector, _spmatrix):
            evals, evecs = _eigsh(
                h_sector,
                k=1,
                which="SA",
                return_eigenvectors=True,
                tol=1e-10,
                maxiter=max(1000, 10 * int(h_sector.shape[0])),
            )
            vec_sector = np.asarray(evecs[:, 0], dtype=complex).reshape(-1)
        else:
            dense = np.asarray(h_sector, dtype=complex)
            evals, evecs = np.linalg.eigh(dense)
            vec_sector = np.asarray(evecs[:, int(np.argmin(np.real(evals)))], dtype=complex).reshape(-1)
        psi_full = np.zeros(1 << int(basis.total_qubits), dtype=complex)
        for local_idx, basis_idx in enumerate(basis.basis_indices):
            psi_full[int(basis_idx)] = complex(vec_sector[int(local_idx)])
        return _normalize_state(psi_full)
    except Exception as exc:
        _ai_log(
            "hardcoded_adapt_exact_reference_state_unavailable",
            error=str(exc),
        )
        return None


# ---------------------------------------------------------------------------
# Windowed reopt helpers (pure, deterministic)
# ---------------------------------------------------------------------------

_VALID_REOPT_POLICIES = {"append_only", "full", "windowed"}


def _resolve_reopt_active_indices(
    *,
    policy: str,
    n: int,
    theta: np.ndarray,
    window_size: int = 3,
    window_topk: int = 0,
    periodic_full_refit_triggered: bool = False,
) -> tuple[list[int], str]:
    """Return (sorted_active_indices, effective_policy_name).

    Active-index selection contract (windowed):
      1. w_eff = min(window_size, n)
      2. newest = [n - w_eff, ..., n - 1]
      3. older  = [0, ..., n - w_eff - 1]
      4. If window_topk > 0, rank older by descending |theta[i]|,
         break ties by ascending index i.
      5. k_eff = min(window_topk, len(older))
      6. active = union(newest, top-k older)
      7. return sorted ascending

    For append_only: active = [n - 1]
    For full or periodic full-refit override: active = [0 .. n-1]
    """
    policy_key = str(policy).strip().lower()
    if n <= 0:
        return [], policy_key

    if policy_key == "append_only":
        return [n - 1], "append_only"

    if policy_key == "full":
        return list(range(n)), "full"

    if policy_key != "windowed":
        raise ValueError(f"Unknown reopt policy '{policy_key}'.")

    # Periodic full-refit override for this depth
    if periodic_full_refit_triggered:
        return list(range(n)), "windowed_periodic_full"

    w_eff = min(int(window_size), n)
    newest = list(range(n - w_eff, n))

    older_start = n - w_eff
    if older_start <= 0 or int(window_topk) <= 0:
        return sorted(newest), "windowed"

    older_candidates = list(range(0, older_start))
    # Rank by descending |theta[i]|, tie-break ascending index i
    older_ranked = sorted(
        older_candidates,
        key=lambda i: (-abs(float(theta[i])), i),
    )
    k_eff = min(int(window_topk), len(older_ranked))
    selected_older = older_ranked[:k_eff]
    active = sorted(set(newest) | set(selected_older))
    return active, "windowed"


def _make_reduced_objective(
    full_theta: np.ndarray,
    active_indices: list[int],
    obj_fn: Any,
) -> tuple[Any, np.ndarray]:
    """Build a reduced-variable objective and its initial point.

    Returns (reduced_obj, x0_reduced) where:
      - reduced_obj(x_active) reconstructs a full theta from frozen+active
        and calls obj_fn(full_theta)
      - x0_reduced = full_theta[active_indices]
    """
    frozen_theta = np.array(full_theta, copy=True)
    active_idx = list(active_indices)
    n_active = len(active_idx)
    x0 = np.array([float(frozen_theta[i]) for i in active_idx], dtype=float)

    if n_active == len(frozen_theta):
        # Full prefix — no wrapping needed
        return obj_fn, np.array(frozen_theta, copy=True)

    def _reduced(x_active: np.ndarray) -> float:
        full = np.array(frozen_theta, copy=True)
        x_arr = np.asarray(x_active, dtype=float).ravel()
        for k, idx in enumerate(active_idx):
            full[idx] = float(x_arr[k])
        return float(obj_fn(full))

    return _reduced, x0


def _resolve_adapt_continuation_mode(*, problem: str, requested_mode: str | None) -> str:
    mode_raw = "legacy" if requested_mode is None else str(requested_mode).strip().lower()
    if mode_raw == "":
        return "legacy"
    if mode_raw not in {"legacy", "phase1_v1", "phase2_v1", "phase3_v1"}:
        raise ValueError("adapt_continuation_mode must be one of {'legacy','phase1_v1','phase2_v1','phase3_v1'}.")
    return str(mode_raw)


@dataclass(frozen=True)
class ResolvedAdaptStopPolicy:
    adapt_drop_floor: float
    adapt_drop_patience: int
    adapt_drop_min_depth: int
    adapt_grad_floor: float
    adapt_drop_floor_source: str
    adapt_drop_patience_source: str
    adapt_drop_min_depth_source: str
    adapt_grad_floor_source: str
    drop_policy_enabled: bool
    drop_policy_source: str
    eps_energy_termination_enabled: bool
    eps_grad_termination_enabled: bool


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


def _phase1_repeated_family_flat(
    *,
    history: list[dict[str, Any]],
    candidate_family: str,
    patience: int,
    weak_drop_threshold: float,
) -> bool:
    if str(candidate_family).strip() == "":
        return False
    tail = [row for row in history if isinstance(row, dict) and row.get("candidate_family") is not None]
    need = max(0, int(patience) - 1)
    if need <= 0:
        return False
    if len(tail) < need:
        return False
    recent = tail[-need:]
    for row in recent:
        if str(row.get("candidate_family")) != str(candidate_family):
            return False
        drop = float(row.get("delta_abs_drop_from_prev", float("inf")))
        if not math.isfinite(drop) or drop > float(weak_drop_threshold):
            return False
    return True


def _splice_candidate_at_position(
    *,
    ops: list[AnsatzTerm],
    theta: np.ndarray,
    op: AnsatzTerm,
    position_id: int,
    init_theta: float = 0.0,
) -> tuple[list[AnsatzTerm], np.ndarray]:
    current_layout = build_parameter_layout(
        ops,
        ignore_identity=True,
        coefficient_tolerance=1e-12,
        sort_terms=True,
    )
    op_layout = build_parameter_layout(
        [op],
        ignore_identity=True,
        coefficient_tolerance=1e-12,
        sort_terms=True,
    )
    pos_logical = max(0, min(int(current_layout.logical_parameter_count), int(position_id)))
    pos_runtime = int(runtime_insert_position(current_layout, pos_logical))
    new_ops = list(ops)
    new_ops.insert(pos_logical, op)
    theta_arr = np.asarray(theta, dtype=float).reshape(-1)
    insert_block = np.full(int(op_layout.runtime_parameter_count), float(init_theta), dtype=float)
    new_theta = np.insert(theta_arr, pos_runtime, insert_block)
    return new_ops, np.asarray(new_theta, dtype=float)


def _predict_reopt_window_for_position(
    *,
    theta: np.ndarray,
    position_id: int,
    policy: str,
    window_size: int,
    window_topk: int,
    periodic_full_refit_triggered: bool,
) -> list[int]:
    theta_arr = np.asarray(theta, dtype=float).reshape(-1)
    append_position = int(theta_arr.size)
    pos = max(0, min(int(append_position), int(position_id)))
    theta_hyp = np.insert(theta_arr, pos, 0.0)
    active, _name = _resolve_reopt_active_indices(
        policy=str(policy),
        n=int(theta_hyp.size),
        theta=np.asarray(theta_hyp, dtype=float),
        window_size=int(window_size),
        window_topk=int(window_topk),
        periodic_full_refit_triggered=bool(periodic_full_refit_triggered),
    )
    return [int(i) for i in active]


def _window_terms_for_position(
    *,
    selected_ops: list[AnsatzTerm],
    refit_window_indices: list[int],
    position_id: int,
) -> tuple[list[AnsatzTerm], list[str]]:
    window_terms: list[AnsatzTerm] = []
    window_labels: list[str] = []
    pos = int(position_id)
    for idx in refit_window_indices:
        j = int(idx)
        if j == pos:
            continue
        mapped = j if j < pos else j - 1
        if 0 <= int(mapped) < len(selected_ops):
            term = selected_ops[int(mapped)]
            window_terms.append(term)
            window_labels.append(str(term.label))
    return window_terms, window_labels


def _phase2_record_sort_key(record: Mapping[str, Any]) -> tuple[float, float, int, int]:
    return (
        -float(record.get("full_v2_score", float("-inf"))),
        -float(record.get("simple_score", float("-inf"))),
        int(record.get("candidate_pool_index", -1)),
        int(record.get("position_id", -1)),
    )


def _run_hardcoded_adapt_vqe(
    *,
    h_poly: Any,
    num_sites: int,
    ordering: str,
    problem: str,
    adapt_pool: str | None,
    t: float,
    u: float,
    dv: float,
    boundary: str,
    omega0: float,
    g_ep: float,
    n_ph_max: int,
    boson_encoding: str,
    max_depth: int,
    eps_grad: float,
    eps_energy: float,
    maxiter: int,
    seed: int,
    adapt_inner_optimizer: str = "SPSA",
    adapt_spsa_a: float = 0.2,
    adapt_spsa_c: float = 0.1,
    adapt_spsa_alpha: float = 0.602,
    adapt_spsa_gamma: float = 0.101,
    adapt_spsa_A: float = 10.0,
    adapt_spsa_avg_last: int = 0,
    adapt_spsa_eval_repeats: int = 1,
    adapt_spsa_eval_agg: str = "mean",
    adapt_spsa_callback_every: int = 1,
    adapt_spsa_progress_every_s: float = 60.0,
    allow_repeats: bool,
    finite_angle_fallback: bool,
    finite_angle: float,
    finite_angle_min_improvement: float,
    adapt_drop_floor: float | None = None,
    adapt_drop_patience: int | None = None,
    adapt_drop_min_depth: int | None = None,
    adapt_grad_floor: float | None = None,
    adapt_eps_energy_min_extra_depth: int = -1,
    adapt_eps_energy_patience: int = -1,
    adapt_ref_base_depth: int = 0,
    paop_r: int = 0,
    paop_split_paulis: bool = False,
    paop_prune_eps: float = 0.0,
    paop_normalization: str = "none",
    disable_hh_seed: bool = False,
    psi_ref_override: np.ndarray | None = None,
    adapt_gradient_parity_check: bool = False,
    adapt_state_backend: str = "compiled",
    adapt_reopt_policy: str = "append_only",
    adapt_window_size: int = 3,
    adapt_window_topk: int = 0,
    adapt_full_refit_every: int = 0,
    adapt_final_full_refit: bool = True,
    exact_gs_override: float | None = None,
    adapt_continuation_mode: str | None = "legacy",
    phase1_lambda_F: float = 1.0,
    phase1_lambda_compile: float = 0.05,
    phase1_lambda_measure: float = 0.02,
    phase1_lambda_leak: float = 0.0,
    phase1_score_z_alpha: float = 0.0,
    phase1_shortlist_size: int = 64,
    phase1_probe_max_positions: int = 6,
    phase1_plateau_patience: int = 2,
    phase1_trough_margin_ratio: float = 1.0,
    phase1_prune_enabled: bool = True,
    phase1_prune_fraction: float = 0.25,
    phase1_prune_max_candidates: int = 6,
    phase1_prune_max_regression: float = 1e-8,
    phase2_shortlist_fraction: float = 0.2,
    phase2_shortlist_size: int = 12,
    phase2_lambda_H: float = 1e-6,
    phase2_rho: float = 0.25,
    phase2_gamma_N: float = 1.0,
    phase2_enable_batching: bool = True,
    phase2_batch_target_size: int = 2,
    phase2_batch_size_cap: int = 3,
    phase2_batch_near_degenerate_ratio: float = 0.9,
    phase3_motif_source_json: Path | None = None,
    phase3_symmetry_mitigation_mode: str = "off",
    phase3_enable_rescue: bool = False,
    phase3_lifetime_cost_mode: str = "phase3_v1",
    phase3_runtime_split_mode: str = "off",
    phase3_backend_cost_mode: str = "proxy",
    phase3_backend_name: str | None = None,
    phase3_backend_shortlist: Sequence[str] | None = None,
    phase3_backend_transpile_seed: int = 7,
    phase3_backend_optimization_level: int = 1,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run standard ADAPT-VQE and return (payload, psi_ground)."""
    if float(finite_angle) <= 0.0:
        raise ValueError("finite_angle must be > 0.")
    if float(finite_angle_min_improvement) < 0.0:
        raise ValueError("finite_angle_min_improvement must be >= 0.")
    adapt_state_backend_key = str(adapt_state_backend).strip().lower()
    if adapt_state_backend_key not in {"legacy", "compiled"}:
        raise ValueError("adapt_state_backend must be one of {'legacy','compiled'}.")
    adapt_reopt_policy_key = str(adapt_reopt_policy).strip().lower()
    if adapt_reopt_policy_key not in _VALID_REOPT_POLICIES:
        raise ValueError(f"adapt_reopt_policy must be one of {_VALID_REOPT_POLICIES}.")
    adapt_window_size_val = int(adapt_window_size)
    adapt_window_topk_val = int(adapt_window_topk)
    adapt_full_refit_every_val = int(adapt_full_refit_every)
    adapt_final_full_refit_val = bool(adapt_final_full_refit)
    phase1_shortlist_size_val = int(phase1_shortlist_size)
    if adapt_window_size_val < 1:
        raise ValueError("adapt_window_size must be >= 1.")
    if adapt_window_topk_val < 0:
        raise ValueError("adapt_window_topk must be >= 0.")
    if adapt_full_refit_every_val < 0:
        raise ValueError("adapt_full_refit_every must be >= 0.")
    if phase1_shortlist_size_val < 1:
        raise ValueError("phase1_shortlist_size must be >= 1.")
    adapt_inner_optimizer_key = str(adapt_inner_optimizer).strip().upper()
    if adapt_inner_optimizer_key not in {"COBYLA", "POWELL", "SPSA"}:
        raise ValueError("adapt_inner_optimizer must be one of {'COBYLA','POWELL','SPSA'}.")
    adapt_spsa_eval_agg_key = str(adapt_spsa_eval_agg).strip().lower()
    if adapt_spsa_eval_agg_key not in {"mean", "median"}:
        raise ValueError("adapt_spsa_eval_agg must be one of {'mean','median'}.")
    if int(adapt_spsa_callback_every) < 1:
        raise ValueError("adapt_spsa_callback_every must be >= 1.")
    if float(adapt_spsa_progress_every_s) < 0.0:
        raise ValueError("adapt_spsa_progress_every_s must be >= 0.")
    if int(adapt_eps_energy_min_extra_depth) < -1:
        raise ValueError("adapt_eps_energy_min_extra_depth must be >= 0 or -1 (auto=L).")
    if int(adapt_eps_energy_patience) < -1 or int(adapt_eps_energy_patience) == 0:
        raise ValueError("adapt_eps_energy_patience must be >= 1 or -1 (auto=L).")
    if int(adapt_ref_base_depth) < 0:
        raise ValueError("adapt_ref_base_depth must be >= 0.")
    problem_key = str(problem).strip().lower()
    continuation_mode = _resolve_adapt_continuation_mode(
        problem=str(problem_key),
        requested_mode=adapt_continuation_mode,
    )
    stop_policy = _resolve_adapt_stop_policy(
        problem=str(problem_key),
        continuation_mode=str(continuation_mode),
        adapt_drop_floor=adapt_drop_floor,
        adapt_drop_patience=adapt_drop_patience,
        adapt_drop_min_depth=adapt_drop_min_depth,
        adapt_grad_floor=adapt_grad_floor,
    )
    adapt_drop_floor = float(stop_policy.adapt_drop_floor)
    adapt_drop_patience = int(stop_policy.adapt_drop_patience)
    adapt_drop_min_depth = int(stop_policy.adapt_drop_min_depth)
    adapt_grad_floor = float(stop_policy.adapt_grad_floor)
    drop_policy_enabled = bool(stop_policy.drop_policy_enabled)
    eps_energy_termination_enabled = bool(stop_policy.eps_energy_termination_enabled)
    eps_grad_termination_enabled = bool(stop_policy.eps_grad_termination_enabled)
    if float(adapt_drop_floor) >= 0.0 and int(adapt_drop_patience) < 1:
        raise ValueError("adapt_drop_patience must be >= 1 when adapt_drop_floor is enabled.")
    if float(adapt_drop_floor) >= 0.0 and int(adapt_drop_min_depth) < 1:
        raise ValueError("adapt_drop_min_depth must be >= 1 when adapt_drop_floor is enabled.")
    if int(adapt_drop_patience) < 0:
        raise ValueError("adapt_drop_patience must be >= 0.")
    if int(adapt_drop_min_depth) < 0:
        raise ValueError("adapt_drop_min_depth must be >= 0.")
    eps_energy_min_extra_depth_effective = (
        int(num_sites)
        if int(adapt_eps_energy_min_extra_depth) == -1
        else int(adapt_eps_energy_min_extra_depth)
    )
    eps_energy_patience_effective = (
        int(num_sites)
        if int(adapt_eps_energy_patience) == -1
        else int(adapt_eps_energy_patience)
    )
    if int(eps_energy_patience_effective) < 1:
        raise ValueError("resolved eps-energy patience must be >= 1.")
    if int(eps_energy_min_extra_depth_effective) < 0:
        raise ValueError("resolved eps-energy min extra depth must be >= 0.")
    phase3_symmetry_mitigation_mode_key = str(phase3_symmetry_mitigation_mode).strip().lower()
    if phase3_symmetry_mitigation_mode_key not in {"off", "verify_only", "postselect_diag_v1", "projector_renorm_v1"}:
        raise ValueError(
            "phase3_symmetry_mitigation_mode must be one of {'off','verify_only','postselect_diag_v1','projector_renorm_v1'}."
        )
    phase3_lifetime_cost_mode_key = str(phase3_lifetime_cost_mode).strip().lower()
    if phase3_lifetime_cost_mode_key not in {"off", "phase3_v1"}:
        raise ValueError("phase3_lifetime_cost_mode must be one of {'off','phase3_v1'}.")
    phase3_runtime_split_mode_key = str(phase3_runtime_split_mode).strip().lower()
    if phase3_runtime_split_mode_key not in {"off", "shortlist_pauli_children_v1"}:
        raise ValueError(
            "phase3_runtime_split_mode must be one of {'off','shortlist_pauli_children_v1'}."
        )
    phase3_backend_cost_mode_key = str(phase3_backend_cost_mode).strip().lower()
    if phase3_backend_cost_mode_key not in {"proxy", "transpile_single_v1", "transpile_shortlist_v1"}:
        raise ValueError(
            "phase3_backend_cost_mode must be one of {'proxy','transpile_single_v1','transpile_shortlist_v1'}."
        )
    phase3_backend_shortlist_tokens = tuple(
        str(tok).strip()
        for tok in (str(phase3_backend_shortlist).split(",") if isinstance(phase3_backend_shortlist, str) else list(phase3_backend_shortlist or []))
        if str(tok).strip() != ""
    )
    if phase3_backend_cost_mode_key != "proxy":
        if str(problem_key) != "hh":
            raise ValueError("phase3_backend_cost_mode is only valid for problem='hh'.")
        if str(continuation_mode) != "phase3_v1":
            raise ValueError("phase3_backend_cost_mode is only valid for adapt_continuation_mode='phase3_v1'.")
        if int(phase3_backend_optimization_level) not in {0, 1, 2, 3}:
            raise ValueError("phase3_backend_optimization_level must be one of {0,1,2,3}.")
        if phase3_backend_cost_mode_key == "transpile_single_v1":
            if phase3_backend_name in {None, ""}:
                raise ValueError("transpile_single_v1 requires --phase3-backend-name.")
            if phase3_backend_shortlist_tokens:
                raise ValueError("transpile_single_v1 does not accept --phase3-backend-shortlist.")
        if phase3_backend_cost_mode_key == "transpile_shortlist_v1":
            if phase3_backend_name not in {None, ""}:
                raise ValueError("transpile_shortlist_v1 does not accept --phase3-backend-name.")
            if len(phase3_backend_shortlist_tokens) < 1:
                raise ValueError("transpile_shortlist_v1 requires --phase3-backend-shortlist.")
    pool_key_input = None if adapt_pool is None else str(adapt_pool).strip().lower()
    adapt_spsa_params = {
        "a": float(adapt_spsa_a),
        "c": float(adapt_spsa_c),
        "alpha": float(adapt_spsa_alpha),
        "gamma": float(adapt_spsa_gamma),
        "A": float(adapt_spsa_A),
        "avg_last": int(adapt_spsa_avg_last),
        "eval_repeats": int(adapt_spsa_eval_repeats),
        "eval_agg": str(adapt_spsa_eval_agg_key),
        "callback_every": int(adapt_spsa_callback_every),
        "progress_every_s": float(adapt_spsa_progress_every_s),
    }
    t0 = time.perf_counter()
    hf_bits = "N/A"
    _ai_log(
        "hardcoded_adapt_vqe_start",
        L=int(num_sites),
        problem=str(problem),
        adapt_pool=(str(pool_key_input) if pool_key_input is not None else None),
        adapt_continuation_mode=str(continuation_mode),
        phase3_motif_source_json=(str(phase3_motif_source_json) if phase3_motif_source_json is not None else None),
        phase3_symmetry_mitigation_mode=str(phase3_symmetry_mitigation_mode_key),
        phase3_runtime_split_mode=str(phase3_runtime_split_mode_key),
        phase3_backend_cost_mode=str(phase3_backend_cost_mode_key),
        phase3_backend_name=(None if phase3_backend_name in {None, ''} else str(phase3_backend_name)),
        phase3_backend_shortlist=[str(x) for x in phase3_backend_shortlist_tokens],
        phase3_backend_transpile_seed=int(phase3_backend_transpile_seed),
        phase3_backend_optimization_level=int(phase3_backend_optimization_level),
        phase3_enable_rescue=bool(phase3_enable_rescue),
        max_depth=int(max_depth),
        maxiter=int(maxiter),
        adapt_inner_optimizer=str(adapt_inner_optimizer_key),
        finite_angle_fallback=bool(finite_angle_fallback),
        finite_angle=float(finite_angle),
        finite_angle_min_improvement=float(finite_angle_min_improvement),
        adapt_gradient_parity_check=bool(adapt_gradient_parity_check),
        adapt_state_backend=str(adapt_state_backend_key),
        adapt_drop_policy_enabled=bool(drop_policy_enabled),
        adapt_drop_floor=(float(adapt_drop_floor) if drop_policy_enabled else None),
        adapt_drop_patience=(int(adapt_drop_patience) if drop_policy_enabled else None),
        adapt_drop_min_depth=(int(adapt_drop_min_depth) if drop_policy_enabled else None),
        adapt_grad_floor=(float(adapt_grad_floor) if float(adapt_grad_floor) >= 0.0 else None),
        adapt_drop_floor_resolved=float(adapt_drop_floor),
        adapt_drop_patience_resolved=int(adapt_drop_patience),
        adapt_drop_min_depth_resolved=int(adapt_drop_min_depth),
        adapt_grad_floor_resolved=float(adapt_grad_floor),
        adapt_drop_floor_source=str(stop_policy.adapt_drop_floor_source),
        adapt_drop_patience_source=str(stop_policy.adapt_drop_patience_source),
        adapt_drop_min_depth_source=str(stop_policy.adapt_drop_min_depth_source),
        adapt_grad_floor_source=str(stop_policy.adapt_grad_floor_source),
        adapt_drop_policy_source=str(stop_policy.drop_policy_source),
        adapt_eps_energy_min_extra_depth=int(adapt_eps_energy_min_extra_depth),
        adapt_eps_energy_patience=int(adapt_eps_energy_patience),
        adapt_ref_base_depth=int(adapt_ref_base_depth),
        adapt_eps_energy_min_extra_depth_effective=int(eps_energy_min_extra_depth_effective),
        adapt_eps_energy_patience_effective=int(eps_energy_patience_effective),
        eps_energy_gate_cumulative_depth=int(adapt_ref_base_depth) + int(eps_energy_min_extra_depth_effective),
        eps_energy_termination_enabled=bool(eps_energy_termination_enabled),
        eps_grad_termination_enabled=bool(eps_grad_termination_enabled),
        adapt_reopt_policy=str(adapt_reopt_policy_key),
        adapt_window_size=int(adapt_window_size_val),
        adapt_window_topk=int(adapt_window_topk_val),
        adapt_full_refit_every=int(adapt_full_refit_every_val),
        adapt_final_full_refit=bool(adapt_final_full_refit_val),
    )

    num_particles = half_filled_num_particles(int(num_sites))
    psi_ref, _, _ = _default_adapt_input_state(
        problem=str(problem_key),
        num_sites=int(num_sites),
        ordering=str(ordering),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
    )
    if problem_key != "hh":
        hf_bits = str(
            hartree_fock_bitstring(
                n_sites=int(num_sites),
                num_particles=num_particles,
                indexing=str(ordering),
            )
        )
    if psi_ref_override is not None:
        psi_ref_override_arr = np.asarray(psi_ref_override, dtype=complex).reshape(-1)
        if int(psi_ref_override_arr.size) != int(psi_ref.size):
            raise ValueError(
                f"psi_ref_override length mismatch: got {psi_ref_override_arr.size}, expected {psi_ref.size}"
            )
        psi_ref = _normalize_state(psi_ref_override_arr)
        _ai_log(
            "hardcoded_adapt_ref_override_applied",
            nq=int(round(math.log2(psi_ref.size))),
            dim=int(psi_ref.size),
        )

    # Build operator pool(s)
    def _build_hh_pool_by_key(pool_key_hh: str) -> tuple[list[AnsatzTerm], str]:
        key = str(pool_key_hh).strip().lower()
        if key == "hva":
            hva_pool = _build_hva_pool(
                int(num_sites),
                float(t),
                float(u),
                float(omega0),
                float(g_ep),
                float(dv),
                int(n_ph_max),
                str(boson_encoding),
                str(ordering),
                str(boundary),
            )
            if abs(float(g_ep)) <= 1e-15:
                return list(hva_pool), "hardcoded_adapt_vqe_hva_hh"
            ham_term_pool = _build_hh_termwise_augmented_pool(h_poly)
            merged_pool = list(hva_pool) + [
                AnsatzTerm(label=f"hh_termwise_{term.label}", polynomial=term.polynomial)
                for term in ham_term_pool
            ]
            seen: set[tuple[tuple[str, float], ...]] = set()
            dedup_pool: list[AnsatzTerm] = []
            for term in merged_pool:
                sig = _polynomial_signature(term.polynomial)
                if sig in seen:
                    continue
                seen.add(sig)
                dedup_pool.append(term)
            return dedup_pool, "hardcoded_adapt_vqe_hva_hh"
        if key == "full_meta":
            pool_full, full_meta_sizes = _build_hh_full_meta_pool(
                h_poly=h_poly,
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                omega0=float(omega0),
                g_ep=float(g_ep),
                dv=float(dv),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                ordering=str(ordering),
                boundary=str(boundary),
                paop_r=int(paop_r),
                paop_split_paulis=bool(paop_split_paulis),
                paop_prune_eps=float(paop_prune_eps),
                paop_normalization=str(paop_normalization),
                num_particles=num_particles,
            )
            _ai_log(
                "hardcoded_adapt_full_meta_pool_built",
                **full_meta_sizes,
                dedup_total=int(len(pool_full)),
            )
            return list(pool_full), "hardcoded_adapt_vqe_full_meta"
        if key == "pareto_lean":
            pool_lean, lean_sizes = _build_hh_pareto_lean_pool(
                h_poly=h_poly,
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                omega0=float(omega0),
                g_ep=float(g_ep),
                dv=float(dv),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                ordering=str(ordering),
                boundary=str(boundary),
                paop_r=int(paop_r),
                paop_split_paulis=bool(paop_split_paulis),
                paop_prune_eps=float(paop_prune_eps),
                paop_normalization=str(paop_normalization),
                num_particles=num_particles,
            )
            _ai_log(
                "hardcoded_adapt_pareto_lean_pool_built",
                **lean_sizes,
                dedup_total=int(len(pool_lean)),
            )
            return list(pool_lean), "hardcoded_adapt_vqe_pareto_lean"
        if key == "pareto_lean_l2":
            pool_lean_l2, lean_l2_sizes = _build_hh_pareto_lean_l2_pool(
                h_poly=h_poly,
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                omega0=float(omega0),
                g_ep=float(g_ep),
                dv=float(dv),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                ordering=str(ordering),
                boundary=str(boundary),
                paop_r=int(paop_r),
                paop_split_paulis=bool(paop_split_paulis),
                paop_prune_eps=float(paop_prune_eps),
                paop_normalization=str(paop_normalization),
                num_particles=num_particles,
            )
            _ai_log(
                "hardcoded_adapt_pareto_lean_l2_pool_built",
                **lean_l2_sizes,
                dedup_total=int(len(pool_lean_l2)),
            )
            return list(pool_lean_l2), "hardcoded_adapt_vqe_pareto_lean_l2"
        if key == "pareto_lean_gate_pruned":
            pool_gp, gp_sizes = _build_hh_pareto_lean_gate_pruned_pool(
                h_poly=h_poly,
                num_sites=int(num_sites),
                t=float(t),
                u=float(u),
                omega0=float(omega0),
                g_ep=float(g_ep),
                dv=float(dv),
                n_ph_max=int(n_ph_max),
                boson_encoding=str(boson_encoding),
                ordering=str(ordering),
                boundary=str(boundary),
                paop_r=int(paop_r),
                paop_split_paulis=bool(paop_split_paulis),
                paop_prune_eps=float(paop_prune_eps),
                paop_normalization=str(paop_normalization),
                num_particles=num_particles,
            )
            _ai_log(
                "hardcoded_adapt_pareto_lean_gate_pruned_pool_built",
                **gp_sizes,
                dedup_total=int(len(pool_gp)),
            )
            return list(pool_gp), "hardcoded_adapt_vqe_pareto_lean_gate_pruned"
        if key == "uccsd_paop_lf_full":
            uccsd_lifted_pool = _build_hh_uccsd_fermion_lifted_pool(
                int(num_sites),
                int(n_ph_max),
                str(boson_encoding),
                str(ordering),
                str(boundary),
                num_particles=num_particles,
            )
            paop_pool = _build_paop_pool(
                int(num_sites),
                int(n_ph_max),
                str(boson_encoding),
                str(ordering),
                str(boundary),
                "paop_lf_full",
                int(paop_r),
                bool(paop_split_paulis),
                float(paop_prune_eps),
                str(paop_normalization),
                num_particles,
            )
            return _deduplicate_pool_terms(list(uccsd_lifted_pool) + list(paop_pool)), "hardcoded_adapt_vqe_uccsd_paop_lf_full"
        if key in {"paop", "paop_min", "paop_std", "paop_full", "paop_lf", "paop_lf_std", "paop_lf2_std", "paop_lf_full"}:
            paop_pool = _build_paop_pool(
                int(num_sites),
                int(n_ph_max),
                str(boson_encoding),
                str(ordering),
                str(boundary),
                key,
                int(paop_r),
                bool(paop_split_paulis),
                float(paop_prune_eps),
                str(paop_normalization),
                num_particles,
            )
            if abs(float(g_ep)) <= 1e-15:
                return list(paop_pool), f"hardcoded_adapt_vqe_{key}"
            hva_pool = _build_hva_pool(
                int(num_sites),
                float(t),
                float(u),
                float(omega0),
                float(g_ep),
                float(dv),
                int(n_ph_max),
                str(boson_encoding),
                str(ordering),
                str(boundary),
            )
            ham_term_pool = _build_hh_termwise_augmented_pool(h_poly)
            merged_pool = list(hva_pool) + [
                AnsatzTerm(label=f"hh_termwise_{term.label}", polynomial=term.polynomial)
                for term in ham_term_pool
            ] + list(paop_pool)
            seen: set[tuple[tuple[str, float], ...]] = set()
            dedup_pool: list[AnsatzTerm] = []
            for term in merged_pool:
                sig = _polynomial_signature(term.polynomial)
                if sig in seen:
                    continue
                seen.add(sig)
                dedup_pool.append(term)
            return dedup_pool, f"hardcoded_adapt_vqe_{key}"
        if key == "full_hamiltonian":
            return _build_full_hamiltonian_pool(h_poly, normalize_coeff=True), "hardcoded_adapt_vqe_full_hamiltonian_hh"
        raise ValueError(
            "For problem='hh', supported ADAPT pools are: "
            "hva, full_meta, pareto_lean, pareto_lean_l2, pareto_lean_gate_pruned, uccsd_paop_lf_full, paop, paop_min, paop_std, paop_full, "
            "paop_lf, paop_lf_std, paop_lf2_std, paop_lf_full, full_hamiltonian"
        )

    pool_stage_family: list[str] = []
    pool_family_ids: list[str] = []
    phase1_core_limit = 0
    phase1_residual_indices: set[int] = set()
    phase1_depth0_full_meta_override = False
    if continuation_mode in {"phase1_v1", "phase2_v1", "phase3_v1"} and problem_key == "hh":
        if pool_key_input in ("full_meta", "pareto_lean", "pareto_lean_l2", "pareto_lean_gate_pruned"):
            phase1_depth0_full_meta_override = True
            pool, _pool_method = _build_hh_pool_by_key(str(pool_key_input))
            phase1_core_limit = int(len(pool))
            phase1_residual_indices = set()
            pool_stage_family = ["core"] * int(len(pool))
            pool_family_ids = [str(pool_key_input)] * int(len(pool))
            _ai_log(
                "hardcoded_adapt_phase1_depth0_full_meta_override",
                continuation_mode=str(continuation_mode),
                pool_key=str(pool_key_input),
                pool_size=int(len(pool)),
            )
        else:
            core_key = str(pool_key_input if pool_key_input is not None else "paop_lf_std")
            core_pool, _core_method = _build_hh_pool_by_key(core_key)
            residual_pool, _residual_method = _build_hh_pool_by_key("full_meta")
            seen_sig = {_polynomial_signature(op.polynomial) for op in core_pool}
            residual_unique: list[AnsatzTerm] = []
            for op in residual_pool:
                sig = _polynomial_signature(op.polynomial)
                if sig in seen_sig:
                    continue
                seen_sig.add(sig)
                residual_unique.append(op)
            pool = list(core_pool) + list(residual_unique)
            phase1_core_limit = int(len(core_pool))
            phase1_residual_indices = set(range(int(phase1_core_limit), int(len(pool))))
            pool_stage_family = (["core"] * int(phase1_core_limit)) + (["residual"] * int(len(residual_unique)))
            pool_family_ids = ([str(core_key)] * int(phase1_core_limit)) + (["full_meta"] * int(len(residual_unique)))
        method_name = f"hardcoded_adapt_vqe_{str(continuation_mode)}_hh"
        pool_key = str(continuation_mode)
    else:
        pool_key = str(pool_key_input if pool_key_input is not None else ("uccsd" if problem_key == "hubbard" else "full_meta"))
        if problem_key == "hh":
            pool, method_name = _build_hh_pool_by_key(pool_key)
        else:
            if pool_key == "uccsd":
                pool = _build_uccsd_pool(int(num_sites), num_particles, str(ordering))
                method_name = "hardcoded_adapt_vqe_uccsd"
            elif pool_key == "cse":
                pool = _build_cse_pool(
                    int(num_sites),
                    str(ordering),
                    float(t),
                    float(u),
                    float(dv),
                    str(boundary),
                )
                method_name = "hardcoded_adapt_vqe_cse"
            elif pool_key == "full_hamiltonian":
                pool = _build_full_hamiltonian_pool(h_poly)
                method_name = "hardcoded_adapt_vqe_full_hamiltonian"
            elif pool_key == "hva":
                raise ValueError(
                    "For problem='hubbard', pool='hva' is not valid. "
                    "Use uccsd, cse, or full_hamiltonian."
                )
            elif pool_key == "full_meta":
                raise ValueError("Pool 'full_meta' is only valid for problem='hh'.")
            elif pool_key == "uccsd_paop_lf_full":
                raise ValueError("Pool 'uccsd_paop_lf_full' is only valid for problem='hh'.")
            else:
                raise ValueError(f"Unsupported adapt pool '{adapt_pool}'.")
        pool_stage_family = [str(pool_key)] * int(len(pool))
        pool_family_ids = [str(pool_key)] * int(len(pool))

    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding))) if problem_key == "hh" else 1
    pool_symmetry_specs: list[dict[str, Any] | None] = [None] * int(len(pool))
    pool_generator_registry: dict[str, dict[str, Any]] = {}
    if problem_key == "hh" and len(pool) > 0:
        base_pool_symmetry_specs = [
            dict(
                build_symmetry_spec(
                    family_id=str(pool_family_ids[idx] if idx < len(pool_family_ids) else "unknown"),
                    mitigation_mode=str(phase3_symmetry_mitigation_mode_key),
                ).__dict__
            )
            for idx in range(len(pool))
        ]
        raw_pool_generator_registry = build_pool_generator_registry(
            terms=pool,
            family_ids=pool_family_ids,
            num_sites=int(num_sites),
            ordering=str(ordering),
            qpb=int(max(1, qpb)),
            symmetry_specs=base_pool_symmetry_specs,
            split_policy=("deliberate_split" if bool(paop_split_paulis) else "preserve"),
        )
        filtered_pool: list[AnsatzTerm] = []
        filtered_stage_family: list[str] = []
        filtered_family_ids: list[str] = []
        filtered_specs: list[dict[str, Any] | None] = []
        filtered_registry: dict[str, dict[str, Any]] = {}
        removed_labels: list[str] = []
        removed_family_ids: list[str] = []
        for idx, term in enumerate(pool):
            label = str(term.label)
            meta = raw_pool_generator_registry.get(label)
            spec = (
                meta.get("symmetry_spec")
                if isinstance(meta, Mapping)
                else base_pool_symmetry_specs[idx]
            )
            if isinstance(spec, Mapping) and bool(spec.get("hard_guard", False)):
                removed_labels.append(label)
                removed_family_ids.append(str(pool_family_ids[idx] if idx < len(pool_family_ids) else "unknown"))
                continue
            filtered_pool.append(term)
            filtered_stage_family.append(str(pool_stage_family[idx] if idx < len(pool_stage_family) else pool_key))
            filtered_family_ids.append(str(pool_family_ids[idx] if idx < len(pool_family_ids) else pool_key))
            filtered_specs.append(dict(spec) if isinstance(spec, Mapping) else None)
            if isinstance(meta, Mapping):
                filtered_registry[label] = dict(meta)
        if removed_labels:
            _ai_log(
                "hardcoded_adapt_hh_pool_symmetry_filtered",
                removed_count=int(len(removed_labels)),
                kept_count=int(len(filtered_pool)),
                removed_labels_sample=[str(x) for x in removed_labels[:12]],
                removed_families_sample=[str(x) for x in removed_family_ids[:12]],
            )
        pool = filtered_pool
        pool_stage_family = filtered_stage_family
        pool_family_ids = filtered_family_ids
        pool_symmetry_specs = filtered_specs
        pool_generator_registry = filtered_registry
        if continuation_mode in {"phase1_v1", "phase2_v1", "phase3_v1"}:
            phase1_core_limit = int(sum(1 for stage in pool_stage_family if str(stage) == "core"))
            phase1_residual_indices = {
                int(idx) for idx, stage in enumerate(pool_stage_family) if str(stage) == "residual"
            }

    if len(pool) == 0:
        raise ValueError(f"ADAPT pool '{pool_key}' produced no operators for problem='{problem_key}'.")
    _ai_log(
        "hardcoded_adapt_pool_built",
        pool_type=str(pool_key),
        pool_size=int(len(pool)),
        continuation_mode=str(continuation_mode),
        phase1_depth0_full_meta_override=bool(phase1_depth0_full_meta_override),
        phase1_core_size=(
            int(phase1_core_limit)
            if continuation_mode in {"phase1_v1", "phase2_v1", "phase3_v1"} and problem_key == "hh"
            else None
        ),
        phase1_residual_size=(
            int(len(phase1_residual_indices))
            if continuation_mode in {"phase1_v1", "phase2_v1", "phase3_v1"} and problem_key == "hh"
            else None
        ),
    )

    phase1_enabled = bool(continuation_mode in {"phase1_v1", "phase2_v1", "phase3_v1"} and problem_key == "hh")
    phase2_enabled = bool(continuation_mode in {"phase2_v1", "phase3_v1"} and problem_key == "hh")
    phase3_enabled = bool(continuation_mode == "phase3_v1" and problem_key == "hh")
    phase3_split_events: list[dict[str, Any]] = []
    phase3_input_motif_library: dict[str, Any] | None = None
    phase3_runtime_split_summary: dict[str, Any] = {
        "mode": (str(phase3_runtime_split_mode_key) if phase3_enabled else "off"),
        "probed_parent_count": 0,
        "evaluated_child_count": 0,
        "rejected_child_count_symmetry": 0,
        "admissible_child_set_count": 0,
        "probe_parent_win_count": 0,
        "probe_child_set_count": 0,
        "selected_child_set_count": 0,
        "selected_child_count": 0,
        "selected_child_labels": [],
    }
    phase3_motif_usage: dict[str, Any] = {
        "enabled": False,
        "source_json": (str(phase3_motif_source_json) if phase3_motif_source_json is not None else None),
        "source_tag": None,
        "seeded_labels": [],
        "seeded_generator_ids": [],
        "seeded_motif_ids": [],
        "selected_match_count": 0,
    }
    phase3_rescue_history: list[dict[str, Any]] = []
    phase3_exact_reference_state: np.ndarray | None = None
    if phase3_enabled and not pool_generator_registry:
        pool_symmetry_specs = [
            dict(
                build_symmetry_spec(
                    family_id=str(pool_family_ids[idx] if idx < len(pool_family_ids) else "unknown"),
                    mitigation_mode=str(phase3_symmetry_mitigation_mode_key),
                ).__dict__
            )
            for idx in range(len(pool))
        ]
        pool_generator_registry = build_pool_generator_registry(
            terms=pool,
            family_ids=pool_family_ids,
            num_sites=int(num_sites),
            ordering=str(ordering),
            qpb=int(max(1, qpb)),
            symmetry_specs=pool_symmetry_specs,
            split_policy=("deliberate_split" if bool(paop_split_paulis) else "preserve"),
        )
    if phase3_enabled and phase3_motif_source_json is not None:
        phase3_input_motif_library = load_motif_library_from_json(Path(phase3_motif_source_json))
        if phase3_input_motif_library is not None:
            phase3_motif_usage["enabled"] = True
            phase3_motif_usage["source_tag"] = str(phase3_input_motif_library.get("source_tag", "payload"))
    if phase3_enabled and bool(phase3_enable_rescue):
        phase3_exact_reference_state = _exact_reference_state_for_hh(
            num_sites=int(num_sites),
            num_particles=num_particles,
            indexing=str(ordering),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            t=float(t),
            u=float(u),
            dv=float(dv),
            omega0=float(omega0),
            g_ep=float(g_ep),
            boundary=str(boundary),
        )
        _ai_log(
            "hardcoded_adapt_phase3_registry_ready",
            pool_size=int(len(pool)),
            generator_count=int(len(pool_generator_registry)),
            motif_source_enabled=bool(phase3_input_motif_library is not None),
            rescue_exact_state_available=bool(phase3_exact_reference_state is not None),
        )

    compile_cache_t0 = time.perf_counter()
    pauli_action_cache: dict[str, CompiledPauliAction] = {}
    h_compiled = _compile_polynomial_action(
        h_poly,
        pauli_action_cache=pauli_action_cache,
    )
    pool_compiled = [
        _compile_polynomial_action(
            op.polynomial,
            pauli_action_cache=pauli_action_cache,
        )
        for op in pool
    ]
    compile_cache_elapsed_s = float(time.perf_counter() - compile_cache_t0)
    pool_compiled_terms_total = int(sum(len(compiled_poly.terms) for compiled_poly in pool_compiled))
    _ai_log(
        "hardcoded_adapt_compiled_cache_ready",
        pool_size=int(len(pool)),
        h_terms=int(len(h_compiled.terms)),
        pool_terms_total=pool_compiled_terms_total,
        unique_pauli_actions=int(len(pauli_action_cache)),
        compile_elapsed_s=compile_cache_elapsed_s,
    )
    _ai_log(
        "hardcoded_adapt_compile_timing",
        pool_size=int(len(pool)),
        pool_terms_total=pool_compiled_terms_total,
        unique_pauli_actions=int(len(pauli_action_cache)),
        compile_elapsed_s=compile_cache_elapsed_s,
    )

    def _build_compiled_executor(ops: list[AnsatzTerm]) -> CompiledAnsatzExecutor:
        return CompiledAnsatzExecutor(
            ops,
            coefficient_tolerance=1e-12,
            ignore_identity=True,
            sort_terms=True,
            pauli_action_cache=pauli_action_cache,
            parameterization_mode="per_pauli_term",
        )

    def _build_selected_layout(ops: list[AnsatzTerm]) -> AnsatzParameterLayout:
        return build_parameter_layout(
            ops,
            ignore_identity=True,
            coefficient_tolerance=1e-12,
            sort_terms=True,
        )

    # ADAPT-VQE main loop
    selected_ops: list[AnsatzTerm] = []
    selected_layout = _build_selected_layout(selected_ops)
    theta = np.zeros(0, dtype=float)
    selected_executor: CompiledAnsatzExecutor | None = None
    history: list[dict[str, Any]] = []
    nfev_total = 0
    stop_reason = "max_depth"

    scipy_minimize = None
    if adapt_inner_optimizer_key in {"COBYLA", "POWELL"}:
        from scipy.optimize import minimize as scipy_minimize

    def _scipy_inner_options(maxiter_value: int) -> dict[str, Any]:
        if adapt_inner_optimizer_key == "COBYLA":
            return {"maxiter": int(maxiter_value), "rhobeg": 0.3}
        if adapt_inner_optimizer_key == "POWELL":
            return {"maxiter": int(maxiter_value), "xtol": 1e-4, "ftol": 1e-8}
        raise ValueError(f"Unsupported SciPy ADAPT inner optimizer: {adapt_inner_optimizer_key}")

    # Pool availability tracking (for no-repeat mode)
    available_indices = (
        set(range(int(phase1_core_limit)))
        if phase1_enabled
        else set(range(len(pool)))
    )
    selection_counts = np.zeros(len(pool), dtype=np.int64)
    phase1_stage_cfg = StageControllerConfig(
        plateau_patience=int(max(1, phase1_plateau_patience)),
        weak_drop_threshold=(
            float(adapt_drop_floor)
            if bool(drop_policy_enabled)
            else float(max(float(eps_energy), 1e-12))
        ),
        probe_margin_ratio=float(max(0.0, phase1_trough_margin_ratio)),
        max_probe_positions=int(max(1, phase1_probe_max_positions)),
        append_admit_threshold=0.05,
        family_repeat_patience=int(max(1, phase1_plateau_patience)),
    )
    phase1_stage = StageController(phase1_stage_cfg)
    if phase1_enabled:
        phase1_stage.start_with_seed()
    phase1_score_cfg = SimpleScoreConfig(
        lambda_F=float(phase1_lambda_F),
        lambda_compile=float(phase1_lambda_compile),
        lambda_measure=float(phase1_lambda_measure),
        lambda_leak=float(phase1_lambda_leak),
        z_alpha=float(phase1_score_z_alpha),
    )
    phase1_compile_oracle = Phase1CompileCostOracle()
    phase1_measure_cache = MeasurementCacheAudit(nominal_shots_per_group=1)
    backend_compile_cfg = BackendCompileConfig(
        mode=str(phase3_backend_cost_mode_key),
        requested_backend_name=(None if phase3_backend_name in {None, ''} else str(phase3_backend_name)),
        requested_backend_shortlist=tuple(str(x) for x in phase3_backend_shortlist_tokens),
        seed_transpiler=int(phase3_backend_transpile_seed),
        optimization_level=int(phase3_backend_optimization_level),
    )
    backend_compile_oracle = (
        BackendCompileOracle(
            config=backend_compile_cfg,
            num_qubits=int(round(math.log2(psi_ref.size))),
            ref_state=np.asarray(psi_ref, dtype=complex),
        )
        if str(phase3_backend_cost_mode_key) != "proxy"
        else None
    )
    if backend_compile_oracle is not None and len(getattr(backend_compile_oracle, "targets", ())) == 0:
        raise RuntimeError("No backend targets could be resolved for phase3 backend-aware scoring.")
    phase2_score_cfg = FullScoreConfig(
        z_alpha=float(phase1_score_z_alpha),
        lambda_F=float(phase1_lambda_F),
        lambda_H=float(max(1e-12, phase2_lambda_H)),
        rho=float(max(1e-6, phase2_rho)),
        gamma_N=float(max(0.0, phase2_gamma_N)),
        shortlist_fraction=float(max(0.05, phase2_shortlist_fraction)),
        shortlist_size=int(max(1, phase2_shortlist_size)),
        batch_target_size=int(max(1, phase2_batch_target_size)),
        batch_size_cap=int(max(1, phase2_batch_size_cap)),
        batch_near_degenerate_ratio=float(max(0.0, min(1.0, phase2_batch_near_degenerate_ratio))),
        lifetime_cost_mode=(
            str(phase3_lifetime_cost_mode_key)
            if phase3_enabled and str(phase3_lifetime_cost_mode_key) != "off"
            else "off"
        ),
        remaining_evaluations_proxy_mode=(
            "remaining_depth"
            if phase3_enabled and str(phase3_lifetime_cost_mode_key) != "off"
            else "none"
        ),
    )
    phase2_novelty_oracle = Phase2NoveltyOracle()
    phase2_curvature_oracle = Phase2CurvatureOracle()
    phase2_memory_adapter = Phase2OptimizerMemoryAdapter()
    phase2_compiled_term_cache: dict[str, Any] = {}
    phase2_optimizer_memory = phase2_memory_adapter.unavailable(
        method=str(adapt_inner_optimizer_key),
        parameter_count=int(theta.size),
        reason="pre_seed_state",
    )
    phase1_residual_opened = False
    phase1_last_probe_reason = "none"
    phase1_last_positions_considered: list[int] = []
    phase1_last_trough_detected = False
    phase1_last_trough_probe_triggered = False
    phase1_last_selected_score: float | None = None
    phase1_features_history: list[dict[str, Any]] = []
    phase1_stage_events: list[dict[str, Any]] = []
    phase1_scaffold_pre_prune: dict[str, Any] | None = None
    phase2_last_shortlist_records: list[dict[str, Any]] = []
    phase2_last_batch_selected = False
    phase2_last_batch_penalty_total = 0.0
    phase2_last_optimizer_memory_reused = False
    phase2_last_optimizer_memory_source = "unavailable"
    phase2_last_shortlist_eval_records: list[dict[str, Any]] = []

    energy_current, _ = energy_via_one_apply(psi_ref, h_compiled)
    energy_current = float(energy_current)
    nfev_total += 1
    _ai_log("hardcoded_adapt_initial_energy", energy=energy_current)
    if exact_gs_override is None:
        exact_gs = _exact_gs_energy_for_problem(
            h_poly,
            problem=problem_key,
            num_sites=int(num_sites),
            num_particles=num_particles,
            indexing=str(ordering),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            t=float(t),
            u=float(u),
            dv=float(dv),
            omega0=float(omega0),
            g_ep=float(g_ep),
            boundary=str(boundary),
        )
    else:
        exact_gs = float(exact_gs_override)
        _ai_log("hardcoded_adapt_exact_override_used", exact_gs=exact_gs)
    drop_prev_delta_abs = float(abs(energy_current - exact_gs))
    drop_plateau_hits = 0
    eps_energy_low_streak = 0

    # HH preconditioning: optimize a compact boson-quadrature e-ph seed block
    # before greedy ADAPT selection. This helps avoid the weak-coupling basin
    # when g is moderate/strong.
    if (
        (not disable_hh_seed)
        and problem_key == "hh"
        and abs(float(g_ep)) > 1e-15
    ):
        n_sites = int(num_sites)
        boson_bits = n_sites * int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
        seed_indices: list[int] = []
        for idx, op in enumerate(pool):
            label = str(op.label)
            if not label.startswith("hh_termwise_ham_quadrature_term("):
                continue
            op_terms = op.polynomial.return_polynomial()
            if not op_terms:
                continue
            pw = str(op_terms[0].pw2strng())
            has_boson_y = any(ch == "y" for ch in pw[:boson_bits])
            has_electron_z = ("z" in pw[boson_bits:])
            if has_boson_y and has_electron_z:
                seed_indices.append(idx)

        if seed_indices:
            seed_ops = [pool[i] for i in seed_indices]
            seed_layout = _build_selected_layout(seed_ops)
            theta_seed0 = np.zeros(int(seed_layout.runtime_parameter_count), dtype=float)
            seed_executor = (
                _build_compiled_executor(seed_ops)
                if adapt_state_backend_key == "compiled"
                else None
            )
            seed_opt_t0 = time.perf_counter()
            seed_cobyla_last_hb_t = seed_opt_t0
            seed_cobyla_nfev_so_far = 0
            seed_cobyla_best_fun = float("inf")

            def _seed_obj(x: np.ndarray) -> float:
                nonlocal seed_cobyla_last_hb_t, seed_cobyla_nfev_so_far, seed_cobyla_best_fun
                if seed_executor is not None:
                    psi_seed = seed_executor.prepare_state(np.asarray(x, dtype=float), psi_ref)
                    seed_energy, _ = energy_via_one_apply(psi_seed, h_compiled)
                    seed_energy_val = float(seed_energy)
                else:
                    seed_energy_val = _adapt_energy_fn(
                        h_poly,
                        psi_ref,
                        seed_ops,
                        x,
                        h_compiled=h_compiled,
                    )
                if adapt_inner_optimizer_key in {"COBYLA", "POWELL"}:
                    seed_cobyla_nfev_so_far += 1
                    if seed_energy_val < seed_cobyla_best_fun:
                        seed_cobyla_best_fun = float(seed_energy_val)
                    now = time.perf_counter()
                    if (now - seed_cobyla_last_hb_t) >= float(adapt_spsa_progress_every_s):
                        _ai_log(
                            "hardcoded_adapt_scipy_heartbeat",
                            stage="hh_seed_preopt",
                            depth=0,
                            opt_method=str(adapt_inner_optimizer_key),
                            nfev_opt_so_far=int(seed_cobyla_nfev_so_far),
                            best_fun=float(seed_cobyla_best_fun),
                            delta_abs_best=(
                                float(abs(seed_cobyla_best_fun - exact_gs))
                                if math.isfinite(seed_cobyla_best_fun)
                                else None
                            ),
                            elapsed_opt_s=float(now - seed_opt_t0),
                        )
                        seed_cobyla_last_hb_t = now
                return float(seed_energy_val)

            seed_maxiter = int(max(100, min(int(maxiter), 600)))
            if adapt_inner_optimizer_key == "SPSA":
                seed_last_hb_t = seed_opt_t0

                def _seed_spsa_callback(ev: dict[str, Any]) -> None:
                    nonlocal seed_last_hb_t
                    now = time.perf_counter()
                    if (now - seed_last_hb_t) < float(adapt_spsa_progress_every_s):
                        return
                    seed_best = float(ev.get("best_fun", float("nan")))
                    _ai_log(
                        "hardcoded_adapt_spsa_heartbeat",
                        stage="hh_seed_preopt",
                        depth=0,
                        iter=int(ev.get("iter", 0)),
                        nfev_opt_so_far=int(ev.get("nfev_so_far", 0)),
                        best_fun=seed_best,
                        delta_abs_best=float(abs(seed_best - exact_gs)) if math.isfinite(seed_best) else None,
                        elapsed_opt_s=float(now - seed_opt_t0),
                    )
                    seed_last_hb_t = now

                seed_result = spsa_minimize(
                    fun=_seed_obj,
                    x0=theta_seed0,
                    maxiter=int(seed_maxiter),
                    seed=int(seed) + 90000,
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
                    callback=_seed_spsa_callback,
                    callback_every=int(adapt_spsa_callback_every),
                )
                seed_theta = np.asarray(seed_result.x, dtype=float)
                seed_energy = float(seed_result.fun)
                seed_nfev = int(seed_result.nfev)
                seed_nit = int(seed_result.nit)
                seed_success = bool(seed_result.success)
                seed_message = str(seed_result.message)
            else:
                if scipy_minimize is None:
                    raise RuntimeError(
                        f"SciPy minimize is unavailable for {adapt_inner_optimizer_key} ADAPT inner optimizer."
                    )
                seed_result = scipy_minimize(
                    _seed_obj,
                    theta_seed0,
                    method=str(adapt_inner_optimizer_key),
                    options=_scipy_inner_options(int(seed_maxiter)),
                )
                seed_theta = np.asarray(seed_result.x, dtype=float)
                seed_energy = float(seed_result.fun)
                seed_nfev = int(getattr(seed_result, "nfev", 0))
                seed_nit = int(getattr(seed_result, "nit", 0))
                seed_success = bool(getattr(seed_result, "success", False))
                seed_message = str(getattr(seed_result, "message", ""))
            nfev_total += int(seed_nfev)

            selected_ops = list(seed_ops)
            selected_layout = _build_selected_layout(selected_ops)
            theta = np.asarray(seed_theta, dtype=float)
            if phase2_enabled:
                if adapt_inner_optimizer_key == "SPSA":
                    phase2_optimizer_memory = phase2_memory_adapter.from_result(
                        seed_result,
                        method=str(adapt_inner_optimizer_key),
                        parameter_count=int(theta.size),
                        source="hh_seed_preopt",
                    )
                else:
                    phase2_optimizer_memory = phase2_memory_adapter.unavailable(
                        method=str(adapt_inner_optimizer_key),
                        parameter_count=int(theta.size),
                        reason="non_spsa_seed_preopt",
                    )
            if not allow_repeats:
                for idx in seed_indices:
                    available_indices.discard(idx)
            energy_current = float(seed_energy)
            selected_executor = (
                seed_executor
                if seed_executor is not None
                else None
            )
            _ai_log(
                "hardcoded_adapt_hh_seed_preopt",
                num_seed_ops=int(len(seed_ops)),
                seed_opt_method=str(adapt_inner_optimizer_key),
                seed_energy=float(seed_energy),
                seed_nfev=int(seed_nfev),
                seed_nit=int(seed_nit),
                seed_success=bool(seed_success),
                seed_message=str(seed_message),
            )
            if phase1_enabled:
                phase1_stage.begin_core()
                phase1_stage_events.append(
                    {
                        "depth": 0,
                        "stage_name": "seed",
                        "reason": "seed_complete",
                        "num_seed_ops": int(len(seed_ops)),
                    }
                )
    if phase1_enabled and phase1_stage.stage_name == "seed":
        phase1_stage.begin_core()
        phase1_stage_events.append(
            {
                "depth": 0,
                "stage_name": "seed",
                "reason": "seed_skipped_or_empty",
                "num_seed_ops": 0,
            }
        )
    if phase2_enabled and int(theta.size) > 0 and int(phase2_optimizer_memory.get("parameter_count", 0)) != int(theta.size):
        phase2_optimizer_memory = phase2_memory_adapter.unavailable(
            method=str(adapt_inner_optimizer_key),
            parameter_count=int(theta.size),
            reason="post_seed_memory_resize",
        )

    if phase3_enabled and isinstance(phase3_input_motif_library, Mapping):
        motif_seed_records = select_tiled_generators_from_library(
            motif_library=phase3_input_motif_library,
            registry_by_label=pool_generator_registry,
            target_num_sites=int(num_sites),
            excluded_labels=[str(op.label) for op in selected_ops],
            max_seed=4,
        )
        label_to_indices: dict[str, list[int]] = {}
        for idx_pool, op_pool in enumerate(pool):
            label_to_indices.setdefault(str(op_pool.label), []).append(int(idx_pool))
        seeded_labels_now: list[str] = []
        seeded_generator_ids_now: list[str] = []
        seeded_motif_ids_now: list[str] = []
        for rec in motif_seed_records:
            label_seed = str(rec.get("candidate_label", ""))
            idx_list = label_to_indices.get(label_seed, [])
            idx_seed = None
            for idx_candidate in idx_list:
                if bool(allow_repeats) or int(idx_candidate) in available_indices:
                    idx_seed = int(idx_candidate)
                    break
            if idx_seed is None:
                continue
            if phase2_enabled:
                phase2_optimizer_memory = phase2_memory_adapter.remap_insert(
                    phase2_optimizer_memory,
                    position_id=int(theta.size),
                    count=1,
                )
            selected_ops, theta = _splice_candidate_at_position(
                ops=selected_ops,
                theta=np.asarray(theta, dtype=float),
                op=pool[int(idx_seed)],
                position_id=int(theta.size),
                init_theta=0.0,
            )
            selection_counts[int(idx_seed)] += 1
            if not allow_repeats:
                available_indices.discard(int(idx_seed))
            seeded_labels_now.append(str(label_seed))
            generator_meta = rec.get("generator_metadata", None)
            if isinstance(generator_meta, Mapping) and generator_meta.get("generator_id") is not None:
                seeded_generator_ids_now.append(str(generator_meta.get("generator_id")))
            motif_meta = rec.get("motif_metadata", None)
            if isinstance(motif_meta, Mapping):
                for motif_id in motif_meta.get("motif_ids", []):
                    seeded_motif_ids_now.append(str(motif_id))
        if seeded_labels_now:
            phase3_motif_usage["seeded_labels"] = [str(x) for x in seeded_labels_now]
            phase3_motif_usage["seeded_generator_ids"] = [str(x) for x in seeded_generator_ids_now]
            phase3_motif_usage["seeded_motif_ids"] = [str(x) for x in seeded_motif_ids_now]
            phase1_stage_events.append(
                {
                    "depth": 0,
                    "stage_name": str(phase1_stage.stage_name if phase1_enabled else "legacy"),
                    "reason": "motif_seed_injected",
                    "seeded_labels": [str(x) for x in seeded_labels_now],
                }
            )
            if adapt_state_backend_key == "compiled":
                selected_executor = _build_compiled_executor(selected_ops)
            _ai_log(
                "hardcoded_adapt_phase3_motif_seeded",
                seeded_count=int(len(seeded_labels_now)),
                seeded_labels=[str(x) for x in seeded_labels_now],
                source_tag=str(phase3_motif_usage.get("source_tag")),
            )

    rescue_cfg = RescueConfig(enabled=bool(phase3_enable_rescue))

    def _phase3_try_rescue(
        *,
        psi_current_state: np.ndarray,
        shortlist_eval_records: list[dict[str, Any]],
        selected_position_append: int,
        history_rows: list[dict[str, Any]],
        trough_detected_now: bool,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        diagnostic = {
            "enabled": bool(phase3_enable_rescue),
            "triggered": False,
            "reason": "disabled",
            "ranked": [],
            "selected_label": None,
            "selected_position": None,
            "overlap_gain": 0.0,
        }
        trigger_on, trigger_reason = should_trigger_rescue(
            enabled=bool(phase3_enable_rescue),
            exact_state_available=bool(phase3_exact_reference_state is not None),
            residual_opened=bool(phase1_residual_opened),
            trough_detected=bool(trough_detected_now),
            history=history_rows,
            shortlist_records=shortlist_eval_records,
            cfg=rescue_cfg,
        )
        diagnostic["reason"] = str(trigger_reason)
        if not bool(trigger_on):
            return None, diagnostic
        diagnostic["triggered"] = True
        psi_exact_ref = np.asarray(phase3_exact_reference_state, dtype=complex)
        overlap_current = float(abs(np.vdot(psi_exact_ref, np.asarray(psi_current_state, dtype=complex))) ** 2)
        probe_theta_by_key: dict[tuple[int, int], float] = {}

        def _overlap_gain(rec: Mapping[str, Any]) -> float:
            idx_sel = int(rec.get("candidate_pool_index", -1))
            pos_sel = int(rec.get("position_id", selected_position_append))
            if idx_sel < 0 or idx_sel >= len(pool):
                return 0.0
            best_gain_local = 0.0
            best_theta_local = 0.0
            for theta_probe in (float(finite_angle), -float(finite_angle)):
                ops_trial, theta_trial = _splice_candidate_at_position(
                    ops=selected_ops,
                    theta=np.asarray(theta, dtype=float),
                    op=pool[int(idx_sel)],
                    position_id=int(pos_sel),
                    init_theta=float(theta_probe),
                )
                if adapt_state_backend_key == "compiled":
                    psi_trial = _build_compiled_executor(ops_trial).prepare_state(theta_trial, psi_ref)
                else:
                    psi_trial = _prepare_adapt_state(
                        psi_ref,
                        ops_trial,
                        theta_trial,
                        parameter_layout=_build_selected_layout(ops_trial),
                    )
                overlap_trial = float(abs(np.vdot(psi_exact_ref, np.asarray(psi_trial, dtype=complex))) ** 2)
                gain = float(overlap_trial - overlap_current)
                if gain > best_gain_local:
                    best_gain_local = float(gain)
                    best_theta_local = float(theta_probe)
            probe_theta_by_key[(int(idx_sel), int(pos_sel))] = float(best_theta_local)
            return float(best_gain_local)

        best_record, rescue_meta = rank_rescue_candidates(
            records=shortlist_eval_records,
            overlap_gain_fn=_overlap_gain,
            cfg=rescue_cfg,
        )
        diagnostic.update(
            {
                "reason": str(rescue_meta.get("reason", trigger_reason)),
                "ranked": [dict(x) for x in rescue_meta.get("ranked", [])],
            }
        )
        if best_record is None:
            return None, diagnostic
        idx_best = int(best_record.get("candidate_pool_index", -1))
        pos_best = int(best_record.get("position_id", selected_position_append))
        diagnostic.update(
            {
                "selected_label": str(best_record.get("candidate_label", "")),
                "selected_position": int(pos_best),
                "overlap_gain": float(best_record.get("overlap_gain", 0.0)),
                "init_theta": float(probe_theta_by_key.get((int(idx_best), int(pos_best)), 0.0)),
            }
        )
        best_out = dict(best_record)
        best_out["rescue_init_theta"] = float(probe_theta_by_key.get((int(idx_best), int(pos_best)), 0.0))
        return best_out, diagnostic

    for depth in range(int(max_depth)):
        iter_t0 = time.perf_counter()

        # 1) Compute the current state
        if adapt_state_backend_key == "compiled":
            if len(selected_ops) == 0:
                psi_current = np.array(psi_ref, copy=True)
            else:
                if selected_executor is None:
                    selected_executor = _build_compiled_executor(selected_ops)
                psi_current = selected_executor.prepare_state(theta, psi_ref)
        else:
            psi_current = _prepare_adapt_state(psi_ref, selected_ops, theta, parameter_layout=selected_layout)
        energy_current, hpsi_current = energy_via_one_apply(psi_current, h_compiled)
        energy_current = float(energy_current)
        theta_logical_current = _logical_theta_alias(theta, selected_layout)
        backend_compile_snapshot = (
            backend_compile_oracle.snapshot_base(selected_ops)
            if backend_compile_oracle is not None
            else None
        )

        # 2) Compute commutator gradients for all pool operators
        gradient_eval_t0 = time.perf_counter()
        gradients = np.zeros(len(pool), dtype=float)
        grad_magnitudes = np.zeros(len(pool), dtype=float)
        for i in available_indices:
            apsi = _apply_compiled_polynomial(psi_current, pool_compiled[i])
            gradients[i] = adapt_commutator_grad_from_hpsi(hpsi_current, apsi)
            grad_magnitudes[i] = abs(float(gradients[i]))
        if bool(adapt_gradient_parity_check) and available_indices:
            parity_idx = max(available_indices, key=lambda idx: grad_magnitudes[int(idx)])
            grad_old = _commutator_gradient(
                h_poly,
                pool[int(parity_idx)],
                psi_current,
                h_compiled=h_compiled,
                pool_compiled=pool_compiled[int(parity_idx)],
            )
            grad_new = float(gradients[int(parity_idx)])
            rel_err = abs(grad_new - grad_old) / max(abs(grad_new), abs(grad_old), 1e-15)
            if rel_err > float(_ADAPT_GRADIENT_PARITY_RTOL):
                raise AssertionError(
                    "ADAPT gradient parity check failed: "
                    f"depth={depth + 1}, idx={int(parity_idx)}, grad_new={grad_new:.16e}, "
                    f"grad_old={float(grad_old):.16e}, rel_err={float(rel_err):.3e}, "
                    f"rtol={_ADAPT_GRADIENT_PARITY_RTOL:.1e}"
                )
        gradient_eval_elapsed_s = float(time.perf_counter() - gradient_eval_t0)
        _ai_log(
            "hardcoded_adapt_gradient_timing",
            depth=int(depth + 1),
            available_count=int(len(available_indices)),
            gradient_eval_elapsed_s=float(gradient_eval_elapsed_s),
        )

        # 2b) Select candidate (legacy argmax or phase1_v1 simple score).
        selected_position = int(len(selected_ops))
        stage_name = "legacy"
        phase1_feature_selected: dict[str, Any] | None = None
        phase1_stage_transition_reason = "legacy"
        append_position = int(len(selected_ops))
        phase1_append_best_score = float("-inf")
        phase2_selected_records: list[dict[str, Any]] = []
        phase2_last_shortlist_records = []
        phase2_last_shortlist_eval_records = []
        phase2_last_batch_selected = False
        phase2_last_batch_penalty_total = 0.0
        if available_indices:
            max_grad = float(max(float(grad_magnitudes[i]) for i in available_indices))
        else:
            max_grad = 0.0
        if phase1_enabled and available_indices:
            stage_name = str(phase1_stage.stage_name)
            append_position = int(theta.size)
            available_sorted = sorted(list(available_indices), key=lambda i: -float(grad_magnitudes[i]))
            shortlist = available_sorted[: min(len(available_sorted), int(phase1_shortlist_size_val))]
            current_active_window_for_probe, _probe_window_name = _resolve_reopt_active_indices(
                policy=str(adapt_reopt_policy_key),
                n=int(max(1, append_position)),
                theta=(np.asarray(theta_logical_current, dtype=float) if append_position > 0 else np.zeros(1, dtype=float)),
                window_size=int(adapt_window_size_val),
                window_topk=int(adapt_window_topk_val),
                periodic_full_refit_triggered=False,
            )

            def _evaluate_phase1_positions(
                positions_considered_local: list[int],
                *,
                trough_probe_triggered_local: bool,
            ) -> dict[str, Any]:
                best_score_local = float("-inf")
                best_idx_local = int(shortlist[0]) if shortlist else int(max(available_indices))
                best_position_local = int(append_position)
                best_feat_local: dict[str, Any] | None = None
                append_best_score_local = float("-inf")
                append_best_g_lcb_local = 0.0
                append_best_family_local = ""
                best_non_append_score_local = float("-inf")
                best_non_append_g_lcb_local = 0.0
                records_local: list[dict[str, Any]] = []
                for idx in shortlist:
                    for pos in positions_considered_local:
                        active_window_guess = _predict_reopt_window_for_position(
                            theta=np.asarray(theta_logical_current, dtype=float),
                            position_id=int(pos),
                            policy=str(adapt_reopt_policy_key),
                            window_size=int(adapt_window_size_val),
                            window_topk=int(adapt_window_topk_val),
                            periodic_full_refit_triggered=False,
                        )
                        proxy_compile_est = phase1_compile_oracle.estimate(
                            candidate_term_count=int(len(pool_compiled[int(idx)].terms)),
                            position_id=int(pos),
                            append_position=int(append_position),
                            refit_active_count=int(len(active_window_guess)),
                            candidate_term=pool[int(idx)],
                        )
                        compile_est = (
                            backend_compile_oracle.estimate_insertion(
                                backend_compile_snapshot,
                                candidate_term=pool[int(idx)],
                                position_id=int(pos),
                                proxy_baseline=proxy_compile_est,
                            )
                            if backend_compile_oracle is not None and backend_compile_snapshot is not None
                            else proxy_compile_est
                        )
                        meas_stats = phase1_measure_cache.estimate(
                            measurement_group_keys_for_term(pool[int(idx)])
                        )
                        is_residual_candidate = bool(int(idx) in phase1_residual_indices)
                        stage_gate_open = (
                            (stage_name == "residual")
                            or (not is_residual_candidate)
                        )
                        generator_meta = (
                            pool_generator_registry.get(str(pool[int(idx)].label))
                            if phase3_enabled
                            else None
                        )
                        symmetry_spec = (
                            pool_symmetry_specs[int(idx)]
                            if phase3_enabled and int(idx) < len(pool_symmetry_specs)
                            else None
                        )
                        feat_obj = build_candidate_features(
                            stage_name=str(stage_name),
                            candidate_label=str(pool[int(idx)].label),
                            candidate_family=str(pool_family_ids[int(idx)]),
                            candidate_pool_index=int(idx),
                            position_id=int(pos),
                            append_position=int(append_position),
                            positions_considered=[int(x) for x in positions_considered_local],
                            gradient_signed=float(gradients[int(idx)]),
                            metric_proxy=float(abs(float(gradients[int(idx)]))),
                            sigma_hat=0.0,
                            refit_window_indices=[int(i) for i in active_window_guess],
                            compile_cost=compile_est,
                            measurement_stats=meas_stats,
                            leakage_penalty=(
                                float(leakage_penalty_from_spec(symmetry_spec))
                                if phase3_enabled
                                else 0.0
                            ),
                            stage_gate_open=bool(stage_gate_open),
                            leakage_gate_open=not bool(
                                isinstance(symmetry_spec, Mapping) and symmetry_spec.get("hard_guard", False)
                            ),
                            trough_probe_triggered=bool(trough_probe_triggered_local),
                            trough_detected=False,
                            cfg=phase1_score_cfg,
                            generator_metadata=(dict(generator_meta) if isinstance(generator_meta, Mapping) else None),
                            symmetry_spec=(dict(symmetry_spec) if isinstance(symmetry_spec, Mapping) else None),
                            symmetry_mode=("shared_phase3_spec" if phase3_enabled else "none"),
                            symmetry_mitigation_mode=str(phase3_symmetry_mitigation_mode_key if phase3_enabled else "off"),
                            current_depth=int(depth),
                            max_depth=int(max_depth),
                            lifetime_cost_mode=(
                                str(phase3_lifetime_cost_mode_key)
                                if phase3_enabled
                                else "off"
                            ),
                            remaining_evaluations_proxy_mode=(
                                "remaining_depth"
                                if phase3_enabled and str(phase3_lifetime_cost_mode_key) != "off"
                                else "none"
                            ),
                        )
                        window_terms, window_labels = _window_terms_for_position(
                            selected_ops=list(selected_ops),
                            refit_window_indices=[int(i) for i in active_window_guess],
                            position_id=int(pos),
                        )
                        feat = dict(feat_obj.__dict__)
                        score_val = float(feat.get("simple_score", float("-inf")))
                        records_local.append(
                            {
                                "feature": feat_obj,
                                "simple_score": float(score_val),
                                "candidate_pool_index": int(idx),
                                "position_id": int(pos),
                                "candidate_term": pool[int(idx)],
                                "window_terms": list(window_terms),
                                "window_labels": list(window_labels),
                            }
                        )
                        if int(pos) == int(append_position) and score_val > append_best_score_local:
                            append_best_score_local = float(score_val)
                            append_best_g_lcb_local = float(feat.get("g_lcb", 0.0))
                            append_best_family_local = str(feat.get("candidate_family", ""))
                        if int(pos) != int(append_position) and score_val > best_non_append_score_local:
                            best_non_append_score_local = float(score_val)
                            best_non_append_g_lcb_local = float(feat.get("g_lcb", 0.0))
                        if score_val > best_score_local:
                            best_score_local = float(score_val)
                            best_idx_local = int(idx)
                            best_position_local = int(pos)
                            best_feat_local = dict(feat)
                return {
                    "best_score": float(best_score_local),
                    "best_idx": int(best_idx_local),
                    "best_position": int(best_position_local),
                    "best_feat": (dict(best_feat_local) if isinstance(best_feat_local, dict) else None),
                    "append_best_score": float(append_best_score_local),
                    "append_best_g_lcb": float(append_best_g_lcb_local),
                    "append_best_family": str(append_best_family_local),
                    "best_non_append_score": float(best_non_append_score_local),
                    "best_non_append_g_lcb": float(best_non_append_g_lcb_local),
                    "records": list(records_local),
                }

            append_eval = _evaluate_phase1_positions(
                [int(append_position)],
                trough_probe_triggered_local=False,
            )
            phase1_append_best_score = float(append_eval["append_best_score"])
            repeated_family_flat = _phase1_repeated_family_flat(
                history=history,
                candidate_family=str(append_eval["append_best_family"]),
                patience=int(phase1_stage_cfg.family_repeat_patience),
                weak_drop_threshold=float(phase1_stage_cfg.weak_drop_threshold),
            )
            probe_on, probe_reason = should_probe_positions(
                stage_name=str(stage_name),
                drop_plateau_hits=int(drop_plateau_hits),
                max_grad=float(max_grad),
                eps_grad=float(eps_grad),
                append_score=float(phase1_append_best_score),
                finite_angle_flat=False,
                repeated_family_flat=bool(repeated_family_flat),
                cfg=phase1_stage_cfg,
            )
            positions_considered = [int(append_position)]
            score_eval = append_eval
            if probe_on:
                positions_considered = allowed_positions(
                    n_params=int(theta.size),
                    append_position=int(append_position),
                    active_window_indices=[int(i) for i in current_active_window_for_probe],
                    max_positions=int(phase1_stage_cfg.max_probe_positions),
                )
                score_eval = _evaluate_phase1_positions(
                    [int(x) for x in positions_considered],
                    trough_probe_triggered_local=True,
                )
            if backend_compile_oracle is not None:
                finite_phase1_records = [
                    rec for rec in score_eval.get("records", [])
                    if math.isfinite(float(rec.get("simple_score", float("-inf"))))
                ]
                if not finite_phase1_records:
                    stop_reason = "backend_compile_exhausted"
                    break
            trough = detect_trough(
                append_score=float(score_eval["append_best_score"]),
                best_non_append_score=float(score_eval["best_non_append_score"]),
                best_non_append_g_lcb=float(score_eval["best_non_append_g_lcb"]),
                margin_ratio=float(phase1_stage_cfg.probe_margin_ratio),
                append_admit_threshold=float(phase1_stage_cfg.append_admit_threshold),
            )
            phase1_last_probe_reason = str(probe_reason)
            phase1_last_positions_considered = [int(x) for x in positions_considered]
            phase1_last_trough_detected = bool(trough)
            phase1_last_trough_probe_triggered = bool(probe_on)
            phase1_last_selected_score = float(score_eval["best_score"])
            best_feat = score_eval["best_feat"]
            best_idx = int(score_eval["best_idx"])
            selected_position = int(score_eval["best_position"])
            selection_mode = "simple_v1_probe" if bool(probe_on) else "simple_v1"
            if phase2_enabled:
                cheap_records = shortlist_records(
                    [
                        {
                            **dict(rec),
                            "feature": rec["feature"],
                            "simple_score": float(rec.get("simple_score", float("-inf"))),
                            "candidate_pool_index": int(rec.get("candidate_pool_index", -1)),
                            "position_id": int(rec.get("position_id", append_position)),
                        }
                        for rec in score_eval.get("records", [])
                    ],
                    cfg=phase2_score_cfg,
                    score_key="simple_score",
                )
                full_records: list[dict[str, Any]] = []
                for rec in cheap_records:
                    feat_base = rec.get("feature")
                    if not isinstance(feat_base, CandidateFeatures):
                        continue
                    window_terms = list(rec.get("window_terms", []))
                    window_labels = [str(x) for x in rec.get("window_labels", [])]
                    parent_label = str(rec.get("candidate_term").label)
                    parent_generator_meta = (
                        dict(feat_base.generator_metadata)
                        if isinstance(feat_base.generator_metadata, Mapping)
                        else (
                            dict(pool_generator_registry.get(parent_label, {}))
                            if phase3_enabled and isinstance(pool_generator_registry.get(parent_label), Mapping)
                            else None
                        )
                    )
                    parent_symmetry_spec = (
                        dict(feat_base.symmetry_spec)
                        if isinstance(feat_base.symmetry_spec, Mapping)
                        else (
                            dict(pool_symmetry_specs[int(feat_base.candidate_pool_index)])
                            if phase3_enabled
                            and int(feat_base.candidate_pool_index) < len(pool_symmetry_specs)
                            and isinstance(pool_symmetry_specs[int(feat_base.candidate_pool_index)], Mapping)
                            else None
                        )
                    )

                    def _full_record_for_candidate(
                        *,
                        candidate_term: AnsatzTerm,
                        candidate_label: str,
                        generator_metadata: Mapping[str, Any] | None,
                        symmetry_spec_candidate: Mapping[str, Any] | None,
                        runtime_split_mode_value: str = "off",
                        runtime_split_parent_label_value: str | None = None,
                        runtime_split_child_index_value: int | None = None,
                        runtime_split_child_count_value: int | None = None,
                        runtime_split_chosen_representation_value: str = "parent",
                        runtime_split_child_indices_value: Sequence[int] | None = None,
                        runtime_split_child_labels_value: Sequence[str] | None = None,
                        runtime_split_child_generator_ids_value: Sequence[str] | None = None,
                    ) -> dict[str, Any]:
                        compiled_candidate = phase2_compiled_term_cache.get(str(candidate_label))
                        if compiled_candidate is None:
                            compiled_candidate = _compile_polynomial_action(
                                candidate_term.polynomial,
                                pauli_action_cache=pauli_action_cache,
                            )
                            phase2_compiled_term_cache[str(candidate_label)] = compiled_candidate
                        grad_candidate = float(
                            adapt_commutator_grad_from_hpsi(
                                hpsi_current,
                                _apply_compiled_polynomial(
                                    np.asarray(psi_current, dtype=complex),
                                    compiled_candidate,
                                ),
                            )
                        )
                        proxy_compile_est_candidate = phase1_compile_oracle.estimate(
                            candidate_term_count=int(len(compiled_candidate.terms)),
                            position_id=int(feat_base.position_id),
                            append_position=int(feat_base.append_position),
                            refit_active_count=int(len(feat_base.refit_window_indices)),
                            candidate_term=candidate_term,
                        )
                        compile_est_candidate = (
                            backend_compile_oracle.estimate_insertion(
                                backend_compile_snapshot,
                                candidate_term=candidate_term,
                                position_id=int(feat_base.position_id),
                                proxy_baseline=proxy_compile_est_candidate,
                            )
                            if backend_compile_oracle is not None and backend_compile_snapshot is not None
                            else proxy_compile_est_candidate
                        )
                        measurement_stats_candidate = phase1_measure_cache.estimate(
                            measurement_group_keys_for_term(candidate_term)
                        )
                        feat_candidate_base = build_candidate_features(
                            stage_name=str(feat_base.stage_name),
                            candidate_label=str(candidate_label),
                            candidate_family=str(feat_base.candidate_family),
                            candidate_pool_index=int(feat_base.candidate_pool_index),
                            position_id=int(feat_base.position_id),
                            append_position=int(feat_base.append_position),
                            positions_considered=[int(x) for x in feat_base.positions_considered],
                            gradient_signed=float(grad_candidate),
                            metric_proxy=float(abs(grad_candidate)),
                            sigma_hat=float(feat_base.sigma_hat),
                            refit_window_indices=[int(i) for i in feat_base.refit_window_indices],
                            compile_cost=compile_est_candidate,
                            measurement_stats=measurement_stats_candidate,
                            leakage_penalty=(
                                float(leakage_penalty_from_spec(symmetry_spec_candidate))
                                if isinstance(symmetry_spec_candidate, Mapping)
                                else float(feat_base.leakage_penalty)
                            ),
                            stage_gate_open=bool(feat_base.stage_gate_open),
                            leakage_gate_open=bool(feat_base.leakage_gate_open),
                            trough_probe_triggered=bool(feat_base.trough_probe_triggered),
                            trough_detected=bool(feat_base.trough_detected),
                            cfg=phase1_score_cfg,
                            generator_metadata=(
                                dict(generator_metadata) if isinstance(generator_metadata, Mapping) else None
                            ),
                            symmetry_spec=(
                                dict(symmetry_spec_candidate)
                                if isinstance(symmetry_spec_candidate, Mapping)
                                else None
                            ),
                            symmetry_mode=str(feat_base.symmetry_mode),
                            symmetry_mitigation_mode=str(feat_base.symmetry_mitigation_mode),
                            motif_metadata=(
                                dict(feat_base.motif_metadata)
                                if isinstance(feat_base.motif_metadata, Mapping)
                                else None
                            ),
                            motif_bonus=float(feat_base.motif_bonus or 0.0),
                            motif_source=str(feat_base.motif_source),
                            current_depth=int(depth),
                            max_depth=int(max_depth),
                            lifetime_cost_mode=str(feat_base.lifetime_cost_mode),
                            remaining_evaluations_proxy_mode=str(feat_base.remaining_evaluations_proxy_mode),
                        )
                        if str(runtime_split_mode_value) != "off":
                            feat_candidate_base = CandidateFeatures(
                                **{
                                    **feat_candidate_base.__dict__,
                                    "runtime_split_mode": str(runtime_split_mode_value),
                                    "runtime_split_parent_label": (
                                        str(runtime_split_parent_label_value)
                                        if runtime_split_parent_label_value is not None
                                        else None
                                    ),
                                    "runtime_split_child_index": (
                                        int(runtime_split_child_index_value)
                                        if runtime_split_child_index_value is not None
                                        else None
                                    ),
                                    "runtime_split_child_count": (
                                        int(runtime_split_child_count_value)
                                        if runtime_split_child_count_value is not None
                                        else None
                                    ),
                                    "runtime_split_chosen_representation": str(
                                        runtime_split_chosen_representation_value
                                    ),
                                    "runtime_split_child_indices": (
                                        [int(x) for x in runtime_split_child_indices_value]
                                        if runtime_split_child_indices_value is not None
                                        else []
                                    ),
                                    "runtime_split_child_labels": (
                                        [str(x) for x in runtime_split_child_labels_value]
                                        if runtime_split_child_labels_value is not None
                                        else []
                                    ),
                                    "runtime_split_child_generator_ids": (
                                        [str(x) for x in runtime_split_child_generator_ids_value]
                                        if runtime_split_child_generator_ids_value is not None
                                        else []
                                    ),
                                }
                            )
                        active_memory = phase2_memory_adapter.select_active(
                            phase2_optimizer_memory,
                            active_indices=list(feat_candidate_base.refit_window_indices),
                            source=f"adapt.depth{int(depth + 1)}.window_subset",
                        )
                        feat_full = build_full_candidate_features(
                            base_feature=feat_candidate_base,
                            psi_state=np.asarray(psi_current, dtype=complex),
                            candidate_term=candidate_term,
                            window_terms=list(window_terms),
                            window_labels=[str(x) for x in window_labels],
                            cfg=phase2_score_cfg,
                            novelty_oracle=phase2_novelty_oracle,
                            curvature_oracle=phase2_curvature_oracle,
                            compiled_cache=phase2_compiled_term_cache,
                            pauli_action_cache=pauli_action_cache,
                            optimizer_memory=active_memory,
                            motif_library=(phase3_input_motif_library if phase3_enabled else None),
                            target_num_sites=int(num_sites),
                        )
                        return {
                            **dict(rec),
                            "feature": feat_full,
                            "simple_score": float(feat_full.simple_score or float("-inf")),
                            "full_v2_score": float(feat_full.full_v2_score or float("-inf")),
                            "candidate_pool_index": int(feat_full.candidate_pool_index),
                            "position_id": int(feat_full.position_id),
                            "candidate_term": candidate_term,
                        }

                    parent_record = _full_record_for_candidate(
                        candidate_term=rec["candidate_term"],
                        candidate_label=parent_label,
                        generator_metadata=parent_generator_meta,
                        symmetry_spec_candidate=parent_symmetry_spec,
                    )
                    candidate_variants = [parent_record]
                    if (
                        phase3_enabled
                        and str(phase3_runtime_split_mode_key) == "shortlist_pauli_children_v1"
                        and isinstance(parent_generator_meta, Mapping)
                        and bool(parent_generator_meta.get("is_macro_generator", False))
                    ):
                        split_children = build_runtime_split_children(
                            parent_label=str(parent_label),
                            polynomial=rec["candidate_term"].polynomial,
                            family_id=str(feat_base.candidate_family),
                            num_sites=int(num_sites),
                            ordering=str(ordering),
                            qpb=int(max(1, qpb)),
                            split_mode=str(phase3_runtime_split_mode_key),
                            parent_generator_metadata=parent_generator_meta,
                            symmetry_spec=parent_symmetry_spec,
                        )
                        if split_children:
                            phase3_runtime_split_summary["probed_parent_count"] = int(
                                phase3_runtime_split_summary.get("probed_parent_count", 0)
                            ) + 1
                        split_child_records: list[dict[str, Any]] = []
                        split_child_record_by_generator_id: dict[str, dict[str, Any]] = {}
                        split_child_scores: dict[str, float] = {}
                        for child in split_children:
                            child_label = str(child.get("child_label"))
                            child_poly = child.get("child_polynomial")
                            child_meta = child.get("child_generator_metadata")
                            child_symmetry_gate = (
                                dict(child.get("symmetry_gate", {}))
                                if isinstance(child.get("symmetry_gate"), Mapping)
                                else {}
                            )
                            if not isinstance(child_poly, PauliPolynomial):
                                continue
                            if not isinstance(child_meta, Mapping):
                                continue
                            pool_generator_registry[str(child_label)] = dict(child_meta)
                            phase3_runtime_split_summary["evaluated_child_count"] = int(
                                phase3_runtime_split_summary.get("evaluated_child_count", 0)
                            ) + 1
                            if not bool(child_symmetry_gate.get("passed", True)):
                                phase3_runtime_split_summary["rejected_child_count_symmetry"] = int(
                                    phase3_runtime_split_summary.get("rejected_child_count_symmetry", 0)
                                ) + 1
                            child_record = _full_record_for_candidate(
                                candidate_term=AnsatzTerm(
                                    label=str(child_label),
                                    polynomial=child_poly,
                                ),
                                candidate_label=str(child_label),
                                generator_metadata=dict(child_meta),
                                symmetry_spec_candidate=(
                                    dict(child_meta.get("symmetry_spec", {}))
                                    if isinstance(child_meta.get("symmetry_spec"), Mapping)
                                    else parent_symmetry_spec
                                ),
                                runtime_split_mode_value=str(phase3_runtime_split_mode_key),
                                runtime_split_parent_label_value=str(parent_label),
                                runtime_split_child_index_value=(
                                    int(child.get("child_index"))
                                    if child.get("child_index") is not None
                                    else None
                                ),
                                runtime_split_child_count_value=(
                                    int(child.get("child_count"))
                                    if child.get("child_count") is not None
                                    else None
                                ),
                                runtime_split_chosen_representation_value="child_atom",
                                runtime_split_child_indices_value=(
                                    [int(child.get("child_index"))]
                                    if child.get("child_index") is not None
                                    else []
                                ),
                                runtime_split_child_labels_value=[str(child_label)],
                                runtime_split_child_generator_ids_value=(
                                    [str(child_meta.get("generator_id"))]
                                    if child_meta.get("generator_id") is not None
                                    else []
                                ),
                            )
                            split_child_records.append(dict(child_record))
                            split_child_scores[str(child_label)] = float(
                                child_record.get("full_v2_score", float("-inf"))
                            )
                            if child_meta.get("generator_id") is not None:
                                split_child_record_by_generator_id[str(child_meta.get("generator_id"))] = dict(child_record)
                        split_candidate_record: dict[str, Any] | None = None
                        admissible_child_subsets: list[list[str]] = []
                        best_split_choice_reason = "no_admissible_child_set"
                        best_split_gate_results: dict[str, Any] = {}
                        best_split_child_ids: list[str] = []
                        child_set_candidates = build_runtime_split_child_sets(
                            parent_label=str(parent_label),
                            family_id=str(feat_base.candidate_family),
                            num_sites=int(num_sites),
                            ordering=str(ordering),
                            qpb=int(max(1, qpb)),
                            split_mode=str(phase3_runtime_split_mode_key),
                            children=split_children,
                            parent_generator_metadata=parent_generator_meta,
                            symmetry_spec=parent_symmetry_spec,
                            max_subset_size=3,
                        )
                        phase3_runtime_split_summary["admissible_child_set_count"] = int(
                            phase3_runtime_split_summary.get("admissible_child_set_count", 0)
                        ) + int(len(child_set_candidates))
                        if child_set_candidates and split_child_records:
                            compat_oracle_split = CompatibilityPenaltyOracle(
                                cfg=phase2_score_cfg,
                                psi_state=np.asarray(psi_current, dtype=complex),
                                compiled_cache=phase2_compiled_term_cache,
                                pauli_action_cache=pauli_action_cache,
                            )
                            best_split_proxy = float("-inf")
                            best_split_payload: dict[str, Any] | None = None
                            for child_set in child_set_candidates:
                                child_set_ids = [str(x) for x in child_set.get("child_generator_ids", [])]
                                child_set_records = [
                                    dict(split_child_record_by_generator_id[str(child_id)])
                                    for child_id in child_set_ids
                                    if str(child_id) in split_child_record_by_generator_id
                                ]
                                if len(child_set_records) != len(child_set_ids):
                                    continue
                                admissible_child_subsets.append(
                                    [str(x) for x in child_set.get("child_labels", [])]
                                )
                                penalty_total = 0.0
                                for left_idx in range(len(child_set_records)):
                                    for right_idx in range(left_idx + 1, len(child_set_records)):
                                        penalty_total += float(
                                            compat_oracle_split.penalty(
                                                child_set_records[left_idx],
                                                child_set_records[right_idx],
                                            ).get("total", 0.0)
                                        )
                                proxy_score = float(
                                    sum(
                                        float(rec_child.get("full_v2_score", float("-inf")))
                                        for rec_child in child_set_records
                                    )
                                    - penalty_total
                                )
                                if proxy_score > best_split_proxy:
                                    best_split_proxy = float(proxy_score)
                                    best_split_payload = dict(child_set)
                            if best_split_payload is not None:
                                split_label = str(best_split_payload.get("candidate_label"))
                                split_poly = best_split_payload.get("candidate_polynomial")
                                split_meta = best_split_payload.get("candidate_generator_metadata")
                                if isinstance(split_poly, PauliPolynomial) and isinstance(split_meta, Mapping):
                                    pool_generator_registry[str(split_label)] = dict(split_meta)
                                    best_split_gate_results = (
                                        dict(best_split_payload.get("symmetry_gate", {}))
                                        if isinstance(best_split_payload.get("symmetry_gate"), Mapping)
                                        else {}
                                    )
                                    best_split_child_ids = [
                                        str(x) for x in best_split_payload.get("child_generator_ids", [])
                                    ]
                                    split_candidate_record = _full_record_for_candidate(
                                        candidate_term=AnsatzTerm(
                                            label=str(split_label),
                                            polynomial=split_poly,
                                        ),
                                        candidate_label=str(split_label),
                                        generator_metadata=dict(split_meta),
                                        symmetry_spec_candidate=(
                                            dict(split_meta.get("symmetry_spec", {}))
                                            if isinstance(split_meta.get("symmetry_spec"), Mapping)
                                            else parent_symmetry_spec
                                        ),
                                        runtime_split_mode_value=str(phase3_runtime_split_mode_key),
                                        runtime_split_parent_label_value=str(parent_label),
                                        runtime_split_child_count_value=int(len(split_children)),
                                        runtime_split_chosen_representation_value="child_set",
                                        runtime_split_child_indices_value=[
                                            int(x) for x in best_split_payload.get("child_indices", [])
                                        ],
                                        runtime_split_child_labels_value=[
                                            str(x) for x in best_split_payload.get("child_labels", [])
                                        ],
                                        runtime_split_child_generator_ids_value=list(best_split_child_ids),
                                    )
                                    candidate_variants.append(dict(split_candidate_record))
                                    parent_score = float(parent_record.get("full_v2_score", float("-inf")))
                                    split_score = float(split_candidate_record.get("full_v2_score", float("-inf")))
                                    split_wins = bool(split_score > parent_score)
                                    if split_wins:
                                        phase3_runtime_split_summary["probe_child_set_count"] = int(
                                            phase3_runtime_split_summary.get("probe_child_set_count", 0)
                                        ) + 1
                                        best_split_choice_reason = "child_set_actual_score_better"
                                    else:
                                        phase3_runtime_split_summary["probe_parent_win_count"] = int(
                                            phase3_runtime_split_summary.get("probe_parent_win_count", 0)
                                        ) + 1
                                        best_split_choice_reason = "parent_actual_score_better"
                                    phase3_split_events.append(
                                        build_split_event(
                                            parent_generator_id=str(parent_generator_meta.get("generator_id")),
                                            child_generator_ids=list(best_split_child_ids),
                                            reason=f"depth{int(depth + 1)}_shortlist_probe",
                                            split_mode=str(phase3_runtime_split_mode_key),
                                            probe_trigger="phase2_shortlist",
                                            choice_reason=str(best_split_choice_reason),
                                            parent_score=float(parent_score),
                                            child_scores=dict(split_child_scores),
                                            admissible_child_subsets=list(admissible_child_subsets),
                                            chosen_representation=("child_set" if split_wins else "parent"),
                                            chosen_child_ids=(list(best_split_child_ids) if split_wins else []),
                                            split_margin=float(split_score - parent_score),
                                            symmetry_gate_results=dict(best_split_gate_results),
                                            compiled_cost_parent=float(
                                                parent_record.get("feature").compile_cost_total
                                            )
                                            if isinstance(parent_record.get("feature"), CandidateFeatures)
                                            else None,
                                            compiled_cost_children=float(
                                                split_candidate_record.get("feature").compile_cost_total
                                            )
                                            if isinstance(split_candidate_record.get("feature"), CandidateFeatures)
                                            else None,
                                            insertion_positions=[int(feat_base.position_id)],
                                        )
                                    )
                        elif split_children:
                            phase3_runtime_split_summary["probe_parent_win_count"] = int(
                                phase3_runtime_split_summary.get("probe_parent_win_count", 0)
                            ) + 1
                            phase3_split_events.append(
                                build_split_event(
                                    parent_generator_id=str(parent_generator_meta.get("generator_id")),
                                    child_generator_ids=[
                                        str(meta.get("generator_id"))
                                        for meta in (
                                            child.get("child_generator_metadata")
                                            for child in split_children
                                            if isinstance(child.get("child_generator_metadata"), Mapping)
                                        )
                                        if meta.get("generator_id") is not None
                                    ],
                                    reason=f"depth{int(depth + 1)}_shortlist_probe",
                                    split_mode=str(phase3_runtime_split_mode_key),
                                    probe_trigger="phase2_shortlist",
                                    choice_reason="no_admissible_child_set",
                                    parent_score=float(parent_record.get("full_v2_score", float("-inf"))),
                                    child_scores=dict(split_child_scores),
                                    admissible_child_subsets=[],
                                    chosen_representation="parent",
                                    chosen_child_ids=[],
                                    symmetry_gate_results={"admissible_child_set_count": 0},
                                    compiled_cost_parent=float(
                                        parent_record.get("feature").compile_cost_total
                                    )
                                    if isinstance(parent_record.get("feature"), CandidateFeatures)
                                    else None,
                                    insertion_positions=[int(feat_base.position_id)],
                                )
                            )
                    candidate_variants = sorted(candidate_variants, key=_phase2_record_sort_key)
                    if candidate_variants:
                        full_records.append(dict(candidate_variants[0]))
                full_records = sorted(full_records, key=_phase2_record_sort_key)
                if backend_compile_oracle is not None:
                    finite_full_records = [
                        rec for rec in full_records
                        if math.isfinite(float(rec.get("full_v2_score", float("-inf"))))
                    ]
                    if not finite_full_records:
                        stop_reason = "backend_compile_exhausted"
                        break
                phase2_last_shortlist_eval_records = [dict(rec) for rec in full_records]
                phase2_last_shortlist_records = [
                    dict(rec["feature"].__dict__)
                    for rec in full_records
                    if isinstance(rec.get("feature"), CandidateFeatures)
                ]
                if full_records:
                    if bool(phase2_enable_batching) and str(stage_name) == "core":
                        compat_oracle = CompatibilityPenaltyOracle(
                            cfg=phase2_score_cfg,
                            psi_state=np.asarray(psi_current, dtype=complex),
                            compiled_cache=phase2_compiled_term_cache,
                            pauli_action_cache=pauli_action_cache,
                        )
                        phase2_selected_records, phase2_last_batch_penalty_total = greedy_batch_select(
                            full_records,
                            compat_oracle,
                            phase2_score_cfg,
                        )
                    else:
                        phase2_selected_records = [dict(full_records[0])]
                    phase2_selected_records = sorted(phase2_selected_records, key=_phase2_record_sort_key)
                    phase2_last_batch_selected = bool(len(phase2_selected_records) > 1)
                    top_feat = phase2_selected_records[0].get("feature")
                    if isinstance(top_feat, CandidateFeatures):
                        phase1_feature_selected = dict(top_feat.__dict__)
                        phase1_feature_selected["trough_detected"] = bool(trough)
                        phase1_last_selected_score = float(
                            top_feat.full_v2_score if top_feat.full_v2_score is not None else top_feat.simple_score or float("-inf")
                        )
                        best_idx = int(top_feat.candidate_pool_index)
                        selected_position = int(top_feat.position_id)
                        split_selected = bool(str(top_feat.runtime_split_mode) != "off")
                        selection_mode = (
                            "full_v2_batch_split"
                            if phase2_last_batch_selected and split_selected
                            else (
                                "full_v2_split"
                                if split_selected
                                else ("full_v2_batch" if phase2_last_batch_selected else "full_v2")
                            )
                        )
                elif best_feat is not None:
                    best_feat["trough_detected"] = bool(trough)
                    phase1_feature_selected = dict(best_feat)
            elif best_feat is not None:
                best_feat["trough_detected"] = bool(trough)
                phase1_feature_selected = dict(best_feat)
            phase1_stage_now, phase1_stage_transition_reason = phase1_stage.resolve_stage_transition(
                drop_plateau_hits=int(drop_plateau_hits),
                trough_detected=bool(trough),
                residual_opened=bool(phase1_residual_opened),
            )
            if (
                phase1_stage_now == "residual"
                and (not phase1_residual_opened)
                and len(phase1_residual_indices) > 0
            ):
                phase1_residual_opened = True
                available_indices |= set(int(i) for i in phase1_residual_indices)
                phase1_stage_events.append(
                    {
                        "depth": int(depth + 1),
                        "stage_name": "residual",
                        "reason": str(phase1_stage_transition_reason),
                    }
                )
                _ai_log(
                    "hardcoded_adapt_phase1_residual_opened",
                    depth=int(depth + 1),
                    residual_count=int(len(phase1_residual_indices)),
                )
        else:
            if allow_repeats:
                repeat_bias = 1.5
                scores = grad_magnitudes / (1.0 + repeat_bias * selection_counts.astype(float))
                best_idx = int(np.argmax(scores))
            else:
                best_idx = int(np.argmax(grad_magnitudes))
            selection_mode = "gradient"

        _ai_log(
            "hardcoded_adapt_iter",
            depth=int(depth + 1),
            max_grad=float(max_grad),
            best_op=(
                str(phase2_selected_records[0].get("candidate_term").label)
                if phase1_enabled
                and phase2_enabled
                and len(phase2_selected_records) > 0
                and phase2_selected_records[0].get("candidate_term") is not None
                else (
                    str(phase1_feature_selected.get("candidate_label"))
                    if isinstance(phase1_feature_selected, dict)
                    and phase1_feature_selected.get("candidate_label") is not None
                    else str(pool[best_idx].label)
                )
            ),
            selected_position=int(selected_position),
            stage_name=str(stage_name),
            selection_score=(float(phase1_last_selected_score) if phase1_enabled else None),
            energy=float(energy_current),
        )

        # 3) Check gradient convergence (with optional finite-angle fallback)
        if not phase1_enabled:
            selection_mode = "gradient"
        init_theta = 0.0
        fallback_scan_size = 0
        fallback_best_probe_delta_e = None
        fallback_best_probe_theta = None
        if max_grad < float(eps_grad):
            if bool(finite_angle_fallback) and available_indices:
                fallback_scan_size = int(len(available_indices))
                best_probe_energy = float(energy_current)
                best_probe_idx = None
                best_probe_theta = None
                fallback_executor_cache: dict[int, CompiledAnsatzExecutor] = {}

                for idx in available_indices:
                    trial_ops = selected_ops + [pool[idx]]
                    for trial_theta in (float(finite_angle), -float(finite_angle)):
                        trial_ops, trial_theta_vec = _splice_candidate_at_position(
                            ops=selected_ops,
                            theta=np.asarray(theta, dtype=float),
                            op=pool[int(idx)],
                            position_id=int(append_position),
                            init_theta=float(trial_theta),
                        )
                        if adapt_state_backend_key == "compiled":
                            trial_executor = fallback_executor_cache.get(int(idx))
                            if trial_executor is None:
                                trial_executor = _build_compiled_executor(trial_ops)
                                fallback_executor_cache[int(idx)] = trial_executor
                            psi_trial = trial_executor.prepare_state(trial_theta_vec, psi_ref)
                            probe_energy, _ = energy_via_one_apply(psi_trial, h_compiled)
                            probe_energy = float(probe_energy)
                        else:
                            probe_energy = _adapt_energy_fn(
                                h_poly,
                                psi_ref,
                                trial_ops,
                                trial_theta_vec,
                                h_compiled=h_compiled,
                            )
                        nfev_total += 1
                        if probe_energy < best_probe_energy:
                            best_probe_energy = float(probe_energy)
                            best_probe_idx = int(idx)
                            best_probe_theta = float(trial_theta)

                fallback_best_probe_delta_e = float(best_probe_energy - energy_current)
                fallback_best_probe_theta = float(best_probe_theta) if best_probe_theta is not None else None
                _ai_log(
                    "hardcoded_adapt_fallback_scan",
                    depth=int(depth + 1),
                    scan_size=int(fallback_scan_size),
                    finite_angle=float(finite_angle),
                    best_probe_idx=(int(best_probe_idx) if best_probe_idx is not None else None),
                    best_probe_op=(str(pool[best_probe_idx].label) if best_probe_idx is not None else None),
                    best_probe_delta_e=float(fallback_best_probe_delta_e),
                )

                if (
                    best_probe_idx is not None
                    and (energy_current - best_probe_energy) > float(finite_angle_min_improvement)
                ):
                    best_idx = int(best_probe_idx)
                    selected_position = int(append_position)
                    phase1_feature_selected = None
                    phase2_selected_records = []
                    phase1_last_selected_score = None
                    phase1_last_positions_considered = [int(append_position)]
                    selection_mode = "finite_angle_fallback"
                    init_theta = float(best_probe_theta)
                    _ai_log(
                        "hardcoded_adapt_fallback_selected",
                        depth=int(depth + 1),
                        selected_idx=int(best_idx),
                        selected_op=str(pool[best_idx].label),
                        init_theta=float(init_theta),
                        probe_delta_e=float(fallback_best_probe_delta_e),
                    )
                else:
                    eps_grad_probe_allowed = bool(phase1_enabled and str(stage_name) != "residual")
                    eps_grad_trough = bool(phase1_last_trough_detected and int(selected_position) != int(append_position))
                    if eps_grad_probe_allowed and (not eps_grad_trough):
                        probe_positions_eps = allowed_positions(
                            n_params=int(theta.size),
                            append_position=int(append_position),
                            active_window_indices=[int(i) for i in current_active_window_for_probe],
                            max_positions=int(phase1_stage_cfg.max_probe_positions),
                        )
                        probe_eval_eps = _evaluate_phase1_positions(
                            [int(x) for x in probe_positions_eps],
                            trough_probe_triggered_local=True,
                        )
                        eps_grad_trough = detect_trough(
                            append_score=float(probe_eval_eps["append_best_score"]),
                            best_non_append_score=float(probe_eval_eps["best_non_append_score"]),
                            best_non_append_g_lcb=float(probe_eval_eps["best_non_append_g_lcb"]),
                            margin_ratio=float(phase1_stage_cfg.probe_margin_ratio),
                            append_admit_threshold=float(phase1_stage_cfg.append_admit_threshold),
                        )
                        if bool(eps_grad_trough) and int(probe_eval_eps["best_position"]) != int(append_position):
                            phase1_last_probe_reason = "eps_grad_flat"
                            phase1_last_positions_considered = [int(x) for x in probe_positions_eps]
                            phase1_last_trough_detected = True
                            phase1_last_trough_probe_triggered = True
                            phase1_last_selected_score = float(probe_eval_eps["best_score"])
                            phase1_feature_selected = dict(probe_eval_eps["best_feat"] or {})
                            phase2_selected_records = []
                            if phase1_feature_selected:
                                phase1_feature_selected["trough_detected"] = True
                            best_idx = int(probe_eval_eps["best_idx"])
                            selected_position = int(probe_eval_eps["best_position"])
                            selection_mode = "simple_v1_probe"
                    if not bool(eps_grad_trough):
                        rescue_record = None
                        rescue_diag: dict[str, Any] | None = None
                        if phase3_enabled:
                            rescue_record, rescue_diag = _phase3_try_rescue(
                                psi_current_state=np.asarray(psi_current, dtype=complex),
                                shortlist_eval_records=list(phase2_last_shortlist_eval_records),
                                selected_position_append=int(append_position),
                                history_rows=list(history),
                                trough_detected_now=bool(phase1_last_trough_detected),
                            )
                            if rescue_diag is not None:
                                phase3_rescue_history.append(dict(rescue_diag))
                        if isinstance(rescue_record, Mapping):
                            best_idx = int(rescue_record.get("candidate_pool_index", best_idx))
                            selected_position = int(rescue_record.get("position_id", append_position))
                            init_theta = float(rescue_record.get("rescue_init_theta", finite_angle))
                            selection_mode = "rescue_overlap"
                            feat_rescue = rescue_record.get("feature")
                            phase2_selected_records = [dict(rescue_record)] if phase2_enabled else []
                            if isinstance(feat_rescue, CandidateFeatures):
                                phase1_feature_selected = dict(feat_rescue.__dict__)
                                phase1_feature_selected["actual_fallback_mode"] = "rescue_overlap"
                                phase1_last_selected_score = float(
                                    feat_rescue.full_v2_score
                                    if feat_rescue.full_v2_score is not None
                                    else feat_rescue.simple_score or float("-inf")
                                )
                                if str(feat_rescue.runtime_split_mode) != "off":
                                    selection_mode = "rescue_overlap_split"
                            _ai_log(
                                "hardcoded_adapt_phase3_rescue_selected",
                                depth=int(depth + 1),
                                selected_idx=int(best_idx),
                                selected_op=(
                                    str(rescue_record.get("candidate_term").label)
                                    if rescue_record.get("candidate_term") is not None
                                    else str(pool[int(best_idx)].label)
                                ),
                                selected_position=int(selected_position),
                                init_theta=float(init_theta),
                                overlap_gain=float(rescue_record.get("overlap_gain", 0.0)),
                            )
                        else:
                            if bool(eps_grad_termination_enabled):
                                stop_reason = "eps_grad"
                                _ai_log(
                                    "hardcoded_adapt_converged_grad",
                                    max_grad=float(max_grad),
                                    eps_grad=float(eps_grad),
                                    fallback_attempted=True,
                                    fallback_best_probe_delta_e=float(fallback_best_probe_delta_e),
                                    finite_angle_min_improvement=float(finite_angle_min_improvement),
                                )
                                break
                            selection_mode = "eps_grad_suppressed_continue"
                            _ai_log(
                                "hardcoded_adapt_eps_grad_termination_suppressed",
                                depth=int(depth + 1),
                                max_grad=float(max_grad),
                                eps_grad=float(eps_grad),
                                fallback_attempted=True,
                                fallback_best_probe_delta_e=float(fallback_best_probe_delta_e),
                                finite_angle_min_improvement=float(finite_angle_min_improvement),
                                continuation_mode=str(continuation_mode),
                                problem=str(problem_key),
                            )
            else:
                eps_grad_trough = bool(phase1_enabled and phase1_last_trough_detected and int(selected_position) != int(append_position))
                if not bool(eps_grad_trough):
                    rescue_record = None
                    rescue_diag: dict[str, Any] | None = None
                    if phase3_enabled:
                        rescue_record, rescue_diag = _phase3_try_rescue(
                            psi_current_state=np.asarray(psi_current, dtype=complex),
                            shortlist_eval_records=list(phase2_last_shortlist_eval_records),
                            selected_position_append=int(append_position),
                            history_rows=list(history),
                            trough_detected_now=bool(phase1_last_trough_detected),
                        )
                        if rescue_diag is not None:
                            phase3_rescue_history.append(dict(rescue_diag))
                    if isinstance(rescue_record, Mapping):
                        best_idx = int(rescue_record.get("candidate_pool_index", best_idx))
                        selected_position = int(rescue_record.get("position_id", append_position))
                        init_theta = float(rescue_record.get("rescue_init_theta", finite_angle))
                        selection_mode = "rescue_overlap"
                        feat_rescue = rescue_record.get("feature")
                        phase2_selected_records = [dict(rescue_record)] if phase2_enabled else []
                        if isinstance(feat_rescue, CandidateFeatures):
                            phase1_feature_selected = dict(feat_rescue.__dict__)
                            phase1_feature_selected["actual_fallback_mode"] = "rescue_overlap"
                            phase1_last_selected_score = float(
                                feat_rescue.full_v2_score
                                if feat_rescue.full_v2_score is not None
                                else feat_rescue.simple_score or float("-inf")
                            )
                            if str(feat_rescue.runtime_split_mode) != "off":
                                selection_mode = "rescue_overlap_split"
                        _ai_log(
                            "hardcoded_adapt_phase3_rescue_selected",
                            depth=int(depth + 1),
                            selected_idx=int(best_idx),
                            selected_op=(
                                str(rescue_record.get("candidate_term").label)
                                if rescue_record.get("candidate_term") is not None
                                else str(pool[int(best_idx)].label)
                            ),
                            selected_position=int(selected_position),
                            init_theta=float(init_theta),
                            overlap_gain=float(rescue_record.get("overlap_gain", 0.0)),
                        )
                    else:
                        if bool(eps_grad_termination_enabled):
                            stop_reason = "eps_grad"
                            _ai_log("hardcoded_adapt_converged_grad", max_grad=float(max_grad), eps_grad=float(eps_grad))
                            break
                        selection_mode = "eps_grad_suppressed_continue"
                        _ai_log(
                            "hardcoded_adapt_eps_grad_termination_suppressed",
                            depth=int(depth + 1),
                            max_grad=float(max_grad),
                            eps_grad=float(eps_grad),
                            fallback_attempted=False,
                            continuation_mode=str(continuation_mode),
                            problem=str(problem_key),
                        )

        # 4) Admit selected operator (append or insertion in continuation modes).
        selected_batch_records_for_history: list[dict[str, Any]] = []
        selected_batch_labels: list[str] = []
        selected_batch_positions: list[int] = []
        selected_batch_indices: list[int] = []
        selected_batch_measurement_keys: list[str] = []
        if phase1_enabled and phase2_enabled and len(phase2_selected_records) > 0:
            original_positions_seen: list[int] = []
            for rec in phase2_selected_records:
                feat_rec = rec.get("feature")
                if not isinstance(feat_rec, CandidateFeatures):
                    continue
                idx_sel = int(feat_rec.candidate_pool_index)
                pos_orig = int(feat_rec.position_id)
                pos_eff = int(pos_orig + sum(1 for prev in original_positions_seen if prev <= pos_orig))
                admitted_term = rec.get("candidate_term")
                if not isinstance(admitted_term, AnsatzTerm):
                    admitted_term = pool[int(idx_sel)]
                admitted_layout = _build_selected_layout([admitted_term])
                runtime_insert_pos = int(runtime_insert_position(selected_layout, int(pos_eff)))
                if phase2_enabled:
                    phase2_optimizer_memory = phase2_memory_adapter.remap_insert(
                        phase2_optimizer_memory,
                        position_id=int(runtime_insert_pos),
                        count=int(admitted_layout.runtime_parameter_count),
                    )
                selected_ops, theta = _splice_candidate_at_position(
                    ops=selected_ops,
                    theta=np.asarray(theta, dtype=float),
                    op=admitted_term,
                    position_id=int(pos_eff),
                    init_theta=0.0,
                )
                selected_layout = _build_selected_layout(selected_ops)
                if (
                    phase3_enabled
                    and str(feat_rec.runtime_split_mode) != "off"
                    and feat_rec.parent_generator_id is not None
                ):
                    selected_child_generator_ids = [str(x) for x in feat_rec.runtime_split_child_generator_ids]
                    phase3_split_events.append(
                        build_split_event(
                            parent_generator_id=str(feat_rec.parent_generator_id),
                            child_generator_ids=(
                                list(selected_child_generator_ids)
                                if selected_child_generator_ids
                                else ([str(feat_rec.generator_id)] if feat_rec.generator_id is not None else [])
                            ),
                            reason=f"depth{int(depth + 1)}_selected",
                            split_mode=str(feat_rec.runtime_split_mode),
                            choice_reason="selected_for_admission",
                            chosen_representation=str(feat_rec.runtime_split_chosen_representation),
                            chosen_child_ids=list(selected_child_generator_ids),
                            symmetry_gate_results=(
                                dict(feat_rec.generator_metadata.get("compile_metadata", {}).get("runtime_split", {}).get("symmetry_gate", {}))
                                if isinstance(feat_rec.generator_metadata, Mapping)
                                and isinstance(feat_rec.generator_metadata.get("compile_metadata"), Mapping)
                                and isinstance(
                                    feat_rec.generator_metadata.get("compile_metadata", {}).get("runtime_split"),
                                    Mapping,
                                )
                                else {}
                            ),
                            insertion_positions=[int(pos_eff)],
                        )
                    )
                    if str(feat_rec.runtime_split_chosen_representation) == "child_set":
                        phase3_runtime_split_summary["selected_child_set_count"] = int(
                            phase3_runtime_split_summary.get("selected_child_set_count", 0)
                        ) + 1
                    phase3_runtime_split_summary["selected_child_count"] = int(
                        phase3_runtime_split_summary.get("selected_child_count", 0)
                    ) + int(
                        max(
                            1,
                            len(selected_child_generator_ids)
                            if selected_child_generator_ids
                            else len(feat_rec.runtime_split_child_labels),
                        )
                    )
                    phase3_runtime_split_summary["selected_child_labels"] = [
                        *list(phase3_runtime_split_summary.get("selected_child_labels", [])),
                        *(
                            [str(x) for x in feat_rec.runtime_split_child_labels]
                            if feat_rec.runtime_split_child_labels
                            else [str(admitted_term.label)]
                        ),
                    ]
                original_positions_seen.append(int(pos_orig))
                selection_counts[idx_sel] += 1
                if not allow_repeats:
                    available_indices.discard(idx_sel)
                selected_batch_records_for_history.append(dict(feat_rec.__dict__))
                selected_batch_labels.append(str(admitted_term.label))
                selected_batch_positions.append(int(pos_orig))
                selected_batch_indices.append(int(idx_sel))
                selected_batch_measurement_keys.extend(measurement_group_keys_for_term(admitted_term))
            if selected_batch_indices:
                best_idx = int(selected_batch_indices[0])
                selected_position = int(selected_batch_positions[0])
        elif phase1_enabled:
            admitted_term = pool[int(best_idx)]
            admitted_layout = _build_selected_layout([admitted_term])
            runtime_insert_pos = int(runtime_insert_position(selected_layout, int(selected_position)))
            if phase2_enabled:
                phase2_optimizer_memory = phase2_memory_adapter.remap_insert(
                    phase2_optimizer_memory,
                    position_id=int(runtime_insert_pos),
                    count=int(admitted_layout.runtime_parameter_count),
                )
            selected_ops, theta = _splice_candidate_at_position(
                ops=selected_ops,
                theta=np.asarray(theta, dtype=float),
                op=admitted_term,
                position_id=int(selected_position),
                init_theta=float(init_theta),
            )
            selected_layout = _build_selected_layout(selected_ops)
            selection_counts[best_idx] += 1
            if not allow_repeats:
                available_indices.discard(best_idx)
            if isinstance(phase1_feature_selected, dict):
                selected_batch_records_for_history.append(dict(phase1_feature_selected))
            selected_batch_labels.append(str(pool[int(best_idx)].label))
            selected_batch_positions.append(int(selected_position))
            selected_batch_indices.append(int(best_idx))
            selected_batch_measurement_keys.extend(measurement_group_keys_for_term(pool[int(best_idx)]))
        else:
            selected_ops, theta = _splice_candidate_at_position(
                ops=selected_ops,
                theta=np.asarray(theta, dtype=float),
                op=pool[int(best_idx)],
                position_id=int(len(selected_ops)),
                init_theta=float(init_theta),
            )
            selected_layout = _build_selected_layout(selected_ops)
            selection_counts[best_idx] += 1
            if not allow_repeats:
                available_indices.discard(best_idx)
            selected_batch_labels.append(str(pool[int(best_idx)].label))
            selected_batch_positions.append(int(selected_position))
            selected_batch_indices.append(int(best_idx))
            selected_batch_measurement_keys.extend(measurement_group_keys_for_term(pool[int(best_idx)]))
        if adapt_state_backend_key == "compiled":
            selected_executor = _build_compiled_executor(selected_ops)
        else:
            selected_executor = None

        # 5) Re-optimize parameters with selected inner optimizer
        # Policy: 'full' re-optimizes all parameters (legacy behavior),
        #         'append_only' freezes the prefix and optimizes only the
        #         newly appended parameter.
        energy_prev = energy_current
        theta_before_opt = np.array(theta, copy=True)
        optimizer_t0 = time.perf_counter()
        cobyla_last_hb_t = optimizer_t0
        cobyla_nfev_so_far = 0
        cobyla_best_fun = float("inf")

        def _obj(x: np.ndarray) -> float:
            nonlocal cobyla_last_hb_t, cobyla_nfev_so_far, cobyla_best_fun
            if adapt_state_backend_key == "compiled":
                assert selected_executor is not None
                psi_obj = selected_executor.prepare_state(np.asarray(x, dtype=float), psi_ref)
                energy_obj, _ = energy_via_one_apply(psi_obj, h_compiled)
                energy_obj_val = float(energy_obj)
            else:
                energy_obj_val = _adapt_energy_fn(
                    h_poly,
                    psi_ref,
                    selected_ops,
                    x,
                    h_compiled=h_compiled,
                    parameter_layout=selected_layout,
                )
            if adapt_inner_optimizer_key in {"COBYLA", "POWELL"}:
                cobyla_nfev_so_far += 1
                if energy_obj_val < cobyla_best_fun:
                    cobyla_best_fun = float(energy_obj_val)
                now = time.perf_counter()
                if (now - cobyla_last_hb_t) >= float(adapt_spsa_progress_every_s):
                    _ai_log(
                        "hardcoded_adapt_scipy_heartbeat",
                        stage="depth_opt",
                        depth=int(depth + 1),
                        opt_method=str(adapt_inner_optimizer_key),
                        nfev_opt_so_far=int(cobyla_nfev_so_far),
                        best_fun=float(cobyla_best_fun),
                        delta_abs_best=(
                            float(abs(cobyla_best_fun - exact_gs))
                            if math.isfinite(cobyla_best_fun)
                            else None
                        ),
                        elapsed_opt_s=float(now - optimizer_t0),
                    )
                    cobyla_last_hb_t = now
            return float(energy_obj_val)

        # -- Resolve active reopt indices for this depth --
        n_theta_runtime = int(theta.size)
        theta_logical_selected = _logical_theta_alias(theta, selected_layout)
        n_theta_logical = int(theta_logical_selected.size)
        depth_local = int(depth + 1)
        depth_cumulative = int(adapt_ref_base_depth) + int(depth_local)
        periodic_full_refit_triggered = bool(
            adapt_reopt_policy_key == "windowed"
            and adapt_full_refit_every_val > 0
            and depth_cumulative % adapt_full_refit_every_val == 0
        )
        reopt_active_indices, reopt_policy_effective = _resolve_reopt_active_indices(
            policy=adapt_reopt_policy_key,
            n=n_theta_logical,
            theta=theta_logical_selected,
            window_size=adapt_window_size_val,
            window_topk=adapt_window_topk_val,
            periodic_full_refit_triggered=periodic_full_refit_triggered,
        )
        reopt_runtime_active_indices = runtime_indices_for_logical_indices(
            selected_layout,
            reopt_active_indices,
        )
        if phase1_enabled and isinstance(phase1_feature_selected, dict):
            phase1_feature_selected["refit_window_indices"] = [int(i) for i in reopt_active_indices]
        if phase1_enabled and selected_batch_records_for_history:
            for rec in selected_batch_records_for_history:
                rec["refit_window_indices"] = [int(i) for i in reopt_active_indices]
        _obj_opt, opt_x0 = _make_reduced_objective(theta, reopt_runtime_active_indices, _obj)
        phase2_active_memory = None
        phase2_last_optimizer_memory_reused = False
        phase2_last_optimizer_memory_source = "unavailable"
        if phase2_enabled and adapt_inner_optimizer_key == "SPSA":
            phase2_active_memory = phase2_memory_adapter.select_active(
                phase2_optimizer_memory,
                active_indices=list(reopt_runtime_active_indices),
                source=f"adapt.depth{int(depth + 1)}.opt_active",
            )
            phase2_last_optimizer_memory_reused = bool(phase2_active_memory.get("reused", False))
            phase2_last_optimizer_memory_source = str(phase2_active_memory.get("source", "unavailable"))

        if adapt_inner_optimizer_key == "SPSA":
            spsa_last_hb_t = optimizer_t0

            def _depth_spsa_callback(ev: dict[str, Any]) -> None:
                nonlocal spsa_last_hb_t
                now = time.perf_counter()
                if (now - spsa_last_hb_t) < float(adapt_spsa_progress_every_s):
                    return
                best_fun = float(ev.get("best_fun", float("nan")))
                _ai_log(
                    "hardcoded_adapt_spsa_heartbeat",
                    stage="depth_opt",
                    depth=int(depth + 1),
                    iter=int(ev.get("iter", 0)),
                    nfev_opt_so_far=int(ev.get("nfev_so_far", 0)),
                    best_fun=best_fun,
                    delta_abs_best=float(abs(best_fun - exact_gs)) if math.isfinite(best_fun) else None,
                    elapsed_opt_s=float(now - optimizer_t0),
                )
                spsa_last_hb_t = now

            result = spsa_minimize(
                fun=_obj_opt,
                x0=opt_x0,
                maxiter=int(maxiter),
                seed=int(seed) + int(depth),
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
                callback=_depth_spsa_callback,
                callback_every=int(adapt_spsa_callback_every),
                memory=(dict(phase2_active_memory) if isinstance(phase2_active_memory, Mapping) else None),
                refresh_every=0,
                precondition_mode=("diag_rms_grad" if phase2_enabled else "none"),
            )
            # Reconstruct full theta from reduced optimizer result
            if len(reopt_runtime_active_indices) == n_theta_runtime:
                theta = np.asarray(result.x, dtype=float)
            else:
                result_x = np.asarray(result.x, dtype=float).ravel()
                for k, idx in enumerate(reopt_runtime_active_indices):
                    theta[idx] = float(result_x[k])
            energy_current = float(result.fun)
            nfev_opt = int(result.nfev)
            nit_opt = int(result.nit)
            opt_success = bool(result.success)
            opt_message = str(result.message)
            if phase2_enabled:
                phase2_optimizer_memory = phase2_memory_adapter.merge_active(
                    phase2_optimizer_memory,
                    active_indices=list(reopt_runtime_active_indices),
                    active_state=phase2_memory_adapter.from_result(
                        result,
                        method=str(adapt_inner_optimizer_key),
                        parameter_count=int(len(reopt_runtime_active_indices)),
                        source=f"adapt.depth{int(depth + 1)}.spsa_result",
                    ),
                    source=f"adapt.depth{int(depth + 1)}.merge",
                )
        else:
            if scipy_minimize is None:
                raise RuntimeError(
                    f"SciPy minimize is unavailable for {adapt_inner_optimizer_key} ADAPT inner optimizer."
                )
            result = scipy_minimize(
                _obj_opt,
                opt_x0,
                method=str(adapt_inner_optimizer_key),
                options=_scipy_inner_options(int(maxiter)),
            )
            # Reconstruct full theta from reduced optimizer result
            if len(reopt_runtime_active_indices) == n_theta_runtime:
                theta = np.asarray(result.x, dtype=float)
            else:
                result_x = np.asarray(result.x, dtype=float).ravel()
                for k, idx in enumerate(reopt_runtime_active_indices):
                    theta[idx] = float(result_x[k])
            energy_current = float(result.fun)
            nfev_opt = int(getattr(result, "nfev", 0))
            nit_opt = int(getattr(result, "nit", 0))
            opt_success = bool(getattr(result, "success", False))
            opt_message = str(getattr(result, "message", ""))
            if phase2_enabled:
                phase2_optimizer_memory = phase2_memory_adapter.unavailable(
                    method=str(adapt_inner_optimizer_key),
                    parameter_count=int(theta.size),
                    reason="non_spsa_depth_opt",
                )

        # -- Depth-level non-improvement rollback guard --
        # If the optimizer returned an energy worse than the entry energy for
        # this depth, roll back to the pre-optimization parameters.  This
        # prevents stochastic optimizer noise from accumulating regressions.
        depth_rollback = False
        if float(energy_current) > float(energy_prev):
            _ai_log(
                "hardcoded_adapt_depth_rollback",
                depth=int(depth + 1),
                energy_before_opt=float(energy_prev),
                energy_after_opt=float(energy_current),
                regression=float(energy_current - energy_prev),
                opt_method=str(adapt_inner_optimizer_key),
            )
            theta = np.array(theta_before_opt, copy=True)
            energy_current = float(energy_prev)
            depth_rollback = True

        optimizer_elapsed_s = float(time.perf_counter() - optimizer_t0)
        nfev_total += int(nfev_opt)
        depth_local = int(depth + 1)
        depth_cumulative = int(adapt_ref_base_depth) + int(depth_local)
        delta_abs_prev = float(drop_prev_delta_abs)
        delta_abs_current = float(abs(energy_current - exact_gs))
        delta_abs_drop = float(delta_abs_prev - delta_abs_current)
        drop_prev_delta_abs = float(delta_abs_current)
        eps_energy_step_abs = float(abs(energy_current - energy_prev))
        eps_energy_low_step = bool(eps_energy_step_abs < float(eps_energy))
        eps_energy_gate_open = bool(depth_local >= int(eps_energy_min_extra_depth_effective))
        if eps_energy_gate_open:
            if eps_energy_low_step:
                eps_energy_low_streak += 1
            else:
                eps_energy_low_streak = 0
        else:
            eps_energy_low_streak = 0
        eps_energy_termination_condition = bool(eps_energy_gate_open) and (
            int(eps_energy_low_streak) >= int(eps_energy_patience_effective)
        )
        drop_low_signal = None
        drop_low_grad = None
        if drop_policy_enabled and int(depth_local) >= int(adapt_drop_min_depth):
            drop_low_signal = bool(delta_abs_drop < float(adapt_drop_floor))
            if float(adapt_grad_floor) >= 0.0:
                drop_low_grad = bool(float(max_grad) < float(adapt_grad_floor))
            else:
                drop_low_grad = True
            if bool(drop_low_signal) and bool(drop_low_grad):
                drop_plateau_hits += 1
            else:
                drop_plateau_hits = 0
        _ai_log(
            "hardcoded_adapt_optimizer_timing",
            depth=int(depth + 1),
            opt_method=str(adapt_inner_optimizer_key),
            nfev_opt=int(nfev_opt),
            nit_opt=int(nit_opt),
            opt_success=bool(opt_success),
            opt_message=str(opt_message),
            optimizer_elapsed_s=float(optimizer_elapsed_s),
        )

        selected_primary_label = (
            str(selected_batch_labels[0])
            if selected_batch_labels
            else (
                str(phase1_feature_selected.get("candidate_label"))
                if isinstance(phase1_feature_selected, dict)
                and phase1_feature_selected.get("candidate_label") is not None
                else str(pool[best_idx].label)
            )
        )
        selected_grad_signed_value = (
            float(phase1_feature_selected.get("g_signed"))
            if isinstance(phase1_feature_selected, dict)
            and phase1_feature_selected.get("g_signed") is not None
            else float(gradients[best_idx])
        )
        selected_grad_abs_value = (
            float(phase1_feature_selected.get("g_abs"))
            if isinstance(phase1_feature_selected, dict)
            and phase1_feature_selected.get("g_abs") is not None
            else float(grad_magnitudes[best_idx])
        )
        history_row = {
            "depth": int(depth + 1),
            "selected_op": str(selected_primary_label),
            "pool_index": int(best_idx),
            "selected_ops": [str(x) for x in selected_batch_labels],
            "selected_pool_indices": [int(x) for x in selected_batch_indices],
            "selection_mode": str(selection_mode),
            "init_theta": float(init_theta),
            "max_grad": float(max_grad),
            "selected_grad_signed": float(selected_grad_signed_value),
            "selected_grad_abs": float(selected_grad_abs_value),
            "fallback_scan_size": int(fallback_scan_size),
            "fallback_best_probe_delta_e": (
                float(fallback_best_probe_delta_e) if fallback_best_probe_delta_e is not None else None
            ),
            "fallback_best_probe_theta": (
                float(fallback_best_probe_theta) if fallback_best_probe_theta is not None else None
            ),
            "energy_before_opt": float(energy_prev),
            "energy_after_opt": float(energy_current),
            "delta_energy": float(energy_current - energy_prev),
            "delta_abs_prev": float(delta_abs_prev),
            "delta_abs_current": float(delta_abs_current),
            "delta_abs_drop_from_prev": float(delta_abs_drop),
            "opt_method": str(adapt_inner_optimizer_key),
            "reopt_policy": str(adapt_reopt_policy_key),
            "nfev_opt": int(nfev_opt),
            "nit_opt": int(nit_opt),
            "opt_success": bool(opt_success),
            "opt_message": str(opt_message),
            "gradient_eval_elapsed_s": float(gradient_eval_elapsed_s),
            "optimizer_elapsed_s": float(optimizer_elapsed_s),
            "iter_elapsed_s": float(time.perf_counter() - iter_t0),
            "drop_policy_enabled": bool(drop_policy_enabled),
            "drop_policy_source": str(stop_policy.drop_policy_source),
            "adapt_drop_floor_resolved": float(adapt_drop_floor),
            "adapt_drop_patience_resolved": int(adapt_drop_patience),
            "adapt_drop_min_depth_resolved": int(adapt_drop_min_depth),
            "adapt_grad_floor_resolved": float(adapt_grad_floor),
            "adapt_drop_floor_source": str(stop_policy.adapt_drop_floor_source),
            "adapt_drop_patience_source": str(stop_policy.adapt_drop_patience_source),
            "adapt_drop_min_depth_source": str(stop_policy.adapt_drop_min_depth_source),
            "adapt_grad_floor_source": str(stop_policy.adapt_grad_floor_source),
            "drop_low_signal": drop_low_signal,
            "drop_low_grad": drop_low_grad,
            "drop_plateau_hits": int(drop_plateau_hits),
            "depth_rollback": bool(depth_rollback),
            "depth_cumulative": int(depth_cumulative),
            "adapt_ref_base_depth": int(adapt_ref_base_depth),
            "eps_energy_step_abs": float(eps_energy_step_abs),
            "eps_energy_low_step": bool(eps_energy_low_step),
            "eps_energy_low_streak": int(eps_energy_low_streak),
            "eps_energy_gate_open": bool(eps_energy_gate_open),
            "eps_energy_min_extra_depth_effective": int(eps_energy_min_extra_depth_effective),
            "eps_energy_patience_effective": int(eps_energy_patience_effective),
            "eps_energy_termination_enabled": bool(eps_energy_termination_enabled),
            "eps_energy_termination_condition": bool(eps_energy_termination_condition),
            "eps_grad_termination_enabled": bool(eps_grad_termination_enabled),
            "eps_grad_threshold_hit": bool(max_grad < float(eps_grad)),
            "reopt_policy_effective": str(reopt_policy_effective),
            "reopt_active_indices": [int(i) for i in reopt_active_indices],
            "reopt_active_count": int(len(reopt_active_indices)),
            "reopt_runtime_active_indices": [int(i) for i in reopt_runtime_active_indices],
            "reopt_runtime_active_count": int(len(reopt_runtime_active_indices)),
            "num_parameters_after_opt": int(theta.size),
            "logical_num_parameters_after_opt": int(len(selected_ops)),
            "parameters_added_this_step": int(theta.size - theta_before_opt.size),
            "logical_parameters_added_this_step": int(len(selected_batch_labels)) if selected_batch_labels else 1,
            "reopt_periodic_full_refit_triggered": bool(periodic_full_refit_triggered),
        }
        if phase1_enabled:
            history_row.update(
                {
                    "continuation_mode": str(continuation_mode),
                    "candidate_family": str(
                        pool_family_ids[int(best_idx)] if int(best_idx) < len(pool_family_ids) else "legacy"
                    ),
                    "stage_name": str(stage_name),
                    "stage_transition_reason": str(phase1_stage_transition_reason),
                    "selected_position": int(selected_position),
                    "selected_positions": [int(x) for x in selected_batch_positions],
                    "batch_selected": bool(phase2_enabled and phase2_last_batch_selected),
                    "batch_size": int(len(selected_batch_labels)),
                    "positions_considered": [int(x) for x in phase1_last_positions_considered],
                    "score_version": (
                        str(phase1_feature_selected.get("score_version"))
                        if isinstance(phase1_feature_selected, dict)
                        else None
                    ),
                    "simple_score": (
                        float(phase1_feature_selected.get("simple_score"))
                        if isinstance(phase1_feature_selected, dict)
                        and phase1_feature_selected.get("simple_score") is not None
                        else None
                    ),
                    "metric_proxy": (
                        float(phase1_feature_selected.get("metric_proxy"))
                        if isinstance(phase1_feature_selected, dict)
                        else None
                    ),
                    "curvature_mode": (
                        str(phase1_feature_selected.get("curvature_mode"))
                        if isinstance(phase1_feature_selected, dict)
                        else None
                    ),
                    "novelty_mode": (
                        str(phase1_feature_selected.get("novelty_mode"))
                        if isinstance(phase1_feature_selected, dict)
                        else None
                    ),
                    "novelty": (
                        phase1_feature_selected.get("novelty")
                        if isinstance(phase1_feature_selected, dict)
                        else None
                    ),
                    "g_lcb": (
                        float(phase1_feature_selected.get("g_lcb"))
                        if isinstance(phase1_feature_selected, dict)
                        else None
                    ),
                    "refit_window_indices": (
                        [int(i) for i in phase1_feature_selected.get("refit_window_indices", [])]
                        if isinstance(phase1_feature_selected, dict)
                        else [int(i) for i in reopt_active_indices]
                    ),
                    "compile_cost_proxy": (
                        dict(phase1_feature_selected.get("compiled_position_cost_proxy", {}))
                        if isinstance(phase1_feature_selected, dict)
                        else None
                    ),
                    "compile_cost_mode": str(phase3_backend_cost_mode_key),
                    "compile_cost_source": (
                        str(phase1_feature_selected.get("compile_cost_source", "proxy"))
                        if isinstance(phase1_feature_selected, dict)
                        else "proxy"
                    ),
                    "compile_cost_total": (
                        float(phase1_feature_selected.get("compile_cost_total", 0.0))
                        if isinstance(phase1_feature_selected, dict)
                        else 0.0
                    ),
                    "compile_gate_open": (
                        bool(phase1_feature_selected.get("compile_gate_open", True))
                        if isinstance(phase1_feature_selected, dict)
                        else True
                    ),
                    "compile_failure_reason": (
                        phase1_feature_selected.get("compile_failure_reason")
                        if isinstance(phase1_feature_selected, dict)
                        else None
                    ),
                    "compile_cost_backend": (
                        dict(phase1_feature_selected.get("compiled_position_cost_backend", {}))
                        if isinstance(phase1_feature_selected, dict)
                        and isinstance(phase1_feature_selected.get("compiled_position_cost_backend"), Mapping)
                        else None
                    ),
                    "measurement_cache_stats": (
                        dict(phase1_feature_selected.get("measurement_cache_stats", {}))
                        if isinstance(phase1_feature_selected, dict)
                        else None
                    ),
                    "actual_fallback_mode": (
                        str(phase1_feature_selected.get("actual_fallback_mode"))
                        if isinstance(phase1_feature_selected, dict)
                        else None
                    ),
                    "trough_probe_triggered": bool(phase1_last_trough_probe_triggered),
                    "trough_detected": bool(phase1_last_trough_detected),
                }
            )
            if phase2_enabled:
                history_row.update(
                    {
                        "full_v2_score": (
                            float(phase1_feature_selected.get("full_v2_score"))
                            if isinstance(phase1_feature_selected, dict)
                            and phase1_feature_selected.get("full_v2_score") is not None
                            else None
                        ),
                        "shortlist_size": int(len(phase2_last_shortlist_records)),
                        "shortlisted_records": [dict(x) for x in phase2_last_shortlist_records],
                        "compatibility_penalty_total": float(phase2_last_batch_penalty_total),
                        "optimizer_memory_reused": bool(phase2_last_optimizer_memory_reused),
                        "optimizer_memory_source": str(phase2_last_optimizer_memory_source),
                    }
                )
            if phase3_enabled:
                history_row.update(
                    {
                        "generator_id": (
                            str(phase1_feature_selected.get("generator_id"))
                            if isinstance(phase1_feature_selected, dict)
                            and phase1_feature_selected.get("generator_id") is not None
                            else None
                        ),
                        "template_id": (
                            str(phase1_feature_selected.get("template_id"))
                            if isinstance(phase1_feature_selected, dict)
                            and phase1_feature_selected.get("template_id") is not None
                            else None
                        ),
                        "is_macro_generator": (
                            bool(phase1_feature_selected.get("is_macro_generator", False))
                            if isinstance(phase1_feature_selected, dict)
                            else None
                        ),
                        "parent_generator_id": (
                            str(phase1_feature_selected.get("parent_generator_id"))
                            if isinstance(phase1_feature_selected, dict)
                            and phase1_feature_selected.get("parent_generator_id") is not None
                            else None
                        ),
                        "symmetry_mode": (
                            str(phase1_feature_selected.get("symmetry_mode"))
                            if isinstance(phase1_feature_selected, dict)
                            else None
                        ),
                        "symmetry_mitigation_mode": (
                            str(phase1_feature_selected.get("symmetry_mitigation_mode"))
                            if isinstance(phase1_feature_selected, dict)
                            else str(phase3_symmetry_mitigation_mode_key)
                        ),
                        "symmetry_spec": (
                            dict(phase1_feature_selected.get("symmetry_spec", {}))
                            if isinstance(phase1_feature_selected, dict)
                            else None
                        ),
                        "motif_bonus": (
                            float(phase1_feature_selected.get("motif_bonus", 0.0))
                            if isinstance(phase1_feature_selected, dict)
                            else 0.0
                        ),
                        "motif_source": (
                            str(phase1_feature_selected.get("motif_source"))
                            if isinstance(phase1_feature_selected, dict)
                            else None
                        ),
                        "motif_metadata": (
                            dict(phase1_feature_selected.get("motif_metadata", {}))
                            if isinstance(phase1_feature_selected, dict)
                            and isinstance(phase1_feature_selected.get("motif_metadata"), Mapping)
                            else None
                        ),
                        "runtime_split_mode": (
                            str(phase1_feature_selected.get("runtime_split_mode", "off"))
                            if isinstance(phase1_feature_selected, dict)
                            else "off"
                        ),
                        "runtime_split_parent_label": (
                            str(phase1_feature_selected.get("runtime_split_parent_label"))
                            if isinstance(phase1_feature_selected, dict)
                            and phase1_feature_selected.get("runtime_split_parent_label") is not None
                            else None
                        ),
                        "runtime_split_child_index": (
                            int(phase1_feature_selected.get("runtime_split_child_index"))
                            if isinstance(phase1_feature_selected, dict)
                            and phase1_feature_selected.get("runtime_split_child_index") is not None
                            else None
                        ),
                        "runtime_split_child_count": (
                            int(phase1_feature_selected.get("runtime_split_child_count"))
                            if isinstance(phase1_feature_selected, dict)
                            and phase1_feature_selected.get("runtime_split_child_count") is not None
                            else None
                        ),
                        "runtime_split_chosen_representation": (
                            str(phase1_feature_selected.get("runtime_split_chosen_representation", "parent"))
                            if isinstance(phase1_feature_selected, dict)
                            else "parent"
                        ),
                        "runtime_split_child_indices": (
                            [int(x) for x in phase1_feature_selected.get("runtime_split_child_indices", [])]
                            if isinstance(phase1_feature_selected, dict)
                            else []
                        ),
                        "runtime_split_child_labels": (
                            [str(x) for x in phase1_feature_selected.get("runtime_split_child_labels", [])]
                            if isinstance(phase1_feature_selected, dict)
                            else []
                        ),
                        "runtime_split_child_generator_ids": (
                            [str(x) for x in phase1_feature_selected.get("runtime_split_child_generator_ids", [])]
                            if isinstance(phase1_feature_selected, dict)
                            else []
                        ),
                        "lifetime_cost_mode": (
                            str(phase1_feature_selected.get("lifetime_cost_mode"))
                            if isinstance(phase1_feature_selected, dict)
                            else str(phase3_lifetime_cost_mode_key)
                        ),
                        "remaining_evaluations_proxy_mode": (
                            str(phase1_feature_selected.get("remaining_evaluations_proxy_mode"))
                            if isinstance(phase1_feature_selected, dict)
                            else None
                        ),
                        "remaining_evaluations_proxy": (
                            float(phase1_feature_selected.get("remaining_evaluations_proxy", 0.0))
                            if isinstance(phase1_feature_selected, dict)
                            else 0.0
                        ),
                        "lifetime_weight_components": (
                            dict(phase1_feature_selected.get("lifetime_weight_components", {}))
                            if isinstance(phase1_feature_selected, dict)
                            else None
                        ),
                    }
                )
        if adapt_inner_optimizer_key == "SPSA":
            history_row["spsa_params"] = dict(adapt_spsa_params)
        history.append(history_row)
        if phase1_enabled:
            if selected_batch_records_for_history:
                for rec in selected_batch_records_for_history:
                    phase1_features_history.append(dict(rec))
            elif isinstance(phase1_feature_selected, dict):
                phase1_features_history.append(dict(phase1_feature_selected))
            phase1_measure_cache.commit(
                selected_batch_measurement_keys
                if selected_batch_measurement_keys
                else measurement_group_keys_for_term(pool[int(best_idx)])
            )

        _ai_log(
            "hardcoded_adapt_iter_done",
            depth=int(depth + 1),
            energy=float(energy_current),
            delta_e=float(energy_current - energy_prev),
            eps_energy_step_abs=float(eps_energy_step_abs),
            eps_energy_low_step=bool(eps_energy_low_step),
            eps_energy_low_streak=int(eps_energy_low_streak),
            eps_energy_gate_open=bool(eps_energy_gate_open),
            eps_energy_min_extra_depth_effective=int(eps_energy_min_extra_depth_effective),
            eps_energy_patience_effective=int(eps_energy_patience_effective),
            eps_energy_termination_enabled=bool(eps_energy_termination_enabled),
            eps_energy_termination_condition=bool(eps_energy_termination_condition),
            eps_grad_termination_enabled=bool(eps_grad_termination_enabled),
            eps_grad_threshold_hit=bool(max_grad < float(eps_grad)),
            depth_cumulative=int(depth_cumulative),
            delta_abs_current=float(delta_abs_current),
            delta_abs_drop_from_prev=float(delta_abs_drop),
            drop_plateau_hits=int(drop_plateau_hits),
            depth_rollback=bool(depth_rollback),
            gradient_eval_elapsed_s=float(gradient_eval_elapsed_s),
            optimizer_elapsed_s=float(optimizer_elapsed_s),
        )

        if (
            drop_policy_enabled
            and int(depth_local) >= int(adapt_drop_min_depth)
            and int(drop_plateau_hits) >= int(adapt_drop_patience)
        ):
            if phase1_enabled and (not phase1_residual_opened) and len(phase1_residual_indices) > 0:
                phase1_residual_opened = True
                available_indices |= set(int(i) for i in phase1_residual_indices)
                phase1_stage.resolve_stage_transition(
                    drop_plateau_hits=int(drop_plateau_hits),
                    trough_detected=bool(phase1_last_trough_detected),
                    residual_opened=True,
                )
                drop_plateau_hits = 0
                phase1_stage_events.append(
                    {
                        "depth": int(depth_local),
                        "stage_name": "residual",
                        "reason": "drop_plateau_open",
                    }
                )
                _ai_log(
                    "hardcoded_adapt_phase1_residual_opened_on_plateau",
                    depth=int(depth_local),
                    residual_count=int(len(phase1_residual_indices)),
                )
                continue
            stop_reason = "drop_plateau"
            _ai_log(
                "hardcoded_adapt_converged_drop_plateau",
                depth=int(depth_local),
                delta_abs_current=float(delta_abs_current),
                delta_abs_drop_from_prev=float(delta_abs_drop),
                drop_floor=float(adapt_drop_floor),
                drop_patience=int(adapt_drop_patience),
                drop_min_depth=int(adapt_drop_min_depth),
                drop_plateau_hits=int(drop_plateau_hits),
                grad_floor=(float(adapt_grad_floor) if float(adapt_grad_floor) >= 0.0 else None),
                max_grad=float(max_grad),
            )
            break

        # 6) Check energy convergence
        if bool(eps_energy_termination_condition):
            if bool(eps_energy_termination_enabled):
                stop_reason = "eps_energy"
                _ai_log(
                    "hardcoded_adapt_converged_energy",
                    depth=int(depth_local),
                    depth_cumulative=int(depth_cumulative),
                    delta_e=float(eps_energy_step_abs),
                    eps_energy=float(eps_energy),
                    eps_energy_low_streak=int(eps_energy_low_streak),
                    eps_energy_patience=int(eps_energy_patience_effective),
                    eps_energy_min_extra_depth=int(eps_energy_min_extra_depth_effective),
                    adapt_ref_base_depth=int(adapt_ref_base_depth),
                    eps_energy_gate_cumulative_depth=int(adapt_ref_base_depth) + int(eps_energy_min_extra_depth_effective),
                )
                break
            _ai_log(
                "hardcoded_adapt_eps_energy_termination_suppressed",
                depth=int(depth_local),
                depth_cumulative=int(depth_cumulative),
                delta_e=float(eps_energy_step_abs),
                eps_energy=float(eps_energy),
                eps_energy_low_streak=int(eps_energy_low_streak),
                eps_energy_patience=int(eps_energy_patience_effective),
                eps_energy_min_extra_depth=int(eps_energy_min_extra_depth_effective),
                adapt_ref_base_depth=int(adapt_ref_base_depth),
                eps_energy_gate_cumulative_depth=int(adapt_ref_base_depth) + int(eps_energy_min_extra_depth_effective),
                continuation_mode=str(continuation_mode),
                problem=str(problem_key),
            )
        if bool(eps_energy_low_step) and (not bool(eps_energy_gate_open)):
            _ai_log(
                "hardcoded_adapt_energy_convergence_gate_wait",
                depth=int(depth_local),
                depth_cumulative=int(depth_cumulative),
                delta_e=float(eps_energy_step_abs),
                eps_energy=float(eps_energy),
                eps_energy_min_extra_depth=int(eps_energy_min_extra_depth_effective),
                adapt_ref_base_depth=int(adapt_ref_base_depth),
                eps_energy_gate_cumulative_depth=int(adapt_ref_base_depth) + int(eps_energy_min_extra_depth_effective),
            )

        # Check if pool exhausted
        if not allow_repeats and not available_indices:
            stop_reason = "pool_exhausted"
            _ai_log("hardcoded_adapt_pool_exhausted")
            break

    # -- Final full-prefix refit (windowed policy only) --
    final_full_refit_meta: dict[str, Any] = {
        "requested": bool(
            adapt_reopt_policy_key == "windowed" and adapt_final_full_refit_val
        ),
        "executed": False,
        "skipped_reason": None,
        "energy_before": None,
        "energy_after": None,
        "nfev": 0,
        "nit": 0,
        "opt_success": None,
        "opt_message": None,
        "rollback": False,
    }
    if (
        adapt_reopt_policy_key == "windowed"
        and adapt_final_full_refit_val
        and len(selected_ops) > 0
    ):
        # Check if already satisfied by last depth being a full-prefix reopt
        last_was_full = bool(
            len(history) > 0
            and str(history[-1].get("reopt_policy_effective", "")).startswith("windowed_periodic_full")
            or (len(history) > 0 and str(history[-1].get("reopt_policy_effective", "")) == "full")
        )
        # Also skip if last depth used all local indices (full or periodic full)
        if last_was_full and len(history) > 0:
            last_active = history[-1].get("reopt_active_count", 0)
            last_was_full = bool(int(last_active) == int(len(selected_ops)))

        if last_was_full:
            final_full_refit_meta["skipped_reason"] = "last_depth_already_full_prefix"
            _ai_log(
                "hardcoded_adapt_final_full_refit_skipped",
                reason="last_depth_already_full_prefix",
            )
        else:
            _ai_log(
                "hardcoded_adapt_final_full_refit_start",
                n_params=int(theta.size),
                energy_before=float(energy_current),
            )
            final_full_refit_meta["energy_before"] = float(energy_current)
            energy_before_final = float(energy_current)

            # Re-use existing _obj and optimizer infrastructure
            if adapt_state_backend_key == "compiled":
                if selected_executor is None:
                    selected_executor = _build_compiled_executor(selected_ops)

            # Reset COBYLA heartbeat state for final refit
            final_opt_t0 = time.perf_counter()
            cobyla_last_hb_t = final_opt_t0
            cobyla_nfev_so_far = 0
            cobyla_best_fun = float("inf")

            def _obj_final(x: np.ndarray) -> float:
                nonlocal cobyla_last_hb_t, cobyla_nfev_so_far, cobyla_best_fun
                if adapt_state_backend_key == "compiled":
                    assert selected_executor is not None
                    psi_obj = selected_executor.prepare_state(np.asarray(x, dtype=float), psi_ref)
                    energy_obj, _ = energy_via_one_apply(psi_obj, h_compiled)
                    energy_obj_val = float(energy_obj)
                else:
                    energy_obj_val = _adapt_energy_fn(
                        h_poly, psi_ref, selected_ops, x, h_compiled=h_compiled,
                    )
                if adapt_inner_optimizer_key in {"COBYLA", "POWELL"}:
                    cobyla_nfev_so_far += 1
                    if energy_obj_val < cobyla_best_fun:
                        cobyla_best_fun = float(energy_obj_val)
                    now = time.perf_counter()
                    if (now - cobyla_last_hb_t) >= float(adapt_spsa_progress_every_s):
                        _ai_log(
                            "hardcoded_adapt_scipy_heartbeat",
                            stage="final_full_refit",
                            opt_method=str(adapt_inner_optimizer_key),
                            nfev_opt_so_far=int(cobyla_nfev_so_far),
                            best_fun=float(cobyla_best_fun),
                            elapsed_opt_s=float(now - final_opt_t0),
                        )
                        cobyla_last_hb_t = now
                return float(energy_obj_val)

            final_x0 = np.array(theta, copy=True)
            if adapt_inner_optimizer_key == "SPSA":
                final_memory = None
                if phase2_enabled:
                    final_memory = phase2_memory_adapter.select_active(
                        phase2_optimizer_memory,
                        active_indices=list(range(int(theta.size))),
                        source="adapt.final_full_refit.active_subset",
                    )
                final_result = spsa_minimize(
                    fun=_obj_final,
                    x0=final_x0,
                    maxiter=int(maxiter),
                    seed=int(seed) + int(max_depth) + 1,
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
                    memory=(dict(final_memory) if isinstance(final_memory, Mapping) else None),
                    refresh_every=0,
                    precondition_mode=("diag_rms_grad" if phase2_enabled else "none"),
                )
            else:
                if scipy_minimize is None:
                    raise RuntimeError(
                        f"SciPy minimize is unavailable for {adapt_inner_optimizer_key} final full refit."
                    )
                final_result = scipy_minimize(
                    _obj_final,
                    final_x0,
                    method=str(adapt_inner_optimizer_key),
                    options=_scipy_inner_options(int(maxiter)),
                )

            final_energy = float(final_result.fun)
            final_nfev = int(getattr(final_result, "nfev", 0))
            final_nit = int(getattr(final_result, "nit", 0))
            final_success = bool(getattr(final_result, "success", False))
            final_message = str(getattr(final_result, "message", ""))

            # Rollback-on-regression semantics
            if final_energy > energy_before_final:
                final_full_refit_meta["rollback"] = True
                _ai_log(
                    "hardcoded_adapt_final_full_refit_rollback",
                    energy_before=float(energy_before_final),
                    energy_after=float(final_energy),
                )
            else:
                theta = np.asarray(final_result.x, dtype=float)
                energy_current = float(final_energy)
                if phase2_enabled and adapt_inner_optimizer_key == "SPSA":
                    phase2_optimizer_memory = phase2_memory_adapter.merge_active(
                        phase2_optimizer_memory,
                        active_indices=list(range(int(theta.size))),
                        active_state=phase2_memory_adapter.from_result(
                            final_result,
                            method=str(adapt_inner_optimizer_key),
                            parameter_count=int(theta.size),
                            source="adapt.final_full_refit.result",
                        ),
                        source="adapt.final_full_refit.merge",
                    )

            nfev_total += final_nfev
            final_full_refit_meta["executed"] = True
            final_full_refit_meta["energy_after"] = float(final_energy)
            final_full_refit_meta["nfev"] = int(final_nfev)
            final_full_refit_meta["nit"] = int(final_nit)
            final_full_refit_meta["opt_success"] = bool(final_success)
            final_full_refit_meta["opt_message"] = str(final_message)
            _ai_log(
                "hardcoded_adapt_final_full_refit_done",
                energy_before=float(energy_before_final),
                energy_after=float(final_energy),
                rollback=bool(final_full_refit_meta["rollback"]),
                nfev=int(final_nfev),
            )

    prune_summary: dict[str, Any] = {
        "enabled": bool(phase1_enabled and phase1_prune_enabled),
        "executed": False,
        "rolled_back": False,
        "accepted_count": 0,
        "candidate_count": 0,
        "decisions": [],
        "energy_before": float(energy_current),
        "energy_after_prune": float(energy_current),
        "energy_after_post_refit": float(energy_current),
        "post_refit_executed": False,
    }
    if phase1_enabled and bool(phase1_prune_enabled) and int(len(selected_ops)) > 1:
        phase1_scaffold_pre_prune = {
            "operators": [str(op.label) for op in selected_ops],
            "optimal_point": [float(x) for x in np.asarray(theta, dtype=float).tolist()],
            "energy": float(energy_current),
        }
        prune_cfg = PruneConfig(
            max_candidates=int(max(1, phase1_prune_max_candidates)),
            min_candidates=2,
            fraction_candidates=float(max(0.0, phase1_prune_fraction)),
            max_regression=float(max(0.0, phase1_prune_max_regression)),
        )

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
            if adapt_state_backend_key == "compiled":
                selected_executor = _build_compiled_executor(selected_ops) if len(selected_ops) > 0 else None
            else:
                selected_executor = None
            theta_post, e_post = post_prune_refit(
                theta=np.asarray(theta, dtype=float),
                refit_fn=lambda x: _refit_given_ops(list(selected_ops), np.asarray(x, dtype=float)),
            )
            theta = np.asarray(theta_post, dtype=float)
            energy_current = float(e_post)
            if adapt_state_backend_key == "compiled":
                selected_executor = _build_compiled_executor(selected_ops) if len(selected_ops) > 0 else None
            else:
                selected_executor = None
            prune_summary["post_refit_executed"] = True
            if phase3_enabled and str(phase3_symmetry_mitigation_mode_key) != "off":
                post_prune_generator_meta = selected_generator_metadata_for_labels(
                    [str(op.label) for op in selected_ops],
                    pool_generator_registry,
                )
                sym_pre = verify_symmetry_sequence(
                    generator_metadata=pre_prune_generator_meta,
                    mitigation_mode=str(phase3_symmetry_mitigation_mode_key),
                )
                sym_post = verify_symmetry_sequence(
                    generator_metadata=post_prune_generator_meta,
                    mitigation_mode=str(phase3_symmetry_mitigation_mode_key),
                )
                prune_summary["symmetry_mitigation"] = {
                    "mode": str(phase3_symmetry_mitigation_mode_key),
                    "pre_prune": dict(sym_pre),
                    "post_prune": dict(sym_post),
                }
                if (not bool(sym_post.get("passed", True))) or (
                    float(sym_post.get("max_leakage_risk", 0.0))
                    > float(sym_pre.get("max_leakage_risk", 0.0)) + 1e-12
                ):
                    selected_ops = list(pre_prune_ops)
                    selected_layout = _build_selected_layout(selected_ops)
                    theta = np.asarray(pre_prune_theta, dtype=float)
                    energy_current = float(pre_prune_energy)
                    if phase2_enabled and isinstance(pre_prune_memory, dict):
                        phase2_optimizer_memory = dict(pre_prune_memory)
                    if adapt_state_backend_key == "compiled":
                        selected_executor = _build_compiled_executor(selected_ops) if len(selected_ops) > 0 else None
                    else:
                        selected_executor = None
                    prune_summary["rolled_back"] = True
                    prune_summary["rollback_reason"] = "symmetry_verify_failed"
            if float(energy_current) > float(pre_prune_energy) + float(prune_cfg.max_regression):
                selected_ops = list(pre_prune_ops)
                selected_layout = _build_selected_layout(selected_ops)
                theta = np.asarray(pre_prune_theta, dtype=float)
                energy_current = float(pre_prune_energy)
                if phase2_enabled and isinstance(pre_prune_memory, dict):
                    phase2_optimizer_memory = dict(pre_prune_memory)
                if adapt_state_backend_key == "compiled":
                    selected_executor = _build_compiled_executor(selected_ops) if len(selected_ops) > 0 else None
                else:
                    selected_executor = None
                prune_summary["rolled_back"] = True
                prune_summary["rollback_reason"] = "post_prune_regression_exceeded"

        prune_summary.update(
            {
                "executed": True,
                "accepted_count": int(accepted_count),
                "energy_after_prune": float(energy_after_prune),
                "energy_after_post_refit": float(energy_current),
                "decisions": [dict(d.__dict__) for d in prune_decisions],
            }
        )
        _ai_log(
            "hardcoded_adapt_phase1_prune_done",
            candidate_count=int(prune_summary["candidate_count"]),
            accepted_count=int(prune_summary["accepted_count"]),
            energy_after_post_refit=float(prune_summary["energy_after_post_refit"]),
        )

    selected_layout = _build_selected_layout(selected_ops)
    theta_logical_final = _logical_theta_alias(theta, selected_layout)

    # Build final state
    if adapt_state_backend_key == "compiled":
        if len(selected_ops) == 0:
            psi_adapt = np.array(psi_ref, copy=True)
        else:
            if selected_executor is None:
                selected_executor = _build_compiled_executor(selected_ops)
            psi_adapt = selected_executor.prepare_state(theta, psi_ref)
    else:
        psi_adapt = _prepare_adapt_state(psi_ref, selected_ops, theta, parameter_layout=selected_layout)
    psi_adapt = _normalize_state(psi_adapt)

    elapsed = time.perf_counter() - t0
    selected_generator_metadata = (
        selected_generator_metadata_for_labels(
            [str(op.label) for op in selected_ops],
            pool_generator_registry,
        )
        if phase3_enabled
        else []
    )
    phase3_symmetry_summary = (
        verify_symmetry_sequence(
            generator_metadata=selected_generator_metadata,
            mitigation_mode=str(phase3_symmetry_mitigation_mode_key),
        )
        if phase3_enabled
        else None
    )
    phase3_output_motif_library = (
        extract_motif_library(
            generator_metadata=selected_generator_metadata,
            theta=[float(x) for x in np.asarray(theta_logical_final, dtype=float).tolist()],
            source_num_sites=int(num_sites),
            source_tag=f"phase3_v1_L{int(num_sites)}",
            ordering=str(ordering),
            boson_encoding=str(boson_encoding),
        )
        if phase3_enabled and selected_generator_metadata
        else None
    )
    if phase3_enabled and isinstance(phase3_input_motif_library, Mapping):
        selected_match_count = 0
        source_records = phase3_input_motif_library.get("records", [])
        if isinstance(source_records, Sequence):
            for meta in selected_generator_metadata:
                matched = False
                for rec in source_records:
                    if not isinstance(rec, Mapping):
                        continue
                    if str(rec.get("family_id", "")) != str(meta.get("family_id", "")):
                        continue
                    if str(rec.get("template_id", "")) != str(meta.get("template_id", "")):
                        continue
                    if [int(x) for x in rec.get("support_site_offsets", [])] != [
                        int(x) for x in meta.get("support_site_offsets", [])
                    ]:
                        continue
                    matched = True
                    break
                if matched:
                    selected_match_count += 1
        phase3_motif_usage["selected_match_count"] = int(selected_match_count)

    continuation_payload: dict[str, Any] = {
        "mode": str(continuation_mode),
        "score_version": (
            str(phase2_score_cfg.score_version)
            if phase2_enabled
            else str(phase1_score_cfg.score_version)
        ),
        "stage_controller": {
            "shortlist_size": int(phase1_shortlist_size_val),
            "plateau_patience": int(phase1_stage_cfg.plateau_patience),
            "probe_margin_ratio": float(phase1_stage_cfg.probe_margin_ratio),
            "max_probe_positions": int(phase1_stage_cfg.max_probe_positions),
            "append_admit_threshold": float(phase1_stage_cfg.append_admit_threshold),
        },
        "stage_events": [dict(row) for row in phase1_stage_events],
        "phase1_feature_rows": [dict(row) for row in phase1_features_history[-200:]],
        "last_probe_reason": str(phase1_last_probe_reason),
        "residual_opened": bool(phase1_residual_opened),
    }
    backend_compile_summary: dict[str, Any] | None = None
    if backend_compile_oracle is not None:
        backend_compile_summary = {
            "mode": str(phase3_backend_cost_mode_key),
            "requested_backend_name": (
                None if phase3_backend_name in {None, ""} else str(phase3_backend_name)
            ),
            "requested_backend_shortlist": [str(x) for x in phase3_backend_shortlist_tokens],
            "optimization_level": int(phase3_backend_optimization_level),
            "seed_transpiler": int(phase3_backend_transpile_seed),
            "resolution_audit": [dict(row) for row in backend_compile_oracle.resolution_audit],
            "cache_summary": dict(backend_compile_oracle.cache_summary()),
            **dict(backend_compile_oracle.final_scaffold_summary(selected_ops)),
        }
    if phase2_enabled:
        continuation_payload.update(
            {
                "phase2_shortlist_rows": [dict(row) for row in phase2_last_shortlist_records[-200:]],
                "optimizer_memory": dict(phase2_optimizer_memory),
                "phase2": {
                    "shortlist_fraction": float(phase2_score_cfg.shortlist_fraction),
                    "shortlist_size": int(phase2_score_cfg.shortlist_size),
                    "batch_target_size": int(phase2_score_cfg.batch_target_size),
                    "batch_size_cap": int(phase2_score_cfg.batch_size_cap),
                    "batch_near_degenerate_ratio": float(phase2_score_cfg.batch_near_degenerate_ratio),
                },
            }
        )
    if phase3_enabled:
        continuation_payload.update(
            {
                "selected_generator_metadata": [dict(x) for x in selected_generator_metadata],
                "generator_split_events": [dict(x) for x in phase3_split_events],
                "runtime_split_summary": dict(phase3_runtime_split_summary),
                "motif_library": (
                    dict(phase3_output_motif_library)
                    if isinstance(phase3_output_motif_library, Mapping)
                    else None
                ),
                "motif_usage": dict(phase3_motif_usage),
                "symmetry_mitigation": (
                    dict(phase3_symmetry_summary)
                    if isinstance(phase3_symmetry_summary, Mapping)
                    else None
                ),
                "rescue_history": [dict(x) for x in phase3_rescue_history],
            }
        )
    if backend_compile_summary is not None:
        continuation_payload["backend_compile_cost_summary"] = dict(backend_compile_summary)

    payload = {
        "success": True,
        "method": method_name,
        "energy": float(energy_current),
        "exact_gs_energy": float(exact_gs),
        "delta_e": float(energy_current - exact_gs),
        "abs_delta_e": float(abs(energy_current - exact_gs)),
        "num_particles": {"n_up": int(num_particles[0]), "n_dn": int(num_particles[1])},
        "ansatz_depth": int(len(selected_ops)),
        "num_parameters": int(theta.size),
        "logical_num_parameters": int(len(selected_ops)),
        "optimal_point": [float(x) for x in theta.tolist()],
        "logical_optimal_point": [float(x) for x in theta_logical_final.tolist()],
        "parameterization": serialize_layout(selected_layout),
        "operators": [str(op.label) for op in selected_ops],
        "pool_size": int(len(pool)),
        "pool_type": str(pool_key),
        "phase1_depth0_full_meta_override": bool(phase1_depth0_full_meta_override),
        "stop_reason": str(stop_reason),
        "nfev_total": int(nfev_total),
        "adapt_inner_optimizer": str(adapt_inner_optimizer_key),
        "adapt_reopt_policy": str(adapt_reopt_policy_key),
        "adapt_window_size": int(adapt_window_size_val),
        "adapt_window_topk": int(adapt_window_topk_val),
        "adapt_full_refit_every": int(adapt_full_refit_every_val),
        "adapt_final_full_refit": bool(adapt_final_full_refit_val),
        "allow_repeats": bool(allow_repeats),
        "finite_angle_fallback": bool(finite_angle_fallback),
        "finite_angle": float(finite_angle),
        "finite_angle_min_improvement": float(finite_angle_min_improvement),
        "adapt_drop_policy_enabled": bool(drop_policy_enabled),
        "adapt_drop_floor": (float(adapt_drop_floor) if drop_policy_enabled else None),
        "adapt_drop_patience": (int(adapt_drop_patience) if drop_policy_enabled else None),
        "adapt_drop_min_depth": (int(adapt_drop_min_depth) if drop_policy_enabled else None),
        "adapt_grad_floor": (float(adapt_grad_floor) if float(adapt_grad_floor) >= 0.0 else None),
        "adapt_drop_floor_resolved": float(adapt_drop_floor),
        "adapt_drop_patience_resolved": int(adapt_drop_patience),
        "adapt_drop_min_depth_resolved": int(adapt_drop_min_depth),
        "adapt_grad_floor_resolved": float(adapt_grad_floor),
        "adapt_drop_floor_source": str(stop_policy.adapt_drop_floor_source),
        "adapt_drop_patience_source": str(stop_policy.adapt_drop_patience_source),
        "adapt_drop_min_depth_source": str(stop_policy.adapt_drop_min_depth_source),
        "adapt_grad_floor_source": str(stop_policy.adapt_grad_floor_source),
        "adapt_drop_policy_source": str(stop_policy.drop_policy_source),
        "adapt_ref_base_depth": int(adapt_ref_base_depth),
        "adapt_eps_energy_min_extra_depth": int(adapt_eps_energy_min_extra_depth),
        "adapt_eps_energy_patience": int(adapt_eps_energy_patience),
        "eps_energy_min_extra_depth_effective": int(eps_energy_min_extra_depth_effective),
        "eps_energy_patience_effective": int(eps_energy_patience_effective),
        "eps_energy_gate_cumulative_depth": int(adapt_ref_base_depth) + int(eps_energy_min_extra_depth_effective),
        "eps_energy_termination_enabled": bool(eps_energy_termination_enabled),
        "eps_grad_termination_enabled": bool(eps_grad_termination_enabled),
        "eps_energy_low_streak_final": int(eps_energy_low_streak),
        "drop_plateau_hits_final": int(drop_plateau_hits),
        "adapt_gradient_parity_check": bool(adapt_gradient_parity_check),
        "adapt_state_backend": str(adapt_state_backend_key),
        "compiled_pauli_cache": {
            "enabled": True,
            "compile_elapsed_s": compile_cache_elapsed_s,
            "h_terms": int(len(h_compiled.terms)),
            "pool_terms_total": int(pool_compiled_terms_total),
            "unique_pauli_actions": int(len(pauli_action_cache)),
        },
        "history": history,
        "final_full_refit": dict(final_full_refit_meta),
        "continuation_mode": str(continuation_mode),
        "elapsed_s": float(elapsed),
        "hf_bitstring_qn_to_q0": str(hf_bits),
    }
    if phase1_enabled:
        measurement_plan = phase1_measure_cache.plan_for([])
        payload.update(
            {
                "continuation": dict(continuation_payload),
                "measurement_cache_summary": {
                    **dict(phase1_measure_cache.summary()),
                    "measurement_plan": dict(measurement_plan.__dict__),
                },
                "compile_cost_proxy_summary": {
                    "version": (
                        "phase3_v1_proxy"
                        if phase3_enabled
                        else ("phase2_v1_proxy" if phase2_enabled else "phase1_v1_proxy")
                    ),
                    "components": [
                        "new_pauli_actions",
                        "new_rotation_steps",
                        "cx_proxy_total",
                        "sq_proxy_total",
                        "gate_proxy_total",
                        "max_pauli_weight",
                        "position_shift_span",
                        "refit_active_count",
                    ],
                },
                "compile_cost_mode": str(phase3_backend_cost_mode_key),
                "backend_compile_cost_summary": (
                    dict(backend_compile_summary) if backend_compile_summary is not None else None
                ),
                "pre_prune_scaffold": (
                    dict(phase1_scaffold_pre_prune) if phase1_scaffold_pre_prune is not None else None
                ),
                "prune_summary": dict(prune_summary),
                "post_prune_refit": {
                    "executed": bool(prune_summary.get("post_refit_executed", False)),
                    "energy": float(prune_summary.get("energy_after_post_refit", energy_current)),
                },
                "scaffold_fingerprint_lite": ScaffoldFingerprintLite(
                    selected_operator_labels=[str(op.label) for op in selected_ops],
                    selected_generator_ids=[
                        str(meta.get("generator_id", ""))
                        for meta in selected_generator_metadata
                        if str(meta.get("generator_id", "")) != ""
                    ],
                    num_parameters=int(theta.size),
                    generator_family=str(pool_key),
                    continuation_mode=str(continuation_mode),
                    compiled_pauli_cache_size=int(len(pauli_action_cache)),
                    measurement_plan_version=str(measurement_plan.plan_version),
                    post_prune=bool(prune_summary.get("executed", False)),
                    split_event_count=int(len(phase3_split_events)),
                    motif_record_ids=(
                        [
                            str(rec.get("motif_id", ""))
                            for rec in phase3_output_motif_library.get("records", [])
                            if isinstance(rec, Mapping) and str(rec.get("motif_id", "")) != ""
                        ]
                        if isinstance(phase3_output_motif_library, Mapping)
                        else []
                    ),
                    compile_cost_mode=str(phase3_backend_cost_mode_key),
                    backend_target_names=(
                        list(
                            dict.fromkeys(
                                [
                                    str(row.get("transpile_backend", ""))
                                    for row in (backend_compile_summary or {}).get("rows", [])
                                    if str(row.get("transpile_backend", "")) != ""
                                ]
                            )
                        )
                        if isinstance(backend_compile_summary, Mapping)
                        else []
                    ),
                    backend_reduction_mode=(
                        "single_backend"
                        if str(phase3_backend_cost_mode_key) == "transpile_single_v1"
                        else (
                            "best_backend_in_shortlist_v1"
                            if str(phase3_backend_cost_mode_key) == "transpile_shortlist_v1"
                            else "none"
                        )
                    ),
                ).__dict__,
            }
        )
    if adapt_inner_optimizer_key == "SPSA":
        payload["adapt_spsa"] = dict(adapt_spsa_params)

    _ai_log(
        "hardcoded_adapt_vqe_done",
        L=int(num_sites),
        adapt_inner_optimizer=str(adapt_inner_optimizer_key),
        energy=float(energy_current),
        exact_gs=float(exact_gs),
        abs_delta_e=float(abs(energy_current - exact_gs)),
        depth=int(len(selected_ops)),
        stop_reason=str(stop_reason),
        elapsed_sec=round(elapsed, 6),
    )
    return payload, psi_adapt


# ---------------------------------------------------------------------------
# Trajectory simulation (identical to VQE pipeline)
# ---------------------------------------------------------------------------

def _simulate_trajectory(
    *,
    num_sites: int,
    psi0: np.ndarray,
    hmat: np.ndarray,
    ordered_labels_exyz: list[str],
    coeff_map_exyz: dict[str, complex],
    trotter_steps: int,
    t_final: float,
    num_times: int,
    suzuki_order: int,
) -> tuple[list[dict[str, float]], list[np.ndarray]]:
    if int(suzuki_order) != 2:
        raise ValueError("This script currently supports suzuki_order=2 only.")

    nq = len(ordered_labels_exyz[0]) if ordered_labels_exyz else int(np.log2(max(1, psi0.size)))
    evals, evecs = np.linalg.eigh(hmat)
    evecs_dag = np.conjugate(evecs).T

    compiled = {lbl: _compile_pauli_action(lbl, nq) for lbl in ordered_labels_exyz}
    times = np.linspace(0.0, float(t_final), int(num_times))
    n_times = int(times.size)
    stride = max(1, n_times // 20)
    t0 = time.perf_counter()
    _ai_log(
        "hardcoded_adapt_trajectory_start",
        L=int(num_sites),
        num_times=n_times,
        t_final=float(t_final),
        trotter_steps=int(trotter_steps),
    )

    rows: list[dict[str, float]] = []
    exact_states: list[np.ndarray] = []

    for idx, time_val in enumerate(times):
        tv = float(time_val)
        psi_exact = evecs @ (np.exp(-1j * evals * tv) * (evecs_dag @ psi0))
        psi_exact = _normalize_state(psi_exact)

        psi_trot = _evolve_trotter_suzuki2_absolute(
            psi0, ordered_labels_exyz, coeff_map_exyz, compiled, tv, int(trotter_steps),
        )

        fidelity = float(abs(np.vdot(psi_exact, psi_trot)) ** 2)
        n_up_exact, n_dn_exact = _occupation_site0(psi_exact, num_sites)
        n_up_trot, n_dn_trot = _occupation_site0(psi_trot, num_sites)

        rows.append({
            "time": tv,
            "fidelity": fidelity,
            "energy_exact": _expectation_hamiltonian(psi_exact, hmat),
            "energy_trotter": _expectation_hamiltonian(psi_trot, hmat),
            "n_up_site0_exact": n_up_exact,
            "n_up_site0_trotter": n_up_trot,
            "n_dn_site0_exact": n_dn_exact,
            "n_dn_site0_trotter": n_dn_trot,
            "doublon_exact": _doublon_total(psi_exact, num_sites),
            "doublon_trotter": _doublon_total(psi_trot, num_sites),
        })
        exact_states.append(psi_exact)
        if idx == 0 or idx == n_times - 1 or ((idx + 1) % stride == 0):
            _ai_log(
                "hardcoded_adapt_trajectory_progress",
                step=int(idx + 1),
                total_steps=n_times,
                frac=round(float((idx + 1) / n_times), 6),
                time=tv,
                fidelity=float(fidelity),
                elapsed_sec=round(time.perf_counter() - t0, 6),
            )

    _ai_log("hardcoded_adapt_trajectory_done", L=int(num_sites), num_times=n_times)
    return rows, exact_states


# ---------------------------------------------------------------------------
# PDF writer (compact — mirrors VQE pipeline)
# ---------------------------------------------------------------------------

def _write_pipeline_pdf(pdf_path: Path, payload: dict[str, Any], run_command: str) -> None:
    require_matplotlib()
    settings = payload.get("settings", {})
    adapt = payload.get("adapt_vqe", {})
    problem = settings.get("problem", "hubbard")
    model_name = "Hubbard-Holstein" if problem == "hh" else "Hubbard"

    manifest_sections: list[tuple[str, list[tuple[str, Any]]]] = [
        (
            "Model and regime",
            [
                ("Model family", model_name),
                ("Ansatz type", f"ADAPT-VQE (pool: {settings.get('adapt_pool', '?')})"),
                ("Drive enabled", False),
                ("L", settings.get("L")),
                ("Boundary", settings.get("boundary")),
                ("Ordering", settings.get("ordering")),
            ],
        ),
        (
            "Core physical parameters",
            [
                ("t", settings.get("t")),
                ("U", settings.get("u")),
                ("dv", settings.get("dv")),
            ],
        ),
        (
            "ADAPT controls",
            [
                ("ADAPT max depth", settings.get("adapt_max_depth", "?")),
                ("ADAPT eps_grad", settings.get("adapt_eps_grad", "?")),
                ("ADAPT eps_energy", settings.get("adapt_eps_energy", "?")),
                ("Inner optimizer", settings.get("adapt_inner_optimizer", "?")),
                ("Finite-angle fallback", settings.get("adapt_finite_angle_fallback", "?")),
                ("Finite-angle probe", settings.get("adapt_finite_angle", "?")),
            ],
        ),
        (
            "Trajectory settings",
            [
                ("trotter_steps", settings.get("trotter_steps")),
                ("t_final", settings.get("t_final")),
                ("Suzuki order", settings.get("suzuki_order")),
                ("Initial state source", settings.get("initial_state_source")),
            ],
        ),
    ]
    if problem == "hh":
        manifest_sections.append(
            (
                "Hubbard-Holstein parameters",
                [
                    ("omega0", settings.get("omega0")),
                    ("g_ep", settings.get("g_ep")),
                    ("n_ph_max", settings.get("n_ph_max")),
                    ("Boson encoding", settings.get("boson_encoding")),
                ],
            )
        )
    if str(settings.get("adapt_inner_optimizer", "")).strip().upper() == "SPSA":
        adapt_spsa = settings.get("adapt_spsa", {})
        if isinstance(adapt_spsa, dict):
            manifest_sections.append(
                (
                    "SPSA settings",
                    [
                        ("a", adapt_spsa.get("a")),
                        ("c", adapt_spsa.get("c")),
                        ("A", adapt_spsa.get("A")),
                        ("alpha", adapt_spsa.get("alpha")),
                        ("gamma", adapt_spsa.get("gamma")),
                        ("eval_repeats", adapt_spsa.get("eval_repeats")),
                        ("eval_agg", adapt_spsa.get("eval_agg")),
                        ("avg_last", adapt_spsa.get("avg_last")),
                    ],
                )
            )

    summary_sections: list[tuple[str, list[tuple[str, Any]]]] = [
        (
            "ADAPT outcome",
            [
                ("ADAPT-VQE energy", adapt.get("energy")),
                ("Exact GS energy", adapt.get("exact_gs_energy")),
                ("|ΔE|", adapt.get("abs_delta_e")),
                ("Ansatz depth", adapt.get("ansatz_depth")),
                ("Pool size", adapt.get("pool_size")),
            ],
        ),
        (
            "Optimization summary",
            [
                ("Stop reason", adapt.get("stop_reason")),
                ("Total nfev", adapt.get("nfev_total")),
                ("Elapsed (s)", adapt.get("elapsed_s")),
                ("Inner optimizer", settings.get("adapt_inner_optimizer")),
            ],
        ),
        (
            "Trajectory grid",
            [
                ("trotter_steps", settings.get("trotter_steps")),
                ("t_final", settings.get("t_final")),
                ("Initial state source", settings.get("initial_state_source")),
            ],
        ),
    ]

    operator_lines = [
        "Selected operators",
        "",
        f"Ansatz depth: {adapt.get('ansatz_depth')}",
        f"Pool size: {adapt.get('pool_size')}",
        f"Stop reason: {adapt.get('stop_reason')}",
        "",
    ]
    for op_label in (adapt.get("operators") or []):
        operator_lines.append(f"  {op_label}")

    with PdfPages(str(pdf_path)) as pdf:
        render_manifest_overview_page(
            pdf,
            title=f"{model_name} ADAPT-VQE report — L={settings.get('L')}",
            experiment_statement="ADAPT-VQE state preparation followed by exact-versus-Trotter trajectory diagnostics.",
            sections=manifest_sections,
            notes=[
                "The full operator list and executed command are moved to the appendix.",
            ],
        )
        render_executive_summary_page(
            pdf,
            title="Executive summary",
            experiment_statement="Prepared-state quality and convergence summary before trajectory pages.",
            sections=summary_sections,
            notes=[
                "Trajectory pages show fidelity, energy, occupations, and doublon from the ADAPT state.",
            ],
        )
        render_section_divider_page(
            pdf,
            title="Trajectory diagnostics",
            summary="Main result pages compare exact and Trotter trajectories starting from the ADAPT-prepared state.",
            bullets=[
                "Fidelity and energy.",
                "Site-0 occupations and doublon.",
            ],
        )

        # Trajectory plots
        rows = payload.get("trajectory", [])
        if rows:
            times = np.array([r["time"] for r in rows])
            fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True)
            ax_f, ax_e = axes[0]
            ax_n, ax_d = axes[1]

            ax_f.plot(times, [r["fidelity"] for r in rows], color="#0b3d91")
            ax_f.set_title("Fidelity (Trotter vs Exact)")
            ax_f.set_ylabel("F(t)")
            ax_f.grid(alpha=0.25)

            ax_e.plot(times, [r["energy_trotter"] for r in rows], label="Trotter", color="#d62728")
            ax_e.plot(times, [r["energy_exact"] for r in rows], label="Exact", color="#111111", linestyle="--")
            ax_e.set_title("Energy")
            ax_e.set_ylabel("E(t)")
            ax_e.legend(fontsize=8)
            ax_e.grid(alpha=0.25)

            ax_n.plot(times, [r["n_up_site0_trotter"] for r in rows], label="n_up trot", color="#17becf")
            ax_n.plot(times, [r["n_dn_site0_trotter"] for r in rows], label="n_dn trot", color="#9467bd")
            ax_n.set_title("Site-0 Occupations (Trotter)")
            ax_n.set_xlabel("Time")
            ax_n.legend(fontsize=8)
            ax_n.grid(alpha=0.25)

            ax_d.plot(times, [r["doublon_trotter"] for r in rows], label="Trotter", color="#e377c2")
            ax_d.plot(times, [r["doublon_exact"] for r in rows], label="Exact", color="#111111", linestyle="--")
            ax_d.set_title("Doublon")
            ax_d.set_xlabel("Time")
            ax_d.legend(fontsize=8)
            ax_d.grid(alpha=0.25)

            fig.suptitle(f"Hardcoded ADAPT-VQE Pipeline L={settings.get('L')}", fontsize=13)
            fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
            pdf.savefig(fig)
            plt.close(fig)

        render_section_divider_page(
            pdf,
            title="Technical appendix",
            summary="Detailed operator provenance and full reproducibility material.",
            bullets=[
                "Selected operator list.",
                "Executed command.",
            ],
        )
        render_text_page(pdf, operator_lines)
        render_command_page(
            pdf,
            run_command,
            script_name="pipelines/hardcoded/adapt_pipeline.py",
        )


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hardcoded ADAPT-VQE Hubbard / Hubbard-Holstein pipeline.")
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--u", type=float, default=4.0)
    p.add_argument("--problem", choices=["hubbard", "hh"], default="hubbard")
    p.add_argument("--dv", type=float, default=0.0)
    p.add_argument("--omega0", type=float, default=0.0)
    p.add_argument("--g-ep", type=float, default=0.0, help="Holstein electron-phonon coupling g.")
    p.add_argument("--n-ph-max", type=int, default=1)
    p.add_argument("--boson-encoding", choices=["binary"], default="binary")
    p.add_argument("--boundary", choices=["periodic", "open"], default="periodic")
    p.add_argument("--ordering", choices=["blocked", "interleaved"], default="blocked")
    p.add_argument("--term-order", choices=["native", "sorted"], default="sorted")

    # ADAPT-VQE controls
    p.add_argument(
        "--adapt-pool",
        choices=[
            "uccsd",
            "cse",
            "full_hamiltonian",
            "hva",
            "full_meta",
            "pareto_lean",
            "pareto_lean_l2",
            "pareto_lean_gate_pruned",
            "uccsd_paop_lf_full",
            "paop",
            "paop_min",
            "paop_std",
            "paop_full",
            "paop_lf",
            "paop_lf_std",
            "paop_lf2_std",
            "paop_lf_full",
        ],
        default=None,
        help=(
            "ADAPT pool family. If omitted, runtime resolves problem-aware defaults: "
            "hubbard->uccsd, hh+legacy->full_meta, hh+phase1_v1/phase2_v1/phase3_v1->paop_lf_std core + residual full_meta. "
            "HH also supports opt-in scaffold-derived preset pareto_lean."
        ),
    )
    p.add_argument(
        "--adapt-continuation-mode",
        choices=["legacy", "phase1_v1", "phase2_v1", "phase3_v1"],
        default="legacy",
        help="Continuation mode for ADAPT. legacy is default; phase1_v1 is staged continuation; phase2_v1 adds shortlist/full scoring and batching; phase3_v1 adds generator/motif/symmetry/rescue metadata.",
    )
    p.add_argument("--adapt-max-depth", type=int, default=20)
    p.add_argument("--adapt-eps-grad", type=float, default=1e-4)
    p.add_argument(
        "--adapt-eps-energy",
        type=float,
        default=1e-8,
        help=(
            "Energy convergence threshold. Acts as a terminating guard for Hubbard and HH legacy runs; "
            "in HH phase1_v1/phase2_v1/phase3_v1 it is telemetry-only."
        ),
    )
    p.add_argument(
        "--adapt-inner-optimizer",
        choices=["COBYLA", "POWELL", "SPSA"],
        default="SPSA",
        help="Inner re-optimizer for HH seed pre-opt and per-depth ADAPT re-optimization.",
    )
    p.add_argument(
        "--adapt-state-backend",
        choices=["compiled", "legacy"],
        default="compiled",
        help="State action backend for ADAPT gradient/energy evaluations (compiled is default production path).",
    )
    p.add_argument(
        "--adapt-reopt-policy",
        choices=["append_only", "full", "windowed"],
        default="append_only",
        help=(
            "Per-depth re-optimization policy. "
            "'append_only' (default): freeze the prefix theta[:k] and optimize only the newly appended parameter. "
            "'full': legacy behavior — re-optimize all parameters jointly. "
            "'windowed': optimize a sliding window of recent parameters plus optional top-k older carry."
        ),
    )
    p.add_argument(
        "--adapt-window-size", type=int, default=3,
        help="Window size for 'windowed' reopt policy (number of newest parameters in active set).",
    )
    p.add_argument(
        "--adapt-window-topk", type=int, default=0,
        help="Number of older high-magnitude parameters to include in windowed active set.",
    )
    p.add_argument(
        "--adapt-full-refit-every", type=int, default=0,
        help="Periodic full-prefix refit cadence for 'windowed' (0=disabled). Uses cumulative depth.",
    )
    p.add_argument(
        "--adapt-final-full-refit",
        choices=["true", "false"],
        default="true",
        help="Run a final full-prefix refit after ADAPT loop when using 'windowed' policy.",
    )
    p.add_argument("--phase1-lambda-F", type=float, default=1.0)
    p.add_argument("--phase1-lambda-compile", type=float, default=0.05)
    p.add_argument("--phase1-lambda-measure", type=float, default=0.02)
    p.add_argument("--phase1-lambda-leak", type=float, default=0.0)
    p.add_argument("--phase1-score-z-alpha", type=float, default=0.0)
    p.add_argument(
        "--phase1-shortlist-size",
        type=int,
        default=64,
        help="Maximum candidate count admitted into phase-1 probing before phase-2 full scoring.",
    )
    p.add_argument("--phase1-probe-max-positions", type=int, default=6)
    p.add_argument("--phase1-plateau-patience", type=int, default=2)
    p.add_argument("--phase1-trough-margin-ratio", type=float, default=1.0)
    p.set_defaults(phase1_prune_enabled=True)
    p.add_argument("--phase1-prune-enabled", dest="phase1_prune_enabled", action="store_true")
    p.add_argument("--phase1-no-prune", dest="phase1_prune_enabled", action="store_false")
    p.add_argument("--phase1-prune-fraction", type=float, default=0.25)
    p.add_argument("--phase1-prune-max-candidates", type=int, default=6)
    p.add_argument("--phase1-prune-max-regression", type=float, default=1e-8)
    p.add_argument(
        "--phase2-shortlist-fraction",
        type=float,
        default=0.2,
        help="Fraction of phase-1 records admitted into phase-2 full scoring before shortlist-size capping.",
    )
    p.add_argument(
        "--phase2-shortlist-size",
        type=int,
        default=12,
        help="Maximum phase-2 shortlist size after cheap screening.",
    )
    p.add_argument(
        "--phase2-lambda-H",
        type=float,
        default=1e-6,
        help="Phase-2 full-score weight for the curvature/H proxy term.",
    )
    p.add_argument(
        "--phase2-rho",
        type=float,
        default=0.25,
        help="Phase-2 diversity penalty weight used during shortlist/batch scoring.",
    )
    p.add_argument(
        "--phase2-gamma-N",
        type=float,
        default=1.0,
        help="Phase-2 novelty multiplier used in the full_v2 score.",
    )
    p.set_defaults(phase2_enable_batching=True)
    p.add_argument("--phase2-enable-batching", dest="phase2_enable_batching", action="store_true")
    p.add_argument("--phase2-no-batching", dest="phase2_enable_batching", action="store_false")
    p.add_argument(
        "--phase2-batch-target-size",
        type=int,
        default=2,
        help="Target number of near-degenerate candidates admitted into a phase-2 batch selection step.",
    )
    p.add_argument(
        "--phase2-batch-size-cap",
        type=int,
        default=3,
        help="Hard cap on candidates admitted into a phase-2 batch selection step.",
    )
    p.add_argument(
        "--phase2-batch-near-degenerate-ratio",
        type=float,
        default=0.9,
        help="Relative full-score threshold for near-degenerate candidates eligible for phase-2 batching.",
    )
    p.add_argument(
        "--phase3-motif-source-json",
        type=Path,
        default=None,
        help="Optional solved HH continuation JSON used to derive a transferable motif library for phase3_v1.",
    )
    p.add_argument(
        "--phase3-symmetry-mitigation-mode",
        choices=["off", "verify_only", "postselect_diag_v1", "projector_renorm_v1"],
        default="off",
        help="Optional Phase 3 symmetry hook. verify_only preserves current behavior; active symmetry modes remain metadata/telemetry hooks here and are enforced in the noise oracle path.",
    )
    p.set_defaults(phase3_enable_rescue=False)
    p.add_argument("--phase3-enable-rescue", dest="phase3_enable_rescue", action="store_true")
    p.add_argument("--phase3-no-rescue", dest="phase3_enable_rescue", action="store_false")
    p.add_argument(
        "--phase3-lifetime-cost-mode",
        choices=["off", "phase3_v1"],
        default="phase3_v1",
        help="Enable deterministic lifetime burden weighting inside the existing full_v2 score.",
    )
    p.add_argument(
        "--phase3-runtime-split-mode",
        choices=["off", "shortlist_pauli_children_v1"],
        default="off",
        help="Opt-in shortlist-only macro splitting. When enabled, shortlisted macro generators are probed through serialized Pauli child atoms, but only symmetry-safe child-set candidates compete for admission at phase2/phase3 scoring time.",
    )
    p.add_argument(
        "--phase3-backend-cost-mode",
        choices=["proxy", "transpile_single_v1", "transpile_shortlist_v1"],
        default="proxy",
        help="Keep ADAPT logical but replace the Phase 3 compile-burden proxy with transpilation-derived burden against one backend or a fixed backend shortlist.",
    )
    p.add_argument(
        "--phase3-backend-name",
        type=str,
        default=None,
        help="Target backend name for --phase3-backend-cost-mode transpile_single_v1 (for example ibm_boston or ibm_miami).",
    )
    p.add_argument(
        "--phase3-backend-shortlist",
        type=str,
        default=None,
        help="Comma-separated backend shortlist for --phase3-backend-cost-mode transpile_shortlist_v1.",
    )
    p.add_argument(
        "--phase3-backend-transpile-seed",
        type=int,
        default=7,
        help="Seed used by the backend-conditioned transpilation oracle.",
    )
    p.add_argument(
        "--phase3-backend-optimization-level",
        type=int,
        default=1,
        help="Qiskit transpiler optimization level used by the backend-conditioned transpilation oracle.",
    )
    p.add_argument("--adapt-maxiter", type=int, default=300, help="Inner optimizer maxiter per re-optimization")
    p.add_argument("--adapt-spsa-a", type=float, default=0.2)
    p.add_argument("--adapt-spsa-c", type=float, default=0.1)
    p.add_argument("--adapt-spsa-alpha", type=float, default=0.602)
    p.add_argument("--adapt-spsa-gamma", type=float, default=0.101)
    p.add_argument("--adapt-spsa-A", type=float, default=10.0)
    p.add_argument("--adapt-spsa-avg-last", type=int, default=0)
    p.add_argument("--adapt-spsa-eval-repeats", type=int, default=1)
    p.add_argument(
        "--adapt-spsa-eval-agg",
        choices=["mean", "median"],
        default="mean",
    )
    p.add_argument("--adapt-spsa-callback-every", type=int, default=5)
    p.add_argument("--adapt-spsa-progress-every-s", type=float, default=60.0)
    p.add_argument("--adapt-seed", type=int, default=7)
    p.set_defaults(adapt_allow_repeats=True)
    p.add_argument("--adapt-allow-repeats", dest="adapt_allow_repeats", action="store_true")
    p.add_argument("--adapt-no-repeats", dest="adapt_allow_repeats", action="store_false")
    p.set_defaults(adapt_finite_angle_fallback=True)
    p.add_argument(
        "--adapt-finite-angle-fallback",
        dest="adapt_finite_angle_fallback",
        action="store_true",
        help="If gradients are below threshold, scan finite ±theta probes to continue ADAPT when beneficial.",
    )
    p.add_argument(
        "--adapt-no-finite-angle-fallback",
        dest="adapt_finite_angle_fallback",
        action="store_false",
        help="Disable finite-angle fallback and stop immediately when gradients are below threshold.",
    )
    p.add_argument(
        "--adapt-finite-angle",
        type=float,
        default=0.1,
        help="Probe angle theta used by finite-angle fallback (tests ±theta).",
    )
    p.add_argument(
        "--adapt-finite-angle-min-improvement",
        type=float,
        default=1e-12,
        help="Minimum required energy drop from finite-angle probe to accept fallback selection.",
    )
    p.add_argument(
        "--adapt-disable-hh-seed",
        action="store_true",
        help="Disable HH preconditioning with the compact quadrature seed block.",
    )
    p.add_argument(
        "--adapt-gradient-parity-check",
        action="store_true",
        help=(
            "Debug-only parity guard: compare one reused-Hpsi gradient per ADAPT depth "
            f"against the legacy commutator path (rtol={_ADAPT_GRADIENT_PARITY_RTOL:.1e})."
        ),
    )
    p.add_argument(
        "--adapt-drop-floor",
        type=float,
        default=None,
        help=(
            "Energy-drop floor for plateau stop policy (drop = ΔE_abs(d-1)-ΔE_abs(d)). "
            "If omitted, HH phase1_v1/phase2_v1/phase3_v1 resolves to 5e-4; Hubbard / HH legacy stay off. "
            "Pass a negative value to disable explicitly."
        ),
    )
    p.add_argument(
        "--adapt-drop-patience",
        type=int,
        default=None,
        help=(
            "Consecutive low-drop depth count required to trigger drop plateau stop. "
            "If omitted, HH phase1_v1/phase2_v1/phase3_v1 resolves to 3; Hubbard / HH legacy stay off."
        ),
    )
    p.add_argument(
        "--adapt-drop-min-depth",
        type=int,
        default=None,
        help=(
            "Minimum ADAPT depth before evaluating the drop plateau stop policy. "
            "If omitted, HH phase1_v1/phase2_v1/phase3_v1 resolves to 12; Hubbard / HH legacy stay off."
        ),
    )
    p.add_argument(
        "--adapt-grad-floor",
        type=float,
        default=None,
        help=(
            "Optional secondary gradient floor for drop plateau stop. "
            "If omitted, HH phase1_v1/phase2_v1/phase3_v1 resolves to 2e-2; Hubbard / HH legacy disable it. "
            "Pass a negative value to disable explicitly."
        ),
    )
    p.add_argument(
        "--adapt-eps-energy-min-extra-depth",
        type=int,
        default=-1,
        help=(
            "Minimum extra ADAPT depth before the eps-energy guard can trigger. "
            "Use -1 to auto-set this to L. Telemetry-only in HH phase1_v1/phase2_v1/phase3_v1."
        ),
    )
    p.add_argument(
        "--adapt-eps-energy-patience",
        type=int,
        default=-1,
        help=(
            "Consecutive low-improvement depth count required for the eps-energy guard. "
            "Use -1 to auto-set this to L. Telemetry-only in HH phase1_v1/phase2_v1/phase3_v1."
        ),
    )
    p.add_argument(
        "--adapt-ref-json",
        type=Path,
        default=None,
        help=(
            "Import reference state from an ADAPT/VQE JSON initial_state.amplitudes_qn_to_q0. "
            "In HH phase1_v1/phase2_v1/phase3_v1 reruns, metadata-compatible warm/ADAPT JSON can also "
            "reuse ground_state exact-energy fields."
        ),
    )
    p.add_argument("--paop-r", type=int, default=1, help="Cloud radius R for paop_full/paop_lf_full pools.")
    p.add_argument(
        "--paop-split-paulis",
        action="store_true",
        help="Split composite PAOP generators into single Pauli terms.",
    )
    p.add_argument(
        "--paop-prune-eps",
        type=float,
        default=0.0,
        help="Prune PAOP Pauli terms below this absolute coefficient threshold.",
    )
    p.add_argument(
        "--paop-normalization",
        choices=["none", "fro", "maxcoeff"],
        default="none",
        help="Normalization mode for PAOP generators before ADAPT search.",
    )

    # Trotter dynamics
    p.add_argument("--t-final", type=float, default=20.0)
    p.add_argument("--num-times", type=int, default=201)
    p.add_argument("--suzuki-order", type=int, default=2)
    p.add_argument("--trotter-steps", type=int, default=64)

    p.add_argument("--initial-state-source", choices=["exact", "adapt_vqe", "hf"], default="adapt_vqe")

    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-pdf", type=Path, default=None)
    p.add_argument(
        "--dense-eigh-max-dim",
        type=int,
        default=8192,
        help="Skip full dense Hamiltonian diagonalization when Hilbert dimension exceeds this threshold.",
    )
    p.add_argument("--skip-pdf", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _ai_log("hardcoded_adapt_main_start", settings=vars(args))
    run_command = current_command_string()
    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    json_dir = artifacts_dir / "json"
    pdf_dir = artifacts_dir / "pdf"
    json_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    prob = "hh" if str(args.problem).strip().lower() == "hh" else "hubbard"
    output_json = args.output_json or (json_dir / f"adapt_{prob}_L{args.L}.json")
    output_pdf = args.output_pdf or (pdf_dir / f"adapt_{prob}_L{args.L}.pdf")

    # 1) Build Hamiltonian
    problem_key = str(args.problem).strip().lower()
    if problem_key == "hh":
        h_poly = build_hubbard_holstein_hamiltonian(
            dims=int(args.L),
            J=float(args.t),
            U=float(args.u),
            omega0=float(args.omega0),
            g=float(args.g_ep),
            n_ph_max=int(args.n_ph_max),
            boson_encoding=str(args.boson_encoding),
            v_t=None,
            v0=float(args.dv),
            t_eval=None,
            repr_mode="JW",
            indexing=str(args.ordering),
            pbc=(str(args.boundary) == "periodic"),
            include_zero_point=True,
        )
    else:
        h_poly = build_hubbard_hamiltonian(
            dims=int(args.L),
            t=float(args.t),
            U=float(args.u),
            v=float(args.dv),
            repr_mode="JW",
            indexing=str(args.ordering),
            pbc=(str(args.boundary) == "periodic"),
        )

    native_order, coeff_map_exyz = _collect_hardcoded_terms_exyz(h_poly)
    _ai_log("hardcoded_adapt_hamiltonian_built", L=int(args.L), num_terms=int(len(coeff_map_exyz)))
    if args.term_order == "native":
        ordered_labels_exyz = list(native_order)
    else:
        ordered_labels_exyz = sorted(coeff_map_exyz)
    nq_total = len(ordered_labels_exyz[0]) if ordered_labels_exyz else 2 * int(args.L)
    hilbert_dim = 1 << int(nq_total)
    dense_eigh_enabled = bool(hilbert_dim <= int(args.dense_eigh_max_dim))
    hmat: np.ndarray | None = None
    if dense_eigh_enabled:
        hmat = _build_hamiltonian_matrix(coeff_map_exyz)
    else:
        _ai_log(
            "hardcoded_adapt_dense_eigh_skipped",
            hilbert_dim=int(hilbert_dim),
            dense_eigh_max_dim=int(args.dense_eigh_max_dim),
        )

    psi_ref_override_for_adapt: np.ndarray | None = None
    adapt_ref_import: dict[str, Any] | None = None
    adapt_ref_meta: Mapping[str, Any] | None = None
    adapt_ref_base_depth = 0
    ansatz_input_state_for_adapt: np.ndarray | None = None
    ansatz_input_state_source = "hf"
    ansatz_input_state_kind: str | None = "reference_state"
    if args.adapt_ref_json is not None:
        psi_ref_override_for_adapt, adapt_ref_meta = _load_adapt_initial_state(
            Path(args.adapt_ref_json),
            int(nq_total),
        )
        ansatz_input_state_for_adapt = np.asarray(psi_ref_override_for_adapt, dtype=complex).reshape(-1)
        adapt_ref_vqe = adapt_ref_meta.get("adapt_vqe", {})
        if isinstance(adapt_ref_vqe, dict):
            ref_depth_raw = adapt_ref_vqe.get("ansatz_depth")
            try:
                ref_depth_val = int(ref_depth_raw)
                if ref_depth_val >= 0:
                    adapt_ref_base_depth = int(ref_depth_val)
            except (TypeError, ValueError):
                adapt_ref_base_depth = 0
        adapt_ref_import = {
            "path": str(Path(args.adapt_ref_json)),
            "initial_state_source": adapt_ref_meta.get("initial_state_source"),
            "initial_state_handoff_state_kind": adapt_ref_meta.get("initial_state_handoff_state_kind"),
            "settings": adapt_ref_meta.get("settings", {}),
            "adapt_vqe": adapt_ref_meta.get("adapt_vqe", {}),
            "adapt_ref_base_depth": int(adapt_ref_base_depth),
        }
        ansatz_input_state_source = str(adapt_ref_meta.get("initial_state_source") or "adapt_ref_json")
        raw_kind = adapt_ref_meta.get("initial_state_handoff_state_kind")
        ansatz_input_state_kind = None if raw_kind in {None, ""} else str(raw_kind)
        _ai_log(
            "hardcoded_adapt_ref_json_loaded",
            path=str(Path(args.adapt_ref_json)),
            initial_state_source=adapt_ref_meta.get("initial_state_source"),
            adapt_ref_base_depth=int(adapt_ref_base_depth),
        )
    else:
        ansatz_input_state_for_adapt, ansatz_input_state_source, ansatz_input_state_kind = _default_adapt_input_state(
            problem=str(problem_key),
            num_sites=int(args.L),
            ordering=str(args.ordering),
            n_ph_max=int(args.n_ph_max),
            boson_encoding=str(args.boson_encoding),
        )

    # Sector-filtered exact ground state: ADAPT-VQE preserves particle number,
    # so compare against the GS within the same (n_alpha, n_beta) sector.
    # For HH: use fermion-only sector filtering (phonon qubits free).
    num_particles_main = half_filled_num_particles(int(args.L))
    gs_energy_exact, gs_energy_source, exact_energy_reuse_mismatches = _resolve_exact_energy_override_from_adapt_ref(
        adapt_ref_meta=adapt_ref_meta,
        args=args,
        problem=problem_key,
        continuation_mode=str(args.adapt_continuation_mode),
    )
    if gs_energy_exact is None:
        gs_energy_exact = _exact_gs_energy_for_problem(
            h_poly,
            problem=problem_key,
            num_sites=int(args.L),
            num_particles=num_particles_main,
            indexing=str(args.ordering),
            n_ph_max=int(args.n_ph_max),
            boson_encoding=str(args.boson_encoding),
            t=float(args.t),
            u=float(args.u),
            dv=float(args.dv),
            omega0=float(args.omega0),
            g_ep=float(args.g_ep),
            boundary=str(args.boundary),
        )
        gs_energy_source = "computed"
    if adapt_ref_import is not None:
        adapt_ref_import["exact_energy_reused"] = bool(gs_energy_source == "adapt_ref_json")
        adapt_ref_import["exact_energy_reuse_mismatches"] = list(exact_energy_reuse_mismatches)
        if gs_energy_source == "adapt_ref_json":
            adapt_ref_import["reused_exact_energy"] = float(gs_energy_exact)
            _ai_log(
                "hardcoded_adapt_exact_energy_reused",
                path=str(Path(args.adapt_ref_json)),
                exact_energy=float(gs_energy_exact),
            )
        elif (
            problem_key == "hh"
            and str(args.adapt_continuation_mode).strip().lower() in _HH_STAGED_CONTINUATION_MODES
        ):
            _ai_log(
                "hardcoded_adapt_exact_energy_reuse_skipped",
                path=str(Path(args.adapt_ref_json)),
                mismatch_count=int(len(exact_energy_reuse_mismatches)),
                has_candidate=bool(_resolve_exact_energy_from_payload(adapt_ref_meta or {})),
            )

    # Full-spectrum eigenvectors are optional (memory heavy for large HH spaces).
    psi_exact_ground: np.ndarray
    if hmat is not None:
        evals_full, evecs_full = np.linalg.eigh(hmat)
        psi_exact_ground_opt: np.ndarray | None = None
        for idx in range(len(evals_full)):
            if abs(evals_full[idx] - gs_energy_exact) < 1e-8:
                psi_exact_ground_opt = _normalize_state(
                    np.asarray(evecs_full[:, idx], dtype=complex).reshape(-1)
                )
                break
        if psi_exact_ground_opt is None:
            gs_idx_fallback = int(np.argmin(evals_full))
            psi_exact_ground_opt = _normalize_state(
                np.asarray(evecs_full[:, gs_idx_fallback], dtype=complex).reshape(-1)
            )
        psi_exact_ground = psi_exact_ground_opt
    elif problem_key == "hh":
        psi_exact_ground = _normalize_state(
            np.asarray(
                hubbard_holstein_reference_state(
                    dims=int(args.L),
                    num_particles=num_particles_main,
                    n_ph_max=int(args.n_ph_max),
                    boson_encoding=str(args.boson_encoding),
                    indexing=str(args.ordering),
                ),
                dtype=complex,
            ).reshape(-1)
        )
    else:
        psi_exact_ground = _normalize_state(
            np.asarray(
                hartree_fock_statevector(int(args.L), num_particles_main, indexing=str(args.ordering)),
                dtype=complex,
            ).reshape(-1)
        )

    # 2) Run ADAPT-VQE
    adapt_payload: dict[str, Any]
    try:
        adapt_payload, psi_adapt = _run_hardcoded_adapt_vqe(
            h_poly=h_poly,
            num_sites=int(args.L),
            ordering=str(args.ordering),
            problem=str(args.problem),
            adapt_pool=(str(args.adapt_pool) if args.adapt_pool is not None else None),
            t=float(args.t),
            u=float(args.u),
            dv=float(args.dv),
            boundary=str(args.boundary),
            omega0=float(args.omega0),
            g_ep=float(args.g_ep),
            n_ph_max=int(args.n_ph_max),
            boson_encoding=str(args.boson_encoding),
            max_depth=int(args.adapt_max_depth),
            eps_grad=float(args.adapt_eps_grad),
            eps_energy=float(args.adapt_eps_energy),
            maxiter=int(args.adapt_maxiter),
            seed=int(args.adapt_seed),
            adapt_inner_optimizer=str(args.adapt_inner_optimizer),
            adapt_spsa_a=float(args.adapt_spsa_a),
            adapt_spsa_c=float(args.adapt_spsa_c),
            adapt_spsa_alpha=float(args.adapt_spsa_alpha),
            adapt_spsa_gamma=float(args.adapt_spsa_gamma),
            adapt_spsa_A=float(args.adapt_spsa_A),
            adapt_spsa_avg_last=int(args.adapt_spsa_avg_last),
            adapt_spsa_eval_repeats=int(args.adapt_spsa_eval_repeats),
            adapt_spsa_eval_agg=str(args.adapt_spsa_eval_agg),
            adapt_spsa_callback_every=int(args.adapt_spsa_callback_every),
            adapt_spsa_progress_every_s=float(args.adapt_spsa_progress_every_s),
            adapt_state_backend=str(args.adapt_state_backend),
            adapt_reopt_policy=str(args.adapt_reopt_policy),
            adapt_window_size=int(args.adapt_window_size),
            adapt_window_topk=int(args.adapt_window_topk),
            adapt_full_refit_every=int(args.adapt_full_refit_every),
            adapt_final_full_refit=bool(str(args.adapt_final_full_refit).strip().lower() == "true"),
            adapt_continuation_mode=str(args.adapt_continuation_mode),
            allow_repeats=bool(args.adapt_allow_repeats),
            finite_angle_fallback=bool(args.adapt_finite_angle_fallback),
            finite_angle=float(args.adapt_finite_angle),
            finite_angle_min_improvement=float(args.adapt_finite_angle_min_improvement),
            adapt_drop_floor=(float(args.adapt_drop_floor) if args.adapt_drop_floor is not None else None),
            adapt_drop_patience=(int(args.adapt_drop_patience) if args.adapt_drop_patience is not None else None),
            adapt_drop_min_depth=(int(args.adapt_drop_min_depth) if args.adapt_drop_min_depth is not None else None),
            adapt_grad_floor=(float(args.adapt_grad_floor) if args.adapt_grad_floor is not None else None),
            adapt_eps_energy_min_extra_depth=int(args.adapt_eps_energy_min_extra_depth),
            adapt_eps_energy_patience=int(args.adapt_eps_energy_patience),
            adapt_ref_base_depth=int(adapt_ref_base_depth),
            paop_r=int(args.paop_r),
            paop_split_paulis=bool(args.paop_split_paulis),
            paop_prune_eps=float(args.paop_prune_eps),
            paop_normalization=str(args.paop_normalization),
            disable_hh_seed=bool(args.adapt_disable_hh_seed),
            psi_ref_override=psi_ref_override_for_adapt,
            adapt_gradient_parity_check=bool(args.adapt_gradient_parity_check),
            exact_gs_override=float(gs_energy_exact),
            phase1_lambda_F=float(args.phase1_lambda_F),
            phase1_lambda_compile=float(args.phase1_lambda_compile),
            phase1_lambda_measure=float(args.phase1_lambda_measure),
            phase1_lambda_leak=float(args.phase1_lambda_leak),
            phase1_score_z_alpha=float(args.phase1_score_z_alpha),
            phase1_shortlist_size=int(args.phase1_shortlist_size),
            phase1_probe_max_positions=int(args.phase1_probe_max_positions),
            phase1_plateau_patience=int(args.phase1_plateau_patience),
            phase1_trough_margin_ratio=float(args.phase1_trough_margin_ratio),
            phase1_prune_enabled=bool(args.phase1_prune_enabled),
            phase1_prune_fraction=float(args.phase1_prune_fraction),
            phase1_prune_max_candidates=int(args.phase1_prune_max_candidates),
            phase1_prune_max_regression=float(args.phase1_prune_max_regression),
            phase2_shortlist_fraction=float(args.phase2_shortlist_fraction),
            phase2_shortlist_size=int(args.phase2_shortlist_size),
            phase2_lambda_H=float(args.phase2_lambda_H),
            phase2_rho=float(args.phase2_rho),
            phase2_gamma_N=float(args.phase2_gamma_N),
            phase2_enable_batching=bool(args.phase2_enable_batching),
            phase2_batch_target_size=int(args.phase2_batch_target_size),
            phase2_batch_size_cap=int(args.phase2_batch_size_cap),
            phase2_batch_near_degenerate_ratio=float(args.phase2_batch_near_degenerate_ratio),
            phase3_motif_source_json=(Path(args.phase3_motif_source_json) if args.phase3_motif_source_json is not None else None),
            phase3_symmetry_mitigation_mode=str(args.phase3_symmetry_mitigation_mode),
            phase3_enable_rescue=bool(args.phase3_enable_rescue),
            phase3_lifetime_cost_mode=str(args.phase3_lifetime_cost_mode),
            phase3_runtime_split_mode=str(args.phase3_runtime_split_mode),
            phase3_backend_cost_mode=str(args.phase3_backend_cost_mode),
            phase3_backend_name=(None if args.phase3_backend_name in {None, ""} else str(args.phase3_backend_name)),
            phase3_backend_shortlist=(
                []
                if args.phase3_backend_shortlist in {None, ""}
                else [str(tok).strip() for tok in str(args.phase3_backend_shortlist).split(",") if str(tok).strip() != ""]
            ),
            phase3_backend_transpile_seed=int(args.phase3_backend_transpile_seed),
            phase3_backend_optimization_level=int(args.phase3_backend_optimization_level),
        )
    except Exception as exc:
        _ai_log("hardcoded_adapt_vqe_failed", L=int(args.L), error=str(exc))
        adapt_payload = {
            "success": False,
            "method": f"hardcoded_adapt_vqe_{str(args.adapt_pool).lower()}",
            "energy": None,
            "adapt_inner_optimizer": str(args.adapt_inner_optimizer),
            "error": str(exc),
        }
        if str(args.adapt_inner_optimizer).strip().upper() == "SPSA":
            adapt_payload["adapt_spsa"] = {
                "a": float(args.adapt_spsa_a),
                "c": float(args.adapt_spsa_c),
                "alpha": float(args.adapt_spsa_alpha),
                "gamma": float(args.adapt_spsa_gamma),
                "A": float(args.adapt_spsa_A),
                "avg_last": int(args.adapt_spsa_avg_last),
                "eval_repeats": int(args.adapt_spsa_eval_repeats),
                "eval_agg": str(args.adapt_spsa_eval_agg),
            }
        psi_adapt = psi_exact_ground

    # 3) Select initial state for dynamics
    num_particles = half_filled_num_particles(int(args.L))
    if problem_key == "hh":
        psi_hf = _normalize_state(
            np.asarray(
                hubbard_holstein_reference_state(
                    dims=int(args.L),
                    num_particles=num_particles,
                    n_ph_max=int(args.n_ph_max),
                    boson_encoding=str(args.boson_encoding),
                    indexing=str(args.ordering),
                ),
                dtype=complex,
            ).reshape(-1)
        )
    else:
        psi_hf = _normalize_state(
            np.asarray(
                hartree_fock_statevector(int(args.L), num_particles, indexing=str(args.ordering)),
                dtype=complex,
            ).reshape(-1)
        )

    if args.initial_state_source == "adapt_vqe" and bool(adapt_payload.get("success", False)):
        psi0 = psi_adapt
        _ai_log("hardcoded_adapt_initial_state_selected", source="adapt_vqe")
    elif args.initial_state_source == "adapt_vqe":
        raise RuntimeError("Requested --initial-state-source adapt_vqe but ADAPT-VQE failed.")
    elif args.initial_state_source == "hf":
        psi0 = psi_hf
        _ai_log("hardcoded_adapt_initial_state_selected", source="hf")
    else:
        psi0 = psi_exact_ground
        _ai_log("hardcoded_adapt_initial_state_selected", source="exact")

    # 4) Trajectory
    if hmat is None:
        trajectory = []
        _ai_log(
            "hardcoded_adapt_trajectory_skipped_no_dense_hmat",
            hilbert_dim=int(hilbert_dim),
            dense_eigh_max_dim=int(args.dense_eigh_max_dim),
        )
    else:
        trajectory, _exact_states = _simulate_trajectory(
            num_sites=int(args.L),
            psi0=psi0,
            hmat=hmat,
            ordered_labels_exyz=ordered_labels_exyz,
            coeff_map_exyz=coeff_map_exyz,
            trotter_steps=int(args.trotter_steps),
            t_final=float(args.t_final),
            num_times=int(args.num_times),
            suzuki_order=int(args.suzuki_order),
        )

    # 5) Emit JSON
    initial_state_source_resolved = str(
        args.initial_state_source if args.initial_state_source != "adapt_vqe" or adapt_payload.get("success") else "exact"
    )
    initial_state_kind_resolved = (
        "prepared_state"
        if (args.initial_state_source == "adapt_vqe" and bool(adapt_payload.get("success", False)))
        else "reference_state"
    )

    payload: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "hardcoded_adapt",
        "settings": {
            "L": int(args.L),
            "t": float(args.t),
            "u": float(args.u),
            "problem": str(args.problem),
            "omega0": float(args.omega0),
            "g_ep": float(args.g_ep),
            "n_ph_max": int(args.n_ph_max),
            "boson_encoding": str(args.boson_encoding),
            "dv": float(args.dv),
            "boundary": str(args.boundary),
            "ordering": str(args.ordering),
            "t_final": float(args.t_final),
            "num_times": int(args.num_times),
            "suzuki_order": int(args.suzuki_order),
            "trotter_steps": int(args.trotter_steps),
            "term_order": str(args.term_order),
            "dense_eigh_max_dim": int(args.dense_eigh_max_dim),
            "dense_eigh_enabled": bool(dense_eigh_enabled),
            "hilbert_dim": int(hilbert_dim),
            "adapt_pool": (str(args.adapt_pool) if args.adapt_pool is not None else None),
            "adapt_continuation_mode": str(args.adapt_continuation_mode),
            "adapt_max_depth": int(args.adapt_max_depth),
            "adapt_eps_grad": float(args.adapt_eps_grad),
            "adapt_eps_energy": float(args.adapt_eps_energy),
            "adapt_inner_optimizer": str(args.adapt_inner_optimizer),
            "adapt_state_backend": str(args.adapt_state_backend),
            "adapt_finite_angle_fallback": bool(args.adapt_finite_angle_fallback),
            "adapt_finite_angle": float(args.adapt_finite_angle),
            "adapt_finite_angle_min_improvement": float(args.adapt_finite_angle_min_improvement),
            "adapt_drop_floor": (float(args.adapt_drop_floor) if args.adapt_drop_floor is not None else None),
            "adapt_drop_patience": (int(args.adapt_drop_patience) if args.adapt_drop_patience is not None else None),
            "adapt_drop_min_depth": (int(args.adapt_drop_min_depth) if args.adapt_drop_min_depth is not None else None),
            "adapt_grad_floor": (float(args.adapt_grad_floor) if args.adapt_grad_floor is not None else None),
            "adapt_drop_floor_resolved": adapt_payload.get("adapt_drop_floor_resolved"),
            "adapt_drop_patience_resolved": adapt_payload.get("adapt_drop_patience_resolved"),
            "adapt_drop_min_depth_resolved": adapt_payload.get("adapt_drop_min_depth_resolved"),
            "adapt_grad_floor_resolved": adapt_payload.get("adapt_grad_floor_resolved"),
            "adapt_drop_floor_source": adapt_payload.get("adapt_drop_floor_source"),
            "adapt_drop_patience_source": adapt_payload.get("adapt_drop_patience_source"),
            "adapt_drop_min_depth_source": adapt_payload.get("adapt_drop_min_depth_source"),
            "adapt_grad_floor_source": adapt_payload.get("adapt_grad_floor_source"),
            "adapt_drop_policy_source": adapt_payload.get("adapt_drop_policy_source"),
            "adapt_eps_energy_min_extra_depth": int(args.adapt_eps_energy_min_extra_depth),
            "adapt_eps_energy_patience": int(args.adapt_eps_energy_patience),
            "adapt_ref_base_depth": int(adapt_ref_base_depth),
            "adapt_gradient_parity_check": bool(args.adapt_gradient_parity_check),
            "adapt_seed": int(args.adapt_seed),
            "adapt_reopt_policy": str(args.adapt_reopt_policy),
            "adapt_window_size": int(args.adapt_window_size),
            "adapt_window_topk": int(args.adapt_window_topk),
            "adapt_full_refit_every": int(args.adapt_full_refit_every),
            "adapt_final_full_refit": str(args.adapt_final_full_refit),
            "phase1_lambda_F": float(args.phase1_lambda_F),
            "phase1_lambda_compile": float(args.phase1_lambda_compile),
            "phase1_lambda_measure": float(args.phase1_lambda_measure),
            "phase1_lambda_leak": float(args.phase1_lambda_leak),
            "phase1_score_z_alpha": float(args.phase1_score_z_alpha),
            "phase1_shortlist_size": int(args.phase1_shortlist_size),
            "phase1_probe_max_positions": int(args.phase1_probe_max_positions),
            "phase1_plateau_patience": int(args.phase1_plateau_patience),
            "phase1_trough_margin_ratio": float(args.phase1_trough_margin_ratio),
            "phase1_prune_enabled": bool(args.phase1_prune_enabled),
            "phase1_prune_fraction": float(args.phase1_prune_fraction),
            "phase1_prune_max_candidates": int(args.phase1_prune_max_candidates),
            "phase1_prune_max_regression": float(args.phase1_prune_max_regression),
            "phase2_shortlist_fraction": float(args.phase2_shortlist_fraction),
            "phase2_shortlist_size": int(args.phase2_shortlist_size),
            "phase2_lambda_H": float(args.phase2_lambda_H),
            "phase2_rho": float(args.phase2_rho),
            "phase2_gamma_N": float(args.phase2_gamma_N),
            "phase2_enable_batching": bool(args.phase2_enable_batching),
            "phase2_batch_target_size": int(args.phase2_batch_target_size),
            "phase2_batch_size_cap": int(args.phase2_batch_size_cap),
            "phase2_batch_near_degenerate_ratio": float(args.phase2_batch_near_degenerate_ratio),
            "phase3_motif_source_json": (
                str(args.phase3_motif_source_json)
                if args.phase3_motif_source_json is not None
                else None
            ),
            "phase3_symmetry_mitigation_mode": str(args.phase3_symmetry_mitigation_mode),
            "phase3_enable_rescue": bool(args.phase3_enable_rescue),
            "phase3_lifetime_cost_mode": str(args.phase3_lifetime_cost_mode),
            "phase3_runtime_split_mode": str(args.phase3_runtime_split_mode),
            "phase3_backend_cost_mode": str(args.phase3_backend_cost_mode),
            "phase3_backend_name": (
                None if args.phase3_backend_name in {None, ""} else str(args.phase3_backend_name)
            ),
            "phase3_backend_shortlist": (
                []
                if args.phase3_backend_shortlist in {None, ""}
                else [str(tok).strip() for tok in str(args.phase3_backend_shortlist).split(",") if str(tok).strip() != ""]
            ),
            "phase3_backend_transpile_seed": int(args.phase3_backend_transpile_seed),
            "phase3_backend_optimization_level": int(args.phase3_backend_optimization_level),
            "adapt_ref_json": (str(args.adapt_ref_json) if args.adapt_ref_json is not None else None),
            "paop_r": int(args.paop_r),
            "paop_split_paulis": bool(args.paop_split_paulis),
            "paop_prune_eps": float(args.paop_prune_eps),
            "paop_normalization": str(args.paop_normalization),
            "initial_state_source": str(args.initial_state_source),
        },
        "hamiltonian": {
            "num_qubits": int(
                len(ordered_labels_exyz[0])
                if ordered_labels_exyz
                else int(round(math.log2(hmat.shape[0])))
            ),
            "num_terms": int(len(coeff_map_exyz)),
            "coefficients_exyz": [
                {
                    "label_exyz": lbl,
                    "coeff": {"re": float(np.real(coeff_map_exyz[lbl])), "im": float(np.imag(coeff_map_exyz[lbl]))},
                }
                for lbl in ordered_labels_exyz
            ],
        },
        "ground_state": {
            "exact_energy": float(gs_energy_exact),
            "exact_energy_source": str(gs_energy_source),
            "method": (EXACT_METHOD if hmat is not None else "sector_exact_only_no_dense_eigh"),
        },
        "adapt_vqe": adapt_payload,
        "initial_state": build_statevector_manifest(
            psi_state=np.asarray(psi0, dtype=complex).reshape(-1),
            source=initial_state_source_resolved,
            handoff_state_kind=initial_state_kind_resolved,
            amplitude_cutoff=1e-12,
        ),
        "ansatz_input_state": build_statevector_manifest(
            psi_state=np.asarray(ansatz_input_state_for_adapt, dtype=complex).reshape(-1),
            source=str(ansatz_input_state_source),
            handoff_state_kind=ansatz_input_state_kind,
            amplitude_cutoff=1e-12,
        ),
        "trajectory": trajectory,
    }
    if str(args.adapt_inner_optimizer).strip().upper() == "SPSA":
        payload["settings"]["adapt_spsa"] = {
            "a": float(args.adapt_spsa_a),
            "c": float(args.adapt_spsa_c),
            "alpha": float(args.adapt_spsa_alpha),
            "gamma": float(args.adapt_spsa_gamma),
            "A": float(args.adapt_spsa_A),
            "avg_last": int(args.adapt_spsa_avg_last),
            "eval_repeats": int(args.adapt_spsa_eval_repeats),
            "eval_agg": str(args.adapt_spsa_eval_agg),
            "callback_every": int(args.adapt_spsa_callback_every),
            "progress_every_s": float(args.adapt_spsa_progress_every_s),
        }
    if adapt_ref_import is not None:
        adapt_ref_import["ansatz_input_state_persisted"] = True
        payload["adapt_ref_import"] = adapt_ref_import

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not args.skip_pdf:
        _write_pipeline_pdf(output_pdf, payload, run_command)

    _ai_log(
        "hardcoded_adapt_main_done",
        L=int(args.L),
        output_json=str(output_json),
        output_pdf=(str(output_pdf) if not args.skip_pdf else None),
        adapt_energy=adapt_payload.get("energy"),
    )
    print(f"Wrote JSON: {output_json}")
    if not args.skip_pdf:
        print(f"Wrote PDF:  {output_pdf}")


if __name__ == "__main__":
    main()

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

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/test/test_hh_continuation_stage_control.py
```py
from __future__ import annotations

from pipelines.hardcoded.hh_continuation_stage_control import (
    StageController,
    StageControllerConfig,
    allowed_positions,
    detect_trough,
    should_probe_positions,
)


def test_allowed_positions_are_unique_and_bounded() -> None:
    out = allowed_positions(
        n_params=8,
        append_position=8,
        active_window_indices=[7, 3, 3, 0],
        max_positions=5,
    )
    assert len(out) <= 5
    assert out[0] == 8
    assert len(out) == len(set(out))
    assert all(0 <= x <= 8 for x in out)
    assert 4 not in out


def test_should_probe_on_plateau() -> None:
    cfg = StageControllerConfig(plateau_patience=2)
    probe, reason = should_probe_positions(
        stage_name="core",
        drop_plateau_hits=2,
        max_grad=1e-2,
        eps_grad=1e-4,
        append_score=1.0,
        finite_angle_flat=False,
        repeated_family_flat=False,
        cfg=cfg,
    )
    assert probe is True
    assert reason == "drop_plateau"


def test_should_probe_on_eps_grad_only_when_fallback_flat() -> None:
    cfg = StageControllerConfig()
    probe, reason = should_probe_positions(
        stage_name="core",
        drop_plateau_hits=0,
        max_grad=1e-5,
        eps_grad=1e-4,
        append_score=1.0,
        finite_angle_flat=True,
        repeated_family_flat=False,
        cfg=cfg,
    )
    assert probe is True
    assert reason == "eps_grad_flat"

    probe, reason = should_probe_positions(
        stage_name="core",
        drop_plateau_hits=0,
        max_grad=1e-5,
        eps_grad=1e-4,
        append_score=1.0,
        finite_angle_flat=False,
        repeated_family_flat=False,
        cfg=cfg,
    )
    assert probe is False
    assert reason == "default_append_only"


def test_should_probe_on_repeated_family_flat() -> None:
    cfg = StageControllerConfig()
    probe, reason = should_probe_positions(
        stage_name="core",
        drop_plateau_hits=0,
        max_grad=1e-2,
        eps_grad=1e-4,
        append_score=1.0,
        finite_angle_flat=False,
        repeated_family_flat=True,
        cfg=cfg,
    )
    assert probe is True
    assert reason == "family_repeat_flat"


def test_detect_trough_requires_positive_non_append_lcb() -> None:
    assert detect_trough(
        append_score=0.1,
        best_non_append_score=0.2,
        best_non_append_g_lcb=0.0,
        margin_ratio=1.0,
        append_admit_threshold=0.05,
    ) is False
    assert detect_trough(
        append_score=0.1,
        best_non_append_score=0.2,
        best_non_append_g_lcb=1e-3,
        margin_ratio=1.0,
        append_admit_threshold=0.05,
    ) is True


def test_detect_trough_accepts_non_append_when_append_below_floor() -> None:
    assert detect_trough(
        append_score=0.03,
        best_non_append_score=0.06,
        best_non_append_g_lcb=1e-3,
        margin_ratio=2.0,
        append_admit_threshold=0.05,
    ) is True


def test_stage_controller_seed_to_core_to_residual() -> None:
    ctrl = StageController(StageControllerConfig(plateau_patience=2))
    ctrl.start_with_seed()
    stage, reason = ctrl.resolve_stage_transition(
        drop_plateau_hits=0,
        trough_detected=False,
        residual_opened=False,
    )
    assert stage == "core"
    assert reason == "seed_complete"
    stage, reason = ctrl.resolve_stage_transition(
        drop_plateau_hits=2,
        trough_detected=False,
        residual_opened=False,
    )
    assert stage == "residual"
    assert reason == "plateau_without_trough"

```

File: /Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/src/quantum/ansatz_parameterization.py
```py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RotationTermSpec:
    pauli_exyz: str
    coeff_real: float
    nq: int


@dataclass(frozen=True)
class GeneratorParameterBlock:
    candidate_label: str
    logical_index: int
    runtime_start: int
    terms: tuple[RotationTermSpec, ...]

    @property
    def runtime_count(self) -> int:
        return int(len(self.terms))

    @property
    def runtime_stop(self) -> int:
        return int(self.runtime_start + len(self.terms))


@dataclass(frozen=True)
class AnsatzParameterLayout:
    mode: str
    term_order: str
    ignore_identity: bool
    coefficient_tolerance: float
    blocks: tuple[GeneratorParameterBlock, ...]

    @property
    def logical_parameter_count(self) -> int:
        return int(len(self.blocks))

    @property
    def runtime_parameter_count(self) -> int:
        if not self.blocks:
            return 0
        return int(self.blocks[-1].runtime_stop)

    def runtime_slice_for_logical_index(self, logical_index: int) -> slice:
        block = self.blocks[int(logical_index)]
        return slice(int(block.runtime_start), int(block.runtime_stop))


"""
H = Σ_j c_j P_j
active_terms(H) = ordered, filtered {(P_j, Re[c_j])}
"""
def iter_runtime_rotation_terms(
    poly: Any,
    *,
    ignore_identity: bool = True,
    coefficient_tolerance: float = 1e-12,
    sort_terms: bool = True,
) -> tuple[RotationTermSpec, ...]:
    poly_terms = list(poly.return_polynomial())
    if not poly_terms:
        return tuple()

    nq = int(poly_terms[0].nqubit())
    id_label = "e" * nq
    ordered = list(poly_terms)
    if sort_terms:
        ordered.sort(key=lambda t: t.pw2strng())

    out: list[RotationTermSpec] = []
    tol = float(coefficient_tolerance)
    for term in ordered:
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if int(term.nqubit()) != nq:
            raise ValueError(
                f"Inconsistent polynomial qubit count: expected {nq}, got {term.nqubit()}."
            )
        if len(label) != nq:
            raise ValueError(f"Invalid Pauli label length for '{label}': expected {nq}.")
        if abs(coeff) < tol:
            continue
        if ignore_identity and label == id_label:
            continue
        if abs(coeff.imag) > tol:
            raise ValueError(f"Non-negligible imaginary coefficient in term {label}: {coeff}.")
        out.append(RotationTermSpec(pauli_exyz=label, coeff_real=float(coeff.real), nq=nq))
    return tuple(out)


"""
layout = ⨆_k block_k, block_k = ordered active Pauli terms for logical generator k
"""
def build_parameter_layout(
    terms: Sequence[Any],
    *,
    ignore_identity: bool = True,
    coefficient_tolerance: float = 1e-12,
    sort_terms: bool = True,
) -> AnsatzParameterLayout:
    blocks: list[GeneratorParameterBlock] = []
    runtime_start = 0
    for logical_index, term in enumerate(terms):
        specs = iter_runtime_rotation_terms(
            getattr(term, "polynomial"),
            ignore_identity=ignore_identity,
            coefficient_tolerance=coefficient_tolerance,
            sort_terms=sort_terms,
        )
        block = GeneratorParameterBlock(
            candidate_label=str(getattr(term, "label", f"term_{logical_index}")),
            logical_index=int(logical_index),
            runtime_start=int(runtime_start),
            terms=tuple(specs),
        )
        blocks.append(block)
        runtime_start += int(block.runtime_count)
    return AnsatzParameterLayout(
        mode="per_pauli_term_v1",
        term_order=("sorted" if sort_terms else "native"),
        ignore_identity=bool(ignore_identity),
        coefficient_tolerance=float(coefficient_tolerance),
        blocks=tuple(blocks),
    )


def runtime_insert_position(layout: AnsatzParameterLayout, logical_position: int) -> int:
    pos = max(0, min(int(logical_position), int(layout.logical_parameter_count)))
    if pos >= int(layout.logical_parameter_count):
        return int(layout.runtime_parameter_count)
    return int(layout.blocks[pos].runtime_start)


def runtime_indices_for_logical_indices(
    layout: AnsatzParameterLayout,
    logical_indices: Sequence[int],
) -> list[int]:
    out: list[int] = []
    for logical_index in logical_indices:
        idx = int(logical_index)
        if idx < 0 or idx >= int(layout.logical_parameter_count):
            continue
        block = layout.blocks[idx]
        out.extend(range(int(block.runtime_start), int(block.runtime_stop)))
    return [int(x) for x in out]


def expand_legacy_logical_theta(
    theta_logical: np.ndarray | Sequence[float],
    layout: AnsatzParameterLayout,
) -> np.ndarray:
    base = np.asarray(theta_logical, dtype=float).reshape(-1)
    if int(base.size) != int(layout.logical_parameter_count):
        raise ValueError(
            f"logical theta length mismatch: got {base.size}, expected {layout.logical_parameter_count}."
        )
    out = np.zeros(int(layout.runtime_parameter_count), dtype=float)
    for block, theta_val in zip(layout.blocks, base):
        if int(block.runtime_count) <= 0:
            continue
        out[block.runtime_start:block.runtime_stop] = float(theta_val)
    return out


def project_runtime_theta_block_mean(
    theta_runtime: np.ndarray | Sequence[float],
    layout: AnsatzParameterLayout,
) -> np.ndarray:
    arr = np.asarray(theta_runtime, dtype=float).reshape(-1)
    if int(arr.size) != int(layout.runtime_parameter_count):
        raise ValueError(
            f"runtime theta length mismatch: got {arr.size}, expected {layout.runtime_parameter_count}."
        )
    out = np.zeros(int(layout.logical_parameter_count), dtype=float)
    for block in layout.blocks:
        if int(block.runtime_count) <= 0:
            out[block.logical_index] = 0.0
            continue
        vals = arr[block.runtime_start:block.runtime_stop]
        out[block.logical_index] = float(np.mean(vals))
    return out


def serialize_layout(layout: AnsatzParameterLayout) -> dict[str, Any]:
    return {
        "mode": str(layout.mode),
        "term_order": str(layout.term_order),
        "ignore_identity": bool(layout.ignore_identity),
        "coefficient_tolerance": float(layout.coefficient_tolerance),
        "logical_operator_count": int(layout.logical_parameter_count),
        "runtime_parameter_count": int(layout.runtime_parameter_count),
        "blocks": [
            {
                "candidate_label": str(block.candidate_label),
                "logical_index": int(block.logical_index),
                "runtime_start": int(block.runtime_start),
                "runtime_count": int(block.runtime_count),
                "runtime_terms_exyz": [
                    {
                        "pauli_exyz": str(spec.pauli_exyz),
                        "coeff_re": float(spec.coeff_real),
                        "coeff_im": 0.0,
                        "nq": int(spec.nq),
                    }
                    for spec in block.terms
                ],
            }
            for block in layout.blocks
        ],
    }


def deserialize_layout(payload: Mapping[str, Any]) -> AnsatzParameterLayout:
    blocks_raw = payload.get("blocks", [])
    if not isinstance(blocks_raw, Sequence):
        raise ValueError("parameterization.blocks must be a sequence.")
    blocks: list[GeneratorParameterBlock] = []
    runtime_start_expected = 0
    for logical_index, raw_block in enumerate(blocks_raw):
        if not isinstance(raw_block, Mapping):
            raise ValueError("Each parameterization block must be a mapping.")
        terms_raw = raw_block.get("runtime_terms_exyz", [])
        if not isinstance(terms_raw, Sequence):
            raise ValueError("parameterization block runtime_terms_exyz must be a sequence.")
        terms: list[RotationTermSpec] = []
        for raw_term in terms_raw:
            if not isinstance(raw_term, Mapping):
                raise ValueError("parameterization runtime term must be a mapping.")
            label = str(raw_term.get("pauli_exyz", "")).strip()
            if label == "":
                raise ValueError("parameterization runtime term missing pauli_exyz.")
            coeff_re = float(raw_term.get("coeff_re", 0.0))
            coeff_im = float(raw_term.get("coeff_im", 0.0))
            if abs(coeff_im) > float(payload.get("coefficient_tolerance", 1e-12)):
                raise ValueError(f"parameterization runtime term {label} has non-zero coeff_im={coeff_im}.")
            nq = int(raw_term.get("nq", len(label)))
            if len(label) != nq:
                raise ValueError(f"parameterization runtime term {label} length mismatch vs nq={nq}.")
            terms.append(RotationTermSpec(pauli_exyz=label, coeff_real=coeff_re, nq=nq))
        runtime_start = int(raw_block.get("runtime_start", runtime_start_expected))
        if runtime_start != runtime_start_expected:
            raise ValueError(
                f"parameterization block runtime_start mismatch: got {runtime_start}, expected {runtime_start_expected}."
            )
        block = GeneratorParameterBlock(
            candidate_label=str(raw_block.get("candidate_label", f"term_{logical_index}")),
            logical_index=int(raw_block.get("logical_index", logical_index)),
            runtime_start=int(runtime_start),
            terms=tuple(terms),
        )
        if int(block.logical_index) != int(logical_index):
            raise ValueError(
                f"parameterization block logical_index mismatch: got {block.logical_index}, expected {logical_index}."
            )
        runtime_count = int(raw_block.get("runtime_count", block.runtime_count))
        if runtime_count != int(block.runtime_count):
            raise ValueError(
                f"parameterization block runtime_count mismatch for {block.candidate_label}: got {runtime_count}, expected {block.runtime_count}."
            )
        blocks.append(block)
        runtime_start_expected = int(block.runtime_stop)

    layout = AnsatzParameterLayout(
        mode=str(payload.get("mode", "per_pauli_term_v1")),
        term_order=str(payload.get("term_order", "sorted")),
        ignore_identity=bool(payload.get("ignore_identity", True)),
        coefficient_tolerance=float(payload.get("coefficient_tolerance", 1e-12)),
        blocks=tuple(blocks),
    )
    runtime_parameter_count = int(payload.get("runtime_parameter_count", layout.runtime_parameter_count))
    if runtime_parameter_count != int(layout.runtime_parameter_count):
        raise ValueError(
            f"parameterization runtime_parameter_count mismatch: got {runtime_parameter_count}, expected {layout.runtime_parameter_count}."
        )
    logical_operator_count = int(payload.get("logical_operator_count", layout.logical_parameter_count))
    if logical_operator_count != int(layout.logical_parameter_count):
        raise ValueError(
            f"parameterization logical_operator_count mismatch: got {logical_operator_count}, expected {layout.logical_parameter_count}."
        )
    return layout


__all__ = [
    "AnsatzParameterLayout",
    "GeneratorParameterBlock",
    "RotationTermSpec",
    "build_parameter_layout",
    "deserialize_layout",
    "expand_legacy_logical_theta",
    "iter_runtime_rotation_terms",
    "project_runtime_theta_block_mean",
    "runtime_indices_for_logical_indices",
    "runtime_insert_position",
    "serialize_layout",
]

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
