"""Anti-drift tests for the Paper-II run-command registry.

The point of these is that a flag renamed or removed in the runner fails here,
in under a second, instead of failing on a cluster six hours into a job -- or
worse, silently falling back to a config default, which is how an ablation arm
once tested a candidate pool of 8 while claiming to test the full pool.
"""

from __future__ import annotations

import pytest

from pipelines.time_dynamics.paper_ii_runs import (
    ARMS,
    AVQDS,
    EXCHANGE,
    EXCHANGE_GAMMA_COMPILED,
    EXCHANGE_LEGACY_PROXY_FIXED,
    EXCHANGE_Q_COMPILED,
    REGIMES,
    SIAM_NB2_PROOF_ARMS,
    SIAM_NB4_ARMS,
    build_siam_nb2_proof_run,
    build_siam_nb4_continuation_run,
    build_siam_nb4_screen_run,
    SeedNotBuiltError,
    available_regimes,
    DRIVES,
    GATES,
    HORIZONS,
    MCLACHLAN_L2_GATE,
    build_run,
    build_selector_comparison_run,
    paper_ii_runs,
)
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
    _build_parser,
)


def _parser():
    return _build_parser()


@pytest.mark.parametrize("arm", sorted(ARMS))
@pytest.mark.parametrize("gate", sorted(GATES))
def test_every_arm_gate_command_parses(arm: str, gate: str) -> None:
    """Every registered command must be valid for the runner's live CLI."""

    run = build_run(
        seed_path="seed.json", arm=arm, gate=gate, drive="fastweak",
        horizon="t2", output_json="out/run.json",
    )
    parsed = _parser().parse_args(list(run.argv()))
    assert parsed.artifact_json == "seed.json"
    assert parsed.output_json == "out/run.json"


def test_full_matrix_parses() -> None:
    parser = _parser()
    for run in paper_ii_runs():
        parser.parse_args(list(run.argv()))


def test_comparator_does_not_carry_this_routes_insertion_gate() -> None:
    """Layering our gate on AVQDS would misrepresent the comparator."""

    run = build_run(
        seed_path="seed.json", arm="avqds", gate=MCLACHLAN_L2_GATE.gate_id,
        drive="fastweak", horizon="t2", output_json="out/run.json",
    )
    assert "--insertion-gate-mode" not in run.argv()
    assert "--residual-ratio-threshold" not in run.argv()


def test_runner_has_no_candidate_pool_cap() -> None:
    """Every eligible direction reaches the structural search."""

    run = build_run(
        seed_path="seed.json", arm="exchange", gate=MCLACHLAN_L2_GATE.gate_id,
        drive="fastweak", horizon="t2", output_json="out/run.json",
    )
    parsed = _parser().parse_args(list(run.argv()))
    assert not hasattr(parsed, "max_structural_pool_size")
    assert not hasattr(parsed, "max_append_candidates")
    for flag in ("--max-structural-pool-size", "--max-append-candidates"):
        with pytest.raises(SystemExit):
            _parser().parse_args([*run.argv(), flag, "8"])


def test_subdivision_budget_is_the_repaired_value() -> None:
    run = build_run(
        seed_path="seed.json", arm="exchange", gate=MCLACHLAN_L2_GATE.gate_id,
        drive="fastweak", horizon="t2", output_json="out/run.json",
    )
    parsed = _parser().parse_args(list(run.argv()))
    assert int(parsed.solve_repair_max_local_subdivisions) >= 10


def test_run_ids_are_unique_across_the_matrix() -> None:
    runs = paper_ii_runs()
    assert len({r.run_id for r in runs}) == len(runs) == len(DRIVES) * len(ARMS)


def test_unknown_names_are_rejected() -> None:
    for kwargs in (
        {"arm": "nope"}, {"gate": "nope"}, {"drive": "nope"}, {"horizon": "nope"}
    ):
        base = dict(
            seed_path="seed.json", arm="exchange", gate=MCLACHLAN_L2_GATE.gate_id,
            drive="fastweak", horizon="t2", output_json="out/run.json",
        )
        base.update(kwargs)
        with pytest.raises(KeyError):
            build_run(**base)


def test_registries_are_non_empty() -> None:
    assert ARMS and GATES and DRIVES and HORIZONS


def test_campaign_cells_agree_with_the_registry() -> None:
    """A campaign cell and a registry run must configure the same trajectory.

    These were separate copies until 2026-08-23 and they drifted: the campaign
    module still carried `max_structural_pool_size: 8` after that cap had been
    identified as the lane's largest accuracy defect, so any campaign built
    from it would have silently reproduced the defect.
    """

    from pipelines.time_dynamics.campaign import (
        CampaignSpec,
        SeedSpec,
        uniform_threshold_plan,
    )

    methods = ("exchange", "append_only", "avqds")
    controls = ("state_motion_1e-2",)
    spec = CampaignSpec(
        campaign_id="parity",
        seeds=(SeedSpec("s", "seed.json", "hh", 1, "weak"),),
        method_ids=methods,
        controller_ids=controls,
        drive_ids=("fastweak",),
        horizon_ids=("t2",),
        threshold_plan=uniform_threshold_plan(
            method_ids=methods,
            controller_ids=controls,
            drive_ids=("fastweak",),
            thresholds=(1.0e-3,),
        ),
        numerics_id="rk4_ridge1e-7",
    )
    parser = _parser()
    by_arm = {}
    for cell in spec.cells():
        parsed = parser.parse_args(list(cell.runner_argv()))
        by_arm[cell.method_id] = parsed

    for arm_id in methods:
        run = build_run(
            seed_path="seed.json", arm=arm_id,
            gate=MCLACHLAN_L2_GATE.gate_id, drive="fastweak", horizon="t2",
            output_json="out/run.json", activation_cut=1.0e-3,
        )
        registry = _parser().parse_args(list(run.argv()))
        campaign = by_arm[arm_id]
        for field in (
            "integrator",
            "solve_repair_max_local_subdivisions",
            "solve_repair_state_motion_l2_step_max",
            "solve_repair_kink_eta_max",
            "append_schur_condition_gate",
            "insertion_gate_mode",
            "insertion_l2_cut",
            "max_joint_patch_evaluations",
            "t_final",
            "num_times",
            "drive_A",
            "drive_omega",
        ):
            assert getattr(campaign, field) == getattr(registry, field), (
                f"{arm_id}.{field}: campaign={getattr(campaign, field)!r} "
                f"registry={getattr(registry, field)!r}"
            )


def test_search_profiles_do_not_restrict_the_candidate_pool() -> None:
    from pipelines.time_dynamics.paper_ii_runs import PRODUCTION_STRUCTURE
    from pipelines.time_dynamics.paper_ii_runs import (
        SEARCH_PROFILES, SIAM_NB2_PROOF_SEARCH, SIAM_NB4_SEARCH_PROFILES,
    )

    profiles = (
        *SEARCH_PROFILES.values(), SIAM_NB2_PROOF_SEARCH,
        *SIAM_NB4_SEARCH_PROFILES.values(),
    )
    assert "--max-structural-pool-size" not in PRODUCTION_STRUCTURE
    for profile in profiles:
        assert "--max-structural-pool-size" not in profile.flags
        assert not hasattr(profile, "pool_size")


@pytest.mark.parametrize("regime", sorted(REGIMES))
def test_every_regime_builds_the_same_algorithm(regime: str) -> None:
    """Sweeping regimes must change the seed path and nothing else."""

    parser = _parser()
    reference = None
    run = build_run(
        regime=regime, arm="exchange", gate=MCLACHLAN_L2_GATE.gate_id,
        drive="fastweak", horizon="t10", output_json="out/run.json",
    )
    parsed = parser.parse_args(list(run.argv()))
    policy = (
        parsed.integrator,
        parsed.solve_repair_max_local_subdivisions,
        parsed.insertion_gate_mode,
        parsed.insertion_l2_cut,
        parsed.prune_target_policy,
        parsed.t_final,
        parsed.num_times,
    )
    baseline = build_run(
        regime="hh_snake_nph1", arm="exchange", gate=MCLACHLAN_L2_GATE.gate_id,
        drive="fastweak", horizon="t10", output_json="out/run.json",
    )
    base_parsed = _parser().parse_args(list(baseline.argv()))
    reference = (
        base_parsed.integrator,
        base_parsed.solve_repair_max_local_subdivisions,
        base_parsed.insertion_gate_mode,
        base_parsed.insertion_l2_cut,
        base_parsed.prune_target_policy,
        base_parsed.t_final,
        base_parsed.num_times,
    )
    assert policy == reference
    assert parsed.artifact_json == REGIMES[regime].seed_path


def test_unbuilt_regime_fails_before_the_runner_sees_it() -> None:
    missing = [k for k, r in REGIMES.items() if not r.available]
    if not missing:
        pytest.skip("every regime seed is built")
    with pytest.raises(SeedNotBuiltError):
        build_run(
            regime=missing[0], arm="exchange", gate=MCLACHLAN_L2_GATE.gate_id,
            drive="fastweak", horizon="t10", output_json="out/run.json",
            require_seed=True,
        )


def test_regime_requires_binary_aligned_phonon_cutoff() -> None:
    from pipelines.time_dynamics.paper_ii_runs import Regime

    with pytest.raises(ValueError, match="binary phonon register"):
        Regime(regime_id="bad", seed_path="x.json", n_ph_max=2)


def test_calibration_regime_is_available() -> None:
    assert "hh_snake_nph1" in available_regimes()


def test_build_run_requires_exactly_one_seed_source() -> None:
    for kwargs in ({}, {"regime": "hh_snake_nph1", "seed_path": "s.json"}):
        with pytest.raises(ValueError, match="exactly one"):
            build_run(
                arm="exchange", gate=MCLACHLAN_L2_GATE.gate_id, drive="fastweak",
                horizon="t2", output_json="out/run.json", **kwargs,
            )


def test_published_avqds_is_arm_plus_its_own_step_control() -> None:
    """Yao et al. specify Euler, Tikhonov xi=1e-6, and d theta_max = 5e-3.

    The published method is now expressed as a structural arm paired with its
    own step-control law, rather than an arm carrying a private numerics blob:
    that is what lets the step control be paired with the other rule too.
    """

    run = build_run(
        regime="hh_snake_nph1", arm="avqds_published",
        gate=MCLACHLAN_L2_GATE.gate_id, drive="strongfast", horizon="t10",
        step_control="delta_theta_5e-3", output_json="out/run.json",
    )
    parsed = _parser().parse_args(list(run.argv()))
    assert parsed.integrator == "euler"
    assert parsed.solve_repair is False
    assert parsed.certification_refit is False
    assert parsed.ridge_lambda == pytest.approx(1.0e-6)
    assert parsed.avqds_delta_theta_max == pytest.approx(5.0e-3)
    assert parsed.dynamics_policy == "avqds"


def test_published_avqds_rejects_nonpublished_numerics_or_control() -> None:
    base = dict(
        regime="hh_snake_nph1", arm="avqds_published",
        gate=MCLACHLAN_L2_GATE.gate_id, drive="strongfast", horizon="t10",
        output_json="out/run.json",
    )
    with pytest.raises(ValueError, match="fixes numerics"):
        build_run(**base, numerics="rk4_ridge1e-7")
    with pytest.raises(ValueError, match="fixes step_control"):
        build_run(**base, step_control="state_motion_1e-2")


def test_shared_stack_defaults_to_rk4() -> None:
    for arm in ("exchange", "append_only", "avqds"):
        run = build_run(
            regime="hh_snake_nph1", arm=arm, gate=MCLACHLAN_L2_GATE.gate_id,
            drive="strongfast", horizon="t10", output_json="out/run.json",
        )
        parsed = _parser().parse_args(list(run.argv()))
        assert parsed.integrator == "rk4"
        assert parsed.ridge_lambda == pytest.approx(1.0e-7)


def test_parameter_guard_composes_with_state_guard_under_rk4() -> None:
    """The controller ablation adds the parameter cap without changing RK4."""

    seen = {}
    for arm in ("exchange", "avqds"):
        for control in (
            "state_motion_1e-2",
            "state_motion_1e-2_plus_parameter_5e-3",
        ):
            run = build_run(
                regime="hh_snake_nph1", arm=arm, gate=MCLACHLAN_L2_GATE.gate_id,
                drive="strongfast", horizon="t10", step_control=control,
                output_json="out/run.json",
            )
            parsed = _parser().parse_args(list(run.argv()))
            seen[(arm, control)] = (
                parsed.solve_repair, parsed.solve_repair_parameter_step_max
            )
            # The inner numerical method is identical across all four cells.
            assert parsed.integrator == "rk4"
            assert parsed.ridge_lambda == pytest.approx(1.0e-7)
    assert seen[("exchange", "state_motion_1e-2_plus_parameter_5e-3")][1] == pytest.approx(5.0e-3)
    assert seen[("avqds", "state_motion_1e-2")][0] is True
    assert len({v for v in seen.values()}) == 2


def test_structural_rule_arms_share_the_same_inner_numerics() -> None:
    """The causal structural comparison excludes the published-method bundle."""

    values = set()
    for arm in ("exchange", "append_only", "avqds"):
        run = build_run(
            regime="hh_snake_nph1", arm=arm, gate=MCLACHLAN_L2_GATE.gate_id,
            drive="strongfast", horizon="t10", output_json="out/run.json",
        )
        parsed = _parser().parse_args(list(run.argv()))
        values.add((parsed.integrator, parsed.ridge_lambda,
                    parsed.solve_damping, parsed.pinv_rcond))
    assert len(values) == 1, values


def test_runner_transport_builds_the_live_exchange_config() -> None:
    """Parsing is insufficient: the CLI namespace must construct the config."""

    from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
        _build_parser,
        _support_patch_config_from_args,
    )

    args = _build_parser().parse_args(
        ["--artifact-json", "seed.json", "--output-json", "run.json"]
    )
    config = _support_patch_config_from_args(args)
    assert config.debt_policy == "drift_ranked"
    assert config.deletion_enabled is True
    assert config.prune_history_lambda == 0.0
    assert not hasattr(config, "append_min_time")


@pytest.mark.parametrize("policy", ("greedy", "block_exchange", "bounded_exhaustive"))
@pytest.mark.parametrize("gain", ("joint_batch", "singleton_factorized"))
def test_explicit_search_axes_build_matched_compiled_commands(policy, gain) -> None:
    from dataclasses import replace
    from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
        _support_patch_config_from_args,
    )

    base = build_run(
        seed_path="seed.json", arm=EXCHANGE_Q_COMPILED.arm_id,
        gate=MCLACHLAN_L2_GATE.gate_id, drive="fastweak", horizon="t2",
        output_json="out/run.json",
    )
    profile = replace(
        base.search_profile, profile_id=f"test_{policy}_{gain}",
        search_policy=policy, gain_evaluation=gain,
    )
    run = replace(base, search_profile=profile)
    parsed = _parser().parse_args(list(run.argv()))
    config = _support_patch_config_from_args(parsed)
    assert config.search_policy == policy
    assert config.gain_evaluation == gain
    assert config.max_insertion_batch_size == base.search_profile.insertion_cardinality
    assert run.search_profile.joint_patch_evaluations == base.search_profile.joint_patch_evaluations
    assert "--search-policy" not in base.argv()
    assert "--gain-evaluation" not in base.argv()


def test_search_profile_rejects_unknown_or_unused_settings() -> None:
    from dataclasses import replace
    from pipelines.time_dynamics.paper_ii_runs import PAPER_DECLARED_SEARCH

    for values in (
        {"search_policy": "exhaustive"},
        {"gain_evaluation": "factorized"},
        {"gain_evaluation": "singleton_factorized"},
        {"search_policy": "greedy", "exchange_insertion_frontier": 8},
    ):
        with pytest.raises(ValueError):
            replace(PAPER_DECLARED_SEARCH, **values)


@pytest.mark.parametrize("policy", ("greedy", "block_exchange", "bounded_exhaustive"))
@pytest.mark.parametrize("gain", ("joint_batch", "singleton_factorized"))
@pytest.mark.parametrize("cardinality", (1, 4))
def test_campaign_helper_pins_effective_axes_and_preserves_declared_budget(
    policy, gain, cardinality,
) -> None:
    from pipelines.time_dynamics.paper_ii_runs import DIAGNOSTIC_FAST_SEARCH
    from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
        _support_patch_config_from_args,
    )

    run = build_selector_comparison_run(
        search_policy=policy, gain_evaluation=gain, insertion_cardinality=cardinality,
        seed_path="seed.json", drive="fastweak", horizon="t2", output_json="out/run.json",
        search_profile=DIAGNOSTIC_FAST_SEARCH.profile_id,
        extra_flags=("--max-compiled-patches-per-checkpoint=23", "--compile-workers", "1"),
    )
    config = _support_patch_config_from_args(_parser().parse_args(list(run.argv())))
    assert config.exchange_route_bundle == "paper_ii_exchange_q_compiled_v1"
    assert (config.search_policy, config.gain_evaluation) == (policy, gain)
    assert config.max_insertion_batch_size == cardinality
    assert config.exchange_insertion_frontier is None
    assert config.interaction_frontier_widths is None
    assert config.max_joint_patch_evaluations == DIAGNOSTIC_FAST_SEARCH.joint_patch_evaluations
    assert config.max_certification_attempts_per_level == DIAGNOSTIC_FAST_SEARCH.certification_attempts_per_level
    assert config.max_certification_attempts_per_deletion_branch == DIAGNOSTIC_FAST_SEARCH.certification_attempts_per_deletion_branch
    assert config.max_insertion_rounds_per_checkpoint == DIAGNOSTIC_FAST_SEARCH.rounds_per_checkpoint
    assert config.max_compiled_patches_per_checkpoint == 23
    assert config.compile_workers == 1
    assert f"__{policy}__{gain}__a{cardinality}" in run.run_id
    assert run.argv().count("--exchange-route-bundle") == 1
    assert run.argv().count("--search-policy") == 1
    assert run.argv().count("--gain-evaluation") == 1


@pytest.mark.parametrize("extra_flags", (
    ("--search-policy", "family_frontiers"),
    ("--gain-eval=joint_batch",),
    ("--exchange-route-bun=paper_ii_generalized_exchange_v2",),
    ("--dynamics-policy", "avqds"),
    ("--exchange-insertion-frontier=8",),
    ("--interaction-frontier-widths", "2,4"),
    ("--max-insertion-batch-size=1",),
    ("--max-joint-patch-evaluations=1",),
))
def test_campaign_helper_rejects_effective_overrides(extra_flags) -> None:
    with pytest.raises(ValueError, match="override declared"):
        build_selector_comparison_run(
            search_policy="greedy", gain_evaluation="singleton_factorized",
            insertion_cardinality=4, seed_path="seed.json", drive="fastweak",
            horizon="t2", output_json="out/run.json", extra_flags=extra_flags,
        )


def test_campaign_helper_keeps_generic_defaults_and_profiles_unchanged() -> None:
    common = dict(seed_path="seed.json", drive="fastweak", horizon="t2", output_json="out/run.json")
    baseline = build_run(arm="exchange", **common)
    baseline_argv = baseline.argv()
    identifiers = set()
    for policy in ("greedy", "block_exchange", "bounded_exhaustive"):
        for gain in ("joint_batch", "singleton_factorized"):
            for cardinality in (1, 4):
                run = build_selector_comparison_run(
                    search_policy=policy, gain_evaluation=gain,
                    insertion_cardinality=cardinality, **common,
                )
                identifiers.add(run.run_id)
                assert (run.seed_path, run.drive, run.horizon, run.numerics, run.step_control) == (
                    baseline.seed_path, baseline.drive, baseline.horizon,
                    baseline.numerics, baseline.step_control,
                )
    assert len(identifiers) == 12
    assert build_run(arm="exchange", **common).argv() == baseline_argv
    assert "--exchange-route-bundle" not in baseline_argv
    assert "--search-policy" not in baseline_argv
    assert "--gain-evaluation" not in baseline_argv
    for overrides in (
        {"search_policy": "family_frontiers"},
        {"arm": "exchange"},
        {"search_profile": "paper_declared_exchange_v1"},
        {"insertion_cardinality": 1.5},
    ):
        settings = dict(search_policy="greedy", gain_evaluation="joint_batch", insertion_cardinality=4)
        settings.update(overrides)
        with pytest.raises(ValueError):
            build_selector_comparison_run(**settings, **common)


def test_runner_rejects_incompatible_search_before_seed_load(monkeypatch, capsys) -> None:
    from pipelines.time_dynamics.runners import ap_append_from_adapt_artifact as runner

    def forbidden_load(*args, **kwargs):
        pytest.fail("invalid search configuration reached seed loading")

    monkeypatch.setattr(runner, "_load_runtime_input_or_raise", forbidden_load)
    with pytest.raises(SystemExit) as exc:
        runner.main([
            "--artifact-json", "seed.json", "--output-json", "run.json",
            "--search-policy", "greedy",
        ])
    assert exc.value.code == 2
    assert "compiled-cost bundle" in capsys.readouterr().err


def test_selector_bundle_arms_are_exchange_plus_one_closed_bundle() -> None:
    from pipelines.time_dynamics.paper_ii_runs import (
        EXCHANGE,
        EXCHANGE_GAMMA_COMPILED,
        EXCHANGE_LEGACY_PROXY_FIXED,
        EXCHANGE_Q_COMPILED,
    )

    expected = {
        EXCHANGE_LEGACY_PROXY_FIXED: "paper_ii_exchange_legacy_proxy_fixed_v1",
        EXCHANGE_Q_COMPILED: "paper_ii_exchange_q_compiled_v1",
        EXCHANGE_GAMMA_COMPILED: "paper_ii_exchange_gamma_compiled_v1",
    }
    for arm, bundle in expected.items():
        assert arm.flags[: len(EXCHANGE.flags)] == EXCHANGE.flags
        parsed = _parser().parse_args(
            list(build_run(arm=arm.arm_id, drive="fastweak", horizon="smoke",
                           regime="hh_snake_nph1", output_json="x.json").argv())
        )
        assert parsed.exchange_route_bundle == bundle
        assert parsed.debt_policy == "drift_ranked"
    historical = _parser().parse_args(
        list(build_run(arm="exchange", drive="fastweak", horizon="smoke",
                       regime="hh_snake_nph1", output_json="x.json").argv())
    )
    assert historical.exchange_route_bundle == "paper_ii_generalized_exchange_v2"


# --- SIAM route arms -------------------------------------------------------
#
# The compiled-Q arm is EXCHANGE plus one --exchange-route-bundle, so it shares
# the exchange *policy* while carrying a different arm id.  The SIAM commands
# used to pass the arm id straight to --dynamics-policy (which admits only
# exchange/avqds) and to gate the route flags on ``arm_id == "exchange"``, so a
# bundle arm lost both its bundle flag and the mclachlan_l2 insertion gate.
#
# The frozen tuples below were captured from the pre-change code.  They exist so
# that any drift in the historical exchange and avqds argv fails here rather
# than silently changing what a locked SIAM run executes.

FROZEN_NB2_EXCHANGE_ARGV: tuple[str, ...] = (
    '--siam-seed-json',
    'SEED.json',
    '--t-final',
    '1.0',
    '--num-times',
    '26',
    '--integrator',
    'euler',
    '--ridge-lambda',
    '1e-06',
    '--solve-damping',
    '0.0',
    '--pinv-rcond',
    '1e-10',
    '--solve-repair',
    '--solve-repair-profile',
    'minimal',
    '--solve-repair-state-motion-l2-step-max',
    '5.0e-5',
    '--solve-repair-kink-eta-max',
    '5.0e-3',
    '--solve-repair-max-local-subdivisions',
    '11',
    '--fail-on-unsupported-steps',
    '--max-joint-patch-evaluations',
    '50000',
    '--max-certification-attempts-per-level',
    '12',
    '--max-certification-attempts-per-deletion-branch',
    '2',
    '--max-insertion-rounds-per-checkpoint',
    '12',
    '--max-insertion-batch-size',
    '1',
    '--no-append-schur-condition-gate',
    '--insertion-gate-mode',
    'mclachlan_l2',
    '--insertion-l2-cut',
    '1.0e-3',
    '--dynamics-policy',
    'exchange',
    '--debt-policy',
    'drift_ranked',
    '--prune-history-lambda',
    '0.0',
    '--prune-target-policy',
    'all_active',
    '--prune-ray-distance-tol',
    '2.0e-3',
    '--parameterization-mode',
    'logical_shared',
    '--record-statevector',
    '--require-complete-candidate-pool',
    '--progress-log-every',
    '1',
    '--code-revision',
    'REV',
    '--output-json',
    'OUT.json',
)

FROZEN_NB2_AVQDS_ARGV: tuple[str, ...] = (
    '--siam-seed-json',
    'SEED.json',
    '--t-final',
    '1.0',
    '--num-times',
    '26',
    '--integrator',
    'euler',
    '--ridge-lambda',
    '1e-06',
    '--solve-damping',
    '0.0',
    '--pinv-rcond',
    '1e-10',
    '--solve-repair',
    '--solve-repair-profile',
    'minimal',
    '--solve-repair-state-motion-l2-step-max',
    '5.0e-5',
    '--solve-repair-kink-eta-max',
    '5.0e-3',
    '--solve-repair-max-local-subdivisions',
    '11',
    '--fail-on-unsupported-steps',
    '--max-joint-patch-evaluations',
    '50000',
    '--max-certification-attempts-per-level',
    '12',
    '--max-certification-attempts-per-deletion-branch',
    '2',
    '--max-insertion-rounds-per-checkpoint',
    '12',
    '--max-insertion-batch-size',
    '1',
    '--no-append-schur-condition-gate',
    '--dynamics-policy',
    'avqds',
    '--avqds-l2-cut',
    '1.0e-3',
    '--parameterization-mode',
    'logical_shared',
    '--record-statevector',
    '--require-complete-candidate-pool',
    '--progress-log-every',
    '1',
    '--code-revision',
    'REV',
    '--output-json',
    'OUT.json',
)

FROZEN_NB4_SCREEN_EXCHANGE_ARGV: tuple[str, ...] = (
    '--siam-seed-json',
    'SEED.json',
    '--times',
    '0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35000000000000003,0.36,0.37,0.38,0.39,0.4,0.41000000000000003,0.42,0.43,0.44,0.45,0.46,0.47000000000000003,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.5700000000000001,0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.6900000000000001,0.7000000000000001,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8,0.81,0.8200000000000001,0.8300000000000001,0.84,0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.9400000000000001,0.9500000000000001,0.96,0.97,0.98,0.99,1.0',
    '--integrator',
    'euler',
    '--ridge-lambda',
    '1e-06',
    '--solve-damping',
    '0.0',
    '--pinv-rcond',
    '1e-10',
    '--solve-repair',
    '--solve-repair-profile',
    'minimal',
    '--solve-repair-state-motion-l2-step-max',
    '5.0e-5',
    '--solve-repair-kink-eta-max',
    '5.0e-3',
    '--solve-repair-max-local-subdivisions',
    '11',
    '--fail-on-unsupported-steps',
    '--max-joint-patch-evaluations',
    '50000',
    '--max-certification-attempts-per-level',
    '12',
    '--max-certification-attempts-per-deletion-branch',
    '2',
    '--max-insertion-rounds-per-checkpoint',
    '12',
    '--max-insertion-batch-size',
    '1',
    '--no-append-schur-condition-gate',
    '--insertion-gate-mode',
    'mclachlan_l2',
    '--insertion-l2-cut',
    '0.001',
    '--avqds-l2-cut',
    '0.001',
    '--dynamics-policy',
    'exchange',
    '--debt-policy',
    'drift_ranked',
    '--prune-history-lambda',
    '0.0',
    '--prune-target-policy',
    'all_active',
    '--prune-ray-distance-tol',
    '2.0e-3',
    '--parameterization-mode',
    'logical_shared',
    '--append-occurrence-policy',
    'layer_reuse',
    '--record-statevector',
    '--require-complete-candidate-pool',
    '--progress-log-every',
    '1',
    '--code-revision',
    'REV',
    '--output-json',
    'OUT.json',
)

FROZEN_NB4_SCREEN_AVQDS_ARGV: tuple[str, ...] = (
    '--siam-seed-json',
    'SEED.json',
    '--times',
    '0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35000000000000003,0.36,0.37,0.38,0.39,0.4,0.41000000000000003,0.42,0.43,0.44,0.45,0.46,0.47000000000000003,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.5700000000000001,0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.6900000000000001,0.7000000000000001,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8,0.81,0.8200000000000001,0.8300000000000001,0.84,0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.9400000000000001,0.9500000000000001,0.96,0.97,0.98,0.99,1.0',
    '--integrator',
    'euler',
    '--ridge-lambda',
    '1e-06',
    '--solve-damping',
    '0.0',
    '--pinv-rcond',
    '1e-10',
    '--solve-repair',
    '--solve-repair-profile',
    'minimal',
    '--solve-repair-state-motion-l2-step-max',
    '5.0e-5',
    '--solve-repair-kink-eta-max',
    '5.0e-3',
    '--solve-repair-max-local-subdivisions',
    '11',
    '--fail-on-unsupported-steps',
    '--max-joint-patch-evaluations',
    '50000',
    '--max-certification-attempts-per-level',
    '12',
    '--max-certification-attempts-per-deletion-branch',
    '2',
    '--max-insertion-rounds-per-checkpoint',
    '12',
    '--max-insertion-batch-size',
    '1',
    '--no-append-schur-condition-gate',
    '--insertion-l2-cut',
    '0.001',
    '--avqds-l2-cut',
    '0.001',
    '--dynamics-policy',
    'avqds',
    '--parameterization-mode',
    'logical_shared',
    '--append-occurrence-policy',
    'layer_reuse',
    '--record-statevector',
    '--require-complete-candidate-pool',
    '--progress-log-every',
    '1',
    '--code-revision',
    'REV',
    '--output-json',
    'OUT.json',
)

FROZEN_NB4_CONTINUATION_EXCHANGE_ARGV: tuple[str, ...] = (
    '--resume-from-run-json',
    'PRIOR.json',
    '--siam-seed-json',
    'SEED.json',
    '--times',
    '1.0,1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09,1.1,1.11,1.12,1.1300000000000001,1.1400000000000001,1.1500000000000001,1.16,1.17,1.18,1.19,1.2,1.21,1.22,1.23,1.24,1.25,1.26,1.27,1.28,1.29,1.3,1.31,1.32,1.33,1.34,1.35,1.36,1.37,1.3800000000000001,1.3900000000000001,1.4000000000000001,1.41,1.42,1.43,1.44,1.45,1.46,1.47,1.48,1.49,1.5,1.51,1.52,1.53,1.54,1.55,1.56,1.57,1.58,1.59,1.6,1.61,1.62,1.6300000000000001,1.6400000000000001,1.6500000000000001,1.6600000000000001,1.67,1.68,1.69,1.7,1.71,1.72,1.73,1.74,1.75,1.76,1.77,1.78,1.79,1.8,1.81,1.82,1.83,1.84,1.85,1.86,1.87,1.8800000000000001,1.8900000000000001,1.9000000000000001,1.9100000000000001,1.92,1.93,1.94,1.95,1.96,1.97,1.98,1.99,2.0,2.0100000000000002,2.02,2.0300000000000002,2.04,2.05,2.06,2.07,2.08,2.09,2.1,2.11,2.12,2.13,2.14,2.15,2.16,2.17,2.18,2.19,2.2,2.21,2.22,2.23,2.24,2.25,2.2600000000000002,2.27,2.2800000000000002,2.29,2.3000000000000003,2.31,2.32,2.33,2.34,2.35,2.36,2.37,2.38,2.39,2.4,2.41,2.42,2.43,2.44,2.45,2.46,2.47,2.48,2.49,2.5,2.5100000000000002,2.52,2.5300000000000002,2.54,2.5500000000000003,2.56,2.57,2.58,2.59,2.6,2.61,2.62,2.63,2.64,2.65,2.66,2.67,2.68,2.69,2.7,2.71,2.72,2.73,2.74,2.75,2.7600000000000002,2.77,2.7800000000000002,2.79,2.8000000000000003,2.81,2.82,2.83,2.84,2.85,2.86,2.87,2.88,2.89,2.9,2.91,2.92,2.93,2.94,2.95,2.96,2.97,2.98,2.99,3.0,3.0100000000000002,3.02,3.0300000000000002,3.04,3.0500000000000003,3.06,3.0700000000000003,3.08,3.09,3.1,3.11,3.12,3.13,3.14,3.15,3.16,3.17,3.18,3.19,3.2,3.21,3.22,3.23,3.24,3.25,3.2600000000000002,3.27,3.2800000000000002,3.29,3.3000000000000003,3.31,3.3200000000000003,3.33,3.34,3.35,3.36,3.37,3.38,3.39,3.4,3.41,3.42,3.43,3.44,3.45,3.46,3.47,3.48,3.49,3.5,3.5100000000000002,3.52,3.5300000000000002,3.54,3.5500000000000003,3.56,3.5700000000000003,3.58,3.59,3.6,3.61,3.62,3.63,3.64,3.65,3.66,3.67,3.68,3.69,3.7,3.71,3.72,3.73,3.74,3.75,3.7600000000000002,3.77,3.7800000000000002,3.79,3.8000000000000003,3.81,3.8200000000000003,3.83,3.84,3.85,3.86,3.87,3.88,3.89,3.9,3.91,3.92,3.93,3.94,3.95,3.96,3.97,3.98,3.99,4.0,4.01,4.0200000000000005,4.03,4.04,4.05,4.0600000000000005,4.07,4.08,4.09,4.1,4.11,4.12,4.13,4.14,4.15,4.16,4.17,4.18,4.19,4.2,4.21,4.22,4.23,4.24,4.25,4.26,4.2700000000000005,4.28,4.29,4.3,4.3100000000000005,4.32,4.33,4.34,4.3500000000000005,4.36,4.37,4.38,4.39,4.4,4.41,4.42,4.43,4.44,4.45,4.46,4.47,4.48,4.49,4.5,4.51,4.5200000000000005,4.53,4.54,4.55,4.5600000000000005,4.57,4.58,4.59,4.6000000000000005,4.61,4.62,4.63,4.64,4.65,4.66,4.67,4.68,4.69,4.7,4.71,4.72,4.73,4.74,4.75,4.76,4.7700000000000005,4.78,4.79,4.8,4.8100000000000005,4.82,4.83,4.84,4.8500000000000005,4.86,4.87,4.88,4.89,4.9,4.91,4.92,4.93,4.94,4.95,4.96,4.97,4.98,4.99,5.0',
    '--integrator',
    'euler',
    '--ridge-lambda',
    '1e-06',
    '--solve-damping',
    '0.0',
    '--pinv-rcond',
    '1e-10',
    '--solve-repair',
    '--solve-repair-profile',
    'minimal',
    '--solve-repair-state-motion-l2-step-max',
    '5.0e-5',
    '--solve-repair-kink-eta-max',
    '5.0e-3',
    '--solve-repair-max-local-subdivisions',
    '11',
    '--fail-on-unsupported-steps',
    '--max-joint-patch-evaluations',
    '50000',
    '--max-certification-attempts-per-level',
    '12',
    '--max-certification-attempts-per-deletion-branch',
    '2',
    '--max-insertion-rounds-per-checkpoint',
    '12',
    '--max-insertion-batch-size',
    '1',
    '--no-append-schur-condition-gate',
    '--insertion-gate-mode',
    'mclachlan_l2',
    '--insertion-l2-cut',
    '0.001',
    '--avqds-l2-cut',
    '0.001',
    '--dynamics-policy',
    'exchange',
    '--debt-policy',
    'drift_ranked',
    '--prune-history-lambda',
    '0.0',
    '--prune-target-policy',
    'all_active',
    '--prune-ray-distance-tol',
    '2.0e-3',
    '--parameterization-mode',
    'logical_shared',
    '--append-occurrence-policy',
    'layer_reuse',
    '--record-statevector',
    '--require-complete-candidate-pool',
    '--progress-log-every',
    '1',
    '--code-revision',
    'REV',
    '--output-json',
    'OUT.json',
)

FROZEN_NB4_CONTINUATION_AVQDS_ARGV: tuple[str, ...] = (
    '--resume-from-run-json',
    'PRIOR.json',
    '--siam-seed-json',
    'SEED.json',
    '--times',
    '1.0,1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09,1.1,1.11,1.12,1.1300000000000001,1.1400000000000001,1.1500000000000001,1.16,1.17,1.18,1.19,1.2,1.21,1.22,1.23,1.24,1.25,1.26,1.27,1.28,1.29,1.3,1.31,1.32,1.33,1.34,1.35,1.36,1.37,1.3800000000000001,1.3900000000000001,1.4000000000000001,1.41,1.42,1.43,1.44,1.45,1.46,1.47,1.48,1.49,1.5,1.51,1.52,1.53,1.54,1.55,1.56,1.57,1.58,1.59,1.6,1.61,1.62,1.6300000000000001,1.6400000000000001,1.6500000000000001,1.6600000000000001,1.67,1.68,1.69,1.7,1.71,1.72,1.73,1.74,1.75,1.76,1.77,1.78,1.79,1.8,1.81,1.82,1.83,1.84,1.85,1.86,1.87,1.8800000000000001,1.8900000000000001,1.9000000000000001,1.9100000000000001,1.92,1.93,1.94,1.95,1.96,1.97,1.98,1.99,2.0,2.0100000000000002,2.02,2.0300000000000002,2.04,2.05,2.06,2.07,2.08,2.09,2.1,2.11,2.12,2.13,2.14,2.15,2.16,2.17,2.18,2.19,2.2,2.21,2.22,2.23,2.24,2.25,2.2600000000000002,2.27,2.2800000000000002,2.29,2.3000000000000003,2.31,2.32,2.33,2.34,2.35,2.36,2.37,2.38,2.39,2.4,2.41,2.42,2.43,2.44,2.45,2.46,2.47,2.48,2.49,2.5,2.5100000000000002,2.52,2.5300000000000002,2.54,2.5500000000000003,2.56,2.57,2.58,2.59,2.6,2.61,2.62,2.63,2.64,2.65,2.66,2.67,2.68,2.69,2.7,2.71,2.72,2.73,2.74,2.75,2.7600000000000002,2.77,2.7800000000000002,2.79,2.8000000000000003,2.81,2.82,2.83,2.84,2.85,2.86,2.87,2.88,2.89,2.9,2.91,2.92,2.93,2.94,2.95,2.96,2.97,2.98,2.99,3.0,3.0100000000000002,3.02,3.0300000000000002,3.04,3.0500000000000003,3.06,3.0700000000000003,3.08,3.09,3.1,3.11,3.12,3.13,3.14,3.15,3.16,3.17,3.18,3.19,3.2,3.21,3.22,3.23,3.24,3.25,3.2600000000000002,3.27,3.2800000000000002,3.29,3.3000000000000003,3.31,3.3200000000000003,3.33,3.34,3.35,3.36,3.37,3.38,3.39,3.4,3.41,3.42,3.43,3.44,3.45,3.46,3.47,3.48,3.49,3.5,3.5100000000000002,3.52,3.5300000000000002,3.54,3.5500000000000003,3.56,3.5700000000000003,3.58,3.59,3.6,3.61,3.62,3.63,3.64,3.65,3.66,3.67,3.68,3.69,3.7,3.71,3.72,3.73,3.74,3.75,3.7600000000000002,3.77,3.7800000000000002,3.79,3.8000000000000003,3.81,3.8200000000000003,3.83,3.84,3.85,3.86,3.87,3.88,3.89,3.9,3.91,3.92,3.93,3.94,3.95,3.96,3.97,3.98,3.99,4.0,4.01,4.0200000000000005,4.03,4.04,4.05,4.0600000000000005,4.07,4.08,4.09,4.1,4.11,4.12,4.13,4.14,4.15,4.16,4.17,4.18,4.19,4.2,4.21,4.22,4.23,4.24,4.25,4.26,4.2700000000000005,4.28,4.29,4.3,4.3100000000000005,4.32,4.33,4.34,4.3500000000000005,4.36,4.37,4.38,4.39,4.4,4.41,4.42,4.43,4.44,4.45,4.46,4.47,4.48,4.49,4.5,4.51,4.5200000000000005,4.53,4.54,4.55,4.5600000000000005,4.57,4.58,4.59,4.6000000000000005,4.61,4.62,4.63,4.64,4.65,4.66,4.67,4.68,4.69,4.7,4.71,4.72,4.73,4.74,4.75,4.76,4.7700000000000005,4.78,4.79,4.8,4.8100000000000005,4.82,4.83,4.84,4.8500000000000005,4.86,4.87,4.88,4.89,4.9,4.91,4.92,4.93,4.94,4.95,4.96,4.97,4.98,4.99,5.0',
    '--integrator',
    'euler',
    '--ridge-lambda',
    '1e-06',
    '--solve-damping',
    '0.0',
    '--pinv-rcond',
    '1e-10',
    '--solve-repair',
    '--solve-repair-profile',
    'minimal',
    '--solve-repair-state-motion-l2-step-max',
    '5.0e-5',
    '--solve-repair-kink-eta-max',
    '5.0e-3',
    '--solve-repair-max-local-subdivisions',
    '11',
    '--fail-on-unsupported-steps',
    '--max-joint-patch-evaluations',
    '50000',
    '--max-certification-attempts-per-level',
    '12',
    '--max-certification-attempts-per-deletion-branch',
    '2',
    '--max-insertion-rounds-per-checkpoint',
    '12',
    '--max-insertion-batch-size',
    '1',
    '--no-append-schur-condition-gate',
    '--insertion-l2-cut',
    '0.001',
    '--avqds-l2-cut',
    '0.001',
    '--dynamics-policy',
    'avqds',
    '--parameterization-mode',
    'logical_shared',
    '--append-occurrence-policy',
    'layer_reuse',
    '--record-statevector',
    '--require-complete-candidate-pool',
    '--progress-log-every',
    '1',
    '--code-revision',
    'REV',
    '--output-json',
    'OUT.json',
)


def _nb2(arm: str) -> tuple[str, ...]:
    return build_siam_nb2_proof_run(
        seed_path="SEED.json",
        arm=arm,
        output_json="OUT.json",
        code_revision="REV",
    ).argv()


def _nb4_screen(arm: str) -> tuple[str, ...]:
    return build_siam_nb4_screen_run(
        seed_path="SEED.json",
        arm=arm,
        output_json="OUT.json",
        code_revision="REV",
    ).argv()


def _nb4_continuation(arm: str) -> tuple[str, ...]:
    return build_siam_nb4_continuation_run(
        seed_path="SEED.json",
        arm=arm,
        resume_from_run_json="PRIOR.json",
        activation_cut=1.0e-3,
        output_json="OUT.json",
        code_revision="REV",
    ).argv()


@pytest.mark.parametrize(
    "actual, frozen",
    [
        (_nb2("exchange"), FROZEN_NB2_EXCHANGE_ARGV),
        (_nb2("avqds"), FROZEN_NB2_AVQDS_ARGV),
        (_nb4_screen("exchange"), FROZEN_NB4_SCREEN_EXCHANGE_ARGV),
        (_nb4_screen("avqds"), FROZEN_NB4_SCREEN_AVQDS_ARGV),
        (_nb4_continuation("exchange"), FROZEN_NB4_CONTINUATION_EXCHANGE_ARGV),
        (_nb4_continuation("avqds"), FROZEN_NB4_CONTINUATION_AVQDS_ARGV),
    ],
    ids=[
        "nb2_exchange", "nb2_avqds",
        "nb4_screen_exchange", "nb4_screen_avqds",
        "nb4_continuation_exchange", "nb4_continuation_avqds",
    ],
)
def test_registered_siam_argv_is_unchanged(
    actual: tuple[str, ...], frozen: tuple[str, ...]
) -> None:
    """The historical exchange and avqds SIAM commands are byte-identical."""

    assert tuple(actual) == frozen


def test_arm_dynamics_policy_is_the_structural_policy_not_the_arm_id() -> None:
    assert EXCHANGE.dynamics_policy == "exchange"
    assert EXCHANGE_Q_COMPILED.dynamics_policy == "exchange"
    assert EXCHANGE_GAMMA_COMPILED.dynamics_policy == "exchange"
    assert EXCHANGE_LEGACY_PROXY_FIXED.dynamics_policy == "exchange"
    assert AVQDS.dynamics_policy == "avqds"
    # Every arm's policy is one the runner actually admits.
    for arm in ARMS.values():
        assert arm.dynamics_policy in {"exchange", "avqds"}


def test_compiled_q_is_a_registered_siam_arm() -> None:
    assert SIAM_NB2_PROOF_ARMS["exchange_q_compiled"] is EXCHANGE_Q_COMPILED
    assert SIAM_NB4_ARMS["exchange_q_compiled"] is EXCHANGE_Q_COMPILED


@pytest.mark.parametrize(
    "argv_of",
    [_nb2, _nb4_screen, _nb4_continuation],
    ids=["nb2_proof", "nb4_screen", "nb4_continuation"],
)
def test_compiled_q_siam_argv_keeps_the_route_and_the_gate(argv_of) -> None:
    argv = list(argv_of("exchange_q_compiled"))
    parsed = _parser().parse_args(argv)

    assert parsed.dynamics_policy == "exchange"
    assert parsed.exchange_route_bundle == "paper_ii_exchange_q_compiled_v1"
    assert parsed.insertion_gate_mode == "mclachlan_l2"
    assert parsed.debt_policy == "drift_ranked"
    # --dynamics-policy is passed exactly once, never as the arm id.
    assert argv.count("--dynamics-policy") == 1
    assert "exchange_q_compiled" not in argv


@pytest.mark.parametrize(
    "argv_of",
    [_nb2, _nb4_screen, _nb4_continuation],
    ids=["nb2_proof", "nb4_screen", "nb4_continuation"],
)
def test_avqds_siam_argv_carries_no_exchange_route(argv_of) -> None:
    argv = list(argv_of("avqds"))
    parsed = _parser().parse_args(argv)

    assert parsed.dynamics_policy == "avqds"
    assert "--exchange-route-bundle" not in argv
    assert "--insertion-gate-mode" not in argv


def test_siam_run_ids_name_the_route() -> None:
    nb2 = build_siam_nb2_proof_run(
        seed_path="SEED.json", arm="exchange_q_compiled",
        output_json="OUT.json", code_revision="REV",
    )
    assert nb2.run_id == "siam_nb2_t1_26_exchange_q_compiled"
    nb4 = build_siam_nb4_screen_run(
        seed_path="SEED.json", arm="exchange_q_compiled",
        output_json="OUT.json", code_revision="REV",
    )
    assert nb4.run_id.startswith("siam_nb4_")
    assert "_exchange_q_compiled_cut" in nb4.run_id
