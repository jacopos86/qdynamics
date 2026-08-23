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
    REGIMES,
    SeedNotBuiltError,
    available_regimes,
    DRIVES,
    GATES,
    HORIZONS,
    MCLACHLAN_L2_GATE,
    build_run,
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


def test_pool_cap_does_not_bind_the_deduplicated_pool() -> None:
    """A cap below the pool size silently discards usable words."""

    run = build_run(
        seed_path="seed.json", arm="exchange", gate=MCLACHLAN_L2_GATE.gate_id,
        drive="fastweak", horizon="t2", output_json="out/run.json",
    )
    parsed = _parser().parse_args(list(run.argv()))
    assert int(parsed.max_structural_pool_size) >= 125


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
        DriveSpec,
        HorizonSpec,
        SeedSpec,
        append_only_arm,
        avqds_arm,
        exchange_arm,
    )

    spec = CampaignSpec(
        campaign_id="parity",
        seeds=(SeedSpec("s", "seed.json", "hh", 1, "weak"),),
        drives=(DriveSpec("fastweak", True, 0.6, 3.0),),
        horizons=(HorizonSpec("t2", 2.0, 51),),
        arms=(exchange_arm(2.0e-3), append_only_arm(), avqds_arm(1.0e-3)),
    )
    parser = _parser()
    by_arm = {}
    for cell in spec.cells():
        parsed = parser.parse_args(list(cell.runner_argv()))
        by_arm[cell.arm.arm_id] = parsed

    for arm_id, registry_arm in (
        ("exchange_tau0.002", "exchange"),
        ("append_only", "append_only"),
        ("avqds_cut0.001", "avqds"),
    ):
        run = build_run(
            seed_path="seed.json", arm=registry_arm,
            gate=MCLACHLAN_L2_GATE.gate_id, drive="fastweak", horizon="t2",
            output_json="out/run.json",
        )
        registry = _parser().parse_args(list(run.argv()))
        campaign = by_arm[arm_id]
        for field in (
            "integrator",
            "solve_repair_max_local_subdivisions",
            "solve_repair_state_motion_l2_step_max",
            "solve_repair_kink_eta_max",
            "max_structural_pool_size",
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


def test_no_campaign_arm_reintroduces_the_pool_cap_of_eight() -> None:
    from pipelines.time_dynamics.campaign import PRODUCTION_STRUCTURE

    flags = list(PRODUCTION_STRUCTURE)
    assert int(flags[flags.index("--max-structural-pool-size") + 1]) >= 125


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
        parsed.max_structural_pool_size,
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
        base_parsed.max_structural_pool_size,
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
    assert parsed.ridge_lambda == pytest.approx(1.0e-7)
    assert parsed.avqds_delta_theta_max == pytest.approx(5.0e-3)
    assert parsed.dynamics_policy == "avqds"


def test_step_control_is_separable_from_the_structural_rule() -> None:
    """The 2x2 that isolates step control from structure must be expressible."""

    seen = {}
    for arm in ("exchange", "avqds"):
        for control in ("state_motion_1e-2", "delta_theta_5e-3"):
            run = build_run(
                regime="hh_snake_nph1", arm=arm, gate=MCLACHLAN_L2_GATE.gate_id,
                drive="strongfast", horizon="t10", step_control=control,
                output_json="out/run.json",
            )
            parsed = _parser().parse_args(list(run.argv()))
            seen[(arm, control)] = (
                parsed.solve_repair, parsed.avqds_delta_theta_max
            )
            # The inner numerical method is identical across all four cells.
            assert parsed.integrator == "euler"
            assert parsed.ridge_lambda == pytest.approx(1.0e-7)
    assert seen[("exchange", "delta_theta_5e-3")][1] == pytest.approx(5.0e-3)
    assert seen[("avqds", "state_motion_1e-2")][0] is True
    assert len({v for v in seen.values()}) == 2


def test_every_arm_shares_the_same_inner_numerics() -> None:
    """Damping, regularization, and integrator may never vary by arm."""

    values = set()
    for arm in ARMS:
        run = build_run(
            regime="hh_snake_nph1", arm=arm, gate=MCLACHLAN_L2_GATE.gate_id,
            drive="strongfast", horizon="t10", output_json="out/run.json",
        )
        parsed = _parser().parse_args(list(run.argv()))
        values.add((parsed.integrator, parsed.ridge_lambda,
                    parsed.solve_damping, parsed.pinv_rcond))
    assert len(values) == 1, values
