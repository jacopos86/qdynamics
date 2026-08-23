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


def test_canonical_numerics_are_not_arm_dependent() -> None:
    """A comparison may vary the arm and nothing else."""

    argvs = {}
    for arm in ARMS:
        run = build_run(
            seed_path="seed.json", arm=arm, gate=MCLACHLAN_L2_GATE.gate_id,
            drive="fastweak", horizon="t2", output_json="out/run.json",
        )
        parsed = _parser().parse_args(list(run.argv()))
        argvs[arm] = (
            parsed.integrator,
            parsed.solve_repair_max_local_subdivisions,
            parsed.solve_repair_state_motion_l2_step_max,
            parsed.max_structural_pool_size,
            parsed.t_final,
            parsed.num_times,
            parsed.drive_A,
            parsed.drive_omega,
        )
    assert len(set(argvs.values())) == 1, argvs


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
