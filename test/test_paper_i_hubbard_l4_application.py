from __future__ import annotations

from pipelines.static_adapt.ra_adapt.append import AppendAdaptRequest
from pipelines.static_adapt.ra_adapt.paper_i_hubbard_l4 import (
    PAPER_I_HUBBARD_L4_EXACT_ENERGY,
    PaperIHubbardL4MacroGradientPhase0CandidateAdapter,
    build_paper_i_hubbard_l4_append_request,
    build_paper_i_hubbard_l4_problem,
    build_paper_i_hubbard_l4_request,
    materialize_paper_i_hubbard_l4_ra_protocol,
    paper_i_hubbard_l4_application_source_contract,
    require_paper_i_hubbard_l4_problem,
)
from pipelines.static_adapt.sr_snake.contracts import (
    BeamOff,
    CombinatorialBatchAdmission,
    ForkLocalBeam,
    MetricPruning,
    PruningOff,
    SingletonAdmission,
)


def test_l4_problem_and_macro_pool_are_exactly_named() -> None:
    problem = build_paper_i_hubbard_l4_problem()
    require_paper_i_hubbard_l4_problem(problem)
    assert problem.family_key == "hubbard"
    assert problem.request.num_sites == 4
    assert problem.sector.num_particles == (2, 2)
    assert abs(problem.exact_target.resolve_energy() - PAPER_I_HUBBARD_L4_EXACT_ENERGY) < 5e-13

    adapter = PaperIHubbardL4MacroGradientPhase0CandidateAdapter()
    parent = adapter.parent_inventory(problem)
    executable = adapter.executable_pool(problem)
    assert parent.receipt.count > 0
    assert executable.receipt.count > 0
    assert executable.receipt.count <= parent.receipt.count
    assert executable.receipt.source_parent_ordered_labels_sha256 == parent.receipt.ordered_labels_sha256


def test_l4_five_arm_requests_change_only_the_typed_extension() -> None:
    baseline = build_paper_i_hubbard_l4_request("baseline", maximum_controller_rounds=3)
    prune = build_paper_i_hubbard_l4_request("prune", maximum_controller_rounds=3)
    batch = build_paper_i_hubbard_l4_request("batch", maximum_controller_rounds=3)
    beam = build_paper_i_hubbard_l4_request("beam", maximum_controller_rounds=3)
    append = build_paper_i_hubbard_l4_append_request(maximum_controller_rounds=3)

    assert isinstance(baseline.method.admission, SingletonAdmission)
    assert isinstance(baseline.method.pruning, PruningOff)
    assert isinstance(baseline.method.beam, BeamOff)

    assert isinstance(prune.method.admission, SingletonAdmission)
    assert isinstance(prune.method.pruning, MetricPruning)
    assert prune.method.pruning.small_theta_abs == 1.0e-10
    assert isinstance(prune.method.beam, BeamOff)

    assert isinstance(batch.method.admission, CombinatorialBatchAdmission)
    assert isinstance(batch.method.pruning, PruningOff)
    assert isinstance(batch.method.beam, BeamOff)

    assert isinstance(beam.method.admission, SingletonAdmission)
    assert isinstance(beam.method.pruning, PruningOff)
    assert isinstance(beam.method.beam, ForkLocalBeam)

    assert all(
        isinstance(request.adapter, PaperIHubbardL4MacroGradientPhase0CandidateAdapter)
        for request in (baseline, prune, batch, beam)
    )
    assert isinstance(append, AppendAdaptRequest)
    assert isinstance(append.adapter, PaperIHubbardL4MacroGradientPhase0CandidateAdapter)


def test_l4_prune_keeps_the_old_theta_nomination_gate() -> None:
    problem = build_paper_i_hubbard_l4_problem()
    baseline = materialize_paper_i_hubbard_l4_ra_protocol(
        problem,
        build_paper_i_hubbard_l4_request(
            "baseline",
            maximum_controller_rounds=3,
        ),
    )
    prune = materialize_paper_i_hubbard_l4_ra_protocol(
        problem,
        build_paper_i_hubbard_l4_request(
            "prune",
            maximum_controller_rounds=3,
        ),
    )

    assert "prune_small_theta_nomination_screen" not in (
        baseline.route_contract["semantic_invariants"]
    )
    assert (
        prune.route_contract["execution_settings"][
            "phase1_prune_small_theta_abs"
        ]
        == 1.0e-10
    )
    assert (
        prune.route_contract["semantic_invariants"][
            "prune_small_theta_nomination_screen"
        ]
        == 1.0e-10
    )


def test_l4_source_contract_binds_matched_tight_powell_settings() -> None:
    source = paper_i_hubbard_l4_application_source_contract(
        build_paper_i_hubbard_l4_problem()
    )
    settings = source["scientific_settings"]
    assert settings["optimizer"] == "powell"
    assert settings["optimizer_maxiter"] == 200
    assert settings["optimizer_xtol"] == 1.0e-5
    assert settings["optimizer_ftol"] == 1.0e-12
