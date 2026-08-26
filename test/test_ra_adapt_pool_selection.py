"""Optional parent-pool selection on the RA route.

The adapter's ``pool_key`` chooses the parent supply. ``None`` keeps the
canonical ``full_meta`` identity bit-for-bit (the refactor parity gate pins
that). An explicit key is validated fail-closed against the problem's
admissible pools, flows into the pool receipts and the route contract, and
therefore changes the protocol sha256: a different pool is a different
protocol identity, never a silent settings drift. Named applications stay
source-locked to their pools. The canonical Paper-I summary attaches only to
canonical pool identities; an ablation identity completes without one.
"""

from __future__ import annotations

import math

import pytest

from pipelines.static_adapt.ra_adapt import run_ra_adapt
from pipelines.static_adapt.ra_adapt import bundles as bundle_module
from pipelines.static_adapt.ra_adapt.adapters import MacroCandidateAdapter
from pipelines.static_adapt.ra_adapt.contracts import (
    CANDIDATE_REPRESENTATION_MACRO,
)
from pipelines.static_adapt.sr_snake.contracts import AppendOnlyInsertion

from test_ra_adapt_facade import _hh_problem, _validated_protocol


def _macro_protocol(problem, *, rounds: int, pool_key: str | None):
    return _validated_protocol(
        problem,
        rounds=rounds,
        adapter=MacroCandidateAdapter(pool_key=pool_key),
        insertion=AppendOnlyInsertion(),
        route_id=bundle_module.ROUTE_RA_MACRO_APPEND_ONLY,
        candidate_representation=CANDIDATE_REPRESENTATION_MACRO,
    )


def test_inadmissible_pool_fails_closed() -> None:
    problem = _hh_problem()
    with pytest.raises(ValueError, match="not admissible"):
        MacroCandidateAdapter(pool_key="no_such_pool").parent_inventory(
            problem
        )


def test_explicit_full_meta_matches_the_default_pool_identity() -> None:
    problem = _hh_problem()
    default = _macro_protocol(problem, rounds=2, pool_key=None)
    explicit = _macro_protocol(problem, rounds=2, pool_key="full_meta")
    # An explicit "full_meta" resolves the identical parent and executable
    # inventories; pool identity is carried by the digested receipts.
    assert (
        explicit.parent_inventory.ordered_pool_sha256
        == default.parent_inventory.ordered_pool_sha256
    )
    assert (
        explicit.executable_pool.ordered_pool_sha256
        == default.executable_pool.ordered_pool_sha256
    )


def test_pool_choice_is_a_distinct_protocol_identity() -> None:
    problem = _hh_problem()
    default = _macro_protocol(problem, rounds=2, pool_key=None)
    pareto = _macro_protocol(problem, rounds=2, pool_key="pareto_lean_l2")
    assert default.sha256 != pareto.sha256
    assert (
        pareto.executable_pool.ordered_pool_sha256
        != default.executable_pool.ordered_pool_sha256
    )
    assert pareto.request.adapter.pool_key == "pareto_lean_l2"


def test_non_default_pool_runs_and_respects_physics() -> None:
    problem = _hh_problem()
    result = run_ra_adapt(
        problem, _macro_protocol(problem, rounds=2, pool_key="pareto_lean_l2")
    )
    energies = [
        float(row.energy) for row in result.run.accepted_trajectory
    ]
    assert energies, "bounded pareto_lean_l2 run recorded no accepted state"
    exact = float(problem.exact_target.resolve_energy(ai_log=None))
    assert all(math.isfinite(e) and e >= exact - 1.0e-9 for e in energies)
    assert all(
        energies[i] <= energies[i - 1] + 1.0e-9
        for i in range(1, len(energies))
    )
    # The executed pool is the requested one, and the canonical Paper-I
    # summary does not stamp an ablation identity.
    assert str(result.run.route.execution.pool) == "pareto_lean_l2"
    assert result.run.paper_i_summary is None


def test_named_applications_reject_pool_overrides() -> None:
    problem = _hh_problem()
    from pipelines.static_adapt.ra_adapt.pools import _parent_pool_spec

    with pytest.raises(ValueError, match="source-locked"):
        _parent_pool_spec(
            problem,
            pool_key="pareto_lean_l2",
            paper_i_l3_page12_application=True,
        )
