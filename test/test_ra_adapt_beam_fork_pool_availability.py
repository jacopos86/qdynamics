"""Beam sibling-diversity exclusions must not contract the candidate pool.

A fork-local beam child is forked with ``excluded_pool_indices`` so that the
siblings of one parent explore distinct admissions.  That exclusion constrains
one selection decision.  Because a winning child is promoted to the next
round's parent, an exclusion written into persistent pool availability deletes
the operator from the surviving lineage for the rest of the run.

These regressions pin the separation: ``available_indices`` stays at full
parent availability and is what gets snapshotted, checkpointed and rehydrated;
``selection_available_indices()`` carries the round-scoped exclusion and is what
the round's gradient surface, Phase-0 screen and candidate domain consume.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pipelines.contracts.problem import ProblemRequest
from pipelines.static_adapt.builders.problem_registry import (
    resolve_problem_context,
)
from pipelines.static_adapt.ra_adapt import (
    MacroGradientPhase0CandidateAdapter,
    RAAdaptRequest,
    run_ra_adapt,
)
from pipelines.static_adapt.ra_adapt import bundles as bundle_module
from pipelines.static_adapt.ra_adapt.contracts import (
    ACTIVE_GRADIENT_STATIONARY,
    CANDIDATE_REPRESENTATION_MACRO,
    RESOURCE_WEIGHTING_ALL_PHASE,
    _attach_validated_bundle_protocol_authority,
)
from pipelines.static_adapt.ra_adapt.engine import (
    RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID,
    build_resolved_ra_protocol,
)
from pipelines.static_adapt.sr_snake import (
    ForkLocalBeam,
    MetricPruning,
    PlateauCommutationInsertion,
    SRExecutionPolicy,
    SRMethodPolicy,
    SRStopPolicy,
)

BEAM_ROUNDS = 6


def _hh_problem() -> Any:
    return resolve_problem_context(
        ProblemRequest(
            problem_key="hh",
            num_sites=2,
            t=1.0,
            u=0.5,
            dv=0.0,
            omega0=1.0,
            g_ep=0.2,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
        )
    )


def _beam_metric_method() -> SRMethodPolicy:
    return SRMethodPolicy(
        insertion=PlateauCommutationInsertion(),
        pruning=MetricPruning(),
        beam=ForkLocalBeam(
            live_parent_branches=3,
            admission_children_per_parent=2,
            maximum_admission_children_per_round=6,
            s_alg_weight=0.005,
        ),
    )


def _validated_execution_protocol(*, rounds: int) -> tuple[Any, Any]:
    problem = _hh_problem()
    request = RAAdaptRequest(
        adapter=MacroGradientPhase0CandidateAdapter(),
        method=_beam_metric_method(),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=int(rounds))
        ),
    )
    cell = bundle_module.BundleCellSpec(
        cell_id="beam_fork_pool_availability_fixture",
        stage="validation",
        regime_id="fixture",
        nph=1,
        route_id="beam_fork_pool_availability_fixture",
        algorithm_id=(
            RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID
        ),
        selector_family="ra_adapt",
        candidate_representation=CANDIDATE_REPRESENTATION_MACRO,
        horizon=int(rounds),
        source_lock_id="fixture_lock",
    )
    authority_kwargs = {
        "cell": cell,
        "bundle_id": bundle_module.STATIONARY_BUNDLE_ID,
        "bundle_manifest_sha256": "8" * 64,
        "source_locks_sha256": "1" * 64,
        "source_lock_refs": {
            "source_locks_manifest_sha256": "1" * 64,
            "implementation_source_inventory_sha256": "2" * 64,
            "cell_source_lock_id": "fixture_lock",
            "cell_source_lock_sha256": "3" * 64,
            "visible_provenance_sha256": "4" * 64,
            "provenance_tracker_sha256": "5" * 64,
            "ed_cutoff_reference_sha256": "6" * 64,
            "resolver_script_sha256": "7" * 64,
        },
        "active_gradient_policy": ACTIVE_GRADIENT_STATIONARY,
        "resource_weighting_scope": RESOURCE_WEIGHTING_ALL_PHASE,
    }
    protocol = build_resolved_ra_protocol(
        problem,
        request,
        materialization_authority=(
            bundle_module._bundle_protocol_materialization_authority(
                **authority_kwargs
            )
        ),
    )
    protocol = _attach_validated_bundle_protocol_authority(
        protocol,
        bundle_module._bundle_protocol_materialization_authority(
            **authority_kwargs,
            protocol_sha256=protocol.sha256,
        ),
    )
    return problem, protocol


def test_fork_exclusion_is_round_scoped_and_not_persistent() -> None:
    """The exclusion filters this round's selection, not branch availability."""

    from pipelines.static_adapt.adapt_pipeline import (
        _DefaultNoPruneNumericalCursor,
    )

    field = _DefaultNoPruneNumericalCursor.__dataclass_fields__[
        "beam_fork_excluded_pool_indices"
    ]
    assert field.default == frozenset()

    cursor = object.__new__(_DefaultNoPruneNumericalCursor)
    cursor.available_indices = {0, 1, 2, 3}
    cursor.beam_fork_excluded_pool_indices = frozenset()
    assert cursor.selection_available_indices() == {0, 1, 2, 3}

    cursor.beam_fork_excluded_pool_indices = frozenset({1, 3})
    # The round's selection may not reach an excluded sibling admission ...
    assert cursor.selection_available_indices() == {0, 2}
    # ... but persistent availability, which is what gets snapshotted and
    # rehydrated on resume, is untouched.
    assert cursor.available_indices == {0, 1, 2, 3}


def test_portable_pool_state_retains_retired_selection_counts() -> None:
    """Retirement narrows availability without erasing selection history."""

    from pipelines.static_adapt.adapt_pipeline import (
        _default_no_prune_portable_pool_state,
    )

    pool = tuple(
        SimpleNamespace(label=f"candidate-{index}") for index in range(3)
    )
    registry = {
        f"candidate-{index}": {"generator_id": f"generator-{index}"}
        for index in range(3)
    }
    available, counts = _default_no_prune_portable_pool_state(
        pool=pool,
        available_indices={0, 2},
        selection_counts=np.asarray([1, 3, 2], dtype=np.int64),
        pool_generator_registry=registry,
    )

    assert available == (
        "generator-0::pool[0]",
        "generator-2::pool[2]",
    )
    assert counts == (
        ("generator-0::pool[0]", 1),
        ("generator-1::pool[1]", 3),
        ("generator-2::pool[2]", 2),
    )


def test_beam_portable_state_preserves_persistent_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sibling exclusion narrows the domain, not portable state."""

    from pipelines.static_adapt import adapt_pipeline

    session_type = adapt_pipeline._DefaultNoPruneNumericalSession
    original_snapshot = (
        adapt_pipeline._default_no_prune_transition_state_snapshot
    )
    original_fork = session_type.fork_beam_branch
    audited_exclusions: list[tuple[int, ...]] = []
    audited_post_transition_states: list[int] = []

    def audited_fork(
        session: Any,
        state: Any,
        *,
        branch_id: str,
        parent_branch_id: str | None,
        excluded_pool_indices: tuple[int, ...] = (),
    ) -> tuple[Any, Any]:
        child, child_input_state = original_fork(
            session,
            state,
            branch_id=branch_id,
            parent_branch_id=parent_branch_id,
            excluded_pool_indices=excluded_pool_indices,
        )
        if excluded_pool_indices:
            audited_exclusions.append(tuple(excluded_pool_indices))
            assert child_input_state.available_generator_ids == (
                state.available_generator_ids
            )
            assert child_input_state.selection_counts == state.selection_counts
        return child, child_input_state

    def audited_snapshot(**kwargs: Any) -> Any:
        snapshot = original_snapshot(**kwargs)
        persistent = kwargs["persistent_generator_state"]
        if int(snapshot.controller_round) > int(
            persistent.controller_round
        ):
            audited_post_transition_states.append(snapshot.controller_round)
            assert snapshot.available_generator_ids == (
                persistent.available_generator_ids
            )
            assert tuple(
                generator_id for generator_id, _count in snapshot.selection_counts
            ) == tuple(
                generator_id
                for generator_id, _count in persistent.selection_counts
            )
        return snapshot

    monkeypatch.setattr(
        session_type,
        "fork_beam_branch",
        audited_fork,
    )
    monkeypatch.setattr(
        adapt_pipeline,
        "_default_no_prune_transition_state_snapshot",
        audited_snapshot,
    )
    problem, protocol = _validated_execution_protocol(rounds=1)
    run_ra_adapt(problem, protocol)

    assert audited_exclusions
    assert audited_post_transition_states


def test_beam_fork_leaves_child_pool_availability_at_parent_width() -> None:
    """A forked child inherits full availability, exclusion held separately."""

    import inspect

    from pipelines.static_adapt.adapt_pipeline import (
        _DefaultNoPruneNumericalSession,
    )

    source = inspect.getsource(
        _DefaultNoPruneNumericalSession.fork_beam_branch
    )
    # The cloned cursor must receive full parent availability; the exclusion
    # must travel in the round-scoped field instead.
    assert "available_indices=available_indices" in source
    assert "beam_fork_excluded_pool_indices=excluded" in source
    assert "selectable_indices = available_indices.difference(excluded)" in (
        source
    )

    clear_source = inspect.getsource(
        _DefaultNoPruneNumericalSession.clear_beam_branch_context
    )
    assert (
        "self.cursor.beam_fork_excluded_pool_indices = frozenset()"
        in clear_source
    )


def test_accepted_state_snapshot_reports_persistent_availability() -> None:
    """The promoted/checkpointed snapshot must never carry the exclusion."""

    import inspect

    from pipelines.static_adapt import adapt_pipeline

    source = inspect.getsource(adapt_pipeline)
    marker = "return _default_no_prune_accepted_selection_snapshot("
    assert source.count(marker) == 1
    index = source.index(marker)
    body = source[index : index + 900]
    assert "available_indices=self.cursor.available_indices," in body
    assert "selection_available_indices" not in body


def test_beam_route_pool_width_is_constant_across_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: the nomination pool never contracts under beam + prune."""

    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    problem, protocol = _validated_execution_protocol(rounds=BEAM_ROUNDS)

    observed = run_ra_adapt(problem, protocol)

    receipts = observed.scientific_receipts["accepted_round_receipts"]
    assert len(receipts) == BEAM_ROUNDS

    widths = [
        int(
            receipt["ra_gradient_phase0_shortlist"]["input_candidate_count"]
        )
        for receipt in receipts
    ]
    populations = [
        {
            int(row["pool_index"])
            for row in receipt["ra_gradient_phase0_shortlist"]["ranking"]
        }
        for receipt in receipts
    ]
    union = set().union(*populations)

    # A promoted child may legitimately have screened one round against its
    # own sibling exclusion, so widths vary.  What must not happen is cumulative
    # loss: the variation is bounded by a single round's exclusion, and at
    # least one round still sees the whole population.
    assert max(widths) == len(union), (
        f"no round screened the full nomination population: {widths}"
    )
    assert max(widths) - min(widths) <= 1, (
        "nomination-pool width varies by more than one round's sibling "
        f"exclusion: {widths}"
    )

    # The defect signature is a terminal absence: once an index left the
    # population it never returned.  A round-scoped exclusion always recovers.
    for index in sorted(union):
        absent = [
            ordinal
            for ordinal, population in enumerate(populations)
            if index not in population
        ]
        if not absent or absent[0] == len(populations) - 1:
            continue
        recovered = any(
            index in population
            for population in populations[absent[0] + 1 :]
        )
        assert recovered, (
            f"pool index {index} left the nomination population at round "
            f"{absent[0] + 1} and never returned"
        )
