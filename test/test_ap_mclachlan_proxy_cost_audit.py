"""The singleton proxy identity saves work without changing search decisions."""

from copy import deepcopy
from itertools import product

import pytest

from pipelines.time_dynamics.ap_mclachlan import append_cost
from pipelines.time_dynamics.ap_mclachlan import exchange_integration as integration
from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import _PruneControllerRuntimeState
from pipelines.time_dynamics.ap_mclachlan.exchange_config import (
    EXCHANGE_ROUTE_BUNDLE_HISTORICAL,
    EXCHANGE_ROUTE_BUNDLE_LEGACY_PROXY_FIXED,
    EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_structural import enumerate_structural_candidates
from pipelines.time_dynamics.ap_mclachlan.geometry_eval import evaluate_mclachlan_geometry
from test.test_ap_mclachlan_exchange_integration import (
    HAM, POLICY, _active_prune_atoms, _config, _run, _state,
)


def _compiled_config(**overrides):
    values = dict(
        exchange_route_bundle=EXCHANGE_ROUTE_BUNDLE_Q_COMPILED,
        debt_policy="drift_ranked", insertion_gate_mode="mclachlan_l2",
        max_insertion_batch_size=2, max_joint_patch_evaluations=256,
        max_compiled_patches_per_checkpoint=256, compile_workers=1,
    )
    values.update(overrides)
    return _config(**values)


def _forbid_proxy_work(monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("singleton-normalized proxy work is mathematically redundant")

    for name in (
        "estimate_append_atom_set_cost", "estimate_prune_atom_set_cost",
        "_paper_i_proxy_denominator", "append_cost_telemetry_for_family",
    ):
        monkeypatch.setattr(integration, name, unexpected)
    monkeypatch.setattr(append_cost.Phase1CompileCostOracle, "estimate", unexpected)


def _with_original_callbacks(build_inputs):
    """Diagnostic reconstruction of the pre-optimization adapter callbacks."""

    def build(**kwargs):
        inputs = build_inputs(**kwargs)
        atoms = inputs["atoms_by_id"]
        active = {
            int(i): atom
            for atom in integration.active_support_atoms(kwargs["state"], kwargs["theta_runtime"])
            for i in atom.runtime_indices
        }
        settings = inputs["cost_settings"]

        def insertion_cost(atom_ids):
            selected = tuple(atoms[str(a)] for a in atom_ids)
            return integration._paper_i_proxy_denominator(
                integration.estimate_append_atom_set_cost(selected), settings=settings,
            )

        def deletion_cost(removed):
            selected = tuple(active[int(i)] for i in removed if i in active)
            if not selected:
                return 1.0
            return integration._paper_i_proxy_denominator(
                integration.estimate_prune_atom_set_cost(selected), settings=settings,
            )

        inputs["structural_kwargs"].update(
            insertion_cost=insertion_cost, deletion_cost=deletion_cost,
        )
        return inputs

    return build


def _scientific_payload(payload):
    result = deepcopy(payload)
    compiler = result["selector_route"].get("compile")
    if compiler is not None:
        compiler.pop("compile_seconds")
        compiler.pop("warm_seconds")
    return result


@pytest.mark.parametrize("policy,gain", tuple(product(
    ("greedy", "block_exchange", "bounded_exhaustive"),
    ("joint_batch", "singleton_factorized"),
)))
def test_compiled_modes_skip_singleton_proxy_work_and_preserve_decisions(monkeypatch, policy, gain):
    config = _compiled_config(
        search_policy=policy, gain_evaluation=gain,
        support_patch_scoring_workers=2 if policy == "bounded_exhaustive" else 1,
    )
    with monkeypatch.context() as guarded:
        _forbid_proxy_work(guarded)
        actual, payload = _run(config)
    assert payload["selector_route"]["compile"]["compile_calls"] > 0
    assert payload["selector_route"]["compile"]["backend"]["backend_name"] == "FakeMarrakesh"
    assert actual.committed.cost_marginal is not None
    assert actual.certification.certified

    with monkeypatch.context() as diagnostic:
        diagnostic.setattr(integration, "build_selector_inputs", _with_original_callbacks(integration.build_selector_inputs))
        expected, reference = _run(config)
    assert actual.committed == expected.committed
    assert actual.attempts == expected.attempts
    assert _scientific_payload(payload) == _scientific_payload(reference)


def _enumerate(config, build_inputs):
    state = _state()
    evaluation = evaluate_mclachlan_geometry(
        state=state, hamiltonian=HAM, theta_runtime=state.theta_runtime,
        time=0.0, include_tangent_matrix=True,
    )
    runtime = _PruneControllerRuntimeState()
    runtime.loss_history = {label: [(1, 0.2), (2, 0.4)] for label in state.runtime_coordinate_labels}
    inputs = build_inputs(
        state=state, evaluation=evaluation, support_config=config,
        runtime_state=runtime, theta_runtime=state.theta_runtime, time_index=3,
        deletable_atoms=_active_prune_atoms(
            state, theta_runtime=state.theta_runtime, support_config=config,
            runtime_state=runtime, time_index=3,
        ),
        insertions_enabled=True, debt_ranking=True, accuracy_debt=True,
    )
    try:
        return enumerate_structural_candidates(**inputs["structural_kwargs"], inverse_policy=POLICY)
    finally:
        if inputs["compiled_cost_runtime"] is not None:
            inputs["compiled_cost_runtime"].close()


@pytest.mark.parametrize("bundle", (EXCHANGE_ROUTE_BUNDLE_HISTORICAL, EXCHANGE_ROUTE_BUNDLE_Q_COMPILED))
def test_identity_preserves_full_frontiers_and_nonzero_utility_hooks(monkeypatch, bundle):
    config = _compiled_config(
        exchange_route_bundle=bundle, interaction_frontier_widths=(1, 2),
        exchange_insertion_frontier=1, prune_history_lambda=0.7,
        prune_condition_lambda_kappa_rel=0.2, prune_condition_lambda_kappa_dam=0.3,
        max_compiled_patches_per_checkpoint=None,
    )
    with monkeypatch.context() as guarded:
        _forbid_proxy_work(guarded)
        actual = _enumerate(config, integration.build_selector_inputs)
    expected = _enumerate(config, _with_original_callbacks(integration.build_selector_inputs))
    assert actual.frontiers_used == 2
    assert any(len(c.inserted_selection) == 2 for c in actual.candidates)
    assert actual.exchange_frontier is not None
    assert len(actual.exchange_frontier) == 1
    assert actual == expected


def test_raw_legacy_denominators_still_use_nonunit_proxy_cost(monkeypatch):
    seen = []
    original = integration._paper_i_proxy_denominator

    def record(*args, **kwargs):
        value = original(*args, **kwargs)
        seen.append(value)
        return value

    monkeypatch.setattr(integration, "_paper_i_proxy_denominator", record)
    _run(_config(cost_normalization_mode="raw_legacy_v1"))
    assert set(seen) == {1.025, 1.075, 1.125}


def test_population_repair_keeps_actual_population_proxy_cost(monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("population repair does not need singleton normalization")

    monkeypatch.setattr(integration, "_paper_i_proxy_denominator", unexpected)
    _, payload = _run(_compiled_config(
        exchange_route_bundle=EXCHANGE_ROUTE_BUNDLE_LEGACY_PROXY_FIXED,
        max_compiled_patches_per_checkpoint=None,
    ))
    level = payload["selector_route"]["levels"][0]
    assert level["insert_population_size"] > 1
    assert max(level["distinct_insert_denominators"]) > 1.0
