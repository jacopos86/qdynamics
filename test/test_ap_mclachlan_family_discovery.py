"""Family discovery must bound refused D without inventing evaluated patches."""
from types import SimpleNamespace

from pipelines.time_dynamics.ap_mclachlan.exchange_config import resolve_route_bundle
from pipelines.time_dynamics.ap_mclachlan.exchange_structural import (
    enumerate_structural_candidates, iter_structural_families,
)
from pipelines.time_dynamics.ap_mclachlan.patch_evaluation import CandidateCostInputs
from test.test_ap_mclachlan_exchange_structural import _setup


def test_refused_deletion_discovery_is_bounded_without_patch_or_compile_charges():
    visits, compilations = [], []

    def permission(removed):
        visits.append(removed)
        return SimpleNamespace(permitted=False)

    def cost(removed, selection, plan):
        compilations.append((removed, selection))
        return CandidateCostInputs(cost_marginal=(0, 0, 0), compile_calls=1)

    *_unused, kwargs = _setup(
        deletable_indices=(0, 1, 2), candidate_pool_for_deletion=lambda _d: (),
        max_joint_patch_evaluations=2, deletion_permission=permission,
        route_bundle=resolve_route_bundle('paper_ii_exchange_q_compiled_v1'),
        candidate_cost_inputs=cost, gamma_rcond=1e-10,
        population_accuracy_debt=False,
    )
    result = enumerate_structural_candidates(**kwargs)
    assert visits == [(0,), (1,)]
    assert [candidate.kind for candidate in result.candidates] == ['stay']
    assert result.guard.scored_count == 1
    assert result.compile_guard.scored_count == 0
    assert compilations == []
    assert result.deletion_discovery['inspected_count'] == 2
    assert result.deletion_discovery['truncated'] is True
    assert result.deletion_discovery['subset_population'] == 6
    assert result.deletion_discovery['max_batch_fraction'] is None


def test_discovery_limit_preserves_permitted_prefix_of_unfinished_rung():
    visits = []

    def permission(removed):
        visits.append(removed)
        return SimpleNamespace(permitted=removed in {(0,), (0, 1)})

    *_unused, kwargs = _setup(
        deletable_indices=(0, 1, 2), candidate_pool_for_deletion=lambda _d: (),
        max_joint_patch_evaluations=4, deletion_permission=permission,
    )
    result = enumerate_structural_candidates(**kwargs)
    assert visits == [(0,), (1,), (2,), (0, 1)]
    assert [candidate.removed_runtime_indices for candidate in result.candidates] == [(), (0,), (0, 1)]
    assert result.guard.scored_count == 3
    assert result.guard.rejected_family is None
    assert result.deletion_discovery['inspected_count'] == 4
    assert result.deletion_discovery['truncated'] is True


def test_discovery_exhaustion_leaves_pure_insertion_frontiers_reachable():
    visits, labels, members = [], [], []

    def permission(removed):
        visits.append(removed)
        return SimpleNamespace(permitted=False)

    *_unused, kwargs = _setup(
        deletable_indices=(0, 1, 2), cuts_by_atom={'ca': (0,), 'cb': (0,)},
        max_joint_patch_evaluations=4, deletion_permission=permission,
        max_insertion_batch_size=2,
    )
    result = None
    for label, candidates, telemetry in iter_structural_families(**kwargs):
        labels.append(label)
        members.extend(candidates)
        if telemetry is not None:
            result = telemetry
    assert visits == [(0,), (1,), (2,), (0, 1)]
    assert any(label.startswith('frontier_') for label in labels)
    assert sum(candidate.kind == 'insert' for candidate in members) >= 2
    assert result.deletion_discovery['truncated'] is True


def test_existing_whole_family_patch_admission_is_not_silently_changed():
    visits = []

    def permission(removed):
        visits.append(removed)
        return SimpleNamespace(permitted=True)

    *_unused, kwargs = _setup(
        deletable_indices=(0, 1, 2), candidate_pool_for_deletion=lambda _d: (),
        max_joint_patch_evaluations=3, deletion_permission=permission,
    )
    result = enumerate_structural_candidates(**kwargs)
    # Three permitted singleton deletions cannot fit after the stay slot.
    # This pre-existing family policy is reported, not repaired incidentally.
    assert visits == [(0,), (1,), (2,)]
    assert result.guard.rejected_family == 'singleton_d1'
    assert result.guard.scored_count == 1
    assert [candidate.kind for candidate in result.candidates] == ['stay']
    assert result.deletion_discovery['inspected_count'] == 3


def test_explicit_deletion_domain_and_unlimited_work_remain_respected():
    visits = []

    def permission(removed):
        visits.append(removed)
        return SimpleNamespace(permitted=False)

    *_unused, kwargs = _setup(
        deletable_indices=(0, 1, 2), candidate_pool_for_deletion=lambda _d: (),
        max_joint_patch_evaluations=None, max_deletion_batch_size=1,
        deletion_permission=permission,
    )
    result = enumerate_structural_candidates(**kwargs)
    assert visits == [(0,), (1,), (2,)]
    assert result.deletion_discovery['subset_population'] == 3
    assert result.deletion_discovery['inspected_count'] == 3
    assert result.deletion_discovery['truncated'] is False



def test_cached_permission_reuse_does_not_spend_another_discovery_slot(monkeypatch):
    import numpy as np
    from pipelines.time_dynamics.ap_mclachlan import exchange_structural as module
    from pipelines.time_dynamics.ap_mclachlan.deletion_permission import DeletionPermissionEvaluator

    state, _evaluation, _cache, _cuts, kwargs = _setup(
        deletable_indices=(0, 1, 2), candidate_pool_for_deletion=lambda _d: (),
        max_joint_patch_evaluations=2,
    )
    permission = DeletionPermissionEvaluator(
        gram=kwargs['base_K'], force=kwargs['base_f'], norm_b_sq=kwargs['norm_b_sq'],
        theta_runtime=state.theta_runtime, rotation_coefficients=np.ones(3),
        ray_distance_max=1e-3, normalized_schur_loss_max=1e-3, epsilon_norm=1e-12,
    )
    # Exercise cache reuse at the scheduler seam without repeating a solve.
    monkeypatch.setattr(module, 'iter_deletion_rungs', lambda *_a, **_k: iter([
        (1, iter(((0,), (0,), (1,), (2,)))),
    ]))
    result = enumerate_structural_candidates(**kwargs, deletion_permission=permission.assess)
    assert permission.summary()['evaluated_deletion_set_count'] == 2
    assert result.deletion_discovery['inspected_count'] == 2
    assert result.deletion_discovery['truncated'] is True
    assert result.guard.scored_count == 1


def test_family_discovery_receipt_reaches_actual_integration_payload():
    from test.test_ap_mclachlan_exchange_integration import _config, _run

    selection, payload = _run(_config(
        prune_ray_distance_tol=1e-15,
        prune_patch_smoothness_eta_max=1e-15,
        structural_score_floor=1e30,
    ))
    discovery = payload['selector_route']['deletion_discovery']
    assert discovery == selection.telemetry.deletion_discovery
    assert discovery['inspected_count'] == payload['deletion_permission']['evaluated_deletion_set_count']
    assert discovery['truncated'] is False

