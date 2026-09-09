"""Worker lifetime and fixed compiler context, independent of search caches."""
from __future__ import annotations

from dataclasses import replace
import os
from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.time_dynamics.ap_mclachlan import compiled_patch_cost as cpc
from pipelines.time_dynamics.ap_mclachlan import exchange_integration as integration
from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import _PruneControllerRuntimeState
from pipelines.time_dynamics.ap_mclachlan.exchange_config import CompileContract
from pipelines.time_dynamics.ap_mclachlan.geometry_eval import evaluate_mclachlan_geometry
from test.test_ap_mclachlan_compiled_patch_cost import CA, CB, XX, ZZ, X0, _state
from test.test_ap_mclachlan_exchange_integration import HAM, POLICY, _config
from test.test_ap_mclachlan_exchange_integration import _state as selection_state


@pytest.fixture
def recording_pool(monkeypatch):
    original = cpc.CompileWarmPool

    class RecordingPool:
        created = []
        fail_next = None
        compile_word = original.compile_word
        _restore_parent_env = original._restore_parent_env

        def __init__(self, **kwargs):
            assert kwargs['base_terms'] == () and kwargs['atom_terms'] == {}
            self.context = kwargs
            self.timeout_seconds = kwargs['contract'].compile_timeout_seconds
            self.prior_env = {k: os.environ.get(k) for k in cpc.WORKER_THREAD_ENV_VARS}
            self.pinned_env = dict.fromkeys(cpc.WORKER_THREAD_ENV_VARS, '1')
            os.environ.update(self.pinned_env)
            self._terminated = False
            self.rows = []
            self.jobs_run = 0
            self.created.append(self)

        def compile_jobs(self, jobs, expected_keys):
            if len(jobs) != len(expected_keys):
                raise cpc.CompiledPatchCostError('compile job count does not match')
            if self.fail_next is not None:
                failure = self.fail_next
                type(self).fail_next = None
                self.terminate()
                raise failure
            rows = []
            for job, expected in zip(jobs, expected_keys):
                assert job[0] == 'word'
                key = cpc.word_key(job[1], contract=self.context['contract'],
                                   nq=self.context['nq'], ref_state=self.context['ref_state'])
                if key != expected:
                    self.terminate()
                    raise cpc.CompiledPatchCostError('worker word identity disagrees')
                # Synthetic resources test orchestration; existing real-Qiskit
                # regressions separately establish compiled-resource parity.
                triple = cpc.CompiledResourceTriple(len(job[1]), len(job[1]), 2 * len(job[1]))
                rows.append((key, triple))
                self.rows.append((key, triple.as_tuple()))
            self.jobs_run += len(rows)
            return rows

        def close(self):
            self._terminated = True
            self._restore_parent_env()

        terminate = close

    monkeypatch.setattr(cpc, 'CompileWarmPool', RecordingPool)
    return RecordingPool


def context(**overrides):
    result = dict(contract=CompileContract(), nq=2,
                  ref_state=np.array([1, 0, 0, 0], dtype=complex), workers=2)
    result.update(overrides)
    return result


def compile_word(lease, terms=(XX,), **overrides):
    kwargs = context(**overrides)
    kwargs.pop('workers')
    return lease.compile_word(terms, backend_obj=None, **kwargs)


@pytest.mark.parametrize('exceptional', [False, True])
def test_session_restores_original_environment_and_closes_leases(recording_pool, monkeypatch, exceptional):
    for index, name in enumerate(cpc.WORKER_THREAD_ENV_VARS):
        if index % 2:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, str(index + 2))
    prior = {name: os.environ.get(name) for name in cpc.WORKER_THREAD_ENV_VARS}
    failure = RuntimeError('outside compiler')
    try:
        with cpc.CompilerSession() as owner:
            assert not recording_pool.created
            first = owner.lease(**context())
            compile_word(first)
            first.close()
            with pytest.raises(cpc.CompiledPatchCostError, match='no longer available'):
                compile_word(first)
            second = owner.lease(**context())
            assert second.prior_env == prior
            assert len(recording_pool.created) == 1
            compile_word(second, (ZZ, XX))
            if exceptional:
                raise failure
    except RuntimeError as caught:
        assert exceptional and caught is failure
    assert recording_pool.created[0]._terminated
    assert {name: os.environ.get(name) for name in prior} == prior
    with pytest.raises(cpc.CompiledPatchCostError, match='no longer available'):
        compile_word(second)
    with pytest.raises(cpc.CompiledPatchCostError, match='closed'):
        owner.lease(**context())


@pytest.mark.parametrize('field,value', [
    ('backend_name', 'different'), ('optimization_level', 0), ('transpile_seed', 9182),
    ('structural_angle', .117), ('scale_floor', .234), ('compile_timeout_seconds', 1.2),
    ('coordinate_weights', {'n2q': 2., 'd2q': 1., 'dc': 1.}),
])
def test_every_contract_field_is_checked_on_lease_and_direct_word(recording_pool, field, value):
    changed = replace(CompileContract(), **{field: value})
    assert changed != CompileContract()
    with cpc.CompilerSession() as owner:
        lease = owner.lease(**context())
        for operation in (lambda: owner.lease(**context(contract=changed)),
                          lambda: compile_word(lease, contract=changed)):
            with pytest.raises(cpc.CompiledPatchCostError, match='context changed'):
                operation()
        assert recording_pool.created[0].jobs_run == 0


def test_reference_snapshot_width_workers_and_effective_environment(recording_pool, monkeypatch):
    current = context()
    reference = current['ref_state']
    with cpc.CompilerSession() as owner:
        lease = owner.lease(**current)
        frozen = recording_pool.created[0].context['ref_state']
        assert not np.shares_memory(reference, frozen) and not frozen.flags.writeable
        reference[0] = 0
        assert frozen[0] == 1
        for changes in ({'ref_state': reference}, {'nq': 3}, {'workers': 1},
                        {'ref_state': frozen.astype(np.complex64)}):
            with pytest.raises(cpc.CompiledPatchCostError, match='context changed'):
                owner.lease(**context(**changes))
        for changes in ({'ref_state': reference}, {'nq': 3}):
            with pytest.raises(cpc.CompiledPatchCostError, match='context changed'):
                compile_word(lease, **changes)
        monkeypatch.setenv(cpc.WORKER_THREAD_ENV_VARS[0], '4')
        with pytest.raises(cpc.CompiledPatchCostError, match='environment changed'):
            compile_word(lease)


@pytest.mark.parametrize('failure_kind', ['timeout', 'word_key'])
def test_failure_preserves_error_and_restarts_only_on_next_lease(recording_pool, failure_kind):
    with cpc.CompilerSession() as owner:
        first = owner.lease(**context())
        if failure_kind == 'timeout':
            failure = cpc.CompileTimeoutError('sealed timeout')
            recording_pool.fail_next = failure
            with pytest.raises(cpc.CompileTimeoutError) as raised:
                compile_word(first)
            assert raised.value is failure
        else:
            with pytest.raises(cpc.CompiledPatchCostError, match='word identity'):
                first.compile_jobs((('word', (XX,)),), ('wrong',))
        assert first._terminated and len(recording_pool.created) == 1
        with pytest.raises(cpc.CompiledPatchCostError, match='no longer available'):
            compile_word(first)
        second = owner.lease(**context())
        assert len(recording_pool.created) == 2
        compile_word(second)
        assert sum(pool.jobs_run for pool in recording_pool.created) == 1


def test_preflight_rejections_do_not_consume_or_replace_pool(recording_pool):
    with cpc.CompilerSession() as owner:
        lease = owner.lease(**context())
        for jobs, keys in (((('word', (XX,)),), ()), ((('trial', 'a', 0),), ('x',))):
            with pytest.raises(cpc.CompiledPatchCostError):
                lease.compile_jobs(jobs, keys)
        assert not lease._terminated and recording_pool.created[0].jobs_run == 0
        compile_word(lease)


def runtime(state, *, session=None, atom=CA):
    return integration.CompiledCostRuntime(
        state=state, atoms_by_id={'a': SimpleNamespace(term=atom)},
        cuts_by_atom={'a': (0,)}, blocks=(), contract=CompileContract(),
        workers=2, compiler_session=session,
    )


def test_runtime_uses_current_words_but_fresh_caches(recording_pool):
    states = (_state((X0, XX), (.1, .2)), _state((ZZ, XX, X0), (.2, .3, .1)))
    atoms = (CA, CB)  # Same candidate ID now denotes a different operator.

    def evaluate(session):
        observations = []
        for state, atom in zip(states, atoms):
            cost = runtime(state, session=session, atom=atom)
            try:
                # Explicitly isolate placement mechanics from this lifetime test.
                cost._class_members['a'] = {0: (0, 1)}
                cost.warm((((0,), (('a', 0),), None),))
                observations.append((cost.cache.compile_calls, cost.cache.cache_hits,
                                     cost.warm_primed, cost.base_word_sha256,
                                     dict(cost.cache._entries)))
            finally:
                cost.close()
        return observations

    separate = evaluate(None)
    separate_words = [row for pool in recording_pool.created for row in pool.rows]
    assert len(recording_pool.created) == 2 and all(p._terminated for p in recording_pool.created)
    with cpc.CompilerSession() as session:
        shared = evaluate(session)
        assert len(recording_pool.created) == 3 and not recording_pool.created[-1]._terminated
        assert shared == separate
        assert recording_pool.created[-1].rows == separate_words


def test_standalone_runtime_constructor_failure_releases_private_session(recording_pool):
    prior = {name: os.environ.get(name) for name in cpc.WORKER_THREAD_ENV_VARS}
    failure = cpc.CompileTimeoutError('base compilation failed')
    recording_pool.fail_next = failure
    with pytest.raises(cpc.CompileTimeoutError) as raised:
        runtime(_state((XX,), (.1,)))
    assert raised.value is failure
    assert len(recording_pool.created) == 1 and recording_pool.created[0]._terminated
    assert {name: os.environ.get(name) for name in prior} == prior


def select(session=None, **config_overrides):
    state = selection_state()
    evaluation = evaluate_mclachlan_geometry(
        state=state, hamiltonian=HAM, theta_runtime=state.theta_runtime,
        time=0., include_tangent_matrix=True,
    )
    from pipelines.time_dynamics.ap_mclachlan.fixed_step import solve_fixed_mclachlan_step
    step = solve_fixed_mclachlan_step(evaluation.geometry, inverse_policy=POLICY)
    return integration.select_generalized_exchange_patch(
        state=state, hamiltonian=HAM, theta_runtime=state.theta_runtime, time=0.,
        base_evaluation=evaluation, base_step=step, inverse_policy=POLICY,
        support_config=_config(exchange_route_bundle='paper_ii_exchange_q_compiled_v1',
                               compile_workers=2, **config_overrides),
        runtime_state=_PruneControllerRuntimeState(), time_index=3,
        compiler_session=session,
    )


@pytest.mark.parametrize('policy', ['greedy', 'bounded_exhaustive', 'block_exchange'])
@pytest.mark.parametrize('gain', ['joint_batch', 'singleton_factorized'])
def test_selector_scientific_payload_and_work_are_identical(recording_pool, policy, gain):
    settings = dict(search_policy=policy, gain_evaluation=gain,
                    max_joint_patch_evaluations=16, max_compiled_patches_per_checkpoint=16)
    _, separate = select(**settings)
    expected_words = list(recording_pool.created[-1].rows)
    with cpc.CompilerSession() as session:
        for _ in range(2):
            _, shared = select(session, **settings)
            for payload in (separate, shared):
                # A shared pool keeps its parent environment pins until run exit.
                payload['structural_scoring_parallelism'].pop('blas_thread_env', None)
                compile_receipt = payload['selector_route']['compile']
                for name in ('compile_seconds', 'warm_seconds'):
                    compile_receipt.pop(name, None)
            assert shared == separate
        assert recording_pool.created[-1].rows == expected_words * 2
        assert len(recording_pool.created) == 2


def test_selector_failure_after_runtime_creation_releases_private_owner(recording_pool, monkeypatch):
    failure = RuntimeError('ranking setup failed')
    def fail(**kwargs):
        raise failure
    monkeypatch.setattr(integration, 'build_population_ranker', fail)
    with pytest.raises(RuntimeError) as raised:
        select()
    assert raised.value is failure
    assert len(recording_pool.created) == 1 and recording_pool.created[0]._terminated


def test_selector_statistics_failure_still_releases_constructed_runtime(recording_pool, monkeypatch):
    failure = RuntimeError('candidate statistics failed')
    build = integration.build_selector_inputs

    class FailedStatistics:
        @property
        def column_count(self):
            raise failure

    def build_with_failed_statistics(**kwargs):
        inputs = build(**kwargs)
        inputs['structural_kwargs']['cache'] = FailedStatistics()
        return inputs

    monkeypatch.setattr(integration, 'build_selector_inputs', build_with_failed_statistics)
    prior = {name: os.environ.get(name) for name in cpc.WORKER_THREAD_ENV_VARS}
    with pytest.raises(RuntimeError) as raised:
        select()
    assert raised.value is failure
    assert len(recording_pool.created) == 1 and recording_pool.created[0]._terminated
    assert {name: os.environ.get(name) for name in prior} == prior
