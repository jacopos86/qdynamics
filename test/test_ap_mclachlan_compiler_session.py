"""Worker lifetime and fixed compiler context, independent of search caches."""
from __future__ import annotations

from dataclasses import replace
import copy
import threading
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
        work_receipt = original.work_receipt
        _work_success = original._work_success
        _work_failure = original._work_failure

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
            self._work_lock = threading.Lock()
            self._work = dict.fromkeys(("jobs_submitted", "jobs_succeeded", "jobs_failed", "jobs_timed_out"), 0)
            self.created.append(self)

        def compile_jobs(self, jobs, expected_keys):
            if len(jobs) != len(expected_keys):
                raise cpc.CompiledPatchCostError('compile job count does not match')
            self._work["jobs_submitted"] += len(jobs)
            if self.fail_next is not None:
                failure = self.fail_next
                type(self).fail_next = None
                self._work_failure(failure)
                self.terminate()
                raise failure
            rows = []
            for job, expected in zip(jobs, expected_keys):
                assert job[0] == 'word'
                key = cpc.word_key(job[1], contract=self.context['contract'],
                                   nq=self.context['nq'], ref_state=self.context['ref_state'])
                if key != expected:
                    self._work_failure(None)
                    self.terminate()
                    raise cpc.CompiledPatchCostError('worker word identity disagrees')
                # Synthetic resources test orchestration; existing real-Qiskit
                # regressions separately establish compiled-resource parity.
                triple = cpc.CompiledResourceTriple(len(job[1]), len(job[1]), 2 * len(job[1]))
                rows.append((key, triple))
                self.rows.append((key, triple.as_tuple()))
                self._work_success(None)
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
        assert [row[3:] for row in shared] == [row[3:] for row in separate]
        assert sum(row[0] for row in shared) <= sum(row[0] for row in separate)
        assert recording_pool.created[-1].rows == list(dict.fromkeys(separate_words))


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
            separate_view, shared_view = copy.deepcopy((separate, shared))
            for payload in (separate_view, shared_view):
                # A shared pool keeps its parent environment pins until run exit.
                payload['structural_scoring_parallelism'].pop('blas_thread_env', None)
                compile_receipt = payload['selector_route']['compile']
                for name in (
                    'compile_seconds', 'warm_seconds', 'compile_calls', 'compiler_work', 'compiler_work_total',
                    'cache_compilations_published', 'cache_hits', 'shared_cache_hits',
                    'session_result_cache', 'warm_primed_words', 'worker_blas_env_pinned',
                    'parent_blas_env_before_pool',
                ):
                    compile_receipt.pop(name, None)
                payload['selector_route'].pop('compile_calls_in_enumeration')
            assert shared_view == separate_view
        assert recording_pool.created[-1].rows == expected_words
        assert shared['selector_route']['compile']['compile_calls'] == 0
        assert shared['selector_route']['compile']['shared_cache_hits'] > 0
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


# Trajectory-local result reuse; these tests exercise the actual runtime facade.
def test_repeated_runtime_reuses_resources_but_not_geometry_or_mutable_content(recording_pool):
    from test.test_ap_mclachlan_compiled_patch_cost import _poly
    from src.quantum.vqe_latex_python_pairs import AnsatzTerm
    state = _state((XX, X0), (.1, .2))
    with cpc.CompilerSession() as session:
        first = runtime(state, session=session)
        expected = first.base_triple
        first_cache = first.cache
        first.close()
        second = runtime(_state((XX, X0), (.6, .9)), session=session)
        assert second.cache is not first_cache
        assert second.base_triple == expected
        assert second.cache.shared_cache_hits == 1
        assert second.telemetry()['compiler_work']['jobs_submitted'] == 0
        with pytest.raises(cpc.CompiledPatchCostError, match='closed'):
            first_cache.lookup(first.base_word_sha256)
        second.close()
        # A new object with the same label, then in-place polynomial mutation,
        # both change the word; no object-ID-based memo is permitted.
        term = AnsatzTerm(label='mutable', polynomial=_poly(('xx', .3)))
        cache = session.cache_for_decision(
            **context(), backend_identity=second.backend_provenance,
            word_compiler=session.lease(**context()).compile_word,
        )
        args = dict(contract=CompileContract(), nq=2, ref_state=context()['ref_state'], backend_obj=None)
        key0, _, hit0 = cache.compile_word_record((term,), **args)
        term.polynomial.return_polynomial()[0].p_coeff = .7
        key1, _, hit1 = cache.compile_word_record((term,), **args)
        assert key0 != key1 and not hit0 and not hit1
        cache.close()


@pytest.mark.parametrize('field,value', [
    ('backend_name', 'different'), ('optimization_level', 0), ('transpile_seed', 9182),
    ('structural_angle', .117), ('scale_floor', .234), ('compile_timeout_seconds', 1.2),
    ('coordinate_weights', {'n2q': 2., 'd2q': 1., 'dc': 1.}),
])
def test_shared_hit_still_checks_every_contract_field(recording_pool, field, value):
    with cpc.CompilerSession() as session:
        lease = session.lease(**context())
        cache = session.cache_for_decision(**context(), backend_identity={'backend': 'fixed'}, word_compiler=lease.compile_word)
        kwargs = dict(nq=2, ref_state=context()['ref_state'], contract=CompileContract(), backend_obj=None)
        cache.compile_word_record((XX,), **kwargs)
        kwargs['contract'] = replace(CompileContract(), **{field: value})
        with pytest.raises(cpc.CompiledPatchCostError, match='context changed'):
            cache.compile_word_record((XX,), **kwargs)
        assert session.work_receipt()['jobs_submitted'] == 1


def test_shared_hit_checks_reference_backend_environment_and_session_close(recording_pool, monkeypatch):
    identity = {'backend': 'fixed', 'version': '1'}
    ctx = context()
    with cpc.CompilerSession() as session:
        lease = session.lease(**ctx)
        cache = session.cache_for_decision(**ctx, backend_identity=identity, word_compiler=lease.compile_word)
        kwargs = dict(nq=2, ref_state=ctx['ref_state'], contract=ctx['contract'], backend_obj=None)
        key, _, _ = cache.compile_word_record((XX,), **kwargs)
        identity['version'] = '2'
        with pytest.raises(cpc.CompiledPatchCostError, match='backend identity changed'):
            cache.lookup(key)
        identity['version'] = '1'
        ctx['ref_state'][0] = 0
        with pytest.raises(cpc.CompiledPatchCostError, match='context changed'):
            cache.lookup(key)
        ctx['ref_state'][0] = 1
        monkeypatch.setenv(cpc.WORKER_THREAD_ENV_VARS[0], '3')
        with pytest.raises(cpc.CompiledPatchCostError, match='environment changed'):
            cache.lookup(key)
        monkeypatch.setenv(cpc.WORKER_THREAD_ENV_VARS[0], '1')
    with pytest.raises(cpc.CompiledPatchCostError, match='closed'):
        cache.lookup(key)


def test_bounded_lru_recompiles_evictions_and_preserves_decision_local_results(recording_pool):
    def cache(session):
        return session.cache_for_decision(**context(), backend_identity={'backend': 'fixed'},
                                          word_compiler=session.lease(**context()).compile_word)
    args = dict(nq=2, ref_state=context()['ref_state'], contract=CompileContract(), backend_obj=None)
    with cpc.CompilerSession(max_cached_words=1) as session:
        first = cache(session)
        x = first.compile_word_record((XX,), **args)
        z = first.compile_word_record((ZZ,), **args)
        assert first.compile_word_record((XX,), **args)[2]
        assert first.compile_calls == 2  # Session eviction does not evict the local memo.
        first.close()
        second = cache(session)
        assert second.compile_word_record((ZZ,), **args)[2]
        again = second.compile_word_record((XX,), **args)
        assert again[:2] == x[:2] and not again[2]
        assert session.work_receipt()['jobs_submitted'] == 3
        assert session.cache_receipt() == {'max_words': 1, 'resident_words': 1, 'evictions': 2}
        second.close()


def test_duplicate_prime_counts_both_actual_results_but_stores_one():
    cache = cpc.CompiledPatchCostCache()
    triple = cpc.CompiledResourceTriple(1, 2, 3)
    assert cache.prime('key', triple)
    assert not cache.prime('key', triple)
    assert cache.compile_calls == 2 and len(cache) == 1


@pytest.mark.parametrize('later_completes', [False, True])
def test_failed_warm_keeps_completed_work_counts_without_publishing_partial_batch(later_completes):
    # Exercise actual CompileWarmPool.compile_jobs with a synchronous fake
    # dispatcher: job0 succeeds, job1 fails, job2 has already succeeded.
    # The old post-loop jobs_run update would erase all three outcomes.
    pool = object.__new__(cpc.CompileWarmPool)
    pool._terminated = False
    pool.timeout_seconds = 1.
    pool.jobs_run = 0
    pool._work_lock = threading.Lock()
    pool._work = dict.fromkeys(('jobs_submitted', 'jobs_succeeded', 'jobs_failed', 'jobs_timed_out'), 0)
    pool.prior_env = {}
    class Result:
        def __init__(self, result=None, error=None): self.result, self.error = result, error
        def get(self, **kwargs):
            if self.error: raise self.error
            return self.result
    class Dispatcher:
        def apply_async(self, fn, args, callback, error_callback):
            key = args[0][1]
            if key == 'bad':
                error = cpc.CompiledPatchCostError('worker failed')
                error_callback(error)
                return Result(error=error)
            result = (key, (1, 2, 3))
            if key != 'later' or later_completes:
                callback(result)
            return Result(result=result)
        def terminate(self): pass
        def join(self): pass
    pool._pool = Dispatcher()
    cache = cpc.CompiledPatchCostCache()
    with pytest.raises(cpc.CompiledPatchCostError, match='worker failed'):
        cpc.warm_word_cache(cache, pool, [(k, ('word', k)) for k in ('ok', 'bad', 'later')])
    assert cache.compile_calls == 0 and len(cache) == 0
    assert pool.jobs_run == 1
    assert pool.work_receipt() == dict(jobs_submitted=3, jobs_succeeded=1+int(later_completes), jobs_failed=1,
                                      jobs_timed_out=0, jobs_unresolved=int(not later_completes))


def test_failed_word_is_not_cached_and_pool_work_survives_replacement(recording_pool):
    with cpc.CompilerSession() as session:
        first = runtime(_state((XX,), (.1,)), session=session)
        first.close()
        recording_pool.fail_next = cpc.CompileTimeoutError('first changed word fails')
        with pytest.raises(cpc.CompileTimeoutError):
            runtime(_state((ZZ,), (.2,)), session=session)
        third = runtime(_state((ZZ,), (.3,)), session=session)
        assert third.cache.shared_cache_hits == 0
        work = session.work_receipt()
        assert work['jobs_submitted'] == 3 and work['jobs_succeeded'] == 2 and work['jobs_failed'] == 1
        third.close()


def test_session_does_not_retain_closed_runtime_callbacks(recording_pool):
    import gc
    import weakref
    with cpc.CompilerSession() as session:
        cost = runtime(_state((XX,), (.1,)), session=session)
        reference = weakref.ref(cost)
        cost.close()
        del cost
        gc.collect()
        assert reference() is None
        assert session.cache_receipt()['resident_words'] == 1


@pytest.mark.parametrize('changes', [{'nq': 3}, {'workers': 1}, {'ref_state': np.array([1,0,0,0], dtype=np.complex64)}])
def test_new_decision_cannot_change_sealed_context_even_if_word_is_cached(recording_pool, changes):
    with cpc.CompilerSession() as session:
        cost = runtime(_state((XX,), (.1,)), session=session)
        cost.close()
        with pytest.raises(cpc.CompiledPatchCostError, match='context changed'):
            session.cache_for_decision(**context(**changes), backend_identity=cost.backend_provenance,
                                       word_compiler=lambda *a, **k: None)
        assert session.work_receipt()['jobs_submitted'] == 1


def test_failed_warm_replays_cached_success_then_replaces_pool_only_for_miss(recording_pool, monkeypatch):
    with cpc.CompilerSession() as session:
        cost = runtime(_state((XX,), (.1,)), session=session)
        cost._class_members['a'] = {0: (0,), 1: (1,)}
        cost.cost_inputs((), (('a', 0),), None)
        failed_lease = cost.pool
        recording_pool.fail_next = cpc.CompiledPatchCostError('warm word failed')
        assert cost.warm((((), (('a', 1),), None),)) == 0
        before = session.work_receipt()
        assert failed_lease._terminated
        with pytest.raises(cpc.CompiledPatchCostError, match='no longer available'):
            compile_word(failed_lease)
        cached = cost.cost_inputs((), (('a', 0),), None)
        assert cached.compile_calls == 0 and session.work_receipt() == before
        assert len(recording_pool.created) == 1
        with monkeypatch.context() as changed:
            changed.setenv(cpc.WORKER_THREAD_ENV_VARS[0], 'changed-after-failure')
            with pytest.raises(cpc.CompiledPatchCostError, match='environment changed'):
                cost.cache.lookup(cost.base_word_sha256)
        cost.cost_inputs((), (('a', 1),), None)
        assert len(recording_pool.created) == 2
        after = session.work_receipt()
        assert all(after[k] >= before[k] for k in before if k != 'jobs_unresolved')
        assert all(v >= 0 for v in cost.telemetry()['compiler_work'].values())
        assert after['jobs_submitted'] == 4 and after['jobs_succeeded'] == 3 and after['jobs_failed'] == 1
        cost.close()


def test_retiring_failed_pool_keeps_unresolved_work_out_of_next_decision_delta(recording_pool):
    with cpc.CompilerSession() as session:
        lease = session.lease(**context())
        recording_pool.fail_next = cpc.CompiledPatchCostError('batch failed with outstanding work')
        with pytest.raises(cpc.CompiledPatchCostError):
            lease.compile_jobs((('word', (XX,)),) * 3, ('unused',) * 3)
        assert session.work_receipt()['jobs_unresolved'] == 2
        cost = runtime(_state((ZZ,), (.2,)), session=session)
        receipt = cost.telemetry()
        assert receipt['compiler_work'] == dict(jobs_submitted=1, jobs_succeeded=1,
            jobs_failed=0, jobs_timed_out=0, jobs_unresolved=0)
        assert receipt['compiler_work_total'] == dict(jobs_submitted=4, jobs_succeeded=1,
            jobs_failed=1, jobs_timed_out=0, jobs_unresolved=2)
        cost.close()
