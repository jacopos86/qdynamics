"""Compiler worker lifetime at the public trajectory boundary."""

from __future__ import annotations

import multiprocessing
import os

import pytest

from pipelines.time_dynamics.ap_mclachlan import adaptive_trajectory as trajectory
from pipelines.time_dynamics.ap_mclachlan import compiled_patch_cost as costs
from pipelines.time_dynamics.ap_mclachlan.fixed_step import SolveRepairConfig
from test.test_ap_mclachlan_exchange_integration import HAM, POLICY, _config, _state


def test_callback_failure_after_compilation_releases_trajectory_workers(monkeypatch):
    """A failed progress sink must not leave the idle shared pool alive."""

    prior = {}
    for index, name in enumerate(costs.WORKER_THREAD_ENV_VARS):
        if name == "RAYON_NUM_THREADS":
            monkeypatch.delenv(name, raising=False)
            prior[name] = None
        else:
            value = str(index + 2)
            monkeypatch.setenv(name, value)
            prior[name] = value
    initial_children = {child.pid for child in multiprocessing.active_children()}
    failure = RuntimeError("progress sink failed after the first compiled checkpoint")
    observed_workers = set()

    def progress(event):
        if event["phase"] != "checkpoint_done":
            return
        observed_workers.update(
            child.pid for child in multiprocessing.active_children()
            if child.pid not in initial_children
        )
        assert observed_workers, "fixture did not reach a live compiled-cost worker"
        raise failure

    with pytest.raises(RuntimeError) as raised:
        trajectory.run_append_mclachlan_trajectory(
            state=_state(),
            hamiltonian=HAM,
            times=(0.0, 0.02),
            inverse_policy=POLICY,
            support_patch_config=_config(
                exchange_route_bundle="paper_ii_exchange_q_compiled_v1",
                search_policy="greedy",
                gain_evaluation="joint_batch",
                insertion_gate_mode="mclachlan_l2",
                insertion_l2_cut=0.0,
                max_joint_patch_evaluations=16,
                max_compiled_patches_per_checkpoint=16,
                max_insertion_rounds_per_checkpoint=1,
                max_insertion_batch_size=1,
                max_deletion_batch_size=1,
                compile_workers=1,
            ),
            solve_repair_config=SolveRepairConfig.minimal_profile(),
            progress_callback=progress,
        )

    assert raised.value is failure
    assert observed_workers
    assert {child.pid for child in multiprocessing.active_children()} <= initial_children
    assert {name: os.environ.get(name) for name in prior} == prior


def test_invalid_time_grid_does_not_start_compiler_workers(monkeypatch):
    """Owning a compiler session does not eagerly spawn workers."""

    def unexpected_pool(*args, **kwargs):
        raise AssertionError("invalid trajectory created compiler workers")

    monkeypatch.setattr(costs, "CompileWarmPool", unexpected_pool)
    prior = {name: os.environ.get(name) for name in costs.WORKER_THREAD_ENV_VARS}
    with pytest.raises(ValueError, match="monotonically nondecreasing"):
        trajectory.run_append_mclachlan_trajectory(
            state=_state(), hamiltonian=HAM, times=(0.02, 0.0),
        )
    assert {name: os.environ.get(name) for name in prior} == prior
