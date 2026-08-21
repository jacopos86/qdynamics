"""Run locks: what they bind, and what they refuse to compare."""

from __future__ import annotations

import pytest

from pipelines.time_dynamics.run_lock import (
    RUN_LOCK_SCHEMA_V1,
    IncomparableRunsError,
    assert_comparable,
    build_run_lock,
    physics_fingerprint,
)


def _lock(**overrides):
    params = dict(
        seed_artifact_json=None, family_key="hh", n_ph_max=1,
        times=[0.0, 0.5, 1.0], drive_profile={"A": 0.6, "omega": 3.0},
        integrator_method="rk4",
        inverse_policy={"pinv_rcond": 1e-10, "ridge_lambda": 1e-7, "solve_damping": 0.0},
        solve_repair={"enabled": True},
        structural_policy="exchange",
        guards={"max_structural_pool_size": 8},
    )
    params.update(overrides)
    return build_run_lock(**params)


def test_lock_records_physics_numerics_policy_and_code() -> None:
    lock = _lock()
    assert lock["schema"] == RUN_LOCK_SCHEMA_V1
    assert lock["physics"]["family_key"] == "hh"
    assert lock["physics"]["n_ph_max"] == 1
    assert lock["physics"]["time_grid"]["point_count"] == 3
    assert lock["numerics"]["integrator_method"] == "rk4"
    assert lock["policy"]["structural_policy"] == "exchange"
    assert lock["code_revision"]  # a revision or "unknown", never absent


def test_arms_differing_only_in_policy_remain_comparable() -> None:
    # Two arms of a comparison must differ in policy and agree on physics.
    exchange = _lock(structural_policy="exchange")
    avqds = _lock(structural_policy="avqds")
    assert exchange["physics_fingerprint"] == avqds["physics_fingerprint"]
    assert assert_comparable([exchange, avqds])


@pytest.mark.parametrize(
    "difference",
    [
        {"n_ph_max": 3},                       # different truncation
        {"family_key": "hubbard"},             # different Hamiltonian
        {"times": [0.0, 0.5, 1.0, 1.5]},       # different reporting grid
        {"drive_profile": {"A": 0.9, "omega": 3.0}},  # different drive
        {"drive_profile": None},               # driven vs undriven
    ],
)
def test_physics_differences_block_comparison(difference) -> None:
    base = _lock()
    other = _lock(**difference)
    assert base["physics_fingerprint"] != other["physics_fingerprint"]
    with pytest.raises(IncomparableRunsError, match="physics fingerprint"):
        assert_comparable([base, other])


def test_exact_reference_pointer_does_not_affect_comparability() -> None:
    # A reference is a reporting artifact, not an input to the trajectory.
    with_ref = _lock(exact_reference_json="somewhere/reference.json")
    without = _lock()
    assert with_ref["physics_fingerprint"] == without["physics_fingerprint"]
    assert assert_comparable([with_ref, without])


def test_empty_input_is_an_error_not_a_pass() -> None:
    with pytest.raises(IncomparableRunsError):
        assert_comparable([])


def test_fingerprint_is_order_independent_over_keys() -> None:
    a = _lock()["physics"]
    b = dict(reversed(list(a.items())))
    assert physics_fingerprint(a) == physics_fingerprint(b)
