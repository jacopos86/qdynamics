"""Small contracts for the interim Paper-II paired-results diagnostic."""

from pipelines.time_dynamics.diagnostics.partial_factorial_report import (
    _dominates,
    _error_no_worse_and_c2q_lower,
)


def test_dominance_requires_no_regression_and_one_strict_improvement() -> None:
    keys = ("a", "b")

    assert _dominates({"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 3.0}, keys)
    assert not _dominates({"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 2.0}, keys)
    assert not _dominates({"a": 0.5, "b": 2.5}, {"a": 1.0, "b": 2.0}, keys)


def test_observable_qualification_requires_both_accuracy_and_lower_cost() -> None:
    apm = {"error": 1.0e-4, "graph_proxy": {"c_hat_2q": 80.0}}
    avqds = {"error": 1.0e-4, "graph_proxy": {"c_hat_2q": 100.0}}

    assert _error_no_worse_and_c2q_lower(apm, avqds, "error")

    apm["error"] = 1.1e-4
    assert not _error_no_worse_and_c2q_lower(apm, avqds, "error")

    apm["error"] = 0.9e-4
    apm["graph_proxy"]["c_hat_2q"] = 100.0
    assert not _error_no_worse_and_c2q_lower(apm, avqds, "error")
