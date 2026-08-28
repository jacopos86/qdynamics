from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.paper_iii_qse_score_ablation import (
    FACTORIAL_SCORE_MODES,
    factorial_score_profiles,
    run_factorial_score_ablation,
    run_factorial_score_ablation_campaign,
)
from pipelines.qse_spectra.core import computational_basis_state, pauli_string_basis_element
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


def test_factorial_profiles_cross_four_scores_with_novelty_reward_on_and_off() -> None:
    profiles = factorial_score_profiles(max_records=11, target_root_count=3)

    assert len(profiles) == 8
    assert {profile.score_mode for profile in profiles} == set(FACTORIAL_SCORE_MODES)
    for score_mode in FACTORIAL_SCORE_MODES:
        cells = [profile for profile in profiles if profile.score_mode == score_mode]
        assert {profile.additive_novelty_reward for profile in cells} == {True, False}
    assert all(profile.selector_config.max_records == 11 for profile in profiles)
    assert all(profile.selector_config.geometry_target_roots == 3 for profile in profiles)
    assert all(profile.selector_config.geometry_min_metric_novelty == 1.0e-12 for profile in profiles)
    assert all(profile.selector_config.geometry_cost_discount_alpha == 1.0 for profile in profiles)


def _tiny_problem() -> SimpleNamespace:
    low = 1.0
    high = 2.0
    hamiltonian = PauliPolynomial("JW")
    for label, coefficient in (
        ("ee", (low + high) / 2.0),
        ("ze", -low / 2.0),
        ("ez", -high / 2.0),
    ):
        hamiltonian.add_term(PauliTerm(2, ps=label, pc=coefficient))
    basis = (
        pauli_string_basis_element("xe", nq=2, name="low"),
        pauli_string_basis_element("ex", nq=2, name="high"),
    )

    def resource_triple(indices):
        value = float(sum((1.0, 2.0)[int(index)] for index in indices))
        return {"n2q": value, "d2q": value, "dc": value}

    def estimator_cost(indices):
        count = len(indices)
        return {
            "pair_count": count * (count + 1) // 2,
            "naive_terms": count,
            "distinct_words": count,
            "qwc_groups": count,
        }

    return SimpleNamespace(
        regime="tiny",
        u=0.0,
        g_ep=0.0,
        n_ph_max=1,
        num_qubits=2,
        hamiltonian=hamiltonian,
        basis=basis,
        costs=(1.0, 2.0),
        resources=({}, {}),
        ground=computational_basis_state(2, "00"),
        spectrum=(0.0, low, high),
        references=(low, high),
        pool_digest="tiny-pool",
        reference_key="tiny-reference",
        resource_triple=resource_triple,
        estimator_cost=estimator_cost,
    )


def test_factorial_runner_reports_every_prefix_before_and_after_exchange() -> None:
    payload = run_factorial_score_ablation(
        _tiny_problem(),
        k_max=2,
        target_root_count=1,
        exchange_max_rounds=2,
    )

    assert payload["schema_version"] == "paper_iii_qse_score_factorial_v2"
    assert payload["profile_count"] == 8
    assert payload["target_root_count"] == 1
    assert payload["growth_rounds"] == [1, 2]
    for profile in payload["profiles"].values():
        assert len(profile["trace"]) == 2
        first, second = profile["trace"]
        assert first["pre_exchange"]["support_size"] == 1
        assert second["pre_exchange"]["support_size"] == 2
        for row in (first, second):
            assert "selected_support" in row["pre_exchange"]
            assert "root_energies" in row["pre_exchange"]
            assert "root_abs_errors" in row["pre_exchange"]
            assert "ky_fan_sum" in row["pre_exchange"]
            assert "target_root_energy" in row["pre_exchange"]
            assert "gap_to_next_root" in row["pre_exchange"]
            assert "generalized_residual_norms" in row["pre_exchange"]
            assert "physical_residual_norms" in row["pre_exchange"]
            assert "retained_rank" in row["pre_exchange"]
            assert "exchange" in row
            assert "post_exchange" in row
        assert second["pre_exchange"]["matching_to_previous"] is not None
        assert second["pre_exchange"]["matching_to_previous"]["principal_cosines"]
        assert first["post_exchange"]["matching_to_pre_exchange"] is not None


def test_campaign_runner_collects_complete_factorials_without_changing_the_cells() -> None:
    payload = run_factorial_score_ablation_campaign(
        [_tiny_problem()],
        k_max=2,
        target_root_count=1,
        exchange_max_rounds=2,
        regime_set="test",
    )

    assert payload["schema_version"] == "paper_iii_qse_score_factorial_campaign_v1"
    assert payload["regime_set"] == "test"
    assert payload["case_count"] == 1
    assert payload["settings"] == {
        "k_max": 2,
        "target_root_count": 1,
        "exchange_max_rounds": 2,
    }
    assert len(payload["cases"]) == 1
    assert payload["cases"][0]["profile_count"] == 8
    assert payload["cases"][0]["problem"]["regime"] == "tiny"


def test_incomplete_prefix_does_not_report_a_partial_window_as_max_root_error() -> None:
    payload = run_factorial_score_ablation(
        _tiny_problem(),
        k_max=2,
        target_root_count=2,
        exchange_max_rounds=1,
    )

    for profile in payload["profiles"].values():
        first, second = profile["trace"]
        assert first["pre_exchange"]["target_window_status"] == "TARGET_WINDOW_NOT_RESOLVED"
        assert first["pre_exchange"]["max_root_abs_error"] is None
        assert first["pre_exchange"]["available_max_root_abs_error"] is not None
        assert second["pre_exchange"]["target_window_status"] == "TARGET_WINDOW_RESOLVED"
        assert second["pre_exchange"]["max_root_abs_error"] is not None


def test_reported_growth_rounds_stop_when_the_shared_novelty_floor_exhausts_the_pool() -> None:
    problem = _tiny_problem()
    problem.basis = (*problem.basis, problem.basis[0])
    problem.costs = (1.0, 2.0, 3.0)
    problem.resources = ({}, {}, {})

    def resource_triple(indices):
        value = float(sum(problem.costs[int(index)] for index in indices))
        return {"n2q": value, "d2q": value, "dc": value}

    problem.resource_triple = resource_triple
    payload = run_factorial_score_ablation(
        problem,
        k_max=3,
        target_root_count=1,
        exchange_max_rounds=1,
    )

    assert payload["requested_growth_cap"] == 3
    assert payload["growth_rounds"] == [1, 2]
    assert all(profile["growth_rounds"] == [1, 2] for profile in payload["profiles"].values())
