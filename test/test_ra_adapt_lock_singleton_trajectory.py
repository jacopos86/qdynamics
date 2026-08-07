from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipelines.contracts.problem import ProblemRequest
from pipelines.static_adapt import adapt_pipeline
from pipelines.static_adapt.builders.problem_registry import (
    resolve_problem_context,
)
from pipelines.static_adapt.sr_snake import (
    SRExecutionPolicy,
    SRRunRequest,
    SRStopPolicy,
)
from pipelines.static_adapt.ra_adapt.runtime import _execute_sr_snake


_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "ra_adapt_singleton_trajectory_nph3.json"
)


def _problem():
    return resolve_problem_context(
        ProblemRequest(
            problem_key="hh",
            num_sites=2,
            t=1.0,
            u=2.0,
            dv=0.0,
            omega0=1.0,
            g_ep=1.0,
            n_ph_max=3,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
        ),
        exact_energy_impl=adapt_pipeline._exact_gs_energy_for_problem,
    )


def _project(result):
    return {
        "accepted_trajectory": [
            {
                "controller_round": receipt.controller_round,
                "energy": receipt.energy,
                "insertion_positions": list(receipt.insertion_positions),
                "operators": list(receipt.operators),
            }
            for receipt in result.accepted_trajectory
        ],
        "accepted_refit_receipts": [
            {
                "scope": receipt.accepted_refit.scope,
                "coordinate_chart": receipt.accepted_refit.coordinate_chart,
                "base_chart_policy": receipt.accepted_refit.base_chart_policy,
                "full_ansatz": receipt.accepted_refit.full_ansatz,
                "supported_rank": receipt.accepted_refit.supported_rank,
            }
            for receipt in result.scientific_replay
        ],
        "trust_receipts": [
            receipt.trust_solve.to_dict()
            for receipt in result.scientific_replay
        ],
        "estimator_accounting": {
            "components": (
                result.estimator_accounting.all_work.components.to_dict()
            ),
            "s_alg": result.estimator_accounting.all_work.s_alg,
            "prefix_components": [
                row.components.to_dict()
                for row in result.canonical_reporting.accepted_prefix_work
            ],
            "prefix_s_alg": [
                row.s_alg
                for row in result.canonical_reporting.accepted_prefix_work
            ],
        },
        "problem": {
            "g_ep": result.problem.g_ep,
            "n_ph_max": result.problem.n_ph_max,
            "num_sites": result.problem.num_sites,
            "omega0": result.problem.omega0,
            "ordering": result.problem.ordering,
            "problem_request_sha256": result.problem.problem_request_sha256,
            "t": result.problem.t,
            "u": result.problem.u,
        },
        "route": {
            "contract_sha256": result.route.contract_sha256,
            "profile": result.route.profile,
            "trust_policy": result.route.execution.trust_policy,
        },
        "schema": "ra_adapt_singleton_trajectory_fixture_v1",
    }


def test_historical_singleton_lock_is_preserved_and_canonical_contract_holds() -> None:
    assert hashlib.sha256(_FIXTURE.read_bytes()).hexdigest() == (
        "722ff9e3503c46a577d18ef9206b5914b8ad7a5b965a6510e42e69ed645ac220"
    )
    expected = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert expected["schema"] == "ra_adapt_singleton_trajectory_fixture_v1"
    assert expected["accepted_trajectory"][0]["operators"] == [
        (
            "uccsd_ferm_lifted::uccsd_sing(alpha:0->1)"
            "::child_set[0]::legal_projected"
        )
    ]
    assert expected["trust_receipts"][0]["transaction_failure"] == (
        "ValueError:Projected trust transaction is missing square G_AA_raw."
    )
    assert expected["trust_receipts"][0]["update_reason"] == (
        "source_metric_transaction_failure_hold"
    )

    result = _execute_sr_snake(
        _problem(),
        SRRunRequest(
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=3)
            )
        ),
    ).result
    observed = _project(result)

    # Post-split selection is global over retained parent identities.  It does
    # not consult the inherited physical-family lane, yet preserves the locked
    # singleton trajectory, refit, and executed-work characterization.
    expected_trajectory = expected["accepted_trajectory"]
    observed_trajectory = observed["accepted_trajectory"]
    assert len(observed_trajectory) == len(expected_trajectory)
    for actual, locked in zip(
        observed_trajectory, expected_trajectory, strict=True
    ):
        assert actual["controller_round"] == locked["controller_round"]
        assert actual["operators"] == locked["operators"]
        assert actual["insertion_positions"] == locked["insertion_positions"]
        assert actual["energy"] == pytest.approx(
            locked["energy"], rel=0.0, abs=2.0e-12
        )

    observed_trust = observed["trust_receipts"]
    assert len(observed_trust) == 3
    assert all(
        receipt["policy"]
        == "source_metric_inverse_sqrt_no_overlap_v1"
        for receipt in observed_trust
    )
    assert all(
        receipt["transaction_complete"] is True
        and receipt["endpoint_overlap_query_charge"] == 0
        and receipt["supported_metric_whitening_active"] is False
        and receipt["supported_metric_inverse_sqrt_constructed"] is False
        and receipt["supported_rank"] >= 1
        for receipt in observed_trust
    )

    assert observed["accepted_refit_receipts"] == expected[
        "accepted_refit_receipts"
    ]
    assert observed["problem"] == expected["problem"]
    # Preserve the historical source-locked fixture while requiring the
    # current cumulative-relative plateau contract to carry its new digest.
    assert observed["route"]["profile"] == expected["route"]["profile"]
    assert observed["route"]["trust_policy"] == expected["route"][
        "trust_policy"
    ]
    assert expected["route"]["contract_sha256"] == (
        "eecd11fccf8f34b5e89042fb75949b54546a297ea5fe4f9969ad91ec569ff08a"
    )
    assert observed["route"]["contract_sha256"] == (
        "9d90a88a353f3adcc9373a223c1523564b9cd1c49712232db74e8f63895c8057"
    )
    assert observed["schema"] == expected["schema"]

    accounting = observed["estimator_accounting"]
    assert accounting == expected["estimator_accounting"]
    assert accounting["s_alg"] == sum(accounting["components"].values())
    assert accounting["prefix_s_alg"] == [
        sum(row.values()) for row in accounting["prefix_components"]
    ]
