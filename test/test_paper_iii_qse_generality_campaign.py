"""Locks for the fresh Peierls/SIAM Paper-III acquisition campaign."""

from __future__ import annotations

from pipelines.exact_bench.paper_iii_qse_generality_campaign import run_campaign
from pipelines.qse_spectra.peierls_static_adapter import (
    PEIERLS_POOL_POLICY_ID,
    PeierlsStaticAdapterError,
    load_peierls_problem,
)
from pipelines.qse_spectra.selection_policy import (
    PAPER_III_LOCKED_SELECTOR_POLICY_ID,
)


def test_peierls_problem_adapter_is_deterministic_and_paper_facing() -> None:
    first = load_peierls_problem("peierls_weak_weak", target_roots=6)
    second = load_peierls_problem("peierls_weak_weak", target_roots=6)
    assert first.pool_digest == second.pool_digest
    assert first.pool_policy_id == PEIERLS_POOL_POLICY_ID
    assert len(first.basis) == len(first.costs) == len(first.resources) == 53
    assert len(first.references) == 6
    assert first.arm_receipt()["uses_exact_excited_state_information_online"] is False

    try:
        load_peierls_problem("not_a_regime")
    except PeierlsStaticAdapterError:
        pass
    else:  # pragma: no cover - fail-closed assertion
        raise AssertionError("undeclared Peierls regime was accepted")


def test_one_case_campaign_uses_locked_routes_and_emits_resource_axes() -> None:
    payload = run_campaign(
        ["peierls_weak_weak"],
        trace_stride=100,
        include_qiskit=False,
        verify_determinism=True,
    )
    assert payload["method_contract"]["selector_policy_id"] == (
        PAPER_III_LOCKED_SELECTOR_POLICY_ID
    )
    case = payload["cases"]["peierls_weak_weak"]
    assert case["selector_replay_lock"] == {
        "selected_support_identical": True,
        "stop_reason_identical": True,
        "final_residual_identical": True,
    }
    assert set(case["routes"]) == {"selected_qse", "direct_davidson", "fixed_qse"}
    for route in case["routes"].values():
        assert route["resources"]["qiskit_compiled"]["status"] == "NOT_REQUESTED"
        assert route["resources"]["estimator"]["queries"] > 0
        assert route["static_spectral_audit"]["status"] == "EVALUATED"
    assert "terminal_ritz_vectors" not in case["routes"]["direct_davidson"]["completion"]
