from __future__ import annotations

import copy

import numpy as np
import pytest

from pipelines.qse_spectra.pool_profiles import (
    HH_POOL_ENCODED_BOSON_COMPLETION_V1,
)
from pipelines.qse_spectra.selection_policy import (
    PAPER_III_LOCKED_SELECTOR_POLICY_ID,
)
from pipelines.reporting.build_paper_iii_static_production_report import (
    StaticProductionReportError,
    build_markdown,
    normalized_l1,
    validate_acquisition_payload,
)


def _crossings() -> list[dict[str, object]]:
    return [
        {"error_target": 1.0e-4, "status": "REACHED", "n2q": 10.0},
        {"error_target": 1.0e-6, "status": "REACHED", "n2q": 20.0},
    ]


def _acquisition_payload() -> dict[str, object]:
    rows = []
    for index in range(12):
        rows.append(
            {
                "case": {"case_key": f"case_{index}"},
                "selected_qse_by_pool_policy": {
                    HH_POOL_ENCODED_BOSON_COMPLETION_V1: {
                        "algorithm_status": "TARGET_WINDOW_CERTIFIED",
                        "pool_policy_id": HH_POOL_ENCODED_BOSON_COMPLETION_V1,
                        "crossings": _crossings(),
                    }
                },
                "direct_davidson": {
                    "algorithm_status": "DAVIDSON_RESIDUAL_CERTIFIED",
                    "crossings": _crossings(),
                },
            }
        )
    return {
        "schema_version": "paper_iii_hh_pool_profile_ablation_v1",
        "selector_policy_id": PAPER_III_LOCKED_SELECTOR_POLICY_ID,
        "controller_boundary": {
            "implicit_fallback_enabled": False,
            "hybrid_enabled": False,
        },
        "lock_gate": {
            "error_target": 1.0e-4,
            "coverage_qualified_pool_policy_ids": [
                HH_POOL_ENCODED_BOSON_COMPLETION_V1
            ],
        },
        "rows": rows,
    }


def test_validate_acquisition_accepts_only_locked_completion_profile() -> None:
    payload = _acquisition_payload()
    assert (
        validate_acquisition_payload(payload)
        == HH_POOL_ENCODED_BOSON_COMPLETION_V1
    )

    drift = copy.deepcopy(payload)
    drift["controller_boundary"]["implicit_fallback_enabled"] = True
    with pytest.raises(StaticProductionReportError, match="fallback/hybrid"):
        validate_acquisition_payload(drift)


def test_validate_acquisition_rejects_one_uncertified_cell() -> None:
    payload = _acquisition_payload()
    payload["rows"][3]["selected_qse_by_pool_policy"][
        HH_POOL_ENCODED_BOSON_COMPLETION_V1
    ]["algorithm_status"] = "POOL_EXHAUSTED"
    with pytest.raises(StaticProductionReportError, match="uncertified"):
        validate_acquisition_payload(payload)


def test_normalized_l1_uses_integrated_reference_weight() -> None:
    grid = np.linspace(0.0, 1.0, 5)
    reference = np.ones(5)
    candidate = np.full(5, 1.25)
    assert normalized_l1(reference, candidate, grid) == pytest.approx(0.25)


def test_markdown_declares_static_only_and_no_fallback() -> None:
    payload = {
        "summary": {
            "case_count": 1,
            "selected_online_certified_count": 1,
            "selected_not_more_expensive_than_direct_at_1e-4": 1,
            "selected_not_more_expensive_than_direct_at_1e-6": 1,
        },
        "rows": [
            {
                "case": {"case_key": "weak_weak_nph1"},
                "crossings": {
                    "1e-4": {"selected_n2q": 10.0, "direct_n2q": 20.0},
                    "1e-6": {"selected_n2q": 12.0, "direct_n2q": 24.0},
                },
                "selected_terminal": {
                    "n2q": 12.0,
                    "max_root_energy_error": 1.0e-7,
                },
                "fixed_linear_response": {"max_root_energy_error": 1.0e-2},
                "transition_observables": {
                    name: {
                        "selected": {"normalized_spectral_l1": 0.1},
                        "fixed_linear_response": {
                            "normalized_spectral_l1": 0.4
                        },
                    }
                    for name in ("site0_density", "site0_displacement")
                },
            }
        ],
    }
    text = build_markdown(payload)
    assert "static and frequency-domain QSE only" in text
    assert "No selected-QSE-to-Davidson fallback" in text
    assert "time-dynamics claim" in text
