from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.qse_spectra.core import (
    QSEBasisVectorPolicy,
    computational_basis_state,
    compute_qse_spectra,
    pauli_string_basis_element,
    polynomial_basis_element,
)
from pipelines.qse_spectra.record_selection import (
    GEOMETRY_SCORE_FULL_PENCIL_KY_FAN,
    GEOMETRY_SCORE_RAW_OVERLAP_RITZ_GAIN,
    GEOMETRY_SCORE_RESIDUALIZED_RITZ_GAIN,
    GEOMETRY_SCORE_RESIDUALIZED_RITZ_GAIN_KY_FAN_SHORTLIST,
    GEOMETRY_SCORE_NORMALIZED_ANGULAR,
    StaticRecordSelectionConfig,
    extract_static_record_candidate,
    finalize_static_record_selection_payload,
    select_static_qse_records,
    static_record_selection_payload,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


def _poly(nq: int, terms: list[tuple[str, complex]]) -> PauliPolynomial:
    out = PauliPolynomial("JW")
    for label, coeff in terms:
        out.add_term(PauliTerm(int(nq), ps=str(label), pc=complex(coeff)))
    return out


def test_input_order_selection_is_deterministic_and_preserves_metadata() -> None:
    basis = [
        pauli_string_basis_element("zz", nq=2, name="first", metadata={"source": "unit", "rank": 0}),
        pauli_string_basis_element("ex", nq=2, name="second", metadata={"source": "unit", "rank": 1}),
        pauli_string_basis_element("xe", nq=2, name="third", metadata={"source": "unit", "rank": 2}),
    ]

    result = select_static_qse_records(
        basis,
        config=StaticRecordSelectionConfig(mode="input_order", max_records=2),
    )
    payload = static_record_selection_payload(result)

    assert result.selected_original_indices == (0, 1)
    assert [element.name for element in result.selected_basis_elements] == ["first", "second"]
    assert payload["summary"]["input_basis_size"] == 3
    assert payload["summary"]["selected_basis_size"] == 2
    assert payload["candidates"][0]["metadata"] == {"source": "unit", "rank": 0}
    assert payload["rejected_records"] == [
        {"original_basis_index": 2, "rejection_reasons": ["rank_limit"], "eligible": True}
    ]


def test_cost_proxy_prefers_lower_operator_cost_and_breaks_ties_by_input_order() -> None:
    expensive = polynomial_basis_element(
        _poly(2, [("xx", 1.0), ("zz", 1.0), ("ez", 1.0)]),
        name="expensive_first",
    )
    basis = [
        expensive,
        pauli_string_basis_element("xe", nq=2, name="tie_a"),
        pauli_string_basis_element("ex", nq=2, name="tie_b"),
    ]

    result = select_static_qse_records(
        basis,
        config=StaticRecordSelectionConfig(mode="cost_proxy", max_records=2),
    )

    assert result.selected_original_indices == (1, 2)
    assert [element.name for element in result.selected_basis_elements] == ["tie_a", "tie_b"]
    assert result.candidates[1].cost_proxy == pytest.approx(result.candidates[2].cost_proxy)
    assert result.candidates[0].cost_proxy > result.candidates[1].cost_proxy


def test_geometry_selected_qse_admits_metric_residual_direction() -> None:
    hamiltonian = _poly(1, [("z", 1.0), ("x", 0.5)])
    psi = computational_basis_state(1, "0")
    basis = [
        pauli_string_basis_element("I", nq=1, name="identity"),
        pauli_string_basis_element("Z", nq=1, name="parallel_z"),
        pauli_string_basis_element("X", nq=1, name="residual_flip"),
    ]

    result = select_static_qse_records(
        basis,
        config=StaticRecordSelectionConfig(mode="geometry_selected", max_records=1),
        hamiltonian=hamiltonian,
        prepared_state=psi,
        basis_vector_policy=QSEBasisVectorPolicy(
            reference_projection="q0",
            basis_vector_normalization="raw_projected",
        ),
    )
    payload = static_record_selection_payload(result)

    assert result.selected_original_indices == (2,)
    assert [element.name for element in result.selected_basis_elements] == ["residual_flip"]
    selected = payload["selected_records"][0]
    assert selected["name"] == "residual_flip"
    assert selected["selection_score"] > 0.0
    candidate_payload = payload["candidates"][2]
    assert candidate_payload["geometry"]["metric_novelty_fraction"] == pytest.approx(1.0)
    assert candidate_payload["geometry"]["residual_capture"] > 0.0
    rejected = {row["original_basis_index"]: row["rejection_reasons"] for row in payload["rejected_records"]}
    assert rejected[0] == ["metric_novelty_floor"]
    assert rejected[1] == ["metric_novelty_floor"]


def test_geometry_score_interface_defaults_to_the_locked_angular_selector() -> None:
    hamiltonian = _poly(1, [("z", 1.0), ("x", 0.5)])
    psi = computational_basis_state(1, "0")
    basis = [
        pauli_string_basis_element("X", nq=1, name="flip"),
        pauli_string_basis_element("Y", nq=1, name="phase_flip"),
    ]
    context = {
        "hamiltonian": hamiltonian,
        "prepared_state": psi,
        "basis_vector_policy": QSEBasisVectorPolicy(
            reference_projection="q0",
            basis_vector_normalization="raw_projected",
        ),
        "compiled_costs": (1.0, 1.0),
    }

    implicit = select_static_qse_records(
        basis,
        config=StaticRecordSelectionConfig(mode="geometry_selected", max_records=2),
        **context,
    )
    explicit = select_static_qse_records(
        basis,
        config=StaticRecordSelectionConfig(
            mode="geometry_selected",
            max_records=2,
            geometry_score_mode=GEOMETRY_SCORE_NORMALIZED_ANGULAR,
            geometry_additive_novelty_reward=True,
        ),
        **context,
    )

    assert explicit.selected_original_indices == implicit.selected_original_indices
    assert explicit.geometry_stop == implicit.geometry_stop
    assert explicit.config.geometry_min_metric_novelty == pytest.approx(1.0e-12)

    with pytest.raises(ValueError, match="geometry_score_mode"):
        StaticRecordSelectionConfig(
            mode="geometry_selected",
            max_records=1,
            geometry_score_mode="not-a-score",
        )
    with pytest.raises(ValueError, match="geometry_additive_novelty_reward"):
        StaticRecordSelectionConfig(
            mode="geometry_selected",
            max_records=1,
            geometry_additive_novelty_reward=1,
        )


def test_disabling_additive_novelty_reward_preserves_the_novelty_floor() -> None:
    hamiltonian = _poly(1, [("z", 1.0), ("x", 0.5)])
    psi = computational_basis_state(1, "0")
    basis = [
        pauli_string_basis_element("I", nq=1, name="dependent"),
        pauli_string_basis_element("X", nq=1, name="novel"),
    ]
    context = {
        "hamiltonian": hamiltonian,
        "prepared_state": psi,
        "basis_vector_policy": QSEBasisVectorPolicy(
            reference_projection="q0",
            basis_vector_normalization="raw_projected",
        ),
        "compiled_costs": (1.0, 1.0),
    }
    with_reward = select_static_qse_records(
        basis,
        config=StaticRecordSelectionConfig(
            mode="geometry_selected",
            max_records=1,
            geometry_additive_novelty_reward=True,
        ),
        **context,
    )
    without_reward = select_static_qse_records(
        basis,
        config=StaticRecordSelectionConfig(
            mode="geometry_selected",
            max_records=1,
            geometry_additive_novelty_reward=False,
        ),
        **context,
    )

    assert without_reward.selected_original_indices == (1,)
    rejected = {
        int(row["original_basis_index"]): list(row["rejection_reasons"])
        for row in static_record_selection_payload(without_reward)["rejected_records"]
    }
    assert rejected[0] == ["metric_novelty_floor"]
    on_score = with_reward.candidate_decisions[1]["selection_score"]
    off_score = without_reward.candidate_decisions[1]["selection_score"]
    assert on_score - off_score == pytest.approx(0.25)


def test_supported_image_admissibility_rejects_roundoff_directions_at_the_solver_floor() -> None:
    nearly_projected_out = polynomial_basis_element(
        _poly(1, [("e", 1.0), ("x", 1.0e-8)]),
        name="roundoff_direction",
    )
    supported = pauli_string_basis_element("x", nq=1, name="supported_direction")

    result = select_static_qse_records(
        [nearly_projected_out, supported],
        config=StaticRecordSelectionConfig(
            mode="geometry_selected",
            max_records=1,
            geometry_target_roots=1,
            geometry_score_mode=GEOMETRY_SCORE_NORMALIZED_ANGULAR,
            geometry_require_supported_image=True,
        ),
        hamiltonian=_poly(1, [("z", 1.0)]),
        prepared_state=computational_basis_state(1, "0"),
        basis_vector_policy=QSEBasisVectorPolicy(
            reference_projection="q0",
            basis_vector_normalization="raw_projected",
        ),
    )

    assert result.selected_original_indices == (1,)
    assert "supported_image_floor" in result.candidate_decisions[0]["rejection_reasons"]


@pytest.mark.parametrize(
    "score_mode",
    [
        GEOMETRY_SCORE_RAW_OVERLAP_RITZ_GAIN,
        GEOMETRY_SCORE_RESIDUALIZED_RITZ_GAIN,
    ],
)
def test_two_dimensional_ritz_scores_report_the_predicted_energy_gain(
    score_mode: str,
) -> None:
    # In {|0>, |1>}, H = [[0, g], [g, d]].  Both the overlap-aware raw
    # pencil and the full-frame-residualized pencil therefore predict the
    # literal lowering (sqrt(d^2 + 4g^2) - d) / 2.
    d = 4.0
    g = 0.5
    hamiltonian = _poly(
        1,
        [("e", d / 2.0), ("z", -d / 2.0), ("x", g)],
    )
    result = select_static_qse_records(
        [pauli_string_basis_element("X", nq=1, name="candidate")],
        config=StaticRecordSelectionConfig(
            mode="geometry_selected",
            max_records=1,
            geometry_target_roots=1,
            geometry_score_mode=score_mode,
            geometry_additive_novelty_reward=False,
            geometry_cost_discount_alpha=1.0,
        ),
        hamiltonian=hamiltonian,
        prepared_state=computational_basis_state(1, "0"),
        basis_vector_policy=QSEBasisVectorPolicy(
            reference_projection="q0",
            basis_vector_normalization="raw_projected",
        ),
        compiled_costs=(1.0,),
    )

    expected = (math.sqrt(d * d + 4.0 * g * g) - d) / 2.0
    geometry = result.candidate_decisions[0]["geometry"]
    assert geometry["score_mode"] == score_mode
    assert geometry["spectral_score_raw"] == pytest.approx(expected)
    assert geometry["spectral_score_normalized"] == pytest.approx(1.0)


def test_full_pencil_ky_fan_oracle_uses_a_common_rank_filling_ceiling() -> None:
    # Before the retained frame reaches R, every novel singleton raises its
    # rank by one.  Padding missing roots by the same round-local ceiling makes
    # the full proposed-pencil comparison select the lower Rayleigh direction
    # without using exact-reference data.
    low = 1.0
    high = 10.0
    hamiltonian = _poly(
        2,
        [
            ("ee", (low + high) / 2.0),
            ("ze", -low / 2.0),
            ("ez", -high / 2.0),
        ],
    )
    result = select_static_qse_records(
        [
            pauli_string_basis_element("xe", nq=2, name="low_rayleigh"),
            pauli_string_basis_element("ex", nq=2, name="high_rayleigh"),
        ],
        config=StaticRecordSelectionConfig(
            mode="geometry_selected",
            max_records=1,
            geometry_target_roots=1,
            geometry_score_mode=GEOMETRY_SCORE_FULL_PENCIL_KY_FAN,
            geometry_additive_novelty_reward=False,
            geometry_cost_discount_alpha=1.0,
        ),
        hamiltonian=hamiltonian,
        prepared_state=computational_basis_state(2, "00"),
        basis_vector_policy=QSEBasisVectorPolicy(
            reference_projection="q0",
            basis_vector_normalization="raw_projected",
        ),
        compiled_costs=(1.0, 1.0),
    )

    assert result.selected_original_indices == (0,)
    low_geometry = result.candidate_decisions[0]["geometry"]
    high_geometry = result.candidate_decisions[1]["geometry"]
    assert low_geometry["ky_fan_bootstrap"] is True
    assert low_geometry["spectral_score_raw"] == pytest.approx(high - low)
    assert high_geometry["spectral_score_raw"] == pytest.approx(0.0)


def test_residualized_shortlist_matches_full_ky_fan_when_it_covers_the_pool() -> None:
    hamiltonian = _poly(
        2,
        [("ee", 5.5), ("ze", -0.5), ("ez", -5.0)],
    )
    basis = [
        pauli_string_basis_element("xe", nq=2, name="low_rayleigh"),
        pauli_string_basis_element("ex", nq=2, name="high_rayleigh"),
    ]
    context = {
        "hamiltonian": hamiltonian,
        "prepared_state": computational_basis_state(2, "00"),
        "basis_vector_policy": QSEBasisVectorPolicy(
            reference_projection="q0",
            basis_vector_normalization="raw_projected",
        ),
        "compiled_costs": (1.0, 1.0),
    }
    common = dict(
        mode="geometry_selected",
        max_records=2,
        geometry_target_roots=1,
        geometry_additive_novelty_reward=False,
        geometry_cost_discount_alpha=1.0,
        geometry_require_supported_image=True,
    )
    oracle = select_static_qse_records(
        basis,
        config=StaticRecordSelectionConfig(
            **common,
            geometry_score_mode=GEOMETRY_SCORE_FULL_PENCIL_KY_FAN,
        ),
        **context,
    )
    shortlist = select_static_qse_records(
        basis,
        config=StaticRecordSelectionConfig(
            **common,
            geometry_score_mode=GEOMETRY_SCORE_RESIDUALIZED_RITZ_GAIN_KY_FAN_SHORTLIST,
            geometry_ky_fan_shortlist_size=2,
        ),
        **context,
    )

    assert shortlist.selected_original_indices == oracle.selected_original_indices
    rounds = shortlist.geometry_stop["round_trace"]
    assert [row["screening"]["full_proposed_pencil_evaluations"] for row in rounds] == [2, 1]
    assert [row["screening"]["two_dimensional_pencil_evaluations"] for row in rounds] == [2, 1]


def test_pauli_string_and_polynomial_feature_extraction() -> None:
    pauli = extract_static_record_candidate(
        pauli_string_basis_element("xe", nq=2, name="x_on_q1"),
        original_basis_index=0,
    )
    poly = extract_static_record_candidate(
        polynomial_basis_element(
            _poly(2, [("xx", 2.0), ("ze", -0.5j)]),
            name="mixed_poly",
            metadata={"full_meta_class": "unit_test"},
        ),
        original_basis_index=1,
    )

    assert pauli.term_count == 1
    assert pauli.max_pauli_weight == 1
    assert pauli.mean_pauli_weight == pytest.approx(1.0)
    assert pauli.support_qubit_count == 1
    assert pauli.coefficient_l1 == pytest.approx(1.0)

    assert poly.nq == 2
    assert poly.term_count == 2
    assert poly.max_pauli_weight == 2
    assert poly.mean_pauli_weight == pytest.approx(1.5)
    assert poly.support_qubit_count == 2
    assert poly.coefficient_l1 == pytest.approx(2.5)
    assert poly.metadata == {"full_meta_class": "unit_test"}


def test_invalid_static_record_selection_configs_raise_clear_errors() -> None:
    with pytest.raises(ValueError, match="mode"):
        StaticRecordSelectionConfig(mode="spectral_gain", max_records=1)
    with pytest.raises(ValueError, match="max_records"):
        StaticRecordSelectionConfig(mode="input_order", max_records=0)
    with pytest.raises(ValueError, match="max_term_count"):
        StaticRecordSelectionConfig(mode="input_order", max_records=1, max_term_count=0)
    with pytest.raises(ValueError, match="max_pauli_weight"):
        StaticRecordSelectionConfig(mode="input_order", max_records=1, max_pauli_weight=-1)
    with pytest.raises(ValueError, match="min_retained_rank"):
        StaticRecordSelectionConfig(mode="input_order", max_records=1, min_retained_rank=-1)
    with pytest.raises(ValueError, match="max_overlap_condition"):
        StaticRecordSelectionConfig(mode="input_order", max_records=1, max_overlap_condition=float("inf"))


def test_hard_caps_reject_candidates_before_rank_limit() -> None:
    basis = [
        polynomial_basis_element(_poly(2, [("xx", 1.0), ("zz", 1.0)]), name="two_terms"),
        pauli_string_basis_element("xx", nq=2, name="too_heavy"),
        pauli_string_basis_element("ex", nq=2, name="kept"),
    ]

    result = select_static_qse_records(
        basis,
        config=StaticRecordSelectionConfig(
            mode="input_order",
            max_records=3,
            max_term_count=1,
            max_pauli_weight=1,
        ),
    )
    payload = static_record_selection_payload(result)

    assert result.selected_original_indices == (2,)
    assert payload["summary"]["eligible_candidate_count"] == 1
    assert payload["summary"]["hard_rejected_count"] == 2
    rejected = {row["original_basis_index"]: row["rejection_reasons"] for row in payload["rejected_records"]}
    assert rejected[0] == ["max_term_count", "max_pauli_weight"]
    assert rejected[1] == ["max_pauli_weight"]


def test_post_qse_guard_finalization_is_diagnostic_only_and_preserves_selection() -> None:
    hamiltonian = _poly(1, [("z", 1.0)])
    psi = computational_basis_state(1, "0")
    basis = [
        pauli_string_basis_element("I", nq=1, name="identity"),
        pauli_string_basis_element("Z", nq=1, name="parallel"),
        pauli_string_basis_element("X", nq=1, name="flip"),
    ]
    selection = select_static_qse_records(
        basis,
        config=StaticRecordSelectionConfig(
            mode="input_order",
            max_records=3,
            min_retained_rank=2,
            max_overlap_condition=0.5,
        ),
    )
    result = compute_qse_spectra(
        hamiltonian,
        psi,
        selection.selected_basis_elements,
        basis_vector_policy=QSEBasisVectorPolicy(
            reference_projection="q0",
            basis_vector_normalization="raw_projected",
        ),
    )

    payload = finalize_static_record_selection_payload(selection, result)

    assert selection.selected_original_indices == (0, 1, 2)
    assert payload["selected_original_basis_indices"] == [0, 1, 2]
    post = payload["post_qse_diagnostics"]
    assert post["retained_rank"] == 1
    assert post["discarded_rank"] == 2
    assert post["basis_vector_zero_count"] == 2
    assert post["basis_vector_projected_out_by_q0_count"] == 2
    assert post["basis_vector_diagnostics_available"] is True
    guards = post["guards"]
    assert guards["min_retained_rank"]["passed"] is False
    assert guards["max_overlap_condition"]["passed"] is False
    assert guards["all_configured_guards_passed"] is False


def test_selection_retaining_zero_candidates_is_rejected() -> None:
    basis = [pauli_string_basis_element("xx", nq=2, name="too_heavy")]

    with pytest.raises(ValueError, match="retained zero"):
        select_static_qse_records(
            basis,
            config=StaticRecordSelectionConfig(mode="input_order", max_records=1, max_pauli_weight=1),
        )
