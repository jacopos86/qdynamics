from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_continuation_generators import (
    build_generator_metadata,
    build_pool_generator_registry,
    build_runtime_split_child_sets,
    build_runtime_split_children,
    build_split_event,
    rebuild_polynomial_from_serialized_terms,
)
from pipelines.hardcoded.hh_continuation_symmetry import build_symmetry_spec
from src.quantum.pauli_polynomial_class import (
    PauliPolynomial,
    fermion_minus_operator,
    fermion_plus_operator,
)
from src.quantum.pauli_words import PauliTerm


def _term(label: str, poly: PauliPolynomial):
    return type("_DummyAnsatzTerm", (), {"label": str(label), "polynomial": poly})()


def _macro_poly() -> PauliPolynomial:
    return PauliPolynomial(
        "JW",
        [
            PauliTerm(6, ps="eyeexy", pc=1.0),
            PauliTerm(6, ps="eyeeyx", pc=-1.0),
        ],
    )


def _number_preserving_macro_poly() -> PauliPolynomial:
    return (-1j) * (
        fermion_plus_operator("JW", 4, 1) * fermion_minus_operator("JW", 4, 0)
        - fermion_plus_operator("JW", 4, 0) * fermion_minus_operator("JW", 4, 1)
    )


def _mixed_macro_poly() -> PauliPolynomial:
    return _number_preserving_macro_poly() + PauliPolynomial(
        "JW",
        [PauliTerm(4, ps="zeee", pc=0.25)],
    )


def test_build_generator_metadata_is_stable_for_same_structure() -> None:
    sym = build_symmetry_spec(family_id="paop_lf_std", mitigation_mode="verify_only")
    first = build_generator_metadata(
        label="cand",
        polynomial=_macro_poly(),
        family_id="paop_lf_std",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_spec=sym.__dict__,
    )
    second = build_generator_metadata(
        label="cand",
        polynomial=_macro_poly(),
        family_id="paop_lf_std",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_spec=sym.__dict__,
    )
    assert first.generator_id == second.generator_id
    assert first.template_id == second.template_id
    assert first.support_site_offsets == [0, 1]
    assert first.is_macro_generator is True


def test_pool_registry_carries_symmetry_metadata() -> None:
    sym = build_symmetry_spec(family_id="paop_lf_std", mitigation_mode="verify_only")
    registry = build_pool_generator_registry(
        terms=[_term("macro", _macro_poly())],
        family_ids=["paop_lf_std"],
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_specs=[sym.__dict__],
    )
    meta = registry["macro"]
    assert meta["family_id"] == "paop_lf_std"
    assert meta["is_macro_generator"] is True
    assert meta["symmetry_spec"]["mitigation_eligible"] is True
    assert meta["symmetry_spec"]["particle_number_mode"] == "preserving"
    assert meta["compile_metadata"]["symmetry_gate"]["passed"] is True
    assert "operator_symmetry_checked" in meta["symmetry_spec"]["tags"]


def test_build_generator_metadata_hard_guards_base_terms_that_break_required_symmetry() -> None:
    sym = build_symmetry_spec(family_id="uccsd", mitigation_mode="verify_only")
    bad_term = _number_preserving_macro_poly().return_polynomial()[0]
    bad_poly = PauliPolynomial("JW", [bad_term])
    meta = build_generator_metadata(
        label="bad_base_term",
        polynomial=bad_poly,
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_spec=sym.__dict__,
    )
    assert meta.symmetry_spec is not None
    assert meta.symmetry_spec["particle_number_mode"] == "violating"
    assert meta.symmetry_spec["spin_sector_mode"] == "violating"
    assert meta.symmetry_spec["hard_guard"] is True
    assert "operator_symmetry_checked" in meta.symmetry_spec["tags"]
    assert "operator_symmetry_rejected" in meta.symmetry_spec["tags"]
    assert meta.compile_metadata["symmetry_intent"]["particle_number_mode"] == "preserving"
    assert meta.compile_metadata["symmetry_gate"]["passed"] is False


def test_deliberate_split_marks_child_metadata() -> None:
    meta = build_generator_metadata(
        label="child",
        polynomial=_macro_poly(),
        family_id="paop_lf_std",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        split_policy="deliberate_split",
        parent_generator_id="gen:parent",
    )
    assert meta.is_macro_generator is False
    assert meta.parent_generator_id == "gen:parent"
    assert meta.split_policy == "deliberate_split"


def test_build_split_event_keeps_parent_child_provenance() -> None:
    event = build_split_event(
        parent_generator_id="gen:parent",
        child_generator_ids=["gen:c1", "gen:c2"],
        reason="compiled_depth_cap",
        split_mode="selective",
    )
    assert event["parent_generator_id"] == "gen:parent"
    assert event["child_generator_ids"] == ["gen:c1", "gen:c2"]
    assert event["reason"] == "compiled_depth_cap"


def test_build_runtime_split_children_marks_atomic_terms_that_break_required_symmetry() -> None:
    sym = build_symmetry_spec(family_id="uccsd", mitigation_mode="verify_only")
    parent_meta = build_generator_metadata(
        label="macro",
        polynomial=_number_preserving_macro_poly(),
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_spec=sym.__dict__,
    )
    children = build_runtime_split_children(
        parent_label="macro",
        polynomial=_number_preserving_macro_poly(),
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        split_mode="shortlist_pauli_children_v1",
        parent_generator_metadata=parent_meta.__dict__,
        symmetry_spec=sym.__dict__,
    )
    assert len(children) == 2
    assert children[0]["child_label"].startswith("macro::split[0]::")
    assert children[1]["child_label"].startswith("macro::split[1]::")
    for idx, child in enumerate(children):
        meta = child["child_generator_metadata"]
        compile_meta = meta["compile_metadata"]
        assert meta["parent_generator_id"] == parent_meta.generator_id
        assert meta["split_policy"] == "deliberate_split"
        assert meta["is_macro_generator"] is False
        assert compile_meta["runtime_split"]["mode"] == "shortlist_pauli_children_v1"
        assert compile_meta["runtime_split"]["parent_label"] == "macro"
        assert compile_meta["runtime_split"]["child_index"] == idx
        assert compile_meta["runtime_split"]["child_count"] == 2
        assert compile_meta["runtime_split"]["representation"] == "child_atom"
        assert compile_meta["runtime_split"]["symmetry_gate"]["passed"] is False
        assert meta["symmetry_spec"]["particle_number_mode"] == "violating"
        assert meta["symmetry_spec"]["hard_guard"] is True
        assert len(compile_meta["serialized_terms_exyz"]) == 1


def test_build_runtime_split_child_sets_only_returns_symmetry_safe_combinations() -> None:
    sym = build_symmetry_spec(family_id="uccsd", mitigation_mode="verify_only")
    parent_meta = build_generator_metadata(
        label="macro",
        polynomial=_mixed_macro_poly(),
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_spec=sym.__dict__,
    )
    children = build_runtime_split_children(
        parent_label="macro",
        polynomial=_mixed_macro_poly(),
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        split_mode="shortlist_pauli_children_v1",
        parent_generator_metadata=parent_meta.__dict__,
        symmetry_spec=sym.__dict__,
    )
    child_sets = build_runtime_split_child_sets(
        parent_label="macro",
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        split_mode="shortlist_pauli_children_v1",
        children=children,
        parent_generator_metadata=parent_meta.__dict__,
        symmetry_spec=sym.__dict__,
        max_subset_size=3,
    )
    labels = {row["candidate_label"] for row in child_sets}
    assert labels == {"macro::child_set[0]", "macro::child_set[1,2]"}
    by_label = {row["candidate_label"]: row for row in child_sets}
    pair_meta = by_label["macro::child_set[1,2]"]["candidate_generator_metadata"]["compile_metadata"]
    singleton_meta = by_label["macro::child_set[0]"]["candidate_generator_metadata"]["compile_metadata"]
    assert pair_meta["runtime_split"]["representation"] == "child_set"
    assert pair_meta["runtime_split"]["child_indices"] == [1, 2]
    assert pair_meta["runtime_split"]["symmetry_gate"]["passed"] is True
    assert singleton_meta["runtime_split"]["child_indices"] == [0]
    assert by_label["macro::child_set[1,2]"]["candidate_generator_metadata"]["symmetry_spec"]["particle_number_mode"] == "preserving"
    assert len(pair_meta["serialized_terms_exyz"]) == 2


def test_build_split_event_records_probe_choice_details() -> None:
    event = build_split_event(
        parent_generator_id="gen:parent",
        child_generator_ids=["gen:c1", "gen:c2"],
        reason="depth4_shortlist_probe",
        split_mode="shortlist_pauli_children_v1",
        probe_trigger="phase2_shortlist",
        choice_reason="parent_actual_score_better",
        parent_score=1.25,
        child_scores={"c1": 0.8, "c2": 0.7},
        admissible_child_subsets=[["c1", "c2"]],
        chosen_representation="parent",
        chosen_child_ids=[],
        split_margin=-0.1,
        symmetry_gate_results={"passed": True},
        compiled_cost_parent=2.0,
        compiled_cost_children=2.4,
        insertion_positions=[3],
    )
    assert event["probe_trigger"] == "phase2_shortlist"
    assert event["choice_reason"] == "parent_actual_score_better"
    assert event["child_scores"] == {"c1": 0.8, "c2": 0.7}
    assert event["admissible_child_subsets"] == [["c1", "c2"]]
    assert event["chosen_representation"] == "parent"
    assert event["compiled_cost_parent"] == 2.0
    assert event["insertion_positions"] == [3]


def test_rebuild_polynomial_from_serialized_terms_preserves_serialized_order() -> None:
    poly = rebuild_polynomial_from_serialized_terms(
        [
            {"pauli_exyz": "eyezee", "coeff_re": 1.0, "coeff_im": 0.0, "nq": 6},
            {"pauli_exyz": "eyeeez", "coeff_re": -1.0, "coeff_im": 0.0, "nq": 6},
        ]
    )
    assert [term.pw2strng() for term in poly.return_polynomial()] == ["eyezee", "eyeeez"]


def test_rebuild_polynomial_from_serialized_terms_respects_custom_drop_tolerance() -> None:
    poly = rebuild_polynomial_from_serialized_terms(
        [
            {"pauli_exyz": "eyezee", "coeff_re": 1.0e-8, "coeff_im": 0.0, "nq": 6},
        ],
        drop_abs_tol=1.0e-12,
    )
    assert [term.pw2strng() for term in poly.return_polynomial()] == ["eyezee"]
