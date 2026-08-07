from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.scaffold.hh_continuation_generators import (
    build_generator_metadata,
    build_pool_generator_registry,
    build_runtime_split_child_sets,
    build_runtime_split_children,
    build_split_event,
    clear_pool_generator_registry_cache_memory,
    rebuild_polynomial_from_serialized_terms,
    _commutator_l1_norm,
    _fermion_number_operators,
    _operator_symmetry_gate,
    _runtime_split_symmetry_gate,
)
from pipelines.scaffold.hh_continuation_symmetry import build_symmetry_spec
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


def test_fast_commutator_l1_matches_polynomial_product_formula() -> None:
    n_up, _n_dn = _fermion_number_operators(nq=4, num_sites=2, ordering="blocked")
    rhs = _mixed_macro_poly()
    slow_comm = n_up * rhs - rhs * n_up
    expected = float(sum(abs(complex(term.p_coeff)) for term in slow_comm.return_polynomial()))
    observed = _commutator_l1_norm(n_up, rhs)
    assert math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12)


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


def test_pool_registry_disk_cache_round_trips(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("STATIC_ADAPT_HH_GENERATOR_REGISTRY_CACHE", "disk")
    monkeypatch.setenv("STATIC_ADAPT_HH_GENERATOR_REGISTRY_CACHE_DIR", str(tmp_path))
    clear_pool_generator_registry_cache_memory()
    sym = build_symmetry_spec(family_id="paop_lf_std", mitigation_mode="verify_only")
    kwargs = dict(
        terms=[_term("macro", _macro_poly())],
        family_ids=["paop_lf_std"],
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_specs=[sym.__dict__],
    )
    first = build_pool_generator_registry(**kwargs)
    cache_files = list(tmp_path.glob("*.pickle"))
    assert len(cache_files) == 1
    clear_pool_generator_registry_cache_memory()
    second = build_pool_generator_registry(**kwargs)
    assert second == first


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


def test_runtime_split_hard_guard_uses_fixed_sector_invariance() -> None:
    sym = build_symmetry_spec(family_id="uccsd", mitigation_mode="verify_only")
    sym_spec = {**sym.__dict__, "hard_guard": True}
    parent_meta = build_generator_metadata(
        label="macro",
        polynomial=_number_preserving_macro_poly(),
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_spec=sym_spec,
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
        symmetry_spec=sym_spec,
        fixed_num_particles=(1, 1),
        hard_guard_required=True,
    )

    assert len(children) == 2
    for child in children:
        gate = child["symmetry_gate"]
        assert gate["passed"] is True
        assert gate["hard_guard_required"] is True
        assert gate["rejected"] is False
        assert gate["rejection_reason"] is None
        assert gate["gate_scope"] == "fixed_count_sector_invariance_v1"
        assert gate["globally_particle_number_commuting"] is False
        assert gate["fixed_count_sector"]["particle_sector_invariant"] is True
        assert gate["fixed_count_sector"]["spin_sector_invariant"] is True

    child_sets = build_runtime_split_child_sets(
        parent_label="macro",
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        split_mode="shortlist_pauli_children_v1",
        children=children,
        parent_generator_metadata=parent_meta.__dict__,
        symmetry_spec=sym_spec,
        fixed_num_particles=(1, 1),
        hard_guard_required=True,
        subset_sizes=(1,),
    )
    assert len(child_sets) == 2
    assert all(row["symmetry_gate"]["passed"] is True for row in child_sets)


def test_runtime_split_required_hard_guard_missing_spec_rejects_children() -> None:
    children = build_runtime_split_children(
        parent_label="macro",
        polynomial=_number_preserving_macro_poly(),
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        split_mode="shortlist_pauli_children_v1",
        symmetry_spec=None,
        fixed_num_particles=(1, 1),
        hard_guard_required=True,
    )

    assert len(children) == 2
    for child in children:
        gate = child["symmetry_gate"]
        assert gate["checked"] is False
        assert gate["passed"] is False
        assert gate["rejected"] is True
        assert gate["rejection_reason"] == "runtime_split_required_symmetry_spec_missing"
        assert gate["hard_guard_required"] is True
        runtime_gate = child["child_generator_metadata"]["compile_metadata"][
            "runtime_split"
        ]["symmetry_gate"]
        assert runtime_gate == gate

    child_sets = build_runtime_split_child_sets(
        parent_label="macro",
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        split_mode="shortlist_pauli_children_v1",
        children=children,
        symmetry_spec=None,
        fixed_num_particles=(1, 1),
        hard_guard_required=True,
        subset_sizes=(1,),
    )
    assert child_sets == []


def test_runtime_split_required_hard_guard_rejects_malformed_contracts() -> None:
    preserving_spec = {
        "particle_number_mode": "preserving",
        "spin_sector_mode": "preserving",
        "hard_guard": True,
    }
    malformed_cases = (
        (
            {
                "particle_number_mode": "preserving",
                "spin_sector_mode": "preserving",
                "hard_guard": False,
            },
            (1, 1),
            "runtime_split_required_symmetry_spec_malformed",
        ),
        (
            {
                "particle_number_mode": "preserving",
                "spin_sector_mode": "preserving",
                "hard_guard": "true",
            },
            (1, 1),
            "runtime_split_required_symmetry_spec_malformed",
        ),
        (
            preserving_spec,
            None,
            "runtime_split_required_fixed_num_particles_missing",
        ),
        (
            preserving_spec,
            (1,),
            "runtime_split_required_fixed_num_particles_malformed",
        ),
        (
            preserving_spec,
            (1, 1.5),
            "runtime_split_required_fixed_num_particles_malformed",
        ),
    )

    for symmetry_spec, fixed_num_particles, expected_reason in malformed_cases:
        children = build_runtime_split_children(
            parent_label="macro",
            polynomial=_number_preserving_macro_poly(),
            family_id="uccsd",
            num_sites=2,
            ordering="blocked",
            qpb=1,
            split_mode="shortlist_pauli_children_v1",
            symmetry_spec=symmetry_spec,
            fixed_num_particles=fixed_num_particles,
            hard_guard_required=True,
        )
        assert len(children) == 2
        for child in children:
            gate = child["symmetry_gate"]
            assert gate["checked"] is False
            assert gate["passed"] is False
            assert gate["rejected"] is True
            assert gate["rejection_reason"] == expected_reason


def test_fixed_sector_guard_rejects_true_fixed_count_leakage() -> None:
    leaking = PauliPolynomial("JW", [PauliTerm(4, ps="eeex", pc=1.0)])
    gate = _operator_symmetry_gate(
        polynomial=leaking,
        num_sites=2,
        ordering="blocked",
        symmetry_spec={
            "particle_number_mode": "preserving",
            "spin_sector_mode": "preserving",
            "hard_guard": True,
        },
        fixed_num_particles=(1, 1),
    )

    assert gate["gate_scope"] == "fixed_count_sector_invariance_v1"
    assert gate["passed"] is False
    assert gate["fixed_count_sector"]["particle_sector_leakage_l1"] > 0.0

    required_gate = _runtime_split_symmetry_gate(
        polynomial=leaking,
        num_sites=2,
        ordering="blocked",
        symmetry_spec={
            "particle_number_mode": "preserving",
            "spin_sector_mode": "preserving",
            "hard_guard": True,
        },
        fixed_num_particles=(1, 1),
        hard_guard_required=True,
    )
    assert required_gate["checked"] is True
    assert required_gate["passed"] is False
    assert required_gate["rejected"] is True
    assert (
        required_gate["rejection_reason"]
        == "runtime_split_fixed_count_sector_violation"
    )


def test_required_hard_guard_retains_one_valid_singleton_after_rejecting_sibling() -> None:
    symmetry_spec = {
        "particle_number_mode": "preserving",
        "spin_sector_mode": "preserving",
        "hard_guard": True,
    }
    parent = PauliPolynomial(
        "JW",
        [
            PauliTerm(4, ps="eeyx", pc=1.0),
            PauliTerm(4, ps="eeex", pc=1.0),
        ],
    )
    children = build_runtime_split_children(
        parent_label="mixed_guard_parent",
        polynomial=parent,
        family_id="test",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        split_mode="shortlist_pauli_children_v1",
        symmetry_spec=symmetry_spec,
        fixed_num_particles=(1, 1),
        hard_guard_required=True,
    )
    passed_indices = [
        int(child["child_index"])
        for child in children
        if bool(child["symmetry_gate"]["passed"])
    ]
    rejected_indices = [
        int(child["child_index"])
        for child in children
        if not bool(child["symmetry_gate"]["passed"])
    ]
    assert len(passed_indices) == 1
    assert len(rejected_indices) == 1

    child_sets = build_runtime_split_child_sets(
        parent_label="mixed_guard_parent",
        family_id="test",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        split_mode="shortlist_pauli_children_v1",
        children=children,
        symmetry_spec=symmetry_spec,
        fixed_num_particles=(1, 1),
        hard_guard_required=True,
        subset_sizes=(1,),
    )
    assert len(child_sets) == 1
    assert child_sets[0]["child_indices"] == passed_indices
    assert child_sets[0]["symmetry_gate"]["checked"] is True
    assert child_sets[0]["symmetry_gate"]["passed"] is True


def test_build_runtime_split_child_sets_only_returns_symmetry_safe_combinations() -> None:
    sym = build_symmetry_spec(family_id="uccsd", mitigation_mode="verify_only")
    sym_spec = dict(sym.__dict__)
    sym_spec["hard_guard"] = True
    parent_meta = build_generator_metadata(
        label="macro",
        polynomial=_mixed_macro_poly(),
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_spec=sym_spec,
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
        symmetry_spec=sym_spec,
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
        symmetry_spec=sym_spec,
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
    assert pair_meta["runtime_split"]["recommended_execution_mode"] == "grouped_exact"
    assert pair_meta["runtime_split"]["termwise_child_gates_all_passed"] is False
    assert singleton_meta["runtime_split"]["child_indices"] == [0]
    assert singleton_meta["runtime_split"]["recommended_execution_mode"] == "termwise_product"
    assert singleton_meta["runtime_split"]["termwise_child_gates_all_passed"] is True
    assert by_label["macro::child_set[1,2]"]["candidate_generator_metadata"]["symmetry_spec"]["particle_number_mode"] == "preserving"
    assert len(pair_meta["serialized_terms_exyz"]) == 2


def test_runtime_split_child_sets_check_preserving_spec_even_without_hard_guard() -> None:
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

    assert child_sets
    assert {row["candidate_label"] for row in child_sets} == {
        "macro::child_set[0]",
        "macro::child_set[1,2]",
    }
    for row in child_sets:
        gate = row["symmetry_gate"]
        assert gate["checked"] is True
        assert gate["passed"] is True
        assert gate.get("skipped_reason") != "runtime_split_symmetry_hard_guard_off"
        runtime_split = row["candidate_generator_metadata"]["compile_metadata"]["runtime_split"]
        runtime_gate = runtime_split["symmetry_gate"]
        assert runtime_gate["checked"] is True
        assert runtime_gate["passed"] is True
        expected_mode = (
            "termwise_product"
            if row["candidate_label"] == "macro::child_set[0]"
            else "grouped_exact"
        )
        assert runtime_split["recommended_execution_mode"] == expected_mode
        assert row["recommended_execution_mode"] == expected_mode


def test_runtime_split_subset_sizes_are_exact_and_independent_of_symmetry() -> None:
    parent_meta = build_generator_metadata(
        label="macro",
        polynomial=_mixed_macro_poly(),
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_spec=None,
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
        symmetry_spec=None,
    )

    pair_only = build_runtime_split_child_sets(
        parent_label="macro",
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        split_mode="shortlist_pauli_children_v1",
        children=children,
        parent_generator_metadata=parent_meta.__dict__,
        symmetry_spec=None,
        subset_sizes=(2,),
        max_subset_size=1,
    )
    singleton_and_pair = build_runtime_split_child_sets(
        parent_label="macro",
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        split_mode="shortlist_pauli_children_v1",
        children=children,
        parent_generator_metadata=parent_meta.__dict__,
        symmetry_spec=None,
        subset_sizes=(1, 2),
    )
    unavailable_four = build_runtime_split_child_sets(
        parent_label="macro",
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        split_mode="shortlist_pauli_children_v1",
        children=children,
        parent_generator_metadata=parent_meta.__dict__,
        symmetry_spec=None,
        subset_sizes=(4,),
    )

    assert {tuple(row["child_indices"]) for row in pair_only} == {
        (0, 1),
        (0, 2),
        (1, 2),
    }
    assert all(row["subset_cardinality"] == 2 for row in pair_only)
    assert all(row["requested_subset_sizes"] == [2] for row in pair_only)
    assert {row["subset_cardinality"] for row in singleton_and_pair} == {1, 2}
    assert len(singleton_and_pair) == 6
    assert unavailable_four == []


def test_archival_runtime_split_child_sets_can_preserve_missing_spec_skip_pass() -> None:
    sym = build_symmetry_spec(family_id="uccsd", mitigation_mode="verify_only")
    hard_guard_spec = dict(sym.__dict__)
    hard_guard_spec["hard_guard"] = True
    parent_meta = build_generator_metadata(
        label="macro",
        polynomial=_mixed_macro_poly(),
        family_id="uccsd",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_spec=hard_guard_spec,
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
        symmetry_spec=hard_guard_spec,
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
        symmetry_spec=None,
        max_subset_size=3,
    )

    assert child_sets
    for row in child_sets:
        gate = row["symmetry_gate"]
        assert gate["checked"] is False
        assert gate["passed"] is True
        assert gate["skipped_reason"] == "runtime_split_symmetry_spec_missing"
        meta = row["candidate_generator_metadata"]
        assert meta["symmetry_spec"] is None
        runtime_split = meta["compile_metadata"]["runtime_split"]
        runtime_gate = runtime_split["symmetry_gate"]
        assert runtime_gate["checked"] is False
        assert runtime_gate["passed"] is True
        assert runtime_gate["skipped_reason"] == "runtime_split_symmetry_spec_missing"
        assert runtime_split["representation"] == "child_set"


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
        parent_collapse_diagnostic={
            "selection_mode": "proxy_child_set_preselection",
            "depth": 4,
        },
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
    assert event["parent_collapse_diagnostic"]["selection_mode"] == "proxy_child_set_preselection"


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


def test_runtime_split_preserves_terms_above_its_declared_tolerance() -> None:
    tiny_coefficient = 5.0e-8
    parent = PauliPolynomial(
        "JW",
        [
            PauliTerm(4, ps="xeee", pc=tiny_coefficient),
            PauliTerm(4, ps="yeee", pc=1.0),
        ],
    )

    children = build_runtime_split_children(
        parent_label="tiny_macro",
        polynomial=parent,
        family_id="test",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        split_mode="shortlist_pauli_children_v1",
        tol=1.0e-12,
    )
    child_sets = build_runtime_split_child_sets(
        parent_label="tiny_macro",
        family_id="test",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        split_mode="shortlist_pauli_children_v1",
        children=children,
        subset_sizes=(1,),
        tol=1.0e-12,
    )

    assert len(children) == 2
    assert len(child_sets) == 2
    tiny_child = min(
        children,
        key=lambda child: abs(
            complex(child["child_polynomial"].return_polynomial()[0].p_coeff)
        ),
    )
    tiny_child_terms = tiny_child["child_polynomial"].return_polynomial()
    assert len(tiny_child_terms) == 1
    assert complex(tiny_child_terms[0].p_coeff) == complex(tiny_coefficient)
