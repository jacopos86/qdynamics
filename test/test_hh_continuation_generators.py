from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_continuation_generators import (
    build_generator_metadata,
    build_pool_generator_registry,
    build_runtime_split_children,
    build_split_event,
)
from pipelines.hardcoded.hh_continuation_symmetry import build_symmetry_spec
from src.quantum.pauli_polynomial_class import PauliPolynomial
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


def test_build_runtime_split_children_emits_serialized_single_term_children() -> None:
    sym = build_symmetry_spec(family_id="paop_lf_std", mitigation_mode="verify_only")
    parent_meta = build_generator_metadata(
        label="macro",
        polynomial=_macro_poly(),
        family_id="paop_lf_std",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_spec=sym.__dict__,
    )
    children = build_runtime_split_children(
        parent_label="macro",
        polynomial=_macro_poly(),
        family_id="paop_lf_std",
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
        assert len(compile_meta["serialized_terms_exyz"]) == 1
