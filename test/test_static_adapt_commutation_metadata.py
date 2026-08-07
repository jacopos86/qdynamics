from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import pipelines.static_adapt.commutation_metadata as retained_owner
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


REPO_ROOT = Path(__file__).resolve().parents[1]


def _term(module, word: str, coefficient: complex = 1.0):
    value = complex(coefficient)
    return module.SerializedPauliExpansionTerm(
        pauli_exyz=word,
        coeff_re=float(value.real),
        coeff_im=float(value.imag),
        nq=len(word),
    )


def _expansion(module, key: str, label: str, words: list[str]):
    terms = tuple(_term(module, word) for word in words)
    support = sorted(
        {
            qubit
            for term in terms
            for qubit in module.support_qubits_from_pauli_word(
                term.pauli_exyz
            )
        }
    )
    return module.GeneratorAlgebraicExpansion(
        key=key,
        label=label,
        generator_id=key,
        terms=terms,
        support_qubits=tuple(support),
        exactness=module.EXACTNESS_EXACT,
        source="parity_test",
    )


def _index_payload(index) -> dict:
    return {
        "expansions_by_key": {
            str(key): asdict(value)
            for key, value in index.expansions_by_key.items()
        },
        "label_to_keys": {
            str(key): tuple(value)
            for key, value in index.label_to_keys.items()
        },
    }


def test_retained_owner_has_no_lane_or_phase_routing_surface() -> None:
    retired_names = {
        "LANE_FLAT",
        "LANE_CURV",
        "LANE_DISJ",
        "LANE_MIX",
        "LANES_PHASE1",
        "AlgebraicLocalContextSummary",
        "summarize_local_context",
        "assign_lane",
        "algebraic_lane_quota_pressure_budgets",
        "phase1_lane_shortlist_records",
        "phase2_lane_health_shortlist_records",
        "build_phase0_weak_algebraic_index",
        "phase0_weak_lane_payload",
    }

    assert all(
        not hasattr(retained_owner, name) for name in retired_names
    )
    assert not hasattr(
        retained_owner.AlgebraicMetadataIndex,
        "summarize_local_context",
    )


def test_downstream_owners_do_not_reach_retired_algebraic_lane_partition() -> None:
    downstream_paths = (
        "pipelines/static_adapt/adapt_pipeline.py",
        "pipelines/static_adapt/commutation_metadata.py",
        "pipelines/static_adapt/lane_routes.py",
        "pipelines/static_adapt/phase_shortlists.py",
        "pipelines/static_adapt/selector_candidate_metadata.py",
        "pipelines/static_adapt/selector_feature_metadata.py",
        "pipelines/static_adapt/prune_schur_payloads.py",
    )
    forbidden = (
        "pipelines.static_adapt.algebraic_metadata",
        "STATIC_LANE_ROUTE_ALGEBRAIC",
        "LANES_PHASE1",
        "LANE_FLAT",
        "LANE_CURV",
        "LANE_DISJ",
        "LANE_MIX",
        "algebraic_lane_policy_active",
    )

    for relative_path in downstream_paths:
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert all(name not in source for name in forbidden), relative_path


def test_retired_algebraic_lane_partition_is_inert_and_manifested() -> None:
    original = REPO_ROOT / "pipelines/static_adapt/algebraic_metadata.py"
    manifest_path = (
        REPO_ROOT
        / "archive/paper_i_static_adapt_legacy_20260727/MANIFEST.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    family_entries = {
        str(entry["original_path"]): entry
        for entry in manifest["entries"]
        if entry["family"] == "algebraic_lane_partition"
    }
    assert set(family_entries) == {
        "pipelines/static_adapt/algebraic_metadata.py",
        "pipelines/static_adapt/algebraic_metadata.py#algebraic_lane_partition",
        (
            "pipelines/static_adapt/selector_candidate_metadata.py"
            "#algebraic_lane_partition_and_policy_gates"
        ),
        "pipelines/static_adapt/lane_routes.py#static_lane_route_algebraic",
        "test/test_static_adapt_algebraic_metadata.py",
    }
    assert not original.exists()
    assert not (
        REPO_ROOT / "test/test_static_adapt_algebraic_metadata.py"
    ).exists()

    for entry in family_entries.values():
        snapshot = REPO_ROOT / str(entry["archive_path"])
        assert snapshot.is_file()
        assert snapshot.name.endswith(".py.txt")
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == entry["sha256"]

    proof = manifest["source_index_proof"]
    assert "algebraic_lane_partition" in (
        proof["pre_archive_family_internal_source_sweep"]
    )
    assert "algebraic_lane_partition" in (
        proof["post_archive_active_source_sweep"]
    )


@pytest.mark.parametrize(
    ("lhs", "rhs", "support", "commutes", "product"),
    [
        ("x", "x", (0,), True, ("e", 1.0 + 0.0j)),
        ("x", "y", (0,), False, ("z", 0.0 + 1.0j)),
        ("xe", "ey", (1,), True, ("xy", 1.0 + 0.0j)),
        ("xyz", "zyx", (0, 1, 2), True, ("yey", 1.0 + 0.0j)),
        ("eex", "zye", (0,), True, ("zyx", 1.0 + 0.0j)),
    ],
)
def test_pauli_support_commutation_and_product_contract(
    lhs: str,
    rhs: str,
    support: tuple[int, ...],
    commutes: bool,
    product: tuple[str, complex],
) -> None:
    assert retained_owner.normalize_pauli_word_exyz(lhs.upper()) == lhs
    assert retained_owner.support_qubits_from_pauli_word(lhs) == support
    assert retained_owner.pauli_words_commute(lhs, rhs) is commutes
    assert retained_owner.multiply_pauli_words(lhs, rhs) == product


def test_serialized_and_registry_expansion_contract() -> None:
    raw_term = {
        "pauli_exyz": "eY",
        "coeff_re": 0.25,
        "coeff_im": -0.5,
        "nq": 2,
    }
    assert asdict(
        retained_owner.SerializedPauliExpansionTerm.from_mapping(raw_term)
    ) == {
        "pauli_exyz": "ey",
        "coeff_re": 0.25,
        "coeff_im": -0.5,
        "nq": 2,
    }

    metadata = {
        "candidate_label": "registered",
        "generator_id": "generator:registered",
        "support_qubits": [0, 1],
        "compile_metadata": {
            "serialized_terms_exyz": [
                {
                    "pauli_exyz": "xe",
                    "coeff_re": 1.0,
                    "coeff_im": 0.0,
                    "nq": 2,
                },
                {
                    "pauli_exyz": "ey",
                    "coeff_re": 0.0,
                    "coeff_im": -0.5,
                    "nq": 2,
                },
            ]
        },
    }
    expansion = retained_owner.expansion_from_generator_metadata(metadata)
    assert expansion.key == "generator:registered"
    assert expansion.label == "registered"
    assert expansion.support_qubits == (0, 1)
    assert expansion.exactness == retained_owner.EXACTNESS_EXACT
    assert [term.pauli_exyz for term in expansion.terms] == ["xe", "ey"]

    approximate = retained_owner.expansion_from_generator_metadata(
        {
            "candidate_label": "approximate",
            "support_qubits": [1],
            "compile_metadata": {},
        },
        require_exact=False,
    )
    assert asdict(approximate) == {
        "key": "approximate",
        "label": "approximate",
        "generator_id": None,
        "terms": (),
        "support_qubits": (1,),
        "exactness": retained_owner.EXACTNESS_APPROX,
        "source": "missing_serialized_terms_approx",
    }


def test_full_expansion_commutator_and_pair_metadata_contract() -> None:
    lhs = _expansion(
        retained_owner,
        "lhs",
        "LHS",
        ["x", "y"],
    )
    rhs = _expansion(
        retained_owner,
        "rhs",
        "RHS",
        ["x", "y"],
    )
    commutes, norm = retained_owner.exact_expansions_commute(lhs, rhs)
    assert commutes is True
    assert norm == pytest.approx(0.0)
    pair = retained_owner.build_pair_metadata(lhs, rhs)
    assert pair.support_overlap is True
    assert pair.commutes is True
    assert pair.relation == retained_owner.RELATION_FLAT_COMM
    assert pair.commutator_l1_norm == pytest.approx(0.0)


def test_polynomial_expansion_and_exact_index_contract() -> None:
    polynomial = PauliPolynomial(
        "JW",
        [
            PauliTerm(2, ps="xe", pc=1.0),
            PauliTerm(2, ps="ey", pc=-0.25j),
        ],
    )
    term = SimpleNamespace(label="polynomial", polynomial=polynomial)
    polynomial_expansion = retained_owner.expansion_from_ansatz_term(term)
    assert polynomial_expansion.key == "polynomial"
    assert polynomial_expansion.label == "polynomial"
    assert polynomial_expansion.support_qubits == (0, 1)
    assert [entry.pauli_exyz for entry in polynomial_expansion.terms] == [
        "ey",
        "xe",
    ]

    registry = {
        "registered": {
            "candidate_label": "registered",
            "generator_id": "generator:registered",
            "compile_metadata": {
                "serialized_terms_exyz": [
                    {
                        "pauli_exyz": "ez",
                        "coeff_re": 2.0,
                        "coeff_im": 0.0,
                        "nq": 2,
                    }
                ]
            },
        }
    }
    retained_index = retained_owner.build_exact_expansion_index(
        registry_by_label=registry
    )
    assert set(_index_payload(retained_index)["expansions_by_key"]) == {
        "generator:registered"
    }
    assert retained_index.resolve_key("registered") == "generator:registered"
    pair = retained_index.pair("registered", "registered")
    assert pair.commutes is True
    assert pair.relation == retained_owner.RELATION_FLAT_COMM


def test_exact_index_can_extend_for_a_new_runtime_split_term() -> None:
    index = retained_owner.build_exact_expansion_index(
        registry_by_label={},
    )
    polynomial = PauliPolynomial(
        "JW",
        [PauliTerm(2, ps="xy", pc=1.0)],
    )
    term = SimpleNamespace(
        label="runtime_split_child",
        polynomial=polynomial,
    )

    assert retained_owner.ensure_exact_expansion_in_index(
        index,
        term,
        {},
    )
    key = index.resolve_key("runtime_split_child")
    assert key == "runtime_split_child"
    assert index.expansions_by_key[key].support_qubits == (0, 1)

    before = _index_payload(index)
    assert retained_owner.ensure_exact_expansion_in_index(
        index,
        term,
        {},
    )
    assert _index_payload(index) == before


def test_exact_index_extension_fails_closed_without_an_exact_source() -> None:
    index = retained_owner.build_exact_expansion_index(
        registry_by_label={},
    )
    term = SimpleNamespace(label="missing_exact_source")

    with pytest.raises(
        retained_owner.AlgebraicMetadataError,
        match="missing exact serialized metadata",
    ):
        retained_owner.ensure_exact_expansion_in_index(
            index,
            term,
            {},
        )


@pytest.mark.parametrize(
    "raw",
    [
        {"pauli_exyz": "x", "coeff_re": 1.0, "nq": 1},
        {
            "pauli_exyz": "xx",
            "coeff_re": 1.0,
            "coeff_im": 0.0,
            "nq": 1,
        },
    ],
)
def test_invalid_serialized_term_fails_closed(raw: dict) -> None:
    with pytest.raises(retained_owner.AlgebraicMetadataError):
        retained_owner.SerializedPauliExpansionTerm.from_mapping(raw)
