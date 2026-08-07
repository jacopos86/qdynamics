from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.scaffold.hh_continuation_generators import build_generator_metadata
from pipelines.scaffold.hh_continuation_motifs import (
    extract_motif_library,
    load_motif_library_from_json,
    load_motif_library_from_payload,
    load_selected_logical_library_from_payload,
    match_selected_logical_generators,
    match_selected_logical_operator_families,
    merge_motif_libraries,
    motif_bonus_for_generator,
    select_tiled_generators_from_library,
)
from pipelines.scaffold.hh_continuation_symmetry import build_symmetry_spec
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.pauli_words import PauliTerm


def _poly(label: str) -> PauliPolynomial:
    return PauliPolynomial("JW", [PauliTerm(len(label), ps=label, pc=1.0)])


def test_extract_motif_library_preserves_structural_metadata() -> None:
    sym = build_symmetry_spec(family_id="paop_lf_std", mitigation_mode="verify_only")
    meta = build_generator_metadata(
        label="seed_left",
        polynomial=_poly("eeeexy"),
        family_id="paop_lf_std",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        symmetry_spec=sym.__dict__,
    )
    library = extract_motif_library(
        generator_metadata=[meta.__dict__],
        theta=[0.2],
        source_num_sites=2,
        source_tag="small_hh",
        ordering="blocked",
        boson_encoding="binary",
    )
    assert library["library_version"] == "phase3_motif_library_v1"
    assert library["records"][0]["family_id"] == "paop_lf_std"
    assert library["records"][0]["generator_ids"] == [meta.generator_id]
    assert library["records"][0]["symmetry_spec"]["mitigation_eligible"] is True


def test_select_tiled_generators_matches_on_metadata_not_labels() -> None:
    src_meta = build_generator_metadata(
        label="source_label",
        polynomial=_poly("eeeexy"),
        family_id="paop_lf_std",
        num_sites=2,
        ordering="blocked",
        qpb=1,
    )
    library = extract_motif_library(
        generator_metadata=[src_meta.__dict__],
        theta=[0.2],
        source_num_sites=2,
        source_tag="small_hh",
        ordering="blocked",
        boson_encoding="binary",
    )
    target_meta = build_generator_metadata(
        label="completely_different_target_name",
        polynomial=_poly("eeeezy"),
        family_id="paop_lf_std",
        num_sites=3,
        ordering="blocked",
        qpb=1,
    )
    target_registry = {
        "zzz_last": {
            **target_meta.__dict__,
            "candidate_label": "completely_different_target_name",
            "template_id": src_meta.template_id,
            "support_site_offsets": src_meta.support_site_offsets,
        }
    }
    seeded = select_tiled_generators_from_library(
        motif_library=library,
        registry_by_label=target_registry,
        target_num_sites=3,
        excluded_labels=[],
        max_seed=2,
    )
    assert [row["candidate_label"] for row in seeded] == ["completely_different_target_name"]
    assert seeded[0]["motif_metadata"]["motif_source"] == "small_hh"


def test_load_motif_library_from_payload_reconstructs_from_generator_metadata() -> None:
    meta_a = build_generator_metadata(
        label="seed_left",
        polynomial=_poly("eeeexy"),
        family_id="paop_lf_std",
        num_sites=2,
        ordering="blocked",
        qpb=1,
    )
    meta_b = build_generator_metadata(
        label="seed_right",
        polynomial=_poly("eeeezy"),
        family_id="paop_lf_std",
        num_sites=2,
        ordering="blocked",
        qpb=1,
    )
    payload = {
        "generated_utc": "2026-03-08T00:00:00Z",
        "settings": {
            "L": 2,
            "ordering": "blocked",
            "boson_encoding": "binary",
        },
        "adapt_vqe": {
            "optimal_point": [0.3, -0.1],
        },
        "continuation": {
            "selected_generator_metadata": [meta_a.__dict__, meta_b.__dict__],
        },
    }

    library = load_motif_library_from_payload(payload)

    assert library is not None
    assert library["source_tag"] == "2026-03-08T00:00:00Z"
    assert len(library["records"]) == 2
    assert library["records"][0]["generator_ids"] == [meta_a.generator_id]
    assert library["records"][1]["generator_ids"] == [meta_b.generator_id]
    assert library["records"][0]["family_id"] == "paop_lf_std"


def test_motif_bonus_and_json_load_round_trip(tmp_path: Path) -> None:
    meta = build_generator_metadata(
        label="seed_left",
        polynomial=_poly("eeeexy"),
        family_id="paop_lf_std",
        num_sites=2,
        ordering="blocked",
        qpb=1,
    )
    library = extract_motif_library(
        generator_metadata=[meta.__dict__],
        theta=[0.3],
        source_num_sites=2,
        source_tag="small_hh",
        ordering="blocked",
        boson_encoding="binary",
    )
    path = tmp_path / "motif.json"
    payload = {"continuation": {"motif_library": library}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_motif_library_from_json(path)
    bonus, motif_meta = motif_bonus_for_generator(
        generator_metadata=meta.__dict__,
        motif_library=loaded,
        target_num_sites=3,
    )
    assert loaded is not None
    assert bonus > 0.0
    assert motif_meta is not None
    assert motif_meta["target_num_sites"] == 3


def test_selected_logical_loader_accepts_nested_history_and_operator_fallback() -> None:
    meta = build_generator_metadata(
        label="hist_label",
        polynomial=_poly("eeeexy"),
        family_id="paop_lf_std",
        num_sites=2,
        ordering="blocked",
        qpb=1,
    )
    payload = {
        "generated_utc": "2026-04-01T00:00:00Z",
        "settings": {"L": 2, "ordering": "blocked", "boson_encoding": "binary"},
        "adapt_vqe": {
            "history": [
                {
                    "selected_feature_rows": [
                        {"generator_metadata": meta.__dict__},
                    ]
                }
            ],
            "operators": ["fallback_label"],
        },
    }

    library = load_selected_logical_library_from_payload(payload)

    assert library is not None
    assert library["schema"] == "selected_logical_library_v1"
    assert library["source_kind"] == "adapt_vqe.history.selected_feature_rows.generator_metadata"
    assert library["records"][0]["generator_id"] == meta.generator_id

    fallback = load_selected_logical_library_from_payload({"adapt_vqe": {"operators": ["fallback_label"]}})
    assert fallback is not None
    assert fallback["source_kind"] == "adapt_vqe.operators"
    assert fallback["records"][0]["candidate_label"] == "fallback_label"


def test_selected_logical_matching_prefers_generator_then_template_then_label() -> None:
    src_meta = build_generator_metadata(
        label="source_label",
        polynomial=_poly("eeeexy"),
        family_id="paop_lf_std",
        num_sites=2,
        ordering="blocked",
        qpb=1,
    )
    target_meta = {
        **src_meta.__dict__,
        "candidate_label": "renamed_target",
    }
    by_id = load_selected_logical_library_from_payload([src_meta.__dict__])
    assert by_id is not None
    matches = match_selected_logical_generators(
        selected_logical_library=by_id,
        registry_by_label={"renamed_target": target_meta},
        target_num_sites=2,
    )
    assert matches[0]["match_method"] == "generator_id"
    assert matches[0]["candidate_label"] == "renamed_target"

    structural_source = {
        "schema": "selected_logical_library_v1",
        "records": [
            {
                "family_id": src_meta.family_id,
                "template_id": src_meta.template_id,
                "support_site_offsets": src_meta.support_site_offsets,
                "boundary_behavior": "interior_only",
            }
        ],
    }
    structural = load_selected_logical_library_from_payload(structural_source)
    assert structural is not None
    structural_matches = match_selected_logical_generators(
        selected_logical_library=structural,
        registry_by_label={"renamed_target": {**target_meta, "generator_id": "different"}},
        target_num_sites=2,
    )
    assert structural_matches[0]["match_method"] == "template_support_offsets"

    label_source = load_selected_logical_library_from_payload({"adapt_vqe": {"operators": ["legacy_label"]}})
    assert label_source is not None
    label_matches = match_selected_logical_generators(
        selected_logical_library=label_source,
        registry_by_label={"legacy_label": {"candidate_label": "legacy_label"}},
        target_num_sites=2,
    )
    assert label_matches[0]["match_method"] == "exact_label"


def test_selected_logical_loader_does_not_stringify_none_metadata() -> None:
    library = load_selected_logical_library_from_payload(
        {
            "schema": "selected_logical_library_v1",
            "records": [
                {
                    "candidate_label": "uccsd_sing(alpha:0->1)",
                    "generator_id": None,
                    "family_id": None,
                    "template_id": "None",
                }
            ],
        }
    )

    assert library is not None
    record = library["records"][0]
    assert record["generator_id"] is None
    assert record["family_id"] is None
    assert record["template_id"] is None
    assert record["operator_family_id"] == "uccsd"


def test_selected_logical_operator_family_closure_uses_family_not_exact_label() -> None:
    source = load_selected_logical_library_from_payload(
        {
            "schema": "selected_logical_library_v1",
            "records": [
                {
                    "candidate_label": "uccsd_sing(alpha:0->1)",
                    "operator_label": "uccsd_sing(alpha:0->1)",
                    "family_id": "uccsd",
                }
            ],
        }
    )
    assert source is not None

    matches = match_selected_logical_operator_families(
        selected_logical_library=source,
        registry_by_label={
            "uccsd_sing(alpha:0->1)": {
                "candidate_label": "uccsd_sing(alpha:0->1)",
                "family_id": "full_meta",
            },
            "uccsd_dbl(ab:0,2->1,3)": {
                "candidate_label": "uccsd_dbl(ab:0,2->1,3)",
                "family_id": "full_meta",
            },
            "ham_full": {"candidate_label": "ham_full", "family_id": "full_meta"},
        },
        target_num_sites=2,
    )

    assert [row["registry_label"] for row in matches] == [
        "uccsd_sing(alpha:0->1)",
        "uccsd_dbl(ab:0,2->1,3)",
    ]
    assert {row["match_method"] for row in matches} == {
        "operator_family_closure",
        "operator_family_closure_from_exact",
    }
    assert all(row["operator_family_id"] == "uccsd" for row in matches)


def test_selected_logical_loader_accepts_motif_libraries() -> None:
    meta = build_generator_metadata(
        label="seed_left",
        polynomial=_poly("eeeexy"),
        family_id="paop_lf_std",
        num_sites=2,
        ordering="blocked",
        qpb=1,
    )
    motif_library = extract_motif_library(
        generator_metadata=[meta.__dict__],
        theta=[0.3],
        source_num_sites=2,
        source_tag="motif_src",
        ordering="blocked",
        boson_encoding="binary",
    )

    selected = load_selected_logical_library_from_payload({"continuation": {"motif_library": motif_library}})

    assert selected is not None
    assert selected["source_kind"] == "continuation.motif_library"
    assert selected["records"][0]["generator_ids"] == [meta.generator_id]


def test_merge_motif_libraries_rejects_incompatible_layout_metadata() -> None:
    meta = build_generator_metadata(
        label="seed_left",
        polynomial=_poly("eeeexy"),
        family_id="paop_lf_std",
        num_sites=2,
        ordering="blocked",
        qpb=1,
    )
    blocked = extract_motif_library(
        generator_metadata=[meta.__dict__],
        theta=[0.3],
        source_num_sites=2,
        source_tag="blocked_src",
        ordering="blocked",
        boson_encoding="binary",
    )
    interleaved = extract_motif_library(
        generator_metadata=[meta.__dict__],
        theta=[0.2],
        source_num_sites=2,
        source_tag="interleaved_src",
        ordering="interleaved",
        boson_encoding="binary",
    )
    with pytest.raises(ValueError, match="mismatched ordering/boson_encoding"):
        merge_motif_libraries([blocked, interleaved])
