from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_continuation_generators import build_generator_metadata
from pipelines.hardcoded.hh_continuation_motifs import (
    extract_motif_library,
    load_motif_library_from_json,
    load_motif_library_from_payload,
    merge_motif_libraries,
    motif_bonus_for_generator,
    select_tiled_generators_from_library,
)
from pipelines.hardcoded.hh_continuation_symmetry import build_symmetry_spec
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
