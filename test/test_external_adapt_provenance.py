#!/usr/bin/env python3
"""Tests for external ADAPT competitor provenance scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

from pipelines.exact_bench.external_adapt.provenance import (
    CEO_ADAPT_VQE_PINNED_COMMIT,
    EXTERNAL_ADAPT_ALGORITHM_IDS,
    external_algorithm_manifest_metadata,
    get_external_reference_spec,
    reference_catalog,
    reference_specs_for_algorithm,
)
from pipelines.exact_bench.external_adapt.repository_manager import (
    catalog_payload,
    checkout_dir_for,
)


def test_external_reference_catalog_contains_expected_competitors() -> None:
    ids = {spec.reference_id for spec in reference_catalog()}

    assert "ceo_adapt_vqe" in ids
    assert "hrgrimsl_adapt" in ids
    assert "jordanovsj_vqe" in ids
    assert "overlap_adapt_vqe_request" in ids


def test_ceo_reference_records_required_public_checkout_pin() -> None:
    spec = get_external_reference_spec("ceo_adapt_vqe")

    assert spec.default_ref == CEO_ADAPT_VQE_PINNED_COMMIT
    assert spec.pinned_commit == CEO_ADAPT_VQE_PINNED_COMMIT
    assert "openfermion" in spec.package_imports
    assert "pyscf" in spec.package_imports


def test_ceo_and_tetris_rows_share_ceo_reference_but_overlap_is_request_only() -> None:
    ceo_refs = reference_specs_for_algorithm("static_ceo_adapt_phase3")
    tetris_refs = reference_specs_for_algorithm("static_tetris_adapt_phase3")
    overlap_refs = reference_specs_for_algorithm("static_overlap_adapt_phase3")

    assert [ref.reference_id for ref in ceo_refs] == ["ceo_adapt_vqe"]
    assert [ref.reference_id for ref in tetris_refs] == ["ceo_adapt_vqe"]
    assert [ref.reference_id for ref in overlap_refs] == ["overlap_adapt_vqe_request"]
    assert overlap_refs[0].availability == "request_only"


def test_external_algorithm_manifest_metadata_marks_no_phase3_emulation() -> None:
    metadata = external_algorithm_manifest_metadata(
        "static_tetris_adapt_phase3",
        status="runnable",
        dispatch="external_static_adapt_tetris_public_code",
    )

    assert metadata["external_algorithm"] is True
    assert metadata["phase3_controller_called"] is False
    assert metadata["external_adapt_policy"] == "do_not_emulate_through_phase3_controller"
    assert metadata["external_adapt_reference_ids"] == ["ceo_adapt_vqe"]
    assert metadata["external_adapt_algorithm_adapter_status"] == (
        "implemented_hubbard_L2_tetris_public_code_parameterized_cases"
    )
    assert metadata["external_adapt_dispatch"] == "external_static_adapt_tetris_public_code"
    assert metadata["external_adapt_pinned_commits"] == {"ceo_adapt_vqe": CEO_ADAPT_VQE_PINNED_COMMIT}
    assert "openfermion" in metadata["external_adapt_package_imports"]["ceo_adapt_vqe"]
    assert "openfermionpyscf" in metadata["external_adapt_package_imports"]["ceo_adapt_vqe"]


def test_ceo_algorithm_metadata_marks_first_slice_adapter() -> None:
    metadata = external_algorithm_manifest_metadata(
        "static_ceo_adapt_phase3",
        status="runnable",
        dispatch="external_static_adapt_ceo_public_code",
    )

    assert metadata["external_adapt_algorithm_adapter_status"] == "implemented_hubbard_L2_public_code_first_slice"
    assert metadata["external_adapt_dispatch"] == "external_static_adapt_ceo_public_code"


def test_non_external_algorithm_gets_empty_metadata() -> None:
    assert external_algorithm_manifest_metadata("static_family_native_adapt_phase3") == {}
    assert "static_tetris_adapt_phase3" in EXTERNAL_ADAPT_ALGORITHM_IDS


def test_catalog_payload_and_checkout_dir_are_json_safe(tmp_path: Path) -> None:
    payload = catalog_payload()
    encoded = json.dumps(payload, sort_keys=True)
    path = checkout_dir_for("ceo_adapt_vqe", cache_root=tmp_path)

    assert "ceo_adapt_vqe" in encoded
    assert path == tmp_path / "ceo_adapt_vqe"


def test_get_unknown_external_reference_fails_clearly() -> None:
    try:
        get_external_reference_spec("missing")
    except ValueError as exc:
        assert "Unknown external reference" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing reference should fail")
