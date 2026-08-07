from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
BUILDER_PATH = (
    ROOT
    / "chtc/phase3_optuna/build_paper_i_hh_sr_material_window_anchor_20260721.py"
)


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "paper_i_material_window_anchor_builder",
        BUILDER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load_builder()


def test_runtime_overlay_is_exactly_six_test1_modules(builder) -> None:
    assert builder.ANCHOR_ID.endswith("_v4_chtc")
    assert builder.ANCHOR_BATCH.endswith("-v4")
    assert set(builder.PRODUCTION_OVERLAYS) == {
        "pipelines/scaffold/hh_continuation_scoring.py",
        "pipelines/static_adapt/adapt_pipeline.py",
        "pipelines/static_adapt/phase3_material_window.py",
        "pipelines/static_adapt/route_a_trust_region.py",
        "pipelines/static_adapt/selector_candidate_metadata.py",
        "pipelines/static_adapt/sr_snake_route_profile.py",
    }
    assert "pipelines/static_adapt/estimator_call_ledger.py" not in (
        builder.PRODUCTION_OVERLAYS
    )
    assert "pipelines/static_adapt/formal_manifold_warm_start.py" not in (
        builder.PRODUCTION_OVERLAYS
    )


def test_inherited_execution_provenance_is_removed(builder, tmp_path: Path) -> None:
    inherited = (
        "route_registration_repair.json",
        "remote_execution_gate.json",
        "submission_artifact_hashes.json",
        "submission_receipt.json",
        "remote_preflight_provenance_inventory.json",
        "post_submission_provenance_inventory.json",
    )
    for name in inherited:
        (tmp_path / name).write_text("stale", encoding="utf-8")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "stale.pyc").write_bytes(b"stale")
    builder.clean_inherited_bundle_state(tmp_path)
    assert all(not (tmp_path / name).exists() for name in inherited)
    assert not cache.exists()


def test_reconstructed_source_is_hash_locked_and_parent_compatible(
    builder,
    tmp_path: Path,
) -> None:
    archive, files, overlays, contracts, compatibility = (
        builder.build_source_archive(tmp_path)
    )
    assert builder.sha256(archive) == builder.EXPECTED_RECONSTRUCTED_SOURCE_SHA256
    assert len(overlays) == 11
    assert compatibility["production_changed_path_count"] == 6
    assert set(compatibility["production_changed_paths"]) == set(
        builder.PRODUCTION_OVERLAYS
    )
    assert set(compatibility["protected_files"]) == set(
        builder.PROTECTED_PARENT_FILES
    )
    assert all(
        receipt["byte_identical_to_parent"]
        for receipt in compatibility["protected_files"].values()
    )
    assert contracts[builder.PARENT_ALIAS]["digest"] == builder.PARENT_DIGEST
    assert contracts[builder.CHILD_ALIAS]["digest"] == builder.CHILD_DIGEST
    assert set(compatibility["changed_paths"]) == set(builder.OVERLAY_FILES)
    assert len(files) > 300


def test_reviewed_adapt_hunk_allowlist_rejects_selected_hunk_drift(
    builder,
    tmp_path: Path,
) -> None:
    parent_archive = builder.BASE / "source_locked.tar.gz"
    overlay_archive = builder.IMMUTABLE_OVERLAY_ARCHIVE
    parent_root = tmp_path / "parent"
    overlay_root = tmp_path / "overlay"
    builder.extract_archive(parent_archive, parent_root)
    builder.extract_archive(overlay_archive, overlay_root)
    parent = parent_root / "pipelines/static_adapt/adapt_pipeline.py"
    overlay = overlay_root / "pipelines/static_adapt/adapt_pipeline.py"
    text = overlay.read_text(encoding="utf-8")
    anchor = "from pipelines.static_adapt.phase3_material_window import ("
    assert text.count(anchor) == 1
    overlay.write_text(
        text.replace(
            anchor,
            "from pipelines.static_adapt.phase3_material_window_drift import (",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reviewed Test-1 adapt hunks are missing"):
        builder.reconstruct_adapt_test1(parent, overlay)


def test_parent_default_geometry_telemetry_and_test2_route_are_absent(
    builder,
    tmp_path: Path,
) -> None:
    builder.build_source_archive(tmp_path)
    source = tmp_path / "source"
    scoring = (
        source / "pipelines/scaffold/hh_continuation_scoring.py"
    ).read_text(encoding="utf-8")
    route = (
        source / "pipelines/static_adapt/sr_snake_route_profile.py"
    ).read_text(encoding="utf-8")
    assert (
        '**({"acquisition_mode": str(acquisition_mode_key)} '
        "if coupling_screen else {})"
    ) in scoring
    assert "FS_PRUNE_VERIFY" not in route
    assert "fs_prune_verify" not in route
