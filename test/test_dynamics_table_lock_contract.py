from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipelines.time_dynamics.tables.dynamics_benchmark_contract import (
    DYNAMICS_COARSE_TUNING_CLASSES,
    DynamicsBenchmarkCase,
)
from pipelines.time_dynamics.tables.table_lock_contract import (
    DYNAMICS_CLASS_SETTINGS_LOCK_MANIFEST_SCHEMA,
    build_locked_or_default_tuning_provenance,
    case_with_class_settings_overrides,
    controller_cli_tokens_from_settings,
    seed_lock_metadata,
    validate_class_settings_lock_manifest,
    validate_same_seed_rows,
    with_class_settings_lock_manifest,
)


def _case(tmp_path: Path, *, metadata: dict | None = None) -> DynamicsBenchmarkCase:
    artifact = tmp_path / "seed.json"
    artifact.write_text('{"ok": true}\n', encoding="utf-8")
    return DynamicsBenchmarkCase(
        case_id="unit_hubbard_A0p2",
        family="hubbard",
        table_class="fermionic_lattice",
        artifact_json=str(artifact),
        metadata={} if metadata is None else metadata,
    )


def test_seed_lock_metadata_hashes_static_seed_and_preserves_group(tmp_path: Path) -> None:
    case = _case(
        tmp_path,
        metadata={
            "seed_lock": {
                "same_seed_comparator_group_id": "hubbard_A0p2_same_seed",
                "seed_selection_policy": "current_best_static_adapt_seed_from_latest_phase3_summary",
            }
        },
    )

    lock = seed_lock_metadata(case)

    expected = hashlib.sha256(Path(case.artifact_json).read_bytes()).hexdigest()
    assert lock["same_seed_comparator_group_id"] == "hubbard_A0p2_same_seed"
    assert lock["static_seed_artifact_sha256"] == expected
    assert lock["same_seed_validation_status"] == "hash_recorded"
    assert lock["seed_selection_policy"] == "current_best_static_adapt_seed_from_latest_phase3_summary"


def test_seed_lock_metadata_flattens_paper_ii_hh_seed_track_provenance(tmp_path: Path) -> None:
    case = _case(
        tmp_path,
        metadata={
            "seed_lock": {
                "same_seed_comparator_group_id": "hh_weak_weak_geo_A0p2_t8_dt321_same_seed_v1",
                "seed_track": "geo",
                "static_algorithm_id": "static_geo_adapt_vqe",
                "static_seed_display_label": "GeoAdapt",
                "hh_regime_id": "weak_weak",
                "hh_static_case_id": "hh_L2_nph2_three_model_sym_weak_weak",
                "hh_u_over_t": 0.25,
                "hh_lambda": 0.25,
                "n_ph_work": 2,
                "n_ph_ref": 5,
                "source_artifact_json": "artifacts/source/runtime_seed.json",
                "source_artifact_sha256": "abc123",
                "normalized_seed_artifact_json": "chtc/generic_time_dynamics_table/input/seed.json",
                "normalized_seed_artifact_sha256": "def456",
                "static_abs_delta_e": 1.5e-4,
                "static_parameter_count": 7,
                "runtime_loadability_status": "payload_fields_present_dry_load_not_run",
                "latest_phase3_source_artifact_missing_locally": False,
            }
        },
    )

    lock = seed_lock_metadata(case)

    assert lock["seed_track"] == "geo"
    assert lock["static_algorithm_id"] == "static_geo_adapt_vqe"
    assert lock["static_seed_display_label"] == "GeoAdapt"
    assert lock["hh_regime_id"] == "weak_weak"
    assert lock["hh_static_case_id"] == "hh_L2_nph2_three_model_sym_weak_weak"
    assert lock["source_artifact_sha256"] == "abc123"
    assert lock["normalized_seed_artifact_sha256"] == "def456"
    assert lock["static_abs_delta_e"] == 1.5e-4
    assert lock["static_parameter_count"] == 7
    assert lock["runtime_loadability_status"] == "payload_fields_present_dry_load_not_run"
    assert lock["latest_phase3_source_artifact_missing_locally"] is False


def test_class_settings_manifest_locks_controller_tuning_provenance(tmp_path: Path) -> None:
    manifest = tmp_path / "class_settings.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": DYNAMICS_CLASS_SETTINGS_LOCK_MANIFEST_SCHEMA,
                "lock_status": "locked",
                "settings": [
                    {
                        "tuning_class": "fermionic",
                        "algorithm_id": "dyn_controller_full",
                        "settings_kind": "controller",
                        "settings_source": "unit_class_optuna_v1",
                        "class_tuned_result_locked": True,
                        "settings_payload": {
                            "miss_threshold": 0.4,
                            "gain_ratio_threshold": 0.02,
                            "append_enabled": True,
                            "prune_mode": "schur_projected_shadow_v1",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    case = with_class_settings_lock_manifest(
        _case(tmp_path),
        manifest_path=manifest,
        require_locked=True,
    )

    provenance = build_locked_or_default_tuning_provenance(
        case=case,
        algorithm_id="dyn_controller_full",
        settings_kind="controller",
        settings_payload={"miss_threshold": 0.9},
        locked=False,
    )

    assert provenance["tuning_class"] == "fermionic"
    assert provenance["settings_source"] == "unit_class_optuna_v1"
    assert provenance["class_tuned_result_locked"] is True
    assert provenance["tuning_validation_status"] == "locked_coarse_class_tuned"
    assert "unit_hubbard_A0p2" not in provenance["settings_id"]


def test_require_locked_class_settings_fails_closed_when_missing_entry(tmp_path: Path) -> None:
    manifest = tmp_path / "class_settings.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": DYNAMICS_CLASS_SETTINGS_LOCK_MANIFEST_SCHEMA,
                "lock_status": "locked",
                "settings": [],
            }
        ),
        encoding="utf-8",
    )
    case = with_class_settings_lock_manifest(
        _case(tmp_path),
        manifest_path=manifest,
        require_locked=True,
    )

    with pytest.raises(ValueError, match="has no entry"):
        build_locked_or_default_tuning_provenance(
            case=case,
            algorithm_id="dyn_controller_full",
            settings_kind="controller",
        )


def test_canonical_class_settings_manifest_requires_exact_three_controller_classes() -> None:
    settings = [
        {
            "tuning_class": tuning_class,
            "algorithm_id": "dyn_controller_full",
            "variant_id": "full_controller",
            "settings_kind": "controller",
            "class_tuned_result_locked": True,
            "strict_online_feedback_exact_free": True,
            "settings_payload": {
                "checkpoint_controller_mode": "observable_v1",
                "checkpoint_controller_exact_input_mode": "off",
                "miss_threshold": 0.05,
            },
        }
        for tuning_class in DYNAMICS_COARSE_TUNING_CLASSES
    ]
    manifest = {
        "schema": DYNAMICS_CLASS_SETTINGS_LOCK_MANIFEST_SCHEMA,
        "lock_status": "locked",
        "require_canonical_controller_classes": True,
        "settings": settings,
    }

    validation = validate_class_settings_lock_manifest(
        manifest,
        require_exact_controller_classes=True,
    )

    assert validation["canonical_controller_classes"] == sorted(DYNAMICS_COARSE_TUNING_CLASSES)
    missing = dict(manifest)
    missing["settings"] = settings[:-1]
    with pytest.raises(ValueError, match="exactly one locked"):
        validate_class_settings_lock_manifest(missing, require_exact_controller_classes=True)


def test_class_settings_manifest_rejects_case_specific_payload_key() -> None:
    manifest = {
        "schema": DYNAMICS_CLASS_SETTINGS_LOCK_MANIFEST_SCHEMA,
        "lock_status": "locked",
        "settings": [
            {
                "tuning_class": "fermionic",
                "algorithm_id": "dyn_controller_full",
                "settings_kind": "controller",
                "class_tuned_result_locked": True,
                "settings_payload": {"case_id": "unit_case", "miss_threshold": 0.05},
            }
        ],
    }

    with pytest.raises(ValueError, match="case-specific settings keys"):
        validate_class_settings_lock_manifest(manifest)


def test_all_algorithm_candidate_lock_requires_every_table_i_algorithm_class() -> None:
    settings = []
    for tuning_class in DYNAMICS_COARSE_TUNING_CLASSES:
        for algorithm_id, settings_kind in {
            "dyn_controller_full": "controller",
            "dyn_fixed_mclachlan": "mclachlan",
            "dyn_product_formula_envelope": "comparator",
            "dyn_qdrift": "comparator",
            "dyn_fixed_pvqd": "comparator",
            "dyn_adaptive_pvqd": "comparator",
            "dyn_avqds": "comparator",
            "dyn_avqds_tetris": "comparator",
        }.items():
            settings.append(
                {
                    "tuning_class": tuning_class,
                    "algorithm_id": algorithm_id,
                    "settings_kind": settings_kind,
                    "settings_payload": {"qdrift_samples_per_interval": 8}
                    if algorithm_id == "dyn_qdrift"
                    else {},
                    "class_tuned_result_locked": False,
                    "candidate_only_not_promoted": True,
                }
            )
    manifest = {
        "schema": DYNAMICS_CLASS_SETTINGS_LOCK_MANIFEST_SCHEMA,
        "lock_status": "candidate_not_promoted",
        "require_all_table_i_algorithm_classes": True,
        "settings": settings,
    }

    validation = validate_class_settings_lock_manifest(manifest)

    assert validation["required_algorithm_class_entry_count"] == 24
    assert validation["candidate_only_entry_count"] == 24
    missing = dict(manifest)
    missing["settings"] = settings[:-1]
    with pytest.raises(ValueError, match="missing required all-algorithm class entries"):
        validate_class_settings_lock_manifest(missing)


def test_case_with_class_settings_overrides_merges_comparator_payload(tmp_path: Path) -> None:
    manifest = tmp_path / "class_settings.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": DYNAMICS_CLASS_SETTINGS_LOCK_MANIFEST_SCHEMA,
                "lock_status": "candidate_not_promoted",
                "require_all_table_i_algorithm_classes": False,
                "settings": [
                    {
                        "tuning_class": "fermionic",
                        "algorithm_id": "dyn_qdrift",
                        "settings_kind": "comparator",
                        "settings_source": "unit_candidate",
                        "settings_payload": {
                            "qdrift_samples_per_interval": 3,
                            "qdrift_rng_seed": 123,
                        },
                        "class_tuned_result_locked": False,
                        "candidate_only_not_promoted": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    case = with_class_settings_lock_manifest(_case(tmp_path), manifest_path=manifest)

    effective = case_with_class_settings_overrides(
        case,
        algorithm_id="dyn_qdrift",
        settings_kind="comparator",
    )

    assert effective.metadata["qdrift_samples_per_interval"] == 3
    assert effective.metadata["qdrift_rng_seed"] == 123
    assert effective.metadata["effective_class_settings_entries"]["dyn_qdrift:comparator"][
        "candidate_only_not_promoted"
    ] is True


def test_require_algorithm_class_settings_fails_closed_for_missing_comparator(tmp_path: Path) -> None:
    manifest = tmp_path / "class_settings.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": DYNAMICS_CLASS_SETTINGS_LOCK_MANIFEST_SCHEMA,
                "lock_status": "candidate_not_promoted",
                "require_all_table_i_algorithm_classes": False,
                "required_algorithm_settings": [
                    {"algorithm_id": "dyn_qdrift", "settings_kind": "comparator"}
                ],
                "settings": [],
            }
        ),
        encoding="utf-8",
    )
    case = with_class_settings_lock_manifest(_case(tmp_path), manifest_path=manifest)

    with pytest.raises(ValueError, match="missing required all-algorithm class entries"):
        case_with_class_settings_overrides(
            case,
            algorithm_id="dyn_qdrift",
            settings_kind="comparator",
        )


def test_require_locked_class_settings_allows_non_controller_defaults_when_missing(tmp_path: Path) -> None:
    manifest = tmp_path / "class_settings.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": DYNAMICS_CLASS_SETTINGS_LOCK_MANIFEST_SCHEMA,
                "lock_status": "locked",
                "settings": [
                    {
                        "tuning_class": "fermionic",
                        "algorithm_id": "dyn_controller_full",
                        "settings_kind": "controller",
                        "class_tuned_result_locked": True,
                        "settings_payload": {"miss_threshold": 0.05},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    case = with_class_settings_lock_manifest(
        _case(tmp_path),
        manifest_path=manifest,
        require_locked=True,
    )

    provenance = build_locked_or_default_tuning_provenance(
        case=case,
        algorithm_id="dyn_fixed_mclachlan",
        settings_kind="mclachlan",
        settings_payload={"lock_fixed_manifold": True},
    )

    assert provenance["class_tuned_result_locked"] is False
    assert provenance["class_settings_lock_manifest"] == str(manifest.resolve())
    assert provenance["settings_kind"] == "mclachlan"


def test_controller_cli_tokens_use_only_algorithm_settings() -> None:
    tokens = controller_cli_tokens_from_settings(
        {
            "miss_threshold": 0.3,
            "gain_ratio_threshold": 0.01,
            "append_enabled": False,
            "lock_fixed_manifold": True,
            "unknown_or_case_specific_ignored": "x",
        }
    )

    assert "--checkpoint-controller-miss-threshold" in tokens
    assert "0.3" in tokens
    assert "--checkpoint-controller-gain-ratio-threshold" in tokens
    assert "--no-checkpoint-controller-append-enabled" in tokens
    assert "--lock-fixed-manifold" in tokens
    assert "unknown_or_case_specific_ignored" not in tokens


def test_same_seed_rows_validation_flags_hash_mismatch() -> None:
    good = {
        "case_id": "case_a",
        "provenance": {
            "same_seed_comparator_group_id": "group",
            "static_seed_artifact_json": "seed.json",
            "static_seed_artifact_sha256": "abc",
        },
    }
    bad = {
        "case_id": "case_a",
        "provenance": {
            "same_seed_comparator_group_id": "group",
            "static_seed_artifact_json": "seed.json",
            "static_seed_artifact_sha256": "def",
        },
    }

    assert validate_same_seed_rows([good])["passed"] is True
    validation = validate_same_seed_rows([good, bad])
    assert validation["passed"] is False
    assert "group" in validation["bad_groups"]
