#!/usr/bin/env python3
"""Table-lock contracts for Paper II time-dynamics evidence runs.

This module is intentionally small and data-only.  It records two invariants
needed before Paper-II dynamics table values are credible:

1. every method for a benchmark point must use the same static ADAPT seed; and
2. checkpoint-controller settings used for paper-facing runs must be locked at
   coarse Hamiltonian-class granularity, not per Hamiltonian instance.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics.tables.dynamics_benchmark_contract import (
    DYNAMICS_CANONICAL_CONTROLLER_ALGORITHM_ID,
    DYNAMICS_CANONICAL_CONTROLLER_VARIANT_ID,
    DYNAMICS_CLASS_TUNING_DEFAULT_SOURCE,
    DYNAMICS_CLASS_SETTINGS_KINDS,
    DYNAMICS_COARSE_TUNING_CLASSES,
    DYNAMICS_TABLE_I_ALGORITHM_SETTINGS_KIND,
    DYNAMICS_FORBIDDEN_TUNING_ID_KEYS,
    DYNAMICS_SETTINGS_KIND_CONTROLLER,
    DynamicsBenchmarkCase,
    normalize_dynamics_tuning_class,
    build_dynamics_tuning_provenance,
    dynamics_tuning_class,
    json_safe,
)

DYNAMICS_TABLE_LOCK_SEED_MANIFEST_SCHEMA = "dynamics_table_lock_seed_manifest_v1"
DYNAMICS_CLASS_SETTINGS_LOCK_MANIFEST_SCHEMA = "dynamics_class_settings_lock_manifest_v1"
DYNAMICS_TABLE_LOCK_SOURCE = "paper_ii_same_seed_table_lock_v1"
DYNAMICS_CLASS_SETTINGS_LOCK_SOURCE = "paper_ii_class_locked_settings_v1"
_LOCKED_MANIFEST_STATUSES = {"locked", "promoted_locked"}

_CLASS_SETTINGS_MANIFEST_METADATA_KEY = "class_settings_lock_manifest"
_REQUIRE_CLASS_SETTINGS_METADATA_KEY = "require_locked_class_settings"
_REQUIRE_ALGORITHM_CLASS_SETTINGS_METADATA_KEY = "require_algorithm_class_settings"
_EFFECTIVE_CLASS_SETTINGS_METADATA_KEY = "effective_class_settings_entries"

# Controller settings emitted by the Optuna class-settings candidate are stored
# without CLI prefixes.  This map is deliberately limited to checkpoint-controller
# and generic route knobs; case-specific physics/time-grid/drive/seed keys remain
# forbidden by build_dynamics_tuning_provenance.
_CONTROLLER_SETTING_CLI_FLAGS: dict[str, str] = {
    "checkpoint_controller_mode": "--checkpoint-controller-mode",
    "checkpoint_controller_exact_input_mode": "--checkpoint-controller-exact-input-mode",
    "checkpoint_controller_reference_mode": "--checkpoint-controller-reference-mode",
    "checkpoint_controller_noise_mode": "--checkpoint-controller-noise-mode",
    "integrator_policy": "--checkpoint-controller-integrator-policy",
    "prune_mode": "--checkpoint-controller-prune-mode",
    "high_miss_no_admit_policy": "--checkpoint-controller-high-miss-no-admit-policy",
    "oracle_selection_policy": "--checkpoint-controller-oracle-selection-policy",
    "confirm_score_mode": "--checkpoint-controller-confirm-score-mode",
    "miss_threshold": "--checkpoint-controller-miss-threshold",
    "gain_ratio_threshold": "--checkpoint-controller-gain-ratio-threshold",
    "candidate_step_scales": "--checkpoint-controller-candidate-step-scales",
    "miss_abs_threshold": "--checkpoint-controller-miss-abs-threshold",
    "append_margin_abs": "--checkpoint-controller-append-margin-abs",
    "append_no_harm_condition_ratio_cap": "--checkpoint-controller-append-no-harm-condition-ratio-cap",
    "append_no_harm_condition_abs_floor": "--checkpoint-controller-append-no-harm-condition-abs-floor",
    "append_no_harm_displacement_ratio_cap": "--checkpoint-controller-append-no-harm-displacement-ratio-cap",
    "append_no_harm_kink_min_step_gain_delta": "--checkpoint-controller-append-no-harm-kink-min-step-gain-delta",
    "append_no_harm_kink_max_condition_ratio": "--checkpoint-controller-append-no-harm-kink-max-condition-ratio",
    "append_no_harm_kink_max_displacement_ratio": "--checkpoint-controller-append-no-harm-kink-max-displacement-ratio",
    "append_no_harm_rho_only_min_step_gain_delta": "--checkpoint-controller-append-no-harm-rho-only-min-step-gain-delta",
    "append_no_harm_rho_only_condition_ratio_cap": "--checkpoint-controller-append-no-harm-rho-only-condition-ratio-cap",
    "append_no_harm_rho_only_step_residual_ratio_cap": "--checkpoint-controller-append-no-harm-rho-only-step-residual-ratio-cap",
    "append_no_harm_rho_only_displacement_ratio_cap": "--checkpoint-controller-append-no-harm-rho-only-displacement-ratio-cap",
    "shortlist_fraction": "--checkpoint-controller-shortlist-fraction",
    "regularization_lambda": "--checkpoint-controller-regularization-lambda",
    "candidate_regularization_lambda": "--checkpoint-controller-candidate-regularization-lambda",
    "pinv_rcond": "--checkpoint-controller-pinv-rcond",
    "compile_penalty_weight": "--checkpoint-controller-compile-penalty-weight",
    "measurement_penalty_weight": "--checkpoint-controller-measurement-penalty-weight",
    "directional_penalty_weight": "--checkpoint-controller-directional-penalty-weight",
    "confirm_compress_fraction": "--checkpoint-controller-confirm-compress-fraction",
    "shortlist_size": "--checkpoint-controller-shortlist-size",
    "active_window_size": "--checkpoint-controller-active-window-size",
    "max_probe_positions": "--checkpoint-controller-max-probe-positions",
    "confirm_compress_min_modes": "--checkpoint-controller-confirm-compress-min-modes",
    "confirm_compress_max_modes": "--checkpoint-controller-confirm-compress-max-modes",
    "progress_observable_window": "--checkpoint-controller-progress-observable-window",
    "integrator_euler_observable_window": "--checkpoint-controller-integrator-euler-observable-window",
    "integrator_columnarity_threshold": "--checkpoint-controller-integrator-columnarity-threshold",
    "integrator_curvature_threshold": "--checkpoint-controller-integrator-curvature-threshold",
    "integrator_euler_fs_error_threshold": "--checkpoint-controller-integrator-euler-fs-error-threshold",
    "integrator_condition_max": "--checkpoint-controller-integrator-condition-max",
    "integrator_euler_min_time_fraction": "--checkpoint-controller-integrator-euler-min-time-fraction",
    "integrator_euler_site_span_max": "--checkpoint-controller-integrator-euler-site-span-max",
    "integrator_euler_primary_density_span_max": "--checkpoint-controller-integrator-euler-primary-density-span-max",
    "integrator_euler_energy_span_max": "--checkpoint-controller-integrator-euler-energy-span-max",
    "progress_early_stop_min_checkpoint": "--checkpoint-controller-progress-early-stop-min-checkpoint",
    "progress_early_stop_site_error_mean_max": "--checkpoint-controller-progress-early-stop-site-error-mean-max",
    "progress_early_stop_primary_density_error_mean_max": "--checkpoint-controller-progress-early-stop-primary-density-error-mean-max",
    "progress_early_stop_energy_error_mean_max": "--checkpoint-controller-progress-early-stop-energy-error-mean-max",
    "progress_early_stop_site_span_max": "--checkpoint-controller-progress-early-stop-site-span-max",
    "progress_early_stop_primary_density_span_max": "--checkpoint-controller-progress-early-stop-primary-density-span-max",
    "progress_early_stop_energy_span_max": "--checkpoint-controller-progress-early-stop-energy-span-max",
    "prune_miss_threshold": "--checkpoint-controller-prune-miss-threshold",
    "prune_loss_threshold": "--checkpoint-controller-prune-loss-threshold",
    "prune_theta_block_tol": "--checkpoint-controller-prune-theta-block-tol",
    "prune_state_jump_l2_tol": "--checkpoint-controller-prune-state-jump-l2-tol",
    "prune_safe_miss_increase_tol": "--checkpoint-controller-prune-safe-miss-increase-tol",
    "prune_schur_monotonicity_tol": "--checkpoint-controller-prune-schur-monotonicity-tol",
    "prune_loss_norm_epsilon": "--checkpoint-controller-prune-loss-norm-epsilon",
    "prune_differential_miss_tol": "--checkpoint-controller-prune-differential-miss-tol",
    "prune_projection_mode": "--checkpoint-controller-prune-projection-mode",
    "prune_projection_trust_radius": "--checkpoint-controller-prune-projection-trust-radius",
    "prune_projection_regularization": "--checkpoint-controller-prune-projection-regularization",
    "prune_ray_distance_tol": "--checkpoint-controller-prune-ray-distance-tol",
    "prune_shadow_score_increase_tol": "--checkpoint-controller-prune-shadow-score-increase-tol",
    "prune_schur_ladder_local_radius": "--checkpoint-controller-prune-schur-ladder-local-radius",
    "prune_projection_rounds": "--checkpoint-controller-prune-projection-rounds",
    "prune_projection_max_active_runtime": "--checkpoint-controller-prune-projection-max-active-runtime",
    "prune_shadow_horizon_steps": "--checkpoint-controller-prune-shadow-horizon-steps",
    "prune_persistence_window": "--checkpoint-controller-prune-persistence-window",
    "prune_persistence_required": "--checkpoint-controller-prune-persistence-required",
    "prune_appended_origin_target_policy": "--checkpoint-controller-prune-appended-origin-target-policy",
    "prune_appended_origin_grace_steps": "--checkpoint-controller-prune-appended-origin-grace-steps",
    "prune_initial_scaffold_grace_steps": "--checkpoint-controller-prune-initial-scaffold-grace-steps",
    "prune_appended_origin_bias_scale": "--checkpoint-controller-prune-appended-origin-bias-scale",
    "prune_appended_origin_bias_max_factor": "--checkpoint-controller-prune-appended-origin-bias-max-factor",
}

_BOOL_SETTING_FLAGS: dict[str, tuple[str, str]] = {
    "strict_qpu_faithful": ("--checkpoint-controller-strict-qpu-faithful", ""),
    "lock_fixed_manifold": ("--lock-fixed-manifold", ""),
    "allow_repeats": ("--allow-repeats", ""),
    "append_enabled": (
        "--checkpoint-controller-append-enabled",
        "--no-checkpoint-controller-append-enabled",
    ),
    "append_no_harm_guard_enabled": (
        "--checkpoint-controller-append-no-harm-guard-enabled",
        "--no-checkpoint-controller-append-no-harm-guard-enabled",
    ),
    "prune_appended_origin_bias_enabled": (
        "--checkpoint-controller-prune-appended-origin-bias-enabled",
        "--no-checkpoint-controller-prune-appended-origin-bias-enabled",
    ),
    "prune_no_harm_guard_enabled": (
        "--checkpoint-controller-prune-no-harm-guard-enabled",
        "--no-checkpoint-controller-prune-no-harm-guard-enabled",
    ),
    "prune_high_miss_differential_enabled": (
        "--checkpoint-controller-prune-high-miss-differential-enabled",
        "--no-checkpoint-controller-prune-high-miss-differential-enabled",
    ),
    "prune_shadow_enabled": (
        "--checkpoint-controller-prune-shadow-enabled",
        "--no-checkpoint-controller-prune-shadow-enabled",
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_metadata(case: DynamicsBenchmarkCase) -> dict[str, Any]:
    return dict(case.metadata) if isinstance(case.metadata, Mapping) else {}


def seed_lock_metadata(case: DynamicsBenchmarkCase) -> dict[str, Any]:
    metadata = _case_metadata(case)
    raw = metadata.get("seed_lock", {})
    seed_lock = dict(raw) if isinstance(raw, Mapping) else {}
    artifact = Path(case.artifact_json).expanduser()
    artifact_hash = seed_lock.get("seed_artifact_sha256")
    if artifact_hash in {None, ""} and artifact.exists() and artifact.is_file():
        artifact_hash = _sha256_file(artifact)
    group_id = seed_lock.get("same_seed_comparator_group_id") or metadata.get(
        "same_seed_comparator_group_id"
    )
    if group_id in {None, ""}:
        group_id = case.case_id
    out = {
        "seed_lock_schema": DYNAMICS_TABLE_LOCK_SEED_MANIFEST_SCHEMA,
        "table_lock_source": seed_lock.get("table_lock_source", DYNAMICS_TABLE_LOCK_SOURCE),
        "same_seed_comparator_group_id": str(group_id),
        "static_seed_artifact_json": str(case.artifact_json),
        "static_seed_artifact_sha256": artifact_hash,
        "seed_selection_policy": seed_lock.get(
            "seed_selection_policy",
            metadata.get("seed_selection_policy", "current_best_static_adapt_seed"),
        ),
        "selected_static_seed_source": seed_lock.get(
            "selected_static_seed_source",
            metadata.get("selected_static_seed_source", "latest_phase3_summary_or_ledger"),
        ),
        "same_seed_validation_status": "hash_recorded" if artifact_hash else "hash_missing",
    }
    optional_seed_track_keys = (
        "seed_track",
        "static_algorithm_id",
        "static_seed_display_label",
        "hh_regime_id",
        "hh_static_case_id",
        "hh_u_over_t",
        "hh_lambda",
        "n_ph_work",
        "n_ph_ref",
        "source_artifact_json",
        "source_artifact_sha256",
        "normalized_seed_artifact_json",
        "normalized_seed_artifact_sha256",
        "static_abs_delta_e",
        "static_parameter_count",
        "runtime_loadability_status",
        "latest_phase3_source_artifact_missing_locally",
    )
    for key in optional_seed_track_keys:
        value = seed_lock.get(key, None)
        if value is not None and value != "":
            out[key] = value
    return json_safe(out)


def table_lock_provenance_for_case(case: DynamicsBenchmarkCase) -> dict[str, Any]:
    lock = seed_lock_metadata(case)
    return {
        "seed_lock": lock,
        **lock,
    }


def with_class_settings_lock_manifest(
    case: DynamicsBenchmarkCase,
    *,
    manifest_path: str | Path | None,
    require_locked: bool = False,
) -> DynamicsBenchmarkCase:
    if manifest_path is None and not require_locked:
        return case
    metadata = _case_metadata(case)
    if manifest_path is not None:
        metadata[_CLASS_SETTINGS_MANIFEST_METADATA_KEY] = str(manifest_path)
    if require_locked:
        metadata[_REQUIRE_CLASS_SETTINGS_METADATA_KEY] = True
    return replace(case, metadata=metadata)


def class_settings_manifest_path(case: DynamicsBenchmarkCase) -> Path | None:
    raw = _case_metadata(case).get(_CLASS_SETTINGS_MANIFEST_METADATA_KEY)
    if raw in {None, ""}:
        return None
    return Path(str(raw)).expanduser().resolve()


def require_locked_class_settings(case: DynamicsBenchmarkCase) -> bool:
    raw = _case_metadata(case).get(_REQUIRE_CLASS_SETTINGS_METADATA_KEY, False)
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def load_class_settings_lock_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("class settings lock manifest must be a JSON object")
    schema = payload.get("schema")
    if schema not in {DYNAMICS_CLASS_SETTINGS_LOCK_MANIFEST_SCHEMA, "dynamics_class_settings_candidate_v1"}:
        raise ValueError(
            f"unsupported class settings manifest schema {schema!r}; "
            f"expected {DYNAMICS_CLASS_SETTINGS_LOCK_MANIFEST_SCHEMA!r}"
        )
    if schema == "dynamics_class_settings_candidate_v1":
        payload = {
            "schema": DYNAMICS_CLASS_SETTINGS_LOCK_MANIFEST_SCHEMA,
            "lock_status": payload.get("lock_status", "candidate_unlocked"),
            "settings": [dict(payload)],
        }
    validate_class_settings_lock_manifest(
        payload,
        require_exact_controller_classes=bool(
            payload.get("require_canonical_controller_classes", False)
            or payload.get("canonical_controller_policy_classes_required", False)
        ),
        require_all_table_i_algorithm_classes=bool(
            payload.get("require_all_table_i_algorithm_classes", False)
        ),
    )
    return dict(payload)


def _settings_entries(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("settings", [])
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, Mapping)]
    if isinstance(raw, Mapping):
        return [dict(raw)]
    return []


def _entry_variant_id(entry: Mapping[str, Any]) -> str | None:
    raw = entry.get("variant_id", None)
    return None if raw in {None, ""} else str(raw)


def _entry_key(entry: Mapping[str, Any]) -> tuple[str, str, str, str | None]:
    return (
        str(entry.get("tuning_class", "")),
        str(entry.get("algorithm_id", "")),
        str(entry.get("settings_kind", "")),
        _entry_variant_id(entry),
    )


def _required_algorithm_settings_from_manifest(payload: Mapping[str, Any]) -> set[tuple[str, str]]:
    """Return algorithm/settings-kind pairs that a manifest marks required."""

    required: set[tuple[str, str]] = set()
    if bool(payload.get("require_all_table_i_algorithm_classes", False)):
        required.update(
            (str(algorithm_id), str(settings_kind))
            for algorithm_id, settings_kind in DYNAMICS_TABLE_I_ALGORITHM_SETTINGS_KIND.items()
        )
    raw = payload.get("required_algorithm_settings", [])
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            algorithm_id = str(item.get("algorithm_id", "")).strip()
            settings_kind = str(
                item.get("settings_kind", DYNAMICS_TABLE_I_ALGORITHM_SETTINGS_KIND.get(algorithm_id, ""))
            ).strip()
            if algorithm_id and settings_kind:
                required.add((algorithm_id, settings_kind))
    return required


def _manifest_requires_algorithm_settings(
    payload: Mapping[str, Any],
    *,
    algorithm_id: str,
    settings_kind: str | None,
) -> bool:
    required = _required_algorithm_settings_from_manifest(payload)
    if not required:
        return False
    wanted_kind = str(settings_kind or DYNAMICS_TABLE_I_ALGORITHM_SETTINGS_KIND.get(str(algorithm_id), ""))
    return (str(algorithm_id), wanted_kind) in required


def _case_metadata_requires_algorithm_settings(
    case: DynamicsBenchmarkCase,
    *,
    algorithm_id: str,
    settings_kind: str | None,
) -> bool:
    raw = _case_metadata(case).get(_REQUIRE_ALGORITHM_CLASS_SETTINGS_METADATA_KEY, False)
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on", "all", "table_i"}
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        wanted_kind = str(settings_kind or DYNAMICS_TABLE_I_ALGORITHM_SETTINGS_KIND.get(str(algorithm_id), ""))
        for item in raw:
            if isinstance(item, Mapping):
                item_algorithm = str(item.get("algorithm_id", ""))
                item_kind = str(item.get("settings_kind", DYNAMICS_TABLE_I_ALGORITHM_SETTINGS_KIND.get(item_algorithm, "")))
                if item_algorithm == str(algorithm_id) and item_kind == wanted_kind:
                    return True
            elif str(item) == str(algorithm_id):
                return True
        return False
    return bool(raw)


def validate_class_settings_lock_manifest(
    payload: Mapping[str, Any],
    *,
    require_exact_controller_classes: bool = False,
    require_all_table_i_algorithm_classes: bool = False,
) -> dict[str, Any]:
    """Validate a class-settings lock/candidate manifest without running physics.

    The canonical Paper-II controller lock is intentionally class-level: exactly
    one full-controller policy may be promoted for each of the three coarse
    Hamiltonian classes.  This helper also rejects case-specific settings
    payload keys that would make a settings ID benchmark-point-specific.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("class settings lock manifest must be a JSON object")
    schema = payload.get("schema")
    if schema != DYNAMICS_CLASS_SETTINGS_LOCK_MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported class settings manifest schema {schema!r}; "
            f"expected {DYNAMICS_CLASS_SETTINGS_LOCK_MANIFEST_SCHEMA!r}"
        )
    entries = _settings_entries(payload)
    required_algorithm_settings = _required_algorithm_settings_from_manifest(payload)
    if require_all_table_i_algorithm_classes:
        required_algorithm_settings.update(
            (str(algorithm_id), str(settings_kind))
            for algorithm_id, settings_kind in DYNAMICS_TABLE_I_ALGORITHM_SETTINGS_KIND.items()
        )
    seen: set[tuple[str, str, str, str | None]] = set()
    controller_classes: set[str] = set()
    covered_required: set[tuple[str, str, str]] = set()
    candidate_only_entries = 0
    for entry in entries:
        tuning_class = normalize_dynamics_tuning_class(entry.get("tuning_class", "")) or ""
        if tuning_class not in DYNAMICS_COARSE_TUNING_CLASSES:
            known = ", ".join(DYNAMICS_COARSE_TUNING_CLASSES)
            raise ValueError(
                f"class settings entry has unsupported tuning_class {tuning_class!r}; "
                f"expected one of {known}"
            )
        algorithm_id = str(entry.get("algorithm_id", "")).strip()
        if not algorithm_id:
            raise ValueError("class settings entry is missing algorithm_id")
        settings_kind = str(entry.get("settings_kind", DYNAMICS_SETTINGS_KIND_CONTROLLER)).strip()
        if settings_kind not in DYNAMICS_CLASS_SETTINGS_KINDS:
            known = ", ".join(DYNAMICS_CLASS_SETTINGS_KINDS)
            raise ValueError(
                f"class settings entry has unsupported settings_kind {settings_kind!r}; "
                f"expected one of {known}"
            )
        key = _entry_key({**entry, "settings_kind": settings_kind})
        if key in seen:
            raise ValueError(f"duplicate class-settings entry for {key}")
        seen.add(key)
        settings_payload = entry.get("settings_payload", {}) or {}
        if not isinstance(settings_payload, Mapping):
            raise ValueError(f"class settings entry {key} settings_payload must be a mapping")
        forbidden = sorted(set(str(item) for item in settings_payload) & DYNAMICS_FORBIDDEN_TUNING_ID_KEYS)
        if forbidden:
            joined = ", ".join(forbidden)
            raise ValueError(f"class settings entry {key} contains case-specific settings keys: {joined}")
        if (algorithm_id, settings_kind) in required_algorithm_settings:
            covered_required.add((tuning_class, algorithm_id, settings_kind))
        if bool(entry.get("candidate_only_not_promoted", False)):
            candidate_only_entries += 1
        if (
            settings_kind == DYNAMICS_SETTINGS_KIND_CONTROLLER
            and algorithm_id == DYNAMICS_CANONICAL_CONTROLLER_ALGORITHM_ID
            and _entry_variant_id(entry) in {None, DYNAMICS_CANONICAL_CONTROLLER_VARIANT_ID}
            and bool(entry.get("class_tuned_result_locked", False))
        ):
            controller_classes.add(tuning_class)
            if require_exact_controller_classes and not bool(
                entry.get("strict_online_feedback_exact_free", False)
            ):
                raise ValueError(
                    f"canonical controller entry for {tuning_class!r} is not "
                    "strict_online_feedback_exact_free"
                )
    if require_exact_controller_classes and controller_classes != set(DYNAMICS_COARSE_TUNING_CLASSES):
        raise ValueError(
            "canonical class-settings manifest must contain exactly one locked "
            "dyn_controller_full/controller policy for each of "
            f"{DYNAMICS_COARSE_TUNING_CLASSES}; got {sorted(controller_classes)}"
        )
    missing_required: list[tuple[str, str, str]] = []
    for algorithm_id, settings_kind in sorted(required_algorithm_settings):
        for tuning_class in DYNAMICS_COARSE_TUNING_CLASSES:
            key = (str(tuning_class), str(algorithm_id), str(settings_kind))
            if key not in covered_required:
                missing_required.append(key)
    if missing_required:
        raise ValueError(
            "class-settings manifest is missing required all-algorithm class entries: "
            + ", ".join("/".join(key) for key in missing_required)
        )
    return json_safe(
        {
            "entry_count": int(len(entries)),
            "canonical_controller_classes": sorted(controller_classes),
            "requires_exact_controller_classes": bool(require_exact_controller_classes),
            "required_algorithm_settings": [
                {"algorithm_id": algorithm_id, "settings_kind": settings_kind}
                for algorithm_id, settings_kind in sorted(required_algorithm_settings)
            ],
            "required_algorithm_class_entry_count": int(len(covered_required)),
            "candidate_only_entry_count": int(candidate_only_entries),
        }
    )


def class_settings_entry_for_case(
    case: DynamicsBenchmarkCase,
    *,
    algorithm_id: str,
    settings_kind: str | None = None,
    variant_id: str | None = None,
) -> dict[str, Any] | None:
    manifest = class_settings_manifest_path(case)
    if manifest is None:
        if require_locked_class_settings(case):
            raise ValueError(f"case {case.case_id}: --require-locked-class-settings set without a manifest")
        return None
    payload = load_class_settings_lock_manifest(manifest)
    lock_status = str(payload.get("lock_status", "")).strip().lower()
    require_locked = require_locked_class_settings(case)
    if require_locked and lock_status not in _LOCKED_MANIFEST_STATUSES:
        raise ValueError(f"class settings manifest {manifest} is not locked; lock_status={lock_status!r}")
    wanted_class = dynamics_tuning_class(case)
    for entry in _settings_entries(payload):
        entry_class = normalize_dynamics_tuning_class(entry.get("tuning_class", "")) or ""
        if str(entry_class) != str(wanted_class):
            continue
        if str(entry.get("algorithm_id", "")) != str(algorithm_id):
            continue
        if settings_kind is not None and str(entry.get("settings_kind", settings_kind)) != str(settings_kind):
            continue
        entry_variant = entry.get("variant_id", None)
        if variant_id not in {None, ""} and entry_variant not in {None, "", variant_id}:
            continue
        if require_locked_class_settings(case) and not bool(entry.get("class_tuned_result_locked", False)):
            raise ValueError(
                f"class settings entry for {wanted_class}/{algorithm_id} is not class_tuned_result_locked"
            )
        out = dict(entry)
        out.setdefault("lock_status", lock_status)
        out.setdefault("manifest_path", str(manifest))
        return out
    wanted_kind = str(settings_kind or DYNAMICS_TABLE_I_ALGORITHM_SETTINGS_KIND.get(str(algorithm_id), DYNAMICS_SETTINGS_KIND_CONTROLLER))
    required_by_manifest = _manifest_requires_algorithm_settings(
        payload,
        algorithm_id=str(algorithm_id),
        settings_kind=wanted_kind,
    )
    required_by_case = _case_metadata_requires_algorithm_settings(
        case,
        algorithm_id=str(algorithm_id),
        settings_kind=wanted_kind,
    )
    required_controller = (
        str(wanted_kind) == DYNAMICS_SETTINGS_KIND_CONTROLLER
        and str(algorithm_id) == DYNAMICS_CANONICAL_CONTROLLER_ALGORITHM_ID
    )
    if required_by_manifest or required_by_case or (require_locked and required_controller):
        raise ValueError(
            f"class settings manifest {manifest} has no entry for "
            f"tuning_class={wanted_class!r}, algorithm_id={algorithm_id!r}, settings_kind={wanted_kind!r}"
        )
    return None


def build_locked_or_default_tuning_provenance(
    *,
    case: DynamicsBenchmarkCase,
    algorithm_id: str,
    settings_kind: str,
    settings_payload: Mapping[str, Any] | None = None,
    settings_source: str = DYNAMICS_CLASS_TUNING_DEFAULT_SOURCE,
    variant_id: str | None = None,
    locked: bool = False,
) -> dict[str, Any]:
    entry = class_settings_entry_for_case(
        case,
        algorithm_id=algorithm_id,
        settings_kind=settings_kind,
        variant_id=variant_id,
    )
    manifest = class_settings_manifest_path(case)
    if entry is not None:
        settings_payload = dict(entry.get("settings_payload", {}) or {})
        settings_source = str(entry.get("settings_source", DYNAMICS_CLASS_SETTINGS_LOCK_SOURCE))
        locked = bool(entry.get("class_tuned_result_locked", True))
    provenance = build_dynamics_tuning_provenance(
        case=case,
        algorithm_id=algorithm_id,
        settings_kind=settings_kind,
        settings_payload=dict(settings_payload or {}),
        settings_source=settings_source,
        variant_id=variant_id,
        locked=bool(locked),
    )
    if manifest is not None:
        provenance["class_settings_lock_manifest"] = str(manifest)
    if entry is not None:
        provenance.update(
            {
                "class_settings_entry_present": True,
                "class_settings_entry_settings_id": entry.get("settings_id"),
                "class_settings_entry_lock_status": entry.get("lock_status"),
                "class_settings_entry_promotion_status": entry.get("promotion_status"),
                "class_settings_candidate_only_not_promoted": bool(
                    entry.get("candidate_only_not_promoted", False)
                ),
                "class_settings_search_profile_id": entry.get("search_profile_id"),
                "class_settings_calibration_status": entry.get("calibration_status"),
                "class_tuned_status": entry.get(
                    "class_tuned_status",
                    "locked_coarse_class_tuned" if locked else "candidate_not_promoted",
                ),
            }
        )
        for source_key, target_key in (
            ("source_candidate_json", "class_settings_entry_source_candidate_json"),
            ("source_summary_json", "class_settings_entry_source_summary_json"),
            ("selected_trial_number", "class_settings_selected_trial_number"),
            ("strict_online_feedback_exact_free", "strict_online_feedback_exact_free"),
            ("candidate_record_json", "class_settings_candidate_record_json"),
        ):
            if entry.get(source_key, None) not in {None, ""}:
                provenance[target_key] = entry.get(source_key)
    return json_safe(provenance)


def case_with_class_settings_overrides(
    case: DynamicsBenchmarkCase,
    *,
    algorithm_id: str,
    settings_kind: str | None = None,
    variant_id: str | None = None,
) -> DynamicsBenchmarkCase:
    """Return ``case`` with class-level algorithm settings merged into metadata.

    Comparator and fixed-McLachlan runners consume settings through metadata
    hooks, not controller CLI tokens.  The class-settings manifest remains the
    source of truth for validation/provenance; this helper only exposes the
    selected class payload to existing runner configuration hooks.
    """

    wanted_kind = str(settings_kind or DYNAMICS_TABLE_I_ALGORITHM_SETTINGS_KIND.get(str(algorithm_id), ""))
    entry = class_settings_entry_for_case(
        case,
        algorithm_id=str(algorithm_id),
        settings_kind=wanted_kind or None,
        variant_id=variant_id,
    )
    if entry is None:
        return case
    payload = entry.get("settings_payload", {}) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"class settings entry for {algorithm_id}/{wanted_kind} settings_payload must be a mapping"
        )
    metadata = _case_metadata(case)
    metadata.update(dict(payload))
    effective_entries = metadata.get(_EFFECTIVE_CLASS_SETTINGS_METADATA_KEY, {})
    if not isinstance(effective_entries, Mapping):
        effective_entries = {}
    entry_key = f"{algorithm_id}:{wanted_kind or entry.get('settings_kind', '')}"
    effective_entries = dict(effective_entries)
    effective_entries[entry_key] = json_safe(
        {
            "algorithm_id": str(algorithm_id),
            "settings_kind": wanted_kind or entry.get("settings_kind"),
            "tuning_class": entry.get("tuning_class"),
            "settings_source": entry.get("settings_source", DYNAMICS_CLASS_SETTINGS_LOCK_SOURCE),
            "settings_id": entry.get("settings_id"),
            "candidate_only_not_promoted": bool(entry.get("candidate_only_not_promoted", False)),
            "class_tuned_result_locked": bool(entry.get("class_tuned_result_locked", False)),
            "settings_payload_keys": sorted(str(key) for key in payload),
        }
    )
    metadata[_EFFECTIVE_CLASS_SETTINGS_METADATA_KEY] = effective_entries
    return replace(case, metadata=json_safe(metadata))


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def controller_cli_tokens_from_settings(settings_payload: Mapping[str, Any]) -> list[str]:
    tokens: list[str] = []
    for key, value in dict(settings_payload).items():
        if value in {None, ""}:
            continue
        key = str(key)
        if key in _BOOL_SETTING_FLAGS:
            true_flag, false_flag = _BOOL_SETTING_FLAGS[key]
            flag = true_flag if _bool_value(value) else false_flag
            if flag:
                tokens.append(flag)
            continue
        flag = _CONTROLLER_SETTING_CLI_FLAGS.get(key)
        if flag is None:
            continue
        tokens.extend([flag, str(value)])
    return tokens


def controller_cli_tokens_for_case(
    case: DynamicsBenchmarkCase,
    *,
    algorithm_id: str,
    settings_kind: str = "controller",
    variant_id: str | None = None,
) -> list[str]:
    entry = class_settings_entry_for_case(
        case,
        algorithm_id=algorithm_id,
        settings_kind=settings_kind,
        variant_id=variant_id,
    )
    if entry is None:
        return []
    payload = entry.get("settings_payload", {})
    return controller_cli_tokens_from_settings(payload if isinstance(payload, Mapping) else {})


def validate_same_seed_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        prov = row.get("provenance", {}) if isinstance(row.get("provenance"), Mapping) else {}
        group = prov.get("same_seed_comparator_group_id")
        if group in {None, ""}:
            continue
        bucket = groups.setdefault(str(group), {"hashes": set(), "artifacts": set(), "case_ids": set()})
        for key, target in (
            ("static_seed_artifact_sha256", "hashes"),
            ("static_seed_artifact_json", "artifacts"),
            ("case_id", "case_ids"),
        ):
            value = prov.get(key, row.get(key))
            if value not in {None, ""}:
                bucket[target].add(str(value))
    bad = {
        group: {name: sorted(values) for name, values in data.items()}
        for group, data in groups.items()
        if len(data.get("hashes", set())) > 1 or len(data.get("artifacts", set())) > 1
    }
    return json_safe(
        {
            "same_seed_group_count": int(len(groups)),
            "passed": not bad,
            "bad_groups": bad,
            "groups": {
                group: {name: sorted(values) for name, values in data.items()}
                for group, data in groups.items()
            },
        }
    )


_SHARED_BENCHMARK_SURFACE_KEYS: tuple[str, ...] = (
    "static_seed_artifact_sha256",
    "drive_signature",
    "time_grid_signature",
    "observable_set_signature",
    "diagnostic_reference_signature",
    "compile_target_signature",
)


def _row_mapping(row: Any) -> Mapping[str, Any]:
    if hasattr(row, "to_dict"):
        payload = row.to_dict()
        return payload if isinstance(payload, Mapping) else {}
    return row if isinstance(row, Mapping) else {}


def _benchmark_surface_for_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = row.get("benchmark_surface")
    if isinstance(direct, Mapping):
        return direct
    provenance = row.get("provenance", {})
    if isinstance(provenance, Mapping) and isinstance(provenance.get("benchmark_surface"), Mapping):
        return provenance["benchmark_surface"]
    return {}


def validate_shared_benchmark_surface_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    required_algorithm_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate same-seed/same-grid/same-drive metadata across comparator rows."""

    required = {str(item) for item in required_algorithm_ids}
    groups: dict[str, dict[str, Any]] = {}
    violations: list[dict[str, Any]] = []
    for raw in rows:
        row = _row_mapping(raw)
        algorithm_id = str(row.get("algorithm_id", ""))
        surface = _benchmark_surface_for_row(row)
        if not surface:
            violations.append(
                {
                    "violation": "missing_benchmark_surface",
                    "algorithm_id": algorithm_id,
                    "case_id": row.get("case_id"),
                }
            )
            continue
        group_id = str(surface.get("same_seed_comparator_group_id", ""))
        if not group_id:
            violations.append(
                {
                    "violation": "missing_same_seed_comparator_group_id",
                    "algorithm_id": algorithm_id,
                    "case_id": row.get("case_id"),
                }
            )
            continue
        group = groups.setdefault(
            group_id,
            {
                "algorithm_ids": [],
                "case_ids": [],
                "values": {key: {} for key in _SHARED_BENCHMARK_SURFACE_KEYS},
            },
        )
        group["algorithm_ids"].append(algorithm_id)
        group["case_ids"].append(str(row.get("case_id", "")))
        for key in _SHARED_BENCHMARK_SURFACE_KEYS:
            value = surface.get(key)
            if value in {None, ""}:
                violations.append(
                    {
                        "violation": "missing_surface_key",
                        "group_id": group_id,
                        "algorithm_id": algorithm_id,
                        "key": key,
                    }
                )
                continue
            group["values"][key].setdefault(str(value), []).append(algorithm_id)
    for group_id, group in groups.items():
        present = {str(item) for item in group.get("algorithm_ids", [])}
        missing_required = sorted(required - present)
        if missing_required:
            violations.append(
                {
                    "violation": "missing_required_algorithm_ids",
                    "group_id": group_id,
                    "missing_algorithm_ids": missing_required,
                }
            )
        for key, values in group.get("values", {}).items():
            if len(values) > 1:
                violations.append(
                    {
                        "violation": "shared_surface_mismatch",
                        "group_id": group_id,
                        "key": key,
                        "value_count": int(len(values)),
                        "values": values,
                    }
                )
    return json_safe(
        {
            "schema": "dynamics_shared_benchmark_surface_validation_v1",
            "passed": not violations,
            "violations": violations,
            "group_count": int(len(groups)),
            "groups": groups,
            "required_algorithm_ids": sorted(required),
        }
    )


__all__ = [
    "DYNAMICS_CLASS_SETTINGS_LOCK_MANIFEST_SCHEMA",
    "DYNAMICS_TABLE_LOCK_SEED_MANIFEST_SCHEMA",
    "DYNAMICS_TABLE_LOCK_SOURCE",
    "build_locked_or_default_tuning_provenance",
    "case_with_class_settings_overrides",
    "class_settings_entry_for_case",
    "controller_cli_tokens_for_case",
    "controller_cli_tokens_from_settings",
    "load_class_settings_lock_manifest",
    "require_locked_class_settings",
    "seed_lock_metadata",
    "table_lock_provenance_for_case",
    "validate_class_settings_lock_manifest",
    "validate_shared_benchmark_surface_rows",
    "validate_same_seed_rows",
    "with_class_settings_lock_manifest",
]
