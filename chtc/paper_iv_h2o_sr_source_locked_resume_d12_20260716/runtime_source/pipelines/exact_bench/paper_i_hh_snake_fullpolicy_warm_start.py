#!/usr/bin/env python3
"""Warm-start audit for the Paper-I HH SNAKE full-policy Optuna campaign."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


OUTPUT_SLUG = "paper_i_hh_snake_fullpolicy_20260622_v1"
FULL_POLICY_PROFILE = "hh_routea_full_policy_v1"
HH_OPTUNA_SOURCE = REPO_ROOT / "pipelines/exact_bench/hh_cost_energy_optuna.py"
DEFAULT_VISIBLE_BASELINE_JSON = (
    REPO_ROOT
    / "MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_native200_qiskit_table_plot_alignment_20260622.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "raw_outputs/local_smokes" / OUTPUT_SLUG
REGIME_ORDER = (
    "weak-weak",
    "intermediate-weak",
    "strong-weak-u8",
    "weak-strong",
    "intermediate-strong",
    "strong-strong-u8",
)
REGIME_SPECS: dict[str, dict[str, Any]] = {
    "weak-weak": {"u": 0.25, "lambda": 0.25, "n_ph_work": 2, "n_ph_ref": 2},
    "intermediate-weak": {"u": 1.25, "lambda": 0.25, "n_ph_work": 2, "n_ph_ref": 2},
    "strong-weak-u8": {"u": 8.0, "lambda": 0.25, "n_ph_work": 2, "n_ph_ref": 2},
    "weak-strong": {"u": 0.25, "lambda": 1.25, "n_ph_work": 4, "n_ph_ref": 4},
    "intermediate-strong": {"u": 1.25, "lambda": 1.25, "n_ph_work": 4, "n_ph_ref": 4},
    "strong-strong-u8": {"u": 8.0, "lambda": 1.25, "n_ph_work": 4, "n_ph_ref": 4},
}
LEGACY_AMBIGUOUS_LABELS = frozenset({"strong-weak", "strong_weak", "strong-strong", "strong_strong"})
_DIRECT_REGIME_LABELS = frozenset(REGIME_ORDER)
_U8_ALIASES = {
    "strong_weak_u8": "strong-weak-u8",
    "strong-weak-u8": "strong-weak-u8",
    "u8-strong-weak": "strong-weak-u8",
    "strong_strong_u8": "strong-strong-u8",
    "strong-strong-u8": "strong-strong-u8",
    "u8-strong-strong": "strong-strong-u8",
}
_BASE_ENQUEUE_PARAMS: dict[str, Any] = {
    "base_preset": "resolved_default",
    "adapt_max_depth": 30,
    "selector_geometry_mode": "base",
    "runtime_split_mode": "shortlist_pauli_children_v1",
    "batching_mode": "on",
    "repeats_mode": "base",
    "selection_cost_mode": "marrakesh_graph_span_v1",
    "motif_mode": "off",
    "phase1_prune_mode": "live",
    "phase0_pilot_profile": "base",
    "phase0_pilot_records_profile": "base",
    "maturity_shortlist_profile": "base",
    "phase1_shortlist_size_profile": "base",
    "phase2_shortlist_fraction_profile": "base",
    "phase2_shortlist_size_profile": "base",
    "maturity_shot_profile": "base",
    "phase_live_profile": "base",
    "prune_witness_profile": "base",
    "prune_prefilter_profile": "base",
    "adapt_window_profile": "base",
    "adapt_history_window_profile": "base",
    "geometry_window_profile": "base",
    "backend_cost_weight_profile": "base",
    "phase2_w_shot_profile": "base",
    "phase2_rho_profile": "base",
    "spsa_profile": "current",
    "ml_candidate_profile": "base",
    "phase1_prune_fraction_profile": "base",
    "batch_near_degenerate_profile": "base",
    "batch_rank_tol_profile": "base",
    "batch_additivity_tol_profile": "base",
}


def _repo_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else REPO_ROOT / raw


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _literal_constants(source_path: Path, wanted: set[str]) -> dict[str, Any]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    constants: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    constants[target.id] = ast.literal_eval(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in wanted:
            constants[node.target.id] = ast.literal_eval(node.value)
    return constants


@lru_cache(maxsize=1)
def _full_policy_param_options() -> dict[str, list[str]]:
    constants = _literal_constants(HH_OPTUNA_SOURCE, {"_FULL_POLICY_PARAM_OPTIONS"})
    raw_options = constants.get("_FULL_POLICY_PARAM_OPTIONS", {})
    if not isinstance(raw_options, Mapping):
        raise ValueError(f"Unable to read _FULL_POLICY_PARAM_OPTIONS from {HH_OPTUNA_SOURCE}.")
    return {str(key): [str(item) for item in values] for key, values in raw_options.items()}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _norm_label(raw: str) -> str:
    return str(raw).strip().lower().replace("_", "-")


def _maybe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _source_settings(source_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    settings = source_payload.get("settings", {})
    return settings if isinstance(settings, Mapping) else {}


def _source_u_value(source_payload: Mapping[str, Any]) -> float | None:
    settings = _source_settings(source_payload)
    hamiltonian = source_payload.get("hamiltonian", {})
    if not isinstance(hamiltonian, Mapping):
        hamiltonian = {}
    return _maybe_float(settings.get("u") if settings.get("u") is not None else hamiltonian.get("u"))


def _source_work_cutoff(source_payload: Mapping[str, Any]) -> int | None:
    settings = _source_settings(source_payload)
    value = _maybe_float(settings.get("n_ph_max"))
    return None if value is None else int(round(value))


def canonical_campaign_regime(
    raw: str,
    *,
    source_payload: Mapping[str, Any] | None = None,
    source_path: str | Path | None = None,
) -> str:
    """Resolve current campaign labels while rejecting ambiguous legacy labels."""
    key = _norm_label(raw)
    raw_key = str(raw).strip().lower()
    if key in _DIRECT_REGIME_LABELS:
        return key
    if raw_key in _U8_ALIASES:
        return _U8_ALIASES[raw_key]
    if key in _U8_ALIASES:
        return _U8_ALIASES[key]

    if raw_key in LEGACY_AMBIGUOUS_LABELS or key in {_norm_label(x) for x in LEGACY_AMBIGUOUS_LABELS}:
        u_value = _source_u_value(source_payload or {})
        work_cutoff = _source_work_cutoff(source_payload or {})
        source_text = "" if source_path is None else str(source_path).lower()
        if key == "strong-weak" and u_value == 8.0 and (work_cutoff == 2 or "strong_weak_u8" in source_text):
            return "strong-weak-u8"
        if key == "strong-strong" and u_value == 8.0 and (work_cutoff == 4 or "strong_strong_u8" in source_text):
            return "strong-strong-u8"
        raise ValueError(
            f"Ambiguous legacy HH regime label {raw!r}; use explicit intermediate-* or strong-*-u8 labels."
        )

    valid = ", ".join(REGIME_ORDER)
    raise ValueError(f"Unknown HH regime label {raw!r}; expected one of: {valid}")


def _choice_for_value(name: str, value: Any) -> str:
    choices = list(_full_policy_param_options()[name])
    if value is None:
        return "base"
    value_text = str(value)
    if value_text in choices:
        return value_text
    value_float = _maybe_float(value)
    if value_float is None:
        return value_text if value_text in choices else "base"
    numeric_choices: list[tuple[str, float]] = []
    for choice in choices:
        choice_float = _maybe_float(choice)
        if choice_float is None:
            continue
        if math.isclose(value_float, choice_float, rel_tol=1e-9, abs_tol=1e-12):
            return str(choice)
        numeric_choices.append((str(choice), float(choice_float)))
    if not numeric_choices:
        return "base"
    if value_float == 0.0:
        return min(numeric_choices, key=lambda item: abs(item[1]))[0]
    return min(
        numeric_choices,
        key=lambda item: abs(math.log10(max(abs(item[1]), 1e-300)) - math.log10(max(abs(value_float), 1e-300))),
    )[0]


def _setting_choice(settings: Mapping[str, Any], field_name: str, setting_name: str, fallback: Any = None) -> str:
    value = settings.get(setting_name, fallback)
    return _choice_for_value(field_name, value)


def _float_slug(value: float) -> str:
    return f"{float(value):.3e}".replace("+", "").replace("-", "m")


def _resolved_study_name(*, lane: str, epsilon_abs_delta_e: float, study_name_prefix: str | None = None) -> str:
    suffix = f"{lane}_eps_{_float_slug(float(epsilon_abs_delta_e))}"
    prefix = "" if study_name_prefix in {None, ""} else str(study_name_prefix).strip()
    return suffix if prefix == "" else f"{prefix}_{suffix}"


def _baseline_trial_params(settings: Mapping[str, Any]) -> dict[str, Any]:
    spsa = settings.get("adapt_spsa", {})
    if not isinstance(spsa, Mapping):
        spsa = {}
    phase2_batch_target = settings.get("phase2_batch_target_size")
    phase2_batch_cap = settings.get("phase2_batch_size_cap")
    phase3_batch_target = settings.get("phase3_batch_target_size", phase2_batch_target)
    phase3_batch_cap = settings.get("phase3_batch_size_cap", phase2_batch_cap)
    phase2_near = settings.get("phase2_batch_near_degenerate_ratio")
    phase3_near = settings.get("phase3_batch_near_degenerate_ratio", phase2_near)
    phase2_rank = settings.get("phase2_batch_rank_rel_tol")
    phase3_rank = settings.get("phase3_batch_rank_rel_tol", phase2_rank)
    phase2_add = settings.get("phase2_batch_additivity_tol")
    phase3_add = settings.get("phase3_batch_additivity_tol", phase2_add)

    params = dict(_BASE_ENQUEUE_PARAMS)
    params["adapt_max_depth"] = int(round(_maybe_float(settings.get("adapt_max_depth")) or 30))
    params.update(
        {
            "full_phase0_pilot_max_records": _setting_choice(
                settings, "full_phase0_pilot_max_records", "phase0_pilot_max_records"
            ),
            "full_phase1_shortlist_size": _setting_choice(settings, "full_phase1_shortlist_size", "phase1_shortlist_size"),
            "full_phase2_shortlist_fraction": _setting_choice(
                settings, "full_phase2_shortlist_fraction", "phase2_shortlist_fraction"
            ),
            "full_phase2_shortlist_size": _setting_choice(settings, "full_phase2_shortlist_size", "phase2_shortlist_size"),
            "full_adapt_window_size": _setting_choice(settings, "full_adapt_window_size", "adapt_window_size"),
            "full_phase3_geometry_window_size": _setting_choice(
                settings, "full_phase3_geometry_window_size", "phase3_geometry_window_size"
            ),
            "full_phase2_w_shot": _setting_choice(settings, "full_phase2_w_shot", "phase2_w_shot"),
            "full_phase2_rho": _setting_choice(settings, "full_phase2_rho", "phase2_rho"),
            "full_phase2_batch_target_size": _choice_for_value("full_phase2_batch_target_size", phase2_batch_target),
            "full_phase2_batch_size_cap": _choice_for_value("full_phase2_batch_size_cap", phase2_batch_cap),
            "full_phase3_batch_target_size": _choice_for_value("full_phase3_batch_target_size", phase3_batch_target),
            "full_phase3_batch_size_cap": _choice_for_value("full_phase3_batch_size_cap", phase3_batch_cap),
            "full_batch_near_degenerate_ratio": _choice_for_value("full_batch_near_degenerate_ratio", phase3_near),
            "full_batch_rank_rel_tol": _choice_for_value("full_batch_rank_rel_tol", phase3_rank),
            "full_batch_additivity_tol": _choice_for_value("full_batch_additivity_tol", phase3_add),
            "full_phase3_batch_order_selection_mode": _setting_choice(
                settings, "full_phase3_batch_order_selection_mode", "phase3_batch_order_selection_mode"
            ),
            "full_phase3_batch_order_max_permutations": _setting_choice(
                settings, "full_phase3_batch_order_max_permutations", "phase3_batch_order_max_permutations"
            ),
            "full_phase2_frontier_ratio": _setting_choice(settings, "full_phase2_frontier_ratio", "phase2_frontier_ratio"),
            "full_phase3_frontier_ratio": _setting_choice(settings, "full_phase3_frontier_ratio", "phase3_frontier_ratio"),
            "full_phase3_tie_beam_score_ratio": _setting_choice(
                settings, "full_phase3_tie_beam_score_ratio", "phase3_tie_beam_score_ratio"
            ),
            "full_phase3_tie_beam_abs_tol": _setting_choice(
                settings, "full_phase3_tie_beam_abs_tol", "phase3_tie_beam_abs_tol"
            ),
            "full_phase3_tie_beam_max_branches": _setting_choice(
                settings, "full_phase3_tie_beam_max_branches", "phase3_tie_beam_max_branches"
            ),
            "full_phase1_prune_mode": _setting_choice(settings, "full_phase1_prune_mode", "phase1_prune_mode"),
            "full_phase1_prune_fraction": _setting_choice(
                settings, "full_phase1_prune_fraction", "phase1_prune_fraction"
            ),
            "full_phase1_prune_min_candidates": _setting_choice(
                settings, "full_phase1_prune_min_candidates", "phase1_prune_min_candidates"
            ),
            "full_phase1_prune_max_candidates": _setting_choice(
                settings, "full_phase1_prune_max_candidates", "phase1_prune_max_candidates"
            ),
            "full_phase1_prune_max_regression": _setting_choice(
                settings, "full_phase1_prune_max_regression", "phase1_prune_max_regression"
            ),
            "full_phase1_prune_tolerance_mode": _setting_choice(
                settings, "full_phase1_prune_tolerance_mode", "phase1_prune_tolerance_mode"
            ),
            "full_phase1_prune_tolerance_shot_coeff": _setting_choice(
                settings, "full_phase1_prune_tolerance_shot_coeff", "phase1_prune_tolerance_shot_coeff"
            ),
            "full_phase1_prune_tolerance_screen_coeff": _setting_choice(
                settings, "full_phase1_prune_tolerance_screen_coeff", "phase1_prune_tolerance_screen_coeff"
            ),
            "full_phase1_prune_tolerance_chem": _setting_choice(
                settings, "full_phase1_prune_tolerance_chem", "phase1_prune_tolerance_chem"
            ),
            "full_phase1_prune_tolerance_rel_coeff": _setting_choice(
                settings, "full_phase1_prune_tolerance_rel_coeff", "phase1_prune_tolerance_rel_coeff"
            ),
            "full_phase1_prune_retained_gain_ratio": _setting_choice(
                settings, "full_phase1_prune_retained_gain_ratio", "phase1_prune_retained_gain_ratio"
            ),
            "full_phase1_prune_protect_steps": _setting_choice(
                settings, "full_phase1_prune_protect_steps", "phase1_prune_protect_steps"
            ),
            "full_phase1_prune_stale_age": _setting_choice(
                settings, "full_phase1_prune_stale_age", "phase1_prune_stale_age"
            ),
            "full_phase1_prune_stagnation_threshold": _setting_choice(
                settings, "full_phase1_prune_stagnation_threshold", "phase1_prune_stagnation_threshold"
            ),
            "full_phase1_prune_small_theta_abs": _setting_choice(
                settings, "full_phase1_prune_small_theta_abs", "phase1_prune_small_theta_abs"
            ),
            "full_phase1_prune_small_theta_relative": _setting_choice(
                settings, "full_phase1_prune_small_theta_relative", "phase1_prune_small_theta_relative"
            ),
            "full_phase1_prune_cooldown_steps": _setting_choice(
                settings, "full_phase1_prune_cooldown_steps", "phase1_prune_cooldown_steps"
            ),
            "full_phase1_prune_local_window_size": _setting_choice(
                settings, "full_phase1_prune_local_window_size", "phase1_prune_local_window_size"
            ),
            "full_phase1_prune_recovery_trust_radius": _setting_choice(
                settings, "full_phase1_prune_recovery_trust_radius", "phase1_prune_recovery_trust_radius"
            ),
            "full_phase1_prune_old_fraction": _setting_choice(
                settings, "full_phase1_prune_old_fraction", "phase1_prune_old_fraction"
            ),
            "full_phase1_prune_checkpoint_period": _setting_choice(
                settings, "full_phase1_prune_checkpoint_period", "phase1_prune_checkpoint_period"
            ),
            "full_phase1_prune_live_min_depth": _setting_choice(
                settings, "full_phase1_prune_live_min_depth", "phase1_prune_live_min_depth"
            ),
            "full_phase1_prune_maturity_threshold": _setting_choice(
                settings, "full_phase1_prune_maturity_threshold", "phase1_prune_maturity_threshold"
            ),
            "full_phase1_prune_snr_threshold": _setting_choice(
                settings, "full_phase1_prune_snr_threshold", "phase1_prune_snr_threshold"
            ),
            "full_spsa_maxiter": _choice_for_value(
                "full_spsa_maxiter",
                settings.get("adapt_maxiter", settings.get("adapt_final_refit_maxiter")),
            ),
            "full_spsa_a": _choice_for_value("full_spsa_a", spsa.get("a")),
            "full_spsa_c": _choice_for_value("full_spsa_c", spsa.get("c")),
            "full_spsa_alpha": _choice_for_value("full_spsa_alpha", spsa.get("alpha")),
            "full_spsa_gamma": _choice_for_value("full_spsa_gamma", spsa.get("gamma")),
            "full_spsa_A": _choice_for_value("full_spsa_A", spsa.get("A")),
            "full_spsa_avg_last": _choice_for_value("full_spsa_avg_last", spsa.get("avg_last")),
            "full_spsa_eval_repeats": _choice_for_value("full_spsa_eval_repeats", spsa.get("eval_repeats")),
            "full_spsa_callback_every": _choice_for_value("full_spsa_callback_every", spsa.get("callback_every")),
        }
    )
    return params


def _normalise_enqueue_params(params: Mapping[str, Any], *, lane: str = "canonical") -> dict[str, Any]:
    del lane
    options = _full_policy_param_options()
    normalised = dict(_BASE_ENQUEUE_PARAMS)
    normalised.update({str(k): v for k, v in params.items() if not str(k).startswith("full_")})
    for name, choices in options.items():
        value = params.get(name, "base")
        normalised[name] = value if value in choices else "base"
    return normalised


def _visible_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    display = row.get("display", {})
    metrics: dict[str, Any] = dict(display if isinstance(display, Mapping) else {})
    for key in (
        "one_minus_F_display",
        "fidelity_status",
        "plot_marker_delta_e",
        "plot_marker_k_pl",
        "marker_minus_table_delta_e",
        "S_prefix_status",
        "S_note",
    ):
        if key in row:
            metrics[key] = row.get(key)
    return metrics


def _phase3_policy_snapshot() -> dict[str, Any]:
    source_path = REPO_ROOT / "pipelines/static_adapt/optimization/phase3_policy_optuna.py"
    try:
        wanted = {
            "_PHASE3_BATCH_SELECTION_MODE_CHOICES",
            "_PHASE3_BATCH_PREFILTER_MODE_CHOICES",
            "_DEFAULT_PHASE1_PRUNE_POLICY",
            "_SPSA_SCHEDULE_PARAM_RANGES",
            "_SPSA_CLI_OPTIONS",
        }
        constants = _literal_constants(source_path, wanted)
    except Exception as exc:  # pragma: no cover - defensive for stripped runtime bundles.
        return {
            "source_module": "pipelines.static_adapt.optimization.phase3_policy_optuna",
            "source_path": str(source_path),
            "source_read_error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "source_module": "pipelines.static_adapt.optimization.phase3_policy_optuna",
        "source_path": str(source_path),
        "phase3_batch_selection_mode_choices": list(constants.get("_PHASE3_BATCH_SELECTION_MODE_CHOICES", ())),
        "phase3_batch_prefilter_mode_choices": list(constants.get("_PHASE3_BATCH_PREFILTER_MODE_CHOICES", ())),
        "phase1_prune_policy": constants.get("_DEFAULT_PHASE1_PRUNE_POLICY"),
        "spsa_schedule_param_ranges": {
            str(k): list(v) for k, v in dict(constants.get("_SPSA_SCHEDULE_PARAM_RANGES", {})).items()
        },
        "spsa_cli_options": dict(constants.get("_SPSA_CLI_OPTIONS", {})),
    }


def fixed_locks_manifest() -> dict[str, Any]:
    return {
        "paper": "Paper-I",
        "table": "HH Table III",
        "method": "SNAKE",
        "speed_surface_profile": FULL_POLICY_PROFILE,
        "route_id": "route_a",
        "static_meta_feature_profile": "paper_i_production_v1",
        "adapt_pool_requested": "full_meta",
        "pool_policy": "full_meta_minus_hva",
        "class_filter_json": "agent_guidance/static-adapt/hh_full_meta_minus_hva_class_filter.json",
        "cutoff_contract": {
            "weak/intermediate/strong weak-Holstein": {"n_ph_work": 2, "n_ph_ref": 2},
            "weak/intermediate/strong strong-Holstein": {"n_ph_work": 4, "n_ph_ref": 4},
        },
        "reference_contract": {
            "displayed_error": "same_cutoff_exact_reference",
            "exact_manifest": "MATH/paper_facing/paper_I_static_scaffold/hh_ed_reference_manifest_20260614.json",
        },
        "route_locks": {
            "phase3_batch_selection_mode": "reduced_plane",
            "phase3_batch_prefilter_mode": "off",
            "phase3_batch_order_selection_mode_default": "finite_step_v1",
            "phase1_prune_enabled": True,
            "phase1_prune_policy": "recoverability_ladder_v1",
            "phase1_prune_mode_default": "both",
            "phase3_selector_policy": "algebraic_nested_v1",
            "phase3_selector_geometry_mode": "reduced",
            "phase2_novelty_mode": "collective_span_v1",
            "hardware_resolution_mode": "ideal",
        },
        "budget_reporting": {
            "native200_table_equivalent_spsa_maxiter": 200,
            "different_spsa_budget_trials_are_diagnostic": True,
        },
    }


def build_warm_start_audit(
    *,
    baseline_json: Path = DEFAULT_VISIBLE_BASELINE_JSON,
    regimes: Sequence[str] = REGIME_ORDER,
    optuna_storage: str | None = None,
    study_name_prefix: str | None = None,
    lane: str = "canonical",
    epsilon_abs_delta_e: float = 1e9,
) -> dict[str, Any]:
    baseline_path = _repo_path(baseline_json)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Visible baseline JSON is missing: {baseline_path}")
    baseline_payload = _load_json(baseline_path)
    rows = baseline_payload.get("snake_rows")
    if not isinstance(rows, list):
        raise ValueError(f"{baseline_path} does not contain a snake_rows list.")

    requested = tuple(canonical_campaign_regime(label) for label in regimes)
    regime_entries: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or str(row.get("method", "")).strip().upper() != "SNAKE":
            continue
        source_raw = row.get("source_json")
        if not source_raw:
            raise ValueError(f"snake_rows[{index}] is missing source_json.")
        source_path = _repo_path(str(source_raw))
        if not source_path.exists():
            raise FileNotFoundError(f"Missing local SNAKE baseline source JSON for row {index}: {source_path}")
        plot_source_path: Path | None = None
        if row.get("plot_source_json") not in {None, ""}:
            plot_source_path = _repo_path(str(row.get("plot_source_json")))
            if not plot_source_path.exists():
                raise FileNotFoundError(f"Missing local SNAKE plot source JSON for row {index}: {plot_source_path}")

        source_payload = _load_json(source_path)
        campaign_regime = canonical_campaign_regime(
            str(row.get("regime", "")),
            source_payload=source_payload,
            source_path=source_raw,
        )
        if campaign_regime not in requested:
            continue
        if campaign_regime in regime_entries:
            raise ValueError(f"Duplicate current SNAKE baseline row for {campaign_regime}.")

        source_sha256 = _sha256(source_path)
        expected_sha256 = row.get("source_sha256")
        if expected_sha256 not in {None, ""} and str(expected_sha256) != source_sha256:
            raise ValueError(
                f"Hash mismatch for {campaign_regime}: visible row records {expected_sha256}, local source is {source_sha256}."
            )
        settings = _source_settings(source_payload)
        extracted_params = _baseline_trial_params(settings)
        enqueue_params = _normalise_enqueue_params(extracted_params, lane=lane)
        regime_entries[campaign_regime] = {
            "display_regime": str(row.get("regime")),
            "campaign_regime": campaign_regime,
            "row_index": int(index),
            "method": "SNAKE",
            "source_json": str(source_raw),
            "source_json_abs": str(source_path),
            "source_sha256": source_sha256,
            "visible_source_sha256": expected_sha256,
            "plot_source_json": None if plot_source_path is None else str(row.get("plot_source_json")),
            "plot_source_json_abs": None if plot_source_path is None else str(plot_source_path),
            "plot_source_sha256": None if plot_source_path is None else _sha256(plot_source_path),
            "visible_metrics": _visible_metrics(row),
            "settings_extract": {
                "u": settings.get("u"),
                "g_ep": settings.get("g_ep"),
                "n_ph_work": settings.get("n_ph_max"),
                "adapt_max_depth": settings.get("adapt_max_depth"),
                "adapt_final_refit_maxiter": settings.get("adapt_final_refit_maxiter"),
                "phase1_prune_mode": settings.get("phase1_prune_mode"),
                "phase1_prune_policy": settings.get("phase1_prune_policy"),
                "phase2_batch_target_size": settings.get("phase2_batch_target_size"),
                "phase3_batch_target_size": settings.get("phase3_batch_target_size"),
            },
            "raw_extracted_params": extracted_params,
            "enqueue_params": [enqueue_params],
        }

    missing = [regime for regime in requested if regime not in regime_entries]
    if missing:
        raise ValueError(f"Missing current visible SNAKE baselines for regimes: {missing}")

    ordered_entries = {regime: regime_entries[regime] for regime in requested}
    return {
        "schema": "paper_i_hh_snake_fullpolicy_warm_start_audit_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "output_slug": OUTPUT_SLUG,
        "baseline_source": {
            "path": str(baseline_json),
            "abs_path": str(baseline_path),
            "sha256": _sha256(baseline_path),
            "schema": baseline_payload.get("schema"),
            "updated_date": baseline_payload.get("updated_date"),
        },
        "fixed_locks": fixed_locks_manifest(),
        "regime_order": list(requested),
        "regime_specs": {regime: REGIME_SPECS[regime] for regime in requested},
        "optuna": {
            "storage": optuna_storage,
            "study_name_prefix": study_name_prefix,
            "lane": str(lane),
            "epsilon_abs_delta_e": float(epsilon_abs_delta_e),
            "resolved_study_names": {
                regime: _resolved_study_name(
                    lane=str(lane),
                    epsilon_abs_delta_e=float(epsilon_abs_delta_e),
                    study_name_prefix=(f"{study_name_prefix}__{regime}" if study_name_prefix else None),
                )
                for regime in requested
            },
        },
        "regimes": ordered_entries,
    }


def enqueue_manifest_from_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    regimes = audit.get("regimes", {})
    if not isinstance(regimes, Mapping):
        raise ValueError("audit.regimes must be a mapping.")
    return {
        "schema": "paper_i_hh_snake_fullpolicy_enqueue_params_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "output_slug": OUTPUT_SLUG,
        "baseline_source": audit.get("baseline_source"),
        "fixed_locks": audit.get("fixed_locks"),
        "regimes": {
            str(regime): {
                "source_json": entry.get("source_json"),
                "source_sha256": entry.get("source_sha256"),
                "visible_metrics": entry.get("visible_metrics"),
                "enqueue_params": list(entry.get("enqueue_params", [])),
            }
            for regime, entry in regimes.items()
            if isinstance(entry, Mapping)
        },
    }


def search_space_manifest_from_audit(
    audit: Mapping[str, Any],
    *,
    warm_start_audit_json: Path | None = None,
    enqueue_params_json: Path | None = None,
    search_space_manifest_json: Path | None = None,
) -> dict[str, Any]:
    regimes = audit.get("regimes", {})
    if not isinstance(regimes, Mapping):
        raise ValueError("audit.regimes must be a mapping.")
    generated_artifact_paths = {
        "warm_start_audit_json": None if warm_start_audit_json is None else str(warm_start_audit_json),
        "enqueue_params_json": None if enqueue_params_json is None else str(enqueue_params_json),
        "search_space_manifest_json": None if search_space_manifest_json is None else str(search_space_manifest_json),
    }
    return {
        "schema": "paper_i_hh_snake_fullpolicy_search_space_manifest_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "output_slug": OUTPUT_SLUG,
        "speed_surface_profile": FULL_POLICY_PROFILE,
        "fixed_locks": audit.get("fixed_locks"),
        "sampled_ranges": {
            key: list(values) for key, values in _full_policy_param_options().items()
        },
        "phase3_policy_optuna_reuse": _phase3_policy_snapshot(),
        "baseline_provenance": audit.get("baseline_source"),
        "enqueued_priors": {
            str(regime): {
                "source_json": entry.get("source_json"),
                "source_sha256": entry.get("source_sha256"),
                "visible_metrics": entry.get("visible_metrics"),
                "enqueue_params": entry.get("enqueue_params"),
            }
            for regime, entry in regimes.items()
            if isinstance(entry, Mapping)
        },
        "optuna": audit.get("optuna"),
        "generated_artifact_paths": generated_artifact_paths,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_regimes(raw: str | None) -> tuple[str, ...]:
    if raw in {None, ""}:
        return REGIME_ORDER
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-json", type=Path, default=DEFAULT_VISIBLE_BASELINE_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--warm-start-audit-json", type=Path, default=None)
    parser.add_argument("--enqueue-params-json", type=Path, default=None)
    parser.add_argument("--search-space-manifest-json", type=Path, default=None)
    parser.add_argument("--regimes", type=str, default=",".join(REGIME_ORDER))
    parser.add_argument("--optuna-storage", type=str, default=None)
    parser.add_argument("--study-name-prefix", type=str, default=OUTPUT_SLUG)
    parser.add_argument("--lane", type=str, default="canonical")
    parser.add_argument("--epsilon-abs-delta-e", type=float, default=1e9)
    parser.add_argument("--print-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    output_dir = Path(args.output_dir)
    warm_start_audit_json = args.warm_start_audit_json or output_dir / "warm_start_audit.json"
    enqueue_params_json = args.enqueue_params_json or output_dir / "enqueue_params.json"
    search_space_manifest_json = args.search_space_manifest_json or output_dir / "search_space_manifest.json"
    audit = build_warm_start_audit(
        baseline_json=Path(args.baseline_json),
        regimes=_parse_regimes(args.regimes),
        optuna_storage=args.optuna_storage,
        study_name_prefix=args.study_name_prefix,
        lane=str(args.lane),
        epsilon_abs_delta_e=float(args.epsilon_abs_delta_e),
    )
    enqueue_manifest = enqueue_manifest_from_audit(audit)
    search_manifest = search_space_manifest_from_audit(
        audit,
        warm_start_audit_json=warm_start_audit_json,
        enqueue_params_json=enqueue_params_json,
        search_space_manifest_json=search_space_manifest_json,
    )
    if bool(args.print_only):
        print(json.dumps(_json_safe({"audit": audit, "enqueue_manifest": enqueue_manifest, "search_manifest": search_manifest}), indent=2, sort_keys=True))
        return
    _write_json(warm_start_audit_json, audit)
    _write_json(enqueue_params_json, enqueue_manifest)
    _write_json(search_space_manifest_json, search_manifest)
    print(
        json.dumps(
            {
                "warm_start_audit_json": str(warm_start_audit_json),
                "enqueue_params_json": str(enqueue_params_json),
                "search_space_manifest_json": str(search_space_manifest_json),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
