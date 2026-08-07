#!/usr/bin/env python3
"""Build the Paper-I HH SNAKE full-policy warm-start manifest.

The manifest is intentionally candidate-only: it records the current visible
Table-III SNAKE sources and emits per-regime Optuna enqueue rows for
``hh_routea_full_policy_v1``.  It does not update manuscript tables or promote
artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench import hh_cost_energy_optuna as optuna_hh  # noqa: E402

DEFAULT_ALIGNMENT_JSON = (
    REPO_ROOT
    / "MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_native200_qiskit_table_plot_alignment_20260622.json"
)
EXPECTED_REGIMES = (
    "weak-weak",
    "intermediate-weak",
    "strong-weak-u8",
    "weak-strong",
    "intermediate-strong",
    "strong-strong-u8",
)
REGIME_EXPECTED_HH = {
    "weak-weak": {"u": 0.25, "lambda": 0.25, "n_ph_work": 2, "n_ph_ref": 2},
    "intermediate-weak": {"u": 1.25, "lambda": 0.25, "n_ph_work": 2, "n_ph_ref": 2},
    "strong-weak-u8": {"u": 8.0, "lambda": 0.25, "n_ph_work": 2, "n_ph_ref": 2},
    "weak-strong": {"u": 0.25, "lambda": 1.25, "n_ph_work": 4, "n_ph_ref": 4},
    "intermediate-strong": {"u": 1.25, "lambda": 1.25, "n_ph_work": 4, "n_ph_ref": 4},
    "strong-strong-u8": {"u": 8.0, "lambda": 1.25, "n_ph_work": 4, "n_ph_ref": 4},
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nested(root: Mapping[str, Any], *keys: str) -> Any:
    current: Any = root
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _nearly_equal(left: Any, right: Any, *, atol: float = 1e-9) -> bool:
    try:
        return abs(float(left) - float(right)) <= float(atol)
    except Exception:
        return False


def _lambda_from_settings(settings: Mapping[str, Any]) -> float | None:
    raw = settings.get("lambda_ep", settings.get("lambda"))
    if raw is not None:
        try:
            return float(raw)
        except Exception:
            return None
    try:
        g_ep = float(settings["g_ep"])
        t = float(settings.get("t", 1.0))
        omega0 = float(settings.get("omega0", 1.0))
        return 2.0 * g_ep * g_ep / (t * omega0)
    except Exception:
        return None


def _choice_value(name: str, raw: Any) -> str:
    if raw is None:
        return "base"
    options = optuna_hh._FULL_POLICY_PARAM_OPTIONS[name]
    text = str(raw)
    if text in options:
        return text
    try:
        raw_float = float(raw)
    except Exception:
        raw_float = None
    if raw_float is not None:
        for option in options:
            if option == "base":
                continue
            try:
                if abs(float(option) - raw_float) <= max(1e-12, abs(raw_float) * 1e-9):
                    return str(option)
            except Exception:
                continue
    return "base"


def _settings_param(settings: Mapping[str, Any], key: str, param_name: str) -> str:
    return _choice_value(param_name, settings.get(key))


def _spsa_param(settings: Mapping[str, Any], key: str, param_name: str) -> str:
    spsa = settings.get("adapt_spsa")
    if not isinstance(spsa, Mapping):
        spsa = _nested(settings, "adapt_vqe", "adapt_spsa")
    if not isinstance(spsa, Mapping):
        return "base"
    return _choice_value(param_name, spsa.get(key))


def _canonical_regime(row: Mapping[str, Any], result_payload: Mapping[str, Any]) -> str:
    raw = str(row.get("regime") or "").strip().lower()
    settings = result_payload.get("settings") if isinstance(result_payload.get("settings"), Mapping) else {}
    u_value = settings.get("u")
    lambda_value = _lambda_from_settings(settings)
    if raw in {"weak-weak", "intermediate-weak", "weak-strong", "intermediate-strong"}:
        return raw
    if raw == "strong-weak" and _nearly_equal(u_value, 8.0) and _nearly_equal(lambda_value, 0.25):
        return "strong-weak-u8"
    if raw == "strong-strong" and _nearly_equal(u_value, 8.0) and _nearly_equal(lambda_value, 1.25):
        return "strong-strong-u8"
    raise ValueError(
        f"Ambiguous or unsupported visible regime {row.get('regime')!r}; use explicit U=8 labels for current strong rows."
    )


def _validate_identity(regime: str, result_payload: Mapping[str, Any]) -> list[str]:
    settings = result_payload.get("settings") if isinstance(result_payload.get("settings"), Mapping) else {}
    adapt_vqe = result_payload.get("adapt_vqe") if isinstance(result_payload.get("adapt_vqe"), Mapping) else {}
    expected = REGIME_EXPECTED_HH[regime]
    errors: list[str] = []
    if not _nearly_equal(settings.get("u"), expected["u"]):
        errors.append(f"{regime}: U mismatch {settings.get('u')!r} != {expected['u']!r}")
    lambda_value = _lambda_from_settings(settings)
    if not _nearly_equal(lambda_value, expected["lambda"]):
        errors.append(f"{regime}: lambda mismatch {lambda_value!r} != {expected['lambda']!r}")
    if int(settings.get("n_ph_max", -1)) != int(expected["n_ph_work"]):
        errors.append(f"{regime}: working cutoff mismatch {settings.get('n_ph_max')!r} != {expected['n_ph_work']!r}")
    if str(settings.get("adapt_pool")) != "full_meta":
        errors.append(f"{regime}: adapt_pool is not full_meta")
    class_filter = str(settings.get("adapt_pool_class_filter_json") or "")
    if class_filter and "hh_full_meta_minus_hva_class_filter" not in class_filter:
        errors.append(f"{regime}: unexpected class-filter provenance {class_filter!r}")
    route_identity = adapt_vqe.get("static_route_identity")
    if isinstance(route_identity, Mapping) and str(route_identity.get("route_id")) not in {"route_a", "None"}:
        errors.append(f"{regime}: static route is not route_a")
    return errors


def _enqueue_params_from_result(result_payload: Mapping[str, Any]) -> dict[str, Any]:
    settings = result_payload.get("settings") if isinstance(result_payload.get("settings"), Mapping) else {}
    return {
        "base_preset": "resolved_default",
        "adapt_max_depth": 30,
        "selector_geometry_mode": "base",
        "runtime_split_mode": "shortlist_pauli_children_v1",
        "batching_mode": "on",
        "repeats_mode": "base",
        "selection_cost_mode": "marrakesh_graph_span_v1",
        "motif_mode": "off",
        "phase1_prune_mode": "live",
        "spsa_profile": "current",
        "full_phase0_pilot_max_records": _settings_param(settings, "phase0_pilot_max_records", "full_phase0_pilot_max_records"),
        "full_phase1_shortlist_size": _settings_param(settings, "phase1_shortlist_size", "full_phase1_shortlist_size"),
        "full_phase2_shortlist_fraction": _settings_param(settings, "phase2_shortlist_fraction", "full_phase2_shortlist_fraction"),
        "full_phase2_shortlist_size": _settings_param(settings, "phase2_shortlist_size", "full_phase2_shortlist_size"),
        "full_adapt_window_size": _settings_param(settings, "adapt_window_size", "full_adapt_window_size"),
        "full_phase3_geometry_window_size": _settings_param(settings, "phase3_geometry_window_size", "full_phase3_geometry_window_size"),
        "full_phase2_w_shot": _settings_param(settings, "phase2_w_shot", "full_phase2_w_shot"),
        "full_phase2_rho": _settings_param(settings, "phase2_rho", "full_phase2_rho"),
        "full_phase2_batch_target_size": _settings_param(settings, "phase2_batch_target_size", "full_phase2_batch_target_size"),
        "full_phase2_batch_size_cap": _settings_param(settings, "phase2_batch_size_cap", "full_phase2_batch_size_cap"),
        "full_batch_near_degenerate_ratio": _settings_param(settings, "phase2_batch_near_degenerate_ratio", "full_batch_near_degenerate_ratio"),
        "full_batch_rank_rel_tol": _settings_param(settings, "phase2_batch_rank_rel_tol", "full_batch_rank_rel_tol"),
        "full_batch_additivity_tol": _settings_param(settings, "phase2_batch_additivity_tol", "full_batch_additivity_tol"),
        "full_phase3_batch_order_max_permutations": _settings_param(settings, "phase3_batch_order_max_permutations", "full_phase3_batch_order_max_permutations"),
        "full_phase2_frontier_ratio": _settings_param(settings, "phase2_frontier_ratio", "full_phase2_frontier_ratio"),
        "full_phase3_frontier_ratio": _settings_param(settings, "phase3_frontier_ratio", "full_phase3_frontier_ratio"),
        "full_phase3_tie_beam_score_ratio": _settings_param(settings, "phase3_tie_beam_score_ratio", "full_phase3_tie_beam_score_ratio"),
        "full_phase3_tie_beam_abs_tol": _settings_param(settings, "phase3_tie_beam_abs_tol", "full_phase3_tie_beam_abs_tol"),
        "full_phase3_tie_beam_max_branches": _settings_param(settings, "phase3_tie_beam_max_branches", "full_phase3_tie_beam_max_branches"),
        "full_phase1_prune_fraction": _settings_param(settings, "phase1_prune_fraction", "full_phase1_prune_fraction"),
        "full_phase1_prune_min_candidates": _settings_param(settings, "phase1_prune_min_candidates", "full_phase1_prune_min_candidates"),
        "full_phase1_prune_max_candidates": _settings_param(settings, "phase1_prune_max_candidates", "full_phase1_prune_max_candidates"),
        "full_phase1_prune_max_regression": _settings_param(settings, "phase1_prune_max_regression", "full_phase1_prune_max_regression"),
        "full_phase1_prune_tolerance_mode": _settings_param(settings, "phase1_prune_tolerance_mode", "full_phase1_prune_tolerance_mode"),
        "full_phase1_prune_tolerance_shot_coeff": _settings_param(settings, "phase1_prune_tolerance_shot_coeff", "full_phase1_prune_tolerance_shot_coeff"),
        "full_phase1_prune_tolerance_screen_coeff": _settings_param(settings, "phase1_prune_tolerance_screen_coeff", "full_phase1_prune_tolerance_screen_coeff"),
        "full_phase1_prune_tolerance_chem": _settings_param(settings, "phase1_prune_tolerance_chem", "full_phase1_prune_tolerance_chem"),
        "full_phase1_prune_tolerance_rel_coeff": _settings_param(settings, "phase1_prune_tolerance_rel_coeff", "full_phase1_prune_tolerance_rel_coeff"),
        "full_phase1_prune_retained_gain_ratio": _settings_param(settings, "phase1_prune_retained_gain_ratio", "full_phase1_prune_retained_gain_ratio"),
        "full_phase1_prune_protect_steps": _settings_param(settings, "phase1_prune_protect_steps", "full_phase1_prune_protect_steps"),
        "full_phase1_prune_stale_age": _settings_param(settings, "phase1_prune_stale_age", "full_phase1_prune_stale_age"),
        "full_phase1_prune_stagnation_threshold": _settings_param(settings, "phase1_prune_stagnation_threshold", "full_phase1_prune_stagnation_threshold"),
        "full_phase1_prune_small_theta_abs": _settings_param(settings, "phase1_prune_small_theta_abs", "full_phase1_prune_small_theta_abs"),
        "full_phase1_prune_small_theta_relative": _settings_param(settings, "phase1_prune_small_theta_relative", "full_phase1_prune_small_theta_relative"),
        "full_phase1_prune_cooldown_steps": _settings_param(settings, "phase1_prune_cooldown_steps", "full_phase1_prune_cooldown_steps"),
        "full_phase1_prune_local_window_size": _settings_param(settings, "phase1_prune_local_window_size", "full_phase1_prune_local_window_size"),
        "full_phase1_prune_recovery_trust_radius": _settings_param(settings, "phase1_prune_recovery_trust_radius", "full_phase1_prune_recovery_trust_radius"),
        "full_phase1_prune_old_fraction": _settings_param(settings, "phase1_prune_old_fraction", "full_phase1_prune_old_fraction"),
        "full_phase1_prune_checkpoint_period": _settings_param(settings, "phase1_prune_checkpoint_period", "full_phase1_prune_checkpoint_period"),
        "full_phase1_prune_live_min_depth": _settings_param(settings, "phase1_prune_live_min_depth", "full_phase1_prune_live_min_depth"),
        "full_phase1_prune_maturity_threshold": _settings_param(settings, "phase1_prune_maturity_threshold", "full_phase1_prune_maturity_threshold"),
        "full_phase1_prune_snr_threshold": _settings_param(settings, "phase1_prune_snr_threshold", "full_phase1_prune_snr_threshold"),
        "full_phase1_prune_collapse_peak_abs_min": _settings_param(settings, "phase1_prune_collapse_peak_abs_min", "full_phase1_prune_collapse_peak_abs_min"),
        "full_phase1_prune_collapse_current_abs_max": _settings_param(settings, "phase1_prune_collapse_current_abs_max", "full_phase1_prune_collapse_current_abs_max"),
        "full_phase1_prune_collapse_ratio": _settings_param(settings, "phase1_prune_collapse_ratio", "full_phase1_prune_collapse_ratio"),
        "full_phase1_prune_collapse_min_abs_drop": _settings_param(settings, "phase1_prune_collapse_min_abs_drop", "full_phase1_prune_collapse_min_abs_drop"),
        "full_phase1_prune_collapse_min_observations": _settings_param(settings, "phase1_prune_collapse_min_observations", "full_phase1_prune_collapse_min_observations"),
        "full_spsa_maxiter": _choice_value(
            "full_spsa_maxiter",
            settings.get("adapt_maxiter", settings.get("adapt_final_refit_maxiter")),
        ),
        "full_spsa_a": _spsa_param(settings, "a", "full_spsa_a"),
        "full_spsa_c": _spsa_param(settings, "c", "full_spsa_c"),
        "full_spsa_alpha": _spsa_param(settings, "alpha", "full_spsa_alpha"),
        "full_spsa_gamma": _spsa_param(settings, "gamma", "full_spsa_gamma"),
        "full_spsa_A": _spsa_param(settings, "A", "full_spsa_A"),
        "full_spsa_avg_last": _spsa_param(settings, "avg_last", "full_spsa_avg_last"),
        "full_spsa_eval_repeats": _spsa_param(settings, "eval_repeats", "full_spsa_eval_repeats"),
        "full_spsa_callback_every": _spsa_param(settings, "callback_every", "full_spsa_callback_every"),
    }


def build_warm_start_manifest(alignment_json: Path = DEFAULT_ALIGNMENT_JSON) -> dict[str, Any]:
    alignment_path = Path(alignment_json)
    if not alignment_path.is_absolute():
        alignment_path = REPO_ROOT / alignment_path
    alignment = _load_json(alignment_path)
    rows = [row for row in alignment.get("snake_rows", []) if isinstance(row, Mapping) and row.get("method") == "SNAKE"]
    regimes: dict[str, Any] = {}
    failures: list[str] = []
    for row in rows:
        source_json = Path(str(row.get("source_json") or ""))
        if not source_json.is_absolute():
            source_json = REPO_ROOT / source_json
        if not source_json.exists():
            failures.append(f"{row.get('regime')}: missing source_json {source_json}")
            continue
        digest = _sha256(source_json)
        expected_digest = row.get("source_sha256")
        if expected_digest not in {None, ""} and str(expected_digest) != digest:
            failures.append(f"{row.get('regime')}: sha256 mismatch for {source_json}")
            continue
        result_payload = _load_json(source_json)
        regime = _canonical_regime(row, result_payload)
        failures.extend(_validate_identity(regime, result_payload))
        params = _enqueue_params_from_result(result_payload)
        regimes[regime] = {
            "source_regime_label": row.get("regime"),
            "source_json": str(source_json.relative_to(REPO_ROOT) if source_json.is_relative_to(REPO_ROOT) else source_json),
            "source_sha256": digest,
            "visible_metrics": dict(row.get("display") or {}),
            "one_minus_F_display": row.get("one_minus_F_display"),
            "plot_source_json": row.get("plot_source_json"),
            "enqueue_params": [
                {
                    "source": "paper_i_hh_native200_qiskit_table_plot_alignment_20260622",
                    "params": params,
                }
            ],
        }
    missing = [regime for regime in EXPECTED_REGIMES if regime not in regimes]
    if missing:
        failures.append(f"Missing expected regimes: {missing}")
    if failures:
        raise ValueError("warm_start_audit_failed: " + "; ".join(failures))
    search_space = optuna_hh._lane_union_param_space(
        "canonical",
        ("resolved_default",),
        speed_surface_profile=optuna_hh._HH_ROUTEA_FULL_POLICY_PROFILE,
    )
    return {
        "schema": "paper_i_hh_snake_full_policy_warm_start_manifest_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True,
        "promotion_policy": "no_table_or_manuscript_update",
        "baseline_alignment_json": str(alignment_path.relative_to(REPO_ROOT) if alignment_path.is_relative_to(REPO_ROOT) else alignment_path),
        "search_surface_profile": optuna_hh._HH_ROUTEA_FULL_POLICY_PROFILE,
        "fixed_identity_locks": {
            "paper": "Paper I",
            "target": "HH Table III",
            "route": "route_a",
            "method": "SNAKE",
            "adapt_pool": "full_meta_minus_hva",
            "batch_selection_mode": "reduced_plane",
            "batch_prefilter_mode": "off",
            "prune_policy": "recoverability_ladder_v1",
            "prune_mode": "both",
        },
        "regimes": {regime: regimes[regime] for regime in EXPECTED_REGIMES},
        "search_space": {key: list(values) for key, values in search_space.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-json", type=Path, default=DEFAULT_ALIGNMENT_JSON)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_warm_start_manifest(Path(args.alignment_json))
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is None:
        print(text)
        return 0
    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = REPO_ROOT / output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(text + "\n", encoding="utf-8")
    print(f"Wrote warm-start manifest: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
