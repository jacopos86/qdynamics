#!/usr/bin/env python3
"""Historical evidence ledger for Paper-I HH SNAKE Optuna bound selection.

This module is intentionally standalone. It reads completed/local JSON evidence,
normalizes a minimal source/settings/prefix ledger, computes curve labels and
regime-local Pareto contributors, and emits a first-pass Optuna surface proposal.

It does not launch runs, mutate existing run configs, or depend on Optuna.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_PIPELINE_NAME = "hh_optuna_evidence_ledger_v1"
_LOG_FLOOR = 1.0e-16
_DEFAULT_EPSILONS = (0.03, 0.05, 0.10)
_DEFAULT_MATERIAL_GEO_WIN_DECADES = 0.05
_DEFAULT_STALLED_SLOPE_DECADES_PER_PREFIX = 0.01
COMPARISON_MODES = ("snake_incumbent", "geo_anchor", "absolute")

REGIME_LABELS = (
    "weak-weak",
    "weak-strong",
    "intermediate-weak",
    "intermediate-strong",
    "strong-weak",
    "strong-strong",
)

ENGINEERING_PRIORS: dict[str, dict[str, Any]] = {
    "lambda_K_scale": {"range": [0.02, 1.0], "sampling": "log"},
    "phase1_prune_fraction": {"range": [0.10, 0.35], "sampling": "linear"},
    "prune_recoverability_slack_scale": {"range": [-0.5, 0.5], "sampling": "linear"},
    "batch_near_degenerate_ratio_shared": {"range": [0.90, 1.0], "sampling": "linear"},
    "batch_rank_rel_tol_shared": {"range": [1.0e-8, 1.0e-5], "sampling": "log"},
    "batch_additivity_slack_scale": {"range": [0.5, 2.0], "sampling": "log"},
}

SEVERE_SOFT_RISKS = {
    "stalled_curve",
    "bad_initial_slope",
    "bad_recent_slope",
    "late_only_energy_gain",
    "cost_explosion_without_energy_gain",
    "batch_size_explosion",
    "batch_nonadditive",
    "prune_rollback_excess",
    "prune_aggressive_without_recovery",
}


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    source_path: str
    source_hash: str
    parser_version: str
    artifact_kind: str
    cluster_id: str | None
    trial_id: str | None
    run_family: str
    benchmark_id: str | None
    regime: str
    method: str
    cutoff: dict[str, Any]
    hamiltonian_fields: dict[str, Any]
    status: str
    created_at: str | None = None
    source_batch: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class SettingsRecord:
    source_id: str
    route_contract: dict[str, Any]
    novelty: dict[str, Any]
    motif_prior: dict[str, Any]
    cost_fields: dict[str, Any]
    batch_fields: dict[str, Any]
    prune_fields: dict[str, Any]
    beam_fields: dict[str, Any]
    runtime_split_fields: dict[str, Any]
    optimizer_fields: dict[str, Any]
    randomness: dict[str, Any]
    effective_params: dict[str, Any]
    field_provenance: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PrefixPoint:
    source_id: str
    k: int
    energy_fields: dict[str, Any]
    selection: dict[str, Any]
    compiled_cost: dict[str, Any]
    proxy_cost: dict[str, Any]
    phase_telemetry: dict[str, Any]
    optimizer_telemetry: dict[str, Any]


@dataclass(frozen=True)
class GeoAnchor:
    anchor_id: str
    regime: str
    cutoff: dict[str, Any]
    method: str
    source_id: str
    delta_E_geo_best: float
    k_geo_useful: int
    k_geo_best: int
    geo_curve: list[dict[str, Any]]
    compiled_or_proxy_cost_geo: dict[str, Any]


@dataclass(frozen=True)
class TrialOutcome:
    source_id: str
    regime: str
    cutoff: dict[str, Any]
    method: str
    route_fidelity_class: str
    status: str
    delta_E_best: float | None
    k_best_energy: int | None
    k_nearbest_by_epsilon: dict[str, int | None]
    k_geo_parity: int | None
    k_geo_win: int | None
    delta_E_at_k_nearbest: float | None
    geo_anchor_id: str | None
    ratios: dict[str, Any]
    slope_features: dict[str, Any]
    pareto_features: dict[str, Any]
    labels: list[str]
    hard_exclude_from_bound_training: bool
    soft_risk_weight: float


@dataclass(frozen=True)
class AnalysisResult:
    sources: list[SourceRecord]
    settings: list[SettingsRecord]
    prefix_points: list[PrefixPoint]
    anchors: list[GeoAnchor]
    outcomes: list[TrialOutcome]
    pareto_contributors: dict[str, list[dict[str, Any]]]
    bound_proposals: list[dict[str, Any]]
    surface: dict[str, Any]
    summary: dict[str, Any]


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {k: _jsonable(v) for k, v in value.__dict__.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
    return value


def _load_jsonish(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows: list[Any] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
        return {"trial_events": rows}
    return json.loads(text)


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_id(path: Path, digest: str, trial_id: str | None = None) -> str:
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", path.stem).strip("_") or "source"
    base = f"{stem}_{digest[:12]}"
    if trial_id not in {None, ""}:
        return f"{base}_trial_{trial_id}"
    return base


def _nested(mapping: Mapping[str, Any] | None, *keys: str) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _finite_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        out = int(value)
    except Exception:
        return None
    return out


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"true", "1", "yes", "on", "enabled"}:
            return True
        if raw in {"false", "0", "no", "off", "disabled"}:
            return False
    return None


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _method_from_payload(payload: Mapping[str, Any], settings: Mapping[str, Any], path: Path | None = None) -> str:
    candidates = [
        payload.get("method"),
        _nested(payload, "adapt_vqe", "method"),
        _nested(payload, "result", "method"),
        settings.get("method"),
        settings.get("method_label"),
        settings.get("algorithm"),
        settings.get("static_method"),
        path.as_posix() if path is not None else None,
    ]
    raw = " ".join(_clean_text(x).lower() for x in candidates if x not in {None, ""})
    profile = _clean_text(settings.get("static_meta_feature_profile")).lower()
    if "snake" in raw or "route a" in raw or "route_a" in raw or profile == "paper_i_production_v1":
        return "snake"
    if "geo-adapt" in raw or "geo_adapt" in raw or "static_geo_adapt" in raw:
        return "geo_adapt"
    if "tetris" in raw:
        return "tetris_adapt"
    if "append" in raw:
        return "append_adapt"
    return "unknown"


def _normalize_regime(raw: Any) -> str:
    text = _clean_text(raw).lower()
    if not text:
        return "unknown"
    text = text.replace("-", "_").replace("/", "_")
    abbreviation_patterns = (
        (r"(?:^|[_\W])(?:hh_)?ss(?:[_\W]|$)", "strong-strong"),
        (r"(?:^|[_\W])(?:hh_)?ws(?:[_\W]|$)", "weak-strong"),
        (r"(?:^|[_\W])(?:hh_)?sw(?:[_\W]|$)", "strong-weak"),
        (r"(?:^|[_\W])(?:hh_)?ww(?:[_\W]|$)", "weak-weak"),
    )
    for pattern, label in abbreviation_patterns:
        if re.search(pattern, text):
            return label
    ordered = (
        ("intermediate_strong", "intermediate-strong"),
        ("intermediate_weak", "intermediate-weak"),
        ("strong_strong", "strong-strong"),
        ("weak_strong", "weak-strong"),
        ("strong_weak", "strong-weak"),
        ("weak_weak", "weak-weak"),
    )
    for needle, label in ordered:
        if needle in text:
            return label
    return "unknown"


def _nph_from_text(text: str) -> int | None:
    match = re.search(r"(?:nph|n_ph|phonon[_-]?cutoff)[_-]?(\d+)", text.lower())
    if match:
        return _finite_int(match.group(1))
    return None


def _regime_from_numeric(settings: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    u = _finite_float(_first_present(settings.get("U_over_t"), settings.get("u_over_t"), settings.get("u"), payload.get("u")))
    t = _finite_float(_first_present(settings.get("t"), payload.get("t"))) or 1.0
    g = _finite_float(_first_present(settings.get("lambda"), settings.get("g_ep"), settings.get("g"), payload.get("g_ep")))
    if u is not None and t not in {0.0, None}:
        u_over_t = float(u) / float(t)
    else:
        u_over_t = None
    if u_over_t is None or g is None:
        return "unknown"
    if u_over_t >= 1.0:
        hub = "strong"
    elif u_over_t >= 0.75:
        hub = "intermediate"
    else:
        hub = "weak"
    hol = "strong" if float(g) >= 1.0 else "weak"
    return f"{hub}-{hol}"


def _extract_settings_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for candidate in (
        payload.get("settings"),
        _nested(payload, "result", "settings"),
        _nested(payload, "adapt_vqe", "settings"),
        payload.get("run_settings"),
    ):
        if isinstance(candidate, Mapping):
            return candidate
    return {}


def _extract_reference_energy(payload: Mapping[str, Any]) -> float | None:
    for value in (
        _nested(payload, "ground_state", "exact_energy_filtered"),
        _nested(payload, "ground_state", "exact_energy"),
        _nested(payload, "adapt_vqe", "exact_gs_energy"),
        _nested(payload, "result", "same_cutoff_exact_gs_energy"),
        _nested(payload, "result", "exact_gs_energy"),
        _nested(payload, "result", "exact_energy"),
        payload.get("same_cutoff_reference_energy"),
        payload.get("reference_energy_same_cutoff"),
    ):
        out = _finite_float(value)
        if out is not None:
            return out
    return None


def _infer_artifact_kind(payload: Mapping[str, Any], path: Path) -> str:
    name = path.name.lower()
    if path.suffix.lower() == ".jsonl" or "trial_events" in payload:
        return "trial_events_jsonl"
    if "current_best" in name:
        return "current_best_json"
    if "current" in name:
        return "current_json"
    if "preflight" in name:
        return "preflight_json"
    if "audit" in name:
        return "audit_json"
    if "adapt_vqe" in payload or "ground_state" in payload:
        return "result_json"
    return "unknown"


def _infer_status(payload: Mapping[str, Any]) -> str:
    raw_status = _clean_text(_first_present(payload.get("status"), _nested(payload, "result", "status"))).lower()
    if raw_status in {"completed", "failed", "held", "partial", "running"}:
        return "partial" if raw_status == "running" else raw_status
    if _nested(payload, "adapt_vqe", "energy") is not None or _nested(payload, "result", "energy") is not None:
        return "completed"
    if _history_from_payload(payload):
        return "completed"
    success = _first_present(payload.get("success"), _nested(payload, "adapt_vqe", "success"), _nested(payload, "result", "success"))
    if success is False:
        return "failed"
    return "unknown"


def _history_from_payload(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates = (
        _nested(payload, "adapt_vqe", "history"),
        _nested(payload, "result", "adapt_history"),
        payload.get("adapt_history"),
        payload.get("history"),
        _nested(payload, "trial", "history"),
    )
    for candidate in candidates:
        if isinstance(candidate, list):
            return [row for row in candidate if isinstance(row, Mapping)]
    return []


def _list_from_any(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value in {None, ""}:
        return []
    return [value]


def _compiled_cost_from_row(row: Mapping[str, Any], payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    nested = row.get("compiled_cost") if isinstance(row.get("compiled_cost"), Mapping) else {}
    payload = payload or {}
    n2q = _finite_int(_first_present(nested.get("N2q"), nested.get("compiled_count_2q"), row.get("N2q"), row.get("compiled_count_2q")))
    d2q = _finite_int(_first_present(nested.get("D2q"), nested.get("compiled_depth"), row.get("D2q"), row.get("compiled_depth")))
    dcirc = _finite_int(_first_present(nested.get("Dcirc"), nested.get("compiled_size"), row.get("Dcirc"), row.get("compiled_size")))
    nparam = _finite_int(_first_present(nested.get("Nparam"), nested.get("runtime_parameter_count"), row.get("Nparam"), row.get("runtime_parameter_count")))
    status_raw = _clean_text(_first_present(nested.get("status"), row.get("compiled_cost_status"))).lower()
    if n2q is not None or d2q is not None or dcirc is not None:
        status = "compiled"
    elif status_raw:
        status = status_raw
    else:
        status = "unavailable"
    return {
        "status": status,
        "N2q": n2q,
        "D2q": d2q,
        "Dcirc": dcirc,
        "rotation_count": _finite_int(_first_present(nested.get("rotation_count"), row.get("rotation_count"))),
        "Nparam": nparam,
        "compiler_name": nested.get("compiler_name"),
        "compiler_version": nested.get("compiler_version"),
        "failure_reason": _first_present(nested.get("failure_reason"), row.get("compiled_cost_failure_reason")),
    }


def _proxy_cost_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    nested = row.get("proxy_cost") if isinstance(row.get("proxy_cost"), Mapping) else {}
    vals = {
        "twoq_proxy": _finite_float(_first_present(nested.get("twoq_proxy"), row.get("twoq_proxy"), row.get("compile_proxy_2q"))),
        "depth_proxy": _finite_float(_first_present(nested.get("depth_proxy"), row.get("depth_proxy"), row.get("compile_proxy_depth"))),
        "shot_proxy": _finite_float(_first_present(nested.get("shot_proxy"), row.get("shot_proxy"))),
        "work_proxy": _finite_float(_first_present(nested.get("work_proxy"), row.get("work_proxy"), row.get("S_proxy"))),
        "proxy_formula_hash": _first_present(nested.get("proxy_formula_hash"), row.get("proxy_formula_hash")),
    }
    return {"available": any(vals[k] is not None for k in ("twoq_proxy", "depth_proxy", "shot_proxy", "work_proxy")), **vals}


def _extract_prefix_points(payload: Mapping[str, Any], source_id: str) -> list[PrefixPoint]:
    ref = _extract_reference_energy(payload)
    history = _history_from_payload(payload)
    points: list[PrefixPoint] = []
    if not history:
        adapt = payload.get("adapt_vqe", payload)
        if isinstance(adapt, Mapping):
            history = [adapt]
    for idx, row in enumerate(history, start=1):
        k = _finite_int(_first_present(row.get("k"), row.get("depth"), row.get("ansatz_depth"), row.get("iteration"), row.get("step"))) or idx
        row_ref = _finite_float(row.get("benchmark_target_reference_energy"))
        if row_ref is None:
            row_ref = ref
        energy = _finite_float(_first_present(row.get("energy_after"), row.get("energy_after_opt"), row.get("energy_current"), row.get("energy"), row.get("optimizer_reported_energy"), row.get("spsa_energy_after"), row.get("energy_before_opt")))
        delta_e = _finite_float(_first_present(row.get("abs_delta_e_same_cutoff"), row.get("abs_delta_e"), row.get("delta_abs_current"), row.get("benchmark_target_abs_delta_current"), row.get("delta_E"), row.get("delta_e"), row.get("error")))
        delta_source = "reported" if delta_e is not None else "unavailable"
        if delta_e is None and energy is not None and row_ref is not None:
            delta_e = abs(float(energy) - float(row_ref))
            delta_source = "recomputed"
        labels_prefix = _list_from_any(_first_present(row.get("selected_operator_labels"), row.get("operator_labels"), row.get("selected_labels"), row.get("operator_label_sequence")))
        selected_records = row.get("selected_records") if isinstance(row.get("selected_records"), list) else []
        if not labels_prefix and selected_records:
            labels_prefix = [
                _first_present(rec.get("operator_label"), rec.get("generator_label"), rec.get("label"))
                for rec in selected_records
                if isinstance(rec, Mapping)
            ]
        new_labels = _list_from_any(_first_present(row.get("newly_selected_generator_labels"), row.get("selected_op"), row.get("selected_label"), row.get("operator_label"), row.get("accepted_label")))
        post_prune = row.get("post_admission_prune") if isinstance(row.get("post_admission_prune"), Mapping) else {}
        phase_telemetry = {
            "phase1_pool_size": _finite_int(row.get("phase1_pool_size")),
            "phase1_survivor_count": _finite_int(row.get("phase1_survivor_count")),
            "phase2_survivor_count": _finite_int(row.get("phase2_survivor_count")),
            "phase3_survivor_count": _finite_int(row.get("phase3_survivor_count")),
            "phase2_best_score": _finite_float(row.get("phase2_best_score")),
            "phase3_best_score": _finite_float(row.get("phase3_best_score")),
            "phase_scores_summary": row.get("phase_scores_summary") if isinstance(row.get("phase_scores_summary"), Mapping) else {},
            "batch_size": _finite_int(row.get("batch_size")),
            "batch_joint_gain": _finite_float(row.get("batch_joint_gain")),
            "batch_additive_gain_sum": _finite_float(row.get("batch_additive_gain_sum")),
            "batch_additivity_ratio": _finite_float(row.get("batch_additivity_ratio")),
            "prune_nomination_count": _finite_int(row.get("prune_nomination_count")),
            "prune_attempt_count": _finite_int(_first_present(row.get("prune_attempt_count"), post_prune.get("attempt_count"), post_prune.get("candidate_count"))),
            "prune_accept_count": _finite_int(_first_present(row.get("prune_accept_count"), post_prune.get("accept_count"), post_prune.get("accepted_count"))),
            "prune_rollback_count": _finite_int(_first_present(row.get("prune_rollback_count"), post_prune.get("rollback_count"))),
            "prune_max_regression": _finite_float(_first_present(row.get("prune_max_regression"), post_prune.get("max_regression"))),
            "prune_enabled": _bool_or_none(_first_present(row.get("prune_enabled"), post_prune.get("enabled"))),
            "beam_branch_id": _finite_int(row.get("beam_branch_id")),
            "beam_parent_id": _finite_int(row.get("beam_parent_id")),
            "beam_live_branch_count": _finite_int(row.get("beam_live_branch_count")),
        }
        points.append(
            PrefixPoint(
                source_id=source_id,
                k=int(k),
                energy_fields={
                    "energy": energy,
                    "same_cutoff_reference_energy": row_ref,
                    "delta_E": delta_e,
                    "delta_E_source": delta_source,
                    "delta_E_floor_applied": False,
                },
                selection={
                    "selected_generator_labels_prefix": [str(x) for x in labels_prefix],
                    "newly_selected_generator_labels": [str(x) for x in new_labels],
                    "generator_family_prefix": [],
                    "new_generator_family": [],
                    "sequence_hash_prefix": None,
                },
                compiled_cost=_compiled_cost_from_row(row),
                proxy_cost=_proxy_cost_from_row(row),
                phase_telemetry=phase_telemetry,
                optimizer_telemetry={
                    "optimizer_status": _clean_text(row.get("optimizer_status") or "unknown"),
                    "n_objective_evals": _finite_int(row.get("n_objective_evals")),
                    "final_gradient_proxy": _finite_float(row.get("final_gradient_proxy")),
                    "spsa_schedule_hash_or_summary": row.get("spsa_schedule_hash_or_summary"),
                },
            )
        )
    points.sort(key=lambda p: int(p.k))
    dedup: dict[int, PrefixPoint] = {}
    for point in points:
        dedup[int(point.k)] = point
    return [dedup[k] for k in sorted(dedup)]


def _setting(settings: Mapping[str, Any], payload: Mapping[str, Any], *names: str) -> tuple[Any, str]:
    for name in names:
        if name in settings:
            return settings[name], f"settings.{name}"
    for name in names:
        if name in payload:
            return payload[name], f"payload.{name}"
    return None, "unavailable"


def _extract_source_and_settings(path: Path, payload: Mapping[str, Any]) -> tuple[SourceRecord, SettingsRecord, list[PrefixPoint]]:
    digest = _sha256_path(path)
    settings = _extract_settings_mapping(payload)
    trial_id = _clean_text(_first_present(payload.get("trial_id"), payload.get("trial_number"))) or None
    sid = _source_id(path, digest, trial_id=trial_id)
    method = _method_from_payload(payload, settings, path)
    regime = _normalize_regime(_first_present(payload.get("regime"), settings.get("regime"), payload.get("benchmark_id"), settings.get("benchmark_id"), payload.get("case_id"), settings.get("case_id"), path.name))
    if regime == "unknown":
        regime = _normalize_regime(path.as_posix())
    if regime == "unknown":
        regime = _regime_from_numeric(settings, payload)
    n_ph = _finite_int(_first_present(settings.get("n_ph_work"), settings.get("n_ph_max"), settings.get("n_ph"), payload.get("n_ph_max")))
    if n_ph is None:
        n_ph = _nph_from_text(path.as_posix())
    if n_ph is None and regime != "unknown" and ("hh" in path.as_posix().lower() or "holstein" in path.as_posix().lower()):
        n_ph = 4 if regime.endswith("strong") else 2
    benchmark_id = _clean_text(_first_present(payload.get("benchmark_id"), settings.get("benchmark_id"), payload.get("case_id"))) or None
    status = _infer_status(payload)
    source = SourceRecord(
        source_id=sid,
        source_path=str(path),
        source_hash=digest,
        parser_version=_PIPELINE_NAME,
        artifact_kind=_infer_artifact_kind(payload, path),
        cluster_id=_clean_text(_first_present(payload.get("cluster_id"), settings.get("cluster_id"))) or None,
        trial_id=trial_id,
        run_family=_clean_text(_first_present(payload.get("run_family"), settings.get("run_family"))) or "unknown",
        benchmark_id=benchmark_id,
        regime=regime,
        method=method,
        cutoff={"n_ph": n_ph, "other_cutoff_fields": {}},
        hamiltonian_fields={
            "U_over_t": _finite_float(_first_present(settings.get("U_over_t"), settings.get("u_over_t"))),
            "g_or_lambda": _finite_float(_first_present(settings.get("lambda"), settings.get("g_ep"), settings.get("g"))),
            "omega": _finite_float(_first_present(settings.get("omega0"), settings.get("omega"))),
            "site_count": _finite_int(_first_present(settings.get("L"), payload.get("L"))),
            "electron_sector": settings.get("num_particles"),
        },
        status=status,
        created_at=_clean_text(_first_present(payload.get("created_at"), payload.get("timestamp"))) or None,
        source_batch=str(path.parent),
    )

    field_prov: dict[str, str] = {}

    def pick(*names: str) -> Any:
        value, origin = _setting(settings, payload, *names)
        field_prov[names[0]] = origin
        return value

    route_profile = pick("static_meta_feature_profile", "meta_feature_profile", "route_profile")
    pool_key = pick("adapt_pool", "pool_key", "route_base_pool_key")
    gamma = _finite_float(pick("phase2_gamma_N"))
    gamma_schedule = _clean_text(pick("phase2_gamma_N_schedule_mode")) or "unknown"
    motif_bonus = _finite_float(pick("phase2_motif_bonus_weight", "novelty_bonus", "motif_bonus_weight"))
    fixed_optimizer = _clean_text(pick("adapt_inner_optimizer", "fixed_inner_optimizer")) or "unknown"
    batch_enabled = _bool_or_none(pick("phase3_enable_batching", "phase2_enable_batching"))
    if batch_enabled is None:
        batch_enabled = _bool_or_none(settings.get("batching_enabled"))
    batch_near = _finite_float(pick("phase2_batch_near_degenerate_ratio", "phase3_batch_near_degenerate_ratio"))
    batch_rank = _finite_float(pick("phase2_batch_rank_rel_tol", "phase3_batch_rank_rel_tol"))
    batch_add = _finite_float(pick("phase2_batch_additivity_tol", "phase3_batch_additivity_tol"))
    prune_enabled = _bool_or_none(pick("phase1_prune_enabled"))
    prune_fraction = _finite_float(pick("phase1_prune_fraction"))
    prune_max_regression = _finite_float(pick("phase1_prune_max_regression"))
    prune_retained_gain_ratio = _finite_float(pick("phase1_prune_retained_gain_ratio"))
    lambda_k_scale = _finite_float(pick("lambda_K_scale", "cost_lambda_K_scale"))
    add_slack = _finite_float(pick("batch_additivity_slack_scale"))
    if add_slack is None and batch_add is not None:
        add_slack = float(batch_add) / 0.25
    prune_slack = _finite_float(pick("prune_recoverability_slack_scale"))
    route_fidelity = _route_fidelity_class(
        method=method,
        route_profile=route_profile,
        pool_key=pool_key,
        gamma=gamma,
        gamma_schedule=gamma_schedule,
        motif_bonus=motif_bonus,
        fixed_optimizer=fixed_optimizer,
    )
    settings_record = SettingsRecord(
        source_id=sid,
        route_contract={
            "route_profile": route_profile,
            "pool_key": pool_key,
            "route_contract_hash": _route_contract_hash(settings),
            "paper_i_fidelity_class": route_fidelity,
        },
        novelty={
            "phase2_gamma_N": gamma,
            "phase2_gamma_N_schedule_mode": gamma_schedule,
            "phase3_gamma_N": _finite_float(pick("phase3_gamma_N")),
            "novelty_schedule_summary": settings.get("phase2_gamma_N_schedule"),
        },
        motif_prior={
            "phase2_motif_bonus_weight": motif_bonus,
            "phase3_motif_bonus_weight": _finite_float(pick("phase3_motif_bonus_weight")),
            "path_prior_enabled": bool(_bool_or_none(pick("path_prior_enabled")) or False),
            "path_prior_fields": {},
        },
        cost_fields={
            "cost_bundle_hash": _cost_bundle_hash(settings),
            "lambda_K_scale_if_recoverable": lambda_k_scale,
            "phase1_lambda_compile": _finite_float(pick("phase1_lambda_compile")),
            "phase1_lambda_measure": _finite_float(pick("phase1_lambda_measure")),
            "phase1_lambda_2q": _finite_float(pick("phase1_lambda_2q")),
            "phase1_lambda_d": _finite_float(pick("phase1_lambda_d")),
            "phase1_lambda_theta": _finite_float(pick("phase1_lambda_theta")),
            "phase1_lambda_shot": _finite_float(pick("phase1_lambda_shot")),
            "phase2_lambda_2q": _finite_float(pick("phase2_lambda_2q")),
            "phase2_lambda_d": _finite_float(pick("phase2_lambda_d")),
            "phase2_lambda_theta": _finite_float(pick("phase2_lambda_theta")),
            "phase2_lambda_shot": _finite_float(pick("phase2_lambda_shot")),
            "phase2_w_depth": _finite_float(pick("phase2_w_depth")),
            "phase2_w_group": _finite_float(pick("phase2_w_group")),
            "phase2_w_shot": _finite_float(pick("phase2_w_shot")),
            "phase2_w_optdim": _finite_float(pick("phase2_w_optdim")),
            "phase2_w_reuse": _finite_float(pick("phase2_w_reuse")),
            "phase2_w_lifetime": _finite_float(pick("phase2_w_lifetime")),
        },
        batch_fields={
            "phase2_enable_batching": batch_enabled,
            "phase3_enable_batching": batch_enabled,
            "phase2_batch_size_cap": _finite_int(pick("phase2_batch_size_cap", "phase3_batch_size_cap")),
            "phase3_batch_size_cap": _finite_int(pick("phase3_batch_size_cap", "phase2_batch_size_cap")),
            "phase2_batch_near_degenerate_ratio": batch_near,
            "phase3_batch_near_degenerate_ratio": batch_near,
            "phase2_batch_rank_rel_tol": batch_rank,
            "phase3_batch_rank_rel_tol": batch_rank,
            "phase2_batch_additivity_tol": batch_add,
            "phase3_batch_additivity_tol": batch_add,
            "phase3_batch_prefilter_mode": _clean_text(pick("phase3_batch_prefilter_mode")) or "unknown",
        },
        prune_fields={
            "phase1_prune_enabled": prune_enabled,
            "phase1_prune_policy": _clean_text(pick("phase1_prune_policy")) or "unknown",
            "phase1_prune_fraction": prune_fraction,
            "phase1_prune_candidate_count": _finite_int(pick("phase1_prune_candidate_count")),
            "phase1_prune_max_regression": prune_max_regression,
            "phase1_prune_retained_gain_ratio": prune_retained_gain_ratio,
            "phase1_prune_protect_steps": _finite_int(pick("phase1_prune_protect_steps")),
            "phase1_prune_stale_age": _finite_int(pick("phase1_prune_stale_age")),
            "phase1_prune_checkpoint_period": _finite_int(pick("phase1_prune_checkpoint_period")),
            "phase1_prune_amplitude_witness_required": _bool_or_none(pick("phase1_prune_amplitude_witness_required")),
        },
        beam_fields={
            "adapt_beam_enabled": (_finite_int(pick("adapt_beam_live_branches")) or 0) > 1,
            "adapt_beam_live_branches": _finite_int(pick("adapt_beam_live_branches")),
            "adapt_beam_children_per_parent": _finite_int(pick("adapt_beam_children_per_parent")),
            "adapt_beam_terminated_keep": _finite_int(pick("adapt_beam_terminated_keep")),
            "beam_selection_policy": settings.get("beam_selection_policy"),
        },
        runtime_split_fields={
            "phase3_runtime_split_mode": _clean_text(pick("phase3_runtime_split_mode", "runtime_split_mode")) or "unknown",
            "phase3_runtime_split_selection_mode": _clean_text(pick("phase3_runtime_split_selection_mode")) or None,
            "child_set_handling": "serial_if_enabled",
        },
        optimizer_fields={
            "fixed_inner_optimizer": fixed_optimizer,
            "spsa_schedule_hash_or_summary": settings.get("spsa_schedule_hash_or_summary"),
            "optimizer_budget": _finite_int(pick("adapt_maxiter", "maxiter")),
            "optimizer_seed": _finite_int(pick("seed", "optimizer_seed")),
        },
        randomness={
            "trial_seed": _finite_int(pick("trial_seed")),
            "sampler_seed": _finite_int(pick("sampler_seed")),
            "backend_seed": _finite_int(pick("phase3_backend_transpile_seed", "backend_seed")),
        },
        effective_params={
            "lambda_K_scale": lambda_k_scale,
            "phase1_prune_fraction": prune_fraction,
            "prune_recoverability_slack_scale": prune_slack,
            "batch_near_degenerate_ratio_shared": batch_near,
            "batch_rank_rel_tol_shared": batch_rank,
            "batch_additivity_slack_scale": add_slack,
        },
        field_provenance=field_prov,
    )
    return source, settings_record, _extract_prefix_points(payload, sid)


def _route_contract_hash(settings: Mapping[str, Any]) -> str:
    keys = (
        "adapt_pool",
        "pool_key",
        "static_meta_feature_profile",
        "phase2_gamma_N",
        "phase2_gamma_N_schedule_mode",
        "phase2_motif_bonus_weight",
        "phase3_enable_batching",
        "phase3_batch_selection_mode",
        "phase3_batch_prefilter_mode",
        "phase1_prune_policy",
        "adapt_inner_optimizer",
        "phase3_runtime_split_mode",
    )
    payload = {k: settings.get(k) for k in keys if k in settings}
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _cost_bundle_hash(settings: Mapping[str, Any]) -> str | None:
    keys = [k for k in sorted(settings) if "lambda" in str(k).lower() or str(k).startswith("phase2_w_")]
    if not keys:
        return None
    raw = json.dumps({k: settings.get(k) for k in keys}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _route_fidelity_class(*, method: str, route_profile: Any, pool_key: Any, gamma: float | None, gamma_schedule: str, motif_bonus: float | None, fixed_optimizer: str) -> str:
    if method != "snake":
        return "noncomparable"
    pool_text = _clean_text(pool_key).lower()
    route_text = _clean_text(route_profile).lower()
    opt_text = _clean_text(fixed_optimizer).lower()
    exact = (
        (pool_text in {"full_meta", "full_meta_minus_hva"} or "full_meta" in pool_text)
        and route_text == "paper_i_production_v1"
        and gamma is not None
        and abs(float(gamma) - 1.0) <= 1.0e-12
        and gamma_schedule == "fixed"
        and (motif_bonus is not None and abs(float(motif_bonus)) <= 1.0e-15)
        and (opt_text in {"spsa", "unknown", ""})
    )
    if exact:
        return "exact"
    if method == "snake" and ("full_meta" in pool_text or route_text):
        return "compatible"
    return "context_only"


def _build_method_anchors(
    sources: Sequence[SourceRecord],
    prefix_points: Sequence[PrefixPoint],
    *,
    method: str,
    anchor_kind: str,
    source_path_allowlist: set[str] | None = None,
) -> list[GeoAnchor]:
    by_source = _points_by_source(prefix_points)
    anchors: dict[tuple[str, Any], GeoAnchor] = {}
    for source in sources:
        if source.method != method or source.regime == "unknown":
            continue
        if source_path_allowlist is not None and str(Path(source.source_path).resolve()) not in source_path_allowlist:
            continue
        pts = [p for p in by_source.get(source.source_id, []) if _finite_float(p.energy_fields.get("delta_E")) is not None]
        if not pts:
            continue
        best = min(pts, key=lambda p: float(p.energy_fields["delta_E"]))
        key = (source.regime, source.cutoff.get("n_ph"))
        candidate = GeoAnchor(
            anchor_id=f"{anchor_kind}|{source.regime}|nph{source.cutoff.get('n_ph')}|{source.source_hash[:12]}",
            regime=source.regime,
            cutoff=source.cutoff,
            method=anchor_kind,
            source_id=source.source_id,
            delta_E_geo_best=float(best.energy_fields["delta_E"]),
            k_geo_useful=int(best.k),
            k_geo_best=int(best.k),
            geo_curve=[{"k": int(p.k), "delta_E": p.energy_fields.get("delta_E")} for p in pts],
            compiled_or_proxy_cost_geo=best.compiled_cost if best.compiled_cost.get("status") == "compiled" else best.proxy_cost,
        )
        old = anchors.get(key)
        if old is None or candidate.delta_E_geo_best < old.delta_E_geo_best:
            anchors[key] = candidate
    return list(anchors.values())


def _build_comparison_anchors(
    sources: Sequence[SourceRecord],
    prefix_points: Sequence[PrefixPoint],
    *,
    comparison_mode: str,
    incumbent_source_paths: Sequence[str | Path] | None = None,
) -> list[GeoAnchor]:
    if comparison_mode == "absolute":
        return []
    if comparison_mode == "geo_anchor":
        return _build_method_anchors(sources, prefix_points, method="geo_adapt", anchor_kind="geo_adapt")
    if comparison_mode == "snake_incumbent":
        allowlist = None
        if incumbent_source_paths:
            allowlist = {str(Path(p).expanduser().resolve()) for p in incumbent_source_paths}
        return _build_method_anchors(sources, prefix_points, method="snake", anchor_kind="snake_incumbent", source_path_allowlist=allowlist)
    raise ValueError(f"Unsupported comparison_mode={comparison_mode!r}; expected one of {COMPARISON_MODES}.")


def _build_geo_anchors(sources: Sequence[SourceRecord], prefix_points: Sequence[PrefixPoint]) -> list[GeoAnchor]:
    return _build_method_anchors(sources, prefix_points, method="geo_adapt", anchor_kind="geo_adapt")


def _points_by_source(points: Sequence[PrefixPoint]) -> dict[str, list[PrefixPoint]]:
    out: dict[str, list[PrefixPoint]] = {}
    for point in points:
        out.setdefault(point.source_id, []).append(point)
    for rows in out.values():
        rows.sort(key=lambda p: int(p.k))
    return out


def _settings_by_source(settings: Sequence[SettingsRecord]) -> dict[str, SettingsRecord]:
    return {row.source_id: row for row in settings}


def _anchor_by_regime_cutoff(anchors: Sequence[GeoAnchor]) -> dict[tuple[str, Any], GeoAnchor]:
    return {(a.regime, a.cutoff.get("n_ph")): a for a in anchors}


def _anchor_from_mapping(row: Mapping[str, Any], idx: int = 0) -> GeoAnchor:
    regime = _normalize_regime(_first_present(row.get("regime"), row.get("regime_label"), row.get("benchmark_id")))
    n_ph = _finite_int(_first_present(row.get("n_ph"), row.get("n_ph_max"), row.get("cutoff_n_ph"), _nested(row, "cutoff", "n_ph")))
    delta = _finite_float(_first_present(row.get("delta_E"), row.get("delta_E_anchor"), row.get("delta_E_geo_best"), row.get("abs_delta_e")))
    if delta is None:
        raise ValueError(f"Fixed anchor row {idx} is missing finite delta_E/abs_delta_e.")
    k_useful = _finite_int(_first_present(row.get("k_useful"), row.get("k"), row.get("adapt_depth"), row.get("depth")))
    if k_useful is None:
        raise ValueError(f"Fixed anchor row {idx} is missing k_useful/k/adapt_depth/depth.")
    method = _clean_text(_first_present(row.get("method"), row.get("anchor_kind"))) or "paper_i_fixed_anchor"
    anchor_id = _clean_text(row.get("anchor_id")) or f"{method}|{regime}|nph{n_ph}|fixed{idx}"
    source_id = _clean_text(row.get("source_id")) or anchor_id
    cost = row.get("compiled_or_proxy_cost_geo") if isinstance(row.get("compiled_or_proxy_cost_geo"), Mapping) else {}
    if not cost:
        cost = {
            "status": _clean_text(row.get("cost_status")) or "paper_i_fixed",
            "N2q": _finite_int(_first_present(row.get("N2q"), row.get("compiled_count_2q"))),
            "D2q": _finite_int(_first_present(row.get("D2q"), row.get("compiled_depth"))),
            "Dcirc": _finite_int(_first_present(row.get("Dcirc"), row.get("compiled_size"))),
        }
    return GeoAnchor(
        anchor_id=anchor_id,
        regime=regime,
        cutoff={"n_ph": n_ph, "other_cutoff_fields": {}},
        method=method,
        source_id=source_id,
        delta_E_geo_best=float(delta),
        k_geo_useful=int(k_useful),
        k_geo_best=_finite_int(_first_present(row.get("k_best"), row.get("k_geo_best"))) or int(k_useful),
        geo_curve=[{"k": int(k_useful), "delta_E": float(delta), "anchor_source": "fixed_paper_i"}],
        compiled_or_proxy_cost_geo=dict(cost),
    )


def _fixed_anchors_from_rows(rows: Sequence[Mapping[str, Any]] | None) -> list[GeoAnchor]:
    if not rows:
        return []
    return [_anchor_from_mapping(row, idx) for idx, row in enumerate(rows)]


def _load_fixed_anchor_rows(path: Path, *, anchor_set: str | None = None) -> list[Mapping[str, Any]]:
    payload = _load_jsonish(path)
    if isinstance(payload, Mapping):
        rows = None
        if anchor_set:
            sets = payload.get("anchor_sets")
            if not isinstance(sets, Mapping) or anchor_set not in sets:
                raise ValueError(f"Fixed anchor file {path} does not contain anchor_sets[{anchor_set!r}].")
            rows = sets[anchor_set]
        if rows is None:
            rows = payload.get("anchors")
        if rows is None:
            rows = payload.get("paper_i_geo_anchors")
        if rows is None and "anchor_sets" in payload and isinstance(payload["anchor_sets"], Mapping):
            raise ValueError(f"Fixed anchor file {path} contains anchor_sets; pass --fixed-anchor-set.")
        if rows is None and {"regime", "delta_E"} <= set(payload):
            rows = [payload]
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError(f"Fixed anchor file must contain a list, anchors[], paper_i_geo_anchors[], or selected anchor_sets[]: {path}")
    out = [row for row in rows if isinstance(row, Mapping)]
    if len(out) != len(rows):
        raise ValueError(f"Fixed anchor file contains non-object rows: {path}")
    return out


def _log_delta(delta_e: float | None) -> float | None:
    if delta_e is None or not math.isfinite(float(delta_e)):
        return None
    return math.log10(max(float(delta_e), _LOG_FLOOR))


def _best_so_far(values: Sequence[float]) -> list[float]:
    out: list[float] = []
    best = float("inf")
    for value in values:
        best = min(best, float(value))
        out.append(best)
    return out


def _slope(xs: Sequence[int], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    dx = int(xs[-1]) - int(xs[0])
    if dx == 0:
        return None
    return float(ys[-1] - ys[0]) / float(dx)


def _curve_metrics(points: Sequence[PrefixPoint], anchor: GeoAnchor | None, epsilons: Sequence[float]) -> dict[str, Any]:
    valid = [(int(p.k), float(p.energy_fields["delta_E"])) for p in points if _finite_float(p.energy_fields.get("delta_E")) is not None]
    if not valid:
        return {
            "delta_E_best": None,
            "k_best_energy": None,
            "k_nearbest_by_epsilon": {str(e): None for e in epsilons},
            "k_geo_parity": None,
            "k_geo_win": None,
            "delta_E_at_k_nearbest": None,
            "slope_features": {},
            "ratios": {},
            "best_so_far": [],
        }
    ks = [k for k, _d in valid]
    logs = [_log_delta(d) for _k, d in valid]
    logs_clean = [float(y) for y in logs if y is not None]
    best_logs = _best_so_far(logs_clean)
    best_idx = min(range(len(valid)), key=lambda i: valid[i][1])
    best_delta = float(valid[best_idx][1])
    best_log = _log_delta(best_delta)
    nearbest: dict[str, int | None] = {}
    for eps in epsilons:
        k_val = None
        if best_log is not None:
            for idx, b in enumerate(best_logs):
                if b <= float(best_log) + float(eps):
                    k_val = int(ks[idx])
                    break
        nearbest[str(eps)] = k_val
    eps_default = 0.05 if 0.05 in [float(e) for e in epsilons] else float(epsilons[0])
    k_near = nearbest.get(str(eps_default))
    delta_at_near = None
    if k_near is not None:
        for k, d in valid:
            if int(k) == int(k_near):
                delta_at_near = float(d)
                break
    y_geo = _log_delta(anchor.delta_E_geo_best) if anchor is not None else None
    k_geo = int(anchor.k_geo_useful) if anchor is not None else None
    k_parity = None
    k_win = None
    if y_geo is not None:
        for idx, b in enumerate(best_logs):
            if k_parity is None and b <= y_geo + float(eps_default):
                k_parity = int(ks[idx])
            if k_win is None and b <= y_geo - _DEFAULT_MATERIAL_GEO_WIN_DECADES:
                k_win = int(ks[idx])

    initial_end = min(len(ks), max(2, min(4, len(ks))))
    recent_w = min(5, max(2, len(ks) // 4 if len(ks) >= 8 else min(len(ks), 3)))
    initial_slope = _slope(ks[:initial_end], best_logs[:initial_end])
    recent_slope = _slope(ks[-recent_w:], best_logs[-recent_w:]) if len(ks) >= 2 else None
    best_window_slope = None
    if len(ks) >= 2:
        w = min(5, max(2, len(ks) // 4 if len(ks) >= 8 else 2))
        slopes = [_slope(ks[i : i + w], best_logs[i : i + w]) for i in range(0, len(ks) - w + 1)]
        slopes = [s for s in slopes if s is not None]
        best_window_slope = min(slopes) if slopes else None
    plateau_length = _plateau_length(ks, best_logs)
    aulc = None
    if y_geo is not None and best_logs:
        aulc = sum((b - y_geo) for b in best_logs) / float(len(best_logs))
    ratios = {
        "energy_ratio_to_geo": (best_delta / float(anchor.delta_E_geo_best) if anchor is not None and anchor.delta_E_geo_best > 0 else None),
        "k_nearbest_ratio_to_geo": (float(k_near) / float(k_geo) if k_near is not None and k_geo not in {None, 0} else None),
    }
    return {
        "delta_E_best": best_delta,
        "k_best_energy": int(ks[best_idx]),
        "k_nearbest_by_epsilon": nearbest,
        "k_geo_parity": k_parity,
        "k_geo_win": k_win,
        "delta_E_at_k_nearbest": delta_at_near,
        "slope_features": {
            "initial_slope": initial_slope,
            "recent_slope": recent_slope,
            "best_window_slope": best_window_slope,
            "plateau_length": plateau_length,
            "plateau_escape_delta_log": None,
            "area_under_log_error_curve": aulc,
        },
        "ratios": ratios,
        "best_so_far": [{"k": int(k), "log_delta_E_best_so_far": float(b)} for k, b in zip(ks, best_logs)],
    }


def _plateau_length(ks: Sequence[int], best_logs: Sequence[float]) -> int:
    if len(ks) < 2:
        return 0
    longest = 1
    current = 1
    for idx in range(1, len(ks)):
        dy = abs(float(best_logs[idx]) - float(best_logs[idx - 1]))
        dx = max(1, int(ks[idx]) - int(ks[idx - 1]))
        if dy / dx <= _DEFAULT_STALLED_SLOPE_DECADES_PER_PREFIX:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return int(longest)


def _assign_labels(
    source: SourceRecord,
    settings: SettingsRecord,
    points: Sequence[PrefixPoint],
    anchor: GeoAnchor | None,
    metrics: Mapping[str, Any],
    *,
    comparison_mode: str,
) -> tuple[list[str], bool, float]:
    labels: list[str] = []
    hard: set[str] = set()
    soft_weight = 1.0
    if source.status in {"failed", "held"}:
        hard.add("failed_or_held_job")
    if source.regime == "unknown" or source.cutoff.get("n_ph") is None:
        hard.add("noncomparable_cutoff_or_regime")
    valid_curve = len([p for p in points if _finite_float(p.energy_fields.get("delta_E")) is not None]) >= 3
    if not valid_curve:
        hard.add("missing_curve_telemetry")
    if source.method == "snake" and comparison_mode != "absolute" and anchor is None:
        hard.add("missing_comparison_anchor")
    if metrics.get("delta_E_best") is None:
        hard.add("invalid_energy_error")
    if source.method == "unknown":
        hard.add("invalid_route_contract")
    labels.extend(sorted(hard))
    if not hard:
        labels.append("valid_completed")
    route_class = str(settings.route_contract.get("paper_i_fidelity_class"))
    if route_class == "exact":
        labels.append("exact_paper_i_fidelity")
    elif route_class == "compatible":
        labels.append("compatible_snake_context")
    gamma = settings.novelty.get("phase2_gamma_N")
    schedule = settings.novelty.get("phase2_gamma_N_schedule_mode")
    if gamma is not None and (abs(float(gamma) - 1.0) > 1.0e-12 or schedule != "fixed"):
        labels.append("nonflat_novelty_context")
        soft_weight += 0.25
    motif = settings.motif_prior.get("phase2_motif_bonus_weight")
    if motif is not None and abs(float(motif)) > 1.0e-15:
        labels.append("motif_prior_context")
        soft_weight += 0.25
    slopes = metrics.get("slope_features", {}) if isinstance(metrics.get("slope_features"), Mapping) else {}
    initial = _finite_float(slopes.get("initial_slope"))
    recent = _finite_float(slopes.get("recent_slope"))
    if initial is not None and initial >= -2.0 * _DEFAULT_STALLED_SLOPE_DECADES_PER_PREFIX:
        labels.append("bad_initial_slope")
        soft_weight += 0.5
    if recent is not None and recent >= -_DEFAULT_STALLED_SLOPE_DECADES_PER_PREFIX:
        labels.append("bad_recent_slope")
        labels.append("stalled_curve")
        soft_weight += 0.5
    ratios = metrics.get("ratios", {}) if isinstance(metrics.get("ratios"), Mapping) else {}
    k_ratio = _finite_float(ratios.get("k_nearbest_ratio_to_geo"))
    energy_ratio = _finite_float(ratios.get("energy_ratio_to_geo"))
    same_source_anchor = bool(anchor is not None and anchor.source_id == source.source_id)
    if same_source_anchor:
        labels.append("comparison_anchor_source")
    if metrics.get("k_geo_parity") is not None:
        labels.append("comparison_parity_reached")
        if comparison_mode == "geo_anchor":
            labels.append("geo_parity_reached")
        elif comparison_mode == "snake_incumbent":
            labels.append("snake_incumbent_parity_reached")
    if metrics.get("k_geo_win") is not None and not same_source_anchor:
        labels.append("material_comparison_win")
        if comparison_mode == "geo_anchor":
            labels.append("material_geo_win")
        elif comparison_mode == "snake_incumbent":
            labels.append("material_snake_incumbent_win")
    if k_ratio is not None and k_ratio <= 1.25:
        labels.append("early_useful_prefix")
    if metrics.get("k_geo_win") is not None and anchor is not None and int(metrics["k_geo_win"]) > 1.5 * max(1, int(anchor.k_geo_useful)):
        labels.append("late_only_energy_gain")
        soft_weight += 0.5
    if energy_ratio is not None and energy_ratio <= 1.0 and k_ratio is not None and k_ratio <= 1.0 and not same_source_anchor:
        labels.append("candidate_pareto_dominates_comparison_energy_iteration")
        if comparison_mode == "geo_anchor":
            labels.append("snake_pareto_dominates_geo_energy_iteration")
        elif comparison_mode == "snake_incumbent":
            labels.append("snake_pareto_dominates_incumbent_energy_iteration")
    return sorted(set(labels)), bool(hard), soft_weight


def _compute_outcomes(
    sources: Sequence[SourceRecord],
    settings_records: Sequence[SettingsRecord],
    prefix_points: Sequence[PrefixPoint],
    anchors: Sequence[GeoAnchor],
    epsilons: Sequence[float],
    *,
    comparison_mode: str,
) -> list[TrialOutcome]:
    points_by_source = _points_by_source(prefix_points)
    settings_by_source = _settings_by_source(settings_records)
    anchors_by_key = _anchor_by_regime_cutoff(anchors)
    outcomes: list[TrialOutcome] = []
    for source in sources:
        settings = settings_by_source[source.source_id]
        anchor = anchors_by_key.get((source.regime, source.cutoff.get("n_ph")))
        pts = points_by_source.get(source.source_id, [])
        metrics = _curve_metrics(pts, anchor, epsilons)
        labels, hard, weight = _assign_labels(source, settings, pts, anchor, metrics, comparison_mode=comparison_mode)
        outcomes.append(
            TrialOutcome(
                source_id=source.source_id,
                regime=source.regime,
                cutoff=source.cutoff,
                method=source.method,
                route_fidelity_class=str(settings.route_contract.get("paper_i_fidelity_class")),
                status=source.status,
                delta_E_best=metrics.get("delta_E_best"),
                k_best_energy=metrics.get("k_best_energy"),
                k_nearbest_by_epsilon=dict(metrics.get("k_nearbest_by_epsilon", {})),
                k_geo_parity=metrics.get("k_geo_parity"),
                k_geo_win=metrics.get("k_geo_win"),
                delta_E_at_k_nearbest=metrics.get("delta_E_at_k_nearbest"),
                geo_anchor_id=(anchor.anchor_id if anchor is not None else None),
                ratios={"comparison_mode": comparison_mode, **dict(metrics.get("ratios", {}))},
                slope_features=dict(metrics.get("slope_features", {})),
                pareto_features={"historical_pareto_contributor": False, "pareto_prefixes": []},
                labels=labels,
                hard_exclude_from_bound_training=hard,
                soft_risk_weight=float(weight),
            )
        )
    return outcomes

def _pareto_front(points: Sequence[dict[str, Any]], eps: Sequence[float]) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    for idx, point in enumerate(points):
        coords = [float(x) for x in point["coords"]]
        dominated = False
        for jdx, other in enumerate(points):
            if idx == jdx:
                continue
            other_coords = [float(x) for x in other["coords"]]
            all_le = all(o <= c + float(eps_i) for o, c, eps_i in zip(other_coords, coords, eps))
            one_strict = any(o < c - float(eps_i) for o, c, eps_i in zip(other_coords, coords, eps))
            if all_le and one_strict:
                dominated = True
                break
        if not dominated:
            front.append(dict(point))
    return front


def _attach_pareto_contributors(outcomes: Sequence[TrialOutcome], prefix_points: Sequence[PrefixPoint]) -> tuple[list[TrialOutcome], dict[str, list[dict[str, Any]]]]:
    points_by_source = _points_by_source(prefix_points)
    out_by_source = {o.source_id: o for o in outcomes}
    grouped: dict[tuple[str, Any], list[dict[str, Any]]] = {}
    for outcome in outcomes:
        if outcome.hard_exclude_from_bound_training or outcome.method not in {"snake", "geo_adapt"}:
            continue
        best = float("inf")
        for point in points_by_source.get(outcome.source_id, []):
            d = _finite_float(point.energy_fields.get("delta_E"))
            if d is None:
                continue
            y = _log_delta(d)
            if y is None:
                continue
            best = min(best, y)
            grouped.setdefault((outcome.regime, outcome.cutoff.get("n_ph")), []).append(
                {
                    "source_id": outcome.source_id,
                    "method": outcome.method,
                    "k": int(point.k),
                    "coords": [float(best), int(point.k)],
                    "delta_E": float(d),
                }
            )
    fronts: dict[str, list[dict[str, Any]]] = {}
    contributor_prefixes: dict[str, list[dict[str, Any]]] = {}
    for key, rows in grouped.items():
        front = _pareto_front(rows, eps=(0.02, 0.0))
        front_key = f"{key[0]}|nph{key[1]}"
        fronts[front_key] = front
        for row in front:
            if row.get("method") == "snake":
                contributor_prefixes.setdefault(str(row["source_id"]), []).append({"regime_key": front_key, "k": row["k"], "coords": row["coords"]})
    new_outcomes: list[TrialOutcome] = []
    for outcome in outcomes:
        prefixes = contributor_prefixes.get(outcome.source_id, [])
        labels = list(outcome.labels)
        if prefixes:
            labels.append("historical_pareto_contributor")
        new_outcomes.append(
            TrialOutcome(
                source_id=outcome.source_id,
                regime=outcome.regime,
                cutoff=outcome.cutoff,
                method=outcome.method,
                route_fidelity_class=outcome.route_fidelity_class,
                status=outcome.status,
                delta_E_best=outcome.delta_E_best,
                k_best_energy=outcome.k_best_energy,
                k_nearbest_by_epsilon=outcome.k_nearbest_by_epsilon,
                k_geo_parity=outcome.k_geo_parity,
                k_geo_win=outcome.k_geo_win,
                delta_E_at_k_nearbest=outcome.delta_E_at_k_nearbest,
                geo_anchor_id=outcome.geo_anchor_id,
                ratios=outcome.ratios,
                slope_features=outcome.slope_features,
                pareto_features={"historical_pareto_contributor": bool(prefixes), "pareto_prefixes": prefixes},
                labels=sorted(set(labels)),
                hard_exclude_from_bound_training=outcome.hard_exclude_from_bound_training,
                soft_risk_weight=outcome.soft_risk_weight,
            )
        )
    return new_outcomes, fronts


def _transform_param(name: str, value: float) -> float:
    if ENGINEERING_PRIORS[name]["sampling"] == "log":
        return math.log10(max(float(value), _LOG_FLOOR))
    return float(value)


def _inverse_transform_param(name: str, value: float) -> float:
    if ENGINEERING_PRIORS[name]["sampling"] == "log":
        return 10.0 ** float(value)
    return float(value)


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute quantile of empty values.")
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return vals[0]
    pos = float(q) * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _param_values(settings_records: Sequence[SettingsRecord]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in settings_records:
        vals: dict[str, float] = {}
        for name in ENGINEERING_PRIORS:
            v = _finite_float(row.effective_params.get(name))
            if v is not None:
                vals[name] = float(v)
        out[row.source_id] = vals
    return out


def _build_bound_proposals(outcomes: Sequence[TrialOutcome], settings_records: Sequence[SettingsRecord], *, min_safe_support: int = 6) -> list[dict[str, Any]]:
    param_by_source = _param_values(settings_records)
    proposals: list[dict[str, Any]] = []
    for name, prior in ENGINEERING_PRIORS.items():
        safe: list[TrialOutcome] = []
        adverse: list[TrialOutcome] = []
        for outcome in outcomes:
            if outcome.method != "snake" or outcome.hard_exclude_from_bound_training:
                continue
            if name not in param_by_source.get(outcome.source_id, {}):
                continue
            labels = set(outcome.labels)
            severe = bool(labels & SEVERE_SOFT_RISKS)
            positive = bool(labels & {"historical_pareto_contributor", "comparison_anchor_source", "comparison_parity_reached", "material_comparison_win", "geo_parity_reached", "material_geo_win"})
            if positive and not severe:
                safe.append(outcome)
            if severe:
                adverse.append(outcome)
        prior_range = [float(prior["range"][0]), float(prior["range"][1])]
        if len(safe) < int(min_safe_support):
            proposals.append(
                {
                    "parameter": name,
                    "status": "inconclusive",
                    "recommended_range": prior_range,
                    "sampling": prior["sampling"],
                    "evidence_rule": "engineering_prior_due_to_insufficient_safe_support",
                    "support_counts": {"safe": len(safe), "adverse": len(adverse), "min_safe_support": int(min_safe_support)},
                    "supporting_source_ids": [o.source_id for o in safe],
                    "excluded_regions": [],
                    "confidence": "low",
                    "validation_needed": "collect clean comparable SNAKE trials with this parameter observed",
                }
            )
            continue
        xs = [_transform_param(name, param_by_source[o.source_id][name]) for o in safe]
        lo = _quantile(xs, 0.10)
        hi = _quantile(xs, 0.90)
        width = max(hi - lo, 0.0)
        pad = max(width * 0.10, 0.0)
        lo -= pad
        hi += pad
        eng_lo_t = _transform_param(name, prior_range[0])
        eng_hi_t = _transform_param(name, prior_range[1])
        lo = max(min(eng_lo_t, eng_hi_t), lo)
        hi = min(max(eng_lo_t, eng_hi_t), hi)
        if lo > hi:
            proposals.append(
                {
                    "parameter": name,
                    "status": "inconclusive",
                    "recommended_range": prior_range,
                    "sampling": prior["sampling"],
                    "evidence_rule": "safe_interval_outside_engineering_prior",
                    "support_counts": {"safe": len(safe), "adverse": len(adverse), "min_safe_support": int(min_safe_support)},
                    "supporting_source_ids": [o.source_id for o in safe],
                    "excluded_regions": [],
                    "confidence": "low",
                    "validation_needed": "inspect parameter extraction and decide whether to widen engineering prior before sampling",
                }
            )
            continue
        rec = [_inverse_transform_param(name, lo), _inverse_transform_param(name, hi)]
        if rec[0] > rec[1]:
            rec = [rec[1], rec[0]]
        proposals.append(
            {
                "parameter": name,
                "status": "recommended",
                "recommended_range": rec,
                "sampling": prior["sampling"],
                "evidence_rule": "deterministic_robust_safe_interval",
                "support_counts": {"safe": len(safe), "adverse": len(adverse), "min_safe_support": int(min_safe_support)},
                "supporting_source_ids": [o.source_id for o in safe],
                "excluded_regions": [],
                "confidence": "medium" if len(safe) >= max(10, min_safe_support) else "low",
                "validation_needed": "leave-one-source-batch-out and sentinel trial near proposed boundary",
            }
        )
    return proposals


def _build_surface(bound_proposals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    proposal_by_name = {str(p["parameter"]): p for p in bound_proposals}
    tunables: dict[str, Any] = {}
    for name in ENGINEERING_PRIORS:
        proposal = proposal_by_name.get(name, {})
        tunables[name] = {
            "range": proposal.get("recommended_range", ENGINEERING_PRIORS[name]["range"]),
            "sampling": proposal.get("sampling", ENGINEERING_PRIORS[name]["sampling"]),
            "source": proposal.get("evidence_rule", "engineering_prior"),
            "status": proposal.get("status", "inconclusive"),
        }
    return {
        "schema": "paper_i_hh_snake_optuna_surface_v1",
        "surface_kind": "six_effective_knob_first_pass",
        "fixed_settings": {
            "phase2_gamma_N": 1.0,
            "phase2_gamma_N_schedule_mode": "fixed",
            "phase2_motif_bonus_weight": 0.0,
            "motif_path_priors": "disabled",
            "phase3_enable_batching": True,
            "phase3_batch_selection_mode": "reduced_plane",
            "phase3_batch_prefilter_mode": "off",
            "adapt_beam_enabled": True,
            "phase1_prune_enabled": True,
            "phase1_prune_policy": "recoverability_ladder_v1",
            "fixed_inner_optimizer": "SPSA",
            "runtime_split": "fixed_not_sampled",
            "compiled_cost_policy": "posthoc_shortlisted_prefixes",
        },
        "tunables": tunables,
        "field_mapping": {
            "batch_near_degenerate_ratio_shared": ["phase2_batch_near_degenerate_ratio", "phase3_batch_near_degenerate_ratio_alias"],
            "batch_rank_rel_tol_shared": ["phase2_batch_rank_rel_tol", "phase3_batch_rank_rel_tol_alias"],
            "batch_additivity_slack_scale": ["phase2_batch_additivity_tol = source_default * scale", "phase3_batch_additivity_tol_alias = same"],
            "prune_recoverability_slack_scale": ["phase1_prune_max_regression = source_default * 10^u", "phase1_prune_retained_gain_ratio = 1 - (1-source_default) * 10^u"],
        },
    }


def analyze_sources(
    source_paths: Sequence[str | Path],
    *,
    min_safe_support: int = 6,
    epsilons: Sequence[float] = _DEFAULT_EPSILONS,
    comparison_mode: str = "snake_incumbent",
    fixed_anchor_rows: Sequence[Mapping[str, Any]] | None = None,
    incumbent_source_paths: Sequence[str | Path] | None = None,
) -> AnalysisResult:
    if comparison_mode not in COMPARISON_MODES:
        raise ValueError(f"comparison_mode must be one of {COMPARISON_MODES}, got {comparison_mode!r}")
    sources: list[SourceRecord] = []
    settings: list[SettingsRecord] = []
    prefix_points: list[PrefixPoint] = []
    for raw_path in source_paths:
        path = Path(raw_path)
        payload = _load_jsonish(path)
        if not isinstance(payload, Mapping):
            payload = {"raw_payload": payload}
        source, setting, points = _extract_source_and_settings(path, payload)
        sources.append(source)
        settings.append(setting)
        prefix_points.extend(points)
    fixed_anchors = _fixed_anchors_from_rows(fixed_anchor_rows)
    inferred_anchors = _build_comparison_anchors(
        sources,
        prefix_points,
        comparison_mode=comparison_mode,
        incumbent_source_paths=incumbent_source_paths,
    )
    anchors_by_key: dict[tuple[str, Any], GeoAnchor] = {(a.regime, a.cutoff.get("n_ph")): a for a in inferred_anchors}
    for anchor in fixed_anchors:
        anchors_by_key[(anchor.regime, anchor.cutoff.get("n_ph"))] = anchor
    anchors = list(anchors_by_key.values())
    outcomes = _compute_outcomes(sources, settings, prefix_points, anchors, epsilons, comparison_mode=comparison_mode)
    outcomes, pareto = _attach_pareto_contributors(outcomes, prefix_points)
    bound_proposals = _build_bound_proposals(outcomes, settings, min_safe_support=min_safe_support)
    surface = _build_surface(bound_proposals)
    summary = {
        "pipeline": _PIPELINE_NAME,
        "comparison_mode": comparison_mode,
        "source_count": len(sources),
        "settings_count": len(settings),
        "prefix_point_count": len(prefix_points),
        "anchor_count": len(anchors),
        "fixed_anchor_count": len(fixed_anchors),
        "inferred_anchor_count": len(inferred_anchors),
        "outcome_count": len(outcomes),
        "snake_count": sum(1 for s in sources if s.method == "snake"),
        "geo_anchor_count": sum(1 for a in anchors if a.method == "geo_adapt"),
        "snake_incumbent_anchor_count": sum(1 for a in anchors if a.method == "snake_incumbent"),
        "bound_parameters": list(ENGINEERING_PRIORS),
        "surface_kind": surface["surface_kind"],
    }
    surface["comparison_mode"] = comparison_mode
    surface["anchor_policy"] = (
        "fixed Paper-I anchors override inferred anchors" if fixed_anchors else "inferred from source set"
    )
    return AnalysisResult(
        sources=sources,
        settings=settings,
        prefix_points=prefix_points,
        anchors=anchors,
        outcomes=outcomes,
        pareto_contributors=pareto,
        bound_proposals=bound_proposals,
        surface=surface,
        summary=summary,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def _surface_yaml(surface: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"schema: {surface['schema']}")
    lines.append(f"surface_kind: {surface['surface_kind']}")
    if "comparison_mode" in surface:
        lines.append(f"comparison_mode: {surface['comparison_mode']}")
    if "anchor_policy" in surface:
        lines.append(f"anchor_policy: {json.dumps(surface['anchor_policy'])}")
    lines.append("fixed_settings:")
    for key, value in surface["fixed_settings"].items():
        lines.append(f"  {key}: {json.dumps(value)}")
    lines.append("tunables:")
    for key, spec in surface["tunables"].items():
        rng = spec["range"]
        lines.append(f"  {key}:")
        lines.append(f"    range: [{rng[0]}, {rng[1]}]")
        lines.append(f"    sampling: {spec['sampling']}")
        lines.append(f"    source: {spec['source']}")
        lines.append(f"    status: {spec['status']}")
    lines.append("field_mapping:")
    for key, values in surface["field_mapping"].items():
        lines.append(f"  {key}:")
        for value in values:
            lines.append(f"    - {json.dumps(value)}")
    return "\n".join(lines) + "\n"


def write_analysis_outputs(result: AnalysisResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "source_manifest.ndjson", result.sources)
    _write_jsonl(output_dir / "settings.ndjson", result.settings)
    _write_jsonl(output_dir / "prefix_points.ndjson", result.prefix_points)
    _write_jsonl(output_dir / "anchors.ndjson", result.anchors)
    _write_jsonl(output_dir / "trial_outcomes.ndjson", result.outcomes)
    _write_json(output_dir / "pareto_fronts_by_regime.json", result.pareto_contributors)
    _write_json(output_dir / "bound_proposals.json", result.bound_proposals)
    _write_json(output_dir / "summary.json", result.summary)
    (output_dir / "optuna_surface.yaml").write_text(_surface_yaml(result.surface), encoding="utf-8")
    _write_bound_support_csv(output_dir / "bound_support_table.csv", result.bound_proposals)


def _write_bound_support_csv(path: Path, proposals: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["parameter", "status", "range_lo", "range_hi", "sampling", "safe", "adverse", "confidence", "evidence_rule"],
        )
        writer.writeheader()
        for proposal in proposals:
            rng = proposal.get("recommended_range", [None, None])
            counts = proposal.get("support_counts", {}) if isinstance(proposal.get("support_counts"), Mapping) else {}
            writer.writerow(
                {
                    "parameter": proposal.get("parameter"),
                    "status": proposal.get("status"),
                    "range_lo": rng[0] if isinstance(rng, Sequence) and len(rng) >= 1 else None,
                    "range_hi": rng[1] if isinstance(rng, Sequence) and len(rng) >= 2 else None,
                    "sampling": proposal.get("sampling"),
                    "safe": counts.get("safe"),
                    "adverse": counts.get("adverse"),
                    "confidence": proposal.get("confidence"),
                    "evidence_rule": proposal.get("evidence_rule"),
                }
            )


def _expand_source_args(paths: Sequence[str], globs: Sequence[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            raise ValueError(f"Source path is a directory; pass explicit files or --source-glob: {path}")
        out.append(path)
    for pattern in globs:
        for match in sorted(glob.glob(str(Path(pattern).expanduser()))):
            path = Path(match)
            if path.is_file():
                out.append(path)
    seen: set[str] = set()
    dedup: list[Path] = []
    for path in out:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        dedup.append(path)
    return dedup


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", action="append", default=[], help="Explicit JSON/JSONL evidence file. May be repeated.")
    p.add_argument("--source-glob", action="append", default=[], help="File glob for JSON/JSONL evidence. Must resolve to files.")
    p.add_argument("--output-dir", required=True, help="Directory for ledger and surface outputs.")
    p.add_argument("--min-safe-support", type=int, default=6, help="Minimum safe rows before data-derived interval replaces engineering prior.")
    p.add_argument("--epsilon", action="append", type=float, default=[], help="Useful-prefix log10 tolerance. May be repeated; defaults to 0.03,0.05,0.10.")
    p.add_argument("--comparison-mode", choices=COMPARISON_MODES, default="snake_incumbent", help="Anchor semantics for parity/win labels; default searches against current SNAKE incumbent.")
    p.add_argument("--fixed-anchor-json", action="append", default=[], help="JSON file containing fixed Paper-I comparator anchors. Fixed anchors override inferred anchors by regime/cutoff.")
    p.add_argument("--fixed-anchor-set", default=None, help="Named anchor_sets key to read from --fixed-anchor-json, e.g. snake_incumbent or paper_i_geo_useful.")
    p.add_argument("--incumbent-source", action="append", default=[], help="Optional SNAKE source file(s) to use as incumbent anchors when --comparison-mode=snake_incumbent.")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    source_paths = _expand_source_args(args.source, args.source_glob)
    if not source_paths:
        raise SystemExit("No source files provided. Use --source or --source-glob.")
    eps = tuple(args.epsilon) if args.epsilon else _DEFAULT_EPSILONS
    anchor_rows: list[Mapping[str, Any]] = []
    for raw_anchor in args.fixed_anchor_json:
        anchor_rows.extend(_load_fixed_anchor_rows(Path(raw_anchor).expanduser(), anchor_set=args.fixed_anchor_set))
    result = analyze_sources(
        source_paths,
        min_safe_support=int(args.min_safe_support),
        epsilons=eps,
        comparison_mode=str(args.comparison_mode),
        fixed_anchor_rows=anchor_rows,
        incumbent_source_paths=args.incumbent_source,
    )
    write_analysis_outputs(result, Path(args.output_dir).expanduser())


if __name__ == "__main__":  # pragma: no cover
    main()
