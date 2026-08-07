#!/usr/bin/env python3
"""Offline prune-risk telemetry and motif prefilter helpers for Paper-I SNAKE."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PRUNE_PREFILTER_OFF = "off"
PRUNE_PREFILTER_MOTIF_RISK_V1 = "motif_risk_v1"
PRUNE_MOTIF_RISK_PROFILE_SCHEMA = "prune_motif_risk_profile_v1"
PRUNE_NOMINATION_RULE_SCHEMA = "prune_nomination_rule_v1"
PRUNE_NOMINATION_TUNING_REPORT_SCHEMA = "prune_nomination_tuning_report_v1"

_CHILD_SET_RE = re.compile(r"child_set\[(\d+)\]")


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _maybe_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _maybe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def prune_motif_key(label: str | None) -> tuple[str, str, str]:
    """Return a stable coarse motif key for SNAKE prune labels."""

    text = str(label or "")
    namespace = "unknown"
    if "::" in text:
        namespace = text.split("::", 1)[0]
    elif ":" in text:
        namespace = text.split(":", 1)[0]
    elif text:
        namespace = text.split("(", 1)[0]

    scope = "child_set" if "child_set" in text else "base"
    child_match = _CHILD_SET_RE.search(text)
    if child_match is not None:
        subtype = f"child_set[{child_match.group(1)}]"
    else:
        tail = text
        if "::" in tail:
            tail = tail.split("::", 1)[1]
        elif ":" in tail:
            tail = tail.split(":", 1)[1]
        subtype = tail.split("(", 1)[0].strip() or "unknown"
    return (str(scope), str(namespace or "unknown"), str(subtype or "unknown"))


def motif_key_id(key: Sequence[str]) -> str:
    parts = [str(x) for x in key]
    while len(parts) < 3:
        parts.append("unknown")
    return "|".join(parts[:3])


def motif_id_for_label(label: str | None) -> str:
    return motif_key_id(prune_motif_key(label))


def _candidate_label_from_row(row: Mapping[str, Any]) -> str:
    for key in (
        "label",
        "candidate_label",
        "selected_label",
        "probe_label",
        "candidate",
        "candidate_label_pretty",
    ):
        value = row.get(key)
        if value is not None and str(value):
            return str(value)
    return ""


def _candidate_index_from_row(row: Mapping[str, Any], *, default: int = -1) -> int:
    for key in ("index", "candidate_index", "selected_index", "probe_index"):
        if key in row:
            return _maybe_int(row.get(key), default)
    return int(default)


def _copy_feature(
    out: dict[str, Any],
    row: Mapping[str, Any],
    source_key: str,
    target_key: str,
) -> None:
    if source_key in row and row.get(source_key) is not None:
        value = row.get(source_key)
        if isinstance(value, (str, bool, int)):
            out[target_key] = value
        elif isinstance(value, float):
            if math.isfinite(float(value)):
                out[target_key] = float(value)
        else:
            maybe = _maybe_float(value, None)
            if maybe is not None:
                out[target_key] = float(maybe)


def _store_candidate_features(
    by_index: dict[int, dict[str, Any]],
    by_label: dict[str, dict[str, Any]],
    *,
    index: int,
    label: str,
    features: Mapping[str, Any],
) -> None:
    clean = {str(k): v for k, v in features.items() if v is not None}
    if int(index) >= 0:
        by_index.setdefault(int(index), {}).update(clean)
    if label:
        by_label.setdefault(str(label), {}).update(clean)


def _build_candidate_feature_resolver(
    prune_payload: Mapping[str, Any],
) -> Any:
    """Build a lightweight lookup for pre-ablation prune nomination features."""

    by_index: dict[int, dict[str, Any]] = {}
    by_label: dict[str, dict[str, Any]] = {}

    def add_rows(
        rows_key: str,
        *,
        fallback_index_from_position: bool,
        feature_map: Mapping[str, str],
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        for position, raw in enumerate(_as_sequence(prune_payload.get(rows_key))):
            if not isinstance(raw, Mapping):
                continue
            label = _candidate_label_from_row(raw)
            index = _candidate_index_from_row(
                raw,
                default=(int(position) if fallback_index_from_position else -1),
            )
            features: dict[str, Any] = {}
            for source_key, target_key in feature_map.items():
                _copy_feature(features, raw, source_key, target_key)
            if extra:
                features.update(dict(extra))
            _store_candidate_features(
                by_index,
                by_label,
                index=int(index),
                label=str(label),
                features=features,
            )

    add_rows(
        "metadata",
        fallback_index_from_position=True,
        feature_map={
            "generator_id": "generator_id",
            "admission_step": "admission_step",
            "first_seen_step": "first_seen_step",
            "selector_score": "selector_score",
            "selector_burden": "selector_burden",
            "cooldown_remaining": "cooldown_remaining",
        },
        extra={"feature_source_metadata": True},
    )
    add_rows(
        "gate_rows",
        fallback_index_from_position=False,
        feature_map={
            "first_seen_age": "first_seen_age",
            "admission_age": "admission_age",
            "cooldown_remaining": "cooldown_remaining",
            "selector_burden": "selector_burden",
            "protected": "protected",
            "cooldown_blocked": "cooldown_blocked",
            "recoverability_eligible": "recoverability_eligible",
        },
        extra={"feature_source_gate": True},
    )
    add_rows(
        "frozen_scores",
        fallback_index_from_position=False,
        feature_map={
            "frozen_energy": "frozen_energy",
            "frozen_regression": "frozen_regression",
            "selector_burden": "selector_burden",
            "cheap_prune_score": "cheap_prune_score",
        },
        extra={"feature_source_frozen_delete_probe": True},
    )
    for raw in _as_sequence(prune_payload.get("frozen_scores")):
        if not isinstance(raw, Mapping):
            continue
        label = _candidate_label_from_row(raw)
        index = _candidate_index_from_row(raw, default=-1)
        refit_window = _as_sequence(raw.get("refit_window_indices"))
        _store_candidate_features(
            by_index,
            by_label,
            index=index,
            label=label,
            features={"frozen_refit_window_size": int(len(refit_window))},
        )
    for rank, idx in enumerate(_as_sequence(prune_payload.get("probe_indices"))):
        _store_candidate_features(
            by_index,
            by_label,
            index=_maybe_int(idx, -1),
            label="",
            features={"probe_rank": int(rank)},
        )
    for rank, label in enumerate(_as_sequence(prune_payload.get("probe_labels"))):
        _store_candidate_features(
            by_index,
            by_label,
            index=-1,
            label=str(label),
            features={"probe_rank": int(rank)},
        )
    for raw in _as_sequence(prune_payload.get("candidate_nomination_sources")):
        if not isinstance(raw, Mapping):
            continue
        label = _candidate_label_from_row(raw)
        index = _candidate_index_from_row(raw, default=-1)
        lanes = [
            str(x)
            for x in _as_sequence(raw.get("lanes"))
            if x is not None and str(x)
        ]
        _store_candidate_features(
            by_index,
            by_label,
            index=index,
            label=label,
            features={
                "nomination_lanes": lanes,
                "nomination_authority": str(raw.get("authority", "")),
            },
        )
    selected_index = _maybe_int(prune_payload.get("selected_index"), -1)
    selected_label = str(prune_payload.get("selected_label") or "")
    if selected_index >= 0 or selected_label:
        _store_candidate_features(
            by_index,
            by_label,
            index=selected_index,
            label=selected_label,
            features={"selected_by_prune_payload": True},
        )
    schur_payload = _as_mapping(prune_payload.get("schur_surrogate_nomination"))
    for raw in _as_sequence(schur_payload.get("rows")):
        if not isinstance(raw, Mapping):
            continue
        label = _candidate_label_from_row(raw)
        index = _candidate_index_from_row(raw, default=-1)
        features: dict[str, Any] = {
            "schur_nomination_active": bool(schur_payload.get("active", False)),
            "schur_used_for_nomination": bool(schur_payload.get("used_for_nomination", False)),
        }
        for source_key, target_key in {
            "score": "schur_score",
            "schur_min": "schur_min",
            "schur_health": "schur_health",
            "schur_monotone": "schur_monotone",
            "surrogate_authority": "schur_surrogate_authority",
            "used_for_acceptance": "schur_used_for_acceptance",
        }.items():
            _copy_feature(features, raw, source_key, target_key)
        window_sizes = [
            int(x)
            for x in _as_sequence(raw.get("window_sizes"))
            if _maybe_int(x, -1) >= 0
        ]
        if window_sizes:
            features["schur_window_size_max"] = int(max(window_sizes))
            features["schur_window_size_min"] = int(min(window_sizes))
        _store_candidate_features(
            by_index,
            by_label,
            index=index,
            label=label,
            features=features,
        )

    def resolve(index: int, label: str) -> dict[str, Any]:
        features: dict[str, Any] = {}
        if int(index) >= 0:
            features.update(by_index.get(int(index), {}))
        if label:
            features.update(by_label.get(str(label), {}))
        selector_score = _maybe_float(features.get("selector_score"), None)
        selector_burden = _maybe_float(features.get("selector_burden"), 0.0)
        if selector_score is not None:
            features["selector_value_per_burden"] = float(selector_score) / float(
                1.0 + max(0.0, float(selector_burden or 0.0))
            )
        return features

    return resolve


def _iter_adapt_history_rows(payload: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    root = _as_mapping(payload.get("adapt_vqe") if isinstance(payload, Mapping) else {})
    if not root:
        root = payload
    rows = _as_sequence(root.get("history"))
    if not rows:
        rows = _as_sequence(root.get("history_tail"))
    for row in rows:
        if isinstance(row, Mapping):
            yield row


def _prune_payload_from_history_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("post_admission_prune", "phase1_prune", "prune_summary"):
        value = row.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _decision_records(
    *,
    prune_payload: Mapping[str, Any],
    step: int,
    history_position: int,
    source_path: str | None,
) -> list[dict[str, Any]]:
    decisions = _as_sequence(prune_payload.get("decisions"))
    out: list[dict[str, Any]] = []
    feature_resolver = _build_candidate_feature_resolver(prune_payload)
    if decisions:
        decision_iterable = list(decisions)
    else:
        grouped: dict[tuple[int, str], dict[str, Any]] = {}
        for rung_row in _as_sequence(prune_payload.get("recoverability_ladder_rows")):
            if not isinstance(rung_row, Mapping):
                continue
            label = str(
                rung_row.get("candidate_label")
                or rung_row.get("label")
                or rung_row.get("selected_label")
                or ""
            )
            index = _maybe_int(rung_row.get("candidate_index", rung_row.get("index")), -1)
            if not label:
                continue
            key = (int(index), str(label))
            current = grouped.setdefault(
                key,
                {
                    "index": int(index),
                    "label": str(label),
                    "accepted": False,
                    "reason": str(rung_row.get("reason", "")),
                },
            )
            if rung_row.get("rung_kind") is not None:
                current["rung_kind"] = str(rung_row.get("rung_kind"))
            if bool(rung_row.get("accepted", False)):
                current["accepted"] = True
                current["reason"] = str(rung_row.get("reason", "accepted"))
        decision_iterable = list(grouped.values())
    for decision in decision_iterable:
        if not isinstance(decision, Mapping):
            continue
        label = (
            decision.get("label")
            or decision.get("candidate_label")
            or decision.get("selected_label")
            or ""
        )
        if not label:
            continue
        index = _maybe_int(decision.get("index", decision.get("candidate_index")), -1)
        accepted = bool(decision.get("accepted", False))
        row = {
            "schema": "prune_candidate_record_v1",
            "step": int(step),
            "history_position": int(history_position),
            "source_path": source_path,
            "candidate_index": int(index),
            "label": str(label),
            "motif_key": list(prune_motif_key(str(label))),
            "motif_id": motif_id_for_label(str(label)),
            "outcome": "accepted" if accepted else "rollback_rejected",
            "accepted": bool(accepted),
            "rollback_rejected": not bool(accepted),
            "legacy_inferred": False,
            "reason": str(decision.get("reason", "")),
        }
        row.update(feature_resolver(int(index), str(label)))
        for source_key, target_key in {
            "regression": "delete_refit_regression",
            "energy_before": "delete_refit_energy_before",
            "energy_after": "delete_refit_energy_after",
            "regression_threshold": "delete_refit_regression_threshold",
            "retained_gain": "delete_refit_retained_gain",
            "retained_gain_threshold": "delete_refit_retained_gain_threshold",
            "rung_index": "delete_refit_rung_index",
            "rung_kind": "delete_refit_rung_kind",
            "safe_regression_ok": "delete_refit_safe_regression_ok",
            "retained_gain_ok": "delete_refit_retained_gain_ok",
            "curvature_guard_ok": "delete_refit_curvature_guard_ok",
            "confidence_guard_ok": "delete_refit_confidence_guard_ok",
        }.items():
            _copy_feature(row, decision, source_key, target_key)
        out.append(row)
    return out


def _legacy_prune_record(
    *,
    prune_payload: Mapping[str, Any],
    step: int,
    history_position: int,
    source_path: str | None,
) -> dict[str, Any] | None:
    selected_label = prune_payload.get("selected_label")
    selected_index = _maybe_int(prune_payload.get("selected_index"), -1)
    if not selected_label:
        probe_labels = _as_sequence(prune_payload.get("probe_labels"))
        if probe_labels:
            selected_label = probe_labels[0]
            probe_indices = _as_sequence(prune_payload.get("probe_indices"))
            if probe_indices:
                selected_index = _maybe_int(probe_indices[0], selected_index)
    if not selected_label:
        return None
    accepted = _maybe_int(prune_payload.get("accepted_count"), 0) > 0
    feature_resolver = _build_candidate_feature_resolver(prune_payload)
    row = {
        "schema": "prune_candidate_record_v1",
        "step": int(step),
        "history_position": int(history_position),
        "source_path": source_path,
        "candidate_index": int(selected_index),
        "label": str(selected_label),
        "motif_key": list(prune_motif_key(str(selected_label))),
        "motif_id": motif_id_for_label(str(selected_label)),
        "outcome": "accepted" if accepted else "rollback_rejected",
        "accepted": bool(accepted),
        "rollback_rejected": not bool(accepted),
        "legacy_inferred": True,
        "reason": str(prune_payload.get("permission_reason", "legacy_selected_label")),
    }
    row.update(feature_resolver(int(selected_index), str(selected_label)))
    return row


def extract_prune_candidate_records(
    payload: Mapping[str, Any],
    *,
    source_path: str | None = None,
    max_step: int | None = None,
    include_legacy_inferred: bool = True,
) -> list[dict[str, Any]]:
    """Extract accepted/rejected prune candidate rows from ADAPT result JSON."""

    records: list[dict[str, Any]] = []
    for history_position, row in enumerate(_iter_adapt_history_rows(payload)):
        prune_payload = _prune_payload_from_history_row(row)
        if not prune_payload:
            continue
        if not bool(prune_payload.get("executed", False)):
            continue
        step = _maybe_int(row.get("depth"), history_position + 1)
        if max_step is not None and int(step) > int(max_step):
            continue
        explicit = _decision_records(
            prune_payload=prune_payload,
            step=int(step),
            history_position=int(history_position),
            source_path=source_path,
        )
        if explicit:
            records.extend(explicit)
            continue
        if include_legacy_inferred and _maybe_int(prune_payload.get("candidate_count"), 0) > 0:
            legacy = _legacy_prune_record(
                prune_payload=prune_payload,
                step=int(step),
                history_position=int(history_position),
                source_path=source_path,
            )
            if legacy is not None:
                records.append(legacy)
    return records


def build_motif_risk_profile(
    records: Iterable[Mapping[str, Any]],
    *,
    default_unseen_risk: float = 1.0,
) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = {}
    labels_by_motif: dict[str, set[str]] = {}
    legacy_by_motif: dict[str, int] = {}
    for raw in records:
        record = _as_mapping(raw)
        motif_id = str(record.get("motif_id") or motif_id_for_label(str(record.get("label", ""))))
        if not motif_id:
            continue
        bucket = counts.setdefault(motif_id, Counter())
        if bool(record.get("accepted", False)):
            bucket["accepted"] += 1
        elif bool(record.get("rollback_rejected", False)) or str(record.get("outcome", "")) != "":
            bucket["rollback_rejected"] += 1
        labels_by_motif.setdefault(motif_id, set()).add(str(record.get("label", "")))
        if bool(record.get("legacy_inferred", False)):
            legacy_by_motif[motif_id] = legacy_by_motif.get(motif_id, 0) + 1

    motifs: dict[str, dict[str, Any]] = {}
    for motif_id, bucket in sorted(counts.items()):
        accepted = int(bucket.get("accepted", 0))
        rejected = int(bucket.get("rollback_rejected", 0))
        total = int(accepted + rejected)
        risk = float(default_unseen_risk) if total <= 0 else float(rejected) / float(total)
        motifs[str(motif_id)] = {
            "motif_id": str(motif_id),
            "motif_key": str(motif_id).split("|"),
            "accepted_count": int(accepted),
            "rollback_rejected_count": int(rejected),
            "total_count": int(total),
            "risk": float(risk),
            "legacy_inferred_count": int(legacy_by_motif.get(motif_id, 0)),
            "example_labels": sorted(label for label in labels_by_motif.get(motif_id, set()) if label)[:8],
        }
    return {
        "schema": PRUNE_MOTIF_RISK_PROFILE_SCHEMA,
        "default_unseen_risk": float(default_unseen_risk),
        "motifs": motifs,
    }


def load_prune_prefilter_profile(path: str | Path) -> dict[str, Any]:
    profile_path = Path(path)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Prune prefilter profile must be a JSON object: {profile_path}")
    schema = str(payload.get("schema", ""))
    if schema != PRUNE_MOTIF_RISK_PROFILE_SCHEMA:
        raise ValueError(
            f"Unsupported prune prefilter profile schema {schema!r}; expected {PRUNE_MOTIF_RISK_PROFILE_SCHEMA!r}"
        )
    motifs = payload.get("motifs")
    if not isinstance(motifs, Mapping):
        raise ValueError(f"Prune prefilter profile has no motifs object: {profile_path}")
    return dict(payload)


def _profile_motif_row(profile: Mapping[str, Any], motif_id: str) -> Mapping[str, Any]:
    motifs = _as_mapping(profile.get("motifs"))
    row = motifs.get(str(motif_id))
    return row if isinstance(row, Mapping) else {}


def _motif_stats(profile: Mapping[str, Any], motif_id: str) -> dict[str, Any]:
    motif_row = _profile_motif_row(profile, motif_id)
    accepted = _maybe_int(motif_row.get("accepted_count"), 0)
    rejected = _maybe_int(motif_row.get("rollback_rejected_count"), 0)
    total = _maybe_int(motif_row.get("total_count"), accepted + rejected)
    default_risk = _maybe_float(profile.get("default_unseen_risk"), 1.0)
    risk = _maybe_float(motif_row.get("risk"), default_risk if default_risk is not None else 1.0)
    return {
        "accepted_count": int(accepted),
        "rollback_rejected_count": int(rejected),
        "total_count": int(total),
        "risk": float(risk if risk is not None else 1.0),
        "seen": bool(total > 0),
    }


def _value_from_record(record: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key in record:
            value = _maybe_float(record.get(key), None)
            if value is not None:
                return float(value)
    return None


def _rule_numeric_checks(record: Mapping[str, Any], rule: Mapping[str, Any]) -> tuple[bool, str]:
    checks: list[tuple[str, Sequence[str], str]] = [
        ("selector_burden_max", ("selector_burden",), "selector_burden_above_max"),
        (
            "selector_value_per_burden_max",
            ("selector_value_per_burden",),
            "selector_value_per_burden_above_max",
        ),
        ("schur_score_max", ("schur_score", "schur_min"), "schur_score_above_max"),
    ]
    for rule_key, record_keys, reason in checks:
        if rule.get(rule_key) is None:
            continue
        threshold = _maybe_float(rule.get(rule_key), None)
        value = _value_from_record(record, record_keys)
        if threshold is None or value is None:
            return False, f"{reason}_missing"
        if rule_key.endswith("_min"):
            if float(value) < float(threshold) - 1e-15:
                return False, reason
        elif float(value) > float(threshold) + 1e-15:
            return False, reason
    return True, "allowed"


def record_passes_nomination_rule(
    record: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
    rule: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    """Evaluate a non-authoritative, pre-ablation nomination filter rule."""

    motif_id = str(record.get("motif_id") or motif_id_for_label(str(record.get("label", ""))))
    stats = _motif_stats(profile, motif_id)
    risk_threshold = _maybe_float(rule.get("risk_threshold"), 0.0)
    if risk_threshold is None:
        risk_threshold = 0.0
    policy = str(rule.get("motif_policy", "clean_prior_motif"))
    accepted_count = int(stats["accepted_count"])
    rejected_count = int(stats["rollback_rejected_count"])
    risk = float(stats["risk"])
    if policy == "allow_all":
        motif_ok = True
        reason = "allowed"
    elif policy == "block_prior_reject":
        motif_ok = bool(rejected_count == 0)
        reason = "prior_rollback_rejection" if not motif_ok else "allowed"
    elif policy == "prior_accept_risk":
        motif_ok = bool(accepted_count > 0 and risk <= float(risk_threshold))
        if accepted_count <= 0:
            reason = "no_prior_acceptance"
        elif risk > float(risk_threshold):
            reason = "risk_above_threshold"
        else:
            reason = "allowed"
    else:
        motif_ok = bool(accepted_count > 0 and rejected_count == 0 and risk <= float(risk_threshold))
        if rejected_count > 0:
            reason = "prior_rollback_rejection"
        elif accepted_count <= 0:
            reason = "no_prior_acceptance"
        elif risk > float(risk_threshold):
            reason = "risk_above_threshold"
        else:
            reason = "allowed"
    if not motif_ok:
        return False, reason, stats
    numeric_ok, numeric_reason = _rule_numeric_checks(record, rule)
    if not numeric_ok:
        return False, numeric_reason, stats
    return True, "allowed", stats


def filter_prune_candidates_by_motif_risk(
    *,
    candidate_indices: Sequence[int],
    labels: Sequence[str],
    profile: Mapping[str, Any],
    risk_threshold: float = 0.0,
    max_candidates: int = 1,
    candidate_features: Mapping[int, Mapping[str, Any]] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Filter candidate indices before delete/refit trials using motif risk."""

    threshold = float(max(0.0, risk_threshold))
    cap = int(max(0, max_candidates))
    default_risk = _maybe_float(profile.get("default_unseen_risk"), 1.0)
    if default_risk is None:
        default_risk = 1.0
    allowed: list[int] = []
    rows: list[dict[str, Any]] = []
    active_rule = _as_mapping(profile.get("active_nomination_rule"))
    if active_rule.get("schema") != PRUNE_NOMINATION_RULE_SCHEMA:
        active_rule = {}
    elif active_rule.get("max_candidates_per_step") is not None:
        cap = int(max(0, _maybe_int(active_rule.get("max_candidates_per_step"), cap)))
    for raw_idx in candidate_indices:
        idx = int(raw_idx)
        label = str(labels[idx]) if 0 <= idx < len(labels) else ""
        motif_id = motif_id_for_label(label)
        feature_row = dict(_as_mapping((candidate_features or {}).get(int(idx))))
        feature_row.update({"label": str(label), "motif_id": str(motif_id)})
        if active_rule:
            row_allowed, reason, stats = record_passes_nomination_rule(
                feature_row,
                profile=profile,
                rule=active_rule,
            )
            accepted_count = int(stats["accepted_count"])
            rejected_count = int(stats["rollback_rejected_count"])
            total_count = int(stats["total_count"])
            risk = float(stats["risk"])
        else:
            motif_row = _profile_motif_row(profile, motif_id)
            accepted_count = _maybe_int(motif_row.get("accepted_count"), 0)
            rejected_count = _maybe_int(motif_row.get("rollback_rejected_count"), 0)
            total_count = _maybe_int(motif_row.get("total_count"), accepted_count + rejected_count)
            risk = _maybe_float(motif_row.get("risk"), float(default_risk))
            if risk is None:
                risk = float(default_risk)
            reason = "allowed"
            row_allowed = bool(
                accepted_count > 0
                and rejected_count == 0
                and float(risk) <= float(threshold)
            )
            if rejected_count > 0:
                reason = "prior_rollback_rejection"
            elif accepted_count <= 0:
                reason = "no_prior_acceptance"
            elif float(risk) > float(threshold):
                reason = "risk_above_threshold"
        row_allowed = bool(row_allowed and (cap <= 0 or len(allowed) < cap))
        if not row_allowed and cap > 0 and len(allowed) >= cap and reason == "allowed":
            reason = "prefilter_cap_reached"
        if row_allowed:
            allowed.append(int(idx))
        rows.append(
            {
                "index": int(idx),
                "label": str(label),
                "motif_id": str(motif_id),
                "motif_key": list(prune_motif_key(label)),
                "risk": float(risk),
                "accepted_count": int(accepted_count),
                "rollback_rejected_count": int(rejected_count),
                "total_count": int(total_count),
                "allowed": bool(row_allowed),
                "reason": str(reason),
                "active_nomination_rule": bool(active_rule),
            }
        )
    blocked = [row for row in rows if not bool(row.get("allowed", False))]
    telemetry = {
        "schema": "prune_prefilter_decision_rows_v1",
        "policy": PRUNE_PREFILTER_MOTIF_RISK_V1,
        "input_count": int(len(candidate_indices)),
        "allowed_count": int(len(allowed)),
        "blocked_count": int(len(blocked)),
        "risk_threshold": float(threshold),
        "max_candidates": int(cap),
        "rows": rows,
        "blocked_indices": [int(row["index"]) for row in blocked],
        "blocked_labels": [str(row["label"]) for row in blocked],
    }
    return allowed, telemetry


def _group_key(record: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(record.get("source_path") or ""),
        _maybe_int(record.get("history_position"), 0),
        _maybe_int(record.get("step"), 0),
    )


def evaluate_nomination_rule(
    records: Iterable[Mapping[str, Any]],
    *,
    profile: Mapping[str, Any],
    rule: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [dict(_as_mapping(record)) for record in records]
    cap = _maybe_int(rule.get("max_candidates_per_step"), 0)
    groups: dict[tuple[str, int, int], list[tuple[int, dict[str, Any]]]] = {}
    for ordinal, row in enumerate(rows):
        groups.setdefault(_group_key(row), []).append((int(ordinal), row))

    metrics = Counter()
    blocked_reasons: Counter[str] = Counter()
    allowed_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    for _key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        group_rows_sorted = sorted(
            group_rows,
            key=lambda item: (
                _maybe_int(item[1].get("probe_rank"), item[0]),
                item[0],
            ),
        )
        allowed_in_group = 0
        for _ordinal, row in group_rows_sorted:
            accepted = bool(row.get("accepted", False))
            rejected = bool(row.get("rollback_rejected", False)) or (
                str(row.get("outcome", "")) == "rollback_rejected"
            )
            metrics["record_count"] += 1
            if accepted:
                metrics["accepted_total"] += 1
            if rejected:
                metrics["rollback_rejected_total"] += 1
            allowed_pre_cap, reason, stats = record_passes_nomination_rule(
                row,
                profile=profile,
                rule=rule,
            )
            allowed = bool(allowed_pre_cap and (cap <= 0 or allowed_in_group < int(cap)))
            if allowed_pre_cap and not allowed and cap > 0:
                reason = "prefilter_cap_reached"
            if allowed:
                allowed_in_group += 1
                metrics["allowed_count"] += 1
                if accepted:
                    metrics["accepted_retained"] += 1
                if rejected:
                    metrics["rollback_rejected_retained"] += 1
                if len(allowed_rows) < 12:
                    allowed_rows.append(
                        {
                            "label": str(row.get("label", "")),
                            "motif_id": str(row.get("motif_id", "")),
                            "outcome": str(row.get("outcome", "")),
                            "step": _maybe_int(row.get("step"), 0),
                            "source_path": str(row.get("source_path") or ""),
                        }
                    )
            else:
                metrics["blocked_count"] += 1
                blocked_reasons[str(reason)] += 1
                if accepted:
                    metrics["accepted_blocked"] += 1
                if rejected:
                    metrics["rollback_rejected_blocked"] += 1
                if len(blocked_rows) < 12:
                    blocked_rows.append(
                        {
                            "label": str(row.get("label", "")),
                            "motif_id": str(row.get("motif_id", "")),
                            "outcome": str(row.get("outcome", "")),
                            "reason": str(reason),
                            "step": _maybe_int(row.get("step"), 0),
                            "source_path": str(row.get("source_path") or ""),
                            "motif_stats": dict(stats),
                        }
                    )
    accepted_total = int(metrics.get("accepted_total", 0))
    rejected_total = int(metrics.get("rollback_rejected_total", 0))
    accepted_retained = int(metrics.get("accepted_retained", 0))
    rejected_blocked = int(metrics.get("rollback_rejected_blocked", 0))
    for key in (
        "record_count",
        "accepted_total",
        "rollback_rejected_total",
        "allowed_count",
        "blocked_count",
        "accepted_retained",
        "accepted_blocked",
        "rollback_rejected_retained",
        "rollback_rejected_blocked",
    ):
        metrics.setdefault(key, 0)
    metrics["accepted_recall"] = (
        float(accepted_retained) / float(accepted_total) if accepted_total > 0 else 1.0
    )
    metrics["rollback_rejection_elimination"] = (
        float(rejected_blocked) / float(rejected_total) if rejected_total > 0 else 1.0
    )
    metrics["zero_rollback_rejections_retained"] = bool(
        int(metrics.get("rollback_rejected_retained", 0)) == 0
    )
    metrics["objective"] = float(
        1_000_000 * int(metrics.get("rollback_rejected_retained", 0))
        + 1_000 * int(metrics.get("accepted_blocked", 0))
        + int(metrics.get("allowed_count", 0))
    )
    return {
        "schema": "prune_nomination_rule_evaluation_v1",
        "rule": dict(rule),
        "metrics": {str(k): v for k, v in metrics.items()},
        "blocked_reasons": dict(sorted(blocked_reasons.items())),
        "sample_allowed_rows": allowed_rows,
        "sample_blocked_rows": blocked_rows,
    }


def _feature_values(records: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[float]:
    values: list[float] = []
    for record in records:
        value = _value_from_record(record, keys)
        if value is not None:
            values.append(float(value))
    return values


def _threshold_candidates(values: Sequence[float], *, max_count: int = 7) -> list[float]:
    finite = sorted({float(x) for x in values if math.isfinite(float(x))})
    if not finite:
        return []
    if len(finite) <= int(max_count):
        return finite
    picks: list[float] = []
    for frac in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        pos = int(round(float(frac) * float(len(finite) - 1)))
        picks.append(float(finite[max(0, min(pos, len(finite) - 1))]))
    return sorted({float(x) for x in picks})


def _nomination_rule_candidates(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selector_value_thresholds = _threshold_candidates(
        _feature_values(records, ("selector_value_per_burden",))
    )
    schur_thresholds = _threshold_candidates(_feature_values(records, ("schur_score", "schur_min")))
    rules: list[dict[str, Any]] = []

    def add(rule: Mapping[str, Any]) -> None:
        clean = {k: v for k, v in dict(rule).items() if v is not None}
        clean.setdefault("schema", PRUNE_NOMINATION_RULE_SCHEMA)
        clean.setdefault("risk_threshold", 0.0)
        clean.setdefault("max_candidates_per_step", 0)
        name_parts = [
            str(clean.get("motif_policy", "clean_prior_motif")),
            f"risk{float(clean.get('risk_threshold', 0.0)):.3g}",
            f"cap{int(clean.get('max_candidates_per_step', 0))}",
        ]
        for key in (
            "selector_value_per_burden_max",
            "schur_score_max",
        ):
            if key in clean:
                name_parts.append(f"{key}={clean[key]}")
        clean.setdefault("name", "_".join(name_parts))
        rules.append(clean)

    motif_policies = ("allow_all", "block_prior_reject", "prior_accept_risk", "clean_prior_motif")
    risk_thresholds = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
    caps = (0, 1, 2, 3)
    for motif_policy in motif_policies:
        for risk_threshold in risk_thresholds:
            for cap in caps:
                base = {
                    "motif_policy": motif_policy,
                    "risk_threshold": float(risk_threshold),
                    "max_candidates_per_step": int(cap),
                }
                add(base)
                for selector_value in selector_value_thresholds:
                    add({**base, "selector_value_per_burden_max": float(selector_value)})
                for schur_score in schur_thresholds:
                    add({**base, "schur_score_max": float(schur_score)})
    deduped: dict[str, dict[str, Any]] = {}
    for rule in rules:
        key = json.dumps({k: v for k, v in rule.items() if k != "name"}, sort_keys=True)
        deduped.setdefault(key, rule)
    return list(deduped.values())


def feature_availability(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    keys = (
        "selector_score",
        "selector_burden",
        "selector_value_per_burden",
        "first_seen_age",
        "admission_age",
        "schur_score",
        "schur_used_for_nomination",
        "frozen_regression",
        "cheap_prune_score",
    )
    counts = Counter()
    total = 0
    for record in records:
        total += 1
        for key in keys:
            if key in record and record.get(key) is not None:
                counts[key] += 1
    counts["record_count"] = int(total)
    return {str(k): int(v) for k, v in counts.items()}


def build_nomination_tuning_report(
    records: Iterable[Mapping[str, Any]],
    *,
    profile: Mapping[str, Any] | None = None,
    max_rules: int = 20,
) -> dict[str, Any]:
    rows = [dict(_as_mapping(record)) for record in records]
    profile_use = dict(profile) if isinstance(profile, Mapping) else build_motif_risk_profile(rows)
    baseline_rule = {
        "schema": PRUNE_NOMINATION_RULE_SCHEMA,
        "name": "baseline_allow_all",
        "motif_policy": "allow_all",
        "risk_threshold": 1.0,
        "max_candidates_per_step": 0,
    }
    strict_rule = {
        "schema": PRUNE_NOMINATION_RULE_SCHEMA,
        "name": "strict_clean_motif_cap1",
        "motif_policy": "clean_prior_motif",
        "risk_threshold": 0.0,
        "max_candidates_per_step": 1,
    }
    evaluations = [
        evaluate_nomination_rule(rows, profile=profile_use, rule=rule)
        for rule in _nomination_rule_candidates(rows)
    ]
    evaluations.sort(
        key=lambda item: (
            int(item["metrics"].get("rollback_rejected_retained", 0)),
            -int(item["metrics"].get("accepted_retained", 0)),
            int(item["metrics"].get("accepted_blocked", 0)),
            int(item["metrics"].get("allowed_count", 0)),
            str(item["rule"].get("name", "")),
        )
    )
    top = evaluations[: max(0, int(max_rules))]
    recommended = dict(top[0]["rule"]) if top else dict(strict_rule)
    outcome_totals = Counter()
    for record in rows:
        if bool(record.get("accepted", False)):
            outcome_totals["accepted"] += 1
        if bool(record.get("rollback_rejected", False)) or str(record.get("outcome", "")) == "rollback_rejected":
            outcome_totals["rollback_rejected"] += 1
    return {
        "schema": PRUNE_NOMINATION_TUNING_REPORT_SCHEMA,
        "record_count": int(len(rows)),
        "outcome_totals": {str(k): int(v) for k, v in sorted(outcome_totals.items())},
        "feature_availability": feature_availability(rows),
        "baseline": evaluate_nomination_rule(rows, profile=profile_use, rule=baseline_rule),
        "strict_motif_cap1": evaluate_nomination_rule(rows, profile=profile_use, rule=strict_rule),
        "top_rules": top,
        "recommended_rule": recommended,
        "note": (
            "Rules are nomination filters evaluated on historical attempted prune candidates. "
            "They intentionally do not use delete/refit outcome fields as decision features."
        ),
    }


def format_nomination_tuning_report_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Prune Nomination Tuning Report")
    lines.append("")
    lines.append(f"- schema: `{report.get('schema', '')}`")
    lines.append(f"- records: `{_maybe_int(report.get('record_count'), 0)}`")
    outcome = _as_mapping(report.get("outcome_totals"))
    lines.append(
        f"- outcomes: accepted `{_maybe_int(outcome.get('accepted'), 0)}`, "
        f"rollback rejected `{_maybe_int(outcome.get('rollback_rejected'), 0)}`"
    )
    recommended = _as_mapping(report.get("recommended_rule"))
    lines.append(f"- recommended rule: `{recommended.get('name', '')}`")
    lines.append("")
    lines.append("## Top Rules")
    lines.append("")
    lines.append(
        "| rank | rule | retained accepts | blocked accepts | retained rejects | blocked rejects | allowed |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for rank, item in enumerate(_as_sequence(report.get("top_rules")), start=1):
        if not isinstance(item, Mapping):
            continue
        rule = _as_mapping(item.get("rule"))
        metrics = _as_mapping(item.get("metrics"))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    f"`{rule.get('name', '')}`",
                    str(_maybe_int(metrics.get("accepted_retained"), 0)),
                    str(_maybe_int(metrics.get("accepted_blocked"), 0)),
                    str(_maybe_int(metrics.get("rollback_rejected_retained"), 0)),
                    str(_maybe_int(metrics.get("rollback_rejected_blocked"), 0)),
                    str(_maybe_int(metrics.get("allowed_count"), 0)),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Feature Availability")
    lines.append("")
    for key, value in sorted(_as_mapping(report.get("feature_availability")).items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append(str(report.get("note", "")))
    lines.append("")
    return "\n".join(lines)


def prune_telemetry_counts(payload: Mapping[str, Any]) -> dict[str, int]:
    """Return aggregate prune metrics from full or compact ADAPT telemetry."""

    candidate_count = 0
    accepted_count = 0
    rejected_delete_attempt_count = 0
    no_accept_restore_pass_count = 0
    accepted_then_guard_rolled_back_count = 0
    actual_rollback_count = 0
    prefilter_blocked_count = 0
    prefilter_allowed_count = 0
    for row in _iter_adapt_history_rows(payload):
        prune_payload = _prune_payload_from_history_row(row)
        if not prune_payload:
            continue
        candidate_count += _maybe_int(prune_payload.get("candidate_count"), 0)
        accepted_count += _maybe_int(prune_payload.get("accepted_count"), 0)
        prefilter_blocked_count += _maybe_int(prune_payload.get("prune_prefilter_blocked_count"), 0)
        prefilter_allowed_count += _maybe_int(prune_payload.get("prune_prefilter_allowed_count"), 0)
        decisions = _decision_records(
            prune_payload=prune_payload,
            step=_maybe_int(row.get("depth"), 0),
            history_position=0,
            source_path=None,
        )
        rejected_here = sum(1 for decision in decisions if bool(decision.get("rollback_rejected", False)))
        refit_rejected_here = sum(
            1
            for decision in decisions
            if bool(decision.get("rollback_rejected", False))
            and str(decision.get("delete_refit_rung_kind", "")) not in {"", "frozen_delete"}
        )
        if rejected_here <= 0:
            rejected_here = _maybe_int(prune_payload.get("rejected_delete_attempt_count"), 0)
        if rejected_here <= 0 and bool(prune_payload.get("rollback_snapshot_restored", False)):
            rejected_here = 1
        rejected_delete_attempt_count += int(rejected_here)
        snapshot_refit_restored = bool(
            prune_payload.get("rollback_snapshot_restored", False)
        ) and bool(prune_payload.get("post_refit_executed", False))
        if snapshot_refit_restored and _maybe_int(prune_payload.get("accepted_count"), 0) <= 0:
            no_accept_restore_pass_count += 1
            actual_rollback_count += int(max(1, int(refit_rejected_here)))
        elif int(refit_rejected_here) > 0:
            actual_rollback_count += int(refit_rejected_here)
        if bool(prune_payload.get("rolled_back", False)) and _maybe_int(prune_payload.get("accepted_count"), 0) > 0:
            accepted_then_guard_rolled_back_count += 1
            actual_rollback_count += 1
    return {
        "prune_candidate_count": int(candidate_count),
        "prune_accepted_count": int(accepted_count),
        "prune_rejected_delete_attempt_count": int(rejected_delete_attempt_count),
        "prune_no_accept_restore_pass_count": int(no_accept_restore_pass_count),
        "prune_accepted_then_guard_rolled_back_count": int(accepted_then_guard_rolled_back_count),
        "prune_actual_rollback_count": int(actual_rollback_count),
        "prune_prefilter_blocked_count": int(prefilter_blocked_count),
        "prune_prefilter_allowed_count": int(prefilter_allowed_count),
    }


def _load_records_from_json_paths(paths: Sequence[Path], *, max_step: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            continue
        records.extend(
            extract_prune_candidate_records(
                payload,
                source_path=str(path),
                max_step=max_step,
                include_legacy_inferred=True,
            )
        )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_paths", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--records-jsonl", type=Path, default=None)
    parser.add_argument("--tuning-report-json", type=Path, default=None)
    parser.add_argument("--tuning-report-md", type=Path, default=None)
    parser.add_argument("--tuning-max-rules", type=int, default=20)
    parser.add_argument(
        "--activate-best-rule",
        action="store_true",
        help="Attach the best local tuning rule as active_nomination_rule in the output profile.",
    )
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--default-unseen-risk", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    records = _load_records_from_json_paths(args.json_paths, max_step=args.max_step)
    profile = build_motif_risk_profile(records, default_unseen_risk=float(args.default_unseen_risk))
    profile["source_json_paths"] = [str(Path(path)) for path in args.json_paths]
    profile["record_count"] = int(len(records))
    tuning_report: dict[str, Any] | None = None
    if (
        args.tuning_report_json is not None
        or args.tuning_report_md is not None
        or bool(args.activate_best_rule)
    ):
        tuning_report = build_nomination_tuning_report(
            records,
            profile=profile,
            max_rules=int(args.tuning_max_rules),
        )
        profile["recommended_nomination_rule"] = dict(tuning_report.get("recommended_rule", {}))
        if bool(args.activate_best_rule):
            profile["active_nomination_rule"] = dict(tuning_report.get("recommended_rule", {}))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    if args.records_jsonl is not None:
        args.records_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.records_jsonl.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    if tuning_report is not None and args.tuning_report_json is not None:
        args.tuning_report_json.parent.mkdir(parents=True, exist_ok=True)
        args.tuning_report_json.write_text(
            json.dumps(tuning_report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if tuning_report is not None and args.tuning_report_md is not None:
        args.tuning_report_md.parent.mkdir(parents=True, exist_ok=True)
        args.tuning_report_md.write_text(
            format_nomination_tuning_report_markdown(tuning_report),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
