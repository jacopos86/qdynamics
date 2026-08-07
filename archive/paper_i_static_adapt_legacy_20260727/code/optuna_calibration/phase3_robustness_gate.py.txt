"""Static ADAPT Phase3 Hubbard L2 robustness gate.

The gate has two layers:

* :func:`evaluate_phase3_robustness_payload` is a pure payload evaluator.  It
  inspects an ADAPT/result payload and returns a JSON-serializable verdict.
* :func:`run_hubbard_l2_robustness_gate` is an invocable preflight runner for
  tiny Hubbard L2 SPSA/POWELL lanes.  It writes ``robustness_gate.json`` and is
  intended to fail before broad static Optuna studies when required.

This module is intentionally static-ADAPT only.  It does not import or touch
``pipelines.time_dynamics``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.static_adapt.selector_measurement_proxy import (
    controller_proxy_from_adapt_payload,
    validate_controller_proxy_for_shot_objective,
)

PHASE3_ROBUSTNESS_GATE_SCHEMA = "phase3_hubbard_l2_robustness_gate_v1"
PHASE3_PAYLOAD_EVALUATION_SCHEMA = "phase3_static_adapt_payload_robustness_evaluation_v1"
PHASE3_HUBBARD_L2_SPSA_CALIBRATION_SCHEMA = "phase3_hubbard_l2_spsa_calibration_v1"
PHASE3_HUBBARD_L2_SPSA_CALIBRATION_CASE_SCHEMA = "phase3_hubbard_l2_spsa_calibration_case_v1"
PHASE3_CANONICAL_SCORE_FORMULA = "DeltaE_TR * N3 / (1 + K3)"

_CANONICAL_CONFIG_KEYS = (
    "canonical_score_formula",
    "primary_selector_score_key",
    "selector_tie_break_score_key",
    "auxiliary_terms_primary_mode",
)
_PHASE3_ROW_KEYS = (
    "phase3_primary_score",
    "phase3_tie_break_score",
    "phase3_auxiliary_score_mode",
    "phase3_canonical_score_formula",
)
_PHASE3_COMPONENT_KEYS = (
    "DeltaE_TR",
    "phase3_delta_e_tr",
    "confidence_factor",
    "phase3_confidence_factor",
    "N3",
    "phase3_N3",
    "K3",
    "phase3_K3",
    "denominator_1_plus_K3",
    "phase3_denominator_1_plus_K3",
    "phase3_reduced_trust_gain",
    "phase3_reduced_novelty",
    "phase3_burden_total",
    "selector_burden",
)
_METADATA_EXPRESSIVE_TOKENS = (
    "expressive",
    "uccsd",
    "double",
    "doubles",
    "two_body",
    "two-body",
    "twobody",
    "fermion_double",
    "cluster_double",
    "clustered_double",
    "t2",
    "quartic",
    "hubbard_interaction",
    "onsite_interaction",
    "density_assisted",
    "spin_exchange",
    "pair_hop",
    "pair_hopping",
    "pair_bridge",
    "bridge",
)
_LABEL_DOUBLE_FALLBACK_TOKENS = (
    "double",
    "doubles",
    "doub",
    "uccsd",
    "t2",
)
_NON_SELECTION_EVIDENCE_KEYS = (
    "non_selection_evidence",
    "non_selection_reason",
    "not_selected_reason",
    "rejection_reason",
    "exclusion_reason",
    "admission_reason",
    "selection_reason",
    "gate_reason",
    "quality_gate_reason",
    "stage_gate_reason",
    "leakage_gate_reason",
    "compile_failure_reason",
    "role_rescue_reason",
    "phase3_rescue_reason",
)


@dataclass(frozen=True)
class GateIssue:
    severity: str
    code: str
    message: str
    path: str | None = None
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PayloadGateEvaluation:
    schema: str
    ok: bool
    issues: tuple[GateIssue, ...]
    summary: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "ok": bool(self.ok),
            "issues": [asdict(issue) for issue in self.issues],
            "summary": _jsonable(self.summary),
        }


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, GateIssue):
        return asdict(value)
    if isinstance(value, PayloadGateEvaluation):
        return value.to_json()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")


def _issue(
    issues: list[GateIssue],
    code: str,
    message: str,
    *,
    severity: str = "error",
    path: str | None = None,
    context: Mapping[str, Any] | None = None,
) -> None:
    issues.append(GateIssue(severity=str(severity), code=str(code), message=str(message), path=path, context=dict(context or {})))


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _adapt_section(payload: Mapping[str, Any] | None) -> Mapping[str, Any]:
    root = _as_mapping(payload)
    adapt = root.get("adapt_vqe")
    return adapt if isinstance(adapt, Mapping) else root


def _continuation(payload: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return _as_mapping(_adapt_section(payload).get("continuation"))


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _dedupe_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        row_dict = dict(row)
        key = (
            str(row_dict.get("candidate_label", row_dict.get("operator_label", row_dict.get("label", "")))) ,
            str(row_dict.get("generator_id", "")),
            str(row_dict.get("position_id", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row_dict)
    return out


def _phase3_rows(payload: Mapping[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cont = _continuation(payload)
    scored = _dedupe_rows(
        [
            *_rows(cont.get("phase2_scored_rows")),
            *_rows(cont.get("phase2_shortlist_rows")),
        ]
    )
    retained = _dedupe_rows(_rows(cont.get("phase2_retained_shortlist_rows")))
    admitted = _dedupe_rows(_rows(cont.get("phase2_admitted_rows")))
    return scored, retained, admitted


def _row_label(row: Mapping[str, Any]) -> str | None:
    for key in ("candidate_label", "operator_label", "label", "selected_label"):
        value = row.get(key)
        if value not in {None, ""}:
            return str(value)
    meta = _row_metadata(row)
    value = meta.get("candidate_label") or meta.get("operator_label")
    return None if value in {None, ""} else str(value)


def _row_generator_id(row: Mapping[str, Any]) -> str | None:
    for key in ("generator_id", "parent_generator_id"):
        value = row.get(key)
        if value not in {None, ""}:
            return str(value)
    meta = _row_metadata(row)
    for key in ("generator_id", "parent_generator_id"):
        value = meta.get(key)
        if value not in {None, ""}:
            return str(value)
    return None


def _row_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for key in ("generator_metadata", "metadata", "operator_metadata", "motif_metadata"):
        value = row.get(key)
        if isinstance(value, Mapping):
            meta.update(dict(value))
    feature = row.get("feature")
    if isinstance(feature, Mapping):
        for key in ("generator_metadata", "metadata", "operator_metadata", "motif_metadata"):
            value = feature.get(key)
            if isinstance(value, Mapping):
                meta.update(dict(value))
        for key in (
            "candidate_family",
            "generator_id",
            "template_id",
            "is_macro_generator",
            "parent_generator_id",
            "candidate_label",
        ):
            if key in feature and key not in meta:
                meta[key] = feature[key]
    for key in (
        "family_id",
        "candidate_family",
        "generator_family",
        "template_id",
        "generator_id",
        "role",
        "operator_role",
        "family_role",
        "expressive_role",
        "generator_role",
        "selected_logical_role",
        "operator_class",
        "candidate_class",
        "is_macro_generator",
    ):
        if key in row and key not in meta:
            meta[key] = row[key]
    return meta


def _flatten_metadata_text(meta: Mapping[str, Any]) -> str:
    pieces: list[str] = []
    for key in (
        "role",
        "operator_role",
        "family_role",
        "expressive_role",
        "generator_role",
        "selected_logical_role",
        "operator_class",
        "candidate_class",
        "family_id",
        "candidate_family",
        "generator_family",
        "template_id",
        "generator_id",
        "parent_generator_id",
        "equivalent_role",
    ):
        value = meta.get(key)
        if isinstance(value, (str, int, float, bool)):
            pieces.append(str(value))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            pieces.extend(str(x) for x in value if isinstance(x, (str, int, float, bool)))
    return " ".join(pieces).replace("-", "_").lower()


def classify_expressive_role(row: Mapping[str, Any]) -> dict[str, Any]:
    """Classify expressive Hubbard/UCCSD-like doubles using metadata first.

    Label text is consulted only as a fallback for double/UCCSD-like labels.
    """

    meta = _row_metadata(row)
    meta_text = _flatten_metadata_text(meta)
    matched = [token for token in _METADATA_EXPRESSIVE_TOKENS if token in meta_text]
    if matched:
        return {
            "is_expressive": True,
            "source": "metadata",
            "matched_tokens": matched,
            "label": _row_label(row),
            "generator_id": _row_generator_id(row),
        }
    label = str(_row_label(row) or "").replace("-", "_").lower()
    label_matches = [token for token in _LABEL_DOUBLE_FALLBACK_TOKENS if token in label]
    if label_matches:
        return {
            "is_expressive": True,
            "source": "label_double_fallback",
            "matched_tokens": label_matches,
            "label": _row_label(row),
            "generator_id": _row_generator_id(row),
        }
    return {"is_expressive": False, "source": "none", "matched_tokens": (), "label": _row_label(row), "generator_id": _row_generator_id(row)}


def _has_canonical_formula(value: Any) -> bool:
    text = str(value or "")
    if text == PHASE3_CANONICAL_SCORE_FORMULA:
        return True
    lowered = text.lower()
    return "confidence" not in lowered and all(token.lower() in lowered for token in ("DeltaE_TR", "N3", "K3"))


def _config_sources(payload: Mapping[str, Any] | None) -> list[tuple[str, Mapping[str, Any]]]:
    cont = _continuation(payload)
    active = _as_mapping(cont.get("active_phase3_surface_summary"))
    sources: list[tuple[str, Mapping[str, Any]]] = []
    phase2 = _as_mapping(cont.get("phase2"))
    if phase2:
        sources.append(("adapt_vqe.continuation.phase2", phase2))
    score_config = _as_mapping(active.get("score_config"))
    if score_config:
        sources.append(("adapt_vqe.continuation.active_phase3_surface_summary.score_config", score_config))
    if active:
        sources.append(("adapt_vqe.continuation.active_phase3_surface_summary", active))
    return sources


def _row_component_mapping(row: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("phase_score_components", "score_components", "phase3_score_components", "components"):
        value = row.get(key)
        if isinstance(value, Mapping):
            return value
    feature = row.get("feature")
    if isinstance(feature, Mapping):
        for key in ("phase_score_components", "score_components", "phase3_score_components", "components"):
            value = feature.get(key)
            if isinstance(value, Mapping):
                return value
    return {}


def _row_has_phase3_components(row: Mapping[str, Any]) -> bool:
    component_map = _row_component_mapping(row)
    if any(key in component_map for key in _PHASE3_COMPONENT_KEYS):
        return True
    return any(key in row for key in _PHASE3_COMPONENT_KEYS)


def _labels_and_ids(rows: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    labels = {str(label) for row in rows if (label := _row_label(row)) not in {None, ""}}
    gids = {str(gid) for row in rows if (gid := _row_generator_id(row)) not in {None, ""}}
    return labels, gids


def _selected_sets(payload: Mapping[str, Any] | None) -> tuple[set[str], set[str]]:
    adapt = _adapt_section(payload)
    cont = _continuation(payload)
    active = _as_mapping(cont.get("active_phase3_surface_summary"))
    labels: set[str] = set(str(x) for x in active.get("selected_operator_labels", []) if x not in {None, ""})
    gids: set[str] = set(str(x) for x in active.get("selected_generator_ids", []) if x not in {None, ""})
    fp = _as_mapping(adapt.get("scaffold_fingerprint_lite"))
    labels.update(str(x) for x in fp.get("selected_operator_labels", []) if x not in {None, ""})
    gids.update(str(x) for x in fp.get("selected_generator_ids", []) if x not in {None, ""})
    for key in ("selected_operator_labels", "operator_labels"):
        labels.update(str(x) for x in adapt.get(key, []) if x not in {None, ""})
    for key in ("selected_generator_ids", "generator_ids"):
        gids.update(str(x) for x in adapt.get(key, []) if x not in {None, ""})
    for meta in _rows(cont.get("selected_generator_metadata")):
        label = _row_label(meta)
        gid = _row_generator_id(meta)
        if label:
            labels.add(str(label))
        if gid:
            gids.add(str(gid))
    return labels, gids


def _row_bool(row: Mapping[str, Any], keys: Sequence[str]) -> bool:
    return any(bool(row.get(key, False)) for key in keys)


def _has_non_selection_evidence(row: Mapping[str, Any]) -> bool:
    for key in _NON_SELECTION_EVIDENCE_KEYS:
        value = row.get(key)
        if value not in {None, "", False}:
            return True
    if row.get("stage_gate_open") is False or row.get("leakage_gate_open") is False or row.get("compile_gate_open") is False:
        return True
    if str(row.get("admission_status", "")).strip().lower() in {"rejected", "not_selected", "not_admitted", "filtered"}:
        return True
    if str(row.get("selection_status", "")).strip().lower() in {"not_selected", "rejected", "filtered"}:
        return True
    return False


def _prune_decision_rows(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    adapt = _adapt_section(payload)
    out: list[dict[str, Any]] = []
    prune = _as_mapping(adapt.get("prune_summary"))
    out.extend(_rows(prune.get("decisions")))
    selected_label = prune.get("selected_label")
    if selected_label not in {None, ""} and bool(prune.get("executed", False)):
        out.append({"label": selected_label, "accepted": bool(prune.get("accepted_count", 0)), "source": "prune_summary.selected_label"})
    for hist in _rows(adapt.get("history")):
        hist_prune = _as_mapping(hist.get("post_admission_prune"))
        out.extend(_rows(hist_prune.get("decisions")))
        label = hist_prune.get("selected_label")
        if label not in {None, ""} and bool(hist_prune.get("executed", False)):
            out.append({"label": label, "accepted": bool(hist_prune.get("accepted_count", 0)), "source": "history.post_admission_prune.selected_label"})
    return out


def _is_pruned(row: Mapping[str, Any], prune_rows: Sequence[Mapping[str, Any]]) -> bool:
    if _row_bool(row, ("pruned", "phase1_pruned", "post_prune_removed")):
        return True
    label = _row_label(row)
    gid = _row_generator_id(row)
    for decision in prune_rows:
        accepted = bool(decision.get("accepted", decision.get("accepted_prune", False)))
        if not accepted:
            continue
        d_label = decision.get("label", decision.get("candidate_label", decision.get("operator_label")))
        d_gid = decision.get("generator_id")
        if label not in {None, ""} and d_label not in {None, ""} and str(label) == str(d_label):
            return True
        if gid not in {None, ""} and d_gid not in {None, ""} and str(gid) == str(d_gid):
            return True
    return False


def _energy_abs_delta(payload: Mapping[str, Any] | None) -> float | None:
    adapt = _adapt_section(payload)
    for key in ("abs_delta_e", "abs_delta_e_reference", "cutoff_abs_delta_e"):
        if key in adapt:
            try:
                value = float(adapt[key])
                return value if math.isfinite(value) else None
            except Exception:
                pass
    root = _as_mapping(payload)
    for key in ("abs_delta_e", "abs_delta_e_reference", "cutoff_abs_delta_e"):
        if key in root:
            try:
                value = float(root[key])
                return value if math.isfinite(value) else None
            except Exception:
                pass
    return None


def _payload_stop_reason(payload: Mapping[str, Any] | None) -> str | None:
    adapt = _adapt_section(payload)
    for source in (adapt, _as_mapping(payload)):
        for key in ("stop_reason", "quality_gate_reason", "failure_reason"):
            value = source.get(key)
            if value not in {None, ""}:
                return str(value)
    return None


def _failure_classification_from_values(
    *,
    stop_reason: str | None,
    abs_delta_e: float | None,
    target_abs_delta_e: float | None,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    target = None if target_abs_delta_e is None else float(target_abs_delta_e)
    ratio = None
    if target is not None and target > 0.0 and abs_delta_e is not None:
        ratio = float(abs_delta_e) / float(target)
    stop = str(stop_reason or "")
    failure = str(failure_reason or "")
    classification = "none"
    severity = "info"
    reason = None
    if target is not None and target > 0.0:
        if abs_delta_e is None:
            classification = "missing_abs_delta_e"
            severity = "error"
            reason = "target_requested_but_metric_missing"
        elif "eps_grad" in stop.lower() and float(abs_delta_e) > max(100.0 * float(target), 1e-3):
            classification = "premature_eps_grad_high_error"
            severity = "error"
            reason = "eps_grad_stop_with_abs_delta_e_far_above_target"
        elif float(abs_delta_e) > float(target):
            classification = "target_abs_delta_e_not_met"
            severity = "error"
            reason = "abs_delta_e_above_target"
    elif "eps_grad" in stop.lower() and abs_delta_e is not None and float(abs_delta_e) > 1e-3:
        classification = "premature_eps_grad_high_error"
        severity = "warning"
        reason = "eps_grad_stop_with_large_abs_delta_e_without_target"
    if classification == "none" and failure:
        classification = "runner_failure"
        severity = "error"
        reason = failure
    return {
        "schema": "phase3_failure_classification_v1",
        "classification": classification,
        "severity": severity,
        "reason": reason,
        "stop_reason": None if stop_reason in {None, ""} else str(stop_reason),
        "failure_reason": None if failure_reason in {None, ""} else str(failure_reason),
        "abs_delta_e": None if abs_delta_e is None else float(abs_delta_e),
        "target_abs_delta_e": target,
        "abs_delta_e_over_target": ratio,
    }


def _failure_classification_from_payload(
    payload: Mapping[str, Any] | None,
    *,
    target_abs_delta_e: float | None,
) -> dict[str, Any]:
    return _failure_classification_from_values(
        stop_reason=_payload_stop_reason(payload),
        abs_delta_e=_energy_abs_delta(payload),
        target_abs_delta_e=target_abs_delta_e,
    )


def _failure_classification_from_benchmark(
    benchmark: Mapping[str, Any] | None,
    *,
    target_abs_delta_e: float | None,
) -> dict[str, Any]:
    bench = _as_mapping(benchmark)
    abs_delta_e = None
    for key in ("abs_delta_e", "abs_delta_e_reference", "cutoff_abs_delta_e"):
        if key in bench:
            try:
                value = float(bench[key])
                if math.isfinite(value):
                    abs_delta_e = value
                    break
            except Exception:
                pass
    return _failure_classification_from_values(
        stop_reason=None if bench.get("stop_reason") in {None, ""} else str(bench.get("stop_reason")),
        failure_reason=None if bench.get("failure_reason") in {None, ""} else str(bench.get("failure_reason")),
        abs_delta_e=abs_delta_e,
        target_abs_delta_e=target_abs_delta_e,
    )


def evaluate_phase3_robustness_payload(
    payload: Mapping[str, Any] | None,
    *,
    shot_objective_relevant: bool = False,
    target_abs_delta_e: float | None = None,
) -> PayloadGateEvaluation:
    """Pure static Phase3 robustness evaluator for an ADAPT/result payload."""

    issues: list[GateIssue] = []
    root = _as_mapping(payload)
    adapt = _adapt_section(root)
    cont = _continuation(root)
    scored_rows, retained_rows, admitted_rows = _phase3_rows(root)

    config_sources = _config_sources(root)
    complete_config_sources = [
        path for path, source in config_sources if all(key in source for key in _CANONICAL_CONFIG_KEYS)
    ]
    formula_sources = [
        path for path, source in config_sources if _has_canonical_formula(source.get("canonical_score_formula"))
    ]
    if not config_sources:
        _issue(issues, "missing_phase3_config", "No Phase3 scoring config payload was found.", path="adapt_vqe.continuation")
    elif not complete_config_sources:
        missing_by_source = {
            path: [key for key in _CANONICAL_CONFIG_KEYS if key not in source]
            for path, source in config_sources
        }
        _issue(
            issues,
            "incomplete_canonical_score_config",
            "Phase3 config does not expose all canonical selector keys.",
            context={"missing_by_source": missing_by_source},
        )
    if not formula_sources:
        _issue(
            issues,
            "missing_canonical_score_formula",
            "Canonical Phase3 formula is missing or not recognizable.",
            context={"expected_formula": PHASE3_CANONICAL_SCORE_FORMULA},
        )

    if not scored_rows:
        _issue(issues, "missing_phase3_rows", "No Phase3 scored rows were found.", path="adapt_vqe.continuation.phase2_scored_rows")
    else:
        missing_row_key_count = 0
        missing_component_count = 0
        for idx, row in enumerate(scored_rows):
            missing_keys = [key for key in _PHASE3_ROW_KEYS if key not in row]
            if missing_keys:
                missing_row_key_count += 1
                if missing_row_key_count <= 5:
                    _issue(
                        issues,
                        "phase3_row_missing_fields",
                        "A Phase3 row does not expose canonical primary/tie-break fields.",
                        path=f"adapt_vqe.continuation.phase2_scored_rows[{idx}]",
                        context={"missing_keys": missing_keys, "candidate_label": _row_label(row)},
                    )
            if not _row_has_phase3_components(row):
                missing_component_count += 1
                if missing_component_count <= 5:
                    _issue(
                        issues,
                        "phase3_row_missing_components",
                        "A Phase3 row does not expose canonical score components.",
                        path=f"adapt_vqe.continuation.phase2_scored_rows[{idx}]",
                        context={"candidate_label": _row_label(row)},
                    )

    selected_labels, selected_gids = _selected_sets(root)
    retained_labels, retained_gids = _labels_and_ids(retained_rows)
    admitted_labels, admitted_gids = _labels_and_ids(admitted_rows)
    expressive_rows: list[dict[str, Any]] = []
    expressive_summaries: list[dict[str, Any]] = []
    prune_rows = _prune_decision_rows(root)
    protected_count = 0
    non_selection_evidence_count = 0
    pruned_count = 0
    pruned_without_equivalent: list[dict[str, Any]] = []

    # Include selected metadata as virtual selected rows only when it has an
    # identifier absent from the scored surface.  Real ADAPT payloads usually
    # carry expressive rows in ``phase2_scored_rows``; blindly appending
    # metadata-only records would double count the same role and can mask a
    # pruned scored row with an unpruned metadata echo.
    candidate_rows = list(scored_rows)
    existing_labels, existing_gids = _labels_and_ids(candidate_rows)
    for meta in _rows(cont.get("selected_generator_metadata")):
        meta_row = {**dict(meta), "selected": True, "_source": "selected_generator_metadata"}
        label = _row_label(meta_row)
        gid = _row_generator_id(meta_row)
        if label in {None, ""} and gid in {None, ""}:
            continue
        if (label not in {None, ""} and str(label) in existing_labels) or (
            gid not in {None, ""} and str(gid) in existing_gids
        ):
            continue
        candidate_rows.append(meta_row)

    for row in _dedupe_rows(candidate_rows):
        role = classify_expressive_role(row)
        if not bool(role.get("is_expressive")):
            continue
        label = _row_label(row)
        gid = _row_generator_id(row)
        selected = bool(
            _row_bool(row, ("selected", "phase3_selected", "admitted", "phase3_admitted"))
            or (label is not None and label in selected_labels)
            or (gid is not None and gid in selected_gids)
        )
        admitted = bool(
            _row_bool(row, ("admitted", "phase3_admitted", "selected_for_admission"))
            or (label is not None and label in admitted_labels)
            or (gid is not None and gid in admitted_gids)
        )
        retained = bool(
            _row_bool(row, ("retained", "phase3_retained", "phase2_retained", "phase3_shortlisted"))
            or (label is not None and label in retained_labels)
            or (gid is not None and gid in retained_gids)
        )
        non_selection_evidence = _has_non_selection_evidence(row)
        pruned = _is_pruned(row, prune_rows)
        protected = bool((selected or admitted or retained) and not pruned)
        if protected:
            protected_count += 1
        if non_selection_evidence:
            non_selection_evidence_count += 1
        if pruned:
            pruned_count += 1
        summary = {
            **role,
            "selected": bool(selected),
            "admitted": bool(admitted),
            "retained": bool(retained),
            "protected": bool(protected),
            "non_selection_evidence": bool(non_selection_evidence),
            "pruned": bool(pruned),
        }
        expressive_rows.append(dict(row))
        expressive_summaries.append(_jsonable(summary))

    if not expressive_rows:
        _issue(
            issues,
            "missing_expressive_role",
            "No expressive Hubbard/UCCSD-like double or equivalent expressive role was found in Phase3 rows or selected metadata.",
        )
    elif protected_count <= 0:
        if non_selection_evidence_count < len(expressive_rows):
            _issue(
                issues,
                "expressive_role_not_retained",
                "Expressive role exists but is not selected/admitted/retained and lacks clear non-selection evidence.",
                context={"expressive_roles": expressive_summaries[:8]},
            )

    if pruned_count > 0 and protected_count <= 0:
        pruned_without_equivalent = [row for row in expressive_summaries if row.get("pruned")]
        _issue(
            issues,
            "expressive_role_pruned_without_equivalent",
            "An expressive role was pruned and no equivalent expressive role remains selected/admitted/retained.",
            context={"pruned_expressive_roles": pruned_without_equivalent[:8]},
        )

    measurement_validation: Mapping[str, Any]
    try:
        proxy = controller_proxy_from_adapt_payload(root)
        measurement_validation = validate_controller_proxy_for_shot_objective(proxy)
    except Exception as exc:  # pragma: no cover - defensive for malformed synthetic payloads
        measurement_validation = {
            "schema": "controller_measurement_proxy_validation_v1",
            "valid": False,
            "reason": f"validation_exception:{type(exc).__name__}:{exc}",
        }
    if bool(shot_objective_relevant) and not bool(measurement_validation.get("valid", False)):
        _issue(
            issues,
            "invalid_measurement_proxy_for_shot_objective",
            "Shot objective is relevant but the controller measurement proxy is not validated native work.",
            path="adapt_vqe.controller_measurement_work_summary",
            context={"measurement_proxy_validation": dict(measurement_validation)},
        )

    abs_delta_e = _energy_abs_delta(root)
    if target_abs_delta_e is not None and float(target_abs_delta_e) > 0.0:
        if abs_delta_e is None:
            _issue(
                issues,
                "missing_abs_delta_e",
                "Target abs_delta_e was requested but no abs_delta_e metric was found.",
                context={"target_abs_delta_e": float(target_abs_delta_e)},
            )
        elif float(abs_delta_e) > float(target_abs_delta_e):
            _issue(
                issues,
                "target_abs_delta_e_not_met",
                "Static Hubbard L2 lane did not meet the target abs_delta_e threshold.",
                context={"abs_delta_e": float(abs_delta_e), "target_abs_delta_e": float(target_abs_delta_e)},
            )

    failure_classification = _failure_classification_from_payload(
        root,
        target_abs_delta_e=target_abs_delta_e,
    )
    if failure_classification.get("classification") == "premature_eps_grad_high_error":
        _issue(
            issues,
            "premature_eps_grad_high_error",
            "Lane stopped on eps_grad while abs_delta_e remained far above the target.",
            context=failure_classification,
        )

    error_issues = [issue for issue in issues if issue.severity == "error"]
    summary = {
        "canonical_score": {
            "expected_formula": PHASE3_CANONICAL_SCORE_FORMULA,
            "config_source_count": int(len(config_sources)),
            "complete_config_sources": complete_config_sources,
            "formula_sources": formula_sources,
        },
        "phase3_rows": {
            "scored_count": int(len(scored_rows)),
            "retained_count": int(len(retained_rows)),
            "admitted_count": int(len(admitted_rows)),
            "rows_with_required_fields": int(
                sum(all(key in row for key in _PHASE3_ROW_KEYS) for row in scored_rows)
            ),
            "rows_with_components": int(sum(_row_has_phase3_components(row) for row in scored_rows)),
        },
        "expressive_role": {
            "expressive_count": int(len(expressive_rows)),
            "protected_count": int(protected_count),
            "non_selection_evidence_count": int(non_selection_evidence_count),
            "pruned_count": int(pruned_count),
            "roles": expressive_summaries[:16],
        },
        "measurement_proxy": {
            "shot_objective_relevant": bool(shot_objective_relevant),
            "validation": dict(measurement_validation),
        },
        "energy": {
            "abs_delta_e": abs_delta_e,
            "target_abs_delta_e": (None if target_abs_delta_e is None else float(target_abs_delta_e)),
        },
        "failure_classification": failure_classification,
    }
    return PayloadGateEvaluation(
        schema=PHASE3_PAYLOAD_EVALUATION_SCHEMA,
        ok=not error_issues,
        issues=tuple(issues),
        summary=summary,
    )


def _default_hubbard_l2_spec(p3opt: Any, lane: str) -> Any:
    return p3opt.HamiltonianBenchmarkSpec(
        benchmark_id=f"hubbard_L2_robustness_{str(lane).lower()}",
        family="hubbard",
        features=p3opt.ProblemFeatureVector(
            problem="hubbard",
            size_label="L2_robustness_gate",
            L=2,
            n_qubits=4,
            pool_size_hint=64,
            spinful=True,
            bosonic=False,
        ),
        base_pipeline_args=(
            "--problem",
            "hubbard",
            "--L",
            "2",
            "--t",
            "1.0",
            "--u",
            "4.0",
            "--dv",
            "0.0",
            "--ordering",
            "blocked",
            "--boundary",
            "periodic",
        ),
        baseline_abs_delta_e=1e-4,
        baseline_count_2q=1000,
        baseline_depth_2q=3000,
        baseline_parameter_count=64,
        split="preflight",
        tags=("static_phase3", "hubbard_l2_robustness_gate", str(lane).upper()),
    )


def run_single_hubbard_l2_lane(
    *,
    lane: str,
    output_dir: Path,
    target_abs_delta_e: float = 1e-5,
    python_bin: str = sys.executable,
    adapt_timeout_s: float | None = None,
    compile_timeout_s: float | None = None,
) -> dict[str, Any]:
    """Run one tiny static Hubbard L2 lane in the current process."""

    import pipelines.static_adapt.optimization.phase3_policy_optuna as p3opt

    lane_key = str(lane).strip().upper()
    active = str(getattr(p3opt, "_ACTIVE_INNER_OPTIMIZER", "")).strip().upper()
    if lane_key != active:
        raise RuntimeError(
            f"run_single_hubbard_l2_lane({lane_key}) requires PHASE3_POLICY_INNER_OPTIMIZER={lane_key}; "
            f"current process is {active}."
        )
    lane_dir = Path(output_dir) / "robustness_gate" / lane_key.lower()
    spec = _default_hubbard_l2_spec(p3opt, lane_key)
    policy = p3opt.AlgorithmPolicy.default()
    preflight_policy: dict[str, Any] | None = None
    if lane_key == "SPSA":
        spec, policy, preflight_policy = _apply_calibrated_spsa_preflight_policy(p3opt, spec)
    weights = p3opt.StaticObjectiveWeights(target_abs_delta_e=float(target_abs_delta_e))
    benchmark = p3opt.run_static_benchmark(
        spec,
        policy,
        output_dir=lane_dir,
        python_bin=python_bin,
        adapt_timeout_s=adapt_timeout_s,
        compile_timeout_s=compile_timeout_s,
        benchmark_target_abs_delta_e=float(target_abs_delta_e),
        objective_weights=weights,
    )
    payload: Mapping[str, Any] | None = None
    if benchmark.result_json not in {None, ""}:
        result_path = Path(str(benchmark.result_json))
        if result_path.exists():
            raw = json.loads(result_path.read_text(encoding="utf-8"))
            payload = raw if isinstance(raw, Mapping) else None
    evaluation = evaluate_phase3_robustness_payload(
        payload,
        shot_objective_relevant=False,
        target_abs_delta_e=float(target_abs_delta_e),
    )
    ok = bool(benchmark.success) and bool(evaluation.ok)
    return {
        "schema": "phase3_hubbard_l2_robustness_lane_v1",
        "lane": lane_key,
        "ok": bool(ok),
        "benchmark": _jsonable(asdict(benchmark)),
        "evaluation": evaluation.to_json(),
        "output_dir": str(lane_dir),
        **({"preflight_policy": preflight_policy} if preflight_policy is not None else {}),
    }


def _run_lane_subprocess(
    *,
    lane: str,
    output_dir: Path,
    target_abs_delta_e: float,
    python_bin: str,
    adapt_timeout_s: float | None,
    compile_timeout_s: float | None,
) -> dict[str, Any]:
    lane_key = str(lane).strip().upper()
    lane_json = Path(output_dir) / "robustness_gate" / lane_key.lower() / "lane_gate.json"
    command = [
        str(python_bin),
        "-m",
        "pipelines.static_adapt.optimization.phase3_robustness_gate",
        "--output-dir",
        str(output_dir),
        "--lane-only",
        lane_key,
        "--target-abs-delta-e",
        str(float(target_abs_delta_e)),
        "--lane-json",
        str(lane_json),
    ]
    if adapt_timeout_s is not None and float(adapt_timeout_s) > 0.0:
        command.extend(["--adapt-timeout-sec", str(float(adapt_timeout_s))])
    if compile_timeout_s is not None and float(compile_timeout_s) > 0.0:
        command.extend(["--compile-timeout-sec", str(float(compile_timeout_s))])
    env = dict(os.environ)
    env["PHASE3_POLICY_INNER_OPTIMIZER"] = lane_key
    started = _now_utc()
    proc = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, capture_output=True)
    payload: dict[str, Any]
    if lane_json.exists():
        try:
            raw = json.loads(lane_json.read_text(encoding="utf-8"))
            payload = dict(raw) if isinstance(raw, Mapping) else {}
        except Exception as exc:
            payload = {"schema": "phase3_hubbard_l2_robustness_lane_v1", "lane": lane_key, "ok": False, "failure_reason": f"lane_json_parse_failed:{exc}"}
    else:
        payload = {"schema": "phase3_hubbard_l2_robustness_lane_v1", "lane": lane_key, "ok": False, "failure_reason": "missing_lane_json"}
    payload.setdefault("subprocess", {})
    payload["subprocess"] = {
        "command": command,
        "returncode": int(proc.returncode),
        "started_utc": started,
        "finished_utc": _now_utc(),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    if int(proc.returncode) != 0:
        payload["ok"] = False
        payload.setdefault("failure_reason", f"lane_subprocess_returncode:{proc.returncode}")
    return _jsonable(payload)


def run_hubbard_l2_robustness_gate(
    *,
    output_dir: Path,
    lanes: Sequence[str] = ("SPSA", "POWELL"),
    target_abs_delta_e: float = 1e-5,
    python_bin: str = sys.executable,
    adapt_timeout_s: float | None = None,
    compile_timeout_s: float | None = None,
    lane_runner: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the static Hubbard L2 robustness preflight and write JSON output."""

    import pipelines.static_adapt.optimization.phase3_policy_optuna as p3opt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_lanes = tuple(dict.fromkeys(str(lane).strip().upper() for lane in lanes if str(lane).strip()))
    if not normalized_lanes:
        normalized_lanes = ("SPSA", "POWELL")
    lane_payloads: dict[str, Any] = {}
    for lane in normalized_lanes:
        if lane_runner is not None:
            try:
                payload = lane_runner(
                    lane=lane,
                    output_dir=output_dir,
                    target_abs_delta_e=float(target_abs_delta_e),
                    python_bin=python_bin,
                    adapt_timeout_s=adapt_timeout_s,
                    compile_timeout_s=compile_timeout_s,
                )
                lane_payloads[lane] = _jsonable(dict(payload))
            except Exception as exc:
                lane_payloads[lane] = {
                    "schema": "phase3_hubbard_l2_robustness_lane_v1",
                    "lane": lane,
                    "ok": False,
                    "failure_reason": f"lane_runner_exception:{type(exc).__name__}:{exc}",
                }
            continue
        active = str(getattr(p3opt, "_ACTIVE_INNER_OPTIMIZER", "")).strip().upper()
        if lane == active:
            try:
                lane_payloads[lane] = run_single_hubbard_l2_lane(
                    lane=lane,
                    output_dir=output_dir,
                    target_abs_delta_e=float(target_abs_delta_e),
                    python_bin=python_bin,
                    adapt_timeout_s=adapt_timeout_s,
                    compile_timeout_s=compile_timeout_s,
                )
            except Exception as exc:
                lane_payloads[lane] = {
                    "schema": "phase3_hubbard_l2_robustness_lane_v1",
                    "lane": lane,
                    "ok": False,
                    "failure_reason": f"lane_exception:{type(exc).__name__}:{exc}",
                }
        else:
            lane_payloads[lane] = _run_lane_subprocess(
                lane=lane,
                output_dir=output_dir,
                target_abs_delta_e=float(target_abs_delta_e),
                python_bin=python_bin,
                adapt_timeout_s=adapt_timeout_s,
                compile_timeout_s=compile_timeout_s,
            )
    ok = all(bool(payload.get("ok", False)) for payload in lane_payloads.values())
    issue_count = 0
    failure_classification_counts: dict[str, int] = {}
    for payload in lane_payloads.values():
        evaluation = payload.get("evaluation") if isinstance(payload, Mapping) else None
        if isinstance(evaluation, Mapping):
            issue_count += int(len(evaluation.get("issues", []) or []))
            summary = evaluation.get("summary")
            classification = _as_mapping(_as_mapping(summary).get("failure_classification")).get("classification")
            if classification not in {None, ""}:
                key = str(classification)
                failure_classification_counts[key] = int(failure_classification_counts.get(key, 0)) + 1
        elif not bool(payload.get("ok", False)):
            issue_count += 1
            classification = _as_mapping(payload.get("failure_classification")).get("classification")
            key = str(classification or "runner_failure")
            failure_classification_counts[key] = int(failure_classification_counts.get(key, 0)) + 1
    gate = {
        "schema": PHASE3_ROBUSTNESS_GATE_SCHEMA,
        "generated_utc": _now_utc(),
        "ok": bool(ok),
        "status": "passed" if ok else "failed",
        "target_abs_delta_e": float(target_abs_delta_e),
        "lanes_requested": list(normalized_lanes),
        "lane_count": int(len(normalized_lanes)),
        "issue_count": int(issue_count),
        "failure_classification_counts": dict(sorted(failure_classification_counts.items())),
        "lanes": lane_payloads,
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "robustness_gate.json", gate)
    return _jsonable(gate)


def default_hubbard_l2_spsa_calibration_matrix() -> tuple[dict[str, Any], ...]:
    """Return the tiny static Hubbard L2 SPSA calibration/triage matrix.

    The matrix is intentionally report-only.  A passing schedule is surfaced as
    a candidate for human review; it is not promoted into defaults here.
    """

    baseline_schedule = {"a": 0.1, "c": 0.1, "A": 10.0, "alpha": 0.602, "gamma": 0.101}
    lower_c_higher_A_schedule = {"a": 0.05, "c": 0.02, "A": 50.0, "alpha": 0.602, "gamma": 0.101}
    lower_c_higher_A_settings = {
        "maxiter": 8000,
        "final_full_refit": True,
        "eval_repeats": 1,
        "eval_agg": "mean",
        "allow_repeats": False,
    }
    return (
        {
            "name": "baseline_current_defaults",
            "description": "Current static Phase3 policy SPSA defaults.",
            "schedule": baseline_schedule,
            "maxiter": 4000,
            "final_full_refit": True,
            "eval_repeats": 1,
            "eval_agg": "mean",
            "allow_repeats": False,
            "diagnostic_only": False,
        },
        {
            "name": "lower_c_higher_A",
            "description": "Lower perturbation c with larger stability offset A and longer SPSA budget.",
            "schedule": lower_c_higher_A_schedule,
            **lower_c_higher_A_settings,
            "diagnostic_only": False,
        },
        {
            "name": "repeats_median",
            "description": "Same as lower_c_higher_A with repeated noisy objective evaluations aggregated by median.",
            "schedule": lower_c_higher_A_schedule,
            **{**lower_c_higher_A_settings, "eval_repeats": 3, "eval_agg": "median"},
            "diagnostic_only": False,
        },
        {
            "name": "repeats_allowed_candidate",
            "description": "Repeat-enabled SPSA recovery candidate for manual promotion review; report-only, not a default change.",
            "schedule": lower_c_higher_A_schedule,
            **{**lower_c_higher_A_settings, "allow_repeats": True},
            "diagnostic_only": False,
            "promotion_eligible": True,
        },
    )


def _spsa_cli_override_args_from_case(case: Mapping[str, Any]) -> list[str]:
    schedule = _as_mapping(case.get("schedule"))
    repeat_flag = "--adapt-allow-repeats" if bool(case.get("allow_repeats", False)) else "--adapt-no-repeats"
    return [
        "--adapt-inner-optimizer",
        "SPSA",
        "--adapt-spsa-a",
        str(float(schedule["a"])),
        "--adapt-spsa-c",
        str(float(schedule["c"])),
        "--adapt-spsa-A",
        str(float(schedule["A"])),
        "--adapt-spsa-alpha",
        str(float(schedule["alpha"])),
        "--adapt-spsa-gamma",
        str(float(schedule["gamma"])),
        "--adapt-maxiter",
        str(int(case["maxiter"])),
        "--adapt-final-full-refit",
        "true" if bool(case.get("final_full_refit", True)) else "false",
        "--adapt-spsa-eval-repeats",
        str(int(case.get("eval_repeats", 1))),
        "--adapt-spsa-eval-agg",
        str(case.get("eval_agg", "mean")),
        repeat_flag,
    ]


def _normalize_spsa_calibration_case(case: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(case)
    name = str(raw.get("name") or "unnamed").strip() or "unnamed"
    schedule_raw = raw.get("schedule") if isinstance(raw.get("schedule"), Mapping) else {}
    schedule = {
        "a": float(schedule_raw.get("a", raw.get("a", raw.get("spsa_a", 0.1)))),
        "c": float(schedule_raw.get("c", raw.get("c", raw.get("spsa_c", 0.1)))),
        "A": float(schedule_raw.get("A", raw.get("A", raw.get("spsa_A", 10.0)))),
        "alpha": float(schedule_raw.get("alpha", raw.get("alpha", raw.get("spsa_alpha", 0.602)))),
        "gamma": float(schedule_raw.get("gamma", raw.get("gamma", raw.get("spsa_gamma", 0.101)))),
    }
    eval_agg = str(raw.get("eval_agg", "mean")).strip().lower()
    if eval_agg not in {"mean", "median"}:
        raise ValueError(f"SPSA calibration case {name!r} has invalid eval_agg={eval_agg!r}")
    diagnostic_only = bool(raw.get("diagnostic_only", False))
    normalized = {
        "name": name,
        "description": str(raw.get("description") or ""),
        "schedule": schedule,
        "maxiter": int(raw.get("maxiter", 4000)),
        "final_full_refit": bool(raw.get("final_full_refit", True)),
        "eval_repeats": max(1, int(raw.get("eval_repeats", 1))),
        "eval_agg": eval_agg,
        "allow_repeats": bool(raw.get("allow_repeats", False)),
        "diagnostic_only": diagnostic_only,
    }
    normalized["promotion_eligible"] = bool(raw.get("promotion_eligible", not diagnostic_only)) and not diagnostic_only
    normalized["cli_overrides"] = {
        "adapt_inner_optimizer": "SPSA",
        "adapt_spsa_a": schedule["a"],
        "adapt_spsa_c": schedule["c"],
        "adapt_spsa_A": schedule["A"],
        "adapt_spsa_alpha": schedule["alpha"],
        "adapt_spsa_gamma": schedule["gamma"],
        "adapt_maxiter": int(normalized["maxiter"]),
        "adapt_final_full_refit": bool(normalized["final_full_refit"]),
        "adapt_spsa_eval_repeats": int(normalized["eval_repeats"]),
        "adapt_spsa_eval_agg": str(normalized["eval_agg"]),
        "adapt_allow_repeats": bool(normalized["allow_repeats"]),
    }
    normalized["cli_flag_overrides"] = {
        "--adapt-inner-optimizer": "SPSA",
        "--adapt-spsa-a": schedule["a"],
        "--adapt-spsa-c": schedule["c"],
        "--adapt-spsa-A": schedule["A"],
        "--adapt-spsa-alpha": schedule["alpha"],
        "--adapt-spsa-gamma": schedule["gamma"],
        "--adapt-maxiter": int(normalized["maxiter"]),
        "--adapt-final-full-refit": "true" if normalized["final_full_refit"] else "false",
        "--adapt-spsa-eval-repeats": int(normalized["eval_repeats"]),
        "--adapt-spsa-eval-agg": str(normalized["eval_agg"]),
        "--adapt-allow-repeats": bool(normalized["allow_repeats"]),
    }
    normalized["cli_override_args"] = _spsa_cli_override_args_from_case(normalized)
    normalized["cli_required_env"] = {"PHASE3_POLICY_INNER_OPTIMIZER": "SPSA"}
    return normalized


def spsa_calibration_cli_overrides_for_candidate(candidate: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return manual CLI override fields/args for a calibration candidate summary."""

    if not isinstance(candidate, Mapping):
        return None
    case_source = candidate.get("case") if isinstance(candidate.get("case"), Mapping) else None
    if case_source is None:
        settings = _as_mapping(candidate.get("settings"))
        case_source = {
            "name": candidate.get("case_name", candidate.get("name", "best_passing_candidate")),
            "schedule": candidate.get("schedule"),
            "maxiter": settings.get("maxiter", candidate.get("maxiter", 4000)),
            "final_full_refit": settings.get("final_full_refit", candidate.get("final_full_refit", True)),
            "eval_repeats": settings.get("eval_repeats", candidate.get("eval_repeats", 1)),
            "eval_agg": settings.get("eval_agg", candidate.get("eval_agg", "mean")),
            "allow_repeats": settings.get("allow_repeats", candidate.get("allow_repeats", False)),
            "diagnostic_only": candidate.get("diagnostic_only", False),
            "promotion_eligible": candidate.get("promotion_eligible", True),
        }
    normalized = _normalize_spsa_calibration_case(case_source)
    return {
        "case_name": normalized["name"],
        "cli_overrides": dict(normalized["cli_overrides"]),
        "cli_flag_overrides": dict(normalized["cli_flag_overrides"]),
        "cli_override_args": list(normalized["cli_override_args"]),
        "cli_required_env": dict(normalized["cli_required_env"]),
    }


def extract_best_passing_spsa_calibration_cli_overrides(
    calibration_report: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Extract manual CLI override fields for ``best_passing_candidate`` from a report."""

    best = _as_mapping(_as_mapping(calibration_report).get("best"))
    candidate = best.get("best_passing_candidate")
    return spsa_calibration_cli_overrides_for_candidate(candidate if isinstance(candidate, Mapping) else None)


def _set_cli_option(args: Sequence[str], flag: str, value: str | int | float) -> tuple[str, ...]:
    out: list[str] = []
    idx = 0
    tokens = [str(x) for x in args]
    while idx < len(tokens):
        tok = tokens[idx]
        if tok == flag:
            idx += 1
            if idx < len(tokens) and not tokens[idx].startswith("--"):
                idx += 1
            continue
        if tok.startswith(flag + "="):
            idx += 1
            continue
        out.append(tok)
        idx += 1
    out.extend([str(flag), str(value)])
    return tuple(out)


def _calibration_spec_for_case(p3opt: Any, case: Mapping[str, Any]) -> Any:
    spec = _default_hubbard_l2_spec(p3opt, "SPSA")
    name = str(case["name"])
    args = tuple(spec.base_pipeline_args)
    args = _set_cli_option(args, "--adapt-spsa-eval-repeats", int(case["eval_repeats"]))
    args = _set_cli_option(args, "--adapt-spsa-eval-agg", str(case["eval_agg"]))
    return replace(
        spec,
        benchmark_id=f"hubbard_L2_spsa_calibration_{name}",
        base_pipeline_args=args,
        split="calibration",
        tags=(*tuple(spec.tags), "spsa_calibration", name),
    )


def _calibration_policy_for_case(p3opt: Any, case: Mapping[str, Any]) -> Any:
    base = p3opt.AlgorithmPolicy.default()
    schedule = _as_mapping(case.get("schedule"))
    static = replace(
        base.static,
        adapt_maxiter=int(case["maxiter"]),
        adapt_final_full_refit=bool(case["final_full_refit"]),
        adapt_allow_repeats=bool(case["allow_repeats"]),
    )
    inner = replace(
        base.inner_optimizer,
        inner_optimizer="SPSA",
        final_optimizer_type="SPSA",
        spsa_a=float(schedule["a"]),
        spsa_c=float(schedule["c"]),
        spsa_A=float(schedule["A"]),
        spsa_alpha=float(schedule["alpha"]),
        spsa_gamma=float(schedule["gamma"]),
        refit_maxiter=int(case["maxiter"]),
        final_maxiter=int(case["maxiter"]),
    )
    return p3opt.AlgorithmPolicy(pool=base.pool, static=static, inner_optimizer=inner)


def _calibrated_spsa_preflight_case() -> dict[str, Any]:
    """Return the calibrated SPSA case promoted for this gate lane only."""

    for case in default_hubbard_l2_spsa_calibration_matrix():
        normalized = _normalize_spsa_calibration_case(case)
        if normalized["name"] == "repeats_allowed_candidate":
            return normalized
    raise RuntimeError("missing repeats_allowed_candidate SPSA calibration case")


def _spsa_preflight_settings(case: Mapping[str, Any]) -> dict[str, Any]:
    schedule = _as_mapping(case.get("schedule"))
    return {
        "adapt_allow_repeats": bool(case.get("allow_repeats", False)),
        "adapt_maxiter": int(case.get("maxiter", 4000)),
        "adapt_final_full_refit": bool(case.get("final_full_refit", True)),
        "spsa_a": float(schedule.get("a", 0.1)),
        "spsa_c": float(schedule.get("c", 0.1)),
        "spsa_A": float(schedule.get("A", 10.0)),
        "spsa_alpha": float(schedule.get("alpha", 0.602)),
        "spsa_gamma": float(schedule.get("gamma", 0.101)),
        "eval_repeats": int(case.get("eval_repeats", 1)),
        "eval_agg": str(case.get("eval_agg", "mean")),
    }


def _apply_calibrated_spsa_preflight_policy(
    p3opt: Any,
    spec: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    """Apply the repeat-enabled SPSA calibration to the robustness gate lane.

    This is intentionally scoped to the Hubbard L2 robustness-gate SPSA
    preflight.  It does not mutate ``phase3_policy_optuna`` defaults or the
    Optuna sample space.
    """

    case = _calibrated_spsa_preflight_case()
    spec_args = tuple(spec.base_pipeline_args)
    spec_args = _set_cli_option(spec_args, "--adapt-spsa-eval-repeats", int(case["eval_repeats"]))
    spec_args = _set_cli_option(spec_args, "--adapt-spsa-eval-agg", str(case["eval_agg"]))
    gated_spec = replace(
        spec,
        base_pipeline_args=spec_args,
        tags=(
            *tuple(spec.tags),
            "calibrated_spsa_preflight_policy",
            str(case["name"]),
        ),
    )
    policy = _calibration_policy_for_case(p3opt, case)
    metadata = {
        "schema": "phase3_hubbard_l2_robustness_preflight_policy_v1",
        "applied": True,
        "scope": "robustness_gate_spsa_lane_only",
        "policy_source": "hubbard_l2_spsa_calibration_matrix",
        "candidate_name": str(case["name"]),
        "description": str(case.get("description", "")),
        "settings": _spsa_preflight_settings(case),
        "calibration_case": _jsonable(case),
        "cli_overrides": dict(case.get("cli_overrides", {})),
        "cli_flag_overrides": dict(case.get("cli_flag_overrides", {})),
        "cli_override_args": list(case.get("cli_override_args", [])),
        "cli_required_env": dict(case.get("cli_required_env", {})),
        "note": "Lane-local calibrated preflight policy; not a global Phase3/Optuna/ADAPT default.",
    }
    return gated_spec, policy, metadata


def run_single_hubbard_l2_spsa_calibration_case(
    *,
    case: Mapping[str, Any],
    output_dir: Path,
    target_abs_delta_e: float = 1e-5,
    python_bin: str = sys.executable,
    adapt_timeout_s: float | None = None,
    compile_timeout_s: float | None = None,
) -> dict[str, Any]:
    """Run one SPSA schedule case for the static Hubbard L2 calibration matrix."""

    import pipelines.static_adapt.optimization.phase3_policy_optuna as p3opt

    active = str(getattr(p3opt, "_ACTIVE_INNER_OPTIMIZER", "")).strip().upper()
    if active != "SPSA":
        raise RuntimeError(
            "run_single_hubbard_l2_spsa_calibration_case requires "
            "PHASE3_POLICY_INNER_OPTIMIZER=SPSA before process start."
        )
    normalized_case = _normalize_spsa_calibration_case(case)
    case_dir = Path(output_dir) / "spsa_calibration" / str(normalized_case["name"])
    spec = _calibration_spec_for_case(p3opt, normalized_case)
    policy = _calibration_policy_for_case(p3opt, normalized_case)
    weights = p3opt.StaticObjectiveWeights(target_abs_delta_e=float(target_abs_delta_e))
    benchmark = p3opt.run_static_benchmark(
        spec,
        policy,
        output_dir=case_dir,
        python_bin=python_bin,
        adapt_timeout_s=adapt_timeout_s,
        compile_timeout_s=compile_timeout_s,
        benchmark_target_abs_delta_e=float(target_abs_delta_e),
        objective_weights=weights,
    )
    benchmark_payload = _jsonable(asdict(benchmark))
    result_payload: Mapping[str, Any] | None = None
    result_json = benchmark_payload.get("result_json")
    if result_json not in {None, ""}:
        result_path = Path(str(result_json))
        if result_path.exists():
            raw = json.loads(result_path.read_text(encoding="utf-8"))
            result_payload = raw if isinstance(raw, Mapping) else None
    evaluation = evaluate_phase3_robustness_payload(
        result_payload,
        shot_objective_relevant=False,
        target_abs_delta_e=float(target_abs_delta_e),
    )
    failure_classification = (
        _failure_classification_from_benchmark(benchmark_payload, target_abs_delta_e=float(target_abs_delta_e))
        if result_payload is None
        else _failure_classification_from_payload(result_payload, target_abs_delta_e=float(target_abs_delta_e))
    )
    abs_delta_e = _as_float_or_none(benchmark_payload.get("abs_delta_e"))
    passed_target = bool(
        benchmark.success
        and evaluation.ok
        and abs_delta_e is not None
        and abs_delta_e <= float(target_abs_delta_e)
    )
    return {
        "schema": PHASE3_HUBBARD_L2_SPSA_CALIBRATION_CASE_SCHEMA,
        "case": normalized_case,
        "ok": bool(benchmark.success and evaluation.ok),
        "completed": True,
        "passed_target": bool(passed_target),
        "promotion_eligible": bool(normalized_case.get("promotion_eligible", True)),
        "abs_delta_e": abs_delta_e,
        "target_abs_delta_e": float(target_abs_delta_e),
        "benchmark": benchmark_payload,
        "evaluation": evaluation.to_json(),
        "failure_classification": failure_classification,
        "output_dir": str(case_dir),
    }


def _case_abs_delta(case_payload: Mapping[str, Any]) -> float | None:
    for source in (
        case_payload,
        _as_mapping(case_payload.get("benchmark")),
        _as_mapping(_as_mapping(case_payload.get("evaluation")).get("summary")).get("energy"),
    ):
        value = source.get("abs_delta_e") if isinstance(source, Mapping) else None
        if value is None:
            continue
        try:
            out = float(value)
            return out if math.isfinite(out) else None
        except Exception:
            continue
    return None


def _calibration_payload_passes_target(case_payload: Mapping[str, Any]) -> bool:
    abs_delta_e = _case_abs_delta(case_payload)
    target_abs_delta_e = _as_float_or_none(case_payload.get("target_abs_delta_e"))
    return bool(
        case_payload.get("ok", False)
        and abs_delta_e is not None
        and target_abs_delta_e is not None
        and abs_delta_e <= target_abs_delta_e
    )


def _normalize_calibration_case_payload(
    raw: Mapping[str, Any] | None,
    *,
    case: Mapping[str, Any],
    target_abs_delta_e: float,
) -> dict[str, Any]:
    payload = dict(raw or {})
    normalized_case = _normalize_spsa_calibration_case(case)
    if payload.get("schema") == PHASE3_HUBBARD_L2_SPSA_CALIBRATION_CASE_SCHEMA:
        payload.setdefault("case", normalized_case)
        payload.setdefault("completed", True)
        payload.setdefault("promotion_eligible", bool(normalized_case.get("promotion_eligible", True)))
        payload.setdefault("target_abs_delta_e", float(target_abs_delta_e))
        payload.setdefault("abs_delta_e", _case_abs_delta(payload))
        payload["passed_target"] = bool(payload.get("passed_target", True) and _calibration_payload_passes_target(payload))
        return _jsonable(payload)

    result_payload = payload.get("result_payload")
    result_payload_mapping = result_payload if isinstance(result_payload, Mapping) else None
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, Mapping) and result_payload_mapping is not None:
        evaluation = evaluate_phase3_robustness_payload(
            result_payload_mapping,
            shot_objective_relevant=False,
            target_abs_delta_e=float(target_abs_delta_e),
        ).to_json()
    benchmark = payload.get("benchmark") if isinstance(payload.get("benchmark"), Mapping) else {}
    failure_classification = payload.get("failure_classification")
    if not isinstance(failure_classification, Mapping):
        failure_classification = (
            _failure_classification_from_payload(
                result_payload_mapping,
                target_abs_delta_e=float(target_abs_delta_e),
            )
            if result_payload_mapping is not None
            else _failure_classification_from_benchmark(
                benchmark,
                target_abs_delta_e=float(target_abs_delta_e),
            )
        )
    abs_delta_e = _case_abs_delta({"abs_delta_e": payload.get("abs_delta_e"), "benchmark": benchmark, "evaluation": evaluation})
    benchmark_success = bool(benchmark.get("success", payload.get("benchmark_success", payload.get("ok", False))))
    evaluation_ok = bool(evaluation.get("ok", True)) if isinstance(evaluation, Mapping) else True
    ok = bool(payload.get("ok", benchmark_success and evaluation_ok))
    passed_target = bool(
        ok
        and abs_delta_e is not None
        and abs_delta_e <= float(target_abs_delta_e)
    )
    return _jsonable(
        {
            "schema": PHASE3_HUBBARD_L2_SPSA_CALIBRATION_CASE_SCHEMA,
            "case": normalized_case,
            "ok": ok,
            "completed": bool(payload.get("completed", True)),
            "passed_target": bool(payload.get("passed_target", True) and passed_target),
            "promotion_eligible": bool(normalized_case.get("promotion_eligible", True)),
            "abs_delta_e": abs_delta_e,
            "target_abs_delta_e": float(target_abs_delta_e),
            "benchmark": _jsonable(benchmark),
            "evaluation": _jsonable(evaluation) if isinstance(evaluation, Mapping) else None,
            "failure_classification": _jsonable(failure_classification),
            "output_dir": payload.get("output_dir"),
        }
    )


def _best_case_summary(cases: Sequence[Mapping[str, Any]], *, passing: bool) -> dict[str, Any] | None:
    candidates: list[tuple[float, str, Mapping[str, Any]]] = []
    for payload in cases:
        if not bool(payload.get("promotion_eligible", True)):
            continue
        is_passing = _calibration_payload_passes_target(payload)
        if bool(passing) != is_passing:
            continue
        abs_delta_e = _case_abs_delta(payload)
        if abs_delta_e is None:
            continue
        candidates.append((float(abs_delta_e), str(_as_mapping(payload.get("case")).get("name", "")), payload))
    if not candidates:
        return None
    _abs_delta, _name, payload = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
    case = _as_mapping(payload.get("case"))
    classification = _as_mapping(payload.get("failure_classification"))
    cli = spsa_calibration_cli_overrides_for_candidate({"case": case}) or {}
    return {
        "case_name": case.get("name"),
        "abs_delta_e": _case_abs_delta(payload),
        "target_abs_delta_e": payload.get("target_abs_delta_e"),
        "passed_target": _calibration_payload_passes_target(payload),
        "failure_classification": classification.get("classification"),
        "schedule": _jsonable(case.get("schedule")),
        "settings": {
            key: case.get(key)
            for key in ("maxiter", "final_full_refit", "eval_repeats", "eval_agg", "allow_repeats")
        },
        "cli_overrides": dict(cli.get("cli_overrides", {})),
        "cli_flag_overrides": dict(cli.get("cli_flag_overrides", {})),
        "cli_override_args": list(cli.get("cli_override_args", [])),
        "cli_required_env": dict(cli.get("cli_required_env", {})),
    }


def run_hubbard_l2_spsa_calibration(
    *,
    output_dir: Path,
    cases: Sequence[Mapping[str, Any]] | None = None,
    target_abs_delta_e: float = 1e-5,
    python_bin: str = sys.executable,
    adapt_timeout_s: float | None = None,
    compile_timeout_s: float | None = None,
    case_runner: Callable[..., Mapping[str, Any]] | None = None,
    calibration_json: Path | None = None,
) -> dict[str, Any]:
    """Run/report the static Hubbard L2 SPSA schedule calibration matrix.

    This is a triage runner, not a gate and not a default-promotion mechanism.
    Tests can pass ``case_runner`` to avoid expensive ADAPT subprocesses.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_cases = tuple(
        _normalize_spsa_calibration_case(case)
        for case in (default_hubbard_l2_spsa_calibration_matrix() if cases is None else cases)
    )
    case_payloads: dict[str, Any] = {}
    for case in normalized_cases:
        name = str(case["name"])
        try:
            if case_runner is None:
                payload = run_single_hubbard_l2_spsa_calibration_case(
                    case=case,
                    output_dir=output_dir,
                    target_abs_delta_e=float(target_abs_delta_e),
                    python_bin=python_bin,
                    adapt_timeout_s=adapt_timeout_s,
                    compile_timeout_s=compile_timeout_s,
                )
            else:
                raw = case_runner(
                    case=case,
                    output_dir=output_dir / "spsa_calibration" / name,
                    target_abs_delta_e=float(target_abs_delta_e),
                    python_bin=python_bin,
                    adapt_timeout_s=adapt_timeout_s,
                    compile_timeout_s=compile_timeout_s,
                )
                payload = _normalize_calibration_case_payload(
                    raw,
                    case=case,
                    target_abs_delta_e=float(target_abs_delta_e),
                )
        except Exception as exc:
            payload = {
                "schema": PHASE3_HUBBARD_L2_SPSA_CALIBRATION_CASE_SCHEMA,
                "case": case,
                "ok": False,
                "completed": False,
                "passed_target": False,
                "promotion_eligible": bool(case.get("promotion_eligible", True)),
                "abs_delta_e": None,
                "target_abs_delta_e": float(target_abs_delta_e),
                "failure_classification": _failure_classification_from_values(
                    stop_reason=None,
                    abs_delta_e=None,
                    target_abs_delta_e=float(target_abs_delta_e),
                    failure_reason=f"case_runner_exception:{type(exc).__name__}:{exc}",
                ),
            }
        case_payloads[name] = _jsonable(payload)

    case_values = tuple(value for value in case_payloads.values() if isinstance(value, Mapping))
    classification_counts: dict[str, int] = {}
    for payload in case_values:
        classification = str(_as_mapping(payload.get("failure_classification")).get("classification", "none"))
        classification_counts[classification] = int(classification_counts.get(classification, 0)) + 1
    best_passing = _best_case_summary(case_values, passing=True)
    best_near_miss = _best_case_summary(case_values, passing=False)
    passed_count = int(sum(_calibration_payload_passes_target(payload) for payload in case_values))
    completed_count = int(sum(bool(payload.get("completed", False)) for payload in case_values))
    report = {
        "schema": PHASE3_HUBBARD_L2_SPSA_CALIBRATION_SCHEMA,
        "generated_utc": _now_utc(),
        "calibration_completed": completed_count == len(case_values),
        "status": "completed" if completed_count == len(case_values) else "incomplete",
        "target_abs_delta_e": float(target_abs_delta_e),
        "case_count": int(len(case_values)),
        "completed_case_count": completed_count,
        "passing_case_count": passed_count,
        "failure_classification_counts": dict(sorted(classification_counts.items())),
        "cases": case_payloads,
        "best": {
            "best_passing_candidate": best_passing,
            "best_near_miss_candidate": best_near_miss,
            "automatic_promotion": False,
            "selected_for_promotion": None,
            "promotion_policy": "report_only_manual_decision_required",
        },
        "output_dir": str(output_dir),
    }
    output_path = Path(calibration_json) if calibration_json is not None else output_dir / "hubbard_l2_spsa_calibration.json"
    _write_json(output_path, report)
    report["calibration_json"] = str(output_path)
    return _jsonable(report)


def should_apply_robustness_gate(
    *,
    mode: str,
    specs: Sequence[Any],
    gate_mode: str = "auto",
    n_trials: int = 1,
) -> bool:
    """Return whether CLI auto/require should run the preflight gate."""

    gate = str(gate_mode or "auto").strip().lower()
    if gate == "off":
        return False
    if gate == "require":
        return True
    if gate != "auto":
        raise ValueError("robustness gate mode must be one of {'auto','require','off'}")
    if str(mode).strip().lower() not in {"global", "oracle-grid"}:
        return False
    # Auto protects broad global/oracle-grid studies while allowing the tiny
    # one-case Hubbard L2 n_trials=1 smoke exception to stay fast.
    selected = tuple(specs)
    if int(n_trials) > 1:
        return True
    if len(selected) > 1:
        return True
    only = selected[0] if len(selected) == 1 else None
    return not bool(
        only is not None
        and str(getattr(only, "family", "")) == "hubbard"
        and int(getattr(getattr(only, "features", None), "L", -1)) == 2
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Static Hubbard L2 Phase3 robustness gate.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lanes", nargs="+", default=["SPSA", "POWELL"])
    parser.add_argument("--target-abs-delta-e", type=float, default=1e-5)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--adapt-timeout-sec", type=float, default=None)
    parser.add_argument("--compile-timeout-sec", type=float, default=None)
    parser.add_argument(
        "--spsa-calibration",
        action="store_true",
        help="Run the report-only static Hubbard L2 SPSA schedule calibration matrix instead of the pass/fail robustness gate.",
    )
    parser.add_argument(
        "--calibration-json",
        type=Path,
        default=None,
        help="Optional output path for --spsa-calibration JSON. Defaults to OUTPUT_DIR/hubbard_l2_spsa_calibration.json.",
    )
    parser.add_argument("--lane-only", choices=["SPSA", "POWELL", "COBYLA"], default=None)
    parser.add_argument("--lane-json", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.spsa_calibration:
        payload = run_hubbard_l2_spsa_calibration(
            output_dir=args.output_dir,
            target_abs_delta_e=float(args.target_abs_delta_e),
            python_bin=str(args.python_bin),
            adapt_timeout_s=args.adapt_timeout_sec,
            compile_timeout_s=args.compile_timeout_sec,
            calibration_json=args.calibration_json,
        )
        return 0 if bool(payload.get("calibration_completed", False)) else 2
    if args.lane_only:
        payload = run_single_hubbard_l2_lane(
            lane=str(args.lane_only),
            output_dir=args.output_dir,
            target_abs_delta_e=float(args.target_abs_delta_e),
            python_bin=str(args.python_bin),
            adapt_timeout_s=args.adapt_timeout_sec,
            compile_timeout_s=args.compile_timeout_sec,
        )
        if args.lane_json is not None:
            _write_json(Path(args.lane_json), payload)
        else:
            _write_json(Path(args.output_dir) / "lane_gate.json", payload)
        return 0 if bool(payload.get("ok", False)) else 2
    gate = run_hubbard_l2_robustness_gate(
        output_dir=args.output_dir,
        lanes=tuple(args.lanes),
        target_abs_delta_e=float(args.target_abs_delta_e),
        python_bin=str(args.python_bin),
        adapt_timeout_s=args.adapt_timeout_sec,
        compile_timeout_s=args.compile_timeout_sec,
    )
    return 0 if bool(gate.get("ok", False)) else 2


__all__ = [
    "PHASE3_CANONICAL_SCORE_FORMULA",
    "PHASE3_HUBBARD_L2_SPSA_CALIBRATION_CASE_SCHEMA",
    "PHASE3_HUBBARD_L2_SPSA_CALIBRATION_SCHEMA",
    "PHASE3_PAYLOAD_EVALUATION_SCHEMA",
    "PHASE3_ROBUSTNESS_GATE_SCHEMA",
    "GateIssue",
    "PayloadGateEvaluation",
    "classify_expressive_role",
    "default_hubbard_l2_spsa_calibration_matrix",
    "evaluate_phase3_robustness_payload",
    "extract_best_passing_spsa_calibration_cli_overrides",
    "run_hubbard_l2_robustness_gate",
    "run_hubbard_l2_spsa_calibration",
    "run_single_hubbard_l2_spsa_calibration_case",
    "run_single_hubbard_l2_lane",
    "spsa_calibration_cli_overrides_for_candidate",
    "should_apply_robustness_gate",
    "build_parser",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
