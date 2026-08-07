"""Candidate metadata assembly for static ADAPT selectors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pipelines.contracts.static_provenance import (
    HH_PHYSICAL_OPERATOR_LANE_OTHER,
    classify_static_physical_operator_lane,
)
from pipelines.scaffold.hh_continuation_types import CandidateFeatures
from pipelines.static_adapt.algebraic_metadata import (
    AlgebraicMetadataError,
    AlgebraicMetadataIndex,
    LANE_MIX,
    build_exact_expansion_index,
)
from pipelines.static_adapt.lane_routes import (
    STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE,
)
from pipelines.static_adapt.nested_windows import (
    COMPILE_PROXY_BASIS_OLD_PRE_INHERITED,
    NestedRefitWindow,
    build_nested_window_accounting,
    map_post_to_pre_old_index,
    serialize_nested_window,
    serialize_nested_window_accounting,
)
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def candidate_feature_with_updates(
    feat: Any,
    updates: Mapping[str, Any],
) -> Any:
    if not isinstance(feat, CandidateFeatures):
        return feat
    return CandidateFeatures(**{**feat.__dict__, **dict(updates)})


def _old_pre_indices_from_post_window(
    *,
    post_indices: Sequence[int],
    position_id: int,
) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for post_index in post_indices:
        old_index = map_post_to_pre_old_index(int(post_index), int(position_id))
        if old_index is None or int(old_index) in seen:
            continue
        seen.add(int(old_index))
        out.append(int(old_index))
    return out


@dataclass
class SelectorCandidateMetadataContext:
    algebraic_metadata_index: AlgebraicMetadataIndex | None
    pool_generator_registry: Mapping[str, Mapping[str, Any]]
    algebraic_lane_policy_active: bool
    adapt_window_size: int
    static_lane_route: str
    problem_key: str
    physical_operator_lane_summary: dict[str, Any]
    physical_operator_lane_classifier_version: str
    phase3_selector_policy: str
    phase3_geometry_window_size: int

    def ensure_algebraic_expansion_for_term(self, term: AnsatzTerm) -> bool:
        index = self.algebraic_metadata_index
        if index is None:
            return False
        label = str(getattr(term, "label", ""))
        try:
            index.resolve_key(label)
            return True
        except AlgebraicMetadataError:
            pass
        try:
            single = build_exact_expansion_index(
                pool=[term],
                registry_by_label=self.pool_generator_registry,
                require_exact=True,
                allow_polynomial_source=True,
            )
        except AlgebraicMetadataError as exc:
            if bool(self.algebraic_lane_policy_active):
                raise RuntimeError(
                    f"Exact algebraic expansion is missing for generator {label!r}."
                ) from exc
            return False
        for key, expansion in single.expansions_by_key.items():
            index.expansions_by_key[str(key)] = expansion
        for key_label, keys in single.label_to_keys.items():
            merged = list(index.label_to_keys.get(str(key_label), ()))
            for key in keys:
                if str(key) not in merged:
                    merged.append(str(key))
            index.label_to_keys[str(key_label)] = tuple(merged)
        return True

    def algebraic_context_terms(
        self,
        *,
        selected_ops_now: Sequence[AnsatzTerm],
        window_terms: Sequence[AnsatzTerm],
    ) -> list[AnsatzTerm]:
        out: list[AnsatzTerm] = []
        seen: set[str] = set()
        for term in list(window_terms):
            label = str(getattr(term, "label", ""))
            if label and label not in seen:
                out.append(term)
                seen.add(label)
        recent_budget = max(1, int(self.adapt_window_size))
        for term in list(selected_ops_now)[-recent_budget:]:
            label = str(getattr(term, "label", ""))
            if label and label not in seen:
                out.append(term)
                seen.add(label)
        return out

    def algebraic_payload_for_candidate(
        self,
        *,
        candidate_term: AnsatzTerm,
        selected_ops_now: Sequence[AnsatzTerm],
        window_terms: Sequence[AnsatzTerm],
    ) -> dict[str, Any]:
        index = self.algebraic_metadata_index
        if index is None:
            return {
                "algebraic_lane": LANE_MIX,
                "algebraic_quality": "inactive",
                "algebraic_context_counts": {
                    "n_flat": 0,
                    "n_curv": 0,
                    "n_disj": 0,
                    "n_approx": 0,
                },
                "algebraic_context_labels": [],
            }
        self.ensure_algebraic_expansion_for_term(candidate_term)
        context_terms = self.algebraic_context_terms(
            selected_ops_now=selected_ops_now,
            window_terms=window_terms,
        )
        context_labels: list[str] = []
        for term in context_terms:
            self.ensure_algebraic_expansion_for_term(term)
            label = str(getattr(term, "label", ""))
            if label:
                context_labels.append(label)
        if not context_labels:
            return {
                "algebraic_lane": LANE_MIX,
                "algebraic_quality": "exact",
                "algebraic_context_counts": {
                    "n_flat": 0,
                    "n_curv": 0,
                    "n_disj": 0,
                    "n_approx": 0,
                },
                "algebraic_context_labels": [],
            }
        summary = index.summarize_local_context(
            str(getattr(candidate_term, "label", "")),
            context_labels,
        )
        return {
            "algebraic_lane": str(summary.lane),
            "algebraic_quality": str(summary.quality),
            "algebraic_context_counts": {
                "n_flat": int(summary.n_flat),
                "n_curv": int(summary.n_curv),
                "n_disj": int(summary.n_disj),
                "n_approx": int(summary.n_approx),
            },
            "algebraic_context_labels": [str(value) for value in summary.context_labels],
        }

    def physical_payload_for_candidate(
        self,
        *,
        feat_obj: CandidateFeatures,
        candidate_term: AnsatzTerm,
    ) -> dict[str, Any]:
        if self.static_lane_route != STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE:
            return {"static_lane_route": str(self.static_lane_route)}

        candidates: list[tuple[str, str]] = []
        seen: set[str] = set()

        def add_label(source: str, value: Any) -> None:
            if value in {None, ""}:
                return
            label = str(value)
            if label == "" or label in seen:
                return
            seen.add(label)
            candidates.append((str(source), label))

        add_label(
            "runtime_split_parent_label",
            getattr(feat_obj, "runtime_split_parent_label", None),
        )
        metadata = getattr(feat_obj, "generator_metadata", None)
        if isinstance(metadata, Mapping):
            for key in (
                "runtime_split_parent_label",
                "parent_label",
                "source_parent_label",
                "parent_generator_label",
                "template_label",
                "base_label",
                "source_label",
                "generator_label",
                "candidate_label",
                "label",
            ):
                add_label(f"generator_metadata.{key}", metadata.get(key))
        add_label("feature.candidate_label", getattr(feat_obj, "candidate_label", None))
        add_label("candidate_term.label", getattr(candidate_term, "label", None))

        chosen: dict[str, Any] | None = None
        chosen_source = "unavailable"
        if not candidates:
            candidates.append(("missing", ""))
        for source, label in candidates:
            payload = classify_static_physical_operator_lane(
                label,
                problem=self.problem_key,
            )
            lane = str(
                payload.get("physical_operator_lane", HH_PHYSICAL_OPERATOR_LANE_OTHER)
            )
            if chosen is None:
                chosen = dict(payload)
                chosen_source = str(source)
            if lane != HH_PHYSICAL_OPERATOR_LANE_OTHER:
                chosen = dict(payload)
                chosen_source = str(source)
                break

        chosen = chosen or classify_static_physical_operator_lane(
            "",
            problem=self.problem_key,
        )
        lane = str(
            chosen.get("physical_operator_lane", HH_PHYSICAL_OPERATOR_LANE_OTHER)
        )
        quality = (
            "classified"
            if lane != HH_PHYSICAL_OPERATOR_LANE_OTHER
            else "unclassified_other"
        )
        summary = self.physical_operator_lane_summary
        summary["classified_count"] = int(summary.get("classified_count", 0)) + 1
        if lane == HH_PHYSICAL_OPERATOR_LANE_OTHER:
            summary["other_count"] = int(summary.get("other_count", 0)) + 1
        lane_counts = summary.setdefault("lane_counts", {})
        if isinstance(lane_counts, dict):
            lane_counts[lane] = int(lane_counts.get(lane, 0)) + 1
        sources = summary.setdefault("sources", {})
        if isinstance(sources, dict):
            sources[chosen_source] = int(sources.get(chosen_source, 0)) + 1
        return {
            "static_lane_route": str(self.static_lane_route),
            "physical_operator_lane": str(lane),
            "physical_operator_quality": str(quality),
            "physical_operator_hh_full_meta_class": chosen.get("hh_full_meta_class"),
            "physical_operator_classifier_version": str(
                chosen.get(
                    "classifier_version",
                    self.physical_operator_lane_classifier_version,
                )
            ),
            "physical_operator_classifier_label": str(chosen.get("label", "")),
            "physical_operator_lane_source": str(chosen_source),
        }

    def geometry_policy_key(self) -> str:
        return (
            "fixed_local_v1"
            if int(self.phase3_geometry_window_size) >= 1
            else "legacy_coupled"
        )

    def attach_selector_metadata(
        self,
        *,
        feat_obj: CandidateFeatures,
        candidate_term: AnsatzTerm,
        selected_ops_now: Sequence[AnsatzTerm],
        window_terms: Sequence[AnsatzTerm],
        nested_window: NestedRefitWindow,
        phase3_geometry_window: NestedRefitWindow | None = None,
    ) -> tuple[CandidateFeatures, dict[str, Any]]:
        physical_payload = self.physical_payload_for_candidate(
            feat_obj=feat_obj,
            candidate_term=candidate_term,
        )
        if not bool(self.algebraic_lane_policy_active):
            if self.static_lane_route == STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE:
                return (
                    candidate_feature_with_updates(feat_obj, physical_payload),
                    dict(physical_payload),
                )
            return feat_obj, {}
        algebraic_payload = self.algebraic_payload_for_candidate(
            candidate_term=candidate_term,
            selected_ops_now=selected_ops_now,
            window_terms=window_terms,
        )
        nested_payload = serialize_nested_window(nested_window)
        accounting = build_nested_window_accounting(
            nested_window,
            compile_proxy_basis=COMPILE_PROXY_BASIS_OLD_PRE_INHERITED,
        )
        accounting_payload = serialize_nested_window_accounting(accounting)
        geometry_window = (
            phase3_geometry_window
            if phase3_geometry_window is not None
            else nested_window
        )
        geometry_payload = serialize_nested_window(geometry_window)
        geometry_accounting = build_nested_window_accounting(
            geometry_window,
            compile_proxy_basis=COMPILE_PROXY_BASIS_OLD_PRE_INHERITED,
        )
        geometry_accounting_payload = serialize_nested_window_accounting(
            geometry_accounting
        )
        window_new_old_pre_indices = _old_pre_indices_from_post_window(
            post_indices=geometry_window.window_new_post_indices,
            position_id=int(geometry_window.position_id),
        )
        window_age_old_pre_indices = _old_pre_indices_from_post_window(
            post_indices=geometry_window.window_age_post_indices,
            position_id=int(geometry_window.position_id),
        )
        policy_key = self.geometry_policy_key()
        updates: dict[str, Any] = {
            **dict(algebraic_payload),
            **dict(physical_payload),
            "phase3_selector_policy": str(self.phase3_selector_policy),
            "phase3_score_policy": "reduced_window_geometry_v1",
            "nested_refit_window": dict(nested_payload),
            "nested_window_accounting": dict(accounting_payload),
            "nested_refit_window_status": "predicted",
            "inherited_refit_window_indices": [
                int(value) for value in nested_window.old_pre_indices
            ],
            "active_post_refit_indices": [
                int(value) for value in nested_window.active_post_indices
            ],
            "selection_inherited_old_indices": [
                int(value) for value in accounting.selection_inherited_old_indices
            ],
            "optimizer_active_refit_indices": [
                int(value) for value in accounting.optimizer_active_refit_indices
            ],
            "optimizer_active_refit_count": int(accounting.optimizer_active_refit_count),
            "compile_proxy_basis": str(accounting.compile_proxy_basis),
            "compile_proxy_refit_count": int(accounting.compile_proxy_refit_count),
            "refit_window_basis": "old_pre_geometry_alias",
            "phase2_geometry_window_indices": [
                int(value) for value in geometry_window.old_pre_indices
            ],
            "phase2_geometry_window_policy": str(policy_key),
            "phase3_geometry_window_policy": str(policy_key),
            "phase3_geometry_window_size": int(self.phase3_geometry_window_size),
            "phase3_geometry_refit_window_indices": [
                int(value) for value in geometry_window.old_pre_indices
            ],
            "phase3_geometry_active_post_indices": [
                int(value) for value in geometry_window.active_post_indices
            ],
            "phase3_geometry_nested_refit_window": dict(geometry_payload),
            "phase3_geometry_window_accounting": dict(geometry_accounting_payload),
            "schur_window_indices": [
                int(value) for value in geometry_window.old_pre_indices
            ],
            "schur_window_policy": "phase3_geometry_refit_window_alias",
            "w3_wopt_decoupled": bool(int(self.phase3_geometry_window_size) >= 1),
            "window_origin": str(geometry_window.origin),
            "window_new_indices": [int(value) for value in window_new_old_pre_indices],
            "window_age_indices": [int(value) for value in window_age_old_pre_indices],
        }
        return candidate_feature_with_updates(feat_obj, updates), dict(updates)


__all__ = [
    "SelectorCandidateMetadataContext",
    "candidate_feature_with_updates",
]
