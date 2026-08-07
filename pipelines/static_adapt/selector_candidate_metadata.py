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
from pipelines.static_adapt.lane_routes import (
    GLOBAL_SINGLE_POPULATION_LANE,
    STATIC_LANE_ROUTE_GLOBAL_SINGLE_POPULATION,
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
    static_lane_route: str
    problem_key: str
    physical_operator_lane_summary: dict[str, Any]
    physical_operator_lane_classifier_version: str
    phase3_selector_policy: str
    phase3_geometry_window_size: int
    phase3_response_coordinate_scope: str

    def physical_payload_for_candidate(
        self,
        *,
        feat_obj: CandidateFeatures,
        candidate_term: AnsatzTerm,
    ) -> dict[str, Any]:
        if self.static_lane_route == STATIC_LANE_ROUTE_GLOBAL_SINGLE_POPULATION:
            # Lanes-off arm: every candidate belongs to one global population,
            # so no physical-family classification is performed and the lane
            # summary records a single bucket.
            summary = self.physical_operator_lane_summary
            summary["classified_count"] = (
                int(summary.get("classified_count", 0)) + 1
            )
            lane_counts = summary.setdefault("lane_counts", {})
            if isinstance(lane_counts, dict):
                lane_counts[GLOBAL_SINGLE_POPULATION_LANE] = (
                    int(lane_counts.get(GLOBAL_SINGLE_POPULATION_LANE, 0)) + 1
                )
            return {
                "static_lane_route": str(self.static_lane_route),
                "physical_operator_lane": GLOBAL_SINGLE_POPULATION_LANE,
                "physical_operator_quality": "global_single_population",
                "physical_operator_hh_full_meta_class": None,
                "physical_operator_classifier_version": (
                    "global_single_population_v1"
                ),
                "physical_operator_classifier_label": "",
                "physical_operator_lane_source": "global_single_population",
            }
        if self.static_lane_route != STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE:
            raise ValueError(
                "Selector candidate metadata supports only the retained "
                "physical-operator-family lane route; got "
                f"{self.static_lane_route!r}."
            )

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

    def response_policy_key(self) -> str:
        return str(self.phase3_response_coordinate_scope)

    def response_geometry_policy_key(self) -> str:
        scope = self.response_policy_key()
        if scope == "full_active_plus_singleton_v1":
            return "full_active_plus_singleton_v1"
        if scope == "candidate_material_coupling_window_v1":
            return "candidate_material_coupling_window_v1"
        if scope == "fixed_local_window_v1":
            return "fixed_local_v1"
        return "legacy_coupled"

    def attach_selector_metadata(
        self,
        *,
        feat_obj: CandidateFeatures,
        candidate_term: AnsatzTerm,
        selected_ops_now: Sequence[AnsatzTerm],
        window_terms: Sequence[AnsatzTerm],
        nested_window: NestedRefitWindow,
        phase2_geometry_window: NestedRefitWindow | None = None,
        phase3_geometry_window: NestedRefitWindow | None = None,
    ) -> tuple[CandidateFeatures, dict[str, Any]]:
        physical_payload = self.physical_payload_for_candidate(
            feat_obj=feat_obj,
            candidate_term=candidate_term,
        )
        nested_payload = serialize_nested_window(nested_window)
        accounting = build_nested_window_accounting(
            nested_window,
            compile_proxy_basis=COMPILE_PROXY_BASIS_OLD_PRE_INHERITED,
        )
        accounting_payload = serialize_nested_window_accounting(accounting)
        phase2_window = (
            phase2_geometry_window
            if phase2_geometry_window is not None
            else nested_window
        )
        response_window = (
            phase3_geometry_window
            if phase3_geometry_window is not None
            else nested_window
        )
        geometry_payload = serialize_nested_window(response_window)
        geometry_accounting = build_nested_window_accounting(
            response_window,
            compile_proxy_basis=COMPILE_PROXY_BASIS_OLD_PRE_INHERITED,
        )
        geometry_accounting_payload = serialize_nested_window_accounting(
            geometry_accounting
        )
        window_new_old_pre_indices = _old_pre_indices_from_post_window(
            post_indices=phase2_window.window_new_post_indices,
            position_id=int(phase2_window.position_id),
        )
        window_age_old_pre_indices = _old_pre_indices_from_post_window(
            post_indices=phase2_window.window_age_post_indices,
            position_id=int(phase2_window.position_id),
        )
        policy_key = self.geometry_policy_key()
        updates: dict[str, Any] = {
            **dict(physical_payload),
            "phase3_selector_policy": str(self.phase3_selector_policy),
            "phase3_score_policy": (
                "full_active_plus_singleton_response_v1"
                if self.response_policy_key() == "full_active_plus_singleton_v1"
                else (
                    "candidate_material_coupling_response_v1"
                    if self.response_policy_key()
                    == "candidate_material_coupling_window_v1"
                    else "reduced_window_geometry_v1"
                )
            ),
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
                int(value) for value in phase2_window.old_pre_indices
            ],
            "phase2_geometry_window_policy": str(policy_key),
            "phase3_geometry_window_policy": str(
                self.response_geometry_policy_key()
            ),
            "phase3_geometry_window_size": int(self.phase3_geometry_window_size),
            "phase3_geometry_refit_window_indices": [
                int(value) for value in response_window.old_pre_indices
            ],
            "phase3_geometry_active_post_indices": [
                int(value) for value in response_window.active_post_indices
            ],
            "phase3_response_coordinate_scope": str(self.response_policy_key()),
            "phase3_response_coordinate_indices": [
                int(value) for value in response_window.active_post_indices
            ],
            "phase3_response_pre_support_count": int(
                len(response_window.active_post_indices)
            ),
            "phase3_active_logical_coordinate_count": int(
                response_window.pre_parameter_count
            ),
            "phase3_geometry_nested_refit_window": dict(geometry_payload),
            "phase3_geometry_window_accounting": dict(geometry_accounting_payload),
            "schur_window_indices": [
                int(value) for value in response_window.old_pre_indices
            ],
            "schur_window_policy": "phase3_geometry_refit_window_alias",
            "w3_wopt_decoupled": bool(
                tuple(response_window.active_post_indices)
                != tuple(nested_window.active_post_indices)
            ),
            "window_origin": str(phase2_window.origin),
            "window_new_indices": [int(value) for value in window_new_old_pre_indices],
            "window_age_indices": [int(value) for value in window_age_old_pre_indices],
        }
        return candidate_feature_with_updates(feat_obj, updates), dict(updates)


__all__ = [
    "SelectorCandidateMetadataContext",
    "candidate_feature_with_updates",
]
