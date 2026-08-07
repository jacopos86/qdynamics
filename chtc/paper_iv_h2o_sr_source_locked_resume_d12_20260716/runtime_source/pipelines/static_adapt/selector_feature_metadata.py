"""Geometry and cache metadata for static ADAPT candidate features."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pipelines.scaffold.hh_continuation_types import CandidateFeatures
from pipelines.static_adapt.nested_windows import (
    COMPILE_PROXY_BASIS_OLD_PRE_INHERITED,
    NestedRefitWindow,
    NestedWindowAccounting,
    NestedWindowError,
    build_nested_window_accounting,
    nested_window_accounting_from_json,
)
from src.quantum.ansatz_parameterization import AnsatzParameterLayout


def selector_int_list(value: Any) -> list[int]:
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return []
    if not isinstance(value, Sequence):
        return []
    out: list[int] = []
    for raw in value:
        try:
            out.append(int(raw))
        except (TypeError, ValueError):
            continue
    return out


def logical_runtime_reduced_position_groups(
    layout: AnsatzParameterLayout,
    logical_indices: Sequence[int],
    runtime_to_reduced: Mapping[int, int],
) -> tuple[list[list[int]], list[int], bool, bool]:
    """Map logical blocks to full runtime reduced-position groups."""

    groups: list[list[int]] = []
    flattened_runtime: list[int] = []
    basis_supported = True
    full_active_window = True
    for raw_idx in logical_indices:
        logical_idx = int(raw_idx)
        if logical_idx < 0 or logical_idx >= int(layout.logical_parameter_count):
            basis_supported = False
            full_active_window = False
            groups.append([])
            continue
        block = layout.blocks[int(logical_idx)]
        if int(block.runtime_count) <= 0:
            basis_supported = False
            full_active_window = False
            groups.append([])
            continue
        runtime_indices = [
            int(i) for i in range(int(block.runtime_start), int(block.runtime_stop))
        ]
        reduced_group: list[int] = []
        for runtime_idx in runtime_indices:
            flattened_runtime.append(int(runtime_idx))
            reduced_pos = runtime_to_reduced.get(int(runtime_idx))
            if reduced_pos is None:
                full_active_window = False
            else:
                reduced_group.append(int(reduced_pos))
        if len(reduced_group) != len(runtime_indices):
            full_active_window = False
        groups.append(reduced_group)
    return groups, flattened_runtime, bool(basis_supported), bool(full_active_window)


def _nested_validation_status(reason: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() else "_" for ch in str(reason).strip().lower()
    ).strip("_")
    if not cleaned:
        cleaned = "unknown"
    return f"invalid_payload_{cleaned[:80]}"


@dataclass(frozen=True)
class SelectorFeatureMetadataContext:
    algebraic_lane_policy_active: bool
    static_lane_route: str
    static_lane_route_lane_key: str
    static_lane_route_lanes: tuple[str, ...]
    physical_operator_lane_classifier_version: str | None
    phase1_shortlist_size_base: int
    phase1_shortlist_size_effective: int
    phase2_shortlist_size_base: int
    phase2_shortlist_size_effective: int
    phase2_shortlist_fraction_base: float
    phase2_shortlist_fraction_effective: float
    physical_lane_shortlist_aggressiveness: int
    cache_jsonable: Callable[[Any], Any]
    validate_nested_window: Callable[..., None]

    def phase2_geometry_indices(self, feat_base: Any) -> list[int]:
        if not isinstance(feat_base, CandidateFeatures):
            return []
        explicit = selector_int_list(
            getattr(feat_base, "phase2_geometry_window_indices", [])
        )
        if explicit:
            return explicit
        return selector_int_list(getattr(feat_base, "refit_window_indices", []))

    def phase3_schur_indices(self, feat_base: Any) -> list[int]:
        if not isinstance(feat_base, CandidateFeatures):
            return []
        raw_schur = getattr(feat_base, "schur_window_indices", None)
        schur = selector_int_list(raw_schur)
        schur_policy = str(
            getattr(
                feat_base,
                "schur_window_policy",
                "phase3_geometry_refit_window_alias",
            )
        )
        p3_policy = str(
            getattr(feat_base, "phase3_geometry_window_policy", "legacy_coupled")
        )
        if raw_schur is not None and (
            schur
            or schur_policy != "phase3_geometry_refit_window_alias"
            or p3_policy == "fixed_local_v1"
        ):
            return schur
        raw_p3 = getattr(feat_base, "phase3_geometry_refit_window_indices", None)
        p3 = selector_int_list(raw_p3)
        if raw_p3 is not None and (p3 or p3_policy != "legacy_coupled"):
            return p3
        return self.phase2_geometry_indices(feat_base)

    def validate_selected_nested_window(
        self,
        *,
        raw_payload: Mapping[str, Any] | None,
        source_inherited_indices: Sequence[int],
        source_active_post_indices: Sequence[int],
        source_accounting_payload: Mapping[str, Any] | None,
        selected_position: int,
        n_theta_logical: int,
    ) -> tuple[list[int], list[int], NestedWindowAccounting | None, str]:
        if not bool(self.algebraic_lane_policy_active):
            return [], [], None, "policy_inactive"
        if not isinstance(raw_payload, Mapping):
            return [], [], None, "missing_payload"
        try:
            pre_n = int(raw_payload.get("pre_parameter_count", int(n_theta_logical) - 1))
            post_n = int(raw_payload.get("post_parameter_count", int(n_theta_logical)))
            window = NestedRefitWindow(
                window_version=str(
                    raw_payload.get("window_version", "nested_refit_window_v1")
                ),
                origin=str(raw_payload.get("origin", "nested_inherited_v1")),
                policy_requested=str(raw_payload.get("policy_requested", "unknown")),
                policy_effective=str(raw_payload.get("policy_effective", "unknown")),
                pre_parameter_count=int(pre_n),
                post_parameter_count=int(post_n),
                position_id=int(raw_payload.get("position_id", selected_position)),
                candidate_post_index=int(
                    raw_payload.get("candidate_post_index", selected_position)
                ),
                old_pre_indices=tuple(selector_int_list(raw_payload.get("old_pre_indices"))),
                old_post_indices=tuple(selector_int_list(raw_payload.get("old_post_indices"))),
                active_post_indices=tuple(
                    selector_int_list(raw_payload.get("active_post_indices"))
                ),
                window_new_post_indices=tuple(
                    selector_int_list(raw_payload.get("window_new_post_indices"))
                ),
                window_age_post_indices=tuple(
                    selector_int_list(raw_payload.get("window_age_post_indices"))
                ),
                periodic_full_refit_triggered=bool(
                    raw_payload.get("periodic_full_refit_triggered", False)
                ),
            )
            if int(window.post_parameter_count) != int(n_theta_logical):
                raise NestedWindowError(
                    "post count does not match current logical theta length"
                )
            if int(window.position_id) != int(selected_position):
                raise NestedWindowError("position_id does not match selected position")
            if int(window.candidate_post_index) != int(selected_position):
                raise NestedWindowError(
                    "candidate_post_index does not match selected position"
                )
            if int(selected_position) not in {
                int(i) for i in window.active_post_indices
            }:
                raise NestedWindowError(
                    "selected position is absent from active_post_indices"
                )
            self.validate_nested_window(
                window,
                allowed_old_pre_indices=list(
                    range(max(0, int(window.pre_parameter_count)))
                ),
            )
            if source_inherited_indices and tuple(
                int(x) for x in source_inherited_indices
            ) != tuple(window.old_pre_indices):
                raise NestedWindowError(
                    "source inherited indices do not match payload"
                )
            if source_active_post_indices and tuple(
                int(x) for x in source_active_post_indices
            ) != tuple(window.active_post_indices):
                raise NestedWindowError(
                    "source active-post indices do not match payload"
                )
            source_accounting: NestedWindowAccounting | None = None
            if isinstance(source_accounting_payload, Mapping):
                source_accounting = nested_window_accounting_from_json(
                    source_accounting_payload
                )
                if tuple(source_accounting.old_pre_indices) != tuple(
                    window.old_pre_indices
                ):
                    raise NestedWindowError(
                        "source accounting old_pre_indices do not match payload"
                    )
                if tuple(source_accounting.active_post_indices) != tuple(
                    window.active_post_indices
                ):
                    raise NestedWindowError(
                        "source accounting active_post_indices do not match payload"
                    )
                if int(source_accounting.candidate_post_index) != int(
                    window.candidate_post_index
                ):
                    raise NestedWindowError(
                        "source accounting candidate_post_index does not match payload"
                    )
            basis = (
                str(source_accounting.compile_proxy_basis)
                if source_accounting is not None
                else COMPILE_PROXY_BASIS_OLD_PRE_INHERITED
            )
            accounting = build_nested_window_accounting(
                window,
                compile_proxy_basis=str(basis),
            )
            if source_accounting is not None:
                if int(source_accounting.compile_proxy_refit_count) != int(
                    accounting.compile_proxy_refit_count
                ):
                    raise NestedWindowError(
                        "source accounting compile_proxy_refit_count does not match payload"
                    )
                if int(source_accounting.optimizer_active_refit_count) != int(
                    accounting.optimizer_active_refit_count
                ):
                    raise NestedWindowError(
                        "source accounting optimizer_active_refit_count does not match payload"
                    )
            return (
                [int(x) for x in window.old_pre_indices],
                [int(x) for x in window.active_post_indices],
                accounting,
                "ready",
            )
        except Exception as exc:
            return [], [], None, _nested_validation_status(str(exc))

    def compile_proxy_refit_count(self, feat_base: Any) -> int:
        refit_window = list(getattr(feat_base, "refit_window_indices", []) or [])
        if not bool(self.algebraic_lane_policy_active):
            return int(len(refit_window))
        accounting_payload = getattr(feat_base, "nested_window_accounting", None)
        if isinstance(accounting_payload, Mapping):
            return int(
                nested_window_accounting_from_json(
                    accounting_payload
                ).compile_proxy_refit_count
            )
        basis = str(getattr(feat_base, "compile_proxy_basis", "legacy"))
        if basis != "legacy":
            return int(getattr(feat_base, "compile_proxy_refit_count", 0))
        return int(len(refit_window))

    def inherited_selector_updates(self, feat_base: Any) -> dict[str, Any]:
        if not isinstance(feat_base, CandidateFeatures):
            return {}
        return {
            "algebraic_lane": feat_base.algebraic_lane,
            "algebraic_quality": feat_base.algebraic_quality,
            "algebraic_context_counts": dict(feat_base.algebraic_context_counts),
            "algebraic_context_labels": [
                str(x) for x in feat_base.algebraic_context_labels
            ],
            "static_lane_route": str(feat_base.static_lane_route),
            "physical_operator_lane": feat_base.physical_operator_lane,
            "physical_operator_quality": feat_base.physical_operator_quality,
            "physical_operator_hh_full_meta_class": (
                feat_base.physical_operator_hh_full_meta_class
            ),
            "physical_operator_classifier_version": (
                feat_base.physical_operator_classifier_version
            ),
            "physical_operator_classifier_label": (
                feat_base.physical_operator_classifier_label
            ),
            "physical_operator_lane_source": feat_base.physical_operator_lane_source,
            "physical_operator_lane_health": feat_base.physical_operator_lane_health,
            "physical_operator_lane_relative_health": (
                feat_base.physical_operator_lane_relative_health
            ),
            "physical_operator_lane_live": feat_base.physical_operator_lane_live,
            "phase3_selector_policy": str(feat_base.phase3_selector_policy),
            "phase3_score_policy": str(feat_base.phase3_score_policy),
            "nested_refit_window": (
                dict(feat_base.nested_refit_window)
                if isinstance(feat_base.nested_refit_window, Mapping)
                else None
            ),
            "nested_window_accounting": (
                dict(feat_base.nested_window_accounting)
                if isinstance(feat_base.nested_window_accounting, Mapping)
                else None
            ),
            "nested_refit_window_status": str(feat_base.nested_refit_window_status),
            "refit_window_basis": str(feat_base.refit_window_basis),
            "phase2_geometry_window_indices": [
                int(x) for x in feat_base.phase2_geometry_window_indices
            ],
            "phase2_geometry_window_policy": str(
                feat_base.phase2_geometry_window_policy
            ),
            "phase3_geometry_window_policy": str(
                feat_base.phase3_geometry_window_policy
            ),
            "phase3_geometry_window_size": int(feat_base.phase3_geometry_window_size),
            "phase3_geometry_refit_window_indices": [
                int(x) for x in feat_base.phase3_geometry_refit_window_indices
            ],
            "phase3_geometry_active_post_indices": [
                int(x) for x in feat_base.phase3_geometry_active_post_indices
            ],
            "phase3_geometry_nested_refit_window": (
                dict(feat_base.phase3_geometry_nested_refit_window)
                if isinstance(feat_base.phase3_geometry_nested_refit_window, Mapping)
                else None
            ),
            "phase3_geometry_window_accounting": (
                dict(feat_base.phase3_geometry_window_accounting)
                if isinstance(feat_base.phase3_geometry_window_accounting, Mapping)
                else None
            ),
            "schur_window_indices": [
                int(x) for x in feat_base.schur_window_indices
            ],
            "schur_window_policy": str(feat_base.schur_window_policy),
            "w3_wopt_decoupled": bool(feat_base.w3_wopt_decoupled),
            "inherited_refit_window_indices": [
                int(x) for x in feat_base.inherited_refit_window_indices
            ],
            "active_post_refit_indices": [
                int(x) for x in feat_base.active_post_refit_indices
            ],
            "selection_inherited_old_indices": [
                int(x) for x in feat_base.selection_inherited_old_indices
            ],
            "optimizer_active_refit_indices": [
                int(x) for x in feat_base.optimizer_active_refit_indices
            ],
            "optimizer_active_refit_count": int(
                feat_base.optimizer_active_refit_count
            ),
            "compile_proxy_basis": str(feat_base.compile_proxy_basis),
            "compile_proxy_refit_count": int(feat_base.compile_proxy_refit_count),
            "window_origin": str(feat_base.window_origin),
            "window_new_indices": [int(x) for x in feat_base.window_new_indices],
            "window_age_indices": [int(x) for x in feat_base.window_age_indices],
        }

    def scoring_cache_payload(
        self,
        feat_base: CandidateFeatures,
    ) -> dict[str, Any]:
        return {
            "stage_name": str(feat_base.stage_name),
            "candidate_family": str(feat_base.candidate_family),
            "candidate_pool_index": int(feat_base.candidate_pool_index),
            "position_id": int(feat_base.position_id),
            "append_position": int(feat_base.append_position),
            "positions_considered": [
                int(x) for x in feat_base.positions_considered
            ],
            "sigma_hat": float(feat_base.sigma_hat),
            "refit_window_indices": [
                int(x) for x in feat_base.refit_window_indices
            ],
            "phase2_geometry_indices": [
                int(x) for x in self.phase2_geometry_indices(feat_base)
            ],
            "phase2_geometry_window_policy": str(
                feat_base.phase2_geometry_window_policy
            ),
            "phase3_schur_indices": [
                int(x) for x in self.phase3_schur_indices(feat_base)
            ],
            "phase3_geometry_active_post_indices": [
                int(x) for x in feat_base.phase3_geometry_active_post_indices
            ],
            "phase3_geometry_window_policy": str(
                feat_base.phase3_geometry_window_policy
            ),
            "phase3_geometry_window_size": int(feat_base.phase3_geometry_window_size),
            "schur_window_policy": str(feat_base.schur_window_policy),
            "inherited_refit_window_indices": [
                int(x) for x in feat_base.inherited_refit_window_indices
            ],
            "active_post_refit_indices": [
                int(x) for x in feat_base.active_post_refit_indices
            ],
            "optimizer_active_refit_indices": [
                int(x) for x in feat_base.optimizer_active_refit_indices
            ],
            "compile_proxy_refit_count": int(
                self.compile_proxy_refit_count(feat_base)
            ),
            "stage_gate_open": bool(feat_base.stage_gate_open),
            "trough_probe_triggered": bool(feat_base.trough_probe_triggered),
            "trough_detected": bool(feat_base.trough_detected),
            "symmetry_mode": str(feat_base.symmetry_mode),
            "symmetry_mitigation_mode": str(feat_base.symmetry_mitigation_mode),
            "motif_metadata": self.cache_jsonable(feat_base.motif_metadata),
            "motif_bonus": float(feat_base.motif_bonus or 0.0),
            "motif_source": str(feat_base.motif_source),
            "lifetime_cost_mode": str(feat_base.lifetime_cost_mode),
            "remaining_evaluations_proxy_mode": str(
                feat_base.remaining_evaluations_proxy_mode
            ),
            "family_repeat_cost": float(feat_base.family_repeat_cost),
            "static_lane_route": str(self.static_lane_route),
            "static_lane_route_lane_key": str(self.static_lane_route_lane_key),
            "static_lane_route_lanes": [
                str(lane) for lane in self.static_lane_route_lanes
            ],
            "physical_operator_lane_classifier_version": (
                None
                if self.physical_operator_lane_classifier_version is None
                else str(self.physical_operator_lane_classifier_version)
            ),
            "phase1_shortlist_size_base": int(self.phase1_shortlist_size_base),
            "phase1_shortlist_size_effective": int(
                self.phase1_shortlist_size_effective
            ),
            "phase2_shortlist_size_base": int(self.phase2_shortlist_size_base),
            "phase2_shortlist_size_effective": int(
                self.phase2_shortlist_size_effective
            ),
            "phase2_shortlist_fraction_base": float(
                self.phase2_shortlist_fraction_base
            ),
            "phase2_shortlist_fraction_effective": float(
                self.phase2_shortlist_fraction_effective
            ),
            "physical_lane_shortlist_aggressiveness": int(
                self.physical_lane_shortlist_aggressiveness
            ),
            "physical_operator_lane": feat_base.physical_operator_lane,
            "physical_operator_hh_full_meta_class": (
                feat_base.physical_operator_hh_full_meta_class
            ),
            "physical_operator_classifier_label": (
                feat_base.physical_operator_classifier_label
            ),
            "physical_operator_lane_source": feat_base.physical_operator_lane_source,
            "inherited_selector_updates": self.cache_jsonable(
                self.inherited_selector_updates(feat_base)
            ),
        }


__all__ = [
    "SelectorFeatureMetadataContext",
    "logical_runtime_reduced_position_groups",
    "selector_int_list",
]
