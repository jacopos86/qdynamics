"""Phase-3 runtime-split policy helpers for static ADAPT."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Mapping, Sequence

from pipelines.scaffold.hh_continuation_generators import (
    build_generator_metadata,
    build_runtime_split_child_sets,
    build_runtime_split_children,
    serialize_polynomial_terms_exyz,
)
from pipelines.static_adapt.route_a_child_padding import (
    ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
    ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_POLICIES,
    RouteAChildPaddingConfig,
    project_route_a_child_polynomial,
)
from pipelines.static_adapt.route_a_shortlists import (
    canonicalize_pauli_child_direction,
    pauli_child_identity,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.vqe_latex_python_pairs import AnsatzTerm

__all__ = [
    "ROUTE_A_CHILD_SYMMETRY_GUARD_FIXED_SECTOR_V1",
    "ROUTE_A_RUNTIME_SPLIT_CHILD_PADDING_SCHEMA",
    "_PHASE3_RUNTIME_SPLIT_CHILD_SET_SYMMETRY_POLICIES",
    "_phase3_runtime_split_needs_proxy_child_set_scoring",
    "_phase3_runtime_split_archival_polynomial_term_count",
    "_phase3_runtime_split_parent_eligible",
    "_normalize_phase3_runtime_split_child_set_symmetry_policy",
    "_phase3_runtime_split_child_set_symmetry_spec",
    "build_global_child_records_for_parent",
    "project_and_deduplicate_runtime_split_child_sets",
]

_PHASE3_RUNTIME_SPLIT_CHILD_SET_SYMMETRY_POLICIES = {"off", "parent", "hard_guard"}
ROUTE_A_CHILD_SYMMETRY_GUARD_FIXED_SECTOR_V1 = "fixed_count_sector_invariance_v1"
ROUTE_A_RUNTIME_SPLIT_CHILD_PADDING_SCHEMA = (
    "route_a_runtime_split_child_set_padding_v1"
)


def _runtime_split_parent_label(
    child_set: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str:
    direct = child_set.get("runtime_split_parent_label")
    if direct not in {None, ""}:
        return str(direct)
    compile_metadata = metadata.get("compile_metadata")
    runtime_split = (
        compile_metadata.get("runtime_split")
        if isinstance(compile_metadata, Mapping)
        else None
    )
    if isinstance(runtime_split, Mapping):
        parent = runtime_split.get("parent_label")
        if parent not in {None, ""}:
            return str(parent)
    return ""


def _projected_child_label(raw_label: str) -> str:
    suffix = "::legal_projected"
    return (
        str(raw_label)
        if str(raw_label).endswith(suffix)
        else f"{raw_label}{suffix}"
    )


def project_and_deduplicate_runtime_split_child_sets(
    child_sets: Sequence[Mapping[str, Any]],
    *,
    config: RouteAChildPaddingConfig,
    num_sites: int,
    ordering: str,
    qpb: int,
    fixed_num_particles: Sequence[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Project archival child sets before scoring and deduplicate per parent.

    The helper is deliberately independent of ``RouteAFunnelConfig`` so an
    archival ``archival_child_set_forward_v1`` recovery can add the approved
    padding enforcement without changing macro shortlist caps or fractions.
    Projection is applied to every candidate polynomial first.  The resulting
    direction is then canonicalized modulo one global scalar, and equivalent
    projected directions are deduplicated only within the same parent and
    insertion position.  Every raw term row is retained in lineage telemetry.
    """

    if str(config.policy) != ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1:
        raise ValueError(
            "Archival runtime-split child projection requires the cutoff-generic "
            f"policy {ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1!r}; "
            f"got {config.policy!r}."
        )
    if int(num_sites) < 1:
        raise ValueError("num_sites must be positive for runtime child projection.")
    if int(qpb) < 1:
        raise ValueError("qpb must be positive for runtime child projection.")

    projected_rows: list[dict[str, Any]] = []
    projection_events: list[dict[str, Any]] = []
    zero_rejections: list[dict[str, Any]] = []
    for source_ordinal, source in enumerate(child_sets):
        row = dict(source)
        raw_polynomial = row.get("candidate_polynomial")
        raw_metadata = row.get("candidate_generator_metadata")
        if not isinstance(raw_polynomial, PauliPolynomial):
            raise ValueError(
                "Runtime-split padding candidate is missing candidate_polynomial: "
                f"ordinal={source_ordinal}."
            )
        if not isinstance(raw_metadata, Mapping):
            raise ValueError(
                "Runtime-split padding candidate is missing candidate_generator_metadata: "
                f"ordinal={source_ordinal}."
            )
        raw_metadata_dict = dict(raw_metadata)
        raw_label = str(row.get("candidate_label", ""))
        parent_label = _runtime_split_parent_label(row, raw_metadata_dict)
        position_id = (
            None
            if row.get("position_id") is None
            else int(row.get("position_id"))
        )
        raw_terms = serialize_polynomial_terms_exyz(raw_polynomial)
        raw_term = AnsatzTerm(
            label=str(raw_label),
            polynomial=raw_polynomial,
            execution_mode=str(
                row.get("recommended_execution_mode") or "termwise_product"
            ),
        )
        raw_identity = pauli_child_identity({"candidate_term": raw_term})
        raw_compile_metadata = raw_metadata_dict.get("compile_metadata")
        raw_runtime_split = (
            dict(raw_compile_metadata.get("runtime_split", {}))
            if isinstance(raw_compile_metadata, Mapping)
            and isinstance(raw_compile_metadata.get("runtime_split"), Mapping)
            else {}
        )
        raw_symmetry_gate = (
            dict(row.get("symmetry_gate", {}))
            if isinstance(row.get("symmetry_gate"), Mapping)
            else {}
        )
        projected_polynomial, projection_payload = project_route_a_child_polynomial(
            raw_polynomial,
            config=config,
        )
        common_lineage = {
            "source_ordinal": int(source_ordinal),
            "parent_label": str(parent_label),
            "position_id": position_id,
            "raw_candidate_label": str(raw_label),
            "raw_generator_id": (
                None
                if raw_metadata_dict.get("generator_id") in {None, ""}
                else str(raw_metadata_dict.get("generator_id"))
            ),
            "raw_parent_generator_id": (
                None
                if raw_metadata_dict.get("parent_generator_id") in {None, ""}
                else str(raw_metadata_dict.get("parent_generator_id"))
            ),
            "raw_identity": str(raw_identity),
            "raw_term_order": "polynomial_iteration_order",
            "raw_serialized_terms_exyz": [dict(term) for term in raw_terms],
            "raw_child_indices": [
                int(value) for value in row.get("child_indices", [])
            ],
            "raw_child_labels": [
                str(value) for value in row.get("child_labels", [])
            ],
            "raw_child_generator_ids": [
                str(value) for value in row.get("child_generator_ids", [])
            ],
            "raw_runtime_split": dict(raw_runtime_split),
            "raw_symmetry_gate": dict(raw_symmetry_gate),
        }
        if projected_polynomial is None:
            rejection = {
                **common_lineage,
                "status": "rejected",
                "reason": str(
                    projection_payload.get("reason", "projection_is_zero")
                ),
                "projection": dict(projection_payload),
            }
            zero_rejections.append(rejection)
            projection_events.append(dict(rejection))
            continue

        projected_label = _projected_child_label(raw_label)
        pre_normalization_terms = serialize_polynomial_terms_exyz(
            projected_polynomial
        )
        canonical_term, normalization_payload = canonicalize_pauli_child_direction(
            AnsatzTerm(
                label=str(projected_label),
                polynomial=projected_polynomial,
                execution_mode="grouped_exact",
            )
        )
        canonical_polynomial = getattr(canonical_term, "polynomial", None)
        if not isinstance(canonical_polynomial, PauliPolynomial):
            raise ValueError(
                "Projected runtime-split child direction could not be canonicalized: "
                f"label={raw_label!r}."
            )
        if str(normalization_payload.get("status")) != "normalized":
            raise ValueError(
                "Projected runtime-split child direction normalization failed: "
                f"label={raw_label!r}, payload={normalization_payload!r}."
            )
        canonical_term = AnsatzTerm(
            label=str(projected_label),
            polynomial=canonical_polynomial,
            execution_mode="grouped_exact",
        )
        canonical_terms = serialize_polynomial_terms_exyz(canonical_polynomial)
        projected_identity = pauli_child_identity(
            {"candidate_term": canonical_term}
        )
        symmetry_spec = (
            dict(raw_metadata_dict.get("symmetry_spec", {}))
            if isinstance(raw_metadata_dict.get("symmetry_spec"), Mapping)
            else None
        )
        projected_metadata = asdict(
            build_generator_metadata(
                label=str(projected_label),
                polynomial=canonical_polynomial,
                family_id=str(
                    raw_metadata_dict.get("family_id")
                    or "runtime_split_projected"
                ),
                num_sites=int(num_sites),
                ordering=str(ordering),
                qpb=int(qpb),
                split_policy="runtime_split_projected_child",
                parent_generator_id=(
                    str(raw_metadata_dict.get("parent_generator_id"))
                    if raw_metadata_dict.get("parent_generator_id")
                    not in {None, ""}
                    else None
                ),
                symmetry_spec=symmetry_spec,
                fixed_num_particles=fixed_num_particles,
                serialized_terms=canonical_terms,
            )
        )
        lineage = {
            **common_lineage,
            "status": "projected",
            "projected_candidate_label": str(projected_label),
            "projected_identity": str(projected_identity),
            "projection": dict(projection_payload),
            "pre_normalization_projected_term_order": (
                "deterministic_pauli_label_order"
            ),
            "pre_normalization_projected_serialized_terms_exyz": [
                dict(term) for term in pre_normalization_terms
            ],
            "direction_normalization": dict(normalization_payload),
            "selected_projected_term_order": (
                "projective_unit_norm_positive_anchor_order"
            ),
            "selected_projected_serialized_terms_exyz": [
                dict(term) for term in canonical_terms
            ],
            "selected_execution_mode": "grouped_exact",
        }
        projected_compile_metadata = dict(
            projected_metadata.get("compile_metadata", {})
        )
        projected_compile_metadata["runtime_split"] = {
            **dict(raw_runtime_split),
            "padding_projection_policy": str(config.policy),
            "raw_candidate_label": str(raw_label),
            "projected_candidate_label": str(projected_label),
            "projected_identity": str(projected_identity),
            "recommended_execution_mode": "grouped_exact",
        }
        projected_compile_metadata[
            "route_a_child_padding_projection"
        ] = dict(projection_payload)
        projected_compile_metadata[
            "route_a_child_padding_lineage"
        ] = dict(lineage)
        projected_metadata["compile_metadata"] = projected_compile_metadata
        projected_metadata["route_a_child_padding_projection"] = dict(
            projection_payload
        )
        projected_metadata["route_a_child_padding_lineage"] = dict(lineage)
        projected_row = {
            **row,
            "candidate_label": str(projected_label),
            "candidate_polynomial": canonical_polynomial,
            "candidate_generator_metadata": projected_metadata,
            "recommended_execution_mode": "grouped_exact",
            "route_a_child_padding_projection": dict(projection_payload),
            "route_a_child_padding_lineage": dict(lineage),
            "route_a_projected_child_identity": str(projected_identity),
            "route_a_child_direction_normalization": dict(
                normalization_payload
            ),
        }
        projected_rows.append(projected_row)
        projection_events.append(dict(lineage))

    grouped: dict[tuple[str, int | None, str], list[dict[str, Any]]] = {}
    for row in projected_rows:
        metadata = row.get("candidate_generator_metadata")
        metadata_mapping = metadata if isinstance(metadata, Mapping) else {}
        parent_label = _runtime_split_parent_label(row, metadata_mapping)
        position_id = (
            None
            if row.get("position_id") is None
            else int(row.get("position_id"))
        )
        identity = str(row.get("route_a_projected_child_identity", ""))
        grouped.setdefault((parent_label, position_id, identity), []).append(row)

    retained: list[dict[str, Any]] = []
    deduplication_events: list[dict[str, Any]] = []
    for (parent_label, position_id, projected_identity), rows in grouped.items():
        representative = dict(rows[0])
        source_lineage = [
            dict(row.get("route_a_child_padding_lineage", {}))
            for row in rows
            if isinstance(row.get("route_a_child_padding_lineage"), Mapping)
        ]
        source_labels = [
            str(lineage.get("raw_candidate_label", ""))
            for lineage in source_lineage
        ]
        deduplication = {
            "parent_label": str(parent_label),
            "position_id": position_id,
            "projected_identity": str(projected_identity),
            "representative_candidate_label": str(
                representative.get("candidate_label", "")
            ),
            "source_count": int(len(rows)),
            "source_candidate_labels_in_raw_order": source_labels,
            "duplicate_count": int(max(0, len(rows) - 1)),
        }
        representative[
            "route_a_child_padding_source_lineage"
        ] = source_lineage
        representative[
            "route_a_child_padding_source_labels"
        ] = source_labels
        representative[
            "route_a_child_padding_source_count"
        ] = int(len(rows))
        representative[
            "route_a_child_padding_deduplication"
        ] = dict(deduplication)
        metadata = representative.get("candidate_generator_metadata")
        if isinstance(metadata, Mapping):
            metadata_out = dict(metadata)
            compile_metadata = dict(metadata_out.get("compile_metadata", {}))
            compile_metadata[
                "route_a_child_padding_source_lineage"
            ] = source_lineage
            compile_metadata[
                "route_a_child_padding_deduplication"
            ] = dict(deduplication)
            metadata_out["compile_metadata"] = compile_metadata
            metadata_out[
                "route_a_child_padding_source_lineage"
            ] = source_lineage
            metadata_out[
                "route_a_child_padding_deduplication"
            ] = dict(deduplication)
            representative["candidate_generator_metadata"] = metadata_out
        retained.append(representative)
        if len(rows) > 1:
            deduplication_events.append(dict(deduplication))

    telemetry = {
        "schema": ROUTE_A_RUNTIME_SPLIT_CHILD_PADDING_SCHEMA,
        "policy_requested": str(config.policy),
        "policy_effective": str(config.policy),
        "active": True,
        "enforcement_stage": (
            "post_runtime_split_child_set_construction_pre_score_v1"
        ),
        "deduplication_scope": (
            "per_parent_and_position_projected_identity_v1"
        ),
        "input_candidate_count": int(len(child_sets)),
        "projection_input_count": int(len(child_sets)),
        "projection_output_count": int(len(projected_rows)),
        "projection_zero_rejection_count": int(len(zero_rejections)),
        "projected_candidate_count_before_deduplication": int(
            len(projected_rows)
        ),
        "projected_identity_count_before_deduplication": int(len(grouped)),
        "retained_candidate_count": int(len(retained)),
        "deduplicated_candidate_count": int(
            len(projected_rows) - len(retained)
        ),
        "all_retained_execution_modes_grouped_exact": bool(
            all(
                str(row.get("recommended_execution_mode")) == "grouped_exact"
                for row in retained
            )
        ),
        "raw_term_order_and_coefficients_preserved_in_lineage": True,
        "projected_relative_coefficients_preserved": True,
        "global_scalar_normalized_before_identity_deduplication": True,
        "projection_events": projection_events,
        "zero_rejections": zero_rejections,
        "deduplication_events": deduplication_events,
    }
    return retained, telemetry


def _phase3_runtime_split_needs_proxy_child_set_scoring(
    *,
    selection_mode: str,
    parent_collapse_debug_enabled: bool,
) -> bool:
    mode_key = str(selection_mode).strip().lower()
    if bool(parent_collapse_debug_enabled):
        return True
    return bool(mode_key == "proxy_child_set_preselection")


def _phase3_runtime_split_archival_polynomial_term_count(polynomial: Any) -> int:
    try:
        return int(len(list(polynomial.return_polynomial())))
    except Exception:
        return 0


def _phase3_runtime_split_parent_eligible(
    *,
    split_mode: str,
    selection_mode: str,
    generator_metadata: Mapping[str, Any] | None,
    candidate_term: Any,
) -> bool:
    if str(split_mode).strip().lower() != "shortlist_pauli_children_v1":
        return False
    if isinstance(generator_metadata, Mapping) and bool(
        generator_metadata.get("is_macro_generator", False)
    ):
        return True
    # Archived June-2026 Phase-III split rows were not consistently keyed by the
    # later macro-generator metadata flag. Keep that recovery behavior explicit
    # and opt-in so canonical/shared-pool runs remain governed by metadata.
    if str(selection_mode).strip().lower() != "archival_child_set_forward_v1":
        return False
    polynomial = getattr(candidate_term, "polynomial", candidate_term)
    return bool(_phase3_runtime_split_archival_polynomial_term_count(polynomial) > 1)


def _normalize_phase3_runtime_split_child_set_symmetry_policy(value: str | None) -> str:
    key = str(value or "parent").strip().lower().replace("-", "_")
    if key in {"", "source", "preserve"}:
        key = "parent"
    if key not in _PHASE3_RUNTIME_SPLIT_CHILD_SET_SYMMETRY_POLICIES:
        allowed = ", ".join(sorted(_PHASE3_RUNTIME_SPLIT_CHILD_SET_SYMMETRY_POLICIES))
        raise ValueError(
            "phase3_runtime_split_child_set_symmetry_policy must be one of "
            f"{{{allowed}}}; got {value!r}."
        )
    return key


def _phase3_runtime_split_child_set_symmetry_spec(
    parent_symmetry_spec: Mapping[str, Any] | None,
    *,
    policy: str,
    fallback_preserving: bool = False,
) -> dict[str, Any] | None:
    policy_key = _normalize_phase3_runtime_split_child_set_symmetry_policy(policy)
    if policy_key == "off":
        return None
    parent_has_symmetry_modes = bool(
        isinstance(parent_symmetry_spec, Mapping)
        and any(
            key in parent_symmetry_spec
            for key in ("particle_number_mode", "spin_sector_mode", "hard_guard")
        )
    )
    if policy_key == "parent" and isinstance(parent_symmetry_spec, Mapping) and parent_has_symmetry_modes:
        return dict(parent_symmetry_spec)
    if policy_key == "parent" and not bool(fallback_preserving):
        return dict(parent_symmetry_spec) if isinstance(parent_symmetry_spec, Mapping) else None
    out = dict(parent_symmetry_spec) if isinstance(parent_symmetry_spec, Mapping) else {}
    out["particle_number_mode"] = "preserving"
    out["spin_sector_mode"] = "preserving"
    out.setdefault("phonon_number_mode", "not_conserved")
    out["hard_guard"] = True
    out["runtime_split_child_set_symmetry_policy"] = (
        "parent_fallback_preserving" if policy_key == "parent" else "hard_guard"
    )
    raw_tags = out.get("tags", [])
    tags = (
        [str(tag) for tag in raw_tags]
        if isinstance(raw_tags, Sequence) and not isinstance(raw_tags, (str, bytes))
        else []
    )
    if "runtime_split_child_set_hard_guard" not in tags:
        tags.append("runtime_split_child_set_hard_guard")
    out["tags"] = tags
    return out


def build_global_child_records_for_parent(
    *,
    parent_label: str,
    parent_term: AnsatzTerm,
    parent_family_id: str,
    parent_generator_metadata: Mapping[str, Any] | None,
    parent_symmetry_spec: Mapping[str, Any] | None,
    child_set_symmetry_policy: str,
    subset_sizes: Sequence[int],
    num_sites: int,
    ordering: str,
    qpb: int,
    problem_key: str,
    fixed_num_particles: Sequence[int] | None,
    evaluate_candidate: Callable[..., Mapping[str, Any]],
    child_padding_config: RouteAChildPaddingConfig | None = None,
    defer_phase1_evaluation: bool = False,
    base_record: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Expand one retained Phase-II macro into child stage-one records."""

    symmetry_spec = _phase3_runtime_split_child_set_symmetry_spec(
        parent_symmetry_spec,
        policy=str(child_set_symmetry_policy),
        fallback_preserving=bool(str(problem_key) == "hh"),
    )
    children = build_runtime_split_children(
        parent_label=str(parent_label),
        polynomial=parent_term.polynomial,
        family_id=str(parent_family_id),
        num_sites=int(num_sites),
        ordering=str(ordering),
        qpb=int(max(1, qpb)),
        split_mode="shortlist_pauli_children_v1",
        parent_generator_metadata=(
            dict(parent_generator_metadata)
            if isinstance(parent_generator_metadata, Mapping)
            else None
        ),
        symmetry_spec=symmetry_spec,
        fixed_num_particles=fixed_num_particles,
    )
    child_sets = build_runtime_split_child_sets(
        parent_label=str(parent_label),
        family_id=str(parent_family_id),
        num_sites=int(num_sites),
        ordering=str(ordering),
        qpb=int(max(1, qpb)),
        split_mode="shortlist_pauli_children_v1",
        children=children,
        parent_generator_metadata=(
            dict(parent_generator_metadata)
            if isinstance(parent_generator_metadata, Mapping)
            else None
        ),
        symmetry_spec=symmetry_spec,
        fixed_num_particles=fixed_num_particles,
        subset_sizes=tuple(int(size) for size in subset_sizes),
    )

    records: list[dict[str, Any]] = []
    projection_requested = bool(
        child_padding_config is not None
        and str(child_padding_config.policy)
        in ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_POLICIES
    )
    projection_input_count = 0
    projection_output_count = 0
    projection_zero_count = 0
    projection_raw_term_count = 0
    projection_projected_term_count = 0
    rejected_child_atom_count = sum(
        1
        for child in children
        if isinstance(child.get("symmetry_gate"), Mapping)
        and not bool(child["symmetry_gate"].get("passed", True))
    )
    for child_set in child_sets:
        polynomial = child_set.get("candidate_polynomial")
        metadata = child_set.get("candidate_generator_metadata")
        if not isinstance(polynomial, PauliPolynomial) or not isinstance(metadata, Mapping):
            continue
        label = str(child_set.get("candidate_label", ""))
        child_indices = [int(value) for value in child_set.get("child_indices", [])]
        child_labels = [str(value) for value in child_set.get("child_labels", [])]
        child_generator_ids = [
            str(value) for value in child_set.get("child_generator_ids", [])
        ]
        projection_payload: dict[str, Any] = {
            "active": False,
            "reason": "projection_policy_inactive",
        }
        recommended_execution_mode = str(
            child_set.get("recommended_execution_mode")
            or "termwise_product"
        )
        if projection_requested:
            projection_input_count += 1
            projected_polynomial, projection_payload = (
                project_route_a_child_polynomial(
                    polynomial,
                    config=child_padding_config,
                )
            )
            projection_raw_term_count += int(
                projection_payload.get("raw_term_count", 0)
            )
            projection_projected_term_count += int(
                projection_payload.get("projected_term_count", 0)
            )
            if projected_polynomial is None:
                projection_zero_count += 1
                continue
            polynomial = projected_polynomial
            projection_output_count += 1
            label = f"{label}::legal_projected"
            raw_compile_metadata = metadata.get("compile_metadata")
            raw_runtime_split = (
                raw_compile_metadata.get("runtime_split")
                if isinstance(raw_compile_metadata, Mapping)
                else None
            )
            projected_serialized_terms = serialize_polynomial_terms_exyz(
                polynomial
            )
            projected_metadata = asdict(
                build_generator_metadata(
                    label=str(label),
                    polynomial=polynomial,
                    family_id=str(parent_family_id),
                    num_sites=int(num_sites),
                    ordering=str(ordering),
                    qpb=int(qpb),
                    split_policy="runtime_split_projected_child",
                    parent_generator_id=(
                        str(metadata.get("parent_generator_id"))
                        if metadata.get("parent_generator_id") is not None
                        else None
                    ),
                    symmetry_spec=(
                        dict(metadata.get("symmetry_spec", {}))
                        if isinstance(metadata.get("symmetry_spec"), Mapping)
                        else symmetry_spec
                    ),
                    fixed_num_particles=fixed_num_particles,
                    serialized_terms=projected_serialized_terms,
                )
            )
            projected_compile_metadata = dict(
                projected_metadata.get("compile_metadata", {})
            )
            if isinstance(raw_runtime_split, Mapping):
                projected_compile_metadata["runtime_split"] = dict(
                    raw_runtime_split
                )
            projected_compile_metadata[
                "route_a_child_padding_projection"
            ] = dict(projection_payload)
            projected_metadata["compile_metadata"] = (
                projected_compile_metadata
            )
            projected_metadata["route_a_child_padding_projection"] = dict(
                projection_payload
            )
            metadata = projected_metadata
            recommended_execution_mode = "grouped_exact"
        candidate_term = AnsatzTerm(
            label=label,
            polynomial=polynomial,
            execution_mode=str(recommended_execution_mode),
        )
        if bool(defer_phase1_evaluation):
            if base_record is None:
                raise ValueError(
                    "Deferred global-child Phase-1 evaluation requires base_record."
                )
            record = {
                **dict(base_record),
                "candidate_label": str(label),
                "candidate_term": candidate_term,
                "generator_metadata": dict(metadata),
                "phase1_active_score": float("-inf"),
                "simple_score": float("-inf"),
                "phase2_raw_score": float("-inf"),
                "route_a_child_phase1_evaluation_deferred": True,
            }
        else:
            record = dict(
                evaluate_candidate(
                    candidate_term=candidate_term,
                    candidate_label=label,
                    generator_metadata=dict(metadata),
                    symmetry_spec_candidate=(
                        dict(metadata.get("symmetry_spec", {}))
                        if isinstance(metadata.get("symmetry_spec"), Mapping)
                        else symmetry_spec
                    ),
                    runtime_split_mode_value="shortlist_pauli_children_v1",
                    runtime_split_parent_label_value=str(parent_label),
                    runtime_split_child_count_value=int(len(children)),
                    runtime_split_chosen_representation_value="child_set",
                    runtime_split_child_indices_value=child_indices,
                    runtime_split_child_labels_value=child_labels,
                    runtime_split_child_generator_ids_value=child_generator_ids,
                )
            )
        record.update(
            {
                "candidate_label": str(label),
                "runtime_split_mode": "shortlist_pauli_children_v1",
                "runtime_split_parent_label": str(parent_label),
                "runtime_split_child_count": int(len(children)),
                "runtime_split_chosen_representation": "child_set",
                "runtime_split_child_indices": child_indices,
                "runtime_split_child_labels": child_labels,
                "runtime_split_child_generator_ids": child_generator_ids,
                "generator_metadata": dict(metadata),
                "route_a_child_padding_projection": dict(
                    projection_payload
                ),
            }
        )
        records.append(record)

    return records, {
        "schema": "route_a_phase3_global_child_parent_expansion_v1",
        "parent_label": str(parent_label),
        "child_atom_count": int(len(children)),
        "rejected_child_atom_count_symmetry": int(rejected_child_atom_count),
        "admissible_child_set_count": int(len(child_sets)),
        "staged_child_set_count": int(len(records)),
        "phase1_evaluated_child_set_count": int(
            0 if defer_phase1_evaluation else len(records)
        ),
        # Compatibility alias for existing payload readers.
        "evaluated_child_set_count": int(
            0 if defer_phase1_evaluation else len(records)
        ),
        "subset_sizes": [int(size) for size in subset_sizes],
        "subset_size_semantics": "exact_allowed_pauli_word_cardinalities",
        "child_set_symmetry_policy": str(child_set_symmetry_policy),
        "child_symmetry_guard_semantics": (
            ROUTE_A_CHILD_SYMMETRY_GUARD_FIXED_SECTOR_V1
            if str(child_set_symmetry_policy) == "hard_guard"
            and fixed_num_particles is not None
            else "disabled_or_global_compatibility"
        ),
        "fixed_num_particles": (
            [int(value) for value in fixed_num_particles]
            if fixed_num_particles is not None
            else None
        ),
        "child_padding_projection_policy": (
            str(child_padding_config.policy)
            if child_padding_config is not None
            else None
        ),
        "child_padding_projection_requested": bool(projection_requested),
        "child_padding_projection_input_count": int(projection_input_count),
        "child_padding_projection_output_count": int(projection_output_count),
        "child_padding_projection_zero_count": int(projection_zero_count),
        "child_padding_projection_raw_term_count": int(
            projection_raw_term_count
        ),
        "child_padding_projection_projected_term_count": int(
            projection_projected_term_count
        ),
        "phase1_evaluation_deferred_until_after_global_deduplication": bool(
            defer_phase1_evaluation
        ),
    }
