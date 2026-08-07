"""Shared Pauli-child pool contract for Paper-I method comparisons.

This module is deliberately neutral: it does not know about SNAKE phases,
append-only selection, or Geo-ADAPT scoring.  Callers provide the same macro
parent pool and receive either the historical ordered parent-plus-child-set
pool or the explicitly projected children-only singleton pool, together with
hashes that report scripts can compare across methods.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from src.quantum.pauli_polynomial_class import PauliPolynomial

from pipelines.scaffold.hh_continuation_generators import (
    build_runtime_split_child_sets,
    build_runtime_split_children,
    normalize_runtime_split_subset_sizes,
    serialize_polynomial_terms_exyz,
)
from pipelines.static_adapt.route_a_child_padding import (
    ROUTE_A_CHILD_PADDING_FULL_BINARY_CODE_SPACE_V1,
    ROUTE_A_CHILD_PADDING_HARD_FILTER_V1,
    ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
    RouteAChildPaddingConfig,
    filter_route_a_child_padding_records,
)
from pipelines.static_adapt.route_a_shortlists import (
    CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1,
    deduplicate_child_position_records,
    pauli_child_identity,
)
from pipelines.static_adapt.runtime_split import (
    project_and_deduplicate_runtime_split_child_sets,
)
from src.quantum.vqe_latex_python_pairs import AnsatzTerm

SHARED_PAULI_POOL_CONTRACT_ID = "paper_i_shared_pauli_child_pool_v1"
SHARED_PAULI_POOL_MODE_OFF = "off"
SHARED_PAULI_POOL_MODE_CHILD_SETS_V1 = "shared_pauli_child_sets_v1"
SHARED_PAULI_POOL_MODE_PROJECTED_SINGLETON_CHILDREN_ONLY_V1 = (
    "projected_singleton_children_only_v1"
)
SHARED_PAULI_POOL_MODE_GUARDED_SINGLETON_CHILDREN_ONLY_V1 = (
    "guarded_singleton_children_only_v1"
)
SHARED_PAULI_POOL_MODE_ALIASES = {
    SHARED_PAULI_POOL_MODE_OFF: SHARED_PAULI_POOL_MODE_OFF,
    "none": SHARED_PAULI_POOL_MODE_OFF,
    "false": SHARED_PAULI_POOL_MODE_OFF,
    "0": SHARED_PAULI_POOL_MODE_OFF,
    "disabled": SHARED_PAULI_POOL_MODE_OFF,
    "pauli_child_sets_v1": SHARED_PAULI_POOL_MODE_CHILD_SETS_V1,
    "global_pauli_child_sets_v1": SHARED_PAULI_POOL_MODE_CHILD_SETS_V1,
    SHARED_PAULI_POOL_MODE_CHILD_SETS_V1: SHARED_PAULI_POOL_MODE_CHILD_SETS_V1,
    SHARED_PAULI_POOL_MODE_PROJECTED_SINGLETON_CHILDREN_ONLY_V1: (
        SHARED_PAULI_POOL_MODE_PROJECTED_SINGLETON_CHILDREN_ONLY_V1
    ),
    SHARED_PAULI_POOL_MODE_GUARDED_SINGLETON_CHILDREN_ONLY_V1: (
        SHARED_PAULI_POOL_MODE_GUARDED_SINGLETON_CHILDREN_ONLY_V1
    ),
}
SHARED_PAULI_POOL_SYMMETRY_POLICY_OFF = "off"
SHARED_PAULI_POOL_SYMMETRY_POLICY_HARD_GUARD = "hard_guard"


@dataclass(frozen=True)
class SharedPauliPoolParent:
    label: str
    polynomial: PauliPolynomial
    family_id: str = "unknown"
    stage_family: str = "shared"
    construction: str = "parent"
    execution_mode: str = "termwise_product"
    symmetry_spec: dict[str, Any] | None = None
    generator_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class SharedPauliPoolCandidate:
    label: str
    polynomial: PauliPolynomial
    family_id: str
    stage_family: str
    construction: str
    execution_mode: str
    representation: str
    parent_label: str | None
    child_indices: tuple[int, ...]
    child_labels: tuple[str, ...]
    symmetry_spec: dict[str, Any] | None
    symmetry_gate: dict[str, Any] | None
    generator_metadata: dict[str, Any]
    serialized_terms_exyz: tuple[dict[str, Any], ...]
    parent_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class SharedPauliPoolResult:
    candidates: tuple[SharedPauliPoolCandidate, ...]
    meta: dict[str, Any]
    manifest: dict[str, Any]


def normalize_shared_pauli_pool_mode(value: str | None) -> str:
    key = str(value or SHARED_PAULI_POOL_MODE_OFF).strip().lower().replace("-", "_")
    if key == "":
        key = SHARED_PAULI_POOL_MODE_OFF
    if key not in SHARED_PAULI_POOL_MODE_ALIASES:
        allowed = ", ".join(sorted(SHARED_PAULI_POOL_MODE_ALIASES))
        raise ValueError(f"shared_pauli_pool_mode must be one of {{{allowed}}}; got {value!r}.")
    return SHARED_PAULI_POOL_MODE_ALIASES[key]


def normalize_shared_pauli_pool_symmetry_policy(value: str | None) -> str:
    key = str(value or SHARED_PAULI_POOL_SYMMETRY_POLICY_OFF).strip().lower().replace("-", "_")
    if key in {"", "none", "false", "0", "disabled"}:
        key = SHARED_PAULI_POOL_SYMMETRY_POLICY_OFF
    if key not in {SHARED_PAULI_POOL_SYMMETRY_POLICY_OFF, SHARED_PAULI_POOL_SYMMETRY_POLICY_HARD_GUARD}:
        raise ValueError("shared_pauli_pool_symmetry_policy must be one of {'off','hard_guard'}.")
    return key


def _child_symmetry_spec(policy: str) -> dict[str, Any]:
    hard_guard = bool(policy == SHARED_PAULI_POOL_SYMMETRY_POLICY_HARD_GUARD)
    return {
        "policy": str(policy),
        "hard_guard": hard_guard,
        "particle_number_mode": "preserving" if hard_guard else "off",
        "spin_sector_mode": "preserving" if hard_guard else "off",
        "source": SHARED_PAULI_POOL_CONTRACT_ID,
    }


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _serialized_terms(polynomial: PauliPolynomial) -> tuple[dict[str, Any], ...]:
    return tuple(dict(row) for row in serialize_polynomial_terms_exyz(polynomial))


def _parent_metadata(parent: SharedPauliPoolParent) -> dict[str, Any]:
    meta = dict(parent.generator_metadata or {})
    meta.setdefault("label", str(parent.label))
    meta.setdefault("candidate_label", str(parent.label))
    meta.setdefault("family_id", str(parent.family_id))
    compile_meta = dict(meta.get("compile_metadata", {})) if isinstance(meta.get("compile_metadata"), Mapping) else {}
    serialized = [dict(row) for row in _serialized_terms(parent.polynomial)]
    compile_meta.setdefault("serialized_terms_exyz", serialized)
    compile_meta.setdefault("num_polynomial_terms", int(len(serialized)))
    compile_meta.setdefault("signature_size", int(len(serialized)))
    meta["compile_metadata"] = compile_meta
    meta.setdefault("is_macro_generator", bool(len(serialized) > 1))
    return meta


def _symmetry_spec_from_metadata(
    meta: Mapping[str, Any],
    fallback: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    spec = meta.get("symmetry_spec")
    if isinstance(spec, Mapping):
        return dict(spec)
    if isinstance(fallback, Mapping):
        return dict(fallback)
    return None


def _candidate_fingerprint(candidate: SharedPauliPoolCandidate) -> dict[str, Any]:
    payload = {
        "label": str(candidate.label),
        "family_id": str(candidate.family_id),
        "stage_family": str(candidate.stage_family),
        "construction": str(candidate.construction),
        "execution_mode": str(candidate.execution_mode),
        "representation": str(candidate.representation),
        "parent_label": candidate.parent_label,
        "child_indices": [int(idx) for idx in candidate.child_indices],
        "child_labels": [str(label) for label in candidate.child_labels],
        "serialized_terms_exyz": [dict(row) for row in candidate.serialized_terms_exyz],
    }
    if candidate.parent_labels:
        payload["parent_labels"] = [str(label) for label in candidate.parent_labels]
    return payload


def _parent_fingerprint(parent: SharedPauliPoolParent) -> dict[str, Any]:
    return {
        "label": str(parent.label),
        "family_id": str(parent.family_id),
        "stage_family": str(parent.stage_family),
        "construction": str(parent.construction),
        "execution_mode": str(parent.execution_mode),
        "serialized_terms_exyz": [dict(row) for row in _serialized_terms(parent.polynomial)],
    }


def _project_singleton_children(
    *,
    children: Sequence[Mapping[str, Any]],
    problem_key: str,
    num_sites: int,
    ordering: str,
    qpb: int,
    n_ph_max: int | None,
    boson_encoding: str | None,
    total_register_width: int | None,
    fixed_num_particles: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the canonical exact padding projection to raw singleton atoms.

    ``subset_size=1`` describes the one raw child direction.  The exact legal
    projection may expand that direction into a grouped multi-Pauli
    polynomial; it remains one logical singleton candidate.
    """

    raw_rows: list[dict[str, Any]] = []
    for child in children:
        gate = child.get("symmetry_gate")
        if not isinstance(gate, Mapping) or not bool(gate.get("checked", False)):
            raise ValueError(
                "projected singleton child lacks a checked symmetry receipt."
            )
        if not bool(gate.get("passed", False)):
            continue
        polynomial = child.get("child_polynomial")
        metadata = child.get("child_generator_metadata")
        if not isinstance(polynomial, PauliPolynomial) or not isinstance(metadata, Mapping):
            raise ValueError(
                "projected singleton child lacks polynomial/metadata payloads."
            )
        label = str(child.get("child_label") or "")
        if not label:
            raise ValueError("projected singleton child lacks a stable label.")
        child_index = int(child.get("child_index", 0))
        raw_rows.append(
            {
                "candidate_label": label,
                "candidate_polynomial": polynomial,
                "candidate_generator_metadata": dict(metadata),
                "recommended_execution_mode": "termwise_product",
                "child_indices": [child_index],
                "child_labels": [label],
                "symmetry_gate": dict(gate),
            }
        )

    problem = str(problem_key).strip().lower()
    if problem == "hubbard":
        return raw_rows, {
            "schema": "projected_singleton_children_padding_v1",
            "policy": "no_boson_register",
            "projection_input_count": int(len(raw_rows)),
            "retained_candidate_count": int(len(raw_rows)),
            "projection_zero_rejection_count": 0,
            "deduplicated_candidate_count": 0,
        }
    if problem != "hh":
        raise ValueError(
            "projected_singleton_children_only_v1 supports only HH/Hubbard; "
            f"got {problem!r}."
        )
    if n_ph_max is None or boson_encoding in {None, ""} or total_register_width is None:
        raise ValueError(
            "projected_singleton_children_only_v1 requires n_ph_max, "
            "boson_encoding, and total_register_width for exact padding projection."
        )
    config = RouteAChildPaddingConfig(
        policy=ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
        problem_key=problem,
        num_sites=int(num_sites),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        total_register_width=int(total_register_width),
    )
    return project_and_deduplicate_runtime_split_child_sets(
        raw_rows,
        config=config,
        num_sites=int(num_sites),
        ordering=str(ordering),
        qpb=int(qpb),
        fixed_num_particles=tuple(int(value) for value in fixed_num_particles),
    )


def _manifest_from_candidates(
    *,
    candidates: Sequence[SharedPauliPoolCandidate],
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [_candidate_fingerprint(candidate) for candidate in candidates]
    label_rows = [str(candidate.label) for candidate in candidates]
    contract_identity = {
        "contract_id": SHARED_PAULI_POOL_CONTRACT_ID,
        "schema": str(meta.get("schema", "shared_pauli_child_pool_contract_v1")),
        "mode": str(meta.get("mode", SHARED_PAULI_POOL_MODE_OFF)),
        "symmetry_policy": str(meta.get("symmetry_policy", SHARED_PAULI_POOL_SYMMETRY_POLICY_OFF)),
        "subset_sizes": [int(size) for size in meta.get("subset_sizes", [])],
        "max_subset_size": int(meta.get("max_subset_size", 0)),
        "pool_policy": str(meta.get("pool_policy", "")),
    }
    if meta.get("candidate_representation_policy") is not None:
        contract_identity.update(
            {
                "candidate_representation_policy": str(
                    meta["candidate_representation_policy"]
                ),
                "padding_policy": str(meta.get("padding_policy", "")),
                "source_parent_ordered_pool_hash": str(
                    meta.get("source_parent_ordered_pool_hash", "")
                ),
            }
        )
    return {
        "contract_id": SHARED_PAULI_POOL_CONTRACT_ID,
        "schema": "shared_pauli_pool_manifest_v1",
        "mode": str(meta.get("mode", SHARED_PAULI_POOL_MODE_OFF)),
        "enabled": bool(meta.get("enabled", False)),
        "symmetry_policy": str(meta.get("symmetry_policy", SHARED_PAULI_POOL_SYMMETRY_POLICY_OFF)),
        "symmetry_gate_enforced": bool(meta.get("symmetry_gate_enforced", False)),
        "explicit_no_guard": bool(meta.get("explicit_no_guard", False)),
        "subset_sizes": [int(size) for size in meta.get("subset_sizes", [])],
        "subset_size_semantics": str(meta.get("subset_size_semantics", "")),
        "max_subset_size": int(meta.get("max_subset_size", 0)),
        "ordered_candidate_count": int(len(rows)),
        "ordered_label_hash": _json_digest(label_rows),
        "ordered_pool_hash": _json_digest({"contract_identity": contract_identity, "candidates": rows}),
        "base_pool_term_count": int(meta.get("base_pool_term_count", 0)),
        "expanded_pool_term_count": int(meta.get("expanded_pool_term_count", len(rows))),
        "candidate_fingerprint_schema": "label_family_stage_representation_parent_children_serialized_terms_v1",
        "contract_identity": contract_identity,
    }


def _base_meta(
    *,
    mode: str,
    symmetry_policy: str,
    subset_sizes: Sequence[int],
    base_count: int,
    explicit_no_guard: bool = False,
) -> dict[str, Any]:
    enabled = bool(mode != SHARED_PAULI_POOL_MODE_OFF)
    symmetry_gate_enforced = bool(enabled and symmetry_policy == SHARED_PAULI_POOL_SYMMETRY_POLICY_HARD_GUARD)
    subset_sizes_tuple = tuple(int(size) for size in subset_sizes)
    meta = {
        "contract_id": SHARED_PAULI_POOL_CONTRACT_ID,
        "schema": "shared_pauli_child_pool_contract_v1",
        "enabled": enabled,
        "mode": str(mode),
        "symmetry_policy": str(symmetry_policy),
        "symmetry_gate_enforced": symmetry_gate_enforced,
        "explicit_no_guard": bool(enabled and explicit_no_guard),
        "subset_sizes": [int(size) for size in subset_sizes_tuple],
        "subset_size_semantics": "exact_allowed_pauli_word_cardinalities",
        "max_subset_size": int(max(subset_sizes_tuple)),
        "source": "paper_i_shared_parent_plus_pauli_child_sets",
        "base_pool_term_count": int(base_count),
        "expanded_pool_term_count": int(base_count),
        "split_parent_count": 0,
        "child_atom_count": 0,
        "child_set_candidate_count": 0,
        "added_child_set_count": 0,
        "duplicate_child_set_count": 0,
        "symmetry_checked_child_atom_count": 0,
        "symmetry_rejected_child_atom_count": 0,
        "symmetry_checked_child_set_count": 0,
        "symmetry_rejected_child_set_count": 0,
        "pool_policy": "same_ordered_parent_plus_child_set_pool_for_snake_geo_append",
        "parent_duplicate_child_set_policy": "exclude_child_set_identical_to_parent_macro",
    }
    if mode == SHARED_PAULI_POOL_MODE_PROJECTED_SINGLETON_CHILDREN_ONLY_V1:
        meta.update(
            {
                "source": "paper_i_unfiltered_full_meta_projected_singleton_children",
                "pool_policy": "projected_singleton_children_only_from_same_ordered_parent_pool",
                "candidate_representation_policy": "projected_singleton_child_only_v1",
                "padding_policy": ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_V1,
                "parent_candidate_count": 0,
                "projected_singleton_source_term_count": 0,
                "projected_singleton_candidate_count": 0,
                "projected_singleton_symmetry_rejected_count": 0,
                "projected_singleton_projection_input_count": 0,
                "projected_singleton_projection_zero_count": 0,
                "projected_singleton_projection_zero_exclusions": [],
                "projected_singleton_projection_deduplicated_count": 0,
                "projected_singleton_grouped_term_count": 0,
                "projected_singleton_null_count": 0,
                "projected_singleton_null_identity_count": 0,
                "projected_singleton_null_exclusions": [],
            }
        )
    if mode == SHARED_PAULI_POOL_MODE_GUARDED_SINGLETON_CHILDREN_ONLY_V1:
        meta.update(
            {
                "source": "paper_i_unfiltered_full_meta_guarded_raw_singleton_children",
                "pool_policy": "guarded_raw_singleton_children_only_from_same_ordered_parent_pool",
                "candidate_representation_policy": "guarded_raw_singleton_child_only_v1",
                "padding_policy": ROUTE_A_CHILD_PADDING_HARD_FILTER_V1,
                "parent_candidate_count": 0,
                "guarded_singleton_source_term_count": 0,
                "guarded_singleton_symmetry_rejected_count": 0,
                "guarded_singleton_global_duplicate_count": 0,
                "guarded_singleton_pre_padding_identity_count": 0,
                "guarded_singleton_padding_rejected_count": 0,
                "guarded_singleton_null_identity_count": 0,
                "guarded_singleton_null_exclusions": [],
                "guarded_singleton_candidate_count": 0,
                "guarded_singleton_projection_applied": False,
                "guarded_singleton_identity_policy": (
                    CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1
                ),
            }
        )
    return meta


def build_shared_pauli_child_pool(
    *,
    parents: Sequence[SharedPauliPoolParent],
    mode: str | None,
    symmetry_policy: str | None,
    max_subset_size: int | str | None = None,
    subset_sizes: Sequence[int] | str | int | None = None,
    problem_key: str,
    num_sites: int,
    ordering: str,
    qpb: int,
    n_ph_max: int | None = None,
    boson_encoding: str | None = None,
    total_register_width: int | None = None,
    fixed_num_particles: Sequence[int] | None = None,
    max_terms: int | None = None,
) -> SharedPauliPoolResult:
    """Build the canonical ordered pool for the shared Paper-I child-pool route."""
    mode_key = normalize_shared_pauli_pool_mode(mode)
    policy_key = normalize_shared_pauli_pool_symmetry_policy(symmetry_policy)
    raw_symmetry_policy = str(symmetry_policy).strip().lower().replace("-", "_") if symmetry_policy is not None else ""
    explicit_no_guard = bool(
        raw_symmetry_policy not in {"", "none", "false", "0", "disabled"}
        and policy_key == SHARED_PAULI_POOL_SYMMETRY_POLICY_OFF
    )
    legacy_max_subset_size = (
        None if max_subset_size in {None, ""} else int(max_subset_size)
    )
    requested_subset_sizes = normalize_runtime_split_subset_sizes(
        subset_sizes,
        legacy_max_subset_size=legacy_max_subset_size,
    )
    parent_rows = tuple(parents)
    projected_children_only = bool(
        mode_key == SHARED_PAULI_POOL_MODE_PROJECTED_SINGLETON_CHILDREN_ONLY_V1
    )
    guarded_children_only = bool(
        mode_key == SHARED_PAULI_POOL_MODE_GUARDED_SINGLETON_CHILDREN_ONLY_V1
    )
    children_only = bool(projected_children_only or guarded_children_only)
    if children_only:
        if tuple(int(size) for size in requested_subset_sizes) != (1,):
            raise ValueError(
                f"{mode_key} requires exact subset size 1."
            )
        if policy_key != SHARED_PAULI_POOL_SYMMETRY_POLICY_HARD_GUARD:
            raise ValueError(
                f"{mode_key} requires symmetry_policy=hard_guard."
            )
        if not parent_rows:
            raise ValueError(
                f"{mode_key} requires a nonempty parent pool."
            )
        if fixed_num_particles is None:
            raise ValueError(
                f"{mode_key} requires fixed_num_particles "
                "for fail-closed symmetry certification."
            )
        problem_key_normalized = str(problem_key).strip().lower()
        guarded_singleton_families = {
            "hh",
            "molecular_vibronic_h2o_linear_fd",
        }
        if guarded_children_only and problem_key_normalized not in guarded_singleton_families:
            raise ValueError(
                "guarded_singleton_children_only_v1 requires an explicitly "
                "supported fixed-sector fermion-boson family."
            )
        if guarded_children_only and (
            n_ph_max is None
            or boson_encoding in {None, ""}
            or total_register_width is None
        ):
            raise ValueError(
                "guarded_singleton_children_only_v1 requires n_ph_max, "
                "boson_encoding, and total_register_width for the legal-codeword guard."
            )
    meta = _base_meta(
        mode=mode_key,
        symmetry_policy=policy_key,
        subset_sizes=requested_subset_sizes,
        base_count=int(len(parent_rows)),
        explicit_no_guard=explicit_no_guard,
    )
    if children_only:
        parent_fingerprints = [_parent_fingerprint(parent) for parent in parent_rows]
        meta["source_parent_ordered_label_hash"] = _json_digest(
            [str(parent.label) for parent in parent_rows]
        )
        meta["source_parent_ordered_pool_hash"] = _json_digest(parent_fingerprints)
    if mode_key != SHARED_PAULI_POOL_MODE_OFF:
        if str(problem_key).strip().lower() not in {
            "hh",
            "hubbard",
            "molecular_vibronic_h2o_linear_fd",
        }:
            raise ValueError(
                "shared_pauli_pool_mode is not registered for this problem."
            )

    candidates: list[SharedPauliPoolCandidate] = []
    guarded_raw_records: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    child_symmetry_spec = _child_symmetry_spec(policy_key)

    for parent in parent_rows:
        parent_meta = _parent_metadata(parent)
        parent_serialized = _serialized_terms(parent.polynomial)
        parent_candidate = SharedPauliPoolCandidate(
            label=str(parent.label),
            polynomial=parent.polynomial,
            family_id=str(parent.family_id),
            stage_family=str(parent.stage_family),
            construction=str(parent.construction),
            execution_mode=str(parent.execution_mode),
            representation="parent",
            parent_label=None,
            child_indices=(),
            child_labels=(),
            symmetry_spec=_symmetry_spec_from_metadata(parent_meta, parent.symmetry_spec),
            symmetry_gate=None,
            generator_metadata=dict(parent_meta),
            serialized_terms_exyz=parent_serialized,
        )
        if not children_only:
            candidates.append(parent_candidate)
            seen_labels.add(str(parent_candidate.label))
        if mode_key == SHARED_PAULI_POOL_MODE_OFF or len(parent_serialized) <= 1:
            if not children_only:
                continue

        children = build_runtime_split_children(
            parent_label=str(parent.label),
            polynomial=parent.polynomial,
            family_id=str(parent.family_id),
            num_sites=int(num_sites),
            ordering=str(ordering),
            qpb=int(max(1, qpb)),
            split_mode=mode_key,
            parent_generator_metadata=parent_meta,
            symmetry_spec=child_symmetry_spec,
            fixed_num_particles=(fixed_num_particles if children_only else None),
            hard_guard_required=children_only,
            include_unsplit_singleton=children_only,
        )
        if not children:
            continue
        meta["split_parent_count"] = int(meta["split_parent_count"]) + 1
        meta["child_atom_count"] = int(meta["child_atom_count"]) + int(len(children))
        for child in children:
            gate = child.get("symmetry_gate")
            if isinstance(gate, Mapping) and bool(gate.get("checked", False)):
                meta["symmetry_checked_child_atom_count"] = int(meta["symmetry_checked_child_atom_count"]) + 1
                if not bool(gate.get("passed", True)):
                    meta["symmetry_rejected_child_atom_count"] = int(meta["symmetry_rejected_child_atom_count"]) + 1

        if projected_children_only:
            meta["projected_singleton_source_term_count"] = int(
                meta["projected_singleton_source_term_count"]
            ) + int(len(children))
            symmetry_rejected = sum(
                1
                for child in children
                if not (
                    isinstance(child.get("symmetry_gate"), Mapping)
                    and bool(child["symmetry_gate"].get("checked", False))
                    and bool(child["symmetry_gate"].get("passed", False))
                )
            )
            meta["projected_singleton_symmetry_rejected_count"] = int(
                meta["projected_singleton_symmetry_rejected_count"]
            ) + int(symmetry_rejected)
            projected_rows, projection_meta = _project_singleton_children(
                children=children,
                problem_key=str(problem_key),
                num_sites=int(num_sites),
                ordering=str(ordering),
                qpb=int(max(1, qpb)),
                n_ph_max=n_ph_max,
                boson_encoding=boson_encoding,
                total_register_width=total_register_width,
                fixed_num_particles=tuple(int(value) for value in fixed_num_particles),
            )
            meta["projected_singleton_projection_input_count"] = int(
                meta["projected_singleton_projection_input_count"]
            ) + int(projection_meta.get("projection_input_count", 0))
            projection_zero_count = int(
                projection_meta.get("projection_zero_rejection_count", 0)
            )
            raw_zero_exclusions = projection_meta.get("zero_rejections", [])
            if not isinstance(raw_zero_exclusions, list) or any(
                not isinstance(row, Mapping) for row in raw_zero_exclusions
            ):
                raise ValueError(
                    "projected singleton padding telemetry lacks normalized "
                    "zero-rejection lineage."
                )
            if len(raw_zero_exclusions) != projection_zero_count:
                raise ValueError(
                    "projected singleton padding zero-rejection count/lineage mismatch."
                )
            meta["projected_singleton_projection_zero_count"] = int(
                meta["projected_singleton_projection_zero_count"]
            ) + projection_zero_count
            if raw_zero_exclusions:
                normalized_zero_exclusions = [
                    {
                        "schema": "projected_singleton_null_child_exclusion_v1",
                        "null_kind": "exact_projection_zero",
                        **dict(row),
                    }
                    for row in raw_zero_exclusions
                ]
                projection_exclusions = list(
                    meta["projected_singleton_projection_zero_exclusions"]
                )
                projection_exclusions.extend(normalized_zero_exclusions)
                meta["projected_singleton_projection_zero_exclusions"] = (
                    projection_exclusions
                )
                null_exclusions = list(meta["projected_singleton_null_exclusions"])
                null_exclusions.extend(normalized_zero_exclusions)
                meta["projected_singleton_null_exclusions"] = null_exclusions
            meta["projected_singleton_projection_deduplicated_count"] = int(
                meta["projected_singleton_projection_deduplicated_count"]
            ) + int(projection_meta.get("deduplicated_candidate_count", 0))
            for projected in projected_rows:
                label = str(projected.get("candidate_label") or "")
                if not label:
                    raise ValueError("projected singleton child lacks a stable label.")
                if label in seen_labels:
                    raise ValueError(
                        "projected singleton child label collision: " f"{label!r}."
                    )
                raw_meta = projected.get("candidate_generator_metadata")
                if not isinstance(raw_meta, Mapping):
                    raise ValueError(
                        "projected singleton child lacks projected metadata."
                    )
                child_meta = dict(raw_meta)
                child_indices = tuple(
                    int(value) for value in projected.get("child_indices", ())
                )
                child_labels = tuple(
                    str(value) for value in projected.get("child_labels", ())
                )
                if len(child_indices) != 1 or len(child_labels) != 1:
                    raise ValueError(
                        "projected singleton candidate lost its one-child lineage."
                    )
                projection_receipt = (
                    dict(projected.get("route_a_child_padding_projection", {}))
                    if isinstance(projected.get("route_a_child_padding_projection"), Mapping)
                    else {
                        "schema": "route_a_child_padding_exact_projection_v1",
                        "active": False,
                        "policy": "no_boson_register",
                        "reason": "no_boson_register",
                    }
                )
                child_meta["shared_pauli_pool_contract"] = {
                    "contract_id": SHARED_PAULI_POOL_CONTRACT_ID,
                    "mode": str(mode_key),
                    "symmetry_policy": str(policy_key),
                    "symmetry_gate_enforced": True,
                    "subset_sizes": [1],
                    "subset_size_semantics": "exact_allowed_pauli_word_cardinalities",
                    "max_subset_size": 1,
                    "parent_label": str(parent.label),
                    "representation": "projected_singleton_child",
                    "padding_projection": dict(projection_receipt),
                }
                polynomial = projected.get("candidate_polynomial")
                if not isinstance(polynomial, PauliPolynomial):
                    raise ValueError(
                        "projected singleton child lacks projected polynomial."
                    )
                projected_terms = _serialized_terms(polynomial)
                if projected_terms and all(
                    set(str(term.get("pauli_exyz", "")).lower()) <= {"e"}
                    for term in projected_terms
                ):
                    meta["projected_singleton_null_identity_count"] = int(
                        meta["projected_singleton_null_identity_count"]
                    ) + 1
                    exclusions = list(meta["projected_singleton_null_exclusions"])
                    exclusions.append(
                        {
                            "schema": "projected_singleton_null_child_exclusion_v1",
                            "null_kind": "identity_global_phase",
                            "reason": "exact_projection_is_identity_global_phase_direction",
                            "parent_label": str(parent.label),
                            "candidate_label": label,
                            "child_indices": [int(value) for value in child_indices],
                            "child_labels": [str(value) for value in child_labels],
                            "serialized_terms_exyz": [dict(row) for row in projected_terms],
                            "padding_projection": dict(projection_receipt),
                        }
                    )
                    meta["projected_singleton_null_exclusions"] = exclusions
                    continue
                raw_gate = projected.get("symmetry_gate")
                gate = dict(raw_gate) if isinstance(raw_gate, Mapping) else None
                candidate = SharedPauliPoolCandidate(
                    label=label,
                    polynomial=polynomial,
                    family_id=str(parent.family_id),
                    stage_family=str(parent.stage_family),
                    construction=f"{parent.construction}::projected_singleton_child",
                    execution_mode=str(
                        projected.get("recommended_execution_mode")
                        or ("grouped_exact" if str(problem_key).strip().lower() == "hh" else "termwise_product")
                    ),
                    representation="projected_singleton_child",
                    parent_label=str(parent.label),
                    child_indices=child_indices,
                    child_labels=child_labels,
                    symmetry_spec=_symmetry_spec_from_metadata(
                        child_meta, parent.symmetry_spec
                    ),
                    symmetry_gate=gate,
                    generator_metadata=child_meta,
                    serialized_terms_exyz=projected_terms,
                )
                if not candidate.serialized_terms_exyz:
                    raise ValueError("projected singleton candidate is zero.")
                candidates.append(candidate)
                seen_labels.add(label)
                meta["projected_singleton_candidate_count"] = int(
                    meta["projected_singleton_candidate_count"]
                ) + 1
                meta["projected_singleton_grouped_term_count"] = int(
                    meta["projected_singleton_grouped_term_count"]
                ) + int(len(candidate.serialized_terms_exyz))
                if max_terms is not None and len(candidates) > int(max_terms):
                    raise ValueError(
                        f"shared Pauli-child pool exceeds cap: {len(candidates)} > {int(max_terms)}"
                    )
            continue

        if guarded_children_only:
            meta["guarded_singleton_source_term_count"] = int(
                meta["guarded_singleton_source_term_count"]
            ) + int(len(children))
            for child in children:
                gate = child.get("symmetry_gate")
                if not isinstance(gate, Mapping) or not bool(
                    gate.get("checked", False)
                ):
                    raise ValueError(
                        "guarded singleton child lacks a checked symmetry receipt."
                    )
                if not bool(gate.get("passed", False)):
                    meta["guarded_singleton_symmetry_rejected_count"] = int(
                        meta["guarded_singleton_symmetry_rejected_count"]
                    ) + 1
                    continue
                polynomial = child.get("child_polynomial")
                raw_meta = child.get("child_generator_metadata")
                child_label = str(child.get("child_label") or "")
                if (
                    not isinstance(polynomial, PauliPolynomial)
                    or not isinstance(raw_meta, Mapping)
                    or not child_label
                ):
                    raise ValueError(
                        "guarded singleton child lacks polynomial, metadata, or label."
                    )
                candidate_term = AnsatzTerm(
                    label=child_label,
                    polynomial=polynomial,
                    execution_mode="termwise_product",
                )
                guarded_raw_records.append(
                    {
                        "candidate_label": child_label,
                        "candidate_term": candidate_term,
                        "candidate_generator_metadata": dict(raw_meta),
                        "candidate_family_id": str(parent.family_id),
                        "candidate_stage_family": str(parent.stage_family),
                        "candidate_construction": str(parent.construction),
                        "runtime_split_parent_label": str(parent.label),
                        "runtime_split_child_index": int(
                            child.get("child_index", 0)
                        ),
                        "runtime_split_child_label": child_label,
                        "route_a_child_parent_labels": [str(parent.label)],
                        "symmetry_gate": dict(gate),
                        "position_id": 0,
                        "_shared_pool_rank_score": 0.0,
                    }
                )
            continue

        child_sets = build_runtime_split_child_sets(
            parent_label=str(parent.label),
            family_id=str(parent.family_id),
            num_sites=int(num_sites),
            ordering=str(ordering),
            qpb=int(max(1, qpb)),
            split_mode=mode_key,
            children=children,
            parent_generator_metadata=parent_meta,
            symmetry_spec=child_symmetry_spec,
            subset_sizes=requested_subset_sizes,
        )
        meta["child_set_candidate_count"] = int(meta["child_set_candidate_count"]) + int(len(child_sets))
        for child_set in child_sets:
            label = str(child_set.get("candidate_label", ""))
            if not label:
                continue
            if label in seen_labels:
                meta["duplicate_child_set_count"] = int(meta["duplicate_child_set_count"]) + 1
                continue
            raw_gate = child_set.get("symmetry_gate")
            gate = dict(raw_gate) if isinstance(raw_gate, Mapping) else None
            if gate is not None and bool(gate.get("checked", False)):
                meta["symmetry_checked_child_set_count"] = int(meta["symmetry_checked_child_set_count"]) + 1
                if not bool(gate.get("passed", True)):
                    meta["symmetry_rejected_child_set_count"] = int(meta["symmetry_rejected_child_set_count"]) + 1
                    continue
            raw_meta = child_set.get("candidate_generator_metadata")
            child_meta = dict(raw_meta) if isinstance(raw_meta, Mapping) else {}
            child_meta["shared_pauli_pool_contract"] = {
                "contract_id": SHARED_PAULI_POOL_CONTRACT_ID,
                "mode": str(mode_key),
                "symmetry_policy": str(policy_key),
                "symmetry_gate_enforced": bool(meta.get("symmetry_gate_enforced", False)),
                "explicit_no_guard": bool(meta.get("explicit_no_guard", False)),
                "subset_sizes": [int(size) for size in requested_subset_sizes],
                "subset_size_semantics": "exact_allowed_pauli_word_cardinalities",
                "max_subset_size": int(max(requested_subset_sizes)),
                "parent_label": str(parent.label),
                "representation": "child_set",
            }
            child_indices = tuple(int(idx) for idx in (child_set.get("child_indices", ()) or ()))
            child_labels = tuple(str(item) for item in (child_set.get("child_labels", ()) or ()))
            polynomial = child_set.get("candidate_polynomial")
            if not isinstance(polynomial, PauliPolynomial):
                continue
            candidate = SharedPauliPoolCandidate(
                label=label,
                polynomial=polynomial,
                family_id=str(parent.family_id),
                stage_family=str(parent.stage_family),
                construction=f"{parent.construction}::shared_pauli_child_set",
                execution_mode=str(child_set.get("recommended_execution_mode") or parent.execution_mode),
                representation="child_set",
                parent_label=str(parent.label),
                child_indices=child_indices,
                child_labels=child_labels,
                symmetry_spec=_symmetry_spec_from_metadata(child_meta, parent.symmetry_spec),
                symmetry_gate=gate,
                generator_metadata=child_meta,
                serialized_terms_exyz=_serialized_terms(polynomial),
            )
            candidates.append(candidate)
            seen_labels.add(str(candidate.label))
            meta["added_child_set_count"] = int(meta["added_child_set_count"]) + 1
            if max_terms is not None and len(candidates) > int(max_terms):
                raise ValueError(f"shared Pauli-child pool exceeds cap: {len(candidates)} > {int(max_terms)}")

    if guarded_children_only:
        lineage_by_identity: dict[str, list[dict[str, Any]]] = {}
        for record in guarded_raw_records:
            identity = str(pauli_child_identity(record))
            lineage_by_identity.setdefault(identity, []).append(
                {
                    "parent_label": str(record["runtime_split_parent_label"]),
                    "child_index": int(record["runtime_split_child_index"]),
                    "child_label": str(record["runtime_split_child_label"]),
                    "family_id": str(record["candidate_family_id"]),
                    "stage_family": str(record["candidate_stage_family"]),
                    "construction": str(record["candidate_construction"]),
                }
            )
        deduplicated, dedup_meta = deduplicate_child_position_records(
            guarded_raw_records,
            score_key="_shared_pool_rank_score",
            identity_policy=CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1,
        )
        meta["guarded_singleton_global_duplicate_count"] = int(
            dedup_meta.get("duplicate_record_count", 0)
        )
        meta["guarded_singleton_pre_padding_identity_count"] = int(
            dedup_meta.get("identity_count", len(deduplicated))
        )
        for record in deduplicated:
            identity = str(record.get("route_a_child_identity") or "")
            lineage = sorted(
                lineage_by_identity.get(identity, []),
                key=lambda row: (
                    str(row["parent_label"]),
                    int(row["child_index"]),
                    str(row["child_label"]),
                ),
            )
            record["route_a_child_lineage_records"] = lineage
        padding_policy = (
            ROUTE_A_CHILD_PADDING_FULL_BINARY_CODE_SPACE_V1
            if str(problem_key).strip().lower()
            == "molecular_vibronic_h2o_linear_fd"
            else ROUTE_A_CHILD_PADDING_HARD_FILTER_V1
        )
        padding_config = RouteAChildPaddingConfig(
            policy=padding_policy,
            problem_key=str(problem_key),
            num_sites=int(num_sites),
            n_ph_max=int(n_ph_max),
            boson_encoding=str(boson_encoding),
            total_register_width=int(total_register_width),
        )
        retained_records, padding_meta = filter_route_a_child_padding_records(
            deduplicated,
            config=padding_config,
        )
        if bool(
            padding_meta.get("projection_applied_before_child_phase1_evaluation")
        ):
            raise RuntimeError(
                "guarded singleton pool unexpectedly applied a projection."
            )
        if not bool(padding_meta.get("applied_before_child_phase1", False)):
            raise RuntimeError(
                "guarded singleton pool lacks a pre-Phase-I padding receipt."
            )
        meta["guarded_singleton_padding_filter"] = dict(padding_meta)
        meta["guarded_singleton_padding_rejected_count"] = int(
            padding_meta.get("rejected_record_count", 0)
        )
        for record in retained_records:
            candidate_term = record.get("candidate_term")
            polynomial = getattr(candidate_term, "polynomial", None)
            if not isinstance(polynomial, PauliPolynomial):
                raise ValueError(
                    "guarded singleton record lacks a canonical polynomial."
                )
            serialized = _serialized_terms(polynomial)
            nonzero = [
                row
                for row in serialized
                if abs(
                    complex(
                        float(row.get("coeff_re", 0.0)),
                        float(row.get("coeff_im", 0.0)),
                    )
                )
                > 1.0e-12
            ]
            if len(nonzero) != 1:
                raise ValueError(
                    "guarded singleton candidate is not exactly one Pauli word."
                )
            pauli_label = str(nonzero[0].get("pauli_exyz", "")).lower()
            if not pauli_label:
                raise ValueError(
                    "guarded singleton candidate has an empty Pauli label."
                )
            if set(pauli_label) <= {"e"}:
                meta["guarded_singleton_null_identity_count"] = int(
                    meta["guarded_singleton_null_identity_count"]
                ) + 1
                meta["guarded_singleton_null_exclusions"].append(
                    {
                        "reason": "raw_singleton_is_identity_global_phase_direction",
                        "pauli_exyz": str(pauli_label),
                        "parent_labels": [
                            str(value)
                            for value in record.get(
                                "route_a_child_parent_labels", []
                            )
                        ],
                    }
                )
                continue
            parent_labels = tuple(
                sorted(
                    {
                        str(value)
                        for value in record.get(
                            "route_a_child_parent_labels", []
                        )
                        if str(value)
                    }
                )
            )
            if not parent_labels:
                raise ValueError(
                    "guarded singleton candidate lost all parent lineage."
                )
            lineage = [
                dict(row)
                for row in record.get("route_a_child_lineage_records", [])
                if isinstance(row, Mapping)
            ]
            raw_meta = record.get("candidate_generator_metadata")
            child_meta = dict(raw_meta) if isinstance(raw_meta, Mapping) else {}
            stable_label = f"guarded_singleton::{pauli_label}"
            child_meta["label"] = stable_label
            child_meta["candidate_label"] = stable_label
            compile_meta = (
                dict(child_meta.get("compile_metadata", {}))
                if isinstance(child_meta.get("compile_metadata"), Mapping)
                else {}
            )
            compile_meta["serialized_terms_exyz"] = [
                dict(row) for row in serialized
            ]
            compile_meta["num_polynomial_terms"] = 1
            compile_meta["signature_size"] = 1
            compile_meta["runtime_split"] = {
                "mode": str(mode_key),
                "representation": "guarded_singleton_child",
                "parent_labels": list(parent_labels),
                "global_deduplication_applied": True,
                "padding_projection_applied": False,
            }
            child_meta["compile_metadata"] = compile_meta
            child_meta["is_macro_generator"] = False
            child_meta["shared_pauli_pool_contract"] = {
                "contract_id": SHARED_PAULI_POOL_CONTRACT_ID,
                "mode": str(mode_key),
                "symmetry_policy": str(policy_key),
                "symmetry_gate_enforced": True,
                "subset_sizes": [1],
                "subset_size_semantics": "exact_allowed_pauli_word_cardinalities",
                "max_subset_size": 1,
                "representation": "guarded_singleton_child",
                "padding_policy": ROUTE_A_CHILD_PADDING_HARD_FILTER_V1,
                "padding_projection_applied": False,
                "global_identity_policy": (
                    CHILD_IDENTITY_POLICY_GLOBAL_PAULI_WORD_V1
                ),
                "parent_labels": list(parent_labels),
                "parent_lineage": lineage,
                "direction_normalization": dict(
                    record.get("route_a_child_direction_normalization", {})
                ),
            }
            candidate = SharedPauliPoolCandidate(
                label=stable_label,
                polynomial=polynomial,
                family_id=str(record.get("candidate_family_id") or "hh"),
                stage_family=str(
                    record.get("candidate_stage_family") or "shared"
                ),
                construction=(
                    f"{record.get('candidate_construction') or 'full_meta'}"
                    "::guarded_singleton_child"
                ),
                execution_mode="termwise_product",
                representation="guarded_singleton_child",
                parent_label=str(parent_labels[0]),
                child_indices=(0,),
                child_labels=(stable_label,),
                symmetry_spec=_symmetry_spec_from_metadata(child_meta, None),
                symmetry_gate=(
                    dict(record["symmetry_gate"])
                    if isinstance(record.get("symmetry_gate"), Mapping)
                    else None
                ),
                generator_metadata=child_meta,
                serialized_terms_exyz=serialized,
                parent_labels=parent_labels,
            )
            candidates.append(candidate)
            meta["guarded_singleton_candidate_count"] = int(
                meta["guarded_singleton_candidate_count"]
            ) + 1
            if max_terms is not None and len(candidates) > int(max_terms):
                raise ValueError(
                    f"shared Pauli-child pool exceeds cap: {len(candidates)} > {int(max_terms)}"
                )

    meta["expanded_pool_term_count"] = int(len(candidates))
    meta["expansion_factor"] = float(len(candidates)) / float(len(parent_rows)) if parent_rows else None
    if projected_children_only:
        meta["projected_singleton_null_count"] = int(
            len(meta["projected_singleton_null_exclusions"])
        )
        if not candidates:
            raise ValueError(
                "projected_singleton_children_only_v1 produced no valid candidates."
            )
        if any(
            candidate.representation != "projected_singleton_child"
            or candidate.parent_label is None
            or len(candidate.child_indices) != 1
            or not candidate.serialized_terms_exyz
            for candidate in candidates
        ):
            raise RuntimeError(
                "projected_singleton_children_only_v1 mixed parent or nonsingleton candidates."
            )
        meta["candidate_representation_counts"] = {
            "parent": 0,
            "child_set": 0,
            "projected_singleton_child": int(len(candidates)),
        }
    if guarded_children_only:
        if not candidates:
            raise ValueError(
                "guarded_singleton_children_only_v1 produced no valid candidates."
            )
        pauli_labels = []
        for candidate in candidates:
            if (
                candidate.representation != "guarded_singleton_child"
                or candidate.parent_label is None
                or not candidate.parent_labels
                or len(candidate.serialized_terms_exyz) != 1
            ):
                raise RuntimeError(
                    "guarded singleton pool mixed parents, macros, or missing lineage."
                )
            pauli_labels.append(
                str(candidate.serialized_terms_exyz[0].get("pauli_exyz", ""))
            )
        if len(set(pauli_labels)) != len(pauli_labels):
            raise RuntimeError(
                "guarded singleton pool contains duplicate Pauli words."
            )
        meta["candidate_representation_counts"] = {
            "parent": 0,
            "child_set": 0,
            "projected_singleton_child": 0,
            "guarded_singleton_child": int(len(candidates)),
        }
    manifest = _manifest_from_candidates(candidates=candidates, meta=meta)
    meta["ordered_label_hash"] = str(manifest["ordered_label_hash"])
    meta["ordered_pool_hash"] = str(manifest["ordered_pool_hash"])
    meta["ordered_candidate_count"] = int(manifest["ordered_candidate_count"])
    meta["contract_identity"] = dict(manifest["contract_identity"])
    meta["manifest"] = dict(manifest)
    return SharedPauliPoolResult(candidates=tuple(candidates), meta=meta, manifest=manifest)


def shared_pauli_pool_fingerprint_rows(candidates: Sequence[SharedPauliPoolCandidate]) -> list[dict[str, Any]]:
    return [_jsonable(_candidate_fingerprint(candidate)) for candidate in candidates]
