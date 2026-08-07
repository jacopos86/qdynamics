"""Shared Pauli-child pool contract for Paper-I method comparisons.

This module is deliberately neutral: it does not know about SNAKE phases,
append-only selection, or Geo-ADAPT scoring.  Callers provide the same macro
parent pool and receive the same ordered parent-plus-child-set pool, together
with hashes that report scripts can compare across methods.
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

SHARED_PAULI_POOL_CONTRACT_ID = "paper_i_shared_pauli_child_pool_v1"
SHARED_PAULI_POOL_MODE_OFF = "off"
SHARED_PAULI_POOL_MODE_CHILD_SETS_V1 = "shared_pauli_child_sets_v1"
SHARED_PAULI_POOL_MODE_ALIASES = {
    SHARED_PAULI_POOL_MODE_OFF: SHARED_PAULI_POOL_MODE_OFF,
    "none": SHARED_PAULI_POOL_MODE_OFF,
    "false": SHARED_PAULI_POOL_MODE_OFF,
    "0": SHARED_PAULI_POOL_MODE_OFF,
    "disabled": SHARED_PAULI_POOL_MODE_OFF,
    "pauli_child_sets_v1": SHARED_PAULI_POOL_MODE_CHILD_SETS_V1,
    "global_pauli_child_sets_v1": SHARED_PAULI_POOL_MODE_CHILD_SETS_V1,
    SHARED_PAULI_POOL_MODE_CHILD_SETS_V1: SHARED_PAULI_POOL_MODE_CHILD_SETS_V1,
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
    return {
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
    return {
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
    meta = _base_meta(
        mode=mode_key,
        symmetry_policy=policy_key,
        subset_sizes=requested_subset_sizes,
        base_count=int(len(parent_rows)),
        explicit_no_guard=explicit_no_guard,
    )
    if mode_key != SHARED_PAULI_POOL_MODE_OFF:
        if str(problem_key).strip().lower() not in {"hh", "hubbard"}:
            raise ValueError("shared_pauli_pool_mode is currently only valid for problem in {'hh','hubbard'}.")

    candidates: list[SharedPauliPoolCandidate] = []
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
        candidates.append(parent_candidate)
        seen_labels.add(str(parent_candidate.label))
        if mode_key == SHARED_PAULI_POOL_MODE_OFF or len(parent_serialized) <= 1:
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

    meta["expanded_pool_term_count"] = int(len(candidates))
    meta["expansion_factor"] = float(len(candidates)) / float(len(parent_rows)) if parent_rows else None
    manifest = _manifest_from_candidates(candidates=candidates, meta=meta)
    meta["ordered_label_hash"] = str(manifest["ordered_label_hash"])
    meta["ordered_pool_hash"] = str(manifest["ordered_pool_hash"])
    meta["ordered_candidate_count"] = int(manifest["ordered_candidate_count"])
    meta["contract_identity"] = dict(manifest["contract_identity"])
    meta["manifest"] = dict(manifest)
    return SharedPauliPoolResult(candidates=tuple(candidates), meta=meta, manifest=manifest)


def shared_pauli_pool_fingerprint_rows(candidates: Sequence[SharedPauliPoolCandidate]) -> list[dict[str, Any]]:
    return [_jsonable(_candidate_fingerprint(candidate)) for candidate in candidates]
