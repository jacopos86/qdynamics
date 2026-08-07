"""Global child-set pool expansion for static ADAPT/SNAKE.

This is intentionally separate from the archival Phase-3 runtime split path.
Global expansion changes the candidate pool before Phase 1, so all staged
selectors see the same parent-plus-child-set pool as the generic append/Geo
comparators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from src.quantum.vqe_latex_python_pairs import AnsatzTerm

from pipelines.scaffold.hh_continuation_generators import (
    build_runtime_split_child_sets,
    build_runtime_split_children,
    normalize_runtime_split_subset_sizes,
)

ADAPT_CHILD_POOL_EXPANSION_OFF = "off"
ADAPT_CHILD_POOL_EXPANSION_GLOBAL_PAULI_CHILD_SETS_V1 = "global_pauli_child_sets_v1"
ADAPT_CHILD_POOL_EXPANSION_MODES = frozenset(
    {
        ADAPT_CHILD_POOL_EXPANSION_OFF,
        ADAPT_CHILD_POOL_EXPANSION_GLOBAL_PAULI_CHILD_SETS_V1,
        "pauli_child_sets_v1",
    }
)
ADAPT_CHILD_POOL_EXPANSION_SYMMETRY_POLICY_OFF = "off"
ADAPT_CHILD_POOL_EXPANSION_SYMMETRY_POLICY_HARD_GUARD = "hard_guard"
ADAPT_CHILD_POOL_EXPANSION_SYMMETRY_POLICIES = frozenset(
    {
        ADAPT_CHILD_POOL_EXPANSION_SYMMETRY_POLICY_OFF,
        ADAPT_CHILD_POOL_EXPANSION_SYMMETRY_POLICY_HARD_GUARD,
    }
)


@dataclass(frozen=True)
class ChildPoolExpansionResult:
    pool: list[AnsatzTerm]
    pool_stage_family: list[str]
    pool_family_ids: list[str]
    pool_symmetry_specs: list[dict[str, Any] | None]
    pool_generator_registry: dict[str, dict[str, Any]]
    phase1_core_limit: int
    phase1_residual_indices: set[int]
    meta: dict[str, Any]


def normalize_adapt_child_pool_expansion_mode(value: str | None) -> str:
    key = str(value or ADAPT_CHILD_POOL_EXPANSION_OFF).strip().lower().replace("-", "_")
    if key in {"", "none", "false", "0", "disabled"}:
        key = ADAPT_CHILD_POOL_EXPANSION_OFF
    if key == "pauli_child_sets_v1":
        key = ADAPT_CHILD_POOL_EXPANSION_GLOBAL_PAULI_CHILD_SETS_V1
    if key not in {
        ADAPT_CHILD_POOL_EXPANSION_OFF,
        ADAPT_CHILD_POOL_EXPANSION_GLOBAL_PAULI_CHILD_SETS_V1,
    }:
        allowed = ", ".join(sorted(ADAPT_CHILD_POOL_EXPANSION_MODES))
        raise ValueError(f"adapt_child_pool_expansion_mode must be one of {{{allowed}}}; got {value!r}.")
    return key


def normalize_adapt_child_pool_expansion_symmetry_policy(value: str | None) -> str:
    key = str(value or ADAPT_CHILD_POOL_EXPANSION_SYMMETRY_POLICY_OFF).strip().lower().replace("-", "_")
    if key in {"", "none", "false", "0", "disabled"}:
        key = ADAPT_CHILD_POOL_EXPANSION_SYMMETRY_POLICY_OFF
    if key not in ADAPT_CHILD_POOL_EXPANSION_SYMMETRY_POLICIES:
        allowed = ", ".join(sorted(ADAPT_CHILD_POOL_EXPANSION_SYMMETRY_POLICIES))
        raise ValueError(
            f"adapt_child_pool_expansion_symmetry_policy must be one of {{{allowed}}}; got {value!r}."
        )
    return key


def _child_pool_symmetry_spec(policy: str) -> dict[str, Any] | None:
    if policy == ADAPT_CHILD_POOL_EXPANSION_SYMMETRY_POLICY_OFF:
        return None
    return {
        "hard_guard": bool(policy == ADAPT_CHILD_POOL_EXPANSION_SYMMETRY_POLICY_HARD_GUARD),
        "particle_number_mode": "preserving",
        "spin_sector_mode": "preserving",
        "source": "snake_global_child_pool_expansion",
    }


def _parent_meta(
    registry: Mapping[str, Mapping[str, Any]],
    term: AnsatzTerm,
) -> dict[str, Any]:
    meta = registry.get(str(term.label))
    if isinstance(meta, Mapping):
        out = dict(meta)
    else:
        out = {}
    out.setdefault("label", str(term.label))
    out.setdefault("candidate_label", str(term.label))
    return out


def _meta_symmetry_spec(meta: Mapping[str, Any], fallback: Mapping[str, Any] | None) -> dict[str, Any] | None:
    spec = meta.get("symmetry_spec")
    if isinstance(spec, Mapping):
        return dict(spec)
    if isinstance(fallback, Mapping):
        return dict(fallback)
    return None


def _base_meta(
    *,
    enabled: bool,
    mode: str,
    symmetry_policy: str,
    subset_sizes: Sequence[int],
    base_pool_size: int,
) -> dict[str, Any]:
    subset_sizes_tuple = tuple(int(size) for size in subset_sizes)
    return {
        "schema": "snake_global_child_pool_expansion_v1",
        "enabled": bool(enabled),
        "mode": str(mode),
        "symmetry_policy": str(symmetry_policy),
        "subset_sizes": [int(size) for size in subset_sizes_tuple],
        "subset_size_semantics": "exact_allowed_pauli_word_cardinalities",
        "max_subset_size": int(max(subset_sizes_tuple)),
        "source": "before_phase1_global_child_set_pool",
        "base_pool_term_count": int(base_pool_size),
        "expanded_pool_term_count": int(base_pool_size),
        "split_parent_count": 0,
        "child_atom_count": 0,
        "child_set_candidate_count": 0,
        "added_child_set_count": 0,
        "duplicate_child_set_count": 0,
        "core_parent_count": 0,
        "core_added_child_set_count": 0,
        "residual_parent_count": 0,
        "residual_added_child_set_count": 0,
        "symmetry_checked_child_atom_count": 0,
        "symmetry_rejected_child_atom_count": 0,
        "symmetry_checked_child_set_count": 0,
        "symmetry_rejected_child_set_count": 0,
        "hh_fermion_symmetry_filter": {
            "active": bool(enabled),
            "rejected_child_count": 0,
            "rejected_child_set_count": 0,
            "unchecked_rejected_count": 0,
            "failed_gate_rejected_count": 0,
            "rejected_labels_sample": [],
        },
        "supports_only": "paper_i_hh_hubbard_snake_global_child_set_pool",
        "matches_generic_append_geo_policy": "child_set_candidates_not_raw_child_atoms",
    }


def _record_hh_fermion_rejection(meta: dict[str, Any], *, label: str, gate: Mapping[str, Any] | None, kind: str) -> None:
    filt = meta.get("hh_fermion_symmetry_filter")
    if not isinstance(filt, dict):
        return
    if kind == "child_set":
        filt["rejected_child_set_count"] = int(filt.get("rejected_child_set_count", 0)) + 1
        meta["symmetry_rejected_child_set_count"] = int(meta.get("symmetry_rejected_child_set_count", 0)) + 1
    else:
        filt["rejected_child_count"] = int(filt.get("rejected_child_count", 0)) + 1
    if not isinstance(gate, Mapping) or not bool(gate.get("checked", False)):
        filt["unchecked_rejected_count"] = int(filt.get("unchecked_rejected_count", 0)) + 1
    elif not bool(gate.get("passed", True)):
        filt["failed_gate_rejected_count"] = int(filt.get("failed_gate_rejected_count", 0)) + 1
    sample = filt.get("rejected_labels_sample")
    if isinstance(sample, list) and len(sample) < 20:
        sample.append(str(label))


def _runtime_split_gate_checked_and_passed(gate: Any) -> bool:
    return bool(isinstance(gate, Mapping) and bool(gate.get("checked", False)) and bool(gate.get("passed", False)))


def expand_snake_pool_with_global_child_sets(
    *,
    pool: Sequence[AnsatzTerm],
    pool_stage_family: Sequence[str],
    pool_family_ids: Sequence[str],
    pool_symmetry_specs: Sequence[Mapping[str, Any] | None],
    pool_generator_registry: Mapping[str, Mapping[str, Any]],
    phase1_core_limit: int,
    phase1_residual_indices: set[int],
    mode: str | None,
    symmetry_policy: str | None,
    max_subset_size: int | None = None,
    subset_sizes: Sequence[int] | str | int | None = None,
    problem_key: str,
    continuation_mode: str,
    num_sites: int,
    ordering: str,
    qpb: int,
    ai_log: Callable[..., None] | None = None,
) -> ChildPoolExpansionResult:
    mode_key = normalize_adapt_child_pool_expansion_mode(mode)
    policy_key = normalize_adapt_child_pool_expansion_symmetry_policy(symmetry_policy)
    requested_subset_sizes = normalize_runtime_split_subset_sizes(
        subset_sizes,
        legacy_max_subset_size=max_subset_size,
    )
    base_pool = list(pool)
    base_stage = [str(x) for x in pool_stage_family]
    base_family = [str(x) for x in pool_family_ids]
    base_specs = [
        (dict(x) if isinstance(x, Mapping) else None)
        for x in pool_symmetry_specs
    ]
    registry = {str(k): dict(v) for k, v in pool_generator_registry.items()}

    meta = _base_meta(
        enabled=bool(mode_key != ADAPT_CHILD_POOL_EXPANSION_OFF),
        mode=mode_key,
        symmetry_policy=policy_key,
        subset_sizes=requested_subset_sizes,
        base_pool_size=len(base_pool),
    )
    if mode_key == ADAPT_CHILD_POOL_EXPANSION_OFF:
        return ChildPoolExpansionResult(
            pool=base_pool,
            pool_stage_family=base_stage,
            pool_family_ids=base_family,
            pool_symmetry_specs=base_specs,
            pool_generator_registry=registry,
            phase1_core_limit=int(phase1_core_limit),
            phase1_residual_indices=set(int(i) for i in phase1_residual_indices),
            meta=meta,
        )

    if str(problem_key).strip().lower() not in {"hh", "hubbard"}:
        raise ValueError("adapt_child_pool_expansion_mode is currently only valid for problem in {'hh','hubbard'}.")
    if str(continuation_mode).strip().lower() != "phase3_v1":
        raise ValueError(
            "adapt_child_pool_expansion_mode is currently only valid for adapt_continuation_mode='phase3_v1'."
        )
    core_indices = [
        int(idx)
        for idx in range(len(base_pool))
        if int(idx) < int(phase1_core_limit) and int(idx) not in set(phase1_residual_indices)
    ]
    residual_indices = [int(idx) for idx in range(len(base_pool)) if int(idx) not in set(core_indices)]
    meta["core_parent_count"] = int(len(core_indices))
    meta["residual_parent_count"] = int(len(residual_indices))
    child_symmetry_spec = _child_pool_symmetry_spec(policy_key)
    seen_labels = {str(term.label) for term in base_pool}

    out_pool: list[AnsatzTerm] = []
    out_stage: list[str] = []
    out_family: list[str] = []
    out_specs: list[dict[str, Any] | None] = []
    child_entries: dict[str, list[tuple[AnsatzTerm, str, str, dict[str, Any] | None, dict[str, Any]]]] = {
        "core": [],
        "residual": [],
    }

    def append_original(idx: int) -> None:
        term = base_pool[int(idx)]
        out_pool.append(term)
        out_stage.append(str(base_stage[int(idx)] if int(idx) < len(base_stage) else "core"))
        out_family.append(str(base_family[int(idx)] if int(idx) < len(base_family) else "hh"))
        out_specs.append(base_specs[int(idx)] if int(idx) < len(base_specs) else None)

    def build_child_entries(idx: int, bucket: str) -> None:
        parent = base_pool[int(idx)]
        family_id = str(base_family[int(idx)] if int(idx) < len(base_family) else "hh")
        parent_spec = base_specs[int(idx)] if int(idx) < len(base_specs) else None
        parent_meta = _parent_meta(registry, parent)
        children = build_runtime_split_children(
            parent_label=str(parent.label),
            polynomial=parent.polynomial,
            family_id=family_id,
            num_sites=int(num_sites),
            ordering=str(ordering),
            qpb=int(max(1, qpb)),
            split_mode=mode_key,
            parent_generator_metadata=parent_meta,
            symmetry_spec=child_symmetry_spec,
        )
        if not children:
            return
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
            family_id=family_id,
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
            label = str(child_set.get("candidate_label"))
            gate = child_set.get("symmetry_gate")
            if (
                policy_key == ADAPT_CHILD_POOL_EXPANSION_SYMMETRY_POLICY_HARD_GUARD
                and not _runtime_split_gate_checked_and_passed(gate)
            ):
                _record_hh_fermion_rejection(meta, label=label, gate=gate if isinstance(gate, Mapping) else None, kind="child_set")
                continue
            if label in seen_labels:
                meta["duplicate_child_set_count"] = int(meta["duplicate_child_set_count"]) + 1
                continue
            child_meta = dict(child_set.get("candidate_generator_metadata") or {})
            child_meta["global_child_pool_expansion"] = {
                "mode": str(mode_key),
                "source": "before_phase1_global_child_set_pool",
                "parent_label": str(parent.label),
                "stage_bucket": str(bucket),
            }
            registry[label] = dict(child_meta)
            if isinstance(gate, Mapping) and bool(gate.get("checked", False)):
                meta["symmetry_checked_child_set_count"] = int(meta["symmetry_checked_child_set_count"]) + 1
            child_term = AnsatzTerm(
                label=label,
                polynomial=child_set["candidate_polynomial"],
                execution_mode=str(
                    child_set.get("recommended_execution_mode")
                    or getattr(parent, "execution_mode", "termwise_product")
                ),
            )
            child_spec = _meta_symmetry_spec(child_meta, parent_spec)
            child_entries[bucket].append((child_term, str(bucket), family_id, child_spec, child_meta))
            seen_labels.add(label)
            meta["added_child_set_count"] = int(meta["added_child_set_count"]) + 1
            if bucket == "core":
                meta["core_added_child_set_count"] = int(meta["core_added_child_set_count"]) + 1
            else:
                meta["residual_added_child_set_count"] = int(meta["residual_added_child_set_count"]) + 1

    for idx in core_indices:
        append_original(idx)
        build_child_entries(idx, "core")
    for child_term, stage, family_id, spec, _child_meta in child_entries["core"]:
        out_pool.append(child_term)
        out_stage.append(stage)
        out_family.append(family_id)
        out_specs.append(spec)
    new_core_limit = int(len(out_pool))
    for idx in residual_indices:
        append_original(idx)
        build_child_entries(idx, "residual")
    for child_term, stage, family_id, spec, _child_meta in child_entries["residual"]:
        out_pool.append(child_term)
        out_stage.append(stage)
        out_family.append(family_id)
        out_specs.append(spec)

    new_residual_indices = set(range(new_core_limit, len(out_pool)))
    meta["expanded_pool_term_count"] = int(len(out_pool))
    meta["expansion_factor"] = (
        float(len(out_pool)) / float(len(base_pool))
        if len(base_pool) > 0
        else None
    )
    if ai_log is not None:
        ai_log("hardcoded_adapt_child_pool_expansion_applied", **meta)
    return ChildPoolExpansionResult(
        pool=out_pool,
        pool_stage_family=out_stage,
        pool_family_ids=out_family,
        pool_symmetry_specs=out_specs,
        pool_generator_registry=registry,
        phase1_core_limit=new_core_limit,
        phase1_residual_indices=new_residual_indices,
        meta=meta,
    )


__all__ = [
    "ADAPT_CHILD_POOL_EXPANSION_GLOBAL_PAULI_CHILD_SETS_V1",
    "ADAPT_CHILD_POOL_EXPANSION_MODES",
    "ADAPT_CHILD_POOL_EXPANSION_OFF",
    "ADAPT_CHILD_POOL_EXPANSION_SYMMETRY_POLICIES",
    "ADAPT_CHILD_POOL_EXPANSION_SYMMETRY_POLICY_HARD_GUARD",
    "ChildPoolExpansionResult",
    "expand_snake_pool_with_global_child_sets",
    "normalize_adapt_child_pool_expansion_mode",
    "normalize_adapt_child_pool_expansion_symmetry_policy",
]
