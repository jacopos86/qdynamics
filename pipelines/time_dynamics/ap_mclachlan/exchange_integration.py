"""Route integration of the deletion-conditioned exchange selector.

Adapts the trajectory loop's objects — state, controller config, prune runtime
state, cost settings — into the pure selector stack (structural cache,
enumeration, certification) and maps the outcome back onto ``PatchDecision``.
This is the single entry point the trajectory loop calls; the retired
append-ladder/prune-ladder/exchange-pairing selectors have no successor other
than this.

Wiring choices, recorded for reproduction:

* deletion feasibility comes from the existing atom gates (target policy,
  drive-aligned protection, cooldown, occurrence identity, minimum surviving
  support) via ``_active_prune_atoms``;
* insertion candidates come from ``candidate_append_atoms`` under the
  configured occurrence policy; the branch pool is deletion-independent under
  ``layer_reuse`` and re-admits deleted bases under ``unique_support``;
* hardware costs use the Paper-I proxy estimators with per-candidate
  denominators (``cost_normalization: per_candidate_raw_v1``); family-robust
  normalization over the enumerated family can supersede this without
  touching enumeration;
* certification gates reuse the route's existing thresholds: prune ray
  tolerance, prune patch-smoothness eta, and the Schur condition bound.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.append_cost import (
    AppendCostSettings,
    append_cost_telemetry_for_family,
    estimate_append_atom_set_cost,
)
from pipelines.time_dynamics.ap_mclachlan.commutation import singleton_insertion_cuts
from pipelines.time_dynamics.ap_mclachlan.exchange_certification import (
    CertificationGates,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_selector import (
    EXCHANGE_SELECTION_POLICY_V1,
    ExchangeSelection,
    select_exchange_patch,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_structural import (
    StructuralScoreWeights,
)
from pipelines.time_dynamics.ap_mclachlan.fixed_step import FixedMcLachlanStep
from pipelines.time_dynamics.ap_mclachlan.geometry_eval import GeometryEvaluation
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import TimeDependentHamiltonian
from pipelines.time_dynamics.ap_mclachlan.insertion_words import (
    tokens_commute_from_terms,
)
from pipelines.time_dynamics.ap_mclachlan.inverse import McLachlanInversePolicy
from pipelines.time_dynamics.ap_mclachlan.prune_cost import (
    estimate_prune_atom_set_cost,
)
from pipelines.time_dynamics.ap_mclachlan.state import (
    APMcLachlanState,
    _per_coordinate_terms,
)
from pipelines.time_dynamics.ap_mclachlan.structural_cache import (
    build_structural_insertion_cache,
)
from src.quantum.ansatz_parameterization import iter_runtime_rotation_terms
from pipelines.time_dynamics.ap_mclachlan.support_atoms import (
    active_support_atoms,
    candidate_append_atoms,
)


def _paper_i_proxy_denominator(
    raw_estimate: Any,
    *,
    settings: AppendCostSettings,
) -> float:
    """Per-candidate Paper-I proxy cost denominator (>= 1)."""

    telemetry = append_cost_telemetry_for_family(
        (raw_estimate,),
        insertion_gains=(None,),
        settings=settings,
    )[0]
    return max(1.0, float(telemetry.hardware_cost_denominator))


def build_selector_inputs(
    *,
    state: APMcLachlanState,
    evaluation: GeometryEvaluation,
    support_config: Any,
    runtime_state: Any,
    theta_runtime: np.ndarray,
    time_index: int,
    active_prune_atoms: Any,
) -> dict[str, Any]:
    """Assemble every selector input from route objects.

    ``active_prune_atoms`` is the route's feasibility function (injected to
    avoid a circular import with the trajectory module); it must apply target
    policy, drive-aligned protection, cooldown, and surviving-support gates.
    """

    raw_atoms = candidate_append_atoms(
        state,
        allow_incomplete_candidate_pool=bool(
            support_config.allow_incomplete_candidate_pool
        ),
        occurrence_policy=str(support_config.append_occurrence_policy),
    )
    # Identity-level deduplication: two candidates with the same single Pauli
    # child generate the same one-parameter family exp(-i theta c P) (the
    # coefficient rescales theta), so they are the same insertion operator.
    # Keep the first occurrence in frozen pool order.  This is not a score
    # prefilter: no geometry or score is consulted, only operator identity.
    atoms = []
    seen_words: set[str] = set()
    for atom in raw_atoms:
        specs = iter_runtime_rotation_terms(
            getattr(atom.term, "polynomial"),
            ignore_identity=bool(state.executor.ignore_identity),
            coefficient_tolerance=float(state.executor.coefficient_tolerance),
            sort_terms=bool(state.executor.sort_terms),
        )
        word = str(specs[0].pauli_exyz) if len(specs) == 1 else None
        if word is not None and word in seen_words:
            continue
        if word is not None:
            seen_words.add(word)
        atoms.append(atom)
    atoms = tuple(atoms)
    pool_dedup_telemetry = {
        "candidate_pool_raw": int(len(raw_atoms)),
        "candidate_pool_deduplicated": int(len(atoms)),
    }
    atoms_by_id = {str(atom.atom_id): atom for atom in atoms}
    ordered_atom_ids = tuple(str(atom.atom_id) for atom in atoms)

    blocks = _per_coordinate_terms(state)
    cuts_by_atom = {
        atom_id: singleton_insertion_cuts(atoms_by_id[atom_id].term, blocks)
        for atom_id in ordered_atom_ids
    }

    terms_by_key: dict[str, Any] = {
        str(label): term
        for label, term in zip(state.runtime_coordinate_labels, blocks)
    }
    for atom_id in ordered_atom_ids:
        terms_by_key[atom_id] = atoms_by_id[atom_id].term

    cache = build_structural_insertion_cache(
        state=state,
        evaluation=evaluation,
        cuts_by_atom=cuts_by_atom,
        atoms_by_id=atoms_by_id,
        checkpoint_key=(int(time_index), state.runtime_coordinate_labels),
    )

    deletable_atoms = active_prune_atoms(
        state,
        theta_runtime=theta_runtime,
        support_config=support_config,
        runtime_state=runtime_state,
        time_index=int(time_index),
    )
    deletable_indices = tuple(
        sorted(
            index
            for atom in deletable_atoms
            for index in atom.runtime_indices
        )
    )
    active_by_index = {
        int(index): atom
        for atom in active_support_atoms(state, theta_runtime)
        for index in atom.runtime_indices
    }
    active_base_ids = {
        str(dict(atom.metadata or {}).get("base_atom_id", atom.atom_id))
        for atom in active_by_index.values()
    }

    occurrence_reuse = (
        str(support_config.append_occurrence_policy).strip().lower()
        != "unique_support"
    )

    def candidate_pool_for_deletion(removed: tuple[int, ...]) -> tuple[str, ...]:
        if occurrence_reuse:
            return ordered_atom_ids
        surviving_bases = {
            str(
                dict(active_by_index[i].metadata or {}).get(
                    "base_atom_id", active_by_index[i].atom_id
                )
            )
            for i in active_by_index
            if i not in set(removed)
        }
        out = []
        for atom_id in ordered_atom_ids:
            atom = atoms_by_id[atom_id]
            base = str(dict(atom.metadata or {}).get("base_atom_id", atom.atom_id))
            if base not in surviving_bases:
                out.append(atom_id)
        return tuple(out)

    cost_settings = AppendCostSettings.from_config(support_config)

    def insertion_cost(atom_ids: tuple[str, ...]) -> float:
        selected = tuple(atoms_by_id[str(a)] for a in atom_ids)
        return _paper_i_proxy_denominator(
            estimate_append_atom_set_cost(selected), settings=cost_settings
        )

    def deletion_cost(removed: tuple[int, ...]) -> float:
        selected = tuple(
            active_by_index[int(i)] for i in removed if int(i) in active_by_index
        )
        if not selected:
            return 1.0
        return _paper_i_proxy_denominator(
            estimate_prune_atom_set_cost(selected), settings=cost_settings
        )

    pauli_by_atom: dict[str, str] = {}
    for atom_id in ordered_atom_ids:
        specs = iter_runtime_rotation_terms(
            getattr(atoms_by_id[atom_id].term, "polynomial"),
            ignore_identity=bool(state.executor.ignore_identity),
            coefficient_tolerance=float(state.executor.coefficient_tolerance),
            sort_terms=bool(state.executor.sort_terms),
        )
        if len(specs) == 1:
            pauli_by_atom[atom_id] = str(specs[0].pauli_exyz)

    def occurrence_label(atom: Any, cut: int, ordinal: int) -> str:
        # Stable child form ("::r0::<pauli>" suffix) so the rebuilt layout
        # keeps the label verbatim instead of re-suffixing it.
        pauli = pauli_by_atom[str(atom.atom_id)]
        return (
            f"{atom.atom_label}::insr{int(time_index)}c{int(cut)}o{int(ordinal)}"
            f"::r0::{pauli}"
        )

    weights = StructuralScoreWeights(
        alpha_ins=float(support_config.append_cost_alpha),
        alpha_del=float(support_config.prune_cost_alpha),
        w_delta=float(support_config.patch_utility_delta_weight),
        epsilon_L=float(support_config.eps_loss),
    )

    max_batch = int(
        getattr(support_config, "max_insertion_batch_size", None)
        or support_config.max_append_batch_size
    )

    structural_kwargs = dict(
        cache=cache,
        base_K=np.asarray(evaluation.geometry.K, dtype=float),
        base_f=np.asarray(evaluation.geometry.f, dtype=float),
        norm_b_sq=float(evaluation.geometry.norm_b_sq),
        weights=weights,
        deletable_indices=deletable_indices,
        min_surviving_support=int(support_config.min_runtime_parameter_count),
        cuts_by_atom=cuts_by_atom,
        candidate_pool_for_deletion=candidate_pool_for_deletion,
        insertion_cost=insertion_cost,
        deletion_cost=deletion_cost,
        tokens_commute=tokens_commute_from_terms(terms_by_key),
        max_insertion_batch_size=max_batch,
        interaction_frontier_widths=getattr(
            support_config, "interaction_frontier_widths", None
        ),
        max_joint_patch_evaluations=getattr(
            support_config, "max_joint_patch_evaluations", None
        ),
    )
    return {
        "atoms_by_id": atoms_by_id,
        "occurrence_label": occurrence_label,
        "structural_kwargs": structural_kwargs,
        "pool_dedup_telemetry": pool_dedup_telemetry,
    }


def select_deletion_conditioned_patch(
    *,
    state: APMcLachlanState,
    hamiltonian: TimeDependentHamiltonian,
    theta_runtime: np.ndarray,
    time: float,
    base_evaluation: GeometryEvaluation,
    base_step: FixedMcLachlanStep,
    inverse_policy: McLachlanInversePolicy,
    support_config: Any,
    runtime_state: Any,
    time_index: int,
    active_prune_atoms: Any,
    solve_repair_config: Any | None = None,
) -> tuple[ExchangeSelection, dict[str, Any]]:
    """Run the exchange selector at one checkpoint with route wiring.

    Returns the selection and a JSON-safe decision payload.  Escalation past
    each level follows the structural-repair predicate: the base residual
    ratio must exceed the configured threshold for the search to keep
    acquiring families after a level certifies nothing.
    """

    inputs = build_selector_inputs(
        state=state,
        evaluation=base_evaluation,
        support_config=support_config,
        runtime_state=runtime_state,
        theta_runtime=theta_runtime,
        time_index=int(time_index),
        active_prune_atoms=active_prune_atoms,
    )
    inputs["structural_kwargs"]["inverse_policy"] = inverse_policy
    gates = CertificationGates(
        ray_distance_max=float(support_config.prune_ray_distance_tol),
        smoothness_eta_max=float(support_config.prune_patch_smoothness_eta_max),
        condition_number_max=float(
            support_config.append_schur_max_condition_number
        ),
    )

    def escalate() -> bool:
        return float(base_step.residual_ratio) >= float(
            support_config.residual_ratio_threshold
        )

    selection = select_exchange_patch(
        state=state,
        hamiltonian=hamiltonian,
        theta_runtime=theta_runtime,
        time=float(time),
        base_evaluation=base_evaluation,
        base_step=base_step,
        inverse_policy=inverse_policy,
        gates=gates,
        atoms_by_id=inputs["atoms_by_id"],
        occurrence_label=inputs["occurrence_label"],
        structural_kwargs=inputs["structural_kwargs"],
        score_floor=float(
            getattr(support_config, "structural_score_floor", 0.0) or 0.0
        ),
        escalate=escalate,
        solve_repair_config=solve_repair_config,
    )

    payload: dict[str, Any] = {
        "selection_policy": EXCHANGE_SELECTION_POLICY_V1,
        **inputs.get("pool_dedup_telemetry", {}),
        "kind": selection.kind,
        "stop_reason": selection.stop_reason,
        "attempt_count": len(selection.attempts),
        "attempts": [a.to_json_dict() for a in selection.attempts[-16:]],
        "cost_normalization": "per_candidate_raw_v1",
    }
    if selection.telemetry is not None:
        payload["work_guard"] = selection.telemetry.guard.to_json_dict()
        payload["frontier_schedule"] = [
            int(w) for w in selection.telemetry.frontier_schedule
        ]
        payload["frontiers_used"] = int(selection.telemetry.frontiers_used)
        payload["q_base"] = float(selection.telemetry.q_base)
    if selection.committed is not None:
        payload["committed"] = {
            "removed_runtime_indices": [
                int(i) for i in selection.committed.removed_runtime_indices
            ],
            "inserted_selection": [
                [str(a), int(p)]
                for a, p in selection.committed.inserted_selection
            ],
            "family": selection.committed.family,
            "score": float(selection.committed.score),
            "q": float(selection.committed.q),
            "insertion_gain": float(selection.committed.insertion_gain),
            "deletion_loss": float(selection.committed.deletion_loss),
        }
        if selection.certification is not None:
            payload["certification"] = selection.certification.to_json_dict()
    return selection, payload


__all__ = [
    "build_selector_inputs",
    "select_deletion_conditioned_patch",
]
