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
    build_local_ray_refit,
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


def build_candidate_pool(
    state: APMcLachlanState,
    *,
    support_config: Any,
    insertions_enabled: bool = True,
) -> tuple[tuple[Any, ...], dict[str, str], dict[str, Any]]:
    """The candidate insertion pool, identically for every structural policy.

    Every policy that ranks insertions must choose from the same set, or a
    comparison between policies measures the pools instead of the rules.  This
    is the single place that decides pool membership and order: occurrence
    policy, single-Pauli-word identity deduplication (two candidates with the
    same word generate the same one-parameter family, since the coefficient
    only rescales theta), and the configuration-level cap.

    The cap keeps the pool's frozen order rather than sorting, because any
    reordering here silently changes which operators a capped policy can see.

    Returns the ordered atoms, their Pauli words, and pool telemetry.
    """

    if insertions_enabled:
        raw_atoms = candidate_append_atoms(
            state,
            allow_incomplete_candidate_pool=bool(
                support_config.allow_incomplete_candidate_pool
            ),
            occurrence_policy=str(support_config.append_occurrence_policy),
        )
    else:
        # Prune-only mode: insertions need new measurements and the
        # structural-repair predicate is inactive, so the pool is empty.
        raw_atoms = ()

    atoms: list[Any] = []
    pauli_by_atom: dict[str, str] = {}
    seen_words: set[str] = set()
    for atom in raw_atoms:
        specs = iter_runtime_rotation_terms(
            getattr(atom.term, "polynomial"),
            ignore_identity=bool(state.executor.ignore_identity),
            coefficient_tolerance=float(state.executor.coefficient_tolerance),
            sort_terms=bool(state.executor.sort_terms),
        )
        word = str(specs[0].pauli_exyz) if len(specs) == 1 else None
        if word is not None:
            if word in seen_words:
                continue
            seen_words.add(word)
            pauli_by_atom[str(atom.atom_id)] = word
        atoms.append(atom)

    pool_cap = getattr(support_config, "max_structural_pool_size", None)
    if pool_cap is not None:
        atoms = atoms[: max(0, int(pool_cap))]
        kept = {str(a.atom_id) for a in atoms}
        pauli_by_atom = {k: v for k, v in pauli_by_atom.items() if k in kept}

    telemetry = {
        "candidate_pool_raw": int(len(raw_atoms)),
        "candidate_pool_deduplicated": int(len(atoms)),
        "candidate_pool_order": "frozen_pool_order",
        "insertions_enabled": bool(insertions_enabled),
    }
    return tuple(atoms), pauli_by_atom, telemetry


def build_selector_inputs(
    *,
    state: APMcLachlanState,
    evaluation: GeometryEvaluation,
    support_config: Any,
    runtime_state: Any,
    theta_runtime: np.ndarray,
    time_index: int,
    active_prune_atoms: Any,
    insertions_enabled: bool = True,
) -> dict[str, Any]:
    """Assemble every selector input from route objects.

    ``active_prune_atoms`` is the route's feasibility function (injected to
    avoid a circular import with the trajectory module); it must apply target
    policy, drive-aligned protection, cooldown, and surviving-support gates.
    """

    atoms, pauli_by_atom, pool_dedup_telemetry = build_candidate_pool(
        state,
        support_config=support_config,
        insertions_enabled=bool(insertions_enabled),
    )
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
        lambda_hist=float(getattr(support_config, "prune_history_lambda", 0.0)),
        lambda_cond_relief=float(
            getattr(support_config, "prune_condition_lambda_kappa_rel", 0.0)
        ),
        lambda_cond_damage=float(
            getattr(support_config, "prune_condition_lambda_kappa_dam", 0.0)
        ),
        epsilon_L=float(support_config.eps_loss),
    )

    max_batch = int(
        getattr(support_config, "max_insertion_batch_size", None)
        or support_config.max_append_batch_size
    )

    # Hook #2: historical deletion-loss prior.  ``runtime_state.loss_history``
    # maps stable runtime coordinate labels to ``(time_index, loss)`` records;
    # a deletion set's history term is the mean over its coordinates of each
    # coordinate's windowed mean loss (coordinates without history contribute
    # zero, so the term vanishes until evidence accumulates).
    runtime_labels = tuple(state.runtime_coordinate_labels)
    history_window = max(1, int(getattr(support_config, "prune_history_window", 3)))
    loss_history: dict[str, list[tuple[int, float]]] = (
        getattr(runtime_state, "loss_history", None) or {}
    )

    def deletion_history_loss(removed: tuple[int, ...]) -> float:
        if not removed:
            return 0.0
        total = 0.0
        for index in removed:
            records = loss_history.get(runtime_labels[int(index)])
            if records:
                recent = records[-history_window:]
                total += sum(loss for _t, loss in recent) / len(recent)
        return total / len(removed)

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
        deletion_history_loss=deletion_history_loss,
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
    insertions_enabled: bool = True,
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
        insertions_enabled=bool(insertions_enabled),
    )
    inputs["structural_kwargs"]["inverse_policy"] = inverse_policy
    gates = CertificationGates(
        ray_distance_max=float(support_config.prune_ray_distance_tol),
        smoothness_eta_max=float(support_config.prune_patch_smoothness_eta_max),
        condition_number_max=(
            None
            if support_config.append_schur_max_condition_number is None
            else float(support_config.append_schur_max_condition_number)
        ),
    )

    def escalate() -> bool:
        return float(base_step.residual_ratio) >= float(
            support_config.residual_ratio_threshold
        )

    # Hook #3: optional bounded local refit toward the frozen checkpoint ray,
    # applied to materialized finalists before the hard gates.  Pure
    # insertions start on the ray and are skipped inside the hook.
    refit = None
    if bool(getattr(support_config, "certification_refit_enabled", False)):
        refit = build_local_ray_refit(
            target_psi=base_evaluation.psi,
            trust_radius=float(
                getattr(support_config, "certification_refit_trust_radius", 0.1)
            ),
            max_iterations=int(
                getattr(support_config, "certification_refit_max_iterations", 15)
            ),
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
        refit=refit,
        solve_repair_config=solve_repair_config,
        max_certification_attempts_per_level=getattr(
            support_config, "max_certification_attempts_per_level", None
        ),
        max_certification_attempts_per_deletion_branch=getattr(
            support_config, "max_certification_attempts_per_deletion_branch", None
        ),
    )

    # Record realized deletion losses so hook #2's prior sees this attempt at
    # future checkpoints.  Loss is attributed equally across the deleted
    # coordinates; keys are stable runtime coordinate labels, so survivors
    # keep their record across later structural edits.
    if runtime_state is not None and getattr(runtime_state, "loss_history", None) is not None:
        window = max(1, int(getattr(support_config, "prune_history_window", 3)))
        labels = tuple(state.runtime_coordinate_labels)
        for attempt in selection.attempts:
            if not attempt.removed_runtime_indices or attempt.deletion_loss is None:
                continue
            share = float(attempt.deletion_loss) / len(attempt.removed_runtime_indices)
            for index in attempt.removed_runtime_indices:
                records = runtime_state.loss_history.setdefault(labels[int(index)], [])
                records.append((int(time_index), share))
                del records[:-window]

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
    "build_candidate_pool",
    "build_selector_inputs",
    "select_deletion_conditioned_patch",
]
