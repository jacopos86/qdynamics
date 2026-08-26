"""The Paper-II generalized-exchange deep module.

One checkpoint search chooses a pair ``(D, I)``: active runtime coordinates
``D`` to delete and positioned candidate occurrences ``I`` to insert.  Pure
insertion is the boundary ``D = empty``; pure deletion is ``I = empty``; true
exchange has both components nonempty.  They are restrictions of one proposal
space, not three algorithms.

``select_generalized_exchange_patch`` is the trajectory route's only public
selection operation.  It owns which faces are admissible, which ranking is in
force, and the realized-L2 commit rule.  In particular, the trajectory caller
does not reconstruct an "append route" or a "prune route":

* above the McLachlan-distance cut, insertions open and deletions remain open
  whenever the support floor permits them, so the complete family is searched;
* below the cut, insertions close and the measurement-free deletion face stays
  available;
* insertion-only debt is retained only as an explicit ablation; and
* ``drift_ranked`` is the canonical debt ranking, not a separate selector.

Wiring choices, recorded for reproduction:

* deletion feasibility first applies the existing atom gates (target policy,
  drive-aligned protection, cooldown, occurrence identity, minimum surviving
  support), then a measurement-free set permission combining an accumulated
  rotation-angle ray bound with normalized reverse-Schur loss;
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

from typing import Any, Mapping

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.append_cost import (
    AppendCostSettings,
    append_cost_telemetry_for_family,
    estimate_append_atom_set_cost,
)
from pipelines.time_dynamics.ap_mclachlan.avqds import (
    mclachlan_distance_squared,
)
from pipelines.time_dynamics.ap_mclachlan.commutation import singleton_insertion_cuts
from pipelines.time_dynamics.ap_mclachlan.deletion_permission import (
    DeletionPermissionEvaluator,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_certification import (
    CertificationGates,
    build_local_ray_refit,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_config import (
    APGeneralizedExchangeConfig,
    ExchangeEligibilityConfig,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_selector import (
    GENERALIZED_EXCHANGE_SELECTION_POLICY_V2,
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
    ActiveSupportAtom,
    active_support_atoms,
    appended_origin_atom_labels,
    candidate_append_atoms,
)
from pipelines.time_dynamics.generalized_exchange import (
    EXCHANGE_RANKING_SIGNED_DRIFT,
    GeneralizedExchange,
    GeneralizedExchangeDomain,
    GeneralizedPatch,
    REALIZED_REFUSE,
    REALIZED_RETRY_INSERT_FACE,
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


def _runtime_rotation_coefficient_l1(state: APMcLachlanState) -> np.ndarray:
    """Per runtime coordinate, the L1 coefficient of its Pauli rotations.

    A per-Pauli coordinate has one coefficient.  A shared parent coordinate
    may drive an ordered product of rotations; summing absolute coefficients
    keeps the deletion ray bound conservative by the Fubini--Study triangle
    inequality.
    """

    values: list[float] = []
    for term in _per_coordinate_terms(state):
        specs = iter_runtime_rotation_terms(
            getattr(term, "polynomial"),
            ignore_identity=bool(state.executor.ignore_identity),
            coefficient_tolerance=float(state.executor.coefficient_tolerance),
            sort_terms=bool(state.executor.sort_terms),
        )
        values.append(float(sum(abs(float(spec.coeff_real)) for spec in specs)))
    out = np.asarray(values, dtype=float)
    if out.size != int(state.runtime_parameter_count):
        raise ValueError(
            "runtime rotation coefficients do not match the active support."
        )
    return out


def _build_deletion_permission_evaluator(
    *,
    state: APMcLachlanState,
    evaluation: GeometryEvaluation,
    theta_runtime: np.ndarray,
    support_config: Any,
) -> DeletionPermissionEvaluator:
    exchange_config = APGeneralizedExchangeConfig.from_route_config(support_config)
    return DeletionPermissionEvaluator(
        gram=np.asarray(evaluation.geometry.K, dtype=float),
        force=np.asarray(evaluation.geometry.f, dtype=float),
        norm_b_sq=float(evaluation.geometry.norm_b_sq),
        theta_runtime=np.asarray(theta_runtime, dtype=float),
        rotation_coefficients=_runtime_rotation_coefficient_l1(state),
        ray_distance_max=float(exchange_config.certification.ray_distance_max),
        normalized_schur_loss_max=float(
            exchange_config.certification.velocity_change_max
        ),
        epsilon_norm=float(StructuralScoreWeights().epsilon_norm),
    )


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

    exchange_config = APGeneralizedExchangeConfig.from_route_config(support_config)

    if insertions_enabled:
        raw_atoms = candidate_append_atoms(
            state,
            allow_incomplete_candidate_pool=bool(
                exchange_config.eligibility.allow_incomplete_candidate_pool
            ),
            occurrence_policy=str(exchange_config.eligibility.occurrence_policy),
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

    pool_cap = exchange_config.search.pool_size
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


def eligible_deletion_atoms(
    state: APMcLachlanState,
    *,
    theta_runtime: np.ndarray,
    eligibility_config: ExchangeEligibilityConfig,
    runtime_state: Any,
    time_index: int,
) -> tuple[ActiveSupportAtom, ...]:
    """Return the deletion face allowed by support and protection limits.

    This is structural feasibility only.  It nominates coordinates that may
    appear in ``D``; it does not decide whether deletion is accurate or useful.
    The realized generalized-exchange solve remains the arbiter.
    """

    if not eligibility_config.deletion_enabled:
        return ()

    target_policy = str(eligibility_config.target_policy).strip().lower()
    appended_labels = appended_origin_atom_labels(state)
    active_atoms = tuple(active_support_atoms(state, theta_runtime))
    appended_base_counts: dict[str, int] = {}
    for atom in active_atoms:
        if str(atom.atom_label) not in appended_labels:
            continue
        base_atom_id = str(
            dict(atom.metadata or {}).get("base_atom_id", atom.atom_id)
        )
        appended_base_counts[base_atom_id] = int(
            appended_base_counts.get(base_atom_id, 0) + 1
        )

    out: list[ActiveSupportAtom] = []
    for atom in active_atoms:
        if (
            target_policy in {"appended_only", "redundant_appended_only"}
            and str(atom.atom_label) not in appended_labels
        ):
            continue
        if target_policy == "redundant_appended_only":
            base_atom_id = str(
                dict(atom.metadata or {}).get("base_atom_id", atom.atom_id)
            )
            if int(appended_base_counts.get(base_atom_id, 0)) <= 1:
                continue
        if bool(eligibility_config.protect_drive_aligned):
            identity = " ".join(
                (str(atom.atom_id), str(atom.atom_label), str(atom.parent_label))
            ).lower()
            if "drive_aligned" in identity:
                continue
        if int(state.runtime_parameter_count) - int(atom.runtime_count) < int(
            eligibility_config.minimum_surviving_support
        ):
            continue
        cooldown_until = runtime_state.cooldown_until_index.get(str(atom.atom_id))
        if cooldown_until is not None and int(time_index) < int(cooldown_until):
            continue
        out.append(atom)
    return tuple(out)


def generalized_exchange_domain(
    *,
    state: APMcLachlanState,
    theta_runtime: np.ndarray,
    base_evaluation: GeometryEvaluation,
    base_step: FixedMcLachlanStep,
    inverse_policy: McLachlanInversePolicy,
    support_config: Any,
    runtime_state: Any,
    time_index: int,
) -> tuple[GeneralizedExchangeDomain, tuple[ActiveSupportAtom, ...]]:
    """Derive the searched face from L2 and the existing support limits."""

    checkpoint_l2 = mclachlan_distance_squared(
        base_evaluation, inverse_policy=inverse_policy
    )
    exchange_config = APGeneralizedExchangeConfig.from_route_config(support_config)
    l2_cut = float(exchange_config.score.l2_cut)
    gate_mode = str(exchange_config.eligibility.insertion_gate_mode).strip().lower()
    drift_threshold = getattr(
        exchange_config.eligibility, "accumulated_drift_threshold", None
    )
    accumulated_drift = float(getattr(runtime_state, "accumulated_drift", 0.0))
    if gate_mode == "mclachlan_l2":
        insertion_gate_open = bool(checkpoint_l2 > l2_cut)
    else:
        insertion_gate_open = bool(
            float(base_step.residual_ratio)
            >= float(exchange_config.eligibility.residual_ratio_threshold)
            or (
                drift_threshold is not None
                and accumulated_drift >= float(drift_threshold)
            )
        )

    insertion_cap = int(exchange_config.search.insertion_cardinality)
    deletable_atoms = eligible_deletion_atoms(
        state,
        theta_runtime=theta_runtime,
        eligibility_config=exchange_config.eligibility,
        runtime_state=runtime_state,
        time_index=int(time_index),
    )
    debt_policy = str(exchange_config.score.debt_policy)
    exchange = GeneralizedExchange(
        l2_cut=float(l2_cut),
        debt_policy=debt_policy,
        support_floor=int(exchange_config.eligibility.minimum_surviving_support),
        insertion_cardinality_cap=int(insertion_cap),
        l2_debt_enabled=bool(gate_mode == "mclachlan_l2"),
    )
    domain = exchange.domain(
        checkpoint_l2=float(checkpoint_l2),
        insertion_gate_open=bool(insertion_gate_open),
        deletion_candidate_count=len(deletable_atoms),
    )
    if not domain.deletion_face_open:
        deletable_atoms = ()
    return domain, tuple(deletable_atoms)


def build_selector_inputs(
    *,
    state: APMcLachlanState,
    evaluation: GeometryEvaluation,
    support_config: Any,
    runtime_state: Any,
    theta_runtime: np.ndarray,
    time_index: int,
    deletable_atoms: tuple[ActiveSupportAtom, ...],
    insertions_enabled: bool = True,
    debt_ranking: bool = False,
    deletion_permission_evaluator: DeletionPermissionEvaluator | None = None,
) -> dict[str, Any]:
    """Assemble every selector input from route objects.

    ``deletable_atoms`` has already passed target, protection, cooldown, and
    surviving-support gates in :func:`eligible_deletion_atoms`.
    """

    exchange_config = APGeneralizedExchangeConfig.from_route_config(support_config)
    permission_evaluator = deletion_permission_evaluator or (
        _build_deletion_permission_evaluator(
            state=state,
            evaluation=evaluation,
            theta_runtime=theta_runtime,
            support_config=support_config,
        )
    )

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
        str(exchange_config.eligibility.occurrence_policy).strip().lower()
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
        alpha_ins=float(exchange_config.score.append_cost_alpha),
        alpha_del=float(exchange_config.score.prune_cost_alpha),
        w_delta=float(exchange_config.score.delta_weight),
        lambda_hist=float(exchange_config.score.history_weight),
        lambda_cond_relief=float(exchange_config.score.condition_relief_weight),
        lambda_cond_damage=float(exchange_config.score.condition_damage_weight),
        epsilon_L=float(exchange_config.score.epsilon_loss),
        debt_ranking=bool(debt_ranking),
    )

    max_batch = int(exchange_config.search.insertion_cardinality)

    # Hook #2: historical deletion-loss prior.  ``runtime_state.loss_history``
    # maps stable runtime coordinate labels to ``(time_index, loss)`` records;
    # a deletion set's history term is the mean over its coordinates of each
    # coordinate's windowed mean loss (coordinates without history contribute
    # zero, so the term vanishes until evidence accumulates).
    runtime_labels = tuple(state.runtime_coordinate_labels)
    history_window = max(1, int(exchange_config.score.history_window))
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
        min_surviving_support=int(
            exchange_config.eligibility.minimum_surviving_support
        ),
        cuts_by_atom=cuts_by_atom,
        candidate_pool_for_deletion=candidate_pool_for_deletion,
        insertion_cost=insertion_cost,
        deletion_cost=deletion_cost,
        deletion_permission=permission_evaluator.assess,
        tokens_commute=tokens_commute_from_terms(terms_by_key),
        deletion_history_loss=deletion_history_loss,
        max_insertion_batch_size=max_batch,
        interaction_frontier_widths=exchange_config.search.interaction_frontier_widths,
        max_joint_patch_evaluations=exchange_config.search.joint_patch_evaluations,
    )
    return {
        "atoms_by_id": atoms_by_id,
        "occurrence_label": occurrence_label,
        "structural_kwargs": structural_kwargs,
        "pool_dedup_telemetry": pool_dedup_telemetry,
        "deletion_permission_evaluator": permission_evaluator,
    }


def select_generalized_exchange_patch(
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
    solve_repair_config: Any | None = None,
) -> tuple[ExchangeSelection, dict[str, Any]]:
    """Select one generalized patch from the admissible ``(D, I)`` domain.

    L2, the support floor, the insertion-cardinality cap, target/protection
    gates, and the explicit debt ablation determine the searched face here.
    A materialized candidate under accuracy debt must strictly reduce L2.  If
    a deletion-containing finalist does not, this operation retries its own
    insert-only boundary instead of making the trajectory caller do so.
    """

    domain, deletable_atoms = generalized_exchange_domain(
        state=state,
        theta_runtime=theta_runtime,
        base_evaluation=base_evaluation,
        base_step=base_step,
        inverse_policy=inverse_policy,
        support_config=support_config,
        runtime_state=runtime_state,
        time_index=int(time_index),
    )
    exchange_config = APGeneralizedExchangeConfig.from_route_config(support_config)
    permission_evaluator = _build_deletion_permission_evaluator(
        state=state,
        evaluation=base_evaluation,
        theta_runtime=theta_runtime,
        support_config=support_config,
    )
    gates = CertificationGates(
        ray_distance_max=float(exchange_config.certification.ray_distance_max),
        smoothness_eta_max=float(
            exchange_config.certification.velocity_change_max
        ),
        condition_number_max=exchange_config.certification.condition_number_max,
    )

    drift_threshold = exchange_config.eligibility.accumulated_drift_threshold
    accumulated_drift = float(getattr(runtime_state, "accumulated_drift", 0.0))

    def escalate() -> bool:
        # L2 debt is the authority for the insertion-bearing family.  Without
        # this, a checkpoint in debt can acquire no families merely because a
        # different normalized residual happens to be below its threshold.
        if bool(domain.accuracy_debt):
            return True
        # Otherwise: locally hard checkpoint (residual ratio) OR a trajectory
        # that has banked error while every checkpoint looked locally easy.
        if float(base_step.residual_ratio) >= float(
            exchange_config.eligibility.residual_ratio_threshold
        ):
            return True
        return drift_threshold is not None and accumulated_drift >= float(
            drift_threshold
        )

    # Hook #3: optional bounded local refit toward the frozen checkpoint ray,
    # applied to materialized finalists before the hard gates.  Pure
    # insertions start on the ray and are skipped inside the hook.
    refit = None
    if bool(exchange_config.certification.refit_enabled):
        refit = build_local_ray_refit(
            target_psi=base_evaluation.psi,
            trust_radius=float(exchange_config.certification.refit_trust_radius),
            max_iterations=int(exchange_config.certification.refit_max_iterations),
        )

    def select_once(
        allowed_deletions: tuple[ActiveSupportAtom, ...],
        *,
        insertions_enabled: bool,
    ) -> tuple[ExchangeSelection, dict[str, Any]]:
        inputs = build_selector_inputs(
            state=state,
            evaluation=base_evaluation,
            support_config=support_config,
            runtime_state=runtime_state,
            theta_runtime=theta_runtime,
            time_index=int(time_index),
            deletable_atoms=allowed_deletions,
            insertions_enabled=bool(insertions_enabled),
            debt_ranking=bool(domain.ranking == EXCHANGE_RANKING_SIGNED_DRIFT),
            deletion_permission_evaluator=permission_evaluator,
        )
        inputs["structural_kwargs"]["inverse_policy"] = inverse_policy
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
            score_floor=float(exchange_config.search.structural_score_floor),
            escalate=escalate,
            refit=refit,
            solve_repair_config=solve_repair_config,
            max_certification_attempts_per_level=(
                exchange_config.search.certification_attempts_per_level
            ),
            max_certification_attempts_per_deletion_branch=(
                exchange_config.search.certification_attempts_per_deletion_branch
            ),
        )
        return selection, inputs

    selection, inputs = select_once(
        deletable_atoms if domain.deletion_face_open else (),
        insertions_enabled=domain.insertion_face_open,
    )
    deletion_fallback_used = False

    def realized_l2(result: ExchangeSelection) -> float | None:
        if result.certification is None or result.certification.evaluation is None:
            return None
        return mclachlan_distance_squared(
            result.certification.evaluation, inverse_policy=inverse_policy
        )

    def realized_disposition(result: ExchangeSelection, value: float | None) -> str:
        if result.committed is None or value is None:
            return REALIZED_REFUSE
        return GeneralizedExchange.assess_realized_candidate(
            domain=domain,
            patch=GeneralizedPatch(
                deletions=tuple(result.committed.removed_runtime_indices),
                insertions=tuple(result.committed.inserted_selection),
            ),
            candidate_l2=float(value),
        )

    candidate_l2 = realized_l2(selection)
    disposition = realized_disposition(selection, candidate_l2)
    if disposition == REALIZED_RETRY_INSERT_FACE:
        deletion_fallback_used = True
        first_attempts = selection.attempts
        first_structural_scored_count = int(selection.structural_scored_count)
        selection, inputs = select_once((), insertions_enabled=True)
        selection = ExchangeSelection(
            committed=selection.committed,
            certification=selection.certification,
            attempts=tuple(first_attempts) + tuple(selection.attempts),
            telemetry=selection.telemetry,
            stop_reason=selection.stop_reason,
            structural_scored_count=int(
                first_structural_scored_count + selection.structural_scored_count
            ),
        )
        candidate_l2 = realized_l2(selection)
        disposition = realized_disposition(selection, candidate_l2)

    if selection.committed is not None and disposition == REALIZED_REFUSE:
        selection = ExchangeSelection(
            committed=None,
            certification=None,
            attempts=selection.attempts,
            telemetry=selection.telemetry,
            stop_reason="refused_non_improving_patch_under_l2_debt",
            structural_scored_count=int(selection.structural_scored_count),
        )

    # Record realized deletion losses so hook #2's prior sees this attempt at
    # future checkpoints.  Loss is attributed equally across the deleted
    # coordinates; keys are stable runtime coordinate labels, so survivors
    # keep their record across later structural edits.
    if runtime_state is not None and getattr(runtime_state, "loss_history", None) is not None:
        window = max(1, int(exchange_config.score.history_window))
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
        "selection_policy": GENERALIZED_EXCHANGE_SELECTION_POLICY_V2,
        "generalized_exchange_domain": domain.to_json_dict(),
        "deletion_fallback_used": bool(deletion_fallback_used),
        "realized_candidate_l2": (
            None if candidate_l2 is None else float(candidate_l2)
        ),
        **inputs.get("pool_dedup_telemetry", {}),
        "kind": selection.kind,
        "stop_reason": selection.stop_reason,
        "attempt_count": len(selection.attempts),
        "structural_scored_count": int(selection.structural_scored_count),
        "attempts": [a.to_json_dict() for a in selection.attempts[-16:]],
        "cost_normalization": "per_candidate_raw_v1",
        "deletion_permission": permission_evaluator.summary(),
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
    payload["accumulated_drift"] = float(accumulated_drift)
    payload["escalation_accumulated_drift_threshold"] = (
        None if drift_threshold is None else float(drift_threshold)
    )
    payload["escalation_source"] = (
        "residual_ratio"
        if float(base_step.residual_ratio)
        >= float(support_config.residual_ratio_threshold)
        else (
            "accumulated_drift"
            if drift_threshold is not None
            and accumulated_drift >= float(drift_threshold)
            else "none"
        )
    )
    return selection, payload


__all__ = [
    "build_candidate_pool",
    "build_selector_inputs",
    "eligible_deletion_atoms",
    "generalized_exchange_domain",
    "select_generalized_exchange_patch",
]
