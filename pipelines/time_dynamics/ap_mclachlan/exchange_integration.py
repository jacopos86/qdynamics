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

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
import time
from typing import Any, Mapping

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.append_cost import (
    AP_APPEND_COST_NORMALIZATION_FAMILY_ROBUST_V1,
    AppendCostSettings,
    append_cost_telemetry_for_family,
    estimate_append_atom_set_cost,
)
from pipelines.time_dynamics.ap_mclachlan.avqds import (
    mclachlan_distance_squared,
)
from pipelines.time_dynamics.ap_mclachlan.compiled_patch_cost import CompilerSession
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
    ExchangeRouteBundle,
    ExchangeRouteConfig,
    permission_ray_distance_max,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_selector import (
    GENERALIZED_EXCHANGE_SELECTION_POLICY_V2,
    ExchangeSelection,
    RankInfo,
    RankedLevel,
    select_exchange_patch,
)
from pipelines.time_dynamics.ap_mclachlan.patch_evaluation import (
    CandidateCostInputs,
    StructuralCandidate,
    StructuralScoreWeights,
)
from pipelines.time_dynamics.ap_mclachlan.objective_ordinates import (
    DEFAULT_GAMMA_PINV_RCOND,
)
from pipelines.time_dynamics.ap_mclachlan.performance import phase
from pipelines.time_dynamics.ap_mclachlan.population_scoring import (
    CompiledResourceMarginal,
    PopulationCostNormalization,
    RankRegime,
    ScoreRecord,
    legacy_family_robust_denominators,
    score_and_rank,
)
from pipelines.time_dynamics.ap_mclachlan.fixed_step import FixedMcLachlanStep
from pipelines.time_dynamics.ap_mclachlan.geometry import residual_denominator
from pipelines.time_dynamics.ap_mclachlan.geometry_eval import GeometryEvaluation
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import TimeDependentHamiltonian
from pipelines.time_dynamics.ap_mclachlan.insertion_words import (
    tokens_commute_from_terms,
)
from pipelines.time_dynamics.ap_mclachlan.inverse import (
    McLachlanInversePolicy,
    solve_theta_dot,
)
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


# BLAS-thread environment recorded with every selection's parallelism
# telemetry (same variable set the runtime-benchmark harness pins/records).
# A worker fan-out multiplies against whatever ambient BLAS threading these
# describe, so the receipt must carry them for oversubscription audits.
BLAS_THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
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
        ray_distance_max=permission_ray_distance_max(exchange_config),
        normalized_schur_loss_max=float(
            exchange_config.certification.velocity_change_max
        ),
        epsilon_norm=float(StructuralScoreWeights().epsilon_norm),
    )


PLACEMENT_RULE_ID = "paper_ii_min_weighted_raw_marginal_placement_v1"


class CompiledCostRuntime:
    """Per-decision compiled-cost adapter for the compiled bundles.

    Resolves the sealed backend once per process, compiles the base word
    once per checkpoint, chooses one materialization cut per (atom, retained
    cut) inside its commutation class, and compiles every complete patch
    exactly once through a word-identity cache.  Failures raise; there is no
    proxy or ``K = 1`` fallback.
    """

    def __init__(
        self,
        *,
        state: APMcLachlanState,
        atoms_by_id: Mapping[str, Any],
        cuts_by_atom: Mapping[str, Sequence[int]],
        blocks: Sequence[Any],
        contract: Any,
        workers: int = 1,
        compiler_session: CompilerSession | None = None,
    ) -> None:
        from pipelines.time_dynamics.ap_mclachlan import compiled_patch_cost as cpc

        self._cpc = cpc
        self.state = state
        self.atoms_by_id = dict(atoms_by_id)
        self.cuts_by_atom = {k: tuple(int(c) for c in v) for k, v in cuts_by_atom.items()}
        self.blocks = tuple(blocks)
        self.contract = contract
        self.backend, self.backend_provenance = _resolve_compile_backend_cached(contract)
        # SIAM seeds start from an empty ansatz, where ``executor`` carries no
        # qubit count; ``state_num_qubits`` falls back to log2 of the
        # reference-state dimension.  Reading ``executor.nq`` directly made
        # every compiled-cost bundle raise at the first checkpoint.
        self.nq = int(cpc.state_num_qubits(state))
        self.ref_state = np.asarray(state.psi_ref, dtype=complex).reshape(-1)
        self.base_terms = tuple(cpc.base_word(state))
        self.base_word_sha256 = cpc.word_key(
            self.base_terms,
            nq=self.nq,
            ref_state=self.ref_state,
            contract=contract,
        )
        self.workers = max(1, int(workers))
        self.compiler_session = (
            compiler_session if compiler_session is not None else CompilerSession()
        )
        self._owns_compiler_session = compiler_session is None
        self._closed = False
        self.pool = None
        self.pool_restarts = 0
        self.pool_failures: list[str] = []
        self.pinned_env: dict[str, str] = {}
        self.prior_env: dict[str, str | None] = {}
        self.warm_primed = 0
        self.warm_seconds = 0.0
        self.seconds = 0.0
        self._compiler_work_start = self.compiler_session.work_receipt()
        self.cache = self.compiler_session.cache_for_decision(
            contract=contract, nq=self.nq, ref_state=self.ref_state,
            workers=self.workers, backend_identity=self.backend_provenance,
            word_compiler=self._compile_word_isolated,
        )
        start = time.perf_counter()
        try:
            with phase("patch.exchange.compile"):
                self.base_triple = self.cache.compile_word(
                    self.base_terms,
                    nq=self.nq,
                    ref_state=self.ref_state,
                    contract=contract,
                    backend_obj=self.backend,
                )
        except BaseException:
            self.close()
            raise
        self.seconds += time.perf_counter() - start
        self._placements: dict[tuple[str, int], Any] = {}
        self._class_members: dict[str, dict[int, tuple[int, ...]]] = {}

    def _ensure_pool(self) -> Any:
        if self._closed:
            raise self._cpc.CompiledPatchCostError("compiled-cost runtime is closed.")
        if self.pool is None:
            self.pool = self.compiler_session.lease(
                contract=self.contract,
                nq=self.nq,
                ref_state=self.ref_state,
                workers=self.workers,
            )
            self.pinned_env = dict(self.pool.pinned_env)
            self.prior_env = dict(self.pool.prior_env)
        return self.pool

    def _compile_word_isolated(self, terms: Sequence[Any], **kwargs: Any) -> Any:
        pool = self._ensure_pool()
        try:
            return pool.compile_word(terms, **kwargs)
        except self._cpc.CompiledPatchCostError as exc:
            self.pool_failures.append(f"{type(exc).__name__}:{exc}")
            if getattr(pool, "_terminated", False):
                self.pool = None
                self.pool_restarts += 1
            raise

    def placement(self, atom_id: str, retained_cut: int) -> Any:
        key = (str(atom_id), int(retained_cut))
        cached = self._placements.get(key)
        if cached is not None:
            return cached
        atom = self.atoms_by_id[str(atom_id)]
        members_by_cut = self._class_members.get(str(atom_id))
        if members_by_cut is None:
            members_by_cut = self._cpc.insertion_class_members(atom.term, self.blocks)
            self._class_members[str(atom_id)] = members_by_cut
        members = members_by_cut[int(retained_cut)]
        start = time.perf_counter()
        with phase("patch.exchange.compile"):
            placement = self._cpc.choose_placement(
                atom.term,
                members,
                base_terms=self.base_terms,
                nq=self.nq,
                ref_state=self.ref_state,
                contract=self.contract,
                backend_obj=self.backend,
                cache=self.cache,
            )
        self.seconds += time.perf_counter() - start
        self._placements[key] = placement
        return placement

    def cost_inputs(
        self,
        removed: tuple[int, ...],
        selection: tuple[tuple[str, int], ...],
        plan: Any,
    ) -> CandidateCostInputs:
        calls_before = int(self.cache.compile_calls)
        entries: list[tuple[int, Any, str]] = []
        placement_cuts: list[tuple[str, int]] = []
        for ordinal, (atom_id, cut) in enumerate(selection):
            placement = self.placement(str(atom_id), int(cut))
            chosen = int(placement.chosen_cut)
            placement_cuts.append((str(atom_id), chosen))
            entries.append((chosen, self.atoms_by_id[str(atom_id)].term, f"{atom_id}@{chosen}#{ordinal}"))
        start = time.perf_counter()
        with phase("patch.exchange.compile"):
            record = self._cpc.compile_complete_patch(
                self.state,
                removed_runtime_indices=tuple(int(i) for i in removed),
                insertions=tuple(entries),
                contract=self.contract,
                backend_obj=self.backend,
                cache=self.cache,
                base_triple=self.base_triple,
            )
        self.seconds += time.perf_counter() - start
        marginal = record.marginal
        return CandidateCostInputs(
            cost_marginal=(int(marginal.n2q), int(marginal.d2q), int(marginal.dc)),
            placement_cuts=tuple(placement_cuts),
            patch_word_sha256=str(record.word_sha256),
            compile_calls=int(self.cache.compile_calls) - calls_before,
        )

    def validates_materialized_word(
        self,
        candidate: StructuralCandidate,
        insertions: Sequence[tuple[int, Any, str]],
    ) -> bool:
        if candidate.patch_word_sha256 is None:
            return False
        word = self._cpc.structural_word(
            self.state,
            candidate.removed_runtime_indices,
            insertions,
        )
        observed = self._cpc.word_key(
            word,
            nq=self.nq,
            ref_state=self.ref_state,
            contract=self.contract,
        )
        return bool(str(observed) == str(candidate.patch_word_sha256))

    def _members_for(self, atom_id: str) -> dict[int, tuple[int, ...]]:
        members_by_cut = self._class_members.get(str(atom_id))
        if members_by_cut is None:
            atom = self.atoms_by_id[str(atom_id)]
            members_by_cut = self._cpc.insertion_class_members(atom.term, self.blocks)
            self._class_members[str(atom_id)] = members_by_cut
        return members_by_cut

    def warm(self, items: Sequence[tuple[tuple[int, ...], tuple[tuple[str, int], ...], Any]]) -> int:
        """Compile every word one family will need, in a process pool.

        Two ordered fan-outs per family: the class-member trial words behind
        every not-yet-chosen placement, then (after placements are fixed
        serially from the primed cache) the complete-patch words.  Returns
        the number of words primed; with one worker this is a no-op and the
        serial path compiles on demand.
        """

        if self.workers <= 1:
            return 0
        start = time.perf_counter()
        pool = self._ensure_pool()
        needed = sorted(
            {(str(a), int(c)) for _removed, selection, _plan in items for a, c in selection}
            - set(self._placements)
        )
        trial_jobs: list[tuple[str, tuple]] = []
        for atom_id, cut in needed:
            term = self.atoms_by_id[atom_id].term
            for position in self._members_for(atom_id)[int(cut)]:
                word = self.base_terms[: int(position)] + (term,) + self.base_terms[int(position) :]
                trial_jobs.append(
                    (
                        self._cpc.word_key(
                            word, nq=self.nq, ref_state=self.ref_state, contract=self.contract
                        ),
                        ("word", word),
                    )
                )
        try:
            primed = self._cpc.warm_word_cache(self.cache, pool, trial_jobs)
        except self._cpc.CompiledPatchCostError as exc:
            self.pool_failures.append(f"warm_trial:{type(exc).__name__}:{exc}")
            if getattr(pool, "_terminated", False):
                self.pool = None
                self.pool_restarts += 1
            self.warm_seconds += time.perf_counter() - start
            return 0
        for atom_id, cut in needed:
            self.placement(atom_id, cut)
        patch_jobs: list[tuple[str, tuple]] = []
        for removed, selection, _plan in items:
            if not removed and not selection:
                continue
            entries = []
            for ordinal, (atom_id, cut) in enumerate(selection):
                chosen = int(self.placement(str(atom_id), int(cut)).chosen_cut)
                entries.append((chosen, str(atom_id), f"{atom_id}@{chosen}#{ordinal}"))
            word = self._cpc.edit_word(
                self.base_terms,
                tuple(int(i) for i in removed),
                [(cut, self.atoms_by_id[a].term, label) for cut, a, label in entries],
            )
            patch_jobs.append(
                (
                    self._cpc.word_key(
                        word, nq=self.nq, ref_state=self.ref_state, contract=self.contract
                    ),
                    ("word", word),
                )
            )
        try:
            primed += self._cpc.warm_word_cache(self.cache, pool, patch_jobs)
        except self._cpc.CompiledPatchCostError as exc:
            self.pool_failures.append(f"warm_patch:{type(exc).__name__}:{exc}")
            if getattr(pool, "_terminated", False):
                self.pool = None
                self.pool_restarts += 1
        self.warm_primed += int(primed)
        self.warm_seconds += time.perf_counter() - start
        return int(primed)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.cache.close()
        if self.pool is not None:
            self.pool.close()
            self.pool = None
        if self._owns_compiler_session:
            self.compiler_session.close()

    def telemetry(self) -> dict[str, Any]:
        work_total = self.compiler_session.work_receipt()
        work = {
            key: value - self._compiler_work_start[key]
            for key, value in work_total.items() if key != "jobs_unresolved"
        }
        # Queue occupancy is derived, not monotonic: do not subtract two
        # unresolved counts as though they were cumulative work counters.
        work["jobs_unresolved"] = max(
            0, work["jobs_submitted"] - work["jobs_succeeded"] - work["jobs_failed"]
        )
        return {
            "placement_rule": PLACEMENT_RULE_ID,
            "compile_workers": int(self.workers),
            "compile_execution_mode": (
                "spawn_pool_serial"
                if self.workers <= 1
                else "spawn_pool_warm_then_serial_replay"
            ),
            "warm_primed_words": int(self.warm_primed),
            "warm_seconds": float(self.warm_seconds),
            "worker_blas_env_pinned": (
                dict(self.pinned_env)
            ),
            "parent_blas_env_before_pool": dict(self.prior_env),
            "compile_timeout_seconds": float(self.contract.compile_timeout_seconds),
            "pool_restarts": int(self.pool_restarts),
            "pool_failures": list(self.pool_failures),
            "backend": self.backend_provenance,
            "base_resources": {
                "n2q": int(self.base_triple.n2q),
                "d2q": int(self.base_triple.d2q),
                "dc": int(self.base_triple.dc),
            },
            "base_word_sha256": str(self.base_word_sha256),
            # Successful worker outcomes, including a failed warm's completed
            # work; cache publication and logical patch slots are distinct.
            "compile_calls": int(work["jobs_succeeded"]),
            "compiler_work": work,
            "compiler_work_total": work_total,
            "cache_compilations_published": int(self.cache.compile_calls),
            "cache_hits": int(self.cache.cache_hits),
            "shared_cache_hits": int(self.cache.shared_cache_hits),
            "session_result_cache": self.compiler_session.cache_receipt(),
            "placements_chosen": int(len(self._placements)),
            "compile_seconds": float(self.seconds),
        }


_BACKEND_CACHE: dict[tuple, tuple[Any, dict[str, Any]]] = {}


def _resolve_compile_backend_cached(contract: Any) -> tuple[Any, dict[str, Any]]:
    from pipelines.time_dynamics.ap_mclachlan import compiled_patch_cost as cpc

    key = (
        str(contract.backend_name),
        int(contract.optimization_level),
        int(contract.transpile_seed),
    )
    cached = _BACKEND_CACHE.get(key)
    if cached is None:
        cached = cpc.resolve_compile_backend(contract)
        _BACKEND_CACHE[key] = cached
    return cached


def _legacy_proxy_cost_inputs(
    *,
    atoms_by_id: Mapping[str, Any],
    active_by_index: Mapping[int, Any],
):
    """Raw Paper-I proxy components per patch, for population repair."""

    def cost_inputs(
        removed: tuple[int, ...],
        selection: tuple[tuple[str, int], ...],
        plan: Any,
    ) -> CandidateCostInputs:
        insert_components = None
        delete_components = None
        if selection:
            atom_ids = tuple(atom_id for atom_id, _cut in selection)
            raw = estimate_append_atom_set_cost(tuple(atoms_by_id[str(a)] for a in atom_ids))
            insert_components = dict(raw.raw_components)
        if removed:
            selected = tuple(
                active_by_index[int(i)] for i in removed if int(i) in active_by_index
            )
            if selected:
                delete_components = dict(estimate_prune_atom_set_cost(selected).raw_components)
        return CandidateCostInputs(
            proxy_insert_components=insert_components,
            proxy_delete_components=delete_components,
        )

    return cost_inputs


def _bounded_rows(rows: Sequence[Mapping[str, Any]], limit: int = 24) -> list:
    return [dict(row) for row in list(rows)[: int(limit)]]


def build_population_ranker(
    *,
    bundle: ExchangeRouteBundle,
    route: ExchangeRouteConfig,
    accuracy_debt: bool,
    weights: StructuralScoreWeights,
    score_floor: float,
    cost_settings: AppendCostSettings,
    population_context: Mapping[str, Any] | None = None,
):
    """Seal one certification level and return its certification order.

    Compiled bundles: population-relative ``K`` over every evaluated
    candidate (before sign filtering), then the shifted Paper-I score under
    the regime fixed by realized ``L^2`` (large miss while in accuracy debt,
    maintenance otherwise).  The legacy repair keeps the historical score and
    debt ordering but normalizes the proxy denominators over the actual
    competing population instead of a singleton.
    """

    regime = RankRegime.LARGE_MISS if bool(accuracy_debt) else RankRegime.MAINTENANCE

    def merit_of(candidate: StructuralCandidate) -> float | None:
        if bundle.needs_gamma:
            return candidate.m_gamma
        return float(candidate.delta)

    def ranker(level_candidates: Sequence[StructuralCandidate]) -> RankedLevel:
        non_stay = [c for c in level_candidates if c.kind != "stay"]
        competing = [
            c
            for c in non_stay
            if (
                bool(c.inserted_selection)
                if regime is RankRegime.LARGE_MISS
                else bool(c.removed_runtime_indices) and not bool(c.inserted_selection)
            )
        ]
        by_key = {c.order_key: c for c in competing}
        if bundle.needs_compiled_cost:
            evaluated = [
                c
                for c in competing
                if c.cost_marginal is not None
                and np.isfinite(float(c.delta))
                and c.ordinate_error is None
                and (
                    (c.m_gamma is not None and np.isfinite(float(c.m_gamma)))
                    or (
                        not bundle.needs_gamma
                        and c.gain_evaluation == "singleton_factorized"
                        and c.joint_validation_required
                    )
                )
            ]
            if not evaluated:
                return RankedLevel(
                    ordered=(),
                    rank_info={},
                    telemetry={
                        "kernel": bundle.kernel_id,
                        "regime": regime.value,
                        "population_size": 0,
                        "competing_candidate_count": int(len(competing)),
                        "compiler_rejections": [
                            {
                                "patch": [
                                    list(c.order_key[0]),
                                    [list(x) for x in c.order_key[1]],
                                ],
                                "reason": str(c.cost_error),
                            }
                            for c in competing
                            if c.cost_error is not None
                        ],
                        "ordinate_rejections": [
                            {
                                "patch": [
                                    list(c.order_key[0]),
                                    [list(x) for x in c.order_key[1]],
                                ],
                                "reason": str(c.ordinate_error),
                            }
                            for c in competing
                            if c.ordinate_error is not None
                        ],
                    },
                )
            sealed = PopulationCostNormalization().seal(
                [(c.order_key, CompiledResourceMarginal(*c.cost_marginal)) for c in evaluated]
            )
            records = [
                ScoreRecord(
                    patch_id=c.order_key,
                    merit=float(merit_of(c)),
                    K=float(sealed.k_of(c.order_key)),
                    raw_merit_tiebreak=float(merit_of(c)),
                    canonical_key=c.order_key,
                )
                for c in evaluated
            ]
            result = score_and_rank(
                records, regime, tau_del=float(route.maintenance_loss_tolerance)
            )
            ordered = tuple(by_key[r.patch_id] for r in result.ranked)
            rank_info = {
                r.patch_id: RankInfo(K=float(r.K), score=float(r.score), merit=float(r.merit))
                for r in result.ranked
            }
            excluded_counts: dict[str, int] = {}
            for ex in result.excluded:
                excluded_counts[str(ex.reason)] = excluded_counts.get(str(ex.reason), 0) + 1
            population_json = sealed.to_json_dict()
            cost_receipt = {
                "schema": "paper_ii_cost_population_receipt_v1",
                "context": dict(population_context or {}),
                "normalization": population_json,
            }
            cost_population_sha256 = hashlib.sha256(
                json.dumps(
                    cost_receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode("utf-8")
            ).hexdigest()
            ranked_rows = [
                {
                    "patch": [list(r.patch_id[0]), [list(x) for x in r.patch_id[1]]],
                    "merit": float(r.merit),
                    "K": float(r.K),
                    "score": float(r.score),
                    "rank_key": list(r.rank_key[:-1]),
                }
                for r in result.ranked
            ]
            excluded_rows = [
                {
                    "patch": [list(ex.patch_id[0]), [list(x) for x in ex.patch_id[1]]],
                    "merit": float(ex.merit),
                    "K": float(ex.K),
                    "score": None if ex.score is None else float(ex.score),
                    "reason": str(ex.reason),
                }
                for ex in result.excluded
            ]
            decision_payload = {
                "cost_population_sha256": cost_population_sha256,
                "objective": bundle.objective_id,
                "kernel": bundle.kernel_id,
                "regime": regime.value,
                "tau": float(result.tau),
                "ranked": ranked_rows,
                "excluded": excluded_rows,
            }
            decision_sha256 = hashlib.sha256(
                json.dumps(
                    decision_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            telemetry = {
                "kernel": bundle.kernel_id,
                "objective": bundle.objective_id,
                "cost_model": bundle.cost_model_id,
                "regime": regime.value,
                "tau": float(result.tau),
                "population_size": int(sealed.size),
                "normalization_population_sha256": str(
                    population_json.get("population_sha256")
                ),
                "population_sha256": str(population_json.get("population_sha256")),
                "cost_population_sha256": cost_population_sha256,
                "decision_receipt_sha256": decision_sha256,
                "population_receipt": cost_receipt,
                "medians": dict(sealed.medians),
                "mads": dict(sealed.mads),
                "mad_scales": dict(sealed.scales),
                "scales": dict(sealed.scales),
                "ranked_count": int(len(result.ranked)),
                "excluded_counts": excluded_counts,
                "ranked": ranked_rows,
                "excluded": excluded_rows,
                "compiler_rejections": [
                    {
                        "patch": [
                            list(c.order_key[0]),
                            [list(x) for x in c.order_key[1]],
                        ],
                        "reason": str(c.cost_error),
                    }
                    for c in competing
                    if c.cost_error is not None
                ],
                "ordinate_rejections": [
                    {
                        "patch": [
                            list(c.order_key[0]),
                            [list(x) for x in c.order_key[1]],
                        ],
                        "reason": str(c.ordinate_error),
                    }
                    for c in competing
                    if c.ordinate_error is not None
                ],
                "top": ranked_rows[:24],
            }
            return RankedLevel(ordered=ordered, rank_info=rank_info, telemetry=telemetry)

        # Legacy proxy, normalized over the actual competing population.
        insert_pop = [
            (c.order_key, c.proxy_insert_components)
            for c in competing
            if c.proxy_insert_components is not None
        ]
        delete_pop = [
            (c.order_key, c.proxy_delete_components)
            for c in competing
            if c.proxy_delete_components is not None
        ]
        c_ins = {
            d.patch_id: max(1.0, float(d.hardware_cost_denominator))
            for d in (
                legacy_family_robust_denominators(insert_pop, settings=cost_settings)
                if insert_pop
                else ()
            )
        }
        c_del = {
            d.patch_id: max(1.0, float(d.hardware_cost_denominator))
            for d in (
                legacy_family_robust_denominators(delete_pop, settings=cost_settings)
                if delete_pop
                else ()
            )
        }
        rows = []
        for c in competing:
            cost_i = c_ins.get(c.order_key, 1.0)
            cost_d = c_del.get(c.order_key, 1.0)
            # The historical utilities were computed with C == 1, so the
            # population denominators enter as exact rescalings.
            insertion_utility = float(c.insertion_utility) / (cost_i ** float(weights.alpha_ins))
            deletion_utility = float(c.deletion_utility) * (cost_d ** float(weights.alpha_del))
            score = insertion_utility + deletion_utility + float(weights.w_delta) * float(c.delta)
            if not np.isfinite(score) or float(score) <= float(score_floor):
                continue
            if bool(weights.debt_ranking):
                tol = float(weights.delta_rank_tolerance)
                primary = float(round(float(c.delta) / tol)) if tol > 0.0 else float(c.delta)
                secondary = float(np.tanh(insertion_utility + deletion_utility))
            else:
                primary = float(score)
                secondary = 0.0
            rows.append((primary, secondary, c, score, cost_i if c.inserted_selection else cost_d))
        rows.sort(key=lambda row: (-row[0], -row[1], row[2].order_key))
        ordered = tuple(row[2] for row in rows)
        rank_info = {
            row[2].order_key: RankInfo(K=float(row[4]), score=float(row[3]), merit=float(row[2].delta))
            for row in rows
        }
        telemetry = {
            "kernel": bundle.kernel_id,
            "objective": bundle.objective_id,
            "cost_model": bundle.cost_model_id,
            "regime": regime.value,
            "population_size": int(len(competing)),
            "insert_population_size": int(len(insert_pop)),
            "delete_population_size": int(len(delete_pop)),
            "ranked_count": int(len(rows)),
            "distinct_insert_denominators": sorted({round(v, 12) for v in c_ins.values()}),
            "distinct_delete_denominators": sorted({round(v, 12) for v in c_del.values()}),
            "top": _bounded_rows(
                [
                    {
                        "patch": [list(row[2].order_key[0]), [list(x) for x in row[2].order_key[1]]],
                        "score": float(row[3]),
                        "denominator": float(row[4]),
                        "delta": float(row[2].delta),
                    }
                    for row in rows
                ]
            ),
        }
        return RankedLevel(ordered=ordered, rank_info=rank_info, telemetry=telemetry)

    return ranker


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
    only rescales theta). Every eligible direction is retained in frozen order.

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

    telemetry = {
        "candidate_pool_raw": int(len(raw_atoms)),
        "candidate_pool_deduplicated": int(len(atoms)),
        "candidate_pool_order": "frozen_pool_order",
        "append_occurrence_policy": str(
            exchange_config.eligibility.occurrence_policy
        ),
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
    accuracy_debt: bool = False,
    deletion_permission_evaluator: DeletionPermissionEvaluator | None = None,
    compiled_cost_runtime: CompiledCostRuntime | None = None,
    compiler_session: CompilerSession | None = None,
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
        theta_runtime=np.asarray(theta_runtime, dtype=float),
        evaluation=evaluation,
        cuts_by_atom=cuts_by_atom,
        atoms_by_id=atoms_by_id,
        checkpoint_key=(int(time_index), state.runtime_coordinate_labels),
        lazy_candidate_pairs=(exchange_config.search.search_policy != "family_frontiers"),
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

    if (
        cost_settings.cost_normalization_mode
        == AP_APPEND_COST_NORMALIZATION_FAMILY_ROBUST_V1
    ):
        # A singleton family is its own median: every normalized component
        # vanishes, so both proxy denominators are exactly one. Select the
        # identity before estimating raw costs; population repair below still
        # acquires its raw components over the actual competing population.
        def unit_proxy_cost(_selection: tuple[Any, ...]) -> float:
            return 1.0

        insertion_cost = unit_proxy_cost
        deletion_cost = unit_proxy_cost
    else:
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
        if state.parameterization_mode == "logical_shared":
            return (
                f"{atom.atom_label}::insr{int(time_index)}c{int(cut)}o{int(ordinal)}"
                "::logical::generator"
            )
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
        max_deletion_batch_size=exchange_config.search.deletion_cardinality,
        interaction_frontier_widths=exchange_config.search.interaction_frontier_widths,
        max_joint_patch_evaluations=exchange_config.search.joint_patch_evaluations,
        scoring_workers=int(exchange_config.search.scoring_workers),
        exchange_insertion_frontier=(
            exchange_config.search.exchange_insertion_frontier
        ),
    )

    if exchange_config.search.search_policy != "family_frontiers":
        structural_kwargs["patch_rung_schedule"] = exchange_config.search.patch_rung_schedule
    route = exchange_config.route
    bundle = route.bundle
    cost_runtime = compiled_cost_runtime
    if bundle.uses_population_scoring:
        gamma_rcond = (
            DEFAULT_GAMMA_PINV_RCOND
            if (bundle.needs_gamma or bundle.needs_compiled_cost)
            else None
        )
        if bundle.needs_compiled_cost:
            if cost_runtime is None:
                cost_runtime = CompiledCostRuntime(
                    state=state,
                    atoms_by_id=atoms_by_id,
                    cuts_by_atom=cuts_by_atom,
                    blocks=blocks,
                    contract=bundle.compiler,
                    workers=int(route.compile_workers),
                    compiler_session=compiler_session,
                )
            cost_inputs = cost_runtime.cost_inputs
            if cost_runtime.workers > 1:
                structural_kwargs["candidate_cost_warm"] = cost_runtime.warm
        else:
            cost_inputs = _legacy_proxy_cost_inputs(
                atoms_by_id=atoms_by_id, active_by_index=active_by_index
            )
        structural_kwargs.update(
            route_bundle=bundle,
            gamma_rcond=gamma_rcond,
            candidate_cost_inputs=cost_inputs,
            max_compiled_patches_per_checkpoint=route.max_compiled_patches_per_checkpoint,
            population_accuracy_debt=bool(accuracy_debt),
        )
    return {
        "atoms_by_id": atoms_by_id,
        "occurrence_label": occurrence_label,
        "structural_kwargs": structural_kwargs,
        "pool_dedup_telemetry": pool_dedup_telemetry,
        "deletion_permission_evaluator": permission_evaluator,
        "weights": weights,
        "cost_settings": cost_settings,
        "compiled_cost_runtime": cost_runtime,
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
    compiler_session: CompilerSession | None = None,
) -> tuple[ExchangeSelection, dict[str, Any]]:
    """Select one generalized patch from the admissible ``(D, I)`` domain.

    L2, the support floor, insertion and optional deletion cardinality caps,
    target/protection gates, and the explicit debt ablation determine the
    searched face here.
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

    route = exchange_config.route
    bundle = route.bundle
    shared_cost_runtime: dict[str, Any] = {"runtime": None}
    selector_passes: list[dict[str, Any]] = []

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
            accuracy_debt=bool(domain.accuracy_debt),
            deletion_permission_evaluator=permission_evaluator,
            compiled_cost_runtime=shared_cost_runtime["runtime"],
            compiler_session=compiler_session,
        )
        shared_cost_runtime["runtime"] = inputs.get("compiled_cost_runtime")
        cache = inputs["structural_kwargs"]["cache"]
        pre_support = int(state.runtime_parameter_count)
        positioned_columns = int(cache.column_count)
        selector_passes.append(
            {
                "pass_index": int(len(selector_passes)),
                "pre_runtime_parameter_count": int(pre_support),
                "positioned_candidate_column_count": int(positioned_columns),
                "candidate_pool_count": int(len(inputs["atoms_by_id"])),
                "allowed_deletion_atom_count": int(len(allowed_deletions)),
                "insertions_enabled": bool(insertions_enabled),
                "logical_shared_cache_entry_count": int(
                    pre_support * positioned_columns
                    + positioned_columns * (positioned_columns + 3) // 2
                ),
                "complete": True,
            }
        )
        inputs["structural_kwargs"]["inverse_policy"] = inverse_policy
        population_ranker = None
        realized_l2_gate = None
        realized_l2_gate_reason = "realized_l2_not_improved"
        candidate_word_validator = None
        if bundle.uses_population_scoring:
            runtime = inputs.get("compiled_cost_runtime")
            population_context = {
                "schema": "paper_ii_cost_population_context_v1",
                "checkpoint_index": int(time_index),
                "checkpoint_time": float(time),
                "regime": (
                    "large_miss" if bool(domain.accuracy_debt) else "maintenance"
                ),
                "cost_model": bundle.cost_model_id,
                "base_word_sha256": (
                    None if runtime is None else str(runtime.base_word_sha256)
                ),
                "base_support_labels_sha256": hashlib.sha256(
                    json.dumps(
                        list(state.runtime_coordinate_labels),
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "compiler": (
                    None if bundle.compiler is None else asdict(bundle.compiler)
                ),
            }
            population_ranker = build_population_ranker(
                bundle=bundle,
                route=route,
                accuracy_debt=bool(domain.accuracy_debt),
                weights=inputs["weights"],
                score_floor=float(exchange_config.search.structural_score_floor),
                cost_settings=inputs["cost_settings"],
                population_context=population_context,
            )
            if bool(domain.accuracy_debt):
                checkpoint_l2 = float(domain.checkpoint_l2)

                def realized_l2_gate(result: Any) -> bool:
                    if result.evaluation is None:
                        return False
                    return bool(
                        mclachlan_distance_squared(
                            result.evaluation, inverse_policy=inverse_policy
                        )
                        < checkpoint_l2
                    )
            else:
                base_solve = solve_theta_dot(
                    np.asarray(base_evaluation.geometry.K, dtype=float),
                    np.asarray(base_evaluation.geometry.f, dtype=float),
                    policy=inverse_policy,
                )
                base_q = float(base_solve.captured_drift)
                denominator = residual_denominator(
                    float(base_evaluation.geometry.norm_b_sq),
                    float(inputs["weights"].epsilon_norm),
                )
                tau_del = float(route.maintenance_loss_tolerance)
                realized_l2_gate_reason = "realized_q_loss_budget_exceeded"

                def realized_l2_gate(result: Any) -> bool:
                    if result.evaluation is None:
                        return False
                    candidate_solve = solve_theta_dot(
                        np.asarray(result.evaluation.geometry.K, dtype=float),
                        np.asarray(result.evaluation.geometry.f, dtype=float),
                        policy=inverse_policy,
                    )
                    merit = float(
                        (float(candidate_solve.captured_drift) - base_q) / denominator
                    )
                    return bool(merit >= -tau_del)

            if runtime is not None:
                candidate_word_validator = runtime.validates_materialized_word

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
            population_ranker=population_ranker,
            realized_l2_gate=realized_l2_gate,
            realized_l2_gate_reason=realized_l2_gate_reason,
            candidate_word_validator=candidate_word_validator,
            search_policy=exchange_config.search.search_policy,
            gain_evaluation=exchange_config.search.gain_evaluation,
        )
        if exchange_config.search.search_policy != "family_frontiers":
            # Each pass owns a fresh tangent cache. Finalize after frozen-ray
            # validation so its newly acquired pairs count even on rejection.
            acquisition = cache.acquisition_receipt()
            selector_passes[-1]["structural_acquisition"] = acquisition
            selector_passes[-1]["logical_shared_cache_entry_count"] = int(
                acquisition["logical_total_estimator_entries"]
            )
            selector_passes[-1]["complete"] = all(
                bool(stage["complete"]) for stage in acquisition["stages"].values()
            )
        return selection, inputs

    try:
        selection, inputs = select_once(
            deletable_atoms if domain.deletion_face_open else (),
            insertions_enabled=bool(
                domain.insertion_face_open
                and (not bundle.uses_population_scoring or domain.accuracy_debt)
            ),
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
            first_levels = tuple(selection.level_telemetry)
            first_certification_work = tuple(selection.certification_work)
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
                level_telemetry=first_levels + tuple(selection.level_telemetry),
                certification_work=(
                    first_certification_work + tuple(selection.certification_work)
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
                level_telemetry=tuple(selection.level_telemetry),
                certification_work=tuple(selection.certification_work),
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

        compile_telemetry = (
            None
            if shared_cost_runtime["runtime"] is None
            else shared_cost_runtime["runtime"].telemetry()
        )
    finally:
        if shared_cost_runtime["runtime"] is not None:
            shared_cost_runtime["runtime"].close()

    certification_work = [dict(row) for row in selection.certification_work]
    attempt_reason_counts = Counter(str(attempt.reason) for attempt in selection.attempts)
    deletion_attempt_reason_counts = Counter(
        str(attempt.reason)
        for attempt in selection.attempts
        if attempt.removed_runtime_indices
    )
    shared_cache_entries = sum(
        int(row["logical_shared_cache_entry_count"]) for row in selector_passes
    )
    finalist_geometry_entries = sum(
        int(row.get("logical_full_geometry_entry_count", 0))
        for row in certification_work
    )
    refit_entries = sum(
        int(row.get("logical_refit_entry_count", 0))
        for row in certification_work
    )
    estimator_work = {
        "schema": "paper_ii_generalized_exchange_estimator_work_v1",
        "selector_pass_count": int(len(selector_passes)),
        "selector_passes": list(selector_passes),
        "positioned_candidate_column_count": int(
            sum(int(row["positioned_candidate_column_count"]) for row in selector_passes)
        ),
        "logical_shared_cache_entry_count": int(shared_cache_entries),
        "structural_candidate_record_count": int(selection.structural_scored_count),
        "certification_attempt_count": int(len(certification_work)),
        "certification_attempts": certification_work,
        "refit_objective_evaluation_attempt_count": int(
            sum(
                int(row.get("refit_objective_evaluation_attempt_count", 0))
                for row in certification_work
            )
        ),
        "refit_objective_evaluation_count": int(
            sum(
                int(row.get("refit_objective_evaluation_count", 0))
                for row in certification_work
            )
        ),
        "logical_refit_entry_count": int(refit_entries),
        "full_geometry_evaluation_attempt_count": int(
            sum(
                int(row.get("full_geometry_evaluation_attempt_count", 0))
                for row in certification_work
            )
        ),
        "full_geometry_evaluation_count": int(
            sum(
                int(row.get("full_geometry_evaluation_count", 0))
                for row in certification_work
            )
        ),
        "logical_finalist_geometry_entry_count": int(finalist_geometry_entries),
        "ray_overlap_check_count": int(
            sum(int(row.get("ray_overlap_check_count", 0)) for row in certification_work)
        ),
        "velocity_check_count": int(
            sum(int(row.get("velocity_check_count", 0)) for row in certification_work)
        ),
        "attempt_record_count": int(len(selection.attempts)),
        "attempt_reason_counts": dict(sorted(attempt_reason_counts.items())),
        "deletion_bearing_attempt_record_count": int(
            sum(deletion_attempt_reason_counts.values())
        ),
        "deletion_bearing_attempt_reason_counts": dict(
            sorted(deletion_attempt_reason_counts.items())
        ),
        "serialized_attempt_record_count": int(min(16, len(selection.attempts))),
        "serialized_attempt_ledger_complete": bool(len(selection.attempts) <= 16),
        "logical_total_selector_estimator_entries": int(
            shared_cache_entries + finalist_geometry_entries + refit_entries
        ),
        "complete": bool(
            all(bool(row.get("complete", False)) for row in selector_passes)
            and all(bool(row.get("complete", False)) for row in certification_work)
        ),
    }

    payload: dict[str, Any] = {
        # ``selection_policy`` names the selection *family* and is pinned to the
        # v2 string that historical readers match on.  The closed route bundle
        # actually in force is reported alongside it, so a compiled-Q run is not
        # mislabelled as the historical route.
        "selection_policy": GENERALIZED_EXCHANGE_SELECTION_POLICY_V2,
        "selection_route_bundle": str(bundle.bundle_id),
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
        "attempts": [
            a.to_json_dict(compiled_score=bundle.needs_compiled_cost)
            for a in selection.attempts[-16:]
        ],
        "estimator_work": estimator_work,
        "cost_normalization": (
            "per_candidate_raw_v1" if bundle.is_historical else bundle.cost_model_id
        ),
        "deletion_permission": permission_evaluator.summary(),
        "selector_route": {
            "schema": "paper_ii_selector_route_v1",
            **(
                {
                    "search_policy": exchange_config.search.search_policy,
                    "gain_evaluation": exchange_config.search.gain_evaluation,
                }
                if exchange_config.search.search_policy != "family_frontiers"
                else {}
            ),
            "bundle": asdict(bundle),
            "regime": "large_miss" if bool(domain.accuracy_debt) else "maintenance",
            "gamma_rcond": (
                DEFAULT_GAMMA_PINV_RCOND
                if (bundle.needs_gamma or bundle.needs_compiled_cost)
                else None
            ),
            "maintenance_loss_tolerance": float(route.maintenance_loss_tolerance),
            "maintenance_loss_tolerance_requested": route.maintenance_loss_tolerance_requested,
            "max_compiled_patches_per_checkpoint": route.max_compiled_patches_per_checkpoint,
            "compile": compile_telemetry,
            "compile_guard": (
                None
                if selection.telemetry is None or selection.telemetry.compile_guard is None
                else selection.telemetry.compile_guard.to_json_dict()
            ),
            "compile_calls_in_enumeration": (
                None if selection.telemetry is None else int(selection.telemetry.compile_calls)
            ),
            "levels": list(selection.level_telemetry),
        },
        "structural_scoring_parallelism": {
            "schema": "paper_ii_exchange_scoring_parallelism_v1",
            "requested_workers": int(exchange_config.search.scoring_workers),
            "effective_workers_max": (
                None
                if selection.telemetry is None
                else int(selection.telemetry.scoring_workers_effective_max)
            ),
            "execution_mode": (
                "serial"
                if int(exchange_config.search.scoring_workers) == 1
                else "speculative_memo_threads"
            ),
            "worker_clamp_policy": "min_requested_and_family_missing_solve_count",
            "blas_thread_env": {
                name: os.environ.get(name) for name in BLAS_THREAD_ENV_VARS
            },
        },
    }
    if selection.telemetry is not None:
        payload["work_guard"] = selection.telemetry.guard.to_json_dict()
        if selection.telemetry.deletion_discovery is not None:
            payload["selector_route"]["deletion_discovery"] = dict(
                selection.telemetry.deletion_discovery
            )
        payload["frontier_schedule"] = [
            int(w) for w in selection.telemetry.frontier_schedule
        ]
        payload["frontiers_used"] = int(selection.telemetry.frontiers_used)
        payload["q_base"] = float(selection.telemetry.q_base)
        if selection.telemetry.exchange_frontier is not None:
            payload["exchange_frontier"] = [
                [str(a), int(c)]
                for a, c in selection.telemetry.exchange_frontier
            ]
    if selection.committed is not None:
        committed_rank = {"score": float(selection.committed.score)}
        if bundle.needs_compiled_cost:
            # The successful attempt retains the score that ordered this
            # finalist, including when joint validation changed its geometry.
            attempt = selection.attempts[-1] if selection.attempts else None
            if (
                attempt is None
                or attempt.removed_runtime_indices != selection.committed.removed_runtime_indices
                or attempt.inserted_selection != selection.committed.inserted_selection
                or attempt.route_score is None
                or attempt.cost_factor_k is None
                or attempt.route_merit is None
            ):
                raise AssertionError("compiled commit is missing its sealed ranking record")
            committed_rank = {
                "score": float(attempt.route_score),
                "cost_factor_k": float(attempt.cost_factor_k),
                "route_merit": float(attempt.route_merit),
            }
        payload["committed"] = {
            "removed_runtime_indices": [
                int(i) for i in selection.committed.removed_runtime_indices
            ],
            "inserted_selection": [
                [str(a), int(p)]
                for a, p in selection.committed.inserted_selection
            ],
            "family": selection.committed.family,
            **committed_rank,
            "q": float(selection.committed.q),
            "insertion_gain": float(selection.committed.insertion_gain),
            "deletion_loss": float(selection.committed.deletion_loss),
            "delta_q_normalized": float(selection.committed.delta),
            "delta_gamma_normalized": (
                None
                if selection.committed.m_gamma is None
                else float(selection.committed.m_gamma)
            ),
            "compiled_cost_marginal": (
                None
                if selection.committed.cost_marginal is None
                else list(selection.committed.cost_marginal)
            ),
            "compiled_placement_cuts": (
                None
                if selection.committed.placement_cuts is None
                else [
                    [str(a), int(c)] for a, c in selection.committed.placement_cuts
                ]
            ),
            "compiled_patch_word_sha256": selection.committed.patch_word_sha256,
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


def build_candidate_pool_contract(
    state: APMcLachlanState,
    *,
    support_config: Any,
) -> dict[str, Any]:
    """Hash the exact ordered insertion pool shared by every policy.

    The source-pool name and its size are insufficient evidence: occurrence
    expansion, Pauli-family deduplication, order, and a finite cap all act
    before either structural rule sees a candidate.  This receipt is built by
    the same function used by APM and AVQDS, so a comparison can fail closed if
    the effective pools differ while avoiding a large pool dump in every run.
    """

    atoms, _pauli_by_atom, telemetry = build_candidate_pool(
        state,
        support_config=support_config,
        insertions_enabled=True,
    )
    ordered_atoms: list[dict[str, Any]] = []
    for atom in atoms:
        rotations = iter_runtime_rotation_terms(
            getattr(atom.term, "polynomial"),
            ignore_identity=bool(state.executor.ignore_identity),
            coefficient_tolerance=float(state.executor.coefficient_tolerance),
            sort_terms=bool(state.executor.sort_terms),
        )
        ordered_atoms.append(
            {
                "atom_id": str(atom.atom_id),
                "atom_label": str(atom.atom_label),
                "rotations": [
                    {
                        "pauli_exyz": str(rotation.pauli_exyz),
                        "coeff_real": float(rotation.coeff_real),
                    }
                    for rotation in rotations
                ],
            }
        )
    blob = json.dumps(ordered_atoms, sort_keys=True, separators=(",", ":"))
    return {
        "schema": "paper_ii_effective_candidate_pool_v1",
        "ordered_atom_contract_sha256": hashlib.sha256(
            blob.encode("utf-8")
        ).hexdigest(),
        "candidate_pool_raw": int(telemetry["candidate_pool_raw"]),
        "candidate_pool_deduplicated": int(
            telemetry["candidate_pool_deduplicated"]
        ),
        "candidate_pool_order": str(telemetry["candidate_pool_order"]),
        "append_occurrence_policy": str(
            telemetry["append_occurrence_policy"]
        ),
    }
