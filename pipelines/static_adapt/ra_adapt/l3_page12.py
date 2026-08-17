"""Named L=3 application seam for the Page-12 RA candidate route.

The ordinary Paper-I facade and ordinary pool builders remain locked to
Hubbard--Holstein ``L=2``.  This module admits the preserved diagnostic
``nph=1`` point plus the explicitly requested three-point weak-Holstein
``nph=3`` sector under one source-locked Page-12 policy composition; it is not
a generic higher-``L`` escape hatch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import threading
from typing import Any, ClassVar, Mapping

from pipelines.contracts.problem import ProblemRequest, ResolvedProblemContext
from pipelines.static_adapt.builders.problem_registry import (
    resolve_problem_context,
)
from pipelines.static_adapt.ra_adapt.adapters import (
    GLOBAL_SINGLETON_GRADIENT_PHASE0_ADAPTER_ID,
    GlobalSingletonGradientPhase0CandidateAdapter,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    ACTIVE_GRADIENT_STATIONARY,
    CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    RAAdaptRequest,
    RESOURCE_WEIGHTING_ALL_PHASE,
    canonical_sha256,
)
from pipelines.static_adapt.ra_adapt.pools import (
    PAPER_I_L3_PAGE12_APPLICATION_ID,
    PAPER_I_L3_PAGE12_HAMILTONIAN_TERMS_SHA256,
    PAPER_I_L3_PAGE12_PARENT_COUNT,
    PAPER_I_L3_PAGE12_PARENT_LABELS_SHA256,
    PAPER_I_L3_PAGE12_PARENT_POOL_SHA256,
    PAPER_I_L3_PAGE12_PROBLEM_REQUEST_SHA256,
    PAPER_I_L3_PAGE12_SINGLETON_COUNT,
    PAPER_I_L3_PAGE12_SINGLETON_LABELS_SHA256,
    PAPER_I_L3_PAGE12_SINGLETON_POOL_SHA256,
    PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_PARENT_COUNT,
    PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_PARENT_LABELS_SHA256,
    PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_PARENT_POOL_SHA256,
    PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_POOL_LOCKS,
    PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_SINGLETON_COUNT,
    PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_SINGLETON_LABELS_SHA256,
    PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_SINGLETON_POOL_SHA256,
    build_paper_i_l3_page12_guarded_single_pauli_pool,
    build_paper_i_l3_page12_parent_template_inventory,
    require_paper_i_l3_page12_problem,
)
from pipelines.static_adapt.sr_snake.contracts import (
    BeamOff,
    CheckpointObservation,
    EstimatorLedgerObservation,
    ExactEDSourceReceipt,
    FreshStart,
    PlateauCommutationInsertion,
    PruningOff,
    SRExecutionPolicy,
    SRMethodPolicy,
    SRObservationPolicy,
    SRStopPolicy,
    SingletonAdmission,
)
from pipelines.scaffold.hh_continuation_generators import (
    serialize_polynomial_terms_exyz,
)


PAPER_I_L3_PAGE12_ADAPTER_ID = (
    "paper_i_l3_page12_global_singleton_gradient_phase0_candidate_adapter_v1"
)
PAPER_I_L3_PAGE12_ALGORITHM_ID = (
    "paper_i_ra_adapt_global_singleton_gradient_phase0_phase1_phase2_phase3_"
    "qiskit_phase2_phase3_plateau_no_lanes_v1"
)
PAPER_I_L3_PAGE12_SOURCE_LOCK_KEY = (
    "paper_i_l3_page12_application_source_sha256"
)
PAPER_I_L3_PAGE12_EXACT_SOURCE_ID = (
    "paper_i_l3_page12_same_cutoff_sector_ed_v1"
)
PAPER_I_L3_PAGE12_EXACT_SOURCE_RECEIPT_SHA256 = (
    "079ef700ed8fd478ccd45b64df740815c2a68ec10dc280bd7ec84bcf71dddd04"
)
PAPER_I_L3_PAGE12_EXACT_ENERGY = -0.6735153809694907
PAPER_I_L3_PAGE12_HORIZON = 50
PAPER_I_L3_PAGE12_APPLICATION_SOURCE_SHA256 = (
    "7ef4bdc24f4dbd751bdfeebed3ab26be1dfece0a33331ba18eff38b35cfad70c"
)
PAPER_I_L3_PAGE12_ROUTE_CONTRACT_SHA256 = (
    "8d5f9a53d79c30abba5c26b9bba68751dea3122b2f692021a44e7db260748e83"
)
PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_REGIMES = (
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
)
PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_U = {
    "weak_weak": 0.25,
    "intermediate_weak": 1.25,
    "strong_weak_u8": 8.0,
}
PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_EXACT_SOURCE_RECEIPT_SHA256 = {
    "weak_weak": (
        "07b9fb66ed1a1114005d3f15698e2420c6470cb984adb09cc684a7dc86bd6904"
    ),
    "intermediate_weak": (
        "79545226a2112f11a4d55f7d164325399733724e3295f0a8ac1039ddf23219b6"
    ),
    "strong_weak_u8": (
        "8f5009c7359e6a052b95e2d60d62befbfc5794a5cc9a2b9c76367bd7ea237784"
    ),
}
PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_EXACT_ENERGY = {
    "weak_weak": -1.2240897788292735,
    "intermediate_weak": -0.6741755664986704,
    "strong_weak_u8": 0.7901779398005324,
}
# Filled from the complete canonical application payloads below.  These are
# independent per-regime locks even though the pool and route bytes are shared.
PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_APPLICATION_SOURCE_SHA256 = {
    "weak_weak": (
        "4c28997afe4b554d6b58f28909e7be11150e061bdf31db18621e5b72c2a943ec"
    ),
    "intermediate_weak": (
        "2bf29f2b2de694ff5b06651514ee07b8b16211918ebc655e80baae1b83a53e6f"
    ),
    "strong_weak_u8": (
        "c5019e08d24537a8874254b7a67f0696bfd7f882d7a914686cb03b8566e9e871"
    ),
}

_L3_EXECUTABLE_POOL_CACHE_ENV = "STATIC_ADAPT_L3_ROUTE_POOL_CACHE"
_L3_EXECUTABLE_POOL_MEMORY_CACHE: dict[str, Any] = {}
_L3_EXECUTABLE_POOL_MEMORY_CACHE_LOCK = threading.RLock()
_L3_EXECUTABLE_POOL_CACHE_DISABLED = {
    "",
    "0",
    "off",
    "false",
    "no",
    "disabled",
    "none",
}
_L3_EXECUTABLE_POOL_CACHE_MEMORY = {"1", "on", "true", "yes", "memory"}


def _paper_i_l3_page12_executable_pool(
    problem: ResolvedProblemContext,
) -> Any:
    """Return the locked L3 pool, optionally memoized within this process.

    The named-problem validator still runs on every call.  A cache entry is
    therefore reachable only after the complete request, register, sector,
    Hamiltonian, and per-regime source locks pass.  The cache is opt-in so
    sealed packages that explicitly disable caches retain their prior
    execution behavior.
    """

    identity = require_paper_i_l3_page12_problem(problem)
    raw_mode = str(
        os.environ.get(_L3_EXECUTABLE_POOL_CACHE_ENV, "off")
    ).strip().lower()
    if raw_mode in _L3_EXECUTABLE_POOL_CACHE_DISABLED:
        return build_paper_i_l3_page12_guarded_single_pauli_pool(problem)
    if raw_mode not in _L3_EXECUTABLE_POOL_CACHE_MEMORY:
        raise ValueError(
            f"{_L3_EXECUTABLE_POOL_CACHE_ENV} must be memory or off."
        )

    with _L3_EXECUTABLE_POOL_MEMORY_CACHE_LOCK:
        cached = _L3_EXECUTABLE_POOL_MEMORY_CACHE.get(identity)
        if cached is None:
            cached = build_paper_i_l3_page12_guarded_single_pauli_pool(
                problem
            )
            _L3_EXECUTABLE_POOL_MEMORY_CACHE[identity] = cached
        return cached


@dataclass(frozen=True)
class PaperIL3Page12GlobalSingletonGradientPhase0CandidateAdapter(
    GlobalSingletonGradientPhase0CandidateAdapter
):
    """Global-singleton Page-12 adapter for the exact named L=3 point."""

    application_family_key: ClassVar[str] = "hh"
    application_id = PAPER_I_L3_PAGE12_APPLICATION_ID
    adapter_id: str = PAPER_I_L3_PAGE12_ADAPTER_ID

    def __post_init__(self) -> None:
        if (
            self.candidate_representation_id
            != CANDIDATE_REPRESENTATION_SINGLE_PAULI
            or self.adapter_id != PAPER_I_L3_PAGE12_ADAPTER_ID
        ):
            raise ValueError("The Page-12 L=3 adapter identity is fixed.")

    def parent_inventory(self, problem: ResolvedProblemContext) -> Any:
        return build_paper_i_l3_page12_parent_template_inventory(
            problem,
            representation_id=self.candidate_representation_id,
        )

    def executable_pool(self, problem: ResolvedProblemContext) -> Any:
        return _paper_i_l3_page12_executable_pool(problem)


def build_paper_i_l3_page12_problem(
    regime_id: str = "intermediate_weak",
    *,
    nph: int = 1,
) -> ResolvedProblemContext:
    """Construct one exact named L=3 physics context.

    The argument-free call preserves the original intermediate--weak
    ``nph=1`` application.  ``nph=3`` is admitted only for the three locked
    weak-Holstein regimes.
    """

    regime = str(regime_id)
    cutoff = int(nph)
    if cutoff == 1 and regime == "intermediate_weak":
        u = 1.25
    elif cutoff == 3 and regime in PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_U:
        u = PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_U[regime]
    else:
        raise ValueError(
            "The named L=3 Page-12 problem supports only the preserved "
            "intermediate_weak/nph=1 point or the three locked nph=3 "
            "weak-Holstein regimes."
        )

    problem = resolve_problem_context(
        ProblemRequest(
            problem_key="hh",
            num_sites=3,
            t=1.0,
            u=u,
            dv=0.0,
            omega0=1.0,
            g_ep=math.sqrt(0.125),
            n_ph_max=cutoff,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            include_zero_point=True,
            n_fermions=None,
        )
    )
    observed = require_paper_i_l3_page12_problem(problem)
    expected = (
        "legacy_intermediate_weak_nph1" if cutoff == 1 else regime
    )
    if observed != expected:
        raise RuntimeError("The named L=3 Page-12 problem identity drifted.")
    return problem


def build_paper_i_l3_page12_request(
    *,
    output_dir: Path | None = None,
) -> RAAdaptRequest:
    """Return the fixed fresh ``k=50`` Page-12 scientific request.

    Exact diagonalization is deliberately absent from the online stopping
    policy.  The same-cutoff reference is authenticated separately for
    reporting by :func:`paper_i_l3_page12_application_source_contract`.
    """

    observation = SRObservationPolicy()
    if output_dir is not None:
        root = Path(output_dir)
        observation = SRObservationPolicy(
            checkpoint=CheckpointObservation(
                path=root / "current.json",
                every_controller_rounds=1,
                keep_history_tail=100,
            ),
            estimator_ledger=EstimatorLedgerObservation(
                path=root / "estimator_ledger.json"
            ),
        )
    return RAAdaptRequest(
        adapter=PaperIL3Page12GlobalSingletonGradientPhase0CandidateAdapter(),
        method=SRMethodPolicy(
            admission=SingletonAdmission(),
            insertion=PlateauCommutationInsertion(),
            pruning=PruningOff(),
            beam=BeamOff(),
        ),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(
                maximum_controller_rounds=PAPER_I_L3_PAGE12_HORIZON
            ),
            resume=FreshStart(),
        ),
        observation=observation,
    )


def require_paper_i_l3_page12_request(request: RAAdaptRequest) -> None:
    """Fail closed on any policy or horizon drift at the named seam."""

    if not isinstance(request, RAAdaptRequest):
        raise TypeError("request must be an RAAdaptRequest.")
    if not (
        isinstance(
            request.adapter,
            PaperIL3Page12GlobalSingletonGradientPhase0CandidateAdapter,
        )
        and request.adapter.adapter_id == PAPER_I_L3_PAGE12_ADAPTER_ID
        and request.adapter.phase0_shortlist_policy_id
        == "global_singleton_absolute_gradient_shortlist_v1"
        and isinstance(request.method.admission, SingletonAdmission)
        and isinstance(
            request.method.insertion,
            PlateauCommutationInsertion,
        )
        and isinstance(request.method.pruning, PruningOff)
        and isinstance(request.method.beam, BeamOff)
        and isinstance(request.execution.resume, FreshStart)
        and int(request.execution.stop.maximum_controller_rounds)
        == PAPER_I_L3_PAGE12_HORIZON
        and request.execution.stop.exact_ed_target is None
    ):
        raise ValueError(
            "The named Page-12 L=3 seam is fixed to a fresh k=50 global-"
            "singleton gradient-Phase-0, singleton I/II/III, plateau, "
            "pruning-off, beam-off request with reporting-only ED."
        )


def paper_i_l3_page12_application_source_contract(
    problem: ResolvedProblemContext,
) -> dict[str, Any]:
    """Build the package source-lock payload for this exact application."""

    identity = require_paper_i_l3_page12_problem(problem)
    legacy = identity == "legacy_intermediate_weak_nph1"
    regime_id = "intermediate_weak" if legacy else identity
    nph = int(problem.request.n_ph_max)
    parent = build_paper_i_l3_page12_parent_template_inventory(
        problem,
        representation_id=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    )
    singleton = _paper_i_l3_page12_executable_pool(problem)
    exact_source_id = (
        PAPER_I_L3_PAGE12_EXACT_SOURCE_ID
        if legacy
        else (
            "paper_i_l3_weak_sector_"
            f"{regime_id}_nph3_same_cutoff_sector_ed_v1"
        )
    )
    exact_source = ExactEDSourceReceipt.from_problem(
        problem,
        source_id=exact_source_id,
    )
    exact_source_sha256 = canonical_sha256(exact_source.to_dict())
    exact_energy = float(problem.exact_target.resolve_energy())
    expected_exact_receipt = (
        PAPER_I_L3_PAGE12_EXACT_SOURCE_RECEIPT_SHA256
        if legacy
        else PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_EXACT_SOURCE_RECEIPT_SHA256[
            regime_id
        ]
    )
    expected_exact_energy = (
        PAPER_I_L3_PAGE12_EXACT_ENERGY
        if legacy
        else PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_EXACT_ENERGY[regime_id]
    )
    if (
        exact_source_sha256 != expected_exact_receipt
        or exact_source.n_ph_max != nph
        or not math.isclose(
            exact_energy,
            expected_exact_energy,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(
            "The Page-12 L=3 same-cutoff ED reference drifted from its "
            "typed problem, sector, cutoff, or energy."
        )
    payload: dict[str, Any] = {
        "schema": (
            "paper_i_l3_page12_application_source_contract_v1"
            if legacy
            else "paper_i_l3_page12_weak_sector_application_source_contract_v1"
        ),
        "application_id": PAPER_I_L3_PAGE12_APPLICATION_ID,
        "algorithm_id": PAPER_I_L3_PAGE12_ALGORITHM_ID,
        "adapter_id": PAPER_I_L3_PAGE12_ADAPTER_ID,
        "problem_request_sha256": exact_source.problem_request_sha256,
        "hamiltonian_terms_sha256": canonical_sha256(
            serialize_polynomial_terms_exyz(problem.hamiltonian)
        ),
        "sector_num_particles": [2, 1],
        "parent_inventory": {
            "count": (
                PAPER_I_L3_PAGE12_PARENT_COUNT
                if legacy
                else PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_PARENT_COUNT
            ),
            "ordered_labels_sha256": (
                PAPER_I_L3_PAGE12_PARENT_LABELS_SHA256
                if legacy
                else PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_PARENT_LABELS_SHA256
            ),
            "ordered_pool_sha256": (
                PAPER_I_L3_PAGE12_PARENT_POOL_SHA256
                if legacy
                else PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_POOL_LOCKS[
                    regime_id
                ]["parent_pool_sha256"]
            ),
            "observed_receipt_sha256": parent.receipt.sha256,
        },
        "singleton_inventory": {
            "count": (
                PAPER_I_L3_PAGE12_SINGLETON_COUNT
                if legacy
                else PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_SINGLETON_COUNT
            ),
            "ordered_labels_sha256": (
                PAPER_I_L3_PAGE12_SINGLETON_LABELS_SHA256
                if legacy
                else PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_SINGLETON_LABELS_SHA256
            ),
            "ordered_pool_sha256": (
                PAPER_I_L3_PAGE12_SINGLETON_POOL_SHA256
                if legacy
                else PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_POOL_LOCKS[
                    regime_id
                ]["singleton_pool_sha256"]
            ),
            "observed_receipt_sha256": singleton.receipt.sha256,
        },
        "same_cutoff_exact_reference": {
            **exact_source.to_dict(),
            "receipt_sha256": exact_source_sha256,
            # Sparse-sector eigensolvers can vary in the final floating-point
            # bits.  Validate the live solve above, but source-lock the stable
            # reference value used by this application contract.
            "energy": expected_exact_energy,
            "controller_input": False,
        },
        "scientific_settings": {
            "maximum_controller_rounds": PAPER_I_L3_PAGE12_HORIZON,
            "optimizer": "powell",
            "optimizer_maxiter": 200,
            "seed": 7,
            "active_gradient_policy": ACTIVE_GRADIENT_STATIONARY,
            "resource_weighting_scope": RESOURCE_WEIGHTING_ALL_PHASE,
            "phase0_shortlist_size": 24,
            "phase0_cost": "gradient_only_no_metric_no_resource_v1",
            "phase1_cost": "structural_proxy_v1",
            "phase2_phase3_cost": (
                "qiskit_full_trial_ansatz_signed_marginal_no_lanes_v1"
            ),
            "insertion": "plateau_commutation_reduced_tau1em4_v2",
        },
    }
    if not legacy:
        payload["regime_id"] = regime_id
        payload["n_ph_max"] = 3
    digest = canonical_sha256(payload)
    expected_digest = (
        PAPER_I_L3_PAGE12_APPLICATION_SOURCE_SHA256
        if legacy
        else PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_APPLICATION_SOURCE_SHA256[
            regime_id
        ]
    )
    if digest != expected_digest:
        raise RuntimeError(
            "The named Page-12 L=3 application source contract drifted: "
            f"observed {digest}."
        )
    return {**payload, "sha256": digest}


def require_paper_i_l3_page12_materialization(
    *,
    problem: ResolvedProblemContext,
    request: RAAdaptRequest,
    algorithm_id: str,
    active_gradient_policy: str,
    resource_weighting_scope: str,
    source_locks: Mapping[str, str],
) -> None:
    """Authenticate the bundle-only Page-12 policy and application lock."""

    require_paper_i_l3_page12_problem(problem)
    require_paper_i_l3_page12_request(request)
    expected_source = paper_i_l3_page12_application_source_contract(problem)
    if (
        str(algorithm_id) != PAPER_I_L3_PAGE12_ALGORITHM_ID
        or str(active_gradient_policy) != ACTIVE_GRADIENT_STATIONARY
        or str(resource_weighting_scope) != RESOURCE_WEIGHTING_ALL_PHASE
        or not isinstance(source_locks, Mapping)
        or source_locks.get(PAPER_I_L3_PAGE12_SOURCE_LOCK_KEY)
        != expected_source["sha256"]
    ):
        raise ValueError(
            "The named Page-12 L=3 application requires the stationary "
            "Qiskit-II/III Page-12 bundle authority and its exact "
            "application source-lock digest."
        )


def is_paper_i_l3_page12_application(
    problem: ResolvedProblemContext | None,
    request: RAAdaptRequest,
) -> bool:
    """Return true only after validating the named adapter and problem."""

    if not isinstance(
        request.adapter,
        PaperIL3Page12GlobalSingletonGradientPhase0CandidateAdapter,
    ):
        return False
    if problem is None:
        raise ValueError("The Page-12 L=3 adapter requires its problem.")
    require_paper_i_l3_page12_problem(problem)
    require_paper_i_l3_page12_request(request)
    return True


__all__ = [
    "PAPER_I_L3_PAGE12_ADAPTER_ID",
    "PAPER_I_L3_PAGE12_ALGORITHM_ID",
    "PAPER_I_L3_PAGE12_APPLICATION_SOURCE_SHA256",
    "PAPER_I_L3_PAGE12_EXACT_ENERGY",
    "PAPER_I_L3_PAGE12_EXACT_SOURCE_ID",
    "PAPER_I_L3_PAGE12_HORIZON",
    "PAPER_I_L3_PAGE12_ROUTE_CONTRACT_SHA256",
    "PAPER_I_L3_PAGE12_SOURCE_LOCK_KEY",
    "PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_APPLICATION_SOURCE_SHA256",
    "PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_EXACT_ENERGY",
    "PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_EXACT_SOURCE_RECEIPT_SHA256",
    "PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_REGIMES",
    "PaperIL3Page12GlobalSingletonGradientPhase0CandidateAdapter",
    "build_paper_i_l3_page12_problem",
    "build_paper_i_l3_page12_request",
    "is_paper_i_l3_page12_application",
    "paper_i_l3_page12_application_source_contract",
    "require_paper_i_l3_page12_materialization",
    "require_paper_i_l3_page12_request",
]
