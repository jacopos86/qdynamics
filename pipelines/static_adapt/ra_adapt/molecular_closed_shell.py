"""Named lane-owned RA-ADAPT application for the molecular closed-shell family.

The ordinary Paper-I facade is locked to the canonical Hubbard--Holstein L=2
problem.  This module admits a strict-intake ``molecular_restricted_closed_shell``
problem (``resolve_molecular_diagnostic_intake``: spatial MO integrals, blocked
alpha-then-beta Jordan--Wigner register, fixed ``(n_up, n_dn)`` sector, HF
reference) the same way the other named applications do: a fixed-identity
candidate adapter that declares ``application_family_key``, a problem predicate
that fails closed, lane-owned pool builders whose executable labels equal the
runtime pool labels, and a route-contract overlay so receipts name this family
and its proxy Phase-III cost instead of the Hubbard--Holstein defaults they would
otherwise inherit.

Scientific scope of the application, stated once:

* candidate representation: intact macro generators (``uccsd`` by default, the
  92-generator molecular UCCSD pool; any admissible repository pool key may be
  named, and the executed pool is the audited subset whose grouped and executed
  unitaries preserve the fixed counts);
* method policy: singleton admission, plateau-commutation or append-only
  insertion, pruning off, beam off (the compositions the probes exercised);
* Phase-III backend cost: ``proxy`` (the H2O precedent for non-HH problems; the
  HH ``marrakesh_graph_span_v1`` oracle is a lattice-specific cost model);
* the exact reference is a reporting input: no ``ExactEDStop`` unless a caller
  asks for one explicitly.

Nothing here changes the Hubbard--Holstein, pure-Hubbard, L=3 or H2O
linear-FD routes: every engine branch is gated on this adapter type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from pipelines.contracts.problem import ResolvedProblemContext
from pipelines.static_adapt.builders.pool_resolution import (
    resolve_pool_plan,
    resolve_requested_pool_filters,
)
from pipelines.static_adapt.ra_adapt.adapters import MacroCandidateAdapter
from pipelines.static_adapt.ra_adapt.contracts import (
    CANDIDATE_REPRESENTATION_MACRO,
    RAAdaptRequest,
    canonical_sha256,
)
from pipelines.static_adapt.ra_adapt import pools as _pools
from pipelines.static_adapt.sector_invariants import (
    audit_candidate_pool_sector_contract,
)
from pipelines.static_adapt.sr_snake.contracts import (
    AppendOnlyInsertion,
    BeamOff,
    ExactEDSourceReceipt,
    ExactEDStop,
    FreshStart,
    PlateauCommutationInsertion,
    PruningOff,
    SRExecutionPolicy,
    SRMethodPolicy,
    SRStopPolicy,
    SingletonAdmission,
)

MOLECULAR_CLOSED_SHELL_FAMILY = "molecular_restricted_closed_shell"
MOLECULAR_CLOSED_SHELL_APPLICATION_ID = "doron_molecular_closed_shell_ra_adapt_v1"
MOLECULAR_CLOSED_SHELL_APPLICATION_LANE = "doron_molecular_closed_shell_ra_macro_v1"
MOLECULAR_CLOSED_SHELL_MACRO_ADAPTER_ID = "doron_molecular_closed_shell_macro_adapter_v1"
MOLECULAR_CLOSED_SHELL_ROUTE_PREFIX = "doron_molecular_ra_adapt__"
MOLECULAR_CLOSED_SHELL_SUPERSESSION_REASON = (
    "doron_molecular_closed_shell_ra_adapt_nonstationary_full_response_v2_20261020"
)
MOLECULAR_CLOSED_SHELL_PHASE3_COST_MODE = "proxy"
MOLECULAR_CLOSED_SHELL_PHASE3_COST_REASON = "generic_non_hh_application_v1"
MOLECULAR_CLOSED_SHELL_DEFAULT_POOL_KEY = "uccsd"
MOLECULAR_CLOSED_SHELL_LOCK_MESSAGE = (
    "The ordinary Paper-I RA-ADAPT facade is locked to the canonical "
    "Hubbard--Holstein L=2 problem. Molecular closed-shell problems are "
    "executable only through the named Doron molecular closed-shell macro "
    "application adapter."
)
_INSERTION_POLICIES = {
    "plateau_commutation": PlateauCommutationInsertion,
    "append_only": AppendOnlyInsertion,
}
# HH-calibrated invariants that do not describe this family's parent pool.
_HH_ONLY_INVARIANT_KEYS = ("candidate_parent_pool", "hh_preseed_policy")

__all__ = [
    "MOLECULAR_CLOSED_SHELL_APPLICATION_ID",
    "MOLECULAR_CLOSED_SHELL_APPLICATION_LANE",
    "MOLECULAR_CLOSED_SHELL_DEFAULT_POOL_KEY",
    "MOLECULAR_CLOSED_SHELL_FAMILY",
    "MOLECULAR_CLOSED_SHELL_LOCK_MESSAGE",
    "MOLECULAR_CLOSED_SHELL_MACRO_ADAPTER_ID",
    "MOLECULAR_CLOSED_SHELL_ROUTE_PREFIX",
    "MOLECULAR_CLOSED_SHELL_SUPERSESSION_REASON",
    "MolecularClosedShellMacroCandidateAdapter",
    "build_molecular_closed_shell_ra_request",
    "is_molecular_closed_shell_application",
    "molecular_closed_shell_adapter_requested",
    "molecular_closed_shell_execution_overlay",
    "molecular_closed_shell_invariants_overlay",
    "require_molecular_closed_shell_problem",
]


def require_molecular_closed_shell_problem(problem: ResolvedProblemContext) -> None:
    """Fail closed unless ``problem`` is a strict-intake molecular closed-shell context."""

    if not isinstance(problem, ResolvedProblemContext):
        raise TypeError("problem must be a ResolvedProblemContext.")
    family = str(problem.family_key).strip().lower()
    if family != MOLECULAR_CLOSED_SHELL_FAMILY:
        raise ValueError(f"The molecular closed-shell RA application requires family {MOLECULAR_CLOSED_SHELL_FAMILY!r}, got {family!r}.")
    request = problem.request
    if str(getattr(request, "problem_key", "")).strip().lower() != MOLECULAR_CLOSED_SHELL_FAMILY:
        raise ValueError("The molecular closed-shell RA application requires a molecular problem request.")
    layout = problem.layout
    num_sites = int(request.num_sites)
    if int(getattr(layout, "boson_qubits", 0)) != 0 or int(getattr(request, "n_ph_max", 0)) != 0:
        raise ValueError("The molecular closed-shell RA application has no boson register.")
    if int(layout.fermion_qubits) != 2 * num_sites or int(layout.total_qubits) != 2 * num_sites:
        raise ValueError("The molecular closed-shell RA application requires a 2 * n_spatial spin-orbital register.")
    if str(layout.ordering).strip().lower() != "blocked":
        raise ValueError("The molecular closed-shell RA application requires blocked alpha-then-beta ordering.")
    particles = problem.sector.num_particles
    if particles is None or len(particles) != 2 or int(particles[0]) != int(particles[1]):
        raise ValueError("The molecular closed-shell RA application requires a closed-shell (n_up == n_dn) sector.")
    if not (0 < int(particles[0]) < num_sites):
        raise ValueError("The molecular closed-shell RA application requires a partially filled active space.")
    runtime = getattr(problem, "runtime_data", None) or {}
    if "molecular_diagnostic_intake_receipt" not in runtime:
        raise ValueError("The molecular closed-shell RA application requires the strict-intake receipt in runtime_data.")


def _candidate_records(problem: ResolvedProblemContext, pool_key: str) -> tuple[tuple[Any, ...], Any]:
    """Parent records from the repository pool resolver, exactly as the runtime re-resolves them."""

    key = str(pool_key).strip().lower()
    admissible = tuple(str(k) for k in problem.admissible_pool_keys)
    if key not in admissible:
        raise ValueError(f"pool_key {key!r} is not admissible for this problem; choose from {list(admissible)}.")
    filters = resolve_requested_pool_filters(
        problem_key=MOLECULAR_CLOSED_SHELL_FAMILY,
        num_sites=int(problem.request.num_sites),
        n_ph_max=0,
        adapt_pool=key,
        adapt_pool_class_filter_json=None,
        adapt_pool_label_filter_json=None,
        adapt_selected_logical_source_json=None,
        adapt_selected_logical_mode="off",
        adapt_selected_logical_transfer_mode="exact_match_v1",
    )
    plan = resolve_pool_plan(
        resolved_problem=problem,
        continuation_mode="phase3_v1",
        adapt_pool=key,
        paop_r=1,
        paop_split_paulis=False,
        paop_prune_eps=0.0,
        paop_normalization="none",
        phase3_symmetry_mitigation_mode="off",
        filter_resolution=filters,
    )
    records = []
    for index, term in enumerate(plan.pool):
        serialized = _pools._serialized_terms(term)
        if not serialized:
            continue
        metadata = _pools._generator_metadata(term)
        identity = str(
            metadata.get("generator_id")
            or metadata.get("generator_identity")
            or f"parent:{canonical_sha256({'label': str(term.label), 'terms': serialized})[:16]}"
        )
        records.append(
            _pools.CandidateRecord(
                label=str(term.label),
                term=term,
                representation_id=CANDIDATE_REPRESENTATION_MACRO,
                generator_identity=identity,
                parent_identities=(),
                family_id=str(plan.pool_family_ids[index] if index < len(plan.pool_family_ids) else key),
                stage_family=str(plan.pool_stage_family[index] if index < len(plan.pool_stage_family) else "shared"),
                construction=f"{key}::{plan.pool_key}",
                execution_mode=str(getattr(term, "execution_mode", "termwise_product") or "termwise_product"),
                serialized_terms_exyz=serialized,
                symmetry_receipt=None,
                generator_metadata=metadata,
            )
        )
    return tuple(records), plan


@dataclass(frozen=True)
class MolecularClosedShellMacroCandidateAdapter(MacroCandidateAdapter):
    """Intact molecular macro generators for the named closed-shell application."""

    application_family_key: ClassVar[str] = MOLECULAR_CLOSED_SHELL_FAMILY
    application_id: ClassVar[str] = MOLECULAR_CLOSED_SHELL_APPLICATION_ID
    adapter_id: str = MOLECULAR_CLOSED_SHELL_MACRO_ADAPTER_ID
    pool_key: str | None = MOLECULAR_CLOSED_SHELL_DEFAULT_POOL_KEY

    def __post_init__(self) -> None:
        if (
            self.candidate_representation_id != CANDIDATE_REPRESENTATION_MACRO
            or self.adapter_id != MOLECULAR_CLOSED_SHELL_MACRO_ADAPTER_ID
        ):
            raise ValueError("The named molecular closed-shell adapter identity is fixed.")
        if self.pool_key is None or not str(self.pool_key).strip():
            raise ValueError("The named molecular closed-shell adapter requires an explicit pool_key.")

    def parent_inventory(self, problem: ResolvedProblemContext) -> Any:
        require_molecular_closed_shell_problem(problem)
        records, plan = _candidate_records(problem, str(self.pool_key))
        return _pools.CandidateInventory(
            candidates=records,
            receipt=_pools._receipt(
                schema=_pools.PARENT_TEMPLATE_INVENTORY_SCHEMA,
                representation_id=CANDIDATE_REPRESENTATION_MACRO,
                candidates=records,
            ),
            metadata={
                "pool_key": str(plan.pool_key),
                "method_name": str(plan.method_name),
                "application_id": MOLECULAR_CLOSED_SHELL_APPLICATION_ID,
            },
        )

    def executable_pool(self, problem: ResolvedProblemContext) -> Any:
        parent = self.parent_inventory(problem)
        audit = audit_candidate_pool_sector_contract(
            [candidate.term for candidate in parent.candidates],
            resolved_problem=problem,
        )
        rejected = set(audit.get("grouped_violation_indices", ())) | set(audit.get("execution_violation_indices", ()))
        kept = tuple(candidate for index, candidate in enumerate(parent.candidates) if index not in rejected)
        removed = tuple(candidate.label for index, candidate in enumerate(parent.candidates) if index in rejected)
        if not kept:
            raise ValueError("The molecular closed-shell executable pool is empty after the sector audit.")
        return _pools.CandidateInventory(
            candidates=kept,
            receipt=_pools._receipt(
                schema=_pools.EXECUTABLE_MACRO_POOL_SCHEMA,
                representation_id=CANDIDATE_REPRESENTATION_MACRO,
                candidates=kept,
                removed_labels=removed,
                source_parent_labels_sha256=parent.receipt.ordered_labels_sha256,
            ),
            metadata={
                "pool_key": str(self.pool_key),
                "sector_audit_removed_count": int(len(removed)),
                "application_id": MOLECULAR_CLOSED_SHELL_APPLICATION_ID,
            },
        )


def molecular_closed_shell_adapter_requested(request: RAAdaptRequest) -> bool:
    """True when the request names the molecular closed-shell adapter (no problem validation)."""

    return isinstance(getattr(request, "adapter", None), MolecularClosedShellMacroCandidateAdapter)


def is_molecular_closed_shell_application(
    problem: ResolvedProblemContext | None,
    request: RAAdaptRequest,
) -> bool:
    """Named-application predicate: adapter match, valid problem, locked method policy."""

    if not molecular_closed_shell_adapter_requested(request):
        return False
    if problem is None:
        raise ValueError("The named molecular closed-shell adapter requires its problem.")
    require_molecular_closed_shell_problem(problem)
    method = request.method
    if not (
        isinstance(method.admission, SingletonAdmission)
        and isinstance(method.insertion, (PlateauCommutationInsertion, AppendOnlyInsertion))
        and isinstance(method.pruning, PruningOff)
        and isinstance(method.beam, BeamOff)
    ):
        raise ValueError(
            "The molecular closed-shell RA application is locked to singleton admission, "
            "plateau or append-only insertion, pruning off, and beam off."
        )
    return True


def molecular_closed_shell_execution_overlay(execution: Mapping[str, Any]) -> dict[str, Any]:
    """Route execution settings that name this family and its proxy Phase-III cost."""

    updated = dict(execution)
    updated.update(
        {
            "problem": MOLECULAR_CLOSED_SHELL_FAMILY,
            "phase3_backend_cost_mode": MOLECULAR_CLOSED_SHELL_PHASE3_COST_MODE,
        }
    )
    updated.pop("phase3_backend_cost_scope", None)
    return updated


def molecular_closed_shell_invariants_overlay(invariants: Mapping[str, Any], *, request: RAAdaptRequest) -> dict[str, Any]:
    """Semantic invariants for the receipts: the family, the lane and the pool actually used."""

    updated = dict(invariants)
    for key in _HH_ONLY_INVARIANT_KEYS:
        updated.pop(key, None)
    updated.update(
        {
            "application_lane": MOLECULAR_CLOSED_SHELL_APPLICATION_LANE,
            "application_id": MOLECULAR_CLOSED_SHELL_APPLICATION_ID,
            "problem_family": MOLECULAR_CLOSED_SHELL_FAMILY,
            "operator_pool_identity": str(getattr(request.adapter, "pool_key", MOLECULAR_CLOSED_SHELL_DEFAULT_POOL_KEY)),
            "phase3_backend_cost_mode": MOLECULAR_CLOSED_SHELL_PHASE3_COST_MODE,
            "phase3_backend_cost_reason": MOLECULAR_CLOSED_SHELL_PHASE3_COST_REASON,
            "singleton_admission": True,
            "raw_single_pauli_child_exposure": False,
            "guarded_single_pauli_child_exposure": False,
        }
    )
    return updated


def build_molecular_closed_shell_ra_request(
    *,
    maximum_controller_rounds: int,
    insertion: str = "plateau_commutation",
    pool_key: str = MOLECULAR_CLOSED_SHELL_DEFAULT_POOL_KEY,
    exact_ed_stop: ExactEDStop | None = None,
) -> RAAdaptRequest:
    """One fresh request for the named application.

    ``exact_ed_stop`` is optional and off by default: the exact sector energy is
    a reporting input, not an online controller input.  Callers who want the
    H2O-lane style chemical-accuracy termination pass an ``ExactEDStop`` built
    with :func:`exact_ed_stop_for_problem`.
    """

    horizon = int(maximum_controller_rounds)
    if horizon < 1:
        raise ValueError("maximum_controller_rounds must be positive.")
    try:
        insertion_policy = _INSERTION_POLICIES[str(insertion)]()
    except KeyError:
        raise ValueError(f"insertion must be one of {sorted(_INSERTION_POLICIES)}; got {insertion!r}.") from None
    return RAAdaptRequest(
        adapter=MolecularClosedShellMacroCandidateAdapter(pool_key=str(pool_key)),
        method=SRMethodPolicy(
            admission=SingletonAdmission(),
            insertion=insertion_policy,
            pruning=PruningOff(),
            beam=BeamOff(),
        ),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=horizon, exact_ed_target=exact_ed_stop),
            resume=FreshStart(),
        ),
    )


def exact_ed_stop_for_problem(
    problem: ResolvedProblemContext,
    *,
    energy: float,
    absolute_tolerance: float,
    source_id: str = "doron_molecular_closed_shell_sector_ed_v1",
) -> ExactEDStop:
    """An explicit, source-receipted exact-energy stop for callers who ask for one."""

    return ExactEDStop(
        energy=float(energy),
        absolute_tolerance=float(absolute_tolerance),
        source=ExactEDSourceReceipt.from_problem(problem, source_id=str(source_id)),
    )


__all__ += ["exact_ed_stop_for_problem"]
