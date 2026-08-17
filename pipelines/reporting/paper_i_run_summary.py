"""Canonical Paper-I accepted-run summary and prefix compilation seam.

This module consumes the immutable, typed result emitted by the canonical
SR-SNAKE facade.  It deliberately does not load JSON artifacts, search legacy
schemas, infer historical route identities, or recover missing fields.

Qiskit remains an observation boundary: imports and compilation happen only
when :func:`summarize_paper_i_run` requests a prefix observation.  A tooling
failure is returned as a retryable observation failure and never changes the
accepted scientific result.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass, replace
import math
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from pipelines.static_adapt.sr_snake.contracts import SRRunResult


EFFECTIVE_PLATEAU_POLICY = "paper_i_effective_plateau_v1"
LOCKED_QISKIT_COMPILE_CONVENTION = "table_i_basis_gate_transpile_v1"
_HORIZON_SCOPES = frozenset(
    {"natural_terminal", "deliberately_stopped_prefix"}
)


@dataclass(frozen=True)
class CanonicalAppendReference:
    """Typed marker for the exact-problem canonical append resolver."""


CANONICAL_APPEND_REFERENCE = CanonicalAppendReference()


@dataclass(frozen=True)
class PaperIReferenceState:
    amplitudes_real: tuple[float, ...]
    amplitudes_imaginary: tuple[float, ...]
    qubit_count: int
    source_label: str
    state_fingerprint: str

    def __post_init__(self) -> None:
        if self.qubit_count <= 0:
            raise ValueError("reference-state qubit_count must be positive.")
        if not self.source_label or not self.state_fingerprint:
            raise ValueError(
                "reference-state source and fingerprint must be nonempty."
            )
        expected_size = 1 << self.qubit_count
        if (
            len(self.amplitudes_real) != expected_size
            or len(self.amplitudes_imaginary) != expected_size
        ):
            raise ValueError(
                "reference-state amplitudes disagree with qubit_count."
            )
        amplitudes = tuple(
            complex(real, imaginary)
            for real, imaginary in zip(
                self.amplitudes_real,
                self.amplitudes_imaginary,
                strict=True,
            )
        )
        if not all(
            math.isfinite(value.real) and math.isfinite(value.imag)
            for value in amplitudes
        ):
            raise ValueError("reference-state amplitudes must be finite.")
        norm_squared = math.fsum(
            value.real * value.real + value.imag * value.imag
            for value in amplitudes
        )
        if not math.isclose(
            norm_squared,
            1.0,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ):
            raise ValueError("reference-state amplitudes must be normalized.")
        from pipelines.static_adapt.estimator_call_ledger import (
            projective_state_fingerprint,
        )

        observed = projective_state_fingerprint(amplitudes)
        if observed != self.state_fingerprint:
            raise ValueError(
                "reference-state fingerprint does not authenticate amplitudes."
            )


@dataclass(frozen=True)
class PaperIPrefixPauliTerm:
    pauli_exyz: str
    coefficient_real: float
    coefficient_imaginary: float
    qubit_count: int


@dataclass(frozen=True)
class PaperIPrefixOperator:
    candidate_label: str
    logical_index: int
    runtime_start: int
    runtime_count: int
    execution_mode: str
    runtime_terms: tuple[PaperIPrefixPauliTerm, ...]


@dataclass(frozen=True)
class PaperIWorkComponents:
    n_h_outer: int
    n_h_refit: int
    n_grad: int
    n_metric: int

    @property
    def s_alg(self) -> int:
        return (
            self.n_h_outer
            + self.n_h_refit
            + self.n_grad
            + self.n_metric
        )


@dataclass(frozen=True)
class PaperIAlgorithmicWork:
    components: PaperIWorkComponents
    s_alg: int


@dataclass(frozen=True)
class PaperIPrefixCompileInput:
    """Complete, typed input for one locked Paper-I prefix compilation."""

    source_method: str
    controller_round: int
    active_ansatz_depth: int
    ordered_operator_labels: tuple[str, ...]
    operators: tuple[PaperIPrefixOperator, ...]
    logical_parameters: tuple[float, ...]
    runtime_parameters: tuple[float, ...]
    reference_state: PaperIReferenceState
    checkpoint_sha256: str
    projective_state_fingerprint: str
    problem_request_sha256: str
    route_profile: str
    route_contract_sha256: str
    algorithmic_work: PaperIAlgorithmicWork

    def __post_init__(self) -> None:
        _validate_prefix_compile_input(self)

    def with_source_method(self, source_method: str) -> "PaperIPrefixCompileInput":
        """Return the same typed prefix under an explicit method identity."""

        method = str(source_method).strip()
        if not method:
            raise ValueError("source_method must not be empty.")
        return replace(self, source_method=method)

    @property
    def compile_cache_key(self) -> tuple[str, str, str, str, int]:
        return (
            self.source_method,
            self.problem_request_sha256,
            self.route_contract_sha256,
            self.checkpoint_sha256,
            self.controller_round,
        )


@dataclass(frozen=True)
class PaperIQiskitResources:
    compile_convention: str
    compiled_two_qubit_count: int
    compiled_two_qubit_depth: int
    compiled_total_depth: int


@dataclass(frozen=True)
class PaperIObservationFailure:
    exception_type: str
    message: str
    retryable: bool = True


@dataclass(frozen=True)
class PaperIAcceptedError:
    controller_round: int
    active_ansatz_depth: int
    accepted_energy: float
    exact_same_cutoff_energy: float
    absolute_energy_error: float
    projective_state_fingerprint: str
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        round_index = int(self.controller_round)
        depth = int(self.active_ansatz_depth)
        energy = float(self.accepted_energy)
        exact = float(self.exact_same_cutoff_energy)
        error = float(self.absolute_energy_error)
        if (
            isinstance(self.controller_round, bool)
            or round_index != self.controller_round
            or round_index < 1
        ):
            raise ValueError("accepted error controller_round must be positive.")
        if (
            isinstance(self.active_ansatz_depth, bool)
            or depth != self.active_ansatz_depth
            or depth < 1
        ):
            raise ValueError("accepted error active_ansatz_depth must be positive.")
        if not all(math.isfinite(value) for value in (energy, exact, error)):
            raise ValueError("accepted error energies must be finite.")
        if error < 0.0 or not math.isclose(
            error,
            abs(energy - exact),
            rel_tol=1.0e-12,
            abs_tol=1.0e-14,
        ):
            raise ValueError("accepted error row is internally inconsistent.")
        if not str(self.projective_state_fingerprint).strip():
            raise ValueError("accepted error state fingerprint is required.")
        _sha256(self.checkpoint_sha256, name="accepted error checkpoint_sha256")


@dataclass(frozen=True)
class PaperIErrorTracePoint:
    """Minimal typed input for Paper-I trajectory selection policies."""

    controller_round: int
    absolute_energy_error: float

    def __post_init__(self) -> None:
        _positive_int(
            self.controller_round,
            name="error trace controller_round",
        )
        error = _finite(
            self.absolute_energy_error,
            name="error trace absolute_energy_error",
        )
        if error < 0.0:
            raise ValueError(
                "error trace absolute_energy_error must be nonnegative."
            )


@dataclass(frozen=True)
class PaperIEffectivePlateauSelection:
    policy: str
    selected_trace_index: int
    controller_round: int
    absolute_energy_error: float
    best_observed_error: float
    selection_threshold: float
    horizon_controller_rounds: int


@dataclass(frozen=True)
class PaperICommonAccuracySelection:
    shared_window_end_controller_round: int
    common_target_absolute_error: float
    sr_snake_window_minimum_error: float
    append_adapt_window_minimum_error: float
    sr_snake_crossing_trace_index: int
    sr_snake_crossing_controller_round: int
    append_adapt_crossing_trace_index: int
    append_adapt_crossing_controller_round: int
    sr_snake_plateau_controller_round: int
    append_adapt_plateau_controller_round: int


@dataclass(frozen=True)
class PaperIPrefixObservation:
    purpose: str
    status: str
    controller_round: int
    active_ansatz_depth: int
    absolute_energy_error: float
    algorithmic_work: PaperIAlgorithmicWork
    prefix: PaperIPrefixCompileInput
    resources: PaperIQiskitResources | None
    failure: PaperIObservationFailure | None


@dataclass(frozen=True)
class PaperIEffectivePlateauObservation:
    policy: str
    status: str
    controller_round: int
    active_ansatz_depth: int
    absolute_energy_error: float
    best_observed_error: float
    selection_threshold: float
    available_horizon_controller_rounds: int
    horizon_scope: str
    algorithmic_work: PaperIAlgorithmicWork
    prefix: PaperIPrefixCompileInput
    resources: PaperIQiskitResources | None
    failure: PaperIObservationFailure | None


@dataclass(frozen=True)
class PaperIComparisonContract:
    problem_request_sha256: str
    optimizer: str
    optimizer_maxiter: int
    seed: int
    candidate_representation: str
    compile_convention: str = LOCKED_QISKIT_COMPILE_CONVENTION


@dataclass(frozen=True)
class PaperIAppendResolutionRequest:
    comparison_contract: PaperIComparisonContract
    exact_same_cutoff_energy: float
    reference_state: PaperIReferenceState


@dataclass(frozen=True)
class PaperIAppendRunSource:
    """Typed append-ADAPT adapter output accepted by the canonical summary.

    Construction of this object belongs to a source-locked append registry or
    adapter.  The summary itself never searches artifact trees or interprets
    historical JSON.
    """

    comparison_contract: PaperIComparisonContract
    accepted_error_trace: tuple[PaperIAcceptedError, ...]
    accepted_prefixes: tuple[PaperIPrefixCompileInput, ...]
    horizon_scope: str


@runtime_checkable
class PaperIAppendReferenceResolver(Protocol):
    """Resolve one source-locked canonical append comparator."""

    def resolve_canonical_append(
        self,
        request: PaperIAppendResolutionRequest,
    ) -> PaperIAppendRunSource | None:
        ...


class _CanonicalAppendRegistryResolver:
    """Lazy source-locked registry boundary for the ordinary sentinel."""

    def resolve_canonical_append(
        self,
        request: PaperIAppendResolutionRequest,
    ) -> PaperIAppendRunSource | None:
        from pipelines.reporting.paper_i_append_registry import (
            default_paper_i_append_reference_resolver,
        )

        return (
            default_paper_i_append_reference_resolver()
            .resolve_canonical_append(request)
        )


_CANONICAL_APPEND_REGISTRY_RESOLVER = _CanonicalAppendRegistryResolver()


@dataclass(frozen=True)
class PaperIAppendMatchedObservation:
    status: str
    reason: str | None
    shared_window_end_controller_round: int | None
    common_target_absolute_error: float | None
    sr_snake: PaperIPrefixObservation | None
    append_adapt: PaperIPrefixObservation | None
    failure: PaperIObservationFailure | None = None


@dataclass(frozen=True)
class PaperIRunProvenance:
    problem_key: str
    problem_request_sha256: str
    problem_family: str
    exact_target_label: str
    exact_same_cutoff_energy: float
    reference_label: str
    reference_source_label: str
    reference_state_fingerprint: str
    route_family: str
    route_profile_request: str
    route_profile: str
    route_contract_sha256: str
    candidate_representation: str
    optimizer: str
    optimizer_maxiter: int
    seed: int
    qiskit_compile_convention: str


@dataclass(frozen=True)
class PaperIRunSummary:
    schema: ClassVar[str] = "paper_i_run_summary_v1"
    accepted_error_trace: tuple[PaperIAcceptedError, ...]
    effective_plateau: PaperIEffectivePlateauObservation
    append_matched: PaperIAppendMatchedObservation
    requested_rounds: tuple[PaperIPrefixObservation, ...]
    canonical_all_work: PaperIAlgorithmicWork
    horizon_scope: str
    available_controller_rounds: int
    provenance: PaperIRunProvenance

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-ready projection for report builders."""

        return {"schema": self.schema, **asdict(self)}


class _PrefixCompiler(Protocol):
    def __call__(
        self,
        prefix: PaperIPrefixCompileInput,
    ) -> PaperIQiskitResources:
        ...


def _required_attribute(owner: Any, name: str, *, context: str) -> Any:
    if isinstance(owner, Mapping):
        raise TypeError(
            f"{context} must be a typed canonical receipt, not a mapping."
        )
    try:
        return getattr(owner, name)
    except AttributeError as exc:
        raise TypeError(
            f"{context} is missing required canonical field {name!r}."
        ) from exc


def _typed_run_source(run_source: SRRunResult) -> SRRunResult:
    from pipelines.static_adapt.sr_snake.contracts import SRRunResult

    if not isinstance(run_source, SRRunResult):
        raise TypeError(
            "run_source must be a typed canonical SRRunResult."
        )
    for name in (
        "accepted_trajectory",
        "problem",
        "route",
        "stop",
        "scientific_replay",
        "estimator_accounting",
        "canonical_reporting",
    ):
        _required_attribute(run_source, name, context="run_source")
    return run_source


def _route_policy_fields_match(route: Any, method: Any) -> bool:
    expected = {
        "admission_policy": method.admission.kind,
        "insertion_policy": method.insertion.kind,
        "pruning_policy": method.pruning.kind,
        "beam_policy": method.beam.kind,
    }
    return all(
        _nonempty(
            _required_attribute(
                route,
                field,
                context="run_source.route",
            ),
            name=f"run_source.route.{field}",
        )
        == value
        for field, value in expected.items()
    )


def _canonical_ra_supersession_identities(
    method: Any,
    *,
    candidate_representation: str,
) -> tuple[tuple[str, str, str, str], ...]:
    """Reconstruct the finite, authenticated RA route-identity set.

    An RA route is accepted only when its complete serialized contract digest
    matches a contract rebuilt by the canonical RA authority.  The invariant
    and lineage checks below make the supersession relationship explicit;
    merely using an ``ra_adapt``-looking family or profile string is
    insufficient.
    """

    from pipelines.static_adapt.ra_adapt.adapters import (
        GlobalSinglePauliWordCandidateAdapter,
        GlobalSingletonGradientPhase0CandidateAdapter,
        MacroCandidateAdapter,
        MacroGradientPhase0CandidateAdapter,
        MacroGradientPhase0ThenSingletonCandidateAdapter,
        MacroThenSingletonPhaseICandidateAdapter,
        SinglePauliWordCandidateAdapter,
    )
    from pipelines.static_adapt.ra_adapt.contracts import (
        ACTIVE_GRADIENT_POLICIES,
        CANDIDATE_REPRESENTATION_MACRO,
        CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        EXACT_ORDERED_INSERTION_CHART,
        ENDPOINT_OVERLAP_DISPLACEMENT_TRUST,
        FULL_ENLARGED_ACCEPTED_REFIT,
        PROJECTED_GENERALIZED_SOLVER,
        RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V1,
        RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V2,
        RA_ADAPT_SINGLETON_LATCHED_PHASE3_ALGORITHM_ID,
        RA_ADAPT_SINGLETON_PHASE3_PLATEAU_ALGORITHM_ID,
        RAAdaptRequest,
        RESOURCE_WEIGHTING_ALL_PHASE,
        RESOURCE_WEIGHTING_SCOPES,
        SOURCE_GRAM_NO_OVERLAP_TRUST,
        SUPPORTED_FS_WHITENED_REFIT_CHART,
        canonical_sha256,
    )
    from pipelines.static_adapt.ra_adapt.engine import (
        RA_ADAPT_ALGORITHM_ID,
        RA_ADAPT_GLOBAL_SINGLETON_INSERTION_KIND_BY_ALGORITHM_ID,
        RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ALGORITHM_ID,
        RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID,
        RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_DENOMINATOR_NO_LANES_ALGORITHM_ID,
        RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_SOURCE_ALGORITHM_ID,
        RA_ADAPT_GLOBAL_SINGLETON_QISKIT_COST_ALGORITHM_IDS,
        RA_ADAPT_LEGACY_ALGORITHM_ID,
        RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ALGORITHM_ID,
        RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID,
        RA_ADAPT_MACRO_GRADIENT_PHASE0_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID,
        RA_ADAPT_MACRO_QISKIT_COST_INSERTION_KIND_BY_ALGORITHM_ID,
        RA_ADAPT_MACRO_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID,
        RA_ADAPT_PHASE23_QISKIT_ALGORITHM_IDS,
        RA_ADAPT_PHASE3_QISKIT_COST_PHASE_REUSE,
        RA_ADAPT_PHASE3_QISKIT_COST_POLICY,
        RA_ADAPT_PHASE3_QISKIT_DENOMINATOR_POLICY,
        RA_ADAPT_PHASE3_QISKIT_ALGORITHM_IDS,
        _macro_parent_contract,
        _repaired_route_contract,
    )
    from pipelines.static_adapt.hh_backend_compile_oracle import (
        BACKEND_COMPILE_SCOPE_PHASE2_PHASE3_QISKIT_ONLY_V1,
        BACKEND_COMPILE_SCOPE_PHASE3_QISKIT_ONLY_V1,
        MARRAKESH_GRAPH_SPAN_MODE,
        ONE_QUBIT_COORDINATE_COMPILED_POSITIVE_DELTA_V1,
    )
    from pipelines.static_adapt.sr_snake._context import (
        _canonical_route_contract_for_request,
    )
    from pipelines.static_adapt.sr_snake.contracts import SRRunRequest

    if candidate_representation == CANDIDATE_REPRESENTATION_SINGLE_PAULI:
        adapters = (
            SinglePauliWordCandidateAdapter(),
            MacroThenSingletonPhaseICandidateAdapter(),
            MacroGradientPhase0ThenSingletonCandidateAdapter(),
            GlobalSinglePauliWordCandidateAdapter(),
            GlobalSingletonGradientPhase0CandidateAdapter(),
        )
    elif candidate_representation == CANDIDATE_REPRESENTATION_MACRO:
        adapters = (
            MacroCandidateAdapter(),
            MacroGradientPhase0CandidateAdapter(),
        )
    else:
        return ()

    identities: set[tuple[str, str, str, str]] = set()
    for adapter in adapters:
        request = RAAdaptRequest(adapter=adapter, method=method)
        algorithm_ids: tuple[str, ...]
        if type(adapter) is MacroGradientPhase0CandidateAdapter:
            algorithm_ids = (
                (
                    RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID,
                    RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ALGORITHM_ID,
                )
                if str(method.insertion.kind) == "plateau_commutation"
                else ()
            )
        elif isinstance(adapter, MacroCandidateAdapter):
            # Algorithm ids do not select insertion semantics; the typed
            # method does.  They do, however, select the authenticated
            # Qiskit-cost route suffix, so reporting must rebuild those
            # identities with the exact insertion-kind discriminator.
            qiskit_ids = tuple(
                sorted(
                    algorithm_id
                    for algorithm_id, insertion_kind in (
                        RA_ADAPT_MACRO_QISKIT_COST_INSERTION_KIND_BY_ALGORITHM_ID.items()
                    )
                    if insertion_kind == method.insertion.kind
                )
            )
            algorithm_ids = (
                RA_ADAPT_ALGORITHM_ID,
                RA_ADAPT_LEGACY_ALGORITHM_ID,
                "paper_i_ra_adapt_macro_always_insertion_repair_v1",
                *qiskit_ids,
            )
        elif isinstance(
            adapter,
            MacroGradientPhase0ThenSingletonCandidateAdapter,
        ):
            algorithm_ids = (
                (
                    RA_ADAPT_MACRO_GRADIENT_PHASE0_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID,
                )
                if str(method.insertion.kind) == "plateau_commutation"
                else ()
            )
        elif isinstance(
            adapter,
            GlobalSingletonGradientPhase0CandidateAdapter,
        ):
            algorithm_ids = (
                (
                    RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ALGORITHM_ID,
                )
                if str(method.insertion.kind) == "plateau_commutation"
                else ()
            )
        elif isinstance(adapter, GlobalSinglePauliWordCandidateAdapter):
            algorithm_ids = (
                RA_ADAPT_ALGORITHM_ID,
                RA_ADAPT_LEGACY_ALGORITHM_ID,
                *tuple(
                    sorted(
                        algorithm_id
                        for algorithm_id, insertion_kind in (
                            RA_ADAPT_GLOBAL_SINGLETON_INSERTION_KIND_BY_ALGORITHM_ID.items()
                        )
                        if insertion_kind == method.insertion.kind
                    )
                ),
            )
        else:
            algorithm_ids = (
                RA_ADAPT_ALGORITHM_ID,
                RA_ADAPT_LEGACY_ALGORITHM_ID,
                "paper_i_ra_adapt_singleton_plateau_insertion_repair_v1",
                *(
                    (
                        RA_ADAPT_SINGLETON_PHASE3_PLATEAU_ALGORITHM_ID,
                        RA_ADAPT_SINGLETON_LATCHED_PHASE3_ALGORITHM_ID,
                    )
                    if str(method.insertion.kind) == "plateau_commutation"
                    else ()
                ),
                *(
                    (
                        RA_ADAPT_MACRO_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID,
                    )
                    if isinstance(
                        adapter,
                        MacroThenSingletonPhaseICandidateAdapter,
                    )
                    and str(method.insertion.kind)
                    == "plateau_commutation"
                    else ()
                ),
            )
        for algorithm_id in algorithm_ids:
            if candidate_representation == CANDIDATE_REPRESENTATION_MACRO:
                parent_contract, parent_sha256 = _macro_parent_contract(
                    request,
                    algorithm_id=algorithm_id,
                )
                parent_profile = str(
                    parent_contract.get("route_profile", "")
                )
            elif (
                algorithm_id == RA_ADAPT_LEGACY_ALGORITHM_ID
                and str(method.insertion.kind) == "plateau_commutation"
            ):
                from pipelines.static_adapt.sr_snake_route_profile import (
                    canonical_sr_snake_insertion_commutation_plateau_v1_contract,
                    canonical_sr_snake_insertion_commutation_plateau_v1_contract_sha256,
                )

                parent_contract = (
                    canonical_sr_snake_insertion_commutation_plateau_v1_contract()
                )
                parent_profile = str(
                    parent_contract.get("route_profile", "")
                )
                parent_sha256 = (
                    canonical_sr_snake_insertion_commutation_plateau_v1_contract_sha256()
                )
            else:
                (
                    _parent_request,
                    parent_profile,
                    _parent_contract,
                    parent_sha256,
                ) = _canonical_route_contract_for_request(
                    SRRunRequest(method=method)
                )
            for active_gradient_policy in sorted(
                ACTIVE_GRADIENT_POLICIES
            ):
                for resource_weighting_scope in sorted(
                    RESOURCE_WEIGHTING_SCOPES
                ):
                    if algorithm_id == RA_ADAPT_ALGORITHM_ID and (
                        active_gradient_policy != "measured_residual_response_v1"
                    ):
                        continue
                    if algorithm_id in {
                        RA_ADAPT_SINGLETON_PHASE3_PLATEAU_ALGORITHM_ID,
                        RA_ADAPT_SINGLETON_LATCHED_PHASE3_ALGORITHM_ID,
                    } and (
                        active_gradient_policy
                        != "stationary_source_response_v1"
                        or resource_weighting_scope
                        != "late_resource_weighting_v1"
                    ):
                        continue
                    if algorithm_id == (
                        RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID
                    ) and (
                        active_gradient_policy
                        != "stationary_source_response_v1"
                        or resource_weighting_scope
                        != RESOURCE_WEIGHTING_ALL_PHASE
                    ):
                        continue
                    phase3_only_qiskit_algorithm = bool(
                        algorithm_id in RA_ADAPT_PHASE3_QISKIT_ALGORITHM_IDS
                    )
                    staged_phase23_qiskit_algorithm = bool(
                        algorithm_id in RA_ADAPT_PHASE23_QISKIT_ALGORITHM_IDS
                    )
                    denominator_no_lanes = bool(
                        algorithm_id
                        == RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_DENOMINATOR_NO_LANES_ALGORITHM_ID
                    )
                    if (
                        phase3_only_qiskit_algorithm
                        or staged_phase23_qiskit_algorithm
                    ) and (
                        active_gradient_policy
                        != "stationary_source_response_v1"
                        or resource_weighting_scope
                        != RESOURCE_WEIGHTING_ALL_PHASE
                    ):
                        continue
                    qiskit_cost_algorithm = bool(
                            phase3_only_qiskit_algorithm
                            or staged_phase23_qiskit_algorithm
                        or algorithm_id
                        in (
                            RA_ADAPT_GLOBAL_SINGLETON_QISKIT_COST_ALGORITHM_IDS
                        )
                        or algorithm_id
                        in (
                            RA_ADAPT_MACRO_QISKIT_COST_INSERTION_KIND_BY_ALGORITHM_ID
                        )
                    )
                    if (
                        qiskit_cost_algorithm
                        and resource_weighting_scope
                        != RESOURCE_WEIGHTING_ALL_PHASE
                    ):
                        continue
                    (
                        profile_request,
                        profile,
                        contract,
                        contract_sha256,
                    ) = _repaired_route_contract(
                        request,
                        active_gradient_policy=active_gradient_policy,
                        resource_weighting_scope=resource_weighting_scope,
                        algorithm_id=algorithm_id,
                    )
                    invariants = contract.get("semantic_invariants")
                    lineage = contract.get("lineage_authority")
                    if not isinstance(
                        invariants, Mapping
                    ) or not isinstance(lineage, Mapping):
                        raise RuntimeError(
                            "Canonical RA contract lost its invariant or "
                            "lineage mapping."
                        )
                    endpoint_overlap_trust = bool(
                        getattr(method, "trust_update", None) is not None
                        and getattr(
                            method.trust_update,
                            "kind",
                            "",
                        )
                        == "endpoint_overlap_displacement"
                    )
                    required_invariants = {
                        "canonical_interface": (
                            "run_ra_adapt_problem_request_v1"
                        ),
                        "candidate_representation": candidate_representation,
                        "result_candidate_representation": (
                            candidate_representation
                        ),
                        "candidate_geometry_chart": (
                            EXACT_ORDERED_INSERTION_CHART
                        ),
                        "phase3_solver": PROJECTED_GENERALIZED_SOLVER,
                        "phase3_metric_ridge": 0.0,
                        "phase3_whitening_active": False,
                        "phase3_inverse_sqrt_constructed": False,
                        "trust_policy": (
                            ENDPOINT_OVERLAP_DISPLACEMENT_TRUST
                            if endpoint_overlap_trust
                            else SOURCE_GRAM_NO_OVERLAP_TRUST
                        ),
                        "endpoint_overlap_required": endpoint_overlap_trust,
                        "endpoint_overlap_measurement_active": (
                            endpoint_overlap_trust
                        ),
                        "endpoint_overlap_query_charge_required": (
                            1 if endpoint_overlap_trust else 0
                        ),
                        "accepted_refit_scope": FULL_ENLARGED_ACCEPTED_REFIT,
                        "accepted_refit_coordinate_chart": (
                            SUPPORTED_FS_WHITENED_REFIT_CHART
                        ),
                        "active_gradient_policy": active_gradient_policy,
                        "resource_weighting_scope": (
                            resource_weighting_scope
                        ),
                    }
                    expected_supersession_reason = (
                        "paper_i_ra_adapt_nonstationary_full_response_v2_20260731"
                        if algorithm_id == RA_ADAPT_ALGORITHM_ID
                        else (
                            "paper_i_ra_adapt_macro_gradient_phase0_proxy_"
                            "no_lanes_candidate_20260810"
                        )
                        if algorithm_id
                        == RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID
                        else (
                            "paper_i_ra_adapt_macro_gradient_phase0_macro_"
                            "phase123_phase23_qiskit_candidate_20260811"
                            if algorithm_id
                            == RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ALGORITHM_ID
                            else
                            "paper_i_ra_adapt_macro_gradient_phase0_then_"
                            "singleton_phase123_phase23_qiskit_candidate_20260807"
                            if algorithm_id
                            == RA_ADAPT_MACRO_GRADIENT_PHASE0_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID
                            else "paper_i_ra_adapt_global_singleton_gradient_"
                            "phase0_phase123_phase23_qiskit_candidate_20260807"
                            if algorithm_id
                            == RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ALGORITHM_ID
                            else "paper_i_ra_adapt_macro_then_singleton_phase123_"
                            "phase23_qiskit_candidate_20260807"
                        )
                        if staged_phase23_qiskit_algorithm
                        else (
                            (
                                "paper_i_ra_adapt_phase3_qiskit_denominator_"
                                "no_lanes_tau1em6_candidate_20260806"
                            )
                            if denominator_no_lanes
                            else (
                                "paper_i_ra_adapt_phase3_only_qiskit_cost_"
                                "candidate_20260806"
                            )
                        )
                        if phase3_only_qiskit_algorithm
                        else (
                            "paper_i_ra_adapt_singleton_phase3_plateau_"
                            "ablation_20260802"
                        )
                        if algorithm_id
                        == RA_ADAPT_SINGLETON_PHASE3_PLATEAU_ALGORITHM_ID
                        else (
                            "paper_i_ra_adapt_singleton_latched_phase3_"
                            "separate_plateau_insertion_20260804"
                        )
                        if algorithm_id
                        == RA_ADAPT_SINGLETON_LATCHED_PHASE3_ALGORITHM_ID
                        else "paper_i_ra_adapt_unification_repair_20260727"
                    )
                    if algorithm_id == RA_ADAPT_ALGORITHM_ID:
                        required_invariants.update(
                            {
                                "phase3_candidate_gain_policy": (
                                    "joint_minus_active_only_supported_trust_v1"
                                ),
                                "phase3_candidate_gain_semantics": (
                                    "full_joint_minus_candidate_independent_"
                                    "active_only_v1"
                                ),
                                "accepted_refit_initialization_policy": (
                                    "exact_applied_joint_step_guarded_v1"
                                ),
                                "accepted_refit_initialization_coordinate_scope": (
                                    "full_existing_active_plus_new_batch_"
                                    "coordinates_v1"
                                ),
                            }
                        )
                    if algorithm_id in {
                        RA_ADAPT_MACRO_GRADIENT_PHASE0_PROXY_NO_LANES_ALGORITHM_ID,
                        RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ALGORITHM_ID,
                    }:
                        required_invariants.update(
                            {
                                "phase0_active": True,
                                "phase0_score": (
                                    "standard_adapt_absolute_gradient_v1"
                                ),
                                "phase0_fubini_metric_active": False,
                                "phase0_resource_cost_active": False,
                                "phase0_compile_cost_active": False,
                                "phase0_estimator_components": ["N_grad"],
                                "physical_operator_lanes_active": False,
                                "shortlist_population_policy": (
                                    "single_global_population_v1"
                                ),
                                "selector_qiskit_compile_cost_active": (
                                    algorithm_id
                                    == RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ALGORITHM_ID
                                ),
                                "macro_generator_identity_preserved_all_phases": (
                                    True
                                ),
                                "singleton_child_exposure_active": False,
                                "plateau_prior_mean_decrease_ratio_threshold": (
                                    1.0e-4
                                ),
                            }
                        )
                    if phase3_only_qiskit_algorithm:
                        required_invariants.update(
                            {
                                "selector_compile_cost_policy": (
                                    RA_ADAPT_PHASE3_QISKIT_DENOMINATOR_POLICY
                                    if denominator_no_lanes
                                    else RA_ADAPT_PHASE3_QISKIT_COST_POLICY
                                ),
                                "selector_compile_cost_phase_reuse": (
                                    RA_ADAPT_PHASE3_QISKIT_COST_PHASE_REUSE
                                ),
                                "selector_compile_cost_scope": (
                                    BACKEND_COMPILE_SCOPE_PHASE3_QISKIT_ONLY_V1
                                ),
                                "phase_i_phase_ii_compile_cost_source": (
                                    MARRAKESH_GRAPH_SPAN_MODE
                                ),
                                "phase_iii_compile_cost_source": (
                                    "backend_transpile_v1"
                                ),
                                "phase_iii_qiskit_backend_fallback_allowed": (
                                    False
                                ),
                                "phase_iii_qiskit_negative_delta_reward_enabled": (
                                    False
                                ),
                                "phase_iii_qiskit_raw_signed_telemetry_required": (
                                    True
                                ),
                                "phase_iii_qiskit_structure_theta_value": 1.0,
                                "phase_iii_qiskit_full_base_trial_ansatz_transpile": (
                                    True
                                ),
                                "phase_iii_qiskit_independent_base_trial_layouts": (
                                    True
                                ),
                                "phase_iii_qiskit_one_qubit_coordinate_policy": (
                                    ONE_QUBIT_COORDINATE_COMPILED_POSITIVE_DELTA_V1
                                ),
                                "phase_iii_qiskit_selector_circuit_coordinates": [
                                    "positive_clip_delta_N2q",
                                    "positive_clip_delta_D2q",
                                    "positive_clip_delta_N1q",
                                ],
                                "phase_iii_qiskit_population_rescore_policy": (
                                    "complete_evaluated_phase3_population_"
                                    "before_ranking_v1"
                                ),
                                "phase_iii_qiskit_population_normalization_policy": (
                                    "family_robust_v1"
                                    if denominator_no_lanes
                                    else "family_robust_symmetric_arctan_v1"
                                ),
                                "phase_iii_qiskit_failure_policy": (
                                    "abort_run_v1"
                                ),
                            }
                        )
                        if denominator_no_lanes:
                            required_invariants.update(
                                {
                                    "physical_operator_lanes_active": False,
                                    "shortlist_population_policy": (
                                        "single_global_population_v1"
                                    ),
                                    "plateau_prior_mean_decrease_ratio_threshold": (
                                        1.0e-6
                                    ),
                                    "phase_iii_score_formula": (
                                        "B3/(1+lambda_2q*cbar_2q+"
                                        "lambda_d*cbar_d+lambda_1q*cbar_1q)"
                                    ),
                                }
                            )
                    if staged_phase23_qiskit_algorithm:
                        required_invariants.update(
                            {
                                "selector_compile_cost_scope": (
                                    BACKEND_COMPILE_SCOPE_PHASE2_PHASE3_QISKIT_ONLY_V1
                                ),
                                "phase_i_compile_cost_source": (
                                    "structural_proxy_v1"
                                ),
                                "phase_ii_compile_cost_source": (
                                    "backend_transpile_v1"
                                ),
                                "phase_iii_compile_cost_source": (
                                    "backend_transpile_v1"
                                ),
                                "phase_ii_phase_iii_qiskit_negative_delta_reward_enabled": (
                                    True
                                ),
                                "physical_operator_lanes_active": False,
                                "shortlist_population_policy": (
                                    "single_global_population_v1"
                                ),
                            }
                        )
                        if algorithm_id in {
                            RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ALGORITHM_ID,
                            RA_ADAPT_MACRO_GRADIENT_PHASE0_THEN_SINGLETON_PHASE23_QISKIT_ALGORITHM_ID,
                            RA_ADAPT_GLOBAL_SINGLETON_GRADIENT_PHASE0_PHASE23_QISKIT_ALGORITHM_ID,
                        }:
                            required_invariants.update(
                                {
                                    "phase0_active": True,
                                    "phase0_score": (
                                        "standard_adapt_absolute_gradient_v1"
                                    ),
                                    "phase0_fubini_metric_active": False,
                                    "phase0_resource_cost_active": False,
                                    "phase0_compile_cost_active": False,
                                    "phase0_estimator_components": ["N_grad"],
                                }
                            )
                    if algorithm_id == (
                        RA_ADAPT_SINGLETON_PHASE3_PLATEAU_ALGORITHM_ID
                    ):
                        required_invariants.update(
                            {
                                "phase1_activation_scope": (
                                    "all_controller_rounds_v1"
                                ),
                                "phase2_activation_scope": (
                                    "all_controller_rounds_v1"
                                ),
                                "phase3_competitive_population_activation": (
                                    "same_round_insertion_plateau_predicate_v1"
                                ),
                                "phase3_preplateau_admission_authority": (
                                    "phase2_raw_score_top_rank_v1"
                                ),
                            }
                        )
                    if algorithm_id == (
                        RA_ADAPT_SINGLETON_LATCHED_PHASE3_ALGORITHM_ID
                    ):
                        required_invariants.update(
                            {
                                "phase1_activation_scope": (
                                    "all_controller_rounds_v1"
                                ),
                                "phase2_activation_scope": (
                                    "all_controller_rounds_v1"
                                ),
                                "phase3_competitive_population_activation": (
                                    "first_open_progress_plateau_predicate_"
                                    "latched_v1"
                                ),
                                "phase3_preplateau_admission_authority": (
                                    "phase2_raw_score_top_rank_v1"
                                ),
                                "phase3_activation_independent_latch": True,
                                "phase3_latch_retirement_policy": (
                                    "never_close_v1"
                                ),
                                "insertion_plateau_history_scope": (
                                    "prior_full_phase3_accepted_transition_"
                                    "global_prior_mean_v1"
                                ),
                            }
                        )
                    expected_parent_profile = parent_profile
                    expected_parent_sha256 = parent_sha256
                    if phase3_only_qiskit_algorithm:
                        (
                            _page7_request,
                            expected_parent_profile,
                            _page7_contract,
                            expected_parent_sha256,
                        ) = _repaired_route_contract(
                            request,
                            active_gradient_policy=active_gradient_policy,
                            resource_weighting_scope=resource_weighting_scope,
                            algorithm_id=(
                                RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_ALGORITHM_ID
                                if denominator_no_lanes
                                else RA_ADAPT_GLOBAL_SINGLETON_PHASE3_QISKIT_SOURCE_ALGORITHM_ID
                            ),
                        )
                    elif algorithm_id == (
                        RA_ADAPT_SINGLETON_LATCHED_PHASE3_ALGORITHM_ID
                    ):
                        (
                            _page8_request,
                            expected_parent_profile,
                            _page8_contract,
                            expected_parent_sha256,
                        ) = _repaired_route_contract(
                            request,
                            active_gradient_policy=active_gradient_policy,
                            resource_weighting_scope=resource_weighting_scope,
                            algorithm_id=(
                                RA_ADAPT_SINGLETON_PHASE3_PLATEAU_ALGORITHM_ID
                            ),
                        )
                    if (
                        contract.get("schema")
                        != (
                            RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V2
                            if algorithm_id == RA_ADAPT_ALGORITHM_ID
                            else RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V1
                        )
                        or contract.get("route_family") != "ra_adapt"
                        or any(
                            invariants.get(key) != value
                            for key, value in required_invariants.items()
                        )
                        or lineage.get("parent_route_profile")
                        != expected_parent_profile
                        or lineage.get("parent_contract_sha256")
                        != expected_parent_sha256
                        or lineage.get("supersession_reason")
                        != expected_supersession_reason
                        or lineage.get("scientific_result_anchor_claimed")
                        is not False
                        or canonical_sha256(contract) != contract_sha256
                    ):
                        raise RuntimeError(
                            "Canonical RA route authority produced an "
                            "invalid supersession contract."
                        )
                    identities.add(
                        (
                            str(contract["route_family"]),
                            str(profile_request),
                            str(profile),
                            str(contract_sha256),
                        )
                    )
    return tuple(sorted(identities))


def _canonical_ra_semantic_closure_identities(
    method: Any,
    *,
    candidate_representation: str,
) -> tuple[tuple[str, str, str, str], ...]:
    """Rebuild executable semantic-route identities for Paper-I reporting."""

    from pipelines.static_adapt.ra_adapt.contracts import (
        ACTIVE_GRADIENT_STATIONARY,
        CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        RAAdaptRequest,
        RESOURCE_WEIGHTING_ALL_PHASE,
        canonical_sha256,
    )
    from pipelines.static_adapt.ra_adapt.engine import (
        _repaired_route_contract,
    )
    from pipelines.static_adapt.ra_adapt.semantic_closure_routes import (
        PAPER_I_RA_PHASE0_EXECUTABLE_ROUTE_VARIANTS,
        PaperIRASemanticClosureGlobalSingletonCandidateAdapter,
        semantic_closure_route_identity,
    )
    from pipelines.static_adapt.sr_snake.contracts import (
        SRExecutionPolicy,
        SRStopPolicy,
    )

    if candidate_representation != CANDIDATE_REPRESENTATION_SINGLE_PAULI:
        return ()
    identities: set[tuple[str, str, str, str]] = set()
    for route_variant in sorted(PAPER_I_RA_PHASE0_EXECUTABLE_ROUTE_VARIANTS):
        identity = semantic_closure_route_identity(route_variant)
        for horizon in range(1, 51):
            request = RAAdaptRequest(
                adapter=PaperIRASemanticClosureGlobalSingletonCandidateAdapter(
                    route_variant=route_variant,
                ),
                method=method,
                execution=SRExecutionPolicy(
                    stop=SRStopPolicy(
                        maximum_controller_rounds=horizon,
                    )
                ),
            )
            try:
                profile_request, profile, contract, contract_sha256 = (
                    _repaired_route_contract(
                        request,
                        active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
                        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
                        algorithm_id=identity.algorithm_id,
                    )
                )
            except (TypeError, ValueError):
                continue
            native = contract.get("native_semantic_contract")
            if (
                contract.get("route_family") != "ra_adapt"
                or contract.get("algorithm_id") != identity.algorithm_id
                or contract.get("route_id") != identity.route_id
                or contract.get("semantic_implementation_version")
                != identity.semantic_implementation_version
                or not isinstance(native, Mapping)
                or native.get("route_variant") != route_variant
                or native.get("horizon") != horizon
                or canonical_sha256(contract) != contract_sha256
            ):
                raise RuntimeError(
                    "Canonical semantic RA authority produced an invalid route."
                )
            identities.add(
                (
                    str(contract["route_family"]),
                    str(profile_request),
                    str(profile),
                    str(contract_sha256),
                )
            )
    return tuple(sorted(identities))


def _validate_canonical_identity(run_source: SRRunResult) -> None:
    from pipelines.static_adapt.ra_adapt.contracts import (
        ACTIVE_GRADIENT_STATIONARY,
        CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        RAAdaptRequest,
        RESOURCE_WEIGHTING_ALL_PHASE,
    )
    from pipelines.static_adapt.ra_adapt.engine import (
        _repaired_route_contract,
    )
    from pipelines.static_adapt.sr_snake._context import (
        _canonical_route_contract_for_request,
    )
    from pipelines.static_adapt.sr_snake.contracts import (
        CANONICAL_CANDIDATE_REPRESENTATION,
        SRMethodPolicy,
        SRRunRequest,
    )
    from pipelines.static_adapt.ra_adapt.l3_page12 import (
        PAPER_I_L3_PAGE12_ROUTE_CONTRACT_SHA256,
    )
    from pipelines.static_adapt.ra_adapt.pools import (
        PAPER_I_L3_PAGE12_PROBLEM_REQUEST_SHA256,
        PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_PROBLEM_LOCKS,
    )
    from pipelines.static_adapt.ra_adapt.pure_hubbard_noise_page12 import (
        PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ALGORITHM_ID,
        PaperIPureHubbardNoisePage12CandidateAdapter,
    )

    problem = run_source.problem
    route = run_source.route
    execution = _required_attribute(
        route,
        "execution",
        context="run_source.route",
    )
    method = _required_attribute(
        route,
        "method",
        context="run_source.route",
    )
    if not isinstance(method, SRMethodPolicy):
        raise TypeError("run_source.route.method must be an SRMethodPolicy.")
    observed_identity = (
        _nonempty(
            _required_attribute(
                route,
                "family",
                context="run_source.route",
            ),
            name="run_source.route.family",
        ),
        _nonempty(
            _required_attribute(
                route,
                "profile_request",
                context="run_source.route",
            ),
            name="run_source.route.profile_request",
        ),
        _nonempty(
            _required_attribute(
                route,
                "profile",
                context="run_source.route",
            ),
            name="run_source.route.profile",
        ),
        _sha256(
            _required_attribute(
                route,
                "contract_sha256",
                context="run_source.route",
            ),
            name="run_source.route.contract_sha256",
        ),
    )
    representation = str(
        _required_attribute(
            run_source.canonical_reporting,
            "candidate_representation",
            context="canonical_reporting",
        )
    )
    num_sites = _positive_int(
        _required_attribute(
            problem,
            "num_sites",
            context="run_source.problem",
        ),
        name="run_source.problem.num_sites",
    )
    named_l3_problem_request_sha256s = {
        PAPER_I_L3_PAGE12_PROBLEM_REQUEST_SHA256,
        *(
            str(lock["problem_request_sha256"])
            for lock in (
                PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_PROBLEM_LOCKS.values()
            )
        ),
    }
    named_l3_page12_application = bool(
        num_sites == 3
        and observed_identity[0] == "ra_adapt"
        and observed_identity[3]
        == PAPER_I_L3_PAGE12_ROUTE_CONTRACT_SHA256
        and str(
            _required_attribute(
                problem,
                "problem_request_sha256",
                context="run_source.problem",
            )
        )
        in named_l3_problem_request_sha256s
    )
    pure_route_identities: set[tuple[str, str, str, str]] = set()
    if (
        observed_identity[0] == "ra_adapt"
        and representation == CANDIDATE_REPRESENTATION_SINGLE_PAULI
        and str(
            _required_attribute(
                problem,
                "family_key",
                context="run_source.problem",
            )
        )
        == "hubbard"
        and num_sites == 2
        and str(method.insertion.kind) == "plateau_commutation"
    ):
        for noise_level_id in ("low", "high", "extreme"):
            pure_request = RAAdaptRequest(
                adapter=PaperIPureHubbardNoisePage12CandidateAdapter(
                    noise_level_id=noise_level_id
                ),
                method=method,
            )
            try:
                (
                    pure_profile_request,
                    pure_profile,
                    _pure_contract,
                    pure_contract_sha256,
                ) = _repaired_route_contract(
                    pure_request,
                    active_gradient_policy=ACTIVE_GRADIENT_STATIONARY,
                    resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
                    algorithm_id=(
                        PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ALGORITHM_ID
                    ),
                )
            except (TypeError, ValueError):
                # A typed lookalike is not the named application.  Let the
                # ordinary canonical-identity check below reject it with the
                # established reporting error instead of leaking route-builder
                # validation from this narrow exception path.
                pure_route_identities.clear()
                break
            pure_route_identities.add(
                (
                    "ra_adapt",
                    pure_profile_request,
                    pure_profile,
                    pure_contract_sha256,
                )
            )
    named_pure_hubbard_application = bool(
        observed_identity in pure_route_identities
        and representation == CANDIDATE_REPRESENTATION_SINGLE_PAULI
        and str(
            _required_attribute(
                problem,
                "family_key",
                context="run_source.problem",
            )
        )
        == "hubbard"
        and num_sites == 2
        and float(
            _required_attribute(problem, "t", context="run_source.problem")
        )
        == 1.0
        and float(
            _required_attribute(problem, "u", context="run_source.problem")
        )
        in {1.5, 8.0}
        and float(
            _required_attribute(problem, "dv", context="run_source.problem")
        )
        == 0.0
        and float(
            _required_attribute(
                problem, "omega0", context="run_source.problem"
            )
        )
        == 0.0
        and float(
            _required_attribute(problem, "g_ep", context="run_source.problem")
        )
        == 0.0
        and int(
            _required_attribute(
                problem, "n_ph_max", context="run_source.problem"
            )
        )
        == 0
        and str(
            _required_attribute(
                problem, "ordering", context="run_source.problem"
            )
        )
        == "blocked"
        and str(
            _required_attribute(
                problem, "boundary", context="run_source.problem"
            )
        )
        == "open"
        and int(
            _required_attribute(
                problem, "n_fermions", context="run_source.problem"
            )
        )
        == 2
        and int(
            _required_attribute(
                problem, "total_qubits", context="run_source.problem"
            )
        )
        == 4
    )
    semantic_route_identities = set(
        _canonical_ra_semantic_closure_identities(
            method,
            candidate_representation=representation,
        )
    )
    named_semantic_application = bool(
        observed_identity in semantic_route_identities
    )
    if observed_identity[0] == "ra_adapt":
        expected_identities = set(
            _canonical_ra_supersession_identities(
                method,
                candidate_representation=representation,
            )
        )
        expected_identities.update(semantic_route_identities)
        route_identity_matches = bool(
            observed_identity in expected_identities
            or named_l3_page12_application
            or named_pure_hubbard_application
        )
        representation_matches = bool(
            expected_identities
            or named_l3_page12_application
            or named_pure_hubbard_application
        )
    else:
        (
            expected_profile_request,
            expected_profile,
            expected_contract,
            expected_contract_sha256,
        ) = _canonical_route_contract_for_request(
            SRRunRequest(method=method)
        )
        expected_identity = (
            _nonempty(
                expected_contract.get("route_family"),
                name="canonical route contract family",
            ),
            expected_profile_request,
            expected_profile,
            expected_contract_sha256,
        )
        expected_identities = {expected_identity}
        if str(method.insertion.kind) == "plateau_commutation":
            from pipelines.static_adapt.sr_snake_route_profile import (
                canonical_sr_snake_insertion_commutation_plateau_v1_contract,
                canonical_sr_snake_insertion_commutation_plateau_v1_contract_sha256,
            )

            legacy_contract = (
                canonical_sr_snake_insertion_commutation_plateau_v1_contract()
            )
            legacy_profile = _nonempty(
                legacy_contract.get("route_profile"),
                name="legacy canonical route profile",
            )
            expected_identities.add(
                (
                    _nonempty(
                        legacy_contract.get("route_family"),
                        name="legacy canonical route family",
                    ),
                    legacy_profile,
                    legacy_profile,
                    canonical_sr_snake_insertion_commutation_plateau_v1_contract_sha256(),
                )
            )
        route_identity_matches = observed_identity in expected_identities
        representation_matches = (
            representation == CANONICAL_CANDIDATE_REPRESENTATION
        )
    if (
        not route_identity_matches
        or not _route_policy_fields_match(route, method)
    ):
        raise ValueError(
            "canonical Paper-I summary route identity disagrees with the "
            "typed canonical route authority."
        )
    if (
        str(
            _required_attribute(
                problem,
                "family_key",
                context="run_source.problem",
            )
        )
        != "hh"
        and not named_pure_hubbard_application
    ):
        raise ValueError(
            "canonical Paper-I summary requires Hubbard-Holstein or the "
            "exact named pure-Hubbard full-noise application."
        )
    if num_sites != 2 and not named_l3_page12_application:
        raise ValueError(
            "canonical Paper-I summary requires Hubbard-Holstein L=2 or "
            "the exact named Page-12 L=3 application."
        )
    insertion_policy = str(
        _required_attribute(
            route,
            "insertion_policy",
            context="run_source.route",
        )
    )
    insertion_policy_supported = bool(
        insertion_policy
        in {
            "always_commutation_reduced",
            "append_commutation_reduced",
            "plateau_commutation",
        }
        or (
            observed_identity[0] == "ra_adapt"
            and insertion_policy == "append_only"
        )
    )
    if not insertion_policy_supported:
        raise ValueError(
            "canonical Paper-I summary requires a typed active insertion "
            "policy, or an authenticated RA append-only route; historical "
            "append-only replay remains on its explicit compatibility path."
        )
    if (
        str(
            _required_attribute(
                execution,
                "pool",
                context="run_source.route.execution",
            )
        )
        != "full_meta"
    ):
        raise ValueError("canonical Paper-I summary requires the full_meta pool.")
    if bool(
        _required_attribute(
            execution,
            "phase0_enabled",
            context="run_source.route.execution",
        )
    ) and not (
        named_pure_hubbard_application or named_semantic_application
    ):
        raise ValueError("canonical Paper-I summary does not accept Phase 0.")
    if bool(
        _required_attribute(
            execution,
            "phase_live_hysteresis_enabled",
            context="run_source.route.execution",
        )
    ):
        raise ValueError(
            "canonical Paper-I summary does not accept phase-live hysteresis."
        )
    if not representation_matches:
        if observed_identity[0] != "ra_adapt":
            raise ValueError(
                "canonical Paper-I summary requires the hard-guarded "
                "cardinality-one Pauli-child representation."
            )
        raise ValueError(
            "canonical Paper-I summary candidate representation disagrees "
            "with its authenticated route identity."
        )


def _finite(value: Any, *, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    return resolved


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer.")
    resolved = int(value)
    if resolved != value or resolved < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return resolved


def canonical_paper_i_algorithmic_work(
    *,
    n_h_outer: int,
    n_h_refit: int,
    n_grad: int,
    n_metric: int,
) -> PaperIAlgorithmicWork:
    """Close the canonical four-component Paper-I estimator-work receipt."""

    components = PaperIWorkComponents(
        n_h_outer=_nonnegative_int(n_h_outer, name="n_h_outer"),
        n_h_refit=_nonnegative_int(n_h_refit, name="n_h_refit"),
        n_grad=_nonnegative_int(n_grad, name="n_grad"),
        n_metric=_nonnegative_int(n_metric, name="n_metric"),
    )
    return PaperIAlgorithmicWork(
        components=components,
        s_alg=components.s_alg,
    )


def _positive_int(value: Any, *, name: str) -> int:
    resolved = _nonnegative_int(value, name=name)
    if resolved < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return resolved


def _nonempty(value: Any, *, name: str) -> str:
    resolved = str(value).strip()
    if not resolved:
        raise ValueError(f"{name} must not be empty.")
    return resolved


def _sha256(value: Any, *, name: str) -> str:
    resolved = str(value).strip().lower()
    if len(resolved) != 64 or any(
        char not in "0123456789abcdef" for char in resolved
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return resolved


def _horizon_scope(value: Any) -> str:
    resolved = str(value).strip()
    if resolved not in _HORIZON_SCOPES:
        raise ValueError(
            "canonical_reporting.horizon_scope must be "
            "'natural_terminal' or 'deliberately_stopped_prefix'."
        )
    return resolved


def _validate_projected_work(
    work: PaperIAlgorithmicWork,
    *,
    name: str,
) -> None:
    if not isinstance(work, PaperIAlgorithmicWork) or not isinstance(
        work.components,
        PaperIWorkComponents,
    ):
        raise TypeError(f"{name} must be typed Paper-I algorithmic work.")
    values = (
        _nonnegative_int(
            work.components.n_h_outer,
            name=f"{name}.components.n_h_outer",
        ),
        _nonnegative_int(
            work.components.n_h_refit,
            name=f"{name}.components.n_h_refit",
        ),
        _nonnegative_int(
            work.components.n_grad,
            name=f"{name}.components.n_grad",
        ),
        _nonnegative_int(
            work.components.n_metric,
            name=f"{name}.components.n_metric",
        ),
    )
    s_alg = _nonnegative_int(work.s_alg, name=f"{name}.s_alg")
    if sum(values) != s_alg:
        raise ValueError(f"{name} fails canonical component closure.")


def _validate_prefix_compile_input(prefix: PaperIPrefixCompileInput) -> None:
    """Validate one compiler-ready prefix without consulting another schema."""

    if prefix.source_method not in {
        "sr_snake",
        "ra_adapt",
        "append_adapt",
    }:
        raise ValueError(
            "prefix source_method must be sr_snake, ra_adapt, or "
            "append_adapt."
        )
    round_index = _positive_int(
        prefix.controller_round,
        name="prefix.controller_round",
    )
    depth = _positive_int(
        prefix.active_ansatz_depth,
        name="prefix.active_ansatz_depth",
    )
    labels = tuple(
        _nonempty(value, name="prefix.ordered_operator_labels")
        for value in prefix.ordered_operator_labels
    )
    if (
        len(labels) != depth
        or len(prefix.operators) != depth
        or len(prefix.logical_parameters) != depth
    ):
        raise ValueError(
            "prefix operator labels, operators, and logical parameters must "
            "match active_ansatz_depth."
        )
    logical = tuple(
        _finite(value, name=f"prefix.logical_parameters[{index}]")
        for index, value in enumerate(prefix.logical_parameters)
    )
    runtime = tuple(
        _finite(value, name=f"prefix.runtime_parameters[{index}]")
        for index, value in enumerate(prefix.runtime_parameters)
    )
    del logical
    reference = prefix.reference_state
    if not isinstance(reference, PaperIReferenceState):
        raise TypeError("prefix.reference_state must be typed.")
    qubits = _positive_int(
        reference.qubit_count,
        name="prefix.reference_state.qubit_count",
    )
    real = tuple(
        _finite(value, name="prefix.reference_state.amplitudes_real")
        for value in reference.amplitudes_real
    )
    imaginary = tuple(
        _finite(value, name="prefix.reference_state.amplitudes_imaginary")
        for value in reference.amplitudes_imaginary
    )
    if len(real) != 2**qubits or len(imaginary) != len(real):
        raise ValueError("prefix reference state does not cover its register.")
    norm_squared = math.fsum(
        real_part * real_part + imaginary_part * imaginary_part
        for real_part, imaginary_part in zip(real, imaginary, strict=True)
    )
    if not math.isclose(
        norm_squared,
        1.0,
        rel_tol=1.0e-10,
        abs_tol=1.0e-12,
    ):
        raise ValueError("prefix reference state must be normalized.")
    _nonempty(
        reference.source_label,
        name="prefix.reference_state.source_label",
    )
    supplied_reference_fingerprint = _nonempty(
        reference.state_fingerprint,
        name="prefix.reference_state.state_fingerprint",
    )
    from pipelines.static_adapt.estimator_call_ledger import (
        projective_state_fingerprint,
    )

    expected_reference_fingerprint = projective_state_fingerprint(
        tuple(
            complex(real_part, imaginary_part)
            for real_part, imaginary_part in zip(
                real,
                imaginary,
                strict=True,
            )
        )
    )
    if supplied_reference_fingerprint != expected_reference_fingerprint:
        raise ValueError(
            "prefix reference-state fingerprint does not authenticate its "
            "amplitudes."
        )
    expected_runtime_start = 0
    for index, operator in enumerate(prefix.operators):
        if not isinstance(operator, PaperIPrefixOperator):
            raise TypeError(f"prefix.operators[{index}] must be typed.")
        if (
            operator.logical_index != index
            or operator.candidate_label != labels[index]
            or operator.runtime_start != expected_runtime_start
            or operator.runtime_count != len(operator.runtime_terms)
            or operator.runtime_count < 1
        ):
            raise ValueError(
                "prefix operators do not form the exact ordered logical/runtime "
                "partition."
            )
        if operator.execution_mode not in {
            "termwise_product",
            "grouped_exact",
        }:
            raise ValueError("prefix operator execution mode is unsupported.")
        for term_index, term in enumerate(operator.runtime_terms):
            if not isinstance(term, PaperIPrefixPauliTerm):
                raise TypeError(
                    f"prefix operator {index} term {term_index} must be typed."
                )
            if (
                term.qubit_count != qubits
                or len(term.pauli_exyz) != qubits
                or any(symbol not in "exyz" for symbol in term.pauli_exyz)
            ):
                raise ValueError(
                    "prefix Pauli term violates the canonical exyz register."
                )
            _finite(
                term.coefficient_real,
                name="prefix Pauli coefficient real part",
            )
            _finite(
                term.coefficient_imaginary,
                name="prefix Pauli coefficient imaginary part",
            )
        expected_runtime_start += operator.runtime_count
    if expected_runtime_start != len(runtime):
        raise ValueError(
            "prefix operators do not exactly partition runtime parameters."
        )
    _sha256(prefix.checkpoint_sha256, name="prefix.checkpoint_sha256")
    _nonempty(
        prefix.projective_state_fingerprint,
        name="prefix.projective_state_fingerprint",
    )
    _sha256(
        prefix.problem_request_sha256,
        name="prefix.problem_request_sha256",
    )
    _nonempty(prefix.route_profile, name="prefix.route_profile")
    _sha256(
        prefix.route_contract_sha256,
        name="prefix.route_contract_sha256",
    )
    _validate_projected_work(
        prefix.algorithmic_work,
        name=f"prefix round {round_index} algorithmic_work",
    )


def _work_from_receipt(value: Any, *, name: str) -> PaperIAlgorithmicWork:
    components = _required_attribute(value, "components", context=name)
    projected = PaperIWorkComponents(
        n_h_outer=_nonnegative_int(
            _required_attribute(components, "n_h_outer", context=f"{name}.components"),
            name=f"{name}.components.n_h_outer",
        ),
        n_h_refit=_nonnegative_int(
            _required_attribute(components, "n_h_refit", context=f"{name}.components"),
            name=f"{name}.components.n_h_refit",
        ),
        n_grad=_nonnegative_int(
            _required_attribute(components, "n_grad", context=f"{name}.components"),
            name=f"{name}.components.n_grad",
        ),
        n_metric=_nonnegative_int(
            _required_attribute(components, "n_metric", context=f"{name}.components"),
            name=f"{name}.components.n_metric",
        ),
    )
    s_alg = _nonnegative_int(
        _required_attribute(value, "s_alg", context=name),
        name=f"{name}.s_alg",
    )
    if projected.s_alg != s_alg:
        raise ValueError(
            f"{name} fails canonical component closure: "
            f"S_alg={s_alg}, components={projected.s_alg}."
        )
    return PaperIAlgorithmicWork(components=projected, s_alg=s_alg)


def _validate_accounting(run_source: Any) -> PaperIAlgorithmicWork:
    accounting = run_source.estimator_accounting
    if not bool(_required_attribute(accounting, "complete", context="estimator_accounting")):
        raise ValueError("canonical estimator accounting must be complete.")
    if not bool(
        _required_attribute(
            accounting,
            "prefix_closure_passed",
            context="estimator_accounting",
        )
    ):
        raise ValueError("canonical estimator prefix closure must pass.")
    blockers = tuple(
        _required_attribute(
            accounting,
            "exact_blockers",
            context="estimator_accounting",
        )
    )
    if blockers:
        raise ValueError(
            f"canonical estimator accounting has blockers: {blockers!r}."
        )
    all_work = _work_from_receipt(
        _required_attribute(accounting, "all_work", context="estimator_accounting"),
        name="estimator_accounting.all_work",
    )
    winning = _work_from_receipt(
        _required_attribute(
            accounting,
            "winning_lineage",
            context="estimator_accounting",
        ),
        name="estimator_accounting.winning_lineage",
    )
    if winning.s_alg > all_work.s_alg:
        raise ValueError("winning-lineage S_alg exceeds canonical all-work S_alg.")
    raw = _required_attribute(
        accounting,
        "raw_occurrences",
        context="estimator_accounting",
    )
    raw_work = _work_from_receipt(
        _WorkProjection(
            components=raw,
            s_alg=_required_attribute(
                accounting,
                "raw_occurrence_total",
                context="estimator_accounting",
            ),
        ),
        name="estimator_accounting.raw_occurrences",
    )
    if raw_work != all_work:
        raise ValueError(
            "canonical all-work receipt disagrees with raw occurrence accounting."
        )
    return all_work


@dataclass(frozen=True)
class _WorkProjection:
    components: Any
    s_alg: Any


def _reference_state(run_source: Any) -> PaperIReferenceState:
    reporting = run_source.canonical_reporting
    source = _required_attribute(
        reporting,
        "reference_state",
        context="canonical_reporting",
    )
    real = tuple(
        _finite(value, name=f"reference_state.amplitudes_real[{index}]")
        for index, value in enumerate(
            _required_attribute(
                source,
                "amplitudes_real",
                context="canonical_reporting.reference_state",
            )
        )
    )
    imaginary = tuple(
        _finite(value, name=f"reference_state.amplitudes_imaginary[{index}]")
        for index, value in enumerate(
            _required_attribute(
                source,
                "amplitudes_imaginary",
                context="canonical_reporting.reference_state",
            )
        )
    )
    qubits = _positive_int(
        _required_attribute(
            source,
            "qubit_count",
            context="canonical_reporting.reference_state",
        ),
        name="canonical_reporting.reference_state.qubit_count",
    )
    if len(real) != 2**qubits or len(imaginary) != len(real):
        raise ValueError(
            "canonical reference-state amplitudes do not match qubit_count."
        )
    norm_squared = math.fsum(
        real_part * real_part + imaginary_part * imaginary_part
        for real_part, imaginary_part in zip(real, imaginary, strict=True)
    )
    if not math.isclose(norm_squared, 1.0, rel_tol=1.0e-10, abs_tol=1.0e-12):
        raise ValueError("canonical reference state must be normalized.")
    problem_qubits = _positive_int(
        _required_attribute(
            run_source.problem,
            "total_qubits",
            context="run_source.problem",
        ),
        name="run_source.problem.total_qubits",
    )
    if qubits != problem_qubits:
        raise ValueError(
            "canonical reference-state qubit count disagrees with the problem."
        )
    return PaperIReferenceState(
        amplitudes_real=real,
        amplitudes_imaginary=imaginary,
        qubit_count=qubits,
        source_label=_nonempty(
            _required_attribute(
                source,
                "source_label",
                context="canonical_reporting.reference_state",
            ),
            name="canonical_reporting.reference_state.source_label",
        ),
        state_fingerprint=_nonempty(
            _required_attribute(
                source,
                "state_fingerprint",
                context="canonical_reporting.reference_state",
            ),
            name="canonical_reporting.reference_state.state_fingerprint",
        ),
    )


def _prefix_work(run_source: Any) -> tuple[PaperIAlgorithmicWork, ...]:
    raw = tuple(
        _required_attribute(
            run_source.canonical_reporting,
            "accepted_prefix_work",
            context="canonical_reporting",
        )
    )
    projected = tuple(
        _work_from_receipt(value, name=f"accepted_prefix_work[{index}]")
        for index, value in enumerate(raw)
    )
    previous = PaperIWorkComponents(0, 0, 0, 0)
    for index, work in enumerate(projected):
        current = work.components
        if (
            current.n_h_outer < previous.n_h_outer
            or current.n_h_refit < previous.n_h_refit
            or current.n_grad < previous.n_grad
            or current.n_metric < previous.n_metric
        ):
            raise ValueError(
                f"accepted_prefix_work[{index}] is not cumulative."
            )
        previous = current
    return projected


def _project_operator(
    block: Any,
    *,
    index: int,
    expected_label: str,
    qubit_count: int,
) -> PaperIPrefixOperator:
    label = _nonempty(
        _required_attribute(block, "candidate_label", context=f"parameter_blocks[{index}]"),
        name=f"parameter_blocks[{index}].candidate_label",
    )
    logical_index = _nonnegative_int(
        _required_attribute(block, "logical_index", context=f"parameter_blocks[{index}]"),
        name=f"parameter_blocks[{index}].logical_index",
    )
    if logical_index != index or label != expected_label:
        raise ValueError(
            "checkpoint parameter-block order disagrees with accepted operator order."
        )
    runtime_start = _nonnegative_int(
        _required_attribute(block, "runtime_start", context=f"parameter_blocks[{index}]"),
        name=f"parameter_blocks[{index}].runtime_start",
    )
    runtime_count = _positive_int(
        _required_attribute(block, "runtime_count", context=f"parameter_blocks[{index}]"),
        name=f"parameter_blocks[{index}].runtime_count",
    )
    terms: list[PaperIPrefixPauliTerm] = []
    for term_index, term in enumerate(
        tuple(
            _required_attribute(
                block,
                "runtime_terms",
                context=f"parameter_blocks[{index}]",
            )
        )
    ):
        word = _nonempty(
            _required_attribute(
                term,
                "pauli_exyz",
                context=f"parameter_blocks[{index}].runtime_terms[{term_index}]",
            ),
            name=f"parameter_blocks[{index}].runtime_terms[{term_index}].pauli_exyz",
        )
        term_qubits = _positive_int(
            _required_attribute(
                term,
                "qubit_count",
                context=f"parameter_blocks[{index}].runtime_terms[{term_index}]",
            ),
            name=f"parameter_blocks[{index}].runtime_terms[{term_index}].qubit_count",
        )
        if (
            term_qubits != qubit_count
            or len(word) != qubit_count
            or any(symbol not in "exyz" for symbol in word)
        ):
            raise ValueError(
                "checkpoint runtime Pauli term violates the canonical exyz "
                "qubit-order contract."
            )
        terms.append(
            PaperIPrefixPauliTerm(
                pauli_exyz=word,
                coefficient_real=_finite(
                    _required_attribute(
                        term,
                        "coefficient_real",
                        context=f"parameter_blocks[{index}].runtime_terms[{term_index}]",
                    ),
                    name=f"parameter_blocks[{index}].runtime_terms[{term_index}].coefficient_real",
                ),
                coefficient_imaginary=_finite(
                    _required_attribute(
                        term,
                        "coefficient_imaginary",
                        context=f"parameter_blocks[{index}].runtime_terms[{term_index}]",
                    ),
                    name=(
                        f"parameter_blocks[{index}].runtime_terms"
                        f"[{term_index}].coefficient_imaginary"
                    ),
                ),
                qubit_count=term_qubits,
            )
        )
    if len(terms) != runtime_count:
        raise ValueError(
            "checkpoint runtime_count disagrees with the serialized term count."
        )
    execution_mode = _nonempty(
        _required_attribute(
            block,
            "execution_mode",
            context=f"parameter_blocks[{index}]",
        ),
        name=f"parameter_blocks[{index}].execution_mode",
    )
    if execution_mode not in {"termwise_product", "grouped_exact"}:
        raise ValueError(
            f"unsupported canonical execution mode {execution_mode!r}."
        )
    return PaperIPrefixOperator(
        candidate_label=label,
        logical_index=logical_index,
        runtime_start=runtime_start,
        runtime_count=runtime_count,
        execution_mode=execution_mode,
        runtime_terms=tuple(terms),
    )


def _reconstruct_sr_prefix(
    run_source: Any,
    zero_index: int,
    *,
    accepted_prefix_work: tuple[PaperIAlgorithmicWork, ...] | None = None,
    reference_state: PaperIReferenceState | None = None,
) -> PaperIPrefixCompileInput:
    """Reconstruct one exact accepted prefix from its scientific checkpoint."""

    run_source = _typed_run_source(run_source)
    trajectory = tuple(run_source.accepted_trajectory)
    replay = tuple(run_source.scientific_replay)
    work = (
        _prefix_work(run_source)
        if accepted_prefix_work is None
        else accepted_prefix_work
    )
    if not (0 <= int(zero_index) < len(trajectory)):
        raise IndexError("accepted prefix index is out of range.")
    if len(trajectory) != len(replay) or len(trajectory) != len(work):
        raise ValueError(
            "accepted trajectory, replay checkpoints, and prefix work must align."
        )
    state = trajectory[zero_index]
    replay_row = replay[zero_index]
    checkpoint = _required_attribute(
        replay_row,
        "checkpoint",
        context=f"scientific_replay[{zero_index}]",
    )
    round_index = _positive_int(
        _required_attribute(
            state,
            "controller_round",
            context=f"accepted_trajectory[{zero_index}]",
        ),
        name=f"accepted_trajectory[{zero_index}].controller_round",
    )
    replay_round = _positive_int(
        _required_attribute(
            replay_row,
            "controller_round",
            context=f"scientific_replay[{zero_index}]",
        ),
        name=f"scientific_replay[{zero_index}].controller_round",
    )
    checkpoint_round = _positive_int(
        _required_attribute(
            checkpoint,
            "outer_iteration",
            context=f"scientific_replay[{zero_index}].checkpoint",
        ),
        name=f"scientific_replay[{zero_index}].checkpoint.outer_iteration",
    )
    if (
        round_index != zero_index + 1
        or replay_round != round_index
        or checkpoint_round != round_index
    ):
        raise ValueError(
            "canonical accepted controller rounds must be complete, contiguous, "
            "and aligned with replay checkpoints."
        )
    replay_state = _required_attribute(
        replay_row,
        "accepted_state",
        context=f"scientific_replay[{zero_index}]",
    )
    if replay_state != state:
        raise ValueError(
            f"scientific_replay[{zero_index}] accepted state disagrees with the trajectory."
        )
    labels = tuple(
        _nonempty(value, name=f"accepted_trajectory[{zero_index}].operators")
        for value in _required_attribute(
            state,
            "operators",
            context=f"accepted_trajectory[{zero_index}]",
        )
    )
    checkpoint_labels = tuple(
        str(value)
        for value in _required_attribute(
            checkpoint,
            "ordered_operator_labels",
            context=f"scientific_replay[{zero_index}].checkpoint",
        )
    )
    depth = _positive_int(
        _required_attribute(
            checkpoint,
            "active_ansatz_depth",
            context=f"scientific_replay[{zero_index}].checkpoint",
        ),
        name=f"scientific_replay[{zero_index}].checkpoint.active_ansatz_depth",
    )
    if labels != checkpoint_labels or len(labels) != depth:
        raise ValueError(
            "accepted operator order/depth disagrees with the replay checkpoint."
        )
    logical = tuple(
        _finite(value, name=f"checkpoint.logical_parameters[{index}]")
        for index, value in enumerate(
            _required_attribute(
                checkpoint,
                "logical_parameters",
                context=f"scientific_replay[{zero_index}].checkpoint",
            )
        )
    )
    runtime = tuple(
        _finite(value, name=f"checkpoint.runtime_parameters[{index}]")
        for index, value in enumerate(
            _required_attribute(
                checkpoint,
                "runtime_parameters",
                context=f"scientific_replay[{zero_index}].checkpoint",
            )
        )
    )
    if (
        logical
        != tuple(
            _required_attribute(
                state,
                "logical_parameters",
                context=f"accepted_trajectory[{zero_index}]",
            )
        )
        or runtime
        != tuple(
            _required_attribute(
                state,
                "runtime_parameters",
                context=f"accepted_trajectory[{zero_index}]",
            )
        )
    ):
        raise ValueError(
            "accepted parameters disagree with the replay checkpoint."
        )
    reference = (
        _reference_state(run_source)
        if reference_state is None
        else reference_state
    )
    raw_blocks = tuple(
        _required_attribute(
            checkpoint,
            "parameter_blocks",
            context=f"scientific_replay[{zero_index}].checkpoint",
        )
    )
    if len(raw_blocks) != depth or len(logical) != depth:
        raise ValueError(
            "checkpoint parameter blocks/logical parameters disagree with depth."
        )
    if (
        str(
            _required_attribute(
                checkpoint,
                "parameterization_mode",
                context=f"scientific_replay[{zero_index}].checkpoint",
            )
        )
        != "per_pauli_term_v1"
        or str(
            _required_attribute(
                checkpoint,
                "parameterization_term_order",
                context=f"scientific_replay[{zero_index}].checkpoint",
            )
        )
        != "sorted"
    ):
        raise ValueError(
            "canonical checkpoint parameterization must be the sorted "
            "per-Pauli-term layout."
        )
    if (
        str(
            _required_attribute(
                checkpoint,
                "estimator_ledger_status",
                context=f"scientific_replay[{zero_index}].checkpoint",
            )
        )
        != "complete"
    ):
        raise ValueError("canonical checkpoint estimator ledger must be complete.")
    operators = tuple(
        _project_operator(
            block,
            index=index,
            expected_label=labels[index],
            qubit_count=reference.qubit_count,
        )
        for index, block in enumerate(raw_blocks)
    )
    expected_runtime_start = 0
    for operator in operators:
        if operator.runtime_start != expected_runtime_start:
            raise ValueError(
                "checkpoint parameter blocks must be an exact contiguous "
                "runtime partition."
            )
        expected_runtime_start += operator.runtime_count
    if expected_runtime_start != len(runtime):
        raise ValueError(
            "checkpoint runtime parameter count disagrees with parameter blocks."
        )
    if not bool(
        _required_attribute(
            checkpoint,
            "strict_replay_passed",
            context=f"scientific_replay[{zero_index}].checkpoint",
        )
    ):
        raise ValueError("canonical prefix strict replay did not pass.")
    if not math.isclose(
        _finite(
            _required_attribute(
                checkpoint,
                "strict_replay_fidelity",
                context=f"scientific_replay[{zero_index}].checkpoint",
            ),
            name="checkpoint.strict_replay_fidelity",
        ),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ):
        raise ValueError("canonical prefix strict replay fidelity is incomplete.")
    route_profile = _nonempty(
        _required_attribute(
            run_source.route,
            "profile",
            context="run_source.route",
        ),
        name="run_source.route.profile",
    )
    route_sha = _sha256(
        _required_attribute(
            run_source.route,
            "contract_sha256",
            context="run_source.route",
        ),
        name="run_source.route.contract_sha256",
    )
    if (
        str(
            _required_attribute(
                checkpoint,
                "route_profile",
                context=f"scientific_replay[{zero_index}].checkpoint",
            )
        )
        != route_profile
        or str(
            _required_attribute(
                checkpoint,
                "route_contract_sha256",
                context=f"scientific_replay[{zero_index}].checkpoint",
            )
        )
        != route_sha
    ):
        raise ValueError("checkpoint route identity disagrees with the run.")
    checkpoint_s_alg = _nonnegative_int(
        _required_attribute(
            checkpoint,
            "estimator_ledger_s_alg",
            context=f"scientific_replay[{zero_index}].checkpoint",
        ),
        name="checkpoint.estimator_ledger_s_alg",
    )
    beam_policy = _nonempty(
        _required_attribute(
            run_source.route,
            "beam_policy",
            context="run_source.route",
        ),
        name="run_source.route.beam_policy",
    )
    if (
        beam_policy == "off"
        and checkpoint_s_alg != work[zero_index].s_alg
    ) or (
        beam_policy == "fork_local"
        and checkpoint_s_alg > work[zero_index].s_alg
    ):
        raise ValueError(
            "checkpoint ledger S_alg disagrees with canonical prefix work."
        )
    fingerprint = _nonempty(
        _required_attribute(
            state,
            "projective_state_fingerprint",
            context=f"accepted_trajectory[{zero_index}]",
        ),
        name=f"accepted_trajectory[{zero_index}].projective_state_fingerprint",
    )
    if (
        str(
            _required_attribute(
                checkpoint,
                "projective_state_fingerprint",
                context=f"scientific_replay[{zero_index}].checkpoint",
            )
        )
        != fingerprint
    ):
        raise ValueError(
            "checkpoint projective fingerprint disagrees with accepted state."
        )
    return PaperIPrefixCompileInput(
        source_method="sr_snake",
        controller_round=round_index,
        active_ansatz_depth=depth,
        ordered_operator_labels=labels,
        operators=operators,
        logical_parameters=logical,
        runtime_parameters=runtime,
        reference_state=reference,
        checkpoint_sha256=_sha256(
            _required_attribute(
                checkpoint,
                "checkpoint_sha256",
                context=f"scientific_replay[{zero_index}].checkpoint",
            ),
            name="checkpoint.checkpoint_sha256",
        ),
        projective_state_fingerprint=fingerprint,
        problem_request_sha256=_sha256(
            _required_attribute(
                run_source.problem,
                "problem_request_sha256",
                context="run_source.problem",
            ),
            name="run_source.problem.problem_request_sha256",
        ),
        route_profile=route_profile,
        route_contract_sha256=route_sha,
        algorithmic_work=work[zero_index],
    )


def _accepted_trace(
    run_source: Any,
    prefixes: Sequence[PaperIPrefixCompileInput],
) -> tuple[PaperIAcceptedError, ...]:
    exact = _finite(
        _required_attribute(
            run_source.canonical_reporting,
            "exact_same_cutoff_energy",
            context="canonical_reporting",
        ),
        name="canonical_reporting.exact_same_cutoff_energy",
    )
    trajectory = tuple(run_source.accepted_trajectory)
    if not trajectory:
        raise ValueError(
            "canonical summary requires at least one complete accepted state."
        )
    rows: list[PaperIAcceptedError] = []
    for zero_index, state in enumerate(trajectory):
        prefix = prefixes[zero_index]
        energy = _finite(
            _required_attribute(
                state,
                "energy",
                context=f"accepted_trajectory[{zero_index}]",
            ),
            name=f"accepted_trajectory[{zero_index}].energy",
        )
        rows.append(
            PaperIAcceptedError(
                controller_round=prefix.controller_round,
                active_ansatz_depth=prefix.active_ansatz_depth,
                accepted_energy=energy,
                exact_same_cutoff_energy=exact,
                absolute_energy_error=abs(energy - exact),
                projective_state_fingerprint=(
                    prefix.projective_state_fingerprint
                ),
                checkpoint_sha256=prefix.checkpoint_sha256,
            )
        )
    stop_rounds = _nonnegative_int(
        _required_attribute(
            run_source.stop,
            "completed_controller_rounds",
            context="run_source.stop",
        ),
        name="run_source.stop.completed_controller_rounds",
    )
    if stop_rounds != len(rows):
        raise ValueError(
            "stop receipt disagrees with the complete accepted trajectory."
        )
    return tuple(rows)


def _validated_error_trace(
    trace: Sequence[PaperIErrorTracePoint],
    *,
    name: str,
) -> tuple[PaperIErrorTracePoint, ...]:
    rows = tuple(trace)
    if not rows:
        raise ValueError(f"{name} requires a nonempty error trace.")
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, PaperIErrorTracePoint):
            raise TypeError(f"{name} requires typed PaperIErrorTracePoint rows.")
        if int(row.controller_round) != index:
            raise ValueError(
                f"{name} controller rounds must be complete and ordered 1..N."
            )
    return rows


def select_paper_i_effective_plateau(
    trace: Sequence[PaperIErrorTracePoint],
    *,
    relative_tolerance: float = 0.10,
) -> PaperIEffectivePlateauSelection:
    """Apply ``paper_i_effective_plateau_v1`` to one complete error trace."""

    rows = _validated_error_trace(trace, name="effective plateau selection")
    tolerance = _finite(
        relative_tolerance,
        name="effective plateau relative_tolerance",
    )
    if tolerance < 0.0:
        raise ValueError(
            "effective plateau relative_tolerance must be nonnegative."
        )
    best = min(row.absolute_energy_error for row in rows)
    threshold = (1.0 + tolerance) * best
    selected_index = next(
        index
        for index, row in enumerate(rows)
        if row.absolute_energy_error <= threshold
    )
    selected = rows[selected_index]
    return PaperIEffectivePlateauSelection(
        policy=EFFECTIVE_PLATEAU_POLICY,
        selected_trace_index=selected_index,
        controller_round=selected.controller_round,
        absolute_energy_error=selected.absolute_energy_error,
        best_observed_error=best,
        selection_threshold=threshold,
        horizon_controller_rounds=len(rows),
    )


def select_paper_i_common_accuracy(
    sr_snake_trace: Sequence[PaperIErrorTracePoint],
    append_adapt_trace: Sequence[PaperIErrorTracePoint],
) -> PaperICommonAccuracySelection:
    """Select first crossings under the canonical shared-plateau window."""

    snake = _validated_error_trace(
        sr_snake_trace,
        name="SR-SNAKE common-accuracy selection",
    )
    append = _validated_error_trace(
        append_adapt_trace,
        name="append-ADAPT common-accuracy selection",
    )
    snake_plateau = select_paper_i_effective_plateau(snake)
    append_plateau = select_paper_i_effective_plateau(append)
    shared_end = min(
        snake_plateau.controller_round,
        append_plateau.controller_round,
    )
    snake_window = tuple(
        row for row in snake if row.controller_round <= shared_end
    )
    append_window = tuple(
        row for row in append if row.controller_round <= shared_end
    )
    snake_minimum = min(
        row.absolute_energy_error for row in snake_window
    )
    append_minimum = min(
        row.absolute_energy_error for row in append_window
    )
    target = max(snake_minimum, append_minimum)
    snake_crossing_index, snake_crossing = next(
        (index, row)
        for index, row in enumerate(snake)
        if row.controller_round <= shared_end
        and row.absolute_energy_error <= target
    )
    append_crossing_index, append_crossing = next(
        (index, row)
        for index, row in enumerate(append)
        if row.controller_round <= shared_end
        and row.absolute_energy_error <= target
    )
    return PaperICommonAccuracySelection(
        shared_window_end_controller_round=shared_end,
        common_target_absolute_error=target,
        sr_snake_window_minimum_error=snake_minimum,
        append_adapt_window_minimum_error=append_minimum,
        sr_snake_crossing_trace_index=snake_crossing_index,
        sr_snake_crossing_controller_round=snake_crossing.controller_round,
        append_adapt_crossing_trace_index=append_crossing_index,
        append_adapt_crossing_controller_round=(
            append_crossing.controller_round
        ),
        sr_snake_plateau_controller_round=snake_plateau.controller_round,
        append_adapt_plateau_controller_round=(
            append_plateau.controller_round
        ),
    )


def _error_trace_points(
    trace: Sequence[PaperIAcceptedError],
) -> tuple[PaperIErrorTracePoint, ...]:
    return tuple(
        PaperIErrorTracePoint(
            controller_round=row.controller_round,
            absolute_energy_error=row.absolute_energy_error,
        )
        for row in trace
    )


def compile_paper_i_prefix_qiskit_payload(
    prefix: PaperIPrefixCompileInput,
) -> dict[str, Any]:
    """Compile one typed Paper-I prefix under the locked Table-I convention.

    Heavy and optional imports are intentionally local to this observation
    seam.  Method-specific adapters must authenticate and construct
    :class:`PaperIPrefixCompileInput` before crossing this shared boundary.
    """

    import numpy as np

    from pipelines.exact_bench.table_i_qiskit_resource_compile import (
        TABLE_I_COMPILED_BASIS_GATES,
        TABLE_I_QISKIT_COMPILE_CONVENTION,
        TableIQiskitCompileConfig,
        compile_table_i_ansatz_terms,
    )
    from src.quantum.pauli_polynomial_class import PauliPolynomial
    from src.quantum.qubitization_module import PauliTerm
    from src.quantum.vqe_latex_python_pairs import AnsatzTerm

    if TABLE_I_QISKIT_COMPILE_CONVENTION != LOCKED_QISKIT_COMPILE_CONVENTION:
        raise RuntimeError(
            "The installed Paper-I Qiskit compiler convention drifted."
        )
    operators = []
    for operator in prefix.operators:
        polynomial = PauliPolynomial("JW")
        for term in operator.runtime_terms:
            polynomial.add_term(
                PauliTerm(
                    term.qubit_count,
                    ps=term.pauli_exyz,
                    pc=complex(
                        term.coefficient_real,
                        term.coefficient_imaginary,
                    ),
                )
            )
        operators.append(
            AnsatzTerm(
                label=operator.candidate_label,
                polynomial=polynomial,
                execution_mode=operator.execution_mode,
            )
        )
    reference = np.asarray(
        prefix.reference_state.amplitudes_real,
        dtype=float,
    ).astype(complex)
    reference += 1.0j * np.asarray(
        prefix.reference_state.amplitudes_imaginary,
        dtype=float,
    )
    config = TableIQiskitCompileConfig(
        basis_gates=TABLE_I_COMPILED_BASIS_GATES,
        optimization_level=0,
        seed_transpiler=7,
        structure_theta_value=1.0,
        include_reference_state=True,
        compile_convention=LOCKED_QISKIT_COMPILE_CONVENTION,
        coefficient_tolerance=1.0e-12,
        grouped_exact_max_active_qubits=5,
    )
    payload = compile_table_i_ansatz_terms(
        ops=tuple(operators),
        num_qubits=prefix.reference_state.qubit_count,
        reference_state=reference,
        source_kind="canonical_paper_i_accepted_prefix",
        config=config,
    )
    if (
        tuple(payload["compiled_basis_gates"]) != TABLE_I_COMPILED_BASIS_GATES
        or int(payload["qiskit_transpile_optimization_level"]) != 0
        or int(payload["qiskit_transpile_seed"]) != 7
        or payload["compiled_circuit_scope"]
        != "ansatz_circuit_including_reference_state"
    ):
        raise RuntimeError("The locked Paper-I Qiskit compiler settings drifted.")
    return dict(payload)


def _locked_prefix_compiler(
    prefix: PaperIPrefixCompileInput,
) -> PaperIQiskitResources:
    """Project the locked compile payload into the stable typed summary row."""

    payload = compile_paper_i_prefix_qiskit_payload(prefix)
    return PaperIQiskitResources(
        compile_convention=str(payload["compile_convention"]),
        compiled_two_qubit_count=_nonnegative_int(
            payload["compiled_count_2q_total"],
            name="compiled_count_2q_total",
        ),
        compiled_two_qubit_depth=_nonnegative_int(
            payload["compiled_depth_2q_total"],
            name="compiled_depth_2q_total",
        ),
        compiled_total_depth=_nonnegative_int(
            payload["compiled_depth_total"],
            name="compiled_depth_total",
        ),
    )


_PREFIX_COMPILER: _PrefixCompiler = _locked_prefix_compiler


def _existing_sr_compile_sidecars(
    run_source: Any,
    prefixes: Sequence[PaperIPrefixCompileInput],
) -> dict[
    tuple[str, str, str, str, int],
    PaperIQiskitResources | PaperIObservationFailure,
]:
    """Index matching resources from an already attached typed summary."""

    previous = _required_attribute(
        run_source,
        "paper_i_summary",
        context="run_source",
    )
    if previous is None:
        return {}
    if not isinstance(previous, PaperIRunSummary):
        raise TypeError(
            "run_source.paper_i_summary must be a typed PaperIRunSummary."
        )
    current = {prefix.compile_cache_key: prefix for prefix in prefixes}
    observations: list[PaperIPrefixObservation] = list(
        previous.requested_rounds
    )
    plateau = previous.effective_plateau
    observations.append(
        PaperIPrefixObservation(
            purpose="existing_effective_plateau_sidecar",
            status=plateau.status,
            controller_round=plateau.controller_round,
            active_ansatz_depth=plateau.active_ansatz_depth,
            absolute_energy_error=plateau.absolute_energy_error,
            algorithmic_work=plateau.algorithmic_work,
            prefix=plateau.prefix,
            resources=plateau.resources,
            failure=plateau.failure,
        )
    )
    if previous.append_matched.sr_snake is not None:
        observations.append(previous.append_matched.sr_snake)
    cache: dict[
        tuple[str, str, str, str, int],
        PaperIQiskitResources | PaperIObservationFailure,
    ] = {}
    for observation in observations:
        if not isinstance(observation, PaperIPrefixObservation):
            raise TypeError(
                "existing Paper-I summary contains an untyped prefix "
                "observation."
            )
        prefix = observation.prefix
        key = prefix.compile_cache_key
        if prefix.source_method != "sr_snake" or key not in current:
            continue
        if prefix != current[key]:
            raise ValueError(
                "existing Paper-I resource sidecar key collides with a "
                "different canonical accepted prefix."
            )
        resources = observation.resources
        if resources is None:
            # Tooling failures remain retryable; do not cache them across
            # summary calls.
            continue
        if not isinstance(resources, PaperIQiskitResources):
            raise TypeError(
                "existing Paper-I summary contains an untyped resource "
                "sidecar."
            )
        if resources.compile_convention != LOCKED_QISKIT_COMPILE_CONVENTION:
            raise ValueError(
                "existing Paper-I resource sidecar uses the wrong compile "
                "convention."
            )
        existing = cache.get(key)
        if existing is not None and existing != resources:
            raise ValueError(
                "existing Paper-I summary contains conflicting resource "
                "sidecars for one accepted prefix."
            )
        cache[key] = resources
    return cache


def _compile_once(
    prefix: PaperIPrefixCompileInput,
    cache: dict[
        tuple[str, str, str, str, int],
        PaperIQiskitResources | PaperIObservationFailure,
    ],
) -> PaperIQiskitResources | PaperIObservationFailure:
    key = prefix.compile_cache_key
    if key in cache:
        return cache[key]
    try:
        resources = _PREFIX_COMPILER(prefix)
        if not isinstance(resources, PaperIQiskitResources):
            raise TypeError(
                "Paper-I prefix compiler returned an untyped resource payload."
            )
        if resources.compile_convention != LOCKED_QISKIT_COMPILE_CONVENTION:
            raise ValueError(
                "Paper-I prefix compiler returned the wrong compile convention."
            )
        for name in (
            "compiled_two_qubit_count",
            "compiled_two_qubit_depth",
            "compiled_total_depth",
        ):
            _nonnegative_int(getattr(resources, name), name=name)
        observed: PaperIQiskitResources | PaperIObservationFailure = resources
    except Exception as exc:
        observed = PaperIObservationFailure(
            exception_type=type(exc).__name__,
            message=str(exc),
            retryable=True,
        )
    cache[key] = observed
    return observed


def _prefix_observation(
    *,
    purpose: str,
    prefix: PaperIPrefixCompileInput,
    absolute_energy_error: float,
    cache: dict[
        tuple[str, str, str, str, int],
        PaperIQiskitResources | PaperIObservationFailure,
    ],
) -> PaperIPrefixObservation:
    compiled = _compile_once(prefix, cache)
    failure = (
        compiled if isinstance(compiled, PaperIObservationFailure) else None
    )
    resources = (
        compiled if isinstance(compiled, PaperIQiskitResources) else None
    )
    return PaperIPrefixObservation(
        purpose=purpose,
        status=(
            "retryable_tooling_error" if failure is not None else "available"
        ),
        controller_round=prefix.controller_round,
        active_ansatz_depth=prefix.active_ansatz_depth,
        absolute_energy_error=absolute_energy_error,
        algorithmic_work=prefix.algorithmic_work,
        prefix=prefix,
        resources=resources,
        failure=failure,
    )


def _comparison_contract(run_source: Any) -> PaperIComparisonContract:
    execution = _required_attribute(
        run_source.route,
        "execution",
        context="run_source.route",
    )
    return PaperIComparisonContract(
        problem_request_sha256=_sha256(
            _required_attribute(
                run_source.problem,
                "problem_request_sha256",
                context="run_source.problem",
            ),
            name="run_source.problem.problem_request_sha256",
        ),
        optimizer=_nonempty(
            _required_attribute(
                execution,
                "optimizer",
                context="run_source.route.execution",
            ),
            name="run_source.route.execution.optimizer",
        ),
        optimizer_maxiter=_positive_int(
            _required_attribute(
                execution,
                "optimizer_maxiter",
                context="run_source.route.execution",
            ),
            name="run_source.route.execution.optimizer_maxiter",
        ),
        seed=_nonnegative_int(
            _required_attribute(
                execution,
                "seed",
                context="run_source.route.execution",
            ),
            name="run_source.route.execution.seed",
        ),
        candidate_representation=_nonempty(
            _required_attribute(
                run_source.canonical_reporting,
                "candidate_representation",
                context="canonical_reporting",
            ),
            name="canonical_reporting.candidate_representation",
        ),
    )


def _resolve_append_source(
    append_reference: (
        CanonicalAppendReference
        | PaperIAppendRunSource
        | PaperIAppendReferenceResolver
    ),
    request: PaperIAppendResolutionRequest,
) -> tuple[PaperIAppendRunSource | None, PaperIAppendMatchedObservation | None]:
    if isinstance(append_reference, CanonicalAppendReference):
        append_reference = _CANONICAL_APPEND_REGISTRY_RESOLVER
    if isinstance(append_reference, PaperIAppendRunSource):
        return append_reference, None
    if isinstance(append_reference, PaperIAppendReferenceResolver):
        try:
            source = append_reference.resolve_canonical_append(request)
        except Exception as exc:
            failure = PaperIObservationFailure(
                exception_type=type(exc).__name__,
                message=str(exc),
                retryable=True,
            )
            return None, PaperIAppendMatchedObservation(
                status="retryable_resolution_error",
                reason="canonical_append_resolver_failed",
                shared_window_end_controller_round=None,
                common_target_absolute_error=None,
                sr_snake=None,
                append_adapt=None,
                failure=failure,
            )
        if source is None:
            return None, PaperIAppendMatchedObservation(
                status="unavailable",
                reason="canonical_append_reference_not_found",
                shared_window_end_controller_round=None,
                common_target_absolute_error=None,
                sr_snake=None,
                append_adapt=None,
            )
        if not isinstance(source, PaperIAppendRunSource):
            return None, PaperIAppendMatchedObservation(
                status="retryable_resolution_error",
                reason="canonical_append_resolver_returned_untyped_source",
                shared_window_end_controller_round=None,
                common_target_absolute_error=None,
                sr_snake=None,
                append_adapt=None,
                failure=PaperIObservationFailure(
                    exception_type="TypeError",
                    message=(
                        "append resolver must return PaperIAppendRunSource or None"
                    ),
                    retryable=True,
                ),
            )
        return source, None
    raise TypeError(
        "append_reference must be a typed canonical append marker, a typed "
        "PaperIAppendRunSource, or a PaperIAppendReferenceResolver."
    )


def _validate_append_source(
    source: PaperIAppendRunSource,
    request: PaperIAppendResolutionRequest,
) -> str | None:
    if source.comparison_contract != request.comparison_contract:
        return "append comparison contract does not match the SR-SNAKE run"
    try:
        horizon = _horizon_scope(source.horizon_scope)
        trace = tuple(source.accepted_error_trace)
        prefixes = tuple(source.accepted_prefixes)
        if not trace or len(trace) != len(prefixes):
            return "append accepted trace and prefixes must be complete and aligned"
        previous_components = PaperIWorkComponents(0, 0, 0, 0)
        for index, (row, prefix) in enumerate(
            zip(trace, prefixes, strict=True),
            start=1,
        ):
            if not isinstance(row, PaperIAcceptedError) or not isinstance(
                prefix,
                PaperIPrefixCompileInput,
            ):
                return "append source contains an untyped row or prefix"
            _validate_prefix_compile_input(prefix)
            if (
                row.controller_round != index
                or prefix.controller_round != index
                or prefix.source_method != "append_adapt"
                or row.active_ansatz_depth != prefix.active_ansatz_depth
            ):
                return "append accepted history is not complete and aligned"
            if (
                prefix.problem_request_sha256
                != request.comparison_contract.problem_request_sha256
            ):
                return "append prefix problem identity disagrees with its contract"
            if (
                prefix.reference_state.qubit_count
                != request.reference_state.qubit_count
                or prefix.reference_state.state_fingerprint
                != request.reference_state.state_fingerprint
            ):
                return (
                    "append prefix reference state disagrees with the "
                    "SR-SNAKE run"
                )
            if (
                row.projective_state_fingerprint
                != prefix.projective_state_fingerprint
                or row.checkpoint_sha256 != prefix.checkpoint_sha256
            ):
                return "append error row is not bound to its typed prefix"
            if not math.isclose(
                row.exact_same_cutoff_energy,
                request.exact_same_cutoff_energy,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                return "append source uses a different same-cutoff exact energy"
            if not math.isclose(
                row.absolute_energy_error,
                abs(row.accepted_energy - row.exact_same_cutoff_energy),
                rel_tol=1.0e-12,
                abs_tol=1.0e-14,
            ):
                return "append source error trace is internally inconsistent"
            components = prefix.algorithmic_work.components
            if (
                components.n_h_outer < previous_components.n_h_outer
                or components.n_h_refit < previous_components.n_h_refit
                or components.n_grad < previous_components.n_grad
                or components.n_metric < previous_components.n_metric
            ):
                return "append prefix work is not cumulative"
            previous_components = components
        del horizon
    except (TypeError, ValueError) as exc:
        return str(exc)
    return None


def _append_matched_observation(
    *,
    source: PaperIAppendRunSource,
    request: PaperIAppendResolutionRequest,
    snake_trace: tuple[PaperIAcceptedError, ...],
    snake_prefixes: tuple[PaperIPrefixCompileInput, ...],
    cache: dict[
        tuple[str, str, str, str, int],
        PaperIQiskitResources | PaperIObservationFailure,
    ],
) -> PaperIAppendMatchedObservation:
    incompatibility = _validate_append_source(source, request)
    if incompatibility is not None:
        return PaperIAppendMatchedObservation(
            status="incompatible",
            reason=incompatibility,
            shared_window_end_controller_round=None,
            common_target_absolute_error=None,
            sr_snake=None,
            append_adapt=None,
        )
    append_trace = tuple(source.accepted_error_trace)
    selection = select_paper_i_common_accuracy(
        _error_trace_points(snake_trace),
        _error_trace_points(append_trace),
    )
    snake_crossing = snake_trace[
        selection.sr_snake_crossing_trace_index
    ]
    append_crossing = append_trace[
        selection.append_adapt_crossing_trace_index
    ]
    snake_observation = _prefix_observation(
        purpose="append_matched_common_accuracy_sr_snake",
        prefix=snake_prefixes[
            selection.sr_snake_crossing_trace_index
        ],
        absolute_energy_error=snake_crossing.absolute_energy_error,
        cache=cache,
    )
    append_observation = _prefix_observation(
        purpose="append_matched_common_accuracy_append_adapt",
        prefix=source.accepted_prefixes[
            selection.append_adapt_crossing_trace_index
        ],
        absolute_energy_error=append_crossing.absolute_energy_error,
        cache=cache,
    )
    failures = tuple(
        observation.failure
        for observation in (snake_observation, append_observation)
        if observation.failure is not None
    )
    return PaperIAppendMatchedObservation(
        status=(
            "retryable_tooling_error" if failures else "available"
        ),
        reason=(
            "one or more common-accuracy prefix compilations failed"
            if failures
            else None
        ),
        shared_window_end_controller_round=(
            selection.shared_window_end_controller_round
        ),
        common_target_absolute_error=(
            selection.common_target_absolute_error
        ),
        sr_snake=snake_observation,
        append_adapt=append_observation,
        failure=(failures[0] if failures else None),
    )


def _requested_rounds(
    values: Sequence[int],
    *,
    available_rounds: int,
) -> tuple[int, ...]:
    resolved: list[int] = []
    seen: set[int] = set()
    for value in values:
        round_index = _positive_int(
            value,
            name="requested controller round",
        )
        if round_index > available_rounds:
            raise ValueError(
                f"requested controller round {round_index} is outside the "
                f"complete accepted history 1..{available_rounds}."
            )
        if round_index not in seen:
            resolved.append(round_index)
            seen.add(round_index)
    return tuple(resolved)


def _provenance(
    run_source: Any,
    *,
    exact_energy: float,
    reference: PaperIReferenceState,
    comparison: PaperIComparisonContract,
) -> PaperIRunProvenance:
    problem = run_source.problem
    route = run_source.route
    return PaperIRunProvenance(
        problem_key=_nonempty(
            _required_attribute(problem, "problem_key", context="run_source.problem"),
            name="run_source.problem.problem_key",
        ),
        problem_request_sha256=comparison.problem_request_sha256,
        problem_family=_nonempty(
            _required_attribute(problem, "family_key", context="run_source.problem"),
            name="run_source.problem.family_key",
        ),
        exact_target_label=_nonempty(
            _required_attribute(
                problem,
                "exact_target_label",
                context="run_source.problem",
            ),
            name="run_source.problem.exact_target_label",
        ),
        exact_same_cutoff_energy=exact_energy,
        reference_label=_nonempty(
            _required_attribute(
                problem,
                "reference_label",
                context="run_source.problem",
            ),
            name="run_source.problem.reference_label",
        ),
        reference_source_label=reference.source_label,
        reference_state_fingerprint=reference.state_fingerprint,
        route_family=_nonempty(
            _required_attribute(route, "family", context="run_source.route"),
            name="run_source.route.family",
        ),
        route_profile_request=_nonempty(
            _required_attribute(
                route,
                "profile_request",
                context="run_source.route",
            ),
            name="run_source.route.profile_request",
        ),
        route_profile=_nonempty(
            _required_attribute(route, "profile", context="run_source.route"),
            name="run_source.route.profile",
        ),
        route_contract_sha256=_sha256(
            _required_attribute(
                route,
                "contract_sha256",
                context="run_source.route",
            ),
            name="run_source.route.contract_sha256",
        ),
        candidate_representation=comparison.candidate_representation,
        optimizer=comparison.optimizer,
        optimizer_maxiter=comparison.optimizer_maxiter,
        seed=comparison.seed,
        qiskit_compile_convention=comparison.compile_convention,
    )


def summarize_paper_i_run(
    run_source: SRRunResult,
    *,
    append_reference: (
        CanonicalAppendReference
        | PaperIAppendRunSource
        | PaperIAppendReferenceResolver
    ) = CANONICAL_APPEND_REFERENCE,
    requested_controller_rounds: Sequence[int] = (),
) -> PaperIRunSummary:
    """Summarize one canonical accepted Paper-I run.

    The input must expose the exact canonical receipt fields.  Mappings and
    historical payloads are rejected; there is no field search or fallback.
    """

    run_source = _typed_run_source(run_source)
    _validate_canonical_identity(run_source)
    all_work = _validate_accounting(run_source)
    horizon = _horizon_scope(
        _required_attribute(
            run_source.canonical_reporting,
            "horizon_scope",
            context="canonical_reporting",
        )
    )
    trajectory = tuple(run_source.accepted_trajectory)
    accepted_prefix_work = _prefix_work(run_source)
    reference = _reference_state(run_source)
    prefixes = tuple(
        _reconstruct_sr_prefix(
            run_source,
            index,
            accepted_prefix_work=accepted_prefix_work,
            reference_state=reference,
        )
        for index in range(len(trajectory))
    )
    trace = _accepted_trace(run_source, prefixes)
    if prefixes[-1].algorithmic_work.s_alg > all_work.s_alg:
        raise ValueError(
            "terminal accepted-prefix work exceeds canonical all-work S_alg."
        )
    terminal_components = prefixes[-1].algorithmic_work.components
    all_components = all_work.components
    if (
        terminal_components.n_h_outer > all_components.n_h_outer
        or terminal_components.n_h_refit > all_components.n_h_refit
        or terminal_components.n_grad > all_components.n_grad
        or terminal_components.n_metric > all_components.n_metric
    ):
        raise ValueError(
            "terminal accepted-prefix components exceed canonical all-work "
            "components."
        )
    requested = _requested_rounds(
        tuple(requested_controller_rounds),
        available_rounds=len(trace),
    )
    cache = _existing_sr_compile_sidecars(run_source, prefixes)

    plateau_selection = select_paper_i_effective_plateau(
        _error_trace_points(trace)
    )
    plateau_row = trace[plateau_selection.selected_trace_index]
    plateau_prefix = prefixes[plateau_selection.selected_trace_index]
    plateau_compiled = _compile_once(plateau_prefix, cache)
    plateau_failure = (
        plateau_compiled
        if isinstance(plateau_compiled, PaperIObservationFailure)
        else None
    )
    plateau_resources = (
        plateau_compiled
        if isinstance(plateau_compiled, PaperIQiskitResources)
        else None
    )
    plateau = PaperIEffectivePlateauObservation(
        policy=plateau_selection.policy,
        status=(
            "retryable_tooling_error"
            if plateau_failure is not None
            else "available"
        ),
        controller_round=plateau_row.controller_round,
        active_ansatz_depth=plateau_row.active_ansatz_depth,
        absolute_energy_error=plateau_row.absolute_energy_error,
        best_observed_error=plateau_selection.best_observed_error,
        selection_threshold=plateau_selection.selection_threshold,
        available_horizon_controller_rounds=(
            plateau_selection.horizon_controller_rounds
        ),
        horizon_scope=horizon,
        algorithmic_work=plateau_prefix.algorithmic_work,
        prefix=plateau_prefix,
        resources=plateau_resources,
        failure=plateau_failure,
    )

    requested_observations = tuple(
        _prefix_observation(
            purpose="requested_controller_round",
            prefix=prefixes[round_index - 1],
            absolute_energy_error=trace[
                round_index - 1
            ].absolute_energy_error,
            cache=cache,
        )
        for round_index in requested
    )

    comparison = _comparison_contract(run_source)
    exact_energy = trace[0].exact_same_cutoff_energy
    append_request = PaperIAppendResolutionRequest(
        comparison_contract=comparison,
        exact_same_cutoff_energy=exact_energy,
        reference_state=reference,
    )
    append_source, early_append = _resolve_append_source(
        append_reference,
        append_request,
    )
    append_matched = (
        early_append
        if early_append is not None
        else _append_matched_observation(
            source=append_source,
            request=append_request,
            snake_trace=trace,
            snake_prefixes=prefixes,
            cache=cache,
        )
    )
    if append_matched is None:  # pragma: no cover - static narrowing guard
        raise RuntimeError("append summary resolution produced no observation.")
    return PaperIRunSummary(
        accepted_error_trace=trace,
        effective_plateau=plateau,
        append_matched=append_matched,
        requested_rounds=requested_observations,
        canonical_all_work=all_work,
        horizon_scope=horizon,
        available_controller_rounds=len(trace),
        provenance=_provenance(
            run_source,
            exact_energy=exact_energy,
            reference=reference,
            comparison=comparison,
        ),
    )


__all__ = [
    "CANONICAL_APPEND_REFERENCE",
    "CanonicalAppendReference",
    "canonical_paper_i_algorithmic_work",
    "compile_paper_i_prefix_qiskit_payload",
    "EFFECTIVE_PLATEAU_POLICY",
    "LOCKED_QISKIT_COMPILE_CONVENTION",
    "PaperIAcceptedError",
    "PaperIAlgorithmicWork",
    "PaperIAppendMatchedObservation",
    "PaperIAppendReferenceResolver",
    "PaperIAppendResolutionRequest",
    "PaperIAppendRunSource",
    "PaperICommonAccuracySelection",
    "PaperIComparisonContract",
    "PaperIEffectivePlateauObservation",
    "PaperIEffectivePlateauSelection",
    "PaperIErrorTracePoint",
    "PaperIObservationFailure",
    "PaperIPrefixCompileInput",
    "PaperIPrefixObservation",
    "PaperIPrefixOperator",
    "PaperIPrefixPauliTerm",
    "PaperIQiskitResources",
    "PaperIReferenceState",
    "PaperIRunProvenance",
    "PaperIRunSummary",
    "PaperIWorkComponents",
    "select_paper_i_common_accuracy",
    "select_paper_i_effective_plateau",
    "summarize_paper_i_run",
]
