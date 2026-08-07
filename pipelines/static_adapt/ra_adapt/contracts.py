"""Typed contracts for the canonical Paper-I RA-ADAPT interfaces.

The public request deliberately exposes only representation and the already
characterized SR controller choices.  Study policies are resolved from a
source-locked run bundle and therefore do not appear on :class:`RAAdaptRequest`
or :class:`AppendAdaptRequest`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, ClassVar, Mapping

from pipelines.static_adapt.sr_snake.contracts import (
    AcceptedStateResume,
    AlwaysCommutationReducedInsertion,
    AppendCommutationReducedInsertion,
    AppendOnlyInsertion,
    BeamOff,
    CheckpointObservation,
    CombinatorialBatchAdmission,
    EstimatorLedgerObservation,
    EndpointOverlapDisplacementTrust,
    ExactEDSourceReceipt,
    ExactEDStop,
    ForkLocalBeam,
    FreshStart,
    FullCombinatorialSearchWindow,
    GreedyBatchAdmission,
    MetricPruning,
    PlateauCommutationInsertion,
    PruningOff,
    RecoverabilityPruning,
    ResolvedProblemReceipt,
    SerializableContract,
    SingletonAdmission,
    SRExecutionPolicy,
    SRMethodPolicy,
    SRObservationPolicy,
    SRRunResult,
    SRStopPolicy,
    TrustRegionPruning,
)
from pipelines.static_adapt.numerical_physical_integrity import (
    NUMERICAL_PHYSICAL_INTEGRITY_SCHEMA,
    NumericalPhysicalIntegrityReceipt,
)


RA_ADAPT_PROTOCOL_SCHEMA_V1 = "paper_i_ra_adapt_resolved_protocol_v1"
RA_ADAPT_PROTOCOL_SCHEMA_V2 = "paper_i_ra_adapt_resolved_protocol_v2"
# Historical bundle code imports the unsuffixed name.  Keep it pinned to the
# immutable v1 namespace; canonical full-response construction opts into v2
# explicitly.
RA_ADAPT_PROTOCOL_SCHEMA = RA_ADAPT_PROTOCOL_SCHEMA_V1
RA_ADAPT_PROTOCOL_SCHEMAS = frozenset(
    {RA_ADAPT_PROTOCOL_SCHEMA_V1, RA_ADAPT_PROTOCOL_SCHEMA_V2}
)
APPEND_ADAPT_PROTOCOL_SCHEMA = "paper_i_append_adapt_resolved_protocol_v1"
RA_ADAPT_RESULT_SCHEMA_V1 = "paper_i_ra_adapt_result_v1"
RA_ADAPT_RESULT_SCHEMA_V2 = "paper_i_ra_adapt_result_v2"
RA_ADAPT_RESULT_SCHEMA = RA_ADAPT_RESULT_SCHEMA_V1
APPEND_ADAPT_RESULT_SCHEMA = "paper_i_append_adapt_result_v1"
BUNDLE_PROTOCOL_MATERIALIZATION_SCHEMA_V1 = (
    "paper_i_ra_adapt_bundle_protocol_materialization_v1"
)
BUNDLE_PROTOCOL_MATERIALIZATION_SCHEMA_V2 = (
    "paper_i_ra_adapt_bundle_protocol_materialization_v2"
)
BUNDLE_PROTOCOL_MATERIALIZATION_SCHEMA = (
    BUNDLE_PROTOCOL_MATERIALIZATION_SCHEMA_V1
)
RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V1 = "paper_i_ra_adapt_route_contract_v1"
RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V2 = "paper_i_ra_adapt_route_contract_v2"
RA_ADAPT_NONSTATIONARY_INCREMENTAL_FULL_RESPONSE_ALGORITHM_ID = (
    "paper_i_ra_adapt_nonstationary_incremental_full_response_v2"
)
RA_ADAPT_SINGLETON_PHASE3_PLATEAU_ALGORITHM_ID = (
    "paper_i_ra_adapt_singleton_phase3_population_on_insertion_plateau_v1"
)
RA_ADAPT_SINGLETON_LATCHED_PHASE3_ALGORITHM_ID = (
    "paper_i_ra_adapt_singleton_latched_phase3_separate_plateau_insertion_v1"
)
RA_ADAPT_PHASE3_POPULATION_ALL_ROUNDS = "all_controller_rounds_v1"
RA_ADAPT_PHASE3_POPULATION_ON_INSERTION_PLATEAU = (
    "same_round_insertion_plateau_predicate_v1"
)
RA_ADAPT_PHASE3_POPULATION_LATCHED_ON_PROGRESS_PLATEAU = (
    "first_open_progress_plateau_predicate_latched_v1"
)
RA_ADAPT_PHASE3_PREPLATEAU_WINNER_MATERIALIZATION = (
    "phase2_winner_only_refit_geometry_v1"
)
RA_ADAPT_POST_LATCH_INSERTION_TRIGGER_SCOPE = (
    "prior_full_phase3_accepted_transition_global_prior_mean_v1"
)
RA_ADAPT_INCREMENTAL_CANDIDATE_GAIN_POLICY = (
    "joint_minus_active_only_supported_trust_v1"
)
RA_ADAPT_FULL_RESPONSE_REFIT_INITIALIZATION_POLICY = (
    "exact_applied_joint_step_guarded_v1"
)
CANDIDATE_LINEAGE_SCHEMA = "ra_adapt_candidate_lineage_receipt_v1"
CANDIDATE_INVENTORY_LINEAGE_SCHEMA = (
    "ra_adapt_candidate_inventory_lineage_receipt_v1"
)

CANDIDATE_REPRESENTATION_MACRO = "macro_generator_v1"
CANDIDATE_REPRESENTATION_SINGLE_PAULI = "single_pauli_word_v1"
CANDIDATE_REPRESENTATIONS = frozenset(
    {
        CANDIDATE_REPRESENTATION_MACRO,
        CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    }
)

ACTIVE_GRADIENT_STATIONARY = "stationary_source_response_v1"
ACTIVE_GRADIENT_MEASURED = "measured_residual_response_v1"
ACTIVE_GRADIENT_POLICIES = frozenset(
    {ACTIVE_GRADIENT_STATIONARY, ACTIVE_GRADIENT_MEASURED}
)

RESOURCE_WEIGHTING_LATE = "late_resource_weighting_v1"
RESOURCE_WEIGHTING_ALL_PHASE = "all_phase_resource_weighting_v1"
RESOURCE_WEIGHTING_SCOPES = frozenset(
    {RESOURCE_WEIGHTING_LATE, RESOURCE_WEIGHTING_ALL_PHASE}
)

EXACT_ORDERED_INSERTION_CHART = "exact_ordered_insertion_zero_angle_v1"
PROJECTED_GENERALIZED_SOLVER = (
    "supported_metric_projected_generalized_trust_v1"
)
SOURCE_GRAM_NO_OVERLAP_TRUST = (
    "supported_source_gram_no_endpoint_overlap_trust_v1"
)
ENDPOINT_OVERLAP_DISPLACEMENT_TRUST = (
    "supported_predicted_fs_endpoint_overlap_trust_v1"
)
FULL_ENLARGED_ACCEPTED_REFIT = "full_ansatz_v1"
SUPPORTED_FS_WHITENED_REFIT_CHART = (
    "supported_fs_whitened_fixed_v1"
)
EXPANDED_RUNTIME_PROJECTED_LOGICAL_BASE_CHART = (
    "expanded_runtime_projected_logical_v1"
)
NATIVE_REFIT_CHART = "native_v1"
LOGICAL_SHARED_REDUCED_REFIT_BASE_CHART = (
    "logical_shared_reduced_v1"
)
RA_STAGED_SELECTOR_ID = "ra_adapt_staged_phase_i_ii_iii_funnel_v1"
APPEND_CONVENTIONAL_SELECTOR_ID = (
    "append_adapt_largest_absolute_commutator_gradient_v1"
)
APPEND_CONVENTIONAL_SELECTOR_SCOPE = (
    "conventional_append_no_phase3_no_trust_v1"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, CanonicalContract):
        return value.to_dict()
    # SR policy objects carry stable ``kind`` discriminators in their own
    # serializer.  Treating them as generic dataclasses silently erased those
    # discriminators and made a materialized request impossible to rehydrate.
    if isinstance(value, SerializableContract):
        return value.to_dict()
    if is_dataclass(value):
        return {
            item.name: _jsonable(getattr(value, item.name))
            for item in fields(value)
            if item.metadata.get("canonical", True)
            if getattr(value, item.name) is not None
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Canonical RA-ADAPT JSON forbids NaN and infinity.")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical byte representation used for protocol hashes."""

    return json.dumps(
        _jsonable(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: str, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return normalized


class CanonicalContract:
    """Deterministic JSON projection for RA-ADAPT contracts and receipts."""

    kind: ClassVar[str | None] = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            item.name: _jsonable(getattr(self, item.name))
            for item in fields(self)
            if item.metadata.get("canonical", True)
            if getattr(self, item.name) is not None
        }
        if self.kind is not None:
            payload = {"kind": self.kind, **payload}
        return payload

    def to_json(self) -> str:
        return canonical_json_bytes(self).decode("utf-8")


def _default_single_pauli_adapter() -> Any:
    from pipelines.static_adapt.ra_adapt.adapters import (
        SinglePauliWordCandidateAdapter,
    )

    return SinglePauliWordCandidateAdapter()


@dataclass(frozen=True)
class RAAdaptRequest(CanonicalContract):
    """Public RA request.

    Active-gradient and resource-weighting choices are intentionally absent.
    They can only enter through a validated :class:`ResolvedRAAdaptProtocol`
    materialized by ``ra_adapt.bundles``.
    """

    kind: ClassVar[str] = "ra_adapt_request"

    adapter: Any = field(default_factory=_default_single_pauli_adapter)
    method: SRMethodPolicy = field(default_factory=SRMethodPolicy)
    execution: SRExecutionPolicy = field(default_factory=SRExecutionPolicy)
    observation: SRObservationPolicy = field(default_factory=SRObservationPolicy)

    def __post_init__(self) -> None:
        representation = getattr(
            self.adapter, "candidate_representation_id", None
        )
        if representation not in CANDIDATE_REPRESENTATIONS:
            raise TypeError(
                "adapter must implement the canonical candidate-representation "
                "adapter contract."
            )
        if not isinstance(self.method, SRMethodPolicy):
            raise TypeError("method must be an SRMethodPolicy.")
        if not isinstance(self.execution, SRExecutionPolicy):
            raise TypeError("execution must be an SRExecutionPolicy.")
        if not isinstance(self.observation, SRObservationPolicy):
            raise TypeError("observation must be an SRObservationPolicy.")


@dataclass(frozen=True)
class RAAdaptOperationalControls(SerializableContract):
    """Bounded execution mechanics for an already-authorized RA protocol.

    These controls may shorten the authorized controller horizon, select a
    typed accepted-state resume checkpoint, and redirect observation
    sidecars.  They never enter or replace the resolved scientific protocol.
    """

    kind: ClassVar[str] = "ra_adapt_operational_controls_v1"

    maximum_controller_rounds: int
    resume: FreshStart | AcceptedStateResume = field(
        default_factory=FreshStart
    )
    observation: SRObservationPolicy = field(
        default_factory=SRObservationPolicy
    )

    def __post_init__(self) -> None:
        rounds = self.maximum_controller_rounds
        if (
            isinstance(rounds, bool)
            or int(rounds) != rounds
            or int(rounds) < 1
        ):
            raise ValueError(
                "maximum_controller_rounds must be a positive integer."
            )
        object.__setattr__(
            self,
            "maximum_controller_rounds",
            int(rounds),
        )
        if not isinstance(self.resume, (FreshStart, AcceptedStateResume)):
            raise TypeError(
                "resume must be FreshStart or AcceptedStateResume."
            )
        if not isinstance(self.observation, SRObservationPolicy):
            raise TypeError(
                "observation must be an SRObservationPolicy."
            )


@dataclass(frozen=True)
class AppendAdaptRequest(CanonicalContract):
    """Public conventional Append-ADAPT request."""

    kind: ClassVar[str] = "append_adapt_request"

    adapter: Any = field(default_factory=_default_single_pauli_adapter)
    execution: SRExecutionPolicy = field(default_factory=SRExecutionPolicy)
    observation: SRObservationPolicy = field(default_factory=SRObservationPolicy)

    def __post_init__(self) -> None:
        representation = getattr(
            self.adapter, "candidate_representation_id", None
        )
        if representation not in CANDIDATE_REPRESENTATIONS:
            raise TypeError(
                "adapter must implement the canonical candidate-representation "
                "adapter contract."
            )
        if not isinstance(self.execution, SRExecutionPolicy):
            raise TypeError("execution must be an SRExecutionPolicy.")
        if not isinstance(self.observation, SRObservationPolicy):
            raise TypeError("observation must be an SRObservationPolicy.")


@dataclass(frozen=True)
class PoolInventoryReceipt(CanonicalContract):
    schema: str
    candidate_representation: str
    ordered_labels: tuple[str, ...]
    ordered_labels_sha256: str
    ordered_pool_sha256: str
    count: int
    removed_labels: tuple[str, ...] = ()
    source_parent_ordered_labels_sha256: str | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        if self.candidate_representation not in CANDIDATE_REPRESENTATIONS:
            raise ValueError("Unknown candidate representation.")
        if int(self.count) != len(self.ordered_labels):
            raise ValueError("Pool count must equal ordered-label count.")
        _require_sha256(
            self.ordered_labels_sha256, name="ordered_labels_sha256"
        )
        if self.ordered_labels_sha256 != canonical_sha256(
            list(self.ordered_labels)
        ):
            raise ValueError(
                "ordered_labels_sha256 does not match the ordered labels."
            )
        _require_sha256(self.ordered_pool_sha256, name="ordered_pool_sha256")
        if self.source_parent_ordered_labels_sha256 is not None:
            _require_sha256(
                self.source_parent_ordered_labels_sha256,
                name="source_parent_ordered_labels_sha256",
            )
        payload = self.to_dict()
        payload.pop("sha256", None)
        expected = canonical_sha256(payload)
        if self.sha256 is None:
            object.__setattr__(self, "sha256", expected)
        elif _require_sha256(self.sha256, name="sha256") != expected:
            raise ValueError(
                "Pool inventory digest does not match its canonical payload."
            )


@dataclass(frozen=True)
class CandidateInventoryLineageRow(CanonicalContract):
    """Compact, ordered lineage identity for one executable candidate."""

    label: str
    representation_id: str
    generator_identity: str
    parent_identities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        label = str(self.label).strip()
        generator_identity = str(self.generator_identity).strip()
        if not label:
            raise ValueError("Candidate lineage labels must be nonempty.")
        if self.representation_id not in CANDIDATE_REPRESENTATIONS:
            raise ValueError("Unknown candidate representation.")
        if not generator_identity:
            raise ValueError(
                "Candidate lineage generator identities must be nonempty."
            )
        parents = tuple(str(value).strip() for value in self.parent_identities)
        if any(not value for value in parents):
            raise ValueError(
                "Candidate lineage parent identities must be nonempty."
            )
        if len(set(parents)) != len(parents):
            raise ValueError(
                "Candidate lineage parent identities must be unique."
            )
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "generator_identity", generator_identity)
        object.__setattr__(self, "parent_identities", parents)


@dataclass(frozen=True)
class CandidateInventoryLineageReceipt(CanonicalContract):
    """Compact full-inventory lineage, kept separate from pool identity."""

    schema: str
    candidate_representation: str
    pool_inventory_sha256: str
    ordered_rows: tuple[CandidateInventoryLineageRow, ...]
    ordered_rows_sha256: str
    count: int
    sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema != CANDIDATE_INVENTORY_LINEAGE_SCHEMA:
            raise ValueError("Unknown candidate-inventory lineage schema.")
        if self.candidate_representation not in CANDIDATE_REPRESENTATIONS:
            raise ValueError("Unknown candidate representation.")
        _require_sha256(
            self.pool_inventory_sha256,
            name="pool_inventory_sha256",
        )
        if isinstance(self.count, bool) or int(self.count) < 1:
            raise ValueError(
                "Candidate-inventory lineage count must be positive."
            )
        if int(self.count) != len(self.ordered_rows):
            raise ValueError(
                "Candidate-inventory lineage count must equal ordered-row "
                "count."
            )
        if any(
            not isinstance(row, CandidateInventoryLineageRow)
            for row in self.ordered_rows
        ):
            raise TypeError(
                "Candidate-inventory lineage rows must be typed receipts."
            )
        if any(
            row.representation_id != self.candidate_representation
            for row in self.ordered_rows
        ):
            raise ValueError(
                "Candidate-inventory lineage rows changed representation."
            )
        _require_sha256(
            self.ordered_rows_sha256,
            name="ordered_rows_sha256",
        )
        expected_rows_sha256 = canonical_sha256(
            [row.to_dict() for row in self.ordered_rows]
        )
        if self.ordered_rows_sha256 != expected_rows_sha256:
            raise ValueError(
                "Candidate-inventory ordered-row digest does not match its "
                "rows."
            )
        payload = self.to_dict()
        payload.pop("sha256", None)
        expected = canonical_sha256(payload)
        if self.sha256 is None:
            object.__setattr__(self, "sha256", expected)
        elif _require_sha256(self.sha256, name="sha256") != expected:
            raise ValueError(
                "Candidate-inventory lineage digest does not match its "
                "canonical payload."
            )

    def authority_binding(self) -> dict[str, Any]:
        """Return the compact protocol binding for this full receipt."""

        return {
            "schema": str(self.schema),
            "candidate_representation": str(
                self.candidate_representation
            ),
            "pool_inventory_sha256": str(self.pool_inventory_sha256),
            "count": int(self.count),
            "ordered_rows_sha256": str(self.ordered_rows_sha256),
            "sha256": str(self.sha256),
        }


@dataclass(frozen=True)
class CandidateLineageReceipt(CanonicalContract):
    """One admitted candidate at its explicit ordered-insertion position."""

    representation_id: str
    candidate_label: str
    generator_identity: str
    parent_identities: tuple[str, ...]
    insertion_position: int
    candidate_manifest_sha256: str | None = None
    schema: str = CANDIDATE_LINEAGE_SCHEMA
    sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema != CANDIDATE_LINEAGE_SCHEMA:
            raise ValueError("Unknown candidate-lineage receipt schema.")
        if self.representation_id not in CANDIDATE_REPRESENTATIONS:
            raise ValueError("Unknown candidate representation.")
        label = str(self.candidate_label).strip()
        generator_identity = str(self.generator_identity).strip()
        if not label:
            raise ValueError("Candidate lineage labels must be nonempty.")
        if not generator_identity:
            raise ValueError(
                "Candidate lineage generator identities must be nonempty."
            )
        parents = tuple(str(value).strip() for value in self.parent_identities)
        if any(not value for value in parents):
            raise ValueError(
                "Candidate lineage parent identities must be nonempty."
            )
        if len(set(parents)) != len(parents):
            raise ValueError(
                "Candidate lineage parent identities must be unique."
            )
        if isinstance(self.insertion_position, bool) or int(
            self.insertion_position
        ) < 0:
            raise ValueError("insertion_position must be nonnegative.")
        if self.candidate_manifest_sha256 is not None:
            _require_sha256(
                self.candidate_manifest_sha256,
                name="candidate_manifest_sha256",
            )
        object.__setattr__(self, "candidate_label", label)
        object.__setattr__(self, "generator_identity", generator_identity)
        object.__setattr__(self, "parent_identities", parents)
        object.__setattr__(
            self, "insertion_position", int(self.insertion_position)
        )
        payload = self.to_dict()
        payload.pop("sha256", None)
        expected = canonical_sha256(payload)
        if self.sha256 is None:
            object.__setattr__(self, "sha256", expected)
        elif _require_sha256(self.sha256, name="sha256") != expected:
            raise ValueError(
                "Candidate-lineage digest does not match its canonical "
                "payload."
            )


@dataclass(frozen=True)
class PhaseIIIStabilizationReceipt(CanonicalContract):
    solver_policy: str
    kappa_stabilization_shift: float
    trust_boundary_multiplier_lambda: float
    total_metric_multiplier_mu: float
    trust_boundary_active: bool
    metric_whitening_active: bool = False
    metric_inverse_sqrt_constructed: bool = False

    def __post_init__(self) -> None:
        if self.solver_policy != PROJECTED_GENERALIZED_SOLVER:
            raise ValueError("The canonical RA engine requires the projected solver.")
        kappa = float(self.kappa_stabilization_shift)
        boundary = float(self.trust_boundary_multiplier_lambda)
        total = float(self.total_metric_multiplier_mu)
        if not all(math.isfinite(value) for value in (kappa, boundary, total)):
            raise ValueError("Phase-III multipliers must be finite.")
        if min(kappa, boundary, total) < 0.0:
            raise ValueError("Phase-III multipliers must be nonnegative.")
        tolerance = 128.0 * math.ulp(max(1.0, abs(total)))
        if abs(total - (kappa + boundary)) > tolerance:
            raise ValueError("Phase-III receipt must satisfy mu = kappa + lambda.")
        if bool(self.trust_boundary_active) != bool(boundary > 0.0):
            raise ValueError("Trust-boundary activity is derived from lambda > 0.")
        if self.metric_whitening_active or self.metric_inverse_sqrt_constructed:
            raise ValueError("The projected generalized solver is raw-Gram only.")


@dataclass(frozen=True)
class PhaseIIIMultiplierContract(CanonicalContract):
    """Field-level semantics for projected generalized Phase-III receipts."""

    schema: str = "ra_adapt_phase3_multiplier_contract_v1"
    kappa_field: str = "kappa_stabilization_shift"
    lambda_field: str = "trust_boundary_multiplier_lambda"
    mu_field: str = "total_metric_multiplier_mu"
    additive_identity: str = "mu_equals_kappa_plus_lambda_v1"
    boundary_activity_rule: str = "boundary_active_iff_lambda_gt_zero_v1"
    curvature_only_regime: str = (
        "kappa_gt_zero_lambda_eq_zero_boundary_inactive_v1"
    )
    trust_bounded_regime: str = (
        "kappa_ge_zero_lambda_gt_zero_boundary_active_v1"
    )

    def __post_init__(self) -> None:
        expected = {
            "schema": "ra_adapt_phase3_multiplier_contract_v1",
            "kappa_field": "kappa_stabilization_shift",
            "lambda_field": "trust_boundary_multiplier_lambda",
            "mu_field": "total_metric_multiplier_mu",
            "additive_identity": "mu_equals_kappa_plus_lambda_v1",
            "boundary_activity_rule": (
                "boundary_active_iff_lambda_gt_zero_v1"
            ),
            "curvature_only_regime": (
                "kappa_gt_zero_lambda_eq_zero_boundary_inactive_v1"
            ),
            "trust_bounded_regime": (
                "kappa_ge_zero_lambda_gt_zero_boundary_active_v1"
            ),
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(
                    f"Phase-III multiplier contract field {name!r} drifted."
                )


@dataclass(frozen=True)
class PolicyEchoReceipt(CanonicalContract):
    active_gradient_policy: str
    resource_weighting_scope: str
    active_gradient_indices_acquired: tuple[int, ...] = ()
    active_gradient_charge: int = 0
    phase3_candidate_gain_policy: str | None = None
    accepted_refit_initialization_policy: str | None = None

    def __post_init__(self) -> None:
        if self.active_gradient_policy not in ACTIVE_GRADIENT_POLICIES:
            raise ValueError("Unknown active-gradient policy.")
        if self.resource_weighting_scope not in RESOURCE_WEIGHTING_SCOPES:
            raise ValueError("Unknown resource-weighting scope.")
        if (
            self.active_gradient_policy == ACTIVE_GRADIENT_STATIONARY
            and (
                self.active_gradient_indices_acquired
                or int(self.active_gradient_charge) != 0
            )
        ):
            raise ValueError(
                "Stationary-source response cannot acquire or charge active "
                "gradients."
            )
        policies = (
            self.phase3_candidate_gain_policy,
            self.accepted_refit_initialization_policy,
        )
        if any(value is None for value in policies) and any(
            value is not None for value in policies
        ):
            raise ValueError(
                "Candidate-gain and accepted-refit initialization policies "
                "must be echoed together."
            )
        if self.phase3_candidate_gain_policy is not None and (
            self.phase3_candidate_gain_policy
            != RA_ADAPT_INCREMENTAL_CANDIDATE_GAIN_POLICY
            or self.accepted_refit_initialization_policy
            != RA_ADAPT_FULL_RESPONSE_REFIT_INITIALIZATION_POLICY
        ):
            raise ValueError("Unknown RA full-response policy echo.")


@dataclass(frozen=True)
class BundleProtocolMaterializationReceipt(CanonicalContract):
    """Serialized proof that one protocol came from a validated bundle cell.

    The receipt is provenance, not permission to execute a run.  Execution
    additionally requires an in-memory capability bound to the final protocol
    digest; that capability is minted only by the validated bundle loader.
    """

    schema: str
    bundle_id: str
    bundle_manifest_sha256: str
    source_locks_sha256: str
    source_lock_refs_sha256: str
    cell_id: str
    source_lock_id: str
    protocol_schema: str
    algorithm_id: str
    candidate_representation: str
    selector_identity: str
    active_gradient_policy: str
    resource_weighting_scope: str
    sha256: str

    def __post_init__(self) -> None:
        if self.schema not in {
            BUNDLE_PROTOCOL_MATERIALIZATION_SCHEMA_V1,
            BUNDLE_PROTOCOL_MATERIALIZATION_SCHEMA_V2,
        }:
            raise ValueError("Unknown bundle protocol materialization schema.")
        if not str(self.bundle_id).strip():
            raise ValueError("Bundle materialization requires a bundle id.")
        if not str(self.cell_id).strip():
            raise ValueError("Bundle materialization requires a cell id.")
        if not str(self.source_lock_id).strip():
            raise ValueError(
                "Bundle materialization requires a source-lock id."
            )
        for name in (
            "bundle_manifest_sha256",
            "source_locks_sha256",
            "source_lock_refs_sha256",
            "sha256",
        ):
            _require_sha256(str(getattr(self, name)), name=name)
        if self.protocol_schema not in {
            *RA_ADAPT_PROTOCOL_SCHEMAS,
            APPEND_ADAPT_PROTOCOL_SCHEMA,
        }:
            raise ValueError("Unknown materialized protocol schema.")
        expected_materialization_schema = (
            BUNDLE_PROTOCOL_MATERIALIZATION_SCHEMA_V2
            if self.protocol_schema == RA_ADAPT_PROTOCOL_SCHEMA_V2
            else BUNDLE_PROTOCOL_MATERIALIZATION_SCHEMA_V1
        )
        if self.schema != expected_materialization_schema:
            raise ValueError(
                "Bundle materialization schema does not match its protocol "
                "schema."
            )
        if (
            self.algorithm_id
            == RA_ADAPT_NONSTATIONARY_INCREMENTAL_FULL_RESPONSE_ALGORITHM_ID
        ) != (self.protocol_schema == RA_ADAPT_PROTOCOL_SCHEMA_V2):
            raise ValueError(
                "RA full-response algorithm and materialized protocol schema "
                "must use the same version."
            )
        if self.candidate_representation not in CANDIDATE_REPRESENTATIONS:
            raise ValueError(
                "Unknown materialized candidate representation."
            )
        if self.active_gradient_policy not in ACTIVE_GRADIENT_POLICIES:
            raise ValueError("Unknown materialized active-gradient policy.")
        if self.resource_weighting_scope not in RESOURCE_WEIGHTING_SCOPES:
            raise ValueError(
                "Unknown materialized resource-weighting scope."
            )
        expected_selector = (
            RA_STAGED_SELECTOR_ID
            if self.protocol_schema in RA_ADAPT_PROTOCOL_SCHEMAS
            else APPEND_CONVENTIONAL_SELECTOR_ID
        )
        if self.selector_identity != expected_selector:
            raise ValueError(
                "Bundle materialization selector identity drifted."
            )
        payload = self.to_dict()
        payload.pop("sha256", None)
        if self.sha256 != canonical_sha256(payload):
            raise ValueError(
                "Bundle materialization digest does not match its "
                "canonical payload."
            )


class BundleProtocolMaterializationAuthority:
    """Private construction capability for bundle-only protocol policies.

    Callers cannot construct this class through its interface.  The bundle
    materializer mints an unbound capability after source-lock validation;
    the bundle loader mints a final capability bound to the serialized
    protocol digest after validating the complete on-disk bundle.
    """

    __slots__ = ("_receipt", "_source_lock_refs", "_protocol_sha256")

    def __new__(cls, *_args: Any, **_kwargs: Any) -> Any:
        raise TypeError(
            "Bundle protocol materialization authority is minted only by "
            "ra_adapt.bundles."
        )

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError(
            "Bundle protocol materialization authority is immutable."
        )

    @property
    def receipt(self) -> BundleProtocolMaterializationReceipt:
        return self._receipt

    @property
    def protocol_sha256(self) -> str | None:
        return self._protocol_sha256

    @property
    def source_lock_refs(self) -> Mapping[str, str]:
        return self._source_lock_refs


def _mint_bundle_protocol_materialization_authority(
    receipt: BundleProtocolMaterializationReceipt,
    *,
    source_lock_refs: Mapping[str, str],
    protocol_sha256: str | None = None,
) -> BundleProtocolMaterializationAuthority:
    """Mint the private capability used only by ``ra_adapt.bundles``."""

    if not isinstance(receipt, BundleProtocolMaterializationReceipt):
        raise TypeError(
            "receipt must be BundleProtocolMaterializationReceipt."
        )
    if not isinstance(source_lock_refs, Mapping) or not source_lock_refs:
        raise ValueError(
            "Bundle authority requires nonempty source-lock refs."
        )
    normalized_refs = {
        str(key): str(value)
        for key, value in sorted(
            source_lock_refs.items(), key=lambda pair: str(pair[0])
        )
    }
    if canonical_sha256(normalized_refs) != (
        receipt.source_lock_refs_sha256
    ):
        raise ValueError(
            "Bundle authority source-lock refs do not match its receipt."
        )
    bound_digest = (
        None
        if protocol_sha256 is None
        else _require_sha256(
            protocol_sha256, name="materialized protocol_sha256"
        )
    )
    authority = object.__new__(BundleProtocolMaterializationAuthority)
    object.__setattr__(authority, "_receipt", receipt)
    object.__setattr__(
        authority, "_source_lock_refs", normalized_refs
    )
    object.__setattr__(
        authority, "_protocol_sha256", bound_digest
    )
    return authority


def bundle_protocol_materialization_receipt(
    *,
    bundle_id: str,
    bundle_manifest_sha256: str,
    source_locks_sha256: str,
    source_lock_refs: Mapping[str, str],
    cell_id: str,
    source_lock_id: str,
    protocol_schema: str,
    algorithm_id: str,
    candidate_representation: str,
    selector_identity: str,
    active_gradient_policy: str,
    resource_weighting_scope: str,
) -> BundleProtocolMaterializationReceipt:
    """Build the canonical receipt payload for a validated bundle cell.

    This helper does not mint an execution capability.  It is kept separate
    so bundle validation can recompute and compare the receipt byte-for-byte.
    """

    if not isinstance(source_lock_refs, Mapping) or not source_lock_refs:
        raise ValueError(
            "Bundle materialization requires nonempty source-lock refs."
        )
    normalized_refs = {
        str(key): str(value)
        for key, value in sorted(
            source_lock_refs.items(), key=lambda pair: str(pair[0])
        )
    }
    for name, value in normalized_refs.items():
        if name == "cell_source_lock_id":
            if not value.strip():
                raise ValueError("cell_source_lock_id cannot be empty.")
        else:
            _require_sha256(
                value, name=f"source_lock_refs.{name}"
            )
    payload = {
        "schema": (
            BUNDLE_PROTOCOL_MATERIALIZATION_SCHEMA_V2
            if str(protocol_schema) == RA_ADAPT_PROTOCOL_SCHEMA_V2
            else BUNDLE_PROTOCOL_MATERIALIZATION_SCHEMA_V1
        ),
        "bundle_id": str(bundle_id),
        "bundle_manifest_sha256": str(bundle_manifest_sha256),
        "source_locks_sha256": str(source_locks_sha256),
        "source_lock_refs_sha256": canonical_sha256(normalized_refs),
        "cell_id": str(cell_id),
        "source_lock_id": str(source_lock_id),
        "protocol_schema": str(protocol_schema),
        "algorithm_id": str(algorithm_id),
        "candidate_representation": str(candidate_representation),
        "selector_identity": str(selector_identity),
        "active_gradient_policy": str(active_gradient_policy),
        "resource_weighting_scope": str(resource_weighting_scope),
    }
    return BundleProtocolMaterializationReceipt(
        **payload,
        sha256=canonical_sha256(payload),
    )


@dataclass(frozen=True)
class ResolvedRAAdaptProtocol(CanonicalContract):
    """Immutable, digested protocol accepted only after bundle validation."""

    schema: str
    algorithm_id: str
    candidate_representation: str
    adapter_id: str
    selector_identity: str
    active_gradient_policy: str
    resource_weighting_scope: str
    derivative_chart_id: str
    trust_policy_id: str
    phase3_solver_id: str
    phase3_multiplier_contract: PhaseIIIMultiplierContract
    accepted_refit_scope: str
    accepted_refit_coordinate_chart: str
    accepted_refit_base_chart_policy: str
    problem: ResolvedProblemReceipt
    parent_inventory: PoolInventoryReceipt
    executable_pool: PoolInventoryReceipt
    optimizer: str
    optimizer_maxiter: int
    stopping_rule: Mapping[str, Any]
    horizon: int
    seeds: Mapping[str, int]
    estimator_accounting_convention: str
    compile_identity: Mapping[str, Any]
    lineage_authority: Mapping[str, Any]
    source_locks: Mapping[str, str]
    bundle_id: str
    bundle_manifest_sha256: str
    execution_authorized: bool
    request: RAAdaptRequest | AppendAdaptRequest
    sha256: str
    selector_scope: str | None = None
    route_contract: Mapping[str, Any] | None = None
    baseline_consumption: Mapping[str, Any] | None = None
    bundle_materialization: (
        BundleProtocolMaterializationReceipt | None
    ) = None
    _materialization_authority: (
        BundleProtocolMaterializationAuthority | None
    ) = field(
        default=None,
        repr=False,
        compare=False,
        metadata={"canonical": False},
    )

    def __post_init__(self) -> None:
        if self.schema not in {
            *RA_ADAPT_PROTOCOL_SCHEMAS,
            APPEND_ADAPT_PROTOCOL_SCHEMA,
        }:
            raise ValueError("Unknown resolved protocol schema.")
        is_ra_protocol = self.schema in RA_ADAPT_PROTOCOL_SCHEMAS
        is_full_response_v2 = bool(
            self.algorithm_id
            == RA_ADAPT_NONSTATIONARY_INCREMENTAL_FULL_RESPONSE_ALGORITHM_ID
        )
        if is_ra_protocol and (
            is_full_response_v2
            != (self.schema == RA_ADAPT_PROTOCOL_SCHEMA_V2)
        ):
            raise ValueError(
                "RA full-response algorithm and protocol schema must use the "
                "same version."
            )
        if self.candidate_representation not in CANDIDATE_REPRESENTATIONS:
            raise ValueError("Unknown candidate representation.")
        if self.active_gradient_policy not in ACTIVE_GRADIENT_POLICIES:
            raise ValueError("Unknown active-gradient policy.")
        if self.resource_weighting_scope not in RESOURCE_WEIGHTING_SCOPES:
            raise ValueError("Unknown resource-weighting scope.")
        if self.derivative_chart_id != EXACT_ORDERED_INSERTION_CHART:
            raise ValueError("Resolved protocols require exact ordered insertion.")
        if self.trust_policy_id not in {
            SOURCE_GRAM_NO_OVERLAP_TRUST,
            ENDPOINT_OVERLAP_DISPLACEMENT_TRUST,
        }:
            raise ValueError("Resolved protocols carry an unknown trust policy.")
        if self.phase3_solver_id != PROJECTED_GENERALIZED_SOLVER:
            raise ValueError("Resolved protocols require the projected solver.")
        if not isinstance(
            self.phase3_multiplier_contract, PhaseIIIMultiplierContract
        ):
            raise TypeError(
                "Resolved protocols require a typed Phase-III multiplier contract."
            )
        if self.accepted_refit_scope != FULL_ENLARGED_ACCEPTED_REFIT:
            raise ValueError("Accepted refit must cover the full enlarged ansatz.")
        if is_ra_protocol:
            if (
                self.accepted_refit_coordinate_chart
                != SUPPORTED_FS_WHITENED_REFIT_CHART
            ):
                raise ValueError(
                    "Resolved RA protocols require the supported-FS chart."
                )
            if (
                self.accepted_refit_base_chart_policy
                != EXPANDED_RUNTIME_PROJECTED_LOGICAL_BASE_CHART
            ):
                raise ValueError(
                    "Resolved RA protocols require the expanded-runtime "
                    "projected-logical accepted-refit base chart."
                )
            if self.selector_scope is not None:
                raise ValueError(
                    "RA protocols must not carry an Append selector scope."
                )
        elif self.selector_scope is None:
            # Historical materializations predate the explicit Append
            # selector-scope field and used RA's supported-FS refit chart.
            # They remain loadable as immutable provenance, but the Append
            # executor rejects them before execution.
            if (
                self.accepted_refit_coordinate_chart
                != SUPPORTED_FS_WHITENED_REFIT_CHART
                or self.accepted_refit_base_chart_policy
                != EXPANDED_RUNTIME_PROJECTED_LOGICAL_BASE_CHART
            ):
                raise ValueError(
                    "Legacy Append protocols require their historical "
                    "supported-FS accepted-refit identity."
                )
        else:
            if (
                self.selector_scope
                != APPEND_CONVENTIONAL_SELECTOR_SCOPE
            ):
                raise ValueError("Unknown Append selector scope.")
            if self.accepted_refit_coordinate_chart != NATIVE_REFIT_CHART:
                raise ValueError(
                    "Conventional Append protocols require the native "
                    "accepted-refit chart."
                )
            if (
                self.accepted_refit_base_chart_policy
                != LOGICAL_SHARED_REDUCED_REFIT_BASE_CHART
            ):
                raise ValueError(
                    "Conventional Append protocols require the logical-shared "
                    "accepted-refit base chart."
                )
        if not isinstance(self.problem, ResolvedProblemReceipt):
            raise TypeError(
                "Resolved protocols require a typed resolved-problem receipt."
            )
        if int(self.horizon) < 1:
            raise ValueError("Resolved protocols require a positive horizon.")
        if int(self.optimizer_maxiter) < 1:
            raise ValueError("optimizer_maxiter must be positive.")
        if bool(self.execution_authorized):
            raise ValueError(
                "Materialized RA-ADAPT bundles are non-executing handoff artifacts."
            )
        if not str(self.bundle_id).strip():
            raise ValueError("Resolved protocols require a bundle id.")
        _require_sha256(
            self.bundle_manifest_sha256, name="bundle_manifest_sha256"
        )
        _require_sha256(self.sha256, name="sha256")
        for name, receipt in (
            ("route_contract", self.route_contract),
            ("baseline_consumption", self.baseline_consumption),
        ):
            if receipt is not None:
                if not isinstance(receipt, Mapping):
                    raise TypeError(f"{name} must be a mapping.")
                observed = _require_sha256(
                    str(receipt.get("sha256", "")),
                    name=f"{name}.sha256",
                )
                digest_payload = dict(receipt)
                digest_payload.pop("sha256", None)
                if observed != canonical_sha256(digest_payload):
                    raise ValueError(
                        f"{name} digest does not match its canonical payload."
                    )
        if self.bundle_materialization is not None:
            if not isinstance(
                self.bundle_materialization,
                BundleProtocolMaterializationReceipt,
            ):
                raise TypeError(
                    "bundle_materialization must be a typed receipt."
                )
            materialization = self.bundle_materialization
            expected_materialization = {
                "bundle_id": self.bundle_id,
                "bundle_manifest_sha256": self.bundle_manifest_sha256,
                "protocol_schema": self.schema,
                "algorithm_id": self.algorithm_id,
                "candidate_representation": self.candidate_representation,
                "selector_identity": self.selector_identity,
                "active_gradient_policy": self.active_gradient_policy,
                "resource_weighting_scope": self.resource_weighting_scope,
            }
            for name, expected_value in expected_materialization.items():
                if getattr(materialization, name) != expected_value:
                    raise ValueError(
                        "Bundle materialization receipt drifted at "
                        f"{name}."
                    )
            if (
                str(
                    self.source_locks.get(
                        "source_locks_manifest_sha256", ""
                    )
                )
                != materialization.source_locks_sha256
                or str(self.source_locks.get("cell_source_lock_id", ""))
                != materialization.source_lock_id
            ):
                raise ValueError(
                    "Bundle materialization source-lock identity drifted."
                )
            authority = self._materialization_authority
            if authority is not None:
                if not isinstance(
                    authority, BundleProtocolMaterializationAuthority
                ):
                    raise TypeError(
                        "Invalid bundle materialization authority."
                    )
                if authority.receipt != materialization:
                    raise ValueError(
                        "Bundle materialization authority receipt drifted."
                    )
        elif self._materialization_authority is not None:
            raise ValueError(
                "A materialization authority requires its serialized receipt."
            )
        lineage_binding = self.lineage_authority.get(
            "candidate_inventory_lineage"
        )
        if not isinstance(lineage_binding, Mapping):
            raise ValueError(
                "Resolved protocol lineage authority is missing the candidate "
                "inventory lineage binding."
            )
        if (
            str(lineage_binding.get("schema", ""))
            != CANDIDATE_INVENTORY_LINEAGE_SCHEMA
        ):
            raise ValueError(
                "Resolved protocol candidate-inventory lineage schema drifted."
            )
        if (
            str(lineage_binding.get("candidate_representation", ""))
            != self.candidate_representation
        ):
            raise ValueError(
                "Resolved protocol candidate-inventory representation drifted."
            )
        if int(lineage_binding.get("count", -1)) != int(
            self.executable_pool.count
        ):
            raise ValueError(
                "Resolved protocol candidate-inventory lineage count drifted."
            )
        if (
            str(lineage_binding.get("pool_inventory_sha256", ""))
            != str(self.executable_pool.sha256)
        ):
            raise ValueError(
                "Resolved protocol candidate-inventory pool binding drifted."
            )
        for name in ("ordered_rows_sha256", "sha256"):
            _require_sha256(
                str(lineage_binding.get(name, "")),
                name=(
                    "lineage_authority.candidate_inventory_lineage."
                    + name
                ),
            )
        if (
            self.schema == RA_ADAPT_PROTOCOL_SCHEMA_V2
            and self.algorithm_id
            == RA_ADAPT_NONSTATIONARY_INCREMENTAL_FULL_RESPONSE_ALGORITHM_ID
        ):
            expected_algorithm_semantics = {
                "active_response": ACTIVE_GRADIENT_MEASURED,
                "candidate_gain_policy": (
                    RA_ADAPT_INCREMENTAL_CANDIDATE_GAIN_POLICY
                ),
                "accepted_refit_initialization_policy": (
                    RA_ADAPT_FULL_RESPONSE_REFIT_INITIALIZATION_POLICY
                ),
                "full_response_coordinate_scope": (
                    "existing_active_plus_new_batch_v1"
                ),
            }
            if (
                self.active_gradient_policy != ACTIVE_GRADIENT_MEASURED
                or self.lineage_authority.get("algorithm_semantics")
                != expected_algorithm_semantics
            ):
                raise ValueError(
                    "Canonical RA v2 algorithm semantics drifted from its "
                    "protocol identity."
                )
        if is_ra_protocol:
            if self.selector_identity != RA_STAGED_SELECTOR_ID:
                raise ValueError("RA protocols require the staged selector.")
            if not isinstance(self.request, RAAdaptRequest):
                raise TypeError("RA protocol request must be RAAdaptRequest.")
        else:
            if self.selector_identity != APPEND_CONVENTIONAL_SELECTOR_ID:
                raise ValueError("Append protocols require the conventional selector.")
            if not isinstance(self.request, AppendAdaptRequest):
                raise TypeError(
                    "Append protocol request must be AppendAdaptRequest."
                )
        expected = canonical_sha256(self.digest_payload())
        if self.sha256 != expected:
            raise ValueError(
                "Resolved protocol digest does not match its canonical payload."
            )

    def digest_payload(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.pop("sha256", None)
        return payload


def _attach_validated_bundle_protocol_authority(
    protocol: ResolvedRAAdaptProtocol,
    authority: BundleProtocolMaterializationAuthority,
) -> ResolvedRAAdaptProtocol:
    """Attach a final protocol-digest binding after bundle validation."""

    if not isinstance(protocol, ResolvedRAAdaptProtocol):
        raise TypeError("protocol must be ResolvedRAAdaptProtocol.")
    if not isinstance(
        authority, BundleProtocolMaterializationAuthority
    ):
        raise TypeError(
            "authority must be BundleProtocolMaterializationAuthority."
        )
    if authority.receipt != protocol.bundle_materialization:
        raise ValueError(
            "Validated bundle authority does not match the protocol receipt."
        )
    if authority.protocol_sha256 != protocol.sha256:
        raise ValueError(
            "Validated bundle authority does not bind the protocol digest."
        )
    return replace(
        protocol,
        _materialization_authority=authority,
    )


def require_protocol_materialization_authority(
    protocol: ResolvedRAAdaptProtocol,
    *,
    ordinary_algorithm_id: str,
    ordinary_bundle_id: str,
    ordinary_bundle_manifest_sha256: str,
    additional_ordinary_identities: tuple[
        tuple[str, str, str], ...
    ] = (),
) -> None:
    """Fail closed on bundle protocols not loaded through bundle validation."""

    materialization = protocol.bundle_materialization
    if materialization is None:
        allowed_ordinary_identities = {
            (
                str(ordinary_algorithm_id),
                str(ordinary_bundle_id),
                str(ordinary_bundle_manifest_sha256),
            ),
            *(
                (str(algorithm), str(bundle), str(digest))
                for algorithm, bundle, digest
                in additional_ordinary_identities
            ),
        }
        if (
            protocol.active_gradient_policy != ACTIVE_GRADIENT_MEASURED
            or protocol.resource_weighting_scope
            != RESOURCE_WEIGHTING_ALL_PHASE
            or (
                str(protocol.algorithm_id),
                str(protocol.bundle_id),
                str(protocol.bundle_manifest_sha256),
            )
            not in allowed_ordinary_identities
        ):
            raise ValueError(
                "Study policies require a validated bundle protocol "
                "materialization authority."
            )
        return
    authority = protocol._materialization_authority
    if not isinstance(
        authority, BundleProtocolMaterializationAuthority
    ):
        raise ValueError(
            "Bundle protocol must be loaded through "
            "ra_adapt.bundles.load_validated_bundle_protocol before "
            "execution."
        )
    if (
        authority.receipt != materialization
        or authority.protocol_sha256 != protocol.sha256
    ):
        raise ValueError(
            "Bundle protocol materialization authority drifted from the "
            "serialized protocol."
        )


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _without_kind(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("kind", None)
    return payload


def _adapter_from_mapping(value: Any) -> Any:
    from pipelines.static_adapt.ra_adapt.adapters import (
        GLOBAL_SINGLE_PAULI_ADAPTER_ID,
        H2O_LINEAR_FD_SECTOR_COMPLETE_PAULI_BLOCK_ADAPTER_ID,
        H2O_LINEAR_FD_SINGLE_PAULI_ADAPTER_ID,
        H2O_LINEAR_FD_SYMMETRY_COMPLETE_ADAPTER_ID,
        MACRO_ADAPTER_ID,
        SINGLE_PAULI_ADAPTER_ID,
        GlobalSinglePauliWordCandidateAdapter,
        H2OLinearFDSectorCompletePauliBlockCandidateAdapter,
        H2OLinearFDSinglePauliWordCandidateAdapter,
        H2OLinearFDSymmetryCompleteCandidateAdapter,
        MacroCandidateAdapter,
        SinglePauliWordCandidateAdapter,
    )

    payload = _mapping(value, name="request.adapter")
    representation = payload.get("candidate_representation_id")
    adapter_id = payload.get("adapter_id")
    if representation == CANDIDATE_REPRESENTATION_MACRO:
        if adapter_id == H2O_LINEAR_FD_SECTOR_COMPLETE_PAULI_BLOCK_ADAPTER_ID:
            return H2OLinearFDSectorCompletePauliBlockCandidateAdapter()
        if adapter_id == H2O_LINEAR_FD_SYMMETRY_COMPLETE_ADAPTER_ID:
            return H2OLinearFDSymmetryCompleteCandidateAdapter()
        if adapter_id not in {None, MACRO_ADAPTER_ID}:
            raise ValueError("Macro adapter id drifted.")
        return MacroCandidateAdapter()
    if representation == CANDIDATE_REPRESENTATION_SINGLE_PAULI:
        if adapter_id == H2O_LINEAR_FD_SINGLE_PAULI_ADAPTER_ID:
            return H2OLinearFDSinglePauliWordCandidateAdapter()
        if adapter_id == GLOBAL_SINGLE_PAULI_ADAPTER_ID:
            return GlobalSinglePauliWordCandidateAdapter()
        if adapter_id not in {None, SINGLE_PAULI_ADAPTER_ID}:
            raise ValueError("Single-Pauli adapter id drifted.")
        return SinglePauliWordCandidateAdapter()
    raise ValueError("Unknown serialized candidate adapter.")


def _policy_from_kind(
    value: Any,
    *,
    name: str,
    constructors: Mapping[str, Any],
) -> Any:
    payload = _mapping(value, name=name)
    kind = str(payload.get("kind", ""))
    constructor = constructors.get(kind)
    if constructor is None:
        raise ValueError(f"Unknown {name} kind {kind!r}.")
    kwargs = _without_kind(payload)
    if (
        constructor is CombinatorialBatchAdmission
        and "search_window_size" in kwargs
        and kwargs["search_window_size"] is None
    ):
        kwargs["search_window_size"] = FullCombinatorialSearchWindow()
    return constructor(**kwargs)


def _method_from_mapping(value: Any) -> SRMethodPolicy:
    payload = _mapping(value, name="request.method")
    return SRMethodPolicy(
        admission=_policy_from_kind(
            payload.get("admission"),
            name="admission policy",
            constructors={
                SingletonAdmission.kind: SingletonAdmission,
                GreedyBatchAdmission.kind: GreedyBatchAdmission,
                CombinatorialBatchAdmission.kind: (
                    CombinatorialBatchAdmission
                ),
            },
        ),
        insertion=_policy_from_kind(
            payload.get("insertion"),
            name="insertion policy",
            constructors={
                PlateauCommutationInsertion.kind: (
                    PlateauCommutationInsertion
                ),
                AppendOnlyInsertion.kind: AppendOnlyInsertion,
                AppendCommutationReducedInsertion.kind: (
                    AppendCommutationReducedInsertion
                ),
                AlwaysCommutationReducedInsertion.kind: (
                    AlwaysCommutationReducedInsertion
                ),
            },
        ),
        pruning=_policy_from_kind(
            payload.get("pruning"),
            name="pruning policy",
            constructors={
                PruningOff.kind: PruningOff,
                MetricPruning.kind: MetricPruning,
                TrustRegionPruning.kind: TrustRegionPruning,
                RecoverabilityPruning.kind: RecoverabilityPruning,
            },
        ),
        beam=_policy_from_kind(
            payload.get("beam"),
            name="beam policy",
            constructors={
                BeamOff.kind: BeamOff,
                ForkLocalBeam.kind: ForkLocalBeam,
            },
        ),
        trust_update=(
            None
            if payload.get("trust_update") is None
            else _policy_from_kind(
                payload.get("trust_update"),
                name="trust-update policy",
                constructors={
                    EndpointOverlapDisplacementTrust.kind: (
                        EndpointOverlapDisplacementTrust
                    ),
                },
            )
        ),
    )


def _stop_from_mapping(value: Any) -> SRStopPolicy:
    payload = dict(_mapping(value, name="execution.stop"))
    exact = payload.get("exact_ed_target")
    if exact is not None:
        exact_payload = dict(_mapping(exact, name="exact_ed_target"))
        source = ExactEDSourceReceipt(
            **dict(
                _mapping(
                    exact_payload.pop("source"),
                    name="exact_ed_target.source",
                )
            )
        )
        payload["exact_ed_target"] = ExactEDStop(
            source=source,
            **exact_payload,
        )
    return SRStopPolicy(**payload)


def _execution_from_mapping(value: Any) -> SRExecutionPolicy:
    payload = _mapping(value, name="request.execution")
    resume_payload = _mapping(payload.get("resume"), name="execution.resume")
    resume = _policy_from_kind(
        resume_payload,
        name="resume policy",
        constructors={
            FreshStart.kind: FreshStart,
            AcceptedStateResume.kind: AcceptedStateResume,
        },
    )
    return SRExecutionPolicy(
        stop=_stop_from_mapping(payload.get("stop")),
        resume=resume,
    )


def _observation_from_mapping(value: Any) -> SRObservationPolicy:
    payload = _mapping(value, name="request.observation")
    checkpoint_payload = payload.get("checkpoint")
    ledger_payload = payload.get("estimator_ledger")
    return SRObservationPolicy(
        checkpoint=(
            None
            if checkpoint_payload is None
            else CheckpointObservation(
                **dict(
                    _mapping(
                        checkpoint_payload,
                        name="observation.checkpoint",
                    )
                )
            )
        ),
        estimator_ledger=(
            None
            if ledger_payload is None
            else EstimatorLedgerObservation(
                **dict(
                    _mapping(
                        ledger_payload,
                        name="observation.estimator_ledger",
                    )
                )
            )
        ),
        resource_rounds=(
            None
            if payload.get("resource_rounds") is None
            else tuple(payload["resource_rounds"])
        ),
    )


def ra_adapt_request_from_mapping(value: Any) -> RAAdaptRequest:
    """Rehydrate a canonical RA request without inferring policy classes."""

    payload = _mapping(value, name="RAAdaptRequest")
    if payload.get("kind") not in {None, RAAdaptRequest.kind}:
        raise ValueError("Serialized request is not an RAAdaptRequest.")
    return RAAdaptRequest(
        adapter=_adapter_from_mapping(payload.get("adapter")),
        method=_method_from_mapping(payload.get("method")),
        execution=_execution_from_mapping(payload.get("execution")),
        observation=_observation_from_mapping(payload.get("observation")),
    )


def append_adapt_request_from_mapping(value: Any) -> AppendAdaptRequest:
    """Rehydrate a canonical Append request without policy guessing."""

    payload = _mapping(value, name="AppendAdaptRequest")
    if payload.get("kind") not in {None, AppendAdaptRequest.kind}:
        raise ValueError("Serialized request is not an AppendAdaptRequest.")
    return AppendAdaptRequest(
        adapter=_adapter_from_mapping(payload.get("adapter")),
        execution=_execution_from_mapping(payload.get("execution")),
        observation=_observation_from_mapping(payload.get("observation")),
    )


def resolved_ra_adapt_protocol_from_mapping(
    value: Any,
) -> ResolvedRAAdaptProtocol:
    """Verify and rehydrate one materialized RA/Append protocol."""

    payload = dict(_mapping(value, name="ResolvedRAAdaptProtocol"))
    schema = payload.get("schema")
    request_payload = payload.pop("request")
    request = (
        ra_adapt_request_from_mapping(request_payload)
        if schema in RA_ADAPT_PROTOCOL_SCHEMAS
        else append_adapt_request_from_mapping(request_payload)
        if schema == APPEND_ADAPT_PROTOCOL_SCHEMA
        else None
    )
    if request is None:
        raise ValueError("Unknown resolved protocol schema.")
    payload["request"] = request
    problem_payload = dict(
        _mapping(payload["problem"], name="protocol.problem")
    )
    problem_payload.setdefault("n_fermions", None)
    payload["problem"] = ResolvedProblemReceipt(**problem_payload)
    for name in ("parent_inventory", "executable_pool"):
        inventory = dict(_mapping(payload[name], name=f"protocol.{name}"))
        inventory["ordered_labels"] = tuple(inventory["ordered_labels"])
        inventory["removed_labels"] = tuple(
            inventory.get("removed_labels", ())
        )
        payload[name] = PoolInventoryReceipt(**inventory)
    payload["phase3_multiplier_contract"] = PhaseIIIMultiplierContract(
        **dict(
            _mapping(
                payload["phase3_multiplier_contract"],
                name="protocol.phase3_multiplier_contract",
            )
        )
    )
    raw_materialization = payload.get("bundle_materialization")
    if raw_materialization is not None:
        payload["bundle_materialization"] = (
            BundleProtocolMaterializationReceipt(
                **dict(
                    _mapping(
                        raw_materialization,
                        name="protocol.bundle_materialization",
                    )
                )
            )
        )
    return ResolvedRAAdaptProtocol(**payload)


def load_resolved_ra_adapt_protocol(
    path: str | Path,
) -> ResolvedRAAdaptProtocol:
    """Load, digest-check, and rehydrate one canonical protocol JSON file."""

    with Path(path).open("r", encoding="utf-8") as stream:
        return resolved_ra_adapt_protocol_from_mapping(json.load(stream))


def _validate_numerical_physical_integrity(
    receipt: NumericalPhysicalIntegrityReceipt,
    *,
    scientific_receipts: Mapping[str, Any],
    expected_method: str,
) -> None:
    if (
        not isinstance(receipt, NumericalPhysicalIntegrityReceipt)
        or receipt.schema != NUMERICAL_PHYSICAL_INTEGRITY_SCHEMA
        or receipt.method != expected_method
    ):
        raise TypeError(
            "Result requires the typed method-aligned numerical/physical "
            "integrity receipt."
        )
    payload = receipt.to_dict()
    if (
        scientific_receipts.get(
            "numerical_physical_integrity"
        )
        != payload
        or scientific_receipts.get(
            "numerical_physical_integrity_sha256"
        )
        != canonical_sha256(receipt)
    ):
        raise ValueError(
            "Scientific receipts must authenticate numerical/physical "
            "integrity evidence."
        )


@dataclass(frozen=True)
class RAAdaptResult(CanonicalContract):
    schema: str
    protocol: ResolvedRAAdaptProtocol
    selector_identity: str
    parent_inventory: PoolInventoryReceipt
    executable_pool: PoolInventoryReceipt
    policy: PolicyEchoReceipt
    run: SRRunResult
    numerical_physical_integrity: NumericalPhysicalIntegrityReceipt
    scientific_receipts: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected_schema = (
            RA_ADAPT_RESULT_SCHEMA_V2
            if self.protocol.algorithm_id
            == RA_ADAPT_NONSTATIONARY_INCREMENTAL_FULL_RESPONSE_ALGORITHM_ID
            else RA_ADAPT_RESULT_SCHEMA_V1
        )
        if self.schema != expected_schema:
            raise ValueError("Unknown RA result schema.")
        if self.selector_identity != RA_STAGED_SELECTOR_ID:
            raise ValueError("RA result selector identity drifted.")
        if (
            self.policy.active_gradient_policy
            != self.protocol.active_gradient_policy
            or self.policy.resource_weighting_scope
            != self.protocol.resource_weighting_scope
        ):
            raise ValueError("RA result policy echo drifted from its protocol.")
        if (
            self.protocol.algorithm_id
            == RA_ADAPT_NONSTATIONARY_INCREMENTAL_FULL_RESPONSE_ALGORITHM_ID
        ):
            expected_scientific_policies = {
                "phase3_candidate_gain_policy": (
                    RA_ADAPT_INCREMENTAL_CANDIDATE_GAIN_POLICY
                ),
                "accepted_refit_initialization_policy": (
                    RA_ADAPT_FULL_RESPONSE_REFIT_INITIALIZATION_POLICY
                ),
            }
            if (
                self.policy.phase3_candidate_gain_policy
                != RA_ADAPT_INCREMENTAL_CANDIDATE_GAIN_POLICY
                or self.policy.accepted_refit_initialization_policy
                != RA_ADAPT_FULL_RESPONSE_REFIT_INITIALIZATION_POLICY
                or any(
                    self.scientific_receipts.get(key) != value
                    for key, value in expected_scientific_policies.items()
                )
            ):
                raise ValueError(
                    "Canonical RA v2 result policies drifted from its "
                    "protocol identity."
                )
            for replay in self.run.scientific_replay:
                accepted_refit = replay.accepted_refit
                status = accepted_refit.initialization_status
                guard_nfev = accepted_refit.initialization_guard_nfev
                if (
                    accepted_refit.initialization_policy
                    != RA_ADAPT_FULL_RESPONSE_REFIT_INITIALIZATION_POLICY
                    or status not in {"accepted", "rejected", "unavailable"}
                    or (
                        status in {"accepted", "rejected"}
                        and guard_nfev != 1
                    )
                    or (status == "unavailable" and guard_nfev != 0)
                ):
                    raise ValueError(
                        "Canonical RA v2 typed accepted-refit initialization "
                        "receipt drifted from its protocol identity."
                    )
        _validate_numerical_physical_integrity(
            self.numerical_physical_integrity,
            scientific_receipts=self.scientific_receipts,
            expected_method="ra_adapt",
        )

    @property
    def final_state(self) -> Any:
        return self.run.final_state

    @property
    def accepted_trajectory(self) -> Any:
        return self.run.accepted_trajectory

    @property
    def accepted_transitions(self) -> Any:
        return self.run.accepted_transitions

    @property
    def estimator_accounting(self) -> Any:
        return self.run.estimator_accounting


@dataclass(frozen=True)
class AppendAdaptResult(CanonicalContract):
    schema: str
    protocol: ResolvedRAAdaptProtocol
    selector_identity: str
    parent_inventory: PoolInventoryReceipt
    executable_pool: PoolInventoryReceipt
    policy: PolicyEchoReceipt
    result_payload: Mapping[str, Any]
    paper_i_summary: Any
    numerical_physical_integrity: NumericalPhysicalIntegrityReceipt
    scientific_receipts: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema != APPEND_ADAPT_RESULT_SCHEMA:
            raise ValueError("Unknown Append result schema.")
        if self.selector_identity != APPEND_CONVENTIONAL_SELECTOR_ID:
            raise ValueError("Append result selector identity drifted.")
        if not isinstance(self.result_payload, Mapping):
            raise TypeError("Append result payload must be a mapping.")
        if not isinstance(self.scientific_receipts, Mapping):
            raise TypeError("Append scientific receipts must be a mapping.")
        _validate_numerical_physical_integrity(
            self.numerical_physical_integrity,
            scientific_receipts=self.scientific_receipts,
            expected_method="append_adapt",
        )
        if self.result_payload.get(
            "numerical_physical_integrity"
        ) != self.numerical_physical_integrity.to_dict():
            raise ValueError(
                "Append result payload must carry the authenticated "
                "numerical/physical integrity receipt."
            )
        if (
            not isinstance(self.paper_i_summary, CanonicalContract)
            or not is_dataclass(self.paper_i_summary)
            or getattr(self.paper_i_summary, "schema", None)
            != "paper_i_append_run_summary_v1"
        ):
            raise ValueError(
                "paper_i_summary must be the typed canonical Paper-I "
                "Append run summary."
            )
        if (
            getattr(self.paper_i_summary, "protocol_sha256", None)
            != self.protocol.sha256
            or getattr(
                self.paper_i_summary,
                "source_result_payload_sha256",
                None,
            )
            != canonical_sha256(self.result_payload)
            or int(
                getattr(
                    self.paper_i_summary,
                    "available_controller_rounds",
                    -1,
                )
            )
            != int(
                self.result_payload.get(
                    "controller_rounds_completed",
                    -1,
                )
            )
        ):
            raise ValueError(
                "paper_i_summary is not aligned with the completed Append "
                "result."
            )
        summary_payload = self.paper_i_summary.to_dict()
        if (
            self.scientific_receipts.get(
                "paper_i_append_run_summary"
            )
            != summary_payload
            or self.scientific_receipts.get(
                "paper_i_append_run_summary_sha256"
            )
            != canonical_sha256(self.paper_i_summary)
        ):
            raise ValueError(
                "Append scientific receipts must authenticate the canonical "
                "run summary."
            )


__all__ = [
    "ACTIVE_GRADIENT_MEASURED",
    "ACTIVE_GRADIENT_POLICIES",
    "ACTIVE_GRADIENT_STATIONARY",
    "APPEND_ADAPT_PROTOCOL_SCHEMA",
    "APPEND_ADAPT_RESULT_SCHEMA",
    "APPEND_CONVENTIONAL_SELECTOR_ID",
    "APPEND_CONVENTIONAL_SELECTOR_SCOPE",
    "AppendAdaptRequest",
    "AppendAdaptResult",
    "BUNDLE_PROTOCOL_MATERIALIZATION_SCHEMA",
    "BUNDLE_PROTOCOL_MATERIALIZATION_SCHEMA_V1",
    "BUNDLE_PROTOCOL_MATERIALIZATION_SCHEMA_V2",
    "BundleProtocolMaterializationAuthority",
    "BundleProtocolMaterializationReceipt",
    "CANDIDATE_REPRESENTATION_MACRO",
    "CANDIDATE_REPRESENTATION_SINGLE_PAULI",
    "CANDIDATE_INVENTORY_LINEAGE_SCHEMA",
    "CANDIDATE_LINEAGE_SCHEMA",
    "CandidateInventoryLineageReceipt",
    "CandidateInventoryLineageRow",
    "CandidateLineageReceipt",
    "EXACT_ORDERED_INSERTION_CHART",
    "ENDPOINT_OVERLAP_DISPLACEMENT_TRUST",
    "EXPANDED_RUNTIME_PROJECTED_LOGICAL_BASE_CHART",
    "FULL_ENLARGED_ACCEPTED_REFIT",
    "LOGICAL_SHARED_REDUCED_REFIT_BASE_CHART",
    "NATIVE_REFIT_CHART",
    "NUMERICAL_PHYSICAL_INTEGRITY_SCHEMA",
    "NumericalPhysicalIntegrityReceipt",
    "PhaseIIIMultiplierContract",
    "PhaseIIIStabilizationReceipt",
    "PolicyEchoReceipt",
    "PoolInventoryReceipt",
    "PROJECTED_GENERALIZED_SOLVER",
    "RA_ADAPT_PROTOCOL_SCHEMA",
    "RA_ADAPT_PROTOCOL_SCHEMAS",
    "RA_ADAPT_PROTOCOL_SCHEMA_V1",
    "RA_ADAPT_PROTOCOL_SCHEMA_V2",
    "RA_ADAPT_RESULT_SCHEMA",
    "RA_ADAPT_RESULT_SCHEMA_V1",
    "RA_ADAPT_RESULT_SCHEMA_V2",
    "RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V1",
    "RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V2",
    "RA_ADAPT_SINGLETON_LATCHED_PHASE3_ALGORITHM_ID",
    "RA_ADAPT_SINGLETON_PHASE3_PLATEAU_ALGORITHM_ID",
    "RA_ADAPT_POST_LATCH_INSERTION_TRIGGER_SCOPE",
    "RA_ADAPT_PHASE3_POPULATION_ALL_ROUNDS",
    "RA_ADAPT_PHASE3_POPULATION_LATCHED_ON_PROGRESS_PLATEAU",
    "RA_ADAPT_PHASE3_POPULATION_ON_INSERTION_PLATEAU",
    "RA_ADAPT_PHASE3_PREPLATEAU_WINNER_MATERIALIZATION",
    "RA_STAGED_SELECTOR_ID",
    "RAAdaptOperationalControls",
    "RAAdaptRequest",
    "RAAdaptResult",
    "RESOURCE_WEIGHTING_ALL_PHASE",
    "RESOURCE_WEIGHTING_LATE",
    "RESOURCE_WEIGHTING_SCOPES",
    "ResolvedRAAdaptProtocol",
    "SOURCE_GRAM_NO_OVERLAP_TRUST",
    "SUPPORTED_FS_WHITENED_REFIT_CHART",
    "append_adapt_request_from_mapping",
    "bundle_protocol_materialization_receipt",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_resolved_ra_adapt_protocol",
    "ra_adapt_request_from_mapping",
    "resolved_ra_adapt_protocol_from_mapping",
    "require_protocol_materialization_authority",
]
