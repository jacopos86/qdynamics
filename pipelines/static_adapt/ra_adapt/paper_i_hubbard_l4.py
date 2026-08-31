"""Named Paper-I half-filled Hubbard ``L=4`` five-arm application.

This seam admits one exact physics point and one matched candidate supply for
conventional Append-ADAPT and the RA baseline/prune/batch/beam arms.  It does
not widen the ordinary Paper-I facades or permit checkpoint splicing from an
older route contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, ClassVar, Literal, Mapping

from pipelines.contracts.problem import ProblemRequest, ResolvedProblemContext
from pipelines.scaffold.hh_continuation_generators import (
    serialize_polynomial_terms_exyz,
)
from pipelines.static_adapt.builders.problem_registry import (
    resolve_problem_context,
)
from pipelines.static_adapt.ra_adapt.adapters import (
    MACRO_GRADIENT_PHASE0_ADAPTER_ID,
    MacroGradientPhase0CandidateAdapter,
    PHASE_I_VISIBILITY_ALL_EXECUTABLE,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    ACTIVE_GRADIENT_MEASURED,
    ACTIVE_GRADIENT_STATIONARY,
    APPEND_ADAPT_PROTOCOL_SCHEMA,
    APPEND_CONVENTIONAL_SELECTOR_ID,
    AppendAdaptRequest,
    CANDIDATE_REPRESENTATION_MACRO,
    RA_ADAPT_PROTOCOL_SCHEMA,
    RA_STAGED_SELECTOR_ID,
    RAAdaptRequest,
    RESOURCE_WEIGHTING_ALL_PHASE,
    ResolvedRAAdaptProtocol,
    _attach_validated_bundle_protocol_authority,
    _mint_bundle_protocol_materialization_authority,
    bundle_protocol_materialization_receipt,
    canonical_sha256,
)
from pipelines.static_adapt.ra_adapt.pools import (
    PAPER_I_HUBBARD_L4_APPLICATION_ID,
    build_paper_i_hubbard_l4_executable_macro_pool,
    build_paper_i_hubbard_l4_parent_template_inventory,
    require_paper_i_hubbard_l4_problem,
)
from pipelines.scaffold.hh_continuation_scoring import (
    PHASE3_ZERO_CENTERED_SIGNED_FACTOR_CONSUMER_SEMANTIC_VERSION,
)
from pipelines.static_adapt.sr_snake.contracts import (
    BeamOff,
    CheckpointObservation,
    CombinatorialBatchAdmission,
    EstimatorLedgerObservation,
    ExactEDSourceReceipt,
    FreshStart,
    ForkLocalBeam,
    FullCombinatorialSearchWindow,
    MetricPruning,
    PlateauCommutationInsertion,
    PruningOff,
    SRExecutionPolicy,
    SRMethodPolicy,
    SRObservationPolicy,
    SRStopPolicy,
    SingletonAdmission,
)


PAPER_I_HUBBARD_L4_ADAPTER_ID = (
    "paper_i_hubbard_l4_macro_gradient_phase0_candidate_adapter_v1"
)
PAPER_I_HUBBARD_L4_ALGORITHM_ID = (
    "paper_i_ra_adapt_macro_gradient_phase0_macro_phase1_phase2_phase3_"
    "qiskit_phase2_phase3_plateau_no_lanes_v1"
)
PAPER_I_HUBBARD_L4_SOURCE_LOCK_KEY = (
    "paper_i_hubbard_l4_application_source_sha256"
)
PAPER_I_HUBBARD_L4_EXACT_SOURCE_ID = (
    "paper_i_hubbard_l4_half_filled_open_same_sector_ed_v1"
)
PAPER_I_HUBBARD_L4_EXACT_ENERGY = -3.3827681604170525
PAPER_I_HUBBARD_L4_HORIZON = 50
PAPER_I_HUBBARD_L4_BUNDLE_ID = "paper_i_hubbard_l4_five_arm_v1"

PaperIHubbardL4Arm = Literal["baseline", "prune", "batch", "beam"]


@dataclass(frozen=True)
class PaperIHubbardL4MacroGradientPhase0CandidateAdapter(
    MacroGradientPhase0CandidateAdapter
):
    """Keep the locked L=4 full-meta macros intact through all RA phases."""

    application_family_key: ClassVar[str] = "hubbard"
    application_id: ClassVar[str] = PAPER_I_HUBBARD_L4_APPLICATION_ID
    phase_i_candidate_visibility_id: ClassVar[str] = (
        PHASE_I_VISIBILITY_ALL_EXECUTABLE
    )
    adapter_id: str = PAPER_I_HUBBARD_L4_ADAPTER_ID

    def __post_init__(self) -> None:
        if (
            self.candidate_representation_id
            != CANDIDATE_REPRESENTATION_MACRO
            or self.adapter_id != PAPER_I_HUBBARD_L4_ADAPTER_ID
            or self.macro_phase0_policy_id
            != "standard_adapt_abs_gradient_macro_phase0_v1"
        ):
            raise ValueError("The named Hubbard L=4 adapter identity is fixed.")

    def parent_inventory(self, problem: ResolvedProblemContext) -> Any:
        return build_paper_i_hubbard_l4_parent_template_inventory(
            problem,
            representation_id=self.candidate_representation_id,
        )

    def executable_pool(self, problem: ResolvedProblemContext) -> Any:
        return build_paper_i_hubbard_l4_executable_macro_pool(problem)


def build_paper_i_hubbard_l4_problem() -> ResolvedProblemContext:
    """Construct the exact open, blocked, half-filled ``L=4`` Hubbard point."""

    problem = resolve_problem_context(
        ProblemRequest(
            problem_key="hubbard",
            num_sites=4,
            t=1.0,
            u=1.25,
            dv=0.0,
            omega0=0.0,
            g_ep=0.0,
            n_ph_max=0,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
            include_zero_point=False,
            v_nn=0.0,
            t_prime=0.0,
            n_fermions=4,
        )
    )
    require_paper_i_hubbard_l4_problem(problem)
    observed = float(problem.exact_target.resolve_energy())
    if not math.isclose(
        observed,
        PAPER_I_HUBBARD_L4_EXACT_ENERGY,
        rel_tol=0.0,
        abs_tol=5.0e-13,
    ):
        raise RuntimeError(
            "The named Hubbard L=4 exact energy drifted: "
            f"{observed} != {PAPER_I_HUBBARD_L4_EXACT_ENERGY}."
        )
    return problem


def _observation_policy(
    output_dir: Path | None,
    *,
    horizon: int,
    checkpoint_enabled: bool = True,
) -> SRObservationPolicy:
    if output_dir is None:
        return SRObservationPolicy()
    root = Path(output_dir)
    return SRObservationPolicy(
        checkpoint=(
            CheckpointObservation(
                path=root / "current.json",
                every_controller_rounds=1,
                keep_history_tail=max(100, int(horizon)),
            )
            if checkpoint_enabled
            else None
        ),
        estimator_ledger=EstimatorLedgerObservation(
            path=root / "estimator_ledger.json"
        ),
    )


def build_paper_i_hubbard_l4_request(
    arm: PaperIHubbardL4Arm,
    *,
    maximum_controller_rounds: int = PAPER_I_HUBBARD_L4_HORIZON,
    output_dir: Path | None = None,
) -> RAAdaptRequest:
    """Build one fresh, source-matchable RA arm request."""

    arm_id = str(arm).strip().lower()
    horizon = int(maximum_controller_rounds)
    if horizon < 1:
        raise ValueError("maximum_controller_rounds must be positive.")
    if arm_id not in {"baseline", "prune", "batch", "beam"}:
        raise ValueError("arm must be baseline, prune, batch, or beam.")

    admission = SingletonAdmission()
    pruning = PruningOff()
    beam = BeamOff()
    if arm_id == "prune":
        pruning = MetricPruning(small_theta_abs=1.0e-10)
    elif arm_id == "batch":
        admission = CombinatorialBatchAdmission(
            maximum_size=3,
            search_window_size=FullCombinatorialSearchWindow(),
        )
    elif arm_id == "beam":
        beam = ForkLocalBeam(
            live_parent_branches=3,
            admission_children_per_parent=2,
            maximum_admission_children_per_round=6,
            s_alg_weight=0.005,
        )
    return RAAdaptRequest(
        adapter=PaperIHubbardL4MacroGradientPhase0CandidateAdapter(),
        method=SRMethodPolicy(
            admission=admission,
            insertion=PlateauCommutationInsertion(),
            pruning=pruning,
            beam=beam,
        ),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=horizon),
            resume=FreshStart(),
        ),
        observation=_observation_policy(
            output_dir,
            horizon=horizon,
            checkpoint_enabled=(arm_id != "batch"),
        ),
    )


def build_paper_i_hubbard_l4_append_request(
    *,
    maximum_controller_rounds: int = PAPER_I_HUBBARD_L4_HORIZON,
    output_dir: Path | None = None,
) -> AppendAdaptRequest:
    """Build the matched conventional Append-ADAPT comparator request."""

    horizon = int(maximum_controller_rounds)
    if horizon < 1:
        raise ValueError("maximum_controller_rounds must be positive.")
    return AppendAdaptRequest(
        adapter=PaperIHubbardL4MacroGradientPhase0CandidateAdapter(),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=horizon),
            resume=FreshStart(),
        ),
        observation=_observation_policy(output_dir, horizon=horizon),
    )


def is_paper_i_hubbard_l4_ra_application(
    problem: ResolvedProblemContext | None,
    request: RAAdaptRequest,
) -> bool:
    if not isinstance(
        request.adapter,
        PaperIHubbardL4MacroGradientPhase0CandidateAdapter,
    ):
        return False
    if problem is None:
        raise ValueError("The named Hubbard L=4 adapter requires its problem.")
    require_paper_i_hubbard_l4_problem(problem)
    return True


def is_paper_i_hubbard_l4_append_application(
    problem: ResolvedProblemContext | None,
    request: AppendAdaptRequest,
) -> bool:
    if not isinstance(
        request.adapter,
        PaperIHubbardL4MacroGradientPhase0CandidateAdapter,
    ):
        return False
    if problem is None:
        raise ValueError("The named Hubbard L=4 adapter requires its problem.")
    require_paper_i_hubbard_l4_problem(problem)
    return True


def paper_i_hubbard_l4_application_source_contract(
    problem: ResolvedProblemContext,
) -> dict[str, Any]:
    """Bind the exact problem, macro supply, and reporting-only ED source."""

    require_paper_i_hubbard_l4_problem(problem)
    parent = build_paper_i_hubbard_l4_parent_template_inventory(
        problem,
        representation_id=CANDIDATE_REPRESENTATION_MACRO,
    )
    executable = build_paper_i_hubbard_l4_executable_macro_pool(problem)
    exact_source = ExactEDSourceReceipt.from_problem(
        problem,
        source_id=PAPER_I_HUBBARD_L4_EXACT_SOURCE_ID,
    )
    payload: dict[str, Any] = {
        "schema": "paper_i_hubbard_l4_application_source_contract_v1",
        "application_id": PAPER_I_HUBBARD_L4_APPLICATION_ID,
        "algorithm_id": PAPER_I_HUBBARD_L4_ALGORITHM_ID,
        "adapter_id": PAPER_I_HUBBARD_L4_ADAPTER_ID,
        "problem_request_sha256": exact_source.problem_request_sha256,
        "hamiltonian_terms_sha256": canonical_sha256(
            serialize_polynomial_terms_exyz(problem.hamiltonian)
        ),
        "sector_num_particles": [2, 2],
        "parent_inventory": parent.receipt.to_dict(),
        "executable_macro_inventory": executable.receipt.to_dict(),
        "same_sector_exact_reference": {
            **exact_source.to_dict(),
            "receipt_sha256": canonical_sha256(exact_source.to_dict()),
            "energy": PAPER_I_HUBBARD_L4_EXACT_ENERGY,
            "controller_input": False,
        },
        "scientific_settings": {
            "maximum_controller_rounds": PAPER_I_HUBBARD_L4_HORIZON,
            "optimizer": "powell",
            "optimizer_maxiter": 200,
            "optimizer_xtol": 1.0e-5,
            "optimizer_ftol": 1.0e-12,
            "seed": 7,
            "active_gradient_policy": ACTIVE_GRADIENT_STATIONARY,
            "resource_weighting_scope": RESOURCE_WEIGHTING_ALL_PHASE,
            "phase0_shortlist_size": 24,
            "phase0_cost": "gradient_only_no_metric_no_resource_v1",
            "phase1_cost": "structural_proxy_v1",
            "phase2_phase3_cost": (
                "qiskit_full_trial_ansatz_signed_marginal_no_lanes_v1"
            ),
            "accepted_refit": "fixed_supported_fs_whitened_chart_v1",
            "phase3_signed_factor_consumer_semantic_version": (
                PHASE3_ZERO_CENTERED_SIGNED_FACTOR_CONSUMER_SEMANTIC_VERSION
            ),
        },
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def _materialization_refs(
    problem: ResolvedProblemContext,
    *,
    request: RAAdaptRequest | AppendAdaptRequest,
    arm: str,
) -> dict[str, str]:
    source = paper_i_hubbard_l4_application_source_contract(problem)
    source_lock_id = f"paper_i_hubbard_l4__{arm}__source_v1"
    cell_source_lock_sha256 = canonical_sha256(
        {
            "schema": "paper_i_hubbard_l4_cell_source_lock_v1",
            "source_lock_id": source_lock_id,
            "application_source_sha256": source["sha256"],
            "request_sha256": canonical_sha256(request.to_dict()),
        }
    )
    source_locks_manifest_sha256 = canonical_sha256(
        {
            "schema": "paper_i_hubbard_l4_source_locks_v1",
            "application_source_sha256": source["sha256"],
            "cell_source_lock_sha256": cell_source_lock_sha256,
        }
    )
    return {
        "source_locks_manifest_sha256": source_locks_manifest_sha256,
        "cell_source_lock_id": source_lock_id,
        "cell_source_lock_sha256": cell_source_lock_sha256,
        PAPER_I_HUBBARD_L4_SOURCE_LOCK_KEY: str(source["sha256"]),
    }


def _materialize(
    problem: ResolvedProblemContext,
    request: RAAdaptRequest | AppendAdaptRequest,
) -> ResolvedRAAdaptProtocol:
    """Mint and bind the narrow in-memory execution authority."""

    from pipelines.static_adapt.ra_adapt.append import (
        APPEND_ADAPT_ALGORITHM_ID,
        build_resolved_append_protocol,
    )
    from pipelines.static_adapt.ra_adapt.engine import (
        RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ALGORITHM_ID,
        build_resolved_ra_protocol,
    )

    if isinstance(request, RAAdaptRequest):
        if not is_paper_i_hubbard_l4_ra_application(problem, request):
            raise ValueError("RA request is not the named Hubbard L=4 application.")
        algorithm_id = RA_ADAPT_MACRO_GRADIENT_PHASE0_MACRO_PHASE23_QISKIT_ALGORITHM_ID
        protocol_schema = RA_ADAPT_PROTOCOL_SCHEMA
        selector_identity = RA_STAGED_SELECTOR_ID
        active_gradient_policy = ACTIVE_GRADIENT_STATIONARY
        builder = build_resolved_ra_protocol
        arm = str(request.method.admission.kind)
        if not isinstance(request.method.pruning, PruningOff):
            arm = "prune"
        elif not isinstance(request.method.beam, BeamOff):
            arm = "beam"
        elif isinstance(request.method.admission, CombinatorialBatchAdmission):
            arm = "batch"
    else:
        if not is_paper_i_hubbard_l4_append_application(problem, request):
            raise ValueError("Append request is not the named Hubbard L=4 application.")
        algorithm_id = APPEND_ADAPT_ALGORITHM_ID
        protocol_schema = APPEND_ADAPT_PROTOCOL_SCHEMA
        selector_identity = APPEND_CONVENTIONAL_SELECTOR_ID
        active_gradient_policy = ACTIVE_GRADIENT_MEASURED
        builder = build_resolved_append_protocol
        arm = "avqe"
    refs = _materialization_refs(
        problem,
        request=request,
        arm=arm,
    )
    arm_payload = request.to_dict()
    bundle_manifest_sha256 = canonical_sha256(
        {
            "schema": "paper_i_hubbard_l4_native_materialization_v1",
            "application_source": refs,
            "request": arm_payload,
        }
    )
    source_locks_sha256 = refs["source_locks_manifest_sha256"]
    receipt = bundle_protocol_materialization_receipt(
        bundle_id=PAPER_I_HUBBARD_L4_BUNDLE_ID,
        bundle_manifest_sha256=bundle_manifest_sha256,
        source_locks_sha256=source_locks_sha256,
        source_lock_refs=refs,
        cell_id=f"paper_i_hubbard_l4__{arm}",
        source_lock_id=refs["cell_source_lock_id"],
        protocol_schema=protocol_schema,
        algorithm_id=algorithm_id,
        candidate_representation=CANDIDATE_REPRESENTATION_MACRO,
        selector_identity=selector_identity,
        active_gradient_policy=active_gradient_policy,
        resource_weighting_scope=RESOURCE_WEIGHTING_ALL_PHASE,
    )
    unbound = _mint_bundle_protocol_materialization_authority(
        receipt,
        source_lock_refs=refs,
    )
    protocol = builder(problem, request, materialization_authority=unbound)
    bound = _mint_bundle_protocol_materialization_authority(
        receipt,
        source_lock_refs=refs,
        protocol_sha256=protocol.sha256,
    )
    return _attach_validated_bundle_protocol_authority(protocol, bound)


def materialize_paper_i_hubbard_l4_ra_protocol(
    problem: ResolvedProblemContext,
    request: RAAdaptRequest,
) -> ResolvedRAAdaptProtocol:
    return _materialize(problem, request)


def materialize_paper_i_hubbard_l4_append_protocol(
    problem: ResolvedProblemContext,
    request: AppendAdaptRequest,
) -> ResolvedRAAdaptProtocol:
    return _materialize(problem, request)


def require_paper_i_hubbard_l4_materialization(
    *,
    problem: ResolvedProblemContext,
    request: RAAdaptRequest | AppendAdaptRequest,
    algorithm_id: str,
    active_gradient_policy: str,
    resource_weighting_scope: str,
    source_locks: Mapping[str, str],
) -> None:
    """Fail closed if a materialized L=4 protocol drifts from its source."""

    named = (
        is_paper_i_hubbard_l4_ra_application(problem, request)
        if isinstance(request, RAAdaptRequest)
        else is_paper_i_hubbard_l4_append_application(problem, request)
    )
    expected_source = paper_i_hubbard_l4_application_source_contract(problem)
    expected_algorithm = (
        PAPER_I_HUBBARD_L4_ALGORITHM_ID
        if isinstance(request, RAAdaptRequest)
        else "paper_i_append_adapt_v1"
    )
    expected_gradient = (
        ACTIVE_GRADIENT_STATIONARY
        if isinstance(request, RAAdaptRequest)
        else ACTIVE_GRADIENT_MEASURED
    )
    if (
        not named
        or str(algorithm_id) != expected_algorithm
        or str(active_gradient_policy) != expected_gradient
        or str(resource_weighting_scope) != RESOURCE_WEIGHTING_ALL_PHASE
        or source_locks.get(PAPER_I_HUBBARD_L4_SOURCE_LOCK_KEY)
        != expected_source["sha256"]
    ):
        raise ValueError(
            "The named Hubbard L=4 protocol lost its application source lock "
            "or matched scientific policy."
        )


__all__ = [
    "PAPER_I_HUBBARD_L4_ADAPTER_ID",
    "PAPER_I_HUBBARD_L4_ALGORITHM_ID",
    "PAPER_I_HUBBARD_L4_BUNDLE_ID",
    "PAPER_I_HUBBARD_L4_EXACT_ENERGY",
    "PAPER_I_HUBBARD_L4_HORIZON",
    "PAPER_I_HUBBARD_L4_SOURCE_LOCK_KEY",
    "PaperIHubbardL4MacroGradientPhase0CandidateAdapter",
    "build_paper_i_hubbard_l4_append_request",
    "build_paper_i_hubbard_l4_problem",
    "build_paper_i_hubbard_l4_request",
    "is_paper_i_hubbard_l4_append_application",
    "is_paper_i_hubbard_l4_ra_application",
    "materialize_paper_i_hubbard_l4_append_protocol",
    "materialize_paper_i_hubbard_l4_ra_protocol",
    "paper_i_hubbard_l4_application_source_contract",
    "require_paper_i_hubbard_l4_materialization",
    "require_paper_i_hubbard_l4_problem",
]
