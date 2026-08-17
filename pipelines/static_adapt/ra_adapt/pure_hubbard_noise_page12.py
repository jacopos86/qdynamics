"""Named pure-Hubbard full-noise application of the Page-12 RA route.

This is a deliberately narrow application seam.  It does not widen the
ordinary Paper-I Hubbard--Holstein facade, ordinary pool builders, or generic
Hubbard execution.  The only admitted physics points are the open, blocked,
half-filled ``L=2`` Hubbard model at ``U/t in {1.5, 8}``, and the only admitted
noise profiles are the three fixed Paper-I appendix rungs below.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, ClassVar, Mapping

from pipelines.contracts.problem import ProblemRequest, ResolvedProblemContext
from pipelines.exact_bench.noise_oracle_defaults import (
    SYNTHETIC_COHERENT_1Q_GATES_DEFAULT,
    SYNTHETIC_COHERENT_2Q_GATES_DEFAULT,
    SYNTHETIC_DEPOLARIZING_1Q_GATES_DEFAULT,
    SYNTHETIC_DEPOLARIZING_2Q_GATES_DEFAULT,
)
from pipelines.static_adapt.builders.problem_registry import resolve_problem_context
from pipelines.static_adapt.ra_adapt.adapters import (
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
    PAPER_I_PURE_HUBBARD_NOISE_PAGE12_APPLICATION_ID,
    build_paper_i_pure_hubbard_noise_page12_guarded_single_pauli_pool,
    build_paper_i_pure_hubbard_noise_page12_parent_template_inventory,
    require_paper_i_pure_hubbard_noise_page12_problem,
)
from pipelines.static_adapt.sr_snake.contracts import (
    AcceptedStateResume,
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


PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ADAPTER_ID = (
    "paper_i_pure_hubbard_noise_page12_global_singleton_gradient_phase0_"
    "candidate_adapter_v1"
)
PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ALGORITHM_ID = (
    "paper_i_ra_adapt_pure_hubbard_full_noise_global_singleton_gradient_"
    "phase0_phase1_phase2_phase3_qiskit_phase2_phase3_plateau_no_lanes_v1"
)
PAPER_I_PURE_HUBBARD_NOISE_PAGE12_SOURCE_LOCK_KEY = (
    "paper_i_pure_hubbard_noise_page12_application_source_sha256"
)
PAPER_I_PURE_HUBBARD_NOISE_PAGE12_EXACT_SOURCE_ID = (
    "paper_i_pure_hubbard_same_cutoff_sector_ed_v1"
)
PAPER_I_PURE_HUBBARD_NOISE_VALUE_SEED = 702688422
PAPER_I_PURE_HUBBARD_NOISE_COHERENT_SEED = 20260609
PAPER_I_PURE_HUBBARD_NOISE_GRADIENT_STEP = 0.1
PAPER_I_PURE_HUBBARD_NOISE_SIMULATOR_SEED = 7
PAPER_I_PURE_HUBBARD_NOISE_TRANSPILER_SEED = 7
_SAME_CUTOFF_EXACT_EVALUATION_POLICY_ID = (
    "runtime_same_cutoff_exact_diagnostic_full_precision_v1"
)
_SAME_CUTOFF_EXACT_FORMULA_ID = (
    "l2_open_half_filled_hubbard_ground_energy_v1"
)
_SAME_CUTOFF_EXACT_U_RATIONAL_BY_VALUE = {
    1.5: (3, 2),
    8.0: (8, 1),
}
_SAME_CUTOFF_EXACT_VALIDATION_ATOL = 1.0e-12

_NOISE_TUPLE_ORDER = ("sigma_E", "p1", "p2", "epsilon1", "epsilon2")
_NOISE_TUPLES: dict[str, tuple[float, float, float, float, float]] = {
    "low": (1.0e-6, 1.0e-8, 1.0e-7, 2.0e-4, 6.0e-4),
    "high": (
        7.071067811865475e-5,
        1.0e-6,
        1.0e-5,
        2.0e-3,
        6.0e-3,
    ),
    "extreme": (1.0e-2, 1.0e-3, 1.0e-2, 6.0e-2, 6.0e-2),
}


def pure_hubbard_noise_level_contract(level: str) -> dict[str, Any]:
    """Return one canonical fixed full-noise rung."""

    key = str(level).strip().lower()
    try:
        sigma_e, p1, p2, epsilon1, epsilon2 = _NOISE_TUPLES[key]
    except KeyError as exc:
        raise ValueError(
            "noise_level_id must be one of {'low', 'high', 'extreme'}."
        ) from exc
    payload: dict[str, Any] = {
        "schema": "paper_i_pure_hubbard_full_noise_level_v1",
        "noise_level_id": key,
        "noise_tuple_order": list(_NOISE_TUPLE_ORDER),
        "noise_tuple": [sigma_e, p1, p2, epsilon1, epsilon2],
        "value_noise": {
            "model": "gaussian_iid_v1",
            "std": sigma_e,
            "seed": PAPER_I_PURE_HUBBARD_NOISE_VALUE_SEED,
            "frozen_keyed": False,
            "semantic": (
                "post_expectation_value_noise_not_physical_shots"
            ),
            "std_source": "explicit_std",
            "physical_shots_unchanged": True,
            "fixed_gate_error_reduction_claimed": False,
        },
        "synthetic_depolarizing": {
            "one_qubit_error": p1,
            "two_qubit_error": p2,
            "one_qubit_gates": list(
                SYNTHETIC_DEPOLARIZING_1Q_GATES_DEFAULT
            ),
            "two_qubit_gates": list(
                SYNTHETIC_DEPOLARIZING_2Q_GATES_DEFAULT
            ),
        },
        "synthetic_coherent": {
            "one_qubit_angle_std": epsilon1,
            "two_qubit_angle_std": epsilon2,
            "generator_mode": "random_pauli_frozen_v1",
            "one_qubit_gates": list(
                SYNTHETIC_COHERENT_1Q_GATES_DEFAULT
            ),
            "two_qubit_gates": list(
                SYNTHETIC_COHERENT_2Q_GATES_DEFAULT
            ),
        },
        "synthetic_coherent_seed": PAPER_I_PURE_HUBBARD_NOISE_COHERENT_SEED,
        "oracle_noise_mode": "aer_density_matrix_synthetic_coherent",
        "execution_surface": "expectation_v1",
        "density_matrix_shotless": True,
        "optimizer_evaluation_order": "serial_v1",
        "gradient_step": PAPER_I_PURE_HUBBARD_NOISE_GRADIENT_STEP,
        # Bind the complete effective OracleConfig rather than inheriting
        # mutable defaults from the shared lifecycle.
        "effective_oracle_config": {
            "noise_mode": "aer_density_matrix_synthetic_coherent",
            "shots": 1,
            "seed": PAPER_I_PURE_HUBBARD_NOISE_SIMULATOR_SEED,
            "seed_transpiler": PAPER_I_PURE_HUBBARD_NOISE_TRANSPILER_SEED,
            "transpile_optimization_level": 1,
            "oracle_repeats": 1,
            "oracle_aggregate": "mean",
            "backend_name": None,
            "use_fake_backend": False,
            "approximation": False,
            "abelian_grouping": True,
            "allow_aer_fallback": True,
            "aer_fallback_mode": "sampler_shots",
            "omp_shm_workaround": True,
            "mitigation": {
                "mode": "none",
                "zne_scales": [],
                "dd_sequence": None,
                "local_readout_strategy": None,
            },
            "symmetry_mitigation": {
                "mode": "off",
                "ordering": "blocked",
                "sector_n_up": None,
                "num_sites": None,
                "sector_n_dn": None,
            },
            "runtime_profile": {
                "name": "legacy_runtime_v0",
                "resilience_level": None,
                "default_shots": None,
                "default_precision": None,
                "max_execution_time": None,
                "init_qubits": None,
                "measure_mitigation": None,
                "measure_twirling": None,
                "gate_twirling": None,
                "gate_twirling_scope": None,
                "twirling_strategy": None,
                "zne_mitigation": None,
                "zne_noise_factors": [],
                "zne_extrapolator": [],
                "pec_mitigation": None,
                "dd_enable": None,
                "dd_sequence": None,
            },
            "runtime_raw_profile": "legacy_runtime_v0",
            "runtime_session": {"mode": "prefer_session"},
            "execution_surface": "expectation_v1",
            "raw_transport": "auto",
            "raw_store_memory": False,
            "raw_grouping_mode": "qwc_basis_cover_reuse",
            "raw_artifact_path": None,
            "value_noise_model": "gaussian_iid_v1",
            "value_noise_std": sigma_e,
            "value_noise_seed": PAPER_I_PURE_HUBBARD_NOISE_VALUE_SEED,
            "value_noise_sigma0_abs": None,
            "value_noise_n_eff": None,
            "value_noise_semantic": (
                "post_expectation_value_noise_not_physical_shots"
            ),
            "value_noise_std_source": "explicit_std",
            "synthetic_depolarizing_1q_error": p1,
            "synthetic_depolarizing_2q_error": p2,
            "synthetic_depolarizing_1q_gates": list(
                SYNTHETIC_DEPOLARIZING_1Q_GATES_DEFAULT
            ),
            "synthetic_depolarizing_2q_gates": list(
                SYNTHETIC_DEPOLARIZING_2Q_GATES_DEFAULT
            ),
            "synthetic_coherent_1q_angle_std": epsilon1,
            "synthetic_coherent_2q_angle_std": epsilon2,
            "synthetic_coherent_seed": (
                PAPER_I_PURE_HUBBARD_NOISE_COHERENT_SEED
            ),
            "synthetic_coherent_generator_mode": (
                "random_pauli_frozen_v1"
            ),
            "synthetic_coherent_1q_gates": list(
                SYNTHETIC_COHERENT_1Q_GATES_DEFAULT
            ),
            "synthetic_coherent_2q_gates": list(
                SYNTHETIC_COHERENT_2Q_GATES_DEFAULT
            ),
        },
    }
    return {**payload, "sha256": canonical_sha256(payload)}


@dataclass(frozen=True)
class PaperIPureHubbardNoisePage12CandidateAdapter(
    GlobalSingletonGradientPhase0CandidateAdapter
):
    """Global-singleton Page-12 adapter for the named Hubbard noise study."""

    application_family_key: ClassVar[str] = "hubbard"
    application_id: ClassVar[str] = (
        PAPER_I_PURE_HUBBARD_NOISE_PAGE12_APPLICATION_ID
    )
    adapter_id: str = PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ADAPTER_ID
    noise_level_id: str = "low"

    def __post_init__(self) -> None:
        pure_hubbard_noise_level_contract(self.noise_level_id)
        if (
            self.candidate_representation_id
            != CANDIDATE_REPRESENTATION_SINGLE_PAULI
            or self.adapter_id
            != PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ADAPTER_ID
        ):
            raise ValueError(
                "The pure-Hubbard Page-12 noise adapter identity is fixed."
            )

    def parent_inventory(self, problem: ResolvedProblemContext) -> Any:
        return build_paper_i_pure_hubbard_noise_page12_parent_template_inventory(
            problem,
            representation_id=self.candidate_representation_id,
        )

    def executable_pool(self, problem: ResolvedProblemContext) -> Any:
        return build_paper_i_pure_hubbard_noise_page12_guarded_single_pauli_pool(
            problem
        )


def build_paper_i_pure_hubbard_noise_page12_problem(
    *,
    u: float,
) -> ResolvedProblemContext:
    """Build either of the two authorized pure-Hubbard physics points."""

    problem = resolve_problem_context(
        ProblemRequest(
            problem_key="hubbard",
            num_sites=2,
            t=1.0,
            u=float(u),
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
            n_fermions=2,
        )
    )
    require_paper_i_pure_hubbard_noise_page12_problem(problem)
    return problem


def build_paper_i_pure_hubbard_noise_page12_request(
    *,
    noise_level: str,
    maximum_controller_rounds: int,
    output_dir: Path | None = None,
    resume: FreshStart | AcceptedStateResume | None = None,
) -> RAAdaptRequest:
    """Build the named scientific request with a package-supplied horizon."""

    horizon = int(maximum_controller_rounds)
    if horizon < 1:
        raise ValueError("maximum_controller_rounds must be positive.")
    observation = SRObservationPolicy()
    if output_dir is not None:
        root = Path(output_dir)
        observation = SRObservationPolicy(
            checkpoint=CheckpointObservation(
                path=root / "current.json",
                every_controller_rounds=1,
                keep_history_tail=max(100, horizon),
            ),
            estimator_ledger=EstimatorLedgerObservation(
                path=root / "estimator_ledger.json"
            ),
        )
    return RAAdaptRequest(
        adapter=PaperIPureHubbardNoisePage12CandidateAdapter(
            noise_level_id=str(noise_level).strip().lower()
        ),
        method=SRMethodPolicy(
            admission=SingletonAdmission(),
            insertion=PlateauCommutationInsertion(),
            pruning=PruningOff(),
            beam=BeamOff(),
        ),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=horizon),
            resume=FreshStart() if resume is None else resume,
        ),
        observation=observation,
    )


def require_paper_i_pure_hubbard_noise_page12_request(
    request: RAAdaptRequest,
) -> None:
    """Reject every unnamed policy, noise, or execution variant."""

    if not isinstance(request, RAAdaptRequest):
        raise TypeError("request must be an RAAdaptRequest.")
    adapter = request.adapter
    if not (
        isinstance(adapter, PaperIPureHubbardNoisePage12CandidateAdapter)
        and type(adapter) is PaperIPureHubbardNoisePage12CandidateAdapter
        and adapter.adapter_id
        == PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ADAPTER_ID
        and isinstance(request.method.admission, SingletonAdmission)
        and isinstance(request.method.insertion, PlateauCommutationInsertion)
        and isinstance(request.method.pruning, PruningOff)
        and isinstance(request.method.beam, BeamOff)
        and isinstance(request.execution.resume, (FreshStart, AcceptedStateResume))
        and int(request.execution.stop.maximum_controller_rounds) >= 1
        and request.execution.stop.exact_ed_target is None
    ):
        raise ValueError(
            "The named pure-Hubbard noise seam requires global-singleton "
            "gradient Phase 0, singleton I/II/III, cumulative-relative "
            "plateau insertion, pruning off, beam off, and reporting-only ED."
        )
    pure_hubbard_noise_level_contract(adapter.noise_level_id)


def paper_i_pure_hubbard_noise_page12_application_source_contract(
    problem: ResolvedProblemContext,
    request: RAAdaptRequest,
) -> dict[str, Any]:
    """Bind physics, pool bytes, controller noise, and exact diagnostics."""

    require_paper_i_pure_hubbard_noise_page12_problem(problem)
    require_paper_i_pure_hubbard_noise_page12_request(request)
    adapter = request.adapter
    assert isinstance(adapter, PaperIPureHubbardNoisePage12CandidateAdapter)
    parent = adapter.parent_inventory(problem)
    singleton = adapter.executable_pool(problem)
    exact_source = ExactEDSourceReceipt.from_problem(
        problem,
        source_id=PAPER_I_PURE_HUBBARD_NOISE_PAGE12_EXACT_SOURCE_ID,
    )
    u_numerator, u_denominator = (
        _SAME_CUTOFF_EXACT_U_RATIONAL_BY_VALUE[
            float(problem.request.u)
        ]
    )
    t_numerator, t_denominator = 1, 1
    analytic_u = float(u_numerator) / float(u_denominator)
    analytic_t = float(t_numerator) / float(t_denominator)
    analytic_exact_energy = 0.5 * (
        analytic_u
        - math.sqrt(analytic_u * analytic_u + 16.0 * analytic_t * analytic_t)
    )
    observed_exact_energy = float(problem.exact_target.resolve_energy())
    if not math.isclose(
        observed_exact_energy,
        analytic_exact_energy,
        rel_tol=0.0,
        abs_tol=_SAME_CUTOFF_EXACT_VALIDATION_ATOL,
    ):
        raise ValueError(
            "The pure-Hubbard Page-12 same-cutoff ED reference drifted "
            "from its analytic problem identity."
        )
    noise = pure_hubbard_noise_level_contract(adapter.noise_level_id)
    payload: dict[str, Any] = {
        "schema": "paper_i_pure_hubbard_noise_page12_application_source_contract_v2",
        "application_id": PAPER_I_PURE_HUBBARD_NOISE_PAGE12_APPLICATION_ID,
        "algorithm_id": PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ALGORITHM_ID,
        "adapter_id": PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ADAPTER_ID,
        "problem_request_sha256": exact_source.problem_request_sha256,
        "problem": {
            "family": "hubbard",
            "L": 2,
            "t": 1.0,
            "U": float(problem.request.u),
            "ordering": "blocked",
            "boundary": "open",
            "sector_num_particles": [1, 1],
            "boson_register_qubits": 0,
        },
        "parent_inventory": parent.receipt.to_dict(),
        "singleton_inventory": singleton.receipt.to_dict(),
        "noise": noise,
        "noise_surface": {
            "candidate_gradient_scoring": "noisy",
            "powell_refit_objective": "noisy",
            "geometry_and_gram": "exact",
            "reported_energy": "exact_diagnostic",
        },
        "same_cutoff_exact_reference": {
            **exact_source.to_dict(),
            "evaluation_policy_id": (
                _SAME_CUTOFF_EXACT_EVALUATION_POLICY_ID
            ),
            "analytic_reference": {
                "formula_id": _SAME_CUTOFF_EXACT_FORMULA_ID,
                "t": {
                    "numerator": t_numerator,
                    "denominator": t_denominator,
                },
                "U": {
                    "numerator": u_numerator,
                    "denominator": u_denominator,
                },
            },
            "controller_input": False,
        },
        "scientific_settings": {
            "maximum_controller_rounds": int(
                request.execution.stop.maximum_controller_rounds
            ),
            "optimizer": "powell",
            "optimizer_maxiter": 200,
            "seed": 7,
            "active_gradient_policy": ACTIVE_GRADIENT_STATIONARY,
            "resource_weighting_scope": RESOURCE_WEIGHTING_ALL_PHASE,
            "phase0_shortlist_size": 24,
            "phase0_cost": "gradient_only_no_metric_no_resource_v1",
            "phase2_phase3_cost": (
                "qiskit_full_trial_ansatz_signed_marginal_no_lanes_v1"
            ),
            "insertion": (
                "cumulative_relative_plateau_commutation_reduced_"
                "tau1em4_v1"
            ),
            "plateau_energy_source": (
                "persisted_noisy_controller_energy_before_after_v1"
            ),
            "optimizer_evaluation_order": "serial_v1",
        },
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def require_paper_i_pure_hubbard_noise_page12_materialization(
    *,
    problem: ResolvedProblemContext,
    request: RAAdaptRequest,
    algorithm_id: str,
    active_gradient_policy: str,
    resource_weighting_scope: str,
    source_locks: Mapping[str, str],
) -> None:
    """Require bundle authority for the exact named noise application."""

    require_paper_i_pure_hubbard_noise_page12_problem(problem)
    require_paper_i_pure_hubbard_noise_page12_request(request)
    expected = paper_i_pure_hubbard_noise_page12_application_source_contract(
        problem,
        request,
    )
    if (
        str(algorithm_id) != PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ALGORITHM_ID
        or str(active_gradient_policy) != ACTIVE_GRADIENT_STATIONARY
        or str(resource_weighting_scope) != RESOURCE_WEIGHTING_ALL_PHASE
        or not isinstance(source_locks, Mapping)
        or source_locks.get(PAPER_I_PURE_HUBBARD_NOISE_PAGE12_SOURCE_LOCK_KEY)
        != expected["sha256"]
    ):
        raise ValueError(
            "The named pure-Hubbard Page-12 noise application requires its "
            "stationary all-phase bundle and exact application source-lock "
            "digest."
        )


def is_paper_i_pure_hubbard_noise_page12_application(
    problem: ResolvedProblemContext | None,
    request: RAAdaptRequest,
) -> bool:
    if not isinstance(
        request.adapter,
        PaperIPureHubbardNoisePage12CandidateAdapter,
    ):
        return False
    if problem is None:
        raise ValueError("The pure-Hubbard noise adapter requires its problem.")
    require_paper_i_pure_hubbard_noise_page12_problem(problem)
    require_paper_i_pure_hubbard_noise_page12_request(request)
    return True


__all__ = [
    "PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ADAPTER_ID",
    "PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ALGORITHM_ID",
    "PAPER_I_PURE_HUBBARD_NOISE_PAGE12_SOURCE_LOCK_KEY",
    "PaperIPureHubbardNoisePage12CandidateAdapter",
    "build_paper_i_pure_hubbard_noise_page12_problem",
    "build_paper_i_pure_hubbard_noise_page12_request",
    "is_paper_i_pure_hubbard_noise_page12_application",
    "paper_i_pure_hubbard_noise_page12_application_source_contract",
    "pure_hubbard_noise_level_contract",
    "require_paper_i_pure_hubbard_noise_page12_materialization",
    "require_paper_i_pure_hubbard_noise_page12_request",
]
