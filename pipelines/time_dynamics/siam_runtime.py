r"""Make the SIAM screening pilot executable by the AP-McLachlan trajectory core.

The AP trajectory core takes an :class:`APMcLachlanState` and a
:class:`TimeDependentHamiltonian` directly.  It does *not* require a scaffold
ADAPT-VQE seed artifact --- that path exists because Hubbard-Holstein seeds come
from a static ADAPT run, which SIAM does not have and should not fake.  This
module builds the two objects the core actually wants, so the pilot runs through
the unmodified controller, integrator, guards, and run-lock machinery.

The post-quench Hamiltonian is time-independent: the quench is a change of
initial condition, not a drive, so ``drive_model`` is ``None`` and
``H(t) = H_static``.

``resolved_problem`` is supplied as a small explicit adapter rather than a
Hubbard-Holstein problem context.  The AP route consults it in exactly two
places --- the static-Hamiltonian parity gate and the observable-family
dispatch --- and SIAM supplies its own observables, so a light object carrying
``family_key`` and ``hamiltonian`` is the honest surface. Passing a fabricated
HH context instead would make the family dispatch silently return doublon and
site-occupation plans that mean nothing here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.siam_pool import build_siam_pool
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, hamiltonian_matrix
from pipelines.contracts.scaffold import CandidatePoolSource, ScaffoldRuntimeInput
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import (
    TimeDependentHamiltonian,
    time_dependent_hamiltonian_from_runtime_input,
)
from pipelines.time_dynamics.ap_mclachlan.state import (
    AP_PARAMETERIZATION_LOGICAL_SHARED,
    APMcLachlanState,
    state_from_scaffold_runtime_input,
)
from pipelines.time_dynamics.siam_screening import (
    SIAMScreeningConfig,
    build_screening_problem,
)

if TYPE_CHECKING:
    from pipelines.time_dynamics.siam_seed import SIAMSeedManifest

SIAM_RUNTIME_V1 = "siam_screening_runtime_v1"
SIAM_FAMILY_KEY = "siam"


@dataclass(frozen=True)
class SIAMProblemContext:
    """Minimal resolved-problem surface the AP route consults.

    Deliberately not a Hubbard-Holstein context: the observable dispatch keys on
    ``family_key``, and claiming ``hh`` here would hand SIAM a doublon plan built
    for a two-site fermion-boson lattice.
    """

    family_key: str
    hamiltonian: Any
    n_bath: int
    n_qubits: int
    capabilities: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SIAMRuntime:
    """Everything needed to call ``run_append_mclachlan_trajectory``."""

    state: APMcLachlanState
    hamiltonian: TimeDependentHamiltonian
    candidate_pool_terms: tuple[AnsatzTerm, ...]
    observables: Mapping[str, Any]
    times: tuple[float, ...]
    provenance: dict[str, Any]


def pool_ansatz_terms(n_bath: int) -> tuple[AnsatzTerm, ...]:
    """The logical pool as ``AnsatzTerm`` generators, in deterministic order."""

    return tuple(
        AnsatzTerm(label=block.label, polynomial=block.operator)
        for block in build_siam_pool(int(n_bath))
    )


def build_siam_runtime(
    config: SIAMScreeningConfig | None = None,
    *,
    seed_terms: Sequence[AnsatzTerm] = (),
) -> SIAMRuntime:
    """Build the AP state and Hamiltonian for a SIAM screening run.

    ``seed_terms`` is normally empty: the quench starts from the decoupled
    Fermi sea with no variational coordinates, and the route grows the support
    itself.  Supplying terms seeds a non-empty starting ansatz.
    """

    cfg = config or SIAMScreeningConfig()
    runtime_input = build_siam_runtime_input(cfg, seed_terms=seed_terms)
    problem = build_screening_problem(cfg)
    state = state_from_scaffold_runtime_input(
        runtime_input,
        parameterization_mode=AP_PARAMETERIZATION_LOGICAL_SHARED,
    )
    hamiltonian = time_dependent_hamiltonian_from_runtime_input(runtime_input)
    return SIAMRuntime(
        state=state,
        hamiltonian=hamiltonian,
        candidate_pool_terms=state.candidate_pool_terms,
        observables=problem.observables,
        times=cfg.times,
        provenance=dict(problem.provenance),
    )


def build_siam_runtime_input(
    config: SIAMScreeningConfig | None = None,
    *,
    seed_terms: Sequence[AnsatzTerm] = (),
    artifact_json: str | None = None,
    seed_manifest: "SIAMSeedManifest | None" = None,
    verified_seed_sha256: str | None = None,
) -> ScaffoldRuntimeInput:
    """Build the family-neutral production input for one verified SIAM seed."""

    problem = build_screening_problem(config)
    cfg = problem.config
    terms = tuple(seed_terms)
    production_binding = (
        artifact_json,
        seed_manifest,
        verified_seed_sha256,
    )
    if any(value is not None for value in production_binding) and not all(
        value is not None for value in production_binding
    ):
        raise ValueError(
            "artifact_json, seed_manifest, and verified_seed_sha256 must be "
            "supplied together so a production SIAM input cannot claim an "
            "unverified seed path."
        )
    if seed_manifest is not None:
        from pipelines.time_dynamics.siam_seed import build_siam_seed_manifest

        if terms:
            raise ValueError(
                "Manifest-backed SIAM production inputs require empty seed_terms; "
                "the immutable prepared-state seed starts with empty support."
            )
        if build_siam_seed_manifest(
            cfg,
            schema=seed_manifest.schema,
        ) != seed_manifest:
            raise ValueError(
                "SIAM runtime physics does not match the verified seed manifest."
            )
        digest = str(verified_seed_sha256).strip().lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("verified_seed_sha256 must be a lowercase SHA-256 digest.")
    else:
        digest = None
    layout = build_parameter_layout(
        terms, ignore_identity=True, coefficient_tolerance=1.0e-12, sort_terms=True
    )
    psi_ref = np.asarray(problem.initial_state, dtype=complex).reshape(-1)
    context = SIAMProblemContext(
        family_key=SIAM_FAMILY_KEY,
        hamiltonian=problem.hamiltonian,
        n_bath=int(cfg.n_bath),
        n_qubits=int(cfg.layout.n_qubits),
        metadata=dict(cfg.physics_provenance()),
    )

    # The pool is declared complete: it is the full enumerated block list from
    # build_siam_pool, not a filtered shortlist. The AP route refuses structural
    # edits from an incomplete pool, and rightly so -- a truncated pool makes a
    # "best available insertion" claim that the enumeration cannot support.
    pool_source = CandidatePoolSource(
        source_kind="resolved_pool",
        pool_key=f"siam_screening_nb{int(cfg.n_bath)}",
        completeness="complete",
        pool_build_kwargs={"n_bath": int(cfg.n_bath)},
    )

    logical_count = int(layout.logical_parameter_count)
    runtime_count = int(layout.runtime_parameter_count)
    provenance = {
        "source": SIAM_RUNTIME_V1,
        **cfg.physics_provenance(),
    }
    if artifact_json is not None:
        provenance["artifact_json"] = str(artifact_json)
    if seed_manifest is not None:
        provenance["seed_schema"] = str(seed_manifest.schema)
        provenance["verified_seed_sha256"] = str(digest)
    return ScaffoldRuntimeInput(
        resolved_problem=context,
        psi_ref=psi_ref,
        psi_initial=psi_ref.copy(),
        base_layout=layout,
        theta_runtime=np.zeros(runtime_count, dtype=float),
        theta_logical=np.zeros(logical_count, dtype=float),
        structure_locked=False,
        exact_energy=None,
        selected_terms=terms,
        candidate_pool_terms=pool_ansatz_terms(cfg.n_bath),
        candidate_pool_source=pool_source,
        provenance=provenance,
        extensions={
            "parameterization_mode": AP_PARAMETERIZATION_LOGICAL_SHARED,
            "seed_manifest_bound": bool(seed_manifest is not None),
        },
    )


def evaluate_screening_observables(
    runtime: SIAMRuntime, psi: np.ndarray
) -> dict[str, float]:
    """Screening observables of a statevector, in the report's names."""

    vector = np.asarray(psi, dtype=complex).reshape(-1)
    out: dict[str, float] = {}
    for name, poly in runtime.observables.items():
        matrix = np.asarray(hamiltonian_matrix(poly), dtype=complex)
        out[name] = float(np.real(np.vdot(vector, matrix @ vector)))
    return out


__all__ = [
    "SIAMProblemContext",
    "SIAMRuntime",
    "SIAM_FAMILY_KEY",
    "SIAM_RUNTIME_V1",
    "build_siam_runtime",
    "build_siam_runtime_input",
    "evaluate_screening_observables",
    "pool_ansatz_terms",
]
