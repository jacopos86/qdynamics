"""Canonical Paper-I parent, macro, and guarded singleton pool ownership."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from pipelines.contracts.problem import ResolvedProblemContext
from pipelines.static_adapt.builders.pool_resolution import (
    resolve_pool_plan,
    resolve_requested_pool_filters,
)
from pipelines.static_adapt.builders.shared_pauli_pool_contract import (
    SHARED_PAULI_POOL_MODE_GUARDED_SINGLETON_CHILDREN_ONLY_V1,
    SharedPauliPoolParent,
    build_shared_pauli_child_pool,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    CANDIDATE_INVENTORY_LINEAGE_SCHEMA,
    CANDIDATE_REPRESENTATION_MACRO,
    CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    CandidateInventoryLineageReceipt,
    CandidateInventoryLineageRow,
    PoolInventoryReceipt,
    canonical_sha256,
)
from pipelines.static_adapt.sector_invariants import (
    audit_generator_sector_contract,
    audit_candidate_pool_sector_contract,
    resolve_fixed_count_qubit_groups,
)
from pipelines.scaffold.hh_continuation_generators import (
    serialize_polynomial_terms_exyz,
)
from src.quantum.vqe_latex_python_pairs import AnsatzTerm
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


PARENT_TEMPLATE_INVENTORY_SCHEMA = "ra_adapt_parent_template_inventory_v1"
EXECUTABLE_MACRO_POOL_SCHEMA = "ra_adapt_executable_macro_pool_v1"
GUARDED_SINGLETON_POOL_SCHEMA = "ra_adapt_guarded_singleton_pool_v1"
STAGED_SINGLETON_POOL_SCHEMA = "ra_adapt_staged_singleton_exposure_v1"
GUARDED_SINGLETON_GENERATOR_IDENTITY_SCHEMA = (
    "ra_adapt_guarded_singleton_intrinsic_generator_identity_v1"
)
H2O_SYMMETRY_COMPLETE_POOL_SCHEMA = (
    "paper_iv_h2o_ra_adapt_symmetry_complete_generator_pool_v1"
)
H2O_SECTOR_COMPLETE_PAULI_BLOCK_POOL_SCHEMA = (
    "paper_iv_h2o_ra_adapt_sector_complete_pauli_block_pool_v1"
)
H2O_SECTOR_COMPLETE_PAULI_BLOCK_IDENTITY_SCHEMA = (
    "paper_iv_h2o_sector_complete_pauli_block_identity_v1"
)

EXPECTED_PARENT_COUNTS = {3: 123, 7: 171}
EXPECTED_EXECUTABLE_MACRO_COUNTS = {3: 102, 7: 148}
EXPECTED_MACRO_REMOVED_COUNTS = {3: 21, 7: 23}
H2O_LINEAR_FD_FAMILY = "molecular_vibronic_h2o_linear_fd"
H2O_DERIVATIVE_RESOLVED_POOL = "full_meta_derivative_resolved_v2"
EXPECTED_H2O_DERIVATIVE_RESOLVED_PARENT_COUNT = 448
PAPER_I_L3_PAGE12_APPLICATION_ID = (
    "paper_i_l3_page12_global_singleton_gradient_phase0_v1"
)
PAPER_I_L3_PAGE12_PROBLEM_REQUEST_SHA256 = (
    "c67ea0c6489b6e35d1fdebdffd6f9ed700d0712c4f2745569dca37ad3fdaca2e"
)
PAPER_I_L3_PAGE12_HAMILTONIAN_TERMS_SHA256 = (
    "017adb185a9be650417682edee1a232d76639a4551d76042c7887a8ed9183b83"
)
PAPER_I_L3_PAGE12_PARENT_COUNT = 199
PAPER_I_L3_PAGE12_PARENT_LABELS_SHA256 = (
    "2a5b4c15f4264e319cec2d6188d1d41f375696d0269f5077d492e2df7f140763"
)
PAPER_I_L3_PAGE12_PARENT_POOL_SHA256 = (
    "529382a677566cd67861cb46edc0b5dbb2a8922536629001f1552de2616c625b"
)
PAPER_I_L3_PAGE12_SINGLETON_COUNT = 74
PAPER_I_L3_PAGE12_SINGLETON_LABELS_SHA256 = (
    "87d95b07689629696cac2fffb72a9018996bcdd5050aab00949095960209a40e"
)
PAPER_I_L3_PAGE12_SINGLETON_POOL_SHA256 = (
    "147afb129aa0f5b254661e082781d8ac0b1a6e0624a21a7948eb0cad3393e894"
)
PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_PROBLEM_LOCKS = {
    "weak_weak": {
        "problem_request_sha256": (
            "560516f13b2af570209a91d5ec809a517622fdb80171165abb84d80900dc8b57"
        ),
        "hamiltonian_terms_sha256": (
            "4eac4bcf62c66fc6203389ae4c1705d9c8d9abfb5e01b540097219d23e50d135"
        ),
    },
    "intermediate_weak": {
        "problem_request_sha256": (
            "bd248fa60a6d9ec06cdc68dc873cdab30b2b445edf4fe4e98eda8daebb03ccb6"
        ),
        "hamiltonian_terms_sha256": (
            "d874caab1c963100723cb6097f88d37c943dd48c4a284352749e63691ee993a1"
        ),
    },
    "strong_weak_u8": {
        "problem_request_sha256": (
            "ca9a650982535cdf90c84fca0f2a20342e1f7c0dc02fd1d48e64424e24ccac1b"
        ),
        "hamiltonian_terms_sha256": (
            "e1a4153820e56408f81b7282726808fe01e55aaefcf9ea53b1c57ea974ed33da"
        ),
    },
}
PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_PARENT_COUNT = 251
PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_PARENT_LABELS_SHA256 = (
    "a22e7b97e0f47006703f74bb1b1477f98ee048ada9a9568314fea07f6974e5f8"
)
PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_PARENT_POOL_SHA256 = (
    "0bd9a12af1945ec06d0eb7d252ddb6ca7c8e391847819f12bfaf35530b984a5c"
)
PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_SINGLETON_COUNT = 314
PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_SINGLETON_LABELS_SHA256 = (
    "2c5a82544bea5c8e1d55c22def7c7cc431315e9da9bf693d25a45ce54704b860"
)
PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_SINGLETON_POOL_SHA256 = (
    "c9e76e6c5679236e5b1b88446690de80676d0cf670f72b58b2848e0d4cb91b23"
)
PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_POOL_LOCKS = {
    "weak_weak": {
        "parent_pool_sha256": (
            "0bd9a12af1945ec06d0eb7d252ddb6ca7c8e391847819f12bfaf35530b984a5c"
        ),
        "singleton_pool_sha256": (
            "c9e76e6c5679236e5b1b88446690de80676d0cf670f72b58b2848e0d4cb91b23"
        ),
    },
    "intermediate_weak": {
        "parent_pool_sha256": (
            "6e1591a11f21eede3332b5570f42a844ab4e9229818bb0128d6634037ee97298"
        ),
        "singleton_pool_sha256": (
            "0963385b7c74622f40975292ed20867730abd73b89f776d4be1f58d512be21e7"
        ),
    },
    "strong_weak_u8": {
        "parent_pool_sha256": (
            "f070fefc4a1c4e45eec81e01db0e8f793fde2fca1297e5159cd6eaec8b9524cd"
        ),
        "singleton_pool_sha256": (
            "8ff33f4f5686249ab59e1c177814d920ee86db5fb53d4334a39730d3642d1a05"
        ),
    },
}
PAPER_I_PURE_HUBBARD_NOISE_PAGE12_APPLICATION_ID = (
    "paper_i_pure_hubbard_noise_page12_global_singleton_gradient_phase0_v1"
)


@dataclass(frozen=True)
class CandidateRecord:
    """One executable candidate plus serialized lineage.

    ``term`` is the live numerical object.  Hashes are derived only from the
    explicit deterministic fields returned by :meth:`manifest_row`.
    """

    label: str
    term: AnsatzTerm
    representation_id: str
    generator_identity: str
    parent_identities: tuple[str, ...]
    family_id: str
    stage_family: str
    construction: str
    execution_mode: str
    serialized_terms_exyz: tuple[Mapping[str, Any], ...]
    symmetry_receipt: Mapping[str, Any] | None = None
    generator_metadata: Mapping[str, Any] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    def manifest_row(self) -> dict[str, Any]:
        return {
            "label": str(self.label),
            "representation_id": str(self.representation_id),
            "generator_identity": str(self.generator_identity),
            "parent_identities": [str(value) for value in self.parent_identities],
            "family_id": str(self.family_id),
            "stage_family": str(self.stage_family),
            "construction": str(self.construction),
            "execution_mode": str(self.execution_mode),
            "serialized_terms_exyz": [
                dict(row) for row in self.serialized_terms_exyz
            ],
            "symmetry_receipt": (
                None
                if self.symmetry_receipt is None
                else dict(self.symmetry_receipt)
            ),
        }


@dataclass(frozen=True)
class CandidateInventory:
    candidates: tuple[CandidateRecord, ...]
    receipt: PoolInventoryReceipt
    metadata: Mapping[str, Any]


def build_candidate_inventory_lineage_receipt(
    inventory: CandidateInventory,
) -> CandidateInventoryLineageReceipt:
    """Project one executable inventory to its compact full-lineage receipt.

    This receipt intentionally sits beside :class:`PoolInventoryReceipt`.
    Adding lineage to the pool receipt itself would rewrite the locked
    123/171 and 102/148 pool identities.
    """

    if not isinstance(inventory, CandidateInventory):
        raise TypeError("inventory must be a CandidateInventory.")
    rows = tuple(
        CandidateInventoryLineageRow(
            label=str(candidate.label),
            representation_id=str(candidate.representation_id),
            generator_identity=str(candidate.generator_identity),
            parent_identities=tuple(
                str(value) for value in candidate.parent_identities
            ),
        )
        for candidate in inventory.candidates
    )
    if tuple(row.label for row in rows) != tuple(
        inventory.receipt.ordered_labels
    ):
        raise RuntimeError(
            "Candidate-inventory lineage order differs from the locked pool "
            "order."
        )
    if any(
        row.representation_id
        != inventory.receipt.candidate_representation
        for row in rows
    ):
        raise RuntimeError(
            "Candidate-inventory lineage representation differs from the "
            "pool receipt."
        )
    if int(len(rows)) != int(inventory.receipt.count):
        raise RuntimeError(
            "Candidate-inventory lineage count differs from the pool receipt."
        )
    return CandidateInventoryLineageReceipt(
        schema=CANDIDATE_INVENTORY_LINEAGE_SCHEMA,
        candidate_representation=str(
            inventory.receipt.candidate_representation
        ),
        pool_inventory_sha256=str(inventory.receipt.sha256),
        ordered_rows=rows,
        ordered_rows_sha256=canonical_sha256(
            [row.to_dict() for row in rows]
        ),
        count=int(len(rows)),
    )


def _require_paper_i_problem(problem: ResolvedProblemContext) -> None:
    if not isinstance(problem, ResolvedProblemContext):
        raise TypeError("problem must be a ResolvedProblemContext.")
    if (
        str(problem.family_key).strip().lower() != "hh"
        or int(problem.request.num_sites) != 2
    ):
        raise ValueError(
            "The canonical Paper-I RA-ADAPT pool is locked to the "
            "Hubbard--Holstein L=2 problem."
        )


def require_h2o_symmetry_complete_problem(
    problem: ResolvedProblemContext,
) -> None:
    """Require the supported three-mode CAS(8e,6o) full-binary H2O lane."""

    if not isinstance(problem, ResolvedProblemContext):
        raise TypeError("problem must be a ResolvedProblemContext.")
    n_ph_max = int(problem.request.n_ph_max)
    level_count = n_ph_max + 1
    qubits_per_mode = max(0, (level_count - 1).bit_length())
    mode_cutoffs = tuple(
        int(value)
        for value in problem.runtime_data.get(
            "vibronic_h2o_linear_fd_mode_cutoffs",
            (),
        )
    )
    if (
        str(problem.family_key).strip().lower() != H2O_LINEAR_FD_FAMILY
        or int(problem.request.num_sites) != 6
        or tuple(int(value) for value in problem.sector.num_particles)
        != (4, 4)
        or str(problem.request.boson_encoding).strip().lower() != "binary"
        or level_count < 1
        or bool(level_count & (level_count - 1))
        or mode_cutoffs != (n_ph_max, n_ph_max, n_ph_max)
        or int(problem.layout.fermion_qubits) != 12
        or int(problem.layout.boson_qubits) != 3 * qubits_per_mode
        or int(problem.layout.total_qubits) != 12 + 3 * qubits_per_mode
    ):
        raise ValueError(
            "The H2O RA application requires the three-mode CAS(8e,6o), "
            "(4 alpha, 4 beta), equal-cutoff full-binary linear-FD problem."
        )


def _require_supported_singleton_problem(
    problem: ResolvedProblemContext,
) -> None:
    if not isinstance(problem, ResolvedProblemContext):
        raise TypeError("problem must be a ResolvedProblemContext.")
    family = str(problem.family_key).strip().lower()
    if family == "hh" and int(problem.request.num_sites) == 2:
        return
    if family == H2O_LINEAR_FD_FAMILY:
        require_h2o_symmetry_complete_problem(problem)
        return
    raise ValueError(
        "The staged RA singleton pool supports canonical Paper-I HH L=2 "
        "or the named Paper-IV H2O linear-FD CAS(8e,6o) lane."
    )


def require_paper_i_l3_page12_problem(
    problem: ResolvedProblemContext,
) -> str:
    """Validate one explicitly locked L=3 Page-12 application point.

    This stays separate from ``_require_supported_singleton_problem`` so the
    ordinary Paper-I pool entrypoints remain locked to Hubbard--Holstein
    ``L=2``.  Only the named L=3 adapter opts into either the preserved
    intermediate--weak ``nph=1`` point or the three requested weak-Holstein
    ``nph=3`` points.
    """

    if not isinstance(problem, ResolvedProblemContext):
        raise TypeError("problem must be a ResolvedProblemContext.")
    from pipelines.static_adapt.sr_snake.contracts import (
        ResolvedProblemReceipt,
    )

    receipt = ResolvedProblemReceipt.from_problem(problem)
    particles = tuple(int(value) for value in problem.sector.num_particles)
    legacy = (
        receipt.problem_request_sha256
        == PAPER_I_L3_PAGE12_PROBLEM_REQUEST_SHA256
    )
    weak_sector_regime = next(
        (
            regime_id
            for regime_id, lock in sorted(
                PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_PROBLEM_LOCKS.items()
            )
            if receipt.problem_request_sha256
            == lock["problem_request_sha256"]
        ),
        None,
    )
    nph = int(problem.request.n_ph_max)
    qubits_per_mode = max(1, int(math.ceil(math.log2(nph + 1))))
    if (
        str(problem.family_key).strip().lower() != "hh"
        or (not legacy and weak_sector_regime is None)
        or int(problem.request.num_sites) != 3
        or particles != (2, 1)
        or int(problem.layout.fermion_qubits) != 6
        or int(problem.layout.boson_qubits) != 3 * qubits_per_mode
        or int(problem.layout.total_qubits) != 6 + 3 * qubits_per_mode
        or str(problem.sector.label) != "half_filled_fermion_sector"
        or str(problem.default_reference_label)
        != "hubbard_holstein_reference_state"
        or str(problem.exact_target_label)
        != "exact_ground_energy_sector_hh"
        or str(problem.exact_comparison_space_label)
        != "fermion_register_with_truncated_bosons"
        or "full_meta" not in set(problem.admissible_pool_keys)
    ):
        raise ValueError(
            "The named Page-12 L=3 application requires either its exact "
            "intermediate--weak HH point at n_ph_max=1 or one of the "
            "three exact weak-Holstein n_ph_max=3 points, with open/binary/"
            "blocked encoding, (N_up,N_down)=(2,1), and same-cutoff ED."
        )
    hamiltonian_sha256 = canonical_sha256(
        serialize_polynomial_terms_exyz(problem.hamiltonian)
    )
    expected_hamiltonian = (
        PAPER_I_L3_PAGE12_HAMILTONIAN_TERMS_SHA256
        if legacy
        else str(
            PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_PROBLEM_LOCKS[
                str(weak_sector_regime)
            ]["hamiltonian_terms_sha256"]
        )
    )
    if hamiltonian_sha256 != expected_hamiltonian:
        raise ValueError(
            "The named Page-12 L=3 Hamiltonian bytes drifted from the "
            "locked application construction."
        )
    return "legacy_intermediate_weak_nph1" if legacy else str(weak_sector_regime)


def require_paper_i_pure_hubbard_noise_page12_problem(
    problem: ResolvedProblemContext,
) -> None:
    """Admit only the two named pure-Hubbard noise benchmark points."""

    if not isinstance(problem, ResolvedProblemContext):
        raise TypeError("problem must be a ResolvedProblemContext.")
    request = problem.request
    particles = tuple(int(value) for value in problem.sector.num_particles)
    if (
        str(problem.family_key).strip().lower() != "hubbard"
        or str(request.problem_key).strip().lower() != "hubbard"
        or int(request.num_sites) != 2
        or not any(
            math.isclose(float(request.u), value, rel_tol=0.0, abs_tol=0.0)
            for value in (1.5, 8.0)
        )
        or float(request.t) != 1.0
        or float(request.dv) != 0.0
        or float(request.v_nn) != 0.0
        or float(request.t_prime) != 0.0
        or float(request.g_ep) != 0.0
        or float(request.omega0) != 0.0
        or int(request.n_ph_max) != 0
        or str(request.ordering).strip().lower() != "blocked"
        or str(request.boundary).strip().lower() != "open"
        or particles != (1, 1)
        or int(problem.layout.fermion_qubits) != 4
        or int(problem.layout.boson_qubits) != 0
        or int(problem.layout.total_qubits) != 4
        or "full_meta" not in set(problem.admissible_pool_keys)
    ):
        raise ValueError(
            "The named pure-Hubbard Page-12 noise application requires "
            "L=2, t=1, U/t in {1.5, 8}, (N_up,N_down)=(1,1), open/blocked, "
            "zero boson coupling/cutoff, and the full_meta pool."
        )


def _parent_pool_spec(
    problem: ResolvedProblemContext,
    *,
    paper_i_l3_page12_application: bool = False,
    paper_i_pure_hubbard_noise_page12_application: bool = False,
) -> dict[str, Any]:
    if paper_i_l3_page12_application and paper_i_pure_hubbard_noise_page12_application:
        raise ValueError("Only one named RA pool application may be active.")
    if paper_i_pure_hubbard_noise_page12_application:
        require_paper_i_pure_hubbard_noise_page12_problem(problem)
        return {
            "problem_key": "hubbard",
            "num_sites": 2,
            "pool_key": "full_meta",
            # Pure Hubbard has no boson register.  The shared pool builder
            # still accepts a positive divisor and observes qpb=0.
            "boson_mode_count": 1,
        }
    if paper_i_l3_page12_application:
        require_paper_i_l3_page12_problem(problem)
        return {
            "problem_key": "hh",
            "num_sites": 3,
            "pool_key": "full_meta",
            "boson_mode_count": 3,
        }
    _require_supported_singleton_problem(problem)
    family = str(problem.family_key).strip().lower()
    if family == "hh":
        return {
            "problem_key": "hh",
            "num_sites": 2,
            "pool_key": "full_meta",
            "boson_mode_count": 2,
        }
    return {
        "problem_key": H2O_LINEAR_FD_FAMILY,
        "num_sites": 6,
        "pool_key": H2O_DERIVATIVE_RESOLVED_POOL,
        "boson_mode_count": 3,
    }


def _serialized_terms(term: AnsatzTerm) -> tuple[dict[str, Any], ...]:
    return tuple(
        dict(row) for row in serialize_polynomial_terms_exyz(term.polynomial)
    )


def _generator_metadata(term: AnsatzTerm) -> dict[str, Any]:
    raw = getattr(term, "generator_metadata", None)
    if not isinstance(raw, Mapping):
        raw = getattr(term, "metadata", None)
    return dict(raw) if isinstance(raw, Mapping) else {}


def _parent_records(
    problem: ResolvedProblemContext,
    *,
    paper_i_l3_page12_application: bool = False,
    paper_i_pure_hubbard_noise_page12_application: bool = False,
) -> tuple[tuple[CandidateRecord, ...], Any]:
    spec = _parent_pool_spec(
        problem,
        paper_i_l3_page12_application=paper_i_l3_page12_application,
        paper_i_pure_hubbard_noise_page12_application=(
            paper_i_pure_hubbard_noise_page12_application
        ),
    )
    problem_key = str(spec["problem_key"])
    num_sites = int(spec["num_sites"])
    pool_key = str(spec["pool_key"])
    filters = resolve_requested_pool_filters(
        problem_key=problem_key,
        num_sites=num_sites,
        n_ph_max=int(problem.request.n_ph_max),
        adapt_pool=pool_key,
        adapt_pool_class_filter_json=None,
        adapt_pool_label_filter_json=None,
        adapt_selected_logical_source_json=None,
        adapt_selected_logical_mode="off",
        adapt_selected_logical_transfer_mode="exact_match_v1",
    )
    plan = resolve_pool_plan(
        resolved_problem=problem,
        continuation_mode="benchmark_static_geo_adapt",
        adapt_pool=pool_key,
        paop_r=2,
        paop_split_paulis=False,
        paop_prune_eps=1.0e-12,
        paop_normalization="none",
        phase3_symmetry_mitigation_mode="off",
        filter_resolution=filters,
    )
    records: list[CandidateRecord] = []
    for index, term in enumerate(plan.pool):
        serialized = _serialized_terms(term)
        if not serialized:
            continue
        metadata = _generator_metadata(term)
        records.append(
            CandidateRecord(
                label=str(term.label),
                term=term,
                representation_id=CANDIDATE_REPRESENTATION_MACRO,
                generator_identity=str(
                    metadata.get("generator_id")
                    or metadata.get("generator_identity")
                    or f"parent:{canonical_sha256({'label': str(term.label), 'terms': serialized})[:16]}"
                ),
                parent_identities=(),
                family_id=str(
                    plan.pool_family_ids[index]
                    if index < len(plan.pool_family_ids)
                    else pool_key
                ),
                stage_family=str(
                    plan.pool_stage_family[index]
                    if index < len(plan.pool_stage_family)
                    else "shared"
                ),
                construction=f"{pool_key}::{plan.pool_key}",
                execution_mode=str(
                    getattr(term, "execution_mode", "termwise_product")
                    or "termwise_product"
                ),
                serialized_terms_exyz=serialized,
                symmetry_receipt=(
                    dict(plan.pool_symmetry_specs[index])
                    if index < len(plan.pool_symmetry_specs)
                    and isinstance(plan.pool_symmetry_specs[index], Mapping)
                    else None
                ),
                generator_metadata=metadata,
            )
        )
    return tuple(records), plan


def _receipt(
    *,
    schema: str,
    representation_id: str,
    candidates: Sequence[CandidateRecord],
    removed_labels: Sequence[str] = (),
    source_parent_labels_sha256: str | None = None,
) -> PoolInventoryReceipt:
    labels = tuple(str(candidate.label) for candidate in candidates)
    pool_identity_rows = []
    for candidate in candidates:
        row = candidate.manifest_row()
        # Representation is a consumer view of the common parent supply, not
        # part of that supply's scientific identity.  This keeps the 123/171
        # parent hash identical for macro and singleton comparisons.
        row.pop("representation_id", None)
        pool_identity_rows.append(row)
    return PoolInventoryReceipt(
        schema=str(schema),
        candidate_representation=str(representation_id),
        ordered_labels=labels,
        ordered_labels_sha256=canonical_sha256(list(labels)),
        ordered_pool_sha256=canonical_sha256(pool_identity_rows),
        count=int(len(candidates)),
        removed_labels=tuple(str(value) for value in removed_labels),
        source_parent_ordered_labels_sha256=source_parent_labels_sha256,
    )


def _build_parent_template_inventory(
    problem: ResolvedProblemContext,
    *,
    representation_id: str,
    paper_i_l3_page12_application: bool,
    paper_i_pure_hubbard_noise_page12_application: bool = False,
) -> CandidateInventory:
    if representation_id not in {
        CANDIDATE_REPRESENTATION_MACRO,
        CANDIDATE_REPRESENTATION_SINGLE_PAULI,
    }:
        raise ValueError("Unknown candidate representation.")
    application_identity = (
        require_paper_i_l3_page12_problem(problem)
        if paper_i_l3_page12_application
        else None
    )
    records, plan = _parent_records(
        problem,
        paper_i_l3_page12_application=paper_i_l3_page12_application,
        paper_i_pure_hubbard_noise_page12_application=(
            paper_i_pure_hubbard_noise_page12_application
        ),
    )
    if representation_id == CANDIDATE_REPRESENTATION_SINGLE_PAULI:
        records = tuple(
            CandidateRecord(
                **{
                    **record.__dict__,
                    "representation_id": CANDIDATE_REPRESENTATION_SINGLE_PAULI,
                }
            )
            for record in records
        )
    nph = int(problem.request.n_ph_max)
    family = str(problem.family_key).strip().lower()
    expected = (
        (
            PAPER_I_L3_PAGE12_PARENT_COUNT
            if int(problem.request.n_ph_max) == 1
            else PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_PARENT_COUNT
        )
        if paper_i_l3_page12_application
        else
        EXPECTED_H2O_DERIVATIVE_RESOLVED_PARENT_COUNT
        if family == H2O_LINEAR_FD_FAMILY
        else EXPECTED_PARENT_COUNTS.get(nph)
    )
    if expected is not None and len(records) != expected:
        raise RuntimeError(
            f"Parent full_meta inventory drifted at nph={nph}: "
            f"{len(records)} != {expected}."
        )
    inventory = CandidateInventory(
        candidates=records,
        receipt=_receipt(
            schema=PARENT_TEMPLATE_INVENTORY_SCHEMA,
            representation_id=representation_id,
            candidates=records,
        ),
        metadata={
            "pool_key": str(plan.pool_key),
            "source": "builders.pool_resolution.resolve_pool_plan",
            "paop": {
                "r": 2,
                "split_paulis": False,
                "prune_eps": 1.0e-12,
                "normalization": "none",
            },
        },
    )
    if paper_i_l3_page12_application:
        expected_parent_labels = (
            PAPER_I_L3_PAGE12_PARENT_LABELS_SHA256
            if nph == 1
            else PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_PARENT_LABELS_SHA256
        )
        expected_parent_pool = (
            PAPER_I_L3_PAGE12_PARENT_POOL_SHA256
            if nph == 1
            else str(
                PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_POOL_LOCKS[
                    str(application_identity)
                ]["parent_pool_sha256"]
            )
        )
        if (
            inventory.receipt.ordered_labels_sha256
            != expected_parent_labels
            or inventory.receipt.ordered_pool_sha256
            != expected_parent_pool
        ):
            raise RuntimeError(
                "The named Page-12 L=3 parent inventory drifted from its "
                "locked ordering or generator bytes."
            )
    return inventory


def build_parent_template_inventory(
    problem: ResolvedProblemContext,
    *,
    representation_id: str,
) -> CandidateInventory:
    """Build the ordinary L=2 or named H2O parent inventory."""

    return _build_parent_template_inventory(
        problem,
        representation_id=representation_id,
        paper_i_l3_page12_application=False,
    )


def build_paper_i_l3_page12_parent_template_inventory(
    problem: ResolvedProblemContext,
    *,
    representation_id: str,
) -> CandidateInventory:
    """Build the source-locked parent inventory for the named L=3 route."""

    return _build_parent_template_inventory(
        problem,
        representation_id=representation_id,
        paper_i_l3_page12_application=True,
    )


def build_paper_i_pure_hubbard_noise_page12_parent_template_inventory(
    problem: ResolvedProblemContext,
    *,
    representation_id: str,
) -> CandidateInventory:
    """Build the named pure-Hubbard parent inventory without widening defaults."""

    return _build_parent_template_inventory(
        problem,
        representation_id=representation_id,
        paper_i_l3_page12_application=False,
        paper_i_pure_hubbard_noise_page12_application=True,
    )


def build_executable_macro_pool(
    problem: ResolvedProblemContext,
) -> CandidateInventory:
    """Return the sector-safe 102/148 macro pool shared by RA and Append."""

    _require_paper_i_problem(problem)
    parent = build_parent_template_inventory(
        problem, representation_id=CANDIDATE_REPRESENTATION_MACRO
    )
    terms = [candidate.term for candidate in parent.candidates]
    audit = audit_candidate_pool_sector_contract(
        terms, resolved_problem=problem
    )
    from pipelines.static_adapt.adapt_pipeline import (
        _resolve_parent_sector_filter_policy,
    )

    policy = _resolve_parent_sector_filter_policy(
        grouped_violation_indices=set(
            int(value)
            for value in audit.get("grouped_violation_indices", ())
        ),
        execution_violation_indices=set(
            int(value)
            for value in audit.get("execution_violation_indices", ())
        ),
        problem_key="hh",
        runtime_split_mode="off",
        runtime_split_selection_mode="off",
        runtime_split_symmetry_policy="hard_guard",
    )
    removed_indices = {
        int(value) for value in policy.get("removed_indices", ())
    }
    retained = tuple(
        candidate
        for index, candidate in enumerate(parent.candidates)
        if index not in removed_indices
    )
    removed_labels = tuple(
        candidate.label
        for index, candidate in enumerate(parent.candidates)
        if index in removed_indices
    )
    nph = int(problem.request.n_ph_max)
    expected_removed = EXPECTED_MACRO_REMOVED_COUNTS.get(nph)
    expected_retained = EXPECTED_EXECUTABLE_MACRO_COUNTS.get(nph)
    if expected_removed is not None and len(removed_labels) != expected_removed:
        raise RuntimeError(
            f"Macro removed-set drifted at nph={nph}: "
            f"{len(removed_labels)} != {expected_removed}."
        )
    if expected_retained is not None and len(retained) != expected_retained:
        raise RuntimeError(
            f"Executable macro pool drifted at nph={nph}: "
            f"{len(retained)} != {expected_retained}."
        )
    return CandidateInventory(
        candidates=retained,
        receipt=_receipt(
            schema=EXECUTABLE_MACRO_POOL_SCHEMA,
            representation_id=CANDIDATE_REPRESENTATION_MACRO,
            candidates=retained,
            removed_labels=removed_labels,
            source_parent_labels_sha256=parent.receipt.ordered_labels_sha256,
        ),
        metadata={
            "parent_inventory": parent.receipt.to_dict(),
            "prefilter_sector_audit": dict(audit),
            "parent_sector_filter_policy": dict(policy),
        },
    )


def build_h2o_symmetry_complete_generator_pool(
    problem: ResolvedProblemContext,
) -> CandidateInventory:
    """Return all sector-preserving derivative-resolved H2O generators.

    A single Jordan--Wigner Pauli child of a fermionic excitation generally
    leaks from the fixed ``(N_alpha, N_beta)`` sector.  The complete grouped
    generator restores the cancellation that makes the excitation physical.
    This pool therefore keeps singleton *admission* while preserving each
    symmetry-complete generator as the executable candidate.
    """

    _require_supported_singleton_problem(problem)
    if str(problem.family_key).strip().lower() != H2O_LINEAR_FD_FAMILY:
        raise ValueError(
            "The H2O symmetry-complete pool requires the named molecular "
            "vibronic linear-FD application."
        )
    parent = build_parent_template_inventory(
        problem,
        representation_id=CANDIDATE_REPRESENTATION_MACRO,
    )
    audit = audit_candidate_pool_sector_contract(
        [candidate.term for candidate in parent.candidates],
        resolved_problem=problem,
    )
    grouped_violations = tuple(
        int(value)
        for value in audit.get("grouped_violation_indices", ())
    )
    execution_violations = tuple(
        int(value)
        for value in audit.get("execution_violation_indices", ())
    )
    if grouped_violations or execution_violations:
        raise RuntimeError(
            "The derivative-resolved H2O generator pool is not fully "
            "sector preserving: "
            f"grouped={grouped_violations!r}, "
            f"execution={execution_violations!r}."
        )
    return CandidateInventory(
        candidates=parent.candidates,
        receipt=_receipt(
            schema=H2O_SYMMETRY_COMPLETE_POOL_SCHEMA,
            representation_id=CANDIDATE_REPRESENTATION_MACRO,
            candidates=parent.candidates,
            source_parent_labels_sha256=(
                parent.receipt.ordered_labels_sha256
            ),
        ),
        metadata={
            "parent_inventory": parent.receipt.to_dict(),
            "sector_audit": dict(audit),
            "admission_cardinality": 1,
            "candidate_semantics": (
                "one_symmetry_complete_generator_per_round_v1"
            ),
            "raw_single_pauli_children_rejected": True,
        },
    )


def _clone_pauli_term(term: PauliTerm) -> PauliTerm:
    return PauliTerm(
        int(term.nqubit()),
        pc=complex(term.p_coeff),
        ps=str(term.pw2strng()),
    )


def _pauli_flip_support(word: str) -> tuple[int, ...]:
    normalized = str(word).strip().lower().replace("i", "e")
    unsupported = sorted(set(normalized) - {"e", "x", "y", "z"})
    if unsupported:
        raise ValueError(
            "H2O sector-block construction received unsupported Pauli "
            f"symbols {unsupported!r}."
        )
    return tuple(
        index
        for index, symbol in enumerate(normalized)
        if symbol in {"x", "y"}
    )


def _sector_block_identity(
    serialized_terms_exyz: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    digest = canonical_sha256(
        {
            "schema": H2O_SECTOR_COMPLETE_PAULI_BLOCK_IDENTITY_SCHEMA,
            "serialized_terms_exyz": [
                dict(row) for row in serialized_terms_exyz
            ],
        }
    )
    return f"h2o_sector_block::{digest[:16]}", f"h2o-sector-block:{digest[:16]}"


def build_h2o_sector_complete_pauli_block_pool(
    problem: ResolvedProblemContext,
    *,
    retained_parents: Sequence[CandidateRecord],
) -> CandidateInventory:
    """Split retained H2O parents only at fixed-sector-safe boundaries.

    A Pauli word that commutes with every declared fixed-count operator is a
    valid singleton block.  Remaining words are grouped by their complete
    X/Y flip support.  Distinct flip supports map a computational basis state
    to distinct output states, so fixed-sector leakage cannot cancel between
    them.  Because each source parent is sector preserving, every resulting
    same-flip-support block must therefore be sector preserving as a grouped
    generator; the executable audit below enforces that argument fail closed.
    """

    require_h2o_symmetry_complete_problem(problem)
    source = tuple(retained_parents)
    if not source:
        raise ValueError(
            "H2O sector-complete child exposure requires retained parents."
        )
    if any(
        candidate.representation_id != CANDIDATE_REPRESENTATION_MACRO
        for candidate in source
    ):
        raise ValueError(
            "H2O sector-complete child exposure requires macro parent "
            "records."
        )

    parent_audit = audit_candidate_pool_sector_contract(
        [candidate.term for candidate in source],
        resolved_problem=problem,
    )
    if not bool(parent_audit.get("passed")) or not bool(
        parent_audit.get("execution_passed")
    ):
        raise RuntimeError(
            "H2O sector-block construction received a nonphysical parent "
            f"shortlist: {parent_audit!r}."
        )

    groups, unsupported = resolve_fixed_count_qubit_groups(problem)
    if unsupported or not groups:
        raise RuntimeError(
            "H2O sector-block construction requires complete fixed-count "
            f"qubit groups; unsupported={unsupported!r}."
        )

    drafts: dict[str, dict[str, Any]] = {}
    source_labels: dict[str, str] = {
        str(candidate.generator_identity): str(candidate.label)
        for candidate in source
    }
    individually_safe_word_count = 0
    cancellation_block_count = 0

    for parent in source:
        safe_components: list[PauliTerm] = []
        unsafe_by_flip_support: dict[tuple[int, ...], list[PauliTerm]] = {}
        for component in parent.term.polynomial.return_polynomial():
            cloned = _clone_pauli_term(component)
            singleton = AnsatzTerm(
                label="h2o_sector_singleton_probe",
                polynomial=PauliPolynomial("JW", [cloned]),
                execution_mode="termwise_product",
            )
            singleton_audit = audit_generator_sector_contract(
                singleton,
                groups=groups,
                total_qubits=int(problem.layout.total_qubits),
            )
            if bool(singleton_audit["execution_preserves_fixed_counts"]):
                safe_components.append(cloned)
            else:
                support = _pauli_flip_support(str(cloned.pw2strng()))
                unsafe_by_flip_support.setdefault(support, []).append(cloned)

        component_groups: list[tuple[list[PauliTerm], bool]] = [
            ([component], True) for component in safe_components
        ]
        component_groups.extend(
            (components, False)
            for _, components in sorted(
                unsafe_by_flip_support.items(),
                key=lambda item: item[0],
            )
        )
        for components, individually_safe in component_groups:
            execution_mode = (
                "termwise_product"
                if individually_safe and len(components) == 1
                else "grouped_exact"
            )
            polynomial = PauliPolynomial(
                "JW", [_clone_pauli_term(term) for term in components]
            )
            serialized = tuple(
                dict(row)
                for row in serialize_polynomial_terms_exyz(polynomial)
            )
            if not serialized:
                continue
            label, generator_identity = _sector_block_identity(serialized)
            block_term = AnsatzTerm(
                label=label,
                polynomial=polynomial,
                execution_mode=execution_mode,
            )
            block_audit = audit_generator_sector_contract(
                block_term,
                groups=groups,
                total_qubits=int(problem.layout.total_qubits),
            )
            if not bool(block_audit["grouped_preserves_fixed_counts"]) or not bool(
                block_audit["execution_preserves_fixed_counts"]
            ):
                raise RuntimeError(
                    "Same-flip-support H2O block failed its fixed-sector "
                    f"audit for parent {parent.label!r}: {block_audit!r}."
                )

            manifest_key = canonical_sha256(
                {
                    "serialized_terms_exyz": [dict(row) for row in serialized],
                    "execution_mode": execution_mode,
                }
            )
            draft = drafts.get(manifest_key)
            if draft is None:
                draft = {
                    "label": label,
                    "term": block_term,
                    "generator_identity": generator_identity,
                    "parent_identities": [],
                    "family_id": str(parent.family_id),
                    "stage_family": str(parent.stage_family),
                    "construction": (
                        "h2o_fixed_sector_pauli_block_from_retained_parent_v1"
                    ),
                    "execution_mode": execution_mode,
                    "serialized_terms_exyz": serialized,
                    "symmetry_receipt": block_audit,
                    "generator_metadata": {
                        "generator_id": generator_identity,
                        "generator_identity_schema": (
                            H2O_SECTOR_COMPLETE_PAULI_BLOCK_IDENTITY_SCHEMA
                        ),
                        "sector_complete_pauli_block": {
                            "schema": (
                                H2O_SECTOR_COMPLETE_PAULI_BLOCK_POOL_SCHEMA
                            ),
                            "grouping_policy": (
                                "individually_safe_else_complete_xy_flip_support_v1"
                            ),
                            "component_count": int(len(serialized)),
                            "individually_sector_preserving": bool(
                                individually_safe
                            ),
                            "raw_single_pauli_child": bool(
                                individually_safe and len(serialized) == 1
                            ),
                        },
                    },
                }
                drafts[manifest_key] = draft
                if individually_safe:
                    individually_safe_word_count += 1
                else:
                    cancellation_block_count += 1
            parent_identities = draft["parent_identities"]
            parent_identity = str(parent.generator_identity)
            if parent_identity not in parent_identities:
                parent_identities.append(parent_identity)

    records: list[CandidateRecord] = []
    for draft in drafts.values():
        parent_identities = tuple(
            str(value) for value in draft["parent_identities"]
        )
        metadata = dict(draft["generator_metadata"])
        metadata["parent_generator_ids"] = list(parent_identities)
        metadata["parent_labels"] = [
            source_labels[parent_identity]
            for parent_identity in parent_identities
        ]
        records.append(
            CandidateRecord(
                label=str(draft["label"]),
                term=draft["term"],
                representation_id=CANDIDATE_REPRESENTATION_MACRO,
                generator_identity=str(draft["generator_identity"]),
                parent_identities=parent_identities,
                family_id=str(draft["family_id"]),
                stage_family=str(draft["stage_family"]),
                construction=str(draft["construction"]),
                execution_mode=str(draft["execution_mode"]),
                serialized_terms_exyz=tuple(
                    dict(row) for row in draft["serialized_terms_exyz"]
                ),
                symmetry_receipt=dict(draft["symmetry_receipt"]),
                generator_metadata=metadata,
            )
        )

    if not records:
        raise RuntimeError(
            "H2O sector-complete child exposure produced no candidates."
        )
    final_audit = audit_candidate_pool_sector_contract(
        [candidate.term for candidate in records],
        resolved_problem=problem,
    )
    if not bool(final_audit.get("passed")) or not bool(
        final_audit.get("execution_passed")
    ):
        raise RuntimeError(
            "H2O sector-complete child inventory failed its final executable "
            f"audit: {final_audit!r}."
        )

    return CandidateInventory(
        candidates=tuple(records),
        receipt=_receipt(
            schema=H2O_SECTOR_COMPLETE_PAULI_BLOCK_POOL_SCHEMA,
            representation_id=CANDIDATE_REPRESENTATION_MACRO,
            candidates=records,
            source_parent_labels_sha256=canonical_sha256(
                [candidate.label for candidate in source]
            ),
        ),
        metadata={
            "exposure_scope": "ra_retained_parent_shortlist_v1",
            "exposure_policy": (
                "sector_complete_pauli_blocks_from_retained_parents_v1"
            ),
            "grouping_policy": (
                "individually_safe_else_complete_xy_flip_support_v1"
            ),
            "source_parent_count": int(len(source)),
            "source_parent_labels": [
                str(candidate.label) for candidate in source
            ],
            "source_parent_generator_identities": [
                str(candidate.generator_identity) for candidate in source
            ],
            "individually_safe_word_count": int(
                individually_safe_word_count
            ),
            "cancellation_block_count": int(cancellation_block_count),
            "candidate_count": int(len(records)),
            "raw_unsafe_single_pauli_words_rejected": True,
            "parent_sector_audit": dict(parent_audit),
            "final_sector_audit": dict(final_audit),
        },
    )
def _shared_parent(candidate: CandidateRecord) -> SharedPauliPoolParent:
    return SharedPauliPoolParent(
        label=str(candidate.label),
        polynomial=candidate.term.polynomial,
        family_id=str(candidate.family_id),
        stage_family=str(candidate.stage_family),
        construction=str(candidate.construction),
        execution_mode=str(candidate.execution_mode),
        symmetry_spec=(
            None
            if candidate.symmetry_receipt is None
            else dict(candidate.symmetry_receipt)
        ),
        generator_metadata=_generator_metadata(candidate.term),
    )


def guarded_singleton_generator_identity(
    *,
    label: str,
    serialized_terms_exyz: Sequence[Mapping[str, Any]],
) -> str:
    """Identify a Pauli direction independently of its staged parent supply."""

    return (
        "child:"
        + canonical_sha256(
            {
                "schema": (
                    GUARDED_SINGLETON_GENERATOR_IDENTITY_SCHEMA
                ),
                "representation_id": (
                    CANDIDATE_REPRESENTATION_SINGLE_PAULI
                ),
                "label": str(label),
                "serialized_terms_exyz": [
                    dict(row)
                    for row in serialized_terms_exyz
                ],
            }
        )[:16]
    )


def _build_guarded_single_pauli_pool(
    problem: ResolvedProblemContext,
    *,
    retained_parents: Sequence[CandidateRecord] | None = None,
    paper_i_l3_page12_application: bool,
    paper_i_pure_hubbard_noise_page12_application: bool = False,
) -> CandidateInventory:
    application_identity = (
        require_paper_i_l3_page12_problem(problem)
        if paper_i_l3_page12_application
        else None
    )
    spec = _parent_pool_spec(
        problem,
        paper_i_l3_page12_application=paper_i_l3_page12_application,
        paper_i_pure_hubbard_noise_page12_application=(
            paper_i_pure_hubbard_noise_page12_application
        ),
    )
    parent = _build_parent_template_inventory(
        problem,
        representation_id=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        paper_i_l3_page12_application=paper_i_l3_page12_application,
        paper_i_pure_hubbard_noise_page12_application=(
            paper_i_pure_hubbard_noise_page12_application
        ),
    )
    source = (
        tuple(parent.candidates)
        if retained_parents is None
        else tuple(retained_parents)
    )
    if not source:
        raise ValueError("Guarded singleton exposure requires parent templates.")
    source_identity_by_label: dict[str, str] = {}
    for source_candidate in source:
        label = str(source_candidate.label)
        identity = str(source_candidate.generator_identity)
        previous = source_identity_by_label.setdefault(label, identity)
        if previous != identity:
            raise ValueError(
                "Guarded singleton exposure found one parent label with "
                "multiple generator identities."
            )
    result = build_shared_pauli_child_pool(
        parents=tuple(_shared_parent(candidate) for candidate in source),
        mode=SHARED_PAULI_POOL_MODE_GUARDED_SINGLETON_CHILDREN_ONLY_V1,
        symmetry_policy="hard_guard",
        max_subset_size=1,
        subset_sizes=(1,),
        problem_key=str(spec["problem_key"]),
        num_sites=int(spec["num_sites"]),
        ordering=str(problem.request.ordering),
        qpb=int(
            (int(problem.layout.total_qubits) - int(problem.layout.fermion_qubits))
            // int(spec["boson_mode_count"])
        ),
        n_ph_max=int(problem.request.n_ph_max),
        boson_encoding=str(problem.request.boson_encoding),
        total_register_width=int(problem.layout.total_qubits),
        fixed_num_particles=tuple(int(value) for value in problem.sector.num_particles),
        max_terms=None,
    )
    records: list[CandidateRecord] = []
    for candidate in result.candidates:
        term = AnsatzTerm(
            label=str(candidate.label),
            polynomial=candidate.polynomial,
            execution_mode=str(candidate.execution_mode),
        )
        parent_labels = tuple(
            str(value)
            for value in (
                candidate.parent_labels
                or (() if candidate.parent_label is None else (candidate.parent_label,))
            )
        )
        missing_parent_labels = tuple(
            label
            for label in parent_labels
            if label not in source_identity_by_label
        )
        if missing_parent_labels:
            raise RuntimeError(
                "Guarded singleton child lineage names parents outside the "
                f"source inventory: {missing_parent_labels!r}."
            )
        parent_identities = tuple(
            source_identity_by_label[label] for label in parent_labels
        )
        generator_identity = (
            guarded_singleton_generator_identity(
                label=str(candidate.label),
                serialized_terms_exyz=(
                    candidate.serialized_terms_exyz
                ),
            )
        )
        generator_metadata = dict(candidate.generator_metadata)
        generator_metadata["generator_id"] = generator_identity
        generator_metadata["generator_identity_schema"] = (
            GUARDED_SINGLETON_GENERATOR_IDENTITY_SCHEMA
        )
        records.append(
            CandidateRecord(
                label=str(candidate.label),
                term=term,
                representation_id=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
                generator_identity=generator_identity,
                parent_identities=parent_identities,
                family_id=str(candidate.family_id),
                stage_family=str(candidate.stage_family),
                construction=str(candidate.construction),
                execution_mode=str(candidate.execution_mode),
                serialized_terms_exyz=tuple(
                    dict(row) for row in candidate.serialized_terms_exyz
                ),
                symmetry_receipt=(
                    None
                    if candidate.symmetry_gate is None
                    else dict(candidate.symmetry_gate)
                ),
                generator_metadata=generator_metadata,
            )
        )
    receipt = _receipt(
        schema=GUARDED_SINGLETON_POOL_SCHEMA,
        representation_id=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
        candidates=records,
        source_parent_labels_sha256=canonical_sha256(
            [candidate.label for candidate in source]
        ),
    )
    inventory = CandidateInventory(
        candidates=tuple(records),
        receipt=receipt,
        metadata={
            "exposure_scope": (
                "global_parent_inventory_v1"
                if retained_parents is None
                else "ra_retained_parent_shortlist_v1"
            ),
            "source_parent_count": int(len(source)),
            "source_parent_labels": [
                str(candidate.label) for candidate in source
            ],
            "source_parent_generator_identities": [
                str(candidate.generator_identity) for candidate in source
            ],
            "shared_pool_meta": dict(result.meta),
            "shared_pool_manifest": dict(result.manifest),
        },
    )
    nph = int(problem.request.n_ph_max)
    if paper_i_l3_page12_application and retained_parents is None:
        expected_singleton_count = (
            PAPER_I_L3_PAGE12_SINGLETON_COUNT
            if nph == 1
            else PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_SINGLETON_COUNT
        )
        expected_singleton_labels = (
            PAPER_I_L3_PAGE12_SINGLETON_LABELS_SHA256
            if nph == 1
            else PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_SINGLETON_LABELS_SHA256
        )
        expected_singleton_pool = (
            PAPER_I_L3_PAGE12_SINGLETON_POOL_SHA256
            if nph == 1
            else str(
                PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_POOL_LOCKS[
                    str(application_identity)
                ]["singleton_pool_sha256"]
            )
        )
        if (
            inventory.receipt.count != expected_singleton_count
            or inventory.receipt.ordered_labels_sha256
            != expected_singleton_labels
            or inventory.receipt.ordered_pool_sha256
            != expected_singleton_pool
        ):
            raise RuntimeError(
                "The named Page-12 L=3 guarded-singleton inventory drifted "
                "from its locked count, ordering, or generator bytes."
            )
    return inventory


def build_guarded_single_pauli_pool(
    problem: ResolvedProblemContext,
    *,
    retained_parents: Sequence[CandidateRecord] | None = None,
) -> CandidateInventory:
    """Build ordinary L=2 or named H2O guarded single-Pauli children."""

    return _build_guarded_single_pauli_pool(
        problem,
        retained_parents=retained_parents,
        paper_i_l3_page12_application=False,
    )


def build_paper_i_l3_page12_guarded_single_pauli_pool(
    problem: ResolvedProblemContext,
    *,
    retained_parents: Sequence[CandidateRecord] | None = None,
) -> CandidateInventory:
    """Build the locked global or retained L=3 Page-12 singleton pool."""

    return _build_guarded_single_pauli_pool(
        problem,
        retained_parents=retained_parents,
        paper_i_l3_page12_application=True,
    )


def build_paper_i_pure_hubbard_noise_page12_guarded_single_pauli_pool(
    problem: ResolvedProblemContext,
    *,
    retained_parents: Sequence[CandidateRecord] | None = None,
) -> CandidateInventory:
    """Build the named pure-Hubbard global guarded-singleton inventory."""

    return _build_guarded_single_pauli_pool(
        problem,
        retained_parents=retained_parents,
        paper_i_l3_page12_application=False,
        paper_i_pure_hubbard_noise_page12_application=True,
    )


def build_staged_single_pauli_pool(
    problem: ResolvedProblemContext,
    *,
    retained_parents: Sequence[CandidateRecord],
) -> CandidateInventory:
    """Expose canonical unit-Pauli children after the RA parent shortlist.

    Staging changes only the parent supply.  The canonical child construction
    remains the same global hard-guarded unit-Pauli construction used by
    Append-ADAPT, including identity removal and cross-parent
    canonicalization/deduplication.
    """

    _require_supported_singleton_problem(problem)
    source = tuple(retained_parents)
    if not source:
        raise ValueError("Staged singleton exposure requires retained parents.")
    if any(
        candidate.representation_id
        != CANDIDATE_REPRESENTATION_SINGLE_PAULI
        for candidate in source
    ):
        raise ValueError(
            "Staged singleton exposure received a non-singleton parent."
        )
    guarded = build_guarded_single_pauli_pool(
        problem,
        retained_parents=source,
    )
    records = guarded.candidates
    return CandidateInventory(
        candidates=records,
        receipt=_receipt(
            schema=STAGED_SINGLETON_POOL_SCHEMA,
            representation_id=CANDIDATE_REPRESENTATION_SINGLE_PAULI,
            candidates=records,
            source_parent_labels_sha256=canonical_sha256(
                [candidate.label for candidate in source]
            ),
        ),
        metadata={
            "exposure_scope": "ra_retained_parent_shortlist_v1",
            "exposure_policy": (
                "split_guard_project_canonicalize_dedupe_"
                "across_retained_parents_v1"
            ),
            "source_parent_count": int(len(source)),
            "source_parent_labels": [
                str(candidate.label) for candidate in source
            ],
            "source_parent_generator_identities": [
                str(candidate.generator_identity) for candidate in source
            ],
            "canonical_child_pool": guarded.receipt.to_dict(),
            "canonical_child_pool_metadata": dict(guarded.metadata),
        },
    )


__all__ = [
    "CandidateInventory",
    "CandidateRecord",
    "EXPECTED_EXECUTABLE_MACRO_COUNTS",
    "EXPECTED_H2O_DERIVATIVE_RESOLVED_PARENT_COUNT",
    "EXPECTED_MACRO_REMOVED_COUNTS",
    "EXPECTED_PARENT_COUNTS",
    "H2O_DERIVATIVE_RESOLVED_POOL",
    "H2O_LINEAR_FD_FAMILY",
    "H2O_SECTOR_COMPLETE_PAULI_BLOCK_IDENTITY_SCHEMA",
    "H2O_SECTOR_COMPLETE_PAULI_BLOCK_POOL_SCHEMA",
    "H2O_SYMMETRY_COMPLETE_POOL_SCHEMA",
    "GUARDED_SINGLETON_GENERATOR_IDENTITY_SCHEMA",
    "PAPER_I_L3_PAGE12_APPLICATION_ID",
    "PAPER_I_L3_PAGE12_HAMILTONIAN_TERMS_SHA256",
    "PAPER_I_L3_PAGE12_PARENT_COUNT",
    "PAPER_I_L3_PAGE12_PARENT_LABELS_SHA256",
    "PAPER_I_L3_PAGE12_PARENT_POOL_SHA256",
    "PAPER_I_L3_PAGE12_PROBLEM_REQUEST_SHA256",
    "PAPER_I_L3_PAGE12_SINGLETON_COUNT",
    "PAPER_I_L3_PAGE12_SINGLETON_LABELS_SHA256",
    "PAPER_I_L3_PAGE12_SINGLETON_POOL_SHA256",
    "PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_PARENT_COUNT",
    "PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_PARENT_LABELS_SHA256",
    "PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_PARENT_POOL_SHA256",
    "PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_POOL_LOCKS",
    "PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_PROBLEM_LOCKS",
    "PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_SINGLETON_COUNT",
    "PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_SINGLETON_LABELS_SHA256",
    "PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_SINGLETON_POOL_SHA256",
    "PAPER_I_PURE_HUBBARD_NOISE_PAGE12_APPLICATION_ID",
    "build_candidate_inventory_lineage_receipt",
    "build_executable_macro_pool",
    "build_guarded_single_pauli_pool",
    "build_h2o_sector_complete_pauli_block_pool",
    "build_h2o_symmetry_complete_generator_pool",
    "build_paper_i_l3_page12_guarded_single_pauli_pool",
    "build_paper_i_l3_page12_parent_template_inventory",
    "build_paper_i_pure_hubbard_noise_page12_guarded_single_pauli_pool",
    "build_paper_i_pure_hubbard_noise_page12_parent_template_inventory",
    "build_staged_single_pauli_pool",
    "build_parent_template_inventory",
    "guarded_singleton_generator_identity",
    "require_paper_i_l3_page12_problem",
    "require_paper_i_pure_hubbard_noise_page12_problem",
]
