#!/usr/bin/env python3
"""Benchmark algorithm applicability registry.

This module is deliberately sidecar-only: it does not run physics kernels and it
must not change production ADAPT/controller behavior.  Its job is to answer the
question "is this algorithm/family pair valid and implemented enough to submit
as a batch job?" before CHTC spends a slot on it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

from pipelines.exact_bench.comparator_provenance import (
    ExecutionSurfaceRole,
    maybe_comparator_source_profile,
)
from pipelines.static_adapt.builders.problem_registry import (
    available_problem_keys,
    get_problem_family_spec,
)

BenchmarkDomain = Literal["static", "dynamics"]
ApplicabilityStatus = Literal[
    "runnable",
    "skipped_unsupported",
    "skipped_not_implemented",
    "skipped_no_runner",
]


@dataclass(frozen=True)
class BenchmarkAlgorithm:
    algorithm_id: str
    domain: BenchmarkDomain
    display_name: str
    method_family: str
    supported_families: tuple[str, ...] | None = None
    required_capabilities: tuple[str, ...] = ()
    required_pool_key: str | None = None
    implemented_families: tuple[str, ...] | None = None
    runner_module: str | None = None
    qpu_faithful: bool | None = None
    exact_assisted: bool = False
    diagnostic: bool = False
    hamiltonian_generic: bool = False
    algorithm_origin: str | None = None
    execution_surface: str | None = None
    execution_surface_role: ExecutionSurfaceRole | None = None
    external_reference_status: ExecutionSurfaceRole | None = None
    external_reference_id: str | None = None
    parity_reference_algorithm_id: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class AlgorithmApplicability:
    family: str
    algorithm_id: str
    domain: BenchmarkDomain
    status: ApplicabilityStatus
    reason: str
    runner_module: str | None
    qpu_faithful: bool | None
    exact_assisted: bool
    diagnostic: bool
    hamiltonian_generic: bool
    required_pool_key: str | None = None
    resolved_pool_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_STATIC_HH_RUNNER = "pipelines.exact_bench.generic_static_benchmark"
_EXTERNAL_STATIC_ADAPT_RUNNER = "pipelines.exact_bench.external_adapt.external_static_adapt_benchmark"
_DYNAMICS_HH_RUNNER = "pipelines.time_dynamics.tables.generic_dynamics_benchmark"
_GENERIC_STATIC_ED_REFERENCE_FAMILIES = (
    "hubbard",
    "ionic_hubbard",
    "extended_hubbard",
    "ttprime_hubbard",
    "spinless_tv",
    "spin_boson",
    "bose_hubbard",
    "harmonic_kerr_chain",
    "molecular_vibronic_h2",
)
_STATIC_HEA_QISKIT_TABLE_I_FAMILIES = (
    "hh",
    "hubbard",
    "ionic_hubbard",
    "extended_hubbard",
    "ttprime_hubbard",
    "spinless_tv",
    "spin_boson",
    "bose_hubbard",
    "harmonic_kerr_chain",
    "molecular_vibronic_h2",
)
_STATIC_FAMILY_INFORMED_VQE_FAMILIES = _STATIC_HEA_QISKIT_TABLE_I_FAMILIES
_GENERIC_QISKIT_ADAPTVQE_TABLE_I_FAMILIES = (
    "hh",
    "hubbard",
    "ionic_hubbard",
    "extended_hubbard",
    "ttprime_hubbard",
    "spinless_tv",
    "spin_boson",
    "bose_hubbard",
    "harmonic_kerr_chain",
    "molecular_vibronic_h2",
)
_GENERIC_STATIC_ADAPT_VARIANT_TABLE_I_FAMILIES = _GENERIC_QISKIT_ADAPTVQE_TABLE_I_FAMILIES
_GENERIC_DYNAMICS_FIRST_SLICE_FAMILIES = (
    "hh",
    "hubbard",
    "ionic_hubbard",
    "extended_hubbard",
    "ttprime_hubbard",
    "spinless_tv",
    "spin_boson",
    "bose_hubbard",
    "harmonic_kerr_chain",
    "molecular_vibronic_h2",
)
_GENERIC_DYNAMICS_CONTROLLER_ABLATION_FAMILIES = _GENERIC_DYNAMICS_FIRST_SLICE_FAMILIES
_GENERIC_DYNAMICS_AVQDS_T_FIXTURE_FAMILIES = (
    *_GENERIC_DYNAMICS_FIRST_SLICE_FAMILIES,
)


def _source_kwargs(algorithm_id: str) -> dict[str, Any]:
    """Registry-facing subset of Paper-I comparator source metadata."""
    profile = maybe_comparator_source_profile(algorithm_id)
    if profile is None:
        return {}
    return {
        "algorithm_origin": profile.algorithm_origin,
        "execution_surface": profile.execution_surface,
        "execution_surface_role": profile.execution_surface_role,
        "external_reference_status": profile.external_reference_status,
        "external_reference_id": profile.external_reference_id,
        "parity_reference_algorithm_id": profile.parity_reference_algorithm_id,
    }


_DEFAULT_ALGORITHMS: tuple[BenchmarkAlgorithm, ...] = (
    # Static, broadly meaningful families.  The generic ED row exposes existing
    # exact-target machinery for non-HH canonical static cases; HH-specific
    # static matrix rows remain separate.
    BenchmarkAlgorithm(
        algorithm_id="static_ed_reference",
        domain="static",
        display_name="Exact diagonalization / reference",
        method_family="classical_reference",
        supported_families=None,
        implemented_families=_GENERIC_STATIC_ED_REFERENCE_FAMILIES,
        runner_module=_STATIC_HH_RUNNER,
        qpu_faithful=False,
        exact_assisted=True,
        diagnostic=True,
        hamiltonian_generic=True,
        notes=(
            "Benchmark-local generic exact-target reference row over "
            "ResolvedProblemContext.exact_target.resolve_energy(...); HH remains "
            "on the existing HH-specific static benchmark surface."
        ),
    ),
    BenchmarkAlgorithm(
        algorithm_id="static_family_native_adapt_phase3",
        domain="static",
        display_name="Resource-adaptive Phase3 ADAPT scaffold/controller",
        method_family="adapt",
        supported_families=None,
        required_pool_key="full_meta",
        implemented_families=None,
        runner_module=_STATIC_HH_RUNNER,
        hamiltonian_generic=True,
        **_source_kwargs("static_family_native_adapt_phase3"),
        notes="Generic static_adapt Phase3 runner over the problem-local full_meta pool.",
    ),
    BenchmarkAlgorithm(
        algorithm_id="static_append_only_adapt_phase3",
        domain="static",
        display_name="Append-only Phase3 ADAPT limit",
        method_family="adapt_limit",
        supported_families=None,
        required_pool_key="full_meta",
        implemented_families=None,
        runner_module=_STATIC_HH_RUNNER,
        hamiltonian_generic=True,
        notes="Generic static_adapt Phase3 runner with append-only insertion/reoptimization controls.",
    ),
    BenchmarkAlgorithm(
        algorithm_id="static_hva_vqe",
        domain="static",
        display_name="Hamiltonian variational ansatz VQE",
        method_family="fixed_ansatz_vqe",
        supported_families=None,
        required_pool_key="hva",
        implemented_families=("hh",),
        runner_module=_STATIC_HH_RUNNER,
        hamiltonian_generic=False,
        notes="Current concrete HVA VQE benchmark implementation is HH-specific.",
    ),
    BenchmarkAlgorithm(
        algorithm_id="static_hea_qiskit_vqe",
        domain="static",
        display_name="Qiskit hardware-efficient ansatz VQE",
        method_family="fixed_ansatz_vqe",
        supported_families=None,
        implemented_families=_STATIC_HEA_QISKIT_TABLE_I_FAMILIES,
        runner_module=_STATIC_HH_RUNNER,
        hamiltonian_generic=True,
        **_source_kwargs("static_hea_qiskit_vqe"),
        notes=(
            "Exact-bench Qiskit HEA VQE row over the canonical Paper-I Table-I "
            "Hamiltonian cases. Fixed-count "
            "and truncated-boson leakage diagnostics are reporting-only after optimization."
        ),
    ),
    BenchmarkAlgorithm(
        algorithm_id="static_family_informed_vqe",
        domain="static",
        display_name="Family-informed fixed VQE",
        method_family="family_informed_fixed_ansatz_vqe",
        supported_families=None,
        required_pool_key="full_meta",
        implemented_families=_STATIC_FAMILY_INFORMED_VQE_FAMILIES,
        runner_module=_STATIC_HH_RUNNER,
        hamiltonian_generic=True,
        **_source_kwargs("static_family_informed_vqe"),
        notes=(
            "Exact-bench-local second fixed-ansatz baseline. Family policies are "
            "UCCSD-style for fermionic/molecular "
            "families, quadrature/HVA-style for bosonic families, and a "
            "Lang-Firsov/UCCSD/quadrature-inspired hybrid for mixed fermion-boson "
            "families, implemented as a fixed family-prioritized subset of the "
            "problem-local full_meta pool."
        ),
    ),
    BenchmarkAlgorithm(
        algorithm_id="static_qiskit_adapt_vqe",
        domain="static",
        display_name="Qiskit AdaptVQE append-only ADAPT reference",
        method_family="library_adapt_reference",
        supported_families=None,
        required_pool_key="full_meta",
        implemented_families=_GENERIC_QISKIT_ADAPTVQE_TABLE_I_FAMILIES,
        runner_module=_STATIC_HH_RUNNER,
        exact_assisted=False,
        diagnostic=False,
        hamiltonian_generic=True,
        **_source_kwargs("static_qiskit_adapt_vqe"),
        notes=(
            "Exact-bench-only Qiskit Algorithms AdaptVQE row over the canonical "
            "Paper-I Table-I Hamiltonian cases. The operator pool is the "
            "problem-local full_meta pool converted into Qiskit SparsePauliOps; exact "
            "references are reporting-only after optimization. This row does not "
            "call Phase3/SNAKE/static_adapt controller code and must not be used "
            "as a CEO/TETRIS/QEB/Geo emulation."
        ),
    ),
    BenchmarkAlgorithm(
        algorithm_id="static_full_meta_append_adapt_vqe",
        domain="static",
        display_name="Append-only ADAPT-VQE (local full_meta)",
        method_family="full_meta_append_only_adapt",
        supported_families=None,
        required_pool_key="full_meta",
        implemented_families=_GENERIC_STATIC_ADAPT_VARIANT_TABLE_I_FAMILIES,
        runner_module=_STATIC_HH_RUNNER,
        exact_assisted=False,
        diagnostic=False,
        hamiltonian_generic=True,
        **_source_kwargs("static_full_meta_append_adapt_vqe"),
        notes=(
            "Exact-bench-local append-only ADAPT-VQE comparator over the problem-local "
            "full_meta pool. Each iteration selects the single largest absolute raw "
            "ADAPT commutator gradient from the undrained pool, appends the candidate "
            "with replacement, and refits the full ansatz with Powell. Paper-I rows use "
            "an explicit fixed iteration horizon. Exact references are reporting-only "
            "after optimization; decision-noise support is provided through local "
            "selector/refit surfaces. This row does not call Phase3/SNAKE/static_adapt "
            "controller code."
        ),
    ),
    BenchmarkAlgorithm(
        algorithm_id="static_qubit_qeb_adapt_vqe",
        domain="static",
        display_name="Qubit/QEB-ADAPT-VQE",
        method_family="qubit_excitation_adapt",
        supported_families=None,
        implemented_families=_GENERIC_STATIC_ADAPT_VARIANT_TABLE_I_FAMILIES,
        runner_module=_STATIC_HH_RUNNER,
        exact_assisted=False,
        diagnostic=False,
        hamiltonian_generic=True,
        **_source_kwargs("static_qubit_qeb_adapt_vqe"),
        notes=(
            "Benchmark-local generic statevector ADAPT row over the canonical "
            "Paper-I Table-I Hamiltonian cases. Uses QEB singles and doubles expanded "
            "into repo exyz Pauli words and resolves exact references only after "
            "optimization. This is an operator-class comparator, not a full_meta "
            "same-pool controller comparison. Does not call Phase3/SNAKE/static_adapt "
            "controller code."
        ),
    ),
    BenchmarkAlgorithm(
        algorithm_id="static_geo_qubit_adapt_vqe",
        domain="static",
        display_name="legacy geometry diagnostic (removed from Table I)",
        method_family="metric_aware_full_meta_adapt",
        supported_families=None,
        required_pool_key="full_meta",
        implemented_families=_GENERIC_STATIC_ADAPT_VARIANT_TABLE_I_FAMILIES,
        runner_module=_STATIC_HH_RUNNER,
        exact_assisted=False,
        diagnostic=False,
        hamiltonian_generic=True,
        **_source_kwargs("static_geo_qubit_adapt_vqe"),
        notes=(
            "Benchmark-local full-meta Geo-style metric selector row over the "
            "problem-local full_meta pool. Candidate selection solves the projected "
            "tangent metric over the remaining pool and ranks by absolute "
            "natural-gradient step, but stopping and inner optimization remain the "
            "legacy raw-gradient/BFGS ADAPT path. This is not the faithful QEB "
            "Geo-ADAPT-VQE comparator. Exact references are reporting-only and "
            "Phase3/SNAKE controller code is not called."
        ),
    ),
    BenchmarkAlgorithm(
        algorithm_id="static_geo_qeb_adapt_vqe",
        domain="static",
        display_name="Geo-ADAPT-VQE (QEB reference)",
        method_family="geo_adapt_qeb_excitation",
        supported_families=None,
        implemented_families=_GENERIC_STATIC_ADAPT_VARIANT_TABLE_I_FAMILIES,
        runner_module=_STATIC_HH_RUNNER,
        exact_assisted=False,
        diagnostic=False,
        hamiltonian_generic=True,
        **_source_kwargs("static_geo_qeb_adapt_vqe"),
        notes=(
            "Exact-bench-local Geo-ADAPT-VQE QEB reference row. Uses the benchmark-local "
            "QEB singles/doubles excitation pool, projected Fubini-Study natural-gradient "
            "selection and stopping, with-replacement selection except immediate repeats, "
            "and a local QNGD-style inner optimizer with seeded energy-only SPSA fallback; "
            "no BFGS polish is used. This is retained for operator-class diagnostics and "
            "is not the Table-I same-pool Pos-Geo row."
        ),
    ),
    BenchmarkAlgorithm(
        algorithm_id="static_geo_adapt_vqe",
        domain="static",
        display_name="Geo-ADAPT-VQE",
        method_family="geo_adapt_full_meta_powell",
        supported_families=None,
        required_pool_key="full_meta",
        implemented_families=_GENERIC_STATIC_ADAPT_VARIANT_TABLE_I_FAMILIES,
        runner_module=_STATIC_HH_RUNNER,
        exact_assisted=False,
        diagnostic=False,
        hamiltonian_generic=True,
        **_source_kwargs("static_geo_adapt_vqe"),
        notes=(
            "Exact-bench-local Geo-ADAPT-VQE comparator row over the problem-local "
            "full_meta pool. Candidate selection solves the projected Fubini-Study "
            "tangent metric over the full undrained pool, ranks by absolute natural-gradient "
            "step, skips the append only when the immediately previous generator wins, and "
            "otherwise appends it. Every iteration refits the full ansatz with Powell; "
            "Paper-I rows use an explicit fixed iteration horizon. This is "
            "the Table-I same-pool Geo comparator; the older Pos-Geo insertion-refit "
            "QNGD row is retained only as a diagnostic."
        ),
    ),
    BenchmarkAlgorithm(
        algorithm_id="static_pos_geo_adapt_vqe",
        domain="static",
        display_name="Pos-Geo-ADAPT-VQE",
        method_family="pos_geo_adapt_full_meta",
        supported_families=None,
        required_pool_key="full_meta",
        implemented_families=_GENERIC_STATIC_ADAPT_VARIANT_TABLE_I_FAMILIES,
        runner_module=_STATIC_HH_RUNNER,
        exact_assisted=False,
        diagnostic=False,
        hamiltonian_generic=True,
        **_source_kwargs("static_pos_geo_adapt_vqe"),
        notes=(
            "Exact-bench-local Pos-Geo-ADAPT-VQE comparator row. Uses the problem-local "
            "full_meta pool where available, projected Fubini-Study natural-gradient "
            "selection and stopping, with-replacement selection except immediate repeats, "
            "and tests all insertion positions by local QNGD refit with seeded energy-only "
            "SPSA fallback if QNGD stalls. No BFGS polish is used. Exact references are "
            "reporting-only after optimization and Phase3/SNAKE/static_adapt controller code "
            "is not called."
        ),
    ),
    BenchmarkAlgorithm(
        algorithm_id="static_uccsd_vqe",
        domain="static",
        display_name="UCCSD / lifted UCCSD VQE",
        method_family="chemistry_vqe",
        supported_families=(
            "hubbard",
            "hh",
            "ionic_hubbard",
            "extended_hubbard",
            "ttprime_hubbard",
            "molecular_vibronic_h2",
        ),
        implemented_families=("hh",),
        runner_module=_STATIC_HH_RUNNER,
        hamiltonian_generic=False,
        notes="Meaningful for spinful fermion/molecular families; implemented benchmark row is HH lifted-UCCSD.",
    ),
    BenchmarkAlgorithm(
        algorithm_id="static_avqite_uccsd",
        domain="static",
        display_name="AVQITE on UCCSD/lifted-UCCSD pool",
        method_family="adaptive_imaginary_time",
        supported_families=(
            "hubbard",
            "hh",
            "ionic_hubbard",
            "extended_hubbard",
            "ttprime_hubbard",
            "molecular_vibronic_h2",
        ),
        implemented_families=("hh",),
        runner_module=_STATIC_HH_RUNNER,
        exact_assisted=False,
        diagnostic=True,
        hamiltonian_generic=False,
        notes="Current concrete benchmark row is HH UCCSD-lifted AVQITE.",
    ),
    BenchmarkAlgorithm(
        algorithm_id="static_ceo_adapt_phase3",
        domain="static",
        display_name="CEO ADAPT benchmark",
        method_family="adapt_selector_variant",
        supported_families=None,
        required_pool_key="__family_default__",
        implemented_families=("hubbard",),
        runner_module=_EXTERNAL_STATIC_ADAPT_RUNNER,
        hamiltonian_generic=False,
        **_source_kwargs("static_ceo_adapt_phase3"),
        notes=(
            "External CEO public-code row is wired only for the benchmark-local "
            "Hubbard L2 first slice via the pinned ceo_adapt_vqe checkout. Do "
            "not emulate it through the Phase3 controller."
        ),
    ),
    BenchmarkAlgorithm(
        algorithm_id="static_tetris_adapt_phase3",
        domain="static",
        display_name="TETRIS-ADAPT benchmark",
        method_family="adapt_selector_variant",
        supported_families=None,
        required_pool_key="__family_default__",
        implemented_families=("hubbard",),
        runner_module=_EXTERNAL_STATIC_ADAPT_RUNNER,
        hamiltonian_generic=False,
        **_source_kwargs("static_tetris_adapt_phase3"),
        notes=(
            "External TETRIS public-code row is wired only for the benchmark-local "
            "Hubbard L2 first slice via the pinned ceo_adapt_vqe checkout with "
            "LinAlgAdapt(tetris=True). Do not emulate it through the Phase3 controller."
        ),
    ),
    BenchmarkAlgorithm(
        algorithm_id="static_overlap_adapt_phase3",
        domain="static",
        display_name="Overlap-ADAPT benchmark",
        method_family="adapt_selector_variant",
        supported_families=None,
        required_pool_key="__family_default__",
        implemented_families=(),
        runner_module=_EXTERNAL_STATIC_ADAPT_RUNNER,
        hamiltonian_generic=False,
        **_source_kwargs("static_overlap_adapt_phase3"),
        notes=(
            "Overlap-ADAPT code is request-only in the catalog; keep this row "
            "skipped unless author code is obtained or a faithful "
            "reimplementation is explicitly labeled. Do not emulate it through "
            "the Phase3 controller."
        ),
    ),
    BenchmarkAlgorithm(
        algorithm_id="static_qeb_sq_lf_adapt",
        domain="static",
        display_name="QEB / SQ-LF ADAPT",
        method_family="electron_phonon_adapt",
        supported_families=("hh",),
        required_pool_key="sq_lf_std",
        implemented_families=("hh",),
        runner_module=_STATIC_HH_RUNNER,
        hamiltonian_generic=False,
        notes="HH/electron-phonon-specific pool; do not submit for pure Hubbard or boson chains.",
    ),
    BenchmarkAlgorithm(
        algorithm_id="static_lang_firsov_vqe",
        domain="static",
        display_name="Lang-Firsov SQ/LF VQE",
        method_family="electron_phonon_vqe",
        supported_families=("hh",),
        required_pool_key="sq_lf_std",
        implemented_families=("hh",),
        runner_module=_STATIC_HH_RUNNER,
        hamiltonian_generic=False,
        notes="HH/electron-phonon-specific; invalid for non-electron-phonon families.",
    ),
    BenchmarkAlgorithm(
        algorithm_id="static_qsci_sqd_sq_lf",
        domain="static",
        display_name="QSCI/SQD SQ-LF sampling diagonalization",
        method_family="sampling_subspace",
        supported_families=("hh",),
        required_pool_key="sq_lf_std",
        implemented_families=("hh",),
        runner_module=_STATIC_HH_RUNNER,
        hamiltonian_generic=False,
        notes="Current QSCI/SQD benchmark pool is HH SQ-LF.",
    ),
    # Dynamics.  Registry captures conceptual capability separately from concrete
    # row coverage.  Generic fixture-backed comparator rows are promoted only
    # when an honest row runner exists.
    BenchmarkAlgorithm(
        algorithm_id="dyn_exact_reference",
        domain="dynamics",
        display_name="Exact/Krylov reference dynamics",
        method_family="classical_reference",
        supported_families=None,
        required_capabilities=("supports_drive_benchmark_exact",),
        implemented_families=_GENERIC_DYNAMICS_FIRST_SLICE_FAMILIES,
        runner_module=_DYNAMICS_HH_RUNNER,
        qpu_faithful=False,
        exact_assisted=True,
        diagnostic=True,
        hamiltonian_generic=True,
        notes="First generic slice supports explicit fixture-backed realtime cases; other families remain skipped until cases/runners exist.",
    ),
    BenchmarkAlgorithm(
        algorithm_id="dyn_product_formula_envelope",
        domain="dynamics",
        display_name="Product-formula envelope",
        method_family="product_formula",
        supported_families=None,
        required_capabilities=("supports_drive_benchmark_exact",),
        implemented_families=_GENERIC_DYNAMICS_FIRST_SLICE_FAMILIES,
        runner_module=_DYNAMICS_HH_RUNNER,
        qpu_faithful=True,
        exact_assisted=False,
        diagnostic=True,
        hamiltonian_generic=True,
        notes="Fixture-backed cases use the same seed, drive waveform, time grid, observables, and compile proxy as the checkpoint-controller benchmark.",
    ),
    BenchmarkAlgorithm(
        algorithm_id="dyn_qdrift",
        domain="dynamics",
        display_name="qDRIFT/randomized product formula",
        method_family="randomized_product_formula",
        supported_families=None,
        required_capabilities=("supports_drive_benchmark_exact",),
        implemented_families=_GENERIC_DYNAMICS_FIRST_SLICE_FAMILIES,
        runner_module=_DYNAMICS_HH_RUNNER,
        qpu_faithful=True,
        exact_assisted=False,
        diagnostic=True,
        hamiltonian_generic=True,
        notes="Fixture-backed cases use the same seed, drive waveform, time grid, observables, and compile proxy as the checkpoint-controller benchmark.",
    ),
    BenchmarkAlgorithm(
        algorithm_id="dyn_fixed_mclachlan",
        domain="dynamics",
        display_name="Fixed-scaffold McLachlan",
        method_family="variational_dynamics",
        supported_families=None,
        required_capabilities=("supports_driven_realtime",),
        implemented_families=_GENERIC_DYNAMICS_FIRST_SLICE_FAMILIES,
        runner_module=_DYNAMICS_HH_RUNNER,
        qpu_faithful=True,
        exact_assisted=False,
        diagnostic=True,
        hamiltonian_generic=True,
        notes="First generic slice supports explicit fixture-backed neutral realtime cases; other families remain skipped until cases/runners exist.",
    ),
    BenchmarkAlgorithm(
        algorithm_id="dyn_fixed_pvqd",
        domain="dynamics",
        display_name="Fixed pVQD",
        method_family="variational_dynamics",
        supported_families=None,
        required_capabilities=("supports_driven_realtime",),
        implemented_families=_GENERIC_DYNAMICS_FIRST_SLICE_FAMILIES,
        runner_module=_DYNAMICS_HH_RUNNER,
        qpu_faithful=True,
        exact_assisted=False,
        diagnostic=True,
        hamiltonian_generic=True,
        notes="Fixture-backed cases use a repo-native fixed-pVQD comparator with product-formula target states; exact references are reporting-only.",
    ),
    BenchmarkAlgorithm(
        algorithm_id="dyn_adaptive_pvqd",
        domain="dynamics",
        display_name="Adaptive pVQD",
        method_family="adaptive_variational_dynamics",
        supported_families=None,
        required_capabilities=("supports_driven_realtime",),
        implemented_families=_GENERIC_DYNAMICS_FIRST_SLICE_FAMILIES,
        runner_module=_DYNAMICS_HH_RUNNER,
        qpu_faithful=True,
        exact_assisted=False,
        diagnostic=True,
        hamiltonian_generic=True,
        notes="Fixture-backed cases use a repo-native adaptive-pVQD comparator with product-formula target states; exact references are reporting-only.",
    ),
    BenchmarkAlgorithm(
        algorithm_id="dyn_qiskit_trotter_qrte",
        domain="dynamics",
        display_name="Qiskit TrotterQRTE",
        method_family="qiskit_community_time_evolver",
        supported_families=None,
        required_capabilities=("supports_driven_realtime",),
        implemented_families=_GENERIC_DYNAMICS_FIRST_SLICE_FAMILIES,
        runner_module=_DYNAMICS_HH_RUNNER,
        qpu_faithful=True,
        exact_assisted=False,
        diagnostic=True,
        hamiltonian_generic=True,
        algorithm_origin="pinned_qiskit_community_dynamics",
        execution_surface="pinned_qiskit_community_time_evolver",
        execution_surface_role="primary_execution_surface",
        notes=(
            "Primary Qiskit-community TrotterQRTE comparator row. This is distinct "
            "from repo-native product-formula rows and from Qiskit parity sidecars; "
            "exact references are reporting-only after the Qiskit trajectory is produced."
        ),
    ),
    BenchmarkAlgorithm(
        algorithm_id="dyn_qiskit_pvqd",
        domain="dynamics",
        display_name="Qiskit PVQD",
        method_family="qiskit_community_time_evolver",
        supported_families=None,
        required_capabilities=("supports_driven_realtime",),
        implemented_families=_GENERIC_DYNAMICS_FIRST_SLICE_FAMILIES,
        runner_module=_DYNAMICS_HH_RUNNER,
        qpu_faithful=True,
        exact_assisted=False,
        diagnostic=True,
        hamiltonian_generic=True,
        algorithm_origin="pinned_qiskit_community_dynamics",
        execution_surface="pinned_qiskit_community_time_evolver",
        execution_surface_role="primary_execution_surface",
        notes=(
            "Primary Qiskit-community PVQD comparator row. This is distinct from "
            "repo-native fixed/adaptive pVQD rows and from Qiskit parity sidecars; "
            "exact references are reporting-only after the Qiskit trajectory is produced."
        ),
    ),
    BenchmarkAlgorithm(
        algorithm_id="dyn_qiskit_varqrte",
        domain="dynamics",
        display_name="Qiskit VarQRTE",
        method_family="qiskit_community_time_evolver",
        supported_families=None,
        required_capabilities=("supports_driven_realtime",),
        implemented_families=_GENERIC_DYNAMICS_FIRST_SLICE_FAMILIES,
        runner_module=_DYNAMICS_HH_RUNNER,
        qpu_faithful=True,
        exact_assisted=False,
        diagnostic=True,
        hamiltonian_generic=True,
        algorithm_origin="pinned_qiskit_community_dynamics",
        execution_surface="pinned_qiskit_community_time_evolver",
        execution_surface_role="primary_execution_surface",
        notes=(
            "Primary Qiskit-community VarQRTE/RealMcLachlan comparator row. This is "
            "distinct from fixed-scaffold McLachlan and AP-McLachlan controller rows; "
            "exact references are reporting-only after the Qiskit trajectory is produced."
        ),
    ),
    BenchmarkAlgorithm(
        algorithm_id="dyn_avqds",
        domain="dynamics",
        display_name="AVQDS",
        method_family="adaptive_variational_dynamics",
        supported_families=None,
        required_capabilities=("supports_driven_realtime",),
        implemented_families=_GENERIC_DYNAMICS_FIRST_SLICE_FAMILIES,
        runner_module=_DYNAMICS_HH_RUNNER,
        qpu_faithful=True,
        exact_assisted=False,
        diagnostic=True,
        hamiltonian_generic=True,
        notes="Fixture-backed cases use a repo-native AVQDS RHS-tangent comparator; exact references are reporting-only.",
    ),
    BenchmarkAlgorithm(
        algorithm_id="dyn_avqds_t",
        domain="dynamics",
        display_name="PF-target adaptive tangent (diagnostic)",
        method_family="product_formula_target_tangent_diagnostic",
        supported_families=None,
        required_capabilities=("supports_driven_realtime",),
        implemented_families=_GENERIC_DYNAMICS_AVQDS_T_FIXTURE_FAMILIES,
        runner_module=_DYNAMICS_HH_RUNNER,
        qpu_faithful=True,
        exact_assisted=False,
        diagnostic=True,
        hamiltonian_generic=True,
        notes=(
            "Historical custom comparator that projects a product-formula target "
            "tangent. It is not published AVQDS(T), where T means TETRIS."
        ),
    ),
    BenchmarkAlgorithm(
        algorithm_id="dyn_avqds_tetris",
        domain="dynamics",
        display_name="AVQDS(T)",
        method_family="adaptive_variational_dynamics_tetris",
        supported_families=None,
        required_capabilities=("supports_driven_realtime",),
        implemented_families=_GENERIC_DYNAMICS_AVQDS_T_FIXTURE_FAMILIES,
        runner_module=_DYNAMICS_HH_RUNNER,
        qpu_faithful=True,
        exact_assisted=False,
        diagnostic=True,
        hamiltonian_generic=True,
        notes=(
            "Repo-native implementation of published AVQDS(T) Method 3: continuous "
            "McLachlan RHS, absolute-eigenvalue truncation, and greedy layers of "
            "score-ranked qubit-disjoint Pauli generators. Exact references are reporting-only."
        ),
    ),
    BenchmarkAlgorithm(
        algorithm_id="dyn_controller_full",
        domain="dynamics",
        display_name="Full strict checkpoint McLachlan controller",
        method_family="checkpoint_controller",
        supported_families=None,
        required_capabilities=("supports_driven_realtime",),
        implemented_families=_GENERIC_DYNAMICS_CONTROLLER_ABLATION_FAMILIES,
        runner_module=_DYNAMICS_HH_RUNNER,
        qpu_faithful=True,
        exact_assisted=False,
        diagnostic=True,
        hamiltonian_generic=True,
        notes=(
            "Table-I isolated row for the full strict ideal-observable checkpoint controller. "
            "Uses the controller-ablation runner with only the full-controller variant."
        ),
    ),
    BenchmarkAlgorithm(
        algorithm_id="dyn_controller_ablation_matrix",
        domain="dynamics",
        display_name="Generic checkpoint-controller ablation matrix",
        method_family="controller_ablation",
        supported_families=None,
        required_capabilities=("supports_driven_realtime",),
        implemented_families=_GENERIC_DYNAMICS_CONTROLLER_ABLATION_FAMILIES,
        runner_module=_DYNAMICS_HH_RUNNER,
        qpu_faithful=True,
        exact_assisted=False,
        diagnostic=True,
        hamiltonian_generic=True,
        notes=(
            "Runs non-HH generic realtime strict ideal-observable ablation variants: "
            "full controller, fixed scaffold, no append, no pruning, fixed integrator, "
            "and no compressed residual-split confirmation."
        ),
    ),
    BenchmarkAlgorithm(
        algorithm_id="dyn_vff_like",
        domain="dynamics",
        display_name="VFF-like exact-trained statefit",
        method_family="variational_compilation_like",
        supported_families=("hh",),
        required_capabilities=("supports_drive_benchmark_exact",),
        implemented_families=("hh",),
        runner_module=_DYNAMICS_HH_RUNNER,
        exact_assisted=True,
        diagnostic=True,
        hamiltonian_generic=False,
        notes="Exact-trained HH diagnostic; not true generic VFF/LSVQC.",
    ),
)


def default_benchmark_algorithms(*, domain: BenchmarkDomain | None = None) -> tuple[BenchmarkAlgorithm, ...]:
    if domain is None:
        return _DEFAULT_ALGORITHMS
    return tuple(alg for alg in _DEFAULT_ALGORITHMS if alg.domain == domain)


def get_benchmark_algorithm(algorithm_id: str) -> BenchmarkAlgorithm:
    for alg in _DEFAULT_ALGORITHMS:
        if alg.algorithm_id == algorithm_id:
            return alg
    known = ", ".join(alg.algorithm_id for alg in _DEFAULT_ALGORITHMS)
    raise ValueError(f"Unknown benchmark algorithm {algorithm_id!r}. Known algorithms: {known}")


def _capability_supported(family: str, capability: str) -> bool:
    spec = get_problem_family_spec(family)
    return bool(getattr(spec.capabilities, capability, False))


def _resolve_required_pool(family: str, required_pool_key: str | None) -> tuple[bool, str | None, str]:
    if required_pool_key is None:
        return True, None, ""
    spec = get_problem_family_spec(family)
    if required_pool_key == "__family_default__":
        # Avoid constructing full problem contexts here; use family spec metadata.
        # This is sufficient for applicability/manifest planning and avoids
        # invoking expensive Hamiltonian builders just to decide skip/runnable.
        pool = spec.default_pool_key
        if pool is None and family == "hh":
            pool = "paop_lf_std"
        if pool is None:
            return False, None, "family has no default runtime pool"
        if str(pool) not in spec.admissible_pool_keys:
            return False, str(pool), f"default pool {pool!r} is not admissible for family"
        return True, str(pool), ""
    if required_pool_key not in spec.admissible_pool_keys:
        return False, required_pool_key, f"required pool {required_pool_key!r} is not admissible for family"
    return True, required_pool_key, ""


def evaluate_algorithm_for_family(
    algorithm: BenchmarkAlgorithm | str,
    family: str,
) -> AlgorithmApplicability:
    alg = get_benchmark_algorithm(algorithm) if isinstance(algorithm, str) else algorithm
    family_key = str(family).strip()
    if family_key not in available_problem_keys():
        return AlgorithmApplicability(
            family=family_key,
            algorithm_id=alg.algorithm_id,
            domain=alg.domain,
            status="skipped_unsupported",
            reason=f"unknown family {family_key!r}",
            runner_module=alg.runner_module,
            qpu_faithful=alg.qpu_faithful,
            exact_assisted=alg.exact_assisted,
            diagnostic=alg.diagnostic,
            hamiltonian_generic=alg.hamiltonian_generic,
            required_pool_key=alg.required_pool_key,
        )
    if alg.supported_families is not None and family_key not in alg.supported_families:
        return AlgorithmApplicability(
            family=family_key,
            algorithm_id=alg.algorithm_id,
            domain=alg.domain,
            status="skipped_unsupported",
            reason="algorithm is not meaningful for this Hamiltonian family",
            runner_module=alg.runner_module,
            qpu_faithful=alg.qpu_faithful,
            exact_assisted=alg.exact_assisted,
            diagnostic=alg.diagnostic,
            hamiltonian_generic=alg.hamiltonian_generic,
            required_pool_key=alg.required_pool_key,
        )
    for capability in alg.required_capabilities:
        if not _capability_supported(family_key, capability):
            return AlgorithmApplicability(
                family=family_key,
                algorithm_id=alg.algorithm_id,
                domain=alg.domain,
                status="skipped_unsupported",
                reason=f"family lacks required capability {capability}",
                runner_module=alg.runner_module,
                qpu_faithful=alg.qpu_faithful,
                exact_assisted=alg.exact_assisted,
                diagnostic=alg.diagnostic,
                hamiltonian_generic=alg.hamiltonian_generic,
                required_pool_key=alg.required_pool_key,
            )
    pool_ok, resolved_pool, pool_reason = _resolve_required_pool(family_key, alg.required_pool_key)
    if not pool_ok:
        return AlgorithmApplicability(
            family=family_key,
            algorithm_id=alg.algorithm_id,
            domain=alg.domain,
            status="skipped_unsupported",
            reason=pool_reason,
            runner_module=alg.runner_module,
            qpu_faithful=alg.qpu_faithful,
            exact_assisted=alg.exact_assisted,
            diagnostic=alg.diagnostic,
            hamiltonian_generic=alg.hamiltonian_generic,
            required_pool_key=alg.required_pool_key,
            resolved_pool_key=resolved_pool,
        )
    if alg.implemented_families is not None and family_key not in alg.implemented_families:
        return AlgorithmApplicability(
            family=family_key,
            algorithm_id=alg.algorithm_id,
            domain=alg.domain,
            status="skipped_not_implemented",
            reason="conceptually compatible, but no benchmark row runner is wired for this family yet",
            runner_module=alg.runner_module,
            qpu_faithful=alg.qpu_faithful,
            exact_assisted=alg.exact_assisted,
            diagnostic=alg.diagnostic,
            hamiltonian_generic=alg.hamiltonian_generic,
            required_pool_key=alg.required_pool_key,
            resolved_pool_key=resolved_pool,
        )
    if alg.runner_module is None:
        return AlgorithmApplicability(
            family=family_key,
            algorithm_id=alg.algorithm_id,
            domain=alg.domain,
            status="skipped_no_runner",
            reason="no standalone benchmark row runner is defined",
            runner_module=None,
            qpu_faithful=alg.qpu_faithful,
            exact_assisted=alg.exact_assisted,
            diagnostic=alg.diagnostic,
            hamiltonian_generic=alg.hamiltonian_generic,
            required_pool_key=alg.required_pool_key,
            resolved_pool_key=resolved_pool,
        )
    return AlgorithmApplicability(
        family=family_key,
        algorithm_id=alg.algorithm_id,
        domain=alg.domain,
        status="runnable",
        reason="implemented runner available",
        runner_module=alg.runner_module,
        qpu_faithful=alg.qpu_faithful,
        exact_assisted=alg.exact_assisted,
        diagnostic=alg.diagnostic,
        hamiltonian_generic=alg.hamiltonian_generic,
        required_pool_key=alg.required_pool_key,
        resolved_pool_key=resolved_pool,
    )


def compatibility_matrix(
    *,
    families: Sequence[str] | None = None,
    algorithms: Sequence[BenchmarkAlgorithm] | None = None,
    domain: BenchmarkDomain | None = None,
) -> list[AlgorithmApplicability]:
    fams = tuple(families) if families is not None else available_problem_keys()
    algs = tuple(algorithms) if algorithms is not None else default_benchmark_algorithms(domain=domain)
    return [evaluate_algorithm_for_family(alg, family) for family in fams for alg in algs]


__all__ = [
    "AlgorithmApplicability",
    "ApplicabilityStatus",
    "BenchmarkAlgorithm",
    "BenchmarkDomain",
    "compatibility_matrix",
    "default_benchmark_algorithms",
    "evaluate_algorithm_for_family",
    "get_benchmark_algorithm",
]
