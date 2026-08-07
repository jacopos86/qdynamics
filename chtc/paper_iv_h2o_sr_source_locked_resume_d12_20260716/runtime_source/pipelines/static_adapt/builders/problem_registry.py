"""Problem family registry and resolved runtime context for static ADAPT."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.quantum.chemistry.molecular_hamiltonian import (
    build_restricted_closed_shell_molecular_hamiltonian,
)
from src.quantum.chemistry.psi4_adapter import (
    load_restricted_closed_shell_problem_from_json,
)
from src.quantum.chemistry.vibronic_h2 import (
    build_cached_vibronic_h2_model,
    load_cached_vibronic_h2_fixture,
)
from src.quantum.chemistry.vibronic_h2o import (
    build_cached_vibronic_h2o_model,
    load_cached_vibronic_h2o_fixture,
)
from src.quantum.chemistry.vibronic_h2o_linear_fd import (
    H2O_LINEAR_FD_FAMILY_KEY,
    load_cached_production_vibronic_h2o_linear_fd_fixture,
)
from src.quantum.hartree_fock_reference_state import hartree_fock_statevector
from src.quantum.hubbard_latex_python_pairs import boson_qubits_per_site
from src.quantum.operator_pools.boson_chains import (
    build_boson_chain_fock_statevector,
    build_boson_chain_vacuum_statevector,
)
from src.quantum.operator_pools.spin_boson import (
    build_spin_boson_reference_statevector,
)
from src.quantum.vqe_latex_python_pairs import half_filled_num_particles

from .lattice_hamiltonians import spinless_reference_statevector
from .problem_setup import (
    _default_adapt_input_state,
    _exact_gs_energy_for_problem,
    build_problem_hamiltonian,
)


from pipelines.contracts.problem import (
    ExactTargetSpec,
    FixedCountConstraint,
    HamiltonianFamilyCapabilities,
    ParityConstraint,
    ProblemFamilySpec,
    ProblemRequest,
    ReferenceStateSpec,
    RegisterBlockSpec,
    RegisterLayoutSpec,
    ResolvedProblemContext,
    SectorSelection,
    TruncationConstraint,
    WeightedChargeConstraint,
    canonical_problem_key,
)


def _problem_request_from_namespace(cls: type[ProblemRequest], args: Any) -> ProblemRequest:
    problem_key = canonical_problem_key(getattr(args, "problem", "hubbard"))
    molecular_problem_json_raw = getattr(args, "molecular_problem_json", None)
    molecular_problem_json = (
        None
        if molecular_problem_json_raw in {None, ""}
        else str(Path(molecular_problem_json_raw))
    )
    molecular_vibronic_h2_fixture_raw = getattr(args, "molecular_vibronic_h2_fixture_json", None)
    molecular_vibronic_h2_fixture_json = (
        None
        if molecular_vibronic_h2_fixture_raw in {None, ""}
        else str(Path(molecular_vibronic_h2_fixture_raw))
    )
    molecular_vibronic_h2o_fixture_raw = getattr(args, "molecular_vibronic_h2o_fixture_json", None)
    molecular_vibronic_h2o_fixture_json = (
        None
        if molecular_vibronic_h2o_fixture_raw in {None, ""}
        else str(Path(molecular_vibronic_h2o_fixture_raw))
    )
    molecular_vibronic_h2o_linear_fd_fixture_raw = getattr(
        args,
        "molecular_vibronic_h2o_linear_fd_fixture_json",
        None,
    )
    molecular_vibronic_h2o_linear_fd_fixture_json = (
        None
        if molecular_vibronic_h2o_linear_fd_fixture_raw in {None, ""}
        else str(Path(molecular_vibronic_h2o_linear_fd_fixture_raw))
    )
    num_sites = int(getattr(args, "L"))
    if problem_key == "molecular_restricted_closed_shell" and molecular_problem_json is not None:
        problem = load_restricted_closed_shell_problem_from_json(Path(molecular_problem_json))
        num_sites = int(problem.n_spatial_orbitals)
    n_ph_max = int(getattr(args, "n_ph_max"))
    if problem_key == H2O_LINEAR_FD_FAMILY_KEY and molecular_vibronic_h2o_linear_fd_fixture_json is not None:
        cached_h2o_linear_fd = load_cached_production_vibronic_h2o_linear_fd_fixture(
            Path(molecular_vibronic_h2o_linear_fd_fixture_json)
        )
        num_sites = int(cached_h2o_linear_fd.model.n_spatial_orbitals)
        n_ph_max = max(int(cutoff) for cutoff in cached_h2o_linear_fd.model.mode_cutoffs)
    return cls(
        problem_key=problem_key,
        num_sites=int(num_sites),
        t=float(getattr(args, "t")),
        u=float(getattr(args, "u")),
        dv=float(getattr(args, "dv")),
        omega0=float(getattr(args, "omega0")),
        g_ep=float(getattr(args, "g_ep")),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(getattr(args, "boson_encoding")),
        ordering=str(getattr(args, "ordering")),
        boundary=str(getattr(args, "boundary")),
        include_zero_point=bool(getattr(args, "include_zero_point", True)),
        molecular_problem_json=molecular_problem_json,
        molecular_vibronic_h2_fixture_json=molecular_vibronic_h2_fixture_json,
        molecular_vibronic_h2o_fixture_json=molecular_vibronic_h2o_fixture_json,
        molecular_vibronic_h2o_linear_fd_fixture_json=molecular_vibronic_h2o_linear_fd_fixture_json,
        v_nn=float(getattr(args, "v_nn", 0.0)),
        t_prime=float(getattr(args, "t_prime", 0.0)),
        n_fermions=(
            None
            if getattr(args, "n_fermions", None) is None
            else int(getattr(args, "n_fermions"))
        ),
    )


ProblemRequest.from_namespace = classmethod(_problem_request_from_namespace)  # type: ignore[attr-defined]


def _problem_family_spec_resolve(
    self: ProblemFamilySpec,
    request: ProblemRequest,
    *,
    hamiltonian: Any | None = None,
    exact_energy_impl: Callable[..., float] | None = None,
) -> ResolvedProblemContext:
    if canonical_problem_key(request.problem_key) != self.family_key:
        raise ValueError(
            f"Problem request key {request.problem_key!r} does not match family {self.family_key!r}."
        )
    if self._context_resolver is not None:
        return self._context_resolver(
            self,
            request,
            hamiltonian=hamiltonian,
            exact_energy_impl=exact_energy_impl,
        )
    if (
        len(self.supported_boson_encodings) > 0
        and str(request.boson_encoding) not in self.supported_boson_encodings
    ):
        raise ValueError(
            f"Unsupported boson encoding {request.boson_encoding!r} for family {self.family_key!r}."
        )
    layout = self._layout_builder(request)
    h_poly = hamiltonian
    if h_poly is None:
        h_poly = build_problem_hamiltonian(
            problem_key=str(request.problem_key),
            num_sites=int(request.num_sites),
            t=float(request.t),
            u=float(request.u),
            dv=float(request.dv),
            omega0=float(request.omega0),
            g_ep=float(request.g_ep),
            n_ph_max=int(request.n_ph_max),
            boson_encoding=str(request.boson_encoding),
            ordering=str(request.ordering),
            boundary=str(request.boundary),
            include_zero_point=bool(request.include_zero_point),
            molecular_vibronic_h2_fixture_json=request.molecular_vibronic_h2_fixture_json,
            molecular_vibronic_h2o_fixture_json=request.molecular_vibronic_h2o_fixture_json,
            molecular_vibronic_h2o_linear_fd_fixture_json=request.molecular_vibronic_h2o_linear_fd_fixture_json,
            v_nn=float(request.v_nn),
            t_prime=float(request.t_prime),
        )
    if str(self.family_key) == "spinless_tv":
        default_num_particles = (int(_default_spinless_fermion_count(request)), 0)
    elif str(self.family_key) in {"spin_boson", "bose_hubbard", "harmonic_kerr_chain"}:
        default_num_particles = (0, 0)
    else:
        default_num_particles = tuple(half_filled_num_particles(int(request.num_sites)))
    sector = _build_sector_selection(
        family_key=str(self.family_key),
        request=request,
        default_num_particles=default_num_particles,
    )
    reference_state = _build_reference_state_spec(
        family_key=str(self.family_key),
        request=request,
    )
    exact_target = _build_exact_target_spec(
        family_key=str(self.family_key),
        request=request,
        h_poly=h_poly,
        sector=sector,
        reference_state=reference_state,
        exact_energy_impl=exact_energy_impl,
    )
    return ResolvedProblemContext(
        family_key=str(self.family_key),
        request=request,
        layout=layout,
        hamiltonian=h_poly,
        sector=sector,
        reference_state=reference_state,
        exact_target=exact_target,
        default_controller_profile=str(self.default_controller_profile),
        default_continuation_mode=str(self.default_continuation_mode),
        admissible_pool_keys=tuple(self.admissible_pool_keys),
        default_pool_key=(None if self.default_pool_key is None else str(self.default_pool_key)),
        default_pool_resolution_scope=str(self.default_pool_resolution_scope),
        default_sector_label=str(self.default_sector_label),
        default_reference_label=str(self.default_reference_label),
        exact_target_label=str(self.exact_target_label),
        exact_comparison_space_label=str(self.exact_comparison_space_label),
        default_num_particles=default_num_particles,
        capabilities=self.capabilities,
        runtime_data=None,
    )


ProblemFamilySpec.resolve = _problem_family_spec_resolve  # type: ignore[attr-defined]

def _build_hubbard_layout(request: ProblemRequest) -> RegisterLayoutSpec:
    fermion_qubits = 2 * int(request.num_sites)
    return RegisterLayoutSpec(
        total_qubits=int(fermion_qubits),
        fermion_qubits=int(fermion_qubits),
        boson_qubits=0,
        ordering=str(request.ordering),
        boson_encoding=None,
        blocks=(
            RegisterBlockSpec(
                name="fermion",
                kind="fermion",
                start_qubit=0,
                stop_qubit=int(fermion_qubits),
            ),
        ),
    )


def _build_spinless_layout(request: ProblemRequest) -> RegisterLayoutSpec:
    fermion_qubits = int(request.num_sites)
    return RegisterLayoutSpec(
        total_qubits=int(fermion_qubits),
        fermion_qubits=int(fermion_qubits),
        boson_qubits=0,
        ordering=str(request.ordering),
        boson_encoding=None,
        blocks=(
            RegisterBlockSpec(
                name="fermion",
                kind="fermion",
                start_qubit=0,
                stop_qubit=int(fermion_qubits),
            ),
        ),
    )


def _build_hh_layout(request: ProblemRequest) -> RegisterLayoutSpec:
    fermion_qubits = 2 * int(request.num_sites)
    boson_bits_per_site = int(
        boson_qubits_per_site(
            int(request.n_ph_max),
            str(request.boson_encoding),
        )
    )
    boson_qubits = int(request.num_sites) * int(boson_bits_per_site)
    total_qubits = int(fermion_qubits + boson_qubits)
    return RegisterLayoutSpec(
        total_qubits=int(total_qubits),
        fermion_qubits=int(fermion_qubits),
        boson_qubits=int(boson_qubits),
        ordering=str(request.ordering),
        boson_encoding=str(request.boson_encoding),
        blocks=(
            RegisterBlockSpec(
                name="fermion",
                kind="fermion",
                start_qubit=0,
                stop_qubit=int(fermion_qubits),
            ),
            RegisterBlockSpec(
                name="boson",
                kind="boson",
                start_qubit=int(fermion_qubits),
                stop_qubit=int(total_qubits),
            ),
        ),
    )


def _build_spin_boson_layout(request: ProblemRequest) -> RegisterLayoutSpec:
    if int(request.num_sites) < 1:
        raise ValueError(f"spin_boson requires L>=1; got L={request.num_sites}.")
    emitter_qubits = 2
    boson_bits_per_site = int(
        boson_qubits_per_site(
            int(request.n_ph_max),
            str(request.boson_encoding),
        )
    )
    boson_qubits = int(request.num_sites) * int(boson_bits_per_site)
    total_qubits = int(emitter_qubits + boson_qubits)
    return RegisterLayoutSpec(
        total_qubits=int(total_qubits),
        fermion_qubits=int(emitter_qubits),
        boson_qubits=int(boson_qubits),
        ordering=str(request.ordering),
        boson_encoding=str(request.boson_encoding),
        blocks=(
            RegisterBlockSpec(
                name="emitter",
                kind="fermion",
                start_qubit=0,
                stop_qubit=int(emitter_qubits),
            ),
            RegisterBlockSpec(
                name="boson",
                kind="boson",
                start_qubit=int(emitter_qubits),
                stop_qubit=int(total_qubits),
            ),
        ),
    )


def _build_molecular_vibronic_h2_layout(request: ProblemRequest) -> RegisterLayoutSpec:
    if int(request.num_sites) != 2:
        raise ValueError(f"molecular_vibronic_h2 supports L=2 only; got L={request.num_sites}.")
    if str(request.ordering).strip().lower() != "blocked":
        raise ValueError("molecular_vibronic_h2 supports ordering='blocked' only.")
    if int(request.n_ph_max) < 1:
        raise ValueError(f"molecular_vibronic_h2 supports n_ph_max>=1 only; got {request.n_ph_max}.")
    if str(request.boson_encoding).strip().lower() != "binary":
        raise ValueError("molecular_vibronic_h2 supports boson_encoding='binary' only.")
    if str(request.boundary).strip().lower() != "open":
        raise ValueError("molecular_vibronic_h2 supports boundary='open' only.")
    if not bool(request.include_zero_point):
        raise ValueError("molecular_vibronic_h2 fixture includes zero-point energy; include_zero_point must be true.")
    fixture_path = None if request.molecular_vibronic_h2_fixture_json in {None, ""} else Path(str(request.molecular_vibronic_h2_fixture_json))
    model = build_cached_vibronic_h2_model(
        n_ph_max=int(request.n_ph_max),
        boson_encoding=str(request.boson_encoding),
        coupling_scale=float(request.g_ep),
        ordering=str(request.ordering),
        fixture_path=fixture_path,
    )
    fermion_qubits = int(model.n_fermion_qubits)
    boson_qubits = int(model.n_boson_qubits)
    total_qubits = int(model.n_total_qubits)
    return RegisterLayoutSpec(
        total_qubits=int(total_qubits),
        fermion_qubits=int(fermion_qubits),
        boson_qubits=int(boson_qubits),
        ordering=str(request.ordering),
        boson_encoding=str(request.boson_encoding),
        blocks=(
            RegisterBlockSpec(
                name="fermion",
                kind="fermion",
                start_qubit=0,
                stop_qubit=int(fermion_qubits),
            ),
            RegisterBlockSpec(
                name="boson",
                kind="boson",
                start_qubit=int(fermion_qubits),
                stop_qubit=int(total_qubits),
            ),
        ),
    )


def _build_molecular_vibronic_h2o_layout(request: ProblemRequest) -> RegisterLayoutSpec:
    if int(request.num_sites) != 2:
        raise ValueError(f"molecular_vibronic_h2o supports active-space L=2 only; got L={request.num_sites}.")
    if str(request.ordering).strip().lower() != "blocked":
        raise ValueError("molecular_vibronic_h2o supports ordering='blocked' only.")
    if int(request.n_ph_max) < 1:
        raise ValueError(f"molecular_vibronic_h2o supports n_ph_max>=1 only; got {request.n_ph_max}.")
    if str(request.boson_encoding).strip().lower() != "binary":
        raise ValueError("molecular_vibronic_h2o supports boson_encoding='binary' only.")
    if str(request.boundary).strip().lower() != "open":
        raise ValueError("molecular_vibronic_h2o supports boundary='open' only.")
    if not bool(request.include_zero_point):
        raise ValueError("molecular_vibronic_h2o fixture includes zero-point energy; include_zero_point must be true.")
    fixture_path = None if request.molecular_vibronic_h2o_fixture_json in {None, ""} else Path(str(request.molecular_vibronic_h2o_fixture_json))
    model = build_cached_vibronic_h2o_model(
        n_ph_max=int(request.n_ph_max),
        boson_encoding=str(request.boson_encoding),
        coupling_scale=float(request.g_ep),
        ordering=str(request.ordering),
        fixture_path=fixture_path,
    )
    fermion_qubits = int(model.n_fermion_qubits)
    boson_qubits = int(model.n_boson_qubits)
    total_qubits = int(model.n_total_qubits)
    return RegisterLayoutSpec(
        total_qubits=int(total_qubits),
        fermion_qubits=int(fermion_qubits),
        boson_qubits=int(boson_qubits),
        ordering=str(request.ordering),
        boson_encoding=str(request.boson_encoding),
        blocks=(
            RegisterBlockSpec(
                name="fermion",
                kind="fermion",
                start_qubit=0,
                stop_qubit=int(fermion_qubits),
            ),
            RegisterBlockSpec(
                name="boson",
                kind="boson",
                start_qubit=int(fermion_qubits),
                stop_qubit=int(total_qubits),
            ),
        ),
    )


def _load_h2o_linear_fd_cached_from_request(request: ProblemRequest) -> Any:
    if request.molecular_vibronic_h2o_linear_fd_fixture_json in {None, ""}:
        raise ValueError(
            "molecular_vibronic_h2o_linear_fd requires "
            "molecular_vibronic_h2o_linear_fd_fixture_json."
        )
    return load_cached_production_vibronic_h2o_linear_fd_fixture(
        Path(str(request.molecular_vibronic_h2o_linear_fd_fixture_json))
    )


def _validate_h2o_linear_fd_request_controls(request: ProblemRequest, model: Any) -> None:
    if str(request.ordering).strip().lower() != "blocked":
        raise ValueError("molecular_vibronic_h2o_linear_fd supports ordering='blocked' only.")
    if str(request.boson_encoding).strip().lower() != "binary":
        raise ValueError("molecular_vibronic_h2o_linear_fd supports boson_encoding='binary' only.")
    if str(request.boundary).strip().lower() != "open":
        raise ValueError("molecular_vibronic_h2o_linear_fd supports boundary='open' only.")
    if not bool(request.include_zero_point):
        raise ValueError(
            "molecular_vibronic_h2o_linear_fd fixture includes zero-point energy; "
            "include_zero_point must be true."
        )
    if int(request.num_sites) != int(model.n_spatial_orbitals):
        raise ValueError(
            "molecular_vibronic_h2o_linear_fd active-space L mismatch: "
            f"got L={request.num_sites}, fixture has {model.n_spatial_orbitals}."
        )
    cutoff_summary = max(int(cutoff) for cutoff in model.mode_cutoffs)
    if int(request.n_ph_max) != int(cutoff_summary):
        raise ValueError(
            "molecular_vibronic_h2o_linear_fd scalar n_ph_max must match the fixture "
            f"cutoff summary {cutoff_summary}; got {request.n_ph_max}."
        )


def _build_molecular_vibronic_h2o_linear_fd_layout(request: ProblemRequest) -> RegisterLayoutSpec:
    cached = _load_h2o_linear_fd_cached_from_request(request)
    model = cached.model
    _validate_h2o_linear_fd_request_controls(request, model)
    mode_blocks: list[RegisterBlockSpec] = []
    for block in cached.fixture.layout.boson_modes:
        mode_blocks.append(
            RegisterBlockSpec(
                name=f"boson_mode_{block.mode_label}",
                kind="boson_mode",
                start_qubit=int(block.qubit_start),
                stop_qubit=int(block.qubit_start) + int(block.n_qubits),
            )
        )
    return RegisterLayoutSpec(
        total_qubits=int(model.n_total_qubits),
        fermion_qubits=int(model.n_fermion_qubits),
        boson_qubits=int(model.n_boson_qubits),
        ordering=str(request.ordering),
        boson_encoding=str(request.boson_encoding),
        blocks=(
            RegisterBlockSpec(
                name="fermion",
                kind="fermion",
                start_qubit=0,
                stop_qubit=int(model.n_fermion_qubits),
            ),
            RegisterBlockSpec(
                name="boson",
                kind="boson",
                start_qubit=int(model.n_fermion_qubits),
                stop_qubit=int(model.n_total_qubits),
            ),
            *tuple(mode_blocks),
        ),
    )


def _build_boson_chain_layout(request: ProblemRequest) -> RegisterLayoutSpec:
    boson_bits_per_site = int(
        boson_qubits_per_site(
            int(request.n_ph_max),
            str(request.boson_encoding),
        )
    )
    boson_qubits = int(request.num_sites) * int(boson_bits_per_site)
    return RegisterLayoutSpec(
        total_qubits=int(boson_qubits),
        fermion_qubits=0,
        boson_qubits=int(boson_qubits),
        ordering=str(request.ordering),
        boson_encoding=str(request.boson_encoding),
        blocks=(
            RegisterBlockSpec(
                name="boson",
                kind="boson",
                start_qubit=0,
                stop_qubit=int(boson_qubits),
            ),
        ),
    )


def _default_spinless_fermion_count(request: ProblemRequest) -> int:
    n_fermions = int(request.n_fermions) if request.n_fermions is not None else (int(request.num_sites) // 2)
    if n_fermions < 0 or n_fermions > int(request.num_sites):
        raise ValueError(
            f"spinless_tv requires 0 <= n_fermions <= num_sites; got n_fermions={n_fermions}, num_sites={request.num_sites}."
        )
    return int(n_fermions)


def _build_sector_selection(
    *,
    family_key: str,
    request: ProblemRequest,
    default_num_particles: tuple[int, int],
) -> SectorSelection:
    if str(family_key) == "spinless_tv":
        return SectorSelection(
            label="fixed_spinless_sector",
            comparison_space_label="spinless_fermion_register",
            constraints=(
                FixedCountConstraint(
                    quantity="n_f",
                    value=int(default_num_particles[0]),
                    scope="full_register",
                ),
            ),
            num_particles=tuple(default_num_particles),
        )
    if str(family_key) in {"bose_hubbard", "harmonic_kerr_chain"}:
        return SectorSelection(
            label="truncated_boson_register",
            comparison_space_label="truncated_boson_register",
            constraints=(
                TruncationConstraint(
                    quantity="boson_occupancy",
                    max_local_occupancy=int(request.n_ph_max),
                    scope="boson_register",
                ),
            ),
            num_particles=None,
        )
    if str(family_key) == "hh":
        return SectorSelection(
            label="half_filled_fermion_sector",
            comparison_space_label="fermion_register_with_truncated_bosons",
            constraints=(
                FixedCountConstraint(
                    quantity="n_up",
                    value=int(default_num_particles[0]),
                    scope="fermion_register",
                ),
                FixedCountConstraint(
                    quantity="n_dn",
                    value=int(default_num_particles[1]),
                    scope="fermion_register",
                ),
                TruncationConstraint(
                    quantity="phonon_occupancy",
                    max_local_occupancy=int(request.n_ph_max),
                    scope="boson_register",
                ),
            ),
            num_particles=tuple(default_num_particles),
        )
    return SectorSelection(
        label="half_filled_spin_sector",
        comparison_space_label="full_register",
        constraints=(
            FixedCountConstraint(
                quantity="n_up",
                value=int(default_num_particles[0]),
                scope="full_register",
            ),
            FixedCountConstraint(
                quantity="n_dn",
                value=int(default_num_particles[1]),
                scope="full_register",
            ),
        ),
        num_particles=tuple(default_num_particles),
    )


def _build_reference_state_spec(
    *,
    family_key: str,
    request: ProblemRequest,
) -> ReferenceStateSpec:
    if str(family_key) == "spinless_tv":
        n_fermions = _default_spinless_fermion_count(request)

        def _build_state_spinless() -> np.ndarray:
            return np.asarray(
                spinless_reference_statevector(
                    num_sites=int(request.num_sites),
                    n_fermions=int(n_fermions),
                ),
                dtype=complex,
            ).reshape(-1)

        return ReferenceStateSpec(
            kind="spinless_fermion_filling",
            source_label="filling",
            state_kind="reference_state",
            build_state=_build_state_spinless,
        )
    if str(family_key) == "bose_hubbard":

        def _build_state_bose_hubbard_one_boson() -> np.ndarray:
            # Bose-Hubbard conserves total boson number. Starting from the
            # vacuum makes the ADAPT commutator gradient vanish identically
            # while the unrestricted truncated-register exact target lies in a
            # one-boson sector for the default benchmark parameters. Seed the
            # scaffold in that accessible sector without using exact-state
            # information.
            onsite = [(((-1.0) ** int(site)) * float(request.dv), int(site)) for site in range(int(request.num_sites))]
            seed_site = min(onsite, key=lambda item: (float(item[0]), int(item[1])))[1]
            occupations = [0 for _ in range(int(request.num_sites))]
            occupations[int(seed_site)] = 1
            return np.asarray(
                build_boson_chain_fock_statevector(
                    num_sites=int(request.num_sites),
                    n_ph_max=int(request.n_ph_max),
                    boson_encoding=str(request.boson_encoding),
                    occupations=tuple(occupations),
                ),
                dtype=complex,
            ).reshape(-1)

        return ReferenceStateSpec(
            kind="bose_hubbard_one_boson_fock",
            source_label="one_boson_fock",
            state_kind="reference_state",
            build_state=_build_state_bose_hubbard_one_boson,
        )

    if str(family_key) == "harmonic_kerr_chain":

        def _build_state_boson_vacuum() -> np.ndarray:
            return np.asarray(
                build_boson_chain_vacuum_statevector(
                    num_sites=int(request.num_sites),
                    n_ph_max=int(request.n_ph_max),
                    boson_encoding=str(request.boson_encoding),
                ),
                dtype=complex,
            ).reshape(-1)

        return ReferenceStateSpec(
            kind="boson_vacuum",
            source_label="boson_vacuum",
            state_kind="reference_state",
            build_state=_build_state_boson_vacuum,
        )

    def _build_state() -> np.ndarray:
        psi_ref, _source, _kind = _default_adapt_input_state(
            problem=str(family_key),
            num_sites=int(request.num_sites),
            ordering=str(request.ordering),
            n_ph_max=int(request.n_ph_max),
            boson_encoding=str(request.boson_encoding),
        )
        return np.asarray(psi_ref, dtype=complex).reshape(-1)

    return ReferenceStateSpec(
        kind=(
            "hubbard_holstein_reference_state"
            if str(family_key) == "hh"
            else "hartree_fock"
        ),
        source_label="hf",
        state_kind="reference_state",
        build_state=_build_state,
    )


def _build_exact_target_spec(
    *,
    family_key: str,
    request: ProblemRequest,
    h_poly: Any,
    sector: SectorSelection,
    reference_state: ReferenceStateSpec,
    exact_energy_impl: Callable[..., float] | None = None,
) -> ExactTargetSpec:
    exact_energy_callable = (
        _exact_gs_energy_for_problem
        if exact_energy_impl is None
        else exact_energy_impl
    )

    def _resolve_energy(*, ai_log: Callable[..., None] | None = None) -> float:
        num_particles = (
            tuple(sector.num_particles)
            if sector.num_particles is not None
            else tuple(half_filled_num_particles(int(request.num_sites)))
        )
        return float(
            exact_energy_callable(
                h_poly,
                problem=str(family_key),
                num_sites=int(request.num_sites),
                num_particles=num_particles,
                indexing=str(request.ordering),
                n_ph_max=int(request.n_ph_max),
                boson_encoding=str(request.boson_encoding),
                t=float(request.t),
                u=float(request.u),
                dv=float(request.dv),
                v_nn=float(request.v_nn),
                t_prime=float(request.t_prime),
                omega0=float(request.omega0),
                g_ep=float(request.g_ep),
                boundary=str(request.boundary),
                include_zero_point=bool(request.include_zero_point),
                ai_log=ai_log,
            )
        )

    return ExactTargetSpec(
        kind=(
            "exact_ground_energy_sector_hh"
            if str(family_key) == "hh"
            else "exact_ground_energy_spinless_fixed_count"
            if str(family_key) == "spinless_tv"
            else "exact_ground_energy_spin_boson"
            if str(family_key) == "spin_boson"
            else "exact_ground_energy_boson_only"
            if str(family_key) in {"bose_hubbard", "harmonic_kerr_chain"}
            else "exact_ground_energy_sector"
        ),
        comparison_space_label=str(sector.comparison_space_label),
        resolve_energy=_resolve_energy,
        exact_state_policy="dense_diagonalization_if_available",
        build_fallback_anchor_state=reference_state.build_state,
        fallback_policy="reference_state_anchor_when_exact_state_unavailable",
    )


def _load_molecular_problem_from_request(request: ProblemRequest) -> Any:
    raw_path = request.molecular_problem_json
    if raw_path in {None, ""}:
        raise ValueError(
            "problem='molecular_restricted_closed_shell' requires --molecular-problem-json."
        )
    return load_restricted_closed_shell_problem_from_json(Path(str(raw_path)))


def _build_molecular_layout(
    *,
    request: ProblemRequest,
    molecular_problem: Any,
) -> RegisterLayoutSpec:
    fermion_qubits = int(molecular_problem.n_spin_orbitals)
    return RegisterLayoutSpec(
        total_qubits=int(fermion_qubits),
        fermion_qubits=int(fermion_qubits),
        boson_qubits=0,
        ordering=str(request.ordering),
        boson_encoding=None,
        blocks=(
            RegisterBlockSpec(
                name="fermion",
                kind="fermion",
                start_qubit=0,
                stop_qubit=int(fermion_qubits),
            ),
        ),
    )


def _resolve_molecular_problem_context(
    family_spec: ProblemFamilySpec,
    request: ProblemRequest,
    *,
    hamiltonian: Any | None = None,
    exact_energy_impl: Callable[..., float] | None = None,
) -> ResolvedProblemContext:
    if str(request.ordering).strip().lower() != "blocked":
        raise ValueError(
            "molecular_restricted_closed_shell currently supports ordering='blocked' only."
        )
    molecular_problem = _load_molecular_problem_from_request(request)
    if int(request.num_sites) != int(molecular_problem.n_spatial_orbitals):
        raise ValueError(
            "molecular_restricted_closed_shell request.num_sites must equal "
            f"n_spatial_orbitals from the JSON payload; got request.num_sites={request.num_sites} "
            f"and n_spatial_orbitals={molecular_problem.n_spatial_orbitals}."
        )
    layout = _build_molecular_layout(
        request=request,
        molecular_problem=molecular_problem,
    )
    h_poly = (
        hamiltonian
        if hamiltonian is not None
        else build_restricted_closed_shell_molecular_hamiltonian(
            molecular_problem,
            ordering=str(request.ordering),
        )
    )
    num_particles = tuple(int(x) for x in molecular_problem.num_particles)

    def _build_state() -> np.ndarray:
        return np.asarray(
            hartree_fock_statevector(
                int(molecular_problem.n_spatial_orbitals),
                num_particles,
                indexing=str(request.ordering),
            ),
            dtype=complex,
        ).reshape(-1)

    exact_energy_callable = (
        _exact_gs_energy_for_problem
        if exact_energy_impl is None
        else exact_energy_impl
    )

    def _resolve_energy(*, ai_log: Callable[..., None] | None = None) -> float:
        return float(
            exact_energy_callable(
                h_poly,
                problem=str(family_spec.family_key),
                num_sites=int(molecular_problem.n_spatial_orbitals),
                num_particles=num_particles,
                indexing=str(request.ordering),
                n_ph_max=0,
                boson_encoding="binary",
                t=float(request.t),
                u=float(request.u),
                dv=float(request.dv),
                omega0=float(request.omega0),
                g_ep=float(request.g_ep),
                boundary=str(request.boundary),
                include_zero_point=bool(request.include_zero_point),
                ai_log=ai_log,
            )
        )

    return ResolvedProblemContext(
        family_key=str(family_spec.family_key),
        request=request,
        layout=layout,
        hamiltonian=h_poly,
        sector=SectorSelection(
            label="closed_shell_fixed_number_sector",
            comparison_space_label="spin_orbital_register",
            constraints=(
                FixedCountConstraint(
                    quantity="n_up",
                    value=int(num_particles[0]),
                    scope="full_register",
                ),
                FixedCountConstraint(
                    quantity="n_dn",
                    value=int(num_particles[1]),
                    scope="full_register",
                ),
            ),
            num_particles=num_particles,
        ),
        reference_state=ReferenceStateSpec(
            kind="restricted_hartree_fock",
            source_label="hf",
            state_kind="reference_state",
            build_state=_build_state,
        ),
        exact_target=ExactTargetSpec(
            kind="exact_ground_energy_sector_molecular",
            comparison_space_label="spin_orbital_register",
            resolve_energy=_resolve_energy,
            exact_state_policy="dense_diagonalization_if_available",
            build_fallback_anchor_state=_build_state,
            fallback_policy="reference_state_anchor_when_exact_state_unavailable",
        ),
        default_controller_profile=str(family_spec.default_controller_profile),
        default_continuation_mode=str(family_spec.default_continuation_mode),
        admissible_pool_keys=tuple(family_spec.admissible_pool_keys),
        default_pool_key=(
            None
            if family_spec.default_pool_key is None
            else str(family_spec.default_pool_key)
        ),
        default_pool_resolution_scope=str(family_spec.default_pool_resolution_scope),
        default_sector_label=str(family_spec.default_sector_label),
        default_reference_label=str(family_spec.default_reference_label),
        exact_target_label=str(family_spec.exact_target_label),
        exact_comparison_space_label=str(family_spec.exact_comparison_space_label),
        default_num_particles=num_particles,
        capabilities=family_spec.capabilities,
        runtime_data={
            "molecular_problem": molecular_problem,
            "molecular_problem_json": (
                None
                if request.molecular_problem_json in {None, ""}
                else str(request.molecular_problem_json)
            ),
        },
    )


def _resolve_molecular_vibronic_h2_context(
    family_spec: ProblemFamilySpec,
    request: ProblemRequest,
    *,
    hamiltonian: Any | None = None,
    exact_energy_impl: Callable[..., float] | None = None,
) -> ResolvedProblemContext:
    if int(request.num_sites) != 2:
        raise ValueError(f"molecular_vibronic_h2 supports L=2 only; got L={request.num_sites}.")
    if str(request.ordering).strip().lower() != "blocked":
        raise ValueError("molecular_vibronic_h2 supports ordering='blocked' only.")
    if int(request.n_ph_max) < 1:
        raise ValueError(f"molecular_vibronic_h2 supports n_ph_max>=1 only; got {request.n_ph_max}.")
    if str(request.boson_encoding).strip().lower() != "binary":
        raise ValueError("molecular_vibronic_h2 supports boson_encoding='binary' only.")
    if str(request.boundary).strip().lower() != "open":
        raise ValueError("molecular_vibronic_h2 supports boundary='open' only.")
    if not bool(request.include_zero_point):
        raise ValueError("molecular_vibronic_h2 fixture includes zero-point energy; include_zero_point must be true.")
    fixture_path = None if request.molecular_vibronic_h2_fixture_json in {None, ""} else Path(str(request.molecular_vibronic_h2_fixture_json))
    fixture = load_cached_vibronic_h2_fixture(fixture_path)
    model = build_cached_vibronic_h2_model(
        n_ph_max=int(request.n_ph_max),
        boson_encoding=str(request.boson_encoding),
        coupling_scale=float(request.g_ep),
        ordering=str(request.ordering),
        fixture_path=fixture_path,
    )
    layout = _build_molecular_vibronic_h2_layout(request)
    h_poly = hamiltonian if hamiltonian is not None else model.h_vibronic
    num_particles = (1, 1)

    def _build_state() -> np.ndarray:
        return np.asarray(model.psi_ref, dtype=complex).reshape(-1).copy()

    def _resolve_energy(*, ai_log: Callable[..., None] | None = None) -> float:
        exact_energy_callable = exact_energy_impl
        if (
            exact_energy_callable is None
            and fixture.exact_ground_energy is not None
            and int(request.n_ph_max) == int(fixture.model.n_ph_max)
            and math.isclose(float(request.g_ep), float(fixture.model.coupling_scale), rel_tol=0.0, abs_tol=1e-15)
        ):
            return float(fixture.exact_ground_energy)
        if exact_energy_callable is None:
            exact_energy_callable = _exact_gs_energy_for_problem
        return float(
            exact_energy_callable(
                h_poly,
                problem=str(family_spec.family_key),
                num_sites=2,
                num_particles=num_particles,
                indexing="blocked",
                n_ph_max=int(request.n_ph_max),
                boson_encoding="binary",
                t=float(request.t),
                u=float(request.u),
                dv=float(request.dv),
                omega0=float(request.omega0),
                g_ep=float(request.g_ep),
                boundary=str(request.boundary),
                include_zero_point=True,
                ai_log=ai_log,
            )
        )

    return ResolvedProblemContext(
        family_key=str(family_spec.family_key),
        request=request,
        layout=layout,
        hamiltonian=h_poly,
        sector=SectorSelection(
            label="closed_shell_fermions_with_truncated_vibration",
            comparison_space_label="spin_orbital_register_with_truncated_vibration",
            constraints=(
                FixedCountConstraint(
                    quantity="n_up",
                    value=1,
                    scope="fermion_register",
                ),
                FixedCountConstraint(
                    quantity="n_dn",
                    value=1,
                    scope="fermion_register",
                ),
                TruncationConstraint(
                    quantity="vibrational_occupancy",
                    max_local_occupancy=int(request.n_ph_max),
                    scope="boson_register",
                ),
            ),
            num_particles=num_particles,
        ),
        reference_state=ReferenceStateSpec(
            kind="restricted_hf_times_boson_vacuum",
            source_label="cached_vibronic_h2_fixture",
            state_kind="reference_state",
            build_state=_build_state,
        ),
        exact_target=ExactTargetSpec(
            kind="exact_ground_energy_molecular_vibronic_h2_physical_sector",
            comparison_space_label="spin_orbital_register_with_truncated_vibration",
            resolve_energy=_resolve_energy,
            exact_state_policy="dense_diagonalization_if_available",
            build_fallback_anchor_state=_build_state,
            fallback_policy="reference_state_anchor_when_exact_state_unavailable",
        ),
        default_controller_profile=str(family_spec.default_controller_profile),
        default_continuation_mode=str(family_spec.default_continuation_mode),
        admissible_pool_keys=tuple(family_spec.admissible_pool_keys),
        default_pool_key=(
            None if family_spec.default_pool_key is None else str(family_spec.default_pool_key)
        ),
        default_pool_resolution_scope=str(family_spec.default_pool_resolution_scope),
        default_sector_label=str(family_spec.default_sector_label),
        default_reference_label=str(family_spec.default_reference_label),
        exact_target_label=str(family_spec.exact_target_label),
        exact_comparison_space_label=str(family_spec.exact_comparison_space_label),
        default_num_particles=num_particles,
        capabilities=family_spec.capabilities,
        runtime_data={
            "vibronic_h2_model": model,
            "vibronic_h2_fixture_path": str(fixture.fixture_path),
            "vibronic_h2_fixture_metadata": dict(fixture.metadata),
            "vibronic_h2_coupling_scale": float(model.coupling_scale),
        },
    )


def _resolve_molecular_vibronic_h2o_context(
    family_spec: ProblemFamilySpec,
    request: ProblemRequest,
    *,
    hamiltonian: Any | None = None,
    exact_energy_impl: Callable[..., float] | None = None,
) -> ResolvedProblemContext:
    if int(request.num_sites) != 2:
        raise ValueError(f"molecular_vibronic_h2o supports active-space L=2 only; got L={request.num_sites}.")
    if str(request.ordering).strip().lower() != "blocked":
        raise ValueError("molecular_vibronic_h2o supports ordering='blocked' only.")
    if int(request.n_ph_max) < 1:
        raise ValueError(f"molecular_vibronic_h2o supports n_ph_max>=1 only; got {request.n_ph_max}.")
    if str(request.boson_encoding).strip().lower() != "binary":
        raise ValueError("molecular_vibronic_h2o supports boson_encoding='binary' only.")
    if str(request.boundary).strip().lower() != "open":
        raise ValueError("molecular_vibronic_h2o supports boundary='open' only.")
    if not bool(request.include_zero_point):
        raise ValueError("molecular_vibronic_h2o fixture includes zero-point energy; include_zero_point must be true.")
    fixture_path = None if request.molecular_vibronic_h2o_fixture_json in {None, ""} else Path(str(request.molecular_vibronic_h2o_fixture_json))
    fixture = load_cached_vibronic_h2o_fixture(fixture_path)
    model = build_cached_vibronic_h2o_model(
        n_ph_max=int(request.n_ph_max),
        boson_encoding=str(request.boson_encoding),
        coupling_scale=float(request.g_ep),
        ordering=str(request.ordering),
        fixture_path=fixture_path,
    )
    layout = _build_molecular_vibronic_h2o_layout(request)
    h_poly = hamiltonian if hamiltonian is not None else model.h_vibronic
    num_particles = (1, 1)

    def _build_state() -> np.ndarray:
        return np.asarray(model.psi_ref, dtype=complex).reshape(-1).copy()

    def _resolve_energy(*, ai_log: Callable[..., None] | None = None) -> float:
        exact_energy_callable = exact_energy_impl
        if (
            exact_energy_callable is None
            and fixture.exact_ground_energy is not None
            and int(request.n_ph_max) == int(fixture.model.n_ph_max)
            and math.isclose(float(request.g_ep), float(fixture.model.coupling_scale), rel_tol=0.0, abs_tol=1e-15)
        ):
            return float(fixture.exact_ground_energy)
        if exact_energy_callable is None:
            exact_energy_callable = _exact_gs_energy_for_problem
        return float(
            exact_energy_callable(
                h_poly,
                problem=str(family_spec.family_key),
                num_sites=2,
                num_particles=num_particles,
                indexing="blocked",
                n_ph_max=int(request.n_ph_max),
                boson_encoding="binary",
                t=float(request.t),
                u=float(request.u),
                dv=float(request.dv),
                omega0=float(request.omega0),
                g_ep=float(request.g_ep),
                boundary=str(request.boundary),
                include_zero_point=True,
                ai_log=ai_log,
            )
        )

    return ResolvedProblemContext(
        family_key=str(family_spec.family_key),
        request=request,
        layout=layout,
        hamiltonian=h_poly,
        sector=SectorSelection(
            label="closed_shell_fermions_with_truncated_vibration",
            comparison_space_label="spin_orbital_register_with_truncated_vibration",
            constraints=(
                FixedCountConstraint(
                    quantity="n_up",
                    value=1,
                    scope="fermion_register",
                ),
                FixedCountConstraint(
                    quantity="n_dn",
                    value=1,
                    scope="fermion_register",
                ),
                TruncationConstraint(
                    quantity="vibrational_occupancy",
                    max_local_occupancy=int(request.n_ph_max),
                    scope="boson_register",
                ),
            ),
            num_particles=num_particles,
        ),
        reference_state=ReferenceStateSpec(
            kind="restricted_hf_times_boson_vacuum",
            source_label="cached_vibronic_h2o_fixture",
            state_kind="reference_state",
            build_state=_build_state,
        ),
        exact_target=ExactTargetSpec(
            kind="exact_ground_energy_molecular_vibronic_h2o_physical_sector",
            comparison_space_label="spin_orbital_register_with_truncated_vibration",
            resolve_energy=_resolve_energy,
            exact_state_policy="dense_diagonalization_if_available",
            build_fallback_anchor_state=_build_state,
            fallback_policy="reference_state_anchor_when_exact_state_unavailable",
        ),
        default_controller_profile=str(family_spec.default_controller_profile),
        default_continuation_mode=str(family_spec.default_continuation_mode),
        admissible_pool_keys=tuple(family_spec.admissible_pool_keys),
        default_pool_key=(
            None if family_spec.default_pool_key is None else str(family_spec.default_pool_key)
        ),
        default_pool_resolution_scope=str(family_spec.default_pool_resolution_scope),
        default_sector_label=str(family_spec.default_sector_label),
        default_reference_label=str(family_spec.default_reference_label),
        exact_target_label=str(family_spec.exact_target_label),
        exact_comparison_space_label=str(family_spec.exact_comparison_space_label),
        default_num_particles=num_particles,
        capabilities=family_spec.capabilities,
        runtime_data={
            "vibronic_h2o_model": model,
            "vibronic_h2o_fixture_path": str(fixture.fixture_path),
            "vibronic_h2o_fixture_metadata": dict(fixture.metadata),
            "vibronic_h2o_coupling_scale": float(model.coupling_scale),
        },
    )


def _resolve_molecular_vibronic_h2o_linear_fd_context(
    family_spec: ProblemFamilySpec,
    request: ProblemRequest,
    *,
    hamiltonian: Any | None = None,
    exact_energy_impl: Callable[..., float] | None = None,
) -> ResolvedProblemContext:
    cached = _load_h2o_linear_fd_cached_from_request(request)
    model = cached.model
    fixture = cached.fixture
    _validate_h2o_linear_fd_request_controls(request, model)
    layout = _build_molecular_vibronic_h2o_linear_fd_layout(request)
    h_poly = hamiltonian if hamiltonian is not None else model.h_vibronic
    num_particles = tuple(int(v) for v in model.num_particles)
    mode_cutoffs = tuple(int(v) for v in model.mode_cutoffs)
    mode_labels = tuple(str(v) for v in model.mode_labels)

    def _build_state() -> np.ndarray:
        return np.asarray(model.psi_ref, dtype=complex).reshape(-1).copy()

    def _resolve_energy(*, ai_log: Callable[..., None] | None = None) -> float:
        exact_energy_callable = exact_energy_impl
        if exact_energy_callable is None:
            energy = fixture.exact_reference.ground_energy_hartree
            if energy is None or not np.isfinite(float(energy)):
                raise ValueError("molecular_vibronic_h2o_linear_fd fixture exact ground energy is unavailable.")
            return float(energy)
        return float(
            exact_energy_callable(
                h_poly,
                problem=str(family_spec.family_key),
                num_sites=int(model.n_spatial_orbitals),
                num_particles=num_particles,
                indexing="blocked",
                n_ph_max=max(mode_cutoffs),
                boson_encoding="binary",
                t=float(request.t),
                u=float(request.u),
                dv=float(request.dv),
                omega0=float(request.omega0),
                g_ep=float(request.g_ep),
                boundary=str(request.boundary),
                include_zero_point=True,
                molecular_vibronic_h2o_linear_fd_fixture_json=str(cached.fixture_path),
                ai_log=ai_log,
            )
        )

    constraints: list[Any] = [
        FixedCountConstraint(
            quantity="n_up",
            value=int(num_particles[0]),
            scope="fermion_register",
        ),
        FixedCountConstraint(
            quantity="n_dn",
            value=int(num_particles[1]),
            scope="fermion_register",
        ),
    ]
    constraints.extend(
        TruncationConstraint(
            quantity="vibrational_occupancy",
            max_local_occupancy=int(cutoff),
            scope=f"boson_mode:{label}",
        )
        for label, cutoff in zip(mode_labels, mode_cutoffs)
    )

    return ResolvedProblemContext(
        family_key=str(family_spec.family_key),
        request=request,
        layout=layout,
        hamiltonian=h_poly,
        sector=SectorSelection(
            label="closed_shell_fermions_with_linear_fd_truncated_vibrations",
            comparison_space_label="spin_orbital_register_with_linear_fd_truncated_vibrations",
            constraints=tuple(constraints),
            num_particles=num_particles,
        ),
        reference_state=ReferenceStateSpec(
            kind="restricted_hf_times_linear_fd_boson_vacuum",
            source_label="production_h2o_linear_fd_fixture",
            state_kind="reference_state",
            build_state=_build_state,
        ),
        exact_target=ExactTargetSpec(
            kind="exact_ground_energy_molecular_vibronic_h2o_linear_fd_physical_sector",
            comparison_space_label="spin_orbital_register_with_linear_fd_truncated_vibrations",
            resolve_energy=_resolve_energy,
            exact_state_policy="fixture_exact_state_if_available",
            build_fallback_anchor_state=_build_state,
            fallback_policy="reference_state_anchor_when_fixture_exact_state_unavailable",
        ),
        default_controller_profile=str(family_spec.default_controller_profile),
        default_continuation_mode=str(family_spec.default_continuation_mode),
        admissible_pool_keys=tuple(family_spec.admissible_pool_keys),
        default_pool_key=(
            None if family_spec.default_pool_key is None else str(family_spec.default_pool_key)
        ),
        default_pool_resolution_scope=str(family_spec.default_pool_resolution_scope),
        default_sector_label=str(family_spec.default_sector_label),
        default_reference_label=str(family_spec.default_reference_label),
        exact_target_label=str(family_spec.exact_target_label),
        exact_comparison_space_label=str(family_spec.exact_comparison_space_label),
        default_num_particles=num_particles,
        capabilities=family_spec.capabilities,
        runtime_data={
            "vibronic_h2o_linear_fd_fixture": fixture,
            "vibronic_h2o_linear_fd_model": model,
            "vibronic_h2o_linear_fd_fixture_path": str(cached.fixture_path),
            "vibronic_h2o_linear_fd_fixture_metadata": dict(cached.metadata),
            "vibronic_h2o_linear_fd_mode_labels": mode_labels,
            "vibronic_h2o_linear_fd_mode_cutoffs": mode_cutoffs,
        },
    )


def _resolve_spin_boson_problem_context(
    family_spec: ProblemFamilySpec,
    request: ProblemRequest,
    *,
    hamiltonian: Any | None = None,
    exact_energy_impl: Callable[..., float] | None = None,
) -> ResolvedProblemContext:
    layout = _build_spin_boson_layout(request)
    h_poly = (
        hamiltonian
        if hamiltonian is not None
        else build_problem_hamiltonian(
            problem_key=str(family_spec.family_key),
            num_sites=int(request.num_sites),
            t=float(request.t),
            u=float(request.u),
            dv=float(request.dv),
            omega0=float(request.omega0),
            g_ep=float(request.g_ep),
            n_ph_max=int(request.n_ph_max),
            boson_encoding=str(request.boson_encoding),
            ordering=str(request.ordering),
            boundary=str(request.boundary),
            include_zero_point=bool(request.include_zero_point),
            v_nn=float(request.v_nn),
            t_prime=float(request.t_prime),
        )
    )

    def _build_state() -> np.ndarray:
        return np.asarray(
            build_spin_boson_reference_statevector(
                num_sites=int(request.num_sites),
                t=float(request.t),
                dv=float(request.dv),
                n_ph_max=int(request.n_ph_max),
                boson_encoding=str(request.boson_encoding),
            ),
            dtype=complex,
        ).reshape(-1)

    exact_energy_callable = (
        _exact_gs_energy_for_problem
        if exact_energy_impl is None
        else exact_energy_impl
    )

    def _resolve_energy(*, ai_log: Callable[..., None] | None = None) -> float:
        return float(
            exact_energy_callable(
                h_poly,
                problem=str(family_spec.family_key),
                num_sites=int(request.num_sites),
                num_particles=(0, 0),
                indexing=str(request.ordering),
                n_ph_max=int(request.n_ph_max),
                boson_encoding=str(request.boson_encoding),
                t=float(request.t),
                u=float(request.u),
                dv=float(request.dv),
                omega0=float(request.omega0),
                g_ep=float(request.g_ep),
                boundary=str(request.boundary),
                include_zero_point=bool(request.include_zero_point),
                ai_log=ai_log,
            )
        )

    sector = SectorSelection(
        label="single_emitter_truncated_boson_sector",
        comparison_space_label="one_emitter_truncated_boson_register",
        constraints=(
            WeightedChargeConstraint(
                quantity="n_emitter",
                weights=(("g", 1), ("e", 1)),
                value=1,
                scope="emitter_register",
            ),
            TruncationConstraint(
                quantity="boson_occupancy",
                max_local_occupancy=int(request.n_ph_max),
                scope="boson_register",
            ),
        ),
        num_particles=None,
    )

    return ResolvedProblemContext(
        family_key=str(family_spec.family_key),
        request=request,
        layout=layout,
        hamiltonian=h_poly,
        sector=sector,
        reference_state=ReferenceStateSpec(
            kind="spin_boson_uncoupled_ground",
            source_label="uncoupled_ground",
            state_kind="reference_state",
            build_state=_build_state,
        ),
        exact_target=ExactTargetSpec(
            kind="exact_ground_energy_spin_boson",
            comparison_space_label="one_emitter_truncated_boson_register",
            resolve_energy=_resolve_energy,
            exact_state_policy="dense_diagonalization_if_available",
            build_fallback_anchor_state=_build_state,
            fallback_policy="reference_state_anchor_when_exact_state_unavailable",
        ),
        default_controller_profile=str(family_spec.default_controller_profile),
        default_continuation_mode=str(family_spec.default_continuation_mode),
        admissible_pool_keys=tuple(family_spec.admissible_pool_keys),
        default_pool_key=(
            None if family_spec.default_pool_key is None else str(family_spec.default_pool_key)
        ),
        default_pool_resolution_scope=str(family_spec.default_pool_resolution_scope),
        default_sector_label=str(family_spec.default_sector_label),
        default_reference_label=str(family_spec.default_reference_label),
        exact_target_label=str(family_spec.exact_target_label),
        exact_comparison_space_label=str(family_spec.exact_comparison_space_label),
        default_num_particles=(0, 0),
        capabilities=family_spec.capabilities,
        runtime_data={
            "trajectory_metric_family": "spin_boson",
            "emitter_mode_labels": ("g", "e"),
            "spin_boson_boson_mode_count": int(request.num_sites),
            "spin_boson_emitter_qubits": 2,
        },
    )


_HUBBARD_POOL_KEYS = (
    "uccsd",
    "uccsd_qeb",
    "uccsd_qeb_hva_blocks",
    "cse",
    "full_hamiltonian",
    "hamiltonian_blocks",
    "full_meta",
)

_SPINFUL_LATTICE_POOL_KEYS = (
    "hamiltonian_quadratures",
    "hva",
    "uccsd",
    "full_hamiltonian",
    "hamiltonian_blocks",
    "family_max",
    "full_meta",
)

_MOLECULAR_POOL_KEYS = (
    "uccsd",
    "full_hamiltonian",
    "hamiltonian_blocks",
    "hva",
    "family_max",
    "full_meta",
)

_MOLECULAR_VIBRONIC_POOL_KEYS = (
    "full_meta",
    "full_hamiltonian",
)

_SPINLESS_POOL_KEYS = (
    "hamiltonian_quadratures",
    "hva",
    "full_hamiltonian",
    "hamiltonian_blocks",
    "family_max",
    "full_meta",
)

_SPIN_BOSON_POOL_KEYS = (
    "hamiltonian_quadratures",
    "hva",
    "full_hamiltonian",
    "hamiltonian_blocks",
    "family_max",
    "full_meta",
)

_BOSON_CHAIN_POOL_KEYS = (
    "hamiltonian_quadratures",
    "hva",
    "full_hamiltonian",
    "hamiltonian_blocks",
    "family_max",
    "full_meta",
)

_HH_POOL_KEYS = (
    "full_hamiltonian",
    "hamiltonian_blocks",
    "hva",
    "full_meta",
    "math_md_full_meta_v1",
    "math_md_full_meta",
    "pareto_lean",
    "pareto_lean_l3",
    "pareto_lean_l2",
    "pareto_lean_gate_pruned",
    "uccsd_paop_lf_full",
    "uccsd_otimes_paop_lf_std",
    "uccsd_otimes_paop_lf2_std",
    "uccsd_otimes_paop_bond_disp_std",
    "uccsd_otimes_paop_lf_std_seq2p",
    "uccsd_otimes_paop_lf2_std_seq2p",
    "uccsd_otimes_paop_bond_disp_std_seq2p",
    "sq_lf_std",
    "paop",
    "paop_min",
    "paop_std",
    "paop_full",
    "paop_lf",
    "paop_lf_std",
    "paop_lf2_std",
    "paop_lf3_std",
    "paop_lf4_std",
    "paop_lf_full",
    "paop_sq_std",
    "paop_sq_full",
    "paop_bond_disp_std",
    "paop_hop_sq_std",
    "paop_pair_sq_std",
    "vlf_only",
    "sq_only",
    "vlf_sq",
    "sq_dens_only",
    "vlf_sq_dens",
)

_SPINFUL_LATTICE_CAPABILITIES = HamiltonianFamilyCapabilities(
    observable_kind="spinful_lattice",
    primary_density_modes=("auto", "pair_difference", "staggered"),
    drive_operator_kind="spinful_lattice_density",
    supports_measurement_observables=True,
    supports_driven_realtime=True,
    supports_drive_exact_v1=True,
    supports_drive_benchmark_exact=True,
    supports_strict_qpu_faithful=True,
    supports_hamiltonian_flow_projective=True,
    report_manifest_fields=(
        "site_occupations",
        "n_up_site",
        "n_dn_site",
        "doublon",
        "staggered",
        "primary_density",
    ),
)
_HH_CAPABILITIES = HamiltonianFamilyCapabilities(
    observable_kind="hh_spinful_boson",
    primary_density_modes=("auto", "pair_difference", "staggered"),
    drive_operator_kind="hh_density_legacy",
    supports_measurement_observables=True,
    supports_driven_realtime=True,
    supports_drive_mode_off=True,
    supports_drive_exact_v1=True,
    supports_drive_benchmark_exact=True,
    supports_strict_qpu_faithful=True,
    report_manifest_fields=(
        "site_occupations",
        "n_up_site",
        "n_dn_site",
        "doublon",
        "staggered",
        "primary_density",
    ),
)
_MOLECULAR_CAPABILITIES = HamiltonianFamilyCapabilities(
    observable_kind="unsupported",
    primary_density_modes=("auto",),
    supports_measurement_observables=False,
    supports_driven_realtime=False,
    supports_drive_benchmark_exact=True,
    supports_strict_qpu_faithful=False,
    report_manifest_fields=("energy_total",),
)
_MOLECULAR_VIBRONIC_CAPABILITIES = HamiltonianFamilyCapabilities(
    observable_kind="molecular_vibronic_h2",
    primary_density_modes=("auto", "vibron_number"),
    drive_operator_kind="molecular_vibronic_h2_dhdr",
    supports_measurement_observables=True,
    supports_driven_realtime=True,
    supports_drive_exact_v1=True,
    supports_drive_benchmark_exact=True,
    supports_strict_qpu_faithful=True,
    supports_hamiltonian_flow_projective=True,
    report_manifest_fields=("energy_total", "vibron_number", "vibronic_h2_dhdr"),
)
_MOLECULAR_VIBRONIC_H2O_CAPABILITIES = HamiltonianFamilyCapabilities(
    observable_kind="molecular_vibronic_h2o",
    primary_density_modes=("auto", "vibron_number"),
    supports_measurement_observables=True,
    supports_driven_realtime=False,
    supports_drive_benchmark_exact=True,
    supports_strict_qpu_faithful=False,
    supports_hamiltonian_flow_projective=False,
    report_manifest_fields=("energy_total", "vibron_number"),
)
_MOLECULAR_VIBRONIC_H2O_LINEAR_FD_CAPABILITIES = HamiltonianFamilyCapabilities(
    observable_kind="molecular_vibronic_h2o_linear_fd",
    primary_density_modes=("auto", "vibron_number"),
    supports_measurement_observables=True,
    supports_driven_realtime=False,
    supports_drive_benchmark_exact=True,
    supports_strict_qpu_faithful=False,
    supports_hamiltonian_flow_projective=False,
    report_manifest_fields=("energy_total", "vibron_number"),
)
_SPINLESS_CAPABILITIES = HamiltonianFamilyCapabilities(
    observable_kind="spinless_lattice",
    primary_density_modes=("auto", "staggered", "pair_difference"),
    drive_operator_kind="spinless_lattice_density",
    supports_measurement_observables=True,
    supports_driven_realtime=True,
    supports_drive_exact_v1=True,
    supports_drive_benchmark_exact=True,
    supports_strict_qpu_faithful=True,
    supports_hamiltonian_flow_projective=True,
    report_manifest_fields=(
        "site_occupations",
        "spinless_particle_number",
        "spinless_staggered_density",
        "staggered",
        "primary_density",
    ),
)
_SPIN_BOSON_CAPABILITIES = HamiltonianFamilyCapabilities(
    observable_kind="spin_boson",
    primary_density_modes=("auto", "imbalance"),
    drive_operator_kind="spin_boson_imbalance",
    supports_measurement_observables=True,
    supports_driven_realtime=True,
    supports_drive_mode_off=True,
    supports_drive_exact_v1=True,
    supports_drive_benchmark_exact=True,
    supports_strict_qpu_faithful=True,
    supports_hamiltonian_flow_projective=True,
    report_manifest_fields=(
        "emitter_ground_occupation",
        "emitter_excited_occupation",
        "boson_number",
        "emitter_imbalance",
        "spin_x",
        "primary_density",
    ),
)
_BOSON_CHAIN_CAPABILITIES = HamiltonianFamilyCapabilities(
    observable_kind="boson_chain",
    primary_density_modes=("auto", "pair_difference", "staggered"),
    drive_operator_kind="boson_chain_number",
    supports_measurement_observables=True,
    supports_driven_realtime=True,
    supports_drive_exact_v1=True,
    supports_drive_benchmark_exact=True,
    supports_strict_qpu_faithful=True,
    supports_hamiltonian_flow_projective=True,
    report_manifest_fields=(
        "site_occupations",
        "boson_number_total",
        "staggered",
        "primary_density",
    ),
)
_HARMONIC_KERR_CHAIN_CAPABILITIES = HamiltonianFamilyCapabilities(
    observable_kind="boson_chain",
    primary_density_modes=("auto", "pair_difference", "staggered"),
    drive_operator_kind="harmonic_kerr_chain_displacement",
    supports_measurement_observables=True,
    supports_driven_realtime=True,
    supports_drive_exact_v1=True,
    supports_drive_benchmark_exact=True,
    supports_strict_qpu_faithful=True,
    supports_hamiltonian_flow_projective=True,
    report_manifest_fields=(
        "site_occupations",
        "boson_number_total",
        "staggered",
        "primary_density",
    ),
)

_PROBLEM_REGISTRY: dict[str, ProblemFamilySpec] = {
    "hubbard": ProblemFamilySpec(
        family_key="hubbard",
        default_controller_profile="phase3_v1",
        default_continuation_mode="phase3_v1",
        admissible_pool_keys=_HUBBARD_POOL_KEYS,
        default_pool_key="uccsd",
        default_pool_resolution_scope="family_default",
        supported_boson_encodings=("binary",),
        default_sector_label="half_filled_spin_sector",
        default_reference_label="hartree_fock",
        exact_target_label="exact_ground_energy_sector",
        exact_comparison_space_label="full_register",
        _layout_builder=_build_hubbard_layout,
        capabilities=_SPINFUL_LATTICE_CAPABILITIES,
    ),
    "hh": ProblemFamilySpec(
        family_key="hh",
        default_controller_profile="phase3_v1",
        default_continuation_mode="phase3_v1",
        admissible_pool_keys=_HH_POOL_KEYS,
        default_pool_key=None,
        default_pool_resolution_scope="controller_resolved",
        supported_boson_encodings=("binary",),
        default_sector_label="half_filled_fermion_sector",
        default_reference_label="hubbard_holstein_reference_state",
        exact_target_label="exact_ground_energy_sector_hh",
        exact_comparison_space_label="fermion_register_with_truncated_bosons",
        _layout_builder=_build_hh_layout,
        capabilities=_HH_CAPABILITIES,
    ),
    "molecular_restricted_closed_shell": ProblemFamilySpec(
        family_key="molecular_restricted_closed_shell",
        default_controller_profile="phase3_v1",
        default_continuation_mode="phase3_v1",
        admissible_pool_keys=_MOLECULAR_POOL_KEYS,
        default_pool_key="uccsd",
        default_pool_resolution_scope="family_default",
        supported_boson_encodings=(),
        default_sector_label="closed_shell_fixed_number_sector",
        default_reference_label="restricted_hartree_fock",
        exact_target_label="exact_ground_energy_sector_molecular",
        exact_comparison_space_label="spin_orbital_register",
        _layout_builder=_build_hubbard_layout,
        capabilities=_MOLECULAR_CAPABILITIES,
        _context_resolver=_resolve_molecular_problem_context,
    ),
    "molecular_vibronic_h2": ProblemFamilySpec(
        family_key="molecular_vibronic_h2",
        default_controller_profile="phase3_v1",
        default_continuation_mode="phase3_v1",
        admissible_pool_keys=_MOLECULAR_VIBRONIC_POOL_KEYS,
        default_pool_key="full_meta",
        default_pool_resolution_scope="family_default",
        supported_boson_encodings=("binary",),
        default_sector_label="closed_shell_fermions_with_truncated_vibration",
        default_reference_label="restricted_hf_times_boson_vacuum",
        exact_target_label="exact_ground_energy_molecular_vibronic_h2_physical_sector",
        exact_comparison_space_label="spin_orbital_register_with_truncated_vibration",
        _layout_builder=_build_molecular_vibronic_h2_layout,
        capabilities=_MOLECULAR_VIBRONIC_CAPABILITIES,
        _context_resolver=_resolve_molecular_vibronic_h2_context,
    ),
    "molecular_vibronic_h2o": ProblemFamilySpec(
        family_key="molecular_vibronic_h2o",
        default_controller_profile="phase3_v1",
        default_continuation_mode="phase3_v1",
        admissible_pool_keys=_MOLECULAR_VIBRONIC_POOL_KEYS,
        default_pool_key="full_meta",
        default_pool_resolution_scope="family_default",
        supported_boson_encodings=("binary",),
        default_sector_label="closed_shell_fermions_with_truncated_vibration",
        default_reference_label="restricted_hf_times_boson_vacuum",
        exact_target_label="exact_ground_energy_molecular_vibronic_h2o_physical_sector",
        exact_comparison_space_label="spin_orbital_register_with_truncated_vibration",
        _layout_builder=_build_molecular_vibronic_h2o_layout,
        capabilities=_MOLECULAR_VIBRONIC_H2O_CAPABILITIES,
        _context_resolver=_resolve_molecular_vibronic_h2o_context,
    ),
    "molecular_vibronic_h2o_linear_fd": ProblemFamilySpec(
        family_key="molecular_vibronic_h2o_linear_fd",
        default_controller_profile="phase3_v1",
        default_continuation_mode="phase3_v1",
        admissible_pool_keys=_MOLECULAR_VIBRONIC_POOL_KEYS,
        default_pool_key="full_meta",
        default_pool_resolution_scope="family_default",
        supported_boson_encodings=("binary",),
        default_sector_label="closed_shell_fermions_with_linear_fd_truncated_vibrations",
        default_reference_label="restricted_hf_times_linear_fd_boson_vacuum",
        exact_target_label="exact_ground_energy_molecular_vibronic_h2o_linear_fd_physical_sector",
        exact_comparison_space_label="spin_orbital_register_with_linear_fd_truncated_vibrations",
        _layout_builder=_build_molecular_vibronic_h2o_linear_fd_layout,
        capabilities=_MOLECULAR_VIBRONIC_H2O_LINEAR_FD_CAPABILITIES,
        _context_resolver=_resolve_molecular_vibronic_h2o_linear_fd_context,
    ),
    "ionic_hubbard": ProblemFamilySpec(
        family_key="ionic_hubbard",
        default_controller_profile="phase3_v1",
        default_continuation_mode="phase3_v1",
        admissible_pool_keys=_SPINFUL_LATTICE_POOL_KEYS,
        default_pool_key="hamiltonian_quadratures",
        default_pool_resolution_scope="family_default",
        supported_boson_encodings=(),
        default_sector_label="half_filled_spin_sector",
        default_reference_label="hartree_fock",
        exact_target_label="exact_ground_energy_sector",
        exact_comparison_space_label="full_register",
        _layout_builder=_build_hubbard_layout,
        capabilities=_SPINFUL_LATTICE_CAPABILITIES,
    ),
    "extended_hubbard": ProblemFamilySpec(
        family_key="extended_hubbard",
        default_controller_profile="phase3_v1",
        default_continuation_mode="phase3_v1",
        admissible_pool_keys=_SPINFUL_LATTICE_POOL_KEYS,
        default_pool_key="hamiltonian_quadratures",
        default_pool_resolution_scope="family_default",
        supported_boson_encodings=(),
        default_sector_label="half_filled_spin_sector",
        default_reference_label="hartree_fock",
        exact_target_label="exact_ground_energy_sector",
        exact_comparison_space_label="full_register",
        _layout_builder=_build_hubbard_layout,
        capabilities=_SPINFUL_LATTICE_CAPABILITIES,
    ),
    "ttprime_hubbard": ProblemFamilySpec(
        family_key="ttprime_hubbard",
        default_controller_profile="phase3_v1",
        default_continuation_mode="phase3_v1",
        admissible_pool_keys=_SPINFUL_LATTICE_POOL_KEYS,
        default_pool_key="hamiltonian_quadratures",
        default_pool_resolution_scope="family_default",
        supported_boson_encodings=(),
        default_sector_label="half_filled_spin_sector",
        default_reference_label="hartree_fock",
        exact_target_label="exact_ground_energy_sector",
        exact_comparison_space_label="full_register",
        _layout_builder=_build_hubbard_layout,
        capabilities=_SPINFUL_LATTICE_CAPABILITIES,
    ),
    "spinless_tv": ProblemFamilySpec(
        family_key="spinless_tv",
        default_controller_profile="phase3_v1",
        default_continuation_mode="phase3_v1",
        admissible_pool_keys=_SPINLESS_POOL_KEYS,
        default_pool_key="hamiltonian_quadratures",
        default_pool_resolution_scope="family_default",
        supported_boson_encodings=(),
        default_sector_label="fixed_spinless_sector",
        default_reference_label="spinless_fermion_filling",
        exact_target_label="exact_ground_energy_spinless_fixed_count",
        exact_comparison_space_label="spinless_fermion_register",
        _layout_builder=_build_spinless_layout,
        capabilities=_SPINLESS_CAPABILITIES,
    ),
    "spin_boson": ProblemFamilySpec(
        family_key="spin_boson",
        default_controller_profile="phase3_v1",
        default_continuation_mode="phase3_v1",
        admissible_pool_keys=_SPIN_BOSON_POOL_KEYS,
        default_pool_key="full_meta",
        default_pool_resolution_scope="family_default",
        supported_boson_encodings=("binary", "unary"),
        default_sector_label="single_emitter_truncated_boson_sector",
        default_reference_label="spin_boson_uncoupled_ground",
        exact_target_label="exact_ground_energy_spin_boson",
        exact_comparison_space_label="one_emitter_truncated_boson_register",
        _layout_builder=_build_spin_boson_layout,
        capabilities=_SPIN_BOSON_CAPABILITIES,
        _context_resolver=_resolve_spin_boson_problem_context,
    ),
    "bose_hubbard": ProblemFamilySpec(
        family_key="bose_hubbard",
        default_controller_profile="phase3_v1",
        default_continuation_mode="phase3_v1",
        admissible_pool_keys=_BOSON_CHAIN_POOL_KEYS,
        default_pool_key="full_meta",
        default_pool_resolution_scope="family_default",
        supported_boson_encodings=("binary", "unary"),
        default_sector_label="truncated_boson_register",
        default_reference_label="boson_vacuum",
        exact_target_label="exact_ground_energy_boson_only",
        exact_comparison_space_label="truncated_boson_register",
        _layout_builder=_build_boson_chain_layout,
        capabilities=_BOSON_CHAIN_CAPABILITIES,
    ),
    "harmonic_kerr_chain": ProblemFamilySpec(
        family_key="harmonic_kerr_chain",
        default_controller_profile="phase3_v1",
        default_continuation_mode="phase3_v1",
        admissible_pool_keys=_BOSON_CHAIN_POOL_KEYS,
        default_pool_key="full_meta",
        default_pool_resolution_scope="family_default",
        supported_boson_encodings=("binary", "unary"),
        default_sector_label="truncated_boson_register",
        default_reference_label="boson_vacuum",
        exact_target_label="exact_ground_energy_boson_only",
        exact_comparison_space_label="truncated_boson_register",
        _layout_builder=_build_boson_chain_layout,
        capabilities=_HARMONIC_KERR_CHAIN_CAPABILITIES,
    ),
}


def available_problem_keys() -> tuple[str, ...]:
    return tuple(_PROBLEM_REGISTRY.keys())


def available_adapt_pool_keys() -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for family_key in available_problem_keys():
        for pool_key in _PROBLEM_REGISTRY[family_key].admissible_pool_keys:
            if pool_key in seen:
                continue
            seen.add(pool_key)
            ordered.append(str(pool_key))
    return tuple(ordered)


def get_problem_family_spec(problem_key: str) -> ProblemFamilySpec:
    key = canonical_problem_key(problem_key)
    try:
        return _PROBLEM_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported problem family {problem_key!r}. Known problems: {available_problem_keys()}."
        ) from exc


def default_continuation_mode_for_problem(problem_key: str) -> str:
    return str(get_problem_family_spec(problem_key).default_continuation_mode)


def resolve_runtime_default_pool_key(
    resolved_problem: ResolvedProblemContext,
    *,
    continuation_mode: str,
) -> str:
    if (
        resolved_problem.default_pool_key is not None
        and str(resolved_problem.default_pool_resolution_scope) == "family_default"
    ):
        return str(resolved_problem.default_pool_key)
    if str(resolved_problem.family_key) == "hh":
        if str(continuation_mode).strip().lower() in {"phase1_v1", "phase2_v1", "phase3_v1"}:
            return "paop_lf_std"
        return "full_meta"
    if resolved_problem.default_pool_key is not None:
        return str(resolved_problem.default_pool_key)
    raise ValueError(
        f"No runtime default pool key is defined for family {resolved_problem.family_key!r}."
    )


def supported_continuation_modes_for_problem(problem_key: str) -> tuple[str, ...]:
    """Return phase-1 compatibility tokens accepted for this family."""
    key = canonical_problem_key(problem_key)
    if key in _PROBLEM_REGISTRY:
        return ("legacy", "phase1_v1", "phase2_v1", "phase3_v1")
    return ("legacy", "phase1_v1", "phase2_v1", "phase3_v1")


def resolve_problem_context(
    request: ProblemRequest,
    *,
    hamiltonian: Any | None = None,
    exact_energy_impl: Callable[..., float] | None = None,
) -> ResolvedProblemContext:
    return get_problem_family_spec(request.problem_key).resolve(
        request,
        hamiltonian=hamiltonian,
        exact_energy_impl=exact_energy_impl,
    )


def resolve_problem_context_from_namespace(
    args: Any,
    *,
    hamiltonian: Any | None = None,
    exact_energy_impl: Callable[..., float] | None = None,
) -> ResolvedProblemContext:
    request = ProblemRequest.from_namespace(args)
    return resolve_problem_context(
        request,
        hamiltonian=hamiltonian,
        exact_energy_impl=exact_energy_impl,
    )


__all__ = [
    "ExactTargetSpec",
    "FixedCountConstraint",
    "HamiltonianFamilyCapabilities",
    "ParityConstraint",
    "ProblemRequest",
    "ProblemFamilySpec",
    "RegisterBlockSpec",
    "RegisterLayoutSpec",
    "ReferenceStateSpec",
    "ResolvedProblemContext",
    "SectorSelection",
    "TruncationConstraint",
    "WeightedChargeConstraint",
    "available_adapt_pool_keys",
    "available_problem_keys",
    "canonical_problem_key",
    "default_continuation_mode_for_problem",
    "get_problem_family_spec",
    "resolve_problem_context",
    "resolve_problem_context_from_namespace",
    "resolve_runtime_default_pool_key",
    "supported_continuation_modes_for_problem",
]
