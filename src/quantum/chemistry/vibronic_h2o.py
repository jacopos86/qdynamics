from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.quantum.chemistry.molecular_hamiltonian import build_restricted_closed_shell_molecular_hamiltonian
from src.quantum.chemistry.molecular_uccsd import build_molecular_uccsd_pool
from src.quantum.chemistry.psi4_adapter import (
    RestrictedClosedShellMolecularProblem,
    load_restricted_closed_shell_problem_from_json,
)
from src.quantum.chemistry.vibronic_h2 import (
    _boson_momentum_operator,
    _clean_real_polynomial,
    _fermion_density_product_operator,
    _fermion_number_operator,
    _fermion_orbital_mixing_operator,
    _fermion_pair_exchange_operator,
    _lift_fermion_polynomial,
    _polynomial_from_jsonable,
    _polynomial_to_jsonable,
    _state_from_sparse_jsonable,
    _state_to_sparse_jsonable,
    build_vibronic_reference_state,
    exact_ground_energy_physical_sector,
)
from src.quantum.hubbard_latex_python_pairs import (
    boson_operator,
    boson_qubits_per_site,
    phonon_qubit_indices_for_site,
)
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


H2O_VIBRONIC_FIXTURE_SCHEMA = "molecular_vibronic_h2o_fixture_v1"
H2O_VIBRONIC_FAMILY_KEY = "molecular_vibronic_h2o"
H2O_VIBRONIC_ACTIVE_SPACE_KIND = "h2o_frontier_two_spatial_orbital_projection"
H2O_VIBRONIC_DERIVATIVE_SOURCE = "frontier_gap_surrogate_dQ_v1"
H2O_VIBRONIC_DEFAULT_OMEGA_AU = 0.017
H2O_VIBRONIC_DEFAULT_REDUCED_MASS_AU = 1728.256
H2O_VIBRONIC_DEFAULT_DERIVATIVE_SCALE_AU_PER_BOHR = 0.05


@dataclass(frozen=True)
class VibronicH2OModel:
    geometry_spec: str
    basis: str
    active_space_kind: str
    selected_spatial_orbital_indices: tuple[int, int]
    source_n_spatial_orbitals: int
    source_n_spin_orbitals: int
    n_ph_max: int
    boson_encoding: str
    n_fermion_qubits: int
    n_boson_qubits: int
    n_total_qubits: int
    omega_au: float
    reduced_mass_au: float
    x_zpf_bohr: float
    derivative_scale_au_per_bohr: float
    derivative_source: str
    electronic_hf_energy_center: float
    h_electronic: Any
    dH_dQ: Any
    h_vibronic: Any
    pool: tuple[AnsatzTerm, ...]
    psi_ref: np.ndarray
    coupling_scale: float = 1.0

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "molecule": "H2O",
            "geometry_spec": str(self.geometry_spec),
            "basis": str(self.basis),
            "active_space_kind": str(self.active_space_kind),
            "selected_spatial_orbital_indices": [int(i) for i in self.selected_spatial_orbital_indices],
            "source_n_spatial_orbitals": int(self.source_n_spatial_orbitals),
            "source_n_spin_orbitals": int(self.source_n_spin_orbitals),
            "n_ph_max": int(self.n_ph_max),
            "boson_encoding": str(self.boson_encoding),
            "n_fermion_qubits": int(self.n_fermion_qubits),
            "n_boson_qubits": int(self.n_boson_qubits),
            "n_total_qubits": int(self.n_total_qubits),
            "omega_au": float(self.omega_au),
            "reduced_mass_au": float(self.reduced_mass_au),
            "x_zpf_bohr": float(self.x_zpf_bohr),
            "coupling_scale": float(self.coupling_scale),
            "derivative_scale_au_per_bohr": float(self.derivative_scale_au_per_bohr),
            "derivative_source": str(self.derivative_source),
            "electronic_hf_energy_center": float(self.electronic_hf_energy_center),
            "pool_size": int(len(self.pool)),
            "model_status": "active_space_one_mode_prototype_not_full_parent_hamiltonian",
        }


@dataclass(frozen=True)
class CachedVibronicH2OFixture:
    model: VibronicH2OModel
    exact_ground_energy: float | None
    fixture_path: Path
    metadata: dict[str, Any]


def default_vibronic_h2o_fixture_path() -> Path:
    return Path(__file__).resolve().parents[3] / "test_support" / "molecular_vibronic_h2o_sto3g_active2_fd001.json"


def default_h2o_source_problem_path() -> Path:
    return Path(__file__).resolve().parent / "h2o_sto3g_fast_result.json"


def _validate_binary_one_mode(*, n_ph_max: int, boson_encoding: str, ordering: str) -> None:
    if str(ordering).strip().lower() != "blocked":
        raise ValueError("molecular_vibronic_h2o currently supports ordering='blocked' only.")
    if int(n_ph_max) < 1:
        raise ValueError("n_ph_max must be >= 1.")
    if str(boson_encoding).strip().lower() != "binary":
        raise ValueError("molecular_vibronic_h2o currently supports boson_encoding='binary' only.")


def select_h2o_frontier_active_space_indices(
    problem: RestrictedClosedShellMolecularProblem,
    *,
    selected_spatial_orbital_indices: Sequence[int] | None = None,
) -> tuple[int, int]:
    if selected_spatial_orbital_indices is None:
        selected = (int(problem.n_alpha) - 1, int(problem.n_alpha))
    else:
        selected = tuple(int(i) for i in selected_spatial_orbital_indices)
    if len(selected) != 2 or tuple(sorted(selected)) != tuple(selected) or len(set(selected)) != 2:
        raise ValueError("H2O vibronic active space requires two unique, increasing spatial orbital indices.")
    if selected[0] < 0 or selected[-1] >= int(problem.n_spatial_orbitals):
        raise ValueError("H2O vibronic selected spatial orbital indices are out of range.")
    if int(problem.n_alpha) < 1 or int(problem.n_beta) < 1:
        raise ValueError("H2O vibronic active-space prototype requires at least one alpha and beta electron.")
    return int(selected[0]), int(selected[1])


def project_h2o_problem_to_frontier_active_space(
    problem: RestrictedClosedShellMolecularProblem,
    *,
    selected_spatial_orbital_indices: Sequence[int] | None = None,
) -> RestrictedClosedShellMolecularProblem:
    selected = select_h2o_frontier_active_space_indices(
        problem,
        selected_spatial_orbital_indices=selected_spatial_orbital_indices,
    )
    idx = np.asarray(selected, dtype=int)
    h_parent = np.asarray(problem.one_body_integrals_mo, dtype=float)
    eri_parent = np.asarray(problem.two_body_integrals_mo, dtype=float)
    h_active = h_parent[np.ix_(idx, idx)]
    eri_active = eri_parent[np.ix_(idx, idx, idx, idx)]
    hf_total = float(problem.nuclear_repulsion_energy + 2.0 * h_active[0, 0] + eri_active[0, 0, 0, 0])
    return RestrictedClosedShellMolecularProblem(
        geometry_spec=str(problem.geometry_spec),
        basis=str(problem.basis),
        charge=int(problem.charge),
        multiplicity=int(problem.multiplicity),
        reference=str(problem.reference),
        n_spatial_orbitals=2,
        n_alpha=1,
        n_beta=1,
        hf_energy=float(hf_total),
        nuclear_repulsion_energy=float(problem.nuclear_repulsion_energy),
        one_body_integrals_mo=np.asarray(h_active, dtype=float),
        two_body_integrals_mo=np.asarray(eri_active, dtype=float),
    )


def _frontier_gap_surrogate_derivative(
    *,
    derivative_scale_au_per_bohr: float,
) -> Any:
    nq_fermion = 4
    occ = _clean_real_polynomial(
        _fermion_number_operator(nq_fermion=nq_fermion, orbital=0)
        + _fermion_number_operator(nq_fermion=nq_fermion, orbital=2)
    )
    virt = _clean_real_polynomial(
        _fermion_number_operator(nq_fermion=nq_fermion, orbital=1)
        + _fermion_number_operator(nq_fermion=nq_fermion, orbital=3)
    )
    return _clean_real_polynomial(float(derivative_scale_au_per_bohr) * (virt - occ))


def _build_pool(
    *,
    n_ph_max: int,
    boson_encoding: str,
    qpb: int,
    n_total_qubits: int,
    dH_dQ_lifted: Any,
) -> tuple[AnsatzTerm, ...]:
    p_b = _boson_momentum_operator(
        nq_total=int(n_total_qubits),
        boson_qubits=phonon_qubit_indices_for_site(0, n_sites=1, qpb=int(qpb), fermion_qubits=4),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
    )
    pool: list[AnsatzTerm] = []
    for term in build_molecular_uccsd_pool(
        n_spatial_orbitals=2,
        num_particles=(1, 1),
        ordering="blocked",
    ):
        lifted = _clean_real_polynomial(_lift_fermion_polynomial(term.polynomial, boson_qubits=int(qpb)))
        if lifted.count_number_terms() > 0:
            pool.append(AnsatzTerm(label=f"el::{term.label}", polynomial=lifted))
    pool.append(AnsatzTerm(label="boson::p", polynomial=p_b))
    coupled_dq_p = _clean_real_polynomial(dH_dQ_lifted * p_b)
    if coupled_dq_p.count_number_terms() > 0:
        pool.append(AnsatzTerm(label="coupled::dH_dQ_times_p", polynomial=coupled_dq_p))

    number_ops = [
        _clean_real_polynomial(
            _lift_fermion_polynomial(
                _fermion_number_operator(nq_fermion=4, orbital=int(j)),
                boson_qubits=int(qpb),
            )
        )
        for j in range(4)
    ]
    occ = _clean_real_polynomial(number_ops[0] + number_ops[2])
    virt = _clean_real_polynomial(number_ops[1] + number_ops[3])
    for label, channel in (
        ("frontier_occ", occ),
        ("frontier_virt", virt),
        ("frontier_imbalance", _clean_real_polynomial(virt - occ)),
    ):
        coupled = _clean_real_polynomial(channel * p_b)
        if coupled.count_number_terms() > 0:
            pool.append(AnsatzTerm(label=f"coupled::occ::{label}::p", polynomial=coupled))

    factored_channels = (
        (
            "frontier_mix_x",
            _clean_real_polynomial(
                _lift_fermion_polynomial(
                    _clean_real_polynomial(
                        _fermion_orbital_mixing_operator(nq_fermion=4, left_orbital=0, right_orbital=1)
                        + _fermion_orbital_mixing_operator(nq_fermion=4, left_orbital=2, right_orbital=3)
                    ),
                    boson_qubits=int(qpb),
                )
            ),
        ),
        (
            "frontier_pair_x",
            _clean_real_polynomial(
                _lift_fermion_polynomial(
                    _fermion_pair_exchange_operator(
                        nq_fermion=4,
                        bond_alpha=0,
                        anti_alpha=1,
                        bond_beta=2,
                        anti_beta=3,
                    ),
                    boson_qubits=int(qpb),
                )
            ),
        ),
        (
            "frontier_occ_pair",
            _clean_real_polynomial(
                _lift_fermion_polynomial(
                    _fermion_density_product_operator(nq_fermion=4, orbitals=(0, 2)),
                    boson_qubits=int(qpb),
                )
            ),
        ),
        (
            "frontier_virt_pair",
            _clean_real_polynomial(
                _lift_fermion_polynomial(
                    _fermion_density_product_operator(nq_fermion=4, orbitals=(1, 3)),
                    boson_qubits=int(qpb),
                )
            ),
        ),
    )
    for label, channel in factored_channels:
        coupled = _clean_real_polynomial(channel * p_b)
        if coupled.count_number_terms() > 0:
            pool.append(AnsatzTerm(label=f"coupled::factored::{label}::p", polynomial=coupled))
    return tuple(pool)


def build_vibronic_h2o_model_from_active_problem(
    active_problem: RestrictedClosedShellMolecularProblem,
    *,
    source_problem: RestrictedClosedShellMolecularProblem,
    selected_spatial_orbital_indices: Sequence[int],
    n_ph_max: int = 1,
    boson_encoding: str = "binary",
    coupling_scale: float = 1.0,
    omega_au: float = H2O_VIBRONIC_DEFAULT_OMEGA_AU,
    reduced_mass_au: float = H2O_VIBRONIC_DEFAULT_REDUCED_MASS_AU,
    derivative_scale_au_per_bohr: float = H2O_VIBRONIC_DEFAULT_DERIVATIVE_SCALE_AU_PER_BOHR,
    derivative_source: str = H2O_VIBRONIC_DERIVATIVE_SOURCE,
    ordering: str = "blocked",
) -> VibronicH2OModel:
    _validate_binary_one_mode(n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding), ordering=str(ordering))
    if int(active_problem.n_spatial_orbitals) != 2 or tuple(active_problem.num_particles) != (1, 1):
        raise ValueError("molecular_vibronic_h2o runtime active problem must be two spatial orbitals with (1,1) electrons.")
    if float(omega_au) <= 0.0 or not np.isfinite(float(omega_au)):
        raise ValueError("omega_au must be positive and finite.")
    if float(reduced_mass_au) <= 0.0 or not np.isfinite(float(reduced_mass_au)):
        raise ValueError("reduced_mass_au must be positive and finite.")
    if not np.isfinite(float(coupling_scale)):
        raise ValueError("coupling_scale must be finite.")
    if not np.isfinite(float(derivative_scale_au_per_bohr)):
        raise ValueError("derivative_scale_au_per_bohr must be finite.")

    n_fermion_qubits = 4
    qpb = int(boson_qubits_per_site(int(n_ph_max), str(boson_encoding)))
    n_total_qubits = int(n_fermion_qubits) + int(qpb)
    boson_qubits = phonon_qubit_indices_for_site(0, n_sites=1, qpb=int(qpb), fermion_qubits=int(n_fermion_qubits))

    h_electronic = build_restricted_closed_shell_molecular_hamiltonian(active_problem, ordering="blocked")
    dH_dQ = _frontier_gap_surrogate_derivative(
        derivative_scale_au_per_bohr=float(derivative_scale_au_per_bohr),
    )
    h_center_lifted = _lift_fermion_polynomial(h_electronic, boson_qubits=int(qpb))
    dH_dQ_lifted = _lift_fermion_polynomial(dH_dQ, boson_qubits=int(qpb))
    n_b = _clean_real_polynomial(
        boson_operator(
            "JW",
            int(n_total_qubits),
            boson_qubits,
            which="n",
            n_ph_max=int(n_ph_max),
            encoding=str(boson_encoding),
        )
    )
    x_b = _clean_real_polynomial(
        boson_operator(
            "JW",
            int(n_total_qubits),
            boson_qubits,
            which="x",
            n_ph_max=int(n_ph_max),
            encoding=str(boson_encoding),
        )
    )
    x_zpf_bohr = float(math.sqrt(1.0 / (2.0 * float(reduced_mass_au) * float(omega_au))))
    h_vibronic = _clean_real_polynomial(
        h_center_lifted
        + (float(omega_au) * n_b)
        + float(0.5 * omega_au)
        + (float(coupling_scale) * float(x_zpf_bohr)) * (dH_dQ_lifted * x_b)
    )
    psi_ref = build_vibronic_reference_state(
        n_spatial_orbitals=2,
        num_particles=(1, 1),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        ordering="blocked",
    )
    pool = _build_pool(
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        qpb=int(qpb),
        n_total_qubits=int(n_total_qubits),
        dH_dQ_lifted=dH_dQ_lifted,
    )
    return VibronicH2OModel(
        geometry_spec=str(source_problem.geometry_spec),
        basis=str(source_problem.basis),
        active_space_kind=H2O_VIBRONIC_ACTIVE_SPACE_KIND,
        selected_spatial_orbital_indices=tuple(int(i) for i in selected_spatial_orbital_indices),
        source_n_spatial_orbitals=int(source_problem.n_spatial_orbitals),
        source_n_spin_orbitals=int(source_problem.n_spin_orbitals),
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        n_fermion_qubits=int(n_fermion_qubits),
        n_boson_qubits=int(qpb),
        n_total_qubits=int(n_total_qubits),
        omega_au=float(omega_au),
        reduced_mass_au=float(reduced_mass_au),
        x_zpf_bohr=float(x_zpf_bohr),
        derivative_scale_au_per_bohr=float(derivative_scale_au_per_bohr),
        derivative_source=str(derivative_source),
        electronic_hf_energy_center=float(active_problem.hf_energy),
        h_electronic=h_electronic,
        dH_dQ=dH_dQ,
        h_vibronic=h_vibronic,
        pool=tuple(pool),
        psi_ref=np.asarray(psi_ref, dtype=complex),
        coupling_scale=float(coupling_scale),
    )


def build_vibronic_h2o_model_from_source_problem(
    source_problem: RestrictedClosedShellMolecularProblem,
    *,
    selected_spatial_orbital_indices: Sequence[int] | None = None,
    n_ph_max: int = 1,
    boson_encoding: str = "binary",
    coupling_scale: float = 1.0,
    omega_au: float = H2O_VIBRONIC_DEFAULT_OMEGA_AU,
    reduced_mass_au: float = H2O_VIBRONIC_DEFAULT_REDUCED_MASS_AU,
    derivative_scale_au_per_bohr: float = H2O_VIBRONIC_DEFAULT_DERIVATIVE_SCALE_AU_PER_BOHR,
    ordering: str = "blocked",
) -> VibronicH2OModel:
    selected = select_h2o_frontier_active_space_indices(
        source_problem,
        selected_spatial_orbital_indices=selected_spatial_orbital_indices,
    )
    active = project_h2o_problem_to_frontier_active_space(
        source_problem,
        selected_spatial_orbital_indices=selected,
    )
    return build_vibronic_h2o_model_from_active_problem(
        active,
        source_problem=source_problem,
        selected_spatial_orbital_indices=selected,
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        coupling_scale=float(coupling_scale),
        omega_au=float(omega_au),
        reduced_mass_au=float(reduced_mass_au),
        derivative_scale_au_per_bohr=float(derivative_scale_au_per_bohr),
        ordering=str(ordering),
    )


def build_default_vibronic_h2o_model(
    *,
    n_ph_max: int = 1,
    boson_encoding: str = "binary",
    coupling_scale: float = 1.0,
    ordering: str = "blocked",
) -> VibronicH2OModel:
    source = load_restricted_closed_shell_problem_from_json(default_h2o_source_problem_path())
    return build_vibronic_h2o_model_from_source_problem(
        source,
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        coupling_scale=float(coupling_scale),
        ordering=str(ordering),
    )


def vibronic_h2o_fixture_to_jsonable(
    model: VibronicH2OModel,
    *,
    exact_ground_energy: float | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": H2O_VIBRONIC_FIXTURE_SCHEMA,
        "family_key": H2O_VIBRONIC_FAMILY_KEY,
        "model": model.to_jsonable(),
        "polynomials": {
            "h_electronic": _polynomial_to_jsonable(model.h_electronic),
            "dH_dQ": _polynomial_to_jsonable(model.dH_dQ),
            "h_vibronic": _polynomial_to_jsonable(model.h_vibronic),
        },
        "pool": [
            {
                "label": str(term.label),
                "polynomial": _polynomial_to_jsonable(term.polynomial),
            }
            for term in model.pool
        ],
        "reference_state": {
            "kind": "restricted_hf_times_boson_vacuum",
            **_state_to_sparse_jsonable(model.psi_ref),
        },
        "exact": {
            "ground_energy_physical_sector": None if exact_ground_energy is None else float(exact_ground_energy),
            "method": "dense_physical_sector",
        },
        "provenance": dict(provenance or {}),
    }


def load_cached_vibronic_h2o_fixture(path: str | Path | None = None) -> CachedVibronicH2OFixture:
    fixture_path = default_vibronic_h2o_fixture_path() if path in {None, ""} else Path(path)
    raw = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Vibronic H2O fixture must be a JSON object.")
    if raw.get("schema") != H2O_VIBRONIC_FIXTURE_SCHEMA:
        raise ValueError(f"Unsupported vibronic H2O fixture schema: {raw.get('schema')!r}")
    if raw.get("family_key") != H2O_VIBRONIC_FAMILY_KEY:
        raise ValueError(f"Unexpected vibronic H2O fixture family_key: {raw.get('family_key')!r}")
    model_payload = raw.get("model")
    polynomials = raw.get("polynomials")
    if not isinstance(model_payload, dict) or not isinstance(polynomials, dict):
        raise ValueError("Vibronic H2O fixture missing model/polynomials payloads.")
    n_total = int(model_payload["n_total_qubits"])
    n_fermion = int(model_payload["n_fermion_qubits"])
    n_boson = int(model_payload["n_boson_qubits"])
    if int(n_fermion) != 4:
        raise ValueError("Vibronic H2O v1 fixture must have four active-space fermion qubits.")
    if int(n_total) != int(n_fermion) + int(n_boson):
        raise ValueError("Vibronic H2O fixture qubit counts are inconsistent.")
    if int(model_payload["n_ph_max"]) != 1 or str(model_payload["boson_encoding"]) != "binary":
        raise ValueError("Vibronic H2O checked fixture must be n_ph_max=1, binary encoding.")
    h_electronic = _polynomial_from_jsonable(polynomials.get("h_electronic"), expected_nq=int(n_fermion))
    dH_dQ = _polynomial_from_jsonable(polynomials.get("dH_dQ"), expected_nq=int(n_fermion))
    h_vibronic = _polynomial_from_jsonable(polynomials.get("h_vibronic"), expected_nq=int(n_total))
    pool_payload = raw.get("pool")
    if not isinstance(pool_payload, list) or not pool_payload:
        raise ValueError("Vibronic H2O fixture must contain a non-empty pool.")
    pool = tuple(
        AnsatzTerm(
            label=str(row["label"]),
            polynomial=_polynomial_from_jsonable(row.get("polynomial"), expected_nq=int(n_total)),
        )
        for row in pool_payload
        if isinstance(row, dict)
    )
    if not pool:
        raise ValueError("Vibronic H2O fixture pool is empty after parsing.")
    psi_ref = _state_from_sparse_jsonable(raw.get("reference_state"), expected_nq=int(n_total))
    norm = float(np.linalg.norm(psi_ref))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("Vibronic H2O fixture reference state has invalid norm.")
    psi_ref = psi_ref / norm
    exact_payload = raw.get("exact", {})
    exact_energy = None
    if isinstance(exact_payload, dict) and exact_payload.get("ground_energy_physical_sector") is not None:
        exact_energy = float(exact_payload["ground_energy_physical_sector"])
    if exact_energy is None:
        exact_energy = exact_ground_energy_physical_sector(
            h_vibronic,
            n_spatial_orbitals=2,
            num_particles=(1, 1),
            n_ph_max=1,
            boson_encoding="binary",
        )
    model = VibronicH2OModel(
        geometry_spec=str(model_payload["geometry_spec"]),
        basis=str(model_payload["basis"]),
        active_space_kind=str(model_payload["active_space_kind"]),
        selected_spatial_orbital_indices=tuple(int(i) for i in model_payload["selected_spatial_orbital_indices"]),
        source_n_spatial_orbitals=int(model_payload["source_n_spatial_orbitals"]),
        source_n_spin_orbitals=int(model_payload["source_n_spin_orbitals"]),
        n_ph_max=int(model_payload["n_ph_max"]),
        boson_encoding=str(model_payload["boson_encoding"]),
        n_fermion_qubits=int(n_fermion),
        n_boson_qubits=int(n_boson),
        n_total_qubits=int(n_total),
        omega_au=float(model_payload["omega_au"]),
        reduced_mass_au=float(model_payload["reduced_mass_au"]),
        x_zpf_bohr=float(model_payload["x_zpf_bohr"]),
        derivative_scale_au_per_bohr=float(model_payload["derivative_scale_au_per_bohr"]),
        derivative_source=str(model_payload["derivative_source"]),
        electronic_hf_energy_center=float(model_payload["electronic_hf_energy_center"]),
        h_electronic=h_electronic,
        dH_dQ=dH_dQ,
        h_vibronic=h_vibronic,
        pool=pool,
        psi_ref=np.asarray(psi_ref, dtype=complex),
        coupling_scale=float(model_payload.get("coupling_scale", 1.0)),
    )
    return CachedVibronicH2OFixture(
        model=model,
        exact_ground_energy=float(exact_energy),
        fixture_path=Path(fixture_path),
        metadata={
            "schema": str(raw.get("schema")),
            "family_key": str(raw.get("family_key")),
            "model": dict(model_payload),
            "provenance": dict(raw.get("provenance", {})) if isinstance(raw.get("provenance", {}), dict) else {},
        },
    )


def _rebuild_from_cached_components(
    base: VibronicH2OModel,
    *,
    n_ph_max: int,
    boson_encoding: str,
    coupling_scale: float,
    ordering: str,
) -> VibronicH2OModel:
    _validate_binary_one_mode(n_ph_max=int(n_ph_max), boson_encoding=str(boson_encoding), ordering=str(ordering))
    active_problem = RestrictedClosedShellMolecularProblem(
        geometry_spec=str(base.geometry_spec),
        basis=str(base.basis),
        charge=0,
        multiplicity=1,
        reference="rhf",
        n_spatial_orbitals=2,
        n_alpha=1,
        n_beta=1,
        hf_energy=float(base.electronic_hf_energy_center),
        nuclear_repulsion_energy=0.0,
        one_body_integrals_mo=np.zeros((2, 2), dtype=float),
        two_body_integrals_mo=np.zeros((2, 2, 2, 2), dtype=float),
    )
    source_problem = RestrictedClosedShellMolecularProblem(
        geometry_spec=str(base.geometry_spec),
        basis=str(base.basis),
        charge=0,
        multiplicity=1,
        reference="rhf",
        n_spatial_orbitals=int(base.source_n_spatial_orbitals),
        n_alpha=1,
        n_beta=1,
        hf_energy=float(base.electronic_hf_energy_center),
        nuclear_repulsion_energy=0.0,
        one_body_integrals_mo=np.zeros((int(base.source_n_spatial_orbitals), int(base.source_n_spatial_orbitals)), dtype=float),
        two_body_integrals_mo=np.zeros(
            (
                int(base.source_n_spatial_orbitals),
                int(base.source_n_spatial_orbitals),
                int(base.source_n_spatial_orbitals),
                int(base.source_n_spatial_orbitals),
            ),
            dtype=float,
        ),
    )
    rebuilt = build_vibronic_h2o_model_from_active_problem(
        active_problem,
        source_problem=source_problem,
        selected_spatial_orbital_indices=base.selected_spatial_orbital_indices,
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        coupling_scale=float(coupling_scale),
        omega_au=float(base.omega_au),
        reduced_mass_au=float(base.reduced_mass_au),
        derivative_scale_au_per_bohr=float(base.derivative_scale_au_per_bohr),
        derivative_source=str(base.derivative_source),
        ordering=str(ordering),
    )
    return VibronicH2OModel(
        geometry_spec=rebuilt.geometry_spec,
        basis=rebuilt.basis,
        active_space_kind=rebuilt.active_space_kind,
        selected_spatial_orbital_indices=rebuilt.selected_spatial_orbital_indices,
        source_n_spatial_orbitals=rebuilt.source_n_spatial_orbitals,
        source_n_spin_orbitals=int(base.source_n_spin_orbitals),
        n_ph_max=rebuilt.n_ph_max,
        boson_encoding=rebuilt.boson_encoding,
        n_fermion_qubits=rebuilt.n_fermion_qubits,
        n_boson_qubits=rebuilt.n_boson_qubits,
        n_total_qubits=rebuilt.n_total_qubits,
        omega_au=rebuilt.omega_au,
        reduced_mass_au=rebuilt.reduced_mass_au,
        x_zpf_bohr=rebuilt.x_zpf_bohr,
        derivative_scale_au_per_bohr=rebuilt.derivative_scale_au_per_bohr,
        derivative_source=rebuilt.derivative_source,
        electronic_hf_energy_center=rebuilt.electronic_hf_energy_center,
        h_electronic=base.h_electronic,
        dH_dQ=base.dH_dQ,
        h_vibronic=_clean_real_polynomial(
            _lift_fermion_polynomial(base.h_electronic, boson_qubits=int(rebuilt.n_boson_qubits))
            + (float(rebuilt.omega_au) * _clean_real_polynomial(
                boson_operator(
                    "JW",
                    int(rebuilt.n_total_qubits),
                    phonon_qubit_indices_for_site(0, n_sites=1, qpb=int(rebuilt.n_boson_qubits), fermion_qubits=4),
                    which="n",
                    n_ph_max=int(n_ph_max),
                    encoding=str(boson_encoding),
                )
            ))
            + float(0.5 * rebuilt.omega_au)
            + (float(coupling_scale) * float(rebuilt.x_zpf_bohr))
            * (
                _lift_fermion_polynomial(base.dH_dQ, boson_qubits=int(rebuilt.n_boson_qubits))
                * _clean_real_polynomial(
                    boson_operator(
                        "JW",
                        int(rebuilt.n_total_qubits),
                        phonon_qubit_indices_for_site(0, n_sites=1, qpb=int(rebuilt.n_boson_qubits), fermion_qubits=4),
                        which="x",
                        n_ph_max=int(n_ph_max),
                        encoding=str(boson_encoding),
                    )
                )
            )
        ),
        pool=rebuilt.pool,
        psi_ref=rebuilt.psi_ref,
        coupling_scale=float(coupling_scale),
    )


def build_cached_vibronic_h2o_model(
    *,
    n_ph_max: int = 1,
    boson_encoding: str = "binary",
    coupling_scale: float = 1.0,
    ordering: str = "blocked",
    fixture_path: str | Path | None = None,
) -> VibronicH2OModel:
    fixture = load_cached_vibronic_h2o_fixture(fixture_path)
    base = fixture.model
    if (
        int(n_ph_max) == int(base.n_ph_max)
        and str(boson_encoding) == str(base.boson_encoding)
        and math.isclose(float(coupling_scale), float(base.coupling_scale), rel_tol=0.0, abs_tol=1e-15)
        and str(ordering).strip().lower() == "blocked"
    ):
        return base
    return _rebuild_from_cached_components(
        base,
        n_ph_max=int(n_ph_max),
        boson_encoding=str(boson_encoding),
        coupling_scale=float(coupling_scale),
        ordering=str(ordering),
    )
