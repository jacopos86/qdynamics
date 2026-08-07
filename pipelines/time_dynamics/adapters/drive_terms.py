from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.quantum.drives_time_potential import (
    DensityDriveTemplate,
    default_spatial_weights,
    gaussian_sinusoid_waveform,
    reference_method_name,
)
from src.quantum.hubbard_latex_python_pairs import (
    boson_qubits_per_site,
    build_hubbard_holstein_drive,
)
from src.quantum.operator_pools.boson_chains import (
    make_boson_chain_observables,
    make_harmonic_kerr_chain_drive_operator,
)
from src.quantum.operator_pools.spin_boson import (
    build_spin_boson_hamiltonian,
    make_spin_boson_observables,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm

from pipelines.time_dynamics.adapters.hamiltonian import (
    adapter_for_resolved_problem,
)


def _poly_coeff_map(poly: Any, *, tol: float = 1.0e-12) -> dict[str, complex]:
    coeff_map: dict[str, complex] = {}
    for term in tuple(poly.return_polynomial()):
        label = str(term.pw2strng())
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        coeff_map[label] = coeff_map.get(label, 0.0 + 0.0j) + coeff
    return {
        str(label): complex(coeff)
        for label, coeff in coeff_map.items()
        if abs(complex(coeff)) > float(tol)
    }


def _polynomials_match(lhs: Any, rhs: Any, *, tol: float = 1.0e-10) -> bool:
    lhs_map = _poly_coeff_map(lhs, tol=float(tol))
    rhs_map = _poly_coeff_map(rhs, tol=float(tol))
    labels = set(lhs_map) | set(rhs_map)
    return all(abs(lhs_map.get(label, 0.0 + 0.0j) - rhs_map.get(label, 0.0 + 0.0j)) <= float(tol) for label in labels)


def _infer_polynomial_num_qubits(poly: Any, *, context: str) -> int:
    labels = [str(term.pw2strng()) for term in tuple(poly.return_polynomial())]
    if not labels:
        raise ValueError(f"{context} produced an empty Pauli polynomial.")
    lengths = {len(label) for label in labels}
    if len(lengths) != 1:
        raise ValueError(f"{context} produced mixed Pauli label lengths: {sorted(lengths)}.")
    return int(next(iter(lengths)))


def _resolved_total_qubits(
    *,
    resolved_problem: Any,
    request: Any,
    family_key: str,
) -> int:
    layout = getattr(resolved_problem, "layout", None)
    total_qubits = getattr(layout, "total_qubits", None)
    if total_qubits not in {None, ""}:
        return int(total_qubits)
    hamiltonian = getattr(resolved_problem, "hamiltonian", None)
    if hamiltonian is not None:
        return _infer_polynomial_num_qubits(
            hamiltonian,
            context=f"{family_key} resolved hamiltonian",
        )
    n_sites = int(getattr(request, "num_sites", 0))
    if n_sites <= 0:
        raise ValueError(f"Driven {family_key} neutral realtime requires positive num_sites.")
    if str(family_key).strip().lower() == "hh":
        qpb = boson_qubits_per_site(
            int(getattr(request, "n_ph_max")),
            str(getattr(request, "boson_encoding", "binary")),
        )
        return int(2 * n_sites + n_sites * int(qpb))
    return int(2 * n_sites)


def _lift_fermion_polynomial_to_total_qubits(
    poly: Any,
    *,
    total_qubits: int,
    fermion_qubits: int,
    context: str,
    tol: float = 1.0e-12,
) -> PauliPolynomial:
    """Lift a fermion-register polynomial to a boson-prefixed total register."""

    fermion_nq = _infer_polynomial_num_qubits(poly, context=str(context))
    if int(fermion_nq) == int(total_qubits):
        return _with_optional_global_identity(poly, include_identity=True, context=str(context))
    if int(fermion_nq) != int(fermion_qubits):
        raise ValueError(
            f"{context} has {fermion_nq} qubits; expected either total={total_qubits} "
            f"or fermion={fermion_qubits}."
        )
    boson_qubits = int(total_qubits) - int(fermion_qubits)
    if boson_qubits < 0:
        raise ValueError(f"{context} has inconsistent total/fermion qubits.")
    coeff_map: dict[str, complex] = {}
    for label, coeff in _poly_coeff_map(poly, tol=float(tol)).items():
        lifted_label = ("e" * int(boson_qubits)) + str(label)
        coeff_map[lifted_label] = coeff_map.get(lifted_label, 0.0 + 0.0j) + complex(coeff)
    return _coeff_map_to_polynomial(
        coeff_map,
        nq=int(total_qubits),
        context=str(context),
        tol=float(tol),
    )


"Built Math: strip_identity(P) removes only the global scalar identity term, preserving all physical non-identity components."
def _with_optional_global_identity(
    poly: Any,
    *,
    include_identity: bool,
    context: str,
) -> PauliPolynomial:
    nq = _infer_polynomial_num_qubits(poly, context=str(context))
    coeff_map = _poly_coeff_map(poly)
    if not bool(include_identity):
        coeff_map.pop("e" * int(nq), None)
    return _coeff_map_to_polynomial(
        coeff_map,
        nq=int(nq),
        context=str(context),
    )


def count_nonidentity_pauli_terms(poly: Any, *, tol: float = 1.0e-12) -> int:
    coeff_map = _poly_coeff_map(poly, tol=float(tol))
    return int(
        sum(
            1
            for label in coeff_map
            if any(str(ch).lower() != "e" for ch in str(label))
        )
    )


"Built Math: P(c) := Σ_label c_label P_label, with real Hermitian coefficients only."
def _coeff_map_to_polynomial(
    coeff_map: Mapping[str, complex | float | int],
    *,
    nq: int,
    context: str,
    tol: float = 1.0e-12,
) -> PauliPolynomial:
    poly = PauliPolynomial("JW")
    nq_i = int(nq)
    for label in sorted(str(k) for k in coeff_map.keys()):
        coeff = complex(coeff_map[label])
        if abs(coeff) <= float(tol):
            continue
        if len(str(label)) != nq_i:
            raise ValueError(
                f"{context} produced Pauli label {label!r} with length {len(str(label))}; expected {nq_i}."
            )
        if abs(float(coeff.imag)) > float(tol):
            raise ValueError(f"{context} produced non-Hermitian coefficient for {label}: {coeff}.")
        poly.add_term(PauliTerm(nq_i, ps=str(label), pc=float(coeff.real)))
    poly._reduce()
    return poly


def _z_label(*, nq_total: int, qubit_index: int) -> str:
    nq = int(nq_total)
    q = int(qubit_index)
    if nq <= 0:
        raise ValueError("nq_total must be positive")
    if q < 0 or q >= nq:
        raise ValueError(f"qubit_index {q} out of range [0, {nq})")
    z_pos = nq - 1 - q
    return ("e" * z_pos) + "z" + ("e" * (nq - 1 - z_pos))


"Built Math: D_spinful := Σ_i s_i Σ_σ n_{iσ}; n_{iσ}=(I-Z_{q(i,σ)})/2."
def _spinful_density_drive_polynomial(
    *,
    n_sites: int,
    nq_total: int,
    ordering: str,
    spatial_weights: tuple[float, ...],
    include_identity: bool,
) -> PauliPolynomial:
    template = DensityDriveTemplate.build(
        n_sites=int(n_sites),
        nq_total=int(nq_total),
        indexing=str(ordering),
        electron_qubit_offset=0,
    )
    coeff_map: dict[str, float] = {}
    for site, site_weight in enumerate(tuple(float(x) for x in spatial_weights)):
        coeff_z = -0.5 * float(site_weight)
        for spin in (0, 1):
            label = str(template.z_label(int(site), int(spin)))
            coeff_map[label] = float(coeff_map.get(label, 0.0) + coeff_z)
    if bool(include_identity):
        coeff_map[str(template.identity_label)] = float(sum(float(x) for x in spatial_weights))
    return _coeff_map_to_polynomial(
        coeff_map,
        nq=int(nq_total),
        context="spinful density drive",
    )


"Built Math: D_spinless := Σ_i s_i n_i; n_i=(I-Z_i)/2."
def _spinless_density_drive_polynomial(
    *,
    n_sites: int,
    spatial_weights: tuple[float, ...],
    include_identity: bool,
) -> PauliPolynomial:
    nq_total = int(n_sites)
    if len(tuple(spatial_weights)) != nq_total:
        raise ValueError(
            f"spinless density drive expected {nq_total} spatial weights; got {len(tuple(spatial_weights))}."
        )
    coeff_map: dict[str, float] = {}
    for site, site_weight in enumerate(tuple(float(x) for x in spatial_weights)):
        coeff_z = -0.5 * float(site_weight)
        label = _z_label(nq_total=int(nq_total), qubit_index=int(site))
        coeff_map[label] = float(coeff_map.get(label, 0.0) + coeff_z)
    if bool(include_identity):
        coeff_map["e" * int(nq_total)] = 0.5 * float(sum(float(x) for x in spatial_weights))
    return _coeff_map_to_polynomial(
        coeff_map,
        nq=int(nq_total),
        context="spinless density drive",
    )


@dataclass(frozen=True)
class RealtimeDriveModel:
    family_key: str
    operator_label: str
    drive_poly: Any
    spatial_weights: tuple[float, ...]
    profile_payload: dict[str, Any]
    drive_term_count: int
    supports_mode_off: bool
    supports_benchmark_exact: bool
    supports_exact_v1: bool
    drive_A: float
    drive_omega: float
    drive_tbar: float
    drive_phi: float

    "Built Math: H(t) = H_static + f(t) D, with D carrying spatial weights."
    def coefficient_at(self, physical_time: float) -> float:
        return gaussian_sinusoid_waveform(
            float(physical_time),
            A=float(self.drive_A),
            omega=float(self.drive_omega),
            tbar=float(self.drive_tbar),
            phi=float(self.drive_phi),
        )


def _custom_weights_or_none(drive_config: Any) -> list[float] | None:
    custom_weights = getattr(drive_config, "drive_custom_weights", None)
    return None if custom_weights is None else [float(x) for x in tuple(custom_weights)]


"Built Math: profile(D,f) records the scalar waveform and static weighted operator identity."
def _drive_profile_payload(
    *,
    drive_config: Any,
    family_key: str,
    operator_label: str,
    spatial_weights: tuple[float, ...],
    include_identity: bool,
) -> dict[str, Any]:
    custom_weights = _custom_weights_or_none(drive_config)
    return {
        "A": float(drive_config.drive_A),
        "omega": float(drive_config.drive_omega),
        "tbar": float(drive_config.drive_tbar),
        "phi": float(drive_config.drive_phi),
        "pattern": str(drive_config.drive_pattern),
        "custom_weights": custom_weights,
        "include_identity": bool(include_identity),
        "time_sampling": str(drive_config.drive_time_sampling),
        "t0": float(drive_config.drive_t0),
        "family_key": str(family_key),
        "operator_label": str(operator_label),
        "spatial_weights": [float(x) for x in tuple(spatial_weights)],
        "reference_method_name": reference_method_name(
            str(drive_config.drive_time_sampling)
        ),
    }


"Built Math: H_drive^sb(t) := f(t) * s_0 * Z_imbalance, where Z_imbalance is the validated static dv operator."
def _resolve_spin_boson_drive_model(
    *,
    resolved_problem: Any,
    drive_config: Any,
) -> RealtimeDriveModel:
    request = getattr(resolved_problem, "request", None)
    if request is None:
        raise ValueError("spin_boson driven neutral realtime requires a resolved_problem request.")
    if int(getattr(request, "num_sites", 0)) != 1:
        raise ValueError(
            "Driven spin_boson neutral realtime currently requires num_sites == 1."
        )
    if bool(getattr(drive_config, "drive_include_identity", False)):
        raise ValueError(
            "Driven spin_boson neutral realtime currently does not support --drive-include-identity."
        )

    observables = make_spin_boson_observables(
        n_ph_max=int(request.n_ph_max),
        boson_encoding=str(request.boson_encoding),
        ordering=str(request.ordering),
    )
    imbalance = observables["imbalance"]
    h_dv0 = build_spin_boson_hamiltonian(
        num_sites=int(request.num_sites),
        t=float(request.t),
        u=float(request.u),
        dv=0.0,
        omega0=float(request.omega0),
        g_ep=float(request.g_ep),
        n_ph_max=int(request.n_ph_max),
        boson_encoding=str(request.boson_encoding),
        ordering=str(request.ordering),
        include_zero_point=bool(request.include_zero_point),
    )
    h_dv1 = build_spin_boson_hamiltonian(
        num_sites=int(request.num_sites),
        t=float(request.t),
        u=float(request.u),
        dv=1.0,
        omega0=float(request.omega0),
        g_ep=float(request.g_ep),
        n_ph_max=int(request.n_ph_max),
        boson_encoding=str(request.boson_encoding),
        ordering=str(request.ordering),
        include_zero_point=bool(request.include_zero_point),
    )
    static_dv_operator = h_dv1 + ((-1.0) * h_dv0)
    static_dv_operator._reduce()
    if not _polynomials_match(imbalance, static_dv_operator):
        raise ValueError(
            "spin_boson driven neutral realtime expected the imbalance observable to match the static Hamiltonian dv operator."
        )

    custom_weights = _custom_weights_or_none(drive_config)
    weights = default_spatial_weights(
        int(getattr(drive_config, "n_sites", 1)),
        mode=str(getattr(drive_config, "drive_pattern", "staggered")),
        custom=custom_weights,
    )
    if int(weights.size) != 1:
        raise ValueError(
            "Driven spin_boson neutral realtime currently requires exactly one spatial weight."
        )
    spatial_weights = tuple(float(x) for x in weights.tolist())
    drive_poly = float(spatial_weights[0]) * imbalance
    drive_poly._reduce()

    return RealtimeDriveModel(
        family_key="spin_boson",
        operator_label="imbalance",
        drive_poly=drive_poly,
        spatial_weights=spatial_weights,
        profile_payload=_drive_profile_payload(
            drive_config=drive_config,
            family_key="spin_boson",
            operator_label="imbalance",
            spatial_weights=spatial_weights,
            include_identity=False,
        ),
        drive_term_count=count_nonidentity_pauli_terms(drive_poly),
        supports_mode_off=True,
        supports_benchmark_exact=True,
        supports_exact_v1=True,
        drive_A=float(drive_config.drive_A),
        drive_omega=float(drive_config.drive_omega),
        drive_tbar=float(drive_config.drive_tbar),
        drive_phi=float(drive_config.drive_phi),
    )


def _validated_request_site_count(
    *,
    family_key: str,
    request: Any,
    drive_config: Any,
) -> int:
    n_sites = int(getattr(request, "num_sites", getattr(drive_config, "n_sites", 0)))
    if n_sites <= 0:
        raise ValueError(f"Driven {family_key} neutral realtime requires a positive num_sites.")
    drive_sites = int(getattr(drive_config, "n_sites", n_sites))
    if drive_sites != n_sites:
        raise ValueError(
            f"Driven {family_key} neutral realtime requires drive n_sites={drive_sites} to match problem num_sites={n_sites}."
        )
    return int(n_sites)


"Built Math: H_drive^lat(t) := f(t) * Σ_i s_i(n_{i↑}+n_{i↓}) for spinful lattice families."
def _resolve_spinful_lattice_density_drive_model(
    *,
    family_key: str,
    resolved_problem: Any,
    drive_config: Any,
) -> RealtimeDriveModel:
    request = getattr(resolved_problem, "request", None)
    if request is None:
        raise ValueError(f"Driven {family_key} neutral realtime requires a resolved_problem request.")
    n_sites = _validated_request_site_count(
        family_key=str(family_key),
        request=request,
        drive_config=drive_config,
    )
    nq_total = 2 * int(n_sites)
    request_ordering = str(getattr(request, "ordering", getattr(drive_config, "ordering", "blocked")))
    drive_ordering = str(getattr(drive_config, "ordering", request_ordering))
    if drive_ordering.strip().lower() != request_ordering.strip().lower():
        raise ValueError(
            f"Driven {family_key} neutral realtime requires drive ordering={drive_ordering!r} "
            f"to match problem ordering={request_ordering!r}."
        )
    custom_weights = _custom_weights_or_none(drive_config)
    weights_arr = default_spatial_weights(
        int(n_sites),
        mode=str(getattr(drive_config, "drive_pattern", "staggered")),
        custom=custom_weights,
    )
    spatial_weights = tuple(float(x) for x in weights_arr.tolist())
    drive_poly = _spinful_density_drive_polynomial(
        n_sites=int(n_sites),
        nq_total=int(nq_total),
        ordering=str(request_ordering),
        spatial_weights=spatial_weights,
        include_identity=bool(getattr(drive_config, "drive_include_identity", False)),
    )
    return RealtimeDriveModel(
        family_key=str(family_key),
        operator_label="spinful_onsite_density",
        drive_poly=drive_poly,
        spatial_weights=spatial_weights,
        profile_payload=_drive_profile_payload(
            drive_config=drive_config,
            family_key=str(family_key),
            operator_label="spinful_onsite_density",
            spatial_weights=spatial_weights,
            include_identity=bool(getattr(drive_config, "drive_include_identity", False)),
        ),
        drive_term_count=count_nonidentity_pauli_terms(drive_poly),
        supports_mode_off=False,
        supports_benchmark_exact=True,
        supports_exact_v1=True,
        drive_A=float(drive_config.drive_A),
        drive_omega=float(drive_config.drive_omega),
        drive_tbar=float(drive_config.drive_tbar),
        drive_phi=float(drive_config.drive_phi),
    )


"Built Math: H_drive^HH(t) := f(t) * Σ_i s_i Σ_σ n_{iσ}, lifted over phonon qubits."
def _resolve_hh_density_drive_model(
    *,
    family_key: str,
    resolved_problem: Any,
    drive_config: Any,
) -> RealtimeDriveModel:
    request = getattr(resolved_problem, "request", None)
    if request is None:
        raise ValueError(f"Driven {family_key} neutral realtime requires a resolved_problem request.")
    n_sites = _validated_request_site_count(
        family_key=str(family_key),
        request=request,
        drive_config=drive_config,
    )
    request_ordering = str(getattr(request, "ordering", getattr(drive_config, "ordering", "blocked")))
    drive_ordering = str(getattr(drive_config, "ordering", request_ordering))
    if drive_ordering.strip().lower() != request_ordering.strip().lower():
        raise ValueError(
            f"Driven {family_key} neutral realtime requires drive ordering={drive_ordering!r} "
            f"to match problem ordering={request_ordering!r}."
        )
    total_qubits = _resolved_total_qubits(
        resolved_problem=resolved_problem,
        request=request,
        family_key=str(family_key),
    )
    custom_weights = _custom_weights_or_none(drive_config)
    weights_arr = default_spatial_weights(
        int(n_sites),
        mode=str(getattr(drive_config, "drive_pattern", "staggered")),
        custom=custom_weights,
    )
    spatial_weights = tuple(float(x) for x in weights_arr.tolist())
    drive_poly = build_hubbard_holstein_drive(
        dims=int(n_sites),
        v_t=spatial_weights,
        v0=tuple(0.0 for _ in range(int(n_sites))),
        repr_mode="JW",
        indexing=str(request_ordering),
        nq_override=int(total_qubits),
    )
    drive_poly = _with_optional_global_identity(
        drive_poly,
        include_identity=bool(getattr(drive_config, "drive_include_identity", False)),
        context="Hubbard-Holstein density drive",
    )
    return RealtimeDriveModel(
        family_key=str(family_key),
        operator_label="hh_spinful_onsite_density",
        drive_poly=drive_poly,
        spatial_weights=spatial_weights,
        profile_payload=_drive_profile_payload(
            drive_config=drive_config,
            family_key=str(family_key),
            operator_label="hh_spinful_onsite_density",
            spatial_weights=spatial_weights,
            include_identity=bool(getattr(drive_config, "drive_include_identity", False)),
        ),
        drive_term_count=count_nonidentity_pauli_terms(drive_poly),
        supports_mode_off=True,
        supports_benchmark_exact=True,
        supports_exact_v1=True,
        drive_A=float(drive_config.drive_A),
        drive_omega=float(drive_config.drive_omega),
        drive_tbar=float(drive_config.drive_tbar),
        drive_phi=float(drive_config.drive_phi),
    )


"Built Math: H_drive^spinless(t) := f(t) * Σ_i s_i n_i for spinless lattice families."
def _resolve_spinless_lattice_density_drive_model(
    *,
    family_key: str,
    resolved_problem: Any,
    drive_config: Any,
) -> RealtimeDriveModel:
    request = getattr(resolved_problem, "request", None)
    if request is None:
        raise ValueError(f"Driven {family_key} neutral realtime requires a resolved_problem request.")
    n_sites = _validated_request_site_count(
        family_key=str(family_key),
        request=request,
        drive_config=drive_config,
    )
    custom_weights = _custom_weights_or_none(drive_config)
    weights_arr = default_spatial_weights(
        int(n_sites),
        mode=str(getattr(drive_config, "drive_pattern", "staggered")),
        custom=custom_weights,
    )
    spatial_weights = tuple(float(x) for x in weights_arr.tolist())
    drive_poly = _spinless_density_drive_polynomial(
        n_sites=int(n_sites),
        spatial_weights=spatial_weights,
        include_identity=bool(getattr(drive_config, "drive_include_identity", False)),
    )
    return RealtimeDriveModel(
        family_key=str(family_key),
        operator_label="spinless_onsite_density",
        drive_poly=drive_poly,
        spatial_weights=spatial_weights,
        profile_payload=_drive_profile_payload(
            drive_config=drive_config,
            family_key=str(family_key),
            operator_label="spinless_onsite_density",
            spatial_weights=spatial_weights,
            include_identity=bool(getattr(drive_config, "drive_include_identity", False)),
        ),
        drive_term_count=count_nonidentity_pauli_terms(drive_poly),
        supports_mode_off=False,
        supports_benchmark_exact=True,
        supports_exact_v1=True,
        drive_A=float(drive_config.drive_A),
        drive_omega=float(drive_config.drive_omega),
        drive_tbar=float(drive_config.drive_tbar),
        drive_phi=float(drive_config.drive_phi),
    )


"Built Math: H_drive^HKC(t) := f(t) * Σ_i s_i X_i for the harmonic-Kerr displacement drive."
def _resolve_harmonic_kerr_chain_displacement_drive_model(
    *,
    family_key: str,
    resolved_problem: Any,
    drive_config: Any,
) -> RealtimeDriveModel:
    request = getattr(resolved_problem, "request", None)
    if request is None:
        raise ValueError(f"Driven {family_key} neutral realtime requires a resolved_problem request.")
    n_sites = _validated_request_site_count(
        family_key=str(family_key),
        request=request,
        drive_config=drive_config,
    )
    custom_weights = _custom_weights_or_none(drive_config)
    weights_arr = default_spatial_weights(
        int(n_sites),
        mode=str(getattr(drive_config, "drive_pattern", "staggered")),
        custom=custom_weights,
    )
    spatial_weights = tuple(float(x) for x in weights_arr.tolist())
    drive_poly = make_harmonic_kerr_chain_drive_operator(
        num_sites=int(n_sites),
        n_ph_max=int(request.n_ph_max),
        boson_encoding=str(request.boson_encoding),
        spatial_weights=spatial_weights,
    )
    drive_poly = _with_optional_global_identity(
        drive_poly,
        include_identity=bool(getattr(drive_config, "drive_include_identity", False)),
        context="harmonic Kerr chain displacement drive",
    )
    return RealtimeDriveModel(
        family_key=str(family_key),
        operator_label="harmonic_kerr_displacement",
        drive_poly=drive_poly,
        spatial_weights=spatial_weights,
        profile_payload=_drive_profile_payload(
            drive_config=drive_config,
            family_key=str(family_key),
            operator_label="harmonic_kerr_displacement",
            spatial_weights=spatial_weights,
            include_identity=bool(getattr(drive_config, "drive_include_identity", False)),
        ),
        drive_term_count=count_nonidentity_pauli_terms(drive_poly),
        supports_mode_off=False,
        supports_benchmark_exact=True,
        supports_exact_v1=True,
        drive_A=float(drive_config.drive_A),
        drive_omega=float(drive_config.drive_omega),
        drive_tbar=float(drive_config.drive_tbar),
        drive_phi=float(drive_config.drive_phi),
    )


"Built Math: H_drive^boson-chain(t) := f(t) * Σ_i s_i n_i for number-driven boson-chain families."
def _resolve_boson_chain_number_drive_model(
    *,
    family_key: str,
    resolved_problem: Any,
    drive_config: Any,
) -> RealtimeDriveModel:
    request = getattr(resolved_problem, "request", None)
    if request is None:
        raise ValueError(f"Driven {family_key} neutral realtime requires a resolved_problem request.")
    n_sites = _validated_request_site_count(
        family_key=str(family_key),
        request=request,
        drive_config=drive_config,
    )
    observables = make_boson_chain_observables(
        num_sites=int(n_sites),
        n_ph_max=int(request.n_ph_max),
        boson_encoding=str(request.boson_encoding),
    )
    custom_weights = _custom_weights_or_none(drive_config)
    weights_arr = default_spatial_weights(
        int(n_sites),
        mode=str(getattr(drive_config, "drive_pattern", "staggered")),
        custom=custom_weights,
    )
    spatial_weights = tuple(float(x) for x in weights_arr.tolist())
    drive_poly = PauliPolynomial("JW")
    for site, site_weight in enumerate(spatial_weights):
        drive_poly = drive_poly + (float(site_weight) * observables[f"n_site_{site}"])
    drive_poly._reduce()
    drive_poly = _with_optional_global_identity(
        drive_poly,
        include_identity=bool(getattr(drive_config, "drive_include_identity", False)),
        context="boson chain number drive",
    )
    return RealtimeDriveModel(
        family_key=str(family_key),
        operator_label="boson_onsite_number",
        drive_poly=drive_poly,
        spatial_weights=spatial_weights,
        profile_payload=_drive_profile_payload(
            drive_config=drive_config,
            family_key=str(family_key),
            operator_label="boson_onsite_number",
            spatial_weights=spatial_weights,
            include_identity=bool(getattr(drive_config, "drive_include_identity", False)),
        ),
        drive_term_count=count_nonidentity_pauli_terms(drive_poly),
        supports_mode_off=False,
        supports_benchmark_exact=True,
        supports_exact_v1=True,
        drive_A=float(drive_config.drive_A),
        drive_omega=float(drive_config.drive_omega),
        drive_tbar=float(drive_config.drive_tbar),
        drive_phi=float(drive_config.drive_phi),
    )


"Built Math: H_drive^vib(t) := f(t) * dH/dR for the molecular vibronic H2 fixture."
def _resolve_molecular_vibronic_h2_drive_model(
    *,
    family_key: str,
    resolved_problem: Any,
    drive_config: Any,
) -> RealtimeDriveModel:
    runtime_data = getattr(resolved_problem, "runtime_data", None)
    model = runtime_data.get("vibronic_h2_model") if isinstance(runtime_data, Mapping) else None
    if model is None or not hasattr(model, "dH_dR"):
        raise ValueError("Driven molecular_vibronic_h2 realtime requires runtime_data['vibronic_h2_model'].")
    lifted_dhdr = _lift_fermion_polynomial_to_total_qubits(
        getattr(model, "dH_dR"),
        total_qubits=int(getattr(model, "n_total_qubits")),
        fermion_qubits=int(getattr(model, "n_fermion_qubits")),
        context="molecular vibronic H2 dH/dR drive",
    )
    drive_poly = _with_optional_global_identity(
        lifted_dhdr,
        include_identity=bool(getattr(drive_config, "drive_include_identity", False)),
        context="molecular vibronic H2 dH/dR drive",
    )
    spatial_weights = (1.0,)
    return RealtimeDriveModel(
        family_key=str(family_key),
        operator_label="molecular_vibronic_dhdr",
        drive_poly=drive_poly,
        spatial_weights=spatial_weights,
        profile_payload=_drive_profile_payload(
            drive_config=drive_config,
            family_key=str(family_key),
            operator_label="molecular_vibronic_dhdr",
            spatial_weights=spatial_weights,
            include_identity=bool(getattr(drive_config, "drive_include_identity", False)),
        ),
        drive_term_count=count_nonidentity_pauli_terms(drive_poly),
        supports_mode_off=False,
        supports_benchmark_exact=True,
        supports_exact_v1=True,
        drive_A=float(drive_config.drive_A),
        drive_omega=float(drive_config.drive_omega),
        drive_tbar=float(drive_config.drive_tbar),
        drive_phi=float(drive_config.drive_phi),
    )


"Built Math: resolve_drive_model(problem, drive) := family-specific driven operator seam for neutral realtime."
def resolve_realtime_drive_model(
    *,
    resolved_problem: Any,
    drive_config: Any,
) -> RealtimeDriveModel:
    adapter = adapter_for_resolved_problem(resolved_problem)
    family_key = str(adapter.family_key)
    drive_kind = adapter.drive_operator_kind
    if drive_kind == "spin_boson_imbalance":
        return _resolve_spin_boson_drive_model(
            resolved_problem=resolved_problem,
            drive_config=drive_config,
        )
    if drive_kind == "spinful_lattice_density":
        return _resolve_spinful_lattice_density_drive_model(
            family_key=family_key,
            resolved_problem=resolved_problem,
            drive_config=drive_config,
        )
    if drive_kind == "hh_density_legacy":
        return _resolve_hh_density_drive_model(
            family_key=family_key,
            resolved_problem=resolved_problem,
            drive_config=drive_config,
        )
    if drive_kind == "spinless_lattice_density":
        return _resolve_spinless_lattice_density_drive_model(
            family_key=family_key,
            resolved_problem=resolved_problem,
            drive_config=drive_config,
        )
    if drive_kind == "boson_chain_number":
        return _resolve_boson_chain_number_drive_model(
            family_key=family_key,
            resolved_problem=resolved_problem,
            drive_config=drive_config,
        )
    if drive_kind == "harmonic_kerr_chain_displacement":
        return _resolve_harmonic_kerr_chain_displacement_drive_model(
            family_key=family_key,
            resolved_problem=resolved_problem,
            drive_config=drive_config,
        )
    if drive_kind == "molecular_vibronic_h2_dhdr":
        return _resolve_molecular_vibronic_h2_drive_model(
            family_key=family_key,
            resolved_problem=resolved_problem,
            drive_config=drive_config,
        )
    raise ValueError(
        f"Driven neutral realtime currently has no drive-term seam for family {family_key!r}."
    )


__all__ = [
    "RealtimeDriveModel",
    "count_nonidentity_pauli_terms",
    "resolve_realtime_drive_model",
]
