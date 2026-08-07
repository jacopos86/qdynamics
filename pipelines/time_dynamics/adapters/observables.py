from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

from src.quantum.compiled_polynomial import (
    apply_compiled_polynomial,
    compile_polynomial_action,
)
from src.quantum.hubbard_latex_python_pairs import (
    SPIN_DN,
    SPIN_UP,
    boson_operator,
    jw_number_operator,
    mode_index,
    phonon_qubit_indices_for_site,
)
from src.quantum.operator_pools.boson_chains import make_boson_chain_observables
from src.quantum.operator_pools.spin_boson import make_spin_boson_observables
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm

from pipelines.time_dynamics.adapters.hamiltonian import (
    BOSON_CHAIN_FAMILIES,
    SPINFUL_LATTICE_FAMILIES,
    SPINLESS_LATTICE_FAMILIES,
    adapter_for_resolved_problem,
)

_BOSON_CHAIN_FAMILIES = BOSON_CHAIN_FAMILIES
_SPINFUL_LATTICE_OBSERVABLE_FAMILIES = SPINFUL_LATTICE_FAMILIES
_SPINLESS_LATTICE_OBSERVABLE_FAMILIES = SPINLESS_LATTICE_FAMILIES


@dataclass(frozen=True)
class ObservableMeasurementDefinition:
    """A measurement-compatible observable needed to rebuild a snapshot field."""

    name: str
    kind: str
    poly: Any


@dataclass(frozen=True)
class ObservableMeasurementBundle:
    """Family-generic observable definitions plus reconstruction labels."""

    observable_family: str
    definitions: tuple[ObservableMeasurementDefinition, ...]
    site_occupations_label: str | None
    site_occupations_component_labels: tuple[str, ...]
    up_component_names: tuple[str, ...]
    dn_component_names: tuple[str, ...]


def observable_family_key(resolved_problem: Any | None) -> str:
    if resolved_problem is None:
        return "hh"
    runtime_data = getattr(resolved_problem, "runtime_data", None)
    if isinstance(runtime_data, Mapping):
        family = runtime_data.get("trajectory_metric_family", None)
        if family not in {None, ""}:
            return str(family)
    family_key = getattr(resolved_problem, "family_key", None)
    return "hh" if family_key in {None, ""} else str(family_key)


def _raw_family_key(resolved_problem: Any | None) -> str:
    if resolved_problem is None:
        return "hh"
    family_key = getattr(resolved_problem, "family_key", None)
    if family_key not in {None, ""}:
        return str(family_key)
    family = observable_family_key(resolved_problem)
    if family in (_SPINFUL_LATTICE_OBSERVABLE_FAMILIES | _SPINLESS_LATTICE_OBSERVABLE_FAMILIES):
        raise ValueError(
            "Spinful/spinless observable measurements require resolved_problem.family_key; "
            f"runtime metric alias {family!r} is not sufficient."
        )
    return str(family)


def _spin_orbital_bit_index(site: int, spin: int, num_sites: int, ordering: str) -> int:
    ord_norm = str(ordering).strip().lower()
    if ord_norm == "blocked":
        return int(site) if int(spin) == 0 else int(num_sites) + int(site)
    if ord_norm == "interleaved":
        return (2 * int(site)) + int(spin)
    raise ValueError(f"Unsupported ordering {ordering!r}")


def _resolved_num_qubits(
    resolved_problem: Any | None,
    *,
    num_qubits: int | None,
    default: int,
) -> int:
    if num_qubits is not None:
        return int(num_qubits)
    layout = None if resolved_problem is None else getattr(resolved_problem, "layout", None)
    total_qubits = None if layout is None else getattr(layout, "total_qubits", None)
    if total_qubits not in {None, ""}:
        return int(total_qubits)
    return int(default)


def _positive_num_sites(num_sites: int) -> int:
    n_sites = int(num_sites)
    if n_sites <= 0:
        raise ValueError(f"num_sites must be positive; got {num_sites!r}.")
    return int(n_sites)


def _validate_request_num_sites(
    resolved_problem: Any | None,
    *,
    num_sites: int,
    family: str,
) -> None:
    request = None if resolved_problem is None else getattr(resolved_problem, "request", None)
    request_num_sites = None if request is None else getattr(request, "num_sites", None)
    if request_num_sites in {None, ""}:
        return
    if int(request_num_sites) != int(num_sites):
        raise ValueError(
            f"Observable measurement num_sites mismatch for {family}: "
            f"argument={int(num_sites)} request={int(request_num_sites)}."
        )


def _reduce_poly(poly: Any) -> Any:
    try:
        poly._reduce()
    except Exception:
        pass
    return poly


def _estimate_mean(estimates: Mapping[str, Any], name: str) -> float:
    rec = estimates.get(str(name), None)
    value = rec.get("mean", None) if isinstance(rec, Mapping) else rec
    if value is None:
        raise KeyError(f"Missing observable estimate {name!r}.")
    value_f = float(value)
    if not np.isfinite(value_f):
        raise ValueError(f"Observable estimate {name!r} is not finite: {value!r}.")
    return float(value_f)


def _emitter_labels_from_component_labels(labels: Sequence[str]) -> list[str]:
    out: list[str] = []
    for raw in tuple(labels)[:2]:
        label = str(raw)
        out.append(label[2:] if label.startswith("n_") else label)
    while len(out) < 2:
        out.append("g" if len(out) == 0 else "e")
    return out


def _site_component_labels(num_sites: int) -> tuple[str, ...]:
    return tuple(f"n_{site}" for site in range(int(num_sites)))


def _spinful_measurement_bundle(
    *,
    family: str,
    resolved_problem: Any | None,
    num_sites: int,
    ordering: str,
    num_qubits: int | None,
) -> ObservableMeasurementBundle:
    n_sites = _positive_num_sites(num_sites)
    nq = _resolved_num_qubits(
        resolved_problem,
        num_qubits=num_qubits,
        default=2 * int(n_sites),
    )
    if int(2 * n_sites) > int(nq):
        raise ValueError(
            f"Spinful observable register unavailable: 2*L={2 * n_sites} > nq={nq}."
        )

    definitions: list[ObservableMeasurementDefinition] = []
    doublon_total = PauliPolynomial("JW", [])
    up_names: list[str] = []
    dn_names: list[str] = []
    for site in range(int(n_sites)):
        up_mode = mode_index(
            int(site),
            SPIN_UP,
            indexing=str(ordering),
            n_sites=int(n_sites),
        )
        dn_mode = mode_index(
            int(site),
            SPIN_DN,
            indexing=str(ordering),
            n_sites=int(n_sites),
        )
        n_up = _reduce_poly(jw_number_operator("JW", int(nq), int(up_mode)))
        n_dn = _reduce_poly(jw_number_operator("JW", int(nq), int(dn_mode)))
        doublon_total = doublon_total + (n_up * n_dn)
        up_name = f"n_up_site_{site}"
        dn_name = f"n_dn_site_{site}"
        up_names.append(up_name)
        dn_names.append(dn_name)
        definitions.append(
            ObservableMeasurementDefinition(
                name=up_name,
                kind="site_occupation_up",
                poly=n_up,
            )
        )
        definitions.append(
            ObservableMeasurementDefinition(
                name=dn_name,
                kind="site_occupation_dn",
                poly=n_dn,
            )
        )
    definitions.append(
        ObservableMeasurementDefinition(
            name="doublon",
            kind="doublon_total",
            poly=_reduce_poly(doublon_total),
        )
    )
    return ObservableMeasurementBundle(
        observable_family=str(family),
        definitions=tuple(definitions),
        site_occupations_label="fermion_site_occupations",
        site_occupations_component_labels=_site_component_labels(n_sites),
        up_component_names=tuple(up_names),
        dn_component_names=tuple(dn_names),
    )


def _spinless_measurement_bundle(
    *,
    family: str,
    resolved_problem: Any | None,
    num_sites: int,
    num_qubits: int | None,
) -> ObservableMeasurementBundle:
    n_sites = _positive_num_sites(num_sites)
    nq = _resolved_num_qubits(
        resolved_problem,
        num_qubits=num_qubits,
        default=int(n_sites),
    )
    if int(n_sites) > int(nq):
        raise ValueError(
            f"Spinless observable register unavailable: L={n_sites} > nq={nq}."
        )
    definitions = tuple(
        ObservableMeasurementDefinition(
            name=f"n_site_{site}",
            kind="site_occupation",
            poly=_reduce_poly(jw_number_operator("JW", int(nq), int(site))),
        )
        for site in range(int(n_sites))
    )
    return ObservableMeasurementBundle(
        observable_family=str(family),
        definitions=definitions,
        site_occupations_label="spinless_site_occupations",
        site_occupations_component_labels=_site_component_labels(n_sites),
        up_component_names=(),
        dn_component_names=(),
    )



def _vibronic_h2_model(resolved_problem: Any) -> Any:
    runtime_data = getattr(resolved_problem, "runtime_data", None)
    model = runtime_data.get("vibronic_h2_model") if isinstance(runtime_data, Mapping) else None
    if model is None:
        raise ValueError("molecular_vibronic_h2 observables require runtime_data['vibronic_h2_model'].")
    return model


def _vibronic_h2_number_poly(resolved_problem: Any) -> Any:
    model = _vibronic_h2_model(resolved_problem)
    qpb = int(getattr(model, "n_boson_qubits", 1))
    n_fermion = int(getattr(model, "n_fermion_qubits", 4))
    n_total = int(getattr(model, "n_total_qubits", n_fermion + qpb))
    block = phonon_qubit_indices_for_site(0, n_sites=1, qpb=qpb, fermion_qubits=n_fermion)
    poly = boson_operator(
        "JW",
        int(n_total),
        block,
        which="n",
        n_ph_max=int(getattr(model, "n_ph_max", 1)),
        encoding=str(getattr(model, "boson_encoding", "binary")),
    )
    poly._reduce()
    return poly


def _vibronic_h2_lifted_dhdr_poly(resolved_problem: Any) -> Any:
    model = _vibronic_h2_model(resolved_problem)
    raw = getattr(model, "dH_dR")
    terms = tuple(raw.return_polynomial())
    if not terms:
        raise ValueError("molecular_vibronic_h2 dH/dR observable is empty.")
    labels = {str(term.pw2strng()) for term in terms}
    lengths = {len(label) for label in labels}
    if len(lengths) != 1:
        raise ValueError("molecular_vibronic_h2 dH/dR observable has mixed Pauli label lengths.")
    raw_nq = int(next(iter(lengths)))
    total_nq = int(getattr(model, "n_total_qubits"))
    fermion_nq = int(getattr(model, "n_fermion_qubits"))
    if raw_nq == total_nq:
        return _reduce_poly(raw)
    if raw_nq != fermion_nq:
        raise ValueError(
            f"molecular_vibronic_h2 dH/dR observable has {raw_nq} qubits; "
            f"expected fermion={fermion_nq} or total={total_nq}."
        )
    boson_nq = int(total_nq) - int(fermion_nq)
    lifted = PauliPolynomial("JW")
    for term in terms:
        coeff = complex(term.p_coeff)
        if abs(coeff) <= 1.0e-12:
            continue
        lifted.add_term(PauliTerm(total_nq, ps=("e" * boson_nq) + str(term.pw2strng()), pc=coeff))
    return _reduce_poly(lifted)


def _vibronic_h2_measurement_bundle(*, resolved_problem: Any) -> ObservableMeasurementBundle:
    definitions = (
        ObservableMeasurementDefinition(
            name="vibron_number",
            kind="vibron_number",
            poly=_reduce_poly(_vibronic_h2_number_poly(resolved_problem)),
        ),
        ObservableMeasurementDefinition(
            name="vibronic_dhdr",
            kind="vibronic_dhdr",
            poly=_vibronic_h2_lifted_dhdr_poly(resolved_problem),
        ),
    )
    return ObservableMeasurementBundle(
        observable_family="molecular_vibronic_h2",
        definitions=definitions,
        site_occupations_label="vibronic_mode_occupation",
        site_occupations_component_labels=("n_vib",),
        up_component_names=(),
        dn_component_names=(),
    )

def _boson_chain_measurement_bundle(
    *,
    family: str,
    resolved_problem: Any,
) -> ObservableMeasurementBundle:
    request = getattr(resolved_problem, "request")
    n_sites = _positive_num_sites(int(request.num_sites))
    observables = make_boson_chain_observables(
        num_sites=int(n_sites),
        n_ph_max=int(request.n_ph_max),
        boson_encoding=str(request.boson_encoding),
    )
    definitions = tuple(
        ObservableMeasurementDefinition(
            name=f"n_site_{site}",
            kind="site_occupation",
            poly=_reduce_poly(observables[f"n_site_{site}"]),
        )
        for site in range(int(n_sites))
    )
    return ObservableMeasurementBundle(
        observable_family=str(family),
        definitions=definitions,
        site_occupations_label="boson_site_occupations",
        site_occupations_component_labels=_site_component_labels(n_sites),
        up_component_names=(),
        dn_component_names=(),
    )


def _spin_boson_measurement_bundle(
    *,
    resolved_problem: Any,
) -> ObservableMeasurementBundle:
    request = getattr(resolved_problem, "request")
    runtime_data = getattr(resolved_problem, "runtime_data", None)
    mode_labels = (
        list(runtime_data.get("emitter_mode_labels", ("g", "e")))
        if isinstance(runtime_data, Mapping)
        else ["g", "e"]
    )
    emitter_mode_labels = [str(x) for x in mode_labels[:2]]
    while len(emitter_mode_labels) < 2:
        emitter_mode_labels.append("g" if len(emitter_mode_labels) == 0 else "e")
    component_labels = tuple(
        [
            (label if label.startswith("n_") else f"n_{label}")
            for label in emitter_mode_labels[:2]
        ]
        + ["n_b"]
    )
    observables = make_spin_boson_observables(
        n_ph_max=int(request.n_ph_max),
        boson_encoding=str(request.boson_encoding),
        ordering=str(request.ordering),
    )
    definitions = tuple(
        ObservableMeasurementDefinition(
            name=name,
            kind=kind,
            poly=_reduce_poly(observables[name]),
        )
        for name, kind in (
            ("n_g", "emitter_ground_occupation"),
            ("n_e", "emitter_excited_occupation"),
            ("n_b", "boson_number"),
            ("spin_x", "spin_x"),
        )
    )
    return ObservableMeasurementBundle(
        observable_family="spin_boson",
        definitions=definitions,
        site_occupations_label="emitter_and_boson_occupations",
        site_occupations_component_labels=component_labels,
        up_component_names=(),
        dn_component_names=(),
    )


def observable_measurement_bundle_for_problem(
    *,
    resolved_problem: Any | None,
    num_sites: int,
    ordering: str = "blocked",
    num_qubits: int | None = None,
) -> ObservableMeasurementBundle:
    """Build family-generic observables sufficient to reconstruct snapshots.

    The returned Pauli-polynomial definitions are measurement-compatible: they
    are ordinary observables of the prepared circuit state.  No exact target or
    future state data is referenced here.
    """

    adapter = adapter_for_resolved_problem(resolved_problem, default_family="hh")
    family = _raw_family_key(resolved_problem)
    observable_kind = str(adapter.observable_kind)
    if observable_kind == "molecular_vibronic_h2":
        if resolved_problem is None:
            raise ValueError("molecular_vibronic_h2 observable measurements require a resolved_problem.")
        return _vibronic_h2_measurement_bundle(resolved_problem=resolved_problem)
    if observable_kind == "spin_boson":
        if resolved_problem is None:
            raise ValueError("spin_boson observable measurements require a resolved_problem.")
        _validate_request_num_sites(resolved_problem, num_sites=int(num_sites), family=str(family))
        return _spin_boson_measurement_bundle(resolved_problem=resolved_problem)
    if observable_kind == "boson_chain":
        if resolved_problem is None:
            raise ValueError("Boson-chain observable measurements require a resolved_problem.")
        _validate_request_num_sites(resolved_problem, num_sites=int(num_sites), family=str(family))
        return _boson_chain_measurement_bundle(
            family=str(family),
            resolved_problem=resolved_problem,
        )
    if observable_kind == "spinless_lattice":
        return _spinless_measurement_bundle(
            family=str(family),
            resolved_problem=resolved_problem,
            num_sites=int(num_sites),
            num_qubits=num_qubits,
        )
    if observable_kind in {"hh_spinful_boson", "spinful_lattice"}:
        return _spinful_measurement_bundle(
            family=str(family),
            resolved_problem=resolved_problem,
            num_sites=int(num_sites),
            ordering=str(ordering),
            num_qubits=num_qubits,
        )
    raise ValueError(f"Unsupported observable measurement family {family!r}.")


def measured_snapshot_from_estimates(
    bundle: ObservableMeasurementBundle,
    estimates: Mapping[str, Any],
    *,
    resolved_problem: Any | None,
    num_sites: int,
    requested_primary_density_mode: str,
) -> dict[str, Any]:
    """Reconstruct the observable snapshot represented by measured means."""

    family = str(bundle.observable_family)
    n_sites = _positive_num_sites(num_sites)
    if family == "spin_boson" or family in _BOSON_CHAIN_FAMILIES:
        _validate_request_num_sites(
            resolved_problem,
            num_sites=int(n_sites),
            family=str(family),
        )
    if family == "spin_boson":
        n_g = _estimate_mean(estimates, "n_g")
        n_e = _estimate_mean(estimates, "n_e")
        n_b = _estimate_mean(estimates, "n_b")
        spin_x = _estimate_mean(estimates, "spin_x")
        emitter_mode_labels = _emitter_labels_from_component_labels(
            bundle.site_occupations_component_labels
        )
        snapshot: dict[str, Any] = {
            "observable_family": "spin_boson",
            "n_up_site": [],
            "n_dn_site": [],
            "site_occupations": [float(n_g), float(n_e), float(n_b)],
            "site_occupations_label": bundle.site_occupations_label,
            "site_occupations_component_labels": [
                str(x) for x in bundle.site_occupations_component_labels
            ],
            "doublon": float("nan"),
            "staggered": float("nan"),
            "emitter_mode_labels": [str(x) for x in emitter_mode_labels],
            "emitter_ground_occupation": float(n_g),
            "emitter_excited_occupation": float(n_e),
            "boson_number": float(n_b),
            "emitter_imbalance": float(n_e - n_g),
            "spin_x": float(spin_x),
        }
        snapshot["primary_density"] = float(
            primary_density_value_from_snapshot(
                snapshot,
                resolved_problem=resolved_problem,
                num_sites=int(n_sites),
                requested_mode=str(requested_primary_density_mode),
            )
        )
        return snapshot

    if family in _SPINLESS_LATTICE_OBSERVABLE_FAMILIES:
        site_occupations = [
            _estimate_mean(estimates, f"n_site_{site}") for site in range(int(n_sites))
        ]
        staggered = float(_staggered_density(site_occupations))
        snapshot = {
            "observable_family": family,
            "n_up_site": [],
            "n_dn_site": [],
            "site_occupations": [float(x) for x in site_occupations],
            "site_occupations_label": bundle.site_occupations_label,
            "site_occupations_component_labels": [
                str(x) for x in bundle.site_occupations_component_labels
            ],
            "doublon": float("nan"),
            "staggered": float(staggered),
            "spinless_particle_number": float(np.sum(site_occupations)),
            "spinless_staggered_density": float(staggered),
        }
        snapshot["primary_density"] = float(
            primary_density_value_from_snapshot(
                snapshot,
                resolved_problem=resolved_problem,
                num_sites=int(n_sites),
                requested_mode=str(requested_primary_density_mode),
            )
        )
        return snapshot

    if family == "molecular_vibronic_h2":
        vibron_number = _estimate_mean(estimates, "vibron_number")
        vibronic_dhdr = _estimate_mean(estimates, "vibronic_dhdr")
        snapshot = {
            "observable_family": "molecular_vibronic_h2",
            "n_up_site": [],
            "n_dn_site": [],
            "site_occupations": [float(vibron_number)],
            "site_occupations_label": bundle.site_occupations_label,
            "site_occupations_component_labels": [
                str(x) for x in bundle.site_occupations_component_labels
            ],
            "doublon": float("nan"),
            "staggered": float("nan"),
            "vibron_number": float(vibron_number),
            "vibronic_dhdr": float(vibronic_dhdr),
        }
        snapshot["primary_density"] = float(
            primary_density_value_from_snapshot(
                snapshot,
                resolved_problem=resolved_problem,
                num_sites=int(n_sites),
                requested_mode=str(requested_primary_density_mode),
            )
        )
        return snapshot

    if family in _BOSON_CHAIN_FAMILIES:
        site_occupations = [
            _estimate_mean(estimates, f"n_site_{site}") for site in range(int(n_sites))
        ]
        snapshot = {
            "observable_family": family,
            "n_up_site": [],
            "n_dn_site": [],
            "site_occupations": [float(x) for x in site_occupations],
            "site_occupations_label": bundle.site_occupations_label,
            "site_occupations_component_labels": [
                str(x) for x in bundle.site_occupations_component_labels
            ],
            "doublon": float("nan"),
            "staggered": float(_staggered_density(site_occupations)),
            "boson_number_total": float(np.sum(site_occupations)),
            "site0_occupation": float(site_occupations[0]) if site_occupations else float("nan"),
        }
        snapshot["primary_density"] = float(
            primary_density_value_from_snapshot(
                snapshot,
                resolved_problem=resolved_problem,
                num_sites=int(n_sites),
                requested_mode=str(requested_primary_density_mode),
            )
        )
        return snapshot

    if family == "hh" or family in _SPINFUL_LATTICE_OBSERVABLE_FAMILIES:
        up_names = tuple(bundle.up_component_names) or tuple(
            f"n_up_site_{site}" for site in range(int(n_sites))
        )
        dn_names = tuple(bundle.dn_component_names) or tuple(
            f"n_dn_site_{site}" for site in range(int(n_sites))
        )
        if len(up_names) != int(n_sites) or len(dn_names) != int(n_sites):
            raise ValueError(
                "Spinful observable bundle component count does not match num_sites."
            )
        n_up = [_estimate_mean(estimates, name) for name in up_names]
        n_dn = [_estimate_mean(estimates, name) for name in dn_names]
        site_occupations = [float(u + d) for u, d in zip(n_up, n_dn)]
        snapshot = {
            "observable_family": family,
            "n_up_site": [float(x) for x in n_up],
            "n_dn_site": [float(x) for x in n_dn],
            "site_occupations": [float(x) for x in site_occupations],
            "site_occupations_up": [float(x) for x in n_up],
            "site_occupations_dn": [float(x) for x in n_dn],
            "site_occupations_label": bundle.site_occupations_label,
            "site_occupations_component_labels": [
                str(x) for x in bundle.site_occupations_component_labels
            ],
            "doublon": float(_estimate_mean(estimates, "doublon")),
            "staggered": float(_staggered_density(site_occupations)),
        }
        snapshot["primary_density"] = float(
            primary_density_value_from_snapshot(
                snapshot,
                resolved_problem=resolved_problem,
                num_sites=int(n_sites),
                requested_mode=str(requested_primary_density_mode),
            )
        )
        return snapshot

    raise ValueError(f"Unsupported observable measurement family {family!r}.")


def hh_observable_snapshot(
    psi: np.ndarray,
    *,
    num_sites: int,
    ordering: str,
) -> dict[str, Any]:
    probs = np.abs(np.asarray(psi, dtype=complex).reshape(-1)) ** 2
    n_up = np.zeros(int(num_sites), dtype=float)
    n_dn = np.zeros(int(num_sites), dtype=float)
    doublon_total = 0.0
    up_bits = [_spin_orbital_bit_index(site, 0, num_sites, ordering) for site in range(int(num_sites))]
    dn_bits = [_spin_orbital_bit_index(site, 1, num_sites, ordering) for site in range(int(num_sites))]

    for idx, prob in enumerate(probs):
        p = float(prob)
        if p <= 0.0:
            continue
        for site in range(int(num_sites)):
            up = int((idx >> up_bits[site]) & 1)
            dn = int((idx >> dn_bits[site]) & 1)
            n_up[site] += float(up) * p
            n_dn[site] += float(dn) * p
            doublon_total += float(up * dn) * p

    n_site = np.asarray(n_up + n_dn, dtype=float)
    if n_site.size == 0:
        staggered = float("nan")
    else:
        signs = np.array(
            [1.0 if (site % 2 == 0) else -1.0 for site in range(int(n_site.size))],
            dtype=float,
        )
        staggered = float(np.sum(signs * n_site) / float(n_site.size))
    return {
        "observable_family": "hh",
        "n_up_site": [float(x) for x in np.asarray(n_up, dtype=float).tolist()],
        "n_dn_site": [float(x) for x in np.asarray(n_dn, dtype=float).tolist()],
        "site_occupations": [float(x) for x in np.asarray(n_site, dtype=float).tolist()],
        "doublon": float(doublon_total),
        "staggered": float(staggered),
    }


def spinful_lattice_observable_snapshot(
    psi: np.ndarray,
    *,
    resolved_problem: Any,
    num_sites: int,
    ordering: str,
) -> dict[str, Any]:
    snapshot = hh_observable_snapshot(
        psi,
        num_sites=int(num_sites),
        ordering=str(ordering),
    )
    family_key = getattr(resolved_problem, "family_key", None)
    snapshot["observable_family"] = (
        str(family_key)
        if family_key not in {None, ""}
        else observable_family_key(resolved_problem)
    )
    snapshot["site_occupations_label"] = "fermion_site_occupations"
    snapshot["site_occupations_component_labels"] = [
        f"n_{site}" for site in range(int(num_sites))
    ]
    return snapshot


def _compiled_expectation(
    psi: np.ndarray,
    *,
    cache_key: str,
    poly: Any,
    compiled_poly_cache: MutableMapping[str, Any] | None,
    pauli_action_cache: MutableMapping[str, Any] | None,
) -> float:
    compiled = None if compiled_poly_cache is None else compiled_poly_cache.get(cache_key)
    if compiled is None:
        compiled = compile_polynomial_action(
            poly,
            tol=1.0e-12,
            pauli_action_cache=pauli_action_cache,
        )
        if compiled_poly_cache is not None:
            compiled_poly_cache[cache_key] = compiled
    op_psi = apply_compiled_polynomial(np.asarray(psi, dtype=complex).reshape(-1), compiled)
    return float(np.real(np.vdot(np.asarray(psi, dtype=complex).reshape(-1), op_psi)))


def spin_boson_observable_snapshot(
    psi: np.ndarray,
    *,
    resolved_problem: Any,
    compiled_poly_cache: MutableMapping[str, Any] | None = None,
    pauli_action_cache: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    request = getattr(resolved_problem, "request")
    runtime_data = getattr(resolved_problem, "runtime_data", None)
    mode_labels = (
        list(runtime_data.get("emitter_mode_labels", ("g", "e")))
        if isinstance(runtime_data, Mapping)
        else ["g", "e"]
    )
    observables = make_spin_boson_observables(
        n_ph_max=int(request.n_ph_max),
        boson_encoding=str(request.boson_encoding),
        ordering=str(request.ordering),
    )
    n_g = _compiled_expectation(
        psi,
        cache_key="spin_boson::n_g",
        poly=observables["n_g"],
        compiled_poly_cache=compiled_poly_cache,
        pauli_action_cache=pauli_action_cache,
    )
    n_e = _compiled_expectation(
        psi,
        cache_key="spin_boson::n_e",
        poly=observables["n_e"],
        compiled_poly_cache=compiled_poly_cache,
        pauli_action_cache=pauli_action_cache,
    )
    n_b = _compiled_expectation(
        psi,
        cache_key="spin_boson::n_b",
        poly=observables["n_b"],
        compiled_poly_cache=compiled_poly_cache,
        pauli_action_cache=pauli_action_cache,
    )
    imbalance = _compiled_expectation(
        psi,
        cache_key="spin_boson::imbalance",
        poly=observables["imbalance"],
        compiled_poly_cache=compiled_poly_cache,
        pauli_action_cache=pauli_action_cache,
    )
    spin_x = _compiled_expectation(
        psi,
        cache_key="spin_boson::spin_x",
        poly=observables["spin_x"],
        compiled_poly_cache=compiled_poly_cache,
        pauli_action_cache=pauli_action_cache,
    )
    emitter_mode_labels = [str(x) for x in mode_labels[:2]]
    while len(emitter_mode_labels) < 2:
        emitter_mode_labels.append("g" if len(emitter_mode_labels) == 0 else "e")
    component_labels = [
        (label if label.startswith("n_") else f"n_{label}")
        for label in emitter_mode_labels[:2]
    ] + ["n_b"]
    return {
        "observable_family": "spin_boson",
        "n_up_site": [],
        "n_dn_site": [],
        "site_occupations": [float(n_g), float(n_e), float(n_b)],
        "site_occupations_label": "emitter_and_boson_occupations",
        "site_occupations_component_labels": [str(x) for x in component_labels],
        "doublon": float("nan"),
        "staggered": float("nan"),
        "emitter_mode_labels": emitter_mode_labels,
        "emitter_ground_occupation": float(n_g),
        "emitter_excited_occupation": float(n_e),
        "boson_number": float(n_b),
        "emitter_imbalance": float(imbalance),
        "spin_x": float(spin_x),
    }


def _staggered_density(site_occupations: Sequence[float]) -> float:
    values = np.asarray(site_occupations, dtype=float).reshape(-1)
    if values.size == 0:
        return float("nan")
    signs = np.array(
        [1.0 if (site % 2 == 0) else -1.0 for site in range(int(values.size))],
        dtype=float,
    )
    return float(np.sum(signs * values) / float(values.size))


r"Built Math: n_i = \sum_b |\psi_b|^2 bit_i(b); s = L^{-1}\sum_i (-1)^i n_i; spinless t-V has no doublon channel."
def spinless_lattice_observable_snapshot(
    psi: np.ndarray,
    *,
    resolved_problem: Any,
    num_sites: int,
) -> dict[str, Any]:
    probs = np.abs(np.asarray(psi, dtype=complex).reshape(-1)) ** 2
    n_site = np.zeros(int(num_sites), dtype=float)
    for idx, prob in enumerate(probs):
        p = float(prob)
        if p <= 0.0:
            continue
        for site in range(int(num_sites)):
            n_site[int(site)] += float((int(idx) >> int(site)) & 1) * p
    site_occupations = [float(x) for x in np.asarray(n_site, dtype=float).tolist()]
    staggered = float(_staggered_density(site_occupations))
    family_key = getattr(resolved_problem, "family_key", None)
    return {
        "observable_family": (
            str(family_key)
            if family_key not in {None, ""}
            else observable_family_key(resolved_problem)
        ),
        "n_up_site": [],
        "n_dn_site": [],
        "site_occupations": site_occupations,
        "site_occupations_label": "spinless_site_occupations",
        "site_occupations_component_labels": [
            f"n_{site}" for site in range(int(num_sites))
        ],
        "doublon": float("nan"),
        "staggered": float(staggered),
        "spinless_particle_number": float(np.sum(n_site)),
        "spinless_staggered_density": float(staggered),
    }


def boson_chain_observable_snapshot(
    psi: np.ndarray,
    *,
    resolved_problem: Any,
    compiled_poly_cache: MutableMapping[str, Any] | None = None,
    pauli_action_cache: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    request = getattr(resolved_problem, "request")
    observables = make_boson_chain_observables(
        num_sites=int(request.num_sites),
        n_ph_max=int(request.n_ph_max),
        boson_encoding=str(request.boson_encoding),
    )
    site_occupations = [
        _compiled_expectation(
            psi,
            cache_key=f"boson_chain::{observable_family_key(resolved_problem)}::n_site_{site}",
            poly=observables[f"n_site_{site}"],
            compiled_poly_cache=compiled_poly_cache,
            pauli_action_cache=pauli_action_cache,
        )
        for site in range(int(request.num_sites))
    ]
    boson_number_total = _compiled_expectation(
        psi,
        cache_key=f"boson_chain::{observable_family_key(resolved_problem)}::n_total",
        poly=observables["n_total"],
        compiled_poly_cache=compiled_poly_cache,
        pauli_action_cache=pauli_action_cache,
    )
    return {
        "observable_family": observable_family_key(resolved_problem),
        "n_up_site": [],
        "n_dn_site": [],
        "site_occupations": [float(x) for x in site_occupations],
        "site_occupations_label": "boson_site_occupations",
        "site_occupations_component_labels": [
            f"n_{site}" for site in range(int(request.num_sites))
        ],
        "doublon": float("nan"),
        "staggered": float(_staggered_density(site_occupations)),
        "boson_number_total": float(boson_number_total),
        "site0_occupation": float(site_occupations[0]),
    }



def vibronic_h2_observable_snapshot(
    psi: np.ndarray,
    *,
    resolved_problem: Any,
    compiled_poly_cache: MutableMapping[str, Any] | None = None,
    pauli_action_cache: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    model = _vibronic_h2_model(resolved_problem)
    n_b = _compiled_expectation(
        psi,
        cache_key="molecular_vibronic_h2::vibron_number",
        poly=_vibronic_h2_number_poly(resolved_problem),
        compiled_poly_cache=compiled_poly_cache,
        pauli_action_cache=pauli_action_cache,
    )
    dhdr = _compiled_expectation(
        psi,
        cache_key="molecular_vibronic_h2::dH_dR",
        poly=_vibronic_h2_lifted_dhdr_poly(resolved_problem),
        compiled_poly_cache=compiled_poly_cache,
        pauli_action_cache=pauli_action_cache,
    )
    return {
        "observable_family": "molecular_vibronic_h2",
        "n_up_site": [],
        "n_dn_site": [],
        "site_occupations": [float(n_b)],
        "site_occupations_label": "vibronic_mode_occupation",
        "site_occupations_component_labels": ["n_vib"],
        "doublon": float("nan"),
        "staggered": float("nan"),
        "vibron_number": float(n_b),
        "vibronic_dhdr": float(dhdr),
        "primary_density": float(n_b),
    }

def observable_snapshot_for_state(
    psi: np.ndarray,
    *,
    resolved_problem: Any | None,
    num_sites: int,
    ordering: str,
    compiled_poly_cache: MutableMapping[str, Any] | None = None,
    pauli_action_cache: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    family = observable_family_key(resolved_problem)
    raw_family = getattr(resolved_problem, "family_key", None)
    raw_family_key = "" if raw_family in {None, ""} else str(raw_family)
    if family == "molecular_vibronic_h2":
        return vibronic_h2_observable_snapshot(
            psi,
            resolved_problem=resolved_problem,
            compiled_poly_cache=compiled_poly_cache,
            pauli_action_cache=pauli_action_cache,
        )
    if family == "spin_boson":
        return spin_boson_observable_snapshot(
            psi,
            resolved_problem=resolved_problem,
            compiled_poly_cache=compiled_poly_cache,
            pauli_action_cache=pauli_action_cache,
        )
    if raw_family_key in _SPINLESS_LATTICE_OBSERVABLE_FAMILIES:
        return spinless_lattice_observable_snapshot(
            psi,
            resolved_problem=resolved_problem,
            num_sites=int(num_sites),
        )
    if raw_family_key in _SPINFUL_LATTICE_OBSERVABLE_FAMILIES:
        return spinful_lattice_observable_snapshot(
            psi,
            resolved_problem=resolved_problem,
            num_sites=int(num_sites),
            ordering=str(ordering),
        )
    if family in _BOSON_CHAIN_FAMILIES:
        return boson_chain_observable_snapshot(
            psi,
            resolved_problem=resolved_problem,
            compiled_poly_cache=compiled_poly_cache,
            pauli_action_cache=pauli_action_cache,
        )
    return hh_observable_snapshot(
        psi,
        num_sites=int(num_sites),
        ordering=str(ordering),
    )


def auto_primary_density_mode(*, resolved_problem: Any | None, num_sites: int) -> str:
    family = observable_family_key(resolved_problem)
    raw_family = getattr(resolved_problem, "family_key", None)
    raw_family_key = "" if raw_family in {None, ""} else str(raw_family)
    if raw_family_key in _SPINLESS_LATTICE_OBSERVABLE_FAMILIES:
        return "staggered"
    if family == "molecular_vibronic_h2":
        return "vibron_number"
    if family == "spin_boson":
        return "imbalance"
    if family in _BOSON_CHAIN_FAMILIES:
        return "pair_difference" if int(max(1, num_sites)) == 2 else "staggered"
    return "pair_difference" if int(max(1, num_sites)) == 2 else "staggered"


def primary_density_value_from_snapshot(
    snapshot: Mapping[str, Any],
    *,
    resolved_problem: Any | None,
    num_sites: int,
    requested_mode: str,
) -> float:
    mode = str(requested_mode).strip().lower()
    if mode == "auto":
        mode = auto_primary_density_mode(
            resolved_problem=resolved_problem,
            num_sites=int(num_sites),
        )
    if mode == "pair_difference":
        site_occ = np.asarray(snapshot.get("site_occupations", ()), dtype=float).reshape(-1)
        if site_occ.size < 2:
            return float("nan")
        return float(site_occ[0] - site_occ[1])
    if mode == "imbalance":
        return float(snapshot.get("emitter_imbalance", float("nan")))
    if mode == "vibron_number":
        return float(snapshot.get("vibron_number", float("nan")))
    return float(snapshot.get("staggered", float("nan")))


def _copy_summary_fields(
    row: Mapping[str, Any],
    out: dict[str, Any],
    pairs: Sequence[tuple[str, str]],
) -> None:
    for src_key, dst_key in pairs:
        value = row.get(src_key, None)
        if value is None:
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            out[dst_key] = [str(x) for x in value]
        elif isinstance(value, (float, int, np.floating, np.integer)):
            out[dst_key] = float(value)
        else:
            out[dst_key] = value


def summary_fields_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    family = str(row.get("observable_family", "hh"))
    out: dict[str, Any] = {"final_observable_family": family}
    if family == "molecular_vibronic_h2":
        _copy_summary_fields(
            row,
            out,
            (
                ("vibron_number", "final_vibron_number"),
                ("vibronic_dhdr", "final_vibronic_dhdr"),
            ),
        )
        return out
    if family == "spin_boson":
        _copy_summary_fields(
            row,
            out,
            (
                ("emitter_mode_labels", "emitter_mode_labels"),
                ("emitter_ground_occupation", "final_emitter_ground_occupation"),
                ("emitter_excited_occupation", "final_emitter_excited_occupation"),
                ("boson_number", "final_boson_number"),
                ("emitter_imbalance", "final_emitter_imbalance"),
                ("spin_x", "final_spin_x"),
            ),
        )
        return out
    if family in _SPINLESS_LATTICE_OBSERVABLE_FAMILIES:
        _copy_summary_fields(
            row,
            out,
            (
                ("spinless_particle_number", "final_spinless_particle_number"),
                ("spinless_staggered_density", "final_spinless_staggered_density"),
            ),
        )
        return out
    if family in _BOSON_CHAIN_FAMILIES:
        _copy_summary_fields(
            row,
            out,
            (
                ("boson_number_total", "final_boson_number_total"),
                ("site0_occupation", "final_site0_occupation"),
            ),
        )
        return out
    return out


__all__ = [
    "ObservableMeasurementBundle",
    "ObservableMeasurementDefinition",
    "auto_primary_density_mode",
    "measured_snapshot_from_estimates",
    "observable_family_key",
    "observable_measurement_bundle_for_problem",
    "observable_snapshot_for_state",
    "primary_density_value_from_snapshot",
    "summary_fields_from_row",
]
