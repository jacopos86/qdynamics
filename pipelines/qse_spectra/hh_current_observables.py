#!/usr/bin/env python3
"""HH charge-current and Peierls-contact observables on the full HH register.

RECONSTRUCTION (2026-08-18): the original module was never committed and was
lost when snapshot commit 6442fbb5 captured its importers without it. This
implementation is reconstructed against the committed behavioral spec in
``test/test_qse_hh_current_observables.py`` and the CLI wiring in
``__main__.py``.

Physics and conventions
-----------------------
For the 1D Hubbard--Holstein chain with hopping ``t`` under the standard
charge Peierls substitution, the paramagnetic charge-current and the
diamagnetic (contact / kinetic) operators on a directed edge ``(i -> j)``
with adjacent Jordan--Wigner modes ``a < b`` per spin are

    J_edge,sigma = (t/2) * (X_b Y_a - Y_b X_a)
    K_edge,sigma = (t/2) * (X_b X_a + Y_b Y_a)

using the repo Pauli-string convention (qubit 0 is the rightmost label
character; blocked ordering puts the spin-up block first, so same-spin
neighboring sites occupy adjacent qubits and need no Z string). Phonon
qubits remain identity: the operators act on the full register so they can
be used directly as QSE transition observables. ``t == 0`` yields explicit
zero operators (a single identity term with coefficient zero) flagged in
metadata rather than silently empty observables. Periodic L=2 is rejected
as ambiguous (the chain edge and the wrap edge coincide); periodic wrap
edges for L>2 would require JW strings and fail closed in this
reconstruction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import json

from pipelines.qse_spectra.core import QSEObservable, polynomial_observable
from src.quantum.hubbard_latex_python_pairs import SPIN_DN, SPIN_UP, mode_index
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm

HH_CURRENT_OBSERVABLES_SCHEMA_VERSION = "hh_current_observables_v1"
HH_CURRENT_PEIERLS_POLICY = "standard_hh_1d_charge_peierls"
HH_CURRENT_CONTACT_POLICY = "peierls_second_derivative_record_only"
HH_CURRENT_EDGE_ORIENTATIONS = ("positive_chain",)

_HOPPING_MATCH_TOLERANCE = 1.0e-12


class HHCurrentObservableError(ValueError):
    """Raised when HH current-observable inputs are invalid or unsupported."""


def _pauli_label(nq: int, placements: Mapping[int, str]) -> str:
    chars = ["e"] * int(nq)
    for qubit, pauli in placements.items():
        index = int(nq) - 1 - int(qubit)
        if index < 0 or index >= int(nq):
            raise HHCurrentObservableError(f"qubit index {qubit} out of range for nq={nq}.")
        chars[index] = str(pauli)
    return "".join(chars)


def _zero_polynomial(nq: int) -> PauliPolynomial:
    # The constructor drops zero-coefficient terms; add_term retains them, and
    # the explicit zero identity term is the bundle's zero-operator contract.
    poly = PauliPolynomial("JW")
    poly.add_term(PauliTerm(int(nq), ps="e" * int(nq), pc=0.0))
    return poly


def directed_hh_current_edges(
    layout: Any,
    *,
    edge_orientation: str = "positive_chain",
) -> tuple[tuple[int, int], ...]:
    """Return the directed site edges for the requested orientation."""

    orientation = str(edge_orientation)
    if orientation not in HH_CURRENT_EDGE_ORIENTATIONS:
        raise HHCurrentObservableError(
            f"HH current edge orientation must be one of {list(HH_CURRENT_EDGE_ORIENTATIONS)!r}; "
            f"got {orientation!r}."
        )
    num_sites = int(layout.num_sites)
    boundary = str(layout.boundary)
    if num_sites < 2:
        raise HHCurrentObservableError("HH current observables require at least two sites.")
    edges = [(site, site + 1) for site in range(num_sites - 1)]
    if boundary == "periodic":
        if num_sites == 2:
            raise HHCurrentObservableError(
                "periodic L=2 HH current edges are ambiguous: the chain edge (0,1) and "
                "the wrap edge (1,0) coincide; use an open boundary or L>2."
            )
        edges.append((num_sites - 1, 0))
    return tuple(edges)


def _edge_pair_terms(
    layout: Any,
    *,
    edge: tuple[int, int],
    spin: int,
    hopping_amplitude: float,
    kind: str,
) -> list[PauliTerm]:
    nq = int(layout.total_qubits)
    num_sites = int(layout.num_sites)
    mode_from = int(
        mode_index(int(edge[0]), spin, indexing=str(layout.ordering), n_sites=num_sites)
    )
    mode_to = int(mode_index(int(edge[1]), spin, indexing=str(layout.ordering), n_sites=num_sites))
    if abs(mode_to - mode_from) != 1:
        raise HHCurrentObservableError(
            f"HH current edge {tuple(edge)!r} maps to non-adjacent JW modes "
            f"({mode_from}, {mode_to}); Jordan-Wigner strings are not supported by "
            "this reconstruction."
        )
    low, high = sorted((mode_from, mode_to))
    direction = 1.0 if mode_to > mode_from else -1.0
    half_t = 0.5 * float(hopping_amplitude)
    if kind == "current":
        return [
            PauliTerm(nq, ps=_pauli_label(nq, {high: "x", low: "y"}), pc=direction * half_t),
            PauliTerm(nq, ps=_pauli_label(nq, {high: "y", low: "x"}), pc=-direction * half_t),
        ]
    if kind == "contact":
        return [
            PauliTerm(nq, ps=_pauli_label(nq, {high: "x", low: "x"}), pc=half_t),
            PauliTerm(nq, ps=_pauli_label(nq, {high: "y", low: "y"}), pc=half_t),
        ]
    raise HHCurrentObservableError(f"Unsupported HH edge operator kind {kind!r}.")


def spin_resolved_hh_edge_current_operator(
    layout: Any,
    *,
    edge: tuple[int, int],
    spin: int,
    hopping_amplitude: float,
) -> PauliPolynomial:
    if float(hopping_amplitude) == 0.0:
        return _zero_polynomial(int(layout.total_qubits))
    return PauliPolynomial(
        "JW",
        _edge_pair_terms(
            layout, edge=edge, spin=spin, hopping_amplitude=float(hopping_amplitude), kind="current"
        ),
    )


def spin_resolved_hh_edge_contact_operator(
    layout: Any,
    *,
    edge: tuple[int, int],
    spin: int,
    hopping_amplitude: float,
) -> PauliPolynomial:
    if float(hopping_amplitude) == 0.0:
        return _zero_polynomial(int(layout.total_qubits))
    return PauliPolynomial(
        "JW",
        _edge_pair_terms(
            layout, edge=edge, spin=spin, hopping_amplitude=float(hopping_amplitude), kind="contact"
        ),
    )


def _total_operator(
    layout: Any,
    *,
    edges: Sequence[tuple[int, int]],
    hopping_amplitude: float,
    kind: str,
) -> PauliPolynomial:
    nq = int(layout.total_qubits)
    if float(hopping_amplitude) == 0.0:
        return _zero_polynomial(nq)
    terms: list[PauliTerm] = []
    for edge in edges:
        for spin in (SPIN_UP, SPIN_DN):
            terms.extend(
                _edge_pair_terms(
                    layout,
                    edge=tuple(edge),
                    spin=spin,
                    hopping_amplitude=float(hopping_amplitude),
                    kind=kind,
                )
            )
    return PauliPolynomial("JW", terms)


def total_hh_charge_current_operator(
    layout: Any,
    *,
    edges: Sequence[tuple[int, int]],
    hopping_amplitude: float,
) -> PauliPolynomial:
    return _total_operator(layout, edges=edges, hopping_amplitude=hopping_amplitude, kind="current")


def total_hh_charge_contact_operator(
    layout: Any,
    *,
    edges: Sequence[tuple[int, int]],
    hopping_amplitude: float,
) -> PauliPolynomial:
    return _total_operator(layout, edges=edges, hopping_amplitude=hopping_amplitude, kind="contact")


@dataclass(frozen=True)
class HHCurrentHoppingResolution:
    """Hopping amplitude resolved from HH settings artifacts."""

    hopping_amplitude: float
    metadata: dict[str, Any] = field(default_factory=dict)


def resolve_hh_current_hopping_from_sources(
    *,
    sources: Mapping[str, Any],
) -> HHCurrentHoppingResolution:
    """Resolve the HH hopping ``t`` from JSON artifacts carrying ``settings``.

    Every supplied source that declares a hopping value must agree; a
    disagreement fails closed with a conflict error.
    """

    resolved_from: list[dict[str, Any]] = []
    for source_key, raw_path in sources.items():
        if raw_path is None:
            continue
        path = Path(raw_path)
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        settings = payload.get("settings")
        if not isinstance(settings, Mapping):
            continue
        value: Any = settings.get("t")
        if value is None:
            value = settings.get("J")
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        hopping = float(value)
        if not math.isfinite(hopping):
            raise HHCurrentObservableError(
                f"HH hopping from {source_key} ({path}) is not finite: {value!r}."
            )
        resolved_from.append(
            {
                "source": str(source_key),
                "path": str(path),
                "field": "t_or_J",
                "value": hopping,
            }
        )
    if not resolved_from:
        raise HHCurrentObservableError(
            "no HH hopping amplitude could be resolved from the supplied sources; "
            "supply --hh-current-hopping-amplitude explicitly."
        )
    reference = float(resolved_from[0]["value"])
    for record in resolved_from[1:]:
        if abs(float(record["value"]) - reference) > _HOPPING_MATCH_TOLERANCE:
            raise HHCurrentObservableError(
                "HH hopping amplitude conflict across sources: "
                f"{resolved_from[0]['source']} declares {reference}, "
                f"{record['source']} declares {record['value']}."
            )
    return HHCurrentHoppingResolution(
        hopping_amplitude=reference,
        metadata={
            "source_schema": "hh_current_hopping_resolution_v1",
            "resolved_from": resolved_from,
        },
    )


@dataclass(frozen=True)
class HHCurrentObservableBundle:
    """Current/contact observables plus bundle-level provenance metadata."""

    observables: tuple[QSEObservable, ...]
    current_labels: tuple[str, ...]
    contact_label: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


def build_hh_current_observable_bundle(
    *,
    layout: Any,
    hopping_amplitude: float,
    edge_orientation: str = "positive_chain",
    include_contact: bool = True,
    contact_policy: str = HH_CURRENT_CONTACT_POLICY,
    peierls_policy: str = HH_CURRENT_PEIERLS_POLICY,
    config: Any = None,
    hopping_source_metadata: Mapping[str, Any] | None = None,
) -> HHCurrentObservableBundle:
    """Build the HH current (and optional contact) QSE transition observables."""

    del config
    if str(peierls_policy) != HH_CURRENT_PEIERLS_POLICY:
        raise HHCurrentObservableError(
            f"HH current Peierls policy must be {HH_CURRENT_PEIERLS_POLICY!r}; "
            f"got {peierls_policy!r}."
        )
    if str(contact_policy) != HH_CURRENT_CONTACT_POLICY:
        raise HHCurrentObservableError(
            f"HH current contact policy must be {HH_CURRENT_CONTACT_POLICY!r}; "
            f"got {contact_policy!r}."
        )
    hopping = float(hopping_amplitude)
    if not math.isfinite(hopping):
        raise HHCurrentObservableError(f"HH hopping amplitude must be finite; got {hopping!r}.")
    edges = directed_hh_current_edges(layout, edge_orientation=str(edge_orientation))
    zero = hopping == 0.0

    current_name = f"hh_J[{edge_orientation}]"
    contact_name = f"hh_K[{edge_orientation}]"
    current_poly = total_hh_charge_current_operator(
        layout, edges=edges, hopping_amplitude=hopping
    )
    observables: list[QSEObservable] = [
        polynomial_observable(
            current_poly,
            name=current_name,
            metadata={
                "source": HH_CURRENT_OBSERVABLES_SCHEMA_VERSION,
                "role": "charge_current",
                "zero_operator": bool(zero),
            },
        )
    ]
    contact_label: str | None = None
    if bool(include_contact):
        contact_poly = total_hh_charge_contact_operator(
            layout, edges=edges, hopping_amplitude=hopping
        )
        observables.append(
            polynomial_observable(
                contact_poly,
                name=contact_name,
                metadata={
                    "source": HH_CURRENT_OBSERVABLES_SCHEMA_VERSION,
                    "role": "peierls_contact",
                    "zero_operator": bool(zero),
                },
            )
        )
        contact_label = contact_name

    metadata: dict[str, Any] = {
        "schema_version": HH_CURRENT_OBSERVABLES_SCHEMA_VERSION,
        "edge_orientation": str(edge_orientation),
        "directed_edges": [[int(edge[0]), int(edge[1])] for edge in edges],
        "peierls_policy": HH_CURRENT_PEIERLS_POLICY,
        "contact_policy": HH_CURRENT_CONTACT_POLICY,
        "hopping_amplitude": hopping,
        "include_contact": bool(include_contact),
        "current_zero_operator": bool(zero),
        "contact_zero_operator": bool(zero) if include_contact else None,
    }
    if hopping_source_metadata:
        metadata["hopping_source"] = dict(hopping_source_metadata)

    return HHCurrentObservableBundle(
        observables=tuple(observables),
        current_labels=(current_name,),
        contact_label=contact_label,
        metadata=metadata,
    )
