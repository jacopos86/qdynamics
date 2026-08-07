"""Hubbard-Holstein neutral-response observable builders for QSE.

The builders in this module are intentionally sidecar-style: they produce
``QSEObservable`` records and response-channel specs for the existing QSE spectra
pipeline, while delegating all fermion/phonon qubit layout conventions to the
same Hubbard-Holstein helpers used by Hamiltonian and pool construction.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.qse_spectra.core import (
    QSEObservable,
    QSEPruningConfig,
    _clean_polynomial_terms,
    normalize_statevector,
    polynomial_observable,
)
from pipelines.qse_spectra.response_functions import ResponseChannel
from src.quantum.compiled_polynomial import apply_compiled_polynomial, compile_polynomial_action
from src.quantum.hubbard_latex_python_pairs import (
    SPIN_DN,
    SPIN_UP,
    boson_displacement_operator,
    boson_operator,
    boson_qubits_per_site,
    jw_number_operator,
    mode_index,
    phonon_qubit_indices_for_site,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm

HH_RESPONSE_OBSERVABLES_SCHEMA_VERSION = "hh_neutral_response_observables_v1"
HH_NEUTRAL_RESPONSE_CHANNELS = ("nn", "XX", "PP", "nX", "C_nX")

_MISSING = object()


class HHResponseObservableError(ValueError):
    """Raised when HH response observables cannot be built safely."""


@dataclass(frozen=True)
class HHResponseLayout:
    """Resolved HH qubit/register layout for response-observable construction."""

    num_sites: int
    n_ph_max: int
    boson_encoding: str
    ordering: str
    boundary: str
    total_qubits: int | None = None
    num_particles: tuple[int, int] | None = None
    source_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        num_sites = _strict_int(self.num_sites, name="num_sites", min_value=1)
        n_ph_max = _strict_int(self.n_ph_max, name="n_ph_max", min_value=0)
        boson_encoding = str(self.boson_encoding).strip().lower()
        if boson_encoding not in {"binary", "unary"}:
            raise HHResponseObservableError("HH response boson_encoding must be 'binary' or 'unary'.")
        ordering = str(self.ordering).strip().lower()
        if ordering not in {"blocked", "interleaved"}:
            raise HHResponseObservableError("HH response ordering must be 'blocked' or 'interleaved'.")
        boundary = str(self.boundary).strip().lower()
        if boundary not in {"open", "periodic"}:
            raise HHResponseObservableError("HH response boundary must be 'open' or 'periodic'.")
        qpb = int(boson_qubits_per_site(int(n_ph_max), boson_encoding))
        inferred_total = int(2 * num_sites + num_sites * qpb)
        total_qubits = inferred_total if self.total_qubits is None else int(self.total_qubits)
        if total_qubits != inferred_total:
            raise HHResponseObservableError(
                "HH response layout qubit count mismatch: "
                f"expected 2*num_sites + num_sites*qpb = {inferred_total}, got {total_qubits}."
            )
        num_particles = None
        if self.num_particles is not None:
            raw = tuple(_strict_int(x, name="num_particles entry", min_value=0) for x in self.num_particles)
            if len(raw) != 2:
                raise HHResponseObservableError("num_particles must contain (n_up, n_dn).")
            if raw[0] < 0 or raw[1] < 0:
                raise HHResponseObservableError("num_particles entries must be non-negative.")
            num_particles = (int(raw[0]), int(raw[1]))
        object.__setattr__(self, "num_sites", num_sites)
        object.__setattr__(self, "n_ph_max", n_ph_max)
        object.__setattr__(self, "boson_encoding", boson_encoding)
        object.__setattr__(self, "ordering", ordering)
        object.__setattr__(self, "boundary", boundary)
        object.__setattr__(self, "total_qubits", total_qubits)
        object.__setattr__(self, "num_particles", num_particles)

    @property
    def fermion_qubits(self) -> int:
        return int(2 * int(self.num_sites))

    @property
    def qubits_per_boson_site(self) -> int:
        return int(boson_qubits_per_site(int(self.n_ph_max), str(self.boson_encoding)))

    def phonon_qubits_for_site(self, site: int) -> tuple[int, ...]:
        return tuple(
            int(q)
            for q in phonon_qubit_indices_for_site(
                int(site),
                n_sites=int(self.num_sites),
                qpb=int(self.qubits_per_boson_site),
                fermion_qubits=int(self.fermion_qubits),
            )
        )

    def to_manifest(self) -> dict[str, Any]:
        payload = {
            "problem_key": "hh",
            "num_sites": int(self.num_sites),
            "n_ph_max": int(self.n_ph_max),
            "boson_encoding": str(self.boson_encoding),
            "ordering": str(self.ordering),
            "boundary": str(self.boundary),
            "fermion_qubits": int(self.fermion_qubits),
            "qubits_per_boson_site": int(self.qubits_per_boson_site),
            "total_qubits": int(self.total_qubits),
        }
        if self.num_particles is not None:
            payload["num_particles"] = [int(self.num_particles[0]), int(self.num_particles[1])]
        if self.source_metadata:
            payload["source_metadata"] = _json_safe(self.source_metadata)
        return payload


@dataclass(frozen=True)
class HHFormFactor:
    """A spatial HH response form factor over lattice sites."""

    label: str
    weights: tuple[complex, ...]
    source: str

    def to_manifest(self) -> dict[str, Any]:
        return {
            "label": str(self.label),
            "source": str(self.source),
            "weights": [_complex_pair(value) for value in self.weights],
        }


@dataclass(frozen=True)
class HHResponseObservableBundle:
    """Generated transition observables plus response-channel records."""

    observables: tuple[QSEObservable, ...]
    response_channels: tuple[ResponseChannel, ...]
    metadata: Mapping[str, Any]


def resolve_hh_response_layout_from_sources(
    *,
    expected_nq: int,
    sources: Mapping[str, Path | str | None],
) -> HHResponseLayout:
    """Resolve HH response layout from existing Hamiltonian/artifact settings.

    The resolver accepts only explicit existing JSON settings from the HH
    Hamiltonian/basis path.  It does not invent a qubit map.  If no consistent
    HH layout can be proven, callers should require explicit observable JSON.
    """

    expected_nq_i = _strict_int(expected_nq, name="expected_nq", min_value=1)
    resolved: list[tuple[str, Path, dict[str, Any]]] = []
    for role, raw_path in sources.items():
        if raw_path is None:
            continue
        path = Path(raw_path)
        payload = _load_json_mapping(path, role=str(role))
        settings = _settings_mapping_or_none(payload)
        if settings is None:
            continue
        problem_key = _problem_key_from_settings(settings)
        if problem_key is None:
            continue
        if problem_key != "hh":
            raise HHResponseObservableError(
                f"HH response requested, but {role} settings identify problem {problem_key!r}."
            )
        layout_payload = _layout_payload_from_settings(payload, settings, role=str(role), path=path)
        resolved.append((str(role), path, layout_payload))

    if not resolved:
        raise HHResponseObservableError(
            "HH neutral response channels require a resolvable HH layout from existing Hamiltonian/basis "
            "artifact settings. Provide HH settings via --hamiltonian-json/--basis-artifact-json, "
            "or provide explicit --transition-observable-json instead."
        )

    merged_layout = dict(resolved[0][2])
    conflicts: list[str] = []
    for key in ("num_sites", "n_ph_max", "boson_encoding", "ordering", "boundary", "num_particles"):
        present = [(role, layout[key]) for role, _path, layout in resolved if key in layout]
        if not present:
            continue
        first_role, first_value = present[0]
        for role, value in present[1:]:
            if _normalize_for_conflict(first_value) != _normalize_for_conflict(value):
                conflicts.append(f"{key}: {first_role}={first_value!r} {role}={value!r}")
        if key not in merged_layout:
            merged_layout[key] = first_value
    if conflicts:
        raise HHResponseObservableError("HH response layout sources conflict: " + "; ".join(conflicts))

    source_metadata = {
        "schema_version": HH_RESPONSE_OBSERVABLES_SCHEMA_VERSION,
        "resolved_from": [
            {"role": role, "path": str(path), "layout_keys": sorted(layout.keys())}
            for role, path, layout in resolved
        ],
    }
    return HHResponseLayout(
        num_sites=int(merged_layout["num_sites"]),
        n_ph_max=int(merged_layout["n_ph_max"]),
        boson_encoding=str(merged_layout["boson_encoding"]),
        ordering=str(merged_layout["ordering"]),
        boundary=str(merged_layout["boundary"]),
        total_qubits=int(expected_nq_i),
        num_particles=(tuple(merged_layout["num_particles"]) if "num_particles" in merged_layout else None),
        source_metadata=source_metadata,
    )


def parse_hh_form_factor(spec: str | Sequence[complex | float | int], *, num_sites: int) -> HHFormFactor:
    """Parse a code-facing HH form-factor specification."""

    n_sites = _strict_int(num_sites, name="num_sites", min_value=1)
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes, bytearray)):
        weights = tuple(complex(value) for value in spec)
        if len(weights) != n_sites:
            raise HHResponseObservableError(f"form factor has length {len(weights)}; expected {n_sites}.")
        return HHFormFactor(label="custom", weights=weights, source="sequence")

    text = str(spec).strip() if spec is not None else "staggered"
    if text == "":
        raise HHResponseObservableError("HH response form factor must be non-empty.")
    key = text.lower()
    if key in {"uniform", "ones"}:
        return HHFormFactor(label="uniform", weights=tuple(1.0 + 0.0j for _ in range(n_sites)), source=text)
    if key in {"uniform_normalized", "uniform_norm"}:
        scale = 1.0 / math.sqrt(float(n_sites))
        return HHFormFactor(label="uniform_normalized", weights=tuple(scale + 0.0j for _ in range(n_sites)), source=text)
    if key in {"staggered", "alternating"}:
        return HHFormFactor(label="staggered", weights=tuple(((-1.0) ** j) + 0.0j for j in range(n_sites)), source=text)
    if key in {"staggered_normalized", "staggered_norm"}:
        scale = 1.0 / math.sqrt(float(n_sites))
        return HHFormFactor(
            label="staggered_normalized",
            weights=tuple((((-1.0) ** j) * scale) + 0.0j for j in range(n_sites)),
            source=text,
        )
    if key.startswith("site:"):
        site = _strict_int(key.split(":", 1)[1], name="site", min_value=0)
        if site >= n_sites:
            raise HHResponseObservableError(f"site form factor index {site} is out of range for L={n_sites}.")
        weights = [0.0 + 0.0j] * n_sites
        weights[site] = 1.0 + 0.0j
        return HHFormFactor(label=f"site_{site}", weights=tuple(weights), source=text)
    if key.startswith("obc_sine:"):
        mode = _strict_int(key.split(":", 1)[1], name="obc_sine mode", min_value=1)
        if mode > n_sites:
            raise HHResponseObservableError(f"obc_sine mode {mode} must be <= L={n_sites}.")
        scale = math.sqrt(2.0 / float(n_sites + 1))
        weights = tuple(
            complex(scale * math.sin(math.pi * float(mode) * float(j + 1) / float(n_sites + 1)), 0.0)
            for j in range(n_sites)
        )
        return HHFormFactor(label=f"obc_sine_m{mode}", weights=weights, source=text)
    if key.startswith("csv:"):
        csv_text = text.split(":", 1)[1]
        weights = _parse_complex_csv(csv_text, expected_len=n_sites)
        return HHFormFactor(label="custom_csv", weights=weights, source=text)
    if "," in text:
        weights = _parse_complex_csv(text, expected_len=n_sites)
        return HHFormFactor(label="custom_csv", weights=weights, source=text)
    raise HHResponseObservableError(
        "Unsupported HH response form factor. Use uniform, uniform_normalized, staggered, "
        "staggered_normalized, site:<i>, obc_sine:<m>, or csv:w0,w1,... ."
    )


def normalize_hh_neutral_response_channels(channels: Sequence[str] | str | None) -> tuple[str, ...]:
    """Normalize repeatable/comma-separated HH response channel names."""

    if channels is None:
        return ()
    raw_items: list[str]
    if isinstance(channels, str):
        raw_items = [channels]
    else:
        raw_items = [str(item) for item in channels]
    normalized: list[str] = []
    for raw in raw_items:
        for token in str(raw).split(","):
            text = token.strip()
            if text == "":
                continue
            key = text.replace("-", "_").lower()
            if key == "all":
                normalized.extend(HH_NEUTRAL_RESPONSE_CHANNELS)
                continue
            aliases = {
                "density": "nn",
                "s_nn": "nn",
                "nn": "nn",
                "x": "XX",
                "xx": "XX",
                "s_xx": "XX",
                "p": "PP",
                "pp": "PP",
                "s_pp": "PP",
                "nx": "nX",
                "n_x": "nX",
                "s_nx": "nX",
                "cnx": "C_nX",
                "c_nx": "C_nX",
                "c_{nx}": "C_nX",
                "c_{n_x}": "C_nX",
            }
            try:
                normalized.append(aliases[key])
            except KeyError as exc:
                raise HHResponseObservableError(
                    f"Unsupported HH neutral response channel {text!r}; expected one of "
                    f"{HH_NEUTRAL_RESPONSE_CHANNELS!r} or all."
                ) from exc
    out: list[str] = []
    seen: set[str] = set()
    for channel in normalized:
        if channel not in seen:
            seen.add(channel)
            out.append(channel)
    return tuple(out)


def density_baseline_from_state(
    layout: HHResponseLayout,
    prepared_state: np.ndarray,
    *,
    config: QSEPruningConfig | None = None,
) -> float:
    """Return ``bar n`` averaged over HH sites in ``prepared_state``."""

    cfg = config if config is not None else QSEPruningConfig()
    psi, _, nq = normalize_statevector(prepared_state)
    if int(nq) != int(layout.total_qubits):
        raise HHResponseObservableError(f"prepared_state has nq={nq}; HH layout has nq={layout.total_qubits}.")
    values: list[float] = []
    max_imag = 0.0
    for site in range(int(layout.num_sites)):
        op = site_density_operator(layout, site=site, density_baseline=None, config=cfg)
        clean = _clean_observable_polynomial(op, config=cfg)
        compiled = compile_polynomial_action(clean, tol=float(cfg.polynomial_drop_abs_tol))
        vec = apply_compiled_polynomial(psi, compiled)
        value = complex(np.vdot(psi, vec))
        max_imag = max(max_imag, abs(float(value.imag)))
        values.append(float(value.real))
    if max_imag > 1.0e-9:
        raise HHResponseObservableError(
            f"site-density expectation has imaginary part {max_imag}, cannot infer real density baseline."
        )
    return float(sum(values) / float(layout.num_sites))


def site_density_operator(
    layout: HHResponseLayout,
    *,
    site: int,
    density_baseline: float | None = None,
    config: QSEPruningConfig | None = None,
) -> PauliPolynomial:
    """Build local HH electronic density or density fluctuation ``delta n_i``."""

    del config
    site_i = _strict_int(site, name="site", min_value=0)
    if site_i >= int(layout.num_sites):
        raise HHResponseObservableError(f"site index {site_i} out of range for L={layout.num_sites}.")
    nq = int(layout.total_qubits)
    up_mode = mode_index(site_i, SPIN_UP, indexing=str(layout.ordering), n_sites=int(layout.num_sites))
    dn_mode = mode_index(site_i, SPIN_DN, indexing=str(layout.ordering), n_sites=int(layout.num_sites))
    op = jw_number_operator("JW", nq, int(up_mode)) + jw_number_operator("JW", nq, int(dn_mode))
    if density_baseline is not None:
        baseline = _finite_float(density_baseline, name="density_baseline")
        if baseline != 0.0:
            op += (-float(baseline)) * _identity_polynomial(nq)
    return op


def phonon_displacement_operator(
    layout: HHResponseLayout,
    *,
    site: int,
    config: QSEPruningConfig | None = None,
) -> PauliPolynomial:
    """Build local HH phonon displacement ``X_i=b_i^dagger+b_i``."""

    del config
    _require_phonon_support(layout)
    return boson_displacement_operator(
        "JW",
        int(layout.total_qubits),
        layout.phonon_qubits_for_site(site),
        n_ph_max=int(layout.n_ph_max),
        encoding=str(layout.boson_encoding),
    )


def phonon_momentum_operator(
    layout: HHResponseLayout,
    *,
    site: int,
    config: QSEPruningConfig | None = None,
) -> PauliPolynomial:
    """Build local HH phonon momentum ``P_i=i(b_i^dagger-b_i)``."""

    del config
    _require_phonon_support(layout)
    qubits = layout.phonon_qubits_for_site(site)
    b = boson_operator(
        "JW",
        int(layout.total_qubits),
        qubits,
        which="b",
        n_ph_max=int(layout.n_ph_max),
        encoding=str(layout.boson_encoding),
    )
    bdag = boson_operator(
        "JW",
        int(layout.total_qubits),
        qubits,
        which="bdag",
        n_ph_max=int(layout.n_ph_max),
        encoding=str(layout.boson_encoding),
    )
    return (1j * bdag) + ((-1j) * b)


def weighted_density_fluctuation_operator(
    layout: HHResponseLayout,
    form_factor: HHFormFactor | Sequence[complex | float | int] | str,
    *,
    density_baseline: float | None,
    config: QSEPruningConfig | None = None,
) -> PauliPolynomial:
    cfg = config if config is not None else QSEPruningConfig()
    form = _ensure_form_factor(form_factor, num_sites=int(layout.num_sites))
    out = PauliPolynomial("JW")
    for site, weight in enumerate(form.weights):
        if abs(complex(weight)) <= float(cfg.polynomial_drop_abs_tol):
            continue
        out += complex(weight) * site_density_operator(
            layout,
            site=int(site),
            density_baseline=float(density_baseline),
            config=cfg,
        )
    return _clean_observable_polynomial(out, config=cfg)


def weighted_phonon_displacement_operator(
    layout: HHResponseLayout,
    form_factor: HHFormFactor | Sequence[complex | float | int] | str,
    *,
    config: QSEPruningConfig | None = None,
) -> PauliPolynomial:
    cfg = config if config is not None else QSEPruningConfig()
    form = _ensure_form_factor(form_factor, num_sites=int(layout.num_sites))
    out = PauliPolynomial("JW")
    for site, weight in enumerate(form.weights):
        if abs(complex(weight)) <= float(cfg.polynomial_drop_abs_tol):
            continue
        out += complex(weight) * phonon_displacement_operator(layout, site=int(site), config=cfg)
    return _clean_observable_polynomial(out, config=cfg)


def weighted_phonon_momentum_operator(
    layout: HHResponseLayout,
    form_factor: HHFormFactor | Sequence[complex | float | int] | str,
    *,
    config: QSEPruningConfig | None = None,
) -> PauliPolynomial:
    cfg = config if config is not None else QSEPruningConfig()
    form = _ensure_form_factor(form_factor, num_sites=int(layout.num_sites))
    out = PauliPolynomial("JW")
    for site, weight in enumerate(form.weights):
        if abs(complex(weight)) <= float(cfg.polynomial_drop_abs_tol):
            continue
        out += complex(weight) * phonon_momentum_operator(layout, site=int(site), config=cfg)
    return _clean_observable_polynomial(out, config=cfg)


def mixed_density_displacement_operator(
    layout: HHResponseLayout,
    *,
    separation: int = 0,
    density_baseline: float,
    config: QSEPruningConfig | None = None,
) -> PauliPolynomial:
    """Build ``C_nX(r)=sum_i delta n_i X_{i+r}`` using HH boundary semantics."""

    cfg = config if config is not None else QSEPruningConfig()
    sep = int(separation)
    out = PauliPolynomial("JW")
    for site_i in range(int(layout.num_sites)):
        site_j = int(site_i) + int(sep)
        if str(layout.boundary) == "periodic":
            site_j %= int(layout.num_sites)
        elif site_j < 0 or site_j >= int(layout.num_sites):
            continue
        n_i = site_density_operator(layout, site=int(site_i), density_baseline=float(density_baseline), config=cfg)
        x_j = phonon_displacement_operator(layout, site=int(site_j), config=cfg)
        out += n_i * x_j
    return _clean_observable_polynomial(out, config=cfg)


def build_hh_neutral_response_observable_bundle(
    *,
    layout: HHResponseLayout,
    channels: Sequence[str] | str,
    form_factor: HHFormFactor | Sequence[complex | float | int] | str = "staggered",
    prepared_state: np.ndarray | None = None,
    density_baseline: float | None = None,
    nx_separation: int = 0,
    config: QSEPruningConfig | None = None,
) -> HHResponseObservableBundle:
    """Build HH neutral-response observables and matching response channels."""

    cfg = config if config is not None else QSEPruningConfig()
    channel_names = normalize_hh_neutral_response_channels(channels)
    if not channel_names:
        return HHResponseObservableBundle(observables=(), response_channels=(), metadata={})
    form = _ensure_form_factor(form_factor, num_sites=int(layout.num_sites))
    label_suffix = _safe_label(form.label)
    n_name = f"hh_n[{label_suffix}]"
    x_name = f"hh_X[{label_suffix}]"
    p_name = f"hh_P[{label_suffix}]"
    c_name = f"hh_C_nX[r={int(nx_separation)}]"

    needed_n = any(channel in {"nn", "nX"} for channel in channel_names)
    needed_x = any(channel in {"XX", "nX"} for channel in channel_names)
    needed_p = any(channel == "PP" for channel in channel_names)
    needed_c = any(channel == "C_nX" for channel in channel_names)
    baseline = None
    if needed_n or needed_c:
        baseline = _resolve_density_baseline(
            layout,
            prepared_state=prepared_state,
            density_baseline=density_baseline,
            config=cfg,
        )

    observables_by_name: dict[str, QSEObservable] = {}
    if needed_n:
        n_poly = weighted_density_fluctuation_operator(
            layout,
            form,
            density_baseline=float(baseline),
            config=cfg,
        )
        observables_by_name[n_name] = polynomial_observable(
            n_poly,
            name=n_name,
            metadata=_observable_metadata(
                layout=layout,
                form_factor=form,
                channel_family="n",
                formula="n[f]=sum_j f(j)(n_j-bar_n I)",
                density_baseline=baseline,
            ),
        )
    if needed_x:
        x_poly = weighted_phonon_displacement_operator(layout, form, config=cfg)
        observables_by_name[x_name] = polynomial_observable(
            x_poly,
            name=x_name,
            metadata=_observable_metadata(
                layout=layout,
                form_factor=form,
                channel_family="X",
                formula="X[f]=sum_j f(j)(b_j^dagger+b_j)",
                density_baseline=baseline,
            ),
        )
    if needed_p:
        p_poly = weighted_phonon_momentum_operator(layout, form, config=cfg)
        observables_by_name[p_name] = polynomial_observable(
            p_poly,
            name=p_name,
            metadata=_observable_metadata(
                layout=layout,
                form_factor=form,
                channel_family="P",
                formula="P[f]=sum_j f(j)i(b_j^dagger-b_j)",
                density_baseline=baseline,
            ),
        )
    if needed_c:
        c_poly = mixed_density_displacement_operator(
            layout,
            separation=int(nx_separation),
            density_baseline=float(baseline),
            config=cfg,
        )
        observables_by_name[c_name] = polynomial_observable(
            c_poly,
            name=c_name,
            metadata=_observable_metadata(
                layout=layout,
                form_factor=form,
                channel_family="C_nX",
                formula="C_nX(r)=sum_i delta n_i X_{i+r}",
                density_baseline=baseline,
                extra={"separation": int(nx_separation), "form_factor_applies": False},
            ),
        )

    response_channels: list[ResponseChannel] = []
    for channel in channel_names:
        if channel == "nn":
            response_channels.append(_response_channel(n_name, n_name, "nn"))
        elif channel == "XX":
            response_channels.append(_response_channel(x_name, x_name, "XX"))
        elif channel == "PP":
            response_channels.append(_response_channel(p_name, p_name, "PP"))
        elif channel == "nX":
            response_channels.append(_response_channel(n_name, x_name, "nX"))
        elif channel == "C_nX":
            response_channels.append(_response_channel(c_name, c_name, "C_nX"))
        else:  # pragma: no cover - normalized earlier.
            raise HHResponseObservableError(f"Unsupported HH channel after normalization: {channel!r}")

    metadata = {
        "schema_version": HH_RESPONSE_OBSERVABLES_SCHEMA_VERSION,
        "channels_requested": list(channel_names),
        "observable_count": int(len(observables_by_name)),
        "response_channel_count": int(len(response_channels)),
        "layout": layout.to_manifest(),
        "form_factor": form.to_manifest(),
        "form_factor_applies_to": ["n", "X", "P"],
        "density_baseline": None if baseline is None else float(baseline),
        "nx_separation": int(nx_separation),
    }
    return HHResponseObservableBundle(
        observables=tuple(observables_by_name.values()),
        response_channels=tuple(response_channels),
        metadata=metadata,
    )


def _response_channel(a_label: str, b_label: str, channel_kind: str) -> ResponseChannel:
    return ResponseChannel(
        A_label=str(a_label),
        B_label=str(b_label),
        channel_kind=str(channel_kind),
        metadata={
            "source": HH_RESPONSE_OBSERVABLES_SCHEMA_VERSION,
            "channel_family": str(channel_kind),
        },
    )


def _observable_metadata(
    *,
    layout: HHResponseLayout,
    form_factor: HHFormFactor,
    channel_family: str,
    formula: str,
    density_baseline: float,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source": HH_RESPONSE_OBSERVABLES_SCHEMA_VERSION,
        "channel_family": str(channel_family),
        "operator_formula": str(formula),
        "layout": layout.to_manifest(),
        "form_factor": form_factor.to_manifest(),
        "density_baseline": None if density_baseline is None else float(density_baseline),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def _resolve_density_baseline(
    layout: HHResponseLayout,
    *,
    prepared_state: np.ndarray | None,
    density_baseline: float | None,
    config: QSEPruningConfig,
) -> float:
    if density_baseline is not None:
        return _finite_float(density_baseline, name="density_baseline")
    if prepared_state is not None:
        return density_baseline_from_state(layout, prepared_state, config=config)
    if layout.num_particles is not None:
        return float((int(layout.num_particles[0]) + int(layout.num_particles[1])) / float(layout.num_sites))
    raise HHResponseObservableError(
        "HH density fluctuation requires density_baseline, prepared_state, or num_particles in resolved layout."
    )


def _ensure_form_factor(
    form_factor: HHFormFactor | Sequence[complex | float | int] | str,
    *,
    num_sites: int,
) -> HHFormFactor:
    if isinstance(form_factor, HHFormFactor):
        if len(form_factor.weights) != int(num_sites):
            raise HHResponseObservableError(
                f"form factor {form_factor.label!r} has length {len(form_factor.weights)}; expected {num_sites}."
            )
        return form_factor
    return parse_hh_form_factor(form_factor, num_sites=int(num_sites))


def _require_phonon_support(layout: HHResponseLayout) -> None:
    if int(layout.n_ph_max) <= 0:
        raise HHResponseObservableError(
            "HH phonon X/P response observables require n_ph_max >= 1; provide explicit observable JSON otherwise."
        )


def _clean_observable_polynomial(poly: PauliPolynomial, *, config: QSEPruningConfig) -> PauliPolynomial:
    clean = _clean_polynomial_terms(
        poly,
        drop_abs_tol=float(config.polynomial_drop_abs_tol),
        require_real_coefficients=False,
        coeff_imag_abs_tol=float(config.hamiltonian_coeff_imag_absolute_tolerance),
        allow_empty_after_pruning=False,
    )
    return clean.polynomial


def _identity_polynomial(nq: int) -> PauliPolynomial:
    return PauliPolynomial("JW", [PauliTerm(int(nq), ps="e" * int(nq), pc=1.0)])


def _load_json_mapping(path: Path, *, role: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HHResponseObservableError(f"{role} JSON does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise HHResponseObservableError(f"{role} JSON is not valid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise HHResponseObservableError(f"{role} JSON must be a mapping: {path}")
    return payload


def _settings_mapping_or_none(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    settings = payload.get("settings")
    if isinstance(settings, Mapping):
        return settings
    if any(key in payload for key in ("problem", "problem_key", "family")):
        return payload
    return None


def _settings_value(settings: Mapping[str, Any], *keys: str, default: Any = _MISSING) -> Any:
    for key in keys:
        if key in settings and settings[key] is not None:
            return settings[key]
    return default


def _problem_key_from_settings(settings: Mapping[str, Any]) -> str | None:
    raw = _settings_value(settings, "problem", "problem_key", "family", default=None)
    if raw is None:
        return None
    key = str(raw).strip().lower()
    aliases = {
        "hubbard_holstein": "hh",
        "hubbard-holstein": "hh",
        "holstein": "hh",
    }
    return aliases.get(key, key)


def _layout_payload_from_settings(
    payload: Mapping[str, Any],
    settings: Mapping[str, Any],
    *,
    role: str,
    path: Path,
) -> dict[str, Any]:
    required = {
        "num_sites": ("L", "num_sites"),
        "n_ph_max": ("n_ph_max", "nph", "n_ph"),
        "boson_encoding": ("boson_encoding",),
        "ordering": ("ordering", "indexing"),
        "boundary": ("boundary",),
    }
    out: dict[str, Any] = {}
    missing: list[str] = []
    for output_key, aliases in required.items():
        value = _settings_value(settings, *aliases)
        if value is _MISSING:
            missing.append(output_key)
            continue
        if output_key in {"num_sites", "n_ph_max"}:
            out[output_key] = _strict_int(value, name=output_key, min_value=0 if output_key == "n_ph_max" else 1)
        else:
            out[output_key] = str(value).strip().lower()
    if missing:
        raise HHResponseObservableError(
            f"HH response layout in {role} JSON {path} is missing required fields: {missing!r}."
        )
    num_particles = _num_particles_from_payload(payload, settings)
    if num_particles is not None:
        out["num_particles"] = [int(num_particles[0]), int(num_particles[1])]
    return out


def _num_particles_from_payload(payload: Mapping[str, Any], settings: Mapping[str, Any]) -> tuple[int, int] | None:
    adapt_vqe = payload.get("adapt_vqe")
    if isinstance(adapt_vqe, Mapping):
        raw = adapt_vqe.get("num_particles")
        if isinstance(raw, Mapping):
            n_up = raw.get("n_up")
            n_dn = raw.get("n_dn")
            if n_up is not None and n_dn is not None:
                return (
                    _strict_int(n_up, name="adapt_vqe.num_particles.n_up", min_value=0),
                    _strict_int(n_dn, name="adapt_vqe.num_particles.n_dn", min_value=0),
                )
    raw_settings = settings.get("num_particles")
    if isinstance(raw_settings, Sequence) and not isinstance(raw_settings, (str, bytes, bytearray)):
        values = list(raw_settings)
        if len(values) == 2:
            return (
                _strict_int(values[0], name="settings.num_particles[0]", min_value=0),
                _strict_int(values[1], name="settings.num_particles[1]", min_value=0),
            )
    return None


def _parse_complex_csv(text: str, *, expected_len: int) -> tuple[complex, ...]:
    raw_values = [item.strip() for item in str(text).split(",")]
    if any(item == "" for item in raw_values):
        raise HHResponseObservableError(f"Invalid HH form-factor csv values {text!r}.")
    values: list[complex] = []
    for item in raw_values:
        try:
            values.append(complex(item.replace("i", "j")))
        except ValueError as exc:
            raise HHResponseObservableError(f"Invalid HH form-factor weight {item!r}.") from exc
    if len(values) != int(expected_len):
        raise HHResponseObservableError(f"form factor has length {len(values)}; expected {expected_len}.")
    return tuple(values)


def _strict_int(value: Any, *, name: str, min_value: int | None = None) -> int:
    if isinstance(value, bool):
        raise HHResponseObservableError(f"{name} must be an integer; got {value!r}.")
    if isinstance(value, int):
        out = int(value)
    elif isinstance(value, float):
        if not math.isfinite(value) or not float(value).is_integer():
            raise HHResponseObservableError(f"{name} must be an integer; got {value!r}.")
        out = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if text == "":
            raise HHResponseObservableError(f"{name} must be an integer; got {value!r}.")
        try:
            out = int(text, 10)
        except Exception as exc:
            raise HHResponseObservableError(f"{name} must be an integer; got {value!r}.") from exc
        canonical = str(out)
        if text not in {canonical, f"+{canonical}"}:
            raise HHResponseObservableError(f"{name} must be an integer; got {value!r}.")
    else:
        raise HHResponseObservableError(f"{name} must be an integer; got {value!r}.")
    if min_value is not None and out < int(min_value):
        raise HHResponseObservableError(f"{name} must be >= {int(min_value)}; got {out}.")
    return out


def _finite_float(value: Any, *, name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise HHResponseObservableError(f"{name} must be numeric; got {value!r}.") from exc
    if not math.isfinite(out):
        raise HHResponseObservableError(f"{name} must be finite; got {out!r}.")
    return float(out)


def _normalize_for_conflict(value: Any) -> Any:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return round(float(value), 15)
    if isinstance(value, str):
        text = value.strip()
        try:
            number = float(text)
        except ValueError:
            return text.lower()
        if math.isfinite(number):
            return round(float(number), 15)
        return text.lower()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_normalize_for_conflict(item) for item in value)
    return value


def _complex_pair(value: complex | float | int) -> list[float]:
    value_c = complex(value)
    return [float(value_c.real), float(value_c.imag)]


def _safe_label(value: str) -> str:
    out = []
    for ch in str(value):
        if ch.isalnum() or ch in {"_", "-", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "custom"


def _json_safe(value: Any) -> Any:
    if isinstance(value, complex):
        return {"re": float(value.real), "im": float(value.imag)}
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(v) for v in value]
    return value


__all__ = [
    "HH_RESPONSE_OBSERVABLES_SCHEMA_VERSION",
    "HH_NEUTRAL_RESPONSE_CHANNELS",
    "HHFormFactor",
    "HHResponseLayout",
    "HHResponseObservableBundle",
    "HHResponseObservableError",
    "build_hh_neutral_response_observable_bundle",
    "density_baseline_from_state",
    "mixed_density_displacement_operator",
    "normalize_hh_neutral_response_channels",
    "parse_hh_form_factor",
    "phonon_displacement_operator",
    "phonon_momentum_operator",
    "resolve_hh_response_layout_from_sources",
    "site_density_operator",
    "weighted_density_fluctuation_operator",
    "weighted_phonon_displacement_operator",
    "weighted_phonon_momentum_operator",
]
