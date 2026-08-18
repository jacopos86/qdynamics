#!/usr/bin/env python3
"""QSE conductivity/current response payloads (diagnostic postprocessing).

RECONSTRUCTION (2026-08-18): the original module was never committed and was
lost when snapshot commit 6442fbb5 captured its importers without it. This
implementation is reconstructed against the committed behavioral spec in
``test/test_qse_conductivity.py`` and the CLI wiring in ``__main__.py``.

Given a solved ``QSEResult`` whose transition observables include a current
operator ``J`` (and optionally a contact/diamagnetic operator ``K``), each
channel reports, per QSE root ``n`` with excitation energy
``omega_n = E_n - E_0`` and current amplitude ``A_n = <phi_n|J|psi_0>``:

    paramagnetic S_JJ(omega) = sum_n |A_n|^2 * kernel(omega - omega_n)
    sigma_reg(omega)         = pi * S_JJ^{omega_n > floor}(omega) / omega
                               (reported as zero at or below the omega floor)

The contact term is recorded as a bare expectation ``<psi_0|K|psi_0>`` and is
never combined into a Drude delta (``contact_record_only_no_drude_delta``);
the Drude weight is explicitly ``not_evaluated``. Zero current sources are
reported explicitly rather than omitted. Complex scalars are encoded as
``[re, im]`` arrays. Everything here is post-run diagnostic reporting and
never feeds controller decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.qse_spectra.core import QSEResult, expect_qse_observable
from pipelines.qse_spectra.spectral_functions import (
    BroadeningKernelConfig,
    SpectralGrid,
    evaluate_broadening_kernel,
)

CONDUCTIVITY_RESPONSE_SCHEMA_VERSION = "qse_conductivity_response_v1"
_PEIERLS_POLICY_DEFAULT = "standard_hh_1d_charge_peierls"
_CONTACT_POLICY_DEFAULT = "contact_record_only_no_drude_delta"
_REGULAR_POLICY_NAME = "positive_frequency_paramagnetic_sjj_over_omega"
_ZERO_SOURCE_NORM_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class ConductivityChannel:
    """One current/contact channel request keyed by transition-observable names."""

    current_label: str
    contact_label: str | None = None
    channel_kind: str = "longitudinal_charge"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.current_label).strip():
            raise ValueError("ConductivityChannel current_label must be non-empty.")


def parse_conductivity_channel_spec(spec: str, *, metadata: Mapping[str, Any] | None = None) -> ConductivityChannel:
    """Parse ``current[:contact[:kind]]`` into a :class:`ConductivityChannel`."""

    parts = [part.strip() for part in str(spec).split(":")]
    if not parts or not parts[0]:
        raise ValueError(f"Invalid conductivity channel spec {spec!r}; expected current[:contact[:kind]].")
    if len(parts) > 3:
        raise ValueError(f"Invalid conductivity channel spec {spec!r}; too many ':' fields.")
    contact = parts[1] if len(parts) >= 2 and parts[1] else None
    kind = parts[2] if len(parts) == 3 and parts[2] else "longitudinal_charge"
    return ConductivityChannel(
        current_label=parts[0],
        contact_label=contact,
        channel_kind=kind,
        metadata=dict(metadata or {}),
    )


def parse_conductivity_channel_specs(
    specs: Sequence[str],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[ConductivityChannel, ...]:
    return tuple(parse_conductivity_channel_spec(spec, metadata=metadata) for spec in specs)


def _complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def _observable_records_by_name(result: QSEResult) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for record in result.transition_observables:
        name = str(record.observable.name)
        if name not in records:
            records[name] = record
    return records


def build_conductivity_response_payload(
    result: QSEResult,
    *,
    grid: SpectralGrid,
    kernel_config: BroadeningKernelConfig,
    channels: Sequence[ConductivityChannel],
    prepared_state: np.ndarray,
    config: Any = None,
    omega_floor: float = 1.0e-12,
    peierls_policy: str = _PEIERLS_POLICY_DEFAULT,
    contact_policy: str = _CONTACT_POLICY_DEFAULT,
) -> dict[str, Any]:
    """Build the additive ``qse_conductivity_response_v1`` manifest payload."""

    floor = float(omega_floor)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("conductivity omega_floor must be finite and positive.")
    if str(contact_policy) != _CONTACT_POLICY_DEFAULT:
        raise ValueError(
            f"conductivity contact policy must be {_CONTACT_POLICY_DEFAULT!r}; got {contact_policy!r}."
        )

    records = _observable_records_by_name(result)
    eigenvalues = np.asarray(result.eigenvalues, dtype=float).reshape(-1)
    ground_energy = float(eigenvalues[0]) if eigenvalues.size else 0.0
    omegas = eigenvalues - ground_energy
    grid_values = grid.values()

    observable_labels: list[str] = []
    channel_payloads: list[dict[str, Any]] = []
    for channel in channels:
        current_label = str(channel.current_label)
        current_record = records.get(current_label)
        if current_record is None:
            raise ValueError(
                f"conductivity channel current label {current_label!r} does not match any "
                "computed transition observable."
            )
        if current_label not in observable_labels:
            observable_labels.append(current_label)

        amplitudes = np.asarray(current_record.transition_amplitudes, dtype=complex).reshape(-1)
        strengths = np.asarray(current_record.transition_strengths, dtype=float).reshape(-1)
        count = min(int(eigenvalues.size), int(amplitudes.size), int(strengths.size))
        source_norm = float(math.sqrt(max(0.0, float(np.sum(strengths[:count])))))
        zero_source = source_norm <= _ZERO_SOURCE_NORM_TOLERANCE

        roots: list[dict[str, Any]] = []
        for index in range(count):
            omega_n = float(omegas[index])
            weight = float(strengths[index])
            included = (omega_n > floor) and (weight > 0.0)
            roots.append(
                {
                    "state_index": int(index),
                    "omega": omega_n,
                    "current_amplitude": _complex_pair(complex(amplitudes[index])),
                    "current_weight": [weight, 0.0],
                    "current_strength": weight,
                    "included_in_regular_conductivity_sum": bool(included),
                }
            )

        paramagnetic = np.zeros_like(grid_values)
        regular = np.zeros_like(grid_values)
        for index in range(count):
            weight = float(strengths[index])
            if weight == 0.0:
                continue
            kernel_values = np.asarray(
                evaluate_broadening_kernel(grid_values - float(omegas[index]), kernel_config),
                dtype=float,
            )
            paramagnetic += weight * kernel_values
            if float(omegas[index]) > floor:
                regular += weight * kernel_values
        with np.errstate(divide="ignore", invalid="ignore"):
            regular = np.where(grid_values > floor, math.pi * regular / grid_values, 0.0)

        if channel.contact_label is not None:
            contact_label = str(channel.contact_label)
            contact_record = records.get(contact_label)
            if contact_record is None:
                raise ValueError(
                    f"conductivity channel contact label {contact_label!r} does not match any "
                    "computed transition observable."
                )
            if contact_label not in observable_labels:
                observable_labels.append(contact_label)
            expectation = complex(
                expect_qse_observable(contact_record.observable, prepared_state, config=config)
            )
            contact_term: dict[str, Any] = {
                "status": "evaluated",
                "label": contact_label,
                "expectation": _complex_pair(expectation),
            }
        else:
            contact_term = {"status": "not_supplied"}

        channel_payloads.append(
            {
                "current_label": current_label,
                "contact_label": channel.contact_label,
                "channel_kind": str(channel.channel_kind),
                "metadata": dict(channel.metadata),
                "current_source": {
                    "status": "evaluated",
                    "source_norm": source_norm,
                    "zero_current_source": bool(zero_source),
                },
                "contact_term": contact_term,
                "drude_weight": {
                    "status": "not_evaluated",
                    "reason": _CONTACT_POLICY_DEFAULT,
                },
                "roots": roots,
                "paramagnetic_current_response": {
                    "values": [[float(value), 0.0] for value in paramagnetic]
                },
                "regular_conductivity": {"values": [[float(value), 0.0] for value in regular]},
            }
        )

    return {
        "schema_version": CONDUCTIVITY_RESPONSE_SCHEMA_VERSION,
        "policy": "diagnostic_only_current_response_postprocessing",
        "response_kind": "conductivity_current",
        "complex_scalar_encoding": "array_real_imag",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "contact_policy": {
            "name": str(contact_policy),
            "combines_contact_into_drude_delta": False,
        },
        "peierls_policy": {"name": str(peierls_policy)},
        "regular_conductivity_policy": {
            "name": _REGULAR_POLICY_NAME,
            "zero_or_negative_frequency_handling": "reported_as_zero_at_or_below_omega_floor",
            "omega_floor": floor,
        },
        "frequency_grid": grid.to_manifest(),
        "kernel": {"kernel": str(kernel_config.kernel), "eta": float(kernel_config.eta)},
        "observables": [{"label": label} for label in observable_labels],
        "channels": channel_payloads,
    }
