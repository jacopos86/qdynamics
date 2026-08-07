#!/usr/bin/env python3
"""Deterministic benchmark shot-proxy schema helpers.

This module owns the comparator-compatible total-shot proxy field names used by
Paper-I static benchmark rows.  The proxy is deterministic accounting metadata,
not a physical hardware shot allocation.
"""

from __future__ import annotations

import math
from typing import Any

DETERMINISTIC_SHOT_PROXY_STATUS = "deterministic_proxy_not_physical_shots"
DETERMINISTIC_SHOT_PROXY_FORMULA = (
    "shots_total = shots_per_pauli_term_proxy * hamiltonian_pauli_term_count * "
    "(energy_eval_count_proxy + gradient_operator_probe_count_proxy + metric_operator_probe_count_proxy)"
)
DETERMINISTIC_SHOT_PROXY_NOTE = "Benchmark-table deterministic proxy only; not a hardware shot allocation."


def _strict_nonnegative_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed) or parsed < 0.0:
        return None
    rounded = int(round(parsed))
    if abs(parsed - float(rounded)) > 1e-9:
        return None
    return rounded


def build_deterministic_shot_proxy_fields(
    *,
    hamiltonian_pauli_term_count: int | float,
    pool_term_count: int | float | None = None,
    energy_eval_count: int | float | None,
    gradient_scan_count: int | float = 0,
    gradient_operator_probe_count: int | float,
    metric_operator_probe_count: int | float,
    shots_per_pauli_term_proxy: int | float,
    comparator_legacy_coercion: bool = False,
) -> dict[str, Any]:
    """Return comparator-compatible deterministic total-shot proxy fields.

    ``comparator_legacy_coercion=True`` preserves the historical comparator
    behavior: counts are coerced with ``int``/``max`` and energy evaluations are
    clamped to at least one.  Strict callers should use the default ``False``;
    invalid or non-integer-like inputs then raise ``ValueError``.
    """

    if comparator_legacy_coercion:
        h_count = max(0, int(hamiltonian_pauli_term_count))
        pool_count = max(0, int(pool_term_count or 0))
        energy_count = max(1, int(energy_eval_count or 0))
        grad_scans = max(0, int(gradient_scan_count))
        grad_probes = max(0, int(gradient_operator_probe_count))
        metric_probes = max(0, int(metric_operator_probe_count))
        shots_per_term = max(0, int(shots_per_pauli_term_proxy))
    else:
        parsed = {
            "hamiltonian_pauli_term_count": _strict_nonnegative_int(hamiltonian_pauli_term_count),
            "pool_term_count": _strict_nonnegative_int(0 if pool_term_count is None else pool_term_count),
            "energy_eval_count_proxy": _strict_nonnegative_int(energy_eval_count),
            "gradient_scan_count_proxy": _strict_nonnegative_int(gradient_scan_count),
            "gradient_operator_probe_count_proxy": _strict_nonnegative_int(gradient_operator_probe_count),
            "metric_operator_probe_count_proxy": _strict_nonnegative_int(metric_operator_probe_count),
            "shots_per_pauli_term_proxy": _strict_nonnegative_int(shots_per_pauli_term_proxy),
        }
        invalid = [key for key, value in parsed.items() if value is None]
        if invalid:
            raise ValueError(f"invalid deterministic shot-proxy inputs: {', '.join(invalid)}")
        h_count = int(parsed["hamiltonian_pauli_term_count"])
        pool_count = int(parsed["pool_term_count"])
        energy_count = int(parsed["energy_eval_count_proxy"])
        grad_scans = int(parsed["gradient_scan_count_proxy"])
        grad_probes = int(parsed["gradient_operator_probe_count_proxy"])
        metric_probes = int(parsed["metric_operator_probe_count_proxy"])
        shots_per_term = int(parsed["shots_per_pauli_term_proxy"])
    shots_total = int(shots_per_term * h_count * (energy_count + grad_probes + metric_probes))
    return {
        "shots_total": shots_total,
        "static_shot_estimate_status": DETERMINISTIC_SHOT_PROXY_STATUS,
        "shot_proxy_formula": DETERMINISTIC_SHOT_PROXY_FORMULA,
        "shot_proxy_note": DETERMINISTIC_SHOT_PROXY_NOTE,
        "shots_per_pauli_term_proxy": shots_per_term,
        "hamiltonian_pauli_term_count": h_count,
        "pool_term_count": pool_count,
        "energy_eval_count_proxy": energy_count,
        "gradient_scan_count_proxy": grad_scans,
        "gradient_operator_probe_count_proxy": grad_probes,
        "metric_operator_probe_count_proxy": metric_probes,
    }
