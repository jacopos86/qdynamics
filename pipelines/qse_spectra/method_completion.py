"""Completion boundary for Paper-III QSE construction routes.

The canonical selected-QSE route is fixed-pool and never starts another
eigensolver. Its terminal status is bound to the actual exchange-maintained
support, not copied from the pre-exchange acquisition receipt. Post-run exact
spectrum qualification remains separate: this module accepts no exact roots
or exact errors.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.qse_spectra.core import (
    _apply_basis_element,
    _apply_polynomial_operator,
    _config as _core_config,
    compute_qse_spectra,
    normalize_statevector,
)
from pipelines.qse_spectra.switch_policy import assert_online_only_receipt


COMPLETION_SCHEMA_VERSION = "paper_iii_qse_method_completion_v2"
SUPPORT_CERTIFICATE_SCHEMA_VERSION = "paper_iii_qse_support_certificate_v1"

ROUTE_SELECTED_QSE_FIXED_POOL = "selected_qse_fixed_pool"
ROUTE_DIRECT_DAVIDSON = "direct_davidson"
ROUTE_SELECTED_QSE_DAVIDSON_HYBRID = "selected_qse_then_davidson"

STATUS_TARGET_WINDOW_CERTIFIED = "TARGET_WINDOW_CERTIFIED"
STATUS_POOL_EXHAUSTED = "POOL_EXHAUSTED"
STATUS_TARGET_WINDOW_NOT_CERTIFIED = "TARGET_WINDOW_NOT_CERTIFIED"
SELECTED_QSE_TERMINAL_STATUSES = (
    STATUS_TARGET_WINDOW_CERTIFIED,
    STATUS_POOL_EXHAUSTED,
    STATUS_TARGET_WINDOW_NOT_CERTIFIED,
)


def certify_fixed_pool_support(
    pool: Sequence[Any],
    support_indices: Sequence[int],
    *,
    hamiltonian: Any,
    prepared_state: np.ndarray,
    basis_vector_policy: Any,
    target_root_count: int,
    residual_stop_threshold: float,
    metric_novelty_floor: float = 1.0e-12,
    qse_config: Any = None,
) -> dict[str, Any]:
    """Certify the actual terminal support using online quantities only.

    Exchange may replace records after the acquisition selector emits its
    terminal receipt. Its Ky-Fan/cost/rank/conditioning transaction does not
    transfer the selector's residual/window certificate. This check rebuilds
    the supported-overlap Ritz frame of the final support, evaluates full-state
    Ritz residuals, and tests window pressure against every novel unused pool
    record. Exact roots and post-run errors are neither accepted nor read.
    """

    indices = tuple(int(index) for index in support_indices)
    if not indices:
        raise ValueError("support certification requires a nonempty support.")
    if len(indices) != len(set(indices)):
        raise ValueError("support certification requires unique indices.")
    roots = int(target_root_count)
    threshold = float(residual_stop_threshold)
    novelty_floor = float(metric_novelty_floor)
    if roots <= 0:
        raise ValueError("target_root_count must be positive.")
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("residual_stop_threshold must be finite and positive.")
    if not math.isfinite(novelty_floor) or novelty_floor < 0.0:
        raise ValueError("metric_novelty_floor must be finite and non-negative.")

    pool_tuple = tuple(pool)
    if any(index < 0 or index >= len(pool_tuple) for index in indices):
        raise ValueError("support index lies outside the fixed pool.")
    cfg = _core_config(qse_config)
    psi, _norm, nq = normalize_statevector(
        np.asarray(prepared_state, dtype=complex).reshape(-1)
    )
    result = compute_qse_spectra(
        hamiltonian,
        psi,
        tuple(pool_tuple[index] for index in indices),
        config=cfg,
        basis_vector_policy=basis_vector_policy,
    )
    phi = np.column_stack(
        tuple(
            np.asarray(vector, dtype=complex).reshape(-1)
            for vector in result.matrices.basis_matrix_vectors
        )
    )

    # Build the physical retained frame from the same supported-overlap
    # eigensystem used by the QSE solve. These units are the novelty frame.
    overlap_values, overlap_vectors = np.linalg.eigh(
        np.asarray(result.matrices.overlap, dtype=complex)
    )
    retained_units: list[np.ndarray] = []
    for position in result.retained_overlap_indices:
        value = float(overlap_values[int(position)])
        if value <= 0.0:
            continue
        retained_units.append(
            np.asarray(
                phi @ overlap_vectors[:, int(position)] / math.sqrt(value),
                dtype=complex,
            ).reshape(-1)
        )

    cache: dict[str, Any] = {}

    def _project_reference(vector: np.ndarray) -> np.ndarray:
        projected = np.asarray(vector, dtype=complex).reshape(-1)
        if str(getattr(basis_vector_policy, "reference_projection", "none")) == "q0":
            projected = projected - complex(np.vdot(psi, projected)) * psi
        return projected

    root_count = min(roots, int(result.eigenvalues.size))
    residual_norms: list[float] = []
    for root in range(root_count):
        state = np.asarray(
            phi @ np.asarray(result.eigenvectors_basis[:, root], dtype=complex),
            dtype=complex,
        ).reshape(-1)
        h_state = np.asarray(
            _apply_polynomial_operator(
                hamiltonian,
                state,
                nq=int(nq),
                name="hamiltonian",
                config=cfg,
                pauli_action_cache=cache,
            ),
            dtype=complex,
        ).reshape(-1)
        residual = _project_reference(
            h_state - float(result.eigenvalues[root]) * state
        )
        residual_norms.append(float(np.linalg.norm(residual)))
    max_residual = max(residual_norms, default=None)
    residual_below_threshold = bool(
        max_residual is not None
        and math.isfinite(max_residual)
        and max_residual < threshold
    )

    window_top = (
        float(result.eigenvalues[roots - 1])
        if int(result.eigenvalues.size) >= roots
        else None
    )
    selected = set(indices)
    novel_unused = 0
    threatening: list[int] = []
    for candidate_index, element in enumerate(pool_tuple):
        if candidate_index in selected:
            continue
        image = _project_reference(
            _apply_basis_element(
                element,
                psi,
                nq=int(nq),
                config=cfg,
                pauli_action_cache=cache,
            )
        )
        norm_sq = float(np.vdot(image, image).real)
        # Match the selector's supported-image admissibility exactly. Under
        # Q0 projection the identity image is mathematically zero, but a
        # literal ``> 0`` test resurrects normalization roundoff as a novel
        # low-energy direction and falsely reopens the target window.
        if norm_sq <= float(cfg.overlap_absolute_cutoff):
            continue
        projected = image.copy()
        # Must survive verbatim: two-pass reorthogonalized Gram-Schmidt.
        for _pass in range(2):
            for unit in retained_units:
                projected = projected - complex(np.vdot(unit, projected)) * unit
        projected_norm_sq = float(np.vdot(projected, projected).real)
        novelty = max(0.0, projected_norm_sq / norm_sq)
        if novelty < novelty_floor or projected_norm_sq <= 0.0:
            continue
        novel_unused += 1
        if window_top is None:
            continue
        unit = projected / math.sqrt(projected_norm_sq)
        h_unit = np.asarray(
            _apply_polynomial_operator(
                hamiltonian,
                unit,
                nq=int(nq),
                name="hamiltonian",
                config=cfg,
                pauli_action_cache=cache,
            ),
            dtype=complex,
        ).reshape(-1)
        # Must survive: h11 is the window-closure test's candidate Ritz value.
        h11 = float(complex(np.vdot(unit, h_unit)).real)
        if h11 < float(window_top) + threshold:
            threatening.append(int(candidate_index))

    retained_rank = int(result.retained_rank)
    window_pressure = bool(threatening)
    window_closed = bool(retained_rank >= roots and not window_pressure)
    certified = bool(
        retained_rank >= roots and residual_below_threshold and window_closed
    )
    return {
        "schema_version": SUPPORT_CERTIFICATE_SCHEMA_VERSION,
        "support_indices": list(indices),
        "support_size": len(indices),
        "target_root_count": roots,
        "retained_frame_size": retained_rank,
        "root_residual_norms": residual_norms,
        "max_target_residual_norm": max_residual,
        "residual_stop_threshold": threshold,
        "residual_below_threshold": residual_below_threshold,
        "window_top": window_top,
        "window_margin": threshold,
        "window_pressure": window_pressure,
        "window_closed": window_closed,
        "novel_unused_candidate_count": novel_unused,
        "threatening_candidate_indices": threatening,
        "metric_novelty_floor": novelty_floor,
        "target_window_certified": certified,
        "uses_exact_reference": False,
        "online_only": True,
    }


def classify_selected_qse_completion(
    record_stop: Mapping[str, Any],
    *,
    trace_terminal_record_count: int,
    selector_terminal_record_count: int,
    target_root_count: int,
    trace_reached_selector_terminal: bool,
    terminal_support_certificate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Classify fixed-pool selected QSE from the actual terminal support."""

    assert_online_only_receipt(record_stop)
    trace_count = int(trace_terminal_record_count)
    selector_count = int(selector_terminal_record_count)
    roots = int(target_root_count)
    if trace_count < 0 or selector_count < 0:
        raise ValueError("record counts must be non-negative.")
    if trace_count > selector_count:
        raise ValueError("trace_terminal_record_count exceeds selector count.")
    if trace_reached_selector_terminal and trace_count != selector_count:
        raise ValueError(
            "a trace marked selector-terminal must have equal trace and selector counts."
        )
    if roots <= 0:
        raise ValueError("target_root_count must be positive.")

    stop_reason = str(record_stop.get("stop_reason", ""))
    certificate = (
        None if terminal_support_certificate is None else dict(terminal_support_certificate)
    )
    if certificate is not None:
        if certificate.get("schema_version") != SUPPORT_CERTIFICATE_SCHEMA_VERSION:
            raise ValueError("unexpected terminal-support certificate schema.")
        if certificate.get("uses_exact_reference") is not False:
            raise ValueError("terminal-support certificate must not use exact references.")
        if certificate.get("online_only") is not True:
            raise ValueError("terminal-support certificate must be online-only.")
        if int(certificate.get("target_root_count", -1)) != roots:
            raise ValueError("terminal-support certificate targets the wrong root count.")
        support_indices = tuple(int(i) for i in certificate.get("support_indices", ()))
        if int(certificate.get("support_size", -1)) != len(support_indices):
            raise ValueError("terminal-support certificate has inconsistent support size.")

    final_window_closed = (
        None if certificate is None else bool(certificate.get("window_closed") is True)
    )
    final_window_pressure = (
        None if certificate is None else bool(certificate.get("window_pressure") is True)
    )
    retained_frame_size = (
        None if certificate is None else int(certificate.get("retained_frame_size", 0))
    )
    raw_residual = (
        None if certificate is None else certificate.get("max_target_residual_norm")
    )
    raw_threshold = (
        None if certificate is None else certificate.get("residual_stop_threshold")
    )
    final_residual = None if raw_residual is None else float(raw_residual)
    residual_threshold = None if raw_threshold is None else float(raw_threshold)
    residual_below_threshold = bool(
        final_residual is not None
        and residual_threshold is not None
        and math.isfinite(final_residual)
        and math.isfinite(residual_threshold)
        and residual_threshold > 0.0
        and final_residual < residual_threshold
    )
    has_online_certificate = bool(
        trace_count > 0
        and residual_below_threshold
        and final_window_closed is True
        and final_window_pressure is False
        and retained_frame_size is not None
        and retained_frame_size >= roots
    )
    if certificate is not None and bool(
        certificate.get("target_window_certified", False)
    ) != has_online_certificate:
        raise ValueError("terminal-support certificate carries a contradictory verdict.")

    if has_online_certificate:
        status = STATUS_TARGET_WINDOW_CERTIFIED
        reason = "terminal_support_residual_and_window_certificate"
    elif stop_reason == "pool_exhausted" and trace_reached_selector_terminal:
        status = STATUS_POOL_EXHAUSTED
        reason = "record_pool_exhausted"
    elif not trace_reached_selector_terminal:
        status = STATUS_TARGET_WINDOW_NOT_CERTIFIED
        reason = "dimension_cap_before_selector_terminal"
    else:
        status = STATUS_TARGET_WINDOW_NOT_CERTIFIED
        reason = "terminal_support_did_not_certify_target_window"

    return {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "completion_route": ROUTE_SELECTED_QSE_FIXED_POOL,
        "terminal_status": status,
        "termination_reason": reason,
        "target_window_certified": has_online_certificate,
        "davidson_invoked": False,
        "online_only": True,
        "record_stop_reason": stop_reason,
        "trace_terminal_record_count": trace_count,
        "selector_terminal_record_count": selector_count,
        "target_root_count": roots,
        "trace_reached_selector_terminal": bool(trace_reached_selector_terminal),
        "terminal_online_metrics_available": certificate is not None,
        "terminal_support_certificate": certificate,
        "final_window_closed": final_window_closed,
        "final_window_pressure": final_window_pressure,
        "final_retained_frame_size": retained_frame_size,
        "final_max_target_residual_norm": final_residual,
        "residual_stop_threshold": residual_threshold,
        "residual_below_threshold": residual_below_threshold,
    }


__all__ = [
    "COMPLETION_SCHEMA_VERSION",
    "ROUTE_DIRECT_DAVIDSON",
    "ROUTE_SELECTED_QSE_DAVIDSON_HYBRID",
    "ROUTE_SELECTED_QSE_FIXED_POOL",
    "SELECTED_QSE_TERMINAL_STATUSES",
    "STATUS_POOL_EXHAUSTED",
    "STATUS_TARGET_WINDOW_CERTIFIED",
    "STATUS_TARGET_WINDOW_NOT_CERTIFIED",
    "SUPPORT_CERTIFICATE_SCHEMA_VERSION",
    "certify_fixed_pool_support",
    "classify_selected_qse_completion",
]
