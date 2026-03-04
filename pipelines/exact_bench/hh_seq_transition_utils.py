#!/usr/bin/env python3
"""Shared helpers for HH sequential stage-transition robustness runs.

This module is intentionally internal to exact-benchmark/report workflows.
It does not modify operator-core algebra classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _to_ixyz(label_exyz: str) -> str:
    return (
        str(label_exyz)
        .replace("e", "I")
        .replace("x", "X")
        .replace("y", "Y")
        .replace("z", "Z")
    )


def polynomial_signature(poly: Any, tol: float = 1e-12) -> tuple[tuple[str, float], ...]:
    """Deterministic PauliPolynomial signature for dedup checks."""
    items: list[tuple[str, float]] = []
    for term in poly.return_polynomial():
        coeff = complex(term.p_coeff)
        if abs(coeff) <= float(tol):
            continue
        if abs(coeff.imag) > 1e-10:
            raise ValueError(
                f"Non-negligible imaginary coeff in signature build: {coeff} for {term.pw2strng()}"
            )
        items.append((str(term.pw2strng()), round(float(coeff.real), 12)))
    items.sort()
    return tuple(items)


@dataclass(frozen=True)
class TransitionConfig:
    """Windowed absolute-slope transition policy."""

    window_k: int = 5
    slope_epsilon: float = 5e-5
    patience: int = 3
    min_points_before_switch: int = 8


@dataclass
class TransitionState:
    """Mutable transition state for stage switch checks."""

    cfg: TransitionConfig
    delta_abs_trace: list[float] = field(default_factory=list)
    slope_trace: list[float] = field(default_factory=list)
    plateau_hits: int = 0
    switch_triggered: bool = False
    trigger_index: int | None = None
    trigger_reason: str = ""


def _window_slope(values: Sequence[float]) -> float:
    """Least-squares slope over equally spaced x in [0, ..., k-1]."""
    y = np.asarray(values, dtype=float).reshape(-1)
    if y.size < 2:
        return float("nan")
    x = np.arange(int(y.size), dtype=float)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    den = float(np.sum((x - x_mean) ** 2))
    if den <= 0.0:
        return 0.0
    num = float(np.sum((x - x_mean) * (y - y_mean)))
    return float(num / den)


def update_transition_state(
    state: TransitionState,
    *,
    delta_abs: float,
) -> dict[str, Any]:
    """Push one |DeltaE| point and update switch decision."""
    val = float(abs(delta_abs))
    state.delta_abs_trace.append(val)

    k = max(2, int(state.cfg.window_k))
    slope = float("nan")
    if len(state.delta_abs_trace) >= k:
        slope = _window_slope(state.delta_abs_trace[-k:])
        state.slope_trace.append(float(slope))

        if abs(float(slope)) <= float(abs(state.cfg.slope_epsilon)):
            state.plateau_hits += 1
        else:
            state.plateau_hits = 0

    if (
        (not state.switch_triggered)
        and len(state.delta_abs_trace) >= int(state.cfg.min_points_before_switch)
        and state.plateau_hits >= int(state.cfg.patience)
    ):
        state.switch_triggered = True
        state.trigger_index = int(len(state.delta_abs_trace) - 1)
        state.trigger_reason = (
            "windowed_abs_slope_plateau: "
            f"|slope|={abs(float(slope)):.6e} <= eps={abs(float(state.cfg.slope_epsilon)):.6e}, "
            f"hits={int(state.plateau_hits)}/{int(state.cfg.patience)}, "
            f"window_k={int(state.cfg.window_k)}"
        )

    return {
        "delta_abs": float(val),
        "slope": (None if not np.isfinite(float(slope)) else float(slope)),
        "window_k": int(state.cfg.window_k),
        "slope_epsilon": float(state.cfg.slope_epsilon),
        "patience": int(state.cfg.patience),
        "plateau_hits": int(state.plateau_hits),
        "min_points_before_switch": int(state.cfg.min_points_before_switch),
        "switch_triggered": bool(state.switch_triggered),
        "trigger_index": (
            None if state.trigger_index is None else int(state.trigger_index)
        ),
        "trigger_reason": str(state.trigger_reason),
    }


def summarize_transition(state: TransitionState) -> dict[str, Any]:
    return {
        "policy": {
            "window_k": int(state.cfg.window_k),
            "slope_epsilon": float(state.cfg.slope_epsilon),
            "patience": int(state.cfg.patience),
            "min_points_before_switch": int(state.cfg.min_points_before_switch),
        },
        "delta_abs_trace": [float(x) for x in state.delta_abs_trace],
        "slope_trace": [float(x) for x in state.slope_trace],
        "plateau_hits": int(state.plateau_hits),
        "switch_triggered": bool(state.switch_triggered),
        "trigger_index": (
            None if state.trigger_index is None else int(state.trigger_index)
        ),
        "trigger_reason": str(state.trigger_reason),
    }


def build_pool_b_strict_union(
    *,
    uccsd_ops: Sequence[AnsatzTerm],
    hva_ops: Sequence[AnsatzTerm],
    paop_full_ops: Sequence[AnsatzTerm],
    signature_tol: float = 1e-12,
) -> tuple[list[AnsatzTerm], dict[str, Any], dict[tuple[tuple[str, float], ...], dict[str, bool]]]:
    """Build strict Pool B = UCCSD_lifted + HVA + PAOP_FULL (deduplicated).

    Deterministic order is preserved by family then source-local order.
    """
    ordered_components: list[tuple[str, list[AnsatzTerm]]] = [
        ("uccsd", list(uccsd_ops)),
        ("hva", list(hva_ops)),
        ("paop_full", list(paop_full_ops)),
    ]

    source_by_sig: dict[tuple[tuple[str, float], ...], dict[str, bool]] = {}
    raw_sizes: dict[str, int] = {}
    for fam, ops in ordered_components:
        raw_sizes[fam] = int(len(ops))
        for op in ops:
            sig = polynomial_signature(op.polynomial, tol=float(signature_tol))
            source_by_sig.setdefault(
                sig,
                {"uccsd": False, "hva": False, "paop_full": False},
            )[fam] = True

    dedup: list[AnsatzTerm] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for _fam, ops in ordered_components:
        for op in ops:
            sig = polynomial_signature(op.polynomial, tol=float(signature_tol))
            if sig in seen:
                continue
            seen.add(sig)
            dedup.append(op)

    presence_counts = {"uccsd": 0, "hva": 0, "paop_full": 0}
    overlap_count = 0
    family_overlap_matrix = {
        "uccsd_hva": 0,
        "uccsd_paop_full": 0,
        "hva_paop_full": 0,
        "all_three": 0,
    }

    for op in dedup:
        sig = polynomial_signature(op.polynomial, tol=float(signature_tol))
        flags = source_by_sig.get(
            sig,
            {"uccsd": False, "hva": False, "paop_full": False},
        )
        active = [name for name, present in flags.items() if bool(present)]
        for name in active:
            presence_counts[name] += 1
        if len(active) >= 2:
            overlap_count += 1
        if bool(flags.get("uccsd")) and bool(flags.get("hva")):
            family_overlap_matrix["uccsd_hva"] += 1
        if bool(flags.get("uccsd")) and bool(flags.get("paop_full")):
            family_overlap_matrix["uccsd_paop_full"] += 1
        if bool(flags.get("hva")) and bool(flags.get("paop_full")):
            family_overlap_matrix["hva_paop_full"] += 1
        if all(bool(flags.get(x)) for x in ("uccsd", "hva", "paop_full")):
            family_overlap_matrix["all_three"] += 1

    meta = {
        "pool_b_definition": "strict_union(uccsd_lifted, hva, paop_full)",
        "raw_sizes": raw_sizes,
        "dedup_total": int(len(dedup)),
        "dedup_source_presence_counts": {
            "uccsd": int(presence_counts["uccsd"]),
            "hva": int(presence_counts["hva"]),
            "paop_full": int(presence_counts["paop_full"]),
        },
        "overlap_count": int(overlap_count),
        "family_overlap_matrix": {k: int(v) for k, v in family_overlap_matrix.items()},
    }
    return dedup, meta, source_by_sig


def build_time_dependent_sparse_qop(
    *,
    ordered_labels_exyz: Sequence[str],
    static_coeff_map_exyz: dict[str, complex],
    drive_coeff_map_exyz: dict[str, complex] | None = None,
    tol: float = 1e-12,
) -> SparsePauliOp:
    """Build ordered SparsePauliOp for H_static + H_drive(t) in IXYZ labels."""
    drive_map = dict(drive_coeff_map_exyz or {})
    labels: list[str] = list(ordered_labels_exyz)

    # Keep deterministic order: static labels first, then new drive-only labels sorted.
    drive_only = sorted(lbl for lbl in drive_map.keys() if lbl not in set(labels))
    labels.extend(drive_only)

    terms: list[tuple[str, complex]] = []
    for lbl in labels:
        coeff = complex(static_coeff_map_exyz.get(lbl, 0.0 + 0.0j)) + complex(drive_map.get(lbl, 0.0 + 0.0j))
        if abs(coeff) <= float(tol):
            continue
        terms.append((_to_ixyz(lbl), coeff))

    if not terms:
        nq = len(labels[0]) if labels else 1
        terms = [("I" * int(nq), 0.0 + 0.0j)]

    return SparsePauliOp.from_list(terms).simplify(atol=float(tol))


def flatten_coeff_map_real_imag(coeff_map: dict[str, complex]) -> dict[str, dict[str, float]]:
    """JSON-safe coeff map serialization helper."""
    out: dict[str, dict[str, float]] = {}
    for label in sorted(coeff_map):
        coeff = complex(coeff_map[label])
        out[str(label)] = {"re": float(np.real(coeff)), "im": float(np.imag(coeff))}
    return out
