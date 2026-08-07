#!/usr/bin/env python3
"""Reusable handoff-state bundle writer for hardcoded HH workflows."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class HandoffStateBundleConfig:
    L: int
    t: float
    U: float
    dv: float
    omega0: float
    g_ep: float
    n_ph_max: int
    boson_encoding: str
    ordering: str
    boundary: str
    sector_n_up: int
    sector_n_dn: int


def build_handoff_settings_manifest(cfg: HandoffStateBundleConfig) -> dict[str, Any]:
    return {
        "L": int(cfg.L),
        "problem": "hh",
        "t": float(cfg.t),
        "u": float(cfg.U),
        "dv": float(cfg.dv),
        "omega0": float(cfg.omega0),
        "g_ep": float(cfg.g_ep),
        "n_ph_max": int(cfg.n_ph_max),
        "boson_encoding": str(cfg.boson_encoding),
        "ordering": str(cfg.ordering),
        "boundary": str(cfg.boundary),
        "sector_n_up": int(cfg.sector_n_up),
        "sector_n_dn": int(cfg.sector_n_dn),
    }


def _statevector_to_amplitudes_qn_to_q0(
    psi_state: np.ndarray,
    *,
    cutoff: float,
) -> dict[str, dict[str, float]]:
    psi = np.asarray(psi_state, dtype=complex).reshape(-1)
    nq_total = int(round(math.log2(int(psi.size))))
    out: dict[str, dict[str, float]] = {}
    for idx, amp in enumerate(psi):
        if abs(amp) <= float(cutoff):
            continue
        out[format(idx, f"0{nq_total}b")] = {
            "re": float(np.real(amp)),
            "im": float(np.imag(amp)),
        }
    return out


_VALID_HANDOFF_STATE_KINDS = {"prepared_state", "reference_state"}


def build_statevector_manifest(
    *,
    psi_state: np.ndarray,
    source: str,
    handoff_state_kind: str | None = None,
    amplitude_cutoff: float = 1e-14,
) -> dict[str, Any]:
    psi = np.asarray(psi_state, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(psi))
    if norm <= 0.0:
        raise ValueError("psi_state must be non-zero.")
    psi = psi / norm
    nq_total = int(round(math.log2(int(psi.size))))
    kind = None if handoff_state_kind is None else str(handoff_state_kind).strip()
    if kind is not None and kind not in _VALID_HANDOFF_STATE_KINDS:
        raise ValueError(
            f"handoff_state_kind must be one of {sorted(_VALID_HANDOFF_STATE_KINDS)} when provided."
        )
    manifest: dict[str, Any] = {
        "source": str(source),
        "nq_total": int(nq_total),
        "amplitudes_qn_to_q0": _statevector_to_amplitudes_qn_to_q0(
            psi,
            cutoff=float(amplitude_cutoff),
        ),
        "amplitude_cutoff": float(amplitude_cutoff),
        "norm": float(np.linalg.norm(psi)),
    }
    if kind is not None:
        manifest["handoff_state_kind"] = kind
    return manifest


def _json_safe_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_tree(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe_tree(value.tolist())
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else str(value)
    return str(value)


def write_handoff_state_bundle(
    *,
    path: Path,
    psi_state: np.ndarray,
    cfg: HandoffStateBundleConfig,
    source: str,
    exact_energy: float,
    energy: float,
    delta_E_abs: float,
    relative_error_abs: float,
    meta: dict[str, Any] | None = None,
    adapt_operators: list[str] | None = None,
    adapt_optimal_point: list[float] | None = None,
    adapt_logical_optimal_point: list[float] | None = None,
    adapt_parameterization: dict[str, Any] | None = None,
    adapt_logical_num_parameters: int | None = None,
    adapt_pool_type: str | None = None,
    handoff_state_kind: str | None = None,
    continuation_mode: str | None = None,
    continuation_scaffold: dict[str, Any] | None = None,
    continuation_details: Mapping[str, Any] | None = None,
    optimizer_memory: dict[str, Any] | None = None,
    selected_generator_metadata: list[dict[str, Any]] | None = None,
    generator_split_events: list[dict[str, Any]] | None = None,
    motif_library: dict[str, Any] | None = None,
    motif_usage: dict[str, Any] | None = None,
    symmetry_mitigation: dict[str, Any] | None = None,
    rescue_history: list[dict[str, Any]] | None = None,
    prune_summary: dict[str, Any] | None = None,
    pre_prune_scaffold: dict[str, Any] | None = None,
    replay_contract_hint: dict[str, Any] | None = None,
    ansatz_input_state: np.ndarray | None = None,
    ansatz_input_state_source: str | None = None,
    ansatz_input_state_handoff_state_kind: str | None = None,
    amplitude_cutoff: float = 1e-14,
) -> None:
    """Write an adapt_json-compatible HH handoff bundle."""

    psi = np.asarray(psi_state, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(psi))
    if norm <= 0.0:
        raise ValueError("psi_state must be non-zero.")
    psi = psi / norm

    adapt_vqe_block: dict[str, Any] = {
        "energy": float(energy),
        "abs_delta_e": float(delta_E_abs),
        "relative_error_abs": float(relative_error_abs),
    }
    if adapt_operators is not None and adapt_optimal_point is not None:
        adapt_vqe_block["operators"] = list(adapt_operators)
        adapt_vqe_block["optimal_point"] = [float(x) for x in adapt_optimal_point]
        adapt_vqe_block["ansatz_depth"] = int(len(adapt_operators))
        adapt_vqe_block["num_parameters"] = int(len(adapt_optimal_point))
        if adapt_logical_optimal_point is not None:
            adapt_vqe_block["logical_optimal_point"] = [float(x) for x in adapt_logical_optimal_point]
        if adapt_logical_num_parameters is not None:
            adapt_vqe_block["logical_num_parameters"] = int(adapt_logical_num_parameters)
        if adapt_parameterization is not None:
            adapt_vqe_block["parameterization"] = dict(adapt_parameterization)
    if adapt_pool_type is not None:
        adapt_vqe_block["pool_type"] = str(adapt_pool_type)
    if pre_prune_scaffold is not None:
        adapt_vqe_block["pre_prune_scaffold"] = dict(pre_prune_scaffold)
    if prune_summary is not None:
        adapt_vqe_block["prune_summary"] = dict(prune_summary)

    payload: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "settings": build_handoff_settings_manifest(cfg),
        "adapt_vqe": adapt_vqe_block,
        "initial_state": build_statevector_manifest(
            psi_state=psi,
            source=str(source),
            handoff_state_kind=handoff_state_kind,
            amplitude_cutoff=float(amplitude_cutoff),
        ),
        "exact": {
            "E_exact_sector": float(exact_energy),
        },
    }
    if ansatz_input_state is not None:
        if ansatz_input_state_source is None or str(ansatz_input_state_source).strip() == "":
            raise ValueError("ansatz_input_state_source is required when ansatz_input_state is provided.")
        payload["ansatz_input_state"] = build_statevector_manifest(
            psi_state=np.asarray(ansatz_input_state, dtype=complex).reshape(-1),
            source=str(ansatz_input_state_source),
            handoff_state_kind=ansatz_input_state_handoff_state_kind,
            amplitude_cutoff=float(amplitude_cutoff),
        )
    continuation_block: dict[str, Any] = (
        dict(_json_safe_tree(continuation_details))
        if isinstance(continuation_details, Mapping)
        else {}
    )
    if continuation_mode is not None:
        continuation_block["mode"] = str(continuation_mode)
    if continuation_scaffold is not None:
        continuation_block["scaffold"] = dict(continuation_scaffold)
    if optimizer_memory is not None:
        continuation_block["optimizer_memory"] = dict(optimizer_memory)
    if selected_generator_metadata is not None:
        continuation_block["selected_generator_metadata"] = [dict(x) for x in selected_generator_metadata]
    if generator_split_events is not None:
        continuation_block["generator_split_events"] = [dict(x) for x in generator_split_events]
    if motif_library is not None:
        continuation_block["motif_library"] = dict(motif_library)
    if motif_usage is not None:
        continuation_block["motif_usage"] = dict(motif_usage)
    if symmetry_mitigation is not None:
        continuation_block["symmetry_mitigation"] = dict(symmetry_mitigation)
    if rescue_history is not None:
        continuation_block["rescue_history"] = [dict(x) for x in rescue_history]
    if replay_contract_hint is not None:
        continuation_block["replay_contract_hint"] = dict(replay_contract_hint)
    if continuation_block:
        payload["continuation"] = continuation_block
    if isinstance(meta, dict):
        payload["meta"] = dict(meta)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
