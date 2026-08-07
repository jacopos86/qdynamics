#!/usr/bin/env python3
"""Recover Paper-I HH Table-III prefix fidelities for the active 3x6 rows.

This support script is intentionally narrow.  It reads the current SNAKE row
block from ``MATH/paper_details/Paper_I.tex`` and the parent Geo/append support
CSV referenced by that block, replays/refits the selected prefix rows, computes
dense exact-state fidelity at the same working cutoff, and writes CSV/JSON
sidecars.  It does not edit the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.contracts.problem import ProblemRequest  # noqa: E402
from pipelines.exact_bench.generic_static_adapt_variants import (  # noqa: E402
    STATIC_FULL_META_APPEND_ADAPT_VQE,
    STATIC_GEO_ADAPT_VQE,
    _PoolCandidate,
    _dense_exact_state_fidelity_for_selected,
    _get_config,
    _namespace_from_base_args,
    _normalize_reference_metric_fields,
    _optimize_selected,
    _prepare_selected_state,
    _resolve_context_from_spec,
    _safe_exact_energy,
    build_full_meta_candidate_pool,
    has_scipy_minimize_support,
    _import_scipy_minimize,
)
from pipelines.static_adapt.builders.problem_registry import resolve_problem_context  # noqa: E402
from src.quantum.compiled_polynomial import (  # noqa: E402
    compile_polynomial_action,
    energy_via_one_apply,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial  # noqa: E402
from src.quantum.qubitization_module import PauliTerm  # noqa: E402

REGIME_ORDER = (
    "weak-weak",
    "intermediate-weak",
    "strong-weak",
    "weak-strong",
    "intermediate-strong",
    "strong-strong",
)
SNAKE_BLOCK_MARKER = "BEGIN_MACHINE_READABLE_HH_PHYSICAL_LANE_DUPLICATE_UPDATE_20260708"
DEFAULT_OUT_DIR = REPO_ROOT / "output/pdf/paper_i_hh_tableiii_prefix_fidelity_20260708"
OLD_REPO_ROOT = Path("/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _int_or_none(value: Any) -> int | None:
    x = _finite_float(value)
    return None if x is None else int(round(x))


def _resolve_path(path: str | Path) -> Path:
    p = Path(str(path))
    if p.is_absolute():
        return p
    local = REPO_ROOT / p
    if local.exists():
        return local
    return OLD_REPO_ROOT / p


def _extract_commented_json_block(tex_path: Path, marker: str) -> dict[str, Any]:
    text = tex_path.read_text(encoding="utf-8", errors="ignore")
    start = text.find(marker)
    if start < 0:
        raise RuntimeError(f"marker not found in {tex_path}: {marker}")
    block = text[start:]
    # Extract the first balanced JSON object whose lines are LaTeX comments.
    json_lines: list[str] = []
    depth = 0
    seen_open = False
    for raw in block.splitlines()[1:]:
        if not raw.lstrip().startswith("%"):
            if seen_open:
                break
            continue
        line = raw.lstrip()[1:]
        if line.startswith(" "):
            line = line[1:]
        if not seen_open and "{" not in line:
            continue
        json_lines.append(line)
        depth += line.count("{") - line.count("}")
        seen_open = seen_open or "{" in line
        if seen_open and depth == 0:
            break
    if not json_lines:
        raise RuntimeError(f"no JSON block found after marker {marker}")
    return json.loads("\n".join(json_lines))


def _snake_rows_from_tex(tex_path: Path) -> tuple[list[dict[str, Any]], Path]:
    block = _extract_commented_json_block(tex_path, SNAKE_BLOCK_MARKER)
    changed = block.get("changed_snake_cells")
    if not isinstance(changed, Mapping):
        raise RuntimeError("changed_snake_cells missing from current SNAKE block")
    rows: list[dict[str, Any]] = []
    for regime in REGIME_ORDER:
        cell = changed.get(regime)
        if not isinstance(cell, Mapping):
            raise RuntimeError(f"missing SNAKE cell for {regime}")
        source = _resolve_path(str(cell["source_json"]))
        rows.append(
            {
                "method_key": "snake",
                "method_display": "SNAKE",
                "regime": regime,
                "selected_prefix_k": int(cell["k_pl"]),
                "visible_abs_delta_e": float(cell["abs_delta_e"]),
                "source_json": str(source),
                "source_json_sha256": str(cell.get("source_json_sha256") or ""),
                "N2q": cell.get("N2q"),
                "D2q": cell.get("D2q"),
                "Dc": cell.get("Dc"),
                "S_alg": cell.get("S_alg"),
            }
        )
    support_csv = _resolve_path(str(block["comparator_support_csv"]))
    return rows, support_csv


def _parent_comparator_rows(support_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with support_csv.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            role = str(raw.get("role_key") or "")
            if role not in {"geo_macro_c", "append_macro_c"}:
                continue
            method_display = "Parent Geo-ADAPT" if role == "geo_macro_c" else "Parent append-only ADAPT"
            rows.append(
                {
                    "method_key": "geo_parent" if role == "geo_macro_c" else "append_parent",
                    "method_display": method_display,
                    "role_key": role,
                    "regime": str(raw["regime"]),
                    "selected_prefix_k": int(float(raw["selected_prefix_k"])),
                    "visible_abs_delta_e": float(raw["abs_delta_e"]),
                    "source_json": str(_resolve_path(str(raw["source_json"]))),
                    "source_json_sha256": str(raw.get("source_sha256") or ""),
                    "N2q": raw.get("N2q"),
                    "D2q": raw.get("D2q"),
                    "Dc": raw.get("Dc"),
                    "S_alg": raw.get("S_alg"),
                }
            )
    by_key = {(row["method_key"], row["regime"]): row for row in rows}
    ordered: list[dict[str, Any]] = []
    for method in ("geo_parent", "append_parent"):
        for regime in REGIME_ORDER:
            row = by_key.get((method, regime))
            if row is None:
                raise RuntimeError(f"missing comparator row {method} {regime}")
            ordered.append(row)
    return ordered


def _context_from_settings(settings: Mapping[str, Any]):
    ns = Namespace(
        problem=settings.get("problem", "hh"),
        L=int(settings.get("L", 2)),
        t=float(settings.get("t", 1.0)),
        u=float(settings.get("u", 0.0)),
        dv=float(settings.get("dv", 0.0)),
        omega0=float(settings.get("omega0", 1.0)),
        g_ep=float(settings.get("g_ep", 0.0)),
        n_ph_max=int(settings.get("n_ph_max", 0)),
        boson_encoding=str(settings.get("boson_encoding", "binary")),
        ordering=str(settings.get("ordering", "blocked")),
        boundary=str(settings.get("boundary", "open")),
        include_zero_point=True if settings.get("include_zero_point") in {None, ""} else bool(settings.get("include_zero_point")),
        molecular_problem_json=settings.get("molecular_problem_json"),
        molecular_vibronic_h2_fixture_json=settings.get("molecular_vibronic_h2_fixture_json"),
        molecular_vibronic_h2o_fixture_json=settings.get("molecular_vibronic_h2o_fixture_json"),
        molecular_vibronic_h2o_linear_fd_fixture_json=settings.get("molecular_vibronic_h2o_linear_fd_fixture_json"),
        v_nn=float(settings.get("v_nn", 0.0) or 0.0),
        t_prime=float(settings.get("t_prime", 0.0) or 0.0),
        n_fermions=settings.get("n_fermions"),
    )
    return resolve_problem_context(ProblemRequest.from_namespace(ns))


def _context_from_generic_payload(payload: Mapping[str, Any]):
    spec = payload.get("spec")
    if not isinstance(spec, Mapping) or "base_pipeline_args" not in spec:
        result = payload.get("result") if isinstance(payload.get("result"), Mapping) else payload
        # Last-resort direct namespace from result fields.
        return _context_from_settings(result)
    spec_ns = Namespace(base_pipeline_args=tuple(spec.get("base_pipeline_args") or ()))
    return _resolve_context_from_spec(spec_ns)


def _candidate_from_serialized_terms(
    *,
    label: str,
    terms_raw: Sequence[Mapping[str, Any]],
    construction: str,
    parent_label: str | None = None,
    generator_metadata: Mapping[str, Any] | None = None,
) -> _PoolCandidate:
    terms: list[PauliTerm] = []
    pauli_labels: list[str] = []
    support: set[int] = set()
    for raw in terms_raw:
        pauli = str(raw.get("pauli_exyz") or raw.get("pauli") or "").strip().lower()
        if not pauli:
            raise RuntimeError(f"{label}: serialized term missing pauli_exyz")
        nq = int(raw.get("nq", len(pauli)))
        coeff = complex(float(raw.get("coeff_re", 0.0)), float(raw.get("coeff_im", 0.0)))
        if abs(coeff.imag) > 1e-12:
            raise RuntimeError(f"{label}: complex Pauli coefficient not supported by replay script: {coeff}")
        terms.append(PauliTerm(nq, ps=pauli, pc=float(coeff.real)))
        pauli_labels.append(pauli)
        support.update(idx for idx, char in enumerate(pauli) if char != "e")
    if not terms:
        raise RuntimeError(f"{label}: no serialized terms")
    return _PoolCandidate(
        label=str(label),
        polynomial=PauliPolynomial("JW", terms),
        support=tuple(sorted(support)),
        pauli_labels_exyz=tuple(pauli_labels),
        construction=str(construction),
        parent_label=parent_label,
        runtime_split_mode="shortlist_pauli_children_v1" if parent_label else "off",
        runtime_split_representation="child_set" if parent_label else "parent",
        generator_metadata=dict(generator_metadata or {}),
    )


def _snake_candidate_from_feature(row: Mapping[str, Any], feature_index: int) -> _PoolCandidate:
    features = row.get("selected_feature_rows")
    if not isinstance(features, Sequence) or isinstance(features, (str, bytes, bytearray)) or feature_index >= len(features):
        raise RuntimeError("SNAKE row missing selected_feature_rows needed for prefix fidelity replay")
    feature = features[feature_index]
    if not isinstance(feature, Mapping):
        raise RuntimeError("SNAKE selected_feature_rows contains non-object entry")
    label = str(feature.get("candidate_label") or (row.get("selected_ops") or [None])[feature_index])
    metadata = feature.get("generator_metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError(f"SNAKE feature {label} missing generator_metadata")
    compile_meta = metadata.get("compile_metadata")
    if not isinstance(compile_meta, Mapping):
        raise RuntimeError(f"SNAKE feature {label} missing compile_metadata")
    terms_raw = compile_meta.get("serialized_terms_exyz")
    if not isinstance(terms_raw, Sequence) or isinstance(terms_raw, (str, bytes, bytearray)):
        raise RuntimeError(f"SNAKE feature {label} missing serialized_terms_exyz")
    return _candidate_from_serialized_terms(
        label=label,
        terms_raw=[term for term in terms_raw if isinstance(term, Mapping)],
        construction="snake_history_selected_feature_row_v1",
        parent_label=feature.get("runtime_split_parent_label"),
        generator_metadata=metadata,
    )


def _generic_selected_candidates(
    *,
    payload: Mapping[str, Any],
    context: Any,
    prefix_k: int,
) -> list[_PoolCandidate]:
    result = payload.get("result") if isinstance(payload.get("result"), Mapping) else payload
    pool = build_full_meta_candidate_pool(context, max_terms=None)
    by_label = {candidate.label: candidate for candidate in pool}
    selected: list[_PoolCandidate] = []
    history = result.get("adapt_history")
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes, bytearray)):
        raise RuntimeError("generic result missing adapt_history")
    for row in history[: int(prefix_k)]:
        if not isinstance(row, Mapping):
            continue
        labels = row.get("selected_batch_labels") or []
        if isinstance(labels, str):
            labels = [labels]
        for label_raw in labels:
            label = str(label_raw)
            candidate = by_label.get(label)
            if candidate is None:
                raise RuntimeError(f"generic selected label not found in full_meta pool: {label}")
            selected.append(candidate)
    return selected


def _runtime_objects(context: Any) -> tuple[np.ndarray, dict[str, Any], Any]:
    psi_ref = np.asarray(context.reference_state.build_state(), dtype=complex).reshape(-1)
    psi_ref = psi_ref / max(float(np.linalg.norm(psi_ref)), 1.0e-300)
    pauli_action_cache: dict[str, Any] = {}
    h_compiled = compile_polynomial_action(context.hamiltonian, tol=1e-12, pauli_action_cache=pauli_action_cache)
    return psi_ref, pauli_action_cache, h_compiled


def _optimize_with_runtime(
    *,
    selected: Sequence[_PoolCandidate],
    x0: np.ndarray,
    psi_ref: np.ndarray,
    pauli_action_cache: dict[str, Any],
    h_compiled: Any,
    optimizer_kind: str,
    optimizer_maxiter: int,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    if not has_scipy_minimize_support():
        raise RuntimeError("scipy optimizer unavailable")
    minimize_fn = _import_scipy_minimize()
    return _optimize_selected(
        minimize_fn=minimize_fn,
        selected=selected,
        x0=np.asarray(x0, dtype=float).reshape(-1),
        psi_ref=psi_ref,
        h_compiled=h_compiled,
        pauli_action_cache=pauli_action_cache,
        optimizer_maxiter=int(optimizer_maxiter),
        optimizer_method=str(optimizer_kind),
        parameterization_mode="logical_shared",
    )


def _energy_for_selected_theta(
    *,
    selected: Sequence[_PoolCandidate],
    theta: np.ndarray,
    psi_ref: np.ndarray,
    pauli_action_cache: dict[str, Any],
    h_compiled: Any,
) -> float:
    psi = _prepare_selected_state(
        selected=selected,
        theta=np.asarray(theta, dtype=float).reshape(-1),
        psi_ref=psi_ref,
        pauli_action_cache=pauli_action_cache,
        parameterization_mode="logical_shared",
    )
    energy, _ = energy_via_one_apply(psi, h_compiled)
    return float(energy)


def _optimize_window_with_runtime(
    *,
    selected: Sequence[_PoolCandidate],
    theta: np.ndarray,
    active_indices: Sequence[int],
    psi_ref: np.ndarray,
    pauli_action_cache: dict[str, Any],
    h_compiled: Any,
    optimizer_kind: str,
    optimizer_maxiter: int,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    theta_full = np.asarray(theta, dtype=float).reshape(-1).copy()
    active = tuple(int(i) for i in active_indices if 0 <= int(i) < int(theta_full.size))
    if len(active) == 0 or len(active) == int(theta_full.size):
        return _optimize_with_runtime(
            selected=selected,
            x0=theta_full,
            psi_ref=psi_ref,
            pauli_action_cache=pauli_action_cache,
            h_compiled=h_compiled,
            optimizer_kind=optimizer_kind,
            optimizer_maxiter=optimizer_maxiter,
        )
    if str(optimizer_kind).strip().lower() not in {"powell", "scipy.optimize.minimize:powell"}:
        raise RuntimeError(f"windowed replay currently supports Powell rows only; got {optimizer_kind!r}")
    if not has_scipy_minimize_support():
        raise RuntimeError("scipy optimizer unavailable")
    minimize_fn = _import_scipy_minimize()
    x0 = theta_full[list(active)]
    eval_count = 0

    def objective(x: np.ndarray) -> float:
        nonlocal eval_count
        eval_count += 1
        trial = theta_full.copy()
        trial[list(active)] = np.asarray(x, dtype=float).reshape(-1)
        return _energy_for_selected_theta(
            selected=selected,
            theta=trial,
            psi_ref=psi_ref,
            pauli_action_cache=pauli_action_cache,
            h_compiled=h_compiled,
        )

    result = minimize_fn(
        objective,
        np.asarray(x0, dtype=float).reshape(-1),
        method="Powell",
        options={"maxiter": int(optimizer_maxiter), "xtol": 1e-5, "ftol": 1e-12},
    )
    theta_full[list(active)] = np.asarray(getattr(result, "x", x0), dtype=float).reshape(-1)
    energy = _energy_for_selected_theta(
        selected=selected,
        theta=theta_full,
        psi_ref=psi_ref,
        pauli_action_cache=pauli_action_cache,
        h_compiled=h_compiled,
    )
    return (
        theta_full,
        float(energy),
        {
            "nfev": getattr(result, "nfev", eval_count),
            "nit": getattr(result, "nit", None),
            "success": bool(getattr(result, "success", False)),
            "message": str(getattr(result, "message", "")),
            "optimizer": "scipy.optimize.minimize:Powell:windowed",
            "active_indices": list(active),
        },
    )


def _optimize_and_fidelity(
    *,
    selected: Sequence[_PoolCandidate],
    context: Any,
    source_payload: Mapping[str, Any],
    source_energy_reference: Mapping[str, Any],
    optimizer_kind: str,
    optimizer_maxiter: int,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, float, dict[str, Any], dict[str, Any]]:
    psi_ref, pauli_action_cache, h_compiled = _runtime_objects(context)
    if x0 is None:
        x0 = np.zeros(len(selected), dtype=float)
    theta, energy, opt = _optimize_with_runtime(
        selected=selected,
        x0=np.asarray(x0, dtype=float).reshape(-1),
        psi_ref=psi_ref,
        pauli_action_cache=pauli_action_cache,
        h_compiled=h_compiled,
        optimizer_maxiter=int(optimizer_maxiter),
        optimizer_kind=str(optimizer_kind),
    )
    ref = _normalize_reference_metric_fields(
        same_cutoff_exact_gs_energy=source_energy_reference.get("same_cutoff_exact_gs_energy"),
        exact_reference_energy=source_energy_reference.get("exact_reference_energy"),
        exact_reference_n_ph_max=source_energy_reference.get("exact_reference_n_ph_max"),
        primary_energy_metric=source_energy_reference.get("primary_energy_metric"),
        same_cutoff_error_role=source_energy_reference.get("same_cutoff_error_role"),
        fallback_same_cutoff_energy=_safe_exact_energy(context),
    )
    fid = _dense_exact_state_fidelity_for_selected(
        selected=selected,
        theta=theta,
        psi_ref=psi_ref,
        h_compiled=h_compiled,
        pauli_action_cache=pauli_action_cache,
        exact_energy=ref.get("same_cutoff_exact_gs_energy"),
        parameterization_mode="logical_shared",
    )
    return theta, float(energy), opt, {**fid, "reference_metrics": ref}


def _fixed_theta_energy_and_fidelity(
    *,
    selected: Sequence[_PoolCandidate],
    theta: np.ndarray,
    context: Any,
    source_energy_reference: Mapping[str, Any],
) -> tuple[float, dict[str, Any]]:
    psi_ref, pauli_action_cache, h_compiled = _runtime_objects(context)
    energy = _energy_for_selected_theta(
        selected=selected,
        theta=np.asarray(theta, dtype=float).reshape(-1),
        psi_ref=psi_ref,
        pauli_action_cache=pauli_action_cache,
        h_compiled=h_compiled,
    )
    ref = _normalize_reference_metric_fields(
        same_cutoff_exact_gs_energy=source_energy_reference.get("same_cutoff_exact_gs_energy"),
        exact_reference_energy=source_energy_reference.get("exact_reference_energy"),
        exact_reference_n_ph_max=source_energy_reference.get("exact_reference_n_ph_max"),
        primary_energy_metric=source_energy_reference.get("primary_energy_metric"),
        same_cutoff_error_role=source_energy_reference.get("same_cutoff_error_role"),
        fallback_same_cutoff_energy=_safe_exact_energy(context),
    )
    fid = _dense_exact_state_fidelity_for_selected(
        selected=selected,
        theta=np.asarray(theta, dtype=float).reshape(-1),
        psi_ref=psi_ref,
        h_compiled=h_compiled,
        pauli_action_cache=pauli_action_cache,
        exact_energy=ref.get("same_cutoff_exact_gs_energy"),
        parameterization_mode="logical_shared",
    )
    return float(energy), {**fid, "reference_metrics": ref}


def _generic_reference_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = payload.get("result") if isinstance(payload.get("result"), Mapping) else payload
    return {
        "same_cutoff_exact_gs_energy": result.get("same_cutoff_exact_gs_energy")
        or result.get("exact_gs_energy")
        or result.get("exact_energy"),
        "exact_reference_energy": result.get("exact_reference_energy"),
        "exact_reference_n_ph_max": result.get("exact_reference_n_ph_max"),
        "primary_energy_metric": result.get("primary_energy_metric", "same_cutoff_abs_delta_e"),
        "same_cutoff_error_role": result.get("same_cutoff_error_role", "primary"),
    }


def _snake_reference_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    adapt = payload.get("adapt_vqe") if isinstance(payload.get("adapt_vqe"), Mapping) else {}
    ground = payload.get("ground_state") if isinstance(payload.get("ground_state"), Mapping) else {}
    return {
        "same_cutoff_exact_gs_energy": adapt.get("exact_gs_energy") or ground.get("exact_energy"),
        "exact_reference_energy": adapt.get("exact_reference_energy"),
        "exact_reference_n_ph_max": adapt.get("exact_reference_n_ph_max"),
        "primary_energy_metric": adapt.get("primary_energy_metric", "same_cutoff_abs_delta_e"),
        "same_cutoff_error_role": adapt.get("same_cutoff_error_role", "primary"),
    }


def _replay_generic_row(row: Mapping[str, Any]) -> dict[str, Any]:
    source = Path(str(row["source_json"]))
    payload = _read_json(source)
    context = _context_from_generic_payload(payload)
    selected = _generic_selected_candidates(payload=payload, context=context, prefix_k=int(row["selected_prefix_k"]))
    result = payload.get("result") if isinstance(payload.get("result"), Mapping) else payload
    optimizer_kind = str(result.get("adapt_optimizer_kind") or result.get("optimizer_kind") or "powell")
    optimizer_maxiter = int(_int_or_none(result.get("optimizer_maxiter")) or _int_or_none(result.get("adapt_spsa_maxiter")) or 200)
    theta, energy, opt, fid = _optimize_and_fidelity(
        selected=selected,
        context=context,
        source_payload=payload,
        source_energy_reference=_generic_reference_fields(payload),
        optimizer_kind=optimizer_kind,
        optimizer_maxiter=optimizer_maxiter,
    )
    return _row_result(row=row, source_payload=payload, selected=selected, theta=theta, energy=energy, opt=opt, fid=fid)


def _snake_selected_through_prefix(
    payload: Mapping[str, Any],
    prefix_k: int,
    *,
    context: Any,
    optimizer_kind: str,
    optimizer_maxiter: int,
) -> tuple[list[_PoolCandidate], np.ndarray, list[dict[str, Any]], float | None, dict[str, Any]]:
    adapt = payload.get("adapt_vqe") if isinstance(payload.get("adapt_vqe"), Mapping) else {}
    history = adapt.get("history")
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes, bytearray)):
        raise RuntimeError("SNAKE source missing adapt_vqe.history")
    selected: list[_PoolCandidate] = []
    theta = np.zeros(0, dtype=float)
    events: list[dict[str, Any]] = []
    psi_ref, pauli_action_cache, h_compiled = _runtime_objects(context)
    last_energy: float | None = None
    last_opt: dict[str, Any] = {}
    for step, step_row in enumerate(history[: int(prefix_k)], start=1):
        if not isinstance(step_row, Mapping):
            continue
        labels = step_row.get("selected_ops") or []
        if isinstance(labels, str):
            labels = [labels]
        positions = step_row.get("selected_positions") or []
        if isinstance(positions, (str, bytes, bytearray)):
            positions = []
        for idx, _label in enumerate(labels):
            candidate = _snake_candidate_from_feature(step_row, idx)
            pos = int(positions[idx]) if idx < len(positions) else len(selected)
            pos = max(0, min(pos, len(selected)))
            selected.insert(pos, candidate)
            theta = np.insert(theta, pos, 0.0)
            events.append({"step": step, "event": "insert", "label": candidate.label, "position": pos})
        features = step_row.get("selected_feature_rows")
        active_indices: Sequence[int] = tuple(range(len(selected)))
        if isinstance(features, Sequence) and not isinstance(features, (str, bytes, bytearray)) and features:
            first_feature = features[0]
            if isinstance(first_feature, Mapping):
                raw_active = first_feature.get("optimizer_active_refit_indices") or first_feature.get("active_post_refit_indices")
                if isinstance(raw_active, Sequence) and not isinstance(raw_active, (str, bytes, bytearray)):
                    active_indices = tuple(int(i) for i in raw_active)
        theta, last_energy, last_opt = _optimize_window_with_runtime(
            selected=selected,
            theta=theta,
            active_indices=active_indices,
            psi_ref=psi_ref,
            pauli_action_cache=pauli_action_cache,
            h_compiled=h_compiled,
            optimizer_kind=optimizer_kind,
            optimizer_maxiter=optimizer_maxiter,
        )
        prune = step_row.get("post_admission_prune")
        if isinstance(prune, Mapping) and int(prune.get("accepted_count") or 0) > 0:
            # Current rows use frozen-delete accepted trials.  Delete the
            # recorded active coordinate.  The final prefix refit below is the
            # fidelity recovery state.
            raw_index = prune.get("selected_index")
            if raw_index is None and isinstance(prune.get("trial"), Mapping):
                raw_index = prune["trial"].get("selected_index")
            if raw_index is None:
                raise RuntimeError(f"SNAKE prune at step {step} accepted without selected_index")
            delete_index = int(raw_index)
            if not (0 <= delete_index < len(selected)):
                raise RuntimeError(f"SNAKE prune index out of range at step {step}: {delete_index} of {len(selected)}")
            deleted = selected.pop(delete_index)
            theta = np.delete(theta, delete_index)
            events.append({"step": step, "event": "accepted_prune", "label": deleted.label, "index": delete_index})
            last_energy = _energy_for_selected_theta(
                selected=selected,
                theta=theta,
                psi_ref=psi_ref,
                pauli_action_cache=pauli_action_cache,
                h_compiled=h_compiled,
            )
            last_opt = {"optimizer": "frozen_delete_no_survivor_refit", "nfev": 1, "success": True}
    return selected, theta, events, last_energy, last_opt


def _replay_snake_row(row: Mapping[str, Any]) -> dict[str, Any]:
    source = Path(str(row["source_json"]))
    payload = _read_json(source)
    settings = payload.get("settings") if isinstance(payload.get("settings"), Mapping) else {}
    context = _context_from_settings(settings)
    adapt = payload.get("adapt_vqe") if isinstance(payload.get("adapt_vqe"), Mapping) else {}
    optimizer_kind = str(adapt.get("adapt_inner_optimizer") or settings.get("adapt_optimizer_kind") or "powell")
    optimizer_maxiter = int(_int_or_none(settings.get("adapt_scipy_maxiter")) or _int_or_none(adapt.get("adapt_scipy_maxfev")) or 200)
    selected, x0, events, seq_energy, seq_opt = _snake_selected_through_prefix(
        payload,
        int(row["selected_prefix_k"]),
        context=context,
        optimizer_kind=optimizer_kind,
        optimizer_maxiter=optimizer_maxiter,
    )
    energy, fid = _fixed_theta_energy_and_fidelity(
        selected=selected,
        theta=x0,
        context=context,
        source_energy_reference=_snake_reference_fields(payload),
    )
    theta = x0
    opt = seq_opt
    out = _row_result(row=row, source_payload=payload, selected=selected, theta=theta, energy=energy, opt=opt, fid=fid)
    out["snake_replay_events"] = events
    out["snake_replay_event_count"] = len(events)
    out["snake_sequential_replay_energy_before_final_refit"] = seq_energy
    out["snake_sequential_replay_optimizer_nfev"] = seq_opt.get("nfev")
    return out


def _visible_reference_delta(row: Mapping[str, Any], energy: float, fid: Mapping[str, Any]) -> tuple[float | None, float | None]:
    ref = fid.get("reference_metrics")
    if not isinstance(ref, Mapping):
        return None, None
    same = _finite_float(ref.get("same_cutoff_exact_gs_energy"))
    if same is None:
        return None, None
    replay_abs = abs(float(energy) - float(same))
    visible = _finite_float(row.get("visible_abs_delta_e"))
    mismatch = None if visible is None else abs(float(replay_abs) - float(visible))
    return replay_abs, mismatch


def _row_result(
    *,
    row: Mapping[str, Any],
    source_payload: Mapping[str, Any],
    selected: Sequence[_PoolCandidate],
    theta: np.ndarray,
    energy: float,
    opt: Mapping[str, Any],
    fid: Mapping[str, Any],
) -> dict[str, Any]:
    replay_abs, mismatch = _visible_reference_delta(row, energy, fid)
    fidelity = _finite_float(fid.get("exact_state_fidelity"))
    one_minus = None if fidelity is None else max(0.0, 1.0 - float(fidelity))
    status = "ok"
    details: list[str] = []
    if fidelity is None:
        status = "blocked"
        details.append(str(fid.get("infidelity_status") or "fidelity_missing"))
    if mismatch is None:
        status = "blocked" if status == "blocked" else "warning"
        details.append("delta_e_mismatch_not_available")
    elif mismatch > 1e-6:
        status = "warning"
        details.append(f"replay_abs_delta_e_mismatch={mismatch:.6e}")
    return {
        "method_key": row.get("method_key"),
        "method_display": row.get("method_display"),
        "regime": row.get("regime"),
        "selected_prefix_k": int(row["selected_prefix_k"]),
        "selected_operator_count": int(len(selected)),
        "visible_abs_delta_e": _finite_float(row.get("visible_abs_delta_e")),
        "replay_energy": float(energy),
        "replay_abs_delta_e_same_cutoff": replay_abs,
        "replay_abs_delta_e_visible_mismatch": mismatch,
        "exact_state_fidelity": fidelity,
        "one_minus_fidelity": one_minus,
        "fidelity_status": status,
        "fidelity_status_detail": "; ".join(details) if details else "computed_prefix_replay_refit",
        "optimizer_nfev": opt.get("nfev"),
        "optimizer_nit": opt.get("nit"),
        "optimizer_success": opt.get("success"),
        "N2q": row.get("N2q"),
        "D2q": row.get("D2q"),
        "Dc": row.get("Dc"),
        "S_alg": row.get("S_alg"),
        "source_json": row.get("source_json"),
        "source_json_sha256": _sha256(Path(str(row["source_json"]))) if Path(str(row["source_json"])).exists() else row.get("source_json_sha256"),
        "selected_labels": [candidate.label for candidate in selected],
        "theta": [float(x) for x in np.asarray(theta, dtype=float).reshape(-1)],
        "dense_fidelity_payload": {k: v for k, v in fid.items() if k != "reference_metrics"},
        "reference_metrics": fid.get("reference_metrics"),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method_display",
        "regime",
        "selected_prefix_k",
        "selected_operator_count",
        "visible_abs_delta_e",
        "replay_abs_delta_e_same_cutoff",
        "replay_abs_delta_e_visible_mismatch",
        "exact_state_fidelity",
        "one_minus_fidelity",
        "fidelity_status",
        "fidelity_status_detail",
        "N2q",
        "D2q",
        "Dc",
        "S_alg",
        "source_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-tex", default=str(REPO_ROOT / "MATH/paper_details/Paper_I.tex"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--method", choices=["snake", "geo_parent", "append_parent", "all"], default="all")
    parser.add_argument("--regime", choices=[*REGIME_ORDER, "all"], default="all")
    args = parser.parse_args(argv)

    tex_path = Path(args.paper_tex)
    output_dir = Path(args.output_dir)
    snake_rows, support_csv = _snake_rows_from_tex(tex_path)
    rows = [*snake_rows, *_parent_comparator_rows(support_csv)]
    if args.method != "all":
        rows = [row for row in rows if row["method_key"] == args.method]
    if args.regime != "all":
        rows = [row for row in rows if row["regime"] == args.regime]

    results: list[dict[str, Any]] = []
    for row in rows:
        print(f"[fidelity] {row['method_display']} {row['regime']} k={row['selected_prefix_k']}", flush=True)
        try:
            if row["method_key"] == "snake":
                result = _replay_snake_row(row)
            else:
                result = _replay_generic_row(row)
        except Exception as exc:
            result = {
                "method_key": row.get("method_key"),
                "method_display": row.get("method_display"),
                "regime": row.get("regime"),
                "selected_prefix_k": row.get("selected_prefix_k"),
                "visible_abs_delta_e": row.get("visible_abs_delta_e"),
                "fidelity_status": "blocked",
                "fidelity_status_detail": f"{type(exc).__name__}: {exc}",
                "source_json": row.get("source_json"),
            }
        results.append(result)

    payload = {
        "schema": "paper_i_hh_tableiii_prefix_fidelity_replay_v1",
        "paper_tex": str(tex_path),
        "snake_block_marker": SNAKE_BLOCK_MARKER,
        "comparator_support_csv": str(support_csv),
        "row_count": len(results),
        "rows": results,
    }
    json_path = output_dir / "paper_i_hh_tableiii_prefix_fidelity_20260708.json"
    csv_path = output_dir / "paper_i_hh_tableiii_prefix_fidelity_20260708.csv"
    _write_json(json_path, payload)
    _write_csv(csv_path, results)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
